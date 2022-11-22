from typing import Optional

import torch
import torch.nn


def _temperature_top_k_top_p_filtering(logits,
                                       temperature: Optional[float] = None,
                                       top_k: Optional[int] = None,
                                       top_p: Optional[float] = None,
                                       filter_value=-float('Inf')):
    # assert logits.dim() == 1
    if temperature is not None:
        assert 0 < temperature <= 1
        logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p is not None:
        assert 0 < top_p <= 1
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits


def generate(model: torch.nn.Module,
             input_ids: torch.Tensor,
             max_length: Optional[int] = None,
             temperature: Optional[float] = None,
             top_k: Optional[int] = None,
             top_p: Optional[float] = None,
             eos_token_id: Optional[int] = None,
             use_cache: Optional[bool] = None):
    encoder = model.config.encoding
    eos_token_id = eos_token_id if eos_token_id is not None else encoder.EOT
    past_key_values = None
    next_tokens = input_ids
    while True:
        batch_size, seq_len = next_tokens.shape
        cached_len = seq_len
        if use_cache and past_key_values is not None:
            cached_len += past_key_values[0][0].shape[2]

        attention_mask = torch.zeros(batch_size, 1, seq_len, cached_len,
                                     dtype=torch.bool,
                                     device=next_tokens.device)
        output = model(next_tokens,
                       attention_mask=attention_mask,
                       past_key_values=past_key_values,
                       use_cache=use_cache)
        logits, past_key_values = output
        last_logits = _temperature_top_k_top_p_filtering(logits[:, -1],
                                                         temperature=temperature,
                                                         top_k=top_k,
                                                         top_p=top_p)
        probs = torch.softmax(last_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        if use_cache != True:
            next_tokens = input_ids

        if (next_tokens == eos_token_id).all():
            break

        if input_ids.shape[1] >= max_length:
            break

    return input_ids
