import torch as th


def temperature_top_k_top_p_filtering(logits, temperature=1, top_k=0, top_p=0, filter_value=-float('Inf')):
    assert logits.dim() == 1

    temperature = min(temperature, 1.0)
    temperature = max(temperature, 0.0)
    logits = logits / (temperature + 0.01)
    top_k = min(top_k, logits.size(-1))

    if top_k > 0:
        indices_to_remove = logits < th.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = th.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits
