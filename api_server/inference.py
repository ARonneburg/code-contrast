import torch
import time
import traceback
import termcolor

from code_contrast import log
from code_contrast.pprint import hlprint
from code_contrast.scratchpad.diff import ScratchpadDiff
from code_contrast.scratchpad.completion import ScratchpadCompletion
from code_contrast.modeling.codify_model import CodifyModel

from typing import Optional, Union, Dict, Any, Iterable


def print_tensor(tensor: torch.Tensor):
    return "·".join(["%i" % i for i in tensor.shape]) + " " + str(tensor.dtype).replace("torch.", "")


class Inference:

    def __init__(self, weights: str, device: str = 'cuda'):
        self._device = device
        self._model = CodifyModel.from_pretrained(weights)
        self._model = self._model.to(self._device).to(torch.half).eval()
        self._encoding = self._model.config.encoding

    def _prepare(self, request: Dict[str, Any]):
        object_type = request["object"]
        assert object_type in ["diff_completion_req", "text_completion_req"]
        if object_type == "diff_completion_req":
            scratchpad = ScratchpadDiff(self._encoding, **request)
        else:
            scratchpad = ScratchpadCompletion(self._encoding, **request)
        p = scratchpad.prompt(self._model.config.T)
        assert len(p) > 0
        tokens_prompt = torch.tensor(p, device=self._device)
        return scratchpad, tokens_prompt

    def _make_mask(self, seq_len: int, past_key_values_length: int):
        if past_key_values_length == 0:
            mask = torch.ones((seq_len, seq_len + past_key_values_length),
                              dtype=torch.bool, device=self._device)
            mask = torch.triu(mask, 1)
        else:
            mask = torch.zeros((seq_len, seq_len + past_key_values_length),
                               dtype=torch.bool, device=self._device)
        return mask

    def _generate_scratchpad(self,
                             input_ids: torch.Tensor,
                             scratchpad: Union[ScratchpadCompletion, ScratchpadDiff],
                             max_length: int,
                             use_cache: bool = True) -> torch.Tensor:
        past_key_values = None
        input_ids = input_ids.unsqueeze(0)
        next_tokens = input_ids
        for token_idx in range(max_length):
            batch_size, seq_len = next_tokens.shape
            cache_len = 0
            if use_cache and past_key_values is not None:
                cache_len = past_key_values[0][0].shape[2]
            attention_mask = self._make_mask(seq_len, cache_len)

            hidden_state, past_key_values = self._model(
                next_tokens,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache)
            logits = self._model.lm_forward(hidden_state)

            next_tokens = scratchpad.new_token(
                self._model, 0, logits[:, -1, :self._encoding.n_vocab], dict(x_bte=hidden_state)).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            if not use_cache:
                next_tokens = input_ids

            yield input_ids[0]

            if scratchpad.finish_reason:
                break

        if not scratchpad.finish_reason:
            scratchpad.finish_reason = "maxlen"

    def __call__(self, request: Dict[str, Any], stream: bool) -> Iterable[Optional[Dict[str, Any]]]:
        ts_batch_started = time.time()
        scratchpad, tokens_prompt = self._prepare(request)
        try:
            with torch.inference_mode():
                tokens = None
                for tokens in self._generate_scratchpad(tokens_prompt, scratchpad,
                                                        max_length=request["max_tokens"]):
                    if scratchpad.needs_upload and stream:
                        yield self._json_result(scratchpad, status="in_progress")
                    else:
                        yield None
                    log("%0.2fs sampling over, result %s" % (time.time() - ts_batch_started, print_tensor(tokens)))
            assert scratchpad.finish_reason
            scratchpad.finalize()

            tokens = tokens.cpu().numpy()
            completion = tokens[len(tokens_prompt):]
            hlcompletion = hlprint(completion, self._model.config.encoding).replace("\n", "\\n")
            log(f"completion {completion} '{hlcompletion}'")
            if isinstance(scratchpad, ScratchpadDiff):
                if scratchpad.diff_out and scratchpad.diff_out.errors:
                    log(termcolor.colored(str(scratchpad.diff_out.errors), "red"))

            yield self._json_result(scratchpad, status="completed")
        except ...:
            log(traceback.format_exc())
            yield None

    @staticmethod
    def _json_result(scratchpad,
                     status: str):
        assert status in ["in_progress", "completed"]
        return {
            "id": scratchpad.id,
            "object": "text_completion",
            "choices": [
                {
                    "index": 0,
                    "files": scratchpad.completion(True),
                    "logprobs": None,
                    "finish_reason": scratchpad.finish_reason
                },
            ],
            "status": status,
            "more_toplevel_fields": (scratchpad.toplevel_fields(),),
            "generated_tokens_n": (scratchpad.generated_tokens_n,),
        }
