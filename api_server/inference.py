import logging

import torch
import time
import traceback

from pathlib import Path
from threading import Thread

from code_contrast import ScratchpadDiff
from code_contrast import ScratchpadCompletion
from code_contrast import CodifyModel

from typing import Optional, Union, Dict, Any, Iterable


class Inference:

    def __init__(self, workdir: Path, force_cpu: bool):
        self._workdir = workdir
        self._device = "cuda" if torch.has_cuda and not force_cpu else "cpu"
        self._model = None
        self._encoding = None

    def _prepare_scratchpad(self, request: Dict[str, Any]):
        created_ts = time.time()

        def logger(*args):
            logging.debug(args)

        object_type = request["object"]
        assert object_type in ["diff_completion_req", "text_completion_req"]
        if object_type == "diff_completion_req":
            scratchpad = ScratchpadDiff(
                enc=self._encoding,
                logger=logger,
                created=created_ts,
                **request)
        else:
            scratchpad = ScratchpadCompletion(
                enc=self._encoding,
                logger=logger,
                created=created_ts,
                **request)
        p = scratchpad.prompt(self._model.config.T)
        if len(p) == 0:
            raise RuntimeError("empty tokens prompt")

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

    def _from_pretrained(self, model: str):
        self._model = CodifyModel.from_pretrained(str(self._workdir / "weights"), repo_id="reymondzzz/testmodel")
        if self._device.startswith("cuda"):
            self._model = self._model.to(torch.half)
        self._model = self._model.eval()
        self._encoding = self._model.config.encoding

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
                    "text": "",
                    "files": scratchpad.completion(True),
                    "logprobs": None,
                    "finish_reason": scratchpad.finish_reason
                },
            ],
            "status": status,
            "created": scratchpad.created,
            "uploaded": time.time(),
            "generated_tokens_n": scratchpad.generated_tokens_n,
            **scratchpad.toplevel_fields(),
        }

    @property
    def ready(self):
        return self._model is not None and self._encoding is not None

    def startup(self, model: str):
        if self.ready:
            return False

        startup_thread = Thread(
            target=self._from_pretrained,
            kwargs={'model': model})
        startup_thread.start()

        return True

    def infer(self, request: Dict[str, Any], stream: bool) -> Iterable[Optional[Dict[str, Any]]]:
        try:
            scratchpad, tokens_prompt = self._prepare_scratchpad(request)
            with torch.inference_mode():
                for _ in self._generate_scratchpad(tokens_prompt, scratchpad, max_length=request["max_tokens"]):
                    if scratchpad.needs_upload and stream:
                        yield self._json_result(scratchpad, status="in_progress")
                    else:
                        yield None
            assert scratchpad.finish_reason
            scratchpad.finalize()

            yield self._json_result(scratchpad, status="completed")
        except Exception as e:
            logging.error(e)
            logging.error(traceback.format_exc())
            yield None
