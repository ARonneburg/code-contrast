import torch
import logging
import requests
import time
import traceback

from pathlib import Path
from threading import Thread
from threading import Lock

from code_contrast import ScratchpadDiff
from code_contrast import ScratchpadCompletion
from code_contrast import CodifyModel

from typing import Optional, Union, Dict, Any, Iterable, Tuple


__all__ = ["Inference", "LockedInference",
           "LockedError", "NoSettedModel", "InvalidModel"]


class LockedError(Exception):
    pass


class NoSettedModel(Exception):
    pass


class InvalidModel(Exception):
    pass


class Inference:

    def __init__(self, token: str, workdir: Path, force_cpu: bool):
        self._device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"

        self._model_lock = Lock()
        self._model = None
        self._encoding = None
        self._model_name = None

        self._model_setup_thread = Thread(
            target=self._model_setup,
            kwargs={
                'workdir': workdir,
                "token": token,
            })
        self._model_setup_thread.start()

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

    @staticmethod
    def _fetch_model(token) -> Tuple[str, str]:
        url = "https://max.smallcloud.ai/v1/codify-model"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        response = requests.get(url=url, headers=headers).json()
        if response["retcode"] != "OK":
            raise RuntimeError(response.get("human_readable_message", "unknown error"))
        print(response)
        model_name = response["tentant_model"]["model_name"]
        model_path = response["tentant_model"]["model_path"]
        model_path_type = response["tentant_model"]["model_path_type"]
        if model_path_type not in ["huggingface"]:
            raise RuntimeError(f"unknown model path type {model_path_type}")
        return model_name, model_path

    def _model_setup(self, token: str, workdir: Path):
        fetch_timeout = 30
        while True:
            # try:
            #     model_name, model_path = self._fetch_model(token)
            # except Exception as e:
            #     self._model = None
            #     self._encoding = None
            #     self._model_name = None
            #     time.sleep(fetch_timeout)
            #     logging.error("model fetch failed:")
            #     logging.error(e)
            #     continue
            model_name = "CONTRASTcode/medium"
            model_path = "smallcloudai/codify_medium_py"
            if model_name == self._model_name:
                time.sleep(fetch_timeout)
                continue
            with self._model_lock:
                try:
                    time.sleep(10)
                    self._model = CodifyModel.from_pretrained(
                        str(workdir / "weights"), repo_id=model_path)
                    self._model.to(self._device)
                    if self._device.startswith("cuda"):
                        self._model = self._model.to(torch.half)
                    self._model = self._model.eval()
                    self._encoding = self._model.config.encoding
                    self._model_name = model_name
                    fetch_timeout = 1
                except Exception as e:
                    self._model = None
                    self._encoding = None
                    self._model_name = None
                    fetch_timeout = 60
                    logging.error("model loading failed:")
                    logging.error(e)

    @staticmethod
    def _json_result(scratchpad, status: str):
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
    def lock(self):
        return self._model_lock

    def infer(self, request: Dict[str, Any], stream: bool) -> Iterable[Optional[Dict[str, Any]]]:
        try:
            scratchpad, tokens_prompt = self._prepare_scratchpad(request)
            with torch.inference_mode():
                for _ in self._generate_scratchpad(tokens_prompt, scratchpad, max_length=request["max_tokens"]):
                    print("token")
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

    def locked_inference(self, model_name):
        print(model_name, self._model_name)
        if not self._model_lock.acquire(blocking=False):
            raise LockedError
        if self._model_name is None:
            raise NoSettedModel
        if model_name != self._model_name:
            raise InvalidModel
        return LockedInference(self)


class LockedInference:

    def __init__(self, inference: Inference):
        self._inference = inference
        assert self._inference.lock.locked()

    def infer(self, request: Dict[str, Any], stream: bool) -> Iterable[Optional[Dict[str, Any]]]:
        for data in self._inference.infer(request, stream):
            yield data

    def __del__(self):
        self._inference.lock.release()
