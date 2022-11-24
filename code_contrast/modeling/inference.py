import asyncio

import torch
import time
import traceback
import termcolor
from uuid import uuid4

from pydantic import BaseModel, Required

from code_contrast import log
from code_contrast.pprint import hlprint
from code_contrast.inf_scratchpad_diff import ScratchpadDiff
from code_contrast.inf_scratchpad_completion import ScratchpadCompletion
from code_contrast.modeling.codify_model import CodifyModel

from typing import Optional, Union, Dict, Any, List, Iterable

import json
import uvicorn
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse


class NlpSamplingParams(BaseModel):
    model: str = Query(default=Required, regex="^[a-z/A-Z0-9_]+$")
    max_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 1.0
    top_n: int = 0
    stop: Union[List[str], str] = []
    stream: bool = False

    def clamp(self):
        def _clamp(a, b, x):
            return max(a, min(b, x))
        self.temperature = _clamp(0, 4, self.temperature)
        self.top_p = _clamp(0.0, 1.0, self.top_p)
        self.top_n = _clamp(0, 1000, self.top_n)
        self.max_tokens = _clamp(0, 8192, self.max_tokens)
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_n": self.top_n,
            "max_tokens": self.max_tokens,
            "stop_tokens": self.stop,
        }


class TextCompletion(NlpSamplingParams):
    prompt: str


class DiffCompletion(NlpSamplingParams):
    intent: str
    sources: Dict[str, str]
    cursor_file: str
    cursor0: int
    cursor1: int
    function: str = Query(
        default=Required,
        regex="^(highlight|infill|diff-anywhere|diff-atcursor|diff-selection|edit-chain)$"
    )
    max_edits: int = 4


def print_tensor(tensor: torch.Tensor):
    return "·".join(["%i" % i for i in tensor.shape]) + " " + str(tensor.dtype).replace("torch.", "")


def _make_mask(seq_len: int, past_key_values_length: int, device: torch.device):
    # prompt
    if past_key_values_length == 0:
        mask = torch.ones((seq_len, seq_len + past_key_values_length), dtype=torch.bool, device=device)
        mask = torch.triu(mask, 1)
    else:
        mask = torch.zeros((seq_len, seq_len + past_key_values_length), dtype=torch.bool, device=device)
    return mask


class Predictor:

    def __init__(self, weights: str, device: str = 'cuda'):
        self._device = device
        self._model = CodifyModel.from_pretrained(weights)
        self._model = self._model.to(self._device).to(torch.half).eval()
        self._encoding = self._model.config.encoding

    def prepare(self, request: Dict[str, Any]):
        # with traces.Profiler("prepare"):
        if True:
            object_type = request["object"]
            if object_type == "diff_completion_req":
                scratchpad = ScratchpadDiff(self._encoding, **request)
            else:
                scratchpad = ScratchpadCompletion(self._encoding, **request)
            p = scratchpad.prompt(self._model.config.T)
            assert len(p) > 0
            # The request is validated by spad
            tokens_prompt = torch.tensor(p, device=self._device)
        return scratchpad, tokens_prompt

    def _make_mask(self, seq_len: int, past_key_values_length: int):
        # prompt
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

    def __call__(self, request, stream) -> Iterable[Optional[Dict[str, Any]]]:
        # ts_batch_started = time.time()
        scratchpad, tokens_prompt = self.prepare(request)
        # with traces.Profiler("sampling"):
        try:
            with torch.inference_mode():
                # tokens = None
                for tokens in self._generate_scratchpad(tokens_prompt, scratchpad,
                                                        max_length=request["max_tokens"]):
                    if scratchpad.needs_upload and stream:
                        yield self._json_result(request["id"], scratchpad, status="in_progress")
                    else:
                        yield None
                    # log("%0.2fs sampling over, result %s" % (time.time() - ts_batch_started, print_tensor(tokens)))
            assert scratchpad.finish_reason
            scratchpad.finalize()
            # logging
            # tokens = ret["tokens"].cpu().numpy()
            # completion = tokens[len(tokens_prompt):]
            # log("completion%i %s '%s'" % (0, str(completion), hlprint(completion, self._model.config.encoding).replace("\n", "\\n")))
            # if isinstance(scratchpad, ScratchpadDiff):
            #     if scratchpad.diff_out and scratchpad.diff_out.errors:
            #         log(termcolor.colored(str(scratchpad.diff_out.errors), "red"))
            yield self._json_result(request["id"], scratchpad, status="completed")
        except Exception as e:
            log(traceback.format_exc())
            yield None

    @staticmethod
    def _json_result(request_id: str,
                     scratchpad,
                     status: str):
        assert status in ["in_progress", "completed"]
        return {
            "id": request_id,
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


async def streamer(request: Dict[str, Any], predictor: Predictor):
    try:
        stream = request["stream"]
        for response in predictor(request, stream):
            if response is None:
                continue
            data = json.dumps(response)
            if stream:
                data = "data: " + data + "\n\n"
            yield data
        if stream:
            yield "data: [DONE]" + "\n\n"
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    predictor = Predictor(weights="/home/mitya/model_weights")
    app = FastAPI(docs_url=None)

    @app.post("/completion")
    async def contrast(post: TextCompletion):
        request = post.clamp()
        request.update({
            "id": str(uuid4()),
            "object": "text_completion_req",
            "prompt": post.prompt,
            "stop_tokens": post.stop,
            "stream": post.stream,
        })
        return StreamingResponse(streamer(request, predictor))

    @app.post("/contrast")
    async def contrast(post: DiffCompletion):
        if post.function != "diff-anywhere":
            if post.cursor_file not in post.sources:
                raise HTTPException(status_code=400, detail="cursor_file='%s' is not in sources=%s" % (
                post.cursor_file, list(post.sources.keys())))
            if post.cursor0 < 0 or post.cursor1 < 0:
                raise HTTPException(status_code=400,
                                    detail="cursor0=%d or cursor1=%d is negative" % (post.cursor0, post.cursor1))
            filetext = post.sources[post.cursor_file]
            if post.cursor0 > len(filetext) or post.cursor1 > len(filetext):
                raise HTTPException(status_code=400, detail="cursor0=%d or cursor1=%d is beyond file length=%d" % (
                post.cursor0, post.cursor1, len(filetext)))
        else:
            post.cursor0 = -1
            post.cursor1 = -1
            post.cursor_file = ""
        if post.function == "highlight":
            post.max_tokens = 1
        request = post.clamp()
        request.update({
            "id": str(uuid4()),
            "object": "diff_completion_req",
            "intent": post.intent,
            "sources": post.sources,
            "cursor_file": post.cursor_file,
            "cursor0": post.cursor0,
            "cursor1": post.cursor1,
            "function": post.function,
            "max_edits": post.max_edits,
            "stop_tokens": post.stop,
            "stream": post.stream,
        })
        return StreamingResponse(streamer(request, predictor))

    uvicorn.run(app, host="127.0.0.1", port=8008)

    # predictor = Predictor(weights="/home/mitya/model_weights")
    # call = {
    #     "model": "CONTRASTcode/medium",
    #     "prompt": "import numpy as np\n\ndef hello_world_in_numpy():\n",
    #     "echo": True,
    #     "stream": False,
    #     "temperature": 0.4,
    #     "max_tokens": 200
    # }
    # call = {
    #     "model": "CONTRASTcode/medium",
    #     "sources": {"hello.py": "def hello_world():\n    pass\n\ndef a_distraction_function():\n    print(\"there to distract!\")\n\n"},
    #     "intent": "Implement hello_world function",
    #     "function": "diff-anywhere",
    #     "cursor_file": "hello.py",
    #     "cursor0": 27,
    #     "cursor1": 27,
    #     "stream": False,
    #     "temperature": 0.7,
    #     "max_tokens": 500,
    #     "max_edits": 1,
    #     "stop": ["\n\n"]
    # }
    # call = {
    #     'model': 'CONTRASTcode/stable',
    #     'sources': {'test.py': "def print_hello(times: int):\n    \n    \nif __name__ == '__main__':\n    print_hello()\n\n\n    \n"},
    #     'intent': 'Infill',
    #     'function': 'infill',
    #     'cursor_file': 'test.py',
    #     'cursor0': 33,
    #     'cursor1': 33,
    #     'temperature': 0.2,
    #     'max_tokens': 150,
    #     'max_edits': 1,
    #     'stop': ['\n\n'],
    #     "stream": True,
    # }
    # for data in predictor(
    #         {
    #             "id": "",
    #             "object": "diff_completion_req",
    #             # "object": "text_completion_req",
    #             "stop_tokens": call.get("stop", []),
    #             **call
    #         }, False):
    #     print(data)
