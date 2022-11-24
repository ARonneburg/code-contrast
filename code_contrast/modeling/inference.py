import torch
import time
import copy
import traceback
import termcolor

from code_contrast.pprint import hlprint
from code_contrast.inf_scratchpad_diff import ScratchpadDiff
from code_contrast.inf_scratchpad_completion import ScratchpadCompletion
from modeling.codify_model import CodifyModel

from typing import List, Optional, Union, Dict, Any


def print_tensor(tensor: torch.Tensor):
    return "·".join(["%i" % i for i in tensor.shape]) + " " + str(tensor.dtype).replace("torch.", "")


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


def _make_mask(seq_len: int, past_key_values_length: int, device: torch.device):
    mask = torch.ones((seq_len, seq_len + past_key_values_length), dtype=torch.bool, device=device)
    mask = torch.triu(mask, 1)
    return mask


def generate_scratchpad(model: torch.nn.Module,
                        input_ids: torch.Tensor,
                        scratchpad: Union[ScratchpadCompletion, ScratchpadDiff],
                        max_length: int,
                        use_cache: Optional[bool] = None) -> torch.Tensor:
    encoder = model.config.encoding
    past_key_values = None
    input_ids = input_ids.unsqueeze(0)
    next_tokens = input_ids
    while True:
        batch_size, seq_len = next_tokens.shape
        cache_len = 0
        if use_cache and past_key_values is not None:
            cache_len += past_key_values[0][0].shape[2]

        attention_mask = _make_mask(seq_len, cache_len, next_tokens.device)

        output = model(next_tokens,
                       attention_mask=attention_mask,
                       past_key_values=past_key_values,
                       use_cache=use_cache)
        hidden_states, past_key_values = output
        logits = model.lm_forward(hidden_states)

        next_tokens = scratchpad.new_token(model, 0, logits[:, -1, :encoder.n_vocab], past_key_values).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        if use_cache is not True:
            next_tokens = input_ids

        if scratchpad.finish_reason:
            break

        if input_ids.shape[1] >= max_length:
            break

    return input_ids[0]


class Predictor:

    def __init__(self, weights: str, device: str = 'cuda'):
        self._device = device
        self._model = CodifyModel.from_pretrained(weights)
        self._model = self._model.to(self._device).to(torch.half).eval()

    def __call__(self, call):
        print(call)
        ts_batch_started = time.time()
        # with traces.Profiler("prepare"):
        if True:
            if call.get("function", "completion") in ["completion"]:
                call["object"] = "text_completion_req"
            else:
                call["object"] = "diff_completion_req"
            from uuid import uuid4
            call["id"] = str(uuid4())
            call["stop_tokens"] = call.get("stop", [])
            object_type = call["object"]
            if object_type == "diff_completion_req":
                scratchpad = ScratchpadDiff(self._model.config.encoding, **call)
            else:
                scratchpad = ScratchpadCompletion(self._model.config.encoding, **call)
            p = scratchpad.prompt(self._model.config.T)
            if len(p) == 0:
                assert False
            # The request is validated by spad
            tokens_prompt = torch.tensor(p, device=self._device)
            max_tokens = call["max_tokens"]
            # temperature = float(call["temperature"])
        # with traces.Profiler("sampling"):
        with torch.inference_mode():
            ret = dict(tokens=None)
            try:
                ret["tokens"] = generate_scratchpad(
                    self._model, tokens_prompt, scratchpad, max_length=max_tokens, use_cache=True)
                print("%0.2fs sampling over, result %s" % (time.time() - ts_batch_started, print_tensor(ret["tokens"])))
            except Exception as e:
                print(traceback.format_exc())
                return None
        if scratchpad.finish_reason == "":
            scratchpad.finish_reason = "maxlen"
        tokens = ret["tokens"].cpu().numpy()
        scratchpad.finalize()
        completion = tokens[len(tokens_prompt):]
        print("completion%i %s '%s'" % (0, str(completion), hlprint(completion, self._model.config.encoding).replace("\n", "\\n")))
        if isinstance(scratchpad, ScratchpadDiff):
            if scratchpad.diff_out and scratchpad.diff_out.errors:
                print(termcolor.colored(str(scratchpad.diff_out.errors), "red"))
        return self._json_result(call, scratchpad, status="completed")

    @staticmethod
    def _json_result(call: Dict[str, Any],
                     scratchpad,
                     status: str):
        assert status in ["in_progress", "completed"]
        return {
            "id": call["id"],
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


if __name__ == "__main__":
    import json
    import uvicorn
    from fastapi import FastAPI, Request, Response

    # signal.signal(signal.SIGUSR1, catch_sigkill)
    # traces.configure()

    # predictor = Predictor(weights="/home/mitya/model_weights")
    # app = FastAPI(docs_url=None, redoc_url="/watch-meh")
    #
    # @app.post("/contrast")
    # async def run_model(request: Request):
    #     request = await request.json()
    #     response = predictor(request)
    #     return Response(content=json.dumps(response))
    #
    # uvicorn.run(app, host="127.0.0.1", port=8008)

    predictor = Predictor(weights='gs://small-storage1/checkpoints/Diffs-v0/11-mix10-medium-tposaft-lr50/000300000/')
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
    call = {
        'model': 'CONTRASTcode/stable',
        'sources': {'test.py': "def print_hello(times: int):\n    \n    \nif __name__ == '__main__':\n    print_hello()\n\n\n    \n"},
        'intent': 'Infill',
        'function': 'completion',
        'cursor_file': 'test.py',
        'cursor0': 33,
        'cursor1': 33,
        'temperature': 0.2,
        'max_tokens': 50,
        'max_edits': 1,
        'stop': ['\n\n']
    }
    print(predictor(call))
