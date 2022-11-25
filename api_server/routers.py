import asyncio
import json

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from uuid import uuid4

from api_server.params import TextSamplingParams
from api_server.params import DiffSamplingParams
from api_server.inference import Inference

from typing import Dict, Any


async def inference_streamer(
        request: Dict[str, Any], inference: Inference):
    try:
        stream = request["stream"]
        for response in inference(request, stream):
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


class CompletionRouter(APIRouter):

    def __init__(self, inference, *args, **kwargs):
        self._inference = inference
        super(CompletionRouter, self).__init__(*args, **kwargs)
        super(CompletionRouter, self).add_api_route("/completion", self._completion, methods=["POST"])

    async def _completion(self, post: TextSamplingParams):
        request = post.clamp()
        request.update({
            "id": str(uuid4()),
            "object": "text_completion_req",
            "prompt": post.prompt,
            "stop_tokens": post.stop,
            "stream": post.stream,
        })
        return StreamingResponse(inference_streamer(request, self._inference))


class ContrastRouter(APIRouter):

    def __init__(self, inference, *args, **kwargs):
        self._inference = inference
        super(ContrastRouter, self).__init__(*args, **kwargs)
        super(ContrastRouter, self).add_api_route("/contrast", self._contrast, methods=["POST"])

    async def _contrast(self, post: DiffSamplingParams):
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
        return StreamingResponse(inference_streamer(request, self._inference))


# if __name__ == "__main__":
#     from argparse import ArgumentParser
#
#     parser = ArgumentParser()
#     parser.add_argument("--host", type=str, default="127.0.0.1")
#     parser.add_argument("--port", type=int, default=8008)
#
#     inference = Inference(weights="/code/model_weights")
#     server = FastAPI(docs_url=None)
#
#     uvicorn.run(server, host="127.0.0.1", port=8008)
