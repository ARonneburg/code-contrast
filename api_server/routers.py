import asyncio
import requests
import json

from fastapi import Header
from fastapi import Response
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from uuid import uuid4

from api_server.params import TextSamplingParams
from api_server.params import DiffSamplingParams
from api_server.utils import TokenHandler
from api_server.inference import Inference

from typing import Dict, Any


__all__ = ["GreetingsRouter", "CompletionRouter", "ContrastRouter"]


async def inference_streamer(
        request: Dict[str, Any], inference: Inference):
    try:
        stream = request["stream"]
        for response in inference.infer(request, stream):
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


def parse_authorization_header(authorization: str = Header(None)) -> str:
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    bearer_hdr = authorization.split(" ")
    if len(bearer_hdr) != 2 or bearer_hdr[0] != "Bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    return bearer_hdr[1]


class GreetingsRouter(APIRouter):

    def __init__(self,
                 token: TokenHandler,
                 inference: Inference,
                 *args, **kwargs):
        self._token = token
        self._inference = inference
        super(GreetingsRouter, self).__init__(*args, **kwargs)
        super(GreetingsRouter, self).add_api_route("/greetings", self._greetings, methods=["GET"])

    @staticmethod
    def _get_user_info(token) -> Dict[str, Any]:
        url = "https://max.smallcloud.ai/v1/tenant-info"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        response = requests.get(url=url, headers=headers)
        return response.json()

    async def _greetings(self, authorization: str = Header(None)):
        token = parse_authorization_header(authorization)

        user_info = self._get_user_info(token)
        if user_info.get("retcode", "") != "OK":
            raise HTTPException(status_code=401, detail="Could not get user info")
        model = user_info.get("tentant_info", {}).get("model", "")  # TODO

        server_token = self._token.get()
        if server_token is None:
            self._token.set(token)
        elif server_token != token:
            raise HTTPException(status_code=401, detail="This server cannot work with your API key")

        if not self._inference.ready:
            self._inference.startup(model)

        return Response(content=json.dumps({
            "status": "ready" if self._inference.ready else "startup",
        }))


class CompletionRouter(APIRouter):

    def __init__(self,
                 token: TokenHandler,
                 inference: Inference,
                 *args, **kwargs):
        self._token = token
        self._inference = inference
        super(CompletionRouter, self).__init__(*args, **kwargs)
        super(CompletionRouter, self).add_api_route("/completion", self._completion, methods=["POST"])

    async def _completion(self,
                          post: TextSamplingParams,
                          authorization: str = Header(None)):
        token = parse_authorization_header(authorization)
        if not self._inference.ready:
            raise HTTPException(status_code=401, detail="Server is not ready")
        if self._token.get() != token:
            raise HTTPException(status_code=401, detail="This server cannot work with your API key")
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

    def __init__(self,
                 token: TokenHandler,
                 inference: Inference,
                 *args, **kwargs):
        self._token = token
        self._inference = inference
        super(ContrastRouter, self).__init__(*args, **kwargs)
        super(ContrastRouter, self).add_api_route("/contrast", self._contrast, methods=["POST"])

    async def _contrast(self,
                        post: DiffSamplingParams,
                        authorization: str = Header(None)):
        token = parse_authorization_header(authorization)
        if not self._inference.ready:
            raise HTTPException(status_code=401, detail="Server is not ready")
        if self._token.get() != token:
            raise HTTPException(status_code=401, detail="This server cannot work with your API key")
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
