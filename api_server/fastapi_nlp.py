import collections
import sys, time, threading, random, json, re, copy, asyncio
from aredis.pipeline import StrictPipeline
import aredis.pubsub
import aiohttp
from fastapi import APIRouter, Request, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from deploy_front_py.fastapi_highlevel_calls import router
from deploy_front_py import fastapi_utils as fu
from deploy_front_py.fastapi_utils import log, safe_for_redis
from pydantic import BaseModel, Required
from typing import List, Dict, Tuple, Optional, Callable, Union


TIMEOUT = 30


router = APIRouter()




class NlpSamplingParams(BaseModel):
    max_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 1.0
    top_n: int = 0
    stop: Union[List[str], str] = []
    def clamp(self):
        self.temperature = fu.clamp(0, 4, self.temperature)
        self.top_p = fu.clamp(0.0, 1.0, self.top_p)
        self.top_n = fu.clamp(0, 1000, self.top_n)
        self.max_tokens = fu.clamp(0, 8192, self.max_tokens)
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_n": self.top_n,
            "max_tokens": self.max_tokens,
            "created": time.time(),
            "stop_tokens": self.stop,
        }



class DiffCompletion(NlpSamplingParams):
    model: str = Query(default=Required, regex="^[a-z/A-Z0-9_]+$")
    intent: str
    sources: Dict[str, str]
    cursor_file: str
    cursor0: int
    cursor1: int
    function: str = Query(
        default=Required, regex="^(highlight|infill|diff-anywhere|diff-atcursor|diff-selection|edit-chain)$"
    )
    max_edits: int = 4
    stream: bool = False
    # n: int = 1



@router.post("/contrast")
async def contrast(
    post: DiffCompletion,
    authorization: str = Header(None),
):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    bearer_hdr = authorization.split(" ")
    if len(bearer_hdr) != 2 or bearer_hdr[0] != "Bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    red = fu.get_red()
    ac_dict = await lookup_bearer(bearer_hdr[1], red)
    if ac_dict is None:
        raise HTTPException(status_code=401, detail="Could not verify your API key")
    # print("account", ac_dict["account"], "karma", ac_dict["karma"])
    account = ac_dict["account"]
    if post.function != "diff-anywhere":
        if post.cursor_file not in post.sources:
            raise HTTPException(status_code=400, detail="cursor_file='%s' is not in sources=%s" % (post.cursor_file, list(post.sources.keys())))
        if post.cursor0 < 0 or post.cursor1 < 0:
            raise HTTPException(status_code=400, detail="cursor0=%d or cursor1=%d is negative" % (post.cursor0, post.cursor1))
        filetext = post.sources[post.cursor_file]
        if post.cursor0 > len(filetext) or post.cursor1 > len(filetext):
            raise HTTPException(status_code=400, detail="cursor0=%d or cursor1=%d is beyond file length=%d" % (post.cursor0, post.cursor1, len(filetext)))
    else:
        post.cursor0 = -1
        post.cursor1 = -1
        post.cursor_file = ""
    ticket = TicketWithQueue("comp")
    if post.function == "highlight":
        post.max_tokens = 0
    req = post.clamp()
    req.update({
        "id": ticket.ticket,
        "object": "diff_completion_req",
        "account": account,
        "model": post.model,
        "intent": post.intent,
        "sources": post.sources,
        "cursor_file": post.cursor_file,
        "cursor0": post.cursor0,
        "cursor1": post.cursor1,
        "function": post.function,
        "max_edits": post.max_edits,
    })
    t0 = time.time()
    reqid = "call_" + ticket.ticket_safe + "_req"
    global process_dispatch_task
    while not process_dispatch_task_working:
        if not process_dispatch_task:
            process_dispatch_task = asyncio.create_task(per_process_dispatch_loop(), name="dispatch_task")
        await asyncio.sleep(0.1)
    line: StrictPipeline = await red.pipeline(transaction=False)
    await line.setex(reqid, TIMEOUT, json.dumps(req))
    await line.incr("stat_api_calls")
    kc = "stat_u_" + safe_for_redis(account) + "_calls"
    kt = "stat_u_" + safe_for_redis(account) + "_tokens"
    kmu = "stat_g_" + post.model + "__" + safe_for_redis(account)
    await line.incr(kc)
    await line.expire(kc, 86400*10)
    await line.incr(kmu)
    await line.expire(kmu, 86400*10)
    await line.append("_to_dispatch", ticket.ticket_safe + ",")
    await line.execute()
    t1 = time.time()
    log("%0.1fms create diff_completion %s" % (1000*(t1 - t0), ticket.ticket))
    return StreamingResponse(diff_streamer(ticket, post, req["created"], kt))


async def diff_streamer(ticket: TicketWithQueue, post: DiffCompletion, created_ts, kt):
    try:
        while 1:
            try:
                msg = await asyncio.wait_for(ticket.queue.get(), TIMEOUT)
            except asyncio.TimeoutError:
                msg = {"status": "error", "human_readable_message": "timeout"}
            if not post.stream:
                if msg.get("status", "") == "in_progress":
                    continue
                yield json.dumps(msg)
                break
            yield "data: " + json.dumps(msg) + "\n\n"
            if msg.get("status", "") != "in_progress":
                break
        if post.stream:
            yield "data: [DONE]" + "\n\n"
        ticket.done()
        stats_accum[kt] += msg.get("generated_tokens_n", 0)
        stats_accum["stat_m_" + post.model + "_completed"] += 1
    finally:
        if ticket.ticket is not None:
            log("  ***  diff_streamer  ***  cancelling %s" % ticket.ticket)
            stats_accum["stat_api_cancelled"] += 1
            stats_accum["stat_m_" + post.model + "_cancelled"] += 1
            red = fu.get_red()
            await red.setex("call_" + ticket.ticket_safe + "_cancelled", TIMEOUT, "1")
        ticket.done()


