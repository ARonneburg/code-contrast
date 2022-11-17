import sys, time, threading, random, json, re
import asyncio
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deploy_front_py import fastapi_highlevel_calls
from deploy_front_py import fastapi_nlp
from deploy_front_py import fastapi_infengine
from deploy_front_py import tv
from deploy_front_py import fastapi_utils as fu
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


app = FastAPI(docs_url=None, redoc_url="/watch-meh")
app.include_router(fastapi_highlevel_calls.router, prefix="/highlevel-v1")
app.include_router(fastapi_nlp.router, prefix="/v1")
app.include_router(fastapi_infengine.router, prefix="/infengine-v1")
app.include_router(tv.router, prefix="/tv")


origins = [
    "https://max.smallcloud.ai",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    fu.red_startup()


@app.on_event("shutdown")
async def shutdown():
    fu.red_shutdown()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/ping")
def ping_handler():
    return {"message": "pong"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8008, debug=True)
    #, loop="asyncio")

# Or run:
# uvicorn deploy_front_py.python_fastapi_server:app --host 0.0.0.0 --port 8008
