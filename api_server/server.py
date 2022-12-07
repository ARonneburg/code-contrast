import sys

import uvicorn
import logging

from datetime import datetime
from pathlib import Path
from fastapi import FastAPI

from api_server.inference import Inference
from api_server.utils import TokenHandler
from api_server.routers import GreetingsRouter
from api_server.routers import CompletionRouter
from api_server.routers import ContrastRouter


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--workdir", type=Path, default=Path("/working_volume"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    logdir = args.workdir / "logs"
    logdir.mkdir(exist_ok=True, parents=False)
    file_handler = logging.FileHandler(filename=logdir / f"server_{datetime.now():%Y-%m-%d-%H-%M-%S}.log")
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])

    token = TokenHandler(workdir=args.workdir)
    inference = Inference(workdir=args.workdir, force_cpu=args.cpu)

    app = FastAPI(docs_url=None)
    app.include_router(GreetingsRouter(token, inference))
    app.include_router(CompletionRouter(token, inference))
    app.include_router(ContrastRouter(token, inference))

    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
