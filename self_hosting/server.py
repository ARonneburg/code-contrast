import logging
import asyncio
import sys

from hypercorn.config import Config
from hypercorn.asyncio import serve

from datetime import datetime
from pathlib import Path
from fastapi import FastAPI

from self_hosting.inference import Inference
from self_hosting.routers import ActivateRouter
from self_hosting.routers import CompletionRouter
from self_hosting.routers import ContrastRouter


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--token", type=str)
    args = parser.parse_args()

    logdir = args.workdir / "logs"
    logdir.mkdir(exist_ok=True, parents=False)
    file_handler = logging.FileHandler(filename=logdir / f"server_{datetime.now():%Y-%m-%d-%H-%M-%S}.log")
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])

    inference = Inference(token=args.token, workdir=args.workdir, force_cpu=args.cpu)

    app = FastAPI(docs_url=None)
    app.include_router(ActivateRouter(args.token))
    app.include_router(CompletionRouter(args.token, inference))
    app.include_router(ContrastRouter(args.token, inference))

    config = Config()
    config.bind = f"{args.host}:{args.port}"
    # config.logconfig = None
    config.accesslog = "-"
    # TODO(d.ageev): this is a hack to make the server run correct with jb
    config.keyfile = "/home/mitya/projects/code-contrast/certs/privkey1.pem"
    config.certfile = "/home/mitya/projects/code-contrast/certs/cert1.pem"

    asyncio.run(serve(app=app, config=config))
