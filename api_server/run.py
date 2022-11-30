import sys

import uvicorn
import logging
from pathlib import Path
from fastapi import FastAPI

from code_contrast.modeling.codify_model import CodifyModel
from api_server.routers import CompletionRouter
from api_server.routers import ContrastRouter
from api_server.inference import Inference


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workdir", type=Path, default=Path("/working_volume"))
    args = parser.parse_args()

    file_handler = logging.FileHandler(filename=args.workdir / "server.log")
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])

    with CodifyModel.from_pretrained(args.workdir / "weights", repo_id="reymondzzz/testmodel") as model:
        inference = Inference(model=model, device=args.device)

    app = FastAPI(docs_url=None)
    app.include_router(CompletionRouter(inference))
    app.include_router(ContrastRouter(inference))

    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
