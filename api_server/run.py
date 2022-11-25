import uvicorn
from fastapi import FastAPI

from api_server.routers import CompletionRouter
from api_server.routers import ContrastRouter
from api_server.inference import Inference


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)

    inference = Inference(weights="/home/mitya/model_weights")

    app = FastAPI(docs_url=None)
    app.include_router(CompletionRouter(inference))
    app.include_router(ContrastRouter(inference))

    uvicorn.run(app, host="127.0.0.1", port=8008)
