"""Prediction API: /health, /predict."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .model_loader import load_model

CLASSES = ["setosa", "versicolor", "virginica"]
state = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Approach B downloads the model from S3 here; requests are served only after it loads.
    state["model"] = load_model()
    yield


app = FastAPI(title="iris-api", lifespan=lifespan)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/health")
def health() -> dict:
    # git SHA + model source show which image and model are running.
    return {
        "status": "ok",
        "git_sha": os.getenv("GIT_SHA", "unknown"),
        "model_source": os.getenv("MODEL_SOURCE", "local"),
        "model_uri": os.getenv("MODEL_URI"),
    }


@app.post("/predict")
def predict(features: IrisFeatures) -> dict:
    x = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    proba = state["model"].predict_proba(x)[0]
    return {
        "class": CLASSES[int(proba.argmax())],
        "probabilities": dict(zip(CLASSES, proba.round(4).tolist())),
    }
