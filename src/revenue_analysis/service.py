"""Staging HTTP inference. Load a promoted local artifact at startup."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from revenue_analysis.ml import FEATURES, detect_drift, load_model, validate_frame

_state: dict[str, Any] = {}


class ScoreRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


def startup():
    directory = Path(os.environ.get("MODEL_DIR", "/app/model"))
    _state["model"], _state["report"] = load_model(directory)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    startup()
    yield
    _state.clear()


app = FastAPI(title="Revenue retention staging inference", version="1", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model_version": _state["report"]["model_version"]}


@app.post("/predict")
async def predict(request: ScoreRequest):
    try:
        if any(set(record) != set(FEATURES) for record in request.records):
            raise ValueError("Each record must contain exactly the documented feature columns")
        frame = validate_frame(pd.DataFrame(request.records), training=False)
        probabilities = _state["model"].predict_proba(frame.loc[:, FEATURES])[:, 0]
        drift = detect_drift(frame, _state["report"]["reference_profile"])
    except (SchemaError, SchemaErrors) as exc:
        raise HTTPException(status_code=422, detail="Input failed the feature schema") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"model_version": _state["report"]["model_version"],
            "p_retained": probabilities.tolist(), "drift": drift}
