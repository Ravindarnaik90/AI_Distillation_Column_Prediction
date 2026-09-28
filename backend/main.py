# -*- coding: utf-8 -*-
"""
main.py — FastAPI backend for AI-Powered Distillation Intelligence.

Endpoints:
    GET  /api/health              — artifact/load status
    GET  /api/model-info          — algorithm, feature list, static eval metrics
    GET  /api/feature-importance  — real importances pulled from model.pkl
    GET  /api/input-ranges        — slider ranges/defaults/units for the frontend
    POST /api/predict             — single prediction
    POST /api/what-if             — two predictions (current vs modified) + delta
    POST /api/batch-predict       — CSV upload -> predictions for every row
    POST /api/chat                — DistilBot (Groq if configured, else rule-based)

Also serves the static frontend/ folder at "/", so the whole app can be started
with a single `python run.py` (or `uvicorn main:app --app-dir backend`) command.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from chat_service import get_ai_response
from model_service import INPUT_RANGES, RawInputs, service
from schemas import (
    BatchPredictResponse,
    BatchPredictRow,
    ChatRequest,
    ChatResponse,
    FeatureImportanceResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    WhatIfRequest,
    WhatIfResponse,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="AI-Powered Distillation Intelligence API",
    description="Machine Learning-Based Prediction, Monitoring and Optimization of Distillation Processes",
    version="1.0.0",
)

# During local development the frontend may be opened from a different origin
# (e.g. a "Live Server" extension on :5500). In production, this same FastAPI
# app also serves the frontend directly (see the routes at the bottom of this
# file), so no cross-origin requests are needed at all. Keep the allowed
# origins list narrow and explicit instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _to_raw(req: PredictRequest) -> RawInputs:
    return RawInputs(pressure=req.pressure, t1=req.t1, L=req.L, V=req.V, D=req.D, B=req.B, F=req.F)


def _to_response(result) -> PredictResponse:
    derived = {
        "reflux_ratio": result.engineered_features["Reflux_Ratio"],
        "reboiler_intensity": result.engineered_features["Reboiler_Intensity"],
        "condenser_load": result.engineered_features["Condenser_Load"],
        "column_load": result.engineered_features["Column_Load"],
        "separation_duty": result.engineered_features["Separation_Duty"],
        "column_efficiency": result.engineered_features["Column_Efficiency"],
    }
    return PredictResponse(
        prediction=result.prediction,
        percent=result.percent,
        status=result.status,
        status_class=result.status_class,
        mass_balance_gap=result.mass_balance_gap,
        mass_balance_ok=result.mass_balance_ok,
        derived_metrics=derived,
    )


def _run_prediction(req: PredictRequest) -> PredictResponse:
    try:
        result = service.predict(_to_raw(req))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return _to_response(result)


# ─────────────────────────────────────────────────────────────────────────────
# Health / model info
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if service.is_ready else "error",
        model_loaded=service.model is not None,
        scaler_loaded=service.scaler is not None,
        feature_count=len(service.feature_names),
        error=service.load_error,
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if not service.is_ready:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {service.load_error}")
    return ModelInfoResponse(**service.model_info())


@app.get("/api/feature-importance", response_model=FeatureImportanceResponse)
def feature_importance() -> FeatureImportanceResponse:
    items = service.feature_importance()
    if not items:
        raise HTTPException(status_code=503, detail="Feature importance unavailable (model not loaded).")
    return FeatureImportanceResponse(items=items)


@app.get("/api/input-ranges")
def input_ranges() -> dict:
    return INPUT_RANGES


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    return _run_prediction(req)


@app.post("/api/what-if", response_model=WhatIfResponse)
def what_if(req: WhatIfRequest) -> WhatIfResponse:
    current = _run_prediction(req.current)
    modified = _run_prediction(req.modified)
    return WhatIfResponse(current=current, modified=modified, delta=modified.prediction - current.prediction)


REQUIRED_CSV_COLUMNS = ["pressure", "t1", "L", "V", "D", "B", "F"]


@app.post("/api/batch-predict", response_model=BatchPredictResponse)
async def batch_predict(file: UploadFile = File(...)) -> BatchPredictResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw_bytes = await file.read()
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode file as UTF-8: {exc}") from exc

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise HTTPException(status_code=400, detail="CSV file appears to be empty.")

    missing_cols = [c for c in REQUIRED_CSV_COLUMNS if c not in reader.fieldnames]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=(f"CSV is missing required column(s): {missing_cols}. "
                    f"Expected columns: {REQUIRED_CSV_COLUMNS}"),
        )

    results: List[BatchPredictRow] = []
    succeeded = 0
    for i, row in enumerate(reader, start=1):
        try:
            values = {col: float(row[col]) for col in REQUIRED_CSV_COLUMNS}
            req = PredictRequest(**values)
            resp = _run_prediction(req)
            results.append(BatchPredictRow(
                row=i, **values,
                prediction=resp.prediction, percent=resp.percent, status=resp.status,
            ))
            succeeded += 1
        except Exception as exc:  # noqa: BLE001
            safe_values = {}
            for col in REQUIRED_CSV_COLUMNS:
                try:
                    safe_values[col] = float(row.get(col, "nan"))
                except (TypeError, ValueError):
                    safe_values[col] = float("nan")
            results.append(BatchPredictRow(row=i, **safe_values, error=str(exc)))

    return BatchPredictResponse(results=results, total=len(results), succeeded=succeeded,
                                 failed=len(results) - succeeded)


@app.get("/api/batch-predict/template")
def batch_predict_template() -> StreamingResponse:
    """Download a starter CSV with the required columns and one example row."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(REQUIRED_CSV_COLUMNS)
    writer.writerow([1.01, 78.0, 780.0, 1040.0, 260.0, 340.0, 580.0])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_predict_template.csv"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    params = req.params.model_dump() if req.params else {}
    if req.params is not None:
        raw = _to_raw(req.params)
        try:
            result = service.predict(raw)
            params["reflux_ratio"] = result.engineered_features["Reflux_Ratio"]
            params["reboiler"] = result.engineered_features["Reboiler_Intensity"]
        except Exception:  # noqa: BLE001
            pass

    history = [h.model_dump() for h in req.history]
    reply, source = get_ai_response(req.message, params, req.prediction, history)
    return ChatResponse(reply=reply, source=source)


# ─────────────────────────────────────────────────────────────────────────────
# Static frontend
# ─────────────────────────────────────────────────────────────────────────────
if FRONTEND_DIR.exists():

    @app.get("/")
    def serve_index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/style.css")
    def serve_css() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "style.css")

    @app.get("/script.js")
    def serve_js() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "script.js")

    @app.get("/feature_importance.png")
    def serve_legacy_chart() -> FileResponse:
        return FileResponse(Path(__file__).parent / "feature_importance.png")
