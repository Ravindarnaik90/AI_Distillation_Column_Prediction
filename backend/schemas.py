# -*- coding: utf-8 -*-
"""Pydantic request/response models for the API."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from model_service import INPUT_RANGES


def _range(key: str):
    r = INPUT_RANGES[key]
    return r["min"], r["max"]


class PredictRequest(BaseModel):
    pressure: float = Field(..., ge=_range("pressure")[0], le=_range("pressure")[1],
                             description="Column operating pressure (bar)")
    t1: float = Field(..., ge=_range("t1")[0], le=_range("t1")[1],
                       description="T1 top tray temperature (°C)")
    L: float = Field(..., ge=_range("L")[0], le=_range("L")[1],
                      description="Reflux flow (kmol/hr)")
    V: float = Field(..., ge=_range("V")[0], le=_range("V")[1],
                      description="Vapor flow (kmol/hr)")
    D: float = Field(..., ge=_range("D")[0], le=_range("D")[1],
                      description="Distillate flow (kmol/hr)")
    B: float = Field(..., ge=_range("B")[0], le=_range("B")[1],
                      description="Bottoms flow (kmol/hr)")
    F: float = Field(..., ge=_range("F")[0], le=_range("F")[1],
                      description="Feed flow (kmol/hr)")


class PredictResponse(BaseModel):
    prediction: float
    percent: float
    status: str
    status_class: str
    mass_balance_gap: float
    mass_balance_ok: bool
    derived_metrics: Dict[str, float]


class WhatIfRequest(BaseModel):
    current: PredictRequest
    modified: PredictRequest


class WhatIfResponse(BaseModel):
    current: PredictResponse
    modified: PredictResponse
    delta: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    feature_count: int
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    algorithm: Optional[str]
    feature_count: int
    feature_names: List[str]
    r2_score: float
    rmse: float
    mae: float
    accuracy_band: str
    optimal_threshold: float
    acceptable_threshold: float
    mean_feed_flow: float


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    items: List[FeatureImportanceItem]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    params: Optional[PredictRequest] = None
    prediction: Optional[float] = None


class ChatResponse(BaseModel):
    reply: str
    source: str  # "groq" | "rule-based" | "rule-based-fallback"


class BatchPredictRow(BaseModel):
    row: int
    pressure: float
    t1: float
    L: float
    V: float
    D: float
    B: float
    F: float
    prediction: Optional[float] = None
    percent: Optional[float] = None
    status: Optional[str] = None
    error: Optional[str] = None


class BatchPredictResponse(BaseModel):
    results: List[BatchPredictRow]
    total: int
    succeeded: int
    failed: int
