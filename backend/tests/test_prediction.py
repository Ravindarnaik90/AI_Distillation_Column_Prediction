# -*- coding: utf-8 -*-
"""
Test suite covering:
  - artifact loading (model.pkl / scaler.pkl / features_names.pkl)
  - correct feature ordering (matches features_names.pkl exactly)
  - the specific ordering-bug regression (old vs new prediction differ)
  - prediction on default inputs
  - the FastAPI endpoints (/api/health, /api/predict, /api/model-info, etc.)

Run with:
    pytest backend/tests/test_prediction.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_service import RawInputs, engineer_features, service  # noqa: E402

DEFAULT_INPUTS = RawInputs(pressure=1.01, t1=78.0, L=780.0, V=1040.0, D=260.0, B=340.0, F=580.0)


# ── Artifact loading ──────────────────────────────────────────────────────
def test_model_loads():
    assert service.model is not None, f"model.pkl failed to load: {service.load_error}"


def test_scaler_loads():
    assert service.scaler is not None, f"scaler.pkl failed to load: {service.load_error}"


def test_feature_names_load():
    assert len(service.feature_names) == 21
    assert service.is_ready


# ── Feature ordering (the bug fix) ────────────────────────────────────────
def test_engineered_features_have_21_keys():
    features = engineer_features(DEFAULT_INPUTS)
    assert len(features) == 21


def test_ordered_vector_matches_features_names_pkl_exactly():
    features = engineer_features(DEFAULT_INPUTS)
    df = service.build_ordered_vector(features)
    assert list(df.columns) == service.feature_names
    assert df.shape == (1, 21)


def test_ordered_vector_raises_on_missing_feature():
    features = engineer_features(DEFAULT_INPUTS)
    del features["Column_Load"]
    with pytest.raises(KeyError):
        service.build_ordered_vector(features)


def test_old_buggy_order_differs_from_corrected_order():
    """Regression test for the original app.py bug: positions 11-14
    (Distillate_Withdrawal, Bottoms_Withdrawal, Column_Load, Feed_Normalized)
    were scrambled. Confirm the corrected order actually changes the vector
    and the resulting prediction."""
    import numpy as np

    features = engineer_features(DEFAULT_INPUTS)

    old_buggy_vector = np.array([[
        features["Pressure"], features["L"], features["V"], features["D"], features["B"], features["F"],
        features["Temp_Bottom"], features["Reflux_Ratio"], features["Reboiler_Intensity"], features["Condenser_Load"],
        features["Feed_Normalized"], features["Distillate_Withdrawal"], features["Bottoms_Withdrawal"], features["Column_Load"],
        features["Reflux_x_Temp_Top"], features["Reflux_x_Temp_Diff"], features["Reboiler_x_Temp_Bottom"],
        features["Feed_x_Reflux"], features["Feed_x_Reboiler"], features["Separation_Duty"], features["Column_Efficiency"],
    ]])
    corrected_df = service.build_ordered_vector(features)

    assert list(old_buggy_vector[0]) != list(corrected_df.iloc[0].values)

    old_pred = float(service.model.predict(service.scaler.transform(old_buggy_vector))[0])
    new_pred = float(service.model.predict(service.scaler.transform(corrected_df))[0])
    assert old_pred != pytest.approx(new_pred, abs=1e-6)


# ── Prediction ─────────────────────────────────────────────────────────────
def test_default_prediction_runs_and_is_in_range():
    result = service.predict(DEFAULT_INPUTS)
    assert 0.0 <= result.prediction <= 1.0
    assert result.status in ("OPTIMAL", "ACCEPTABLE", "BELOW SPEC")


def test_default_prediction_matches_known_corrected_value():
    """Locks in the corrected value found during the migration investigation."""
    result = service.predict(DEFAULT_INPUTS)
    assert result.prediction == pytest.approx(0.8246, abs=0.001)


def test_feature_importance_sums_reasonably():
    items = service.feature_importance()
    assert len(items) == 21
    total = sum(i["importance"] for i in items)
    assert total == pytest.approx(1.0, abs=0.05)
    # sorted descending
    values = [i["importance"] for i in items]
    assert values == sorted(values, reverse=True)


# ── FastAPI endpoints ───────────────────────────────────────────────────────
@pytest.fixture
def client():
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


def test_api_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["feature_count"] == 21


def test_api_model_info(client):
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["feature_count"] == 21
    assert len(body["feature_names"]) == 21


def test_api_predict_default(client):
    payload = {"pressure": 1.01, "t1": 78.0, "L": 780.0, "V": 1040.0, "D": 260.0, "B": 340.0, "F": 580.0}
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == pytest.approx(0.8246, abs=0.001)
    assert body["status"] == "OPTIMAL"


def test_api_predict_rejects_out_of_range_input(client):
    payload = {"pressure": 999, "t1": 78.0, "L": 780.0, "V": 1040.0, "D": 260.0, "B": 340.0, "F": 580.0}
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 422  # pydantic validation error


def test_api_feature_importance(client):
    resp = client.get("/api/feature-importance")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 21


def test_api_what_if(client):
    base = {"pressure": 1.01, "t1": 78.0, "L": 780.0, "V": 1040.0, "D": 260.0, "B": 340.0, "F": 580.0}
    modified = dict(base, L=900.0)
    resp = client.post("/api/what-if", json={"current": base, "modified": modified})
    assert resp.status_code == 200
    body = resp.json()
    assert "delta" in body


def test_api_batch_predict_csv(client):
    csv_content = "pressure,t1,L,V,D,B,F\n1.01,78.0,780.0,1040.0,260.0,340.0,580.0\n1.05,80.0,800.0,1100.0,270.0,330.0,590.0\n"
    files = {"file": ("sample.csv", csv_content, "text/csv")}
    resp = client.post("/api/batch-predict", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert body["succeeded"] == 2


def test_api_batch_predict_rejects_missing_columns(client):
    csv_content = "pressure,t1\n1.01,78.0\n"
    files = {"file": ("sample.csv", csv_content, "text/csv")}
    resp = client.post("/api/batch-predict", files=files)
    assert resp.status_code == 400


def test_api_chat_rule_based_without_key(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    resp = client.post("/api/chat", json={"message": "is this purity good?", "prediction": 0.83})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "rule-based"
    assert len(body["reply"]) > 0
