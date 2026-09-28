# -*- coding: utf-8 -*-
"""
model_service.py
─────────────────
Loads the trained artifacts (model.pkl, scaler.pkl, features_names.pkl) and
performs feature engineering + prediction.

WHY THIS FILE EXISTS (bug fix context)
---------------------------------------
The original Streamlit app (`app.py`) built its 21-value input vector as a
hand-written Python list:

    input_vector = np.array([[
        pressure, l_input, v_input, d_input, b_input, f_input,
        temp_bottom, reflux_ratio, reboiler_intensity, condenser_load,
        feed_normalized, distillate_w, bottoms_w, column_load,     # <- WRONG ORDER
        reflux_x_temp_top, reflux_x_temp_diff, reboiler_x_temp_bot,
        feed_x_reflux, feed_x_reboiler, separation_duty, column_efficiency
    ]])

but the model was trained on columns in the order stored in
`features_names.pkl`:

    ... Condenser_Load, Distillate_Withdrawal, Bottoms_Withdrawal,
        Column_Load, Feed_Normalized, ...

Positions 11-14 were scrambled (`feed_normalized` was placed where
`Distillate_Withdrawal` belongs, and everything after it shifted by one).
This silently fed the scaler/model the wrong numbers for those four
features on every single prediction the old app ever made.

FIX: instead of relying on a hand-ordered list, we build a *name -> value*
dictionary, and construct the final vector with:

    vector = [features[name] for name in feature_names]

`feature_names` is read directly from `features_names.pkl` (the ground
truth used at training time), so the ordering can never drift again even
if new features are added or the dict's insertion order changes.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ARTIFACT_DIR = Path(__file__).parent

# Small constant to avoid division-by-zero on V / F, matches original app.py
EPSILON = 1e-6

# Mean feed flow used to normalize F during training (see model_training_script.py
# STEP 3, "Feed_Normalized" — printed as "Mean Feed Flow (F)" during training and
# hardcoded into app.py as MEAN_FEED_FLOW). Preserved exactly as in the original app.
MEAN_FEED_FLOW = 545.23

# Purity classification thresholds, taken verbatim from app.py (STEP 11 / line ~742)
OPTIMAL_THRESHOLD = 0.82
ACCEPTABLE_THRESHOLD = 0.75

# Slider ranges/defaults, taken verbatim from app.py (STEP 8, lines ~604-620)
INPUT_RANGES = {
    "pressure": {"min": 0.5, "max": 3.0, "default": 1.01, "step": 0.01, "unit": "bar",
                 "label": "Pressure"},
    "t1": {"min": 60.0, "max": 120.0, "default": 78.0, "step": 0.5, "unit": "°C",
           "label": "T1 — Top Tray Temperature"},
    "L": {"min": 300.0, "max": 1200.0, "default": 780.0, "step": 10.0, "unit": "kmol/hr",
          "label": "Reflux Flow (L)"},
    "V": {"min": 600.0, "max": 1500.0, "default": 1040.0, "step": 10.0, "unit": "kmol/hr",
          "label": "Vapor Flow (V)"},
    "D": {"min": 100.0, "max": 500.0, "default": 260.0, "step": 10.0, "unit": "kmol/hr",
          "label": "Distillate (D)"},
    "B": {"min": 100.0, "max": 500.0, "default": 340.0, "step": 10.0, "unit": "kmol/hr",
          "label": "Bottoms (B)"},
    "F": {"min": 350.0, "max": 700.0, "default": 580.0, "step": 10.0, "unit": "kmol/hr",
          "label": "Feed Flow (F)"},
}


@dataclass
class RawInputs:
    pressure: float
    t1: float
    L: float
    V: float
    D: float
    B: float
    F: float


@dataclass
class PredictionResult:
    prediction: float
    percent: float
    status: str            # OPTIMAL | ACCEPTABLE | BELOW SPEC
    status_class: str      # good | warn | danger
    mass_balance_gap: float
    mass_balance_ok: bool
    engineered_features: Dict[str, float]


def engineer_features(raw: RawInputs) -> Dict[str, float]:
    """
    Reproduce the exact feature-engineering formulas from app.py / model_training_script.py.
    Returns a name -> value dict (order doesn't matter here; ordering is applied later
    using features_names.pkl).
    """
    v_safe = max(raw.V, EPSILON)
    f_safe = max(raw.F, EPSILON)

    temp_top = raw.t1
    # NOTE: the deployed app only collects T1 from the user (not the full T1..T14
    # temperature profile used at training time), so it approximates the reboiler
    # temperature as T1 + 20°C. Preserved exactly as in the original app.
    temp_bottom = raw.t1 + 20.0
    temp_diff = temp_bottom - temp_top

    feed_normalized = raw.F / MEAN_FEED_FLOW

    reflux_ratio = raw.L / v_safe
    reboiler_intensity = raw.V / f_safe
    condenser_load = raw.L / f_safe
    distillate_withdrawal = raw.D / f_safe
    bottoms_withdrawal = raw.B / f_safe
    column_load = (raw.L + raw.V) / f_safe

    reflux_x_temp_top = reflux_ratio * temp_top
    reflux_x_temp_diff = reflux_ratio * temp_diff
    reboiler_x_temp_bottom = reboiler_intensity * temp_bottom
    feed_x_reflux = feed_normalized * reflux_ratio
    feed_x_reboiler = feed_normalized * reboiler_intensity

    separation_duty = reflux_ratio * reboiler_intensity
    column_efficiency = reflux_ratio * column_load

    return {
        "Pressure": raw.pressure,
        "L": raw.L,
        "V": raw.V,
        "D": raw.D,
        "B": raw.B,
        "F": raw.F,
        "Temp_Bottom": temp_bottom,
        "Reflux_Ratio": reflux_ratio,
        "Reboiler_Intensity": reboiler_intensity,
        "Condenser_Load": condenser_load,
        "Distillate_Withdrawal": distillate_withdrawal,
        "Bottoms_Withdrawal": bottoms_withdrawal,
        "Column_Load": column_load,
        "Feed_Normalized": feed_normalized,
        "Reflux_x_Temp_Top": reflux_x_temp_top,
        "Reflux_x_Temp_Diff": reflux_x_temp_diff,
        "Reboiler_x_Temp_Bottom": reboiler_x_temp_bottom,
        "Feed_x_Reflux": feed_x_reflux,
        "Feed_x_Reboiler": feed_x_reboiler,
        "Separation_Duty": separation_duty,
        "Column_Efficiency": column_efficiency,
    }


class DistillationModelService:
    """Loads artifacts once and serves predictions using the correct,
    name-driven feature ordering."""

    def __init__(self, artifact_dir: Path = ARTIFACT_DIR):
        self.artifact_dir = artifact_dir
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.load_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        try:
            with open(self.artifact_dir / "model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open(self.artifact_dir / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open(self.artifact_dir / "features_names.pkl", "rb") as f:
                self.feature_names = list(pickle.load(f))
        except Exception as exc:  # noqa: BLE001 - surfaced via /api/health
            self.load_error = str(exc)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None and bool(self.feature_names)

    def build_ordered_vector(self, features: Dict[str, float]) -> pd.DataFrame:
        """The core bug fix: order the vector using features_names.pkl, by name,
        never by hand-written position.

        Returns a single-row DataFrame with columns in the exact order (and with the
        exact names) StandardScaler was fitted on, which also avoids the sklearn
        "X does not have valid feature names" warning that the original app ignored.
        """
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise KeyError(f"Missing engineered feature(s) required by the model: {missing}")
        ordered = {name: [features[name]] for name in self.feature_names}
        return pd.DataFrame(ordered, columns=self.feature_names)

    def predict(self, raw: RawInputs) -> PredictionResult:
        if not self.is_ready:
            raise RuntimeError(f"Model artifacts not loaded: {self.load_error}")

        features = engineer_features(raw)
        vector = self.build_ordered_vector(features)

        scaled = self.scaler.transform(vector)
        pred = float(self.model.predict(scaled)[0])
        pct = pred * 100.0

        if pred >= OPTIMAL_THRESHOLD:
            status, status_class = "OPTIMAL", "good"
        elif pred >= ACCEPTABLE_THRESHOLD:
            status, status_class = "ACCEPTABLE", "warn"
        else:
            status, status_class = "BELOW SPEC", "danger"

        mb_gap = abs((raw.D + raw.B) - raw.F)

        return PredictionResult(
            prediction=pred,
            percent=pct,
            status=status,
            status_class=status_class,
            mass_balance_gap=mb_gap,
            mass_balance_ok=mb_gap <= 30,
            engineered_features=features,
        )

    def feature_importance(self) -> List[Dict[str, float]]:
        """Real feature importances pulled from the loaded model, paired with
        their names from features_names.pkl (same ordering guarantee as prediction)."""
        if not self.is_ready or not hasattr(self.model, "feature_importances_"):
            return []
        importances = self.model.feature_importances_
        pairs = list(zip(self.feature_names, (float(x) for x in importances)))
        pairs.sort(key=lambda p: p[1], reverse=True)
        return [{"feature": name, "importance": value} for name, value in pairs]

    def model_info(self) -> Dict:
        return {
            "algorithm": type(self.model).__name__ if self.model is not None else None,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            # Static evaluation metrics reported by the original training run
            # (see model_training_script.py STEP 6/9 and app.py sidebar).
            "r2_score": 0.98,
            "rmse": 0.0155,
            "mae": 0.0122,
            "accuracy_band": "±1.55%",
            "optimal_threshold": OPTIMAL_THRESHOLD,
            "acceptable_threshold": ACCEPTABLE_THRESHOLD,
            "mean_feed_flow": MEAN_FEED_FLOW,
        }


# Singleton used by the FastAPI app
service = DistillationModelService()
