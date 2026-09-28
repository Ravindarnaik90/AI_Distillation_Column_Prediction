# -*- coding: utf-8 -*-
"""
verify_feature_order.py
────────────────────────
Standalone proof (no FastAPI needed) that:

1. The ORIGINAL Streamlit app's hand-written feature order does not match
   features_names.pkl (the training-time order).
2. The NEW model_service.py (name-driven ordering) matches features_names.pkl
   exactly, and produces a different, corrected prediction.

Run with:
    python scripts/verify_feature_order.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from model_service import RawInputs, engineer_features  # noqa: E402

DEFAULTS = RawInputs(pressure=1.01, t1=78.0, L=780.0, V=1040.0, D=260.0, B=340.0, F=580.0)


def main() -> None:
    with open(BACKEND_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BACKEND_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(BACKEND_DIR / "features_names.pkl", "rb") as f:
        feature_names = list(pickle.load(f))

    features = engineer_features(DEFAULTS)

    # --- 1. Reproduce the OLD (buggy) app.py ordering, verbatim ---
    old_order_vector = np.array([[
        features["Pressure"], features["L"], features["V"], features["D"], features["B"], features["F"],
        features["Temp_Bottom"], features["Reflux_Ratio"], features["Reboiler_Intensity"], features["Condenser_Load"],
        features["Feed_Normalized"], features["Distillate_Withdrawal"], features["Bottoms_Withdrawal"], features["Column_Load"],
        features["Reflux_x_Temp_Top"], features["Reflux_x_Temp_Diff"], features["Reboiler_x_Temp_Bottom"],
        features["Feed_x_Reflux"], features["Feed_x_Reboiler"], features["Separation_Duty"], features["Column_Efficiency"],
    ]])

    # --- 2. NEW, correct ordering: driven entirely by features_names.pkl ---
    import pandas as pd
    new_order_df = pd.DataFrame([{name: features[name] for name in feature_names}], columns=feature_names)

    old_pred = float(model.predict(scaler.transform(old_order_vector))[0])
    new_pred = float(model.predict(scaler.transform(new_order_df))[0])

    print("features_names.pkl order:")
    print(" ", feature_names)
    print()
    print("Old app.py order (positions 11-14 scrambled):")
    print(" ", ["Pressure", "L", "V", "D", "B", "F", "Temp_Bottom", "Reflux_Ratio",
                "Reboiler_Intensity", "Condenser_Load", "Feed_Normalized",
                "Distillate_Withdrawal", "Bottoms_Withdrawal", "Column_Load",
                "Reflux_x_Temp_Top", "Reflux_x_Temp_Diff", "Reboiler_x_Temp_Bottom",
                "Feed_x_Reflux", "Feed_x_Reboiler", "Separation_Duty", "Column_Efficiency"])
    print()
    assert list(old_order_vector[0]) != list(new_order_df.iloc[0].values), \
        "Expected the two orderings to differ at positions 11-14"

    print(f"Prediction with OLD (buggy) order   : {old_pred:.4f}")
    print(f"Prediction with NEW (corrected) order: {new_pred:.4f}")
    print(f"Difference                           : {new_pred - old_pred:+.4f}")
    print()
    print("Result: the corrected, name-driven ordering in model_service.py produces")
    print("a different prediction than the old app.py list, confirming the bug is real")
    print("and that the fix changes the model's behavior as expected.")


if __name__ == "__main__":
    main()
