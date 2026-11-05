"""Inference and What-If Simulator Service for Student Academic Risk (Phase 6).

Exposes real-time model scoring, scenario simulation (what-if analysis),
and individualized intervention recommenders.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from ml.features import FEATURE_COLUMNS

MODELS_DIR = BASE_DIR / "models"


class RiskPredictorService:
    _instance: RiskPredictorService | None = None

    def __init__(self):
        self.model = None
        self.regressor = None
        self.kmeans = None
        self.predictions_df = None
        self._load_artifacts()

    @classmethod
    def get_instance(cls) -> RiskPredictorService:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_artifacts(self):
        model_path = MODELS_DIR / "risk_model_rf.joblib"
        reg_path = MODELS_DIR / "sgpa_predictor.joblib"
        pred_csv = MODELS_DIR / "ml_predictions.csv"

        if model_path.exists():
            self.model = joblib.load(model_path)
        if reg_path.exists():
            self.regressor = joblib.load(reg_path)
        if pred_csv.exists():
            self.predictions_df = pd.read_csv(pred_csv)

    def get_student_risk(self, enrollment_no: str) -> dict[str, Any] | None:
        if self.predictions_df is None:
            self._load_artifacts()
        if self.predictions_df is None or self.predictions_df.empty:
            return None

        key = enrollment_no.strip().upper()
        match = self.predictions_df[self.predictions_df["Enrollment_No"].str.upper() == key]
        if match.empty:
            return None
        row = match.iloc[0].to_dict()

        # Parse string lists if stored as string representations
        import ast
        for list_col in ("Top_Reasons", "Interventions"):
            if isinstance(row.get(list_col), str) and row[list_col].startswith("["):
                try:
                    row[list_col] = ast.literal_eval(row[list_col])
                except Exception:
                    pass
        return row

    def simulate_what_if(
        self,
        base_features: dict[str, float],
        attendance_delta: float = 0.0,
        internal_delta: float = 0.0,
    ) -> dict[str, Any]:
        """Simulate change in risk probability given hypothetical improvements."""
        if self.model is None:
            self._load_artifacts()
            if self.model is None:
                return {"simulated_prob": 0.5, "risk_band": "Medium", "delta": 0.0}

        row = base_features.copy()
        row["overall_attendance_pct"] = float(np.clip(row.get("overall_attendance_pct", 75.0) + attendance_delta, 0.0, 100.0))
        row["internal_marks_avg_pct"] = float(np.clip(row.get("internal_marks_avg_pct", 65.0) + internal_delta, 0.0, 100.0))

        # Re-derive subjects below 75 if attendance increased significantly
        if attendance_delta > 5.0 and row.get("subjects_below_75", 0) > 0:
            row["subjects_below_75"] = max(0, int(row["subjects_below_75"] - 1))

        df_input = pd.DataFrame([row])[FEATURE_COLUMNS]
        sim_prob = float(self.model.predict_proba(df_input)[0, 1])
        orig_prob = base_features.get("original_prob", sim_prob)

        return {
            "simulated_prob": round(sim_prob, 4),
            "simulated_pct": round(sim_prob * 100.0, 1),
            "risk_band": config.get_risk_band(sim_prob),
            "prob_reduction": round((orig_prob - sim_prob) * 100.0, 1),
            "new_attendance": round(row["overall_attendance_pct"], 1),
            "new_internals": round(row["internal_marks_avg_pct"], 1),
        }

