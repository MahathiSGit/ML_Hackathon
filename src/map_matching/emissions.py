
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


FEATURE_COLUMNS = [
    "candidate_rank",
    "distance_m",
    "heading_difference_deg",
    "road_length_m",
    "gps_speed_mps",
]


class XGBoostEmissionScorer:
    """Generate normalized log emission scores for road candidates."""

    def __init__(self, project_root=None):
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]

        project_root = Path(project_root)

        model_path = (
            project_root / "models" / "artifacts" / "xgboost_tuned.json"
        )
        imputer_path = (
            project_root / "models" / "artifacts" / "median_imputer.joblib"
        )

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not imputer_path.exists():
            raise FileNotFoundError(f"Imputer not found: {imputer_path}")

        self.model = XGBClassifier()
        self.model.load_model(str(model_path))

        self.imputer = joblib.load(imputer_path)

    def score(self, candidate_rows):
        """
        Score candidates and return normalized log emissions.

        candidate_rows: DataFrame containing FEATURE_COLUMNS.

        Returns:
            numpy array of normalized log emission scores.
        """
        if candidate_rows.empty:
            return np.array([], dtype=float)

        missing = [
            column
            for column in FEATURE_COLUMNS
            if column not in candidate_rows.columns
        ]

        if missing:
            raise ValueError(f"Missing candidate features: {missing}")

        X = candidate_rows[FEATURE_COLUMNS].copy()

        X = X.replace([np.inf, -np.inf], np.nan)
        X = self.imputer.transform(X)

        probabilities = self.model.predict_proba(X)[:, 1]

        # Avoid log(0).
        probabilities = np.clip(probabilities, 1e-9, 1.0)

        # Normalize candidate scores within each observation.
        log_scores = np.log(probabilities)
        log_scores -= np.logaddexp.reduce(log_scores)

        return log_scores
