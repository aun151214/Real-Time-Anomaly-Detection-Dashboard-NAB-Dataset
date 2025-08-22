
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class IFDetector:
    """Simple wrapper around IsolationForest for 1-D time series."""
    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
        )

    def fit(self, y: pd.Series):
        arr = np.asarray(y).reshape(-1, 1)
        self.model.fit(arr)
        return self

    def predict_scores(self, y: pd.Series) -> pd.Series:
        arr = np.asarray(y).reshape(-1, 1)
        scores = -self.model.decision_function(arr)  # larger => more abnormal
        return pd.Series(scores, index=y.index, name="score")

    def predict_labels(self, y: pd.Series, threshold: float | None = None) -> pd.Series:
        """Return 1 for anomaly, 0 for normal. If threshold None, use model's default."""
        if threshold is None:
            arr = np.asarray(y).reshape(-1, 1)
            raw = self.model.predict(arr)
            labels = (raw == -1).astype(int)
            return pd.Series(labels, index=y.index, name="pred")
        else:
            scores = self.predict_scores(y)
            labels = (scores >= threshold).astype(int)
            labels.name = "pred"
            return labels
