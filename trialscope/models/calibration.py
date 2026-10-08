"""Probability calibration. Platt (sigmoid) scaling instead of isotonic regression.

Isotonic regression fitted on a validation year with ~1,000 trials produces step functions whose top step can sit at
exactly 1.0 (a handful of terminated trials at the highest scores), so unusual trials were scored "99% risk".
A sigmoid is monotone, smooth, cannot saturate that way, and had a lower out-of-time Brier score here (0.086 vs 0.089).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    def fit(self, raw, y) -> "PlattCalibrator":
        self.lr_ = LogisticRegression(C=1e6).fit(_logit(raw), np.asarray(y).astype(int))
        return self

    def predict(self, raw) -> np.ndarray:
        return self.lr_.predict_proba(_logit(raw))[:, 1]
