"""Learned residual heads (LightGBM, one-vs-rest, multi-label).

Design: the rule layer's per-mode log-odds is used as each head's ``init_score``,
so the trees fit only the *residual* between the interpretable prior and the
truth. The total pre-calibration logit is therefore

    z_mode = rule_logit_mode  +  trees_raw_margin_mode

which is one additive, decomposable ledger. Cold start (no training data) => no
models => residual 0 => the system runs on pure rules.

LightGBM is chosen for native NaN handling (missing families route around a
default split), interaction capture (dilution needs position x length x attention),
seconds-fast CPU training, and exact TreeSHAP attribution.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from tokentrace.core.types import ALL_MODES, FailureMode, FeatureVector
from tokentrace.signals.features import ALL_FEATURES


class ResidualClassifier:
    def __init__(self, feature_names: Optional[list[str]] = None):
        self.feature_names = feature_names or ALL_FEATURES
        self.models: dict[FailureMode, object] = {}

    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: np.ndarray,             # [N, F] features (NaN allowed)
        rule_logits: np.ndarray,   # [N, 5] per-mode prior logit (init_score)
        Y: np.ndarray,             # [N, 5] binary multi-label
        sample_weight: Optional[np.ndarray] = None,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        min_child_samples: int = 10,
    ) -> "ResidualClassifier":
        from lightgbm import LGBMClassifier

        for i, mode in enumerate(ALL_MODES):
            y = Y[:, i].astype(int)
            init = rule_logits[:, i].astype(float)
            if len(set(y.tolist())) < 2:
                # Degenerate label column -> keep pure-rule prior for this mode.
                continue
            model = LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                verbose=-1,
            )
            model.fit(X, y, init_score=init, sample_weight=sample_weight)
            self.models[mode] = model
        return self

    # ------------------------------------------------------------------ #
    def residual(self, fv: FeatureVector) -> dict[FailureMode, float]:
        """Trees-only raw margin per mode (0 where a head was not trained)."""
        if not self.models:
            return {m: 0.0 for m in ALL_MODES}
        x = fv.to_array(self.feature_names).reshape(1, -1)
        out: dict[FailureMode, float] = {}
        for m in ALL_MODES:
            model = self.models.get(m)
            out[m] = float(model.predict(x, raw_score=True)[0]) if model else 0.0
        return out

    def shap(self, fv: FeatureVector) -> dict[FailureMode, dict[str, float]]:
        """Per-mode TreeSHAP feature contributions (log-odds) for the learned part."""
        if not self.models:
            return {m: {} for m in ALL_MODES}
        x = fv.to_array(self.feature_names).reshape(1, -1)
        out: dict[FailureMode, dict[str, float]] = {}
        for m in ALL_MODES:
            model = self.models.get(m)
            if not model:
                out[m] = {}
                continue
            contrib = model.predict(x, pred_contrib=True)[0]  # [F+1], last = base
            out[m] = {
                self.feature_names[j]: float(contrib[j])
                for j in range(len(self.feature_names))
                if abs(contrib[j]) > 1e-6
            }
        return out

    @property
    def is_fitted(self) -> bool:
        return bool(self.models)

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        blob = {"feature_names": self.feature_names, "models": self.models}
        Path(path).write_bytes(pickle.dumps(blob))

    def load(self, path: str | Path) -> "ResidualClassifier":
        blob = pickle.loads(Path(path).read_bytes())
        self.feature_names = blob["feature_names"]
        self.models = blob["models"]
        return self
