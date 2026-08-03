"""Adaptive prediction sets (APS-style) to directly target Top-3 coverage.

Given calibrated per-mode probabilities, produce the smallest ranked set whose
cumulative mass clears a threshold ``tau``. ``tau`` is fit on held-out data so the
true primary mode lands in the set with >= (1 - alpha) probability. The set size is
itself a difficulty readout: size 1 = easy, size 3-4 = genuinely ambiguous.

Cold start (unfitted): tau defaults to 0.9, giving a sensible top-until-90%-mass
set capped at ``max_size``.
"""

from __future__ import annotations

from typing import Optional

from tokentrace.core.types import ALL_MODES, FailureMode


class ConformalPredictor:
    def __init__(self, alpha: float = 0.1, max_size: int = 4, tau: float = 0.9):
        self.alpha = alpha
        self.max_size = max_size
        self.tau = tau
        self._fitted = False

    def fit(self, prob_rows: list[dict[FailureMode, float]], true_modes: list[FailureMode]) -> "ConformalPredictor":
        """Calibrate tau via the APS nonconformity score (cumulative prob up to and
        including the true mode)."""
        scores: list[float] = []
        for probs, y in zip(prob_rows, true_modes):
            ranked = sorted(ALL_MODES, key=lambda m: -probs[m])
            cum = 0.0
            for m in ranked:
                cum += probs[m]
                if m == y:
                    break
            scores.append(cum)
        if not scores:
            return self
        scores.sort()
        n = len(scores)
        # (1-alpha) empirical quantile with finite-sample correction.
        import math

        k = min(n - 1, max(0, math.ceil((n + 1) * (1 - self.alpha)) - 1))
        self.tau = float(scores[k])
        self._fitted = True
        return self

    def predict(self, probs: dict[FailureMode, float]) -> list[FailureMode]:
        ranked = sorted(ALL_MODES, key=lambda m: -probs[m])
        out: list[FailureMode] = []
        cum = 0.0
        for m in ranked:
            out.append(m)
            cum += probs[m]
            if cum >= self.tau or len(out) >= self.max_size:
                break
        return out

    @property
    def is_fitted(self) -> bool:
        return self._fitted
