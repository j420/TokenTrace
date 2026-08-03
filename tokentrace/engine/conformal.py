"""Adaptive prediction sets (APS-style) to directly target Top-3 coverage.

Given calibrated per-mode probabilities, produce the smallest ranked set whose
cumulative mass clears a threshold ``tau``. ``tau`` is fit on held-out data so the
true primary mode lands in the set with >= (1 - alpha) probability. The set size is
itself a difficulty readout: size 1 = easy, size 3-4 = genuinely ambiguous.

Cold start (unfitted): tau defaults to 0.9, giving a sensible top-until-90%-mass
set capped at ``max_size``.
"""

from __future__ import annotations


from tokentrace.core.types import ALL_MODES, FailureMode


class ConformalPredictor:
    def __init__(self, alpha: float = 0.1, max_size: int = 4, tau: float = 0.9):
        self.alpha = alpha
        self.max_size = max_size
        self.tau = tau
        self._fitted = False

    @staticmethod
    def _normalize(probs: dict[FailureMode, float]) -> dict[FailureMode, float]:
        """APS assumes a distribution, but the engine emits INDEPENDENT one-vs-rest
        marginals whose sum ranges over [0, 2]. Without this the cumulative score is
        not in [0,1]: tau fits to 1.0, set size becomes anti-correlated with real
        ambiguity, and coverage degenerates to plain top-1 accuracy.
        """
        pos = {m: max(0.0, probs.get(m, 0.0)) for m in ALL_MODES}
        tot = sum(pos.values())
        if tot <= 0:
            return pos
        return {m: v / tot for m, v in pos.items()}

    def fit(self, prob_rows: list[dict[FailureMode, float]], true_modes: list[FailureMode]) -> "ConformalPredictor":
        """Calibrate tau via the APS nonconformity score (cumulative normalized
        probability up to and including the true mode)."""
        scores: list[float] = []
        for raw, y in zip(prob_rows, true_modes):
            probs = self._normalize(raw)
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
        # Clamp below 1.0: a saturated tau makes the mass test unsatisfiable, so every
        # set would fall through to the max_size cap regardless of confidence.
        self.tau = min(float(scores[k]), 1.0 - 1e-6)
        self._fitted = True
        return self

    def predict(self, raw: dict[FailureMode, float]) -> list[FailureMode]:
        # Normalized so the accumulated mass is on the same scale as the fitted tau
        # (see _normalize). Structurally-impossible modes (prob 0, e.g. retrieval
        # modes hard-masked for non-RAG) are excluded so they never pad the set.
        probs = self._normalize(raw)
        ranked = [m for m in sorted(ALL_MODES, key=lambda m: -probs[m]) if probs[m] > 0.0]
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
