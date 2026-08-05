"""Per-mode, per-missingness-signature probability calibration.

Raw score distributions shift when a signal family is absent, so a single global
map would make black-box probabilities meaningless. We therefore fit a separate
isotonic map for each (mode, missingness-signature) that has enough data, and fall
back to (mode, global) then to a plain sigmoid. This is what keeps Top-3 and the
abstention thresholds meaningful under tier degradation.
"""

from __future__ import annotations

import math
import pickle
from collections import defaultdict
from pathlib import Path

from tokentrace.core.types import FailureMode


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _coarse(signature: str) -> str:
    """Coarse bucket for the fallback ladder: keep only whether a usable reference
    was available, discarding which signal families were present."""
    return "__noref__" if signature.endswith("|noref") else "__ref__"


class _Platt:
    """Two-parameter logistic calibration (sigmoid(a*z + b)).

    Picklable and dependency-light; used where isotonic would overfit.
    """

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0

    def fit(self, zs: list[float], ys: list[int], max_iter: int = 60) -> "_Platt":
        """Platt scaling via the Lin-Weng-Lin (2007) formulation.

        Implemented directly rather than through ``LogisticRegression``: our logits
        are near-separable (the residual head saturates around +-10), which makes a
        general-purpose solver run its full iteration budget on every bucket — it
        turned a 5s ablation run into minutes. The Lin et al. target smoothing makes
        the objective finite under separability, so this converges in a few Newton
        steps.
        """
        n_pos = sum(1 for y in ys if y > 0)
        n_neg = len(ys) - n_pos
        hi = (n_pos + 1.0) / (n_pos + 2.0)
        lo = 1.0 / (n_neg + 2.0)
        t = [hi if y > 0 else lo for y in ys]

        a, b = 0.0, math.log((n_neg + 1.0) / (n_pos + 1.0))
        for _ in range(max_iter):
            # gradient + Hessian of the regularized log-loss
            g1 = g2 = h11 = h22 = h21 = 0.0
            for z, ti in zip(zs, t):
                fx = a * z + b
                p = sigmoid(fx)
                d1, d2 = p - ti, p * (1.0 - p)
                g1 += z * d1
                g2 += d1
                h11 += z * z * d2
                h22 += d2
                h21 += z * d2
            if abs(g1) < 1e-6 and abs(g2) < 1e-6:
                break
            det = h11 * h22 - h21 * h21
            if abs(det) < 1e-12:
                break
            da = -(h22 * g1 - h21 * g2) / det
            db = -(-h21 * g1 + h11 * g2) / det
            a, b = a + da, b + db
            if abs(da) < 1e-8 and abs(db) < 1e-8:
                break
        self.a, self.b = a, b
        return self

    def predict(self, zs):
        return [sigmoid(self.a * float(z) + self.b) for z in zs]


class Calibrator:
    def __init__(self, min_samples: int = 40, isotonic_min: int = 60):
        self.min_samples = min_samples
        #: Below this many records a bucket is calibrated with Platt rather than
        #: isotonic (a non-parametric fit needs substantially more data; on a thin
        #: bucket isotonic overfits and measured WORSE than the raw sigmoid).
        self.isotonic_min = isotonic_min
        self.maps: dict[tuple[str, str], object] = {}
        #: 0 exact per-signature map, 1 coarse ref/noref, 2 global, 3 raw sigmoid.
        #: Surfaced so a caller can tell the user a probability came from a pooled map.
        self.last_fallback: int = 0
        #: How many reference-free records the ``__global__`` pool was fitted from.
        #: See :attr:`global_admits_noref` — a *presence* test is not enough.
        self.n_noref: int = 0

    @property
    def global_admits_noref(self) -> bool:
        """May a reference-free trace be served the pooled ``__global__`` map?

        Only when that pool contains ENOUGH reference-free records to have shaped
        it. An earlier version asked merely whether it had seen *any*, which a
        single stray record satisfied — 400 reference-bearing records plus one
        reference-free one produced a 99.75%-reference-bearing ``__global__`` that
        the guard then happily served, restoring the exact bug it was written to
        stop. That is reachable in the real workflow: one row with
        ``ground_truth=None`` in the calibration split (``load_ragtruth`` produces
        them) would have disabled the guard globally.

        The bar is ``min_samples`` — the same data-sufficiency threshold every
        bucket must clear to be fitted at all, rather than a new invented constant.
        """
        return self.n_noref >= self.min_samples

    def fit(self, records: list[dict]) -> "Calibrator":
        """records: dicts with keys ``mode`` (FailureMode), ``signature`` (str),
        ``z`` (float raw logit), ``label`` (0/1).

        Refitting REPLACES prior state. Accumulating instead let stale maps from a
        discarded training set survive a refit and keep being served.
        """
        self.maps = {}
        self.n_noref = 0

        groups: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
        for r in records:
            mode = r["mode"].value if isinstance(r["mode"], FailureMode) else str(r["mode"])
            sig = r["signature"]
            groups[(mode, sig)].append((r["z"], int(r["label"])))
            # Intermediate bucket so a thin signature degrades one STEP (to traces
            # with the same reference-availability) rather than straight to a fully
            # pooled map — the pooling this module exists to avoid.
            groups[(mode, _coarse(sig))].append((r["z"], int(r["label"])))
            groups[(mode, "__global__")].append((r["z"], int(r["label"])))
            if sig.endswith("|noref"):
                self.n_noref += 1

        for key, pairs in groups.items():
            if len(pairs) < self.min_samples:
                continue
            ys = [y for _, y in pairs]
            if len(set(ys)) < 2:
                continue
            zs = [z for z, _ in pairs]
            fitted = self._fit_validated(zs, ys)
            if fitted is not None:
                self.maps[key] = fitted
        return self

    # ------------------------------------------------------------------ #
    def _fit_validated(self, zs: list[float], ys: list[int]):
        """Fit a calibration map and KEEP IT ONLY IF IT BEATS THE RAW SIGMOID.

        Calibration is not free: when the underlying scores are already sharp and
        accurate (a saturated residual head on a separable corpus), any smoothing
        map makes probabilities strictly worse — measured ECE 0.024 calibrated vs
        0.010 raw. Silently applying it anyway would degrade exactly the numbers this
        module exists to improve.

        So we hold out a deterministic third of the bucket, compare the candidate's
        Brier score against the identity (raw sigmoid) baseline, and only adopt the
        map when it actually helps. Isotonic is tried on large buckets and Platt on
        thin ones (a non-parametric fit overfits when data is scarce).
        """
        from sklearn.isotonic import IsotonicRegression

        n = len(zs)
        idx = sorted(range(n), key=lambda i: (hash((round(zs[i], 6), ys[i], i)) & 0xFFFF))
        hold = set(idx[: max(1, n // 3)])
        tr_z = [zs[i] for i in range(n) if i not in hold]
        tr_y = [ys[i] for i in range(n) if i not in hold]
        va_z = [zs[i] for i in range(n) if i in hold]
        va_y = [ys[i] for i in range(n) if i in hold]
        if len(set(tr_y)) < 2 or not va_z:
            return None

        def brier(preds, ys_):
            return sum((p - y) ** 2 for p, y in zip(preds, ys_)) / len(ys_)

        baseline = brier([sigmoid(z) for z in va_z], va_y)

        if n >= self.isotonic_min:
            cand = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            cand.fit(tr_z, tr_y)
        else:
            cand = _Platt().fit(tr_z, tr_y)
        if brier(list(cand.predict(va_z)), va_y) >= baseline:
            return None   # calibration does not help here; keep the raw sigmoid

        # Adopted: refit on the full bucket now that the family is validated.
        if n >= self.isotonic_min:
            final = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            final.fit(zs, ys)
            return final
        return _Platt().fit(zs, ys)

    def transform(self, mode: FailureMode, signature: str, z: float) -> float:
        mkey = mode.value if isinstance(mode, FailureMode) else str(mode)
        ladder = [(mkey, signature), (mkey, _coarse(signature))]
        # The __global__ rung is only admissible if the pool it was fitted from
        # actually contains records like this trace. A calibrator trained on
        # reference-bearing data alone has a __global__ that IS the reference-bearing
        # map under another name, so offering it to a reference-free trace defeats
        # the |noref guard one rung below where _coarse() blocks it — measured as
        # 326/412 production-shape probabilities served by a pooled reference-bearing
        # map, ECE 0.0053 -> 0.0691. Falling through to the raw sigmoid is the honest
        # outcome: no map fits this trace, so do not pretend one does.
        if self.global_admits_noref or not signature.endswith("|noref"):
            ladder.append((mkey, "__global__"))
        for level, key in enumerate(ladder):
            m = self.maps.get(key)
            if m is not None:
                self.last_fallback = level      # 0 = exact map, >0 = pooled
                return float(m.predict([z])[0])
        self.last_fallback = 3                  # uncalibrated sigmoid
        return sigmoid(z)

    @property
    def is_fitted(self) -> bool:
        return bool(self.maps)

    def save(self, path: str | Path) -> None:
        # The fitted maps alone are NOT the calibrator's state. Persisting only
        # `maps` silently dropped n_noref (so a calibrator legitimately fitted on
        # reference-free data denied itself the __global__ rung after a reload) and
        # reset the data-sufficiency thresholds to defaults.
        Path(path).write_bytes(pickle.dumps({
            "maps": self.maps,
            "n_noref": self.n_noref,
            "min_samples": self.min_samples,
            "isotonic_min": self.isotonic_min,
        }))

    def load(self, path: str | Path) -> "Calibrator":
        state = pickle.loads(Path(path).read_bytes())
        if not isinstance(state, dict) or "maps" not in state:
            self.maps = state           # a file written before state was versioned
            return self
        self.maps = state["maps"]
        self.n_noref = state.get("n_noref", 0)
        self.min_samples = state.get("min_samples", self.min_samples)
        self.isotonic_min = state.get("isotonic_min", self.isotonic_min)
        return self
