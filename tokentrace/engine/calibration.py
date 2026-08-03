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


class Calibrator:
    def __init__(self, min_samples: int = 40):
        self.min_samples = min_samples
        self.maps: dict[tuple[str, str], object] = {}

    def fit(self, records: list[dict]) -> "Calibrator":
        """records: dicts with keys ``mode`` (FailureMode), ``signature`` (str),
        ``z`` (float raw logit), ``label`` (0/1)."""
        from sklearn.isotonic import IsotonicRegression

        groups: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
        for r in records:
            mode = r["mode"].value if isinstance(r["mode"], FailureMode) else str(r["mode"])
            groups[(mode, r["signature"])].append((r["z"], int(r["label"])))
            groups[(mode, "__global__")].append((r["z"], int(r["label"])))

        for key, pairs in groups.items():
            if len(pairs) < self.min_samples:
                continue
            ys = [y for _, y in pairs]
            if len(set(ys)) < 2:
                continue
            zs = [z for z, _ in pairs]
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(zs, ys)
            self.maps[key] = ir
        return self

    def transform(self, mode: FailureMode, signature: str, z: float) -> float:
        mkey = mode.value if isinstance(mode, FailureMode) else str(mode)
        for key in ((mkey, signature), (mkey, "__global__")):
            m = self.maps.get(key)
            if m is not None:
                return float(m.predict([z])[0])
        return sigmoid(z)

    @property
    def is_fitted(self) -> bool:
        return bool(self.maps)

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self.maps))

    def load(self, path: str | Path) -> "Calibrator":
        self.maps = pickle.loads(Path(path).read_bytes())
        return self
