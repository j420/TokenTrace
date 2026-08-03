"""Model-backed scorers (higher fidelity than the heuristics).

Drop-in replacement for :class:`HeuristicScorers` that upgrades the signals where
a model genuinely helps:

* ``relevance``  — cosine similarity of sentence-transformer embeddings.
* ``support``    — NLI entailment (does a context sentence entail the hypothesis?).
* ``semantic_entropy`` / ``self_consistency`` — cluster resampled answers by
  *bidirectional* NLI entailment (Farquhar 2024), instead of token overlap.

``ambiguity`` and ``match`` keep the heuristic implementations (token-F1 is the
standard QA correctness proxy; a real ambiguity detector would need its own trained
model — noted as future work), so this class composes with :class:`HeuristicScorers`
rather than reimplementing everything.

All heavy imports are lazy; needs the ``retrieval`` extra
(``pip install 'tokentrace[retrieval]'``). Everything runs on CPU.
"""

from __future__ import annotations

import math
from typing import Optional

from tokentrace.signals.scorers import HeuristicScorers, norm


class ModelScorers(HeuristicScorers):
    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model: str = "cross-encoder/nli-deberta-v3-small",
        entail_threshold: float = 0.5,
        device: str = "cpu",
    ):
        try:
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError as e:  # pragma: no cover - optional extra
            raise ImportError(
                "ModelScorers needs the 'retrieval' extra: pip install 'tokentrace[retrieval]'"
            ) from e
        self._embedder = SentenceTransformer(embed_model, device=device)
        self._nli = CrossEncoder(nli_model, device=device)
        self.entail_threshold = entail_threshold
        # Resolve the entailment column from the model's own label map instead of
        # hard-coding 1: index 1 is the deberta-v3 NLI order
        # ([contradiction, entailment, neutral]) but e.g. the MNLI-standard order
        # puts entailment at 0/2, which would silently score the wrong class.
        # Still assignable by callers for exotic checkpoints.
        self._num_labels, self._entail_idx = self._resolve_labels()

    def _resolve_labels(self) -> tuple[Optional[int], int]:
        cfg = getattr(self._nli, "config", None)
        if cfg is None:
            cfg = getattr(getattr(self._nli, "model", None), "config", None)
        num: Optional[int] = None
        for holder in (self._nli, cfg):
            v = getattr(holder, "num_labels", None) if holder is not None else None
            if isinstance(v, int) and v > 0:
                num = v
                break
        label2id = getattr(cfg, "label2id", None) if cfg is not None else None
        if not isinstance(label2id, dict) or not label2id:
            id2label = getattr(cfg, "id2label", None) if cfg is not None else None
            label2id = ({v: k for k, v in id2label.items()}
                        if isinstance(id2label, dict) else {})
        idx: Optional[int] = None
        for name, i in label2id.items():
            if str(name).strip().lower().startswith("entail"):
                try:
                    idx = int(i)
                except (TypeError, ValueError):
                    idx = None
                break
        if idx is None:
            idx = 0 if num == 1 else 1
        if num is not None and idx >= num:
            idx = max(0, num - 1)
        return num, idx

    # ------------------------------------------------------------------ #
    def relevance(self, query: str, text: str) -> float:
        if not query or not text:
            return 0.0
        emb = self._embedder.encode([query, text], normalize_embeddings=True,
                                    convert_to_numpy=True)
        return round(max(0.0, float(emb[0] @ emb[1])), 4)

    def support(self, hypothesis: str, texts: list[str]) -> float:
        """Max entailment probability of ``hypothesis`` by any premise in ``texts``."""
        if not hypothesis or not texts:
            return 0.0
        pairs = [(t, hypothesis) for t in texts if t]
        if not pairs:
            return 0.0
        probs = self._entail_probs(pairs)
        return round(float(max(probs)), 4)

    def semantic_entropy(self, samples: list[str]) -> float:
        if not samples:
            return 0.0
        clusters = self._nli_cluster(samples)
        total = len(samples)
        h = -sum((n / total) * math.log(n / total) for n in clusters)
        h = h / math.log(len(samples)) if len(samples) > 1 else 0.0
        return round(max(0.0, h), 4)

    def self_consistency(self, samples: list[str], answer: str) -> float:
        if not samples:
            return 0.0
        same = sum(1 for s in samples if self._equiv_nli(s, answer))
        return round(same / len(samples), 4)

    # ------------------------------------------------------------------ #
    def _entail_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Entailment probability per pair, guaranteed to be in [0, 1].

        Two fixes over the previous version:
        * It decided whether to softmax by testing ``allclose(sum, 1.0)``. Raw
          3-class logits sum to ~1.0 about 0.9% of the time (measured on N(0,2.5)
          single-pair calls) and were then returned verbatim — e.g.
          [3.2, -1.4, -0.8] made ``support()`` return -1.4 where the true
          entailment probability is 0.0098. A negative "probability" then feeds a
          LightGBM residual head and an isotonic calibrator that were both fit on
          [0,1] data. The softmax is now applied deterministically.
        * A 1-logit cross-encoder returns a flat array of length ``len(pairs)``;
          ``np.atleast_2d`` reshaped that to (1, n_pairs) — i.e. it softmaxed
          ACROSS PAIRS and made ``_equiv_nli`` raise IndexError on its two-pair
          call. The output is now reshaped using the model's declared
          ``num_labels``.
        """
        import numpy as np

        if not pairs:
            return []
        # CrossEncoder.predict kwargs shifted across sentence-transformers 3/4/5
        # (apply_softmax -> activation_fn); fall back to a bare call.
        try:
            raw = self._nli.predict(pairs, convert_to_numpy=True, apply_softmax=False)
        except TypeError:  # pragma: no cover - version-dependent
            raw = self._nli.predict(pairs)
        scores = np.asarray(raw, dtype=float)

        n = len(pairs)
        nl = self._num_labels
        if scores.ndim == 1:
            if nl and nl > 1 and scores.size == n * nl:
                scores = scores.reshape(n, nl)
            elif scores.size == n:
                scores = scores.reshape(n, 1)       # single-logit cross-encoder
            else:  # pragma: no cover - defensive
                scores = scores.reshape(n, -1)
        elif scores.ndim > 2:  # pragma: no cover - defensive
            scores = scores.reshape(n, -1)

        if scores.shape[1] == 1:
            # num_labels == 1: a binary entailment/relevance head. sentence-
            # transformers applies Sigmoid by default in that case, so the value
            # is normally already a probability; squash only when it is not, then
            # clip so the [0,1] scorer contract holds either way.
            col = scores[:, 0]
            if float(col.min()) < 0.0 or float(col.max()) > 1.0:
                col = 1.0 / (1.0 + np.exp(-col))
            return np.clip(col, 0.0, 1.0).tolist()

        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        idx = min(max(int(self._entail_idx), 0), probs.shape[1] - 1)
        return np.clip(probs[:, idx], 0.0, 1.0).tolist()

    def _equiv_nli(self, a: str, b: str) -> bool:
        if " ".join(norm(a)) == " ".join(norm(b)):
            return True
        p = self._entail_probs([(a, b), (b, a)])
        return p[0] >= self.entail_threshold and p[1] >= self.entail_threshold

    def _nli_cluster(self, samples: list[str]) -> list[int]:
        reps: list[str] = []
        counts: list[int] = []
        for s in samples:
            placed = False
            for i, rep in enumerate(reps):
                if self._equiv_nli(s, rep):
                    counts[i] += 1
                    placed = True
                    break
            if not placed:
                reps.append(s)
                counts.append(1)
        return counts
