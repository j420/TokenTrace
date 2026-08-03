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
        # NLI label index for "entailment" — deberta-v3 nli order is
        # [contradiction, entailment, neutral]; override if your model differs.
        self._entail_idx = 1

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
        import numpy as np

        logits = self._nli.predict(pairs, convert_to_numpy=True, apply_softmax=False)
        logits = np.atleast_2d(logits)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        soft = e / e.sum(axis=1, keepdims=True)
        return soft[:, self._entail_idx].tolist()

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
