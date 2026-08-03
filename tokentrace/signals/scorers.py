"""Pluggable scorers used by the signal extractors.

The default :class:`HeuristicScorers` are pure-Python and dependency-free: they
run on CPU with no downloads, giving real (if approximate) signals from the text.
They implement the same interface a model-backed scorer would, so
:class:`~tokentrace.signals.model_scorers.ModelScorers` (NLI + sentence embeddings)
can be dropped in for higher fidelity without touching the extractors.

Every scorer returns a value in [0, 1] (or a count), never None — "missingness"
is decided by the extractor/tier, not by the scorer.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

_WORD = re.compile(r"[a-z0-9]+")
_PRONOUNS = {"it", "its", "they", "them", "their", "this", "that", "these",
             "those", "he", "she", "him", "her", "his", "one", "there"}
_VAGUE_TIME = {"recently", "soon", "later", "now", "then", "today", "yesterday"}


def norm(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def _tokset(text: str) -> set[str]:
    return set(norm(text))


class Scorers(Protocol):
    def support(self, hypothesis: str, texts: list[str]) -> float: ...
    def relevance(self, query: str, text: str) -> float: ...
    def ambiguity(self, question: str) -> float: ...
    def match(self, answer: str, references: list[str]) -> float: ...
    def semantic_entropy(self, samples: list[str]) -> float: ...


class HeuristicScorers:
    """Dependency-free CPU scorers. Good enough to drive the whole pipeline;
    swap in :class:`ModelScorers` for NLI/embedding fidelity."""

    # -- entailment proxy: is `hypothesis` supported by any of `texts`? -- #
    def support(self, hypothesis: str, texts: list[str]) -> float:
        h = _tokset(hypothesis)
        if not h:
            return 0.0
        best = 0.0
        for t in texts:
            tt = _tokset(t)
            if not tt:
                continue
            # recall of hypothesis content words, with a light idf-free weighting
            overlap = len(h & tt) / len(h)
            best = max(best, overlap)
        return round(best, 4)

    # -- lexical relevance of a chunk to the query -- #
    def relevance(self, query: str, text: str) -> float:
        q, t = _tokset(query), _tokset(text)
        if not q or not t:
            return 0.0
        return round(len(q & t) / len(q | t), 4)  # Jaccard

    # -- prompt ambiguity heuristic in [0,1] -- #
    def ambiguity(self, question: str) -> float:
        toks = norm(question)
        if not toks:
            return 0.5
        score = 0.0
        content = [w for w in toks if w not in _PRONOUNS]
        # 1) unresolved pronouns with no proper-noun antecedent
        n_pron = sum(1 for w in toks if w in _PRONOUNS)
        has_proper = bool(re.search(r"\b[A-Z][a-z]{2,}", question[1:]))  # capitalized mid-sentence
        if n_pron and not has_proper:
            score += 0.4
        # 2) underspecified: very few content words
        if len(content) <= 4:
            score += 0.3
        # 3) missing named entity entirely
        if not has_proper:
            score += 0.2
        # 4) vague temporal language on a 'when' question
        if any(w in ("when", "how") for w in toks) and any(w in _VAGUE_TIME for w in toks):
            score += 0.1
        return round(min(1.0, score), 4)

    # -- answer correctness vs references (token-F1, set-valued refs) -- #
    def match(self, answer: str, references: list[str]) -> float:
        a = _tokset(answer)
        if not a or not references:
            return 0.0
        best = 0.0
        for ref in references:
            r = _tokset(ref)
            if not r:
                continue
            inter = len(a & r)
            if inter == 0:
                continue
            prec, rec = inter / len(a), inter / len(r)
            best = max(best, 2 * prec * rec / (prec + rec))
        return round(best, 4)

    # -- semantic entropy over resampled answers (cluster + entropy) -- #
    def semantic_entropy(self, samples: list[str]) -> float:
        if not samples:
            return 0.0
        clusters = self._cluster(samples)
        total = len(samples)
        h = -sum((n / total) * math.log(n / total) for n in clusters.values())
        # normalize by log(k) so it is comparable across different sample counts
        h = h / math.log(len(samples)) if len(samples) > 1 else 0.0
        return round(max(0.0, h), 4)  # max() clears -0.0

    def self_consistency(self, samples: list[str], answer: str) -> float:
        if not samples:
            return 0.0
        key = " ".join(norm(answer))
        same = sum(1 for s in samples if self._equiv(" ".join(norm(s)), key))
        return round(same / len(samples), 4)

    # -- clustering by (proxy) semantic equivalence -- #
    def _cluster(self, samples: list[str]) -> Counter:
        reps: list[str] = []
        labels: list[int] = []
        for s in samples:
            ns = " ".join(norm(s))
            placed = False
            for i, rep in enumerate(reps):
                if self._equiv(ns, rep):
                    labels.append(i)
                    placed = True
                    break
            if not placed:
                reps.append(ns)
                labels.append(len(reps) - 1)
        return Counter(labels)

    @staticmethod
    def _equiv(a: str, b: str) -> bool:
        """Proxy for bidirectional NLI entailment: high token-Jaccard equivalence."""
        if a == b:
            return True
        sa, sb = set(a.split()), set(b.split())
        if not sa or not sb:
            return a == b
        return len(sa & sb) / len(sa | sb) >= 0.6


DEFAULT_SCORERS = HeuristicScorers()
