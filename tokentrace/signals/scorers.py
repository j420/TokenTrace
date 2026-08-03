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
# Function words carry no evidential weight; counting them inflates bag-of-words
# support/containment for verbose answers.
_STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
         "was", "were", "are", "be", "been", "by", "with", "from", "as", "that",
         "this", "it", "its", "into", "about", "than", "then", "there"}
# Capitalized words that are NOT proper-noun antecedents (question/stop words that
# can appear capitalized at the start of a sentence).
_QUESTION_STOP = {"what", "when", "who", "where", "why", "how", "which", "whose",
                  "is", "was", "were", "are", "did", "does", "do", "the", "a", "an",
                  "in", "on", "at", "of", "to", "for"}


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
        # CONTENT words only: with stop words included, a verbose answer saturates
        # this score purely on "the/of/is" overlap, and `answer_supported_by_context`
        # is the gate both hallucination rules key on. Falls back to the raw token
        # set when an answer is nothing but stop words.
        h = _tokset(hypothesis) - _STOP or _tokset(hypothesis)
        if not h:
            return 0.0
        best = 0.0
        for t in texts:
            tt = _tokset(t)
            if not tt:
                continue
            # recall of hypothesis content words, maximised over chunks (a hypothesis
            # whose tokens are SCATTERED across several unrelated chunks must not
            # count as supported by any of them)
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
        # Case-preserving word tokens (need capitalization to spot proper nouns).
        words = re.findall(r"[A-Za-z0-9]+", question)
        lower = [w.lower() for w in words]
        if not lower:
            return 0.5
        content = [w for w in lower if w not in _PRONOUNS]

        pron_positions = [i for i, w in enumerate(lower) if w in _PRONOUNS]
        n_pron = len(pron_positions)

        def is_proper(i: int) -> bool:
            # A proper-noun antecedent: capitalized, not sentence-initial, and not a
            # capitalized question/stop word.
            return (i > 0 and len(words[i]) >= 3 and words[i][0].isupper()
                    and lower[i] not in _QUESTION_STOP)

        # An antecedent only counts if it PRECEDES the pronoun — a capitalized unit or
        # place that comes AFTER the pronoun (Celsius, Moon, Earth) is not one.
        has_antecedent = bool(pron_positions) and any(
            is_proper(i) for i in range(pron_positions[0])
        )

        score = 0.0
        # 1) an unresolved pronoun (no preceding antecedent) is strong ambiguity on
        #    its own — weighted to cross the gate regardless of question length.
        if n_pron and not has_antecedent:
            score += 0.6
        # 2) underspecified: very few content words
        if len(content) <= 4:
            score += 0.2
        # 3) no named entity anywhere AND no pronoun (bare underspecified query)
        if n_pron == 0 and not any(is_proper(i) for i in range(len(words))) and len(content) <= 4:
            score += 0.2
        # 4) vague temporal language on a 'when/how' question
        if any(w in ("when", "how") for w in lower) and any(w in _VAGUE_TIME for w in lower):
            score += 0.1
        return round(min(1.0, score), 4)

    # -- answer correctness vs references (token-F1, set-valued refs) -- #
    def match(self, answer: str, references: list[str]) -> float:
        """Correctness score in [0,1] against a SET of acceptable references.

        Symmetric token-F1 alone punishes a correct answer phrased as a full
        sentence ("The Eiffel Tower was completed in 1889" vs "1889" scores ~0.3 and
        would be recorded as *incorrect*), and `is_correct` gates every verification
        gate and metric downstream. So we also allow CONTAINMENT — the reference's
        content words all appearing in the answer — but only while the answer stays
        close to the reference's length, so a rambling answer cannot be scored
        correct merely for containing the gold tokens somewhere.
        """
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
            f1 = 2 * prec * rec / (prec + rec)
            rc = r - _STOP or r
            contained = len(a & rc) / len(rc)
            if contained >= 1.0 and len(a) <= 4 * len(r) + 5:
                f1 = max(f1, 1.0)
            best = max(best, f1)
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
