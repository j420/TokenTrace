"""FAISS-backed RAG retrieval (require the ``retrieval`` extra).

Builds a dense index over a corpus and returns :class:`Chunk` lists for an
:class:`Inference`, so the retrieval-signal family runs on a real retriever. All
CPU (``faiss-cpu`` + a small sentence-transformer). Lazy imports keep the core
dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tokentrace.core.types import Chunk, Inference


@dataclass
class RagIndex:
    docs: list[str]
    source_ids: list[str] = field(default_factory=list)
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    _index: object = None
    _embedder: object = None

    def build(self) -> "RagIndex":
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - optional extra
            raise ImportError("RAG needs the 'retrieval' extra: pip install 'tokentrace[retrieval]'") from e

        if not self.source_ids:
            self.source_ids = [f"doc{i}" for i in range(len(self.docs))]
        self._embedder = SentenceTransformer(self.model_name)
        emb = self._embedder.encode(self.docs, normalize_embeddings=True,
                                    convert_to_numpy=True).astype("float32")
        self._index = faiss.IndexFlatIP(emb.shape[1])  # cosine via normalized inner product
        self._index.add(emb)
        return self

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        if self._index is None:
            self.build()
        q = self._embedder.encode([query], normalize_embeddings=True,
                                  convert_to_numpy=True).astype("float32")
        scores, idxs = self._index.search(q, k)
        chunks: list[Chunk] = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            chunks.append(Chunk(text=self.docs[i], retriever_score=float(score),
                                source_id=self.source_ids[i]))
        return chunks

    def make_inference(
        self, question: str, generated_answer: str, k: int = 5,
        ground_truth: Optional[list[str]] = None, gold_source_ids: Optional[set[str]] = None,
    ) -> Inference:
        """Retrieve context and assemble an Inference. If ``gold_source_ids`` is
        known (evaluation), mark those chunks as gold so retrieval-vs-dilution can be
        scored directly."""
        chunks = self.retrieve(question, k)
        if gold_source_ids:
            for c in chunks:
                c.gold = c.source_id in gold_source_ids
        prompt = f"{question}\n" + "\n".join(c.text for c in chunks)
        return Inference(prompt=prompt, generated_answer=generated_answer,
                         retrieved_context=chunks, ground_truth=ground_truth, question=question)
