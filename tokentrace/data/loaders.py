"""Real dataset loaders (require the ``data`` extra: ``pip install 'tokentrace[data]'``).

These convert the proposal's datasets into TokenTrace's structures so the same
pipeline that runs on the bundled seed pool runs on real data:

* HotpotQA / Natural Questions / TruthfulQA -> :class:`Fact` / :class:`MultiHopFact`
  clean seeds for the injection harness (with the correctness gate ensuring the
  model answers correctly with gold context before any failure is injected).
* RAGTruth -> :class:`LabeledInference` with **REAL** provenance and human
  hallucination labels — reserved for headline evaluation.
* BEIR -> a corpus for the FAISS retriever (see :mod:`tokentrace.retrieval.rag`).

Everything is lazy: importing this module does not import ``datasets``. Field
mappings follow each dataset's published schema; adjust ``split``/field names if a
dataset version differs.
"""

from __future__ import annotations

from typing import Optional

from tokentrace.core.types import Chunk, FailureMode, Inference, LabeledInference, Provenance
from tokentrace.data.samples import Fact, MultiHopFact


def _load(path: str, name: Optional[str] = None, split: str = "validation"):
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - optional extra
        raise ImportError("Real loaders need the 'data' extra: pip install 'tokentrace[data]'") from e
    return load_dataset(path, name, split=split)


# --------------------------------------------------------------------------- #
def natural_questions_facts(limit: int = 200, split: str = "validation") -> list[Fact]:
    """Short-answer NQ items -> single-hop Fact seeds."""
    ds = _load("natural_questions", split=split)
    out: list[Fact] = []
    for row in ds:
        q = row["question"]["text"] if isinstance(row.get("question"), dict) else row.get("question")
        answer = _first_short_answer(row)
        if not q or not answer:
            continue
        gold = _nq_gold_sentence(row, answer)
        out.append(Fact(
            question=q if q.endswith("?") else q + "?", answer=answer,
            gold_fact=gold or f"The answer is {answer}.", wrong_answer="a different entity",
            entity=answer, ambiguous_question=_stripped(q, answer),
            parametric=False, distractors=[],
        ))
        if len(out) >= limit:
            break
    return out


def hotpotqa_multihop(limit: int = 200, split: str = "validation") -> list[MultiHopFact]:
    """HotpotQA comparison/bridge items -> MultiHopFact seeds (supporting sentences
    are the gold hops)."""
    ds = _load("hotpot_qa", "distractor", split=split)
    out: list[MultiHopFact] = []
    for row in ds:
        facts = _hotpot_supporting_sentences(row)
        if len(facts) < 2:
            continue
        out.append(MultiHopFact(
            question=row["question"], answer=row["answer"], facts=facts,
            wrong_answer="the other entity",
            distractors=[s for para in row["context"]["sentences"] for s in para][:4],
        ))
        if len(out) >= limit:
            break
    return out


def truthfulqa_facts(limit: int = 100) -> list[Fact]:
    """TruthfulQA -> Fact seeds prone to parametric misconceptions (good for the
    hallucination recipe)."""
    ds = _load("truthful_qa", "generation", split="validation")
    out: list[Fact] = []
    for row in ds:
        out.append(Fact(
            question=row["question"], answer=row["best_answer"],
            gold_fact=row["best_answer"], wrong_answer=(row.get("incorrect_answers") or ["a myth"])[0],
            entity="", ambiguous_question=row["question"], parametric=True,
            distractors=row.get("incorrect_answers", [])[:3],
        ))
        if len(out) >= limit:
            break
    return out


def ragtruth_labeled(limit: int = 500, split: str = "test") -> list[LabeledInference]:
    """RAGTruth -> LabeledInference with REAL provenance. Responses with annotated
    hallucination spans are labeled HALLUCINATION; clean responses get no label.

    RAGTruth ships as (source_info, response, annotations). We map the prompt +
    reference passages to context chunks and the model output to the answer.
    """
    ds = _load("ragtruth", split=split)   # adjust to your local RAGTruth HF path/name
    out: list[LabeledInference] = []
    for row in ds:
        passages = row.get("reference") or row.get("passages") or []
        if isinstance(passages, str):
            passages = [passages]
        chunks = [Chunk(text=p, retriever_score=1.0, source_id=f"p{i}") for i, p in enumerate(passages)]
        inf = Inference(
            prompt=row.get("prompt") or row.get("question", ""),
            generated_answer=row.get("response", ""), retrieved_context=chunks or None,
            ground_truth=None, question=row.get("question"),
            meta={"task_type": row.get("task_type", "QA"), "source": "ragtruth"},
        )
        has_halluc = bool(row.get("labels") or row.get("hallucination_spans"))
        labels = [FailureMode.HALLUCINATION] if has_halluc else []
        out.append(LabeledInference(
            inference=inf, labels=labels, provenance=Provenance.REAL,
            injection_recipe=None, verification={"source": "ragtruth_human"},
        ))
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
def _first_short_answer(row) -> Optional[str]:
    ann = row.get("annotations") or {}
    sa = ann.get("short_answers") or []
    for block in sa:
        text = block.get("text") if isinstance(block, dict) else None
        if text:
            return text[0] if isinstance(text, list) else text
    return None


def _nq_gold_sentence(row, answer: str) -> Optional[str]:
    doc = row.get("document", {})
    tokens = doc.get("tokens", {}).get("token") if isinstance(doc.get("tokens"), dict) else None
    if tokens and answer in " ".join(tokens):
        return f"... {answer} ..."
    return None


def _hotpot_supporting_sentences(row) -> list[str]:
    titles = row["context"]["title"]
    sents = row["context"]["sentences"]
    sup_titles = row["supporting_facts"]["title"]
    sup_ids = row["supporting_facts"]["sent_id"]
    by_title = dict(zip(titles, sents))
    out = []
    for t, sid in zip(sup_titles, sup_ids):
        para = by_title.get(t)
        if para and 0 <= sid < len(para):
            out.append(para[sid].strip())
    return out


def _stripped(question: str, entity: str) -> str:
    """Best-effort ambiguity variant: replace the entity with a pronoun."""
    if entity and entity in question:
        return question.replace(entity, "it", 1)
    return question
