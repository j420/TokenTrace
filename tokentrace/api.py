"""High-level facade tying the pieces together.

    from tokentrace import TokenTrace
    tt = TokenTrace.default()             # mock model, engine trained on synthetic
    report = tt.analyze(inference)        # -> DiagnosisReport

Used by the CLI and the Streamlit app. Keeps the wiring (model + pipeline +
engine + recommender) in one place.
"""

from __future__ import annotations

from typing import Optional

from tokentrace.core.types import DiagnosisReport, Inference, Tier
from tokentrace.engine.diagnosis import DiagnosisEngine
from tokentrace.models.base import ModelHandle
from tokentrace.models.registry import load_model
from tokentrace.recommend.recommender import Recommender
from tokentrace.signals.registry import SignalPipeline


class TokenTrace:
    def __init__(
        self,
        model: ModelHandle,
        engine: Optional[DiagnosisEngine] = None,
        pipeline: Optional[SignalPipeline] = None,
    ):
        self.model = model
        self.pipeline = pipeline or SignalPipeline()
        self.engine = engine or DiagnosisEngine(recommender=Recommender(model=model))

    # ------------------------------------------------------------------ #
    @classmethod
    def default(
        cls,
        backend: str = "mock",
        model_name: str = "mock-4b",
        tier: Tier = Tier.WHITE,
        train: bool = True,
        seeds: tuple[int, ...] = (0, 1, 2, 3),
    ) -> "TokenTrace":
        """Build a ready-to-use instance. With ``train=True`` (default) the engine
        is trained + calibrated on the offline synthetic corpus; otherwise it runs
        cold-start (interpretable rules only)."""
        model = load_model(model_name, backend=backend, tier=tier)
        pipeline = SignalPipeline()
        if not train:
            engine = DiagnosisEngine(recommender=Recommender(model=model))
            return cls(model, engine, pipeline)

        # Train on the synthetic corpus (offline, no downloads), with the SAME
        # family-dropout tier schedule as train_and_evaluate / run_ablations /
        # run_noref. This is the parity that makes the published tables evidence
        # about THIS engine: without the schedule, default() trained at the
        # handle's own tier only, so demo/analyze/triage and the Streamlit app
        # were running an engine the benchmarks never measured. (`with_tier` can
        # only down-cap, so on a handle opened below WHITE the schedule
        # degenerates to the handle's tier — the same behaviour as before.)
        from tokentrace.data.synthetic import build_dataset
        from tokentrace.eval.benchmark import (
            family_dropout_schedule,
            split_dataset,
            train_engine,
        )

        dataset = build_dataset(model, pipeline, seeds=seeds)
        train, cal, _ = split_dataset(dataset)
        engine = train_engine(train, cal, model, pipeline,
                              train_tiers=family_dropout_schedule(len(train)))
        return cls(model, engine, pipeline)

    # ------------------------------------------------------------------ #
    def analyze(self, inference: Inference, tier: Optional[Tier] = None) -> DiagnosisReport:
        """Diagnose one inference. If ``inference.generated_answer`` is empty, the
        model is run to produce it first (so you can diagnose a fresh query)."""
        model = self.model if tier is None else self.model.with_tier(tier)
        if not inference.generated_answer:
            inference.generated_answer = model.bound_to(inference).generate(inference.prompt).text
        fv = self.pipeline.run(inference, model)
        return self.engine.diagnose(inference, fv, model.tier)

    def features(self, inference: Inference, tier: Optional[Tier] = None):
        model = self.model if tier is None else self.model.with_tier(tier)
        return self.pipeline.run(inference, model)
