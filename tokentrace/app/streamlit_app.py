"""TokenTrace — interactive trace explorer.

    pip install 'tokentrace[app]'
    streamlit run tokentrace/app/streamlit_app.py

Loads a trace (a canonical scenario, a pasted JSON, or an uploaded file), runs the
diagnosis, and shows: ranked root-cause diagnoses, the additive evidence ledger,
the causal chain, and recommendations with a "try this fix" button that runs the
simulated intervention and reports whether the answer corrects. Everything is
CPU-only; with the mock backend it needs no downloads.
"""

from __future__ import annotations

import json

import streamlit as st

from tokentrace.api import TokenTrace
from tokentrace.core.serialize import inference_from_dict, inference_to_dict
from tokentrace.core.types import Tier
from tokentrace.recommend.recommender import INTERVENTIONS

_TIER = {"black-box": Tier.BLACK, "grey-box": Tier.GREY, "white-box": Tier.WHITE}
_MODE_COLOR = {
    "retrieval_failure": "#e45756", "context_dilution": "#f58518",
    "hallucination": "#b279a2", "reasoning_failure": "#4c78a8",
    "prompt_ambiguity": "#54a24b",
}


def _supported(fn, **kwargs) -> dict:
    """Keep only the kwargs this installed Streamlit actually accepts.

    Streamlit's layout API is a moving target across the versions a user might
    have: ``use_container_width`` was deprecated with a removal date that has now
    passed and is superseded by ``width="stretch"``, and ``bar_chart(horizontal=)``
    only exists on newer releases. Pinning a narrow version floor to dodge that
    would be worse than adapting — the app is a demo, not a library, and it should
    render on whatever the user already has. So we feature-detect per call instead
    of encoding version numbers we would have to keep correct.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):       # builtins / C-implemented: pass nothing extra
        return {}
    return {k: v for k, v in kwargs.items() if k in params}


def _fit(fn) -> dict:
    """Width kwargs for a full-width element, whichever spelling this version uses.

    Detecting the NAME ``width`` is not enough, and assuming it was cost this app its
    whole supported range below Streamlit 1.49: ``width`` existed long before
    ``"stretch"`` was legal, typed ``int | None`` (pixels). Passing the string into
    that int protobuf field raises ``TypeError: 'str' object cannot be interpreted as
    an integer`` at the first table — the app crashed rather than degraded on every
    version ``pyproject.toml`` allows. So detect the SEMANTICS: only the modern
    parameter carries a string default.
    """
    import inspect

    try:
        param = inspect.signature(fn).parameters.get("width")
    except (TypeError, ValueError):
        param = None
    if param is not None and isinstance(param.default, str):
        return {"width": "stretch"}
    return _supported(fn, use_container_width=True)


@st.cache_resource(show_spinner="Training engine on the offline synthetic corpus…")
def get_engine(backend: str, model_name: str, train: bool) -> TokenTrace:
    return TokenTrace.default(backend=backend, model_name=model_name, tier=Tier.WHITE, train=train)


def _canonical_traces() -> dict:
    from tokentrace.cli import _canonical

    return {name: inf for name, inf in _canonical()}


def main() -> None:
    st.set_page_config(page_title="TokenTrace", page_icon="🔎", layout="wide")
    st.title("🔎 TokenTrace — evidence-based root-cause analysis for LLMs")
    st.caption("Why did the model get this wrong — and what should I change? "
               "Ranked diagnoses · auditable evidence · validated fixes. CPU-only.")

    with st.sidebar:
        st.header("Model")
        backend = st.selectbox("Backend", ["mock", "gguf", "hf"], index=0,
                               help="mock = no downloads. gguf/hf need the extras installed.")
        model_name = st.text_input("Model", "mock-4b")
        train = st.checkbox("Train engine (else cold-start rules)", value=True)
        tier_label = st.select_slider("Observability tier",
                                      list(_TIER.keys()), value="white-box")
        st.header("Trace")
        canon = _canonical_traces()
        choice = st.selectbox("Canonical scenario", ["— choose —", *canon.keys()])
        uploaded = st.file_uploader("…or upload an inference JSON", type="json")
        pasted = st.text_area("…or paste inference JSON", height=140)

    inf = None
    if uploaded is not None:
        inf = inference_from_dict(json.load(uploaded))
    elif pasted.strip():
        inf = inference_from_dict(json.loads(pasted))
    elif choice in canon:
        inf = canon[choice]

    if inf is None:
        st.info("Pick a canonical scenario, upload, or paste an inference JSON to begin.")
        st.code(json.dumps(inference_to_dict(next(iter(canon.values()))), indent=2)[:900] + " …",
                language="json")
        return

    tt = get_engine(backend, model_name, train)
    tier = _TIER[tier_label]
    report = tt.analyze(inf, tier=tier)

    # ---- header: Q / A ---- #
    left, right = st.columns([3, 1])
    with left:
        st.subheader("Inference")
        st.markdown(f"**Question:** {inf.query}")
        st.markdown(f"**Answer:** `{inf.generated_answer}`")
        if inf.has_ground_truth:
            st.markdown(f"**Ground truth:** {inf.ground_truth}")
        if inf.retrieved_context:
            with st.expander(f"Retrieved context ({len(inf.retrieved_context)} chunks)"):
                for c in inf.retrieved_context:
                    tag = "🟢 gold" if c.gold else "⚪"
                    st.markdown(f"{tag} `{c.source_id}` — {c.text[:200]}")
    with right:
        st.metric("Diagnostic confidence", f"{report.diagnostic_confidence:.2f}")
        st.metric("Tier", report.tier.label)
        if report.abstained:
            st.warning("Abstained")

    # ---- ranked diagnoses ---- #
    st.subheader("Ranked root-cause diagnoses")
    st.bar_chart({d.mode.pretty: d.probability for d in report.diagnoses},
                 **_supported(st.bar_chart, horizontal=True))

    for d in report.diagnoses:
        if d.probability < 0.15:
            continue
        role = f" · {d.role.value.replace('_', ' ')}" if d.role else ""
        parents = ("  ⟵ " + ", ".join(p.pretty for p in d.causal_parents)) if d.causal_parents else ""
        with st.expander(f"**{d.mode.pretty}** — {d.probability:.0%}{role}{parents}",
                         expanded=(d.rank == 0 and not report.abstained)):
            if d.evidence:
                st.markdown("**Evidence ledger** (log-odds contributions)")
                st.dataframe(
                    [{"signal": e.signal, "family": e.family.value,
                      "Δ log-odds": round(e.contribution_logodds, 3),
                      "source": e.source, "detail": e.rendered} for e in d.evidence],
                    hide_index=True, **_fit(st.dataframe),
                )
            for r in d.recommendations:
                _render_recommendation(tt, inf, r)

    if report.conformal_set:
        st.markdown("**Top-k (conformal) set:** "
                    + ", ".join(m.pretty for m in report.conformal_set))
    for n in report.notes:
        st.info(n)

    with st.expander("Signal feature vector (missingness-aware)"):
        fv = tt.features(inf, tier=tier)
        st.markdown(f"missingness signature: `{fv.missingness_signature()}`")
        st.dataframe([{"feature": k, "value": v} for k, v in sorted(fv.values.items())],
                     hide_index=True, **_fit(st.dataframe))


def _render_recommendation(tt: TokenTrace, inf, rec) -> None:
    badge = {True: "✅ validated", False: "❌ no effect", None: "ℹ️ advisory"}[rec.validated]
    st.markdown(f"**Fix ({rec.targets_mode.pretty}):** {rec.description}  \n_{badge}_")
    if rec.action in INTERVENTIONS and st.button(f"▶ Try this fix: `{rec.action}`",
                                                 key=f"fix_{rec.targets_mode.value}_{rec.action}"):
        modified = INTERVENTIONS[rec.action](inf)
        if modified is None:
            st.write("No structural fix available — recommend the model abstain.")
            return
        new_answer = tt.model.bound_to(modified).generate(modified.prompt).text
        # Three outcomes, not two. Without ground truth we cannot say whether the fix
        # worked, and rendering "unknown" in success green (as this did) overstates the
        # result on exactly the traces a production user brings — where there is no
        # reference to check against.
        rec_er = tt.engine.recommender
        if rec_er is None or not inf.has_ground_truth:
            st.info(f"After fix, answer → `{new_answer}` · outcome unknown (no ground truth)")
        elif rec_er._correct(new_answer, inf.ground_truth):
            st.success(f"After fix, answer → `{new_answer}` · ✅ now correct")
        else:
            st.warning(f"After fix, answer → `{new_answer}` · ❌ still wrong")


if __name__ == "__main__":
    main()
