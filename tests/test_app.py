"""The Streamlit trace explorer, executed headlessly.

The app is one of the project's three deliverables and, until this file existed,
had never actually been run — it was written, shipped in the docs, and never
imported. Streamlit's own ``AppTest`` runs the script in-process and exposes the
resulting element tree, so we can assert on real rendered output rather than on
"the module imports". Skipped when Streamlit is not installed (it lives in the
``app`` extra, not the CPU core).
"""

from __future__ import annotations

import inspect
import json

import pytest

pytest.importorskip("streamlit", reason="streamlit is in the [app] extra")

import streamlit as st                     # noqa: E402
from streamlit.testing.v1 import AppTest   # noqa: E402

from tokentrace.app import streamlit_app   # noqa: E402
from tokentrace.core.serialize import inference_to_dict   # noqa: E402

APP_PATH = streamlit_app.__file__
_SCENARIO_BOX = "Canonical scenario"
#: What ``st.metric("Tier", report.tier.label)`` must read for each slider position.
_TIER_LABEL = {"black-box": "black", "grey-box": "grey", "white-box": "white"}


def _fresh(train: bool = False) -> AppTest:
    """Load the app with the engine in cold-start mode.

    Cold-start (rules only) keeps the test to a couple of seconds; the trained
    engine is exercised exhaustively by the benchmark tests. What is under test
    here is the *app*, not the engine.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    box = [c for c in at.checkbox if "Train" in c.label]
    assert box, "expected the 'Train engine' checkbox to exist"
    if box[0].value != train:
        box[0].set_value(train).run()
    return at


def _select_scenario(at: AppTest, name: str) -> AppTest:
    box = [sb for sb in at.selectbox if sb.label == _SCENARIO_BOX]
    assert box, f"expected a {_SCENARIO_BOX!r} selectbox"
    assert name in box[0].options, f"{name!r} not among {box[0].options}"
    return box[0].select(name).run()


def _paste(at: AppTest, inference) -> AppTest:
    """Drive the app through the paste-JSON path, as a user with their own trace would."""
    box = [t for t in at.text_area if "paste" in t.label.lower()]
    assert box, "expected the paste-JSON text area"
    return box[0].set_value(json.dumps(inference_to_dict(inference))).run()


def _click_fix(at: AppTest, action: str) -> AppTest:
    """Click the 'Try this fix' button for one intervention.

    Always driven from a freshly loaded app: Streamlit carries widget state across
    ``run()``, so clicking a second button on an app that already clicked a first one
    fires both and merges their verdicts.
    """
    fix = [b for b in at.button if action in b.label]
    assert fix, f"no 'Try this fix: {action}' button among {[b.label for b in at.button]}"
    return fix[0].click().run()


def _verdicts(at: AppTest) -> dict[str, list[str]]:
    """The post-fix verdict lines, split by the severity they were rendered at.

    Only lines reporting an intervention outcome are kept: the app also emits engine
    notes as ``st.info`` and an "Abstained" ``st.warning``, and folding those in is
    what made the previous version of this assertion unable to fail.
    """
    return {kind: [e.value for e in getattr(at, kind) if "After fix, answer" in e.value]
            for kind in ("success", "warning", "info")}


def _badges(at: AppTest) -> list[str]:
    """The ``✅ validated`` / ``❌ no effect`` / ``ℹ️ advisory`` line of each fix."""
    return [m.value for m in at.markdown if m.value.startswith("**Fix (")]


def test_app_loads_without_a_trace_selected():
    at = _fresh()
    assert not at.exception, [str(e.value) for e in at.exception]
    # Nothing chosen yet -> the app must prompt rather than render an empty report.
    assert any("scenario" in m.value.lower() or "paste" in m.value.lower() for m in at.info)


@pytest.mark.parametrize("scenario", [
    "grounded (healthy)", "retrieval failure", "context dilution",
    "prompt ambiguity", "reasoning failure",
])
def test_every_canonical_scenario_renders(scenario):
    at = _select_scenario(_fresh(), scenario)
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("Ranked root-cause diagnoses" in s.value for s in at.subheader)
    # The feature-vector table always renders, so a diagnosis view is never empty.
    assert at.dataframe, "expected at least the signal feature-vector table"


def test_failing_scenario_offers_a_fix_and_healthy_one_does_not():
    """The discriminating assertion: the app must not offer a repair for a trace
    that is already correct, and must offer one for a trace that is not."""
    broken = _select_scenario(_fresh(), "retrieval failure")
    healthy = _select_scenario(_fresh(), "grounded (healthy)")
    assert not broken.exception and not healthy.exception
    broken_fixes = [b.label for b in broken.button if "Try this fix" in b.label]
    healthy_fixes = [b.label for b in healthy.button if "Try this fix" in b.label]
    assert broken_fixes, "a retrieval failure must surface at least one actionable fix"
    assert not healthy_fixes, f"a healthy trace must not be 'fixed': {healthy_fixes}"


def test_a_working_fix_is_reported_as_a_success_and_nothing_else():
    """`add_gold_context` really does correct this trace, so the verdict must be green.

    Half of the pair below: a test that accepts "some outcome was rendered somewhere"
    passes just as happily when the success and warning branches are swapped, which is
    the one mistake that matters here — it tells a developer their fix worked when it
    did not.
    """
    at = _click_fix(_select_scenario(_fresh(), "retrieval failure"), "add_gold_context")
    assert not at.exception, [str(e.value) for e in at.exception]
    v = _verdicts(at)
    assert len(v["success"]) == 1 and "now correct" in v["success"][0], v
    assert v["warning"] == [] and v["info"] == [], v


def test_a_fix_that_does_not_correct_the_answer_is_reported_as_still_wrong():
    """The other half: on the reasoning-failure trace the *context dilution* fix is
    applied, the model is re-run, and the answer stays wrong. That must not read green.
    """
    at = _click_fix(_select_scenario(_fresh(), "reasoning failure"), "rerank_gold_first")
    assert not at.exception, [str(e.value) for e in at.exception]
    v = _verdicts(at)
    assert len(v["warning"]) == 1 and "still wrong" in v["warning"][0], v
    assert v["success"] == [] and v["info"] == [], v
    # ...and the badge above it agrees, rather than contradicting it on the same screen.
    assert any("❌ no effect" in b for b in _badges(at)), _badges(at)


def test_fix_outcome_is_not_reported_as_success_without_ground_truth():
    """Regression: the button previously rendered every post-fix answer in success
    green, including when there was no reference to judge it against."""
    from tokentrace.cli import _canonical

    inf = dict(_canonical())["retrieval failure"]
    inf.ground_truth = None
    at = _paste(_fresh(), inf)
    assert not at.exception, [str(e.value) for e in at.exception]
    if not [b for b in at.button if "Try this fix" in b.label]:
        pytest.skip("no fix offered for the reference-free variant")
    at = _click_fix(at, "Try this fix")
    assert not at.exception, [str(e.value) for e in at.exception]
    v = _verdicts(at)
    assert v["success"] == [], "outcome must not be green when it cannot be verified"
    assert len(v["info"]) == 1 and "unknown" in v["info"][0], v


def test_a_fix_on_an_already_correct_answer_is_not_reported_as_a_repair():
    """Right answer for the wrong reasons: the fix changes nothing, so nothing is green.

    ``Recommender._maybe_validate`` refuses to score an intervention when the original
    answer was already correct ("advisory, leave it unscored"), which is why the badge
    reads `ℹ️ advisory`. The verdict underneath judged only the NEW answer, so it
    printed "✅ now correct" in success green for a no-op — the two disagreeing about
    the same fix, one line apart.
    """
    from _helpers import parametric_fragile

    at = _paste(_fresh(), parametric_fragile())
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("ℹ️ advisory" in b for b in _badges(at)), _badges(at)
    at = _click_fix(at, "add_gold_context")
    assert not at.exception, [str(e.value) for e in at.exception]
    v = _verdicts(at)
    assert v["success"] == [], "a fix that repaired nothing must not be reported green"
    assert v["warning"] == [], "...nor as a failed fix: it was never scored"
    assert len(v["info"]) == 1, v
    assert "already correct" in v["info"][0] and "not scored" in v["info"][0], v


def test_the_tier_slider_actually_changes_the_analysis():
    """Rendering without an exception at each slider position proves nothing — an app
    that ignored the slider and analyzed everything at white-box would pass that. The
    tier has to reach the engine, so assert on what it changes: the tier the report
    came back at, and which signal families the feature vector could populate.
    """
    signatures = {}
    for tier, label in _TIER_LABEL.items():
        at = _fresh()
        sliders = [s for s in at.select_slider if "tier" in s.label.lower()]
        assert sliders, "expected the observability-tier slider"
        at = _select_scenario(sliders[0].set_value(tier).run(), "retrieval failure")
        assert not at.exception, (tier, [str(e.value) for e in at.exception])
        metrics = {m.label: m.value for m in at.metric}
        assert metrics.get("Tier") == label, (tier, metrics)
        sig = [m.value for m in at.markdown if "missingness signature" in m.value]
        assert len(sig) == 1, (tier, sig)
        signatures[tier] = sig[0]
    # Black-box cannot see inside the model, so the mechanistic family must be absent
    # there and present at white-box. (Grey and white agree on this corpus.)
    assert "mechanistic" not in signatures["black-box"], signatures
    assert "mechanistic" in signatures["white-box"], signatures


def test_fit_returns_a_full_width_kwarg_this_streamlit_actually_accepts():
    """`_fit` degrading to ``{}`` renders every table at its natural width — no
    exception, nothing in the element tree to observe (Streamlit keeps the width on
    the enclosing Element proto, which ``AppTest`` drops), so only its contract can be
    pinned. Two live versions and the semantic trap it was written for:
    """
    def modern(width: str = "content"):        # >=1.49: width takes "stretch"
        ...

    def legacy(use_container_width: bool = False):
        ...

    def pixels_only(width=None):               # width exists but is int|None pixels
        ...

    assert streamlit_app._fit(modern) == {"width": "stretch"}
    assert streamlit_app._fit(legacy) == {"use_container_width": True}
    # Passing "stretch" into the int field raises TypeError at the first table, so the
    # correct degrade is to pass nothing at all.
    assert streamlit_app._fit(pixels_only) == {}

    live = streamlit_app._fit(st.dataframe)
    assert live, "the installed Streamlit takes one of the two spellings; pick it"
    assert set(live) <= set(inspect.signature(st.dataframe).parameters), live
