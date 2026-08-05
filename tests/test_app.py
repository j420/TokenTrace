"""The Streamlit trace explorer, executed headlessly.

The app is one of the project's three deliverables and, until this file existed,
had never actually been run — it was written, shipped in the docs, and never
imported. Streamlit's own ``AppTest`` runs the script in-process and exposes the
resulting element tree, so we can assert on real rendered output rather than on
"the module imports". Skipped when Streamlit is not installed (it lives in the
``app`` extra, not the CPU core).
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit", reason="streamlit is in the [app] extra")

from streamlit.testing.v1 import AppTest   # noqa: E402

from tokentrace.app import streamlit_app   # noqa: E402

APP_PATH = streamlit_app.__file__
_SCENARIO_BOX = "Canonical scenario"


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


def test_try_this_fix_runs_the_intervention_and_reports_an_outcome():
    at = _select_scenario(_fresh(), "retrieval failure")
    fix = [b for b in at.button if "Try this fix" in b.label]
    assert fix, "no fix button to click"
    at = fix[0].click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    rendered = ([m.value for m in at.success] + [m.value for m in at.warning]
                + [m.value for m in at.info] + [m.value for m in at.markdown])
    assert any("After fix, answer" in t for t in rendered), rendered


def test_fix_outcome_is_not_reported_as_success_without_ground_truth():
    """Regression: the button previously rendered every post-fix answer in success
    green, including when there was no reference to judge it against."""
    at = _fresh()
    import json

    from tokentrace.cli import _canonical
    from tokentrace.core.serialize import inference_to_dict

    payload = dict(inference_to_dict(dict(_canonical())["retrieval failure"]))
    payload["ground_truth"] = None
    box = [t for t in at.text_area if "paste" in t.label.lower()]
    assert box, "expected the paste-JSON text area"
    at = box[0].set_value(json.dumps(payload)).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    fix = [b for b in at.button if "Try this fix" in b.label]
    if not fix:                     # no actionable fix on this trace -> nothing to assert
        pytest.skip("no fix offered for the reference-free variant")
    at = fix[0].click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.success, "outcome must not be green when it cannot be verified"
    assert any("unknown" in m.value for m in at.info), [m.value for m in at.info]


def test_all_three_tiers_render():
    for tier in ("black-box", "grey-box", "white-box"):
        at = _fresh()
        sliders = [s for s in at.select_slider if "tier" in s.label.lower()]
        assert sliders, "expected the observability-tier slider"
        at = sliders[0].set_value(tier).run()
        at = _select_scenario(at, "retrieval failure")
        assert not at.exception, (tier, [str(e.value) for e in at.exception])
