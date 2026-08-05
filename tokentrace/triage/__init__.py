"""Fleet triage — aggregate many diagnosed traces into a ranked fix list.

    from tokentrace import TokenTrace
    from tokentrace.triage import render_text, triage

    tt = TokenTrace.default()
    report = triage(load_my_traces(), tt)     # -> TriageReport
    print(render_text(report))
    json.dumps(report.to_dict())              # strict JSON, no NaN/Infinity

See :mod:`tokentrace.triage.aggregate` for the honesty contract stating which
numbers are measured and which are descriptive counts.
"""

from __future__ import annotations

from tokentrace.triage.aggregate import (
    NO_SIGNAL,
    TriageCluster,
    TriageFailure,
    TriageReport,
    render_text,
    triage,
)

__all__ = [
    "triage",
    "render_text",
    "TriageReport",
    "TriageCluster",
    "TriageFailure",
    "NO_SIGNAL",
]
