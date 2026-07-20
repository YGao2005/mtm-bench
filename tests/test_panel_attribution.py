"""Tests for the attribution axis scorer (panel_attribution) — failure localization.

Load-bearing assertions (EXP-0092 + synthesis §2/§3):
  - localization accuracy = predicted critical span == gold critical_span_id, per cell;
  - unfalsifiable rows (AEB 14% empty) are dropped from the denominator, counted separately, never a
    silent miss;
  - an abstain (None) on a falsifiable record counts as a miss (cannot inflate accuracy);
  - fire-on-clean is N/A on every AEB cell (failure-only) — nan, never a number.
"""

from __future__ import annotations

import math

from mtm_bench.panel import (
    AttributionLabel,
    Firewall,
    GoldRecord,
    HumanReliability,
    Provenance,
)
from mtm_bench.panel_attribution import (
    AttributionScore,
    LocalizerEntry,
    render_attribution,
    score_attribution,
)
from mtm_bench.schema import Outcome, OutcomeStatus, Span, SpanKind, TaskSpec, Trace


def _attr_record(tid: str, critical: str, *, falsifiable: bool = True) -> GoldRecord:
    return GoldRecord(
        trace_id=tid,
        substrate="aeb",
        domain="alfworld",
        outcome=OutcomeStatus.FAILURE,  # AEB is failure-only
        label=AttributionLabel(
            critical_span_id=critical,
            category="inefficient_plan",
            vocab_id="aeb_cognitive_v1",
            falsifiable=falsifiable,
        ),
        provenance=Provenance(
            source="human",
            reliability=HumanReliability(
                n_annotators=10, kappa_status="reported", cohen_kappa=0.55, kappa_subset_n=30
            ),
        ),
        firewall=Firewall(answer_key_fields=["critical_failure_step"]),
    )


def _failed_trace(tid: str, span_ids: list[str]) -> Trace:
    spans = [Span(span_id=s, index=i, kind=SpanKind.AGENT_MSG) for i, s in enumerate(span_ids)]
    return Trace(
        trace_id=tid, agent_id="a", task_type="alfworld",
        spec=TaskSpec(goal="t", allowed_tools=[]), spans=spans,
        outcome=Outcome(status=OutcomeStatus.FAILURE), label=None,
    )


def test_localization_accuracy() -> None:
    recs = [_attr_record("t1", "s3"), _attr_record("t2", "s1")]
    traces = {
        "t1": _failed_trace("t1", ["s0", "s1", "s2", "s3"]),
        "t2": _failed_trace("t2", ["s0", "s1"]),
    }
    # localizer always points at s3: hits t1, misses t2.
    entry = LocalizerEntry("always_s3", lambda t: "s3")
    out = score_attribution(recs, traces, [entry])
    sc = out["always_s3"]["aeb|alfworld"]
    assert sc.n_falsifiable == 2 and sc.localized_hits == 1
    assert sc.localization_acc == 0.5


def test_unfalsifiable_rows_dropped_from_denominator() -> None:
    recs = [
        _attr_record("ok", "s1", falsifiable=True),
        _attr_record("empty", "s1", falsifiable=False),  # 14% AEB empty-label case
    ]
    traces = {
        "ok": _failed_trace("ok", ["s0", "s1"]),
        "empty": _failed_trace("empty", ["s0", "s1"]),
    }
    entry = LocalizerEntry("g", lambda t: "s1")
    sc = score_attribution(recs, traces, [entry])["g"]["aeb|alfworld"]
    assert sc.n_falsifiable == 1  # only the falsifiable row enters the denominator
    assert sc.n_unfalsifiable == 1
    assert sc.localized_hits == 1 and sc.localization_acc == 1.0


def test_abstain_counts_as_miss() -> None:
    recs = [_attr_record("t1", "s2")]
    traces = {"t1": _failed_trace("t1", ["s0", "s1", "s2"])}
    entry = LocalizerEntry("abstainer", lambda t: None)
    sc = score_attribution(recs, traces, [entry])["abstainer"]["aeb|alfworld"]
    assert sc.n_falsifiable == 1 and sc.n_abstained == 1 and sc.localized_hits == 0
    assert sc.localization_acc == 0.0  # abstain cannot inflate accuracy


def test_fire_on_clean_is_structurally_na() -> None:
    sc = AttributionScore(n_falsifiable=5, localized_hits=3)
    assert sc.n_clean == 0
    assert math.isnan(sc.fire_on_clean)  # failure-only → never a number


def test_render_marks_fire_on_clean_na() -> None:
    recs = [_attr_record("t1", "s1")]
    traces = {"t1": _failed_trace("t1", ["s0", "s1"])}
    out = score_attribution(recs, traces, [LocalizerEntry("g", lambda t: "s1")])
    rendered = render_attribution(out)
    assert "n/a" in rendered  # fire-on-clean column is n/a
    assert "localization" in rendered.lower()
