"""Tests for the process-quality axis scorer (panel_scoring) — the per-step two numbers.

The load-bearing assertions:
  - the two numbers are computed correctly on the harmful (-1) class (recall + fire-on-helpful);
  - a span the grader does not label counts as "+1 / no flag" (cannot inflate recall by abstaining);
  - excluded_from_pooling records (APB tau2 slice) are dropped by default, never pooled;
  - CROSS-VALIDATION: the panel's (recall_hits, n_harmful, clean_fires) equal apb_leaderboard's
    -1-class confusion (cls_tp[-1], cls_tp[-1]+cls_fn[-1], and the +1→-1 false alarm) for the SAME
    prediction — proving this is a re-view of the EXP-0030-validated scorer, not a parallel copy.
"""

from __future__ import annotations

from mtm_bench.apb_loader import load_apb_record
from mtm_bench.apb_leaderboard import APBRecord, StepVerifierEntry, _score_one
from mtm_bench.panel import (
    Anchoring,
    Firewall,
    GoldRecord,
    HumanReliability,
    ProcessQualityLabel,
    Provenance,
)
from mtm_bench.panel_adapters import apb_gold_from_record
from mtm_bench.panel_scoring import StepGraderEntry, score_pq_entries
from mtm_bench.schema import Outcome, OutcomeStatus, Span, SpanKind, TaskSpec, Trace


def _pq_record(tid: str, span_values: dict[str, int], *, excluded: bool = False) -> GoldRecord:
    return GoldRecord(
        trace_id=tid,
        substrate="apb",
        domain="apb_test",
        outcome=OutcomeStatus.UNKNOWN,
        label=ProcessQualityLabel(scale=[-1, 0, 1], span_values=span_values),
        provenance=Provenance(
            source="hybrid",
            reliability=HumanReliability(
                n_annotators=3, kappa_status="unreported_raw_only", raw_agreement=0.891
            ),
            anchoring=Anchoring(anchor_source="llm_judge", strength="collapsed_ondemand"),
        ),
        excluded_from_pooling=excluded,
        eval_fault=False,
        firewall=Firewall(answer_key_fields=["step_labels"]),
    )


def _trace_with_spans(tid: str, span_ids: list[str]) -> Trace:
    spans = [Span(span_id=s, index=i, kind=SpanKind.AGENT_MSG) for i, s in enumerate(span_ids)]
    return Trace(
        trace_id=tid, agent_id="a", task_type="apb_test",
        spec=TaskSpec(goal="t", allowed_tools=[]), spans=spans,
        outcome=Outcome(status=OutcomeStatus.UNKNOWN), label=None,
    )


def test_two_numbers_on_harmful_class() -> None:
    # gold: s1=-1 (harmful), s2=+1 (helpful), s3=-1 (harmful), s4=0 (neutral)
    rec = _pq_record("t1", {"s1": -1, "s2": 1, "s3": -1, "s4": 0})
    trace = _trace_with_spans("t1", ["s1", "s2", "s3", "s4"])
    # grader flags s1 (correct -1) and s2 (false alarm on a helpful step); misses s3.
    grader = StepGraderEntry("g", lambda t: {"s1": -1, "s2": -1})
    out = score_pq_entries([rec], {"t1": trace}, [grader])
    sc = out["g"]["apb|apb_test"]
    assert sc.n_harmful == 2 and sc.recall_hits == 1 and sc.recall == 0.5
    assert sc.n_helpful == 1 and sc.clean_fires == 1 and sc.fire_on_clean == 1.0
    assert sc.n_neutral == 1
    assert sc.n_trajectories == 1


def test_unlabeled_span_counts_as_no_flag() -> None:
    rec = _pq_record("t1", {"s1": -1, "s2": -1})
    trace = _trace_with_spans("t1", ["s1", "s2"])
    # grader labels nothing → both harmful steps are missed (recall 0), not silently caught.
    out = score_pq_entries([rec], {"t1": trace}, [StepGraderEntry("silent", lambda t: {})])
    sc = out["silent"]["apb|apb_test"]
    assert sc.n_harmful == 2 and sc.recall_hits == 0 and sc.recall == 0.0


def test_excluded_from_pooling_dropped_by_default() -> None:
    rec = _pq_record("tau2slice", {"s1": -1}, excluded=True)
    trace = _trace_with_spans("tau2slice", ["s1"])
    g = StepGraderEntry("g", lambda t: {"s1": -1})
    assert score_pq_entries([rec], {"tau2slice": trace}, [g])["g"] == {}
    # include_excluded scores it in isolation.
    incl = score_pq_entries([rec], {"tau2slice": trace}, [g], include_excluded=True)
    assert incl["g"]["apb|apb_test"].recall == 1.0


def test_cross_validates_against_apb_leaderboard_minus1_confusion() -> None:
    """The panel's harmful-class numbers must EQUAL apb_leaderboard's -1-class confusion for the
    same prediction — proving the panel scorer re-views the validated scorer, not re-derives it."""
    apb_rec = {
        "data_source": "tau2_airline",
        "total_index": 12345,
        "final_label": -1,
        "step_labels": {"2": -1, "4": 1, "6": -1, "8": 0},
        "ground_truth": {"x": 1},
        "messages": [
            {"role": "system", "content": "p"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a2"},
            {"role": "tool", "name": "t", "content": "ok"},
            {"role": "assistant", "content": "a4"},
            {"role": "tool", "name": "t", "content": "ok"},
            {"role": "assistant", "content": "a6"},
            {"role": "tool", "name": "t", "content": "ok"},
            {"role": "assistant", "content": "a8"},
        ],
    }
    trace = load_apb_record(apb_rec)
    # Build the panel gold (tau2 slice excluded; use include_excluded to compare directly).
    gold = apb_gold_from_record(trace, apb_rec)
    gold_spans = gold.label.span_values  # {span_id: gold value}, keyed on real msg-index spans

    # A prediction in BOTH views: catch msg2's -1, false-alarm msg4 (+1→-1), miss msg6's -1.
    # Map gold msg-index keys → the representative span_id the panel used.
    span_for_msg = {}
    msgidx = trace.meta["apb_msg_index"]
    from collections import defaultdict
    inv = defaultdict(list)
    for sid, mi in msgidx.items():
        inv[mi].append(sid)
    span_order = {s.span_id: s.index for s in trace.spans}
    for mi, sids in inv.items():
        span_for_msg[mi] = sorted(sids, key=lambda s: span_order[s])[0]

    pred_steps = {"2": -1, "4": -1, "6": 1, "8": 0}  # apb-style {msg_index: label}
    pred_spans = {span_for_msg[int(k)]: v for k, v in pred_steps.items()}

    # Panel scorer (include_excluded since tau2 slice is excluded by default).
    panel = score_pq_entries(
        [gold], {trace.trace_id: trace},
        [StepGraderEntry("g", lambda t: pred_spans)], include_excluded=True,
    )["g"]["apb|tau2_airline"]

    # apb_leaderboard scorer over the SAME reference + prediction.
    ref = {"k": APBRecord(key="k", dataset="tau2", data_source="tau2_airline",
                          step_labels={"2": -1, "4": 1, "6": -1, "8": 0}, final_label=-1)}
    entry = StepVerifierEntry("g", lambda r: (pred_steps, -1))
    axis = _score_one(ref, entry)

    # -1-class confusion equality.
    assert panel.recall_hits == axis.cls_tp[-1]                 # caught -1 steps
    assert panel.n_harmful == axis.cls_tp[-1] + axis.cls_fn[-1]  # all gold -1 steps
    # fire-on-helpful = a gold +1 step predicted -1 = part of cls_fp[-1].
    assert panel.clean_fires == 1
    assert gold_spans  # sanity: the join produced span-keyed gold
