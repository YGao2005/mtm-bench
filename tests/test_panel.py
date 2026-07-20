"""Tests for the provenance-first unified gold sidecar (cross-domain meta-eval PANEL).

These pin the load-bearing properties of the data layer (EXP-0094 + the synthesis memo):
  - the reliability type FORCES kappa honesty (raw agreement cannot masquerade as kappa);
  - process_quality span_values must lie inside the declared ordinal scale;
  - the loader keeps gold in a sidecar (a Trace carries no gold) — the firewall as structure;
  - the firewall auditor CATCHES a residual answer-key surface left in the trace (GRAFT 3);
  - ingest invariants REJECT a per-step/attribution join that points off-trace (GRAFT 4);
  - has_clean_pool is False exactly when no in-pool record is SUCCESS (the AEB n_clean==0 case),
    which is what makes leaderboard.py print fire-on-clean = "n/a" with no new scoring code.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mtm_bench.leaderboard import _CallableEntry, score_leaderboard
from mtm_bench.panel import (
    AttributionLabel,
    Firewall,
    FirewallViolation,
    GoldRecord,
    GoldStore,
    HumanReliability,
    OutcomeLabel,
    ProcessQualityLabel,
    Provenance,
    assert_ingest_invariants,
    find_leaks,
    has_clean_pool,
    to_gold_items,
)
from mtm_bench.schema import (
    Outcome,
    OutcomeStatus,
    Span,
    SpanKind,
    TaskSpec,
    ToolResult,
    Trace,
)


def _trace(tid: str, *, status: OutcomeStatus = OutcomeStatus.SUCCESS, spans=None) -> Trace:
    return Trace(
        trace_id=tid,
        agent_id="a",
        task_type="tau2_airline",
        spec=TaskSpec(goal="t", allowed_tools=[]),
        spans=spans
        or [Span(span_id="s0", index=0, kind=SpanKind.USER_MSG, content="hi")],
        outcome=Outcome(status=status, final_response=None),
        label=None,
        meta={"domain": "tau2_airline"},
    )


def _outcome_gold(tid: str, *, value: bool, status: OutcomeStatus) -> GoldRecord:
    return GoldRecord(
        trace_id=tid,
        substrate="tau2_verified",
        domain="tau2_airline",
        outcome=status,
        label=OutcomeLabel(value=value, basis=["state_hash"]),
        provenance=Provenance(
            source="deterministic",
            reliability={"kind": "deterministic", "grader_id": "tau2_state_hash"},
        ),
        firewall=Firewall(answer_key_fields=["state_hash"]),
    )


# ───────────────────────── reliability type forces kappa honesty ─────────────────────────


def test_kappa_status_reported_requires_kappa() -> None:
    with pytest.raises(ValidationError):
        HumanReliability(n_annotators=2, kappa_status="reported", cohen_kappa=None)
    # The valid form: a reported kappa carries a number.
    ok = HumanReliability(n_annotators=2, kappa_status="reported", cohen_kappa=0.55)
    assert ok.cohen_kappa == 0.55


def test_raw_only_cannot_carry_a_kappa() -> None:
    # APB's case: raw agreement 89.1%, kappa NOT re-derivable (EXP-0094). The type forbids
    # presenting a kappa under unreported_raw_only — raw can never masquerade as chance-corrected.
    with pytest.raises(ValidationError):
        HumanReliability(
            n_annotators=3,
            kappa_status="unreported_raw_only",
            cohen_kappa=0.7,
            raw_agreement=0.891,
        )
    ok = HumanReliability(
        n_annotators=3, kappa_status="unreported_raw_only", raw_agreement=0.891
    )
    assert ok.cohen_kappa is None and ok.raw_agreement == 0.891


def test_process_quality_values_must_lie_in_scale() -> None:
    with pytest.raises(ValidationError):
        ProcessQualityLabel(scale=[-1, 0, 1], span_values={"s1": 2})
    ok = ProcessQualityLabel(scale=[-1, 0, 1], span_values={"s1": -1, "s3": 1})
    assert ok.span_values["s1"] == -1


# ───────────────────────── the loader contract / firewall as code ─────────────────────────


def test_gold_store_separates_trace_from_gold() -> None:
    store = GoldStore()
    t = _trace("t1")
    g = _outcome_gold("t1", value=True, status=OutcomeStatus.SUCCESS)
    store.register(t, g)
    # A contestant gets the trace and has no path to the gold from it.
    loaded = store.load_trace("t1")
    assert loaded.label is None  # the Trace carries no gold
    assert not hasattr(loaded, "gold")
    # The scorer (only) can fetch gold by id.
    assert store.load_gold("t1").label.value is True


def test_register_rejects_trace_id_mismatch() -> None:
    store = GoldStore()
    other = _outcome_gold("OTHER", value=True, status=OutcomeStatus.SUCCESS)
    with pytest.raises(ValueError, match="trace_id mismatch"):
        store.register(_trace("t1"), other)


def test_firewall_auditor_catches_residual_answer_key_surface() -> None:
    # A trace whose tool_result.value still carries an answer-key field NAME the gold forbids:
    # find_leaks must report it, and strict GoldStore.register must raise.
    leaky_span = Span(
        span_id="s1",
        index=1,
        kind=SpanKind.TOOL_RESULT,
        tool_result=ToolResult(name="grade", value={"output_columns": "REDACT_ME"}),
    )
    t = _trace("leak", spans=[Span(span_id="s0", index=0, kind=SpanKind.USER_MSG), leaky_span])
    g = GoldRecord(
        trace_id="leak",
        substrate="sop_bench",
        domain="order_fulfillment",
        outcome=OutcomeStatus.SUCCESS,
        label=OutcomeLabel(value=True),
        provenance=Provenance(
            source="deterministic",
            reliability={"kind": "deterministic", "grader_id": "sopbench_v1"},
        ),
        firewall=Firewall(answer_key_fields=["output_columns"]),
    )
    leaks = find_leaks(t, g)
    assert leaks == ["s1.output_columns"]
    with pytest.raises(FirewallViolation):
        GoldStore(strict=True).register(t, g)
    # Non-strict store registers but the leak is still discoverable by the auditor.
    GoldStore(strict=False).register(t, g)


def test_clean_trace_has_no_leak() -> None:
    t = _trace("clean")
    g = _outcome_gold("clean", value=True, status=OutcomeStatus.SUCCESS)
    assert find_leaks(t, g) == []


# ───────────────────────── ingest invariants (GRAFT 4: span_id join) ─────────────────────────


def test_attribution_join_must_resolve_to_a_real_span() -> None:
    t = _trace("attr", spans=[Span(span_id="s0", index=0, kind=SpanKind.AGENT_MSG)])
    bad = GoldRecord(
        trace_id="attr",
        substrate="aeb",
        domain="alfworld",
        outcome=OutcomeStatus.FAILURE,
        label=AttributionLabel(
            critical_span_id="NOT_A_SPAN", category="inefficient_plan", vocab_id="aeb_cognitive_v1"
        ),
        provenance=Provenance(
            source="human",
            reliability=HumanReliability(
                n_annotators=10, kappa_status="reported", cohen_kappa=0.55, kappa_subset_n=30
            ),
        ),
        firewall=Firewall(answer_key_fields=["critical_failure_step"]),
    )
    with pytest.raises(ValueError, match="not a span of the trace"):
        assert_ingest_invariants(t, bad)


def test_process_quality_join_rejects_offtrace_keys() -> None:
    t = _trace("pq", spans=[Span(span_id="s0", index=0, kind=SpanKind.AGENT_MSG)])
    bad = GoldRecord(
        trace_id="pq",
        substrate="apb",
        domain="tau2",
        outcome=OutcomeStatus.UNKNOWN,
        label=ProcessQualityLabel(scale=[-1, 0, 1], span_values={"s99": -1}),
        provenance=Provenance(
            source="hybrid",
            reliability=HumanReliability(
                n_annotators=3, kappa_status="unreported_raw_only", raw_agreement=0.891
            ),
        ),
        firewall=Firewall(answer_key_fields=["step_labels", "final_label", "ground_truth"]),
    )
    with pytest.raises(ValueError, match="do not resolve"):
        assert_ingest_invariants(t, bad)


# ───────────────────────── has_clean_pool composes with leaderboard.py ─────────────────────────


def test_has_clean_pool_true_when_a_success_record_exists() -> None:
    recs = [
        _outcome_gold("a", value=True, status=OutcomeStatus.SUCCESS),
        _outcome_gold("b", value=False, status=OutcomeStatus.FAILURE),
    ]
    assert has_clean_pool(recs) is True


def test_has_clean_pool_false_for_aeb_failure_only() -> None:
    # AEB: 200/200 failures -> no clean pool -> fire-on-clean must be N/A downstream.
    recs = [
        GoldRecord(
            trace_id=f"f{i}",
            substrate="aeb",
            domain="alfworld",
            outcome=OutcomeStatus.FAILURE,
            label=AttributionLabel(
                critical_span_id="s0", category="step_limit", vocab_id="aeb_cognitive_v1"
            ),
            provenance=Provenance(
                source="human",
                reliability=HumanReliability(
                    n_annotators=10, kappa_status="reported", cohen_kappa=0.55
                ),
            ),
            firewall=Firewall(answer_key_fields=["critical_failure_step"]),
        )
        for i in range(3)
    ]
    assert has_clean_pool(recs) is False


def test_excluded_from_pooling_record_does_not_count_as_clean() -> None:
    # APB tau2 slice: a SUCCESS record flagged excluded_from_pooling must NOT supply a clean denom
    # (it double-counts the depth leg).
    rec = _outcome_gold("tau2_overlap", value=True, status=OutcomeStatus.SUCCESS)
    rec = rec.model_copy(update={"excluded_from_pooling": True})
    assert has_clean_pool([rec]) is False


def test_eval_fault_requires_a_reason() -> None:
    with pytest.raises(ValidationError):
        GoldRecord(
            trace_id="ef",
            substrate="sop_bench",
            domain="order_fulfillment",
            outcome=OutcomeStatus.SUCCESS,
            label=OutcomeLabel(value=True),
            provenance=Provenance(
                source="deterministic",
                reliability={"kind": "deterministic", "grader_id": "sopbench_v1"},
            ),
            firewall=Firewall(answer_key_fields=["output_columns"]),
            eval_fault=True,  # missing eval_fault_reason
        )


# ───────── projection → GoldItem reuses the SHIPPED scorer (synthesis §4 step 4) ─────────


def _aeb_failure_gold(tid: str) -> GoldRecord:
    return GoldRecord(
        trace_id=tid,
        substrate="aeb",
        domain="alfworld",
        outcome=OutcomeStatus.FAILURE,
        label=AttributionLabel(
            critical_span_id="s0", category="inefficient_plan", vocab_id="aeb_cognitive_v1"
        ),
        provenance=Provenance(
            source="human",
            reliability=HumanReliability(
                n_annotators=10, kappa_status="reported", cohen_kappa=0.55
            ),
        ),
        firewall=Firewall(answer_key_fields=["critical_failure_step"]),
    )


def test_to_gold_items_drops_unknown_and_excluded() -> None:
    recs = [
        _outcome_gold("ok", value=True, status=OutcomeStatus.SUCCESS),
        _outcome_gold("bad", value=False, status=OutcomeStatus.FAILURE),
        # APB final==0 → UNKNOWN: not scoreable on the binary cell, dropped.
        GoldRecord(
            trace_id="neutral",
            substrate="apb",
            domain="tau2",
            outcome=OutcomeStatus.UNKNOWN,
            label=ProcessQualityLabel(scale=[-1, 0, 1], span_values={}),
            provenance=Provenance(
                source="hybrid",
                reliability=HumanReliability(
                    n_annotators=3, kappa_status="unreported_raw_only", raw_agreement=0.891
                ),
            ),
            firewall=Firewall(answer_key_fields=["step_labels"]),
        ),
    ]
    items = to_gold_items(recs)
    by_id = {i.trace_id: i for i in items}
    assert set(by_id) == {"ok", "bad"}  # 'neutral' (UNKNOWN) dropped
    assert by_id["bad"].corrupt_success is True  # FAILURE = the positive class
    assert by_id["ok"].corrupt_success is False  # SUCCESS = clean
    assert by_id["ok"].cell == "tau2_verified|tau2_airline"


def test_aeb_failure_only_projects_to_na_fire_on_clean() -> None:
    # The load-bearing composition: AEB has no clean pool, so the SHIPPED scorer reports its
    # fire-on-clean as n/a (n_clean==0 -> nan), with NO new scoring code — the leaderboard.py path.
    # tau2 (with a clean trace) is scored in its OWN cell, never pooled with AEB.
    recs = [
        _aeb_failure_gold("aeb1"),
        _aeb_failure_gold("aeb2"),
        _outcome_gold("tau_clean", value=True, status=OutcomeStatus.SUCCESS),
        _outcome_gold("tau_fail", value=False, status=OutcomeStatus.FAILURE),
    ]
    assert has_clean_pool([r for r in recs if r.substrate == "aeb"]) is False
    items = to_gold_items(recs)
    traces = [_trace(i.trace_id, status=OutcomeStatus.SUCCESS) for i in items]
    # A detector that flags everything: recall 1.0 on failures; on AEB there are no clean traces.
    entry = _CallableEntry("flag_all", lambda t, s: True)
    rep = score_leaderboard(items, traces, [entry])

    aeb_cell = rep.scores["flag_all"]["aeb|alfworld"]["instance_label"]
    assert aeb_cell.n_clean == 0  # AEB failure-only → empty clean denominator
    assert aeb_cell.clean_fire_rate != aeb_cell.clean_fire_rate  # nan (n/a in render)
    assert aeb_cell.n_corrupt == 2 and aeb_cell.recall == 1.0

    # tau2 cell is separate and DOES have a clean trace (so fire-on-clean is a real number).
    tau_cell = rep.scores["flag_all"]["tau2_verified|tau2_airline"]["instance_label"]
    assert tau_cell.n_clean == 1 and tau_cell.clean_fire_rate == 1.0
    # The two cells are NOT pooled — distinct keys in the report.
    assert "aeb|alfworld" in rep.cells and "tau2_verified|tau2_airline" in rep.cells

    # The render shows "n/a" for the AEB fire-on-clean column.
    out = rep.render()
    assert "n/a" in out
