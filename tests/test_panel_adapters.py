"""Tests for the substrate ingest adapters (tau2-verified reference + APB process-quality).

These run against REAL on-disk data where available (the committed tau2 airline fixture; the APB
clone if present, else a hermetic hand-built APB record matching the verified shape), and pin the
load-bearing adapter properties (EXP-0094 + synthesis §2):
  - tau2-verified emits an outcome/binary GoldRecord whose value/basis/outcome come from the
    deterministic reward, with deterministic+human-confirmed provenance;
  - APB emits a process_quality/ordinal GoldRecord keyed on REAL span_ids (the GRAFT-4 join), with
    final_label==0 mapping to UNKNOWN (not failure), kappa_status forced raw-only, and the tau2
    slice flagged excluded_from_pooling;
  - both records pass the GoldStore firewall + ingest invariants (the join resolves to real spans).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtm_bench.apb_loader import load_apb_record
from mtm_bench.tau2_loader import load_tau2_results
from mtm_bench.panel import GoldStore, has_clean_pool
from mtm_bench.panel_adapters import apb_gold_from_record, tau2_verified_gold
from mtm_bench.schema import OutcomeStatus

_REPO = Path(__file__).resolve().parents[1]
_TAU2_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "airline_sample.json"
_APB_CLONE = (
    Path.home() / "scratch" / "agentprocessbench" / "data" / "AgentProcessBench" / "tau2.jsonl"
)


# ───────────────────────── tau2-verified reference adapter ─────────────────────────


def test_tau2_verified_gold_from_real_fixture() -> None:
    traces = load_tau2_results(_TAU2_FIXTURE, "airline")
    assert traces, "fixture should yield traces"
    recs = [tau2_verified_gold(t) for t in traces]

    for t, r in zip(traces, recs, strict=True):
        assert r.substrate == "tau2_verified"
        assert r.domain == "tau2_airline"
        assert r.label.axis == "outcome"
        # The outcome label value mirrors the deterministic reward on the trace.
        assert r.label.value == (t.outcome.status == OutcomeStatus.SUCCESS)
        assert r.outcome == t.outcome.status
        # Deterministic provenance with the human-confirmation anchor (verified subset).
        assert r.provenance.source == "deterministic"
        assert r.provenance.anchoring is not None
        assert r.provenance.anchoring.anchor_source == "human"
        # basis carried from the reward (airline fixture is DB/COMMUNICATE).
        assert r.label.basis  # non-empty

    # A SUCCESS fixture trace gives a clean pool (the depth leg supplies fire-on-clean).
    if any(t.outcome.status == OutcomeStatus.SUCCESS for t in traces):
        assert has_clean_pool(recs) is True


def test_tau2_verified_records_pass_the_firewall() -> None:
    traces = load_tau2_results(_TAU2_FIXTURE, "airline")
    store = GoldStore(strict=True)
    for t in traces:
        # The reward fields live on meta/outcome, not in the span stream — no leak.
        store.register(t, tau2_verified_gold(t))
    assert len(store.trace_ids()) == len(traces)


# ───────────────────────── APB process-quality adapter ─────────────────────────


def _hermetic_apb_record() -> dict:
    """A small airline APB record (OpenAI chat), matching the verified tau2.jsonl shape. step_labels
    are keyed by assistant-message index; final_label==0 exercises the UNKNOWN mapping."""
    return {
        "data_source": "tau2_airline",
        "total_index": 999,
        "final_label": 0,
        "step_labels": {"2": 1, "4": -1, "6": 0},
        "ground_truth": {"should_not_be_read": True},
        "messages": [
            {"role": "system", "content": "# Airline Agent Policy"},
            {"role": "user", "content": "Update my flight."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "get_user_details", "arguments": '{"user_id": "u_1"}'}}]},
            {"role": "tool", "name": "get_user_details", "content": "{\"ok\": true}"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c2", "type": "function",
                 "function": {"name": "book_reservation", "arguments": "{}"}}]},
            {"role": "tool", "name": "book_reservation", "content": "Error: bad"},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "###STOP###"},
        ],
    }


def _apb_records():
    """Yield (trace, record) for the hermetic record + (if present) the first real clone record."""
    rec = _hermetic_apb_record()
    yield load_apb_record(rec), rec
    if _APB_CLONE.exists():
        real = json.loads(_APB_CLONE.read_text().splitlines()[0])
        yield load_apb_record(real), real


def test_apb_gold_shape_and_join() -> None:
    trace, rec = next(_apb_records())  # hermetic
    g = apb_gold_from_record(trace, rec)
    assert g.substrate == "apb"
    assert g.label.axis == "process_quality"
    assert g.label.scale == [-1, 0, 1]
    # final_label==0 maps to UNKNOWN, never FAILURE (apb_leaderboard final==0-excluded semantics).
    assert g.outcome == OutcomeStatus.UNKNOWN
    # span_values keyed on REAL span ids, one per labeled assistant message.
    span_ids = {s.span_id for s in trace.spans}
    assert set(g.label.span_values).issubset(span_ids)
    assert len(g.label.span_values) == len(rec["step_labels"])
    # the -1 / 0 / +1 values survive the join.
    assert sorted(g.label.span_values.values()) == [-1, 0, 1]
    # reliability: raw-only, kappa absent (EXP-0094).
    assert g.provenance.source == "hybrid"
    assert g.provenance.reliability.kappa_status == "unreported_raw_only"
    assert g.provenance.reliability.cohen_kappa is None
    assert g.provenance.reliability.raw_agreement == pytest.approx(0.891)
    # tau2 slice is excluded from pooling (double-counts the depth leg).
    assert g.excluded_from_pooling is True


def test_apb_gold_passes_firewall_and_invariants_on_all_records() -> None:
    store = GoldStore(strict=True)
    n = 0
    for trace, rec in _apb_records():
        # The APB Trace carries no answer key in its span stream (load_apb_record keeps gold off the
        # trace), so registration (which runs find_leaks + assert_ingest_invariants) must succeed.
        store.register(trace, apb_gold_from_record(trace, rec))
        n += 1
    assert n >= 1


def test_apb_excluded_slice_has_no_clean_pool_contribution() -> None:
    # Even a SUCCESS-final APB tau2 record must not contribute a clean denominator (it double-counts
    # the depth leg) — has_clean_pool ignores excluded_from_pooling records.
    rec = _hermetic_apb_record()
    rec["final_label"] = 1  # SUCCESS final
    trace = load_apb_record(rec)
    g = apb_gold_from_record(trace, rec)
    assert g.outcome == OutcomeStatus.SUCCESS
    assert g.excluded_from_pooling is True
    assert has_clean_pool([g]) is False


def test_apb_gold_raises_on_unjoinable_label() -> None:
    rec = _hermetic_apb_record()
    rec["step_labels"]["99"] = -1  # a msg index with no message/span
    trace = load_apb_record(rec)
    with pytest.raises(ValueError, match="no span in the trace"):
        apb_gold_from_record(trace, rec)
