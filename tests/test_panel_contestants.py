"""Tests for the reused-prediction judge contestants (APB's 20 shipped LLM verifiers).

Load-bearing assertions:
  - a shipped LLM verifier's predicted step_labels (msg-index keyed) replay onto the SAME span_ids
    the gold uses (the apb_gold inversion), so process-quality scoring is a real head-to-head;
  - a missing/failed prediction is "no flag" (cannot inflate recall);
  - the outcome predictor flags FAILURE iff predicted final_label == -1;
  - the localizer's critical step = first predicted -1 msg → its representative span_id;
  - (if the APB clone is present) the judges load and score against gold without key-join errors.
"""

from __future__ import annotations

from pathlib import Path

from mtm_bench.apb_loader import load_apb_record
from mtm_bench.panel_adapters import apb_gold_from_record
from mtm_bench.panel_contestants import (
    _msg_index_to_span,
    apb_judge_localizers,
    apb_judge_outcome_predictors,
    apb_judge_step_graders,
    tau2_cached_outcome_predictors,
)
from mtm_bench.panel_scoring import score_pq_entries
from mtm_bench.schema import Outcome, OutcomeStatus, Span, SpanKind, TaskSpec, Trace

_APB_CLONE = Path.home() / "scratch" / "agentprocessbench"


def _apb_record() -> dict:
    return {
        "data_source": "tau2_airline",
        "total_index": 760,
        "final_label": -1,
        "step_labels": {"2": 1, "4": -1, "6": 1},
        "ground_truth": {"x": 1},
        "messages": [
            {"role": "system", "content": "p"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a2"},
            {"role": "tool", "name": "t", "content": "ok"},
            {"role": "assistant", "content": "a4"},
            {"role": "tool", "name": "t", "content": "ok"},
            {"role": "assistant", "content": "a6"},
            {"role": "user", "content": "###STOP###"},
        ],
    }


def test_msg_index_to_span_matches_gold_span_keys() -> None:
    trace = load_apb_record(_apb_record())
    gold = apb_gold_from_record(trace, _apb_record())
    m2s = _msg_index_to_span(trace)
    # Every gold span_values key is the representative span of some labeled message.
    assert set(gold.label.span_values).issubset(set(m2s.values()))


def test_missing_prediction_is_no_flag(tmp_path: Path) -> None:
    # Build a tiny fake APB clone with ONE judge that has NO prediction for our trace.
    res = tmp_path / "eval" / "results" / "FakeJudge"
    res.mkdir(parents=True)
    # prediction for a DIFFERENT total_index (770) → our trace (760) gets no flag.
    (res / "tau2__blind_fake.jsonl").write_text(
        '{"data_source":"tau2_airline","query_index":4,"sample_index":0,'
        '"step_labels":{"2":-1},"final_label":-1,"status":"done"}\n'
    )
    trace = load_apb_record(_apb_record())  # total_index 760
    graders = apb_judge_step_graders(tmp_path)
    assert len(graders) == 1
    # ti for (qi=4,si=0) = 750+4*5+0 = 770 ≠ 760 → no prediction → empty grade.
    assert graders[0].grader(trace) == {}


def test_outcome_predictor_flags_failure_on_minus1_final(tmp_path: Path) -> None:
    res = tmp_path / "eval" / "results" / "J"
    res.mkdir(parents=True)
    # (qi=2, si=0) → ti 760, matching our trace; final_label -1 → should flag failure.
    (res / "tau2__blind_j.jsonl").write_text(
        '{"data_source":"tau2_airline","query_index":2,"sample_index":0,'
        '"step_labels":{"4":-1},"final_label":-1,"status":"done"}\n'
    )
    trace = load_apb_record(_apb_record())  # ti 760
    preds = apb_judge_outcome_predictors(tmp_path)
    assert "judge:blind_j" in preds
    assert preds["judge:blind_j"](trace) is True


def test_localizer_picks_first_predicted_minus1(tmp_path: Path) -> None:
    res = tmp_path / "eval" / "results" / "J"
    res.mkdir(parents=True)
    (res / "tau2__blind_j.jsonl").write_text(
        '{"data_source":"tau2_airline","query_index":2,"sample_index":0,'
        '"step_labels":{"4":-1,"6":-1},"final_label":-1,"status":"done"}\n'
    )
    trace = load_apb_record(_apb_record())
    loc = apb_judge_localizers(tmp_path)[0]
    m2s = _msg_index_to_span(trace)
    # first predicted -1 is msg 4 → its representative span.
    assert loc.localize(trace) == m2s[4]


def test_join_uses_total_index_block_not_data_source() -> None:
    """Regression lock: predictions key by total_index BLOCK, not the trace data_source string.

    A bfcl trace has data_source 'bfcl_multi_turn_base' but its prediction file is keyed 'bfcl' — a
    data_source join silently misses (R=0∧F=0 fabricated 'no flag'). The total_index block
    (500-749 = bfcl) is the correct shared key."""
    from mtm_bench.panel_contestants import _dataset_for_total_index

    assert _dataset_for_total_index(0) == "hotpotqa"      # 0-249
    assert _dataset_for_total_index(250) == "gaia_dev"    # 250-499
    assert _dataset_for_total_index(500) == "bfcl"        # 500-749
    assert _dataset_for_total_index(750) == "tau2"        # 750-999
    assert _dataset_for_total_index(9999) is None

    # A bfcl trace whose data_source is the CONTENT source must still find its bfcl prediction.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        res = Path(td) / "eval" / "results" / "J"
        res.mkdir(parents=True)
        # ti = 500 + qi*5 + si = 500 → qi=0,si=0
        (res / "bfcl__blind_j.jsonl").write_text(
            '{"data_source":"bfcl_multi_turn_base","query_index":0,"sample_index":0,'
            '"step_labels":{"2":-1},"final_label":-1,"status":"done"}\n'
        )
        rec = {
            "data_source": "bfcl_multi_turn_base", "total_index": 500, "final_label": -1,
            "step_labels": {"2": -1},
            "messages": [
                {"role": "system", "content": "p"}, {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a2"}, {"role": "user", "content": "###STOP###"},
            ],
        }
        trace = load_apb_record(rec)
        preds = apb_judge_outcome_predictors(Path(td))
        # The judge's prediction joins despite data_source != dataset key, so it flags failure.
        assert preds["judge:blind_j"](trace) is True


def _tau2_trace(tid: str) -> Trace:
    return Trace(
        trace_id=tid, agent_id="airline", task_type="airline",
        spec=TaskSpec(goal="t", allowed_tools=[]),
        spans=[Span(span_id="s0", index=0, kind=SpanKind.USER_MSG)],
        outcome=Outcome(status=OutcomeStatus.SUCCESS), label=None,
        meta={"domain": "airline"},
    )


def test_tau2_broad_prompt_cache_violated_field(tmp_path: Path) -> None:
    # broad-prompt shape: {"verdicts": {uuid: {"violated": bool}}}.
    cache = tmp_path / "broad.json"
    cache.write_text(
        '{"model":"m","verdicts":{"u1":{"violated":true,"confidence":0.9},'
        '"u2":{"violated":false}}}'
    )
    preds = tau2_cached_outcome_predictors({"bp": cache})
    assert set(preds) == {"judge:bp"}
    assert preds["judge:bp"](_tau2_trace("u1"), None) is True
    assert preds["judge:bp"](_tau2_trace("u2"), None) is False
    assert preds["judge:bp"](_tau2_trace("uncovered"), None) is False  # absent → no flag


def test_tau2_det_cache_per_model(tmp_path: Path) -> None:
    # det-leaderboard shape: {"verdicts": {model: {uuid: bool}}} → one contestant per model.
    cache = tmp_path / "det.json"
    cache.write_text('{"verdicts":{"modelA":{"u1":true,"u2":false},"modelB":{"u1":false}}}')
    preds = tau2_cached_outcome_predictors({"det": cache})
    assert set(preds) == {"judge:det:modelA", "judge:det:modelB"}
    assert preds["judge:det:modelA"](_tau2_trace("u1"), None) is True
    assert preds["judge:det:modelB"](_tau2_trace("u1"), None) is False


def test_real_clone_judges_score_without_join_errors() -> None:
    if not (_APB_CLONE / "eval" / "results").exists():
        return  # clone absent → hermetic tests above cover the logic
    import json

    # Load a handful of real tau2 gold records + traces.
    recs, traces = [], {}
    with (_APB_CLONE / "data" / "AgentProcessBench" / "tau2.jsonl").open() as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            rec = json.loads(line)
            t = load_apb_record(rec)
            recs.append(apb_gold_from_record(t, rec))
            traces[t.trace_id] = t

    graders = apb_judge_step_graders(_APB_CLONE)
    assert len(graders) >= 10  # ~20 shipped verifiers
    # Score them (include_excluded since tau2 slice is excluded by default).
    scores = score_pq_entries(recs, traces, graders[:3], include_excluded=True)
    # At least one judge produced a non-trivial harmful-class denominator on these records.
    any_scored = any(
        sc.n_harmful + sc.n_helpful > 0
        for by_cell in scores.values()
        for sc in by_cell.values()
    )
    assert any_scored
