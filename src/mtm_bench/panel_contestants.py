"""Reused-prediction CONTESTANTS for the panel: APB's 20 shipped LLM step-verifiers, free.

AgentProcessBench ships 20 LLM verifiers' blind predictions under ``eval/results/<MODEL>/`` — each
a real frontier judge (GPT-5.x, Gemini-3, DeepSeek, Kimi, Qwen, Llama families) that read the
trajectory and emitted per-assistant-message ``step_labels`` + a trajectory ``final_label``. We
replay those frozen predictions as panel contestants (the "reused-prediction adapter" pattern,
EXP-0033) — NO new model spend. This is what turns the panel from a free-ladder demo into the
over-flagging hook + the "strong on one axis, weak on another" claim (synthesis §3 claim #1).

The join is the apb_leaderboard ``_record_key`` (validated EXP-0033): a prediction's
``(query_index, sample_index)`` resolves to the same ``ti:<total_index>`` as the gold record, and
the panel's APB ``trace_id`` is ``apb:<data_source>:<total_index>`` — so a contestant looks its
prediction up by the trace's ``apb_total_index``. A missing/failed prediction = "no flag" (+1 /
outcome ok), the apb_leaderboard convention, so a judge that didn't answer cannot inflate recall.

FIREWALL note: a reused prediction is keyed by trace identity and looked up — the judge already ran
blind on the trajectory (it never saw gold). Replaying its output is not an answer-key read; the
gold ``step_labels``/``final_label`` never enter the contestant. (Same status as the apb_leaderboard
LLM rungs.)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .schema import Trace

from .apb_leaderboard import _DATASET_OFFSETS, APBRecord, _load_apb
from .panel_attribution import LocalizerEntry
from .panel_scoring import StepGraderEntry

APB_DATASETS = ("tau2", "hotpotqa", "bfcl", "gaia_dev")


def _dataset_for_total_index(ti: int) -> str | None:
    """Map an APB ``total_index`` to its dataset via the fixed 250-wide offset blocks
    (``_DATASET_OFFSETS``: hotpotqa 0-249, gaia_dev 250-499, bfcl 500-749, tau2 750-999).

    NOTE: the trace's ``data_source`` (e.g. ``bfcl_multi_turn_base``, ``searchR1_hotpotqa``) is the
    CONTENT source, NOT the dataset key the prediction files use (``bfcl``/``hotpotqa``) — joining
    on ``data_source`` silently misses (the bug this fixes). The total_index block is the
    unambiguous key both sides share."""
    for ds, off in _DATASET_OFFSETS.items():
        if off <= ti < off + 250:
            return ds
    return None


def _msg_index_to_span(trace: Trace) -> dict[int, str]:
    """Map each APB assistant-message index → the representative (first-in-order) span_id of that
    message — the SAME inversion ``apb_gold`` uses, so a predicted step_labels keyed by msg-index
    lands on the exact span_ids the gold span_values use."""
    span_msg = trace.meta.get("apb_msg_index") or {}
    span_order = {s.span_id: s.index for s in trace.spans}
    msg_to_spans: dict[int, list[str]] = defaultdict(list)
    for sid, mi in span_msg.items():
        msg_to_spans[int(mi)].append(sid)
    return {
        mi: sorted(sids, key=lambda s: span_order.get(s, 1 << 30))[0]
        for mi, sids in msg_to_spans.items()
    }


def _load_apb_predictions(apb_dir: Path) -> dict[str, dict[str, dict[str, APBRecord]]]:
    """{run_name: {dataset: {record_key: APBRecord}}} for every shipped LLM verifier."""
    res = Path(apb_dir).expanduser() / "eval" / "results"
    by_run: dict[str, dict[str, dict[str, APBRecord]]] = {}
    for mdir in sorted(p for p in res.iterdir() if p.is_dir()):
        for f in sorted(mdir.glob("*.jsonl")):
            if any(p in {"raw", "_raw", "llm_annotations_raw"} for p in f.parts):
                continue
            ds = f.name.split("__", 1)[0]
            if ds not in APB_DATASETS:
                continue
            run = f.name.split("__", 1)[1].removesuffix(".jsonl") if "__" in f.name else f.stem
            by_run.setdefault(run, {})[ds] = _load_apb(f, ds)
    return by_run


def _pred_for_trace(
    preds_by_ds: dict[str, dict[str, APBRecord]], trace: Trace
) -> APBRecord | None:
    """Look a contestant's prediction up by the trace's APB ``total_index`` (the shared join key).

    The dataset is resolved from the total_index BLOCK (``_dataset_for_total_index``), never from
    the trace's ``data_source`` string — the latter is the content source and does not match the
    prediction files' dataset keys (the silent-miss bug)."""
    ti = trace.meta.get("apb_total_index")
    if ti is None:
        return None
    ds = _dataset_for_total_index(int(ti))
    if ds is None or ds not in preds_by_ds:
        return None
    rec = preds_by_ds[ds].get(f"ti:{int(ti)}")
    if rec is None or rec.failed:
        return None
    return rec


def apb_judge_step_graders(apb_dir: Path) -> list[StepGraderEntry]:
    """One process-quality contestant per shipped LLM verifier: replay its predicted ``step_labels``
    (msg-index keyed) onto the trace's representative span_ids → ``{span_id: -1|0|1}``."""
    by_run = _load_apb_predictions(apb_dir)
    entries: list[StepGraderEntry] = []
    for run, preds_by_ds in sorted(by_run.items()):
        def _grader(trace: Trace, _preds=preds_by_ds):
            rec = _pred_for_trace(_preds, trace)
            if rec is None:
                return {}  # no flag everywhere (missing prediction)
            m2s = _msg_index_to_span(trace)
            out: dict[str, int] = {}
            for msg_key, val in rec.step_labels.items():
                span = m2s.get(int(msg_key)) if msg_key.lstrip("-").isdigit() else None
                if span is not None:
                    out[span] = val
            return out
        entries.append(StepGraderEntry(name=f"judge:{run}", grader=_grader))
    return entries


def apb_judge_outcome_predictors(apb_dir: Path) -> dict[str, callable]:
    """One outcome predictor per shipped LLM verifier: ``trace -> bool`` (True == predicts FAILURE),
    from the replayed ``final_label`` (-1 → failure-flag; +1/0/missing → no flag). Plugs into the
    leaderboard's predicate-entry shape so judges score on the OUTCOME axis too."""
    by_run = _load_apb_predictions(apb_dir)
    out: dict[str, callable] = {}
    for run, preds_by_ds in sorted(by_run.items()):
        def _predict(trace: Trace, _spec=None, _preds=preds_by_ds) -> bool:
            rec = _pred_for_trace(_preds, trace)
            return bool(rec is not None and rec.final_label == -1)
        out[f"judge:{run}"] = _predict
    return out


def tau2_cached_outcome_predictors(
    cache_paths: dict[str, str | Path],
) -> dict[str, callable]:
    """Outcome-axis judge contestants for tau2 from FROZEN verdict caches (no spend).

    ``cache_paths`` maps a contestant name → a tau2 judge cache JSON. Two shapes are recognized,
    both keyed by the tau2 simulation UUID (== the panel trace_id):
      • broad-prompt diagnostic: ``{"model":..., "verdicts": {uuid: {"violated": bool, ...}}}``
        — one model, full held-out coverage;
      • deterministic leaderboard: ``{"verdicts": {model: {uuid: bool}}}`` — several models, the
        config_only cell's coverage.
    Each becomes a ``trace -> bool`` predicate (True == the judge flags a policy violation/failure).
    A trace the cache does not cover → ``False`` (no flag), so partial coverage cannot inflate
    recall (missing traces get no fire but the scorer still counts them via the gold pool, so a
    thin-coverage judge shows LOW recall honestly — verify coverage before quoting a number)."""
    out: dict[str, callable] = {}
    for name, path in cache_paths.items():
        blob = json.loads(Path(path).expanduser().read_text())
        verds = blob.get("verdicts", blob)
        # Detect shape: broad-prompt has {uuid: {violated}}; det-cache has {model: {uuid: bool}}.
        sample = next(iter(verds.values())) if verds else None
        if isinstance(sample, dict) and "violated" in sample:
            flat = {uid: bool(v.get("violated")) for uid, v in verds.items()}
            out[f"judge:{name}"] = (lambda t, _s=None, _f=flat: bool(_f.get(t.trace_id, False)))
        elif isinstance(sample, dict):  # {model: {uuid: bool}}
            for model, mverds in verds.items():
                flat = {uid: bool(b) for uid, b in mverds.items()}
                label = f"judge:{name}:{model}"
                out[label] = (lambda t, _s=None, _f=flat: bool(_f.get(t.trace_id, False)))
    return out


def apb_judge_localizers(apb_dir: Path) -> list[LocalizerEntry]:
    """One attribution contestant per shipped LLM verifier: its predicted critical step = the FIRST
    msg with a predicted ``-1`` (apb_leaderboard's first-neg1 convention), mapped to that message's
    representative span_id. None when the judge predicts no -1 (an abstain → a miss)."""
    by_run = _load_apb_predictions(apb_dir)
    entries: list[LocalizerEntry] = []
    for run, preds_by_ds in sorted(by_run.items()):
        def _localize(trace: Trace, _preds=preds_by_ds) -> str | None:
            rec = _pred_for_trace(_preds, trace)
            if rec is None:
                return None
            neg = [int(k) for k, v in rec.step_labels.items()
                   if v == -1 and k.lstrip("-").isdigit()]
            if not neg:
                return None
            return _msg_index_to_span(trace).get(min(neg))
        entries.append(LocalizerEntry(name=f"judge:{run}", localize=_localize))
    return entries


__all__ = [
    "apb_judge_step_graders",
    "apb_judge_outcome_predictors",
    "apb_judge_localizers",
    "tau2_cached_outcome_predictors",
]
