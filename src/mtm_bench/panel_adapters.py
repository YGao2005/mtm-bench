"""Substrate ingest adapters: an upstream ``Trace`` + its native gold → a panel ``GoldRecord``.

One adapter per substrate. Each takes a Trace ALREADY built by the substrate's ``argus_adapters``
loader (so span construction / step_id / the answer-key firewall on the Trace side is reused, never
duplicated) plus that substrate's native gold blob, and emits the provenance-first ``GoldRecord``
(``panel.py``). The adapters are where the per-substrate lossy mapping (synthesis §2) and the
verified reliability facts (EXP-0094) are encoded ONCE.

This increment ships the two adapters whose data shapes were verified on-disk (EXP-0094 session):
  - ``tau2_verified_gold`` — the REFERENCE adapter (outcome axis, deterministic state-hash gold);
  - ``apb_gold`` — the process-quality axis (ordinal [-1,0,1], hybrid human-of-record gold).

SOP-Bench (needs paid Bedrock rollouts) and AEB (needs the Drive trajectory download) are the next
build increment — their reliability/firewall facts are already pinned in EXP-0090/0091/0092 and the
synthesis §2, so adding them is a localized addition here, not a redesign.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schema import OutcomeStatus, Trace

from .panel import (
    Anchoring,
    AttributionLabel,
    Firewall,
    GoldRecord,
    HumanReliability,
    OutcomeLabel,
    ProcessQualityLabel,
    Provenance,
)

# ───────────────────────── tau2-verified (REFERENCE adapter) ─────────────────────────


def tau2_verified_gold(trace: Trace, *, human_confirmed: bool = True) -> GoldRecord:
    """tau2-verified → outcome/binary GoldRecord (the DEPTH leg, highest-trust anchor).

    The Trace is built by ``mtm_bench.tau2_loader.simulation_to_trace``; its ``outcome.status`` is the
    state-hash reward (SUCCESS iff reward>=1.0) and ``meta['reward_basis']`` is the scored basis
    (e.g. ``["DB","COMMUNICATE"]`` / ``["state_hash"]``). The outcome gold IS that deterministic
    reward — so this adapter READS ``trace.outcome`` deliberately (unlike a detector, the SCORER
    may; ADR-0008 forbids only *detectors* reading the oracle). Provenance is ``deterministic`` with
    an optional human-confirmation ``anchoring`` for the verified subset (EXP-0074/0087).

    Firewall: ``state_hash`` is an environment property, NOT in the transcript → leak risk LOW (the
    reward fields are on ``meta``/``outcome``, redacted from a contestant's view by the GoldStore
    contract; they are not in the span stream a detector reads)."""
    status = trace.outcome.status
    basis = trace.meta.get("reward_basis")
    if not isinstance(basis, list):
        basis = ["state_hash"]
    domain = f"tau2_{trace.task_type}"

    provenance = Provenance(
        source="deterministic",
        reliability={"kind": "deterministic", "grader_id": "tau2_state_hash"},
        # The verified subset is a deterministic grade a human confirmed equals the reference —
        # recorded as anchoring so the depth leg's higher trust is visible in the type, not prose.
        anchoring=(
            Anchoring(anchor_source="human", strength="collapsed_ondemand")
            if human_confirmed
            else None
        ),
    )
    return GoldRecord(
        trace_id=trace.trace_id,
        substrate="tau2_verified",
        domain=domain,
        outcome=status,
        label=OutcomeLabel(value=(status == OutcomeStatus.SUCCESS), basis=basis),
        provenance=provenance,
        firewall=Firewall(
            answer_key_fields=["reward", "reward_info", "reward_basis"],
            leak_surfaces=["state_hash is an environment property, not in the transcript (LOW)"],
        ),
    )


# ───────────────────────── APB / AgentProcessBench (process-quality leg) ─────────────────────────

# APB ternary effectiveness scale (guide §2). +1 helpful / 0 neutral / -1 harmful.
_APB_SCALE: list[int] = [-1, 0, 1]
# Reliability facts pinned in EXP-0094: 89.1% RAW agreement (paper abstract), kappa NOT re-derivable
# from the release (consensus-only; per-annotator exports absent) → kappa_status forced raw-only.
_APB_RAW_AGREEMENT = 0.891


def _final_label_to_outcome(final_label: int | None) -> OutcomeStatus:
    """APB ``final_label`` → OutcomeStatus, honestly (EXP-0094 / synthesis graft 1).

    +1 → SUCCESS; -1 → FAILURE; **0 → UNKNOWN** (NOT failure) — matching ``apb_leaderboard.py``'s
    ``final==0``-excluded semantics; a forced-boolean mapping would mis-bucket the neutral final
    into the failure pool and corrupt the clean/failure partition. Missing → UNKNOWN."""
    if final_label is None:
        return OutcomeStatus.UNKNOWN
    if final_label >= 1:
        return OutcomeStatus.SUCCESS
    if final_label <= -1:
        return OutcomeStatus.FAILURE
    return OutcomeStatus.UNKNOWN


def apb_gold(
    trace: Trace,
    step_labels: dict[str, int],
    final_label: int | None,
) -> GoldRecord:
    """APB → process_quality/ordinal GoldRecord (the per-step PRM-credit leg).

    ``trace`` is built by ``argus_adapters.apb.load_apb_record`` (carries
    ``meta['apb_msg_index'] = {span_id: enclosing-msg-index}``). ``step_labels`` is APB's native
    per-assistant-message gold (``{str(msg_index): -1|0|1}``); ``final_label`` is the trajectory
    final.

    GRAFT 4 join: APB keys gold by the assistant MESSAGE index, and one message may explode into
    several spans (parallel calls / content+calls). The label is the "net contribution of the WHOLE
    assistant message" (guide §0 — one label per message), so we key ``span_values`` on the
    **representative (first-in-trace-order) span** of each labeled message. That span_id resolves to
    a real span (the ingest invariant), and we never fabricate N independent labels from one.

    Reliability: hybrid (human-of-record + an LLM reference panel collapsed behind a ``<details>``);
    ``kappa_status='unreported_raw_only'`` with ``raw_agreement=0.891`` (EXP-0094 — kappa NOT
    re-derivable). The tau2 slice is the SAME substrate as the depth leg, so it is flagged
    ``excluded_from_pooling`` (41.8% by step label — synthesis §2).
    """
    span_msg = trace.meta.get("apb_msg_index") or {}
    # Invert {span_id: msg_index} → {msg_index: [span_ids in trace order]}.
    msg_to_spans: dict[int, list[str]] = defaultdict(list)
    span_order = {s.span_id: s.index for s in trace.spans}
    for sid, mi in span_msg.items():
        msg_to_spans[int(mi)].append(sid)
    for mi in msg_to_spans:
        msg_to_spans[mi].sort(key=lambda sid: span_order.get(sid, 1 << 30))

    span_values: dict[str, int] = {}
    for key, val in step_labels.items():
        mi = int(key)
        spans = msg_to_spans.get(mi)
        if not spans:
            # A labeled message with no span in the trace would break the join — surface it loudly
            # rather than silently dropping a human label (the ingest invariant catches it too).
            raise ValueError(
                f"APB gold for {trace.trace_id!r}: step_label msg index {mi} has no span in the "
                f"trace (step_labels keys must == assistant-message span set)"
            )
        span_values[spans[0]] = int(val)  # representative span = first in trace order

    data_source = trace.meta.get("data_source", "")
    domain = str(data_source) or trace.task_type
    is_tau2_slice = str(data_source).startswith("tau2")

    provenance = Provenance(
        source="hybrid",  # human-of-record + collapsed llm anchoring
        reliability=HumanReliability(
            n_annotators=3,  # APB annotation platform; consensus released (EXP-0094)
            kappa_status="unreported_raw_only",
            raw_agreement=_APB_RAW_AGREEMENT,
        ),
        anchoring=Anchoring(anchor_source="llm_judge", strength="collapsed_ondemand"),
    )

    return GoldRecord(
        trace_id=trace.trace_id,
        substrate="apb",
        domain=domain,
        outcome=_final_label_to_outcome(final_label),
        label=ProcessQualityLabel(
            scale=list(_APB_SCALE),
            span_values=span_values,
            propagation_rule="-1 sticks until corrected",  # autocorrelation (synthesis §2)
        ),
        provenance=provenance,
        # The tau2 slice double-counts the depth leg (same substrate) → exclude from pooling.
        excluded_from_pooling=is_tau2_slice,
        firewall=Firewall(
            answer_key_fields=["step_labels", "final_label", "ground_truth"],
            leak_surfaces=[
                "AnswerOnly/*_final.jsonl (bare answer key)",
                "inline ground_truth/answer_text beside the trajectory",
                "--mode reference (GT-fed eval path)",
            ],
        ),
    )


def apb_gold_from_record(trace: Trace, record: dict[str, Any]) -> GoldRecord:
    """Convenience: pull ``step_labels``/``final_label`` straight off a raw APB record dict."""
    return apb_gold(trace, record.get("step_labels") or {}, record.get("final_label"))


# ───────────────────────── AEB / AgentErrorBench (attribution leg) ─────────────────────────

# AEB IAA: reported Cohen's kappa 0.55 (moderate, subset-only) — inherited, NOT re-derivable from
# the release (EXP-0092/0094: consensus-only labels). 10 expert annotators (paper).
_AEB_KAPPA = 0.55
_AEB_VOCAB = "aeb_cognitive_v1"  # the 17-type cognitive taxonomy — UNCONTROLLED (plan vs planning)


def aeb_gold(trace: Trace, label: dict[str, Any]) -> GoldRecord:
    """AEB consensus label → attribution/categorical GoldRecord (the failure-LOCALIZATION leg).

    ``trace`` is built by ``argus_adapters.aeb.load_aeb_trajectory`` (carries
    ``meta['aeb_step_to_span']`` = 1-based assistant-step → span_id). ``label`` is one AEB consensus
    record (``critical_failure_step`` + ``critical_failure_module``/``failure_type``). The critical
    step maps to its assistant-message span_id (GRAFT 4 join).

    FALSIFIABILITY (EXP-0092): a label is unfalsifiable when BOTH ``failure_type`` and ``reasoning``
    are empty (14% of AEB) — flagged ``falsifiable=False`` so the attribution scorer drops it from
    the localization denominator rather than scoring it as a silent miss. AEB is FAILURE-only →
    outcome FAILURE, fire-on-clean N/A. The category vocab is uncontrolled → ``vocab_id`` flags it;
    the axis is single-substrate (generality nominal)."""
    step_to_span = trace.meta.get("aeb_step_to_span") or {}
    crit_step = label.get("critical_failure_step")
    # The critical step → its assistant span. step_to_span keys are ints (1-based assistant turns).
    # Verified on ALFWorld: 75/75 critical steps map directly (no fallback). A step that does NOT
    # resolve is a real join error → raise, never silently re-point (the answer-key firewall + the
    # honest-localization contract both depend on the gold span being the TRUE critical step).
    crit_span = step_to_span.get(crit_step) or step_to_span.get(str(crit_step))
    if crit_span is None:
        raise ValueError(
            f"AEB gold for {trace.trace_id!r}: critical_failure_step {crit_step!r} does not "
            f"resolve to an assistant span (have {sorted(int(k) for k in step_to_span)})"
        )

    # falsifiable iff SOME step annotation carries a non-empty failure_type or reasoning (EXP-0092:
    # 14% overall are empty-both → unfalsifiable). Structure: step_annotations[].{module}.{...}.
    falsifiable = False
    for ann in label.get("step_annotations") or []:
        if not isinstance(ann, dict):
            continue
        for key, mod in ann.items():
            if key == "step" or not isinstance(mod, dict):
                continue
            if mod.get("failure_type") or mod.get("reasoning"):
                falsifiable = True
                break
        if falsifiable:
            break
    category = str(label.get("critical_failure_module") or "unknown")

    return GoldRecord(
        trace_id=trace.trace_id,
        substrate="aeb",
        domain=str(label.get("task_type") or trace.task_type),
        outcome=OutcomeStatus.FAILURE,  # AEB is failure-only
        label=AttributionLabel(
            critical_span_id=crit_span,
            step_index=crit_step if isinstance(crit_step, int) else None,
            category=category,
            vocab_id=_AEB_VOCAB,
            falsifiable=falsifiable,
        ),
        provenance=Provenance(
            source="human",
            reliability=HumanReliability(
                n_annotators=10, kappa_status="reported", cohen_kappa=_AEB_KAPPA
            ),
        ),
        firewall=Firewall(
            answer_key_fields=["critical_failure_step", "critical_failure_module",
                               "failure_type", "step_annotations"],
            leak_surfaces=["the label is a sidecar; the trajectory itself carries no answer key"],
        ),
    )


__all__ = [
    "tau2_verified_gold",
    "apb_gold",
    "apb_gold_from_record",
    "aeb_gold",
]
