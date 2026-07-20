"""Provenance-first unified gold sidecar for the cross-domain meta-eval PANEL.

Four agent-trajectory gold corpora (SOP-Bench, AgentErrorBench, AgentProcessBench, tau2-verified)
carry gold on **three different axes** — outcome, failure-attribution, and per-step process-quality.
This module is the *data layer* the grader-of-graders panel reads: it puts four heterogeneous golds
behind ONE loader contract such that

  (a) a contestant detector physically cannot read the answer key — the FIREWALL, made structural
      by keeping gold in a sidecar (``GoldRecord``) keyed by ``trace_id``, NEVER merged into the
      ``Trace`` a detector sees;
  (b) the two un-pooled numbers (recall-on-failure, fire-on-clean) are computable per substrate —
      by projecting a ``GoldRecord`` down to the shipped ``leaderboard.GoldItem`` (no new scoring);
  (c) provenance/reliability heterogeneity (deterministic-but-eval-faulted vs human-kappa-0.55 vs
      human-raw-agreement-only vs human-confirmed-state-hash) is forced INTO THE TYPE, not buried
      in a footnote — ``kappa_status`` makes "raw agreement, not kappa" un-presentable as kappa.

Design provenance: ``docs/planning/UNIFIED-GOLD-SCHEMA-SYNTHESIS-2026-06-29.md`` (the winning
provenance-first design + four grafts) and EXP-0094 (APB panel-adapter red-team, verified facts).

GRAFTS folded in (each closes a real gap the runner-up design exposed):
  1. REUSE ``OutcomeStatus`` (success|failure|partial|unknown), not a bespoke enum
     → gives ``partial``/``unknown`` free; ``has_clean_pool = any(outcome==SUCCESS)``; composes
     with leaderboard.py's ``n_clean==0 -> nan -> "n/a"`` path with ZERO new scoring code.
  2. Mechanical ``eval_fault`` / ``excluded_from_pooling`` flags the harness READS — droppable
     slices (SOP-Bench order_fulfillment tool-unreachable labels; APB tau2 41.8%-by-step overlap
     vs the depth leg), not free-text provenance notes.
  3. ``firewall.answer_key_fields`` is an AUDITED ledger (``find_leaks`` is the backstop), not a
     passive doc string — the hook the EXP-0080 firewall ablation (+35pp) actually measures.
  4. Per-step gold joins on the UNIQUE ``Span.span_id``, never the decode-grouping ``Span.step_id``
     (which is shared across a parallel decode and ``None`` when unknown → silent mis-alignment).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from .schema import OutcomeStatus, Trace

from .leaderboard import GoldItem

Substrate = Literal["sop_bench", "aeb", "apb", "tau2_verified"]
Source = Literal["deterministic", "human", "llm_judge", "hybrid"]
# hybrid = human-of-record with visible-but-collapsed llm anchoring (the APB case).


# ───────────────────────── 1. PROVENANCE & RELIABILITY ─────────────────────────
# Top of the type tree, not a footnote. The reliability tagged union makes it
# impossible to emit a human label without declaring HOW its agreement was measured.


class DeterministicReliability(BaseModel):
    kind: Literal["deterministic"] = "deterministic"
    grader_id: str  # e.g. "sopbench_fuzzy_v1", "tau2_state_hash"


class HumanReliability(BaseModel):
    """kappa-PRIMARY enforced (the project's annotation-personnel discipline): raw agreement alone
    is INVALID without a ``kappa_status`` that says why kappa is absent. A reader can never be shown
    raw agreement as if it were chance-corrected."""

    kind: Literal["human"] = "human"
    n_annotators: int
    cohen_kappa: float | None = None
    kappa_status: Literal["reported", "unreported_raw_only", "single_annotator"]
    raw_agreement: float | None = None
    kappa_subset_n: int | None = None

    @model_validator(mode="after")
    def _kappa_primary(self) -> HumanReliability:
        if self.kappa_status == "reported" and self.cohen_kappa is None:
            raise ValueError("kappa_status='reported' requires a non-null cohen_kappa")
        if self.kappa_status == "unreported_raw_only" and self.cohen_kappa is not None:
            raise ValueError(
                "kappa_status='unreported_raw_only' must have cohen_kappa=None "
                "(if a kappa exists, status is 'reported')"
            )
        if self.kappa_status == "single_annotator" and self.n_annotators > 1:
            raise ValueError("kappa_status='single_annotator' contradicts n_annotators>1")
        return self


class LlmJudgeReliability(BaseModel):
    kind: Literal["llm_judge"] = "llm_judge"
    model_id: str
    saw_ground_truth: bool  # blind vs reference-fed


Reliability = Annotated[
    DeterministicReliability | HumanReliability | LlmJudgeReliability,
    Field(discriminator="kind"),
]


class Anchoring(BaseModel):
    """How a human-of-record label was anchored to an automated suggestion (APB collapses an LLM
    reference panel behind a ``<details>`` element; the human is authority-of-record)."""

    anchor_source: Source
    strength: Literal["none", "collapsed_ondemand", "prefilled"]


class Provenance(BaseModel):
    source: Source
    reliability: Reliability
    anchoring: Anchoring | None = None


# ───────────────────────── 2. THE THREE LABEL SHAPES ─────────────────────────
# One axis per gold record, as a tagged union discriminated by ``axis``.


class OutcomeLabel(BaseModel):
    axis: Literal["outcome"] = "outcome"
    kind: Literal["binary"] = "binary"
    value: bool  # pass
    basis: list[str] | None = None  # e.g. ["state_hash"], ["DB","COMMUNICATE"]


class AttributionLabel(BaseModel):
    axis: Literal["attribution"] = "attribution"
    kind: Literal["categorical"] = "categorical"
    critical_span_id: str  # GRAFT 4: the UNIQUE Span.span_id, NOT step_id
    step_index: int | None = None  # int mirror for position-keyed substrates
    category: str  # failure_type / module
    vocab_id: str  # names the (un)controlled vocabulary, e.g. "aeb_cognitive_v1"
    free_text: str | None = None
    falsifiable: bool = True  # AEB 14% empty failure_type+reasoning -> False


class ProcessQualityLabel(BaseModel):
    axis: Literal["process_quality"] = "process_quality"
    kind: Literal["ordinal"] = "ordinal"
    scale: list[int]  # APB: [-1, 0, 1]
    span_values: dict[str, int]  # GRAFT 4: keyed on span_id, not step_id
    propagation_rule: str | None = None  # e.g. "-1 sticks until corrected" (autocorrelation)

    @model_validator(mode="after")
    def _values_in_scale(self) -> ProcessQualityLabel:
        bad = {v for v in self.span_values.values() if v not in self.scale}
        if bad:
            raise ValueError(f"span_values {bad} outside declared scale {self.scale}")
        return self


Label = Annotated[
    OutcomeLabel | AttributionLabel | ProcessQualityLabel,
    Field(discriminator="axis"),
]


# ───────────────────────── 3. THE GOLD RECORD ─────────────────────────


class AuxOutcome(BaseModel):
    """A second, demoted axis (APB carries per-step process-quality AND a trajectory final)."""

    value: bool
    provenance: Provenance


class Firewall(BaseModel):
    """GRAFT 3: the answer-key ledger as an AUDITED hook. ``answer_key_fields`` are the field NAMES
    a contestant must never read; ``find_leaks`` is the backstop that verifies ingest redacted them
    from the Trace a detector sees."""

    answer_key_fields: list[str]
    leak_surfaces: list[str] = Field(default_factory=list)  # free-text recompute-trap notes


class GoldRecord(BaseModel):
    """Provenance-first gold for ONE trace. Keyed to a ``Trace`` by ``trace_id`` — the ONLY join
    key. The ``Trace`` itself carries no gold; that separation IS the firewall, made structural."""

    trace_id: str  # FK to Trace
    substrate: Substrate
    domain: str  # "tau2_airline","alfworld","gaia","order_fulfillment", ...
    outcome: OutcomeStatus  # GRAFT 1: MANDATORY on every record, reusing the shipped enum.
    #   APB final==0 lands in 'unknown'/'partial', NEVER mis-bucketed into 'failure'.
    label: Label  # exactly one axis per record
    provenance: Provenance

    # GRAFT 2: mechanical, harness-readable flags (not free-text provenance notes).
    eval_fault: bool = False  # SOP-Bench order_fulfillment tool-unreachable labels (EXP-0090)
    eval_fault_reason: str = ""
    excluded_from_pooling: bool = False  # APB tau2 41.8%-by-step overlap; SOP-Bench faulted slice

    aux_outcome: AuxOutcome | None = None  # optional demoted second axis (APB final_label)
    firewall: Firewall

    @model_validator(mode="after")
    def _eval_fault_reason_present(self) -> GoldRecord:
        if self.eval_fault and not self.eval_fault_reason:
            raise ValueError("eval_fault=True requires a non-empty eval_fault_reason")
        if not self.eval_fault and self.eval_fault_reason:
            raise ValueError("eval_fault_reason set but eval_fault=False")
        return self


# ───────────────────────── 4. THE LOADER CONTRACT (firewall as code) ─────────────────────────


def has_clean_pool(records: list[GoldRecord]) -> bool:
    """True iff any in-pool record's outcome is SUCCESS — the denominator behind fire-on-clean.

    Composes with leaderboard.py: when this is False (AEB: all 200 failures), the projected GoldItem
    set has ``n_clean==0`` and the scorer's ``clean_fire_rate`` is ``nan`` -> printed "n/a". Records
    flagged ``excluded_from_pooling`` do not contribute (APB tau2 slice double-counts the depth)."""
    return any(
        r.outcome == OutcomeStatus.SUCCESS and not r.excluded_from_pooling for r in records
    )


def to_gold_items(
    records: list[GoldRecord],
    *,
    tier: str = "instance_label",
    include_excluded: bool = False,
) -> list[GoldItem]:
    """Project ``GoldRecord``s down to ``leaderboard.GoldItem``s so the SHIPPED two-number scorer
    runs over the panel with NO new scoring code (synthesis §4 step 4).

    The projection reads the record's MANDATORY ``outcome`` (graft 1), never its label axis — that
    is the whole point of forcing ``outcome`` onto every record: a clean/failure partition is
    computable for the attribution (AEB) and process-quality (APB) legs alike, so they flow through
    the outcome-shaped scorer that reports recall-on-failure + fire-on-clean. A record whose
    outcome is neither SUCCESS nor FAILURE (``unknown``/``partial`` — e.g. APB ``final==0``) is NOT
    scoreable on this binary cell and is dropped (the caller records it; never mis-bucketed).

    ``corrupt_success`` semantics match the leaderboard: a FAILURE record is the positive class the
    detector should flag (``corrupt_success=True``); a SUCCESS record is clean
    (``corrupt_success=False``) → its firing is a false positive. ``cell`` is ``substrate|domain``
    so the report is stratified per substrate (never pooled). ``excluded_from_pooling`` records are
    dropped by default (the APB tau2 slice double-counts the depth leg); pass ``include_excluded``
    to score them in isolation. ``source`` carries the provenance kind for the tautology guard."""
    items: list[GoldItem] = []
    for r in records:
        if r.excluded_from_pooling and not include_excluded:
            continue
        if r.outcome == OutcomeStatus.SUCCESS:
            corrupt = False
        elif r.outcome == OutcomeStatus.FAILURE:
            corrupt = True
        else:
            continue  # unknown/partial: not scoreable on the binary outcome cell
        items.append(
            GoldItem(
                trace_id=r.trace_id,
                cell=f"{r.substrate}|{r.domain}",
                tier=tier,
                corrupt_success=corrupt,
                source=f"{r.substrate}:{r.provenance.reliability.kind}",
            )
        )
    return items


class GoldStore:
    """Holds traces and gold SEPARATELY. A contestant detector is handed a ``Trace`` via
    ``load_trace`` and has no reference to the ``GoldRecord`` (available only to the scorer via
    ``load_gold``). That separation is the firewall; ``find_leaks`` audits that ingest actually
    redacted the in-trace answer-key surfaces (SOP-Bench ``output_columns`` in the tool-result
    stream; APB inline ``ground_truth`` — both live INSIDE the trace's own stream, §4 open risk #1).

    Traces handed in here are expected to be ALREADY redacted by the substrate adapter. The store
    re-audits on registration (``strict=True`` raises on a residual leak) so a forgotten redaction
    fails loudly at ingest instead of silently defeating the firewall at score time."""

    def __init__(self, *, strict: bool = True) -> None:
        self._traces: dict[str, Trace] = {}
        self._gold: dict[str, GoldRecord] = {}
        self._strict = strict

    def register(self, trace: Trace, gold: GoldRecord) -> None:
        if trace.trace_id != gold.trace_id:
            raise ValueError(
                f"trace_id mismatch: trace={trace.trace_id!r} gold={gold.trace_id!r}"
            )
        assert_ingest_invariants(trace, gold)
        leaks = find_leaks(trace, gold)
        if leaks and self._strict:
            raise FirewallViolation(trace.trace_id, leaks)
        self._traces[trace.trace_id] = trace
        self._gold[trace.trace_id] = gold

    def load_trace(self, trace_id: str) -> Trace:
        """Return the (redacted) trace a contestant detector sees. No gold reachable from here."""
        return self._traces[trace_id]

    def load_gold(self, trace_id: str) -> GoldRecord:
        """Return the gold record. SCORER-ONLY — never pass the result into a detector."""
        return self._gold[trace_id]

    def trace_ids(self) -> list[str]:
        return list(self._traces)

    def records(self) -> list[GoldRecord]:
        return list(self._gold.values())


class FirewallViolation(Exception):
    """Raised when a trace still carries an answer-key surface the firewall forbids."""

    def __init__(self, trace_id: str, leaks: list[str]) -> None:
        self.trace_id = trace_id
        self.leaks = leaks
        super().__init__(f"firewall: trace {trace_id!r} leaks answer-key fields: {leaks}")


def _walk_keys(obj: Any) -> list[str]:
    """Every dict key appearing anywhere in a nested structure (a surface a detector could read)."""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.append(str(k))
            found.extend(_walk_keys(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found.extend(_walk_keys(v))
    return found


def find_leaks(trace: Trace, gold: GoldRecord) -> list[str]:
    """Backstop audit (GRAFT 3): return every ``answer_key_fields`` name that still appears as a
    structured KEY anywhere in the trace's spans (``meta`` + ``tool_result.value``). A non-empty
    result means ingest forgot to redact an answer-key surface — the #1 silent firewall failure.

    Reports ``"<span_id>.<field>"`` per leak. Detects the field-NAME-as-key failure mode (the
    forgot-to-strip case); literal answer-text leakage is the adapter's redaction responsibility,
    which this auditor cannot reconstruct (documented limitation, §4 open risk #1)."""
    banned = set(gold.firewall.answer_key_fields)
    if not banned:
        return []
    leaks: list[str] = []
    for span in trace.spans:
        surfaces: list[Any] = []
        if span.meta:
            surfaces.append(span.meta)
        if span.tool_result is not None and span.tool_result.value is not None:
            surfaces.append(span.tool_result.value)
        for surface in surfaces:
            for key in _walk_keys(surface):
                if key in banned:
                    leaks.append(f"{span.span_id}.{key}")
    return leaks


def assert_ingest_invariants(trace: Trace, gold: GoldRecord) -> None:
    """GRAFT 4 backstop: per-step / attribution gold must resolve to REAL spans of THIS trace (no
    out-of-bounds), since neither design enforces the join at the type level. Runs at register()."""
    span_ids = {s.span_id for s in trace.spans}
    label = gold.label
    if isinstance(label, AttributionLabel):
        if label.critical_span_id not in span_ids:
            raise ValueError(
                f"attribution gold for {gold.trace_id!r}: critical_span_id "
                f"{label.critical_span_id!r} not a span of the trace"
            )
    elif isinstance(label, ProcessQualityLabel):
        unresolved = [k for k in label.span_values if k not in span_ids]
        if unresolved:
            raise ValueError(
                f"process_quality gold for {gold.trace_id!r}: {len(unresolved)} span_values "
                f"keys do not resolve to trace spans (first few: {unresolved[:5]})"
            )


__all__ = [
    "Substrate",
    "Source",
    "DeterministicReliability",
    "HumanReliability",
    "LlmJudgeReliability",
    "Reliability",
    "Anchoring",
    "Provenance",
    "OutcomeLabel",
    "AttributionLabel",
    "ProcessQualityLabel",
    "Label",
    "AuxOutcome",
    "Firewall",
    "GoldRecord",
    "GoldStore",
    "FirewallViolation",
    "has_clean_pool",
    "to_gold_items",
    "find_leaks",
    "assert_ingest_invariants",
]
