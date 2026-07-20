"""Canonical, framework-agnostic agent-trace schema.

This is the contract between capture (argus-sdk) and everything downstream
(detectors, evals, benchmark). See docs/architecture/trace-schema.md.

Design rules (enforced in review):
- No framework or network imports here. Pure data + Pydantic.
- ``SpanKind`` is *what an event is*; ``Phase`` is *where in the agent lifecycle it
  sits*. Phase powers phase-attribution (ADR-0006), never causal claims.
- A trace may carry a ground-truth ``Label`` (datasets) and a measured ``Outcome``
  (programmatic checks). Both are optional so production traces work without them.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = 2


class SpanKind(StrEnum):
    """What an execution event *is*."""

    USER_MSG = "user_msg"
    AGENT_MSG = "agent_msg"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RETRIEVAL = "retrieval"
    PLANNING = "planning"
    REFLECTION = "reflection"
    HANDOFF = "handoff"  # multi-agent control transfer
    SYSTEM = "system"


class Phase(StrEnum):
    """Where in the agent lifecycle an event sits. Coarse by design.

    Used only for phase-attribution: we report which phase the anomaly signal
    peaked in, never "span N caused the failure" (ADR-0006).
    """

    INTAKE = "intake"
    PLANNING = "planning"
    RETRIEVAL = "retrieval"
    ACTION = "action"
    VERIFICATION = "verification"
    RESPONSE = "response"


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A tool invocation. The *signature* (name + arg types) is what detectors
    consume; literal arg values are kept for the judge/human but are not the
    primary structural substrate (ADR-0003)."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    # Optional declared schema: arg name -> type label ("INT", "MONEY", "STR", ...).
    arg_schema: dict[str, str] | None = None
    # Who emitted this call. In dual-control settings (tau2-bench) the simulated
    # USER also has tools — attribution matters for process detectors (ADR-0009).
    requestor: Role | None = None


class ToolResult(BaseModel):
    name: str
    ok: bool = True
    error: str | None = None
    value: Any | None = None
    latency_ms: float | None = None


class Span(BaseModel):
    """One ordered execution event."""

    span_id: str
    index: int  # position in the trace, 0-based
    kind: SpanKind
    phase: Phase = Phase.ACTION
    parent_id: str | None = None
    role: Role | None = None
    content: str | None = None  # raw prose — for judge/display, NOT primary for rules

    # Model-step grouping key: identical for all spans emitted by ONE LLM decode
    # (one assistant message). THE load-bearing field for parallel-tool-call
    # detection — without it, batched calls are indistinguishable from
    # sequentially-logged ones (ADR-0008). Populated faithfully by the tau2
    # adapter (AssistantMessage.tool_calls is a list). None when unknown.
    step_id: str | None = None

    tool: ToolCall | None = None
    tool_result: ToolResult | None = None

    tokens: int | None = None
    latency_ms: float | None = None
    cost: float | None = None
    model: str | None = None

    meta: dict[str, Any] = Field(default_factory=dict)


class Check(BaseModel):
    """A programmatic ground-truth assertion about the run (e.g., 'refund_issued')."""

    name: str
    passed: bool
    detail: str | None = None


class OutcomeStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class Outcome(BaseModel):
    status: OutcomeStatus = OutcomeStatus.UNKNOWN
    checks: list[Check] = Field(default_factory=list)
    final_response: str | None = None


class Label(BaseModel):
    """Ground truth — present in datasets only, not in production traces."""

    passed: bool
    failure_type: str | None = None  # see detection-strategy.md taxonomy
    first_error_span_id: str | None = None  # best-effort; research says this is hard
    notes: str | None = None
    annotator: str | None = None


class ToolInfo(BaseModel):
    """Per-tool process metadata, derived once per domain from the policy + tool
    schema. This is how the mutating/terminal/auth partition and read-before-mutate
    evidence sets enter declaratively (SABER's mutating/non-mutating partition;
    Near-Miss read-equivalence sets) — detectors never hardcode tool names."""

    mutating: bool = False  # changes external state (book/cancel/refund/...)
    terminal: bool = False  # ends the interaction (e.g. transfer_to_human_agents)
    auth: bool = False  # establishes/verifies identity
    # Read-only tools that constitute sufficient "evidence" before this tool may
    # be called (Near-Miss read-before-mutate). Any one occurring earlier suffices.
    read_set: list[str] = Field(default_factory=list)


class ReferenceAction(BaseModel):
    """One step of a benchmark-authored reference trajectory (τ² evaluation_criteria.actions):
    the tool call a correct agent is expected to make. Used by the Layer-2 deviation oracle
    (ADR-0008) to label process violations set-wise — overshoot (a mutation not in the
    reference) / missing (a reference mutation the agent skipped) — independently of the
    outcome oracle. This is benchmark ground truth, NOT our own annotation."""

    name: str
    requestor: Role = Role.ASSISTANT
    arguments: dict[str, Any] = Field(default_factory=dict)


class GuardType(StrEnum):
    """The four R10 checkable-clause buckets (the finite, closed set)."""

    VALUE_CONDITIONAL = "value_conditional"  # status/attr-gated, enum-allowed, threshold, tiered
    NUMERIC_CAP = "numeric_cap"  # <=N passengers / payment-instrument / per-object
    TIME_WINDOW = "time_window"  # timestamp arithmetic (cancel within 24h of booking)
    CROSS_OBJECT_JOIN = "cross_object_join"  # quantify a predicate over a collection / attr-exists


class JoinKind(StrEnum):
    """The cross_object_join variants, made explicit so the resolver dispatches cleanly."""

    SAME_ATTR_ACROSS_COLLECTION = "same_attr_across_collection"  # every elem.attr OP a reference
    ATTRIBUTE_EXISTS_IN_RELATED = "attribute_exists_in_related"  # collection ⊆ related's collection
    NO_SEGMENT_FLOWN = "no_segment_flown"  # none(elem.attr in disallowed-set)


class Guard(BaseModel):
    """One deterministically-checkable policy clause — the R11 ``typed_check`` tier.

    A guard FIRES (emits an over_action Signal) ONLY when its typed predicate is decidably
    False on operands that ALL resolved from the trace. If ANY operand it needs is unresolvable
    (structurally absent or ambiguous), the guard ABSTAINS — it emits a loud, recorded,
    NON-firing abstain Signal (confidence=0.0), never a guessed pass and never a guessed fail.
    That is the R7 over-bind defense made structural.

    The operand grammar (``arg:``/``result:``/``count:`` + the new ``ts:``/``each:`` kinds) and
    the ``op`` keys are the SAME family ``InvariantViolationDetector`` already evaluates; this
    model is the typed, provenance-carrying wrapper around them.
    """

    name: str
    guard_type: GuardType
    applies_to: str | None = None  # the mutating tool this guard gates (triage only)

    # --- ADR-0015 Increment-1: record-scoping trigger -------------------------------------------
    # "on:<tool>" names the mutating tool call whose TOUCHED record this guard checks. When set,
    # the interpreter (i) only evaluates if that tool call is present (else no occasion -> []), and
    # (ii) binds EVERY operand to spans carrying the triggering call's record key — not the first
    # tool/result by name across the whole trace. trigger=None preserves the legacy whole-trace
    # first-match semantics byte-identically (every existing committed guard keeps working). It is
    # orthogonal to the 4 GuardType buckets (the cross-cutting axis of the A7/A30/A50/A58 shapes).
    trigger: str | None = None

    # --- comparison core (numeric_cap, value_conditional threshold/tiered, time_window) ---
    # left/right are operand refs (str, e.g. "count:book_reservation.passengers",
    # "result:get_flight_status.status", "ts:get_reservation_details.created_at") OR literals.
    left: Any | None = None
    op: str | None = None  # key into InvariantViolationDetector._OPS (reused verbatim)
    right: Any | None = None

    # --- value_conditional enum form ---
    subject: Any | None = None  # operand ref whose resolved value must be in/notin the set
    allowed_values: list[Any] | None = None  # the enumerated allow-set (e.g. ["available"])

    # --- value_conditional tiered-lookup form (e.g. baggage allowance matrix) ---
    tier_table: dict[str, dict[str, Any]] | None = None  # {"regular": {"basic_economy": 0, ...}}
    # operand refs whose resolved values index the table [k1][k2]:
    tier_keys: list[Any] | None = None
    # After the table lookup yields a bound, compare it against this operand using ``op``.
    # tier_compare_to is the operand ref for the actual value (e.g. count:...added_bags):
    tier_compare_to: Any | None = None

    # --- cross_object_join form ---
    join: JoinKind | None = None
    collection: Any | None = None  # ``each:tool.field`` operand ref -> a list
    element_attr: str | None = None  # subfield read off each element
    related: Any | None = None  # ATTRIBUTE_EXISTS_IN_RELATED: the related list operand ref
    disallowed: list[Any] | None = None  # NO_SEGMENT_FLOWN: the disallowed status set

    # --- firing metadata (mirrors the rules.py Signal pattern) ---
    failure_type: str = "over_action"
    # NOTE: severity is a plain float, NOT a Severity enum. base.Severity is a class of float
    # constants (Severity.CRITICAL == 1.0) and schema.py MUST NOT import argus_detectors
    # (no framework/network imports here). 1.0 maps to Severity.CRITICAL.
    severity: float = 1.0
    confidence: float = 0.85

    # --- R11 provenance (MANDATORY: the no-guess criterion is enforced structurally) ---
    policy_ref: str | None = None  # verbatim policy sentence this compiled from
    source_policy_name: str | None = None  # the ToolGuard policy_name it came from

    # --- abstain behavior (R7). Default ON; the compiled tau2/ToolGuard guards never opt out. ---
    abstain_on_missing: bool = True

    @field_validator("policy_ref")
    @classmethod
    def _policy_ref_required(cls, v: str | None) -> str | None:
        # A Guard with no quoted policy span is rejected at load: this enforces the falsifiable
        # criterion "abstains, does not guess caps/windows" at COMPILE/LOAD time, not check time.
        if v is None or not v.strip():
            raise ValueError("Guard.policy_ref is mandatory (verbatim policy span) — no-guess rule")
        return v

    @model_validator(mode="after")
    def _require_fields_per_type(self) -> Guard:
        gt = self.guard_type
        if gt == GuardType.NUMERIC_CAP:
            if self.left is None or self.op is None or self.right is None:
                raise ValueError("numeric_cap requires left/op/right")
        elif gt == GuardType.VALUE_CONDITIONAL:
            enum_form = self.subject is not None and self.allowed_values is not None
            tier_form = (
                self.tier_table is not None
                and self.tier_keys is not None
                and self.tier_compare_to is not None
                and self.op is not None
            )
            thresh_form = self.left is not None and self.op is not None and self.right is not None
            if not (enum_form or tier_form or thresh_form):
                raise ValueError(
                    "value_conditional requires one of: (subject+allowed_values), "
                    "(tier_table+tier_keys+tier_compare_to+op), or (left+op+right)"
                )
        elif gt == GuardType.TIME_WINDOW:
            if self.left is None or self.op is None or self.right is None:
                raise ValueError("time_window requires left/op/right (left a ts: delta in secs)")
        elif gt == GuardType.CROSS_OBJECT_JOIN:
            if self.join is None or self.collection is None:
                raise ValueError("cross_object_join requires join + collection")
            # SAME_ATTR mirrors the numeric_cap left/op/right discipline: an omitted reference
            # operand must be rejected at load, not resolved as a None literal and fired (the R7
            # over-bind the join path otherwise re-introduces).
            if self.join == JoinKind.SAME_ATTR_ACROSS_COLLECTION and (
                self.right is None or self.op is None or self.element_attr is None
            ):
                raise ValueError("same_attr_across_collection requires right + op + element_attr")
            if self.join == JoinKind.ATTRIBUTE_EXISTS_IN_RELATED and self.related is None:
                raise ValueError("attribute_exists_in_related requires `related`")
            if self.join == JoinKind.NO_SEGMENT_FLOWN and (
                not self.disallowed or self.element_attr is None
            ):
                raise ValueError("no_segment_flown requires `disallowed` + element_attr")
        return self


class TaskSpec(BaseModel):
    """Declared expectations for a task. Domain knowledge enters here, not via
    hardcoding in detectors. Optional — detectors degrade gracefully without it."""

    goal: str | None = None
    allowed_tools: list[str] | None = None
    required_steps: list[str] | None = None  # tool names that must occur, in order if listed
    # Benchmark-authored reference trajectory (τ² evaluation_criteria.actions). Powers the
    # Layer-2 reference-deviation oracle — process ground truth from the benchmark's authors,
    # never from the outcome oracle (ADR-0008). None when the task ships no reference.
    reference_actions: list[ReferenceAction] | None = None
    # Invariants as small declarative rules the rule-tier can evaluate. LEGACY free-form alias:
    # subsumed by the typed numeric_cap/value_conditional guards below, but kept working — the
    # rule tier still reads it (byte-for-byte) so existing call sites are untouched.
    # e.g. {"name": "refund_le_order_value", "expr": "refund_amount <= order_value"}
    invariants: list[dict[str, Any]] | None = None
    # Typed, deterministically-checkable policy clauses (R11 typed_check tier). The closed set
    # of R10 buckets. Each guard abstains loudly (no firing signal, recorded) when its operand is
    # unresolvable from the trace — never guesses. Read by the rule tier alongside `invariants`.
    guards: list[Guard] | None = None
    success_criteria: list[str] | None = None

    # Per-tool process registry (tool name -> ToolInfo). Powers the
    # read-before-mutate, authenticate-before-act, and post-handoff detectors
    # without hardcoding domain tool names (ADR-0008).
    tool_registry: dict[str, ToolInfo] | None = None
    # Authoritative NL policy text (tau policy.md / wiki.md) for judge-tier
    # grounding and offline guard generation. The policy document IS the spec.
    policy_text: str | None = None

    # A verbatim message the agent must send after invoking a terminal/handoff tool
    # (tau retail+airline: "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.").
    # Powers the required-utterance detector declaratively (the policy string IS the spec).
    required_utterance_after_terminal: str | None = None

    # Whether the policy requires explicit user confirmation before EACH state-mutating action
    # (tau retail+airline: "list the action details and obtain explicit user confirmation").
    # Opt-in: the confirm-before-mutate detector abstains unless a domain's policy sets this,
    # so domains without a confirmation rule (e.g. cs-agent-v0) are not spuriously flagged.
    confirm_before_mutate_required: bool = False

    # Whether the policy forbids emitting more than one tool call per model step
    # (tau retail+airline: "You should make at most one tool call at a time"). Opt-in: the
    # parallel-tool-call detector abstains unless a domain's policy sets this — many agent
    # systems *encourage* parallel calls, so firing unconditionally would be a false positive
    # there. The policy claim lives here, not hardcoded in the detector (ADR-0009).
    parallel_calls_forbidden: bool = False

    # --- convenience accessors ---

    def mutating_tools(self) -> set[str]:
        if not self.tool_registry:
            return set()
        return {name for name, info in self.tool_registry.items() if info.mutating}

    def tool_info(self, name: str) -> ToolInfo | None:
        return self.tool_registry.get(name) if self.tool_registry else None

    def all_guards(self) -> list[Guard]:
        return list(self.guards or [])


class Trace(BaseModel):
    """A complete agent execution."""

    trace_id: str
    agent_id: str | None = None
    task_type: str | None = None
    schema_version: int = SCHEMA_VERSION

    spec: TaskSpec | None = None
    spans: list[Span] = Field(default_factory=list)
    outcome: Outcome | None = None
    label: Label | None = None

    meta: dict[str, Any] = Field(default_factory=dict)

    # --- convenience accessors used across detectors ---

    def tool_calls(self) -> list[Span]:
        return [s for s in self.spans if s.kind == SpanKind.TOOL_CALL]

    def tool_results(self) -> list[Span]:
        return [s for s in self.spans if s.kind == SpanKind.TOOL_RESULT]

    def span_by_id(self, span_id: str) -> Span | None:
        return next((s for s in self.spans if s.span_id == span_id), None)

    def ordered(self) -> list[Span]:
        return sorted(self.spans, key=lambda s: s.index)

    def step_groups(self) -> dict[str, list[Span]]:
        """Group spans by model-step (step_id), preserving order. Spans with no
        step_id are skipped. Used by parallel-tool-call / mixed-turn detectors."""
        groups: dict[str, list[Span]] = {}
        for s in self.ordered():
            if s.step_id is not None:
                groups.setdefault(s.step_id, []).append(s)
        return groups
