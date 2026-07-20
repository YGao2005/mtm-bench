"""Adapter: AgentProcessBench (APB) trajectory → ``argus_core.Trace``.

APB (arXiv 2603.14465, MIT) ships 1,000 human-labeled agent trajectories in the **OpenAI
chat-message** format (``system``/``user``/``assistant``-with-``tool_calls``/``tool``), one
per record, with per-assistant-message human gold labels. This is a DIFFERENT input shape
from the τ²-bench ``SimulationRun`` the ``tau2`` adapter consumes (that one is Sierra's own
data model), so this is a net-new adapter — but it mirrors how ``tau2.messages_to_spans``
builds spans/``step_id``/the tool registry, so the SAME deterministic detectors run unchanged.

Built for the strata-1+3 meta-eval leaderboard (metaeval-difficulty-prereg.md §2/§7,
EXP-0033): Argus's deterministic detectors enter as ONE step-verifier entry alongside APB's
20 released LLMs. Only the τ² slice (``data_source`` in {tau2_airline, tau2_retail,
tau2_telecom}) carries a policy/registry; the other 3 APB datasets (hotpotqa/gaia/bfcl) have
no Argus tool registry and are out of scope for this adapter.

Verified against the released ``data/AgentProcessBench/tau2.jsonl``:

- An APB record's ``messages`` is OpenAI chat. An ``assistant`` message's ``tool_calls`` is a
  list in OpenAI shape: ``{"id", "type": "function", "function": {"name", "arguments"}}`` where
  ``arguments`` is a **JSON string** (parsed here). >1 call in one assistant message == parallel
  calls in one decode → all those spans get the SAME ``step_id`` so
  ``ParallelToolCallDetector`` fires (mirrors the τ² adapter's contract).
- Unlike τ², an APB assistant message MAY carry BOTH ``content`` and ``tool_calls`` (345/~?
  in tau2.jsonl). We keep the content on a leading AGENT_MSG span and still explode the calls.
- A ``tool`` message carries its tool ``name`` directly (plus a ``tool_call_id``); error is
  signaled by the ``content`` string starting with ``"Error"`` (no boolean field) — so
  ``ToolResult.ok = not content.startswith("Error")``.
- **APB ``step_labels`` are keyed by the assistant message's index in the ``messages`` array**
  (one label per ``role="assistant"`` message — guide §0), even when that message emits several
  tool_calls. We set each span's ``step_id`` to that message index (str) AND record a
  ``span_id -> assistant-message-index`` map in ``Trace.meta["apb_msg_index"]`` so the WHERE
  projection (prereg §7) can map an Argus per-call firing UP to its enclosing assistant message.

ANTI-ORACLE-COPYING (V5 / drift-trap #1): an APB record also carries ``ground_truth`` (the answer
key). We deliberately DO NOT feed it as ``TaskSpec.reference_actions`` — doing so would let
``UnexpectedMutationDetector`` read the answer key (oracle-copying). With ``reference_actions=None``
that detector abstains, so Argus's firings stay purely process/policy-structural.
"""

from __future__ import annotations

import json
from typing import Any

from .schema import (
    Outcome,
    OutcomeStatus,
    Phase,
    Role,
    Span,
    SpanKind,
    TaskSpec,
    ToolCall,
    ToolResult,
    Trace,
)

from .tau2_loader import task_spec_for_domain

# APB ``data_source`` → our τ² domain key (selects the registry/policy). The 3 non-τ² APB
# datasets have no Argus tool registry, so this adapter only handles the τ² slice.
APB_DATA_SOURCE_TO_DOMAIN: dict[str, str] = {
    "tau2_airline": "airline",
    "tau2_retail": "retail",
    "tau2_telecom": "telecom",  # no policy/registry shipped — spec-less (detectors abstain)
}

_ROLE_TO_KIND = {
    "system": (SpanKind.SYSTEM, Phase.INTAKE, Role.SYSTEM),
    "user": (SpanKind.USER_MSG, Phase.INTAKE, Role.USER),
    "assistant": (SpanKind.AGENT_MSG, Phase.RESPONSE, Role.ASSISTANT),
}


def _tool_call_name(tc: dict[str, Any]) -> str:
    """OpenAI tool_call → tool name (``function.name``; fall back to a flat ``name``)."""
    fn = tc.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return str(tc.get("name", "unknown"))


def _tool_call_args(tc: dict[str, Any]) -> dict[str, Any]:
    """OpenAI tool_call arguments (a JSON **string** under ``function.arguments``) → dict.
    Tolerates an already-parsed dict or unparseable junk (→ empty dict)."""
    fn = tc.get("function")
    raw = fn.get("arguments") if isinstance(fn, dict) else tc.get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def messages_to_spans(
    messages: list[dict[str, Any]],
) -> tuple[list[Span], dict[str, int]]:
    """Walk APB OpenAI chat messages → ordered Argus spans.

    Returns ``(spans, span_id_to_msg_index)`` where the second item maps every span's id to the
    index of its enclosing message in the original ``messages`` array — the alignment the WHERE
    projection needs (APB labels are keyed by that index).

    Mapping: system/user/assistant content → SYSTEM/USER_MSG/AGENT_MSG; an assistant message's
    ``tool_calls`` → one TOOL_CALL span each, all sharing ``step_id = str(message_index)`` (one
    decode); a ``tool`` message → TOOL_RESULT (``ok = not content.startswith('Error')``). A mixed
    assistant message (content AND tool_calls) emits a leading AGENT_MSG span then the calls.
    """
    spans: list[Span] = []
    span_to_msg: dict[str, int] = {}
    idx = 0

    def _emit(span: Span, msg_i: int) -> None:
        nonlocal idx
        spans.append(span)
        span_to_msg[span.span_id] = msg_i
        idx += 1

    for i, m in enumerate(messages):
        role = m.get("role")

        if role == "tool":
            content = m.get("content")
            ctext = content if isinstance(content, str) else str(content or "")
            _emit(
                Span(
                    span_id=f"r_{i}",
                    index=idx,
                    kind=SpanKind.TOOL_RESULT,
                    phase=Phase.ACTION,
                    role=Role.TOOL,
                    tool_result=ToolResult(
                        name=str(m.get("name", "unknown")),
                        ok=not ctext.startswith("Error"),
                        error=ctext if ctext.startswith("Error") else None,
                        value=content,
                    ),
                ),
                i,
            )
            continue

        calls = m.get("tool_calls") or []
        content = m.get("content")

        # A mixed assistant decode (content AND tool_calls) keeps a leading content span; the
        # step_id ties it to the same model step as its calls (so the WHERE projection sees the
        # whole assistant message as one unit).
        if content and (role != "tool"):
            kind, phase, span_role = _ROLE_TO_KIND.get(
                role or "", (SpanKind.AGENT_MSG, Phase.RESPONSE, Role.ASSISTANT)
            )
            _emit(
                Span(
                    span_id=f"m{i}",
                    index=idx,
                    kind=kind,
                    phase=phase,
                    role=span_role,
                    content=content,
                    step_id=str(i) if role in ("assistant", "user") else None,
                ),
                i,
            )

        # Tool calls: one TOOL_CALL span each, all sharing step_id == this message index.
        for j, tc in enumerate(calls):
            call_id = tc.get("id") or f"m{i}_c{j}"
            _emit(
                Span(
                    span_id=f"{call_id}",
                    index=idx,
                    kind=SpanKind.TOOL_CALL,
                    phase=Phase.ACTION,
                    role=Role.ASSISTANT,
                    step_id=str(i),
                    tool=ToolCall(
                        name=_tool_call_name(tc),
                        args=_tool_call_args(tc),
                        requestor=Role.ASSISTANT,
                    ),
                ),
                i,
            )

        # A bare message with neither content nor calls (rare) still needs a placeholder span so
        # the trace isn't silently dropping a labeled assistant message.
        if not content and not calls:
            kind, phase, span_role = _ROLE_TO_KIND.get(
                role or "", (SpanKind.AGENT_MSG, Phase.RESPONSE, Role.ASSISTANT)
            )
            _emit(
                Span(
                    span_id=f"m{i}",
                    index=idx,
                    kind=kind,
                    phase=phase,
                    role=span_role,
                    content=None,
                    step_id=str(i) if role in ("assistant", "user") else None,
                ),
                i,
            )

    return spans, span_to_msg


# Domains the APB path scores SPEC-LESS regardless of whether the τ² adapter now has a registry.
# telecom gained a policy-derived registry for the generality-lane R20 probe (EXP-0081, which reads
# raw τ² RESULT files), but the APB lane's contract + leaderboard treat APB-telecom as spec-less;
# keep that path byte-stable here (the registry is reached via task_spec_for_domain directly, not
# through this APB front door).
_APB_SPEC_LESS_DOMAINS = frozenset({"telecom"})


def apb_task_spec(domain: str, policy_text: str | None = None) -> TaskSpec | None:
    """Build the TaskSpec for an APB τ² record.

    Reuses the τ² adapter's registry/invariants/flags (the policy IS the spec).
    ``reference_actions`` is deliberately omitted (None) — APB's ``ground_truth`` is the answer
    key and feeding it would
    make ``UnexpectedMutationDetector`` an oracle-copier (V5 / drift-trap #1). Returns ``None`` for
    a domain the APB lane scores spec-less (telecom), so the caller scores it as spec-less (all
    process detectors abstain — a coverage gap, not a pass).
    """
    if domain in _APB_SPEC_LESS_DOMAINS:
        return None
    try:
        return task_spec_for_domain(domain, policy_text=policy_text, reference_actions=None)
    except ValueError:
        return None  # unknown/registry-less domain


def load_apb_record(
    record: dict[str, Any],
    *,
    policy_by_domain: dict[str, str] | None = None,
) -> Trace:
    """Convert one APB record (``messages`` + ``tools`` + ``data_source``) into a ``Trace``.

    ``policy_by_domain`` maps a τ² domain ("airline"/"retail") → its policy text (read from
    ``datasets/tau2/policy/<domain>.md``); attached to the spec for judge-tier grounding only —
    the deterministic detectors don't read it. The trace's ``meta`` carries:
      - ``apb_msg_index``: ``{span_id: enclosing-message-index}`` (the WHERE-projection key);
      - ``data_source``/``domain``/``apb_total_index`` for provenance.

    We do NOT set ``Trace.outcome`` from APB's ``final_label`` (that's the human gold the
    leaderboard scores AGAINST — keeping it off the trace avoids any oracle leak into detectors).
    """
    messages = record.get("messages") or []
    spans, span_to_msg = messages_to_spans(messages)

    data_source = str(record.get("data_source", ""))
    domain = APB_DATA_SOURCE_TO_DOMAIN.get(data_source, data_source)
    policy_text = (policy_by_domain or {}).get(domain)
    spec = apb_task_spec(domain, policy_text=policy_text)

    total_index = record.get("total_index")
    trace_id = f"apb:{data_source}:{total_index}"

    return Trace(
        trace_id=trace_id,
        agent_id=domain,
        task_type=domain,
        spec=spec,
        spans=spans,
        # No outcome from APB gold — the leaderboard holds the gold separately (no detector leak).
        outcome=Outcome(status=OutcomeStatus.UNKNOWN, final_response=None),
        label=None,
        meta={
            "source": "agentprocessbench",
            "data_source": data_source,
            "domain": domain,
            "apb_total_index": total_index,
            "apb_msg_index": span_to_msg,
            "spec_less": spec is None,
            # APB/τ² end conversations with a sentinel user turn (###STOP###/###TRANSFER###,
            # 235/250 in tau2.jsonl) — the simulator closing the call, NOT "the agent never
            # responded." Flag it so PrematureTerminationDetector doesn't misfire on the
            # transcript-format artifact (same contract the τ² adapter uses for user_stop).
            "terminated_cleanly": True,
        },
    )


__all__ = [
    "APB_DATA_SOURCE_TO_DOMAIN",
    "apb_task_spec",
    "load_apb_record",
    "messages_to_spans",
]
