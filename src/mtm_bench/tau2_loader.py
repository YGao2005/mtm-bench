"""Adapter: τ²-bench (Sierra) simulation results → ``mtm_bench Trace``.

τ²-bench retail + airline is our first external, in-domain benchmark (multi-turn
customer-support + tool-calling, with real policy documents). This module turns a
saved τ² ``Results`` JSON (one file per model/domain run) into Argus traces so the
deterministic ensemble runs on real data. See ADR-0009.

Everything here was verified against the public source (github.com/sierra-research/
tau2-bench, MIT) — message model ``src/tau2/data_model/message.py``, results model
``simulation.py``, tool decorations ``domains/<d>/tools.py``, and the committed
``data/tau2/results/final`` runs. Specifically:

- One ``AssistantMessage`` is ONE model decode; its ``tool_calls`` is a *list*. So
  >1 call in a single message == parallel calls in one decode → all those spans get
  the SAME ``step_id`` and ``process.parallel_tool_calls`` fires (ADR-0009). Real
  airline/retail runs are ~20% multi-call messages despite the "one tool call at a
  time" policy — these are exactly the latent failures Argus exists to catch.
- ``ToolMessage`` carries only the call ``id`` (no tool name); we join result→call
  by id to recover the name that ``verify_before_mutate`` consumes.
- The outcome oracle is ``reward_info.reward`` (binary {0.0, 1.0} on these domains)
  → ``Trace.outcome`` (Layer-0). We do NOT set ``Trace.label`` from it: process
  detectors are never graded against the outcome oracle (ADR-0008).
- ``mutating`` / ``terminal`` / ``auth`` are derived from the ``@is_tool`` decorators
  (``ToolType.WRITE`` ⇒ mutating; ``transfer_to_human_agents`` is the terminal
  transfer; retail ``find_user_id_*`` establish identity) — declarative, never
  hardcoded inside a detector (ADR-0008).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import (
    Outcome,
    OutcomeStatus,
    Phase,
    ReferenceAction,
    Role,
    Span,
    SpanKind,
    TaskSpec,
    ToolCall,
    ToolInfo,
    ToolResult,
    Trace,
)

# --- Per-domain tool registry (compiled once from the verified @is_tool decorators).
# mutating == ToolType.WRITE; terminal == the transfer-to-human handoff; auth == the
# identity-establishing reads. read_set == the lookup the policy requires before a
# state mutation (Near-Miss read-before-mutate); left empty where no single read is
# unambiguously required (the detector then abstains — a coverage gap, not a pass).
_AIRLINE_REGISTRY: dict[str, ToolInfo] = {
    "search_direct_flight": ToolInfo(),
    "search_onestop_flight": ToolInfo(),
    "list_all_airports": ToolInfo(),
    "get_flight_status": ToolInfo(),
    "get_reservation_details": ToolInfo(),
    "get_user_details": ToolInfo(),
    "calculate": ToolInfo(),
    "book_reservation": ToolInfo(mutating=True),  # creates new state — no prior read required
    "cancel_reservation": ToolInfo(mutating=True, read_set=["get_reservation_details"]),
    "update_reservation_baggages": ToolInfo(mutating=True, read_set=["get_reservation_details"]),
    "update_reservation_flights": ToolInfo(mutating=True, read_set=["get_reservation_details"]),
    "update_reservation_passengers": ToolInfo(mutating=True, read_set=["get_reservation_details"]),
    "send_certificate": ToolInfo(mutating=True),  # compensation; required reads are not single
    # GENERIC + mutates_state=False in tau2; the policy's terminal handoff.
    "transfer_to_human_agents": ToolInfo(terminal=True),
}

_RETAIL_REGISTRY: dict[str, ToolInfo] = {
    # Policy: authenticate via email OR name+zip before anything else.
    "find_user_id_by_email": ToolInfo(auth=True),
    "find_user_id_by_name_zip": ToolInfo(auth=True),
    "get_user_details": ToolInfo(),
    "get_order_details": ToolInfo(),
    "get_product_details": ToolInfo(),
    "get_item_details": ToolInfo(),
    "list_all_product_types": ToolInfo(),
    "calculate": ToolInfo(),
    "cancel_pending_order": ToolInfo(mutating=True, read_set=["get_order_details"]),
    "exchange_delivered_order_items": ToolInfo(mutating=True, read_set=["get_order_details"]),
    "modify_pending_order_address": ToolInfo(mutating=True, read_set=["get_order_details"]),
    "modify_pending_order_items": ToolInfo(mutating=True, read_set=["get_order_details"]),
    "modify_pending_order_payment": ToolInfo(mutating=True, read_set=["get_order_details"]),
    "return_delivered_order_items": ToolInfo(mutating=True, read_set=["get_order_details"]),
    "modify_user_address": ToolInfo(mutating=True),  # profile field; policy mandates no prior read
    "transfer_to_human_agents": ToolInfo(terminal=True),
}

# Telecom registry — added for the R20 cross-domain delta probe (EXP-0081). mutating == the
# @is_tool(ToolType.WRITE) tools (suspend/resume/send_payment_request/refuel/en|disable_roaming);
# read_set is assigned ONLY where main_policy.md unambiguously mandates a single prior read,
# leaving the rest empty so the detector ABSTAINS (a coverage gap, not a pass) — same discipline
# as airline/retail above. Derived from policy prose + the @is_tool decorators, never the oracle /
# reference trajectory (ADR-0008 / drift-trap #1).
_TELECOM_REGISTRY: dict[str, ToolInfo] = {
    # reads / lookups
    "get_customer_by_phone": ToolInfo(auth=True),  # identity lookup
    "get_customer_by_id": ToolInfo(),
    "get_customer_by_name": ToolInfo(auth=True),
    "get_details_by_id": ToolInfo(),  # polymorphic line/bill getter
    "get_bills_for_customer": ToolInfo(),
    "get_data_usage": ToolInfo(),
    # mutations
    # policy line 117: "always check that the bill is overdue before sending a payment request".
    # read_set = the TYPED bill getter only (same single-typed-read discipline as airline/retail).
    # The bill status is ALSO readable via the polymorphic get_details_by_id(bill_id), but that
    # tool is polymorphic (returns a line for L*, a bill for B*, …), so a NAME-based read_set
    # cannot tell a bill-read from a line-read — putting it here masks real "never-checked-the-bill"
    # bypasses (EXP-0081 audit: 49/90 o4-mini payment traces read only a LINE via get_details_by_id
    # yet passed). The polymorphic-getter LEGIT case (agent genuinely read the bill via
    # get_details_by_id) is handled by the probe's VALUE-aware correction, not by widening read_set.
    "send_payment_request": ToolInfo(mutating=True, read_set=["get_bills_for_customer"]),
    # policy line 125: lift suspension only AFTER overdue bills are paid → gated on bill status.
    "resume_line": ToolInfo(mutating=True, read_set=["get_bills_for_customer"]),
    # no single unambiguous required prior read in policy → empty read_set (detector abstains).
    "suspend_line": ToolInfo(mutating=True),
    "refuel_data": ToolInfo(mutating=True),
    "enable_roaming": ToolInfo(mutating=True),
    "disable_roaming": ToolInfo(mutating=True),
    "transfer_to_human_agents": ToolInfo(terminal=True),
}

TOOL_REGISTRIES: dict[str, dict[str, ToolInfo]] = {
    "airline": _AIRLINE_REGISTRY,
    "retail": _RETAIL_REGISTRY,
    "telecom": _TELECOM_REGISTRY,
}

# τ² TerminationReason values that mean the conversation ended normally (the user
# simulator chose to stop / transfer). A trailing user message under these is the
# user closing the call, NOT "the agent never responded" — see premature-termination
# handling in argus_detectors.rules. Anything else (max_steps, *_error, timeout,
# context_window_exceeded) is an abnormal/premature end.
_CLEAN_TERMINATIONS = frozenset({"user_stop", "agent_stop"})

_ROLE_TO_KIND = {
    "system": (SpanKind.SYSTEM, Phase.INTAKE, Role.SYSTEM),
    "user": (SpanKind.USER_MSG, Phase.INTAKE, Role.USER),
    "assistant": (SpanKind.AGENT_MSG, Phase.RESPONSE, Role.ASSISTANT),
}

# Verbatim policy utterance both domains require after a transfer_to_human_agents call.
_TRANSFER_UTTERANCE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."

# Declarative cap invariants compiled from the policy prose (the COUNT grammar in
# argus_detectors.rules). Airline book_reservation: "at most five passengers"; payment "at most
# one travel certificate, at most one credit card, and at most three gift cards." Retail caps are
# value/balance constraints (not pure counts), so none are expressed here.
_DOMAIN_INVARIANTS: dict[str, list[dict[str, Any]]] = {
    "airline": [
        {"name": "max_5_passengers", "left": "count:book_reservation.passengers",
         "op": "<=", "right": 5},
        {"name": "max_3_gift_cards", "left": "count:book_reservation.payment_methods[gift_card]",
         "op": "<=", "right": 3},
        {"name": "max_1_credit_card",
         "left": "count:book_reservation.payment_methods[credit_card]", "op": "<=", "right": 1},
        {"name": "max_1_certificate",
         "left": "count:book_reservation.payment_methods[certificate]", "op": "<=", "right": 1},
    ],
    "retail": [],
    # policy line 134: "The maximum amount of data that can be refueled is 2GB." (numeric cap)
    "telecom": [
        {"name": "max_2gb_refuel", "left": "arg:refuel_data.gb_amount", "op": "<=", "right": 2},
    ],
}


def _parse_reference_actions(task: dict[str, Any]) -> list[ReferenceAction] | None:
    """Pull the benchmark-authored reference trajectory from a τ² task's
    ``evaluation_criteria.actions`` (ordered tool calls a correct agent should make). Returns
    None when the task ships no actions (e.g. refuse-to-act tasks) — the Layer-2 oracle then
    abstains rather than treating 'no reference' as 'no required actions'."""
    actions = ((task.get("evaluation_criteria") or {}).get("actions")) or []
    if not actions:
        return None
    parsed: list[ReferenceAction] = []
    for a in actions:
        raw = a.get("requestor", "assistant")
        requestor = Role(raw) if raw in (r.value for r in Role) else Role.ASSISTANT
        parsed.append(
            ReferenceAction(
                name=a["name"],
                requestor=requestor,
                arguments=a.get("arguments") or {},
            )
        )
    return parsed


def task_spec_for_domain(
    domain: str,
    policy_text: str | None = None,
    reference_actions: list[ReferenceAction] | None = None,
) -> TaskSpec:
    """Build the TaskSpec for a τ² domain: the tool_registry (process metadata), the declarative
    cap invariants compiled from the policy, the required post-transfer utterance, optional policy
    text for judge-tier grounding, and the optional benchmark-authored ``reference_actions`` (for
    the Layer-2 deviation oracle, ADR-0008). We still do NOT set ``required_steps`` (the rule-tier
    ordering check stays off; the reference trajectory feeds the oracle, not a hardcoded order)."""
    registry = TOOL_REGISTRIES.get(domain)
    if registry is None:
        raise ValueError(f"Unknown τ² domain {domain!r}; known: {sorted(TOOL_REGISTRIES)}")
    invariants = _DOMAIN_INVARIANTS.get(domain) or None
    return TaskSpec(
        goal=f"τ²-bench {domain} customer-support task.",
        allowed_tools=sorted(registry),
        tool_registry=dict(registry),
        invariants=invariants,
        policy_text=policy_text,
        required_utterance_after_terminal=_TRANSFER_UTTERANCE,
        confirm_before_mutate_required=True,  # both τ² domains' policies require it
        parallel_calls_forbidden=True,  # both τ² domains: "at most one tool call at a time"
        reference_actions=reference_actions,
    )


def _explode_tool_calls(
    raw_calls: list[dict[str, Any]],
    *,
    step_id: str,
    start_index: int,
    span_prefix: str,
    call_name_by_id: dict[str, str],
) -> list[Span]:
    """Turn a message's ``tool_calls`` list into one TOOL_CALL span each, all sharing
    ``step_id`` (one model decode) so parallel calls are recoverable."""
    spans: list[Span] = []
    for j, tc in enumerate(raw_calls):
        call_id = tc.get("id") or f"{span_prefix}_c{j}"
        name = tc["name"]
        call_name_by_id[call_id] = name
        requestor = tc.get("requestor", "assistant")
        spans.append(
            Span(
                span_id=call_id,
                index=start_index + j,
                kind=SpanKind.TOOL_CALL,
                phase=Phase.ACTION,
                role=Role.ASSISTANT if requestor == "assistant" else Role.USER,
                step_id=step_id,
                tool=ToolCall(
                    name=name,
                    args=tc.get("arguments", {}) or {},
                    requestor=Role(requestor),
                ),
            )
        )
    return spans


def messages_to_spans(messages: list[dict[str, Any]]) -> list[Span]:
    """Walk τ² messages → ordered Argus spans.

    Mapping (verified against the τ² message model): system/user/assistant content →
    SYSTEM/USER_MSG/AGENT_MSG; an assistant or user message's ``tool_calls`` → one
    TOOL_CALL span each sharing one ``step_id``; a ``tool`` message → TOOL_RESULT
    (``ok = not error``), joined to its call by id to recover the tool name.
    """
    spans: list[Span] = []
    call_name_by_id: dict[str, str] = {}
    idx = 0

    for i, m in enumerate(messages):
        role = m.get("role")

        # Environment tool result(s). Saved half-duplex runs store flat ToolMessages;
        # a MultiToolMessage envelope (role == "tool" AND a tool_messages list) is
        # handled defensively — check it first, since it also has role "tool".
        tool_msgs: list[dict[str, Any]]
        if "tool_messages" in m:
            tool_msgs = m["tool_messages"]
        elif role == "tool":
            tool_msgs = [m]
        else:
            tool_msgs = []
        if tool_msgs:
            for tm in tool_msgs:
                call_id = tm["id"]
                spans.append(
                    Span(
                        span_id=f"r_{call_id}",
                        index=idx,
                        kind=SpanKind.TOOL_RESULT,
                        phase=Phase.ACTION,
                        role=Role.TOOL,
                        tool_result=ToolResult(
                            name=call_name_by_id.get(call_id, "unknown"),
                            ok=not tm.get("error", False),
                            error=tm.get("content") if tm.get("error") else None,
                            value=tm.get("content"),
                        ),
                    )
                )
                idx += 1
            continue

        calls = m.get("tool_calls")
        if calls:
            # An assistant/user decode that emitted tool calls (never mixed with content
            # in τ² — verified). All calls share one step_id == this decode.
            new_spans = _explode_tool_calls(
                calls,
                step_id=f"m{i}",
                start_index=idx,
                span_prefix=f"m{i}",
                call_name_by_id=call_name_by_id,
            )
            spans.extend(new_spans)
            idx += len(new_spans)
            continue

        # Plain content message (system / user / assistant).
        kind, phase, span_role = _ROLE_TO_KIND.get(
            role or "", (SpanKind.AGENT_MSG, Phase.RESPONSE, Role.ASSISTANT)
        )
        spans.append(
            Span(
                span_id=f"m{i}",
                index=idx,
                kind=kind,
                phase=phase,
                role=span_role,
                content=m.get("content"),
                step_id=f"m{i}" if role in ("assistant", "user") else None,
            )
        )
        idx += 1

    return spans


def simulation_to_trace(
    simulation: dict[str, Any],
    task: dict[str, Any],
    domain: str,
    *,
    policy_text: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> Trace:
    """Convert one τ² ``SimulationRun`` (+ its ``Task``) into a ``Trace``."""
    spans = messages_to_spans(simulation.get("messages") or [])

    reward_info = simulation.get("reward_info") or {}
    reward = reward_info.get("reward")
    if reward is None:
        status = OutcomeStatus.UNKNOWN
    elif reward >= 1.0:
        status = OutcomeStatus.SUCCESS
    else:
        status = OutcomeStatus.FAILURE

    termination = simulation.get("termination_reason")
    meta: dict[str, Any] = {
        "source": "tau2-bench",
        "domain": domain,
        "task_id": task.get("id"),
        "termination_reason": termination,
        # Generic, framework-agnostic flag the rule tier consults so a clean
        # user-initiated end isn't misread as "agent never responded" (ADR-0009).
        "terminated_cleanly": termination in _CLEAN_TERMINATIONS,
        "reward": reward,
        "reward_basis": reward_info.get("reward_basis"),
    }
    if provenance:
        meta["provenance"] = provenance

    return Trace(
        trace_id=str(simulation.get("id")),
        agent_id=domain,
        task_type=domain,
        spec=task_spec_for_domain(
            domain,
            policy_text=policy_text,
            reference_actions=_parse_reference_actions(task),
        ),
        spans=spans,
        outcome=Outcome(status=status, final_response=None),
        # No Layer-1 process label from τ² — outcome is Layer-0 only (ADR-0008). The Layer-2
        # reference-deviation oracle can attach a process label downstream (see argus_benchmark).
        label=None,
        meta=meta,
    )


def load_tau2_results(
    path: str | Path,
    domain: str,
    *,
    policy_text: str | None = None,
    policy_path: str | Path | None = None,
) -> list[Trace]:
    """Load a τ² ``Results`` JSON file and return one ``Trace`` per simulation.

    ``domain`` selects the tool registry/spec (a τ² results file is single-domain).
    Pass ``policy_text`` or ``policy_path`` to attach the NL policy for judge-tier
    grounding (Phase-1 task 4); omitted otherwise.
    """
    path = Path(path)
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.loads(f.read())
    else:
        data = json.loads(path.read_text())

    if policy_text is None and policy_path is not None:
        policy_text = Path(policy_path).read_text()

    tasks_by_id = {str(t["id"]): t for t in data.get("tasks", [])}
    provenance = data.get("_argus_fixture_provenance")

    traces: list[Trace] = []
    for sim in data.get("simulations", []):
        task = tasks_by_id.get(str(sim.get("task_id")), {"id": sim.get("task_id")})
        traces.append(
            simulation_to_trace(
                sim, task, domain, policy_text=policy_text, provenance=provenance
            )
        )
    return traces


def _sim_content_hash(sim: dict[str, Any]) -> str:
    """Stable hash of a simulation's messages + reward — must match the manifest extractor
    (datasets/tau2/extract_manifest.py) so drift in the source file is detectable."""
    import hashlib

    payload = json.dumps(
        {"messages": sim.get("messages"), "reward": (sim.get("reward_info") or {}).get("reward")},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def held_out_task_ids(manifest_path: str | Path) -> dict[str, set[str]]:
    """The frozen held-out *task_id* set per domain, read from the committed manifest.

    The split is by task_id (``sha1(task_id)`` stratified alternation — see
    datasets/tau2/extract_manifest.py), and task_ids are SHARED across every model in the τ²
    suite. So this set is model-invariant: it's the right key for an apples-to-apples
    second-model generality test (hold the held-out tasks fixed, swap only the model). The
    manifest's per-sim ``sim_id``/``content_hash`` are gpt-4.1-specific and intentionally NOT
    used here — those guard the gpt-4.1 regression baseline, not the cross-model comparison.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    return {
        domain: {e["task_id"] for e in dom["entries"]}
        for domain, dom in manifest["domains"].items()
    }


def load_model_held_out(
    results_dir: str | Path,
    manifest_path: str | Path,
    policy_dir: str | Path,
    source_files: dict[str, str],
) -> list[Trace]:
    """Resolve the frozen held-out task_id split against an ARBITRARY model's τ² runs.

    Same held-out tasks as the gpt-4.1 gate (``held_out_task_ids``), but ``source_files`` names
    THIS model's per-domain result files. Keyed on task_id only (a different model produces
    different sim UUIDs and different message hashes by construction, so the gpt-4.1
    content-hash check cannot and must not apply). Used by the report-only second-model
    generality run (Phase 1.5 step 2) — NOT gated against the gpt-4.1 baseline.
    """
    results_dir = Path(results_dir)
    policy_dir = Path(policy_dir)
    held = held_out_task_ids(manifest_path)

    traces: list[Trace] = []
    for domain, task_ids in held.items():
        if domain not in source_files:
            raise ValueError(f"no source file given for held-out domain {domain!r}")
        src = results_dir / source_files[domain]
        if not src.exists():
            raise FileNotFoundError(f"τ² run for second-model split not found: {src}")
        data = json.loads(src.read_text())
        tasks_by_id = {str(t["id"]): t for t in data.get("tasks", [])}
        policy_text = (policy_dir / f"{domain}.md").read_text()

        for sim in data["simulations"]:
            if str(sim.get("task_id")) not in task_ids:
                continue
            task = tasks_by_id.get(str(sim["task_id"]), {"id": sim["task_id"]})
            traces.append(simulation_to_trace(sim, task, domain, policy_text=policy_text))
    return traces


def load_held_out_tau2(
    results_dir: str | Path,
    manifest_path: str | Path,
    policy_dir: str | Path,
) -> list[Trace]:
    """Resolve the committed held-out manifest against the (un-committed) full τ² runs.

    Reads ``eval_manifest.json`` (the frozen held-out split), loads the named source files
    from ``results_dir``, and returns one ``Trace`` per held-out simulation — with each
    sim's ``content_hash`` re-verified so a source file that drifted under the manifest is a
    hard error (you cannot silently move the goalposts). Used by the superset gate (ADR-0010).
    """
    results_dir = Path(results_dir)
    policy_dir = Path(policy_dir)
    manifest = json.loads(Path(manifest_path).read_text())
    source_files: dict[str, str] = manifest["source_files"]

    traces: list[Trace] = []
    for domain, dom_manifest in manifest["domains"].items():
        src = results_dir / source_files[domain]
        if not src.exists():
            raise FileNotFoundError(f"τ² run for held-out manifest not found: {src}")
        data = json.loads(src.read_text())
        sims_by_id = {s["id"]: s for s in data["simulations"]}
        tasks_by_id = {str(t["id"]): t for t in data.get("tasks", [])}
        policy_text = (policy_dir / f"{domain}.md").read_text()

        for entry in dom_manifest["entries"]:
            sim = sims_by_id.get(entry["sim_id"])
            if sim is None:
                raise ValueError(f"manifest sim {entry['sim_id']} absent from {src.name}")
            if _sim_content_hash(sim) != entry["content_hash"]:
                raise ValueError(
                    f"content hash drift for sim {entry['sim_id']} in {src.name} — "
                    "the eval data changed under the manifest; re-extract deliberately"
                )
            task = tasks_by_id.get(str(sim["task_id"]), {"id": sim["task_id"]})
            traces.append(
                simulation_to_trace(sim, task, domain, policy_text=policy_text)
            )
    return traces
