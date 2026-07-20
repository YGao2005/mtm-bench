"""Adapter: AgentErrorBench (AEB / AgentDebug) failure trajectory → ``argus_core.Trace``.

AEB (arXiv 2509.25370, github.com/ulab-uiuc/AgentDebug, MIT) ships HUMAN-annotated failure
trajectories with a single critical-failure step + a cognitive-error category per trajectory
(EXP-0092). Trajectories are OpenAI-ish chat (``{messages: [...], metadata: {...}}``) where the
agent alternates user-observation / assistant-action turns. This adapter turns one trajectory into a
``Trace`` and joins its consensus label to the attribution-axis ``GoldRecord`` (built in
``argus_benchmark.panel_adapters`` — kept there so this adapter has no benchmark→panel dependency).

Verified against the downloaded ALFWorld split (EXP-0092):
  - ``messages`` alternate ``user`` (observation) and ``assistant`` (action); ALFWorld has no
    structured tool_calls (actions are prose), so each assistant message is one AGENT_MSG span.
  - The label's ``critical_failure_step`` is 1-based over the ASSISTANT turns (step N == the Nth
    assistant message); ``metadata.steps`` == the assistant-turn count. We map step N → that
    assistant message's span_id so the panel's attribution join lands on a real span (GRAFT 4).
  - The label is FAILURE-ONLY (``metadata.won`` is false for every annotated trajectory) → the
    panel marks the record FAILURE and the attribution scorer reports fire-on-clean as N/A.

ANTI-ORACLE-COPYING: the label (``critical_failure_step``/``failure_type``) is the answer key — it
is NEVER written onto the Trace; it stays in the sidecar ``GoldRecord`` (the panel firewall). The
Trace a contestant sees carries only the messages.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .schema import Outcome, OutcomeStatus, Phase, Role, Span, SpanKind, TaskSpec, Trace

# trajectory_id / filename both start "<LLM>_<idx>_..." — the (LLM, idx) join key (EXP-0092).
_KEY_RE = re.compile(r"(.+?)_(\d+)_")


def _join_key(s: str) -> tuple[str, int] | None:
    m = _KEY_RE.match(s)
    return (m.group(1), int(m.group(2))) if m else None


def messages_to_spans(messages: list[dict[str, Any]]) -> tuple[list[Span], dict[int, str]]:
    """AEB chat messages → spans, plus ``{assistant_step_number(1-based): span_id}`` so a label's
    ``critical_failure_step`` resolves to the right span. user → USER_MSG, assistant → AGENT_MSG."""
    spans: list[Span] = []
    step_to_span: dict[int, str] = {}
    assistant_n = 0
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant":
            assistant_n += 1
            kind, phase, srole = SpanKind.AGENT_MSG, Phase.RESPONSE, Role.ASSISTANT
        else:
            kind, phase, srole = SpanKind.USER_MSG, Phase.INTAKE, Role.USER
        span_id = f"m{i}"
        spans.append(
            Span(span_id=span_id, index=i, kind=kind, phase=phase, role=srole,
                 content=m.get("content"))
        )
        if role == "assistant":
            step_to_span[assistant_n] = span_id
    return spans, step_to_span


def load_aeb_trajectory(path: str | Path, *, task_type: str = "alfworld") -> Trace:
    """Load one AEB trajectory JSON (``{messages, metadata}``) → a ``Trace``.

    ``meta`` carries ``aeb_step_to_span`` (the 1-based assistant-step → span_id map the attribution
    GoldRecord uses) + the trajectory id/LLM for the label join. The outcome is FAILURE (AEB is
    failure-only; ``metadata.won`` is false). No gold (critical step) is written onto the Trace."""
    import json

    path = Path(path)
    data = json.loads(path.read_text())
    messages = data.get("messages") or []
    meta_in = data.get("metadata") or {}
    spans, step_to_span = messages_to_spans(messages)

    llm_idx = _join_key(path.stem)
    trace_id = f"aeb:{task_type}:{path.stem}"
    return Trace(
        trace_id=trace_id,
        agent_id=str(meta_in.get("model", "unknown")),
        task_type=task_type,
        spec=TaskSpec(goal=f"AgentErrorBench {task_type} task.", allowed_tools=[]),
        spans=spans,
        # AEB is failure-only; the gold critical-step lives in the sidecar, not here.
        outcome=Outcome(status=OutcomeStatus.FAILURE, final_response=None),
        label=None,
        meta={
            "source": "agenterrorbench",
            "task_type": task_type,
            "aeb_step_to_span": step_to_span,
            "aeb_n_steps": int(meta_in.get("steps", len(step_to_span))),
            "aeb_join_key": list(llm_idx) if llm_idx else None,
            "won": bool(meta_in.get("won", False)),
        },
    )


def index_trajectories(traj_dir: str | Path) -> dict[tuple[str, int], Path]:
    """{(LLM, idx): path} for every trajectory file in a split dir — the label-join index."""
    out: dict[tuple[str, int], Path] = {}
    for p in sorted(Path(traj_dir).glob("*.json")):
        k = _join_key(p.name)
        if k is not None:
            out.setdefault(k, p)
    return out


def load_aeb_labels(label_path: str | Path) -> list[dict[str, Any]]:
    """Load an AEB ``Label/<split>_labels.json`` (a list of consensus label records)."""
    import json

    return json.loads(Path(label_path).read_text())


def label_join_key(label: dict[str, Any]) -> tuple[str, int] | None:
    """The (LLM, idx) join key for a label, parsed from its ``trajectory_id`` (EXP-0092)."""
    tid = label.get("trajectory_id", "")
    m = _KEY_RE.match(tid)
    return (str(label.get("LLM", m.group(1) if m else "")), int(m.group(2))) if m else None


__all__ = [
    "messages_to_spans",
    "load_aeb_trajectory",
    "index_trajectories",
    "load_aeb_labels",
    "label_join_key",
]
