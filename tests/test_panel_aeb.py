"""Tests for the AEB attribution adapter + gold + scoring (the attribution axis end-to-end).

Load-bearing assertions (EXP-0092 + synthesis §2):
  - the trajectory adapter maps assistant step N (1-based) → the Nth assistant span;
  - aeb_gold maps critical_failure_step → that span (the GRAFT-4 join), FAILURE outcome, kappa=0.55;
  - an empty-failure_type+reasoning label is flagged unfalsifiable;
  - a critical step that does not resolve RAISES (never silently re-points — answer-key firewall);
  - the attribution scorer drops unfalsifiable rows + reports fire-on-clean N/A (failure-only);
  - (if the ALFWorld split is present) the real labels join + register through the firewall.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mtm_bench.aeb_loader import (
    index_trajectories,
    label_join_key,
    load_aeb_labels,
    load_aeb_trajectory,
    messages_to_spans,
)
from mtm_bench.panel import GoldStore
from mtm_bench.panel_adapters import aeb_gold
from mtm_bench.panel_attribution import LocalizerEntry, score_attribution

_AEB = Path(__file__).resolve().parents[1] / "datasets" / "agenterrorbench"
_ALF_LABELS = _AEB / "Label" / "alfworld_labels.json"
_ALF_TRAJ = _AEB / "Original_Failure_Trajectory" / "ALFWorld"


def _traj(n_steps: int) -> dict:
    msgs = []
    for _ in range(n_steps):
        msgs.append({"role": "user", "content": "obs"})
        msgs.append({"role": "assistant", "content": "act"})
    return {"messages": msgs, "metadata": {"steps": n_steps, "won": False, "model": "m"}}


def _label(crit: int, *, falsifiable: bool = True) -> dict:
    ann = {"step": crit, "plan": {
        "failure_type": "inefficient_plan" if falsifiable else "",
        "reasoning": "repeated the same action" if falsifiable else "",
    }}
    return {
        "trajectory_id": "GPT-4o_001_id", "LLM": "GPT-4o", "task_type": "alfworld",
        "critical_failure_step": crit, "critical_failure_module": "plan",
        "step_annotations": [ann],
    }


def _write_traj(tmp: Path, name: str, n_steps: int) -> Path:
    import json

    p = tmp / f"{name}.json"
    p.write_text(json.dumps(_traj(n_steps)))
    return p


def test_step_to_span_maps_nth_assistant() -> None:
    spans, step_to_span = messages_to_spans(_traj(3)["messages"])
    asst = [s.span_id for s in spans if s.kind.value == "agent_msg"]
    assert step_to_span[1] == asst[0]
    assert step_to_span[3] == asst[2]


def test_aeb_gold_maps_critical_step_to_span(tmp_path: Path) -> None:
    t = load_aeb_trajectory(_write_traj(tmp_path, "GPT-4o_001_x", 5))
    g = aeb_gold(t, _label(3))
    asst = [s.span_id for s in t.spans if s.kind.value == "agent_msg"]
    assert g.substrate == "aeb"
    assert g.outcome.value == "failure"  # AEB failure-only
    assert g.label.critical_span_id == asst[2]  # step 3 → 3rd assistant span
    assert g.label.step_index == 3
    assert g.provenance.reliability.cohen_kappa == 0.55
    assert g.label.falsifiable is True


def test_unfalsifiable_label_flagged(tmp_path: Path) -> None:
    t = load_aeb_trajectory(_write_traj(tmp_path, "GPT-4o_002_x", 5))
    g = aeb_gold(t, _label(2, falsifiable=False))
    assert g.label.falsifiable is False


def test_unresolvable_critical_step_raises(tmp_path: Path) -> None:
    t = load_aeb_trajectory(_write_traj(tmp_path, "GPT-4o_003_x", 3))  # only 3 assistant steps
    with pytest.raises(ValueError, match="does not"):
        aeb_gold(t, _label(99))  # step 99 has no span


def test_scorer_drops_unfalsifiable_and_fire_on_clean_na(tmp_path: Path) -> None:
    t1 = load_aeb_trajectory(_write_traj(tmp_path, "GPT-4o_001_a", 5))
    t2 = load_aeb_trajectory(_write_traj(tmp_path, "GPT-4o_002_b", 5))
    recs = [aeb_gold(t1, _label(3, falsifiable=True)),
            aeb_gold(t2, _label(2, falsifiable=False))]
    traces = {r.trace_id: t for r, t in [(recs[0], t1), (recs[1], t2)]}
    asst1 = [s.span_id for s in t1.spans if s.kind.value == "agent_msg"]
    entry = LocalizerEntry("hit_step3", lambda tr: asst1[2] if tr.trace_id == t1.trace_id else None)
    sc = score_attribution(recs, traces, [entry])["hit_step3"]["aeb|alfworld"]
    assert sc.n_falsifiable == 1  # the unfalsifiable record is dropped from the denominator
    assert sc.n_unfalsifiable == 1
    assert sc.localized_hits == 1 and sc.localization_acc == 1.0
    import math
    assert math.isnan(sc.fire_on_clean)  # failure-only


def test_real_alfworld_split_joins_and_registers() -> None:
    if not _ALF_LABELS.exists() or not _ALF_TRAJ.exists():
        return  # split absent → hermetic tests cover the logic
    idx = index_trajectories(_ALF_TRAJ)
    labels = load_aeb_labels(_ALF_LABELS)
    store = GoldStore(strict=True)
    joined = 0
    for label in labels:
        k = label_join_key(label)
        if k not in idx:
            continue
        t = load_aeb_trajectory(idx[k], task_type="alfworld")
        store.register(t, aeb_gold(t, label))  # exercises firewall + ingest invariant
        joined += 1
    assert joined >= 50  # ~75 of 100 ALFWorld labels join to a present trajectory
