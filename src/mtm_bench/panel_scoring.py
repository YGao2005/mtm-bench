"""Per-axis panel scoring: the two un-pooled numbers, expressed once per gold axis.

The panel grades contestant graders on THREE orthogonal gold axes (synthesis §3 claim #1). Each
axis needs the two un-pooled numbers (recall-on-failure ↑, fire-on-clean ↓) in its OWN units:

  • outcome (binary, trajectory)    → ``panel.to_gold_items`` + the shipped ``score_leaderboard``
                                       (done — EXP-0096). NOT re-implemented here.
  • process_quality (ordinal, step) → ``score_pq_entries`` (this module): the "corrupt" unit is the
                                       harmful step (gold ``-1``); recall = caught -1 steps,
                                       fire-on-clean = -1 predicted on a gold ``+1`` (helpful) step.
  • attribution (categorical, step) → ``panel_attribution.score_attribution`` (Task 8).

The process-quality numbers are the SAME quantities ``apb_leaderboard.AxisScore`` already tracks as
the ``-1``-class confusion (``cls_tp[-1]`` / ``cls_fp[-1]`` / ``cls_fn[-1]``) — recall on the -1
class and its false-alarm rate against +1. We express them in the panel's ``span_values`` view here
and CROSS-VALIDATE against ``apb_leaderboard._score_one`` in the tests (one scoring truth, two
views), so this is a re-view of the validated scorer, not a parallel untested copy.

AUTOCORRELATION CAVEAT (synthesis §2 / open-risk 4): APB ``-1`` propagates ("-1 sticks until
corrected"), so a trajectory's -1 steps are NOT independent. ``StepTwoNumber`` therefore reports
both the pooled step counts AND ``n_trajectories`` so a reader can see the effective sample is
closer to the trajectory count than the step count — the schema records the rule; this scorer
surfaces it, never hides it behind a big-N precision.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .schema import Trace

from .panel import GoldRecord, ProcessQualityLabel
from .stats import wilson_ci

# A process-quality contestant: read the (redacted) Trace, emit {span_id: -1|0|1}. A span it does
# not label is "no flag" (+1), matching apb_leaderboard's missing-prediction == no-flag rule.
StepGrader = Callable[[Trace], dict[str, int]]


@dataclass
class StepTwoNumber:
    """One contestant's two step-level numbers in one (substrate|domain) cell, on the -1 class.

    ``n_trajectories`` is carried beside the step counts because APB -1 propagation makes steps
    autocorrelated — the effective N is nearer the trajectory count than the step counts."""

    n_harmful: int = 0  # gold -1 steps (the "corrupt" positive class)
    recall_hits: int = 0  # gold -1 AND predicted -1
    n_helpful: int = 0  # gold +1 steps (the "clean" class)
    clean_fires: int = 0  # gold +1 BUT predicted -1 (false alarm)
    n_neutral: int = 0  # gold 0 steps (reported, not scored on this binary view)
    n_trajectories: int = 0

    @property
    def recall(self) -> float:
        return self.recall_hits / self.n_harmful if self.n_harmful else float("nan")

    @property
    def fire_on_clean(self) -> float:
        return self.clean_fires / self.n_helpful if self.n_helpful else float("nan")

    @property
    def recall_ci(self) -> tuple[float, float]:
        return wilson_ci(self.recall_hits, self.n_harmful)

    @property
    def clean_ci(self) -> tuple[float, float]:
        return wilson_ci(self.clean_fires, self.n_helpful)


@dataclass
class StepGraderEntry:
    """A process-quality contestant: ``name`` + a ``StepGrader`` (trace → {span_id: -1|0|1})."""

    name: str
    grader: StepGrader


def score_pq_entries(
    records: list[GoldRecord],
    traces_by_id: dict[str, Trace],
    entries: list[StepGraderEntry],
    *,
    include_excluded: bool = False,
) -> dict[str, dict[str, StepTwoNumber]]:
    """Score process-quality contestants over the panel's process_quality records.

    ``traces_by_id`` are the REDACTED traces a contestant sees (the GoldStore.load_trace view); the
    grader reads the trace and emits ``{span_id: -1|0|1}``; we score that prediction vs the gold
    ``span_values`` on the -1 class. Returns ``{entry_name: {cell: StepTwoNumber}}``. Per cell, two
    numbers, never pooled."""
    out: dict[str, dict[str, StepTwoNumber]] = {e.name: {} for e in entries}
    for r in records:
        if not isinstance(r.label, ProcessQualityLabel):
            continue
        if r.excluded_from_pooling and not include_excluded:
            continue
        trace = traces_by_id.get(r.trace_id)
        if trace is None:
            continue
        cell = f"{r.substrate}|{r.domain}"
        gold = r.label.span_values
        for entry in entries:
            pred = entry.grader(trace)
            sc = out[entry.name].setdefault(cell, StepTwoNumber())
            sc.n_trajectories += 1
            for span_id, gv in gold.items():
                pv = pred.get(span_id, 1)  # missing prediction == no flag (+1)
                if gv == -1:
                    sc.n_harmful += 1
                    sc.recall_hits += int(pv == -1)
                elif gv == 1:
                    sc.n_helpful += 1
                    sc.clean_fires += int(pv == -1)
                else:
                    sc.n_neutral += 1
    return out


def render_pq(scores: dict[str, dict[str, StepTwoNumber]]) -> str:
    """Render the process-quality two-number table (per cell, never pooled; -1 = harmful class)."""
    lines = ["══ Panel process-quality axis (per-step, -1=harmful class) ══",
             "  R = recall-on-harmful ↑   F = fire-on-helpful ↓   (two numbers, never pooled)",
             "  ⚠ APB -1 propagates → steps autocorrelated; effective N ≈ n_traj, not n_steps\n"]
    cells = sorted({c for e in scores.values() for c in e})
    for cell in cells:
        lines.append(f"── cell: {cell} ──")
        for name, by_cell in scores.items():
            sc = by_cell.get(cell)
            if sc is None:
                continue
            rec = "  n/a " if not sc.n_harmful else f"{sc.recall:5.2f}"
            fir = "  n/a " if not sc.n_helpful else f"{sc.fire_on_clean:5.2f}"
            lines.append(
                f"    {name:<26} R={rec} F={fir} "
                f"(harmful={sc.n_harmful}, helpful={sc.n_helpful}, "
                f"neutral={sc.n_neutral}, n_traj={sc.n_trajectories})"
            )
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "StepGrader",
    "StepTwoNumber",
    "StepGraderEntry",
    "score_pq_entries",
    "render_pq",
]
