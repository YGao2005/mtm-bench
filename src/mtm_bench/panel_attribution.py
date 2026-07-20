"""Attribution axis scoring: failure LOCALIZATION (which step, and — optionally — why).

The attribution axis (AEB / AgentErrorBench) asks a different question than outcome or
process-quality: given a trajectory that DID fail, which step is the critical failure, and what KIND
of error is it? Gold is one ``critical_span_id`` + a ``category`` per failure-only trajectory
(EXP-0092). The natural metric is **localization accuracy**: of the trajectories with a falsifiable
gold critical step, on how many did the grader point at the right step.

Hard constraints carried from EXP-0092 + the synthesis (§2/§3):
  • **Failure-only → fire-on-clean is N/A, structurally.** Every AEB record is a failure; there is
    no clean pool, so a false-positive-rate cannot be computed on AEB alone (this is WHY AEB is
    composed with tau2/SOP-Bench, not used solo). ``AttributionScore`` reports localization +
    ``n_clean=0`` and never invents a fire-on-clean number.
  • **Drop the unfalsifiable rows.** 14% of AEB labels are unfalsifiable (empty failure_type AND
    reasoning); ``falsifiable=false`` on the panel label. They are excluded from the localization
    denominator (counted separately as ``n_unfalsifiable``), never scored as a silent miss.
  • **Uncontrolled vocabulary.** The category vocab is substrate-private + inconsistent (``plan`` vs
    ``planning``); the attribution axis is effectively SINGLE-substrate (AEB only). This module
    scores STEP localization (objective); it does NOT score category-match (that would require
    canonicalizing the vocab first — a separate, flagged step). Cross-domain generality is nominal.

A contestant here is a LOCALIZER: read the (redacted) failed Trace, return the span_id it judges the
critical failure step (or None to abstain). Abstain counts as a miss (cannot inflate accuracy)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .schema import Trace

from .panel import AttributionLabel, GoldRecord
from .stats import wilson_ci

# An attribution contestant: read the failed Trace, return the span_id of the predicted critical
# step (None = abstain = a miss). It never sees the gold critical_span_id (the firewall).
Localizer = Callable[[Trace], str | None]


@dataclass
class AttributionScore:
    """One localizer's score in one (substrate|domain) cell. Localization only; no fire-on-clean."""

    n_falsifiable: int = 0  # failure trajectories with a falsifiable gold critical step
    localized_hits: int = 0  # predicted span == gold critical_span_id
    n_unfalsifiable: int = 0  # dropped from the denominator (14% empty AEB rows), reported
    n_abstained: int = 0  # localizer returned None on a falsifiable record (counts as a miss)

    # AEB is failure-only: there is NO clean pool, so fire-on-clean is undefined by construction.
    n_clean: int = 0  # always 0 for AEB; kept so the report can print "n/a" via the shared path

    @property
    def localization_acc(self) -> float:
        return self.localized_hits / self.n_falsifiable if self.n_falsifiable else float("nan")

    @property
    def localization_ci(self) -> tuple[float, float]:
        return wilson_ci(self.localized_hits, self.n_falsifiable)

    @property
    def fire_on_clean(self) -> float:
        # Structurally N/A — AEB is failure-only. Never a number.
        return float("nan")


@dataclass
class LocalizerEntry:
    name: str
    localize: Localizer


def score_attribution(
    records: list[GoldRecord],
    traces_by_id: dict[str, Trace],
    entries: list[LocalizerEntry],
    *,
    include_excluded: bool = False,
) -> dict[str, dict[str, AttributionScore]]:
    """Score localizers over the panel's attribution records, per ``substrate|domain`` cell.

    Returns ``{entry_name: {cell: AttributionScore}}``. Unfalsifiable rows are counted in
    ``n_unfalsifiable`` and excluded from the localization denominator (never a silent miss). An
    abstain (None) on a falsifiable record counts as a miss (so abstaining cannot inflate accuracy).
    fire-on-clean is N/A on every AEB cell (failure-only) — reported as such, never invented."""
    out: dict[str, dict[str, AttributionScore]] = {e.name: {} for e in entries}
    for r in records:
        if not isinstance(r.label, AttributionLabel):
            continue
        if r.excluded_from_pooling and not include_excluded:
            continue
        trace = traces_by_id.get(r.trace_id)
        if trace is None:
            continue
        cell = f"{r.substrate}|{r.domain}"
        gold_span = r.label.critical_span_id
        falsifiable = r.label.falsifiable
        for entry in entries:
            sc = out[entry.name].setdefault(cell, AttributionScore())
            if not falsifiable:
                sc.n_unfalsifiable += 1
                continue
            sc.n_falsifiable += 1
            pred = entry.localize(trace)
            if pred is None:
                sc.n_abstained += 1
            elif pred == gold_span:
                sc.localized_hits += 1
    return out


def render_attribution(scores: dict[str, dict[str, AttributionScore]]) -> str:
    """Render the attribution localization table (per cell; no fire-on-clean — AEB failure-only)."""
    lines = ["══ Panel attribution axis (failure localization) ══",
             "  L = localization accuracy ↑ (predicted critical step == gold)   "
             "F = fire-on-clean = n/a (AEB failure-only)",
             "  ⚠ single-substrate (AEB); category vocab uncontrolled → step localization only\n"]
    cells = sorted({c for e in scores.values() for c in e})
    for cell in cells:
        lines.append(f"── cell: {cell} ──")
        for name, by_cell in scores.items():
            sc = by_cell.get(cell)
            if sc is None:
                continue
            loc = "  n/a " if not sc.n_falsifiable else f"{sc.localization_acc:5.2f}"
            lines.append(
                f"    {name:<26} L={loc} F=  n/a  "
                f"(falsifiable={sc.n_falsifiable}, hits={sc.localized_hits}, "
                f"abstained={sc.n_abstained}, unfalsifiable_dropped={sc.n_unfalsifiable})"
            )
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "Localizer",
    "AttributionScore",
    "LocalizerEntry",
    "score_attribution",
    "render_attribution",
]
