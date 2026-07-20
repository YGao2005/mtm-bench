"""Detector-as-unit meta-eval leaderboard (the shippable instrument, EXP-0020 charter step 8).

PAE fixes ONE judge and ranks AGENTS. This module does the inverse: it scores ANY failure-detector
as the unit-under-test against a fixed **corrupt-success gold cell**, so a native oracle, a no-LLM
name baseline, Argus's typed checks, N disjoint-family LLM judges, and PAE's own judge all rank on
the same ruler. That is the differentiation EXP-0020 said we must ship (PAE only promised one).

Two numbers per entry, ALWAYS side by side, NEVER pooled (R11 + the two-numbers discipline):
  - recall-on-corrupt-success: of the gold corrupt-success traces, how many the detector flags;
  - firing-rate-on-clean:      of the gold clean traces, how many it (wrongly) flags.
Both are reported PER R11 certificate_tier (config_only / threshold_const / instance_label /
truth_oracle), with Wilson CIs, because a detector that wins the typed tier can lose the semantic
tier and pooling would hide it.

ANTI-OVERFIT / no-oracle-copying (drift-trap #1, ADR-0008): the gold's `corrupt_success` bit is
supplied by a mechanism INDEPENDENT of every scored entry (human-primary gold for the semantic
cell; the deterministic `step_id` fact for the parallel-call cell). The scorer does not read
`Trace.outcome` to score detectors — outcome only marks which traces are eligible (oracle=SUCCESS).
When a detector's predicate IS the gold mechanism (Argus's typed check on the deterministic cell),
its score is tautological-by-construction and MUST be labelled so, never touted (EXP-0015 lesson).

Dependency-free (reuses `wilson_ci`); an entry is just a name + a `predict` callable, so adding
PAE's judge or a new model is a three-line adapter, not a framework change.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .schema import TaskSpec, Trace

from .serialize import ci_or_none, nan_to_none
from .stats import wilson_ci

# The R11 certificate tiers, in cost order. Every row is reported per tier, never pooled.
CERTIFICATE_TIERS: tuple[str, ...] = (
    "config_only",
    "threshold_const",
    "instance_label",
    "truth_oracle",
)


@dataclass(frozen=True)
class GoldItem:
    """One human-primary (or mechanism-independent) gold label for one oracle=SUCCESS trace.

    ``corrupt_success`` is the bit the leaderboard scores recall against: True iff the oracle said
    SUCCESS but the policy was actually violated. ``tier`` is the R11 certificate tier of the
    violated clause (or of the trace's check class for clean traces) — the stratification key.
    ``source`` records HOW the bit was decided (e.g. ``human_blind`` / ``step_id_fact``) so an
    entry that shares the gold's mechanism can be flagged tautological."""

    trace_id: str
    cell: str  # "<agent>|<domain>"
    tier: str  # one of CERTIFICATE_TIERS
    corrupt_success: bool
    source: str = "unspecified"
    violated_clause_id: str | None = None


@runtime_checkable
class DetectorEntry(Protocol):
    """A leaderboard contestant. ``predict`` returns True iff the entry flags the trace as a
    failure (corrupt-success). Stateless w.r.t. scoring; may carry its own cache/model handle.

    ``tautological_on`` lists the gold ``source`` values this entry's predicate IS — so the scorer
    can mark that cell's number "by construction" instead of letting it masquerade as a win."""

    name: str
    tautological_on: frozenset[str]

    def predict(self, trace: Trace, spec: TaskSpec | None) -> bool: ...


@dataclass
class _CallableEntry:
    """Concrete DetectorEntry built from a plain predicate — the common adapter."""

    name: str
    _predict: Callable[[Trace, TaskSpec | None], bool]
    tautological_on: frozenset[str] = frozenset()

    def predict(self, trace: Trace, spec: TaskSpec | None) -> bool:
        return self._predict(trace, spec)


def detector_entry(name: str, detector, *, tautological_on: frozenset[str] = frozenset()):
    """Wrap any Argus ``Detector`` (rule/process/judge) as a leaderboard entry: it flags a trace
    iff the detector emits ≥1 signal. The detector's own abstention (empty list) = no flag."""
    return _CallableEntry(
        name=name,
        _predict=lambda t, s: bool(detector.analyze(t, s if s is not None else t.spec)),
        tautological_on=tautological_on,
    )


@dataclass
class CellScore:
    """One entry's two numbers in one (cell, tier), with Wilson CIs and tautology flag."""

    n_corrupt: int
    recall_hits: int
    n_clean: int
    clean_fires: int
    tautological: bool = False

    @property
    def recall(self) -> float:
        return self.recall_hits / self.n_corrupt if self.n_corrupt else float("nan")

    @property
    def clean_fire_rate(self) -> float:
        return self.clean_fires / self.n_clean if self.n_clean else float("nan")

    @property
    def recall_ci(self) -> tuple[float, float]:
        return wilson_ci(self.recall_hits, self.n_corrupt)

    @property
    def clean_ci(self) -> tuple[float, float]:
        return wilson_ci(self.clean_fires, self.n_clean)

    def to_dict(self) -> dict:
        """Raw counts AND the derived two numbers (recall, fire-on-clean, Wilson CIs), JSON-safe.
        The derived fields are what a consumer wants; asdict() alone would drop them, and their
        nan-when-empty values are mapped to null (a bare NaN is not valid JSON)."""
        return {
            "n_corrupt": self.n_corrupt,
            "recall_hits": self.recall_hits,
            "n_clean": self.n_clean,
            "clean_fires": self.clean_fires,
            "tautological": self.tautological,
            "recall": nan_to_none(self.recall),
            "recall_ci": ci_or_none(self.recall_ci, self.n_corrupt),
            "clean_fire_rate": nan_to_none(self.clean_fire_rate),
            "clean_ci": ci_or_none(self.clean_ci, self.n_clean),
        }


@dataclass
class LeaderboardReport:
    # scores[entry_name][cell][tier] = CellScore
    scores: dict[str, dict[str, dict[str, CellScore]]] = field(default_factory=dict)
    entry_names: list[str] = field(default_factory=list)
    cells: list[str] = field(default_factory=list)
    tiers: list[str] = field(default_factory=list)
    n_gold: int = 0
    n_scored: int = 0  # gold items whose trace was found in the trace set

    def render(self) -> str:
        lines: list[str] = []
        lines.append("══ Detector-as-unit meta-eval leaderboard (EXP-0020) ══")
        lines.append(
            f"  {self.n_scored}/{self.n_gold} gold items scored "
            f"(R = recall-on-corrupt-success ↑better, F = firing-rate-on-clean ↓better; "
            f"two numbers, never pooled — R11)"
        )
        lines.append("  '†' = tautological by construction (entry predicate == gold mechanism; "
                     "not a win, the EXP-0015 lesson)\n")
        for cell in self.cells:
            lines.append(f"── cell: {cell} ──")
            for tier in self.tiers:
                # Skip tiers with no gold in this cell.
                any_here = any(
                    tier in self.scores[e].get(cell, {}) for e in self.entry_names
                )
                if not any_here:
                    continue
                # n is entry-invariant; read it off the first entry that has it.
                sample = next(
                    self.scores[e][cell][tier] for e in self.entry_names
                    if tier in self.scores[e].get(cell, {})
                )
                lines.append(
                    f"  tier={tier}  (corrupt={sample.n_corrupt}, clean={sample.n_clean})"
                )
                for e in self.entry_names:
                    cs = self.scores[e].get(cell, {}).get(tier)
                    if cs is None:
                        continue
                    taut = "†" if cs.tautological else " "
                    rec = "  n/a " if cs.n_corrupt == 0 else f"{cs.recall:5.2f}"
                    fir = "  n/a " if cs.n_clean == 0 else f"{cs.clean_fire_rate:5.2f}"
                    lines.append(
                        f"    {taut}{e:<26} R={rec} {_ci(cs.recall_ci, cs.n_corrupt)}  "
                        f"F={fir} {_ci(cs.clean_ci, cs.n_clean)}"
                    )
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """The full leaderboard as a JSON-safe dict: metadata + scores[entry][cell][tier], each
        cell carrying raw counts and the derived two numbers. Round-trips through json.dumps."""
        return {
            "schema": "mtm.leaderboard.v1",
            "n_gold": self.n_gold,
            "n_scored": self.n_scored,
            "entry_names": list(self.entry_names),
            "cells": list(self.cells),
            "tiers": list(self.tiers),
            "scores": {
                entry: {
                    cell: {tier: cs.to_dict() for tier, cs in tiers.items()}
                    for cell, tiers in cells.items()
                }
                for entry, cells in self.scores.items()
            },
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize ``to_dict()`` to a JSON string (indent=None for a compact single line)."""
        return json.dumps(self.to_dict(), indent=indent)


def _ci(ci: tuple[float, float], n: int) -> str:
    return "        " if n == 0 else f"[{ci[0]:.2f},{ci[1]:.2f}]"


def score_leaderboard(
    gold: list[GoldItem],
    traces: list[Trace],
    entries: list[DetectorEntry],
) -> LeaderboardReport:
    """Score every entry against the gold cell, stratified by (cell, tier). Two numbers each.

    Only gold items whose trace is present in ``traces`` are scored (a missing trace is counted in
    n_gold but not n_scored, and reported — never silently dropped). An entry's prediction is taken
    verbatim from ``entry.predict``; the scorer adds NO outcome/answer-key info of its own."""
    trace_by_id = {t.trace_id: t for t in traces}
    cells = sorted({g.cell for g in gold})
    tiers = [t for t in CERTIFICATE_TIERS if any(g.tier == t for g in gold)]
    entry_names = [e.name for e in entries]

    # Pre-bucket gold by (cell, tier).
    buckets: dict[tuple[str, str], list[GoldItem]] = {}
    n_scored = 0
    for g in gold:
        if g.trace_id in trace_by_id:
            n_scored += 1
        buckets.setdefault((g.cell, g.tier), []).append(g)

    scores: dict[str, dict[str, dict[str, CellScore]]] = {e.name: {} for e in entries}
    for entry in entries:
        for (cell, tier), items in buckets.items():
            n_corrupt = recall_hits = n_clean = clean_fires = 0
            taut = False
            for g in items:
                trace = trace_by_id.get(g.trace_id)
                if trace is None:
                    continue  # unscored (counted in n_gold only)
                if g.source in entry.tautological_on:
                    taut = True
                fired = bool(entry.predict(trace, trace.spec))
                if g.corrupt_success:
                    n_corrupt += 1
                    recall_hits += int(fired)
                else:
                    n_clean += 1
                    clean_fires += int(fired)
            scores[entry.name].setdefault(cell, {})[tier] = CellScore(
                n_corrupt=n_corrupt, recall_hits=recall_hits,
                n_clean=n_clean, clean_fires=clean_fires, tautological=taut,
            )

    return LeaderboardReport(
        scores=scores, entry_names=entry_names, cells=cells, tiers=tiers,
        n_gold=len(gold), n_scored=n_scored,
    )


__all__ = [
    "CERTIFICATE_TIERS",
    "GoldItem",
    "DetectorEntry",
    "detector_entry",
    "CellScore",
    "LeaderboardReport",
    "score_leaderboard",
]
