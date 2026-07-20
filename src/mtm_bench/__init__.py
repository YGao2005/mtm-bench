"""MtM-Bench — Measuring the Measurers: a meta-evaluation benchmark for agent-failure detectors.

Grades the *graders* of agent transcripts. Any detector — a no-LLM keyword rule, a deterministic
compiled check, an LLM judge — is scored on one ruler with the two-numbers discipline:
recall-on-corrupt ↑ and fire-on-clean ↓, reported side by side per certificate tier, never pooled
into a single F1. An answer-key firewall keeps every contestant from scoring by copying the oracle.

Three contestant axes, one callable each (see docs/SUBMIT.md):
  outcome          predict(trace, spec) -> bool          via ``DetectorEntry`` / ``detector_entry``
  process-quality  grader(trace) -> {span_id: -1|0|1}    via ``StepGraderEntry``
  attribution      localize(trace) -> span_id | None     via ``LocalizerEntry``
"""

from .apb_leaderboard import (
    APB_DATASETS,
    APBLeaderboardReport,
    APBRecord,
    AxisScore,
    StepVerifierEntry,
    degenerate_entries,
    load_apb_reference,
    reused_llm_entries,
    score_apb_leaderboard,
)
from .leaderboard import (
    CERTIFICATE_TIERS,
    CellScore,
    DetectorEntry,
    GoldItem,
    LeaderboardReport,
    detector_entry,
    score_leaderboard,
)
from .panel import (
    FirewallViolation,
    GoldRecord,
    GoldStore,
    has_clean_pool,
    to_gold_items,
)
from .panel_adapters import aeb_gold, apb_gold, apb_gold_from_record, tau2_verified_gold
from .panel_attribution import LocalizerEntry, render_attribution, score_attribution
from .panel_contestants import (
    apb_judge_localizers,
    apb_judge_outcome_predictors,
    apb_judge_step_graders,
    tau2_cached_outcome_predictors,
)
from .panel_scoring import StepGraderEntry, render_pq, score_pq_entries
from .schema import (
    Label,
    Outcome,
    OutcomeStatus,
    Span,
    SpanKind,
    TaskSpec,
    Trace,
)
from .stats import wilson_ci
from .tau2_loader import load_tau2_results

__all__ = [
    # trace schema (vendored, self-contained)
    "Trace", "Span", "SpanKind", "Label", "Outcome", "OutcomeStatus", "TaskSpec",
    # detector-as-unit leaderboard (outcome axis)
    "CERTIFICATE_TIERS", "CellScore", "DetectorEntry", "GoldItem", "LeaderboardReport",
    "detector_entry", "score_leaderboard",
    # unified gold panel + firewall
    "GoldRecord", "GoldStore", "FirewallViolation", "has_clean_pool", "to_gold_items",
    "tau2_verified_gold", "apb_gold", "apb_gold_from_record", "aeb_gold",
    # process-quality axis
    "StepGraderEntry", "score_pq_entries", "render_pq",
    # attribution axis
    "LocalizerEntry", "score_attribution", "render_attribution",
    # APB strata leaderboard + replayed judges
    "APB_DATASETS", "APBLeaderboardReport", "APBRecord", "AxisScore", "StepVerifierEntry",
    "degenerate_entries", "load_apb_reference", "reused_llm_entries", "score_apb_leaderboard",
    "apb_judge_step_graders", "apb_judge_outcome_predictors", "apb_judge_localizers",
    "tau2_cached_outcome_predictors",
    # data loading + stats
    "load_tau2_results", "wilson_ci",
]
