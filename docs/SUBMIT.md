# Submit a detector to MtM-Bench

A contestant reads a trace and makes a decision. It never sees the gold — the firewall is
structural (`GoldStore` keeps gold in a sidecar; `find_leaks` audits residual answer-key fields),
and a contestant whose predicate *is* the gold mechanism must declare `tautological_on=` so its
number is flagged "by construction."

Runnable end-to-end version of everything below: `python scripts/submit_detector_example.py` (FREE).

## Axis 1 — Outcome (corrupt-success detection)

Decide: *is this oracle-SUCCESS trace actually a policy violation?*

```python
from mtm_bench import GoldItem, detector_entry, score_leaderboard
from mtm_bench.leaderboard import _CallableEntry

# Option A: a plain predicate
entry = _CallableEntry("my_detector", lambda trace, spec: my_predicate(trace))

# Option B: any object with .analyze(trace, spec) -> list (flags iff ≥1 signal; [] = abstain)
entry = detector_entry("my_detector", MyDetector())

report = score_leaderboard(gold_items, traces, [entry])
print(report.render())          # recall-on-corrupt ↑ and fire-on-clean ↓, per tier, Wilson CIs
report.to_json()                # JSON export (schema mtm.leaderboard.v1)
```

Score it on the shipped human-census cell by adapting `scripts/reproduce_paper.py` §C — replace the
cached judge entry with yours.

## Axis 2 — Process-quality (per-step grading)

Decide: for each assistant step, is it harmful (−1), neutral (0), or helpful (+1)?

```python
from mtm_bench import StepGraderEntry, score_pq_entries, render_pq

entry = StepGraderEntry(name="my_grader", grader=lambda trace: {span_id: -1, ...})
scores = score_pq_entries(records, traces_by_id, [entry])
print(render_pq(scores))        # recall on gold −1 steps ↑, fire on gold +1 steps ↓
```

Gold for this axis comes from AgentProcessBench (see `data/external/README.md`); its 20 released
LLM verifiers are pre-seated as baseline rungs via `apb_judge_step_graders(apb_dir)`.

## Axis 3 — Attribution (failure localization)

Decide: which step is the critical failure? Return its `span_id`, or `None` to abstain.

```python
from mtm_bench import LocalizerEntry, score_attribution, render_attribution

entry = LocalizerEntry(name="my_localizer", localize=lambda trace: "span-17")
scores = score_attribution(records, traces_by_id, [entry])
print(render_attribution(scores))   # localization accuracy over falsifiable labels
```

Position baselines (first/mid/last step) score ≈0.02–0.06 on the AgentErrorBench ALFWorld split —
a real localizer must clear that bar.

## Reporting rules (what a submission must include)

1. **Both numbers, never pooled.** Recall and fire-on-clean side by side per cell/tier. No F1.
2. **The firewall holds.** Your callable reads the trace only. If your predicate shares the gold's
   mechanism, declare `tautological_on=` — the scorer will dagger that cell.
3. **Beat the degenerates.** Every cell carries constant-flag/constant-pass baselines; a
   submission that doesn't separate from them (CI-aware) is reported as non-discriminating,
   which is a result about the cell, not a win.
