# MtM-Bench — Measuring the Measurers

A **meta-evaluation benchmark for agent-failure detectors** — it grades the *graders* of agent
transcripts. Drop any detector (a no-LLM keyword rule, a deterministic compiled check, an LLM
judge) onto **one ruler** and see how it separates corrupt from clean with the **two-numbers
discipline** — never a single pooled score.

Companion artifact to the paper *"Measuring the Measurers: Grading Agent-Failure Detectors, and
Knowing When a Benchmark Cannot."*

## What you get from `pip install` alone

**4,448 traces** across **12 cells** (4 agent models × 3 domains), each with deterministic
oracle gold — no downloads, no API keys, no external dependencies:

| Domain | Tasks | Traces (4 trials × 4 models) | Policy |
|---|---|---|---|
| airline | 50 | 800 | customer-service SOP |
| retail | 114 | 1,824 | e-commerce support SOP |
| telecom | 114 | 1,824 | tech-support / billing SOP |

Agent models: **gpt-4.1**, **gpt-4.1-mini**, **o4-mini**, **Claude 3.7 Sonnet** — all run on
[τ²-bench](https://github.com/sierra-research/tau2-bench) (MIT) with the deterministic
state-hash reward oracle.

Every trace carries a frozen dev/test split (task_id-keyed, model-invariant, `sha1` stratified
alternation). Score on `test`, tune on `dev` — one command for each.

## Why two numbers, never pooled

A single accuracy/F1 hides the failure mode that matters. So every cell reports **both** numbers
side by side with Wilson CIs:

- **recall-on-corrupt** — of the traces that are *actually* bad, how many the detector flags. ↑
- **fire-on-clean** — of the genuinely-clean traces, how many it *wrongly* flags. ↓

A contestant sitting on the `recall ≈ fire-on-clean` diagonal has no separating signal — the
ruler is built to make that visible. There is deliberately no `f1` on any report type.

## The firewall (the rule you must not break)

Your detector sees the **trace only** — never the gold label, reference trajectory, or answer-key
field. Reading the answer key is *oracle-copying*: it rebuilds an inferior copy of the oracle and
destroys the disagreement the benchmark measures. The `GoldStore` enforces this structurally
(gold lives in a sidecar; `find_leaks` audits residual answer-key fields). In the paper's
ablation, letting a judge see the gold inflates its balanced accuracy from 65% to 100%. If your
contestant's predicate *is* the gold mechanism, declare `tautological_on=` so its number is
flagged "by construction."

## Install

```bash
pip install -e .            # only dependency: pydantic
```

## Quickstart

```bash
# score all 12 cells at once (test split, text table):
python -m mtm_bench run-all

# single cell, your own judge cache:
python -m mtm_bench tau2-leaderboard \
    --traces data/tau2/traces_airline_gpt41.json.gz \
    --domain airline \
    --judge-cache broad_prompt=data/tau2/judge_caches/broad_prompt_diagnostic.json

# reproduce the paper's numbers (offline, self-checking, 13 assertions):
python scripts/reproduce_paper.py
```

Or use the library directly:

```python
from mtm_bench import GoldItem, detector_entry, score_leaderboard

report = score_leaderboard(gold_items, traces, [my_entry])
print(report.render())                     # text table, both numbers + Wilson CIs
open("out.json", "w").write(report.to_json())   # JSON-safe (nan → null)
```

## Shipped baseline numbers

Every judge below replays from a frozen cache — all numbers reproduce offline. R = recall-on-corrupt
(↑), F = fire-on-clean (↓). The gpt-4.1 cells (test split):

| Judge | airline R / F | retail R / F | telecom R / F |
|---|---|---|---|
| Claude Sonnet 5 | 0.62 / 0.14 | 0.35 / 0.26 | 0.12 / 0.38 |
| Claude Haiku 4.5 | 0.75 / 0.32 | 0.60 / 0.38 | 0.11 / 0.42 |
| broad_prompt (Sonnet 4.6, paper) | 0.88 / 0.48 | 0.73 / 0.49 | — |

The Sonnet 5 and Haiku 4.5 caches cover **all 12 cells** (2,256 test verdicts each) — the full
table is one command: `python -m mtm_bench run-all`. Two things the table already shows: (1) no
judge dominates — Haiku fires more and catches more, Sonnet 5 is conservative on both numbers, so
neither is "better" without picking an operating point (this is why the ruler never pools); (2)
telecom breaks every judge (R ≤ 0.2 at F ≥ 0.3 — *below* the random diagonal): its oracle
failures are mostly process/workflow breakdowns, which a semantic policy judge is explicitly
scoped to ignore. A detector that clears telecom recalls a failure class no shipped judge sees.

**Truth-clean calibration audit** (n=40 human-labeled airline census):

| Metric | Value | 95% CI |
|---|---|---|
| Genuine over-flag rate | 9/27 = 0.33 | [0.19, 0.52] |
| Real catch rate | 4/9 = 0.44 | [0.19, 0.73] |

This audit demonstrates that "fire-on-clean" isn't simply "false positive rate" — a third of
oracle-clean fires were genuine catches the oracle missed.

## Submit a detector

Three axes, one contestant protocol each — see [docs/SUBMIT.md](docs/SUBMIT.md) for worked examples:

| Axis | What it decides | Callable signature | Scorer |
|---|---|---|---|
| **Outcome** | is this SUCCESS trace a corrupt success? | `predict(trace, spec) -> bool` | `score_leaderboard` |
| **Process-quality** | per step: harmful / neutral / helpful | `grader(trace) -> dict[span_id, -1\|0\|1]` | `score_pq_entries` |
| **Attribution** | which step is the critical failure? | `localize(trace) -> span_id \| None` | `score_attribution` |

## Data architecture

### Core (self-contained, no external downloads)

The 12-cell τ² corpus. Every trace has deterministic oracle gold (the state-hash reward), a
frozen dev/test split, and the domain policy the agent was supposed to follow. A detector author
gets a complete, non-trivial benchmark from `pip install` alone.

### Optional seats (external repos, not redistributed)

| Substrate | What it adds | Clone from |
|---|---|---|
| [AgentProcessBench](https://github.com/RUCBM/AgentProcessBench) | per-step gold (helpful/neutral/harmful) + 20 pre-seated LLM verifier baselines across 4 domains | `data/external/README.md` |
| [AgentErrorBench](https://github.com/RUCBM/AgentErrorBench) | failure-point localization gold (ALFWorld split) | `data/external/README.md` |

These adapt into the same `GoldRecord` panel and two-number scorer via the `apb_gold` /
`aeb_gold` adapters — they broaden the ruler, not a load-bearing dependency.

### Calibration annex (human gold)

The n=40 blind census (`data/tau2/census_labels.jsonl`) — the only cell where the truth-clean
decomposition is available. Not an evaluation axis per se, but the evidence that the benchmark's
disagreement-is-the-product thesis holds empirically.

## Dev/test split

```bash
python -m mtm_bench splits                              # show the split table
python -m mtm_bench tau2-leaderboard --split test ...   # score only held-out tasks (default)
python -m mtm_bench tau2-leaderboard --split dev ...    # score only dev tasks
python -m mtm_bench tau2-leaderboard --split all ...    # ignore the split
```

The split is `sha1(task_id)` stratified alternation — every trial of a task lands on the same
side. The partition is model-invariant: the same held-out task IDs apply to gpt-4.1, gpt-4.1-mini,
o4-mini, and Claude 3.7 Sonnet alike, so cross-model comparisons are apples-to-apples.

**What gold exists on each side.** The human-labeled census and the frozen judge cache cover
**test-split tasks only**. On `dev` you have oracle gold but no human labels or judge verdicts.
To tune a detector: generate your own verdicts on the dev traces (grounding them in the shipped
`data/tau2/policy/` files), score with `--split dev`, and report final numbers on `--split test`.

## Repo layout

```
src/mtm_bench/     the scorers: leaderboard (outcome), panel (gold + firewall), panel_scoring
                   (process-quality), panel_attribution, apb_leaderboard, schema (self-contained
                   pydantic trace model), tau2_loader
data/tau2/         4,448 traces (gzipped), policies, frozen caches, eval manifest, human census
scripts/           reproduce_paper.py — the paper's numbers from shipped data, offline
docs/SUBMIT.md     how to seat your detector on the ruler
tests/             the scoring-math + firewall regression suite (61 tests)
```

## License

MIT (see LICENSE). The τ² policies/traces derive from τ²-bench (MIT). AgentProcessBench and
AgentErrorBench data are **not** redistributed; download scripts point at the original releases.
