"""Reproduce the paper's τ²/census numbers from the data shipped in this repo — offline, no keys.

Self-checking: each section asserts the recomputed numbers against the values printed in the paper
(results.tex macros) and the script exits non-zero on any mismatch. Judge-dependent numbers replay
from the frozen caches in data/tau2/judge_caches/ (no API spend).

  A. Truth-clean over-flag decomposition (paper §crown: OverflagTruthClean / OverflagRealCatch)
  B. Human-anchored judge-FP-gap          (paper §crown: FpgapHuman / FpgapHumanStrict / config tier)
  C. Over-flag on the held-out depth leg  (paper §strata: airline R/F, retail R/F, broad-prompt judge)

Run:  python scripts/reproduce_paper.py
"""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
TAU2 = DATA / "tau2"

sys.path.insert(0, str(ROOT / "src"))

from mtm_bench import (  # noqa: E402
    GoldStore,
    score_leaderboard,
    tau2_cached_outcome_predictors,
    tau2_verified_gold,
    to_gold_items,
)
from mtm_bench.leaderboard import _CallableEntry  # noqa: E402
from mtm_bench.stats import wilson_ci  # noqa: E402
from mtm_bench.tau2_loader import held_out_task_ids, load_tau2_results  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want, tol: float = 0.0) -> None:
    ok = (abs(got - want) <= tol) if tol else (got == want)
    print(f"  {'✓' if ok else '✗ MISMATCH'}  {name}: got {got}  (paper: {want})")
    if not ok:
        FAILURES.append(name)


def verify_manifest() -> None:
    print("── data integrity (MANIFEST.json) ──")
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    for rel, meta in manifest.items():
        digest = hashlib.sha256((DATA / rel).read_bytes()).hexdigest()
        if digest != meta["sha256"]:
            FAILURES.append(f"manifest:{rel}")
            print(f"  ✗ HASH DRIFT  {rel}")
    print(f"  ✓ {len(manifest)} files verified\n")


def section_a_truthclean() -> None:
    print("── A. truth-clean over-flag decomposition (n=40 airline census, judge held fixed) ──")
    census = [json.loads(x) for x in (TAU2 / "census_labels.jsonl").read_text().splitlines() if x.strip()]
    human = {r["trace_id"]: r.get("blind", {}).get("corrupt_success") for r in census}
    cache = json.loads((TAU2 / "judge_caches" / "shipped_semantic_n40.json").read_text())
    judge = {r["trace_id"]: bool(r["judge_violated"]) for r in cache["rows"]}

    n_corrupt = sum(1 for v in human.values() if v == "yes")
    n_clean = sum(1 for v in human.values() if v == "no")
    catch = sum(1 for t, h in human.items() if h == "yes" and judge.get(t))
    over = sum(1 for t, h in human.items() if h == "no" and judge.get(t))
    naive = sum(1 for t in human if judge.get(t))

    check("census n", len(human), 40)
    check("corrupt / clean", (n_corrupt, n_clean), (9, 27))
    check("naive oracle-clean fires", naive, 14)
    check("GENUINE over-flag (human-clean)", round(over / n_clean, 2), 0.33)
    check("REAL catch (human-corrupt)", round(catch / n_corrupt, 2), 0.44)
    lo, hi = wilson_ci(over, n_clean)
    print(f"     over-flag {over}/{n_clean} CI [{lo:.2f},{hi:.2f}]  ·  catch {catch}/{n_corrupt}")
    print()


def section_b_fpgap() -> None:
    print("── B. human-anchored judge-FP-gap (blind pilot, complete census) ──")
    # Run the ported scorer for its full report (it exits via SystemExit — contain it), then
    # re-derive the headline number here with the scorer's own flagged-stratum rule.
    try:
        runpy.run_path(str(ROOT / "scripts" / "fp_gap.py"), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            FAILURES.append("fp_gap scorer exited non-zero")
    census = [json.loads(x) for x in (TAU2 / "census_labels.jsonl").read_text().splitlines() if x.strip()]
    sealed = json.loads((TAU2 / "reveal.sealed.json").read_text())["reveal"]
    flagged_clean = flagged_n = 0
    for r in census:
        s = sealed.get(r["item_id"], {})  # the scorer's own join: item_id, stratum from record first
        stratum = r.get("stratum") or s.get("stratum")
        flagged = stratum in {"majority", "union_only"} or bool(s.get("assist_flags"))
        if flagged:
            flagged_n += 1
            if r["blind"]["corrupt_success"] != "yes":
                flagged_clean += 1
    check("FP-gap incl-unsure", f"{flagged_clean}/{flagged_n}", "23/31")
    check("FP-gap rate", round(flagged_clean / flagged_n, 3), 0.742)
    print()


def section_c_depth_leg() -> None:
    print("── C. over-flag on the held-out depth leg (broad-prompt judge, frozen cache) ──")
    held = held_out_task_ids(TAU2 / "eval_manifest.json")
    trace_files = {
        "airline": TAU2 / "traces_airline_gpt41.json.gz",
        "retail": TAU2 / "traces_retail_gpt41.json.gz",
    }
    store = GoldStore(strict=True)
    for domain, path in trace_files.items():
        policy = (TAU2 / "policy" / f"{domain}.md").read_text()
        for t in load_tau2_results(path, domain, policy_text=policy):
            if t.meta.get("task_id") not in held.get(domain, set()):
                continue
            store.register(t, tau2_verified_gold(t))
    records = store.records()
    items = to_gold_items(records, tier="instance_label")
    traces = [store.load_trace(i.trace_id) for i in items]

    preds = tau2_cached_outcome_predictors(
        {"broad_prompt": TAU2 / "judge_caches" / "broad_prompt_diagnostic.json"}
    )
    entries = [_CallableEntry(name, fn) for name, fn in preds.items()]
    report = score_leaderboard(items, traces, entries)
    print(report.render())

    by_cell = report.scores["judge:broad_prompt"]
    paper = {"tau2_airline": (0.88, 0.48), "tau2_retail": (0.73, 0.49)}
    for cell, tiers in by_cell.items():
        cs = tiers["instance_label"]
        dom = cell.split("|")[-1]
        if dom in paper:
            want_r, want_f = paper[dom]
            check(f"{dom} recall", round(cs.recall, 2), want_r)
            check(f"{dom} fire-on-clean", round(cs.clean_fire_rate, 2), want_f)
    print()


def main() -> int:
    verify_manifest()
    section_a_truthclean()
    section_b_fpgap()
    section_c_depth_leg()
    if FAILURES:
        print(f"✗ {len(FAILURES)} MISMATCH(ES): {FAILURES}")
        return 1
    print("✓ all paper numbers reproduce from shipped data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
