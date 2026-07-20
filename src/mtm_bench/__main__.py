"""``python -m mtm_bench`` — run a benchmark cell from the command line, text or JSON.

Only the file-drivable rulers are exposed here. The APB process-judgment leaderboard reads a
cloned AgentProcessBench repo (human gold + the released LLM predictions + the degenerate rows)
and needs no API key or new labels — so it runs end-to-end from a directory path. The τ² outcome
leaderboard is likewise file-drivable when the contestants are FROZEN verdict caches (the shipped
judges, or your own ``{"verdicts": {...}}`` JSON) — ``tau2-leaderboard`` seats them beside the
constant-flag/constant-pass degenerates on any shipped (or newly generated) rollout file. The
generic ``score_leaderboard`` over live Python *callables* stays a library entry (see the package
README's "submit a detector" section); this CLI does not fake one.

Examples:
  python -m mtm_bench apb-leaderboard --apb-dir ~/scratch/agentprocessbench
  python -m mtm_bench tau2-leaderboard --traces data/tau2/traces_airline_gpt41.json.gz \\
      --domain airline --judge-cache broad_prompt=data/tau2/judge_caches/broad_prompt_diagnostic.json
  python -m mtm_bench tau2-leaderboard --traces data/tau2/traces_airline_o4mini.json.gz \\
      --domain airline --split test --json
  python -m mtm_bench splits --manifest data/tau2/eval_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _apb_leaderboard(args: argparse.Namespace) -> int:
    from .apb_leaderboard import (
        degenerate_entries,
        load_apb_reference,
        reused_llm_entries,
        score_apb_leaderboard,
    )

    apb = Path(args.apb_dir).expanduser()
    if not (apb / "data" / "AgentProcessBench").exists():
        print(
            f"AgentProcessBench not found under {apb} "
            "(clone github.com/RUCBM/AgentProcessBench, or pass --apb-dir / set APB_DIR).",
            file=sys.stderr,
        )
        return 2

    reference = load_apb_reference(apb)
    entries = [*degenerate_entries(), *reused_llm_entries(apb)]
    report = score_apb_leaderboard(reference, entries)

    if args.json:
        print(report.to_json(indent=None if args.compact else 2))
        return 0

    print(report.render(args.dataset))
    if args.dataset != "AVG":
        print("\n" + report.render("AVG"))
    return 0


def _tau2_leaderboard(args: argparse.Namespace) -> int:
    from .leaderboard import _CallableEntry, score_leaderboard
    from .panel import GoldStore, to_gold_items
    from .panel_adapters import tau2_verified_gold
    from .panel_contestants import tau2_cached_outcome_predictors
    from .tau2_loader import held_out_task_ids, load_tau2_results

    traces_path = Path(args.traces).expanduser()
    if not traces_path.exists():
        print(f"traces file not found: {traces_path}", file=sys.stderr)
        return 2

    policy_path = (
        Path(args.policy).expanduser()
        if args.policy
        else traces_path.parent / "policy" / f"{args.domain}.md"
    )
    policy = policy_path.read_text() if policy_path.exists() else None

    # Split filter. The manifest's held-out set is task_id-keyed and model-invariant, so the same
    # test partition applies to every agent model's rollout file (apples-to-apples across cells).
    keep = None
    if args.split != "all":
        manifest_path = (
            Path(args.manifest).expanduser()
            if args.manifest
            else traces_path.parent / "eval_manifest.json"
        )
        if not manifest_path.exists():
            print(f"eval manifest not found: {manifest_path} (needed for --split)", file=sys.stderr)
            return 2
        held = held_out_task_ids(manifest_path).get(args.domain, set())
        keep = (lambda tid: tid in held) if args.split == "test" else (lambda tid: tid not in held)

    store = GoldStore(strict=True)
    for t in load_tau2_results(traces_path, args.domain, policy_text=policy):
        if keep is not None and not keep(t.meta.get("task_id")):
            continue
        store.register(t, tau2_verified_gold(t))
    items = to_gold_items(store.records(), tier="instance_label")
    if not items:
        print("no scoreable traces after split/outcome filtering.", file=sys.stderr)
        return 2
    traces = [store.load_trace(i.trace_id) for i in items]

    # Every cell carries the degenerate baselines (SUBMIT.md reporting rule 3).
    entries = [
        _CallableEntry("baseline:always_flag", lambda t, s: True),
        _CallableEntry("baseline:never_flag", lambda t, s: False),
    ]
    caches: dict[str, str] = {}
    for kv in args.judge_cache or []:
        if "=" not in kv:
            print(f"--judge-cache expects NAME=PATH, got: {kv}", file=sys.stderr)
            return 2
        name, path = kv.split("=", 1)
        caches[name] = path
    entries += [
        _CallableEntry(name, fn) for name, fn in tau2_cached_outcome_predictors(caches).items()
    ]

    report = score_leaderboard(items, traces, entries)
    if args.json:
        print(report.to_json(indent=None if args.compact else 2))
    else:
        print(report.render())
    return 0


def _splits(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.exists():
        print(f"eval manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())
    rows = {
        domain: {
            "tasks_total": dom["n_tasks_total"],
            "tasks_test": dom["n_tasks_held_out"],
            "tasks_dev": dom["n_tasks_total"] - dom["n_tasks_held_out"],
            "sims_test_gpt41": dom["n_sims_held_out"],
        }
        for domain, dom in manifest["domains"].items()
    }
    if args.json:
        print(json.dumps({"split_algo": manifest.get("split_algo"), "domains": rows}, indent=2))
        return 0
    print("frozen dev/test split (task_id-keyed, model-invariant — tune on dev, report on test)")
    print(f"  algo: {manifest.get('split_algo')}")
    for domain, r in rows.items():
        print(
            f"  {domain:10s} tasks: {r['tasks_total']:4d} total = {r['tasks_dev']} dev"
            f" + {r['tasks_test']} test   (gpt-4.1 test sims: {r['sims_test_gpt41']})"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="mtm_bench",
        description="Run an MtM-Bench meta-eval cell (text or JSON).",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    apb = sub.add_parser(
        "apb-leaderboard",
        help="AgentProcessBench strata-1+3 process-judgment leaderboard (label-free, no API).",
    )
    apb.add_argument(
        "--apb-dir",
        default=os.environ.get("APB_DIR", "~/scratch/agentprocessbench"),
        help="path to a cloned AgentProcessBench repo (default: $APB_DIR or ~/scratch/...).",
    )
    apb.add_argument(
        "--dataset",
        default="AVG",
        help="cell to render in text mode: AVG | hotpotqa | gaia_dev | bfcl | tau2.",
    )
    apb.add_argument("--json", action="store_true", help="emit the full report as JSON.")
    apb.add_argument(
        "--compact", action="store_true", help="with --json, emit a single compact line."
    )
    apb.set_defaults(func=_apb_leaderboard)

    t2 = sub.add_parser(
        "tau2-leaderboard",
        help="two-number outcome leaderboard on a τ² rollout file (frozen caches, no API).",
    )
    t2.add_argument("--traces", required=True, help="τ² results JSON (.json or .json.gz).")
    t2.add_argument("--domain", required=True, help="τ² domain of the file: airline | retail.")
    t2.add_argument(
        "--policy",
        default=None,
        help="domain policy markdown (default: <traces_dir>/policy/<domain>.md if present).",
    )
    t2.add_argument(
        "--judge-cache",
        action="append",
        metavar="NAME=PATH",
        help="frozen verdict cache to seat as a contestant (repeatable).",
    )
    t2.add_argument(
        "--split",
        choices=["test", "dev", "all"],
        default="test",
        help="which side of the frozen task split to score (default: test).",
    )
    t2.add_argument(
        "--manifest",
        default=None,
        help="eval manifest with the frozen split (default: <traces_dir>/eval_manifest.json).",
    )
    t2.add_argument("--json", action="store_true", help="emit the full report as JSON.")
    t2.add_argument(
        "--compact", action="store_true", help="with --json, emit a single compact line."
    )
    t2.set_defaults(func=_tau2_leaderboard)

    sp = sub.add_parser("splits", help="show the frozen dev/test task split per domain.")
    sp.add_argument(
        "--manifest",
        default="data/tau2/eval_manifest.json",
        help="eval manifest path (default: data/tau2/eval_manifest.json).",
    )
    sp.add_argument("--json", action="store_true", help="emit the split table as JSON.")
    sp.set_defaults(func=_splits)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
