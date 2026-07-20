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
  python -m mtm_bench run-all --split test --json
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


def _score_tau2_traces(
    traces_path: Path,
    domain: str,
    split: str,
    caches: dict[str, str | Path],
    *,
    policy_path: Path | None = None,
    manifest_path: Path | None = None,
):
    """Scoring core shared by ``tau2-leaderboard`` and ``run-all``: one rollout file → one
    ``LeaderboardReport`` (degenerate baselines + the frozen-cache judges, split-filtered).

    Returns None when the split/outcome filter leaves nothing scoreable; raises
    ``FileNotFoundError`` when a split is requested but the eval manifest is missing."""
    from .leaderboard import _CallableEntry, score_leaderboard
    from .panel import GoldStore, to_gold_items
    from .panel_adapters import tau2_verified_gold
    from .panel_contestants import tau2_cached_outcome_predictors
    from .tau2_loader import held_out_task_ids, load_tau2_results

    if policy_path is None:
        policy_path = traces_path.parent / "policy" / f"{domain}.md"
    policy = policy_path.read_text() if policy_path.exists() else None

    # Split filter. The manifest's held-out set is task_id-keyed and model-invariant, so the same
    # test partition applies to every agent model's rollout file (apples-to-apples across cells).
    keep = None
    if split != "all":
        if manifest_path is None:
            manifest_path = traces_path.parent / "eval_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"eval manifest not found: {manifest_path} (needed for --split)"
            )
        held = held_out_task_ids(manifest_path).get(domain, set())
        keep = (lambda tid: tid in held) if split == "test" else (lambda tid: tid not in held)

    store = GoldStore(strict=True)
    for t in load_tau2_results(traces_path, domain, policy_text=policy):
        if keep is not None and not keep(t.meta.get("task_id")):
            continue
        store.register(t, tau2_verified_gold(t))
    items = to_gold_items(store.records(), tier="instance_label")
    if not items:
        return None
    traces = [store.load_trace(i.trace_id) for i in items]

    # Every cell carries the degenerate baselines (SUBMIT.md reporting rule 3).
    entries = [
        _CallableEntry("baseline:always_flag", lambda t, s: True),
        _CallableEntry("baseline:never_flag", lambda t, s: False),
    ]
    entries += [
        _CallableEntry(name, fn) for name, fn in tau2_cached_outcome_predictors(caches).items()
    ]
    return score_leaderboard(items, traces, entries)


def _tau2_leaderboard(args: argparse.Namespace) -> int:
    traces_path = Path(args.traces).expanduser()
    if not traces_path.exists():
        print(f"traces file not found: {traces_path}", file=sys.stderr)
        return 2

    caches: dict[str, str | Path] = {}
    for kv in args.judge_cache or []:
        if "=" not in kv:
            print(f"--judge-cache expects NAME=PATH, got: {kv}", file=sys.stderr)
            return 2
        name, path = kv.split("=", 1)
        caches[name] = path

    try:
        report = _score_tau2_traces(
            traces_path,
            args.domain,
            args.split,
            caches,
            policy_path=Path(args.policy).expanduser() if args.policy else None,
            manifest_path=Path(args.manifest).expanduser() if args.manifest else None,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    if report is None:
        print("no scoreable traces after split/outcome filtering.", file=sys.stderr)
        return 2

    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.write_text(report.to_json(indent=None if args.compact else 2) + "\n")
        print(f"wrote JSON report to {out_path}", file=sys.stderr)
    elif args.json:
        print(report.to_json(indent=None if args.compact else 2))
    else:
        print(report.render())
    return 0


def _run_all(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).expanduser()
    trace_files = sorted(data_dir.glob("traces_*.json.gz"))
    if not trace_files:
        print(f"no trace files matching traces_*.json.gz under {data_dir}", file=sys.stderr)
        return 2

    # Seat every shipped frozen cache as a contestant. Files whose shape the cache loader does not
    # recognize as a verdict cache (e.g. the agreement-audit exports) are skipped by it, and a
    # trace a cache does not cover gets no flag — so airline-only caches score honestly on retail.
    cache_dir = data_dir / "judge_caches"
    caches: dict[str, str | Path] = (
        {p.stem: p for p in sorted(cache_dir.glob("*.json"))} if cache_dir.exists() else {}
    )

    # results[model][domain] = LeaderboardReport (filenames are traces_{domain}_{model}.json.gz).
    results: dict[str, dict] = {}
    for tf in trace_files:
        parts = tf.name.removesuffix(".json.gz").split("_", 2)
        if len(parts) != 3:
            print(f"skipping unrecognized trace filename: {tf.name}", file=sys.stderr)
            continue
        _, domain, model = parts
        try:
            report = _score_tau2_traces(tf, domain, args.split, caches)
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            return 2
        if report is None:
            print(
                f"skipping {tf.name}: no scoreable traces on split={args.split}", file=sys.stderr
            )
            continue
        results.setdefault(model, {})[domain] = report
    if not results:
        print("no scoreable trace files.", file=sys.stderr)
        return 2

    # The caches that actually seated (entries are shared across cells) — offered files whose
    # shape the loader rejected are not claimed here.
    any_report = next(iter(next(iter(results.values())).values()))
    seated = sorted(
        e.removeprefix("judge:") for e in any_report.entry_names if e.startswith("judge:")
    )

    if args.json:
        combined = {
            "schema": "mtm.tau2_run_all.v1",
            "split": args.split,
            "judge_caches": seated,
            "models": {
                model: {domain: rep.to_dict() for domain, rep in sorted(domains.items())}
                for model, domains in sorted(results.items())
            },
        }
        print(json.dumps(combined, indent=None if args.compact else 2))
        return 0

    lines: list[str] = []
    lines.append(f"══ τ² outcome leaderboard — every shipped rollout file (split={args.split}) ══")
    lines.append(
        "  (R = recall-on-corrupt-success ↑better, F = firing-rate-on-clean ↓better; "
        "two numbers, never pooled — R11)"
    )
    if seated:
        lines.append(f"  judge caches seated: {', '.join(seated)}")
    lines.append("")
    for model in sorted(results):
        lines.append(f"── model: {model} ──")
        for domain in sorted(results[model]):
            rep = results[model][domain]
            for cell in rep.cells:
                for tier in rep.tiers:
                    sample = next(
                        (
                            rep.scores[e][cell][tier]
                            for e in rep.entry_names
                            if tier in rep.scores[e].get(cell, {})
                        ),
                        None,
                    )
                    if sample is None:
                        continue
                    lines.append(
                        f"  {domain}  (corrupt={sample.n_corrupt}, clean={sample.n_clean})"
                    )
                    for e in rep.entry_names:
                        cs = rep.scores[e].get(cell, {}).get(tier)
                        if cs is None:
                            continue
                        rec = "  n/a " if cs.n_corrupt == 0 else f"{cs.recall:5.2f}"
                        fir = "  n/a " if cs.n_clean == 0 else f"{cs.clean_fire_rate:5.2f}"
                        lines.append(f"    {e:<36} R={rec}  F={fir}")
        lines.append("")
    print("\n".join(lines).rstrip())
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
    t2.add_argument(
        "-o",
        "--output",
        default=None,
        help="write the JSON report to this path instead of stdout (always JSON).",
    )
    t2.set_defaults(func=_tau2_leaderboard)

    ra = sub.add_parser(
        "run-all",
        help="score every shipped τ² rollout file (traces_*.json.gz) in one combined report.",
    )
    ra.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent.parent.parent / "data" / "tau2"),
        help="directory holding traces_*.json.gz + judge_caches/ (default: the repo's data/tau2).",
    )
    ra.add_argument(
        "--split",
        choices=["test", "dev", "all"],
        default="test",
        help="which side of the frozen task split to score (default: test).",
    )
    ra.add_argument("--json", action="store_true", help="emit the combined report as JSON.")
    ra.add_argument(
        "--compact", action="store_true", help="with --json, emit a single compact line."
    )
    ra.set_defaults(func=_run_all)

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
