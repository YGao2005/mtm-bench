"""``python -m mtm_bench`` — run a benchmark cell from the command line, text or JSON.

Only the file-drivable rulers are exposed here. The APB process-judgment leaderboard reads a
cloned AgentProcessBench repo (human gold + the released LLM predictions + the degenerate rows)
and needs no API key or new labels — so it runs end-to-end from a directory path. The generic
detector-as-unit ``score_leaderboard`` takes Python *callables* as contestants (a detector's
``predict``), which cannot be passed on a command line, so it stays a library entry (see the
package README's "submit a detector" section); this CLI does not fake one.

Examples:
  python -m mtm_bench apb-leaderboard --apb-dir ~/scratch/agentprocessbench
  python -m mtm_bench apb-leaderboard --dataset tau2
  python -m mtm_bench apb-leaderboard --json > leaderboard.json
"""

from __future__ import annotations

import argparse
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="mtm_bench",
        description="Run an Argus meta-eval benchmark cell (text or JSON).",
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
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
