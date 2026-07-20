"""Human-anchored judge-FP-gap from the blind pilot -- fills the sec.4 slot of
docs/planning/judge-fp-gap-finding.md.

READ-ONLY over datasets/tau2/metaeval_pilot/{labels.jsonl,reveal.sealed.json}. Touches NOTHING
the labeling UI writes; safe to run while labeling is in flight (recomputes from scratch each run).

What it computes (and how it differs from the EXP-0046 68.5%):
  EXP-0046's 68.5% is a JUDGE-vs-JUDGE proxy -- of comms-TYPED ensemble flags, the fraction whose
  cited clause maps (Jaccard->R11 tier) to `config_only`. It needs a per-flag `violation_kind`,
  which THIS pilot's sealed flags do not carry. This pilot instead supports the stronger,
  HUMAN-PRIMARY quantity the finding doc's sec.4 slot actually asks for:

    judge-FP rate = of items the ensemble FLAGGED, the fraction a human blind-judged NOT corrupt.

  Stratified by approx_tier (config_only vs truth_oracle): the clause-misattribution thesis
  predicts the FPs concentrate on `config_only`. Reported with the two-numbers partner (judge-MISS
  rate on the unflagged stratum) and a planted-FP attention check. This is an INDEPENDENT sample
  (only 3/48 pilot traces overlap the EXP-0044 harvest), so it CORROBORATES the proxy from the
  human side; it does not recompute it.

Run:  uv run python examples/metaeval_pilot_fp_gap.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
PILOT = ROOT / "data" / "tau2"
LABELS = PILOT / "census_labels.jsonl"
SEALED = PILOT / "reveal.sealed.json"

# strata where >=1 model flagged the trace (judge "fired"); "unflagged" = judge silent.
FLAGGED_STRATA = {"majority", "union_only"}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point rate + Wilson 95% CI. Returns (rate, lo, hi); (0,0,0) for n==0."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def _load() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not LABELS.exists():
        return [], {}
    recs = [json.loads(ln) for ln in LABELS.read_text().splitlines() if ln.strip()]
    sealed = json.loads(SEALED.read_text())["reveal"] if SEALED.exists() else {}
    return recs, sealed


def _fmt(label: str, k: int, n: int) -> str:
    r, lo, hi = _wilson(k, n)
    if n == 0:
        return f"  {label:42s}  n=0  (no items yet)"
    return f"  {label:42s}  {k}/{n} = {r:5.1%}  CI[{lo:.3f}, {hi:.3f}]"


def report() -> int:
    recs, sealed = _load()
    if not recs:
        print(f"no labels yet at {LABELS} -- label some items first (webui or review CLI).")
        return 0

    # Join each labeled item to its sealed entry; classify.
    flagged_clean = flagged_n = 0  # judge fired AND human NOT corrupt -> the FP cell
    flagged_clean_strict = 0  # human == "no" (excludes unsure)
    unflagged_corrupt = unflagged_n = 0  # judge silent AND human corrupt -> the MISS cell
    by_tier: dict[str, list[int]] = {}  # tier -> [fp_count, flagged_count]
    # Planted-FP attention check. A plant the human marks no/unsure = cleanly rejected. A plant the
    # human marks "yes" is INDETERMINATE, not a failure: we cannot tell from structured fields
    # whether the plant baited the "yes" or the human rejected the plant's trap and flagged an
    # orthogonal issue (P001 did the latter -- `violated_clause_id` is unused/None across the pilot,
    # so a clause-match auto-adjudication is impossible). Indeterminate plants are named for a read.
    planted_total = planted_rejected = 0
    planted_indeterminate: list[str] = []

    for r in recs:
        iid = r["item_id"]
        s = sealed.get(iid, {})
        stratum = r.get("stratum") or s.get("stratum")
        cs = r["blind"]["corrupt_success"]
        human_corrupt = cs == "yes"
        human_clean_any = cs != "yes"  # no OR unsure
        human_clean_strict = cs == "no"
        flagged = stratum in FLAGGED_STRATA or bool(s.get("assist_flags"))

        if s.get("planted_fp"):
            planted_total += 1
            if human_clean_any:
                planted_rejected += 1
            else:  # human said "yes" on a plant -- indeterminate, needs a rationale read
                planted_indeterminate.append(iid)

        if flagged:
            flagged_n += 1
            flagged_clean += int(human_clean_any)
            flagged_clean_strict += int(human_clean_strict)
            tier = s.get("approx_tier") or "untiered"
            t = by_tier.setdefault(tier, [0, 0])
            t[1] += 1
            t[0] += int(human_clean_any)
        else:  # unflagged stratum
            unflagged_n += 1
            unflagged_corrupt += int(human_corrupt)

    print(f"\n=== HUMAN-ANCHORED judge-FP-gap (pilot, {len(recs)} items labeled) ===")
    print("    blind-first human verdict vs the disjoint-family ensemble's flags. INDEPENDENT")
    print("    sample; corroborates the EXP-0046 68.5% judge-vs-judge proxy, not a recompute.\n")

    print("[1] Judge-FP rate  (ensemble FLAGGED + human NOT corrupt = cried wolf vs human gold):")
    print(_fmt("incl-unsure (human != yes)", flagged_clean, flagged_n))
    print(_fmt("strict (human == no)", flagged_clean_strict, flagged_n))

    print("\n   by approx_tier (misattribution thesis => FPs concentrate on config_only):")
    for tier, (fp, n) in sorted(by_tier.items()):
        print(_fmt(f"tier={tier}", fp, n))

    print("\n[2] Judge-MISS rate  (two-numbers partner: ensemble SILENT + human corrupt = missed):")
    print(_fmt("unflagged + human==yes", unflagged_corrupt, unflagged_n))

    print("\n[3] Planted-FP sanity (attention check: planted clean traces should be REJECTED):")
    print(_fmt("planted cleanly rejected (human no/unsure)", planted_rejected, planted_total))
    if planted_indeterminate:
        print("   INDETERMINATE (human 'yes' on a plant -- read the rationale, NOT auto-failed): "
              + ", ".join(planted_indeterminate))
        print("   (a 'yes' may reject the plant's trap yet flag an orthogonal issue -- e.g. P001.)")

    done_cells = {r.get("cell") for r in recs}
    print(f"\n   cells touched: {sorted(c for c in done_cells if c)}")
    if flagged_n < 12:
        print("   [!] n_flagged < 12 -- UNDERPOWERED; wide CIs, provisional not a verdict.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(report())
