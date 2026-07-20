"""Strata-1+3 process-judgment leaderboard over AgentProcessBench (the label-free instrument).

This is the meta-eval's general-IF + WHERE strata (metaeval-difficulty-prereg.md §2), built on
APB's released human gold — NO new labels needed. The unit-under-test is a **step-verifier**: any
system that reads a trajectory and emits per-step ternary labels (+1/0/-1) and a final-result label.
APB's 20 LLMs ship pre-computed predictions we reuse as baseline rungs; degenerate baselines compute
trivially; Argus (on the τ² slice) is a future entry once the APB→Trace adapter lands.

What this adds OVER APB's own `compare.py` (which only ranked 20 LLMs on StepAcc/FirstErrAcc):
  • detector-as-unit framing — degenerates + LLMs (+ eventually Argus) on one ruler;
  • the IF axis as a first-class binary (process-failure-present = final_label==-1), with the
    two-numbers discipline (recall-on-fail AND firing-on-ok), never just per-step accuracy;
  • a BALANCED step metric (macro-F1 over the 3 classes) beside micro StepAcc — because EXP-0030
    measured that micro StepAcc has a 62.8% all-+1 degenerate floor that beats 9/21 real LLMs, so a
    bare-StepAcc ranking is gameable;
  • the anti-triviality SEPARATION VERDICT (prereg V2/V3 + §6): per axis, do real entries separate
    from the best degenerate, CI-aware? If not → the cell is "too easy", flagged mechanically.

Ingestion (`_load_apb`, keying) is the V1-validated path (EXP-0030: reproduced APB byte-exact),
promoted here from the probe. Dependency-free; reuses `wilson_ci`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .serialize import nan_to_none

# APB's 4 datasets and their total_index offsets (compare.py:DATASET_OFFSETS) — for record keying.
APB_DATASETS: tuple[str, ...] = ("hotpotqa", "gaia_dev", "bfcl", "tau2")
_DATASET_OFFSETS = {"hotpotqa": 0, "gaia_dev": 250, "bfcl": 500, "tau2": 750}
STEP_CLASSES: tuple[int, ...] = (1, 0, -1)


@dataclass(frozen=True)
class APBRecord:
    """One APB trajectory's labels (reference OR a verifier's prediction). We score over labels
    only — `messages`/`tools` aren't needed to score reused predictions."""

    key: str
    dataset: str
    data_source: str
    step_labels: dict[str, int]  # {step_index_str: +1|0|-1}
    final_label: int  # +1|0|-1
    failed: bool = False  # prediction marked llm_annotate_failed (counts as "no flag")


# A step-verifier entry: given a reference record, return its (step_labels, final_label), or None
# if it has no/failed prediction for that record.
PredictFn = Callable[[APBRecord], tuple[dict[str, int], int] | None]


@dataclass
class StepVerifierEntry:
    name: str
    predict: PredictFn
    is_degenerate: bool = False


# ── ingestion (V1-validated, EXP-0030) ──
def _to_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v.strip())
    return None


def _record_key(rec: dict, dataset: str) -> str:
    ti = _to_int(rec.get("total_index"))
    if ti is not None:
        return f"ti:{ti}"
    qi, si = _to_int(rec.get("query_index")), _to_int(rec.get("sample_index"))
    if qi is not None and si is not None and dataset in _DATASET_OFFSETS:
        return f"ti:{_DATASET_OFFSETS[dataset] + qi * 5 + si}"
    rid = rec.get("record_id")
    if isinstance(rid, str) and rid.strip():
        return f"rid:{rid.strip()}"
    raise KeyError(f"no record key for fields {sorted(rec)}")


def _norm_steps(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {str(k): _to_int(v) for k, v in value.items() if _to_int(v) is not None}
    if isinstance(value, list):
        return {str(i): _to_int(v) for i, v in enumerate(value) if _to_int(v) is not None}
    return {}


def _is_failed(rec: dict) -> bool:
    c = rec.get("comment")
    return isinstance(c, str) and c.strip().startswith("llm_annotate_failed:")


def _load_apb(path: Path, dataset: str) -> dict[str, APBRecord]:
    """Load a reference or prediction .jsonl into {key: APBRecord} (last occurrence per key wins,
    matching APB's latest-by-line behavior on the released clean files)."""
    out: dict[str, APBRecord] = {}
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: {e}") from e
            key = _record_key(rec, dataset)
            fl = _to_int(rec.get("final_label"))
            out[key] = APBRecord(
                key=key,
                dataset=dataset,
                data_source=str(rec.get("data_source", dataset)),
                step_labels=_norm_steps(rec.get("step_labels")),
                final_label=fl if fl is not None else 0,
                failed=_is_failed(rec),
            )
    return out


def load_apb_reference(apb_dir: Path) -> dict[str, dict[str, APBRecord]]:
    """{dataset: {key: APBRecord}} for the 4 APB reference files under data/AgentProcessBench/."""
    base = Path(apb_dir).expanduser() / "data" / "AgentProcessBench"
    return {
        ds: _load_apb(base / f"{ds}.jsonl", ds)
        for ds in APB_DATASETS
        if (base / f"{ds}.jsonl").exists()
    }


def reused_llm_entries(apb_dir: Path) -> list[StepVerifierEntry]:
    """One entry per released LLM verifier under eval/results/<MODEL>/. The entry looks each
    prediction up by record key (reused-prediction adapter; missing/failed prediction = no flag)."""
    res = Path(apb_dir).expanduser() / "eval" / "results"
    entries: list[StepVerifierEntry] = []
    by_run: dict[str, dict[str, dict[str, APBRecord]]] = {}
    for mdir in sorted(p for p in res.iterdir() if p.is_dir()):
        for f in sorted(mdir.glob("*.jsonl")):
            if any(p in {"raw", "_raw", "llm_annotations_raw"} for p in f.parts):
                continue
            ds = f.name.split("__", 1)[0]
            if ds not in APB_DATASETS:
                continue
            run = f.name.split("__", 1)[1].removesuffix(".jsonl") if "__" in f.name else f.stem
            by_run.setdefault(run, {})[ds] = _load_apb(f, ds)
    for run, preds_by_ds in sorted(by_run.items()):
        def _mk(preds: dict[str, dict[str, APBRecord]]) -> PredictFn:
            def predict(ref: APBRecord) -> tuple[dict[str, int], int] | None:
                p = preds.get(ref.dataset, {}).get(ref.key)
                if p is None or p.failed:
                    return None
                return p.step_labels, p.final_label
            return predict
        entries.append(StepVerifierEntry(name=run, predict=_mk(preds_by_ds)))
    return entries


def degenerate_entries() -> list[StepVerifierEntry]:
    """The mandatory degenerate rows (prereg §6). const_pass = 'everything is fine' (the all-+1
    floor EXP-0030 flagged); const_fail = 'everything broken'; const_neutral = 'all 0'."""

    def _const(step_c: int, final_c: int) -> PredictFn:
        return lambda ref: ({k: step_c for k in ref.step_labels}, final_c)

    return [
        StepVerifierEntry("degenerate:const_pass(+1)", _const(1, 1), is_degenerate=True),
        StepVerifierEntry("degenerate:const_fail(-1)", _const(-1, -1), is_degenerate=True),
        StepVerifierEntry("degenerate:const_neutral(0)", _const(0, 0), is_degenerate=True),
    ]


# ── metrics ──
@dataclass
class AxisScore:
    """Scores for one entry on one (dataset, axis)."""

    # IF axis (binary: positive = process-failure-present = final_label == -1; final==0 excluded)
    n_fail: int = 0
    if_recall_hits: int = 0  # ref fail & pred fail
    n_ok: int = 0
    if_fire_on_ok: int = 0  # ref ok & pred fail (false alarm)
    # WHERE axis
    n_records: int = 0
    firsterr_hits: int = 0
    # STEP axis (micro + per-class for macro-F1)
    step_total: int = 0
    step_micro_hits: int = 0
    # per-class confusion for macro-F1: cls -> [tp, fp, fn]
    cls_tp: dict[int, int] = field(default_factory=lambda: {c: 0 for c in STEP_CLASSES})
    cls_fp: dict[int, int] = field(default_factory=lambda: {c: 0 for c in STEP_CLASSES})
    cls_fn: dict[int, int] = field(default_factory=lambda: {c: 0 for c in STEP_CLASSES})

    @property
    def if_recall(self) -> float:
        return self.if_recall_hits / self.n_fail if self.n_fail else float("nan")

    @property
    def if_fire_rate(self) -> float:
        return self.if_fire_on_ok / self.n_ok if self.n_ok else float("nan")

    @property
    def if_balanced_acc(self) -> float:
        if not self.n_fail or not self.n_ok:
            return float("nan")
        return (self.if_recall + (1 - self.if_fire_rate)) / 2

    @property
    def firsterr_acc(self) -> float:
        return self.firsterr_hits / self.n_records if self.n_records else float("nan")

    @property
    def step_micro_acc(self) -> float:
        return self.step_micro_hits / self.step_total if self.step_total else float("nan")

    @property
    def step_macro_f1(self) -> float:
        f1s = []
        for c in STEP_CLASSES:
            tp, fp, fn = self.cls_tp[c], self.cls_fp[c], self.cls_fn[c]
            if tp + fn == 0:  # class absent in reference → skip
                continue
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
        return sum(f1s) / len(f1s) if f1s else float("nan")

    def to_dict(self) -> dict:
        """Raw counts (IF/WHERE/STEP) AND the derived metrics, JSON-safe (nan → null)."""
        return {
            "n_fail": self.n_fail,
            "if_recall_hits": self.if_recall_hits,
            "n_ok": self.n_ok,
            "if_fire_on_ok": self.if_fire_on_ok,
            "n_records": self.n_records,
            "firsterr_hits": self.firsterr_hits,
            "step_total": self.step_total,
            "step_micro_hits": self.step_micro_hits,
            "if_recall": nan_to_none(self.if_recall),
            "if_fire_rate": nan_to_none(self.if_fire_rate),
            "if_balanced_acc": nan_to_none(self.if_balanced_acc),
            "firsterr_acc": nan_to_none(self.firsterr_acc),
            "step_micro_acc": nan_to_none(self.step_micro_acc),
            "step_macro_f1": nan_to_none(self.step_macro_f1),
        }


def _first_neg1(steps: dict[str, int]) -> int:
    idxs = [int(k) for k, v in steps.items() if v == -1 and k.lstrip("-").isdigit()]
    return min(idxs) if idxs else -1


def _score_one(ref: dict[str, APBRecord], entry: StepVerifierEntry) -> AxisScore:
    s = AxisScore()
    for rec in ref.values():
        pred = entry.predict(rec)
        # missing/failed prediction = "no flag" (empty steps, final ok)
        psteps, pfinal = (pred if pred is not None else ({}, 1))

        # IF axis (binary; drop ref final == 0)
        if rec.final_label == -1:
            s.n_fail += 1
            s.if_recall_hits += int(pfinal == -1)
        elif rec.final_label == 1:
            s.n_ok += 1
            s.if_fire_on_ok += int(pfinal == -1)

        # WHERE axis
        s.n_records += 1
        s.firsterr_hits += int(_first_neg1(psteps) == _first_neg1(rec.step_labels))

        # STEP axis (micro + per-class)
        for sk, sv in rec.step_labels.items():
            s.step_total += 1
            pv = psteps.get(sk)
            if pv == sv:
                s.step_micro_hits += 1
                s.cls_tp[sv] += 1
            else:
                s.cls_fn[sv] += 1
                if pv in s.cls_fp:
                    s.cls_fp[pv] += 1
    return s


@dataclass
class APBLeaderboardReport:
    # scores[entry][dataset] = AxisScore   (dataset includes synthetic "AVG")
    scores: dict[str, dict[str, AxisScore]]
    entry_names: list[str]
    datasets: list[str]
    degenerate_names: list[str]

    def _best(
        self, dataset: str, metric: Callable[[AxisScore], float], *, degenerate: bool
    ) -> float:
        """Best (max) metric over the degenerate-or-real entries present in this cell; nan if none.
        NaN metric values (axis undefined for that entry) are skipped."""
        vals = []
        for e in self.entry_names:
            if (e in self.degenerate_names) != degenerate or dataset not in self.scores[e]:
                continue
            v = metric(self.scores[e][dataset])
            if v == v:  # not nan
                vals.append(v)
        return max(vals) if vals else float("nan")

    def _best_degen(self, dataset: str, metric: Callable[[AxisScore], float]) -> float:
        return self._best(dataset, metric, degenerate=True)

    def _best_real(self, dataset: str, metric: Callable[[AxisScore], float]) -> float:
        return self._best(dataset, metric, degenerate=False)

    def separation_verdict(self, dataset: str = "AVG") -> dict[str, str]:
        """Prereg V2/V3 + §6: per axis, do the real entries separate from the best degenerate by a
        fixed margin? Returns {axis: 'discriminating'|'too-easy'|'n/a'}."""
        margin = 0.05

        def verdict(metric: Callable[[AxisScore], float], too_easy: str = "too-easy") -> str:
            best_real = self._best_real(dataset, metric)
            best_degen = self._best_degen(dataset, metric)
            if best_real != best_real or best_degen != best_degen:  # nan
                return "n/a"
            return "discriminating" if best_real - best_degen > margin else too_easy

        # STEP micro vs macro — EXP-0030: macro separates where micro is gamed by the +1 majority.
        return {
            "IF_balanced_acc": verdict(lambda s: s.if_balanced_acc),
            "STEP_micro_acc": verdict(lambda s: s.step_micro_acc, "too-easy(use macro-F1)"),
            "STEP_macro_f1": verdict(lambda s: s.step_macro_f1),
            "WHERE_firsterr_acc": verdict(lambda s: s.firsterr_acc),
        }

    def render(self, dataset: str = "AVG") -> str:
        lines = [
            f"══ APB strata-1+3 process-judgment leaderboard · cell={dataset} ══",
            "  IF = process-failure (R=recall-on-fail↑ Fire=firing-on-ok↓ Bal=bal-acc↑)",
            "  WHERE = FirstErrAcc↑ · STEP = micro | macro-F1↑ (macro de-games the +1 majority)",
            "  † = degenerate baseline (must NOT beat real entries — prereg §6)\n",
        ]

        def _sort_key(e: str) -> float:
            v = self.scores[e][dataset].if_balanced_acc
            return v if v == v else -1.0  # nan → sort last

        order = sorted(
            (e for e in self.entry_names if dataset in self.scores[e]),
            key=_sort_key,
            reverse=True,
        )
        lines.append(
            f"  {'entry':<42} {'IF_R':>6} {'IF_Fire':>8} {'IF_Bal':>7}  "
            f"{'WHERE':>6}  {'STEP_mic':>8} {'STEP_mac':>8}"
        )
        for e in order:
            s = self.scores[e][dataset]
            d = "†" if e in self.degenerate_names else " "

            def _f(x: float) -> str:
                return "  n/a " if x != x else f"{x:5.1%}"
            lines.append(
                f"  {d}{e:<41} {_f(s.if_recall):>6} {_f(s.if_fire_rate):>8} "
                f"{_f(s.if_balanced_acc):>7}  {_f(s.firsterr_acc):>6}  "
                f"{_f(s.step_micro_acc):>8} {_f(s.step_macro_f1):>8}"
            )
        lines.append("\n  ── anti-triviality separation verdict (prereg V2/V3 + §6) ──")
        for axis, verdict in self.separation_verdict(dataset).items():
            lines.append(f"    {axis:<22} {verdict}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """The full APB leaderboard as a JSON-safe dict: scores[entry][dataset] with raw + derived
        numbers, the degenerate flags, and the per-dataset separation verdicts. json.dumps-ready."""
        return {
            "schema": "mtm.apb_leaderboard.v1",
            "entry_names": list(self.entry_names),
            "datasets": list(self.datasets),
            "degenerate_names": list(self.degenerate_names),
            "separation_verdict": {ds: self.separation_verdict(ds) for ds in self.datasets},
            "scores": {
                entry: {ds: s.to_dict() for ds, s in by_ds.items()}
                for entry, by_ds in self.scores.items()
            },
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize ``to_dict()`` to a JSON string (indent=None for a compact single line)."""
        return json.dumps(self.to_dict(), indent=indent)


def score_apb_leaderboard(
    reference: dict[str, dict[str, APBRecord]],
    entries: list[StepVerifierEntry],
) -> APBLeaderboardReport:
    """Score every entry per dataset + an AVG (micro-pooled) cell."""
    datasets = list(reference.keys())
    scores: dict[str, dict[str, AxisScore]] = {}
    for entry in entries:
        scores[entry.name] = {}
        agg = AxisScore()
        for ds in datasets:
            s = _score_one(reference[ds], entry)
            scores[entry.name][ds] = s
            # pool into AVG
            agg.n_fail += s.n_fail
            agg.if_recall_hits += s.if_recall_hits
            agg.n_ok += s.n_ok
            agg.if_fire_on_ok += s.if_fire_on_ok
            agg.n_records += s.n_records
            agg.firsterr_hits += s.firsterr_hits
            agg.step_total += s.step_total
            agg.step_micro_hits += s.step_micro_hits
            for c in STEP_CLASSES:
                agg.cls_tp[c] += s.cls_tp[c]
                agg.cls_fp[c] += s.cls_fp[c]
                agg.cls_fn[c] += s.cls_fn[c]
        scores[entry.name]["AVG"] = agg
    return APBLeaderboardReport(
        scores=scores,
        entry_names=[e.name for e in entries],
        datasets=[*datasets, "AVG"],
        degenerate_names=[e.name for e in entries if e.is_degenerate],
    )


__all__ = [
    "APB_DATASETS",
    "STEP_CLASSES",
    "APBRecord",
    "StepVerifierEntry",
    "AxisScore",
    "APBLeaderboardReport",
    "load_apb_reference",
    "reused_llm_entries",
    "degenerate_entries",
    "score_apb_leaderboard",
]
