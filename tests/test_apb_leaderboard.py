"""Tests for the APB strata-1+3 leaderboard metric math (argus_benchmark.apb_leaderboard).

The instrument must be provably correct BEFORE we trust real-data numbers (CLAUDE.md §4 — verify,
don't vibe). These tests hand-compute every metric on a tiny synthetic fixture so a regression in
IF/WHERE/STEP scoring or the separation verdict fails loudly.
"""

from __future__ import annotations

from mtm_bench.apb_leaderboard import (
    APBRecord,
    StepVerifierEntry,
    degenerate_entries,
    score_apb_leaderboard,
)


def _ref() -> dict[str, dict[str, APBRecord]]:
    """3 records in one synthetic dataset 'd':
      r1: steps {0:+1, 1:-1, 2:+1}, final -1   (fail; first-err idx 1)
      r2: steps {0:+1, 1:+1},       final +1   (ok; no -1)
      r3: steps {0:-1},             final  0   (excluded from IF binary; first-err idx 0)
    """
    return {
        "d": {
            "ti:0": APBRecord("ti:0", "d", "d", {"0": 1, "1": -1, "2": 1}, -1),
            "ti:1": APBRecord("ti:1", "d", "d", {"0": 1, "1": 1}, 1),
            "ti:2": APBRecord("ti:2", "d", "d", {"0": -1}, 0),
        }
    }


def _perfect_entry() -> StepVerifierEntry:
    # A perfect verifier just echoes each record's own labels (works across any dataset).
    return StepVerifierEntry("perfect", lambda rec: (rec.step_labels, rec.final_label))


def test_perfect_entry_scores_unity():
    rep = score_apb_leaderboard(_ref(), [_perfect_entry()])
    s = rep.scores["perfect"]["d"]
    # IF: 1 fail (r1), 1 ok (r2); r3 (final 0) excluded.
    assert s.n_fail == 1 and s.n_ok == 1
    assert s.if_recall == 1.0 and s.if_fire_rate == 0.0
    assert s.if_balanced_acc == 1.0
    # WHERE: first-(-1) index matches on all 3 records (r1→1, r2→-1/none, r3→0).
    assert s.firsterr_hits == 3 and s.firsterr_acc == 1.0
    # STEP: 6 labels total, all matched.
    assert s.step_total == 6 and s.step_micro_hits == 6
    assert s.step_micro_acc == 1.0
    assert abs(s.step_macro_f1 - 1.0) < 1e-9


def test_const_pass_degenerate_floor():
    """const_pass(+1) predicts every step +1, final +1 — the EXP-0030 'all-+1' degenerate.
    Hand math on the fixture: 4 of 6 step labels are +1 → micro 4/6."""
    rep = score_apb_leaderboard(_ref(), degenerate_entries())
    cp = rep.scores["degenerate:const_pass(+1)"]["d"]
    assert cp.step_total == 6
    assert cp.step_micro_hits == 4  # steps r1[0],r1[2],r2[0],r2[1] are +1
    assert abs(cp.step_micro_acc - 4 / 6) < 1e-9
    # IF: predicts final +1 always → recall on the 1 fail = 0; firing on the 1 ok = 0.
    assert cp.if_recall == 0.0 and cp.if_fire_rate == 0.0
    # WHERE: const_pass has no -1 → predicted first-err -1 (none); matches only r2 (also none).
    assert cp.firsterr_hits == 1  # only r2
    # macro-F1 should be POOR even though micro is 0.67 (the whole point): only the +1 class has TP.
    assert cp.step_macro_f1 < cp.step_micro_acc


def test_const_fail_degenerate():
    rep = score_apb_leaderboard(_ref(), degenerate_entries())
    cf = rep.scores["degenerate:const_fail(-1)"]["d"]
    # predicts every step -1 → matches the 2 ref -1 labels (r1[1], r3[0]).
    assert cf.step_micro_hits == 2
    # IF: predicts final -1 always → recall on fail = 1.0 BUT fires on the ok trace = 1.0.
    assert cf.if_recall == 1.0 and cf.if_fire_rate == 1.0
    assert cf.if_balanced_acc == 0.5  # (1 + (1-1))/2 — exactly chance, the degenerate signature
    # WHERE: predicted first -1 = index 0 for every record; matches only r3 (true first -1 = 0).
    assert cf.firsterr_hits == 1


def test_missing_prediction_counts_as_no_flag():
    """An entry with no prediction for a record = 'no flag' (final ok, empty steps)."""
    entry = StepVerifierEntry("empty", lambda rec: None)
    rep = score_apb_leaderboard(_ref(), [entry])
    s = rep.scores["empty"]["d"]
    assert s.if_recall == 0.0  # never flags the fail
    assert s.if_fire_rate == 0.0  # never fires on ok
    assert s.step_micro_hits == 0  # empty steps match nothing


def test_avg_pools_across_datasets():
    ref = _ref()
    ref["d2"] = {  # a second dataset, 1 record
        "ti:250": APBRecord("ti:250", "d2", "d2", {"0": 1}, 1),
    }
    rep = score_apb_leaderboard(ref, [_perfect_entry()])
    avg = rep.scores["perfect"]["AVG"]
    # pooled step_total = 6 (d) + 1 (d2) = 7
    assert avg.step_total == 7
    assert "AVG" in rep.datasets


def test_separation_verdict_flags_too_easy_on_micro():
    """With ONLY degenerates present, no real entry exists → micro/where verdicts must not claim
    'discriminating' (best_real over an empty set is nan → not > floor)."""
    rep = score_apb_leaderboard(_ref(), degenerate_entries())
    v = rep.separation_verdict("d")
    # no real entries → IF is n/a; step micro/macro/where can't be 'discriminating'
    assert v["IF_balanced_acc"] == "n/a"
    assert "discriminating" not in v["STEP_micro_acc"]
    assert "discriminating" not in v["WHERE_firsterr_acc"]


def test_separation_verdict_discriminates_with_perfect_entry():
    entries = [*degenerate_entries(), _perfect_entry()]
    rep = score_apb_leaderboard(_ref(), entries)
    v = rep.separation_verdict("d")
    # perfect entry beats every degenerate on all axes
    assert v["IF_balanced_acc"] == "discriminating"
    assert v["STEP_macro_f1"] == "discriminating"
    assert v["WHERE_firsterr_acc"] == "discriminating"


def test_json_export_round_trips_with_derived_and_verdicts():
    """to_json() must be valid JSON, carry derived metrics + the separation verdicts, and map an
    undefined axis (nan) to null. const_pass on the fixture has if_recall/if_fire but a perfect
    entry gives clean derived numbers to check."""
    import json

    rep = score_apb_leaderboard(_ref(), [*degenerate_entries(), _perfect_entry()])
    back = json.loads(rep.to_json())
    assert back["schema"] == "mtm.apb_leaderboard.v1"
    assert "AVG" in back["datasets"]
    assert "degenerate:const_pass(+1)" in back["degenerate_names"]
    perfect_d = back["scores"]["perfect"]["d"]
    assert perfect_d["if_balanced_acc"] == 1.0 and perfect_d["step_micro_acc"] == 1.0
    # separation verdicts are serialized per dataset
    assert back["separation_verdict"]["d"]["IF_balanced_acc"] == "discriminating"


def test_json_export_maps_undefined_axis_to_null():
    """A dataset where every record has final_label == 0 → n_fail == n_ok == 0 → if_balanced_acc
    is nan → must serialize as null (no NaN token in the JSON)."""
    import json

    ref = {"d0": {"ti:0": APBRecord("ti:0", "d0", "d0", {"0": 0}, 0)}}
    rep = score_apb_leaderboard(ref, [_perfect_entry()])
    cell = json.loads(rep.to_json())["scores"]["perfect"]["d0"]
    assert cell["if_balanced_acc"] is None and cell["if_recall"] is None
