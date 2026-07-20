"""Tests for the detector-as-unit meta-eval leaderboard (EXP-0020 charter step 8).

The leaderboard is the shippable instrument: it scores arbitrary detector entries against a
corrupt-success gold cell, two numbers each (recall-on-corrupt-success + firing-rate-on-clean),
stratified by R11 certificate tier, never pooled. These tests pin the load-bearing properties:
the two numbers are computed correctly, tiers are NOT pooled, the tautology flag fires when an
entry's predicate is the gold mechanism (the EXP-0015 lesson), and a missing trace is counted but
not scored (never silently dropped).
"""

from __future__ import annotations

from mtm_bench import GoldItem, detector_entry, score_leaderboard
from mtm_bench.leaderboard import _CallableEntry
from mtm_bench.schema import Outcome, OutcomeStatus, Span, SpanKind, TaskSpec, Trace


def _trace(tid: str) -> Trace:
    """A minimal SUCCESS trace; the entries below decide flags from the trace_id, so the content
    is irrelevant — these tests exercise the SCORING, not any detector."""
    return Trace(
        trace_id=tid,
        agent_id="a",
        task_type="airline",
        spec=TaskSpec(goal="t", allowed_tools=[]),
        spans=[Span(span_id="s0", index=0, kind=SpanKind.USER_MSG, content="hi")],
        outcome=Outcome(status=OutcomeStatus.SUCCESS, final_response=None),
        label=None,
        meta={"domain": "airline"},
    )


def _gold(tid: str, tier: str, corrupt: bool, source: str = "human_blind") -> GoldItem:
    return GoldItem(trace_id=tid, cell="gpt-4.1|airline", tier=tier,
                    corrupt_success=corrupt, source=source)


def test_two_numbers_computed_correctly() -> None:
    # 2 corrupt + 2 clean in one tier. An entry that flags exactly one corrupt and one clean
    # must report recall 0.5 and clean-fire-rate 0.5.
    gold = [
        _gold("c1", "config_only", True), _gold("c2", "config_only", True),
        _gold("k1", "config_only", False), _gold("k2", "config_only", False),
    ]
    traces = [_trace(t) for t in ("c1", "c2", "k1", "k2")]
    flags = {"c1", "k1"}
    entry = _CallableEntry("e", lambda t, s: t.trace_id in flags)
    rep = score_leaderboard(gold, traces, [entry])
    cs = rep.scores["e"]["gpt-4.1|airline"]["config_only"]
    assert cs.n_corrupt == 2 and cs.recall_hits == 1 and cs.recall == 0.5
    assert cs.n_clean == 2 and cs.clean_fires == 1 and cs.clean_fire_rate == 0.5


def test_tiers_not_pooled() -> None:
    # Same entry scores DIFFERENTLY across tiers — the report must keep them separate.
    gold = [
        _gold("a", "config_only", True),       # flagged → config recall 1.0
        _gold("b", "truth_oracle", True),       # NOT flagged → truth_oracle recall 0.0
    ]
    traces = [_trace("a"), _trace("b")]
    entry = _CallableEntry("e", lambda t, s: t.trace_id == "a")
    rep = score_leaderboard(gold, traces, [entry])
    cfg = rep.scores["e"]["gpt-4.1|airline"]["config_only"]
    tru = rep.scores["e"]["gpt-4.1|airline"]["truth_oracle"]
    assert cfg.recall == 1.0
    assert tru.recall == 0.0
    # Tiers present and ordered by cost.
    assert rep.tiers == ["config_only", "truth_oracle"]


def test_tautology_flag_fires_on_shared_mechanism() -> None:
    # An entry tautological on the gold source must be flagged so its number isn't touted.
    gold = [_gold("a", "config_only", True, source="step_id_fact")]
    entry = _CallableEntry("argus_typed", lambda t, s: True,
                           tautological_on=frozenset({"step_id_fact"}))
    rep = score_leaderboard(gold, [_trace("a")], [entry])
    cs = rep.scores["argus_typed"]["gpt-4.1|airline"]["config_only"]
    assert cs.tautological is True
    # A DIFFERENT-mechanism entry on the same gold is NOT tautological.
    judge = _CallableEntry("judge", lambda t, s: True, tautological_on=frozenset({"human_blind"}))
    rep2 = score_leaderboard(gold, [_trace("a")], [judge])
    assert rep2.scores["judge"]["gpt-4.1|airline"]["config_only"].tautological is False


def test_missing_trace_counted_not_scored() -> None:
    gold = [_gold("present", "config_only", True), _gold("absent", "config_only", True)]
    traces = [_trace("present")]  # 'absent' trace not supplied
    entry = _CallableEntry("e", lambda t, s: True)
    rep = score_leaderboard(gold, traces, [entry])
    assert rep.n_gold == 2
    assert rep.n_scored == 1
    # Only the present trace enters the denominator.
    cs = rep.scores["e"]["gpt-4.1|airline"]["config_only"]
    assert cs.n_corrupt == 1 and cs.recall_hits == 1


def test_native_oracle_floor_is_zero_recall() -> None:
    # The native outcome oracle, by construction, NEVER flags an oracle=SUCCESS trace as a
    # failure (that's the cell it's blind to) — recall on corrupt-success is 0. This is the
    # benchmark's whole reason to exist; the test pins it.
    gold = [_gold("c", "config_only", True), _gold("k", "config_only", False)]
    oracle = _CallableEntry("native_oracle", lambda t, s: False)  # never fires on SUCCESS
    rep = score_leaderboard(gold, [_trace("c"), _trace("k")], [oracle])
    cs = rep.scores["native_oracle"]["gpt-4.1|airline"]["config_only"]
    assert cs.recall == 0.0
    assert cs.clean_fire_rate == 0.0


def test_detector_entry_wraps_a_detector_object() -> None:
    # detector_entry() flags iff the wrapped detector emits ≥1 signal from .analyze(trace, spec),
    # and treats an empty result (the detector's own abstention) as a non-flag.
    class _StubDetector:
        def __init__(self, signals: list) -> None:
            self._signals = signals

        def analyze(self, trace, spec):
            return self._signals

    t = _trace("stub")
    firing = detector_entry("stub_firing", _StubDetector(["signal"]))
    silent = detector_entry("stub_silent", _StubDetector([]))
    assert firing.predict(t, t.spec) is True
    assert silent.predict(t, t.spec) is False


def test_json_export_round_trips_with_derived_numbers() -> None:
    # to_json() must be valid JSON (json.loads succeeds → no bare NaN token) and must carry the
    # DERIVED two numbers, not just raw counts (asdict() would drop them). Entry flags 1/2 corrupt
    # and 1/2 clean → recall 0.5, fire-on-clean 0.5.
    import json

    gold = [
        _gold("c1", "config_only", True), _gold("c2", "config_only", True),
        _gold("k1", "config_only", False), _gold("k2", "config_only", False),
    ]
    flag_c1_k1 = _CallableEntry("half", lambda t, s: t.trace_id in {"c1", "k1"})
    rep = score_leaderboard(gold, [_trace(x) for x in ("c1", "c2", "k1", "k2")], [flag_c1_k1])
    back = json.loads(rep.to_json())
    assert back["schema"] == "mtm.leaderboard.v1"
    assert back["n_gold"] == 4 and back["n_scored"] == 4
    cell = back["scores"]["half"]["gpt-4.1|airline"]["config_only"]
    assert cell["recall"] == 0.5 and cell["clean_fire_rate"] == 0.5
    assert cell["recall_ci"] is not None and len(cell["recall_ci"]) == 2


def test_json_export_maps_nan_to_null_on_empty_denominator() -> None:
    # A tier with corrupt-only gold → n_clean == 0 → clean_fire_rate is nan → must serialize as
    # null (JSON has no NaN), and the CI must be null too (undefined, not (0,0)).
    import json

    gold = [_gold("c1", "config_only", True)]
    rep = score_leaderboard(gold, [_trace("c1")], [_CallableEntry("f", lambda t, s: True)])
    cell = json.loads(rep.to_json())["scores"]["f"]["gpt-4.1|airline"]["config_only"]
    assert cell["clean_fire_rate"] is None and cell["clean_ci"] is None
    assert cell["recall"] == 1.0
