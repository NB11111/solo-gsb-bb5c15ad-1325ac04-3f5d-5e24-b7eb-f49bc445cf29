"""Unit tests for the rule fault sequence pairing algorithm (app/pairing.py).

The matching is one-to-one and time-order-preserving; it first maximizes the
pair count, then minimizes the total lag, and breaks remaining ties stably by
fault id.
"""

import pytest

from app.pairing import FaultAnchor, pair_faults

BASE = 1_790_121_600.0  # 2026-09-23T00:00:00Z


def anchors(specs):
    """Build FaultAnchors from (id, start_offset) or (id, start, end) triples."""
    out = []
    for spec in specs:
        fid, start = spec[0], spec[1]
        end = spec[2] if len(spec) > 2 else start + 60.0
        out.append(FaultAnchor(id=fid, start=BASE + start, end=BASE + end))
    return out


def pair_ids(result):
    return [(p.antecedent.id, p.subsequent.id) for p in result.pairs]


# --------------------------------------------------------------------------- #
# admissibility
# --------------------------------------------------------------------------- #


def test_basic_pair_and_lag():
    result = pair_faults(anchors([(1, 0)]), anchors([(2, 45)]), 60)
    assert pair_ids(result) == [(1, 2)]
    assert result.pairs[0].lag_seconds == pytest.approx(45.0)
    assert result.total_lag_seconds == pytest.approx(45.0)
    assert result.unmatched_antecedent == []
    assert result.unmatched_subsequent == []


def test_subsequent_must_start_strictly_later():
    # identical starts are not "later": no pair
    result = pair_faults(anchors([(1, 0)]), anchors([(2, 0)]), 60)
    assert result.pair_count == 0
    assert [f.id for f in result.unmatched_antecedent] == [1]
    assert [f.id for f in result.unmatched_subsequent] == [2]
    # an earlier subsequent start is rejected too
    result = pair_faults(anchors([(1, 100)]), anchors([(2, 50)]), 60)
    assert result.pair_count == 0


def test_max_lag_boundary_is_inclusive():
    # lag exactly equal to max_lag pairs; one second beyond does not
    result = pair_faults(anchors([(1, 0)]), anchors([(2, 120)]), 120)
    assert pair_ids(result) == [(1, 2)]
    result = pair_faults(anchors([(1, 0)]), anchors([(2, 121)]), 120)
    assert result.pair_count == 0


def test_each_interval_pairs_at_most_once():
    result = pair_faults(anchors([(1, 0), (2, 10)]), anchors([(3, 20)]), 60)
    assert result.pair_count == 1  # the single subsequent pairs only once
    result = pair_faults(anchors([(1, 0)]), anchors([(2, 10), (3, 20)]), 60)
    assert result.pair_count == 1  # the single antecedent pairs only once


# --------------------------------------------------------------------------- #
# optimality: count first, then total lag, then fault-id tie-break
# --------------------------------------------------------------------------- #


def test_maximizes_pair_count():
    a = anchors([(1, 0), (2, 100)])
    b = anchors([(3, 10), (4, 110)])
    result = pair_faults(a, b, 60)
    assert pair_ids(result) == [(1, 3), (2, 4)]


def test_count_beats_total_lag():
    # matching both pairs costs more total lag than matching only the close
    # one, but the pair count wins
    a = anchors([(1, 0), (2, 100)])
    b = anchors([(3, 50), (4, 150)])
    result = pair_faults(a, b, 60)
    assert pair_ids(result) == [(1, 3), (2, 4)]
    assert result.total_lag_seconds == pytest.approx(100.0)


def test_minimizes_total_lag_over_earliest_antecedent():
    # a plain "earliest antecedent first" greedy would pair a1; the smaller
    # total lag pairs a2 instead (same pair count)
    a = anchors([(1, 0), (2, 1)])
    b = anchors([(3, 2)])
    result = pair_faults(a, b, 10)
    assert pair_ids(result) == [(2, 3)]
    assert result.total_lag_seconds == pytest.approx(1.0)
    assert [f.id for f in result.unmatched_antecedent] == [1]


def test_time_order_is_preserved():
    # the only 2-pair non-crossing matching pairs them in order
    a = anchors([(1, 0), (2, 1)])
    b = anchors([(3, 2), (4, 3)])
    result = pair_faults(a, b, 10)
    assert pair_ids(result) == [(1, 3), (2, 4)]


def test_tie_broken_by_fault_id():
    # identical starts on the antecedent side: same count and lag either way,
    # so the smaller antecedent fault id wins
    a = anchors([(2, 0), (1, 0)])
    b = anchors([(3, 30)])
    result = pair_faults(a, b, 60)
    assert pair_ids(result) == [(1, 3)]
    # identical starts on the subsequent side: smaller subsequent id wins
    a = anchors([(1, 0)])
    b = anchors([(5, 30), (4, 30)])
    result = pair_faults(a, b, 60)
    assert pair_ids(result) == [(1, 4)]


def test_tie_break_is_stable_across_input_order():
    # two antecedents compete for one subsequent with identical starts and
    # lags: the smaller antecedent fault id wins regardless of input order
    specs_a = [(7, 0), (3, 0)]
    specs_b = [(9, 10)]
    forward = pair_faults(anchors(specs_a), anchors(specs_b), 60)
    backward = pair_faults(anchors(specs_a[::-1]), anchors(specs_b[::-1]), 60)
    assert pair_ids(forward) == pair_ids(backward) == [(3, 9)]
    # fully symmetric two-by-two: everything pairs, in time order
    both = pair_faults(anchors([(7, 0), (3, 0)]), anchors([(9, 10), (2, 10)]), 60)
    assert pair_ids(both) == [(3, 2), (7, 9)]


# --------------------------------------------------------------------------- #
# result payload
# --------------------------------------------------------------------------- #


def test_unmatched_and_rates():
    a = anchors([(1, 0), (2, 100), (3, 200)])
    b = anchors([(4, 10), (5, 110)])
    result = pair_faults(a, b, 60)
    assert pair_ids(result) == [(1, 4), (2, 5)]
    assert [f.id for f in result.unmatched_antecedent] == [3]
    assert result.unmatched_subsequent == []
    payload = result.to_dict()
    assert payload["pair_count"] == 2
    assert payload["antecedent"]["total"] == 3
    assert payload["antecedent"]["matched"] == 2
    assert payload["antecedent"]["unmatched"] == 1
    assert payload["antecedent"]["match_rate"] == pytest.approx(2 / 3)
    assert payload["subsequent"]["match_rate"] == pytest.approx(1.0)
    assert payload["antecedent"]["unmatched_faults"] == [
        {"fault_id": 3, "start": BASE + 200, "end": BASE + 260}
    ]
    pair = payload["pairs"][0]
    assert pair["antecedent_fault_id"] == 1
    assert pair["subsequent_fault_id"] == 4
    assert pair["lag_seconds"] == pytest.approx(10.0)


def test_empty_inputs():
    result = pair_faults([], [], 60)
    payload = result.to_dict()
    assert payload["pair_count"] == 0
    assert payload["antecedent"]["match_rate"] == 0.0
    assert payload["subsequent"]["match_rate"] == 0.0
    result = pair_faults(anchors([(1, 0)]), [], 60)
    assert result.pair_count == 0
    assert [f.id for f in result.unmatched_antecedent] == [1]


def test_invalid_max_lag_rejected():
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            pair_faults(anchors([(1, 0)]), anchors([(2, 10)]), bad)
