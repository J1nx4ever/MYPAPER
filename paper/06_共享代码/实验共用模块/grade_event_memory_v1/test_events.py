"""Synthetic acceptance only: never opens RPC or starts a simulator/optimizer."""

from dataclasses import replace
import math

import pytest

from .events import (
    ContractError, Event, build_memory, grade_reward, validate_event,
)


def synthetic_events():
    """A partial FIFO batch survives W0 -> W1; W2/W3 are empty."""
    return [
        Event(1, "DECISION", 0, 0, request_id="q0", decision_sequence=0),
        Event(2, "UNLOAD", 1, 0, "b1", 0, 0, 40, 10.8, .27, True),
        Event(3, "PROCESS", 10, 0, "b1", 0, 0, 10, 2.7, .27, True, 1, 10),
        Event(4, "PROCESS", 20, 0, "b1", 0, 0, 10, 2.7, .27, True, 10, 20),
        Event(5, "DECISION", 20, 0, request_id="q1", decision_sequence=1),
        Event(6, "PROCESS", 30, 0, "b1", 0, 0, 10, 2.7, .27, True, 20, 30),
        Event(7, "WINDOW_CLOSE", 7200, 0, window_q=30, window_m=8.1,
              window_grade=.27, window_grade_valid=True, grade_reward_raw=1),
        Event(8, "DECISION", 7200, 1, request_id="q2", decision_sequence=2),
        Event(9, "PROCESS", 7210, 1, "b1", 0, 0, 10, 2.7, .27, True, 7200, 7210),
        Event(10, "WINDOW_CLOSE", 14400, 1, window_q=10, window_m=2.7,
              window_grade=.27, window_grade_valid=True, grade_reward_raw=1),
        Event(11, "WINDOW_CLOSE", 21600, 2, grade_reward_raw=-1),
        Event(12, "WINDOW_CLOSE", 28800, 3, grade_reward_raw=-1),
        Event(13, "END", 28800, 3, request_id="end", decision_sequence=3),
    ]


def test_conservative_compression_partial_fifo_cross_window_and_grade_validity():
    tape = build_memory(synthetic_events())
    assert len(tape.events) == 13 and len(tape.tokens) == 8
    compressed = tape.tokens[1]
    assert (compressed.seq_first, compressed.seq_last) == (3, 4)
    assert (compressed.time_first, compressed.time) == (10, 20)
    assert (compressed.interval_start, compressed.interval_end) == (1, 20)
    assert (compressed.tons, compressed.metal) == (20, 5.4)
    assert not compressed.grade_before_valid and compressed.grade_before == 0
    assert compressed.grade_after_valid and compressed.grade_after == pytest.approx(.27)
    assert sum(token.tons for token in tape.tokens if token.kind == "PROCESS") == 40
    assert sum(token.metal for token in tape.tokens if token.kind == "PROCESS") == pytest.approx(10.8)
    assert [s.tons for s in tape.settlements] == [30, 10, 0, 0]
    assert [s.cost for s in tape.settlements] == [0, 0, 2, 2]
    assert tape.tokens[3].kind == "WINDOW_CLOSE"
    assert tape.tokens[3].q_after == 0 and not tape.tokens[3].grade_after_valid
    # The history is not reset at W0; the W1 query still sees W0 tokens.
    assert [q.current_cursor for q in tape.queries] == [0, 2, 4]
    assert [q.next_cursor for q in tape.queries] == [2, 4, 8]
    assert [q.terminal_next for q in tape.queries] == [False, False, True]


def test_same_timestamp_uses_append_order_and_exclusive_cursors():
    tape = build_memory(synthetic_events())
    q = tape.queries[2]
    assert q.current_seq == 8
    assert tape.tokens[q.current_cursor - 1].seq_last == 7
    assert all(token.seq_last < q.current_seq for token in tape.tokens[:q.current_cursor])
    for query in tape.queries:
        assert all(token.seq_last < query.current_seq
                   for token in tape.tokens[:query.current_cursor])
        assert all(token.seq_first > query.current_seq
                   for token in tape.tokens[query.current_cursor:query.next_cursor])
    # DECISION at seq5 is a compression barrier even with same batch/window.
    assert [token.seq_first for token in tape.tokens if token.kind == "PROCESS"] == [3, 6, 9]


def test_record_visibility_and_physical_interval_are_distinct():
    events = synthetic_events()
    events[2] = replace(events[2], time=20)
    events[3] = replace(events[3], time=20)
    tape = build_memory(events)
    token = tape.tokens[1]
    assert token.time_first == 20 and token.interval_start == 1
    assert token.time == 20 and token.interval_end == 20


def test_unload_barrier_retains_fifo_and_unknown_categories():
    events = synthetic_events()
    # Replace original one 40t batch with two 20t batches separated by UNLOAD.
    events[1] = replace(events[1], tons=20, metal=5.4, source_id=-1, truck_id=-1)
    events[2] = replace(events[2], source_id=-1, truck_id=-1)
    events[3] = replace(events[3], source_id=-1, truck_id=-1)
    events.insert(4, Event(5, "UNLOAD", 20, 0, "b2", 1, 1, 20, 5.4, .27, True))
    events = [replace(e, seq=i + 1) for i, e in enumerate(events)]
    events[6] = replace(events[6], batch_id="b2", source_id=1, truck_id=1)
    events[9] = replace(events[9], batch_id="b2", source_id=1, truck_id=1)
    tape = build_memory(events)
    assert tape.tokens[0].source_id == -1
    assert any(t.kind == "UNLOAD" and t.source_id == 1 for t in tape.tokens)


@pytest.mark.parametrize("grade,cost", [(.27, 0), (.26, 1), (.28, 1), (.255, 1.5), (.29, 2), (.4, 2)])
def test_existing_reward_formula_and_saturation(grade, cost):
    observed, valid, raw = grade_reward(100, 100 * grade)
    assert observed == pytest.approx(grade) and valid
    assert 1 - raw == pytest.approx(cost)
    assert grade_reward(0, 0) == (0, False, -1)


@pytest.mark.parametrize("index,changes,match", [
    (3, {"seq": 3}, "seq"),
    (3, {"time": 9}, "future physical|visibility"),
    (3, {"interval_start": 9}, "overlap"),
    (2, {"interval_start": 0}, "before material"),
    (2, {"interval_end": 7201, "time": 7201}, "crosses physical"),
    (2, {"tons": 41, "metal": 11.07}, "exceeds remaining"),
    (2, {"batch_id": "unknown"}, "FIFO"),
    (2, {"source_id": 1}, "source/truck"),
    (2, {"grade": .28, "metal": 2.8}, "differs from UNLOAD"),
    (2, {"metal": 3}, "metal/grade"),
    (1, {"source_id": -2}, "categorical"),
    (1, {"grade_valid": False}, "grade/tons"),
    (6, {"grade_reward_raw": 0}, "formula"),
    (6, {"window_q": 31, "window_m": 8.37}, "conserve"),
    (6, {"window": 1, "time": 14400}, "active/unsettled"),
    (7, {"request_id": "q0"}, "duplicate request"),
    (7, {"decision_sequence": 9}, "contiguous"),
    (7, {"batch_id": "future"}, "nonmaterial"),
    (7, {"window_q": 40}, "non-settlement"),
    (10, {"window_grade": .27, "window_grade_valid": True}, "validity"),
    (12, {"time": 28799}, "true shift end"),
    (0, {"seq": True}, "integer"),
    (1, {"grade": math.nan}, "nonfinite"),
])
def test_bad_input_is_rejected(index, changes, match):
    events = synthetic_events()
    events[index] = replace(events[index], **changes)
    with pytest.raises(ContractError, match=match):
        build_memory(events)


def test_missing_duplicate_settlement_and_abnormal_terminal_rejected():
    events = synthetic_events()
    with pytest.raises(ContractError, match="normal END missing"):
        build_memory(events[:-1])
    with pytest.raises(ContractError, match="four settlements"):
        build_memory(events[:11] + events[12:])
    with pytest.raises(ContractError, match="event after END"):
        build_memory(events + [replace(events[-1], seq=14, request_id="again", decision_sequence=4)])
    duplicated = events[:7] + [replace(events[6], seq=8)] + [replace(e, seq=e.seq + 1) for e in events[7:]]
    with pytest.raises(ContractError, match="active/unsettled"):
        build_memory(duplicated)


def test_fixed_fields_round_trip_and_no_silent_future_fill():
    value = synthetic_events()[1]
    assert Event.from_dict(value.to_dict()) == value
    with pytest.raises(ContractError, match="fields"):
        Event.from_dict({**value.to_dict(), "future_grade": .27})
    missing = value.to_dict()
    del missing["grade_valid"]
    with pytest.raises(ContractError, match="fields"):
        Event.from_dict(missing)
    with pytest.raises(ContractError, match="bool"):
        validate_event(replace(value, grade_valid=1))


def test_consecutive_canonical_seq_gap_is_preserved_as_compression_barrier():
    events = synthetic_events()
    events = [replace(e, seq=e.seq + (1 if e.seq >= 4 else 0)) for e in events]
    tape = build_memory(events)
    assert len(tape.tokens) == 9
    assert tape.tokens[1].seq_first == tape.tokens[1].seq_last == 3
    assert tape.tokens[2].seq_first == tape.tokens[2].seq_last == 5


def test_unload_interleaving_and_physical_gap_both_block_compression():
    events = synthetic_events()
    events.insert(3, Event(4, "UNLOAD", 10, 0, "pending", 1, 1, 1, .3, .3, True))
    events = [replace(e, seq=i + 1) for i, e in enumerate(events)]
    tape = build_memory(events)
    assert len(tape.tokens) == 10  # UNLOAD token + no merge of the two PROCESS records
    assert tape.tokens[1].seq_first == tape.tokens[1].seq_last == 3
    assert tape.tokens[2].kind == "UNLOAD"
    assert tape.tokens[3].seq_first == tape.tokens[3].seq_last == 5
    gap = synthetic_events()
    gap[3] = replace(gap[3], interval_start=11)
    assert len(build_memory(gap).tokens) == 9


def test_duplicate_batch_or_process_before_unload_rejected():
    events = synthetic_events()
    events.insert(2, replace(events[1], seq=3))
    events = [replace(e, seq=i + 1) for i, e in enumerate(events)]
    with pytest.raises(ContractError, match="duplicate UNLOAD"):
        build_memory(events)
    events = synthetic_events()
    del events[1]
    with pytest.raises(ContractError, match="without available"):
        build_memory(events)
