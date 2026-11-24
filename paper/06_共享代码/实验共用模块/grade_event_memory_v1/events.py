"""Causal, whole-shift material-event contract; no simulation or learning.

``time`` is information visibility time. PROCESS physical intervals are separate
and must never be used to reorder events. Cursors are exclusive token counts.
Event IDs and canonical journal sequence numbers are audit/join fields, not GRU
features. A producer may aggregate contiguous physical callbacks before writing
one record, when its declared journal contract preserves every query/unload/
window barrier. Token spans and ``time_first`` describe canonical records; they
do not reconstruct callback counts or the first visibility time of merged ticks.
"""

from collections import deque
from dataclasses import asdict, dataclass, fields, replace
import math
from typing import Iterable, Mapping


SCHEMA = "grade-event-memory-v1"
KINDS = ("UNLOAD", "PROCESS", "WINDOW_CLOSE", "DECISION", "END")
WINDOW_SECONDS = 7200.0
SHIFT_SECONDS = 4 * WINDOW_SECONDS
EPS = 1e-8


class ContractError(ValueError):
    """A sealed shift cannot be admitted to the memory dataset."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _same(left: float, right: float) -> bool:
    return math.isclose(left, right, abs_tol=EPS, rel_tol=EPS)


@dataclass(frozen=True)
class Event:
    seq: int
    kind: str
    time: float
    window: int
    batch_id: str = ""
    source_id: int = -1
    truck_id: int = -1
    tons: float = 0.0
    metal: float = 0.0
    grade: float = 0.0
    grade_valid: bool = False
    interval_start: float = 0.0
    interval_end: float = 0.0
    window_q: float = 0.0
    window_m: float = 0.0
    window_grade: float = 0.0
    window_grade_valid: bool = False
    grade_reward_raw: float = 0.0
    request_id: str = ""
    decision_sequence: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping) -> "Event":
        _require(isinstance(value, Mapping), "event must be a mapping")
        _require(set(value) == set(EVENT_FIELDS), "event fields differ from frozen schema")
        event = cls(**value)
        validate_event(event)
        return event


EVENT_FIELDS = tuple(field.name for field in fields(Event))


@dataclass(frozen=True)
class Token:
    seq_first: int
    seq_last: int
    kind: str
    time_first: float
    time: float
    window: int
    batch_id: str
    source_id: int
    truck_id: int
    tons: float
    metal: float
    grade: float
    grade_valid: bool
    interval_start: float
    interval_end: float
    window_q: float
    window_m: float
    window_grade: float
    window_grade_valid: bool
    grade_reward_raw: float
    q_before: float
    m_before: float
    grade_before: float
    grade_before_valid: bool
    q_after: float
    m_after: float
    grade_after: float
    grade_after_valid: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Query:
    request_id: str
    decision_sequence: int
    current_cursor: int
    next_cursor: int
    current_seq: int
    next_seq: int
    terminal_next: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Settlement:
    window: int
    seq: int
    time: float
    tons: float
    metal: float
    grade: float
    grade_valid: bool
    grade_reward_raw: float
    cost: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MemoryTape:
    events: tuple[Event, ...]
    tokens: tuple[Token, ...]
    queries: tuple[Query, ...]
    settlements: tuple[Settlement, ...]


def grade_reward(tons: float, metal: float) -> tuple[float, bool, float]:
    """Actual existing clipped grade reward; empty windows have explicit cost 2."""
    _require(math.isfinite(tons) and math.isfinite(metal), "nonfinite window material")
    _require(tons >= 0 and metal >= 0 and metal <= tons + EPS, "invalid window material")
    if tons == 0:
        _require(metal == 0, "empty window cannot contain metal")
        return 0.0, False, -1.0
    grade = metal / tons
    return grade, True, max(-1.0, 1.0 - abs(grade - 0.27) / 0.01)


def validate_event(event: Event) -> None:
    _require(isinstance(event, Event), "event must be Event")
    for name in ("seq", "window", "source_id", "truck_id", "decision_sequence"):
        _require(type(getattr(event, name)) is int, f"{name} must be an integer")
    _require(event.seq > 0, "seq must be positive")
    _require(event.kind in KINDS, "unknown event kind")
    _require(0 <= event.window < 4, "window must be 0..3")
    _require(event.source_id >= -1 and event.truck_id >= -1, "invalid categorical ID")
    _require(event.decision_sequence >= 0, "negative decision sequence")
    for name in ("batch_id", "request_id"):
        value = getattr(event, name)
        _require(isinstance(value, str) and value.strip() == value
                 and "\x00" not in value and "\n" not in value and "\r" not in value,
                 f"invalid {name}")
    for name in ("grade_valid", "window_grade_valid"):
        _require(type(getattr(event, name)) is bool, f"{name} must be a bool")
    numeric = ("time", "tons", "metal", "grade", "interval_start", "interval_end",
               "window_q", "window_m", "window_grade", "grade_reward_raw")
    for name in numeric:
        value = getattr(event, name)
        _require(type(value) in (int, float) and math.isfinite(value), f"nonfinite/nonreal {name}")
    _require(0 <= event.time <= SHIFT_SECONDS + EPS, "event time outside shift")

    material = event.kind in ("UNLOAD", "PROCESS")
    if material:
        _require(bool(event.batch_id), "material batch ID missing")
        _require(event.tons > 0 and 0 <= event.grade <= 1 and event.grade_valid,
                 "invalid material grade/tons")
        _require(_same(event.metal, event.tons * event.grade), "material metal/grade mismatch")
    else:
        _require(event.batch_id == "" and event.source_id == -1 and event.truck_id == -1,
                 "nonmaterial event has material IDs")
        _require(event.tons == event.metal == event.grade == 0 and not event.grade_valid,
                 "nonmaterial event has material payload")
    if event.kind == "PROCESS":
        _require(event.interval_start >= event.window * WINDOW_SECONDS - EPS
                 and event.interval_end <= (event.window + 1) * WINDOW_SECONDS + EPS,
                 "PROCESS interval crosses physical window")
        _require(event.interval_end > event.interval_start
                 and event.interval_end <= event.time + EPS,
                 "invalid PROCESS interval or future physical interval")
    else:
        _require(event.interval_start == event.interval_end == 0,
                 "non-PROCESS physical interval must be zero")
    if event.kind == "WINDOW_CLOSE":
        _require(event.time + EPS >= (event.window + 1) * WINDOW_SECONDS,
                 "early WINDOW_CLOSE")
        grade, valid, reward = grade_reward(event.window_q, event.window_m)
        _require(event.window_grade_valid == valid and _same(event.window_grade, grade),
                 "window grade/validity mismatch")
        _require(_same(event.grade_reward_raw, reward), "raw grade reward formula differs")
    else:
        _require(event.window_q == event.window_m == event.window_grade
                 == event.grade_reward_raw == 0 and not event.window_grade_valid,
                 "non-settlement event has settlement payload")
    if event.kind in ("DECISION", "END"):
        _require(bool(event.request_id) and event.decision_sequence >= 0,
                 "query marker identity missing")
    else:
        _require(event.request_id == "" and event.decision_sequence == 0,
                 "nonquery event has query identity")
    if event.kind == "END":
        _require(event.window == 3 and _same(event.time, SHIFT_SECONDS), "END is not true shift end")


def _grade(q: float, m: float) -> tuple[float, bool]:
    return (m / q, True) if q > 0 else (0.0, False)


def _token(event: Event, before: tuple[float, float], after: tuple[float, float]) -> Token:
    bg, bv = _grade(*before)
    ag, av = _grade(*after)
    return Token(event.seq, event.seq, event.kind, event.time, event.time, event.window,
                 event.batch_id, event.source_id, event.truck_id, event.tons, event.metal,
                 event.grade, event.grade_valid, event.interval_start, event.interval_end,
                 event.window_q, event.window_m, event.window_grade, event.window_grade_valid,
                 event.grade_reward_raw, *before, bg, bv, *after, ag, av)


def _mergeable(left: Token, right: Token) -> bool:
    return (left.kind == right.kind == "PROCESS"
            and left.seq_last + 1 == right.seq_first
            and left.window == right.window and left.batch_id == right.batch_id
            and left.source_id == right.source_id and left.truck_id == right.truck_id
            and _same(left.grade, right.grade)
            and _same(left.interval_end, right.interval_start))


def build_memory(raw_events: Iterable[Event | Mapping]) -> MemoryTape:
    """Validate one sealed, complete shift and derive conservative causal memory.

    FIFO IDs are sidecar joins: source_id must be the LOAD_COMPLETE source and
    must not be inferred from the unload destination. Unknown IDs stay -1.
    No reconstructed future value is ever backfilled into a prior token.
    """
    events = tuple(value if isinstance(value, Event) else Event.from_dict(value)
                   for value in raw_events)
    _require(bool(events), "empty shift")
    tokens: list[Token] = []
    markers: list[tuple[Event, int]] = []
    settlements: list[Settlement] = []
    fifo = deque()
    seen_batches: set[str] = set()
    seen_requests: set[str] = set()
    q = m = 0.0
    active_window = 0
    last_seq = 0
    last_time = 0.0
    last_physical_end = 0.0
    can_merge = False
    ended = False
    for event in events:
        validate_event(event)
        _require(not ended, "event after END")
        _require(event.seq > last_seq, "canonical record seq not strictly increasing")
        _require(event.time + EPS >= last_time, "visibility time moved backwards")
        last_seq, last_time = event.seq, event.time
        if event.kind == "END":
            _require(active_window == 4, "END before all four settlements")
            ended = True
        else:
            _require(event.window == active_window, "event does not match active/unsettled window")
        before = q, m
        if event.kind == "UNLOAD":
            _require(event.batch_id not in seen_batches, "duplicate UNLOAD batch")
            seen_batches.add(event.batch_id)
            fifo.append([event, event.tons])
        elif event.kind == "PROCESS":
            _require(bool(fifo), "PROCESS without available FIFO material")
            original, remaining = fifo[0]
            _require(event.batch_id == original.batch_id, "PROCESS violates FIFO order")
            _require(event.source_id == original.source_id and event.truck_id == original.truck_id,
                     "PROCESS source/truck differs from UNLOAD sidecar")
            _require(_same(event.grade, original.grade), "PROCESS grade differs from UNLOAD batch")
            _require(event.tons <= remaining + EPS, "PROCESS exceeds remaining batch mass")
            _require(event.interval_start + EPS >= last_physical_end,
                     "physical PROCESS intervals overlap or move backwards")
            _require(event.interval_start + EPS >= original.time,
                     "PROCESS occurs physically before material is unloaded")
            last_physical_end = event.interval_end
            remaining -= event.tons
            if remaining <= EPS:
                fifo.popleft()
            else:
                fifo[0][1] = remaining
            q += event.tons
            m += event.metal
        elif event.kind == "WINDOW_CLOSE":
            _require(_same(event.window_q, q) and _same(event.window_m, m),
                     "WINDOW_CLOSE does not conserve actual processed Q/M")
            settlements.append(Settlement(event.window, event.seq, event.time, q, m,
                                          event.window_grade, event.window_grade_valid,
                                          event.grade_reward_raw, 1.0 - event.grade_reward_raw))
            q = m = 0.0
            active_window += 1
        if event.kind in ("DECISION", "END"):
            _require(event.request_id not in seen_requests, "duplicate request marker")
            seen_requests.add(event.request_id)
            _require(event.decision_sequence == len(markers),
                     "decision_sequence must be contiguous including terminal")
            markers.append((event, len(tokens)))
            can_merge = False
        else:
            token = _token(event, before, (q, m))
            if can_merge and tokens and _mergeable(tokens[-1], token):
                left = tokens[-1]
                tokens[-1] = replace(left, seq_last=token.seq_last, time=token.time,
                                     tons=left.tons + token.tons, metal=left.metal + token.metal,
                                     interval_end=token.interval_end, q_after=token.q_after,
                                     m_after=token.m_after, grade_after=token.grade_after,
                                     grade_after_valid=token.grade_after_valid)
            else:
                tokens.append(token)
            can_merge = event.kind == "PROCESS"
    _require(ended and events[-1].kind == "END", "normal END missing; truncation is not terminal")
    _require(len(settlements) == 4, "four unique window settlements required")
    _require(len(markers) >= 2 and markers[0][0].kind == "DECISION", "no actionable decision")
    _require(markers[-1][0].kind == "END", "last query marker must be END")
    # A settlement preceding the first query would have no transition to own it.
    _require(markers[0][0].seq < settlements[0].seq, "first settlement precedes first decision")
    queries = tuple(Query(current.request_id, current.decision_sequence, cc, nc,
                          current.seq, nxt.seq, nxt.kind == "END")
                    for (current, cc), (nxt, nc) in zip(markers[:-1], markers[1:]))
    return MemoryTape(events, tuple(tokens), queries, tuple(settlements))
