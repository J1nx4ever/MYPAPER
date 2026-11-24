"""Pure conditional dump/buffer/FIFO projection; no files, RNG, model or RPC.

Arrival times are caller-owned *dump entrance* conditions, excluding dump waits,
unloading service and buffer delays. Loading and road feedback are not solved
here. This finite-cohort arithmetic is not a real-environment counterfactual.
Public inputs and outputs are frozen dataclasses containing immutable values.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
import heapq
import math


WINDOW_SECONDS = 7200.0
SHIFT_END = 28800.0
EPS = 1e-8
MASS_TOL = .002  # Existing1519D float32 mass admission scale; not a risk certificate.
STAGES = frozenset(('fifo', 'arrival', 'queued', 'capacity_wait', 'admission_pending',
                    'unloading_reserved', 'unmodeled'))


@dataclass(frozen=True)
class Job:
    key: str
    tons: float
    grade: float | None
    commodity: str = 'ore'
    stage: str = 'arrival'
    arrival_time: float | None = None
    service_seconds: float | None = None
    remaining_service_seconds: float | None = None
    origin: str = 'existing_pipeline'
    source_id: int | None = None
    truck_id: int | None = None
    grade_kind: str = 'conditional_prediction'


@dataclass(frozen=True)
class Snapshot:
    now: float
    current_window: int
    window_q: float
    window_m: float
    initial_fifo: tuple[Job, ...] = ()
    jobs: tuple[Job, ...] = ()
    buffer_capacity: float = 500.
    buffer_stored: float | None = None
    buffer_reserved: float = 0.
    ore_servers: int = 3
    waste_servers: int = 2
    rate: float = 2500. / 3600.
    tick_period: float = .2
    tick_anchor: float = 0.
    tick_phase_observed: bool = False
    resource_owners_verified: bool = True
    reservation_owners_verified: bool = True


@dataclass(frozen=True)
class Candidate:
    action: int
    legal: bool
    job: Job | None = None


@dataclass(frozen=True)
class QM:
    q: float
    m: float | None


@dataclass(frozen=True)
class WindowResult:
    index: int
    start: float
    end: float
    future_q: float
    future_m: float | None
    total_q: float
    total_m: float | None
    grade: float | None
    delivered_waste_q: float


@dataclass(frozen=True)
class PacketResult:
    key: str
    origin: str
    commodity: str
    source_id: int | None
    truck_id: int | None
    input_q: float
    input_m: float | None
    processed: tuple[QM, ...]
    delivered_waste: tuple[float, ...]
    pending_q: float
    pending_m: float | None
    pending_stage: str
    modeled: bool
    unload_start: float | None
    unload_complete: float | None


@dataclass(frozen=True)
class Projection:
    now: float
    end: float
    windows: tuple[WindowResult, ...]
    packets: tuple[PacketResult, ...]
    validity: str
    flags: tuple[str, ...]
    modeled_q: float
    unmodeled_q: float
    unknown_grade_q: float
    basis: str

    @property
    def valid(self):
        return self.validity == 'valid'

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ActionEffect:
    action: int
    valid: bool
    candidate_self: tuple[QM, ...]
    existing_change: tuple[QM, ...]
    system_delta: tuple[QM, ...]
    existing_loss: tuple[QM, ...]
    existing_gain: tuple[QM, ...]
    per_existing_packet_change: tuple[tuple[str, tuple[QM, ...]], ...]
    delivered_waste_self: tuple[float, ...]
    delivered_waste_existing_change: tuple[float, ...]
    delivered_waste_system_delta: tuple[float, ...]
    projection: Projection | None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Estimate:
    baseline: Projection
    comparisons: tuple[ActionEffect, ...]

    def to_dict(self):
        return asdict(self)


def _finite(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError('finite ' + ('positive ' if positive else 'nonnegative ') + name + ' required')


def _job(job):
    if type(job) is not Job or not isinstance(job.key, str) or not job.key:
        raise ValueError('immutable Job with explicit key required')
    _finite(job.tons, 'job tons', positive=True)
    if job.commodity not in ('ore', 'waste') or job.stage not in STAGES:
        raise ValueError('unknown commodity or material stage')
    if job.origin not in ('initial_fifo', 'existing_pipeline', 'candidate'):
        raise ValueError('unknown material origin')
    if job.grade is not None:
        _finite(job.grade, 'grade')
        if job.grade > 1:
            raise ValueError('grade must be within0..1')
    for name in ('arrival_time', 'service_seconds', 'remaining_service_seconds'):
        if getattr(job, name) is not None:
            _finite(getattr(job, name), name)
    for name in ('source_id', 'truck_id'):
        if getattr(job, name) is not None and (type(getattr(job, name)) is not int or getattr(job, name) < 0):
            raise ValueError('causal source/truck identifiers must be nonnegative or None')


def _validated(snapshot):
    if type(snapshot) is not Snapshot or type(snapshot.initial_fifo) is not tuple or type(snapshot.jobs) is not tuple:
        raise ValueError('immutable Snapshot and tuple cohorts required')
    for name in ('now', 'window_q', 'window_m', 'buffer_reserved', 'tick_anchor', 'rate'):
        _finite(getattr(snapshot, name), name)
    for name in ('buffer_capacity', 'tick_period'):
        _finite(getattr(snapshot, name), name, positive=True)
    if snapshot.buffer_stored is not None:
        _finite(snapshot.buffer_stored, 'buffer_stored')
    if (type(snapshot.current_window) is not int or not 0 <= snapshot.current_window < 4 or
            snapshot.now > SHIFT_END or snapshot.window_m > snapshot.window_q + MASS_TOL or
            not snapshot.current_window * WINDOW_SECONDS - .01 <= snapshot.now <= (snapshot.current_window + 1) * WINDOW_SECONDS + .01):
        raise ValueError('invalid current window, cut time or processed Q/M')
    for count in (snapshot.ore_servers, snapshot.waste_servers):
        if type(count) is not int or count < 1:
            raise ValueError('positive integer resource capacities required')
    for name in ('tick_phase_observed', 'resource_owners_verified', 'reservation_owners_verified'):
        if type(getattr(snapshot, name)) is not bool:
            raise ValueError('explicit boolean owner/tick evidence required')
    jobs = snapshot.initial_fifo + snapshot.jobs
    for job in jobs:
        _job(job)
    if len({job.key for job in jobs}) != len(jobs):
        raise ValueError('duplicate material key; buffer reservation is not extra material')
    if len(snapshot.initial_fifo) > 16:
        raise ValueError('FIFO snapshot overflow; no truncation')
    if any(j.stage != 'fifo' or j.commodity != 'ore' for j in snapshot.initial_fifo):
        raise ValueError('initial FIFO must contain only ore FIFO jobs')
    if any(j.stage == 'fifo' for j in snapshot.jobs):
        raise ValueError('FIFO material cannot also be in the pipeline')
    flags = []
    stored = sum(j.tons for j in snapshot.initial_fifo)
    if snapshot.buffer_stored is not None and abs(stored - snapshot.buffer_stored) > MASS_TOL:
        flags.append('buffer_fifo_mass_inconsistent')
    if stored + snapshot.buffer_reserved > snapshot.buffer_capacity + MASS_TOL:
        flags.append('buffer_reservation_capacity_inconsistent')
    expected_r = sum(j.tons for j in snapshot.jobs if j.commodity == 'ore' and j.stage == 'unloading_reserved')
    if abs(expected_r - snapshot.buffer_reserved) > MASS_TOL:
        flags.append('reservation_owner_mass_inconsistent')
    pending_q = sum(j.tons for j in snapshot.jobs if j.commodity == 'ore' and j.stage == 'admission_pending')
    if pending_q and stored + snapshot.buffer_reserved + pending_q > snapshot.buffer_capacity + MASS_TOL:
        flags.append('admission_pending_capacity_inconsistent')
    if not snapshot.resource_owners_verified:
        flags.append('resource_owners_unverified')
    if not snapshot.reservation_owners_verified:
        flags.append('reservation_owners_unverified')
    for commodity, capacity in (('ore', snapshot.ore_servers), ('waste', snapshot.waste_servers)):
        holders = [j for j in snapshot.jobs if j.commodity == commodity and
                   j.stage in ('unloading_reserved', 'capacity_wait', 'admission_pending')]
        if len(holders) > capacity:
            flags.append('resource_holders_exceed_capacity:' + commodity)
        if sum(j.stage == 'admission_pending' for j in holders) > 1:
            flags.append('multiple_admission_pending:' + commodity)
        if commodity == 'ore' and any(j.tons > snapshot.buffer_capacity + MASS_TOL for j in snapshot.jobs if j.commodity == 'ore'):
            flags.append('ore_payload_exceeds_buffer_capacity')
    return tuple(flags)


def _windows(snapshot):
    end = min((snapshot.current_window + 2) * WINDOW_SECONDS, SHIFT_END)
    stop = (snapshot.current_window + 1) * WINDOW_SECONDS
    windows = [(snapshot.current_window, snapshot.now, stop)]
    if snapshot.current_window < 3:
        windows.append((snapshot.current_window + 1, stop, end))
    return tuple(windows), end


def _metal(q, grade):
    return 0. if q <= EPS else (None if grade is None else q * grade)


def _sum_m(values):
    values = tuple(values)
    return None if any(v is None for v in values) else sum(values)


def project(snapshot: Snapshot, candidate: Job | None = None) -> Projection:
    """Jointly reschedule a finite cohort under explicit conditional inputs.

    Unknown arrival/service jobs are excluded from the modeled cohort, retained
    as unmodeled pending mass, and make aggregate output partial. Unknown held
    service times disable that commodity's future pipeline, since dropping a
    resource holder would incorrectly free capacity. Initial FIFO still has its
    independent work-conserving prefix. At H<T, boundary events update pending
    stage but cannot contribute to [t,H) production. Events at T are excluded.
    """
    invalid = _validated(snapshot)
    if candidate is not None:
        _job(candidate)
        if candidate.key in {j.key for j in snapshot.initial_fifo + snapshot.jobs}:
            raise ValueError('candidate key collides with existing material')
        if candidate.stage not in ('arrival', 'unmodeled'):
            raise ValueError('one new candidate must enter via dump-arrival condition')
        candidate = replace(candidate, origin='candidate')
        if candidate.commodity == 'ore' and candidate.tons > snapshot.buffer_capacity + MASS_TOL:
            invalid += ('candidate_ore_payload_exceeds_buffer_capacity',)
    windows, horizon = _windows(snapshot)
    all_jobs = tuple(replace(j, origin='initial_fifo') for j in snapshot.initial_fifo) + snapshot.jobs + ((candidate,) if candidate else ())
    info = {j.key: j for j in all_jobs}
    stage = {j.key: j.stage for j in all_jobs}
    remaining = {j.key: j.tons for j in all_jobs}
    amounts = {j.key: [0.] * len(windows) for j in all_jobs}
    deliveries = {j.key: [0.] * len(windows) for j in all_jobs}
    starts = {j.key: None for j in all_jobs}
    completions = {j.key: None for j in all_jobs}
    flags = ['conditional_dump_arrival_and_service_inputs', 'loading_and_road_feedback_not_solved'] + list(invalid)
    observed_stock = sum(j.tons for j in snapshot.initial_fifo)
    observed_reserved = sum(j.tons for j in snapshot.jobs if j.commodity == 'ore' and j.stage == 'unloading_reserved')
    if snapshot.buffer_stored is not None and EPS < abs(snapshot.buffer_stored - observed_stock) <= MASS_TOL:
        flags.append('buffer_mass_rounding_normalized_to_FIFO')
    if EPS < abs(snapshot.buffer_reserved - observed_reserved) <= MASS_TOL:
        flags.append('reservation_mass_rounding_normalized_to_owners')
    unmodeled = set()
    pipeline = tuple(j for j in all_jobs if j.stage != 'fifo')
    disabled = set(('ore', 'waste')) if invalid else set()
    for job in pipeline:
        duration = job.remaining_service_seconds if job.stage == 'unloading_reserved' else job.service_seconds
        if job.stage in ('queued', 'capacity_wait', 'admission_pending', 'unloading_reserved') and duration is None:
            disabled.add(job.commodity)
            flags.append('held_or_queued_service_unknown:' + job.key)
    for job in pipeline:
        if (job.commodity in disabled or job.stage == 'unmodeled' or
                (job.stage == 'arrival' and (job.arrival_time is None or job.service_seconds is None))):
            unmodeled.add(job.key)
            stage[job.key] = 'unmodeled'
            flags.append('unmodeled_job:' + job.key)
    fifo = deque(j.key for j in snapshot.initial_fifo)
    queued = {c: deque() for c in ('ore', 'waste')}
    waiting = {c: deque() for c in ('ore', 'waste')}
    held = {c: set() for c in ('ore', 'waste')}
    capacity = dict(ore=snapshot.ore_servers, waste=snapshot.waste_servers)
    reserved = [sum(j.tons for j in pipeline if j.key not in unmodeled and
                    j.commodity == 'ore' and j.stage == 'unloading_reserved')]
    events = []
    serial = [0]
    retry_version = [0]
    clock = [snapshot.now]

    def push(time, kind, key='', version=0):
        if time < clock[0] - EPS:
            flags.append('past_arrival_clamped:' + key)
            time = clock[0]
        if time <= horizon + EPS and time < SHIFT_END:
            serial[0] += 1
            # Future equal-time order is a declared stable nominal policy.
            heapq.heappush(events, (max(clock[0], time), {'complete': 0, 'arrival': 1, 'retry': 2}[kind], key, serial[0], kind, version))

    def stock():
        return sum(remaining[key] for key in fifo)

    def start_service(key):
        job = info[key]
        if job.commodity == 'ore':
            if stock() + reserved[0] + job.tons > snapshot.buffer_capacity + MASS_TOL:
                return False
            reserved[0] += job.tons
        stage[key] = 'unloading_reserved'
        starts[key] = clock[0]
        push(clock[0] + job.service_seconds, 'complete', key)
        return True

    def admit(commodity):
        while waiting[commodity]:
            key = waiting[commodity][0]
            if not start_service(key):
                break
            waiting[commodity].popleft()

    def acquire(commodity):
        while queued[commodity] and len(held[commodity]) < capacity[commodity]:
            key = queued[commodity].popleft()
            held[commodity].add(key)
            stage[key] = 'capacity_wait'
            waiting[commodity].append(key)
            # Wait entry retries the head, not necessarily this entering truck.
            admit(commodity)

    def schedule_retry():
        retry_version[0] += 1
        if not waiting['ore']:
            return
        key = waiting['ore'][0]
        needed = stock() + reserved[0] + info[key].tons - snapshot.buffer_capacity
        if needed > stock() + MASS_TOL:
            return  # Reserved unloading material is not yet crushable FIFO.
        if needed > EPS and snapshot.rate == 0:
            return
        eligible = clock[0] + (max(0., needed) / snapshot.rate if snapshot.rate else 0.)
        period, anchor = snapshot.tick_period, snapshot.tick_anchor
        tick = anchor + math.ceil((eligible - anchor) / period - 1e-10) * period
        if tick <= clock[0] + EPS:
            tick = anchor + (math.floor((clock[0] - anchor) / period + 1e-10) + 1) * period
        if not snapshot.tick_phase_observed:
            flags.append('crusher_tick_phase_assumed')
        push(tick, 'retry', version=retry_version[0])

    def advance(end):
        begin = clock[0]
        if snapshot.rate == 0:
            clock[0] = end
            return
        while fifo and begin < end - EPS:
            key = fifo[0]
            finish = min(end, begin + remaining[key] / snapshot.rate)
            for index, (_, lo, hi) in enumerate(windows):
                overlap = max(0., min(finish, hi) - max(begin, lo))
                amounts[key][index] += overlap * snapshot.rate
            used = min(remaining[key], (finish - begin) * snapshot.rate)
            remaining[key] = max(0., remaining[key] - used)
            begin = finish
            if remaining[key] <= EPS:
                remaining[key] = 0.
                stage[key] = 'processed'
                fifo.popleft()
        clock[0] = end

    for job in pipeline:
        key, commodity = job.key, job.commodity
        if key in unmodeled:
            continue
        if job.stage in ('unloading_reserved', 'capacity_wait', 'admission_pending'):
            held[commodity].add(key)
        if job.stage == 'unloading_reserved':
            # True start was not supplied by this DTO; do not invent one at cut.
            flags.append('active_service_continuation:' + key)
            push(snapshot.now + job.remaining_service_seconds, 'complete', key)
        elif job.stage == 'capacity_wait':
            waiting[commodity].append(key)
        elif job.stage == 'queued':
            queued[commodity].append(key)
        elif job.stage == 'arrival':
            push(job.arrival_time, 'arrival', key)
    # A declared pending owner is an already released Wait→Delay transfer.
    for job in pipeline:
        if job.key not in unmodeled and job.stage == 'admission_pending':
            flags.append('admission_pending_atomic_transfer_assumed')
            waiting[job.commodity].appendleft(job.key)
            admit(job.commodity)
    for commodity in ('ore', 'waste'):
        acquire(commodity)
    schedule_retry()
    previous_event_time = None
    while events:
        time, _, key, _, kind, version = heapq.heappop(events)
        if kind == 'retry' and version != retry_version[0]:
            continue
        if time > horizon + EPS:
            break
        if previous_event_time is not None and abs(previous_event_time - time) <= EPS:
            flags.append('future_event_tie_stable_nominal_order_assumed')
        previous_event_time = time
        advance(min(time, horizon))
        if kind == 'arrival':
            commodity = info[key].commodity
            stage[key] = 'queued'
            queued[commodity].append(key)
            acquire(commodity)
        elif kind == 'complete':
            job = info[key]
            completions[key] = clock[0]
            if job.commodity == 'ore':
                reserved[0] = max(0., reserved[0] - job.tons)
                fifo.append(key)
                stage[key] = 'fifo'
            else:
                for index, (_, lo, hi) in enumerate(windows):
                    if lo <= clock[0] < hi:
                        deliveries[key][index] += job.tons
                # At H<T completion is a boundary fact, not a [t,H) delivery.
                if clock[0] < horizon:
                    remaining[key] = 0.
                    stage[key] = 'delivered'
                else:
                    stage[key] = 'uncredited_boundary_delivery'
                    flags.append('uncredited_boundary_delivery:' + key)
            held[job.commodity].discard(key)
            acquire(job.commodity)
        else:
            admit('ore')
        if stock() + reserved[0] > snapshot.buffer_capacity + MASS_TOL:
            raise AssertionError('conditional buffer conservation failed')
        schedule_retry()
    advance(horizon)
    packets = []
    for job in all_jobs:
        ore = job.commodity == 'ore'
        qs = amounts[job.key]
        ds = deliveries[job.key]
        processed = tuple(QM(q, _metal(q, job.grade)) for q in qs)
        pending_q = max(0., job.tons - (sum(qs) if ore else sum(ds)))
        pending_m = _metal(pending_q, job.grade) if ore else None
        packets.append(PacketResult(job.key, job.origin, job.commodity, job.source_id, job.truck_id,
            job.tons, _metal(job.tons, job.grade) if ore else None, processed, tuple(ds),
            pending_q, pending_m, stage[job.key] if pending_q > EPS else ('processed' if ore else 'delivered'),
            job.key not in unmodeled, starts[job.key], completions[job.key]))
        if abs(job.tons - sum(qs if ore else ds) - pending_q) > MASS_TOL:
            raise AssertionError('per-packet quantity conservation failed')
    results = []
    for index, (number, lo, hi) in enumerate(windows):
        qs = [p.processed[index].q for p in packets if p.commodity == 'ore']
        ms = [p.processed[index].m for p in packets if p.commodity == 'ore']
        fq, fm = sum(qs), _sum_m(ms)
        tq = fq + (snapshot.window_q if index == 0 else 0.)
        tm = None if fm is None else fm + (snapshot.window_m if index == 0 else 0.)
        grade = tm / tq if tq > EPS and tm is not None else None
        results.append(WindowResult(number, lo, hi, fq, fm, tq, tm, grade,
            sum(p.delivered_waste[index] for p in packets)))
    unknown_grade = sum(j.tons for j in all_jobs if j.commodity == 'ore' and j.grade is None)
    if unknown_grade:
        flags.append('unknown_ore_grade')
    validity = 'invalid' if invalid else ('partial' if unmodeled or unknown_grade else 'valid')
    unmodeled_q = sum(info[key].tons for key in unmodeled)
    return Projection(snapshot.now, horizon, tuple(results), tuple(packets), validity,
        tuple(dict.fromkeys(flags)), sum(j.tons for j in all_jobs) - unmodeled_q,
        unmodeled_q, unknown_grade, 'modeled_subset_only' if unmodeled else 'all_existing_plus_candidate')


def _difference(a, b):
    return None if a is None or b is None else a - b


def estimate(snapshot: Snapshot, candidates: tuple[Candidate, ...] = ()) -> Estimate:
    """Return a baseline and per-action signed effects under shared conditions."""
    if type(candidates) is not tuple or any(type(c) is not Candidate for c in candidates):
        raise ValueError('immutable tuple of Candidate required')
    if len({c.action for c in candidates}) != len(candidates):
        raise ValueError('duplicate candidate action')
    baseline = project(snapshot)
    count = len(baseline.windows)
    zero = tuple(QM(0., 0.) for _ in range(count))
    comparisons = []
    base_packets = {p.key: p for p in baseline.packets}
    for candidate in candidates:
        if type(candidate.action) is not int or candidate.action not in range(4) or type(candidate.legal) is not bool:
            raise ValueError('explicit existing4-action index and legal boolean required')
        if not candidate.legal or candidate.job is None:
            comparisons.append(ActionEffect(candidate.action, False, zero, zero, zero, zero, zero, (),
                (0.,) * count, (0.,) * count, (0.,) * count, None,
                ('illegal_action' if not candidate.legal else 'missing_candidate_job',)))
            continue
        projection = project(snapshot, candidate.job)
        packets = {p.key: p for p in projection.packets}
        own = packets[candidate.job.key]
        changes = tuple((key, tuple(QM(packets[key].processed[i].q - p.processed[i].q,
            _difference(packets[key].processed[i].m, p.processed[i].m)) for i in range(count)))
            for key, p in base_packets.items())
        existing, loss, gain, delta = [], [], [], []
        waste_existing = []
        for i in range(count):
            q_changes = [values[i].q for _, values in changes]
            m_changes = [values[i].m for _, values in changes]
            ec = QM(sum(q_changes), _sum_m(m_changes))
            existing.append(ec)
            loss.append(QM(sum(max(0., -v) for v in q_changes),
                None if any(v is None for v in m_changes) else sum(max(0., -v) for v in m_changes)))
            gain.append(QM(sum(max(0., v) for v in q_changes),
                None if any(v is None for v in m_changes) else sum(max(0., v) for v in m_changes)))
            delta.append(QM(own.processed[i].q + ec.q,
                None if own.processed[i].m is None or ec.m is None else own.processed[i].m + ec.m))
            waste_existing.append(sum(packets[key].delivered_waste[i] - p.delivered_waste[i]
                for key, p in base_packets.items()))
        comparisons.append(ActionEffect(candidate.action, baseline.valid and projection.valid,
            own.processed, tuple(existing), tuple(delta), tuple(loss), tuple(gain), changes,
            own.delivered_waste, tuple(waste_existing),
            tuple(own.delivered_waste[i] + waste_existing[i] for i in range(count)), projection,
            projection.flags))
    return Estimate(baseline, tuple(comparisons))
