"""Causal 128D material/task summary of one dispatch.physical.v3 snapshot.

All tons and metal refer to *physical* quantities, not reward bookkeeping.
FIFO order and metal are known. Loaded ore is known but its arrival is not.
Unloaded reservations use observed loadingGradeMean/Std only while loading,
otherwise the observed source prior. No hidden block grade or future event is
accepted by this module. Reservations and loaded payloads are mutually exclusive.

The two-window projection is a nominal, work-conserving FIFO calculation with
only existing commitments. Arrival estimates use current route geometry/speed,
observed queues and mean service times. They are NOT actual ETA or guaranteed
window membership. Each window also carries a FIFO-only quantity lower bound
and a deliberately loose quantity upper bound allowing all pipeline tons to
arrive in that window. The upper bounds for different windows need not be
jointly achievable. The bounds assume the observed crusher rate is maintained;
they do not predict a future outage. Grade standard deviations are uncertainty
summaries, not confidence intervals; shared unknown blocks can correlate trucks.
free_capacity is nominal unfilled processing capacity, not guaranteed adjustable
tons. required_interval_intersects_ore_support tests only the convex grade hull
of currently legal ore sources, not transport/queues/FIFO dynamic feasibility.

Frozen modeling assumptions: nominal loaded/empty speeds 5.0/6.5mps are causal
fallback choices, capped by observed road speed limits (or by current speed for
an already moving truck). They are not measured free-flow guarantees. The
0.70min load setup and type-specific dump means reproduce the known public mean
formulas of LoadSite.obsMeanLoadMinutes and DumpSite.obsMeanDumpMinutes. While
service is already active, its observed mean and observed elapsed age are used.

Scaling: window/pipeline Q and M /5000t; shift Q,M /20000t; waste /4000t;
metal balances and metal standard deviations /(5000t * 0.01); grade center
errors /0.01; clocks /28800s or /7200s; action tons /193t. Grade fractions and
required-new-grade bounds retain their physical units and are not clipped.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from python import observation_contract as _observation_contract
from python.observation_contract import (
    FIELD_NAMES, OBSERVATION_SCHEMA_ID, OBSERVATION_SIZE, SCHEMA,
    OBSERVATION_LOW, OBSERVATION_HIGH,
)


FEATURE_VERSION = 'commitment-summary-v1'
FEATURE_CONFIG = {
    'schema': 'dispatch.physical.v3',
    'schema_sha256': 'c242ce76856d1cd929af738da3942bab312749712603626d6c2747ac47d12637',
    'shift_seconds': 28800.0,
    'window_seconds': 7200.0,
    'window_tons_scale': 5000.0,
    'shift_tons_scale': 20000.0,
    'waste_tons_scale': 4000.0,
    'truck_tons_scale': 193.0,
    'grade_deviation_scale': 0.01,
    'loaded_nominal_mps': 5.0,
    'empty_nominal_mps': 6.5,
    'minimum_nominal_mps': 0.5,
    'load_setup_minutes': 0.70,
    'ore_dump_mean_minutes_by_type': [1.8, 2.0, 2.3],
    'waste_dump_mean_minutes_by_type': [1.5, 1.7, 2.0],
    'load_node_indices': [5, 6, 7, 8],
    'dump_node_indices': [0, 1],
    'minimum_adjustable_tons': 1.0,
    'quantity_epsilon_tons': 1e-5,
    'projection': 'existing_commitments_nominal_fifo_plus_quantity_bounds',
    'arrival_time': 'causal_nominal_not_actual_eta',
    'unknown_grade': 'observed_loading_stats_else_source_uniform_prior',
    'variance': 'independent_diagnostic_and_correlated_std_upper_summary',
    'history': 'pure_snapshot_no_window_reset_or_future_access',
}
_FROZEN_CONFIG = deepcopy(FEATURE_CONFIG)

_GLOBAL_NAMES = (
    'shift.elapsed', 'shift.remaining', 'window.index', 'window.remaining',
    'window.active', 'next.active', 'terminal',
    'window.processed_q', 'window.processed_m', 'window.grade',
    'window.grade_valid', 'window.center_error', 'window.lower_balance',
    'window.upper_balance', 'shift.ore', 'shift.metal', 'shift.waste',
    'shift.ore_gap', 'shift.waste_gap', 'crusher.rate', 'crusher.buffer_free',
)
_POOL_NAMES = (
    'fifo.q', 'fifo.m', 'fifo.center_balance',
    'loaded_ore.q', 'loaded_ore.m', 'loaded_ore.center_balance',
    'loaded_transit.q', 'loaded_transit.m',
    'loaded_waiting.q', 'loaded_waiting.m',
    'loaded_unloading.q', 'loaded_unloading.m',
    'reserved_ore.q', 'reserved_ore.m_expected',
    'reserved_ore.metal_std_independent', 'reserved_ore.metal_std_correlated',
    'reserved_ore.grade_known_fraction',
    'loaded_waste.q', 'reserved_waste.q', 'pending.grade_unknown_q',
    'pending.eta_unavailable_q', 'pending.arrival_time_uncertain', 'pending.count',
)
_WINDOW_NAMES = (
    'duration', 'capacity', 'fifo_q', 'fifo_m',
    'nominal_q', 'nominal_m', 'metal_std_correlated',
    'nominal_grade', 'grade_valid', 'lower_balance', 'upper_balance',
    'center_error', 'free_capacity',
    'required_grade_lower', 'required_grade_upper', 'required_grade_valid',
    'required_interval_intersects_ore_support', 'q_lower', 'q_upper',
    'unknown_grade_fraction', 'arrival_uncertain_fraction', 'unmodeled_q',
)
_ACTION_NAMES = (
    'legal', 'dispatch_q', 'ore', 'grade_mean', 'grade_std',
    'available_q', 'queue_count', 'busy_fraction', 'nominal_eta', 'eta_valid',
)
FEATURE_NAMES = (_GLOBAL_NAMES + _POOL_NAMES
                 + tuple(f'{w}.{name}' for w in ('current', 'next') for name in _WINDOW_NAMES)
                 + tuple(f'action.{a}.{name}' for a in range(4) for name in _ACTION_NAMES))
FEATURE_SIZE = len(FEATURE_NAMES)
assert FEATURE_SIZE == 128 and len(set(FEATURE_NAMES)) == FEATURE_SIZE

_BINARY = np.array([f['index'] for f in SCHEMA if f['kind'] == 'binary'])
_SLOTS = np.array([f['index'] for f in SCHEMA if f['kind'] == 'truck_slot'])
_LOW, _HIGH = np.asarray(OBSERVATION_LOW), np.asarray(OBSERVATION_HIGH)
_ONEHOT = tuple(slice(27 + 120*t + start, 27 + 120*t + start + count)
                for t in range(9)
                for start, count in ((1, 3), (4, 12), (26, 4), (30, 2), (32, 9), (45, 16)))
_RAW_SCHEMA_PATH = Path(_observation_contract.__file__).with_name('observation_schema.json')
if (OBSERVATION_SCHEMA_ID != 'dispatch.physical.v3' or OBSERVATION_SIZE != 1519
        or hashlib.sha256(_RAW_SCHEMA_PATH.read_bytes()).hexdigest() != FEATURE_CONFIG['schema_sha256']):
    raise RuntimeError('commitment summary requires the frozen dispatch.physical.v3 schema')
for _i, _name in ((2, 'global.windowIndex'), (4, 'global.windowTons'),
                  (44, 'truck.0.loaded'), (47, 'truck.0.reservedLoadTons'),
                  (49, 'truck.0.payloadGrade'), (89, 'truck.0.routeRemaining'),
                  (1107, 'load.0.available'), (1304, 'dump.0.crusherRate'),
                  (1391, 'fifo.0.valid'), (1439, 'road.0.from')):
    if FIELD_NAMES[_i] != _name:
        raise RuntimeError(f'raw schema field {_i} differs from frozen feature contract')


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def _config(config):
    result = deepcopy(_FROZEN_CONFIG if config is None else config)
    # This experiment freezes physical/scaling assumptions, not just key names.
    if result != _FROZEN_CONFIG:
        raise ValueError('feature config differs from frozen commitment-summary-v1')
    return result


def feature_contract(config=None):
    """Exact JSON contract shared by prepare, model, admission and checkpoints."""
    cfg = _config(config)
    return {'version': FEATURE_VERSION, 'names_sha256': _json_sha(FEATURE_NAMES),
            'config': cfg, 'config_sha256': _json_sha(cfg), 'dimension': FEATURE_SIZE,
            'scope': 'causal_decision_summary_whole_shift', 'source': 'dispatch.physical.v3'}


def feature_config_sha256(config=None):
    return _json_sha(_config(config))


def _state(state):
    s = np.asarray(state)
    if s.dtype != np.float32 or s.shape != (1519,):
        raise ValueError('summary requires float32 state[1519]')
    if not np.isfinite(s).all() or np.any(s < _LOW) or np.any(s > _HIGH):
        raise ValueError('state contains nonfinite or out-of-schema values')
    if np.any((s[_BINARY] != 0) & (s[_BINARY] != 1)):
        raise ValueError('state binary fields must be 0/1')
    if np.any(np.abs(s[_SLOTS]*9 - np.rint(s[_SLOTS]*9)) > 2e-6):
        raise ValueError('state queue slots must encode truck ids')
    if any(float(s[sl].sum()) > 1.000001 for sl in _ONEHOT):
        raise ValueError('state categorical fields must be one-hot or empty')
    ego = s[27:1107:120]
    if s[17]:
        if np.any(ego) or np.any(s[19:27]):
            raise ValueError('terminal state must have no ego/action')
    elif ego.sum() != 1 or not np.any(s[19:23]):
        raise ValueError('decision state requires exactly one ego and legal action')
    wi = float(s[2])*4
    if abs(wi-round(wi)) > 1e-6 or (not s[17] and wi >= 4):
        raise ValueError('invalid physical window index')
    if np.any(s[[4, 5, 7, 8, 9, 10, 11, 12, 1300, 1301, 1303, 1304]] < 0):
        raise ValueError('physical quantities must be nonnegative')
    if not 0 <= s[13] < s[14] <= 1 or not s[13] <= s[15] <= s[14]:
        raise ValueError('invalid grade interval/target')
    return s.astype(np.float64)


def _onehot_index(x):
    found = np.flatnonzero(x)
    return int(found[0]) if len(found) == 1 else None


def _ids(x):
    return [int(round(v*9))-1 for v in x if v > 0]


def _road_times(s, cfg, *, loaded):
    """Observed open graph, fixed nominal speeds capped by observed speed limits."""
    times = np.full((9, 9), np.inf)
    np.fill_diagonal(times, 0)
    base = cfg['loaded_nominal_mps'] if loaded else cfg['empty_nominal_mps']
    for road in s[1439:1519].reshape(16, 5):
        if not road[0] or not road[1] or road[4] or road[2] <= 0 or road[3] <= 0:
            continue
        a, b = int(round(road[0]*9))-1, int(round(road[1]*9))-1
        if not (0 <= a < 9 and 0 <= b < 9):
            raise ValueError('invalid road node encoding')
        times[a, b] = min(times[a, b], road[2]*10000 / min(base, road[3]*10))
    for k in range(9):
        times = np.minimum(times, times[:, k, None] + times[None, k, :])
    return times


def _load_mean(load, q, cfg):
    bucket = load[18]*30
    if q <= 0 or bucket <= 0 or load[19] < 0:
        return math.inf
    return 60*(cfg['load_setup_minutes'] + load[19]*math.ceil(max(0, q-1e-9)/bucket))


def _dump_mean(truck, cfg, *, ore=True):
    typ = _onehot_index(truck[1:4])
    if typ is None:
        return math.inf
    return 60*cfg['ore_dump_mean_minutes_by_type' if ore else 'waste_dump_mean_minutes_by_type'][typ]


def _queue_seconds(s, site, *, load, truck_id=None, cfg):
    """Current recorded queue work only; no future queue-length prediction."""
    trucks = s[27:1107].reshape(9, 120)
    if load:
        queue, active = _ids(site[20:29]) + _ids(site[29:38]), _ids(site[38:47])
        capacity = max(1.0, site[14])
    else:
        queue = _ids(site[12:21]) + _ids(site[21:30]) + _ids(site[30:39])
        active = _ids(site[39:48])
        capacity = max(1.0, site[2]*3)
    # A truck may appear in two observation lists at an event boundary.
    queue = list(dict.fromkeys(queue))
    active = list(dict.fromkeys(active))
    if truck_id in queue:
        queue = queue[:queue.index(truck_id)]
    total = 0.0
    for tid in active + queue:
        if tid == truck_id:
            continue
        t = trucks[tid]
        if load:
            mean = _load_mean(site, t[20]*193, cfg)
            if t[84] > 0:  # loadMean is only present while loading.
                mean = max(0.0, t[85]*600 - t[84]*600)
        else:
            mean = _dump_mean(t, cfg, ore=bool(site[1]))
            if t[88] > 0:
                mean = max(0.0, t[88]*600 - t[87]*600)
        total += mean
    return total/capacity


def _moving_seconds(t, cfg, *, loaded):
    distance = t[62]*10000  # routeRemaining is distance, never actual ETA.
    if distance < 0:
        raise ValueError('negative route distance')
    nominal = cfg['loaded_nominal_mps'] if loaded else cfg['empty_nominal_mps']
    speed = t[43]*10
    if speed >= cfg['minimum_nominal_mps']:
        nominal = min(nominal, speed)
    return distance/max(cfg['minimum_nominal_mps'], nominal)


def _arrival(s, tid, cfg, loaded_paths):
    """Nominal unloading-completion time; inf means unavailable, never zero-fill."""
    t = s[27+120*tid:147+120*tid]
    phase = _onehot_index(t[4:16])
    ore_dump = s[1295:1343]
    if phase is None:
        return math.inf
    q = t[19]*193 if t[17] else t[20]*193
    dump = _dump_mean(t, cfg)
    buffer_delay = max(0.0, s[1300]*500 + q - s[1303]*500)
    rate = s[1304]*2500/3600
    buffer_delay = buffer_delay/rate if rate > 0 else math.inf
    dump_queue = _queue_seconds(s, ore_dump, load=False, truck_id=tid, cfg=cfg)
    if t[17]:
        if phase == 9:  # already unloading: observed mean minus observed age.
            return max(0.0, t[88]*600 - t[87]*600)
        if phase == 5:
            return _moving_seconds(t, cfg, loaded=True) + dump_queue + buffer_delay + dump
        if phase in (6, 7, 8):
            return dump_queue + buffer_delay + dump
        return math.inf
    target = _onehot_index(t[26:30])
    if target is None or phase not in (2, 3, 4):
        return math.inf
    load = s[1107+47*target:1154+47*target]
    loaded_trip = loaded_paths[cfg['load_node_indices'][target], cfg['dump_node_indices'][0]]
    if phase == 4:
        own_load = max(0.0, t[85]*600 - t[84]*600)
        to_load = 0.0
        load_queue = 0.0
    else:
        to_load = _moving_seconds(t, cfg, loaded=False) if phase == 2 else 0.0
        own_load = _load_mean(load, q, cfg)
        load_queue = _queue_seconds(s, load, load=True, truck_id=tid, cfg=cfg)
    return to_load + load_queue + own_load + loaded_trip + dump_queue + buffer_delay + dump


def _fifo(s, cfg):
    rows = s[1391:1439].reshape(16, 3)
    packets, ended = [], False
    for valid, scaled_q, grade in rows:
        if not valid:
            ended = True
            if scaled_q != 0 or grade != 0:
                raise ValueError('absent FIFO slots must be zero')
            continue
        if ended or scaled_q <= 0 or not 0 <= grade <= 1:
            raise ValueError('FIFO must be contiguous, positive and graded')
        packets.append((scaled_q*500, grade, 0.0, True, False))
    q = sum(p[0] for p in packets)
    if not math.isclose(q, s[1300]*500, rel_tol=2e-5, abs_tol=0.002):
        raise ValueError('FIFO tons disagree with crusher buffer snapshot')
    return packets


def _project(fifo, arrivals, durations, rate):
    """Integrate existing arrivals and FIFO without dropping boundary leftovers."""
    totals = [dict(q=0.0, m=0.0, sd=0.0, unknown=0.0, pipeline=0.0) for _ in durations]
    queue = deque([list(p) for p in fifo])
    events = sorted((eta, tid, p) for eta, tid, p in arrivals if math.isfinite(eta))
    now, ei = 0.0, 0
    for wi, duration in enumerate(durations):
        end = now+duration
        while now < end-1e-8:
            while ei < len(events) and events[ei][0] <= now+1e-8:
                queue.append(list(events[ei][2]))
                ei += 1
            stop = min(end, events[ei][0]) if ei < len(events) else end
            budget = max(0.0, stop-now)*rate
            while budget > 1e-8 and queue:
                p = queue[0]
                q = min(p[0], budget)
                result = totals[wi]
                result['q'] += q
                result['m'] += q*p[1]
                result['sd'] += q*p[2]  # safe shared-latent upper summary, not independent.
                result['unknown'] += q*(not p[3])
                result['pipeline'] += q*p[4]
                p[0] -= q
                budget -= q
                if p[0] <= 1e-8:
                    queue.popleft()
            now = stop
    return totals


def _action_eta(s, aid, cfg, empty_paths, loaded_paths):
    egos = np.flatnonzero(s[27:1107:120])
    if len(egos) != 1 or not s[19+aid]:
        return math.inf
    tid = int(egos[0])
    t = s[27+120*tid:147+120*tid]
    node = _onehot_index(t[32:41])
    if node is None:
        return math.inf
    load = s[1107+47*aid:1154+47*aid]
    ore = bool(load[1])
    dump_index = 0 if ore else 1
    ds = s[1295+48*dump_index:1343+48*dump_index]
    return (empty_paths[node, cfg['load_node_indices'][aid]]
            + _queue_seconds(s, load, load=True, cfg=cfg)
            + _load_mean(load, s[23+aid]*193, cfg)
            + loaded_paths[cfg['load_node_indices'][aid], cfg['dump_node_indices'][dump_index]]
            + _queue_seconds(s, ds, load=False, cfg=cfg) + _dump_mean(t, cfg, ore=ore))


def build_features(state, *, config=None):
    """Return a fresh float32[128]; one row depends only on this raw snapshot."""
    cfg, s = _config(config), _state(state)
    scale, gs = cfg['window_tons_scale'], cfg['grade_deviation_scale']
    metal_scale = scale*gs
    lower, upper, center = s[13:16]
    active = not bool(s[17]) and int(round(s[2]*4)) < 4
    next_active = active and int(round(s[2]*4)) < 3
    durations = [s[3]*cfg['window_seconds'] if active else 0.0,
                 cfg['window_seconds'] if next_active else 0.0]
    rate = s[1304]*2500/3600
    current_q, current_m = s[4]*scale, s[5]*scale
    valid = current_q > cfg['quantity_epsilon_tons']
    f = dict(zip(_GLOBAL_NAMES, (
        s[0], s[1], s[2], s[3], float(active), float(next_active), s[17],
        current_q/scale, current_m/scale, current_m/current_q if valid else 0.0,
        float(valid), (current_m/current_q-center)/gs if valid else 0.0,
        (current_m-lower*current_q)/metal_scale, (upper*current_q-current_m)/metal_scale,
        s[7], s[8], s[9], max(0.0, s[11]-s[7]), max(0.0, s[12]-s[9]),
        s[1304], s[1302],
    )))
    for name in _POOL_NAMES:
        f[name] = 0.0
    fifo = _fifo(s, cfg)
    fq, fm = sum(p[0] for p in fifo), sum(p[0]*p[1] for p in fifo)
    f.update({'fifo.q': fq/scale, 'fifo.m': fm/scale,
              'fifo.center_balance': (fm-center*fq)/metal_scale})
    loaded_paths, empty_paths = _road_times(s, cfg, loaded=True), _road_times(s, cfg, loaded=False)
    arrivals, pending_q, unknown_q, unmodeled_q = [], 0.0, 0.0, 0.0
    reserved_known_q, reserved_variance, reserved_sd = 0.0, 0.0, 0.0
    for tid, t in enumerate(s[27:1107].reshape(9, 120)):
        loaded, ore = bool(t[17]), bool(t[18])
        q = t[19]*193 if loaded else t[20]*193
        if q < 0:
            raise ValueError('negative truck payload/reservation')
        if q <= cfg['quantity_epsilon_tons']:
            continue
        if not ore:
            f['loaded_waste.q' if loaded else 'reserved_waste.q'] += q/scale
            continue
        phase = _onehot_index(t[4:16])
        if loaded:
            if t[21] != 1:
                raise ValueError('loaded ore must have observed gradeKnown')
            grade, sd, known = t[22], 0.0, True
            if not 0 <= grade <= 1:
                raise ValueError('invalid observed payload grade')
            f['loaded_ore.q'] += q/scale
            f['loaded_ore.m'] += q*grade/scale
            f['loaded_ore.center_balance'] += q*(grade-center)/metal_scale
            stage = {5: 'transit', 6: 'waiting', 7: 'waiting', 8: 'waiting', 9: 'unloading'}.get(phase)
            if stage:
                f[f'loaded_{stage}.q'] += q/scale
                f[f'loaded_{stage}.m'] += q*grade/scale
        else:
            target = _onehot_index(t[26:30])
            if target is None:
                raise ValueError('reserved ore requires observed loadTarget')
            load = s[1107+47*target:1154+47*target]
            if phase == 4:
                grade, sd, known = t[23], t[24], bool(t[25])
            else:
                grade, sd = load[2], load[3]/math.sqrt(3)
                known = sd == 0.0
            if not 0 <= grade <= 1 or sd < 0:
                raise ValueError('invalid causal reservation grade statistics')
            f['reserved_ore.q'] += q/scale
            f['reserved_ore.m_expected'] += q*grade/scale
            reserved_known_q += q*known
            reserved_variance += (q*sd)**2
            reserved_sd += q*sd
        eta = _arrival(s, tid, cfg, loaded_paths)
        arrivals.append((eta, tid, (q, grade, sd, known, True)))
        pending_q += q
        unknown_q += q*(not known)
        unmodeled_q += q*(not math.isfinite(eta))
        f['pending.count'] += 1/9
    reserved_q = f['reserved_ore.q']*scale
    f.update({'reserved_ore.metal_std_independent': math.sqrt(reserved_variance)/metal_scale,
              'reserved_ore.metal_std_correlated': reserved_sd/metal_scale,
              'reserved_ore.grade_known_fraction': reserved_known_q/reserved_q if reserved_q else 0.0,
              'pending.grade_unknown_q': unknown_q/scale,
              'pending.eta_unavailable_q': unmodeled_q/scale,
              'pending.arrival_time_uncertain': float(pending_q > 0)})
    baseline = _project(fifo, [], durations, rate)
    nominal = _project(fifo, arrivals, durations, rate)
    supports = [(load[2]-load[3], load[2]+load[3])
                for aid, load in enumerate(s[1107:1295].reshape(4, 47))
                if load[1] and s[19+aid]]
    for wi, prefix in enumerate(('current', 'next')):
        for name in _WINDOW_NAMES:
            f[f'{prefix}.{name}'] = 0.0
        if not (active if wi == 0 else next_active):
            continue
        q0, m0 = (current_q, current_m) if wi == 0 else (0.0, 0.0)
        n, b = nominal[wi], baseline[wi]
        capacity = durations[wi]*rate
        q, m = q0+n['q'], m0+n['m']
        free = max(0.0, capacity-n['q'])
        grade_valid = q > cfg['quantity_epsilon_tons']
        required_valid = free >= cfg['minimum_adjustable_tons']
        gl = (lower*(q+free)-m)/free if required_valid else 0.0
        gu = (upper*(q+free)-m)/free if required_valid else 0.0
        intersects = (required_valid and bool(supports)
                      and gl <= max(hi for _, hi in supports)
                      and gu >= min(lo for lo, _ in supports))
        values = (
            durations[wi]/cfg['window_seconds'], capacity/scale, b['q']/scale, b['m']/scale,
            q/scale, m/scale, n['sd']/metal_scale, m/q if grade_valid else 0.0,
            float(grade_valid), (m-lower*q)/metal_scale, (upper*q-m)/metal_scale,
            (m/q-center)/gs if grade_valid else 0.0, free/scale,
            gl, gu, float(required_valid), float(intersects),
            (q0+b['q'])/scale, (q0+min(capacity, b['q']+pending_q))/scale,
            n['unknown']/q if grade_valid else 0.0,
            n['pipeline']/q if grade_valid else 0.0, unmodeled_q/scale,
        )
        f.update({f'{prefix}.{name}': value for name, value in zip(_WINDOW_NAMES, values)})
    for aid, load in enumerate(s[1107:1295].reshape(4, 47)):
        eta = _action_eta(s, aid, cfg, empty_paths, loaded_paths)
        queue_count = len(set(_ids(load[20:29])+_ids(load[29:38])))
        values = (s[19+aid], s[23+aid], load[1], load[2], load[3]/math.sqrt(3),
                  load[7], queue_count/9, load[15]/max(1.0, load[14]),
                  eta/cfg['window_seconds'] if math.isfinite(eta) else 0.0,
                  float(math.isfinite(eta)))
        f.update({f'action.{aid}.{name}': value for name, value in zip(_ACTION_NAMES, values)})
    result = np.asarray([f[name] for name in FEATURE_NAMES], dtype=np.float32)
    if result.shape != (FEATURE_SIZE,) or not np.isfinite(result).all():
        raise ValueError('derived summary is not finite float32[128]')
    return result


def build_sequence(states, *, config=None):
    """Per-row phi, preserving prefixes exactly; no future normalization/fitting."""
    cfg = _config(config)
    values = np.asarray(states)
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 1519:
        raise ValueError('summary sequence requires float32 states[N,1519]')
    if not len(values):
        return np.empty((0, FEATURE_SIZE), dtype=np.float32)
    return np.stack([build_features(row, config=cfg) for row in values])


__all__ = ['FEATURE_VERSION', 'FEATURE_CONFIG', 'FEATURE_NAMES', 'FEATURE_SIZE',
           'feature_contract', 'feature_config_sha256', 'build_features', 'build_sequence']
