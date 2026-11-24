"""Causal adapter for a single existing float32 dispatch.physical.v3 snapshot.

This module does not load episodes, journals, models, seeds or future labels.
Dump/FIFO resource scheduling is delegated to ``engine``. Load queues and travel
provide explicitly approximate, frozen-current-work release times, not a joint
shovel/traffic counterfactual. The historical 128D feature definition is untouched.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from grade_commitment_ppo_v1 import features as old
from grade_event_memory_v1.events import Event, validate_event
from .engine import Candidate, Job, Snapshot


VERSION = 'physical-realization-observation-v1'
EVIDENCE_PROFILES = ('deployable_observation_prefix', 'prefix_verified_audit')
Q_ATOL = 0.002  # Existing raw float32 versus double-ledger admission scale.


@dataclass(frozen=True)
class CausalPrefix:
    """A caller-created slice, never a full tape or suffix retrieval capability."""
    marker_seq: int
    decision_sequence: int
    time_seconds: float
    events: tuple[Event, ...]


@dataclass(frozen=True)
class Adaptation:
    snapshot: Snapshot | None
    candidates: tuple[Candidate, ...]
    fifo_snapshot: Snapshot | None
    status: str
    reason_codes: tuple[str, ...]
    assumptions: tuple[str, ...]
    coverage: dict
    observed: dict
    state_sha256: str | None
    evidence_profile: str


def _near(a, b):
    return math.isclose(float(a), float(b), abs_tol=Q_ATOL, rel_tol=2e-6)


def _prefix_fifo(prefix, s, now):
    if type(prefix) is not CausalPrefix or type(prefix.events) is not tuple:
        raise ValueError('prefix_requires_immutable_causal_slice')
    if (type(prefix.marker_seq) is not int or prefix.marker_seq <= 0
            or type(prefix.decision_sequence) is not int or prefix.decision_sequence < 0
            or not math.isfinite(prefix.time_seconds)
            or abs(prefix.time_seconds-now) > .004):
        raise ValueError('prefix_cut_metadata_mismatch')
    fifo, q, m, seq = deque(), 0., 0., 0
    decisions = 0
    for e in prefix.events:
        if (type(e) is not Event or not seq < e.seq < prefix.marker_seq
                or e.kind == 'END' or e.time > prefix.time_seconds+1e-7):
            raise ValueError('prefix_contains_future_or_unordered_event')
        validate_event(e)
        seq = e.seq
        if e.kind == 'UNLOAD':
            fifo.append([e.batch_id, e.tons, e.grade, e.source_id, e.truck_id])
        elif e.kind == 'PROCESS':
            if not fifo or fifo[0][0] != e.batch_id or e.tons > fifo[0][1]+1e-7:
                raise ValueError('prefix_FIFO_order_or_mass_mismatch')
            fifo[0][1] -= e.tons
            if fifo[0][1] <= 1e-8:
                fifo.popleft()
            q += e.tons
            m += e.metal
        elif e.kind == 'WINDOW_CLOSE':
            q = m = 0.
        elif e.kind == 'DECISION':
            if e.decision_sequence != decisions:
                raise ValueError('prefix_decision_sequence_mismatch')
            decisions += 1
        else:
            raise ValueError('prefix_unknown_event_kind')
    if decisions != prefix.decision_sequence:
        raise ValueError('prefix_decision_count_mismatch')
    rows = s[1391:1439].reshape(16, 3)
    active = [r for r in rows if r[0]]
    if len(active) != len(fifo) or not _near(s[4]*5000, q) or not _near(s[5]*5000, m):
        raise ValueError('prefix_snapshot_material_mismatch')
    for row, packet in zip(active, fifo):
        if not _near(row[1]*500, packet[1]) or abs(row[2]-packet[2]) > 2e-7:
            raise ValueError('prefix_snapshot_FIFO_mismatch')
    return tuple(tuple(p) for p in fifo)


def _remaining_service(mean, age, *, deterministic=False):
    """E[D-age | D>age] for public ordinary triangular(.9,1,1.1)*mean.

    This is a declared duration model, never a read of the latent service draw.
    Special long-load profiles are outside this adapter's support.
    """
    if not math.isfinite(mean) or mean <= 0 or age < 0:
        return None, 'active_service_mean_or_age_invalid'
    if deterministic:
        return (mean-age, None) if age < mean-1e-7 else (None,'active_service_outside_duration_support')
    x = age/mean
    if x >= 1.1-1e-9:
        return None, 'active_service_outside_duration_support'
    if x <= .9:
        return mean-age, None
    if x >= 1.:
        return mean*(1.1-x)/3, None
    survival = 1-(x-.9)**2/.02
    integrated_survival = (1-x)+(x-.9)**3/.06
    return mean*integrated_survival/survival, None


def adapt_observation(state, *, prefix=None, evidence_profile='deployable_observation_prefix',
                      tick_period=.2, tick_anchor=0., tick_phase_observed=False):
    """Return partial/invalid coverage rather than aborting a whole shift.

    Only the raw snapshot is accepted; ``prefix`` is an optional, already cut
    immutable audit slice. Resource ambiguity invalidates the joint snapshot,
    while ``fifo_snapshot`` retains a clearly labeled FIFO-only fallback.
    All four original mask entries are retained. No action is made illegal here.
    """
    reasons, assumptions = [], [
        'fixed_work_conserving_crusher_rate',
        'load_calendar_frozen_current_queue_work_not_joint_shovel_schedule',
        'observed_route_nominal_speed_not_future_traffic',
        'future_equal_time_order_is_engine_scenario_not_observed_stack',
        'ordinary_service_triangular_0.9_1_1.1_not_special_long_load_profile',
    ]
    digest, observed, coverage = None, {}, {}
    candidates, fifo_snapshot = (), None
    try:
        if evidence_profile not in EVIDENCE_PROFILES:
            raise ValueError('unknown_evidence_profile')
        if (prefix is None) != (evidence_profile == 'deployable_observation_prefix'):
            raise ValueError('evidence_profile_prefix_mismatch')
        if (type(tick_phase_observed) is not bool or tick_phase_observed
                or not math.isfinite(tick_period) or tick_period <= 0
                or not math.isfinite(tick_anchor)):
            raise ValueError('crusher_tick_phase_not_observed_in_1519')
        cfg, s = old._config(None), old._state(state)
        digest = hashlib.sha256(np.asarray(state).tobytes(order='C')).hexdigest()
        if s[17]:
            raise ValueError('terminal_is_not_an_actionable_cut')
        now, wi = float(s[0]*28800), int(round(s[2]*4))
        remaining = float(s[3]*7200)
        if abs((wi+1)*7200-now-remaining) > .005:
            raise ValueError('snapshot_clocks_disagree')
        rate, capacity = float(s[1304]*2500/3600), float(s[1303]*500)
        if rate <= 0 or capacity <= 0:
            raise ValueError('nonpositive_crusher_rate_or_buffer_capacity')
        raw_fifo = old._fifo(s, cfg)
        identities = _prefix_fifo(prefix, s, now) if prefix is not None else None
        fifo = tuple(Job(
            key=(identities[i][0] if identities else f'fifo_slot_{i}'),
            tons=float(p[0]), grade=float(p[1]), stage='fifo', origin='initial_fifo',
            source_id=(identities[i][3] if identities and identities[i][3] >= 0 else None),
            truck_id=(identities[i][4] if identities and identities[i][4] >= 0 else None),
            grade_kind='observed_known') for i, p in enumerate(raw_fifo))
        base = dict(now=now, current_window=wi, window_q=float(s[4]*5000),
                    window_m=float(s[5]*5000), initial_fifo=fifo,
                    buffer_capacity=capacity, buffer_stored=sum(p.tons for p in fifo),
                    rate=rate, tick_period=tick_period, tick_anchor=tick_anchor,
                    tick_phase_observed=False)
        fifo_snapshot = Snapshot(**base, jobs=(), buffer_reserved=0.)
        assumptions += ['crusher_tick_anchor_is_documented_assumption_not_observed_phase']
        loaded_paths = old._road_times(s, cfg, loaded=True)
        empty_paths = old._road_times(s, cfg, loaded=False)
        trucks = s[27:1107].reshape(9, 120)
        jobs, by_id = [], {}
        unknown_source_q = expected_grade_q = unmodeled_q = 0.
        active_over_mean = 0
        fatal = []
        for tid, t in enumerate(trucks):
            loaded, payload, reserved = bool(t[17]), float(t[19]*193), float(t[20]*193)
            if loaded and reserved > 1e-5:
                fatal.append('loaded_and_reserved_overlap')
                continue
            if not loaded and payload > 1e-5:
                fatal.append('payload_without_loaded_flag')
                continue
            q = payload if loaded else reserved
            if q <= 1e-5:
                continue
            ore, phase = bool(t[18]), old._onehot_index(t[4:16])
            target = old._onehot_index(t[26:30])
            source = target if not loaded else None
            if loaded and ore:
                grade = float(t[22]) if t[21] else None
                kind = 'observed_known' if t[21] else 'unknown'
            elif not loaded and target is not None and ore:
                load = s[1107+47*target:1154+47*target]
                grade = float(t[23] if phase == 4 else load[2])
                kind = 'expected_loading_statistics' if phase == 4 else 'expected_source_prior'
                expected_grade_q += q
            else:
                grade, kind = None, 'not_applicable' if not ore else 'unknown'
            if source is None:
                unknown_source_q += q
            service = old._dump_mean(t, cfg, ore=ore)
            stage, arrival, remaining_service = 'unmodeled', None, None
            if loaded and phase in (6, 7, 8, 9):
                stage = {6:'queued', 7:'capacity_wait', 8:'admission_pending', 9:'unloading_reserved'}[phase]
                if phase == 9:
                    active_over_mean += int(t[87] >= t[88])
                    remaining_service, error = _remaining_service(float(t[88]*600), float(t[87]*600),
                                                                   deterministic=bool(s[16]))
                    if error:
                        fatal.append(error)
            elif loaded and phase == 5:
                stage = 'arrival'
                arrival = now+old._moving_seconds(t, cfg, loaded=True)
            elif not loaded and target is not None and phase in (2, 3, 4):
                load = s[1107+47*target:1154+47*target]
                if phase == 4:
                    active_over_mean += int(t[84] >= t[85])
                    duration, error = _remaining_service(float(t[85]*600), float(t[84]*600),
                                                         deterministic=bool(s[16]))
                    if error:
                        reasons.append(error)
                    to_load, load_queue = 0., 0.
                else:
                    duration = old._load_mean(load, q, cfg)
                    to_load = old._moving_seconds(t, cfg, loaded=False) if phase == 2 else 0.
                    load_queue = old._queue_seconds(s, load, load=True, truck_id=tid, cfg=cfg)
                trip = loaded_paths[cfg['load_node_indices'][target], cfg['dump_node_indices'][0 if ore else 1]]
                if duration is not None and all(math.isfinite(v) for v in (duration,to_load,load_queue,trip)):
                    stage, arrival = 'arrival', now+duration+to_load+load_queue+trip
            if ore and grade is None:
                reasons.append('loaded_ore_grade_not_observed')
                stage, arrival = 'unmodeled', None
            if not math.isfinite(service):
                stage, arrival = 'unmodeled', None
                service = None
            if stage == 'unmodeled':
                unmodeled_q += q
                reasons.append('existing_commitment_eta_or_grade_unmodeled')
            job = Job(key=f'truck_{tid}_current_commitment', tons=q, grade=grade,
                      commodity='ore' if ore else 'waste', stage=stage,
                      arrival_time=float(arrival) if arrival is not None else None,
                      service_seconds=float(service) if service is not None else None,
                      remaining_service_seconds=float(remaining_service) if remaining_service is not None else None,
                      origin='existing_pipeline',
                      source_id=source, truck_id=tid, grade_kind=kind)
            jobs.append(job)
            by_id[tid] = job
        # Validate explicit dump owners. Busy includes capacity-waiting holders.
        ordered, server_counts = [], []
        for did in range(2):
            ds = s[1295+48*did:1343+48*did]
            servers, busy = int(round(ds[2]*3)), float(ds[3]*3)
            if servers <= 0 or abs(ds[2]*3-servers) > 2e-6:
                fatal.append('dump_server_capacity_invalid')
            server_counts.append(servers)
            lists = [old._ids(ds[a:b]) for a,b in ((12,21),(21,30),(30,39),(39,48))]
            if any(len(ids) != len(set(ids)) for ids in lists):
                fatal.append('dump_owner_duplicate_within_list')
            queue = list(dict.fromkeys(lists[0]+lists[1]))
            capwait, services = lists[2], lists[3]
            pending_flag = bool(ds[10])
            pending_id = int(round(ds[11]*9))-1 if ds[11] else None
            if pending_flag or pending_id is not None:
                fatal.append('admission_pending_reservation_owner_ambiguous')
            if len(set(capwait+services)) != len(capwait)+len(services):
                fatal.append('dump_holder_stage_overlap')
            expected = {'queued':queue, 'capacity_wait':capwait, 'unloading_reserved':services}
            for stage, ids in expected.items():
                for tid in ids:
                    job = by_id.get(tid)
                    if job is None or job.commodity != ('ore' if did == 0 else 'waste') or job.stage != stage:
                        fatal.append('dump_owner_phase_or_commodity_mismatch')
                    elif job not in ordered:
                        ordered.append(job)
            for job in jobs:
                if job.commodity == ('ore' if did == 0 else 'waste') and job.stage in expected:
                    if job.truck_id not in expected[job.stage]:
                        fatal.append('dump_phase_missing_observed_owner')
            if not _near(busy, len(capwait)+len(services)):
                fatal.append('dump_busy_owner_count_mismatch')
            if did == 0:
                service_q = sum(by_id[t].tons for t in services if t in by_id)
                if not _near(ds[6]*500, service_q):
                    fatal.append('buffer_reservation_owner_mismatch')
                if not _near(ds[5]*500+ds[6]*500+ds[7]*500, capacity):
                    fatal.append('buffer_stored_reserved_free_inconsistent')
                if ds[5]*500+ds[6]*500 > capacity+Q_ATOL:
                    fatal.append('buffer_capacity_exceeded')
        ordered += [j for j in jobs if j not in ordered]
        ego = int(np.flatnonzero(s[27:1107:120])[0])
        et = trucks[ego]
        node = old._onehot_index(et[32:41])
        cands = []
        for aid in range(4):
            legal = bool(s[19+aid])
            candidate = None
            if legal:
                load = s[1107+47*aid:1154+47*aid]
                q, ore = float(s[23+aid]*193), bool(load[1])
                if q <= 1e-5:
                    reasons.append('legal_candidate_nonpositive_quantity')
                    cands.append(Candidate(action=aid,legal=True,job=None))
                    continue
                service = old._dump_mean(et, cfg, ore=ore)
                eta = (empty_paths[node, cfg['load_node_indices'][aid]]
                       + old._queue_seconds(s, load, load=True, cfg=cfg)
                       + old._load_mean(load, q, cfg)
                       + loaded_paths[cfg['load_node_indices'][aid], cfg['dump_node_indices'][0 if ore else 1]]) if node is not None else math.inf
                valid = q > 1e-5 and math.isfinite(eta) and math.isfinite(service)
                candidate = Job(key=f'candidate_action_{aid}', tons=q,
                    grade=float(load[2]) if ore else None, commodity='ore' if ore else 'waste',
                    stage='arrival' if valid else 'unmodeled', arrival_time=float(now+eta) if valid else None,
                    service_seconds=float(service) if math.isfinite(service) else None,
                    origin='candidate', source_id=aid, truck_id=ego,
                    grade_kind='expected_source_prior' if ore else 'not_applicable')
                if not valid:
                    reasons.append('candidate_route_or_service_unmodeled')
            cands.append(Candidate(action=aid, legal=legal, job=candidate))
        candidates = tuple(cands)
        coverage = dict(existing_input_q=sum(j.tons for j in jobs),
                        fifo_input_q=sum(j.tons for j in fifo),
                        existing_unmodeled_q=unmodeled_q, expected_grade_q=expected_grade_q,
                        active_service_at_or_over_mean_count=active_over_mean,
                        source_unknown_q=unknown_source_q, tick_assumed=True,
                        tie_assumed=True, load_calendar_approximate=True,
                        routing_feedback_unmodeled=True,
                        reservation_owners_verified=not any('reservation' in r for r in fatal),
                        resource_owners_verified=not fatal)
        observed = dict(now=now,current_window=wi,current_window_q=base['window_q'],
                        current_window_m=base['window_m'],ore_shift_q=float(s[7]*20000),
                        waste_shift_q=float(s[9]*4000),grade_lower=float(s[13]),
                        grade_upper=float(s[14]),grade_center=float(s[15]),
                        buffer_stored=float(s[1300]*500),buffer_reserved=float(s[1301]*500),
                        buffer_capacity=capacity,rate=rate,remaining_seconds=remaining,
                        source_grade_support_upper=max(
                            [float(load[2]+load[3]) for load in s[1107:1295].reshape(4,47) if load[1]]
                            + [float(t[22]) for t in trucks if t[17] and t[18] and t[21]], default=None),
                        mask=tuple(bool(v) for v in s[19:23]))
        reasons += fatal
        snapshot = None if fatal else Snapshot(**base,jobs=tuple(ordered),
            buffer_reserved=float(s[1301]*500),ore_servers=server_counts[0],waste_servers=server_counts[1],
            resource_owners_verified=True,reservation_owners_verified=True)
        return Adaptation(snapshot,candidates,fifo_snapshot,'invalid' if fatal else 'partial',
            tuple(dict.fromkeys(reasons)),tuple(assumptions),coverage,observed,digest,evidence_profile)
    except (ValueError,TypeError,IndexError,OverflowError) as error:
        return Adaptation(None,candidates,fifo_snapshot,'invalid',
            tuple(dict.fromkeys(reasons+[str(error)])),tuple(assumptions),coverage,observed,digest,evidence_profile)


__all__ = ['VERSION','CausalPrefix','Adaptation','adapt_observation']
