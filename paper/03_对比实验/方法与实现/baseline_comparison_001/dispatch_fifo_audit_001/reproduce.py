"""Read-only FIFO audit of the frozen DISPATCH-inspired V2 implementation."""
from hashlib import sha256
import json
from math import fsum, isclose
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from traditional_dispatch_v2 import dispatch_policy as dp, flow_plan as fp
from traditional_dispatch_v2.test_flow_plan import fast_snapshot


ARCHIVE = ROOT / 'runs/bg128-continuous-mixed-20261120-s02/evaluation/nominal/new/material/run_10004025.h5'


def checked_archive():
    digest = sha256(ARCHIVE.read_bytes()).hexdigest()
    seal = json.loads(ARCHIVE.with_suffix('.h5.seal.json').read_bytes())
    assert digest == seal['hdf5_sha256']
    with h5py.File(ARCHIVE) as archive:
        observations = archive['states'][[0, 8]]
        journal = archive['material_events']
        batch = (journal['batch_id'].asstr()[:] == 'batch-28')
        records = [{key: float(journal[key][i]) for key in ('interval_start', 'interval_end', 'tons', 'grade')}
                   for i in np.flatnonzero(batch & (journal['seq'][:] >= 145) & (journal['seq'][:] <= 147))]
    assert len(records) == 2 and records[0]['interval_end'] == records[1]['interval_start'] == 7200.
    assert isclose(fsum(row['tons'] for row in records), 91.7)
    return observations, digest, records


def fixed_backlog_example():
    # Time is relative to the current window. A committed packet arrives at t=20.
    # A new packet dispatched at d enters the crusher FIFO at d+10.
    def committed_balance(t):
        crushed = min(80., max(0., t - 20.))
        return (80. - crushed if t >= 20. else 0., crushed)

    replay = SimpleNamespace(_crusher_rate=1., ore_balance_at=committed_balance)
    route = fp._Route(0, 0, 10., 0, .38, 0., 0., 0., 10.)
    start, end = fp._ore_offsets(route, 0., replay)
    frozen_fraction = fp._mean_overlap((0., 100.), (0., 100.), start, end) / (end - start)
    # For d>10 the committed packet is first and uses t=20..100.
    fifo_fraction = 10. / 100.
    assert isclose(frozen_fraction, .85) and isclose(fifo_fraction, .10)
    return dict(frozen_fraction=frozen_fraction, fifo_fraction=fifo_fraction,
                new_packet_tons=10, committed_packet_arrival=20, committed_packet_tons=80,
                example_late_dispatch=70, frozen_processing_start=80, fifo_processing_start=100)


def fitted_kernel(observation, replay):
    windows = [(max(replay.now, i * fp.WINDOW_SECONDS), (i + 1) * fp.WINDOW_SECONDS)
               for i in range(replay.window_index, 4)]
    replay._move(check_deadline=None, dispatch=False)
    trucks = observation[27:1107].reshape(9, 120)
    types = np.argmax(trucks[:, 1:4], axis=1).tolist()
    launches, cost, matrix, limits, ore, metal, _ = fp._lp(replay, types, windows, replay.score_inputs())
    result = fp.linprog(cost, A_ub=matrix, b_ub=limits, bounds=(0., None), method='highs')
    assert result.success
    return launches, np.maximum(0., result.x), ore, metal


def published_peak(observation):
    replay = fp._CommitmentReplay.from_observation(observation)
    assert replay.now == 0 and not replay.initial_stock and not any(replay.initial_commitments)
    launches, x, ore, metal = fitted_kernel(observation, replay)
    kernels = [(float(q), launch.departure, launch.ore_processing_offsets)
               for q, launch in zip(x[:len(launches)], launches, strict=True)
               if q > 0 and launch.ore_processing_offsets is not None]
    knots = {time + offset for _, departure, offsets in kernels for time in departure for offset in offsets}

    def rate(t):
        return fsum(
            q * max(0., min(d1, t - o0) - max(d0, t - o1)) / ((d1 - d0) * (o1 - o0))
            for q, (d0, d1), (o0, o1) in kernels
        )

    # Each convolution is piecewise linear, so one of these knots attains the maximum.
    peak_rate, peak_time = max((rate(t), t) for t in knots if 0 <= t < fp.SHIFT_END)
    true_rate = float(replay._crusher_rate)
    assert peak_rate > true_rate + .01
    q, m = ore @ x, metal @ x
    return dict(predicted_window_tons=q.tolist(), predicted_window_grades=(m/q).tolist(),
                continuous_flow_peak_tph=peak_rate*3600, nominal_physical_limit_tph=true_rate*3600,
                peak_time_seconds=peak_time)


class RecordedAdmissions(fp._CommitmentReplay):
    def __init__(self, snapshot):
        self.admissions = []
        super().__init__(snapshot)

    def _push(self, timestamp, kind, resource, truck):
        if kind == fp._FINISH and resource == fp._DUMP_OFFSET:
            self.admissions.append((self.now, self._trips[truck].tons))
        super()._push(timestamp, kind, resource, truck)


def published_storage(observation):
    replay = RecordedAdmissions.from_observation(observation)
    launches, x, _, _ = fitted_kernel(observation, replay)
    t = 1633.
    entered = fsum(q for admitted, q in replay.admissions if admitted <= t)
    crushed = replay.ore_balance_at(t)[1]
    new_entries, new_processing = [], []
    for q, launch in zip(x[:len(launches)], launches, strict=True):
        if not q or launch.ore_processing_offsets is None:
            continue
        start, end = launch.departure
        lo, hi = launch.ore_processing_offsets
        entered_share = min(1., max(0., (t - launch.route.dump_start - start) / (end - start)))
        new_entries.append(q * entered_share)
        new_processing.append(q * fp._mean_overlap(launch.departure, (replay.origin, t), lo, hi) / (hi - lo))
    new_entry, new_crushed = fsum(new_entries), fsum(new_processing)
    combined = replay.initial_stock + entered - crushed + new_entry - new_crushed
    assert combined > replay._crusher_capacity
    return dict(evaluation_time_seconds=t, initial_stock_tons=replay.initial_stock,
                committed_entered_tons=entered, committed_crushed_tons=crushed,
                new_entered_tons=new_entry, new_crushed_tons=new_crushed,
                implied_buffer_plus_reservations_tons=combined, physical_limit_tons=replay._crusher_capacity)


def matching_counterexamples():
    truck = dp._TruckOptions(0, (), 10.)
    first = dp._Option(0, 100., 10., 10., 100., 30.)
    second = dp._Option(1, 100., 20., 20., 110., 30.)
    targets = [(dp._Target(0., 100., 100.),)] * 4
    servers = [[0.] for _ in range(4)]
    assigned = [0.] * 4

    def losses(priority):
        bookings = [[(10., priority, 8, 30., None)], [], [], []]
        return [dp._loss((option,), (truck,), targets, assigned, servers, bookings)
                for option in (first, second)]

    frozen, heap_tie = losses(1), losses(2)
    assert frozen[0] > frozen[1] and heap_tie[0] < heap_tie[1]

    tiny_target = [(dp._Target(0., 900., 1.),), (), (), ()]
    tiny_loss = dp._loss((first,), (truck,), tiny_target, assigned, servers, [[], [], [], []])
    assert tiny_loss == 0. and first.tons == 100.

    measured_stock = float(np.float32(147./30000)) * 30000
    measured_payload = float(np.float32(147./193)) * 193
    root = dp._Option(0, measured_payload, 10., 10., 100., 30.)
    observed = SimpleNamespace(ore=True, service=lambda q: 30. if q > 0. else 0.)
    accepted = dp._reserve((root, root), [measured_stock, 0., 0., 0.], (observed,), 0.)
    assert accepted is not None and 0. < accepted[1].tons < fp.MASS_TOL and accepted[1].loading_seconds == 30.

    boundary_time, _ = dp._need_time((dp._Target(0., 100., 100.), dp._Target(7200., 14400., 1.1e-6)),
                                     99.9999995)
    assert 100. < boundary_time < 7200.
    return dict(same_time_shovel_losses_frozen=frozen, same_time_losses_shared_heap_order=heap_tie,
                offered_tons=first.tons, target_tons=1., frozen_target_penalty=tiny_loss,
                available_tons_decoded=measured_stock, planned_tons_decoded=measured_payload,
                phantom_second_load_tons=accepted[1].tons,
                phantom_second_load_service_seconds=accepted[1].loading_seconds,
                inverse_due_time_seconds=boundary_time, next_target_start_seconds=7200)


def solver_clip_example():
    observation = fast_snapshot(window_index=3)
    observation[7], observation[9] = 16000/20000, 4000/4000
    observation[19:27] = 0
    observation[19], observation[23] = 1., 31/193
    loads = observation[1107:1295].reshape(4, 47)
    loads[:, 7] = 0.
    loads[0, 2], loads[0, 7] = .27, 31/30000
    solver = fp.linprog
    evidence = {}

    def limit_result(cost, **kwargs):
        result = solver(cost, **kwargs)
        columns = np.flatnonzero(kwargs['A_ub'][0])
        positive = next(i for i in columns if result.x[i] > 30.)
        zero = next(i for i in columns if result.x[i] == 0.)
        result.x[positive] += .0025
        result.x[zero] -= .0015
        result.x[-1] += .01
        result.status = 1
        matrix, limits = kwargs['A_ub'], kwargs['b_ub']
        evidence.update(raw_max_violation=float(max(matrix @ result.x - limits)),
                        clipped_supply_violation=float((matrix @ np.maximum(0., result.x) - limits)[0]))
        return result

    with patch.object(fp, 'linprog', limit_result):
        released = fp.solve_flow_plan(observation)
    assert released.status == 'feasible_limit'
    assert evidence['raw_max_violation'] <= fp.MASS_TOL < evidence['clipped_supply_violation']
    return dict(**evidence, published_status=released.status)


if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError('run with assertions enabled')
    obs, digest, crossing = checked_archive()
    print(json.dumps(dict(archive=dict(path=str(ARCHIVE), sha256=digest, frames=[0, 8]),
                          recorded_fifo_boundary_crossing=crossing,
                          fixed_backlog=fixed_backlog_example(), archive_peak=published_peak(obs[0]),
                          archive_storage=published_storage(obs[1]),
                          matching=matching_counterexamples(), solver_clip=solver_clip_example()), indent=2))
