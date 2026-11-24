"""Offline regression: replay archived S1519 and RPC telemetry with v2 healthy-fleet fields."""
import numpy as np
from types import SimpleNamespace

from resilience_rpc import PROJECT, Shift, check_fleet, check_snapshot, wire
from resilience_runtime import LiveRPC
from physical import verify_terminal
from python.hdf5_storage import read_dataset
from python.reward_config import partition_components

ARCHIVE = (PROJECT/'runs/bg128-continuous-mixed-20261120-s02/arms/s01_BG128/train_u0014'
           /'episodes/run_10003424/samples.h5')


def expect_rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('invalid physical state accepted')


def test_retirement(states, infos, job):
    for s,info in zip(states,infos,strict=True):
        s = np.asarray(s, dtype=np.float32)
        trucks = s[27:1107].reshape(9,120)
        waste = np.flatnonzero((trucks[:,17] == 1) & (trucks[:,18] == 0) & (trucks[:,19] > 0))
        if len(waste):
            tid = int(waste[0])
            break
    else:
        raise AssertionError('no archived physical waste shipment')
    job = dict(job,fault_truck_id=tid,fault_trigger_seconds=6000.)
    survivor = 511 ^ (1 << tid)
    c = dict(info['reward_components'])
    def frame(now, trig, exit_time, d, p, ego, state=s):
        c.update({'fleet.trigger_seconds':trig, 'fleet.exit_seconds':exit_time,
                  'fleet.dispatchable_mask':d, 'fleet.in_system_mask':p})
        r = wire.DispatchRequest(sim_time_seconds=now,truck_id=ego,action_required=True,reward_components=c)
        delivered = wire.DispatchRequest.FromString(r.SerializeToString())
        return check_fleet(job,delivered,dict(delivered.reward_components),state)

    other = (tid+1)%9
    assert frame(5999.,-1.,-1.,511,511,tid) == (511,511)
    assert frame(6000.,6000.,-1.,survivor,511,other) == (survivor,511)
    check_snapshot(job,s,c)
    carried = float(trucks[tid,19])*193.
    missing = dict(c)
    missing['audit.waste_task.committed_tons'] -= carried
    expect_rejected(lambda: check_snapshot(job,s,missing))
    expect_rejected(lambda: frame(6000.,6000.,-1.,survivor,511,tid))
    expect_rejected(lambda: frame(6500.,6000.,6500.,survivor,survivor,other))

    parked = s.copy()
    fleet = parked[27:1107].reshape(9,120)
    fleet[tid,0] = 0
    fleet[tid,4:16] = 0
    fleet[tid,15] = 1
    fleet[tid,17:41] = 0
    fleet[tid,43:] = 0
    for refs in (fleet[:,81], parked[1107:1295].reshape(4,47)[:,20:],
                 parked[1295:1391].reshape(2,48)[:,11:]):
        refs[np.rint(refs*9) == tid+1] = 0
    assert frame(6500.,6000.,6500.,survivor,survivor,other,parked) == (survivor,survivor)
    return {'truck_id':tid, 'archived_waste_payload_tons':carried, 'verified_phases':3}


def run():
    rows, episodes, _, _ = read_dataset(ARCHIVE, lazy=True)
    with rows:
        ep = episodes[0]
        first = ep['start_info']
        states = [ep['initial_observation']]
        infos = [first]
        actions = []
        for row in rows:
            states.append(row['next_observation'])
            infos.append(row['settlement_info'])
            actions.append(row['action'])

        class RecordedActor:
            policy_version = 'offline-recorded-actions'
            seen = 0

            def act(self, state, dispatchable, in_system, generator, *, opening_mask):
                assert opening_mask == 511
                action = actions[self.seen]
                self.seen += 1
                return action, 0., 0.

        expected, _, _ = partition_components(first['reward_components'], terminal=False)
        fleet_config = {'config.fleet.outage_version':2.,'config.fleet.disabled_mask':0.,
                        'config.fleet.active_count':9.}
        job = dict(expected_rpc_config=dict(expected, **fleet_config), recording_id='archive_only',
                   ore_target=16000.,waste_target=4000.,initial_mask=511, fault_truck_id=-1,
                   fault_trigger_seconds=-1.,actor_sampling_seed=1, run_index=ep['experiment_run_index'],
                   experiment_seed=ep['experiment_seed'])
        shift = Shift(job, RecordedActor(), verify_terminal)
        service = LiveRPC()
        service.arm(shift, SimpleNamespace(claim=lambda *,final: final))

        class OfflineContext:
            def abort(self, status, details):
                raise ValueError(f'{status}: {details}')

        context = OfflineContext()
        settled_q = settled_m = 0.
        for i,(s,info) in enumerate(zip(states,infos,strict=True)):
            terminal = i == len(actions)
            c = dict(info['reward_components'], **fleet_config)
            c.update({'fleet.dispatchable_mask':511., 'fleet.in_system_mask':511.,
                      'fleet.trigger_seconds':-1., 'fleet.exit_seconds':-1.})
            windows = [wire.WindowSummary(**w) for w in info['closed_windows']]
            settled_q += sum(w.ore_tons for w in windows)
            settled_m += sum(w.metal_tons for w in windows)
            if terminal:
                fifo = [s[k:k+3] for k in range(1391,1439,3)]
                fifo_q = sum(float(v[1])*500 for v in fifo if v[0])
                fifo_m = sum(float(v[1])*float(v[2])*500 for v in fifo if v[0])
                c.update({'fleet.window_ore_tons':0., 'fleet.window_metal_tons':0.,
                          'fleet.processed_ore_tons':settled_q,'fleet.processed_metal_tons':settled_m,
                          'fleet.incoming_ore_tons':settled_q+fifo_q,'fleet.incoming_metal_tons':settled_m+fifo_m,
                          'fleet.fifo_ore_tons':fifo_q, 'fleet.fifo_metal_tons':fifo_m})
            request = wire.DispatchRequest(
                kind=wire.EPISODE_END if terminal else wire.EPISODE_START if i==0 else wire.DECISION,
                action_required=not terminal, terminated=terminal, obs=wire.Observation(
                    values=s, schema_id=info['observation_schema_id']), reward_components=c,
                transition_reward=info['transition_reward_raw'],closed_windows=windows,
                **{k:info[k] for k in ('decision_sequence','request_id','truck_id','episode_id',
                  'experiment_run_index','experiment_seed','reward_contract_id','previous_action',
                  'sim_time_seconds','sim_time_delta_seconds')})
            request = wire.DispatchRequest.FromString(request.SerializeToString())
            reply = service.GetNextDestination(request,context)
            assert reply == service.GetNextDestination(request,context)
        assert service.finished.is_set() and service.failure is None
        service.shift = None
        assert reply == service.GetNextDestination(request,context)
        assert len(actions) == shift.receipt['decisions'] and len(shift.closures) == 4
        return dict(archived_decisions=len(actions), verified_callbacks=len(infos), physical_runs=0,
                    synthetic_retirement=test_retirement(states,infos,job),
                    check='historical callbacks with supplied synthetic healthy-fleet telemetry')


if __name__ == '__main__':
    import json
    print(json.dumps(run()))
