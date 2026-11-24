"""v2 RPC fixtures; no simulator is constructed by this suite."""
import numpy as np
import pytest

from grade_commitment_ppo_v1.test_features import synthetic_state, fifo_state, put_truck
from .runtime import (RequestIdentity, RPCAdmission, configuration_from_rpc,
                      validate_snapshot, quantization_error)


def components(committed=4500., delivered=4500.):
    return {'config.'+key: value for key, value in dict(
        production_reward_version=2, wait_kappa=10, shift_seconds=28800,
        window_seconds=7200, target_ore_tons_shift=16000,
        target_waste_tons_shift=4000, w_production=1, waste_shortfall_weight=1,
        grade_lower=.26, grade_upper=.28, target_grade=.27, grade_epsilon=1e-12,
        tonnage_epsilon=1e-6, mass_balance_epsilon=1e-6, num_trucks=9,
        waste_task_limit_enabled=1, waste_task_limit_version=2,
        waste_task_dispatch_stop_tons=6000, waste_task_upper_tons=6193,
        waste_task_overrun_tons=193).items()} | {
        'audit.waste_task.committed_tons':committed,
        'audit.waste_task.delivered_tons':delivered,
        'audit.waste_task.assignment_closed':float(committed >= 6000-1e-6)}


def snapshot(delivered=4500., *, reserved=0., loaded=0., terminal=False):
    x = synthetic_state(terminal=terminal)
    x[9:11] = [delivered/4000, max(0, 4000-delivered)/4000]
    sites = x[1107:1295].reshape(4,47)
    sites[3,6] = reserved/30000
    sites[3,8] = (delivered+loaded)/30000
    sites[3,5] = 1-sites[3,8]
    sites[3,7] = sites[3,5]-sites[3,6]
    if reserved:
        x = put_truck(x, tid=8, phase=2, q=reserved, loaded=False, ore=False, target=3)
    if loaded:
        x = put_truck(x, tid=7, q=loaded, loaded=True, ore=False)
    if delivered+reserved+loaded >= 6000-1e-6:
        x[22] = x[26] = 0
    return x


def identity(sequence=0, request=None):
    return RequestIdentity('offline-fixture',1,17,'episode-1',request or str(sequence),sequence)


def test_configuration_missing_legacy_and_nonfinite_are_rejected():
    cfg, fingerprint = configuration_from_rpc(components())
    assert cfg.waste_stop == 6000 and cfg.max_truck_capacity == 193 and len(fingerprint)==64
    for key, value in [('config.waste_task_limit_version',1),
                       ('config.waste_task_upper_tons',4193),
                       ('config.grade_upper',float('nan'))]:
        c=components(); c[key]=value
        with pytest.raises(ValueError): configuration_from_rpc(c)
    c=components(); del c['config.waste_task_dispatch_stop_tons']
    with pytest.raises(ValueError): configuration_from_rpc(c)


def test_loaded_reserved_delivered_exclusive_bins_and_last_truck_overrun():
    x=snapshot(4500,reserved=150,loaded=193)
    result=validate_snapshot(x,components(4843,4500))
    assert result['committed_tons']==4843 and not result['closed']
    bridge=RPCAdmission()
    bridge.accept(identity(),snapshot(5900),components(5900,5900))
    result=bridge.accept(identity(1),snapshot(5900,reserved=193),components(6093,5900))
    assert result['closed'] and result['committed_tons']==6093
    x=snapshot(5900,reserved=193);x[22]=1;x[26]=.5
    with pytest.raises(ValueError): validate_snapshot(x,components(6093,5900))


def test_float32_threshold_never_replaces_original_hard_mask():
    # Float32 reconstruction crosses to the wrong side of the threshold.
    x=snapshot(5999.9999)
    assert float(x[1107+47*3+8])*30000 > 6000
    assert not validate_snapshot(x,components(5999.9999,5999.9999))['closed']
    x=snapshot(5965,reserved=35)
    assert float(x[1107+47*3+8])*30000+float(x[1107+47*3+6])*30000 < 6000
    assert validate_snapshot(x,components(6000,5965))['closed']
    assert quantization_error(np.asarray([5999.9999/30000],np.float32),30000)>0


def test_ore_progress_conserves_buffer_loaded_and_reservations_without_double_count():
    x=fifo_state(snapshot(),[(100,.27)])
    x[7]=600/20000
    x=put_truck(x,tid=6,q=193,ore=True)
    x=put_truck(x,tid=7,q=107,ore=True)
    x=put_truck(x,tid=4,phase=2,q=100,loaded=False,ore=True,target=0)
    x=put_truck(x,tid=5,phase=2,q=100,loaded=False,ore=True,target=0)
    sites=x[1107:1295].reshape(4,47)
    sites[0,8]=1000/30000;sites[0,6]=200/30000
    sites[0,5]=1-sites[0,8];sites[0,7]=sites[0,5]-sites[0,6]
    x[1301]=107/500;x[1302]=1-x[1300]-x[1301]
    result=validate_snapshot(x,components())
    assert result['crushed_tons']==pytest.approx(600,abs=.001)
    # Q_o is already part of loaded ore; adding it to produced violates mass.
    bad=x.copy();bad[1107+8]+=.01
    with pytest.raises(ValueError,match='ore bins'): validate_snapshot(bad,components())


def test_rpc_identity_lock_config_drift_and_duplicate_end_fail_closed():
    bridge=RPCAdmission();bridge.accept(identity(),snapshot(),components())
    with pytest.raises(ValueError): bridge.accept(identity(),snapshot(),components())
    with pytest.raises(ValueError): bridge.accept(identity(2),snapshot(),components())
    c=components();c['config.wait_kappa']=11
    with pytest.raises(ValueError,match='drift'): bridge.accept(identity(1),snapshot(),c)
    bridge.accept(identity(1),snapshot(terminal=True),components(),terminal=True)
    with pytest.raises(ValueError,match='after true END'):
        bridge.accept(identity(2),snapshot(),components())


def test_nonadjacent_request_id_cannot_be_reused():
    bridge=RPCAdmission()
    bridge.accept(identity(0,'first'),snapshot(),components())
    bridge.accept(identity(1,'second'),snapshot(),components())
    with pytest.raises(ValueError,match='reused'):
        bridge.accept(identity(2,'first'),snapshot(),components())


def test_missing_audit_wrong_ledger_and_nonfinite_fail_closed():
    for key, value in [('audit.waste_task.committed_tons',4501),
                       ('audit.waste_task.delivered_tons',4501),
                       ('audit.waste_task.assignment_closed',1),
                       ('audit.waste_task.delivered_tons',float('inf'))]:
        c=components();c[key]=value
        with pytest.raises(ValueError): validate_snapshot(snapshot(),c)
    c=components();del c['audit.waste_task.committed_tons']
    with pytest.raises(ValueError): validate_snapshot(snapshot(),c)
