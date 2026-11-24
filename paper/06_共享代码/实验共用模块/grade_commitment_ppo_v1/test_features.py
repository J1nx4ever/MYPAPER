"""Analytical synthetic snapshots only: no simulator, RPC or policy training."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from .features import (
    FEATURE_CONFIG, FEATURE_NAMES, FEATURE_SIZE, FEATURE_VERSION,
    build_features, build_sequence, feature_contract, feature_config_sha256,
)


def synthetic_state(*, window_index=0, remaining_seconds=7200.0, terminal=False):
    """Schema-valid artificial decision snapshot for feature/model unit tests."""
    s = np.zeros(1519, dtype=np.float32)
    now = window_index*7200+7200-remaining_seconds if window_index < 4 else 28800
    s[0], s[1], s[2], s[3] = now/28800, max(0, 28800-now)/28800, window_index/4, remaining_seconds/7200
    s[6] = 1
    s[11:16] = [0.8, 1, 0.26, 0.28, 0.27]
    s[17] = terminal
    if not terminal:
        s[19:23], s[23:27] = 1, 91.7/193
    for tid in range(9):
        start = 27+120*tid
        s[start+1+tid//3] = 1
        s[start+4+11] = 1
        s[start+16] = (91.7, 147, 193)[tid//3]/193
        s[start+32] = 1  # all parked at the crusher for this synthetic fixture.
    if not terminal:
        s[27], s[31:43] = 1, 0
        s[31] = 1
    for aid, (ore, grade) in enumerate(((1, .35), (1, .30), (1, .15), (0, 0))):
        start = 1107+47*aid
        s[start:start+8] = [1, ore, grade, .005 if ore else 0, 1, 1, 0, 1]
        s[start+10] = 1  # 500t blocks.
        s[start+14] = 1  # one shovel.
        s[start+18], s[start+19] = .5, .45  # 15t buckets, 0.45min/bucket.
    for did in range(2):
        start = 1295+48*did
        s[start:start+4] = [1, not did, 1/3, 0]
        if did == 0:
            s[start+7:start+10] = [1, 1, 1]  # 500t free/capacity, 2500t/h.
    edges = [(0, 2), (2, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3),
             (3, 5), (5, 3), (3, 6), (6, 3), (4, 7), (7, 4), (4, 8), (8, 4)]
    for rid, (a, b) in enumerate(edges):
        s[1439+5*rid:1444+5*rid] = [(a+1)/9, (b+1)/9, .1, .8, 0]
    return s


def fifo_state(s, packets):
    result = s.copy()
    result[1391:1439] = 0
    for k, (q, g) in enumerate(packets):
        result[1391+3*k:1394+3*k] = [1, q/500, g]
    result[1300] = sum(q for q, _ in packets)/500
    result[1302] = 1-result[1300]-result[1301]
    return result


def put_truck(s, tid=1, *, phase=5, q=100, grade=.3, loaded=True, ore=True, target=1):
    result = s.copy()
    start = 27+120*tid
    result[start+4:start+16] = 0
    result[start+4+phase] = 1
    result[start+17:start+26] = [loaded, ore, q/193 if loaded else 0,
                                0 if loaded else q/193, loaded,
                                grade if loaded else 0, 0, 0, 0]
    result[start+26:start+30] = 0
    if not loaded:
        result[start+26+target] = 1
    result[start+30:start+32] = [int(loaded and ore), int(loaded and not ore)]
    if phase == 5:
        result[start+62] = .1  # 1000m remaining, nominal 200s.
        result[start+43] = .5
    if phase == 9:
        result[start+88] = (1.8, 2, 2.3)[tid//3]/10
    if phase == 4:
        result[start+23:start+26] = [grade, .002, 0]
        result[start+85] = (0.7+.45*math.ceil(q/15))/10
    return result


def features(s):
    return dict(zip(FEATURE_NAMES, map(float, build_features(s))))


def test_contract_exact_dimensions_hashes_and_defensive_config_copy():
    assert FEATURE_SIZE == 128 == len(set(FEATURE_NAMES))
    contract = feature_contract()
    assert set(contract) == {'version', 'names_sha256', 'config', 'config_sha256',
                             'dimension', 'scope', 'source'}
    assert contract['version'] == FEATURE_VERSION
    assert contract['scope'] == 'causal_decision_summary_whole_shift'
    assert contract['source'] == 'dispatch.physical.v3'
    encoded = json.dumps(FEATURE_NAMES, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    assert contract['names_sha256'] == hashlib.sha256(encoded).hexdigest()
    assert contract['config_sha256'] == feature_config_sha256()
    contract['config']['load_node_indices'][0] = 8
    assert FEATURE_CONFIG['load_node_indices'][0] == 5
    changed = deepcopy(FEATURE_CONFIG)
    changed['loaded_nominal_mps'] = 7
    with pytest.raises(ValueError, match='frozen'):
        build_features(synthetic_state(), config=changed)


def test_empty_snapshot_validity_and_units():
    s = synthetic_state()
    x = build_features(s)
    f = features(s)
    assert x.dtype == np.float32 and x.shape == (128,) and np.isfinite(x).all()
    assert f['window.grade_valid'] == f['current.grade_valid'] == 0
    assert f['current.capacity'] == pytest.approx(1)
    assert f['next.capacity'] == pytest.approx(1)
    assert f['current.free_capacity'] == pytest.approx(1)
    assert f['current.required_grade_lower'] == pytest.approx(.26)
    assert f['current.required_grade_upper'] == pytest.approx(.28)
    assert f['current.required_grade_valid'] == 1
    # A blend of the .15 and .30 sources can lie in [.26,.28].
    assert f['current.required_interval_intersects_ore_support'] == 1
    assert f['action.0.dispatch_q'] == pytest.approx(91.7/193)
    assert f['action.0.grade_std'] == pytest.approx(.005/math.sqrt(3))
    assert f['action.3.ore'] == 0
    assert f['action.0.eta_valid'] == 1
    assert f['pending.arrival_time_uncertain'] == 0


def test_actual_processed_mass_grade_gaps_and_task_progress():
    s = synthetic_state(remaining_seconds=3600)
    s[4:6] = [1000/5000, 250/5000]
    s[7:10] = [10000/20000, 2650/20000, 3500/4000]
    f = features(s)
    assert f['window.processed_q']*5000 == pytest.approx(1000)
    assert f['window.processed_m']*5000 == pytest.approx(250)
    assert f['window.grade'] == pytest.approx(.25)
    assert f['window.center_error'] == pytest.approx(-2, abs=4e-6)
    assert f['window.lower_balance']*50 == pytest.approx(-10, abs=2e-5)
    assert f['window.upper_balance']*50 == pytest.approx(30, abs=2e-5)
    assert f['shift.ore_gap'] == pytest.approx(.3)
    assert f['shift.waste_gap'] == pytest.approx(.125)


def test_fifo_order_quantity_metal_conservation_and_boundary_carry():
    s = fifo_state(synthetic_state(remaining_seconds=144), [(200, .30), (300, .15)])
    f = features(s)
    assert f['fifo.q']*5000 == pytest.approx(500)
    assert f['fifo.m']*5000 == pytest.approx(105, abs=1e-5)
    assert f['current.fifo_q']*5000 == pytest.approx(100, abs=1e-5)
    assert f['current.fifo_m']*5000 == pytest.approx(30, abs=1e-5)
    assert f['next.fifo_q']*5000 == pytest.approx(400, abs=4e-5)
    assert f['next.fifo_m']*5000 == pytest.approx(75, abs=1e-5)
    assert (f['current.nominal_q']+f['next.nominal_q'])*5000 == pytest.approx(500, abs=4e-5)
    assert (f['current.nominal_m']+f['next.nominal_m'])*5000 == pytest.approx(105, abs=1e-5)
    assert f['current.q_lower'] == f['current.q_upper']
    assert f['current.arrival_uncertain_fraction'] == 0
    reverse = features(fifo_state(synthetic_state(remaining_seconds=144), [(300, .15), (200, .30)]))
    assert reverse['current.fifo_m']*5000 == pytest.approx(15, abs=1e-5)


def test_known_loaded_unloading_nominal_arrival_fifo_and_required_grade():
    s = fifo_state(synthetic_state(remaining_seconds=180), [(50, .15)])
    s[4:6] = [1000/5000, 270/5000]
    s = put_truck(s, phase=9, q=100, grade=.3)
    # Observed unloading mean1.8min, age0 -> nominal completion108s.
    f = features(s)
    assert f['loaded_unloading.q']*5000 == pytest.approx(100, abs=1e-5)
    assert f['loaded_ore.m']*5000 == pytest.approx(30, abs=1e-5)
    assert f['current.nominal_q']*5000 == pytest.approx(1100, abs=2e-4)
    assert f['current.nominal_m']*5000 == pytest.approx(292.5, abs=2e-4)
    assert f['next.nominal_q']*5000 == pytest.approx(50, abs=2e-5)
    assert f['next.nominal_m']*5000 == pytest.approx(15, abs=1e-5)
    assert f['current.free_capacity']*5000 == pytest.approx(25, abs=2e-5)
    assert f['current.required_grade_lower'] == pytest.approx(0, abs=3e-6)
    assert f['current.required_grade_upper'] == pytest.approx(.9, abs=4e-6)
    assert f['current.required_grade_valid'] == 1
    assert f['pending.arrival_time_uncertain'] == 1
    assert f['current.arrival_uncertain_fraction'] > 0
    assert f['current.q_lower'] <= f['current.nominal_q'] <= f['current.q_upper']


def test_reservations_use_observed_prior_not_unloaded_true_payload_grade():
    s = put_truck(synthetic_state(), loaded=False, phase=2, q=100, grade=.99, target=2)
    f = features(s)
    assert f['reserved_ore.q']*5000 == pytest.approx(100, abs=1e-5)
    assert f['reserved_ore.m_expected']*5000 == pytest.approx(15, abs=1e-5)
    assert f['reserved_ore.metal_std_correlated']*50 == pytest.approx(100*.005/math.sqrt(3))
    hidden = s.copy()
    hidden[27+120+22] = .99
    hidden[27+120+23:27+120+26] = [.8, .1, 1]  # not loading: these are not consumed.
    assert np.array_equal(build_features(s), build_features(hidden))
    assert f['pending.grade_unknown_q']*5000 == pytest.approx(100, abs=1e-5)
    assert f['reserved_ore.grade_known_fraction'] == 0


def test_loading_stats_and_max_correlated_std_not_false_independence():
    s = put_truck(synthetic_state(), tid=1, phase=4, loaded=False, q=100, grade=.31)
    s = put_truck(s, tid=2, phase=4, loaded=False, q=100, grade=.29)
    f = features(s)
    assert f['reserved_ore.m_expected']*5000 == pytest.approx(60, abs=1e-5)
    assert f['reserved_ore.metal_std_independent']*50 == pytest.approx(math.sqrt(2)*.2)
    assert f['reserved_ore.metal_std_correlated']*50 == pytest.approx(.4)
    known = s.copy()
    known[27+120+24:27+120+26] = [0, 1]
    g = features(known)
    assert g['reserved_ore.grade_known_fraction'] == pytest.approx(.5)
    assert g['pending.grade_unknown_q']*5000 == pytest.approx(100, abs=1e-5)


def test_loaded_reservation_not_double_counted_and_waste_not_ore():
    s = put_truck(synthetic_state(), tid=1, q=100)
    s[27+120+20] = 100/193  # retained reservation metadata must not count twice.
    s = put_truck(s, tid=2, q=90, ore=False, grade=0)
    s = put_truck(s, tid=3, q=120, loaded=False, ore=False, phase=2, target=3)
    f = features(s)
    assert f['loaded_ore.q']*5000 == pytest.approx(100, abs=1e-5)
    assert f['loaded_transit.q'] == f['loaded_ore.q']
    assert f['reserved_ore.q'] == 0
    assert f['loaded_waste.q']*5000 == pytest.approx(90, abs=1e-5)
    assert f['reserved_waste.q']*5000 == pytest.approx(120, abs=1e-5)
    assert f['pending.count'] == pytest.approx(1/9)


def test_missing_route_phase_carries_unknown_not_fabricated_eta():
    s = put_truck(synthetic_state(), phase=11, q=100)
    f = features(s)
    assert f['pending.eta_unavailable_q']*5000 == pytest.approx(100, abs=1e-5)
    assert f['current.nominal_q'] == 0
    assert f['current.unmodeled_q']*5000 == pytest.approx(100, abs=1e-5)
    assert f['current.q_upper']*5000 == pytest.approx(100, abs=1e-5)
    assert f['current.q_lower'] == 0
    closed = synthetic_state()
    closed[1443:1519:5] = 1
    g = features(closed)
    assert g['action.0.eta_valid'] == g['action.0.nominal_eta'] == 0


def test_queue_work_delays_nominal_commitment_without_exact_eta_claim():
    s = put_truck(synthetic_state(remaining_seconds=400), tid=1, q=100, grade=.15)
    baseline = features(s)
    queued = put_truck(s, tid=2, q=100, phase=9, grade=.35)
    queued[1334] = 3/9  # dump.0.service.0 contains truck2.
    f = features(queued)
    # Extra current service shifts this truck's completion into the next window.
    assert baseline['current.nominal_q']*5000 == pytest.approx((400-200-108)*2500/3600, abs=1e-4)
    assert f['current.nominal_q']*5000 == pytest.approx(100, abs=1e-5)
    assert f['current.nominal_m']*5000 == pytest.approx(35, abs=1e-5)
    assert f['next.nominal_q']*5000 == pytest.approx(100, abs=1e-5)
    assert f['next.nominal_m']*5000 == pytest.approx(15, abs=1e-5)
    assert f['loaded_unloading.q']*5000 == pytest.approx(100, abs=1e-5)
    # Removing all trucks and adding a source queue changes only action ETA.
    src = synthetic_state()
    before = features(src)
    src = put_truck(src, tid=1, phase=3, q=100, loaded=False)
    src[1127+47] = 2/9
    after = features(src)
    assert after['action.1.queue_count'] == pytest.approx(1/9)
    assert after['action.1.nominal_eta'] > before['action.1.nominal_eta']


def test_no_adjustable_capacity_mask_and_last_window_terminal_behavior():
    s = fifo_state(synthetic_state(window_index=3, remaining_seconds=144), [(100, .30)])
    f = features(s)
    assert f['next.active'] == 0
    assert all(v == 0 for k, v in f.items() if k.startswith('next.') and k != 'next.active')
    assert f['current.required_grade_valid'] == 0
    assert f['current.required_grade_lower'] == f['current.required_grade_upper'] == 0
    terminal = synthetic_state(window_index=4, remaining_seconds=0, terminal=True)
    terminal[7:10] = [.9, .24, 1.02]
    g = features(terminal)
    assert g['terminal'] == 1 and g['window.active'] == g['next.active'] == 0
    assert g['shift.ore'] == pytest.approx(.9)
    assert all(v == 0 for k, v in g.items() if k.startswith('current.') or k.startswith('next.'))
    assert all(g[f'action.{a}.legal'] == g[f'action.{a}.eta_valid'] == 0 for a in range(4))


def test_sequence_prefix_invariant_row_snapshot_and_input_immutable():
    a = synthetic_state(remaining_seconds=180)
    b = fifo_state(synthetic_state(window_index=1), [(200, .3)])
    c = put_truck(synthetic_state(window_index=2), q=100, grade=.15)
    raw = np.stack((a, b, c))
    original = raw.copy()
    full = build_sequence(raw)
    assert full.shape == (3, 128) and full.dtype == np.float32
    assert np.array_equal(full[:2], build_sequence(raw[:2]))
    assert np.array_equal(full[1], build_features(b))
    changed_future = raw.copy()
    changed_future[2] = a
    assert np.array_equal(full[:2], build_sequence(changed_future)[:2])
    assert np.array_equal(raw, original)
    assert build_sequence(np.empty((0, 1519), np.float32)).shape == (0, 128)


def test_existing_real_shift_readonly_feature_compatibility():
    """Existing data only; no environment creation, simulator, fit or forward."""
    import h5py
    path = (Path(__file__).resolve().parents[1] / 'runs'
            / 'grade-only-adv-ab-seed1-20261108-001/arms/a0_gae095/t0/material/run_8910001.h5')
    if not path.is_file():
        pytest.skip('optional historical real run8910001 is not present in this checkout')
    with h5py.File(path, 'r') as handle:
        raw = handle['states'][...]
    before = hashlib.sha256(raw.tobytes()).hexdigest()
    result = build_sequence(raw)
    assert raw.shape == (174, 1519)
    assert result.shape == (174, 128) and np.isfinite(result).all()
    assert result[-1, FEATURE_NAMES.index('terminal')] == 1
    assert hashlib.sha256(raw.tobytes()).hexdigest() == before
    # Pure snapshots preserve the first observation independent of later data.
    assert np.array_equal(result[0], build_features(raw[0]))


@pytest.mark.parametrize('mutation,match', [
    (lambda s: s.astype(np.float64), 'float32'),
    (lambda s: s[:-1], '1519'),
    (lambda s: np.where(np.arange(1519) == 5, np.nan, s).astype(np.float32), 'nonfinite'),
    (lambda s: _changed(s, 19, .5), 'binary'),
    (lambda s: _changed(s, 27, 0), 'ego'),
    (lambda s: _changed(s, 2, .13), 'window'),
    (lambda s: _changed(s, 1300, .2), 'FIFO'),
    (lambda s: _changed(s, 1127, .05), 'queue'),
])
def test_strict_raw_shape_schema_validation(mutation, match):
    with pytest.raises(ValueError, match=match):
        build_features(mutation(synthetic_state()))


def _changed(s, i, v):
    s = s.copy()
    s[i] = v
    return s


def test_fifo_invalid_slots_and_loaded_grade_validity_are_rejected():
    s = _changed(synthetic_state(), 1393, .3)
    with pytest.raises(ValueError, match='absent FIFO'):
        build_features(s)
    s = fifo_state(synthetic_state(), [(50, .3), (50, .15)])
    s[1391:1394] = 0
    s[1300] = .1
    with pytest.raises(ValueError, match='contiguous'):
        build_features(s)
    s = put_truck(synthetic_state())
    s[27+120+21] = 0
    with pytest.raises(ValueError, match='gradeKnown'):
        build_features(s)


@pytest.mark.parametrize('states', [np.zeros(1519, np.float32), np.zeros((1, 1518), np.float32),
                                  np.zeros((0, 1519), np.float64)])
def test_sequence_rejects_invalid_shape_dtype(states):
    with pytest.raises(ValueError, match='sequence'):
        build_sequence(states)
