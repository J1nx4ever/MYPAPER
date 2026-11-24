from types import SimpleNamespace

import pytest

from traditional_dispatch_v1.runtime import _physical, _wait_breakdown


def _components():
    diagnostics = dict(breakdown_version=1.0, load_seconds=20.0, counted_load_seconds=20.0,
                       dump_total_seconds=30.0, dump_resource_queue_seconds=10.0,
                       dump_capacity_admission_seconds=20.0,
                       load_reconciliation_residual_seconds=0.0,
                       dump_reconciliation_residual_seconds=0.0,
                       total_reconciliation_residual_seconds=0.0, unclassified_flagged_seconds=0.0,
                       stage_errors=0.0, load_fallback_hits=0.0)
    return dict(ore_tons_shift=12000.0, waste_tons_shift=2000.0, wait_vehicle_seconds=50.0,
                **{f'diag.wait.{key}': value for key, value in diagnostics.items()})


def test_low_frequency_scoring_keeps_empty_window_and_physical_shortfalls():
    windows = [dict(window_index=i, ore_tons=q, metal_tons=q * g, grade=g)
               for i, (q, g) in enumerate(((4000., .27), (4000., .28), (4000., .27), (0., 0.)))]
    adapter = SimpleNamespace(config=dict(grade_epsilon=1e-12, grade_lower=.26, grade_upper=.28),
                              windows=windows, total_reward=-3.002)
    result = _physical(adapter, dict(reward_components=_components()))
    assert result['windows'][1]['compliant'] and result['windows'][1]['continuous_cost'] == pytest.approx(1.)
    assert result['windows'][3]['grade'] is None and not result['windows'][3]['compliant']
    assert not result['complete_task']
    assert result['production_cost'] == 6.
    assert result['combined_cost'] == pytest.approx(9. + .2 * 50. / 28800.)
    assert result['legacy_rpc_reward'] == -3.002


def test_terminal_wait_breakdown_rejects_unreconciled_or_missing_components():
    components = _components()
    assert _wait_breakdown(components)['dump_capacity_admission_seconds'] == 20.
    components['diag.wait.dump_capacity_admission_seconds'] = 19.
    with pytest.raises(ValueError, match='waiting reconciliation'):
        _wait_breakdown(components)
    del components['diag.wait.load_fallback_hits']
    with pytest.raises(ValueError, match='terminal wait breakdown'):
        _wait_breakdown(components)
