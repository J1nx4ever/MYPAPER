"""Settle the B trial against the targets its opening fleet actually received."""

import math
import sys

from prepare_scaled import HERE, check_fresh, load_case, require, witness

import collect_trial


def verified_targets(scene, task):
    """Independently derive and check the targets from native opening slot payloads."""
    fleet = scene['opening_fleet']
    require(len(fleet) == 9 and sorted(row[0] for row in fleet) == list(range(9))
            and all(type(row[1]) is bool and type(row[3]) is int and row[3] > 0 for row in fleet),
            'invalid opening fleet in the scaled-target trial')
    active = [row[3] for row in fleet if row[1]]
    full = sum(row[3] for row in fleet) / 10.
    require(len(active) == 7 and full == 1295.1, 'unexpected B opening fleet or reference capacity')
    opening = sum(active) / 10.
    alpha = opening / full
    allowance = max(active) / 10.
    expected = dict(ore_target_tons=16000. * alpha, waste_target_tons=4000. * alpha,
                    waste_dispatch_stop_tons=6000. * alpha, waste_upper_tons=6000. * alpha + allowance)
    native = scene['exogenous_config']
    for key, value in expected.items():
        require(math.isclose(task[key], value, rel_tol=0, abs_tol=1e-9)
                and native[key] == task[key], f'{key} differs from the opening-capacity target')
    window_target = task['ore_target_tons'] / 4.
    require(native['ore_target_tons_2h'] == window_target, 'native two-hour ore target differs')
    return dict(active_trucks=len(active), opening_payload_tons=opening, nine_truck_payload_tons=full,
                capacity_ratio=alpha, one_truck_allowance_tons=allowance, ore_target_tons_2h=window_target,
                **{key: task[key] for key in expected})


def no_comparison(label, outcomes, task):
    # The earlier physical witness used a different target and stopping rule.
    return {}


assess_trial = witness.bind(collect_trial.assess, source_comparison=no_comparison)


def assess(label, config, job, finalize_incomplete):
    targets = verified_targets(config['expected_runtime_scene'], config['task'])
    physical, native = assess_trial(label, config, job, finalize_incomplete)
    physical['capacity_targets'] = targets
    if native is not None:
        observed = native[2]['config']['actual_runtime_scene']
        require(verified_targets(observed, config['task']) == targets, 'actual opening targets differ')
    return physical, native


register = witness.bind(witness.ledger.register, check_fresh=check_fresh)
settle = witness.bind(witness.settle, register=register)
main = witness.bind(collect_trial.main, HERE=HERE, CASES=('B',), load_case=load_case,
                    assess=assess, settle=settle)


if __name__ == '__main__':
    sys.exit(main())
