"""Settle and verify the standalone loop_B retry after AnyLogic has closed."""

import sys

from prepare_retry import HERE, check_fresh, load_case, require, witness

import collect_trial


def load_retry(label):
    require(label == 'B', 'repair contains only loop_B')
    return load_case()


def assess(label, config, job, finalize_incomplete):
    physical, native = collect_trial.assess(label, config, job, finalize_incomplete)
    physical['original_aborted_trial'] = config['original_aborted_trial']
    return physical, native


register = witness.bind(witness.ledger.register, check_fresh=check_fresh)
settle = witness.bind(witness.settle, register=register)
main = witness.bind(collect_trial.main, HERE=HERE, CASES=('B',), load_case=load_retry,
                    assess=assess, settle=settle)


if __name__ == '__main__':
    sys.exit(main())
