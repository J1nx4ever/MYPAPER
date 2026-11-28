"""Freeze and settle the second, independently identified native prefix attempt."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import prepare_diagnostic as diagnostic


def main():
    campaign = 'site-layout-frozen-policy-diagnostic-20261127-002'
    run = 10054045
    diagnostic.HERE = HERE
    diagnostic.CAMPAIGN = campaign
    diagnostic.RUN = run
    diagnostic.SEED = 68869771655423
    diagnostic.RECORDING = f'{campaign}-{run}'
    diagnostic.SCRATCH = Path(f'/tmp/faith-topology-prefix-diagnostic-{run}')
    diagnostic.OUTPUT = diagnostic.PROJECT / 'runs' / f'{campaign}-B'
    diagnostic.CONFIG = HERE / 'configs' / 'B.json'
    diagnostic.TAPE = HERE / 'action_tape.jsonl'
    witness = diagnostic.witness
    prepared_rows = witness.bind(witness.prepared_rows, HERE=HERE)
    fresh = witness.bind(witness.check_fresh, prepared_rows=prepared_rows)
    diagnostic.check_fresh = fresh
    diagnostic.register = witness.bind(witness.ledger.register, check_fresh=fresh, identity=diagnostic.identity)
    diagnostic.transition = witness.bind(witness.ledger.transition, identity=diagnostic.identity)
    diagnostic.main()


if __name__ == '__main__':
    main()
