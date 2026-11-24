"""Execute a single serial AnyLogic campaign with a strict 812-shift ledger."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from extension_common import (
    APPROVAL, CAMPAIGN, CONTROL, CORE, OUTPUT, PREPARED, REPORT, SUITE, TOTAL,
    checked, load_package, reference, require,
)
from business_graph_summary_campaign_v1.plan_io import encoded
from grade_only_ppo_v2.runtime import _controller_lock
from online_v1_1.operations import _publish
from extension_budget import ExtensionLedger
from evaluate_extension import evaluate_phase, write_paired_report
from train_extension import train_phase, verify_initial_checkpoint


def progress(status, **fields):
    row = dict(schema='bg128-training-extension-progress-v1', campaign_id=CAMPAIGN, status=status,
               updated_at_utc=datetime.now(timezone.utc).isoformat(), total_shifts=TOTAL, **fields)
    _publish(OUTPUT / 'run_status.json', encoded(row), replace=True)


def execute(suite, phases, ledger):
    import torch

    torch.set_num_threads(suite['cpu_threads'])
    proof = verify_initial_checkpoint(suite['parent_checkpoint'], suite['learner_config'])
    _publish(OUTPUT / 'initial_checkpoint_verification.json', encoded(proof))
    ledger.register()
    evaluations, checkpoints = {}, {16: suite['parent_checkpoint']}
    model = None
    shifts = 0
    for phase, cfg in zip(suite['phases'], phases):
        update = phase['update_index']
        role = phase['role']
        progress('EVALUATING_CHECKPOINT' if role == 'eval' else 'TRAINING_UPDATE',
                 phase_id=phase['phase_id'], update_index=update, completed_preceding_shifts=shifts)
        if role == 'eval':
            evaluations[update] = evaluate_phase(suite, phase, ledger, checkpoints[update])
        else:
            model, checkpoints[update] = train_phase(
                SUITE, suite, cfg, Path(phase['phase_config']['path']), ledger, model)
        shifts += len(cfg['jobs'])
        progress('PHASE_COMPLETE', phase_id=phase['phase_id'], update_index=update,
                 completed_shifts=shifts)
    state = ledger.snapshot()
    counts = Counter(row['status'] for row in state['jobs'].values())
    commits = state['campaigns'][CAMPAIGN]['commits']
    require(counts == {'COMPLETE': TOTAL} and set(map(int, commits)) == set(range(17, 33))
            and all(commits[str(i)]['checkpoint_ref'] == checkpoints[i] for i in range(17, 33))
            and model.update_index == 32 and len(model.consumed_episodes) == 1024,
            'extension completion counts, training lineage or immutable checkpoints differ')
    for ref in suite['parent_artifacts'].values():
        checked(ref)
    paired = write_paired_report(suite, list(evaluations.values()), REPORT)
    result = dict(status='COMPLETE', campaign=reference(SUITE), original_updates=16,
                  added_updates=list(range(17, 33)), cumulative_updates=32, physical_shifts=TOTAL,
                  new_training_shifts=512, cumulative_training_shifts=1024,
                  fixed_evaluation_shifts=300, checkpoints=checkpoints, fixed_evaluations=evaluations,
                  paired_reports=paired, protected_core=reference(CORE), ledger_status_counts=dict(counts))
    _publish(OUTPUT / 'suite_complete.json', encoded(result))
    progress('COMPLETE', update_index=32, completed_shifts=TOTAL)
    print('EXTENSION COMPLETE: 812 shifts, 16 added updates, three paired fixed evaluations.', flush=True)


def main():
    suite, phases = load_package()
    intent = json.loads((PREPARED / 'launch/launch_intent.json').read_bytes())
    require(intent['suite'] == reference(SUITE) and intent['shifts'] == TOTAL, 'launch binding differs')
    ledger = ExtensionLedger(suite, suite_path=SUITE, approval_path=APPROVAL)
    with _controller_lock(OUTPUT):
        require(not (CONTROL / 'STOP.properties').exists() and not ledger.path.exists(),
                'existing production execution requires inspection')
        try:
            execute(suite, phases, ledger)
        except BaseException as error:
            CONTROL.mkdir(parents=True, exist_ok=True)
            stop = CONTROL / 'STOP.properties'
            if not stop.exists():
                _publish(stop, ('reason=' + repr(error).replace('\n', ' ') + '\n').encode())
            progress('STOP', reason=repr(error))
            raise


if __name__ == '__main__':
    main()
