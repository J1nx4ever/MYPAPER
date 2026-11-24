"""One fixed-checkpoint evaluation, then a predeclared first-window inventory."""
from datetime import datetime, timezone
from pathlib import Path
import torch

from discovery_common import (CAMPAIGN, CONTROL, CORE, CORE_SHA, COUNT, OUTPUT, REPORT, SUITE,
                              check_package, checked, read, reference, require, sha, write)
from discovery_budget import DiscoveryLedger
from business_graph_summary_evaluation_v1.runtime import _status, _trial
from grade_only_ppo_v2.runtime import _controller_lock
from online_v1_1.operations import _publish


def discovery_catalog(suite, finished):
    evaluation_ref = finished['reports']['evaluation']
    evaluation = read(checked(evaluation_ref))
    records = evaluation['per_episode']
    require(len(records)==COUNT and
            [r['identity']['run_index'] for r in records]==[j['run_index'] for j in suite['jobs']],
            'discovery reporting inventory differs')
    failures, all_shifts = [], []
    for record in records:
        physical = record['physical']
        windows = physical['windows']
        require([w['window'] for w in windows]==[0,1,2,3], 'four windows required')
        row = dict(run_index=record['identity']['run_index'],
            environment_seed=record['identity']['environment_seed'],
            actor_sampling_seed=record['identity']['actor_sampling_seed'],
            windows=windows, complete_task=physical['complete_task'],
            ore_tons=physical['ore_tons'], waste_tons=physical['waste_tons'],
            total_wait_seconds=physical['total_wait_seconds'], combined_cost=physical['combined_cost'])
        all_shifts.append(row)
        if not windows[0]['compliant']:
            failures.append(dict(**row, kind='empty' if windows[0]['empty'] else
                                        ('low' if windows[0]['grade']<.26 else 'high')))
    summary = dict(status='DISCOVERY_COMPLETE', campaign_id=CAMPAIGN, physical_shifts=COUNT,
        checkpoint=suite['parent_artifacts']['checkpoint'], sampling=suite['sampling'],
        optimizer_updates=0, training_updates=0, first_window_failures=len(failures),
        first_window_compliant=COUNT-len(failures), sealed_evaluation=evaluation_ref,
        frozen_protocol=suite['protocol'], reports=finished['reports'],
        first_window_failed_shifts=failures, all_shifts=all_shifts)
    write(REPORT/'discovery_result.json',summary)
    print(f'DISCOVERY_COMPLETE first_window_failures={len(failures)}/{COUNT}',flush=True)


def main():
    suite, cfg = check_package()
    with _controller_lock(OUTPUT):
        CONTROL.mkdir(exist_ok=True)
        require(not (CONTROL/'STOP.properties').exists(), 'preserved STOP')
        ledger = None
        torch.set_num_threads(suite['cpu_threads'])
        try:
            ledger = DiscoveryLedger(suite)
            finished = _trial(SUITE,suite,cfg,ledger)
            require(sha(CORE)==CORE_SHA, 'protected PPO changed')
            discovery_catalog(suite,finished)
        except BaseException as error:
            if not (CONTROL/'STOP.properties').exists():
                _publish(CONTROL/'STOP.properties', ('reason='+repr(error).replace('\n',' ')+'\n').encode())
            done = 0 if ledger is None else sum(row['status']=='COMPLETE' for row in ledger.snapshot()['jobs'].values())
            _status(suite,'STOPPED',completed_shifts=done,error=repr(error))
            raise


if __name__=='__main__':
    main()
