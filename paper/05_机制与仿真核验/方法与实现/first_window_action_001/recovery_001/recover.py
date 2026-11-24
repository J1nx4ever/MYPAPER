"""Restart the failed Python controller while the original Java run waits."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys

RECOVERY = Path(__file__).resolve().parent
sys.path.insert(0, str(RECOVERY.parent))
from common import CORE, CORE_SHA, MATERIALS, PLAN, PREPARED, RUN, checked, read, reference, require, save, sha
from online_v1_1.operations import _publish
import budget
import runtime

SUITE = PREPARED / 'suite.json'
CONTROL = RUN / 'control'
ATTEMPT = CONTROL / 'controller_recovery_001'
OLD_PID = 848493
JAVA_PID = 848188


def verify_zero_actions(cfg, config_reference):
    receipt = read(CONTROL/'launch_receipt.json')
    require(receipt['pid'] == OLD_PID and receipt['suite'] == reference(SUITE),
            'original launch receipt differs')
    log = (CONTROL/'controller.log').read_text()
    require(log.count("NameError: name '_publish' is not defined") == 2,
            'unexpected original failure')
    for path in (CONTROL/'ready', CONTROL/'java_claims', RUN/'evaluation/episodes', MATERIALS):
        require(not path.exists() or (path.is_dir() and not any(path.iterdir())),
                'existing evidence requires reconciliation: ' + str(path))
    require(not (CONTROL/'STOP.properties').exists() and not (RUN/'suite_complete.json').exists(),
            'campaign has STOP or completion evidence')
    require(reference(RUN/'evaluation/phase_config.json')['sha256'] == config_reference['sha256'],
            'registered phase snapshot differs')
    state = read(budget.LEDGER)
    expected_slots = {str(job['run_index']): {'status':'PLANNED'} for job in cfg['jobs']}
    require(state['schema'] == 'bg128-first-window-action-budget-v1' and
            state['config'] == config_reference and state['slots'] == expected_slots,
            'only the same untouched four-slot registration can be resumed')
    require(sha(CORE)==CORE_SHA, 'protected PPO changed')
    return state


class RegisteredLedger(budget.Ledger):
    def __init__(self, cfg, config_reference):
        # runtime.run acquires the original exclusive campaign lock first.
        self.state = verify_zero_actions(cfg, config_reference)


def install_repairs():
    # Keep the frozen sources intact. Bind the missing publication helper in
    # the two modules and use the already registered ledger under the same lock.
    budget._publish = _publish
    runtime._publish = _publish
    runtime.Ledger = RegisteredLedger


def check_java_wait():
    require(not Path(f'/proc/{OLD_PID}').exists(), 'original Python PID is still present')
    cmd = ['/home/jinx/miniconda3/bin/jcmd', str(JAVA_PID), 'Thread.print']
    stack = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10).stdout
    require('fusion.Baseline1CampaignPlan.awaitConfigSha256' in stack and 'fusion.Main.onStartup' in stack,
            'original AnyLogic process no longer waits for ready')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('check','start','controller'):
        modes.add_argument('--'+mode, action='store_true')
    args = parser.parse_args()
    suite, cfg = runtime.verify(SUITE)
    state = verify_zero_actions(cfg, suite['config'])
    if args.check:
        print('PASS preserved suite/config, no ready/Java claim/RPC/material rows, four PLANNED slots')
        return
    if args.controller:
        metadata = read(ATTEMPT/'identity.json')
        require(metadata['script'] == reference(__file__) and
                metadata['original_ledger'] == reference(budget.LEDGER) and
                metadata['suite'] == reference(SUITE), 'recovery identity changed')
        install_repairs()
        runtime.run(SUITE)
        return

    check_java_wait()
    with socket.socket() as probe:
        probe.bind((cfg['execution']['host'], cfg['execution']['port']))
    ATTEMPT.mkdir()
    identity = dict(schema='first-window-same-launch-recovery-v1',
        script=reference(__file__), suite=reference(SUITE), config=suite['config'],
        original_ledger=reference(budget.LEDGER), original_launch=reference(CONTROL/'launch_receipt.json'),
        original_log=reference(CONTROL/'controller.log'), java_pid=JAVA_PID, failed_pid=OLD_PID,
        run_indices=list(map(int,state['slots'])), recording_id=cfg['recording_id'],
        recovered_existing_planned_shifts=4, added_shifts=0)
    save(ATTEMPT/'identity.json', identity)
    save(ATTEMPT/'original_ledger_snapshot.json', state)
    with (ATTEMPT/'controller.log').open('xb') as stream:
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--controller'],
            cwd=PLAN.parents[3], stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
        os.fsync(stream.fileno())
    save(ATTEMPT/'launch_receipt.json', dict(pid=process.pid, identity=reference(ATTEMPT/'identity.json')))
    print(f'Recovery controller PID {process.pid}')


if __name__=='__main__':
    main()
