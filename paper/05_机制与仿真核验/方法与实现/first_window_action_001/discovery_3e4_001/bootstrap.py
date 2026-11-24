"""Start once, or attach Java startup to the already running controller."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from discovery_common import (CONTROL, MATERIALS, OUTPUT, PLAN, PREPARED, ROOT, SUITE,
                              check_fresh_jobs, check_package, previous_budget, reference, require)
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.launch import _exclusive_json, _launch_lock

PYTHON = '/home/jinx/miniconda3/envs/anylogic-env/bin/python'


def start(suite,cfg):
    launch = PREPARED/'launch'
    command = [PYTHON,'-B',str(PLAN/'run_discovery.py')]
    suite_ref = reference(SUITE)
    with _launch_lock(launch):
        require(not (CONTROL/'STOP.properties').exists(), 'preserved STOP')
        receipt_path = launch/'launch_receipt.json'
        intent_path = launch/'launch_intent.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_bytes())
            process = _process_identity(receipt['pid'])
            require(receipt['suite']==suite_ref and type(receipt['process_start_ticks']) is int and
                    process.get('start_ticks')==receipt['process_start_ticks'] and process.get('state')!='Z' and
                    process.get('command')==command, 'existing discovery controller is not live and exact')
            return dict(status='ATTACHED_TO_EXISTING_DISCOVERY_CONTROLLER',pid=receipt['pid'])
        require(not intent_path.exists() and not OUTPUT.exists() and not MATERIALS.parent.exists(),
                'existing discovery launch/output requires inspection')
        previous = previous_budget()
        check_fresh_jobs(cfg['jobs'],previous)
        require(previous['maximum_total_shifts']==suite['occupied_before'], 'merged budget changed')
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
            probe.bind((cfg['execution']['host'],cfg['execution']['port']))
        _exclusive_json(intent_path,dict(suite=suite_ref,command=command,physical_shifts=100))
        log_path = launch/'controller.log'
        with log_path.open('xb',buffering=0) as log:
            child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
        process = _process_identity(child.pid)
        _exclusive_json(receipt_path,dict(suite=suite_ref,pid=child.pid,process_start_ticks=process.get('start_ticks'),
            log=str(log_path),command=command))
        return dict(status='DISCOVERY_CONTROLLER_STARTED',pid=child.pid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite',type=lambda p:Path(p).resolve(),required=True)
    parser.add_argument('--run-index',type=int,required=True)
    parser.add_argument('--environment-seed',type=int,required=True)
    parser.add_argument('--recording-id',required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for option in ('check','from-anylogic-run','start-controller'):
        mode.add_argument('--'+option,action='store_true')
    args = parser.parse_args()
    require(args.suite==SUITE,'unexpected discovery suite')
    suite,cfg = check_package()
    first = cfg['jobs'][0]
    require((args.run_index,args.environment_seed,args.recording_id)==
            (first['run_index'],first['experiment_seed'],first['recording_id']), 'first frozen job identity differs')
    if args.check:
        print(json.dumps(dict(status='PREFLIGHT_PASS_NO_LISTENER',config=suite['phase_config'],physical_shifts=100)))
        return
    print(json.dumps(start(suite,cfg)))


if __name__=='__main__':
    main()
