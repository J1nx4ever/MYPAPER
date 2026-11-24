"""Start one audited recovery controller without a new budget registration."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from business_graph_summary_campaign_v1.launch import _launch_lock,_exclusive_json
from business_graph_summary_campaign_v1.health import _process_identity
from .gate import preflight,reference

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/jinx/miniconda3/envs/anylogic-env/bin/python'


def start(recovery_path, *, spawn=None):
    path=Path(recovery_path).resolve()
    suite,_,_=preflight(path)
    launch=path.parent/'launch'
    control=Path(suite['control_directory'])
    if (control/'STOP.properties').exists():
        raise ValueError('preserved recovery STOP requires review; do not relaunch')
    with _launch_lock(launch):
        if any((launch/name).exists() for name in ('launch_intent.json','launch_receipt.json')):
            raise ValueError('existing recovery launch; attach exact live process, never resubmit')
        command=[PYTHON,'-B','-m','business_graph_summary_recovery_v1.runtime','run','--recovery',str(path)]
        intent=dict(schema='business-graph-summary-recovery-launch-v1',recovery=reference(path),
            original_suite=reference(suite['suite_path']),command=command,cwd=str(ROOT),
            started_at_utc=datetime.now(timezone.utc).isoformat(),new_budget_slots=0)
        _exclusive_json(launch/'launch_intent.json',intent)
        env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        with (launch/'controller.log').open('ab',buffering=0) as log:
            child=(spawn or subprocess.Popen)(command,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        identity=_process_identity(child.pid)
        receipt=dict(intent,pid=child.pid,process_start_ticks=identity.get('start_ticks'),
            controller_log=str(launch/'controller.log'))
        _exclusive_json(launch/'launch_receipt.json',receipt)
    return dict(status='RECOVERY_CONTROLLER_STARTED_WAIT_FOR_READY',recovery=reference(path),
        pid=child.pid,controller_log=receipt['controller_log'],new_budget_slots=0,
        remaining_physical_shifts=511,manual_anylogic_launches_required=1)
