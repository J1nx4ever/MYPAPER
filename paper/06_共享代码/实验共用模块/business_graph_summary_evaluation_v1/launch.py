"""Launch exactly one frozen100-shift read-only Actor/Critic controller."""
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
from business_graph_summary_campaign_v1.launch import _launch_lock,_exclusive_json
from business_graph_summary_campaign_v1.health import _process_identity
from .gate import ROOT,preflight,reference,require


def start(suite_path,*,spawn=None):
    path=Path(suite_path).resolve();suite,_,approval=preflight(path)
    require(approval is not None,'user100-shift authorization required')
    require(not (Path(suite['control_directory'])/'STOP.properties').exists(),'preserved evaluation STOP; inspect before retry')
    launch=path.parent/'launch'
    with _launch_lock(launch):
        require(not any((launch/n).exists() for n in ('launch_intent.json','launch_receipt.json')),
            'evaluation already launched; never start a second controller')
        command=['/home/jinx/miniconda3/envs/anylogic-env/bin/python','-B','-m',
            'business_graph_summary_evaluation_v1.runtime','run','--suite',str(path),
            '--approval',str(path.parent/'approval.json')]
        intent=dict(schema='bg128-independent-evaluation-launch-v1',suite=reference(path),
            approval=reference(path.parent/'approval.json'),command=command,cwd=str(ROOT),
            physical_shifts=100,optimizer_steps=0,started_at_utc=datetime.now(timezone.utc).isoformat())
        _exclusive_json(launch/'launch_intent.json',intent)
        with (launch/'controller.log').open('ab',buffering=0) as log:
            child=(spawn or subprocess.Popen)(command,cwd=str(ROOT),env=dict(os.environ,PYTHONUNBUFFERED='1',
                PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True)
        observed=_process_identity(child.pid)
        receipt=dict(intent,pid=child.pid,process_start_ticks=observed.get('start_ticks'),
            controller_log=str(launch/'controller.log'))
        _exclusive_json(launch/'launch_receipt.json',receipt)
    return dict(status='EVALUATION_CONTROLLER_STARTED_WAIT_FOR_READY',pid=child.pid,
        physical_shifts=100,optimizer_steps=0,controller_log=receipt['controller_log'])
