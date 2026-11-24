"""Compact, read-only controller status; ordinary status never probes a process.

This module reads frozen plans, the shared ledger, and a single overwritten
progress record.  It opens no socket, constructs no learner, and never changes
STOP/UNKNOWN/claim records.  Optional explicit PID inspection uses /proc only.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path, default=None):
    path = Path(path)
    return json.loads(path.read_bytes()) if path.is_file() else default


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _age(timestamp, now):
    if not isinstance(timestamp, str):
        return None
    try:
        value = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        if value.tzinfo is None:
            return None
        return max(0., (now - value).total_seconds())
    except ValueError:
        return None


def _process_identity(pid):
    """Only called when a PID is explicitly supplied or a child was just born."""
    if type(pid) is not int or pid <= 0:
        raise ValueError('positive explicit controller PID required')
    folder = Path('/proc') / str(pid)
    try:
        stat = (folder / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        command = (folder / 'cmdline').read_bytes().split(b'\0')
        command = [word.decode('utf-8', errors='replace') for word in command if word]
        return dict(pid=pid, state=fields[0], start_ticks=int(fields[19]),
                    cpu_seconds=(int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK'),
                    command=command)
    except (FileNotFoundError, ProcessLookupError):
        return dict(pid=pid, exists=False)
    except PermissionError:
        return dict(pid=pid, inspection_denied=True)


def _batch_eta(progress):
    elapsed = progress.get('elapsed_seconds')
    done, total = progress.get('completed_steps'), progress.get('total_steps')
    if not (isinstance(done, (int, float)) and isinstance(total, (int, float)) and 0 < done <= total):
        done, total = progress.get('completed_epochs'), progress.get('epochs')
    if (isinstance(elapsed, (int, float)) and elapsed >= 0 and
            isinstance(done, (int, float)) and isinstance(total, (int, float)) and 0 < done <= total):
        return dict(seconds=max(0., elapsed * (total / done - 1)), scope='current_update_only',
                    basis='completed_work_linear_extrapolation', guaranteed=False)
    return None


def inspect(suite_path, *, launch_directory=None, pid=None, now=None):
    from .budget import GLOBAL_PATH, connection, _payload
    from .plan_io import load_suite
    suite,_=load_suite(suite_path)
    output=Path(suite['python_output']);control=Path(suite['control_directory'])
    launch_directory=Path(launch_directory or Path(suite_path).parent/'launch')
    counts=Counter();commits=0
    if GLOBAL_PATH.exists():
        with connection(GLOBAL_PATH,readonly=True) as con:
            for state,count in con.execute('SELECT status,count(*) FROM slots WHERE campaign=? GROUP BY status',(suite['campaign_id'],)):
                counts[state]=count
            commits=con.execute('SELECT count(*) FROM commits WHERE campaign=?',(suite['campaign_id'],)).fetchone()[0]
    status='PREPARED_NOT_STARTED'
    if (output/'suite_complete.json').exists(): status='COMPLETE'
    elif (control/'STOP.properties').exists(): status='STOP_REQUIRES_REVIEW'
    elif counts['UNKNOWN']: status='UNKNOWN_REQUIRES_RECONCILIATION'
    elif counts: status='REGISTERED'
    result=dict(status=status,suite=_reference(suite_path),physical_shifts=dict(
        planned=counts['PLANNED'],claimed=counts['CLAIMED'],complete=counts['COMPLETE'],unknown=counts['UNKNOWN'],
        proposed=suite['new_shift_cap']),committed_updates=commits,
        progress=_read(output/'training_progress.json'),execution_authorized=False)
    if pid is not None:
        receipt=_read(launch_directory/'launch_receipt.json',{})
        if receipt.get('pid')!=pid or receipt.get('suite')!=result['suite']: raise ValueError('PID/suite binding differs')
        process=_process_identity(pid)
        if process.get('start_ticks')!=receipt.get('process_start_ticks'): raise ValueError('PID start identity differs')
        result['process']=process
    return result
