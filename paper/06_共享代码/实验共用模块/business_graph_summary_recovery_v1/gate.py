"""Audited continuation of the original 512 slots; no replacement experiment."""
import copy
import json
from pathlib import Path

from business_graph_summary_campaign_v1 import gate as original
from business_graph_summary_campaign_v1.plan_io import (
    ROOT,encoded,sha,reference,load_reference,load_suite)
from business_graph_summary_campaign_v1.budget import (
    GLOBAL_PATH,connection,_payload,_state,digest)

require=original.require
PHASE_CONTRACT=original.PHASE_CONTRACT
CAMPAIGN_CONTRACT=original.CAMPAIGN_CONTRACT
TRAINING_REWARD=original.TRAINING_REWARD
DIAGNOSTIC_SCOPE=original.DIAGNOSTIC_SCOPE
PROTECTED_SHA=original.PROTECTED_SHA
PLAN=ROOT/'change_plans/token_attention_001/business_graph_summary_009/formal_training_001/recovery_001'
SCHEMA='business-graph-summary-recovery-v1'


def source_pins():
    paths=[]
    for package in ('business_graph_summary_storage_v1','business_graph_summary_recovery_v1'):
        paths += [p for p in (ROOT/package).glob('*.py') if not p.name.startswith('test_')]
    paths += [PLAN/'bootstrap.py',PLAN/'prepare_recovery.py']
    require(all(p.is_file() for p in paths),'recovery source inventory is incomplete')
    return {str(p.resolve()):sha(p) for p in sorted(paths)}


def check_sources(pins,recovery_ref=None):
    original.check_sources(pins)
    if recovery_ref is not None:
        descriptor=load_reference(recovery_ref)
        require(descriptor['new_source_pins']==source_pins(),'corrective implementation changed')


def check_phase(cfg):return original.check_phase(cfg)


def check_jobs(suite):return original.check_jobs(suite)


def _reconciled(descriptor,original_suite):
    proof=load_reference(descriptor['reconciliation'])
    require(proof.get('schema')=='bg128-budget-reconciliation-applied-v1' and proof.get('status')=='APPLIED',
        'explicit applied evidence reconciliation required')
    require(proof['campaign_id']==original_suite['campaign_id'] and proof['run_index']==9900001 and
        proof['old_suite']==descriptor['old_suite'] and proof['original_status']=='UNKNOWN' and
        proof['new_status']=='COMPLETE' and proof['new_shift_cap']==0,'reconciliation scope differs')
    # Immutable before-images and the full offline admission chain are retained.
    for key in ('plan','ledger_before','old_row','admission_proof','derived_manifest','stop'):
        require(reference(proof[key]['path'])==proof[key],'reconciliation evidence changed: '+key)
    first=original_suite['jobs'][0]
    with connection(GLOBAL_PATH,readonly=True) as con:
        row=con.execute('SELECT status,payload,checksum FROM slots WHERE campaign=? AND run=?',
            (original_suite['campaign_id'],first['run_index'])).fetchone()
        require(row is not None,'original first slot absent')
        record=_payload((row[1],row[2]))
        require(row[0]==record['status']=='COMPLETE' and record['identity_sha256']==digest(first),
            'first slot is not the evidenced canonical complete run')
        require(record['dataset_ref']==proof['derived_manifest'] and
            record.get('reconciled_from')=='UNKNOWN','reconciled data binding differs')
    return proof


def preflight(path, *, require_reconciled=True, **unused):
    path=Path(path).resolve()
    descriptor=json.loads(path.read_bytes())
    require(descriptor['schema']==SCHEMA and descriptor['recovery_path']==str(path),
        'recovery descriptor identity differs')
    require(descriptor['new_source_pins']==source_pins(),'recovery source SHA inventory changed')
    old_ref=descriptor['old_suite']
    require(reference(old_ref['path'])==old_ref,'original suite changed')
    approval_path=Path(old_ref['path']).parent/'approval.json'
    suite,phases,approval=original.preflight(old_ref['path'],approval_path=approval_path)
    require(approval is not None,'original scoped authorization required')
    require(descriptor['campaign_id']==suite['campaign_id'] and descriptor['resume_offset']==1 and
        descriptor['job_count']==descriptor['remaining_physical_shifts']==511 and
        descriptor['frozen_job_count']==512 and descriptor['new_physical_shift_cap']==0,
        'continuation may only use the original remaining 511 slots')
    require(descriptor['remaining_jobs']==suite['jobs'][1:],'remaining original identities changed or reordered')
    require(descriptor['master_plan']==suite['master_plan'],'original 512-row master must be retained')
    for key in ('java_helper','master_plan','bridge','initial_checkpoint','initial_actor',
            'initial_checkpoint_record','original_stop','original_first_claim',
            'original_launch','original_approval'):
        ref=descriptor[key]
        require(reference(ref['path'])==ref,'recovery artifact changed: '+key)
    for ref in descriptor['initial_artifact_references']:
        require(reference(ref['path'])==ref,'original initial artifact seal changed')
    require(descriptor['original_approval']==reference(approval_path) and
        descriptor['protected_ppo_sha256']==PROTECTED_SHA,'original authorization or PPO core differs')
    require(Path(descriptor['initial_checkpoint']['path'])==Path(suite['arms'][0]['python_output'])/'initial.pt' and
        Path(descriptor['initial_actor']['path'])==Path(phases[0]['python_output'])/'actor.pt',
        'recovery cannot load another policy origin')
    new_control=Path(descriptor['control_directory']).resolve()
    require(new_control==Path(suite['python_output'])/'recovery_001/control' and
        new_control!=Path(suite['control_directory']),'independent recovery control directory required')
    if require_reconciled:
        reconciliation=_reconciled(descriptor,suite)
        require(reconciliation['stop']==descriptor['original_stop'],
            'original failure control evidence differs')
    runtime_suite=copy.deepcopy(suite)
    runtime_suite.update(control_directory=str(new_control),_recovery_ref=reference(path),
        _original_approval_path=str(approval_path.resolve()))
    # Existing registration is validated, never replaced or extended.
    if require_reconciled:
        with connection(GLOBAL_PATH,readonly=True) as con:
            registered=con.execute('SELECT payload,checksum FROM campaigns WHERE id=?',
                (suite['campaign_id'],)).fetchone()
            require(registered is not None,'original registration is required; recovery cannot register')
            entry=_payload(registered)
            state=_state(con,suite,entry)
        require(all(r['status'] not in ('UNKNOWN','CLAIMED') for r in state['jobs'].values()),
            'another unresolved physical claim requires explicit review')
        require(len(state['jobs'])==512,'approved physical inventory differs')
    return runtime_suite,phases,approval
