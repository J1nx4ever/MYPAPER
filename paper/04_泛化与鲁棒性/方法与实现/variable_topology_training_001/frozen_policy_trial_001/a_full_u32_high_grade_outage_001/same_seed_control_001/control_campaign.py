"""Freeze one no-outage control paired with the authenticated tree A physical outage shift."""
import argparse
from contextlib import contextmanager
import csv
from functools import lru_cache
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a_outage_campaign as treatment
from a_outage_campaign import (CHECKPOINT_SHA, PROJECT, budget_snapshot, canonical, ledger_entry,
                               require, stats, witness, write_fresh)

CAMPAIGN = 'site-layout-a-full-u32-high-grade-outage-control-20261127-001'
CAMPAIGN_OUTPUT = PROJECT/'runs'/CAMPAIGN
REPORTS = PROJECT/'reports'/HERE.relative_to(PROJECT/'change_plans')
OVERVIEW = HERE/'launch_identity.json'
EXPERIMENT = 'Topology_A_Full_NoOutageControl'
PURPOSE = 'topology_a_full_no_outage_control'
CASES = ('C_001',)
PAIRED_CASE = 'O_001'
RUN = 10056052
ORIGINAL_RUN = 10056051
SEED = 92953747099209
_receipt = witness.receipt_payload


@lru_cache(maxsize=1)
def original():
    cfg, suite, job = treatment.load_case(PAIRED_CASE)
    out = Path(job['python_output'])
    status, receipt = ledger_entry(suite, job)
    manifest_path, manifest, manifest_sha = witness.native_manifest(Path(job['event_directory']), job)
    native = dict(path=str(manifest_path), sha256=manifest_sha)
    verified = witness.reference(out/'policy_verification.json')
    audit = json.loads(Path(verified['path']).read_bytes())
    material = json.loads((out/'receipt.json').read_bytes())
    sidecar = witness.reference(manifest_path.parent/'outage_evidence.json')
    require(status == 'COMPLETE' and receipt == witness.reference(out/'receipt.json')
            and (job['run_index'], job['experiment_seed']) == (ORIGINAL_RUN, SEED)
            and material['physical_status'] == 'COMPLETE' and material['material_balance_verified'] is True
            and material['manifest'] == native and material['config_sha256'] == job['config_sha256']
            and material['actor_sampling_seed_consumed'] is False, 'outage physical identity or receipt differs')
    require(manifest['complete'] and manifest['terminated'] and manifest['truncated'] is False
            and manifest['native_policy_update'] == 32
            and manifest['config']['actual_runtime_scene'] == cfg['expected_runtime_scene']
            and manifest['outage_evidence_sha256'] == sidecar['sha256']
            and audit['status'] == 'VERIFIED' and audit['action_mode'] == cfg['action_mode'] == 'argmax'
            and manifest['native_policy_decisions'] == audit['decision_count']
            and manifest['native_checkpoint_sha256'] == audit['checkpoint_sha256'] == cfg['checkpoint_sha256'] == CHECKPOINT_SHA
            and audit['inference_manifest'] == witness.reference(out/'inference_manifest.json')
            and audit['inference_sha256'] == witness.reference(out/'inference.jsonl')['sha256'],
            'outage native scene, source files or policy evidence differs')
    pin = dict(run_index=ORIGINAL_RUN, recording_id=job['recording_id'], environment_seed=SEED,
               config=witness.reference(suite['suite_path']), receipt=receipt, manifest=native,
               policy_verification=verified, outage_evidence=sidecar)
    return cfg, pin


def identity(label):
    require(label in CASES, 'unknown control slot')
    folder = PROJECT/'runs'/f'{CAMPAIGN}-{label}'
    return dict(case=label, slot=1, group='A', phase='test', experiment=EXPERIMENT, native_case_code=4002,
                run_index=RUN, recording_id=f'{CAMPAIGN}-{RUN}', environment_seed=SEED, action_mode='argmax',
                actor_sampling_seed=None, slot_permutation_seed=0, event_directory=str(folder/'anylogic_events'),
                python_output_directory=str(folder/'python'))


def make_config(label, model):
    ref, pin = original()
    source = ref['expected_runtime_scene']
    scene = dict(source, model_sha256=model['sha256'],
                 exogenous_config=dict(source['exogenous_config'], load_site_outage=False))
    require(ref['outage'] == source['exogenous_config']['load_site_outage'] == treatment.OUTAGE
            and ref['environment_seed'] == SEED and ref['slot_permutation_seed'] == 0
            and ref['actor_sampling_seed'] is None, 'unexpected outage configuration')
    additions = [witness.reference(HERE/k) for k in ('control_service.py', 'control_campaign.py')]
    return dict(ref, **dict(identity(label), purpose=PURPOSE, model_path=model['path'], model_sha256=model['sha256'],
                expected_runtime_scene=scene, expected_runtime_scene_sha256=sha256(canonical(scene)).hexdigest(),
                outage=None, paired_outage=pin, inference_sources=ref['inference_sources']+additions))


suite_job = witness.bind(treatment.suite_job, HERE=HERE)


@lru_cache(maxsize=1)
def index_doc():
    data = OVERVIEW.read_bytes()
    index = json.loads(data)
    require(sha256(data).hexdigest() == OVERVIEW.with_suffix('.sha256').read_text().strip()
            and index['campaign_id'] == CAMPAIGN and index['outage'] is None
            and index['paired_outage'] == original()[1] and tuple(r['case'] for r in index['cases']) == CASES,
            'control identity index or paired source changed')
    return index


case_labels = witness.bind(treatment.case_labels, index_doc=index_doc)
frozen_index = witness.bind(treatment.frozen_index, index_doc=index_doc)
load_case = witness.bind(treatment.load_case, HERE=HERE, frozen_index=frozen_index,index_doc=index_doc,
                         identity=identity,make_config=make_config,suite_job=suite_job)
read_case = witness.bind(treatment.read_case, load_case=load_case)


def historical_check(identities, history):
    """Admit the sealed O_001 environment seed only for its identified C_001 control."""
    ident, = identities
    require(all(ident[k] == v for k, v in identity('C_001').items()), 'unrecognized control identity')
    require(str(RUN) not in history['jobs'] and f'{CAMPAIGN}-C_001' not in history['campaigns'],
            'physical control registered already')
    _, pin = original()
    original_job = history['jobs'][str(ORIGINAL_RUN)]
    require(original_job['status'] == 'COMPLETE' and original_job['receipt'] == pin['receipt']
            and original_job['campaign_id'] == f'{treatment.CAMPAIGN}-{PAIRED_CASE}', 'paired ledger owner differs')
    mask = witness.ledger.JAVA_MASK
    states = set(witness.ledger.java_random_states(SEED).values())
    require(0 < SEED <= mask and len(states) == 5, 'invalid environment stream derivation')
    prior = witness.bind(witness.ledger.previous.paired10._prior_rows, SUITE=OVERVIEW)(history)
    prior.extend(r['identity'] for r in history['jobs'].values() if isinstance(r.get('identity'),dict))
    for path in witness.GRAPH.rglob('launch_identity.json'):
        if path == OVERVIEW:
            continue
        rows = json.loads(path.read_bytes()).get('cases', [])
        prior.extend(rows.values() if isinstance(rows,dict) else rows)
    other = set()
    for row in prior:
        prior_run, recording = row.get('run_index'), row.get('recording_id')
        require(prior_run != RUN and recording != ident['recording_id']
                and row.get('source_recording_id') != ident['recording_id'], 'physical control collides with history')
        for key in witness.SEED_FIELDS:
            value = row.get(key)
            if type(value) is not int or value <= 0:
                continue
            if value & mask == SEED:
                require(value == SEED and key in ('environment_seed','experiment_seed') and prior_run == ORIGINAL_RUN
                        and recording == pin['recording_id'], 'unauthorized control seed reuse')
            else:
                other.add(value & mask)
    require(all(states.isdisjoint(witness.ledger.java_random_states(s).values()) for s in other),
            'effective control Java streams collide with another registered or reserved seed')


register_started = witness.bind(stats.register_started,CASES=CASES,identity=identity,load_case=load_case,
                                historical_check=historical_check)




@contextmanager
def settlement_session():
    register_started()
    def settle(suite,job,physical,*,collect=True):
        require(not collect or witness.ledger.ledger_path(suite).is_file(), 'cannot settle an unstarted native control')
        with witness.ledger._launch_lock(Path(suite['output'])/'collector'):
            return witness.settle(suite,job,physical,collect=collect)
    yield settle


verify_budget = witness.bind(treatment.verify_budget,CAMPAIGN=CAMPAIGN,OVERVIEW=OVERVIEW,CASES=CASES,
                             load_case=load_case,REPORTS=REPORTS)


def prepare(path):
    label, = CASES
    model = stats.model_pin(path.resolve(strict=True))
    model_bytes = Path(model['path']).read_bytes()
    require(path.suffix=='.alp' and sha256(model_bytes).hexdigest()==model['sha256'],'invalid control model')
    experiment, = [e for e in ET.fromstring(model_bytes).findall('./Model/Experiments/ParamVariationExperiment')
                   if e.findtext('Name')==EXPERIMENT]
    require(experiment.findtext('NumberOfRuns')=='1' and experiment.findtext('AllowParallelEvaluations')=='false'
            and experiment.findtext('FreeformParamValue/Expression/Code')=='4002 + index', 'wrong control experiment')
    cfg = make_config(label,model)
    raw = canonical(cfg)
    sha = sha256(raw).hexdigest()
    suite,job = suite_job(cfg,sha)
    row = dict(identity(label), config_path=str(suite['suite_path']),config_sha256=sha,
               checkpoint_sha256=CHECKPOINT_SHA,task_sha256=cfg['task_sha256'],paired_outage_run=ORIGINAL_RUN)
    with witness.ledger._launch_lock(witness.ledger.previous.LOCKS/'budget'):
        require(not any(p.exists() for p in (OVERVIEW,OVERVIEW.with_suffix('.sha256'),HERE/'configs',
                    HERE/'run_index_seed_manifest.csv',CAMPAIGN_OUTPUT,Path(job['event_directory']).parent)),
                'existing control identity or physical evidence')
        history = budget_snapshot()
        historical_check([row],history)
        archive = REPORTS/Path(model['path']).name
        require(not archive.exists() or archive.read_bytes()==model_bytes, 'archived model differs')
        REPORTS.mkdir(parents=True,exist_ok=True)
        (HERE/'configs').mkdir()
        CAMPAIGN_OUTPUT.mkdir()
        Path(job['event_directory']).parent.mkdir()
        write_fresh(archive,model_bytes)
        write_fresh(suite['suite_path'],raw)
        write_fresh(suite['suite_path'].with_suffix('.sha256'),(sha+'\n').encode())
        stream=io.StringIO(newline='')
        writer=csv.DictWriter(stream,fieldnames=list(row),lineterminator='\n')
        writer.writeheader()
        writer.writerow(row)
        write_fresh(HERE/'run_index_seed_manifest.csv',stream.getvalue().encode())
        doc=dict(campaign_id=CAMPAIGN,cases=[row],outage=None,paired_outage=cfg['paired_outage'],
                 model=model, archived_model=witness.reference(archive), registered_by_preparation=0,
                 registered_at_preparation=history['maximum_total_shifts'])
        data=canonical(doc)
        write_fresh(OVERVIEW,data)
        write_fresh(OVERVIEW.with_suffix('.sha256'),(sha256(data).hexdigest()+'\n').encode())
    return doc


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    doc=prepare(parser.parse_args().model)
    print(json.dumps(dict(index=witness.reference(OVERVIEW),case=doc['cases'][0],before=doc['registered_at_preparation'])))
