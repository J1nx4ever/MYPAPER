"""Create and verify the independent AnyLogic suffix model."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET


PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[6]
SOURCE_DIR = ROOT / 'anylogic/3d/BG128-REINFORCE-S02-MATCHED-EVAL'
SOURCE = SOURCE_DIR / (SOURCE_DIR.name + '.alp')
MODEL_DIR = SOURCE_DIR.with_name('BG128-REINFORCE-S02-MATCHED-EVAL-RECOVERY')
MODEL = MODEL_DIR / (MODEL_DIR.name + '.alp')
REPORT = ROOT / 'reports' / PLAN.relative_to(ROOT / 'change_plans')
PACK = PLAN / 'prepared_001'
CAMPAIGN = 'bg128-reinforce-ppo-steps-s02-u32-paired700-recovery-20261128-001'
EXPERIMENT = 'RL_Training'
RUNS = 658
FIRST = 10097001


def reference(path):
    return {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()}


def edit_model(source, before_helper, after_helper):
    changes = [
        (f'<Name><![CDATA[{SOURCE_DIR.name}]]></Name>'.encode(),
         f'<Name><![CDATA[{MODEL_DIR.name}]]></Name>'.encode()),
        (b'<NumberOfRuns>700</NumberOfRuns>', b'<NumberOfRuns>658</NumberOfRuns>'),
        (before_helper, after_helper),
    ]
    data = source
    for before, after in changes:
        if data.count(before) != 1:
            raise ValueError('Expected a unique original AnyLogic field')
        data = data.replace(before, after, 1)
    reconstructed = data
    for before, after in reversed(changes):
        if reconstructed.count(after) != 1:
            raise ValueError('Expected a unique recovery AnyLogic field')
        reconstructed = reconstructed.replace(after, before, 1)
    if reconstructed != source:
        raise ValueError('Original AnyLogic bytes failed round-trip check')
    return data


def verify_model(data, helper):
    model = ET.fromstring(data).find('Model')
    experiment, = (item for item in model.findall('Experiments/ParamVariationExperiment')
                   if item.findtext('Name') == EXPERIMENT)
    java, = (item for item in model.findall('JavaClasses/JavaClass')
             if item.findtext('Name') == 'Baseline1CampaignPlan')
    endpoints = {item.findtext('Name'): item.findtext('Properties/InitialValue/Code')
                 for item in model.findall('.//Variable')
                 if item.findtext('Name') in ('dispatchInferenceHost', 'dispatchInferencePort')}
    if (model.findtext('Name') != MODEL_DIR.name or model.findtext('JavaPackageName') != 'fusion'
            or experiment.findtext('NumberOfRuns') != str(RUNS)
            or experiment.findtext('UseFreeformParameters') != 'true'
            or experiment.findtext('AllowParallelEvaluations') != 'false'
            or experiment.findtext('ReplicationsProperties/UseReplication') != 'false'
            or java.findtext('Text') != helper.decode('utf-8')
            or endpoints != {'dispatchInferenceHost': '"127.0.0.1"', 'dispatchInferencePort': '50051'}):
        raise ValueError('Recovery experiment, Java helper or RPC endpoint differs')
    return {item.text for item in model.findall('ModelResources/Resource/Path')}


def build():
    prior = json.loads((REPORT.parent / 'model_copy_s02_matched_eval.json').read_bytes())
    previous_helper = PLAN.parent / 'prepared_001/Baseline1CampaignPlan.java'
    if reference(SOURCE) != prior['model'] or reference(previous_helper) != prior['frozen_helper']:
        raise ValueError('Original evaluation model or helper changed after launch')
    launch_file = PACK / 'launch_identity.json'
    launch = json.loads(launch_file.read_bytes())
    master = PACK / 'master_plan.csv'
    helper = PACK / previous_helper.name
    if (reference(master) != launch['master_plan'] or reference(helper) != launch['java_helper']
            or launch['campaign_id'] != CAMPAIGN or launch['evaluation_shifts'] != RUNS
            or launch['training_shifts'] != 0 or launch['rpc_port'] != 50051):
        raise ValueError('Frozen recovery plan and model input differ')
    with master.open(newline='', encoding='utf-8') as stream:
        indices = [int(job['run_index']) for job in csv.DictReader(stream)]
    if (indices != list(range(FIRST, FIRST + RUNS))
            or launch['first']['run_index'] != indices[0] or launch['last']['run_index'] != indices[-1]):
        raise ValueError('Recovery physical run range differs')

    helper_bytes = helper.read_bytes()
    data = edit_model(SOURCE.read_bytes(), previous_helper.read_bytes(), helper_bytes)
    resources = verify_model(data, helper_bytes)
    expected = prior['relative_resources']
    if resources != set(expected) or any(Path(name).is_absolute() or '..' in Path(name).parts for name in resources):
        raise ValueError('Recovery model resource inventory differs')
    if not MODEL_DIR.exists():
        with TemporaryDirectory(prefix='.bg128-matched-eval-recovery-', dir=MODEL_DIR.parent) as staging:
            destination = Path(staging) / MODEL_DIR.name
            destination.mkdir()
            for name in sorted(resources):
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(SOURCE_DIR / name, target)
                if reference(target)['sha256'] != expected[name]:
                    raise ValueError(f'Copied resource differs: {name}')
            (destination / MODEL.name).write_bytes(data)
            destination.rename(MODEL_DIR)
    if MODEL.read_bytes() != data or any(reference(MODEL_DIR / name)['sha256'] != expected[name] for name in resources):
        raise ValueError('Existing recovery copy or its resources differ')
    evidence = dict(source=reference(SOURCE), model=reference(MODEL), frozen_helper=reference(helper),
                    launch=reference(launch_file), master_plan=reference(master), experiment=EXPERIMENT,
                    campaign_id=CAMPAIGN, runs=RUNS, first_run=indices[0], last_run=indices[-1],
                    freeform=True, parallel=False, replications=False, java_package='fusion',
                    rpc_host='127.0.0.1', rpc_port=50051, source_reconstructed_byte_equal=True,
                    changes=['Model.Name', 'RL_Training.NumberOfRuns', 'Baseline1CampaignPlan.Text'],
                    relative_resources=expected)
    REPORT.mkdir(parents=True, exist_ok=True)
    audit = REPORT / 'model_copy_recovery.json'
    rendered = json.dumps(evidence, indent=2, sort_keys=True) + '\n'
    if audit.exists():
        if audit.read_text(encoding='utf-8') != rendered:
            raise ValueError('Existing recovery model evidence differs')
    else:
        with audit.open('x', encoding='utf-8') as stream:
            stream.write(rendered)
    return evidence


if __name__ == '__main__':
    print(json.dumps(build()['model']))
