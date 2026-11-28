"""Build the independent AnyLogic copy for matched S02 paired evaluation."""

import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET


PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[5]
SOURCE_DIR = ROOT / 'anylogic/3d/BG128-REINFORCE-S02-MATCHED'
SOURCE = SOURCE_DIR / (SOURCE_DIR.name + '.alp')
MODEL_DIR = SOURCE_DIR.with_name('BG128-REINFORCE-S02-MATCHED-EVAL')
MODEL = MODEL_DIR / (MODEL_DIR.name + '.alp')
PACK = PLAN / 'prepared_001'
REPORT = ROOT / 'reports' / PLAN.relative_to(ROOT / 'change_plans')
CAMPAIGN = 'bg128-reinforce-ppo-steps-s02-u32-paired700-20261128-001'
RUNS = 700
FIRST = 10096001
EXPERIMENT = 'RL_Training'


def reference(path):
    return {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()}


def model_bytes(old_helper, new_helper):
    changes = (
        (f'<Name><![CDATA[{SOURCE_DIR.name}]]></Name>'.encode(),
         f'<Name><![CDATA[{MODEL_DIR.name}]]></Name>'.encode()),
        (b'<NumberOfRuns>1024</NumberOfRuns>', b'<NumberOfRuns>700</NumberOfRuns>'),
        (old_helper.read_bytes(), new_helper.read_bytes()),
    )
    source = SOURCE.read_bytes()
    result = source
    for before, after in changes:
        if result.count(before) != 1:
            raise ValueError('AnyLogic source does not contain one unique original field')
        result = result.replace(before, after, 1)
    reconstructed = result
    for before, after in reversed(changes):
        if reconstructed.count(after) != 1:
            raise ValueError('AnyLogic copy does not contain one unique new field')
        reconstructed = reconstructed.replace(after, before, 1)
    if reconstructed != source:
        raise ValueError('Original ALP does not reconstruct byte-for-byte')
    return result


def verify_model(data, helper):
    model = ET.fromstring(data).find('Model')
    experiment, = (item for item in model.findall('Experiments/ParamVariationExperiment')
                   if item.findtext('Name') == EXPERIMENT)
    helper_class, = (item for item in model.findall('JavaClasses/JavaClass')
                     if item.findtext('Name') == 'Baseline1CampaignPlan')
    endpoint = {item.findtext('Name'): item.findtext('Properties/InitialValue/Code')
                for item in model.findall('.//Variable')
                if item.findtext('Name') in ('dispatchInferenceHost', 'dispatchInferencePort')}
    if (model.findtext('Name') != MODEL_DIR.name or model.findtext('JavaPackageName') != 'fusion'
            or experiment.findtext('NumberOfRuns') != str(RUNS)
            or experiment.findtext('UseFreeformParameters') != 'true'
            or experiment.findtext('AllowParallelEvaluations') != 'false'
            or experiment.findtext('ReplicationsProperties/UseReplication') != 'false'
            or helper_class.findtext('Text') != helper.read_text(encoding='utf-8')
            or endpoint != {'dispatchInferenceHost': '"127.0.0.1"', 'dispatchInferencePort': '50051'}):
        raise ValueError('AnyLogic experiment, Java helper or RPC endpoint differs from the plan')
    return {item.text for item in model.findall('ModelResources/Resource/Path')}


def build():
    previous = json.loads((REPORT.parent / 'model_copy_s02_matched.json').read_bytes())
    previous_helper = PLAN.parent / 'prepared_001/Baseline1CampaignPlan.java'
    if reference(SOURCE) != previous['model'] or reference(previous_helper) != previous['frozen_helper']:
        raise ValueError('Source training ALP or its helper differs from its recorded seal')
    launch_file = PACK / 'launch_identity.json'
    launch = json.loads(launch_file.read_bytes())
    master = PACK / 'master_plan.csv'
    helper = PACK / previous_helper.name
    if (reference(master) != launch['master_plan'] or reference(helper) != launch['java_helper']
            or launch['evaluation_shifts'] != RUNS or launch['training_shifts'] != 0
            or launch['campaign_id'] != CAMPAIGN or launch['rpc_port'] != 50051):
        raise ValueError('Frozen evaluation launch and model inputs differ')
    with master.open(newline='', encoding='utf-8') as stream:
        run_indices = [int(item['run_index']) for item in csv.DictReader(stream)]
    if (run_indices != list(range(FIRST, FIRST + RUNS))
            or launch['first']['run_index'] != run_indices[0]
            or launch['last']['run_index'] != run_indices[-1]):
        raise ValueError('Unexpected physical run range in master plan')

    data = model_bytes(previous_helper, helper)
    resources = verify_model(data, helper)
    if resources != set(previous['relative_resources']) or any(
        Path(name).is_absolute() or '..' in Path(name).parts for name in resources
    ):
        raise ValueError('Relative resource inventory differs from the verified training model')
    expected_resources = previous['relative_resources']
    if not MODEL_DIR.exists():
        with TemporaryDirectory(dir=MODEL_DIR.parent, prefix='.bg128-matched-eval-') as staging:
            destination = Path(staging) / MODEL_DIR.name
            destination.mkdir()
            for relative in sorted(resources):
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(SOURCE_DIR / relative, target)
                if reference(target)['sha256'] != expected_resources[relative]:
                    raise ValueError(f'Copied resource differs from its seal: {relative}')
            (destination / MODEL.name).write_bytes(data)
            destination.rename(MODEL_DIR)
    if MODEL.read_bytes() != data or any(
        reference(MODEL_DIR / name)['sha256'] != expected_resources[name] for name in resources
    ):
        raise ValueError('Independent evaluation ALP or its resources differ')

    evidence = dict(source=reference(SOURCE), model=reference(MODEL), frozen_helper=reference(helper),
                    launch=reference(launch_file), master_plan=reference(master), experiment=EXPERIMENT,
                    campaign_id=CAMPAIGN, runs=RUNS, first_run=run_indices[0], last_run=run_indices[-1],
                    freeform=True, parallel=False, replications=False, java_package='fusion',
                    rpc_host='127.0.0.1', rpc_port=50051, source_reconstructed_byte_equal=True,
                    changes=['Model.Name', 'RL_Training.NumberOfRuns', 'Baseline1CampaignPlan.Text'],
                    relative_resources=expected_resources)
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / 'model_copy_s02_matched_eval.json'
    rendered = json.dumps(evidence, indent=2, sort_keys=True) + '\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != rendered:
            raise ValueError('Existing evaluation model evidence differs')
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(rendered)
    return evidence


if __name__ == '__main__':
    print(json.dumps(build()['model']))
