"""Materialize an independent AnyLogic model for the matched S02 campaign."""

import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET


PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[4]
SOURCE_DIR = ROOT / 'anylogic/3d/BG128-REINFORCE-S02'
SOURCE = SOURCE_DIR / (SOURCE_DIR.name + '.alp')
MODEL_DIR = SOURCE_DIR.with_name('BG128-REINFORCE-S02-MATCHED')
MODEL = MODEL_DIR / (MODEL_DIR.name + '.alp')
PACK = PLAN / 'prepared_001'
REPORT = ROOT / 'reports' / PLAN.relative_to(ROOT / 'change_plans')
RUNS = 1024


def reference(path):
    return {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()}


def replace_once(data, before, after):
    if data.count(before) != 1:
        raise ValueError('Expected one uniquely identified AnyLogic field')
    return data.replace(before, after, 1)


def main():
    previous = json.loads((REPORT.parent / 'model_copy_s02.json').read_bytes())
    previous_helper = Path(previous['frozen_helper']['path'])
    if reference(SOURCE) != previous['model'] or reference(previous_helper) != previous['frozen_helper']:
        raise ValueError('Source model or its original Java helper differs from its seal')
    launch_path = PACK / 'launch_identity.json'
    launch = json.loads(launch_path.read_bytes())
    suite_path = PACK / 'suite.json'
    suite_ref = reference(suite_path)
    suite = json.loads(suite_path.read_bytes())
    helper = PACK / 'Baseline1CampaignPlan.java'
    helper_ref = reference(helper)
    master = PACK / 'master_plan.csv'
    if (suite_ref != launch['suite'] or helper_ref != suite['java_helper'] or helper_ref != launch['java_helper']
            or reference(master) != suite['master_plan'] or suite['training_shifts'] != RUNS):
        raise ValueError('Frozen launch, suite, helper or master plan differs')
    with master.open(newline='') as stream:
        run_indices = [int(row['run_index']) for row in csv.DictReader(stream)]
    if run_indices != list(range(10094001, 10095025)) or run_indices[0] != launch['first']['run_index']:
        raise ValueError('Unexpected run range in the frozen master plan')

    source = SOURCE.read_bytes()
    updates = [
        (f'<Name><![CDATA[{SOURCE_DIR.name}]]></Name>'.encode(),
         f'<Name><![CDATA[{MODEL_DIR.name}]]></Name>'.encode()),
        (previous_helper.read_bytes(), helper.read_bytes()),
    ]
    model = source
    for before, after in updates:
        model = replace_once(model, before, after)
    reconstructed = model
    for before, after in reversed(updates):
        reconstructed = replace_once(reconstructed, after, before)
    if reconstructed != source:
        raise ValueError('Model source does not reconstruct byte-for-byte')

    root = ET.fromstring(model).find('Model')
    experiment, = [node for node in root.findall('Experiments/ParamVariationExperiment')
                   if node.findtext('Name') == 'RL_Training']
    plan_class, = [node for node in root.findall('JavaClasses/JavaClass')
                   if node.findtext('Name') == 'Baseline1CampaignPlan']
    endpoint = {}
    for variable in root.findall('.//Variable'):
        name = variable.findtext('Name')
        if name in ('dispatchInferenceHost', 'dispatchInferencePort'):
            endpoint[name] = variable.findtext('Properties/InitialValue/Code')
    if (root.findtext('Name') != MODEL_DIR.name or root.findtext('JavaPackageName') != 'fusion'
            or experiment.findtext('NumberOfRuns') != str(RUNS)
            or experiment.findtext('AllowParallelEvaluations') != 'false'
            or experiment.findtext('UseFreeformParameters') != 'true'
            or experiment.findtext('ReplicationsProperties/UseReplication') != 'false'
            or plan_class.findtext('Text') != helper.read_text(encoding='utf-8')
            or endpoint != {'dispatchInferenceHost': '"127.0.0.1"', 'dispatchInferencePort': '50051'}):
        raise ValueError('AnyLogic XML experiment, Java class or RPC endpoint differs')

    resources = {entry.text for entry in root.findall('ModelResources/Resource/Path')}
    if resources != set(previous['relative_resources']):
        raise ValueError('External model resources differ from the sealed resource set')
    if not MODEL_DIR.exists():
        with TemporaryDirectory(dir=MODEL_DIR.parent, prefix='.bg128-matched-') as staging:
            destination = Path(staging) / MODEL_DIR.name
            destination.mkdir()
            for relative in sorted(resources):
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(SOURCE_DIR / relative, target)
                if reference(target)['sha256'] != previous['relative_resources'][relative]:
                    raise ValueError(f'Model resource differs from its seal: {relative}')
            (destination / MODEL.name).write_bytes(model)
            destination.rename(MODEL_DIR)
    if MODEL.read_bytes() != model or any(
        reference(MODEL_DIR / relative)['sha256'] != previous['relative_resources'][relative]
        for relative in resources
    ):
        raise ValueError('Independent model or copied resources differ from the expected bytes')

    evidence = dict(source=reference(SOURCE), model=reference(MODEL), frozen_helper=helper_ref,
                    suite=suite_ref, launch=reference(launch_path), experiment='RL_Training',
                    runs=RUNS, first_run=run_indices[0], last_run=run_indices[-1],
                    freeform=True, parallel=False, replications=False, java_package='fusion',
                    rpc_host='127.0.0.1', rpc_port=50051,
                    changes=['Model.Name', 'JavaClass[Baseline1CampaignPlan].Text'],
                    source_reconstructed_byte_equal=True,
                    relative_resources=previous['relative_resources'])
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / 'model_copy_s02_matched.json'
    rendered = json.dumps(evidence, indent=2, sort_keys=True) + '\n'
    if path.exists():
        if path.read_text() != rendered:
            raise ValueError('Existing model evidence differs')
    else:
        with path.open('x') as stream:
            stream.write(rendered)
    print(json.dumps(evidence['model']))


if __name__ == '__main__':
    main()
