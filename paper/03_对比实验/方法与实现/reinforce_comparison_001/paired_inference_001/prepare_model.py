"""Create and verify the independent AnyLogic inference model from the S02 training copy."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import xml.etree.ElementTree as ET

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[4]
SOURCE_DIR = ROOT / 'anylogic/3d/BG128-REINFORCE-S02'
SOURCE = SOURCE_DIR / (SOURCE_DIR.name + '.alp')
MODEL_NAME = 'BG128-REINFORCE-S02-U32-EVAL'
MODEL_DIR = SOURCE_DIR.with_name(MODEL_NAME)
MODEL = MODEL_DIR / (MODEL_NAME + '.alp')
PACK = PLAN / 'prepared_001'
REPORT = ROOT / 'reports' / PLAN.relative_to(ROOT / 'change_plans')


def reference(path):
    return {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()}


def main():
    prior_evidence = json.loads((REPORT.parent / 'model_copy_s02.json').read_bytes())
    if reference(SOURCE) != prior_evidence['model']:
        raise ValueError('source S02 training model differs from its sealed copy')
    suite = json.loads((PACK / 'suite.json').read_bytes())
    previous_helper = PLAN.parent / 'prepared_001/Baseline1CampaignPlan.java'
    helper = PACK / previous_helper.name
    if reference(helper) != suite['java_helper'] or reference(previous_helper) != prior_evidence['frozen_helper']:
        raise ValueError('frozen helper digest differs')
    source_bytes = SOURCE.read_bytes()
    model_bytes = source_bytes
    modifications = [
        (f'<Name><![CDATA[{SOURCE_DIR.name}]]></Name>'.encode(), f'<Name><![CDATA[{MODEL_NAME}]]></Name>'.encode()),
        (b'<NumberOfRuns>1024</NumberOfRuns>', b'<NumberOfRuns>700</NumberOfRuns>'),
        (previous_helper.read_bytes(), helper.read_bytes()),
    ]
    for before, after in modifications:
        if model_bytes.count(before) != 1:
            raise ValueError('S02 model field no longer has a unique original value')
        model_bytes = model_bytes.replace(before, after, 1)
    root = ET.fromstring(model_bytes).find('Model')
    experiment, = (node for node in root.findall('Experiments/ParamVariationExperiment')
                   if node.findtext('Name') == 'RL_Training')
    plan_class, = (node for node in root.findall('JavaClasses/JavaClass')
                   if node.findtext('Name') == 'Baseline1CampaignPlan')
    if (root.findtext('Name') != MODEL_NAME or root.findtext('JavaPackageName') != 'fusion'
            or experiment.findtext('NumberOfRuns') != '700' or experiment.findtext('UseFreeformParameters') != 'true'
            or experiment.findtext('AllowParallelEvaluations') != 'false'
            or experiment.findtext('ReplicationsProperties/UseReplication') != 'false'
            or plan_class.findtext('Text') != helper.read_text(encoding='utf-8')):
        raise ValueError('copied model experiment, package or Java class differs')
    reconstructed = model_bytes
    for before, after in reversed(modifications):
        if reconstructed.count(after) != 1:
            raise ValueError('cannot unambiguously reverse the model field change')
        reconstructed = reconstructed.replace(after, before, 1)
    if reconstructed != source_bytes:
        raise ValueError('model has changes beyond name, run count and helper')
    resources = {path.text for path in root.findall('ModelResources/Resource/Path')}
    if resources != set(prior_evidence['relative_resources']):
        raise ValueError('S02 relative resource inventory changed')
    if MODEL_DIR.exists():
        raise FileExistsError(MODEL_DIR)
    with TemporaryDirectory(dir=MODEL_DIR.parent, prefix='.bg128-eval-') as staging:
        directory = Path(staging) / MODEL_NAME
        directory.mkdir()
        for relative in sorted(resources):
            destination = directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE_DIR / relative, destination)
            if reference(destination)['sha256'] != prior_evidence['relative_resources'][relative]:
                raise ValueError('copied resource differs: ' + relative)
        (directory / MODEL.name).write_bytes(model_bytes)
        directory.rename(MODEL_DIR)
    evidence = dict(source=reference(SOURCE), model=reference(MODEL), frozen_helper=reference(helper),
                    suite=reference(PACK / 'suite.json'), experiment='RL_Training', runs=700,
                    freeform=True, parallel=False, replications=False, java_package='fusion',
                    changes=prior_evidence['changes'], source_reconstructed_byte_equal=True,
                    relative_resources=prior_evidence['relative_resources'])
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / 'model_copy_s02_u32_eval.json').open('x', encoding='utf-8') as evidence_file:
        json.dump(evidence, evidence_file, indent=2)
        evidence_file.write('\n')
    print(json.dumps(evidence['model']))


if __name__ == '__main__':
    main()
