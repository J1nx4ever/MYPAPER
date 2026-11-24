"""Regenerate the original slot mapping with an isolated recovery startup."""

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from baseline1_v1.java_plan import _replace_once, _require
from business_graph_summary_campaign_v1.plan_io import sha


PLAN = Path(__file__).resolve().parent


def write_java(original_helper, destination, original_control, recovery_control,
               original_bridge, recovery_bridge, original_suite, recovery_json):
    """Pin the new bridge while preserving the original scenario and master tables."""
    original_helper, destination = (Path(path).resolve() for path in (original_helper, destination))
    original_control, recovery_control = (Path(path).resolve() for path in (original_control, recovery_control))
    original_bridge, recovery_bridge = (Path(path).resolve() for path in (original_bridge, recovery_bridge))
    original_suite, recovery_json = (Path(path).resolve() for path in (original_suite, recovery_json))
    _require(destination.name == 'Baseline1CampaignPlan.java' and not destination.exists(),
             'new Baseline1CampaignPlan.java destination required')
    _require(original_control != recovery_control and original_bridge != recovery_bridge
             and original_suite != recovery_json, 'recovery startup must use a separate identity')
    _require(recovery_control.name == 'control' and recovery_control.parent.name == 'recovery_001'
             and recovery_control.parent.parent == original_control.parent,
             'recovery control must be isolated beside the original control')
    source = original_helper.read_text(encoding='utf-8')
    pairs = (
        (f'CONTROL_DIRECTORY = {json.dumps(str(original_control))};',
         f'CONTROL_DIRECTORY = {json.dumps(str(recovery_control))};'),
        (f'absolute({json.dumps(str(original_bridge))}, "bridge path")',
         f'absolute({json.dumps(str(recovery_bridge))}, "bridge path")'),
        (f'sha256(java.nio.file.Files.readAllBytes(bridge)).equals("{sha(original_bridge)}")',
         f'sha256(java.nio.file.Files.readAllBytes(bridge)).equals("{sha(recovery_bridge)}")'),
        (f'"--suite", {json.dumps(str(original_suite))}, "--run-index"',
         f'"--suite", {json.dumps(str(recovery_json))}, "--run-index"'),
    )
    revised = source
    for before, after in pairs:
        revised = _replace_once(revised, before, after)
    restored = revised
    for before, after in reversed(pairs):
        restored = _replace_once(restored, after, before)
    _require(restored == source, 'recovery renderer changed the original job bindings')
    _require(all(json.dumps(str(path)) not in revised
                 for path in (original_control, original_bridge, original_suite)),
             'original startup location remains embedded in the recovery helper')
    _require('require(row.ordinal == 0,' in revised and 'resumeOffset() { return 0; }' in revised,
             'first physical run must start at original ordinal zero')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(revised)
    return destination


def verify_java(helper, scenarios_path):
    """Compile the exact recovery helper and verify its original slot views offline."""
    module_path = PLAN.parent / 'mixed_java.py'
    spec = spec_from_file_location('continuous_training_java_verifier', module_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_java(helper, scenarios_path)
