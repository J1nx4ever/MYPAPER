"""Narrow compatibility review of historical environment evidence.

The old certificate remains evidence for its original environment. This review
only preserves its unchanged accounting checks while allowing explicit scenario
v2 parsing for the new smoke. It is not live AnyLogic v2 acceptance.
"""
import json
from pathlib import Path

from python.collect_offline import ENV_CHECKS, environment_pins
from python.storage_compatibility import source_pins_match
from .contract import require
from .plan_io import sha

SCENARIO_BEFORE = '0dd856d5f6e6b1392ad770bdb00ab8d40fb3ed237ca3c8b043f0c7c783f9a33a'
SCENARIO_AFTER = 'd7af39f3b52e055765ee8d8cfa29e0ed9162eebbc77d4430a4702d9d6ba96b71'


def reviewed_pins_match(recorded, current):
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return False
    key='python/scenario_contract.py'
    if recorded.get(key)!=SCENARIO_BEFORE or current.get(key)!=SCENARIO_AFTER:
        return False
    adjusted=dict(current)
    adjusted[key]=SCENARIO_BEFORE
    return source_pins_match(recorded,adjusted,storage_migration=True)


def verify_environment(spec, reward):
    require(isinstance(spec,dict) and spec.get('path') and spec.get('sha256'),
            'missing environment acceptance')
    path=Path(spec['path'])
    require(sha(path)==spec['sha256'],'environment certificate hash mismatch')
    certificate=json.loads(path.read_bytes())
    require(certificate.get('scope')=='actual_anylogic_environment_three_criteria' and
            certificate.get('status')=='passed' and
            set(certificate.get('passed_checks',[]))==ENV_CHECKS,
            'complete actual environment evidence required')
    require(reviewed_pins_match(certificate.get('source_sha256'),environment_pins()),
            'environment differs from the fixed scenario v2 compatibility review')
    require(certificate.get('reward_settings')==reward,'physical reward accounting differs')
    require(bool(certificate.get('evidence')),'missing environment evidence')
    for item in certificate['evidence']:
        require(sha(item['path'])==item['sha256'],'environment evidence changed')
    return certificate
