"""Environment rule identity, independent of the frozen Reward V1.0 identity."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib
import json
import math

PREFIX='config.waste_task_'
FIELDS={'limit_enabled','limit_version','upper_tons','overrun_tons'}
FIELDS_V2=FIELDS|{'dispatch_stop_tons'}


@dataclass(frozen=True)
class ScenarioConfiguration:
    waste_task_limit_enabled: bool
    waste_task_limit_version: int
    waste_target_tons: float
    waste_task_upper_tons: float
    waste_task_overrun_tons: float
    waste_task_dispatch_stop_tons: float | None = None

    def __post_init__(self):
        if type(self.waste_task_limit_enabled) is not bool or type(self.waste_task_limit_version) is not int or self.waste_task_limit_version not in (1,2):
            raise ValueError('scenario requires a boolean flag and supported integer rule version 1 or 2')
        for name in ('waste_target_tons','waste_task_upper_tons','waste_task_overrun_tons'):
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError('invalid scenario quantity: '+name)
            object.__setattr__(self,name,float(value))
        if self.waste_task_limit_version==1:
            if self.waste_task_dispatch_stop_tons is not None:
                raise ValueError('v1 scenario must not carry a dispatch stop')
            stop=self.waste_target_tons
        else:
            stop=self.waste_task_dispatch_stop_tons
            if type(stop) not in (int,float) or not math.isfinite(stop) or stop<0:
                raise ValueError('v2 scenario requires an explicit finite nonnegative dispatch stop')
            stop=float(stop)
            object.__setattr__(self,'waste_task_dispatch_stop_tons',stop)
            if self.waste_task_limit_enabled and stop<self.waste_target_tons:
                raise ValueError('v2 dispatch stop must cover the waste delivery target')
        if self.waste_task_limit_enabled:
            if not math.isclose(self.waste_task_upper_tons,stop+self.waste_task_overrun_tons,abs_tol=1e-9,rel_tol=0):
                raise ValueError('waste upper must equal '+('target' if self.waste_task_limit_version==1 else 'dispatch stop')+' plus one-truck tolerance')
            if self.waste_target_tons==0 and (self.waste_task_upper_tons or self.waste_task_overrun_tons):
                raise ValueError('zero waste target requires zero upper/tolerance')
            if self.waste_target_tons>0 and self.waste_task_overrun_tons<=0:
                raise ValueError('positive waste task requires positive truck tolerance')
        elif self.waste_task_upper_tons or self.waste_task_overrun_tons or (self.waste_task_limit_version==2 and stop):
            raise ValueError('disabled waste limit must encode inactive upper/tolerance/stop as zero')

    def to_dict(self):
        result=asdict(self)
        # Preserve the exact historical v1 artifact identity and fingerprint.
        if self.waste_task_limit_version==1:result.pop('waste_task_dispatch_stop_tons')
        return result
    @property
    def scenario_id(self):return 'dispatch-scenario-waste-task-v'+str(self.waste_task_limit_version)
    @property
    def fingerprint(self):
        payload=dict(scenario_id=self.scenario_id,**self.to_dict())
        return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def from_components(components):
    present={key[len(PREFIX):] for key in components if key.startswith(PREFIX)}
    if not present:
        if any(key.startswith('audit.waste_task.') for key in components):
            raise ValueError('waste audit labels require explicit scenario configuration')
        return None  # unversioned historical metadata, NOT inferred disabled
    if 'limit_version' not in present:raise ValueError('explicit waste scenario version missing')
    version=float(components[PREFIX+'limit_version'])
    if version not in (1,2):raise ValueError('unsupported waste scenario version')
    fields=FIELDS if version==1 else FIELDS_V2
    if present!=fields:raise ValueError('missing/unknown waste scenario fields: '+str(present^fields))
    values={name:float(components[PREFIX+name]) for name in fields}
    if not all(math.isfinite(x) for x in values.values()):raise ValueError('nonfinite waste scenario metadata')
    if values['limit_enabled'] not in (0,1):raise ValueError('unsupported waste scenario flag')
    if 'config.target_waste_tons_shift' not in components:raise ValueError('scenario target missing from RPC')
    return ScenarioConfiguration(bool(values['limit_enabled']),int(values['limit_version']),
        float(components['config.target_waste_tons_shift']),values['upper_tons'],values['overrun_tons'],
        values.get('dispatch_stop_tons'))


def to_components(config):
    result={PREFIX+'limit_enabled':float(config.waste_task_limit_enabled),
            PREFIX+'limit_version':float(config.waste_task_limit_version),
            PREFIX+'upper_tons':config.waste_task_upper_tons,PREFIX+'overrun_tons':config.waste_task_overrun_tons,
            'config.target_waste_tons_shift':config.waste_target_tons}
    if config.waste_task_limit_version==2:
        result[PREFIX+'dispatch_stop_tons']=config.waste_task_dispatch_stop_tons
    return result


def metadata(config):
    if config is None:
        return dict(scenario_id='legacy-unversioned',scenario_configuration=None,scenario_fingerprint=None)
    return dict(scenario_id=config.scenario_id,scenario_configuration=config.to_dict(),scenario_fingerprint=config.fingerprint)


def assert_same_scenario(*artifacts):
    identities=[]
    for item in artifacts:
        if 'scenario_configuration' not in item:
            if any(k in item for k in ('scenario_id','scenario_fingerprint')):raise ValueError('partial scenario artifact identity')
            identities.append(metadata(None));continue
        raw=item['scenario_configuration']
        expected=metadata(None if raw is None else ScenarioConfiguration(**raw))
        if any(item.get(k)!=v for k,v in expected.items()):raise ValueError('scenario artifact fingerprint/configuration differs')
        identities.append(expected)
    if identities and any(x!=identities[0] for x in identities):raise ValueError('cross-scenario checkpoint/evaluation/report forbidden')
    return identities[0] if identities else metadata(None)
