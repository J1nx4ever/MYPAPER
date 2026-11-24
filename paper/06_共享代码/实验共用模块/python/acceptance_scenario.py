"""Explicit opt-in boundary scenario identity, independent of training settings."""
import math

PREFIX = 'config.acceptance.'
LONG_LOAD_RPC = {
    PREFIX + 'scenario_version': 1.0,
    PREFIX + 'scenario_code': 1.0,
    PREFIX + 'load_not_before_seconds': 14500.0,
}


def boundary_rpc(mode):
    return dict(LONG_LOAD_RPC, **{
        PREFIX+'scenario_code': 2.0,
        PREFIX+'boundary_mode': float(mode),
        PREFIX+'target_truck_id': 0.0,
        PREFIX+'load_target_seconds': 7200.0 if mode==1 else (28800.0 if mode==2 else 0.0),
        PREFIX+'unload_target_seconds': 14400.0 if mode==1 else (28800.0 if mode==3 else 0.0),
    })


BOUNDARY_SCENES = {1:'boundary_internal_7200_14400',2:'boundary_load_cutoff',3:'boundary_unload_cutoff'}


def validate_scenario_config(config):
    if 'acceptance_scenario' not in config:
        return None  # Historical ordinary acceptance profiles remain readable.
    spec = config['acceptance_scenario']
    if not isinstance(spec, dict) or set(spec) != {'name', 'expected_rpc', 'episodes', 'scene_id'}:
        raise ValueError('invalid acceptance_scenario fields')
    expected = spec['expected_rpc']
    if spec['name']=='boundary_service_v1':
        if not isinstance(expected,dict):raise ValueError('invalid boundary RPC settings')
        mode=expected.get(PREFIX+'boundary_mode')
        if type(mode) not in (int,float) or mode not in BOUNDARY_SCENES:
            raise ValueError('unsupported boundary mode')
        if (set(expected)!=set(boundary_rpc(mode)) or expected!=boundary_rpc(mode)
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in expected.values())
                or type(spec['episodes']) is not int or spec['episodes']!=1
                or spec['scene_id']!=BOUNDARY_SCENES[mode]):
            raise ValueError('unsupported boundary acceptance_scenario settings')
        return spec
    if (spec['name'] != 'long_load_floor_v1' or not isinstance(expected, dict)
            or set(expected) != set(LONG_LOAD_RPC)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in expected.values())
            or expected != LONG_LOAD_RPC):
        raise ValueError('unsupported acceptance_scenario RPC settings')
    if type(spec['episodes']) is not int or spec['episodes'] != 2 or spec['scene_id'] != 'long_load_floor_14500':
        raise ValueError('long-load acceptance requires two explicit same-scene episodes')
    return spec


def validate_scenario_jobs(config, jobs):
    spec = validate_scenario_config(config)
    if spec is None:
        return
    if (len(jobs) != spec['episodes']
            or any(j.get('scene_id') != spec['scene_id'] for j in jobs)
            or any(type(j.get(k)) is not int or j[k] <= 0 for j in jobs for k in ('run_index', 'experiment_seed'))
            or any(len({j[k] for j in jobs}) != len(jobs) for k in ('run_index', 'experiment_seed'))):
        raise ValueError('acceptance scenario jobs/count/identity mismatch')


def validate_scenario_rpc(config, components):
    spec = validate_scenario_config(config)
    actual = {k: v for k, v in components.items() if k.startswith(PREFIX)}
    if spec is None and not actual and config.get('acceptance_interruption') is None:
        return
    expected = spec['expected_rpc'] if spec is not None else dict(LONG_LOAD_RPC, **{PREFIX + 'scenario_code': 0.0})
    if (set(actual) != set(expected)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in actual.values())
            or actual != expected):
        raise ValueError(f'RPC acceptance scenario mismatch: expected {expected}, received {actual}')
