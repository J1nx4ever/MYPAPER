"""Explicit configuration, source, engineering and budget gates; no listeners."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
CAMPAIGN_CONTRACT = 'baseline1-campaign-v1'
PHASE_CONTRACT = 'baseline1-rpc-phase-v1'
PROTECTED_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p))


def load_reference(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'exact file reference required')
    p = Path(ref['path'])
    require(p.is_absolute() and sha(p) == ref['sha256'], 'file reference changed: ' + str(p))
    return json.loads(p.read_bytes())


def source_pins():
    historical = ROOT / 'reports/baseline1_single_critic_001/source_pins_before.json'
    # Local ALP bytes are provenance, not the live runtime configuration oracle.
    # Keep Python/schema/protocol pins; the simulator is checked through RPC.
    paths = [REPO / p for p in json.loads(historical.read_text()) if Path(p).suffix.lower() != '.alp']
    paths += [p for p in (ROOT / 'baseline1_v1').glob('*.py') if not p.name.startswith('test_')]
    paths += [ROOT / 'change_plans/stage8_seed1_one_click_001/H2CampaignPlan.java.template']
    paths += [ROOT/'python'/name for name in ('hdf5_storage.py','hdf5_events.py','hdf5_arrays.py',
              'storage_contract.py','storage_compatibility.py','storage_audit.py')]
    paths += [ROOT/'storage_requirements.txt']
    return {str(p.resolve()): sha(p) for p in sorted(set(paths))}


def check_sources(pins):
    require(pins == source_pins(), 'implementation/source inventory changed; prepare a new version')
    require(sha(REPO / 'python/episode_budget_ppo.py') == PROTECTED_SHA, 'protected PPO core changed')


def check_phase(cfg):
    require(cfg.get('contract') == PHASE_CONTRACT, 'Baseline1 RPC phase contract required')
    from python.storage_contract import HDF5, storage_options
    storage=storage_options(cfg.get('storage'))
    if storage['format']==HDF5:
        from python.hdf5_storage import _h5py
        _h5py()
    require(cfg.get('rpc_configuration_authoritative') is True and cfg.get('require_local_alp_audit') is False,
            'RPC-owned configuration and schema checks must be retained')
    require(cfg.get('kind') in ('calibration', 'single_seed_trial'), 'unknown phase kind')
    require(cfg.get('recording_id') and cfg.get('campaign_id') and cfg.get('phase_id'), 'phase identity required')
    for key in ('python_output', 'event_directory'):
        p = Path(cfg[key])
        require(p.is_absolute() and p.is_relative_to(ROOT / 'runs') and p != ROOT / 'runs',
                'new dedicated runs path required: ' + key)
    execution = cfg['execution']
    require(execution['host'] == '127.0.0.1' and type(execution['port']) is int
            and 1024 <= execution['port'] <= 65535, 'explicit local RPC address required')
    require(type(execution['callback_timeout_seconds']) in (int, float)
            and math.isfinite(execution['callback_timeout_seconds'])
            and execution['callback_timeout_seconds'] > 0, 'explicit callback timeout required')
    require(type(cfg['capture_physical_events']) is bool, 'event recording choice must be explicit')
    require(cfg['diagnostic_recording'] == dict(transition_jsonl=False, rpc_jsonl=False,
                                               rollout_snapshots=False), 'detailed logging must stay off')
    jobs = cfg.get('jobs')
    require(isinstance(jobs, list) and jobs, 'complete phase job manifest required')
    require(len({j['run_index'] for j in jobs}) == len(jobs)
            and len({j['experiment_seed'] for j in jobs}) == len(jobs), 'duplicate phase job identity')
    for job in jobs:
        require(job['phase_id'] == cfg['phase_id'] and job['recording_id'] == cfg['recording_id'],
                'phase job belongs to another recording')
        require(type(job['run_index']) is int and 0 < job['run_index'] < 2**31
                and type(job['experiment_seed']) is int and 0 < job['experiment_seed'] < 2**48,
                'invalid explicit simulator run/seed')
        require(Path(job['event_bundle']).parent == Path(cfg['event_directory']), 'event parent differs')
        require(Path(job['python_output']) == Path(cfg['python_output']), 'Python output differs')
        require(job['expected_rpc_config'], 'explicit expected RPC configuration required')
    if cfg['kind'] == 'calibration':
        from python.native_behavior import validate_spec
        validate_spec(cfg['behavior'], check_files=True)
        require(cfg['capture_physical_events'] is True,
                'existing native RPC implementation needs scoped physical event evidence')
        require(all(j['role'] == 'calibration' for j in jobs), 'calibration is not PPO training')
    else:
        require(cfg['capture_physical_events'] is False, 'trial must keep detailed physical events off')
        require(all(j['group'] == 'BASELINE' for j in jobs), 'not a single Critic phase')
    return cfg


def preflight(campaign_path, *, approval_path=None):
    """Read-only. An approval is mandatory only for execution, never for review."""
    path = Path(campaign_path).resolve()
    campaign = json.loads(path.read_bytes())
    require(campaign.get('contract') == CAMPAIGN_CONTRACT, 'Baseline1 campaign contract required')
    require(campaign['legacy_reserved_shifts'] == 11280, 'old H2 allocation cannot be borrowed')
    check_sources(campaign['source_pins'])
    phases = [check_phase(load_reference(ref)) for ref in campaign['phase_configs']]
    jobs = [j for p in phases for j in p['jobs']]
    require(jobs == campaign['jobs'] and len(jobs) == campaign['new_shift_cap'], 'campaign budget/job mismatch')
    require(all(p['campaign_id'] == campaign['campaign_id'] and p['kind'] == campaign['kind'] for p in phases),
            'cross-campaign phase')
    require(len({j['run_index'] for j in jobs}) == len(jobs)
            and len({j['experiment_seed'] for j in jobs}) == len(jobs), 'global identity collision')
    from python.protocol_artifacts import verify_protocol_artifacts
    from python.collect_offline import verify_environment
    verify_protocol_artifacts()
    for p in phases:
        verify_environment(p['environment_acceptance'], p['reward'],storage_migration=True)
    master = campaign['master_plan']
    require(sha(master['path']) == master['sha256'], 'master plan changed')
    require(sha(campaign['java_helper']['path']) == campaign['java_helper']['sha256'], 'Java helper changed')
    engineering = load_reference(campaign['engineering_acceptance'])
    require(engineering['status'] == 'PASS_SYNTHETIC_ONLY' and engineering['source_pins'] == campaign['source_pins'],
            'matching new engineering acceptance required')
    if campaign['kind'] == 'single_seed_trial':
        cfg = campaign['learner_config']
        require(cfg['scope'] == 'formal_online' and cfg['formal_execution_authorized'] is False,
                'prepared formal learner must remain unapproved until the budget gate releases it')
        require(cfg['beta'] in load_reference(campaign['calibration_scale'])['beta_candidates'],
                'trial beta must be an explicit new-scale candidate')
        b, u = cfg['collection']['episodes_per_batch'], cfg['collection']['batches']
        require(sum(j['role'] == 'train' for j in jobs) == b * u, 'training role budget differs')
        require(phases[0]['jobs'][0]['role'] == 't0' and phases[-1]['jobs'][0]['role'] == 'final',
                'independent t0 and final evaluation cannot be skipped')
        require(all(j['target_training_update'] <= u for j in jobs), 'job exceeds update budget')
    approval = None
    if approval_path is not None:
        approval = json.loads(Path(approval_path).read_bytes())
        require(approval.get('decision') == 'approved' and approval.get('campaign_id') == campaign['campaign_id']
                and approval.get('campaign_sha256') == sha(path)
                and approval.get('max_new_shifts') == len(jobs)
                and approval.get('legacy_reserved_shifts') == 11280
                and approval.get('accounting') == 'additional_to_h2_reserved',
                'exact separate budget approval required before any listener or output creation')
    return campaign, phases, approval
