"""Read-only completion audit; no model loading, fitting, RPC or collection.

Run only after suite_complete.json and run_status=COMPLETE are published.
The sole writes are new audit publications under reports/.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from business_graph_summary_evaluation_v1 import budget
from business_graph_summary_evaluation_v1.gate import check_jobs, check_sources
from business_graph_summary_campaign_v1.plan_io import encoded
from business_graph_summary_v1.contract import ablation_contract
from business_graph_summary_v1.targets import composite_contract
from grade_event_memory_v1.journal import read_journal

PLAN = ROOT / 'change_plans/token_attention_001/business_graph_summary_009/checkpoint_evaluation_001'
REPORT = ROOT / 'reports/token_attention_001/business_graph_summary_009/checkpoint_evaluation_001'
HASH_CACHE = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def file_ref(path, *, cache=True):
    path = Path(path).resolve()
    if not cache or str(path) not in HASH_CACHE:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        value = dict(path=str(path), sha256=digest.hexdigest())
        if cache:
            HASH_CACHE[str(path)] = value
        return value
    return HASH_CACHE[str(path)]


def verify(ref):
    require(file_ref(ref['path']) == ref, 'artifact SHA differs: ' + ref['path'])
    return read(ref['path'])


def canonical_hash(value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    return hashlib.sha256(raw).hexdigest()


def audit(suite_path):
    suite_path = Path(suite_path).resolve()
    suite, suite_ref = read(suite_path), file_ref(suite_path)
    output = Path(suite['python_output'])
    complete_path, status_path = output / 'suite_complete.json', output / 'run_status.json'
    require(complete_path.is_file(), 'not ready: final suite completion not published')
    status = read(status_path)
    require(status['status'] == 'COMPLETE' and status['completed_shifts'] == 100,
            'not ready: final run status must be COMPLETE100')
    check_jobs(suite)
    check_sources(suite['source_pins'])
    require(not (Path(suite['control_directory']) / 'STOP.properties').exists(),
            'evaluation STOP exists; audit requires explicit investigation')
    complete = read(complete_path)
    require(complete['campaign'] == suite_ref and complete['checkpoint_update'] == 13 and
            complete['physical_shifts'] == 100 and
            all(complete[k] == 0 for k in ('training_shifts', 'training_updates',
                'optimizer_updates', 'gradient_updates')), 'completion scope differs')
    selection = verify(suite['selection'])
    require(selection['selected_update'] == 13 and
            selection['selected_checkpoint'] == suite['parent_artifacts']['checkpoint'] and
            selection['selection_frozen_before_100_new_evaluation_shifts'] is True and
            complete['selection_review'] == suite['selection'] and
            suite['checkpoint_selection_uses_evaluation_results'] is False,
            'pre-evaluation checkpoint selection binding differs')
    parent = suite['parent_artifacts']
    for ref in parent.values():
        verify(ref) if ref['path'].endswith('.json') else require(
            file_ref(ref['path']) == ref, 'checkpoint bytes changed')
    parent_manifest, parent_seal = read(parent['manifest']['path']), read(parent['seal']['path'])
    require(parent_manifest['update_index'] == 13 and
            parent_manifest['sha256'] == parent_seal['sha256'] == parent['checkpoint']['sha256'] and
            parent_seal['manifest_sha256'] == parent['manifest']['sha256'],
            'parent checkpoint manifest/seal binding differs')
    require(complete['parent_artifacts_before'] == complete['parent_artifacts_after'] == parent,
            'fixed parent artifacts before/after differ')
    reports = complete['reports']
    for ref in reports.values():
        require(file_ref(ref['path']) == ref, 'final report bytes changed')
    receipt = read(reports['receipt']['path'])
    evaluation = read(reports['evaluation']['path'])
    require(receipt['campaign'] == evaluation['campaign'] == suite_ref and
            receipt['phase_config'] == evaluation['phase_config'] == suite['phase_config'] and
            receipt['parent_artifacts_before'] == receipt['parent_artifacts_after'] == parent,
            'final receipt provenance differs')
    require(receipt['model_state_before'] == receipt['model_state_after'] ==
            evaluation['model_state_before'] == evaluation['model_state_after'],
            'recorded full model/optimizer/training state before/after differ')
    require(evaluation['training_state_unchanged'] is True and
            evaluation['checkpoint_provenance'] == parent['checkpoint'] and
            evaluation['n_shifts'] == 100 and
            all(evaluation[k] == 0 for k in ('optimizer_updates', 'gradient_updates',
                                            'actor_updates', 'critic_updates')),
            'fixed-policy zero-update evaluation receipt differs')
    require(evaluation['fingerprints'] == complete['fingerprints'] and
            complete['fingerprints']['actor'] == parent_manifest['actor_digest'] and
            isinstance(complete['fingerprints']['critic'], str) and
            len(complete['fingerprints']['critic']) == 64,
            'selected Actor/Critic parameter digest differs')
    require(receipt['reports'] == {k: v for k, v in reports.items() if k != 'receipt'},
            'receipt report inventory differs')
    actor_ref = {k: receipt['actor'][k] for k in ('path', 'sha256')}
    require(file_ref(actor_ref['path']) == actor_ref, 'evaluation Actor artifact changed')
    actor_manifest = read(actor_ref['path'] + '.manifest.json')
    actor_seal = read(actor_ref['path'] + '.seal.json')
    require(actor_manifest['update_index'] == 13 and
            actor_manifest['actor_digest'] == parent_manifest['actor_digest'] and
            actor_manifest['sha256'] == actor_seal['sha256'] == actor_ref['sha256'] and
            actor_seal['manifest_sha256'] == file_ref(actor_ref['path'] + '.manifest.json')['sha256'],
            'evaluation Actor seal or selected checkpoint binding differs')

    sidecar_ref_before = file_ref(suite['evaluation_budget_path'], cache=False)
    with budget.connection(suite['evaluation_budget_path'], readonly=True) as con:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        require(tables == {'metadata', 'campaigns', 'slots'}, 'unexpected sidecar table or commits found')
        budget._metadata(con, False)
        registered = con.execute('SELECT payload,checksum FROM campaigns WHERE id=?',
                                 (suite['campaign_id'],)).fetchone()
        require(registered is not None, 'independent evaluation registration missing')
        entry = budget._payload(registered)
        require(entry['suite_reference'] == suite_ref and entry['cap'] == 100 and
                entry['approval_sha256'] == file_ref(suite_path.parent / 'approval.json')['sha256'],
                'registered suite, approval or cap differs')
        state = budget._state(con, suite, entry)
        require(con.execute('SELECT count(*) FROM slots WHERE campaign=?',
                (suite['campaign_id'],)).fetchone()[0] == 100, 'wrong sidecar slot count')
    campaign = state['campaigns'][suite['campaign_id']]
    local_counts = dict(Counter(row['status'] for row in state['jobs'].values()))
    require(local_counts == {'COMPLETE': 100} and campaign['commits'] == {},
            'all100 COMPLETE slots with zero commits required')

    cfg = verify(suite['phase_config'])
    require(cfg['jobs'] == suite['jobs'], 'phase job inventory differs')
    expected_refs, per_run = [], []
    graph_hash = canonical_hash(ablation_contract())
    for job in suite['jobs']:
        run = job['run_index']
        row = state['jobs'][str(run)]
        r = row['receipt']
        raw_folder = Path(job['python_output']) / 'episodes' / ('run_' + str(run))
        refs = r['references']
        require(set(refs) == {'config', 'actor', 'journal', 'raw_manifest', 'episode',
                            'episode_manifest', 'episode_seal'}, 'incomplete admitted source bindings')
        for ref in refs.values():
            require(file_ref(ref['path']) == ref, 'admitted source SHA changed: ' + ref['path'])
        require(row['dataset_ref'] == refs['episode_manifest'] and refs['config'] == suite['phase_config'] and
                refs['actor'] == actor_ref and r['phase_config'] == suite['phase_config'] and
                read(raw_folder / 'material_receipt.json') == r,
                'budget receipt, saved receipt or fixed input provenance differs')
        require(refs['raw_manifest'] == file_ref(raw_folder / 'manifest.json') and
                Path(refs['journal']['path']) == Path(job['material_journal']),
                'job-local raw manifest or material path differs')
        manifest, seal = read(refs['episode_manifest']['path']), read(refs['episode_seal']['path'])
        identity = manifest['identity']
        require(manifest['status'] == seal['status'] == 'complete' and
                manifest['schema'] == seal['schema'] == 'business-graph-summary-joint-memory-episode-hdf5-v1' and
                manifest['hdf5_sha256'] == seal['hdf5_sha256'] == refs['episode']['sha256'] and
                seal['manifest_sha256'] == refs['episode_manifest']['sha256'],
                'derived manifest/seal/hash chain differs')
        expected_identity = dict(run_index=run, environment_seed=job['experiment_seed'],
            actor_sampling_seed=job['actor_sampling_seed'], recording_id=job['recording_id'],
            candidate_id='s01_BG128', split='validation', source_kind='fixed_actor',
            config_sha256=suite['phase_config']['sha256'], policy_sha256=actor_ref['sha256'],
            policy_version=parent_manifest['actor_digest'], graph_contract_sha256=graph_hash)
        require(identity == r['identity'] and all(identity.get(k) == v for k, v in expected_identity.items()),
                'sealed episode identity differs from fixed checkpoint/new seed/job')
        raw = read(refs['raw_manifest']['path'])
        raw_receipt = read(raw_folder / 'receipt.json')
        require(raw['status'] == 'complete' and raw['source_kind'] == 'real_anylogic_rpc' and
                raw['provenance']['configuration'] == suite['phase_config'] and
                raw['provenance']['run_index'] == run and raw['provenance']['seed'] == job['experiment_seed'] and
                raw['provenance']['recording_id'] == job['recording_id'] and
                raw_receipt == r['raw_receipt'] and raw_receipt['normal_completion'] is True and
                raw_receipt['four_windows'] == 4 and r['four_windows'] == 4 and
                raw_receipt['run_index'] == run and raw_receipt['experiment_seed'] == job['experiment_seed'] and
                raw['policy_version'] == raw_receipt['policy_version'] == parent_manifest['actor_digest'] and
                raw_receipt['phase_config'] == suite['phase_config'] and raw_receipt['role'] == 'final',
                'actual raw RPC completion or source identity differs')
        require(raw_receipt['dataset_files'] == raw['files_sha256'], 'raw dataset hash inventories differ')
        for name, sha in raw['files_sha256'].items():
            require(file_ref(raw_folder / name)['sha256'] == sha, 'raw dataset bytes changed')
        for name, sha in raw['source_sha256'].items():
            require(file_ref(ROOT / name)['sha256'] == sha, 'raw transport implementation changed')
        journal = read_journal(refs['journal']['path'])
        require(journal.header['source_kind'] == 'actual_anylogic' and
                journal.header['run_index'] == run and journal.header['seed'] == job['experiment_seed'] and
                journal.header['recording_id'] == job['recording_id'] and
                journal.header['config_sha256'] == suite['phase_config']['sha256'] and
                journal.logical_sha256 == identity['journal_logical_sha256'] and
                journal.file_sha256 == identity['journal_file_sha256'] == refs['journal']['sha256'] and
                journal.raw_process_calls == identity['raw_process_calls'] and
                len(journal.events) == manifest['counts']['events'],
                'complete material journal identity/logical seal differs')
        n = manifest['counts']['transitions']
        require(n == raw['transitions'] == raw_receipt['transitions'] == r['transitions'] and n > 0 and
                manifest['counts']['states'] == n + 1 and manifest['counts']['settlements'] == 4 and
                manifest['composite_contract'] == r['composite_contract'] == composite_contract() and
                manifest['ppo_authorized'] is False,
                'episode count or frozen joint-cost contract differs')
        with h5py.File(refs['episode']['path'], 'r') as source:
            require(source.attrs['schema'] == manifest['schema'] and source.attrs['status'] == 'complete' and
                    json.loads(source.attrs['identity_json']) == identity and
                    json.loads(source.attrs['composite_contract_json']) == composite_contract() and
                    source.attrs['terminal_mc_composite_cost'] == 0.0 and
                    source['states'].shape == (n + 1, 1519) and source['states'].dtype == np.dtype('float32') and
                    source['transitions/action'].shape == (n,) and
                    np.isfinite(source['states'][...]).all(),
                    'HDF5 internal identity, counts or raw observation validity differs')
        expected_refs.append(row['dataset_ref'])
        per_run.append(dict(run_index=run, environment_seed=job['experiment_seed'],
            actor_sampling_seed=job['actor_sampling_seed'], status='COMPLETE', transitions=n,
            states=n+1, windows=4, manifest=refs['episode_manifest'], seal=refs['episode_seal'],
            hdf5=refs['episode'], material_journal=refs['journal'],
            raw_manifest=refs['raw_manifest'], material_receipt=file_ref(raw_folder / 'material_receipt.json')))
    require(len(per_run) == 100 and complete['source_refs'] == receipt['source_refs'] ==
            evaluation['source_refs'] == expected_refs, 'final source_refs do not bind exactly the100 canonical slots')
    require(evaluation['n_states'] == sum(row['transitions'] for row in per_run),
            'final decision-state count differs from canonical complete episodes')
    expected_runs = {job['run_index'] for job in suite['jobs']}
    actual_runs = {int(path.name[4:]) for path in (output / 'evaluation/episodes').glob('run_*') if path.is_dir()}
    require(actual_runs == expected_runs and len(list((output / 'evaluation/material').glob('*.h5.manifest.json'))) == 100,
            'unexpected extra or missing physical/derived episode')
    unchanged = {}
    for key, ref in suite['budget_state_before'].items():
        if isinstance(ref, dict) and set(ref) == {'path', 'sha256'}:
            require(file_ref(ref['path']) == ref, 'historical budget/policy bytes changed: ' + key)
            unchanged[key] = ref
    merged = budget.budget_snapshot()
    merged_counts = dict(Counter(row['status'] for row in merged['jobs'].values()))
    require(merged['maximum_total_shifts'] == merged['additional_reserved_shifts'] == 21083 and
            len(merged['jobs']) == 21083 and merged_counts == {'COMPLETE': 21073, 'UNKNOWN': 1, 'PLANNED': 9} and
            merged['active_h2_reserved_shifts'] == merged['legacy_reserved_shifts'] == 0,
            'complete merged21083 accounting or preserved historical states differ')
    require(file_ref(suite['evaluation_budget_path'], cache=False) == sidecar_ref_before,
            'evaluation sidecar changed while audit was running')
    protected = ROOT.parents[1] / 'python/episode_budget_ppo.py'
    require(file_ref(protected)['sha256'] == '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96',
            'protected PPO core bytes changed')
    return dict(schema='bg128-independent-evaluation-completion-audit-v1', status='PASS',
        created_at_utc=datetime.now(timezone.utc).isoformat(), suite=suite_ref,
        audit_implementation=file_ref(__file__), completion=file_ref(complete_path),
        evaluation_receipt=reports['receipt'], evaluation_report=reports['evaluation'],
        checkpoint_update=13, parent_artifacts_before=parent, parent_artifacts_after=parent,
        actor_artifact=actor_ref, selected_actor_digest=parent_manifest['actor_digest'],
        physical_shifts=100, decision_states=evaluation['n_states'], window_settlements=400,
        training_updates=0, optimizer_updates=0, gradient_updates=0, model_loads_during_this_audit=0,
        rpc_starts_during_this_audit=0, collection_calls_during_this_audit=0,
        sidecar=sidecar_ref_before, sidecar_status_counts=local_counts, sidecar_commits=0,
        merged=dict(maximum_total_shifts=21083, jobs=len(merged['jobs']), campaigns=len(merged['campaigns']),
            status_counts=merged_counts, active_h2_reserved_shifts=0),
        unchanged_prior_budget_references=unchanged, protected_ppo=file_ref(protected),
        audit_scope=dict(independent_checks='file seals, identities, provenance, counts, finite inputs, budget accounting',
            model_state_evidence='checkpoint bytes plus source-bound runtime ReadOnlyTrainingGuard before/after receipts',
            model_parameters_deserialized=False, model_predictions_recomputed=False,
            evaluation_metrics_recomputed=False),
        checks=dict(all100_slots_complete=True, no_unknown_claimed_or_planned_in_this_campaign=True,
            no_optimizer_or_training_commits=True, all100_manifests_and_seals_verified=True,
            all100_actual_rpc_and_material_sources_verified=True,
            all100_finite_raw_observation_hdf5_inputs_verified=True,
            all100_source_refs_bound_to_final_receipt=True,
            checkpoint13_actor_critic_provenance_unchanged=True,
            recorded_model_optimizer_training_state_unchanged=True,
            original_independent_json_unchanged=True, original_sampling_json_unchanged=True,
            original_bg128_training_ledger_unchanged=True, full_merged21083_verified=True,
            preserved_historical_unknown1_planned9=True, protected_ppo_unchanged=True),
        file_hashes_verified=len(HASH_CACHE), per_run=per_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default=str(PLAN / 'prepared_001/suite.json'))
    parser.add_argument('--output', default=str(REPORT / 'completion_independent_audit.json'))
    args = parser.parse_args(argv)
    result = audit(args.suite)
    path = Path(args.output).resolve()
    require(path.parent == REPORT.resolve(), 'audit publications must stay in the dedicated report directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(encoded(result))
    markdown = path.with_suffix('.md')
    text = ('# BG128 100班独评完成独立审计\n\n'
        f"结果：**PASS**。100班均 COMPLETE，{result['decision_states']} 个决策状态，400个结算窗口。\n\n"
        '逐班核对预算receipt、原始RPC文件、物料journal、派生HDF5、manifest、seal及最终source_refs，'
        '全部绑定固定 checkpoint13 和该班独立环境／动作seed。模型、Actor/Critic参数摘要与优化器训练状态前后凭证一致；'
        '本审计没有加载模型、训练、采集或启动RPC。\n\n'
        '本轮sidecar为100 COMPLETE，UNKNOWN／CLAIMED／PLANNED均为0，训练commit为0。'
        '完整合并总数21083：COMPLETE 21073、历史 UNKNOWN 1、历史 PLANNED 9；H2未来预留0。\n\n'
        '固定checkpoint的三件artifact、旧独评JSON、采样对比JSON、BG128训练账本、其余冻结预算引用与受保护PPO核心字节均未改变。'
        '独立审计只验证数据、身份、预算和只读推理证据，不重新选择checkpoint，也不重新计算模型预测指标。\n')
    with markdown.open('x') as stream:
        stream.write(text)
    print(json.dumps(dict(status='PASS', physical_shifts=100, merged_total=21083,
                          report=file_ref(path), markdown=file_ref(markdown)), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
