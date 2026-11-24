"""Prepare immutable new-run identities, never approve or start execution."""
import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path

from .gate import ROOT, CAMPAIGN_CONTRACT, PHASE_CONTRACT, encoded, reference, require, source_pins
from .java_plan import COLUMNS, render_java
from .reward import CONTRACT


def _seed(campaign, purpose, ordinal):
    return int.from_bytes(hashlib.sha256(f'{campaign}:{purpose}:{ordinal}'.encode()).digest()[:6], 'big') or 1


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.open('xb').write(encoded(value))


def _reference_configs():
    old = ROOT / 'change_plans/offline_data_readiness_001/campaign_003'
    return {p: json.loads((old / f'policy{p}_train.json').read_bytes()) for p in (0, 1, 2, 4)}


def _phase(campaign_id, phase_id, kind, template, jobs, execution, behavior=None):
    recording = campaign_id + '-' + phase_id
    output = ROOT / 'runs' / campaign_id / phase_id
    events = ROOT / 'runs' / (campaign_id + '_events') / phase_id
    for job in jobs:
        job.update(phase_id=phase_id, recording_id=recording, source_recording_id=recording,
                   event_directory=str(events), python_output=str(output),
                   event_bundle=str(events / f"dispatch-production-wait-shortfall-v2-run-{job['run_index']}-seed-{job['experiment_seed']}"))
    return dict(contract=PHASE_CONTRACT, campaign_id=campaign_id, phase_id=phase_id, kind=kind,
        recording_id=recording, python_output=str(output), event_directory=str(events),
        rpc_configuration_authoritative=True, require_local_alp_audit=False,
        reward=copy.deepcopy(template['reward']), environment_acceptance=copy.deepcopy(template['environment_acceptance']),
        execution=copy.deepcopy(execution),
        storage=dict(format='hdf5-v1',chunk_rows=64),
        capture_physical_events=kind == 'calibration',
        diagnostic_recording=dict(transition_jsonl=False, rpc_jsonl=False, rollout_snapshots=False),
        behavior=copy.deepcopy(behavior), jobs=jobs)


def _finish(directory, campaign_id, kind, phases, engineering_acceptance, extra):
    directory = Path(directory).resolve()
    require(not directory.exists(), 'prepared package is immutable; choose a new version')
    directory.mkdir(parents=True)
    jobs = [j for p in phases for j in p['jobs']]
    require([j['ordinal'] for j in jobs] == list(range(len(jobs))), 'ordinals must be frozen before writing')
    configs = []
    for phase in phases:
        path = directory / 'configs' / (phase['phase_id'] + '.json')
        _write(path, phase)
        configs.append(reference(path))
    master = directory / 'master_plan.csv'
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator='\n')
    writer.writeheader()
    writer.writerows({key: j[key] for key in COLUMNS} for j in jobs)
    master.open('xb').write(stream.getvalue().encode())
    templates = _reference_configs()
    from python.native_behavior import identity
    digests = {'P' + str(p): identity(c['behavior']) for p, c in templates.items()}
    helper = directory / 'Baseline1CampaignPlan.java'
    control = ROOT / 'runs' / campaign_id / 'control'
    helper.open('x').write(render_java(master_path=str(master), master_sha256=reference(master)['sha256'],
        control_directory=str(control), campaign_id=campaign_id,
        jobs=[{key: j[key] for key in COLUMNS} for j in jobs], behavior_digests=digests))
    campaign = dict(contract=CAMPAIGN_CONTRACT, campaign_id=campaign_id, kind=kind,
        approval_status='PENDING_SEPARATE_BUDGET_APPROVAL', legacy_reserved_shifts=11280,
        new_shift_cap=len(jobs), maximum_total_if_approved=11280 + extra.get('prior_baseline1_reserved_shifts', 0) + len(jobs),
        python_output=str(ROOT / 'runs' / campaign_id), control_directory=str(control),
        source_pins=source_pins(), engineering_acceptance=reference(engineering_acceptance),
        phase_configs=configs, jobs=jobs, master_plan=reference(master), java_helper=reference(helper),
        legacy_budget_snapshot=reference(ROOT / 'reports/stage8_seed1_collected_training_001/current_state.json'),
        **extra)
    campaign_path = directory / 'campaign.json'
    _write(campaign_path, campaign)
    request = dict(decision='PENDING', campaign_id=campaign_id, campaign_sha256=reference(campaign_path)['sha256'],
        kind=kind, max_new_shifts=len(jobs), legacy_reserved_shifts=11280,
        accounting='additional_to_h2_reserved',
        note='A request only; never pass this pending file as execution approval.')
    _write(directory / 'budget_request.json', request)
    manifest = dict(status='PREPARED_NOT_STARTED_NOT_APPROVED', campaign=reference(campaign_path),
        new_physical_shifts=len(jobs), roles={role: sum(j['role'] == role for j in jobs) for role in sorted({j['role'] for j in jobs})},
        configurations=[dict(recording_id=p['recording_id'], config=ref, python_output=p['python_output'],
            anylogic_event_directory=p['event_directory'], jobs=[dict(run_index=j['run_index'],
                experiment_seed=j['experiment_seed'], actor_sampling_seed=j['actor_sampling_seed']) for j in p['jobs']])
            for p, ref in zip(phases, configs)])
    _write(directory / 'launch_identity.json', manifest)
    _manual(directory, campaign, manifest)
    return manifest


def prepare_calibration(directory, *, engineering_acceptance, campaign_id, per_policy, first_run_index,
                        minimum_positive_cost_shifts):
    require(type(per_policy) is int and per_policy > 0, 'explicit calibration count required')
    require(type(minimum_positive_cost_shifts) is int and 0 < minimum_positive_cost_shifts <= 4 * per_policy,
            'explicit minimum positive-K count required')
    templates = _reference_configs()
    phases = []
    ordinal = 0
    for policy, template in templates.items():
        jobs = []
        expected = template['jobs'][0]['expected_rpc_config']
        for _ in range(per_policy):
            jobs.append(dict(ordinal=ordinal, actor_seed=2026110401, group='P' + str(policy), role='calibration',
                target_training_update=0, checkpoint_update=0, run_index=first_run_index + ordinal,
                experiment_seed=_seed(campaign_id, 'environment', ordinal),
                actor_sampling_seed=_seed(campaign_id, 'actor', ordinal),
                scene_id='existing_normal_runtime_v1', scene_group=campaign_id + f'-r{ordinal:04d}',
                realization_group_id=campaign_id + f'-r{ordinal:04d}', partition='train',
                runtime_scene_sha256=template['jobs'][0]['runtime_scene_sha256'],
                expected_rpc_config=copy.deepcopy(expected)))
            ordinal += 1
        phases.append(_phase(campaign_id, 'p' + str(policy), 'calibration', template, jobs,
            dict(host='127.0.0.1', port=50051, callback_timeout_seconds=template['callback_timeout_seconds'],
                 event_finalize_timeout_seconds=10), template['behavior']))
    return _finish(directory, campaign_id, 'calibration', phases, engineering_acceptance,
        dict(calibration=dict(expected_policy_counts={str(p): per_policy for p in templates},
             minimum_positive_cost_shifts=minimum_positive_cost_shifts,
             reference_change='New calibration set replaces inaccessible historical train140 for this version only; old design preserved.',
             offline_only=True, critic_pretraining=False)))


def prepare_trial(directory, *, engineering_acceptance, campaign_id, scale_path, beta, first_run_index,
                  actor_seed, batches, episodes_per_batch, evaluation_shifts, epochs):
    scale_path = Path(scale_path).resolve()
    scale = json.loads(scale_path.read_bytes())
    require(scale.get('admission_verified_here') is True and scale.get('calibration_campaign'),
            'strictly admitted new calibration scale is required before a trial can be frozen')
    require(type(beta) in (int, float) and beta in scale['beta_candidates'], 'choose one numeric new beta candidate explicitly')
    require(all(type(x) is int and x > 0 for x in (actor_seed, batches, episodes_per_batch, evaluation_shifts, epochs)),
            'explicit positive trial counts and seed required')
    require(batches >= 2, 'trial needs multiple complete batch updates and a separate monitoring point')
    native = _reference_configs()[0]
    old = json.loads((ROOT / 'change_plans/stage8_seed1_launch_001/formal_configs/actor_1_A.json').read_bytes())
    old_config = old['ppo_config']
    expected = old['shared_controls']['simulation_run_index_seed_list'][0]
    network = {k: old_config['network'][k] for k in ('hidden_sizes', 'activation', 'initialization')}
    old_ppo = old_config['ppo']
    ppo = {key: old_ppo[key] for key in ('minibatch_size', 'clip_range', 'actor_lr', 'critic_lr',
                                       'adam_eps', 'entropy_coef', 'max_grad_norm', 'actor_aggregation')}
    ppo.update(gamma=1, gae_lambda=old_ppo['gae_reward'], epochs=epochs, value_coef=old_ppo['reward_value_coef'],
               value_loss='mse', advantage_processing='none')
    learner = dict(scope='formal_online', formal_execution_authorized=False, network=network, ppo=ppo,
        collection=dict(episodes_per_batch=episodes_per_batch, batches=batches, seed=actor_seed),
        beta=float(beta), reward_contract=CONTRACT, shuffle_seed=_seed(campaign_id, 'shuffle', 0),
        sampling_seed=_seed(campaign_id, 'sampling', 0))
    phases, ordinal = [], 0
    schedule = [('t0', 't0', 0, evaluation_shifts)]
    for update in range(1, batches + 1):
        schedule.append((f'train_u{update:04d}', 'train', update, episodes_per_batch))
        if update == batches // 2:
            schedule.append((f'monitor_u{update:04d}', 'monitor', update, evaluation_shifts))
    schedule.append((f'final_u{batches:04d}', 'final', batches, evaluation_shifts))
    for phase_id, role, update, count in schedule:
        jobs = []
        for _ in range(count):
            jobs.append(dict(ordinal=ordinal, actor_seed=actor_seed, group='BASELINE', role=role,
                target_training_update=update, checkpoint_update=update - 1 if role == 'train' else update,
                run_index=first_run_index + ordinal, experiment_seed=_seed(campaign_id, 'environment', ordinal),
                actor_sampling_seed=_seed(campaign_id, 'actor', ordinal), scene_id=expected['scene_id'],
                scene_group=campaign_id + f'-r{ordinal:04d}', realization_group_id=campaign_id + f'-r{ordinal:04d}',
                partition='train' if role == 'train' else 'validation',
                runtime_scene_sha256=expected['runtime_scene_sha256'],
                expected_rpc_config=copy.deepcopy(expected['expected_rpc_config'])))
            ordinal += 1
        phases.append(_phase(campaign_id, phase_id, 'single_seed_trial', native, jobs,
            dict(host=old['execution']['host'], port=old['execution']['port'],
                 callback_timeout_seconds=old['execution']['callback_timeout_seconds'],
                 event_finalize_timeout_seconds=10)))
    from .gate import load_reference
    calibration_campaign = load_reference(scale['calibration_campaign'])
    require(calibration_campaign['kind'] == 'calibration'
            and scale.get('calibration_physical_shifts') == calibration_campaign['new_shift_cap'],
            'new calibration source and count binding differ')
    require(len(scale.get('dataset_manifests', [])) == calibration_campaign['new_shift_cap'],
            'complete calibration dataset inventory is required')
    for ref in scale['dataset_manifests']:
        require(reference(ref['path']) == ref, 'calibration input changed')
    return _finish(directory, campaign_id, 'single_seed_trial', phases, engineering_acceptance,
        dict(learner_config=learner, cpu_threads=old['cpu_threads'], calibration_scale=reference(scale_path),
             prior_baseline1_reserved_shifts=calibration_campaign['new_shift_cap'],
             trial_only=True, formal_beta_selected=False,
             interpretation='Single seed learning feasibility; not a formal multi-seed or final beta-selection result.'))


def _manual(directory, campaign, manifest):
    expressions = {'experimentRunIndex': 'runIndex', 'experimentSeed': 'environmentSeed',
        'boundaryRecordingId': 'recordingId', 'boundaryEventRoot': 'eventRoot',
        'currentDispatchPolicy': 'dispatchPolicy', 'offlineNativeCollection': 'offlineEnabled',
        'offlineBehaviorSha256': 'offlineSha256', 'enableBoundaryEventRecording': 'eventRecordingEnabled'}
    trial = campaign['kind'] == 'single_seed_trial'
    purpose = '单种子 PPO 试运行' if trial else '原生规则离线校准'
    flags = ('`currentDispatchPolicy=3`、`offlineNativeCollection=false`、`enableBoundaryEventRecording=false`'
             if trial else '`currentDispatchPolicy` 按阶段为 0/1/2/4、`offlineNativeCollection=true`、`enableBoundaryEventRecording=true`')
    lines = [f'# {purpose}启动卡', '',
        f"本包 `{campaign['campaign_id']}`，共 **{campaign['new_shift_cap']} 班**。预算状态：**待单独批准**；本卡不表示监听已启动。",
        '适用于已配置好 Main 启动屏障及 Freeform helper 表达式的 `DRL-PPO-baseline1.alp` 副本。原模型与历史运行记录保留。', '',
        '## 需要你手动做的操作', '',
        '1. 副本 → Additional Class → `Baseline1CampaignPlan`：将旧包的类全文替换为[本包 Java 全文](Baseline1CampaignPlan.java)，不要只改部分常量。',
        f"2. `RL_Training → Properties → Number of runs`：由上一包班数改为 **{campaign['new_shift_cap']}**；保持 Allow parallel evaluations=false、Use replications=false、index 从 0 起。",
        '3. Save/Build，通过后告诉 Codex“已 Build”。既有 Main 与 Freeform 表达式正确时无需重填。',
        '4. 待预算获批且 Codex 确认 **LISTENING + 首班 ready** 后，在 `RL_Training` 点击 **Run 一次**。', '',
        '## 修改效果与检查', '',
        '|位置|修改前 → 修改后|理由与影响|', '|---|---|---|',
        f"|Additional Class → Baseline1CampaignPlan|旧 helper 绑定上一包身份 → 本包 helper 绑定 `{campaign['campaign_id']}` 的逐班 run/seed 与配置 SHA|隔离新运行；启动屏障及独占 claim 防止重复班|",
        f"|RL_Training → Number of runs|上一包班数 → {campaign['new_shift_cap']}|严格对应本包预算与 master 行数|",
        f'|既有 Freeform helper 的返回值|上一包策略/记录开关 → {flags}|随新 helper 自动切换，无需另填策略或开关|', '',
        'HDF5 存储在 Python 接收 RPC 后完成，无需修改 AnyLogic 存储代码或安装 Java HDF5 库。',
        '保留 1519D、动作、生产目标、品位区间、物料 FIFO、同刻事件 LIFO 及现有奖励/场景；RPC 地址仍为 127.0.0.1:50051。',
        ('本次训练和独立评价均采用 RPC Actor；物理明细默认关闭，必要训练输入及目标写入 HDF5，逐 RPC/transition 文本日志和 rollout 快照关闭。'
         if trial else '本次各规则阶段显式采集物理事件用于等待/FIFO/四窗独立重放；训练输入写入 HDF5，逐 RPC/transition 文本日志和 rollout 快照关闭，不执行 PPO 更新。'),
        'Build 通过后，Codex 仍须核对首回调身份、RPC 配置与 schema；每班须真实 28800 秒正常终止、四窗及奖励账本通过。',
        '逐班屏障及训练/评价切换可暂留仿真时间 0。若报错或出现 STOP/UNKNOWN，保留所有记录并告知 Codex；不要删除 claim 或重复 Run。', '',
        '## 本次身份（仅供本包）', '',
        f"- campaign 字节 SHA256：`{manifest['campaign']['sha256']}`",
        f"- 统一预算：`{ROOT / 'runs/baseline1-budget-v1.json'}`；原 11280 班保留，本包新增 {campaign['new_shift_cap']} 班待批。",
        '- 完整 recording/config/目录及逐班 run_index/experiment_seed/actor_sampling_seed：[launch_identity.json](launch_identity.json)；主清单：[master_plan.csv](master_plan.csv)。', '',
        '|recording_id|配置字节 SHA256|Python 输出目录|AnyLogic 事件父目录|', '|---|---|---|---|']
    lines += [f"|`{c['recording_id']}`|`{c['config']['sha256']}`|`{c['python_output']}`|`{c['anylogic_event_directory']}`|"
              for c in manifest['configurations']]
    lines += ['', '<details>', '<summary>仅在旧副本缺少启动屏障或 helper 表达式时展开</summary>', '',
        'Main → Properties → Agent actions → On startup：第一段应为下方调用；若仍为 H2CampaignPlan，仅替换类名；若缺失，在原 startup 开头插入。后续初始化/物理/RNG 代码保留。', '',
        '```java', 'boundaryAcceptanceConfigSha256 = Baseline1CampaignPlan.awaitConfigSha256(',
        '    experimentRunIndex, experimentSeed, boundaryRecordingId,',
        '    boundaryEventRoot, boundaryInstrumentationRevision);', '```', '',
        'RL_Training → Properties → Parameters → Freeform：仅修复缺失或旧常量项，helper 调用不要加引号。', '',
        '|精确参数|完整表达式|', '|---|---|']
    lines += [f'|{key}|`Baseline1CampaignPlan.{method}((int) index)`|' for key, method in expressions.items()]
    lines += ['|boundaryAcceptanceConfigSha256|`"' + '0' * 64 + '"`（启动屏障置换真实配置 SHA）|',
        '|boundaryInstrumentationRevision|`"boundary-events-v1-native-rpc-runtime-scene-v1"`|', '',
        '保持 usePhysicalObservation=true、enableOnnxDispatch=true、obsDryRun=false、enableGradeActionReward=false、diagEvalEnabled=false、bcCollectEnabled=false、enableP0SnapshotAudit=false、acceptanceLongLoadEnabled=false、acceptanceBoundaryMode=0、enableWasteTaskLimit=true。', '',
        '修改理由：恢复现有 RPC/1519D 启动契约；影响仅为新 helper 提供身份及策略/记录开关。Save/Build 后交由 Codex 核对 RPC，不凭缺失参数或零奖励推断配置。', '',
        '</details>']
    (directory / 'anylogic_manual.md').open('x').write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cal = sub.add_parser('calibration')
    for name in ('directory', 'engineering-acceptance', 'campaign-id'):
        cal.add_argument('--' + name, required=True)
    for name in ('per-policy', 'first-run-index', 'minimum-positive-cost-shifts'):
        cal.add_argument('--' + name, required=True, type=int)
    trial = sub.add_parser('trial')
    for name in ('directory', 'engineering-acceptance', 'campaign-id', 'scale-path'):
        trial.add_argument('--' + name, required=True)
    trial.add_argument('--beta', type=float, required=True)
    for name in ('first-run-index', 'actor-seed', 'batches', 'episodes-per-batch', 'evaluation-shifts', 'epochs'):
        trial.add_argument('--' + name, type=int, required=True)
    options = vars(parser.parse_args())
    command = options.pop('command')
    result = prepare_calibration(**options) if command == 'calibration' else prepare_trial(**options)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
