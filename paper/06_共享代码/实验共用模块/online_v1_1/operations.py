"""Publish the exact manually started AnyLogic block after its listener is live.

This module creates instructions and immutable configuration/job snapshots only.
It neither opens a listener nor starts a simulation or consumes a budget slot.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _publish(path, data, *, replace=False):
    """Same-directory fsync and publish; immutable history uses exclusive link."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=path.parent, prefix='.operation-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
            temporary = None
        else:
            os.link(temporary, path)
        directory = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _reference(path, data):
    return dict(path=str(path), sha256=_sha(data))


@contextmanager
def _current_lock(directory):
    with (directory/'.current_operation.lock').open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _identity(config):
    controls = config.get('shared_controls', {})
    recording = config.get('recording_id') or controls.get('recording_id')
    output = config.get('python_output') or config.get('output') or controls.get('python_output')
    events = config.get('anylogic_event_directory') or controls.get('anylogic_event_directory')
    _require(isinstance(recording, str) and recording.strip(), 'operation recording_id required')
    _require(output and Path(output).is_absolute(), 'absolute Python output required')
    _require(events and Path(events).is_absolute(), 'absolute AnyLogic event directory required')
    return recording, str(Path(output).resolve()), str(Path(events).resolve())


def _java_array(values, *, kind):
    suffix = 'L' if kind == 'long' else ''
    return 'new ' + kind + '[]{' + ', '.join(str(v) + suffix for v in values) + '}'


def _manual(operation):
    quote = lambda value: json.dumps(value, ensure_ascii=False)
    expr = operation['java_expressions']
    n = operation['complete_shifts']
    return '\n'.join([
        '# 当前 AnyLogic 操作：监听已就绪', '',
        '**LISTENING_WAITING_ANYLOGIC**。只运行本页精确班数；本页不表示仿真已经开始。', '',
        f"Actor seed `{operation['actor_seed']}`；组 `{operation['group']}`；用途 `{operation['role']}`；监听 `{operation['listener_address']}`。", '',
        '| 精确位置 | 本批完整值 |', '|---|---|',
        f'| RL_Training（ParameterVariation）→ Number of runs | `{n}` |',
        f"| RL_Training → Properties → Parameters → Main.boundaryRecordingId → Freeform | `{quote(operation['recording_id'])}` |",
        f"| 同上 → Main.boundaryAcceptanceConfigSha256 | `{quote(operation['config_byte_sha256'])}` |",
        f"| 同上 → Main.boundaryEventRoot | `{quote(operation['anylogic_event_directory'])}` |", '',
        '实验参数覆盖 Main 同名默认值时以 RL_Training 的 Freeform 为准。沿用已通过的 policy3 RPC/1519D/正常场景设置，不改奖励、成本或模型函数。', '',
        'RL_Training → Properties → Parameters → experimentRunIndex → Freeform：', '',
        '```java', expr['experimentRunIndex'], '```', '',
        '同位置 → experimentSeed → Freeform：', '',
        '```java', expr['experimentSeed'], '```', '',
        '`index` 是原已核验手工清单及本地模型使用的零起点参数变化运行序号；本批数组每次 Run 从第一个元素开始。不要再叠加外层重复、并行运行或第二个参数组合。', '',
        f"- recording_id：`{operation['recording_id']}`",
        f"- 本批运行配置+active_jobs快照：`{operation['active_snapshot']['path']}`",
        f"- 快照原始字节 SHA256（填 boundaryAcceptanceConfigSha256）：`{operation['config_byte_sha256']}`",
        f"- resolved config 单独字节 SHA256：`{operation['resolved_configuration']['sha256']}`",
        f"- Python实际输出：`{operation['python_output']}`",
        f"- Python base output：`{operation['python_base_output']}`",
        f"- AnyLogic事件父目录：`{operation['anylogic_event_directory']}`",
        f"- 完整逐项清单：`{operation['active_jobs_reference']['path']}`", '',
        '保存/Build 后，只在本页仍是 current_operation 且对应监听就绪时 Run。首批共享t0为16班；正式训练每个采集批32班，监测16班，末轮盲测96班，均以本页实际数字为准。',
        '本批结束后等待 Python 更新/评价/保存并发布下一份 READY，再更换身份及数组；禁止一次 Run 128/1024 班跨过更新和监听切换。不要预建每班 event_bundle 末级目录，不覆盖或删除旧事件。',
        'actor_sampling_seed 由 Python 私有采样器使用，不填入 experimentSeed。以下数组供逐项核对：', '',
        '```json', json.dumps(dict(run_index=operation['run_indices'],
            experiment_seed=operation['experiment_seeds'], actor_sampling_seed=operation['actor_sampling_seeds']),
            ensure_ascii=False, indent=2), '```', '',
        '本操作页只含配置、身份、路径和种子，不含观测、请求、盲测预测或表现。Seed2/3仍预留；本页不会自动释放后续种子。', '',
    ])


def publish_operation(config, jobs, *, operation_directory, role, actor_seed, group, listener_address):
    """Call only after server.start succeeds; returns the immutable READY record.

    config_byte_sha256 identifies the exact resolved-config + active-jobs snapshot
    whose bytes are preserved in active_snapshot.path.  It is the value the user
    enters in Main.boundaryAcceptanceConfigSha256 for this physical block.
    """
    _require(isinstance(jobs, list) and jobs, 'explicit nonempty active jobs required')
    _require(role in ('train', 'monitor', 'monitor_t0_shared', 'final_blind'), 'unknown operation role')
    _require(type(actor_seed) is int and actor_seed > 0 and group in ('A', 'B', 'C', 'SHARED_T0'),
             'explicit operation Actor seed/group required')
    _require(isinstance(listener_address, str) and listener_address.strip(), 'listener address required')
    directory = Path(operation_directory)
    _require(directory.is_absolute(), 'absolute operation_directory required')
    directory = directory.resolve()
    recording, output, events = _identity(config)
    indices, seeds, sampling = [], [], []
    for job in jobs:
        for name in ('run_index', 'experiment_seed', 'actor_sampling_seed'):
            _require(type(job.get(name)) is int and job[name] > 0, 'positive integer job '+name+' required')
        _require(job['run_index'] <= 2147483647, 'run_index exceeds Java int array range')
        _require(job['experiment_seed'] <= 9223372036854775807 and
                 job['actor_sampling_seed'] <= 9223372036854775807, 'seed exceeds Java long range')
        _require(job.get('recording_id') == recording, 'active job recording_id differs')
        leaf = Path(job['event_bundle'])
        _require(leaf.is_absolute() and leaf.parent.resolve() == Path(events), 'active event identity misplaced')
        _require(not leaf.exists(), 'active event leaf already exists; do not repeat collection')
        indices.append(job['run_index']); seeds.append(job['experiment_seed']); sampling.append(job['actor_sampling_seed'])
    _require(len(set(indices)) == len(indices) and len(set(seeds)) == len(seeds), 'duplicate active job identity')
    config_bytes, jobs_bytes = _bytes(config), _bytes(jobs)
    snapshot_bytes = _bytes(dict(schema='stage8-resolved-operation-snapshot-v1',
        role=role, actor_seed=actor_seed, group=group, resolved_configuration=config, active_jobs=jobs))
    history = directory/'history'
    history.mkdir(parents=True, exist_ok=True)
    prefix = history/f'op_{indices[0]}'
    # A durable exclusive claim prevents any second publisher for the same block.
    with Path(str(prefix)+'.claim').open('xb') as stream:
        stream.write(_bytes(dict(first_run_index=indices[0], active_snapshot_sha256=_sha(snapshot_bytes))))
        stream.flush(); os.fsync(stream.fileno())
    config_path = Path(str(prefix)+'.resolved_config.json')
    jobs_path = Path(str(prefix)+'.active_jobs.json')
    snapshot_path = Path(str(prefix)+'.snapshot.json')
    _publish(config_path, config_bytes); _publish(jobs_path, jobs_bytes); _publish(snapshot_path, snapshot_bytes)
    index = '[(int) index]'
    operation = dict(schema='stage8-listening-operation-v1', status='LISTENING_WAITING_ANYLOGIC',
        created_at_utc=datetime.now(timezone.utc).isoformat(), role=role, actor_seed=actor_seed, group=group,
        recording_id=recording, python_output=output,
        python_base_output=str(Path(config.get('python_base_output') or output).resolve()),
        anylogic_event_directory=events,
        complete_shifts=len(jobs), listener_address=listener_address,
        run_indices=indices, experiment_seeds=seeds, actor_sampling_seeds=sampling,
        config_byte_sha256=_sha(snapshot_bytes), active_snapshot_sha256=_sha(snapshot_bytes),
        resolved_configuration=_reference(config_path, config_bytes),
        active_jobs_reference=_reference(jobs_path, jobs_bytes),
        active_snapshot=_reference(snapshot_path, snapshot_bytes),
        java_expressions=dict(experimentRunIndex=_java_array(indices, kind='int')+index,
            experimentSeed=_java_array(seeds, kind='long')+index),
        history_json=str(Path(str(prefix)+'.json')), history_markdown=str(Path(str(prefix)+'.md')))
    op_bytes, manual = _bytes(operation), _manual(operation).encode('utf-8')
    _publish(Path(operation['history_json']), op_bytes)
    _publish(Path(operation['history_markdown']), manual)
    with _current_lock(directory):
        _publish(directory/'current_operation.json', op_bytes, replace=True)
        _publish(directory/'current_operation.md', manual, replace=True)
    readme = directory/'README.md'
    if not readme.exists():
        text = ('# AnyLogic逐批操作入口\n\n当前操作见 [current_operation.md](current_operation.md)，'
            '机器身份见 [current_operation.json](current_operation.json)。只在 LISTENING_WAITING_ANYLOGIC 后按精确班数 Run。\n\n'
            '每批在 RL_Training（ParameterVariation）设置 Number of runs、Parameters → Main.boundaryRecordingId、'
            'Main.boundaryAcceptanceConfigSha256、Main.boundaryEventRoot、experimentRunIndex和experimentSeed。'
            '修改前为上一批身份/数组或未知；修改后完整值都在当前页。身份变更确保事件和配置可追溯；'
            '32班训练批结束后监听会切换并执行PPO，不能连续跨批Run。\n\n'
            'history/中的操作、resolved config、active jobs和快照不覆盖；'
            'current_operation只更新指向当前就绪任务。事件父目录由collector创建，末级目录由AnyLogic创建。'
            '默认不写观测、逐请求或表现。本入口不启动AnyLogic、不新增预算、不自动释放Seed2/3。\n')
        try:
            _publish(readme, text.encode('utf-8'))
        except FileExistsError:
            pass
    return operation


def close_operation(operation, *, operation_directory):
    """Retire only this current READY card, without changing immutable history."""
    directory = Path(operation_directory).resolve()
    path = directory/'current_operation.json'
    if not path.is_file():
        return False
    with _current_lock(directory):
        current = json.loads(path.read_text())
        if current.get('active_snapshot_sha256') != operation.get('active_snapshot_sha256'):
            return False
        if current.get('status') != 'LISTENING_WAITING_ANYLOGIC':
            return False
        current.update(status='NOT_LISTENING_WAITING_NEXT_READY',
                       listener_closed_at_utc=datetime.now(timezone.utc).isoformat())
        message = ('# 当前监听已关闭\n\n**NOT_LISTENING_WAITING_NEXT_READY**。现在不要在 AnyLogic 点击 Run。'
            'Python正在执行更新、评价、保存或任务交接；等待下一份 LISTENING READY 操作页，'
            '再按新页面的身份、数组和精确班数运行。\n\n'
            f"上一批 recording_id：`{current['recording_id']}`。"
            f"原始操作留存于 [{Path(current['history_markdown']).name}]({current['history_markdown']})。\n")
        _publish(path, _bytes(current), replace=True)
        _publish(directory/'current_operation.md', message.encode('utf-8'), replace=True)
    return True
