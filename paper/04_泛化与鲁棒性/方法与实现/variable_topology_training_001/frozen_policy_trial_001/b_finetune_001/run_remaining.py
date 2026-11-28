"""Start the complete B campaign once, or inspect its progress without starting a shift."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import os
import subprocess
import time

import campaign

HERE = Path(__file__).resolve().parent
LOG = campaign.CAMPAIGN_OUTPUT/'logs'
ANYLOGIC = Path('/home/jinx/apps/AnyLoigc/anylogic/anylogic')
MODEL = Path('/home/jinx/Models/DRL-PPO-baseline-topologydemo/DRL-PPO-baseline-topologydemo-b-finetune.alp')
CONTROL = campaign.CAMPAIGN_OUTPUT/'controller'
IDENTITY = CONTROL/'campaign.json'
REPORT = campaign.REPORTS/'finetune_report.md'
PYTHON = Path('/home/jinx/miniconda3/envs/anylogic-env/bin/python')


def python_command(script, *args):
    return [str(PYTHON), '-B', '-u', str(HERE/script), *args]


def run_offline(name, script, *args):
    print(f'Starting {name}',flush=True)
    with (LOG/f'{name}.log').open('ab') as output:
        output.write(f'\n{datetime.now(timezone.utc).isoformat()} {name} pid={os.getpid()}\n'.encode())
        output.flush()
        subprocess.run(python_command(script,*args),stdout=output,stderr=subprocess.STDOUT,check=True)
    print(f'Finished {name}',flush=True)


def native(service, experiment, rows):
    log_path=LOG/f'{experiment}-policy.log'
    with ExitStack() as stack:
        log=stack.enter_context(log_path.open('ab'))
        reader=stack.enter_context(log_path.open('rb'))
        reader.seek(0, os.SEEK_END)
        process=subprocess.Popen(service,stdout=log,stderr=subprocess.STDOUT)
        try:
            while b'Policy ready on' not in reader.read():
                if process.poll() is not None:
                    raise RuntimeError(f'{experiment} exited before accepting requests')
                time.sleep(1)
            print(f'Launching {experiment}',flush=True)
            env=dict(os.environ, DISPLAY=os.environ.get('DISPLAY') or ':1')
            al_log=stack.enter_context((LOG/f'{experiment}-anylogic.log').open('xb'))
            stamp(CONTROL/f'{experiment}.json', dict(campaign=campaign.CAMPAIGN, experiment=experiment,
                  controller_pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(),
                  runs=[{k: r[k] for k in ('case', 'run_index', 'recording_id', 'config_sha256')} for r in rows]))
            launcher=subprocess.Popen([str(ANYLOGIC),'-r',str(MODEL),experiment],
                                      cwd=ANYLOGIC.parent,env=env,stdout=al_log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if launcher.poll() not in (None,0):
                    raise RuntimeError(f'AnyLogic launcher failed for {experiment}')
                time.sleep(2)
            if process.returncode:
                raise RuntimeError(f'Policy failed for {experiment}, see {log_path}')
            print(f'Sealed {experiment}',flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def stamp(path, data):
    """Record intent before starting a native process. Existing claims are immutable."""
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def groups():
    cases = campaign.frozen_index()
    return {name: [r for r in cases.values() if r['experiment'] == name] for name in campaign.EXPERIMENTS}


def cohort_state(rows, *, include_activation=True):
    sealed = 0
    touched = False
    for row in rows:
        out = Path(row['python_output_directory'])
        events = Path(row['event_directory'])
        touched |= out.exists() or events.exists() or (include_activation and Path(row['activation_path']).exists())
        a = events/'events_manifest.json'
        b = out/'inference_manifest.json'
        buffer = out/'ppo_buffer_receipt.json'
        if not (a.is_file() and b.is_file() and (row['phase'] != 'train' or buffer.is_file())):
            continue
        try:
            native_seal, actor_seal = (json.loads(p.read_bytes()) for p in (a, b))
        except (FileNotFoundError, json.JSONDecodeError):
            continue  # A reader can see the manifest while the native writer flushes it.
        if not (native_seal.get('complete') and actor_seal.get('complete')):
            continue
        if not (native_seal.get('terminated') and native_seal.get('truncated') is False and
                all(s.get(k) == row[k] for s in (native_seal, actor_seal)
                    for k in ('run_index', 'recording_id', 'config_sha256'))):
            raise ValueError(f"Completed manifests differ from frozen case {row['case']}")
        sealed += 1
    return sealed, touched


def finished():
    return all((campaign.REPORTS/name).is_file() for name in
               ('summary.json', 'finetune_report.md', 'finetune_curves.svg', 'finetune_curves.png'))


def progress(cohorts):
    counts = {k: cohort_state(rows) for k, rows in cohorts.items()}
    stages = '  '.join(f'{k.removeprefix("Topology_B_")} {n}/{len(cohorts[k])}' for k, (n, _) in counts.items())
    updates = sum((campaign.MODEL_OUTPUT/f'u{i:04d}'/'metrics.json').is_file() for i in range(1, 5))
    selected = campaign.CAMPAIGN_OUTPUT/'selection.json'
    selection = f"u{json.loads(selected.read_bytes())['selected_update']}" if selected.is_file() else '待选'
    return counts, f'{stages}  |  更新 {updates}/4  |  入选 {selection}'


def status(cohorts, follow):
    previous = None
    while True:
        _, message = progress(cohorts)
        if message != previous:
            print(message, flush=True)
            previous = message
        if finished():
            print(f'活动已完成，报告位于 {REPORT}', flush=True)
            return
        if not follow:
            return
        time.sleep(5)


def native_stage(service, experiment, rows):
    claim = CONTROL/f'{experiment}.json'
    sealed, touched = cohort_state(rows, include_activation=False)
    if sealed == len(rows):
        return
    if claim.exists() or touched:
        raise RuntimeError(f'{experiment} 已有启动记录及 {sealed}/{len(rows)} 份封存回执，'
                           '请用 --follow 观察原进程或检查原日志。')
    native(service, experiment, rows)
    if cohort_state(rows)[0] != len(rows):
        raise RuntimeError(f'{experiment} 策略进程已退出，但仍有班次缺少封存回执。')


def pipeline(cohorts):
    for batch in range(1,5):
        experiment = f'Topology_B_Train{batch}'
        native_stage(python_command('train_service.py','serve','--batch',str(batch)),experiment,cohorts[experiment])
        run_offline(f'update{batch}','train_service.py','update','--batch',str(batch))
    experiment = 'Topology_B_Validate80'
    native_stage(python_command('eval_service.py','validation'),experiment,cohorts[experiment])
    run_offline('selection','results.py','select')
    experiment = 'Topology_B_Test50'
    native_stage(python_command('eval_service.py','test'),experiment,cohorts[experiment])
    run_offline('final-audit','results.py','final')


def legacy_activity(counts):
    return not IDENTITY.exists() and (any(touched for _, touched in counts.values())
                                      or any(LOG.glob('*.log')))


def main():
    parser = argparse.ArgumentParser(description='B 图训练、选模和复测的一次启动入口')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--status', action='store_true', help='只读显示已封存班次、更新和选模进度')
    mode.add_argument('--follow', action='store_true', help='只读跟踪当前活动直至最终报告写入')
    args = parser.parse_args()
    cohorts = groups()
    if args.status or args.follow or finished():
        status(cohorts, args.follow)
        return
    counts, _ = progress(cohorts)
    if legacy_activity(counts):
        print('检测到本活动已有独立启动记录。使用 --follow 可持续查看。')
        status(cohorts, False)
        return
    CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL/'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('本活动的控制器正在运行。使用 --follow 可持续查看。')
            status(cohorts, False)
            return
        # Recheck after acquiring the lock, including completion by a previous owner.
        counts, _ = progress(cohorts)
        if finished() or legacy_activity(counts):
            status(cohorts, False)
            return
        identity = dict(campaign=campaign.CAMPAIGN,
                        index_sha256=campaign.OVERVIEW.with_suffix('.sha256').read_text().strip())
        if IDENTITY.exists():
            if json.loads(IDENTITY.read_bytes()) != identity:
                raise ValueError(f'活动身份不匹配 {IDENTITY}')
        else:
            stamp(IDENTITY, identity)
        if not (os.access(ANYLOGIC, os.X_OK) and MODEL.is_file()):
            raise FileNotFoundError(f'请核对 AnyLogic 可执行文件 {ANYLOGIC} 及冻结模型 {MODEL}')
        LOG.mkdir(exist_ok=True)
        pipeline(cohorts)
        status(cohorts, False)


if __name__ == '__main__':
    main()
