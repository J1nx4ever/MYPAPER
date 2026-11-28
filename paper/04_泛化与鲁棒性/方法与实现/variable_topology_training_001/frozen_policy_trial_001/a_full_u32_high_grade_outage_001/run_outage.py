"""Launch the sealed A outage shifts once and verify its physical results."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent),str(HERE.parent/'b_finetune_001')]
from b_finetune_001 import run_remaining as native_runner
import a_outage_campaign as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true', help='show physical and audit status')
    args = parser.parse_args()
    rows = list(campaign.frozen_index().values())
    count, touched = native_runner.cohort_state(rows, include_activation=False)
    report = campaign.REPORTS/'summary.md'
    print(f'AnyLogic 已封存 {count}/{len(rows)} 班；结果报告{"已生成" if report.is_file() else "待生成"}。', flush=True)
    if args.status or report.is_file():
        return
    root = campaign.PROJECT/'runs'/campaign.CAMPAIGN
    control = root/'controller'
    log = root/'logs'
    control.mkdir(parents=True, exist_ok=True)
    log.mkdir(exist_ok=True)
    with (control/'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('当前活动的控制器已经在运行。', flush=True)
            return
        count, touched = native_runner.cohort_state(rows, include_activation=False)
        if count != len(rows):
            spec, _, _ = campaign.load_case(rows[0]['case'])
            experiment = spec['experiment']
            if touched or (control/f'{experiment}.json').exists():
                raise RuntimeError('已有物理启动记录，请核对本次原始日志与原生回执。')
            native = campaign.witness.bind(native_runner.native, MODEL=Path(spec['model_path']),
                                           LOG=log, CONTROL=control, campaign=campaign)
            native([sys.executable,'-B','-u',str(HERE/'outage_service.py')], experiment, rows)
        if native_runner.cohort_state(rows, include_activation=False)[0] != len(rows):
            raise RuntimeError('策略已退出，仍有班次缺少完整回执。')
        subprocess.run([sys.executable,'-B','-u',str(HERE/'outage_results.py'),'--collect'],check=True)


if __name__ == '__main__':
    main()
