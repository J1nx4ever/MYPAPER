"""Run the recovery in the installed AnyLogic engine, using the original compiled nine-truck model."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
import xml.etree.ElementTree as ET

from recovery_java import FIRST, HERE, LAST, NAME, ORIGINAL, _source, digest, once


BUILD = Path.home() / '.AnyLogicUniversity/Workspace8.8/DRL-PPO-baseline1_BUILD'
MODEL = next(p for p in HERE.parents if p.name == 'lagrangian_ppo_v1') / 'anylogic/3d/DRL-PPO-baseline1'
PINNED = {
    'src.generated/fusion/RL_Training.java': '520cbad953517d79cd6d402527955e32e2cce630433411a0f6154772aa91cd23',
    'classes/fusion/RL_Training.class': '14a1d7ee1713e2270b608dc3afa6a57e7c217a6f9f61e56add47cd5d3623290d',
    'classes/fusion/Main.class': 'c2f18cd60ce0e63e3b8b6967f034e0483b7349e95aaaf77a75882045be5a7e13',
}
HOST = '''
package fusion;
import com.anylogic.engine.Engine;
import com.anylogic.engine.Experiment.State;
import com.anylogic.engine.gui.ExperimentHost;

public final class RecoveryHost {
    private static final class FactoryProbe extends RL_Training {
        @Override public Main createRoot(Engine engine) {
            System.out.println("ANYLOGIC_HEADLESS_CHECK_OK no physical model created");
            System.exit(0);
            throw new AssertionError();
        }
    }
    public static void main(String[] args) throws Exception {
        boolean check = args[0].equals("--check");
        RL_Training experiment = check ? new FactoryProbe() : new RL_Training();
        ExperimentHost host = new ExperimentHost(experiment);
        experiment.setup(host);
        int count = Baseline1CampaignPlan.jobCount();
        if (experiment.getMaximumIterations() != count || Baseline1CampaignPlan.runIndex(0) != 10049255L
            || Baseline1CampaignPlan.runIndex(count-1) != 10052896L || experiment.isUseReplications()) {
            throw new IllegalStateException("experiment count, range or replication differs");
        }
        System.out.printf("RL_Training mode=%s runs=%d first=%d last=%d%n", args[0], count,
            Baseline1CampaignPlan.runIndex(0), Baseline1CampaignPlan.runIndex(count-1));
        experiment.run();
        long started = System.nanoTime();
        while (true) {
            State state = experiment.getState();
            if (state == State.ERROR) throw new IllegalStateException("AnyLogic experiment entered ERROR");
            if (state == State.FINISHED) {
                int completed = experiment.getNumberOfCompletedIterations();
                if (check || completed != count) throw new IllegalStateException("unexpected completion: "+completed);
                System.out.println("ANYLOGIC_ALL_ITERATIONS_FINISHED " + completed);
                System.exit(0);
            }
            if (check && System.nanoTime() - started > 15_000_000_000L)
                throw new IllegalStateException("headless engine did not reach the root factory");
            Thread.sleep(check ? 100L : 1000L);
        }
    }
}
'''


def wait_for_last_report():
    """Keep the user service alive while the last Python controller seals the five-seed report."""
    suite = json.loads((ORIGINAL / 'gat1-s05/suite.json').read_bytes())
    output = Path(suite['python_output'])
    receipt = json.loads((output / 'launch/launch_receipt.json').read_bytes())
    process = Path('/proc') / str(receipt['pid']) / 'stat'
    while True:
        state = json.loads((output / 'run_status.json').read_bytes())
        if state['status'] == 'COMPLETE':
            summary = Path(suite['report_directory']).parent / 'paired_evaluation_summary.json'
            result = json.loads(summary.read_bytes())
            if (state['completed_shifts'] != 1724 or result['training_seed_schedules'] !=
                    ['s01', 's02', 's03', 's04', 's05'] or result['evaluation_slots_per_checkpoint'] != 700):
                raise ValueError('Final controller or five-seed report count differs')
            print(f'BG128_FIVE_SEEDS_COMPLETE {summary}', flush=True)
            return
        if state['status'] == 'STOP':
            raise RuntimeError(state.get('reason', 'last Python controller stopped'))
        try:
            fields = process.read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            fields = []
        if not fields or fields[0] == 'Z' or int(fields[19]) != receipt['process_start_ticks']:
            if json.loads((output / 'run_status.json').read_bytes())['status'] == 'COMPLETE':
                continue
            raise RuntimeError('Last controller exited before publishing its final report')
        time.sleep(2)


def run(check):
    helper = HERE / 'prepared_001' / NAME
    if helper.read_text() != _source(FIRST, HERE / 'recovery_bridge.py'):
        raise ValueError('Recovery helper or bridge differs from the prepared launch')
    for relative, expected in PINNED.items():
        if digest(BUILD / relative) != expected:
            raise ValueError(f'Original installed model bytes changed: {relative}')
    baseline_code = (BUILD / 'src.generated/fusion' / NAME).read_text()
    if (ORIGINAL / NAME).read_text() not in baseline_code:
        raise ValueError('Original generated project does not contain the frozen four-unit plan')
    libs = [Path(e.attrib['path']).resolve(strict=True)
            for e in ET.parse(BUILD / '.classpath').findall("./classpathentry[@kind='lib']")]
    engine = next(p for p in libs if p.name == 'com.anylogic.engine.jar')
    java = engine.parents[2] / 'jre/bin/java'
    if not java.is_file():
        raise FileNotFoundError(java)
    with TemporaryDirectory(prefix='bg128-rl-training-') as scratch:
        folder = Path(scratch)
        classes = folder / 'classes'
        shutil.copytree(BUILD / 'classes', classes)
        classpath = os.pathsep.join(map(str, [classes, *libs]))
        (folder / NAME).write_text('package fusion;\n' + helper.read_text())
        experiment = (BUILD / 'src.generated/fusion/RL_Training.java').read_text()
        (folder / 'RL_Training.java').write_text(once(experiment, 'return 6896;',
                                                     'return Baseline1CampaignPlan.jobCount();'))
        (folder / 'RecoveryHost.java').write_text(HOST)
        subprocess.run(['javac', '--release', '17', '-encoding', 'UTF-8', '-cp', classpath, '-d', str(classes),
                        str(folder / NAME), str(folder / 'RL_Training.java'), str(folder / 'RecoveryHost.java')],
                       check=True)
        subprocess.run([str(java), '-Xmx1024m', '-Djava.awt.headless=true', '-DparallelWorkersCount_xjal=1',
                        '-cp', classpath, 'fusion.RecoveryHost', '--check' if check else '--run'],
                       check=True, cwd=MODEL, timeout=30 if check else None)
        if not check:
            wait_for_last_report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check', action='store_true', help='stop at the engine root factory, before model creation')
    modes.add_argument('--run', action='store_true', help='run the frozen suffix of the original experiment')
    run(parser.parse_args().check)
