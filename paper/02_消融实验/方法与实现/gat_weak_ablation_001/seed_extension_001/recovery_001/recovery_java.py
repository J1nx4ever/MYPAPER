"""Expose the unclaimed suffix of the sealed 6,896-run Java sweep."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'prepared_001'
NAME = 'Baseline1CampaignPlan.java'
FIRST = 10049255
ORIGINAL_FIRST, LAST = 10046001, 10052896
UNITS = ('gat1-s04', 'full-s05', 'gat1-s05')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Unexpected source layout for {old!r}')
    return source.replace(old, new)


def _source(first_run, runner):
    if first_run != FIRST:
        raise ValueError('Recovery requires the admitted prefix ending at run 10049254')
    launch = json.loads((ORIGINAL / 'launch_identity.json').read_bytes())
    original = ORIGINAL / NAME
    if launch['java_helper'] != {'path': str(original), 'sha256': digest(original)}:
        raise ValueError('Original combined helper differs from its frozen launch manifest')
    source = original.read_text()
    front, units = source.split('    private static final class Unit0 {', 1)
    offset = first_run - ORIGINAL_FIRST
    count = LAST - first_run + 1
    front = once(front, 'return 6896;', f'return {count};')
    front = once(front, 'frozenPlanJobCount() { return jobCount(); }',
                 'frozenPlanJobCount() { return 6896; }')
    front = once(front, 'resumeOffset() { return 0; }',
                 f'resumeOffset() {{ return {offset}; }}')
    front = once(front, 'return index / PER_UNIT;', 'return (index + resumeOffset()) / PER_UNIT;')
    front = once(front, 'private static int forRun(long run) {',
                 'private static int forRun(long run) {\n'
                 f'        if (run < {first_run}L) throw new IllegalStateException("run before recovery");')
    if front.count('(index % PER_UNIT)') != 60:
        raise ValueError('Unexpected original combined Freeform delegation')
    front = front.replace('(index % PER_UNIT)', '((index + resumeOffset()) % PER_UNIT)')
    full_s04, units = units.split('    private static final class Unit1 {', 1)
    gat_s04, s05 = units.split('    private static final class Unit2 {', 1)
    original_bridge = HERE.parent / 'extension_run.py'
    gat_s04 = once(gat_s04, str(original_bridge), str(runner))
    gat_s04 = once(gat_s04, digest(original_bridge), digest(runner))
    gat_s04 = once(gat_s04, 'row.ordinal == 0, "one-click startup requires index 0; recovery must be reviewed"',
                   f'row.run == {first_run}L && row.ordinal == {offset % 1724}, '
                   '"first recovery callback differs from admitted prefix"')
    return (front + '    private static final class Unit0 {' + full_s04
            + '    private static final class Unit1 {' + gat_s04
            + '    private static final class Unit2 {' + s05)


def generate_java(prepared, first_run=FIRST, runner=None):
    destination = Path(prepared) / NAME
    runner = Path(runner or HERE / 'recovery_bridge.py').resolve(strict=True)
    content = _source(first_run, runner).encode()
    if destination.exists():
        if destination.read_bytes() != content:
            raise FileExistsError(f'Refusing to replace a different helper: {destination}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(content)
    return destination


PROBE = '''
public final class RecoveryProbe {
    private static void check(boolean condition) {
        if (!condition) throw new AssertionError("recovery route differs");
    }
    public static void main(String[] args) throws Exception {
        int first = Integer.parseInt(args[0]), index = 0;
        check(Baseline1CampaignPlan.rpcPort() == 50051);
        check(Baseline1CampaignPlan.resumeOffset() == first - 10046001);
        check(Baseline1CampaignPlan.frozenPlanJobCount() == 6896);
        for (String line : java.nio.file.Files.readAllLines(java.nio.file.Path.of(args[1]))) {
            String[] c = line.split(",", -1);
            long run = Baseline1CampaignPlan.runIndex(index);
            check(run == Long.parseLong(c[0]) && run == first + index);
            check(Baseline1CampaignPlan.environmentSeed(index) == Long.parseLong(c[1]));
            check(Baseline1CampaignPlan.actorSamplingSeed(index) == Long.parseLong(c[2]));
            check(Baseline1CampaignPlan.recordingId(index).equals(c[3]));
            check(Baseline1CampaignPlan.eventRoot(index).equals(c[4]));
            check(Baseline1CampaignPlan.materialRoot(index).equals(c[5]));
            check(Baseline1CampaignPlan.scenarioId(run).equals(c[6]));
            for (int site = 0; site < 4; ++site) {
                check(Baseline1CampaignPlan.loadIsOre(run, site) == c[7+site].equals("1"));
                check(Baseline1CampaignPlan.loadGrade(run, site) == Double.parseDouble(c[11+site]));
                check(Baseline1CampaignPlan.loadGradeHalfRange(run, site) == Double.parseDouble(c[15+site]));
            }
            check(Baseline1CampaignPlan.roadSpeedLimit(run, "intersection1", "Crusher", 1.0)
                == Double.parseDouble(c[19]));
            check(Baseline1CampaignPlan.dispatchPolicy(index) == 3 && Baseline1CampaignPlan.materialEnabled(index));
            check(!Baseline1CampaignPlan.offlineEnabled(index) && !Baseline1CampaignPlan.eventRecordingEnabled(index));
            check(Baseline1CampaignPlan.freeSpeedFactorStd(index) == .05);
            ++index;
        }
        check(index == 10052896 - first + 1 && index == Baseline1CampaignPlan.jobCount());
        for (int invalid : new int[]{-1, index}) {
            try { Baseline1CampaignPlan.runIndex(invalid); throw new AssertionError("bad index"); }
            catch (IllegalStateException expected) {}
        }
        for (long invalid : new long[]{first-1, first+index}) {
            try { Baseline1CampaignPlan.scenarioId(invalid); throw new AssertionError("bad run"); }
            catch (IllegalStateException expected) {}
        }
    }
}
'''


def verify_java(prepared, first_run=FIRST):
    helper = Path(prepared) / NAME
    with TemporaryDirectory(prefix='bg128-depth-recovery-java-') as temp:
        folder = Path(temp)
        (folder / 'RecoveryProbe.java').write_text(PROBE)
        inventory = folder / 'expected.csv'
        with inventory.open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            for unit in UNITS:
                directory = ORIGINAL / unit
                with ((directory / 'master_plan.csv').open() as master,
                      (directory / 'scenarios.csv').open() as scenarios,
                      (directory / 'identities.csv').open() as identities):
                    for row, scene, identity in zip(csv.DictReader(master), csv.DictReader(scenarios),
                                                    csv.DictReader(identities), strict=True):
                        if row['run_index'] != scene['run_index'] or row['run_index'] != identity['run_index']:
                            raise ValueError('Original run tables disagree')
                        if int(row['run_index']) >= first_run:
                            writer.writerow([row[k] for k in ('run_index', 'experiment_seed', 'actor_sampling_seed',
                                                              'recording_id', 'event_directory')]
                                            + [identity['material_event_directory'], *list(scene.values())[1:]])
        subprocess.run(['javac', '-encoding', 'UTF-8', '-d', temp, str(helper), str(folder / 'RecoveryProbe.java')],
                       check=True)
        subprocess.run(['java', '-cp', temp, 'RecoveryProbe', str(first_run), str(inventory)], check=True)
    return {'first_run': first_run, 'last_run': LAST, 'verified_slots': LAST - first_run + 1,
            'sha256': digest(helper)}
