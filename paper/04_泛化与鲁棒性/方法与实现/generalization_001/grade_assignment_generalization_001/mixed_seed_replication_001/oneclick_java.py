"""Route one AnyLogic parameter sweep through four sealed serial campaigns."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import indent

from replica_plan import CAP, PREPARED, REPLICAS, locations, parent
from business_graph_summary_campaign_v1.plan_io import load_reference, reference

INDEX_METHODS = {
    'runIndex': 'long', 'environmentSeed': 'long', 'recordingId': 'String',
    'eventRoot': 'String', 'materialEnabled': 'boolean', 'phaseId': 'String',
    'materialRoot': 'String', 'stageId': 'String', 'waitWeight': 'double',
    'actorSamplingSeed': 'long', 'dispatchPolicy': 'int', 'offlineEnabled': 'boolean',
    'offlineSha256': 'String', 'eventRecordingEnabled': 'boolean', 'freeSpeedFactorStd': 'double',
}
RUN_METHODS = {
    'scenarioId': ('String', '', ''),
    'loadIsOre': ('boolean', ', int site', ', site'),
    'loadGrade': ('double', ', int site', ', site'),
    'loadGradeHalfRange': ('double', ', int site', ', site'),
    'roadSpeedLimit': ('double', ', String from, String to, double nominal', ', from, to, nominal'),
    'awaitConfigSha256': ('String',
                         ', long seed, String recording, String eventRoot, String revision',
                         ', seed, recording, eventRoot, revision'),
}


def _method(name, return_type, parameters, selector, arguments):
    arms = '\n'.join(f'            case {index}: return Seed{replica[1:]}.{name}({arguments});'
                     for index, replica in enumerate(REPLICAS))
    return f'''    public static {return_type} {name}({parameters}) {{
        switch ({selector}) {{
{arms}
            default: throw new IllegalStateException("unreachable seed index");
        }}
    }}
'''


def _source(identities):
    run_routes = '\n'.join(
        f"        if (run >= {item['first']['run_index']}L && run < {item['first']['run_index'] + CAP}L) return {index};"
        for index, item in enumerate(identities))
    text = f'''public final class Baseline1CampaignPlan {{
    private static final int SHIFTS_PER_SEED = {CAP};
    private static final int SEEDS = {len(REPLICAS)};
    private Baseline1CampaignPlan() {{ }}

    private static int seedForIndex(int index) {{
        if (index < 0 || index >= jobCount()) throw new IllegalStateException("index outside four-seed plan");
        return index / SHIFTS_PER_SEED;
    }}

    private static int seedForRun(long run) {{
{run_routes}
        throw new IllegalStateException("run_index outside four-seed plan");
    }}

    public static int rpcPort() {{ return 50051; }}
    public static int jobCount() {{ return SHIFTS_PER_SEED * SEEDS; }}
    public static int frozenPlanJobCount() {{ return jobCount(); }}
    public static int resumeOffset() {{ return 0; }}
    public static boolean preparationOnly() {{ return false; }}
'''
    for name, return_type in INDEX_METHODS.items():
        text += _method(name, return_type, 'int index', 'seedForIndex(index)', 'index % SHIFTS_PER_SEED')
    for name, (return_type, parameters, arguments) in RUN_METHODS.items():
        text += _method(name, return_type, 'long run' + parameters, 'seedForRun(run)', 'run' + arguments)
    for item, replica in zip(identities, REPLICAS, strict=True):
        helper = item['java_helper']
        parent.require(reference(helper['path']) == helper, 'embedded seed helper byte SHA changed')
        nested = Path(helper['path']).read_text(encoding='utf-8')
        sentinel = 'public final class Baseline1CampaignPlan {'
        parent.require(nested.count(sentinel) == 1 and not any(line.startswith(('package ', 'import '))
                                                             for line in nested.splitlines()),
                       'seed helper requires a package-independent single class')
        nested = nested[nested.index(sentinel):].replace('public final class', 'private static final class', 1)
        nested = nested.replace('Baseline1CampaignPlan', f'Seed{replica[1:]}')
        text += '\n' + indent(nested, '    ')
    return text + '}\n'


def _check(helper, masters):
    probe = '''public final class CombinedProbe {
    private static void check(boolean condition) {
        if (!condition) throw new AssertionError("combined Java phase routing differs");
    }
    public static void main(String[] args) throws Exception {
        check(Baseline1CampaignPlan.rpcPort() == 50051 && Baseline1CampaignPlan.jobCount() == 6896);
        int index = 0;
        for (String master : args) {
            java.util.List<String> data = java.nio.file.Files.readAllLines(java.nio.file.Path.of(master));
            check(data.size() == 1725);
            for (String row : data.subList(1, data.size())) {
                String[] fields = row.split(",", -1);
                long run = Baseline1CampaignPlan.runIndex(index);
                check(run == Long.parseLong(fields[6])
                    && Baseline1CampaignPlan.environmentSeed(index) == Long.parseLong(fields[7])
                    && Baseline1CampaignPlan.actorSamplingSeed(index) == Long.parseLong(fields[8])
                    && Baseline1CampaignPlan.recordingId(index).equals(fields[5])
                    && Baseline1CampaignPlan.eventRoot(index).equals(fields[9])
                    && Baseline1CampaignPlan.materialEnabled(index)
                    && Baseline1CampaignPlan.dispatchPolicy(index) == 3
                    && !Baseline1CampaignPlan.offlineEnabled(index)
                    && Baseline1CampaignPlan.freeSpeedFactorStd(index) == 0.05
                    && Baseline1CampaignPlan.scenarioId(run) != null);
                index++;
            }
        }
        check(index == 6896);
        try {
            Baseline1CampaignPlan.runIndex(index);
            throw new AssertionError("out-of-range index was accepted");
        } catch (IllegalStateException expected) {
            check(expected.getMessage().contains("outside four-seed plan"));
        }
        try {
            Baseline1CampaignPlan.awaitConfigSha256(
                Baseline1CampaignPlan.runIndex(1724), 0L, "wrong", "wrong", "wrong");
            throw new AssertionError("mismatched first handoff identity was accepted");
        } catch (IllegalStateException expected) {
            check(expected.getMessage().contains("Main parameters differ"));
        }
    }
}
'''
    with TemporaryDirectory(prefix='bg128-combined-java-') as scratch:
        folder = Path(scratch)
        probe_path = folder / 'CombinedProbe.java'
        probe_path.write_text(probe, encoding='utf-8')
        subprocess.run(['javac', '-encoding', 'UTF-8', '-d', str(folder), str(helper), str(probe_path)], check=True)
        subprocess.run(['java', '-cp', str(folder), 'CombinedProbe', *masters], check=True)


def prepare_combined(index):
    identities = [load_reference(ref) for ref in index]
    parent.require([item['replica'] for item in identities] == list(REPLICAS)
                   and all(item['rpc_port'] == 50051 and item['physical_shifts'] == CAP for item in identities),
                   'one-click identities do not match serial seed order')
    helper = PREPARED / 'Baseline1CampaignPlan.java'
    with helper.open('x', encoding='utf-8') as stream:
        stream.write(_source(identities))
    masters = [str(Path(locations(replica)['suite_path']).with_name('master_plan.csv')) for replica in REPLICAS]
    _check(helper, masters)
    info = PREPARED / 'launch_identity.json'
    configs = dict(order=list(REPLICAS), physical_shifts=CAP * len(REPLICAS), shifts_per_seed=CAP,
                   training_shifts=4096, evaluation_shifts=2800, rpc_port=50051, free_speed_factor_std=.05,
                   first=identities[0]['first'], java_helper=reference(helper), launch_identities=index)
    parent.write(info, configs)
    return dict(helper=reference(helper), launch_identity=reference(info), checked_indices=CAP * len(REPLICAS),
                rejected_out_of_range_index=True, rejected_wrong_handoff_identity=True, simulation_started=False)
