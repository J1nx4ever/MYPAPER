"""Read a selected baseline2 ALP and type-check the manual's hooks in a temporary Java source tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

PLAN = Path(__file__).resolve().parent
BASE_SHA = 'eb8c46b7f7410daeb9997bf18ac0aa07e53ca5e5a652e0d79e15ceb425baae26'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def named(tree, tag, name):
    node, = [node for node in tree.iter(tag) if node.findtext('Name') == name]
    return node


def replace_once(body, before, after):
    if body.count(before) != 1:
        raise ValueError('Expected one occurrence of model hook: ' + before)
    return body.replace(before, after)


def compiled_bodies(model, manual):
    blocks = re.findall(r'```java\n(.*?)\n```', manual, re.S)
    if len(blocks) != 12:
        raise ValueError('Manual hook list changed; review the offline compiler')
    (fault_action, init_state, arm_fault, roster_guard, disabled, gate, reject, speed, delay,
     after_dump, slice_audit, rpc) = blocks
    def body(name):
        return named(model, 'Function', name).findtext('Body')

    def action(name):
        return named(model, 'Event', name).findtext('Action')

    def load(path):
        return (PLAN / path).read_text()

    m = named(model, 'ActiveObjectClass', 'Main')
    startup = replace_once(m.findtext('StartupCode'), 'if (!Double.isFinite(targetOreTonsShift)',
                           init_state + '\nif (!Double.isFinite(targetOreTonsShift)')
    startup = replace_once(startup, 'init1();', 'init1();\n' + arm_fault)
    old_guard = '''if (numTrucks != 9 || (fleetDisabledTruckMask != 0 && fleetDisabledTruckMask != 292)) {
    throw new IllegalStateException("Fleet outage requires nine truck slots and mask 0 or 292");
}'''
    init1 = replace_once(body('init1'), old_guard, roster_guard)
    decide = replace_once(body('selectNextLoadSite'), 'if (isFleetTruckDisabled(t.truckId)) {', reject)
    rewards = body('buildRewardConfigComponents')
    old_config, = re.findall(r'if \(fleetDisabledTruckMask != 0\) \{.*?\n}', rewards, re.S)
    rewards = replace_once(rewards, old_config, rpc)
    advance = replace_once(body('advanceCrusherTo'), 'boundaryAuditCursorSec = boundaryAuditEndSec;',
                           slice_audit + '\n boundaryAuditCursorSec = boundaryAuditEndSec;')
    exit_block = named(model, 'EmbeddedObject', 'exitDump')
    exit_code, = [x.findtext('Value/Code') for x in exit_block.findall('Parameters/Parameter')
                  if x.findtext('Name') == 'onExit']
    exit_code = replace_once(exit_code, 't.onDumpingFinished(this);\nt.send("restart", t);', after_dump)
    before = named(model, 'StatechartElement', 'NeedDispatchToLoadSite').findtext('Properties/EntryAction')
    truck = load('Truck.additional.java.txt')
    if ''.join(before.split()) not in ''.join(truck.split()):
        raise ValueError('Original trip commitment code differs from Truck.fleetDispatchNow()')
    speed_filters = []
    for event in ('evApplySpeedActions', 'evUpdateSpeedObservation'):
        loop = 'for (Truck t : trucks) {'
        code = replace_once(action(event), loop, loop + '\n if (isFleetTruckDisabled(t.truckId)) continue;')
        speed_filters.append(f'void offline{event}() {{ {code} }}')
    crc = named(model, 'StatechartElement', '收集数据并重置')
    restart = named(model, 'StatechartElement', 'transition7').find('Properties')
    trip_target = crc.findtext('Id')
    if (restart.get('Target') != trip_target or restart.findtext('EqualsExpression') != '"restart"'
            or 'main.completedTrips++;' not in crc.findtext('Properties/EntryAction')):
        raise ValueError('Original final-trip completion edge changed')
    return {
        'Main': '\n'.join([load('Main.additional.java.txt'), *speed_filters,
                          'EventTimeout evFleetFault; int fleetFaultTruckId;',
                          'double fleetFaultTriggerSeconds, fleetOreTarget, fleetWasteTarget;',
                          f'void offlineStartup() {{ {startup} }}', f'void offlineInit1() {{ {init1} }}',
                          f'void offlineFault() {{ {fault_action} }}', f'void offlineSpeed() {{ {speed} }}',
                          f'boolean offlineDisabled(int truckId) {{ {disabled} }}',
                          f'int offlineSelect(Truck t) {{ {decide} }}',
                          f'java.util.Map<String,Double> offlineConfig() {{ {rewards} }}']),
        'Truck': truck + f'\nvoid offlineReady() {{ {gate} }}',
        'DumpSite': '\n'.join([f'double offlineDelay(Agent agent) {{ return {delay}; }}',
                               f'void offlineExit(Agent agent) {{ {exit_code} }}',
                               f'void offlineCrusher(double nowSec) {{ {advance} }}'])}


def audit(alp, ide):
    if digest(alp) != BASE_SHA:
        raise ValueError('The selected ALP bytes differ from the documented baseline2 revision.')
    model = ET.parse(alp)
    additions = compiled_bodies(model, (PLAN/'manual_anylogic.md').read_text())
    cp_file = ide / '.classpath'
    libraries = [Path(n.get('path')) for n in ET.parse(cp_file).getroot() if n.get('kind') == 'lib']
    sources = sorted((ide/'src.generated/fusion').glob('*.java'))
    if not libraries or not all(path.is_file() for path in libraries) or not set(additions) <= {x.stem for x in sources}:
        raise FileNotFoundError('Missing local SDK classpath or generated baseline2 IDE types')
    cp = os.pathsep.join(str(path) for path in libraries)
    def cmd(args):
        proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f'{args[0]} failed with code {proc.returncode}:\n{proc.stdout}')
        return proc.stdout.strip()

    with TemporaryDirectory(prefix='fleet-outage-java-') as temporary:
        dest = Path(temporary)
        paths = []
        for source in sources:
            content = source.read_text()
            if source.stem in additions:
                brace = content.rindex('}')
                content = content[:brace] + '\n' + additions[source.stem] + '\n' + content[brace:]
            copy = dest/source.name
            copy.write_text(content)
            paths.append(str(copy))
        for name in ('FleetResilienceState','FleetResilienceChecks'):
            copy = dest/(name+'.java')
            copy.write_text('package fusion;\n' + (PLAN/(name+'.java.txt')).read_text())
            paths.append(str(copy))
        javac = cmd(['javac','-version'])
        cmd(['javac','-encoding','UTF-8','-cp',cp,'-d',temporary,*paths])
        result = cmd(['java','-cp',temporary,'fusion.FleetResilienceChecks'])
    if digest(alp) != BASE_SHA:
        raise ValueError('Selected ALP changed during the offline check')
    e = named(model, 'ParamVariationExperiment', 'RL_Training')
    return dict(result='PASS', mode='offline Java type-check and synthetic event order test',
                model=str(alp), model_sha256_before_after=BASE_SHA, javac=javac, ide_project=str(ide),
                sdk_classpath_sha256=digest(cp_file), java_type_files=len(sources), sdk_libraries=len(libraries),
                code_sha256={k: digest(PLAN/k) for k in ('Main.additional.java.txt','Truck.additional.java.txt',
                             'FleetResilienceState.java.txt','FleetResilienceChecks.java.txt','manual_anylogic.md')},
                integration='Original hook bodies and manual replacements compiled against generated IDE model types',
                baseline_experiment=dict(number_of_runs=e.findtext('NumberOfRuns'),
                    simultaneous=e.findtext('SelectionModeForSimultaneousEvents'),
                    parallel=e.findtext('AllowParallelEvaluations'), freeform=e.findtext('UseFreeformParameters'),
                    replication=e.findtext('ReplicationsProperties/UseReplication'),
                    simulation_cutoff_seconds=e.findtext('ModelTimeProperties/FinalTime')),
                original_final_trip_edge='restart -> 收集数据并重置 -> completedTrips++', fixture_result=result,
                physical_shifts=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alp', type=Path, required=True, help='source file selected for offline auditing')
    parser.add_argument('--ide-project', type=Path, required=True, help='matching AnyLogic IDE workspace project')
    parser.add_argument('--report', type=Path, help='optional new JSON evidence file, created only after success')
    args = parser.parse_args()
    record = audit(args.alp.resolve(strict=True), args.ide_project.resolve(strict=True))
    result = json.dumps(record, ensure_ascii=False, indent=2) + '\n'
    if args.report:
        if args.report.suffix.lower() != '.json':
            parser.error('--report must be a new JSON file')
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as output:
            output.write(result)
    print(result, end='')
