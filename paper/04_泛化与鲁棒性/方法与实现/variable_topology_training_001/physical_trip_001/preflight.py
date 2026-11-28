"""Compile the hooked model with its SDK and test native Main startup at t=0."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

from prepare_trip_alp import replace

HERE = Path(__file__).resolve().parent
JAVA = Path('/home/jinx/apps/AnyLoigc/anylogic/jre/bin/java')
BASE = Path('/home/jinx/.AnyLogicUniversity/Workspace8.8/DRL-PPO-baseline-topologydemo-training-topology_BUILD')
WIRING_SMOKE = """
package fusion;
import com.anylogic.engine.Engine;
import static com.anylogic.engine.TimeUnits.SECOND;

public class TopologyTripStartupProbe {
    public static void main(String[] args) {
        for (int i = 1; i <= 2; i++) {
            Engine engine = new Engine();
            engine.setTimeUnit(SECOND);
            Main main = new Main(engine, null, null);
            main.topologyTripCase = i;
            if (!engine.start(main) || engine.getState() != Engine.PAUSED || engine.time() != 0.0
                    || main.trucks.size() != 9 || main.loadSites.size() != 4 || main.dumpSites.size() != 2
                    || Integer.bitCount(main.fleetResilience.openingMask) != (i == 1 ? 9 : 7)
                    || main.segmentMap.size() != (i == 1 ? 12 : 16)) {
                throw new AssertionError("Physical world startup did not match the case");
            }
            main.topologyTripSmoke.close();
            System.out.printf("Native Main t=0 case %d: 9 physical slots, %d in service, %d directed roads%n",
                    i, Integer.bitCount(main.fleetResilience.openingMask), main.segmentMap.size());
        }
    }
}
"""


def patch_generated_main(java, model):
    startup = next(item.findtext('StartupCode') for item in model.findall('./ActiveObjectClasses/ActiveObjectClass')
                   if item.findtext('Name') == 'Main')
    stop = startup.index('boundaryAcceptanceConfigSha256 =')
    fragment = startup[:stop]
    java = replace(java, 'FleetResilienceState fleetResilience;',
                   'FleetResilienceState fleetResilience;\nTopologyTripSmoke topologyTripSmoke;\nint topologyTripCase;')
    java = replace(java, 'boundaryAcceptanceConfigSha256 = Baseline1CampaignPlan.awaitConfigSha256(',
                   fragment + 'boundaryAcceptanceConfigSha256 = Baseline1CampaignPlan.awaitConfigSha256(')
    java = replace(java, 'int  intDestination = 0;',
                   'if (topologyTripSmoke != null) return topologyTripSmoke.modelNodeNumber(destination);\n'
                   'int  intDestination = 0;')
    java = replace(java, 'EventTimeout self = _e;\n\nfor (Truck t : trucks) {\n    t.updateBasicSpeedState();',
                   'EventTimeout self = _e;\n\n'
                   'if (topologyTripSmoke != null) { topologyTripSmoke.sample(); return; }\n'
                   'for (Truck t : trucks) {\n    t.updateBasicSpeedState();')
    return java


def patch_generated_truck(java):
    java = replace(java, 'targetLoadSiteId = main.selectNextLoadSite(this);',
                   'targetLoadSiteId = main.topologyTripSmoke == null\n'
                   '        ? main.selectNextLoadSite(this) : main.topologyTripSmoke.dispatch(this);')
    for sites, field, func in (('Load', 'targetLoadSiteId', 'loadNode'), ('Dump', 'targetDumpSiteId', 'dumpNode')):
        spaces = 8 if sites == 'Load' else 4
        legacy = f"main.collection{sites}Sites\n{' ' * spaces}.get({field}).getName()"
        java = replace(java, legacy, f'(main.topologyTripSmoke == null ? {legacy}'
                       f' : main.topologyTripSmoke.{func}({field}))')
    java = replace(java, 'moveTo(node);',
                   'if (main.topologyTripSmoke != null) main.topologyTripSmoke.departing(this, node);\n'
                   'moveTo(node);', count=2)
    for state in ('MoveSegment', 'MoveSegment1'):
        java = replace(java, f'exitState( {state}, self, true );\n          enterState( ArriveSegmentEnd',
                       f'exitState( {state}, self, true );\n'
                       '      if (main.topologyTripSmoke != null) main.topologyTripSmoke.arrived(this);\n'
                       '          enterState( ArriveSegmentEnd')
    java = replace(java, '    main.completedTrips++;\n} ',
                   '    main.completedTrips++;\n}\n'
                   'if (main.topologyTripSmoke != null) main.topologyTripSmoke.finish(this);')
    return replace(java, 'if ( _t == transition15 ) return \ntrue \n;',
                   'if ( _t == transition15 ) return main.topologyTripSmoke == null;')


def scratch_config(source, work):
    config = (HERE / 'configs' / source).read_bytes()
    lines = config.decode().splitlines()
    scratch = work / 'configs'
    scratch.mkdir(exist_ok=True)
    fields = {'event_directory': str(work / ('receipt_' + source)), 'run_index': '0',
              'recording_id': 'startup-only-scratch'}
    updated = []
    for line in lines:
        key, value = line.split('=', 1)
        updated.append(f'{key}={fields.get(key, value)}')
    data = ('\n'.join(updated) + '\n').encode()
    (scratch / source).write_bytes(data)
    return sha256(config).hexdigest(), sha256(data).hexdigest()


def run(candidate, base):
    model = ET.parse(candidate).getroot().find('Model')
    libs = ET.parse(base / '.classpath').getroot().findall("./classpathentry[@kind='lib']")
    cp = os.pathsep.join(str(entry.attrib['path']) for entry in libs)
    with TemporaryDirectory(prefix='faith-physical-trip-preflight-') as directory:
        work = Path(directory)
        src = work / 'src'
        shutil.copytree(base / 'src.generated', src)
        main = src / 'fusion' / 'Main.java'
        truck = src / 'fusion' / 'Truck.java'
        main.write_text(patch_generated_main(main.read_text(), model), encoding='utf-8')
        truck.write_text(patch_generated_truck(truck.read_text()), encoding='utf-8')
        java = next(item.findtext('Text') for item in model.findall('./JavaClasses/JavaClass')
                    if item.findtext('Name') == 'TopologyTripSmoke')
        java = replace(java, str(HERE / 'configs') + '/', str(work / 'configs') + '/')
        for case in ('A.properties', 'B.properties'):
            old, new = scratch_config(case, work)
            java = replace(java, old, new)
        (src / 'fusion' / 'TopologyTripSmoke.java').write_text('package fusion;\n' + java)
        (src / 'fusion' / 'TopologyTripStartupProbe.java').write_text(WIRING_SMOKE)
        classes = work / 'classes'
        sources = list(src.rglob('*.java'))
        subprocess.run(['javac', '--release', '17', '-encoding', 'UTF-8', '-cp', cp,
                        '-d', str(classes), *map(str, sources)], check=True)
        print(f'Compiled {len(sources)} generated or hooked Java sources', flush=True)
        subprocess.run([str(JAVA), '-Djava.awt.headless=true', '-cp',
                        os.pathsep.join((str(classes), str(base / 'classes'), cp)),
                        'fusion.TopologyTripStartupProbe'], cwd=work, check=True)
        print('No simulation-clock advance and no physical-trial receipt.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--build', type=Path, default=BASE)
    args = parser.parse_args()
    run(args.candidate, args.build)
