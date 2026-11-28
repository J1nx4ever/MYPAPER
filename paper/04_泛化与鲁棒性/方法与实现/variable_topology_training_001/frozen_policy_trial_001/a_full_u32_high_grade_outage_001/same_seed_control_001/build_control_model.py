"""Derive the one-shift untreated A sibling from the sealed native outage model."""
import argparse
from hashlib import sha256
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from build_outage_model import bridge, named_block, replace

BASE = 'DRL-PPO-baseline-topologydemo-a-full-u32-high-grade-outage'
BASE_SHA = '9022e92f066b4e9709e1636a177e10b5efd26dff02d22d56b41962af7ed65280'
MODEL = 'DRL-PPO-baseline-topologydemo-a-full-u32-no-outage-control'
OLD_EXPERIMENT = 'Topology_A_Full_HighGradeOutage'
EXPERIMENT = 'Topology_A_Full_NoOutageControl'


def witness(code):
    code = replace(code, str(HERE.parent/'configs'), str(HERE/'configs'))
    code = bridge(code, '        int slot = number - 4000;', '        int policyUpdate = 32;', f'''        require(number == 4002, "Expected A control case 4002");
        String key = "C_001";
        String phase = "test", experiment = "{EXPERIMENT}";
''')
    code = replace(code, '"topology_a_full_high_grade_outage".equals(text("purpose"))',
                   '"topology_a_full_no_outage_control".equals(text("purpose"))')
    code = replace(code, '        readOutage();',
                   '        require(config.has("outage") && config.get("outage").isJsonNull(), "Expected a control with no scheduled outage");')
    code = replace(code, '"topology-high-grade-outage-v1"', '"topology-no-outage-control-v1"')
    code = replace(code, '        main.evHighGradeStop.restart(stopSec, SECOND);\n'
                         '        main.evHighGradeResume.restart(resumeSec, SECOND);', '        verifyNormalLoader();')
    code = replace(code, '"load_site_outage", physicalOutagePlan()', '"load_site_outage", false')
    code = replace(code, '            syncOutage();', '''            verifyNormalLoader();
            availableDecisions++;
            if (now() >= 14400 && now() < 16200) availableWindowDecisions++;''')
    code = bridge(code, '    private int outageSite, stopSec', '    public void finish(double cumulativeWait)', '''
    private long availableDecisions, availableWindowDecisions;

    private void verifyNormalLoader() {
        LoadSite site = main.getLoadSiteById(0);
        require(site != null && site.available && site.isOre && site.shovelCapacity == 1
                && site.shovelResource.capacity == 1 && wireId(site.loadNode) == 5
                && Math.abs(site.grade - 0.32) < 1e-12, "High-grade loader is not operating normally");
        for (int i = 0; i < site.delay.size(); i++)
            require(!site.delay.isSuspended(site.delay.get(i)), "Unexpected suspended control service");
    }

''')
    return replace(code, '            main.boundaryEventRecorder.bindOutage(publishOutage());', '''            verifyNormalLoader();
            require(availableDecisions == decisionSequence, "Incomplete native loader checks during control");
            main.boundaryEventRecorder.bindControl(availableDecisions,availableWindowDecisions);''')


def recorder(code):
    code = replace(code, '"topology-high-grade-outage-v1"', '"topology-no-outage-control-v1"')
    return bridge(code, '    public synchronized long outageCursor()', '    public synchronized void bindPolicyUpdate', '''    public synchronized void bindControl(long decisions, long windowDecisions) {
        if (closed || manifest.containsKey("control_site_id") || decisions < 1
                || windowDecisions < 0 || windowDecisions > decisions)
            throw new IllegalStateException("Invalid or duplicate native control checks");
        manifest.put("control_site_id",0);
        manifest.put("control_available_decisions",decisions);
        manifest.put("control_available_window_decisions",windowDecisions);
    }

''')


def build(base, target):
    raw = base.read_bytes()
    if sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError('Expected the archived A s02/u32 intervention model')
    if target.name != MODEL+'.alp':
        raise ValueError(f'Expected a separate sibling named {MODEL}.alp')
    code = raw.decode()
    case = named_block(code, 'ParamVariationExperiment', OLD_EXPERIMENT)
    new = replace(case, OLD_EXPERIMENT, EXPERIMENT, count=case.count(OLD_EXPERIMENT))
    new = replace(new, '<![CDATA[4001 + index]]>', '<![CDATA[4002 + index]]>')
    code = replace(code, case, new)
    for name in ('evHighGradeStop', 'evHighGradeResume'):
        code = replace(code, named_block(code, 'Event', name), '')
    for name, adapt in (('TopologyWitness',witness),('AcceptanceEventRecorder',recorder)):
        block = named_block(code,'JavaClass',name)
        code = replace(code,block,adapt(block))
    code = replace(code,f'<Name><![CDATA[{BASE}]]>',f'<Name><![CDATA[{MODEL}]]>')
    exp, = [e for e in ET.fromstring(code).findall('./Model/Experiments/ParamVariationExperiment') if e.findtext('Name') == EXPERIMENT]
    if (exp.findtext('Name'), exp.findtext('NumberOfRuns'), exp.findtext('AllowParallelEvaluations'),
            exp.findtext('FreeformParamValue/Expression/Code')) != (EXPERIMENT,'1','false','4002 + index'):
        raise ValueError('Unexpected control experiment or iteration configuration')
    if any(t in code for t in ('evHighGradeStop','evHighGradeResume','syncOutage()', 'publishOutage()', 'delay.suspend(')):
        raise ValueError('Residual loader outage code')
    with target.open('xb') as out:
        out.write(code.encode())
    return sha256(target.read_bytes()).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    print(build(args.base,args.destination),args.destination)
