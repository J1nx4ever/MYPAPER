"""Build the single native serial A-map paired-severity experiment, from its audited pilot."""
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
MODEL = 'DRL-PPO-baseline-topologydemo-a-paired-severity'
EXPERIMENT = 'Topology_A_HighGradePairedSeverity'


def witness(code, count):
    code = replace(code, str(HERE.parent/'configs'), str(HERE/'configs'))
    code = bridge(code, '        int slot = number - 4000;', '        try {', f'''        int ordinal = number - 5000;
        require(ordinal >= 1 && ordinal <= {count}, "Invalid severity run ordinal");
        String key = String.format(java.util.Locale.ROOT, "P_%03d", ordinal);
        String phase = "test", experiment = "{EXPERIMENT}";
        int slot = (ordinal - 1) / 9 + 1;
        String strategy = new String[] {{"full","flow","flat"}}[((ordinal - 1) % 9) / 3];
        int duration = new int[] {{0,30,60}}[(ordinal - 1) % 3];
''')
    code = replace(code, '&& config.get("policy_update").getAsBigDecimal().intValueExact() == policyUpdate',
                   '&& config.get("slot").getAsBigDecimal().intValueExact() == slot\n'
                   '                    && config.get("strategy").getAsString().equals(strategy)\n'
                   '                    && config.get("duration_minutes").getAsBigDecimal().intValueExact() == duration')
    code = bridge(code, '        require("topology_a_full_high_grade_outage".equals', '        modelSha = hash', '''
        require("topology_a_high_grade_paired_severity".equals(text("purpose")), "Wrong paired campaign purpose");
        policyUpdate = config.get("policy_update").getAsBigDecimal().intValueExact();
        checkpointSha = text("checkpoint_sha256");
        require(checkpointSha.matches("[0-9a-f]{64}") && !text("policy_id").isBlank(), "Invalid policy identity");
        require("flow".equals(text("strategy")) ? policyUpdate == 0 : policyUpdate > 0,
                "Unexpected strategy version");
        durationMinutes = config.get("duration_minutes").getAsBigDecimal().intValueExact();
        if (durationMinutes == 0) {
            require(config.has("outage") && config.get("outage").isJsonNull(), "Expected uninterrupted loader");
        } else {
            readOutage();
            require(stopSec == 14400 && resumeSec == stopSec + 60 * durationMinutes, "Wrong stoppage window");
        }
''')
    code = replace(code, '"topology-high-grade-outage-v1"', '"topology-paired-severity-v1"')
    code = replace(code, '"load_site_outage", physicalOutagePlan()',
                   '"load_site_outage", durationMinutes == 0 ? Boolean.FALSE : physicalOutagePlan()')
    code = replace(code, '        main.evHighGradeStop.restart(stopSec, SECOND);\n'
                         '        main.evHighGradeResume.restart(resumeSec, SECOND);',
                   '''        if (durationMinutes != 0) {
            main.evHighGradeStop.restart(stopSec, SECOND);
            main.evHighGradeResume.restart(resumeSec, SECOND);
        } else {
            verifyNormalLoader();
        }''')
    code = replace(code, '            syncOutage();', '''            syncOutage();
            if (durationMinutes == 0) {
                verifyNormalLoader();
                controlDecisions++;
                double t = now();
                if (t >= 14400 && t < 16200) controlDecisions30++;
                if (t >= 14400 && t < 18000) controlDecisions60++;
            }''')
    code = replace(code, '    private int outageSite, stopSec', '''
    private final int durationMinutes;
    private long controlDecisions, controlDecisions30, controlDecisions60;
    private void verifyNormalLoader() {
        LoadSite site = main.getLoadSiteById(0);
        require(site != null && site.available && site.isOre && site.shovelCapacity == 1
                && site.shovelResource.capacity == 1 && wireId(site.loadNode) == 5
                && Math.abs(site.grade - 0.32) < 1e-12, "High-grade loader unavailable during reference run");
        for (int i = 0; i < site.delay.size(); i++)
            require(!site.delay.isSuspended(site.delay.get(i)), "Suspended service during reference run");
    }

    private int outageSite, stopSec''')
    code = replace(code, '    public void syncOutage() {',
                   '    public void syncOutage() {\n        if (durationMinutes == 0) return;')
    code = replace(code, '            main.boundaryEventRecorder.bindOutage(publishOutage());',
                   '''            var recorder = main.boundaryEventRecorder;
            recorder.bindSeverity(text("strategy"),text("policy_id"),durationMinutes);
            if (durationMinutes == 0) {
                verifyNormalLoader();
                require(controlDecisions == decisionSequence, "Incomplete control observations");
                recorder.bindAvailability(controlDecisions,controlDecisions30,controlDecisions60);
            } else {
                recorder.bindOutage(publishOutage());
            }''')
    return code


def recorder(code):
    code = replace(code, '"topology-high-grade-outage-v1"', '"topology-paired-severity-v1"')
    return replace(code, '    public synchronized void bindOutage(String sha)', '''
    public synchronized void bindSeverity(String strategy, String policy, int durationMinutes) {
        if (closed || manifest.containsKey("native_policy_id"))
            throw new IllegalStateException("Duplicate severity evidence");
        manifest.put("native_strategy",strategy);
        manifest.put("native_policy_id",policy);
        manifest.put("native_duration_minutes",durationMinutes);
    }
    public synchronized void bindAvailability(long all, long thirty, long sixty) {
        if (closed || manifest.containsKey("control_site_id") || all < 1 || thirty < 0 || thirty > sixty || sixty > all)
            throw new IllegalStateException("Invalid control availability evidence");
        manifest.put("control_site_id",0);
        manifest.put("control_available_decisions",all);
        manifest.put("control_available_window_decisions",thirty);
        manifest.put("control_available_60min_decisions",sixty);
    }
    public synchronized void bindOutage(String sha)''')


def build(base, target, slots):
    if not (30 <= slots <= 50):
        raise ValueError('Expected 30–50 matched seeds')
    count = slots * 9
    raw = base.read_bytes()
    if sha256(raw).hexdigest() != BASE_SHA or target.name != MODEL+'.alp':
        raise ValueError('Expected the sealed outage pilot and a new paired-severity sibling')
    source = raw.decode()
    pilot_name = 'Topology_A_Full_HighGradeOutage'
    old = named_block(source,'ParamVariationExperiment',pilot_name)
    new = replace(old,pilot_name,EXPERIMENT,count=old.count(pilot_name))
    new = replace(new,'<NumberOfRuns>1</NumberOfRuns>',f'<NumberOfRuns>{count}</NumberOfRuns>')
    new = replace(new,'<![CDATA[4001 + index]]>', '<![CDATA[5001 + index]]>')
    source = replace(source,old,new)
    for name, patch in [('TopologyWitness',lambda text: witness(text,count)), ('AcceptanceEventRecorder',recorder)]:
        old = named_block(source,'JavaClass',name)
        source = replace(source,old,patch(old))
    source = replace(source,f'<Name><![CDATA[{BASE}]]>',f'<Name><![CDATA[{MODEL}]]>')
    exp, = [e for e in ET.fromstring(source).findall('./Model/Experiments/ParamVariationExperiment')
            if e.findtext('Name') == EXPERIMENT]
    if (exp.findtext('NumberOfRuns'),exp.findtext('AllowParallelEvaluations'),
        exp.findtext('FreeformParamValue/Expression/Code')) != (str(count),'false','5001 + index'):
        raise ValueError('Unexpected native variation controls')
    with target.open('xb') as out:
        out.write(source.encode())
    return sha256(target.read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base',type=Path)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--slots',type=int,required=True)
    args = parser.parse_args()
    print(build(args.base,args.destination,args.slots),args.destination)
