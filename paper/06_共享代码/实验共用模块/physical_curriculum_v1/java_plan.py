"""Render an immutable 6,844-job helper; accessors never claim or open RPC."""
from pathlib import Path
import re
from .gate import CAMPAIGN_ID,CAP,JAVA_TEMPLATE_PATH,READY_PROTOCOL,ROOT,require,sha

def render_helper(master,jobs):
    master=Path(master).resolve();require(len(jobs)==CAP and master.is_file(),'exact6844 master required')
    text=JAVA_TEMPLATE_PATH.read_text()
    def once(old,new):
        nonlocal text
        require(text.count(old)==1,'Java helper anchor changed: '+old[:70]);text=text.replace(old,new,1)
    def one_regex(pattern,new):
        nonlocal text
        matches=list(re.finditer(pattern,text))
        require(len(matches)==1,'Java helper regex anchor changed: '+pattern[:70])
        hit=matches[0];text=text[:hit.start()]+new+text[hit.end():]
    one_regex(r'private static final String MASTER_PLAN_PATH = "[^"]+";',
        'private static final String MASTER_PLAN_PATH = "'+str(master)+'";')
    one_regex(r'private static final String MASTER_PLAN_SHA256 = "[0-9a-f]+";',
        'private static final String MASTER_PLAN_SHA256 = "'+sha(master)+'";')
    old_campaign='u8-wait-efficiency-seed1-20261109-001'
    require(text.count(old_campaign)==2,'Java helper campaign/control anchors changed')
    text=text.replace(old_campaign,CAMPAIGN_ID)
    once('private static final int JOB_COUNT = 812;','private static final int JOB_COUNT = 6844;')
    once('values[3].equals("train") || values[3].equals("final")','values[3].equals("train") || values[3].equals("monitor") || values[3].equals("final")')
    once('stage.equals("reference") || stage.equals("wait01") || stage.equals("wait015")',
        'stage.equals("reference") || stage.equals("wait015") || stage.equals("wait02") || stage.equals("fixed015") || stage.equals("fixed02")')
    once('train_u000[1-8]|final_0100','train_u00(?:0[1-9]|[1-3][0-9]|4[0-8])|benchmark_0100|blind_0100|monitor_n(?:0768|1024|1280|1536)')
    once('require(values[3].equals("train") == localPhase.startsWith("train_"), "phase role differs");',
        'require(values[3].equals(localPhase.startsWith("train_") ? "train" : (localPhase.startsWith("monitor_") ? "monitor" : "final")), "phase role differs");\n'
        '            require(!stage.equals("reference") || localPhase.equals("benchmark_0100"), "reference benchmark only");\n'
        '            require(!localPhase.equals("benchmark_0100") || stage.equals("reference") || stage.equals("wait015") || stage.equals("wait02"), "benchmark stage differs");\n'
        '            require(!localPhase.equals("blind_0100") || !stage.equals("reference"), "blind stage differs");\n'
        '            if (localPhase.startsWith("train_")) {\n'
        '                int update = Integer.parseInt(localPhase.substring(7));\n'
        '                require(update >= 1 && update <= (stage.startsWith("fixed") ? 48 : 32), "training update outside arm range");\n'
        '            }')
    once('stage.equals("wait015") ? "0.15" : "0.1"',
        'stage.equals("reference") ? "0.1" : ((stage.equals("wait015") || stage.equals("fixed015")) ? "0.15" : "0.2")')
    once('T0[1-8]-[0-9]{2}|E-[0-9]{3}','(?:train|benchmark|monitor|blind)-[0-9]{4}')
    once('"u8-wait-efficiency-ready-v1"','"'+READY_PROTOCOL+'"')
    once('            if (row.cells[3].equals("final")) {',
        '            same(properties, "checkpoint_snapshot_verified", "true");\n            if (!row.cells[3].equals("train")) {')
    once('/** Frozen u8 comparison helper. Freeform is read-only; execution requires authorized ready/claim. */',
        '/** Frozen6844 curriculum/fixed control. Build first; one authorized Python controller and one AnyLogic Run. */')
    expected=(
        'private static final String MASTER_PLAN_PATH = "'+str(master)+'";',
        'private static final String MASTER_PLAN_SHA256 = "'+sha(master)+'";',
        'private static final String CONTROL_DIRECTORY = "'+str(ROOT/'runs'/CAMPAIGN_ID/'control')+'";',
        'private static final String CAMPAIGN_ID = "'+CAMPAIGN_ID+'";',
        'private static final int JOB_COUNT = 6844;',
        'private static final boolean PREPARATION_ONLY = false;',
        'same(properties, "protocol", "'+READY_PROTOCOL+'");',
        'same(properties, "checkpoint_snapshot_verified", "true");',
        'same(properties, "all_comparison_checkpoints_frozen", "true");',
    )
    require(all(text.count(marker)==1 for marker in expected),'rendered Java identity or ready protocol differs')
    require('u8-wait-efficiency' not in text and 'train_u000[1-8]' not in text and
        'stage.equals("wait01")' not in text,'old Java campaign/phase/stage remained')
    return text
