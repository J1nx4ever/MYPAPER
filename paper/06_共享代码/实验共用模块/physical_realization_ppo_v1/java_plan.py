"""Render the existing one-start ready/claim helper for the frozen608 master."""
from pathlib import Path
import re

from .gate import ROOT,CAMPAIGN_ID,CAP,JAVA_TEMPLATE_PATH,require,sha


def render_helper(master,jobs):
    require(len(jobs)==CAP,'exact608 Java rows required')
    master=Path(master).resolve()
    require(master.is_file(),'frozen master bytes required before Java rendering')
    text=JAVA_TEMPLATE_PATH.read_text()
    def once(before,after):
        nonlocal text
        require(text.count(before)==1,'reviewed helper anchor absent/ambiguous: '+before[:60])
        text=text.replace(before,after,1)
    text=re.sub(r'private static final String MASTER_PLAN_PATH = "[^"]+";',
        'private static final String MASTER_PLAN_PATH = "'+str(master)+'";',text,count=1)
    text=re.sub(r'private static final String MASTER_PLAN_SHA256 = "[0-9a-f]+";',
        'private static final String MASTER_PLAN_SHA256 = "'+sha(master)+'";',text,count=1)
    old_campaign='u8-wait-efficiency-seed1-20261109-001'
    text=text.replace(old_campaign,CAMPAIGN_ID)
    once('private static final int JOB_COUNT = 812;','private static final int JOB_COUNT = 608;')
    once('require(values[3].equals("train") || values[3].equals("final"), "invalid role");',
         'require(values[3].equals("t0") || values[3].equals("train") || values[3].equals("monitor") || values[3].equals("final"), "invalid role");')
    once('require(stage.equals("reference") || stage.equals("wait01") || stage.equals("wait015"), "invalid stage");',
         'require(stage.equals("production") || stage.equals("wait01"), "invalid stage");')
    once('            require(!stage.equals("reference") || values[3].equals("final"), "reference cannot train");\n','')
    once('require(localPhase.matches("train_u000[1-8]|final_0100"), "invalid phase");',
         'require(localPhase.matches("t0|train_u000[1-8]|monitor_u0008|final_u0008"), "invalid phase");')
    once('require(values[17].equals(stage.equals("wait015") ? "0.15" : "0.1"), "wait weight differs");',
         'require(values[17].equals(stage.equals("production") ? "0.0" : "0.1"), "wait weight differs");')
    once('require(values[18].matches("T0[1-8]-[0-9]{2}|E-[0-9]{3}"), "invalid pairing identity");',
         'require(values[18].matches("[A-Za-z0-9_-]+"), "invalid scenario identity");')
    once('same(properties, "protocol", "u8-wait-efficiency-ready-v1");',
         'same(properties, "protocol", "physical-realization-ppo-ready-v1");')
    once('            if (row.cells[3].equals("final")) {\n                same(properties, "all_comparison_checkpoints_frozen", "true");\n            }',
         '            same(properties, "checkpoint_snapshot_verified", "true");\n'
         '            if (!row.cells[3].equals("train")) {\n'
         '                same(properties, "evaluation_checkpoint_frozen", "true");\n            }')
    once('/** Frozen u8 comparison helper. Freeform is read-only; execution requires authorized ready/claim. */',
         '/** Frozen608 single-model physical PPO curriculum. No Freeform side effects. */')
    return text


__all__=['render_helper']
