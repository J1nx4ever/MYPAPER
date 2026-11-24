"""Render a new 812-row helper from the sealed historical one-start template."""
from pathlib import Path
import re

from .gate import CAMPAIGN_ID, CAP, JAVA_TEMPLATE_PATH, READY_PROTOCOL, require, sha


def render_helper(master, jobs):
    require(len(jobs) == CAP, 'exact812 helper rows required')
    master = Path(master).resolve()
    require(master.is_file(), 'frozen master must precede Java rendering')
    text = JAVA_TEMPLATE_PATH.read_text()
    def once(old, new):
        nonlocal text
        require(text.count(old) == 1, 'helper anchor absent/ambiguous: '+old[:60])
        text = text.replace(old, new, 1)
    text = re.sub(r'private static final String MASTER_PLAN_PATH = "[^"]+";',
                  'private static final String MASTER_PLAN_PATH = "'+str(master)+'";', text, count=1)
    text = re.sub(r'private static final String MASTER_PLAN_SHA256 = "[0-9a-f]+";',
                  'private static final String MASTER_PLAN_SHA256 = "'+sha(master)+'";', text, count=1)
    text = text.replace('u8-wait-efficiency-seed1-20261109-001', CAMPAIGN_ID)
    # One replacement at a time prevents silently accepting a changed template.
    once('stage.equals("reference") || stage.equals("wait01") || stage.equals("wait015")',
         'stage.equals("reference") || stage.equals("wait015") || stage.equals("wait02")')
    once('stage.equals("wait015") ? "0.15" : "0.1"',
         'stage.equals("wait015") ? "0.15" : (stage.equals("wait02") ? "0.2" : "0.1")')
    once('"u8-wait-efficiency-ready-v1"', '"'+READY_PROTOCOL+'"')
    once('            if (row.cells[3].equals("final")) {',
         '            same(properties, "checkpoint_snapshot_verified", "true");\n'
         '            if (row.cells[3].equals("final")) {')
    once('/** Frozen u8 comparison helper. Freeform is read-only; execution requires authorized ready/claim. */',
         '/** Frozen physical46 RAB tradeoff. Freeform is read-only; no additional launches per phase. */')
    return text
