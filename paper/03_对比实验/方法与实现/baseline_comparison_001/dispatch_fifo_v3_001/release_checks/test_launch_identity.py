"""A display card must match the identities the Java and Python launchers consume."""
import json
from pathlib import Path

import pytest

from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.dispatch_fifo_v3_001 import prepare


def test_card_cannot_substitute_another_slots_recording(tmp_path, monkeypatch):
    suite = prepare.load_suite()
    card = json.loads((prepare.PREPARED / 'launch_identity.json').read_bytes())
    card['jobs'][1]['recording_id'] = card['jobs'][0]['recording_id']
    (tmp_path / 'launch_identity.json').write_text(json.dumps(card))
    monkeypatch.setattr(prepare, 'PREPARED', tmp_path)
    with pytest.raises(ValueError, match='identity card'):
        prepare.verify_launch_identity(suite)


def test_retired_helper_cannot_launch_the_final_suite():
    from v3_plan import RETIRED_SUITE, reference
    old = json.loads(Path(RETIRED_SUITE['path']).read_bytes())
    helper = Path(old['java_helper']['path']).read_text()
    launcher = Path(prepare.SUITE).parents[1] / 'launch.py'
    assert reference(launcher)['sha256'] not in helper
    with pytest.raises(ValueError, match='suite path'):
        prepare.load_suite(RETIRED_SUITE['path'])
