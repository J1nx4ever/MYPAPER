"""Frozen seed plan cannot masquerade as an executable evaluation config."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from business_graph_stability_campaign_v1.pairing import evaluation_source,contract,TARGET_EVAL
from business_graph_stability_campaign_v1.plan_io import ROOT,reference
from .gate import PLAN,check_jobs,SUITE_CONTRACT,PHASE_ID
from .runtime import _load_model


def test_pending_seed_plan_has_no_fabricated_checkpoint_or_config_sha():
    pending=json.loads((PLAN/'seed_plan.json').read_bytes())
    assert pending['checkpoint_sha256'] is None and pending['final_config_sha256'] is None
    assert pending['status']=='AWAITING_NEW_U13_CHECKPOINT'
    source=evaluation_source()
    assert len(pending['jobs'])==100
    assert all(new['experiment_seed']==old['experiment_seed'] and
        new['actor_sampling_seed']==old['actor_sampling_seed'] for new,old in zip(pending['jobs'],source['jobs']))


def test_future_evaluation_jobs_use_exact_pairs_and_new_physical_directories():
    suite=deepcopy(evaluation_source())
    output=ROOT/'runs'/TARGET_EVAL;material=output.with_name(TARGET_EVAL+'_materials')/PHASE_ID
    suite.update(contract=SUITE_CONTRACT,campaign_id=TARGET_EVAL,pairing=contract(),
                 python_output=str(output),control_directory=str(output/'control'))
    for ordinal,j in enumerate(suite['jobs']):
        run=9970001+ordinal;ep=f"dispatch-production-wait-shortfall-v2-run-{run}-seed-{j['experiment_seed']}"
        j.update(run_index=run,recording_id=TARGET_EVAL+'-'+PHASE_ID,source_recording_id=TARGET_EVAL+'-'+PHASE_ID,
            python_output=str(output/'evaluation'),material_event_directory=str(material),
            material_event_bundle=str(material/ep),material_journal=str(material/ep/'material_events.bin'))
    check_jobs(suite)
    suite['jobs'][0]['experiment_seed']+=1
    with pytest.raises(ValueError,match='corresponding frozen source row'):
        check_jobs(suite)


def test_historical_u13_is_rejected_instead_of_used_as_new_model(tmp_path):
    folder=ROOT/'runs/business-graph-summary-formal-seed1-20261118-001/arms/s01_BG128/updates/0013'
    checkpoint=folder/'checkpoint.pt'
    selection=tmp_path/'selection.json';selection.write_text('{}')
    suite=dict(parent_artifacts={k:reference(p) for k,p in
        [('checkpoint',checkpoint),('manifest',str(checkpoint)+'.manifest.json'),('seal',str(checkpoint)+'.seal.json')]},
        selection=reference(selection),learner_config={},checkpoint_update=13)
    with pytest.raises(ValueError,match='checkpoint schema differs'):
        _load_model(suite)
