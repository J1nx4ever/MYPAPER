"""Actual completed BG128 episode storage regression, no optimizer or RPC."""
import copy
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest
import torch

from .episode import (
    DATASET_SCHEMA, _valid_identity, read_episode_hdf5, write_episode_hdf5,
)
from business_graph_summary_recovery_v1.collection import load_admitted_episode
from business_graph_summary_recovery_v1.plan_io import ROOT
from business_graph_summary_v1.targets import verify_episode_targets

REPORT=ROOT/'reports/token_attention_001/business_graph_summary_009/formal_training_001/recovery_001'


@pytest.fixture(scope='module')
def episode():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    try:
        receipt=json.loads((REPORT/'material_receipt.json').read_bytes())
        value=load_admitted_episode(receipt)
        assert value['identity']['run_index']==9900001
        assert len(value['actions'])==184 and value['states'].shape==(185,1519)
        yield value
    finally:
        torch.set_num_threads(previous)


def values(episode):
    return {name:episode[name] for name in
            ('identity','states','actions','old_log_probs','memory','evaluation','wait_deltas')}


def validator(original):
    def check(derived):
        assert derived['identity']==original['identity']
        for name in ('states','actions','old_log_probs','wait_deltas'):
            assert np.array_equal(derived[name],original[name])
        assert derived['memory']==original['memory']
        assert verify_episode_targets(derived)==original['targets']
    return check


def test_actual_bg_episode_canonical_write_strong_read_roundtrip(tmp_path,episode):
    path=tmp_path/'bg_actual.h5'
    meta=write_episode_hdf5(path,**values(episode),validator=validator(episode))
    assert meta['schema']==DATASET_SCHEMA and meta['counts']['transitions']==184
    restored=read_episode_hdf5(path,expected_identity=episode['identity'],validator=validator(episode))
    assert restored['targets']==episode['targets']
    assert np.array_equal(restored['states'],episode['states'])
    with pytest.raises(FileExistsError,match='never overwrite'):
        write_episode_hdf5(path,**values(episode),validator=validator(episode))


@pytest.mark.parametrize('field,value,message',[
    ('candidate_id','s01_TQ128','unknown business-graph candidate'),
    ('graph_contract_sha256','0'*64,'graph architecture contract identity differs'),
    ('split','synthetic','source_kind/split'),
])
def test_wrong_architecture_or_partition_rejected_before_admission(episode,field,value,message):
    identity=copy.deepcopy(episode['identity']);identity[field]=value
    with pytest.raises(ValueError,match=message):_valid_identity(identity)


def test_real_writer_and_reader_require_substantive_runtime_validator(tmp_path,episode):
    with pytest.raises(ValueError,match='nonsynthetic admission requires'):
        write_episode_hdf5(tmp_path/'no_validator.h5',**values(episode))
    assert not (tmp_path/'no_validator.h5').exists()
    source=json.loads((REPORT/'first_episode_evidence.json').read_bytes())['derived_h5']['path']
    with pytest.raises(ValueError,match='nonsynthetic admission requires'):
        read_episode_hdf5(source,expected_identity=episode['identity'])


@pytest.mark.parametrize('tamper',['manifest','hdf5','missing_seal'])
def test_manifest_hdf5_and_publication_tamper_are_rejected(tmp_path,episode,tamper):
    source=Path(json.loads((REPORT/'first_episode_evidence.json').read_bytes())['derived_h5']['path'])
    target=tmp_path/'copied.h5'
    for suffix in ('','.manifest.json','.seal.json'):
        shutil.copyfile(str(source)+suffix,str(target)+suffix)
    if tamper=='manifest':
        with Path(str(target)+'.manifest.json').open('ab') as handle:handle.write(b'\n')
        message='manifest SHA256 differs'
    elif tamper=='hdf5':
        with h5py.File(target,'r+') as handle:handle['states'][0,0]+=1e-3
        message='HDF5 SHA256 differs'
    else:
        Path(str(target)+'.seal.json').unlink()
        message='incomplete episode'
    with pytest.raises(ValueError,match=message):
        read_episode_hdf5(target,validator=validator(episode))


def test_regression_old_storage_still_rejects_BG_and_is_not_patched(episode):
    from single_critic_task_query_v1.storage import _valid_identity as old_validator
    with pytest.raises(ValueError,match='unknown task-query candidate'):
        old_validator(episode['identity'])
