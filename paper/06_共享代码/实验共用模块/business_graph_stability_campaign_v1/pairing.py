"""Explicitly requested common initialization and corresponding seed replay.

Only seed fields may repeat; every physical run and recording stays fresh.
The exception is bound to the exact completed BG128 source manifests.
"""
from pathlib import Path
from .plan_io import ROOT, reference, load_reference
from .contract import require

BASE = ROOT/'change_plans/token_attention_001/business_graph_summary_009'
TRAIN_SUITE = BASE/'formal_training_001/prepared_001/suite.json'
EVAL_SUITE = BASE/'checkpoint_evaluation_001/prepared_001/suite.json'
INITIAL = ROOT/'runs/business-graph-summary-formal-seed1-20261118-001/arms/s01_BG128/initial.pt'
INITIAL_SHA = '75bd2d54a78dad799c9f51864def284b915e0246fa2189cd32f5b6bd7a6948d3'
TARGET_TRAIN = 'business-graph-stability-lr1e4-seed1-20261118-001'
TARGET_EVAL = 'business-graph-stability-paired-eval100-20261118-001'


def training_source():
    from business_graph_summary_campaign_v1.plan_io import load_suite
    suite,_=load_suite(TRAIN_SUITE)
    complete=load_reference(reference(Path(suite['python_output'])/'suite_complete.json'))
    require(len(suite['jobs'])==512 and complete['global_updates']==16 and
        complete['campaign']==reference(TRAIN_SUITE),'completed original BG128 training required')
    return suite


def evaluation_source():
    suite=load_reference(reference(EVAL_SUITE))
    require(len(suite['jobs'])==100 and suite['checkpoint_update']==13 and
        suite['sampling']=='stochastic_private_rng','original fixed-u13 evaluation required')
    return suite


def contract():
    require(reference(INITIAL)['sha256']==INITIAL_SHA,'original common initialization bytes changed')
    return dict(schema='bg128-explicit-seed-pairing-v1',
        user_seed_instruction='用和这次相同的种子进行训练和推理。',
        user_checkpoint_instruction='同时应该从原始共同初始化 Checkpoint 重新开始',
        training_suite=reference(TRAIN_SUITE),evaluation_suite=reference(EVAL_SUITE),
        common_initial_checkpoint=reference(INITIAL),
        common_initial_manifest=reference(str(INITIAL)+'.manifest.json'),
        common_initial_seal=reference(str(INITIAL)+'.seal.json'),
        reusable_fields=['actor_seed','experiment_seed','actor_sampling_seed'],
        fresh_fields=['campaign_id','recording_id','source_recording_id','run_index','python_output'],
        accounting='every_paired_physical_shift_is_new_consumption')


def check_seed_pairing(suite, *, evaluation=False):
    require(suite.get('pairing')==contract(),'explicit frozen seed-pairing exception required')
    source=evaluation_source() if evaluation else training_source()
    require(suite['campaign_id']==(TARGET_EVAL if evaluation else TARGET_TRAIN),
            'seed exception is limited to this one paired experiment')
    require(len(suite['jobs'])==len(source['jobs']),'paired source job count differs')
    for ordinal,(new,old) in enumerate(zip(suite['jobs'],source['jobs'])):
        require(new['run_index']==(9970001 if evaluation else 9960001)+ordinal,
            'seed exception physical identity differs')
        require(all(new[k]==old[k] for k in ('actor_seed','experiment_seed','actor_sampling_seed')),
            'seed reuse allowed only for the corresponding frozen source row')
        require(all(new[k]!=old[k] for k in ('run_index','recording_id','source_recording_id')),
            'paired seeds cannot reuse physical identity')
    return source
