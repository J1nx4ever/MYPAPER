"""Bind the three dispatch implementations to their archived policy artifacts."""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a_outage_campaign as original

PROJECT = original.PROJECT
BUSINESS = HERE.parents[3]
ABLATION = BUSINESS/'structural_ablation_001'
GRADE = BUSINESS/'baseline_comparison_001/dispatch_flow_matching_001/grade_control_001'
FLAT_RUN = PROJECT/'runs/bg128-ablation-20261121-flat-s01'
FLAT_MODEL = FLAT_RUN/'arms/s01_BG128/updates/0032/checkpoint.pt'
FLAT_SHA = '40ae504cb159d12837d7c22fc7621e563dabad340fe0a39a7e07aa30766e4804'
FLAT_ACTOR = '131391585ac1f4a4b388d7e9af4c27523df879e61ca0087cec1748455c0f2e9c'
FLOW_SUITE = GRADE/'prepared_001/suite.json'
FLOW_SHA = '866c8822bef87f89e26bd786bac138d1001e97d3fe8ea578aa403afbf9e8e248'
FLOW_SETTINGS = dict(match_trucks=3, planning_ms=800, replan_seconds=900,
                     response_target_ms=1000, soft_replan_seconds=120)


def pin(path):
    path = Path(path)
    return dict(path=str(path),sha256=sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(Path(path).read_bytes())


@lru_cache(maxsize=3)
def policy_metadata(strategy):
    # Shared dispatch transport, feature extraction and protected core come from
    # the exact A-pilot contract. The two new runtime sources are byte-pinned too.
    base = original.policy()
    runtime = [*base['inference_sources'], pin(__file__), pin(HERE/'batch_service.py')]
    common = dict(inference_host='127.0.0.1', inference_port=50871)
    if strategy=='full':
        return dict(base, artifact_kind='actor_checkpoint', action_mode='argmax',inference_sources=runtime)
    if strategy=='flat':
        cp = pin(FLAT_MODEL)
        manifest = pin(str(FLAT_MODEL)+'.manifest.json')
        seal = pin(str(FLAT_MODEL)+'.seal.json')
        info,signature = read(manifest['path']),read(seal['path'])
        archive_path=FLAT_RUN/'suite_complete.json'
        archive=read(archive_path)
        suite=read(archive['suite']['path'])
        original.require(archive['status']=='COMPLETE' and archive['training_updates']==32
                         and archive['evaluation_optimizer_updates']==0 and archive['new_checkpoint']==cp
                         and archive['suite']==pin(archive['suite']['path'])
                         and suite['ablation_variant']=='flat_mlp' and cp['sha256']==FLAT_SHA
                         and signature['sha256']==info['sha256']==FLAT_SHA
                         and signature['manifest_sha256']==manifest['sha256']
                         and info['update_index']==32 and info['actor_digest']==FLAT_ACTOR,'Flat-PPO seal differs')
        extra = [pin(PROJECT/'business_graph_ablation_v1/models.py'),pin(ABLATION/'ablation_runtime.py')]
        original.require(all(suite['source_pins'][r['path']]==r['sha256'] for r in extra),
                         'Flat-PPO policy implementation differs from training')
        return dict(**common, artifact_kind='actor_checkpoint', policy_id='flat-ppo-s01-u32',
                    policy_update=32,action_mode='argmax',checkpoint_path=cp['path'],checkpoint_sha256=FLAT_SHA,
                    actor_digest=FLAT_ACTOR, checkpoint_manifest=manifest,checkpoint_seal=seal,
                    policy_suite=archive['suite'],completion=pin(archive_path),inference_sources=runtime+extra)
    original.require(strategy=='flow','unknown dispatch policy')
    cp=pin(FLOW_SUITE)
    suite=read(FLOW_SUITE)
    completion=PROJECT/'runs/bg128-dispatch-flow-grade-control10-20261123-001/suite_complete.json'
    info=read(completion)
    original.require(cp['sha256']==FLOW_SHA and info['status']=='COMPLETE' and info['suite']==cp
                     and info['optimizer_updates']==0 and suite['policy_kind']=='dispatch_flow_grade_control'
                     and suite['method']==FLOW_SETTINGS and all(pin(r['path'])==r for r in suite['source_pins']),
                     'path-flow policy, grade guard or settings differ from the sealed suite')
    return dict(**common,artifact_kind='sealed_deterministic_policy_suite',
                policy_id='dispatch-flow-matching-grade-control',policy_update=0,action_mode='deterministic',
                checkpoint_path=cp['path'],checkpoint_sha256=FLOW_SHA,policy_suite=cp,completion=pin(completion),
                inference_sources=runtime+suite['source_pins'],method=suite['method'])
