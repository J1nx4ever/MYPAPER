"""Serve the nine-truck Full s02/u32 Actor to the native tree A outage experiment."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parent
TOPOLOGY = TRIAL.parent
PROJECT = TOPOLOGY.parents[3]
sys.path[:0] = [str(PROJECT), str(TOPOLOGY), str(TRIAL), str(TRIAL/'capacity_scaled_001')]

import trial_service
from scaled_service import ScaledTrial
from business_graph_summary_v1 import models
from business_graph_summary_v1.features_cache import SummaryFeatureCache
from relation_gat_ego_v1.adapter import validate_state
from topology_policy import attach_prototype_encoder, require_layout

FULL_SHA = '67b7b2c7104079836ccef8f5747e7aebb4e9e5a92dcaac67c94e42412f104f41'


class FullActor:
    def __init__(self, spec, edges):
        path = Path(spec['checkpoint_path'])
        require = trial_service.require
        require(trial_service.digest(path) == spec['checkpoint_sha256'] == FULL_SHA, 'wrong Full checkpoint')
        manifest = json.loads(Path(str(path)+'.manifest.json').read_bytes())
        saved = torch.load(path, weights_only=False, map_location='cpu')
        require(saved['update_index'] == manifest['update_index'] == 32, 'wrong Full update')
        self.actor = models.actor_from_config(saved['config'])
        self.actor.load_state_dict(saved['actor'], strict=True)
        attach_prototype_encoder(self.actor)
        self.actor.eval()
        self.actor.requires_grad_(False)
        require(models.digest(self.actor) == saved['actor_digest'] == spec['actor_digest'] == manifest['actor_digest'],
                'Full Actor parameters changed at the topology adapter')
        self.edges = edges

    @torch.inference_mode()
    def action(self, obs, mask, dispatchable, present, truck):
        raw = np.asarray(obs, dtype=np.float32)
        validate_state(raw, truck_id=truck)
        trial_service.require(dispatchable == present == 511 and np.array_equal(raw[19:23],mask),
                              'roster or action mask differs')
        state = raw[None,:]
        require_layout(state,self.edges)
        prefix = SummaryFeatureCache(state, feature_config=self.actor.feature_config).prefix()
        scores = self.actor(torch.from_numpy(state),prefix)[0]
        return int(scores.argmax().item())


class OutageTrial(ScaledTrial):
    def __init__(self, label):
        import a_outage_campaign as campaign
        spec, self.sha, self.edges = campaign.read_case(label)
        trial_service.require(spec['layout_id']=='tree_A' and spec['outage']['site_id']==0
                              and spec['policy_update']==32 and spec['action_mode']=='argmax'
                              and spec['actor_sampling_seed'] is None and spec['slot_permutation_seed']==0,
                              'unexpected A pressure-test configuration')
        self.spec=spec
        self.output=Path(spec['python_output_directory'])
        self.events=Path(spec['event_directory'])
        trial_service.require(not self.events.exists() and not (self.output/'inference.jsonl').exists(),
                              'physical identity already started')
        self.active=sum(1 << row[0] for row in spec['expected_runtime_scene']['opening_fleet'] if row[1])
        trial_service.require(self.active==511, 'expected nine active trucks')
        self.policy=FullActor(spec,self.edges)
        self.sequence=0
        self.previous_time=-1.
        self.last_key=self.reply=self.writer=None
        self.sealed=False

    def decide(self,request):
        fault=self.spec['outage']
        closed=fault['start_seconds'] <= request['sim_time_seconds'] < fault['end_seconds']
        obs=request['obs']
        trial_service.require(obs[16]==0 and obs[1107]==int(not closed),
                              'native randomness or high-grade site availability differs')
        if closed:
            trial_service.require(request['action_mask'][0]==obs[19]==obs[23]==obs[1114]==0,
                                  'stopped high-grade loader offers an action or uncommitted tons')
        return super().decide(request)


if __name__ == '__main__':
    import a_outage_campaign as campaign
    trial_service.Trial=OutageTrial
    trial_service.serve(campaign.case_labels())
