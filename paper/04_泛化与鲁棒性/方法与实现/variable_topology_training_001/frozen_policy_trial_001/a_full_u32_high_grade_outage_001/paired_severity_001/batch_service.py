"""Serve fixed learned and grade-controlled policies to the paired A-map shifts."""
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from outage_service import FullActor, ScaledTrial, SummaryFeatureCache, models, trial_service, validate_state
from business_graph_ablation_v1 import models as flat_models
from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.dispatch_flow_matching_001.grade_control_001.grade_policy import GradeControlPolicy
from topology_policy import require_layout
from resilience_runtime import record
import batch_policy_pins as pins


class FlatActor:
    def __init__(self,spec,edges):
        self.edges=edges
        suite=pins.read(spec['policy_suite']['path'])
        saved=torch.load(spec['checkpoint_path'],map_location='cpu',weights_only=False)
        raw=saved['config']
        trial_service.require(raw==suite['arms'][0]['learner_config'],'Flat-PPO learner configuration differs')
        variant=suite['ablation_variant']
        cfg=dict(raw,ablation_variant=variant,ablation_contract=suite['ablation'],
                 network=flat_models.ACTOR_SPECS[variant],memory_critic=flat_models.CRITIC_SPECS[variant])
        self.actor=flat_models.actor_from_config(cfg)
        self.actor.load_state_dict(saved['actor'],strict=True)
        self.actor.eval().requires_grad_(False)
        trial_service.require(saved['update_index']==32 and models.digest(self.actor)==saved['actor_digest']==spec['actor_digest'],
                              'Flat-PPO Actor changed')

    @torch.inference_mode()
    def action(self,obs,mask,dispatchable,present,truck):
        raw=np.asarray(obs,dtype=np.float32)
        validate_state(raw,truck_id=truck)
        trial_service.require(dispatchable==present==511 and np.array_equal(raw[19:23],mask),
                              'Flat-PPO roster or action mask differs')
        state=raw[None,:]
        require_layout(state,self.edges)
        prefix=SummaryFeatureCache(state,feature_config=self.actor.feature_config).prefix()
        return int(self.actor(torch.from_numpy(state),prefix)[0].argmax().item())


class FlowActor:
    def __init__(self,spec):
        self.engine=GradeControlPolicy(**{k:spec['method'][k] for k in
                                        ('match_trucks','planning_ms','soft_replan_seconds','replan_seconds')})
        self.pending=None
        self.by_time=defaultdict(Counter)
        self.t=0.
        self.outage=spec['outage']

    def commit_previous(self):
        if self.pending is not None:
            self.engine.commit(self.pending)
            self.pending=None

    def action(self,obs,mask,dispatchable,present,truck):
        # A new ordinal from this native witness arrives only after the previous
        # action was applied. Retried RPCs are deduplicated by Trial.decide.
        self.commit_previous()
        choice=self.engine.choose(obs,mask)
        self.pending=choice
        now=self.t
        interval='before_4h' if now<14400 else ('4h_to_4h30' if now<16200 else '4h30_to_5h' if now<18000 else 'after_5h')
        names=['total',interval]
        if self.outage and self.outage['start_seconds']<=now<self.outage['end_seconds']:
            names.append('during_outage')
        for phase in names:
            counts=self.by_time[phase]
            counts['decisions']+=1
            counts['replanned']+=int(choice.replanned)
            counts['timed_out']+=int(choice.timed_out)
            counts['fallback_decisions']+=int(choice.kind=='fallback')
            counts['branch/'+(choice.matching_kind or choice.kind)]+=1
            if choice.fallback_reason:
                counts['fallback_reason/'+choice.fallback_reason]+=1
            if choice.plan_error:
                counts['plan_error/'+choice.plan_error]+=1
        return choice.action

    def finish(self,spec,config_sha):
        # Called only after native and Python seals agree on count/actor identity.
        self.commit_previous()
        summary=self.engine.summary()
        counts=dict(summary['branch_counts'])
        trial_service.require(sum(counts.values())==self.by_time['total']['decisions'],'grade-flow counter mismatch')
        return dict(run_index=spec['run_index'],recording_id=spec['recording_id'],
                    config_sha256=config_sha,policy_digest=spec['checkpoint_sha256'],
                    plan_versions=summary['plan_versions'],planned_branches=counts,
                    solver_outcomes=summary['accepted_solver_outcomes'],grade_control=summary['grade_control'],
                    phases={k:dict(v) for k,v in self.by_time.items()})


ACTORS={}


class BatchTrial(ScaledTrial):
    def __init__(self,label):
        import batch_campaign as campaign
        spec,self.sha,self.edges=campaign.read_case(label)
        self.spec=spec
        self.output=Path(spec['python_output_directory'])
        self.events=Path(spec['event_directory'])
        strategy=spec['strategy']
        trial_service.require(spec['layout_id']=='tree_A' and spec['actor_sampling_seed'] is None
                              and spec['slot_permutation_seed']==0 and
                              all(spec[k]==v for k,v in pins.policy_metadata(strategy).items()),
                              'invalid dispatch policy binding')
        trial_service.require(not self.events.exists() and not (self.output/'inference.jsonl').exists()
                              and not (self.output/'inference_manifest.json').exists(),'physical identity already started')
        self.active=sum(1<<r[0] for r in spec['expected_runtime_scene']['opening_fleet'] if r[1])
        trial_service.require(self.active==511,'expected nine A trucks')
        if strategy=='flow':
            self.policy=FlowActor(spec)
        else:
            if strategy not in ACTORS:
                factory=FullActor if strategy=='full' else FlatActor
                ACTORS[strategy]=factory(spec,self.edges)
            self.policy=ACTORS[strategy]
        self.sequence=0
        self.previous_time=-1.
        self.reply=self.last_key=self.writer=None
        self.sealed=False

    def decide(self,request):
        fault=self.spec['outage']
        t=request['sim_time_seconds']
        closed=fault is not None and fault['start_seconds']<=t<fault['end_seconds']
        obs=request['obs']
        trial_service.require(obs[16]==0 and obs[1107]==int(not closed),'native high-shovel availability differs')
        if closed:
            trial_service.require(request['action_mask'][0]==obs[19]==obs[23]==obs[1114]==0,
                                  'unavailable shovel is dispatchable')
        if isinstance(self.policy,FlowActor):
            self.policy.t=t
        return super().decide(request)

    def try_seal(self):
        was_complete=self.sealed
        super().try_seal()
        if not was_complete and self.sealed and isinstance(self.policy,FlowActor):
            diagnostics=self.policy.finish(self.spec,self.sha)
            path=self.output/'flow_diagnostics.json'
            record(path,diagnostics)
            seal=dict(run_index=self.spec['run_index'],recording_id=self.spec['recording_id'],
                      config_sha256=self.sha,policy_digest=self.spec['checkpoint_sha256'],
                      diagnostic=pins.pin(path),inference_manifest=pins.pin(self.output/'inference_manifest.json'),
                      native_manifest=pins.pin(self.events/'events_manifest.json'))
            record(self.output/'flow_diagnostics_seal.json',seal)
            self.policy=None


if __name__=='__main__':
    import batch_campaign as campaign
    labels=sys.argv[1:] or campaign.case_labels()
    trial_service.require(len(set(labels))==len(labels) and all(x in campaign.case_labels() for x in labels),
                          'unknown or repeated case label')
    trial_service.Trial=BatchTrial
    trial_service.serve(labels)
