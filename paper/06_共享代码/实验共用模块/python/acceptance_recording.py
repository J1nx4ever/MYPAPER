"""Opt-in boundary evidence; records RPC bytes and actual float32 env inputs.

This format is acceptance evidence, not a PPO batch or a training dataset.
No simulator is launched and no policy/value/multiplier is updated here.
"""
import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from collections import Counter

import numpy as np

from .observation_contract import FIELD_NAMES, validate_observation

SCHEMA = 'boundary-recording-v1'


def config_bytes(config):
    return (json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()


def config_sha256(config):
    return hashlib.sha256(config_bytes(config)).hexdigest()


def validate_recording_options(config, *, enabled=False, legacy_trace=False, capture=False):
    spec = config.get('recording', {})
    if not isinstance(spec, dict):
        raise ValueError('recording must be an object')
    if spec and type(spec.get('required')) is not bool:
        raise ValueError('recording.required must be an explicit boolean')
    if spec.get('required') and not enabled:
        raise ValueError('this configuration requires --record-boundary-evidence')
    if enabled:
        if spec.get('schema') != SCHEMA or not isinstance(spec.get('recording_id'), str) or not spec['recording_id'].strip():
            raise ValueError('boundary recording requires explicit recording.schema and recording_id')
        if legacy_trace or capture:
            raise ValueError('--record-boundary-evidence already captures observations; do not mix legacy trace flags')
    elif capture and not legacy_trace:
        raise ValueError('capture_observations requires explicit record_transitions')


def describe_recording(config):
    """Configuration-only preview, never a run approval or a listener."""
    validate_recording_options(config, enabled=True)
    return dict(schema=SCHEMA, recording_id=config['recording']['recording_id'],
                config_sha256=config_sha256(config),
                acceptance_run_approved=config.get('acceptance_run_approved') is True,
                listener_started=False, collection_started=False,
                files=['rpc.jsonl','transitions.jsonl','recording_manifest.json'],
                note='Changing any config byte-equivalent value changes this identity; refresh the simulator audit declaration.')


class BoundaryRecorder:
    """Thread-safe journal. A complete marker requires normal complete shifts."""
    def __init__(self, output, config, jobs, *, source_kind='actual_anylogic_acceptance', source_sha256=None,execution_options=None):
        validate_recording_options(config, enabled=True)
        if source_kind not in ('actual_anylogic_acceptance','actual_anylogic_offline','synthetic_unit_test'):
            raise ValueError('unsupported recording source kind')
        self.output = Path(output)
        expected = config_bytes(config)
        path = self.output/'acceptance_run_config.json'
        if not path.is_file() or path.read_bytes() != expected:
            raise ValueError('recording requires exact acceptance_run_config.json bytes before opening')
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.request_ids = set()
        self.response_ids = set()
        self.request_counts = Counter()
        self.response_counts = Counter()
        self.files = {}
        self.counts = {'rpc.jsonl':0, 'transitions.jsonl':0}
        self.active = None
        self.total_reward = 0.
        self.steps = 0
        self.last_terminal = False
        self.closed = False
        self.finished = []
        self.jobs = copy.deepcopy(list(jobs))
        if not self.jobs:
            raise ValueError('boundary recording requires an explicit nonempty job manifest')
        self.manifest = dict(schema=SCHEMA,scope=source_kind,source_kind=source_kind,
            recording_id=config['recording']['recording_id'],config_sha256=hashlib.sha256(expected).hexdigest(),
            status='recording',complete=False,formal_training_data=False,
            expected_jobs=self.jobs,episodes=[],files={},source_sha256=source_sha256 or {},
            observation_dtype='float32',observation_width=1519,consumer='acceptance_rule_no_network',
            rpc_payload_encoding='protobuf_deterministic_reserialization_not_network_frames',
            field_names_sha256=hashlib.sha256(('\n'.join(FIELD_NAMES)+'\n').encode()).hexdigest(),
            simulator_event_evidence='separate_sidecar_required_for_physical_recomputation')
        if source_kind == 'actual_anylogic_offline':
            self.manifest.update(consumer='offline_behavior_no_updates', behavior=copy.deepcopy(config['behavior']))
        if execution_options is not None:
            data=(self.output/'execution_options.json').read_bytes()
            if json.loads(data)!=execution_options:raise ValueError('execution options file mismatch')
            self.manifest.update(execution_options=copy.deepcopy(execution_options),
                                 execution_options_sha256=hashlib.sha256(data).hexdigest())
        try:
            # Exclusive creation: never truncate prior evidence.
            with (self.output/'recording_manifest.json').open('x') as f:
                json.dump(self.manifest,f,ensure_ascii=False,indent=2,allow_nan=False)
                f.write('\n')
            for name in self.counts:
                self.files[name]=(self.output/name).open('x',encoding='utf-8')
        except BaseException:
            for f in self.files.values(): f.close()
            raise

    def _append(self, name, event, values):
        with self.lock:
            if self.closed:
                raise RuntimeError('recording is closed')
            row=dict(record_seq=self.counts[name],event=event,**values)
            payload=json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)
            self.files[name].write(payload+'\n')
            self.files[name].flush()
            self.counts[name]+=1

    def rpc_received(self, request):
        data=request.SerializeToString(deterministic=True)
        with self.condition:
            self._append('rpc.jsonl','request',dict(episode_id=request.episode_id,request_id=request.request_id,
                received_monotonic_ns=time.perf_counter_ns(),payload_base64=base64.b64encode(data).decode(),
                payload_sha256=hashlib.sha256(data).hexdigest()))
            self.request_ids.add(request.request_id)
            self.request_counts[request.request_id]+=1

    def rpc_response(self, request, response):
        data=response.SerializeToString(deterministic=True)
        with self.condition:
            self._append('rpc.jsonl','response',dict(episode_id=request.episode_id,request_id=request.request_id,
                payload_base64=base64.b64encode(data).decode(),payload_sha256=hashlib.sha256(data).hexdigest()))
            self.response_ids.add(request.request_id)
            self.response_counts[request.request_id]+=1
            self.condition.notify_all()

    def rpc_rejected(self, request, error):
        self._append('rpc.jsonl','rejected',dict(episode_id=request.episode_id,request_id=request.request_id,
            reason=str(error),exception_type=type(error).__name__))

    @staticmethod
    def _observation(obs, info, *, terminal=False):
        values=np.asarray(obs)
        if values.dtype != np.float32 or values.shape != (1519,):
            raise ValueError('record the actual 1519D float32 environment array; no projection or recasting')
        validate_observation(info['observation_schema_id'],values,terminal=terminal,truck_id=info['truck_id'])
        return values.tolist()

    def start_episode(self, job, observation, info):
        with self.lock:
            if self.active is not None or len(self.finished)>=len(self.jobs):
                raise ValueError('cannot overwrite an active/excess episode')
            if job != self.jobs[len(self.finished)]:
                raise ValueError('episode job differs from recording manifest')
            values=self._observation(observation,info)
            self.active=info['episode_id'];self.steps=0;self.total_reward=0.
            self.last_terminal=False
            self._append('transitions.jsonl','episode_start',dict(episode_id=self.active,job=copy.deepcopy(job),
                observation=values,info=copy.deepcopy(info)))

    def action(self, observation, info, action):
        with self.lock:
            if info['episode_id'] != self.active:
                raise ValueError('action has no matching active episode')
            if type(action) is not int or not 0<=action<4 or not info['action_mask'][action]:
                raise ValueError('cannot record an illegal action')
            self._append('transitions.jsonl','action',dict(episode_id=self.active,
                decision_sequence=info['decision_sequence'],request_id=info['request_id'],action=action,
                action_mask=list(info['action_mask']),observation=self._observation(observation,info)))

    def transition(self, obs, info, action, next_obs, env_reward, next_info, result):
        with self.lock:
            if info['episode_id'] != self.active or next_info['episode_id'] != self.active:
                raise ValueError('transition crossed episode identity')
            if not all(math.isfinite(float(x)) for x in (env_reward,next_info['transition_reward_raw'],result['reward'])):
                raise ValueError('nonfinite reward cannot enter a successful transition record')
            self._append('transitions.jsonl','transition',dict(episode_id=self.active,step_index=self.steps,
                action=action,action_mask=list(info['action_mask']),observation=self._observation(obs,info),
                next_observation=self._observation(next_obs,next_info,terminal=result['terminated']),
                action_info=copy.deepcopy(info),next_info=copy.deepcopy(next_info),
                env_reward=float(env_reward),raw_reward=float(next_info['transition_reward_raw']),
                formal_reward=float(result['reward']),
                **{k:copy.deepcopy(result[k]) for k in ('grade_costs','grade_valid','grade_bootstrap','grade_trace',
                    'next_grade_valid','reward_bootstrap','reward_trace','terminated','truncated',
                    'abnormal_interruption','sampler_truncated')}))
            self.steps+=1;self.total_reward+=float(result['reward'])
            self.last_terminal=bool(result['terminated'] and not result['truncated'] and not result['abnormal_interruption'])

    def finish_episode(self, job, final_info):
        with self.lock:
            if self.active != final_info['episode_id'] or job!=self.jobs[len(self.finished)] or self.steps<=0:
                raise ValueError('invalid episode completion')
            if final_info['sim_time_seconds']!=28800 or not self.last_terminal:
                raise ValueError('only true shift end can finish an episode')
            row=dict(episode_id=self.active,job=copy.deepcopy(job),transitions=self.steps,
                total_reward=self.total_reward,final_info=copy.deepcopy(final_info),normal_completion=True)
            self._append('transitions.jsonl','episode_end',row)
            self.finished.append({k:copy.deepcopy(row[k]) for k in ('episode_id','job','transitions','total_reward','normal_completion')})
            self.active=None

    def _seal(self, status, error=None):
        for f in self.files.values(): f.flush();f.close()
        self.manifest.update(status=status,complete=status=='complete',episodes=copy.deepcopy(self.finished),
            files={name:dict(records=self.counts[name],sha256=hashlib.sha256((self.output/name).read_bytes()).hexdigest())
                   for name in self.counts})
        if error is not None:self.manifest['error']=error
        temp=self.output/'recording_manifest.json.tmp'
        temp.write_text(json.dumps(self.manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        temp.replace(self.output/'recording_manifest.json')
        self.closed=True

    def complete(self):
        with self.condition:
            if self.closed or self.active is not None or len(self.finished)!=len(self.jobs):
                raise ValueError('cannot seal an incomplete recording')
            # Terminal ACK is prepared by an RPC worker; the sampler can finish
            # its episode first. Do not close the log before that worker writes.
            if not self.request_ids:
                raise ValueError('cannot seal recording without RPC evidence')
            if not self.condition.wait_for(lambda: all(self.response_counts[k]>=n for k,n in self.request_counts.items()), timeout=5):
                raise ValueError('cannot seal recording with missing RPC responses')
            self._seal('complete')

    def abort(self, error, context=None):
        with self.lock:
            if self.closed:return
            self._append('transitions.jsonl','episode_aborted',dict(episode_id=self.active,
                reason=str(error),exception_type=type(error).__name__,context=copy.deepcopy(context)))
            self._seal('aborted',dict(reason=str(error),exception_type=type(error).__name__,context=context))
