"""Serve the frozen policy to the two local AnyLogic trial experiments."""

import argparse
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from compile_layouts import compile_layout, load_layouts


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_case(case):
    path = HERE / 'configs' / f'{case}.json'
    raw = path.read_bytes()
    config_sha = sha256(raw).hexdigest()
    require(path.with_suffix('.sha256').read_text().strip() == config_sha, 'changed trial configuration')
    spec = json.loads(raw)
    require(spec['purpose'] == 'topology_frozen_policy_trial' and spec['action_mode'] == 'argmax',
            'invalid trial purpose or execution mode')
    require(spec['inference_port'] == 50871 and spec['slot_permutation_seed'] == 0,
            'unexpected server port or slot permutation seed')
    require(digest(Path(spec['checkpoint_path'])) == spec['checkpoint_sha256'], 'checkpoint bytes changed')
    sources = spec['inference_sources']
    expected_sources = {HERE / 'trial_service.py', HERE / 'frozen_topology_policy.py',
                        HERE.parent / 'topology_policy.py',
                        HERE.parents[1] / 'fleet_outage_resilience_001' / 'fleet_policy.py'}
    require({Path(pin['path']) for pin in sources} == expected_sources and
            all(digest(Path(pin['path'])) == pin['sha256'] for pin in sources),
            'inference adapter sources differ from the frozen configuration')
    layouts = {layout['id']: layout for layout in load_layouts()}
    scene, edges = compile_layout(layouts[spec['layout_id']])
    require(sha256(scene).hexdigest() == spec['layout_sha256'], 'changed trial layout')
    return spec, config_sha, edges


class Trial:
    def __init__(self, case):
        from frozen_topology_policy import FrozenTopologyPolicy, CHECKPOINT_SHA256
        self.spec, self.sha, self.edges = read_case(case)
        spec = self.spec
        require(spec['checkpoint_sha256'] == CHECKPOINT_SHA256, 'policy does not match the frozen configuration')
        self.policy = FrozenTopologyPolicy(Path(spec['checkpoint_path']), spec['layout_id'],
                                           actor_sampling_seed=spec['slot_permutation_seed'])
        self.output = Path(spec['python_output_directory'])
        self.events = Path(spec['event_directory'])
        self.active = sum(1 << row[0] for row in spec['expected_runtime_scene']['opening_fleet'] if row[1])
        self.sequence = 0
        self.previous_time = -1.
        self.last_key = self.reply = self.writer = None
        self.sealed = False
        require(not (self.output / 'inference.jsonl').exists() and
                not (self.output / 'inference_manifest.json').exists() and not self.events.exists(),
                'frozen identity already has native or inference evidence')

    def decide(self, request):
        from topology_policy import require_layout
        spec = self.spec
        require(not self.sealed, 'shift already sealed')
        for name in ('run_index', 'environment_seed', 'recording_id', 'layout_id'):
            require(request.get(name) == spec[name], f'wrong trial {name}')
        require(request.get('config_sha256') == self.sha, 'wrong configuration SHA')
        step, truck, now = (request[k] for k in ('decision_sequence', 'truck_id', 'sim_time_seconds'))
        require(type(step) is int and type(truck) is int and 0 <= truck < 9,
                'invalid decision ordinal or truck')
        require(isinstance(now, (int, float)) and math.isfinite(now) and 0 <= now < 28800,
                'invalid decision time')
        dispatchable, present = (request[k] for k in ('dispatchable_mask', 'in_system_mask'))
        require(type(dispatchable) is int and type(present) is int and
                dispatchable == self.active == present and ((1 << truck) & dispatchable),
                'unexpected runtime fleet or ineligible truck')
        mask = request['action_mask']
        require(type(mask) is list and len(mask) == 4 and
                all(type(value) is int and value in (0, 1) for value in mask) and any(mask),
                'invalid physical dispatch mask')
        obs = np.asarray(request['obs'], dtype=np.float32)
        require(obs.shape == (1519,) and np.isfinite(obs).all(), 'incomplete or nonfinite native observation')
        require(np.array_equal(obs[19:23], mask), 'action mask differs from physical observation')
        require(abs(float(obs[0]) - now / 28800) < 1e-6 and obs[17] == 0, 'observation clock mismatch')
        require_layout(obs[None], self.edges)
        fingerprint = sha256(obs.astype('<f4', copy=False).tobytes()).hexdigest()
        key = (step, truck, now, tuple(mask), fingerprint)
        if key == self.last_key:
            return self.reply
        require(step == self.sequence and now >= self.previous_time, 'decision out of sequence')
        action = int(self.policy.action(obs, mask, dispatchable, present, truck))
        require(0 <= action < 4 and mask[action] == 1, 'actor returned an ineligible action')
        reply = dict(action=action, decision_sequence=step, config_sha256=self.sha,
                     checkpoint_sha256=spec['checkpoint_sha256'])
        audit = dict(reply, truck_id=truck, sim_time_seconds=now, observation_sha256=fingerprint,
                     run_index=spec['run_index'], recording_id=spec['recording_id'],
                     action_mask=mask, dispatchable_mask=dispatchable, in_system_mask=present)
        if self.writer is None:
            self.output.mkdir(parents=True, exist_ok=True)
            self.writer = (self.output / 'inference.jsonl').open('xb')
        self.writer.write(encode(audit))
        self.writer.flush()
        self.sequence += 1
        self.previous_time = now
        self.last_key, self.reply = key, reply
        return reply

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def try_seal(self):
        if self.sealed or self.sequence == 0:
            return
        manifest_path = self.events / 'events_manifest.json'
        if not manifest_path.is_file():
            return
        try:
            manifest = json.loads(manifest_path.read_bytes())
        except json.JSONDecodeError:
            return  # Native recorder is publishing this file.
        if not manifest.get('complete'):
            return
        spec = self.spec
        require(manifest['terminated'] and not manifest['truncated'] and
                manifest['source_kind'] == 'actual_anylogic' and
                manifest['recording_id'] == spec['recording_id'] and manifest['seed'] == spec['environment_seed']
                and manifest['run_index'] == spec['run_index'] and manifest['config_sha256'] == self.sha
                and manifest['native_policy_decisions'] == self.sequence and
                manifest['native_checkpoint_sha256'] == spec['checkpoint_sha256'],
                'native seal differs from the policy that answered')
        self.close()
        receipt = dict(complete=True, source_kind='frozen_actor_http', run_index=spec['run_index'],
                       recording_id=spec['recording_id'], config_sha256=self.sha,
                       checkpoint_sha256=spec['checkpoint_sha256'], decision_count=self.sequence,
                       inference_sha256=digest(self.output / 'inference.jsonl'))
        with (self.output / 'inference_manifest.json').open('xb') as stream:
            stream.write(encode(receipt))
        self.sealed = True
        print(f"Sealed {spec['layout_id']} with {self.sequence} decisions", flush=True)


def serve(cases):
    torch.set_num_threads(1)
    trials = [Trial(case) for case in cases]
    by_run = {trial.spec['run_index']: trial for trial in trials}
    require(len(by_run) == len(trials), 'duplicate run identity')

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, data):
            raw = encode(data)
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            try:
                require(self.path == '/decide', 'unknown RPC method')
                size = int(self.headers.get('Content-Length', 0))
                require(0 < size <= 128 * 1024, 'invalid request length')
                request = json.loads(self.rfile.read(size))
                trial = by_run.get(request.get('run_index'))
                require(trial is not None, 'unexpected run index')
                reply = trial.decide(request)
            except Exception as error:
                print(f'Policy request rejected: {error!r}', file=sys.stderr, flush=True)
                self.send_json(422, {'error': str(error)})
                return
            self.send_json(200, reply)

        def log_message(self, fmt, *args):
            pass

    try:
        with HTTPServer(('127.0.0.1', 50871), Handler) as server:
            server.timeout = 0.5
            print('Policy ready on 127.0.0.1:50871 for ' + ', '.join(cases), flush=True)
            next_check = 0.
            while not all(trial.sealed for trial in trials):
                server.handle_request()
                if time.monotonic() >= next_check:
                    for trial in trials:
                        trial.try_seal()
                    next_check = time.monotonic() + 1.
    finally:
        for trial in trials:
            trial.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('A', 'B'), action='append')
    args = parser.parse_args()
    serve(args.case or ['A', 'B'])
