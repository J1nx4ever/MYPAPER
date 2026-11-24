"""Sequential AnyLogic campaign. One service, one frozen policy per complete 32-shift batch."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time

import grpc
import torch

from resilience_rpc import Shift, wire
from python.generated import dispatch_pb2_grpc
from python.reward_config import require

REVISION = 'boundary-events-v1-native-rpc-runtime-scene-v1'


def atomic(path, data, *, replace=False):
    """Publish complete bytes, with exclusive creation for immutable records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        name = Path(f.name)
        try:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            name.unlink(missing_ok=True)
            raise
    try:
        if replace:
            os.replace(name, path)
        else:
            os.link(name, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        name.unlink(missing_ok=True)


def record(path, obj, *, replace=False):
    atomic(path, (json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n').encode(),
           replace=replace)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Handshake:
    def __init__(self, suite, job, master_row, claim_budget):
        self.suite, self.job, self.row = suite, job, master_row
        self.control = Path(suite['control_directory'])
        self.claim_budget = claim_budget
        self.claim_file = self.control/'java_claims'/f"run_{job['run_index']}.once.claim"
        self.ready_file = self.control/'ready'/f"run_{job['run_index']}.properties"
        self.claimed = False
        self.lock = threading.Lock()

    def publish(self):
        j = self.job
        cfg = Path(j['config_path']).resolve()
        require(sha(cfg) == j['config_sha256'] and self.row['run_index'] == str(j['run_index'])
                and self.row['recording_id'] == j['recording_id'], 'ready row or config identity differs')
        require(not self.claim_file.exists() and not (self.control/'STOP.properties').exists(),
                'shift already claimed or campaign stopped')
        props = dict(self.row, protocol='baseline1-one-click-ready-v1', campaign_id=self.suite['campaign_id'],
                     instrumentation_revision=REVISION, master_plan_sha256=self.suite['master_plan']['sha256'],
                     config_sha256=j['config_sha256'], snapshot_sha256=j['config_sha256'], snapshot_path=str(cfg))
        require(all(not any(ch in str(v) for ch in '\r\n\\') for v in props.values()), 'invalid ready property')
        content = ''.join(f'{key}={value}\n' for key,value in props.items()).encode('ascii')
        if self.ready_file.exists():
            require(self.ready_file.read_bytes() == content, 'previously published ready bytes differ')
        else:
            atomic(self.ready_file, content)

    def claim_bytes(self):
        j = self.job
        expected = dict(protocol='baseline1-one-click-java-start-v1', campaign_id=self.suite['campaign_id'],
                        master_plan_sha256=self.suite['master_plan']['sha256'], ordinal=str(j['ordinal']),
                        run_index=str(j['run_index']), snapshot_sha256=j['config_sha256'],
                        config_sha256=j['config_sha256'])
        return ''.join(f'{key}={value}\n' for key,value in expected.items()).encode('ascii')

    def claim(self, *, final=False):
        with self.lock:
            if self.claimed:
                return True
            if not self.claim_file.exists():
                require(not final, 'RPC received before exclusive Java start claim')
                return False
            actual = self.claim_file.read_bytes()
            expected = self.claim_bytes()
            if not final and len(actual) < len(expected) and expected.startswith(actual):
                return False
            require(actual == expected, 'Java physical start identity differs')
            self.claim_budget(self.suite, self.job)
            self.claimed = True
            return True


class LiveRPC(dispatch_pb2_grpc.DispatchServiceServicer):
    def __init__(self):
        self.lock = threading.RLock()
        self.shift = self.handshake = None
        self.finished = threading.Event()
        self.failure = None
        self.terminals = {}

    def arm(self, shift, handshake):
        with self.lock:
            self.shift = shift
            self.handshake = handshake
            self.finished.clear()

    def GetNextDestination(self, request, context):
        with self.lock:
            try:
                if request.kind == wire.EPISODE_END:
                    old = self.terminals.get((request.experiment_run_index, request.request_id))
                    if old:
                        digest = hashlib.sha256(request.SerializeToString(deterministic=True)).digest()
                        require(old[0] == digest, 'terminal retry payload changed')
                        return old[1]
                require(self.shift is not None and self.failure is None, 'campaign not ready for RPC')
                self.handshake.claim(final=True)
                response = self.shift.accept(request)
                if request.kind == wire.EPISODE_END:
                    self.terminals[(request.experiment_run_index, request.request_id)] = self.shift.processed[request.request_id]
                    self.finished.set()
                return response
            except Exception as exc:
                self.failure = exc
                self.finished.set()
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))


def recovered_prefix(suite, budget, rows):
    """Only resume after a sealed optimizer update or a completed frozen evaluation."""
    jobs = suite['jobs']
    root = Path(suite['output'])
    with sqlite3.connect(budget.ledger_path(suite).as_uri()+'?mode=ro', uri=True) as db:
        ledger = {run:(state,evidence) for run,state,evidence in db.execute('SELECT run,status,evidence FROM slots')}
    require(len(ledger) == len(jobs), 'resume ledger length differs')
    offset = 0
    updates = dict(warmup=0, static=0, dynamic=0)
    pending = []
    for j,row in zip(jobs,rows,strict=True):
        ordinal, run = j['ordinal'], j['run_index']
        status,evidence = ledger[run]
        hand = Handshake(suite,j,row,budget.claim)
        receipt = Path(j['python_output'])/'receipt.json'
        if status == 'PLANNED':
            require(not hand.claim_file.exists() and not receipt.exists() and
                    (ordinal == offset or not hand.ready_file.exists()), 'unsettled physical run or future ready file')
            continue
        require(ordinal == offset and status == 'COMPLETE' and hand.claim_file.read_bytes() == hand.claim_bytes(),
                'cannot resume a consumed or out-of-order physical run')
        require(json.loads(evidence) == dict(path=str(receipt),sha256=sha(receipt)), 'completed receipt digest differs')
        body = json.loads(receipt.read_bytes())
        for key in ('recording_id','run_index','experiment_seed','actor_sampling_seed','config_sha256','role','arm','update'):
            require(body[key] == j[key], 'completed receipt identity differs: '+key)
        if j['role'] == 'train':
            pending.append(run)
            if len(pending) == 32:
                arm, u = j['arm'], j['update']
                folder = root/'models'/arm/f'u{u:04d}'
                intent = json.loads((folder/'intent.json').read_bytes())
                settled = json.loads((folder/'update.json').read_bytes())
                checkpoint = folder/'checkpoint.pt'
                manifest_path = folder/'checkpoint.pt.manifest.json'
                manifest = json.loads(manifest_path.read_bytes())
                seal = json.loads((folder/'checkpoint.pt.seal.json').read_bytes())
                key = f"{suite['campaign_id']}:{arm}:{u}"
                require(u == updates[arm]+1 and intent['batch_id'] == settled['summary']['batch_id'] == key
                        and settled['summary']['update_index'] == manifest['update_index'] == u
                        and intent['runs'] == pending and checkpoint.stat().st_size == manifest['bytes']
                        and seal['manifest_sha256'] == sha(manifest_path)
                        and sha(checkpoint) == settled['checkpoint_sha256'] == manifest['sha256'] == seal['sha256'],
                        'unsealed training boundary')
                updates[arm] = u
                pending.clear()
        else:
            require(not pending, 'evaluation after incomplete training batch')
        offset += 1
    require(not pending, 'partial training batch cannot restore its in-memory rollouts')
    return offset, updates


def restore_arm(suite, arm, updates, factory, learner):
    base = Path(suite['output'])/'models'/arm
    u = updates[arm]
    path = base/f'u{u:04d}'/'checkpoint.pt'
    if path.exists():
        model = learner.load(path)
        require(model.update_index == u and (not u or f"{suite['campaign_id']}:{arm}:{u}" in model.consumed_batches),
                'restored checkpoint does not match completed batch')
        if not u:
            require(model.policy_version == factory().policy_version, 'initial checkpoint differs from lineage')
        return model
    require(u == 0, 'missing learned checkpoint')
    path.parent.mkdir(parents=True, exist_ok=False)
    model = factory()
    model.save(path)
    return model


def serve(suite, *, address='127.0.0.1:50051', budget=None, verify_physical=None):
    from fleet_policy import FleetLearner
    from physical import verify_terminal
    from reports import make_report
    if budget is None:
        import resilience_budget as budget

    verify_physical = verify_physical or verify_terminal
    jobs = suite['jobs']
    root = Path(suite['output'])
    master = suite['master_plan']
    require(sha(master['path']) == master['sha256'], 'master digest mismatch')
    with open(master['path'], encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len(jobs) and all(int(row['ordinal']) == i for i,row in enumerate(rows)),
            'master does not cover planned jobs')
    torch.set_num_threads(2)
    batch = []
    batch_key = None
    service = LiveRPC()
    server = grpc.server(ThreadPoolExecutor(max_workers=3), options=[('grpc.so_reuseport', 0)])
    dispatch_pb2_grpc.add_DispatchServiceServicer_to_server(service, server)
    require(server.add_insecure_port(address) != 0, 'unable to bind campaign RPC port')
    server.start()
    handshake = job = None
    try:
        budget.register(suite)
        offset,arm_updates = recovered_prefix(suite,budget,rows)
        models = {'warmup':restore_arm(suite, 'warmup', arm_updates,
                            lambda: FleetLearner.warm_start(suite['learner_seed']), FleetLearner)}
        for job,row in zip(jobs[offset:], rows[offset:], strict=True):
            arm = job['arm']
            if arm not in models:
                require(not batch and arm in ('dynamic','static') and arm_updates['warmup'] > 0,
                        'branch without completed warmup')
                models[arm] = restore_arm(suite, arm, arm_updates,
                                          lambda: models['warmup'].fork(suite['branch_seed']), FleetLearner)
            model = models[arm]
            key = arm, job['update']
            training = job['role'] == 'train'
            require((not batch or key == batch_key) and job['update'] == arm_updates[arm]+int(training),
                    'batch boundary or update index differs')
            shift = Shift(job, model, verify_physical)
            handshake = Handshake(suite, job, row, budget.claim)
            service.arm(shift, handshake)
            handshake.publish()
            record(root/'run_status.json', dict(state='WAITING_FOR_SHIFT', run_index=job['run_index'],
                   ordinal=job['ordinal'], arm=arm, update=job['update']), replace=True)
            clock_start = time.monotonic()
            while not service.finished.wait(.5):
                handshake.claim()
                require(not (handshake.control/'STOP.properties').exists(), 'campaign STOP')
                require(time.monotonic()-clock_start < 3600, 'shift waiting/collection timed out')
            with service.lock:
                if service.failure is not None:
                    raise service.failure
                require(handshake.claimed and shift.receipt is not None, 'incomplete physical shift')
                receipt = Path(job['python_output'])/'receipt.json'
                record(receipt, shift.receipt)
                budget.complete(suite, job, receipt)
                service.shift = None
            if training:
                batch_key = key
                batch.append(shift.episode)
                if len(batch) == 32:
                    folder = root/'models'/arm/f"u{job['update']:04d}"
                    folder.mkdir(parents=True, exist_ok=False)
                    batch_id = f"{suite['campaign_id']}:{arm}:{job['update']}"
                    record(folder/'intent.json', dict(batch_id=batch_id, policy_before=batch[0]['policy_version'],
                           runs=[e['identity']['run_index'] for e in batch]))
                    summary = model.update(batch, batch_id=batch_id)
                    model.save(folder/'checkpoint.pt')
                    record(folder/'update.json', dict(checkpoint_sha256=sha(folder/'checkpoint.pt'), summary=summary))
                    arm_updates[arm] += 1
                    batch.clear()
            else:
                require(not batch, 'evaluation encountered an unfinished batch')
        require(not batch, 'campaign ended with incomplete training batch')
        make_report(suite)
        marker = root/'suite_complete.json'
        result = dict(physical_shifts=len(jobs), updates=arm_updates,
                      campaign_id=suite['campaign_id'], master_plan_sha256=master['sha256'])
        if marker.exists():
            require(json.loads(marker.read_bytes()) == result, 'suite completion identity differs')
        else:
            record(marker, result)
        record(root/'run_status.json', dict(state='COMPLETE', updates=arm_updates, physical_shifts=len(jobs)),
               replace=True)
    except BaseException as exc:
        if handshake is not None and handshake.job is job and handshake.claimed:
            if not (Path(job['python_output'])/'receipt.json').is_file():
                budget.unknown(suite, job, str(exc))
        raise
    finally:
        server.stop(2).wait()
