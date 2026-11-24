"""Recovery training with a sealed source model and the original Java/RPC handshake."""
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import sys
import time

import grpc
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resilience_runtime import Handshake, LiveRPC, Shift, dispatch_pb2_grpc, record, require, restore_arm, sha
from resilience_campaign import PLAN, ROOT, reference
from physical import verify_terminal
from reports import summarize


def source_model(suite, learner):
    source = suite['source_u0']
    path = Path(source['path'])
    require(sha(path) == source['sha256'], 'source u0 file differs from sealed recovery plan')
    model = learner.load(path)
    require(model.update_index == 0 and not model.consumed_episodes and not model.consumed_batches
            and model.config['seed'] == suite['learner_seed'], 'source model is not the expected starting u0')
    return model


def settle_shift(suite, job, shift, model, budget):
    """Save actual physical completion even when a training consistency check fails."""
    receipt = Path(job['python_output'])/'receipt.json'
    if job['role'] == 'train':
        try:
            model.verify_collected(shift.episode)
        except Exception as error:
            shift.receipt['likelihood_replay'] = dict(passed=False,error=str(error))
            record(receipt, shift.receipt)
            budget.complete(suite, job, receipt)
            raise
        shift.receipt['likelihood_replay'] = dict(passed=True,decisions=len(shift.episode['actions']))
    record(receipt, shift.receipt)
    budget.complete(suite, job, receipt)


def make_report(suite, budget):
    states = budget.state_view(suite)
    for job in suite['jobs']:
        state, pin = states[job['run_index']]
        receipt = Path(job['python_output'])/'receipt.json'
        require(state == 'COMPLETE' and json.loads(pin) == reference(receipt),
                'physical receipt is incomplete or altered: '+str(job['run_index']))
    report = summarize(suite)
    archive = ROOT/'reports'/PLAN.parent.parent.name/PLAN.parent.name/PLAN.name/suite['campaign_id']/'evaluation.json'
    if archive.exists():
        require(json.loads(archive.read_bytes()) == report,'existing paired evaluation differs')
    else:
        record(archive,report)
    return report


def recovered_prefix(suite, budget, rows):
    """Only resume after a sealed optimizer update or a completed frozen evaluation."""
    jobs = suite['jobs']
    root = Path(suite['output'])
    ledger = budget.state_view(suite)
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
            require(body.get('likelihood_replay',{}).get('passed') is True, 'completed shift failed likelihood replay')
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


def serve(suite, *, address='127.0.0.1:50051', budget=None, verify_physical=verify_terminal):
    from replay_policy import FleetLearner
    if budget is None:
        import recovery_budget as budget
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
                            lambda: source_model(suite,FleetLearner), FleetLearner)}
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
                settle_shift(suite, job, shift, model, budget)
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
        make_report(suite, budget)
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
