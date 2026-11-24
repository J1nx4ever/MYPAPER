"""Prepare auditable paired initializations; never collect or optimize a model.

Run from the experiment root with ``python -m adaptation_v1_1.prepare
--design path``. A preparation manifest is neither a training result nor an
environment/data admission certificate.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from python.constrained_ppo import build_models, digest


ROOT = Path(__file__).resolve().parents[1]
DESIGN_CONTRACT = 'mining-fixed-actor-adaptation-v1.1-design'
BUNDLE_FORMAT = 'prepared_initialization_not_training_result'
FROZEN_NETWORK = dict(hidden_sizes=[128, 128], activation='relu',
                      initialization='orthogonal_sqrt2_hidden_unit_output',
                      risk_output='sigmoid')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def finite_state(module):
    require(all(bool(torch.isfinite(v).all()) for v in module.state_dict().values()),
            'nonfinite model initialization')


def prepare_pair(spec, seed, payload):
    """Pure initialization operation; tiny network specs support synthetic tests.

    Historical checkpoint configuration/data hashes are checked by ``prepare``;
    this helper verifies the model/Actor boundary without any optimizer object.
    """
    require(payload.get('format') == 'mining-critic-mc-v1.1', 'P3 checkpoint format differs')
    require(type(seed) is int and type(payload.get('seed')) is int and payload['seed'] == seed,
            'P3 checkpoint seed differs')
    require(type(payload.get('epoch')) is int and payload['epoch'] > 0, 'P3 epoch invalid')
    require(payload.get('label_version') == 'mining-reward-mc-v1.1', 'P3 label version differs')
    require(isinstance(payload.get('actor_initial_hash'), str) and
            payload['actor_initial_hash'] == payload.get('actor_final_hash'),
            'historical Actor changed during pretraining')
    actor, p0 = build_models(spec, seed)
    require(digest(actor) == payload['actor_initial_hash'], 'reconstructed Actor digest differs')
    # A separate model prevents any accidental P0/P3 tensor aliasing.
    _, p3 = build_models(spec, seed)
    p3.load_state_dict(payload['critics'], strict=True)
    for model in (actor, p0, p3):
        finite_state(model)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return dict(format=BUNDLE_FORMAT, preparation_only=True, execution_authorized=False,
                training_result=False, admission_certificate=False, seed=seed,
                p3_selected_epoch=payload['epoch'], network=copy.deepcopy(spec),
                actor=state(actor), p0_critics=state(p0), p3_critics=state(p3),
                actor_digest=digest(actor), p0_critics_digest=digest(p0),
                p3_critics_digest=digest(p3),
                historical_actor_initial_hash=payload['actor_initial_hash'],
                historical_actor_final_hash=payload['actor_final_hash'],
                optimizer_states_omitted=True)


def verify_bundle(bundle):
    """Readback check on synthetic masked observations only, never real data."""
    require(bundle.get('format') == BUNDLE_FORMAT and
            bundle.get('execution_authorized') is False and
            bundle.get('training_result') is False and
            bundle.get('admission_certificate') is False and
            bundle.get('optimizer_states_omitted') is True,
            'initialization bundle must remain unapproved and preparation-only')
    spec, seed = bundle['network'], bundle['seed']
    a0, c0 = build_models(spec, seed)
    a3, c3 = build_models(spec, seed)
    a0.load_state_dict(bundle['actor'], strict=True)
    a3.load_state_dict(bundle['actor'], strict=True)
    c0.load_state_dict(bundle['p0_critics'], strict=True)
    c3.load_state_dict(bundle['p3_critics'], strict=True)
    for model in (a0, a3, c0, c3):
        finite_state(model)
        model.eval()
    require(digest(a0) == digest(a3) == bundle['actor_digest'] ==
            bundle['historical_actor_initial_hash'] == bundle['historical_actor_final_hash'],
            'Actor readback digest differs')
    require(digest(c0) == bundle['p0_critics_digest'] and
            digest(c3) == bundle['p3_critics_digest'], 'Critic readback digest differs')
    probe = torch.zeros((2, 1519), dtype=torch.float32)
    probe[:, 19:23] = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]])
    with torch.no_grad():
        l0, l3 = a0(probe), a3(probe)
    valid = probe[:, 19:23].bool()
    require(torch.equal(l0, l3) and bool(torch.isfinite(l0[valid]).all()) and
            bool(torch.isneginf(l0[~valid]).all()), 'paired masked Actor logits differ')
    return dict(model_digest_readback=True, paired_actor_masked_logits_equal=True,
                probe='two synthetic 1519D states; two alternating legal actions each',
                actual_observations_read=0, environment_interactions=0)


def prepare(design_path, *, root=ROOT):
    root = Path(root).resolve()
    design_path = Path(design_path).resolve()
    cfg = read(design_path)
    require(cfg.get('contract') == DESIGN_CONTRACT and
            cfg.get('execution_authorized') is False, 'proposal-only design required')
    require(cfg.get('network') == FROZEN_NETWORK, 'production network differs from frozen architecture')
    sources = {str(design_path): sha(design_path)}

    def pin(path, expected):
        path = Path(path)
        require(path.is_absolute(), 'source path must be absolute')
        path = path.resolve()
        require(isinstance(expected, str) and sha(path) == expected, 'source hash differs: ' + str(path))
        require(str(path) not in sources or sources[str(path)] == expected, 'conflicting source hashes')
        sources[str(path)] = expected
        return path

    require(isinstance(cfg.get('source_pins'), dict) and cfg['source_pins'], 'explicit source pins required')
    for path, expected in cfg['source_pins'].items():
        pin(path, expected)
    refs = {}
    for name in ('pretraining_config', 'formal_label_admission', 'core'):
        ref = cfg[name]
        refs[name] = pin(ref['path'], ref['sha256'])
    historical = read(refs['pretraining_config'])
    require(historical.get('contract') == 'mining-critic-mc-v1.1-execution' and
            historical.get('approved') is True and historical.get('network') == cfg['network'],
            'historical pretraining configuration differs')
    require(type(historical.get('cpu_threads')) is int and historical['cpu_threads'] > 0,
            'historical CPU thread count required for exact initialization reproduction')
    torch.set_num_threads(historical['cpu_threads'])
    require((root / historical['labels'] / 'admission.json').resolve() == refs['formal_label_admission'],
            'pretraining data location differs from formal label admission')
    certificate = read(refs['formal_label_admission'])
    require(certificate.get('contract') == 'mining-reward-mc-v1.1' and
            certificate.get('status') == 'admitted_train_validation_only' and
            certificate.get('diagnostic_only') is False and
            certificate.get('formal_training_data') is True and
            certificate.get('reward', {}).get('beta') == .25,
            'formal V1.1 train/validation admission required')
    pairs = cfg.get('pairs')
    require(isinstance(pairs, list) and pairs, 'explicit paired seeds required')
    require(len({p['seed'] for p in pairs}) == len(pairs), 'duplicate paired seed')
    entries = {r['seed']: r for r in historical['runs']}
    require(set(entries) == {p['seed'] for p in pairs}, 'all pretraining seeds must be retained')
    destination = (root / cfg['preparation_output']).resolve()
    require(destination.is_relative_to(root / 'runs') and destination != root / 'runs',
            'preparation output must be a dedicated runs directory')
    require(not destination.exists(), 'refusing preparation output overwrite')
    prepared = []
    for pair in pairs:
        require(type(pair.get('epoch')) is int and pair['epoch'] > 0, 'explicit selected epoch required')
        ref = pair['p3_checkpoint']
        path = pin(ref['path'], ref['sha256'])
        require(path.parent == (root / entries[pair['seed']]['output']).resolve(),
                'P3 checkpoint source run differs')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        require(payload.get('epoch') == pair['epoch'], 'P3 selected epoch differs')
        require(payload.get('config_sha256') == cfg['pretraining_config']['sha256'] and
                payload.get('data_sha256') == cfg['formal_label_admission']['sha256'],
                'P3 checkpoint configuration/data hash differs')
        require(payload.get('run_id') == entries[pair['seed']]['run_id'], 'P3 run identity differs')
        bundle = prepare_pair(cfg['network'], pair['seed'], payload)
        bundle['source'] = dict(p3_checkpoint=dict(path=str(path), sha256=ref['sha256']),
            pretraining_config=cfg['pretraining_config'], formal_label_admission=cfg['formal_label_admission'],
            core=cfg['core'],
            design=dict(path=str(design_path), sha256=sources[str(design_path)]))
        bundle['checks'] = verify_bundle(bundle)
        prepared.append(bundle)
    # Catch concurrent source modifications before any output and again at end.
    for path, expected in sources.items():
        require(sha(path) == expected, 'source changed during preparation: ' + path)
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for bundle in prepared:
            bundle['source_sha256'] = copy.deepcopy(sources)
            path = destination / ('seed_' + str(bundle['seed']) + '.pt')
            with path.open('xb') as f:
                torch.save(bundle, f)
            recovered = torch.load(path, map_location='cpu', weights_only=True)
            check = verify_bundle(recovered)
            require(all(torch.equal(bundle[group][key], recovered[group][key])
                        for group in ('actor', 'p0_critics', 'p3_critics')
                        for key in bundle[group]), 'saved model tensors changed')
            records.append(dict(seed=bundle['seed'], p3_selected_epoch=bundle['p3_selected_epoch'],
                path=str(path), sha256=sha(path), actor_digest=bundle['actor_digest'],
                p0_critics_digest=bundle['p0_critics_digest'], p3_critics_digest=bundle['p3_critics_digest'],
                source=bundle['source'], checks=check))
        for path, expected in sources.items():
            require(sha(path) == expected, 'source changed during preparation: ' + path)
        manifest = dict(contract='mining-fixed-actor-preparation-v1.1', status='prepared_only',
            execution_authorized=False, training_result=False, admission_certificate=False,
            optimizer_states_omitted=True, actor_updates=0, critic_updates=0, optimizer_updates=0,
            ppo_updates=0, multiplier_updates=0, environment_interactions=0,
            recording_id=None, anylogic_event_directory=None, validation_queries=0, test_queries=0,
            cpu_threads=torch.get_num_threads(),
            source_hashes_unchanged=True, source_sha256=sources, pairs=records)
        write(destination / 'manifest.json', manifest)
    except BaseException as exc:
        write(destination / 'FAILED.json', dict(status='preparation_failed', trainable=False, error=str(exc)))
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.design)
    print(json.dumps(dict(status=result['status'], prepared_pairs=len(result['pairs']),
                          execution_authorized=False, training_result=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
