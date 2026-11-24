"""Compare fixed u32 actors on the previously frozen common grade replay."""

from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from statistics import fmean

import h5py
import numpy as np
import torch

from weak_plan import GRADE_ARCHIVE, REPORT, UNITS, VARIANTS, bind, load_package, locations, parent
from ablation_report import complete_report as previous_complete_report, paired_report
from business_graph_summary_campaign_v1.plan_io import load_reference, reference, sha
from business_graph_summary_v1.admission import frozen_actor
from business_graph_summary_v1.c_features import FEATURE_NAMES
from business_graph_summary_v1.models import digest, pack_summary_prefixes
from weak_runtime import runtime_scope

complete_report = bind(previous_complete_report, UNITS=UNITS, VARIANTS=VARIANTS, locations=locations, REPORT=REPORT)
SCHEMA = 'bg128-weak-gat-s01-conditioned-candidate-grade-v1'
GRADE_KEY = 'expected_dispatch_ore_grade_pct'


def _archive_states(archive):
    selections = defaultdict(list)
    for row in archive['decisions']:
        selections[row['full_run_index']].append(row['decision_index'])
    records, sources = {}, {}
    for episode in archive['sampled_episodes']:
        receipt = load_reference(episode['receipt'])
        material = episode['material']
        parent.require(receipt['references']['episode'] == material, 'grade archive receipt differs')
        identity = receipt['identity']
        run = identity['run_index']
        parent.require(run not in sources and identity['scene_id'] == episode['scenario']
                       and identity['environment_seed'] == episode['environment_seed']
                       and identity['actor_sampling_seed'] == episode['actor_sampling_seed'],
                       'grade archive identity differs')
        sources[run] = episode
        data = Path(material['path']).read_bytes()
        parent.require(sha256(data).hexdigest() == material['sha256'], 'archived states changed')
        with h5py.File(BytesIO(data), 'r') as store:
            parent.require(json.loads(store.attrs['identity_json']) == identity, 'HDF5 identity changed')
            for index in selections[run]:
                records[run, index] = store['states'][index]
    rows = archive['decisions']
    parent.require(len(sources) == 50 and len(records) == len(rows) == 400, 'grade sample inventory differs')
    raw = np.stack([records[row['full_run_index'], row['decision_index']] for row in rows])
    parent.require(raw.shape == (400, 1519) and raw.dtype == np.float32
                   and np.all(raw[:, 17] == 0), 'grade archive changed shape or state kind')
    return raw, rows


def _scores(actor, raw, rows):
    actor.eval()
    before = digest(actor)
    sites = raw[:, 1107:1295].reshape(-1, 4, 47)
    ore = sites[:, :, 1] == 1
    legal = raw[:, 19:23] == 1
    parent.require(np.all((ore & legal).sum(1) == 3), 'source sample has illegal ore candidates')
    grades = 100 * sites[:, :, 2]
    expected_nominal = np.array([r['nominal_current_grade_minus_center_pp'] for r in rows])
    prefix = pack_summary_prefixes(raw, feature_config=actor.feature_config)
    current = FEATURE_NAMES.index('current.nominal_grade')
    parent.require(np.array_equal(100 * (prefix.features[:, current] - raw[:, 15]), expected_nominal),
                   'C128 forecast differs from the frozen grade replay')
    with torch.inference_mode():
        probs = actor.distribution(torch.from_numpy(raw), prefix).probs.numpy()
    parent.require(np.isfinite(probs).all() and (probs[~legal] == 0).all()
                   and before == digest(actor), 'invalid probability, mask or Actor state')
    ore_prob = (probs * ore).sum(1)
    parent.require(np.all(ore_prob > 0), 'Actor assigns zero probability to all ore')
    return (probs * ore * grades).sum(1) / ore_prob, ore_prob


def _summaries(rows, expected, probability):
    strata = defaultdict(dict)
    for row, actual in zip(rows, expected, strict=True):
        cell = strata[row['scenario'], row['window'], row['rank']]
        cell.setdefault(row['grade_bin'], []).append(float(actual))
    by_scenario = defaultdict(list)
    all_contrasts = []
    for (scenario, _, _), cell in strata.items():
        low, high = cell['below_center'], cell['above_center']
        parent.require(len(low) + len(high) == 10, 'fixed replay stratum size differs')
        delta = fmean(low) - fmean(high)
        by_scenario[scenario].append(delta)
        all_contrasts.append(delta)
    parent.require(len(strata) == 40 and len(by_scenario) == 5, 'grade replay strata differs')
    return dict(compensation_pp=fmean(all_contrasts), positive_strata=sum(x > 0 for x in all_contrasts),
                ore_choice_probability=fmean(probability),
                by_scenario_pp={key: fmean(values) for key, values in sorted(by_scenario.items())})


def baseline_replay():
    archive = load_reference(GRADE_ARCHIVE)
    raw, rows = _archive_states(archive)
    base = archive['frozen_actors']['full']
    parent.require(reference(base['artifact']['path']) == base['artifact'], 'Full Actor bytes differ')
    actor, fingerprint = frozen_actor(base['artifact']['path'])
    parent.require(fingerprint == base['fingerprint'], 'Full Actor fingerprint differs')
    expected, probability = _scores(actor, raw, rows)
    locked = np.array([row[GRADE_KEY]['full'] for row in rows])
    error = float(np.max(np.abs(expected - locked)))
    parent.require(error < 5e-5, 'Full grade response does not match the frozen replay')
    return raw, rows, base, _summaries(rows, expected, probability), error


def grade_replay():
    torch.set_num_threads(2)
    parent.require(sha(parent.CORE) == parent.PROTECTED_SHA, 'protected PPO core differs')
    raw, rows, base, baseline, error = baseline_replay()
    outcomes, refs = dict(full=baseline), dict(full=base)
    for unit in UNITS:
        suite, _ = load_package(unit)
        root = Path(suite['python_output'])
        complete = load_reference(reference(root / 'suite_complete.json'))
        parent.require(complete['status'] == 'COMPLETE' and complete['suite'] == reference(suite['suite_path'])
                       and complete['training_updates'] == parent.UPDATES,
                       'grade replay needs both complete u32 checkpoints')
        summary = load_reference(complete['phase_results'][0]['report'])
        actor_ref = summary['actor_artifact']
        parent.require(summary['checkpoint_provenance'] == complete['new_checkpoint']
                       and reference(actor_ref['path']) == actor_ref
                       and summary['training_state_unchanged'] and summary['evaluation_optimizer_updates'] == 0,
                       'weak Actor provenance differs')
        with runtime_scope(suite) as runtime:
            actor, fingerprint = runtime.frozen_actor(actor_ref['path'])
            parent.require(fingerprint == summary['fingerprints']['actor'], 'weak Actor fingerprint differs')
            expected, probability = _scores(actor, raw, rows)
        outcomes[unit] = _summaries(rows, expected, probability)
        outcomes[unit]['delta_from_full_pp'] = outcomes[unit]['compensation_pp'] - baseline['compensation_pp']
        refs[unit] = dict(actor=actor_ref, checkpoint=complete['new_checkpoint'], fingerprint=fingerprint)
    output = REPORT / 'fixed_grade_response_replay.json'
    payload = dict(schema=SCHEMA, grade_archive=GRADE_ARCHIVE, sampled_historical_decisions=len(rows),
                   original_grade_replay_max_abs_error_pp=error, model_artifacts=refs, results=outcomes,
                   protected_core_sha256=sha(parent.CORE), new_shifts=0, optimizer_updates=0)
    from business_graph_summary_campaign_v1.runtime import _same_or_save

    _same_or_save(output, payload)
    return reference(output)
