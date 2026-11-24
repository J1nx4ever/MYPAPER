"""Verify the frozen helper, phase contracts and u32 inference without AnyLogic."""
from collections import Counter
from pathlib import Path
import tempfile

from gen_common import (
    CAMPAIGN, CORE, EXTENSION_OUT, MATERIALS, OUTPUT, REPORT, SUITE, TOTAL, authorization_for,
    checked, load_package, reference, require,
)
from baseline1_v1.evaluation import ReadOnlyTrainingGuard
from business_graph_summary_campaign_v1.plan_io import write
from business_graph_summary_recovery_v1.collection import MaterialPhaseCollector
from business_graph_summary_v1.features_cache import pack_summary_prefixes
from gen_budget import GenLedger, check_collisions
from gen_evaluation import EXPECTED_OBSERVATIONS, _phase, load_fixed_model
from java_scene import verify_java_scene
from python.observation_contract import FIELD_NAMES
from python.runtime_scenario import PREFIX, validate_runtime_fields


def verify_tensor_inputs(suite, phases):
    import h5py
    import numpy as np
    import torch

    torch.set_num_threads(suite['cpu_threads'])
    # Retain the sealed nominal observation and test copies with just the seven
    # visible fields replaced. This checks compatibility, never simulates a run.
    fixture = EXTENSION_OUT / 'evaluation/eval_u0032/material/run_9990713.h5'
    with h5py.File(fixture, 'r') as stream:
        state = stream['states'][0].astype(np.float32)
    model = load_fixed_model(suite)
    states = []
    rpc_sources = []
    for cfg in phases:
        values = cfg['jobs'][0]['expected_rpc_config']
        physical = {key: v for key, v in values.items() if key.startswith(PREFIX)}
        validate_runtime_fields(values, physical)
        rpc_sources.append(physical)
        altered = state.copy()
        for field, (key, scale) in EXPECTED_OBSERVATIONS.items():
            altered[FIELD_NAMES.index(field)] = values[key] / scale
        states.append(altered)
    require(np.array_equal(states[0], state), 'nominal observation values differ from recorded input')
    for test in rpc_sources[1:]:
        try:
            validate_runtime_fields(rpc_sources[0], test)
        except ValueError:
            pass
        else:
            raise AssertionError('nominal RPC values accepted for an altered scene')
    batch = np.stack(states)
    prefix = pack_summary_prefixes(batch, feature_config=model.config['feature_config'])
    with ReadOnlyTrainingGuard(model), torch.inference_mode():
        tensor = torch.from_numpy(batch)
        probabilities = model.actor.distribution(tensor, prefix).probs
        values = model.critic(tensor, prefix)
        require(probabilities.shape == (len(phases), 4)
                and bool(torch.isfinite(probabilities).all()) and bool(torch.isfinite(values).all())
                and bool(torch.allclose(probabilities.sum(-1), torch.ones(len(phases)))),
                'frozen u32 cannot read the scenario-coded observations')
    return dict(checkpoint_update=model.update_index, actor_digest=model.policy_version,
                source=reference(fixture), validated_candidate_shapes=list(probabilities.shape),
                validation_scope='sealed observation with scenario inputs changed offline')


def verify():
    suite, phases = load_package()
    check_collisions(suite)
    compile_result = verify_java_scene(checked(suite['java_helper']), checked(suite['scenario_table']))
    inference = verify_tensor_inputs(suite, phases)
    for phase, cfg in zip(suite['phases'], phases, strict=True):
        require(_phase(suite, phase) == cfg, 'evaluation phase reader differs')
        MaterialPhaseCollector(cfg, checked(phase['phase_config']),
                               publish_ready=lambda job, path: None,
                               startup=dict(control_directory=suite['control_directory'],
                                            campaign_id=CAMPAIGN,
                                            master_plan_sha256=suite['master_plan']['sha256']))
    with tempfile.TemporaryDirectory(prefix='bg128-generalization-verify-') as temp:
        d = Path(temp)
        authority = d / 'fixture_authority.json'
        write(authority, authorization_for(suite))
        ledger = GenLedger(suite, suite_path=SUITE, approval_path=authority, path=d / 'fixture.sqlite')
        ledger.register()
        state = ledger.snapshot()
        require(Counter(row['status'] for row in state['jobs'].values()) == {'PLANNED': TOTAL}
                and state['campaigns'][CAMPAIGN]['commits'] == {}, 'temporary registration differs')
    require(not OUTPUT.exists() and not MATERIALS.exists(), 'offline verification produced live run state')
    path = REPORT / 'preparation_validation.json'
    write(path, dict(status='PASS_OFFLINE_PREPARATION', suite=reference(SUITE), java=compile_result,
                     no_training_inference_check=inference, checked_phase_collectors=len(phases),
                     checked_slots=TOTAL, protected_core=reference(CORE),
                     live_simulator_runs=0, live_budget_registration=False))
    print(path)


if __name__ == '__main__':
    verify()
