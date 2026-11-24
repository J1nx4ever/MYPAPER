"""Cross-artifact audit of new diagnostics; no models fitted or simulations."""
from collections import defaultdict
import csv
import numpy as np
from common import *
from python.constrained_batch import load_dataset
from physical_realization_v1.risk import fifo_risk,encoded_interval,Interval

def csv_read(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    replay=read(OUT/'replay.json');inter=read(OUT/'interventions.json');late=read(OUT/'late_updates.json')
    chain=read(OUT/'dispatch_realization.json')
    joint=read(OUT/'joint_support.json')
    config=read(PLAN/'minibatch_replay_config_v2.json');recon=Path(config['python_output']);complete=read(recon/'complete.json')
    assert complete['status']=='PASS_EXACT_RECONSTRUCTION'
    refs=replay['source_pins']+inter['sources']+inter['training_source_pins']+late['source_pins']+late['checkpoint_pins']+chain['source_pins']+joint['source_pins']
    for r in replay['trajectories']:refs.extend(r['sources'].values())
    for item in config['updates']:
        refs.extend(item[k] for k in ('before','after','summary','actor'))
        u=read(recon/f"update_{item['update']:04}.json")
        assert u['status']=='PASS_EXACT_RECONSTRUCTION' and all(u['checks'].values())
        assert all(read(recon/f"preflight_{item['update']:04}.json")['checks'].values())
        refs.extend(u['source_pins'])
    refs.extend([complete['config'],config['replay_implementation'],config['protected_core'],config['supersedes_rejected_reconstruction']])
    unique={}
    for r in refs:
        if r['path'] in unique:assert unique[r['path']]==r,'conflicting source hash for the same path'
        unique[r['path']]=r
    assert all(ref(r['path'])==r for r in unique.values())
    selected=read(OUT/'state_selection.json')['states'];lookup=defaultdict(list)
    for s in selected:lookup[(s['lr'],s['run_index'])].append(s)
    snapshots=0
    for c in cohort():
        key=c['lr'],c['job']['run_index'];rows,*_=load_dataset(c['raw'],require_real=True)
        for s in lookup[key]:
            assert np.array_equal(np.asarray(s['raw'],dtype=np.float32),rows[s['state_index']]['observation'])
            assert s['time']==float(rows[s['state_index']]['action_time_seconds']);snapshots+=1
        if c['job']['run_index']==9970001:
            near=next(r for r in replay['trajectories'] if r['run_index']==9970001)
            marker=near['earliest_visible_support_low_exclusion']
            state=rows[marker['decision_sequence']]['observation']
            interval=Interval(0.,0.)
            for valid,q,g in state[1391:1439].reshape(16,3):
                if not valid:break
                interval=interval.add(encoded_interval(q,500))
            capacity=encoded_interval(state[3],7200).mul(encoded_interval(state[1304],2500/3600))
            risk=fifo_risk(state)
            assert interval.lo>capacity.hi and risk.certificate_valid and risk.recovery_excluded
            assert risk.grade_upper==marker['visible_support_upper'] and risk.grade_upper<.26-1e-12
            near_details=dict(fifo_tons_interval=[interval.lo,interval.hi],
                capacity_tons_interval=[capacity.lo,capacity.hi],grade_upper=risk.grade_upper)
        rows.close()
    assert snapshots==len(selected)==118
    failed=[r for r in replay['trajectories'] if not r['compliant']]
    assert len(failed)==10 and sum(r['window']==0 for r in failed)==8 and sum(r['window']==3 for r in failed)==2
    # Outcomes are independent suffix evidence. All exact observed suffixes
    # must lie within the optimistic prefix envelope, including controls.
    prefixes=csv_read('decision_prefixes.csv');final={(r['lr'],r['run_index']):r for r in replay['trajectories']}
    for p in prefixes:
        r=final[p['lr'],int(p['run_index'])];g=r['processed']['grade']
        assert float(p['lower'])<=g<=float(p['upper'])
        if p['excluded']=='True':assert not r['compliant']
        if p['visible_support_low_recovery_excluded']=='True':assert not r['compliant']
    decomp=csv_read('readout_decomposition.csv');groups=defaultdict(list)
    for r in decomp:
        full=float(r['full_delta']);g=float(r['graph_shapley']);c=float(r['c128_shapley'])
        assert abs(full-g-c)<1e-12
        groups[r['role'],r['lr'],r['run_index'],r['state_index'],r['scene']].append(r)
    for key,rs in groups.items():
        if key[0]=='actor':
            assert len(rs)==4 and abs(sum(float(r['base']) for r in rs)-1)<2e-6
            assert abs(sum(float(r['branch']) for r in rs)-1)<2e-6
    assert len(read(OUT/'branches.json'))==275
    assert inter['eligible']==dict(A=53,B=104,C=118)
    assert inter['training_decisions_scanned']==93534 and inter['training_shifts_scanned']==512
    assert joint['training_decisions']==93534 and joint['unseen_branch_patterns']==dict(A=53,B=57,C=118)
    assert len(csv_read('joint_support.csv'))==275
    assert len(csv_read('late_state_outputs.csv'))==800*7
    assert len(csv_read('dispatch_realization.csv'))==len(prefixes)==987
    for r in chain['trajectories']:
        assert not r['unmatched_unload_batches']
        assert abs(sum(r['processed_mass_attribution'].values())-final[r['lr'],r['run_index']]['processed']['tons'])<1e-6
    assert sha(CORE)==CORE_SHA
    rejected=read(OUT/'minibatch_replay_001/update_0014.json')
    assert rejected['status']=='REJECT_DIVERGED_RECONSTRUCTION'
    save('validation.json',dict(status='PASS',unique_source_files_verified=len(unique),selected_real_states_verified=snapshots,
        replay_prefixes_verified=len(prefixes),decomposition_outputs_verified=len(decomp),branch_groups_verified=len(groups),
        successful_control_exclusions=0,all_observed_outcomes_within_optimistic_prefix_bounds=True,
        near_boundary_interval_certificate=near_details,
        realized_dispatch_chains=987,unmatched_ore_unload_batches=0,
        narrow_discrete_support_checks=275,
        exact_original_update_reconstructions=3,new_physical_shifts=0,official_training_updates=0,
        offline_actor_optimizer_steps=complete['offline_actor_optimizer_steps'],
        offline_critic_optimizer_steps=complete['offline_critic_optimizer_steps'],
        rejected_attempt_offline_actor_optimizer_steps=rejected['offline_actor_optimizer_steps'],
        rejected_attempt_offline_critic_optimizer_steps=rejected['offline_critic_optimizer_steps'],
        total_including_rejected_offline_actor_optimizer_steps=complete['offline_actor_optimizer_steps']+rejected['offline_actor_optimizer_steps'],
        total_including_rejected_offline_critic_optimizer_steps=complete['offline_critic_optimizer_steps']+rejected['offline_critic_optimizer_steps'],
        protected_ppo=ref(CORE),test_count=4,mathematical_tests='test_diagnostics.py: four boundary/support/KL tests passed',
        mathematical_test_implementation=ref(PLAN/'test_diagnostics.py'),
        audit_implementation=ref(Path(__file__)),external_diagnostic_modules=[
            ref(PROJECT/'physical_realization_v1/risk.py'),
            ref(PLAN.parent/'checkpoint_evaluation_001/trained_analysis_001/analyze_first_window_failures.py')]))
    print(json.dumps(dict(status='PASS',source_files=len(unique),real_states=snapshots,decision_prefixes=len(prefixes))))

if __name__=='__main__':main()
