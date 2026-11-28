"""Audit the native shifts, select on the validation block, then compare paired B shifts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import fmean, stdev

import campaign
from campaign import REPORTS, witness, require
import collect_stats
from collect_scaled import verified_targets
from resilience_runtime import record
from topology_learner import checkpoint_sha, checkpoint_path


def load(path):
    return json.loads(Path(path).read_bytes())


def load_evidence(pin):
    data = Path(pin['path']).read_bytes()
    require(sha256(data).hexdigest() == pin['sha256'], f"historical evidence changed at {pin['path']}")
    return json.loads(data)


def baseline_row(slot):
    cfg, _, _ = campaign.load_case(f'E_{slot:03d}')
    source = cfg['paired_baseline']
    require(source == campaign.baseline_slot(slot), f'B{slot:03d} historical pairing differs from frozen config')
    receipt = load_evidence(source['receipt'])
    audit = load_evidence(source['policy_verification'])
    require(receipt['run_index'] == source['run_index'] and receipt['recording_id'] == source['recording_id']
            and receipt['experiment_seed'] == receipt['environment_seed'] == source['environment_seed']
            and receipt['config_sha256'] == source['config']['sha256'] and receipt['manifest'] == source['manifest']
            and receipt['physical_status'] == 'COMPLETE' and receipt['material_balance_verified'] is True
            and audit['status'] == 'VERIFIED' and audit['action_mode'] == 'argmax'
            and audit['checkpoint_sha256'] == cfg['parent_checkpoint_sha256'], f'B{slot:03d} historical result differs')
    return dict(receipt, slot=slot, seed=source['environment_seed'], receipt_sha256=source['receipt']['sha256'])


def freeze(path, value):
    if path.exists():
        require(load(path) == value, f'previously frozen evidence differs at {path}')
    else:
        record(path, value)


def opening_targets(_, cfg):
    scene = cfg['expected_runtime_scene']
    task = cfg['task']
    native = scene['exogenous_config']
    require(scene['layout_id'] == 'loop_B' and (task['grade_lower'], task['grade_upper'], task['window_seconds'])
            == (.26, .28, 7200) and native['deterministic'] is False and native['fault_truck_id'] == -1
            and native['endogenous_congestion'] is False, 'unexpected road, equipment or grade rule')
    training = cfg['phase'] == 'train'
    require(cfg['action_mode'] == ('sample' if training else 'argmax') and
            (bool(cfg['actor_sampling_seed']) if training else cfg['actor_sampling_seed'] is None),
            'incorrect action generation')
    return verified_targets(scene, task)


assess = witness.bind(collect_stats.assess_one, opening_targets=opening_targets)
pinned_checkpoint = lru_cache(maxsize=5)(checkpoint_sha)


def read_shift(label):
    cfg, _, job = campaign.load_case(label)
    binding = load(cfg['activation_path'])
    update = binding['policy_update']
    require(update == (load(campaign.CAMPAIGN_OUTPUT/'selection.json')['selected_update']
                       if cfg['phase'] == 'test' else cfg['policy_update'])
            and binding['run_index'] == job['run_index'] and binding['recording_id'] == job['recording_id']
            and binding['config_sha256'] == job['config_sha256']
            and binding['checkpoint_path'] == str(checkpoint_path(update))
            and binding['checkpoint_sha256'] == pinned_checkpoint(update), f'{label} checkpoint binding differs')
    physical, audit = assess((label, dict(cfg, checkpoint_sha256=binding['checkpoint_sha256']), job, False))
    require(physical['physical_status'] == 'COMPLETE' and audit['status'] == 'VERIFIED'
            and physical.get('material_balance_verified') is True, f'{label}: {physical.get("reason")}; {audit}')
    manifest = load(physical['manifest']['path'])
    require(manifest['native_policy_update'] == update and len(physical['windows']) == 4,
            f'{label} native checkpoint or windows differ')
    if cfg['phase'] == 'train':
        terminal = Path(job['python_output'])/'terminal_observation.json'
        require(manifest['terminal_observation_sha256'] == witness.reference(terminal)['sha256'],
                f'{label} terminal data differ')
    return dict(case=label, phase=cfg['phase'], slot=cfg['slot'], policy_update=update,
                seed=cfg['environment_seed'], physical=physical, inference_audit=audit)


def get_shifts(labels):
    with ProcessPoolExecutor(max_workers=5) as pool:
        return list(pool.map(read_shift, labels, chunksize=2))


def cohort(rows, update=None):
    n = len(rows)
    joint = sum(r['task_status'] == 'MET' for r in rows)
    windows = [[r['windows'][i] for r in rows] for i in range(4)]
    def distance(w):
        grade = 100 * w['metal_tons']/w['ore_tons'] if w['ore_tons'] > 0 else 0.
        return max(26.-grade, grade-28., 0.)
    ore = [r['ore_tons'] for r in rows]
    waste = [r['waste_tons'] for r in rows]
    return dict(update=update, n=n, joint=joint, four=sum(r['compliant_windows'] == 4 for r in rows),
                ore=sum(r['ore_target_met'] for r in rows), waste=sum(r['waste_target_met'] for r in rows),
                window_pass=[sum(w['ore_tons'] > 0 and w['cost'] == 0 for w in group) for group in windows],
                grade_excess_pp=math.fsum(distance(w) for group in windows for w in group),
                ore_mean=fmean(ore), waste_mean=fmean(waste), ore_sd=stdev(ore), waste_sd=stdev(waste),
                joint_ci=collect_stats.success_rate(joint,n)['wilson_95'])


def select(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[row['policy_update']].append(row)
    require(set(groups) == set(range(5)) and len(rows) == 80, 'incomplete validation')
    reference=[(i['slot'], i['seed']) for i in groups[0]]
    require(len(reference)==len(set(s for _,s in reference))==16 and
            all([(i['slot'],i['seed']) for i in group]==reference for group in groups.values()),
            'validation slot mismatch')
    ranking=[cohort([r['physical'] for r in groups[k]],k) for k in range(5)]
    best=max(ranking, key=lambda g:(g['joint'],sum(g['window_pass']),-g['grade_excess_pp'],-g['update']))
    path=REPORTS/'validation.json'
    freeze(path, dict(candidates=ranking,rows=rows))
    update=best['update']
    chosen=dict(selected_update=update,checkpoint_path=str(checkpoint_path(update)),checkpoint_sha256=pinned_checkpoint(update),
                validation_cases=80,validation_report=witness.reference(path),selection_order=campaign.PROTOCOL['selection_order'])
    freeze(campaign.CAMPAIGN_OUTPUT/'selection.json',chosen)
    return chosen,ranking


def verify_budget(rows):
    snapshot = campaign.budget_snapshot()
    expected = {}
    receipts = []
    for row in rows:
        _, suite, job = campaign.load_case(row['case'])
        run = str(job['run_index'])
        ledger_status, receipt = campaign.ledger_entry(suite, job)
        entry = snapshot['jobs'][run]
        evidence = load_evidence(receipt)
        require(ledger_status == entry['status'] == 'COMPLETE' and entry['receipt'] == receipt
                and entry['campaign_id'] == suite['campaign_id'] and entry['identity'] == witness.ledger.identity(job)
                and evidence == campaign._receipt(job, row['physical']),
                f"merged ledger or physical receipt differs at {row['case']}")
        expected[run] = suite['campaign_id']
        receipts.append([run, receipt['sha256']])
    prefix = campaign.CAMPAIGN + '-'
    actual = {run: j['campaign_id'] for run, j in snapshot['jobs'].items() if str(j.get('campaign_id', '')).startswith(prefix)}
    registered_names = {name for name in snapshot['campaigns'] if name.startswith(prefix)}
    require(len(expected) == len(registered_names) == len(rows) == len(campaign.CASES)
            and actual == expected and registered_names == set(expected.values()),
            'campaign registrations differ from the sealed physical roster')
    overview = load_evidence(dict(path=campaign.OVERVIEW, sha256=campaign.OVERVIEW.with_suffix('.sha256').read_text().strip()))
    before = overview['registered_at_preparation']
    baseline = dict(campaign_id=campaign.CAMPAIGN, registered_at_preparation=before,
                    registered_by_preparation=overview['registered_by_preparation'], registered_by_campaign=len(expected),
                    complete_physical_receipts=len(receipts), receipt_index_sha256=sha256(campaign.canonical(receipts)).hexdigest())
    total = snapshot['maximum_total_shifts']
    require(baseline['registered_by_preparation'] == 0 and total >= before + len(expected),
            'merged count does not include every new physical shift')
    path = REPORTS / 'budget_audit.json'
    if path.exists():
        first = load(path)
        audited_total = first.get('registered_at_final_audit', -1)
        require(first == dict(baseline, registered_at_final_audit=audited_total,
                              other_registrations_since_preparation=audited_total-before-len(expected))
                and before + len(expected) <= audited_total <= total, 'sealed budget audit differs')
        return first
    result = dict(baseline, registered_at_final_audit=total,
                  other_registrations_since_preparation=total-before-len(expected))
    freeze(path, result)
    return result


def finish():
    rows=get_shifts(campaign.CASES)
    selected,validation=select([r for r in rows if r['phase']=='validation'])
    training=[]
    previous_sha=pinned_checkpoint(0)
    for k in range(1,5):
        group=[r for r in rows if r['phase']=='train' and r['policy_update']==k-1]
        metric=load(checkpoint_path(k).parent/'metrics.json')
        g=cohort([r['physical'] for r in group],k)
        require(metric['checkpoint_sha256']==pinned_checkpoint(k) and metric['parent_checkpoint_sha256']==previous_sha
                and g['n']==32 and g['joint']==metric['physical']['joint']
                and g['window_pass']==metric['physical']['compliant_by_window'], f'update {k} lineage or outcomes differ')
        g.update(actor_loss=metric['mean']['actor_loss'],critic_mse=metric['mean']['critic_mc_mse'],
                 mean_cost=metric['metrics']['combined_cost_mean'], metrics=witness.reference(checkpoint_path(k).parent/'metrics.json'))
        training.append(g)
        previous_sha=metric['checkpoint_sha256']
    old=[baseline_row(slot) for slot in range(1,51)]
    new=[r for r in rows if r['phase']=='test']
    require(len(old)==len(new)==50, 'paired test requires 50 slots')
    pairs=[]
    for prior, candidate in zip(old,new,strict=True):
        c=candidate['physical']
        require(prior['slot']==candidate['slot'] and prior['seed']==candidate['seed']==c['environment_seed']
                and prior['run_index'] != c['run_index'] and prior['recording_id'] != c['recording_id']
                and all(c['opening_targets'][k] == v for k,v in prior['opening_targets'].items()),
                'inconsistent paired reference')
        pairs.append(dict(slot=prior['slot'],environment_seed=candidate['seed'],previous=prior['run_index'],new=c['run_index'],
                          previous_receipt_sha256=prior['receipt_sha256'],
                          previous_met=prior['task_status']=='MET',new_met=c['task_status']=='MET'))
    freeze(REPORTS/'paired_test.json',pairs)
    b=sum(x['previous_met'] and x['new_met'] for x in pairs)
    i=sum(not x['previous_met'] and x['new_met'] for x in pairs)
    r=sum(x['previous_met'] and not x['new_met'] for x in pairs)
    disc=i+r
    p=min(1.,2*sum(math.comb(disc,j) for j in range(min(i,r)+1))/2**disc)
    paired=dict(both_pass=b,improved=i,regressed=r,both_fail=50-b-i-r,delta_pp=2*(i-r),exact_mcnemar_p=p)
    require(len(rows)==len({x['physical']['recording_id'] for x in rows})==258,'identity count differs')
    with campaign.settlement_session() as settle:
        for row in rows:
            cfg,suite,job=campaign.load_case(row['case'])
            row['ledger_status']=settle(suite,job,row['physical'],collect=True)
            require(row['ledger_status']=='COMPLETE','incomplete physical ledger')
            freeze(Path(job['python_output'])/'policy_verification.json',row['inference_audit'])
    freeze(REPORTS/'cases.json', rows)
    summary=dict(training=training,validation=validation,selection=selected,test=cohort([x['physical'] for x in new],selected['selected_update']),
                 baseline=cohort(old,0),paired=paired,total_actions=sum(x['inference_audit']['decision_count'] for x in rows),
                 verified_shifts=len(rows), protocol=campaign.PROTOCOL, budget=verify_budget(rows))
    freeze(REPORTS/'summary.json',summary)
    from curves import write_figures_and_report
    write_figures_and_report(REPORTS,summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('select','final'))
    mode=parser.parse_args().operation
    if mode=='select':
        result=select(get_shifts([k for k in campaign.CASES if k.startswith('V_')]))[0]
    else:
        result=finish()
    print(json.dumps(result,ensure_ascii=False),flush=True)
