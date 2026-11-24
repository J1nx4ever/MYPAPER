"""Read frozen training configurations; do not scan, collect or fit new data."""
from pathlib import Path
import csv
import json
import sys

PROJECT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(PROJECT))
from business_graph_summary_campaign_v1.plan_io import load_suite,reference

PLAN=Path(__file__).resolve().parent
OUT=PROJECT/'reports/token_attention_001/business_graph_summary_009/trained_intervention_001'


def main():
    path=PLAN.parent/'formal_training_001/prepared_001/suite.json'
    suite,phases=load_suite(path)
    key='config.runtime_scene.dump.0.crusherRateTph'
    values=sorted({job['expected_rpc_config'][key] for job in suite['jobs']})
    assert len(phases)==16 and len(suite['jobs'])==512 and values==[2500.]
    assert all(phase['rpc_configuration_authoritative'] is True and
        phase['require_local_alp_audit'] is False for phase in phases)
    with (OUT/'c128_features.csv').open(encoding='utf-8-sig',newline='') as f:
        changes=[dict(feature=r['feature'],before=float(r['base']),after=float(r['value']),delta=float(r['delta']))
            for r in csv.DictReader(f) if r['scene']=='C' and float(r['delta'])!=0.]
    assert len(changes)==8
    cfg=Path(suite['phase_configs'][0]['path'])
    common=json.loads(cfg.read_bytes())['common_reference']
    report=dict(schema='bg128-trained-interventions-training-support-audit-v1',status='PASS_CONFIG_ONLY',
        training_campaign=suite['campaign_id'],phase_count=16,configured_physical_shifts=512,
        field=key,all_training_expected_rpc_values=values,
        raw_feature_index=1304,raw_feature_scale_tph=2500.,BASE_raw_feature=1.,C_raw_feature=.5,
        C_current_rate_tph=1250.,C_within_configured_training_rate_support=False,
        observations_scanned=False,scope='All frozen configured training jobs and phase references; no new trajectory or per-state raw-column sweep.',
        RPC_config_checks_retained=True,
        deterministic_C128_rate_response=dict(implementation='grade_commitment_ppo_v1.features.build_features',
            rate_formula='rate_tons_per_second = raw[1304] * 2500 / 3600',
            operations=['nominal FIFO processing projection','window capacity = remaining_duration * rate','available adjustable quantity and required-grade calculations'],
            changed_features=changes),
        interpretation='C is a current processing-capacity parameter extrapolation beyond the configured training rate. Weak graph-branch response here cannot by itself diagnose two-layer GAT failure. C128 engineering projections change deterministically with rate; trained readout use of these projections is distinct from having observed rate changes in training.',
        A_B_scope='A is a completed-load transient boundary; B is exogenous reassignment not an available PPO action. Neither is an executed on-policy transition.',
        source_refs=[reference(path),common]+suite['phase_configs']+[reference(PROJECT/'grade_commitment_ppo_v1/features.py')],
        optimizer_updates=0,new_physical_shifts=0,budget_registered=False)
    with (OUT/'training_support_audit.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='PASS_CONFIG_ONLY',jobs=512,configured_rate_tph=values,C_extrapolation=True,C128_changed_features=len(changes))))


if __name__=='__main__':main()
