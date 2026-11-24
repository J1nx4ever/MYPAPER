"""Source-bound 89-check offline evidence, never a physical-run receipt.

The mapping below names actual executable assertions. A complete JUnit run and
the separately pinned real historical stream are mandatory; design documents,
synthetic collection fixtures and a claimed PASS string cannot substitute.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from .contract import CANDIDATES, require
from .gate import source_pins
from .plan_io import ROOT, reference

SCHEMA='relation-gat-ego-offline-acceptance-v1'
PACKAGE=Path(__file__).resolve().parent
PYTHON='/home/jinx/miniconda3/envs/anylogic-env/bin/python'
IDS=tuple([f'R3-{prefix}{i:02d}' for prefix,n in [('N',32),('EA',13),('X',12),('CP',8)]
          for i in range(1,n+1)]+[f'RG{i:02d}' for i in range(1,17)]+[f'GC{i:02d}' for i in range(1,9)])


def _t(file,function):
    return f'relation_gat_ego_v1/test_{file}.py::test_{function}'


# Short internal names expand to explicit file/function selectors. Every
# parametrized instance of every selected function is required, not just one.
TESTS={
 'shapes':_t('token_encoder','token_complete_shapes_inventory_and_fixed_names'),
 'invalid':_t('token_encoder','common_rejects_invalid_original_rows'),
 'rawgrad':_t('token_encoder','raw_torch_validation_preserves_continuous_grad_and_bytes'),
 'tokenbatch':_t('token_encoder','token_batch_current_row_isolation_and_readonly_rng'),
 'site':_t('token_encoder','site_namespace_structural_masks_and_closed_entities_retained'),
 'motion':_t('token_encoder','motion_grade_gates_preserve_order_and_original'),
 'node':_t('token_encoder','node_content_independent_mean_count_oracle_and_observed_segment_count'),
 'refs':_t('token_encoder','reference_sources_typed_ids_missing_and_true_zero'),
 'missing':_t('token_encoder','missing_versus_true_zero_has_neural_canary'),
 'roadid':_t('token_encoder','route_same_base_different_road_identity_canary'),
 'targetid':_t('token_encoder','same_base_target_identity_reaches_neural_truck_token'),
 'memberid':_t('token_encoder','same_base_ordered_member_identity_reaches_neural_site_token'),
 'ordered':_t('token_encoder','route_order_and_ordered_list_slots_are_not_sorted_by_identity'),
 'service':_t('token_encoder','service_multiset_permutation_invariant_counts_distinguish_multiplicity'),
 'serviceid':_t('token_encoder','service_identity_histogram_has_neural_canary_with_identical_base'),
 'fifo':_t('token_encoder','fifo_original_order_zero_slots_empty_bias_gate_and_attachment'),
 'sidecar':_t('token_encoder','action_sidecar_never_enters_z0_and_terminal_diagnostic_has_no_ego'),
 'tokenfinite':_t('token_encoder','nonfinite_intermediate_fails_without_sanitizing_input'),
 'ccontract':_t('c_features','c_contract_source_pin_and_outer_config_is_defensive'),
 'cexact':_t('c_features','c_legacy_common_domain_byte_exact'),
 'czero':_t('c_features','c_zero_mass_private_projection_equivalence'),
 'cboundary':_t('c_features','c_zero_mass_projection_at_arrival_boundaries'),
 'cinvalid':_t('c_features','c_rejects_invalid_original_before_projection'),
 'ctiny':_t('c_features','c_batch_independent_and_preserves_positive_subnormal'),
 'edgeoracle':_t('relation_edges','rg01_all_27_types_named_schema_oracle_and_batch_offsets'),
 'edgebound':_t('relation_edges','rg01_dense_bound_1811_not_truncated'),
 'edgemissing':_t('relation_edges','rg02_missing_distinct_from_zero_and_role_global_errors'),
 'edgeinvalid':_t('relation_edges','rg02_invalid_reference_rejected'),
 'edgeroute':_t('relation_edges','rg02_onehot_and_route_identity_errors'),
 'edgeorder':_t('relation_edges','rg03_ordered_slots_route_and_service_multiset'),
 'self':_t('relation_edges','rg04_self_exact_and_mutation_invalidates_validation'),
 'closed':_t('relation_edges','rg05_closed_unavailable_and_irrelevant_fields_do_not_change_graph'),
 'count':_t('relation_edges','rg08_counts_exact_and_metadata_errors'),
 'edgereadonly':_t('relation_edges','rg15_raw_and_rng_readonly_cache_invalidates_raw'),
 'edgecache':_t('relation_edges','rg15_per_row_cache_reordered_minibatches_equivalent_and_bounded'),
 'graphoracle':_t('graph_layers','gc08_rg06_exact_stage_oracle_inventory_and_gradients'),
 'uniform':_t('graph_layers','gc01_rg14_two_layer_non_degenerate_uniform_oracle'),
 'rank':_t('graph_layers','gc02_rank_source_binding_nonlinearity_and_singleton_edge_value'),
 'message':_t('graph_layers','gc02_single_message_boundary_type_and_rank_canary'),
 'countpath':_t('graph_layers','gc03_rg08_count_path_preserves_multiplicity_after_uniform_mean'),
 'graphperm':_t('graph_layers','gc04_rg07_storage_permutation_softmax_domain_and_batch'),
 'twohop':_t('graph_layers','rg09_rg10_two_synchronous_layers_two_hop_and_storage_isolation'),
 'global':_t('graph_layers','rg11_global_is_outside_graph_interface'),
 'graphfinite':_t('graph_layers','gc08_rg07_finite_inputs_overflow_intermediate_rejected'),
 'graphreadonly':_t('graph_layers','rg15_forward_metadata_input_and_rng_readonly'),
 'egoformula':_t('models','ego_independent_formula_and_standard_mha_oracle'),
 'egouniform':_t('models','ego_uniform_and_single_output_bias_oracle_with_residual'),
 'egograd':_t('models','ego_permutation_and_all_value_residual_gradients'),
 'gather':_t('models','full_inventory_graph_updated_gather_mask_and_global_hooks'),
 'fulluniform':_t('models','uniform_graph_oracle_extends_to_ego_action_actor_and_critic'),
 'isolation':_t('models','named_initialization_common_bytes_all_roles_no_rng_consumption_or_storage_alias'),
 'terminal':_t('models','terminal_production_skips_s_mixed_restores_positions_and_diagnostic_uses_graph'),
 'recompute':_t('models','current_parameter_recompute_and_bidirectional_actor_critic_isolation'),
 'modelbatch':_t('models','batch_microbatch_global_and_input_only_readonly'),
 'globalratio':_t('models','global_changes_do_not_change_physical_query_or_fixed_physical_key_weight_ratio'),
 'egofinite':_t('models','nonfinite_and_layernorm_overflow_rejected'),
 'diagram':_t('models','first_decision_export_is_once_pinned_readonly_and_does_not_start_run'),
 'init':_t('initialization','gc06_all_operators_exact_bytes_and_rng_readonly'),
 'initpair':_t('initialization','gc06_cross_arm_same_role_pairing_and_independent_storage'),
 'initorder':_t('initialization','gc06_unique_score_creation_order_does_not_disturb_common_streams'),
 'initinvalid':_t('initialization','gc06_invalid_role_unknown_parameters_and_aliases_rejected'),
 'future':_t('features_cache','future_canary_and_forged_cache_rejected'),
 'nop':_t('features_cache','zero_mass_raw_retained_c_equal_and_legacy_physical_chain_unused'),
 'cachegrad':_t('state_cache','cached_vs_uncached_full_s_outputs_and_gradients'),
 'cachecurrent':_t('state_cache','cached_edges_never_cache_learned_states_or_graphs'),
 'bypass':_t('state_cache','g0_has_no_graph_cache_and_diagnostic_uses_strict_bypass'),
 'cacheinvalidate':_t('state_cache','raw_edge_change_invalidates_full_s_cache_but_other_row_is_reused'),
 'update':_t('learner','joint_gae_mc_update_and_exact_restore'),
 'optimizer':_t('learner','actor_and_critic_optimizer_storage_isolated'),
 'terminalvalue':_t('learner','true_terminal_prediction_has_zero_bootstrap_and_cannot_be_action'),
 'sampling':_t('collection','online_private_rng_replays_each_raw_decision_and_resets_at_shift'),
 'admission':_t('collection','source_bound_reload_keeps_failed_production_and_cost_only'),
 'phase':_t('collection','actor_and_phase_feature_branches_cannot_be_relabelled'),
 'timeout':_t('collection','ack_seal_wait_timeout_preserves_incomplete_without_recapture'),
 'canonical':_t('acceptance','complete_canonical_storage_mapping_through_graph_ego_actions_and_heads'),
 'ownvalue':_t('acceptance','ego_own_value_and_residual_are_independent_paths_and_uniform_gradients'),
 'headgrad':_t('acceptance','each_graph_updated_action_and_ego_reaches_both_heads_with_c_only_at_fusion'),
 'labels':_t('acceptance','label_future_sampled_action_interfaces_are_rejected_without_state_change'),
 'stamp':_t('acceptance','terminal_flags_and_old_or_same_shape_checkpoint_contract_rejected'),
 'tokenvariance':_t('acceptance','token_finite_variance_overflow_is_rejected'),
 'autocast':_t('acceptance','token_cpu_autocast_rejected'),
 'zeroentry':_t('acceptance','zero_mass_original_survives_sampling_admission_training_and_offline_replay'),
 'proof':_t('acceptance','proof_requires_complete_passed_pinned_tests_and_exact_89_mapping'),
 'streamproof':_t('acceptance','historical_artifact_requires_sources_complete_rows_and_all_arms'),
}

_ROWS={
 'N01':'invalid ccontract phase stamp edgeoracle',
 'N02':'shapes gather graphoracle count',
 'N03':'invalid cinvalid czero edgeinvalid edgemissing edgeroute edgebound count self',
 'N04':'shapes twohop graphoracle',
 'N05':'site closed',
 'N06':'missing refs edgemissing',
 'N07':'site fifo refs edgeoracle edgemissing',
 'N08':'fifo czero',
 'N09':'fifo czero zeroentry closed',
 'N10':'fifo cinvalid',
 'N11':'motion rawgrad',
 'N12':'refs missing edgemissing',
 'N13':'refs node edgeoracle',
 'N14':'canonical graphperm',
 'N15':'canonical edgeoracle edgemissing edgeinvalid',
 'N16':'ordered motion fifo service edgeorder graphperm',
 'N17':'node closed count',
 'N18':'sidecar gather closed',
 'N19':'terminal stamp invalid terminalvalue',
 'N20':'tokenbatch labels future modelbatch',
 'N21':'modelbatch terminal graphperm tokenbatch',
 'N22':'tokenfinite tokenvariance autocast graphfinite graphreadonly egofinite modelbatch',
 'N23':'rawgrad graphoracle twohop egograd headgrad optimizer cachegrad',
 'N24':'shapes init initpair initorder initinvalid twohop',
 'N25':'cexact czero nop headgrad labels',
 'N26':'streamproof',
 'N27':'uniform fulluniform gather bypass',
 'N28':'egoformula graphperm closed count',
 'N29':'gather canonical',
 'N30':'isolation recompute optimizer twohop cachecurrent',
 'N31':'headgrad gather ownvalue',
 'N32':'modelbatch diagram graphreadonly tokenbatch',
 'EA01':'egoformula gather',
 'EA02':'egoformula',
 'EA03':'ownvalue gather',
 'EA04':'egouniform gather',
 'EA05':'modelbatch egoformula',
 'EA06':'egograd',
 'EA07':'gather labels headgrad optimizer',
 'EA08':'terminal stamp terminalvalue',
 'EA09':'terminal stamp',
 'EA10':'invalid refs egofinite',
 'EA11':'terminalvalue update admission timeout',
 'EA12':'egograd ownvalue headgrad terminal update',
 'EA13':'egofinite graphfinite autocast',
 'X01':'roadid targetid memberid serviceid missing',
 'X02':'service serviceid edgeorder countpath',
 'X03':'cexact ccontract',
 'X04':'czero cboundary ctiny fifo zeroentry',
 'X05':'cinvalid invalid',
 'X06':'egouniform',
 'X07':'isolation initpair initorder',
 'X08':'gather labels future closed',
 'X09':'isolation recompute optimizer twohop cachecurrent',
 'X10':'sampling admission update recompute cachecurrent cacheinvalidate zeroentry nop',
 'X11':'twohop global globalratio gather',
 'X12':'terminal stamp terminalvalue update',
 'CP01':'egouniform',
 'CP02':'egouniform ownvalue',
 'CP03':'egouniform egoformula',
 'CP04':'ownvalue egograd',
 'CP05':'init initpair initorder isolation',
 'CP06':'recompute optimizer twohop isolation',
 'CP07':'fulluniform terminal gather ccontract',
 'CP08':'gather graphoracle headgrad shapes egoformula',
 'RG01':'edgeoracle edgebound edgemissing edgeinvalid',
 'RG02':'edgemissing edgeinvalid edgeroute canonical',
 'RG03':'edgeorder graphperm countpath',
 'RG04':'self graphperm',
 'RG05':'closed site',
 'RG06':'graphoracle rank message',
 'RG07':'graphperm graphfinite',
 'RG08':'count countpath edgeorder',
 'RG09':'twohop graphperm',
 'RG10':'twohop',
 'RG11':'global gather globalratio',
 'RG12':'gather headgrad ownvalue',
 'RG13':'terminal stamp terminalvalue',
 'RG14':'uniform fulluniform',
 'RG15':'graphperm graphreadonly edgereadonly labels gather modelbatch future',
 'RG16':'gather graphoracle initpair isolation optimizer sampling update cachecurrent stamp edgecache',
 'GC01':'uniform fulluniform graphoracle',
 'GC02':'rank message count',
 'GC03':'count edgeorder countpath graphperm',
 'GC04':'graphperm twohop',
 'GC05':'gather bypass terminal stamp',
 'GC06':'init initpair initorder initinvalid isolation',
 'GC07':'isolation optimizer twohop graphoracle headgrad labels cachecurrent',
 'GC08':'gather graphoracle graphfinite tokenvariance egofinite',
}
COVERAGE={('R3-'+k if not k.startswith(('RG','GC')) else k):tuple(TESTS[t] for t in v.split())
          for k,v in _ROWS.items()}
assert set(COVERAGE)==set(IDS) and len(IDS)==89


def test_pins():
    return {str(p.resolve()):reference(p)['sha256'] for p in sorted(PACKAGE.rglob('test_*.py'))}


def collect_test_nodeids():
    require(str(Path(sys.executable).resolve())==str(Path(PYTHON).resolve()),'anylogic-env interpreter required')
    result=subprocess.run([PYTHON,'-B','-m','pytest','relation_gat_ego_v1','--collect-only','-q',
                           '-p','no:cacheprovider'],cwd=ROOT,text=True,capture_output=True,check=False,timeout=60)
    require(result.returncode==0,'test collection failed: '+result.stderr[-1000:])
    nodes=[line.strip() for line in result.stdout.splitlines() if line.startswith('relation_gat_ego_v1/') and '::test_' in line]
    require(nodes and len(nodes)==len(set(nodes)),'test collection empty or duplicated')
    return sorted(nodes)


def _expand(nodes):
    expanded={};missing={}
    for identifier,selectors in COVERAGE.items():
        matches=[];absent=[]
        for selector in selectors:
            found=[n for n in nodes if n==selector or n.startswith(selector+'[')]
            matches.extend(found)
            if not found:absent.append(selector)
        expanded[identifier]=sorted(set(matches))
        if absent:missing[identifier]=absent
    return expanded,missing


def _load_ref(ref):
    require(isinstance(ref,dict) and set(ref)=={'path','sha256'},'artifact reference fields differ')
    require(reference(ref['path'])==ref,'artifact bytes changed: '+str(ref.get('path')))
    return Path(ref['path'])


def _junit(path,production,tests):
    root=ET.parse(path).getroot()
    props={}
    for p in root.findall('.//properties/property'):
        if p.get('name','').startswith('relation_'):
            require(p.get('name') not in props,'duplicate JUnit source property')
            props[p.get('name')]=p.get('value')
    require(json.loads(props.get('relation_source_pins','null'))==production and
            json.loads(props.get('relation_test_pins','null'))==tests,'JUnit source/test pins differ')
    require(props.get('relation_evidence_scope')=='offline_synthetic_and_historical_only','JUnit evidence scope differs')
    outcomes={}
    for case in root.findall('.//testcase'):
        node=case.get('classname','').replace('.','/')+'.py::'+case.get('name','')
        require(node not in outcomes,'duplicate JUnit testcase')
        outcomes[node]='FAIL' if any(case.find(k) is not None for k in ('failure','error')) else 'SKIPPED' if case.find('skipped') is not None else 'PASS'
    require(outcomes,'empty JUnit evidence')
    for suite in root.iter('testsuite'):
        require(int(suite.get('errors','0'))==0,'JUnit suite error')
    return outcomes


def _historical_sources():
    base=ROOT/'runs/state-branch-matrix-seed1-20261112-001/arms/s01_A1/train_u0001/material'
    return [base/'run_9400033.h5',base/'run_9400034.h5']


def _validate_historical(path,production):
    import h5py
    import numpy as np
    value=json.loads(Path(path).read_bytes())
    require(value.get('schema')=='relation-gat-ego-historical-stream-v1' and value.get('status')=='PASS' and
            value.get('source_pins')==production,'historical stream version/status/source pins differ')
    require(value.get('new_physical_shifts')==0 and value.get('model_updates')==0 and
            value.get('labels_used_as_inputs') is False and value.get('raw_observations_copied_to_report') is False,
            'historical stream scope differs')
    refs=value.get('input_source_refs',[]);sources=_historical_sources()
    require(len(refs)==len(sources)==2 and [r.get('path') for r in refs]==[str(p.resolve()) for p in sources],
            'historical fixed complete source inventory differs')
    decisions=terminals=rows=0
    for ref,path in zip(refs,sources):
        require(reference(path)=={k:ref[k] for k in ('path','sha256')} and
                ref.get('selection')=='all_rows_in_original_order','historical source bytes or selection differ')
        digest=hashlib.sha256();n=0
        with h5py.File(path,'r') as handle:
            data=handle['states']
            require(data.dtype==np.float32 and data.ndim==2 and data.shape[1]==1519 and len(data)>1,
                    'historical raw schema differs')
            terminal_flags=[]
            for start in range(0,len(data),16):
                block=np.asarray(data[start:start+16]);digest.update(block.tobytes());n+=len(block)
                require(np.isfinite(block).all(),'historical source nonfinite')
                terminal_flags.extend(block[:,17].tolist())
            require(terminal_flags[-1]==1 and all(t==0 for t in terminal_flags[:-1]),
                    'historical source must be one complete original shift')
        require(ref.get('rows')==n and ref.get('states_sha256')==digest.hexdigest(),'historical raw states hash/count differs')
        rows+=n;terminals+=1;decisions+=n-1
    require(value.get('rows')==rows and rows>2,'historical total rows differ')
    arms=value.get('arm_results',{})
    require(set(arms)==set(CANDIDATES),'historical three-arm inventory differs')
    for result in [value,*arms.values()]:
        lo,hi=result.get('edge_count_min',0),result.get('edge_count_max',0)
        require(type(lo) is int and type(hi) is int and 31<=lo<=hi<=1811,'historical edge bounds differ')
        elapsed=result.get('elapsed_seconds',-1)
        require(isinstance(elapsed,(int,float)) and math.isfinite(elapsed) and elapsed>=0 and
                type(result.get('peak_rss_bytes')) is int and result['peak_rss_bytes']>0,
                'historical time/RSS evidence missing')
    for result in arms.values():
        require(result.get('finite') is True and result.get('decision_rows')==decisions and
                result.get('terminal_rows')==terminals,'historical arm did not verify all original rows')
    return value


def build_acceptance(junit,historical,output=None):
    production=source_pins();tests=test_pins();nodes=collect_test_nodeids()
    junit_ref=reference(junit);historical_ref=reference(historical)
    outcomes=_junit(junit_ref['path'],production,tests)
    require(set(outcomes)==set(nodes),'JUnit does not contain exactly the current full test collection')
    _validate_historical(historical_ref['path'],production)
    expanded,missing=_expand(nodes)
    results={identifier:dict(status='NOT_COVERED' if identifier in missing else
                'PASS' if all(outcomes[n]=='PASS' for n in expanded[identifier]) else 'FAIL',
                tests=expanded[identifier],artifacts=[historical_ref] if identifier=='R3-N26' else [],
                missing_selectors=missing.get(identifier,[])) for identifier in IDS}
    passed=all(v=='PASS' for v in outcomes.values()) and all(r['status']=='PASS' for r in results.values())
    proof=dict(schema=SCHEMA,status='PASS_OFFLINE_ONLY' if passed else 'FAIL_OFFLINE_ONLY',
        source_pins=production,test_pins=tests,pytest_junit=junit_ref,historical_stream=historical_ref,
        collected_tests=nodes,results=results,new_physical_shifts=0,
        evidence_scope='offline_synthetic_tests_and_two_historical_complete_shifts',
        physical_smoke_completed=False,performance_conclusion=False,
        interpreter=str(Path(sys.executable).resolve()))
    if output is not None:
        path=Path(output);require(not path.exists(),'acceptance evidence is immutable')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:json.dump(proof,stream,ensure_ascii=False,sort_keys=True,indent=2);stream.write('\n')
    return proof


def verify_acceptance(proof):
    require(isinstance(proof,dict) and proof.get('schema')==SCHEMA and
            proof.get('status')=='PASS_OFFLINE_ONLY','completed offline acceptance required')
    require(proof.get('new_physical_shifts')==0 and proof.get('physical_smoke_completed') is False and
            proof.get('performance_conclusion') is False and
            proof.get('evidence_scope')=='offline_synthetic_tests_and_two_historical_complete_shifts',
            'offline acceptance scope differs')
    require(proof.get('source_pins')==source_pins() and proof.get('test_pins')==test_pins(),
            'acceptance source/test inventory changed')
    require(proof.get('interpreter')==str(Path(PYTHON).resolve()),'acceptance interpreter differs')
    junit=_load_ref(proof.get('pytest_junit'));historical=_load_ref(proof.get('historical_stream'))
    nodes=collect_test_nodeids();require(proof.get('collected_tests')==nodes,'acceptance collection changed')
    outcomes=_junit(junit,proof['source_pins'],proof['test_pins'])
    require(set(outcomes)==set(nodes) and all(v=='PASS' for v in outcomes.values()),
            'all current collected pytest cases must pass without skips')
    expanded,missing=_expand(nodes);require(not missing,'acceptance contains uncovered requirements')
    results=proof.get('results',{})
    require(set(results)==set(IDS) and len(results)==89,'exact 89-ID acceptance inventory required')
    for identifier in IDS:
        expected=dict(status='PASS',tests=expanded[identifier],missing_selectors=[],
                      artifacts=[proof['historical_stream']] if identifier=='R3-N26' else [])
        require(results[identifier]==expected,'acceptance test/artifact mapping differs: '+identifier)
    _validate_historical(historical,proof['source_pins'])
    return True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--junit',required=True);parser.add_argument('--historical',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args(argv)
    proof=build_acceptance(args.junit,args.historical,args.output)
    verify_acceptance(proof)
    print(json.dumps(dict(status=proof['status'],checks=len(proof['results']),
                         pytest_cases=len(proof['collected_tests']),new_physical_shifts=0)))


if __name__=='__main__':main()
