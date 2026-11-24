"""Compare only completed original/alternative groups with their historical context."""
from common import require


def compare(cfg, results, suite_ref):
    pairs = []
    for run, candidate in cfg['cases'].items():
        original, alternative = [r for r in results if str(r['historical_run']) == run]
        require(original['branch']=='original' and alternative['branch']=='alternative' and
                original['historical_complete_match'] and original['common_prefix_match'] and
                alternative['common_prefix_match'], 'paired source binding differs')
        old, new = [r['physical']['windows'][0] for r in (original, alternative)]
        require(not old['compliant'], 'historical first-window failure was not reproduced')
        change = {key: new[key]-old[key] for key in ('tons','metal_tons')}
        change['grade'] = new['grade']-old['grade'] if new['grade'] is not None else None
        displaced = {}
        for source, old_source in original['source_windows'][0]['sources'].items():
            new_source = alternative['source_windows'][0]['sources'][source]
            displaced[source] = {k: new_source[k]-old_source[k] for k in ('q','m')}
        altered = any(abs(v)>1e-5 for row in displaced.values() for v in row.values())
        for k, field in (('q','tons'),('m','metal_tons')):
            require(abs(sum(v[k] for v in displaced.values())-change[field])<=1e-5,
                    'first-window source attribution does not close')
        if new['compliant'] and alternative['physical']['complete_task']:
            conclusion = 'single_substitution_repaired_complete_task'
        elif new['compliant']:
            conclusion = 'single_substitution_repaired_first_window'
        elif altered:
            conclusion = 'processed_mix_changed_without_first_window_repair'
        else:
            conclusion = 'no_measured_processed_mix_change'
        pairs.append(dict(historical_failed_run=int(run), historical_success=candidate['historical_success'],
            original=original, alternative=alternative, first_window_change=change,
            first_window_source_qm_change=displaced, processed_mix_changed=altered,
            substituted_trip_processed_before_deadline=alternative['chosen_trip']['first_window_processed_tons'] > 1e-6,
            conclusion=conclusion))
    return dict(status='COMPLETE', suite=suite_ref, cases=pairs, physical_shifts=4,
        optimizer_updates=0, inference_checkpoint=cfg['checkpoint'],
        random_coupling='same initialized simulator and Actor PRNG streams; post-branch draws can be assigned to different events',
        negative_result_scope='The tested substitution only. Earlier physical feasibility beyond the existing bounds remains open.')
