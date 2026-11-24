"""Add the four experiment slots to the complete read-only budget view."""
from common import *
from business_graph_stability_evaluation_v1.budget import budget_snapshot as prior_snapshot
from physical_curriculum_v1.budget import _history_rows

LEDGER = ROOT / 'runs/independent-evaluation-budget-v1.bg128-first-window-action.json'


def check_new_slots(cfg, before):
    rows = _history_rows(before)
    jobs = cfg['jobs']
    require(tuple(j['run_index'] for j in jobs) == RUN_IDS, 'four prescribed slots required')
    require(CID not in before['campaigns'], 'campaign already registered')
    for job in jobs:
        case = cfg['cases'][str(job['historical_run'])]
        permitted = {case['historical_failed']['run_index'], case['historical_success']['run_index']}
        for row in rows:
            require(row.get('run_index') != job['run_index'] and
                    RECORDING not in (row.get('recording_id'), row.get('source_recording_id')),
                    'physical run or recording identity occupied')
            for field in ('experiment_seed', 'actor_sampling_seed'):
                value = job[field]
                require(value not in (row.get('actor_sampling_seed' if field == 'experiment_seed' else 'experiment_seed'),
                                      row.get('actor_seed')), 'seed stream collision')
                if row.get(field) == value:
                    require(row.get('run_index') in permitted, 'seed overlap outside the case pair')
        for field in ('experiment_seed', 'actor_sampling_seed'):
            sources = {row.get('run_index') for row in rows if row.get(field) == job[field]}
            require(sources == permitted, 'the historical seed pairing is incomplete')


def budget_snapshot():
    result = prior_snapshot()
    if not LEDGER.exists():
        return result
    ledger = read(LEDGER)
    require(ledger['schema'] == 'bg128-first-window-action-budget-v1', 'unknown action experiment ledger')
    cfg = read(checked(ledger['config']))
    require(cfg['campaign_id'] == CID and len(ledger['slots']) == 4, 'ledger campaign differs')
    check_new_slots(cfg, result)
    for job in cfg['jobs']:
        key = str(job['run_index'])
        slot = ledger['slots'][key]
        require(slot['status'] in ('PLANNED', 'CLAIMED', 'COMPLETE', 'UNKNOWN'), 'invalid physical slot status')
        result['jobs'][key] = dict(slot, identity=job, campaign_id=CID)
    result['campaigns'][CID] = dict(campaign=cfg, commits={}, cap=4)
    result['maximum_total_shifts'] += 4
    result['additional_reserved_shifts'] += 4
    return result


class Ledger:
    """Single campaign writer under the existing exclusive controller lock."""
    def __init__(self, cfg, config_ref):
        before = budget_snapshot()
        check_new_slots(cfg, before)
        require(before['maximum_total_shifts'] == cfg['occupied_before'], 'full budget changed since preparation')
        self.state = dict(schema='bg128-first-window-action-budget-v1', config=config_ref,
            slots={str(job['run_index']): dict(status='PLANNED') for job in cfg['jobs']})
        save(LEDGER, self.state)

    def set(self, job, previous, status, **details):
        key = str(job['run_index'])
        require(self.state['slots'][key]['status'] == previous, 'physical slot state differs')
        self.state['slots'][key] = dict(status=status, **details)
        _publish(LEDGER, encoded(self.state), replace=True)
