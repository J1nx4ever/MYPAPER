"""Read-only merge of all predecessor ledgers and the completed four-shift batch."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from budget import LEDGER as FORMER_LEDGER, check_new_slots, prior_snapshot
from common import CID, REPORT, RUN, RUN_IDS, checked, read, reference, require

LEDGER = RUN / 'independent_evaluation_ledger.json'


def closed_ledger(path):
    completion = read(REPORT/'completion_validation.json')
    ledger_ref = reference(path)
    require(ledger_ref['sha256'] == completion['ledger']['sha256'] and
            completion['ledger']['path'] == str(FORMER_LEDGER), 'completed ledger SHA differs')
    data = read(path)
    require(data['schema'] == 'bg128-first-window-action-budget-v1' and
            set(data['slots']) == {str(r) for r in RUN_IDS}, 'closed ledger identity differs')
    cfg = read(checked(data['config']))
    require(cfg['campaign_id'] == CID and len(cfg['jobs']) == len(data['slots']),
            'ledger config inventory differs')
    for run, slot in data['slots'].items():
        require(slot['status'] == 'COMPLETE' and read(checked(slot['receipt']))['run_index'] == int(run),
                'expected a completed slot with a matching receipt')
    return data, cfg


def budget_snapshot():
    require(not FORMER_LEDGER.exists(), 'relocate the closed ledger before merging')
    data, cfg = closed_ledger(LEDGER)
    result = prior_snapshot()
    check_new_slots(cfg, result)
    for job in cfg['jobs']:
        key = str(job['run_index'])
        result['jobs'][key] = dict(data['slots'][key], identity=job, campaign_id=CID)
    result['campaigns'][CID] = dict(campaign=cfg, commits={}, cap=len(data['slots']))
    result['maximum_total_shifts'] += len(data['slots'])
    result['additional_reserved_shifts'] += len(data['slots'])
    return result


if __name__ == '__main__':
    result = budget_snapshot()
    print(json.dumps(dict(maximum_total_shifts=result['maximum_total_shifts'],
        ledger=reference(LEDGER), slots={str(run):result['jobs'][str(run)]['status'] for run in RUN_IDS})))
