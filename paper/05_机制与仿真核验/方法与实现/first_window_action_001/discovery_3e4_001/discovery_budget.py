"""Single-writer discovery ledger under the campaign lock, with complete history."""
import json

from discovery_common import (APPROVAL, CAMPAIGN, COUNT, OUTPUT, SUITE, check_fresh_jobs,
                              checked, encoded, previous_budget, read, reference, require)
from online_v1_1.operations import _publish

LEDGER = OUTPUT/'independent_discovery_ledger.json'
FORMAT = 'bg128-discovery-100-ledger-v1'


def _state(suite, data):
    jobs = {str(job['run_index']):job for job in suite['jobs']}
    require(data['schema']==FORMAT and data['suite']==reference(SUITE) and
            data['approval']==reference(APPROVAL) and set(data['slots'])==set(jobs), 'ledger header or inventory differs')
    states = {}
    for key, row in data['slots'].items():
        require(row['status'] in ('PLANNED','CLAIMED','COMPLETE','UNKNOWN'), 'unexpected slot state')
        if row['status']=='COMPLETE':
            receipt = row['receipt']
            require(receipt['run_index']==jobs[key]['run_index'] and receipt['experiment_seed']==jobs[key]['experiment_seed'] and
                    receipt['phase_config']==suite['phase_config'] and receipt['normal_completion'] is True and
                    receipt['references']['episode_manifest']==row['dataset_ref'], 'completed physical receipt differs')
        states[key] = dict(row, identity=jobs[key], campaign_id=CAMPAIGN)
    return dict(jobs=states, campaigns={CAMPAIGN:dict(campaign=suite,cap=COUNT,commits={})})


def budget_snapshot():
    previous = previous_budget()
    if not LEDGER.exists():
        return previous
    data = read(LEDGER)
    suite = read(checked(data['suite']))
    check_fresh_jobs(suite['jobs'], previous)
    state = _state(suite,data)
    previous['jobs'].update(state['jobs'])
    previous['campaigns'].update(state['campaigns'])
    previous['maximum_total_shifts'] += COUNT
    previous['additional_reserved_shifts'] += COUNT
    return previous


class DiscoveryLedger:
    def __init__(self, suite):
        previous = previous_budget()
        check_fresh_jobs(suite['jobs'], previous)
        require(previous['maximum_total_shifts']==suite['occupied_before'] and
                previous['maximum_total_shifts']+COUNT==suite['maximum_total_if_approved'],
                'merged budget changed since freeze')
        self.suite = suite
        self.data = dict(schema=FORMAT,suite=reference(SUITE),approval=reference(APPROVAL),
            slots={str(j['run_index']):dict(status='PLANNED') for j in suite['jobs']})
        _publish(LEDGER,encoded(self.data))

    def snapshot(self):
        require(read(LEDGER)==self.data, 'concurrent ledger mutation')
        return _state(self.suite,self.data)

    def resume(self):
        state = self.snapshot()
        require(all(row['status'] not in ('CLAIMED','UNKNOWN') for row in state['jobs'].values()),
                'outstanding physical slot requires reconciliation')
        return state

    def _set(self, job, previous, state):
        key = str(job['run_index'])
        row = self.snapshot()['jobs'][key]
        require(row['identity']==job and row['status']==previous,'unexpected physical slot transition')
        updated = dict(self.data, slots=dict(self.data['slots'], **{key: state}))
        _state(self.suite, updated)
        _publish(LEDGER,encoded(updated),replace=True)
        self.data = updated

    def claim(self, job):
        self._set(job,'PLANNED',dict(status='CLAIMED'))

    def complete(self, job, dataset_ref, receipt):
        checked(dataset_ref)
        self._set(job,'CLAIMED',dict(status='COMPLETE',dataset_ref=dataset_ref,receipt=receipt))

    def unknown(self, job, reason):
        self._set(job,'CLAIMED',dict(status='UNKNOWN',reason=reason))


if __name__=='__main__':
    result = budget_snapshot()
    print(json.dumps(dict(maximum_total_shifts=result['maximum_total_shifts'],
        campaign_registered=CAMPAIGN in result['campaigns'])))
