"""Exact four-slot smoke wrapper around the shared durable physical ledger."""
from contextlib import contextmanager
import copy
import hashlib

from baseline1_v1.budget import BudgetLedger as SharedBudgetLedger,GLOBAL_PATH
from .gate import CAMPAIGN_CONTRACT,PROFILE,PURPOSE,DIAGNOSTIC_SCOPE,check_approval,require


class BudgetLedger(SharedBudgetLedger):
    def __init__(self,path,campaign,*,campaign_sha256,approval,approval_sha256,test_only=False):
        require(campaign.get('contract')==CAMPAIGN_CONTRACT and campaign.get('execution_profile')==PROFILE
                and campaign.get('purpose')==PURPOSE and campaign.get('diagnostic_scope')==DIAGNOSTIC_SCOPE
                and campaign.get('new_shift_cap')==len(campaign.get('jobs',[]))==4,
                'only exact four-job real smoke may register through this wrapper')
        check_approval(approval,campaign,campaign_sha256)
        require(test_only or {'budget_before','maximum_total_if_approved'}<=campaign.keys(),
                'production smoke registration needs reviewed snapshot and exact total')
        self._registration_pending=True;self._registration_campaign=copy.deepcopy(campaign)
        super().__init__(path,campaign,campaign_sha256=campaign_sha256,approval=approval,
                         approval_sha256=approval_sha256,test_only=test_only)

    @contextmanager
    def _locked(self):
        with super()._locked():
            if self._registration_pending:
                if not self.test_only:
                    c=self._registration_campaign
                    require(self.path.is_file(),'reviewed shared budget must already exist')
                    data=self._read()
                    if c['campaign_id'] not in data['campaigns']:
                        before=c['budget_before']
                        require(str(self.path)==before.get('path') and
                                hashlib.sha256(self.path.read_bytes()).hexdigest()==before.get('sha256'),
                                'budget changed before atomic smoke registration; reconcile first')
                        require(data['maximum_total_shifts']+4==c['maximum_total_if_approved'],
                                'approved atomic smoke global cap differs')
                    else:
                        require(data['maximum_total_shifts']==c['maximum_total_if_approved'],
                                'registered global cap changed; review before smoke resume')
                self._registration_pending=False
            yield
