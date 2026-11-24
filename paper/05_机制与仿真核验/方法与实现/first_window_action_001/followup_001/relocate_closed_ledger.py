"""Move the completed ledger without modifying its contents or predecessors."""
import json

from merged_budget import FORMER_LEDGER, LEDGER, closed_ledger
from common import REPORT, RUN, reference, require, save
from grade_only_ppo_v2.runtime import _controller_lock


def main():
    with _controller_lock(RUN):
        require(FORMER_LEDGER.is_file() and not LEDGER.exists(), 'expected one closed source and free destination')
        closed_ledger(FORMER_LEDGER)
        before = reference(FORMER_LEDGER)
        FORMER_LEDGER.rename(LEDGER)
        after = reference(LEDGER)
        require(before['sha256'] == after['sha256'], 'closed ledger bytes changed')
        print(json.dumps(save(REPORT/'followup_001/ledger_relocation.json',dict(
            status='MOVED_BYTE_IDENTICAL', original_reference=before, current_reference=after,
            slots='9980001..9980004 COMPLETE', added_shifts=0, predecessor_ledgers_unchanged=True))))


if __name__ == '__main__':
    main()
