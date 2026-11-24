"""Offline checks for schedule, paired goals, full-window cutoff and FIFO invariants."""
from collections import Counter
from itertools import product
import math

from physical import verify_terminal
from schedule import fault_cells,schedule


def verify():
    cells=Counter()
    for u in range(1,10):
        batch=list(fault_cells(u))
        assert len(set(batch))==16
        for i, expected in [(0,[5,5,6]),(1,[5,5,6]),(2,[4,4,4,4])]:
            assert sorted(Counter(c[i] for c in batch).values())==expected
        cells.update(batch)
    assert cells==Counter({c:4 for c in product(range(7,10),range(3),range(4))})
    jobs=list(schedule('offline',1,1,36,36))
    branches=[[j for j in jobs if j['arm']==arm and j['role']=='train'] for arm in ('static','dynamic')]
    for a,b in zip(*branches,strict=True):
        assert all(a[k]==b[k] for k in ('initial_mask','grades','assignment','half_ranges','ore_target','waste_target'))
    for j in jobs:
        assert j['initial_mask'].bit_count()==j['starting_count'] and 6<=j['starting_count']<=9
        assert math.isclose(j['ore_target'],4*j['waste_target'])
        if j['fault_truck_id']!=-1:
            assert j['initial_mask'] & (1<<j['fault_truck_id'])
    job=dict(initial_mask=511,fault_truck_id=8,fault_trigger_seconds=19000)
    c={'config.mass_balance_epsilon':1e-5}
    c.update({'fleet.'+k:v for k,v in dict(trigger_seconds=19000.,exit_seconds=21600.,dispatchable_mask=255,
            in_system_mask=255,window_ore_tons=2000.,window_metal_tons=540.,incoming_ore_tons=10000.,
            incoming_metal_tons=2700.,processed_ore_tons=9800.,processed_metal_tons=2640.,
            fifo_ore_tons=200.,fifo_metal_tons=60.).items()})
    assert verify_terminal(job,c)['window_compliant']
    for t,valid in ((21599.99,True),(21600.,True),(21600.0000001,False)):
        assert verify_terminal(job,dict(c,**{'fleet.exit_seconds':t}))['window_valid']==valid
    empty=verify_terminal(job,dict(c,**{'fleet.window_ore_tons':0.,'fleet.window_metal_tons':0.}))
    assert empty['window_valid'] and empty['window_empty'] and not empty['window_compliant']
    for key in ('fleet.fifo_ore_tons','fleet.fifo_metal_tons'):
        try:
            verify_terminal(job,dict(c,**{key:c[key]+1.}))
        except ValueError:
            pass
        else:
            raise AssertionError('conservation failed to reject '+key)
    return dict(fault_cells=len(cells),observations_per_cell=4,equal_opening_train_slots=len(branches[0]),
                exit_cutoff_seconds=21600.,rejects_ore_and_metal_imbalance=True)


if __name__=='__main__':
    print(verify())
