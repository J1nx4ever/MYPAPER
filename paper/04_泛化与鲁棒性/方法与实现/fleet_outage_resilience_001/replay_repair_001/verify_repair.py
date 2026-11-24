"""Check physical identities and exercise all Java Freeform rows without starting a run."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from repair_campaign import IDENTITY_FIELDS, load_suite, source_suite
from verify_package import PROBE
from resilience_campaign import reference


def check(path):
    suite=load_suite(path)
    src=source_suite()
    with TemporaryDirectory() as tmp:
        probe=Path(tmp)/'FreeformProbe.java'
        probe.write_text(PROBE)
        subprocess.run(['javac','-d',tmp,suite['java_helper']['path'],str(probe)],check=True)
        out=subprocess.run(['java','-cp',tmp,'FreeformProbe'],check=True,capture_output=True,text=True)
    with open(suite['identities']['path'],newline='') as f:
        ids=list(csv.DictReader(f))
    for j,row,identity in zip(suite['jobs'],csv.reader(out.stdout.splitlines()),ids,strict=True):
        if any(identity[k]!=str(j[k]) for k in IDENTITY_FIELDS):
            raise ValueError('run/seed CSV mismatch')
        expected=[j['run_index'],j['recording_id'],j['experiment_seed'],j['actor_sampling_seed'],j['event_directory'],
                  511 ^ j['initial_mask'],j['fault_truck_id'],j['fault_trigger_seconds'],j['ore_target'],
                  j['waste_target'],j['waste_stop'],'false','false','false',3,
                  j['expected_rpc_config']['config.runtime_scene.road.6.speedLimit_mps']]
        for i,role in enumerate(j['assignment']):
            expected.extend([str(role!='W').lower(),j['grades'][i],j['half_ranges'][i]])
        if not all(float(v)==e if isinstance(e,float) else v==str(e) for e,v in zip(expected,row[:-1],strict=True)):
            raise ValueError('Java Freeform row differs for '+str(j['run_index']))
    if suite['jobs'][32:]!=src['jobs'][32:]:
        raise ValueError('original 960 frozen rows changed')
    return dict(suite=reference(path),source_u0=suite['source_u0'],java_freeform_rows=len(ids),
                new_shifts=32, original_unclaimed_rows_unchanged=960, excluded_old_receipts=32,
                phase_counts={arm:sum(j['arm']==arm and j['role']=='train' for j in suite['jobs'])
                              for arm in ('warmup','static','dynamic')},
                eval_shifts=sum(j['role']=='eval' for j in suite['jobs']),
                first=ids[0],first_old=ids[32],last=ids[-1],budget=suite['budget_at_preparation'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,required=True)
    print(json.dumps(check(parser.parse_args().suite),ensure_ascii=False,indent=2))
