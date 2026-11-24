"""Inventory the final diagnostic artifacts, including the rejected attempt."""
from datetime import datetime,timezone
from common import *

def main():
    assert read(OUT/'validation.json')['status']=='PASS'
    assert read(OUT/'research_findings.json')['status']=='PASS_BOUNDED_OFFLINE_RESEARCH'
    for name in ('minibatch_replay_config.json','minibatch_replay_config_v2.json'):
        cfg=read(PLAN/name);assert ref(cfg['replay_implementation']['path'])==cfg['replay_implementation']
    files=[p for folder in (PLAN,OUT) for p in folder.rglob('*')
           if p.is_file() and p.suffix in ('.py','.json','.csv','.md') and p.name!='artifact_manifest.json']
    save('artifact_manifest.json',dict(schema='bg128-next-stage-mechanism-artifact-manifest-v1',
        status='COMPLETE_BOUNDED_OFFLINE_ANALYSIS',created_at_utc=datetime.now(timezone.utc).isoformat(),
        files=[ref(p) for p in sorted(files)],protected_ppo=ref(CORE),
        new_physical_shifts=0,official_training_updates=0,
        note='Includes the rejected one-thread reconstruction for audit history; only exact accepted two-thread reconstructions enter historical conclusions.'))
    print(json.dumps(dict(status='SEALED',files=len(files),manifest=ref(OUT/'artifact_manifest.json'))))

if __name__=='__main__':main()
