"""Reproduce/verify the corrected offline identity without starting it."""
from common import *

def main():
    cfg=read(PLAN/'minibatch_replay_config.json')
    suite=read(cfg['source_training_suite']['path']);assert suite['cpu_threads']==2
    cfg['diagnostic_id']='bg128-late-minibatch-replay-20261118-002'
    cfg['cpu_threads']=suite['cpu_threads'];cfg['python_output']=str(OUT/'minibatch_replay_002')
    cfg['replay_implementation']=ref(PLAN/'minibatch_replay_v2.py')
    cfg['supersedes_rejected_reconstruction']=ref(PLAN/'minibatch_replay_config.json')
    payload=(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode()
    p=PLAN/'minibatch_replay_config_v2.json'
    if p.exists():assert p.read_bytes()==payload,'do not overwrite an existing identity'
    else:
        with p.open('xb') as f:f.write(payload)
    print(json.dumps(dict(status='CONFIG_VERIFIED_NO_LAUNCH',config=ref(p))))

if __name__=='__main__':main()
