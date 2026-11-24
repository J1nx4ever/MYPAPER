"""Fast replay first; resolve roundoff only with exact original online replay."""
import struct
import numpy as np
import torch
from .contract import require
from .features_cache import SummaryFeatureCache

def exact_replay(actor, states, actions):
    cache=SummaryFeatureCache(feature_config=actor.feature_config)
    probabilities=[];logs=[]
    with torch.no_grad():
        for index,state in enumerate(states):
            cache.append(state)
            distribution=actor.distribution(torch.tensor(state.copy())[None,:],cache.prefix([index]))
            probabilities.append(distribution.probs[0].clone())
            logs.append(distribution.log_prob(torch.tensor([int(actions[index])]))[0].clone())
    return dict(probabilities=torch.stack(probabilities),log_probs=torch.stack(logs))

def verify(actor, states, actions, logs, replay, *, rows=None, job=None):
    from .admission import _verify_sampling
    from grade_event_memory_v1.events import ContractError
    mismatch=not np.allclose(replay['log_probs'].detach().numpy(),logs,rtol=1e-6,atol=1e-6)
    if rows is not None:
        try: _verify_sampling(rows,replay['probabilities'],job)
        except ContractError as error:
            if str(error)!='actual Actor sampled action/RNG differs': raise
            mismatch=True
    if not mismatch: return 'batch'
    exact=exact_replay(actor,states,actions)
    require(np.allclose(replay['probabilities'].detach().numpy(),exact['probabilities'].numpy(),rtol=1e-6,atol=1e-6),
            'batch/online probability difference exceeds roundoff tolerance')
    require(all(struct.pack('!d',float(a))==struct.pack('!d',float(b)) for a,b in zip(exact['log_probs'],logs)),
            'exact online log probability differs from stored bits')
    if rows is not None: _verify_sampling(rows,exact['probabilities'],job)
    return 'exact_online'

def verify_learner(model, episode, current):
    logs=np.asarray(episode['old_log_probs'])
    require(np.isfinite(current.detach().numpy()).all(),'nonfinite current log probability')
    if np.allclose(current.detach().numpy(),logs,rtol=1e-6,atol=1e-6): return
    require(model.config['source_kind']!='synthetic','synthetic old action probabilities differ')
    from pathlib import Path
    from .plan_io import reference
    from python.constrained_batch import load_dataset
    artifact=model._actor_artifact;identity=episode['identity']
    require(isinstance(artifact,dict) and artifact['actor_digest']==model.policy_version==identity['policy_version']
            and artifact['sha256']==identity['policy_sha256'],'fallback requires current sealed Actor')
    model._validated_actor_snapshot(artifact['path'])
    raw=Path(artifact['path']).parent/'episodes'/f"run_{identity['run_index']}"
    require(reference(raw/'manifest.json')['sha256']==identity['raw_manifest_sha256'],'raw manifest binding differs')
    rows,_,_,_,_=load_dataset(raw,require_real=True)
    try:
        originals=list(rows)
        states=np.stack([originals[0]['observation']]+[r['next_observation'] for r in originals]).astype(np.float32)
        require(states.tobytes()==episode['states'].tobytes() and
                np.array_equal(episode['actions'],[r['action'] for r in originals]) and
                np.asarray([r['old_log_prob'] for r in originals],dtype=np.float64).tobytes()==logs.tobytes(),
                'fallback input differs from sealed raw trajectory')
        with torch.no_grad():
            prefix=model._prefix(states[:-1],list(range(len(originals))))
            distribution=model.actor.distribution(torch.tensor(states[:-1].copy()),prefix)
            replay=dict(probabilities=distribution.probs,log_probs=current)
            verify(model.actor,states[:-1],episode['actions'],logs,replay,rows=originals,
                   job=dict(actor_sampling_seed=identity['actor_sampling_seed']))
    finally:
        if hasattr(rows,'close'): rows.close()
