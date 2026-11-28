"""Evaluate the sealed step-matched u32 through the original read-only runtime."""
from .. import runtime as training
from ...paired_inference_001 import runtime as prior
from weak_plan import bind


_frozen_policy = bind(prior._frozen_policy, training=training)
verify = bind(prior.verify, _frozen_policy=_frozen_policy)
run = bind(prior.run, _frozen_policy=_frozen_policy, verify=verify)
