"""Serve the five fixed B policies on validation, or the selected policy on test."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "capacity_scaled_001")]

import trial_service
from scaled_service import ScaledTrial
import campaign
from topology_learner import activate, checkpoint_sha, InferenceActor, load_learner


@lru_cache(maxsize=1)
def selected_update():
    path = campaign.PROJECT / "runs" / campaign.CAMPAIGN / "selection.json"
    selection = json.loads(path.read_bytes())
    update = selection["selected_update"]
    if selection["validation_cases"] != 80 or selection["checkpoint_sha256"] != checkpoint_sha(update):
        raise ValueError("selection does not bind a verified validation checkpoint")
    evidence = selection["validation_report"]
    if trial_service.digest(Path(evidence["path"])) != evidence["sha256"]:
        raise ValueError("validation results differ from frozen selection")
    return update


@lru_cache(maxsize=5)
def learner(update):
    return load_learner(update)


class EvalTrial(ScaledTrial):
    def __init__(self, label):
        spec, self.sha, self.edges = campaign.read_case(label)
        trial_service.require(spec["phase"] in ("validation", "test") and spec["action_mode"] == "argmax"
                              and spec["actor_sampling_seed"] is None, "expected a frozen evaluation shift")
        update = selected_update() if spec["phase"] == "test" else spec["policy_update"]
        binding = activate(label, update)
        self.spec = dict(spec, checkpoint_sha256=binding["checkpoint_sha256"])
        self.output = Path(spec["python_output_directory"])
        self.events = Path(spec["event_directory"])
        trial_service.require(not self.events.exists() and not (self.output / "inference.jsonl").exists(),
                              "evaluation identity already started")
        self.active = sum(1 << r[0] for r in spec["expected_runtime_scene"]["opening_fleet"] if r[1])
        self.policy = InferenceActor(learner(update), self.spec, self.edges)
        self.sequence = 0
        self.previous_time = -1.
        self.last_key = self.reply = self.writer = None
        self.sealed = False

    def decide(self, request):
        trial_service.require(request["obs"][16] == 0, "expected stochastic native B conditions")
        return super().decide(request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validation", "test"))
    phase = parser.parse_args().phase
    prefix = "V_" if phase == "validation" else "E_"
    trial_service.Trial = EvalTrial
    trial_service.serve([label for label in campaign.CASES if label.startswith(prefix)])
