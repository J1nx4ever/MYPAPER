"""Serve the frozen s02/u32 Actor for a single uninterrupted A-map shift."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from outage_service import FullActor, ScaledTrial, trial_service
import control_campaign as campaign


class ControlTrial(ScaledTrial):
    def __init__(self, label):
        spec, self.sha, self.edges = campaign.read_case(label)
        trial_service.require(spec['layout_id'] == 'tree_A' and spec['outage'] is None
                              and spec['policy_update'] == 32 and spec['action_mode'] == 'argmax'
                              and spec['actor_sampling_seed'] is None and spec['slot_permutation_seed'] == 0,
                              'unexpected control configuration')
        self.spec = spec
        self.output = Path(spec['python_output_directory'])
        self.events = Path(spec['event_directory'])
        trial_service.require(not self.events.exists() and not (self.output/'inference.jsonl').exists(),
                              'physical control identity already started')
        self.active = sum(1 << row[0] for row in spec['expected_runtime_scene']['opening_fleet'] if row[1])
        trial_service.require(self.active == 511, 'expected all nine A trucks')
        self.policy = FullActor(spec, self.edges)
        self.sequence = 0
        self.previous_time = -1.
        self.last_key = self.reply = self.writer = None
        self.sealed = False

    def decide(self, request):
        obs = request['obs']
        trial_service.require(obs[16] == 0 and obs[1107] == 1,
                              'expected stochastic environment and available high-grade shovel')
        return super().decide(request)


if __name__ == '__main__':
    trial_service.Trial = ControlTrial
    trial_service.serve(campaign.case_labels())
