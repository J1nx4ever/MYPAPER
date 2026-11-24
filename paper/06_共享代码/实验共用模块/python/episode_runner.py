"""Wait for replications started manually in the user's AnyLogic GUI."""
from typing import Optional


class ParameterVariationEpisodeRunner:
    def __init__(self, total_episodes: Optional[int] = None):
        if total_episodes is not None and total_episodes <= 0:
            raise ValueError('total_episodes must be positive')
        self.total_episodes = total_episodes
        self.episodes_started = 0

    def start_episode(self, seed=None, options=None):
        # The approved manifest and RPC handshake verify seeds; Python never starts the simulator.
        if self.total_episodes is not None and self.episodes_started >= self.total_episodes:
            return False
        self.episodes_started += 1
        return True

    def close(self):
        # The manually started AnyLogic process belongs to the user.
        return None
