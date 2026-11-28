"""Serve the frozen argmax policy for independently seeded A/B validation shifts."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "capacity_scaled_001")]

import trial_service
from scaled_service import ScaledTrial
from prepare_stats import read_case


class ValidationTrial(ScaledTrial):
    def decide(self, request):
        obs = request.get("obs", [])
        trial_service.require(len(obs) == 1519 and obs[16] == 0,
                              "expected the stochastic native environment in the Actor observation")
        return super().decide(request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("A", "B"), action="append")
    args = parser.parse_args()
    labels = [f"{group}_{slot:03d}" for group in args.group or ("A", "B") for slot in range(1, 51)]
    trial_service.read_case = read_case
    trial_service.Trial = ValidationTrial
    trial_service.serve(labels)
