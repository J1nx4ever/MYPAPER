"""Serve u9 and verify the goal and remaining waste that it sees at each dispatch."""

from pathlib import Path
import sys
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import trial_service
from prepare_scaled import read_case


def check_observed_targets(obs, task):
    values = np.asarray(obs, dtype=np.float32)
    if values.shape != (1519,):
        raise ValueError("missing physical dispatch observation")
    expected = np.float32([task["ore_target_tons"] / 20000, task["waste_target_tons"] / 4000])
    if not np.array_equal(values[11:13], expected):
        raise ValueError("Actor observation disagrees with opening ore and waste targets")
    remaining_waste = max(0., task["waste_target_tons"] / 4000 - float(values[9]))
    if not np.isclose(values[10], remaining_waste, rtol=0., atol=2e-7):
        raise ValueError("Actor waste shortfall disagrees with scaled target and discharged waste")


class ScaledTrial(trial_service.Trial):
    def decide(self, request):
        try:
            check_observed_targets(request["obs"], self.spec["task"])
            return super().decide(request)
        except Exception:
            self.output.mkdir(parents=True, exist_ok=True)
            error_file = self.output / "first_inference_error.json"
            if not error_file.exists():
                with error_file.open("xb") as stream:
                    stream.write(trial_service.encode(dict(request=request, traceback=traceback.format_exc())))
            raise


if __name__ == "__main__":
    trial_service.read_case = read_case
    trial_service.Trial = ScaledTrial
    trial_service.serve(["B"])
