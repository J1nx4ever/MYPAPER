"""Serve the recovered B shift using its independent identity and the frozen u9 actor."""

from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import trial_service
from prepare_retry import read_case


class RepairTrial(trial_service.Trial):
    def decide(self, request):
        try:
            return super().decide(request)
        except Exception:
            self.output.mkdir(parents=True, exist_ok=True)
            failure = self.output / 'first_inference_error.json'
            if not failure.exists():
                with failure.open('xb') as stream:
                    stream.write(trial_service.encode(dict(request=request, traceback=traceback.format_exc())))
            raise


if __name__ == '__main__':
    trial_service.read_case = read_case
    trial_service.Trial = RepairTrial
    trial_service.serve(['B'])
