"""Low-frequency progress only; no transitions, hidden states or model writes."""
from datetime import datetime, timezone
from pathlib import Path

from online_v1_1.operations import _publish
from .gate import encoded


def write_progress(suite, **values):
    report = dict(schema='business-graph-summary-campaign-progress-v1',
                  campaign_id=suite['campaign_id'],
                  updated_at_utc=datetime.now(timezone.utc).isoformat(), **values)
    path = Path(suite['python_output']) / 'training_progress.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish(path, encoded(report), replace=True)
    return report


def update_callback(suite, cfg, local):
    def publish(event):
        report = write_progress(suite, phase_id=cfg['phase_id'],
            candidate_id=cfg['candidate_id'], update_index=local, **event)
        print('TRAINING_PROGRESS phase=' + cfg['phase_id'] +
              ' status=' + str(report.get('status')) +
              ' epochs=' + str(report.get('completed_epochs', 0)) + '/' +
              str(report.get('epochs', 4)) +
              ' elapsed_seconds=' + str(round(report.get('elapsed_seconds', 0.), 1)), flush=True)
    return publish
