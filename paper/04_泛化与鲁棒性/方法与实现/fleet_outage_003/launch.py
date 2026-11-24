"""Check a prepared run and start the Python listener on the matching Java START."""
import importlib.util
from pathlib import Path
import sys

import campaign as run_plan
from campaign import bind


def _template():
    spec = importlib.util.spec_from_file_location('fleet_template_launcher', run_plan.TEMPLATE/'launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preflight(suite):
    card = run_plan.load_reference(run_plan.reference(run_plan.PREPARED/'launch_identity.json'))
    run_plan.require(card['suite'] == run_plan.reference(run_plan.SUITE)
                     and card['java_helper'] == suite['java_helper'] and card['run_seed'] == suite['run_seed'],
                     'launch identity card differs from the frozen suite')
    run_plan.require(not run_plan.OUTPUT.exists(), 'existing six-truck activity requires reconciliation')
    return dict(status='PREFLIGHT_PASS_NO_LISTENER', campaign_id=run_plan.CAMPAIGN, physical_shifts=run_plan.COUNT,
                suite=card['suite'], java_helper=card['java_helper'], run_seed=card['run_seed'],
                budget=run_plan.check_fresh(suite, run_plan.preceding_snapshot()))


def main():
    # The inherited controller imports the six-truck runtime by module name.
    sys.path.insert(0, str(run_plan.PLAN))
    launcher = _template()
    entry = Path(__file__).resolve()
    launch_dir = run_plan.OUTPUT/'launch'
    shared = {key: getattr(run_plan, key) for key in
              ('CAMPAIGN', 'CORE', 'COUNT', 'CONTROL', 'OUTPUT', 'PREPARED', 'SUITE',
               'AUTHORITY', 'LEDGER', 'Ledger', 'authority_for', 'load_suite')}
    shared.update(ENTRY=entry, LAUNCH_DIR=launch_dir, INTENT=launch_dir/'launch_intent.json',
                  LANE=run_plan.LOCKS/f"port-{run_plan.EXECUTION['port']}.lock",
                  STATUS=run_plan.OUTPUT/'run_status.json', STOP=run_plan.CONTROL/'STOP.properties')
    shared['_intent'] = bind(launcher._intent, **shared)
    shared['_status'] = bind(launcher._status, **shared)
    shared['_preflight'] = _preflight
    shared['_start'] = bind(launcher._start, **shared)
    entrypoint = bind(launcher.main, **shared, _controller=bind(launcher._controller, **shared),
                      _view_status=bind(launcher._view_status, **shared))
    entrypoint()


if __name__ == '__main__':
    main()
