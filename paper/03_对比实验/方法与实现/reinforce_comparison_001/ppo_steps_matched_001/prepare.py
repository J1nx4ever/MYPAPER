"""Freeze the S02 physical schedule and the sealed PPO Actor step budget for each batch."""

from functools import lru_cache
from pathlib import Path

from .. import prepare as parent
# The historical budget module also establishes the existing mixed-plan imports.
from .. import budget as _training_budget
from business_graph_summary_campaign_v1.plan_io import load_reference, reference, write
from weak_plan import bind


PLAN = Path(__file__).resolve().parent
ROOT = parent.ROOT
CAMPAIGN = 'bg128-reinforce-ppo-steps-20261128-s02'
FIRST_RUN = 10094001
COUNT, UPDATES, BATCH_SIZE, RPC_PORT = parent.COUNT, parent.UPDATES, parent.BATCH_SIZE, parent.RPC_PORT
SCHEMA = 'bg128-full-return-reinforce-ppo-steps-s02-v1'
PREPARED = PLAN / 'prepared_001'
SUITE = PREPARED / 'suite.json'
OUTPUT = ROOT / 'runs' / CAMPAIGN
MATERIAL = OUTPUT.with_name(CAMPAIGN + '_materials')
CONTROL = OUTPUT / 'control'
ARM = OUTPUT / 'arms/s01_BG128'
REPORT = ROOT / 'reports' / PLAN.relative_to(ROOT / 'change_plans')
LEDGER = OUTPUT / 'budget.sqlite'
INITIAL_ACTOR = parent.INITIAL_ACTOR
historical = parent.historical
original = parent.original


@lru_cache(maxsize=1)
def ppo_evidence():
    """Trace each summary's reported step count to the actual Adam moments in the sealed checkpoint."""
    import torch

    suite, phases, completion, *_ = historical()
    conf = suite['learner_config']['ppo']
    epochs, batch_size = conf['epochs'], conf['minibatch_size']
    original.require((epochs, batch_size) == (4, 32), 'Full S02 Actor replay settings differ')
    rows, previous_steps = [], 0
    for update, (phase, ckpt) in enumerate(zip(phases, completion['training_checkpoints'], strict=True), 1):
        original.require(reference(ckpt['path']) == ckpt, 'sealed PPO checkpoint bytes differ')
        saved = torch.load(ckpt['path'], map_location='cpu', weights_only=False)
        state = saved['runtime_state']
        summary_ref = state['training_summary']
        summary = load_reference(summary_ref)
        info = summary['update']
        transitions, steps = info['transitions'], info['actor_optimizer_steps']
        per_epoch = (transitions + batch_size - 1) // batch_size
        max_adam_step = max(int(value['step']) for value in saved['actor_optimizer']['state'].values())
        original.require(saved['update_index'] == info['update_index'] == update
                         and state['campaign'] == completion['suite']
                         and summary['transaction'] == state['transaction']
                         and summary['transaction']['phase_config'] == suite['phase_configs'][update-1]
                         and summary['transaction']['run_indices'] == [j['run_index'] for j in phase['jobs']]
                         and info['epochs'] == epochs and info['episodes'] == BATCH_SIZE
                         and steps == epochs * per_epoch == max_adam_step - previous_steps,
                         'PPO step schedule differs from its physical or optimizer transaction')
        previous_steps = max_adam_step
        rows.append(dict(update=update, transitions=transitions, epochs=epochs, actor_optimizer_steps=steps,
                         minibatches_per_epoch=per_epoch, training_summary=summary_ref, checkpoint=ckpt))
    original.require(len(rows) == UPDATES and previous_steps == 23612, 'Full S02 cumulative step budget differs')
    return rows


_original_documents = bind(parent.documents, CAMPAIGN=CAMPAIGN, SCHEMA=SCHEMA, FIRST_RUN=FIRST_RUN,
                           OUTPUT=OUTPUT, MATERIAL=MATERIAL, CONTROL=CONTROL, ARM=ARM, REPORT=REPORT,
                           SUITE=SUITE, LEDGER=LEDGER)


def documents():
    suite, phases = _original_documents()
    evidence = ppo_evidence()
    suite['reinforce']['actor_steps_by_update'] = [item['actor_optimizer_steps'] for item in evidence]
    suite.update(ppo_minibatches_per_epoch=[item['minibatches_per_epoch'] for item in evidence],
                 ppo_epochs=4, full_s02_step_evidence=evidence, actor_optimizer_steps_total=23612)
    return suite, phases


def source_pins():
    pins = parent.source_pins()
    learner = ROOT / 'business_graph_matched_reinforce_v1'
    original.require((learner / 'learner.py').is_file(), 'matched learner implementation missing')
    files = [path for path in learner.glob('*.py') if not path.name.startswith('test_')]
    files += [PLAN / name for name in ('prepare.py', 'prepare_java.py', 'budget.py', 'run.py', 'runtime.py')]
    pins.update({str(path): reference(path)['sha256'] for path in files})
    return dict(sorted(pins.items()))


def write_prepared(path, document):
    # The operator's launch roster includes the optimizer schedule alongside the physical identities.
    if path == PREPARED / 'launch_identity.json':
        rows = ppo_evidence()
        document.update(actor_optimizer_steps_total=sum(row['actor_optimizer_steps'] for row in rows),
                        full_s02_step_evidence=rows)
    write(path, document)


_load_package = bind(parent.load_package, documents=documents, SUITE=SUITE, PREPARED=PREPARED, source_pins=source_pins)


def load_package():
    suite, phases = _load_package()
    launch = load_reference(reference(PREPARED / 'launch_identity.json'))
    original.require(launch['suite'] == reference(SUITE) and launch['identities'] == suite['identities']
                     and launch['full_s02_step_evidence'] == suite['full_s02_step_evidence']
                     and launch['actor_optimizer_steps_total'] == suite['actor_optimizer_steps_total']
                     and launch['campaign_id'] == CAMPAIGN and launch['first'] ==
                     next(original._identity_rows(phases, suite['phase_configs'])),
                     'launch roster or published optimizer schedule differs from the frozen suite')
    return suite, phases


prepare = bind(parent.prepare, __package__=__package__, PLAN=PLAN, PREPARED=PREPARED, SUITE=SUITE,
               OUTPUT=OUTPUT, MATERIAL=MATERIAL, CAMPAIGN=CAMPAIGN, documents=documents,
               load_package=_load_package, source_pins=source_pins, write=write_prepared)
