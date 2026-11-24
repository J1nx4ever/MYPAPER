"""Compare whole ore loads against the same FIFO commitments."""
from dataclasses import dataclass
from time import perf_counter

from single_critic_task_query_v1.targets import GRADE_CENTER
from traditional_dispatch_v2.nominal import MASS_EPS, MASS_TOL, NominalModelError

if __package__:
    from .grade_flow_plan import INNER_LOWER, INNER_UPPER
else:
    from grade_flow_plan import INNER_LOWER, INNER_UPPER


OUTER_LOWER = .26
OUTER_UPPER = .28
MIN_METAL_TONS_GAIN = 2.0
MIN_EFFECT_GRADE = .002


@dataclass(frozen=True)
class Assessment:
    action: int
    ore_tons: float
    windows: tuple[tuple[float, float], ...]

    @property
    def violations(self):
        return tuple(max(0., OUTER_LOWER * q - m, m - OUTER_UPPER * q) for q, m in self.windows)

    @property
    def margin_shortfall(self):
        return sum(weight * max(0., INNER_LOWER * q - m, m - INNER_UPPER * q)
                   for (q, m), weight in zip(self.windows, (1., .6)))

    @property
    def target_deviation(self):
        return sum(weight * abs(m - GRADE_CENTER * q)
                   for (q, m), weight in zip(self.windows, (1., .6)))


@dataclass(frozen=True)
class GuardResult:
    action: int
    reason: str
    original: Assessment | None = None
    selected: Assessment | None = None


class _Deadline(Exception):
    pass


def _run_branch(state, action, check):
    branch = state.clone()
    branch.apply_action(action)
    branch._move(check_deadline=check, dispatch=False)
    inputs = branch.score_inputs()
    window = inputs.first_window
    windows = tuple(zip(inputs.fifo_q[window:window + 2], inputs.fifo_m[window:window + 2], strict=True))
    return Assessment(action, inputs.ore_tons, windows)


def _preserves_production(candidate, original):
    return (candidate.ore_tons + MASS_TOL >= original.ore_tons
            and sum(q for q, _ in candidate.windows) + MASS_TOL >= sum(q for q, _ in original.windows)
            and all(nq > MASS_EPS or oq <= MASS_EPS
                    for (nq, _), (oq, _) in zip(candidate.windows, original.windows, strict=True)))


def guard_action(state, selected, *, deadline):
    """Keep the ore/waste decision and correct only a material ore-blend risk."""
    choices = {option.action: option for option in state.candidates()}
    if selected not in choices or not choices[selected].ore:
        return GuardResult(selected, 'commodity_preserved')
    ore_sites = [action for action, option in choices.items() if option.ore]
    if len(ore_sites) < 2:
        return GuardResult(selected, 'single_ore_route')

    def check():
        if perf_counter() >= deadline:
            raise _Deadline

    try:
        assessments = {action: _run_branch(state, action, check) for action in ore_sites}
    except _Deadline:
        return GuardResult(selected, 'projection_deadline')
    except NominalModelError:
        return GuardResult(selected, 'projection_invalid')
    original = assessments[selected]
    candidates = [row for row in assessments.values()
                  if _preserves_production(row, original)
                  and all(new <= old + MASS_TOL
                          for new, old in zip(row.violations, original.violations, strict=True))]
    winner = min(candidates, key=lambda row: (sum(row.violations), row.margin_shortfall,
                                              row.target_deviation, row.action != selected, row.action))
    materiality = max(MIN_METAL_TONS_GAIN, MIN_EFFECT_GRADE * max(q for q, _ in original.windows))
    if sum(original.violations) - sum(winner.violations) >= materiality:
        reason = 'outer_grade_risk'
    elif (original.margin_shortfall - winner.margin_shortfall >= materiality
          and original.target_deviation > winner.target_deviation + MASS_TOL):
        reason = 'eroded_grade_margin'
    else:
        return GuardResult(selected, 'matching_retained', original, original)
    return GuardResult(winner.action, reason, original, winner)
