"""DISPATCH flow matching with an executable, bounded FIFO decision check."""

from dataclasses import dataclass
from itertools import product
from time import perf_counter

from single_critic_task_query_v1.targets import ORE_TARGET_TONS, WASTE_TARGET_TONS
from traditional_dispatch_v2 import decision as fifo
from traditional_dispatch_v2 import dispatch_policy as industrial
from traditional_dispatch_v2.nominal import MASS_TOL, SHIFT_END, NominalModelError, NominalState

if __package__:
    from .v3_flow import FlowPlanError, solve_flow_plan
    from .v3_matching import loss, reserve, targets
else:
    from v3_flow import FlowPlanError, solve_flow_plan
    from v3_matching import loss, reserve, targets


FIFO_POLICY_CONTRACT = {
    'schema': 'dispatch-inspired-fifo-policy-v3',
    'root_coverage': 'all_legal_roots_followed_by_one_common_deterministic_tail',
    'forced_single_action_shortcut': True,
    'cohort_trial_limit': 12,
    'cohort_nomination': 'three_largest_then_lowest_industrial_loss_per_root_rank_then_root',
    'physical_prefix_check': 'expected_truck_id_legal_action_and_reserved_payload',
    'grade_bounds': (.26, .28),
    'grade_epsilon': 1e-12,
    'physical_cost': 'frozen_fifo_joint_cost',
    'feasible_order': 'industrial_loss_then_fifo_joint_cost',
    'infeasible_order': 'fifo_joint_cost_then_industrial_loss',
    'industrial_comparison': 'same_length_actual_dispatch_prefix',
    'partial_cohort_timeout': 'restore_full_root_incumbent',
    'multitruck_evidence': 'executable_nominations_in_completed_final_comparison',
}


@dataclass(frozen=True)
class FIFOChoice(industrial.Choice):
    fifo_candidates: int = 0
    fifo_roots_completed: int = 0
    fifo_trials_nominated: int = 0
    fifo_trials_completed: int = 0
    fifo_trials_rejected: int = 0
    fifo_trials_compared: int = 0
    fifo_selected_from_cohort: bool = False
    fifo_root_cost: float | None = None
    fifo_selected_cost: float | None = None
    fifo_root_feasible: bool | None = None
    fifo_selected_feasible: bool | None = None
    fifo_selected_industrial_loss: float | None = None
    fifo_selected_actions: tuple[int, ...] = ()
    fifo_selected_truck_ids: tuple[int, ...] = ()
    fifo_guard_restored_root: bool = False
    fifo_guidance_error: str | None = None
    fifo_model_error: str | None = None


class _UnexecutablePrefix(NominalModelError):
    """A virtual match differs from the next physically legal dispatch."""


@dataclass(frozen=True, slots=True)
class _Projection:
    cost: float
    feasible: bool
    trucks: tuple
    options: tuple

    @property
    def actions(self):
        return tuple(option.site for option in self.options)

    @property
    def truck_ids(self):
        return tuple(truck.truck_id for truck in self.trucks)


def _complete(inputs):
    lo, hi = FIFO_POLICY_CONTRACT['grade_bounds']
    epsilon = FIFO_POLICY_CONTRACT['grade_epsilon']
    return (inputs.ore_tons >= ORE_TARGET_TONS and inputs.waste_tons >= WASTE_TARGET_TONS
            and all(q > 0 and lo - epsilon <= m / q <= hi + epsilon
                    for q, m in zip(inputs.fifo_q[inputs.first_window:],
                                    inputs.fifo_m[inputs.first_window:], strict=True)))


def _project(state, prefix, cache, match_trucks, check):
    branch = state.clone()
    trucks, options = [], []
    while not branch.finished and len(options) < match_trucks:
        check()
        index = len(options)
        expected = prefix[index] if index < len(prefix) else None
        if expected is None:
            action = fifo._tail_action(branch, check)
        else:
            truck_id, action, _ = expected
            if branch.current_truck != truck_id:
                raise _UnexecutablePrefix('forecast truck is not the next physically actionable truck')
        truck = industrial._options(branch)
        option = next((item for item in truck.choices if item.site == action), None)
        if option is None:
            if expected is None:
                raise NominalModelError('common continuation selected a route outside the physical action mask')
            raise _UnexecutablePrefix('proposed route is outside the physical action mask')
        if expected is not None and expected[2] is not None and abs(option.tons - expected[2]) > MASS_TOL:
            raise _UnexecutablePrefix('proposed payload differs from the physically dispatched load')
        trucks.append(truck)
        options.append(option)
        branch.apply_action(action)
        branch.advance(check_deadline=check)
    if len(options) < len(prefix):
        raise _UnexecutablePrefix('shift ends before the proposed dispatch prefix is complete')
    actions = tuple(option.site for option in options)
    inputs, _ = fifo._finish(branch, actions, cache, match_trucks, SHIFT_END, check)
    return _Projection(fifo._cost(inputs, unload=False), _complete(inputs), tuple(trucks), tuple(options))


def _best(projections, guidance, check):
    if guidance is None:
        scores = [0.0] * len(projections)
    else:
        depth = min(len(projection.options) for projection in projections)
        scores = []
        for projection in projections:
            check()
            scores.append(loss(projection.options[:depth], projection.trucks[:depth], *guidance))

    def key(item):
        projection, match_loss = item
        score = (match_loss, projection.cost) if projection.feasible else (projection.cost, match_loss)
        return (not projection.feasible, *score, projection.actions, projection.truck_ids)

    winner, industrial_loss = min(zip(projections, scores, strict=True), key=key)
    return winner, industrial_loss if guidance is not None else None


def _nominate(state, actions, match_trucks, guidance, check, *, ranks_per_root=3):
    crew = industrial._ready_trucks(state, match_trucks, check)
    nominations = {action: [] for action in actions}
    scored = len(actions)
    waste = state._waste_committed()
    for size in range(len(crew), 1, -1):
        cohort = crew[:size]
        ranked = {action: [] for action in actions}
        for proposed in product(*(truck.choices for truck in cohort)):
            check()
            root = proposed[0].site
            if len(nominations[root]) == ranks_per_root:
                continue
            allocated = reserve(proposed, state._available_tons, state._loads, waste)
            if allocated is None:
                continue
            scored += 1
            rank = (loss(allocated, cohort, *guidance), tuple(option.site for option in allocated))
            prefix = tuple((truck.truck_id, option.site, option.tons)
                           for truck, option in zip(cohort, allocated, strict=True))
            ranked[root].append((rank, prefix))
        for root, candidates in ranked.items():
            remaining = ranks_per_root - len(nominations[root])
            nominations[root].extend(prefix for _, prefix in sorted(candidates)[:remaining])
        if all(len(items) == ranks_per_root for items in nominations.values()):
            break
    prefixes = [nominations[action][rank] for rank in range(ranks_per_root) for action in actions
                if rank < len(nominations[action])]
    return prefixes, scored


class FIFODispatchPolicy(industrial.DispatchPolicy):
    """Replan tonnes as in V2 and check each decision against the shared FIFO model."""

    def __init__(self, *, planning_ms=800, replan_seconds=900, match_trucks=3, cohort_trial_limit=12,
                 solve_flow=solve_flow_plan):
        super().__init__(planning_ms=planning_ms, replan_seconds=replan_seconds, match_trucks=match_trucks)
        if type(cohort_trial_limit) is not int or not 0 <= cohort_trial_limit <= FIFO_POLICY_CONTRACT['cohort_trial_limit']:
            raise ValueError('cohort trial limit must be an integer from zero to twelve')
        self.cohort_trial_limit = cohort_trial_limit
        self._solve_flow = solve_flow

    def choose(self, observation, mask):
        choice = super().choose(observation, mask)
        if isinstance(choice, FIFOChoice):
            return choice
        # Only the inherited one-legal-action shortcut constructs the base type.
        outcome = FIFOChoice(choice.action, choice.kind, fifo_candidates=1)
        self._pending = (outcome, self._pending[1])
        return outcome

    def _guide(self, observation, state, deadline, evidence):
        signature = industrial._signature(state, observation)
        if (self.plan is None or state.now - self.plan.now >= self.replan_seconds
                or signature != self._plan_signature):
            evidence['lp_status'] = 'failed'
            plan = self._solve_flow(observation, state, time_limit_seconds=max(.001, deadline - perf_counter()))
            if plan.status not in ('optimal', 'feasible_limit') or not plan.periods:
                raise FlowPlanError('no feasible flow plan')
            self.plan = plan
            self._plan_signature = signature
            self._assigned = [0.0] * 4
            evidence.update(replanned=True, lp_status=plan.status,
                            flow_shortfall_ore_tons=plan.ore_shortfall_tons,
                            flow_shortfall_waste_tons=plan.waste_shortfall_tons,
                            flow_grade_slack_tons=sum(low + high for low, high in plan.grade_slack_tons))
        return targets(self.plan), self._assigned, *industrial._shovel_calendars(state)

    def _select(self, obs, actions, deadline):
        fallback = industrial._fallback_action(obs.astype('float64'), actions)
        evidence = dict(fifo_candidates=len(actions))
        roots, cache = [], {}

        def check():
            if perf_counter() >= deadline:
                raise industrial._Deadline

        try:
            check()
            state = NominalState.from_observation(obs)
            if tuple(option.site for option in industrial._options(state).choices) != actions:
                raise NominalModelError('nominal root candidate routes differ from the legal mask')
            for action in actions:
                prefix = ((state.current_truck, action, None),)
                roots.append(_project(state, prefix, cache, self.match_trucks, check))
                evidence['fifo_roots_completed'] = len(roots)
        except industrial._Deadline:
            return FIFOChoice(fallback, 'fallback', 'incomplete_fifo_roots_at_deadline', timed_out=True, **evidence)
        except NominalModelError as error:
            return FIFOChoice(fallback, 'fallback', f'fifo_root_projection: {error}',
                              fifo_model_error=str(error), **evidence)

        incumbent, incumbent_loss = _best(roots, None, check)
        selected, selected_loss = incumbent, incumbent_loss
        try:
            check()
            guidance = self._guide(obs, state, deadline, evidence)
            incumbent, incumbent_loss = _best(roots, guidance, check)
            selected, selected_loss = incumbent, incumbent_loss
            evidence.update(matched_trucks=1, assignments_scored=len(actions))
            if self.cohort_trial_limit and self.match_trucks > 1:
                prefixes, scored = _nominate(state, actions, self.match_trucks, guidance, check)
                prefixes = prefixes[:self.cohort_trial_limit]
                evidence.update(fifo_trials_nominated=len(prefixes), assignments_scored=scored)
                trials = []
                compared_depth = 1
                for prefix in prefixes:
                    check()
                    try:
                        trials.append(_project(state, prefix, cache, self.match_trucks, check))
                        evidence['fifo_trials_completed'] = len(trials)
                        compared_depth = max(compared_depth, len(prefix))
                    except _UnexecutablePrefix:
                        evidence['fifo_trials_rejected'] = evidence.get('fifo_trials_rejected', 0) + 1
                if trials:
                    selected, selected_loss = _best(roots + trials, guidance, check)
                    evidence.update(matched_trucks=compared_depth, fifo_trials_compared=len(trials),
                                    fifo_selected_from_cohort=any(selected is trial for trial in trials))
        except industrial._Deadline:
            evidence.update(timed_out=True, fifo_guard_restored_root=True)
            selected, selected_loss = incumbent, incumbent_loss
        except FlowPlanError as error:
            evidence['fifo_guidance_error'] = str(error)
            selected, selected_loss = incumbent, incumbent_loss
        except NominalModelError as error:
            evidence['fifo_model_error'] = str(error)
            selected, selected_loss = incumbent, incumbent_loss

        return FIFOChoice(selected.actions[0], 'planned', **evidence,
                          fifo_root_cost=incumbent.cost, fifo_selected_cost=selected.cost,
                          fifo_root_feasible=incumbent.feasible, fifo_selected_feasible=selected.feasible,
                          fifo_selected_industrial_loss=selected_loss, fifo_selected_actions=selected.actions,
                          fifo_selected_truck_ids=selected.truck_ids)
