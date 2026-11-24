"""Grade-controlled strategy with the sealed path-matching policy interface."""
from collections import Counter
from dataclasses import replace
from time import perf_counter

from traditional_dispatch_v2.nominal import NominalModelError

if __package__:
    from ..flow_matching_policy import FlowMatchingPolicy, _Calendar, _remaining_window_tons, _truck_offers
    from .grade_flow_plan import solve_grade_plan
    from .grade_guard import guard_action
else:
    from flow_matching_policy import FlowMatchingPolicy, _Calendar, _remaining_window_tons, _truck_offers
    from grade_flow_plan import solve_grade_plan
    from grade_guard import guard_action


class GradeControlPolicy(FlowMatchingPolicy):
    """Retain industrial flow matching before checking the actual candidate payload."""

    def __init__(self, *, solve_plan=solve_grade_plan, **settings):
        super().__init__(solve_plan=solve_plan, **settings)
        self.guard_counts = Counter()
        self._guard_reasons = Counter()

    def choose(self, observation, mask):
        deadline = perf_counter() + self.planning_ms / 1000.
        choice = super().choose(observation, mask)
        if choice.route[3] or sum(mask) < 2:
            return choice
        if perf_counter() >= deadline:
            self._guard_reasons['projection_deadline'] += 1
            return choice
        try:
            state = _Calendar.from_observation(observation)
        except NominalModelError:
            self._guard_reasons['projection_invalid'] += 1
            return choice
        result = guard_action(state, choice.action, deadline=deadline)
        self._guard_reasons[result.reason] += 1
        if result.action == choice.action:
            return choice
        trucks = tuple(int(kind) for kind in observation[27:1107].reshape(9, 120)[:, 1:4].argmax(axis=1))
        offer = next((item for item in _truck_offers(state, trucks) if item.site == result.action), None)
        if offer is None or not mask[result.action]:
            return choice
        outstanding = _remaining_window_tons(self.plan.routes.get(offer.route, ()),
                                              self._assigned[offer.route], offer.ready) if self.plan else 0.
        new = replace(choice, action=offer.site, route=offer.route, assigned_tons=offer.tons,
                      need_time=None, loss_tons=None, oversupply_tons=max(0., offer.tons - outstanding)
                      if self.plan else 0., deadhead_seconds=offer.empty, matched_trucks=1,
                      matching_kind='grade_correction', provisional_trucks=(offer.truck,))
        self._pending = new, self._pending[1]
        self.guard_counts[choice.matching_kind or choice.kind] += 1
        return new

    def summary(self):
        result = super().summary()
        result['grade_control'] = dict(corrections_by_original_branch=dict(self.guard_counts),
                                       assessments_by_reason=dict(self._guard_reasons))
        return result
