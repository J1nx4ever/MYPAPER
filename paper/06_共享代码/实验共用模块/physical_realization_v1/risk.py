"""FIFO-only optimistic grade relaxation, with outward interval arithmetic.

This is a one-sided *conditional* exclusion, never a feasibility test or mask.
Only the caller's current float32 snapshot is accepted. No journal, RNG, model,
file or future trajectory is used. A fixed work-conserving crusher and immutable
initial FIFO are assumptions, not guarantees of actual future operation.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from grade_commitment_ppo_v1 import features as old


def _down(x):
    return math.nextafter(float(x), -math.inf)


def _up(x):
    return math.nextafter(float(x), math.inf)


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self):
        if not (math.isfinite(self.lo) and math.isfinite(self.hi) and self.lo <= self.hi):
            raise ValueError('finite ordered interval required')

    def add(self, other):
        return Interval(_down(self.lo+other.lo), _up(self.hi+other.hi))

    def sub(self, other):
        return Interval(_down(self.lo-other.hi), _up(self.hi-other.lo))

    def mul(self, other):
        products = [a*b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return Interval(_down(min(products)), _up(max(products)))

    def nonnegative(self):
        return Interval(max(0., self.lo), max(0., self.hi))

    def minimum(self, other):
        return Interval(min(self.lo, other.lo), min(self.hi, other.hi))

    def ratio_upper(self, denominator):
        if denominator.lo <= 0:
            return None
        return _up(self.hi/denominator.lo)


def encoded_interval(value, scale=1.):
    """Envelope of round-to-nearest float32 encoding, plus float64 guard.

    Use a full neighbouring ULP instead of a half ULP so tie rounding and
    arithmetic in Java's normalization are covered conservatively. The source
    must be an ordinary finite nonnegative physical quantity divided by scale.
    """
    x = np.float32(value)
    if not np.isfinite(x) or x < 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('nonnegative finite float32 field and positive scale required')
    before = float(np.nextafter(x, np.float32(-math.inf)))
    after = float(np.nextafter(x, np.float32(math.inf)))
    return Interval(max(0., _down(before*scale)), _up(after*scale))


@dataclass(frozen=True)
class FifoRisk:
    grade_upper: float | None
    point_grade_upper: float | None
    certificate_valid: bool
    recovery_excluded: bool
    post_fifo_ideal_q: float
    fifo_prefix_q: float
    fifo_prefix_m: float
    assumptions: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


def fifo_risk(state):
    s = old._state(state)
    assumptions = ('fixed_work_conserving_crusher_rate', 'immutable_initial_fifo',
                   'all_future_ore_grades_within_visible_source_or_loaded_support',
                   'float32_round_to_nearest_normalized_snapshot')
    if s[17]:
        return FifoRisk(None, None, False, False, 0., 0., 0., assumptions, ('terminal',))
    cfg = old.FEATURE_CONFIG
    fifo = old._fifo(s, cfg)
    q0, m0 = s[4]*5000, s[5]*5000
    duration, rate = s[3]*7200, s[1304]*2500/3600
    capacity = duration*rate
    remaining = capacity
    fq = fm = 0.
    for q, grade, *_ in fifo:
        taken = min(q, remaining)
        fq += taken
        fm += taken*grade
        remaining -= taken
    support = [min(1., load[2]+load[3]) for load in s[1107:1295].reshape(4, 47) if load[1]]
    support += [t[22] for t in s[27:1107].reshape(9, 120) if t[17] and t[18] and t[21]]
    unknown_loaded_grade = any(t[17] and t[18] and not t[21]
                               for t in s[27:1107].reshape(9, 120))
    if unknown_loaded_grade:
        # The source prior is not a safe upper support for an unidentified
        # loaded packet. Relax to the global physical grade domain instead.
        support.append(1.)
    if not support:
        return FifoRisk(None, None, False, False, remaining, fq, fm, assumptions,
                        ('grade_support_unavailable',))
    gmax = max(support)
    qb, mb = q0+fq, m0+fm
    point = None if qb <= 0 else max(mb/qb, (mb+gmax*remaining)/(qb+remaining))

    q0i, m0i = encoded_interval(s[4], 5000), encoded_interval(s[5], 5000)
    ci = encoded_interval(s[3], 7200).mul(encoded_interval(s[1304], 2500/3600))
    left = ci
    fqi, fmi = Interval(0., 0.), Interval(0., 0.)
    for valid, q, grade in s[1391:1439].reshape(16, 3):
        if not valid:
            break
        take = encoded_interval(q, 500).minimum(left).nonnegative()
        fqi = fqi.add(take)
        fmi = fmi.add(take.mul(encoded_interval(grade)))
        left = left.sub(take).nonnegative()
    gmax_hi = []
    for load in s[1107:1295].reshape(4, 47):
        if load[1]:
            gmax_hi.append(min(1., encoded_interval(load[2]).add(encoded_interval(load[3])).hi))
    for truck in s[27:1107].reshape(9, 120):
        if truck[17] and truck[18] and truck[21]:
            gmax_hi.append(min(1., encoded_interval(truck[22]).hi))
    if unknown_loaded_grade:
        gmax_hi.append(1.)
    qi, mi = q0i.add(fqi), m0i.add(fmi)
    tail = ci.sub(fqi).nonnegative()
    upper_without = mi.ratio_upper(qi)
    upper_with = mi.add(tail.mul(Interval(0., max(gmax_hi)))).ratio_upper(qi.add(tail))
    # With no processed/prefix material, the optimistic solution may process
    # nothing and leave grade undefined. Do not issue a numerical certificate.
    if upper_without is None or upper_with is None:
        return FifoRisk(None, point, False, False, remaining, fq, fm, assumptions,
                        ('undefined_positive_denominator',))
    upper = min(1., max(upper_without, upper_with))
    # lower's encoding uncertainty is conservative for the one-sided test.
    lower = encoded_interval(s[13]).lo
    reasons = ('unknown_loaded_grade_relaxed_to_physical_upper_1',) if unknown_loaded_grade else ()
    return FifoRisk(upper, point, True, upper < lower, remaining, fq, fm, assumptions, reasons)
