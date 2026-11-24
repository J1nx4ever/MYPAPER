"""Compare truck assignments against fluid tonnage targets and visible shovel queues."""
from dataclasses import replace
from heapq import heapify, heapreplace
from math import inf, isfinite

from traditional_dispatch_v2.dispatch_policy import _targets as targets
from traditional_dispatch_v2.nominal import MASS_EPS, MASS_TOL, WASTE_STOP_TONS


def need_time(targets, assigned):
    """Inverse site target at the next assigned tonne, using half-open intervals."""
    remaining = max(0.0, assigned)
    for target in targets:
        if target.tons <= 0:
            continue
        if remaining < target.tons:
            rate = target.tons / (target.stop - target.start)
            return min(target.stop, target.start + remaining / rate), rate
        remaining = max(0.0, remaining - target.tons)
    return inf, 0.0


def loss(options, trucks, targets, committed, servers, bookings):
    queue = [events.copy() for events in bookings]
    assigned = committed.copy()
    remaining_targets = [sum(target.tons for target in site) for site in targets]
    needs = []
    for index, option in enumerate(options):
        site = option.site
        due, rate = need_time(targets[site], assigned[site])
        covered = min(option.tons, max(0.0, remaining_targets[site] - assigned[site]))
        needs.append((due, rate, covered))
        assigned[site] += option.tons
        # The already-waiting line precedes same-time arrivals. Visible and
        # newly predicted arrivals share the nominal model's truck-ID tie.
        queue[site].append((option.arrival, 1, trucks[index].truck_id, option.loading_seconds, index))
    total = 0.0
    for site, arrivals in enumerate(queue):
        available = servers[site].copy()
        heapify(available)
        for arrival, _, _, duration, choice in sorted(arrivals, key=lambda row: row[:3]):
            start = max(available[0], arrival)
            heapreplace(available, start + duration)
            if choice is None:
                continue
            option = options[choice]
            due, feed, covered = needs[choice]
            truck_rate = option.tons / option.cycle_seconds
            total += truck_rate * (start - arrival + option.empty_seconds - trucks[choice].shortest_empty)
            total += option.tons - covered
            if isfinite(due) and covered:
                dispatch_time = option.arrival - option.empty_seconds
                total += min(covered, feed * abs(dispatch_time - due))
    return total


def reserve(options, available, loads, waste):
    remaining = available.copy()
    allocated = []
    for index, option in enumerate(options):
        site, load = option.site, loads[option.site]
        # The root offer and its legality come from the live double-precision
        # mask, while stocks and the virtual waste ledger decode from float32.
        if index and not load.ore and waste >= WASTE_STOP_TONS - MASS_EPS:
            return None
        tons = option.tons if index == 0 else min(option.tons, remaining[site])
        if tons <= MASS_EPS or tons > remaining[site] + MASS_TOL:
            return None
        if not load.ore:
            waste += tons
        residual = remaining[site] - tons
        # Use the same tolerance as NominalState.apply_action so a decoding
        # residual cannot release a phantom second truck in the matcher.
        remaining[site] = residual if residual > MASS_TOL else 0.0
        if tons != option.tons:
            duration = load.service(tons)
            option = replace(option, tons=tons, loading_seconds=duration,
                             cycle_seconds=option.cycle_seconds - option.loading_seconds + duration)
        allocated.append(option)
    return tuple(allocated)
