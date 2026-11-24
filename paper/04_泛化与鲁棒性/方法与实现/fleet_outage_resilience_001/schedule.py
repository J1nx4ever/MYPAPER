"""Build independent training streams and frozen four-way evaluation pairs."""
import hashlib
from itertools import permutations
from random import Random

CAPACITIES = (91.7, 147., 193.)
EXTRAS = ((0, 1), (1, 0), (2, 2), (0, 1))
HELD_ORDERS = frozenset(("LMHW", "HLMW", "MHLW", "LHWM", "HMWL"))


def seed(campaign, *parts):
    data = "|".join(map(str, (campaign, *parts))).encode()
    return 1 + int.from_bytes(hashlib.sha256(data).digest()[:8], "big") % ((1 << 48) - 1)


def fault_cells(update):
    step = (update - 1) % 9
    n_shift, type_shift = step % 3, (step % 3 + step // 3) % 3
    for site, extra in enumerate(EXTRAS):
        base = [(count, (count + site) % 3) for count in range(3)]
        for count, kind in [*base, extra]:
            yield 7 + (count + n_shift) % 3, (kind + type_shift) % 3, site


def make_scene(rng, n, waste_site, fault_type=None, *, held=False):
    counts = [2, 2, 2]
    for kind in rng.sample(range(3), n - 6):
        counts[kind] += 1
    fleet = [rng.sample(range(k*3, k*3+3), count) for k, count in enumerate(counts)]
    mask = sum(1 << tid for ids in fleet for tid in ids)
    orders = [''.join(p) for p in permutations('HMLW') if p[waste_site] == 'W']
    choices = [order for order in orders if (order in HELD_ORDERS) == held]
    # The historical held-out roles cover only some waste sites.
    order = rng.choice(choices or [o for o in orders if o not in HELD_ORDERS])
    grades = dict(zip("HMLW", [rng.uniform(.36,.38), rng.uniform(.29,.30), rng.uniform(.15,.17), 0.]))
    half = rng.uniform(.005,.01)
    alpha = sum(count*q for count,q in zip(counts, CAPACITIES)) / 1295.1
    return dict(initial_mask=mask, starting_count=n, assignment=order, waste_site=waste_site,
                grades=[grades[role] for role in order], half_ranges=[.005 if role=="W" else half for role in order],
                ore_target=16000.*alpha, waste_target=4000.*alpha, waste_stop=6000.*alpha,
                fault_truck_id=-1 if fault_type is None else rng.choice(fleet[fault_type]),
                fault_trigger_seconds=-1. if fault_type is None else rng.uniform(5400.,19800.))


def schedule(campaign, warmups, cycles, validation_slots, test_slots):
    if min(warmups, cycles, validation_slots, test_slots) < 1:
        raise ValueError("each stage requires a positive size")
    for arm in ("warmup", "static", "dynamic"):
        for update in range(1, (warmups if arm == "warmup" else 9*cycles) + 1):
            stage = "common" if arm == "warmup" else "branches"
            rng = Random(seed(campaign, "design", stage, update))
            if arm == "warmup":
                cells = [(n, None, s) for n in range(6, 10) for s in range(4) for _ in range(2)]
            else:
                cells = [(n, None, s) for n in range(6, 10) for s in range(4)] + list(fault_cells(update))
            rng.shuffle(cells)
            for n, kind, site in cells:
                scene = make_scene(rng, n, site, kind)
                if arm == "static":
                    scene.update(fault_truck_id=-1, fault_trigger_seconds=-1.)
                yield dict(phase=arm, arm=arm, role="train", update=update, split="train", pair_ordinal=-1,
                           condition="outage" if scene["fault_truck_id"] >= 0 else "healthy", **scene)
    for split, slots in (("validation", validation_slots), ("test",test_slots)):
        rng = Random(seed(campaign,"design",split))
        cells = [(n,k,s) for n in range(7,10) for k in range(3) for s in range(4)]
        for slot in range(slots):
            if slot % 36 == 0:
                rng.shuffle(cells)
            n,kind,site = cells[slot%36]
            scene = make_scene(rng,n,site,kind,held=bool(slot % 2))
            for arm in ("static","dynamic"):
                for condition in ("healthy","outage"):
                    fields = dict(scene)
                    if condition == "healthy":
                        fields.update(fault_truck_id=-1, fault_trigger_seconds=-1.)
                    yield dict(phase="evaluation", arm=arm, role="eval", update=9*cycles, split=split,
                               pair_ordinal=slot, condition=condition, **fields)
