"""Use opening rated payload to scale a nine-truck reference task once per shift."""


def scaled_task(base_task, opening_fleet):
    # Runtime scene capacities are integer tenths of a tonne, including parked slots.
    if len(opening_fleet) != 9 or sorted(row[0] for row in opening_fleet) != list(range(9)):
        raise ValueError("expected the nine distinct roster slots")
    if any(type(row[1]) is not bool or type(row[3]) is not int or row[3] <= 0 for row in opening_fleet):
        raise ValueError("invalid opening attendance or rated payload")
    if not 6 <= sum(row[1] for row in opening_fleet) <= 9:
        raise ValueError("opening fleet must contain six to nine active trucks")
    full = sum(row[3] for row in opening_fleet) / 10
    active = sum(row[3] for row in opening_fleet if row[1]) / 10
    if full != 1295.1:
        raise ValueError("nine-slot fleet differs from the dynamic u9 reference")
    alpha = active / full
    task = dict(base_task)
    for key in ("ore_target_tons", "waste_target_tons", "waste_dispatch_stop_tons"):
        task[key] = base_task[key] * alpha
    allowance = base_task["waste_upper_tons"] - base_task["waste_dispatch_stop_tons"]
    if allowance != max(row[3] for row in opening_fleet) / 10:
        raise ValueError("reference task must allow one full truck beyond the stop line")
    task["waste_upper_tons"] = task["waste_dispatch_stop_tons"] + allowance
    return task
