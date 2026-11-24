"""Business acceptance from actual END counters, never inferred from dispatches."""
import math


def verify_terminal(job, components):
    def field(name):
        value = float(components["fleet." + name])
        if not math.isfinite(value):
            raise ValueError("nonfinite fleet." + name)
        return value

    triggered, exited = field("trigger_seconds"), field("exit_seconds")
    initial = job["initial_mask"]
    fault = job["fault_truck_id"]
    if fault >= 0:
        if not (initial & (1 << fault)) or triggered != job["fault_trigger_seconds"]:
            raise ValueError("fault did not trigger on the planned clock and roster")
        if exited != -1 and not triggered <= exited <= 28800:
            raise ValueError("actual exit outside the trigger-to-END interval")
        eligible = initial & ~(1 << fault)
    elif triggered == exited == -1:
        eligible = initial
    else:
        raise ValueError("fault counters present in healthy shift")
    present = eligible if exited >= 0 else initial
    if field("dispatchable_mask") != eligible or field("in_system_mask") != present:
        raise ValueError("terminal fleet masks disagree with actual retirement")

    window_ore = field("window_ore_tons")
    window_metal = field("window_metal_tons")
    masses = {name: field(name) for name in (
        "incoming_ore_tons", "incoming_metal_tons", "processed_ore_tons",
        "processed_metal_tons", "fifo_ore_tons", "fifo_metal_tons")}
    tolerance = float(components["config.mass_balance_epsilon"])
    if not (math.isfinite(tolerance) and 0 < tolerance <= 1e-5):
        raise ValueError("invalid RPC mass balance tolerance")
    for origin in ("incoming", "processed", "fifo"):
        ore, metal = (masses[f"{origin}_{substance}_tons"] for substance in ("ore", "metal"))
        if not 0 <= metal <= ore + tolerance or ore < 0:
            raise ValueError("nonphysical material at " + origin)
    residuals = {}
    for substance in ("ore", "metal"):
        residual = math.fsum((masses[f"incoming_{substance}_tons"],
                              -masses[f"processed_{substance}_tons"],
                              -masses[f"fifo_{substance}_tons"]))
        if abs(residual) > tolerance:
            raise ValueError(f"crusher FIFO {substance} conservation failed: {residual:g} t")
        residuals[substance] = residual
    if (not 0 <= window_metal <= window_ore + tolerance or window_ore < 0
            or window_ore > masses["processed_ore_tons"] + tolerance
            or window_metal > masses["processed_metal_tons"] + tolerance
            or (exited == -1 and (window_metal or window_ore))):
        raise ValueError("invalid exit-anchored window counters")
    eligible_window = 0 <= exited <= 21600
    grade = window_metal / window_ore if eligible_window and window_ore > 0 else None
    return dict(trigger_seconds=triggered, exit_seconds=exited, window_valid=eligible_window,
                exit_delay_seconds=exited-triggered if exited >= 0 else None,
                window_ore_tons=window_ore, window_metal_tons=window_metal, window_grade=grade,
                window_compliant=(grade is not None and .26 <= grade <= .28) if eligible_window else None,
                window_empty=(window_ore == 0) if eligible_window else None,
                mass_balance_residual_tons=residuals, **masses)
