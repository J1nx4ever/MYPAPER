"""Strict terminal-only waiting decomposition from the independent Java ledger.

The original fleet-wait reward ledger remains authoritative. These diagnostics
are required evidence; missing values are never inferred or replaced by zero.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Mapping

WAIT_RPC_FIELDS = {
    'config.wait_breakdown_enabled': 1.0,
    'config.wait_breakdown_version': 1.0,
}
PREFIX = 'diag.wait.'
DURATION_NAMES = ('load_seconds', 'counted_load_seconds', 'dump_total_seconds',
                  'dump_resource_queue_seconds', 'dump_capacity_admission_seconds')
RESIDUAL_NAMES = ('load_reconciliation_residual_seconds',
                  'dump_reconciliation_residual_seconds',
                  'total_reconciliation_residual_seconds',
                  'unclassified_flagged_seconds')
WAIT_FIELDS = tuple(PREFIX + name for name in
                    ('breakdown_version', *DURATION_NAMES, *RESIDUAL_NAMES,
                     'stage_errors', 'load_fallback_hits'))
WAIT_DIAGNOSTICS_CONTRACT = dict(
    schema='u8-wait-breakdown-terminal-v1', version=1, terminal_only=True,
    required_fields=list(WAIT_FIELDS), rpc_fields=dict(WAIT_RPC_FIELDS),
    residual_absolute_tolerance_seconds=1e-6, residual_relative_tolerance=1e-9,
    stage_errors=0, load_fallback_hits=0)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError('finite explicit waiting diagnostic required: '+name)
    return float(value)


def require_wait_rpc_configuration(components: Mapping) -> None:
    if not isinstance(components, Mapping):
        raise ValueError('explicit RPC waiting breakdown configuration required')
    for name, expected in WAIT_RPC_FIELDS.items():
        if name not in components or _number(components[name], name) != expected:
            raise ValueError('RPC waiting breakdown configuration differs: '+name)


def validate_wait_breakdown(components: Mapping, *, total_wait_seconds: float) -> dict:
    """Validate 12 fields and four independently recomputed conservation laws."""
    if not isinstance(components, Mapping) or not set(WAIT_FIELDS) <= set(components):
        raise ValueError('all 12 explicit terminal waiting breakdown fields are required')
    values = {name: _number(components[name], name) for name in WAIT_FIELDS}
    total = _number(total_wait_seconds, 'total_wait_seconds')
    if total < 0 or values[PREFIX+'breakdown_version'] != 1.:
        raise ValueError('nonnegative fleet waiting and breakdown version 1 required')
    if any(values[PREFIX+name] < 0 for name in DURATION_NAMES):
        raise ValueError('waiting breakdown durations must be nonnegative')
    if any(values[PREFIX+name] != 0 for name in ('stage_errors','load_fallback_hits')):
        raise ValueError('waiting stage errors and fallback hits must both be zero')
    tolerance = max(1e-6, 1e-9 * max(1., total))
    load, counted, dump, resource, capacity = (values[PREFIX+name] for name in DURATION_NAMES)
    recomputed = (load-counted, dump-resource-capacity, total-load-dump,
                  total-counted-resource-capacity)
    for name, actual in zip(RESIDUAL_NAMES, recomputed):
        reported = values[PREFIX+name]
        if abs(actual) > tolerance or abs(reported) > tolerance or abs(reported-actual) > tolerance:
            raise ValueError('waiting breakdown conservation differs: '+name)
    return values


def validate_rpc_wait_breakdown(infos) -> dict:
    """Check actual RPC config at every callback and diagnostic evidence at END."""
    infos = tuple(infos)
    if not infos:
        raise ValueError('complete RPC waiting diagnostics evidence required')
    for info in infos:
        require_wait_rpc_configuration(info['reward_components'])
    terminal = infos[-1]['reward_components']
    if 'wait_vehicle_seconds' not in terminal:
        raise ValueError('explicit terminal fleet wait total required')
    return validate_wait_breakdown(terminal, total_wait_seconds=terminal['wait_vehicle_seconds'])
