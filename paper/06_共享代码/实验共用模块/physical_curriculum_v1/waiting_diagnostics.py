"""Explicit optional diagnostic schema; basic RPC fleet-wait stays mandatory.

Absence of a decomposition is recorded as unavailable, never as zero waiting.
If any breakdown field is present, the full declared schema must validate.
"""
from collections.abc import Mapping
import math

from u8_wait_efficiency_v1 import waiting_diagnostics as original

WAIT_RPC_FIELDS = dict(original.WAIT_RPC_FIELDS)
WAIT_FIELDS = original.WAIT_FIELDS
PREFIX = original.PREFIX
DURATION_NAMES = original.DURATION_NAMES
WAIT_DIAGNOSTICS_CONTRACT = {'mode':'optional_if_present',
                             'schema':dict(original.WAIT_DIAGNOSTICS_CONTRACT)}
STATUS = 'waiting_breakdown_status'
AVAILABLE = 'waiting_breakdown_available'


def _number(value,name):
    if isinstance(value,bool) or type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('finite explicit waiting diagnostic required: '+name)
    return float(value)


def require_wait_rpc_configuration(components):
    if not isinstance(components,Mapping):
        raise ValueError('explicit RPC waiting configuration required')
    present = set(WAIT_RPC_FIELDS)&set(components)
    if not present:
        return 'not_declared_by_rpc_schema'
    if present != set(WAIT_RPC_FIELDS):
        raise ValueError('partial waiting diagnostics RPC configuration')
    enabled = _number(components['config.wait_breakdown_enabled'],'config.wait_breakdown_enabled')
    version = _number(components['config.wait_breakdown_version'],'config.wait_breakdown_version')
    if enabled == 1. and version == 1.:
        return 'declared_enabled_v1'
    if enabled == 0. and version in (0.,1.):
        return 'declared_disabled'
    raise ValueError('unsupported waiting diagnostics RPC configuration')


def validate_wait_breakdown(components,*,total_wait_seconds):
    if not isinstance(components,Mapping):
        raise ValueError('explicit terminal waiting diagnostics availability required')
    if _number(total_wait_seconds,'total_wait_seconds') < 0:
        raise ValueError('negative fleet waiting')
    present = set(WAIT_FIELDS)&set(components)
    if present:
        if present != set(WAIT_FIELDS):
            raise ValueError('partial terminal waiting diagnostics schema')
        if components.get(AVAILABLE,True) is not True or components.get(STATUS,'available_v1') != 'available_v1':
            raise ValueError('diagnostic fields contradict explicit availability')
        return dict(original.validate_wait_breakdown(components,total_wait_seconds=total_wait_seconds),
                    **{AVAILABLE:True,STATUS:'available_v1'})
    if (components.get(AVAILABLE) is not False or components.get(STATUS) not in
            ('not_declared_by_rpc_schema','declared_disabled')):
        raise ValueError('explicit unavailable waiting diagnostics status required')
    return {AVAILABLE:False,STATUS:components[STATUS]}


def validate_rpc_wait_breakdown(infos):
    infos = tuple(infos)
    if not infos:
        raise ValueError('complete RPC wait ledger required')
    modes = [require_wait_rpc_configuration(info['reward_components']) for info in infos]
    if len(set(modes)) != 1:
        raise ValueError('RPC waiting diagnostics configuration changed during shift')
    terminal = infos[-1]['reward_components']
    total = _number(terminal.get('wait_vehicle_seconds'),'wait_vehicle_seconds')
    present = set(WAIT_FIELDS)&set(terminal)
    if any(set(WAIT_FIELDS)&set(info['reward_components']) for info in infos[:-1]):
        raise ValueError('waiting diagnostic decomposition is terminal only')
    if modes[0] == 'declared_enabled_v1':
        if present != set(WAIT_FIELDS):
            raise ValueError('enabled RPC schema requires all terminal waiting diagnostics')
        return validate_wait_breakdown(terminal,total_wait_seconds=total)
    if present or any(set(WAIT_FIELDS)&set(info['reward_components']) for info in infos[:-1]):
        raise ValueError('waiting diagnostic values lack enabled RPC schema')
    return {AVAILABLE:False,STATUS:modes[0]}


__all__ = ['WAIT_RPC_FIELDS','WAIT_FIELDS','PREFIX','DURATION_NAMES','WAIT_DIAGNOSTICS_CONTRACT',
           'require_wait_rpc_configuration','validate_wait_breakdown','validate_rpc_wait_breakdown']
