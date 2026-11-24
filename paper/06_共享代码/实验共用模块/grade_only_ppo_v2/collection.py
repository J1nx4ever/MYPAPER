"""Reuse the frozen material/RPC data path for the MC-advantage smoke.

Advantage estimation changes only after complete-shift admission.  Raw RPC,
1519D observations, fixed-Actor sampling, ACK-before-journal-seal ordering,
source-byte rebinding and compact HDF5 are exactly the reviewed v1 data path.
The receipt schema remains v1 because the stored physical-data contract is
unchanged.  Independent v2 campaign/phase gates own scope and authorization.
"""
from grade_only_ppo_v1.collection import (
    MATERIAL_RPC_FIELDS,
    RECEIPT_SCHEMA,
    MaterialPhaseCollector,
    OnlineActor,
    load_admitted_episode,
    material_journal_path,
    wait_for_material_journal,
)

__all__ = [
    'MATERIAL_RPC_FIELDS', 'RECEIPT_SCHEMA', 'MaterialPhaseCollector',
    'OnlineActor', 'load_admitted_episode', 'material_journal_path',
    'wait_for_material_journal',
]
