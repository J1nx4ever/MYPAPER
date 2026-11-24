"""Narrow source-pin equivalence for the reviewed HDF5 storage migration.

This does not grant execution, collection or data admission. Callers must still
verify their certificates, evidence, schema, RPC configuration and data bytes.
Both revisions are fixed so later edits cannot inherit this storage review.
"""
import re


_SHA256 = re.compile(r'^[0-9a-f]{64}$')
LEGACY_BATCH_SHA256 = '01fc6e33effe81f43c20a8f2f1e8cce80694f6a6e94b2f2920401609f7430ec2'
# Reviewed storage migration only. A future revision requires a fresh review.
REVIEWED_BATCH_SHA256 = '7eb3b3d459f85bbe84a62dfe247523ab69edc716d6f551112026bcc95d8ac48b'


def source_pins_match(recorded, current, *, storage_migration=False):
    """Compare exact pin inventories, optionally accepting one reviewed pair.

    A caller must explicitly opt into the storage review. Every unchanged pin
    still matches exactly, and the only permitted difference is the fixed old
    -> fixed new ``python/constrained_batch.py`` pair. Unknown, missing, added,
    malformed and future revisions fail closed.
    """
    if type(recorded) is not dict or type(current) is not dict or not recorded:
        return False
    if recorded.keys() != current.keys():
        return False
    for key, previous in recorded.items():
        present = current[key]
        if (type(key) is not str or type(previous) is not str or type(present) is not str
                or not _SHA256.fullmatch(previous) or not _SHA256.fullmatch(present)):
            return False
        if previous == present:
            continue
        if not (storage_migration is True and key == 'python/constrained_batch.py'
                and previous == LEGACY_BATCH_SHA256 and REVIEWED_BATCH_SHA256 is not None
                and present == REVIEWED_BATCH_SHA256):
            return False
    return True
