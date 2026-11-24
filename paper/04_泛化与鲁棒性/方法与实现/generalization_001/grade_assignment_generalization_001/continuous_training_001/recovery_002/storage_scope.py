"""Use the canonical BG128 episode store inside this recovery process."""
from contextlib import contextmanager

from business_graph_summary_campaign_v1 import admission, collection, storage
from business_graph_summary_storage_v1.episode import read_episode_hdf5, write_episode_hdf5


@contextmanager
def bg128_storage():
    """Rebind the campaign's module aliases and their already-imported copies."""
    bindings = (
        (storage, "read_episode_hdf5", read_episode_hdf5),
        (storage, "write_episode_hdf5", write_episode_hdf5),
        (admission, "read_episode_hdf5", read_episode_hdf5),
        (admission, "write_episode_hdf5", write_episode_hdf5),
        (collection, "read_episode_hdf5", read_episode_hdf5),
    )
    originals = [(module, name, getattr(module, name)) for module, name, _ in bindings]
    try:
        for module, name, function in bindings:
            setattr(module, name, function)
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)
