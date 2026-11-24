"""Give each physical shift a fresh, strictly validated RPC configuration."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from baseline1_v1.collection import PhaseCollector
from business_graph_summary_campaign_v1.collection import MaterialPhaseCollector
from business_graph_summary_campaign_v1.plan_io import reference
from business_graph_summary_v1.contract import require
from python.constrained_batch import load_dataset


class PerJobMaterialCollector(MaterialPhaseCollector):
    """Keep phase-level evidence and close the RPC listener after each shift."""

    def __enter__(self):
        return self

    def collect(self, job, behavior, *, actor_path):
        require(job in self.cfg["jobs"], "job absent from immutable phase")
        phase_transport = self._collector
        job_transport = PhaseCollector(
            dict(self.cfg, jobs=[job]),
            self.config_path,
            publish_ready=phase_transport.publish_ready,
            startup=phase_transport.startup,
        )
        self._collector = job_transport
        try:
            with job_transport:
                return super().collect(job, behavior, actor_path=actor_path)
        finally:
            self._collector = phase_transport

    def __exit__(self, *error):
        return None


@contextmanager
def per_job_transport():
    """Install the recovery transport in the serial mixed campaign controller."""
    import mixed_runtime

    original = mixed_runtime.MaterialPhaseCollector
    mixed_runtime.MaterialPhaseCollector = PerJobMaterialCollector
    try:
        yield
    finally:
        mixed_runtime.MaterialPhaseCollector = original


def verify_replacement_learner(model, episode, current, job):
    """Apply the original exact learner proof to the separately stored replacement."""
    from mixed_runtime import verify_material_observation, verify_online

    identity = episode["identity"]
    artifact = model._actor_artifact
    require(
        np.isfinite(current.detach().numpy()).all()
        and isinstance(artifact, dict)
        and artifact["actor_digest"] == model.policy_version == identity["policy_version"]
        and artifact["sha256"] == identity["policy_sha256"]
        and identity["run_index"] == job["run_index"]
        and identity["actor_sampling_seed"] == job["actor_sampling_seed"],
        "replacement learner trajectory differs from the sealed Actor or planned run",
    )
    require(
        model._validated_actor_snapshot(artifact["path"])["sha256"] == artifact["sha256"],
        "bound Actor seal changed",
    )
    raw = Path(job["python_output"]) / "episodes" / f"run_{job['run_index']}"
    require(
        reference(raw / "manifest.json")["sha256"] == identity["raw_manifest_sha256"],
        "replacement raw trajectory manifest changed",
    )
    reader, _, _, _, _ = load_dataset(raw, require_real=True)
    try:
        rows = list(reader)
    finally:
        if hasattr(reader, "close"):
            reader.close()
    require(bool(rows), "replacement learner trajectory is empty")
    states = np.stack([rows[0]["observation"]] + [row["next_observation"] for row in rows])
    actions = np.array([row["action"] for row in rows], dtype=np.int64)
    logs = np.array([row["old_log_prob"] for row in rows], dtype=np.float64)
    require(
        states.tobytes() == episode["states"].tobytes()
        and np.array_equal(actions, episode["actions"])
        and logs.tobytes() == np.asarray(episode["old_log_probs"], dtype=np.float64).tobytes(),
        "replacement learner inputs differ from sealed observations, actions or probabilities",
    )
    verify_material_observation(states, job)
    return verify_online(model.actor, states[:-1], actions, logs, rows=rows, job=job)
