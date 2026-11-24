from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .generated import dispatch_pb2
from .reward_config import RewardConfiguration, configuration_from_components
from .reward_contract import ADDITIVE_COMPONENTS
from .scenario_contract import ScenarioConfiguration, from_components as scenario_from_components
from .observation_contract import OBSERVATION_SCHEMA_ID, validate_observation
from .reward_contract import (
    REWARD_CONTRACT_ID,
    SHIFT_SECONDS,
    WINDOWS_PER_EPISODE,
    validate_reward_accounting,
    validate_window_summary,
    window_summary_to_dict,
)

SCHEMA_ID = OBSERVATION_SCHEMA_ID
ACTION_COUNT = 4


class ProtocolError(ValueError):
    """The AnyLogic callback violates the frozen dispatch protocol."""


@dataclass
class CallbackTicket:
    request: dispatch_pb2.DispatchRequest
    received_ns: int = field(default_factory=time.perf_counter_ns)
    response: Optional[dispatch_pb2.DispatchResponse] = None
    response_ready: threading.Event = field(default_factory=threading.Event)
    error: Optional[Exception] = None


def observation_array(request: dispatch_pb2.DispatchRequest) -> np.ndarray:
    if not request.HasField("obs"):
        raise ProtocolError("request.obs is required")
    try:
        return validate_observation(
            request.obs.schema_id, request.obs.values,
            terminal=request.kind == dispatch_pb2.EPISODE_END,
            truck_id=request.truck_id,
        )
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc



class DispatchCallbackBridge:
    """Thread-safe rendezvous between blocking gRPC handlers and Gymnasium."""

    def __init__(
        self,
        queue_capacity: int = 1,
        expected_episode_metadata: Optional[Sequence[tuple[int, int]]] = None,
        rollout_audit=None,
        window_grade_bounds: bool = True,
        reward_config: Optional[RewardConfiguration] = None,
        runtime_configuration: bool = True,
        expected_scenario: Optional[ScenarioConfiguration] = None,
        evidence_sink=None,
    ) -> None:
        self._incoming: queue.Queue[CallbackTicket] = queue.Queue(queue_capacity)
        self._lock = threading.Lock()
        self._tickets_by_request_id: dict[str, CallbackTicket] = {}
        self._request_bytes: dict[str, bytes] = {}
        self._last_sequence_by_episode: dict[str, int] = {}
        self._next_window_index_by_episode: dict[str, int] = {}
        self._metadata_by_episode: dict[str, tuple[int, int, str]] = {}
        self._seen_experiment_run_indices: set[int] = set()
        self._seen_experiment_seeds: set[int] = set()
        self._expected_episode_metadata = (
            tuple(expected_episode_metadata)
            if expected_episode_metadata is not None
            else None
        )
        self._started_episode_count = 0
        self._last_ticket_by_episode: dict[str, CallbackTicket] = {}
        self._closed = False
        self._fatal_error = None
        self.last_failure = None
        self.rollout_audit = rollout_audit
        # Only the explicitly enabled acceptance path installs this sink.
        self.evidence_sink = evidence_sink
        self.window_grade_bounds = window_grade_bounds
        self.expected_scenario = expected_scenario
        self.scenario_config = None
        self.runtime_configuration = runtime_configuration
        self.reward_config = reward_config
        if not runtime_configuration:
            raise ValueError("production-v2 requires explicit RPC configuration")
        self._configuration_ready = threading.Event()
        if self.reward_config is not None:
            self._configuration_ready.set()

    def submit_callback(
        self,
        request: dispatch_pb2.DispatchRequest,
        timeout: Optional[float] = None,
    ) -> dispatch_pb2.DispatchResponse:
        if self.rollout_audit is not None:
            self.rollout_audit.event("request_received", self._audit_info(request),
                                     action_required=bool(request.action_required))
        try:
            if self.evidence_sink is not None:
                self.evidence_sink.rpc_received(request)
            ticket = self._register(request)
        except Exception as exc:
            # Refused RPCs must reach the waiting sampler immediately, without fabricating a terminal.
            self.last_failure = dict(episode_id=request.episode_id,request_id=request.request_id,
                kind=int(request.kind),sim_time_seconds=request.sim_time_seconds,
                terminated=bool(request.terminated),rpc_truncated=bool(request.truncated))
            exc.failure_context = dict(self.last_failure)
            # RPC truncated does not identify whether the sampler or simulator caused it.
            exc.sampler_truncated = None if request.truncated else False
            self._fatal_error = exc
            self.close()
            if self.evidence_sink is not None:
                try:
                    self.evidence_sink.rpc_rejected(request, exc)
                except Exception as recording_error:
                    exc.recording_error = str(recording_error)
            if self.rollout_audit is not None:
                self.rollout_audit.event("request_registration_rejected", self._audit_info(request),
                                         reason=str(exc), exception_type=type(exc).__name__)
            raise
        if self.rollout_audit is not None:
            self.rollout_audit.event("request_registration_accepted", self._audit_info(request),
                                     reason="request passed registration and was accepted")
        if self.rollout_audit is not None and request.action_required:
            self.rollout_audit.event("request_action_status", self._audit_info(request),
                                     waiting_for_action=not ticket.response_ready.is_set())
        if not request.action_required and not ticket.response_ready.is_set():
            self._set_response(
                ticket,
                dispatch_pb2.DispatchResponse(
                    has_action=False,
                    episode_id=request.episode_id,
                    decision_sequence=request.decision_sequence,
                ),
            )
        if not ticket.response_ready.wait(timeout):
            raise TimeoutError(
                f"timed out waiting for action for request_id={request.request_id}"
            )
        if ticket.error is not None:
            raise ticket.error
        assert ticket.response is not None
        if self.evidence_sink is not None:
            try:
                self.evidence_sink.rpc_response(request, ticket.response)
            except Exception as exc:
                self._fatal_error = exc
                self.close()
                raise
        return ticket.response

    def next_callback(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._fatal_error is not None:
                raise self._fatal_error
            if self._closed:
                raise RuntimeError("bridge is closed")
            wait = 0.1 if deadline is None else min(0.1,max(0.0,deadline-time.monotonic()))
            try:
                ticket=self._incoming.get(timeout=wait)
                if self._fatal_error is not None:
                    raise self._fatal_error
                return ticket
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for AnyLogic callback")

    def respond_action(
        self,
        ticket: CallbackTicket,
        action: int,
        *,
        value: float = 0.0,
        logits: tuple[float, ...] = (),
    ) -> None:
        if not ticket.request.action_required:
            raise ProtocolError("cannot attach an action to this callback")
        if not 0 <= int(action) < ACTION_COUNT:
            raise ProtocolError(f"action must be in [0, 3], got {action}")
        if ticket.request.obs.values[19 + int(action)] != 1.0:
            raise ProtocolError(f"action {action} is masked out by ALP")
        latency_us = max(0, (time.perf_counter_ns() - ticket.received_ns) // 1000)
        if self.rollout_audit is not None:
            self.rollout_audit.event("action_response", self._audit_info(ticket.request),
                                     action=int(action), response_status="about_to_release_RPC_response",
                                     anylogic_execution="UNVERIFIED")
        self._set_response(
            ticket,
            dispatch_pb2.DispatchResponse(
                action=int(action),
                value=float(value),
                logits=logits,
                latency_us=latency_us,
                has_action=True,
                episode_id=ticket.request.episode_id,
                decision_sequence=ticket.request.decision_sequence,
            ),
        )

    def wait_for_configuration(self, timeout=None):
        if not self._configuration_ready.wait(timeout):
            raise TimeoutError('waiting for AnyLogic EPISODE_START runtime configuration')
        if self._closed:
            raise RuntimeError('bridge closed before runtime configuration was consumed')
        return self.reward_config

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for ticket in self._tickets_by_request_id.values():
                if not ticket.response_ready.is_set():
                    ticket.error = RuntimeError('training service closed before action response')
                    ticket.response_ready.set()
            self._configuration_ready.set()

    @staticmethod
    def _audit_info(request):
        kind = dispatch_pb2.DispatchKind.DESCRIPTOR.values_by_number.get(request.kind)
        return {"episode_id": request.episode_id, "request_id": request.request_id,
                "decision_sequence": request.decision_sequence,
                "sim_time_seconds": request.sim_time_seconds,
                "run_index": int(request.experiment_run_index),
                "experiment_seed": int(request.experiment_seed),
                "dispatch_kind": kind.name if kind is not None else f"UNKNOWN({request.kind})",
                "action_required": bool(request.action_required),
                "truck_id": int(request.truck_id),
                "reward_contract_id": request.reward_contract_id,
                "observation_schema_id": request.obs.schema_id if request.HasField("obs") else None}

    def _register(self, request: dispatch_pb2.DispatchRequest) -> CallbackTicket:
        try:
            incoming_scenario = scenario_from_components(request.reward_components)
        except (ValueError,TypeError) as exc:
            raise ProtocolError(str(exc)) from exc
        if self.reward_config is None and request.kind == dispatch_pb2.EPISODE_START and self.expected_scenario is None:
            self.expected_scenario = incoming_scenario
        if incoming_scenario != self.expected_scenario:
            raise ProtocolError('AnyLogic scenario differs from expected scenario; legacy entries reject declared waste-task rules. Use the dedicated scenario entry.')
        candidate = self.reward_config
        if self.runtime_configuration:
            try:
                incoming_config = configuration_from_components(request.reward_components, terminal=request.kind == dispatch_pb2.EPISODE_END)
            except ValueError as exc:
                raise ProtocolError(str(exc)) from exc
            if candidate is None:
                if request.kind != dispatch_pb2.EPISODE_START:
                    raise ProtocolError('first configuration must arrive with EPISODE_START')
                candidate = incoming_config
            elif incoming_config != candidate:
                raise ProtocolError('AnyLogic reward configuration changed during this run')
        self._validate_request(request, grade_bounds=self.window_grade_bounds, config=candidate)
        serialized = request.SerializeToString(deterministic=True)
        with self._lock:
            if self._closed:
                raise RuntimeError("bridge is closed")
            if self.runtime_configuration and self.reward_config is not None and candidate != self.reward_config:
                raise ProtocolError("concurrent runtime configuration changed")
            existing = self._tickets_by_request_id.get(request.request_id)
            if existing is not None:
                if self._request_bytes[request.request_id] != serialized:
                    raise ProtocolError(
                        "request_id was reused with different request contents"
                    )
                return existing
            last = self._last_ticket_by_episode.get(request.episode_id)
            if last is not None:
                if last.request.kind == dispatch_pb2.EPISODE_END:
                    raise ProtocolError("callback received after episode terminal")
                if last.response is None or not last.response.has_action:
                    raise ProtocolError("next transition arrived before previous action response")
                if request.previous_action != last.response.action:
                    raise ProtocolError("previous_action disagrees with last action response")
                if request.decision_sequence != last.request.decision_sequence + 1:
                    raise ProtocolError("decision_sequence must be contiguous")
                elapsed = request.sim_time_seconds - last.request.sim_time_seconds
                if elapsed < 0 or abs(elapsed - request.sim_time_delta_seconds) > 1e-6:
                    raise ProtocolError("simulation delta disagrees with previous callback time")
            previous = self._last_sequence_by_episode.get(request.episode_id)
            if request.kind == dispatch_pb2.EPISODE_START:
                if previous is not None:
                    raise ProtocolError("duplicate episode start with a new request_id")
                run_index = int(request.experiment_run_index)
                experiment_seed = int(request.experiment_seed)
                if self._expected_episode_metadata is not None:
                    if self._started_episode_count >= len(
                        self._expected_episode_metadata
                    ):
                        raise ProtocolError(
                            "received more EPISODE_START callbacks than the seed manifest"
                        )
                    expected = self._expected_episode_metadata[
                        self._started_episode_count
                    ]
                    if (run_index, experiment_seed) != expected:
                        raise ProtocolError(
                            "episode metadata does not match seed manifest at row "
                            f"{self._started_episode_count + 1}: expected "
                            f"run_index={expected[0]}, experiment_seed={expected[1]}, "
                            f"got run_index={run_index}, experiment_seed={experiment_seed}"
                        )
                if run_index in self._seen_experiment_run_indices:
                    raise ProtocolError("experiment_run_index was reused")
                if experiment_seed in self._seen_experiment_seeds:
                    raise ProtocolError("experiment_seed was reused")
                self._seen_experiment_run_indices.add(run_index)
                self._seen_experiment_seeds.add(experiment_seed)
                self._metadata_by_episode[request.episode_id] = (
                    run_index,
                    experiment_seed,
                    request.reward_contract_id,
                )
                self._next_window_index_by_episode[request.episode_id] = 0
                self._started_episode_count += 1
            elif previous is None:
                raise ProtocolError("DECISION/END received before EPISODE_START")
            elif request.decision_sequence <= previous:
                raise ProtocolError(
                    "decision_sequence must increase strictly within an episode"
                )
            else:
                expected_metadata = self._metadata_by_episode[request.episode_id]
                actual_metadata = (
                    int(request.experiment_run_index),
                    int(request.experiment_seed),
                    request.reward_contract_id,
                )
                if actual_metadata != expected_metadata:
                    raise ProtocolError("episode metadata changed before terminal callback")

            next_window_index = self._next_window_index_by_episode[request.episode_id]
            for summary in request.closed_windows:
                if summary.window_index != next_window_index:
                    raise ProtocolError(
                        "closed_windows must contain each T_win exactly once in order"
                    )
                next_window_index += 1
            if next_window_index > WINDOWS_PER_EPISODE:
                raise ProtocolError("episode reported more than four T_win summaries")
            if (
                request.kind == dispatch_pb2.EPISODE_END
                and next_window_index != WINDOWS_PER_EPISODE
            ):
                raise ProtocolError("EPISODE_END requires exactly four T_win summaries")
            self._next_window_index_by_episode[request.episode_id] = next_window_index
            ticket = CallbackTicket(request=request)
            self._tickets_by_request_id[request.request_id] = ticket
            self._request_bytes[request.request_id] = serialized
            self._last_sequence_by_episode[request.episode_id] = request.decision_sequence
            self._last_ticket_by_episode[request.episode_id] = ticket
            self.scenario_config = incoming_scenario
            if self.runtime_configuration and self.reward_config is None:
                self.reward_config = candidate
        self._incoming.put(ticket)
        self._configuration_ready.set()
        return ticket

    @staticmethod
    def _set_response(
        ticket: CallbackTicket,
        response: dispatch_pb2.DispatchResponse,
    ) -> None:
        if ticket.response_ready.is_set():
            return
        ticket.response = response
        ticket.response_ready.set()

    @staticmethod
    def _validate_request(request, *, grade_bounds=True, config=None):
        try:
            if config is None:
                raise ValueError("explicit RPC configuration required")
            terminal = request.kind == dispatch_pb2.EPISODE_END
            if not request.episode_id or not request.request_id or request.decision_sequence < 0:
                raise ValueError("episode/request identity required")
            if request.kind not in (dispatch_pb2.EPISODE_START, dispatch_pb2.DECISION, dispatch_pb2.EPISODE_END):
                raise ValueError("unknown lifecycle event")
            if request.truncated:
                raise ValueError("truncated/abnormal shift is not accepted as normal training data")
            if terminal:
                if not request.terminated or request.action_required or request.sim_time_seconds != SHIFT_SECONDS:
                    raise ValueError("normal END requires true cutoff and no action")
            elif request.terminated or not request.action_required or request.sim_time_seconds >= SHIFT_SECONDS:
                raise ValueError("invalid decision lifecycle")
            if request.kind == dispatch_pb2.EPISODE_START:
                if (request.decision_sequence != 0 or request.previous_action != -1 or
                    request.transition_reward != 0 or request.sim_time_delta_seconds != 0 or request.closed_windows):
                    raise ValueError("invalid START")
                if request.reward_components.get('wait_vehicle_seconds') != 0:
                    raise ValueError("nonzero waiting before first decision")
            if (not np.isfinite(request.sim_time_seconds) or request.sim_time_seconds < 0 or
                not np.isfinite(request.sim_time_delta_seconds) or request.sim_time_delta_seconds < 0):
                raise ValueError("invalid simulation time")
            if request.experiment_run_index <= 0 or request.experiment_seed <= 0:
                raise ValueError("explicit run index and seed required")
            if request.reward_contract_id != REWARD_CONTRACT_ID:
                raise ValueError("legacy reward contract rejected; use experimental ALP")
            from .observation_contract import FIELD_NAMES
            import hashlib
            order = hashlib.sha256(('\n'.join(FIELD_NAMES)+'\n').encode()).hexdigest()
            for i in range(8):
                if request.reward_components.get(f'config.obs.schema_word_{i}') != int(order[8*i:8*i+8], 16):
                    raise ValueError("runtime observation field order hash missing/mismatched")
            windows = [window_summary_to_dict(w) for w in request.closed_windows]
            if any(w['window_end_seconds'] > request.sim_time_seconds for w in windows):
                raise ValueError("future window result in request")
            validate_reward_accounting(request.transition_reward, dict(request.reward_components), windows,
                                       config=config, terminal=terminal)
            observation_array(request)
        except (ValueError, KeyError, TypeError) as exc:
            raise ProtocolError(str(exc)) from exc
