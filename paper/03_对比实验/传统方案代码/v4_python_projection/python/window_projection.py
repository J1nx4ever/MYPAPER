"""V4 / dispatch.physical.v3: FIFO-only (P0) time-window projection.

Only uses the *current* 1519-D public observation and fixed, audited V4
constants. No AnyLogic future-state access, no simulation-state mutation.

IMPORTANT: P0 projects the material *already in crusher FIFO*, not all
committed material. P1--P3 are NOT implemented here. Do not describe the
result as the final F1 committed-material projection in a research paper.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

RAW_DIM = 1519
SHIFT_SECONDS = 28800.0
WINDOW_SECONDS = 7200.0
WINDOW_COUNT = 4
PROJECTION_SCALE_TONS = 5000.0
FIFO_SLOTS = 16
FIFO_BASE = 1391
FIFO_STRIDE = 3
BUFFER_TONS_INDEX = 1300
CRUSHER_RATE_INDEX = 1304
SHIFT_ELAPSED_INDEX = 0
WINDOW_INDEX = 2
TERMINAL_INDEX = 17

# Critical offsets are cross-checked against the uploaded observation table.
EXPECTED_FIELDS = {
    0: "global.shiftElapsed",
    2: "global.windowIndex",
    17: "global.terminal",
    1300: "dump.0.bufferTons",
    1304: "dump.0.crusherRate",
    1391: "fifo.0.valid",
    1392: "fifo.0.tons",
    1393: "fifo.0.grade",
    1436: "fifo.15.valid",
    1437: "fifo.15.tons",
    1438: "fifo.15.grade",
}


def assert_schema_fields(field_names: Iterable[str]) -> None:
    names = tuple(field_names)
    if len(names) != RAW_DIM:
        raise ValueError(f"expected {RAW_DIM} schema names, got {len(names)}")
    for i, wanted in EXPECTED_FIELDS.items():
        if names[i] != wanted:
            raise ValueError(f"index {i}: expected {wanted!r}, got {names[i]!r}")


@dataclass(frozen=True)
class Batch:
    remaining_tons: float
    grade_fraction: float


@dataclass(frozen=True)
class Projection:
    current_window: int
    produced_tons_by_window: tuple[float, float, float, float]
    produced_metal_by_window: tuple[float, float, float, float]
    remaining_after_shift_tons: float
    initial_fifo_tons: float
    # Six fields in exact V1.0 order, normalized before returning.
    features: np.ndarray


def _checked_raw(raw: np.ndarray) -> np.ndarray:
    a = np.asarray(raw, dtype=np.float64)
    if a.shape != (RAW_DIM,):
        raise ValueError(f"expected shape ({RAW_DIM},), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite raw observation")
    return a


def decode_fifo(raw: np.ndarray, *, check_buffer: bool = True) -> tuple[Batch, ...]:
    """Read contiguous V4 FIFO slots; never double-count reserved/dumping ore.

    dump.0.bufferTons represents crusherStoredTons and includes the FIFO.
    The consistency check is at float32 transport precision, not Java double.
    """
    a = _checked_raw(raw)
    result = []
    seen_empty = False
    for k in range(FIFO_SLOTS):
        base = FIFO_BASE + k * FIFO_STRIDE
        valid, tons_norm, grade = (float(x) for x in a[base:base+3])
        if valid not in (0.0, 1.0):
            raise ValueError(f"fifo.{k}.valid must be binary, got {valid}")
        if valid == 0.0:
            seen_empty = True
            if abs(tons_norm) > 1e-8 or abs(grade) > 1e-8:
                raise ValueError(f"fifo.{k} invalid but nonzero data")
            continue
        if seen_empty:
            raise ValueError(f"fifo.{k} valid after empty slot: invalid FIFO order")
        tons = tons_norm * 500.0
        if tons <= 0.0 or not 0.0 <= grade <= 1.0:
            raise ValueError(f"fifo.{k} invalid remaining tons or grade")
        result.append(Batch(remaining_tons=tons, grade_fraction=grade))

    if check_buffer:
        stored = float(a[BUFFER_TONS_INDEX]) * 500.0
        in_fifo = sum(batch.remaining_tons for batch in result)
        if not math.isclose(stored, in_fifo, abs_tol=1e-3, rel_tol=2e-6):
            raise ValueError(
                f"crusher stored mass != FIFO mass: stored={stored}, "
                f"sum_fifo={in_fifo}; check observation/version"
            )
    return tuple(result)


def project_fifo_only(raw: np.ndarray, *, check_buffer: bool = True) -> Projection:
    """Exact fluid FIFO integration for existing batches under V4 fixed rate.

    Starts *after* the current observation time. No new batches are admitted.
    Cross-window portions are split by actual processed tonnage; neither
    truck arrival time nor whole-truck window assignment is used.
    """
    a = _checked_raw(raw)
    batches = decode_fifo(a, check_buffer=check_buffer)
    now = float(a[SHIFT_ELAPSED_INDEX]) * SHIFT_SECONDS
    raw_index = float(a[WINDOW_INDEX]) * WINDOW_COUNT
    current = int(round(raw_index))
    if abs(raw_index - current) > 1e-5:
        raise ValueError(f"unrecognized normalized window index: {raw_index}")
    if not 0.0 <= now <= SHIFT_SECONDS + 0.01:
        raise ValueError(f"invalid elapsed time {now}")
    if not 0 <= current <= WINDOW_COUNT:
        raise ValueError(f"invalid window index {current}")
    terminal = bool(a[TERMINAL_INDEX])
    if terminal != (current == WINDOW_COUNT):
        # V4 end-of-shift observation should have all windows closed.
        # Fail instead of silently attributing ore to an invalid window.
        raise ValueError("terminal/windowIndex inconsistency")
    rate = float(a[CRUSHER_RATE_INDEX]) * 2500.0 / 3600.0
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"invalid crusher rate t/s: {rate}")

    left = [b.remaining_tons for b in batches]
    grade = [b.grade_fraction for b in batches]
    cursor = 0
    q = [0.0] * WINDOW_COUNT
    m = [0.0] * WINDOW_COUNT
    if not terminal:
        # V4 uses windowIndex and current time; never credit prior output here.
        if now < current * WINDOW_SECONDS - 0.01:
            raise ValueError("observation time precedes windowIndex")
        for w in range(current, WINDOW_COUNT):
            start_sec = max(now, w * WINDOW_SECONDS)
            end_sec = (w+1) * WINDOW_SECONDS
            available = max(0.0, (end_sec - start_sec) * rate)
            while available > 1e-9 and cursor < len(left):
                used = min(available, left[cursor])
                q[w] += used
                m[w] += used * grade[cursor]
                left[cursor] -= used
                available -= used
                if left[cursor] <= 1e-9:
                    cursor += 1

    pending = sum(max(0.0, x) for x in left)
    initial = sum(b.remaining_tons for b in batches)
    if not math.isclose(sum(q) + pending, initial, abs_tol=2e-6, rel_tol=1e-8):
        raise AssertionError("FIFO mass conservation violated")
    if not math.isclose(
        sum(m) + sum(max(0.0,left[k])*grade[k] for k in range(len(left))),
        sum(b.remaining_tons*b.grade_fraction for b in batches),
        abs_tol=2e-6, rel_tol=1e-8,
    ):
        raise AssertionError("FIFO metal conservation violated")

    feat = np.zeros(6, dtype=np.float32)
    if not terminal:
        feat[0] = q[current] / PROJECTION_SCALE_TONS
        feat[1] = m[current] / PROJECTION_SCALE_TONS
        feat[2] = 1.0
        if current + 1 < WINDOW_COUNT:
            feat[3] = q[current+1] / PROJECTION_SCALE_TONS
            feat[4] = m[current+1] / PROJECTION_SCALE_TONS
            feat[5] = 1.0
    return Projection(
        current_window=current,
        produced_tons_by_window=tuple(q),
        produced_metal_by_window=tuple(m),
        remaining_after_shift_tons=pending,
        initial_fifo_tons=initial,
        features=feat,
    )
