"""Shared nominal forecast with roundoff-safe bucket counting."""

from dataclasses import astuple
import math

from traditional_dispatch_v1.nominal import (
    MASS_EPS, MASS_TOL, SHIFT_END, WASTE_STOP_TONS, WINDOW_SECONDS,
    NominalModelError, NominalState as FrozenNominalState, _ARRIVE, _CFG, _DUMP_OFFSET, _FINISH, _Load,
)


class _ObservedLoad(_Load):
    __slots__ = ()

    def service(self, tons):
        ratio = tons / self.bucket_tons
        nearest = round(ratio)
        # The two masses were independently encoded in float32. A last
        # bucket filled to its decoded precision does not require a new pass.
        boundary = nearest * self.bucket_tons
        rounding = max(tons, boundary) * 2**-23
        if nearest > 0 and abs(tons - boundary) <= rounding:
            buckets = nearest
        else:
            buckets = math.ceil(ratio)
        return 60.0 * (_CFG['load_setup_minutes'] + self.bucket_minutes * buckets)


class NominalState(FrozenNominalState):
    """Keep frozen event semantics and replace the decoded shovel service model."""

    def __init__(self, snapshot):
        super().__init__(snapshot)
        self._loads = tuple(_ObservedLoad(*astuple(load)) for load in self._loads)
