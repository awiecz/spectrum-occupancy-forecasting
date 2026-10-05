"""Neutral contracts implemented by retained feedback banks and count filtering."""

from typing import Protocol


class ModelAdapter(Protocol):
    """A factory initializes a bank; replay supplies only delivered feedback snapshots."""

    preprocessing: dict
    failures: list

    def update(self, time, target_end, y, base, experts, vectors):
        """Assimilate completed-target feedback released by `time`."""
        ...

    def forecasts(self, time, base, selected, experts, vectors, age, coverage, active=None):
        """Issue immutable records after processing every already-delivered event."""
        ...

    def checkpoint(self) -> bytes:
        """Serialize local state; never load untrusted checkpoint files."""
        ...


class DeliveredMeasurementAdapter(ModelAdapter, Protocol):
    def advance(self, time):
        """Assimilate released measurement bins, independently of scoring labels."""
        ...
