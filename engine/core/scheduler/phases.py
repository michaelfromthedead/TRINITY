"""Phase definitions for the system scheduler."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List


class Phase(enum.IntEnum):
    """Frame execution phases in order."""
    PRE_UPDATE = 0
    UPDATE = 1
    POST_UPDATE = 2
    PRE_RENDER = 3
    RENDER = 4
    POST_RENDER = 5


DEFAULT_PHASE_ORDER: List[Phase] = list(Phase)


@dataclass
class PhaseGroup:
    """Ordered collection of system IDs within a single phase."""
    phase: Phase
    system_ids: List[int] = field(default_factory=list)

    def add(self, system_id: int) -> None:
        if system_id not in self.system_ids:
            self.system_ids.append(system_id)

    def remove(self, system_id: int) -> None:
        self.system_ids.remove(system_id)
