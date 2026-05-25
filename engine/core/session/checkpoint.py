"""Checkpoint system: rolling checkpoints with auto-prune."""

from __future__ import annotations

import collections
from typing import Optional
import uuid

from engine.core.constants import CHECKPOINT_ID_LENGTH, MAX_CHECKPOINTS
from engine.core.session.session import SessionData


class CheckpointManager:
    """Maintains a rolling set of checkpoints, auto-pruning oldest."""

    def __init__(self, max_checkpoints: int = MAX_CHECKPOINTS) -> None:
        self.max_checkpoints = max_checkpoints
        self._checkpoints: dict[str, SessionData] = {}
        self._order: collections.deque[str] = collections.deque()

    def create_checkpoint(self, session_data: SessionData) -> str:
        """Store a checkpoint, pruning oldest if over limit."""
        checkpoint_id = uuid.uuid4().hex[:CHECKPOINT_ID_LENGTH]
        self._checkpoints[checkpoint_id] = session_data
        self._order.append(checkpoint_id)
        self._prune()
        return checkpoint_id

    def restore_checkpoint(self, checkpoint_id: str) -> Optional[SessionData]:
        """Retrieve a checkpoint by id, or None if not found."""
        return self._checkpoints.get(checkpoint_id)

    def list_checkpoints(self) -> list[str]:
        """Return checkpoint ids in creation order."""
        return list(self._order)

    def _prune(self) -> None:
        while len(self._order) > self.max_checkpoints:
            oldest = self._order.popleft()
            self._checkpoints.pop(oldest, None)
