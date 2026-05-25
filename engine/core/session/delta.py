"""Delta encoding: compute diffs between SessionData snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.core.session.session import SessionData


@dataclass
class DeltaData:
    """Represents the diff between two snapshots."""

    added: dict[str, Any] = field(default_factory=dict)
    removed: dict[str, Any] = field(default_factory=dict)
    modified: dict[str, Any] = field(default_factory=dict)


class DeltaEncoder:
    """Computes and applies deltas between SessionData snapshots."""

    @staticmethod
    def encode_delta(old: SessionData, new: SessionData) -> DeltaData:
        """Compute the diff between two session snapshots (world_snapshot)."""
        old_snap = old.world_snapshot
        new_snap = new.world_snapshot
        old_keys = set(old_snap.keys())
        new_keys = set(new_snap.keys())

        added = {k: new_snap[k] for k in new_keys - old_keys}
        removed = {k: old_snap[k] for k in old_keys - new_keys}
        modified = {
            k: new_snap[k]
            for k in old_keys & new_keys
            if old_snap[k] != new_snap[k]
        }

        return DeltaData(added=added, removed=removed, modified=modified)

    @staticmethod
    def apply_delta(base: SessionData, delta: DeltaData) -> SessionData:
        """Apply a delta to a base snapshot, returning a new SessionData."""
        snapshot = dict(base.world_snapshot)

        for k in delta.removed:
            snapshot.pop(k, None)

        snapshot.update(delta.added)
        snapshot.update(delta.modified)

        return SessionData(
            version=base.version,
            timestamp=base.timestamp,
            frame_count=base.frame_count,
            total_time=base.total_time,
            world_snapshot=snapshot,
            metadata=dict(base.metadata),
        )
