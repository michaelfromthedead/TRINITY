"""Session manager: save/load entire engine state."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from engine.core.constants import SESSION_VERSION


@dataclass
class SessionData:
    """Serializable snapshot of engine state."""

    version: int = SESSION_VERSION
    timestamp: float = 0.0
    frame_count: int = 0
    total_time: float = 0.0
    world_snapshot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionData:
        return cls(
            version=data.get("version", SESSION_VERSION),
            timestamp=data.get("timestamp", 0.0),
            frame_count=data.get("frame_count", 0),
            total_time=data.get("total_time", 0.0),
            world_snapshot=data.get("world_snapshot", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Result:
    """Simple result wrapper."""

    success: bool
    error: Optional[str] = None
    data: Any = None


class Session:
    """Manages saving and loading of engine state."""

    def __init__(
        self,
        frame_count: int = 0,
        total_time: float = 0.0,
        world_snapshot: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.frame_count = frame_count
        self.total_time = total_time
        self.world_snapshot: dict[str, Any] = world_snapshot or {}
        self.metadata: dict[str, Any] = metadata or {}

    def to_session_data(self) -> SessionData:
        return SessionData(
            version=SESSION_VERSION,
            timestamp=time.time(),
            frame_count=self.frame_count,
            total_time=self.total_time,
            world_snapshot=dict(self.world_snapshot),
            metadata=dict(self.metadata),
        )

    def save(self, filepath: str) -> Result:
        """Serialize session state to a JSON file."""
        try:
            data = self.to_session_data()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data.to_dict(), f, indent=2)
            return Result(success=True, data=data)
        except Exception as exc:
            return Result(success=False, error=str(exc))

    def load(self, filepath: str) -> Result:
        """Deserialize and restore state from a JSON file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = json.load(f)
            data = SessionData.from_dict(raw)
            self.frame_count = data.frame_count
            self.total_time = data.total_time
            self.world_snapshot = data.world_snapshot
            self.metadata = data.metadata
            return Result(success=True, data=data)
        except Exception as exc:
            return Result(success=False, error=str(exc))
