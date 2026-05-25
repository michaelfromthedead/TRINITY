"""Animation asset type with channels and keyframes."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from engine.resource.types.base_asset import BaseAsset

__all__ = [
    "InterpolationMode", "Keyframe", "AnimChannel", "AnimationAsset",
]


class InterpolationMode(Enum):
    """Keyframe interpolation modes."""
    STEP = auto()
    LINEAR = auto()
    CUBIC = auto()


@dataclass(frozen=True, slots=True)
class Keyframe:
    """A single animation keyframe."""
    time: float
    value: tuple
    interpolation: InterpolationMode = InterpolationMode.LINEAR


@dataclass(slots=True)
class AnimChannel:
    """An animation channel targeting a specific property path."""
    target_path: str
    keyframes: list[Keyframe] = field(default_factory=list)


class AnimationAsset(BaseAsset):
    """An animation clip asset."""

    __slots__ = ("_duration_seconds", "_frame_count", "_channels", "_loaded")

    def __init__(
        self,
        asset_id: int,
        name: str,
        path: str,
        size_bytes: int,
        duration_seconds: float,
        frame_count: int,
        channels: list[AnimChannel] | None = None,
        version: int = 1,
    ) -> None:
        super().__init__(asset_id, name, path, size_bytes, version)
        self._duration_seconds = duration_seconds
        self._frame_count = frame_count
        self._channels: list[AnimChannel] = channels or []
        self._loaded = False

    @property
    def duration_seconds(self) -> float:
        return self._duration_seconds

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def channels(self) -> list[AnimChannel]:
        return list(self._channels)

    # --- BaseAsset interface ---

    def load(self, data: bytes) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded
