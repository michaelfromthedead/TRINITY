"""Audio asset type with format and sample metadata."""
from __future__ import annotations

from enum import Enum, auto

from engine.resource.types.base_asset import BaseAsset

__all__ = ["AudioFormat", "AudioAsset"]


class AudioFormat(Enum):
    """Audio encoding formats."""
    PCM16 = auto()
    PCM24 = auto()
    FLOAT32 = auto()
    VORBIS = auto()
    OPUS = auto()


class AudioAsset(BaseAsset):
    """An audio clip asset."""

    __slots__ = (
        "_sample_rate", "_channels", "_bit_depth",
        "_duration_seconds", "_format", "_audio_data",
    )

    def __init__(
        self,
        asset_id: int,
        name: str,
        path: str,
        size_bytes: int,
        sample_rate: int,
        channels: int,
        bit_depth: int,
        duration_seconds: float,
        fmt: AudioFormat,
        version: int = 1,
    ) -> None:
        super().__init__(asset_id, name, path, size_bytes, version)
        self._sample_rate = sample_rate
        self._channels = channels
        self._bit_depth = bit_depth
        self._duration_seconds = duration_seconds
        self._format = fmt
        self._audio_data: bytes | None = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def bit_depth(self) -> int:
        return self._bit_depth

    @property
    def duration_seconds(self) -> float:
        return self._duration_seconds

    @property
    def format(self) -> AudioFormat:
        return self._format

    @property
    def audio_data(self) -> bytes | None:
        return self._audio_data

    # --- BaseAsset interface ---

    def load(self, data: bytes) -> None:
        self._audio_data = data

    def unload(self) -> None:
        self._audio_data = None

    def is_loaded(self) -> bool:
        return self._audio_data is not None

    @property
    def memory_footprint(self) -> int:
        if self._audio_data is not None:
            return len(self._audio_data)
        return 0
