"""Audio chunk streaming with ring buffer."""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.resource.constants import AUDIO_CHUNK_SIZE

__all__ = ["AudioChunk", "AudioStreamManager", "AUDIO_CHUNK_SIZE"]


@dataclass(slots=True)
class AudioChunk:
    """A single audio data chunk."""

    chunk_index: int = 0
    sample_offset: int = 0
    sample_count: int = AUDIO_CHUNK_SIZE
    data: bytes | None = None


class AudioStreamManager:
    """Manages audio chunk streaming with a ring buffer per audio asset."""

    __slots__ = ("_buffers", "_pending")

    def __init__(self) -> None:
        # audio_id -> dict[chunk_index, AudioChunk]
        self._buffers: dict[str, dict[int, AudioChunk]] = {}
        self._pending: list[tuple[str, int, int]] = []  # (audio_id, start, count)

    def request_chunks(self, audio_id: str, start_chunk: int, count: int) -> None:
        """Request a range of audio chunks for streaming."""
        self._pending.append((audio_id, start_chunk, count))

    def get_buffered_range(self, audio_id: str) -> tuple[int, int]:
        """Return (min_chunk, max_chunk+1) of buffered chunks, or (0, 0)."""
        buf = self._buffers.get(audio_id)
        if not buf:
            return (0, 0)
        indices = sorted(buf.keys())
        return (indices[0], indices[-1] + 1)

    def update(self) -> None:
        """Process pending chunk requests."""
        for audio_id, start, count in self._pending:
            if audio_id not in self._buffers:
                self._buffers[audio_id] = {}
            buf = self._buffers[audio_id]
            for i in range(count):
                idx = start + i
                chunk = AudioChunk(
                    chunk_index=idx,
                    sample_offset=idx * AUDIO_CHUNK_SIZE,
                    sample_count=AUDIO_CHUNK_SIZE,
                    data=b"\x00" * AUDIO_CHUNK_SIZE,
                )
                buf[idx] = chunk
        self._pending.clear()
