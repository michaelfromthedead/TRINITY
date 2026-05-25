"""World chunk streaming based on camera position."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from engine.resource.constants import CHUNK_SIZE, DEFAULT_LOAD_RADIUS

__all__ = [
    "ChunkState",
    "WorldChunk",
    "WorldStreamManager",
    "CHUNK_SIZE",
    "DEFAULT_LOAD_RADIUS",
]


class ChunkState(enum.Enum):
    """Lifecycle states of a world chunk."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"


@dataclass(slots=True)
class WorldChunk:
    """A single world grid chunk."""

    chunk_x: int = 0
    chunk_y: int = 0
    state: ChunkState = ChunkState.UNLOADED


class WorldStreamManager:
    """Manages world chunk loading/unloading based on camera position."""

    __slots__ = ("_chunks", "_camera_x", "_camera_y", "_load_radius")

    def __init__(self) -> None:
        self._chunks: dict[tuple[int, int], WorldChunk] = {}
        self._camera_x: int = 0
        self._camera_y: int = 0
        self._load_radius: int = DEFAULT_LOAD_RADIUS

    def update_camera(self, x: float, y: float) -> None:
        """Update camera position and trigger chunk loading/unloading."""
        self._camera_x = int(x // CHUNK_SIZE)
        self._camera_y = int(y // CHUNK_SIZE)
        self._update_chunks()

    def _update_chunks(self) -> None:
        """Load chunks within radius, unload those outside."""
        r = self._load_radius
        cx, cy = self._camera_x, self._camera_y

        # Determine desired set.
        desired: set[tuple[int, int]] = set()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                desired.add((cx + dx, cy + dy))

        # Load new chunks.
        for key in desired:
            if key not in self._chunks:
                chunk = WorldChunk(chunk_x=key[0], chunk_y=key[1], state=ChunkState.LOADING)
                self._chunks[key] = chunk
            elif self._chunks[key].state == ChunkState.UNLOADING:
                self._chunks[key].state = ChunkState.LOADED

        # Promote loading -> loaded.
        for key in desired:
            chunk = self._chunks[key]
            if chunk.state == ChunkState.LOADING:
                chunk.state = ChunkState.LOADED

        # Unload chunks outside radius.
        to_remove: list[tuple[int, int]] = []
        for key, chunk in self._chunks.items():
            if key not in desired:
                if chunk.state in (ChunkState.LOADED, ChunkState.LOADING):
                    chunk.state = ChunkState.UNLOADING
                elif chunk.state == ChunkState.UNLOADING:
                    chunk.state = ChunkState.UNLOADED
                    to_remove.append(key)

        for key in to_remove:
            del self._chunks[key]

    def get_loaded_chunks(self) -> list[WorldChunk]:
        """Return all chunks in LOADED state."""
        return [c for c in self._chunks.values() if c.state == ChunkState.LOADED]

    def get_loading_radius(self) -> int:
        return self._load_radius

    def set_loading_radius(self, r: int) -> None:
        self._load_radius = max(0, r)
