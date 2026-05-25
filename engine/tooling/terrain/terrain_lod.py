"""
Terrain LOD system for the AI Game Engine.

Provides chunk-based terrain streaming with multiple LOD levels
for efficient rendering of large terrains.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any, Callable
import math


class LODLevel(Enum):
    """Terrain LOD levels."""
    LOD0 = 0  # Full resolution
    LOD1 = 1  # 1/2 resolution
    LOD2 = 2  # 1/4 resolution
    LOD3 = 3  # 1/8 resolution
    LOD4 = 4  # 1/16 resolution


class ChunkState(Enum):
    """State of a terrain chunk."""
    UNLOADED = auto()
    LOADING = auto()
    LOADED = auto()
    STREAMING_IN = auto()
    STREAMING_OUT = auto()
    ERROR = auto()


@dataclass(slots=True)
class ChunkBounds:
    """Axis-aligned bounding box for a chunk."""
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @property
    def center_x(self) -> float:
        """Get center X coordinate."""
        return (self.min_x + self.max_x) / 2

    @property
    def center_y(self) -> float:
        """Get center Y coordinate."""
        return (self.min_y + self.max_y) / 2

    @property
    def center_z(self) -> float:
        """Get center Z coordinate."""
        return (self.min_z + self.max_z) / 2

    def distance_to_point(self, x: float, y: float, z: float) -> float:
        """Calculate distance from point to nearest edge of bounds."""
        dx = max(self.min_x - x, 0, x - self.max_x)
        dy = max(self.min_y - y, 0, y - self.max_y)
        dz = max(self.min_z - z, 0, z - self.max_z)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Check if point is inside bounds."""
        return (self.min_x <= x <= self.max_x and
                self.min_y <= y <= self.max_y and
                self.min_z <= z <= self.max_z)


@dataclass(slots=True)
class TerrainChunk:
    """
    A chunk of terrain data.

    Represents a portion of the terrain with its own height data,
    LOD state, and streaming status.
    """
    id: int
    grid_x: int
    grid_z: int
    bounds: ChunkBounds
    state: ChunkState = ChunkState.UNLOADED
    current_lod: LODLevel = LODLevel.LOD0
    target_lod: LODLevel = LODLevel.LOD0
    heights: Optional[list[list[float]]] = None
    lod_heights: dict[LODLevel, list[list[float]]] = field(default_factory=dict)
    neighbors: dict[str, Optional[int]] = field(default_factory=lambda: {
        "north": None, "south": None, "east": None, "west": None
    })
    last_access_time: float = 0.0
    priority: float = 0.0

    @property
    def is_loaded(self) -> bool:
        """Check if chunk is loaded."""
        return self.state == ChunkState.LOADED

    @property
    def is_transitioning(self) -> bool:
        """Check if chunk is transitioning between LOD levels."""
        return self.current_lod != self.target_lod

    def get_height_at(self, local_x: int, local_z: int) -> float:
        """Get height at local coordinates."""
        if self.heights is None:
            return 0.0
        if 0 <= local_z < len(self.heights) and 0 <= local_x < len(self.heights[0]):
            return self.heights[local_z][local_x]
        return 0.0

    def set_heights(self, heights: list[list[float]], lod: LODLevel = LODLevel.LOD0) -> None:
        """Set height data for a LOD level."""
        if lod == LODLevel.LOD0:
            self.heights = heights
        self.lod_heights[lod] = heights


@dataclass(slots=True)
class TerrainLODSettings:
    """Settings for terrain LOD system."""
    chunk_size: int = 64  # Vertices per chunk side
    chunk_world_size: float = 64.0  # World units per chunk
    lod0_distance: float = 100.0
    lod1_distance: float = 200.0
    lod2_distance: float = 400.0
    lod3_distance: float = 800.0
    lod4_distance: float = 1600.0
    max_loaded_chunks: int = 256
    stream_distance: float = 2000.0
    morph_range: float = 0.15  # Fraction of LOD distance for morphing
    priority_bias: float = 1.0

    def get_lod_for_distance(self, distance: float) -> LODLevel:
        """Determine LOD level for a given distance."""
        if distance <= self.lod0_distance:
            return LODLevel.LOD0
        elif distance <= self.lod1_distance:
            return LODLevel.LOD1
        elif distance <= self.lod2_distance:
            return LODLevel.LOD2
        elif distance <= self.lod3_distance:
            return LODLevel.LOD3
        else:
            return LODLevel.LOD4

    def get_lod_resolution(self, lod: LODLevel) -> int:
        """Get resolution for a LOD level."""
        divisor = 2 ** lod.value
        return max(2, self.chunk_size // divisor)


class TerrainChunkManager:
    """
    Manages terrain chunks and their streaming.

    Handles chunk loading, unloading, and LOD transitions
    based on camera position.
    """
    __slots__ = (
        "_settings",
        "_chunks",
        "_chunk_grid",
        "_grid_width",
        "_grid_height",
        "_next_chunk_id",
        "_pending_loads",
        "_pending_unloads",
        "_height_source",
    )

    def __init__(
        self,
        terrain_width: int,
        terrain_height: int,
        settings: Optional[TerrainLODSettings] = None
    ):
        """
        Initialize chunk manager.

        Args:
            terrain_width: Total terrain width in vertices
            terrain_height: Total terrain height in vertices
            settings: LOD settings
        """
        self._settings = settings or TerrainLODSettings()
        self._chunks: dict[int, TerrainChunk] = {}
        self._next_chunk_id = 0

        # Calculate grid dimensions
        self._grid_width = math.ceil(terrain_width / self._settings.chunk_size)
        self._grid_height = math.ceil(terrain_height / self._settings.chunk_size)

        # Create chunk grid
        self._chunk_grid: list[list[Optional[int]]] = [
            [None for _ in range(self._grid_width)]
            for _ in range(self._grid_height)
        ]

        self._pending_loads: list[int] = []
        self._pending_unloads: list[int] = []
        self._height_source: Optional[Callable[[int, int, int, int], list[list[float]]]] = None

        # Initialize chunks
        self._create_chunks()

    @property
    def settings(self) -> TerrainLODSettings:
        """Get LOD settings."""
        return self._settings

    @property
    def grid_width(self) -> int:
        """Get grid width in chunks."""
        return self._grid_width

    @property
    def grid_height(self) -> int:
        """Get grid height in chunks."""
        return self._grid_height

    def set_height_source(
        self,
        source: Callable[[int, int, int, int], list[list[float]]]
    ) -> None:
        """
        Set the height data source function.

        Args:
            source: Function that returns height data for a region
                   (start_x, start_z, width, height) -> heights
        """
        self._height_source = source

    def _create_chunks(self) -> None:
        """Create all terrain chunks."""
        for gz in range(self._grid_height):
            for gx in range(self._grid_width):
                chunk_id = self._next_chunk_id
                self._next_chunk_id += 1

                world_x = gx * self._settings.chunk_world_size
                world_z = gz * self._settings.chunk_world_size

                chunk = TerrainChunk(
                    id=chunk_id,
                    grid_x=gx,
                    grid_z=gz,
                    bounds=ChunkBounds(
                        min_x=world_x,
                        min_y=-1000.0,
                        min_z=world_z,
                        max_x=world_x + self._settings.chunk_world_size,
                        max_y=1000.0,
                        max_z=world_z + self._settings.chunk_world_size,
                    ),
                )

                self._chunks[chunk_id] = chunk
                self._chunk_grid[gz][gx] = chunk_id

        # Set up neighbor references
        self._setup_neighbors()

    def _setup_neighbors(self) -> None:
        """Set up neighbor references for all chunks."""
        for gz in range(self._grid_height):
            for gx in range(self._grid_width):
                chunk_id = self._chunk_grid[gz][gx]
                if chunk_id is None:
                    continue

                chunk = self._chunks[chunk_id]

                # North
                if gz > 0:
                    chunk.neighbors["north"] = self._chunk_grid[gz - 1][gx]
                # South
                if gz < self._grid_height - 1:
                    chunk.neighbors["south"] = self._chunk_grid[gz + 1][gx]
                # West
                if gx > 0:
                    chunk.neighbors["west"] = self._chunk_grid[gz][gx - 1]
                # East
                if gx < self._grid_width - 1:
                    chunk.neighbors["east"] = self._chunk_grid[gz][gx + 1]

    def get_chunk(self, chunk_id: int) -> Optional[TerrainChunk]:
        """Get a chunk by ID."""
        return self._chunks.get(chunk_id)

    def get_chunk_at_grid(self, grid_x: int, grid_z: int) -> Optional[TerrainChunk]:
        """Get chunk at grid coordinates."""
        if 0 <= grid_x < self._grid_width and 0 <= grid_z < self._grid_height:
            chunk_id = self._chunk_grid[grid_z][grid_x]
            if chunk_id is not None:
                return self._chunks[chunk_id]
        return None

    def get_chunk_at_world(self, world_x: float, world_z: float) -> Optional[TerrainChunk]:
        """Get chunk at world coordinates."""
        grid_x = int(world_x / self._settings.chunk_world_size)
        grid_z = int(world_z / self._settings.chunk_world_size)
        return self.get_chunk_at_grid(grid_x, grid_z)

    def update(
        self,
        camera_x: float,
        camera_y: float,
        camera_z: float,
        current_time: float
    ) -> dict[str, int]:
        """
        Update chunk LODs and streaming based on camera position.

        Args:
            camera_x, camera_y, camera_z: Camera position
            current_time: Current time for access tracking

        Returns:
            Statistics about the update
        """
        stats = {
            "chunks_loaded": 0,
            "chunks_unloaded": 0,
            "lod_changes": 0,
            "total_loaded": 0,
        }

        # Calculate priorities and target LODs
        chunks_by_priority: list[tuple[float, int]] = []

        for chunk_id, chunk in self._chunks.items():
            distance = chunk.bounds.distance_to_point(camera_x, camera_y, camera_z)

            # Determine target LOD
            target_lod = self._settings.get_lod_for_distance(distance)

            # Calculate priority (lower distance = higher priority)
            priority = 1.0 / (distance + 1.0) * self._settings.priority_bias
            chunk.priority = priority
            chunk.target_lod = target_lod

            # Determine if chunk should be loaded
            if distance <= self._settings.stream_distance:
                chunks_by_priority.append((priority, chunk_id))
                chunk.last_access_time = current_time

        # Sort by priority (highest first)
        chunks_by_priority.sort(reverse=True)

        # Load/unload chunks based on priority and limits
        loaded_count = sum(1 for c in self._chunks.values() if c.is_loaded)

        for priority, chunk_id in chunks_by_priority:
            chunk = self._chunks[chunk_id]

            if not chunk.is_loaded:
                if loaded_count < self._settings.max_loaded_chunks:
                    self._load_chunk(chunk)
                    stats["chunks_loaded"] += 1
                    loaded_count += 1
            else:
                # Update LOD if needed
                if chunk.current_lod != chunk.target_lod:
                    self._transition_lod(chunk)
                    stats["lod_changes"] += 1

        # Unload distant chunks if over limit
        if loaded_count > self._settings.max_loaded_chunks:
            # Find lowest priority loaded chunks
            loaded_chunks = [
                (c.priority, c.id) for c in self._chunks.values()
                if c.is_loaded
            ]
            loaded_chunks.sort()

            while loaded_count > self._settings.max_loaded_chunks and loaded_chunks:
                _, chunk_id = loaded_chunks.pop(0)
                chunk = self._chunks[chunk_id]
                self._unload_chunk(chunk)
                stats["chunks_unloaded"] += 1
                loaded_count -= 1

        stats["total_loaded"] = loaded_count
        return stats

    def _load_chunk(self, chunk: TerrainChunk) -> None:
        """Load a chunk's height data."""
        chunk.state = ChunkState.LOADING

        if self._height_source:
            start_x = chunk.grid_x * self._settings.chunk_size
            start_z = chunk.grid_z * self._settings.chunk_size

            heights = self._height_source(
                start_x, start_z,
                self._settings.chunk_size, self._settings.chunk_size
            )
            chunk.set_heights(heights, LODLevel.LOD0)

            # Generate lower LOD heights
            self._generate_lod_heights(chunk)

        chunk.state = ChunkState.LOADED

    def _unload_chunk(self, chunk: TerrainChunk) -> None:
        """Unload a chunk's height data."""
        chunk.state = ChunkState.STREAMING_OUT
        chunk.heights = None
        chunk.lod_heights.clear()
        chunk.state = ChunkState.UNLOADED

    def _transition_lod(self, chunk: TerrainChunk) -> None:
        """Transition chunk to target LOD."""
        chunk.current_lod = chunk.target_lod

    def _generate_lod_heights(self, chunk: TerrainChunk) -> None:
        """Generate lower LOD height data from full resolution."""
        if chunk.heights is None:
            return

        for lod in [LODLevel.LOD1, LODLevel.LOD2, LODLevel.LOD3, LODLevel.LOD4]:
            resolution = self._settings.get_lod_resolution(lod)
            step = self._settings.chunk_size // resolution

            lod_heights: list[list[float]] = []
            for z in range(resolution):
                row: list[float] = []
                for x in range(resolution):
                    src_x = x * step
                    src_z = z * step
                    row.append(chunk.get_height_at(src_x, src_z))
                lod_heights.append(row)

            chunk.lod_heights[lod] = lod_heights

    def get_loaded_chunks(self) -> list[TerrainChunk]:
        """Get all loaded chunks."""
        return [c for c in self._chunks.values() if c.is_loaded]

    def get_visible_chunks(
        self,
        camera_x: float,
        camera_y: float,
        camera_z: float,
        view_distance: float
    ) -> list[TerrainChunk]:
        """Get chunks within view distance."""
        visible = []
        for chunk in self._chunks.values():
            if chunk.is_loaded:
                distance = chunk.bounds.distance_to_point(camera_x, camera_y, camera_z)
                if distance <= view_distance:
                    visible.append(chunk)
        return visible

    def force_load_region(
        self,
        min_x: float,
        min_z: float,
        max_x: float,
        max_z: float
    ) -> int:
        """
        Force load all chunks in a region.

        Returns:
            Number of chunks loaded
        """
        loaded = 0
        min_gx = max(0, int(min_x / self._settings.chunk_world_size))
        min_gz = max(0, int(min_z / self._settings.chunk_world_size))
        max_gx = min(self._grid_width - 1, int(max_x / self._settings.chunk_world_size))
        max_gz = min(self._grid_height - 1, int(max_z / self._settings.chunk_world_size))

        for gz in range(min_gz, max_gz + 1):
            for gx in range(min_gx, max_gx + 1):
                chunk = self.get_chunk_at_grid(gx, gz)
                if chunk and not chunk.is_loaded:
                    self._load_chunk(chunk)
                    loaded += 1

        return loaded

    def unload_all(self) -> None:
        """Unload all chunks."""
        for chunk in self._chunks.values():
            if chunk.is_loaded:
                self._unload_chunk(chunk)


class TerrainLODSystem:
    """
    High-level terrain LOD system.

    Provides a unified interface for terrain LOD management,
    combining chunk management with rendering hints.
    """
    __slots__ = (
        "_chunk_manager",
        "_terrain_heights",
        "_terrain_width",
        "_terrain_height",
    )

    def __init__(
        self,
        terrain_width: int,
        terrain_height: int,
        terrain_heights: list[list[float]],
        settings: Optional[TerrainLODSettings] = None
    ):
        """
        Initialize LOD system.

        Args:
            terrain_width: Terrain width in vertices
            terrain_height: Terrain height in vertices
            terrain_heights: Full terrain height data
            settings: LOD settings
        """
        self._terrain_width = terrain_width
        self._terrain_height = terrain_height
        self._terrain_heights = terrain_heights

        self._chunk_manager = TerrainChunkManager(
            terrain_width, terrain_height, settings
        )

        # Set up height source
        self._chunk_manager.set_height_source(self._get_heights_for_region)

    def _get_heights_for_region(
        self,
        start_x: int,
        start_z: int,
        width: int,
        height: int
    ) -> list[list[float]]:
        """Extract height data for a region."""
        heights: list[list[float]] = []

        for z in range(height):
            row: list[float] = []
            for x in range(width):
                sx = start_x + x
                sz = start_z + z

                if 0 <= sx < self._terrain_width and 0 <= sz < self._terrain_height:
                    row.append(self._terrain_heights[sz][sx])
                else:
                    row.append(0.0)
            heights.append(row)

        return heights

    @property
    def chunk_manager(self) -> TerrainChunkManager:
        """Get the chunk manager."""
        return self._chunk_manager

    def update(
        self,
        camera_x: float,
        camera_y: float,
        camera_z: float,
        current_time: float = 0.0
    ) -> dict[str, int]:
        """
        Update LOD system.

        Args:
            camera_x, camera_y, camera_z: Camera position
            current_time: Current time

        Returns:
            Update statistics
        """
        return self._chunk_manager.update(camera_x, camera_y, camera_z, current_time)

    def get_height_at(self, world_x: float, world_z: float) -> float:
        """
        Get height at world position.

        Uses loaded chunk data when available.
        """
        chunk = self._chunk_manager.get_chunk_at_world(world_x, world_z)

        if chunk and chunk.is_loaded and chunk.heights:
            # Convert to local coordinates
            local_x = int(world_x - chunk.bounds.min_x)
            local_z = int(world_z - chunk.bounds.min_z)
            return chunk.get_height_at(local_x, local_z)

        # Fallback to full terrain data
        ix = int(world_x)
        iz = int(world_z)
        if 0 <= ix < self._terrain_width and 0 <= iz < self._terrain_height:
            return self._terrain_heights[iz][ix]

        return 0.0

    def get_render_batches(
        self,
        camera_x: float,
        camera_y: float,
        camera_z: float
    ) -> dict[LODLevel, list[TerrainChunk]]:
        """
        Get chunks grouped by LOD level for rendering.

        Returns:
            Dictionary mapping LOD levels to chunk lists
        """
        batches: dict[LODLevel, list[TerrainChunk]] = {
            lod: [] for lod in LODLevel
        }

        for chunk in self._chunk_manager.get_loaded_chunks():
            batches[chunk.current_lod].append(chunk)

        return batches

    def force_full_load(self) -> int:
        """Force load all chunks."""
        return self._chunk_manager.force_load_region(
            0, 0,
            self._terrain_width * self._chunk_manager.settings.chunk_world_size / self._chunk_manager.settings.chunk_size,
            self._terrain_height * self._chunk_manager.settings.chunk_world_size / self._chunk_manager.settings.chunk_size,
        )
