"""Tests for terrain LOD system."""

import pytest
from engine.tooling.terrain.terrain_lod import (
    LODLevel,
    ChunkState,
    ChunkBounds,
    TerrainChunk,
    TerrainLODSettings,
    TerrainChunkManager,
    TerrainLODSystem,
)


class TestChunkBounds:
    """Tests for chunk bounds."""

    def test_creation(self):
        """Test bounds creation."""
        bounds = ChunkBounds(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
        assert bounds.min_x == 0.0
        assert bounds.max_x == 10.0

    def test_center(self):
        """Test center calculation."""
        bounds = ChunkBounds(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
        assert bounds.center_x == 5.0
        assert bounds.center_y == 5.0
        assert bounds.center_z == 5.0

    def test_contains_point_inside(self):
        """Test point containment inside."""
        bounds = ChunkBounds(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
        assert bounds.contains_point(5.0, 5.0, 5.0)

    def test_contains_point_outside(self):
        """Test point containment outside."""
        bounds = ChunkBounds(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
        assert not bounds.contains_point(20.0, 20.0, 20.0)

    def test_distance_inside(self):
        """Test distance from inside point."""
        bounds = ChunkBounds(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
        dist = bounds.distance_to_point(5.0, 5.0, 5.0)
        assert dist == 0.0

    def test_distance_outside(self):
        """Test distance from outside point."""
        bounds = ChunkBounds(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
        dist = bounds.distance_to_point(20.0, 5.0, 5.0)
        assert dist == 10.0


class TestTerrainChunk:
    """Tests for terrain chunk."""

    def test_creation(self):
        """Test chunk creation."""
        chunk = TerrainChunk(
            id=0,
            grid_x=0,
            grid_z=0,
            bounds=ChunkBounds(0.0, 0.0, 0.0, 64.0, 100.0, 64.0),
        )
        assert chunk.id == 0
        assert chunk.state == ChunkState.UNLOADED

    def test_is_loaded(self):
        """Test loaded state check."""
        chunk = TerrainChunk(
            id=0, grid_x=0, grid_z=0,
            bounds=ChunkBounds(0.0, 0.0, 0.0, 64.0, 100.0, 64.0),
        )
        assert not chunk.is_loaded

        chunk.state = ChunkState.LOADED
        assert chunk.is_loaded

    def test_is_transitioning(self):
        """Test transitioning state."""
        chunk = TerrainChunk(
            id=0, grid_x=0, grid_z=0,
            bounds=ChunkBounds(0.0, 0.0, 0.0, 64.0, 100.0, 64.0),
        )
        chunk.current_lod = LODLevel.LOD0
        chunk.target_lod = LODLevel.LOD1

        assert chunk.is_transitioning

    def test_get_height(self):
        """Test height retrieval."""
        chunk = TerrainChunk(
            id=0, grid_x=0, grid_z=0,
            bounds=ChunkBounds(0.0, 0.0, 0.0, 64.0, 100.0, 64.0),
        )
        chunk.heights = [[0.5 for _ in range(4)] for _ in range(4)]

        assert chunk.get_height_at(0, 0) == 0.5
        assert chunk.get_height_at(10, 10) == 0.0  # Out of bounds

    def test_set_heights(self):
        """Test setting height data."""
        chunk = TerrainChunk(
            id=0, grid_x=0, grid_z=0,
            bounds=ChunkBounds(0.0, 0.0, 0.0, 64.0, 100.0, 64.0),
        )
        heights = [[0.5 for _ in range(4)] for _ in range(4)]
        chunk.set_heights(heights, LODLevel.LOD0)

        assert chunk.heights == heights
        assert LODLevel.LOD0 in chunk.lod_heights


class TestTerrainLODSettings:
    """Tests for LOD settings."""

    def test_default_values(self):
        """Test default LOD settings."""
        settings = TerrainLODSettings()
        assert settings.chunk_size == 64
        assert settings.lod0_distance == 100.0
        assert settings.max_loaded_chunks == 256

    def test_get_lod_for_distance(self):
        """Test LOD level calculation."""
        settings = TerrainLODSettings()
        # Default distances: lod0=100, lod1=200, lod2=400, lod3=800, lod4=1600
        # Implementation uses <= so boundary values go to lower LOD

        assert settings.get_lod_for_distance(50.0) == LODLevel.LOD0   # 50 <= 100
        assert settings.get_lod_for_distance(100.0) == LODLevel.LOD0  # 100 <= 100 (boundary)
        assert settings.get_lod_for_distance(150.0) == LODLevel.LOD1  # 150 <= 200
        assert settings.get_lod_for_distance(300.0) == LODLevel.LOD2  # 300 <= 400

    def test_get_lod_resolution(self):
        """Test LOD resolution calculation."""
        settings = TerrainLODSettings(chunk_size=64)

        assert settings.get_lod_resolution(LODLevel.LOD0) == 64
        assert settings.get_lod_resolution(LODLevel.LOD1) == 32
        assert settings.get_lod_resolution(LODLevel.LOD2) == 16


class TestTerrainChunkManager:
    """Tests for chunk manager."""

    def setup_method(self):
        """Set up test manager."""
        self.manager = TerrainChunkManager(
            terrain_width=256,
            terrain_height=256,
            settings=TerrainLODSettings(chunk_size=64),
        )

    def test_creation(self):
        """Test manager creation."""
        assert self.manager.grid_width == 4
        assert self.manager.grid_height == 4

    def test_get_chunk_at_grid(self):
        """Test getting chunk by grid coordinates."""
        chunk = self.manager.get_chunk_at_grid(0, 0)
        assert chunk is not None
        assert chunk.grid_x == 0
        assert chunk.grid_z == 0

    def test_get_chunk_at_world(self):
        """Test getting chunk by world coordinates."""
        chunk = self.manager.get_chunk_at_world(32.0, 32.0)
        assert chunk is not None
        assert chunk.grid_x == 0

    def test_get_nonexistent_chunk(self):
        """Test getting nonexistent chunk."""
        chunk = self.manager.get_chunk_at_grid(100, 100)
        assert chunk is None

    def test_neighbor_setup(self):
        """Test neighbor references."""
        chunk = self.manager.get_chunk_at_grid(1, 1)
        assert chunk.neighbors["north"] is not None
        assert chunk.neighbors["south"] is not None
        assert chunk.neighbors["east"] is not None
        assert chunk.neighbors["west"] is not None

    def test_corner_chunk_neighbors(self):
        """Test corner chunk has partial neighbors."""
        chunk = self.manager.get_chunk_at_grid(0, 0)
        assert chunk.neighbors["north"] is None
        assert chunk.neighbors["west"] is None
        assert chunk.neighbors["south"] is not None
        assert chunk.neighbors["east"] is not None

    def test_update_loads_chunks(self):
        """Test update loads nearby chunks."""
        # Set height source
        def height_source(sx, sz, w, h):
            return [[0.0 for _ in range(w)] for _ in range(h)]

        self.manager.set_height_source(height_source)

        stats = self.manager.update(32.0, 0.0, 32.0, 0.0)
        assert stats["chunks_loaded"] > 0

    def test_get_loaded_chunks(self):
        """Test getting loaded chunks."""
        def height_source(sx, sz, w, h):
            return [[0.0 for _ in range(w)] for _ in range(h)]

        self.manager.set_height_source(height_source)
        self.manager.update(32.0, 0.0, 32.0, 0.0)

        loaded = self.manager.get_loaded_chunks()
        assert len(loaded) > 0

    def test_force_load_region(self):
        """Test force loading a region."""
        def height_source(sx, sz, w, h):
            return [[0.0 for _ in range(w)] for _ in range(h)]

        self.manager.set_height_source(height_source)

        count = self.manager.force_load_region(0.0, 0.0, 128.0, 128.0)
        assert count > 0

    def test_unload_all(self):
        """Test unloading all chunks."""
        def height_source(sx, sz, w, h):
            return [[0.0 for _ in range(w)] for _ in range(h)]

        self.manager.set_height_source(height_source)
        self.manager.force_load_region(0.0, 0.0, 128.0, 128.0)

        self.manager.unload_all()
        loaded = self.manager.get_loaded_chunks()
        assert len(loaded) == 0


class TestTerrainLODSystem:
    """Tests for high-level LOD system."""

    def setup_method(self):
        """Set up test LOD system."""
        self.heights = [[0.5 for _ in range(64)] for _ in range(64)]
        self.system = TerrainLODSystem(
            terrain_width=64,
            terrain_height=64,
            terrain_heights=self.heights,
            settings=TerrainLODSettings(chunk_size=32),
        )

    def test_creation(self):
        """Test system creation."""
        assert self.system.chunk_manager is not None

    def test_update(self):
        """Test LOD update."""
        stats = self.system.update(16.0, 0.0, 16.0)
        assert "chunks_loaded" in stats

    def test_get_height_at(self):
        """Test height retrieval."""
        height = self.system.get_height_at(10.0, 10.0)
        assert height == 0.5

    def test_get_render_batches(self):
        """Test render batch grouping."""
        self.system.update(16.0, 0.0, 16.0)
        batches = self.system.get_render_batches(16.0, 0.0, 16.0)

        assert LODLevel.LOD0 in batches

    def test_force_full_load(self):
        """Test force loading all chunks."""
        count = self.system.force_full_load()
        # Should load some chunks
