"""Tests for WorldStreamManager."""

from engine.resource.streaming.world_streaming import (
    CHUNK_SIZE,
    DEFAULT_LOAD_RADIUS,
    ChunkState,
    WorldChunk,
    WorldStreamManager,
)


class TestWorldChunk:
    def test_default_state(self) -> None:
        chunk = WorldChunk()
        assert chunk.state is ChunkState.UNLOADED


class TestWorldStreamManager:
    def test_initial_no_loaded(self) -> None:
        mgr = WorldStreamManager()
        assert mgr.get_loaded_chunks() == []

    def test_default_loading_radius(self) -> None:
        mgr = WorldStreamManager()
        assert mgr.get_loading_radius() == DEFAULT_LOAD_RADIUS

    def test_set_loading_radius(self) -> None:
        mgr = WorldStreamManager()
        mgr.set_loading_radius(5)
        assert mgr.get_loading_radius() == 5

    def test_set_loading_radius_clamps_negative(self) -> None:
        mgr = WorldStreamManager()
        mgr.set_loading_radius(-1)
        assert mgr.get_loading_radius() == 0

    def test_camera_update_loads_chunks(self) -> None:
        mgr = WorldStreamManager()
        mgr.set_loading_radius(1)
        mgr.update_camera(0.0, 0.0)
        loaded = mgr.get_loaded_chunks()
        # Radius 1 => 3x3 = 9 chunks
        expected_count = (2 * 1 + 1) ** 2
        assert len(loaded) == expected_count
        assert all(c.state is ChunkState.LOADED for c in loaded)

    def test_camera_move_unloads_old_chunks(self) -> None:
        mgr = WorldStreamManager()
        mgr.set_loading_radius(0)  # Only center chunk
        mgr.update_camera(0.0, 0.0)
        assert len(mgr.get_loaded_chunks()) == 1

        # Move far away
        mgr.update_camera(CHUNK_SIZE * 10.0, 0.0)
        loaded = mgr.get_loaded_chunks()
        # New center loaded; old center is unloading (one tick)
        assert len(loaded) == 1
        assert loaded[0].chunk_x == 10

    def test_chunk_coordinates(self) -> None:
        mgr = WorldStreamManager()
        mgr.set_loading_radius(0)
        mgr.update_camera(CHUNK_SIZE * 3.5, CHUNK_SIZE * 2.5)
        loaded = mgr.get_loaded_chunks()
        assert len(loaded) == 1
        assert loaded[0].chunk_x == 3
        assert loaded[0].chunk_y == 2
