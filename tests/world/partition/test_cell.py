"""
Tests for the StreamingCell module.

Tests CellState transitions, actor management, and neighbor lookup.
"""

import pytest
from engine.core.math.vec import Vec3
from engine.world.partition.cell import (
    CellActor,
    CellCoord,
    CellState,
    StreamingCell,
)


# =============================================================================
# CellCoord Tests
# =============================================================================

class TestCellCoord:
    """Tests for CellCoord class."""

    def test_coord_creation(self):
        """Test coordinate creation."""
        coord = CellCoord(5, 10)
        assert coord.x == 5
        assert coord.y == 10

    def test_coord_immutable(self):
        """Test that coordinates are immutable (frozen dataclass)."""
        coord = CellCoord(5, 10)
        with pytest.raises(AttributeError):
            coord.x = 6

    def test_coord_equality(self):
        """Test coordinate equality."""
        coord1 = CellCoord(5, 10)
        coord2 = CellCoord(5, 10)
        coord3 = CellCoord(5, 11)
        assert coord1 == coord2
        assert coord1 != coord3

    def test_coord_hash(self):
        """Test coordinates can be used as dict keys."""
        coord1 = CellCoord(5, 10)
        coord2 = CellCoord(5, 10)
        d = {coord1: "test"}
        assert d[coord2] == "test"

    def test_coord_add(self):
        """Test coordinate addition."""
        coord1 = CellCoord(5, 10)
        coord2 = CellCoord(3, 2)
        result = coord1 + coord2
        assert result.x == 8
        assert result.y == 12

    def test_coord_subtract(self):
        """Test coordinate subtraction."""
        coord1 = CellCoord(5, 10)
        coord2 = CellCoord(3, 2)
        result = coord1 - coord2
        assert result.x == 2
        assert result.y == 8

    def test_coord_distance_manhattan(self):
        """Test Manhattan distance calculation."""
        coord1 = CellCoord(0, 0)
        coord2 = CellCoord(3, 4)
        assert coord1.distance_manhattan(coord2) == 7

    def test_coord_distance_chebyshev(self):
        """Test Chebyshev distance calculation."""
        coord1 = CellCoord(0, 0)
        coord2 = CellCoord(3, 4)
        assert coord1.distance_chebyshev(coord2) == 4

    def test_coord_as_tuple(self):
        """Test conversion to tuple."""
        coord = CellCoord(5, 10)
        assert coord.as_tuple() == (5, 10)

    def test_coord_from_tuple(self):
        """Test creation from tuple."""
        coord = CellCoord.from_tuple((5, 10))
        assert coord.x == 5
        assert coord.y == 10


# =============================================================================
# CellActor Tests
# =============================================================================

class TestCellActor:
    """Tests for CellActor class."""

    def test_actor_creation_default(self):
        """Test default actor creation."""
        actor = CellActor()
        assert actor.id == ""
        assert actor.name == ""
        assert actor.priority == 0
        assert actor.persistent is False

    def test_actor_creation_custom(self):
        """Test custom actor creation."""
        actor = CellActor(
            id="actor_001",
            name="Test Actor",
            position=Vec3(100, 50, 200),
            priority=10,
            persistent=True,
            tags={"enemy", "spawnable"},
        )
        assert actor.id == "actor_001"
        assert actor.position.x == 100
        assert "enemy" in actor.tags


# =============================================================================
# StreamingCell Creation Tests
# =============================================================================

class TestStreamingCellCreation:
    """Tests for StreamingCell creation and initialization."""

    def test_cell_creation_default(self):
        """Test default cell creation."""
        cell = StreamingCell()
        assert cell.coord == CellCoord(0, 0)
        assert cell.state == CellState.UNLOADED
        assert cell.load_progress == 0.0

    def test_cell_creation_custom(self):
        """Test custom cell creation."""
        cell = StreamingCell(
            coord=CellCoord(5, 10),
            bounds_min=Vec3(1280, 0, 2560),
            bounds_max=Vec3(1536, 256, 2816),
            priority=5,
        )
        assert cell.coord.x == 5
        assert cell.coord.y == 10
        assert cell.priority == 5

    def test_cell_bounds_property(self):
        """Test bounds AABB property."""
        cell = StreamingCell(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(256, 256, 256),
        )
        aabb = cell.bounds
        assert aabb.min.x == 0
        assert aabb.max.x == 256

    def test_cell_center_property(self):
        """Test center calculation."""
        cell = StreamingCell(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(256, 256, 256),
        )
        center = cell.center
        assert center.x == 128
        assert center.y == 128
        assert center.z == 128

    def test_cell_size_property(self):
        """Test size calculation."""
        cell = StreamingCell(
            bounds_min=Vec3(100, 50, 200),
            bounds_max=Vec3(300, 150, 450),
        )
        size = cell.size
        assert size.x == 200
        assert size.y == 100
        assert size.z == 250


# =============================================================================
# StreamingCell State Tests
# =============================================================================

class TestStreamingCellState:
    """Tests for cell state transitions."""

    def test_cell_is_loaded_property(self):
        """Test is_loaded property for various states."""
        cell = StreamingCell()
        assert cell.is_loaded is False

        cell.state = CellState.LOADED
        assert cell.is_loaded is True

        cell.state = CellState.ACTIVATED
        assert cell.is_loaded is True

    def test_cell_is_active_property(self):
        """Test is_active property."""
        cell = StreamingCell()
        assert cell.is_active is False

        cell.state = CellState.ACTIVATED
        assert cell.is_active is True

    def test_cell_is_loading_property(self):
        """Test is_loading property."""
        cell = StreamingCell()
        assert cell.is_loading is False

        cell.state = CellState.LOADING
        assert cell.is_loading is True

    def test_cell_is_unloaded_property(self):
        """Test is_unloaded property."""
        cell = StreamingCell()
        assert cell.is_unloaded is True

        cell.state = CellState.LOADED
        assert cell.is_unloaded is False

    def test_cell_load_from_unloaded(self):
        """Test loading from unloaded state."""
        cell = StreamingCell()
        result = cell.load()
        assert result is True
        assert cell.state == CellState.LOADING
        assert cell.load_progress == 0.0

    def test_cell_load_already_loading(self):
        """Test loading when already loading."""
        cell = StreamingCell()
        cell.state = CellState.LOADING
        result = cell.load()
        assert result is False

    def test_cell_complete_load(self):
        """Test completing load."""
        cell = StreamingCell()
        cell.load()
        result = cell.complete_load(timestamp=123.0)
        assert result is True
        assert cell.state == CellState.LOADED
        assert cell.load_progress == 1.0
        assert cell.last_load_time == 123.0

    def test_cell_complete_load_wrong_state(self):
        """Test completing load when not loading."""
        cell = StreamingCell()
        result = cell.complete_load()
        assert result is False

    def test_cell_unload_from_loaded(self):
        """Test unloading from loaded state."""
        cell = StreamingCell()
        cell.state = CellState.LOADED
        result = cell.unload()
        assert result is True
        assert cell.state == CellState.UNLOADING

    def test_cell_unload_from_activated(self):
        """Test unloading from activated state."""
        cell = StreamingCell()
        cell.state = CellState.ACTIVATED
        result = cell.unload()
        assert result is True
        assert cell.state == CellState.UNLOADING

    def test_cell_unload_from_unloaded(self):
        """Test unloading when already unloaded."""
        cell = StreamingCell()
        result = cell.unload()
        assert result is False

    def test_cell_complete_unload(self):
        """Test completing unload."""
        cell = StreamingCell()
        cell.state = CellState.LOADED
        cell.add_actor(CellActor(id="a1"))
        cell.unload()
        result = cell.complete_unload()
        assert result is True
        assert cell.state == CellState.UNLOADED
        assert len(cell.actors) == 0

    def test_cell_activate(self):
        """Test activating a loaded cell."""
        cell = StreamingCell()
        cell.state = CellState.LOADED
        result = cell.activate(timestamp=456.0)
        assert result is True
        assert cell.state == CellState.ACTIVATED
        assert cell.last_activate_time == 456.0

    def test_cell_activate_not_loaded(self):
        """Test activating a non-loaded cell fails."""
        cell = StreamingCell()
        result = cell.activate()
        assert result is False

    def test_cell_deactivate(self):
        """Test deactivating a cell."""
        cell = StreamingCell()
        cell.state = CellState.ACTIVATED
        result = cell.deactivate()
        assert result is True
        assert cell.state == CellState.LOADED

    def test_cell_deactivate_not_activated(self):
        """Test deactivating non-activated cell fails."""
        cell = StreamingCell()
        cell.state = CellState.LOADED
        result = cell.deactivate()
        assert result is False

    def test_cell_update_load_progress(self):
        """Test updating load progress."""
        cell = StreamingCell()
        cell.update_load_progress(0.5)
        assert cell.load_progress == 0.5

    def test_cell_load_progress_clamping(self):
        """Test load progress is clamped."""
        cell = StreamingCell()
        cell.update_load_progress(1.5)
        assert cell.load_progress == 1.0
        cell.update_load_progress(-0.5)
        assert cell.load_progress == 0.0


# =============================================================================
# StreamingCell Actor Management Tests
# =============================================================================

class TestStreamingCellActors:
    """Tests for cell actor management."""

    def test_cell_add_actor(self):
        """Test adding actor to cell."""
        cell = StreamingCell()
        actor = CellActor(id="a1", name="Actor1")
        cell.add_actor(actor)
        assert len(cell.actors) == 1
        assert cell.actors[0].id == "a1"

    def test_cell_remove_actor(self):
        """Test removing actor from cell."""
        cell = StreamingCell()
        cell.add_actor(CellActor(id="a1"))
        result = cell.remove_actor("a1")
        assert result is True
        assert len(cell.actors) == 0

    def test_cell_remove_actor_not_found(self):
        """Test removing non-existent actor."""
        cell = StreamingCell()
        result = cell.remove_actor("nonexistent")
        assert result is False

    def test_cell_get_actor(self):
        """Test getting actor by ID."""
        cell = StreamingCell()
        cell.add_actor(CellActor(id="a1", name="Test"))
        actor = cell.get_actor("a1")
        assert actor is not None
        assert actor.name == "Test"

    def test_cell_get_actor_not_found(self):
        """Test getting non-existent actor."""
        cell = StreamingCell()
        actor = cell.get_actor("nonexistent")
        assert actor is None

    def test_cell_get_actors_by_tag(self):
        """Test getting actors by tag."""
        cell = StreamingCell()
        cell.add_actor(CellActor(id="a1", tags={"enemy", "boss"}))
        cell.add_actor(CellActor(id="a2", tags={"player"}))
        cell.add_actor(CellActor(id="a3", tags={"enemy"}))

        enemies = cell.get_actors_by_tag("enemy")
        assert len(enemies) == 2

    def test_cell_get_persistent_actors(self):
        """Test getting persistent actors."""
        cell = StreamingCell()
        cell.add_actor(CellActor(id="a1", persistent=True))
        cell.add_actor(CellActor(id="a2", persistent=False))
        cell.add_actor(CellActor(id="a3", persistent=True))

        persistent = cell.get_persistent_actors()
        assert len(persistent) == 2


# =============================================================================
# StreamingCell Spatial Tests
# =============================================================================

class TestStreamingCellSpatial:
    """Tests for cell spatial operations."""

    def test_cell_contains_point_inside(self):
        """Test point containment inside cell."""
        cell = StreamingCell(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(256, 256, 256),
        )
        assert cell.contains_point(Vec3(128, 128, 128)) is True

    def test_cell_contains_point_on_edge(self):
        """Test point containment on edge."""
        cell = StreamingCell(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(256, 256, 256),
        )
        assert cell.contains_point(Vec3(0, 0, 0)) is True
        assert cell.contains_point(Vec3(256, 256, 256)) is True

    def test_cell_contains_point_outside(self):
        """Test point containment outside cell."""
        cell = StreamingCell(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(256, 256, 256),
        )
        assert cell.contains_point(Vec3(-1, 128, 128)) is False
        assert cell.contains_point(Vec3(257, 128, 128)) is False

    def test_cell_overlaps_true(self):
        """Test overlapping bounds."""
        cell = StreamingCell(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(256, 256, 256),
        )
        assert cell.overlaps(Vec3(128, 128, 128), Vec3(300, 300, 300)) is True

    def test_cell_overlaps_false(self):
        """Test non-overlapping bounds."""
        cell = StreamingCell(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(256, 256, 256),
        )
        assert cell.overlaps(Vec3(300, 300, 300), Vec3(400, 400, 400)) is False

    def test_cell_distance_to_point(self):
        """Test distance to point calculation."""
        cell = StreamingCell(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(256, 256, 256),
        )
        # Center is at (128, 128, 128)
        distance = cell.distance_to_point(Vec3(128, 128, 128))
        assert distance == pytest.approx(0, abs=0.001)

    def test_cell_distance_squared_to_point(self):
        """Test squared distance to point."""
        cell = StreamingCell(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(256, 256, 256),
        )
        dist_sq = cell.distance_squared_to_point(Vec3(128 + 100, 128, 128))
        assert dist_sq == pytest.approx(10000, abs=0.1)


# =============================================================================
# StreamingCell Neighbor Tests
# =============================================================================

class TestStreamingCellNeighbors:
    """Tests for cell neighbor operations."""

    def test_cell_get_neighbors(self):
        """Test getting all neighbors (8-connected)."""
        cell = StreamingCell(coord=CellCoord(5, 5))
        neighbors = cell.get_neighbors()
        assert len(neighbors) == 8

    def test_cell_get_neighbors_contains_expected(self):
        """Test neighbor coordinates are correct."""
        cell = StreamingCell(coord=CellCoord(5, 5))
        neighbors = cell.get_neighbors()
        coords = {(n.x, n.y) for n in neighbors}
        assert (4, 4) in coords  # NW
        assert (5, 4) in coords  # N
        assert (6, 4) in coords  # NE
        assert (4, 5) in coords  # W
        assert (6, 5) in coords  # E
        assert (4, 6) in coords  # SW
        assert (5, 6) in coords  # S
        assert (6, 6) in coords  # SE

    def test_cell_get_cardinal_neighbors(self):
        """Test getting only cardinal neighbors."""
        cell = StreamingCell(coord=CellCoord(5, 5))
        neighbors = cell.get_cardinal_neighbors()
        assert len(neighbors) == 4

    def test_cell_get_cardinal_neighbors_coords(self):
        """Test cardinal neighbor coordinates."""
        cell = StreamingCell(coord=CellCoord(5, 5))
        neighbors = cell.get_cardinal_neighbors()
        coords = {(n.x, n.y) for n in neighbors}
        assert (5, 4) in coords  # N
        assert (6, 5) in coords  # E
        assert (5, 6) in coords  # S
        assert (4, 5) in coords  # W


# =============================================================================
# StreamingCell Callbacks Tests
# =============================================================================

class TestStreamingCellCallbacks:
    """Tests for cell callbacks."""

    def test_cell_on_load_callback(self):
        """Test load callback is invoked."""
        cell = StreamingCell()
        callback_called = []
        cell.on_load(lambda c: callback_called.append(c.coord))
        cell.load()
        cell.complete_load()
        assert len(callback_called) == 1

    def test_cell_on_unload_callback(self):
        """Test unload callback is invoked."""
        cell = StreamingCell()
        callback_called = []
        cell.on_unload(lambda c: callback_called.append(c.coord))
        cell.state = CellState.LOADED
        cell.unload()
        cell.complete_unload()
        assert len(callback_called) == 1

    def test_cell_on_activate_callback(self):
        """Test activate callback is invoked."""
        cell = StreamingCell()
        callback_called = []
        cell.on_activate(lambda c: callback_called.append(c.coord))
        cell.state = CellState.LOADED
        cell.activate()
        assert len(callback_called) == 1

    def test_cell_on_deactivate_callback(self):
        """Test deactivate callback is invoked."""
        cell = StreamingCell()
        callback_called = []
        cell.on_deactivate(lambda c: callback_called.append(c.coord))
        cell.state = CellState.ACTIVATED
        cell.deactivate()
        assert len(callback_called) == 1

    def test_cell_clear_callbacks(self):
        """Test clearing all callbacks."""
        cell = StreamingCell()
        cell.on_load(lambda c: None)
        cell.on_unload(lambda c: None)
        cell.on_activate(lambda c: None)
        cell.on_deactivate(lambda c: None)
        cell.clear_callbacks()
        # No way to directly verify, but should not raise on state changes
        cell.load()
        cell.complete_load()
        cell.activate()
        cell.deactivate()


# =============================================================================
# StreamingCell Serialization Tests
# =============================================================================

class TestStreamingCellSerialization:
    """Tests for cell serialization."""

    def test_cell_serialize(self):
        """Test cell serialization."""
        cell = StreamingCell(
            coord=CellCoord(5, 10),
            bounds_min=Vec3(1280, 0, 2560),
            bounds_max=Vec3(1536, 256, 2816),
            priority=5,
        )
        cell.add_actor(CellActor(id="a1", name="Actor1", position=Vec3(100, 50, 200)))

        data = cell.serialize()
        assert data["coord"] == [5, 10]
        assert data["priority"] == 5
        assert len(data["actors"]) == 1

    def test_cell_deserialize(self):
        """Test cell deserialization."""
        data = {
            "coord": [5, 10],
            "bounds_min": [1280, 0, 2560],
            "bounds_max": [1536, 256, 2816],
            "priority": 5,
            "actors": [
                {
                    "id": "a1",
                    "name": "Actor1",
                    "position": [100, 50, 200],
                    "priority": 10,
                    "persistent": True,
                    "tags": ["enemy"],
                }
            ],
            "metadata": {"custom": "value"},
        }
        cell = StreamingCell.deserialize(data)
        assert cell.coord.x == 5
        assert cell.coord.y == 10
        assert cell.priority == 5
        assert len(cell.actors) == 1
        assert cell.actors[0].persistent is True

    def test_cell_get_memory_estimate(self):
        """Test memory estimate calculation."""
        cell = StreamingCell()
        cell.add_actor(CellActor(id="a1"))
        cell.add_actor(CellActor(id="a2"))
        estimate = cell.get_memory_estimate()
        assert estimate > 0

    def test_cell_repr(self):
        """Test string representation."""
        cell = StreamingCell(coord=CellCoord(5, 10))
        cell.state = CellState.LOADED
        cell.add_actor(CellActor(id="a1"))
        repr_str = repr(cell)
        assert "5" in repr_str
        assert "10" in repr_str
        assert "LOADED" in repr_str
