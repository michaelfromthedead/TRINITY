"""
Tests for the Level Architecture module.

Tests Level, StreamingLevel, SubLevel, LevelInstance, and WorldComposition classes.
"""

import pytest
from engine.core.math.vec import Vec3
from engine.world.level import (
    Actor,
    Level,
    LevelBounds,
    LevelInstance,
    LevelLayer,
    LevelState,
    LevelType,
    StreamingLevel,
    SubLevel,
    WorldComposition,
)


# =============================================================================
# LevelBounds Tests
# =============================================================================

class TestLevelBounds:
    """Tests for LevelBounds class."""

    def test_bounds_creation_default(self):
        """Test default bounds creation."""
        bounds = LevelBounds()
        assert bounds.min_point == Vec3(0, 0, 0)
        assert bounds.max_point == Vec3(0, 0, 0)

    def test_bounds_creation_custom(self):
        """Test custom bounds creation."""
        bounds = LevelBounds(
            min_point=Vec3(-100, -50, -100),
            max_point=Vec3(100, 50, 100),
        )
        assert bounds.min_point.x == -100
        assert bounds.max_point.z == 100

    def test_bounds_center(self):
        """Test bounds center calculation."""
        bounds = LevelBounds(
            min_point=Vec3(0, 0, 0),
            max_point=Vec3(100, 100, 100),
        )
        center = bounds.center
        assert center.x == 50
        assert center.y == 50
        assert center.z == 50

    def test_bounds_extents(self):
        """Test bounds extents calculation."""
        bounds = LevelBounds(
            min_point=Vec3(0, 0, 0),
            max_point=Vec3(200, 100, 50),
        )
        extents = bounds.extents
        assert extents.x == 100
        assert extents.y == 50
        assert extents.z == 25

    def test_bounds_size(self):
        """Test bounds size calculation."""
        bounds = LevelBounds(
            min_point=Vec3(-50, -25, -10),
            max_point=Vec3(50, 25, 10),
        )
        size = bounds.size
        assert size.x == 100
        assert size.y == 50
        assert size.z == 20

    def test_bounds_contains_point_inside(self):
        """Test point containment for inside point."""
        bounds = LevelBounds(
            min_point=Vec3(0, 0, 0),
            max_point=Vec3(100, 100, 100),
        )
        assert bounds.contains_point(Vec3(50, 50, 50)) is True

    def test_bounds_contains_point_on_edge(self):
        """Test point containment for edge point."""
        bounds = LevelBounds(
            min_point=Vec3(0, 0, 0),
            max_point=Vec3(100, 100, 100),
        )
        assert bounds.contains_point(Vec3(0, 0, 0)) is True
        assert bounds.contains_point(Vec3(100, 100, 100)) is True

    def test_bounds_contains_point_outside(self):
        """Test point containment for outside point."""
        bounds = LevelBounds(
            min_point=Vec3(0, 0, 0),
            max_point=Vec3(100, 100, 100),
        )
        assert bounds.contains_point(Vec3(-1, 50, 50)) is False
        assert bounds.contains_point(Vec3(101, 50, 50)) is False

    def test_bounds_intersects_overlapping(self):
        """Test intersection with overlapping bounds."""
        bounds1 = LevelBounds(Vec3(0, 0, 0), Vec3(100, 100, 100))
        bounds2 = LevelBounds(Vec3(50, 50, 50), Vec3(150, 150, 150))
        assert bounds1.intersects(bounds2) is True

    def test_bounds_intersects_non_overlapping(self):
        """Test intersection with non-overlapping bounds."""
        bounds1 = LevelBounds(Vec3(0, 0, 0), Vec3(100, 100, 100))
        bounds2 = LevelBounds(Vec3(200, 200, 200), Vec3(300, 300, 300))
        assert bounds1.intersects(bounds2) is False

    def test_bounds_expand(self):
        """Test bounds expansion with a point."""
        bounds = LevelBounds(Vec3(0, 0, 0), Vec3(100, 100, 100))
        bounds.expand(Vec3(150, -50, 200))
        assert bounds.min_point.y == -50
        assert bounds.max_point.x == 150
        assert bounds.max_point.z == 200

    def test_bounds_merge(self):
        """Test merging two bounds."""
        bounds1 = LevelBounds(Vec3(0, 0, 0), Vec3(100, 100, 100))
        bounds2 = LevelBounds(Vec3(-50, 50, -25), Vec3(50, 200, 75))
        merged = bounds1.merge(bounds2)
        assert merged.min_point.x == -50
        assert merged.max_point.y == 200


# =============================================================================
# Actor Tests
# =============================================================================

class TestActor:
    """Tests for Actor class."""

    def test_actor_creation_default(self):
        """Test default actor creation."""
        actor = Actor()
        assert actor.id == ""
        assert actor.name == ""
        assert actor.layer == LevelLayer.GAMEPLAY

    def test_actor_creation_custom(self):
        """Test custom actor creation."""
        actor = Actor(
            id="actor_001",
            name="Test Actor",
            transform_position=Vec3(10, 20, 30),
            layer=LevelLayer.LIGHTING,
            tags={"enemy", "spawnable"},
        )
        assert actor.id == "actor_001"
        assert actor.name == "Test Actor"
        assert actor.transform_position.x == 10
        assert LevelLayer.LIGHTING == actor.layer
        assert "enemy" in actor.tags

    def test_actor_position_getter(self):
        """Test actor position getter."""
        actor = Actor(transform_position=Vec3(100, 200, 300))
        pos = actor.get_position()
        assert pos.x == 100
        assert pos.y == 200
        assert pos.z == 300

    def test_actor_position_setter(self):
        """Test actor position setter."""
        actor = Actor()
        actor.set_position(Vec3(50, 60, 70))
        assert actor.transform_position.x == 50
        assert actor.transform_position.y == 60


# =============================================================================
# Level Tests
# =============================================================================

class TestLevel:
    """Tests for Level class."""

    def test_level_creation_default(self):
        """Test default level creation."""
        level = Level()
        assert level.name == ""
        assert level.level_type == LevelType.PERSISTENT
        assert level.state == LevelState.UNLOADED

    def test_level_creation_custom(self):
        """Test custom level creation."""
        level = Level(
            name="TestLevel",
            level_type=LevelType.STREAMING,
        )
        assert level.name == "TestLevel"
        assert level.level_type == LevelType.STREAMING

    def test_level_add_actor(self):
        """Test adding actors to level."""
        level = Level(name="TestLevel")
        actor = Actor(id="a1", name="Actor1", transform_position=Vec3(10, 0, 10))
        level.add_actor(actor)
        assert len(level.actors) == 1
        assert level.actors[0].id == "a1"

    def test_level_remove_actor(self):
        """Test removing actors from level."""
        level = Level()
        actor = Actor(id="a1")
        level.add_actor(actor)
        result = level.remove_actor(actor)
        assert result is True
        assert len(level.actors) == 0

    def test_level_remove_actor_not_found(self):
        """Test removing non-existent actor."""
        level = Level()
        actor = Actor(id="a1")
        result = level.remove_actor(actor)
        assert result is False

    def test_level_get_actor_by_id(self):
        """Test finding actor by ID."""
        level = Level()
        actor = Actor(id="unique_id", name="Test")
        level.add_actor(actor)
        found = level.get_actor_by_id("unique_id")
        assert found is not None
        assert found.name == "Test"

    def test_level_get_actor_by_id_not_found(self):
        """Test finding non-existent actor by ID."""
        level = Level()
        found = level.get_actor_by_id("nonexistent")
        assert found is None

    def test_level_get_actors_by_layer(self):
        """Test filtering actors by layer."""
        level = Level()
        level.add_actor(Actor(id="a1", layer=LevelLayer.GAMEPLAY))
        level.add_actor(Actor(id="a2", layer=LevelLayer.LIGHTING))
        level.add_actor(Actor(id="a3", layer=LevelLayer.GAMEPLAY))

        gameplay_actors = level.get_actors_by_layer(LevelLayer.GAMEPLAY)
        assert len(gameplay_actors) == 2

    def test_level_get_actors_by_tag(self):
        """Test filtering actors by tag."""
        level = Level()
        level.add_actor(Actor(id="a1", tags={"enemy", "boss"}))
        level.add_actor(Actor(id="a2", tags={"player"}))
        level.add_actor(Actor(id="a3", tags={"enemy"}))

        enemies = level.get_actors_by_tag("enemy")
        assert len(enemies) == 2

    def test_level_get_actors_in_bounds(self):
        """Test filtering actors by bounds."""
        level = Level()
        level.add_actor(Actor(id="a1", transform_position=Vec3(50, 50, 50)))
        level.add_actor(Actor(id="a2", transform_position=Vec3(200, 50, 50)))
        level.add_actor(Actor(id="a3", transform_position=Vec3(25, 75, 25)))

        search_bounds = LevelBounds(Vec3(0, 0, 0), Vec3(100, 100, 100))
        found = level.get_actors_in_bounds(search_bounds)
        assert len(found) == 2

    def test_level_layer_enabled_default(self):
        """Test default layer enabled state."""
        level = Level()
        assert level.is_layer_enabled(LevelLayer.GAMEPLAY) is True

    def test_level_set_layer_enabled(self):
        """Test setting layer enabled state."""
        level = Level()
        level.set_layer_enabled(LevelLayer.AUDIO, False)
        assert level.is_layer_enabled(LevelLayer.AUDIO) is False

    def test_level_load(self):
        """Test level loading."""
        level = Level()
        result = level.load()
        assert result is True
        assert level.state == LevelState.LOADED

    def test_level_load_already_loaded(self):
        """Test loading already loaded level."""
        level = Level()
        level.load()
        result = level.load()
        assert result is False

    def test_level_unload(self):
        """Test level unloading."""
        level = Level()
        level.load()
        result = level.unload()
        assert result is True
        assert level.state == LevelState.UNLOADED

    def test_level_unload_not_loaded(self):
        """Test unloading non-loaded level."""
        level = Level()
        result = level.unload()
        assert result is False

    def test_level_on_load_callback(self):
        """Test load callback invocation."""
        level = Level()
        callback_called = []
        level.on_load(lambda l: callback_called.append(l.name))
        level.name = "TestLevel"
        level.load()
        assert "TestLevel" in callback_called

    def test_level_on_unload_callback(self):
        """Test unload callback invocation."""
        level = Level()
        callback_called = []
        level.on_unload(lambda l: callback_called.append(l.name))
        level.name = "TestLevel"
        level.load()
        level.unload()
        assert "TestLevel" in callback_called

    def test_level_serialize(self):
        """Test level serialization."""
        level = Level(name="TestLevel", level_type=LevelType.PERSISTENT)
        level.add_actor(Actor(id="a1", name="Actor1", transform_position=Vec3(10, 20, 30)))

        data = level.serialize()
        assert data["name"] == "TestLevel"
        assert data["level_type"] == "PERSISTENT"
        assert len(data["actors"]) == 1

    def test_level_deserialize(self):
        """Test level deserialization."""
        data = {
            "name": "TestLevel",
            "level_type": "STREAMING",
            "bounds": {"min": [0, 0, 0], "max": [100, 100, 100]},
            "actors": [
                {
                    "id": "a1",
                    "name": "Actor1",
                    "position": [10, 20, 30],
                    "rotation": [0, 0, 0, 1],
                    "scale": [1, 1, 1],
                    "layer": "GAMEPLAY",
                    "tags": ["test"],
                    "persistent": False,
                }
            ],
            "layers_enabled": {},
        }
        level = Level.deserialize(data)
        assert level.name == "TestLevel"
        assert level.level_type == LevelType.STREAMING
        assert len(level.actors) == 1

    def test_level_recalculate_bounds(self):
        """Test bounds recalculation."""
        level = Level()
        level.add_actor(Actor(transform_position=Vec3(10, 10, 10)))
        level.add_actor(Actor(transform_position=Vec3(100, 50, 80)))
        level.recalculate_bounds()
        assert level.bounds.min_point.x == 10
        assert level.bounds.max_point.x == 100

    def test_level_recalculate_bounds_empty(self):
        """Test bounds recalculation with no actors."""
        level = Level()
        level.recalculate_bounds()
        assert level.bounds.min_point == Vec3(0, 0, 0)


# =============================================================================
# StreamingLevel Tests
# =============================================================================

class TestStreamingLevel:
    """Tests for StreamingLevel class."""

    def test_streaming_level_creation(self):
        """Test streaming level creation."""
        level = StreamingLevel(
            name="StreamingLevel",
            load_distance=3000.0,
            unload_distance=4000.0,
        )
        assert level.level_type == LevelType.STREAMING
        assert level.load_distance == 3000.0
        assert level.unload_distance == 4000.0

    def test_streaming_level_hysteresis_enforcement(self):
        """Test hysteresis ensures unload > load distance."""
        level = StreamingLevel(
            load_distance=5000.0,
            unload_distance=4000.0,  # Less than load
            hysteresis=1000.0,
        )
        assert level.unload_distance > level.load_distance

    def test_streaming_level_should_load_in_range(self):
        """Test should_load returns True when in range."""
        level = StreamingLevel(
            load_distance=5000.0,
            reference_point=Vec3(0, 0, 0),
        )
        result = level.should_load(Vec3(1000, 0, 0))
        assert result is True

    def test_streaming_level_should_load_already_loaded(self):
        """Test should_load returns False when already loaded."""
        level = StreamingLevel(load_distance=5000.0)
        level.state = LevelState.LOADED
        result = level.should_load(Vec3(0, 0, 0))
        assert result is False

    def test_streaming_level_should_load_out_of_range(self):
        """Test should_load returns False when out of range."""
        level = StreamingLevel(
            load_distance=5000.0,
            reference_point=Vec3(0, 0, 0),
        )
        result = level.should_load(Vec3(10000, 0, 0))
        assert result is False

    def test_streaming_level_should_unload_in_range(self):
        """Test should_unload returns False when still in range."""
        level = StreamingLevel(
            unload_distance=6000.0,
            reference_point=Vec3(0, 0, 0),
        )
        level.state = LevelState.LOADED
        result = level.should_unload(Vec3(1000, 0, 0))
        assert result is False

    def test_streaming_level_should_unload_out_of_range(self):
        """Test should_unload returns True when out of range."""
        level = StreamingLevel(
            unload_distance=6000.0,
            reference_point=Vec3(0, 0, 0),
        )
        level.state = LevelState.LOADED
        result = level.should_unload(Vec3(10000, 0, 0))
        assert result is True

    def test_streaming_level_update_load_progress(self):
        """Test load progress update."""
        level = StreamingLevel()
        level.state = LevelState.LOADING
        level.update_load_progress(0.5)
        assert level.load_progress == 0.5

    def test_streaming_level_progress_clamp(self):
        """Test load progress clamping."""
        level = StreamingLevel()
        level.update_load_progress(1.5)
        assert level.load_progress == 1.0
        level.update_load_progress(-0.5)
        assert level.load_progress == 0.0

    def test_streaming_level_auto_complete_on_full_progress(self):
        """Test automatic state change on full progress."""
        level = StreamingLevel()
        level.state = LevelState.LOADING
        level.update_load_progress(1.0)
        assert level.state == LevelState.LOADED


# =============================================================================
# SubLevel Tests
# =============================================================================

class TestSubLevel:
    """Tests for SubLevel class."""

    def test_sublevel_creation(self):
        """Test sub-level creation."""
        sub = SubLevel(name="SubLevel1")
        assert sub.level_type == LevelType.SUB_LEVEL
        assert sub.parent is None

    def test_sublevel_set_parent(self):
        """Test setting parent level."""
        parent = Level(name="Parent")
        sub = SubLevel(name="Child")
        sub.set_parent(parent)
        assert sub.parent is parent

    def test_sublevel_get_world_position(self):
        """Test local to world position conversion."""
        sub = SubLevel(relative_offset=Vec3(100, 50, 200))
        world_pos = sub.get_world_position(Vec3(10, 20, 30))
        assert world_pos.x == 110
        assert world_pos.y == 70
        assert world_pos.z == 230

    def test_sublevel_get_local_position(self):
        """Test world to local position conversion."""
        sub = SubLevel(relative_offset=Vec3(100, 50, 200))
        local_pos = sub.get_local_position(Vec3(110, 70, 230))
        assert local_pos.x == 10
        assert local_pos.y == 20
        assert local_pos.z == 30

    def test_sublevel_inherit_layers(self):
        """Test layer inheritance from parent."""
        parent = Level()
        parent.set_layer_enabled(LevelLayer.AUDIO, False)
        sub = SubLevel(inherit_layers=True)
        sub.set_parent(parent)
        sub.load()
        # Should inherit parent's disabled audio layer
        assert sub.is_layer_enabled(LevelLayer.AUDIO) is False


# =============================================================================
# LevelInstance Tests
# =============================================================================

class TestLevelInstance:
    """Tests for LevelInstance class."""

    def test_level_instance_creation(self):
        """Test level instance creation."""
        instance = LevelInstance(name="Instance1")
        assert instance.level_type == LevelType.INSTANCE
        assert instance.instance_id != ""  # Auto-generated

    def test_level_instance_from_source(self):
        """Test creating instance from source level."""
        source = Level(name="SourceLevel")
        source.add_actor(Actor(id="a1", name="Actor1"))

        instance = LevelInstance.from_source(source, "inst_001")
        assert instance.source_level is source
        assert instance.instance_id == "inst_001"

    def test_level_instance_get_actors(self):
        """Test getting actors from instance."""
        source = Level(name="Source")
        source.add_actor(Actor(id="a1"))
        source.add_actor(Actor(id="a2"))

        instance = LevelInstance.from_source(source)
        actors = instance.get_actors()
        assert len(actors) == 2

    def test_level_instance_spawn_actor(self):
        """Test spawning instance-specific actor."""
        source = Level(name="Source")
        instance = LevelInstance.from_source(source)

        instance.spawn_actor(Actor(id="new_actor"))
        assert len(instance.spawned_actors) == 1

    def test_level_instance_destroy_actor(self):
        """Test destroying actor in instance."""
        source = Level(name="Source")
        source.add_actor(Actor(id="a1"))

        instance = LevelInstance.from_source(source)
        instance.destroy_actor("a1")

        actors = instance.get_actors()
        assert len(actors) == 0

    def test_level_instance_reset(self):
        """Test resetting instance to source state."""
        source = Level(name="Source")
        source.add_actor(Actor(id="a1"))

        instance = LevelInstance.from_source(source)
        instance.spawn_actor(Actor(id="new"))
        instance.destroy_actor("a1")
        instance.set_state("key", "value")

        instance.reset_instance()

        assert len(instance.spawned_actors) == 0
        assert len(instance.destroyed_actors) == 0
        assert instance.get_state("key") is None

    def test_level_instance_runtime_state(self):
        """Test runtime state management."""
        instance = LevelInstance()
        instance.set_state("score", 100)
        assert instance.get_state("score") == 100
        assert instance.get_state("nonexistent", "default") == "default"


# =============================================================================
# WorldComposition Tests
# =============================================================================

class TestWorldComposition:
    """Tests for WorldComposition class."""

    def test_world_composition_creation(self):
        """Test world composition creation."""
        world = WorldComposition(name="TestWorld")
        assert world.name == "TestWorld"
        assert len(world.levels) == 0

    def test_world_add_level(self):
        """Test adding levels to world."""
        world = WorldComposition()
        level = Level(name="Level1")
        world.add_level(level)
        assert len(world.levels) == 1

    def test_world_add_streaming_level(self):
        """Test adding streaming level."""
        world = WorldComposition()
        level = StreamingLevel(name="StreamingLevel1")
        world.add_level(level)
        assert len(world.streaming_levels) == 1

    def test_world_add_sublevel(self):
        """Test adding sub-level."""
        world = WorldComposition()
        level = SubLevel(name="SubLevel1")
        world.add_level(level)
        assert len(world.sub_levels) == 1

    def test_world_remove_level(self):
        """Test removing level from world."""
        world = WorldComposition()
        level = Level(name="Level1")
        world.add_level(level)
        result = world.remove_level(level)
        assert result is True
        assert len(world.levels) == 0

    def test_world_get_level(self):
        """Test getting level by name."""
        world = WorldComposition()
        level = Level(name="TestLevel")
        world.add_level(level)
        found = world.get_level("TestLevel")
        assert found is level

    def test_world_get_levels_at_position(self):
        """Test getting levels containing a position."""
        world = WorldComposition()
        level1 = Level(name="L1")
        level1.bounds = LevelBounds(Vec3(0, 0, 0), Vec3(100, 100, 100))
        level2 = Level(name="L2")
        level2.bounds = LevelBounds(Vec3(50, 50, 50), Vec3(150, 150, 150))
        world.add_level(level1)
        world.add_level(level2)

        found = world.get_levels_at_position(Vec3(75, 75, 75))
        assert len(found) == 2

    def test_world_get_loaded_levels(self):
        """Test getting loaded levels."""
        world = WorldComposition()
        level1 = Level(name="L1")
        level1.load()
        level2 = Level(name="L2")
        world.add_level(level1)
        world.add_level(level2)

        loaded = world.get_loaded_levels()
        assert len(loaded) == 1
        assert loaded[0].name == "L1"

    def test_world_update_streaming(self):
        """Test streaming update."""
        world = WorldComposition()
        level = StreamingLevel(
            name="SL1",
            load_distance=5000.0,
            reference_point=Vec3(0, 0, 0),
        )
        world.add_level(level)

        to_load, to_unload = world.update_streaming(Vec3(1000, 0, 0))
        assert len(to_load) == 1
        assert len(to_unload) == 0

    def test_world_origin_rebase_needed(self):
        """Test origin rebase check."""
        world = WorldComposition(origin_shift_threshold=10000.0)
        assert world.check_origin_rebase_needed(Vec3(5000, 0, 0)) is False
        assert world.check_origin_rebase_needed(Vec3(15000, 0, 0)) is True

    def test_world_perform_origin_rebase(self):
        """Test performing origin rebase."""
        world = WorldComposition()
        level = Level()
        level.load()
        actor = Actor(transform_position=Vec3(1000, 500, 2000))
        level.add_actor(actor)
        world.add_level(level)

        delta = world.perform_origin_rebase(Vec3(1000, 0, 0))

        assert delta.x == 1000
        assert actor.transform_position.x == 0  # Shifted by delta

    def test_world_get_tile_at_position(self):
        """Test tile coordinate calculation."""
        world = WorldComposition(tile_size=2048.0)
        tile = world.get_tile_at_position(Vec3(3000, 0, 5000))
        assert tile[0] == 1  # 3000 // 2048
        assert tile[1] == 2  # 5000 // 2048

    def test_world_get_tile_bounds(self):
        """Test tile bounds calculation."""
        world = WorldComposition(tile_size=1024.0)
        bounds = world.get_tile_bounds(2, 3)
        assert bounds.min_point.x == 2048  # 2 * 1024
        assert bounds.min_point.z == 3072  # 3 * 1024

    def test_world_get_surrounding_tiles(self):
        """Test getting surrounding tiles."""
        world = WorldComposition()
        tiles = world.get_surrounding_tiles((5, 5), radius=1)
        assert len(tiles) == 9  # 3x3 grid

    def test_world_get_world_bounds(self):
        """Test combined world bounds."""
        world = WorldComposition()
        level1 = Level()
        level1.bounds = LevelBounds(Vec3(0, 0, 0), Vec3(100, 100, 100))
        level2 = Level()
        level2.bounds = LevelBounds(Vec3(50, 50, 50), Vec3(200, 200, 200))
        world.add_level(level1)
        world.add_level(level2)

        bounds = world.get_world_bounds()
        assert bounds.min_point.x == 0
        assert bounds.max_point.x == 200

    def test_world_iteration(self):
        """Test iterating over world levels."""
        world = WorldComposition()
        world.add_level(Level(name="L1"))
        world.add_level(Level(name="L2"))
        world.add_level(Level(name="L3"))

        names = [l.name for l in world]
        assert len(names) == 3

    def test_world_len(self):
        """Test world level count."""
        world = WorldComposition()
        world.add_level(Level(name="L1"))
        world.add_level(Level(name="L2"))
        assert len(world) == 2
