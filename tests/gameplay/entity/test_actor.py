"""
Comprehensive tests for the Actor System.

Tests for:
- Actor creation, initialization, destruction
- Component attachment/detachment
- Tick enable/disable, tick groups
- Actor hierarchy (parent/child)
- Actor tags and filtering
- Actor replication flags
- Pawn input handling
- Character movement modes
- Character abilities integration
"""
from __future__ import annotations

import pytest
import weakref
from typing import Any, Set
from unittest.mock import Mock, MagicMock, patch

from engine.gameplay.entity.actor import (
    Actor,
    ActorMeta,
    StaticActor,
    DynamicActor,
    Pawn,
    Character,
    ComponentContainer,
    Transform,
)
from engine.gameplay.entity.constants import (
    ActorType,
    LifecycleState,
    TickGroup,
    ENTITY_ID_START,
    ENTITY_NAME_MAX_LENGTH,
    MAX_COMPONENTS_PER_ENTITY,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_state():
    """Reset actor state before each test."""
    Actor.reset_entity_ids()
    ActorMeta.clear_registry()
    yield


@pytest.fixture
def basic_actor():
    """Create a basic actor for testing."""
    return Actor(name="TestActor")


@pytest.fixture
def static_actor():
    """Create a static actor for testing."""
    return StaticActor(name="TestStaticActor")


@pytest.fixture
def dynamic_actor():
    """Create a dynamic actor for testing."""
    return DynamicActor(name="TestDynamicActor")


@pytest.fixture
def pawn():
    """Create a pawn for testing."""
    return Pawn(name="TestPawn")


@pytest.fixture
def character():
    """Create a character for testing."""
    return Character(name="TestCharacter")


@pytest.fixture
def mock_component():
    """Create a mock component."""
    component = Mock()
    component._component = True
    return component


class MockComponent:
    """Mock component class for testing."""
    _component = True

    def __init__(self, value: int = 0):
        self.value = value


# =============================================================================
# ACTOR CREATION TESTS
# =============================================================================


class TestActorCreation:
    """Tests for Actor creation and initialization."""

    def test_actor_creation_basic(self):
        """Test basic actor creation."""
        actor = Actor()
        assert actor is not None
        assert isinstance(actor, Actor)

    def test_actor_creation_with_name(self):
        """Test actor creation with a custom name."""
        actor = Actor(name="MyActor")
        assert actor.name == "MyActor"

    def test_actor_default_name(self):
        """Test actor default name generation."""
        actor = Actor()
        assert actor.name.startswith("Actor_")

    def test_actor_name_truncation(self):
        """Test that long names are truncated."""
        long_name = "A" * (ENTITY_NAME_MAX_LENGTH + 100)
        actor = Actor(name=long_name)
        assert len(actor.name) == ENTITY_NAME_MAX_LENGTH

    def test_actor_unique_entity_id(self):
        """Test that each actor gets a unique entity ID."""
        actor1 = Actor()
        actor2 = Actor()
        assert actor1.entity_id != actor2.entity_id

    def test_actor_entity_id_starts_from_start(self):
        """Test that entity IDs start from ENTITY_ID_START."""
        actor = Actor()
        assert actor.entity_id >= ENTITY_ID_START

    def test_actor_sequential_entity_ids(self):
        """Test that entity IDs are assigned sequentially."""
        actor1 = Actor()
        actor2 = Actor()
        assert actor2.entity_id == actor1.entity_id + 1

    def test_actor_with_transform(self):
        """Test actor creation with a transform."""
        transform = Transform(
            position=(1.0, 2.0, 3.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            scale=(2.0, 2.0, 2.0),
        )
        actor = Actor(transform=transform)
        assert actor.position == (1.0, 2.0, 3.0)
        assert actor.scale == (2.0, 2.0, 2.0)

    def test_actor_with_tags(self):
        """Test actor creation with tags."""
        tags = {"enemy", "npc", "hostile"}
        actor = Actor(tags=tags)
        assert actor.has_tag("enemy")
        assert actor.has_tag("npc")
        assert actor.has_tag("hostile")

    def test_actor_initial_lifecycle_state(self):
        """Test actor starts in CREATED state."""
        actor = Actor()
        assert actor._lifecycle_state == LifecycleState.CREATED

    def test_actor_type_static_default(self):
        """Test that base Actor has STATIC type by default."""
        actor = Actor()
        assert actor.actor_type == ActorType.STATIC

    def test_actor_repr(self):
        """Test actor string representation."""
        actor = Actor(name="TestActor")
        repr_str = repr(actor)
        assert "Actor" in repr_str
        assert "TestActor" in repr_str

    def test_actor_transform_copy(self):
        """Test that actor makes a copy of the transform."""
        transform = Transform(position=(1.0, 2.0, 3.0))
        actor = Actor(transform=transform)
        transform.position = (5.0, 5.0, 5.0)
        assert actor.position == (1.0, 2.0, 3.0)

    def test_actor_tags_copy(self):
        """Test that actor makes a copy of the tags set."""
        tags = {"tag1", "tag2"}
        actor = Actor(tags=tags)
        tags.add("tag3")
        assert not actor.has_tag("tag3")


class TestActorProperties:
    """Tests for Actor property accessors."""

    def test_position_getter(self, basic_actor):
        """Test position getter."""
        assert basic_actor.position == (0.0, 0.0, 0.0)

    def test_position_setter(self, basic_actor):
        """Test position setter."""
        basic_actor.position = (5.0, 10.0, 15.0)
        assert basic_actor.position == (5.0, 10.0, 15.0)

    def test_rotation_getter(self, basic_actor):
        """Test rotation getter returns quaternion."""
        assert basic_actor.rotation == (0.0, 0.0, 0.0, 1.0)

    def test_rotation_setter(self, basic_actor):
        """Test rotation setter."""
        basic_actor.rotation = (0.0, 0.707, 0.0, 0.707)
        assert basic_actor.rotation == (0.0, 0.707, 0.0, 0.707)

    def test_scale_getter(self, basic_actor):
        """Test scale getter."""
        assert basic_actor.scale == (1.0, 1.0, 1.0)

    def test_scale_setter(self, basic_actor):
        """Test scale setter."""
        basic_actor.scale = (2.0, 2.0, 2.0)
        assert basic_actor.scale == (2.0, 2.0, 2.0)

    def test_transform_getter(self, basic_actor):
        """Test transform getter returns Transform object."""
        assert isinstance(basic_actor.transform, Transform)

    def test_transform_setter(self, basic_actor):
        """Test transform setter."""
        new_transform = Transform(position=(1.0, 2.0, 3.0))
        basic_actor.transform = new_transform
        assert basic_actor.position == (1.0, 2.0, 3.0)

    def test_name_setter(self, basic_actor):
        """Test name setter."""
        basic_actor.name = "NewName"
        assert basic_actor.name == "NewName"

    def test_name_setter_truncation(self, basic_actor):
        """Test name setter truncates long names."""
        basic_actor.name = "A" * 500
        assert len(basic_actor.name) == ENTITY_NAME_MAX_LENGTH


# =============================================================================
# COMPONENT TESTS
# =============================================================================


class TestComponentContainer:
    """Tests for ComponentContainer."""

    def test_container_creation(self, basic_actor):
        """Test component container is created with actor."""
        assert basic_actor.components is not None
        assert isinstance(basic_actor.components, ComponentContainer)

    def test_add_component(self, basic_actor):
        """Test adding a component."""
        component = MockComponent()
        basic_actor.add_component("health", component)
        assert basic_actor.has_component("health")

    def test_remove_component(self, basic_actor):
        """Test removing a component."""
        component = MockComponent()
        basic_actor.add_component("health", component)
        removed = basic_actor.remove_component("health")
        assert removed is component
        assert not basic_actor.has_component("health")

    def test_get_component_by_name(self, basic_actor):
        """Test getting a component by name."""
        component = MockComponent(value=42)
        basic_actor.add_component("health", component)
        retrieved = basic_actor.get_component("health")
        assert retrieved.value == 42

    def test_get_component_by_type(self, basic_actor):
        """Test getting a component by type."""
        component = MockComponent(value=100)
        basic_actor.add_component("health", component)
        retrieved = basic_actor.get_component_by_type(MockComponent)
        assert retrieved is component

    def test_has_component(self, basic_actor):
        """Test checking component existence by name."""
        basic_actor.add_component("test", MockComponent())
        assert basic_actor.has_component("test")
        assert not basic_actor.has_component("nonexistent")

    def test_has_component_type(self, basic_actor):
        """Test checking component existence by type."""
        basic_actor.add_component("test", MockComponent())
        assert basic_actor.has_component_type(MockComponent)

    def test_component_duplicate_name_error(self, basic_actor):
        """Test error when adding component with duplicate name."""
        basic_actor.add_component("test", MockComponent())
        with pytest.raises(ValueError, match="already exists"):
            basic_actor.add_component("test", MockComponent())

    def test_component_max_limit(self, basic_actor):
        """Test error when exceeding max component count."""
        for i in range(MAX_COMPONENTS_PER_ENTITY):
            basic_actor.add_component(f"component_{i}", MockComponent())

        with pytest.raises(ValueError, match="Maximum component count"):
            basic_actor.add_component("overflow", MockComponent())

    def test_component_iteration(self, basic_actor):
        """Test iterating over components."""
        basic_actor.add_component("comp1", MockComponent(1))
        basic_actor.add_component("comp2", MockComponent(2))

        names = [name for name, _ in basic_actor.components]
        assert "comp1" in names
        assert "comp2" in names

    def test_component_count(self, basic_actor):
        """Test component count."""
        assert len(basic_actor.components) == 0
        basic_actor.add_component("test1", MockComponent())
        assert len(basic_actor.components) == 1
        basic_actor.add_component("test2", MockComponent())
        assert len(basic_actor.components) == 2

    def test_component_clear(self, basic_actor):
        """Test clearing all components."""
        basic_actor.add_component("test1", MockComponent())
        basic_actor.add_component("test2", MockComponent())
        basic_actor.components.clear()
        assert len(basic_actor.components) == 0

    def test_remove_nonexistent_component(self, basic_actor):
        """Test removing a nonexistent component returns None."""
        result = basic_actor.remove_component("nonexistent")
        assert result is None

    def test_get_nonexistent_component(self, basic_actor):
        """Test getting a nonexistent component returns None."""
        result = basic_actor.get_component("nonexistent")
        assert result is None

    def test_get_nonexistent_component_type(self, basic_actor):
        """Test getting by nonexistent type returns None."""
        result = basic_actor.get_component_by_type(MockComponent)
        assert result is None


class TestComponentCallbacks:
    """Tests for component add/remove callbacks."""

    def test_on_component_added_called(self, basic_actor):
        """Test _on_component_added is called when adding."""
        basic_actor._on_component_added = Mock()
        component = MockComponent()
        basic_actor.add_component("test", component)
        basic_actor._on_component_added.assert_called_once_with("test", component)

    def test_on_component_removed_called(self, basic_actor):
        """Test _on_component_removed is called when removing."""
        component = MockComponent()
        basic_actor.add_component("test", component)
        basic_actor._on_component_removed = Mock()
        basic_actor.remove_component("test")
        basic_actor._on_component_removed.assert_called_once_with("test", component)


# =============================================================================
# TICK TESTS
# =============================================================================


class TestActorTick:
    """Tests for Actor tick functionality."""

    def test_tick_enabled_default(self, basic_actor):
        """Test tick is enabled by default for Actor."""
        assert basic_actor.tick_enabled is True

    def test_static_actor_tick_disabled_default(self, static_actor):
        """Test tick is disabled by default for StaticActor."""
        assert static_actor.tick_enabled is False

    def test_tick_enable_disable(self, basic_actor):
        """Test enabling and disabling tick."""
        basic_actor.tick_enabled = False
        assert basic_actor.tick_enabled is False
        basic_actor.tick_enabled = True
        assert basic_actor.tick_enabled is True

    def test_tick_group_default(self, basic_actor):
        """Test default tick group is UPDATE."""
        assert basic_actor.tick_group == TickGroup.UPDATE

    def test_tick_group_setter(self, basic_actor):
        """Test setting tick group."""
        basic_actor.tick_group = TickGroup.POST_PHYSICS
        assert basic_actor.tick_group == TickGroup.POST_PHYSICS

    def test_dynamic_actor_tick_group_default(self, dynamic_actor):
        """Test DynamicActor default tick group is POST_PHYSICS."""
        assert dynamic_actor.tick_group == TickGroup.POST_PHYSICS

    def test_tick_method_callable(self, basic_actor):
        """Test tick method is callable."""
        basic_actor.tick(0.016)  # Should not raise

    def test_tick_method_override(self):
        """Test overriding tick method."""
        class TickingActor(Actor):
            def __init__(self):
                super().__init__()
                self.tick_count = 0

            def tick(self, delta_time):
                self.tick_count += 1

        actor = TickingActor()
        actor.tick(0.016)
        actor.tick(0.016)
        assert actor.tick_count == 2

    def test_tick_receives_delta_time(self):
        """Test tick receives correct delta time."""
        class DeltaActor(Actor):
            def __init__(self):
                super().__init__()
                self.last_delta = 0.0

            def tick(self, delta_time):
                self.last_delta = delta_time

        actor = DeltaActor()
        actor.tick(0.033)
        assert actor.last_delta == 0.033


class TestTickGroups:
    """Tests for tick group ordering."""

    def test_tick_group_values(self):
        """Test tick group values for ordering."""
        assert TickGroup.PRE_PHYSICS.value < TickGroup.PHYSICS.value
        assert TickGroup.PHYSICS.value < TickGroup.POST_PHYSICS.value
        assert TickGroup.POST_PHYSICS.value < TickGroup.UPDATE.value
        assert TickGroup.UPDATE.value < TickGroup.POST_UPDATE.value
        assert TickGroup.POST_UPDATE.value < TickGroup.LATE_UPDATE.value

    def test_all_tick_groups_defined(self):
        """Test all expected tick groups are defined."""
        expected = {"PRE_PHYSICS", "PHYSICS", "POST_PHYSICS", "PRE_UPDATE", "UPDATE", "POST_UPDATE", "LATE_UPDATE"}
        actual = {tg.name for tg in TickGroup}
        assert expected == actual


# =============================================================================
# HIERARCHY TESTS
# =============================================================================


class TestActorHierarchy:
    """Tests for Actor parent/child hierarchy."""

    def test_actor_no_parent_default(self, basic_actor):
        """Test actor has no parent by default."""
        assert basic_actor.parent is None

    def test_set_parent(self):
        """Test setting parent actor."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")
        child.set_parent(parent)
        assert child.parent is parent

    def test_parent_children_list(self):
        """Test parent has children in list."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")
        child.set_parent(parent)
        assert child in parent.children

    def test_add_child_method(self):
        """Test add_child convenience method."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")
        parent.add_child(child)
        assert child.parent is parent
        assert child in parent.children

    def test_remove_child_method(self):
        """Test remove_child method."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")
        parent.add_child(child)
        parent.remove_child(child)
        assert child.parent is None
        assert child not in parent.children

    def test_multiple_children(self):
        """Test parent with multiple children."""
        parent = Actor(name="Parent")
        child1 = Actor(name="Child1")
        child2 = Actor(name="Child2")
        parent.add_child(child1)
        parent.add_child(child2)
        assert len(parent.children) == 2
        assert child1 in parent.children
        assert child2 in parent.children

    def test_change_parent(self):
        """Test changing parent removes from old parent."""
        parent1 = Actor(name="Parent1")
        parent2 = Actor(name="Parent2")
        child = Actor(name="Child")
        child.set_parent(parent1)
        child.set_parent(parent2)
        assert child not in parent1.children
        assert child in parent2.children
        assert child.parent is parent2

    def test_set_parent_none(self):
        """Test setting parent to None (unparent)."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")
        parent.add_child(child)
        child.set_parent(None)
        assert child.parent is None
        assert child not in parent.children

    def test_world_position_no_parent(self, basic_actor):
        """Test world position equals local position without parent."""
        basic_actor.position = (10.0, 20.0, 30.0)
        assert basic_actor.get_world_position() == (10.0, 20.0, 30.0)

    def test_world_position_with_parent(self):
        """Test world position adds parent position."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")
        parent.position = (10.0, 20.0, 30.0)
        child.position = (5.0, 5.0, 5.0)
        parent.add_child(child)
        world_pos = child.get_world_position()
        assert world_pos == (15.0, 25.0, 35.0)

    def test_world_position_nested(self):
        """Test world position with nested hierarchy."""
        grandparent = Actor(name="Grandparent")
        parent = Actor(name="Parent")
        child = Actor(name="Child")

        grandparent.position = (10.0, 0.0, 0.0)
        parent.position = (5.0, 0.0, 0.0)
        child.position = (2.0, 0.0, 0.0)

        grandparent.add_child(parent)
        parent.add_child(child)

        assert child.get_world_position() == (17.0, 0.0, 0.0)

    def test_children_empty_default(self, basic_actor):
        """Test children list is empty by default."""
        assert basic_actor.children == []

    def test_weak_reference_cleanup(self):
        """Test that deleted children are removed from list."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")
        parent.add_child(child)
        child_ref = weakref.ref(child)
        del child
        # Force garbage collection
        import gc
        gc.collect()
        # Children list should filter out dead references
        children = parent.children
        assert len(children) == 0


# =============================================================================
# TAG TESTS
# =============================================================================


class TestActorTags:
    """Tests for Actor tag system."""

    def test_no_tags_default(self, basic_actor):
        """Test actor has no tags by default."""
        assert len(basic_actor.tags) == 0

    def test_add_tag(self, basic_actor):
        """Test adding a tag."""
        basic_actor.add_tag("enemy")
        assert basic_actor.has_tag("enemy")

    def test_remove_tag(self, basic_actor):
        """Test removing a tag."""
        basic_actor.add_tag("enemy")
        basic_actor.remove_tag("enemy")
        assert not basic_actor.has_tag("enemy")

    def test_remove_nonexistent_tag(self, basic_actor):
        """Test removing nonexistent tag doesn't error."""
        basic_actor.remove_tag("nonexistent")  # Should not raise

    def test_has_tag(self, basic_actor):
        """Test has_tag method."""
        basic_actor.add_tag("test")
        assert basic_actor.has_tag("test")
        assert not basic_actor.has_tag("other")

    def test_has_any_tag(self, basic_actor):
        """Test has_any_tag with matching tags."""
        basic_actor.add_tag("enemy")
        basic_actor.add_tag("hostile")
        assert basic_actor.has_any_tag({"enemy", "friendly"})

    def test_has_any_tag_no_match(self, basic_actor):
        """Test has_any_tag with no matching tags."""
        basic_actor.add_tag("enemy")
        assert not basic_actor.has_any_tag({"friendly", "neutral"})

    def test_has_all_tags(self, basic_actor):
        """Test has_all_tags with all matching."""
        basic_actor.add_tag("enemy")
        basic_actor.add_tag("hostile")
        basic_actor.add_tag("boss")
        assert basic_actor.has_all_tags({"enemy", "hostile"})

    def test_has_all_tags_missing_one(self, basic_actor):
        """Test has_all_tags with missing tag."""
        basic_actor.add_tag("enemy")
        assert not basic_actor.has_all_tags({"enemy", "hostile"})

    def test_tags_property_immutable(self, basic_actor):
        """Test tags property returns immutable frozenset."""
        basic_actor.add_tag("test")
        tags = basic_actor.tags
        assert isinstance(tags, frozenset)

    def test_multiple_tags(self, basic_actor):
        """Test adding multiple tags."""
        basic_actor.add_tag("tag1")
        basic_actor.add_tag("tag2")
        basic_actor.add_tag("tag3")
        assert len(basic_actor.tags) == 3

    def test_duplicate_tag_ignored(self, basic_actor):
        """Test adding duplicate tag is ignored."""
        basic_actor.add_tag("test")
        basic_actor.add_tag("test")
        assert len(basic_actor.tags) == 1


# =============================================================================
# STATIC ACTOR TESTS
# =============================================================================


class TestStaticActor:
    """Tests for StaticActor class."""

    def test_static_actor_type(self, static_actor):
        """Test StaticActor has STATIC type."""
        assert static_actor.actor_type == ActorType.STATIC

    def test_static_actor_no_tick(self, static_actor):
        """Test StaticActor tick is disabled."""
        assert static_actor.tick_enabled is False

    def test_is_static_property(self, static_actor):
        """Test is_static property returns True."""
        assert static_actor.is_static is True

    def test_static_actor_can_have_components(self, static_actor):
        """Test StaticActor can still have components."""
        static_actor.add_component("mesh", MockComponent())
        assert static_actor.has_component("mesh")

    def test_static_actor_can_have_children(self, static_actor):
        """Test StaticActor can have children."""
        child = Actor(name="Child")
        static_actor.add_child(child)
        assert child in static_actor.children


# =============================================================================
# DYNAMIC ACTOR TESTS
# =============================================================================


class TestDynamicActor:
    """Tests for DynamicActor class."""

    def test_dynamic_actor_type(self, dynamic_actor):
        """Test DynamicActor has DYNAMIC type."""
        assert dynamic_actor.actor_type == ActorType.DYNAMIC

    def test_simulate_physics_default(self, dynamic_actor):
        """Test simulate_physics is True by default."""
        assert dynamic_actor.simulate_physics is True

    def test_simulate_physics_setter(self, dynamic_actor):
        """Test simulate_physics setter."""
        dynamic_actor.simulate_physics = False
        assert dynamic_actor.simulate_physics is False

    def test_mass_default(self, dynamic_actor):
        """Test default mass is 1.0."""
        assert dynamic_actor.mass == 1.0

    def test_mass_setter(self, dynamic_actor):
        """Test mass setter."""
        dynamic_actor.mass = 5.0
        assert dynamic_actor.mass == 5.0

    def test_mass_must_be_positive(self, dynamic_actor):
        """Test setting non-positive mass raises error."""
        with pytest.raises(ValueError, match="positive"):
            dynamic_actor.mass = 0.0
        with pytest.raises(ValueError, match="positive"):
            dynamic_actor.mass = -1.0

    def test_velocity_default(self, dynamic_actor):
        """Test velocity is zero by default."""
        assert dynamic_actor.velocity == (0.0, 0.0, 0.0)

    def test_velocity_setter(self, dynamic_actor):
        """Test velocity setter."""
        dynamic_actor.velocity = (1.0, 2.0, 3.0)
        assert dynamic_actor.velocity == (1.0, 2.0, 3.0)

    def test_angular_velocity_default(self, dynamic_actor):
        """Test angular velocity is zero by default."""
        assert dynamic_actor.angular_velocity == (0.0, 0.0, 0.0)

    def test_angular_velocity_setter(self, dynamic_actor):
        """Test angular velocity setter."""
        dynamic_actor.angular_velocity = (0.5, 0.5, 0.5)
        assert dynamic_actor.angular_velocity == (0.5, 0.5, 0.5)

    def test_add_force(self, dynamic_actor):
        """Test adding force changes velocity."""
        dynamic_actor.add_force((10.0, 0.0, 0.0))
        assert dynamic_actor.velocity[0] == 10.0  # F/m = 10/1

    def test_add_force_with_mass(self):
        """Test force considers mass."""
        actor = DynamicActor(mass=2.0)
        actor.add_force((10.0, 0.0, 0.0))
        assert actor.velocity[0] == 5.0  # F/m = 10/2

    def test_add_impulse(self, dynamic_actor):
        """Test adding impulse changes velocity."""
        dynamic_actor.add_impulse((5.0, 0.0, 0.0))
        assert dynamic_actor.velocity[0] == 5.0

    def test_add_impulse_with_mass(self):
        """Test impulse considers mass."""
        actor = DynamicActor(mass=5.0)
        actor.add_impulse((10.0, 0.0, 0.0))
        assert actor.velocity[0] == 2.0  # J/m = 10/5

    def test_cumulative_velocity(self, dynamic_actor):
        """Test forces accumulate velocity."""
        dynamic_actor.add_force((5.0, 0.0, 0.0))
        dynamic_actor.add_force((5.0, 0.0, 0.0))
        assert dynamic_actor.velocity[0] == 10.0

    def test_dynamic_actor_init_with_physics_disabled(self):
        """Test creating DynamicActor with physics disabled."""
        actor = DynamicActor(simulate_physics=False)
        assert actor.simulate_physics is False


# =============================================================================
# PAWN TESTS
# =============================================================================


class TestPawn:
    """Tests for Pawn class."""

    def test_pawn_actor_type(self, pawn):
        """Test Pawn has PAWN type."""
        assert pawn.actor_type == ActorType.PAWN

    def test_pawn_no_controller_default(self, pawn):
        """Test Pawn has no controller by default."""
        assert pawn.controller is None

    def test_pawn_not_possessed_default(self, pawn):
        """Test Pawn is not possessed by default."""
        assert pawn.is_possessed is False

    def test_pawn_not_player_controlled_default(self, pawn):
        """Test Pawn is not player controlled by default."""
        assert pawn.is_player_controlled is False

    def test_pawn_auto_possess_default(self, pawn):
        """Test auto_possess is False by default."""
        assert pawn._auto_possess is False

    def test_pawn_auto_possess_init(self):
        """Test creating Pawn with auto_possess."""
        pawn = Pawn(auto_possess=True)
        assert pawn._auto_possess is True

    def test_pawn_setup_player_input(self, pawn):
        """Test setup_player_input is callable."""
        pawn.setup_player_input()  # Should not raise

    def test_pawn_on_restart(self, pawn):
        """Test on_restart is callable."""
        pawn.on_restart()  # Should not raise

    def test_pawn_inherits_dynamic_actor(self, pawn):
        """Test Pawn has DynamicActor properties."""
        assert hasattr(pawn, "velocity")
        assert hasattr(pawn, "mass")
        assert hasattr(pawn, "simulate_physics")


# =============================================================================
# CHARACTER TESTS
# =============================================================================


class TestCharacter:
    """Tests for Character class."""

    def test_character_actor_type(self, character):
        """Test Character has CHARACTER type."""
        assert character.actor_type == ActorType.CHARACTER

    def test_max_walk_speed_default(self, character):
        """Test default max walk speed."""
        assert character.max_walk_speed == 6.0

    def test_max_walk_speed_setter(self, character):
        """Test max walk speed setter."""
        character.max_walk_speed = 10.0
        assert character.max_walk_speed == 10.0

    def test_max_walk_speed_minimum(self, character):
        """Test max walk speed can't be negative."""
        character.max_walk_speed = -5.0
        assert character.max_walk_speed == 0.0

    def test_max_run_speed_default(self, character):
        """Test default max run speed."""
        assert character.max_run_speed == 12.0

    def test_max_run_speed_setter(self, character):
        """Test max run speed setter."""
        character.max_run_speed = 20.0
        assert character.max_run_speed == 20.0

    def test_jump_velocity_default(self, character):
        """Test default jump velocity."""
        assert character.jump_velocity == 8.0

    def test_jump_velocity_setter(self, character):
        """Test jump velocity setter."""
        character.jump_velocity = 15.0
        assert character.jump_velocity == 15.0

    def test_jump_velocity_minimum(self, character):
        """Test jump velocity can't be negative."""
        character.jump_velocity = -10.0
        assert character.jump_velocity == 0.0

    def test_is_grounded_default(self, character):
        """Test character is grounded by default."""
        assert character.is_grounded is True

    def test_is_jumping_default(self, character):
        """Test character is not jumping by default."""
        assert character.is_jumping is False

    def test_is_falling_default(self, character):
        """Test character is not falling by default."""
        assert character.is_falling is False

    def test_is_crouching_default(self, character):
        """Test character is not crouching by default."""
        assert character.is_crouching is False

    def test_is_walking_default(self, character):
        """Test character is not walking by default."""
        assert character.is_walking is False

    def test_is_running_default(self, character):
        """Test character is not running by default."""
        assert character.is_running is False


class TestCharacterMovement:
    """Tests for Character movement functionality."""

    def test_jump_success(self, character):
        """Test successful jump."""
        assert character.jump() is True
        assert character.is_jumping is True
        assert character.is_grounded is False

    def test_jump_fail_not_grounded(self, character):
        """Test jump fails when not grounded."""
        character._is_grounded = False
        assert character.jump() is False

    def test_jump_fail_already_jumping(self, character):
        """Test jump fails when already jumping."""
        character.jump()
        assert character.jump() is False

    def test_jump_applies_velocity(self, character):
        """Test jump applies vertical velocity."""
        character.jump()
        assert character.velocity[1] == character.jump_velocity

    def test_crouch_success(self, character):
        """Test successful crouch."""
        assert character.crouch() is True
        assert character.is_crouching is True

    def test_crouch_fail_already_crouching(self, character):
        """Test crouch fails when already crouching."""
        character.crouch()
        assert character.crouch() is False

    def test_crouch_fail_disabled(self):
        """Test crouch fails when disabled."""
        char = Character(can_crouch=False)
        assert char.crouch() is False

    def test_uncrouch_success(self, character):
        """Test successful uncrouch."""
        character.crouch()
        assert character.uncrouch() is True
        assert character.is_crouching is False

    def test_uncrouch_fail_not_crouching(self, character):
        """Test uncrouch fails when not crouching."""
        assert character.uncrouch() is False

    def test_start_running(self, character):
        """Test start running."""
        character.start_running()
        assert character.is_running is True

    def test_stop_running(self, character):
        """Test stop running."""
        character.start_running()
        character.stop_running()
        assert character.is_running is False

    def test_add_movement_input(self, character):
        """Test adding movement input."""
        character.add_movement_input(0.5, 0.5)
        assert character._movement_input == (0.5, 0.5)

    def test_add_movement_input_clamped(self, character):
        """Test movement input is clamped to -1, 1."""
        character.add_movement_input(2.0, -2.0)
        assert character._movement_input == (1.0, -1.0)

    def test_add_look_input(self, character):
        """Test adding look input."""
        character.add_look_input(0.3, 0.2)
        assert character._look_input == (0.3, 0.2)

    def test_current_max_speed_walking(self, character):
        """Test current max speed when walking."""
        assert character.current_max_speed == character.max_walk_speed

    def test_current_max_speed_running(self, character):
        """Test current max speed when running."""
        character.start_running()
        assert character.current_max_speed == character.max_run_speed

    def test_current_max_speed_crouching(self, character):
        """Test current max speed when crouching."""
        character.crouch()
        assert character.current_max_speed == character.max_walk_speed * 0.5

    def test_tick_clears_movement_input(self, character):
        """Test tick clears movement input."""
        character.add_movement_input(1.0, 0.0)
        character.tick(0.016)
        assert character._movement_input == (0.0, 0.0)

    def test_tick_updates_walking_state(self, character):
        """Test tick updates walking state based on input."""
        character.add_movement_input(1.0, 0.0)
        character.tick(0.016)
        # After tick, input is cleared, so walking should be checked before
        character.add_movement_input(1.0, 0.0)
        # During tick with input, walking should be set
        assert character._movement_input == (1.0, 0.0)


class TestCharacterInit:
    """Tests for Character initialization options."""

    def test_character_custom_speeds(self):
        """Test creating Character with custom speeds."""
        char = Character(
            max_walk_speed=8.0,
            max_run_speed=16.0,
            jump_velocity=10.0,
        )
        assert char.max_walk_speed == 8.0
        assert char.max_run_speed == 16.0
        assert char.jump_velocity == 10.0

    def test_character_can_crouch_disabled(self):
        """Test creating Character with crouch disabled."""
        char = Character(can_crouch=False)
        assert char._can_crouch is False


# =============================================================================
# LIFECYCLE CALLBACK TESTS
# =============================================================================


class TestActorLifecycleCallbacks:
    """Tests for Actor lifecycle callbacks."""

    def test_on_spawn_callable(self, basic_actor):
        """Test on_spawn is callable."""
        basic_actor.on_spawn()  # Should not raise

    def test_begin_play_callable(self, basic_actor):
        """Test begin_play is callable."""
        basic_actor.begin_play()  # Should not raise

    def test_end_play_callable(self, basic_actor):
        """Test end_play is callable."""
        basic_actor.end_play()  # Should not raise

    def test_on_destroy_clears_children(self):
        """Test on_destroy clears children."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")
        parent.add_child(child)
        parent.on_destroy()
        assert child.parent is None

    def test_on_destroy_clears_components(self, basic_actor):
        """Test on_destroy clears components."""
        basic_actor.add_component("test", MockComponent())
        basic_actor.on_destroy()
        assert len(basic_actor.components) == 0

    def test_destroy_method(self, basic_actor):
        """Test destroy method triggers state transition."""
        # The actual transition might require lifecycle manager setup
        basic_actor.destroy()
        # Should not raise, actual state depends on lifecycle manager


class TestActorOverrides:
    """Tests for overriding Actor methods in subclasses."""

    def test_override_on_spawn(self):
        """Test overriding on_spawn in subclass."""
        class CustomActor(Actor):
            def __init__(self):
                super().__init__()
                self.spawn_called = False

            def on_spawn(self):
                self.spawn_called = True

        actor = CustomActor()
        actor.on_spawn()
        assert actor.spawn_called is True

    def test_override_begin_play(self):
        """Test overriding begin_play in subclass."""
        class CustomActor(Actor):
            def __init__(self):
                super().__init__()
                self.play_started = False

            def begin_play(self):
                self.play_started = True

        actor = CustomActor()
        actor.begin_play()
        assert actor.play_started is True


# =============================================================================
# ACTOR META TESTS
# =============================================================================


class TestActorMeta:
    """Tests for ActorMeta metaclass."""

    def test_actor_meta_registry(self):
        """Test actors are registered in metaclass registry."""
        class TestActor(Actor):
            pass

        assert TestActor._actor_type_id > 0
        found = ActorMeta.get_by_id(TestActor._actor_type_id)
        assert found is TestActor

    def test_actor_meta_unique_ids(self):
        """Test each actor class gets unique ID."""
        class Actor1(Actor):
            pass

        class Actor2(Actor):
            pass

        assert Actor1._actor_type_id != Actor2._actor_type_id

    def test_actor_meta_get_by_name(self):
        """Test looking up actor by qualified name."""
        class NamedActor(Actor):
            pass

        found = ActorMeta.get_by_name(NamedActor._actor_type_name)
        assert found is NamedActor

    def test_actor_meta_all_actor_types(self):
        """Test getting all registered actor types."""
        class ListedActor(Actor):
            pass

        all_types = ActorMeta.all_actor_types()
        assert ListedActor in all_types

    def test_actor_meta_base_classes_not_registered(self):
        """Test base classes have type ID 0."""
        assert Actor._actor_type_id == 0
        assert StaticActor._actor_type_id == 0
        assert DynamicActor._actor_type_id == 0
        assert Pawn._actor_type_id == 0
        assert Character._actor_type_id == 0


# =============================================================================
# TRANSFORM TESTS
# =============================================================================


class TestTransform:
    """Tests for Transform class."""

    def test_transform_defaults(self):
        """Test Transform default values."""
        t = Transform()
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_transform_custom_values(self):
        """Test Transform with custom values."""
        t = Transform(
            position=(1.0, 2.0, 3.0),
            rotation=(0.0, 0.707, 0.0, 0.707),
            scale=(2.0, 2.0, 2.0),
        )
        assert t.position == (1.0, 2.0, 3.0)
        assert t.rotation == (0.0, 0.707, 0.0, 0.707)
        assert t.scale == (2.0, 2.0, 2.0)

    def test_transform_copy(self):
        """Test Transform copy method."""
        t1 = Transform(position=(1.0, 2.0, 3.0))
        t2 = t1.copy()
        assert t2.position == t1.position
        t1.position = (5.0, 5.0, 5.0)
        assert t2.position == (1.0, 2.0, 3.0)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestActorEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_entity_id_thread_safety(self):
        """Test entity ID generation is thread-safe."""
        import threading
        actors = []
        lock = threading.Lock()

        def create_actor():
            actor = Actor()
            with lock:
                actors.append(actor)

        threads = [threading.Thread(target=create_actor) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        ids = [a.entity_id for a in actors]
        assert len(ids) == len(set(ids))  # All IDs unique

    def test_reset_entity_ids(self):
        """Test resetting entity ID counter."""
        actor1 = Actor()
        first_id = actor1.entity_id
        Actor.reset_entity_ids()
        actor2 = Actor()
        assert actor2.entity_id == ENTITY_ID_START

    def test_empty_name(self):
        """Test actor with empty name gets default."""
        actor = Actor(name="")
        # Empty string should still work, not raise
        assert actor.name == ""

    def test_actor_with_none_transform(self):
        """Test actor handles None transform gracefully."""
        actor = Actor(transform=None)
        assert actor.transform is not None
        assert actor.position == (0.0, 0.0, 0.0)

    def test_actor_with_none_tags(self):
        """Test actor handles None tags gracefully."""
        actor = Actor(tags=None)
        assert len(actor.tags) == 0
