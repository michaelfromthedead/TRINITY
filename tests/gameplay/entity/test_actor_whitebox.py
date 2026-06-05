"""
WHITEBOX Tests for Actor System

Tests:
- Actor creation and identification
- Transform hierarchy
- Component management
- Tag system
- Tick configuration
- Actor subclasses (StaticActor, DynamicActor, Pawn, Character)
- ActorMeta metaclass behavior
"""
import pytest
import threading
from typing import Any, Optional
from uuid import uuid4

from engine.gameplay.entity.actor import (
    ActorMeta,
    ComponentContainer,
    Transform,
    Actor,
    StaticActor,
    DynamicActor,
    Pawn,
    Character,
)
from engine.gameplay.entity.constants import (
    ActorType,
    LifecycleState,
    TickGroup,
    ENTITY_ID_START,
    ENTITY_NAME_MAX_LENGTH,
    MAX_COMPONENTS_PER_ENTITY,
    DEFAULT_MAX_WALK_SPEED,
    DEFAULT_MAX_RUN_SPEED,
    DEFAULT_JUMP_VELOCITY,
    CROUCH_SPEED_MULTIPLIER,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_actors():
    """Reset actor state before each test."""
    ActorMeta.clear_registry()
    Actor.reset_entity_ids()
    yield


@pytest.fixture
def basic_actor():
    """Create a basic actor."""
    return Actor(name="TestActor")


@pytest.fixture
def positioned_actor():
    """Create an actor with a specific transform."""
    transform = Transform(
        position=(10.0, 20.0, 30.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        scale=(2.0, 2.0, 2.0),
    )
    return Actor(name="PositionedActor", transform=transform)


class DummyComponent:
    """Dummy component for testing."""
    _component = True

    def __init__(self, value: int = 0):
        self.value = value


class AnotherComponent:
    """Another dummy component for testing."""
    _component = True

    def __init__(self, name: str = "default"):
        self.name = name


# =============================================================================
# TRANSFORM TESTS
# =============================================================================


class TestTransform:
    """Whitebox tests for Transform."""

    def test_default_creation(self):
        """Default transform should have identity values."""
        t = Transform()
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_custom_creation(self):
        """Transform with custom values."""
        t = Transform(
            position=(1.0, 2.0, 3.0),
            rotation=(0.5, 0.5, 0.5, 0.5),
            scale=(2.0, 2.0, 2.0),
        )
        assert t.position == (1.0, 2.0, 3.0)
        assert t.rotation == (0.5, 0.5, 0.5, 0.5)
        assert t.scale == (2.0, 2.0, 2.0)

    def test_copy(self):
        """copy should create independent copy."""
        original = Transform(position=(5.0, 10.0, 15.0))
        copied = original.copy()
        assert copied.position == original.position
        # Modifying copy shouldn't affect original
        copied.position = (0.0, 0.0, 0.0)
        assert original.position == (5.0, 10.0, 15.0)


# =============================================================================
# COMPONENT CONTAINER TESTS
# =============================================================================


class TestComponentContainer:
    """Whitebox tests for ComponentContainer."""

    def test_basic_creation(self, basic_actor):
        """Container should be created with owner."""
        container = ComponentContainer(basic_actor)
        assert len(container) == 0

    def test_add_component(self, basic_actor):
        """add should add component."""
        container = ComponentContainer(basic_actor)
        comp = DummyComponent(value=42)
        container.add("dummy", comp)
        assert container.get("dummy") == comp

    def test_add_duplicate_raises(self, basic_actor):
        """Adding duplicate name should raise."""
        container = ComponentContainer(basic_actor)
        container.add("dummy", DummyComponent())
        with pytest.raises(ValueError, match="already exists"):
            container.add("dummy", DummyComponent())

    def test_add_exceeds_max_raises(self, basic_actor):
        """Adding beyond max should raise."""
        container = ComponentContainer(basic_actor)
        for i in range(MAX_COMPONENTS_PER_ENTITY):
            container.add(f"comp_{i}", DummyComponent())
        with pytest.raises(ValueError, match="Maximum component count"):
            container.add("one_more", DummyComponent())

    def test_remove_component(self, basic_actor):
        """remove should remove and return component."""
        container = ComponentContainer(basic_actor)
        comp = DummyComponent()
        container.add("dummy", comp)
        removed = container.remove("dummy")
        assert removed == comp
        assert container.get("dummy") is None

    def test_remove_nonexistent_returns_none(self, basic_actor):
        """Removing nonexistent should return None."""
        container = ComponentContainer(basic_actor)
        assert container.remove("nonexistent") is None

    def test_get_by_type(self, basic_actor):
        """get_by_type should find component by type."""
        container = ComponentContainer(basic_actor)
        comp = DummyComponent(value=99)
        container.add("my_dummy", comp)
        found = container.get_by_type(DummyComponent)
        assert found == comp

    def test_get_by_type_not_found(self, basic_actor):
        """get_by_type should return None if not found."""
        container = ComponentContainer(basic_actor)
        assert container.get_by_type(DummyComponent) is None

    def test_has_by_name(self, basic_actor):
        """has should check by name."""
        container = ComponentContainer(basic_actor)
        container.add("dummy", DummyComponent())
        assert container.has("dummy") is True
        assert container.has("other") is False

    def test_has_by_type(self, basic_actor):
        """has_type should check by type."""
        container = ComponentContainer(basic_actor)
        container.add("dummy", DummyComponent())
        assert container.has_type(DummyComponent) is True
        assert container.has_type(AnotherComponent) is False

    def test_iteration(self, basic_actor):
        """Container should be iterable."""
        container = ComponentContainer(basic_actor)
        container.add("a", DummyComponent())
        container.add("b", AnotherComponent())
        items = list(container)
        assert len(items) == 2
        names = {name for name, comp in items}
        assert "a" in names
        assert "b" in names

    def test_clear(self, basic_actor):
        """clear should remove all components."""
        container = ComponentContainer(basic_actor)
        container.add("a", DummyComponent())
        container.add("b", AnotherComponent())
        container.clear()
        assert len(container) == 0


# =============================================================================
# ACTOR META TESTS
# =============================================================================


class TestActorMeta:
    """Whitebox tests for ActorMeta."""

    def test_base_classes_not_registered(self):
        """Base classes should not be in registry."""
        assert ActorMeta.get_by_name("Actor") is None

    def test_subclass_registered(self):
        """Custom actor subclasses should be registered."""
        class CustomActor(Actor):
            pass

        assert CustomActor._actor_type_id > 0
        found = ActorMeta.get_by_id(CustomActor._actor_type_id)
        assert found == CustomActor

    def test_get_by_name(self):
        """get_by_name should find registered actors."""
        class NamedActor(Actor):
            pass

        found = ActorMeta.get_by_name(NamedActor._actor_type_name)
        assert found == NamedActor

    def test_all_actor_types(self):
        """all_actor_types should return all registered."""
        class Actor1(Actor):
            pass

        class Actor2(Actor):
            pass

        all_types = ActorMeta.all_actor_types()
        assert Actor1 in all_types
        assert Actor2 in all_types

    def test_unique_type_ids(self):
        """Each actor type should have unique ID."""
        class ActorA(Actor):
            pass

        class ActorB(Actor):
            pass

        assert ActorA._actor_type_id != ActorB._actor_type_id


# =============================================================================
# ACTOR BASIC TESTS
# =============================================================================


class TestActor:
    """Whitebox tests for Actor."""

    def test_basic_creation(self, basic_actor):
        """Basic actor creation."""
        assert basic_actor.name == "TestActor"
        assert basic_actor.entity_id >= ENTITY_ID_START
        assert basic_actor.actor_type == ActorType.STATIC

    def test_auto_generated_name(self):
        """Actor without name should get auto-generated name."""
        actor = Actor()
        assert actor.name.startswith("Actor_")

    def test_name_truncation(self):
        """Long names should be truncated."""
        long_name = "x" * (ENTITY_NAME_MAX_LENGTH + 100)
        actor = Actor(name=long_name)
        assert len(actor.name) == ENTITY_NAME_MAX_LENGTH

    def test_unique_entity_ids(self):
        """Each actor should have unique entity ID."""
        actors = [Actor() for _ in range(10)]
        ids = [a.entity_id for a in actors]
        assert len(set(ids)) == 10

    def test_lifecycle_state_initial(self, basic_actor):
        """Initial lifecycle state should be CREATED."""
        assert basic_actor._lifecycle_state == LifecycleState.CREATED

    def test_tick_defaults(self, basic_actor):
        """Default tick settings."""
        assert basic_actor.tick_enabled is True
        assert basic_actor.tick_group == TickGroup.UPDATE

    def test_repr(self, basic_actor):
        """repr should include useful info."""
        r = repr(basic_actor)
        assert "Actor" in r
        assert str(basic_actor.entity_id) in r
        assert basic_actor.name in r


class TestActorTransform:
    """Tests for actor transform properties."""

    def test_transform_property(self, positioned_actor):
        """transform property should return transform."""
        assert positioned_actor.transform.position == (10.0, 20.0, 30.0)

    def test_position_property(self, positioned_actor):
        """position property shortcut."""
        assert positioned_actor.position == (10.0, 20.0, 30.0)

    def test_position_setter(self, basic_actor):
        """position setter should work."""
        basic_actor.position = (5.0, 5.0, 5.0)
        assert basic_actor.position == (5.0, 5.0, 5.0)

    def test_rotation_property(self, positioned_actor):
        """rotation property shortcut."""
        assert positioned_actor.rotation == (0.0, 0.0, 0.0, 1.0)

    def test_rotation_setter(self, basic_actor):
        """rotation setter should work."""
        basic_actor.rotation = (0.5, 0.5, 0.5, 0.5)
        assert basic_actor.rotation == (0.5, 0.5, 0.5, 0.5)

    def test_scale_property(self, positioned_actor):
        """scale property shortcut."""
        assert positioned_actor.scale == (2.0, 2.0, 2.0)

    def test_scale_setter(self, basic_actor):
        """scale setter should work."""
        basic_actor.scale = (3.0, 3.0, 3.0)
        assert basic_actor.scale == (3.0, 3.0, 3.0)


class TestActorHierarchy:
    """Tests for actor parent-child hierarchy."""

    def test_initial_no_parent(self, basic_actor):
        """New actor should have no parent."""
        assert basic_actor.parent is None

    def test_initial_no_children(self, basic_actor):
        """New actor should have no children."""
        assert basic_actor.children == []

    def test_set_parent(self):
        """set_parent should establish relationship."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")
        child.set_parent(parent)
        assert child.parent == parent
        assert child in parent.children

    def test_add_child(self):
        """add_child should establish relationship."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")
        parent.add_child(child)
        assert child.parent == parent
        assert child in parent.children

    def test_remove_child(self):
        """remove_child should break relationship."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")
        parent.add_child(child)
        parent.remove_child(child)
        assert child.parent is None
        assert child not in parent.children

    def test_change_parent(self):
        """Changing parent should update both old and new parent."""
        parent1 = Actor(name="Parent1")
        parent2 = Actor(name="Parent2")
        child = Actor(name="Child")

        child.set_parent(parent1)
        assert child in parent1.children

        child.set_parent(parent2)
        assert child not in parent1.children
        assert child in parent2.children
        assert child.parent == parent2

    def test_get_world_position_no_parent(self, positioned_actor):
        """World position without parent equals local position."""
        world_pos = positioned_actor.get_world_position()
        assert world_pos == positioned_actor.position

    def test_get_world_position_with_parent(self):
        """World position should accumulate parent position."""
        parent = Actor(
            name="Parent",
            transform=Transform(position=(100.0, 0.0, 0.0)),
        )
        child = Actor(
            name="Child",
            transform=Transform(position=(10.0, 0.0, 0.0)),
        )
        child.set_parent(parent)

        world_pos = child.get_world_position()
        assert world_pos == (110.0, 0.0, 0.0)


class TestActorComponents:
    """Tests for actor component management."""

    def test_components_property(self, basic_actor):
        """components property should return container."""
        assert isinstance(basic_actor.components, ComponentContainer)

    def test_add_component(self, basic_actor):
        """add_component should add to container."""
        comp = DummyComponent(value=42)
        basic_actor.add_component("dummy", comp)
        assert basic_actor.get_component("dummy") == comp

    def test_remove_component(self, basic_actor):
        """remove_component should remove from container."""
        comp = DummyComponent()
        basic_actor.add_component("dummy", comp)
        removed = basic_actor.remove_component("dummy")
        assert removed == comp
        assert basic_actor.get_component("dummy") is None

    def test_get_component_by_type(self, basic_actor):
        """get_component_by_type should find by type."""
        comp = DummyComponent(value=99)
        basic_actor.add_component("my_comp", comp)
        found = basic_actor.get_component_by_type(DummyComponent)
        assert found == comp

    def test_has_component(self, basic_actor):
        """has_component should check existence."""
        basic_actor.add_component("dummy", DummyComponent())
        assert basic_actor.has_component("dummy") is True
        assert basic_actor.has_component("other") is False

    def test_has_component_type(self, basic_actor):
        """has_component_type should check by type."""
        basic_actor.add_component("dummy", DummyComponent())
        assert basic_actor.has_component_type(DummyComponent) is True
        assert basic_actor.has_component_type(AnotherComponent) is False


class TestActorTags:
    """Tests for actor tag system."""

    def test_initial_no_tags(self, basic_actor):
        """New actor should have no tags."""
        assert basic_actor.tags == frozenset()

    def test_initial_with_tags(self):
        """Actor can be created with tags."""
        actor = Actor(tags={"player", "friendly"})
        assert "player" in actor.tags
        assert "friendly" in actor.tags

    def test_add_tag(self, basic_actor):
        """add_tag should add tag."""
        basic_actor.add_tag("enemy")
        assert basic_actor.has_tag("enemy")

    def test_remove_tag(self, basic_actor):
        """remove_tag should remove tag."""
        basic_actor.add_tag("enemy")
        basic_actor.remove_tag("enemy")
        assert not basic_actor.has_tag("enemy")

    def test_remove_nonexistent_tag(self, basic_actor):
        """Removing nonexistent tag should not raise."""
        basic_actor.remove_tag("nonexistent")  # Should not raise

    def test_has_any_tag(self, basic_actor):
        """has_any_tag should check for any matching tag."""
        basic_actor.add_tag("player")
        assert basic_actor.has_any_tag({"player", "enemy"}) is True
        assert basic_actor.has_any_tag({"enemy", "boss"}) is False

    def test_has_all_tags(self, basic_actor):
        """has_all_tags should check for all matching tags."""
        basic_actor.add_tag("player")
        basic_actor.add_tag("friendly")
        assert basic_actor.has_all_tags({"player", "friendly"}) is True
        assert basic_actor.has_all_tags({"player", "enemy"}) is False

    def test_tags_immutable(self, basic_actor):
        """tags property should return immutable set."""
        basic_actor.add_tag("test")
        tags = basic_actor.tags
        assert isinstance(tags, frozenset)


class TestActorTick:
    """Tests for actor tick functionality."""

    def test_tick_enabled_default(self, basic_actor):
        """Tick should be enabled by default."""
        assert basic_actor.tick_enabled is True

    def test_tick_enabled_setter(self, basic_actor):
        """tick_enabled setter should work."""
        basic_actor.tick_enabled = False
        assert basic_actor.tick_enabled is False

    def test_tick_group_default(self, basic_actor):
        """Default tick group should be UPDATE."""
        assert basic_actor.tick_group == TickGroup.UPDATE

    def test_tick_group_setter(self, basic_actor):
        """tick_group setter should work."""
        basic_actor.tick_group = TickGroup.POST_PHYSICS
        assert basic_actor.tick_group == TickGroup.POST_PHYSICS

    def test_tick_method_exists(self, basic_actor):
        """tick method should be callable."""
        basic_actor.tick(0.016)  # Should not raise


class TestActorLifecycle:
    """Tests for actor lifecycle callbacks."""

    def test_on_spawn_called(self):
        """on_spawn should be callable."""
        calls = []

        class SpawnActor(Actor):
            def on_spawn(self):
                calls.append("spawn")

        actor = SpawnActor()
        actor.on_spawn()
        assert "spawn" in calls

    def test_begin_play_called(self):
        """begin_play should be callable."""
        calls = []

        class PlayActor(Actor):
            def begin_play(self):
                calls.append("begin")

        actor = PlayActor()
        actor.begin_play()
        assert "begin" in calls

    def test_end_play_called(self):
        """end_play should be callable."""
        calls = []

        class EndActor(Actor):
            def end_play(self):
                calls.append("end")

        actor = EndActor()
        actor.end_play()
        assert "end" in calls

    def test_on_destroy_clears_children(self):
        """on_destroy should clear children."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")
        parent.add_child(child)

        parent.on_destroy()
        assert child.parent is None

    def test_on_destroy_clears_components(self, basic_actor):
        """on_destroy should clear components."""
        basic_actor.add_component("dummy", DummyComponent())
        basic_actor.on_destroy()
        assert len(basic_actor.components) == 0


# =============================================================================
# STATIC ACTOR TESTS
# =============================================================================


class TestStaticActor:
    """Whitebox tests for StaticActor."""

    def test_creation(self):
        """StaticActor creation."""
        actor = StaticActor(name="Static")
        assert actor.actor_type == ActorType.STATIC
        assert actor.is_static is True

    def test_tick_disabled_by_default(self):
        """StaticActor should have tick disabled by default."""
        actor = StaticActor()
        assert actor.tick_enabled is False


# =============================================================================
# DYNAMIC ACTOR TESTS
# =============================================================================


class TestDynamicActor:
    """Whitebox tests for DynamicActor."""

    def test_creation(self):
        """DynamicActor creation."""
        actor = DynamicActor(name="Dynamic")
        assert actor.actor_type == ActorType.DYNAMIC

    def test_simulate_physics_default(self):
        """simulate_physics should be True by default."""
        actor = DynamicActor()
        assert actor.simulate_physics is True

    def test_simulate_physics_setter(self):
        """simulate_physics setter should work."""
        actor = DynamicActor()
        actor.simulate_physics = False
        assert actor.simulate_physics is False

    def test_mass_default(self):
        """Default mass should be 1."""
        actor = DynamicActor()
        assert actor.mass == 1.0

    def test_mass_custom(self):
        """Custom mass should be settable."""
        actor = DynamicActor(mass=10.0)
        assert actor.mass == 10.0

    def test_mass_setter_positive(self):
        """Mass setter should require positive value."""
        actor = DynamicActor()
        with pytest.raises(ValueError, match="must be positive"):
            actor.mass = 0

    def test_velocity_default(self):
        """Default velocity should be zero."""
        actor = DynamicActor()
        assert actor.velocity == (0.0, 0.0, 0.0)

    def test_velocity_setter(self):
        """velocity setter should work."""
        actor = DynamicActor()
        actor.velocity = (10.0, 5.0, 0.0)
        assert actor.velocity == (10.0, 5.0, 0.0)

    def test_angular_velocity(self):
        """angular_velocity property should work."""
        actor = DynamicActor()
        actor.angular_velocity = (1.0, 2.0, 3.0)
        assert actor.angular_velocity == (1.0, 2.0, 3.0)

    def test_add_force(self):
        """add_force should change velocity."""
        actor = DynamicActor(mass=2.0)
        actor.velocity = (0.0, 0.0, 0.0)
        actor.add_force((10.0, 0.0, 0.0))  # F=10, m=2, a=5
        assert actor.velocity == (5.0, 0.0, 0.0)

    def test_add_impulse(self):
        """add_impulse should change velocity instantly."""
        actor = DynamicActor(mass=2.0)
        actor.velocity = (0.0, 0.0, 0.0)
        actor.add_impulse((10.0, 0.0, 0.0))  # J=10, m=2, dv=5
        assert actor.velocity == (5.0, 0.0, 0.0)

    def test_tick_group_post_physics(self):
        """DynamicActor should tick in POST_PHYSICS."""
        actor = DynamicActor()
        assert actor.tick_group == TickGroup.POST_PHYSICS


# =============================================================================
# PAWN TESTS
# =============================================================================


class TestPawn:
    """Whitebox tests for Pawn."""

    def test_creation(self):
        """Pawn creation."""
        pawn = Pawn(name="TestPawn")
        assert pawn.actor_type == ActorType.PAWN

    def test_initially_not_possessed(self):
        """Pawn should start unpossessed."""
        pawn = Pawn()
        assert pawn.is_possessed is False
        assert pawn.controller is None

    def test_auto_possess_flag(self):
        """auto_possess flag should be settable."""
        pawn = Pawn(auto_possess=True)
        assert pawn._auto_possess is True

    def test_is_player_controlled_false_initially(self):
        """is_player_controlled should be False when not possessed."""
        pawn = Pawn()
        assert pawn.is_player_controlled is False


# =============================================================================
# CHARACTER TESTS
# =============================================================================


class TestCharacter:
    """Whitebox tests for Character."""

    def test_creation(self):
        """Character creation."""
        char = Character(name="TestChar")
        assert char.actor_type == ActorType.CHARACTER

    def test_default_movement_speeds(self):
        """Default movement speed values."""
        char = Character()
        assert char.max_walk_speed == DEFAULT_MAX_WALK_SPEED
        assert char.max_run_speed == DEFAULT_MAX_RUN_SPEED
        assert char.jump_velocity == DEFAULT_JUMP_VELOCITY

    def test_custom_movement_speeds(self):
        """Custom movement speed values."""
        char = Character(
            max_walk_speed=10.0,
            max_run_speed=20.0,
            jump_velocity=15.0,
        )
        assert char.max_walk_speed == 10.0
        assert char.max_run_speed == 20.0
        assert char.jump_velocity == 15.0

    def test_movement_speed_setters(self):
        """Movement speed setters should clamp to 0."""
        char = Character()
        char.max_walk_speed = -5.0
        assert char.max_walk_speed == 0.0

    def test_initial_movement_state(self):
        """Initial movement state should be idle."""
        char = Character()
        assert char.is_walking is False
        assert char.is_running is False
        assert char.is_jumping is False
        assert char.is_falling is False
        assert char.is_crouching is False
        assert char.is_grounded is True

    def test_current_max_speed_walking(self):
        """current_max_speed should return walk speed when walking."""
        char = Character()
        assert char.current_max_speed == char.max_walk_speed

    def test_current_max_speed_running(self):
        """current_max_speed should return run speed when running."""
        char = Character()
        char.start_running()
        assert char.current_max_speed == char.max_run_speed

    def test_current_max_speed_crouching(self):
        """current_max_speed should reduce when crouching."""
        char = Character()
        char.crouch()
        expected = char.max_walk_speed * CROUCH_SPEED_MULTIPLIER
        assert char.current_max_speed == pytest.approx(expected)

    def test_add_movement_input(self):
        """add_movement_input should set movement input."""
        char = Character()
        char.add_movement_input(0.5, 0.3)
        assert char._movement_input == (0.5, 0.3)

    def test_add_movement_input_clamped(self):
        """add_movement_input should clamp values."""
        char = Character()
        char.add_movement_input(2.0, -2.0)
        assert char._movement_input == (1.0, -1.0)

    def test_add_look_input(self):
        """add_look_input should set look input."""
        char = Character()
        char.add_look_input(45.0, -15.0)
        assert char._look_input == (45.0, -15.0)

    def test_jump_when_grounded(self):
        """jump should work when grounded."""
        char = Character()
        assert char.jump() is True
        assert char.is_jumping is True
        assert char.is_grounded is False

    def test_jump_when_already_jumping(self):
        """jump should fail when already jumping."""
        char = Character()
        char.jump()
        assert char.jump() is False

    def test_jump_when_not_grounded(self):
        """jump should fail when not grounded."""
        char = Character()
        char._is_grounded = False
        assert char.jump() is False

    def test_crouch(self):
        """crouch should set crouching state."""
        char = Character()
        assert char.crouch() is True
        assert char.is_crouching is True

    def test_crouch_when_cant_crouch(self):
        """crouch should fail if can_crouch is False."""
        char = Character(can_crouch=False)
        assert char.crouch() is False

    def test_crouch_when_already_crouching(self):
        """crouch should fail when already crouching."""
        char = Character()
        char.crouch()
        assert char.crouch() is False

    def test_uncrouch(self):
        """uncrouch should clear crouching state."""
        char = Character()
        char.crouch()
        assert char.uncrouch() is True
        assert char.is_crouching is False

    def test_uncrouch_when_not_crouching(self):
        """uncrouch should fail when not crouching."""
        char = Character()
        assert char.uncrouch() is False

    def test_start_stop_running(self):
        """start/stop running should toggle running state."""
        char = Character()
        char.start_running()
        assert char.is_running is True
        char.stop_running()
        assert char.is_running is False

    def test_tick_updates_walking_state(self):
        """tick should update walking state based on input."""
        char = Character()
        char.add_movement_input(1.0, 0.0)
        char.tick(0.016)
        # Walking state should be set based on input
        # Input is cleared after tick
        assert char._movement_input == (0.0, 0.0)

    def test_tick_applies_movement(self):
        """tick should apply movement when grounded with input."""
        char = Character()
        char._is_grounded = True
        char.add_movement_input(1.0, 0.0)
        char.tick(0.016)
        # Velocity should be updated (forward input)
        assert char.velocity[0] == char.max_walk_speed
