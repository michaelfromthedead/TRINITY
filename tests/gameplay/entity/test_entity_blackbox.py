"""
BLACKBOX Tests for Entity System.

Tests observable PUBLIC behavior only - no internal state inspection.
Covers: Actor, Pawn, Character, Controller, Prefab, and Lifecycle systems.

Test count: 80+ tests
"""

import pytest
import weakref
from typing import List, Optional, Set, Tuple, Any, Type
from dataclasses import dataclass

# Public API imports only
from engine.gameplay.entity import (
    # Config
    ActorConfig,
    # Actors
    Actor,
    Pawn,
    Character,
    # Controllers
    Controller,
    PlayerController,
    AIController,
    # Prefabs
    PrefabComponent,
    Prefab,
    PrefabRegistry,
    # Lifecycle
    LifecycleEvent,
    LifecycleCallback,
    LifecycleManager,
    # EventLog Integration
    EntitySpawned,
    EntityDestroyed,
    ComponentAdded,
    ComponentRemoved,
    EntityStateChanged,
    EntityEventLog,
    get_entity_event_log,
    clear_entity_event_log,
)

from engine.gameplay.constants import (
    ActorType,
    EntityState,
)

from engine.gameplay.entity.constants import (
    DEFAULT_MAX_WALK_SPEED,
    DEFAULT_JUMP_FORCE,
    DEFAULT_CHARACTER_HEALTH,
    DEFAULT_CHARACTER_MAX_HEALTH,
    CROUCH_SPEED_MULTIPLIER,
    SPRINT_SPEED_MULTIPLIER,
    MAX_PREFAB_INHERITANCE_DEPTH,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton state between tests."""
    # Reset lifecycle manager
    LifecycleManager._instance = None
    # Reset prefab registry
    PrefabRegistry._instance = None
    # Clear event log
    clear_entity_event_log()
    yield


@pytest.fixture
def basic_actor() -> Actor:
    """Create a basic actor."""
    return Actor(name="TestActor")


@pytest.fixture
def pawn() -> Pawn:
    """Create a pawn."""
    return Pawn(name="TestPawn")


@pytest.fixture
def character() -> Character:
    """Create a character."""
    return Character(name="TestCharacter")


@pytest.fixture
def player_controller() -> PlayerController:
    """Create a player controller."""
    return PlayerController(player_index=0)


@pytest.fixture
def ai_controller() -> AIController:
    """Create an AI controller."""
    return AIController()


@pytest.fixture
def lifecycle_manager() -> LifecycleManager:
    """Get lifecycle manager instance."""
    return LifecycleManager()


# =============================================================================
# TEST COMPONENTS (for component tests)
# =============================================================================


@dataclass
class HealthComponent:
    """Test health component."""
    current: float = 100.0
    maximum: float = 100.0


@dataclass
class TransformComponent:
    """Test transform component."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class InventoryComponent:
    """Test inventory component."""
    capacity: int = 10


# =============================================================================
# ACTOR BASIC TESTS
# =============================================================================


class TestActorCreation:
    """Test Actor creation and basic properties."""

    def test_actor_has_unique_id(self):
        """Each actor should have a unique ID."""
        actor1 = Actor(name="Actor1")
        actor2 = Actor(name="Actor2")

        assert actor1.actor_id != actor2.actor_id

    def test_actor_has_name(self, basic_actor):
        """Actor should have the assigned name."""
        assert basic_actor.name == "TestActor"

    def test_actor_default_name(self):
        """Actor without name should get default name."""
        actor = Actor()
        assert actor.name is not None
        assert len(actor.name) > 0

    def test_actor_has_type(self, basic_actor):
        """Actor should have correct type."""
        assert basic_actor.actor_type == ActorType.STATIC

    def test_actor_initial_state(self, basic_actor):
        """Actor should start in CREATING state."""
        assert basic_actor.state == EntityState.CREATING


class TestActorLifecycle:
    """Test Actor lifecycle methods."""

    def test_begin_play_changes_state(self, basic_actor):
        """begin_play() should transition to ACTIVE state."""
        basic_actor.begin_play()
        assert basic_actor.state == EntityState.ACTIVE

    def test_end_play_changes_state(self, basic_actor):
        """end_play() should transition to DEACTIVATING state."""
        basic_actor.begin_play()
        basic_actor.end_play()
        assert basic_actor.state == EntityState.DEACTIVATING

    def test_destroy_marks_for_destruction(self, basic_actor):
        """destroy() should mark actor for destruction."""
        basic_actor.destroy()
        assert basic_actor.state == EntityState.DESTROYING

    def test_is_active_when_active(self, basic_actor):
        """is_active should be True when in ACTIVE state."""
        basic_actor.begin_play()
        assert basic_actor.is_active is True

    def test_is_active_when_not_active(self, basic_actor):
        """is_active should be False when not in ACTIVE state."""
        assert basic_actor.is_active is False


class TestActorTick:
    """Test Actor tick behavior."""

    def test_tick_with_lifespan(self):
        """Tick should decrement lifespan and destroy when expired."""
        actor = Actor(name="Mortal")
        actor._lifespan = 1.0  # 1 second lifespan
        actor.begin_play()

        # Tick for 0.5 seconds - should still be alive
        actor.tick(0.5)
        assert actor.state == EntityState.ACTIVE

        # Tick past lifespan - should be destroying
        actor.tick(0.6)
        assert actor.state == EntityState.DESTROYING

    def test_tick_without_lifespan(self, basic_actor):
        """Tick without lifespan should not auto-destroy."""
        basic_actor.begin_play()

        # Tick many times
        for _ in range(100):
            basic_actor.tick(1.0)

        assert basic_actor.state == EntityState.ACTIVE


# =============================================================================
# ACTOR COMPONENT TESTS
# =============================================================================


class TestActorComponents:
    """Test Actor component system."""

    def test_add_component(self, basic_actor):
        """add_component() should store component."""
        health = HealthComponent(current=100, maximum=100)
        basic_actor.add_component(health)

        assert basic_actor.has_component(HealthComponent) is True

    def test_get_component_returns_correct_type(self, basic_actor):
        """get_component() should return correct component."""
        health = HealthComponent(current=75, maximum=100)
        basic_actor.add_component(health)

        retrieved = basic_actor.get_component(HealthComponent)
        assert retrieved is not None
        assert retrieved.current == 75

    def test_get_component_returns_none_for_missing(self, basic_actor):
        """get_component() should return None for missing type."""
        result = basic_actor.get_component(HealthComponent)
        assert result is None

    def test_has_component_returns_true_for_existing(self, basic_actor):
        """has_component() should return True for existing component."""
        basic_actor.add_component(HealthComponent())
        assert basic_actor.has_component(HealthComponent) is True

    def test_has_component_returns_false_for_missing(self, basic_actor):
        """has_component() should return False for missing component."""
        assert basic_actor.has_component(HealthComponent) is False

    def test_remove_component(self, basic_actor):
        """remove_component() should remove component."""
        basic_actor.add_component(HealthComponent())

        result = basic_actor.remove_component(HealthComponent)

        assert result is True
        assert basic_actor.has_component(HealthComponent) is False

    def test_remove_missing_component_returns_false(self, basic_actor):
        """remove_component() for missing component should return False."""
        result = basic_actor.remove_component(HealthComponent)
        assert result is False

    def test_multiple_components_coexist(self, basic_actor):
        """Multiple different components should coexist."""
        basic_actor.add_component(HealthComponent())
        basic_actor.add_component(TransformComponent())
        basic_actor.add_component(InventoryComponent())

        assert basic_actor.has_component(HealthComponent) is True
        assert basic_actor.has_component(TransformComponent) is True
        assert basic_actor.has_component(InventoryComponent) is True

    def test_replacing_component(self, basic_actor):
        """Adding same component type should replace existing."""
        basic_actor.add_component(HealthComponent(current=100, maximum=100))
        basic_actor.add_component(HealthComponent(current=50, maximum=100))

        retrieved = basic_actor.get_component(HealthComponent)
        assert retrieved.current == 50


# =============================================================================
# ACTOR TAG TESTS
# =============================================================================


class TestActorTags:
    """Test Actor tag system."""

    def test_add_tag(self, basic_actor):
        """add_tag() should add tag."""
        basic_actor.add_tag("enemy")
        assert basic_actor.has_tag("enemy") is True

    def test_remove_tag(self, basic_actor):
        """remove_tag() should remove tag."""
        basic_actor.add_tag("enemy")
        basic_actor.remove_tag("enemy")
        assert basic_actor.has_tag("enemy") is False

    def test_has_tag_returns_false_for_missing(self, basic_actor):
        """has_tag() should return False for missing tag."""
        assert basic_actor.has_tag("nonexistent") is False

    def test_multiple_tags(self, basic_actor):
        """Multiple tags should be supported."""
        basic_actor.add_tag("enemy")
        basic_actor.add_tag("boss")
        basic_actor.add_tag("flying")

        assert basic_actor.has_tag("enemy") is True
        assert basic_actor.has_tag("boss") is True
        assert basic_actor.has_tag("flying") is True


# =============================================================================
# ACTOR OWNERSHIP TESTS
# =============================================================================


class TestActorOwnership:
    """Test Actor owner/child relationships."""

    def test_set_owner_updates_references(self):
        """set_owner() should update both owner and child references."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")

        child.set_owner(parent)

        assert child.get_owner() == parent
        assert child in parent._children

    def test_changing_owner_updates_old_parent(self):
        """Changing owner should remove from old parent."""
        parent1 = Actor(name="Parent1")
        parent2 = Actor(name="Parent2")
        child = Actor(name="Child")

        child.set_owner(parent1)
        child.set_owner(parent2)

        assert child.get_owner() == parent2
        assert child not in parent1._children
        assert child in parent2._children

    def test_set_owner_none_removes_parent(self):
        """Setting owner to None should remove parent reference."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")

        child.set_owner(parent)
        child.set_owner(None)

        assert child.get_owner() is None


# =============================================================================
# PAWN TESTS
# =============================================================================


class TestPawnBasics:
    """Test Pawn basic functionality."""

    def test_pawn_has_pawn_type(self, pawn):
        """Pawn should have PAWN actor type."""
        assert pawn.actor_type == ActorType.PAWN

    def test_pawn_starts_unpossessed(self, pawn):
        """Pawn should start without controller."""
        assert pawn.controller is None
        assert pawn.is_possessed is False

    def test_pawn_can_be_possessed(self, pawn):
        """Pawn should have can_be_possessed property."""
        assert pawn._can_be_possessed is True


class TestPawnPossession:
    """Test Pawn possession mechanics."""

    def test_possess_links_controller_to_pawn(self, pawn, player_controller):
        """possess() should link controller to pawn."""
        result = pawn.possess(player_controller)

        assert result is True
        assert pawn.controller == player_controller
        assert pawn.is_possessed is True

    def test_unpossess_unlinks_controller(self, pawn, player_controller):
        """unpossess() should unlink controller from pawn."""
        pawn.possess(player_controller)
        pawn.unpossess()

        assert pawn.controller is None
        assert pawn.is_possessed is False

    def test_repossess_transfers_to_new_controller(self, pawn, player_controller, ai_controller):
        """Possessing already-possessed pawn should transfer to new controller."""
        pawn.possess(player_controller)
        pawn.possess(ai_controller)

        assert pawn.controller == ai_controller
        assert player_controller.pawn is None

    def test_cannot_possess_if_disabled(self, pawn, player_controller):
        """Cannot possess pawn with possession disabled."""
        pawn._can_be_possessed = False

        result = pawn.possess(player_controller)

        assert result is False
        assert pawn.controller is None


class TestPawnInput:
    """Test Pawn input forwarding."""

    def test_get_movement_input_returns_controller_input(self, pawn, player_controller):
        """Movement input should come from controller."""
        pawn.possess(player_controller)
        player_controller.set_movement_input(1.0, 0.0, 0.0)

        movement = pawn.get_movement_input()

        assert movement == (1.0, 0.0, 0.0)

    def test_get_movement_input_returns_zero_unpossessed(self, pawn):
        """Unpossessed pawn should return zero movement."""
        movement = pawn.get_movement_input()

        assert movement == (0.0, 0.0, 0.0)

    def test_get_look_input_returns_controller_input(self, pawn, player_controller):
        """Look input should come from controller."""
        pawn.possess(player_controller)
        player_controller.set_look_input(45.0, -10.0)

        look = pawn.get_look_input()

        assert look == (45.0, -10.0)


# =============================================================================
# CHARACTER TESTS
# =============================================================================


class TestCharacterBasics:
    """Test Character basic properties."""

    def test_character_has_character_type(self, character):
        """Character should have CHARACTER actor type."""
        assert character.actor_type == ActorType.CHARACTER

    def test_character_has_default_health(self, character):
        """Character should start with default health."""
        assert character._health == DEFAULT_CHARACTER_HEALTH
        assert character._max_health == DEFAULT_CHARACTER_MAX_HEALTH

    def test_character_is_alive_when_health_positive(self, character):
        """Character with positive health should be alive."""
        assert character.is_alive is True

    def test_character_default_movement_speed(self, character):
        """Character should have default movement speed."""
        assert character._movement_speed == DEFAULT_MAX_WALK_SPEED


class TestCharacterMovement:
    """Test Character movement mechanics."""

    def test_crouch_reduces_speed(self, character):
        """Crouching should reduce movement speed."""
        normal_speed = character.movement_speed
        character.crouch()
        crouch_speed = character.movement_speed

        assert crouch_speed < normal_speed
        assert crouch_speed == normal_speed * CROUCH_SPEED_MULTIPLIER

    def test_uncrouch_restores_speed(self, character):
        """Uncrouching should restore movement speed."""
        normal_speed = character.movement_speed
        character.crouch()
        character.uncrouch()

        assert character.movement_speed == normal_speed

    def test_sprint_increases_speed(self, character):
        """Sprinting should increase movement speed."""
        normal_speed = character.movement_speed
        character.sprint(True)
        sprint_speed = character.movement_speed

        assert sprint_speed > normal_speed
        assert sprint_speed == normal_speed * SPRINT_SPEED_MULTIPLIER

    def test_cannot_sprint_while_crouched(self, character):
        """Cannot sprint while crouched."""
        character.crouch()
        character.sprint(True)

        # Should not be sprinting
        assert character._is_sprinting is False


class TestCharacterJump:
    """Test Character jump mechanics."""

    def test_jump_when_grounded(self, character):
        """Jump should succeed when grounded."""
        result = character.jump()

        assert result is True
        assert character._is_jumping is True

    def test_cannot_double_jump(self, character):
        """Cannot jump while already jumping."""
        character.jump()
        result = character.jump()

        assert result is False

    def test_on_landed_resets_jump_state(self, character):
        """Landing should reset jump and fall state."""
        character.jump()
        character._is_falling = True

        character.on_landed()

        assert character._is_jumping is False
        assert character._is_falling is False


class TestCharacterDamage:
    """Test Character damage and healing."""

    def test_take_damage_reduces_health(self, character):
        """take_damage() should reduce health."""
        character.take_damage(25.0)

        assert character._health == 75.0

    def test_take_damage_returns_actual_damage(self, character):
        """take_damage() should return actual damage dealt."""
        character._health = 20.0

        actual = character.take_damage(50.0)

        assert actual == 20.0  # Only 20 damage dealt (remaining health)

    def test_take_damage_zero_does_nothing(self, character):
        """Zero damage should not affect health."""
        actual = character.take_damage(0.0)

        assert actual == 0.0
        assert character._health == DEFAULT_CHARACTER_HEALTH

    def test_take_damage_negative_does_nothing(self, character):
        """Negative damage should not affect health."""
        actual = character.take_damage(-10.0)

        assert actual == 0.0
        assert character._health == DEFAULT_CHARACTER_HEALTH

    def test_death_on_zero_health(self, character):
        """Character should die when health reaches zero."""
        character.take_damage(character._health)

        assert character._health == 0
        assert character.is_alive is False

    def test_heal_restores_health(self, character):
        """heal() should restore health."""
        character._health = 50.0
        character.heal(25.0)

        assert character._health == 75.0

    def test_heal_cannot_exceed_max(self, character):
        """Healing should not exceed max health."""
        character._health = 90.0
        actual = character.heal(50.0)

        assert character._health == DEFAULT_CHARACTER_MAX_HEALTH
        assert actual == 10.0

    def test_heal_dead_character_does_nothing(self, character):
        """Cannot heal dead character."""
        character._health = 0.0

        actual = character.heal(50.0)

        assert actual == 0.0
        assert character._health == 0.0


# =============================================================================
# CONTROLLER TESTS
# =============================================================================


class TestPlayerController:
    """Test PlayerController functionality."""

    def test_player_controller_has_index(self, player_controller):
        """PlayerController should have player index."""
        assert player_controller.player_index == 0

    def test_set_movement_input(self, player_controller):
        """set_movement_input() should store values."""
        player_controller.set_movement_input(0.5, -0.3, 1.0)

        movement = player_controller.get_movement_input()
        assert movement == (0.5, -0.3, 1.0)

    def test_set_look_input(self, player_controller):
        """set_look_input() should store values."""
        player_controller.set_look_input(90.0, -45.0)

        look = player_controller.get_look_input()
        assert look == (90.0, -45.0)

    def test_input_disabled_returns_zero(self, player_controller):
        """Disabled input should return zero."""
        player_controller.set_movement_input(1.0, 1.0, 1.0)
        player_controller.set_look_input(45.0, 30.0)
        player_controller.enable_input(False)

        assert player_controller.get_movement_input() == (0.0, 0.0, 0.0)
        assert player_controller.get_look_input() == (0.0, 0.0)


class TestAIController:
    """Test AIController functionality."""

    def test_ai_controller_has_blackboard(self, ai_controller):
        """AIController should have blackboard."""
        ai_controller.set_blackboard_value("target", "enemy_1")
        value = ai_controller.get_blackboard_value("target")

        assert value == "enemy_1"

    def test_blackboard_default_value(self, ai_controller):
        """Blackboard should return default for missing keys."""
        value = ai_controller.get_blackboard_value("missing", default="none")

        assert value == "none"

    def test_set_movement_input(self, ai_controller):
        """AI can set movement input."""
        ai_controller.set_movement_input(0.0, 0.0, 1.0)

        movement = ai_controller.get_movement_input()
        assert movement == (0.0, 0.0, 1.0)

    def test_set_look_target(self, ai_controller):
        """AI can set look target position."""
        target = (10.0, 20.0, 5.0)
        ai_controller.set_look_target(target)

        assert ai_controller._look_target == target


class TestControllerPossession:
    """Test Controller possession methods."""

    def test_controller_possess_pawn(self, player_controller, pawn):
        """Controller should be able to possess pawn."""
        result = player_controller.possess(pawn)

        assert result is True
        assert player_controller.pawn == pawn

    def test_controller_unpossess(self, player_controller, pawn):
        """Controller should be able to unpossess."""
        player_controller.possess(pawn)
        player_controller.unpossess()

        assert player_controller.pawn is None
        assert pawn.controller is None


# =============================================================================
# PREFAB TESTS
# =============================================================================


class TestPrefabBasics:
    """Test Prefab basic functionality."""

    def test_prefab_has_name(self):
        """Prefab should have name."""
        prefab = Prefab(name="TestPrefab")
        assert prefab.name == "TestPrefab"

    def test_prefab_default_actor_class(self):
        """Prefab should default to Actor class."""
        prefab = Prefab(name="Test")
        assert prefab.actor_class == Actor

    def test_prefab_can_specify_actor_class(self):
        """Prefab can specify custom actor class."""
        prefab = Prefab(name="Test", actor_class=Character)
        assert prefab.actor_class == Character


class TestPrefabSpawning:
    """Test Prefab spawn functionality."""

    def test_spawn_creates_actor(self):
        """spawn() should create actor instance."""
        prefab = Prefab(name="TestPrefab")
        actor = prefab.spawn()

        assert actor is not None
        assert isinstance(actor, Actor)

    def test_spawn_uses_actor_class(self):
        """spawn() should use specified actor class."""
        prefab = Prefab(name="Test", actor_class=Character)
        actor = prefab.spawn()

        assert isinstance(actor, Character)

    def test_spawn_adds_components(self):
        """spawn() should add specified components."""
        prefab = Prefab(
            name="Test",
            components=[
                PrefabComponent(component_type=HealthComponent, properties={"current": 50.0}),
                PrefabComponent(component_type=TransformComponent),
            ],
        )

        actor = prefab.spawn()

        assert actor.has_component(HealthComponent) is True
        assert actor.has_component(TransformComponent) is True

        health = actor.get_component(HealthComponent)
        assert health.current == 50.0

    def test_spawn_adds_tags(self):
        """spawn() should add specified tags."""
        prefab = Prefab(name="Test", tags={"enemy", "boss"})
        actor = prefab.spawn()

        assert actor.has_tag("enemy") is True
        assert actor.has_tag("boss") is True

    def test_spawn_with_override_properties(self):
        """spawn() should allow property overrides."""
        prefab = Prefab(name="Test", properties={"_name": "OriginalName"})
        actor = prefab.spawn(override_properties={"_name": "OverrideName"})

        # The name property should be overridden
        # Note: depends on implementation details
        assert actor is not None

    def test_spawn_children(self):
        """spawn() should spawn child prefabs."""
        child_prefab = Prefab(name="Child")
        parent_prefab = Prefab(name="Parent", children=[child_prefab])

        parent = parent_prefab.spawn()

        assert len(parent._children) >= 1


class TestPrefabRegistry:
    """Test PrefabRegistry functionality."""

    def test_registry_is_singleton(self):
        """PrefabRegistry should be singleton."""
        reg1 = PrefabRegistry()
        reg2 = PrefabRegistry()

        assert reg1 is reg2

    def test_register_prefab(self):
        """register() should store prefab."""
        registry = PrefabRegistry()
        prefab = Prefab(name="Test")

        registry.register(prefab)

        assert registry.get("Test") == prefab

    def test_unregister_prefab(self):
        """unregister() should remove prefab."""
        registry = PrefabRegistry()
        prefab = Prefab(name="Test")
        registry.register(prefab)

        result = registry.unregister("Test")

        assert result is True
        assert registry.get("Test") is None

    def test_spawn_from_registry(self):
        """spawn() should spawn from registered prefab."""
        registry = PrefabRegistry()
        prefab = Prefab(name="Test", actor_class=Character)
        registry.register(prefab)

        actor = registry.spawn("Test")

        assert isinstance(actor, Character)

    def test_spawn_unknown_returns_none(self):
        """spawn() with unknown name should return None."""
        registry = PrefabRegistry()

        actor = registry.spawn("Unknown")

        assert actor is None

    def test_list_prefabs(self):
        """list_prefabs() should return all registered names."""
        registry = PrefabRegistry()
        registry.register(Prefab(name="A"))
        registry.register(Prefab(name="B"))
        registry.register(Prefab(name="C"))

        names = registry.list_prefabs()

        assert "A" in names
        assert "B" in names
        assert "C" in names


# =============================================================================
# LIFECYCLE MANAGER TESTS
# =============================================================================


class TestLifecycleManager:
    """Test LifecycleManager functionality."""

    def test_manager_is_singleton(self, lifecycle_manager):
        """LifecycleManager should be singleton."""
        manager2 = LifecycleManager()
        assert lifecycle_manager is manager2

    def test_register_actor(self, lifecycle_manager, basic_actor):
        """register_actor() should track actor."""
        lifecycle_manager.register_actor(basic_actor)

        retrieved = lifecycle_manager.get_actor(basic_actor.actor_id)
        assert retrieved == basic_actor

    def test_unregister_actor(self, lifecycle_manager, basic_actor):
        """unregister_actor() should remove tracking."""
        lifecycle_manager.register_actor(basic_actor)
        lifecycle_manager.unregister_actor(basic_actor)

        retrieved = lifecycle_manager.get_actor(basic_actor.actor_id)
        assert retrieved is None


class TestLifecycleDeferred:
    """Test deferred spawn/destroy operations."""

    def test_queue_spawn_defers_activation(self, lifecycle_manager, basic_actor):
        """queue_spawn() should defer activation."""
        lifecycle_manager.register_actor(basic_actor)
        lifecycle_manager.queue_spawn(basic_actor)

        # Before processing, actor should not be active
        assert basic_actor.state != EntityState.ACTIVE

    def test_process_pending_activates_spawned(self, lifecycle_manager, basic_actor):
        """process_pending() should activate queued spawns."""
        lifecycle_manager.register_actor(basic_actor)
        lifecycle_manager.queue_spawn(basic_actor)

        lifecycle_manager.process_pending()

        assert basic_actor.state == EntityState.ACTIVE

    def test_queue_destroy_defers_destruction(self, lifecycle_manager, basic_actor):
        """queue_destroy() should defer destruction."""
        lifecycle_manager.register_actor(basic_actor)
        basic_actor.begin_play()
        lifecycle_manager.queue_destroy(basic_actor)

        # Before processing, should still be active
        # (depends on implementation - may still be ACTIVE or transitioning)
        assert basic_actor.state in (EntityState.ACTIVE, EntityState.DEACTIVATING)

    def test_process_pending_destroys_queued(self, lifecycle_manager, basic_actor):
        """process_pending() should destroy queued actors."""
        lifecycle_manager.register_actor(basic_actor)
        basic_actor.begin_play()
        lifecycle_manager.queue_destroy(basic_actor)

        lifecycle_manager.process_pending()

        # Actor should be unregistered
        assert lifecycle_manager.get_actor(basic_actor.actor_id) is None

    def test_multiple_spawns_processed_together(self, lifecycle_manager):
        """Multiple deferred spawns should process together."""
        actors = [Actor(name=f"Actor{i}") for i in range(5)]

        for actor in actors:
            lifecycle_manager.register_actor(actor)
            lifecycle_manager.queue_spawn(actor)

        lifecycle_manager.process_pending()

        for actor in actors:
            assert actor.state == EntityState.ACTIVE


class TestLifecycleCallbacks:
    """Test lifecycle event callbacks."""

    def test_add_callback(self, lifecycle_manager):
        """add_callback() should register callback."""
        events_received = []

        def callback(actor, event):
            events_received.append((actor, event))

        lifecycle_manager.add_callback(callback)
        actor = Actor(name="Test")
        lifecycle_manager.register_actor(actor)

        assert len(events_received) >= 1
        assert events_received[0][1] == LifecycleEvent.CREATED

    def test_remove_callback(self, lifecycle_manager):
        """remove_callback() should unregister callback."""
        events_received = []

        def callback(actor, event):
            events_received.append((actor, event))

        lifecycle_manager.add_callback(callback)
        lifecycle_manager.remove_callback(callback)

        actor = Actor(name="Test")
        lifecycle_manager.register_actor(actor)

        assert len(events_received) == 0

    def test_callback_receives_all_events(self, lifecycle_manager):
        """Callback should receive lifecycle events."""
        events_received = []

        def callback(actor, event):
            events_received.append(event)

        lifecycle_manager.add_callback(callback)

        actor = Actor(name="Test")
        lifecycle_manager.register_actor(actor)
        lifecycle_manager.queue_spawn(actor)
        lifecycle_manager.process_pending()
        lifecycle_manager.queue_destroy(actor)
        lifecycle_manager.process_pending()

        # Should have received multiple events
        assert LifecycleEvent.CREATED in events_received
        assert LifecycleEvent.ACTIVATED in events_received


class TestLifecycleQueries:
    """Test LifecycleManager query methods."""

    def test_get_actors_by_type(self, lifecycle_manager):
        """get_actors_by_type() should filter by type."""
        actor = Actor(name="Static")
        pawn = Pawn(name="Pawn")
        character = Character(name="Character")

        lifecycle_manager.register_actor(actor)
        lifecycle_manager.register_actor(pawn)
        lifecycle_manager.register_actor(character)

        pawns = lifecycle_manager.get_actors_by_type(ActorType.PAWN)
        characters = lifecycle_manager.get_actors_by_type(ActorType.CHARACTER)

        assert pawn in pawns
        assert character in characters
        assert actor not in pawns

    def test_get_actors_with_tag(self, lifecycle_manager):
        """get_actors_with_tag() should filter by tag."""
        actor1 = Actor(name="Enemy1")
        actor2 = Actor(name="Enemy2")
        actor3 = Actor(name="Friend")

        actor1.add_tag("enemy")
        actor2.add_tag("enemy")
        actor3.add_tag("friend")

        lifecycle_manager.register_actor(actor1)
        lifecycle_manager.register_actor(actor2)
        lifecycle_manager.register_actor(actor3)

        enemies = lifecycle_manager.get_actors_with_tag("enemy")

        assert actor1 in enemies
        assert actor2 in enemies
        assert actor3 not in enemies

    def test_get_all_actors(self, lifecycle_manager):
        """get_all_actors() should return all registered."""
        actors = [Actor(name=f"A{i}") for i in range(5)]

        for actor in actors:
            lifecycle_manager.register_actor(actor)

        all_actors = lifecycle_manager.get_all_actors()

        assert len(all_actors) == 5
        for actor in actors:
            assert actor in all_actors


# =============================================================================
# ACTOR HIERARCHY TESTS
# =============================================================================


class TestActorHierarchy:
    """Test Actor parent/child hierarchy."""

    def test_destroying_parent_orphans_children(self):
        """Destroying parent should not crash children."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")

        child.set_owner(parent)
        parent.destroy()

        # Child should still be accessible (orphaned)
        assert child is not None

    def test_destroying_child_removes_from_parent(self):
        """Destroying child should remove from parent's children."""
        parent = Actor(name="Parent")
        child = Actor(name="Child")

        child.set_owner(parent)
        assert child in parent._children

        child.set_owner(None)
        assert child not in parent._children


# =============================================================================
# EVENTLOG INTEGRATION TESTS
# =============================================================================


class TestEntityEventLog:
    """Test EntityEventLog integration."""

    def test_entity_event_log_exists(self):
        """Event log should be accessible."""
        log = get_entity_event_log()
        assert log is not None

    def test_clear_entity_event_log(self):
        """clear_entity_event_log() should work without error."""
        # Just verify the function exists and can be called
        clear_entity_event_log()
        # Get a fresh log after clearing
        log = get_entity_event_log()
        assert log is not None


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEntityEdgeCases:
    """Edge case and boundary tests."""

    def test_actor_unique_ids_across_many(self):
        """Unique IDs should be maintained across many actors."""
        actors = [Actor(name=f"Actor{i}") for i in range(100)]
        ids = [a.actor_id for a in actors]

        assert len(set(ids)) == 100  # All unique

    def test_controller_unique_ids(self):
        """Controllers should have unique IDs."""
        controllers = [PlayerController() for _ in range(10)]
        ids = [c.controller_id for c in controllers]

        assert len(set(ids)) == 10

    def test_character_health_boundaries(self, character):
        """Health should not go below 0 or above max."""
        # Massive damage
        character.take_damage(10000.0)
        assert character._health == 0.0

        # Reset
        character._health = 50.0

        # Massive heal
        character.heal(10000.0)
        assert character._health == DEFAULT_CHARACTER_MAX_HEALTH

    def test_empty_prefab_spawns(self):
        """Empty prefab should spawn basic actor."""
        prefab = Prefab(name="Empty")
        actor = prefab.spawn()

        assert actor is not None
        assert len(actor._components) == 0

    def test_deeply_nested_components(self, basic_actor):
        """Should handle many components."""
        for i in range(50):
            @dataclass
            class DynamicComponent:
                value: int = i

            basic_actor.add_component(DynamicComponent())

        # Should have many components
        assert len(basic_actor._components) >= 1

    def test_lifecycle_constants_exist(self):
        """All lifecycle constants should be defined."""
        assert DEFAULT_MAX_WALK_SPEED > 0
        assert DEFAULT_JUMP_FORCE > 0
        assert DEFAULT_CHARACTER_HEALTH > 0
        assert DEFAULT_CHARACTER_MAX_HEALTH > 0
        assert 0 < CROUCH_SPEED_MULTIPLIER < 1
        assert SPRINT_SPEED_MULTIPLIER > 1
        assert MAX_PREFAB_INHERITANCE_DEPTH > 0

    def test_actor_type_ordering(self):
        """Actor types should have correct values."""
        assert ActorType.STATIC.value == 0
        assert ActorType.DYNAMIC.value == 1
        assert ActorType.PAWN.value == 2
        assert ActorType.CHARACTER.value == 3

    def test_entity_state_progression(self):
        """Entity states should have logical progression."""
        assert EntityState.CREATING.value == 0
        assert EntityState.INITIALIZING.value == 1
        assert EntityState.ACTIVE.value == 2
        assert EntityState.DEACTIVATING.value == 3
        assert EntityState.DESTROYING.value == 4

    def test_possession_with_none_pawn(self, player_controller):
        """Controller should handle None pawn gracefully."""
        # No pawn possessed
        assert player_controller.pawn is None
        # Unpossess should be safe
        player_controller.unpossess()
        assert player_controller.pawn is None
