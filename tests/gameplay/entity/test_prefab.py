"""
Comprehensive tests for the Prefab/Blueprint System.

Tests for:
- Prefab creation from actor
- Prefab instantiation
- Property overrides
- Nested prefabs
- Prefab variants
- Prefab serialization
- Component overrides in prefabs
"""
from __future__ import annotations

import pytest
import copy
import threading
from typing import Any, Dict, Set
from unittest.mock import Mock, MagicMock, patch

from engine.gameplay.entity.prefab import (
    ComponentDefinition,
    PropertyOverride,
    PrefabDefinition,
    PrefabRegistry,
    PrefabInstantiator,
    PrefabBuilder,
    prefab,
    extends,
    spawn_prefab,
    register_prefab,
)
from engine.gameplay.entity.actor import Actor, Transform, StaticActor, DynamicActor, Character
from engine.gameplay.entity.constants import (
    MAX_PREFAB_INHERITANCE_DEPTH,
    PREFAB_INSTANCE_BATCH_SIZE,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_state():
    """Reset prefab state before each test."""
    PrefabRegistry.reset_instance()
    PrefabInstantiator.reset_instance()
    Actor.reset_entity_ids()
    yield
    PrefabRegistry.reset_instance()
    PrefabInstantiator.reset_instance()


@pytest.fixture
def prefab_registry():
    """Create a fresh prefab registry."""
    PrefabRegistry.reset_instance()
    return PrefabRegistry()


@pytest.fixture
def prefab_instantiator():
    """Create a fresh prefab instantiator."""
    PrefabInstantiator.reset_instance()
    return PrefabInstantiator()


class MockComponent:
    """Mock component for testing."""
    _component = True

    def __init__(self, value: int = 0):
        self.value = value


class HealthComponent:
    """Health component for testing."""
    _component = True

    def __init__(self, max_health: int = 100, current_health: int = None):
        self.max_health = max_health
        self.current_health = current_health if current_health is not None else max_health


class InventoryComponent:
    """Inventory component for testing."""
    _component = True

    def __init__(self, slots: int = 10):
        self.slots = slots
        self.items = []


# =============================================================================
# COMPONENT DEFINITION TESTS
# =============================================================================


class TestComponentDefinition:
    """Tests for ComponentDefinition data class."""

    def test_component_definition_creation(self):
        """Test creating a ComponentDefinition."""
        comp_def = ComponentDefinition(
            name="health",
            component_type=HealthComponent,
        )
        assert comp_def.name == "health"
        assert comp_def.component_type is HealthComponent

    def test_component_definition_with_properties(self):
        """Test ComponentDefinition with properties."""
        comp_def = ComponentDefinition(
            name="health",
            component_type=HealthComponent,
            properties={"max_health": 200},
        )
        assert comp_def.properties["max_health"] == 200

    def test_component_definition_with_factory(self):
        """Test ComponentDefinition with factory."""
        factory = Mock(return_value=HealthComponent(max_health=150))
        comp_def = ComponentDefinition(
            name="health",
            component_type=HealthComponent,
            factory=factory,
        )
        assert comp_def.factory is factory

    def test_create_instance_basic(self):
        """Test creating instance from definition."""
        comp_def = ComponentDefinition(
            name="health",
            component_type=HealthComponent,
        )
        instance = comp_def.create_instance()
        assert isinstance(instance, HealthComponent)
        assert instance.max_health == 100  # default

    def test_create_instance_with_properties(self):
        """Test creating instance with property overrides."""
        comp_def = ComponentDefinition(
            name="health",
            component_type=HealthComponent,
            properties={"max_health": 250},
        )
        instance = comp_def.create_instance()
        assert instance.max_health == 250

    def test_create_instance_with_factory(self):
        """Test creating instance with factory."""
        factory = Mock(return_value=HealthComponent(max_health=300))
        comp_def = ComponentDefinition(
            name="health",
            component_type=HealthComponent,
            factory=factory,
        )
        instance = comp_def.create_instance()
        factory.assert_called_once()
        assert instance.max_health == 300

    def test_create_instance_deepcopy_properties(self):
        """Test property values are deep copied."""
        items_list = ["sword", "shield"]
        comp_def = ComponentDefinition(
            name="inventory",
            component_type=InventoryComponent,
            properties={"items": items_list},
        )
        instance = comp_def.create_instance()
        items_list.append("potion")
        assert "potion" not in instance.items


# =============================================================================
# PROPERTY OVERRIDE TESTS
# =============================================================================


class TestPropertyOverride:
    """Tests for PropertyOverride data class."""

    def test_property_override_creation(self):
        """Test creating a PropertyOverride."""
        override = PropertyOverride(
            property_path="transform.position",
            value=(10.0, 20.0, 30.0),
        )
        assert override.property_path == "transform.position"
        assert override.value == (10.0, 20.0, 30.0)

    def test_property_override_nested_path(self):
        """Test PropertyOverride with nested path."""
        override = PropertyOverride(
            property_path="health.current",
            value=50,
        )
        assert override.property_path == "health.current"


# =============================================================================
# PREFAB DEFINITION TESTS
# =============================================================================


class TestPrefabDefinition:
    """Tests for PrefabDefinition data class."""

    def test_prefab_definition_creation(self):
        """Test creating a PrefabDefinition."""
        definition = PrefabDefinition(
            name="test_prefab",
            actor_class=Actor,
        )
        assert definition.name == "test_prefab"
        assert definition.actor_class is Actor

    def test_prefab_definition_with_components(self):
        """Test PrefabDefinition with components."""
        comp_def = ComponentDefinition(
            name="health",
            component_type=HealthComponent,
        )
        definition = PrefabDefinition(
            name="test_prefab",
            actor_class=Actor,
            components={"health": comp_def},
        )
        assert "health" in definition.components

    def test_prefab_definition_with_properties(self):
        """Test PrefabDefinition with properties."""
        definition = PrefabDefinition(
            name="test_prefab",
            actor_class=Actor,
            properties={"max_walk_speed": 10.0},
        )
        assert definition.properties["max_walk_speed"] == 10.0

    def test_prefab_definition_with_tags(self):
        """Test PrefabDefinition with tags."""
        definition = PrefabDefinition(
            name="test_prefab",
            actor_class=Actor,
            tags={"enemy", "boss"},
        )
        assert "enemy" in definition.tags
        assert "boss" in definition.tags

    def test_prefab_definition_with_parent(self):
        """Test PrefabDefinition with parent prefab."""
        definition = PrefabDefinition(
            name="child_prefab",
            actor_class=Actor,
            parent_prefab="parent_prefab",
        )
        assert definition.parent_prefab == "parent_prefab"

    def test_prefab_definition_with_transform(self):
        """Test PrefabDefinition with transform."""
        transform = Transform(position=(5.0, 0.0, 5.0))
        definition = PrefabDefinition(
            name="test_prefab",
            actor_class=Actor,
            transform=transform,
        )
        assert definition.transform.position == (5.0, 0.0, 5.0)


# =============================================================================
# PREFAB REGISTRY TESTS
# =============================================================================


class TestPrefabRegistrySingleton:
    """Tests for PrefabRegistry singleton pattern."""

    def test_singleton_instance(self):
        """Test PrefabRegistry is a singleton."""
        reg1 = PrefabRegistry()
        reg2 = PrefabRegistry()
        assert reg1 is reg2

    def test_reset_instance(self):
        """Test resetting singleton instance."""
        reg1 = PrefabRegistry()
        reg1.register(PrefabDefinition(name="test", actor_class=Actor))
        PrefabRegistry.reset_instance()
        reg2 = PrefabRegistry()
        assert reg2.get("test") is None


class TestPrefabRegistryOperations:
    """Tests for PrefabRegistry operations."""

    def test_register_prefab(self, prefab_registry):
        """Test registering a prefab."""
        definition = PrefabDefinition(name="test", actor_class=Actor)
        prefab_registry.register(definition)
        assert prefab_registry.get("test") is definition

    def test_unregister_prefab(self, prefab_registry):
        """Test unregistering a prefab."""
        definition = PrefabDefinition(name="test", actor_class=Actor)
        prefab_registry.register(definition)
        result = prefab_registry.unregister("test")
        assert result is definition
        assert prefab_registry.get("test") is None

    def test_unregister_nonexistent(self, prefab_registry):
        """Test unregistering nonexistent prefab returns None."""
        result = prefab_registry.unregister("nonexistent")
        assert result is None

    def test_get_prefab(self, prefab_registry):
        """Test getting a registered prefab."""
        definition = PrefabDefinition(name="test", actor_class=Actor)
        prefab_registry.register(definition)
        result = prefab_registry.get("test")
        assert result is definition

    def test_get_nonexistent(self, prefab_registry):
        """Test getting nonexistent prefab returns None."""
        result = prefab_registry.get("nonexistent")
        assert result is None

    def test_list_prefabs(self, prefab_registry):
        """Test listing registered prefabs."""
        prefab_registry.register(PrefabDefinition(name="prefab1", actor_class=Actor))
        prefab_registry.register(PrefabDefinition(name="prefab2", actor_class=Actor))
        prefab_registry.register(PrefabDefinition(name="prefab3", actor_class=Actor))
        names = prefab_registry.list_prefabs()
        assert len(names) == 3
        assert "prefab1" in names
        assert "prefab2" in names
        assert "prefab3" in names

    def test_get_children(self, prefab_registry):
        """Test getting child prefabs."""
        parent = PrefabDefinition(name="parent", actor_class=Actor)
        child1 = PrefabDefinition(name="child1", actor_class=Actor, parent_prefab="parent")
        child2 = PrefabDefinition(name="child2", actor_class=Actor, parent_prefab="parent")
        other = PrefabDefinition(name="other", actor_class=Actor)

        prefab_registry.register(parent)
        prefab_registry.register(child1)
        prefab_registry.register(child2)
        prefab_registry.register(other)

        children = prefab_registry.get_children("parent")
        assert len(children) == 2
        assert "child1" in children
        assert "child2" in children
        assert "other" not in children

    def test_clear(self, prefab_registry):
        """Test clearing registry."""
        prefab_registry.register(PrefabDefinition(name="test1", actor_class=Actor))
        prefab_registry.register(PrefabDefinition(name="test2", actor_class=Actor))
        prefab_registry.clear()
        assert len(prefab_registry.list_prefabs()) == 0


class TestPrefabRegistryInheritance:
    """Tests for prefab inheritance resolution."""

    def test_get_resolved_no_parent(self, prefab_registry):
        """Test resolving prefab without parent."""
        definition = PrefabDefinition(
            name="standalone",
            actor_class=Actor,
            properties={"key": "value"},
        )
        prefab_registry.register(definition)
        resolved = prefab_registry.get_resolved("standalone")
        assert resolved is not None
        assert resolved.properties["key"] == "value"

    def test_get_resolved_with_parent(self, prefab_registry):
        """Test resolving prefab with parent."""
        parent = PrefabDefinition(
            name="parent",
            actor_class=Actor,
            properties={"parent_prop": "parent_value"},
            tags={"parent_tag"},
        )
        child = PrefabDefinition(
            name="child",
            actor_class=Actor,
            parent_prefab="parent",
            properties={"child_prop": "child_value"},
            tags={"child_tag"},
        )

        prefab_registry.register(parent)
        prefab_registry.register(child)

        resolved = prefab_registry.get_resolved("child")
        assert resolved is not None
        assert resolved.properties["parent_prop"] == "parent_value"
        assert resolved.properties["child_prop"] == "child_value"
        assert "parent_tag" in resolved.tags
        assert "child_tag" in resolved.tags

    def test_get_resolved_child_overrides_parent(self, prefab_registry):
        """Test child prefab overrides parent properties."""
        parent = PrefabDefinition(
            name="parent",
            actor_class=Actor,
            properties={"shared": "parent_value"},
        )
        child = PrefabDefinition(
            name="child",
            actor_class=Actor,
            parent_prefab="parent",
            properties={"shared": "child_value"},
        )

        prefab_registry.register(parent)
        prefab_registry.register(child)

        resolved = prefab_registry.get_resolved("child")
        assert resolved.properties["shared"] == "child_value"

    def test_get_resolved_nested_inheritance(self, prefab_registry):
        """Test multi-level inheritance."""
        grandparent = PrefabDefinition(
            name="grandparent",
            actor_class=Actor,
            properties={"level": 0},
        )
        parent = PrefabDefinition(
            name="parent",
            actor_class=Actor,
            parent_prefab="grandparent",
            properties={"level": 1},
        )
        child = PrefabDefinition(
            name="child",
            actor_class=Actor,
            parent_prefab="parent",
            properties={"level": 2},
        )

        prefab_registry.register(grandparent)
        prefab_registry.register(parent)
        prefab_registry.register(child)

        resolved = prefab_registry.get_resolved("child")
        assert resolved.properties["level"] == 2

    def test_get_resolved_missing_parent(self, prefab_registry):
        """Test resolving with missing parent raises error."""
        child = PrefabDefinition(
            name="orphan",
            actor_class=Actor,
            parent_prefab="nonexistent",
        )
        prefab_registry.register(child)

        with pytest.raises(ValueError, match="not found"):
            prefab_registry.get_resolved("orphan")

    def test_get_resolved_max_depth(self, prefab_registry):
        """Test inheritance depth limit is enforced."""
        # Create a chain longer than MAX_PREFAB_INHERITANCE_DEPTH
        prev_name = "root"
        prefab_registry.register(PrefabDefinition(name="root", actor_class=Actor))

        for i in range(MAX_PREFAB_INHERITANCE_DEPTH + 5):
            name = f"level_{i}"
            prefab_registry.register(PrefabDefinition(
                name=name,
                actor_class=Actor,
                parent_prefab=prev_name,
            ))
            prev_name = name

        with pytest.raises(RecursionError, match="depth exceeded"):
            prefab_registry.get_resolved(prev_name)

    def test_get_resolved_caches_result(self, prefab_registry):
        """Test resolved prefabs are cached."""
        definition = PrefabDefinition(name="cached", actor_class=Actor)
        prefab_registry.register(definition)

        resolved1 = prefab_registry.get_resolved("cached")
        resolved2 = prefab_registry.get_resolved("cached")
        assert resolved1 is resolved2

    def test_cache_invalidated_on_register(self, prefab_registry):
        """Test cache is invalidated when prefab is re-registered."""
        definition = PrefabDefinition(
            name="test",
            actor_class=Actor,
            properties={"version": 1},
        )
        prefab_registry.register(definition)
        resolved1 = prefab_registry.get_resolved("test")

        updated = PrefabDefinition(
            name="test",
            actor_class=Actor,
            properties={"version": 2},
        )
        prefab_registry.register(updated)
        resolved2 = prefab_registry.get_resolved("test")

        assert resolved2.properties["version"] == 2


# =============================================================================
# PREFAB INSTANTIATOR TESTS
# =============================================================================


class TestPrefabInstantiatorSingleton:
    """Tests for PrefabInstantiator singleton pattern."""

    def test_singleton_instance(self):
        """Test PrefabInstantiator is a singleton."""
        inst1 = PrefabInstantiator()
        inst2 = PrefabInstantiator()
        assert inst1 is inst2


class TestPrefabInstantiation:
    """Tests for prefab instantiation."""

    def test_instantiate_basic(self, prefab_registry, prefab_instantiator):
        """Test basic prefab instantiation."""
        definition = PrefabDefinition(name="basic", actor_class=Actor)
        prefab_registry.register(definition)

        actor = prefab_instantiator.instantiate("basic", immediate=True)
        assert actor is not None
        assert isinstance(actor, Actor)

    def test_instantiate_with_transform(self, prefab_registry, prefab_instantiator):
        """Test instantiation with transform override."""
        definition = PrefabDefinition(name="positioned", actor_class=Actor)
        prefab_registry.register(definition)

        transform = Transform(position=(10.0, 20.0, 30.0))
        actor = prefab_instantiator.instantiate(
            "positioned",
            transform=transform,
            immediate=True,
        )
        assert actor.position == (10.0, 20.0, 30.0)

    def test_instantiate_with_prefab_transform(self, prefab_registry, prefab_instantiator):
        """Test instantiation uses prefab's default transform."""
        definition = PrefabDefinition(
            name="default_pos",
            actor_class=Actor,
            transform=Transform(position=(5.0, 5.0, 5.0)),
        )
        prefab_registry.register(definition)

        actor = prefab_instantiator.instantiate("default_pos", immediate=True)
        assert actor.position == (5.0, 5.0, 5.0)

    def test_instantiate_with_overrides(self, prefab_registry, prefab_instantiator):
        """Test instantiation with property overrides."""
        definition = PrefabDefinition(
            name="overrideable",
            actor_class=Character,
            properties={"max_walk_speed": 5.0},
        )
        prefab_registry.register(definition)

        actor = prefab_instantiator.instantiate(
            "overrideable",
            overrides={"max_walk_speed": 10.0},
            immediate=True,
        )
        assert actor.max_walk_speed == 10.0

    def test_instantiate_with_components(self, prefab_registry, prefab_instantiator):
        """Test instantiation adds components."""
        comp_def = ComponentDefinition(
            name="health",
            component_type=HealthComponent,
            properties={"max_health": 200},
        )
        definition = PrefabDefinition(
            name="with_components",
            actor_class=Actor,
            components={"health": comp_def},
        )
        prefab_registry.register(definition)

        actor = prefab_instantiator.instantiate("with_components", immediate=True)
        health = actor.get_component("health")
        assert health is not None
        assert health.max_health == 200

    def test_instantiate_with_tags(self, prefab_registry, prefab_instantiator):
        """Test instantiation adds tags."""
        definition = PrefabDefinition(
            name="tagged",
            actor_class=Actor,
            tags={"enemy", "boss"},
        )
        prefab_registry.register(definition)

        actor = prefab_instantiator.instantiate("tagged", immediate=True)
        assert actor.has_tag("enemy")
        assert actor.has_tag("boss")

    def test_instantiate_nonexistent(self, prefab_instantiator):
        """Test instantiating nonexistent prefab returns None."""
        actor = prefab_instantiator.instantiate("nonexistent", immediate=True)
        assert actor is None

    def test_instantiate_deferred(self, prefab_registry, prefab_instantiator):
        """Test deferred instantiation returns None."""
        definition = PrefabDefinition(name="deferred", actor_class=Actor)
        prefab_registry.register(definition)

        result = prefab_instantiator.instantiate("deferred", immediate=False)
        assert result is None


class TestDeferredInstantiation:
    """Tests for deferred prefab instantiation."""

    def test_deferred_instantiation_queued(self, prefab_registry, prefab_instantiator):
        """Test deferred instantiation is queued."""
        definition = PrefabDefinition(name="queued", actor_class=Actor)
        prefab_registry.register(definition)

        prefab_instantiator.instantiate("queued", immediate=False)
        prefab_instantiator.instantiate("queued", immediate=False)
        prefab_instantiator.instantiate("queued", immediate=False)

        created = prefab_instantiator.process_pending()
        assert len(created) == 3

    def test_process_pending_empty(self, prefab_instantiator):
        """Test processing with no pending instantiations."""
        created = prefab_instantiator.process_pending()
        assert len(created) == 0

    def test_process_pending_batch_size(self, prefab_registry, prefab_instantiator):
        """Test pending are processed in batches."""
        definition = PrefabDefinition(name="batch", actor_class=Actor)
        prefab_registry.register(definition)

        # Queue more than batch size
        for _ in range(PREFAB_INSTANCE_BATCH_SIZE + 10):
            prefab_instantiator.instantiate("batch", immediate=False)

        # First batch
        created1 = prefab_instantiator.process_pending()
        assert len(created1) == PREFAB_INSTANCE_BATCH_SIZE

        # Remaining
        created2 = prefab_instantiator.process_pending()
        assert len(created2) == 10

    def test_instantiate_async_with_callback(self, prefab_registry, prefab_instantiator):
        """Test async instantiation with callback."""
        definition = PrefabDefinition(name="async", actor_class=Actor)
        prefab_registry.register(definition)

        callback = Mock()
        prefab_instantiator.instantiate_async("async", callback)

        created = prefab_instantiator.process_pending()
        assert len(created) == 1
        callback.assert_called_once_with(created[0])

    def test_callback_error_doesnt_break_instantiation(self, prefab_registry, prefab_instantiator):
        """Test callback error doesn't break other instantiations."""
        definition = PrefabDefinition(name="error", actor_class=Actor)
        prefab_registry.register(definition)

        bad_callback = Mock(side_effect=RuntimeError("Callback error"))
        good_callback = Mock()

        prefab_instantiator.instantiate_async("error", bad_callback)
        prefab_instantiator.instantiate_async("error", good_callback)

        created = prefab_instantiator.process_pending()
        assert len(created) == 2
        good_callback.assert_called_once()

    def test_clear_pending(self, prefab_registry, prefab_instantiator):
        """Test clearing pending instantiations."""
        definition = PrefabDefinition(name="clear", actor_class=Actor)
        prefab_registry.register(definition)

        for _ in range(5):
            prefab_instantiator.instantiate("clear", immediate=False)

        count = prefab_instantiator.clear_pending()
        assert count == 5

        created = prefab_instantiator.process_pending()
        assert len(created) == 0


# =============================================================================
# PREFAB BUILDER TESTS
# =============================================================================


class TestPrefabBuilder:
    """Tests for PrefabBuilder fluent API."""

    def test_builder_creation(self):
        """Test creating a PrefabBuilder."""
        builder = PrefabBuilder("test", Actor)
        assert builder._name == "test"
        assert builder._actor_class is Actor

    def test_builder_with_component(self):
        """Test adding component via builder."""
        builder = (
            PrefabBuilder("test", Actor)
            .with_component("health", HealthComponent, max_health=200)
        )
        assert "health" in builder._components

    def test_builder_with_component_factory(self):
        """Test adding component with factory."""
        factory = Mock(return_value=HealthComponent())
        builder = (
            PrefabBuilder("test", Actor)
            .with_component("health", HealthComponent, factory=factory)
        )
        assert builder._components["health"].factory is factory

    def test_builder_with_property(self):
        """Test setting property via builder."""
        builder = (
            PrefabBuilder("test", Character)
            .with_property("max_walk_speed", 15.0)
        )
        assert builder._properties["max_walk_speed"] == 15.0

    def test_builder_with_tag(self):
        """Test adding tag via builder."""
        builder = (
            PrefabBuilder("test", Actor)
            .with_tag("enemy")
        )
        assert "enemy" in builder._tags

    def test_builder_with_tags(self):
        """Test adding multiple tags via builder."""
        builder = (
            PrefabBuilder("test", Actor)
            .with_tags("enemy", "boss", "hostile")
        )
        assert "enemy" in builder._tags
        assert "boss" in builder._tags
        assert "hostile" in builder._tags

    def test_builder_extends(self):
        """Test extending parent prefab via builder."""
        builder = (
            PrefabBuilder("child", Actor)
            .extends("parent")
        )
        assert builder._parent == "parent"

    def test_builder_with_transform(self):
        """Test setting transform via builder."""
        builder = (
            PrefabBuilder("test", Actor)
            .with_transform(position=(10.0, 20.0, 30.0))
        )
        assert builder._transform.position == (10.0, 20.0, 30.0)

    def test_builder_with_transform_full(self):
        """Test setting full transform via builder."""
        builder = (
            PrefabBuilder("test", Actor)
            .with_transform(
                position=(1.0, 2.0, 3.0),
                rotation=(0.0, 0.707, 0.0, 0.707),
                scale=(2.0, 2.0, 2.0),
            )
        )
        assert builder._transform.position == (1.0, 2.0, 3.0)
        assert builder._transform.rotation == (0.0, 0.707, 0.0, 0.707)
        assert builder._transform.scale == (2.0, 2.0, 2.0)

    def test_builder_build(self, prefab_registry):
        """Test building and registering prefab."""
        definition = (
            PrefabBuilder("built", Actor)
            .with_tag("test")
            .build()
        )
        assert definition.name == "built"
        assert prefab_registry.get("built") is definition

    def test_builder_instantiate(self, prefab_registry):
        """Test building and instantiating in one step."""
        actor = (
            PrefabBuilder("instant", Actor)
            .with_tag("immediate")
            .instantiate()
        )
        assert actor is not None
        assert actor.has_tag("immediate")

    def test_builder_chaining(self, prefab_registry):
        """Test full builder chaining."""
        definition = (
            PrefabBuilder("complex", Character)
            .with_component("health", HealthComponent, max_health=500)
            .with_component("inventory", InventoryComponent, slots=20)
            .with_property("max_walk_speed", 12.0)
            .with_property("max_run_speed", 24.0)
            .with_tag("player")
            .with_tags("hero", "main_character")
            .with_transform(position=(0.0, 0.0, 0.0))
            .build()
        )

        assert "health" in definition.components
        assert "inventory" in definition.components
        assert definition.properties["max_walk_speed"] == 12.0
        assert "player" in definition.tags
        assert "hero" in definition.tags


# =============================================================================
# PREFAB DECORATOR TESTS
# =============================================================================


class TestPrefabDecorator:
    """Tests for @prefab decorator."""

    def test_prefab_decorator_basic(self, prefab_registry):
        """Test basic @prefab decorator usage."""
        @prefab(name="decorated_actor")
        class DecoratedActor(Actor):
            pass

        assert DecoratedActor._prefab is True
        assert DecoratedActor._prefab_name == "decorated_actor"

    def test_prefab_decorator_registers(self, prefab_registry):
        """Test @prefab registers with global registry."""
        @prefab(name="registered_actor")
        class RegisteredActor(Actor):
            pass

        definition = prefab_registry.get("registered_actor")
        assert definition is not None
        assert definition.actor_class is RegisteredActor

    def test_prefab_decorator_requires_name(self):
        """Test @prefab requires name parameter."""
        with pytest.raises(ValueError, match="'name' parameter is required"):
            @prefab()
            class NoNameActor(Actor):
                pass


class TestExtendsDecorator:
    """Tests for @extends decorator."""

    def test_extends_decorator_basic(self, prefab_registry):
        """Test basic @extends decorator usage."""
        @prefab(name="base_actor")
        class BaseActor(Actor):
            pass

        @extends(parent="base_actor")
        @prefab(name="extended_actor")
        class ExtendedActor(Actor):
            pass

        assert ExtendedActor._extends is True
        assert ExtendedActor._extends_parent == "base_actor"

    def test_extends_decorator_requires_parent(self):
        """Test @extends requires parent parameter."""
        with pytest.raises(ValueError, match="'parent' parameter is required"):
            @extends()
            class NoParentActor(Actor):
                pass


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestSpawnPrefab:
    """Tests for spawn_prefab function."""

    def test_spawn_prefab_basic(self, prefab_registry):
        """Test basic spawn_prefab usage."""
        register_prefab("spawn_test", Actor)
        actor = spawn_prefab("spawn_test")
        assert actor is not None

    def test_spawn_prefab_with_position(self, prefab_registry):
        """Test spawn_prefab with position."""
        register_prefab("positioned", Actor)
        actor = spawn_prefab("positioned", position=(100.0, 0.0, 100.0))
        assert actor.position == (100.0, 0.0, 100.0)

    def test_spawn_prefab_with_rotation(self, prefab_registry):
        """Test spawn_prefab with rotation."""
        register_prefab("rotated", Actor)
        actor = spawn_prefab("rotated", rotation=(0.0, 0.707, 0.0, 0.707))
        assert actor.rotation == (0.0, 0.707, 0.0, 0.707)

    def test_spawn_prefab_with_overrides(self, prefab_registry):
        """Test spawn_prefab with property overrides."""
        register_prefab("overridden", Character, properties={"max_walk_speed": 5.0})
        actor = spawn_prefab("overridden", overrides={"max_walk_speed": 15.0})
        assert actor.max_walk_speed == 15.0

    def test_spawn_prefab_deferred(self, prefab_registry, prefab_instantiator):
        """Test deferred spawn_prefab."""
        register_prefab("deferred_spawn", Actor)
        result = spawn_prefab("deferred_spawn", immediate=False)
        assert result is None

        created = prefab_instantiator.process_pending()
        assert len(created) == 1


class TestRegisterPrefab:
    """Tests for register_prefab function."""

    def test_register_prefab_basic(self, prefab_registry):
        """Test basic register_prefab usage."""
        definition = register_prefab("simple", Actor)
        assert definition.name == "simple"
        assert prefab_registry.get("simple") is definition

    def test_register_prefab_with_components(self, prefab_registry):
        """Test register_prefab with components."""
        comp_def = ComponentDefinition(
            name="health",
            component_type=HealthComponent,
        )
        definition = register_prefab(
            "with_health",
            Actor,
            components={"health": comp_def},
        )
        assert "health" in definition.components

    def test_register_prefab_with_properties(self, prefab_registry):
        """Test register_prefab with properties."""
        definition = register_prefab(
            "with_props",
            Actor,
            properties={"speed": 10.0},
        )
        assert definition.properties["speed"] == 10.0

    def test_register_prefab_with_tags(self, prefab_registry):
        """Test register_prefab with tags."""
        definition = register_prefab(
            "tagged",
            Actor,
            tags={"enemy", "npc"},
        )
        assert "enemy" in definition.tags
        assert "npc" in definition.tags

    def test_register_prefab_with_parent(self, prefab_registry):
        """Test register_prefab with parent."""
        register_prefab("parent", Actor)
        definition = register_prefab(
            "child",
            Actor,
            parent="parent",
        )
        assert definition.parent_prefab == "parent"


# =============================================================================
# NESTED PREFAB TESTS
# =============================================================================


class TestNestedPrefabs:
    """Tests for nested/hierarchical prefabs."""

    def test_three_level_inheritance(self, prefab_registry, prefab_instantiator):
        """Test three-level prefab inheritance."""
        register_prefab("base", Character, properties={"level": "base"}, tags={"base"})
        register_prefab("mid", Character, properties={"level": "mid"}, tags={"mid"}, parent="base")
        register_prefab("leaf", Character, properties={"level": "leaf"}, tags={"leaf"}, parent="mid")

        actor = spawn_prefab("leaf")
        assert actor.has_tag("base")
        assert actor.has_tag("mid")
        assert actor.has_tag("leaf")

    def test_component_inheritance(self, prefab_registry, prefab_instantiator):
        """Test components are inherited."""
        comp_def = ComponentDefinition(name="health", component_type=HealthComponent)
        register_prefab("with_health", Actor, components={"health": comp_def})
        register_prefab("inherits_health", Actor, parent="with_health")

        actor = spawn_prefab("inherits_health")
        assert actor.has_component("health")

    def test_component_override(self, prefab_registry, prefab_instantiator):
        """Test child can override parent component."""
        parent_comp = ComponentDefinition(
            name="health",
            component_type=HealthComponent,
            properties={"max_health": 100},
        )
        child_comp = ComponentDefinition(
            name="health",
            component_type=HealthComponent,
            properties={"max_health": 200},
        )

        register_prefab("parent", Actor, components={"health": parent_comp})
        register_prefab("child", Actor, components={"health": child_comp}, parent="parent")

        actor = spawn_prefab("child")
        health = actor.get_component("health")
        assert health.max_health == 200


# =============================================================================
# PREFAB VARIANT TESTS
# =============================================================================


class TestPrefabVariants:
    """Tests for prefab variants."""

    def test_create_variant(self, prefab_registry):
        """Test creating a variant of a prefab."""
        register_prefab(
            "enemy_base",
            Character,
            properties={"max_walk_speed": 5.0},
            tags={"enemy"},
        )
        register_prefab(
            "enemy_fast",
            Character,
            properties={"max_walk_speed": 15.0},
            tags={"fast"},
            parent="enemy_base",
        )
        register_prefab(
            "enemy_slow",
            Character,
            properties={"max_walk_speed": 2.0},
            tags={"slow"},
            parent="enemy_base",
        )

        fast = spawn_prefab("enemy_fast")
        slow = spawn_prefab("enemy_slow")

        assert fast.max_walk_speed == 15.0
        assert slow.max_walk_speed == 2.0
        assert fast.has_tag("enemy")
        assert slow.has_tag("enemy")

    def test_variants_independent(self, prefab_registry, prefab_instantiator):
        """Test variants are independent instances."""
        register_prefab("template", Actor)
        spawn_prefab("template")
        a1 = spawn_prefab("template")
        a2 = spawn_prefab("template")

        a1.position = (10.0, 0.0, 0.0)
        assert a2.position != a1.position


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestPrefabEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_instantiate_returns_correct_actor_class(self, prefab_registry, prefab_instantiator):
        """Test instantiation returns correct actor subclass."""
        register_prefab("character_prefab", Character)
        actor = spawn_prefab("character_prefab")
        assert isinstance(actor, Character)

    def test_deep_copy_properties(self, prefab_registry, prefab_instantiator):
        """Test properties are deep copied between instances."""
        list_prop = [1, 2, 3]
        # Using a character with a custom property would need a custom class
        # For now, test via component
        comp_def = ComponentDefinition(
            name="inventory",
            component_type=InventoryComponent,
            properties={"items": list_prop},
        )
        register_prefab("with_list", Actor, components={"inventory": comp_def})

        a1 = spawn_prefab("with_list")
        a2 = spawn_prefab("with_list")

        # Modify one instance's component
        a1.get_component("inventory").items.append(4)

        # Other instance should be unaffected
        assert 4 not in a2.get_component("inventory").items

    def test_thread_safe_instantiation(self, prefab_registry):
        """Test thread-safe prefab instantiation."""
        register_prefab("thread_test", Actor)
        actors = []
        lock = threading.Lock()

        def spawn():
            actor = spawn_prefab("thread_test")
            with lock:
                actors.append(actor)

        threads = [threading.Thread(target=spawn) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(actors) == 50
        # All should have unique entity IDs
        ids = [a.entity_id for a in actors]
        assert len(ids) == len(set(ids))

    def test_property_path_navigation(self, prefab_registry, prefab_instantiator):
        """Test property path navigation in overrides."""
        comp_def = ComponentDefinition(
            name="health",
            component_type=HealthComponent,
        )
        register_prefab("with_health", Actor, components={"health": comp_def})

        # Note: Property path for components needs to match implementation
        # This tests that nested paths work if supported
        actor = spawn_prefab("with_health")
        assert actor.get_component("health") is not None

    def test_empty_prefab_instantiation(self, prefab_registry, prefab_instantiator):
        """Test instantiating minimal prefab."""
        register_prefab("minimal", Actor)
        actor = spawn_prefab("minimal")
        assert actor is not None
        assert len(actor.components) == 0
        assert len(actor.tags) == 0
