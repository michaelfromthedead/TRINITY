"""
Comprehensive tests for ShellLang module.

Tests cover:
    - Core primitives (Entity, World, Snapshot, Change)
    - Sugar layer (EntityProxy, QueryResult, TypeQuery, TimeManager)
    - AI interface (execute, validate, dry_run)
    - REPL (Shell, Feedback)
"""

import math
import pytest
from dataclasses import dataclass
from typing import Any

# =============================================================================
# TEST CONSTANTS (imported from modules under test)
# =============================================================================

from foundation.shelllang.core import ENTITY_ID_START, DEFAULT_HISTORY_COUNT
from foundation.shelllang.sugar import MAX_DISPLAY_ENTITIES, MAX_UNDO_STACK
from foundation.shelllang.ai import DEFAULT_QUERY_LIMIT


# =============================================================================
# TEST COMPONENTS
# =============================================================================


@dataclass
class Health:
    """Test health component."""
    current: int = 100
    maximum: int = 100


@dataclass
class Position:
    """Test position component."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Enemy:
    """Test enemy marker component."""
    name: str = "goblin"
    level: int = 1


@dataclass
class Player:
    """Test player marker component."""
    name: str = "hero"


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def world():
    """Create a fresh world for each test."""
    from foundation.shelllang.core import World
    return World()


@pytest.fixture
def populated_world(world):
    """Create a world with some entities."""
    # Create player
    player = world.create()
    world.attach(player, Player("Hero"))
    world.attach(player, Position(0, 0, 0))
    world.attach(player, Health(100, 100))

    # Create enemies
    enemy1 = world.create()
    world.attach(enemy1, Enemy("Goblin", 1))
    world.attach(enemy1, Position(5, 0, 0))
    world.attach(enemy1, Health(30, 30))

    enemy2 = world.create()
    world.attach(enemy2, Enemy("Orc", 3))
    world.attach(enemy2, Position(10, 0, 0))
    world.attach(enemy2, Health(80, 80))

    enemy3 = world.create()
    world.attach(enemy3, Enemy("Dragon", 10))
    world.attach(enemy3, Position(100, 50, 0))
    world.attach(enemy3, Health(500, 500))

    return world


@pytest.fixture
def registry():
    """Create a component registry."""
    return {
        "Health": Health,
        "Position": Position,
        "Enemy": Enemy,
        "Player": Player,
    }


@pytest.fixture
def sugar_setup(populated_world, registry):
    """Set up sugar module with world and registry."""
    from foundation.shelllang import sugar
    sugar.set_world(populated_world)
    sugar.set_registry(registry)
    # Use a no-op echo for tests
    sugar.set_echo(lambda msg: None)
    return populated_world, registry


@pytest.fixture
def ai_interface(populated_world, registry):
    """Create an AI interface."""
    from foundation.shelllang.ai import AIInterface
    return AIInterface(populated_world, registry)


# =============================================================================
# CORE TESTS
# =============================================================================


class TestEntity:
    """Tests for Entity class."""

    def test_entity_creation(self, world):
        """Test entity creation returns unique IDs."""
        e1 = world.create()
        e2 = world.create()
        assert e1.id != e2.id
        assert e1.id >= ENTITY_ID_START
        assert e2.id > e1.id

    def test_entity_equality(self):
        """Test entity equality by ID."""
        from foundation.shelllang.core import Entity
        e1 = Entity(1)
        e2 = Entity(1)
        e3 = Entity(2)
        assert e1 == e2
        assert e1 != e3

    def test_entity_hash(self):
        """Test entity can be used in sets/dicts."""
        from foundation.shelllang.core import Entity
        e1 = Entity(1)
        e2 = Entity(1)
        s = {e1}
        assert e2 in s

    def test_entity_repr(self):
        """Test entity string representation."""
        from foundation.shelllang.core import Entity
        e = Entity(42)
        assert "42" in repr(e)


class TestWorld:
    """Tests for World class."""

    def test_create_and_exists(self, world):
        """Test entity creation and existence check."""
        e = world.create()
        assert world.exists(e)

    def test_destroy(self, world):
        """Test entity destruction."""
        e = world.create()
        assert world.exists(e)
        world.destroy(e)
        assert not world.exists(e)

    def test_attach_and_get(self, world):
        """Test component attachment and retrieval."""
        e = world.create()
        health = Health(50, 100)
        world.attach(e, health)

        retrieved = world.get(e, Health)
        assert retrieved is not None
        assert retrieved.current == 50
        assert retrieved.maximum == 100

    def test_has_component(self, world):
        """Test component existence check."""
        e = world.create()
        assert not world.has(e, Health)

        world.attach(e, Health())
        assert world.has(e, Health)

    def test_detach(self, world):
        """Test component detachment."""
        e = world.create()
        world.attach(e, Health())
        assert world.has(e, Health)

        world.detach(e, Health)
        assert not world.has(e, Health)

    def test_set_tracks_changes(self, world):
        """Test that set() tracks changes."""
        e = world.create()
        world.attach(e, Health(100, 100))

        world.set(e, Health, "current", 50)

        changes = world.recent_changes(1)
        assert len(changes) == 1
        assert changes[0].old_value == 100
        assert changes[0].new_value == 50

    def test_query_single_component(self, populated_world):
        """Test querying for single component."""
        enemies = populated_world.query(Enemy)
        assert len(enemies) == 3

    def test_query_multiple_components(self, populated_world):
        """Test querying for multiple components."""
        # All entities with both Health and Position
        entities = populated_world.query(Health, Position)
        assert len(entities) == 4  # 1 player + 3 enemies

    def test_entity_count(self, populated_world):
        """Test entity count."""
        assert populated_world.entity_count() == 4

    def test_entities_iterator(self, populated_world):
        """Test iterating over all entities."""
        entities = list(populated_world.entities)
        assert len(entities) == 4

    def test_components_of(self, populated_world):
        """Test getting component types of an entity."""
        from foundation.shelllang.core import Entity
        # First entity (player) has Player, Position, Health
        player = Entity(1)
        components = populated_world.components_of(player)
        assert len(components) == 3


class TestSnapshot:
    """Tests for snapshot operations."""

    def test_snap_creates_snapshot(self, populated_world):
        """Test creating a snapshot."""
        snap = populated_world.snap("test")
        assert snap.name == "test"
        assert len(snap.entities) == 4

    def test_restore_reverts_changes(self, populated_world):
        """Test restoring from snapshot."""
        from foundation.shelllang.core import Entity

        # Take snapshot
        snap = populated_world.snap()

        # Make changes
        player = Entity(1)
        populated_world.set(player, Health, "current", 1)

        # Verify change
        health = populated_world.get(player, Health)
        assert health.current == 1

        # Restore
        populated_world.restore(snap)

        # Verify restored
        health = populated_world.get(player, Health)
        assert health.current == 100

    def test_diff_detects_changes(self, populated_world):
        """Test diff between snapshots."""
        from foundation.shelllang.core import Entity

        snap_before = populated_world.snap()

        player = Entity(1)
        populated_world.set(player, Health, "current", 50)

        snap_after = populated_world.snap()

        changes = populated_world.diff(snap_before, snap_after)
        assert len(changes) >= 1

        health_changes = [c for c in changes if c.field_name == "current"]
        assert len(health_changes) == 1
        assert health_changes[0].old_value == 100
        assert health_changes[0].new_value == 50


class TestChange:
    """Tests for Change tracking."""

    def test_change_repr(self):
        """Test Change string representation."""
        from foundation.shelllang.core import Change
        change = Change(
            entity_id=1,
            component_type="Health",
            field_name="current",
            old_value=100,
            new_value=50,
        )
        repr_str = repr(change)
        assert "Health" in repr_str
        assert "current" in repr_str
        assert "100" in repr_str
        assert "50" in repr_str


class TestConstants:
    """Tests verifying constants are respected."""

    def test_entity_id_starts_at_constant(self, world):
        """Test entity IDs start at ENTITY_ID_START."""
        e = world.create()
        assert e.id == ENTITY_ID_START

    def test_recent_changes_default_limit(self, world):
        """Test recent_changes uses DEFAULT_HISTORY_COUNT."""
        e = world.create()
        world.attach(e, Health(100, 100))

        # Make more changes than default limit
        for i in range(DEFAULT_HISTORY_COUNT + 5):
            world.set(e, Health, "current", i)

        # Default should return DEFAULT_HISTORY_COUNT
        changes = world.recent_changes()
        assert len(changes) == DEFAULT_HISTORY_COUNT

    def test_query_limit_default(self, ai_interface, populated_world):
        """Test AI query uses DEFAULT_QUERY_LIMIT."""
        # Create more entities than the default limit
        for i in range(DEFAULT_QUERY_LIMIT + 10):
            e = populated_world.create()
            populated_world.attach(e, Health(i, i))

        result = ai_interface.execute({
            "op": "query",
            "components": ["Health"],
        })

        # Should be capped at DEFAULT_QUERY_LIMIT
        assert result["count"] == DEFAULT_QUERY_LIMIT

    def test_query_result_set_requires_double_underscore(self, sugar_setup):
        """Test QueryResult.set() requires component__field format."""
        from foundation.shelllang.sugar import QueryResult

        world, _ = sugar_setup
        entities = list(world.query(Health))
        result = QueryResult(entities)

        # Should raise ValueError for invalid format
        with pytest.raises(ValueError) as exc_info:
            result.set(invalid_format=100)

        assert "component__field" in str(exc_info.value)


# =============================================================================
# SUGAR TESTS
# =============================================================================


class TestEntityProxy:
    """Tests for EntityProxy sugar."""

    def test_proxy_component_access(self, sugar_setup):
        """Test accessing components via proxy."""
        from foundation.shelllang.sugar import EntityProxy
        from foundation.shelllang.core import Entity

        world, _ = sugar_setup
        proxy = EntityProxy(Entity(1))

        # Access health component
        assert proxy.health.current == 100
        assert proxy.health.maximum == 100

    def test_proxy_component_mutation(self, sugar_setup):
        """Test mutating components via proxy."""
        from foundation.shelllang.sugar import EntityProxy
        from foundation.shelllang.core import Entity

        world, _ = sugar_setup
        proxy = EntityProxy(Entity(1))

        proxy.health.current = 50
        assert proxy.health.current == 50

    def test_proxy_id(self, sugar_setup):
        """Test proxy exposes entity ID."""
        from foundation.shelllang.sugar import EntityProxy
        from foundation.shelllang.core import Entity

        proxy = EntityProxy(Entity(42))
        assert proxy.id == 42

    def test_proxy_equality(self, sugar_setup):
        """Test proxy equality."""
        from foundation.shelllang.sugar import EntityProxy
        from foundation.shelllang.core import Entity

        p1 = EntityProxy(Entity(1))
        p2 = EntityProxy(Entity(1))
        p3 = EntityProxy(Entity(2))

        assert p1 == p2
        assert p1 != p3
        assert p1 == 1  # Can compare to int


class TestQueryResult:
    """Tests for QueryResult sugar."""

    def test_query_result_where(self, sugar_setup):
        """Test filtering with where()."""
        from foundation.shelllang.sugar import QueryResult

        world, _ = sugar_setup
        entities = list(world.query(Enemy))
        result = QueryResult(entities)

        # Filter low-level enemies
        low_level = result.where(lambda e: e.enemy.level < 5)
        assert low_level.count() == 2  # Goblin (1) and Orc (3)

    def test_query_result_without(self, sugar_setup):
        """Test excluding components with without()."""
        from foundation.shelllang.sugar import QueryResult

        world, _ = sugar_setup
        entities = list(world.query(Health))
        result = QueryResult(entities)

        # Exclude enemies
        non_enemies = result.without(Enemy)
        assert non_enemies.count() == 1  # Just the player

    def test_query_result_near(self, sugar_setup):
        """Test distance filtering with near()."""
        from foundation.shelllang.sugar import QueryResult, EntityProxy
        from foundation.shelllang.core import Entity

        world, _ = sugar_setup
        enemies = list(world.query(Enemy))
        result = QueryResult(enemies)

        player = EntityProxy(Entity(1))
        nearby = result.near(player, 15)  # Within 15 units

        # Goblin at (5,0,0) and Orc at (10,0,0) are nearby
        # Dragon at (100,50,0) is far
        assert nearby.count() == 2

    def test_query_result_first(self, sugar_setup):
        """Test first() accessor."""
        from foundation.shelllang.sugar import QueryResult

        world, _ = sugar_setup
        entities = list(world.query(Enemy))
        result = QueryResult(entities)

        first = result.first()
        assert first is not None
        assert hasattr(first, 'enemy')

    def test_query_result_ids(self, sugar_setup):
        """Test ids() accessor."""
        from foundation.shelllang.sugar import QueryResult

        world, _ = sugar_setup
        entities = list(world.query(Enemy))
        result = QueryResult(entities)

        ids = result.ids()
        assert len(ids) == 3
        assert all(isinstance(i, int) for i in ids)

    def test_query_result_iteration(self, sugar_setup):
        """Test iterating over QueryResult."""
        from foundation.shelllang.sugar import QueryResult, EntityProxy

        world, _ = sugar_setup
        entities = list(world.query(Enemy))
        result = QueryResult(entities)

        for e in result:
            assert isinstance(e, EntityProxy)

    def test_query_result_indexing(self, sugar_setup):
        """Test indexing QueryResult."""
        from foundation.shelllang.sugar import QueryResult, EntityProxy

        world, _ = sugar_setup
        entities = list(world.query(Enemy))
        result = QueryResult(entities)

        first = result[0]
        assert isinstance(first, EntityProxy)


class TestTypeQuery:
    """Tests for TypeQuery sugar."""

    def test_type_query_all(self, sugar_setup):
        """Test TypeQuery.all property."""
        from foundation.shelllang.sugar import TypeQuery

        query = TypeQuery(Enemy)
        result = query.all

        assert result.count() == 3

    def test_type_query_where(self, sugar_setup):
        """Test TypeQuery.where() shortcut."""
        from foundation.shelllang.sugar import TypeQuery

        query = TypeQuery(Enemy)
        result = query.where(lambda e: e.enemy.level >= 3)

        assert result.count() == 2  # Orc and Dragon

    def test_type_query_count(self, sugar_setup):
        """Test TypeQuery.count() shortcut."""
        from foundation.shelllang.sugar import TypeQuery

        query = TypeQuery(Enemy)
        assert query.count() == 3


class TestTimeManager:
    """Tests for TimeManager sugar."""

    def test_mark_and_rewind(self, sugar_setup):
        """Test mark() and rewind()."""
        from foundation.shelllang.sugar import TimeManager, EntityProxy
        from foundation.shelllang.core import Entity

        world, _ = sugar_setup
        time = TimeManager()

        # Mark initial state
        time.mark("start")

        # Make changes
        player = EntityProxy(Entity(1))
        player.health.current = 1
        assert player.health.current == 1

        # Rewind
        time.rewind("start")

        # Verify restored
        assert player.health.current == 100

    def test_undo_redo(self, sugar_setup):
        """Test undo() and redo()."""
        from foundation.shelllang.sugar import TimeManager, EntityProxy
        from foundation.shelllang.core import Entity

        world, _ = sugar_setup
        time = TimeManager()

        player = EntityProxy(Entity(1))

        # Checkpoint before change
        time.checkpoint()
        player.health.current = 50

        # Undo
        time.undo()
        assert player.health.current == 100

        # Redo
        time.redo()
        assert player.health.current == 50

    def test_marks_list(self, sugar_setup):
        """Test marks() returns list of mark names."""
        from foundation.shelllang.sugar import TimeManager

        time = TimeManager()
        time.mark("alpha")
        time.mark("beta")
        time.mark("gamma")

        marks = time.marks()
        assert "alpha" in marks
        assert "beta" in marks
        assert "gamma" in marks


# =============================================================================
# AI INTERFACE TESTS
# =============================================================================


class TestAIInterfaceValidate:
    """Tests for AIInterface.validate()."""

    def test_validate_missing_op(self, ai_interface):
        """Test validation catches missing op."""
        result = ai_interface.validate({})
        assert result["valid"] is False
        assert "op" in result["error"]

    def test_validate_unknown_op(self, ai_interface):
        """Test validation catches unknown op."""
        result = ai_interface.validate({"op": "explode"})
        assert result["valid"] is False
        assert "Unknown" in result["error"]

    def test_validate_set_missing_fields(self, ai_interface):
        """Test validation catches missing set fields."""
        result = ai_interface.validate({"op": "set"})
        assert result["valid"] is False
        assert "requires" in result["error"]

    def test_validate_set_unknown_component(self, ai_interface):
        """Test validation catches unknown component."""
        result = ai_interface.validate({
            "op": "set",
            "entity": 1,
            "component": "Mana",  # Does not exist
            "field": "current",
            "value": 50,
        })
        assert result["valid"] is False
        assert "Unknown component" in result["error"]

    def test_validate_set_nonexistent_entity(self, ai_interface):
        """Test validation catches nonexistent entity."""
        result = ai_interface.validate({
            "op": "set",
            "entity": 999,  # Does not exist
            "component": "Health",
            "field": "current",
            "value": 50,
        })
        assert result["valid"] is False
        assert "does not exist" in result["error"]

    def test_validate_valid_command(self, ai_interface):
        """Test validation passes for valid command."""
        result = ai_interface.validate({
            "op": "set",
            "entity": 1,
            "component": "Health",
            "field": "current",
            "value": 50,
        })
        assert result["valid"] is True


class TestAIInterfaceExecute:
    """Tests for AIInterface.execute()."""

    def test_execute_query(self, ai_interface):
        """Test executing query command."""
        result = ai_interface.execute({
            "op": "query",
            "components": ["Enemy"],
        })

        assert "entities" in result
        assert result["count"] == 3

    def test_execute_query_with_where(self, ai_interface):
        """Test query with where clause."""
        result = ai_interface.execute({
            "op": "query",
            "components": ["Enemy"],
            "where": {
                "Enemy.level": {">=": 3}
            }
        })

        assert result["count"] == 2  # Orc and Dragon

    def test_execute_set(self, ai_interface, populated_world):
        """Test executing set command."""
        from foundation.shelllang.core import Entity

        result = ai_interface.execute({
            "op": "set",
            "entity": 1,
            "component": "Health",
            "field": "current",
            "value": 50,
        })

        assert result["old"] == 100
        assert result["new"] == 50

        # Verify change persisted
        health = populated_world.get(Entity(1), Health)
        assert health.current == 50

    def test_execute_spawn(self, ai_interface, populated_world):
        """Test executing spawn command."""
        initial_count = populated_world.entity_count()

        result = ai_interface.execute({
            "op": "spawn",
            "component": "Enemy",
            "fields": {"name": "Slime", "level": 1},
        })

        assert "entity" in result
        assert populated_world.entity_count() == initial_count + 1

    def test_execute_destroy(self, ai_interface, populated_world):
        """Test executing destroy command."""
        initial_count = populated_world.entity_count()

        result = ai_interface.execute({
            "op": "destroy",
            "entity": 2,  # First enemy
        })

        assert "destroyed" in result
        assert populated_world.entity_count() == initial_count - 1

    def test_execute_inspect(self, ai_interface):
        """Test executing inspect command."""
        result = ai_interface.execute({
            "op": "inspect",
            "entity": 1,
        })

        assert result["entity"] == 1
        assert "components" in result
        assert "Player" in result["components"]
        assert "Health" in result["components"]
        assert "Position" in result["components"]

    def test_execute_schema(self, ai_interface):
        """Test executing schema command."""
        result = ai_interface.execute({
            "op": "schema",
            "type": "Health",
        })

        assert result["name"] == "Health"
        assert "fields" in result
        assert "current" in result["fields"]
        assert "maximum" in result["fields"]

    def test_execute_list_types(self, ai_interface):
        """Test executing list_types command."""
        result = ai_interface.execute({
            "op": "list_types",
        })

        assert "types" in result
        assert "Health" in result["types"]
        assert "Enemy" in result["types"]

    def test_execute_count(self, ai_interface):
        """Test executing count command."""
        result = ai_interface.execute({
            "op": "count",
            "components": ["Enemy"],
        })

        assert result["count"] == 3

    def test_execute_snap(self, ai_interface):
        """Test executing snap command."""
        result = ai_interface.execute({
            "op": "snap",
            "name": "test_snapshot",
        })

        assert "snapshot" in result
        assert result["snapshot"]["name"] == "test_snapshot"
        assert result["snapshot"]["entity_count"] == 4


class TestAIInterfaceDryRun:
    """Tests for AIInterface.dry_run()."""

    def test_dry_run_set(self, ai_interface, populated_world):
        """Test dry_run shows what would change."""
        from foundation.shelllang.core import Entity

        result = ai_interface.dry_run({
            "op": "set",
            "entity": 1,
            "component": "Health",
            "field": "current",
            "value": 50,
        })

        assert "would_change" in result
        assert result["would_change"]["from"] == 100
        assert result["would_change"]["to"] == 50

        # Verify no actual change
        health = populated_world.get(Entity(1), Health)
        assert health.current == 100

    def test_dry_run_spawn(self, ai_interface):
        """Test dry_run shows what would be created."""
        result = ai_interface.dry_run({
            "op": "spawn",
            "component": "Enemy",
            "fields": {"name": "Test", "level": 1},
        })

        assert "would_create" in result
        assert result["would_create"]["component"] == "Enemy"

    def test_dry_run_destroy(self, ai_interface):
        """Test dry_run shows what would be destroyed."""
        result = ai_interface.dry_run({
            "op": "destroy",
            "entity": 1,
        })

        assert "would_destroy" in result
        assert result["would_destroy"]["id"] == 1

    def test_dry_run_invalid_command(self, ai_interface):
        """Test dry_run returns validation error."""
        result = ai_interface.dry_run({
            "op": "set",
            # Missing required fields
        })

        assert result["valid"] is False


# =============================================================================
# REPL TESTS
# =============================================================================


class TestFeedback:
    """Tests for Feedback class."""

    def test_feedback_enabled(self, capsys):
        """Test feedback when enabled."""
        from foundation.shelllang.repl import Feedback

        fb = Feedback(enabled=True)
        fb("test message")

        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_feedback_disabled(self, capsys):
        """Test feedback when disabled."""
        from foundation.shelllang.repl import Feedback

        fb = Feedback(enabled=False)
        fb("test message")

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_feedback_toggle(self, capsys):
        """Test enabling/disabling feedback."""
        from foundation.shelllang.repl import Feedback

        fb = Feedback(enabled=True)
        fb("message 1")

        fb.disable()
        fb("message 2")

        fb.enable()
        fb("message 3")

        captured = capsys.readouterr()
        assert "message 1" in captured.out
        assert "message 2" not in captured.out
        assert "message 3" in captured.out

    def test_feedback_history(self):
        """Test feedback history tracking."""
        from foundation.shelllang.repl import Feedback

        fb = Feedback(enabled=True)
        fb.set_callback(lambda msg: None)  # Suppress output

        fb("one")
        fb("two")
        fb("three")

        history = fb.history(2)
        assert len(history) == 2
        assert history[0] == "two"
        assert history[1] == "three"

    def test_feedback_custom_callback(self):
        """Test custom callback."""
        from foundation.shelllang.repl import Feedback

        messages = []
        fb = Feedback(enabled=True)
        fb.set_callback(lambda msg: messages.append(msg))

        fb("hello")
        fb("world")

        assert messages == ["hello", "world"]


class TestShell:
    """Tests for Shell class."""

    def test_shell_execute_expression(self, populated_world, registry):
        """Test executing expressions."""
        from foundation.shelllang.repl import Shell, Feedback

        fb = Feedback(enabled=False)
        shell = Shell(populated_world, registry, fb)

        result = shell.execute("1 + 1")
        assert result == 2

    def test_shell_execute_statement(self, populated_world, registry):
        """Test executing statements."""
        from foundation.shelllang.repl import Shell, Feedback

        fb = Feedback(enabled=False)
        shell = Shell(populated_world, registry, fb)

        result = shell.execute("x = 42")
        assert result is None
        assert shell.namespace["x"] == 42

    def test_shell_has_world_functions(self, populated_world, registry):
        """Test shell namespace includes world functions."""
        from foundation.shelllang.repl import Shell, Feedback

        fb = Feedback(enabled=False)
        shell = Shell(populated_world, registry, fb)

        ns = shell.namespace
        assert "create" in ns
        assert "destroy" in ns
        assert "query" in ns
        assert "snap" in ns

    def test_shell_has_time_functions(self, populated_world, registry):
        """Test shell namespace includes time functions."""
        from foundation.shelllang.repl import Shell, Feedback

        fb = Feedback(enabled=False)
        shell = Shell(populated_world, registry, fb)

        ns = shell.namespace
        assert "mark" in ns
        assert "rewind" in ns
        assert "undo" in ns
        assert "redo" in ns

    def test_shell_has_component_types(self, populated_world, registry):
        """Test shell namespace includes component types."""
        from foundation.shelllang.repl import Shell, Feedback

        fb = Feedback(enabled=False)
        shell = Shell(populated_world, registry, fb)

        ns = shell.namespace
        assert "Health" in ns
        assert "Enemy" in ns
        assert "Position" in ns

    def test_shell_has_ai_interface(self, populated_world, registry):
        """Test shell namespace includes AI interface."""
        from foundation.shelllang.repl import Shell, Feedback

        fb = Feedback(enabled=False)
        shell = Shell(populated_world, registry, fb)

        assert "ai" in shell.namespace

    def test_shell_exit_commands(self, populated_world, registry):
        """Test exit commands."""
        from foundation.shelllang.repl import Shell, Feedback

        fb = Feedback(enabled=False)
        shell = Shell(populated_world, registry, fb)

        result = shell.execute("quit")
        assert "Goodbye" in result

    def test_shell_help_command(self, populated_world, registry):
        """Test help command."""
        from foundation.shelllang.repl import Shell, Feedback

        fb = Feedback(enabled=False)
        shell = Shell(populated_world, registry, fb)

        result = shell.execute("help")
        assert "Entity Operations" in result
        assert "Queries" in result
        assert "Time Travel" in result


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestShellLangIntegration:
    """Integration tests combining multiple modules."""

    def test_full_workflow(self, world, registry):
        """Test complete workflow: create, query, mutate, snapshot."""
        from foundation.shelllang.repl import Shell, Feedback
        from foundation.shelllang.ai import AIInterface

        fb = Feedback(enabled=False)
        shell = Shell(world, registry, fb)
        ai = AIInterface(world, registry)

        # Create entities via shell
        shell.execute("e1 = create()")
        shell.execute("e2 = create()")

        # Attach components
        shell.execute("world.attach(e1, Health(100, 100))")
        shell.execute("world.attach(e1, Enemy('Goblin', 1))")
        shell.execute("world.attach(e2, Health(50, 50))")
        shell.execute("world.attach(e2, Enemy('Orc', 3))")

        # Query via AI
        result = ai.execute({
            "op": "query",
            "components": ["Enemy"],
        })
        assert result["count"] == 2

        # Mutate via AI
        ai.execute({
            "op": "set",
            "entity": shell.namespace["e1"].id,
            "component": "Health",
            "field": "current",
            "value": 1,
        })

        # Verify via shell
        result = shell.execute("world.get(e1, Health).current")
        assert result == 1

    def test_ai_human_interop(self, populated_world, registry):
        """Test AI and human interfaces work together."""
        from foundation.shelllang.sugar import EntityProxy, TypeQuery, set_world, set_registry, set_echo
        from foundation.shelllang.ai import AIInterface
        from foundation.shelllang.core import Entity

        # Set up sugar
        set_world(populated_world)
        set_registry(registry)
        set_echo(lambda msg: None)

        # Human: query via sugar
        enemies = TypeQuery(Enemy)
        assert enemies.count() == 3

        # AI: mutate via interface
        ai = AIInterface(populated_world, registry)
        ai.execute({
            "op": "set",
            "entity": 2,
            "component": "Enemy",
            "field": "level",
            "value": 99,
        })

        # Human: verify via sugar
        proxy = EntityProxy(Entity(2))
        assert proxy.enemy.level == 99


# =============================================================================
# MODULE EXPORTS TEST
# =============================================================================


class TestModuleExports:
    """Test that all expected symbols are exported."""

    def test_core_exports(self):
        """Test core.py exports."""
        from foundation.shelllang.core import (
            Entity,
            Component,
            Change,
            Snapshot,
            World,
            ENTITY_ID_START,
            DEFAULT_HISTORY_COUNT,
        )

    def test_sugar_exports(self):
        """Test sugar.py exports."""
        from foundation.shelllang.sugar import (
            EntityProxy,
            ComponentProxy,
            QueryResult,
            TypeQuery,
            TimeManager,
            set_world,
            set_echo,
            set_registry,
            MAX_DISPLAY_ENTITIES,
            MAX_UNDO_STACK,
            MAX_REDO_STACK,
            DEFAULT_HISTORY_COUNT,
        )

    def test_ai_exports(self):
        """Test ai.py exports."""
        from foundation.shelllang.ai import (
            AIInterface,
            VALID_OPERATIONS,
            DEFAULT_QUERY_LIMIT,
        )

    def test_repl_exports(self):
        """Test repl.py exports."""
        from foundation.shelllang.repl import (
            Feedback,
            Shell,
            echo,
            DEFAULT_PROMPT,
            DEFAULT_HISTORY_COUNT,
            EXIT_COMMANDS,
            HELP_COMMANDS,
        )

    def test_package_exports(self):
        """Test package-level exports."""
        from foundation.shelllang import (
            # Core
            World,
            Entity,
            Component,
            Snapshot,
            Change,
            # Sugar
            EntityProxy,
            ComponentProxy,
            QueryResult,
            TypeQuery,
            TimeManager,
            # AI
            AIInterface,
            # REPL
            Shell,
            Feedback,
            echo,
        )
