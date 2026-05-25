"""
Tests for EventLog and @traced decorator.

Verifies:
- Event and Change dataclasses
- EventLog indexing and querying
- @traced decorator behavior
- Causal chain tracking
"""
import pytest
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from foundation.eventlog import (
    Change,
    Event,
    EventLog,
    traced,
    set_current_tick,
    get_current_tick,
    get_event_log,
    get_current_event,
    add_change_to_current_event,
    clear_event_log,
)


@pytest.fixture(autouse=True)
def reset_event_log():
    """Reset event log and tick between tests."""
    clear_event_log()
    set_current_tick(0)
    yield
    clear_event_log()
    set_current_tick(0)


class TestChangeDataclass:
    """Test Change dataclass."""

    def test_change_creation(self):
        """Change should store all fields."""
        change = Change(entity=1, field="health", old_value=100, new_value=50)

        assert change.entity == 1
        assert change.field == "health"
        assert change.old_value == 100
        assert change.new_value == 50

    def test_change_with_none_values(self):
        """Change should handle None values."""
        change = Change(entity=0, field="data", old_value=None, new_value="set")

        assert change.old_value is None
        assert change.new_value == "set"


class TestEventDataclass:
    """Test Event dataclass."""

    def test_event_creation_minimal(self):
        """Event with minimal fields."""
        event = Event(tick=0, operation="test")

        assert event.tick == 0
        assert event.operation == "test"
        assert event.changes == []
        assert event.entity is None
        assert event.result is None
        assert event.error is None
        assert event.depth == 0

    def test_event_creation_full(self):
        """Event with all fields."""
        change = Change(entity=1, field="x", old_value=0, new_value=10)
        event = Event(
            tick=5,
            operation="Player.move",
            operation_args={"dx": 10},
            entity=1,
            changes=[change],
            result=True,
            immediate_parent="World.update",
            immediate_parent_entity=None,
            root_cause="Monster.think",
            root_cause_entity=2,
            depth=3,
        )

        assert event.tick == 5
        assert event.operation == "Player.move"
        assert event.operation_args == {"dx": 10}
        assert event.entity == 1
        assert len(event.changes) == 1
        assert event.result is True
        assert event.immediate_parent == "World.update"
        assert event.root_cause == "Monster.think"
        assert event.root_cause_entity == 2
        assert event.depth == 3


class TestEventLogBasics:
    """Test EventLog basic operations."""

    def test_record_event(self):
        """Recording an event adds it to the log."""
        log = EventLog()
        event = Event(tick=0, operation="test")

        log.record(event)

        assert len(log) == 1

    def test_all_events(self):
        """all_events returns all recorded events."""
        log = EventLog()
        e1 = Event(tick=0, operation="op1")
        e2 = Event(tick=1, operation="op2")

        log.record(e1)
        log.record(e2)

        events = log.all_events()
        assert len(events) == 2
        assert e1 in events
        assert e2 in events

    def test_clear(self):
        """clear removes all events."""
        log = EventLog()
        log.record(Event(tick=0, operation="test"))
        log.record(Event(tick=1, operation="test"))

        log.clear()

        assert len(log) == 0
        assert log.all_events() == []


class TestEventLogIndexes:
    """Test EventLog indexing."""

    def test_index_by_tick(self):
        """events_at returns events for a specific tick."""
        log = EventLog()
        e1 = Event(tick=0, operation="op1")
        e2 = Event(tick=0, operation="op2")
        e3 = Event(tick=1, operation="op3")

        log.record(e1)
        log.record(e2)
        log.record(e3)

        tick0_events = log.events_at(0)
        assert len(tick0_events) == 2
        assert e1 in tick0_events
        assert e2 in tick0_events
        assert e3 not in tick0_events

    def test_index_by_entity(self):
        """events_for_entity returns events for a specific entity."""
        log = EventLog()
        e1 = Event(tick=0, operation="op1", entity=1)
        e2 = Event(tick=0, operation="op2", entity=2)
        e3 = Event(tick=0, operation="op3", entity=1)

        log.record(e1)
        log.record(e2)
        log.record(e3)

        entity1_events = log.events_for_entity(1)
        assert len(entity1_events) == 2
        assert e1 in entity1_events
        assert e3 in entity1_events
        assert e2 not in entity1_events

    def test_index_by_operation(self):
        """events_for_operation returns events for a specific operation."""
        log = EventLog()
        e1 = Event(tick=0, operation="Player.move")
        e2 = Event(tick=0, operation="Player.attack")
        e3 = Event(tick=0, operation="Player.move")

        log.record(e1)
        log.record(e2)
        log.record(e3)

        move_events = log.events_for_operation("Player.move")
        assert len(move_events) == 2

    def test_index_by_root_cause(self):
        """events_caused_by returns events with a specific root cause entity."""
        log = EventLog()
        e1 = Event(tick=0, operation="op1", root_cause_entity=10)
        e2 = Event(tick=0, operation="op2", root_cause_entity=20)
        e3 = Event(tick=0, operation="op3", root_cause_entity=10)

        log.record(e1)
        log.record(e2)
        log.record(e3)

        caused_by_10 = log.events_caused_by(10)
        assert len(caused_by_10) == 2
        assert e1 in caused_by_10
        assert e3 in caused_by_10


class TestEventLogQuerying:
    """Test EventLog query methods."""

    def test_events_where_single_filter(self):
        """events_where with single filter."""
        log = EventLog()
        log.record(Event(tick=0, operation="op1", entity=1))
        log.record(Event(tick=0, operation="op2", entity=2))

        result = log.events_where(entity=1)
        assert len(result) == 1

    def test_events_where_multiple_filters(self):
        """events_where with multiple filters."""
        log = EventLog()
        log.record(Event(tick=0, operation="op", entity=1))
        log.record(Event(tick=1, operation="op", entity=1))
        log.record(Event(tick=0, operation="op", entity=2))

        result = log.events_where(tick=0, entity=1)
        assert len(result) == 1

    def test_events_where_has_error(self):
        """events_where can filter by error presence."""
        log = EventLog()
        log.record(Event(tick=0, operation="success"))
        log.record(Event(tick=0, operation="failure", error=ValueError("test")))

        errors = log.events_where(has_error=True)
        assert len(errors) == 1
        assert errors[0].operation == "failure"

    def test_events_where_depth_range(self):
        """events_where can filter by depth range."""
        log = EventLog()
        log.record(Event(tick=0, operation="d0", depth=0))
        log.record(Event(tick=0, operation="d1", depth=1))
        log.record(Event(tick=0, operation="d2", depth=2))
        log.record(Event(tick=0, operation="d3", depth=3))

        shallow = log.events_where(max_depth=1)
        assert len(shallow) == 2

        deep = log.events_where(min_depth=2)
        assert len(deep) == 2

    def test_changes_where(self):
        """changes_where filters changes."""
        log = EventLog()
        c1 = Change(entity=1, field="x", old_value=0, new_value=1)
        c2 = Change(entity=2, field="x", old_value=0, new_value=2)
        c3 = Change(entity=1, field="y", old_value=0, new_value=3)

        log.record(Event(tick=0, operation="op", changes=[c1, c2, c3]))

        entity1_changes = log.changes_where(entity=1)
        assert len(entity1_changes) == 2

        x_changes = log.changes_where(field="x")
        assert len(x_changes) == 2


class TestTracedDecoratorBasics:
    """Test @traced decorator basic functionality."""

    def test_traced_records_event(self):
        """@traced records an event for the operation."""
        class Entity:
            id = 1

            @traced
            def do_something(self):
                pass

        entity = Entity()
        entity.do_something()

        log = get_event_log()
        assert len(log) == 1
        # __qualname__ includes enclosing scope, so check suffix
        assert log.all_events()[0].operation.endswith("Entity.do_something")

    def test_traced_captures_entity_id(self):
        """@traced captures entity ID from self.id."""
        class Player:
            def __init__(self):
                self.id = 42

            @traced
            def move(self):
                pass

        player = Player()
        player.move()

        event = get_event_log().all_events()[0]
        assert event.entity == 42

    def test_traced_returns_result(self):
        """@traced passes through return values."""
        class Calculator:
            id = 1

            @traced
            def add(self, a, b):
                return a + b

        calc = Calculator()
        result = calc.add(2, 3)

        assert result == 5

    def test_traced_captures_result_in_event(self):
        """@traced stores return value in event."""
        class Service:
            id = 1

            @traced
            def compute(self):
                return 42

        Service().compute()

        event = get_event_log().all_events()[0]
        assert event.result == 42

    def test_traced_captures_exception(self):
        """@traced captures exceptions in event."""
        class Failer:
            id = 1

            @traced
            def fail(self):
                raise ValueError("test error")

        failer = Failer()
        with pytest.raises(ValueError):
            failer.fail()

        event = get_event_log().all_events()[0]
        assert event.error is not None
        assert "test error" in str(event.error)

    def test_traced_preserves_exception(self):
        """@traced re-raises exceptions."""
        class Failer:
            id = 1

            @traced
            def fail(self):
                raise RuntimeError("propagate me")

        with pytest.raises(RuntimeError) as exc_info:
            Failer().fail()

        assert "propagate me" in str(exc_info.value)


class TestTracedCausalChain:
    """Test @traced causal chain tracking."""

    def test_root_cause_is_first_entity_operation(self):
        """Root cause is the first entity-bound operation."""
        class Player:
            def __init__(self, id):
                self.id = id

            @traced
            def take_damage(self, amount):
                pass

        set_current_tick(1)
        player = Player(1)
        player.take_damage(10)

        event = get_event_log().all_events()[0]
        assert event.root_cause.endswith("Player.take_damage")
        assert event.root_cause_entity == 1

    def test_nested_calls_preserve_root_cause(self):
        """Nested calls maintain the original root cause."""
        class Monster:
            def __init__(self, id):
                self.id = id

            @traced
            def think(self):
                return self.attack()

            @traced
            def attack(self):
                return "attacked"

        monster = Monster(10)
        monster.think()

        events = get_event_log().all_events()
        # Both events should have Monster.think as root cause
        for event in events:
            assert event.root_cause.endswith("Monster.think")
            assert event.root_cause_entity == 10

    def test_immediate_parent_tracking(self):
        """Immediate parent is tracked for nested calls."""
        class Entity:
            def __init__(self, id):
                self.id = id

            @traced
            def outer(self):
                return self.inner()

            @traced
            def inner(self):
                return "inner"

        Entity(1).outer()

        events = get_event_log().all_events()
        inner_event = [e for e in events if e.operation.endswith("Entity.inner")][0]
        assert inner_event.immediate_parent.endswith("Entity.outer")
        assert inner_event.immediate_parent_entity == 1

    def test_depth_increases_with_nesting(self):
        """Call depth increases with nesting."""
        class DeepEntity:
            id = 1

            @traced
            def level0(self):
                return self.level1()

            @traced
            def level1(self):
                return self.level2()

            @traced
            def level2(self):
                return "deep"

        DeepEntity().level0()

        events = get_event_log().all_events()
        depths = {e.operation.split('.')[-1]: e.depth for e in events}
        assert depths["level0"] == 0
        assert depths["level1"] == 1
        assert depths["level2"] == 2

    def test_system_is_not_root_cause(self):
        """Operations without entity ID don't become root cause."""
        class System:
            # No id attribute - this is a "pass-through" system

            @traced
            def update(self, player):
                player.tick()

        class Player:
            def __init__(self, id):
                self.id = id

            @traced
            def tick(self):
                pass

        system = System()
        player = Player(42)
        system.update(player)

        events = get_event_log().all_events()
        # Player.tick should be the root cause, not System.update
        for event in events:
            if event.root_cause is not None:
                assert event.root_cause.endswith("Player.tick")
                assert event.root_cause_entity == 42


class TestCurrentTickAndEvent:
    """Test tick and current event management."""

    def test_set_and_get_tick(self):
        """Tick can be set and retrieved."""
        set_current_tick(42)
        assert get_current_tick() == 42

    def test_events_use_current_tick(self):
        """Events use the current tick value."""
        set_current_tick(10)

        class Entity:
            id = 1

            @traced
            def action(self):
                pass

        Entity().action()

        event = get_event_log().all_events()[0]
        assert event.tick == 10

    def test_get_current_event_inside_traced(self):
        """get_current_event returns the event inside @traced."""
        captured_event = None

        class Entity:
            id = 1

            @traced
            def capture(self):
                nonlocal captured_event
                captured_event = get_current_event()

        Entity().capture()

        assert captured_event is not None
        assert captured_event.operation.endswith("Entity.capture")

    def test_get_current_event_outside_traced(self):
        """get_current_event returns None outside @traced."""
        assert get_current_event() is None


class TestAddChangeToCurrentEvent:
    """Test add_change_to_current_event function."""

    def test_add_change_inside_traced(self):
        """Changes can be added to current event inside @traced."""
        class Entity:
            id = 1

            @traced
            def modify(self):
                change = Change(entity=1, field="x", old_value=0, new_value=1)
                add_change_to_current_event(change)

        Entity().modify()

        event = get_event_log().all_events()[0]
        assert len(event.changes) == 1
        assert event.changes[0].field == "x"

    def test_add_change_outside_traced_returns_false(self):
        """add_change_to_current_event returns False outside @traced."""
        change = Change(entity=1, field="x", old_value=0, new_value=1)
        result = add_change_to_current_event(change)
        assert result is False


class TestClearEventLog:
    """Test clear_event_log function."""

    def test_clear_event_log(self):
        """clear_event_log clears the global log."""
        class Entity:
            id = 1

            @traced
            def action(self):
                pass

        Entity().action()
        assert len(get_event_log()) == 1

        clear_event_log()
        assert len(get_event_log()) == 0
