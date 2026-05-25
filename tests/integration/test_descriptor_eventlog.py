"""
Integration tests for TrackedDescriptor → EventLog integration.

Verifies that changes made to tracked fields are properly recorded
in the EventLog when inside a @traced context.
"""
import pytest
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from foundation.eventlog import (
    traced,
    get_event_log,
    clear_event_log,
    set_current_tick,
)
from trinity import Component


@pytest.fixture(autouse=True)
def reset_eventlog():
    """Reset event log between tests."""
    clear_event_log()
    set_current_tick(0)
    yield
    clear_event_log()
    set_current_tick(0)


class TestTrackedDescriptorEventLogIntegration:
    """Test that TrackedDescriptor records changes to EventLog."""

    def test_changes_recorded_in_traced_context(self, component_meta):
        """Changes inside @traced are recorded in the event."""

        class Player(Component):
            _track_changes = True
            health: int = 100
            mana: int = 50

            @traced
            def take_damage(self, amount: int) -> None:
                self.health -= amount

        player = Player()
        player.id = 1  # Set entity ID
        player.take_damage(25)

        events = get_event_log().all_events()
        assert len(events) == 1

        event = events[0]
        assert event.entity == 1
        assert len(event.changes) == 1

        change = event.changes[0]
        assert change.field == "health"
        assert change.old_value == 100
        assert change.new_value == 75

    def test_multiple_changes_in_single_operation(self, component_meta):
        """Multiple field changes in one operation are all recorded."""

        class Character(Component):
            _track_changes = True
            health: int = 100
            mana: int = 100
            stamina: int = 100

            @traced
            def rest(self) -> None:
                self.health = 100
                self.mana = 100
                self.stamina = 100

        char = Character(health=50, mana=30, stamina=20)
        char.id = 2
        char.rest()

        events = get_event_log().all_events()
        assert len(events) == 1

        event = events[0]
        assert len(event.changes) == 3

        field_names = {c.field for c in event.changes}
        assert field_names == {"health", "mana", "stamina"}

    def test_unchanged_values_not_recorded(self, component_meta):
        """Setting same value doesn't record a change."""

        class Entity(Component):
            _track_changes = True
            value: int = 0

            @traced
            def no_change(self) -> None:
                self.value = 42  # Same value as already set

        entity = Entity()
        entity.id = 3
        entity.value = 42  # Set value before @traced call
        clear_event_log()  # Clear any events from the initial set

        entity.no_change()

        events = get_event_log().all_events()
        assert len(events) == 1
        assert len(events[0].changes) == 0

    def test_nested_traced_operations(self, component_meta):
        """Changes in nested @traced operations are recorded correctly."""

        class Monster(Component):
            _track_changes = True
            health: int = 100
            rage: int = 0

            @traced
            def take_damage(self, amount: int) -> None:
                self.health -= amount
                self.enrage()

            @traced
            def enrage(self) -> None:
                self.rage += 10

        monster = Monster()
        monster.id = 4
        monster.take_damage(20)

        events = get_event_log().all_events()
        assert len(events) == 2  # take_damage and enrage

        # Check health change is in take_damage event
        take_damage_event = [e for e in events if e.operation.endswith("take_damage")][0]
        health_changes = [c for c in take_damage_event.changes if c.field == "health"]
        assert len(health_changes) == 1
        assert health_changes[0].old_value == 100
        assert health_changes[0].new_value == 80

        # Check rage change is in enrage event
        enrage_event = [e for e in events if e.operation.endswith("enrage")][0]
        rage_changes = [c for c in enrage_event.changes if c.field == "rage"]
        assert len(rage_changes) == 1
        assert rage_changes[0].old_value == 0
        assert rage_changes[0].new_value == 10

    def test_changes_outside_traced_not_in_eventlog(self, component_meta):
        """Changes outside @traced don't create events in EventLog."""

        class Item(Component):
            _track_changes = True
            count: int = 0

        item = Item()
        item.id = 5
        item.count = 10  # Not inside @traced

        events = get_event_log().all_events()
        # Should have no events (change is only tracked via Tracker)
        assert len(events) == 0

    def test_entity_id_from_id_attribute(self, component_meta):
        """Entity ID comes from .id attribute if present."""

        class Entity(Component):
            _track_changes = True
            value: int = 0

            @traced
            def modify(self) -> None:
                self.value = 999

        entity = Entity()
        entity.id = 42
        entity.modify()

        event = get_event_log().all_events()[0]
        assert event.changes[0].entity == 42

    def test_entity_id_fallback_to_object_id(self, component_meta):
        """Entity ID falls back to id(obj) if no .id attribute."""

        class NoIdEntity(Component):
            _track_changes = True
            value: int = 0

            @traced
            def modify(self) -> None:
                self.value = 123

        entity = NoIdEntity()
        # Don't set entity.id
        entity.modify()

        event = get_event_log().all_events()[0]
        # Change entity should be id(entity)
        assert event.changes[0].entity == id(entity)

    def test_causal_chain_with_changes(self, component_meta):
        """Changes include proper entity context in causal chain."""

        class Attacker(Component):
            _track_changes = True
            damage: int = 10

            @traced
            def attack(self, target) -> None:
                target.receive_damage(self.damage)

        class Target(Component):
            _track_changes = True
            health: int = 100

            @traced
            def receive_damage(self, amount: int) -> None:
                self.health -= amount

        attacker = Attacker()
        attacker.id = 1
        target = Target()
        target.id = 2

        attacker.attack(target)

        events = get_event_log().all_events()
        assert len(events) == 2

        # Both events should trace back to attacker
        for event in events:
            assert event.root_cause_entity == 1


class TestEventLogQueryingWithChanges:
    """Test querying events with changes."""

    def test_query_changes_by_field(self, component_meta):
        """Can query changes by field name."""

        class Entity(Component):
            _track_changes = True
            x: int = 0
            y: int = 0

            @traced
            def move(self, dx: int, dy: int) -> None:
                self.x += dx
                self.y += dy

        entity = Entity()
        entity.id = 1
        entity.move(10, 20)

        log = get_event_log()
        x_changes = log.changes_where(field="x")
        y_changes = log.changes_where(field="y")

        assert len(x_changes) == 1
        assert x_changes[0].new_value == 10

        assert len(y_changes) == 1
        assert y_changes[0].new_value == 20
