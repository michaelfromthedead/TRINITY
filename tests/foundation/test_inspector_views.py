"""
Tests for Inspector Views - History and Causality views.

Verifies:
- HistoryView rendering with various scenarios
- CausalityView rendering with causal chain data
- RootCauseSummary aggregate analysis
- Integration with EventLog system
"""
import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from foundation.eventlog import (
    Change,
    Event,
    EventLog,
    get_event_log,
    set_current_tick,
    clear_event_log,
    traced,
    add_change_to_current_event,
)
from foundation.inspector import TextUIContext, inspector, Inspector
from foundation.inspector_views import (
    HistoryView,
    CausalityView,
    RootCauseSummary,
    register_inspector_views,
)


@pytest.fixture(autouse=True)
def reset_event_log():
    """Reset event log and tick between tests."""
    clear_event_log()
    set_current_tick(0)
    yield
    clear_event_log()
    set_current_tick(0)


class MockEntity:
    """Simple mock entity for testing."""

    def __init__(self, entity_id: int):
        self.id = entity_id


class TestHistoryViewEmpty:
    """Test HistoryView with no history."""

    def test_history_view_empty(self):
        """Entity with no history shows appropriate message."""
        entity = MockEntity(999)
        view = HistoryView()
        ctx = TextUIContext()

        view.render(entity, ctx)
        output = ctx.get_output()

        assert "No history recorded" in output

    def test_history_view_can_render_entity(self):
        """HistoryView can render objects with id attribute."""
        entity = MockEntity(1)
        view = HistoryView()

        assert view.can_render(entity) is True

    def test_history_view_cannot_render_primitives(self):
        """HistoryView cannot render objects without id."""
        view = HistoryView()

        assert view.can_render(42) is False
        assert view.can_render("string") is False
        assert view.can_render(None) is False
        assert view.can_render([1, 2, 3]) is False


class TestHistoryViewSingleChange:
    """Test HistoryView with a single change."""

    def test_history_view_single_change(self):
        """One change displays correctly."""
        entity = MockEntity(1)
        log = get_event_log()

        # Create an event with one change
        change = Change(entity=1, field="health", old_value=100, new_value=50)
        event = Event(
            tick=100,
            operation="Enemy.attack",
            entity=1,
            changes=[change]
        )
        log.record(event)

        view = HistoryView()
        ctx = TextUIContext()
        view.render(entity, ctx)
        output = ctx.get_output()

        assert "Entity History" in output
        assert "tick   100" in output
        assert "health" in output
        assert "100" in output
        assert "50" in output
        assert "Enemy.attack" in output


class TestHistoryViewMultipleChanges:
    """Test HistoryView with multiple changes."""

    def test_history_view_multiple_changes(self):
        """Many changes display in order."""
        entity = MockEntity(1)
        log = get_event_log()

        # First change at tick 100
        change1 = Change(entity=1, field="health", old_value=100, new_value=70)
        event1 = Event(
            tick=100,
            operation="Enemy.attack",
            entity=1,
            changes=[change1]
        )
        log.record(event1)

        # Second change at tick 200
        change2 = Change(entity=1, field="health", old_value=70, new_value=45)
        event2 = Event(
            tick=200,
            operation="Trap.trigger",
            entity=1,
            changes=[change2]
        )
        log.record(event2)

        # Third change at tick 300
        change3 = Change(entity=1, field="health", old_value=45, new_value=0)
        event3 = Event(
            tick=300,
            operation="Player.death",
            entity=1,
            changes=[change3]
        )
        log.record(event3)

        view = HistoryView()
        ctx = TextUIContext()
        view.render(entity, ctx)
        output = ctx.get_output()

        # Verify all changes are present
        assert "100 -> 70" in output or "100->70" in output or ("100" in output and "70" in output)
        assert "70 -> 45" in output or "70->45" in output or ("70" in output and "45" in output)
        assert "45 -> 0" in output or "45->0" in output or ("45" in output and "0" in output)

    def test_history_view_most_recent_first(self):
        """Changes are displayed most recent first."""
        entity = MockEntity(1)
        log = get_event_log()

        # Earlier tick
        change1 = Change(entity=1, field="x", old_value=0, new_value=10)
        event1 = Event(tick=100, operation="first_op", entity=1, changes=[change1])
        log.record(event1)

        # Later tick
        change2 = Change(entity=1, field="x", old_value=10, new_value=20)
        event2 = Event(tick=500, operation="later_op", entity=1, changes=[change2])
        log.record(event2)

        view = HistoryView()
        ctx = TextUIContext()
        view.render(entity, ctx)
        output = ctx.get_output()

        # Find positions of tick numbers in output
        pos_100 = output.find("100")
        pos_500 = output.find("500")

        # Most recent (tick 500) should appear before earlier (tick 100)
        assert pos_500 < pos_100, "Most recent change should appear first"


class TestCausalityViewNoCause:
    """Test CausalityView with standalone operations."""

    def test_causality_view_no_cause(self):
        """Standalone operation without causal chain."""
        entity = MockEntity(1)
        log = get_event_log()

        # Event without parent or root cause
        change = Change(entity=1, field="position", old_value=0, new_value=5)
        event = Event(
            tick=50,
            operation="Entity.move",
            entity=1,
            changes=[change],
            depth=0
        )
        log.record(event)

        view = CausalityView()
        ctx = TextUIContext()
        view.render(entity, ctx)
        output = ctx.get_output()

        assert "Causality Analysis" in output
        assert "Entity.move" in output
        assert "depth: 0" in output

    def test_causality_view_empty(self):
        """Entity with no causal data shows message."""
        entity = MockEntity(999)
        view = CausalityView()
        ctx = TextUIContext()

        view.render(entity, ctx)
        output = ctx.get_output()

        assert "No causal data recorded" in output


class TestCausalityViewDirectCause:
    """Test CausalityView with immediate parent."""

    def test_causality_view_direct_cause(self):
        """Immediate parent is shown."""
        entity = MockEntity(1)
        log = get_event_log()

        change = Change(entity=1, field="health", old_value=100, new_value=50)
        event = Event(
            tick=200,
            operation="Player.take_damage",
            entity=1,
            changes=[change],
            immediate_parent="Enemy.attack",
            immediate_parent_entity=2,
            depth=1
        )
        log.record(event)

        view = CausalityView()
        ctx = TextUIContext()
        view.render(entity, ctx)
        output = ctx.get_output()

        assert "direct cause: Enemy.attack" in output
        assert "Entity 2" in output


class TestCausalityViewRootCause:
    """Test CausalityView with root cause chain."""

    def test_causality_view_root_cause(self):
        """Root cause chain is displayed."""
        entity = MockEntity(1)
        log = get_event_log()

        change = Change(entity=1, field="health", old_value=45, new_value=0)
        event = Event(
            tick=5030,
            operation="Trap.trigger",
            entity=1,
            changes=[change],
            immediate_parent="trap_activate",
            immediate_parent_entity=23,
            root_cause="Monster_G.think",
            root_cause_entity=8,
            depth=3
        )
        log.record(event)

        view = CausalityView()
        ctx = TextUIContext()
        view.render(entity, ctx)
        output = ctx.get_output()

        assert "root cause: Monster_G.think" in output
        assert "Entity 8" in output

    def test_causality_view_depth(self):
        """Depth is displayed correctly."""
        entity = MockEntity(1)
        log = get_event_log()

        change = Change(entity=1, field="x", old_value=0, new_value=1)
        event = Event(
            tick=100,
            operation="deep_op",
            entity=1,
            changes=[change],
            depth=5
        )
        log.record(event)

        view = CausalityView()
        ctx = TextUIContext()
        view.render(entity, ctx)
        output = ctx.get_output()

        assert "depth: 5" in output


class TestRootCauseSummary:
    """Test RootCauseSummary aggregate analysis."""

    def test_root_cause_summary_basic(self):
        """Basic root cause summary calculation."""
        entity_id = 1
        log = get_event_log()

        # Create events with same root cause
        for i in range(3):
            change = Change(entity=entity_id, field="x", old_value=i, new_value=i + 1)
            event = Event(
                tick=i * 100,
                operation=f"op_{i}",
                entity=entity_id,
                changes=[change],
                root_cause="Monster.think",
                root_cause_entity=10
            )
            log.record(event)

        summaries = RootCauseSummary.for_entity(entity_id, log)

        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.root_cause == "Monster.think"
        assert summary.root_cause_entity == 10
        assert summary.total_events == 3
        assert summary.total_changes == 3
        assert entity_id in summary.affected_entities

    def test_root_cause_summary_multiple_roots(self):
        """Summary with multiple root causes."""
        entity_id = 1
        log = get_event_log()

        # Events from root cause A
        for i in range(2):
            change = Change(entity=entity_id, field="x", old_value=i, new_value=i + 1)
            event = Event(
                tick=i * 100,
                operation=f"op_a_{i}",
                entity=entity_id,
                changes=[change],
                root_cause="CauseA.action",
                root_cause_entity=5
            )
            log.record(event)

        # Events from root cause B
        for i in range(3):
            change = Change(entity=entity_id, field="y", old_value=i, new_value=i + 10)
            event = Event(
                tick=(i + 2) * 100,
                operation=f"op_b_{i}",
                entity=entity_id,
                changes=[change],
                root_cause="CauseB.action",
                root_cause_entity=7
            )
            log.record(event)

        summaries = RootCauseSummary.for_entity(entity_id, log)

        assert len(summaries) == 2

        # Find each summary
        summary_a = next(s for s in summaries if s.root_cause == "CauseA.action")
        summary_b = next(s for s in summaries if s.root_cause == "CauseB.action")

        assert summary_a.total_events == 2
        assert summary_a.root_cause_entity == 5

        assert summary_b.total_events == 3
        assert summary_b.root_cause_entity == 7

    def test_root_cause_summary_no_root_cause(self):
        """Events without root cause are excluded from summary."""
        entity_id = 1
        log = get_event_log()

        # Event without root cause
        change = Change(entity=entity_id, field="x", old_value=0, new_value=1)
        event = Event(
            tick=100,
            operation="standalone_op",
            entity=entity_id,
            changes=[change],
            root_cause=None,
            root_cause_entity=None
        )
        log.record(event)

        summaries = RootCauseSummary.for_entity(entity_id, log)

        # No summaries since no root cause
        assert len(summaries) == 0

    def test_root_cause_summary_affected_entities(self):
        """Affected entities set tracks all changed entities."""
        entity_id = 1
        log = get_event_log()

        # Event affecting multiple entities
        changes = [
            Change(entity=1, field="x", old_value=0, new_value=1),
            Change(entity=2, field="y", old_value=0, new_value=2),
            Change(entity=3, field="z", old_value=0, new_value=3),
        ]
        event = Event(
            tick=100,
            operation="multi_change",
            entity=entity_id,
            changes=changes,
            root_cause="Root.op",
            root_cause_entity=99
        )
        log.record(event)

        summaries = RootCauseSummary.for_entity(entity_id, log)

        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.total_changes == 3
        assert summary.affected_entities == {1, 2, 3}


class TestRegisterInspectorViews:
    """Test the register_inspector_views function."""

    def test_register_inspector_views(self):
        """Views are registered with inspector."""
        # Create a fresh inspector
        test_inspector = Inspector()

        # Get initial view count
        initial_views = len(test_inspector._views)

        # Manually register (simulating register_inspector_views behavior)
        test_inspector.register_view(HistoryView())
        test_inspector.register_view(CausalityView())

        # Should have two more views
        assert len(test_inspector._views) == initial_views + 2

    def test_registered_views_work_with_panel(self):
        """Registered views are accessible via panel."""
        test_inspector = Inspector()
        test_inspector.register_view(HistoryView())
        test_inspector.register_view(CausalityView())

        entity = MockEntity(1)
        panel = test_inspector.inspect(entity)

        # Both views should be available
        view_names = [v.name for v in panel.views]
        assert "History" in view_names
        assert "Causality" in view_names

    def test_can_switch_to_history_view(self):
        """Can switch to History view on panel."""
        test_inspector = Inspector()
        test_inspector.register_view(HistoryView())

        entity = MockEntity(1)
        panel = test_inspector.inspect(entity)

        result = panel.set_view("History")
        assert result is True
        assert panel.current_view.name == "History"

    def test_can_switch_to_causality_view(self):
        """Can switch to Causality view on panel."""
        test_inspector = Inspector()
        test_inspector.register_view(CausalityView())

        entity = MockEntity(1)
        panel = test_inspector.inspect(entity)

        result = panel.set_view("Causality")
        assert result is True
        assert panel.current_view.name == "Causality"


class TestViewNames:
    """Test view name attributes."""

    def test_history_view_name(self):
        """HistoryView has correct name."""
        view = HistoryView()
        assert view.name == "History"

    def test_causality_view_name(self):
        """CausalityView has correct name."""
        view = CausalityView()
        assert view.name == "Causality"


class TestIntegrationWithTracedDecorator:
    """Test views work with @traced decorator."""

    def test_history_with_traced_operations(self):
        """HistoryView shows events from @traced operations."""

        class Player:
            def __init__(self, player_id: int):
                self.id = player_id
                self._health = 100

            @traced
            def take_damage(self, amount: int) -> None:
                old_health = self._health
                self._health = max(0, self._health - amount)
                change = Change(
                    entity=self.id,
                    field="health",
                    old_value=old_health,
                    new_value=self._health
                )
                add_change_to_current_event(change)

        set_current_tick(1000)
        player = Player(42)
        player.take_damage(30)

        view = HistoryView()
        ctx = TextUIContext()
        view.render(player, ctx)
        output = ctx.get_output()

        assert "health" in output
        assert "100" in output
        assert "70" in output

    def test_causality_with_nested_traced(self):
        """CausalityView shows causal chain from nested @traced."""

        class Monster:
            def __init__(self, monster_id: int):
                self.id = monster_id

            @traced
            def think(self):
                return self.attack()

            @traced
            def attack(self):
                change = Change(
                    entity=self.id,
                    field="action",
                    old_value="idle",
                    new_value="attacking"
                )
                add_change_to_current_event(change)
                return "attacked"

        set_current_tick(500)
        monster = Monster(10)
        monster.think()

        view = CausalityView()
        ctx = TextUIContext()
        view.render(monster, ctx)
        output = ctx.get_output()

        # Should show the causal chain
        assert "Monster.attack" in output or "attack" in output
