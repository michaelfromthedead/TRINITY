"""Tests for watch variables - watch, breakpoints, updates."""

import pytest
import time
from engine.tooling.debug.watch_variables import (
    WatchWindow,
    WatchVariable,
    Breakpoint,
    ConditionalWatch,
    VariableTracker,
    WatchType,
    BreakpointType,
    ValueChangeType,
    WatchHistoryEntry,
)


class TestWatchVariable:
    """Tests for WatchVariable class."""

    def test_watch_creation(self):
        watch = WatchVariable(
            name="counter",
            getter=lambda: 42,
        )
        assert watch.name == "counter"
        assert watch.watch_type == WatchType.VALUE

    def test_watch_get_value(self):
        counter = [0]
        watch = WatchVariable(
            name="counter",
            getter=lambda: counter[0],
        )
        assert watch.get_value() == 0
        counter[0] = 10
        assert watch.get_value() == 10

    def test_watch_get_value_error(self):
        def failing_getter():
            raise RuntimeError("Failed")

        watch = WatchVariable(name="failing", getter=failing_getter)
        value = watch.get_value()
        assert "error" in str(value).lower()

    def test_watch_format_value(self):
        watch = WatchVariable(
            name="test",
            getter=lambda: 42.5,
            format_string="{value:.1f}",
        )
        assert watch.format_value(42.5) == "42.5"

    def test_watch_update(self):
        counter = [0]
        watch = WatchVariable(
            name="counter",
            getter=lambda: counter[0],
        )
        watch.update(frame=1)
        assert watch.last_value == 0
        assert watch.update_count == 1

    def test_watch_update_detects_change(self):
        counter = [0]
        watch = WatchVariable(
            name="counter",
            getter=lambda: counter[0],
        )
        watch.update()
        counter[0] = 10
        changed = watch.update()
        assert changed is True
        assert watch.last_value == 10

    def test_watch_update_disabled(self):
        watch = WatchVariable(name="test", getter=lambda: 42)
        watch.enabled = False
        changed = watch.update()
        assert changed is False

    def test_watch_history(self):
        counter = [0]
        watch = WatchVariable(
            name="counter",
            getter=lambda: counter[0],
            max_history=5,
        )
        for i in range(10):
            counter[0] = i
            watch.update()

        assert len(watch.history) == 5  # Max history
        assert watch.history[-1].value == 9

    def test_watch_clear_history(self):
        counter = [0]
        watch = WatchVariable(name="counter", getter=lambda: counter[0])
        watch.update()
        counter[0] = 1
        watch.update()
        watch.clear_history()
        assert len(watch.history) == 0


class TestBreakpoint:
    """Tests for Breakpoint class."""

    def test_breakpoint_creation(self):
        bp = Breakpoint(
            breakpoint_id="bp1",
            name="Test Breakpoint",
            condition=lambda: True,
        )
        assert bp.breakpoint_id == "bp1"
        assert bp.breakpoint_type == BreakpointType.ALWAYS

    def test_breakpoint_check_always(self):
        bp = Breakpoint(
            breakpoint_id="bp",
            name="Test",
            condition=lambda: True,
            breakpoint_type=BreakpointType.ALWAYS,
        )
        result = bp.check()
        assert result is True
        assert bp.triggered is True
        assert bp.hit_count == 1

    def test_breakpoint_check_condition_false(self):
        bp = Breakpoint(
            breakpoint_id="bp",
            name="Test",
            condition=lambda: False,
        )
        result = bp.check()
        assert result is False
        assert bp.triggered is False

    def test_breakpoint_disabled(self):
        bp = Breakpoint(
            breakpoint_id="bp",
            name="Test",
            condition=lambda: True,
        )
        bp.enabled = False
        result = bp.check()
        assert result is False

    def test_breakpoint_hit_count(self):
        bp = Breakpoint(
            breakpoint_id="bp",
            name="Test",
            condition=lambda: True,
            breakpoint_type=BreakpointType.HIT_COUNT,
            target_hit_count=3,
        )
        assert bp.check() is False  # Hit 1
        assert bp.check() is False  # Hit 2
        assert bp.check() is True   # Hit 3

    def test_breakpoint_log_only(self):
        bp = Breakpoint(
            breakpoint_id="bp",
            name="Test",
            condition=lambda: True,
            breakpoint_type=BreakpointType.LOG_ONLY,
        )
        # Should return False even when triggered (no break)
        result = bp.check()
        assert result is False
        assert bp.triggered is True

    def test_breakpoint_action(self):
        action_called = [False]

        def action():
            action_called[0] = True

        bp = Breakpoint(
            breakpoint_id="bp",
            name="Test",
            condition=lambda: True,
            action=action,
        )
        bp.check()
        assert action_called[0] is True

    def test_breakpoint_reset(self):
        bp = Breakpoint(
            breakpoint_id="bp",
            name="Test",
            condition=lambda: True,
        )
        bp.check()
        bp.check()
        assert bp.hit_count == 2
        bp.reset()
        assert bp.hit_count == 0
        assert bp.triggered is False


class TestConditionalWatch:
    """Tests for ConditionalWatch class."""

    def test_conditional_watch_creation(self):
        watch = ConditionalWatch(
            name="test",
            getter=lambda: 42,
            change_type=ValueChangeType.ANY,
        )
        assert watch.watch_type == WatchType.CONDITIONAL

    def test_conditional_watch_any_change(self):
        counter = [0]
        triggered = [False]

        watch = ConditionalWatch(
            name="counter",
            getter=lambda: counter[0],
            change_type=ValueChangeType.ANY,
            on_trigger=lambda old, new: triggered.__setitem__(0, True),
        )

        watch.update()
        counter[0] = 1
        watch.update()
        assert triggered[0] is True
        assert watch.triggered is True

    def test_conditional_watch_increase(self):
        counter = [5]
        triggered_count = [0]

        watch = ConditionalWatch(
            name="counter",
            getter=lambda: counter[0],
            change_type=ValueChangeType.INCREASE,
            on_trigger=lambda old, new: triggered_count.__setitem__(0, triggered_count[0] + 1),
        )

        watch.update()  # Initial
        counter[0] = 3  # Decrease
        watch.update()
        assert triggered_count[0] == 0

        counter[0] = 10  # Increase
        watch.update()
        assert triggered_count[0] == 1

    def test_conditional_watch_decrease(self):
        counter = [5]
        triggered = [False]

        watch = ConditionalWatch(
            name="counter",
            getter=lambda: counter[0],
            change_type=ValueChangeType.DECREASE,
            on_trigger=lambda old, new: triggered.__setitem__(0, True),
        )

        watch.update()
        counter[0] = 3
        watch.update()
        assert triggered[0] is True

    def test_conditional_watch_equals(self):
        counter = [0]
        triggered = [False]

        watch = ConditionalWatch(
            name="counter",
            getter=lambda: counter[0],
            change_type=ValueChangeType.EQUALS,
            target_value=10,
            on_trigger=lambda old, new: triggered.__setitem__(0, True),
        )

        watch.update()
        counter[0] = 5
        watch.update()
        assert triggered[0] is False

        counter[0] = 10
        watch.update()
        assert triggered[0] is True

    def test_conditional_watch_greater(self):
        counter = [0]
        triggered = [False]

        watch = ConditionalWatch(
            name="counter",
            getter=lambda: counter[0],
            change_type=ValueChangeType.GREATER,
            target_value=5,
            on_trigger=lambda old, new: triggered.__setitem__(0, True),
        )

        watch.update()
        counter[0] = 3
        watch.update()
        assert triggered[0] is False

        counter[0] = 10
        watch.update()
        assert triggered[0] is True

    def test_conditional_watch_less(self):
        counter = [10]
        triggered = [False]

        watch = ConditionalWatch(
            name="counter",
            getter=lambda: counter[0],
            change_type=ValueChangeType.LESS,
            target_value=5,
            on_trigger=lambda old, new: triggered.__setitem__(0, True),
        )

        watch.update()
        counter[0] = 8
        watch.update()
        assert triggered[0] is False

        counter[0] = 2
        watch.update()
        assert triggered[0] is True

    def test_conditional_watch_trigger_count(self):
        counter = [0]
        watch = ConditionalWatch(
            name="counter",
            getter=lambda: counter[0],
            change_type=ValueChangeType.ANY,
        )

        for i in range(5):
            counter[0] = i
            watch.update()

        # 5 changes: None->0, 0->1, 1->2, 2->3, 3->4
        # (initial last_value is None, so first update counts as a change)
        assert watch.trigger_count == 5

    def test_conditional_watch_reset_trigger_count(self):
        counter = [0]
        watch = ConditionalWatch(
            name="counter",
            getter=lambda: counter[0],
            change_type=ValueChangeType.ANY,
        )

        watch.update()
        counter[0] = 1
        watch.update()
        watch.reset_trigger_count()
        assert watch.trigger_count == 0


class TestVariableTracker:
    """Tests for VariableTracker class."""

    def test_tracker_object_attribute(self):
        class TestObj:
            def __init__(self):
                self.value = 42

        obj = TestObj()
        tracker = VariableTracker(obj, "value")
        assert tracker.get_value() == 42

        obj.value = 100
        assert tracker.get_value() == 100

    def test_tracker_dict(self):
        data = {"key": "value"}
        tracker = VariableTracker(data, "key")
        assert tracker.get_value() == "value"

    def test_tracker_missing_property(self):
        class TestObj:
            pass

        obj = TestObj()
        tracker = VariableTracker(obj, "nonexistent")
        value = tracker.get_value()
        assert "no property" in value.lower()

    def test_tracker_is_valid(self):
        class TestObj:
            value = 42

        obj = TestObj()
        tracker = VariableTracker(obj, "value")
        assert tracker.is_valid is True


class TestWatchWindow:
    """Tests for WatchWindow singleton."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        WatchWindow.reset_instance()
        yield
        WatchWindow.reset_instance()

    def test_singleton(self):
        w1 = WatchWindow.get_instance()
        w2 = WatchWindow.get_instance()
        assert w1 is w2

    def test_enable_disable(self):
        window = WatchWindow.get_instance()
        window.enable()
        assert window.is_enabled
        window.disable()
        assert not window.is_enabled

    def test_show_hide(self):
        window = WatchWindow.get_instance()
        window.show()
        assert window.is_visible
        window.hide()
        assert not window.is_visible

    def test_toggle(self):
        window = WatchWindow.get_instance()
        window.hide()
        result = window.toggle()
        assert result is True
        result = window.toggle()
        assert result is False

    def test_pause_resume(self):
        window = WatchWindow.get_instance()
        window.pause()
        assert window.is_paused
        window.resume()
        assert not window.is_paused

    def test_add_watch(self):
        window = WatchWindow.get_instance()
        watch = window.add_watch("test", lambda: 42)
        assert window.watch_count == 1
        assert window.get_watch("test") is watch

    def test_add_conditional_watch(self):
        window = WatchWindow.get_instance()
        watch = window.add_conditional_watch(
            "test",
            lambda: 42,
            change_type=ValueChangeType.EQUALS,
            target_value=42,
        )
        assert isinstance(watch, ConditionalWatch)

    def test_add_property_watch(self):
        window = WatchWindow.get_instance()

        class TestObj:
            value = 42

        obj = TestObj()
        watch = window.add_property_watch("obj_value", obj, "value")
        assert watch.get_value() == 42

    def test_remove_watch(self):
        window = WatchWindow.get_instance()
        window.add_watch("test", lambda: 42)
        removed = window.remove_watch("test")
        assert removed is not None
        assert window.watch_count == 0

    def test_get_watches_by_category(self):
        window = WatchWindow.get_instance()
        window.add_watch("w1", lambda: 1, category="physics")
        window.add_watch("w2", lambda: 2, category="ai")
        window.add_watch("w3", lambda: 3, category="physics")

        physics = window.get_watches_by_category("physics")
        assert len(physics) == 2

    def test_clear_watches(self):
        window = WatchWindow.get_instance()
        window.add_watch("w1", lambda: 1)
        window.add_watch("w2", lambda: 2)
        window.clear_watches()
        assert window.watch_count == 0

    def test_add_breakpoint(self):
        window = WatchWindow.get_instance()
        bp = window.add_breakpoint(
            "bp1",
            "Test Breakpoint",
            condition=lambda: True,
        )
        assert window.breakpoint_count == 1
        assert window.get_breakpoint("bp1") is bp

    def test_add_value_breakpoint(self):
        window = WatchWindow.get_instance()
        counter = [0]
        bp = window.add_value_breakpoint(
            "bp",
            "Counter equals 10",
            getter=lambda: counter[0],
            target_value=10,
            comparison=ValueChangeType.EQUALS,
        )
        assert bp is not None

    def test_remove_breakpoint(self):
        window = WatchWindow.get_instance()
        window.add_breakpoint("bp1", "Test", condition=lambda: True)
        removed = window.remove_breakpoint("bp1")
        assert removed is not None
        assert window.breakpoint_count == 0

    def test_enable_disable_breakpoint(self):
        window = WatchWindow.get_instance()
        bp = window.add_breakpoint("bp1", "Test", condition=lambda: True)
        window.disable_breakpoint("bp1")
        assert bp.enabled is False
        window.enable_breakpoint("bp1")
        assert bp.enabled is True

    def test_clear_breakpoints(self):
        window = WatchWindow.get_instance()
        window.add_breakpoint("bp1", "Test1", condition=lambda: True)
        window.add_breakpoint("bp2", "Test2", condition=lambda: True)
        window.clear_breakpoints()
        assert window.breakpoint_count == 0

    def test_on_breakpoint_hit(self):
        window = WatchWindow.get_instance()
        hit_breakpoints = []

        def on_hit(bp):
            hit_breakpoints.append(bp)

        window.on_breakpoint_hit(on_hit)
        window.add_breakpoint("bp1", "Test", condition=lambda: True)
        window.update()
        assert len(hit_breakpoints) == 1

    def test_update(self):
        window = WatchWindow.get_instance()
        counter = [0]
        window.add_watch("counter", lambda: counter[0])

        counter[0] = 10
        triggered = window.update()
        # Returns list of triggered breakpoints
        assert isinstance(triggered, list)

    def test_update_paused(self):
        window = WatchWindow.get_instance()
        window.add_breakpoint("bp", "Test", condition=lambda: True)
        window.pause()
        triggered = window.update()
        assert triggered == []

    def test_update_disabled(self):
        window = WatchWindow.get_instance()
        window.add_breakpoint("bp", "Test", condition=lambda: True)
        window.disable()
        triggered = window.update()
        assert triggered == []

    def test_update_interval(self):
        window = WatchWindow.get_instance()
        window.set_update_interval(1.0)  # 1 second

        counter = [0]
        window.add_watch("counter", lambda: counter[0])

        window.update()  # Should update
        window.update()  # Should skip (within interval)

    def test_get_all_values(self):
        window = WatchWindow.get_instance()
        window.add_watch("a", lambda: 1)
        window.add_watch("b", lambda: 2)

        values = window.get_all_values()
        assert values["a"] == 1
        assert values["b"] == 2

    def test_get_triggered_breakpoints(self):
        window = WatchWindow.get_instance()
        window.add_breakpoint("bp1", "Test1", condition=lambda: True)
        window.add_breakpoint("bp2", "Test2", condition=lambda: False)

        window.update()
        triggered = window.get_triggered_breakpoints()
        assert len(triggered) == 1
        assert triggered[0].breakpoint_id == "bp1"

    def test_reset_all_breakpoints(self):
        window = WatchWindow.get_instance()
        bp = window.add_breakpoint("bp", "Test", condition=lambda: True)
        window.update()
        assert bp.triggered is True

        window.reset_all_breakpoints()
        assert bp.triggered is False

    def test_categories(self):
        window = WatchWindow.get_instance()
        window.add_watch("w1", lambda: 1, category="physics")
        window.add_watch("w2", lambda: 2, category="ai")

        categories = window.categories
        assert "physics" in categories
        assert "ai" in categories

    def test_current_frame(self):
        window = WatchWindow.get_instance()
        frame1 = window.current_frame
        window.update()
        frame2 = window.current_frame
        assert frame2 == frame1 + 1

    def test_render(self):
        window = WatchWindow.get_instance()
        window.show()
        window.add_watch("test", lambda: 42)
        window.add_breakpoint("bp", "Test", condition=lambda: False)

        render_data = window.render()
        assert render_data["type"] == "watch_window"
        assert render_data["visible"] is True
        assert len(render_data["watches"]) == 1
        assert len(render_data["breakpoints"]) == 1

    def test_render_hidden(self):
        window = WatchWindow.get_instance()
        window.hide()
        render_data = window.render()
        assert render_data == {}
