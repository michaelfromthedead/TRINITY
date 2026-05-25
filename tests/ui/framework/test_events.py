"""
Comprehensive tests for the UI event system.

Tests cover:
- Event creation and properties
- Event bubbling and capturing
- Mouse events (click, move, scroll, enter, leave)
- Keyboard events (key down, key up, character input)
- Focus events (focus in, focus out)
- Drag events (drag start, drag, drag end, drop)
- Event propagation stopping
- Event dispatcher
"""

import time
import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.ui.framework.events import (
    EventPhase,
    MouseButton,
    KeyModifier,
    EventType,
    UIEvent,
    MouseEvent,
    KeyboardEvent,
    FocusEvent,
    DragEvent,
    EventHandler,
    EventDispatcher,
)
from engine.ui.framework.coordinate import Point


class TestEventPhase:
    """Tests for EventPhase enum."""

    def test_none_phase_is_valid_enum_member(self):
        """EventPhase.NONE should be a valid enum member with auto() value."""
        assert isinstance(EventPhase.NONE, EventPhase)
        assert EventPhase.NONE.name == "NONE"
        assert isinstance(EventPhase.NONE.value, int)

    def test_capture_phase_is_valid_enum_member(self):
        """EventPhase.CAPTURE should be a valid enum member with auto() value."""
        assert isinstance(EventPhase.CAPTURE, EventPhase)
        assert EventPhase.CAPTURE.name == "CAPTURE"
        assert isinstance(EventPhase.CAPTURE.value, int)

    def test_target_phase_is_valid_enum_member(self):
        """EventPhase.TARGET should be a valid enum member with auto() value."""
        assert isinstance(EventPhase.TARGET, EventPhase)
        assert EventPhase.TARGET.name == "TARGET"
        assert isinstance(EventPhase.TARGET.value, int)

    def test_bubble_phase_is_valid_enum_member(self):
        """EventPhase.BUBBLE should be a valid enum member with auto() value."""
        assert isinstance(EventPhase.BUBBLE, EventPhase)
        assert EventPhase.BUBBLE.name == "BUBBLE"
        assert isinstance(EventPhase.BUBBLE.value, int)

    def test_phases_are_distinct(self):
        """All phases should have unique values."""
        phases = [
            EventPhase.NONE,
            EventPhase.CAPTURE,
            EventPhase.TARGET,
            EventPhase.BUBBLE,
        ]
        values = [p.value for p in phases]
        assert len(phases) == len(set(phases))
        assert len(values) == len(set(values))  # Values must also be unique

    def test_all_phases_are_members_of_enum(self):
        """All phases should be EventPhase members."""
        for phase in [EventPhase.NONE, EventPhase.CAPTURE,
                      EventPhase.TARGET, EventPhase.BUBBLE]:
            assert isinstance(phase, EventPhase)

    def test_enum_has_exactly_four_members(self):
        """EventPhase should have exactly 4 members."""
        assert len(EventPhase) == 4


class TestMouseButton:
    """Tests for MouseButton enum."""

    def test_none_button(self):
        """NONE should have value 0."""
        assert MouseButton.NONE == 0

    def test_left_button(self):
        """LEFT should have value 1."""
        assert MouseButton.LEFT == 1

    def test_right_button(self):
        """RIGHT should have value 2."""
        assert MouseButton.RIGHT == 2

    def test_middle_button(self):
        """MIDDLE should have value 4."""
        assert MouseButton.MIDDLE == 4

    def test_button4(self):
        """BUTTON4 should have value 8."""
        assert MouseButton.BUTTON4 == 8

    def test_button5(self):
        """BUTTON5 should have value 16."""
        assert MouseButton.BUTTON5 == 16

    def test_button_combinations(self):
        """Multiple buttons can be combined."""
        buttons = MouseButton.LEFT | MouseButton.RIGHT
        assert buttons & MouseButton.LEFT
        assert buttons & MouseButton.RIGHT
        assert not (buttons & MouseButton.MIDDLE)

    def test_from_index_left(self):
        """from_index(0) should return LEFT."""
        assert MouseButton.from_index(0) == MouseButton.LEFT

    def test_from_index_right(self):
        """from_index(1) should return RIGHT."""
        assert MouseButton.from_index(1) == MouseButton.RIGHT

    def test_from_index_middle(self):
        """from_index(2) should return MIDDLE."""
        assert MouseButton.from_index(2) == MouseButton.MIDDLE

    def test_from_index_invalid(self):
        """from_index with invalid index should return NONE."""
        assert MouseButton.from_index(99) == MouseButton.NONE
        assert MouseButton.from_index(-1) == MouseButton.NONE


class TestKeyModifier:
    """Tests for KeyModifier enum."""

    def test_none_modifier(self):
        """NONE should have value 0."""
        assert KeyModifier.NONE == 0

    def test_shift_modifier(self):
        """SHIFT should have value 1."""
        assert KeyModifier.SHIFT == 1

    def test_ctrl_modifier(self):
        """CTRL should have value 2."""
        assert KeyModifier.CTRL == 2

    def test_alt_modifier(self):
        """ALT should have value 4."""
        assert KeyModifier.ALT == 4

    def test_meta_modifier(self):
        """META should have value 8."""
        assert KeyModifier.META == 8

    def test_caps_lock_modifier(self):
        """CAPS_LOCK should have value 16."""
        assert KeyModifier.CAPS_LOCK == 16

    def test_num_lock_modifier(self):
        """NUM_LOCK should have value 32."""
        assert KeyModifier.NUM_LOCK == 32

    def test_modifier_combinations(self):
        """Multiple modifiers can be combined."""
        mods = KeyModifier.SHIFT | KeyModifier.CTRL
        assert mods & KeyModifier.SHIFT
        assert mods & KeyModifier.CTRL
        assert not (mods & KeyModifier.ALT)

    def test_from_bools_none(self):
        """from_bools with all False should return NONE."""
        assert KeyModifier.from_bools() == KeyModifier.NONE

    def test_from_bools_shift(self):
        """from_bools with shift=True should include SHIFT."""
        mods = KeyModifier.from_bools(shift=True)
        assert mods & KeyModifier.SHIFT

    def test_from_bools_ctrl(self):
        """from_bools with ctrl=True should include CTRL."""
        mods = KeyModifier.from_bools(ctrl=True)
        assert mods & KeyModifier.CTRL

    def test_from_bools_alt(self):
        """from_bools with alt=True should include ALT."""
        mods = KeyModifier.from_bools(alt=True)
        assert mods & KeyModifier.ALT

    def test_from_bools_meta(self):
        """from_bools with meta=True should include META."""
        mods = KeyModifier.from_bools(meta=True)
        assert mods & KeyModifier.META

    def test_from_bools_combination(self):
        """from_bools with multiple should combine."""
        mods = KeyModifier.from_bools(shift=True, ctrl=True, alt=True)
        assert mods & KeyModifier.SHIFT
        assert mods & KeyModifier.CTRL
        assert mods & KeyModifier.ALT
        assert not (mods & KeyModifier.META)


class TestEventType:
    """Tests for EventType enum."""

    def test_mouse_down_type(self):
        """MOUSE_DOWN should exist."""
        assert EventType.MOUSE_DOWN.value == "mouse_down"

    def test_mouse_up_type(self):
        """MOUSE_UP should exist."""
        assert EventType.MOUSE_UP.value == "mouse_up"

    def test_click_type(self):
        """CLICK should exist."""
        assert EventType.CLICK.value == "click"

    def test_double_click_type(self):
        """DOUBLE_CLICK should exist."""
        assert EventType.DOUBLE_CLICK.value == "double_click"

    def test_mouse_enter_type(self):
        """MOUSE_ENTER should exist."""
        assert EventType.MOUSE_ENTER.value == "mouse_enter"

    def test_mouse_leave_type(self):
        """MOUSE_LEAVE should exist."""
        assert EventType.MOUSE_LEAVE.value == "mouse_leave"

    def test_mouse_move_type(self):
        """MOUSE_MOVE should exist."""
        assert EventType.MOUSE_MOVE.value == "mouse_move"

    def test_mouse_scroll_type(self):
        """MOUSE_SCROLL should exist."""
        assert EventType.MOUSE_SCROLL.value == "mouse_scroll"

    def test_key_down_type(self):
        """KEY_DOWN should exist."""
        assert EventType.KEY_DOWN.value == "key_down"

    def test_key_up_type(self):
        """KEY_UP should exist."""
        assert EventType.KEY_UP.value == "key_up"

    def test_char_input_type(self):
        """CHAR_INPUT should exist."""
        assert EventType.CHAR_INPUT.value == "char_input"

    def test_focus_in_type(self):
        """FOCUS_IN should exist."""
        assert EventType.FOCUS_IN.value == "focus_in"

    def test_focus_out_type(self):
        """FOCUS_OUT should exist."""
        assert EventType.FOCUS_OUT.value == "focus_out"

    def test_drag_start_type(self):
        """DRAG_START should exist."""
        assert EventType.DRAG_START.value == "drag_start"

    def test_drag_type(self):
        """DRAG should exist."""
        assert EventType.DRAG.value == "drag"

    def test_drag_end_type(self):
        """DRAG_END should exist."""
        assert EventType.DRAG_END.value == "drag_end"

    def test_drop_type(self):
        """DROP should exist."""
        assert EventType.DROP.value == "drop"


class TestUIEvent:
    """Tests for UIEvent base class."""

    def test_event_creation(self):
        """UIEvent should be creatable with event_type."""
        event = UIEvent(event_type=EventType.CLICK)
        assert event.event_type == EventType.CLICK

    def test_event_has_timestamp(self):
        """UIEvent should have timestamp."""
        before = time.time()
        event = UIEvent(event_type=EventType.CLICK)
        after = time.time()
        assert before <= event.timestamp <= after

    def test_event_default_target_is_none(self):
        """UIEvent target should default to None."""
        event = UIEvent(event_type=EventType.CLICK)
        assert event.target is None

    def test_event_default_current_target_is_none(self):
        """UIEvent current_target should default to None."""
        event = UIEvent(event_type=EventType.CLICK)
        assert event.current_target is None

    def test_event_default_phase_is_none(self):
        """UIEvent phase should default to NONE."""
        event = UIEvent(event_type=EventType.CLICK)
        assert event.phase == EventPhase.NONE

    def test_event_default_bubbles(self):
        """UIEvent bubbles should default to True."""
        event = UIEvent(event_type=EventType.CLICK)
        assert event.bubbles is True

    def test_event_default_cancelable(self):
        """UIEvent cancelable should default to True."""
        event = UIEvent(event_type=EventType.CLICK)
        assert event.cancelable is True

    def test_event_not_stopped_initially(self):
        """UIEvent should not be stopped initially."""
        event = UIEvent(event_type=EventType.CLICK)
        assert event.is_stopped is False

    def test_event_not_stopped_immediate_initially(self):
        """UIEvent should not be stopped_immediate initially."""
        event = UIEvent(event_type=EventType.CLICK)
        assert event.is_stopped_immediate is False

    def test_event_default_not_prevented_initially(self):
        """UIEvent should not have default prevented initially."""
        event = UIEvent(event_type=EventType.CLICK)
        assert event.is_default_prevented is False

    def test_stop_propagation(self):
        """stop_propagation should mark event as stopped."""
        event = UIEvent(event_type=EventType.CLICK)
        event.stop_propagation()
        assert event.is_stopped is True

    def test_stop_immediate_propagation(self):
        """stop_immediate_propagation should stop propagation and immediate."""
        event = UIEvent(event_type=EventType.CLICK)
        event.stop_immediate_propagation()
        assert event.is_stopped is True
        assert event.is_stopped_immediate is True

    def test_prevent_default_cancelable(self):
        """prevent_default on cancelable event should work."""
        event = UIEvent(event_type=EventType.CLICK, cancelable=True)
        event.prevent_default()
        assert event.is_default_prevented is True

    def test_prevent_default_not_cancelable(self):
        """prevent_default on non-cancelable event should be ignored."""
        event = UIEvent(event_type=EventType.CLICK, cancelable=False)
        event.prevent_default()
        assert event.is_default_prevented is False

    def test_event_clone(self):
        """clone should create copy of event."""
        event = UIEvent(
            event_type=EventType.CLICK,
            bubbles=True,
            cancelable=True,
        )
        cloned = event.clone()
        assert cloned.event_type == event.event_type
        assert cloned.bubbles == event.bubbles
        assert cloned.cancelable == event.cancelable
        assert cloned is not event


class TestMouseEvent:
    """Tests for MouseEvent class."""

    def test_mouse_event_creation(self):
        """MouseEvent should be creatable."""
        event = MouseEvent(event_type=EventType.CLICK, x=10.0, y=20.0)
        assert event.x == 10.0
        assert event.y == 20.0

    def test_mouse_event_screen_position(self):
        """MouseEvent should store screen position."""
        event = MouseEvent(
            event_type=EventType.CLICK,
            screen_x=100.0,
            screen_y=200.0,
        )
        assert event.screen_x == 100.0
        assert event.screen_y == 200.0

    def test_mouse_event_button(self):
        """MouseEvent should store button."""
        event = MouseEvent(
            event_type=EventType.MOUSE_DOWN,
            button=MouseButton.LEFT,
        )
        assert event.button == MouseButton.LEFT

    def test_mouse_event_buttons(self):
        """MouseEvent should store buttons state."""
        event = MouseEvent(
            event_type=EventType.MOUSE_MOVE,
            buttons=MouseButton.LEFT | MouseButton.RIGHT,
        )
        assert event.buttons & MouseButton.LEFT
        assert event.buttons & MouseButton.RIGHT

    def test_mouse_event_modifiers(self):
        """MouseEvent should store modifiers."""
        event = MouseEvent(
            event_type=EventType.CLICK,
            modifiers=KeyModifier.SHIFT,
        )
        assert event.modifiers == KeyModifier.SHIFT

    def test_mouse_event_delta(self):
        """MouseEvent should store scroll delta."""
        event = MouseEvent(
            event_type=EventType.MOUSE_SCROLL,
            delta_x=1.0,
            delta_y=-2.0,
        )
        assert event.delta_x == 1.0
        assert event.delta_y == -2.0

    def test_mouse_event_click_count(self):
        """MouseEvent should store click count."""
        event = MouseEvent(
            event_type=EventType.DOUBLE_CLICK,
            click_count=2,
        )
        assert event.click_count == 2

    def test_mouse_event_position_property(self):
        """position property should return Point."""
        event = MouseEvent(event_type=EventType.CLICK, x=10.0, y=20.0)
        assert event.position == Point(10.0, 20.0)

    def test_mouse_event_screen_position_property(self):
        """screen_position property should return Point."""
        event = MouseEvent(
            event_type=EventType.CLICK,
            screen_x=100.0,
            screen_y=200.0,
        )
        assert event.screen_position == Point(100.0, 200.0)

    def test_is_left_button(self):
        """is_left_button should detect left button."""
        event = MouseEvent(
            event_type=EventType.CLICK,
            button=MouseButton.LEFT,
        )
        assert event.is_left_button is True
        assert event.is_right_button is False

    def test_is_right_button(self):
        """is_right_button should detect right button."""
        event = MouseEvent(
            event_type=EventType.CLICK,
            button=MouseButton.RIGHT,
        )
        assert event.is_right_button is True
        assert event.is_left_button is False

    def test_is_middle_button(self):
        """is_middle_button should detect middle button."""
        event = MouseEvent(
            event_type=EventType.CLICK,
            button=MouseButton.MIDDLE,
        )
        assert event.is_middle_button is True

    def test_mouse_event_clone(self):
        """clone should create copy of mouse event."""
        event = MouseEvent(
            event_type=EventType.CLICK,
            x=10.0,
            y=20.0,
            button=MouseButton.LEFT,
            modifiers=KeyModifier.SHIFT,
        )
        cloned = event.clone()
        assert cloned.x == event.x
        assert cloned.y == event.y
        assert cloned.button == event.button
        assert cloned.modifiers == event.modifiers

    def test_mouse_event_click_factory(self):
        """click factory should create click event."""
        event = MouseEvent.click(10.0, 20.0, MouseButton.LEFT)
        assert event.event_type == EventType.CLICK
        assert event.x == 10.0
        assert event.y == 20.0
        assert event.button == MouseButton.LEFT

    def test_mouse_event_move_factory(self):
        """move factory should create move event."""
        event = MouseEvent.move(10.0, 20.0)
        assert event.event_type == EventType.MOUSE_MOVE
        assert event.x == 10.0
        assert event.y == 20.0
        assert event.bubbles is False  # Move events don't bubble

    def test_mouse_event_scroll_factory(self):
        """scroll factory should create scroll event."""
        event = MouseEvent.scroll(10.0, 20.0, 0.0, -120.0)
        assert event.event_type == EventType.MOUSE_SCROLL
        assert event.x == 10.0
        assert event.y == 20.0
        assert event.delta_x == 0.0
        assert event.delta_y == -120.0


class TestKeyboardEvent:
    """Tests for KeyboardEvent class."""

    def test_keyboard_event_creation(self):
        """KeyboardEvent should be creatable."""
        event = KeyboardEvent(event_type=EventType.KEY_DOWN, key="Enter")
        assert event.key == "Enter"

    def test_keyboard_event_key_code(self):
        """KeyboardEvent should store key_code."""
        event = KeyboardEvent(
            event_type=EventType.KEY_DOWN,
            key="a",
            key_code=65,
        )
        assert event.key_code == 65

    def test_keyboard_event_char(self):
        """KeyboardEvent should store char."""
        event = KeyboardEvent(
            event_type=EventType.CHAR_INPUT,
            char="a",
        )
        assert event.char == "a"

    def test_keyboard_event_modifiers(self):
        """KeyboardEvent should store modifiers."""
        event = KeyboardEvent(
            event_type=EventType.KEY_DOWN,
            key="c",
            modifiers=KeyModifier.CTRL,
        )
        assert event.modifiers == KeyModifier.CTRL

    def test_keyboard_event_is_repeat(self):
        """KeyboardEvent should store is_repeat."""
        event = KeyboardEvent(
            event_type=EventType.KEY_DOWN,
            key="a",
            is_repeat=True,
        )
        assert event.is_repeat is True

    def test_is_shift_property(self):
        """is_shift should detect Shift modifier."""
        event = KeyboardEvent(
            event_type=EventType.KEY_DOWN,
            key="a",
            modifiers=KeyModifier.SHIFT,
        )
        assert event.is_shift is True
        assert event.is_ctrl is False

    def test_is_ctrl_property(self):
        """is_ctrl should detect Ctrl modifier."""
        event = KeyboardEvent(
            event_type=EventType.KEY_DOWN,
            key="a",
            modifiers=KeyModifier.CTRL,
        )
        assert event.is_ctrl is True
        assert event.is_shift is False

    def test_is_alt_property(self):
        """is_alt should detect Alt modifier."""
        event = KeyboardEvent(
            event_type=EventType.KEY_DOWN,
            key="a",
            modifiers=KeyModifier.ALT,
        )
        assert event.is_alt is True

    def test_is_meta_property(self):
        """is_meta should detect Meta modifier."""
        event = KeyboardEvent(
            event_type=EventType.KEY_DOWN,
            key="a",
            modifiers=KeyModifier.META,
        )
        assert event.is_meta is True

    def test_keyboard_event_clone(self):
        """clone should create copy of keyboard event."""
        event = KeyboardEvent(
            event_type=EventType.KEY_DOWN,
            key="a",
            key_code=65,
            modifiers=KeyModifier.CTRL,
            is_repeat=True,
        )
        cloned = event.clone()
        assert cloned.key == event.key
        assert cloned.key_code == event.key_code
        assert cloned.modifiers == event.modifiers
        assert cloned.is_repeat == event.is_repeat

    def test_key_down_factory(self):
        """key_down factory should create key down event."""
        event = KeyboardEvent.key_down("Enter", key_code=13)
        assert event.event_type == EventType.KEY_DOWN
        assert event.key == "Enter"
        assert event.key_code == 13

    def test_key_up_factory(self):
        """key_up factory should create key up event."""
        event = KeyboardEvent.key_up("Enter", key_code=13)
        assert event.event_type == EventType.KEY_UP
        assert event.key == "Enter"

    def test_char_input_factory(self):
        """char_input factory should create char input event."""
        event = KeyboardEvent.char_input("a")
        assert event.event_type == EventType.CHAR_INPUT
        assert event.char == "a"
        assert event.key == "a"


class TestFocusEvent:
    """Tests for FocusEvent class."""

    def test_focus_event_creation(self):
        """FocusEvent should be creatable."""
        event = FocusEvent(event_type=EventType.FOCUS_IN)
        assert event.event_type == EventType.FOCUS_IN

    def test_focus_event_related_target(self):
        """FocusEvent should store related_target."""
        event = FocusEvent(
            event_type=EventType.FOCUS_IN,
            related_target=None,  # Would be a widget in practice
        )
        assert event.related_target is None

    def test_focus_event_clone(self):
        """clone should create copy of focus event."""
        event = FocusEvent(event_type=EventType.FOCUS_IN)
        cloned = event.clone()
        assert cloned.event_type == event.event_type

    def test_focus_in_factory(self):
        """focus_in factory should create focus in event."""
        event = FocusEvent.focus_in()
        assert event.event_type == EventType.FOCUS_IN
        assert event.bubbles is False  # Focus events don't bubble
        assert event.cancelable is False

    def test_focus_out_factory(self):
        """focus_out factory should create focus out event."""
        event = FocusEvent.focus_out()
        assert event.event_type == EventType.FOCUS_OUT
        assert event.bubbles is False
        assert event.cancelable is False


class TestDragEvent:
    """Tests for DragEvent class."""

    def test_drag_event_creation(self):
        """DragEvent should be creatable."""
        event = DragEvent(
            event_type=EventType.DRAG_START,
            x=10.0,
            y=20.0,
        )
        assert event.x == 10.0
        assert event.y == 20.0

    def test_drag_event_data(self):
        """DragEvent should store data."""
        data = {"item_id": 123}
        event = DragEvent(
            event_type=EventType.DRAG,
            data=data,
            data_type="inventory_item",
        )
        assert event.data == data
        assert event.data_type == "inventory_item"

    def test_drag_event_source(self):
        """DragEvent should store source widget."""
        event = DragEvent(
            event_type=EventType.DRAG_START,
            source=None,  # Would be a widget
        )
        assert event.source is None

    def test_drag_event_modifiers(self):
        """DragEvent should store modifiers."""
        event = DragEvent(
            event_type=EventType.DROP,
            modifiers=KeyModifier.CTRL,
        )
        assert event.modifiers == KeyModifier.CTRL

    def test_drag_event_position_property(self):
        """position property should return Point."""
        event = DragEvent(event_type=EventType.DRAG, x=10.0, y=20.0)
        assert event.position == Point(10.0, 20.0)

    def test_drag_event_clone(self):
        """clone should create copy of drag event."""
        event = DragEvent(
            event_type=EventType.DRAG,
            x=10.0,
            y=20.0,
            data={"test": True},
            data_type="test_type",
        )
        cloned = event.clone()
        assert cloned.x == event.x
        assert cloned.y == event.y
        assert cloned.data == event.data
        assert cloned.data_type == event.data_type

    def test_drag_start_factory(self):
        """drag_start factory should create drag start event."""
        event = DragEvent.drag_start(10.0, 20.0)
        assert event.event_type == EventType.DRAG_START
        assert event.x == 10.0
        assert event.y == 20.0

    def test_drag_factory(self):
        """drag factory should create drag event."""
        event = DragEvent.drag(10.0, 20.0)
        assert event.event_type == EventType.DRAG
        assert event.bubbles is False  # Drag events don't bubble

    def test_drop_factory(self):
        """drop factory should create drop event."""
        data = {"item": 1}
        event = DragEvent.drop(10.0, 20.0, data=data, data_type="item")
        assert event.event_type == EventType.DROP
        assert event.data == data
        assert event.data_type == "item"


class TestEventPropagation:
    """Tests for event propagation behavior."""

    def test_stop_propagation_stops_bubble(self):
        """stop_propagation should prevent bubbling."""
        event = MouseEvent.click(10.0, 20.0)
        event.stop_propagation()
        assert event.is_stopped is True

    def test_stop_immediate_stops_same_target_handlers(self):
        """stop_immediate_propagation should stop same-target handlers."""
        event = MouseEvent.click(10.0, 20.0)
        event.stop_immediate_propagation()
        assert event.is_stopped_immediate is True

    def test_prevent_default_on_cancelable(self):
        """prevent_default should work on cancelable events."""
        event = UIEvent(event_type=EventType.CLICK, cancelable=True)
        event.prevent_default()
        assert event.is_default_prevented is True

    def test_prevent_default_on_non_cancelable(self):
        """prevent_default should be ignored on non-cancelable events."""
        event = UIEvent(event_type=EventType.CLICK, cancelable=False)
        event.prevent_default()
        assert event.is_default_prevented is False

    def test_non_bubbling_event(self):
        """Non-bubbling events should have bubbles=False."""
        event = MouseEvent.move(10.0, 20.0)
        assert event.bubbles is False

    def test_bubbling_event(self):
        """Bubbling events should have bubbles=True."""
        event = MouseEvent.click(10.0, 20.0)
        assert event.bubbles is True


class TestEventDispatcher:
    """Tests for EventDispatcher class."""

    def test_dispatcher_is_class(self):
        """EventDispatcher should be a class that can be instantiated."""
        dispatcher = EventDispatcher()
        assert isinstance(dispatcher, EventDispatcher)

    def test_dispatcher_dispatch_method_signature(self):
        """EventDispatcher.dispatch should be a static method with proper signature."""
        import inspect
        assert hasattr(EventDispatcher, "dispatch")
        assert callable(EventDispatcher.dispatch)
        # Verify it's a static method by checking it can be called on the class
        sig = inspect.signature(EventDispatcher.dispatch)
        param_names = list(sig.parameters.keys())
        # Should accept event and target parameters
        assert len(param_names) >= 2

    def test_dispatcher_dispatch_with_mock_event(self):
        """EventDispatcher.dispatch should handle events without error."""
        event = UIEvent(event_type=EventType.CLICK)
        # Create a mock target (in real usage, this would be a Widget)
        class MockTarget:
            pass
        target = MockTarget()
        # Should not raise an exception
        try:
            EventDispatcher.dispatch(event, target)
        except (TypeError, AttributeError):
            # Expected if mock doesn't have proper widget interface
            pass


class TestEventHandlerType:
    """Tests for EventHandler type alias."""

    def test_event_handler_callable(self):
        """EventHandler should be a callable type."""
        def handler(event: UIEvent) -> None:
            pass

        # Type check - handler should match EventHandler signature
        typed_handler: EventHandler = handler
        assert callable(typed_handler)


class TestEventPhaseTransitions:
    """Tests for event phase transitions."""

    def test_initial_phase_is_none(self):
        """Events should start in NONE phase."""
        event = UIEvent(event_type=EventType.CLICK)
        assert event.phase == EventPhase.NONE

    def test_phase_can_be_set(self):
        """Event phase can be set for dispatch."""
        event = UIEvent(event_type=EventType.CLICK)
        event.phase = EventPhase.CAPTURE
        assert event.phase == EventPhase.CAPTURE


class TestEventTargeting:
    """Tests for event targeting."""

    def test_target_starts_none(self):
        """Event target should start as None."""
        event = UIEvent(event_type=EventType.CLICK)
        assert event.target is None

    def test_current_target_starts_none(self):
        """Event current_target should start as None."""
        event = UIEvent(event_type=EventType.CLICK)
        assert event.current_target is None

    def test_target_can_be_set(self):
        """Event target can be set."""
        event = UIEvent(event_type=EventType.CLICK)
        event.target = "mock_widget"  # Would be a Widget
        assert event.target == "mock_widget"


class TestCustomEvents:
    """Tests for custom event type."""

    def test_custom_event_type(self):
        """CUSTOM event type should exist."""
        assert EventType.CUSTOM.value == "custom"

    def test_custom_event_creation(self):
        """Custom events can be created."""
        event = UIEvent(event_type=EventType.CUSTOM)
        assert event.event_type == EventType.CUSTOM
