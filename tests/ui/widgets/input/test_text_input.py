"""
Comprehensive tests for the TextInput widget.

Tests cover:
- Initialization and default values
- Text entry and editing
- Cursor movement and positioning
- Text selection
- Copy/cut/paste operations
- Input validation
- Single-line and multi-line modes
- Placeholder text
- Password masking

Note: The text_input.py source file may not exist yet. These tests are written
based on the expected API from the input __init__.py exports.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def text_input_class():
    """Get TextInput class if available."""
    try:
        from engine.ui.widgets.input.text_input import TextInput
        if hasattr(TextInput, 'reset_id_counter'):
            TextInput.reset_id_counter()
        return TextInput
    except ImportError:
        pytest.skip("text_input.py not yet implemented")


class TestInputMode:
    """Tests for InputMode enumeration."""

    def test_input_mode_text(self):
        """Test TEXT input mode exists."""
        try:
            from engine.ui.widgets.input.text_input import InputMode
            assert InputMode.TEXT is not None
        except ImportError:
            pytest.skip("text_input.py not yet implemented")

    def test_input_mode_password(self):
        """Test PASSWORD input mode exists."""
        try:
            from engine.ui.widgets.input.text_input import InputMode
            assert InputMode.PASSWORD is not None
        except ImportError:
            pytest.skip("text_input.py not yet implemented")

    def test_input_mode_number(self):
        """Test NUMBER input mode exists."""
        try:
            from engine.ui.widgets.input.text_input import InputMode
            assert InputMode.NUMBER is not None
        except ImportError:
            pytest.skip("text_input.py not yet implemented")

    def test_input_mode_email(self):
        """Test EMAIL input mode exists."""
        try:
            from engine.ui.widgets.input.text_input import InputMode
            assert InputMode.EMAIL is not None
        except ImportError:
            pytest.skip("text_input.py not yet implemented")


class TestSelectionRange:
    """Tests for SelectionRange."""

    def test_selection_range_creation(self):
        """Test creating selection range."""
        try:
            from engine.ui.widgets.input.text_input import SelectionRange
            selection = SelectionRange(start=5, end=10)
            assert selection.start == 5
            assert selection.end == 10
        except ImportError:
            pytest.skip("text_input.py not yet implemented")

    def test_selection_range_length(self):
        """Test selection length calculation."""
        try:
            from engine.ui.widgets.input.text_input import SelectionRange
            selection = SelectionRange(start=5, end=15)
            assert selection.length == 10
        except ImportError:
            pytest.skip("text_input.py not yet implemented")

    def test_selection_range_is_collapsed(self):
        """Test collapsed selection (cursor position)."""
        try:
            from engine.ui.widgets.input.text_input import SelectionRange
            selection = SelectionRange(start=5, end=5)
            assert selection.is_collapsed is True

            selection2 = SelectionRange(start=5, end=10)
            assert selection2.is_collapsed is False
        except ImportError:
            pytest.skip("text_input.py not yet implemented")

    def test_selection_range_normalized(self):
        """Test normalized selection (start <= end)."""
        try:
            from engine.ui.widgets.input.text_input import SelectionRange
            selection = SelectionRange(start=10, end=5)
            normalized = selection.normalized()
            assert normalized.start == 5
            assert normalized.end == 10
        except ImportError:
            pytest.skip("text_input.py not yet implemented")


class TestTextInputInitialization:
    """Tests for TextInput initialization."""

    def test_text_input_default_initialization(self, text_input_class):
        """Test TextInput initializes with correct defaults."""
        text_input = text_input_class()
        assert text_input.text == ""
        assert text_input.placeholder == ""
        assert text_input.enabled is True
        assert text_input.visible is True
        assert text_input.multiline is False

    def test_text_input_with_initial_text(self, text_input_class):
        """Test TextInput with initial text."""
        text_input = text_input_class(text="Hello")
        assert text_input.text == "Hello"

    def test_text_input_with_placeholder(self, text_input_class):
        """Test TextInput with placeholder."""
        text_input = text_input_class(placeholder="Enter name...")
        assert text_input.placeholder == "Enter name..."

    def test_text_input_multiline(self, text_input_class):
        """Test TextInput in multiline mode."""
        text_input = text_input_class(multiline=True)
        assert text_input.multiline is True

    def test_text_input_password_mode(self, text_input_class):
        """Test TextInput in password mode."""
        try:
            from engine.ui.widgets.input.text_input import InputMode
            text_input = text_input_class(input_mode=InputMode.PASSWORD)
            assert text_input.input_mode == InputMode.PASSWORD
        except ImportError:
            pytest.skip("InputMode not available")

    def test_text_input_max_length(self, text_input_class):
        """Test TextInput with max length."""
        text_input = text_input_class(max_length=100)
        assert text_input.max_length == 100


class TestTextInputProperties:
    """Tests for TextInput property getters and setters."""

    def test_text_input_text_setter(self, text_input_class):
        """Test setting text."""
        text_input = text_input_class()
        text_input.text = "New text"
        assert text_input.text == "New text"

    def test_text_input_text_setter_truncates(self, text_input_class):
        """Test setting text respects max length."""
        text_input = text_input_class(max_length=5)
        text_input.text = "Hello World"
        assert len(text_input.text) <= 5

    def test_text_input_placeholder_setter(self, text_input_class):
        """Test setting placeholder."""
        text_input = text_input_class()
        text_input.placeholder = "Type here..."
        assert text_input.placeholder == "Type here..."

    def test_text_input_cursor_position(self, text_input_class):
        """Test cursor position property."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 3
        assert text_input.cursor_position == 3

    def test_text_input_cursor_position_clamped(self, text_input_class):
        """Test cursor position is clamped to text length."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 100
        assert text_input.cursor_position == 5  # len("Hello")

    def test_text_input_cursor_position_negative(self, text_input_class):
        """Test negative cursor position clamps to 0."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = -5
        assert text_input.cursor_position == 0

    def test_text_input_selection(self, text_input_class):
        """Test selection property."""
        try:
            from engine.ui.widgets.input.text_input import SelectionRange
            text_input = text_input_class(text="Hello World")
            text_input.selection = SelectionRange(0, 5)
            assert text_input.selection.start == 0
            assert text_input.selection.end == 5
        except ImportError:
            pytest.skip("SelectionRange not available")

    def test_text_input_selected_text(self, text_input_class):
        """Test getting selected text."""
        try:
            from engine.ui.widgets.input.text_input import SelectionRange
            text_input = text_input_class(text="Hello World")
            text_input.selection = SelectionRange(0, 5)
            assert text_input.selected_text == "Hello"
        except ImportError:
            pytest.skip("SelectionRange not available")

    def test_text_input_has_selection(self, text_input_class):
        """Test has_selection property."""
        try:
            from engine.ui.widgets.input.text_input import SelectionRange
            text_input = text_input_class(text="Hello")
            text_input.selection = SelectionRange(0, 3)
            assert text_input.has_selection is True

            text_input.selection = SelectionRange(3, 3)
            assert text_input.has_selection is False
        except ImportError:
            pytest.skip("SelectionRange not available")


class TestTextInputCursorMovement:
    """Tests for cursor movement."""

    def test_text_input_move_cursor_left(self, text_input_class):
        """Test moving cursor left."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 3
        text_input.move_cursor(-1)
        assert text_input.cursor_position == 2

    def test_text_input_move_cursor_right(self, text_input_class):
        """Test moving cursor right."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 2
        text_input.move_cursor(1)
        assert text_input.cursor_position == 3

    def test_text_input_move_cursor_word_left(self, text_input_class):
        """Test moving cursor by word left."""
        text_input = text_input_class(text="Hello World")
        text_input.cursor_position = 8
        text_input.move_cursor_by_word(-1)
        assert text_input.cursor_position == 6  # Start of "World"

    def test_text_input_move_cursor_word_right(self, text_input_class):
        """Test moving cursor by word right."""
        text_input = text_input_class(text="Hello World")
        text_input.cursor_position = 2
        text_input.move_cursor_by_word(1)
        assert text_input.cursor_position == 5  # End of "Hello"

    def test_text_input_move_cursor_to_start(self, text_input_class):
        """Test moving cursor to start."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 3
        text_input.move_cursor_to_start()
        assert text_input.cursor_position == 0

    def test_text_input_move_cursor_to_end(self, text_input_class):
        """Test moving cursor to end."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 2
        text_input.move_cursor_to_end()
        assert text_input.cursor_position == 5


class TestTextInputSelection:
    """Tests for text selection."""

    def test_text_input_select_all(self, text_input_class):
        """Test selecting all text."""
        text_input = text_input_class(text="Hello World")
        text_input.select_all()
        assert text_input.selected_text == "Hello World"

    def test_text_input_select_word(self, text_input_class):
        """Test selecting a word."""
        text_input = text_input_class(text="Hello World")
        text_input.cursor_position = 2
        text_input.select_word()
        assert text_input.selected_text == "Hello"

    def test_text_input_clear_selection(self, text_input_class):
        """Test clearing selection."""
        try:
            from engine.ui.widgets.input.text_input import SelectionRange
            text_input = text_input_class(text="Hello")
            text_input.selection = SelectionRange(0, 3)
            text_input.clear_selection()
            assert text_input.has_selection is False
        except ImportError:
            pytest.skip("SelectionRange not available")

    def test_text_input_select_range(self, text_input_class):
        """Test selecting a range."""
        text_input = text_input_class(text="Hello World")
        text_input.select_range(6, 11)
        assert text_input.selected_text == "World"

    def test_text_input_extend_selection_left(self, text_input_class):
        """Test extending selection left."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 3
        text_input.extend_selection(-2)
        assert text_input.selection.start == 1
        assert text_input.selection.end == 3

    def test_text_input_extend_selection_right(self, text_input_class):
        """Test extending selection right."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 2
        text_input.extend_selection(2)
        assert text_input.selection.start == 2
        assert text_input.selection.end == 4


class TestTextInputEditing:
    """Tests for text editing operations."""

    def test_text_input_insert_text(self, text_input_class):
        """Test inserting text at cursor."""
        text_input = text_input_class(text="Helo")
        text_input.cursor_position = 3
        text_input.insert_text("l")
        assert text_input.text == "Hello"

    def test_text_input_insert_text_replaces_selection(self, text_input_class):
        """Test inserting text replaces selection."""
        text_input = text_input_class(text="Hello World")
        text_input.select_range(0, 5)
        text_input.insert_text("Hi")
        assert text_input.text == "Hi World"

    def test_text_input_delete_character_forward(self, text_input_class):
        """Test deleting character forward (Delete key)."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 1
        text_input.delete_forward()
        assert text_input.text == "Hllo"

    def test_text_input_delete_character_backward(self, text_input_class):
        """Test deleting character backward (Backspace)."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 2
        text_input.delete_backward()
        assert text_input.text == "Hllo"
        assert text_input.cursor_position == 1

    def test_text_input_delete_selection(self, text_input_class):
        """Test deleting selected text."""
        text_input = text_input_class(text="Hello World")
        text_input.select_range(5, 11)
        text_input.delete_selection()
        assert text_input.text == "Hello"

    def test_text_input_delete_word_backward(self, text_input_class):
        """Test deleting word backward."""
        text_input = text_input_class(text="Hello World")
        text_input.cursor_position = 11
        text_input.delete_word_backward()
        assert text_input.text == "Hello "

    def test_text_input_delete_word_forward(self, text_input_class):
        """Test deleting word forward."""
        text_input = text_input_class(text="Hello World")
        text_input.cursor_position = 6
        text_input.delete_word_forward()
        assert text_input.text == "Hello "

    def test_text_input_clear(self, text_input_class):
        """Test clearing all text."""
        text_input = text_input_class(text="Hello World")
        text_input.clear()
        assert text_input.text == ""
        assert text_input.cursor_position == 0


class TestTextInputClipboard:
    """Tests for copy/cut/paste operations."""

    def test_text_input_copy(self, text_input_class):
        """Test copying selected text."""
        text_input = text_input_class(text="Hello World")
        text_input.select_range(0, 5)
        clipboard_text = text_input.copy()
        assert clipboard_text == "Hello"
        # Original text unchanged
        assert text_input.text == "Hello World"

    def test_text_input_cut(self, text_input_class):
        """Test cutting selected text."""
        text_input = text_input_class(text="Hello World")
        text_input.select_range(0, 6)
        clipboard_text = text_input.cut()
        assert clipboard_text == "Hello "
        assert text_input.text == "World"

    def test_text_input_paste(self, text_input_class):
        """Test pasting text."""
        text_input = text_input_class(text="World")
        text_input.cursor_position = 0
        text_input.paste("Hello ")
        assert text_input.text == "Hello World"

    def test_text_input_paste_replaces_selection(self, text_input_class):
        """Test pasting replaces selection."""
        text_input = text_input_class(text="Hello World")
        text_input.select_range(6, 11)
        text_input.paste("Universe")
        assert text_input.text == "Hello Universe"

    def test_text_input_paste_respects_max_length(self, text_input_class):
        """Test paste respects max length."""
        text_input = text_input_class(text="", max_length=10)
        text_input.paste("This is a very long string")
        assert len(text_input.text) <= 10


class TestTextInputValidation:
    """Tests for input validation."""

    def test_text_input_validation_passes(self, text_input_class):
        """Test validation that passes."""
        def validate(text):
            return len(text) >= 3, "Too short"

        text_input = text_input_class(validator=validate)
        text_input.text = "Hello"

        result = text_input.validate()
        assert result.is_valid is True

    def test_text_input_validation_fails(self, text_input_class):
        """Test validation that fails."""
        def validate(text):
            if len(text) < 3:
                return False, "Too short"
            return True, None

        text_input = text_input_class(validator=validate)
        text_input.text = "Hi"

        result = text_input.validate()
        assert result.is_valid is False
        assert result.message == "Too short"

    def test_text_input_regex_pattern(self, text_input_class):
        """Test regex pattern validation."""
        text_input = text_input_class(pattern=r"^\d+$")
        text_input.text = "123"
        assert text_input.is_valid is True

        text_input.text = "abc"
        assert text_input.is_valid is False

    def test_text_input_number_mode_filters(self, text_input_class):
        """Test number mode filters non-numeric input."""
        try:
            from engine.ui.widgets.input.text_input import InputMode
            text_input = text_input_class(input_mode=InputMode.NUMBER)
            text_input.insert_text("abc123def")
            assert text_input.text == "123"
        except ImportError:
            pytest.skip("InputMode not available")


class TestTextInputMultiline:
    """Tests for multiline mode."""

    def test_text_input_newline_in_multiline(self, text_input_class):
        """Test newlines allowed in multiline mode."""
        text_input = text_input_class(multiline=True)
        text_input.text = "Line 1\nLine 2"
        assert "\n" in text_input.text

    def test_text_input_newline_in_single_line(self, text_input_class):
        """Test newlines filtered in single-line mode."""
        text_input = text_input_class(multiline=False)
        text_input.insert_text("Line 1\nLine 2")
        assert "\n" not in text_input.text

    def test_text_input_line_count(self, text_input_class):
        """Test line count in multiline mode."""
        text_input = text_input_class(multiline=True, text="Line 1\nLine 2\nLine 3")
        assert text_input.line_count == 3

    def test_text_input_current_line(self, text_input_class):
        """Test getting current line number."""
        text_input = text_input_class(multiline=True, text="Line 1\nLine 2\nLine 3")
        text_input.cursor_position = 8  # In "Line 2"
        assert text_input.current_line == 1  # 0-indexed

    def test_text_input_move_cursor_line_up(self, text_input_class):
        """Test moving cursor up a line."""
        text_input = text_input_class(multiline=True, text="Line 1\nLine 2")
        text_input.cursor_position = 10  # In "Line 2"
        text_input.move_cursor_line(-1)
        assert text_input.current_line == 0

    def test_text_input_move_cursor_line_down(self, text_input_class):
        """Test moving cursor down a line."""
        text_input = text_input_class(multiline=True, text="Line 1\nLine 2")
        text_input.cursor_position = 2  # In "Line 1"
        text_input.move_cursor_line(1)
        assert text_input.current_line == 1


class TestTextInputPasswordMode:
    """Tests for password mode."""

    def test_text_input_password_display_text(self, text_input_class):
        """Test password mode shows masked text."""
        try:
            from engine.ui.widgets.input.text_input import InputMode
            text_input = text_input_class(
                text="secret",
                input_mode=InputMode.PASSWORD
            )
            assert text_input.display_text != "secret"
            assert len(text_input.display_text) == 6
        except ImportError:
            pytest.skip("InputMode not available")

    def test_text_input_password_mask_character(self, text_input_class):
        """Test custom password mask character."""
        try:
            from engine.ui.widgets.input.text_input import InputMode
            text_input = text_input_class(
                text="pass",
                input_mode=InputMode.PASSWORD,
                password_char="*"
            )
            assert text_input.display_text == "****"
        except ImportError:
            pytest.skip("InputMode not available")

    def test_text_input_copy_disabled_in_password(self, text_input_class):
        """Test copy is disabled in password mode."""
        try:
            from engine.ui.widgets.input.text_input import InputMode
            text_input = text_input_class(
                text="secret",
                input_mode=InputMode.PASSWORD
            )
            text_input.select_all()
            clipboard = text_input.copy()
            assert clipboard == "" or clipboard is None
        except ImportError:
            pytest.skip("InputMode not available")


class TestTextInputEvents:
    """Tests for text input events."""

    def test_text_input_text_change_event(self, text_input_class):
        """Test text change event is emitted."""
        text_input = text_input_class()
        handler = MagicMock()
        text_input.on_text_change(handler)

        text_input.text = "Hello"

        assert handler.called
        event = handler.call_args[0][0]
        assert event.new_text == "Hello"
        assert event.previous_text == ""

    def test_text_input_selection_change_event(self, text_input_class):
        """Test selection change event is emitted."""
        text_input = text_input_class(text="Hello World")
        handler = MagicMock()
        text_input.on_selection_change(handler)

        text_input.select_all()

        assert handler.called

    def test_text_input_submit_event(self, text_input_class):
        """Test submit event (Enter key in single-line)."""
        text_input = text_input_class(text="Hello")
        handler = MagicMock()
        text_input.on_submit(handler)

        text_input.submit()

        assert handler.called

    def test_text_input_unsubscribe(self, text_input_class):
        """Test unsubscribing from events."""
        text_input = text_input_class()
        handler = MagicMock()

        unsubscribe = text_input.on_text_change(handler)
        unsubscribe()

        text_input.text = "Hello"
        assert not handler.called


class TestTextInputKeyboardHandling:
    """Tests for keyboard input handling."""

    def test_text_input_character_input(self, text_input_class):
        """Test character input."""
        text_input = text_input_class()
        text_input.handle_text_input("H")
        text_input.handle_text_input("i")
        assert text_input.text == "Hi"

    def test_text_input_backspace_key(self, text_input_class):
        """Test backspace key handling."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 5
        text_input.handle_key_down("backspace")
        assert text_input.text == "Hell"

    def test_text_input_delete_key(self, text_input_class):
        """Test delete key handling."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 0
        text_input.handle_key_down("delete")
        assert text_input.text == "ello"

    def test_text_input_left_arrow(self, text_input_class):
        """Test left arrow key."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 3
        text_input.handle_key_down("left")
        assert text_input.cursor_position == 2

    def test_text_input_right_arrow(self, text_input_class):
        """Test right arrow key."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 2
        text_input.handle_key_down("right")
        assert text_input.cursor_position == 3

    def test_text_input_home_key(self, text_input_class):
        """Test home key."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 3
        text_input.handle_key_down("home")
        assert text_input.cursor_position == 0

    def test_text_input_end_key(self, text_input_class):
        """Test end key."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 2
        text_input.handle_key_down("end")
        assert text_input.cursor_position == 5

    def test_text_input_ctrl_a_select_all(self, text_input_class):
        """Test Ctrl+A selects all."""
        text_input = text_input_class(text="Hello World")
        text_input.handle_key_down("a", ctrl=True)
        assert text_input.selected_text == "Hello World"

    def test_text_input_shift_arrow_extends_selection(self, text_input_class):
        """Test Shift+Arrow extends selection."""
        text_input = text_input_class(text="Hello")
        text_input.cursor_position = 2
        text_input.handle_key_down("right", shift=True)
        text_input.handle_key_down("right", shift=True)
        assert text_input.selected_text == "ll"


class TestTextInputMouseHandling:
    """Tests for mouse input handling."""

    def test_text_input_click_positions_cursor(self, text_input_class):
        """Test click positions cursor."""
        text_input = text_input_class(text="Hello World", x=0, y=0, width=200, height=30)
        # Simulate click at a position
        text_input.handle_mouse_down(50, 15)
        text_input.handle_mouse_up(50, 15)
        # Cursor should move (exact position depends on font metrics)
        assert text_input.cursor_position >= 0

    def test_text_input_double_click_selects_word(self, text_input_class):
        """Test double-click selects word."""
        text_input = text_input_class(text="Hello World", x=0, y=0, width=200, height=30)
        text_input.handle_double_click(30, 15)
        # Should select "Hello" or "World" depending on position
        assert text_input.has_selection

    def test_text_input_triple_click_selects_line(self, text_input_class):
        """Test triple-click selects line."""
        text_input = text_input_class(text="Hello World", x=0, y=0, width=200, height=30)
        text_input.handle_triple_click(50, 15)
        assert text_input.selected_text == "Hello World"

    def test_text_input_drag_creates_selection(self, text_input_class):
        """Test dragging creates selection."""
        text_input = text_input_class(text="Hello World", x=0, y=0, width=200, height=30)
        text_input.handle_mouse_down(20, 15)
        text_input.handle_mouse_move(80, 15)
        text_input.handle_mouse_up(80, 15)
        assert text_input.has_selection


class TestTextInputDirtyState:
    """Tests for dirty state tracking."""

    def test_text_input_dirty_after_text_change(self, text_input_class):
        """Test input is dirty after text changes."""
        text_input = text_input_class()
        text_input.mark_clean()
        text_input.text = "Hello"
        assert text_input.is_dirty

    def test_text_input_dirty_after_cursor_move(self, text_input_class):
        """Test input is dirty after cursor moves."""
        text_input = text_input_class(text="Hello")
        text_input.mark_clean()
        text_input.cursor_position = 3
        assert text_input.is_dirty

    def test_text_input_mark_clean(self, text_input_class):
        """Test mark_clean clears dirty state."""
        text_input = text_input_class()
        text_input.mark_clean()
        assert text_input.is_dirty is False
