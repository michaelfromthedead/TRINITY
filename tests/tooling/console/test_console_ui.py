"""Tests for the console UI system.

Tests input/output handling, scrollback, and autocomplete.
"""

import pytest
from datetime import datetime

from engine.tooling.console.console_ui import (
    ConsoleUI,
    ConsoleMode,
    ConsoleConfig,
    OutputType,
    OutputLine,
    AutocompleteResult,
)
from engine.tooling.console.command_history import CommandHistory


class TestConsoleConfig:
    """Tests for ConsoleConfig."""

    def test_defaults(self):
        config = ConsoleConfig()
        assert config.max_scrollback == 1000
        assert config.max_input_length == 4096
        assert config.echo_commands is True
        assert config.autocomplete_enabled is True
        assert config.history_enabled is True


class TestOutputLine:
    """Tests for OutputLine."""

    def test_basic_creation(self):
        line = OutputLine(
            text="Hello, World!",
            output_type=OutputType.INFO
        )
        assert line.text == "Hello, World!"
        assert line.output_type == OutputType.INFO
        assert isinstance(line.timestamp, datetime)

    def test_formatted_output(self):
        config = ConsoleConfig(timestamp_format="%H:%M:%S")
        line = OutputLine(
            text="Test message",
            category="Engine"
        )
        formatted = line.formatted(config)
        assert "Test message" in formatted
        assert "[Engine]" in formatted

    def test_formatted_no_timestamp(self):
        config = ConsoleConfig(timestamp_format=None)
        line = OutputLine(text="Test")
        formatted = line.formatted(config)
        assert formatted == "Test"


class TestConsoleUI:
    """Tests for ConsoleUI."""

    @pytest.fixture
    def console(self):
        """Create a console with default config."""
        return ConsoleUI()

    def test_basic_creation(self, console):
        assert console.mode == ConsoleMode.USER
        assert console.visible is False
        assert console.scrollback_count == 0

    def test_toggle_visibility(self, console):
        assert console.visible is False
        result = console.toggle()
        assert result is True
        assert console.visible is True
        result = console.toggle()
        assert result is False
        assert console.visible is False

    def test_set_mode(self, console):
        console.mode = ConsoleMode.DEVELOPER
        assert console.mode == ConsoleMode.DEVELOPER


class TestConsoleOutput:
    """Tests for console output."""

    @pytest.fixture
    def console(self):
        return ConsoleUI()

    def test_write(self, console):
        console.write("Test message")
        assert console.scrollback_count == 1

    def test_write_with_type(self, console):
        console.write("Error!", OutputType.ERROR)
        lines = console.get_scrollback()
        assert lines[0].output_type == OutputType.ERROR

    def test_write_with_category(self, console):
        console.write("Physics error", category="Physics")
        lines = console.get_scrollback()
        assert lines[0].category == "Physics"

    def test_convenience_write_methods(self, console):
        console.write_info("Info message")
        console.write_warning("Warning message")
        console.write_error("Error message")
        console.write_success("Success message")

        lines = console.get_scrollback()
        assert len(lines) == 4
        assert lines[0].output_type == OutputType.INFO
        assert lines[1].output_type == OutputType.WARNING
        assert lines[2].output_type == OutputType.ERROR
        assert lines[3].output_type == OutputType.SUCCESS

    def test_scrollback_limit(self):
        config = ConsoleConfig(max_scrollback=5)
        console = ConsoleUI(config=config)

        for i in range(10):
            console.write(f"Message {i}")

        assert console.scrollback_count == 5
        lines = console.get_scrollback()
        assert lines[0].text == "Message 5"

    def test_get_scrollback_count(self, console):
        console.write("Line 1")
        console.write("Line 2")
        console.write("Line 3")
        console.write("Line 4")
        console.write("Line 5")

        lines = console.get_scrollback(count=3)
        assert len(lines) == 3
        assert lines[0].text == "Line 3"

    def test_get_scrollback_offset(self, console):
        console.write("Line 1")
        console.write("Line 2")
        console.write("Line 3")
        console.write("Line 4")
        console.write("Line 5")

        # Offset from end, then get count from that position
        # With offset=2, we get lines [1,2,3] (excluding last 2)
        # Then count=2 gets last 2 of those: [2,3]
        lines = console.get_scrollback(count=2, offset=2)
        assert len(lines) == 2
        assert lines[0].text == "Line 2"
        assert lines[1].text == "Line 3"

    def test_clear_scrollback(self, console):
        console.write("Message 1")
        console.write("Message 2")
        console.clear_scrollback()

        # Scrollback cleared but clear message added
        assert console.scrollback_count == 1

    def test_output_callback(self, console):
        received = []

        def callback(line: OutputLine):
            received.append(line)

        console.add_output_callback(callback)
        console.write("Test")

        assert len(received) == 1
        assert received[0].text == "Test"

    def test_remove_output_callback(self, console):
        received = []

        def callback(line: OutputLine):
            received.append(line)

        console.add_output_callback(callback)
        console.remove_output_callback(callback)
        console.write("Test")

        assert len(received) == 0


class TestConsoleFilter:
    """Tests for console output filtering."""

    @pytest.fixture
    def console(self):
        return ConsoleUI()

    def test_filter_by_pattern(self, console):
        console.set_filter(pattern="error")
        console.write("This is an error")
        console.write("This is info")
        console.write("Another error occurred")

        lines = console.get_scrollback()
        assert len(lines) == 2

    def test_filter_by_type(self, console):
        console.set_filter(types={OutputType.ERROR, OutputType.WARNING})
        console.write_info("Info")
        console.write_warning("Warning")
        console.write_error("Error")

        lines = console.get_scrollback()
        assert len(lines) == 2

    def test_clear_filter(self, console):
        console.set_filter(pattern="test")
        console.write("test message")
        console.write("other message")

        console.clear_filter()
        console.write("now everything")

        lines = console.get_scrollback()
        # Only messages written before clear_filter that matched
        # plus messages after clear_filter
        assert len(lines) == 2


class TestConsoleInput:
    """Tests for console input handling."""

    @pytest.fixture
    def console(self):
        return ConsoleUI()

    def test_set_input(self, console):
        console.set_input("test input")
        assert console.input_buffer == "test input"
        assert console.cursor_position == 10

    def test_set_input_length_limit(self):
        config = ConsoleConfig(max_input_length=10)
        console = ConsoleUI(config=config)

        console.set_input("this is a very long input")
        assert len(console.input_buffer) == 10

    def test_insert_char(self, console):
        console.set_input("hello")
        console.move_cursor_home()
        console.insert_char("X")

        assert console.input_buffer == "Xhello"
        assert console.cursor_position == 1

    def test_insert_at_cursor(self, console):
        console.set_input("hello")
        console.move_cursor_left()
        console.move_cursor_left()
        console.insert_char("X")

        assert console.input_buffer == "helXlo"

    def test_delete_char(self, console):
        console.set_input("hello")
        console.delete_char()  # Backspace at end

        assert console.input_buffer == "hell"

    def test_delete_char_at_start(self, console):
        console.set_input("hello")
        console.move_cursor_home()
        console.delete_char()  # Backspace at start - no effect

        assert console.input_buffer == "hello"

    def test_delete_forward(self, console):
        console.set_input("hello")
        console.move_cursor_home()
        console.delete_forward()  # Delete key

        assert console.input_buffer == "ello"

    def test_delete_word(self, console):
        console.set_input("hello world test")
        console.delete_word()

        assert console.input_buffer == "hello world "

    def test_clear_input(self, console):
        console.set_input("test input")
        console.clear_input()

        assert console.input_buffer == ""
        assert console.cursor_position == 0


class TestCursorMovement:
    """Tests for cursor movement."""

    @pytest.fixture
    def console(self):
        console = ConsoleUI()
        console.set_input("hello world")
        return console

    def test_move_cursor_left(self, console):
        console.move_cursor_left()
        assert console.cursor_position == 10

    def test_move_cursor_right(self, console):
        console.move_cursor_home()
        console.move_cursor_right()
        assert console.cursor_position == 1

    def test_move_cursor_home(self, console):
        console.move_cursor_home()
        assert console.cursor_position == 0

    def test_move_cursor_end(self, console):
        console.move_cursor_home()
        console.move_cursor_end()
        assert console.cursor_position == 11

    def test_move_cursor_word_left(self, console):
        console.move_cursor_word_left()
        assert console.cursor_position == 6  # Before "world"

    def test_move_cursor_word_right(self, console):
        console.move_cursor_home()
        console.move_cursor_word_right()
        assert console.cursor_position == 6  # After "hello "


class TestConsoleHistoryNavigation:
    """Tests for history navigation in console."""

    @pytest.fixture
    def console(self):
        history = CommandHistory()
        history.add("command1")
        history.add("command2")
        history.add("command3")
        return ConsoleUI(history=history)

    def test_history_previous(self, console):
        result = console.history_previous()
        assert result == "command3"
        assert console.input_buffer == "command3"

    def test_history_next(self, console):
        console.history_previous()  # command3
        console.history_previous()  # command2
        console.history_previous()  # command1

        result = console.history_next()
        assert result == "command2"

    def test_history_search(self, console):
        results = console.history_search("command")
        assert len(results) == 3

    def test_history_disabled(self):
        config = ConsoleConfig(history_enabled=False)
        console = ConsoleUI(config=config)

        assert console.history_previous() is None
        assert console.history_next() is None


class TestConsoleAutocomplete:
    """Tests for console autocomplete."""

    @pytest.fixture
    def console(self):
        console = ConsoleUI()
        console.set_autocomplete_handler(
            lambda text: [c for c in ["help", "hello", "history", "give"]
                         if c.startswith(text.lower())]
        )
        return console

    def test_get_completions(self, console):
        console.set_input("he")
        result = console.get_completions()

        assert "help" in result.suggestions
        assert "hello" in result.suggestions
        assert "history" not in result.suggestions

    def test_common_prefix(self, console):
        console.set_input("he")
        result = console.get_completions()

        assert result.common_prefix == "hel"

    def test_apply_completion(self, console):
        console.set_input("he")
        console.apply_completion("help")

        assert console.input_buffer == "help"

    def test_tab_complete_single_match(self, console):
        console.set_input("giv")
        completed = console.tab_complete()

        assert completed is True
        assert console.input_buffer == "give"

    def test_tab_complete_multiple_matches(self, console):
        console.set_input("he")
        completed = console.tab_complete()

        # Should extend to common prefix
        assert console.input_buffer == "hel"

    def test_tab_complete_shows_options(self, console):
        console.set_input("hel")
        console.tab_complete()

        # Should show completions in output
        lines = console.get_scrollback()
        assert any("Completions" in line.text for line in lines)

    def test_autocomplete_disabled(self):
        config = ConsoleConfig(autocomplete_enabled=False)
        console = ConsoleUI(config=config)

        result = console.get_completions()
        assert result.suggestions == []

    def test_autocomplete_min_chars(self):
        config = ConsoleConfig(autocomplete_min_chars=3)
        console = ConsoleUI(config=config)
        console.set_autocomplete_handler(lambda t: ["test"])

        console.set_input("te")
        result = console.get_completions()
        assert result.suggestions == []

        console.set_input("tes")
        result = console.get_completions()
        assert "test" in result.suggestions


class TestConsoleExecution:
    """Tests for command execution."""

    @pytest.fixture
    def console(self):
        console = ConsoleUI()
        return console

    def test_execute_empty(self, console):
        result = console.execute()
        assert result is False

    def test_execute_with_handler(self, console):
        executed = []

        def handler(cmd: str):
            executed.append(cmd)

        console.set_command_handler(handler)
        console.set_input("test command")
        result = console.execute()

        assert result is True
        assert executed == ["test command"]
        assert console.input_buffer == ""

    def test_execute_echoes_command(self, console):
        console.set_input("test")
        console.execute()

        lines = console.get_scrollback()
        assert any("> test" in line.text for line in lines)

    def test_execute_no_echo(self):
        config = ConsoleConfig(echo_commands=False)
        console = ConsoleUI(config=config)
        console.set_input("test")
        console.execute()

        lines = console.get_scrollback()
        assert not any(">" in line.text for line in lines)

    def test_execute_adds_to_history(self):
        history = CommandHistory()
        console = ConsoleUI(history=history)

        console.set_input("test command")
        console.execute()

        # Check history contains the command
        entries = list(history)
        assert any(e.command == "test command" for e in entries)

    def test_execute_command_directly(self, console):
        executed = []
        console.set_command_handler(lambda cmd: executed.append(cmd))

        result = console.execute_command("direct command")
        assert result is True
        assert "direct command" in executed

    def test_execute_handler_error(self, console):
        def bad_handler(cmd: str):
            raise ValueError("Handler error")

        console.set_command_handler(bad_handler)
        console.set_input("test")
        result = console.execute()

        assert result is False
        lines = console.get_scrollback()
        assert any("Error" in line.text for line in lines)
