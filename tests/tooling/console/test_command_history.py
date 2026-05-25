"""Tests for command history management.

Tests history storage, search, navigation, and persistence.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from engine.tooling.console.command_history import CommandHistory, HistoryEntry


class TestHistoryEntry:
    """Tests for HistoryEntry."""

    def test_basic_creation(self):
        entry = HistoryEntry(
            command="test command",
            timestamp=datetime.now(),
            success=True,
            result_summary="OK"
        )
        assert entry.command == "test command"
        assert entry.success is True
        assert entry.result_summary == "OK"

    def test_to_dict(self):
        timestamp = datetime.now()
        entry = HistoryEntry(
            command="test",
            timestamp=timestamp,
            success=True,
            session_id="session1"
        )
        data = entry.to_dict()

        assert data["command"] == "test"
        assert data["timestamp"] == timestamp.isoformat()
        assert data["success"] is True
        assert data["session_id"] == "session1"

    def test_from_dict(self):
        timestamp = datetime.now()
        data = {
            "command": "test",
            "timestamp": timestamp.isoformat(),
            "success": False,
            "result_summary": "Error",
            "session_id": "session1"
        }
        entry = HistoryEntry.from_dict(data)

        assert entry.command == "test"
        assert entry.success is False
        assert entry.result_summary == "Error"


class TestCommandHistory:
    """Tests for CommandHistory."""

    def test_basic_add(self):
        history = CommandHistory()
        entry = history.add("test command")

        assert entry.command == "test command"
        assert history.count == 1

    def test_add_empty_raises(self):
        history = CommandHistory()
        with pytest.raises(ValueError, match="cannot be empty"):
            history.add("")

    def test_add_strips_whitespace(self):
        history = CommandHistory()
        entry = history.add("  test  ")
        assert entry.command == "test"

    def test_capacity_limit(self):
        history = CommandHistory(capacity=5)
        for i in range(10):
            history.add(f"command_{i}")

        assert history.count == 5
        # Oldest commands should be removed
        assert history.get(0).command == "command_5"

    def test_get_by_index(self):
        history = CommandHistory()
        history.add("first")
        history.add("second")
        history.add("third")

        assert history.get(0).command == "first"
        assert history.get(-1).command == "third"
        assert history.get(100) is None

    def test_get_recent(self):
        history = CommandHistory()
        for i in range(10):
            history.add(f"command_{i}")

        recent = history.get_recent(3)
        assert len(recent) == 3
        assert recent[0].command == "command_7"
        assert recent[2].command == "command_9"

    def test_ignore_duplicates(self):
        history = CommandHistory(ignore_duplicates=True)
        history.add("test")
        history.add("test")  # Duplicate
        history.add("other")
        history.add("other")  # Duplicate

        assert history.count == 2

    def test_allow_duplicates(self):
        history = CommandHistory(ignore_duplicates=False)
        history.add("test")
        history.add("test")

        assert history.count == 2


class TestHistoryNavigation:
    """Tests for history navigation."""

    def test_previous(self):
        history = CommandHistory()
        history.add("first")
        history.add("second")
        history.add("third")

        assert history.previous() == "third"
        assert history.previous() == "second"
        assert history.previous() == "first"
        assert history.previous() == "first"  # At start

    def test_next(self):
        history = CommandHistory()
        history.add("first")
        history.add("second")
        history.add("third")

        history.previous()  # third
        history.previous()  # second
        history.previous()  # first

        assert history.next() == "second"
        assert history.next() == "third"
        assert history.next() is None  # At end

    def test_navigation_reset_on_add(self):
        history = CommandHistory()
        history.add("first")
        history.add("second")

        history.previous()  # second
        history.previous()  # first

        history.add("third")

        assert history.previous() == "third"

    def test_empty_history_navigation(self):
        history = CommandHistory()
        assert history.previous() is None
        assert history.next() is None


class TestHistorySearch:
    """Tests for history search."""

    def test_prefix_search(self):
        history = CommandHistory()
        history.add("give health")
        history.add("give ammo")
        history.add("teleport home")
        history.add("give shield")

        results = history.search("give", match_type="prefix")
        assert len(results) == 3
        # Most recent first
        assert results[0].command == "give shield"

    def test_substring_search(self):
        history = CommandHistory()
        history.add("player health 100")
        history.add("set health_max 200")
        history.add("show stats")

        results = history.search("health", match_type="substring")
        assert len(results) == 2

    def test_regex_search(self):
        history = CommandHistory()
        history.add("give weapon_ak47")
        history.add("give weapon_m4a1")
        history.add("give ammo")

        results = history.search(r"give weapon_\w+", match_type="regex")
        assert len(results) == 2

    def test_regex_invalid_pattern(self):
        history = CommandHistory()
        history.add("test")

        results = history.search("[invalid", match_type="regex")
        assert results == []

    def test_search_max_results(self):
        history = CommandHistory()
        for i in range(100):
            history.add(f"test_{i}")

        results = history.search("test", max_results=10)
        assert len(results) == 10

    def test_empty_query_returns_recent(self):
        history = CommandHistory()
        history.add("first")
        history.add("second")
        history.add("third")

        results = history.search("", max_results=2)
        assert len(results) == 2

    def test_reverse_search(self):
        history = CommandHistory()
        history.add("alpha")
        history.add("beta")
        history.add("alpha_2")
        history.add("gamma")

        results = list(history.reverse_search("alpha"))
        assert len(results) == 2
        assert results[0].command == "alpha_2"
        assert results[1].command == "alpha"


class TestHistoryUnique:
    """Tests for getting unique commands."""

    def test_get_unique_commands(self):
        history = CommandHistory(ignore_duplicates=False)
        history.add("first")
        history.add("second")
        history.add("first")  # Duplicate
        history.add("third")
        history.add("second")  # Duplicate

        unique = history.get_unique_commands()
        assert len(unique) == 3
        # Most recent first
        assert unique[0] == "second"
        assert unique[1] == "third"
        assert unique[2] == "first"

    def test_get_unique_commands_limit(self):
        history = CommandHistory()
        for i in range(100):
            history.add(f"unique_{i}")

        unique = history.get_unique_commands(max_count=10)
        assert len(unique) == 10


class TestHistorySession:
    """Tests for session-based history."""

    def test_session_id(self):
        history = CommandHistory(session_id="session1")
        entry = history.add("test")
        assert entry.session_id == "session1"

    def test_get_by_session(self):
        history = CommandHistory()
        history.session_id = "session1"
        history.add("cmd1")
        history.add("cmd2")

        history.session_id = "session2"
        history.add("cmd3")

        session1_entries = history.get_by_session("session1")
        assert len(session1_entries) == 2

        session2_entries = history.get_by_session("session2")
        assert len(session2_entries) == 1

    def test_clear_session(self):
        history = CommandHistory()
        history.session_id = "session1"
        history.add("cmd1")
        history.add("cmd2")

        history.session_id = "session2"
        history.add("cmd3")

        removed = history.clear_session("session1")
        assert removed == 2
        assert history.count == 1


class TestHistoryFailed:
    """Tests for failed command tracking."""

    def test_get_failed_commands(self):
        history = CommandHistory()
        history.add("success1", success=True)
        history.add("failed1", success=False)
        history.add("success2", success=True)
        history.add("failed2", success=False)

        failed = history.get_failed_commands()
        assert len(failed) == 2
        assert failed[0].command == "failed2"
        assert failed[1].command == "failed1"


class TestHistoryPersistence:
    """Tests for history persistence."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.json"

            history1 = CommandHistory(file_path=path)
            history1.add("command1")
            history1.add("command2")
            history1.save()

            history2 = CommandHistory(file_path=path)
            count = history2.load()

            assert count == 2
            assert history2.get(0).command == "command1"
            assert history2.get(1).command == "command2"

    def test_load_nonexistent(self):
        history = CommandHistory(file_path=Path("/nonexistent/path/history.json"))
        count = history.load()
        assert count == 0

    def test_load_corrupted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.json"
            path.write_text("not valid json")

            history = CommandHistory(file_path=path)
            count = history.load()
            assert count == 0

    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "history.json"

            history = CommandHistory(file_path=path)
            history.add("test")
            history.save()

            assert path.exists()


class TestHistoryMerge:
    """Tests for merging histories."""

    def test_merge(self):
        history1 = CommandHistory()
        history1.add("cmd1")
        history1.add("cmd2")

        history2 = CommandHistory()
        history2.add("cmd3")
        history2.add("cmd4")

        added = history1.merge(history2)
        assert added == 2
        assert history1.count == 4

    def test_merge_removes_duplicates(self):
        now = datetime.now()

        history1 = CommandHistory()
        history1.add("cmd1")

        history2 = CommandHistory()
        # Add same command at same time
        entry = HistoryEntry(command="cmd1", timestamp=history1.get(0).timestamp)
        history2._entries.append(entry)
        history2.add("cmd2")

        history1.merge(history2)
        # cmd1 should only appear once
        assert history1.count == 2

    def test_merge_respects_capacity(self):
        history1 = CommandHistory(capacity=5)
        for i in range(3):
            history1.add(f"h1_cmd{i}")

        history2 = CommandHistory()
        for i in range(3):
            history2.add(f"h2_cmd{i}")

        history1.merge(history2)
        assert history1.count == 5


class TestHistoryClear:
    """Tests for clearing history."""

    def test_clear(self):
        history = CommandHistory()
        history.add("cmd1")
        history.add("cmd2")
        history.clear()

        assert history.count == 0

    def test_clear_resets_navigation(self):
        history = CommandHistory()
        history.add("cmd1")
        history.add("cmd2")
        history.previous()

        history.clear()

        assert history.previous() is None


class TestHistoryContains:
    """Tests for checking if command in history."""

    def test_contains(self):
        history = CommandHistory()
        history.add("test command")

        assert "test command" in history
        assert "other" not in history


class TestHistoryIteration:
    """Tests for iterating over history."""

    def test_iter(self):
        history = CommandHistory()
        history.add("first")
        history.add("second")
        history.add("third")

        commands = [e.command for e in history]
        assert commands == ["first", "second", "third"]

    def test_len(self):
        history = CommandHistory()
        history.add("cmd1")
        history.add("cmd2")

        assert len(history) == 2
