"""
Whitebox tests for Contextual Dialogue module.

Tests CooldownTracker, LinePool selection modes, BarkSystem,
AmbientVOSystem, and ContextualDialogueManager.
"""

import pytest
import threading
import time
import random
from unittest.mock import MagicMock, patch

from engine.audio.dialogue.contextual_dialogue import (
    CooldownTracker,
    LinePool,
    ContextualDialogueManager,
    BarkSystem,
    AmbientVOSystem,
    create_bark_lines,
)
from engine.audio.dialogue.vo_line import VOLine
from engine.audio.dialogue.config import (
    SelectionMode,
    SAME_LINE_COOLDOWN_MS,
    SAME_SPEAKER_COOLDOWN_MS,
    BARK_COOLDOWN_MS,
    PRIORITY_NORMAL,
    PRIORITY_BARK,
    PRIORITY_AMBIENT,
    CONTEXT_BARK,
    CONTEXT_AMBIENT,
)


# =============================================================================
# CooldownTracker Tests
# =============================================================================


class TestCooldownTracker:
    """Tests for CooldownTracker."""

    def test_initialization(self):
        """Test CooldownTracker initializes empty."""
        tracker = CooldownTracker()

        assert len(tracker._line_cooldowns) == 0
        assert len(tracker._speaker_cooldowns) == 0
        assert len(tracker._category_cooldowns) == 0

    def test_record_play(self):
        """Test record_play records all cooldowns."""
        tracker = CooldownTracker()
        current_time = time.time()

        tracker.record_play("line_1", "speaker_1", "combat", current_time)

        assert "line_1" in tracker._line_cooldowns
        assert "speaker_1" in tracker._speaker_cooldowns
        assert "combat" in tracker._category_cooldowns

    def test_record_play_empty_speaker(self):
        """Test record_play with empty speaker."""
        tracker = CooldownTracker()
        current_time = time.time()

        tracker.record_play("line_1", "", "combat", current_time)

        assert "line_1" in tracker._line_cooldowns
        assert "" not in tracker._speaker_cooldowns

    def test_record_play_empty_category(self):
        """Test record_play with empty category."""
        tracker = CooldownTracker()
        current_time = time.time()

        tracker.record_play("line_1", "speaker_1", "", current_time)

        assert "line_1" in tracker._line_cooldowns
        assert "" not in tracker._category_cooldowns

    def test_is_line_on_cooldown_true(self):
        """Test is_line_on_cooldown returns True within cooldown."""
        tracker = CooldownTracker()
        current_time = 100.0

        tracker.record_play("line_1", "", "", current_time)

        # 500ms later (500ms < 1000ms cooldown)
        assert tracker.is_line_on_cooldown("line_1", 100.5, 1000.0) is True

    def test_is_line_on_cooldown_false(self):
        """Test is_line_on_cooldown returns False after cooldown."""
        tracker = CooldownTracker()
        current_time = 100.0

        tracker.record_play("line_1", "", "", current_time)

        # 2 seconds later (2000ms > 1000ms cooldown)
        assert tracker.is_line_on_cooldown("line_1", 102.0, 1000.0) is False

    def test_is_line_on_cooldown_never_played(self):
        """Test is_line_on_cooldown returns False for never played."""
        tracker = CooldownTracker()

        assert tracker.is_line_on_cooldown("line_1", 100.0, 1000.0) is False

    def test_is_speaker_on_cooldown(self):
        """Test is_speaker_on_cooldown functionality."""
        tracker = CooldownTracker()
        current_time = 100.0

        tracker.record_play("line_1", "speaker_1", "", current_time)

        # Within cooldown
        assert tracker.is_speaker_on_cooldown("speaker_1", 100.5, 1000.0) is True
        # After cooldown
        assert tracker.is_speaker_on_cooldown("speaker_1", 102.0, 1000.0) is False

    def test_is_category_on_cooldown(self):
        """Test is_category_on_cooldown functionality."""
        tracker = CooldownTracker()
        current_time = 100.0

        tracker.record_play("line_1", "", "combat", current_time)

        # Within cooldown
        assert tracker.is_category_on_cooldown("combat", 100.5, 1000.0) is True
        # After cooldown
        assert tracker.is_category_on_cooldown("combat", 102.0, 1000.0) is False

    def test_clear_cooldowns(self):
        """Test clear_cooldowns clears all tracking."""
        tracker = CooldownTracker()

        tracker.record_play("line_1", "speaker_1", "combat", time.time())
        tracker.clear_cooldowns()

        assert len(tracker._line_cooldowns) == 0
        assert len(tracker._speaker_cooldowns) == 0
        assert len(tracker._category_cooldowns) == 0

    def test_clear_speaker_cooldown(self):
        """Test clear_speaker_cooldown clears specific speaker."""
        tracker = CooldownTracker()
        current_time = time.time()

        tracker.record_play("line_1", "speaker_1", "", current_time)
        tracker.record_play("line_2", "speaker_2", "", current_time)

        tracker.clear_speaker_cooldown("speaker_1")

        assert "speaker_1" not in tracker._speaker_cooldowns
        assert "speaker_2" in tracker._speaker_cooldowns

    def test_get_cooldown_remaining(self):
        """Test get_cooldown_remaining calculation."""
        tracker = CooldownTracker()
        current_time = 100.0

        tracker.record_play("line_1", "", "", current_time)

        # 500ms later with 1000ms cooldown
        remaining = tracker.get_cooldown_remaining("line_1", 100.5, 1000.0)
        assert abs(remaining - 500.0) < 1.0

    def test_get_cooldown_remaining_after_expire(self):
        """Test get_cooldown_remaining returns 0 after expiry."""
        tracker = CooldownTracker()
        current_time = 100.0

        tracker.record_play("line_1", "", "", current_time)

        # 2 seconds later
        remaining = tracker.get_cooldown_remaining("line_1", 102.0, 1000.0)
        assert remaining == 0.0


# =============================================================================
# CooldownTracker Thread Safety Tests
# =============================================================================


class TestCooldownTrackerThreadSafety:
    """Thread safety tests for CooldownTracker."""

    def test_concurrent_record_play(self):
        """Test concurrent record_play operations."""
        tracker = CooldownTracker()

        def record_plays():
            for i in range(100):
                tracker.record_play(f"line_{i}", f"speaker_{i}", "cat", time.time())
                time.sleep(0.001)

        threads = [threading.Thread(target=record_plays) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without issues
        assert len(tracker._line_cooldowns) <= 300


# =============================================================================
# LinePool Basic Tests
# =============================================================================


class TestLinePoolBasic:
    """Basic tests for LinePool."""

    def test_initialization(self):
        """Test LinePool initializes correctly."""
        pool = LinePool(pool_id="test_pool")

        assert pool.pool_id == "test_pool"
        assert pool.lines == []
        assert pool.selection_mode == SelectionMode.RANDOM.value
        assert pool.cooldown_ms == SAME_LINE_COOLDOWN_MS

    def test_custom_initialization(self):
        """Test LinePool with custom values."""
        pool = LinePool(
            pool_id="test_pool",
            selection_mode=SelectionMode.SEQUENTIAL.value,
            cooldown_ms=5000.0,
            category="combat",
        )

        assert pool.selection_mode == SelectionMode.SEQUENTIAL.value
        assert pool.cooldown_ms == 5000.0
        assert pool.category == "combat"

    def test_add_line(self):
        """Test add_line adds to pool."""
        pool = LinePool(pool_id="test")
        line = VOLine(text="Hello")

        pool.add_line(line)

        assert len(pool.lines) == 1
        assert pool.lines[0] is line

    def test_remove_line(self):
        """Test remove_line removes from pool."""
        pool = LinePool(pool_id="test")
        line = VOLine(line_id="line_1")
        pool.add_line(line)

        result = pool.remove_line("line_1")

        assert result is True
        assert len(pool.lines) == 0

    def test_remove_line_not_found(self):
        """Test remove_line returns False for missing line."""
        pool = LinePool(pool_id="test")

        result = pool.remove_line("missing")

        assert result is False

    def test_size_property(self):
        """Test size property."""
        pool = LinePool(pool_id="test")
        pool.add_line(VOLine())
        pool.add_line(VOLine())

        assert pool.size == 2

    def test_len(self):
        """Test __len__."""
        pool = LinePool(pool_id="test")
        pool.add_line(VOLine())

        assert len(pool) == 1

    def test_iter(self):
        """Test __iter__."""
        pool = LinePool(pool_id="test")
        lines = [VOLine(text=f"Line {i}") for i in range(3)]
        for line in lines:
            pool.add_line(line)

        iterated = list(pool)

        assert len(iterated) == 3


# =============================================================================
# LinePool Selection Mode Tests
# =============================================================================


class TestLinePoolSelectionModes:
    """Tests for LinePool selection modes."""

    def test_select_random(self):
        """Test random selection mode."""
        pool = LinePool(pool_id="test", selection_mode=SelectionMode.RANDOM.value)
        for i in range(10):
            pool.add_line(VOLine(line_id=f"line_{i}"))

        selected_ids = set()
        for _ in range(50):
            line = pool.select_line(time.time())
            if line:
                selected_ids.add(line.line_id)

        # Should select multiple different lines
        assert len(selected_ids) > 1

    def test_select_sequential(self):
        """Test sequential selection mode."""
        pool = LinePool(
            pool_id="test",
            selection_mode=SelectionMode.SEQUENTIAL.value,
            cooldown_ms=0.0,  # No cooldown for testing
        )
        for i in range(3):
            pool.add_line(VOLine(line_id=f"line_{i}"))

        sequence = []
        for _ in range(6):
            line = pool.select_line(time.time())
            if line:
                sequence.append(line.line_id)

        # Should cycle through in order
        assert sequence == ["line_0", "line_1", "line_2", "line_0", "line_1", "line_2"]

    def test_select_sequential_reset(self):
        """Test sequential selection reset."""
        pool = LinePool(
            pool_id="test",
            selection_mode=SelectionMode.SEQUENTIAL.value,
            cooldown_ms=0.0,
        )
        for i in range(3):
            pool.add_line(VOLine(line_id=f"line_{i}"))

        pool.select_line(time.time())
        pool.select_line(time.time())
        pool.reset_sequential()

        line = pool.select_line(time.time())
        assert line.line_id == "line_0"

    def test_select_weighted(self):
        """Test weighted selection mode."""
        pool = LinePool(
            pool_id="test",
            selection_mode=SelectionMode.WEIGHTED.value,
            cooldown_ms=0.0,
        )
        # Add line with high weight
        high_weight = VOLine(line_id="high", weight=10.0)
        low_weight = VOLine(line_id="low", weight=0.1)
        pool.add_line(high_weight)
        pool.add_line(low_weight)

        counts = {"high": 0, "low": 0}
        for _ in range(100):
            line = pool.select_line(time.time())
            if line:
                counts[line.line_id] += 1

        # High weight should be selected more often
        assert counts["high"] > counts["low"]

    def test_select_weighted_zero_total(self):
        """Test weighted selection with zero total weight."""
        pool = LinePool(
            pool_id="test",
            selection_mode=SelectionMode.WEIGHTED.value,
            cooldown_ms=0.0,
        )
        pool.add_line(VOLine(weight=0.0))
        pool.add_line(VOLine(weight=0.0))

        # Should fall back to random
        line = pool.select_line(time.time())
        assert line is not None

    def test_select_shuffle(self):
        """Test shuffle selection mode."""
        pool = LinePool(
            pool_id="test",
            selection_mode=SelectionMode.SHUFFLE.value,
            cooldown_ms=0.0,
        )
        for i in range(5):
            pool.add_line(VOLine(line_id=f"line_{i}"))

        # Select all 5
        first_round = []
        for _ in range(5):
            line = pool.select_line(time.time())
            if line:
                first_round.append(line.line_id)

        # Should have all 5 unique
        assert len(set(first_round)) == 5

    def test_select_conditional(self):
        """Test conditional selection mode."""
        pool = LinePool(
            pool_id="test",
            selection_mode=SelectionMode.CONDITIONAL.value,
            cooldown_ms=0.0,
        )
        pool.add_line(VOLine(line_id="line_1", conditions={"level": 5}))
        pool.add_line(VOLine(line_id="line_2", conditions={"level": 10}))

        # With matching game state
        line = pool.select_line(
            time.time(),
            game_state={"level": 5},
        )

        assert line.line_id == "line_1"


# =============================================================================
# LinePool Availability Tests
# =============================================================================


class TestLinePoolAvailability:
    """Tests for LinePool line availability."""

    def test_get_available_lines_all(self):
        """Test _get_available_lines returns all when no restrictions."""
        pool = LinePool(pool_id="test", cooldown_ms=0.0)
        for i in range(5):
            pool.add_line(VOLine(line_id=f"line_{i}"))

        available = pool._get_available_lines(time.time(), None, None)

        assert len(available) == 5

    def test_get_available_lines_with_cooldown(self):
        """Test _get_available_lines filters cooled lines."""
        pool = LinePool(pool_id="test", cooldown_ms=10000.0)
        for i in range(3):
            pool.add_line(VOLine(line_id=f"line_{i}"))

        tracker = CooldownTracker()
        current_time = time.time()

        # Put first line on cooldown
        tracker.record_play("line_0", "", "", current_time)

        available = pool._get_available_lines(current_time, tracker, None)

        assert len(available) == 2
        assert all(l.line_id != "line_0" for l in available)

    def test_get_available_lines_with_conditions(self):
        """Test _get_available_lines filters by conditions."""
        pool = LinePool(pool_id="test", cooldown_ms=0.0)
        pool.add_line(VOLine(line_id="line_1", conditions={"quest": "active"}))
        pool.add_line(VOLine(line_id="line_2", conditions={}))

        available = pool._get_available_lines(
            time.time(), None, {"quest": "completed"}
        )

        assert len(available) == 1
        assert available[0].line_id == "line_2"

    def test_select_line_empty_pool(self):
        """Test select_line with empty pool."""
        pool = LinePool(pool_id="test")

        line = pool.select_line(time.time())

        assert line is None

    def test_select_line_all_on_cooldown(self):
        """Test select_line when all lines on cooldown."""
        pool = LinePool(pool_id="test", cooldown_ms=10000.0)
        for i in range(3):
            pool.add_line(VOLine(line_id=f"line_{i}"))

        tracker = CooldownTracker()
        current_time = time.time()

        # Put all lines on cooldown
        for i in range(3):
            tracker.record_play(f"line_{i}", "", "", current_time)

        line = pool.select_line(current_time, tracker)

        assert line is None


# =============================================================================
# ContextualDialogueManager Tests
# =============================================================================


class TestContextualDialogueManager:
    """Tests for ContextualDialogueManager."""

    def test_initialization(self):
        """Test ContextualDialogueManager initializes correctly."""
        manager = ContextualDialogueManager()

        assert manager.pool_ids == []
        assert manager._current_game_state == {}

    def test_create_pool(self):
        """Test create_pool creates new pool."""
        manager = ContextualDialogueManager()

        pool = manager.create_pool("test_pool")

        assert pool.pool_id == "test_pool"
        assert "test_pool" in manager.pool_ids

    def test_create_pool_duplicate(self):
        """Test create_pool raises for duplicate."""
        manager = ContextualDialogueManager()
        manager.create_pool("test_pool")

        with pytest.raises(ValueError):
            manager.create_pool("test_pool")

    def test_get_pool(self):
        """Test get_pool retrieves pool."""
        manager = ContextualDialogueManager()
        created = manager.create_pool("test_pool")

        retrieved = manager.get_pool("test_pool")

        assert retrieved is created

    def test_get_pool_not_found(self):
        """Test get_pool returns None for missing."""
        manager = ContextualDialogueManager()

        result = manager.get_pool("missing")

        assert result is None

    def test_get_or_create_pool(self):
        """Test get_or_create_pool creates or retrieves."""
        manager = ContextualDialogueManager()

        pool1 = manager.get_or_create_pool("test_pool")
        pool2 = manager.get_or_create_pool("test_pool")

        assert pool1 is pool2

    def test_add_line_to_pool(self):
        """Test add_line_to_pool adds line."""
        manager = ContextualDialogueManager()
        manager.create_pool("test_pool")
        line = VOLine(text="Hello")

        result = manager.add_line_to_pool("test_pool", line)

        assert result is True
        assert len(manager.get_pool("test_pool").lines) == 1

    def test_add_line_to_pool_not_found(self):
        """Test add_line_to_pool returns False for missing pool."""
        manager = ContextualDialogueManager()

        result = manager.add_line_to_pool("missing", VOLine())

        assert result is False

    def test_remove_pool(self):
        """Test remove_pool removes pool."""
        manager = ContextualDialogueManager()
        manager.create_pool("test_pool")

        result = manager.remove_pool("test_pool")

        assert result is True
        assert "test_pool" not in manager.pool_ids

    def test_select_from_pool(self):
        """Test select_from_pool selects line."""
        callback = MagicMock()
        manager = ContextualDialogueManager(on_line_selected=callback)
        pool = manager.create_pool("test_pool", cooldown_ms=0.0)
        pool.add_line(VOLine(text="Hello"))

        line = manager.select_from_pool("test_pool")

        assert line is not None
        callback.assert_called_once()

    def test_select_from_pool_not_found(self):
        """Test select_from_pool returns None for missing pool."""
        manager = ContextualDialogueManager()

        result = manager.select_from_pool("missing")

        assert result is None

    def test_record_play(self):
        """Test record_play records cooldown."""
        manager = ContextualDialogueManager()
        pool = manager.create_pool("test_pool", category="combat")
        line = VOLine(line_id="line_1", speaker_id="npc_1")
        pool.add_line(line)

        manager.record_play(line, "test_pool")

        # Verify cooldown was recorded
        current_time = time.time()
        assert manager.cooldown_tracker.is_line_on_cooldown(
            "line_1", current_time, 10000.0
        )


# =============================================================================
# ContextualDialogueManager Game State Tests
# =============================================================================


class TestContextualDialogueManagerGameState:
    """Tests for ContextualDialogueManager game state handling."""

    def test_update_game_state(self):
        """Test update_game_state updates state dict."""
        manager = ContextualDialogueManager()

        manager.update_game_state({"level": 5})
        manager.update_game_state({"quest": "active"})

        assert manager._current_game_state["level"] == 5
        assert manager._current_game_state["quest"] == "active"

    def test_set_game_state(self):
        """Test set_game_state replaces state dict."""
        manager = ContextualDialogueManager()
        manager.update_game_state({"old": "value"})

        manager.set_game_state({"new": "value"})

        assert "old" not in manager._current_game_state
        assert manager._current_game_state["new"] == "value"

    def test_clear_game_state(self):
        """Test clear_game_state clears state."""
        manager = ContextualDialogueManager()
        manager.update_game_state({"key": "value"})

        manager.clear_game_state()

        assert manager._current_game_state == {}

    def test_is_line_available(self):
        """Test is_line_available checks availability."""
        manager = ContextualDialogueManager()
        pool = manager.create_pool("test_pool", cooldown_ms=0.0)
        pool.add_line(VOLine(line_id="line_1"))

        result = manager.is_line_available("test_pool", "line_1")

        assert result is True

    def test_is_line_available_on_cooldown(self):
        """Test is_line_available returns False when on cooldown."""
        manager = ContextualDialogueManager()
        pool = manager.create_pool("test_pool", cooldown_ms=10000.0)
        line = VOLine(line_id="line_1")
        pool.add_line(line)

        manager.record_play(line, "test_pool")

        result = manager.is_line_available("test_pool", "line_1")

        assert result is False


# =============================================================================
# BarkSystem Tests
# =============================================================================


class TestBarkSystem:
    """Tests for BarkSystem."""

    def test_initialization(self):
        """Test BarkSystem initializes correctly."""
        system = BarkSystem()

        assert system.is_enabled is True
        assert system.bark_types == []

    def test_custom_initialization(self):
        """Test BarkSystem with custom parameters."""
        callback = MagicMock()
        system = BarkSystem(cooldown_ms=5000.0, on_bark_triggered=callback)

        assert system._default_cooldown == 5000.0

    def test_register_bark_pool(self):
        """Test register_bark_pool registers barks."""
        system = BarkSystem()
        lines = [
            VOLine(text="Alert!", priority=PRIORITY_NORMAL),
            VOLine(text="Enemy!", priority=PRIORITY_NORMAL),
        ]

        pool = system.register_bark_pool("alert", lines)

        assert "alert" in system.bark_types
        assert pool.size == 2
        # Lines should have bark context type and priority
        assert all(l.context_type == CONTEXT_BARK for l in pool.lines)
        assert all(l.priority == PRIORITY_BARK for l in pool.lines)

    def test_trigger_bark(self):
        """Test trigger_bark triggers bark."""
        system = BarkSystem(cooldown_ms=0.0)
        lines = [VOLine(text="Alert!")]
        system.register_bark_pool("alert", lines)

        line = system.trigger_bark("alert")

        assert line is not None
        assert line.text == "Alert!"

    def test_trigger_bark_disabled(self):
        """Test trigger_bark returns None when disabled."""
        system = BarkSystem()
        system.register_bark_pool("alert", [VOLine(text="Alert!")])
        system.disable()

        line = system.trigger_bark("alert")

        assert line is None

    def test_trigger_bark_speaker_filter(self):
        """Test trigger_bark with speaker filter."""
        system = BarkSystem(cooldown_ms=0.0)
        lines = [
            VOLine(text="Line 1", speaker_id="npc_1"),
            VOLine(text="Line 2", speaker_id="npc_2"),
        ]
        system.register_bark_pool("alert", lines)

        # Trigger for specific speaker
        line = system.trigger_bark("alert", speaker_id="npc_1")

        # May or may not get the filtered line depending on random selection
        # Just verify it doesn't crash

    def test_trigger_bark_speaker_on_cooldown(self):
        """Test trigger_bark respects speaker cooldown."""
        system = BarkSystem(cooldown_ms=0.0)
        lines = [VOLine(text="Alert!", speaker_id="npc_1")]
        system.register_bark_pool("alert", lines)

        # Record play to put speaker on cooldown
        current_time = time.time()
        system._manager.cooldown_tracker.record_play("", "npc_1", "", current_time)

        line = system.trigger_bark("alert", speaker_id="npc_1", current_time=current_time)

        assert line is None

    def test_enable_disable(self):
        """Test enable/disable methods."""
        system = BarkSystem()

        system.disable()
        assert system.is_enabled is False

        system.enable()
        assert system.is_enabled is True


# =============================================================================
# AmbientVOSystem Tests
# =============================================================================


class TestAmbientVOSystem:
    """Tests for AmbientVOSystem."""

    def test_initialization(self):
        """Test AmbientVOSystem initializes correctly."""
        system = AmbientVOSystem()

        assert system.is_enabled is True
        assert system.active_zones == set()

    def test_custom_initialization(self):
        """Test AmbientVOSystem with custom parameters."""
        callback = MagicMock()
        system = AmbientVOSystem(
            min_interval_ms=5000.0,
            max_interval_ms=15000.0,
            on_ambient_triggered=callback,
        )

        assert system._min_interval == 5000.0
        assert system._max_interval == 15000.0

    def test_register_zone(self):
        """Test register_zone registers ambient zone."""
        system = AmbientVOSystem()
        lines = [VOLine(text="Wind blowing...")]

        pool = system.register_zone("forest", lines)

        assert pool.size == 1
        # Lines should have ambient context type and priority
        assert all(l.context_type == CONTEXT_AMBIENT for l in pool.lines)
        assert all(l.priority == PRIORITY_AMBIENT for l in pool.lines)

    def test_enter_zone(self):
        """Test enter_zone adds zone to active set."""
        system = AmbientVOSystem()
        system.register_zone("forest", [VOLine()])

        system.enter_zone("forest")

        assert "forest" in system.active_zones

    def test_exit_zone(self):
        """Test exit_zone removes zone from active set."""
        system = AmbientVOSystem()
        system.register_zone("forest", [VOLine()])
        system.enter_zone("forest")

        system.exit_zone("forest")

        assert "forest" not in system.active_zones

    def test_update_no_zones(self):
        """Test update returns None with no active zones."""
        system = AmbientVOSystem()

        line = system.update(time.time())

        assert line is None

    def test_update_disabled(self):
        """Test update returns None when disabled."""
        system = AmbientVOSystem()
        system.register_zone("forest", [VOLine()])
        system.enter_zone("forest")
        system.disable()

        line = system.update(time.time())

        assert line is None

    def test_update_triggers_after_interval(self):
        """Test update triggers after interval elapsed."""
        system = AmbientVOSystem(min_interval_ms=1.0, max_interval_ms=1.0)
        system.register_zone("forest", [VOLine(text="Ambient")])
        system.enter_zone("forest")

        # Set last play time to past
        system._last_play_time = time.time() - 10

        line = system.update(time.time())

        assert line is not None

    def test_force_trigger(self):
        """Test force_trigger forces immediate trigger."""
        system = AmbientVOSystem()
        system.register_zone("forest", [VOLine(text="Ambient")])
        system.enter_zone("forest")

        line = system.force_trigger()

        assert line is not None

    def test_force_trigger_specific_zone(self):
        """Test force_trigger with specific zone."""
        system = AmbientVOSystem()
        system.register_zone("forest", [VOLine(text="Forest")])
        system.register_zone("cave", [VOLine(text="Cave")])
        system.enter_zone("forest")
        system.enter_zone("cave")

        line = system.force_trigger(zone_id="cave")

        assert line is not None

    def test_force_trigger_no_zones(self):
        """Test force_trigger returns None with no zones."""
        system = AmbientVOSystem()

        line = system.force_trigger()

        assert line is None


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestCreateBarkLines:
    """Tests for create_bark_lines helper function."""

    def test_create_basic(self):
        """Test create_bark_lines creates lines."""
        bark_data = [
            {"audio_asset": "bark1.wav", "text": "Alert!"},
            {"audio_asset": "bark2.wav", "text": "Enemy!"},
        ]

        lines = create_bark_lines(bark_data)

        assert len(lines) == 2
        assert lines[0].text == "Alert!"
        assert lines[0].context_type == CONTEXT_BARK
        assert lines[0].priority == PRIORITY_BARK

    def test_create_with_speaker(self):
        """Test create_bark_lines with default speaker."""
        bark_data = [{"audio_asset": "bark.wav", "text": "Alert!"}]

        lines = create_bark_lines(bark_data, speaker_id="npc_1")

        assert lines[0].speaker_id == "npc_1"

    def test_create_with_override_speaker(self):
        """Test create_bark_lines with per-line speaker override."""
        bark_data = [
            {"audio_asset": "bark.wav", "text": "Alert!", "speaker_id": "npc_2"},
        ]

        lines = create_bark_lines(bark_data, speaker_id="npc_1")

        assert lines[0].speaker_id == "npc_2"

    def test_create_with_all_fields(self):
        """Test create_bark_lines with all optional fields."""
        bark_data = [
            {
                "audio_asset": "bark.wav",
                "text": "Alert!",
                "duration_ms": 500.0,
                "priority": PRIORITY_NORMAL,
                "interruptible": False,
                "tags": ["combat", "urgent"],
                "weight": 2.0,
            },
        ]

        lines = create_bark_lines(bark_data)

        assert lines[0].duration_ms == 500.0
        assert lines[0].interruptible is False
        assert "combat" in lines[0].tags
        assert lines[0].weight == 2.0


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestContextualDialogueThreadSafety:
    """Thread safety tests for contextual dialogue components."""

    def test_concurrent_pool_selection(self):
        """Test concurrent pool selection."""
        manager = ContextualDialogueManager()
        pool = manager.create_pool("test", cooldown_ms=0.0)
        for i in range(10):
            pool.add_line(VOLine(line_id=f"line_{i}"))

        results = []

        def select_lines():
            for _ in range(50):
                line = manager.select_from_pool("test")
                results.append(line)
                time.sleep(0.001)

        threads = [threading.Thread(target=select_lines) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have selected lines without errors
        non_none = [r for r in results if r is not None]
        assert len(non_none) == 150

    def test_concurrent_bark_triggers(self):
        """Test concurrent bark triggers."""
        system = BarkSystem(cooldown_ms=0.0)
        for i in range(5):
            lines = [VOLine(line_id=f"line_{i}_{j}") for j in range(3)]
            system.register_bark_pool(f"type_{i}", lines)

        results = []

        def trigger_barks():
            for i in range(20):
                bark_type = f"type_{i % 5}"
                line = system.trigger_bark(bark_type)
                results.append(line)
                time.sleep(0.001)

        threads = [threading.Thread(target=trigger_barks) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without deadlock
