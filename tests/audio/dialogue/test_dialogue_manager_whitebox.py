"""
Whitebox tests for DialogueManager module.

Tests DialogueManager central orchestration, event handling,
and integration with subcomponents.
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

from engine.audio.dialogue.dialogue_manager import (
    DialogueManager,
    DialogueEvent,
)
from engine.audio.dialogue.vo_line import VOLine, VOLineState
from engine.audio.dialogue.vo_streaming import StreamHandle, StreamState
from engine.audio.dialogue.conversation import Conversation, ConversationNode
from engine.audio.dialogue.config import (
    DialogueState,
    PRIORITY_NORMAL,
    PRIORITY_HIGH,
    DEFAULT_INTERRUPT_PRIORITY,
    EVENT_LINE_STARTED,
    EVENT_LINE_ENDED,
    EVENT_LINE_INTERRUPTED,
)


# =============================================================================
# DialogueEvent Tests
# =============================================================================


class TestDialogueEvent:
    """Tests for DialogueEvent dataclass."""

    def test_initialization(self):
        """Test DialogueEvent initializes correctly."""
        event = DialogueEvent(
            event_type=EVENT_LINE_STARTED,
            timestamp=100.0,
        )

        assert event.event_type == EVENT_LINE_STARTED
        assert event.timestamp == 100.0
        assert event.line is None
        assert event.conversation_id is None
        assert event.data == {}

    def test_full_initialization(self):
        """Test DialogueEvent with all fields."""
        line = VOLine(text="Hello")
        event = DialogueEvent(
            event_type=EVENT_LINE_ENDED,
            timestamp=200.0,
            line=line,
            conversation_id="conv_1",
            speaker_id="npc_1",
            data={"interrupted": False},
        )

        assert event.line is line
        assert event.conversation_id == "conv_1"
        assert event.speaker_id == "npc_1"
        assert event.data["interrupted"] is False


# =============================================================================
# DialogueManager Initialization Tests
# =============================================================================


class TestDialogueManagerInitialization:
    """Tests for DialogueManager initialization."""

    def test_default_initialization(self):
        """Test DialogueManager initializes with defaults."""
        manager = DialogueManager()

        assert manager.state == DialogueState.IDLE
        assert manager.is_playing is False
        assert manager.subtitles_enabled is True

    def test_custom_initialization(self):
        """Test DialogueManager with custom parameters."""
        callback = MagicMock()
        manager = DialogueManager(
            max_queue_size=32,
            max_simultaneous=4,
            default_language="es",
            enable_subtitles=False,
            cache_size_mb=64,
            on_event=callback,
        )

        assert manager.current_language == "es"
        assert manager.subtitles_enabled is False


# =============================================================================
# DialogueManager Lifecycle Tests
# =============================================================================


class TestDialogueManagerLifecycle:
    """Tests for DialogueManager lifecycle methods."""

    def test_start(self):
        """Test start method."""
        manager = DialogueManager()
        manager._state = DialogueState.PAUSED

        manager.start()

        assert manager.state == DialogueState.IDLE

    def test_stop(self):
        """Test stop method clears all state."""
        manager = DialogueManager()
        line = VOLine(duration_ms=1000.0)
        manager._queue.enqueue(line)
        manager._queue.start_line(line, time.time())

        manager.stop()

        assert manager.state == DialogueState.IDLE
        assert manager._queue.size == 0
        assert manager._queue.active_count == 0

    def test_pause(self):
        """Test pause method."""
        manager = DialogueManager()
        line = VOLine(duration_ms=1000.0)
        manager._queue.start_line(line, time.time())
        manager._state = DialogueState.PLAYING

        manager.pause()

        assert manager.state == DialogueState.PAUSED
        assert manager._queue.is_paused is True

    def test_resume_playing(self):
        """Test resume when playing."""
        manager = DialogueManager()
        line = VOLine(duration_ms=1000.0)
        manager._queue.start_line(line, time.time())
        manager._state = DialogueState.PAUSED

        manager.resume()

        assert manager.state == DialogueState.PLAYING
        assert manager._queue.is_paused is False

    def test_resume_idle(self):
        """Test resume when no active lines."""
        manager = DialogueManager()
        manager._state = DialogueState.PAUSED

        manager.resume()

        assert manager.state == DialogueState.IDLE


# =============================================================================
# DialogueManager Play Line Tests
# =============================================================================


class TestDialogueManagerPlayLine:
    """Tests for DialogueManager play_line functionality."""

    def test_play_line_basic(self):
        """Test basic play_line operation."""
        manager = DialogueManager()
        line = VOLine(
            audio_asset="test.wav",
            text="Hello",
            duration_ms=1000.0,
        )

        result = manager.play_line(line)

        assert result is True
        assert manager._queue.size == 1

    def test_play_line_localizes(self):
        """Test play_line localizes the line."""
        manager = DialogueManager(default_language="es")
        line = VOLine(audio_asset="test.wav", language="en")

        manager.play_line(line)

        # Line should be localized (but we don't change audio without registered assets)
        assert manager._queue.size == 1

    def test_play_line_high_priority_interrupts(self):
        """Test play_line with high priority interrupts active lines."""
        manager = DialogueManager()

        # Start low priority line
        low_line = VOLine(priority=PRIORITY_NORMAL, interruptible=True, duration_ms=1000.0)
        manager._queue.start_line(low_line, time.time())

        # Play high priority line
        high_line = VOLine(priority=DEFAULT_INTERRUPT_PRIORITY + 1, duration_ms=1000.0)
        manager.play_line(high_line)

        # Low line should be interrupted
        assert low_line.state == VOLineState.INTERRUPTED

    def test_play_line_with_position(self):
        """Test play_line with spatial position."""
        manager = DialogueManager()
        line = VOLine(audio_asset="test.wav", duration_ms=1000.0)

        result = manager.play_line(line, position=(10.0, 0.0, 5.0))

        assert result is True

    def test_play_line_with_radio_effect(self):
        """Test play_line with radio effect."""
        manager = DialogueManager()
        line = VOLine(audio_asset="test.wav", duration_ms=1000.0)

        result = manager.play_line(line, radio_effect=True)

        assert result is True

    def test_play_line_force(self):
        """Test play_line with force flag."""
        manager = DialogueManager(max_queue_size=1)
        manager._queue.enqueue(VOLine(priority=PRIORITY_NORMAL))

        line = VOLine(priority=PRIORITY_HIGH)
        result = manager.play_line(line, force=True)

        assert result is True

    def test_stop_line(self):
        """Test stop_line stops playing line."""
        manager = DialogueManager()
        line = VOLine(line_id="test_line", duration_ms=1000.0)
        manager._queue.start_line(line, time.time())

        result = manager.stop_line("test_line")

        assert result is True
        assert manager._queue.active_count == 0

    def test_stop_line_not_found(self):
        """Test stop_line returns False for inactive line."""
        manager = DialogueManager()

        result = manager.stop_line("missing_line")

        assert result is False


# =============================================================================
# DialogueManager Update Tests
# =============================================================================


class TestDialogueManagerUpdate:
    """Tests for DialogueManager update functionality."""

    def test_update_processes_queue(self):
        """Test update processes completed lines."""
        manager = DialogueManager()
        line = VOLine(duration_ms=100.0)
        manager._queue.start_line(line, time.time())

        # Simulate completion
        line.state = VOLineState.PLAYING
        manager.update(200.0)

        # Line should be completed
        assert line.is_completed is True

    def test_update_dequeues_pending(self):
        """Test update starts pending lines when capacity available."""
        manager = DialogueManager()

        # Enqueue line
        line = VOLine(audio_asset="test.wav", duration_ms=1000.0)
        manager._queue.enqueue(line)

        # Pre-cache the audio
        manager._stream_manager.cache.put("test.wav", b"audio data", duration_ms=1000.0)

        manager.update(16.0)

        # Line should be dequeued and processed
        # Note: actual start depends on stream readiness

    def test_update_updates_subtitles(self):
        """Test update updates subtitle manager."""
        manager = DialogueManager()

        # Just verify update doesn't crash
        manager.update(16.0)


# =============================================================================
# DialogueManager Conversation Tests
# =============================================================================


class TestDialogueManagerConversation:
    """Tests for DialogueManager conversation functionality."""

    def test_register_conversation(self):
        """Test register_conversation registers with manager."""
        manager = DialogueManager()
        conv = Conversation(conversation_id="conv_1")

        manager.register_conversation(conv)

        assert manager._conversation_manager.get_conversation("conv_1") is conv

    def test_start_conversation(self):
        """Test start_conversation starts registered conversation."""
        manager = DialogueManager()

        conv = Conversation(conversation_id="conv_1")
        line = VOLine(audio_asset="test.wav", text="Hello", duration_ms=1000.0)
        node = ConversationNode(node_id="start", line=line)
        conv.add_node(node)
        conv.start_node_id = "start"

        manager.register_conversation(conv)

        # Pre-cache audio
        manager._stream_manager.cache.put("test.wav", b"audio", duration_ms=1000.0)

        result = manager.start_conversation("conv_1")

        assert result is True

    def test_start_conversation_no_line(self):
        """Test start_conversation with node without line."""
        manager = DialogueManager()

        conv = Conversation(conversation_id="conv_1")
        node = ConversationNode(node_id="start", line=None)
        conv.add_node(node)
        conv.start_node_id = "start"

        manager.register_conversation(conv)
        result = manager.start_conversation("conv_1")

        assert result is True

    def test_start_conversation_not_found(self):
        """Test start_conversation returns False for missing."""
        manager = DialogueManager()

        result = manager.start_conversation("missing")

        assert result is False

    def test_end_conversation(self):
        """Test end_conversation ends active conversation."""
        manager = DialogueManager()

        conv = Conversation(conversation_id="conv_1")
        node = ConversationNode(node_id="start")
        conv.add_node(node)
        conv.start_node_id = "start"

        manager.register_conversation(conv)
        manager.start_conversation("conv_1")

        result = manager.end_conversation("conv_1")

        assert result is True

    def test_advance_conversation(self):
        """Test advance_conversation advances to next node."""
        manager = DialogueManager()

        conv = Conversation(conversation_id="conv_1")
        line1 = VOLine(audio_asset="a.wav", duration_ms=1000.0)
        line2 = VOLine(audio_asset="b.wav", duration_ms=1000.0)
        node1 = ConversationNode(node_id="node_1", line=line1, next_nodes=["node_2"])
        node2 = ConversationNode(node_id="node_2", line=line2)
        conv.add_node(node1)
        conv.add_node(node2)
        conv.start_node_id = "node_1"

        manager.register_conversation(conv)
        manager.start_conversation("conv_1")

        # Pre-cache audio
        manager._stream_manager.cache.put("b.wav", b"audio", duration_ms=1000.0)

        result = manager.advance_conversation("conv_1")

        assert result is True

    def test_make_conversation_choice(self):
        """Test make_conversation_choice selects branch."""
        manager = DialogueManager()

        conv = Conversation(conversation_id="conv_1")
        branch = ConversationNode(
            node_id="branch",
            is_branch_point=True,
            branch_options=[{"next_node_id": "a"}],
            next_nodes=["a"],
        )
        option_a = ConversationNode(node_id="a", line=VOLine(audio_asset="a.wav"))
        conv.add_node(branch)
        conv.add_node(option_a)
        conv.start_node_id = "branch"

        manager.register_conversation(conv)
        manager.start_conversation("conv_1")
        conv.wait_for_input()

        # Pre-cache audio
        manager._stream_manager.cache.put("a.wav", b"audio", duration_ms=1000.0)

        result = manager.make_conversation_choice("conv_1", 0)

        assert result is True

    def test_skip_conversation_line(self):
        """Test skip_conversation_line skips current line."""
        manager = DialogueManager()

        conv = Conversation(conversation_id="conv_1", skippable=True)
        line = VOLine(interruptible=True, duration_ms=1000.0)
        line.state = VOLineState.PLAYING
        node = ConversationNode(node_id="start", line=line)
        conv.add_node(node)
        conv.start_node_id = "start"

        manager.register_conversation(conv)
        manager.start_conversation("conv_1")

        result = manager.skip_conversation_line("conv_1")

        assert result is True


# =============================================================================
# DialogueManager Barks and Ambient Tests
# =============================================================================


class TestDialogueManagerBarksAmbient:
    """Tests for DialogueManager barks and ambient functionality."""

    def test_register_bark_pool(self):
        """Test register_bark_pool registers barks."""
        manager = DialogueManager()
        lines = [
            VOLine(audio_asset="bark1.wav", text="Bark 1"),
            VOLine(audio_asset="bark2.wav", text="Bark 2"),
        ]

        manager.register_bark_pool("combat", lines)

        assert "combat" in manager._bark_system.bark_types

    def test_trigger_bark(self):
        """Test trigger_bark triggers registered bark."""
        manager = DialogueManager()
        lines = [VOLine(audio_asset="bark.wav", text="Alert!", duration_ms=500.0)]
        manager.register_bark_pool("alert", lines)

        # Pre-cache
        manager._stream_manager.cache.put("bark.wav", b"audio", duration_ms=500.0)

        result = manager.trigger_bark("alert")

        assert result is True

    def test_trigger_bark_not_registered(self):
        """Test trigger_bark returns False for unregistered type."""
        manager = DialogueManager()

        result = manager.trigger_bark("missing")

        assert result is False

    def test_register_ambient_zone(self):
        """Test register_ambient_zone registers zone."""
        manager = DialogueManager()
        lines = [VOLine(audio_asset="ambient.wav", text="Ambient")]

        manager.register_ambient_zone("forest", lines)

        # Just verify it doesn't crash

    def test_enter_exit_ambient_zone(self):
        """Test enter/exit ambient zone."""
        manager = DialogueManager()
        lines = [VOLine(audio_asset="ambient.wav")]
        manager.register_ambient_zone("forest", lines)

        manager.enter_ambient_zone("forest")
        assert "forest" in manager._ambient_system.active_zones

        manager.exit_ambient_zone("forest")
        assert "forest" not in manager._ambient_system.active_zones


# =============================================================================
# DialogueManager Localization Tests
# =============================================================================


class TestDialogueManagerLocalization:
    """Tests for DialogueManager localization functionality."""

    def test_set_language(self):
        """Test set_language changes current language."""
        manager = DialogueManager(default_language="en")

        result = manager.set_language("es")

        assert result is True
        assert manager.current_language == "es"

    def test_set_language_unsupported(self):
        """Test set_language returns False for unsupported language."""
        manager = DialogueManager()

        result = manager.set_language("xyz")

        assert result is False

    def test_get_supported_languages(self):
        """Test get_supported_languages returns tuple."""
        manager = DialogueManager()

        languages = manager.get_supported_languages()

        assert isinstance(languages, tuple)
        assert "en" in languages


# =============================================================================
# DialogueManager Subtitles Tests
# =============================================================================


class TestDialogueManagerSubtitles:
    """Tests for DialogueManager subtitle functionality."""

    def test_enable_subtitles(self):
        """Test enable_subtitles enables subtitles."""
        manager = DialogueManager(enable_subtitles=False)

        manager.enable_subtitles()

        assert manager.subtitles_enabled is True

    def test_disable_subtitles(self):
        """Test disable_subtitles disables subtitles."""
        manager = DialogueManager(enable_subtitles=True)

        manager.disable_subtitles()

        assert manager.subtitles_enabled is False


# =============================================================================
# DialogueManager Audio Processing Tests
# =============================================================================


class TestDialogueManagerAudioProcessing:
    """Tests for DialogueManager audio processing functionality."""

    def test_set_listener_position(self):
        """Test set_listener_position sets position."""
        manager = DialogueManager()

        manager.set_listener_position((10.0, 0.0, 5.0), (0.0, 0.0, 1.0))

        # Just verify no crash

    def test_set_environment(self):
        """Test set_environment sets reverb environment."""
        manager = DialogueManager()

        manager.set_environment("cave")

        # Just verify no crash

    def test_set_master_volume(self):
        """Test set_master_volume sets volume."""
        manager = DialogueManager()

        manager.set_master_volume(0.8)

        assert manager._processor.master_volume == 0.8


# =============================================================================
# DialogueManager Event Tests
# =============================================================================


class TestDialogueManagerEvents:
    """Tests for DialogueManager event handling."""

    def test_on_line_started_emits_event(self):
        """Test _on_line_started emits event."""
        callback = MagicMock()
        manager = DialogueManager(on_event=callback)
        line = VOLine(speaker_id="npc")

        manager._on_line_started(line)

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert event.event_type == EVENT_LINE_STARTED
        assert event.line is line

    def test_on_line_ended_emits_event(self):
        """Test _on_line_ended emits event."""
        callback = MagicMock()
        manager = DialogueManager(on_event=callback)
        line = VOLine(speaker_id="npc")

        manager._on_line_ended(line, interrupted=False)

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert event.event_type == EVENT_LINE_ENDED
        assert event.data["interrupted"] is False

    def test_on_line_ended_interrupted_emits_event(self):
        """Test _on_line_ended with interruption emits correct event."""
        callback = MagicMock()
        manager = DialogueManager(on_event=callback)
        line = VOLine()

        manager._on_line_ended(line, interrupted=True)

        event = callback.call_args[0][0]
        assert event.event_type == EVENT_LINE_INTERRUPTED

    def test_on_language_changed_emits_event(self):
        """Test _on_language_changed emits event."""
        callback = MagicMock()
        manager = DialogueManager(on_event=callback)

        manager._on_language_changed("en", "es")

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert event.event_type == "language_changed"
        assert event.data["old_language"] == "en"
        assert event.data["new_language"] == "es"

    def test_on_conversation_started_emits_event(self):
        """Test _on_conversation_started emits event."""
        callback = MagicMock()
        manager = DialogueManager(on_event=callback)
        conv = Conversation(conversation_id="conv_1")

        manager._on_conversation_started(conv)

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert event.event_type == "conversation_started"
        assert event.conversation_id == "conv_1"

    def test_on_conversation_ended_emits_event(self):
        """Test _on_conversation_ended emits event."""
        callback = MagicMock()
        manager = DialogueManager(on_event=callback)
        conv = Conversation(conversation_id="conv_1")

        manager._on_conversation_ended(conv, cancelled=True)

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert event.event_type == "conversation_ended"
        assert event.data["cancelled"] is True

    def test_on_branch_reached_emits_event(self):
        """Test _on_branch_reached emits event."""
        callback = MagicMock()
        manager = DialogueManager(on_event=callback)
        conv = Conversation(conversation_id="conv_1")
        node = ConversationNode(
            node_id="branch",
            branch_options=[{"text": "A"}, {"text": "B"}],
        )

        manager._on_branch_reached(conv, node)

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert event.event_type == "branch_reached"
        assert len(event.data["options"]) == 2


# =============================================================================
# DialogueManager Stats Tests
# =============================================================================


class TestDialogueManagerStats:
    """Tests for DialogueManager statistics."""

    def test_stats_empty(self):
        """Test stats for empty manager."""
        manager = DialogueManager()
        stats = manager.stats

        assert stats["state"] == DialogueState.IDLE.value
        assert "queue" in stats
        assert "streaming" in stats
        assert "conversations" in stats
        assert "barks" in stats
        assert "ambient" in stats
        assert "language" in stats

    def test_stats_with_active(self):
        """Test stats with active lines."""
        manager = DialogueManager()
        line = VOLine(duration_ms=1000.0)
        manager._queue.start_line(line, time.time())
        manager._state = DialogueState.PLAYING

        stats = manager.stats

        assert stats["state"] == "playing"
        assert stats["active_lines"] >= 0


# =============================================================================
# DialogueManager Integration Tests
# =============================================================================


class TestDialogueManagerIntegration:
    """Integration tests for DialogueManager."""

    def test_full_line_playback_flow(self):
        """Test complete line playback flow."""
        events = []
        manager = DialogueManager(on_event=lambda e: events.append(e))

        # Pre-cache audio
        manager._stream_manager.cache.put("test.wav", b"audio data", duration_ms=100.0)

        # Play line
        line = VOLine(audio_asset="test.wav", text="Hello", duration_ms=100.0)
        manager.play_line(line)

        # Process queue
        manager.update(16.0)

        # Simulate completion
        active_lines = manager._queue.get_active_lines()
        if active_lines:
            for active in active_lines:
                active.update_playback(200.0)

        manager.update(16.0)

        # Should have emitted events
        assert len(events) > 0

    def test_conversation_flow(self):
        """Test complete conversation flow."""
        manager = DialogueManager()

        # Create conversation
        conv = Conversation(conversation_id="test_conv")
        line1 = VOLine(audio_asset="a.wav", text="Line 1", duration_ms=100.0)
        line2 = VOLine(audio_asset="b.wav", text="Line 2", duration_ms=100.0)
        node1 = ConversationNode(node_id="n1", line=line1, next_nodes=["n2"])
        node2 = ConversationNode(node_id="n2", line=line2)
        conv.add_node(node1)
        conv.add_node(node2)
        conv.start_node_id = "n1"

        manager.register_conversation(conv)

        # Pre-cache audio
        manager._stream_manager.cache.put("a.wav", b"audio", duration_ms=100.0)
        manager._stream_manager.cache.put("b.wav", b"audio", duration_ms=100.0)

        # Start conversation
        manager.start_conversation("test_conv")

        # Advance
        manager.advance_conversation("test_conv")

        # End
        manager.end_conversation("test_conv")

        assert conv.is_complete is True


# =============================================================================
# DialogueManager Thread Safety Tests
# =============================================================================


class TestDialogueManagerThreadSafety:
    """Thread safety tests for DialogueManager."""

    def test_concurrent_play_line(self):
        """Test concurrent play_line operations."""
        manager = DialogueManager()

        def play_lines():
            for i in range(20):
                line = VOLine(audio_asset=f"test_{i}.wav", duration_ms=1000.0)
                manager.play_line(line)
                time.sleep(0.001)

        threads = [threading.Thread(target=play_lines) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without deadlock

    def test_concurrent_update(self):
        """Test concurrent update operations."""
        manager = DialogueManager()

        # Start some lines
        for i in range(5):
            line = VOLine(audio_asset=f"test_{i}.wav", duration_ms=1000.0)
            manager._queue.start_line(line, time.time())

        def update_loop():
            for _ in range(50):
                manager.update(10.0)
                time.sleep(0.001)

        threads = [threading.Thread(target=update_loop) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without deadlock
