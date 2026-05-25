"""
Dialogue Manager Module.

Main coordination class for the dialogue system. Integrates voice-over
queue, streaming, processing, subtitles, localization, and conversations.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .config import (
    DEFAULT_INTERRUPT_PRIORITY,
    DEFAULT_LANGUAGE,
    DialogueState,
    EVENT_LINE_ENDED,
    EVENT_LINE_INTERRUPTED,
    EVENT_LINE_STARTED,
    MAX_QUEUE_SIZE,
    MAX_SIMULTANEOUS_VO,
    QUEUE_PROCESS_INTERVAL_MS,
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
    VO_DUCKING_RELEASE_MS,
    VO_DUCKING_TIME_MS,
)
from .contextual_dialogue import BarkSystem, AmbientVOSystem, ContextualDialogueManager
from .conversation import Conversation, ConversationManager, ConversationNode
from .localization import LocalizationManager
from .subtitle_sync import SubtitleManager, SubtitlePosition
from .vo_line import VOLine, VOLineState
from .vo_processing import VOProcessor, VOProcessingState
from .vo_queue import VOQueue, VOQueueManager
from .vo_streaming import VOStreamManager, StreamHandle


@dataclass
class DialogueEvent:
    """Event emitted by the dialogue system."""
    event_type: str
    timestamp: float
    line: Optional[VOLine] = None
    conversation_id: Optional[str] = None
    speaker_id: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)


class DialogueManager:
    """
    Central manager for all dialogue and voice-over functionality.

    Coordinates:
    - Voice-over queue and priority management
    - Audio streaming and caching
    - Audio processing (radio, distance, reverb)
    - Subtitle display
    - Localization
    - Conversations
    - Contextual dialogue (barks, ambient)
    """

    def __init__(
        self,
        max_queue_size: int = MAX_QUEUE_SIZE,
        max_simultaneous: int = MAX_SIMULTANEOUS_VO,
        default_language: str = DEFAULT_LANGUAGE,
        enable_subtitles: bool = True,
        cache_size_mb: int = 32,
        on_event: Optional[Callable[[DialogueEvent], None]] = None,
        audio_play_callback: Optional[Callable[[VOLine, StreamHandle], None]] = None,
        audio_stop_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Initialize the dialogue manager.

        Args:
            max_queue_size: Maximum VO queue size
            max_simultaneous: Maximum concurrent VO playback
            default_language: Default language code
            enable_subtitles: Enable subtitle display
            cache_size_mb: Audio cache size in MB
            on_event: Callback for dialogue events
            audio_play_callback: Callback to play audio (line, stream) -> None
            audio_stop_callback: Callback to stop audio (line_id) -> None
        """
        self._lock = threading.RLock()
        self._state = DialogueState.IDLE

        # Core components
        self._queue = VOQueue(
            max_size=max_queue_size,
            max_simultaneous=max_simultaneous,
            on_line_started=self._on_line_started,
            on_line_ended=self._on_line_ended,
        )

        self._stream_manager = VOStreamManager(
            cache_size_mb=cache_size_mb,
            on_stream_ready=self._on_stream_ready,
        )

        self._processor = VOProcessor(
            on_state_changed=self._on_processing_changed,
        )

        self._subtitle_manager = SubtitleManager(
            on_subtitle_show=self._on_subtitle_show,
            on_subtitle_hide=self._on_subtitle_hide,
        )
        self._subtitle_manager.enabled = enable_subtitles

        self._localization = LocalizationManager(
            default_language=default_language,
            on_language_changed=self._on_language_changed,
        )

        self._conversation_manager = ConversationManager(
            on_conversation_started=self._on_conversation_started,
            on_conversation_ended=self._on_conversation_ended,
            on_line_started=self._on_conversation_line_started,
            on_line_ended=self._on_conversation_line_ended,
            on_branch_reached=self._on_branch_reached,
        )

        self._bark_system = BarkSystem(
            on_bark_triggered=self._on_bark_triggered,
        )

        self._ambient_system = AmbientVOSystem(
            on_ambient_triggered=self._on_ambient_triggered,
        )

        self._contextual_manager = ContextualDialogueManager()

        # Callbacks
        self._on_event = on_event
        self._audio_play_callback = audio_play_callback
        self._audio_stop_callback = audio_stop_callback

        # Active streams by line ID
        self._active_streams: dict[str, StreamHandle] = {}

        # Timing
        self._last_update_time = time.time()

        # Ducking state
        self._is_ducking = False
        self._duck_target_db = 0.0

    @property
    def state(self) -> DialogueState:
        """Get current dialogue manager state."""
        return self._state

    @property
    def is_playing(self) -> bool:
        """Check if any dialogue is playing."""
        return self._queue.is_playing

    @property
    def current_language(self) -> str:
        """Get current language."""
        return self._localization.current_language

    # =========================================================================
    # Core Operations
    # =========================================================================

    def start(self) -> None:
        """Start the dialogue manager."""
        with self._lock:
            self._state = DialogueState.IDLE
            self._last_update_time = time.time()

    def stop(self) -> None:
        """Stop the dialogue manager and clear all active VO."""
        with self._lock:
            # Stop all active lines
            for line in self._queue.get_active_lines():
                self._stop_line(line, interrupted=True)

            self._queue.clear()
            self._conversation_manager.cancel_all()
            self._subtitle_manager.hide_all()
            self._state = DialogueState.IDLE

    def pause(self) -> None:
        """Pause all dialogue playback."""
        with self._lock:
            self._state = DialogueState.PAUSED
            self._queue.pause()
            self._conversation_manager.pause_all()

            # Pause active streams
            for stream in self._active_streams.values():
                self._stream_manager.pause_stream(stream.stream_id)

    def resume(self) -> None:
        """Resume dialogue playback."""
        with self._lock:
            self._state = DialogueState.PLAYING if self._queue.is_playing else DialogueState.IDLE
            self._queue.resume()
            self._conversation_manager.resume_all()

            # Resume active streams
            for stream in self._active_streams.values():
                self._stream_manager.resume_stream(stream.stream_id)

    def update(self, delta_ms: float) -> None:
        """
        Update the dialogue manager.

        Should be called each frame.

        Args:
            delta_ms: Time since last update in milliseconds
        """
        current_time = time.time()

        with self._lock:
            # Update queue
            completed = self._queue.update(delta_ms)
            for line in completed:
                self._cleanup_line(line)

            # Update streams
            for line_id, stream in list(self._active_streams.items()):
                self._stream_manager.update_stream(stream.stream_id, delta_ms)

            # Update subtitles
            self._subtitle_manager.update(delta_ms, current_time)

            # Update conversations
            self._conversation_manager.update(delta_ms, current_time)

            # Update ambient system
            ambient_line = self._ambient_system.update(current_time)
            if ambient_line:
                self.play_line(ambient_line)

            # Update processing
            self._processor.update(delta_ms)

            # Process queue if idle
            if self._queue.can_play_more and not self._queue.is_empty:
                self._process_queue(current_time)

        self._last_update_time = current_time

    # =========================================================================
    # Line Playback
    # =========================================================================

    def play_line(
        self,
        line: VOLine,
        position: Optional[tuple[float, float, float]] = None,
        radio_effect: bool = False,
        force: bool = False,
    ) -> bool:
        """
        Play a voice-over line.

        Args:
            line: The VO line to play
            position: Optional 3D position for spatialization
            radio_effect: Apply radio/communication effect
            force: Force play even if queue is full

        Returns:
            True if line was queued
        """
        with self._lock:
            # Localize line
            localized = self._localization.localize_line(line)

            # Check for interrupt
            if localized.priority >= DEFAULT_INTERRUPT_PRIORITY:
                self._queue.interrupt_for(localized.priority)

            # Enqueue
            if not self._queue.enqueue(localized, force=force):
                return False

            # Setup processing
            state = self._processor.create_state(
                localized.line_id,
                radio_enabled=radio_effect,
                spatial_enabled=position is not None,
            )

            if position:
                self._processor.set_position(localized.line_id, position)

            # Preload audio
            self._stream_manager.preload_line(localized)

            return True

    def stop_line(self, line_id: str) -> bool:
        """
        Stop a playing line.

        Args:
            line_id: ID of line to stop

        Returns:
            True if line was stopped
        """
        with self._lock:
            for line in self._queue.get_active_lines():
                if line.line_id == line_id:
                    return self._stop_line(line, interrupted=True)
            return False

    def _stop_line(self, line: VOLine, interrupted: bool = False) -> bool:
        """Internal stop line implementation."""
        # Stop audio
        if self._audio_stop_callback:
            self._audio_stop_callback(line.line_id)

        # Stop stream
        stream = self._active_streams.get(line.line_id)
        if stream:
            self._stream_manager.stop_stream(stream.stream_id)
            del self._active_streams[line.line_id]

        # Hide subtitle
        self._subtitle_manager.hide_for_line(line.line_id)

        # Update queue
        self._queue.end_line(line, interrupted)

        # Cleanup processing
        self._processor.remove_state(line.line_id)

        return True

    def _cleanup_line(self, line: VOLine) -> None:
        """Cleanup after line completion."""
        if line.line_id in self._active_streams:
            stream = self._active_streams[line.line_id]
            self._stream_manager.stop_stream(stream.stream_id)
            del self._active_streams[line.line_id]

        self._subtitle_manager.hide_for_line(line.line_id)
        self._processor.remove_state(line.line_id)

    def _process_queue(self, current_time: float) -> None:
        """Process the queue and start next line."""
        line = self._queue.dequeue()
        if not line:
            return

        # Start stream
        stream = self._stream_manager.start_stream(
            line,
            on_ready=lambda s: self._start_line_playback(line, s, current_time),
            on_complete=lambda s: self._on_stream_complete(line, s),
        )

        if stream:
            self._active_streams[line.line_id] = stream

    def _start_line_playback(
        self,
        line: VOLine,
        stream: StreamHandle,
        current_time: float,
    ) -> None:
        """Start actual line playback when stream is ready."""
        with self._lock:
            # Start in queue
            if not self._queue.start_line(line, current_time):
                return

            # Play stream
            self._stream_manager.play_stream(stream.stream_id)

            # Show subtitle
            self._subtitle_manager.show_subtitle(line, current_time)

            # Create subtitle track for sync
            self._subtitle_manager.create_track_from_line(line)

            # Call audio callback
            if self._audio_play_callback:
                self._audio_play_callback(line, stream)

            self._state = DialogueState.PLAYING

    def _on_stream_complete(self, line: VOLine, stream: StreamHandle) -> None:
        """Handle stream completion."""
        with self._lock:
            self._cleanup_line(line)

            if not self._queue.is_playing:
                self._state = DialogueState.IDLE

    # =========================================================================
    # Conversations
    # =========================================================================

    def register_conversation(self, conversation: Conversation) -> None:
        """Register a conversation."""
        self._conversation_manager.register_conversation(conversation)

    def start_conversation(self, conversation_id: str) -> bool:
        """
        Start a registered conversation.

        Args:
            conversation_id: ID of conversation to start

        Returns:
            True if conversation started
        """
        with self._lock:
            node = self._conversation_manager.start_conversation(
                conversation_id, time.time()
            )

            if node and node.has_line:
                self.play_line(node.line)
                return True

            return node is not None

    def end_conversation(self, conversation_id: str, cancelled: bool = False) -> bool:
        """End a conversation."""
        return self._conversation_manager.end_conversation(conversation_id, cancelled)

    def advance_conversation(self, conversation_id: str) -> bool:
        """Advance to next line in conversation."""
        with self._lock:
            node = self._conversation_manager.advance_conversation(
                conversation_id, current_time=time.time()
            )

            if node and node.has_line:
                self.play_line(node.line)
                return True

            return node is not None

    def make_conversation_choice(
        self,
        conversation_id: str,
        choice_index: int,
    ) -> bool:
        """Make a choice at a conversation branch point."""
        with self._lock:
            node = self._conversation_manager.make_choice(
                conversation_id, choice_index
            )

            if node and node.has_line:
                self.play_line(node.line)
                return True

            return node is not None

    def skip_conversation_line(self, conversation_id: str) -> bool:
        """Skip current line in a conversation."""
        return self._conversation_manager.skip_line(conversation_id)

    # =========================================================================
    # Barks and Ambient
    # =========================================================================

    def register_bark_pool(
        self,
        bark_type: str,
        lines: list[VOLine],
    ) -> None:
        """Register a pool of barks."""
        self._bark_system.register_bark_pool(bark_type, lines)

    def trigger_bark(
        self,
        bark_type: str,
        speaker_id: Optional[str] = None,
        position: Optional[tuple[float, float, float]] = None,
    ) -> bool:
        """
        Trigger a bark.

        Args:
            bark_type: Type of bark to trigger
            speaker_id: Optional speaker filter
            position: Optional 3D position

        Returns:
            True if bark was triggered
        """
        line = self._bark_system.trigger_bark(bark_type, speaker_id)
        if line:
            return self.play_line(line, position=position)
        return False

    def register_ambient_zone(
        self,
        zone_id: str,
        lines: list[VOLine],
    ) -> None:
        """Register ambient VO for a zone."""
        self._ambient_system.register_zone(zone_id, lines)

    def enter_ambient_zone(self, zone_id: str) -> None:
        """Notify entering an ambient zone."""
        self._ambient_system.enter_zone(zone_id)

    def exit_ambient_zone(self, zone_id: str) -> None:
        """Notify exiting an ambient zone."""
        self._ambient_system.exit_zone(zone_id)

    # =========================================================================
    # Localization
    # =========================================================================

    def set_language(self, language: str) -> bool:
        """
        Set the current language.

        Args:
            language: Language code

        Returns:
            True if language was changed
        """
        return self._localization.set_language(language)

    def get_supported_languages(self) -> tuple[str, ...]:
        """Get supported languages."""
        return self._localization.supported_languages

    # =========================================================================
    # Subtitles
    # =========================================================================

    def enable_subtitles(self) -> None:
        """Enable subtitle display."""
        self._subtitle_manager.enabled = True

    def disable_subtitles(self) -> None:
        """Disable subtitle display."""
        self._subtitle_manager.enabled = False

    @property
    def subtitles_enabled(self) -> bool:
        """Check if subtitles are enabled."""
        return self._subtitle_manager.enabled

    # =========================================================================
    # Audio Processing
    # =========================================================================

    def set_listener_position(
        self,
        position: tuple[float, float, float],
        forward: Optional[tuple[float, float, float]] = None,
    ) -> None:
        """Set listener position for spatial audio."""
        self._processor.set_listener_position(position, forward)

    def set_environment(self, environment_type: str) -> None:
        """Set environment for reverb."""
        self._processor.set_environment(environment_type)

    def set_master_volume(self, volume: float) -> None:
        """Set master VO volume."""
        self._processor.master_volume = volume

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_line_started(self, line: VOLine) -> None:
        """Handle line started."""
        self._emit_event(DialogueEvent(
            event_type=EVENT_LINE_STARTED,
            timestamp=time.time(),
            line=line,
            speaker_id=line.speaker_id,
        ))

    def _on_line_ended(self, line: VOLine, interrupted: bool) -> None:
        """Handle line ended."""
        event_type = EVENT_LINE_INTERRUPTED if interrupted else EVENT_LINE_ENDED
        self._emit_event(DialogueEvent(
            event_type=event_type,
            timestamp=time.time(),
            line=line,
            speaker_id=line.speaker_id,
            data={"interrupted": interrupted},
        ))

    def _on_stream_ready(self, stream: StreamHandle) -> None:
        """Handle stream ready."""
        pass  # Handled by callback in start_stream

    def _on_processing_changed(self, source_id: str, state: VOProcessingState) -> None:
        """Handle processing state change."""
        pass

    def _on_subtitle_show(self, subtitle: Any) -> None:
        """Handle subtitle show."""
        pass

    def _on_subtitle_hide(self, subtitle: Any) -> None:
        """Handle subtitle hide."""
        pass

    def _on_language_changed(self, old_lang: str, new_lang: str) -> None:
        """Handle language change."""
        self._emit_event(DialogueEvent(
            event_type="language_changed",
            timestamp=time.time(),
            data={"old_language": old_lang, "new_language": new_lang},
        ))

    def _on_conversation_started(self, conversation: Conversation) -> None:
        """Handle conversation started."""
        self._emit_event(DialogueEvent(
            event_type="conversation_started",
            timestamp=time.time(),
            conversation_id=conversation.conversation_id,
        ))

    def _on_conversation_ended(
        self, conversation: Conversation, cancelled: bool
    ) -> None:
        """Handle conversation ended."""
        self._emit_event(DialogueEvent(
            event_type="conversation_ended",
            timestamp=time.time(),
            conversation_id=conversation.conversation_id,
            data={"cancelled": cancelled},
        ))

    def _on_conversation_line_started(
        self, conversation: Conversation, line: VOLine
    ) -> None:
        """Handle conversation line started."""
        pass

    def _on_conversation_line_ended(
        self, conversation: Conversation, line: VOLine
    ) -> None:
        """Handle conversation line ended."""
        pass

    def _on_branch_reached(
        self, conversation: Conversation, node: ConversationNode
    ) -> None:
        """Handle branch point reached."""
        self._emit_event(DialogueEvent(
            event_type="branch_reached",
            timestamp=time.time(),
            conversation_id=conversation.conversation_id,
            data={
                "node_id": node.node_id,
                "options": node.branch_options,
            },
        ))

    def _on_bark_triggered(self, line: VOLine, bark_type: str) -> None:
        """Handle bark triggered."""
        pass

    def _on_ambient_triggered(self, line: VOLine) -> None:
        """Handle ambient VO triggered."""
        pass

    def _emit_event(self, event: DialogueEvent) -> None:
        """Emit a dialogue event."""
        if self._on_event:
            self._on_event(event)

    # =========================================================================
    # Statistics
    # =========================================================================

    @property
    def stats(self) -> dict[str, Any]:
        """Get dialogue manager statistics."""
        with self._lock:
            return {
                "state": self._state.value,
                "queue": self._queue.stats,
                "streaming": self._stream_manager.stats,
                "active_lines": len(self._active_streams),
                "conversations": {
                    "active": self._conversation_manager.active_count,
                },
                "barks": {
                    "enabled": self._bark_system.is_enabled,
                    "types": len(self._bark_system.bark_types),
                },
                "ambient": {
                    "enabled": self._ambient_system.is_enabled,
                    "active_zones": len(self._ambient_system.active_zones),
                },
                "language": self._localization.current_language,
                "subtitles_enabled": self._subtitle_manager.enabled,
            }
