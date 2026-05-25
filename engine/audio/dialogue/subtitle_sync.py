"""
Subtitle Synchronization Module.

Handles subtitle timing, display, styling, and synchronization with audio playback.
Supports multiple simultaneous speakers and various display modes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Optional

from .config import (
    MAX_SUBTITLE_LINES,
    SUBTITLE_CHARS_PER_SECOND,
    SUBTITLE_FADE_TIME_MS,
    SUBTITLE_LINE_HEIGHT,
    SUBTITLE_MAX_WIDTH_PERCENT,
    SUBTITLE_MIN_DISPLAY_MS,
    EVENT_SUBTITLE_SHOW,
    EVENT_SUBTITLE_HIDE,
)
from .vo_line import SubtitleData, VOLine


class SubtitlePosition(str, Enum):
    """Subtitle display position."""
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    TOP_CENTER = "top_center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    CENTER = "center"
    SPEAKER_RELATIVE = "speaker_relative"


class SubtitleState(str, Enum):
    """State of a subtitle display."""
    HIDDEN = "hidden"
    FADING_IN = "fading_in"
    VISIBLE = "visible"
    FADING_OUT = "fading_out"


@dataclass
class SubtitleStyle:
    """Visual style for subtitles."""
    font_family: str = "default"
    font_size: int = 24
    font_weight: str = "normal"  # normal, bold
    text_color: str = "#FFFFFF"
    background_color: str = "#000000"
    background_opacity: float = 0.7
    outline_color: str = "#000000"
    outline_width: int = 2
    shadow_offset: tuple[int, int] = (2, 2)
    shadow_color: str = "#000000"
    line_spacing: float = 1.2
    letter_spacing: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "font_family": self.font_family,
            "font_size": self.font_size,
            "font_weight": self.font_weight,
            "text_color": self.text_color,
            "background_color": self.background_color,
            "background_opacity": self.background_opacity,
            "outline_color": self.outline_color,
            "outline_width": self.outline_width,
            "shadow_offset": self.shadow_offset,
            "shadow_color": self.shadow_color,
            "line_spacing": self.line_spacing,
            "letter_spacing": self.letter_spacing,
        }


@dataclass
class ActiveSubtitle:
    """An actively displayed subtitle."""
    subtitle_id: str
    line_id: str
    text: str
    speaker_id: str = ""
    speaker_name: str = ""
    speaker_color: str = "#FFFFFF"
    start_time: float = 0.0
    end_time: float = 0.0
    position: SubtitlePosition = SubtitlePosition.BOTTOM_CENTER
    style: SubtitleStyle = field(default_factory=SubtitleStyle)
    priority: int = 0

    # Runtime state
    _state: SubtitleState = field(default=SubtitleState.HIDDEN, init=False)
    _opacity: float = field(default=0.0, init=False)
    _fade_progress: float = field(default=0.0, init=False)

    @property
    def state(self) -> SubtitleState:
        """Get subtitle state."""
        return self._state

    @property
    def opacity(self) -> float:
        """Get current opacity."""
        return self._opacity

    @property
    def duration_ms(self) -> float:
        """Get subtitle duration."""
        return (self.end_time - self.start_time) * 1000

    @property
    def is_visible(self) -> bool:
        """Check if subtitle is visible."""
        return self._state in (
            SubtitleState.FADING_IN,
            SubtitleState.VISIBLE,
            SubtitleState.FADING_OUT,
        )

    def show(self) -> None:
        """Start showing the subtitle."""
        self._state = SubtitleState.FADING_IN
        self._fade_progress = 0.0

    def hide(self) -> None:
        """Start hiding the subtitle."""
        if self._state != SubtitleState.HIDDEN:
            self._state = SubtitleState.FADING_OUT
            self._fade_progress = 0.0

    def update(self, delta_ms: float, fade_time_ms: float) -> None:
        """Update subtitle animation state."""
        if self._state == SubtitleState.FADING_IN:
            self._fade_progress += delta_ms
            self._opacity = min(1.0, self._fade_progress / fade_time_ms)

            if self._fade_progress >= fade_time_ms:
                self._state = SubtitleState.VISIBLE
                self._opacity = 1.0

        elif self._state == SubtitleState.FADING_OUT:
            self._fade_progress += delta_ms
            self._opacity = max(0.0, 1.0 - (self._fade_progress / fade_time_ms))

            if self._fade_progress >= fade_time_ms:
                self._state = SubtitleState.HIDDEN
                self._opacity = 0.0


@dataclass
class SubtitleCue:
    """A timed subtitle cue point."""
    time_ms: float
    text: str
    duration_ms: float = 0.0
    speaker_id: str = ""

    @property
    def end_time_ms(self) -> float:
        """Get end time of cue."""
        return self.time_ms + self.duration_ms


class SubtitleTrack:
    """
    A track of timed subtitle cues for a VO line.
    """

    def __init__(self, line_id: str) -> None:
        """Initialize subtitle track."""
        self._line_id = line_id
        self._cues: list[SubtitleCue] = []
        self._current_index = 0

    @property
    def line_id(self) -> str:
        """Get line ID."""
        return self._line_id

    def add_cue(self, cue: SubtitleCue) -> None:
        """Add a cue to the track."""
        self._cues.append(cue)
        # Keep sorted by time
        self._cues.sort(key=lambda c: c.time_ms)

    def get_cue_at_time(self, time_ms: float) -> Optional[SubtitleCue]:
        """Get the cue that should be displayed at a given time."""
        for cue in self._cues:
            if cue.time_ms <= time_ms < cue.end_time_ms:
                return cue
        return None

    def get_next_cue(self, current_time_ms: float) -> Optional[SubtitleCue]:
        """Get the next upcoming cue."""
        for cue in self._cues:
            if cue.time_ms > current_time_ms:
                return cue
        return None

    def reset(self) -> None:
        """Reset track position."""
        self._current_index = 0

    @property
    def cue_count(self) -> int:
        """Get number of cues."""
        return len(self._cues)

    def __iter__(self) -> Iterator[SubtitleCue]:
        return iter(self._cues)


class SubtitleManager:
    """
    Manages subtitle display and synchronization.
    """

    def __init__(
        self,
        max_lines: int = MAX_SUBTITLE_LINES,
        fade_time_ms: float = SUBTITLE_FADE_TIME_MS,
        min_display_ms: float = SUBTITLE_MIN_DISPLAY_MS,
        chars_per_second: int = SUBTITLE_CHARS_PER_SECOND,
        on_subtitle_show: Optional[Callable[[ActiveSubtitle], None]] = None,
        on_subtitle_hide: Optional[Callable[[ActiveSubtitle], None]] = None,
        on_subtitle_update: Optional[Callable[[list[ActiveSubtitle]], None]] = None,
    ) -> None:
        """
        Initialize the subtitle manager.

        Args:
            max_lines: Maximum simultaneous subtitle lines
            fade_time_ms: Fade in/out duration
            min_display_ms: Minimum display time
            chars_per_second: Characters per second for timing
            on_subtitle_show: Callback when subtitle appears
            on_subtitle_hide: Callback when subtitle disappears
            on_subtitle_update: Callback for display updates
        """
        self._max_lines = max_lines
        self._fade_time_ms = fade_time_ms
        self._min_display_ms = min_display_ms
        self._chars_per_second = chars_per_second
        self._lock = threading.RLock()

        # Active subtitles
        self._active_subtitles: dict[str, ActiveSubtitle] = {}

        # Subtitle tracks by line ID
        self._tracks: dict[str, SubtitleTrack] = {}

        # Default styles by speaker
        self._speaker_styles: dict[str, SubtitleStyle] = {}
        self._default_style = SubtitleStyle()

        # Callbacks
        self._on_subtitle_show = on_subtitle_show
        self._on_subtitle_hide = on_subtitle_hide
        self._on_subtitle_update = on_subtitle_update

        # Enabled state
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Check if subtitles are enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable subtitles."""
        self._enabled = value
        if not value:
            self.hide_all()

    @property
    def active_count(self) -> int:
        """Get number of active subtitles."""
        with self._lock:
            return sum(
                1 for s in self._active_subtitles.values() if s.is_visible
            )

    def set_speaker_style(self, speaker_id: str, style: SubtitleStyle) -> None:
        """Set style for a specific speaker."""
        self._speaker_styles[speaker_id] = style

    def get_speaker_style(self, speaker_id: str) -> SubtitleStyle:
        """Get style for a speaker."""
        return self._speaker_styles.get(speaker_id, self._default_style)

    def set_default_style(self, style: SubtitleStyle) -> None:
        """Set the default subtitle style."""
        self._default_style = style

    def calculate_display_duration(self, text: str) -> float:
        """Calculate appropriate display duration for text."""
        # Based on reading speed
        char_duration = (len(text) / self._chars_per_second) * 1000
        return max(self._min_display_ms, char_duration)

    # =========================================================================
    # Subtitle Track Management
    # =========================================================================

    def create_track_from_line(self, line: VOLine) -> SubtitleTrack:
        """
        Create a subtitle track from a VO line.

        Args:
            line: The VO line

        Returns:
            SubtitleTrack with timing info
        """
        track = SubtitleTrack(line.line_id)

        if line.subtitle:
            cue = SubtitleCue(
                time_ms=line.subtitle.start_time_ms,
                text=line.subtitle.text,
                duration_ms=line.subtitle.duration_ms,
                speaker_id=line.speaker_id,
            )
            track.add_cue(cue)
        elif line.text:
            # Create a simple cue for the full text
            duration = self.calculate_display_duration(line.text)
            cue = SubtitleCue(
                time_ms=0.0,
                text=line.text,
                duration_ms=duration,
                speaker_id=line.speaker_id,
            )
            track.add_cue(cue)

        self._tracks[line.line_id] = track
        return track

    def get_track(self, line_id: str) -> Optional[SubtitleTrack]:
        """Get subtitle track for a line."""
        return self._tracks.get(line_id)

    def remove_track(self, line_id: str) -> bool:
        """Remove a subtitle track."""
        if line_id in self._tracks:
            del self._tracks[line_id]
            return True
        return False

    # =========================================================================
    # Subtitle Display
    # =========================================================================

    def show_subtitle(
        self,
        line: VOLine,
        current_time: Optional[float] = None,
        position: SubtitlePosition = SubtitlePosition.BOTTOM_CENTER,
    ) -> Optional[ActiveSubtitle]:
        """
        Show a subtitle for a VO line.

        Args:
            line: The VO line
            current_time: Current time
            position: Display position

        Returns:
            The active subtitle or None if disabled
        """
        if not self._enabled:
            return None

        if current_time is None:
            current_time = time.time()

        with self._lock:
            # Check if at capacity
            visible_count = sum(
                1 for s in self._active_subtitles.values() if s.is_visible
            )
            if visible_count >= self._max_lines:
                # Hide oldest subtitle
                oldest = min(
                    (s for s in self._active_subtitles.values() if s.is_visible),
                    key=lambda s: s.start_time,
                    default=None,
                )
                if oldest:
                    oldest.hide()

            # Get or calculate duration
            if line.subtitle:
                text = line.subtitle.text
                speaker_name = line.subtitle.speaker_name
                speaker_color = line.subtitle.speaker_color
                duration_ms = line.subtitle.duration_ms
            else:
                text = line.text
                speaker_name = ""
                speaker_color = "#FFFFFF"
                duration_ms = self.calculate_display_duration(text)

            if not text:
                return None

            # Create active subtitle
            style = self.get_speaker_style(line.speaker_id)
            subtitle = ActiveSubtitle(
                subtitle_id=f"sub_{line.line_id}",
                line_id=line.line_id,
                text=text,
                speaker_id=line.speaker_id,
                speaker_name=speaker_name,
                speaker_color=speaker_color,
                start_time=current_time,
                end_time=current_time + (duration_ms / 1000),
                position=position,
                style=style,
                priority=line.priority,
            )

            subtitle.show()
            self._active_subtitles[subtitle.subtitle_id] = subtitle

            if self._on_subtitle_show:
                self._on_subtitle_show(subtitle)

            return subtitle

    def hide_subtitle(self, subtitle_id: str) -> bool:
        """
        Hide a subtitle.

        Args:
            subtitle_id: ID of subtitle to hide

        Returns:
            True if subtitle was found and hidden
        """
        with self._lock:
            subtitle = self._active_subtitles.get(subtitle_id)
            if subtitle:
                subtitle.hide()
                return True
            return False

    def hide_for_line(self, line_id: str) -> bool:
        """Hide subtitle for a specific line."""
        with self._lock:
            for subtitle in self._active_subtitles.values():
                if subtitle.line_id == line_id:
                    subtitle.hide()
                    return True
            return False

    def hide_for_speaker(self, speaker_id: str) -> int:
        """Hide all subtitles for a speaker."""
        count = 0
        with self._lock:
            for subtitle in self._active_subtitles.values():
                if subtitle.speaker_id == speaker_id and subtitle.is_visible:
                    subtitle.hide()
                    count += 1
        return count

    def hide_all(self) -> int:
        """Hide all active subtitles."""
        count = 0
        with self._lock:
            for subtitle in self._active_subtitles.values():
                if subtitle.is_visible:
                    subtitle.hide()
                    count += 1
        return count

    def update(self, delta_ms: float, current_time: Optional[float] = None) -> None:
        """
        Update subtitle states.

        Args:
            delta_ms: Time since last update
            current_time: Current time
        """
        if not self._enabled:
            return

        if current_time is None:
            current_time = time.time()

        to_remove = []

        with self._lock:
            for subtitle_id, subtitle in self._active_subtitles.items():
                # Update animation
                subtitle.update(delta_ms, self._fade_time_ms)

                # Check if should start hiding
                if (
                    subtitle.state == SubtitleState.VISIBLE
                    and current_time >= subtitle.end_time
                ):
                    subtitle.hide()

                # Check if fully hidden
                if subtitle.state == SubtitleState.HIDDEN:
                    to_remove.append(subtitle_id)

                    if self._on_subtitle_hide:
                        self._on_subtitle_hide(subtitle)

            # Remove hidden subtitles
            for subtitle_id in to_remove:
                del self._active_subtitles[subtitle_id]

            # Notify update
            if self._on_subtitle_update and self._active_subtitles:
                visible = [
                    s for s in self._active_subtitles.values() if s.is_visible
                ]
                if visible:
                    self._on_subtitle_update(visible)

    def get_visible_subtitles(self) -> list[ActiveSubtitle]:
        """Get list of visible subtitles in display order."""
        with self._lock:
            visible = [
                s for s in self._active_subtitles.values() if s.is_visible
            ]
            # Sort by priority (higher first) then start time (older first)
            visible.sort(key=lambda s: (-s.priority, s.start_time))
            return visible[:self._max_lines]

    def get_subtitle_for_line(self, line_id: str) -> Optional[ActiveSubtitle]:
        """Get active subtitle for a line."""
        with self._lock:
            for subtitle in self._active_subtitles.values():
                if subtitle.line_id == line_id:
                    return subtitle
            return None

    # =========================================================================
    # Synchronization
    # =========================================================================

    def sync_with_playback(
        self,
        line: VOLine,
        playback_time_ms: float,
        current_time: float,
    ) -> None:
        """
        Synchronize subtitle display with audio playback position.

        Args:
            line: The playing VO line
            playback_time_ms: Current playback position
            current_time: Current system time
        """
        if not self._enabled:
            return

        track = self._tracks.get(line.line_id)
        if not track:
            return

        with self._lock:
            current_cue = track.get_cue_at_time(playback_time_ms)
            active_sub = self.get_subtitle_for_line(line.line_id)

            if current_cue:
                if not active_sub or active_sub.text != current_cue.text:
                    # Show new cue
                    if active_sub:
                        active_sub.hide()

                    self.show_subtitle(
                        VOLine(
                            line_id=line.line_id,
                            text=current_cue.text,
                            speaker_id=current_cue.speaker_id or line.speaker_id,
                            priority=line.priority,
                        ),
                        current_time,
                    )
            elif active_sub and active_sub.is_visible:
                # No cue at this time, hide subtitle
                active_sub.hide()


# =============================================================================
# Helper Functions
# =============================================================================


def create_subtitle_data(
    text: str,
    speaker_name: str = "",
    speaker_color: str = "#FFFFFF",
    duration_ms: Optional[float] = None,
    chars_per_second: int = SUBTITLE_CHARS_PER_SECOND,
) -> SubtitleData:
    """
    Create subtitle data with automatic duration calculation.

    Args:
        text: Subtitle text
        speaker_name: Display name of speaker
        speaker_color: Color for speaker name
        duration_ms: Optional explicit duration
        chars_per_second: Reading speed for auto-calculation

    Returns:
        SubtitleData instance
    """
    if duration_ms is None:
        duration_ms = max(
            SUBTITLE_MIN_DISPLAY_MS,
            (len(text) / chars_per_second) * 1000,
        )

    return SubtitleData(
        text=text,
        speaker_name=speaker_name,
        speaker_color=speaker_color,
        start_time_ms=0.0,
        end_time_ms=duration_ms,
    )
