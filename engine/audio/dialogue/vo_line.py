"""
Voice-Over Line Module.

Defines the VOLine class representing an individual voice-over line with
all associated metadata including audio reference, subtitles, timing,
speaker info, lip sync data, and playback properties.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .config import (
    PRIORITY_NORMAL,
    VOPriority,
    ContextType,
)


class VOLineState(str, Enum):
    """Playback state of a VO line."""
    PENDING = "pending"
    LOADING = "loading"
    READY = "ready"
    PLAYING = "playing"
    PAUSED = "paused"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass
class LipSyncData:
    """Lip sync animation data for a VO line."""
    phonemes: list[tuple[float, str]] = field(default_factory=list)  # (time, phoneme)
    visemes: list[tuple[float, int]] = field(default_factory=list)   # (time, viseme_id)
    blend_shapes: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

    def get_phoneme_at(self, time_ms: float) -> Optional[str]:
        """Get the phoneme at a specific time."""
        for i, (t, phoneme) in enumerate(self.phonemes):
            if i + 1 < len(self.phonemes):
                if t <= time_ms < self.phonemes[i + 1][0]:
                    return phoneme
            elif t <= time_ms:
                return phoneme
        return None

    def get_viseme_at(self, time_ms: float) -> Optional[int]:
        """Get the viseme ID at a specific time."""
        for i, (t, viseme) in enumerate(self.visemes):
            if i + 1 < len(self.visemes):
                if t <= time_ms < self.visemes[i + 1][0]:
                    return viseme
            elif t <= time_ms:
                return viseme
        return None


@dataclass
class SubtitleData:
    """Subtitle display data for a VO line."""
    text: str
    speaker_name: str = ""
    speaker_color: str = "#FFFFFF"
    start_time_ms: float = 0.0
    end_time_ms: float = 0.0
    position: tuple[float, float] = (0.5, 0.9)  # Normalized screen coords
    alignment: str = "center"
    font_size: int = 24

    @property
    def duration_ms(self) -> float:
        """Get subtitle duration in milliseconds."""
        return self.end_time_ms - self.start_time_ms


@dataclass
class VOLine:
    """
    Represents a single voice-over line with all metadata.

    Attributes:
        line_id: Unique identifier for this line
        audio_asset: Reference to the audio asset (path or ID)
        text: The spoken text content
        speaker_id: Identifier of the speaking character
        duration_ms: Duration of the audio in milliseconds
        priority: Playback priority for queue ordering
        interruptible: Whether this line can be interrupted
        context_type: Type of dialogue context
        tags: Custom tags for filtering and selection
        conditions: Conditions that must be true for playback
        lip_sync: Lip sync animation data
        subtitle: Subtitle display data
        language: Language code for this line
        on_start: Callback when line starts playing
        on_end: Callback when line finishes or is interrupted
    """
    line_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    audio_asset: str = ""
    text: str = ""
    speaker_id: str = ""
    duration_ms: float = 0.0
    priority: int = PRIORITY_NORMAL
    interruptible: bool = True
    context_type: str = ContextType.BARK.value
    tags: set[str] = field(default_factory=set)
    conditions: dict[str, Any] = field(default_factory=dict)
    lip_sync: Optional[LipSyncData] = None
    subtitle: Optional[SubtitleData] = None
    language: str = "en"
    weight: float = 1.0  # For weighted selection
    cooldown_ms: float = 0.0  # Override cooldown for this line

    # Callbacks
    on_start: Optional[Callable[[VOLine], None]] = field(default=None, repr=False)
    on_end: Optional[Callable[[VOLine, bool], None]] = field(default=None, repr=False)

    # Runtime state (not part of definition)
    _state: VOLineState = field(default=VOLineState.PENDING, init=False)
    _playback_position_ms: float = field(default=0.0, init=False)
    _last_played_time: float = field(default=0.0, init=False)
    _play_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate and initialize the VO line."""
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
        if isinstance(self.priority, VOPriority):
            self.priority = int(self.priority)

    @property
    def state(self) -> VOLineState:
        """Get the current playback state."""
        return self._state

    @state.setter
    def state(self, value: VOLineState) -> None:
        """Set the playback state."""
        self._state = value

    @property
    def playback_position_ms(self) -> float:
        """Get current playback position in milliseconds."""
        return self._playback_position_ms

    @playback_position_ms.setter
    def playback_position_ms(self, value: float) -> None:
        """Set playback position."""
        self._playback_position_ms = max(0.0, min(value, self.duration_ms))

    @property
    def progress(self) -> float:
        """Get playback progress as a normalized value (0-1)."""
        if self.duration_ms <= 0:
            return 0.0
        return self._playback_position_ms / self.duration_ms

    @property
    def remaining_ms(self) -> float:
        """Get remaining playback time in milliseconds."""
        return max(0.0, self.duration_ms - self._playback_position_ms)

    @property
    def is_playing(self) -> bool:
        """Check if the line is currently playing."""
        return self._state == VOLineState.PLAYING

    @property
    def is_completed(self) -> bool:
        """Check if the line has completed playback."""
        return self._state in (VOLineState.COMPLETED, VOLineState.INTERRUPTED)

    @property
    def can_interrupt(self) -> bool:
        """Check if this line can be interrupted based on settings."""
        return self.interruptible and self._state == VOLineState.PLAYING

    def can_be_interrupted_by(self, other_priority: int) -> bool:
        """Check if this line can be interrupted by a line with given priority."""
        if not self.interruptible:
            return False
        return other_priority > self.priority

    def start_playback(self, current_time: float) -> None:
        """Mark the line as started."""
        self._state = VOLineState.PLAYING
        self._playback_position_ms = 0.0
        self._last_played_time = current_time
        self._play_count += 1
        if self.on_start:
            self.on_start(self)

    def update_playback(self, delta_ms: float) -> None:
        """Update playback position."""
        if self._state == VOLineState.PLAYING:
            self._playback_position_ms += delta_ms
            if self._playback_position_ms >= self.duration_ms:
                self.complete_playback(interrupted=False)

    def complete_playback(self, interrupted: bool = False) -> None:
        """Mark the line as completed."""
        self._state = VOLineState.INTERRUPTED if interrupted else VOLineState.COMPLETED
        if self.on_end:
            self.on_end(self, interrupted)

    def pause(self) -> None:
        """Pause playback."""
        if self._state == VOLineState.PLAYING:
            self._state = VOLineState.PAUSED

    def resume(self) -> None:
        """Resume playback."""
        if self._state == VOLineState.PAUSED:
            self._state = VOLineState.PLAYING

    def reset(self) -> None:
        """Reset line to pending state."""
        self._state = VOLineState.PENDING
        self._playback_position_ms = 0.0

    def is_on_cooldown(self, current_time: float, cooldown_ms: float) -> bool:
        """Check if line is on cooldown."""
        if self._last_played_time <= 0:
            return False
        effective_cooldown = self.cooldown_ms if self.cooldown_ms > 0 else cooldown_ms
        return (current_time - self._last_played_time) < effective_cooldown

    def matches_conditions(self, game_state: dict[str, Any]) -> bool:
        """Check if game state matches line conditions."""
        for key, expected in self.conditions.items():
            actual = game_state.get(key)
            if callable(expected):
                if not expected(actual):
                    return False
            elif actual != expected:
                return False
        return True

    def has_tag(self, tag: str) -> bool:
        """Check if line has a specific tag."""
        return tag in self.tags

    def has_all_tags(self, tags: set[str]) -> bool:
        """Check if line has all specified tags."""
        return tags.issubset(self.tags)

    def has_any_tag(self, tags: set[str]) -> bool:
        """Check if line has any of the specified tags."""
        return bool(self.tags.intersection(tags))

    def clone(self) -> VOLine:
        """Create a copy of this line with a new ID."""
        return VOLine(
            line_id=str(uuid.uuid4()),
            audio_asset=self.audio_asset,
            text=self.text,
            speaker_id=self.speaker_id,
            duration_ms=self.duration_ms,
            priority=self.priority,
            interruptible=self.interruptible,
            context_type=self.context_type,
            tags=set(self.tags),
            conditions=dict(self.conditions),
            lip_sync=self.lip_sync,
            subtitle=self.subtitle,
            language=self.language,
            weight=self.weight,
            cooldown_ms=self.cooldown_ms,
            on_start=self.on_start,
            on_end=self.on_end,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize line to dictionary."""
        return {
            "line_id": self.line_id,
            "audio_asset": self.audio_asset,
            "text": self.text,
            "speaker_id": self.speaker_id,
            "duration_ms": self.duration_ms,
            "priority": self.priority,
            "interruptible": self.interruptible,
            "context_type": self.context_type,
            "tags": list(self.tags),
            "conditions": self.conditions,
            "language": self.language,
            "weight": self.weight,
            "cooldown_ms": self.cooldown_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VOLine:
        """Deserialize line from dictionary."""
        tags = data.get("tags", [])
        if isinstance(tags, list):
            tags = set(tags)
        return cls(
            line_id=data.get("line_id", str(uuid.uuid4())),
            audio_asset=data.get("audio_asset", ""),
            text=data.get("text", ""),
            speaker_id=data.get("speaker_id", ""),
            duration_ms=data.get("duration_ms", 0.0),
            priority=data.get("priority", PRIORITY_NORMAL),
            interruptible=data.get("interruptible", True),
            context_type=data.get("context_type", ContextType.BARK.value),
            tags=tags,
            conditions=data.get("conditions", {}),
            language=data.get("language", "en"),
            weight=data.get("weight", 1.0),
            cooldown_ms=data.get("cooldown_ms", 0.0),
        )


def create_vo_line(
    audio_asset: str,
    text: str,
    speaker_id: str = "",
    duration_ms: float = 0.0,
    priority: int = PRIORITY_NORMAL,
    interruptible: bool = True,
    context_type: str = ContextType.BARK.value,
    tags: Optional[set[str]] = None,
    **kwargs: Any,
) -> VOLine:
    """Factory function to create a VOLine with common defaults."""
    return VOLine(
        audio_asset=audio_asset,
        text=text,
        speaker_id=speaker_id,
        duration_ms=duration_ms,
        priority=priority,
        interruptible=interruptible,
        context_type=context_type,
        tags=tags or set(),
        **kwargs,
    )
