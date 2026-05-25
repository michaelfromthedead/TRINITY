"""
Sound Cue System

Sound cues with variation support: random, sequence, switch, shuffle.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import IntEnum, auto

from .config import (
    SoundCueType,
    AudioCategory,
    PITCH_VARIATION_RANGE,
    VOLUME_VARIATION_DB,
    MAX_START_OFFSET_RATIO,
    PRIORITY_NORMAL,
    MINIMUM_DB_VALUE,
)
from .audio_clip import AudioClip
from .audio_source import AudioSource, SourceType


def db_to_linear(db: float) -> float:
    """Convert decibels to linear scale."""
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear scale to decibels."""
    import math
    if linear <= 0:
        return MINIMUM_DB_VALUE
    return 20.0 * math.log10(linear)


@dataclass
class SoundEntry:
    """An entry in a sound cue with its clip and settings."""
    clip: AudioClip
    weight: float = 1.0  # For random selection
    volume_offset_db: float = 0.0
    pitch_offset: float = 0.0
    delay_ms: float = 0.0
    condition: Optional[str] = None  # For switch cues

    def __hash__(self) -> int:
        return hash(self.clip.id)


@dataclass
class CueVariation:
    """Variation settings for sound cues."""
    # Pitch variation
    enable_pitch_variation: bool = False
    pitch_variation_range: float = PITCH_VARIATION_RANGE  # ±10%
    pitch_min: float = 0.9
    pitch_max: float = 1.1

    # Volume variation
    enable_volume_variation: bool = False
    volume_variation_db: float = VOLUME_VARIATION_DB  # ±3dB
    volume_min_db: float = -3.0
    volume_max_db: float = 3.0

    # Start offset variation
    enable_start_offset: bool = False
    max_start_offset_ratio: float = MAX_START_OFFSET_RATIO  # Max 10%

    # Round robin tracking (for random/shuffle)
    avoid_repeats: bool = True
    repeat_avoid_count: int = 2  # Avoid last N played

    def apply_pitch_variation(self, base_pitch: float) -> float:
        """Apply pitch variation to a base pitch."""
        if not self.enable_pitch_variation:
            return base_pitch

        variation = random.uniform(-self.pitch_variation_range, self.pitch_variation_range)
        return base_pitch * (1.0 + variation)

    def apply_volume_variation(self, base_volume: float) -> float:
        """Apply volume variation to a base volume."""
        if not self.enable_volume_variation:
            return base_volume

        variation_db = random.uniform(-self.volume_variation_db, self.volume_variation_db)
        return base_volume * db_to_linear(variation_db)

    def get_start_offset(self, clip_duration: float) -> float:
        """Get random start offset in seconds."""
        if not self.enable_start_offset or clip_duration <= 0:
            return 0.0

        max_offset = clip_duration * self.max_start_offset_ratio
        return random.uniform(0, max_offset)


@dataclass
class SoundCue:
    """
    A sound cue that can contain multiple clips with variation.

    Supports:
    - Simple: Single clip
    - Random: Weighted random selection
    - Sequence: Ordered playback
    - Switch: Parameter-based selection
    - Shuffle: Random without immediate repeats
    """

    id: str
    name: str
    cue_type: SoundCueType = SoundCueType.SIMPLE

    # Entries
    entries: List[SoundEntry] = field(default_factory=list)

    # Base settings
    category: AudioCategory = AudioCategory.SFX
    base_volume: float = 1.0
    base_pitch: float = 1.0
    priority: int = PRIORITY_NORMAL

    # 3D settings
    is_3d: bool = False
    min_distance: float = 1.0
    max_distance: float = 100.0

    # Variation
    variation: CueVariation = field(default_factory=CueVariation)

    # Playback settings
    looping: bool = False
    loop_count: int = -1  # -1 = infinite

    # Sequence tracking
    _sequence_index: int = 0

    # Shuffle tracking
    _shuffle_history: List[int] = field(default_factory=list)
    _shuffle_pool: List[int] = field(default_factory=list)

    # Switch parameter
    _switch_parameter: Optional[str] = None

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_entry(
        self,
        clip: AudioClip,
        weight: float = 1.0,
        volume_offset_db: float = 0.0,
        pitch_offset: float = 0.0,
        delay_ms: float = 0.0,
        condition: Optional[str] = None
    ) -> None:
        """
        Add an entry to the cue.

        Args:
            clip: The audio clip
            weight: Weight for random selection
            volume_offset_db: Volume offset in dB
            pitch_offset: Pitch offset
            delay_ms: Delay before playback
            condition: Condition for switch cues
        """
        entry = SoundEntry(
            clip=clip,
            weight=weight,
            volume_offset_db=volume_offset_db,
            pitch_offset=pitch_offset,
            delay_ms=delay_ms,
            condition=condition
        )
        self.entries.append(entry)

        # Reset shuffle pool when entries change
        self._shuffle_pool = list(range(len(self.entries)))
        random.shuffle(self._shuffle_pool)

    def remove_entry(self, clip: AudioClip) -> bool:
        """
        Remove an entry by clip.

        Args:
            clip: The clip to remove

        Returns:
            True if removed
        """
        for entry in self.entries:
            if entry.clip.id == clip.id:
                self.entries.remove(entry)
                self._shuffle_pool = list(range(len(self.entries)))
                return True
        return False

    def set_switch_parameter(self, parameter: str) -> None:
        """
        Set the switch parameter for switch cues.

        Args:
            parameter: The condition parameter to match
        """
        with self._lock:
            self._switch_parameter = parameter

    def select_entry(self) -> Optional[SoundEntry]:
        """
        Select the next entry based on cue type.

        Returns:
            Selected SoundEntry or None
        """
        if not self.entries:
            return None

        with self._lock:
            if self.cue_type == SoundCueType.SIMPLE:
                return self._select_simple()
            elif self.cue_type == SoundCueType.RANDOM:
                return self._select_random()
            elif self.cue_type == SoundCueType.SEQUENCE:
                return self._select_sequence()
            elif self.cue_type == SoundCueType.SWITCH:
                return self._select_switch()
            elif self.cue_type == SoundCueType.SHUFFLE:
                return self._select_shuffle()

            return self.entries[0]

    def _select_simple(self) -> Optional[SoundEntry]:
        """Select for simple cue (first entry)."""
        return self.entries[0] if self.entries else None

    def _select_random(self) -> Optional[SoundEntry]:
        """Select weighted random entry."""
        if not self.entries:
            return None

        total_weight = sum(e.weight for e in self.entries)
        if total_weight <= 0:
            return random.choice(self.entries)

        # Weighted selection avoiding recent entries
        available = self.entries.copy()

        if self.variation.avoid_repeats and len(available) > self.variation.repeat_avoid_count:
            for idx in self._shuffle_history[-self.variation.repeat_avoid_count:]:
                if 0 <= idx < len(self.entries) and self.entries[idx] in available:
                    available.remove(self.entries[idx])

        total_weight = sum(e.weight for e in available)
        if total_weight <= 0:
            return random.choice(available)

        r = random.uniform(0, total_weight)
        cumulative = 0.0
        for i, entry in enumerate(available):
            cumulative += entry.weight
            if r <= cumulative:
                # Track for repeat avoidance
                idx = self.entries.index(entry)
                self._shuffle_history.append(idx)
                if len(self._shuffle_history) > self.variation.repeat_avoid_count * 2:
                    self._shuffle_history = self._shuffle_history[-self.variation.repeat_avoid_count:]
                return entry

        return available[-1]

    def _select_sequence(self) -> Optional[SoundEntry]:
        """Select next in sequence."""
        if not self.entries:
            return None

        entry = self.entries[self._sequence_index]
        self._sequence_index = (self._sequence_index + 1) % len(self.entries)
        return entry

    def _select_switch(self) -> Optional[SoundEntry]:
        """Select based on switch parameter."""
        if not self.entries:
            return None

        if self._switch_parameter:
            for entry in self.entries:
                if entry.condition == self._switch_parameter:
                    return entry

        # Default to first entry
        return self.entries[0]

    def _select_shuffle(self) -> Optional[SoundEntry]:
        """Select from shuffle pool without immediate repeats."""
        if not self.entries:
            return None

        # Refill pool if empty
        if not self._shuffle_pool:
            self._shuffle_pool = list(range(len(self.entries)))
            random.shuffle(self._shuffle_pool)

            # Avoid repeat of last played
            if self._shuffle_history and self._shuffle_pool:
                last = self._shuffle_history[-1]
                if self._shuffle_pool[0] == last and len(self._shuffle_pool) > 1:
                    # Swap with random position
                    swap_idx = random.randint(1, len(self._shuffle_pool) - 1)
                    self._shuffle_pool[0], self._shuffle_pool[swap_idx] = \
                        self._shuffle_pool[swap_idx], self._shuffle_pool[0]

        idx = self._shuffle_pool.pop(0)
        self._shuffle_history.append(idx)
        if len(self._shuffle_history) > len(self.entries):
            self._shuffle_history = self._shuffle_history[-len(self.entries):]

        return self.entries[idx]

    def create_source(self) -> Optional[AudioSource]:
        """
        Create an audio source configured for this cue.

        Returns:
            Configured AudioSource or None
        """
        entry = self.select_entry()
        if not entry:
            return None

        source = AudioSource(
            id="",
            name=f"{self.name}_instance",
            clip=entry.clip,
            source_type=SourceType.LOOPING if self.looping else SourceType.ONE_SHOT,
            category=self.category,
            priority=self.priority,
            is_3d=self.is_3d,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
            loop_count=self.loop_count,
        )

        # Apply base settings
        volume = self.base_volume * db_to_linear(entry.volume_offset_db)
        pitch = self.base_pitch + entry.pitch_offset

        # Apply variation
        volume = self.variation.apply_volume_variation(volume)
        pitch = self.variation.apply_pitch_variation(pitch)

        source.volume = volume
        source.pitch = pitch

        # Apply start offset
        if self.variation.enable_start_offset and entry.clip.duration > 0:
            offset = self.variation.get_start_offset(entry.clip.duration)
            source.seek(offset)

        return source

    def reset_sequence(self) -> None:
        """Reset sequence to beginning."""
        with self._lock:
            self._sequence_index = 0

    def reset_shuffle(self) -> None:
        """Reset shuffle pool."""
        with self._lock:
            self._shuffle_pool = list(range(len(self.entries)))
            random.shuffle(self._shuffle_pool)
            self._shuffle_history.clear()

    @property
    def entry_count(self) -> int:
        """Get number of entries."""
        return len(self.entries)

    @property
    def total_weight(self) -> float:
        """Get total weight of all entries."""
        return sum(e.weight for e in self.entries)


class SoundCueManager:
    """
    Manages sound cues and their instances.
    """

    def __init__(self) -> None:
        """Initialize cue manager."""
        self._cues: Dict[str, SoundCue] = {}
        self._active_instances: Dict[str, List[AudioSource]] = {}
        self._lock = threading.RLock()

    def register_cue(self, cue: SoundCue) -> None:
        """
        Register a sound cue.

        Args:
            cue: The sound cue to register
        """
        with self._lock:
            self._cues[cue.id] = cue

    def unregister_cue(self, cue_id: str) -> bool:
        """
        Unregister a sound cue.

        Args:
            cue_id: The cue ID to unregister

        Returns:
            True if unregistered
        """
        with self._lock:
            if cue_id in self._cues:
                del self._cues[cue_id]
                return True
            return False

    def get_cue(self, cue_id: str) -> Optional[SoundCue]:
        """
        Get a cue by ID.

        Args:
            cue_id: The cue ID

        Returns:
            The SoundCue or None
        """
        with self._lock:
            return self._cues.get(cue_id)

    def play_cue(self, cue_id: str) -> Optional[AudioSource]:
        """
        Play a sound cue.

        Args:
            cue_id: The cue ID to play

        Returns:
            The created AudioSource or None
        """
        with self._lock:
            cue = self._cues.get(cue_id)
            if not cue:
                return None

            source = cue.create_source()
            if not source:
                return None

            # Track instance
            if cue_id not in self._active_instances:
                self._active_instances[cue_id] = []
            self._active_instances[cue_id].append(source)

            return source

    def stop_cue(self, cue_id: str, fade_ms: float = 0) -> None:
        """
        Stop all instances of a cue.

        Args:
            cue_id: The cue ID to stop
            fade_ms: Fade out duration
        """
        with self._lock:
            instances = self._active_instances.get(cue_id, [])
            for source in instances:
                source.stop(fade_ms)
            self._active_instances[cue_id] = []

    def stop_all(self, fade_ms: float = 0) -> None:
        """Stop all cue instances."""
        with self._lock:
            for cue_id in list(self._active_instances.keys()):
                self.stop_cue(cue_id, fade_ms)

    def update(self, delta_time: float) -> None:
        """
        Update cue manager and clean up stopped instances.

        Args:
            delta_time: Time since last update
        """
        with self._lock:
            for cue_id, instances in list(self._active_instances.items()):
                # Remove stopped instances
                self._active_instances[cue_id] = [
                    s for s in instances if not s.is_stopped
                ]

    def get_active_instance_count(self, cue_id: str) -> int:
        """Get number of active instances of a cue."""
        with self._lock:
            return len(self._active_instances.get(cue_id, []))

    def get_all_cues(self) -> List[SoundCue]:
        """Get all registered cues."""
        with self._lock:
            return list(self._cues.values())


@dataclass
class SoundCueBuilder:
    """
    Builder pattern for creating sound cues.
    """

    _cue: SoundCue = field(default_factory=lambda: SoundCue(id="", name=""))

    def with_id(self, id: str) -> 'SoundCueBuilder':
        """Set cue ID."""
        self._cue.id = id
        return self

    def with_name(self, name: str) -> 'SoundCueBuilder':
        """Set cue name."""
        self._cue.name = name
        return self

    def with_type(self, cue_type: SoundCueType) -> 'SoundCueBuilder':
        """Set cue type."""
        self._cue.cue_type = cue_type
        return self

    def with_category(self, category: AudioCategory) -> 'SoundCueBuilder':
        """Set audio category."""
        self._cue.category = category
        return self

    def with_volume(self, volume: float) -> 'SoundCueBuilder':
        """Set base volume."""
        self._cue.base_volume = volume
        return self

    def with_pitch(self, pitch: float) -> 'SoundCueBuilder':
        """Set base pitch."""
        self._cue.base_pitch = pitch
        return self

    def with_priority(self, priority: int) -> 'SoundCueBuilder':
        """Set priority."""
        self._cue.priority = priority
        return self

    def with_3d(
        self,
        min_distance: float = 1.0,
        max_distance: float = 100.0
    ) -> 'SoundCueBuilder':
        """Enable 3D audio."""
        self._cue.is_3d = True
        self._cue.min_distance = min_distance
        self._cue.max_distance = max_distance
        return self

    def with_looping(self, loop_count: int = -1) -> 'SoundCueBuilder':
        """Enable looping."""
        self._cue.looping = True
        self._cue.loop_count = loop_count
        return self

    def with_pitch_variation(
        self,
        range: float = PITCH_VARIATION_RANGE
    ) -> 'SoundCueBuilder':
        """Enable pitch variation."""
        self._cue.variation.enable_pitch_variation = True
        self._cue.variation.pitch_variation_range = range
        return self

    def with_volume_variation(
        self,
        range_db: float = VOLUME_VARIATION_DB
    ) -> 'SoundCueBuilder':
        """Enable volume variation."""
        self._cue.variation.enable_volume_variation = True
        self._cue.variation.volume_variation_db = range_db
        return self

    def with_start_offset(
        self,
        max_ratio: float = MAX_START_OFFSET_RATIO
    ) -> 'SoundCueBuilder':
        """Enable start offset variation."""
        self._cue.variation.enable_start_offset = True
        self._cue.variation.max_start_offset_ratio = max_ratio
        return self

    def with_avoid_repeats(
        self,
        count: int = 2
    ) -> 'SoundCueBuilder':
        """Enable repeat avoidance."""
        self._cue.variation.avoid_repeats = True
        self._cue.variation.repeat_avoid_count = count
        return self

    def add_clip(
        self,
        clip: AudioClip,
        weight: float = 1.0,
        volume_offset_db: float = 0.0,
        pitch_offset: float = 0.0,
        condition: Optional[str] = None
    ) -> 'SoundCueBuilder':
        """Add a clip entry."""
        self._cue.add_entry(clip, weight, volume_offset_db, pitch_offset, 0.0, condition)
        return self

    def build(self) -> SoundCue:
        """Build the sound cue."""
        return self._cue
