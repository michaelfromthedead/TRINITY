"""
Voice Manager

Manages voice allocation, limiting, priority, and stealing for audio playback.
"""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Callable

from .config import (
    MAX_VOICES,
    MAX_INSTANCES_PER_SOUND,
    VoiceState,
    VoiceStealStrategy,
    AudioCategory,
    CATEGORY_VOICE_LIMITS,
    PRIORITY_NORMAL,
    PRIORITY_CRITICAL,
    VOICE_STEAL_FADE_MS,
    VOLUME_COMPARISON_THRESHOLD,
    DISTANCE_COMPARISON_THRESHOLD,
    VIRTUAL_VOICE_PROMOTION_COOLDOWN_MS,
    VIRTUAL_VOICE_FORCE_PROMOTE_GRACE_MS,
)
from .audio_source import AudioSource
from .audio_listener import Vector3
from .virtual_voice import VirtualVoiceTracker


@dataclass
class Voice:
    """
    Represents an active voice in the audio system.
    """
    id: int
    source: Optional[AudioSource] = None
    category: AudioCategory = AudioCategory.SFX
    priority: int = PRIORITY_NORMAL
    volume: float = 1.0
    distance: float = 0.0
    start_time: float = 0.0
    is_virtual: bool = False
    is_active: bool = False

    def __lt__(self, other: 'Voice') -> bool:
        """Compare for heap ordering (lower = more stealable)."""
        # Higher priority = less stealable, so invert for min heap
        if self.priority != other.priority:
            return self.priority < other.priority
        # Quieter = more stealable
        if abs(self.volume - other.volume) > VOLUME_COMPARISON_THRESHOLD:
            return self.volume < other.volume
        # Farther = more stealable
        if abs(self.distance - other.distance) > DISTANCE_COMPARISON_THRESHOLD:
            return self.distance > other.distance
        # Older = more stealable
        return self.start_time < other.start_time


@dataclass
class VoiceAllocationResult:
    """Result of a voice allocation attempt."""
    success: bool
    voice_id: Optional[int] = None
    stolen_from: Optional[AudioSource] = None
    made_virtual: bool = False
    error: Optional[str] = None


class VoiceManager:
    """
    Manages voice allocation with limits, priorities, and stealing.

    Features:
    - Global voice limit
    - Per-category limits
    - Per-sound instance limits
    - Priority-based allocation
    - Multiple stealing strategies
    - Virtual voice tracking
    """

    def __init__(
        self,
        max_voices: int = MAX_VOICES,
        steal_strategy: VoiceStealStrategy = VoiceStealStrategy.LOWEST_PRIORITY,
        enable_virtual_voices: bool = True
    ) -> None:
        """
        Initialize voice manager.

        Args:
            max_voices: Maximum number of simultaneous voices
            steal_strategy: Strategy for voice stealing
            enable_virtual_voices: Whether to track virtual voices
        """
        self._max_voices = max_voices
        self._steal_strategy = steal_strategy
        self._enable_virtual_voices = enable_virtual_voices

        # Virtual voice tracker
        self._virtual_tracker = VirtualVoiceTracker()
        self._last_virtual_promotion_time: float = 0.0

        # Voice pool
        self._voices: dict[int, Voice] = {}
        self._next_voice_id = 1

        # Active tracking
        self._active_voices: set[int] = set()
        self._virtual_voices: set[int] = set()

        # Category tracking
        self._category_counts: dict[AudioCategory, int] = {cat: 0 for cat in AudioCategory}
        self._category_limits: dict[AudioCategory, int] = dict(CATEGORY_VOICE_LIMITS)

        # Sound instance tracking
        self._sound_instances: dict[str, list[int]] = {}  # clip_id -> voice_ids
        self._max_instances_per_sound = MAX_INSTANCES_PER_SOUND

        # Listener position for distance calculations
        self._listener_position = Vector3()

        # Callbacks
        self.on_voice_stolen: Optional[Callable[[AudioSource, AudioSource], None]] = None
        self.on_voice_virtualized: Optional[Callable[[AudioSource], None]] = None

        # Thread safety
        self._lock = threading.RLock()

        # Pre-allocate voice slots
        for i in range(max_voices):
            voice_id = self._get_next_voice_id()
            self._voices[voice_id] = Voice(id=voice_id)

    def _get_next_voice_id(self) -> int:
        """Generate next voice ID."""
        voice_id = self._next_voice_id
        self._next_voice_id += 1
        return voice_id

    @property
    def active_voice_count(self) -> int:
        """Get number of active (non-virtual) voices."""
        with self._lock:
            return len(self._active_voices)

    @property
    def virtual_voice_count(self) -> int:
        """Get number of virtual voices."""
        with self._lock:
            return len(self._virtual_voices)

    @property
    def total_voice_count(self) -> int:
        """Get total number of voices (active + virtual)."""
        return self.active_voice_count + self.virtual_voice_count

    @property
    def available_voices(self) -> int:
        """Get number of available voice slots."""
        with self._lock:
            return self._max_voices - len(self._active_voices)

    def set_listener_position(self, position: Vector3) -> None:
        """
        Update listener position for distance calculations.

        Args:
            position: Listener world position
        """
        with self._lock:
            self._listener_position = position

    def set_category_limit(self, category: AudioCategory, limit: int) -> None:
        """
        Set voice limit for a category.

        Args:
            category: Audio category
            limit: Maximum voices for this category
        """
        with self._lock:
            self._category_limits[category] = max(0, limit)

    def set_steal_strategy(self, strategy: VoiceStealStrategy) -> None:
        """
        Set voice stealing strategy.

        Args:
            strategy: The stealing strategy to use
        """
        with self._lock:
            self._steal_strategy = strategy

    def allocate_voice(self, source: AudioSource) -> VoiceAllocationResult:
        """
        Allocate a voice for an audio source.

        Args:
            source: The audio source requesting a voice

        Returns:
            VoiceAllocationResult with success status and voice ID
        """
        with self._lock:
            # Check per-sound instance limit
            if source.clip:
                clip_id = source.clip.id
                instances = self._sound_instances.get(clip_id, [])
                if len(instances) >= self._max_instances_per_sound:
                    # Try to steal from same sound
                    stolen = self._steal_same_sound_voice(source, instances)
                    if stolen:
                        return stolen
                    return VoiceAllocationResult(
                        success=False,
                        error=f"Instance limit ({self._max_instances_per_sound}) reached for sound"
                    )

            # Check category limit
            category = source.category
            if self._category_counts[category] >= self._category_limits.get(category, MAX_VOICES):
                # Try to steal from same category
                stolen = self._steal_category_voice(source)
                if stolen:
                    return stolen
                return VoiceAllocationResult(
                    success=False,
                    error=f"Category limit reached for {category.name}"
                )

            # Check global limit
            if len(self._active_voices) >= self._max_voices:
                # Try to steal any voice
                stolen = self._steal_voice(source)
                if stolen:
                    return stolen

                # Try to make a voice virtual
                if self._enable_virtual_voices:
                    virtualized = self._virtualize_voice(source)
                    if virtualized:
                        return virtualized

                return VoiceAllocationResult(
                    success=False,
                    error="Global voice limit reached"
                )

            # Allocate from pool
            return self._allocate_from_pool(source)

    def _allocate_from_pool(self, source: AudioSource) -> VoiceAllocationResult:
        """Allocate a voice from the available pool."""
        # Find available voice slot
        for voice_id, voice in self._voices.items():
            if not voice.is_active:
                voice.source = source
                voice.category = source.category
                voice.priority = source.priority
                voice.volume = source.effective_volume
                voice.distance = self._calculate_distance(source)
                voice.start_time = time.time()
                voice.is_virtual = False
                voice.is_active = True

                self._active_voices.add(voice_id)
                self._category_counts[source.category] += 1

                # Track sound instance
                if source.clip:
                    clip_id = source.clip.id
                    if clip_id not in self._sound_instances:
                        self._sound_instances[clip_id] = []
                    self._sound_instances[clip_id].append(voice_id)

                source.assign_voice(voice_id)
                return VoiceAllocationResult(success=True, voice_id=voice_id)

        return VoiceAllocationResult(success=False, error="No available voice slots")

    def _steal_voice(self, requester: AudioSource) -> Optional[VoiceAllocationResult]:
        """Attempt to steal a voice based on strategy."""
        if self._steal_strategy == VoiceStealStrategy.NONE:
            return None

        candidates = self._get_steal_candidates(requester)
        if not candidates:
            return None

        # Sort by stealability (lowest priority first)
        if self._steal_strategy == VoiceStealStrategy.OLDEST:
            candidates.sort(key=lambda v: v.start_time)
        elif self._steal_strategy == VoiceStealStrategy.QUIETEST:
            candidates.sort(key=lambda v: v.volume)
        elif self._steal_strategy == VoiceStealStrategy.FARTHEST:
            candidates.sort(key=lambda v: -v.distance)
        elif self._steal_strategy == VoiceStealStrategy.LOWEST_PRIORITY:
            candidates.sort(key=lambda v: v.priority)

        # Steal from the best candidate
        victim = candidates[0]
        return self._perform_steal(victim, requester)

    def _steal_same_sound_voice(
        self,
        requester: AudioSource,
        voice_ids: list[int]
    ) -> Optional[VoiceAllocationResult]:
        """Steal from instances of the same sound."""
        candidates = []
        for vid in voice_ids:
            voice = self._voices.get(vid)
            if voice and voice.is_active and voice.source:
                if voice.source.priority < requester.priority:
                    candidates.append(voice)
                elif voice.source.priority == requester.priority:
                    candidates.append(voice)

        if not candidates:
            return None

        # Steal oldest instance (lower priority first)
        candidates.sort(key=lambda v: (v.priority, v.start_time))
        return self._perform_steal(candidates[0], requester)

    def _steal_category_voice(self, requester: AudioSource) -> Optional[VoiceAllocationResult]:
        """Steal from same category."""
        candidates = [
            v for v in self._voices.values()
            if v.is_active and v.category == requester.category and v.source
        ]

        if not candidates:
            return None

        candidates.sort(key=lambda v: (v.priority, v.volume, -v.distance, v.start_time))
        victim = candidates[0]

        if victim.priority > requester.priority:
            return None  # Can't steal from higher priority

        return self._perform_steal(victim, requester)

    def _perform_steal(self, victim: Voice, requester: AudioSource) -> VoiceAllocationResult:
        """Perform the voice steal operation."""
        old_source = victim.source

        # Stop the victim
        if old_source:
            old_source.stop(fade_out_ms=VOICE_STEAL_FADE_MS)
            old_source.release_voice()

            # Notify callback
            if self.on_voice_stolen:
                self.on_voice_stolen(old_source, requester)

        # Release voice
        self._release_voice_internal(victim.id)

        # Allocate to requester
        return self._allocate_from_pool(requester)

    def _virtualize_voice(self, requester: AudioSource) -> Optional[VoiceAllocationResult]:
        """Convert a real voice to virtual to make room."""
        # Find lowest priority voice that can be virtualized
        candidates = [
            v for v in self._voices.values()
            if v.is_active and not v.is_virtual and v.source and
            v.priority < requester.priority
        ]

        if not candidates:
            return None

        candidates.sort(key=lambda v: (v.priority, v.volume, -v.distance))
        victim = candidates[0]

        # Capture playback position before virtualizing
        position_samples = victim.source.playback_position if victim.source else 0

        # Make virtual
        if victim.source:
            victim.source.make_virtual()
            if self.on_voice_virtualized:
                self.on_voice_virtualized(victim.source)

        victim.is_virtual = True
        self._active_voices.discard(victim.id)
        self._virtual_voices.add(victim.id)

        # Register with virtual voice tracker
        if victim.source:
            self._virtual_tracker.track_virtualization(
                voice_id=victim.id,
                source=victim.source,
                position_samples=position_samples,
            )

        # Now allocate from pool
        return self._allocate_from_pool(requester)

    def _get_steal_candidates(self, requester: AudioSource) -> list[Voice]:
        """Get voices that can be stolen by requester."""
        candidates = []
        for voice in self._voices.values():
            if not voice.is_active or not voice.source:
                continue

            # Can't steal critical priority unless we're also critical
            if voice.priority >= PRIORITY_CRITICAL and requester.priority < PRIORITY_CRITICAL:
                continue

            # Can't steal higher priority
            if voice.priority > requester.priority:
                continue

            candidates.append(voice)

        return candidates

    def _calculate_distance(self, source: AudioSource) -> float:
        """Calculate distance from listener to source."""
        if not source.is_3d:
            return 0.0
        return self._listener_position.distance_to(source.position)

    def release_voice(self, voice_id: int) -> None:
        """
        Release a voice back to the pool.

        Args:
            voice_id: The voice ID to release
        """
        with self._lock:
            self._release_voice_internal(voice_id)

    def _release_voice_internal(self, voice_id: int) -> None:
        """Internal voice release without lock."""
        voice = self._voices.get(voice_id)
        if not voice:
            return

        # Update category count
        if voice.is_active:
            self._category_counts[voice.category] = max(
                0, self._category_counts[voice.category] - 1
            )

        # Remove from instance tracking
        if voice.source and voice.source.clip:
            clip_id = voice.source.clip.id
            if clip_id in self._sound_instances:
                if voice_id in self._sound_instances[clip_id]:
                    self._sound_instances[clip_id].remove(voice_id)
                if not self._sound_instances[clip_id]:
                    del self._sound_instances[clip_id]

        # Notify virtual voice tracker
        self._virtual_tracker.on_released(voice_id)

        # Clear voice
        if voice.source:
            voice.source.release_voice()
        voice.source = None
        voice.is_active = False
        voice.is_virtual = False

        # Remove from tracking
        self._active_voices.discard(voice_id)
        self._virtual_voices.discard(voice_id)

    def update(self, delta_time: float) -> None:
        """
        Update voice manager state.

        Args:
            delta_time: Time since last update in seconds
        """
        with self._lock:
            # Update voice priorities and distances
            for voice in self._voices.values():
                if voice.is_active and voice.source:
                    voice.priority = voice.source.priority
                    voice.volume = voice.source.effective_volume
                    voice.distance = self._calculate_distance(voice.source)

            # Update virtual voice tracker
            self._virtual_tracker.update(delta_time)

            # Check for completed sources (active only)
            to_release = []
            for voice_id in self._active_voices:
                voice = self._voices.get(voice_id)
                if voice and voice.source:
                    if voice.source.is_stopped:
                        to_release.append(voice_id)

            for voice_id in to_release:
                self._release_voice_internal(voice_id)

            # Try to promote virtual voices to real
            if self._enable_virtual_voices:
                self._try_promote_virtual_voices()

    def _try_promote_virtual_voices(self) -> None:
        """Try to promote virtual voices using urgency-based ordering."""
        # Cooldown to prevent thrashing
        now = time.time()
        since_last = (now - self._last_virtual_promotion_time) * 1000.0
        if since_last < VIRTUAL_VOICE_PROMOTION_COOLDOWN_MS:
            return

        available = self._max_voices - len(self._active_voices)
        if available <= 0 or not self._virtual_voices:
            return

        # Get urgency-ranked candidates from tracker
        candidates = self._virtual_tracker.get_promotion_candidates(available)
        if not candidates:
            return

        promoted = 0
        for candidate in candidates:
            if promoted >= available:
                break

            vid = candidate.voice_id
            voice = self._voices.get(vid)
            if not voice or not voice.source:
                continue

            # Find an available slot
            for new_id, new_voice in self._voices.items():
                if not new_voice.is_active and new_id not in self._virtual_voices:
                    # Restore playback position from tracker
                    tracked_pos = candidate.position_samples
                    voice.source.seek_samples(tracked_pos)

                    # Promote
                    voice.source.make_real(new_id)

                    self._virtual_voices.discard(vid)
                    self._virtual_tracker.on_promoted(vid)

                    # Allocate to new slot
                    new_voice.source = voice.source
                    new_voice.category = voice.category
                    new_voice.priority = voice.priority
                    new_voice.volume = voice.volume
                    new_voice.distance = voice.distance
                    new_voice.start_time = voice.start_time
                    new_voice.is_virtual = False
                    new_voice.is_active = True

                    self._active_voices.add(new_id)
                    self._category_counts[voice.category] += 1

                    # Clear old slot
                    voice.source = None
                    voice.is_active = False
                    voice.is_virtual = False

                    promoted += 1
                    self._last_virtual_promotion_time = now
                    break

    def get_voice(self, voice_id: int) -> Optional[Voice]:
        """
        Get a voice by ID.

        Args:
            voice_id: The voice ID

        Returns:
            Voice or None
        """
        with self._lock:
            return self._voices.get(voice_id)

    def get_active_voices(self) -> list[Voice]:
        """Get all active (non-virtual) voices."""
        with self._lock:
            return [
                self._voices[vid] for vid in self._active_voices
                if self._voices.get(vid)
            ]

    def get_virtual_voices(self) -> list[Voice]:
        """Get all virtual voices."""
        with self._lock:
            return [
                self._voices[vid] for vid in self._virtual_voices
                if self._voices.get(vid)
            ]

    def get_category_voice_count(self, category: AudioCategory) -> int:
        """Get number of voices in a category."""
        with self._lock:
            return self._category_counts.get(category, 0)

    def get_sound_instance_count(self, clip_id: str) -> int:
        """Get number of playing instances of a sound."""
        with self._lock:
            return len(self._sound_instances.get(clip_id, []))

    def stop_all(self, fade_ms: float = 0) -> None:
        """
        Stop all voices.

        Args:
            fade_ms: Fade out duration
        """
        with self._lock:
            for voice_id in list(self._active_voices | self._virtual_voices):
                voice = self._voices.get(voice_id)
                if voice and voice.source:
                    voice.source.stop(fade_ms)
                self._release_voice_internal(voice_id)

    def stop_category(self, category: AudioCategory, fade_ms: float = 0) -> None:
        """
        Stop all voices in a category.

        Args:
            category: Category to stop
            fade_ms: Fade out duration
        """
        with self._lock:
            to_stop = []
            for voice_id in self._active_voices | self._virtual_voices:
                voice = self._voices.get(voice_id)
                if voice and voice.category == category:
                    to_stop.append(voice_id)

            for voice_id in to_stop:
                voice = self._voices.get(voice_id)
                if voice and voice.source:
                    voice.source.stop(fade_ms)
                self._release_voice_internal(voice_id)

    def pause_all(self) -> None:
        """Pause all active voices."""
        with self._lock:
            for voice_id in self._active_voices:
                voice = self._voices.get(voice_id)
                if voice and voice.source:
                    voice.source.pause()

    def resume_all(self) -> None:
        """Resume all paused voices."""
        with self._lock:
            for voice_id in self._active_voices:
                voice = self._voices.get(voice_id)
                if voice and voice.source:
                    voice.source.resume()

    def get_stats(self) -> dict:
        """Get voice manager statistics."""
        with self._lock:
            return {
                'max_voices': self._max_voices,
                'active_voices': len(self._active_voices),
                'virtual_voices': len(self._virtual_voices),
                'available_voices': self._max_voices - len(self._active_voices),
                'category_counts': dict(self._category_counts),
                'sound_instances': {k: len(v) for k, v in self._sound_instances.items()},
                'steal_strategy': self._steal_strategy.name,
        'virtual_tracker': self._virtual_tracker.get_stats(),
            }
