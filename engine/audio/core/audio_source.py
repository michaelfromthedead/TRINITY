"""
Audio Source

Audio source types: one-shot, looping, and streaming with 3D positioning.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from enum import IntEnum, auto

from .config import (
    SourceType,
    VoiceState,
    AudioCategory,
    AttenuationModel,
    DEFAULT_MIN_DISTANCE,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_ROLLOFF_FACTOR,
    DEFAULT_FADE_IN_MS,
    DEFAULT_FADE_OUT_MS,
    PRIORITY_NORMAL,
    PAN_ANGLE_MULTIPLIER,
)
from .audio_clip import AudioClip
from .audio_listener import Vector3


class SourcePlaybackMode(IntEnum):
    """Playback mode for audio source."""
    ONCE = 0
    LOOP = auto()
    PING_PONG = auto()


@dataclass
class FadeState:
    """Fade in/out state tracking."""
    is_fading: bool = False
    fade_in: bool = True
    start_volume: float = 0.0
    target_volume: float = 1.0
    duration_ms: float = 0.0
    elapsed_ms: float = 0.0
    callback: Optional[Callable[[], None]] = None

    def update(self, delta_ms: float) -> float:
        """
        Update fade and return current multiplier.

        Args:
            delta_ms: Time elapsed since last update

        Returns:
            Current volume multiplier (0-1)
        """
        if not self.is_fading:
            return self.target_volume

        self.elapsed_ms += delta_ms

        if self.elapsed_ms >= self.duration_ms:
            self.is_fading = False
            if self.callback:
                self.callback()
            return self.target_volume

        progress = self.elapsed_ms / self.duration_ms
        return self.start_volume + (self.target_volume - self.start_volume) * progress


@dataclass
class AudioSource:
    """
    Represents an audio source that can play clips in 3D space.

    Supports one-shot, looping, and streaming playback modes with
    full 3D positioning and effects.
    """

    # Identity
    id: str = ""
    name: str = ""

    # Clip reference
    clip: Optional[AudioClip] = None

    # Source type
    source_type: SourceType = SourceType.ONE_SHOT

    # Playback mode
    playback_mode: SourcePlaybackMode = SourcePlaybackMode.ONCE

    # Category for voice management
    category: AudioCategory = AudioCategory.SFX

    # Priority (higher = more important)
    priority: int = PRIORITY_NORMAL

    # Volume (0.0 to 1.0)
    _volume: float = 1.0

    # Pitch (1.0 = normal, 2.0 = octave up, 0.5 = octave down)
    _pitch: float = 1.0

    # Pan (-1.0 = full left, 0.0 = center, 1.0 = full right)
    _pan: float = 0.0

    # 3D positioning
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)

    # 3D settings
    is_3d: bool = False
    min_distance: float = DEFAULT_MIN_DISTANCE
    max_distance: float = DEFAULT_MAX_DISTANCE
    rolloff_factor: float = DEFAULT_ROLLOFF_FACTOR
    attenuation_model: AttenuationModel = AttenuationModel.INVERSE_CLAMPED

    # Doppler
    doppler_level: float = 1.0

    # State
    state: VoiceState = VoiceState.STOPPED

    # Playback position (in samples)
    _playback_position: int = 0

    # Time tracking
    _start_time: float = 0.0
    _pause_time: float = 0.0
    _total_paused_time: float = 0.0

    # Fade state
    fade_state: FadeState = field(default_factory=FadeState)

    # Looping
    loop_start: int = 0
    loop_end: int = 0
    loop_count: int = -1  # -1 = infinite
    _loops_played: int = 0

    # Callbacks
    on_complete: Optional[Callable[['AudioSource'], None]] = None
    on_loop: Optional[Callable[['AudioSource', int], None]] = None

    # Voice ID (assigned when playing)
    _voice_id: Optional[int] = None

    # Virtual voice tracking
    is_virtual: bool = False

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Output levels (after processing)
    _output_volume_left: float = 1.0
    _output_volume_right: float = 1.0

    # Mute state
    _muted: bool = False

    def __post_init__(self) -> None:
        """Initialize source after creation."""
        if not self.id:
            self.id = f"source_{id(self)}"

    def __hash__(self) -> int:
        """Hash based on unique id for set/dict usage."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on id."""
        if not isinstance(other, AudioSource):
            return NotImplemented
        return self.id == other.id

    @property
    def volume(self) -> float:
        """Get source volume."""
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        """Set source volume (clamped 0-1)."""
        self._volume = max(0.0, min(1.0, value))

    @property
    def pitch(self) -> float:
        """Get source pitch."""
        return self._pitch

    @pitch.setter
    def pitch(self, value: float) -> None:
        """Set source pitch (clamped 0.1-4.0)."""
        self._pitch = max(0.1, min(4.0, value))

    @property
    def pan(self) -> float:
        """Get source pan."""
        return self._pan

    @pan.setter
    def pan(self, value: float) -> None:
        """Set source pan (clamped -1 to 1)."""
        self._pan = max(-1.0, min(1.0, value))

    @property
    def muted(self) -> bool:
        """Check if source is muted."""
        return self._muted

    @muted.setter
    def muted(self, value: bool) -> None:
        """Set mute state."""
        self._muted = value

    @property
    def is_playing(self) -> bool:
        """Check if source is currently playing."""
        return self.state == VoiceState.PLAYING

    @property
    def is_paused(self) -> bool:
        """Check if source is paused."""
        return self.state == VoiceState.PAUSED

    @property
    def is_stopped(self) -> bool:
        """Check if source is stopped."""
        return self.state == VoiceState.STOPPED

    @property
    def is_looping(self) -> bool:
        """Check if source is set to loop."""
        return self.playback_mode == SourcePlaybackMode.LOOP or self.loop_count != 0

    @property
    def playback_position(self) -> int:
        """Get current playback position in samples."""
        return self._playback_position

    @property
    def playback_time(self) -> float:
        """Get current playback time in seconds."""
        if not self.clip:
            return 0.0
        return self._playback_position / self.clip.sample_rate

    @property
    def duration(self) -> float:
        """Get clip duration in seconds."""
        if not self.clip:
            return 0.0
        return self.clip.duration

    @property
    def effective_volume(self) -> float:
        """Get effective volume after fade and mute."""
        if self._muted:
            return 0.0
        fade_mult = self.fade_state.target_volume if not self.fade_state.is_fading else \
            self.fade_state.update(0)
        return self._volume * fade_mult

    @property
    def voice_id(self) -> Optional[int]:
        """Get assigned voice ID."""
        return self._voice_id

    def set_clip(self, clip: AudioClip) -> None:
        """
        Set the audio clip for this source.

        Args:
            clip: The audio clip to use
        """
        with self._lock:
            # Release old clip
            if self.clip:
                self.clip.release_ref()

            self.clip = clip
            if clip:
                clip.add_ref()

                # Set up loop points from clip if available
                if clip.metadata.has_loop_points:
                    self.loop_start = clip.metadata.loop_start
                    self.loop_end = clip.metadata.loop_end

            self._playback_position = 0
            self._loops_played = 0

    def play(self, fade_in_ms: float = 0) -> bool:
        """
        Start playback.

        Args:
            fade_in_ms: Fade in duration in milliseconds

        Returns:
            True if playback started
        """
        with self._lock:
            if not self.clip or not self.clip.is_loaded:
                return False

            if self.state == VoiceState.PAUSED:
                # Resume from pause
                self._total_paused_time += time.time() - self._pause_time
            else:
                # Start fresh
                self._playback_position = 0
                self._loops_played = 0
                self._start_time = time.time()
                self._total_paused_time = 0.0

            self.state = VoiceState.PLAYING
            self.is_virtual = False

            # Set up fade in
            if fade_in_ms > 0:
                self.fade_state = FadeState(
                    is_fading=True,
                    fade_in=True,
                    start_volume=0.0,
                    target_volume=1.0,
                    duration_ms=fade_in_ms,
                    elapsed_ms=0.0
                )
            else:
                self.fade_state = FadeState(target_volume=1.0)

            return True

    def pause(self) -> None:
        """Pause playback."""
        with self._lock:
            if self.state == VoiceState.PLAYING:
                self.state = VoiceState.PAUSED
                self._pause_time = time.time()

    def resume(self) -> None:
        """Resume from pause."""
        with self._lock:
            if self.state == VoiceState.PAUSED:
                self._total_paused_time += time.time() - self._pause_time
                self.state = VoiceState.PLAYING

    def stop(self, fade_out_ms: float = 0) -> None:
        """
        Stop playback.

        Args:
            fade_out_ms: Fade out duration in milliseconds
        """
        with self._lock:
            if self.state == VoiceState.STOPPED:
                return

            if fade_out_ms > 0 and self.state == VoiceState.PLAYING:
                self.state = VoiceState.STOPPING
                self.fade_state = FadeState(
                    is_fading=True,
                    fade_in=False,
                    start_volume=self.effective_volume,
                    target_volume=0.0,
                    duration_ms=fade_out_ms,
                    elapsed_ms=0.0,
                    callback=self._complete_stop
                )
            else:
                self._complete_stop()

    def _complete_stop(self) -> None:
        """Complete the stop operation."""
        self.state = VoiceState.STOPPED
        self._playback_position = 0
        self._voice_id = None
        self.is_virtual = False

        if self.on_complete:
            self.on_complete(self)

    def seek(self, position_seconds: float) -> None:
        """
        Seek to a position.

        Args:
            position_seconds: Target position in seconds
        """
        with self._lock:
            if not self.clip:
                return

            sample_position = int(position_seconds * self.clip.sample_rate)
            self._playback_position = max(0, min(sample_position, self.clip.metadata.total_samples))

    def seek_samples(self, sample_position: int) -> None:
        """
        Seek to a sample position.

        Args:
            sample_position: Target position in samples
        """
        with self._lock:
            if not self.clip:
                return

            self._playback_position = max(0, min(sample_position, self.clip.metadata.total_samples))

    def set_position_3d(self, x: float, y: float, z: float) -> None:
        """
        Set 3D position.

        Args:
            x, y, z: World position
        """
        with self._lock:
            self.position = Vector3(x, y, z)

    def set_velocity_3d(self, x: float, y: float, z: float) -> None:
        """
        Set 3D velocity for Doppler.

        Args:
            x, y, z: Velocity vector
        """
        with self._lock:
            self.velocity = Vector3(x, y, z)

    def set_distance_settings(
        self,
        min_distance: float,
        max_distance: float,
        rolloff: float = DEFAULT_ROLLOFF_FACTOR
    ) -> None:
        """
        Set distance attenuation settings.

        Args:
            min_distance: Distance at which sound is at full volume
            max_distance: Distance at which sound is inaudible
            rolloff: Rolloff factor
        """
        with self._lock:
            self.min_distance = max(0.001, min_distance)
            self.max_distance = max(min_distance, max_distance)
            self.rolloff_factor = max(0.0, rolloff)

    def update(self, delta_ms: float) -> None:
        """
        Update source state.

        Args:
            delta_ms: Time elapsed since last update in milliseconds
        """
        with self._lock:
            if self.state not in (VoiceState.PLAYING, VoiceState.STOPPING, VoiceState.VIRTUAL):
                return

            # Update fade (skip for virtual voices - tracker handles advancement)
            if self.state != VoiceState.VIRTUAL and self.fade_state.is_fading:
                self.fade_state.update(delta_ms)

            # Update playback position (based on pitch-adjusted time)
            if self.clip:
                samples_per_ms = (self.clip.sample_rate / 1000.0) * self._pitch
                samples_advanced = int(delta_ms * samples_per_ms)
                self._advance_playback(samples_advanced)

    def _advance_playback(self, samples: int) -> None:
        """Advance playback position handling loops."""
        if not self.clip:
            return

        total_samples = self.clip.metadata.total_samples
        loop_end = self.loop_end if self.loop_end > 0 else total_samples
        loop_start = self.loop_start

        self._playback_position += samples

        if self._playback_position >= loop_end:
            if self.is_looping and (self.loop_count < 0 or self._loops_played < self.loop_count):
                # Loop back
                overflow = self._playback_position - loop_end
                loop_length = loop_end - loop_start
                if loop_length > 0:
                    self._playback_position = loop_start + (overflow % loop_length)
                else:
                    # Invalid loop points - just restart from beginning
                    self._playback_position = loop_start
                self._loops_played += 1

                if self.on_loop:
                    self.on_loop(self, self._loops_played)
            else:
                # End of playback
                self._complete_stop()

    def get_samples(self, num_samples: int) -> Optional[bytes]:
        """
        Get audio samples for mixing.

        Args:
            num_samples: Number of samples to retrieve

        Returns:
            Audio sample data or None
        """
        with self._lock:
            if not self.clip or not self.is_playing:
                return None

            return self.clip.get_samples(self._playback_position, num_samples)

    def calculate_output_volumes(
        self,
        attenuation: float = 1.0,
        spatial_pan: float = 0.0,
        doppler: float = 1.0
    ) -> tuple[float, float]:
        """
        Calculate final output volumes for left/right channels.

        Args:
            attenuation: Distance attenuation (0-1)
            spatial_pan: Spatial panning (-1 to 1)
            doppler: Doppler pitch factor

        Returns:
            Tuple of (left_volume, right_volume)
        """
        base_volume = self.effective_volume * attenuation

        # Combine source pan with spatial pan
        total_pan = max(-1.0, min(1.0, self._pan + spatial_pan))

        # Equal power panning
        import math
        angle = (total_pan + 1.0) * PAN_ANGLE_MULTIPLIER * math.pi  # 0 to pi/2
        left = base_volume * math.cos(angle)
        right = base_volume * math.sin(angle)

        self._output_volume_left = left
        self._output_volume_right = right

        return (left, right)

    def make_virtual(self) -> None:
        """Convert to virtual voice (tracked but not rendered)."""
        with self._lock:
            if self.state == VoiceState.PLAYING:
                self.is_virtual = True
                self.state = VoiceState.VIRTUAL
                self._voice_id = None

    def make_real(self, voice_id: int) -> None:
        """
        Convert from virtual to real voice.

        Args:
            voice_id: Assigned voice ID
        """
        with self._lock:
            if self.state == VoiceState.VIRTUAL:
                self.is_virtual = False
                self.state = VoiceState.PLAYING
                self._voice_id = voice_id

    def assign_voice(self, voice_id: int) -> None:
        """
        Assign a voice ID to this source.

        Args:
            voice_id: The voice ID
        """
        self._voice_id = voice_id

    def release_voice(self) -> None:
        """Release the assigned voice."""
        self._voice_id = None

    def reset(self) -> None:
        """Reset source to initial state."""
        with self._lock:
            self.state = VoiceState.STOPPED
            self._playback_position = 0
            self._loops_played = 0
            self._voice_id = None
            self.is_virtual = False
            self.fade_state = FadeState()

    def clone(self) -> 'AudioSource':
        """
        Create a copy of this source.

        Returns:
            New AudioSource with same settings
        """
        with self._lock:
            new_source = AudioSource(
                id="",
                name=f"{self.name}_clone",
                clip=self.clip,
                source_type=self.source_type,
                playback_mode=self.playback_mode,
                category=self.category,
                priority=self.priority,
                _volume=self._volume,
                _pitch=self._pitch,
                _pan=self._pan,
                position=Vector3(self.position.x, self.position.y, self.position.z),
                velocity=Vector3(self.velocity.x, self.velocity.y, self.velocity.z),
                is_3d=self.is_3d,
                min_distance=self.min_distance,
                max_distance=self.max_distance,
                rolloff_factor=self.rolloff_factor,
                attenuation_model=self.attenuation_model,
                doppler_level=self.doppler_level,
                loop_start=self.loop_start,
                loop_end=self.loop_end,
                loop_count=self.loop_count,
            )

            if self.clip:
                self.clip.add_ref()

            return new_source

    def __del__(self) -> None:
        """Clean up on destruction."""
        if self.clip:
            self.clip.release_ref()


class AudioSourcePool:
    """
    Pool of reusable audio sources to reduce allocations.
    """

    def __init__(self, initial_size: int = 32, max_size: int = 128) -> None:
        """
        Initialize source pool.

        Args:
            initial_size: Initial pool size
            max_size: Maximum pool size
        """
        self._available: list[AudioSource] = []
        self._in_use: set[AudioSource] = set()
        self._max_size = max_size
        self._lock = threading.Lock()

        # Pre-allocate
        for _ in range(initial_size):
            self._available.append(AudioSource())

    def acquire(self) -> Optional[AudioSource]:
        """
        Acquire a source from the pool.

        Returns:
            An AudioSource or None if pool is exhausted
        """
        with self._lock:
            if self._available:
                source = self._available.pop()
                source.reset()
                self._in_use.add(source)
                return source

            # Create new if under limit
            if len(self._in_use) < self._max_size:
                source = AudioSource()
                self._in_use.add(source)
                return source

            return None

    def release(self, source: AudioSource) -> None:
        """
        Return a source to the pool.

        Args:
            source: The source to release
        """
        with self._lock:
            if source in self._in_use:
                self._in_use.remove(source)
                source.reset()
                self._available.append(source)

    @property
    def available_count(self) -> int:
        """Get number of available sources."""
        with self._lock:
            return len(self._available)

    @property
    def in_use_count(self) -> int:
        """Get number of sources in use."""
        with self._lock:
            return len(self._in_use)

    def clear(self) -> None:
        """Clear all sources."""
        with self._lock:
            self._available.clear()
            self._in_use.clear()
