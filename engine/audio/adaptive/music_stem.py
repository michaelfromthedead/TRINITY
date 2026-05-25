"""
Layered music system with stems (instrument groups).

Provides dynamic control over individual music layers for adaptive audio.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, List, Callable, Any
import threading
import time
import math

from .config import (
    MAX_STEMS,
    STEM_FADE_TIME,
    STEM_VOLUME_SMOOTHING,
    STEM_CROSSFADE_CURVE,
    DEFAULT_VOLUME,
    MIN_VOLUME,
    MAX_VOLUME,
    LAYER_DRUMS,
    LAYER_BASS,
    LAYER_MELODY,
    LAYER_PADS,
    LAYER_STRINGS,
    LAYER_PERCUSSION,
    DEFAULT_LAYERS,
    FADE_CURVE_LINEAR,
    FADE_CURVE_EQUAL_POWER,
    FADE_CURVE_S_CURVE,
    FADE_CURVE_EXPONENTIAL,
    EXPONENTIAL_CURVE_FACTOR,
)


class StemState(Enum):
    """State of a music stem."""
    INACTIVE = auto()
    ACTIVE = auto()
    FADING_IN = auto()
    FADING_OUT = auto()
    MUTED = auto()


@dataclass
class StemInfo:
    """Information about a music stem (layer).

    Attributes:
        stem_id: Unique identifier for the stem
        name: Display name (e.g., "drums", "bass")
        layer_type: Type of layer (from config)
        path: Audio file path or resource ID
        volume: Base volume (0.0-1.0)
        pan: Stereo pan (-1.0 to 1.0)
        priority: Priority for resource management
        metadata: Additional metadata
    """
    stem_id: str
    name: str
    layer_type: str
    path: str
    volume: float = DEFAULT_VOLUME
    pan: float = 0.0
    priority: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class StemPlaybackState:
    """Runtime state of a playing stem.

    Attributes:
        stem_info: Static stem information
        state: Current playback state
        current_volume: Current effective volume
        target_volume: Target volume for fading
        fade_start_volume: Volume at fade start
        fade_start_time: Time fade started
        fade_duration: Duration of fade
        is_muted: Whether stem is muted
        solo: Whether stem is soloed
    """
    stem_info: StemInfo
    state: StemState = StemState.INACTIVE
    current_volume: float = 0.0
    target_volume: float = 1.0
    fade_start_volume: float = 0.0
    fade_start_time: float = 0.0
    fade_duration: float = STEM_FADE_TIME
    is_muted: bool = False
    solo: bool = False


class FadeCurve:
    """Fade curve calculations for smooth volume transitions."""

    @staticmethod
    def linear(t: float) -> float:
        """Linear fade curve.

        Args:
            t: Progress (0.0-1.0)

        Returns:
            Volume multiplier (0.0-1.0)
        """
        return max(0.0, min(1.0, t))

    @staticmethod
    def equal_power(t: float) -> float:
        """Equal power (cosine) fade curve.

        Maintains constant perceived loudness during crossfades.

        Args:
            t: Progress (0.0-1.0)

        Returns:
            Volume multiplier (0.0-1.0)
        """
        t = max(0.0, min(1.0, t))
        return math.sin(t * math.pi / 2)

    @staticmethod
    def s_curve(t: float) -> float:
        """S-curve (smoothstep) fade.

        Smooth start and end, fast middle.

        Args:
            t: Progress (0.0-1.0)

        Returns:
            Volume multiplier (0.0-1.0)
        """
        t = max(0.0, min(1.0, t))
        return t * t * (3 - 2 * t)

    @staticmethod
    def exponential(t: float) -> float:
        """Exponential fade curve.

        Args:
            t: Progress (0.0-1.0)

        Returns:
            Volume multiplier (0.0-1.0)
        """
        t = max(0.0, min(1.0, t))
        factor = EXPONENTIAL_CURVE_FACTOR
        return (math.exp(t * factor) - 1) / (math.exp(factor) - 1)

    @staticmethod
    def get_curve(curve_type: str) -> Callable[[float], float]:
        """Get fade curve function by name.

        Args:
            curve_type: Curve type name

        Returns:
            Curve function
        """
        curves = {
            FADE_CURVE_LINEAR: FadeCurve.linear,
            FADE_CURVE_EQUAL_POWER: FadeCurve.equal_power,
            FADE_CURVE_S_CURVE: FadeCurve.s_curve,
            FADE_CURVE_EXPONENTIAL: FadeCurve.exponential,
        }
        return curves.get(curve_type, FadeCurve.linear)


class MusicStem:
    """Manager for a single music stem/layer.

    Handles volume, fading, muting for one audio layer.
    """

    def __init__(
        self,
        stem_info: StemInfo,
        fade_curve: str = STEM_CROSSFADE_CURVE,
    ):
        """Initialize music stem.

        Args:
            stem_info: Static stem information
            fade_curve: Fade curve type to use
        """
        self._info = stem_info
        self._state = StemPlaybackState(stem_info=stem_info)
        self._fade_curve = FadeCurve.get_curve(fade_curve)
        self._lock = threading.RLock()

    @property
    def stem_id(self) -> str:
        """Get stem ID."""
        return self._info.stem_id

    @property
    def name(self) -> str:
        """Get stem name."""
        return self._info.name

    @property
    def layer_type(self) -> str:
        """Get layer type."""
        return self._info.layer_type

    @property
    def info(self) -> StemInfo:
        """Get stem info."""
        return self._info

    @property
    def current_state(self) -> StemState:
        """Get current state."""
        return self._state.state

    @property
    def volume(self) -> float:
        """Get current effective volume."""
        with self._lock:
            if self._state.is_muted:
                return 0.0
            return self._state.current_volume * self._info.volume

    @property
    def is_active(self) -> bool:
        """Check if stem is active (not inactive or muted)."""
        return self._state.state in (StemState.ACTIVE, StemState.FADING_IN)

    @property
    def is_muted(self) -> bool:
        """Check if stem is muted."""
        return self._state.is_muted

    @property
    def is_solo(self) -> bool:
        """Check if stem is soloed."""
        return self._state.solo

    def activate(self, fade_time: float = STEM_FADE_TIME):
        """Activate the stem with fade in.

        Args:
            fade_time: Fade duration in seconds
        """
        with self._lock:
            if self._state.state == StemState.ACTIVE:
                return

            self._state.fade_start_volume = self._state.current_volume
            self._state.target_volume = 1.0
            self._state.fade_start_time = time.perf_counter()
            self._state.fade_duration = fade_time

            if fade_time <= 0:
                self._state.current_volume = 1.0
                self._state.state = StemState.ACTIVE
            else:
                self._state.state = StemState.FADING_IN

    def deactivate(self, fade_time: float = STEM_FADE_TIME):
        """Deactivate the stem with fade out.

        Args:
            fade_time: Fade duration in seconds
        """
        with self._lock:
            if self._state.state == StemState.INACTIVE:
                return

            self._state.fade_start_volume = self._state.current_volume
            self._state.target_volume = 0.0
            self._state.fade_start_time = time.perf_counter()
            self._state.fade_duration = fade_time

            if fade_time <= 0:
                self._state.current_volume = 0.0
                self._state.state = StemState.INACTIVE
            else:
                self._state.state = StemState.FADING_OUT

    def set_volume(self, volume: float, fade_time: float = 0.0):
        """Set stem volume.

        Args:
            volume: Target volume (0.0-1.0)
            fade_time: Fade duration in seconds
        """
        if volume < MIN_VOLUME or volume > MAX_VOLUME:
            raise ValueError(f"Volume must be between {MIN_VOLUME} and {MAX_VOLUME}")

        with self._lock:
            if fade_time <= 0:
                self._state.current_volume = volume
                self._state.target_volume = volume
            else:
                self._state.fade_start_volume = self._state.current_volume
                self._state.target_volume = volume
                self._state.fade_start_time = time.perf_counter()
                self._state.fade_duration = fade_time

                if volume > self._state.current_volume:
                    self._state.state = StemState.FADING_IN
                else:
                    self._state.state = StemState.FADING_OUT

    def mute(self):
        """Mute the stem."""
        with self._lock:
            self._state.is_muted = True
            self._state.state = StemState.MUTED

    def unmute(self):
        """Unmute the stem."""
        with self._lock:
            self._state.is_muted = False
            if self._state.current_volume > 0:
                self._state.state = StemState.ACTIVE
            else:
                self._state.state = StemState.INACTIVE

    def set_solo(self, solo: bool):
        """Set solo state.

        Args:
            solo: Whether to solo this stem
        """
        with self._lock:
            self._state.solo = solo

    def update(self, delta_time: float = None):
        """Update stem state (process fades).

        Args:
            delta_time: Time since last update (not used, uses wall time)
        """
        with self._lock:
            if self._state.state not in (StemState.FADING_IN, StemState.FADING_OUT):
                return

            elapsed = time.perf_counter() - self._state.fade_start_time
            progress = min(1.0, elapsed / self._state.fade_duration)
            curved_progress = self._fade_curve(progress)

            # Interpolate volume
            self._state.current_volume = (
                self._state.fade_start_volume +
                (self._state.target_volume - self._state.fade_start_volume) * curved_progress
            )

            # Check if fade complete
            if progress >= 1.0:
                self._state.current_volume = self._state.target_volume
                if self._state.target_volume > 0:
                    self._state.state = StemState.ACTIVE
                else:
                    self._state.state = StemState.INACTIVE

    def get_state_snapshot(self) -> StemPlaybackState:
        """Get a snapshot of current state.

        Returns:
            Copy of current state
        """
        with self._lock:
            return StemPlaybackState(
                stem_info=self._state.stem_info,
                state=self._state.state,
                current_volume=self._state.current_volume,
                target_volume=self._state.target_volume,
                fade_start_volume=self._state.fade_start_volume,
                fade_start_time=self._state.fade_start_time,
                fade_duration=self._state.fade_duration,
                is_muted=self._state.is_muted,
                solo=self._state.solo,
            )


class StemGroup:
    """A group of related stems (e.g., all percussion stems).

    Allows controlling multiple stems together.
    """

    def __init__(self, name: str):
        """Initialize stem group.

        Args:
            name: Group name
        """
        self.name = name
        self._stems: Dict[str, MusicStem] = {}
        self._group_volume = DEFAULT_VOLUME
        self._group_muted = False
        self._lock = threading.RLock()

    def add_stem(self, stem: MusicStem):
        """Add a stem to the group.

        Args:
            stem: Stem to add
        """
        with self._lock:
            self._stems[stem.stem_id] = stem

    def remove_stem(self, stem_id: str) -> Optional[MusicStem]:
        """Remove a stem from the group.

        Args:
            stem_id: ID of stem to remove

        Returns:
            Removed stem or None
        """
        with self._lock:
            return self._stems.pop(stem_id, None)

    def get_stem(self, stem_id: str) -> Optional[MusicStem]:
        """Get a stem by ID.

        Args:
            stem_id: Stem ID

        Returns:
            Stem or None
        """
        return self._stems.get(stem_id)

    def set_group_volume(self, volume: float, fade_time: float = STEM_FADE_TIME):
        """Set volume for all stems in group.

        Args:
            volume: Target volume
            fade_time: Fade duration
        """
        with self._lock:
            self._group_volume = volume
            for stem in self._stems.values():
                stem.set_volume(volume, fade_time)

    def mute_group(self):
        """Mute all stems in group."""
        with self._lock:
            self._group_muted = True
            for stem in self._stems.values():
                stem.mute()

    def unmute_group(self):
        """Unmute all stems in group."""
        with self._lock:
            self._group_muted = False
            for stem in self._stems.values():
                stem.unmute()

    def activate_all(self, fade_time: float = STEM_FADE_TIME):
        """Activate all stems in group.

        Args:
            fade_time: Fade duration
        """
        with self._lock:
            for stem in self._stems.values():
                stem.activate(fade_time)

    def deactivate_all(self, fade_time: float = STEM_FADE_TIME):
        """Deactivate all stems in group.

        Args:
            fade_time: Fade duration
        """
        with self._lock:
            for stem in self._stems.values():
                stem.deactivate(fade_time)

    def update(self, delta_time: float = None):
        """Update all stems in group.

        Args:
            delta_time: Time since last update
        """
        with self._lock:
            for stem in self._stems.values():
                stem.update(delta_time)

    @property
    def stems(self) -> List[MusicStem]:
        """Get all stems in group."""
        with self._lock:
            return list(self._stems.values())

    @property
    def stem_count(self) -> int:
        """Get number of stems in group."""
        return len(self._stems)


class LayeredMusicPlayer:
    """Music player with stem/layer support.

    Manages multiple synchronized audio stems for adaptive music.
    """

    def __init__(
        self,
        max_stems: int = MAX_STEMS,
        fade_curve: str = STEM_CROSSFADE_CURVE,
    ):
        """Initialize layered music player.

        Args:
            max_stems: Maximum number of stems
            fade_curve: Default fade curve type
        """
        self._max_stems = max_stems
        self._fade_curve = fade_curve
        self._stems: Dict[str, MusicStem] = {}
        self._groups: Dict[str, StemGroup] = {}
        self._master_volume = DEFAULT_VOLUME
        self._lock = threading.RLock()

        # Solo management
        self._has_solo = False

        # Update thread
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def master_volume(self) -> float:
        """Get master volume."""
        return self._master_volume

    @master_volume.setter
    def master_volume(self, value: float):
        """Set master volume."""
        if value < MIN_VOLUME or value > MAX_VOLUME:
            raise ValueError(f"Volume must be between {MIN_VOLUME} and {MAX_VOLUME}")
        self._master_volume = value

    @property
    def stem_count(self) -> int:
        """Get current stem count."""
        return len(self._stems)

    def add_stem(self, stem_info: StemInfo) -> MusicStem:
        """Add a new stem.

        Args:
            stem_info: Stem information

        Returns:
            Created MusicStem

        Raises:
            ValueError: If max stems reached or ID already exists
        """
        with self._lock:
            if len(self._stems) >= self._max_stems:
                raise ValueError(f"Maximum stems ({self._max_stems}) reached")

            if stem_info.stem_id in self._stems:
                raise ValueError(f"Stem ID already exists: {stem_info.stem_id}")

            stem = MusicStem(stem_info, self._fade_curve)
            self._stems[stem_info.stem_id] = stem
            return stem

    def remove_stem(self, stem_id: str) -> bool:
        """Remove a stem.

        Args:
            stem_id: ID of stem to remove

        Returns:
            True if stem was found and removed
        """
        with self._lock:
            if stem_id in self._stems:
                # Remove from any groups
                for group in self._groups.values():
                    group.remove_stem(stem_id)
                del self._stems[stem_id]
                return True
            return False

    def get_stem(self, stem_id: str) -> Optional[MusicStem]:
        """Get a stem by ID.

        Args:
            stem_id: Stem ID

        Returns:
            MusicStem or None
        """
        return self._stems.get(stem_id)

    def get_stem_by_type(self, layer_type: str) -> List[MusicStem]:
        """Get all stems of a layer type.

        Args:
            layer_type: Layer type (e.g., "drums")

        Returns:
            List of matching stems
        """
        with self._lock:
            return [
                stem for stem in self._stems.values()
                if stem.layer_type == layer_type
            ]

    def create_group(self, name: str) -> StemGroup:
        """Create a new stem group.

        Args:
            name: Group name

        Returns:
            Created StemGroup
        """
        with self._lock:
            if name in self._groups:
                return self._groups[name]
            group = StemGroup(name)
            self._groups[name] = group
            return group

    def get_group(self, name: str) -> Optional[StemGroup]:
        """Get a group by name.

        Args:
            name: Group name

        Returns:
            StemGroup or None
        """
        return self._groups.get(name)

    def add_stem_to_group(self, stem_id: str, group_name: str) -> bool:
        """Add a stem to a group.

        Args:
            stem_id: Stem ID
            group_name: Group name

        Returns:
            True if successful
        """
        with self._lock:
            stem = self._stems.get(stem_id)
            group = self._groups.get(group_name)
            if stem is not None and group is not None:
                group.add_stem(stem)
                return True
            return False

    def activate_layer(self, layer_type: str, fade_time: float = STEM_FADE_TIME):
        """Activate all stems of a layer type.

        Args:
            layer_type: Layer type to activate
            fade_time: Fade duration
        """
        for stem in self.get_stem_by_type(layer_type):
            stem.activate(fade_time)

    def deactivate_layer(self, layer_type: str, fade_time: float = STEM_FADE_TIME):
        """Deactivate all stems of a layer type.

        Args:
            layer_type: Layer type to deactivate
            fade_time: Fade duration
        """
        for stem in self.get_stem_by_type(layer_type):
            stem.deactivate(fade_time)

    def set_layer_volume(
        self,
        layer_type: str,
        volume: float,
        fade_time: float = STEM_FADE_TIME,
    ):
        """Set volume for all stems of a layer type.

        Args:
            layer_type: Layer type
            volume: Target volume
            fade_time: Fade duration
        """
        for stem in self.get_stem_by_type(layer_type):
            stem.set_volume(volume, fade_time)

    def mute_layer(self, layer_type: str):
        """Mute all stems of a layer type.

        Args:
            layer_type: Layer type to mute
        """
        for stem in self.get_stem_by_type(layer_type):
            stem.mute()

    def unmute_layer(self, layer_type: str):
        """Unmute all stems of a layer type.

        Args:
            layer_type: Layer type to unmute
        """
        for stem in self.get_stem_by_type(layer_type):
            stem.unmute()

    def solo_stem(self, stem_id: str):
        """Solo a stem (mute all others).

        Args:
            stem_id: ID of stem to solo
        """
        with self._lock:
            stem = self._stems.get(stem_id)
            if stem is None:
                return

            self._has_solo = True
            stem.set_solo(True)

    def unsolo_stem(self, stem_id: str):
        """Remove solo from a stem.

        Args:
            stem_id: ID of stem to unsolo
        """
        with self._lock:
            stem = self._stems.get(stem_id)
            if stem is not None:
                stem.set_solo(False)

            # Check if any stems still soloed
            self._has_solo = any(s.is_solo for s in self._stems.values())

    def clear_solo(self):
        """Clear all solos."""
        with self._lock:
            for stem in self._stems.values():
                stem.set_solo(False)
            self._has_solo = False

    def get_effective_volume(self, stem_id: str) -> float:
        """Get effective volume for a stem considering master and solo.

        Args:
            stem_id: Stem ID

        Returns:
            Effective volume (0.0-1.0)
        """
        with self._lock:
            stem = self._stems.get(stem_id)
            if stem is None:
                return 0.0

            # If any stem is soloed and this isn't it, return 0
            if self._has_solo and not stem.is_solo:
                return 0.0

            return stem.volume * self._master_volume

    def activate_stems_by_intensity(
        self,
        intensity: float,
        layer_order: List[str] = None,
        fade_time: float = STEM_FADE_TIME,
    ):
        """Activate stems based on intensity level.

        Higher intensity activates more layers.

        Args:
            intensity: Intensity level (0.0-1.0)
            layer_order: Order of layers by intensity (low to high)
            fade_time: Fade duration
        """
        if layer_order is None:
            layer_order = list(DEFAULT_LAYERS)

        intensity = max(0.0, min(1.0, intensity))
        num_layers = int(len(layer_order) * intensity)

        with self._lock:
            for i, layer_type in enumerate(layer_order):
                if i < num_layers:
                    self.activate_layer(layer_type, fade_time)
                else:
                    self.deactivate_layer(layer_type, fade_time)

    def set_blend(self, blend_map: Dict[str, float], fade_time: float = STEM_FADE_TIME):
        """Set volume blend for multiple stems.

        Args:
            blend_map: Dictionary mapping stem_id or layer_type to volume
            fade_time: Fade duration
        """
        with self._lock:
            for key, volume in blend_map.items():
                # Check if it's a stem ID
                stem = self._stems.get(key)
                if stem is not None:
                    stem.set_volume(volume, fade_time)
                else:
                    # Treat as layer type
                    self.set_layer_volume(key, volume, fade_time)

    def update(self, delta_time: float = None):
        """Update all stems.

        Args:
            delta_time: Time since last update
        """
        with self._lock:
            for stem in self._stems.values():
                stem.update(delta_time)
            for group in self._groups.values():
                group.update(delta_time)

    def get_all_volumes(self) -> Dict[str, float]:
        """Get effective volumes for all stems.

        Returns:
            Dictionary mapping stem_id to effective volume
        """
        with self._lock:
            return {
                stem_id: self.get_effective_volume(stem_id)
                for stem_id in self._stems
            }

    def get_active_stems(self) -> List[MusicStem]:
        """Get all active (non-silent) stems.

        Returns:
            List of active stems
        """
        with self._lock:
            return [stem for stem in self._stems.values() if stem.is_active]

    def start_update_loop(self, interval_ms: float = 10.0):
        """Start the update loop thread.

        Args:
            interval_ms: Update interval in milliseconds
        """
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        def update_loop():
            while not self._stop_event.is_set():
                self.update()
                time.sleep(interval_ms / 1000.0)

        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()

    def stop_update_loop(self):
        """Stop the update loop thread."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._update_thread is not None:
            self._update_thread.join(timeout=1.0)
            self._update_thread = None

    def clear(self):
        """Remove all stems and groups."""
        with self._lock:
            self._stems.clear()
            self._groups.clear()
            self._has_solo = False
