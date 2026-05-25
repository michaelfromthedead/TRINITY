"""
Adaptive music system with horizontal re-sequencing and vertical layering.

Provides gameplay-driven music that responds to player actions and game state.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, List, Callable, Any, Tuple
import threading
import random
import time

from .config import (
    INTENSITY_SMOOTHING,
    INTENSITY_MIN,
    INTENSITY_MAX,
    DANGER_MIN,
    DANGER_MAX,
    DANGER_THRESHOLD_LOW,
    DANGER_THRESHOLD_HIGH,
    VERTICAL_THRESHOLD_LOW,
    VERTICAL_THRESHOLD_MED,
    VERTICAL_THRESHOLD_HIGH,
    HORIZONTAL_BRANCH_PROBABILITY,
    HORIZONTAL_MIN_SECTION_LENGTH,
    HORIZONTAL_MAX_SECTION_LENGTH,
    PARAM_INTENSITY,
    PARAM_DANGER,
    PARAM_TENSION,
    PARAM_ENERGY,
    DEFAULT_PARAMETERS,
    STEM_FADE_TIME,
    DEFAULT_LAYERS,
)
from .music_timing import MusicClock, BeatGrid
from .music_stem import LayeredMusicPlayer, StemInfo
from .music_callback import MusicCallbackManager, CallbackEvent
from .music_state import MusicStateManager


class AdaptiveMode(Enum):
    """Mode of adaptive music behavior."""
    NONE = auto()
    VERTICAL = auto()    # Layer-based intensity
    HORIZONTAL = auto()  # Section-based branching
    COMBINED = auto()    # Both vertical and horizontal


class BranchType(Enum):
    """Type of horizontal branching."""
    SEQUENTIAL = auto()   # Play sections in order
    RANDOM = auto()       # Random section selection
    RULE_BASED = auto()   # Based on game rules
    WEIGHTED = auto()     # Weighted random selection


@dataclass
class MusicSection:
    """A section of music for horizontal re-sequencing.

    Attributes:
        section_id: Unique identifier
        name: Display name
        start_bar: Starting bar number
        end_bar: Ending bar number
        can_loop: Whether section can loop
        loop_count: Number of times to loop (0 = infinite)
        next_sections: Valid next sections
        weights: Weights for next section selection
        intensity_range: Valid intensity range for this section
        tags: Tags for filtering
    """
    section_id: str
    name: str
    start_bar: int
    end_bar: int
    can_loop: bool = True
    loop_count: int = 0
    next_sections: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    intensity_range: Tuple[float, float] = (0.0, 1.0)
    tags: frozenset[str] = field(default_factory=frozenset)

    @property
    def length_bars(self) -> int:
        """Get section length in bars."""
        return self.end_bar - self.start_bar


@dataclass
class IntensityLevel:
    """Configuration for an intensity level.

    Attributes:
        level_id: Unique identifier
        threshold: Minimum intensity to activate
        layers: Layer configuration (layer_type -> volume)
        sections: Valid sections for this level
        name: Display name
    """
    level_id: str
    threshold: float
    layers: Dict[str, float]
    sections: List[str] = field(default_factory=list)
    name: str = ""


@dataclass
class AdaptiveParameters:
    """Current adaptive music parameters.

    Attributes:
        intensity: Overall intensity (0.0-1.0)
        danger: Danger level (0.0-1.0)
        tension: Tension level (0.0-1.0)
        energy: Energy level (0.0-1.0)
        custom: Custom parameters
    """
    intensity: float = 0.5
    danger: float = 0.0
    tension: float = 0.0
    energy: float = 0.5
    custom: Dict[str, float] = field(default_factory=dict)

    def get(self, name: str, default: float = 0.0) -> float:
        """Get parameter value."""
        if name == PARAM_INTENSITY:
            return self.intensity
        elif name == PARAM_DANGER:
            return self.danger
        elif name == PARAM_TENSION:
            return self.tension
        elif name == PARAM_ENERGY:
            return self.energy
        return self.custom.get(name, default)

    def set(self, name: str, value: float):
        """Set parameter value."""
        value = max(0.0, min(1.0, value))
        if name == PARAM_INTENSITY:
            self.intensity = value
        elif name == PARAM_DANGER:
            self.danger = value
        elif name == PARAM_TENSION:
            self.tension = value
        elif name == PARAM_ENERGY:
            self.energy = value
        else:
            self.custom[name] = value


class VerticalRemixer:
    """Handles vertical (layer-based) adaptive music.

    Controls which layers are active based on intensity and parameters.
    """

    def __init__(
        self,
        stem_player: LayeredMusicPlayer,
        smoothing: float = INTENSITY_SMOOTHING,
    ):
        """Initialize vertical remixer.

        Args:
            stem_player: Layered music player for stem control
            smoothing: Smoothing factor for parameter changes
        """
        self._stem_player = stem_player
        self._smoothing = smoothing
        self._intensity_levels: Dict[str, IntensityLevel] = {}
        self._current_level: Optional[IntensityLevel] = None
        self._target_intensity = 0.5
        self._current_intensity = 0.5
        self._lock = threading.RLock()

        # Default intensity levels
        self._setup_default_levels()

    def _setup_default_levels(self):
        """Set up default intensity levels."""
        self.add_intensity_level(IntensityLevel(
            level_id="low",
            threshold=0.0,
            layers={"pads": 1.0, "melody": 0.3, "bass": 0.2, "drums": 0.0},
            name="Low Intensity",
        ))
        self.add_intensity_level(IntensityLevel(
            level_id="medium",
            threshold=VERTICAL_THRESHOLD_LOW,
            layers={"pads": 0.8, "melody": 0.7, "bass": 0.6, "drums": 0.4},
            name="Medium Intensity",
        ))
        self.add_intensity_level(IntensityLevel(
            level_id="high",
            threshold=VERTICAL_THRESHOLD_MED,
            layers={"pads": 0.5, "melody": 1.0, "bass": 0.9, "drums": 0.8},
            name="High Intensity",
        ))
        self.add_intensity_level(IntensityLevel(
            level_id="maximum",
            threshold=VERTICAL_THRESHOLD_HIGH,
            layers={"pads": 0.3, "melody": 1.0, "bass": 1.0, "drums": 1.0},
            name="Maximum Intensity",
        ))

    def add_intensity_level(self, level: IntensityLevel):
        """Add an intensity level.

        Args:
            level: Intensity level configuration
        """
        with self._lock:
            self._intensity_levels[level.level_id] = level

    def remove_intensity_level(self, level_id: str) -> bool:
        """Remove an intensity level.

        Args:
            level_id: Level to remove

        Returns:
            True if level was found and removed
        """
        with self._lock:
            if level_id in self._intensity_levels:
                del self._intensity_levels[level_id]
                return True
            return False

    def set_intensity(self, intensity: float, immediate: bool = False):
        """Set target intensity.

        Args:
            intensity: Target intensity (0.0-1.0)
            immediate: Whether to apply immediately without smoothing
        """
        intensity = max(INTENSITY_MIN, min(INTENSITY_MAX, intensity))
        with self._lock:
            self._target_intensity = intensity
            if immediate:
                self._current_intensity = intensity
                self._apply_intensity()

    def get_intensity(self) -> float:
        """Get current intensity.

        Returns:
            Current intensity value
        """
        return self._current_intensity

    def _get_level_for_intensity(self, intensity: float) -> Optional[IntensityLevel]:
        """Get the appropriate level for an intensity value.

        Args:
            intensity: Intensity value

        Returns:
            Matching IntensityLevel
        """
        sorted_levels = sorted(
            self._intensity_levels.values(),
            key=lambda l: l.threshold,
            reverse=True,
        )
        for level in sorted_levels:
            if intensity >= level.threshold:
                return level
        return sorted_levels[-1] if sorted_levels else None

    def _apply_intensity(self):
        """Apply current intensity to stem player."""
        level = self._get_level_for_intensity(self._current_intensity)
        if level is None:
            return

        if level != self._current_level:
            self._current_level = level
            self._stem_player.set_blend(level.layers, STEM_FADE_TIME)

    def update(self, delta_time: float = 0.016):
        """Update vertical remixer.

        Args:
            delta_time: Time since last update in seconds
        """
        from .config import INTENSITY_SMOOTHING_RATE
        with self._lock:
            # Smooth intensity changes
            if self._current_intensity != self._target_intensity:
                diff = self._target_intensity - self._current_intensity
                # Use configurable smoothing rate instead of magic number
                self._current_intensity += diff * self._smoothing * delta_time * INTENSITY_SMOOTHING_RATE
                if abs(diff) < 0.01:
                    self._current_intensity = self._target_intensity
                self._apply_intensity()


class HorizontalSequencer:
    """Handles horizontal (section-based) adaptive music.

    Controls which sections play and in what order.
    """

    def __init__(
        self,
        clock: MusicClock,
        callback_manager: MusicCallbackManager,
    ):
        """Initialize horizontal sequencer.

        Args:
            clock: Music clock for timing
            callback_manager: Callback manager for bar events
        """
        self._clock = clock
        self._callback_manager = callback_manager
        self._sections: Dict[str, MusicSection] = {}
        self._current_section: Optional[MusicSection] = None
        self._next_section: Optional[MusicSection] = None
        self._section_play_count = 0
        self._branch_type = BranchType.WEIGHTED
        self._lock = threading.RLock()

        # Callbacks
        self._on_section_change: Optional[Callable[[MusicSection, MusicSection], None]] = None

        # Register bar callback
        self._bar_callback_id = callback_manager.register_bar_callback(
            self._on_bar,
        )

    def add_section(self, section: MusicSection):
        """Add a music section.

        Args:
            section: Section to add
        """
        with self._lock:
            self._sections[section.section_id] = section

    def remove_section(self, section_id: str) -> bool:
        """Remove a section.

        Args:
            section_id: Section to remove

        Returns:
            True if section was found and removed
        """
        with self._lock:
            if section_id in self._sections:
                del self._sections[section_id]
                return True
            return False

    def get_section(self, section_id: str) -> Optional[MusicSection]:
        """Get a section by ID.

        Args:
            section_id: Section ID

        Returns:
            MusicSection or None
        """
        return self._sections.get(section_id)

    def set_branch_type(self, branch_type: BranchType):
        """Set branching behavior.

        Args:
            branch_type: Type of branching to use
        """
        self._branch_type = branch_type

    def start_section(self, section_id: str):
        """Start playing a specific section.

        Args:
            section_id: Section to start
        """
        with self._lock:
            section = self._sections.get(section_id)
            if section is not None:
                old_section = self._current_section
                self._current_section = section
                self._section_play_count = 0
                self._clock.seek_to_bar(section.start_bar)

                if self._on_section_change is not None and old_section is not section:
                    self._on_section_change(old_section, section)

    def queue_next_section(self, section_id: str):
        """Queue a section to play next.

        Args:
            section_id: Section to queue
        """
        with self._lock:
            section = self._sections.get(section_id)
            if section is not None:
                self._next_section = section

    def _on_bar(self, event: CallbackEvent, user_data: Any):
        """Handle bar event for section management."""
        with self._lock:
            if self._current_section is None:
                return

            current_bar = event.bar

            # Check if we've reached the end of current section
            if current_bar >= self._current_section.end_bar:
                self._handle_section_end()

    def _handle_section_end(self):
        """Handle section ending."""
        self._section_play_count += 1

        # Check for queued next section
        if self._next_section is not None:
            old_section = self._current_section
            self._current_section = self._next_section
            self._next_section = None
            self._section_play_count = 0

            if self._on_section_change is not None:
                self._on_section_change(old_section, self._current_section)
            return

        # Check for looping
        if (self._current_section.can_loop and
            (self._current_section.loop_count == 0 or
             self._section_play_count < self._current_section.loop_count)):
            # Stay in current section
            self._clock.seek_to_bar(self._current_section.start_bar)
            return

        # Determine next section
        next_section = self._choose_next_section()
        if next_section is not None:
            old_section = self._current_section
            self._current_section = next_section
            self._section_play_count = 0

            if self._on_section_change is not None:
                self._on_section_change(old_section, self._current_section)

    def _choose_next_section(self) -> Optional[MusicSection]:
        """Choose the next section based on branch type.

        Returns:
            Next section to play
        """
        if self._current_section is None or not self._current_section.next_sections:
            return None

        valid_sections = [
            self._sections.get(sid)
            for sid in self._current_section.next_sections
            if sid in self._sections
        ]
        valid_sections = [s for s in valid_sections if s is not None]

        if not valid_sections:
            return None

        if self._branch_type == BranchType.SEQUENTIAL:
            return valid_sections[0]
        elif self._branch_type == BranchType.RANDOM:
            return random.choice(valid_sections)
        elif self._branch_type == BranchType.WEIGHTED:
            weights = [
                self._current_section.weights.get(s.section_id, 1.0)
                for s in valid_sections
            ]
            return random.choices(valid_sections, weights=weights)[0]
        else:
            return valid_sections[0]

    def set_on_section_change(
        self,
        callback: Optional[Callable[[MusicSection, MusicSection], None]],
    ):
        """Set callback for section changes.

        Args:
            callback: Function to call (old_section, new_section)
        """
        self._on_section_change = callback

    @property
    def current_section(self) -> Optional[MusicSection]:
        """Get current section."""
        return self._current_section

    def clear(self):
        """Remove all sections."""
        with self._lock:
            self._sections.clear()
            self._current_section = None
            self._next_section = None


class AdaptiveMusicSystem:
    """Complete adaptive music system combining all components.

    Integrates vertical layering, horizontal sequencing, and state management.
    """

    def __init__(
        self,
        clock: MusicClock,
        stem_player: LayeredMusicPlayer,
        callback_manager: MusicCallbackManager,
        state_manager: Optional[MusicStateManager] = None,
    ):
        """Initialize adaptive music system.

        Args:
            clock: Music clock for timing
            stem_player: Layered music player
            callback_manager: Callback manager
            state_manager: Optional state manager
        """
        self._clock = clock
        self._stem_player = stem_player
        self._callback_manager = callback_manager
        self._state_manager = state_manager

        # Create sub-systems
        self._vertical_remixer = VerticalRemixer(stem_player)
        self._horizontal_sequencer = HorizontalSequencer(clock, callback_manager)

        # Parameters
        self._parameters = AdaptiveParameters()
        self._mode = AdaptiveMode.COMBINED
        self._lock = threading.RLock()

        # Parameter rules
        self._parameter_rules: List[Callable[[AdaptiveParameters], None]] = []

        # Update thread
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def mode(self) -> AdaptiveMode:
        """Get current adaptive mode."""
        return self._mode

    @mode.setter
    def mode(self, value: AdaptiveMode):
        """Set adaptive mode."""
        self._mode = value

    @property
    def parameters(self) -> AdaptiveParameters:
        """Get current parameters."""
        return self._parameters

    @property
    def vertical_remixer(self) -> VerticalRemixer:
        """Get vertical remixer."""
        return self._vertical_remixer

    @property
    def horizontal_sequencer(self) -> HorizontalSequencer:
        """Get horizontal sequencer."""
        return self._horizontal_sequencer

    def set_parameter(
        self,
        name: str,
        value: float,
        immediate: bool = False,
    ):
        """Set a music parameter.

        Args:
            name: Parameter name
            value: Parameter value (0.0-1.0)
            immediate: Whether to apply immediately
        """
        with self._lock:
            self._parameters.set(name, value)

            # Apply to vertical remixer if intensity
            if name == PARAM_INTENSITY:
                self._vertical_remixer.set_intensity(value, immediate)

            # Run parameter rules
            for rule in self._parameter_rules:
                rule(self._parameters)

    def get_parameter(self, name: str) -> float:
        """Get a parameter value.

        Args:
            name: Parameter name

        Returns:
            Parameter value
        """
        return self._parameters.get(name)

    def add_parameter_rule(self, rule: Callable[[AdaptiveParameters], None]):
        """Add a parameter processing rule.

        Rules are called when parameters change and can trigger
        state changes or other adaptive behavior.

        Args:
            rule: Rule function that receives current parameters
        """
        self._parameter_rules.append(rule)

    def trigger_combat(self):
        """Convenience method to trigger combat music."""
        self.set_parameter(PARAM_INTENSITY, 0.9)
        self.set_parameter(PARAM_DANGER, 0.8)
        if self._state_manager is not None:
            from .config import STATE_COMBAT
            self._state_manager.change_state(STATE_COMBAT)

    def trigger_exploration(self):
        """Convenience method to trigger exploration music."""
        self.set_parameter(PARAM_INTENSITY, 0.3)
        self.set_parameter(PARAM_DANGER, 0.1)
        if self._state_manager is not None:
            from .config import STATE_EXPLORATION
            self._state_manager.change_state(STATE_EXPLORATION)

    def trigger_stealth(self):
        """Convenience method to trigger stealth music."""
        self.set_parameter(PARAM_INTENSITY, 0.4)
        self.set_parameter(PARAM_TENSION, 0.7)
        if self._state_manager is not None:
            from .config import STATE_STEALTH
            self._state_manager.change_state(STATE_STEALTH)

    def increase_intensity(self, amount: float = 0.1):
        """Increase intensity by an amount.

        Args:
            amount: Amount to increase (default 0.1)
        """
        current = self.get_parameter(PARAM_INTENSITY)
        self.set_parameter(PARAM_INTENSITY, current + amount)

    def decrease_intensity(self, amount: float = 0.1):
        """Decrease intensity by an amount.

        Args:
            amount: Amount to decrease (default 0.1)
        """
        current = self.get_parameter(PARAM_INTENSITY)
        self.set_parameter(PARAM_INTENSITY, current - amount)

    def update(self, delta_time: float = 0.016):
        """Update adaptive music system.

        Args:
            delta_time: Time since last update in seconds
        """
        with self._lock:
            if self._mode in (AdaptiveMode.VERTICAL, AdaptiveMode.COMBINED):
                self._vertical_remixer.update(delta_time)

            # Horizontal sequencing is callback-driven, no update needed

            # Update stem player
            self._stem_player.update(delta_time)

    def start_update_loop(self, interval_ms: float = 16.0):
        """Start the update loop thread.

        Args:
            interval_ms: Update interval in milliseconds
        """
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        def update_loop():
            last_time = time.perf_counter()
            while not self._stop_event.is_set():
                current_time = time.perf_counter()
                delta_time = current_time - last_time
                last_time = current_time

                self.update(delta_time)
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
