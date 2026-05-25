"""
State-driven music system for gameplay states.

Provides music management based on game states like exploration,
combat, stealth, victory, and defeat.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, List, Callable, Any, Set
import threading
import time

from .config import (
    STATE_EXPLORATION,
    STATE_COMBAT,
    STATE_STEALTH,
    STATE_VICTORY,
    STATE_DEFEAT,
    STATE_BOSS,
    STATE_MENU,
    STATE_CUTSCENE,
    STATE_AMBIENT,
    STATE_TENSION,
    DEFAULT_STATE,
    VALID_STATES,
    STATE_PRIORITY,
    STATE_TRANSITION_TIME,
    CROSSFADE_DEFAULT_DURATION,
    TRANSITION_CROSSFADE,
    TRANSITION_BAR_SYNC,
    DANGER_THRESHOLD_HIGH,
    DANGER_THRESHOLD_LOW,
    PARAM_DANGER,
)
from .music_timing import MusicClock
from .music_transition import TransitionManager, TransitionConfig


class StateChangeReason(Enum):
    """Reason for a state change."""
    EXPLICIT = auto()  # Directly requested
    PRIORITY = auto()  # Higher priority state took over
    TIMEOUT = auto()   # State timed out
    TRIGGER = auto()   # Triggered by game event
    DEFAULT = auto()   # Returned to default state


@dataclass
class MusicStateConfig:
    """Configuration for a music state.

    Attributes:
        state_id: State identifier
        track_ids: Track IDs for this state
        stem_config: Stem volume configuration
        transition_in_type: Transition type when entering
        transition_out_type: Transition type when leaving
        transition_duration_ms: Default transition duration
        loop: Whether to loop
        min_duration_ms: Minimum time in this state
        priority: State priority (higher = more important)
        can_interrupt: Whether this state can be interrupted
        auto_exit_to: State to exit to after track ends
        tags: Tags for state filtering
    """
    state_id: str
    track_ids: List[str] = field(default_factory=list)
    stem_config: Dict[str, float] = field(default_factory=dict)
    transition_in_type: str = TRANSITION_BAR_SYNC
    transition_out_type: str = TRANSITION_CROSSFADE
    transition_duration_ms: float = STATE_TRANSITION_TIME * 1000
    loop: bool = True
    min_duration_ms: float = 0.0
    priority: int = 0
    can_interrupt: bool = True
    auto_exit_to: Optional[str] = None
    tags: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if self.priority == 0 and self.state_id in STATE_PRIORITY:
            self.priority = STATE_PRIORITY[self.state_id]


@dataclass
class StateTransition:
    """Configuration for transition between specific states.

    Attributes:
        from_state: Source state
        to_state: Destination state
        transition_type: Type of transition
        duration_ms: Transition duration
        stinger_id: Stinger to play during transition
        conditions: Conditions that must be met
    """
    from_state: str
    to_state: str
    transition_type: str = TRANSITION_BAR_SYNC
    duration_ms: float = STATE_TRANSITION_TIME * 1000
    stinger_id: Optional[str] = None
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateHistoryEntry:
    """Entry in state history.

    Attributes:
        state_id: State that was active
        enter_time: When state was entered
        exit_time: When state was exited
        reason: Why state changed
        duration_ms: Time spent in state
    """
    state_id: str
    enter_time: float
    exit_time: Optional[float] = None
    reason: StateChangeReason = StateChangeReason.EXPLICIT
    duration_ms: float = 0.0


class MusicState:
    """A single music state with associated tracks and configuration."""

    def __init__(self, config: MusicStateConfig):
        """Initialize music state.

        Args:
            config: State configuration
        """
        self._config = config
        self._current_track_index = 0
        self._enter_time: Optional[float] = None
        self._is_active = False

    @property
    def state_id(self) -> str:
        """Get state ID."""
        return self._config.state_id

    @property
    def config(self) -> MusicStateConfig:
        """Get state config."""
        return self._config

    @property
    def priority(self) -> int:
        """Get state priority."""
        return self._config.priority

    @property
    def is_active(self) -> bool:
        """Check if state is active."""
        return self._is_active

    @property
    def time_in_state_ms(self) -> float:
        """Get time spent in this state."""
        if not self._is_active or self._enter_time is None:
            return 0.0
        return (time.perf_counter() - self._enter_time) * 1000

    @property
    def can_exit(self) -> bool:
        """Check if state can be exited (minimum duration met)."""
        return self.time_in_state_ms >= self._config.min_duration_ms

    def get_current_track_id(self) -> Optional[str]:
        """Get current track ID.

        Returns:
            Track ID or None if no tracks
        """
        if not self._config.track_ids:
            return None
        return self._config.track_ids[self._current_track_index]

    def get_next_track_id(self) -> Optional[str]:
        """Get next track ID.

        Returns:
            Next track ID or None
        """
        if not self._config.track_ids:
            return None
        if self._config.loop:
            return self._config.track_ids[
                (self._current_track_index + 1) % len(self._config.track_ids)
            ]
        elif self._current_track_index < len(self._config.track_ids) - 1:
            return self._config.track_ids[self._current_track_index + 1]
        return None

    def advance_track(self) -> Optional[str]:
        """Advance to next track.

        Returns:
            New track ID or None if at end
        """
        if not self._config.track_ids:
            return None

        if self._config.loop:
            self._current_track_index = (
                (self._current_track_index + 1) % len(self._config.track_ids)
            )
            return self._config.track_ids[self._current_track_index]
        elif self._current_track_index < len(self._config.track_ids) - 1:
            self._current_track_index += 1
            return self._config.track_ids[self._current_track_index]
        return None

    def enter(self):
        """Enter this state."""
        self._is_active = True
        self._enter_time = time.perf_counter()
        self._current_track_index = 0

    def exit(self):
        """Exit this state."""
        self._is_active = False

    def reset(self):
        """Reset state to initial configuration."""
        self._current_track_index = 0
        self._enter_time = None
        self._is_active = False


class MusicStateManager:
    """Manages music states and transitions between them.

    Coordinates state-driven music based on gameplay conditions.
    """

    def __init__(
        self,
        clock: MusicClock,
        transition_manager: Optional[TransitionManager] = None,
    ):
        """Initialize state manager.

        Args:
            clock: Music clock for timing
            transition_manager: Transition manager for state changes
        """
        self._clock = clock
        self._transition_manager = transition_manager
        self._states: Dict[str, MusicState] = {}
        self._state_transitions: Dict[tuple[str, str], StateTransition] = {}
        self._current_state: Optional[MusicState] = None
        self._previous_state: Optional[MusicState] = None
        self._default_state_id: str = DEFAULT_STATE
        self._state_stack: List[str] = []  # For push/pop semantics
        self._history: List[StateHistoryEntry] = []
        self._max_history = 100
        self._lock = threading.RLock()

        # Callbacks
        self._on_state_enter: Optional[Callable[[str, str], None]] = None
        self._on_state_exit: Optional[Callable[[str, StateChangeReason], None]] = None

        # Parameters that can affect state
        self._parameters: Dict[str, Any] = {}

        # Update thread
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def current_state_id(self) -> Optional[str]:
        """Get current state ID."""
        return self._current_state.state_id if self._current_state else None

    @property
    def current_state(self) -> Optional[MusicState]:
        """Get current state."""
        return self._current_state

    @property
    def previous_state_id(self) -> Optional[str]:
        """Get previous state ID."""
        return self._previous_state.state_id if self._previous_state else None

    def register_state(self, config: MusicStateConfig) -> MusicState:
        """Register a new music state.

        Args:
            config: State configuration

        Returns:
            Created MusicState
        """
        with self._lock:
            state = MusicState(config)
            self._states[config.state_id] = state
            return state

    def unregister_state(self, state_id: str) -> bool:
        """Unregister a state.

        Args:
            state_id: State to unregister

        Returns:
            True if state was found and removed
        """
        with self._lock:
            if state_id in self._states:
                state = self._states.pop(state_id)
                if self._current_state is state:
                    self._current_state = None
                return True
            return False

    def get_state(self, state_id: str) -> Optional[MusicState]:
        """Get a state by ID.

        Args:
            state_id: State ID

        Returns:
            MusicState or None
        """
        return self._states.get(state_id)

    def register_state_transition(self, transition: StateTransition):
        """Register a custom transition between states.

        Args:
            transition: State transition configuration
        """
        with self._lock:
            key = (transition.from_state, transition.to_state)
            self._state_transitions[key] = transition

    def set_default_state(self, state_id: str):
        """Set the default state.

        Args:
            state_id: State ID to use as default
        """
        if state_id not in self._states:
            raise ValueError(f"Unknown state: {state_id}")
        self._default_state_id = state_id

    def change_state(
        self,
        state_id: str,
        reason: StateChangeReason = StateChangeReason.EXPLICIT,
        force: bool = False,
    ) -> bool:
        """Change to a new state.

        Args:
            state_id: State to change to
            reason: Reason for the change
            force: Force change even if current state can't exit

        Returns:
            True if state change was initiated
        """
        with self._lock:
            new_state = self._states.get(state_id)
            if new_state is None:
                return False

            # Check if we can exit current state
            if (self._current_state is not None and
                not force and
                not self._current_state.can_exit and
                self._current_state.config.can_interrupt is False):
                return False

            # Check priority
            if (self._current_state is not None and
                not force and
                new_state.priority < self._current_state.priority):
                return False

            # Get transition config
            transition_config = self._get_transition_config(
                self._current_state.state_id if self._current_state else None,
                state_id,
            )

            # Exit current state
            if self._current_state is not None:
                self._exit_state(reason)

            # Enter new state
            self._enter_state(new_state, transition_config)

            return True

    def _get_transition_config(
        self,
        from_state: Optional[str],
        to_state: str,
    ) -> TransitionConfig:
        """Get transition configuration for state change.

        Args:
            from_state: Source state
            to_state: Destination state

        Returns:
            TransitionConfig to use
        """
        # Check for custom transition
        if from_state is not None:
            key = (from_state, to_state)
            if key in self._state_transitions:
                st = self._state_transitions[key]
                return TransitionConfig(
                    transition_type=st.transition_type,
                    duration_ms=st.duration_ms,
                    stinger_id=st.stinger_id,
                )

        # Use destination state's default
        dest_state = self._states.get(to_state)
        if dest_state is not None:
            return TransitionConfig(
                transition_type=dest_state.config.transition_in_type,
                duration_ms=dest_state.config.transition_duration_ms,
            )

        # Default
        return TransitionConfig()

    def _enter_state(self, state: MusicState, transition_config: TransitionConfig):
        """Enter a new state.

        Args:
            state: State to enter
            transition_config: Transition configuration
        """
        self._previous_state = self._current_state
        self._current_state = state
        state.enter()

        # Record history
        entry = StateHistoryEntry(
            state_id=state.state_id,
            enter_time=time.perf_counter(),
        )
        self._history.append(entry)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Request transition if we have a transition manager
        if self._transition_manager is not None:
            track_id = state.get_current_track_id()
            if track_id is not None:
                self._transition_manager.request_transition(
                    destination_id=track_id,
                    transition_type=transition_config.transition_type,
                    duration_ms=transition_config.duration_ms,
                    stinger_id=transition_config.stinger_id,
                )

        # Callback
        if self._on_state_enter is not None:
            prev_id = self._previous_state.state_id if self._previous_state else None
            self._on_state_enter(state.state_id, prev_id)

    def _exit_state(self, reason: StateChangeReason):
        """Exit current state.

        Args:
            reason: Reason for exiting
        """
        if self._current_state is None:
            return

        state = self._current_state
        state.exit()

        # Update history
        if self._history:
            self._history[-1].exit_time = time.perf_counter()
            self._history[-1].reason = reason
            self._history[-1].duration_ms = state.time_in_state_ms

        # Callback
        if self._on_state_exit is not None:
            self._on_state_exit(state.state_id, reason)

    def push_state(self, state_id: str) -> bool:
        """Push a new state onto the stack.

        Args:
            state_id: State to push

        Returns:
            True if successful
        """
        with self._lock:
            if self._current_state is not None:
                self._state_stack.append(self._current_state.state_id)
            return self.change_state(state_id)

    def pop_state(self) -> Optional[str]:
        """Pop and return to previous state.

        Returns:
            State that was returned to, or None
        """
        with self._lock:
            if not self._state_stack:
                return self.return_to_default()

            previous = self._state_stack.pop()
            self.change_state(previous, StateChangeReason.PRIORITY)
            return previous

    def return_to_default(self) -> str:
        """Return to default state.

        Returns:
            Default state ID
        """
        self.change_state(self._default_state_id, StateChangeReason.DEFAULT)
        return self._default_state_id

    def set_parameter(self, name: str, value: Any):
        """Set a state parameter.

        Parameters can be used by state conditions.

        Args:
            name: Parameter name
            value: Parameter value
        """
        with self._lock:
            self._parameters[name] = value
            self._evaluate_parameter_triggers()

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a state parameter.

        Args:
            name: Parameter name
            default: Default value if not set

        Returns:
            Parameter value
        """
        return self._parameters.get(name, default)

    def _evaluate_parameter_triggers(self):
        """Evaluate parameter-based state triggers.

        Checks parameters against thresholds and triggers appropriate state changes.
        Override or extend this method for custom parameter-driven behavior.
        """
        danger = self._parameters.get(PARAM_DANGER, 0.0)

        # Example trigger logic - high danger triggers combat
        if danger >= DANGER_THRESHOLD_HIGH:
            if self.current_state_id not in (STATE_COMBAT,):
                combat_state = self._states.get(STATE_COMBAT)
                if combat_state is not None:
                    self.change_state(STATE_COMBAT, StateChangeReason.TRIGGER)
        elif danger >= DANGER_THRESHOLD_LOW:
            if self.current_state_id not in (STATE_TENSION, STATE_COMBAT):
                tension_state = self._states.get(STATE_TENSION)
                if tension_state is not None:
                    self.change_state(STATE_TENSION, StateChangeReason.TRIGGER)

    def update(self):
        """Update state manager."""
        with self._lock:
            if self._current_state is None:
                return

            # Check for auto-exit
            if (self._current_state.config.auto_exit_to is not None and
                self._current_state.can_exit):
                # Check if track ended (would need integration with player)
                pass

    def set_callbacks(
        self,
        on_state_enter: Optional[Callable[[str, str], None]] = None,
        on_state_exit: Optional[Callable[[str, StateChangeReason], None]] = None,
    ):
        """Set manager callbacks.

        Args:
            on_state_enter: Called when entering state (new_state, prev_state)
            on_state_exit: Called when exiting state (state, reason)
        """
        self._on_state_enter = on_state_enter
        self._on_state_exit = on_state_exit

    def get_state_history(self, limit: int = 10) -> List[StateHistoryEntry]:
        """Get recent state history.

        Args:
            limit: Maximum entries to return

        Returns:
            List of history entries (most recent last)
        """
        with self._lock:
            return self._history[-limit:]

    def get_all_states(self) -> List[MusicState]:
        """Get all registered states.

        Returns:
            List of all states
        """
        with self._lock:
            return list(self._states.values())

    def start_update_loop(self, interval_ms: float = 50.0):
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
        """Clear all states."""
        with self._lock:
            if self._current_state is not None:
                self._exit_state(StateChangeReason.EXPLICIT)
            self._states.clear()
            self._state_transitions.clear()
            self._state_stack.clear()
            self._current_state = None
            self._previous_state = None


# Pre-defined state configurations for common gameplay scenarios
def create_exploration_state(track_ids: List[str]) -> MusicStateConfig:
    """Create exploration state configuration."""
    return MusicStateConfig(
        state_id=STATE_EXPLORATION,
        track_ids=track_ids,
        stem_config={"drums": 0.3, "bass": 0.5, "melody": 0.8, "pads": 1.0},
        priority=STATE_PRIORITY[STATE_EXPLORATION],
        loop=True,
    )


def create_combat_state(track_ids: List[str]) -> MusicStateConfig:
    """Create combat state configuration."""
    return MusicStateConfig(
        state_id=STATE_COMBAT,
        track_ids=track_ids,
        stem_config={"drums": 1.0, "bass": 1.0, "melody": 0.8, "pads": 0.4},
        transition_in_type=TRANSITION_BAR_SYNC,
        priority=STATE_PRIORITY[STATE_COMBAT],
        min_duration_ms=5000,
        loop=True,
    )


def create_stealth_state(track_ids: List[str]) -> MusicStateConfig:
    """Create stealth state configuration."""
    return MusicStateConfig(
        state_id=STATE_STEALTH,
        track_ids=track_ids,
        stem_config={"drums": 0.2, "bass": 0.6, "melody": 0.4, "pads": 0.8},
        priority=STATE_PRIORITY[STATE_STEALTH],
        loop=True,
    )


def create_boss_state(track_ids: List[str]) -> MusicStateConfig:
    """Create boss battle state configuration."""
    return MusicStateConfig(
        state_id=STATE_BOSS,
        track_ids=track_ids,
        stem_config={"drums": 1.0, "bass": 1.0, "melody": 1.0, "pads": 0.6},
        transition_in_type=TRANSITION_BAR_SYNC,
        priority=STATE_PRIORITY[STATE_BOSS],
        min_duration_ms=10000,
        can_interrupt=False,
        loop=True,
    )


def create_victory_state(track_ids: List[str]) -> MusicStateConfig:
    """Create victory state configuration."""
    return MusicStateConfig(
        state_id=STATE_VICTORY,
        track_ids=track_ids,
        priority=STATE_PRIORITY[STATE_VICTORY],
        loop=False,
        auto_exit_to=STATE_EXPLORATION,
    )


def create_defeat_state(track_ids: List[str]) -> MusicStateConfig:
    """Create defeat state configuration."""
    return MusicStateConfig(
        state_id=STATE_DEFEAT,
        track_ids=track_ids,
        priority=STATE_PRIORITY[STATE_DEFEAT],
        loop=False,
        auto_exit_to=STATE_MENU,
    )
