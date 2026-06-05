"""T-CC-4.3: Time-travel UI components for visual debugging.

This module provides UI components for time-travel debugging that integrate
with the DebugUI framework from T-CC-3.6 and the TimeTravel system from T-CC-4.1.

Key components:
- TimelinePanel: Visual timeline with tick markers and scrub bar
- ScrubBar: Seekable slider for navigating simulation history
- StateDiffView: Side-by-side view showing state changes between ticks
- StepControls: Forward/backward step buttons with configurable step sizes
- SnapshotMarkerWidget: Visual markers for snapshot positions on timeline

Example:
    from engine.debug.time_travel_ui import TimeTravelUI, TimeTravelUIConfig
    from engine.debug.time_travel import TimeTravel

    # Create UI bound to time travel system
    config = TimeTravelUIConfig(
        show_diff_view=True,
        step_sizes=[1, 10, 60],
    )
    ui = TimeTravelUI(time_travel, debug_ui, config)

    # Update each frame
    ui.update()

Dependencies:
    - T-CC-4.1 (time_travel): TimeTravel, TickSnapshot, SnapshotComparison
    - T-CC-3.6 (debug_ui): DebugUI, Widget, ContainerWidget, CollapsibleSection
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from engine.tooling.editor.app_shell import editor, reloadable
from engine.tooling.editor.debug_ui import (
    ButtonWidget,
    CheckboxWidget,
    CollapsibleSection,
    Color,
    ContainerWidget,
    DebugUI,
    DebugUIContext,
    DropdownWidget,
    FloatSliderWidget,
    IntInputWidget,
    IntSliderWidget,
    LabelWidget,
    PropertyPanel,
    SeparatorWidget,
    UIEvent,
    UIState,
    Vec2,
    Widget,
    WidgetConfig,
    WidgetType,
)

if TYPE_CHECKING:
    from engine.debug.time_travel import (
        SnapshotComparison,
        TickRange,
        TickSnapshot,
        TimeTravel,
        TimeTravelEvent,
        TimeTravelState,
    )


__all__ = [
    # Configuration
    "TimeTravelUIConfig",
    # Main UI components
    "TimeTravelUI",
    "TimelinePanel",
    "ScrubBar",
    "StepControls",
    "StateDiffView",
    # Timeline widgets
    "SnapshotMarkerWidget",
    "TickMarkerWidget",
    "PlayheadWidget",
    # Diff widgets
    "DiffEntry",
    "DiffEntryWidget",
    "DiffTreeNode",
    # Events
    "UIActionType",
    "UIAction",
    # Playback
    "PlaybackState",
    "PlaybackController",
]


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass(slots=True, frozen=True)
class TimeTravelUIConfig:
    """Configuration for time-travel UI components.

    Attributes:
        show_timeline: Whether to show the timeline panel.
        show_diff_view: Whether to show the state diff view.
        show_step_controls: Whether to show step forward/back buttons.
        step_sizes: Available step sizes for navigation (in ticks).
        timeline_height: Height of timeline panel in pixels.
        scrub_bar_height: Height of scrub bar in pixels.
        marker_size: Size of snapshot markers in pixels.
        diff_max_depth: Maximum depth for nested diff display.
        auto_scroll_timeline: Auto-scroll timeline to follow playhead.
        highlight_snapshots: Highlight ticks with snapshots.
        show_tick_labels: Show tick number labels on timeline.
        tick_label_interval: Interval for tick labels (show every N ticks).
        playback_speeds: Available playback speed multipliers.
        colors: Custom color scheme for UI elements.
    """

    show_timeline: bool = True
    show_diff_view: bool = True
    show_step_controls: bool = True
    step_sizes: Tuple[int, ...] = (1, 10, 60, 600)
    timeline_height: int = 80
    scrub_bar_height: int = 24
    marker_size: int = 8
    diff_max_depth: int = 10
    auto_scroll_timeline: bool = True
    highlight_snapshots: bool = True
    show_tick_labels: bool = True
    tick_label_interval: int = 60
    playback_speeds: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    colors: Optional[Dict[str, Color]] = None

    def get_color(self, name: str) -> Color:
        """Get color by name with defaults."""
        defaults = {
            "timeline_bg": Color(0.15, 0.15, 0.15, 1.0),
            "timeline_tick": Color(0.4, 0.4, 0.4, 1.0),
            "timeline_major_tick": Color(0.6, 0.6, 0.6, 1.0),
            "playhead": Color(1.0, 0.3, 0.3, 1.0),
            "snapshot_marker": Color(0.3, 0.8, 0.3, 1.0),
            "scrub_bar_bg": Color(0.2, 0.2, 0.2, 1.0),
            "scrub_bar_fill": Color(0.4, 0.6, 0.8, 1.0),
            "scrub_bar_handle": Color(0.8, 0.8, 0.8, 1.0),
            "diff_added": Color(0.3, 0.8, 0.3, 1.0),
            "diff_removed": Color(0.8, 0.3, 0.3, 1.0),
            "diff_modified": Color(0.8, 0.7, 0.2, 1.0),
            "diff_unchanged": Color(0.5, 0.5, 0.5, 1.0),
        }
        if self.colors and name in self.colors:
            return self.colors[name]
        return defaults.get(name, Color(1.0, 1.0, 1.0, 1.0))


# =============================================================================
# UI ACTIONS
# =============================================================================


class UIActionType(Enum):
    """Types of UI actions triggered by user interaction."""

    SEEK_TO_TICK = auto()
    STEP_FORWARD = auto()
    STEP_BACKWARD = auto()
    PLAY = auto()
    PAUSE = auto()
    STOP = auto()
    TOGGLE_RECORDING = auto()
    CAPTURE_SNAPSHOT = auto()
    COMPARE_TICKS = auto()
    ZOOM_IN = auto()
    ZOOM_OUT = auto()
    SCROLL_LEFT = auto()
    SCROLL_RIGHT = auto()
    SET_PLAYBACK_SPEED = auto()


@dataclass(slots=True)
class UIAction:
    """Represents a UI action with associated data.

    Attributes:
        action_type: The type of action.
        tick: Target tick for seek/compare actions.
        tick_b: Second tick for compare actions.
        step_size: Number of ticks for step actions.
        speed: Playback speed multiplier.
    """

    action_type: UIActionType
    tick: int = 0
    tick_b: int = 0
    step_size: int = 1
    speed: float = 1.0


# =============================================================================
# PLAYBACK CONTROLLER
# =============================================================================


class PlaybackState(Enum):
    """Playback state for time-travel UI."""

    STOPPED = auto()
    PLAYING_FORWARD = auto()
    PLAYING_BACKWARD = auto()
    PAUSED = auto()


@editor(category="TimeTravelUI")
@reloadable()
class PlaybackController:
    """Controls playback of recorded simulation history.

    Manages play/pause/stop states and playback speed for
    automatic stepping through history.

    Attributes:
        state: Current playback state.
        speed: Playback speed multiplier (1.0 = real-time).
        loop: Whether to loop at end of history.
    """

    __slots__ = (
        "_state",
        "_speed",
        "_loop",
        "_last_step_time",
        "_ticks_per_second",
        "_accumulated_time",
    )

    def __init__(
        self,
        ticks_per_second: int = 60,
        initial_speed: float = 1.0,
        loop: bool = False,
    ):
        """Initialize playback controller.

        Args:
            ticks_per_second: Base tick rate for playback timing.
            initial_speed: Initial playback speed multiplier.
            loop: Whether to loop playback.
        """
        self._state = PlaybackState.STOPPED
        self._speed = initial_speed
        self._loop = loop
        self._last_step_time = 0.0
        self._ticks_per_second = ticks_per_second
        self._accumulated_time = 0.0

    @property
    def state(self) -> PlaybackState:
        """Current playback state."""
        return self._state

    @property
    def speed(self) -> float:
        """Current playback speed multiplier."""
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        """Set playback speed (clamped to 0.1 - 10.0)."""
        self._speed = max(0.1, min(10.0, value))

    @property
    def loop(self) -> bool:
        """Whether playback loops."""
        return self._loop

    @loop.setter
    def loop(self, value: bool) -> None:
        """Set loop mode."""
        self._loop = value

    @property
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._state in (
            PlaybackState.PLAYING_FORWARD,
            PlaybackState.PLAYING_BACKWARD,
        )

    @property
    def is_paused(self) -> bool:
        """Check if paused."""
        return self._state == PlaybackState.PAUSED

    @property
    def is_stopped(self) -> bool:
        """Check if stopped."""
        return self._state == PlaybackState.STOPPED

    def play_forward(self) -> None:
        """Start playing forward."""
        self._state = PlaybackState.PLAYING_FORWARD
        self._last_step_time = time.time()
        self._accumulated_time = 0.0

    def play_backward(self) -> None:
        """Start playing backward."""
        self._state = PlaybackState.PLAYING_BACKWARD
        self._last_step_time = time.time()
        self._accumulated_time = 0.0

    def pause(self) -> None:
        """Pause playback."""
        if self.is_playing:
            self._state = PlaybackState.PAUSED

    def resume(self) -> None:
        """Resume playback from pause."""
        if self._state == PlaybackState.PAUSED:
            self._state = PlaybackState.PLAYING_FORWARD
            self._last_step_time = time.time()

    def stop(self) -> None:
        """Stop playback."""
        self._state = PlaybackState.STOPPED
        self._accumulated_time = 0.0

    def toggle_play_pause(self) -> None:
        """Toggle between play and pause states."""
        if self.is_playing:
            self.pause()
        elif self._state == PlaybackState.PAUSED:
            self.resume()
        else:
            self.play_forward()

    def update(self, delta_time: float) -> int:
        """Update playback and return number of ticks to step.

        Args:
            delta_time: Time elapsed since last update in seconds.

        Returns:
            Number of ticks to step (positive for forward, negative for backward).
        """
        if not self.is_playing:
            return 0

        # Calculate ticks to step based on speed and time
        self._accumulated_time += delta_time * self._speed
        tick_duration = 1.0 / self._ticks_per_second
        ticks_to_step = int(self._accumulated_time / tick_duration)

        if ticks_to_step > 0:
            self._accumulated_time -= ticks_to_step * tick_duration

        # Apply direction
        if self._state == PlaybackState.PLAYING_BACKWARD:
            ticks_to_step = -ticks_to_step

        return ticks_to_step

    def reset(self) -> None:
        """Reset controller to initial state."""
        self._state = PlaybackState.STOPPED
        self._accumulated_time = 0.0
        self._last_step_time = 0.0


# =============================================================================
# TIMELINE WIDGETS
# =============================================================================


@editor(category="TimeTravelUI")
@reloadable()
class TickMarkerWidget(Widget):
    """Widget displaying a tick marker on the timeline."""

    __slots__ = ("tick", "is_major", "marker_height")

    def __init__(
        self,
        tick: int,
        is_major: bool = False,
        marker_height: int = 12,
        config: Optional[WidgetConfig] = None,
    ):
        """Initialize tick marker.

        Args:
            tick: The tick number this marker represents.
            is_major: Whether this is a major tick (shown larger).
            marker_height: Height of the marker line in pixels.
            config: Widget configuration.
        """
        super().__init__(WidgetType.CUSTOM, config)
        self.tick = tick
        self.is_major = is_major
        self.marker_height = marker_height

    def render(self, ctx: DebugUIContext) -> None:
        """Render the tick marker."""
        if not self.visible:
            return
        ctx._draw_commands.append(
            {
                "type": "tick_marker",
                "tick": self.tick,
                "is_major": self.is_major,
                "height": self.marker_height,
                "x": ctx.cursor_pos.x,
                "y": ctx.cursor_pos.y,
            }
        )


@editor(category="TimeTravelUI")
@reloadable()
class SnapshotMarkerWidget(Widget):
    """Widget displaying a snapshot marker on the timeline."""

    __slots__ = ("tick", "snapshot_id", "marker_size", "highlight", "_snapshot_ref")

    def __init__(
        self,
        tick: int,
        snapshot_id: str = "",
        marker_size: int = 8,
        highlight: bool = False,
        config: Optional[WidgetConfig] = None,
    ):
        """Initialize snapshot marker.

        Args:
            tick: The tick number where snapshot was taken.
            snapshot_id: Unique identifier for the snapshot.
            marker_size: Size of the marker in pixels.
            highlight: Whether to highlight this marker.
            config: Widget configuration.
        """
        super().__init__(WidgetType.CUSTOM, config)
        self.tick = tick
        self.snapshot_id = snapshot_id or f"snapshot_{tick}"
        self.marker_size = marker_size
        self.highlight = highlight
        self._snapshot_ref: Optional[Any] = None

    def set_snapshot(self, snapshot: "TickSnapshot") -> None:
        """Associate with a snapshot object."""
        self._snapshot_ref = snapshot

    def render(self, ctx: DebugUIContext) -> None:
        """Render the snapshot marker."""
        if not self.visible:
            return
        ctx._draw_commands.append(
            {
                "type": "snapshot_marker",
                "tick": self.tick,
                "id": self.snapshot_id,
                "size": self.marker_size,
                "highlight": self.highlight,
                "x": ctx.cursor_pos.x,
                "y": ctx.cursor_pos.y,
            }
        )


@editor(category="TimeTravelUI")
@reloadable()
class PlayheadWidget(Widget):
    """Widget displaying the current playhead position on timeline."""

    __slots__ = ("tick", "dragging", "_drag_start_tick")

    def __init__(
        self,
        tick: int = 0,
        config: Optional[WidgetConfig] = None,
    ):
        """Initialize playhead widget.

        Args:
            tick: Current tick position.
            config: Widget configuration.
        """
        super().__init__(WidgetType.CUSTOM, config)
        self.tick = tick
        self.dragging = False
        self._drag_start_tick = 0

    def set_tick(self, tick: int) -> None:
        """Update playhead position."""
        self.tick = max(0, tick)

    def start_drag(self) -> None:
        """Begin dragging the playhead."""
        self.dragging = True
        self._drag_start_tick = self.tick

    def end_drag(self) -> int:
        """End dragging and return final tick position."""
        self.dragging = False
        return self.tick

    def cancel_drag(self) -> None:
        """Cancel drag and restore original position."""
        self.dragging = False
        self.tick = self._drag_start_tick

    def render(self, ctx: DebugUIContext) -> None:
        """Render the playhead."""
        if not self.visible:
            return
        ctx._draw_commands.append(
            {
                "type": "playhead",
                "tick": self.tick,
                "dragging": self.dragging,
                "x": ctx.cursor_pos.x,
                "y": ctx.cursor_pos.y,
            }
        )


# =============================================================================
# SCRUB BAR
# =============================================================================


@editor(category="TimeTravelUI")
@reloadable()
class ScrubBar(Widget):
    """Seekable slider for navigating simulation history.

    Provides a visual timeline bar that users can click or drag to
    seek to any point in the recorded history. Shows current position
    relative to available history range.

    Attributes:
        min_tick: Minimum tick in history (leftmost position).
        max_tick: Maximum tick in history (rightmost position).
        current_tick: Current playhead position.
        on_seek: Callback when user seeks to a new tick.
    """

    __slots__ = (
        "min_tick",
        "max_tick",
        "current_tick",
        "on_seek",
        "_dragging",
        "_hover_tick",
        "_bar_width",
        "_bar_height",
    )

    def __init__(
        self,
        min_tick: int = 0,
        max_tick: int = 0,
        current_tick: int = 0,
        bar_width: int = 400,
        bar_height: int = 24,
        config: Optional[WidgetConfig] = None,
    ):
        """Initialize scrub bar.

        Args:
            min_tick: Minimum tick in range.
            max_tick: Maximum tick in range.
            current_tick: Initial position.
            bar_width: Width of bar in pixels.
            bar_height: Height of bar in pixels.
            config: Widget configuration.
        """
        super().__init__(WidgetType.CUSTOM, config)
        self.min_tick = min_tick
        self.max_tick = max_tick
        self.current_tick = current_tick
        self.on_seek: Optional[Callable[[int], None]] = None
        self._dragging = False
        self._hover_tick = -1
        self._bar_width = bar_width
        self._bar_height = bar_height

    @property
    def range(self) -> int:
        """Total tick range."""
        return max(1, self.max_tick - self.min_tick)

    @property
    def position(self) -> float:
        """Current position as fraction [0, 1]."""
        if self.range <= 0:
            return 0.0
        return (self.current_tick - self.min_tick) / self.range

    @property
    def is_empty(self) -> bool:
        """Check if scrub bar has no range."""
        return self.max_tick <= self.min_tick

    def set_range(self, min_tick: int, max_tick: int) -> None:
        """Update the tick range."""
        self.min_tick = min_tick
        self.max_tick = max(min_tick, max_tick)
        # Clamp current tick to new range
        self.current_tick = max(self.min_tick, min(self.max_tick, self.current_tick))

    def set_current(self, tick: int) -> None:
        """Set current tick position."""
        self.current_tick = max(self.min_tick, min(self.max_tick, tick))

    def seek_to_fraction(self, fraction: float) -> int:
        """Seek to position as fraction [0, 1]. Returns actual tick."""
        fraction = max(0.0, min(1.0, fraction))
        tick = self.min_tick + int(fraction * self.range)
        self.current_tick = tick
        if self.on_seek:
            self.on_seek(tick)
        return tick

    def seek_relative(self, delta_ticks: int) -> int:
        """Seek relative to current position. Returns actual tick."""
        new_tick = max(self.min_tick, min(self.max_tick, self.current_tick + delta_ticks))
        if new_tick != self.current_tick:
            self.current_tick = new_tick
            if self.on_seek:
                self.on_seek(new_tick)
        return new_tick

    def tick_to_x(self, tick: int) -> float:
        """Convert tick to x position within bar."""
        if self.range <= 0:
            return 0.0
        fraction = (tick - self.min_tick) / self.range
        return fraction * self._bar_width

    def x_to_tick(self, x: float) -> int:
        """Convert x position to tick."""
        if self._bar_width <= 0:
            return self.min_tick
        fraction = max(0.0, min(1.0, x / self._bar_width))
        return self.min_tick + int(fraction * self.range)

    def start_drag(self) -> None:
        """Begin dragging the scrub bar."""
        self._dragging = True

    def end_drag(self) -> None:
        """End dragging."""
        self._dragging = False

    def update_hover(self, x: float) -> int:
        """Update hover position and return tick under cursor."""
        self._hover_tick = self.x_to_tick(x)
        return self._hover_tick

    def clear_hover(self) -> None:
        """Clear hover state."""
        self._hover_tick = -1

    def handle_input(self, event: UIEvent) -> bool:
        """Handle input events."""
        if event.type == "mouse_down":
            self.start_drag()
            tick = self.x_to_tick(event.x)
            self.current_tick = tick
            if self.on_seek:
                self.on_seek(tick)
            return True
        elif event.type == "mouse_up":
            if self._dragging:
                self.end_drag()
                return True
        elif event.type == "mouse_move":
            if self._dragging:
                tick = self.x_to_tick(event.x)
                self.current_tick = tick
                if self.on_seek:
                    self.on_seek(tick)
                return True
            else:
                self.update_hover(event.x)
        return False

    def render(self, ctx: DebugUIContext) -> None:
        """Render the scrub bar."""
        if not self.visible:
            return
        ctx._draw_commands.append(
            {
                "type": "scrub_bar",
                "id": self.id,
                "min_tick": self.min_tick,
                "max_tick": self.max_tick,
                "current_tick": self.current_tick,
                "position": self.position,
                "dragging": self._dragging,
                "hover_tick": self._hover_tick,
                "width": self._bar_width,
                "height": self._bar_height,
                "x": ctx.cursor_pos.x,
                "y": ctx.cursor_pos.y,
            }
        )
        ctx.next_line(self._bar_height + 4)


# =============================================================================
# STEP CONTROLS
# =============================================================================


@editor(category="TimeTravelUI")
@reloadable()
class StepControls(ContainerWidget):
    """Forward/backward step buttons with configurable step sizes.

    Provides buttons for:
    - Step backward (various sizes)
    - Step forward (various sizes)
    - Play/pause toggle
    - Stop button
    - Recording toggle

    Attributes:
        step_sizes: Available step sizes in ticks.
        on_step: Callback when step button pressed (receives signed step count).
        on_play_pause: Callback when play/pause toggled.
        on_stop: Callback when stop pressed.
        on_record_toggle: Callback when recording toggled.
    """

    __slots__ = (
        "step_sizes",
        "current_step_size_index",
        "is_playing",
        "is_recording",
        "on_step",
        "on_play_pause",
        "on_stop",
        "on_record_toggle",
        "_step_back_buttons",
        "_step_forward_buttons",
        "_play_button",
        "_stop_button",
        "_record_button",
        "_step_size_dropdown",
    )

    def __init__(
        self,
        step_sizes: Tuple[int, ...] = (1, 10, 60),
        config: Optional[WidgetConfig] = None,
    ):
        """Initialize step controls.

        Args:
            step_sizes: Available step sizes in ticks.
            config: Widget configuration.
        """
        super().__init__(WidgetType.CUSTOM, config)
        self.step_sizes = step_sizes
        self.current_step_size_index = 0
        self.is_playing = False
        self.is_recording = False

        # Callbacks
        self.on_step: Optional[Callable[[int], None]] = None
        self.on_play_pause: Optional[Callable[[], None]] = None
        self.on_stop: Optional[Callable[[], None]] = None
        self.on_record_toggle: Optional[Callable[[bool], None]] = None

        # Create child widgets
        self._step_back_buttons: List[ButtonWidget] = []
        self._step_forward_buttons: List[ButtonWidget] = []
        self._create_widgets()

    @property
    def current_step_size(self) -> int:
        """Get current step size."""
        if 0 <= self.current_step_size_index < len(self.step_sizes):
            return self.step_sizes[self.current_step_size_index]
        return 1

    def _create_widgets(self) -> None:
        """Create all button widgets."""
        # Step backward buttons (largest to smallest)
        for size in reversed(self.step_sizes):
            btn = ButtonWidget(f"<<{size}", icon="step_back")
            btn.on_click = lambda s=size: self._on_step_back(s)
            self._step_back_buttons.append(btn)
            self.add_child(btn)

        # Play/pause button
        self._play_button = ButtonWidget("Play", icon="play")
        self._play_button.on_click = self._on_play_pause
        self.add_child(self._play_button)

        # Stop button
        self._stop_button = ButtonWidget("Stop", icon="stop")
        self._stop_button.on_click = self._on_stop
        self.add_child(self._stop_button)

        # Step forward buttons (smallest to largest)
        for size in self.step_sizes:
            btn = ButtonWidget(f"{size}>>", icon="step_forward")
            btn.on_click = lambda s=size: self._on_step_forward(s)
            self._step_forward_buttons.append(btn)
            self.add_child(btn)

        # Record toggle button
        self._record_button = ButtonWidget("Record", icon="record")
        self._record_button.on_click = self._on_record_toggle
        self.add_child(self._record_button)

        # Step size dropdown
        options = [f"{s} tick{'s' if s > 1 else ''}" for s in self.step_sizes]
        self._step_size_dropdown = DropdownWidget("Step Size", options, 0)
        self._step_size_dropdown.on_change = self._on_step_size_change
        self.add_child(self._step_size_dropdown)

    def _on_step_back(self, step_size: int) -> None:
        """Handle step backward button click."""
        if self.on_step:
            self.on_step(-step_size)

    def _on_step_forward(self, step_size: int) -> None:
        """Handle step forward button click."""
        if self.on_step:
            self.on_step(step_size)

    def _on_play_pause(self) -> None:
        """Handle play/pause button click."""
        self.is_playing = not self.is_playing
        self._play_button.label = "Pause" if self.is_playing else "Play"
        if self.on_play_pause:
            self.on_play_pause()

    def _on_stop(self) -> None:
        """Handle stop button click."""
        self.is_playing = False
        self._play_button.label = "Play"
        if self.on_stop:
            self.on_stop()

    def _on_record_toggle(self) -> None:
        """Handle record toggle button click."""
        self.is_recording = not self.is_recording
        self._record_button.label = "Stop Rec" if self.is_recording else "Record"
        if self.on_record_toggle:
            self.on_record_toggle(self.is_recording)

    def _on_step_size_change(self, value: str) -> None:
        """Handle step size dropdown change."""
        try:
            idx = self._step_size_dropdown.selected_index
            if 0 <= idx < len(self.step_sizes):
                self.current_step_size_index = idx
        except (ValueError, IndexError):
            pass

    def set_playing(self, playing: bool) -> None:
        """Update playing state."""
        self.is_playing = playing
        self._play_button.label = "Pause" if playing else "Play"

    def set_recording(self, recording: bool) -> None:
        """Update recording state."""
        self.is_recording = recording
        self._record_button.label = "Stop Rec" if recording else "Record"

    def step_backward(self) -> None:
        """Trigger step backward with current step size."""
        self._on_step_back(self.current_step_size)

    def step_forward(self) -> None:
        """Trigger step forward with current step size."""
        self._on_step_forward(self.current_step_size)

    def render(self, ctx: DebugUIContext) -> None:
        """Render all step controls."""
        if not self.visible:
            return
        ctx._draw_commands.append(
            {
                "type": "step_controls",
                "id": self.id,
                "is_playing": self.is_playing,
                "is_recording": self.is_recording,
                "step_sizes": self.step_sizes,
                "current_step_size": self.current_step_size,
                "x": ctx.cursor_pos.x,
                "y": ctx.cursor_pos.y,
            }
        )
        # Render child buttons
        super().render(ctx)


# =============================================================================
# STATE DIFF VIEW
# =============================================================================


class DiffType(IntEnum):
    """Type of difference between two values."""

    UNCHANGED = 0
    ADDED = 1
    REMOVED = 2
    MODIFIED = 3


@dataclass(slots=True)
class DiffEntry:
    """A single entry in the state diff.

    Attributes:
        path: Dot-separated path to the value (e.g., "entities.42.position.x").
        diff_type: Type of change (added, removed, modified, unchanged).
        old_value: Value in tick A (None if added).
        new_value: Value in tick B (None if removed).
        depth: Nesting depth for display.
    """

    path: str
    diff_type: DiffType
    old_value: Any = None
    new_value: Any = None
    depth: int = 0

    @property
    def key(self) -> str:
        """Get the final key name from path."""
        parts = self.path.split(".")
        return parts[-1] if parts else ""

    @property
    def is_changed(self) -> bool:
        """Check if entry represents a change."""
        return self.diff_type != DiffType.UNCHANGED


@editor(category="TimeTravelUI")
@reloadable()
class DiffEntryWidget(Widget):
    """Widget displaying a single diff entry."""

    __slots__ = ("entry", "expanded", "on_expand_toggle")

    def __init__(
        self,
        entry: DiffEntry,
        config: Optional[WidgetConfig] = None,
    ):
        """Initialize diff entry widget.

        Args:
            entry: The diff entry to display.
            config: Widget configuration.
        """
        super().__init__(WidgetType.CUSTOM, config)
        self.entry = entry
        self.expanded = True
        self.on_expand_toggle: Optional[Callable[[bool], None]] = None

    def toggle_expand(self) -> None:
        """Toggle expanded state."""
        self.expanded = not self.expanded
        if self.on_expand_toggle:
            self.on_expand_toggle(self.expanded)

    def render(self, ctx: DebugUIContext) -> None:
        """Render the diff entry."""
        if not self.visible:
            return
        ctx._draw_commands.append(
            {
                "type": "diff_entry",
                "id": self.id,
                "path": self.entry.path,
                "key": self.entry.key,
                "diff_type": self.entry.diff_type.name,
                "old_value": str(self.entry.old_value) if self.entry.old_value is not None else None,
                "new_value": str(self.entry.new_value) if self.entry.new_value is not None else None,
                "depth": self.entry.depth,
                "expanded": self.expanded,
                "x": ctx.cursor_pos.x,
                "y": ctx.cursor_pos.y,
            }
        )
        ctx.next_line()


@dataclass(slots=True)
class DiffTreeNode:
    """A node in the diff tree structure.

    Allows hierarchical display of nested state changes.

    Attributes:
        path: Full path to this node.
        key: Key name for this node.
        diff_type: Type of change at this node.
        old_value: Old value (for leaf nodes).
        new_value: New value (for leaf nodes).
        children: Child nodes for nested values.
        expanded: Whether node is expanded in UI.
    """

    path: str
    key: str
    diff_type: DiffType = DiffType.UNCHANGED
    old_value: Any = None
    new_value: Any = None
    children: List["DiffTreeNode"] = field(default_factory=list)
    expanded: bool = True

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    @property
    def has_changes(self) -> bool:
        """Check if this node or any children have changes."""
        if self.diff_type != DiffType.UNCHANGED:
            return True
        return any(child.has_changes for child in self.children)

    def iter_entries(self, max_depth: int = 10, current_depth: int = 0) -> Iterator[DiffEntry]:
        """Iterate over all diff entries in this tree.

        Args:
            max_depth: Maximum depth to traverse.
            current_depth: Current traversal depth.

        Yields:
            DiffEntry for each node.
        """
        if current_depth > max_depth:
            return

        if self.is_leaf or current_depth == max_depth:
            yield DiffEntry(
                path=self.path,
                diff_type=self.diff_type,
                old_value=self.old_value,
                new_value=self.new_value,
                depth=current_depth,
            )
        else:
            # Yield this node as a header
            yield DiffEntry(
                path=self.path,
                diff_type=self.diff_type,
                depth=current_depth,
            )
            # Yield children
            for child in self.children:
                yield from child.iter_entries(max_depth, current_depth + 1)


@editor(category="TimeTravelUI")
@reloadable()
class StateDiffView(ContainerWidget):
    """Side-by-side view showing state changes between ticks.

    Displays a hierarchical diff of state between two ticks,
    highlighting added, removed, and modified values.

    Attributes:
        tick_a: First tick for comparison.
        tick_b: Second tick for comparison.
        entries: List of diff entries to display.
        filter_unchanged: Whether to hide unchanged values.
        max_depth: Maximum nesting depth to display.
    """

    __slots__ = (
        "tick_a",
        "tick_b",
        "entries",
        "filter_unchanged",
        "max_depth",
        "_diff_tree",
        "_entry_widgets",
        "_expanded_paths",
        "_stats",
    )

    def __init__(
        self,
        tick_a: int = 0,
        tick_b: int = 0,
        filter_unchanged: bool = True,
        max_depth: int = 10,
        config: Optional[WidgetConfig] = None,
    ):
        """Initialize state diff view.

        Args:
            tick_a: First tick for comparison.
            tick_b: Second tick for comparison.
            filter_unchanged: Whether to hide unchanged values.
            max_depth: Maximum nesting depth.
            config: Widget configuration.
        """
        super().__init__(WidgetType.CUSTOM, config)
        self.tick_a = tick_a
        self.tick_b = tick_b
        self.entries: List[DiffEntry] = []
        self.filter_unchanged = filter_unchanged
        self.max_depth = max_depth
        self._diff_tree: Optional[DiffTreeNode] = None
        self._entry_widgets: List[DiffEntryWidget] = []
        self._expanded_paths: Set[str] = set()
        self._stats = {"added": 0, "removed": 0, "modified": 0, "unchanged": 0}

    @property
    def stats(self) -> Dict[str, int]:
        """Get diff statistics."""
        return dict(self._stats)

    @property
    def has_changes(self) -> bool:
        """Check if diff contains any changes."""
        return self._stats["added"] + self._stats["removed"] + self._stats["modified"] > 0

    @property
    def total_changes(self) -> int:
        """Get total number of changes."""
        return self._stats["added"] + self._stats["removed"] + self._stats["modified"]

    def set_comparison(
        self,
        tick_a: int,
        tick_b: int,
        comparison: Optional["SnapshotComparison"] = None,
    ) -> None:
        """Set comparison ticks and update diff entries.

        Args:
            tick_a: First tick.
            tick_b: Second tick.
            comparison: Comparison result from TimeTravel.compare_ticks().
        """
        self.tick_a = tick_a
        self.tick_b = tick_b
        self.entries.clear()
        self._entry_widgets.clear()
        self.clear_children()
        self._reset_stats()

        if comparison is None:
            return

        # Build diff entries from comparison
        self._build_diff_entries(comparison)

    def _reset_stats(self) -> None:
        """Reset diff statistics."""
        self._stats = {"added": 0, "removed": 0, "modified": 0, "unchanged": 0}

    def _build_diff_entries(self, comparison: "SnapshotComparison") -> None:
        """Build diff entries from comparison result."""
        for key, (old_val, new_val) in comparison.differences.items():
            if old_val is None and new_val is not None:
                diff_type = DiffType.ADDED
                self._stats["added"] += 1
            elif old_val is not None and new_val is None:
                diff_type = DiffType.REMOVED
                self._stats["removed"] += 1
            else:
                diff_type = DiffType.MODIFIED
                self._stats["modified"] += 1

            entry = DiffEntry(
                path=key,
                diff_type=diff_type,
                old_value=old_val,
                new_value=new_val,
                depth=0,
            )
            self.entries.append(entry)

            # Create widget for entry
            widget = DiffEntryWidget(entry)
            self._entry_widgets.append(widget)
            self.add_child(widget)

    def set_diff_data(
        self,
        state_a: Dict[str, Any],
        state_b: Dict[str, Any],
    ) -> None:
        """Build diff from raw state dictionaries.

        Args:
            state_a: State at tick A.
            state_b: State at tick B.
        """
        self.entries.clear()
        self._entry_widgets.clear()
        self.clear_children()
        self._reset_stats()

        # Build diff tree
        self._diff_tree = self._build_diff_tree("root", state_a, state_b)

        # Flatten to entries
        for entry in self._diff_tree.iter_entries(self.max_depth):
            if self.filter_unchanged and entry.diff_type == DiffType.UNCHANGED:
                continue
            self.entries.append(entry)

            # Update stats
            if entry.diff_type == DiffType.ADDED:
                self._stats["added"] += 1
            elif entry.diff_type == DiffType.REMOVED:
                self._stats["removed"] += 1
            elif entry.diff_type == DiffType.MODIFIED:
                self._stats["modified"] += 1
            else:
                self._stats["unchanged"] += 1

            # Create widget
            widget = DiffEntryWidget(entry)
            self._entry_widgets.append(widget)
            self.add_child(widget)

    def _build_diff_tree(
        self,
        path: str,
        old_val: Any,
        new_val: Any,
    ) -> DiffTreeNode:
        """Recursively build diff tree from two values."""
        key = path.split(".")[-1] if "." in path else path

        # Handle None cases
        if old_val is None and new_val is None:
            return DiffTreeNode(path, key, DiffType.UNCHANGED)
        if old_val is None:
            return DiffTreeNode(path, key, DiffType.ADDED, new_value=new_val)
        if new_val is None:
            return DiffTreeNode(path, key, DiffType.REMOVED, old_value=old_val)

        # Handle dict comparison
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            node = DiffTreeNode(path, key)
            all_keys = set(old_val.keys()) | set(new_val.keys())
            has_changes = False

            for k in sorted(all_keys):
                child_path = f"{path}.{k}" if path else k
                child = self._build_diff_tree(
                    child_path,
                    old_val.get(k),
                    new_val.get(k),
                )
                node.children.append(child)
                if child.diff_type != DiffType.UNCHANGED or child.has_changes:
                    has_changes = True

            node.diff_type = DiffType.MODIFIED if has_changes else DiffType.UNCHANGED
            return node

        # Handle list comparison
        if isinstance(old_val, list) and isinstance(new_val, list):
            if old_val == new_val:
                return DiffTreeNode(path, key, DiffType.UNCHANGED, old_val, new_val)
            node = DiffTreeNode(path, key, DiffType.MODIFIED)
            max_len = max(len(old_val), len(new_val))
            for i in range(max_len):
                child_path = f"{path}[{i}]"
                old_item = old_val[i] if i < len(old_val) else None
                new_item = new_val[i] if i < len(new_val) else None
                child = self._build_diff_tree(child_path, old_item, new_item)
                node.children.append(child)
            return node

        # Handle primitive comparison
        if old_val == new_val:
            return DiffTreeNode(path, key, DiffType.UNCHANGED, old_val, new_val)
        return DiffTreeNode(path, key, DiffType.MODIFIED, old_val, new_val)

    def toggle_path_expansion(self, path: str) -> None:
        """Toggle expansion state of a path."""
        if path in self._expanded_paths:
            self._expanded_paths.discard(path)
        else:
            self._expanded_paths.add(path)

    def expand_all(self) -> None:
        """Expand all nodes."""
        for entry in self.entries:
            self._expanded_paths.add(entry.path)
        for widget in self._entry_widgets:
            widget.expanded = True

    def collapse_all(self) -> None:
        """Collapse all nodes."""
        self._expanded_paths.clear()
        for widget in self._entry_widgets:
            widget.expanded = False

    def set_filter_unchanged(self, filter_unchanged: bool) -> None:
        """Update filter setting."""
        if filter_unchanged != self.filter_unchanged:
            self.filter_unchanged = filter_unchanged
            # Rebuild if we have a diff tree
            if self._diff_tree:
                self._rebuild_from_tree()

    def _rebuild_from_tree(self) -> None:
        """Rebuild entries from existing diff tree."""
        if not self._diff_tree:
            return

        self.entries.clear()
        self._entry_widgets.clear()
        self.clear_children()
        self._reset_stats()

        for entry in self._diff_tree.iter_entries(self.max_depth):
            if self.filter_unchanged and entry.diff_type == DiffType.UNCHANGED:
                continue
            self.entries.append(entry)

            if entry.diff_type == DiffType.ADDED:
                self._stats["added"] += 1
            elif entry.diff_type == DiffType.REMOVED:
                self._stats["removed"] += 1
            elif entry.diff_type == DiffType.MODIFIED:
                self._stats["modified"] += 1
            else:
                self._stats["unchanged"] += 1

            widget = DiffEntryWidget(entry)
            widget.expanded = entry.path in self._expanded_paths
            self._entry_widgets.append(widget)
            self.add_child(widget)

    def render(self, ctx: DebugUIContext) -> None:
        """Render the diff view."""
        if not self.visible:
            return

        # Draw header with stats
        ctx._draw_commands.append(
            {
                "type": "diff_view_header",
                "id": self.id,
                "tick_a": self.tick_a,
                "tick_b": self.tick_b,
                "stats": self._stats,
                "x": ctx.cursor_pos.x,
                "y": ctx.cursor_pos.y,
            }
        )
        ctx.next_line()

        # Draw entries
        super().render(ctx)


# =============================================================================
# TIMELINE PANEL
# =============================================================================


@editor(category="TimeTravelUI")
@reloadable()
class TimelinePanel(ContainerWidget):
    """Visual timeline with tick markers and scrub bar.

    Shows a graphical representation of simulation history with:
    - Tick markers at regular intervals
    - Snapshot markers showing captured states
    - Current playhead position
    - Scrub bar for navigation

    Attributes:
        min_tick: Minimum tick in view.
        max_tick: Maximum tick in view.
        current_tick: Current playhead position.
        zoom_level: Zoom level (1.0 = default, higher = more zoomed in).
        scroll_offset: Horizontal scroll offset in pixels.
    """

    __slots__ = (
        "min_tick",
        "max_tick",
        "current_tick",
        "zoom_level",
        "scroll_offset",
        "on_seek",
        "_scrub_bar",
        "_playhead",
        "_snapshot_markers",
        "_tick_markers",
        "_panel_width",
        "_panel_height",
        "_ticks_per_pixel",
        "_config",
    )

    def __init__(
        self,
        min_tick: int = 0,
        max_tick: int = 0,
        panel_width: int = 600,
        panel_height: int = 80,
        ui_config: Optional[TimeTravelUIConfig] = None,
        config: Optional[WidgetConfig] = None,
    ):
        """Initialize timeline panel.

        Args:
            min_tick: Initial minimum tick.
            max_tick: Initial maximum tick.
            panel_width: Panel width in pixels.
            panel_height: Panel height in pixels.
            ui_config: Time-travel UI configuration.
            config: Widget configuration.
        """
        super().__init__(WidgetType.CUSTOM, config)
        self.min_tick = min_tick
        self.max_tick = max_tick
        self.current_tick = 0
        self.zoom_level = 1.0
        self.scroll_offset = 0.0

        self.on_seek: Optional[Callable[[int], None]] = None

        self._panel_width = panel_width
        self._panel_height = panel_height
        self._ticks_per_pixel = 1.0
        self._config = ui_config or TimeTravelUIConfig()

        # Create child widgets
        self._scrub_bar = ScrubBar(
            min_tick, max_tick, 0,
            bar_width=panel_width,
            bar_height=self._config.scrub_bar_height,
        )
        self._scrub_bar.on_seek = self._on_scrub_seek
        self.add_child(self._scrub_bar)

        self._playhead = PlayheadWidget(0)
        self.add_child(self._playhead)

        self._snapshot_markers: List[SnapshotMarkerWidget] = []
        self._tick_markers: List[TickMarkerWidget] = []

        self._update_ticks_per_pixel()

    def _update_ticks_per_pixel(self) -> None:
        """Update ticks per pixel based on range and zoom."""
        tick_range = max(1, self.max_tick - self.min_tick)
        self._ticks_per_pixel = tick_range / (self._panel_width * self.zoom_level)

    def _on_scrub_seek(self, tick: int) -> None:
        """Handle scrub bar seek."""
        self.current_tick = tick
        self._playhead.set_tick(tick)
        if self.on_seek:
            self.on_seek(tick)

    def set_range(self, min_tick: int, max_tick: int) -> None:
        """Update the visible tick range."""
        self.min_tick = min_tick
        self.max_tick = max(min_tick, max_tick)
        self._scrub_bar.set_range(min_tick, max_tick)
        self._update_ticks_per_pixel()
        self._rebuild_tick_markers()

    def set_current_tick(self, tick: int) -> None:
        """Update current tick position."""
        self.current_tick = max(self.min_tick, min(self.max_tick, tick))
        self._scrub_bar.set_current(self.current_tick)
        self._playhead.set_tick(self.current_tick)

        # Auto-scroll to keep playhead visible
        if self._config.auto_scroll_timeline:
            self._auto_scroll_to_tick(self.current_tick)

    def add_snapshot_marker(self, tick: int, snapshot_id: str = "") -> SnapshotMarkerWidget:
        """Add a snapshot marker at the given tick."""
        marker = SnapshotMarkerWidget(
            tick, snapshot_id, self._config.marker_size
        )
        self._snapshot_markers.append(marker)
        self.add_child(marker)
        return marker

    def remove_snapshot_marker(self, tick: int) -> bool:
        """Remove snapshot marker at tick."""
        for marker in self._snapshot_markers:
            if marker.tick == tick:
                self._snapshot_markers.remove(marker)
                self.remove_child(marker)
                return True
        return False

    def clear_snapshot_markers(self) -> None:
        """Remove all snapshot markers."""
        for marker in self._snapshot_markers:
            self.remove_child(marker)
        self._snapshot_markers.clear()

    def update_snapshot_markers(self, snapshot_ticks: List[int]) -> None:
        """Update snapshot markers to match given tick list."""
        # Build set of existing marker ticks
        existing = {m.tick for m in self._snapshot_markers}

        # Add missing markers
        for tick in snapshot_ticks:
            if tick not in existing:
                self.add_snapshot_marker(tick)

        # Remove extra markers
        to_remove = []
        for marker in self._snapshot_markers:
            if marker.tick not in snapshot_ticks:
                to_remove.append(marker)
        for marker in to_remove:
            self._snapshot_markers.remove(marker)
            self.remove_child(marker)

    def _rebuild_tick_markers(self) -> None:
        """Rebuild tick markers based on current range and zoom."""
        # Clear existing markers
        for marker in self._tick_markers:
            self.remove_child(marker)
        self._tick_markers.clear()

        if not self._config.show_tick_labels:
            return

        # Calculate tick interval based on zoom
        interval = self._config.tick_label_interval
        while interval / self._ticks_per_pixel < 40:  # Ensure spacing
            interval *= 2
        while interval / self._ticks_per_pixel > 200:
            interval = max(1, interval // 2)

        # Create markers
        tick = (self.min_tick // interval) * interval
        while tick <= self.max_tick:
            is_major = tick % (interval * 5) == 0
            marker = TickMarkerWidget(tick, is_major)
            self._tick_markers.append(marker)
            self.add_child(marker)
            tick += interval

    def zoom_in(self, factor: float = 1.5) -> None:
        """Zoom in on timeline."""
        self.zoom_level = min(10.0, self.zoom_level * factor)
        self._update_ticks_per_pixel()
        self._rebuild_tick_markers()

    def zoom_out(self, factor: float = 1.5) -> None:
        """Zoom out on timeline."""
        self.zoom_level = max(0.1, self.zoom_level / factor)
        self._update_ticks_per_pixel()
        self._rebuild_tick_markers()

    def zoom_to_fit(self) -> None:
        """Zoom to fit entire range in view."""
        self.zoom_level = 1.0
        self.scroll_offset = 0.0
        self._update_ticks_per_pixel()
        self._rebuild_tick_markers()

    def scroll_left(self, amount: float = 50.0) -> None:
        """Scroll left by amount in pixels."""
        self.scroll_offset = max(0.0, self.scroll_offset - amount)

    def scroll_right(self, amount: float = 50.0) -> None:
        """Scroll right by amount in pixels."""
        max_scroll = max(0.0, self._panel_width * self.zoom_level - self._panel_width)
        self.scroll_offset = min(max_scroll, self.scroll_offset + amount)

    def scroll_to_tick(self, tick: int) -> None:
        """Scroll to center given tick in view."""
        tick_pos = (tick - self.min_tick) / max(1, self._ticks_per_pixel)
        center_offset = tick_pos - self._panel_width / 2
        max_scroll = max(0.0, self._panel_width * self.zoom_level - self._panel_width)
        self.scroll_offset = max(0.0, min(max_scroll, center_offset))

    def _auto_scroll_to_tick(self, tick: int) -> None:
        """Auto-scroll to keep tick visible."""
        tick_pos = (tick - self.min_tick) / max(1, self._ticks_per_pixel)
        view_left = self.scroll_offset
        view_right = self.scroll_offset + self._panel_width

        if tick_pos < view_left + 50:
            self.scroll_offset = max(0.0, tick_pos - 50)
        elif tick_pos > view_right - 50:
            max_scroll = max(0.0, self._panel_width * self.zoom_level - self._panel_width)
            self.scroll_offset = min(max_scroll, tick_pos - self._panel_width + 50)

    def tick_to_x(self, tick: int) -> float:
        """Convert tick to x position in panel."""
        return (tick - self.min_tick) / max(1, self._ticks_per_pixel) - self.scroll_offset

    def x_to_tick(self, x: float) -> int:
        """Convert x position to tick."""
        return self.min_tick + int((x + self.scroll_offset) * self._ticks_per_pixel)

    def get_snapshot_ticks(self) -> List[int]:
        """Get list of ticks with snapshot markers."""
        return [m.tick for m in self._snapshot_markers]

    def handle_input(self, event: UIEvent) -> bool:
        """Handle input events."""
        # Forward to scrub bar
        if self._scrub_bar.handle_input(event):
            return True

        # Handle zoom with scroll
        if event.type == "scroll":
            if event.y > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            return True

        # Handle click to seek
        if event.type == "mouse_down":
            tick = self.x_to_tick(event.x)
            if self.min_tick <= tick <= self.max_tick:
                self.set_current_tick(tick)
                if self.on_seek:
                    self.on_seek(tick)
                return True

        return False

    def render(self, ctx: DebugUIContext) -> None:
        """Render the timeline panel."""
        if not self.visible:
            return

        # Draw panel background
        ctx._draw_commands.append(
            {
                "type": "timeline_panel",
                "id": self.id,
                "min_tick": self.min_tick,
                "max_tick": self.max_tick,
                "current_tick": self.current_tick,
                "zoom_level": self.zoom_level,
                "scroll_offset": self.scroll_offset,
                "width": self._panel_width,
                "height": self._panel_height,
                "x": ctx.cursor_pos.x,
                "y": ctx.cursor_pos.y,
            }
        )

        # Render children (markers, scrub bar, playhead)
        super().render(ctx)

        ctx.next_line(self._panel_height + 4)


# =============================================================================
# MAIN TIME-TRAVEL UI
# =============================================================================


@editor(category="TimeTravelUI")
@reloadable()
class TimeTravelUI:
    """Main time-travel debugging UI integrating all components.

    Combines TimelinePanel, ScrubBar, StepControls, and StateDiffView
    into a cohesive debugging interface bound to a TimeTravel instance.

    Example:
        time_travel = TimeTravel(scheduler, world, config)
        debug_ui = DebugUI()

        tt_ui = TimeTravelUI(time_travel, debug_ui)
        tt_ui.show()

        # Each frame:
        tt_ui.update(delta_time)
    """

    __slots__ = (
        "_time_travel",
        "_debug_ui",
        "_config",
        "_timeline_panel",
        "_step_controls",
        "_diff_view",
        "_playback_controller",
        "_root_section",
        "_visible",
        "_last_tick",
        "_comparison_tick_a",
        "_comparison_tick_b",
        "_pending_actions",
    )

    def __init__(
        self,
        time_travel: "TimeTravel",
        debug_ui: DebugUI,
        config: Optional[TimeTravelUIConfig] = None,
    ):
        """Initialize time-travel UI.

        Args:
            time_travel: The TimeTravel instance to bind to.
            debug_ui: The DebugUI manager.
            config: UI configuration.
        """
        self._time_travel = time_travel
        self._debug_ui = debug_ui
        self._config = config or TimeTravelUIConfig()
        self._visible = False
        self._last_tick = -1
        self._comparison_tick_a = 0
        self._comparison_tick_b = 0
        self._pending_actions: List[UIAction] = []

        # Create playback controller
        self._playback_controller = PlaybackController()

        # Create UI components
        self._create_ui()

        # Register event handlers
        self._register_event_handlers()

    @property
    def visible(self) -> bool:
        """Whether UI is visible."""
        return self._visible

    @property
    def config(self) -> TimeTravelUIConfig:
        """UI configuration."""
        return self._config

    @property
    def timeline_panel(self) -> TimelinePanel:
        """Timeline panel component."""
        return self._timeline_panel

    @property
    def step_controls(self) -> StepControls:
        """Step controls component."""
        return self._step_controls

    @property
    def diff_view(self) -> StateDiffView:
        """State diff view component."""
        return self._diff_view

    @property
    def playback_controller(self) -> PlaybackController:
        """Playback controller."""
        return self._playback_controller

    def _create_ui(self) -> None:
        """Create all UI components."""
        # Root section
        self._root_section = self._debug_ui.create_section(
            "Time Travel Debugger", expanded=True
        )

        # Timeline panel
        if self._config.show_timeline:
            self._timeline_panel = TimelinePanel(
                panel_height=self._config.timeline_height,
                ui_config=self._config,
            )
            self._timeline_panel.on_seek = self._on_seek
            self._root_section.add_child(self._timeline_panel)

        # Step controls
        if self._config.show_step_controls:
            self._step_controls = StepControls(
                step_sizes=self._config.step_sizes,
            )
            self._step_controls.on_step = self._on_step
            self._step_controls.on_play_pause = self._on_play_pause
            self._step_controls.on_stop = self._on_stop
            self._step_controls.on_record_toggle = self._on_record_toggle
            self._root_section.add_child(self._step_controls)

        # Diff view
        if self._config.show_diff_view:
            self._diff_view = StateDiffView(
                max_depth=self._config.diff_max_depth,
            )
            self._root_section.add_child(self._diff_view)

    def _register_event_handlers(self) -> None:
        """Register handlers for TimeTravel events."""
        from engine.debug.time_travel import TimeTravelEvent

        def on_event(event: "TimeTravelEvent", data: Any) -> None:
            if event == TimeTravelEvent.SNAPSHOT_CAPTURED:
                self._on_snapshot_captured(data)
            elif event == TimeTravelEvent.STATE_CHANGED:
                self._on_time_travel_state_changed(data)
            elif event == TimeTravelEvent.SEEK_COMPLETED:
                self._on_seek_completed(data)

        self._time_travel.on_event(on_event)

    def _on_snapshot_captured(self, snapshot: "TickSnapshot") -> None:
        """Handle snapshot captured event."""
        if self._config.show_timeline and self._config.highlight_snapshots:
            self._timeline_panel.add_snapshot_marker(snapshot.tick)

    def _on_time_travel_state_changed(self, state: "TimeTravelState") -> None:
        """Handle time travel state change."""
        from engine.debug.time_travel import TimeTravelState

        if state == TimeTravelState.RECORDING:
            self._step_controls.set_recording(True)
        else:
            self._step_controls.set_recording(False)

    def _on_seek_completed(self, tick: int) -> None:
        """Handle seek completed event."""
        self._update_current_tick(tick)

    def _on_seek(self, tick: int) -> None:
        """Handle seek action from UI."""
        self._time_travel.seek_to_tick(tick)

    def _on_step(self, step: int) -> None:
        """Handle step action from UI."""
        if step > 0:
            self._time_travel.step_forward(step)
        else:
            self._time_travel.step_backward(-step)

    def _on_play_pause(self) -> None:
        """Handle play/pause toggle."""
        self._playback_controller.toggle_play_pause()
        self._step_controls.set_playing(self._playback_controller.is_playing)

    def _on_stop(self) -> None:
        """Handle stop action."""
        self._playback_controller.stop()
        self._step_controls.set_playing(False)

    def _on_record_toggle(self, recording: bool) -> None:
        """Handle record toggle."""
        if recording:
            self._time_travel.enable()
        else:
            self._time_travel.pause()

    def _update_current_tick(self, tick: int) -> None:
        """Update UI for current tick."""
        if self._config.show_timeline:
            self._timeline_panel.set_current_tick(tick)

        # Update diff view if tick changed
        if tick != self._last_tick and self._config.show_diff_view:
            self._update_diff_view(self._last_tick, tick)

        self._last_tick = tick

    def _update_diff_view(self, old_tick: int, new_tick: int) -> None:
        """Update diff view to compare old and new ticks."""
        if old_tick < 0:
            return

        comparison = self._time_travel.compare_ticks(old_tick, new_tick)
        if comparison:
            self._diff_view.set_comparison(old_tick, new_tick, comparison)
            self._comparison_tick_a = old_tick
            self._comparison_tick_b = new_tick

    def show(self) -> None:
        """Show the time-travel UI."""
        self._visible = True
        self._root_section.visible = True
        self._sync_from_time_travel()

    def hide(self) -> None:
        """Hide the time-travel UI."""
        self._visible = False
        self._root_section.visible = False

    def toggle(self) -> None:
        """Toggle UI visibility."""
        if self._visible:
            self.hide()
        else:
            self.show()

    def _sync_from_time_travel(self) -> None:
        """Sync UI state from TimeTravel instance."""
        # Update range
        available = self._time_travel.available_range
        if available and self._config.show_timeline:
            self._timeline_panel.set_range(available.start, available.end)
            self._timeline_panel.set_current_tick(self._time_travel.current_tick)

            # Sync snapshot markers
            snapshot_ticks = []
            for snapshot in self._time_travel._buffer.iter_snapshots():
                snapshot_ticks.append(snapshot.tick)
            self._timeline_panel.update_snapshot_markers(snapshot_ticks)

        # Update recording state
        if self._config.show_step_controls:
            self._step_controls.set_recording(self._time_travel.is_recording)

    def update(self, delta_time: float = 0.0) -> None:
        """Update UI state each frame.

        Args:
            delta_time: Time elapsed since last frame in seconds.
        """
        if not self._visible:
            return

        # Update playback
        if self._playback_controller.is_playing:
            ticks_to_step = self._playback_controller.update(delta_time)
            if ticks_to_step != 0:
                if ticks_to_step > 0:
                    self._time_travel.step_forward(ticks_to_step)
                else:
                    self._time_travel.step_backward(-ticks_to_step)

        # Sync current tick
        current = self._time_travel.current_tick
        if current != self._last_tick:
            self._update_current_tick(current)

        # Sync range if recording
        if self._time_travel.is_recording:
            available = self._time_travel.available_range
            if available and self._config.show_timeline:
                self._timeline_panel.set_range(available.start, available.end)

    def compare_ticks(self, tick_a: int, tick_b: int) -> None:
        """Manually compare two ticks in diff view."""
        comparison = self._time_travel.compare_ticks(tick_a, tick_b)
        if comparison and self._config.show_diff_view:
            self._diff_view.set_comparison(tick_a, tick_b, comparison)
            self._comparison_tick_a = tick_a
            self._comparison_tick_b = tick_b

    def seek_to_tick(self, tick: int) -> bool:
        """Seek to specific tick."""
        return self._time_travel.seek_to_tick(tick)

    def step_forward(self, ticks: int = 1) -> bool:
        """Step forward by ticks."""
        return self._time_travel.step_forward(ticks)

    def step_backward(self, ticks: int = 1) -> bool:
        """Step backward by ticks."""
        return self._time_travel.step_backward(ticks)

    def capture_snapshot(self) -> None:
        """Manually capture a snapshot."""
        self._time_travel.capture_snapshot()

    def set_playback_speed(self, speed: float) -> None:
        """Set playback speed multiplier."""
        self._playback_controller.speed = speed

    def zoom_in(self) -> None:
        """Zoom in on timeline."""
        if self._config.show_timeline:
            self._timeline_panel.zoom_in()

    def zoom_out(self) -> None:
        """Zoom out on timeline."""
        if self._config.show_timeline:
            self._timeline_panel.zoom_out()

    def zoom_to_fit(self) -> None:
        """Zoom timeline to fit all history."""
        if self._config.show_timeline:
            self._timeline_panel.zoom_to_fit()

    def get_diff_stats(self) -> Dict[str, int]:
        """Get current diff statistics."""
        if self._config.show_diff_view:
            return self._diff_view.stats
        return {}

    def render(self) -> List[Dict[str, Any]]:
        """Render the UI and return draw commands."""
        if not self._visible:
            return []
        return self._debug_ui.render()

    def handle_event(self, event: UIEvent) -> bool:
        """Handle input event."""
        if not self._visible:
            return False

        # Handle timeline input
        if self._config.show_timeline:
            if self._timeline_panel.handle_input(event):
                return True

        # Handle keyboard shortcuts
        if event.type == "key_down":
            return self._handle_key(event.key, event.modifiers)

        return False

    def _handle_key(self, key: str, modifiers: Set[str]) -> bool:
        """Handle keyboard input."""
        # Step controls
        if key == "Left":
            step = self._step_controls.current_step_size if "shift" not in modifiers else 1
            self._on_step(-step)
            return True
        elif key == "Right":
            step = self._step_controls.current_step_size if "shift" not in modifiers else 1
            self._on_step(step)
            return True
        elif key == "Space":
            self._on_play_pause()
            return True
        elif key == "Home":
            self.seek_to_tick(self._timeline_panel.min_tick)
            return True
        elif key == "End":
            self.seek_to_tick(self._timeline_panel.max_tick)
            return True
        elif key == "Plus" or key == "=":
            self.zoom_in()
            return True
        elif key == "Minus" or key == "-":
            self.zoom_out()
            return True
        elif key == "0":
            self.zoom_to_fit()
            return True

        return False

    def dispose(self) -> None:
        """Clean up resources."""
        self._playback_controller.stop()
        self._debug_ui.remove_panel("time_travel")
