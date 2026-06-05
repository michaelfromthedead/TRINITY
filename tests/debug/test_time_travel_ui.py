"""Tests for T-CC-4.3: Time-travel UI components.

Tests for TimelinePanel, ScrubBar, StateDiffView, StepControls,
and the main TimeTravelUI integration.
"""
from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

# Import test subjects
from engine.debug.time_travel_ui import (
    # Configuration
    TimeTravelUIConfig,
    # Main UI components
    TimeTravelUI,
    TimelinePanel,
    ScrubBar,
    StepControls,
    StateDiffView,
    # Timeline widgets
    SnapshotMarkerWidget,
    TickMarkerWidget,
    PlayheadWidget,
    # Diff widgets
    DiffEntry,
    DiffEntryWidget,
    DiffTreeNode,
    DiffType,
    # Events
    UIActionType,
    UIAction,
    # Playback
    PlaybackState,
    PlaybackController,
)

from engine.tooling.editor.debug_ui import (
    DebugUI,
    DebugUIContext,
    UIEvent,
    UIState,
    Vec2,
    Color,
    WidgetConfig,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def ui_config() -> TimeTravelUIConfig:
    """Create default UI configuration."""
    return TimeTravelUIConfig()


@pytest.fixture
def custom_ui_config() -> TimeTravelUIConfig:
    """Create custom UI configuration."""
    return TimeTravelUIConfig(
        show_timeline=True,
        show_diff_view=True,
        show_step_controls=True,
        step_sizes=(1, 5, 10, 30),
        timeline_height=100,
        scrub_bar_height=30,
        marker_size=10,
        diff_max_depth=5,
        auto_scroll_timeline=False,
        highlight_snapshots=False,
        tick_label_interval=30,
    )


@pytest.fixture
def debug_ui_context() -> DebugUIContext:
    """Create debug UI context."""
    return DebugUIContext(width=800, height=600)


@pytest.fixture
def debug_ui() -> DebugUI:
    """Create debug UI manager."""
    return DebugUI(width=800, height=600)


@pytest.fixture
def mock_time_travel() -> MagicMock:
    """Create mock TimeTravel instance."""
    mock = MagicMock()
    mock.current_tick = 100
    mock.is_enabled = True
    mock.is_recording = True
    mock.available_range = MagicMock()
    mock.available_range.start = 0
    mock.available_range.end = 1000
    mock._buffer = MagicMock()
    mock._buffer.iter_snapshots.return_value = []
    mock.compare_ticks.return_value = None
    mock.on_event = MagicMock()
    return mock


@pytest.fixture
def scrub_bar() -> ScrubBar:
    """Create scrub bar with test range."""
    return ScrubBar(
        min_tick=0,
        max_tick=1000,
        current_tick=100,
        bar_width=400,
        bar_height=24,
    )


@pytest.fixture
def timeline_panel(ui_config: TimeTravelUIConfig) -> TimelinePanel:
    """Create timeline panel."""
    return TimelinePanel(
        min_tick=0,
        max_tick=1000,
        panel_width=600,
        panel_height=80,
        ui_config=ui_config,
    )


@pytest.fixture
def step_controls() -> StepControls:
    """Create step controls."""
    return StepControls(step_sizes=(1, 10, 60))


@pytest.fixture
def diff_view() -> StateDiffView:
    """Create state diff view."""
    return StateDiffView(tick_a=0, tick_b=100)


@pytest.fixture
def playback_controller() -> PlaybackController:
    """Create playback controller."""
    return PlaybackController(ticks_per_second=60)


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestTimeTravelUIConfig:
    """Tests for TimeTravelUIConfig."""

    def test_default_values(self, ui_config: TimeTravelUIConfig):
        """Test default configuration values."""
        assert ui_config.show_timeline is True
        assert ui_config.show_diff_view is True
        assert ui_config.show_step_controls is True
        assert ui_config.step_sizes == (1, 10, 60, 600)
        assert ui_config.timeline_height == 80
        assert ui_config.scrub_bar_height == 24
        assert ui_config.marker_size == 8
        assert ui_config.diff_max_depth == 10
        assert ui_config.auto_scroll_timeline is True
        assert ui_config.highlight_snapshots is True

    def test_custom_values(self, custom_ui_config: TimeTravelUIConfig):
        """Test custom configuration values."""
        assert custom_ui_config.step_sizes == (1, 5, 10, 30)
        assert custom_ui_config.timeline_height == 100
        assert custom_ui_config.scrub_bar_height == 30
        assert custom_ui_config.marker_size == 10
        assert custom_ui_config.diff_max_depth == 5
        assert custom_ui_config.auto_scroll_timeline is False
        assert custom_ui_config.highlight_snapshots is False

    def test_get_default_color(self, ui_config: TimeTravelUIConfig):
        """Test getting default colors."""
        playhead_color = ui_config.get_color("playhead")
        assert isinstance(playhead_color, Color)
        assert playhead_color.r == 1.0
        assert playhead_color.g == 0.3

    def test_get_unknown_color_returns_white(self, ui_config: TimeTravelUIConfig):
        """Test getting unknown color returns white."""
        color = ui_config.get_color("unknown_color_name")
        assert color.r == 1.0
        assert color.g == 1.0
        assert color.b == 1.0

    def test_custom_colors_override_defaults(self):
        """Test custom colors override defaults."""
        custom_colors = {"playhead": Color(0.0, 1.0, 0.0, 1.0)}
        config = TimeTravelUIConfig(colors=custom_colors)
        playhead_color = config.get_color("playhead")
        assert playhead_color.g == 1.0
        assert playhead_color.r == 0.0

    def test_config_is_frozen(self, ui_config: TimeTravelUIConfig):
        """Test that config is immutable (frozen)."""
        with pytest.raises(AttributeError):
            ui_config.show_timeline = False


# =============================================================================
# SCRUB BAR TESTS
# =============================================================================


class TestScrubBar:
    """Tests for ScrubBar widget."""

    def test_initialization(self, scrub_bar: ScrubBar):
        """Test scrub bar initialization."""
        assert scrub_bar.min_tick == 0
        assert scrub_bar.max_tick == 1000
        assert scrub_bar.current_tick == 100

    def test_range_property(self, scrub_bar: ScrubBar):
        """Test range calculation."""
        assert scrub_bar.range == 1000

    def test_position_property(self, scrub_bar: ScrubBar):
        """Test position fraction."""
        assert scrub_bar.position == pytest.approx(0.1)

    def test_is_empty_with_valid_range(self, scrub_bar: ScrubBar):
        """Test is_empty with valid range."""
        assert scrub_bar.is_empty is False

    def test_is_empty_with_zero_range(self):
        """Test is_empty with zero range."""
        bar = ScrubBar(min_tick=100, max_tick=100)
        assert bar.is_empty is True

    def test_set_range(self, scrub_bar: ScrubBar):
        """Test setting range."""
        scrub_bar.set_range(50, 500)
        assert scrub_bar.min_tick == 50
        assert scrub_bar.max_tick == 500

    def test_set_range_clamps_current(self, scrub_bar: ScrubBar):
        """Test that set_range clamps current tick."""
        scrub_bar.current_tick = 800
        scrub_bar.set_range(0, 500)
        assert scrub_bar.current_tick == 500

    def test_set_current(self, scrub_bar: ScrubBar):
        """Test setting current tick."""
        scrub_bar.set_current(500)
        assert scrub_bar.current_tick == 500

    def test_set_current_clamps_to_min(self, scrub_bar: ScrubBar):
        """Test current tick clamping to minimum."""
        scrub_bar.set_current(-100)
        assert scrub_bar.current_tick == 0

    def test_set_current_clamps_to_max(self, scrub_bar: ScrubBar):
        """Test current tick clamping to maximum."""
        scrub_bar.set_current(2000)
        assert scrub_bar.current_tick == 1000

    def test_seek_to_fraction(self, scrub_bar: ScrubBar):
        """Test seeking by fraction."""
        tick = scrub_bar.seek_to_fraction(0.5)
        assert tick == 500
        assert scrub_bar.current_tick == 500

    def test_seek_to_fraction_with_callback(self, scrub_bar: ScrubBar):
        """Test seek callback is invoked."""
        callback_values = []
        scrub_bar.on_seek = lambda t: callback_values.append(t)
        scrub_bar.seek_to_fraction(0.25)
        assert callback_values == [250]

    def test_seek_relative_forward(self, scrub_bar: ScrubBar):
        """Test relative seeking forward."""
        scrub_bar.current_tick = 100
        tick = scrub_bar.seek_relative(50)
        assert tick == 150
        assert scrub_bar.current_tick == 150

    def test_seek_relative_backward(self, scrub_bar: ScrubBar):
        """Test relative seeking backward."""
        scrub_bar.current_tick = 100
        tick = scrub_bar.seek_relative(-50)
        assert tick == 50
        assert scrub_bar.current_tick == 50

    def test_tick_to_x(self, scrub_bar: ScrubBar):
        """Test tick to x conversion."""
        x = scrub_bar.tick_to_x(500)
        assert x == pytest.approx(200.0)  # 500/1000 * 400

    def test_x_to_tick(self, scrub_bar: ScrubBar):
        """Test x to tick conversion."""
        tick = scrub_bar.x_to_tick(200)
        assert tick == 500  # 200/400 * 1000

    def test_drag_operations(self, scrub_bar: ScrubBar):
        """Test drag start/end."""
        assert scrub_bar._dragging is False
        scrub_bar.start_drag()
        assert scrub_bar._dragging is True
        scrub_bar.end_drag()
        assert scrub_bar._dragging is False

    def test_hover_update(self, scrub_bar: ScrubBar):
        """Test hover position update."""
        tick = scrub_bar.update_hover(200)
        assert tick == 500
        assert scrub_bar._hover_tick == 500

    def test_clear_hover(self, scrub_bar: ScrubBar):
        """Test clearing hover state."""
        scrub_bar.update_hover(200)
        scrub_bar.clear_hover()
        assert scrub_bar._hover_tick == -1

    def test_render_outputs_draw_command(self, scrub_bar: ScrubBar, debug_ui_context: DebugUIContext):
        """Test render produces draw command."""
        debug_ui_context.begin_frame()
        scrub_bar.render(debug_ui_context)
        commands = debug_ui_context.end_frame()
        assert len(commands) > 0
        assert commands[0]["type"] == "scrub_bar"
        assert commands[0]["min_tick"] == 0
        assert commands[0]["max_tick"] == 1000


# =============================================================================
# TIMELINE PANEL TESTS
# =============================================================================


class TestTimelinePanel:
    """Tests for TimelinePanel widget."""

    def test_initialization(self, timeline_panel: TimelinePanel):
        """Test timeline panel initialization."""
        assert timeline_panel.min_tick == 0
        assert timeline_panel.max_tick == 1000
        assert timeline_panel.zoom_level == 1.0
        assert timeline_panel.scroll_offset == 0.0

    def test_set_range(self, timeline_panel: TimelinePanel):
        """Test setting tick range."""
        timeline_panel.set_range(100, 2000)
        assert timeline_panel.min_tick == 100
        assert timeline_panel.max_tick == 2000

    def test_set_current_tick(self, timeline_panel: TimelinePanel):
        """Test setting current tick."""
        timeline_panel.set_current_tick(500)
        assert timeline_panel.current_tick == 500

    def test_add_snapshot_marker(self, timeline_panel: TimelinePanel):
        """Test adding snapshot marker."""
        marker = timeline_panel.add_snapshot_marker(300, "snap_1")
        assert marker.tick == 300
        assert marker.snapshot_id == "snap_1"
        assert len(timeline_panel._snapshot_markers) == 1

    def test_remove_snapshot_marker(self, timeline_panel: TimelinePanel):
        """Test removing snapshot marker."""
        timeline_panel.add_snapshot_marker(300)
        assert timeline_panel.remove_snapshot_marker(300) is True
        assert len(timeline_panel._snapshot_markers) == 0

    def test_remove_nonexistent_marker(self, timeline_panel: TimelinePanel):
        """Test removing non-existent marker returns False."""
        assert timeline_panel.remove_snapshot_marker(999) is False

    def test_clear_snapshot_markers(self, timeline_panel: TimelinePanel):
        """Test clearing all markers."""
        timeline_panel.add_snapshot_marker(100)
        timeline_panel.add_snapshot_marker(200)
        timeline_panel.add_snapshot_marker(300)
        timeline_panel.clear_snapshot_markers()
        assert len(timeline_panel._snapshot_markers) == 0

    def test_update_snapshot_markers(self, timeline_panel: TimelinePanel):
        """Test updating markers to match list."""
        timeline_panel.add_snapshot_marker(100)
        timeline_panel.add_snapshot_marker(200)
        timeline_panel.update_snapshot_markers([200, 300, 400])
        ticks = timeline_panel.get_snapshot_ticks()
        assert 100 not in ticks
        assert 200 in ticks
        assert 300 in ticks
        assert 400 in ticks

    def test_zoom_in(self, timeline_panel: TimelinePanel):
        """Test zooming in."""
        initial_zoom = timeline_panel.zoom_level
        timeline_panel.zoom_in(2.0)
        assert timeline_panel.zoom_level > initial_zoom

    def test_zoom_out(self, timeline_panel: TimelinePanel):
        """Test zooming out."""
        timeline_panel.zoom_level = 2.0
        timeline_panel.zoom_out(2.0)
        assert timeline_panel.zoom_level == pytest.approx(1.0)

    def test_zoom_to_fit(self, timeline_panel: TimelinePanel):
        """Test zoom to fit."""
        timeline_panel.zoom_level = 5.0
        timeline_panel.scroll_offset = 100.0
        timeline_panel.zoom_to_fit()
        assert timeline_panel.zoom_level == 1.0
        assert timeline_panel.scroll_offset == 0.0

    def test_scroll_left(self, timeline_panel: TimelinePanel):
        """Test scrolling left."""
        timeline_panel.scroll_offset = 100.0
        timeline_panel.scroll_left(50.0)
        assert timeline_panel.scroll_offset == 50.0

    def test_scroll_left_clamps_to_zero(self, timeline_panel: TimelinePanel):
        """Test scroll left clamps to zero."""
        timeline_panel.scroll_offset = 20.0
        timeline_panel.scroll_left(50.0)
        assert timeline_panel.scroll_offset == 0.0

    def test_scroll_right(self, timeline_panel: TimelinePanel):
        """Test scrolling right."""
        timeline_panel.zoom_level = 2.0  # Make room to scroll
        timeline_panel.scroll_right(50.0)
        assert timeline_panel.scroll_offset == 50.0

    def test_tick_to_x(self, timeline_panel: TimelinePanel):
        """Test tick to x conversion."""
        x = timeline_panel.tick_to_x(500)
        assert x >= 0

    def test_x_to_tick(self, timeline_panel: TimelinePanel):
        """Test x to tick conversion."""
        tick = timeline_panel.x_to_tick(300)
        assert timeline_panel.min_tick <= tick <= timeline_panel.max_tick

    def test_render_outputs_draw_command(self, timeline_panel: TimelinePanel, debug_ui_context: DebugUIContext):
        """Test render produces draw command."""
        debug_ui_context.begin_frame()
        timeline_panel.render(debug_ui_context)
        commands = debug_ui_context.end_frame()
        timeline_cmds = [c for c in commands if c.get("type") == "timeline_panel"]
        assert len(timeline_cmds) > 0

    def test_seek_callback(self, timeline_panel: TimelinePanel):
        """Test seek callback is invoked."""
        callback_values = []
        timeline_panel.on_seek = lambda t: callback_values.append(t)
        timeline_panel._on_scrub_seek(500)
        assert callback_values == [500]


# =============================================================================
# STEP CONTROLS TESTS
# =============================================================================


class TestStepControls:
    """Tests for StepControls widget."""

    def test_initialization(self, step_controls: StepControls):
        """Test step controls initialization."""
        assert step_controls.step_sizes == (1, 10, 60)
        assert step_controls.is_playing is False
        assert step_controls.is_recording is False

    def test_current_step_size(self, step_controls: StepControls):
        """Test current step size property."""
        assert step_controls.current_step_size == 1
        step_controls.current_step_size_index = 1
        assert step_controls.current_step_size == 10

    def test_step_backward_callback(self, step_controls: StepControls):
        """Test step backward triggers callback."""
        callback_values = []
        step_controls.on_step = lambda s: callback_values.append(s)
        step_controls._on_step_back(10)
        assert callback_values == [-10]

    def test_step_forward_callback(self, step_controls: StepControls):
        """Test step forward triggers callback."""
        callback_values = []
        step_controls.on_step = lambda s: callback_values.append(s)
        step_controls._on_step_forward(10)
        assert callback_values == [10]

    def test_play_pause_toggle(self, step_controls: StepControls):
        """Test play/pause toggling."""
        callback_count = [0]
        step_controls.on_play_pause = lambda: callback_count.__setitem__(0, callback_count[0] + 1)

        step_controls._on_play_pause()
        assert step_controls.is_playing is True
        assert callback_count[0] == 1

        step_controls._on_play_pause()
        assert step_controls.is_playing is False
        assert callback_count[0] == 2

    def test_stop(self, step_controls: StepControls):
        """Test stop action."""
        step_controls.is_playing = True
        callback_called = [False]
        step_controls.on_stop = lambda: callback_called.__setitem__(0, True)

        step_controls._on_stop()
        assert step_controls.is_playing is False
        assert callback_called[0] is True

    def test_record_toggle(self, step_controls: StepControls):
        """Test record toggling."""
        callback_values = []
        step_controls.on_record_toggle = lambda r: callback_values.append(r)

        step_controls._on_record_toggle()
        assert step_controls.is_recording is True
        assert callback_values == [True]

        step_controls._on_record_toggle()
        assert step_controls.is_recording is False
        assert callback_values == [True, False]

    def test_set_playing_state(self, step_controls: StepControls):
        """Test setting playing state."""
        step_controls.set_playing(True)
        assert step_controls.is_playing is True
        assert step_controls._play_button.label == "Pause"

        step_controls.set_playing(False)
        assert step_controls.is_playing is False
        assert step_controls._play_button.label == "Play"

    def test_set_recording_state(self, step_controls: StepControls):
        """Test setting recording state."""
        step_controls.set_recording(True)
        assert step_controls.is_recording is True
        assert step_controls._record_button.label == "Stop Rec"

    def test_step_backward_method(self, step_controls: StepControls):
        """Test step_backward convenience method."""
        callback_values = []
        step_controls.on_step = lambda s: callback_values.append(s)
        step_controls.step_backward()
        assert callback_values == [-1]

    def test_step_forward_method(self, step_controls: StepControls):
        """Test step_forward convenience method."""
        callback_values = []
        step_controls.on_step = lambda s: callback_values.append(s)
        step_controls.step_forward()
        assert callback_values == [1]


# =============================================================================
# PLAYBACK CONTROLLER TESTS
# =============================================================================


class TestPlaybackController:
    """Tests for PlaybackController."""

    def test_initialization(self, playback_controller: PlaybackController):
        """Test playback controller initialization."""
        assert playback_controller.state == PlaybackState.STOPPED
        assert playback_controller.speed == 1.0
        assert playback_controller.loop is False

    def test_play_forward(self, playback_controller: PlaybackController):
        """Test playing forward."""
        playback_controller.play_forward()
        assert playback_controller.state == PlaybackState.PLAYING_FORWARD
        assert playback_controller.is_playing is True

    def test_play_backward(self, playback_controller: PlaybackController):
        """Test playing backward."""
        playback_controller.play_backward()
        assert playback_controller.state == PlaybackState.PLAYING_BACKWARD
        assert playback_controller.is_playing is True

    def test_pause(self, playback_controller: PlaybackController):
        """Test pausing."""
        playback_controller.play_forward()
        playback_controller.pause()
        assert playback_controller.state == PlaybackState.PAUSED
        assert playback_controller.is_paused is True

    def test_resume(self, playback_controller: PlaybackController):
        """Test resuming from pause."""
        playback_controller.play_forward()
        playback_controller.pause()
        playback_controller.resume()
        assert playback_controller.state == PlaybackState.PLAYING_FORWARD

    def test_stop(self, playback_controller: PlaybackController):
        """Test stopping."""
        playback_controller.play_forward()
        playback_controller.stop()
        assert playback_controller.state == PlaybackState.STOPPED
        assert playback_controller.is_stopped is True

    def test_toggle_play_pause(self, playback_controller: PlaybackController):
        """Test toggle play/pause."""
        playback_controller.toggle_play_pause()
        assert playback_controller.is_playing is True

        playback_controller.toggle_play_pause()
        assert playback_controller.is_paused is True

        playback_controller.toggle_play_pause()
        assert playback_controller.is_playing is True

    def test_speed_clamping(self, playback_controller: PlaybackController):
        """Test speed is clamped."""
        playback_controller.speed = 0.01
        assert playback_controller.speed >= 0.1

        playback_controller.speed = 100.0
        assert playback_controller.speed <= 10.0

    def test_update_returns_ticks_when_playing(self, playback_controller: PlaybackController):
        """Test update returns ticks when playing."""
        playback_controller.play_forward()
        playback_controller.speed = 60.0  # Fast playback
        ticks = playback_controller.update(1.0)  # 1 second
        assert ticks > 0

    def test_update_returns_zero_when_stopped(self, playback_controller: PlaybackController):
        """Test update returns zero when stopped."""
        ticks = playback_controller.update(1.0)
        assert ticks == 0

    def test_update_returns_negative_when_backward(self, playback_controller: PlaybackController):
        """Test update returns negative ticks when playing backward."""
        playback_controller.play_backward()
        playback_controller.speed = 60.0
        ticks = playback_controller.update(1.0)
        assert ticks < 0

    def test_reset(self, playback_controller: PlaybackController):
        """Test reset."""
        playback_controller.play_forward()
        playback_controller.speed = 2.0
        playback_controller.reset()
        assert playback_controller.state == PlaybackState.STOPPED


# =============================================================================
# DIFF ENTRY TESTS
# =============================================================================


class TestDiffEntry:
    """Tests for DiffEntry."""

    def test_key_from_path(self):
        """Test extracting key from path."""
        entry = DiffEntry(
            path="entities.42.position.x",
            diff_type=DiffType.MODIFIED,
        )
        assert entry.key == "x"

    def test_key_from_simple_path(self):
        """Test key from simple path."""
        entry = DiffEntry(path="health", diff_type=DiffType.ADDED)
        assert entry.key == "health"

    def test_is_changed_for_modified(self):
        """Test is_changed for modified entry."""
        entry = DiffEntry(path="x", diff_type=DiffType.MODIFIED)
        assert entry.is_changed is True

    def test_is_changed_for_unchanged(self):
        """Test is_changed for unchanged entry."""
        entry = DiffEntry(path="x", diff_type=DiffType.UNCHANGED)
        assert entry.is_changed is False


class TestDiffTreeNode:
    """Tests for DiffTreeNode."""

    def test_is_leaf(self):
        """Test is_leaf property."""
        leaf = DiffTreeNode(path="x", key="x")
        assert leaf.is_leaf is True

        parent = DiffTreeNode(path="obj", key="obj", children=[leaf])
        assert parent.is_leaf is False

    def test_has_changes(self):
        """Test has_changes property."""
        unchanged = DiffTreeNode(path="x", key="x", diff_type=DiffType.UNCHANGED)
        assert unchanged.has_changes is False

        modified = DiffTreeNode(path="y", key="y", diff_type=DiffType.MODIFIED)
        assert modified.has_changes is True

    def test_has_changes_with_changed_children(self):
        """Test has_changes with changed children."""
        child = DiffTreeNode(path="x.y", key="y", diff_type=DiffType.ADDED)
        parent = DiffTreeNode(path="x", key="x", diff_type=DiffType.UNCHANGED, children=[child])
        assert parent.has_changes is True

    def test_iter_entries(self):
        """Test iterating over entries."""
        node = DiffTreeNode(
            path="root",
            key="root",
            diff_type=DiffType.MODIFIED,
            old_value=1,
            new_value=2,
        )
        entries = list(node.iter_entries())
        assert len(entries) == 1
        assert entries[0].path == "root"


# =============================================================================
# STATE DIFF VIEW TESTS
# =============================================================================


class TestStateDiffView:
    """Tests for StateDiffView widget."""

    def test_initialization(self, diff_view: StateDiffView):
        """Test diff view initialization."""
        assert diff_view.tick_a == 0
        assert diff_view.tick_b == 100
        assert diff_view.filter_unchanged is True
        assert diff_view.max_depth == 10

    def test_initial_stats(self, diff_view: StateDiffView):
        """Test initial stats are zero."""
        stats = diff_view.stats
        assert stats["added"] == 0
        assert stats["removed"] == 0
        assert stats["modified"] == 0
        assert stats["unchanged"] == 0

    def test_has_changes_empty(self, diff_view: StateDiffView):
        """Test has_changes when empty."""
        assert diff_view.has_changes is False

    def test_total_changes_empty(self, diff_view: StateDiffView):
        """Test total_changes when empty."""
        assert diff_view.total_changes == 0

    def test_set_diff_data_added(self, diff_view: StateDiffView):
        """Test setting diff data with added values."""
        state_a = {"x": 1}
        state_b = {"x": 1, "y": 2}
        diff_view.set_diff_data(state_a, state_b)
        assert diff_view.stats["added"] > 0
        assert diff_view.has_changes is True

    def test_set_diff_data_removed(self, diff_view: StateDiffView):
        """Test setting diff data with removed values."""
        state_a = {"x": 1, "y": 2}
        state_b = {"x": 1}
        diff_view.set_diff_data(state_a, state_b)
        assert diff_view.stats["removed"] > 0

    def test_set_diff_data_modified(self, diff_view: StateDiffView):
        """Test setting diff data with modified values."""
        state_a = {"x": 1}
        state_b = {"x": 2}
        diff_view.set_diff_data(state_a, state_b)
        assert diff_view.stats["modified"] > 0

    def test_set_diff_data_unchanged(self, diff_view: StateDiffView):
        """Test setting diff data with unchanged values."""
        diff_view.filter_unchanged = False
        state_a = {"x": 1}
        state_b = {"x": 1}
        diff_view.set_diff_data(state_a, state_b)
        assert diff_view.stats["unchanged"] > 0

    def test_set_diff_data_nested(self, diff_view: StateDiffView):
        """Test setting diff data with nested structures."""
        state_a = {"obj": {"x": 1, "y": 2}}
        state_b = {"obj": {"x": 1, "y": 3}}
        diff_view.set_diff_data(state_a, state_b)
        assert diff_view.has_changes is True

    def test_set_diff_data_lists(self, diff_view: StateDiffView):
        """Test setting diff data with lists."""
        state_a = {"items": [1, 2, 3]}
        state_b = {"items": [1, 2, 4]}
        diff_view.set_diff_data(state_a, state_b)
        assert diff_view.has_changes is True

    def test_expand_all(self, diff_view: StateDiffView):
        """Test expanding all nodes."""
        state_a = {"obj": {"x": 1}}
        state_b = {"obj": {"x": 2}}
        diff_view.set_diff_data(state_a, state_b)
        diff_view.expand_all()
        for widget in diff_view._entry_widgets:
            assert widget.expanded is True

    def test_collapse_all(self, diff_view: StateDiffView):
        """Test collapsing all nodes."""
        state_a = {"obj": {"x": 1}}
        state_b = {"obj": {"x": 2}}
        diff_view.set_diff_data(state_a, state_b)
        diff_view.collapse_all()
        for widget in diff_view._entry_widgets:
            assert widget.expanded is False

    def test_toggle_path_expansion(self, diff_view: StateDiffView):
        """Test toggling path expansion."""
        diff_view.toggle_path_expansion("test.path")
        assert "test.path" in diff_view._expanded_paths
        diff_view.toggle_path_expansion("test.path")
        assert "test.path" not in diff_view._expanded_paths

    def test_set_filter_unchanged(self, diff_view: StateDiffView):
        """Test changing filter setting."""
        state_a = {"x": 1}
        state_b = {"x": 1}
        diff_view.filter_unchanged = False
        diff_view.set_diff_data(state_a, state_b)
        initial_count = len(diff_view.entries)

        diff_view.set_filter_unchanged(True)
        diff_view._rebuild_from_tree()
        assert len(diff_view.entries) < initial_count


# =============================================================================
# MARKER WIDGET TESTS
# =============================================================================


class TestTickMarkerWidget:
    """Tests for TickMarkerWidget."""

    def test_initialization(self):
        """Test tick marker initialization."""
        marker = TickMarkerWidget(tick=100, is_major=True)
        assert marker.tick == 100
        assert marker.is_major is True

    def test_render(self, debug_ui_context: DebugUIContext):
        """Test tick marker render."""
        marker = TickMarkerWidget(tick=100)
        debug_ui_context.begin_frame()
        marker.render(debug_ui_context)
        commands = debug_ui_context.end_frame()
        assert len(commands) > 0
        assert commands[0]["type"] == "tick_marker"


class TestSnapshotMarkerWidget:
    """Tests for SnapshotMarkerWidget."""

    def test_initialization(self):
        """Test snapshot marker initialization."""
        marker = SnapshotMarkerWidget(tick=200, snapshot_id="snap_1")
        assert marker.tick == 200
        assert marker.snapshot_id == "snap_1"

    def test_default_snapshot_id(self):
        """Test default snapshot ID generation."""
        marker = SnapshotMarkerWidget(tick=300)
        assert "300" in marker.snapshot_id

    def test_highlight(self):
        """Test highlight property."""
        marker = SnapshotMarkerWidget(tick=100, highlight=True)
        assert marker.highlight is True


class TestPlayheadWidget:
    """Tests for PlayheadWidget."""

    def test_initialization(self):
        """Test playhead initialization."""
        playhead = PlayheadWidget(tick=150)
        assert playhead.tick == 150
        assert playhead.dragging is False

    def test_set_tick(self):
        """Test setting tick."""
        playhead = PlayheadWidget()
        playhead.set_tick(500)
        assert playhead.tick == 500

    def test_set_tick_clamps_negative(self):
        """Test setting tick clamps negative values."""
        playhead = PlayheadWidget()
        playhead.set_tick(-100)
        assert playhead.tick == 0

    def test_drag_operations(self):
        """Test drag start/end."""
        playhead = PlayheadWidget(tick=100)
        playhead.start_drag()
        assert playhead.dragging is True
        assert playhead._drag_start_tick == 100

        playhead.set_tick(200)
        final_tick = playhead.end_drag()
        assert playhead.dragging is False
        assert final_tick == 200

    def test_cancel_drag(self):
        """Test cancelling drag."""
        playhead = PlayheadWidget(tick=100)
        playhead.start_drag()
        playhead.set_tick(200)
        playhead.cancel_drag()
        assert playhead.tick == 100
        assert playhead.dragging is False


# =============================================================================
# UI ACTION TESTS
# =============================================================================


class TestUIAction:
    """Tests for UIAction."""

    def test_seek_action(self):
        """Test creating seek action."""
        action = UIAction(UIActionType.SEEK_TO_TICK, tick=500)
        assert action.action_type == UIActionType.SEEK_TO_TICK
        assert action.tick == 500

    def test_step_action(self):
        """Test creating step action."""
        action = UIAction(UIActionType.STEP_FORWARD, step_size=10)
        assert action.action_type == UIActionType.STEP_FORWARD
        assert action.step_size == 10

    def test_speed_action(self):
        """Test creating speed action."""
        action = UIAction(UIActionType.SET_PLAYBACK_SPEED, speed=2.0)
        assert action.speed == 2.0

    def test_compare_action(self):
        """Test creating compare action."""
        action = UIAction(UIActionType.COMPARE_TICKS, tick=100, tick_b=200)
        assert action.tick == 100
        assert action.tick_b == 200


# =============================================================================
# MAIN TIME-TRAVEL UI TESTS
# =============================================================================


class TestTimeTravelUI:
    """Tests for TimeTravelUI main class."""

    def test_initialization(
        self,
        mock_time_travel: MagicMock,
        debug_ui: DebugUI,
        ui_config: TimeTravelUIConfig,
    ):
        """Test TimeTravelUI initialization."""
        ui = TimeTravelUI(mock_time_travel, debug_ui, ui_config)
        assert ui.visible is False
        assert ui.config == ui_config

    def test_show_hide(
        self,
        mock_time_travel: MagicMock,
        debug_ui: DebugUI,
    ):
        """Test show/hide functionality."""
        ui = TimeTravelUI(mock_time_travel, debug_ui)

        ui.show()
        assert ui.visible is True

        ui.hide()
        assert ui.visible is False

    def test_toggle(
        self,
        mock_time_travel: MagicMock,
        debug_ui: DebugUI,
    ):
        """Test toggle functionality."""
        ui = TimeTravelUI(mock_time_travel, debug_ui)

        ui.toggle()
        assert ui.visible is True

        ui.toggle()
        assert ui.visible is False

    def test_seek_to_tick(
        self,
        mock_time_travel: MagicMock,
        debug_ui: DebugUI,
    ):
        """Test seeking to tick."""
        ui = TimeTravelUI(mock_time_travel, debug_ui)
        ui.seek_to_tick(500)
        mock_time_travel.seek_to_tick.assert_called_with(500)

    def test_step_forward(
        self,
        mock_time_travel: MagicMock,
        debug_ui: DebugUI,
    ):
        """Test stepping forward."""
        ui = TimeTravelUI(mock_time_travel, debug_ui)
        ui.step_forward(10)
        mock_time_travel.step_forward.assert_called_with(10)

    def test_step_backward(
        self,
        mock_time_travel: MagicMock,
        debug_ui: DebugUI,
    ):
        """Test stepping backward."""
        ui = TimeTravelUI(mock_time_travel, debug_ui)
        ui.step_backward(10)
        mock_time_travel.step_backward.assert_called_with(10)

    def test_capture_snapshot(
        self,
        mock_time_travel: MagicMock,
        debug_ui: DebugUI,
    ):
        """Test capturing snapshot."""
        ui = TimeTravelUI(mock_time_travel, debug_ui)
        ui.capture_snapshot()
        mock_time_travel.capture_snapshot.assert_called_once()

    def test_set_playback_speed(
        self,
        mock_time_travel: MagicMock,
        debug_ui: DebugUI,
    ):
        """Test setting playback speed."""
        ui = TimeTravelUI(mock_time_travel, debug_ui)
        ui.set_playback_speed(2.0)
        assert ui.playback_controller.speed == 2.0

    def test_zoom_controls(
        self,
        mock_time_travel: MagicMock,
        debug_ui: DebugUI,
    ):
        """Test zoom controls."""
        ui = TimeTravelUI(mock_time_travel, debug_ui)
        ui.show()

        initial_zoom = ui.timeline_panel.zoom_level
        ui.zoom_in()
        assert ui.timeline_panel.zoom_level > initial_zoom

        ui.zoom_out()
        ui.zoom_to_fit()
        assert ui.timeline_panel.zoom_level == 1.0

    def test_compare_ticks(
        self,
        mock_time_travel: MagicMock,
        debug_ui: DebugUI,
    ):
        """Test comparing ticks."""
        ui = TimeTravelUI(mock_time_travel, debug_ui)
        ui.show()
        ui.compare_ticks(100, 200)
        mock_time_travel.compare_ticks.assert_called_with(100, 200)

    def test_update_when_hidden(
        self,
        mock_time_travel: MagicMock,
        debug_ui: DebugUI,
    ):
        """Test update does nothing when hidden."""
        ui = TimeTravelUI(mock_time_travel, debug_ui)
        # Should not raise or have side effects
        ui.update(0.016)

    def test_dispose(
        self,
        mock_time_travel: MagicMock,
        debug_ui: DebugUI,
    ):
        """Test dispose cleans up resources."""
        ui = TimeTravelUI(mock_time_travel, debug_ui)
        ui.dispose()
        assert ui.playback_controller.is_stopped is True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for time-travel UI."""

    def test_full_workflow(
        self,
        mock_time_travel: MagicMock,
        debug_ui: DebugUI,
    ):
        """Test complete workflow: show, seek, step, compare, hide."""
        ui = TimeTravelUI(mock_time_travel, debug_ui)

        # Show UI
        ui.show()
        assert ui.visible is True

        # Seek to tick
        ui.seek_to_tick(500)
        mock_time_travel.seek_to_tick.assert_called_with(500)

        # Step forward
        ui.step_forward(10)
        mock_time_travel.step_forward.assert_called_with(10)

        # Compare ticks
        ui.compare_ticks(490, 500)
        mock_time_travel.compare_ticks.assert_called()

        # Hide UI
        ui.hide()
        assert ui.visible is False

    def test_playback_integration(
        self,
        mock_time_travel: MagicMock,
        debug_ui: DebugUI,
    ):
        """Test playback control integration."""
        ui = TimeTravelUI(mock_time_travel, debug_ui)
        ui.show()

        # Start playback
        ui.playback_controller.play_forward()
        assert ui.playback_controller.is_playing is True

        # Update should step time travel
        ui.playback_controller.speed = 60.0  # Fast
        ui.update(1.0)  # 1 second = ~60 ticks at speed 60

        # Pause
        ui.playback_controller.pause()
        assert ui.playback_controller.is_paused is True

        # Stop
        ui.playback_controller.stop()
        assert ui.playback_controller.is_stopped is True

    def test_timeline_marker_sync(
        self,
        mock_time_travel: MagicMock,
        debug_ui: DebugUI,
    ):
        """Test timeline marker synchronization."""
        # Setup mock snapshots
        mock_snapshot = MagicMock()
        mock_snapshot.tick = 100
        mock_time_travel._buffer.iter_snapshots.return_value = [mock_snapshot]

        ui = TimeTravelUI(mock_time_travel, debug_ui)
        ui.show()
        ui._sync_from_time_travel()

        # Verify marker was added
        ticks = ui.timeline_panel.get_snapshot_ticks()
        assert 100 in ticks
