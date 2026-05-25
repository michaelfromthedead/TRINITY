"""Tests for debug overlays - categories, filtering, visibility."""

import pytest
import time
from engine.tooling.debug.debug_overlays import (
    DebugOverlay,
    OverlayManager,
    OverlayPosition,
    OverlayVisibility,
    OverlayStyle,
    OverlayEntry,
    TextOverlay,
    StatsOverlay,
    GraphOverlay,
    FPSOverlay,
    MemoryOverlay,
)


class TestOverlayEntry:
    """Tests for OverlayEntry class."""

    def test_entry_creation(self):
        entry = OverlayEntry(key="fps", value=60)
        assert entry.key == "fps"
        assert entry.value == 60

    def test_entry_format(self):
        entry = OverlayEntry(key="fps", value=60)
        assert entry.format() == "fps: 60"

    def test_entry_custom_format(self):
        entry = OverlayEntry(
            key="FPS",
            value=60.5,
            format_string="{key} = {value:.1f}",
        )
        assert entry.format() == "FPS = 60.5"

    def test_entry_priority(self):
        entry1 = OverlayEntry(key="a", value=1, priority=1)
        entry2 = OverlayEntry(key="b", value=2, priority=2)
        assert entry2.priority > entry1.priority


class TestOverlayStyle:
    """Tests for OverlayStyle class."""

    def test_default_style(self):
        style = OverlayStyle()
        assert style.font_size == 12.0
        assert style.padding == 8.0

    def test_custom_style(self):
        style = OverlayStyle(
            font_size=16.0,
            padding=10.0,
            background_color=(1.0, 0.0, 0.0, 0.5),
        )
        assert style.font_size == 16.0
        assert style.padding == 10.0
        assert style.background_color == (1.0, 0.0, 0.0, 0.5)


class TestTextOverlay:
    """Tests for TextOverlay class."""

    def test_text_overlay_creation(self):
        overlay = TextOverlay(
            overlay_id="test_overlay",
            title="Test",
            position=OverlayPosition.TOP_LEFT,
        )
        assert overlay.id == "test_overlay"
        assert overlay.title == "Test"
        assert overlay.position == OverlayPosition.TOP_LEFT

    def test_text_overlay_add_entry(self):
        overlay = TextOverlay(overlay_id="test")
        entry = overlay.add_entry("key1", "value1")
        assert entry.key == "key1"
        assert len(overlay.entries) == 1

    def test_text_overlay_update_entry(self):
        overlay = TextOverlay(overlay_id="test")
        overlay.add_entry("fps", 30)
        result = overlay.update_entry("fps", 60)
        assert result is True
        assert overlay.get_entry("fps").value == 60

    def test_text_overlay_remove_entry(self):
        overlay = TextOverlay(overlay_id="test")
        overlay.add_entry("key1", "value1")
        result = overlay.remove_entry("key1")
        assert result is True
        assert len(overlay.entries) == 0

    def test_text_overlay_clear_entries(self):
        overlay = TextOverlay(overlay_id="test")
        overlay.add_entry("key1", "value1")
        overlay.add_entry("key2", "value2")
        overlay.clear_entries()
        assert len(overlay.entries) == 0

    def test_text_overlay_render(self):
        overlay = TextOverlay(
            overlay_id="test",
            title="Stats",
        )
        overlay.add_entry("FPS", 60)
        render_data = overlay.render()
        assert render_data["type"] == "text"
        assert render_data["id"] == "test"
        assert "FPS: 60" in render_data["lines"]


class TestStatsOverlay:
    """Tests for StatsOverlay class."""

    def test_stats_overlay_creation(self):
        overlay = StatsOverlay(
            overlay_id="stats",
            title="Performance",
        )
        assert overlay.id == "stats"
        assert overlay.title == "Performance"

    def test_stats_overlay_register_stat(self):
        overlay = StatsOverlay(overlay_id="stats")
        counter = [0]

        def get_counter():
            counter[0] += 1
            return counter[0]

        overlay.register_stat("counter", get_counter)
        overlay.update()
        entry = overlay.get_entry("counter")
        assert entry.value == 1

    def test_stats_overlay_unregister_stat(self):
        overlay = StatsOverlay(overlay_id="stats")
        overlay.register_stat("test", lambda: 42)
        overlay.unregister_stat("test")
        assert overlay.get_entry("test") is None

    def test_stats_overlay_update_interval(self):
        overlay = StatsOverlay(
            overlay_id="stats",
            update_interval=1.0,
        )
        # Should update first time
        assert overlay.should_update(time.time()) is True
        # Should not update immediately after
        assert overlay.should_update(time.time()) is False


class TestGraphOverlay:
    """Tests for GraphOverlay class."""

    def test_graph_overlay_creation(self):
        overlay = GraphOverlay(
            overlay_id="fps_graph",
            title="FPS",
            max_points=100,
        )
        assert overlay.id == "fps_graph"
        assert len(overlay.data_points) == 0

    def test_graph_overlay_add_data(self):
        overlay = GraphOverlay(overlay_id="graph", max_points=10)
        overlay.add_data_point(60.0)
        overlay.add_data_point(55.0)
        assert len(overlay.data_points) == 2
        assert overlay.current_value == 55.0

    def test_graph_overlay_max_points(self):
        overlay = GraphOverlay(overlay_id="graph", max_points=5)
        for i in range(10):
            overlay.add_data_point(float(i))
        assert len(overlay.data_points) == 5
        # Should have last 5 values
        assert overlay.data_points == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_graph_overlay_auto_scale(self):
        overlay = GraphOverlay(
            overlay_id="graph",
            auto_scale=True,
            min_value=0,
            max_value=100,
        )
        overlay.add_data_point(50.0)
        overlay.add_data_point(200.0)
        assert overlay._max_value == 200.0

    def test_graph_overlay_average(self):
        overlay = GraphOverlay(overlay_id="graph")
        overlay.add_data_point(10.0)
        overlay.add_data_point(20.0)
        overlay.add_data_point(30.0)
        assert overlay.average_value == 20.0

    def test_graph_overlay_clear(self):
        overlay = GraphOverlay(overlay_id="graph")
        overlay.add_data_point(10.0)
        overlay.clear_data()
        assert len(overlay.data_points) == 0

    def test_graph_overlay_render(self):
        overlay = GraphOverlay(overlay_id="graph", title="Test")
        overlay.add_data_point(50.0)
        render_data = overlay.render()
        assert render_data["type"] == "graph"
        assert render_data["title"] == "Test"
        assert 50.0 in render_data["data"]


class TestOverlayVisibility:
    """Tests for overlay visibility settings."""

    def test_visibility_always(self):
        overlay = TextOverlay(
            overlay_id="test",
            visibility=OverlayVisibility.ALWAYS,
        )
        assert overlay.is_visible() is True

    def test_visibility_hidden(self):
        overlay = TextOverlay(
            overlay_id="test",
            visibility=OverlayVisibility.HIDDEN,
        )
        assert overlay.is_visible() is False

    def test_visibility_conditional(self):
        condition_value = [True]
        overlay = TextOverlay(
            overlay_id="test",
            visibility=OverlayVisibility.CONDITIONAL,
        )
        overlay.set_condition(lambda: condition_value[0])
        assert overlay.is_visible() is True
        condition_value[0] = False
        assert overlay.is_visible() is False

    def test_visibility_toggle(self):
        overlay = TextOverlay(
            overlay_id="test",
            visibility=OverlayVisibility.TOGGLE,
        )
        assert overlay.enabled is True
        overlay.toggle()
        assert overlay.enabled is False
        overlay.toggle()
        assert overlay.enabled is True


class TestOverlayPosition:
    """Tests for overlay positioning."""

    def test_custom_position(self):
        overlay = TextOverlay(overlay_id="test")
        overlay.set_custom_position(100.0, 200.0)
        assert overlay.position == OverlayPosition.CUSTOM
        assert overlay.get_custom_position() == (100.0, 200.0)


class TestOverlayManager:
    """Tests for OverlayManager singleton."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        OverlayManager.reset_instance()
        yield
        OverlayManager.reset_instance()

    def test_singleton(self):
        manager1 = OverlayManager.get_instance()
        manager2 = OverlayManager.get_instance()
        assert manager1 is manager2

    def test_register_overlay(self):
        manager = OverlayManager.get_instance()
        overlay = TextOverlay(overlay_id="test", category="debug")
        manager.register_overlay(overlay)
        assert manager.get_overlay("test") is overlay

    def test_unregister_overlay(self):
        manager = OverlayManager.get_instance()
        overlay = TextOverlay(overlay_id="test")
        manager.register_overlay(overlay)
        removed = manager.unregister_overlay("test")
        assert removed is overlay
        assert manager.get_overlay("test") is None

    def test_enable_disable(self):
        manager = OverlayManager.get_instance()
        manager.enable()
        assert manager.is_enabled is True
        manager.disable()
        assert manager.is_enabled is False

    def test_category_management(self):
        manager = OverlayManager.get_instance()
        overlay = TextOverlay(overlay_id="test", category="physics")
        manager.register_overlay(overlay)

        manager.disable_category("physics")
        assert not manager.is_category_enabled("physics")
        assert not overlay.enabled

        manager.enable_category("physics")
        assert manager.is_category_enabled("physics")
        assert overlay.enabled

    def test_get_overlays_by_category(self):
        manager = OverlayManager.get_instance()
        overlay1 = TextOverlay(overlay_id="o1", category="physics")
        overlay2 = TextOverlay(overlay_id="o2", category="ai")
        overlay3 = TextOverlay(overlay_id="o3", category="physics")

        manager.register_overlay(overlay1)
        manager.register_overlay(overlay2)
        manager.register_overlay(overlay3)

        physics_overlays = manager.get_overlays_by_category("physics")
        assert len(physics_overlays) == 2

    def test_toggle_overlay(self):
        manager = OverlayManager.get_instance()
        overlay = TextOverlay(overlay_id="test")
        manager.register_overlay(overlay)

        new_state = manager.toggle_overlay("test")
        assert new_state is False  # Was True, now False
        new_state = manager.toggle_overlay("test")
        assert new_state is True

    def test_update_overlays(self):
        manager = OverlayManager.get_instance()
        counter = [0]

        def get_value():
            counter[0] += 1
            return counter[0]

        stats = StatsOverlay(overlay_id="stats", update_interval=0)
        stats.register_stat("counter", get_value)
        manager.register_overlay(stats)

        manager.update()
        assert counter[0] >= 1

    def test_render_all(self):
        manager = OverlayManager.get_instance()
        overlay1 = TextOverlay(overlay_id="o1")
        overlay2 = TextOverlay(overlay_id="o2")
        manager.register_overlay(overlay1)
        manager.register_overlay(overlay2)

        render_data = manager.render_all()
        assert len(render_data) == 2

    def test_render_disabled(self):
        manager = OverlayManager.get_instance()
        overlay = TextOverlay(overlay_id="test")
        manager.register_overlay(overlay)
        manager.disable()

        render_data = manager.render_all()
        assert len(render_data) == 0

    def test_clear_all(self):
        manager = OverlayManager.get_instance()
        manager.register_overlay(TextOverlay(overlay_id="o1"))
        manager.register_overlay(TextOverlay(overlay_id="o2"))
        manager.clear_all()
        assert manager.overlay_count == 0

    def test_visible_count(self):
        manager = OverlayManager.get_instance()
        overlay1 = TextOverlay(overlay_id="o1", visibility=OverlayVisibility.ALWAYS)
        overlay2 = TextOverlay(overlay_id="o2", visibility=OverlayVisibility.HIDDEN)
        manager.register_overlay(overlay1)
        manager.register_overlay(overlay2)

        assert manager.visible_count == 1

    def test_create_text_overlay(self):
        manager = OverlayManager.get_instance()
        overlay = manager.create_text_overlay("test", "Title")
        assert isinstance(overlay, TextOverlay)
        assert manager.get_overlay("test") is overlay

    def test_create_stats_overlay(self):
        manager = OverlayManager.get_instance()
        overlay = manager.create_stats_overlay("stats", "Stats")
        assert isinstance(overlay, StatsOverlay)
        assert manager.get_overlay("stats") is overlay

    def test_create_graph_overlay(self):
        manager = OverlayManager.get_instance()
        overlay = manager.create_graph_overlay("graph", "Graph")
        assert isinstance(overlay, GraphOverlay)
        assert manager.get_overlay("graph") is overlay


class TestFPSOverlay:
    """Tests for FPSOverlay class."""

    def test_fps_overlay_creation(self):
        overlay = FPSOverlay()
        assert overlay.id == "fps_overlay"

    def test_fps_overlay_record_frame(self):
        overlay = FPSOverlay()
        overlay.record_frame()
        time.sleep(0.01)
        overlay.record_frame()
        # Should have recorded frame times
        assert len(overlay._frame_times) >= 1

    def test_fps_calculation(self):
        overlay = FPSOverlay()
        # Simulate consistent frame times
        overlay._frame_times = [1/60] * 60  # 60 FPS
        fps = overlay._get_fps()
        assert abs(fps - 60.0) < 1.0


class TestMemoryOverlay:
    """Tests for MemoryOverlay class."""

    def test_memory_overlay_creation(self):
        overlay = MemoryOverlay()
        assert overlay.id == "memory_overlay"

    def test_memory_stats(self):
        overlay = MemoryOverlay()
        overlay.update()
        # Stats should be available (or N/A if psutil not installed)
        used_entry = overlay.get_entry("Used")
        assert used_entry is not None
