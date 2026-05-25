"""Tests for the profiler overlay module."""

from __future__ import annotations

import time

import pytest

from engine.tooling.profiling.profiler_overlay import (
    ProfilerOverlay,
    OverlayConfig,
    OverlayPanel,
    OverlayPosition,
    OverlayStyle,
    OverlayStats,
)


class TestOverlayConfig:
    """Tests for OverlayConfig."""

    def test_default_creation(self):
        """Test default configuration."""
        config = OverlayConfig()
        assert config.enabled is True
        assert config.position == OverlayPosition.TOP_LEFT
        assert config.show_fps is True

    def test_custom_creation(self):
        """Test custom configuration."""
        config = OverlayConfig(
            position=OverlayPosition.BOTTOM_RIGHT,
            style=OverlayStyle.DETAILED,
            opacity=0.5,
            show_network=True,
        )
        assert config.position == OverlayPosition.BOTTOM_RIGHT
        assert config.style == OverlayStyle.DETAILED
        assert config.opacity == 0.5
        assert config.show_network is True

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = OverlayConfig(
            show_fps=True,
            show_memory=True,
            font_size=14,
        )
        data = config.to_dict()

        assert data["show_fps"] is True
        assert data["show_memory"] is True
        assert data["font_size"] == 14


class TestOverlayPanel:
    """Tests for OverlayPanel."""

    def test_creation(self):
        """Test panel creation."""
        panel = OverlayPanel(
            name="custom_stats",
            position=OverlayPosition.TOP_RIGHT,
        )
        assert panel.name == "custom_stats"
        assert panel.visible is True

    def test_content_callback(self):
        """Test content callback."""
        def get_stats():
            return {"custom_value": 42}

        panel = OverlayPanel(
            name="test",
            content_callback=get_stats,
        )

        content = panel.get_content()
        assert content["custom_value"] == 42

    def test_set_content(self):
        """Test setting custom content."""
        panel = OverlayPanel(name="test")

        panel.set_content(value1=10, value2="test")

        content = panel.get_content()
        assert content["value1"] == 10
        assert content["value2"] == "test"


class TestOverlayStats:
    """Tests for OverlayStats."""

    def test_creation(self):
        """Test stats creation."""
        stats = OverlayStats()
        assert stats.fps == 0.0
        assert stats.frame_time_ms == 0.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        stats = OverlayStats(
            fps=60.0,
            frame_time_ms=16.67,
            memory_mb=256.0,
        )
        data = stats.to_dict()

        assert data["fps"] == 60.0
        assert data["frame_time_ms"] == 16.67
        assert data["memory_mb"] == 256.0


class TestProfilerOverlay:
    """Tests for ProfilerOverlay."""

    @pytest.fixture
    def overlay(self):
        """Create a fresh overlay instance."""
        config = OverlayConfig(update_interval_ms=0.0)  # No throttling for tests
        return ProfilerOverlay(config)

    def test_initial_state(self, overlay):
        """Test initial overlay state."""
        assert overlay.is_visible is True

    def test_show_hide(self, overlay):
        """Test show/hide operations."""
        overlay.hide()
        assert overlay.is_visible is False

        overlay.show()
        assert overlay.is_visible is True

    def test_toggle(self, overlay):
        """Test toggle operation."""
        initial = overlay.is_visible

        result = overlay.toggle()
        assert result != initial
        assert overlay.is_visible != initial

        result = overlay.toggle()
        assert overlay.is_visible == initial

    def test_update_stats(self, overlay):
        """Test stats update."""
        overlay.update(
            fps=60.0,
            frame_time_ms=16.67,
            cpu_time_ms=10.0,
            gpu_time_ms=8.0,
            memory_mb=512.0,
        )

        stats = overlay.stats
        assert stats.fps == 60.0
        assert stats.frame_time_ms == 16.67
        assert stats.cpu_time_ms == 10.0
        assert stats.gpu_time_ms == 8.0
        assert stats.memory_mb == 512.0

    def test_update_network_stats(self, overlay):
        """Test network stats update."""
        overlay.update(
            network_sent_kbps=10.5,
            network_recv_kbps=25.0,
            network_rtt_ms=50.0,
        )

        stats = overlay.stats
        assert stats.network_sent_kbps == 10.5
        assert stats.network_recv_kbps == 25.0
        assert stats.network_rtt_ms == 50.0

    def test_frame_history(self, overlay):
        """Test frame time history."""
        for i in range(10):
            overlay.update(frame_time_ms=16.0 + i)

        history = overlay.get_frame_graph_data()
        assert len(history) == 10

    def test_frame_history_limit(self):
        """Test frame history limiting."""
        config = OverlayConfig(graph_history=5, update_interval_ms=0.0)
        overlay = ProfilerOverlay(config)

        for i in range(10):
            overlay.update(frame_time_ms=float(i))

        history = overlay.get_frame_graph_data()
        assert len(history) == 5

    def test_set_config(self, overlay):
        """Test config update."""
        new_config = OverlayConfig(
            position=OverlayPosition.BOTTOM_LEFT,
            show_fps=False,
        )

        overlay.set_config(new_config)

        assert overlay.config.position == OverlayPosition.BOTTOM_LEFT
        assert overlay.config.show_fps is False

    def test_update_config(self, overlay):
        """Test partial config update."""
        overlay.update_config(
            show_memory=False,
            font_size=16,
        )

        assert overlay.config.show_memory is False
        assert overlay.config.font_size == 16

    def test_set_position(self, overlay):
        """Test position update."""
        overlay.set_position(OverlayPosition.CENTER)
        assert overlay.config.position == OverlayPosition.CENTER

    def test_set_style(self, overlay):
        """Test style update."""
        overlay.set_style(OverlayStyle.GRAPH)
        assert overlay.config.style == OverlayStyle.GRAPH

    def test_add_panel(self, overlay):
        """Test adding custom panels."""
        panel = OverlayPanel(name="custom")
        overlay.add_panel(panel)

        assert "custom" in overlay.list_panels()

    def test_remove_panel(self, overlay):
        """Test removing panels."""
        panel = OverlayPanel(name="temp")
        overlay.add_panel(panel)
        overlay.remove_panel("temp")

        assert "temp" not in overlay.list_panels()

    def test_get_panel(self, overlay):
        """Test getting panel by name."""
        panel = OverlayPanel(name="stats")
        overlay.add_panel(panel)

        retrieved = overlay.get_panel("stats")
        assert retrieved is not None
        assert retrieved.name == "stats"

    def test_set_panel_visible(self, overlay):
        """Test panel visibility control."""
        panel = OverlayPanel(name="test", visible=True)
        overlay.add_panel(panel)

        overlay.set_panel_visible("test", False)

        retrieved = overlay.get_panel("test")
        assert retrieved.visible is False

    def test_get_display_text(self, overlay):
        """Test display text generation."""
        overlay.update(fps=60.0, frame_time_ms=16.67, memory_mb=256.0)

        lines = overlay.get_display_text()

        assert len(lines) > 0
        assert any("FPS" in line for line in lines)
        assert any("Frame" in line for line in lines)

    def test_get_detailed_stats(self, overlay):
        """Test detailed stats retrieval."""
        for i in range(5):
            overlay.update(frame_time_ms=16.0 + i)

        details = overlay.get_detailed_stats()

        assert "fps" in details
        assert "frame_time" in details
        assert "avg" in details["frame_time"]
        assert "min" in details["frame_time"]
        assert "max" in details["frame_time"]

    def test_update_callback(self, overlay):
        """Test update callbacks."""
        received_stats = []

        def on_update(stats):
            received_stats.append(stats)

        overlay.add_update_callback(on_update)
        overlay.update(fps=60.0)

        assert len(received_stats) == 1
        assert received_stats[0].fps == 60.0

        overlay.remove_update_callback(on_update)

    def test_render_callback(self, overlay):
        """Test render callback."""
        render_called = []

        def on_render(stats, config):
            render_called.append((stats, config))

        overlay.set_render_callback(on_render)
        overlay.render()

        assert len(render_called) == 1

    def test_render_when_hidden(self, overlay):
        """Test render doesn't call callback when hidden."""
        render_called = []

        def on_render(stats, config):
            render_called.append(True)

        overlay.set_render_callback(on_render)
        overlay.hide()
        overlay.render()

        assert len(render_called) == 0

    def test_to_dict(self, overlay):
        """Test dictionary export."""
        overlay.update(fps=60.0)

        panel = OverlayPanel(name="custom")
        overlay.add_panel(panel)

        data = overlay.to_dict()

        assert "visible" in data
        assert "config" in data
        assert "stats" in data
        assert "panels" in data
        assert "custom" in data["panels"]

    def test_update_throttling(self):
        """Test update throttling."""
        config = OverlayConfig(update_interval_ms=100.0)
        overlay = ProfilerOverlay(config)

        overlay.update(fps=60.0)
        overlay.update(fps=30.0)  # Should be throttled

        # First update should have gone through
        assert overlay.stats.fps == 60.0

    def test_memory_peak_tracking(self, overlay):
        """Test memory peak tracking."""
        overlay.update(memory_mb=100.0, memory_peak_mb=150.0)

        stats = overlay.stats
        assert stats.memory_mb == 100.0
        assert stats.memory_peak_mb == 150.0

    def test_rendering_stats(self, overlay):
        """Test rendering stats (draw calls, triangles)."""
        overlay.update(draw_calls=1000, triangles=500000)

        stats = overlay.stats
        assert stats.draw_calls == 1000
        assert stats.triangles == 500000
