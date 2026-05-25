"""
Profiler Overlay for the AI Game Engine.

Provides in-game real-time profiling visualization with:
- FPS counter and frame time graph
- CPU/GPU timing breakdown
- Memory usage display
- Network stats
- Customizable panels
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)


class OverlayPosition(Enum):
    """Overlay panel positions."""
    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_RIGHT = auto()
    TOP_CENTER = auto()
    BOTTOM_CENTER = auto()
    CENTER = auto()


class OverlayStyle(Enum):
    """Overlay visual styles."""
    MINIMAL = auto()
    COMPACT = auto()
    DETAILED = auto()
    GRAPH = auto()


@dataclass
class OverlayConfig:
    """Configuration for the profiler overlay."""
    enabled: bool = True
    position: OverlayPosition = OverlayPosition.TOP_LEFT
    style: OverlayStyle = OverlayStyle.COMPACT
    opacity: float = 0.8
    scale: float = 1.0
    show_fps: bool = True
    show_frame_time: bool = True
    show_cpu_time: bool = True
    show_gpu_time: bool = True
    show_memory: bool = True
    show_network: bool = False
    show_draw_calls: bool = False
    show_triangles: bool = False
    show_frame_graph: bool = True
    graph_width: int = 200
    graph_height: int = 60
    graph_history: int = 120
    update_interval_ms: float = 100.0
    font_size: int = 12
    background_color: Tuple[int, int, int, int] = (0, 0, 0, 200)
    text_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    warning_color: Tuple[int, int, int, int] = (255, 200, 0, 255)
    critical_color: Tuple[int, int, int, int] = (255, 50, 50, 255)
    good_color: Tuple[int, int, int, int] = (50, 255, 50, 255)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "position": self.position.name,
            "style": self.style.name,
            "opacity": self.opacity,
            "scale": self.scale,
            "show_fps": self.show_fps,
            "show_frame_time": self.show_frame_time,
            "show_cpu_time": self.show_cpu_time,
            "show_gpu_time": self.show_gpu_time,
            "show_memory": self.show_memory,
            "show_network": self.show_network,
            "show_draw_calls": self.show_draw_calls,
            "show_triangles": self.show_triangles,
            "show_frame_graph": self.show_frame_graph,
            "graph_width": self.graph_width,
            "graph_height": self.graph_height,
            "graph_history": self.graph_history,
            "update_interval_ms": self.update_interval_ms,
            "font_size": self.font_size,
        }


@dataclass
class OverlayPanel:
    """A customizable overlay panel."""
    name: str
    position: OverlayPosition = OverlayPosition.TOP_LEFT
    width: int = 200
    height: int = 100
    visible: bool = True
    opacity: float = 0.8
    content_callback: Optional[Callable[[], Dict[str, Any]]] = None
    render_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def get_content(self) -> Dict[str, Any]:
        """Get panel content."""
        if self.content_callback:
            return self.content_callback()
        return self.custom_data

    def set_content(self, **kwargs: Any) -> None:
        """Set custom content data."""
        self.custom_data.update(kwargs)


@dataclass
class OverlayStats:
    """Statistics for overlay display."""
    fps: float = 0.0
    frame_time_ms: float = 0.0
    cpu_time_ms: float = 0.0
    gpu_time_ms: float = 0.0
    memory_mb: float = 0.0
    memory_peak_mb: float = 0.0
    draw_calls: int = 0
    triangles: int = 0
    network_sent_kbps: float = 0.0
    network_recv_kbps: float = 0.0
    network_rtt_ms: float = 0.0
    frame_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fps": self.fps,
            "frame_time_ms": self.frame_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "gpu_time_ms": self.gpu_time_ms,
            "memory_mb": self.memory_mb,
            "memory_peak_mb": self.memory_peak_mb,
            "draw_calls": self.draw_calls,
            "triangles": self.triangles,
            "network_sent_kbps": self.network_sent_kbps,
            "network_recv_kbps": self.network_recv_kbps,
            "network_rtt_ms": self.network_rtt_ms,
            "frame_history_count": len(self.frame_history),
        }


class ProfilerOverlay:
    """
    In-game profiler overlay for real-time stats display.

    Features:
    - FPS counter and frame time graph
    - CPU/GPU timing breakdown
    - Memory usage display
    - Network stats
    - Customizable panels
    - Multiple visual styles
    """

    __slots__ = (
        "_config",
        "_stats",
        "_panels",
        "_visible",
        "_lock",
        "_last_update",
        "_update_callbacks",
        "_render_callback",
        "_frame_history",
        "_history_max",
    )

    def __init__(self, config: Optional[OverlayConfig] = None) -> None:
        """
        Initialize the profiler overlay.

        Args:
            config: Overlay configuration
        """
        self._config = config or OverlayConfig()
        self._stats = OverlayStats()
        self._panels: Dict[str, OverlayPanel] = {}
        self._visible = self._config.enabled
        self._lock = threading.RLock()
        self._last_update = 0.0
        self._update_callbacks: Set[Callable[[OverlayStats], None]] = set()
        self._render_callback: Optional[Callable[[OverlayStats, OverlayConfig], None]] = None
        self._frame_history: List[float] = []
        self._history_max = self._config.graph_history

    @property
    def is_visible(self) -> bool:
        """Check if overlay is visible."""
        return self._visible

    @property
    def config(self) -> OverlayConfig:
        """Get overlay configuration."""
        return self._config

    @property
    def stats(self) -> OverlayStats:
        """Get current overlay stats."""
        with self._lock:
            return OverlayStats(
                fps=self._stats.fps,
                frame_time_ms=self._stats.frame_time_ms,
                cpu_time_ms=self._stats.cpu_time_ms,
                gpu_time_ms=self._stats.gpu_time_ms,
                memory_mb=self._stats.memory_mb,
                memory_peak_mb=self._stats.memory_peak_mb,
                draw_calls=self._stats.draw_calls,
                triangles=self._stats.triangles,
                network_sent_kbps=self._stats.network_sent_kbps,
                network_recv_kbps=self._stats.network_recv_kbps,
                network_rtt_ms=self._stats.network_rtt_ms,
                frame_history=list(self._stats.frame_history),
            )

    def show(self) -> None:
        """Show the overlay."""
        self._visible = True

    def hide(self) -> None:
        """Hide the overlay."""
        self._visible = False

    def toggle(self) -> bool:
        """Toggle overlay visibility."""
        self._visible = not self._visible
        return self._visible

    def set_config(self, config: OverlayConfig) -> None:
        """Set overlay configuration."""
        with self._lock:
            self._config = config
            self._visible = config.enabled
            self._history_max = config.graph_history

    def update_config(self, **kwargs: Any) -> None:
        """Update specific configuration values."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)

    def set_position(self, position: OverlayPosition) -> None:
        """Set overlay position."""
        self._config.position = position

    def set_style(self, style: OverlayStyle) -> None:
        """Set overlay visual style."""
        self._config.style = style

    def set_render_callback(
        self,
        callback: Callable[[OverlayStats, OverlayConfig], None],
    ) -> None:
        """Set the render callback for custom rendering."""
        self._render_callback = callback

    def add_update_callback(
        self,
        callback: Callable[[OverlayStats], None],
    ) -> None:
        """Add an update callback."""
        self._update_callbacks.add(callback)

    def remove_update_callback(
        self,
        callback: Callable[[OverlayStats], None],
    ) -> None:
        """Remove an update callback."""
        self._update_callbacks.discard(callback)

    def update(
        self,
        fps: Optional[float] = None,
        frame_time_ms: Optional[float] = None,
        cpu_time_ms: Optional[float] = None,
        gpu_time_ms: Optional[float] = None,
        memory_mb: Optional[float] = None,
        memory_peak_mb: Optional[float] = None,
        draw_calls: Optional[int] = None,
        triangles: Optional[int] = None,
        network_sent_kbps: Optional[float] = None,
        network_recv_kbps: Optional[float] = None,
        network_rtt_ms: Optional[float] = None,
    ) -> None:
        """
        Update overlay statistics.

        Args:
            fps: Frames per second
            frame_time_ms: Frame time in milliseconds
            cpu_time_ms: CPU time in milliseconds
            gpu_time_ms: GPU time in milliseconds
            memory_mb: Memory usage in megabytes
            memory_peak_mb: Peak memory usage in megabytes
            draw_calls: Number of draw calls
            triangles: Number of triangles rendered
            network_sent_kbps: Network send bandwidth (KB/s)
            network_recv_kbps: Network receive bandwidth (KB/s)
            network_rtt_ms: Network round-trip time (ms)
        """
        current_time = time.time() * 1000.0
        if current_time - self._last_update < self._config.update_interval_ms:
            return

        with self._lock:
            self._last_update = current_time

            if fps is not None:
                self._stats.fps = fps
            if frame_time_ms is not None:
                self._stats.frame_time_ms = frame_time_ms
                self._add_to_history(frame_time_ms)
            if cpu_time_ms is not None:
                self._stats.cpu_time_ms = cpu_time_ms
            if gpu_time_ms is not None:
                self._stats.gpu_time_ms = gpu_time_ms
            if memory_mb is not None:
                self._stats.memory_mb = memory_mb
            if memory_peak_mb is not None:
                self._stats.memory_peak_mb = memory_peak_mb
            if draw_calls is not None:
                self._stats.draw_calls = draw_calls
            if triangles is not None:
                self._stats.triangles = triangles
            if network_sent_kbps is not None:
                self._stats.network_sent_kbps = network_sent_kbps
            if network_recv_kbps is not None:
                self._stats.network_recv_kbps = network_recv_kbps
            if network_rtt_ms is not None:
                self._stats.network_rtt_ms = network_rtt_ms

            # Update frame history for stats
            self._stats.frame_history = list(self._frame_history)

        # Notify callbacks
        for callback in self._update_callbacks:
            try:
                callback(self._stats)
            except Exception:
                pass

    def _add_to_history(self, frame_time_ms: float) -> None:
        """Add frame time to history."""
        self._frame_history.append(frame_time_ms)
        while len(self._frame_history) > self._history_max:
            self._frame_history.pop(0)

    def render(self) -> None:
        """Render the overlay (calls render callback if set)."""
        if not self._visible:
            return

        if self._render_callback:
            self._render_callback(self._stats, self._config)

    def add_panel(self, panel: OverlayPanel) -> None:
        """Add a custom panel."""
        with self._lock:
            self._panels[panel.name] = panel

    def remove_panel(self, name: str) -> None:
        """Remove a custom panel."""
        with self._lock:
            self._panels.pop(name, None)

    def get_panel(self, name: str) -> Optional[OverlayPanel]:
        """Get a panel by name."""
        with self._lock:
            return self._panels.get(name)

    def list_panels(self) -> List[str]:
        """List all panel names."""
        with self._lock:
            return list(self._panels.keys())

    def set_panel_visible(self, name: str, visible: bool) -> None:
        """Set panel visibility."""
        with self._lock:
            panel = self._panels.get(name)
            if panel:
                panel.visible = visible

    def get_display_text(self) -> List[str]:
        """
        Get display text for the overlay.

        Returns:
            List of text lines to display
        """
        lines = []

        if self._config.show_fps:
            color = self._get_fps_color()
            lines.append(f"FPS: {self._stats.fps:.1f}")

        if self._config.show_frame_time:
            lines.append(f"Frame: {self._stats.frame_time_ms:.2f} ms")

        if self._config.show_cpu_time:
            lines.append(f"CPU: {self._stats.cpu_time_ms:.2f} ms")

        if self._config.show_gpu_time:
            lines.append(f"GPU: {self._stats.gpu_time_ms:.2f} ms")

        if self._config.show_memory:
            lines.append(f"Mem: {self._stats.memory_mb:.1f} MB")

        if self._config.show_draw_calls:
            lines.append(f"Draw: {self._stats.draw_calls}")

        if self._config.show_triangles:
            lines.append(f"Tris: {self._stats.triangles:,}")

        if self._config.show_network:
            lines.append(f"Net: {self._stats.network_sent_kbps:.1f}/{self._stats.network_recv_kbps:.1f} KB/s")
            lines.append(f"RTT: {self._stats.network_rtt_ms:.1f} ms")

        return lines

    def _get_fps_color(self) -> Tuple[int, int, int, int]:
        """Get color based on FPS."""
        if self._stats.fps >= 60:
            return self._config.good_color
        elif self._stats.fps >= 30:
            return self._config.warning_color
        else:
            return self._config.critical_color

    def get_frame_graph_data(self) -> List[float]:
        """Get frame time history for graph rendering."""
        with self._lock:
            return list(self._frame_history)

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics for display."""
        with self._lock:
            history = self._frame_history
            if history:
                avg_frame = sum(history) / len(history)
                min_frame = min(history)
                max_frame = max(history)
            else:
                avg_frame = min_frame = max_frame = 0.0

            return {
                "fps": {
                    "current": self._stats.fps,
                    "avg": 1000.0 / avg_frame if avg_frame > 0 else 0,
                    "min": 1000.0 / max_frame if max_frame > 0 else 0,
                    "max": 1000.0 / min_frame if min_frame > 0 else float("inf"),
                },
                "frame_time": {
                    "current": self._stats.frame_time_ms,
                    "avg": avg_frame,
                    "min": min_frame,
                    "max": max_frame,
                },
                "cpu_time": self._stats.cpu_time_ms,
                "gpu_time": self._stats.gpu_time_ms,
                "memory": {
                    "current": self._stats.memory_mb,
                    "peak": self._stats.memory_peak_mb,
                },
                "rendering": {
                    "draw_calls": self._stats.draw_calls,
                    "triangles": self._stats.triangles,
                },
                "network": {
                    "sent_kbps": self._stats.network_sent_kbps,
                    "recv_kbps": self._stats.network_recv_kbps,
                    "rtt_ms": self._stats.network_rtt_ms,
                },
            }

    def to_dict(self) -> Dict[str, Any]:
        """Export overlay data as dictionary."""
        with self._lock:
            return {
                "visible": self._visible,
                "config": self._config.to_dict(),
                "stats": self._stats.to_dict(),
                "panels": list(self._panels.keys()),
            }
