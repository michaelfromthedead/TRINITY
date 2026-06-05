"""GI Debug Visualization Overlay (T-GIR-P10.1).

Provides debug visualization for all GI systems in TRINITY:
- Probe grid positions with irradiance color-coding
- Voxel occupancy wireframe rendering
- SSGI confidence heatmap
- Path tracer comparison difference heatmap
- Reflection technique mask per pixel

All debug visualizations are gated by the @debug decorator and have
zero overhead when disabled in release builds.

Key Features:
    - ProbeGridVisualization: Colour-coded probe positions by irradiance
    - VoxelOccupancyVisualization: Wireframe/slice view of voxel data
    - SSGIConfidenceHeatmap: Green (high) to Red (low) confidence map
    - PathTracerComparisonHeatmap: Difference magnitude with RMSE/PSNR
    - ReflectionTechniqueMask: Per-pixel technique identification
    - GIDebugOverlay: Unified overlay system with compositing

Architecture:
    - Debug-only frame graph passes
    - Toggle via @debug decorator
    - Zero overhead in release builds
    - Configurable opacity and thresholds

References:
    - UE5 Lumen Debug Visualizations
    - "Practical Real-Time GI" (SIGGRAPH 2019)
    - NVIDIA RTXGI Debug Tools
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
)

if TYPE_CHECKING:
    from engine.platform.rhi.device import Device
    from engine.platform.rhi.resources import Texture


# =============================================================================
# Constants
# =============================================================================

# Debug mode flag (should be set by build system)
_DEBUG_MODE_ENABLED: bool = True

# Default overlay opacity
DEFAULT_OVERLAY_OPACITY: float = 0.5

# Default difference threshold (10% = 0.1)
DEFAULT_DIFFERENCE_THRESHOLD: float = 0.1

# SSGI confidence thresholds
DEFAULT_CONFIDENCE_LOW: float = 0.3
DEFAULT_CONFIDENCE_HIGH: float = 0.8

# Probe visualization sphere radius (world units)
DEFAULT_PROBE_SPHERE_RADIUS: float = 0.15

# Voxel wireframe line width (pixels)
DEFAULT_WIREFRAME_LINE_WIDTH: float = 1.0

# Color gradient steps for heatmaps
GRADIENT_STEPS: int = 256

# Workgroup size for debug compute shaders
DEBUG_WORKGROUP_SIZE: int = 8

# PSNR quality thresholds (dB)
PSNR_EXCELLENT: float = 40.0
PSNR_GOOD: float = 30.0
PSNR_ACCEPTABLE: float = 25.0
PSNR_POOR: float = 20.0


# =============================================================================
# Debug Mode Control
# =============================================================================


def set_debug_mode(enabled: bool) -> None:
    """Set the global debug mode flag.

    In release builds, this should be set to False at startup to ensure
    zero overhead from debug visualization code.

    Args:
        enabled: Whether debug mode is enabled.
    """
    global _DEBUG_MODE_ENABLED
    _DEBUG_MODE_ENABLED = enabled


def is_debug_enabled() -> bool:
    """Check if debug mode is currently enabled.

    Returns:
        True if debug visualizations are active.
    """
    return _DEBUG_MODE_ENABLED


F = TypeVar("F", bound=Callable[..., Any])


def debug(func: F) -> F:
    """Decorator to gate debug-only code.

    Functions decorated with @debug will:
    - Execute normally when _DEBUG_MODE_ENABLED is True
    - Return None (or the default return value) when disabled
    - Have zero overhead in release builds after JIT compilation

    Usage:
        @debug
        def render_debug_overlay(self) -> None:
            # This code only runs in debug mode
            pass

    Args:
        func: The function to wrap.

    Returns:
        Wrapped function that checks debug mode.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _DEBUG_MODE_ENABLED:
            return None
        return func(*args, **kwargs)
    return wrapper  # type: ignore


# Debug pass registry for frame graph integration
_registered_debug_passes: Dict[str, Callable[[], None]] = {}


def register_debug_pass(name: str, pass_func: Callable[[], None]) -> None:
    """Register a debug pass with the frame graph.

    Args:
        name: Unique name for the debug pass.
        pass_func: Function to execute for this debug pass.
    """
    if _DEBUG_MODE_ENABLED:
        _registered_debug_passes[name] = pass_func


def get_registered_debug_passes() -> Dict[str, Callable[[], None]]:
    """Get all registered debug passes.

    Returns:
        Dictionary of pass names to pass functions.
    """
    return _registered_debug_passes.copy()


def clear_debug_passes() -> None:
    """Clear all registered debug passes."""
    _registered_debug_passes.clear()


# =============================================================================
# Enums
# =============================================================================


class ProbeColorMode(IntEnum):
    """Color mode for probe visualization."""

    IRRADIANCE = 0      # Color by irradiance magnitude (cold->hot)
    LUMINANCE = 1       # Color by luminance value
    STATE = 2           # Color by probe state (active/dormant/invalid)
    DEPTH = 3           # Color by octree depth (for adaptive grids)
    VARIANCE = 4        # Color by variance (for debug sampling)


class ProbeState(IntEnum):
    """State for probe visualization."""

    ACTIVE = 0          # Normal active probe
    DORMANT = 1         # Probe temporarily inactive
    INVALID = 2         # Probe marked invalid
    FADING_IN = 3       # Probe fading in
    FADING_OUT = 4      # Probe fading out


class ReflectionTechnique(IntEnum):
    """Reflection technique identifiers for mask visualization."""

    NONE = 0            # No reflection
    RAY_TRACED = 1      # Hardware ray tracing
    SSR = 2             # Screen-space reflections
    PROBES = 3          # Environment/reflection probes
    ENVIRONMENT = 4     # Fallback environment map
    PLANAR = 5          # Planar reflections


class VoxelSliceAxis(IntEnum):
    """Axis for voxel slice visualization."""

    X = 0               # YZ plane slice
    Y = 1               # XZ plane slice
    Z = 2               # XY plane slice


class HeatmapColorScale(IntEnum):
    """Color scale for heatmap visualization."""

    BLUE_RED = 0        # Blue (low) -> Red (high)
    GREEN_RED = 1       # Green (low) -> Red (high)
    VIRIDIS = 2         # Perceptually uniform (scientific)
    INFERNO = 3         # Perceptually uniform (high contrast)


# =============================================================================
# Color Utilities
# =============================================================================


@dataclass(frozen=True)
class Color:
    """RGBA color with normalized components [0, 1]."""

    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    a: float = 1.0

    def __post_init__(self) -> None:
        """Clamp values to valid range."""
        object.__setattr__(self, "r", max(0.0, min(1.0, self.r)))
        object.__setattr__(self, "g", max(0.0, min(1.0, self.g)))
        object.__setattr__(self, "b", max(0.0, min(1.0, self.b)))
        object.__setattr__(self, "a", max(0.0, min(1.0, self.a)))

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Convert to RGBA tuple."""
        return (self.r, self.g, self.b, self.a)

    def to_rgb_tuple(self) -> Tuple[float, float, float]:
        """Convert to RGB tuple."""
        return (self.r, self.g, self.b)

    def luminance(self) -> float:
        """Calculate perceived luminance."""
        return 0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b

    def blend(self, other: Color, t: float) -> Color:
        """Linear blend with another color.

        Args:
            other: Color to blend towards.
            t: Blend factor [0, 1]. 0 = self, 1 = other.

        Returns:
            Blended color.
        """
        t = max(0.0, min(1.0, t))
        return Color(
            r=self.r + (other.r - self.r) * t,
            g=self.g + (other.g - self.g) * t,
            b=self.b + (other.b - self.b) * t,
            a=self.a + (other.a - self.a) * t,
        )

    @staticmethod
    def from_hex(hex_code: str) -> Color:
        """Create color from hex string (e.g., '#FF0000' or 'FF0000')."""
        hex_code = hex_code.lstrip("#")
        if len(hex_code) == 6:
            r = int(hex_code[0:2], 16) / 255.0
            g = int(hex_code[2:4], 16) / 255.0
            b = int(hex_code[4:6], 16) / 255.0
            return Color(r, g, b, 1.0)
        elif len(hex_code) == 8:
            r = int(hex_code[0:2], 16) / 255.0
            g = int(hex_code[2:4], 16) / 255.0
            b = int(hex_code[4:6], 16) / 255.0
            a = int(hex_code[6:8], 16) / 255.0
            return Color(r, g, b, a)
        else:
            raise ValueError(f"Invalid hex color: {hex_code}")


# Predefined colors for visualization
COLOR_RED = Color(1.0, 0.0, 0.0)
COLOR_GREEN = Color(0.0, 1.0, 0.0)
COLOR_BLUE = Color(0.0, 0.0, 1.0)
COLOR_YELLOW = Color(1.0, 1.0, 0.0)
COLOR_CYAN = Color(0.0, 1.0, 1.0)
COLOR_MAGENTA = Color(1.0, 0.0, 1.0)
COLOR_WHITE = Color(1.0, 1.0, 1.0)
COLOR_BLACK = Color(0.0, 0.0, 0.0)
COLOR_GRAY = Color(0.5, 0.5, 0.5)
COLOR_ORANGE = Color(1.0, 0.5, 0.0)

# Technique mask colors
TECHNIQUE_COLOR_RT = COLOR_BLUE
TECHNIQUE_COLOR_SSR = COLOR_GREEN
TECHNIQUE_COLOR_PROBES = COLOR_YELLOW
TECHNIQUE_COLOR_ENV = COLOR_GRAY
TECHNIQUE_COLOR_PLANAR = COLOR_CYAN
TECHNIQUE_COLOR_NONE = COLOR_BLACK

# Probe state colors
STATE_COLOR_ACTIVE = COLOR_GREEN
STATE_COLOR_DORMANT = COLOR_YELLOW
STATE_COLOR_INVALID = COLOR_RED
STATE_COLOR_FADING_IN = COLOR_CYAN
STATE_COLOR_FADING_OUT = COLOR_ORANGE


def create_heatmap_gradient(
    scale: HeatmapColorScale = HeatmapColorScale.BLUE_RED,
    steps: int = GRADIENT_STEPS,
) -> List[Color]:
    """Create a color gradient for heatmap visualization.

    Args:
        scale: Color scale to use.
        steps: Number of discrete steps in the gradient.

    Returns:
        List of colors from low to high values.
    """
    gradient: List[Color] = []

    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0.0

        if scale == HeatmapColorScale.BLUE_RED:
            # Blue -> Cyan -> Green -> Yellow -> Red
            if t < 0.25:
                gradient.append(COLOR_BLUE.blend(COLOR_CYAN, t * 4.0))
            elif t < 0.5:
                gradient.append(COLOR_CYAN.blend(COLOR_GREEN, (t - 0.25) * 4.0))
            elif t < 0.75:
                gradient.append(COLOR_GREEN.blend(COLOR_YELLOW, (t - 0.5) * 4.0))
            else:
                gradient.append(COLOR_YELLOW.blend(COLOR_RED, (t - 0.75) * 4.0))

        elif scale == HeatmapColorScale.GREEN_RED:
            # Green -> Yellow -> Red
            if t < 0.5:
                gradient.append(COLOR_GREEN.blend(COLOR_YELLOW, t * 2.0))
            else:
                gradient.append(COLOR_YELLOW.blend(COLOR_RED, (t - 0.5) * 2.0))

        elif scale == HeatmapColorScale.VIRIDIS:
            # Perceptually uniform viridis approximation
            r = 0.267 + 0.051 * t + 0.329 * t * t
            g = 0.004 + 1.037 * t - 0.445 * t * t
            b = 0.329 + 1.255 * t - 1.557 * t * t
            gradient.append(Color(r, g, b))

        elif scale == HeatmapColorScale.INFERNO:
            # Perceptually uniform inferno approximation
            r = 0.001 + 0.847 * t + 0.152 * t * t
            g = 0.001 + 0.054 * t + 0.945 * t * t * t
            b = 0.014 + 0.735 * t - 0.749 * t * t
            gradient.append(Color(r, g, b))

    return gradient


def sample_gradient(
    gradient: List[Color],
    value: float,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> Color:
    """Sample a color from a gradient based on a value.

    Args:
        gradient: List of colors in the gradient.
        value: Value to look up.
        min_value: Minimum value (maps to gradient start).
        max_value: Maximum value (maps to gradient end).

    Returns:
        Interpolated color from the gradient.
    """
    if not gradient:
        return COLOR_BLACK

    # Normalize value to [0, 1]
    if max_value <= min_value:
        t = 0.0
    else:
        t = (value - min_value) / (max_value - min_value)

    t = max(0.0, min(1.0, t))

    # Find gradient indices
    idx_float = t * (len(gradient) - 1)
    idx_low = int(idx_float)
    idx_high = min(idx_low + 1, len(gradient) - 1)
    frac = idx_float - idx_low

    return gradient[idx_low].blend(gradient[idx_high], frac)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GIDebugConfig:
    """Configuration for GI debug visualization overlay.

    Attributes:
        show_probes: Enable probe grid visualization.
        show_voxels: Enable voxel occupancy visualization.
        show_ssgi_confidence: Enable SSGI confidence heatmap.
        show_path_tracer_diff: Enable path tracer comparison.
        show_reflection_mask: Enable reflection technique mask.
        overlay_opacity: Global opacity for all overlays [0, 1].
        difference_threshold: Threshold for path tracer diff (0.1 = 10%).
        probe_color_mode: Color mode for probe visualization.
        probe_sphere_radius: Radius of probe spheres in world units.
        voxel_slice_axis: Axis for voxel slice view.
        voxel_slice_depth: Normalized depth [0, 1] for slice view.
        confidence_threshold: SSGI confidence threshold for highlight mode.
        heatmap_scale: Color scale for heatmaps.
    """

    show_probes: bool = False
    show_voxels: bool = False
    show_ssgi_confidence: bool = False
    show_path_tracer_diff: bool = False
    show_reflection_mask: bool = False
    overlay_opacity: float = DEFAULT_OVERLAY_OPACITY
    difference_threshold: float = DEFAULT_DIFFERENCE_THRESHOLD
    probe_color_mode: ProbeColorMode = ProbeColorMode.IRRADIANCE
    probe_sphere_radius: float = DEFAULT_PROBE_SPHERE_RADIUS
    voxel_slice_axis: VoxelSliceAxis = VoxelSliceAxis.Y
    voxel_slice_depth: float = 0.5
    confidence_threshold: float = DEFAULT_CONFIDENCE_LOW
    heatmap_scale: HeatmapColorScale = HeatmapColorScale.GREEN_RED

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.overlay_opacity <= 1.0:
            raise ValueError(
                f"overlay_opacity must be in [0.0, 1.0], got {self.overlay_opacity}"
            )
        if not 0.0 <= self.difference_threshold <= 1.0:
            raise ValueError(
                f"difference_threshold must be in [0.0, 1.0], "
                f"got {self.difference_threshold}"
            )
        if self.probe_sphere_radius <= 0.0:
            raise ValueError(
                f"probe_sphere_radius must be > 0, got {self.probe_sphere_radius}"
            )
        if not 0.0 <= self.voxel_slice_depth <= 1.0:
            raise ValueError(
                f"voxel_slice_depth must be in [0.0, 1.0], "
                f"got {self.voxel_slice_depth}"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], "
                f"got {self.confidence_threshold}"
            )

    @property
    def any_enabled(self) -> bool:
        """Check if any debug visualization is enabled."""
        return (
            self.show_probes
            or self.show_voxels
            or self.show_ssgi_confidence
            or self.show_path_tracer_diff
            or self.show_reflection_mask
        )

    def count_enabled(self) -> int:
        """Count number of enabled visualizations."""
        return sum([
            self.show_probes,
            self.show_voxels,
            self.show_ssgi_confidence,
            self.show_path_tracer_diff,
            self.show_reflection_mask,
        ])


# =============================================================================
# Probe Data Structures
# =============================================================================


@dataclass
class ProbeVisualizationData:
    """Data for a single probe visualization.

    Attributes:
        position: World position of the probe.
        irradiance: RGB irradiance values.
        state: Current probe state.
        depth: Octree depth (for adaptive grids).
        variance: Variance value (for debug sampling).
        blend_weight: Blend weight during transitions.
    """

    position: Tuple[float, float, float]
    irradiance: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    state: ProbeState = ProbeState.ACTIVE
    depth: int = 0
    variance: float = 0.0
    blend_weight: float = 1.0

    def get_luminance(self) -> float:
        """Calculate luminance from irradiance."""
        return (
            0.2126 * self.irradiance[0]
            + 0.7152 * self.irradiance[1]
            + 0.0722 * self.irradiance[2]
        )

    def get_irradiance_magnitude(self) -> float:
        """Calculate irradiance magnitude."""
        return math.sqrt(
            self.irradiance[0] ** 2
            + self.irradiance[1] ** 2
            + self.irradiance[2] ** 2
        )


# =============================================================================
# Probe Grid Visualization
# =============================================================================


class ProbeGridVisualization:
    """Visualization for probe grid positions with irradiance color-coding.

    Renders probe positions as spheres or billboards with colors based on
    irradiance magnitude, luminance, state, depth, or variance.

    The visualization supports:
    - Cold (blue) to hot (red) irradiance gradient
    - State-based coloring (active/dormant/invalid)
    - Depth-based coloring for adaptive grids
    - Configurable sphere radius and opacity
    """

    def __init__(self, config: GIDebugConfig) -> None:
        """Initialize probe grid visualization.

        Args:
            config: Debug configuration.
        """
        self._config = config
        self._probes: List[ProbeVisualizationData] = []
        self._gradient = create_heatmap_gradient(
            HeatmapColorScale.BLUE_RED, GRADIENT_STEPS
        )
        self._max_irradiance: float = 1.0
        self._min_irradiance: float = 0.0
        self._max_depth: int = 4

    @property
    def probe_count(self) -> int:
        """Get the number of probes being visualized."""
        return len(self._probes)

    def set_probes(self, probes: List[ProbeVisualizationData]) -> None:
        """Set probe data for visualization.

        Args:
            probes: List of probe visualization data.
        """
        self._probes = probes

        # Update irradiance range for normalization
        if probes:
            magnitudes = [p.get_irradiance_magnitude() for p in probes]
            self._min_irradiance = min(magnitudes)
            self._max_irradiance = max(magnitudes) or 1.0

            depths = [p.depth for p in probes]
            self._max_depth = max(depths) or 1

    def set_color_mode(self, mode: ProbeColorMode) -> None:
        """Set the color mode for probe visualization.

        Args:
            mode: New color mode.
        """
        self._config.probe_color_mode = mode

    def get_probe_color(self, probe: ProbeVisualizationData) -> Color:
        """Get the color for a probe based on current color mode.

        Args:
            probe: Probe data.

        Returns:
            Color for the probe.
        """
        mode = self._config.probe_color_mode

        if mode == ProbeColorMode.IRRADIANCE:
            magnitude = probe.get_irradiance_magnitude()
            return sample_gradient(
                self._gradient,
                magnitude,
                self._min_irradiance,
                self._max_irradiance,
            )

        elif mode == ProbeColorMode.LUMINANCE:
            luminance = probe.get_luminance()
            return sample_gradient(
                self._gradient,
                luminance,
                0.0,
                1.0,
            )

        elif mode == ProbeColorMode.STATE:
            state_colors = {
                ProbeState.ACTIVE: STATE_COLOR_ACTIVE,
                ProbeState.DORMANT: STATE_COLOR_DORMANT,
                ProbeState.INVALID: STATE_COLOR_INVALID,
                ProbeState.FADING_IN: STATE_COLOR_FADING_IN,
                ProbeState.FADING_OUT: STATE_COLOR_FADING_OUT,
            }
            return state_colors.get(probe.state, STATE_COLOR_INVALID)

        elif mode == ProbeColorMode.DEPTH:
            t = probe.depth / max(self._max_depth, 1)
            return sample_gradient(self._gradient, t)

        elif mode == ProbeColorMode.VARIANCE:
            return sample_gradient(self._gradient, probe.variance, 0.0, 0.1)

        return COLOR_WHITE

    @debug
    def render_probes(self) -> List[Tuple[Tuple[float, float, float], Color, float]]:
        """Render probes as position, color, radius tuples.

        Returns:
            List of (position, color, radius) for each probe.
            Returns None if debug mode is disabled.
        """
        result = []
        radius = self._config.probe_sphere_radius

        for probe in self._probes:
            color = self.get_probe_color(probe)
            # Apply blend weight to alpha
            color = Color(color.r, color.g, color.b, color.a * probe.blend_weight)
            result.append((probe.position, color, radius))

        return result

    def get_probe_at_index(self, index: int) -> Optional[ProbeVisualizationData]:
        """Get probe data at a specific index.

        Args:
            index: Probe index.

        Returns:
            Probe data or None if index is out of bounds.
        """
        if 0 <= index < len(self._probes):
            return self._probes[index]
        return None


# =============================================================================
# Voxel Occupancy Visualization
# =============================================================================


@dataclass
class VoxelData:
    """Data for a single voxel.

    Attributes:
        position: Grid coordinates (x, y, z).
        density: Occupancy density [0, 1].
        color: Optional albedo color.
    """

    position: Tuple[int, int, int]
    density: float = 1.0
    color: Optional[Tuple[float, float, float]] = None


class VoxelOccupancyVisualization:
    """Visualization for voxel occupancy with wireframe and slice views.

    Renders occupied voxels as:
    - Wireframe cubes with opacity based on density
    - 2D slice at configurable depth
    - Color-coded by density or custom color
    """

    def __init__(self, config: GIDebugConfig) -> None:
        """Initialize voxel occupancy visualization.

        Args:
            config: Debug configuration.
        """
        self._config = config
        self._voxels: List[VoxelData] = []
        self._resolution: Tuple[int, int, int] = (64, 64, 64)
        self._world_bounds: Tuple[
            Tuple[float, float, float],
            Tuple[float, float, float],
        ] = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        self._gradient = create_heatmap_gradient(
            HeatmapColorScale.VIRIDIS, GRADIENT_STEPS
        )

    def set_voxels(
        self,
        voxels: List[VoxelData],
        resolution: Tuple[int, int, int],
        world_bounds: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ] = None,
    ) -> None:
        """Set voxel data for visualization.

        Args:
            voxels: List of voxel data.
            resolution: Grid resolution (x, y, z).
            world_bounds: Optional world space bounds ((min), (max)).
        """
        self._voxels = voxels
        self._resolution = resolution
        if world_bounds:
            self._world_bounds = world_bounds

    def set_slice_depth(self, depth: float) -> None:
        """Set the depth for slice visualization.

        Args:
            depth: Normalized depth [0, 1] along the slice axis.
        """
        self._config.voxel_slice_depth = max(0.0, min(1.0, depth))

    def _grid_to_world(
        self,
        grid_pos: Tuple[int, int, int],
    ) -> Tuple[float, float, float]:
        """Convert grid coordinates to world position.

        Args:
            grid_pos: Grid coordinates (x, y, z).

        Returns:
            World position (x, y, z).
        """
        min_b = self._world_bounds[0]
        max_b = self._world_bounds[1]
        res = self._resolution

        return (
            min_b[0] + (grid_pos[0] + 0.5) / res[0] * (max_b[0] - min_b[0]),
            min_b[1] + (grid_pos[1] + 0.5) / res[1] * (max_b[1] - min_b[1]),
            min_b[2] + (grid_pos[2] + 0.5) / res[2] * (max_b[2] - min_b[2]),
        )

    def _get_voxel_size(self) -> Tuple[float, float, float]:
        """Get the size of a single voxel in world units."""
        min_b = self._world_bounds[0]
        max_b = self._world_bounds[1]
        res = self._resolution

        return (
            (max_b[0] - min_b[0]) / res[0],
            (max_b[1] - min_b[1]) / res[1],
            (max_b[2] - min_b[2]) / res[2],
        )

    @debug
    def render_wireframe(
        self,
    ) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Color]]:
        """Render voxel wireframe as line segments.

        Returns:
            List of (start_pos, end_pos, color) for wireframe lines.
            Returns None if debug mode is disabled.
        """
        lines = []
        voxel_size = self._get_voxel_size()
        half_size = (voxel_size[0] / 2, voxel_size[1] / 2, voxel_size[2] / 2)

        for voxel in self._voxels:
            if voxel.density <= 0.0:
                continue

            center = self._grid_to_world(voxel.position)
            color = sample_gradient(self._gradient, voxel.density)
            color = Color(color.r, color.g, color.b, voxel.density)

            # Generate cube wireframe (12 edges)
            corners = []
            for dx in (-1, 1):
                for dy in (-1, 1):
                    for dz in (-1, 1):
                        corners.append((
                            center[0] + dx * half_size[0],
                            center[1] + dy * half_size[1],
                            center[2] + dz * half_size[2],
                        ))

            # Edges along X axis (4)
            for i in (0, 2, 4, 6):
                lines.append((corners[i], corners[i + 1], color))

            # Edges along Y axis (4)
            for i in (0, 1, 4, 5):
                lines.append((corners[i], corners[i + 2], color))

            # Edges along Z axis (4)
            for i in range(4):
                lines.append((corners[i], corners[i + 4], color))

        return lines

    @debug
    def render_slice(
        self,
    ) -> List[Tuple[Tuple[int, int], Color]]:
        """Render a 2D slice of the voxel grid.

        Returns:
            List of (grid_2d_coord, color) for the slice.
            Returns None if debug mode is disabled.
        """
        axis = self._config.voxel_slice_axis
        depth = self._config.voxel_slice_depth
        res = self._resolution

        # Calculate slice index
        if axis == VoxelSliceAxis.X:
            slice_idx = int(depth * (res[0] - 1))
        elif axis == VoxelSliceAxis.Y:
            slice_idx = int(depth * (res[1] - 1))
        else:  # Z
            slice_idx = int(depth * (res[2] - 1))

        # Filter voxels at the slice
        pixels = []
        for voxel in self._voxels:
            pos = voxel.position

            # Check if voxel is on the slice plane
            if axis == VoxelSliceAxis.X and pos[0] == slice_idx:
                pixels.append(((pos[1], pos[2]), sample_gradient(
                    self._gradient, voxel.density
                )))
            elif axis == VoxelSliceAxis.Y and pos[1] == slice_idx:
                pixels.append(((pos[0], pos[2]), sample_gradient(
                    self._gradient, voxel.density
                )))
            elif axis == VoxelSliceAxis.Z and pos[2] == slice_idx:
                pixels.append(((pos[0], pos[1]), sample_gradient(
                    self._gradient, voxel.density
                )))

        return pixels

    @property
    def voxel_count(self) -> int:
        """Get the number of occupied voxels."""
        return len(self._voxels)

    @property
    def total_voxels(self) -> int:
        """Get total possible voxels at current resolution."""
        return self._resolution[0] * self._resolution[1] * self._resolution[2]

    @property
    def occupancy_ratio(self) -> float:
        """Get the ratio of occupied voxels."""
        total = self.total_voxels
        if total == 0:
            return 0.0
        return self.voxel_count / total


# =============================================================================
# SSGI Confidence Heatmap
# =============================================================================


@dataclass
class ConfidencePixel:
    """Per-pixel SSGI confidence data.

    Attributes:
        x: Pixel X coordinate.
        y: Pixel Y coordinate.
        confidence: Confidence value [0, 1].
        hit_count: Number of SSGI hits for this pixel.
        miss_count: Number of SSGI misses for this pixel.
    """

    x: int
    y: int
    confidence: float
    hit_count: int = 0
    miss_count: int = 0


class SSGIConfidenceHeatmap:
    """SSGI confidence visualization with per-pixel heatmap.

    Shows confidence values as a heatmap:
    - Green: High confidence (good SSGI coverage)
    - Yellow: Medium confidence
    - Red: Low confidence (fallback to probes/env)

    Supports threshold-based highlight mode to identify
    pixels below a confidence threshold.
    """

    def __init__(self, config: GIDebugConfig) -> None:
        """Initialize SSGI confidence heatmap.

        Args:
            config: Debug configuration.
        """
        self._config = config
        self._pixels: List[ConfidencePixel] = []
        self._width: int = 0
        self._height: int = 0
        self._gradient = create_heatmap_gradient(
            HeatmapColorScale.GREEN_RED, GRADIENT_STEPS
        )

    def set_confidence_data(
        self,
        pixels: List[ConfidencePixel],
        width: int,
        height: int,
    ) -> None:
        """Set confidence data for the heatmap.

        Args:
            pixels: List of per-pixel confidence data.
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        self._pixels = pixels
        self._width = width
        self._height = height

    def set_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for highlight mode.

        Args:
            threshold: Confidence threshold [0, 1].
        """
        self._config.confidence_threshold = max(0.0, min(1.0, threshold))

    def get_confidence_color(self, confidence: float) -> Color:
        """Get the color for a confidence value.

        Args:
            confidence: Confidence value [0, 1].

        Returns:
            Color from the heatmap gradient (green->yellow->red).
        """
        # Invert so high confidence = green, low = red
        return sample_gradient(self._gradient, 1.0 - confidence)

    @debug
    def render_heatmap(self) -> List[Tuple[Tuple[int, int], Color]]:
        """Render the confidence heatmap.

        Returns:
            List of (pixel_coord, color) for the heatmap.
            Returns None if debug mode is disabled.
        """
        result = []
        threshold = self._config.confidence_threshold

        for pixel in self._pixels:
            color = self.get_confidence_color(pixel.confidence)

            # Highlight mode: dim pixels above threshold
            if pixel.confidence > threshold:
                color = Color(
                    color.r * 0.3,
                    color.g * 0.3,
                    color.b * 0.3,
                    color.a * 0.3,
                )

            result.append(((pixel.x, pixel.y), color))

        return result

    @debug
    def render_threshold_mask(self) -> List[Tuple[int, int]]:
        """Render pixels below the confidence threshold.

        Returns:
            List of pixel coordinates below threshold.
            Returns None if debug mode is disabled.
        """
        threshold = self._config.confidence_threshold
        return [
            (pixel.x, pixel.y)
            for pixel in self._pixels
            if pixel.confidence < threshold
        ]

    def get_statistics(self) -> Dict[str, float]:
        """Get confidence statistics.

        Returns:
            Dictionary with min, max, mean, and std confidence values.
        """
        if not self._pixels:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "below_threshold_ratio": 0.0,
            }

        confidences = [p.confidence for p in self._pixels]
        mean = sum(confidences) / len(confidences)
        variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
        threshold = self._config.confidence_threshold
        below_count = sum(1 for c in confidences if c < threshold)

        return {
            "min": min(confidences),
            "max": max(confidences),
            "mean": mean,
            "std": math.sqrt(variance),
            "below_threshold_ratio": below_count / len(confidences),
        }


# =============================================================================
# Path Tracer Comparison Heatmap
# =============================================================================


@dataclass
class DifferencePixel:
    """Per-pixel difference between GI result and path tracer reference.

    Attributes:
        x: Pixel X coordinate.
        y: Pixel Y coordinate.
        gi_color: GI result RGB [0, 1].
        reference_color: Path tracer reference RGB [0, 1].
        difference: Absolute difference magnitude.
    """

    x: int
    y: int
    gi_color: Tuple[float, float, float]
    reference_color: Tuple[float, float, float]
    difference: float = 0.0

    def __post_init__(self) -> None:
        """Compute difference if not provided."""
        if self.difference == 0.0 and (
            self.gi_color != (0.0, 0.0, 0.0)
            or self.reference_color != (0.0, 0.0, 0.0)
        ):
            self.difference = self._compute_difference()

    def _compute_difference(self) -> float:
        """Compute the relative difference between GI and reference."""
        ref_luminance = (
            0.2126 * self.reference_color[0]
            + 0.7152 * self.reference_color[1]
            + 0.0722 * self.reference_color[2]
        )

        diff_r = abs(self.gi_color[0] - self.reference_color[0])
        diff_g = abs(self.gi_color[1] - self.reference_color[1])
        diff_b = abs(self.gi_color[2] - self.reference_color[2])

        abs_diff = (diff_r + diff_g + diff_b) / 3.0

        # Relative difference (avoid division by zero)
        if ref_luminance > 0.001:
            return abs_diff / ref_luminance
        return abs_diff


@dataclass
class ComparisonStats:
    """Statistics for path tracer comparison.

    Attributes:
        rmse: Root Mean Square Error.
        psnr: Peak Signal-to-Noise Ratio (dB).
        max_diff: Maximum difference value.
        mean_diff: Mean difference value.
        above_threshold_ratio: Ratio of pixels above threshold.
    """

    rmse: float = 0.0
    psnr: float = float("inf")
    max_diff: float = 0.0
    mean_diff: float = 0.0
    above_threshold_ratio: float = 0.0


class PathTracerComparisonHeatmap:
    """Comparison heatmap between GI result and path tracer reference.

    Visualizes the difference magnitude:
    - Dark/Blue: Low difference (good match)
    - Yellow: Medium difference
    - Red: High difference (>10% by default)

    Also computes RMSE and PSNR metrics for quality assessment.
    """

    def __init__(self, config: GIDebugConfig) -> None:
        """Initialize path tracer comparison heatmap.

        Args:
            config: Debug configuration.
        """
        self._config = config
        self._pixels: List[DifferencePixel] = []
        self._width: int = 0
        self._height: int = 0
        self._stats: Optional[ComparisonStats] = None
        self._gradient = create_heatmap_gradient(
            HeatmapColorScale.BLUE_RED, GRADIENT_STEPS
        )

    def set_comparison_data(
        self,
        pixels: List[DifferencePixel],
        width: int,
        height: int,
    ) -> None:
        """Set comparison data for the heatmap.

        Args:
            pixels: List of per-pixel difference data.
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        self._pixels = pixels
        self._width = width
        self._height = height
        self._stats = None  # Invalidate cached stats

    def get_difference_color(self, difference: float) -> Color:
        """Get the color for a difference value.

        Args:
            difference: Difference value (0 = perfect match).

        Returns:
            Color from the heatmap gradient.
        """
        threshold = self._config.difference_threshold
        # Normalize difference to [0, 1] based on threshold * 2
        normalized = min(difference / (threshold * 2), 1.0)
        return sample_gradient(self._gradient, normalized)

    def compute_stats(self) -> ComparisonStats:
        """Compute comparison statistics.

        Returns:
            ComparisonStats with RMSE, PSNR, and other metrics.
        """
        if self._stats is not None:
            return self._stats

        if not self._pixels:
            self._stats = ComparisonStats()
            return self._stats

        # Compute MSE
        squared_errors = []
        differences = []
        threshold = self._config.difference_threshold
        above_threshold = 0

        for pixel in self._pixels:
            diff_r = pixel.gi_color[0] - pixel.reference_color[0]
            diff_g = pixel.gi_color[1] - pixel.reference_color[1]
            diff_b = pixel.gi_color[2] - pixel.reference_color[2]

            se = (diff_r ** 2 + diff_g ** 2 + diff_b ** 2) / 3.0
            squared_errors.append(se)
            differences.append(pixel.difference)

            if pixel.difference > threshold:
                above_threshold += 1

        mse = sum(squared_errors) / len(squared_errors)
        rmse = math.sqrt(mse)

        # PSNR (assuming max signal value of 1.0)
        if mse > 0:
            psnr = 10 * math.log10(1.0 / mse)
        else:
            psnr = float("inf")

        self._stats = ComparisonStats(
            rmse=rmse,
            psnr=psnr,
            max_diff=max(differences),
            mean_diff=sum(differences) / len(differences),
            above_threshold_ratio=above_threshold / len(self._pixels),
        )

        return self._stats

    @debug
    def render_difference(self) -> List[Tuple[Tuple[int, int], Color]]:
        """Render the difference heatmap.

        Returns:
            List of (pixel_coord, color) for the heatmap.
            Returns None if debug mode is disabled.
        """
        result = []
        threshold = self._config.difference_threshold

        for pixel in self._pixels:
            color = self.get_difference_color(pixel.difference)

            # Highlight pixels above threshold
            if pixel.difference > threshold:
                # Make high-difference pixels more saturated
                color = Color(
                    min(1.0, color.r * 1.5),
                    color.g * 0.5,
                    color.b * 0.5,
                    1.0,
                )

            result.append(((pixel.x, pixel.y), color))

        return result

    def get_psnr_quality(self) -> str:
        """Get quality assessment based on PSNR.

        Returns:
            Quality string: 'excellent', 'good', 'acceptable', or 'poor'.
        """
        stats = self.compute_stats()

        if stats.psnr >= PSNR_EXCELLENT:
            return "excellent"
        elif stats.psnr >= PSNR_GOOD:
            return "good"
        elif stats.psnr >= PSNR_ACCEPTABLE:
            return "acceptable"
        else:
            return "poor"


# =============================================================================
# Reflection Technique Mask
# =============================================================================


@dataclass
class ReflectionPixel:
    """Per-pixel reflection technique data.

    Attributes:
        x: Pixel X coordinate.
        y: Pixel Y coordinate.
        technique: Active reflection technique.
        fallback_reason: Optional reason for fallback.
        contribution: Contribution weight [0, 1].
    """

    x: int
    y: int
    technique: ReflectionTechnique
    fallback_reason: Optional[str] = None
    contribution: float = 1.0


class ReflectionTechniqueMask:
    """Per-pixel visualization of active reflection technique.

    Color-codes each pixel by the reflection technique used:
    - Blue: Ray traced reflections
    - Green: Screen-space reflections (SSR)
    - Yellow: Environment/reflection probes
    - Gray: Fallback environment map
    - Cyan: Planar reflections
    - Black: No reflections
    """

    def __init__(self, config: GIDebugConfig) -> None:
        """Initialize reflection technique mask.

        Args:
            config: Debug configuration.
        """
        self._config = config
        self._pixels: List[ReflectionPixel] = []
        self._width: int = 0
        self._height: int = 0

        # Technique color mapping
        self._technique_colors: Dict[ReflectionTechnique, Color] = {
            ReflectionTechnique.NONE: TECHNIQUE_COLOR_NONE,
            ReflectionTechnique.RAY_TRACED: TECHNIQUE_COLOR_RT,
            ReflectionTechnique.SSR: TECHNIQUE_COLOR_SSR,
            ReflectionTechnique.PROBES: TECHNIQUE_COLOR_PROBES,
            ReflectionTechnique.ENVIRONMENT: TECHNIQUE_COLOR_ENV,
            ReflectionTechnique.PLANAR: TECHNIQUE_COLOR_PLANAR,
        }

    def set_technique_data(
        self,
        pixels: List[ReflectionPixel],
        width: int,
        height: int,
    ) -> None:
        """Set reflection technique data.

        Args:
            pixels: List of per-pixel technique data.
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        self._pixels = pixels
        self._width = width
        self._height = height

    def get_technique_color(self, technique: ReflectionTechnique) -> Color:
        """Get the color for a reflection technique.

        Args:
            technique: Reflection technique.

        Returns:
            Color for the technique.
        """
        return self._technique_colors.get(technique, TECHNIQUE_COLOR_NONE)

    @debug
    def render_mask(self) -> List[Tuple[Tuple[int, int], Color]]:
        """Render the reflection technique mask.

        Returns:
            List of (pixel_coord, color) for the mask.
            Returns None if debug mode is disabled.
        """
        result = []

        for pixel in self._pixels:
            color = self.get_technique_color(pixel.technique)
            # Apply contribution as alpha
            color = Color(color.r, color.g, color.b, pixel.contribution)
            result.append(((pixel.x, pixel.y), color))

        return result

    def get_technique_coverage(self) -> Dict[ReflectionTechnique, float]:
        """Get coverage ratio for each reflection technique.

        Returns:
            Dictionary mapping technique to coverage ratio.
        """
        if not self._pixels:
            return {t: 0.0 for t in ReflectionTechnique}

        counts: Dict[ReflectionTechnique, int] = {t: 0 for t in ReflectionTechnique}
        for pixel in self._pixels:
            counts[pixel.technique] += 1

        total = len(self._pixels)
        return {t: c / total for t, c in counts.items()}

    def get_transition_pixels(self) -> List[Tuple[int, int]]:
        """Find pixels at technique transition boundaries.

        Returns:
            List of pixel coordinates at transitions.
        """
        if not self._pixels or self._width == 0 or self._height == 0:
            return []

        # Build grid for fast lookup
        grid: Dict[Tuple[int, int], ReflectionTechnique] = {}
        for pixel in self._pixels:
            grid[(pixel.x, pixel.y)] = pixel.technique

        transitions = []
        for pixel in self._pixels:
            x, y = pixel.x, pixel.y
            current = pixel.technique

            # Check 4-connected neighbours
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in grid and grid[(nx, ny)] != current:
                    transitions.append((x, y))
                    break

        return transitions


# =============================================================================
# Unified GI Debug Overlay
# =============================================================================


class GIDebugOverlay:
    """Unified debug overlay system for all GI visualizations.

    Manages and composites multiple debug visualizations:
    - Probe grid positions
    - Voxel occupancy
    - SSGI confidence
    - Path tracer comparison
    - Reflection technique mask

    Supports:
    - Individual visualization toggles
    - Global opacity control
    - Overlay compositing
    """

    def __init__(self, config: Optional[GIDebugConfig] = None) -> None:
        """Initialize the GI debug overlay.

        Args:
            config: Debug configuration. Uses defaults if None.
        """
        self._config = config or GIDebugConfig()

        # Initialize all visualizations
        self._probe_vis = ProbeGridVisualization(self._config)
        self._voxel_vis = VoxelOccupancyVisualization(self._config)
        self._ssgi_vis = SSGIConfidenceHeatmap(self._config)
        self._comparison_vis = PathTracerComparisonHeatmap(self._config)
        self._reflection_vis = ReflectionTechniqueMask(self._config)

    @property
    def config(self) -> GIDebugConfig:
        """Get the current debug configuration."""
        return self._config

    @property
    def probe_visualization(self) -> ProbeGridVisualization:
        """Get the probe grid visualization."""
        return self._probe_vis

    @property
    def voxel_visualization(self) -> VoxelOccupancyVisualization:
        """Get the voxel occupancy visualization."""
        return self._voxel_vis

    @property
    def ssgi_visualization(self) -> SSGIConfidenceHeatmap:
        """Get the SSGI confidence visualization."""
        return self._ssgi_vis

    @property
    def comparison_visualization(self) -> PathTracerComparisonHeatmap:
        """Get the path tracer comparison visualization."""
        return self._comparison_vis

    @property
    def reflection_visualization(self) -> ReflectionTechniqueMask:
        """Get the reflection technique visualization."""
        return self._reflection_vis

    def toggle(
        self,
        visualization: str,
        enabled: Optional[bool] = None,
    ) -> bool:
        """Toggle a specific visualization on or off.

        Args:
            visualization: Name of the visualization:
                'probes', 'voxels', 'ssgi', 'comparison', 'reflection'
            enabled: Explicitly set state, or toggle if None.

        Returns:
            New enabled state of the visualization.

        Raises:
            ValueError: If visualization name is unknown.
        """
        attr_map = {
            "probes": "show_probes",
            "voxels": "show_voxels",
            "ssgi": "show_ssgi_confidence",
            "comparison": "show_path_tracer_diff",
            "reflection": "show_reflection_mask",
        }

        if visualization not in attr_map:
            raise ValueError(
                f"Unknown visualization: {visualization}. "
                f"Valid options: {list(attr_map.keys())}"
            )

        attr = attr_map[visualization]
        current = getattr(self._config, attr)

        if enabled is None:
            new_value = not current
        else:
            new_value = enabled

        setattr(self._config, attr, new_value)
        return new_value

    def set_opacity(self, opacity: float) -> None:
        """Set the global overlay opacity.

        Args:
            opacity: Opacity value [0, 1].
        """
        self._config.overlay_opacity = max(0.0, min(1.0, opacity))

    def enable_all(self) -> None:
        """Enable all debug visualizations."""
        self._config.show_probes = True
        self._config.show_voxels = True
        self._config.show_ssgi_confidence = True
        self._config.show_path_tracer_diff = True
        self._config.show_reflection_mask = True

    def disable_all(self) -> None:
        """Disable all debug visualizations."""
        self._config.show_probes = False
        self._config.show_voxels = False
        self._config.show_ssgi_confidence = False
        self._config.show_path_tracer_diff = False
        self._config.show_reflection_mask = False

    @debug
    def render_all(
        self,
    ) -> Dict[str, Any]:
        """Render all enabled debug visualizations.

        Returns:
            Dictionary containing render data for each enabled visualization:
            {
                'probes': [...] or None,
                'voxels_wireframe': [...] or None,
                'voxels_slice': [...] or None,
                'ssgi_heatmap': [...] or None,
                'comparison_heatmap': [...] or None,
                'reflection_mask': [...] or None,
            }
            Returns None if debug mode is disabled.
        """
        result: Dict[str, Any] = {
            "probes": None,
            "voxels_wireframe": None,
            "voxels_slice": None,
            "ssgi_heatmap": None,
            "comparison_heatmap": None,
            "reflection_mask": None,
        }

        if self._config.show_probes:
            result["probes"] = self._probe_vis.render_probes()

        if self._config.show_voxels:
            result["voxels_wireframe"] = self._voxel_vis.render_wireframe()
            result["voxels_slice"] = self._voxel_vis.render_slice()

        if self._config.show_ssgi_confidence:
            result["ssgi_heatmap"] = self._ssgi_vis.render_heatmap()

        if self._config.show_path_tracer_diff:
            result["comparison_heatmap"] = self._comparison_vis.render_difference()

        if self._config.show_reflection_mask:
            result["reflection_mask"] = self._reflection_vis.render_mask()

        return result

    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics from all visualization systems.

        Returns:
            Dictionary containing statistics for each system.
        """
        return {
            "probes": {
                "count": self._probe_vis.probe_count,
            },
            "voxels": {
                "count": self._voxel_vis.voxel_count,
                "total": self._voxel_vis.total_voxels,
                "occupancy_ratio": self._voxel_vis.occupancy_ratio,
            },
            "ssgi": self._ssgi_vis.get_statistics(),
            "comparison": {
                "stats": self._comparison_vis.compute_stats(),
                "quality": self._comparison_vis.get_psnr_quality(),
            },
            "reflection": {
                "coverage": self._reflection_vis.get_technique_coverage(),
            },
        }


# =============================================================================
# WGSL Shader Generation
# =============================================================================


def generate_debug_overlay_wgsl() -> str:
    """Generate WGSL shader code for debug overlay compositing.

    Returns:
        WGSL shader source code.
    """
    return '''
// GI Debug Overlay Composite Shader
// T-GIR-P10.1

struct DebugConfig {
    overlay_opacity: f32,
    show_probes: u32,
    show_voxels: u32,
    show_ssgi: u32,
    show_comparison: u32,
    show_reflection: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> config: DebugConfig;
@group(0) @binding(1) var scene_texture: texture_2d<f32>;
@group(0) @binding(2) var ssgi_heatmap: texture_2d<f32>;
@group(0) @binding(3) var comparison_heatmap: texture_2d<f32>;
@group(0) @binding(4) var reflection_mask: texture_2d<f32>;
@group(0) @binding(5) var output_texture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(scene_texture);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let uv = vec2<i32>(global_id.xy);
    var color = textureLoad(scene_texture, uv, 0);
    let opacity = config.overlay_opacity;

    // Composite SSGI confidence heatmap
    if (config.show_ssgi != 0u) {
        let ssgi_color = textureLoad(ssgi_heatmap, uv, 0);
        color = mix(color, ssgi_color, ssgi_color.a * opacity);
    }

    // Composite path tracer comparison
    if (config.show_comparison != 0u) {
        let comp_color = textureLoad(comparison_heatmap, uv, 0);
        color = mix(color, comp_color, comp_color.a * opacity);
    }

    // Composite reflection technique mask
    if (config.show_reflection != 0u) {
        let refl_color = textureLoad(reflection_mask, uv, 0);
        color = mix(color, refl_color, refl_color.a * opacity * 0.5);
    }

    textureStore(output_texture, uv, color);
}
'''


def generate_probe_billboard_wgsl() -> str:
    """Generate WGSL shader for probe billboard rendering.

    Returns:
        WGSL shader source code.
    """
    return '''
// GI Debug Probe Billboard Shader
// T-GIR-P10.1

struct ProbeData {
    position: vec3<f32>,
    radius: f32,
    color: vec4<f32>,
}

struct CameraData {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraData;
@group(0) @binding(1) var<storage, read> probes: array<ProbeData>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let probe = probes[instance_idx];

    // Quad vertices
    let quad_verts = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );

    let local_pos = quad_verts[vertex_idx];

    // Billboard orientation (face camera)
    let to_camera = normalize(camera.camera_pos - probe.position);
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), to_camera));
    let up = cross(to_camera, right);

    let world_pos = probe.position
        + right * local_pos.x * probe.radius
        + up * local_pos.y * probe.radius;

    var output: VertexOutput;
    output.position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    output.color = probe.color;
    output.uv = local_pos * 0.5 + 0.5;

    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Circle mask
    let dist = length(input.uv - vec2<f32>(0.5));
    if (dist > 0.5) {
        discard;
    }

    // Soft edge
    let alpha = 1.0 - smoothstep(0.4, 0.5, dist);

    return vec4<f32>(input.color.rgb, input.color.a * alpha);
}
'''


def generate_voxel_wireframe_wgsl() -> str:
    """Generate WGSL shader for voxel wireframe rendering.

    Returns:
        WGSL shader source code.
    """
    return '''
// GI Debug Voxel Wireframe Shader
// T-GIR-P10.1

struct LineVertex {
    position: vec3<f32>,
    color: vec4<f32>,
}

struct CameraData {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraData;
@group(0) @binding(1) var<storage, read> vertices: array<LineVertex>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    let vertex = vertices[vertex_idx];

    var output: VertexOutput;
    output.position = camera.view_proj * vec4<f32>(vertex.position, 1.0);
    output.color = vertex.color;

    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
'''


# =============================================================================
# Utility Functions
# =============================================================================


def estimate_debug_memory(
    probe_count: int,
    voxel_count: int,
    width: int,
    height: int,
) -> int:
    """Estimate memory usage for debug visualization.

    Args:
        probe_count: Number of probes to visualize.
        voxel_count: Number of occupied voxels.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Estimated memory usage in bytes.
    """
    # ProbeVisualizationData: ~48 bytes each
    probe_memory = probe_count * 48

    # VoxelData: ~24 bytes each
    voxel_memory = voxel_count * 24

    # Heatmap textures: 4 bytes per pixel (RGBA8)
    pixel_count = width * height
    heatmap_memory = pixel_count * 4 * 3  # SSGI, comparison, reflection

    return probe_memory + voxel_memory + heatmap_memory


def create_test_probes(
    grid_size: Tuple[int, int, int] = (4, 4, 2),
    spacing: float = 2.0,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> List[ProbeVisualizationData]:
    """Create test probe data for visualization testing.

    Args:
        grid_size: Number of probes in each dimension.
        spacing: Distance between probes.
        origin: World position of the first probe.

    Returns:
        List of test probe data.
    """
    probes = []

    for z in range(grid_size[2]):
        for y in range(grid_size[1]):
            for x in range(grid_size[0]):
                position = (
                    origin[0] + x * spacing,
                    origin[1] + y * spacing,
                    origin[2] + z * spacing,
                )

                # Varying irradiance based on position
                irradiance = (
                    x / max(grid_size[0] - 1, 1),
                    y / max(grid_size[1] - 1, 1),
                    z / max(grid_size[2] - 1, 1),
                )

                probes.append(ProbeVisualizationData(
                    position=position,
                    irradiance=irradiance,
                    state=ProbeState.ACTIVE,
                    depth=0,
                ))

    return probes


def create_test_voxels(
    resolution: Tuple[int, int, int] = (8, 8, 8),
    fill_ratio: float = 0.3,
) -> List[VoxelData]:
    """Create test voxel data for visualization testing.

    Args:
        resolution: Grid resolution.
        fill_ratio: Ratio of voxels to fill.

    Returns:
        List of test voxel data.
    """
    import random

    voxels = []
    total = resolution[0] * resolution[1] * resolution[2]
    target_count = int(total * fill_ratio)

    positions = [
        (x, y, z)
        for x in range(resolution[0])
        for y in range(resolution[1])
        for z in range(resolution[2])
    ]

    random.shuffle(positions)

    for pos in positions[:target_count]:
        voxels.append(VoxelData(
            position=pos,
            density=random.uniform(0.3, 1.0),
        ))

    return voxels
