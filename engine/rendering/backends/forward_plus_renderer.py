"""Forward+ Renderer for Low-tier quality settings (T-CC-0.10).

Forward+ rendering pipeline optimized for mobile/low-end hardware:
1. Depth pre-pass: Early-Z culling
2. Light culling: Screen-space tile-based or view-space
3. Forward shading: Combined material + lighting pass
4. Tone mapping: LDR output

This renderer is used when QualityTier.LOW is selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple

from trinity.decorators import component
from trinity.types import QualityTier

if TYPE_CHECKING:
    from engine.platform.rhi.resources import Buffer, Texture

# Constants
MAX_LIGHTS_LOW_TIER = 8  # Maximum lights for Low tier
MAX_LIGHTS_MEDIUM_TIER = 32  # Maximum lights for Medium tier
MAX_LIGHTS_HIGH_TIER = 64  # Maximum lights for High/Ultra tier
TILE_SIZE = 16  # Screen-space tile size for light culling
MAX_LIGHTS_PER_TILE = 32


class ForwardPassType(Enum):
    """Forward rendering pass types."""

    DEPTH_PREPASS = auto()
    LIGHT_CULL = auto()
    FORWARD_SHADE = auto()
    TONEMAP = auto()


class ToneMapOperator(Enum):
    """Tone mapping operator types."""

    REINHARD = auto()
    REINHARD_EXTENDED = auto()
    ACES = auto()
    UNCHARTED2 = auto()
    NONE = auto()


@dataclass(slots=True)
class ForwardPlusConfig:
    """Configuration for Forward+ renderer.

    Attributes:
        max_lights: Maximum number of active lights.
        tile_size: Screen-space tile size for light culling (pixels).
        max_lights_per_tile: Maximum lights per screen tile.
        enable_depth_prepass: Enable depth pre-pass for early-Z.
        enable_light_culling: Enable tile-based light culling.
        tonemap_enabled: Enable tone mapping.
        tonemap_operator: Tone mapping operator type.
        exposure: Exposure value for tone mapping.
        quality_tier: Quality tier this config is optimized for.
    """

    max_lights: int = MAX_LIGHTS_LOW_TIER
    tile_size: int = TILE_SIZE
    max_lights_per_tile: int = MAX_LIGHTS_PER_TILE
    enable_depth_prepass: bool = True
    enable_light_culling: bool = True
    tonemap_enabled: bool = True
    tonemap_operator: ToneMapOperator = ToneMapOperator.REINHARD
    exposure: float = 1.0
    quality_tier: QualityTier = QualityTier.LOW

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_lights < 0:
            raise ValueError(
                f"max_lights must be non-negative, got {self.max_lights}"
            )
        if self.tile_size < 1:
            raise ValueError(
                f"tile_size must be positive, got {self.tile_size}"
            )
        if self.max_lights_per_tile < 1:
            raise ValueError(
                f"max_lights_per_tile must be positive, got {self.max_lights_per_tile}"
            )
        if self.exposure <= 0.0:
            raise ValueError(
                f"exposure must be positive, got {self.exposure}"
            )


@dataclass(slots=True)
class LightTile:
    """Screen-space tile containing light indices.

    Attributes:
        x: Tile X coordinate.
        y: Tile Y coordinate.
        light_count: Number of lights affecting this tile.
        light_indices: Indices into the light buffer.
    """

    x: int
    y: int
    light_count: int = 0
    light_indices: List[int] = field(default_factory=list)

    def clear(self) -> None:
        """Clear light data for new frame."""
        self.light_count = 0
        self.light_indices.clear()

    def add_light(self, light_index: int, max_per_tile: int) -> bool:
        """Add a light to this tile.

        Args:
            light_index: Index of the light.
            max_per_tile: Maximum lights allowed per tile.

        Returns:
            True if light was added, False if tile is full.
        """
        if self.light_count >= max_per_tile:
            return False
        self.light_indices.append(light_index)
        self.light_count += 1
        return True


@dataclass(slots=True)
class ForwardPlusPass:
    """Describes a single pass in the Forward+ pipeline.

    Attributes:
        pass_type: Type of this pass.
        name: Human-readable name.
        enabled: Whether this pass is enabled.
    """

    pass_type: ForwardPassType
    name: str
    enabled: bool = True


@dataclass(slots=True)
class ForwardPlusStats:
    """Runtime statistics for Forward+ renderer.

    Attributes:
        depth_prepass_time_ms: Time spent in depth pre-pass.
        light_cull_time_ms: Time spent in light culling.
        forward_shade_time_ms: Time spent in forward shading.
        tonemap_time_ms: Time spent in tone mapping.
        total_time_ms: Total frame time.
        visible_lights: Number of visible lights this frame.
        tiles_with_lights: Number of tiles with at least one light.
        total_tiles: Total number of tiles.
        draw_calls: Number of draw calls issued.
        triangles_rendered: Approximate triangles rendered.
    """

    depth_prepass_time_ms: float = 0.0
    light_cull_time_ms: float = 0.0
    forward_shade_time_ms: float = 0.0
    tonemap_time_ms: float = 0.0
    total_time_ms: float = 0.0
    visible_lights: int = 0
    tiles_with_lights: int = 0
    total_tiles: int = 0
    draw_calls: int = 0
    triangles_rendered: int = 0

    def reset(self) -> None:
        """Reset all statistics."""
        self.depth_prepass_time_ms = 0.0
        self.light_cull_time_ms = 0.0
        self.forward_shade_time_ms = 0.0
        self.tonemap_time_ms = 0.0
        self.total_time_ms = 0.0
        self.visible_lights = 0
        self.tiles_with_lights = 0
        self.total_tiles = 0
        self.draw_calls = 0
        self.triangles_rendered = 0

    @property
    def average_lights_per_tile(self) -> float:
        """Calculate average lights per tile."""
        if self.tiles_with_lights == 0:
            return 0.0
        return self.visible_lights / self.tiles_with_lights


@dataclass(slots=True)
class LightData:
    """Light data for Forward+ rendering.

    Attributes:
        position: World-space position (x, y, z).
        radius: Light radius/range.
        color: Light color (r, g, b).
        intensity: Light intensity multiplier.
        light_type: Type identifier (0=point, 1=spot, 2=directional).
    """

    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 10.0
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    light_type: int = 0  # 0=point, 1=spot, 2=directional


@component
class ForwardPlusRenderer:
    """Forward+ renderer for low-tier quality settings.

    Implements a Forward+ rendering pipeline optimized for mobile
    and low-end hardware. Uses tile-based light culling to reduce
    per-pixel lighting cost.

    Pipeline:
        1. Depth pre-pass: Render depth buffer only
        2. Light culling: Build per-tile light lists
        3. Forward shade: Render materials with tiled lighting
        4. Tone map: Convert HDR to LDR output

    Example:
        renderer = ForwardPlusRenderer(config=ForwardPlusConfig())
        renderer.initialize(width=1280, height=720)
        renderer.begin_frame()
        renderer.execute_depth_prepass(scene)
        renderer.execute_light_culling(lights)
        renderer.execute_forward_shading(scene)
        renderer.execute_tonemapping()
        renderer.end_frame()

    Attributes:
        config: Renderer configuration.
        stats: Runtime statistics.
    """

    _component_name: str = "ForwardPlusRenderer"

    def __init__(self, config: Optional[ForwardPlusConfig] = None) -> None:
        """Initialize Forward+ renderer.

        Args:
            config: Renderer configuration. Uses defaults if None.
        """
        self._config = config or ForwardPlusConfig()
        self._stats = ForwardPlusStats()
        self._initialized = False
        self._width = 0
        self._height = 0
        self._tiles: List[List[LightTile]] = []
        self._passes: List[ForwardPlusPass] = []
        self._depth_buffer: Optional["Texture"] = None
        self._light_buffer: Optional["Buffer"] = None
        self._active_lights: List[LightData] = []
        self._frame_count = 0
        self._setup_passes()

    def _setup_passes(self) -> None:
        """Configure render passes based on config."""
        self._passes = [
            ForwardPlusPass(
                ForwardPassType.DEPTH_PREPASS,
                "depth_prepass",
                self._config.enable_depth_prepass,
            ),
            ForwardPlusPass(
                ForwardPassType.LIGHT_CULL,
                "light_cull",
                self._config.enable_light_culling,
            ),
            ForwardPlusPass(
                ForwardPassType.FORWARD_SHADE,
                "forward_shade",
                True,  # Always enabled
            ),
            ForwardPlusPass(
                ForwardPassType.TONEMAP,
                "tonemap",
                self._config.tonemap_enabled,
            ),
        ]

    @property
    def config(self) -> ForwardPlusConfig:
        """Get renderer configuration."""
        return self._config

    @property
    def stats(self) -> ForwardPlusStats:
        """Get runtime statistics."""
        return self._stats

    @property
    def is_initialized(self) -> bool:
        """Check if renderer has been initialized."""
        return self._initialized

    @property
    def width(self) -> int:
        """Get render width."""
        return self._width

    @property
    def height(self) -> int:
        """Get render height."""
        return self._height

    @property
    def frame_count(self) -> int:
        """Get total frames rendered."""
        return self._frame_count

    @property
    def tile_count(self) -> Tuple[int, int]:
        """Get tile grid dimensions (tiles_x, tiles_y)."""
        if self._width <= 0 or self._height <= 0:
            return (0, 0)
        return (
            (self._width + self._config.tile_size - 1) // self._config.tile_size,
            (self._height + self._config.tile_size - 1) // self._config.tile_size,
        )

    def initialize(self, width: int, height: int) -> bool:
        """Initialize renderer for given resolution.

        Args:
            width: Render width in pixels.
            height: Render height in pixels.

        Returns:
            True if initialization succeeded.

        Raises:
            RuntimeError: If already initialized.
        """
        if self._initialized:
            raise RuntimeError("ForwardPlusRenderer already initialized")

        if width <= 0 or height <= 0:
            return False

        self._width = width
        self._height = height
        self._create_tiles()
        self._initialized = True
        return True

    def _create_tiles(self) -> None:
        """Create screen-space tile grid."""
        tiles_x, tiles_y = self.tile_count
        self._tiles = []
        for y in range(tiles_y):
            row: List[LightTile] = []
            for x in range(tiles_x):
                row.append(LightTile(x=x, y=y))
            self._tiles.append(row)
        self._stats.total_tiles = tiles_x * tiles_y

    def resize(self, width: int, height: int) -> bool:
        """Resize render targets.

        Args:
            width: New render width.
            height: New render height.

        Returns:
            True if resize succeeded.
        """
        if width <= 0 or height <= 0:
            return False

        if not self._initialized:
            return self.initialize(width, height)

        self._width = width
        self._height = height
        self._create_tiles()
        return True

    def begin_frame(self) -> None:
        """Begin a new frame.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            raise RuntimeError("ForwardPlusRenderer not initialized")

        self._stats.reset()
        self._stats.total_tiles = self.tile_count[0] * self.tile_count[1]

        # Clear tile light lists
        for row in self._tiles:
            for tile in row:
                tile.clear()

        self._active_lights.clear()

    def execute_depth_prepass(self, drawables: Optional[List] = None) -> None:
        """Execute depth pre-pass.

        Renders scene geometry to depth buffer only for early-Z culling.

        Args:
            drawables: List of drawable objects. Can be None for stub.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            raise RuntimeError("ForwardPlusRenderer not initialized")

        if not self._config.enable_depth_prepass:
            return

        # Stub: In real impl, issue draw calls with depth-only shader
        if drawables:
            self._stats.draw_calls += len(drawables)

    def execute_light_culling(self, lights: Optional[List[LightData]] = None) -> None:
        """Cull lights into screen-space tiles.

        Args:
            lights: List of light data. Can be None for stub.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            raise RuntimeError("ForwardPlusRenderer not initialized")

        if not self._config.enable_light_culling:
            return

        if lights is None:
            lights = []

        # Limit to max lights for tier
        self._active_lights = lights[: self._config.max_lights]
        self._stats.visible_lights = len(self._active_lights)

        if not self._active_lights:
            return

        # Simple culling: assign lights to tiles based on screen-space projection
        # Real implementation would do frustum-tile intersection
        tiles_with_lights = 0
        for row in self._tiles:
            for tile in row:
                for i, _light in enumerate(self._active_lights):
                    # Simplified: add all lights to all tiles (placeholder)
                    # Real impl: project light sphere to screen, test tile overlap
                    tile.add_light(i, self._config.max_lights_per_tile)
                # Count tile if it has any lights
                if tile.light_count > 0:
                    tiles_with_lights += 1

        self._stats.tiles_with_lights = tiles_with_lights

    def execute_forward_shading(self, drawables: Optional[List] = None) -> None:
        """Execute forward shading pass with tiled lighting.

        Args:
            drawables: List of drawable objects. Can be None for stub.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            raise RuntimeError("ForwardPlusRenderer not initialized")

        # Stub: Issue draw calls with forward shader, sampling per-tile lights
        if drawables:
            self._stats.draw_calls += len(drawables)

    def execute_tonemapping(self) -> None:
        """Apply tone mapping.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            raise RuntimeError("ForwardPlusRenderer not initialized")

        if not self._config.tonemap_enabled:
            return

        # Stub: Apply tone mapping operator (Reinhard, ACES, etc.)
        self._stats.draw_calls += 1

    def end_frame(self) -> None:
        """End frame and collect stats.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            raise RuntimeError("ForwardPlusRenderer not initialized")

        self._stats.total_time_ms = (
            self._stats.depth_prepass_time_ms
            + self._stats.light_cull_time_ms
            + self._stats.forward_shade_time_ms
            + self._stats.tonemap_time_ms
        )
        self._frame_count += 1

    def get_enabled_passes(self) -> List[ForwardPlusPass]:
        """Get list of enabled passes.

        Returns:
            List of enabled ForwardPlusPass objects.
        """
        return [p for p in self._passes if p.enabled]

    def is_pass_enabled(self, pass_type: ForwardPassType) -> bool:
        """Check if a pass type is enabled.

        Args:
            pass_type: Pass type to check.

        Returns:
            True if pass is enabled.
        """
        for p in self._passes:
            if p.pass_type == pass_type:
                return p.enabled
        return False

    def set_pass_enabled(self, pass_type: ForwardPassType, enabled: bool) -> None:
        """Enable or disable a pass.

        Args:
            pass_type: Pass type to modify.
            enabled: Whether to enable or disable.
        """
        for p in self._passes:
            if p.pass_type == pass_type:
                p.enabled = enabled
                break

    def get_tile(self, x: int, y: int) -> Optional[LightTile]:
        """Get tile at grid coordinates.

        Args:
            x: Tile X coordinate.
            y: Tile Y coordinate.

        Returns:
            LightTile at coordinates, or None if out of bounds.
        """
        tiles_x, tiles_y = self.tile_count
        if not (0 <= x < tiles_x and 0 <= y < tiles_y):
            return None
        return self._tiles[y][x]

    def get_tile_at_pixel(self, px: int, py: int) -> Optional[LightTile]:
        """Get tile containing a pixel coordinate.

        Args:
            px: Pixel X coordinate.
            py: Pixel Y coordinate.

        Returns:
            LightTile containing pixel, or None if out of bounds.
        """
        if not self._initialized:
            return None
        if not (0 <= px < self._width and 0 <= py < self._height):
            return None

        tile_x = px // self._config.tile_size
        tile_y = py // self._config.tile_size
        return self.get_tile(tile_x, tile_y)

    def set_exposure(self, exposure: float) -> None:
        """Set tone mapping exposure.

        Args:
            exposure: Exposure value (must be positive).

        Raises:
            ValueError: If exposure is not positive.
        """
        if exposure <= 0.0:
            raise ValueError(f"exposure must be positive, got {exposure}")
        self._config.exposure = exposure

    def set_tonemap_operator(self, operator: ToneMapOperator) -> None:
        """Set tone mapping operator.

        Args:
            operator: Tone mapping operator to use.
        """
        self._config.tonemap_operator = operator

    def destroy(self) -> None:
        """Release renderer resources."""
        self._tiles.clear()
        self._passes.clear()
        self._active_lights.clear()
        self._depth_buffer = None
        self._light_buffer = None
        self._initialized = False
        self._width = 0
        self._height = 0


def create_forward_plus_for_tier(tier: QualityTier) -> ForwardPlusRenderer:
    """Factory function to create Forward+ renderer for quality tier.

    Args:
        tier: Quality tier to configure for.

    Returns:
        Configured ForwardPlusRenderer.
    """
    if tier == QualityTier.LOW:
        config = ForwardPlusConfig(
            max_lights=MAX_LIGHTS_LOW_TIER,
            tile_size=TILE_SIZE,
            max_lights_per_tile=MAX_LIGHTS_PER_TILE,
            enable_depth_prepass=True,
            enable_light_culling=True,
            tonemap_enabled=True,
            tonemap_operator=ToneMapOperator.REINHARD,
            quality_tier=QualityTier.LOW,
        )
    elif tier == QualityTier.MEDIUM:
        config = ForwardPlusConfig(
            max_lights=MAX_LIGHTS_MEDIUM_TIER,
            tile_size=TILE_SIZE,
            max_lights_per_tile=MAX_LIGHTS_PER_TILE,
            enable_depth_prepass=True,
            enable_light_culling=True,
            tonemap_enabled=True,
            tonemap_operator=ToneMapOperator.ACES,
            quality_tier=QualityTier.MEDIUM,
        )
    else:
        # HIGH/ULTRA typically use deferred, but can fall back to Forward+
        config = ForwardPlusConfig(
            max_lights=MAX_LIGHTS_HIGH_TIER,
            tile_size=TILE_SIZE,
            max_lights_per_tile=MAX_LIGHTS_PER_TILE,
            enable_depth_prepass=True,
            enable_light_culling=True,
            tonemap_enabled=True,
            tonemap_operator=ToneMapOperator.ACES,
            quality_tier=tier,
        )
    return ForwardPlusRenderer(config=config)


def get_tier_max_lights(tier: QualityTier) -> int:
    """Get maximum light count for a quality tier.

    Args:
        tier: Quality tier.

    Returns:
        Maximum number of lights.
    """
    if tier == QualityTier.LOW:
        return MAX_LIGHTS_LOW_TIER
    elif tier == QualityTier.MEDIUM:
        return MAX_LIGHTS_MEDIUM_TIER
    else:
        return MAX_LIGHTS_HIGH_TIER
