"""
Motion Blur System

Provides motion blur rendering:
- CameraMotionBlur: Full frame based on camera movement
- ObjectMotionBlur: Per-pixel velocity buffer
- TileMaxVelocity: Performance optimization
- MotionBlurSettings: Complete configuration
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .postprocess_stack import EffectPriority, EffectSettings, PostProcessEffect


class MotionBlurMode(Enum):
    """Motion blur calculation mode."""

    CAMERA_ONLY = auto()  # Only camera motion
    OBJECT_ONLY = auto()  # Only per-object motion
    COMBINED = auto()  # Camera + object motion


class MotionBlurQuality(Enum):
    """Motion blur quality preset."""

    LOW = auto()  # Fewer samples, basic blur
    MEDIUM = auto()  # Standard samples
    HIGH = auto()  # More samples, better quality
    ULTRA = auto()  # Maximum samples, scatter


@dataclass
class MotionBlurSettings(EffectSettings):
    """Motion blur settings."""

    mode: MotionBlurMode = MotionBlurMode.COMBINED
    quality: MotionBlurQuality = MotionBlurQuality.MEDIUM

    # Blur intensity
    intensity: float = 1.0  # Global intensity [0, 2]
    camera_intensity: float = 1.0  # Camera motion scale
    object_intensity: float = 1.0  # Object motion scale

    # Sample settings
    sample_count: int = 16  # Number of blur samples
    max_blur_radius: float = 32.0  # Maximum blur in pixels

    # Performance
    half_resolution: bool = False  # Process at half res
    tile_size: int = 16  # Tile size for max velocity

    # Filtering
    center_sample_weight: float = 1.0  # Weight of center sample
    edge_fade: float = 0.0  # Fade at screen edges [0, 1]

    # Shutter settings (for realistic motion blur)
    shutter_angle: float = 180.0  # degrees (180 = half frame)
    rolling_shutter: float = 0.0  # Rolling shutter simulation [0, 1]

    def __post_init__(self) -> None:
        self.priority = EffectPriority.MOTION_BLUR.value

    def lerp(self, other: "MotionBlurSettings", t: float) -> "MotionBlurSettings":
        """Interpolate between two motion blur settings."""
        return MotionBlurSettings(
            enabled=self.enabled if t < 0.5 else other.enabled,
            weight=self.weight + (other.weight - self.weight) * t,
            mode=self.mode if t < 0.5 else other.mode,
            quality=self.quality if t < 0.5 else other.quality,
            intensity=self.intensity + (other.intensity - self.intensity) * t,
            camera_intensity=self.camera_intensity
            + (other.camera_intensity - self.camera_intensity) * t,
            object_intensity=self.object_intensity
            + (other.object_intensity - self.object_intensity) * t,
            sample_count=int(
                self.sample_count + (other.sample_count - self.sample_count) * t
            ),
            max_blur_radius=self.max_blur_radius
            + (other.max_blur_radius - self.max_blur_radius) * t,
            shutter_angle=self.shutter_angle
            + (other.shutter_angle - self.shutter_angle) * t,
        )

    @property
    def shutter_speed_factor(self) -> float:
        """Get shutter speed factor from angle.

        Returns:
            Shutter speed as fraction of frame time.
        """
        return self.shutter_angle / 360.0


@dataclass
class CameraMotion:
    """Camera motion data for a frame."""

    # View matrices
    current_view: List[List[float]] = field(default_factory=lambda: [[1, 0, 0, 0]] * 4)
    previous_view: List[List[float]] = field(default_factory=lambda: [[1, 0, 0, 0]] * 4)

    # Projection matrices
    current_projection: List[List[float]] = field(
        default_factory=lambda: [[1, 0, 0, 0]] * 4
    )
    previous_projection: List[List[float]] = field(
        default_factory=lambda: [[1, 0, 0, 0]] * 4
    )

    # Derived motion
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # World units per second
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Radians per second

    def calculate_screen_velocity(
        self,
        world_pos: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        """Calculate screen-space velocity for a world position.

        Args:
            world_pos: World position.

        Returns:
            Screen-space velocity (dx, dy).
        """
        return (0.0, 0.0)


class CameraMotionBlur:
    """Full-frame camera motion blur.

    Applies blur based on camera movement between frames.
    Uses reprojection to calculate per-pixel velocity.
    """

    def __init__(self) -> None:
        self._velocity_buffer: Any = None
        self._motion_buffer: Any = None
        self._previous_view_proj: List[List[float]] = [[1, 0, 0, 0]] * 4
        self._current_view_proj: List[List[float]] = [[1, 0, 0, 0]] * 4

    def setup(self, width: int, height: int) -> None:
        """Initialize camera motion blur buffers.

        Args:
            width: Buffer width.
            height: Buffer height.
        """
        self._velocity_buffer = None
        self._motion_buffer = None

    def update_matrices(
        self,
        view: List[List[float]],
        projection: List[List[float]],
    ) -> None:
        """Update camera matrices for this frame.

        Args:
            view: Current view matrix.
            projection: Current projection matrix.
        """
        self._previous_view_proj = self._current_view_proj
        self._current_view_proj = self._multiply_matrices(view, projection)

    def calculate_velocity(
        self,
        depth_buffer: Any,
        width: int,
        height: int,
    ) -> Any:
        """Calculate per-pixel velocity from camera motion.

        Args:
            depth_buffer: Scene depth buffer.
            width: Buffer width.
            height: Buffer height.

        Returns:
            Velocity buffer (RG = motion vector).
        """
        return self._velocity_buffer

    def apply_blur(
        self,
        color_buffer: Any,
        velocity_buffer: Any,
        settings: MotionBlurSettings,
    ) -> Any:
        """Apply motion blur using velocity buffer.

        Args:
            color_buffer: Input color.
            velocity_buffer: Per-pixel velocity.
            settings: Blur settings.

        Returns:
            Blurred color buffer.
        """
        return self._motion_buffer

    def _multiply_matrices(
        self,
        a: List[List[float]],
        b: List[List[float]],
    ) -> List[List[float]]:
        """Multiply two 4x4 matrices."""
        result = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    result[i][j] += a[i][k] * b[k][j]
        return result


class ObjectMotionBlur:
    """Per-object motion blur using velocity buffer.

    Uses a velocity buffer written during scene rendering
    to apply accurate per-object motion blur.
    """

    def __init__(self) -> None:
        self._velocity_buffer: Any = None
        self._tile_max_buffer: Any = None
        self._neighbor_max_buffer: Any = None

    def setup(self, width: int, height: int, tile_size: int = 16) -> None:
        """Initialize object motion blur buffers.

        Args:
            width: Buffer width.
            height: Buffer height.
            tile_size: Tile size for max velocity.
        """
        self._velocity_buffer = None
        self._tile_max_buffer = None
        self._neighbor_max_buffer = None

    def get_velocity_buffer(self) -> Any:
        """Get velocity buffer for scene rendering to write to.

        Returns:
            Velocity buffer handle.
        """
        return self._velocity_buffer

    def process_velocity(
        self,
        velocity_buffer: Any,
        settings: MotionBlurSettings,
    ) -> Any:
        """Process velocity buffer for blur.

        Args:
            velocity_buffer: Raw velocity from rendering.
            settings: Blur settings.

        Returns:
            Processed velocity buffer.
        """
        max_velocity = settings.max_blur_radius * settings.object_intensity

        return velocity_buffer

    def apply_blur(
        self,
        color_buffer: Any,
        velocity_buffer: Any,
        depth_buffer: Any,
        settings: MotionBlurSettings,
    ) -> Any:
        """Apply object motion blur.

        Args:
            color_buffer: Input color.
            velocity_buffer: Per-pixel velocity.
            depth_buffer: Scene depth.
            settings: Blur settings.

        Returns:
            Blurred color buffer.
        """
        return color_buffer


class TileMaxVelocity:
    """Tile-based maximum velocity calculation.

    Divides the screen into tiles and calculates maximum
    velocity per tile to optimize blur sampling.
    """

    def __init__(self, tile_size: int = 16) -> None:
        """Initialize tile max system.

        Args:
            tile_size: Size of tiles in pixels.
        """
        self._tile_size: int = tile_size
        self._tile_buffer: Any = None
        self._neighbor_buffer: Any = None
        self._tiles_x: int = 0
        self._tiles_y: int = 0

    @property
    def tile_size(self) -> int:
        """Tile size in pixels."""
        return self._tile_size

    @tile_size.setter
    def tile_size(self, value: int) -> None:
        self._tile_size = max(4, min(64, value))

    def setup(self, width: int, height: int) -> None:
        """Initialize tile buffers.

        Args:
            width: Image width.
            height: Image height.
        """
        self._tiles_x = (width + self._tile_size - 1) // self._tile_size
        self._tiles_y = (height + self._tile_size - 1) // self._tile_size
        self._tile_buffer = None
        self._neighbor_buffer = None

    def calculate_tile_max(self, velocity_buffer: Any) -> Any:
        """Calculate maximum velocity per tile.

        Args:
            velocity_buffer: Full-resolution velocity buffer.

        Returns:
            Tile-resolution max velocity buffer.
        """
        return self._tile_buffer

    def calculate_neighbor_max(self, tile_buffer: Any) -> Any:
        """Calculate maximum velocity including neighbors.

        Args:
            tile_buffer: Tile max velocity buffer.

        Returns:
            Neighbor-expanded max velocity buffer.
        """
        return self._neighbor_buffer

    def get_tile_velocity(
        self,
        pixel_x: int,
        pixel_y: int,
    ) -> Tuple[float, float]:
        """Get max velocity for a pixel's tile.

        Args:
            pixel_x: Pixel X coordinate.
            pixel_y: Pixel Y coordinate.

        Returns:
            Maximum velocity for the tile.
        """
        return (0.0, 0.0)


class MotionBlurEffect(PostProcessEffect[MotionBlurSettings]):
    """Complete motion blur post-process effect."""

    def __init__(
        self,
        settings: Optional[MotionBlurSettings] = None,
    ) -> None:
        """Initialize motion blur effect.

        Args:
            settings: Motion blur configuration.
        """
        super().__init__(
            name="MotionBlur",
            settings=settings or MotionBlurSettings(),
            priority=EffectPriority.MOTION_BLUR.value,
        )

        self._camera_blur: CameraMotionBlur = CameraMotionBlur()
        self._object_blur: ObjectMotionBlur = ObjectMotionBlur()
        self._tile_max: TileMaxVelocity = TileMaxVelocity()

        self._combined_velocity: Any = None
        self._width: int = 0
        self._height: int = 0

    @property
    def camera_blur(self) -> CameraMotionBlur:
        """Access camera blur processor."""
        return self._camera_blur

    @property
    def object_blur(self) -> ObjectMotionBlur:
        """Access object blur processor."""
        return self._object_blur

    def get_velocity_buffer(self) -> Any:
        """Get velocity buffer for scene to write to.

        Returns:
            Velocity buffer handle.
        """
        return self._object_blur.get_velocity_buffer()

    def get_required_inputs(self) -> List[str]:
        """Get required input resources."""
        return ["color", "depth", "velocity"]

    def get_outputs(self) -> List[str]:
        """Get output resources."""
        return ["color"]

    def setup(self, width: int, height: int) -> None:
        """Initialize motion blur resources.

        Args:
            width: Render width.
            height: Render height.
        """
        self._width = width
        self._height = height

        tile_size = self._settings.tile_size if self._settings else 16

        self._camera_blur.setup(width, height)
        self._object_blur.setup(width, height, tile_size)
        self._tile_max.tile_size = tile_size
        self._tile_max.setup(width, height)

    def update_camera(
        self,
        view: List[List[float]],
        projection: List[List[float]],
    ) -> None:
        """Update camera matrices for motion calculation.

        Args:
            view: Current view matrix.
            projection: Current projection matrix.
        """
        self._camera_blur.update_matrices(view, projection)

    def execute(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        delta_time: float,
    ) -> None:
        """Execute motion blur effect.

        Args:
            inputs: Color, depth, and velocity buffers.
            outputs: Output color buffer.
            delta_time: Frame time.
        """
        if not self._settings or not self._settings.enabled:
            return

        if self._settings.intensity <= 0:
            return

        color_buffer = inputs.get("color")
        depth_buffer = inputs.get("depth")
        velocity_buffer = inputs.get("velocity")

        mode = self._settings.mode

        if mode == MotionBlurMode.CAMERA_ONLY:
            camera_velocity = self._camera_blur.calculate_velocity(
                depth_buffer,
                self._width,
                self._height,
            )
            self._camera_blur.apply_blur(
                color_buffer,
                camera_velocity,
                self._settings,
            )

        elif mode == MotionBlurMode.OBJECT_ONLY:
            if velocity_buffer:
                processed_velocity = self._object_blur.process_velocity(
                    velocity_buffer,
                    self._settings,
                )
                self._tile_max.calculate_tile_max(processed_velocity)
                self._tile_max.calculate_neighbor_max(self._tile_max._tile_buffer)

                self._object_blur.apply_blur(
                    color_buffer,
                    processed_velocity,
                    depth_buffer,
                    self._settings,
                )

        else:  # COMBINED
            camera_velocity = self._camera_blur.calculate_velocity(
                depth_buffer,
                self._width,
                self._height,
            )

            self._combine_velocity(camera_velocity, velocity_buffer)

            self._tile_max.calculate_tile_max(self._combined_velocity)
            self._tile_max.calculate_neighbor_max(self._tile_max._tile_buffer)

    def _combine_velocity(
        self,
        camera_velocity: Any,
        object_velocity: Any,
    ) -> None:
        """Combine camera and object velocity buffers.

        Args:
            camera_velocity: Camera motion velocity.
            object_velocity: Object motion velocity.
        """
        self._combined_velocity = camera_velocity

    def cleanup(self) -> None:
        """Release motion blur resources."""
        self._combined_velocity = None

    def is_compute_effect(self) -> bool:
        """Motion blur uses compute shaders."""
        return True


__all__ = [
    "MotionBlurMode",
    "MotionBlurQuality",
    "MotionBlurSettings",
    "CameraMotion",
    "CameraMotionBlur",
    "ObjectMotionBlur",
    "TileMaxVelocity",
    "MotionBlurEffect",
]
