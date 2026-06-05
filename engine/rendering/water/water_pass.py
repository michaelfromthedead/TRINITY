"""Water rendering pass stub (T-ENV-1.12).

This module provides the base water rendering pass that coordinates
water surface generation, reflection/refraction, and caustics.

Expanded by:
- T-ENV-1.7: Full ocean integration
- T-ENV-1.8: PBR water material
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, List, Optional, Tuple

from trinity.decorators import component

if TYPE_CHECKING:
    from engine.rendering.framegraph import FrameGraph, ResourceHandle
    from engine.platform.rhi.resources import Texture


class WaterQuality(IntEnum):
    """Water rendering quality presets.

    Controls reflection resolution, wave detail, and caustics quality.
    """

    LOW = 0       # No reflections, simple waves
    MEDIUM = 1    # SSR only, medium waves
    HIGH = 2      # SSR + planar, detailed waves
    ULTRA = 3     # Full RT reflections, FFT ocean, caustics


class ReflectionMode(IntEnum):
    """Water reflection rendering modes."""

    NONE = 0          # No reflections
    SKYBOX = 1        # Skybox only
    SSR = 2           # Screen-space reflections
    PLANAR = 3        # Planar reflection render
    RAYTRACED = 4     # Ray-traced reflections


@dataclass
class WaterPassConfig:
    """Configuration for water rendering pass.

    Attributes:
        quality: Quality preset controlling detail levels.
        reflection_mode: Reflection rendering technique.
        refraction_enabled: Enable underwater refraction.
        caustics_enabled: Enable underwater caustics.
        foam_enabled: Enable surface foam.
        underwater_enabled: Enable underwater rendering.
        max_wave_height: Maximum displacement height.
    """

    quality: WaterQuality = WaterQuality.MEDIUM
    reflection_mode: ReflectionMode = ReflectionMode.SSR
    refraction_enabled: bool = True
    caustics_enabled: bool = False
    foam_enabled: bool = True
    underwater_enabled: bool = True
    max_wave_height: float = 10.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_wave_height <= 0.0:
            raise ValueError(
                f"max_wave_height must be positive, got {self.max_wave_height}"
            )
        # Auto-adjust reflection mode based on quality
        if self.quality == WaterQuality.LOW and self.reflection_mode != ReflectionMode.NONE:
            # Allow override but default to cheaper mode
            pass
        if self.quality == WaterQuality.ULTRA and self.reflection_mode == ReflectionMode.NONE:
            self.reflection_mode = ReflectionMode.RAYTRACED


@dataclass
class WaterBody:
    """Definition of a water body in the scene.

    Attributes:
        name: Identifier for this water body.
        bounds: World-space AABB as (min_point, max_point).
        water_level: Base water surface height.
        depth: Maximum water depth.
        flow_direction: Surface flow direction (x, z).
        flow_speed: Flow speed multiplier.
    """

    name: str
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    water_level: float = 0.0
    depth: float = 100.0
    flow_direction: Tuple[float, float] = (1.0, 0.0)
    flow_speed: float = 1.0


@component
class WaterPass:
    """Base water rendering pass.

    Coordinates water rendering including surface generation,
    reflections, refractions, caustics, and underwater effects.

    This is a stub class that will be expanded by T-ENV-1.7 (ocean)
    and T-ENV-1.8 (water materials).

    Example:
        water_pass = WaterPass(config=WaterPassConfig(quality=WaterQuality.HIGH))
        water_pass.setup(frame_graph)
        water_pass.add_water_body(ocean)
        water_pass.execute(view_info)

    Attributes:
        config: Water pass configuration.
        name: Pass name for frame graph registration.
    """

    # Class-level attributes for Trinity component system
    _component_name: str = "WaterPass"

    def __init__(
        self,
        config: Optional[WaterPassConfig] = None,
        name: str = "water_pass",
    ) -> None:
        """Initialize water rendering pass.

        Args:
            config: Pass configuration. Uses defaults if None.
            name: Pass name for identification and debugging.
        """
        self._config = config or WaterPassConfig()
        self._name = name
        self._initialized = False
        self._frame_graph: Optional["FrameGraph"] = None
        self._water_bodies: List[WaterBody] = []
        self._reflection_texture: Optional["Texture"] = None
        self._refraction_texture: Optional["Texture"] = None

    @property
    def config(self) -> WaterPassConfig:
        """Get pass configuration."""
        return self._config

    @property
    def name(self) -> str:
        """Get pass name."""
        return self._name

    @property
    def is_initialized(self) -> bool:
        """Check if pass has been initialized."""
        return self._initialized

    @property
    def water_bodies(self) -> List[WaterBody]:
        """Get registered water bodies."""
        return list(self._water_bodies)

    def setup(self, frame_graph: "FrameGraph") -> "ResourceHandle":
        """Set up water pass in frame graph.

        Registers the pass and declares resource dependencies.

        Args:
            frame_graph: Frame graph to register with.

        Returns:
            Handle to water output resource.

        Raises:
            RuntimeError: If already initialized.
        """
        if self._initialized:
            raise RuntimeError("WaterPass already initialized")

        self._frame_graph = frame_graph
        self._initialized = True

        # Stub: In full implementation, would:
        # 1. Declare depth buffer read
        # 2. Declare color buffer read/write
        # 3. Create reflection/refraction textures
        # 4. Set up caustics resources

        return None  # type: ignore

    def add_water_body(self, water_body: WaterBody) -> None:
        """Add a water body to render.

        Args:
            water_body: Water body definition.

        Raises:
            ValueError: If water body name already exists.
        """
        for existing in self._water_bodies:
            if existing.name == water_body.name:
                raise ValueError(f"Water body '{water_body.name}' already exists")

        self._water_bodies.append(water_body)

    def remove_water_body(self, name: str) -> bool:
        """Remove a water body.

        Args:
            name: Name of water body to remove.

        Returns:
            True if removed, False if not found.
        """
        for i, body in enumerate(self._water_bodies):
            if body.name == name:
                self._water_bodies.pop(i)
                return True
        return False

    def get_water_body(self, name: str) -> Optional[WaterBody]:
        """Get water body by name.

        Args:
            name: Water body name.

        Returns:
            Water body if found, None otherwise.
        """
        for body in self._water_bodies:
            if body.name == name:
                return body
        return None

    def execute(
        self,
        view_position: Tuple[float, float, float],
        view_direction: Tuple[float, float, float],
        time: float,
    ) -> None:
        """Execute water rendering pass.

        Args:
            view_position: Camera world position.
            view_direction: Camera forward direction.
            time: Current simulation time in seconds.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            raise RuntimeError("WaterPass not initialized. Call setup() first.")

        # Stub: In full implementation, would:
        # 1. Update wave simulation
        # 2. Render reflections if enabled
        # 3. Render refractions if enabled
        # 4. Render water surface
        # 5. Apply underwater effects if camera below water

    def is_underwater(self, position: Tuple[float, float, float]) -> bool:
        """Check if a position is underwater.

        Args:
            position: World-space position to check.

        Returns:
            True if position is below any water surface.
        """
        for body in self._water_bodies:
            min_pt, max_pt = body.bounds
            # Check if within horizontal bounds
            if (min_pt[0] <= position[0] <= max_pt[0] and
                min_pt[2] <= position[2] <= max_pt[2]):
                # Check if below water level
                if position[1] < body.water_level:
                    return True
        return False

    def get_water_level_at(self, x: float, z: float) -> Optional[float]:
        """Get water surface height at a position.

        Args:
            x: World X coordinate.
            z: World Z coordinate.

        Returns:
            Water level if within a water body, None otherwise.
        """
        for body in self._water_bodies:
            min_pt, max_pt = body.bounds
            if min_pt[0] <= x <= max_pt[0] and min_pt[2] <= z <= max_pt[2]:
                # Stub: Returns base level, full impl adds wave height
                return body.water_level
        return None

    def destroy(self) -> None:
        """Release pass resources."""
        self._initialized = False
        self._frame_graph = None
        self._water_bodies.clear()
        self._reflection_texture = None
        self._refraction_texture = None
