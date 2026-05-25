"""
Fluid Volume Component.

Provides fluid simulation component for water volumes, buoyancy,
fluid flow, and interaction with physics objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from ..character.character_controller import Vector3


class FluidType(str, Enum):
    """Type of fluid."""
    WATER = "water"
    LAVA = "lava"
    OIL = "oil"
    MUD = "mud"
    ACID = "acid"
    CUSTOM = "custom"


class FlowType(str, Enum):
    """Type of fluid flow."""
    STATIC = "static"        # No flow
    DIRECTIONAL = "directional"  # Constant direction
    RADIAL = "radial"        # Flow from/to center
    VORTEX = "vortex"        # Swirling flow
    RIVER = "river"          # Following spline path


@dataclass
class FluidConfig:
    """
    Configuration for fluid volume.

    Attributes:
        fluid_type: Type of fluid
        density: Fluid density (kg/m^3)
        viscosity: Fluid viscosity
        surface_tension: Surface tension coefficient
        buoyancy_scale: Buoyancy force multiplier
        drag_coefficient: Drag force coefficient
        wave_enabled: Enable surface waves
        wave_height: Wave amplitude
        wave_frequency: Wave frequency
    """
    fluid_type: FluidType = FluidType.WATER
    density: float = 1000.0  # Water density
    viscosity: float = 0.001
    surface_tension: float = 0.073
    buoyancy_scale: float = 1.0
    drag_coefficient: float = 0.5
    wave_enabled: bool = True
    wave_height: float = 0.1
    wave_frequency: float = 1.0
    temperature: float = 20.0  # Celsius


@dataclass
class FlowConfig:
    """
    Configuration for fluid flow.

    Attributes:
        flow_type: Type of flow
        direction: Flow direction (for directional)
        speed: Flow speed
        center: Center point (for radial/vortex)
        turbulence: Turbulence factor
    """
    flow_type: FlowType = FlowType.STATIC
    direction: Vector3 = field(default_factory=lambda: Vector3(1.0, 0.0, 0.0))
    speed: float = 1.0
    center: Vector3 = field(default_factory=Vector3.zero)
    turbulence: float = 0.0


@dataclass
class SubmergedObject:
    """
    Tracking data for a submerged object.

    Attributes:
        entity_id: Entity ID
        body_id: Physics body ID
        submerged_volume: Volume currently underwater
        submerged_ratio: Ratio of volume submerged (0-1)
        entry_time: When object entered fluid
    """
    entity_id: int = 0
    body_id: int = 0
    submerged_volume: float = 0.0
    submerged_ratio: float = 0.0
    entry_time: float = 0.0
    last_position: Vector3 = field(default_factory=Vector3.zero)


class FluidVolumeComponent:
    """
    Component for fluid volumes.

    Provides:
    - Buoyancy calculation
    - Fluid drag
    - Surface wave simulation
    - Flow velocity fields
    - Object tracking
    """

    def __init__(
        self,
        entity_id: int,
        config: Optional[FluidConfig] = None,
        flow_config: Optional[FlowConfig] = None,
    ):
        self._entity_id = entity_id
        self._config = config or FluidConfig()
        self._flow_config = flow_config or FlowConfig()

        # Volume bounds
        self._bounds_min = Vector3.zero()
        self._bounds_max = Vector3.one()
        self._surface_height = 1.0

        # Submerged objects
        self._submerged_objects: dict[int, SubmergedObject] = {}

        # Wave state
        self._wave_time = 0.0
        self._wave_sources: list[tuple[Vector3, float, float]] = []  # (pos, amplitude, time)

        # State
        self._enabled = True
        self._trigger_id: Optional[int] = None

        # Callbacks
        self._on_enter: Optional[Callable[[int], None]] = None
        self._on_exit: Optional[Callable[[int], None]] = None
        self._on_submerge: Optional[Callable[[int], None]] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def entity_id(self) -> int:
        """Entity this component belongs to."""
        return self._entity_id

    @property
    def config(self) -> FluidConfig:
        """Fluid configuration."""
        return self._config

    @property
    def flow_config(self) -> FlowConfig:
        """Flow configuration."""
        return self._flow_config

    @property
    def surface_height(self) -> float:
        """Current surface height."""
        return self._surface_height

    @property
    def submerged_count(self) -> int:
        """Number of submerged objects."""
        return len(self._submerged_objects)

    @property
    def enabled(self) -> bool:
        """Whether fluid simulation is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_enter_callback(self, callback: Optional[Callable[[int], None]]) -> None:
        """Set callback for objects entering fluid."""
        self._on_enter = callback

    def set_exit_callback(self, callback: Optional[Callable[[int], None]]) -> None:
        """Set callback for objects exiting fluid."""
        self._on_exit = callback

    def set_submerge_callback(self, callback: Optional[Callable[[int], None]]) -> None:
        """Set callback for fully submerged objects."""
        self._on_submerge = callback

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def set_bounds(self, min_point: Vector3, max_point: Vector3) -> None:
        """Set the volume bounds."""
        self._bounds_min = min_point
        self._bounds_max = max_point
        self._surface_height = max_point.y

    def set_surface_height(self, height: float) -> None:
        """Set the water surface height."""
        self._surface_height = height
        self._bounds_max = Vector3(
            self._bounds_max.x, height, self._bounds_max.z
        )

    def set_flow(
        self,
        flow_type: FlowType,
        direction: Optional[Vector3] = None,
        speed: float = 1.0,
        center: Optional[Vector3] = None,
    ) -> None:
        """Configure fluid flow."""
        self._flow_config.flow_type = flow_type
        self._flow_config.speed = speed

        if direction is not None:
            self._flow_config.direction = direction.normalized()
        if center is not None:
            self._flow_config.center = center

    # -------------------------------------------------------------------------
    # Buoyancy
    # -------------------------------------------------------------------------

    def calculate_buoyancy(
        self,
        position: Vector3,
        volume: float,
        object_density: float,
    ) -> Vector3:
        """
        Calculate buoyancy force for an object.

        Args:
            position: Object center position
            volume: Object volume
            object_density: Object density

        Returns:
            Buoyancy force vector
        """
        if not self.is_point_in_volume(position):
            return Vector3.zero()

        # Calculate submerged volume
        depth = self._surface_height - position.y
        if depth <= 0:
            return Vector3.zero()

        submerged_ratio = min(1.0, depth / 1.0)  # Simplified
        submerged_volume = volume * submerged_ratio

        # Archimedes principle: F = rho * g * V
        gravity = 9.81
        buoyancy_force = (
            self._config.density *
            gravity *
            submerged_volume *
            self._config.buoyancy_scale
        )

        return Vector3(0.0, buoyancy_force, 0.0)

    def calculate_drag(
        self,
        velocity: Vector3,
        cross_section: float,
    ) -> Vector3:
        """
        Calculate drag force.

        Args:
            velocity: Object velocity
            cross_section: Cross-sectional area

        Returns:
            Drag force vector (opposes motion)
        """
        speed = velocity.magnitude()
        if speed < 0.001:
            return Vector3.zero()

        # Drag equation: F = 0.5 * rho * v^2 * Cd * A
        drag_mag = (
            0.5 *
            self._config.density *
            speed * speed *
            self._config.drag_coefficient *
            cross_section
        )

        return velocity.normalized() * (-drag_mag)

    # -------------------------------------------------------------------------
    # Flow
    # -------------------------------------------------------------------------

    def get_flow_velocity(self, position: Vector3) -> Vector3:
        """
        Get flow velocity at a position.

        Args:
            position: World position

        Returns:
            Flow velocity vector
        """
        if not self.is_point_in_volume(position):
            return Vector3.zero()

        flow = self._flow_config
        base_velocity = Vector3.zero()

        if flow.flow_type == FlowType.STATIC:
            pass

        elif flow.flow_type == FlowType.DIRECTIONAL:
            base_velocity = flow.direction * flow.speed

        elif flow.flow_type == FlowType.RADIAL:
            to_center = flow.center - position
            distance = to_center.magnitude()
            if distance > 0.001:
                direction = to_center.normalized()
                # Velocity decreases with distance (simplified)
                base_velocity = direction * flow.speed / max(1.0, distance)

        elif flow.flow_type == FlowType.VORTEX:
            to_center = position - flow.center
            # Tangent to radius (XZ plane)
            tangent = Vector3(-to_center.z, 0, to_center.x).normalized()
            distance = to_center.horizontal().magnitude()
            if distance > 0.001:
                base_velocity = tangent * flow.speed / max(1.0, distance)

        # Add turbulence
        if flow.turbulence > 0:
            import math
            t = self._wave_time
            turb = Vector3(
                math.sin(t * 3.7 + position.x) * flow.turbulence,
                0,
                math.cos(t * 2.3 + position.z) * flow.turbulence,
            )
            base_velocity = base_velocity + turb

        return base_velocity

    # -------------------------------------------------------------------------
    # Waves
    # -------------------------------------------------------------------------

    def get_surface_height_at(self, x: float, z: float) -> float:
        """
        Get water surface height at XZ position.

        Args:
            x: X coordinate
            z: Z coordinate

        Returns:
            Surface height
        """
        if not self._config.wave_enabled:
            return self._surface_height

        import math

        height = self._surface_height

        # Base wave
        wave_offset = math.sin(
            self._wave_time * self._config.wave_frequency * 2 * math.pi +
            x * 0.5 + z * 0.3
        ) * self._config.wave_height

        height += wave_offset

        # Additional wave sources (splashes)
        for source_pos, amplitude, source_time in self._wave_sources:
            age = self._wave_time - source_time
            if age > 5.0:  # Decay after 5 seconds
                continue

            distance = math.sqrt((x - source_pos.x) ** 2 + (z - source_pos.z) ** 2)
            wave_speed = 2.0  # m/s

            # Ring wave
            ring_distance = age * wave_speed
            ring_width = 1.0

            if abs(distance - ring_distance) < ring_width:
                decay = math.exp(-age * 0.5)
                ring_amplitude = amplitude * decay * (
                    1.0 - abs(distance - ring_distance) / ring_width
                )
                height += ring_amplitude

        return height

    def add_splash(
        self,
        position: Vector3,
        amplitude: float,
    ) -> None:
        """
        Add a splash/wave source.

        Args:
            position: Splash position
            amplitude: Wave amplitude
        """
        self._wave_sources.append((position, amplitude, self._wave_time))

        # Limit wave sources
        if len(self._wave_sources) > 10:
            self._wave_sources.pop(0)

    def update_waves(self, dt: float) -> None:
        """Update wave simulation."""
        self._wave_time += dt

        # Remove old wave sources
        self._wave_sources = [
            (pos, amp, time)
            for pos, amp, time in self._wave_sources
            if self._wave_time - time < 5.0
        ]

    # -------------------------------------------------------------------------
    # Object Tracking
    # -------------------------------------------------------------------------

    def on_object_enter(
        self,
        entity_id: int,
        body_id: int,
        position: Vector3,
        current_time: float,
    ) -> None:
        """Handle object entering fluid."""
        if entity_id not in self._submerged_objects:
            self._submerged_objects[entity_id] = SubmergedObject(
                entity_id=entity_id,
                body_id=body_id,
                entry_time=current_time,
                last_position=position,
            )

            # Add splash
            entry_speed = 0.0  # Would calculate from velocity
            self.add_splash(position, min(0.5, entry_speed * 0.1))

            if self._on_enter:
                self._on_enter(entity_id)

    def on_object_exit(self, entity_id: int) -> None:
        """Handle object exiting fluid."""
        if entity_id in self._submerged_objects:
            del self._submerged_objects[entity_id]

            if self._on_exit:
                self._on_exit(entity_id)

    def update_submerged_object(
        self,
        entity_id: int,
        position: Vector3,
        volume: float,
    ) -> None:
        """Update tracking for a submerged object."""
        if entity_id not in self._submerged_objects:
            return

        obj = self._submerged_objects[entity_id]

        # Calculate submerged ratio
        surface = self.get_surface_height_at(position.x, position.z)
        depth = surface - position.y

        if depth <= 0:
            obj.submerged_ratio = 0.0
            obj.submerged_volume = 0.0
        else:
            # Simplified - assumes spherical object
            obj.submerged_ratio = min(1.0, depth / 2.0)
            obj.submerged_volume = volume * obj.submerged_ratio

        # Check for full submersion
        if obj.submerged_ratio >= 1.0 and self._on_submerge:
            self._on_submerge(entity_id)

        obj.last_position = position

    def is_object_submerged(self, entity_id: int) -> bool:
        """Check if an object is in the fluid."""
        return entity_id in self._submerged_objects

    def get_submerged_ratio(self, entity_id: int) -> float:
        """Get how much of an object is submerged."""
        if entity_id in self._submerged_objects:
            return self._submerged_objects[entity_id].submerged_ratio
        return 0.0

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def is_point_in_volume(self, point: Vector3) -> bool:
        """Check if a point is inside the fluid volume."""
        return (
            self._bounds_min.x <= point.x <= self._bounds_max.x and
            self._bounds_min.y <= point.y <= self._bounds_max.y and
            self._bounds_min.z <= point.z <= self._bounds_max.z
        )

    def is_point_underwater(self, point: Vector3) -> bool:
        """Check if a point is below the water surface."""
        if not self.is_point_in_volume(point):
            return False
        surface = self.get_surface_height_at(point.x, point.z)
        return point.y < surface

    def get_depth_at_point(self, point: Vector3) -> float:
        """Get water depth at a point (negative if above surface)."""
        surface = self.get_surface_height_at(point.x, point.z)
        return surface - point.y

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self, trigger_id: int) -> None:
        """Initialize with physics trigger ID."""
        self._trigger_id = trigger_id

    def cleanup(self) -> None:
        """Cleanup component."""
        self._trigger_id = None
        self._submerged_objects.clear()
        self._wave_sources.clear()

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Get serializable state."""
        return {
            "entity_id": self._entity_id,
            "fluid_type": self._config.fluid_type.value,
            "density": self._config.density,
            "surface_height": self._surface_height,
            "flow_type": self._flow_config.flow_type.value,
            "flow_speed": self._flow_config.speed,
            "submerged_count": len(self._submerged_objects),
            "wave_time": self._wave_time,
            "enabled": self._enabled,
        }


__all__ = [
    "FluidType",
    "FlowType",
    "FluidConfig",
    "FlowConfig",
    "SubmergedObject",
    "FluidVolumeComponent",
]
