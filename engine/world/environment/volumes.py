"""
Environment Volumes System - Physics, trigger, visual, and audio volumes.

Provides a comprehensive volume system for the game engine world layer,
supporting various volume types for physics, gameplay, visual effects,
and audio processing. Uses the Trinity Pattern with enter/exit hooks
and ObservableDescriptor for state changes.
"""

from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

from engine.world.environment.constants import (
    DEFAULT_VOLUME_BLEND_RADIUS,
    DEFAULT_VOLUME_PRIORITY,
    DEFAULT_GRAVITY_STRENGTH,
    DEFAULT_GRAVITY_DIRECTION,
    DEFAULT_WATER_BUOYANCY,
    DEFAULT_WATER_DRAG,
    DEFAULT_PAIN_DAMAGE_PER_SECOND,
    DEFAULT_FOG_DENSITY,
    DEFAULT_FOG_HEIGHT_FALLOFF,
)

T = TypeVar("T")

# Type aliases
Point3D = Tuple[float, float, float]
Observer = Callable[[Any, str, Any, Any], None]


class VolumeType(Enum):
    """Types of environment volumes."""
    # Physics volumes
    GRAVITY = auto()
    WATER = auto()
    PAIN = auto()
    KILL = auto()
    # Gameplay volumes
    TRIGGER = auto()
    BLOCKING = auto()
    CAMERA = auto()
    SPAWN = auto()
    # Visual volumes
    FOG = auto()
    POST_PROCESS = auto()
    REFLECTION = auto()
    LIGHTMASS = auto()
    # Audio volumes
    REVERB = auto()
    AMBIENT = auto()
    # Navigation volumes
    NAV_MODIFIER = auto()
    NAV_LINK = auto()
    NAV_EXCLUDE = auto()


class VolumeShape(Enum):
    """Shape types for volumes."""
    BOX = auto()
    SPHERE = auto()
    CAPSULE = auto()
    CONVEX = auto()


@dataclass
class BoundingBox:
    """Axis-aligned bounding box for volume bounds."""
    min_point: Point3D = (0.0, 0.0, 0.0)
    max_point: Point3D = (1.0, 1.0, 1.0)

    @property
    def center(self) -> Point3D:
        """Get center point of the bounding box."""
        return (
            (self.min_point[0] + self.max_point[0]) / 2,
            (self.min_point[1] + self.max_point[1]) / 2,
            (self.min_point[2] + self.max_point[2]) / 2,
        )

    @property
    def extents(self) -> Point3D:
        """Get half-extents of the bounding box."""
        return (
            (self.max_point[0] - self.min_point[0]) / 2,
            (self.max_point[1] - self.min_point[1]) / 2,
            (self.max_point[2] - self.min_point[2]) / 2,
        )

    def contains_point(self, point: Point3D) -> bool:
        """Check if a point is inside the bounding box."""
        return (
            self.min_point[0] <= point[0] <= self.max_point[0] and
            self.min_point[1] <= point[1] <= self.max_point[1] and
            self.min_point[2] <= point[2] <= self.max_point[2]
        )

    def distance_to_edge(self, point: Point3D) -> float:
        """
        Calculate distance from point to nearest edge.

        Returns positive value if inside, negative if outside.
        """
        # Calculate distance to each face
        dx_min = point[0] - self.min_point[0]
        dx_max = self.max_point[0] - point[0]
        dy_min = point[1] - self.min_point[1]
        dy_max = self.max_point[1] - point[1]
        dz_min = point[2] - self.min_point[2]
        dz_max = self.max_point[2] - point[2]

        # If inside, return distance to nearest face (positive)
        min_dist = min(dx_min, dx_max, dy_min, dy_max, dz_min, dz_max)

        if min_dist >= 0:
            return min_dist

        # If outside, calculate signed distance (negative)
        return min_dist


@dataclass
class SphereBounds:
    """Spherical bounds for volumes."""
    center: Point3D = (0.0, 0.0, 0.0)
    radius: float = 1.0

    def contains_point(self, point: Point3D) -> bool:
        """Check if a point is inside the sphere."""
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        dz = point[2] - self.center[2]
        return (dx * dx + dy * dy + dz * dz) <= (self.radius * self.radius)

    def distance_to_edge(self, point: Point3D) -> float:
        """
        Calculate distance from point to sphere surface.

        Returns positive value if inside, negative if outside.
        """
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        dz = point[2] - self.center[2]
        distance_to_center = math.sqrt(dx * dx + dy * dy + dz * dz)
        return self.radius - distance_to_center


@dataclass
class CapsuleBounds:
    """Capsule bounds for volumes."""
    point_a: Point3D = (0.0, 0.0, 0.0)
    point_b: Point3D = (0.0, 1.0, 0.0)
    radius: float = 0.5

    def contains_point(self, point: Point3D) -> bool:
        """Check if a point is inside the capsule."""
        return self.distance_to_edge(point) >= 0

    def distance_to_edge(self, point: Point3D) -> float:
        """Calculate distance from point to capsule surface."""
        # Vector from A to B
        ab = (
            self.point_b[0] - self.point_a[0],
            self.point_b[1] - self.point_a[1],
            self.point_b[2] - self.point_a[2],
        )
        # Vector from A to point
        ap = (
            point[0] - self.point_a[0],
            point[1] - self.point_a[1],
            point[2] - self.point_a[2],
        )

        # Project point onto line segment AB
        ab_length_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]
        if ab_length_sq < 1e-10:
            # Degenerate case: A and B are the same point
            closest = self.point_a
        else:
            t = max(0, min(1, (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab_length_sq))
            closest = (
                self.point_a[0] + t * ab[0],
                self.point_a[1] + t * ab[1],
                self.point_a[2] + t * ab[2],
            )

        # Distance from point to closest point on line segment
        dx = point[0] - closest[0]
        dy = point[1] - closest[1]
        dz = point[2] - closest[2]
        distance_to_axis = math.sqrt(dx * dx + dy * dy + dz * dz)

        return self.radius - distance_to_axis


class BaseVolume(ABC):
    """
    Base class for all volume types.

    Provides common functionality for containment testing, blending,
    and priority-based overlap resolution.
    """

    def __init__(
        self,
        volume_id: Optional[str] = None,
        volume_type: VolumeType = VolumeType.TRIGGER,
        shape: VolumeShape = VolumeShape.BOX,
        bounds: Optional[BoundingBox | SphereBounds | CapsuleBounds] = None,
        priority: int = 0,
        blend_radius: float = 0.0,
        is_active: bool = True,
    ) -> None:
        """
        Initialize a volume.

        Args:
            volume_id: Unique identifier for the volume.
            volume_type: Type of the volume.
            shape: Shape of the volume.
            bounds: Bounds for containment testing.
            priority: Priority for overlap resolution (higher = more important).
            blend_radius: Radius for smooth blending at edges.
            is_active: Whether the volume is currently active.
        """
        self.volume_id = volume_id or str(uuid.uuid4())
        self.volume_type = volume_type
        self.shape = shape
        self.bounds = bounds or BoundingBox()
        self.priority = priority
        self.blend_radius = blend_radius
        self.is_active = is_active
        self._observers: Dict[str, List[Observer]] = {}

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """
        Check if a point is inside this volume.

        Args:
            x: X coordinate.
            y: Y coordinate.
            z: Z coordinate.

        Returns:
            True if the point is inside the volume.
        """
        if not self.is_active:
            return False
        return self.bounds.contains_point((x, y, z))

    def get_blend_weight(self, point: Point3D) -> float:
        """
        Get blend weight for a point based on distance from edge.

        Args:
            point: The point to check.

        Returns:
            Blend weight from 0.0 (at edge) to 1.0 (fully inside).
        """
        if not self.is_active:
            return 0.0

        distance = self.bounds.distance_to_edge(point)

        if distance < 0:
            return 0.0  # Outside volume

        if self.blend_radius <= 0:
            return 1.0  # No blending, fully inside

        if distance >= self.blend_radius:
            return 1.0  # Fully inside blend region

        # Smooth blend using smoothstep
        t = distance / self.blend_radius
        return t * t * (3 - 2 * t)

    def add_observer(self, field_name: str, callback: Observer) -> None:
        """Add an observer for a field."""
        if field_name not in self._observers:
            self._observers[field_name] = []
        self._observers[field_name].append(callback)

    def remove_observer(self, field_name: str, callback: Observer) -> None:
        """Remove an observer for a field."""
        if field_name in self._observers:
            try:
                self._observers[field_name].remove(callback)
            except ValueError:
                pass

    def _notify_observers(self, field_name: str, old_value: Any, new_value: Any) -> None:
        """Notify all observers of a field change."""
        if field_name in self._observers:
            for callback in self._observers[field_name]:
                try:
                    callback(self, field_name, old_value, new_value)
                except Exception:
                    pass  # Don't let observer errors break the volume

    @abstractmethod
    def get_settings_dict(self) -> Dict[str, Any]:
        """Get volume settings as a dictionary."""
        pass


# =============================================================================
# Physics Volumes
# =============================================================================


class PhysicsVolume(BaseVolume):
    """
    Volume that modifies physics behavior.

    Affects gravity direction, strength, and fluid properties
    for objects inside the volume.
    """

    def __init__(
        self,
        gravity_direction: Point3D = (0.0, -1.0, 0.0),
        gravity_strength: float = 9.81,
        fluid_friction: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.GRAVITY, **kwargs)
        # Normalize gravity direction on construction
        length = math.sqrt(gravity_direction[0]**2 + gravity_direction[1]**2 + gravity_direction[2]**2)
        if length > 0:
            self._gravity_direction = (
                gravity_direction[0] / length,
                gravity_direction[1] / length,
                gravity_direction[2] / length,
            )
        else:
            self._gravity_direction = (0.0, -1.0, 0.0)
        self._gravity_strength = max(0.0, gravity_strength)
        self._fluid_friction = max(0.0, fluid_friction)

    @property
    def gravity_direction(self) -> Point3D:
        return self._gravity_direction

    @gravity_direction.setter
    def gravity_direction(self, value: Point3D) -> None:
        old_value = self._gravity_direction
        # Normalize the direction
        length = math.sqrt(value[0]**2 + value[1]**2 + value[2]**2)
        if length > 0:
            self._gravity_direction = (value[0]/length, value[1]/length, value[2]/length)
        else:
            self._gravity_direction = (0.0, -1.0, 0.0)
        self._notify_observers("gravity_direction", old_value, self._gravity_direction)

    @property
    def gravity_strength(self) -> float:
        return self._gravity_strength

    @gravity_strength.setter
    def gravity_strength(self, value: float) -> None:
        old_value = self._gravity_strength
        self._gravity_strength = max(0.0, value)
        self._notify_observers("gravity_strength", old_value, self._gravity_strength)

    @property
    def fluid_friction(self) -> float:
        return self._fluid_friction

    @fluid_friction.setter
    def fluid_friction(self, value: float) -> None:
        old_value = self._fluid_friction
        self._fluid_friction = max(0.0, value)
        self._notify_observers("fluid_friction", old_value, self._fluid_friction)

    def get_gravity_vector(self) -> Point3D:
        """Get the gravity vector (direction * strength)."""
        return (
            self._gravity_direction[0] * self._gravity_strength,
            self._gravity_direction[1] * self._gravity_strength,
            self._gravity_direction[2] * self._gravity_strength,
        )

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "gravity_direction": self._gravity_direction,
            "gravity_strength": self._gravity_strength,
            "fluid_friction": self._fluid_friction,
        }


class WaterVolume(BaseVolume):
    """
    Volume representing a body of water.

    Provides buoyancy, drag, and current simulation for objects.
    """

    def __init__(
        self,
        water_height: float = 0.0,
        buoyancy: float = 1.0,
        drag: float = 0.5,
        current_direction: Point3D = (0.0, 0.0, 0.0),
        current_strength: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.WATER, **kwargs)
        self._water_height = water_height
        self._buoyancy = max(0.0, buoyancy)
        self._drag = max(0.0, min(1.0, drag))  # Clamp on construction
        self._current_direction = current_direction
        self._current_strength = max(0.0, current_strength)

    @property
    def water_height(self) -> float:
        return self._water_height

    @water_height.setter
    def water_height(self, value: float) -> None:
        old_value = self._water_height
        self._water_height = value
        self._notify_observers("water_height", old_value, self._water_height)

    @property
    def buoyancy(self) -> float:
        return self._buoyancy

    @buoyancy.setter
    def buoyancy(self, value: float) -> None:
        old_value = self._buoyancy
        self._buoyancy = max(0.0, value)
        self._notify_observers("buoyancy", old_value, self._buoyancy)

    @property
    def drag(self) -> float:
        return self._drag

    @drag.setter
    def drag(self, value: float) -> None:
        old_value = self._drag
        self._drag = max(0.0, min(1.0, value))
        self._notify_observers("drag", old_value, self._drag)

    @property
    def current_direction(self) -> Point3D:
        return self._current_direction

    @current_direction.setter
    def current_direction(self, value: Point3D) -> None:
        old_value = self._current_direction
        self._current_direction = value
        self._notify_observers("current_direction", old_value, self._current_direction)

    @property
    def current_strength(self) -> float:
        return self._current_strength

    @current_strength.setter
    def current_strength(self, value: float) -> None:
        old_value = self._current_strength
        self._current_strength = max(0.0, value)
        self._notify_observers("current_strength", old_value, self._current_strength)

    def get_current_vector(self) -> Point3D:
        """Get the water current vector."""
        return (
            self._current_direction[0] * self._current_strength,
            self._current_direction[1] * self._current_strength,
            self._current_direction[2] * self._current_strength,
        )

    def get_submersion_depth(self, point: Point3D) -> float:
        """
        Get how deep a point is submerged below water surface.

        Returns:
            Depth below surface (positive if underwater, negative if above).
        """
        return self._water_height - point[1]

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "water_height": self._water_height,
            "buoyancy": self._buoyancy,
            "drag": self._drag,
            "current_direction": self._current_direction,
            "current_strength": self._current_strength,
        }


class PainVolume(BaseVolume):
    """Volume that deals damage to actors inside it."""

    def __init__(
        self,
        damage_per_second: float = 10.0,
        damage_type: str = "generic",
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.PAIN, **kwargs)
        self._damage_per_second = damage_per_second
        self._damage_type = damage_type

    @property
    def damage_per_second(self) -> float:
        return self._damage_per_second

    @damage_per_second.setter
    def damage_per_second(self, value: float) -> None:
        old_value = self._damage_per_second
        self._damage_per_second = max(0.0, value)
        self._notify_observers("damage_per_second", old_value, self._damage_per_second)

    @property
    def damage_type(self) -> str:
        return self._damage_type

    @damage_type.setter
    def damage_type(self, value: str) -> None:
        old_value = self._damage_type
        self._damage_type = value
        self._notify_observers("damage_type", old_value, self._damage_type)

    def calculate_damage(self, dt: float) -> float:
        """Calculate damage for a time delta."""
        return self._damage_per_second * dt

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "damage_per_second": self._damage_per_second,
            "damage_type": self._damage_type,
        }


class KillVolume(BaseVolume):
    """Volume that instantly kills actors inside it."""

    def __init__(
        self,
        kill_message: str = "Fell out of world",
        respawn_point: Optional[Point3D] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.KILL, **kwargs)
        self._kill_message = kill_message
        self._respawn_point = respawn_point

    @property
    def kill_message(self) -> str:
        return self._kill_message

    @property
    def respawn_point(self) -> Optional[Point3D]:
        return self._respawn_point

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "kill_message": self._kill_message,
            "respawn_point": self._respawn_point,
        }


# =============================================================================
# Gameplay Volumes
# =============================================================================


class OverlapState(Enum):
    """State of an actor's overlap with a trigger volume."""
    OUTSIDE = auto()
    ENTERED = auto()
    INSIDE = auto()
    EXITED = auto()


class TriggerVolume(BaseVolume):
    """
    Volume that triggers callbacks on actor overlap.

    Supports enter, exit, and continuous overlap callbacks with
    tag-based filtering.
    """

    def __init__(
        self,
        filter_tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.TRIGGER, **kwargs)
        self.filter_tags: List[str] = filter_tags or []
        self.on_enter_callbacks: List[Callable[[Any], None]] = []
        self.on_exit_callbacks: List[Callable[[Any], None]] = []
        self.on_overlap_callbacks: List[Callable[[Any], None]] = []
        self.overlapping_actors: Set[str] = set()

    def check_overlap(self, actor_id: str, position: Point3D, tags: Optional[List[str]] = None) -> OverlapState:
        """
        Check overlap state for an actor.

        Args:
            actor_id: Unique identifier for the actor.
            position: Current position of the actor.
            tags: Tags associated with the actor.

        Returns:
            The overlap state (ENTERED, EXITED, INSIDE, or OUTSIDE).
        """
        if not self.is_active:
            return OverlapState.OUTSIDE

        # Check tag filter
        if self.filter_tags and tags:
            if not any(tag in self.filter_tags for tag in tags):
                return OverlapState.OUTSIDE
        elif self.filter_tags and not tags:
            return OverlapState.OUTSIDE

        is_inside = self.contains_point(position[0], position[1], position[2])
        was_inside = actor_id in self.overlapping_actors

        if is_inside and not was_inside:
            self.overlapping_actors.add(actor_id)
            return OverlapState.ENTERED
        elif not is_inside and was_inside:
            self.overlapping_actors.discard(actor_id)
            return OverlapState.EXITED
        elif is_inside:
            return OverlapState.INSIDE
        else:
            return OverlapState.OUTSIDE

    def add_on_enter(self, callback: Callable[[Any], None]) -> None:
        """Add an enter callback."""
        self.on_enter_callbacks.append(callback)

    def add_on_exit(self, callback: Callable[[Any], None]) -> None:
        """Add an exit callback."""
        self.on_exit_callbacks.append(callback)

    def add_on_overlap(self, callback: Callable[[Any], None]) -> None:
        """Add an overlap callback."""
        self.on_overlap_callbacks.append(callback)

    def trigger_callbacks(self, state: OverlapState, actor: Any) -> None:
        """Trigger appropriate callbacks based on overlap state."""
        if state == OverlapState.ENTERED:
            for callback in self.on_enter_callbacks:
                try:
                    callback(actor)
                except Exception:
                    pass
        elif state == OverlapState.EXITED:
            for callback in self.on_exit_callbacks:
                try:
                    callback(actor)
                except Exception:
                    pass
        elif state == OverlapState.INSIDE:
            for callback in self.on_overlap_callbacks:
                try:
                    callback(actor)
                except Exception:
                    pass

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "filter_tags": self.filter_tags,
            "overlapping_count": len(self.overlapping_actors),
        }


class BlockingVolume(BaseVolume):
    """Volume that blocks actor movement."""

    def __init__(
        self,
        collision_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.BLOCKING, **kwargs)
        self.collision_enabled = collision_enabled

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "collision_enabled": self.collision_enabled,
        }


class CameraVolume(BaseVolume):
    """Volume that modifies camera behavior."""

    def __init__(
        self,
        camera_mode: str = "default",
        blend_time: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.CAMERA, **kwargs)
        self.camera_mode = camera_mode
        self.blend_time = blend_time

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "camera_mode": self.camera_mode,
            "blend_time": self.blend_time,
        }


class SpawnVolume(BaseVolume):
    """Volume that defines spawn areas."""

    def __init__(
        self,
        spawn_team: Optional[str] = None,
        max_occupants: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.SPAWN, **kwargs)
        self.spawn_team = spawn_team
        self.max_occupants = max_occupants
        self.current_occupants: Set[str] = set()

    def get_random_point(self) -> Point3D:
        """Get a random spawn point within the volume."""
        import random

        if isinstance(self.bounds, BoundingBox):
            return (
                random.uniform(self.bounds.min_point[0], self.bounds.max_point[0]),
                random.uniform(self.bounds.min_point[1], self.bounds.max_point[1]),
                random.uniform(self.bounds.min_point[2], self.bounds.max_point[2]),
            )
        elif isinstance(self.bounds, SphereBounds):
            # Random point in sphere
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            r = self.bounds.radius * (random.random() ** (1/3))
            return (
                self.bounds.center[0] + r * math.sin(phi) * math.cos(theta),
                self.bounds.center[1] + r * math.sin(phi) * math.sin(theta),
                self.bounds.center[2] + r * math.cos(phi),
            )
        else:
            # Fallback to center
            if hasattr(self.bounds, 'center'):
                return self.bounds.center
            return (0.0, 0.0, 0.0)

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "spawn_team": self.spawn_team,
            "max_occupants": self.max_occupants,
            "current_occupants": len(self.current_occupants),
        }


# =============================================================================
# Visual Volumes
# =============================================================================


class PostProcessVolume(BaseVolume):
    """
    Volume that applies post-processing effects.

    Supports exposure, saturation, contrast, bloom, color grading,
    and vignette effects.
    """

    def __init__(
        self,
        exposure: float = 1.0,
        saturation: float = 1.0,
        contrast: float = 1.0,
        bloom_intensity: float = 0.0,
        color_grading_lut: Optional[str] = None,
        vignette_intensity: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.POST_PROCESS, **kwargs)
        self._exposure = exposure
        self._saturation = saturation
        self._contrast = contrast
        self._bloom_intensity = bloom_intensity
        self._color_grading_lut = color_grading_lut
        self._vignette_intensity = vignette_intensity

    @property
    def exposure(self) -> float:
        return self._exposure

    @exposure.setter
    def exposure(self, value: float) -> None:
        old_value = self._exposure
        self._exposure = max(0.01, value)
        self._notify_observers("exposure", old_value, self._exposure)

    @property
    def saturation(self) -> float:
        return self._saturation

    @saturation.setter
    def saturation(self, value: float) -> None:
        old_value = self._saturation
        self._saturation = max(0.0, value)
        self._notify_observers("saturation", old_value, self._saturation)

    @property
    def contrast(self) -> float:
        return self._contrast

    @contrast.setter
    def contrast(self, value: float) -> None:
        old_value = self._contrast
        self._contrast = max(0.0, value)
        self._notify_observers("contrast", old_value, self._contrast)

    @property
    def bloom_intensity(self) -> float:
        return self._bloom_intensity

    @bloom_intensity.setter
    def bloom_intensity(self, value: float) -> None:
        old_value = self._bloom_intensity
        self._bloom_intensity = max(0.0, value)
        self._notify_observers("bloom_intensity", old_value, self._bloom_intensity)

    @property
    def color_grading_lut(self) -> Optional[str]:
        return self._color_grading_lut

    @color_grading_lut.setter
    def color_grading_lut(self, value: Optional[str]) -> None:
        old_value = self._color_grading_lut
        self._color_grading_lut = value
        self._notify_observers("color_grading_lut", old_value, self._color_grading_lut)

    @property
    def vignette_intensity(self) -> float:
        return self._vignette_intensity

    @vignette_intensity.setter
    def vignette_intensity(self, value: float) -> None:
        old_value = self._vignette_intensity
        self._vignette_intensity = max(0.0, min(1.0, value))
        self._notify_observers("vignette_intensity", old_value, self._vignette_intensity)

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "exposure": self._exposure,
            "saturation": self._saturation,
            "contrast": self._contrast,
            "bloom_intensity": self._bloom_intensity,
            "color_grading_lut": self._color_grading_lut,
            "vignette_intensity": self._vignette_intensity,
        }


class FogVolume(BaseVolume):
    """Volume that defines local fog settings."""

    def __init__(
        self,
        fog_density: float = 0.02,
        fog_color: Tuple[float, float, float] = (0.5, 0.6, 0.7),
        fog_height_falloff: float = 0.2,
        inscattering_color: Tuple[float, float, float] = (0.8, 0.9, 1.0),
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.FOG, **kwargs)
        self._fog_density = fog_density
        self._fog_color = fog_color
        self._fog_height_falloff = fog_height_falloff
        self._inscattering_color = inscattering_color

    @property
    def fog_density(self) -> float:
        return self._fog_density

    @fog_density.setter
    def fog_density(self, value: float) -> None:
        old_value = self._fog_density
        self._fog_density = max(0.0, value)
        self._notify_observers("fog_density", old_value, self._fog_density)

    @property
    def fog_color(self) -> Tuple[float, float, float]:
        return self._fog_color

    @fog_color.setter
    def fog_color(self, value: Tuple[float, float, float]) -> None:
        old_value = self._fog_color
        self._fog_color = tuple(max(0.0, min(1.0, c)) for c in value)  # type: ignore
        self._notify_observers("fog_color", old_value, self._fog_color)

    @property
    def fog_height_falloff(self) -> float:
        return self._fog_height_falloff

    @fog_height_falloff.setter
    def fog_height_falloff(self, value: float) -> None:
        old_value = self._fog_height_falloff
        self._fog_height_falloff = max(0.0, value)
        self._notify_observers("fog_height_falloff", old_value, self._fog_height_falloff)

    @property
    def inscattering_color(self) -> Tuple[float, float, float]:
        return self._inscattering_color

    @inscattering_color.setter
    def inscattering_color(self, value: Tuple[float, float, float]) -> None:
        old_value = self._inscattering_color
        self._inscattering_color = tuple(max(0.0, min(1.0, c)) for c in value)  # type: ignore
        self._notify_observers("inscattering_color", old_value, self._inscattering_color)

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "fog_density": self._fog_density,
            "fog_color": self._fog_color,
            "fog_height_falloff": self._fog_height_falloff,
            "inscattering_color": self._inscattering_color,
        }


class ReflectionVolume(BaseVolume):
    """Volume that captures environment reflections."""

    def __init__(
        self,
        capture_offset: Point3D = (0.0, 0.0, 0.0),
        influence_radius: float = 100.0,
        brightness: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.REFLECTION, **kwargs)
        self.capture_offset = capture_offset
        self.influence_radius = influence_radius
        self.brightness = brightness

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "capture_offset": self.capture_offset,
            "influence_radius": self.influence_radius,
            "brightness": self.brightness,
        }


class LightmassVolume(BaseVolume):
    """Volume that controls lightmass importance."""

    def __init__(
        self,
        importance_multiplier: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.LIGHTMASS, **kwargs)
        self.importance_multiplier = importance_multiplier

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "importance_multiplier": self.importance_multiplier,
        }


# =============================================================================
# Audio Volumes
# =============================================================================


class ReverbVolume(BaseVolume):
    """Volume that applies reverb effects."""

    def __init__(
        self,
        reverb_preset: str = "default",
        room_size: float = 0.5,
        decay_time: float = 1.0,
        wet_dry_mix: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.REVERB, **kwargs)
        self._reverb_preset = reverb_preset
        self._room_size = room_size
        self._decay_time = decay_time
        self._wet_dry_mix = wet_dry_mix

    @property
    def reverb_preset(self) -> str:
        return self._reverb_preset

    @reverb_preset.setter
    def reverb_preset(self, value: str) -> None:
        old_value = self._reverb_preset
        self._reverb_preset = value
        self._notify_observers("reverb_preset", old_value, self._reverb_preset)

    @property
    def room_size(self) -> float:
        return self._room_size

    @room_size.setter
    def room_size(self, value: float) -> None:
        old_value = self._room_size
        self._room_size = max(0.0, min(1.0, value))
        self._notify_observers("room_size", old_value, self._room_size)

    @property
    def decay_time(self) -> float:
        return self._decay_time

    @decay_time.setter
    def decay_time(self, value: float) -> None:
        old_value = self._decay_time
        self._decay_time = max(0.0, value)
        self._notify_observers("decay_time", old_value, self._decay_time)

    @property
    def wet_dry_mix(self) -> float:
        return self._wet_dry_mix

    @wet_dry_mix.setter
    def wet_dry_mix(self, value: float) -> None:
        old_value = self._wet_dry_mix
        self._wet_dry_mix = max(0.0, min(1.0, value))
        self._notify_observers("wet_dry_mix", old_value, self._wet_dry_mix)

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "reverb_preset": self._reverb_preset,
            "room_size": self._room_size,
            "decay_time": self._decay_time,
            "wet_dry_mix": self._wet_dry_mix,
        }


class AmbientVolume(BaseVolume):
    """Volume that plays ambient sounds."""

    def __init__(
        self,
        ambient_sound_id: str = "",
        volume_multiplier: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.AMBIENT, **kwargs)
        self.ambient_sound_id = ambient_sound_id
        self.volume_multiplier = volume_multiplier

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "ambient_sound_id": self.ambient_sound_id,
            "volume_multiplier": self.volume_multiplier,
        }


# =============================================================================
# Navigation Volumes
# =============================================================================


class NavModifierVolume(BaseVolume):
    """Volume that modifies navigation mesh behavior."""

    def __init__(
        self,
        area_class: str = "default",
        cost_modifier: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.NAV_MODIFIER, **kwargs)
        self.area_class = area_class
        self.cost_modifier = cost_modifier

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "area_class": self.area_class,
            "cost_modifier": self.cost_modifier,
        }


class NavLinkVolume(BaseVolume):
    """Volume that defines navigation links (jumps, ladders, etc.)."""

    def __init__(
        self,
        link_type: str = "jump",
        target_point: Point3D = (0.0, 0.0, 0.0),
        bidirectional: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(volume_type=VolumeType.NAV_LINK, **kwargs)
        self.link_type = link_type
        self.target_point = target_point
        self.bidirectional = bidirectional

    def get_settings_dict(self) -> Dict[str, Any]:
        return {
            "link_type": self.link_type,
            "target_point": self.target_point,
            "bidirectional": self.bidirectional,
        }


class NavExcludeVolume(BaseVolume):
    """Volume that excludes area from navigation mesh."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(volume_type=VolumeType.NAV_EXCLUDE, **kwargs)

    def get_settings_dict(self) -> Dict[str, Any]:
        return {}


# =============================================================================
# Volume Manager
# =============================================================================


class VolumeManager:
    """
    Manages all volumes in the world.

    Provides efficient lookup by type, position-based queries,
    and blended settings calculation for overlapping volumes.
    """

    def __init__(self) -> None:
        self.volumes_by_type: Dict[VolumeType, List[BaseVolume]] = {
            vol_type: [] for vol_type in VolumeType
        }
        self._all_volumes: Dict[str, BaseVolume] = {}

    def add_volume(self, volume: BaseVolume) -> None:
        """Add a volume to the manager."""
        self.volumes_by_type[volume.volume_type].append(volume)
        self._all_volumes[volume.volume_id] = volume

    def remove_volume(self, volume_id: str) -> Optional[BaseVolume]:
        """Remove a volume from the manager."""
        volume = self._all_volumes.pop(volume_id, None)
        if volume:
            self.volumes_by_type[volume.volume_type].remove(volume)
        return volume

    def get_volume(self, volume_id: str) -> Optional[BaseVolume]:
        """Get a volume by ID."""
        return self._all_volumes.get(volume_id)

    def get_volumes_at_point(
        self,
        point: Point3D,
        volume_type: Optional[VolumeType] = None,
    ) -> List[BaseVolume]:
        """
        Get all volumes containing a point, sorted by priority.

        Args:
            point: The point to check.
            volume_type: Optional filter by volume type.

        Returns:
            List of volumes containing the point, sorted by priority (highest first).
        """
        result: List[BaseVolume] = []

        if volume_type:
            volumes_to_check = self.volumes_by_type[volume_type]
        else:
            volumes_to_check = list(self._all_volumes.values())

        for volume in volumes_to_check:
            if volume.contains_point(point[0], point[1], point[2]):
                result.append(volume)

        # Sort by priority (highest first)
        result.sort(key=lambda v: v.priority, reverse=True)
        return result

    def get_blended_settings(
        self,
        point: Point3D,
        volume_type: VolumeType,
    ) -> Dict[str, Any]:
        """
        Get blended settings from all volumes of a type at a point.

        Args:
            point: The point to check.
            volume_type: Type of volumes to blend.

        Returns:
            Dictionary of blended settings.
        """
        volumes = self.get_volumes_at_point(point, volume_type)

        if not volumes:
            return {}

        if len(volumes) == 1:
            return volumes[0].get_settings_dict()

        # Calculate blend weights
        weights: List[float] = []
        total_weight = 0.0

        for volume in volumes:
            weight = volume.get_blend_weight(point) * (volume.priority + 1)
            weights.append(weight)
            total_weight += weight

        if total_weight <= 0:
            return volumes[0].get_settings_dict()

        # Normalize weights
        weights = [w / total_weight for w in weights]

        # Blend settings
        result: Dict[str, Any] = {}

        for i, volume in enumerate(volumes):
            settings = volume.get_settings_dict()
            weight = weights[i]

            for key, value in settings.items():
                if key not in result:
                    result[key] = self._init_blend_value(value)
                result[key] = self._blend_value(result[key], value, weight)

        return result

    def _init_blend_value(self, value: Any) -> Any:
        """Initialize a value for blending."""
        if isinstance(value, (int, float)):
            return 0.0
        elif isinstance(value, tuple) and all(isinstance(v, (int, float)) for v in value):
            return tuple(0.0 for _ in value)
        else:
            return value

    def _blend_value(self, current: Any, new: Any, weight: float) -> Any:
        """Blend a value with weight."""
        if isinstance(new, (int, float)) and isinstance(current, (int, float)):
            return current + new * weight
        elif isinstance(new, tuple) and isinstance(current, tuple):
            if all(isinstance(v, (int, float)) for v in new):
                return tuple(c + n * weight for c, n in zip(current, new))

        # For non-blendable values, take highest weighted value
        return new if weight > 0.5 else current

    def get_all_volumes(self) -> List[BaseVolume]:
        """Get all volumes."""
        return list(self._all_volumes.values())

    def clear(self) -> None:
        """Remove all volumes."""
        self._all_volumes.clear()
        for vol_type in VolumeType:
            self.volumes_by_type[vol_type].clear()


__all__ = [
    # Enums
    "VolumeType",
    "VolumeShape",
    "OverlapState",
    # Bounds
    "BoundingBox",
    "SphereBounds",
    "CapsuleBounds",
    # Base
    "BaseVolume",
    # Physics volumes
    "PhysicsVolume",
    "WaterVolume",
    "PainVolume",
    "KillVolume",
    # Gameplay volumes
    "TriggerVolume",
    "BlockingVolume",
    "CameraVolume",
    "SpawnVolume",
    # Visual volumes
    "PostProcessVolume",
    "FogVolume",
    "ReflectionVolume",
    "LightmassVolume",
    # Audio volumes
    "ReverbVolume",
    "AmbientVolume",
    # Navigation volumes
    "NavModifierVolume",
    "NavLinkVolume",
    "NavExcludeVolume",
    # Manager
    "VolumeManager",
    # Types
    "Point3D",
    "Observer",
]
