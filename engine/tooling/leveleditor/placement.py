"""
Object Placement System - Tools for placing objects in the scene.

Provides multiple placement modes:
- Single: Place one object at a time
- Paint Brush: Paint objects with brush strokes
- Scatter: Random scatter placement within regions
- Foliage: Specialized foliage painting with density control
- Spline: Place objects along spline paths

All placement operations integrate with the Foundation Tracker for undo/redo.
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Protocol

from foundation.tracker import tracker


# =============================================================================
# Decorators
# =============================================================================

def editor(cls: type) -> type:
    """Mark a class as editor-only (not included in runtime builds)."""
    cls._editor_only = True
    cls._editor_class = True
    return cls


def track_changes(method: Callable) -> Callable:
    """Decorator to track changes for undo/redo support."""
    def wrapper(self, *args, **kwargs):
        # Check if we're already in a transaction (nested call)
        owns_transaction = tracker._txn is None
        if owns_transaction:
            transaction_name = f"{self.__class__.__name__}.{method.__name__}"
            tracker.begin_transaction(transaction_name)
        try:
            result = method(self, *args, **kwargs)
            if owns_transaction:
                tracker.commit_transaction()
            return result
        except Exception:
            if owns_transaction:
                tracker.rollback_transaction()
            raise
    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper


# =============================================================================
# Types and Enums
# =============================================================================

class PlacementMode(Enum):
    """Available placement modes."""
    SINGLE = auto()
    PAINT_BRUSH = auto()
    SCATTER = auto()
    FOLIAGE = auto()
    SPLINE = auto()


class AxisConstraint(Enum):
    """Axis constraints for placement."""
    NONE = auto()
    X = auto()
    Y = auto()
    Z = auto()
    XY = auto()
    XZ = auto()
    YZ = auto()


class SurfaceAlignment(Enum):
    """How to align objects to surfaces."""
    NONE = auto()
    NORMAL = auto()
    WORLD_UP = auto()
    CUSTOM = auto()


class ScatterPattern(Enum):
    """Scatter distribution patterns."""
    RANDOM = auto()
    POISSON_DISK = auto()
    GRID_JITTER = auto()
    CLUSTER = auto()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(slots=True)
class Vector3:
    """3D vector for positions and directions."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> "Vector3":
        length = self.length()
        if length == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / length, self.y / length, self.z / length)

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )


@dataclass(slots=True)
class Quaternion:
    """Quaternion for rotations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(0, 0, 0, 1)

    @staticmethod
    def from_axis_angle(axis: Vector3, angle: float) -> "Quaternion":
        """Create quaternion from axis and angle (radians)."""
        half_angle = angle / 2
        s = math.sin(half_angle)
        normalized = axis.normalized()
        return Quaternion(
            normalized.x * s,
            normalized.y * s,
            normalized.z * s,
            math.cos(half_angle)
        )

    @staticmethod
    def from_euler(pitch: float, yaw: float, roll: float) -> "Quaternion":
        """Create quaternion from Euler angles (radians)."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        return Quaternion(
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        )


@dataclass(slots=True)
class Transform:
    """Transform with position, rotation, and scale."""
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))


@dataclass(slots=True)
class PlacementResult:
    """Result of a placement operation."""
    success: bool
    object_id: Optional[str] = None
    position: Optional[Vector3] = None
    rotation: Optional[Quaternion] = None
    scale: Optional[Vector3] = None
    error_message: Optional[str] = None


@dataclass(slots=True)
class BrushSettings:
    """Settings for paint brush placement mode."""
    radius: float = 5.0
    density: float = 1.0
    falloff: float = 0.5
    spacing: float = 1.0
    random_rotation: bool = True
    random_scale_min: float = 0.8
    random_scale_max: float = 1.2
    align_to_surface: bool = True
    surface_offset: float = 0.0


@dataclass(slots=True)
class ScatterSettings:
    """Settings for scatter placement mode."""
    region_min: Vector3 = field(default_factory=Vector3)
    region_max: Vector3 = field(default_factory=lambda: Vector3(10, 0, 10))
    count: int = 10
    pattern: ScatterPattern = ScatterPattern.RANDOM
    min_distance: float = 1.0
    seed: Optional[int] = None
    random_rotation: bool = True
    random_rotation_axis: Vector3 = field(default_factory=lambda: Vector3(0, 1, 0))
    random_scale_min: float = 0.8
    random_scale_max: float = 1.2
    align_to_surface: bool = False
    surface_layer_mask: int = 0xFFFFFFFF


@dataclass(slots=True)
class FoliageSettings:
    """Settings for foliage placement mode."""
    brush_radius: float = 10.0
    density: float = 100.0  # Instances per 100 square units
    falloff_curve: float = 0.5
    align_to_normal: bool = True
    random_yaw: bool = True
    random_pitch_max: float = 0.1  # Radians
    scale_min: float = 0.8
    scale_max: float = 1.2
    height_min: float = -1000.0
    height_max: float = 1000.0
    slope_min: float = 0.0
    slope_max: float = 45.0  # Degrees
    ground_layer_mask: int = 0xFFFFFFFF
    cull_distance: float = 1000.0


@dataclass(slots=True)
class SplinePoint:
    """A point on a spline."""
    position: Vector3 = field(default_factory=Vector3)
    tangent_in: Vector3 = field(default_factory=Vector3)
    tangent_out: Vector3 = field(default_factory=Vector3)
    roll: float = 0.0


@dataclass(slots=True)
class SplineSettings:
    """Settings for spline placement mode."""
    points: list[SplinePoint] = field(default_factory=list)
    spacing: float = 2.0
    align_to_spline: bool = True
    follow_roll: bool = True
    random_offset: float = 0.0
    random_scale_min: float = 1.0
    random_scale_max: float = 1.0
    closed_loop: bool = False
    interpolation_steps: int = 10


# =============================================================================
# Protocols
# =============================================================================

class Placeable(Protocol):
    """Protocol for objects that can be placed."""
    def get_bounds(self) -> tuple[Vector3, Vector3]: ...
    def get_pivot(self) -> Vector3: ...


class Surface(Protocol):
    """Protocol for surface raycasting."""
    def raycast(self, origin: Vector3, direction: Vector3) -> Optional[tuple[Vector3, Vector3]]:
        """Cast ray and return hit point and normal, or None."""
        ...


# =============================================================================
# Placement Tool
# =============================================================================

@editor
class PlacementTool:
    """
    Main placement tool supporting multiple placement modes.

    Integrates with Foundation Tracker for full undo/redo support.
    Uses @track_changes decorator to automatically record operations.
    """

    __slots__ = (
        "_mode",
        "_brush_settings",
        "_scatter_settings",
        "_foliage_settings",
        "_spline_settings",
        "_axis_constraint",
        "_surface_alignment",
        "_custom_alignment_direction",
        "_placed_objects",
        "_preview_transform",
        "_active",
        "_surface",
        "_callbacks",
        "__weakref__",
    )

    def __init__(self):
        """Initialize placement tool with default settings."""
        self._mode = PlacementMode.SINGLE
        self._brush_settings = BrushSettings()
        self._scatter_settings = ScatterSettings()
        self._foliage_settings = FoliageSettings()
        self._spline_settings = SplineSettings()
        self._axis_constraint = AxisConstraint.NONE
        self._surface_alignment = SurfaceAlignment.NONE
        self._custom_alignment_direction = Vector3(0, 1, 0)
        self._placed_objects: list[str] = []
        self._preview_transform: Optional[Transform] = None
        self._active = False
        self._surface: Optional[Surface] = None
        self._callbacks: dict[str, list[Callable]] = {
            "on_place": [],
            "on_preview": [],
            "on_mode_change": [],
        }

    @property
    def mode(self) -> PlacementMode:
        """Get current placement mode."""
        return self._mode

    @mode.setter
    def mode(self, value: PlacementMode) -> None:
        """Set placement mode."""
        old_mode = self._mode
        self._mode = value
        for callback in self._callbacks["on_mode_change"]:
            callback(old_mode, value)

    @property
    def brush_settings(self) -> BrushSettings:
        """Get brush settings."""
        return self._brush_settings

    @property
    def scatter_settings(self) -> ScatterSettings:
        """Get scatter settings."""
        return self._scatter_settings

    @property
    def foliage_settings(self) -> FoliageSettings:
        """Get foliage settings."""
        return self._foliage_settings

    @property
    def spline_settings(self) -> SplineSettings:
        """Get spline settings."""
        return self._spline_settings

    @property
    def axis_constraint(self) -> AxisConstraint:
        """Get axis constraint."""
        return self._axis_constraint

    @axis_constraint.setter
    def axis_constraint(self, value: AxisConstraint) -> None:
        """Set axis constraint."""
        self._axis_constraint = value

    @property
    def surface_alignment(self) -> SurfaceAlignment:
        """Get surface alignment mode."""
        return self._surface_alignment

    @surface_alignment.setter
    def surface_alignment(self, value: SurfaceAlignment) -> None:
        """Set surface alignment mode."""
        self._surface_alignment = value

    def set_surface(self, surface: Surface) -> None:
        """Set the surface for raycasting."""
        self._surface = surface

    def on(self, event: str, callback: Callable) -> None:
        """Register callback for placement events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Unregister callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def apply_axis_constraint(self, position: Vector3, reference: Vector3) -> Vector3:
        """Apply axis constraints to position."""
        if self._axis_constraint == AxisConstraint.NONE:
            return position
        elif self._axis_constraint == AxisConstraint.X:
            return Vector3(position.x, reference.y, reference.z)
        elif self._axis_constraint == AxisConstraint.Y:
            return Vector3(reference.x, position.y, reference.z)
        elif self._axis_constraint == AxisConstraint.Z:
            return Vector3(reference.x, reference.y, position.z)
        elif self._axis_constraint == AxisConstraint.XY:
            return Vector3(position.x, position.y, reference.z)
        elif self._axis_constraint == AxisConstraint.XZ:
            return Vector3(position.x, reference.y, position.z)
        elif self._axis_constraint == AxisConstraint.YZ:
            return Vector3(reference.x, position.y, position.z)
        return position

    def compute_alignment_rotation(
        self,
        surface_normal: Optional[Vector3] = None
    ) -> Quaternion:
        """Compute rotation based on alignment settings."""
        if self._surface_alignment == SurfaceAlignment.NONE:
            return Quaternion.identity()
        elif self._surface_alignment == SurfaceAlignment.WORLD_UP:
            return Quaternion.identity()
        elif self._surface_alignment == SurfaceAlignment.NORMAL and surface_normal:
            # Rotate to align Y-axis with surface normal
            up = Vector3(0, 1, 0)
            if abs(surface_normal.dot(up)) > 0.999:
                return Quaternion.identity()
            axis = up.cross(surface_normal)
            angle = math.acos(up.dot(surface_normal))
            return Quaternion.from_axis_angle(axis, angle)
        elif self._surface_alignment == SurfaceAlignment.CUSTOM:
            # Align to custom direction
            up = Vector3(0, 1, 0)
            target = self._custom_alignment_direction.normalized()
            if abs(target.dot(up)) > 0.999:
                return Quaternion.identity()
            axis = up.cross(target)
            angle = math.acos(up.dot(target))
            return Quaternion.from_axis_angle(axis, angle)
        return Quaternion.identity()

    @track_changes
    def place_single(
        self,
        position: Vector3,
        rotation: Optional[Quaternion] = None,
        scale: Optional[Vector3] = None,
        prefab_id: Optional[str] = None,
    ) -> PlacementResult:
        """
        Place a single object at the specified position.

        Args:
            position: World position for placement
            rotation: Optional rotation override
            scale: Optional scale override
            prefab_id: Optional prefab identifier

        Returns:
            PlacementResult with success status and object ID
        """
        object_id = str(uuid.uuid4())

        final_rotation = rotation if rotation else self.compute_alignment_rotation()
        final_scale = scale if scale else Vector3(1, 1, 1)

        # Record the placement
        self._placed_objects.append(object_id)

        # Mark placement data as dirty for tracking
        tracker.mark_dirty(self, "_placed_objects",
                          self._placed_objects[:-1],
                          self._placed_objects.copy())

        result = PlacementResult(
            success=True,
            object_id=object_id,
            position=position,
            rotation=final_rotation,
            scale=final_scale,
        )

        for callback in self._callbacks["on_place"]:
            callback(result)

        return result

    @track_changes
    def place_with_brush(
        self,
        center: Vector3,
        prefab_id: Optional[str] = None,
    ) -> list[PlacementResult]:
        """
        Place objects using brush settings.

        Args:
            center: Center position of brush
            prefab_id: Optional prefab identifier

        Returns:
            List of PlacementResult for each placed object
        """
        results = []
        settings = self._brush_settings

        # Calculate number of instances based on density and radius
        area = math.pi * settings.radius ** 2
        count = int(area * settings.density)

        for _ in range(max(1, count)):
            # Random position within brush radius
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, settings.radius)
            # Apply falloff
            distance *= (1 - (distance / settings.radius) * settings.falloff)

            offset_x = math.cos(angle) * distance
            offset_z = math.sin(angle) * distance

            pos = Vector3(
                center.x + offset_x,
                center.y + settings.surface_offset,
                center.z + offset_z,
            )

            # Random rotation if enabled
            rotation = Quaternion.identity()
            if settings.random_rotation:
                yaw = random.uniform(0, 2 * math.pi)
                rotation = Quaternion.from_axis_angle(Vector3(0, 1, 0), yaw)

            # Random scale
            scale_factor = random.uniform(
                settings.random_scale_min,
                settings.random_scale_max
            )
            scale = Vector3(scale_factor, scale_factor, scale_factor)

            object_id = str(uuid.uuid4())
            self._placed_objects.append(object_id)

            result = PlacementResult(
                success=True,
                object_id=object_id,
                position=pos,
                rotation=rotation,
                scale=scale,
            )
            results.append(result)

            for callback in self._callbacks["on_place"]:
                callback(result)

        tracker.mark_dirty(self, "_placed_objects",
                          self._placed_objects[:-len(results)],
                          self._placed_objects.copy())

        return results

    @track_changes
    def place_scatter(
        self,
        prefab_id: Optional[str] = None,
    ) -> list[PlacementResult]:
        """
        Scatter place objects within the defined region.

        Args:
            prefab_id: Optional prefab identifier

        Returns:
            List of PlacementResult for each placed object
        """
        results = []
        settings = self._scatter_settings

        # Set random seed if specified
        if settings.seed is not None:
            random.seed(settings.seed)

        positions = self._generate_scatter_positions(settings)

        for pos in positions:
            rotation = Quaternion.identity()
            if settings.random_rotation:
                angle = random.uniform(0, 2 * math.pi)
                rotation = Quaternion.from_axis_angle(
                    settings.random_rotation_axis, angle
                )

            scale_factor = random.uniform(
                settings.random_scale_min,
                settings.random_scale_max
            )
            scale = Vector3(scale_factor, scale_factor, scale_factor)

            object_id = str(uuid.uuid4())
            self._placed_objects.append(object_id)

            result = PlacementResult(
                success=True,
                object_id=object_id,
                position=pos,
                rotation=rotation,
                scale=scale,
            )
            results.append(result)

            for callback in self._callbacks["on_place"]:
                callback(result)

        tracker.mark_dirty(self, "_placed_objects",
                          self._placed_objects[:-len(results)],
                          self._placed_objects.copy())

        return results

    def _generate_scatter_positions(
        self,
        settings: ScatterSettings
    ) -> list[Vector3]:
        """Generate positions based on scatter pattern."""
        positions = []
        region_size = settings.region_max - settings.region_min

        if settings.pattern == ScatterPattern.RANDOM:
            for _ in range(settings.count):
                pos = Vector3(
                    settings.region_min.x + random.uniform(0, region_size.x),
                    settings.region_min.y + random.uniform(0, region_size.y),
                    settings.region_min.z + random.uniform(0, region_size.z),
                )
                positions.append(pos)

        elif settings.pattern == ScatterPattern.POISSON_DISK:
            positions = self._poisson_disk_sampling(settings)

        elif settings.pattern == ScatterPattern.GRID_JITTER:
            positions = self._grid_jitter_sampling(settings)

        elif settings.pattern == ScatterPattern.CLUSTER:
            positions = self._cluster_sampling(settings)

        return positions[:settings.count]

    def _poisson_disk_sampling(self, settings: ScatterSettings) -> list[Vector3]:
        """Generate Poisson disk distributed positions."""
        positions = []
        region_size = settings.region_max - settings.region_min
        cell_size = settings.min_distance / math.sqrt(2)

        # Grid dimensions
        grid_width = int(math.ceil(region_size.x / cell_size))
        grid_height = int(math.ceil(region_size.z / cell_size))
        grid: dict[tuple[int, int], Vector3] = {}

        # Start with a random point
        first_pos = Vector3(
            settings.region_min.x + random.uniform(0, region_size.x),
            settings.region_min.y,
            settings.region_min.z + random.uniform(0, region_size.z),
        )
        positions.append(first_pos)
        active = [first_pos]
        grid_x = int((first_pos.x - settings.region_min.x) / cell_size)
        grid_z = int((first_pos.z - settings.region_min.z) / cell_size)
        grid[(grid_x, grid_z)] = first_pos

        k = 30  # Number of attempts

        while active and len(positions) < settings.count * 2:
            idx = random.randint(0, len(active) - 1)
            point = active[idx]
            found = False

            for _ in range(k):
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(settings.min_distance, 2 * settings.min_distance)
                new_pos = Vector3(
                    point.x + math.cos(angle) * distance,
                    settings.region_min.y,
                    point.z + math.sin(angle) * distance,
                )

                # Check bounds
                if (new_pos.x < settings.region_min.x or
                    new_pos.x > settings.region_max.x or
                    new_pos.z < settings.region_min.z or
                    new_pos.z > settings.region_max.z):
                    continue

                gx = int((new_pos.x - settings.region_min.x) / cell_size)
                gz = int((new_pos.z - settings.region_min.z) / cell_size)

                # Check nearby cells
                valid = True
                for dx in range(-2, 3):
                    for dz in range(-2, 3):
                        neighbor = grid.get((gx + dx, gz + dz))
                        if neighbor:
                            dist = (new_pos - neighbor).length()
                            if dist < settings.min_distance:
                                valid = False
                                break
                    if not valid:
                        break

                if valid:
                    positions.append(new_pos)
                    active.append(new_pos)
                    grid[(gx, gz)] = new_pos
                    found = True
                    break

            if not found:
                active.pop(idx)

        return positions

    def _grid_jitter_sampling(self, settings: ScatterSettings) -> list[Vector3]:
        """Generate grid positions with random jitter."""
        positions = []
        region_size = settings.region_max - settings.region_min

        # Calculate grid spacing
        grid_count = int(math.sqrt(settings.count))
        if grid_count < 1:
            grid_count = 1

        step_x = region_size.x / grid_count
        step_z = region_size.z / grid_count
        jitter = settings.min_distance * 0.5

        for i in range(grid_count):
            for j in range(grid_count):
                pos = Vector3(
                    settings.region_min.x + (i + 0.5) * step_x + random.uniform(-jitter, jitter),
                    settings.region_min.y,
                    settings.region_min.z + (j + 0.5) * step_z + random.uniform(-jitter, jitter),
                )
                positions.append(pos)

        return positions

    def _cluster_sampling(self, settings: ScatterSettings) -> list[Vector3]:
        """Generate clustered positions."""
        positions = []
        region_size = settings.region_max - settings.region_min

        # Generate cluster centers
        cluster_count = max(1, settings.count // 5)
        cluster_radius = settings.min_distance * 3

        centers = []
        for _ in range(cluster_count):
            center = Vector3(
                settings.region_min.x + random.uniform(0, region_size.x),
                settings.region_min.y,
                settings.region_min.z + random.uniform(0, region_size.z),
            )
            centers.append(center)

        # Generate points around clusters
        points_per_cluster = settings.count // cluster_count

        for center in centers:
            for _ in range(points_per_cluster):
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, cluster_radius)
                pos = Vector3(
                    center.x + math.cos(angle) * distance,
                    center.y,
                    center.z + math.sin(angle) * distance,
                )
                # Clamp to region
                pos.x = max(settings.region_min.x, min(settings.region_max.x, pos.x))
                pos.z = max(settings.region_min.z, min(settings.region_max.z, pos.z))
                positions.append(pos)

        return positions

    @track_changes
    def place_foliage(
        self,
        center: Vector3,
        prefab_id: Optional[str] = None,
    ) -> list[PlacementResult]:
        """
        Place foliage using specialized foliage settings.

        Args:
            center: Center position for foliage brush
            prefab_id: Optional prefab identifier

        Returns:
            List of PlacementResult for each placed instance
        """
        results = []
        settings = self._foliage_settings

        # Calculate instance count
        area = math.pi * settings.brush_radius ** 2
        count = int(area * settings.density / 100.0)

        for _ in range(max(1, count)):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, settings.brush_radius)
            # Apply falloff
            falloff = 1 - (distance / settings.brush_radius) ** settings.falloff_curve
            if random.random() > falloff:
                continue

            pos = Vector3(
                center.x + math.cos(angle) * distance,
                center.y,
                center.z + math.sin(angle) * distance,
            )

            # Check height constraints
            if pos.y < settings.height_min or pos.y > settings.height_max:
                continue

            # Random rotation
            rotation = Quaternion.identity()
            if settings.random_yaw:
                yaw = random.uniform(0, 2 * math.pi)
                rotation = Quaternion.from_axis_angle(Vector3(0, 1, 0), yaw)

            # Random pitch if enabled
            if settings.random_pitch_max > 0:
                pitch = random.uniform(-settings.random_pitch_max, settings.random_pitch_max)
                pitch_quat = Quaternion.from_axis_angle(Vector3(1, 0, 0), pitch)
                # Combine rotations (simplified)
                rotation.x += pitch_quat.x
                rotation.y += pitch_quat.y
                rotation.z += pitch_quat.z

            # Random scale
            scale_factor = random.uniform(settings.scale_min, settings.scale_max)
            scale = Vector3(scale_factor, scale_factor, scale_factor)

            object_id = str(uuid.uuid4())
            self._placed_objects.append(object_id)

            result = PlacementResult(
                success=True,
                object_id=object_id,
                position=pos,
                rotation=rotation,
                scale=scale,
            )
            results.append(result)

            for callback in self._callbacks["on_place"]:
                callback(result)

        if results:
            tracker.mark_dirty(self, "_placed_objects",
                              self._placed_objects[:-len(results)],
                              self._placed_objects.copy())

        return results

    @track_changes
    def place_along_spline(
        self,
        prefab_id: Optional[str] = None,
    ) -> list[PlacementResult]:
        """
        Place objects along the defined spline.

        Args:
            prefab_id: Optional prefab identifier

        Returns:
            List of PlacementResult for each placed object
        """
        results = []
        settings = self._spline_settings

        if len(settings.points) < 2:
            return results

        # Calculate total spline length
        total_length = self._calculate_spline_length(settings)
        if total_length <= 0:
            return results

        # Place objects at regular intervals
        current_distance = 0.0

        while current_distance < total_length:
            pos, tangent = self._sample_spline(settings, current_distance / total_length)

            rotation = Quaternion.identity()
            if settings.align_to_spline:
                # Align forward direction with tangent
                forward = tangent.normalized()
                up = Vector3(0, 1, 0)
                right = up.cross(forward).normalized()
                if right.length() < 0.001:
                    right = Vector3(1, 0, 0)
                # Build rotation from basis vectors (simplified)
                yaw = math.atan2(forward.x, forward.z)
                rotation = Quaternion.from_axis_angle(Vector3(0, 1, 0), yaw)

            # Apply random offset
            if settings.random_offset > 0:
                offset = random.uniform(-settings.random_offset, settings.random_offset)
                right = Vector3(-tangent.z, 0, tangent.x).normalized()
                pos = pos + right * offset

            # Random scale
            scale_factor = random.uniform(
                settings.random_scale_min,
                settings.random_scale_max
            )
            scale = Vector3(scale_factor, scale_factor, scale_factor)

            object_id = str(uuid.uuid4())
            self._placed_objects.append(object_id)

            result = PlacementResult(
                success=True,
                object_id=object_id,
                position=pos,
                rotation=rotation,
                scale=scale,
            )
            results.append(result)

            for callback in self._callbacks["on_place"]:
                callback(result)

            current_distance += settings.spacing

        if results:
            tracker.mark_dirty(self, "_placed_objects",
                              self._placed_objects[:-len(results)],
                              self._placed_objects.copy())

        return results

    def _calculate_spline_length(self, settings: SplineSettings) -> float:
        """Calculate approximate total spline length."""
        if len(settings.points) < 2:
            return 0.0

        total = 0.0
        steps = settings.interpolation_steps

        for i in range(len(settings.points) - 1):
            p0 = settings.points[i].position
            p1 = settings.points[i + 1].position

            for j in range(steps):
                t0 = j / steps
                t1 = (j + 1) / steps
                pos0 = self._interpolate_segment(
                    settings.points[i], settings.points[i + 1], t0
                )
                pos1 = self._interpolate_segment(
                    settings.points[i], settings.points[i + 1], t1
                )
                total += (pos1 - pos0).length()

        return total

    def _sample_spline(
        self,
        settings: SplineSettings,
        t: float
    ) -> tuple[Vector3, Vector3]:
        """Sample position and tangent at parameter t (0-1)."""
        if len(settings.points) < 2:
            return Vector3(), Vector3(0, 0, 1)

        num_segments = len(settings.points) - 1
        segment_t = t * num_segments
        segment_idx = min(int(segment_t), num_segments - 1)
        local_t = segment_t - segment_idx

        pos = self._interpolate_segment(
            settings.points[segment_idx],
            settings.points[segment_idx + 1],
            local_t
        )

        # Calculate tangent using finite difference
        dt = 0.01
        pos_next = self._interpolate_segment(
            settings.points[segment_idx],
            settings.points[segment_idx + 1],
            min(1.0, local_t + dt)
        )
        tangent = (pos_next - pos).normalized()

        return pos, tangent

    def _interpolate_segment(
        self,
        p0: SplinePoint,
        p1: SplinePoint,
        t: float
    ) -> Vector3:
        """Hermite interpolation between two spline points."""
        t2 = t * t
        t3 = t2 * t

        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        return (
            p0.position * h00 +
            p0.tangent_out * h10 +
            p1.position * h01 +
            p1.tangent_in * h11
        )

    def update_preview(self, position: Vector3) -> Transform:
        """
        Update placement preview at position.

        Args:
            position: Current cursor world position

        Returns:
            Preview transform
        """
        transform = Transform(
            position=position,
            rotation=self.compute_alignment_rotation(),
            scale=Vector3(1, 1, 1),
        )
        self._preview_transform = transform

        for callback in self._callbacks["on_preview"]:
            callback(transform)

        return transform

    def get_placed_objects(self) -> list[str]:
        """Get list of all placed object IDs."""
        return self._placed_objects.copy()

    def clear_placed_objects(self) -> None:
        """Clear the list of placed objects."""
        old_list = self._placed_objects.copy()
        self._placed_objects.clear()
        tracker.mark_dirty(self, "_placed_objects", old_list, [])


__all__ = [
    "PlacementMode",
    "PlacementTool",
    "PlacementResult",
    "ScatterSettings",
    "FoliageSettings",
    "SplineSettings",
    "BrushSettings",
    "SplinePoint",
    "AxisConstraint",
    "SurfaceAlignment",
    "ScatterPattern",
    "Vector3",
    "Quaternion",
    "Transform",
    "editor",
    "track_changes",
]
