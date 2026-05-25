"""
Measurement Tools - Distance, angle, and area measurement utilities.

Provides:
- Distance measurement (point to point, point to surface)
- Angle measurement (3 points, surface normals)
- Area measurement (polygon area, surface area)
- Unit conversion (metric, imperial)
- Measurement visualization

All measurement operations integrate with Foundation Tracker for undo/redo.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from .placement import Vector3, editor, track_changes
from foundation.tracker import tracker


# =============================================================================
# Enums
# =============================================================================

class MeasurementUnit(Enum):
    """Units of measurement."""
    # Metric
    MILLIMETERS = ("mm", 0.001)
    CENTIMETERS = ("cm", 0.01)
    METERS = ("m", 1.0)
    KILOMETERS = ("km", 1000.0)
    # Imperial
    INCHES = ("in", 0.0254)
    FEET = ("ft", 0.3048)
    YARDS = ("yd", 0.9144)
    MILES = ("mi", 1609.344)
    # Game units
    UNITS = ("u", 1.0)

    def __init__(self, symbol: str, to_meters: float):
        self.symbol = symbol
        self.to_meters = to_meters

    @staticmethod
    def convert(value: float, from_unit: "MeasurementUnit", to_unit: "MeasurementUnit") -> float:
        """Convert value between units."""
        # Convert to meters first, then to target unit
        meters = value * from_unit.to_meters
        return meters / to_unit.to_meters


class MeasurementType(Enum):
    """Types of measurements."""
    DISTANCE = auto()
    ANGLE = auto()
    AREA = auto()
    PERIMETER = auto()
    VOLUME = auto()


class AngleUnit(Enum):
    """Units for angle measurement."""
    DEGREES = auto()
    RADIANS = auto()
    GRADIANS = auto()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(slots=True)
class MeasurementPoint:
    """A point used in measurement."""
    position: Vector3
    label: str = ""
    snapped: bool = False
    surface_normal: Optional[Vector3] = None


@dataclass(slots=True)
class MeasurementResult:
    """Result of a measurement operation."""
    measurement_type: MeasurementType
    value: float
    unit: MeasurementUnit
    formatted: str
    points: list[MeasurementPoint] = field(default_factory=list)
    auxiliary_values: dict[str, float] = field(default_factory=dict)


# =============================================================================
# Distance Measurement
# =============================================================================

@editor
class DistanceMeasurement:
    """
    Measure distance between points.

    Supports:
    - Point to point distance
    - Point to surface distance
    - Cumulative distance along path
    - Axis-aligned distances
    """

    __slots__ = (
        "_id",
        "_points",
        "_unit",
        "_show_axis_distances",
        "_accumulated",
        "__weakref__",
    )

    def __init__(self, unit: MeasurementUnit = MeasurementUnit.METERS):
        """
        Initialize distance measurement.

        Args:
            unit: Default measurement unit
        """
        self._id = str(uuid.uuid4())
        self._points: list[MeasurementPoint] = []
        self._unit = unit
        self._show_axis_distances = False
        self._accumulated = False

    @property
    def id(self) -> str:
        return self._id

    @property
    def points(self) -> list[MeasurementPoint]:
        return self._points.copy()

    @property
    def unit(self) -> MeasurementUnit:
        return self._unit

    @unit.setter
    def unit(self, value: MeasurementUnit) -> None:
        self._unit = value

    @property
    def show_axis_distances(self) -> bool:
        return self._show_axis_distances

    @show_axis_distances.setter
    def show_axis_distances(self, value: bool) -> None:
        self._show_axis_distances = value

    @property
    def accumulated(self) -> bool:
        return self._accumulated

    @accumulated.setter
    def accumulated(self, value: bool) -> None:
        self._accumulated = value

    @track_changes
    def add_point(
        self,
        position: Vector3,
        label: str = "",
        snapped: bool = False,
        surface_normal: Optional[Vector3] = None
    ) -> MeasurementPoint:
        """
        Add a measurement point.

        Args:
            position: World position
            label: Optional label
            snapped: Whether point was snapped
            surface_normal: Surface normal if snapped to surface

        Returns:
            Created measurement point
        """
        point = MeasurementPoint(
            position=position,
            label=label or f"P{len(self._points) + 1}",
            snapped=snapped,
            surface_normal=surface_normal,
        )
        old_points = self._points.copy()
        self._points.append(point)
        tracker.mark_dirty(self, "_points", old_points, self._points.copy())
        return point

    @track_changes
    def remove_point(self, index: int) -> bool:
        """Remove a point by index."""
        if 0 <= index < len(self._points):
            old_points = self._points.copy()
            self._points.pop(index)
            tracker.mark_dirty(self, "_points", old_points, self._points.copy())
            return True
        return False

    @track_changes
    def clear_points(self) -> None:
        """Clear all points."""
        old_points = self._points.copy()
        self._points.clear()
        tracker.mark_dirty(self, "_points", old_points, [])

    def calculate_distance(self, p1: Vector3, p2: Vector3) -> float:
        """Calculate distance between two points in meters."""
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dz = p2.z - p1.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def measure(self) -> Optional[MeasurementResult]:
        """
        Perform the distance measurement.

        Returns:
            MeasurementResult or None if insufficient points
        """
        if len(self._points) < 2:
            return None

        if self._accumulated:
            # Cumulative distance along path
            total = 0.0
            for i in range(len(self._points) - 1):
                total += self.calculate_distance(
                    self._points[i].position,
                    self._points[i + 1].position
                )
            distance_meters = total
        else:
            # Simple point to point
            distance_meters = self.calculate_distance(
                self._points[0].position,
                self._points[-1].position
            )

        # Convert to target unit
        value = MeasurementUnit.convert(
            distance_meters,
            MeasurementUnit.METERS,
            self._unit
        )

        # Calculate axis-aligned distances
        auxiliary = {}
        if self._show_axis_distances and len(self._points) >= 2:
            p1 = self._points[0].position
            p2 = self._points[-1].position
            auxiliary["x"] = MeasurementUnit.convert(
                abs(p2.x - p1.x), MeasurementUnit.METERS, self._unit
            )
            auxiliary["y"] = MeasurementUnit.convert(
                abs(p2.y - p1.y), MeasurementUnit.METERS, self._unit
            )
            auxiliary["z"] = MeasurementUnit.convert(
                abs(p2.z - p1.z), MeasurementUnit.METERS, self._unit
            )

        return MeasurementResult(
            measurement_type=MeasurementType.DISTANCE,
            value=value,
            unit=self._unit,
            formatted=f"{value:.3f} {self._unit.symbol}",
            points=self._points.copy(),
            auxiliary_values=auxiliary,
        )

    def get_segment_distances(self) -> list[float]:
        """Get distances of each segment in current unit."""
        distances = []
        for i in range(len(self._points) - 1):
            dist_m = self.calculate_distance(
                self._points[i].position,
                self._points[i + 1].position
            )
            distances.append(MeasurementUnit.convert(
                dist_m, MeasurementUnit.METERS, self._unit
            ))
        return distances


# =============================================================================
# Angle Measurement
# =============================================================================

@editor
class AngleMeasurement:
    """
    Measure angles between points or surfaces.

    Supports:
    - 3-point angle measurement
    - Surface normal angles
    - Angle to axis/plane
    """

    __slots__ = (
        "_id",
        "_points",
        "_angle_unit",
        "_reference_direction",
        "__weakref__",
    )

    def __init__(self, angle_unit: AngleUnit = AngleUnit.DEGREES):
        """
        Initialize angle measurement.

        Args:
            angle_unit: Unit for angle output
        """
        self._id = str(uuid.uuid4())
        self._points: list[MeasurementPoint] = []
        self._angle_unit = angle_unit
        self._reference_direction: Optional[Vector3] = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def points(self) -> list[MeasurementPoint]:
        return self._points.copy()

    @property
    def angle_unit(self) -> AngleUnit:
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self, value: AngleUnit) -> None:
        self._angle_unit = value

    @property
    def reference_direction(self) -> Optional[Vector3]:
        return self._reference_direction

    @reference_direction.setter
    def reference_direction(self, value: Optional[Vector3]) -> None:
        self._reference_direction = value

    @track_changes
    def add_point(
        self,
        position: Vector3,
        label: str = "",
        surface_normal: Optional[Vector3] = None
    ) -> MeasurementPoint:
        """Add a measurement point."""
        point = MeasurementPoint(
            position=position,
            label=label or f"A{len(self._points) + 1}",
            surface_normal=surface_normal,
        )
        old_points = self._points.copy()
        self._points.append(point)
        tracker.mark_dirty(self, "_points", old_points, self._points.copy())
        return point

    @track_changes
    def clear_points(self) -> None:
        """Clear all points."""
        old_points = self._points.copy()
        self._points.clear()
        tracker.mark_dirty(self, "_points", old_points, [])

    def _radians_to_unit(self, radians: float) -> float:
        """Convert radians to current angle unit."""
        if self._angle_unit == AngleUnit.DEGREES:
            return math.degrees(radians)
        elif self._angle_unit == AngleUnit.GRADIANS:
            return radians * (200 / math.pi)
        return radians

    def _get_unit_symbol(self) -> str:
        """Get symbol for current angle unit."""
        if self._angle_unit == AngleUnit.DEGREES:
            return "°"
        elif self._angle_unit == AngleUnit.RADIANS:
            return " rad"
        elif self._angle_unit == AngleUnit.GRADIANS:
            return " grad"
        return ""

    def measure_three_point(self) -> Optional[MeasurementResult]:
        """
        Measure angle at middle point of three points.

        Points define: P1 -> vertex -> P2

        Returns:
            MeasurementResult or None if insufficient points
        """
        if len(self._points) < 3:
            return None

        p1 = self._points[0].position
        vertex = self._points[1].position
        p2 = self._points[2].position

        # Vectors from vertex to other points
        v1 = Vector3(p1.x - vertex.x, p1.y - vertex.y, p1.z - vertex.z)
        v2 = Vector3(p2.x - vertex.x, p2.y - vertex.y, p2.z - vertex.z)

        # Normalize
        len1 = v1.length()
        len2 = v2.length()
        if len1 < 0.0001 or len2 < 0.0001:
            return None

        v1 = v1 * (1.0 / len1)
        v2 = v2 * (1.0 / len2)

        # Calculate angle
        dot = v1.dot(v2)
        dot = max(-1.0, min(1.0, dot))  # Clamp for numerical stability
        angle_rad = math.acos(dot)

        angle = self._radians_to_unit(angle_rad)
        symbol = self._get_unit_symbol()

        return MeasurementResult(
            measurement_type=MeasurementType.ANGLE,
            value=angle,
            unit=MeasurementUnit.UNITS,  # Angle unit stored separately
            formatted=f"{angle:.2f}{symbol}",
            points=self._points[:3],
            auxiliary_values={
                "radians": angle_rad,
                "degrees": math.degrees(angle_rad),
            },
        )

    def measure_surface_angle(self) -> Optional[MeasurementResult]:
        """
        Measure angle between two surface normals.

        Requires at least 2 points with surface normals.

        Returns:
            MeasurementResult or None
        """
        points_with_normals = [
            p for p in self._points
            if p.surface_normal is not None
        ]

        if len(points_with_normals) < 2:
            return None

        n1 = points_with_normals[0].surface_normal
        n2 = points_with_normals[1].surface_normal

        if n1 is None or n2 is None:
            return None

        # Normalize
        n1 = n1.normalized()
        n2 = n2.normalized()

        # Calculate angle
        dot = n1.dot(n2)
        dot = max(-1.0, min(1.0, dot))
        angle_rad = math.acos(dot)

        angle = self._radians_to_unit(angle_rad)
        symbol = self._get_unit_symbol()

        return MeasurementResult(
            measurement_type=MeasurementType.ANGLE,
            value=angle,
            unit=MeasurementUnit.UNITS,
            formatted=f"{angle:.2f}{symbol}",
            points=points_with_normals[:2],
            auxiliary_values={
                "radians": angle_rad,
                "degrees": math.degrees(angle_rad),
            },
        )

    def measure_to_reference(self) -> Optional[MeasurementResult]:
        """
        Measure angle from first point to reference direction.

        Returns:
            MeasurementResult or None
        """
        if len(self._points) < 2 or self._reference_direction is None:
            return None

        p1 = self._points[0].position
        p2 = self._points[1].position

        direction = Vector3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z).normalized()
        reference = self._reference_direction.normalized()

        dot = direction.dot(reference)
        dot = max(-1.0, min(1.0, dot))
        angle_rad = math.acos(dot)

        angle = self._radians_to_unit(angle_rad)
        symbol = self._get_unit_symbol()

        return MeasurementResult(
            measurement_type=MeasurementType.ANGLE,
            value=angle,
            unit=MeasurementUnit.UNITS,
            formatted=f"{angle:.2f}{symbol}",
            points=self._points[:2],
            auxiliary_values={
                "radians": angle_rad,
                "degrees": math.degrees(angle_rad),
            },
        )


# =============================================================================
# Area Measurement
# =============================================================================

@editor
class AreaMeasurement:
    """
    Measure area of polygons and surfaces.

    Supports:
    - Polygon area (3+ points)
    - Rectangle from 2 points
    - Circle from center and radius
    """

    __slots__ = (
        "_id",
        "_points",
        "_unit",
        "_closed",
        "__weakref__",
    )

    def __init__(self, unit: MeasurementUnit = MeasurementUnit.METERS):
        """
        Initialize area measurement.

        Args:
            unit: Linear unit (area will be unit squared)
        """
        self._id = str(uuid.uuid4())
        self._points: list[MeasurementPoint] = []
        self._unit = unit
        self._closed = True

    @property
    def id(self) -> str:
        return self._id

    @property
    def points(self) -> list[MeasurementPoint]:
        return self._points.copy()

    @property
    def unit(self) -> MeasurementUnit:
        return self._unit

    @unit.setter
    def unit(self, value: MeasurementUnit) -> None:
        self._unit = value

    @property
    def closed(self) -> bool:
        return self._closed

    @closed.setter
    def closed(self, value: bool) -> None:
        self._closed = value

    @track_changes
    def add_point(self, position: Vector3, label: str = "") -> MeasurementPoint:
        """Add a boundary point."""
        point = MeasurementPoint(
            position=position,
            label=label or f"V{len(self._points) + 1}",
        )
        old_points = self._points.copy()
        self._points.append(point)
        tracker.mark_dirty(self, "_points", old_points, self._points.copy())
        return point

    @track_changes
    def clear_points(self) -> None:
        """Clear all points."""
        old_points = self._points.copy()
        self._points.clear()
        tracker.mark_dirty(self, "_points", old_points, [])

    def _calculate_polygon_area_2d(self, points: list[Vector3]) -> float:
        """Calculate polygon area using Shoelace formula (XZ plane)."""
        n = len(points)
        if n < 3:
            return 0.0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i].x * points[j].z
            area -= points[j].x * points[i].z

        return abs(area) / 2.0

    def _calculate_polygon_area_3d(self, points: list[Vector3]) -> float:
        """Calculate polygon area in 3D using cross product."""
        n = len(points)
        if n < 3:
            return 0.0

        # Use Newell's method to find normal and area
        normal = Vector3(0, 0, 0)

        for i in range(n):
            j = (i + 1) % n
            normal.x += (points[i].y - points[j].y) * (points[i].z + points[j].z)
            normal.y += (points[i].z - points[j].z) * (points[i].x + points[j].x)
            normal.z += (points[i].x - points[j].x) * (points[i].y + points[j].y)

        return normal.length() / 2.0

    def measure(self) -> Optional[MeasurementResult]:
        """
        Calculate polygon area.

        Returns:
            MeasurementResult or None if insufficient points
        """
        if len(self._points) < 3:
            return None

        positions = [p.position for p in self._points]

        # Calculate area in square meters
        area_sq_meters = self._calculate_polygon_area_3d(positions)

        # Convert linear unit scale
        unit_scale = MeasurementUnit.convert(1.0, MeasurementUnit.METERS, self._unit)
        area = area_sq_meters * (unit_scale * unit_scale)

        # Calculate perimeter
        perimeter = 0.0
        n = len(positions)
        for i in range(n):
            j = (i + 1) % n if self._closed else i + 1
            if j < n:
                dx = positions[j].x - positions[i].x
                dy = positions[j].y - positions[i].y
                dz = positions[j].z - positions[i].z
                perimeter += math.sqrt(dx*dx + dy*dy + dz*dz)

        perimeter_converted = MeasurementUnit.convert(
            perimeter, MeasurementUnit.METERS, self._unit
        )

        return MeasurementResult(
            measurement_type=MeasurementType.AREA,
            value=area,
            unit=self._unit,
            formatted=f"{area:.3f} {self._unit.symbol}²",
            points=self._points.copy(),
            auxiliary_values={
                "perimeter": perimeter_converted,
                "vertex_count": len(self._points),
            },
        )

    def measure_rectangle(
        self,
        p1: Vector3,
        p2: Vector3
    ) -> MeasurementResult:
        """
        Calculate area of rectangle defined by two corner points.

        Args:
            p1: First corner
            p2: Opposite corner

        Returns:
            MeasurementResult
        """
        width = abs(p2.x - p1.x)
        depth = abs(p2.z - p1.z)
        area_sq_meters = width * depth

        unit_scale = MeasurementUnit.convert(1.0, MeasurementUnit.METERS, self._unit)
        area = area_sq_meters * (unit_scale * unit_scale)

        width_converted = MeasurementUnit.convert(width, MeasurementUnit.METERS, self._unit)
        depth_converted = MeasurementUnit.convert(depth, MeasurementUnit.METERS, self._unit)

        return MeasurementResult(
            measurement_type=MeasurementType.AREA,
            value=area,
            unit=self._unit,
            formatted=f"{area:.3f} {self._unit.symbol}²",
            points=[MeasurementPoint(p1), MeasurementPoint(p2)],
            auxiliary_values={
                "width": width_converted,
                "depth": depth_converted,
                "perimeter": 2 * (width_converted + depth_converted),
            },
        )

    def measure_circle(
        self,
        center: Vector3,
        radius: float
    ) -> MeasurementResult:
        """
        Calculate area of circle.

        Args:
            center: Center point
            radius: Radius in meters

        Returns:
            MeasurementResult
        """
        area_sq_meters = math.pi * radius * radius
        circumference = 2 * math.pi * radius

        unit_scale = MeasurementUnit.convert(1.0, MeasurementUnit.METERS, self._unit)
        area = area_sq_meters * (unit_scale * unit_scale)

        radius_converted = MeasurementUnit.convert(radius, MeasurementUnit.METERS, self._unit)
        circumference_converted = MeasurementUnit.convert(
            circumference, MeasurementUnit.METERS, self._unit
        )

        return MeasurementResult(
            measurement_type=MeasurementType.AREA,
            value=area,
            unit=self._unit,
            formatted=f"{area:.3f} {self._unit.symbol}²",
            points=[MeasurementPoint(center, "Center")],
            auxiliary_values={
                "radius": radius_converted,
                "diameter": radius_converted * 2,
                "circumference": circumference_converted,
            },
        )


# =============================================================================
# Measurement Tool (Main Interface)
# =============================================================================

@editor
class MeasurementTool:
    """
    Main measurement tool combining distance, angle, and area measurements.
    """

    __slots__ = (
        "_active_measurement",
        "_distance",
        "_angle",
        "_area",
        "_history",
        "_callbacks",
        "_unit",
        "_angle_unit",
        "__weakref__",
    )

    def __init__(self):
        """Initialize measurement tool."""
        self._active_measurement: Optional[MeasurementType] = None
        self._distance = DistanceMeasurement()
        self._angle = AngleMeasurement()
        self._area = AreaMeasurement()
        self._history: list[MeasurementResult] = []
        self._callbacks: dict[str, list[Callable]] = {
            "on_measure": [],
            "on_point_add": [],
            "on_clear": [],
        }
        self._unit = MeasurementUnit.METERS
        self._angle_unit = AngleUnit.DEGREES

    @property
    def active_measurement(self) -> Optional[MeasurementType]:
        return self._active_measurement

    @property
    def distance(self) -> DistanceMeasurement:
        return self._distance

    @property
    def angle(self) -> AngleMeasurement:
        return self._angle

    @property
    def area(self) -> AreaMeasurement:
        return self._area

    @property
    def history(self) -> list[MeasurementResult]:
        return self._history.copy()

    @property
    def unit(self) -> MeasurementUnit:
        return self._unit

    @unit.setter
    def unit(self, value: MeasurementUnit) -> None:
        self._unit = value
        self._distance.unit = value
        self._area.unit = value

    @property
    def angle_unit(self) -> AngleUnit:
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self, value: AngleUnit) -> None:
        self._angle_unit = value
        self._angle.angle_unit = value

    def on(self, event: str, callback: Callable) -> None:
        """Register callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Unregister callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def start_measurement(self, measurement_type: MeasurementType) -> None:
        """Start a new measurement of specified type."""
        self._active_measurement = measurement_type
        self.clear_current()

    @track_changes
    def add_point(
        self,
        position: Vector3,
        label: str = "",
        surface_normal: Optional[Vector3] = None
    ) -> Optional[MeasurementPoint]:
        """
        Add a point to the active measurement.

        Args:
            position: World position
            label: Optional label
            surface_normal: Surface normal if applicable

        Returns:
            Created point or None if no active measurement
        """
        if self._active_measurement is None:
            return None

        point = None

        if self._active_measurement == MeasurementType.DISTANCE:
            point = self._distance.add_point(position, label, False, surface_normal)
        elif self._active_measurement == MeasurementType.ANGLE:
            point = self._angle.add_point(position, label, surface_normal)
        elif self._active_measurement == MeasurementType.AREA:
            point = self._area.add_point(position, label)

        if point:
            for callback in self._callbacks["on_point_add"]:
                callback(point)

        return point

    def measure(self) -> Optional[MeasurementResult]:
        """
        Perform measurement and return result.

        Returns:
            MeasurementResult or None
        """
        result = None

        if self._active_measurement == MeasurementType.DISTANCE:
            result = self._distance.measure()
        elif self._active_measurement == MeasurementType.ANGLE:
            result = self._angle.measure_three_point()
        elif self._active_measurement == MeasurementType.AREA:
            result = self._area.measure()

        if result:
            self._history.append(result)
            for callback in self._callbacks["on_measure"]:
                callback(result)

        return result

    def clear_current(self) -> None:
        """Clear current measurement points."""
        if self._active_measurement == MeasurementType.DISTANCE:
            self._distance.clear_points()
        elif self._active_measurement == MeasurementType.ANGLE:
            self._angle.clear_points()
        elif self._active_measurement == MeasurementType.AREA:
            self._area.clear_points()

        for callback in self._callbacks["on_clear"]:
            callback()

    def clear_all(self) -> None:
        """Clear all measurements and history."""
        self._distance.clear_points()
        self._angle.clear_points()
        self._area.clear_points()
        self._history.clear()
        self._active_measurement = None

    def clear_history(self) -> None:
        """Clear measurement history."""
        self._history.clear()

    def get_last_measurement(self) -> Optional[MeasurementResult]:
        """Get the most recent measurement."""
        return self._history[-1] if self._history else None

    def convert_value(
        self,
        value: float,
        from_unit: MeasurementUnit,
        to_unit: MeasurementUnit
    ) -> float:
        """Convert a value between units."""
        return MeasurementUnit.convert(value, from_unit, to_unit)


__all__ = [
    "MeasurementUnit",
    "MeasurementType",
    "MeasurementTool",
    "MeasurementResult",
    "MeasurementPoint",
    "DistanceMeasurement",
    "AngleMeasurement",
    "AreaMeasurement",
    "AngleUnit",
]
