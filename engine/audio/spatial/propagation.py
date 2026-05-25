"""Sound Propagation System.

Calculates how sound travels through the environment:
- Direct paths
- Reflections (image source method)
- Diffraction around edges
- Transmission through materials
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.audio.spatial.config import (
    DIFFRACTION_ANGLE_THRESHOLD,
    MAX_DIFFRACTION_PATHS,
    MAX_PROPAGATION_DISTANCE,
    MAX_REFLECTION_ORDER,
    MIN_REFLECTION_COEFFICIENT,
    PROPAGATION_UPDATE_RATE,
    SPEED_OF_SOUND,
    SQRT_ONE_HALF,
    UTD_PATH_DECAY_FACTOR,
)
from engine.core.math.vec import Vec3


class PathType(Enum):
    """Types of sound propagation paths."""

    DIRECT = auto()
    """Direct line-of-sight path."""

    REFLECTION = auto()
    """Reflected off a surface."""

    DIFFRACTION = auto()
    """Diffracted around an edge."""

    TRANSMISSION = auto()
    """Transmitted through a surface."""

    COUPLED = auto()
    """Sound coupled between rooms/spaces."""


@dataclass
class PropagationPath:
    """A single sound propagation path."""

    path_type: PathType
    """Type of this propagation path."""

    points: List[Vec3] = field(default_factory=list)
    """Waypoints along the path (source, reflections, listener)."""

    total_distance: float = 0.0
    """Total path length in meters."""

    attenuation: float = 1.0
    """Additional attenuation factor (0-1)."""

    delay_seconds: float = 0.0
    """Propagation delay in seconds."""

    direction: Vec3 = field(default_factory=Vec3)
    """Arrival direction at listener (normalized)."""

    order: int = 0
    """Reflection/diffraction order (0 = direct)."""

    frequency_response: Optional[List[Tuple[float, float]]] = None
    """Optional frequency-dependent attenuation [(freq, gain), ...]."""

    material_id: Optional[str] = None
    """Material ID for reflections/transmissions."""


@dataclass
class PropagationResult:
    """Result of propagation calculation."""

    paths: List[PropagationPath] = field(default_factory=list)
    """All calculated propagation paths."""

    total_energy: float = 0.0
    """Total sound energy from all paths."""

    dominant_direction: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 1.0))
    """Energy-weighted dominant arrival direction."""

    reverb_contribution: float = 0.0
    """Estimated reverb contribution (0-1)."""

    direct_path_blocked: bool = False
    """Whether the direct path is occluded."""


@dataclass
class ReflectionSurface:
    """A surface that can reflect sound."""

    surface_id: int
    """Unique identifier."""

    center: Vec3
    """Center point of the surface."""

    normal: Vec3
    """Surface normal (facing away from reflective side)."""

    half_extents: Vec3
    """Half-size of the surface rectangle."""

    reflection_coefficient: float = 0.8
    """How much sound is reflected (0-1)."""

    material_id: Optional[str] = None
    """Material identifier."""


@dataclass
class DiffractionEdge:
    """An edge that can diffract sound."""

    edge_id: int
    """Unique identifier."""

    point_a: Vec3
    """Start point of the edge."""

    point_b: Vec3
    """End point of the edge."""

    normal_a: Vec3
    """Normal of first adjacent surface."""

    normal_b: Vec3
    """Normal of second adjacent surface."""

    wedge_angle: float = 90.0
    """Angle between the two surfaces (degrees)."""


# Type aliases for geometry query callbacks
GeometryRaycastFunction = Callable[[Vec3, Vec3, float], Optional[Dict[str, Any]]]
EdgeQueryFunction = Callable[[Vec3, Vec3], List[DiffractionEdge]]
SurfaceQueryFunction = Callable[[Vec3, float], List[ReflectionSurface]]


class PropagationCalculator:
    """Calculates sound propagation paths through the environment."""

    def __init__(
        self,
        max_reflection_order: int = MAX_REFLECTION_ORDER,
        max_diffraction_paths: int = MAX_DIFFRACTION_PATHS,
        speed_of_sound: float = SPEED_OF_SOUND
    ) -> None:
        """Initialize the propagation calculator.

        Args:
            max_reflection_order: Maximum reflection bounces to calculate.
            max_diffraction_paths: Maximum diffraction paths per source.
            speed_of_sound: Speed of sound in m/s.
        """
        self._max_reflection_order = max(0, min(4, max_reflection_order))
        self._max_diffraction_paths = max(0, max_diffraction_paths)
        self._speed_of_sound = max(1.0, speed_of_sound)

        self._geometry_raycast: Optional[GeometryRaycastFunction] = None
        self._edge_query: Optional[EdgeQueryFunction] = None
        self._surface_query: Optional[SurfaceQueryFunction] = None

    @property
    def max_reflection_order(self) -> int:
        """Get maximum reflection order."""
        return self._max_reflection_order

    @max_reflection_order.setter
    def max_reflection_order(self, value: int) -> None:
        """Set maximum reflection order."""
        self._max_reflection_order = max(0, min(4, value))

    @property
    def max_diffraction_paths(self) -> int:
        """Get maximum diffraction paths."""
        return self._max_diffraction_paths

    @max_diffraction_paths.setter
    def max_diffraction_paths(self, value: int) -> None:
        """Set maximum diffraction paths."""
        self._max_diffraction_paths = max(0, value)

    @property
    def speed_of_sound(self) -> float:
        """Get speed of sound."""
        return self._speed_of_sound

    @speed_of_sound.setter
    def speed_of_sound(self, value: float) -> None:
        """Set speed of sound."""
        self._speed_of_sound = max(1.0, value)

    def set_geometry_raycast(self, func: Optional[GeometryRaycastFunction]) -> None:
        """Set function for geometry raycasts.

        Args:
            func: Function(origin, direction, max_distance) -> hit dict or None.
                  Hit dict should contain: hit, point, normal, distance, material_id, reflection.
        """
        self._geometry_raycast = func

    def set_edge_query(self, func: Optional[EdgeQueryFunction]) -> None:
        """Set function for finding diffraction edges.

        Args:
            func: Function(source, listener) -> list of DiffractionEdge.
        """
        self._edge_query = func

    def set_surface_query(self, func: Optional[SurfaceQueryFunction]) -> None:
        """Set function for finding reflection surfaces.

        Args:
            func: Function(position, radius) -> list of ReflectionSurface.
        """
        self._surface_query = func

    def calculate(
        self,
        source_pos: Vec3,
        listener_pos: Vec3,
        include_reflections: bool = True,
        include_diffraction: bool = True,
        max_distance: float = MAX_PROPAGATION_DISTANCE
    ) -> PropagationResult:
        """Calculate all propagation paths between source and listener.

        Args:
            source_pos: Sound source position.
            listener_pos: Listener position.
            include_reflections: Whether to calculate reflection paths.
            include_diffraction: Whether to calculate diffraction paths.
            max_distance: Maximum path distance to consider.

        Returns:
            PropagationResult with all calculated paths.
        """
        paths: List[PropagationPath] = []
        direct_blocked = False

        # Calculate direct path
        direct = self._calculate_direct_path(source_pos, listener_pos, max_distance)
        if direct is not None:
            if direct.attenuation > 0.0:
                paths.append(direct)
            else:
                direct_blocked = True

        # Calculate reflections
        if include_reflections and self._max_reflection_order > 0:
            reflections = self._calculate_reflections(
                source_pos, listener_pos, max_distance
            )
            paths.extend(reflections)

        # Calculate diffraction
        if include_diffraction and self._max_diffraction_paths > 0:
            diffractions = self._calculate_diffraction(
                source_pos, listener_pos, max_distance
            )
            paths.extend(diffractions)

        # Calculate totals
        total_energy = sum(p.attenuation for p in paths)
        dominant_dir = self._calculate_dominant_direction(paths)
        reverb = self._estimate_reverb_contribution(paths)

        return PropagationResult(
            paths=paths,
            total_energy=total_energy,
            dominant_direction=dominant_dir,
            reverb_contribution=reverb,
            direct_path_blocked=direct_blocked
        )

    def _calculate_direct_path(
        self,
        source: Vec3,
        listener: Vec3,
        max_distance: float
    ) -> Optional[PropagationPath]:
        """Calculate direct path between source and listener.

        Args:
            source: Source position.
            listener: Listener position.
            max_distance: Maximum distance to consider.

        Returns:
            Direct PropagationPath or None if too far.
        """
        direction = listener - source
        distance = direction.length()

        if distance < 0.0001:
            return PropagationPath(
                path_type=PathType.DIRECT,
                points=[source, listener],
                total_distance=0.0,
                attenuation=1.0,
                delay_seconds=0.0,
                direction=Vec3(0.0, 0.0, 1.0),
                order=0
            )

        if distance > max_distance:
            return None

        direction = direction.normalized()
        delay = distance / self._speed_of_sound

        # Check for occlusion using raycast
        attenuation = 1.0
        if self._geometry_raycast is not None:
            hit = self._geometry_raycast(source, direction, distance)
            if hit is not None and hit.get("hit", False):
                hit_dist = hit.get("distance", distance)
                if hit_dist < distance - 0.01:
                    # Path is blocked
                    attenuation = 0.0

        return PropagationPath(
            path_type=PathType.DIRECT,
            points=[source, listener],
            total_distance=distance,
            attenuation=attenuation,
            delay_seconds=delay,
            direction=-direction,  # Direction TO listener
            order=0
        )

    def _calculate_reflections(
        self,
        source: Vec3,
        listener: Vec3,
        max_distance: float
    ) -> List[PropagationPath]:
        """Calculate reflection paths using simplified image source method.

        Args:
            source: Source position.
            listener: Listener position.
            max_distance: Maximum path distance.

        Returns:
            List of reflection PropagationPaths.
        """
        paths: List[PropagationPath] = []

        if self._geometry_raycast is None:
            return paths

        # Cast rays in multiple directions to find potential reflectors
        directions = self._get_sample_directions()

        for direction in directions:
            hit = self._geometry_raycast(source, direction, max_distance)

            if hit is None or not hit.get("hit", False):
                continue

            reflection_point = hit.get("point")
            if reflection_point is None:
                continue

            if isinstance(reflection_point, dict):
                reflection_point = Vec3(
                    reflection_point.get("x", 0.0),
                    reflection_point.get("y", 0.0),
                    reflection_point.get("z", 0.0)
                )
            elif isinstance(reflection_point, (list, tuple)):
                reflection_point = Vec3(
                    reflection_point[0],
                    reflection_point[1],
                    reflection_point[2]
                )

            normal = hit.get("normal", Vec3(0.0, 1.0, 0.0))
            if isinstance(normal, dict):
                normal = Vec3(
                    normal.get("x", 0.0),
                    normal.get("y", 1.0),
                    normal.get("z", 0.0)
                )
            elif isinstance(normal, (list, tuple)):
                normal = Vec3(normal[0], normal[1], normal[2])

            reflection_coeff = hit.get("reflection", 0.8)

            if reflection_coeff < MIN_REFLECTION_COEFFICIENT:
                continue

            # Calculate image source
            to_reflection = reflection_point - source
            d1 = to_reflection.length()

            # Calculate reflected direction toward listener
            to_listener = listener - reflection_point
            d2 = to_listener.length()

            total_dist = d1 + d2

            if total_dist > max_distance:
                continue

            # Verify listener is visible from reflection point
            if self._geometry_raycast is not None:
                listener_dir = to_listener.normalized() if d2 > 0.0001 else Vec3(0, 0, 1)
                block_check = self._geometry_raycast(reflection_point, listener_dir, d2)
                if block_check is not None and block_check.get("hit", False):
                    block_dist = block_check.get("distance", d2)
                    if block_dist < d2 - 0.01:
                        continue  # Path to listener is blocked

            arrival_dir = -to_listener.normalized() if d2 > 0.0001 else Vec3(0, 0, 1)

            paths.append(PropagationPath(
                path_type=PathType.REFLECTION,
                points=[source, reflection_point, listener],
                total_distance=total_dist,
                attenuation=reflection_coeff,
                delay_seconds=total_dist / self._speed_of_sound,
                direction=arrival_dir,
                order=1,
                material_id=hit.get("material_id")
            ))

        # Limit to reasonable number
        paths.sort(key=lambda p: p.total_distance)
        max_reflections = self._max_reflection_order * 6
        return paths[:max_reflections]

    def _calculate_diffraction(
        self,
        source: Vec3,
        listener: Vec3,
        max_distance: float
    ) -> List[PropagationPath]:
        """Calculate diffraction paths around edges.

        Uses simplified Uniform Theory of Diffraction (UTD).

        Args:
            source: Source position.
            listener: Listener position.
            max_distance: Maximum path distance.

        Returns:
            List of diffraction PropagationPaths.
        """
        paths: List[PropagationPath] = []

        if self._edge_query is None:
            return paths

        edges = self._edge_query(source, listener)

        for edge in edges[:self._max_diffraction_paths]:
            # Find closest point on edge
            edge_point = self._closest_point_on_segment(
                source, listener, edge.point_a, edge.point_b
            )

            d1 = (edge_point - source).length()
            d2 = (listener - edge_point).length()
            total_dist = d1 + d2
            direct_dist = (listener - source).length()

            if total_dist > max_distance:
                continue

            # Calculate diffraction loss based on path length difference
            # and wedge angle (simplified UTD)
            path_diff = total_dist - direct_dist

            # Fresnel parameter approximation
            # Higher path difference = more attenuation
            wedge_factor = 1.0 - (edge.wedge_angle / 180.0)
            diffraction_loss = self._calculate_utd_attenuation(
                path_diff, wedge_factor
            )

            if diffraction_loss < 0.01:
                continue

            arrival_dir = (listener - edge_point)
            if arrival_dir.length() > 0.0001:
                arrival_dir = -arrival_dir.normalized()
            else:
                arrival_dir = Vec3(0, 0, 1)

            paths.append(PropagationPath(
                path_type=PathType.DIFFRACTION,
                points=[source, edge_point, listener],
                total_distance=total_dist,
                attenuation=diffraction_loss,
                delay_seconds=total_dist / self._speed_of_sound,
                direction=arrival_dir,
                order=1
            ))

        return paths

    def _calculate_utd_attenuation(
        self,
        path_difference: float,
        wedge_factor: float
    ) -> float:
        """Calculate diffraction attenuation using simplified UTD.

        Args:
            path_difference: Extra path length vs direct (meters).
            wedge_factor: Factor based on wedge angle (0-1).

        Returns:
            Attenuation factor (0-1).
        """
        # Simplified model: exponential decay with path difference
        # Modified by wedge angle (sharper = less diffraction)
        base_attenuation = math.exp(-path_difference * UTD_PATH_DECAY_FACTOR)
        return base_attenuation * (0.3 + 0.7 * wedge_factor)

    def _closest_point_on_segment(
        self,
        source: Vec3,
        listener: Vec3,
        seg_a: Vec3,
        seg_b: Vec3
    ) -> Vec3:
        """Find closest point on a line segment to the source-listener line.

        Args:
            source: Source position.
            listener: Listener position.
            seg_a: Segment start.
            seg_b: Segment end.

        Returns:
            Closest point on the segment.
        """
        # Simplified: find point on segment closest to midpoint of source-listener
        midpoint = Vec3(
            (source.x + listener.x) * 0.5,
            (source.y + listener.y) * 0.5,
            (source.z + listener.z) * 0.5
        )

        seg_dir = seg_b - seg_a
        seg_len = seg_dir.length()

        if seg_len < 0.0001:
            return seg_a

        seg_dir = seg_dir / seg_len

        to_mid = midpoint - seg_a
        t = to_mid.x * seg_dir.x + to_mid.y * seg_dir.y + to_mid.z * seg_dir.z
        t = max(0.0, min(seg_len, t))

        return Vec3(
            seg_a.x + seg_dir.x * t,
            seg_a.y + seg_dir.y * t,
            seg_a.z + seg_dir.z * t
        )

    def _calculate_dominant_direction(
        self,
        paths: List[PropagationPath]
    ) -> Vec3:
        """Calculate energy-weighted dominant direction.

        Args:
            paths: List of propagation paths.

        Returns:
            Normalized dominant direction vector.
        """
        if not paths:
            return Vec3(0.0, 0.0, 1.0)

        weighted_x = 0.0
        weighted_y = 0.0
        weighted_z = 0.0
        total_weight = 0.0

        for path in paths:
            weight = path.attenuation
            weighted_x += path.direction.x * weight
            weighted_y += path.direction.y * weight
            weighted_z += path.direction.z * weight
            total_weight += weight

        if total_weight < 0.0001:
            return Vec3(0.0, 0.0, 1.0)

        result = Vec3(
            weighted_x / total_weight,
            weighted_y / total_weight,
            weighted_z / total_weight
        )

        length = result.length()
        if length < 0.0001:
            return Vec3(0.0, 0.0, 1.0)

        return result / length

    def _estimate_reverb_contribution(
        self,
        paths: List[PropagationPath]
    ) -> float:
        """Estimate how much the paths contribute to perceived reverb.

        Args:
            paths: List of propagation paths.

        Returns:
            Reverb contribution factor (0-1).
        """
        if not paths:
            return 0.0

        direct_energy = 0.0
        indirect_energy = 0.0

        for path in paths:
            if path.path_type == PathType.DIRECT:
                direct_energy += path.attenuation
            else:
                indirect_energy += path.attenuation

        total_energy = direct_energy + indirect_energy
        if total_energy < 0.0001:
            return 0.0

        return min(1.0, indirect_energy / total_energy)

    def _get_sample_directions(self) -> List[Vec3]:
        """Get sample directions for reflection tracing.

        Returns:
            List of normalized direction vectors.
        """
        # Cardinal and diagonal directions
        dirs = [
            Vec3(1.0, 0.0, 0.0),
            Vec3(-1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, -1.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
            Vec3(0.0, 0.0, -1.0),
        ]

        # Add diagonal directions using SQRT_ONE_HALF for normalized diagonal components
        diagonals = [
            Vec3(SQRT_ONE_HALF, SQRT_ONE_HALF, 0.0),
            Vec3(-SQRT_ONE_HALF, SQRT_ONE_HALF, 0.0),
            Vec3(SQRT_ONE_HALF, -SQRT_ONE_HALF, 0.0),
            Vec3(-SQRT_ONE_HALF, -SQRT_ONE_HALF, 0.0),
            Vec3(SQRT_ONE_HALF, 0.0, SQRT_ONE_HALF),
            Vec3(-SQRT_ONE_HALF, 0.0, SQRT_ONE_HALF),
            Vec3(SQRT_ONE_HALF, 0.0, -SQRT_ONE_HALF),
            Vec3(-SQRT_ONE_HALF, 0.0, -SQRT_ONE_HALF),
            Vec3(0.0, SQRT_ONE_HALF, SQRT_ONE_HALF),
            Vec3(0.0, -SQRT_ONE_HALF, SQRT_ONE_HALF),
            Vec3(0.0, SQRT_ONE_HALF, -SQRT_ONE_HALF),
            Vec3(0.0, -SQRT_ONE_HALF, -SQRT_ONE_HALF),
        ]
        dirs.extend(diagonals)

        return dirs


@dataclass
class PropagationSettings:
    """Settings for propagation calculation."""

    enabled: bool = True
    """Whether propagation is enabled."""

    include_reflections: bool = True
    """Calculate reflection paths."""

    include_diffraction: bool = True
    """Calculate diffraction paths."""

    max_reflection_order: int = MAX_REFLECTION_ORDER
    """Maximum reflection bounces."""

    max_diffraction_paths: int = MAX_DIFFRACTION_PATHS
    """Maximum diffraction paths."""

    max_distance: float = MAX_PROPAGATION_DISTANCE
    """Maximum propagation distance."""

    update_rate: float = PROPAGATION_UPDATE_RATE
    """Update rate in Hz."""


class PropagationCache:
    """Cache for propagation results to avoid recalculation."""

    def __init__(self, max_entries: int = 64) -> None:
        """Initialize the cache.

        Args:
            max_entries: Maximum cached entries.
        """
        self._max_entries = max(1, max_entries)
        self._cache: Dict[int, Tuple[float, PropagationResult]] = {}
        self._position_tolerance = 0.5  # meters

    def get(
        self,
        source_id: int,
        source_pos: Vec3,
        listener_pos: Vec3
    ) -> Optional[PropagationResult]:
        """Get cached result if available and still valid.

        Args:
            source_id: Source identifier.
            source_pos: Current source position.
            listener_pos: Current listener position.

        Returns:
            Cached PropagationResult or None.
        """
        entry = self._cache.get(source_id)
        if entry is None:
            return None

        # Check if positions have changed significantly
        timestamp, result = entry
        if len(result.paths) == 0:
            return None

        # Use first path's source point as cached position
        cached_source = result.paths[0].points[0] if result.paths[0].points else None
        if cached_source is None:
            return None

        dist = (source_pos - cached_source).length()
        if dist > self._position_tolerance:
            return None

        return result

    def store(
        self,
        source_id: int,
        result: PropagationResult,
        timestamp: float
    ) -> None:
        """Store a propagation result in cache.

        Args:
            source_id: Source identifier.
            result: Propagation result to cache.
            timestamp: Current time for cache invalidation.
        """
        if len(self._cache) >= self._max_entries:
            # Remove oldest entry
            oldest_id = min(self._cache.keys(), key=lambda k: self._cache[k][0])
            del self._cache[oldest_id]

        self._cache[source_id] = (timestamp, result)

    def invalidate(self, source_id: int) -> None:
        """Invalidate cache for a source."""
        self._cache.pop(source_id, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
