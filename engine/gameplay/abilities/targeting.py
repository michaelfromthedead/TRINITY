"""
Targeting System.

Provides targeting modes for abilities including Self, Actor, Point, Area/AOE,
and Confirmation targeting. Handles target validation, area calculations, and
target filtering.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from engine.gameplay.abilities.constants import (
    DEFAULT_AOE_MAX_RADIUS,
    DEFAULT_AOE_MIN_RADIUS,
    DEFAULT_AOE_RADIUS,
    DEFAULT_CONE_ANGLE,
    DEFAULT_LINE_WIDTH,
    DEFAULT_MAX_RANGE,
    DEFAULT_MELEE_RANGE,
    DEFAULT_MIN_RANGE,
    EPSILON,
    AreaShape,
    TargetingMode,
)
from engine.gameplay.abilities.tags import GameplayTag, GameplayTagContainer

T = TypeVar("T")


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================


@dataclass(frozen=True, slots=True)
class Vector3:
    """Simple 3D vector for positions and directions."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vector3:
        return self.__mul__(scalar)

    def __neg__(self) -> Vector3:
        return Vector3(-self.x, -self.y, -self.z)

    @property
    def magnitude(self) -> float:
        """Get the length of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @property
    def magnitude_squared(self) -> float:
        """Get the squared length (faster than magnitude)."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    @property
    def normalized(self) -> Vector3:
        """Get a normalized copy of this vector."""
        mag = self.magnitude
        if mag < EPSILON:
            return Vector3(0, 0, 0)
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

    def dot(self, other: Vector3) -> float:
        """Dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3) -> Vector3:
        """Cross product with another vector."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def distance_to(self, other: Vector3) -> float:
        """Distance to another point."""
        return (self - other).magnitude

    def distance_squared_to(self, other: Vector3) -> float:
        """Squared distance to another point (faster)."""
        return (self - other).magnitude_squared

    def angle_to(self, other: Vector3) -> float:
        """Angle in radians to another vector."""
        dot = self.dot(other)
        mags = self.magnitude * other.magnitude
        if mags < EPSILON:
            return 0.0
        cos_angle = max(-1.0, min(1.0, dot / mags))
        return math.acos(cos_angle)

    @staticmethod
    def zero() -> Vector3:
        """Return zero vector."""
        return Vector3(0, 0, 0)

    @staticmethod
    def up() -> Vector3:
        """Return up vector (positive Y)."""
        return Vector3(0, 1, 0)

    @staticmethod
    def forward() -> Vector3:
        """Return forward vector (positive Z)."""
        return Vector3(0, 0, 1)


class Targetable(Protocol):
    """Protocol for objects that can be targeted."""

    @property
    def position(self) -> Vector3:
        """Get world position."""
        ...

    @property
    def tags(self) -> GameplayTagContainer:
        """Get gameplay tags."""
        ...

    @property
    def is_valid(self) -> bool:
        """Check if still valid (alive, not destroyed)."""
        ...


# =============================================================================
# TARGET DATA
# =============================================================================


@dataclass
class TargetData:
    """Data about a targeting selection."""

    mode: TargetingMode = TargetingMode.SELF
    targets: List[Any] = field(default_factory=list)  # Targeted actors
    point: Optional[Vector3] = None  # World position
    direction: Optional[Vector3] = None  # Targeting direction
    confirmed: bool = False
    cancelled: bool = False
    distance: float = 0.0  # Distance to target/point
    hit_location: Optional[Vector3] = None  # Precise hit location

    @property
    def has_targets(self) -> bool:
        """Check if any targets are selected."""
        return len(self.targets) > 0

    @property
    def primary_target(self) -> Optional[Any]:
        """Get the primary (first) target."""
        return self.targets[0] if self.targets else None

    @property
    def target_count(self) -> int:
        """Get number of targets."""
        return len(self.targets)


# =============================================================================
# TARGET FILTER
# =============================================================================


@dataclass
class TargetFilter:
    """
    Filter for validating potential targets.

    Supports tag-based filtering, team filtering, and custom predicates.
    """

    require_tags: Set[GameplayTag | str] = field(default_factory=set)
    exclude_tags: Set[GameplayTag | str] = field(default_factory=set)
    require_any_tags: Set[GameplayTag | str] = field(default_factory=set)
    allow_self: bool = False
    allow_dead: bool = False
    allow_friendly: bool = True
    allow_hostile: bool = True
    allow_neutral: bool = True
    max_targets: int = 0  # 0 = unlimited
    custom_filter: Optional[Callable[[Any], bool]] = None

    def passes(
        self,
        target: Any,
        source: Optional[Any] = None,
    ) -> bool:
        """Check if a target passes the filter."""
        # Self check
        if target is source and not self.allow_self:
            return False

        # Dead check (if target has is_alive attribute)
        if hasattr(target, "is_alive") and not target.is_alive and not self.allow_dead:
            return False

        # Tag checks
        if hasattr(target, "tags"):
            target_tags = target.tags

            # Require all tags
            for tag in self.require_tags:
                tag_obj = tag if isinstance(tag, GameplayTag) else GameplayTag(tag)
                if not target_tags.has(tag_obj):
                    return False

            # Exclude tags
            for tag in self.exclude_tags:
                tag_obj = tag if isinstance(tag, GameplayTag) else GameplayTag(tag)
                if target_tags.has(tag_obj):
                    return False

            # Require any tags
            if self.require_any_tags:
                has_any = False
                for tag in self.require_any_tags:
                    tag_obj = tag if isinstance(tag, GameplayTag) else GameplayTag(tag)
                    if target_tags.has(tag_obj):
                        has_any = True
                        break
                if not has_any:
                    return False

        # Custom filter
        if self.custom_filter is not None:
            if not self.custom_filter(target):
                return False

        return True

    def filter_targets(
        self,
        targets: List[Any],
        source: Optional[Any] = None,
    ) -> List[Any]:
        """Filter a list of targets."""
        filtered = [t for t in targets if self.passes(t, source)]
        if self.max_targets > 0:
            filtered = filtered[: self.max_targets]
        return filtered


# =============================================================================
# BASE TARGETING SYSTEM
# =============================================================================


class TargetingSystem(ABC):
    """
    Base class for targeting systems.

    Handles target acquisition, validation, and area calculations.
    """

    def __init__(
        self,
        mode: TargetingMode,
        min_range: float = DEFAULT_MIN_RANGE,
        max_range: float = DEFAULT_MAX_RANGE,
        target_filter: Optional[TargetFilter] = None,
    ) -> None:
        self.mode = mode
        self.min_range = min_range
        self.max_range = max_range
        self.target_filter = target_filter or TargetFilter()

    @abstractmethod
    def acquire_targets(
        self,
        source: Any,
        aim_point: Vector3,
        aim_direction: Vector3,
        candidates: List[Any],
    ) -> TargetData:
        """Acquire targets based on the targeting mode."""
        pass

    def is_in_range(self, source_pos: Vector3, target_pos: Vector3) -> bool:
        """Check if a target position is in range."""
        distance = source_pos.distance_to(target_pos)
        return self.min_range <= distance <= self.max_range

    def validate_target(
        self,
        target: Any,
        source: Any,
        source_pos: Vector3,
    ) -> bool:
        """Validate a single target."""
        if not hasattr(target, "position"):
            return False

        if not hasattr(target, "is_valid") or not target.is_valid:
            return False

        if not self.is_in_range(source_pos, target.position):
            return False

        if not self.target_filter.passes(target, source):
            return False

        return True


# =============================================================================
# SELF TARGETING
# =============================================================================


class SelfTargeting(TargetingSystem):
    """
    Self-targeting system.

    Always targets the source entity.
    """

    def __init__(self) -> None:
        super().__init__(
            mode=TargetingMode.SELF,
            min_range=0.0,
            max_range=0.0,
            target_filter=TargetFilter(allow_self=True),
        )

    def acquire_targets(
        self,
        source: Any,
        aim_point: Vector3,
        aim_direction: Vector3,
        candidates: List[Any],
    ) -> TargetData:
        """Return self as the only target."""
        return TargetData(
            mode=self.mode,
            targets=[source],
            point=aim_point if hasattr(source, "position") else None,
            confirmed=True,
        )


# =============================================================================
# ACTOR TARGETING
# =============================================================================


class ActorTargeting(TargetingSystem):
    """
    Single actor targeting system.

    Targets the closest valid actor to the aim point.
    """

    def __init__(
        self,
        min_range: float = DEFAULT_MIN_RANGE,
        max_range: float = DEFAULT_MAX_RANGE,
        target_filter: Optional[TargetFilter] = None,
        require_line_of_sight: bool = True,
    ) -> None:
        super().__init__(
            mode=TargetingMode.ACTOR,
            min_range=min_range,
            max_range=max_range,
            target_filter=target_filter,
        )
        self.require_line_of_sight = require_line_of_sight

    def acquire_targets(
        self,
        source: Any,
        aim_point: Vector3,
        aim_direction: Vector3,
        candidates: List[Any],
    ) -> TargetData:
        """Find the best target actor."""
        source_pos = source.position if hasattr(source, "position") else Vector3.zero()

        valid_targets = []
        for candidate in candidates:
            if self.validate_target(candidate, source, source_pos):
                distance = source_pos.distance_to(candidate.position)
                valid_targets.append((candidate, distance))

        # Sort by distance to aim point
        valid_targets.sort(key=lambda x: aim_point.distance_to(x[0].position))

        if valid_targets:
            target, distance = valid_targets[0]
            return TargetData(
                mode=self.mode,
                targets=[target],
                point=target.position,
                distance=distance,
                confirmed=False,
            )

        return TargetData(mode=self.mode, cancelled=True)


# =============================================================================
# POINT TARGETING
# =============================================================================


class PointTargeting(TargetingSystem):
    """
    Point targeting system.

    Targets a world position rather than an actor.
    """

    def __init__(
        self,
        min_range: float = DEFAULT_MIN_RANGE,
        max_range: float = DEFAULT_MAX_RANGE,
    ) -> None:
        super().__init__(
            mode=TargetingMode.POINT,
            min_range=min_range,
            max_range=max_range,
        )

    def acquire_targets(
        self,
        source: Any,
        aim_point: Vector3,
        aim_direction: Vector3,
        candidates: List[Any],
    ) -> TargetData:
        """Target the aim point."""
        source_pos = source.position if hasattr(source, "position") else Vector3.zero()
        distance = source_pos.distance_to(aim_point)

        if not self.is_in_range(source_pos, aim_point):
            # Clamp to max range
            direction = (aim_point - source_pos).normalized
            aim_point = source_pos + direction * self.max_range
            distance = self.max_range

        return TargetData(
            mode=self.mode,
            targets=[],
            point=aim_point,
            direction=aim_direction,
            distance=distance,
            confirmed=False,
        )


# =============================================================================
# AREA TARGETING
# =============================================================================


class AreaTargeting(TargetingSystem):
    """
    Area of effect targeting system.

    Targets all entities within an area shape (circle, rectangle, cone, line).
    """

    def __init__(
        self,
        shape: AreaShape = AreaShape.CIRCLE,
        radius: float = DEFAULT_AOE_RADIUS,
        min_range: float = DEFAULT_MIN_RANGE,
        max_range: float = DEFAULT_MAX_RANGE,
        target_filter: Optional[TargetFilter] = None,
        # Shape-specific parameters
        cone_angle: float = DEFAULT_CONE_ANGLE,
        rectangle_width: float = DEFAULT_AOE_RADIUS,
        rectangle_height: float = DEFAULT_AOE_RADIUS,
        line_width: float = DEFAULT_LINE_WIDTH,
    ) -> None:
        super().__init__(
            mode=TargetingMode.AREA,
            min_range=min_range,
            max_range=max_range,
            target_filter=target_filter,
        )
        self.shape = shape
        self.radius = max(DEFAULT_AOE_MIN_RADIUS, min(DEFAULT_AOE_MAX_RADIUS, radius))
        self.cone_angle = cone_angle
        self.rectangle_width = rectangle_width
        self.rectangle_height = rectangle_height
        self.line_width = line_width

    def acquire_targets(
        self,
        source: Any,
        aim_point: Vector3,
        aim_direction: Vector3,
        candidates: List[Any],
    ) -> TargetData:
        """Find all targets in the area."""
        source_pos = source.position if hasattr(source, "position") else Vector3.zero()

        # Clamp aim point to range
        distance = source_pos.distance_to(aim_point)
        if distance > self.max_range:
            direction = (aim_point - source_pos).normalized
            aim_point = source_pos + direction * self.max_range
            distance = self.max_range

        # Find targets in area
        targets_in_area = []
        for candidate in candidates:
            if not self.target_filter.passes(candidate, source):
                continue

            if not hasattr(candidate, "position"):
                continue

            if self._is_in_area(aim_point, aim_direction, candidate.position):
                targets_in_area.append(candidate)

        return TargetData(
            mode=self.mode,
            targets=targets_in_area,
            point=aim_point,
            direction=aim_direction,
            distance=distance,
            confirmed=False,
        )

    def _is_in_area(
        self,
        center: Vector3,
        direction: Vector3,
        point: Vector3,
    ) -> bool:
        """Check if a point is within the area shape."""
        if self.shape == AreaShape.CIRCLE:
            return self._is_in_circle(center, point)
        elif self.shape == AreaShape.CONE:
            return self._is_in_cone(center, direction, point)
        elif self.shape == AreaShape.RECTANGLE:
            return self._is_in_rectangle(center, direction, point)
        elif self.shape == AreaShape.LINE:
            return self._is_in_line(center, direction, point)
        elif self.shape == AreaShape.CAPSULE:
            return self._is_in_capsule(center, direction, point)
        return False

    def _is_in_circle(self, center: Vector3, point: Vector3) -> bool:
        """Check if point is in circle."""
        return center.distance_squared_to(point) <= self.radius * self.radius

    def _is_in_cone(
        self,
        apex: Vector3,
        direction: Vector3,
        point: Vector3,
    ) -> bool:
        """Check if point is in cone."""
        to_point = point - apex
        distance = to_point.magnitude

        if distance > self.radius or distance < EPSILON:
            return False

        # Normalize direction
        dir_normalized = direction.normalized
        to_point_normalized = to_point.normalized

        # Check angle
        cos_angle = dir_normalized.dot(to_point_normalized)
        half_angle_rad = math.radians(self.cone_angle / 2)
        return cos_angle >= math.cos(half_angle_rad)

    def _is_in_rectangle(
        self,
        center: Vector3,
        direction: Vector3,
        point: Vector3,
    ) -> bool:
        """Check if point is in oriented rectangle."""
        # Transform point to local space
        to_point = point - center
        dir_normalized = direction.normalized

        # Simple 2D check (XZ plane)
        forward_dist = to_point.x * dir_normalized.x + to_point.z * dir_normalized.z
        right_dir = Vector3(dir_normalized.z, 0, -dir_normalized.x)
        right_dist = to_point.x * right_dir.x + to_point.z * right_dir.z

        return (
            abs(forward_dist) <= self.rectangle_height / 2
            and abs(right_dist) <= self.rectangle_width / 2
        )

    def _is_in_line(
        self,
        start: Vector3,
        direction: Vector3,
        point: Vector3,
    ) -> bool:
        """Check if point is within line area."""
        dir_normalized = direction.normalized
        end = start + dir_normalized * self.radius

        # Distance from point to line segment
        line_vec = end - start
        point_vec = point - start

        line_len_sq = line_vec.magnitude_squared
        if line_len_sq < EPSILON:
            return point.distance_to(start) <= self.line_width / 2

        # Project point onto line
        t = max(0, min(1, point_vec.dot(line_vec) / line_len_sq))
        projection = start + line_vec * t

        return point.distance_to(projection) <= self.line_width / 2

    def _is_in_capsule(
        self,
        start: Vector3,
        direction: Vector3,
        point: Vector3,
    ) -> bool:
        """Check if point is in capsule (line with spherical ends)."""
        dir_normalized = direction.normalized
        end = start + dir_normalized * self.radius

        # Capsule is a line segment with radius
        line_vec = end - start
        point_vec = point - start

        line_len_sq = line_vec.magnitude_squared
        if line_len_sq < EPSILON:
            return point.distance_to(start) <= self.line_width / 2

        t = max(0, min(1, point_vec.dot(line_vec) / line_len_sq))
        projection = start + line_vec * t

        return point.distance_to(projection) <= self.line_width / 2

    def get_area_bounds(self, center: Vector3, direction: Vector3) -> Tuple[Vector3, Vector3]:
        """Get axis-aligned bounding box for the area."""
        # Simple approximation using radius
        min_bound = Vector3(
            center.x - self.radius,
            center.y - self.radius,
            center.z - self.radius,
        )
        max_bound = Vector3(
            center.x + self.radius,
            center.y + self.radius,
            center.z + self.radius,
        )
        return min_bound, max_bound


# =============================================================================
# CONFIRMATION TARGETING
# =============================================================================


class ConfirmationTargeting(TargetingSystem):
    """
    Confirmation targeting system.

    Wraps another targeting system and requires explicit confirmation.
    """

    def __init__(
        self,
        inner_targeting: TargetingSystem,
    ) -> None:
        super().__init__(
            mode=TargetingMode.CONFIRMATION,
            min_range=inner_targeting.min_range,
            max_range=inner_targeting.max_range,
            target_filter=inner_targeting.target_filter,
        )
        self.inner_targeting = inner_targeting
        self._pending_data: Optional[TargetData] = None

    def acquire_targets(
        self,
        source: Any,
        aim_point: Vector3,
        aim_direction: Vector3,
        candidates: List[Any],
    ) -> TargetData:
        """Acquire targets but don't confirm."""
        data = self.inner_targeting.acquire_targets(
            source, aim_point, aim_direction, candidates
        )
        data.mode = TargetingMode.CONFIRMATION
        data.confirmed = False
        self._pending_data = data
        return data

    def confirm(self) -> Optional[TargetData]:
        """Confirm the current targeting selection."""
        if self._pending_data is not None:
            self._pending_data.confirmed = True
            data = self._pending_data
            self._pending_data = None
            return data
        return None

    def cancel(self) -> None:
        """Cancel the current targeting selection."""
        if self._pending_data is not None:
            self._pending_data.cancelled = True
            self._pending_data = None


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_self_targeting() -> SelfTargeting:
    """Create a self-targeting system."""
    return SelfTargeting()


def create_single_target(
    max_range: float = DEFAULT_MAX_RANGE,
    require_hostile: bool = True,
) -> ActorTargeting:
    """Create a single-target targeting system."""
    filter = TargetFilter(
        allow_self=False,
        allow_hostile=require_hostile,
        allow_friendly=not require_hostile,
    )
    return ActorTargeting(max_range=max_range, target_filter=filter)


def create_point_target(max_range: float = DEFAULT_MAX_RANGE) -> PointTargeting:
    """Create a point targeting system."""
    return PointTargeting(max_range=max_range)


def create_aoe(
    radius: float = DEFAULT_AOE_RADIUS,
    max_range: float = DEFAULT_MAX_RANGE,
    include_self: bool = False,
) -> AreaTargeting:
    """Create a circular AOE targeting system."""
    filter = TargetFilter(allow_self=include_self)
    return AreaTargeting(
        shape=AreaShape.CIRCLE,
        radius=radius,
        max_range=max_range,
        target_filter=filter,
    )


def create_cone(
    angle: float = DEFAULT_CONE_ANGLE,
    length: float = DEFAULT_AOE_RADIUS,
    max_range: float = DEFAULT_MIN_RANGE,  # Cone is placed at source
) -> AreaTargeting:
    """Create a cone AOE targeting system."""
    return AreaTargeting(
        shape=AreaShape.CONE,
        radius=length,
        max_range=max_range,
        cone_angle=angle,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "Vector3",
    "TargetData",
    "TargetFilter",
    # Systems
    "TargetingSystem",
    "SelfTargeting",
    "ActorTargeting",
    "PointTargeting",
    "AreaTargeting",
    "ConfirmationTargeting",
    # Factory functions
    "create_self_targeting",
    "create_single_target",
    "create_point_target",
    "create_aoe",
    "create_cone",
]
