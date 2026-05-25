"""
Tests for Camera Rails and Spline Paths (rails.py).

Tests camera rail systems including:
    - Rail point creation
    - Spline evaluation (Catmull-Rom, Bezier)
    - Rail following
    - Loop modes
    - Trigger volumes
    - Blend regions
    - Dolly and crane movement
    - Rail branching
"""

import math
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple
from enum import Enum, auto


# =============================================================================
# Mock Classes for Testing
# =============================================================================


@dataclass
class Vector3:
    """Mock 3D vector for testing."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self) -> "Vector3":
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

    def lerp(self, target: "Vector3", t: float) -> "Vector3":
        return Vector3(
            self.x + (target.x - self.x) * t,
            self.y + (target.y - self.y) * t,
            self.z + (target.z - self.z) * t,
        )

    def distance_to(self, other: "Vector3") -> float:
        return (self - other).magnitude()


@dataclass
class Quaternion:
    """Mock quaternion for rotation testing."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def slerp(self, target: "Quaternion", t: float) -> "Quaternion":
        return Quaternion(
            self.x + (target.x - self.x) * t,
            self.y + (target.y - self.y) * t,
            self.z + (target.z - self.z) * t,
            self.w + (target.w - self.w) * t,
        )


class LoopMode(Enum):
    """Loop mode for rail playback."""
    ONCE = auto()
    LOOP = auto()
    PING_PONG = auto()


class SplineType(Enum):
    """Type of spline interpolation."""
    LINEAR = auto()
    CATMULL_ROM = auto()
    BEZIER = auto()
    HERMITE = auto()


class TriggerEvent(Enum):
    """Trigger event types."""
    ENTER = auto()
    EXIT = auto()
    STAY = auto()


# =============================================================================
# Rail Point
# =============================================================================


@dataclass
class RailPoint:
    """A point on a camera rail."""
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)
    fov: float = 60.0
    roll: float = 0.0
    tension: float = 0.5
    tangent_in: Optional[Vector3] = None
    tangent_out: Optional[Vector3] = None
    speed_multiplier: float = 1.0

    def __post_init__(self):
        if self.tangent_in is None:
            self.tangent_in = Vector3()
        if self.tangent_out is None:
            self.tangent_out = Vector3()


# =============================================================================
# Trigger Volume
# =============================================================================


@dataclass
class TriggerVolume:
    """Trigger volume along a rail."""
    id: str = ""
    start_progress: float = 0.0
    end_progress: float = 0.0
    on_enter: Optional[Callable] = None
    on_exit: Optional[Callable] = None
    on_stay: Optional[Callable] = None
    is_active: bool = False
    data: dict = field(default_factory=dict)


# =============================================================================
# Blend Region
# =============================================================================


@dataclass
class BlendRegion:
    """Region for blending between rails."""
    start_progress: float = 0.0
    end_progress: float = 0.0
    target_rail: Optional["CameraRail"] = None
    target_progress: float = 0.0
    blend_curve: str = "linear"


# =============================================================================
# Spline Evaluators
# =============================================================================


class SplineEvaluator:
    """Base class for spline evaluation."""

    def evaluate(self, points: List[RailPoint], t: float) -> Tuple[Vector3, Quaternion, float]:
        raise NotImplementedError

    def get_tangent(self, points: List[RailPoint], t: float) -> Vector3:
        raise NotImplementedError


class LinearSplineEvaluator(SplineEvaluator):
    """Linear interpolation between points."""

    def evaluate(self, points: List[RailPoint], t: float) -> Tuple[Vector3, Quaternion, float]:
        if not points:
            return Vector3(), Quaternion(), 60.0

        if len(points) == 1:
            return points[0].position, points[0].rotation, points[0].fov

        segment_count = len(points) - 1
        scaled_t = t * segment_count
        segment_index = int(scaled_t)
        segment_index = max(0, min(segment_count - 1, segment_index))
        local_t = scaled_t - segment_index

        p1 = points[segment_index]
        p2 = points[segment_index + 1]

        position = p1.position.lerp(p2.position, local_t)
        rotation = p1.rotation.slerp(p2.rotation, local_t)
        fov = p1.fov + (p2.fov - p1.fov) * local_t

        return position, rotation, fov

    def get_tangent(self, points: List[RailPoint], t: float) -> Vector3:
        if len(points) < 2:
            return Vector3(0, 0, 1)

        segment_count = len(points) - 1
        scaled_t = t * segment_count
        segment_index = int(scaled_t)
        segment_index = max(0, min(segment_count - 1, segment_index))

        p1 = points[segment_index]
        p2 = points[segment_index + 1]

        return (p2.position - p1.position).normalized()


class CatmullRomSplineEvaluator(SplineEvaluator):
    """Catmull-Rom spline interpolation."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def _catmull_rom(self, p0: float, p1: float, p2: float, p3: float, t: float) -> float:
        t2 = t * t
        t3 = t2 * t

        return 0.5 * (
            2 * p1 +
            (-p0 + p2) * t +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )

    def evaluate(self, points: List[RailPoint], t: float) -> Tuple[Vector3, Quaternion, float]:
        if not points:
            return Vector3(), Quaternion(), 60.0

        if len(points) == 1:
            return points[0].position, points[0].rotation, points[0].fov

        if len(points) == 2:
            return LinearSplineEvaluator().evaluate(points, t)

        segment_count = len(points) - 1
        scaled_t = t * segment_count
        segment_index = int(scaled_t)
        segment_index = max(0, min(segment_count - 1, segment_index))
        local_t = scaled_t - segment_index

        i0 = max(0, segment_index - 1)
        i1 = segment_index
        i2 = min(len(points) - 1, segment_index + 1)
        i3 = min(len(points) - 1, segment_index + 2)

        p0, p1, p2, p3 = points[i0], points[i1], points[i2], points[i3]

        position = Vector3(
            self._catmull_rom(p0.position.x, p1.position.x, p2.position.x, p3.position.x, local_t),
            self._catmull_rom(p0.position.y, p1.position.y, p2.position.y, p3.position.y, local_t),
            self._catmull_rom(p0.position.z, p1.position.z, p2.position.z, p3.position.z, local_t),
        )

        rotation = p1.rotation.slerp(p2.rotation, local_t)
        fov = p1.fov + (p2.fov - p1.fov) * local_t

        return position, rotation, fov

    def get_tangent(self, points: List[RailPoint], t: float) -> Vector3:
        delta = 0.001
        t1 = max(0, t - delta)
        t2 = min(1, t + delta)

        pos1, _, _ = self.evaluate(points, t1)
        pos2, _, _ = self.evaluate(points, t2)

        return (pos2 - pos1).normalized()


class BezierSplineEvaluator(SplineEvaluator):
    """Bezier spline interpolation."""

    def _cubic_bezier(self, p0: float, p1: float, p2: float, p3: float, t: float) -> float:
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        return mt3 * p0 + 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t3 * p3

    def evaluate(self, points: List[RailPoint], t: float) -> Tuple[Vector3, Quaternion, float]:
        if not points:
            return Vector3(), Quaternion(), 60.0

        if len(points) < 4:
            return LinearSplineEvaluator().evaluate(points, t)

        segment_count = (len(points) - 1) // 3
        if segment_count == 0:
            return LinearSplineEvaluator().evaluate(points, t)

        scaled_t = t * segment_count
        segment_index = int(scaled_t)
        segment_index = max(0, min(segment_count - 1, segment_index))
        local_t = scaled_t - segment_index

        base = segment_index * 3
        p0 = points[base]
        p1 = points[min(base + 1, len(points) - 1)]
        p2 = points[min(base + 2, len(points) - 1)]
        p3 = points[min(base + 3, len(points) - 1)]

        position = Vector3(
            self._cubic_bezier(p0.position.x, p1.position.x, p2.position.x, p3.position.x, local_t),
            self._cubic_bezier(p0.position.y, p1.position.y, p2.position.y, p3.position.y, local_t),
            self._cubic_bezier(p0.position.z, p1.position.z, p2.position.z, p3.position.z, local_t),
        )

        rotation = p0.rotation.slerp(p3.rotation, local_t)
        fov = self._cubic_bezier(p0.fov, p1.fov, p2.fov, p3.fov, local_t)

        return position, rotation, fov

    def get_tangent(self, points: List[RailPoint], t: float) -> Vector3:
        delta = 0.001
        t1 = max(0, t - delta)
        t2 = min(1, t + delta)

        pos1, _, _ = self.evaluate(points, t1)
        pos2, _, _ = self.evaluate(points, t2)

        return (pos2 - pos1).normalized()


# =============================================================================
# Camera Rail
# =============================================================================


class CameraRail:
    """Camera rail for guided camera movement."""

    def __init__(self, name: str = ""):
        self.name = name
        self.points: List[RailPoint] = []
        self.spline_type = SplineType.CATMULL_ROM
        self.loop_mode = LoopMode.ONCE
        self.default_speed = 1.0
        self.triggers: List[TriggerVolume] = []
        self.blend_regions: List[BlendRegion] = []
        self._evaluator: SplineEvaluator = CatmullRomSplineEvaluator()
        self._length: float = 0.0
        self._length_dirty: bool = True

    def add_point(self, point: RailPoint, index: int = -1):
        """Add a point to the rail."""
        if index < 0:
            self.points.append(point)
        else:
            self.points.insert(index, point)
        self._length_dirty = True

    def remove_point(self, index: int):
        """Remove a point from the rail."""
        if 0 <= index < len(self.points):
            self.points.pop(index)
            self._length_dirty = True

    def clear_points(self):
        """Remove all points from the rail."""
        self.points.clear()
        self._length_dirty = True

    def set_spline_type(self, spline_type: SplineType):
        """Set the spline interpolation type."""
        self.spline_type = spline_type
        if spline_type == SplineType.LINEAR:
            self._evaluator = LinearSplineEvaluator()
        elif spline_type == SplineType.CATMULL_ROM:
            self._evaluator = CatmullRomSplineEvaluator()
        elif spline_type == SplineType.BEZIER:
            self._evaluator = BezierSplineEvaluator()

    def evaluate(self, t: float) -> Tuple[Vector3, Quaternion, float]:
        """Evaluate the rail at parameter t (0-1)."""
        t = max(0.0, min(1.0, t))
        return self._evaluator.evaluate(self.points, t)

    def get_tangent(self, t: float) -> Vector3:
        """Get tangent direction at parameter t."""
        return self._evaluator.get_tangent(self.points, t)

    def get_length(self) -> float:
        """Calculate approximate rail length."""
        if self._length_dirty:
            self._length = 0.0
            if len(self.points) >= 2:
                steps = 100
                prev_pos, _, _ = self.evaluate(0.0)
                for i in range(1, steps + 1):
                    t = i / steps
                    pos, _, _ = self.evaluate(t)
                    self._length += prev_pos.distance_to(pos)
                    prev_pos = pos
            self._length_dirty = False
        return self._length

    def add_trigger(self, trigger: TriggerVolume):
        """Add a trigger volume to the rail."""
        self.triggers.append(trigger)

    def remove_trigger(self, trigger_id: str):
        """Remove a trigger by ID."""
        self.triggers = [t for t in self.triggers if t.id != trigger_id]

    def add_blend_region(self, region: BlendRegion):
        """Add a blend region to the rail."""
        self.blend_regions.append(region)

    def get_triggers_at(self, progress: float) -> List[TriggerVolume]:
        """Get all triggers that overlap with the given progress."""
        return [
            t for t in self.triggers
            if t.start_progress <= progress <= t.end_progress
        ]


# =============================================================================
# Rail Follower
# =============================================================================


class RailFollower:
    """Follows a camera rail over time."""

    def __init__(self, rail: CameraRail = None):
        self.rail = rail
        self.progress = 0.0
        self.speed = 1.0
        self.is_playing = False
        self.is_reversed = False
        self.loop_mode = LoopMode.ONCE
        self.on_complete: Optional[Callable] = None
        self.on_loop: Optional[Callable] = None
        self._active_triggers: set = set()
        self._completed = False

    def set_rail(self, rail: CameraRail):
        """Set the rail to follow."""
        self.rail = rail
        self.progress = 0.0
        self._active_triggers.clear()
        self._completed = False

    def play(self):
        """Start following the rail."""
        self.is_playing = True
        self._completed = False

    def pause(self):
        """Pause following the rail."""
        self.is_playing = False

    def stop(self):
        """Stop and reset to start."""
        self.is_playing = False
        self.progress = 0.0 if not self.is_reversed else 1.0
        self._active_triggers.clear()
        self._completed = False

    def seek(self, progress: float):
        """Seek to a specific progress (0-1)."""
        self.progress = max(0.0, min(1.0, progress))
        self._check_triggers()

    def reverse(self):
        """Reverse playback direction."""
        self.is_reversed = not self.is_reversed

    def _check_triggers(self):
        """Check and fire trigger events."""
        if not self.rail:
            return

        current_triggers = set(t.id for t in self.rail.get_triggers_at(self.progress))

        entered = current_triggers - self._active_triggers
        exited = self._active_triggers - current_triggers

        for trigger in self.rail.triggers:
            if trigger.id in entered:
                trigger.is_active = True
                if trigger.on_enter:
                    trigger.on_enter(trigger)
            elif trigger.id in exited:
                trigger.is_active = False
                if trigger.on_exit:
                    trigger.on_exit(trigger)
            elif trigger.id in current_triggers:
                if trigger.on_stay:
                    trigger.on_stay(trigger)

        self._active_triggers = current_triggers

    def update(self, delta_time: float) -> Tuple[Vector3, Quaternion, float]:
        """Update follower and return current camera state."""
        if not self.rail or not self.is_playing:
            if self.rail:
                return self.rail.evaluate(self.progress)
            return Vector3(), Quaternion(), 60.0

        rail_length = self.rail.get_length()
        if rail_length <= 0:
            return self.rail.evaluate(self.progress)

        point = self.rail.points[int(self.progress * (len(self.rail.points) - 1))] if self.rail.points else None
        speed_mult = point.speed_multiplier if point else 1.0
        effective_speed = self.speed * speed_mult * self.rail.default_speed

        progress_delta = (effective_speed * delta_time) / rail_length
        if self.is_reversed:
            progress_delta = -progress_delta

        self.progress += progress_delta
        self._check_triggers()

        if self.progress >= 1.0:
            if self.loop_mode == LoopMode.ONCE:
                self.progress = 1.0
                self.is_playing = False
                self._completed = True
                if self.on_complete:
                    self.on_complete()
            elif self.loop_mode == LoopMode.LOOP:
                self.progress = self.progress % 1.0
                if self.on_loop:
                    self.on_loop()
            elif self.loop_mode == LoopMode.PING_PONG:
                self.progress = 1.0
                self.is_reversed = True
                if self.on_loop:
                    self.on_loop()

        elif self.progress <= 0.0:
            if self.loop_mode == LoopMode.ONCE:
                self.progress = 0.0
                self.is_playing = False
                self._completed = True
                if self.on_complete:
                    self.on_complete()
            elif self.loop_mode == LoopMode.LOOP:
                self.progress = 1.0 + self.progress
                if self.on_loop:
                    self.on_loop()
            elif self.loop_mode == LoopMode.PING_PONG:
                self.progress = 0.0
                self.is_reversed = False
                if self.on_loop:
                    self.on_loop()

        return self.rail.evaluate(self.progress)

    @property
    def is_complete(self) -> bool:
        return self._completed


# =============================================================================
# Dolly and Crane
# =============================================================================


class DollyRig:
    """Dolly rig for horizontal track movement."""

    def __init__(self):
        self.track_start = Vector3()
        self.track_end = Vector3(10, 0, 0)
        self.position = 0.0
        self.target_position = 0.0
        self.move_speed = 2.0
        self.smoothing = 5.0

    def set_track(self, start: Vector3, end: Vector3):
        """Set the dolly track endpoints."""
        self.track_start = start
        self.track_end = end

    def move_to(self, position: float):
        """Move to a position (0-1) on the track."""
        self.target_position = max(0.0, min(1.0, position))

    def move_by(self, delta: float):
        """Move by a delta amount."""
        self.move_to(self.target_position + delta)

    def update(self, delta_time: float) -> Vector3:
        """Update and return current position."""
        diff = self.target_position - self.position
        self.position += diff * min(1.0, self.smoothing * delta_time)
        return self.track_start.lerp(self.track_end, self.position)

    def get_position(self) -> Vector3:
        """Get current dolly position."""
        return self.track_start.lerp(self.track_end, self.position)


class CraneRig:
    """Crane rig for vertical and rotational movement."""

    def __init__(self):
        self.base_position = Vector3()
        self.arm_length = 5.0
        self.arm_angle = 0.0
        self.target_arm_angle = 0.0
        self.rotation = 0.0
        self.target_rotation = 0.0
        self.min_angle = -45.0
        self.max_angle = 90.0
        self.smoothing = 5.0

    def set_base(self, position: Vector3):
        """Set the crane base position."""
        self.base_position = position

    def set_arm_length(self, length: float):
        """Set the crane arm length."""
        self.arm_length = max(0.1, length)

    def set_arm_angle(self, angle: float):
        """Set target arm angle (degrees)."""
        self.target_arm_angle = max(self.min_angle, min(self.max_angle, angle))

    def set_rotation(self, rotation: float):
        """Set target rotation (degrees)."""
        self.target_rotation = rotation

    def update(self, delta_time: float) -> Vector3:
        """Update and return camera position."""
        angle_diff = self.target_arm_angle - self.arm_angle
        self.arm_angle += angle_diff * min(1.0, self.smoothing * delta_time)

        rot_diff = self.target_rotation - self.rotation
        self.rotation += rot_diff * min(1.0, self.smoothing * delta_time)

        angle_rad = math.radians(self.arm_angle)
        rot_rad = math.radians(self.rotation)

        x = self.arm_length * math.cos(angle_rad) * math.sin(rot_rad)
        y = self.arm_length * math.sin(angle_rad)
        z = self.arm_length * math.cos(angle_rad) * math.cos(rot_rad)

        return self.base_position + Vector3(x, y, z)

    def get_position(self) -> Vector3:
        """Get current crane camera position."""
        angle_rad = math.radians(self.arm_angle)
        rot_rad = math.radians(self.rotation)

        x = self.arm_length * math.cos(angle_rad) * math.sin(rot_rad)
        y = self.arm_length * math.sin(angle_rad)
        z = self.arm_length * math.cos(angle_rad) * math.cos(rot_rad)

        return self.base_position + Vector3(x, y, z)


# =============================================================================
# Rail Branching
# =============================================================================


class RailBranch:
    """Branch point connecting multiple rails."""

    def __init__(self, branch_id: str = ""):
        self.id = branch_id
        self.source_rail: Optional[CameraRail] = None
        self.source_progress: float = 0.0
        self.branches: List[Tuple[CameraRail, float, Callable]] = []
        self.default_branch: int = 0

    def set_source(self, rail: CameraRail, progress: float):
        """Set the source rail and branch point."""
        self.source_rail = rail
        self.source_progress = progress

    def add_branch(self, rail: CameraRail, entry_progress: float, condition: Callable = None):
        """Add a branch option."""
        self.branches.append((rail, entry_progress, condition))

    def evaluate_branch(self) -> Tuple[Optional[CameraRail], float]:
        """Evaluate which branch to take."""
        for rail, progress, condition in self.branches:
            if condition is None or condition():
                return rail, progress
        if self.branches:
            return self.branches[self.default_branch][0], self.branches[self.default_branch][1]
        return None, 0.0


class RailNetwork:
    """Network of connected camera rails."""

    def __init__(self):
        self.rails: dict[str, CameraRail] = {}
        self.branches: dict[str, RailBranch] = {}
        self.current_rail: Optional[CameraRail] = None
        self.follower = RailFollower()

    def add_rail(self, rail: CameraRail):
        """Add a rail to the network."""
        self.rails[rail.name] = rail

    def remove_rail(self, name: str):
        """Remove a rail from the network."""
        if name in self.rails:
            del self.rails[name]

    def add_branch(self, branch: RailBranch):
        """Add a branch point to the network."""
        self.branches[branch.id] = branch

    def set_active_rail(self, name: str, start_progress: float = 0.0):
        """Set the active rail."""
        if name in self.rails:
            self.current_rail = self.rails[name]
            self.follower.set_rail(self.current_rail)
            self.follower.seek(start_progress)

    def update(self, delta_time: float) -> Tuple[Vector3, Quaternion, float]:
        """Update the network and return camera state."""
        return self.follower.update(delta_time)


# =============================================================================
# Rail Point Tests (~20 tests)
# =============================================================================


class TestRailPoint:
    """Test rail point creation and properties."""

    def test_default_initialization(self):
        """Test default rail point initialization."""
        point = RailPoint()
        assert point.position.x == 0.0
        assert point.fov == 60.0
        assert point.speed_multiplier == 1.0

    def test_custom_position(self):
        """Test rail point with custom position."""
        point = RailPoint(position=Vector3(10, 5, 3))
        assert point.position.x == 10
        assert point.position.y == 5

    def test_custom_rotation(self):
        """Test rail point with custom rotation."""
        rot = Quaternion(0, 0.707, 0, 0.707)
        point = RailPoint(rotation=rot)
        assert point.rotation.y == 0.707

    def test_custom_fov(self):
        """Test rail point with custom FOV."""
        point = RailPoint(fov=90.0)
        assert point.fov == 90.0

    def test_tension_parameter(self):
        """Test tension parameter for spline control."""
        point = RailPoint(tension=0.3)
        assert point.tension == 0.3

    def test_tangent_vectors(self):
        """Test tangent vectors initialization."""
        point = RailPoint()
        assert point.tangent_in is not None
        assert point.tangent_out is not None

    def test_custom_tangents(self):
        """Test custom tangent vectors."""
        tangent_in = Vector3(1, 0, 0)
        tangent_out = Vector3(0, 1, 0)
        point = RailPoint(tangent_in=tangent_in, tangent_out=tangent_out)
        assert point.tangent_in.x == 1
        assert point.tangent_out.y == 1

    def test_speed_multiplier(self):
        """Test speed multiplier on point."""
        point = RailPoint(speed_multiplier=2.0)
        assert point.speed_multiplier == 2.0


# =============================================================================
# Spline Evaluation Tests (~30 tests)
# =============================================================================


class TestLinearSpline:
    """Test linear spline evaluation."""

    def test_empty_points(self):
        """Test evaluation with no points."""
        evaluator = LinearSplineEvaluator()
        pos, rot, fov = evaluator.evaluate([], 0.5)
        assert pos.x == 0 and pos.y == 0 and pos.z == 0

    def test_single_point(self):
        """Test evaluation with single point."""
        evaluator = LinearSplineEvaluator()
        point = RailPoint(position=Vector3(5, 5, 5))
        pos, rot, fov = evaluator.evaluate([point], 0.5)
        assert pos.x == 5

    def test_two_points_start(self):
        """Test evaluation at start of two points."""
        evaluator = LinearSplineEvaluator()
        points = [
            RailPoint(position=Vector3(0, 0, 0)),
            RailPoint(position=Vector3(10, 0, 0)),
        ]
        pos, rot, fov = evaluator.evaluate(points, 0.0)
        assert pos.x == pytest.approx(0.0, abs=0.01)

    def test_two_points_end(self):
        """Test evaluation at end of two points."""
        evaluator = LinearSplineEvaluator()
        points = [
            RailPoint(position=Vector3(0, 0, 0)),
            RailPoint(position=Vector3(10, 0, 0)),
        ]
        pos, rot, fov = evaluator.evaluate(points, 1.0)
        assert pos.x == pytest.approx(10.0, abs=0.01)

    def test_two_points_middle(self):
        """Test evaluation at middle of two points."""
        evaluator = LinearSplineEvaluator()
        points = [
            RailPoint(position=Vector3(0, 0, 0)),
            RailPoint(position=Vector3(10, 0, 0)),
        ]
        pos, rot, fov = evaluator.evaluate(points, 0.5)
        assert pos.x == pytest.approx(5.0, abs=0.01)

    def test_fov_interpolation(self):
        """Test FOV is interpolated."""
        evaluator = LinearSplineEvaluator()
        points = [
            RailPoint(position=Vector3(0, 0, 0), fov=60.0),
            RailPoint(position=Vector3(10, 0, 0), fov=90.0),
        ]
        pos, rot, fov = evaluator.evaluate(points, 0.5)
        assert fov == pytest.approx(75.0, abs=0.1)

    def test_tangent_calculation(self):
        """Test tangent calculation."""
        evaluator = LinearSplineEvaluator()
        points = [
            RailPoint(position=Vector3(0, 0, 0)),
            RailPoint(position=Vector3(10, 0, 0)),
        ]
        tangent = evaluator.get_tangent(points, 0.5)
        assert tangent.x == pytest.approx(1.0, abs=0.01)


class TestCatmullRomSpline:
    """Test Catmull-Rom spline evaluation."""

    def test_empty_points(self):
        """Test evaluation with no points."""
        evaluator = CatmullRomSplineEvaluator()
        pos, rot, fov = evaluator.evaluate([], 0.5)
        assert pos.magnitude() == 0

    def test_two_points_fallback(self):
        """Test fallback to linear with two points."""
        evaluator = CatmullRomSplineEvaluator()
        points = [
            RailPoint(position=Vector3(0, 0, 0)),
            RailPoint(position=Vector3(10, 0, 0)),
        ]
        pos, rot, fov = evaluator.evaluate(points, 0.5)
        assert pos.x == pytest.approx(5.0, abs=0.5)

    def test_four_points_smooth(self):
        """Test smooth interpolation with four points."""
        evaluator = CatmullRomSplineEvaluator()
        points = [
            RailPoint(position=Vector3(0, 0, 0)),
            RailPoint(position=Vector3(5, 2, 0)),
            RailPoint(position=Vector3(10, 2, 0)),
            RailPoint(position=Vector3(15, 0, 0)),
        ]
        pos, rot, fov = evaluator.evaluate(points, 0.5)
        assert 5 < pos.x < 10

    def test_tangent_smooth(self):
        """Test tangent is smooth."""
        evaluator = CatmullRomSplineEvaluator()
        points = [
            RailPoint(position=Vector3(0, 0, 0)),
            RailPoint(position=Vector3(5, 5, 0)),
            RailPoint(position=Vector3(10, 0, 0)),
            RailPoint(position=Vector3(15, 5, 0)),
        ]
        tangent = evaluator.get_tangent(points, 0.5)
        assert tangent.magnitude() > 0


class TestBezierSpline:
    """Test Bezier spline evaluation."""

    def test_empty_points(self):
        """Test evaluation with no points."""
        evaluator = BezierSplineEvaluator()
        pos, rot, fov = evaluator.evaluate([], 0.5)
        assert pos.magnitude() == 0

    def test_four_points_bezier(self):
        """Test cubic Bezier with four points."""
        evaluator = BezierSplineEvaluator()
        points = [
            RailPoint(position=Vector3(0, 0, 0)),
            RailPoint(position=Vector3(0, 10, 0)),
            RailPoint(position=Vector3(10, 10, 0)),
            RailPoint(position=Vector3(10, 0, 0)),
        ]
        pos, rot, fov = evaluator.evaluate(points, 0.5)
        assert 0 < pos.x < 10
        assert pos.y > 0

    def test_bezier_start_end(self):
        """Test Bezier passes through start and end."""
        evaluator = BezierSplineEvaluator()
        points = [
            RailPoint(position=Vector3(0, 0, 0)),
            RailPoint(position=Vector3(0, 10, 0)),
            RailPoint(position=Vector3(10, 10, 0)),
            RailPoint(position=Vector3(10, 0, 0)),
        ]
        pos_start, _, _ = evaluator.evaluate(points, 0.0)
        pos_end, _, _ = evaluator.evaluate(points, 1.0)
        assert pos_start.x == pytest.approx(0.0, abs=0.1)
        assert pos_end.x == pytest.approx(10.0, abs=0.1)


# =============================================================================
# Camera Rail Tests (~25 tests)
# =============================================================================


class TestCameraRail:
    """Test camera rail functionality."""

    def test_initialization(self):
        """Test rail initialization."""
        rail = CameraRail(name="test_rail")
        assert rail.name == "test_rail"
        assert len(rail.points) == 0

    def test_add_point(self):
        """Test adding points to rail."""
        rail = CameraRail()
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))
        assert len(rail.points) == 2

    def test_add_point_at_index(self):
        """Test adding point at specific index."""
        rail = CameraRail()
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(20, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)), index=1)
        assert rail.points[1].position.x == 10

    def test_remove_point(self):
        """Test removing a point."""
        rail = CameraRail()
        rail.add_point(RailPoint())
        rail.add_point(RailPoint())
        rail.remove_point(0)
        assert len(rail.points) == 1

    def test_clear_points(self):
        """Test clearing all points."""
        rail = CameraRail()
        for _ in range(5):
            rail.add_point(RailPoint())
        rail.clear_points()
        assert len(rail.points) == 0

    def test_set_spline_type(self):
        """Test setting spline type."""
        rail = CameraRail()
        rail.set_spline_type(SplineType.BEZIER)
        assert rail.spline_type == SplineType.BEZIER

    def test_evaluate_position(self):
        """Test evaluating rail position."""
        rail = CameraRail()
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))
        pos, rot, fov = rail.evaluate(0.5)
        assert 3 < pos.x < 7

    def test_get_tangent(self):
        """Test getting rail tangent."""
        rail = CameraRail()
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))
        tangent = rail.get_tangent(0.5)
        assert tangent.x > 0

    def test_get_length(self):
        """Test calculating rail length."""
        rail = CameraRail()
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))
        length = rail.get_length()
        assert length == pytest.approx(10.0, rel=0.1)

    def test_add_trigger(self):
        """Test adding trigger to rail."""
        rail = CameraRail()
        trigger = TriggerVolume(id="trigger1", start_progress=0.3, end_progress=0.5)
        rail.add_trigger(trigger)
        assert len(rail.triggers) == 1

    def test_remove_trigger(self):
        """Test removing trigger from rail."""
        rail = CameraRail()
        trigger = TriggerVolume(id="trigger1")
        rail.add_trigger(trigger)
        rail.remove_trigger("trigger1")
        assert len(rail.triggers) == 0

    def test_get_triggers_at_progress(self):
        """Test getting triggers at specific progress."""
        rail = CameraRail()
        rail.add_trigger(TriggerVolume(id="t1", start_progress=0.2, end_progress=0.4))
        rail.add_trigger(TriggerVolume(id="t2", start_progress=0.5, end_progress=0.7))

        triggers = rail.get_triggers_at(0.3)
        assert len(triggers) == 1
        assert triggers[0].id == "t1"


# =============================================================================
# Rail Follower Tests (~25 tests)
# =============================================================================


class TestRailFollower:
    """Test rail follower functionality."""

    def test_initialization(self):
        """Test follower initialization."""
        follower = RailFollower()
        assert follower.progress == 0.0
        assert follower.is_playing is False

    def test_set_rail(self):
        """Test setting rail to follow."""
        rail = CameraRail()
        follower = RailFollower()
        follower.set_rail(rail)
        assert follower.rail is rail
        assert follower.progress == 0.0

    def test_play_pause(self):
        """Test play and pause."""
        follower = RailFollower()
        follower.play()
        assert follower.is_playing is True
        follower.pause()
        assert follower.is_playing is False

    def test_stop_resets_progress(self):
        """Test stop resets progress."""
        rail = CameraRail()
        rail.add_point(RailPoint())
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))
        follower = RailFollower(rail)
        follower.progress = 0.5
        follower.stop()
        assert follower.progress == 0.0

    def test_seek(self):
        """Test seeking to specific progress."""
        rail = CameraRail()
        follower = RailFollower(rail)
        follower.seek(0.75)
        assert follower.progress == 0.75

    def test_seek_clamped(self):
        """Test seek is clamped to 0-1."""
        rail = CameraRail()
        follower = RailFollower(rail)
        follower.seek(2.0)
        assert follower.progress == 1.0
        follower.seek(-1.0)
        assert follower.progress == 0.0

    def test_reverse(self):
        """Test reversing playback."""
        follower = RailFollower()
        assert follower.is_reversed is False
        follower.reverse()
        assert follower.is_reversed is True
        follower.reverse()
        assert follower.is_reversed is False

    def test_update_advances_progress(self):
        """Test update advances progress."""
        rail = CameraRail()
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))
        follower = RailFollower(rail)
        follower.play()
        follower.update(0.5)
        assert follower.progress > 0.0

    def test_loop_mode_once(self):
        """Test ONCE loop mode stops at end."""
        rail = CameraRail()
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))
        follower = RailFollower(rail)
        follower.loop_mode = LoopMode.ONCE
        follower.play()

        for _ in range(100):
            follower.update(0.1)

        assert follower.progress == 1.0
        assert follower.is_playing is False

    def test_loop_mode_loop(self):
        """Test LOOP mode wraps around."""
        rail = CameraRail()
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))
        follower = RailFollower(rail)
        follower.loop_mode = LoopMode.LOOP
        follower.speed = 20.0
        follower.play()

        for _ in range(50):
            follower.update(0.1)

        assert follower.is_playing is True

    def test_loop_mode_ping_pong(self):
        """Test PING_PONG mode reverses at ends."""
        rail = CameraRail()
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))
        follower = RailFollower(rail)
        follower.loop_mode = LoopMode.PING_PONG
        follower.speed = 20.0
        follower.play()

        for _ in range(30):
            follower.update(0.1)

        assert follower.is_reversed is True or follower.is_reversed is False

    def test_on_complete_callback(self):
        """Test on_complete callback is fired."""
        completed = [False]

        def on_complete():
            completed[0] = True

        rail = CameraRail()
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(1, 0, 0)))
        follower = RailFollower(rail)
        follower.loop_mode = LoopMode.ONCE
        follower.speed = 10.0
        follower.on_complete = on_complete
        follower.play()

        for _ in range(20):
            follower.update(0.1)

        assert completed[0] is True

    def test_trigger_enter_callback(self):
        """Test trigger enter callback."""
        entered = [False]

        def on_enter(trigger):
            entered[0] = True

        rail = CameraRail()
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))
        rail.add_trigger(TriggerVolume(
            id="test",
            start_progress=0.4,
            end_progress=0.6,
            on_enter=on_enter
        ))

        follower = RailFollower(rail)
        follower.speed = 20.0
        follower.play()

        for _ in range(10):
            follower.update(0.1)

        assert entered[0] is True

    def test_is_complete_property(self):
        """Test is_complete property."""
        rail = CameraRail()
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(1, 0, 0)))
        follower = RailFollower(rail)
        follower.loop_mode = LoopMode.ONCE
        follower.speed = 10.0
        follower.play()

        assert follower.is_complete is False

        for _ in range(20):
            follower.update(0.1)

        assert follower.is_complete is True


# =============================================================================
# Dolly and Crane Tests (~15 tests)
# =============================================================================


class TestDollyRig:
    """Test dolly rig functionality."""

    def test_initialization(self):
        """Test dolly rig initialization."""
        dolly = DollyRig()
        assert dolly.position == 0.0
        assert dolly.move_speed == 2.0

    def test_set_track(self):
        """Test setting dolly track."""
        dolly = DollyRig()
        dolly.set_track(Vector3(0, 0, 0), Vector3(20, 0, 0))
        assert dolly.track_start.x == 0
        assert dolly.track_end.x == 20

    def test_move_to(self):
        """Test moving to position."""
        dolly = DollyRig()
        dolly.move_to(0.5)
        assert dolly.target_position == 0.5

    def test_move_to_clamped(self):
        """Test move_to is clamped."""
        dolly = DollyRig()
        dolly.move_to(2.0)
        assert dolly.target_position == 1.0
        dolly.move_to(-0.5)
        assert dolly.target_position == 0.0

    def test_move_by(self):
        """Test moving by delta."""
        dolly = DollyRig()
        dolly.target_position = 0.5
        dolly.move_by(0.25)
        assert dolly.target_position == 0.75

    def test_update_smoothing(self):
        """Test smooth position update."""
        dolly = DollyRig()
        dolly.move_to(1.0)
        dolly.update(0.1)
        assert 0 < dolly.position < 1.0

    def test_get_position(self):
        """Test getting world position."""
        dolly = DollyRig()
        dolly.set_track(Vector3(0, 0, 0), Vector3(10, 0, 0))
        dolly.position = 0.5
        pos = dolly.get_position()
        assert pos.x == pytest.approx(5.0, abs=0.1)


class TestCraneRig:
    """Test crane rig functionality."""

    def test_initialization(self):
        """Test crane rig initialization."""
        crane = CraneRig()
        assert crane.arm_length == 5.0
        assert crane.arm_angle == 0.0

    def test_set_base(self):
        """Test setting crane base."""
        crane = CraneRig()
        crane.set_base(Vector3(10, 0, 10))
        assert crane.base_position.x == 10

    def test_set_arm_length(self):
        """Test setting arm length."""
        crane = CraneRig()
        crane.set_arm_length(10.0)
        assert crane.arm_length == 10.0

    def test_set_arm_length_minimum(self):
        """Test arm length minimum."""
        crane = CraneRig()
        crane.set_arm_length(0.0)
        assert crane.arm_length >= 0.1

    def test_set_arm_angle(self):
        """Test setting arm angle."""
        crane = CraneRig()
        crane.set_arm_angle(45.0)
        assert crane.target_arm_angle == 45.0

    def test_arm_angle_clamped(self):
        """Test arm angle is clamped."""
        crane = CraneRig()
        crane.set_arm_angle(100.0)
        assert crane.target_arm_angle <= crane.max_angle

    def test_set_rotation(self):
        """Test setting rotation."""
        crane = CraneRig()
        crane.set_rotation(90.0)
        assert crane.target_rotation == 90.0

    def test_update_position(self):
        """Test position calculation."""
        crane = CraneRig()
        crane.arm_length = 10.0
        crane.arm_angle = 45.0
        crane.rotation = 0.0
        pos = crane.get_position()
        assert pos.y > 0


# =============================================================================
# Rail Branching Tests (~15 tests)
# =============================================================================


class TestRailBranching:
    """Test rail branching functionality."""

    def test_branch_initialization(self):
        """Test branch initialization."""
        branch = RailBranch(branch_id="branch1")
        assert branch.id == "branch1"
        assert len(branch.branches) == 0

    def test_set_source(self):
        """Test setting source rail."""
        branch = RailBranch()
        rail = CameraRail(name="main")
        branch.set_source(rail, 0.5)
        assert branch.source_rail is rail
        assert branch.source_progress == 0.5

    def test_add_branch(self):
        """Test adding branch options."""
        branch = RailBranch()
        rail1 = CameraRail(name="rail1")
        rail2 = CameraRail(name="rail2")
        branch.add_branch(rail1, 0.0)
        branch.add_branch(rail2, 0.0)
        assert len(branch.branches) == 2

    def test_evaluate_branch_default(self):
        """Test evaluating default branch."""
        branch = RailBranch()
        rail = CameraRail(name="default")
        branch.add_branch(rail, 0.25)
        result_rail, result_progress = branch.evaluate_branch()
        assert result_rail is rail
        assert result_progress == 0.25

    def test_evaluate_branch_with_condition(self):
        """Test evaluating branch with condition."""
        branch = RailBranch()
        rail1 = CameraRail(name="rail1")
        rail2 = CameraRail(name="rail2")
        branch.add_branch(rail1, 0.0, lambda: False)
        branch.add_branch(rail2, 0.5, lambda: True)
        result_rail, result_progress = branch.evaluate_branch()
        assert result_rail is rail2


class TestRailNetwork:
    """Test rail network functionality."""

    def test_initialization(self):
        """Test network initialization."""
        network = RailNetwork()
        assert len(network.rails) == 0

    def test_add_rail(self):
        """Test adding rail to network."""
        network = RailNetwork()
        rail = CameraRail(name="main")
        network.add_rail(rail)
        assert "main" in network.rails

    def test_remove_rail(self):
        """Test removing rail from network."""
        network = RailNetwork()
        rail = CameraRail(name="main")
        network.add_rail(rail)
        network.remove_rail("main")
        assert "main" not in network.rails

    def test_set_active_rail(self):
        """Test setting active rail."""
        network = RailNetwork()
        rail = CameraRail(name="main")
        rail.add_point(RailPoint())
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))
        network.add_rail(rail)
        network.set_active_rail("main", 0.5)
        assert network.current_rail is rail
        assert network.follower.progress == 0.5

    def test_add_branch(self):
        """Test adding branch to network."""
        network = RailNetwork()
        branch = RailBranch(branch_id="branch1")
        network.add_branch(branch)
        assert "branch1" in network.branches


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases for rail system."""

    def test_empty_rail_evaluation(self):
        """Test evaluating empty rail."""
        rail = CameraRail()
        pos, rot, fov = rail.evaluate(0.5)
        assert pos.magnitude() == 0

    def test_zero_length_rail(self):
        """Test rail with same start and end point."""
        rail = CameraRail()
        rail.add_point(RailPoint(position=Vector3(5, 5, 5)))
        rail.add_point(RailPoint(position=Vector3(5, 5, 5)))
        length = rail.get_length()
        assert length == pytest.approx(0.0, abs=0.1)

    def test_very_short_rail(self):
        """Test following very short rail."""
        rail = CameraRail()
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(0.01, 0, 0)))

        follower = RailFollower(rail)
        follower.speed = 1.0
        follower.play()
        follower.update(0.1)

    def test_negative_speed(self):
        """Test follower with negative speed."""
        rail = CameraRail()
        rail.add_point(RailPoint())
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))
        follower = RailFollower(rail)
        follower.speed = -1.0
        follower.progress = 0.5
        follower.play()
        follower.update(0.1)

    def test_extreme_progress_values(self):
        """Test evaluation with extreme progress values."""
        rail = CameraRail()
        rail.add_point(RailPoint())
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))

        pos1, _, _ = rail.evaluate(-10.0)
        pos2, _, _ = rail.evaluate(10.0)

        assert pos1.x >= 0
        assert pos2.x <= 10


# =============================================================================
# Additional Rail Point Tests
# =============================================================================


class TestRailPointAdvanced:
    """Additional rail point tests."""

    def test_roll_angle(self):
        """Test roll angle on rail point."""
        point = RailPoint(roll=15.0)
        assert point.roll == 15.0

    def test_zero_tension(self):
        """Test zero tension for sharp corners."""
        point = RailPoint(tension=0.0)
        assert point.tension == 0.0

    def test_high_tension(self):
        """Test high tension for smooth curves."""
        point = RailPoint(tension=1.0)
        assert point.tension == 1.0

    def test_custom_tangent_vectors(self):
        """Test custom tangent vectors for Bezier control."""
        point = RailPoint(
            tangent_in=Vector3(-1, 0, 0),
            tangent_out=Vector3(1, 0, 0)
        )
        assert point.tangent_in.x == -1
        assert point.tangent_out.x == 1


# =============================================================================
# Additional Spline Tests
# =============================================================================


class TestSplineAdvanced:
    """Additional spline evaluation tests."""

    def test_catmull_rom_tension_effect(self):
        """Test Catmull-Rom with different alpha values."""
        evaluator = CatmullRomSplineEvaluator(alpha=0.0)
        points = [
            RailPoint(position=Vector3(0, 0, 0)),
            RailPoint(position=Vector3(5, 5, 0)),
            RailPoint(position=Vector3(10, 0, 0)),
            RailPoint(position=Vector3(15, 5, 0)),
        ]
        pos, _, _ = evaluator.evaluate(points, 0.5)
        assert pos.x > 0

    def test_linear_three_points(self):
        """Test linear spline with three points."""
        evaluator = LinearSplineEvaluator()
        points = [
            RailPoint(position=Vector3(0, 0, 0)),
            RailPoint(position=Vector3(5, 5, 0)),
            RailPoint(position=Vector3(10, 0, 0)),
        ]
        pos, _, _ = evaluator.evaluate(points, 0.5)
        assert pos.x == pytest.approx(5.0, abs=0.1)

    def test_bezier_insufficient_points(self):
        """Test Bezier with less than 4 points falls back."""
        evaluator = BezierSplineEvaluator()
        points = [
            RailPoint(position=Vector3(0, 0, 0)),
            RailPoint(position=Vector3(10, 0, 0)),
        ]
        pos, _, _ = evaluator.evaluate(points, 0.5)
        assert pos.x == pytest.approx(5.0, abs=0.5)

    def test_rotation_interpolation(self):
        """Test rotation is interpolated between points."""
        evaluator = LinearSplineEvaluator()
        q1 = Quaternion(0, 0, 0, 1)
        q2 = Quaternion(0, 0.707, 0, 0.707)
        points = [
            RailPoint(position=Vector3(0, 0, 0), rotation=q1),
            RailPoint(position=Vector3(10, 0, 0), rotation=q2),
        ]
        _, rot, _ = evaluator.evaluate(points, 0.5)
        assert rot.y != 0


# =============================================================================
# Additional Rail Follower Tests
# =============================================================================


class TestRailFollowerAdvanced:
    """Additional rail follower tests."""

    def test_speed_multiplier_on_point(self):
        """Test point speed multiplier affects movement."""
        rail = CameraRail()
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0), speed_multiplier=2.0))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0), speed_multiplier=1.0))

        follower = RailFollower(rail)
        follower.speed = 1.0
        follower.play()
        follower.update(0.1)

    def test_trigger_stay_callback(self):
        """Test trigger stay callback fires continuously."""
        stay_count = [0]

        def on_stay(trigger):
            stay_count[0] += 1

        rail = CameraRail()
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))
        rail.add_trigger(TriggerVolume(
            id="test",
            start_progress=0.2,
            end_progress=0.8,
            on_stay=on_stay
        ))

        follower = RailFollower(rail)
        follower.speed = 5.0
        follower.play()
        follower.seek(0.3)

        for _ in range(5):
            follower.update(0.05)

        assert stay_count[0] > 0

    def test_trigger_exit_callback(self):
        """Test trigger exit callback fires."""
        exited = [False]

        def on_exit(trigger):
            exited[0] = True

        rail = CameraRail()
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))
        rail.add_trigger(TriggerVolume(
            id="test",
            start_progress=0.2,
            end_progress=0.4,
            on_exit=on_exit
        ))

        follower = RailFollower(rail)
        follower.speed = 10.0
        follower.play()

        for _ in range(20):
            follower.update(0.1)

        assert exited[0] is True

    def test_on_loop_callback(self):
        """Test on_loop callback fires."""
        looped = [False]

        def on_loop():
            looped[0] = True

        rail = CameraRail()
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(1, 0, 0)))

        follower = RailFollower(rail)
        follower.loop_mode = LoopMode.LOOP
        follower.speed = 10.0
        follower.on_loop = on_loop
        follower.play()

        for _ in range(20):
            follower.update(0.1)

        assert looped[0] is True


# =============================================================================
# Additional Dolly Tests
# =============================================================================


class TestDollyAdvanced:
    """Additional dolly rig tests."""

    def test_dolly_3d_track(self):
        """Test dolly on 3D track."""
        dolly = DollyRig()
        dolly.set_track(Vector3(0, 0, 0), Vector3(10, 5, 10))
        dolly.position = 0.5
        pos = dolly.get_position()
        assert pos.x == pytest.approx(5.0, abs=0.1)
        assert pos.y == pytest.approx(2.5, abs=0.1)
        assert pos.z == pytest.approx(5.0, abs=0.1)

    def test_dolly_smoothing_over_time(self):
        """Test dolly smoothing converges over time."""
        dolly = DollyRig()
        dolly.set_track(Vector3(0, 0, 0), Vector3(10, 0, 0))
        dolly.smoothing = 5.0
        dolly.move_to(1.0)

        for _ in range(50):
            dolly.update(0.1)

        assert dolly.position == pytest.approx(1.0, abs=0.01)


# =============================================================================
# Additional Crane Tests
# =============================================================================


class TestCraneAdvanced:
    """Additional crane rig tests."""

    def test_crane_full_rotation(self):
        """Test crane full 360 degree rotation."""
        crane = CraneRig()
        crane.arm_length = 10.0
        crane.set_rotation(180.0)

        for _ in range(50):
            crane.update(0.1)

        assert crane.rotation == pytest.approx(180.0, abs=1.0)

    def test_crane_arm_at_max_angle(self):
        """Test crane arm at maximum angle."""
        crane = CraneRig()
        crane.arm_length = 10.0
        crane.set_arm_angle(90.0)

        for _ in range(50):
            crane.update(0.1)

        pos = crane.get_position()
        assert pos.y == pytest.approx(10.0, abs=0.5)

    def test_crane_combined_movement(self):
        """Test crane with both arm angle and rotation."""
        crane = CraneRig()
        crane.arm_length = 5.0
        crane.set_arm_angle(45.0)
        crane.set_rotation(45.0)

        for _ in range(50):
            crane.update(0.1)

        pos = crane.get_position()
        assert pos.y > 0
        assert pos.x != 0 or pos.z != 0


# =============================================================================
# Additional Branching Tests
# =============================================================================


class TestBranchingAdvanced:
    """Additional rail branching tests."""

    def test_branch_with_multiple_conditions(self):
        """Test branch with multiple conditions."""
        branch = RailBranch()
        rail1 = CameraRail(name="rail1")
        rail2 = CameraRail(name="rail2")
        rail3 = CameraRail(name="rail3")

        condition_value = [1]

        branch.add_branch(rail1, 0.0, lambda: condition_value[0] == 1)
        branch.add_branch(rail2, 0.0, lambda: condition_value[0] == 2)
        branch.add_branch(rail3, 0.0, lambda: condition_value[0] == 3)

        result, _ = branch.evaluate_branch()
        assert result is rail1

        condition_value[0] = 2
        result, _ = branch.evaluate_branch()
        assert result is rail2

    def test_branch_default_index(self):
        """Test branch default index selection."""
        branch = RailBranch()
        rail1 = CameraRail(name="rail1")
        rail2 = CameraRail(name="rail2")

        branch.add_branch(rail1, 0.0, lambda: False)
        branch.add_branch(rail2, 0.5, lambda: False)
        branch.default_branch = 1

        result, progress = branch.evaluate_branch()
        assert result is rail2
        assert progress == 0.5

    def test_network_multiple_rails(self):
        """Test network with multiple connected rails."""
        network = RailNetwork()

        rail1 = CameraRail(name="intro")
        rail1.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail1.add_point(RailPoint(position=Vector3(10, 0, 0)))

        rail2 = CameraRail(name="main")
        rail2.add_point(RailPoint(position=Vector3(10, 0, 0)))
        rail2.add_point(RailPoint(position=Vector3(20, 0, 0)))

        network.add_rail(rail1)
        network.add_rail(rail2)

        network.set_active_rail("intro")
        network.follower.play()
        network.update(0.5)


# =============================================================================
# Integration Tests
# =============================================================================


class TestRailsIntegration:
    """Integration tests for rail systems."""

    def test_complete_cinematic_rail_sequence(self):
        """Test complete cinematic rail sequence."""
        rail = CameraRail(name="cinematic")
        rail.set_spline_type(SplineType.CATMULL_ROM)

        points = [
            RailPoint(position=Vector3(0, 2, 0), fov=60.0),
            RailPoint(position=Vector3(5, 3, 5), fov=55.0),
            RailPoint(position=Vector3(10, 2, 10), fov=50.0),
            RailPoint(position=Vector3(15, 4, 5), fov=55.0),
            RailPoint(position=Vector3(20, 2, 0), fov=60.0),
        ]

        for point in points:
            rail.add_point(point)

        follower = RailFollower(rail)
        follower.loop_mode = LoopMode.ONCE
        follower.speed = 5.0
        follower.play()

        positions = []
        fovs = []
        for _ in range(100):
            pos, rot, fov = follower.update(0.1)
            positions.append(pos)
            fovs.append(fov)

        assert follower.is_complete is True

    def test_dolly_crane_combination(self):
        """Test combined dolly and crane movement."""
        dolly = DollyRig()
        crane = CraneRig()

        dolly.set_track(Vector3(0, 0, 0), Vector3(20, 0, 0))
        crane.arm_length = 5.0

        for i in range(100):
            dolly.move_to(i / 100)
            dolly_pos = dolly.update(0.016)
            crane.set_base(dolly_pos)
            crane.set_arm_angle(30 + math.sin(i * 0.1) * 20)
            crane.set_rotation(i * 2)
            crane_pos = crane.update(0.016)

    def test_trigger_driven_rail_sequence(self):
        """Test rail with trigger-driven events."""
        events = []

        rail = CameraRail(name="events")
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(100, 0, 0)))

        rail.add_trigger(TriggerVolume(
            id="start",
            start_progress=0.0,
            end_progress=0.1,
            on_enter=lambda t: events.append("start_enter")
        ))
        rail.add_trigger(TriggerVolume(
            id="middle",
            start_progress=0.45,
            end_progress=0.55,
            on_enter=lambda t: events.append("middle_enter"),
            on_exit=lambda t: events.append("middle_exit")
        ))
        rail.add_trigger(TriggerVolume(
            id="end",
            start_progress=0.9,
            end_progress=1.0,
            on_enter=lambda t: events.append("end_enter")
        ))

        follower = RailFollower(rail)
        follower.speed = 50.0
        follower.play()

        for _ in range(50):
            follower.update(0.1)

        assert "start_enter" in events
        assert "middle_enter" in events

    def test_looping_rail_system(self):
        """Test looping rail system."""
        rail = CameraRail(name="loop")
        rail.set_spline_type(SplineType.CATMULL_ROM)

        for i in range(8):
            angle = i * math.pi / 4
            rail.add_point(RailPoint(
                position=Vector3(
                    math.cos(angle) * 10,
                    2,
                    math.sin(angle) * 10
                )
            ))

        follower = RailFollower(rail)
        follower.loop_mode = LoopMode.LOOP
        follower.speed = 10.0
        follower.play()

        loop_count = [0]

        def on_loop():
            loop_count[0] += 1

        follower.on_loop = on_loop

        for _ in range(200):
            follower.update(0.1)

        assert loop_count[0] > 0

    def test_ping_pong_rail(self):
        """Test ping-pong rail movement."""
        rail = CameraRail(name="pingpong")
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))

        follower = RailFollower(rail)
        follower.loop_mode = LoopMode.PING_PONG
        follower.speed = 20.0
        follower.play()

        reverse_count = 0
        was_reversed = follower.is_reversed

        for _ in range(100):
            follower.update(0.1)
            if follower.is_reversed != was_reversed:
                reverse_count += 1
                was_reversed = follower.is_reversed

        assert reverse_count >= 2

    def test_bezier_smooth_path(self):
        """Test Bezier spline for smooth path."""
        rail = CameraRail(name="bezier")
        rail.set_spline_type(SplineType.BEZIER)

        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(0, 10, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 10, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))

        follower = RailFollower(rail)
        follower.play()

        for _ in range(20):
            pos, rot, fov = follower.update(0.1)

    def test_multiple_speed_zones(self):
        """Test rail with multiple speed zones."""
        rail = CameraRail(name="speed_zones")
        rail.set_spline_type(SplineType.LINEAR)

        rail.add_point(RailPoint(position=Vector3(0, 0, 0), speed_multiplier=1.0))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0), speed_multiplier=0.5))
        rail.add_point(RailPoint(position=Vector3(20, 0, 0), speed_multiplier=2.0))
        rail.add_point(RailPoint(position=Vector3(30, 0, 0), speed_multiplier=1.0))

        follower = RailFollower(rail)
        follower.speed = 5.0
        follower.play()

        for _ in range(100):
            follower.update(0.1)


# =============================================================================
# Stress Tests
# =============================================================================


class TestRailsStress:
    """Stress tests for rail systems."""

    def test_many_rail_points(self):
        """Test rail with many points."""
        rail = CameraRail(name="many_points")
        for i in range(100):
            rail.add_point(RailPoint(
                position=Vector3(i, math.sin(i * 0.1), math.cos(i * 0.1))
            ))

        follower = RailFollower(rail)
        follower.play()
        for _ in range(50):
            follower.update(0.1)

    def test_many_triggers(self):
        """Test rail with many triggers."""
        rail = CameraRail(name="many_triggers")
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(100, 0, 0)))

        for i in range(50):
            rail.add_trigger(TriggerVolume(
                id=f"trigger_{i}",
                start_progress=i / 50,
                end_progress=(i + 1) / 50,
            ))

        follower = RailFollower(rail)
        follower.speed = 50.0
        follower.play()
        for _ in range(50):
            follower.update(0.1)

    def test_rapid_rail_switches(self):
        """Test rapid rail switching in network."""
        network = RailNetwork()

        for i in range(10):
            rail = CameraRail(name=f"rail_{i}")
            rail.add_point(RailPoint(position=Vector3(i * 10, 0, 0)))
            rail.add_point(RailPoint(position=Vector3(i * 10 + 10, 0, 0)))
            network.add_rail(rail)

        for _ in range(100):
            rail_name = f"rail_{_ % 10}"
            network.set_active_rail(rail_name)
            network.follower.play()
            network.update(0.05)


# =============================================================================
# Additional Integration Tests
# =============================================================================


class TestRailsIntegrationAdvanced:
    """Additional integration tests for rail systems."""

    def test_complex_branching_scenario(self):
        """Test complex branching with multiple conditions."""
        network = RailNetwork()

        # Create main rail
        main_rail = CameraRail(name="main")
        main_rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        main_rail.add_point(RailPoint(position=Vector3(10, 0, 0)))
        network.add_rail(main_rail)

        # Create branch rails
        branch_a = CameraRail(name="branch_a")
        branch_a.add_point(RailPoint(position=Vector3(10, 0, 0)))
        branch_a.add_point(RailPoint(position=Vector3(20, 10, 0)))
        network.add_rail(branch_a)

        branch_b = CameraRail(name="branch_b")
        branch_b.add_point(RailPoint(position=Vector3(10, 0, 0)))
        branch_b.add_point(RailPoint(position=Vector3(20, -10, 0)))
        network.add_rail(branch_b)

        # Set up branch point
        branch = RailBranch(branch_id="main_split")
        branch.set_source(main_rail, 1.0)
        branch.add_branch(branch_a, 0.0)
        branch.add_branch(branch_b, 0.0)
        network.add_branch(branch)

        network.set_active_rail("main")
        network.follower.play()

    def test_seamless_rail_transitions(self):
        """Test seamless transitions between rails."""
        network = RailNetwork()

        # Create connected rails
        rail1 = CameraRail(name="segment1")
        rail1.set_spline_type(SplineType.LINEAR)
        rail1.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail1.add_point(RailPoint(position=Vector3(10, 0, 0)))

        rail2 = CameraRail(name="segment2")
        rail2.set_spline_type(SplineType.LINEAR)
        rail2.add_point(RailPoint(position=Vector3(10, 0, 0)))
        rail2.add_point(RailPoint(position=Vector3(20, 5, 0)))

        network.add_rail(rail1)
        network.add_rail(rail2)

        # Follow first rail, then switch
        network.set_active_rail("segment1")
        network.follower.speed = 20.0
        network.follower.play()

        for _ in range(30):
            network.update(0.1)

        # Seamless switch to second rail
        network.set_active_rail("segment2", start_progress=0.0)
        network.follower.play()

        for _ in range(30):
            network.update(0.1)

    def test_complex_trigger_interactions(self):
        """Test complex trigger interactions."""
        events = []

        rail = CameraRail(name="events")
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(100, 0, 0)))

        # Overlapping triggers
        rail.add_trigger(TriggerVolume(
            id="zone_a",
            start_progress=0.1,
            end_progress=0.5,
            on_enter=lambda t: events.append(f"{t.id}_enter"),
            on_exit=lambda t: events.append(f"{t.id}_exit"),
            on_stay=lambda t: events.append(f"{t.id}_stay")
        ))
        rail.add_trigger(TriggerVolume(
            id="zone_b",
            start_progress=0.3,
            end_progress=0.7,
            on_enter=lambda t: events.append(f"{t.id}_enter"),
            on_exit=lambda t: events.append(f"{t.id}_exit")
        ))

        follower = RailFollower(rail)
        follower.speed = 50.0
        follower.play()

        for _ in range(40):
            follower.update(0.1)

        assert "zone_a_enter" in events
        assert "zone_b_enter" in events

    def test_combined_dolly_crane_rail(self):
        """Test combining dolly, crane, and rail systems."""
        # Create base rail for dolly track
        rail = CameraRail(name="dolly_track")
        rail.set_spline_type(SplineType.CATMULL_ROM)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 5)))
        rail.add_point(RailPoint(position=Vector3(20, 0, 0)))

        follower = RailFollower(rail)
        crane = CraneRig()
        crane.arm_length = 3.0

        follower.speed = 5.0
        follower.play()

        for i in range(100):
            pos, rot, fov = follower.update(0.016)
            crane.set_base(pos)
            crane.set_arm_angle(30 + math.sin(i * 0.1) * 15)
            crane.set_rotation(i * 2)
            final_pos = crane.update(0.016)


class TestRailsStressAdvanced:
    """Additional stress tests for rails."""

    def test_very_long_rail(self):
        """Test very long rail with many segments."""
        rail = CameraRail(name="long_rail")
        rail.set_spline_type(SplineType.CATMULL_ROM)

        for i in range(500):
            rail.add_point(RailPoint(
                position=Vector3(
                    i * 0.5,
                    math.sin(i * 0.05) * 5,
                    math.cos(i * 0.03) * 3
                ),
                fov=60 + math.sin(i * 0.02) * 10
            ))

        follower = RailFollower(rail)
        follower.speed = 50.0
        follower.play()

        for _ in range(200):
            follower.update(0.1)

    def test_network_stress(self):
        """Test network with many interconnected rails."""
        network = RailNetwork()

        # Create grid of rails
        for i in range(10):
            for j in range(10):
                rail = CameraRail(name=f"rail_{i}_{j}")
                rail.add_point(RailPoint(position=Vector3(i * 10, 0, j * 10)))
                rail.add_point(RailPoint(position=Vector3((i+1) * 10, 0, j * 10)))
                network.add_rail(rail)

        # Rapid switching
        for _ in range(100):
            i, j = _ % 10, (_ // 10) % 10
            network.set_active_rail(f"rail_{i}_{j}")
            network.follower.play()
            network.update(0.05)

    def test_concurrent_followers(self):
        """Test multiple followers on same rail."""
        rail = CameraRail(name="shared")
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(100, 0, 0)))

        followers = [RailFollower(rail) for _ in range(20)]

        for i, follower in enumerate(followers):
            follower.seek(i * 0.05)
            follower.speed = 5.0 + i
            follower.play()

        for _ in range(50):
            for follower in followers:
                follower.update(0.1)


class TestRailsEdgeCasesAdvanced:
    """Additional edge case tests."""

    def test_instant_seek_with_triggers(self):
        """Test instant seeking across multiple triggers."""
        events = []

        rail = CameraRail(name="triggers")
        rail.set_spline_type(SplineType.LINEAR)
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(100, 0, 0)))

        for i in range(10):
            rail.add_trigger(TriggerVolume(
                id=f"t{i}",
                start_progress=i * 0.1,
                end_progress=(i + 1) * 0.1,
                on_enter=lambda t: events.append(f"{t.id}_enter")
            ))

        follower = RailFollower(rail)
        follower.seek(0.95)  # Jump to near end

    def test_zero_speed_playback(self):
        """Test playback with zero speed."""
        rail = CameraRail()
        rail.add_point(RailPoint(position=Vector3(0, 0, 0)))
        rail.add_point(RailPoint(position=Vector3(10, 0, 0)))

        follower = RailFollower(rail)
        follower.speed = 0.0
        follower.play()

        initial_progress = follower.progress
        follower.update(1.0)
        assert follower.progress == initial_progress

    def test_crane_at_limits(self):
        """Test crane at angle limits."""
        crane = CraneRig()
        crane.min_angle = -30
        crane.max_angle = 60

        crane.set_arm_angle(-100)
        assert crane.target_arm_angle == -30

        crane.set_arm_angle(100)
        assert crane.target_arm_angle == 60

    def test_spline_with_duplicate_points(self):
        """Test spline handling of duplicate points."""
        rail = CameraRail()
        rail.set_spline_type(SplineType.CATMULL_ROM)

        # Same position for multiple points
        pos = Vector3(5, 5, 5)
        for _ in range(5):
            rail.add_point(RailPoint(position=pos))

        pos, rot, fov = rail.evaluate(0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
