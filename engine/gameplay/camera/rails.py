"""Camera Rails and Paths - Spline-based camera movement and trigger volumes.

This module provides camera rails, dolly tracks, crane movements, and trigger
volumes for cinematic camera control and level-driven camera changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import math

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat

from engine.gameplay.camera.constants import (
    RAIL_SPLINE_RESOLUTION,
    MIN_RAIL_SEGMENT_LENGTH,
    MAX_RAIL_POINTS,
    DEFAULT_SPLINE_TENSION,
    DEFAULT_RAIL_SPEED,
    DEFAULT_DOLLY_SPEED,
    DEFAULT_CRANE_ARC_ANGLE,
    DEFAULT_CRANE_ARM_LENGTH,
    DEFAULT_FOV,
    TRIGGER_CHECK_INTERVAL,
    TRIGGER_HYSTERESIS,
    MAX_ACTIVE_TRIGGERS,
    CAMERA_EPSILON,
    MIN_DELTA_TIME,
    MAX_DELTA_TIME,
    DEG_TO_RAD,
    BEZIER_TANGENT_SCALE,
    TANGENT_CALC_DELTA,
)

if TYPE_CHECKING:
    from engine.gameplay.camera.controller import BaseCameraController


class LoopMode(Enum):
    """How the rail handles reaching the end."""
    ONCE = auto()           # Stop at end
    LOOP = auto()           # Restart from beginning
    PING_PONG = auto()      # Reverse direction at ends
    CLAMP = auto()          # Stay at end position


class SplineType(Enum):
    """Type of spline interpolation."""
    LINEAR = auto()         # Linear interpolation between points
    CATMULL_ROM = auto()    # Catmull-Rom spline (smooth)
    BEZIER = auto()         # Bezier curves
    HERMITE = auto()        # Hermite spline


@dataclass(slots=True)
class RailPoint:
    """
    A point along a camera rail.

    Attributes:
        position: World position
        rotation: Camera rotation at this point
        fov: Field of view at this point
        tension: Spline tension (affects curve tightness)
        speed_multiplier: Speed adjustment at this point
        event_callback: Optional callback when passing this point
    """
    position: Vec3
    rotation: Optional[Quat] = None
    fov: float = DEFAULT_FOV
    tension: float = DEFAULT_SPLINE_TENSION
    speed_multiplier: float = 1.0
    event_callback: Optional[Callable[[], None]] = None


class CameraRail:
    """
    Spline-based camera path.

    Features:
    - Multiple spline types (linear, Catmull-Rom, Bezier)
    - Position, rotation, and FOV interpolation
    - Variable speed along path
    - Event triggers at waypoints
    - Length calculation and uniform parameterization
    """

    __slots__ = (
        "_points",
        "_spline_type",
        "_resolution",
        "_total_length",
        "_segment_lengths",
        "_parameterization",  # Maps uniform t to non-uniform
        "_closed_loop",
        "_name",
    )

    def __init__(
        self,
        points: Optional[List[RailPoint]] = None,
        spline_type: SplineType = SplineType.CATMULL_ROM,
        resolution: int = RAIL_SPLINE_RESOLUTION,
        closed_loop: bool = False,
        name: str = "",
    ) -> None:
        """
        Initialize camera rail.

        Args:
            points: Initial rail points
            spline_type: Type of spline interpolation
            resolution: Points per segment for length calculation
            closed_loop: Whether the rail forms a closed loop
            name: Optional identifier
        """
        self._points: List[RailPoint] = points if points is not None else []
        self._spline_type = spline_type
        self._resolution = resolution
        self._total_length = 0.0
        self._segment_lengths: List[float] = []
        self._parameterization: List[float] = []
        self._closed_loop = closed_loop
        self._name = name

        if len(self._points) >= 2:
            self._recalculate_length()

    @property
    def name(self) -> str:
        """Get rail name."""
        return self._name

    @property
    def point_count(self) -> int:
        """Get number of points."""
        return len(self._points)

    @property
    def total_length(self) -> float:
        """Get total rail length."""
        return self._total_length

    @property
    def is_closed(self) -> bool:
        """Check if rail is closed loop."""
        return self._closed_loop

    @property
    def spline_type(self) -> SplineType:
        """Get spline type."""
        return self._spline_type

    @spline_type.setter
    def spline_type(self, value: SplineType) -> None:
        """Set spline type and recalculate."""
        self._spline_type = value
        self._recalculate_length()

    def add_point(self, point: RailPoint) -> None:
        """Add a point to the rail."""
        if len(self._points) < MAX_RAIL_POINTS:
            self._points.append(point)
            if len(self._points) >= 2:
                self._recalculate_length()

    def insert_point(self, index: int, point: RailPoint) -> None:
        """Insert a point at specific index."""
        if len(self._points) < MAX_RAIL_POINTS:
            self._points.insert(index, point)
            if len(self._points) >= 2:
                self._recalculate_length()

    def remove_point(self, index: int) -> Optional[RailPoint]:
        """Remove point at index."""
        if 0 <= index < len(self._points):
            point = self._points.pop(index)
            if len(self._points) >= 2:
                self._recalculate_length()
            return point
        return None

    def get_point(self, index: int) -> Optional[RailPoint]:
        """Get point at index."""
        if 0 <= index < len(self._points):
            return self._points[index]
        return None

    def clear(self) -> None:
        """Clear all points."""
        self._points.clear()
        self._total_length = 0.0
        self._segment_lengths.clear()
        self._parameterization.clear()

    def _recalculate_length(self) -> None:
        """Recalculate total length and parameterization."""
        if len(self._points) < 2:
            self._total_length = 0.0
            return

        self._segment_lengths.clear()
        self._parameterization.clear()
        self._total_length = 0.0

        # Sample along spline to calculate length
        num_segments = len(self._points) - 1 if not self._closed_loop else len(self._points)

        accumulated = 0.0
        self._parameterization.append(0.0)

        for seg in range(num_segments):
            segment_length = 0.0
            prev_pos = self._evaluate_raw(seg / num_segments)

            for i in range(1, self._resolution + 1):
                t = (seg + i / self._resolution) / num_segments
                curr_pos = self._evaluate_raw(t)
                segment_length += (curr_pos - prev_pos).length()
                prev_pos = curr_pos

            self._segment_lengths.append(segment_length)
            accumulated += segment_length
            self._parameterization.append(accumulated)

        self._total_length = accumulated

        # Normalize parameterization
        if self._total_length > CAMERA_EPSILON:
            for i in range(len(self._parameterization)):
                self._parameterization[i] /= self._total_length

    def _evaluate_raw(self, t: float) -> Vec3:
        """Evaluate position at raw parameter t (non-uniform)."""
        if len(self._points) < 2:
            return self._points[0].position if self._points else Vec3.zero()

        t = max(0.0, min(1.0, t))

        num_segments = len(self._points) - 1 if not self._closed_loop else len(self._points)
        segment_t = t * num_segments
        segment = int(segment_t)
        local_t = segment_t - segment

        if segment >= num_segments:
            segment = num_segments - 1
            local_t = 1.0

        return self._interpolate_segment(segment, local_t)

    def _interpolate_segment(self, segment: int, t: float) -> Vec3:
        """Interpolate within a single segment."""
        n = len(self._points)

        if self._spline_type == SplineType.LINEAR:
            p0 = self._points[segment].position
            p1 = self._points[(segment + 1) % n].position if self._closed_loop else self._points[min(segment + 1, n - 1)].position
            return p0.lerp(p1, t)

        elif self._spline_type == SplineType.CATMULL_ROM:
            return self._catmull_rom_interpolate(segment, t)

        elif self._spline_type == SplineType.BEZIER:
            return self._bezier_interpolate(segment, t)

        elif self._spline_type == SplineType.HERMITE:
            return self._hermite_interpolate(segment, t)

        return self._points[segment].position

    def _catmull_rom_interpolate(self, segment: int, t: float) -> Vec3:
        """Catmull-Rom spline interpolation."""
        n = len(self._points)

        # Get 4 control points
        if self._closed_loop:
            p0 = self._points[(segment - 1) % n].position
            p1 = self._points[segment % n].position
            p2 = self._points[(segment + 1) % n].position
            p3 = self._points[(segment + 2) % n].position
            tension = self._points[segment % n].tension
        else:
            i0 = max(0, segment - 1)
            i1 = segment
            i2 = min(segment + 1, n - 1)
            i3 = min(segment + 2, n - 1)
            p0 = self._points[i0].position
            p1 = self._points[i1].position
            p2 = self._points[i2].position
            p3 = self._points[i3].position
            tension = self._points[i1].tension

        # Catmull-Rom matrix coefficients
        t2 = t * t
        t3 = t2 * t

        # Tension-adjusted basis functions
        s = (1.0 - tension) / 2.0

        b0 = -s * t3 + 2 * s * t2 - s * t
        b1 = (2 - s) * t3 + (s - 3) * t2 + 1
        b2 = (s - 2) * t3 + (3 - 2 * s) * t2 + s * t
        b3 = s * t3 - s * t2

        return Vec3(
            p0.x * b0 + p1.x * b1 + p2.x * b2 + p3.x * b3,
            p0.y * b0 + p1.y * b1 + p2.y * b2 + p3.y * b3,
            p0.z * b0 + p1.z * b1 + p2.z * b2 + p3.z * b3,
        )

    def _bezier_interpolate(self, segment: int, t: float) -> Vec3:
        """Cubic Bezier interpolation."""
        n = len(self._points)

        # For Bezier, we need to generate control points
        # Using adjacent points to estimate tangents
        if self._closed_loop:
            p0 = self._points[segment % n].position
            p3 = self._points[(segment + 1) % n].position
            prev_pos = self._points[(segment - 1) % n].position
            next_pos = self._points[(segment + 2) % n].position
        else:
            i0 = segment
            i3 = min(segment + 1, n - 1)
            p0 = self._points[i0].position
            p3 = self._points[i3].position
            prev_pos = self._points[max(0, segment - 1)].position
            next_pos = self._points[min(segment + 2, n - 1)].position

        # Generate control points from tangents
        tangent0 = (p3 - prev_pos) * BEZIER_TANGENT_SCALE
        tangent1 = (next_pos - p0) * BEZIER_TANGENT_SCALE
        p1 = p0 + tangent0
        p2 = p3 - tangent1

        # De Casteljau's algorithm
        u = 1.0 - t
        return (p0 * (u * u * u) +
                p1 * (3 * u * u * t) +
                p2 * (3 * u * t * t) +
                p3 * (t * t * t))

    def _hermite_interpolate(self, segment: int, t: float) -> Vec3:
        """Hermite spline interpolation."""
        n = len(self._points)

        # Get positions and calculate tangents
        if self._closed_loop:
            p0 = self._points[segment % n].position
            p1 = self._points[(segment + 1) % n].position
            prev_pos = self._points[(segment - 1) % n].position
            next_pos = self._points[(segment + 2) % n].position
        else:
            i0 = segment
            i1 = min(segment + 1, n - 1)
            p0 = self._points[i0].position
            p1 = self._points[i1].position
            prev_pos = self._points[max(0, segment - 1)].position
            next_pos = self._points[min(segment + 2, n - 1)].position

        # Tangents (finite differences)
        m0 = (p1 - prev_pos) * 0.5
        m1 = (next_pos - p0) * 0.5

        # Hermite basis functions
        t2 = t * t
        t3 = t2 * t

        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        return p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11

    def _arc_length_to_parameter(self, s: float) -> float:
        """Convert arc length parameter to spline parameter."""
        if self._total_length < CAMERA_EPSILON or len(self._parameterization) < 2:
            return s

        s = max(0.0, min(1.0, s))

        # Binary search for segment
        low = 0
        high = len(self._parameterization) - 1

        while low < high:
            mid = (low + high) // 2
            if self._parameterization[mid] < s:
                low = mid + 1
            else:
                high = mid

        if low == 0:
            return 0.0

        # Linear interpolation within segment
        seg_start = self._parameterization[low - 1]
        seg_end = self._parameterization[low]
        seg_len = seg_end - seg_start

        if seg_len < CAMERA_EPSILON:
            return (low - 1) / (len(self._parameterization) - 1)

        local_t = (s - seg_start) / seg_len
        return ((low - 1) + local_t) / (len(self._parameterization) - 1)

    def evaluate_at(self, t: float, uniform: bool = True) -> Tuple[Vec3, Optional[Quat], float]:
        """
        Evaluate rail at parameter t.

        Args:
            t: Parameter (0.0 to 1.0)
            uniform: Use arc-length parameterization

        Returns:
            Tuple of (position, rotation, fov)
        """
        if len(self._points) == 0:
            return Vec3.zero(), None, DEFAULT_FOV

        if len(self._points) == 1:
            p = self._points[0]
            return p.position, p.rotation, p.fov

        # Convert to uniform if requested
        if uniform:
            t = self._arc_length_to_parameter(t)

        # Get position
        position = self._evaluate_raw(t)

        # Interpolate rotation and FOV
        num_segments = len(self._points) - 1 if not self._closed_loop else len(self._points)
        segment_t = t * num_segments
        segment = int(min(segment_t, num_segments - 1))
        local_t = segment_t - segment

        n = len(self._points)
        if self._closed_loop:
            p0 = self._points[segment % n]
            p1 = self._points[(segment + 1) % n]
        else:
            p0 = self._points[segment]
            p1 = self._points[min(segment + 1, n - 1)]

        # Interpolate rotation
        rotation = None
        if p0.rotation is not None and p1.rotation is not None:
            rotation = p0.rotation.slerp(p1.rotation, local_t)
        elif p0.rotation is not None:
            rotation = p0.rotation
        elif p1.rotation is not None:
            rotation = p1.rotation

        # Interpolate FOV
        fov = p0.fov + (p1.fov - p0.fov) * local_t

        return position, rotation, fov

    def get_tangent_at(self, t: float, uniform: bool = True) -> Vec3:
        """
        Get tangent direction at parameter t.

        Args:
            t: Parameter (0.0 to 1.0)
            uniform: Use arc-length parameterization

        Returns:
            Normalized tangent vector
        """
        delta = TANGENT_CALC_DELTA
        t0 = max(0.0, t - delta)
        t1 = min(1.0, t + delta)

        pos0, _, _ = self.evaluate_at(t0, uniform)
        pos1, _, _ = self.evaluate_at(t1, uniform)

        tangent = pos1 - pos0
        if tangent.length_squared() < CAMERA_EPSILON:
            return Vec3.forward()
        return tangent.normalized()

    def get_length(self) -> float:
        """Get total rail length."""
        return self._total_length

    def get_closest_point(self, position: Vec3, samples: int = 50) -> Tuple[float, Vec3]:
        """
        Find closest point on rail to given position.

        Args:
            position: Query position
            samples: Number of samples to check

        Returns:
            Tuple of (parameter t, closest position)
        """
        min_dist = float("inf")
        closest_t = 0.0
        closest_pos = Vec3.zero()

        for i in range(samples + 1):
            t = i / samples
            pos, _, _ = self.evaluate_at(t)
            dist = (pos - position).length_squared()
            if dist < min_dist:
                min_dist = dist
                closest_t = t
                closest_pos = pos

        return closest_t, closest_pos


class RailFollower:
    """
    Follows a camera rail with speed and loop control.

    Features:
    - Speed control with easing
    - Multiple loop modes
    - Event triggering at waypoints
    - Pause and resume
    """

    __slots__ = (
        "_rail",
        "_progress",
        "_speed",
        "_target_speed",
        "_speed_interpolation",
        "_loop_mode",
        "_direction",  # 1 or -1 for ping-pong
        "_is_playing",
        "_is_paused",
        "_last_waypoint",
        "_on_complete",
        "_on_waypoint",
    )

    def __init__(
        self,
        rail: CameraRail,
        speed: float = DEFAULT_RAIL_SPEED,
        loop_mode: LoopMode = LoopMode.ONCE,
    ) -> None:
        """
        Initialize rail follower.

        Args:
            rail: Rail to follow
            speed: Movement speed (units per second)
            loop_mode: Behavior at rail end
        """
        self._rail = rail
        self._progress = 0.0
        self._speed = speed
        self._target_speed = speed
        self._speed_interpolation = 5.0
        self._loop_mode = loop_mode
        self._direction = 1
        self._is_playing = False
        self._is_paused = False
        self._last_waypoint = -1
        self._on_complete: List[Callable[[], None]] = []
        self._on_waypoint: List[Callable[[int], None]] = []

    @property
    def rail(self) -> CameraRail:
        """Get the rail being followed."""
        return self._rail

    @rail.setter
    def rail(self, value: CameraRail) -> None:
        """Set new rail and reset progress."""
        self._rail = value
        self._progress = 0.0
        self._last_waypoint = -1

    @property
    def progress(self) -> float:
        """Get current progress (0.0 to 1.0)."""
        return self._progress

    @progress.setter
    def progress(self, value: float) -> None:
        """Set progress (clamped 0.0 to 1.0)."""
        self._progress = max(0.0, min(1.0, value))

    @property
    def speed(self) -> float:
        """Get current speed."""
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        """Set target speed."""
        self._target_speed = max(0.0, value)

    @property
    def loop_mode(self) -> LoopMode:
        """Get loop mode."""
        return self._loop_mode

    @loop_mode.setter
    def loop_mode(self, value: LoopMode) -> None:
        """Set loop mode."""
        self._loop_mode = value

    @property
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._is_playing and not self._is_paused

    @property
    def is_complete(self) -> bool:
        """Check if playback is complete (for non-looping)."""
        if self._loop_mode in (LoopMode.LOOP, LoopMode.PING_PONG):
            return False
        return self._progress >= 1.0 or self._progress <= 0.0

    def play(self) -> None:
        """Start or resume playback."""
        self._is_playing = True
        self._is_paused = False

    def pause(self) -> None:
        """Pause playback."""
        self._is_paused = True

    def stop(self) -> None:
        """Stop and reset playback."""
        self._is_playing = False
        self._is_paused = False
        self._progress = 0.0
        self._direction = 1
        self._last_waypoint = -1

    def seek(self, progress: float) -> None:
        """Seek to specific progress."""
        self._progress = max(0.0, min(1.0, progress))

    def on_complete(self, callback: Callable[[], None]) -> None:
        """Register completion callback."""
        self._on_complete.append(callback)

    def on_waypoint(self, callback: Callable[[int], None]) -> None:
        """Register waypoint callback."""
        self._on_waypoint.append(callback)

    def evaluate(self) -> Tuple[Vec3, Optional[Quat], float]:
        """Get current position, rotation, and FOV."""
        return self._rail.evaluate_at(self._progress)

    def update(self, delta_time: float) -> Tuple[Vec3, Optional[Quat], float]:
        """
        Update follower and return current state.

        Args:
            delta_time: Time since last update

        Returns:
            Tuple of (position, rotation, fov)
        """
        if not self._is_playing or self._is_paused:
            return self.evaluate()

        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Interpolate speed
        speed_diff = self._target_speed - self._speed
        if abs(speed_diff) > CAMERA_EPSILON:
            self._speed += speed_diff * min(1.0, self._speed_interpolation * delta_time)

        # Get speed multiplier from current rail point
        rail_length = self._rail.get_length()
        if rail_length < CAMERA_EPSILON:
            return self.evaluate()

        # Calculate progress delta
        progress_delta = (self._speed * delta_time / rail_length) * self._direction

        # Get current waypoint index before moving
        num_points = self._rail.point_count
        if num_points > 0:
            current_waypoint = int(self._progress * (num_points - 1))

            # Apply speed multiplier
            point = self._rail.get_point(current_waypoint)
            if point is not None:
                progress_delta *= point.speed_multiplier

        # Update progress
        self._progress += progress_delta

        # Handle end of rail
        if self._progress >= 1.0:
            self._handle_end_reached()
        elif self._progress <= 0.0:
            self._handle_start_reached()

        # Check for waypoint events
        if num_points > 0:
            new_waypoint = int(self._progress * (num_points - 1))
            if new_waypoint != self._last_waypoint:
                self._trigger_waypoint(new_waypoint)
                self._last_waypoint = new_waypoint

        return self.evaluate()

    def _handle_end_reached(self) -> None:
        """Handle reaching end of rail."""
        if self._loop_mode == LoopMode.ONCE:
            self._progress = 1.0
            self._is_playing = False
            for callback in self._on_complete:
                callback()
        elif self._loop_mode == LoopMode.LOOP:
            self._progress = self._progress % 1.0
            self._last_waypoint = -1
        elif self._loop_mode == LoopMode.PING_PONG:
            self._progress = 1.0 - (self._progress - 1.0)
            self._direction = -1
        elif self._loop_mode == LoopMode.CLAMP:
            self._progress = 1.0

    def _handle_start_reached(self) -> None:
        """Handle reaching start of rail (ping-pong mode)."""
        if self._loop_mode == LoopMode.PING_PONG:
            self._progress = -self._progress
            self._direction = 1
        else:
            self._progress = 0.0

    def _trigger_waypoint(self, waypoint: int) -> None:
        """Trigger waypoint events."""
        point = self._rail.get_point(waypoint)
        if point is not None and point.event_callback is not None:
            point.event_callback()

        for callback in self._on_waypoint:
            callback(waypoint)


@dataclass(slots=True)
class TriggerBounds:
    """Axis-aligned bounding box for trigger volumes."""
    min_point: Vec3
    max_point: Vec3

    def contains(self, point: Vec3) -> bool:
        """Check if point is inside bounds."""
        return (
            self.min_point.x <= point.x <= self.max_point.x and
            self.min_point.y <= point.y <= self.max_point.y and
            self.min_point.z <= point.z <= self.max_point.z
        )

    def distance_to(self, point: Vec3) -> float:
        """Calculate distance from point to bounds."""
        # Clamp point to bounds
        closest = Vec3(
            max(self.min_point.x, min(self.max_point.x, point.x)),
            max(self.min_point.y, min(self.max_point.y, point.y)),
            max(self.min_point.z, min(self.max_point.z, point.z)),
        )
        return (point - closest).length()


class TriggerVolume:
    """
    Volume that triggers camera changes when entered/exited.

    Features:
    - Enter, exit, and stay callbacks
    - Bounds-based detection
    - Hysteresis to prevent flickering
    - Priority for overlapping volumes
    """

    __slots__ = (
        "_bounds",
        "_is_inside",
        "_last_distance",
        "_on_enter",
        "_on_exit",
        "_on_stay",
        "_priority",
        "_enabled",
        "_name",
        "_target_camera_mode",
        "_target_rail",
    )

    def __init__(
        self,
        bounds: TriggerBounds,
        priority: int = 0,
        name: str = "",
    ) -> None:
        """
        Initialize trigger volume.

        Args:
            bounds: Volume bounds
            priority: Priority for overlapping volumes (higher = takes precedence)
            name: Optional identifier
        """
        self._bounds = bounds
        self._is_inside = False
        self._last_distance = float("inf")
        self._on_enter: List[Callable[[], None]] = []
        self._on_exit: List[Callable[[], None]] = []
        self._on_stay: List[Callable[[float], None]] = []
        self._priority = priority
        self._enabled = True
        self._name = name
        self._target_camera_mode: Optional[Any] = None
        self._target_rail: Optional[CameraRail] = None

    @property
    def bounds(self) -> TriggerBounds:
        """Get trigger bounds."""
        return self._bounds

    @property
    def is_inside(self) -> bool:
        """Check if currently inside trigger."""
        return self._is_inside

    @property
    def priority(self) -> int:
        """Get trigger priority."""
        return self._priority

    @property
    def enabled(self) -> bool:
        """Check if trigger is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable/disable trigger."""
        self._enabled = value
        if not value and self._is_inside:
            self._is_inside = False
            for callback in self._on_exit:
                callback()

    @property
    def name(self) -> str:
        """Get trigger name."""
        return self._name

    def set_target_camera_mode(self, mode: Any) -> None:
        """Set camera mode to switch to when entering."""
        self._target_camera_mode = mode

    def set_target_rail(self, rail: CameraRail) -> None:
        """Set rail to follow when entering."""
        self._target_rail = rail

    def on_enter(self, callback: Callable[[], None]) -> None:
        """Register enter callback."""
        self._on_enter.append(callback)

    def on_exit(self, callback: Callable[[], None]) -> None:
        """Register exit callback."""
        self._on_exit.append(callback)

    def on_stay(self, callback: Callable[[float], None]) -> None:
        """Register stay callback (receives delta time)."""
        self._on_stay.append(callback)

    def check(self, position: Vec3, delta_time: float) -> bool:
        """
        Check position against trigger volume.

        Args:
            position: Position to check
            delta_time: Time since last check

        Returns:
            True if inside trigger
        """
        if not self._enabled:
            return False

        is_inside_now = self._bounds.contains(position)
        distance = self._bounds.distance_to(position)

        # Apply hysteresis
        if self._is_inside and not is_inside_now:
            # Only exit if clearly outside
            if distance > TRIGGER_HYSTERESIS:
                self._is_inside = False
                for callback in self._on_exit:
                    callback()
        elif not self._is_inside and is_inside_now:
            self._is_inside = True
            for callback in self._on_enter:
                callback()

        # Stay callbacks
        if self._is_inside:
            for callback in self._on_stay:
                callback(delta_time)

        self._last_distance = distance
        return self._is_inside


class BlendRegion:
    """
    Region that blends between camera modes or rails.

    Features:
    - Smooth transitions based on position
    - Blend weight calculation
    - Support for mode and rail transitions
    """

    __slots__ = (
        "_bounds",
        "_blend_axis",  # Which axis to blend along
        "_from_mode",
        "_to_mode",
        "_from_rail",
        "_to_rail",
        "_enabled",
        "_name",
    )

    def __init__(
        self,
        bounds: TriggerBounds,
        blend_axis: str = "x",  # "x", "y", or "z"
        name: str = "",
    ) -> None:
        """
        Initialize blend region.

        Args:
            bounds: Region bounds
            blend_axis: Axis along which to calculate blend weight
            name: Optional identifier
        """
        self._bounds = bounds
        self._blend_axis = blend_axis.lower()
        self._from_mode: Optional[Any] = None
        self._to_mode: Optional[Any] = None
        self._from_rail: Optional[CameraRail] = None
        self._to_rail: Optional[CameraRail] = None
        self._enabled = True
        self._name = name

    def set_mode_blend(self, from_mode: Any, to_mode: Any) -> None:
        """Set modes to blend between."""
        self._from_mode = from_mode
        self._to_mode = to_mode

    def set_rail_blend(self, from_rail: CameraRail, to_rail: CameraRail) -> None:
        """Set rails to blend between."""
        self._from_rail = from_rail
        self._to_rail = to_rail

    def get_blend_weight(self, position: Vec3) -> float:
        """
        Calculate blend weight for position.

        Args:
            position: Query position

        Returns:
            Blend weight (0.0 = from, 1.0 = to)
        """
        if not self._enabled or not self._bounds.contains(position):
            return 0.0

        # Get bounds range along blend axis
        if self._blend_axis == "x":
            min_val = self._bounds.min_point.x
            max_val = self._bounds.max_point.x
            pos_val = position.x
        elif self._blend_axis == "y":
            min_val = self._bounds.min_point.y
            max_val = self._bounds.max_point.y
            pos_val = position.y
        else:  # z
            min_val = self._bounds.min_point.z
            max_val = self._bounds.max_point.z
            pos_val = position.z

        range_val = max_val - min_val
        if range_val < CAMERA_EPSILON:
            return 0.5

        return (pos_val - min_val) / range_val


class Dolly:
    """
    Linear dolly track for camera movement.

    Simpler than a full rail - just a straight line path.
    """

    __slots__ = (
        "_start",
        "_end",
        "_position",
        "_speed",
        "_is_playing",
        "_direction",
    )

    def __init__(
        self,
        start: Vec3,
        end: Vec3,
        speed: float = DEFAULT_DOLLY_SPEED,
    ) -> None:
        """
        Initialize dolly track.

        Args:
            start: Start position
            end: End position
            speed: Movement speed
        """
        self._start = start
        self._end = end
        self._position = 0.0  # 0.0 = start, 1.0 = end
        self._speed = speed
        self._is_playing = False
        self._direction = 1

    @property
    def position(self) -> float:
        """Get position along dolly (0.0 to 1.0)."""
        return self._position

    @position.setter
    def position(self, value: float) -> None:
        """Set position along dolly."""
        self._position = max(0.0, min(1.0, value))

    @property
    def current_position(self) -> Vec3:
        """Get current world position."""
        return self._start.lerp(self._end, self._position)

    @property
    def length(self) -> float:
        """Get dolly track length."""
        return (self._end - self._start).length()

    def play_forward(self) -> None:
        """Start moving toward end."""
        self._is_playing = True
        self._direction = 1

    def play_backward(self) -> None:
        """Start moving toward start."""
        self._is_playing = True
        self._direction = -1

    def stop(self) -> None:
        """Stop movement."""
        self._is_playing = False

    def reset(self) -> None:
        """Reset to start position."""
        self._position = 0.0
        self._is_playing = False
        self._direction = 1

    def update(self, delta_time: float) -> Vec3:
        """
        Update dolly position.

        Args:
            delta_time: Time since last update

        Returns:
            Current world position
        """
        if not self._is_playing:
            return self.current_position

        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        track_length = self.length
        if track_length < CAMERA_EPSILON:
            return self._start

        progress_delta = (self._speed * delta_time / track_length) * self._direction
        self._position = max(0.0, min(1.0, self._position + progress_delta))

        # Stop at ends
        if self._position <= 0.0 or self._position >= 1.0:
            self._is_playing = False

        return self.current_position


class Crane:
    """
    Crane arm for vertical arc movement.

    Features:
    - Vertical arc motion
    - Configurable arm length and angle range
    - Pivot point offset
    """

    __slots__ = (
        "_pivot",
        "_arm_length",
        "_min_angle",
        "_max_angle",
        "_current_angle",
        "_target_angle",
        "_speed",
        "_is_playing",
    )

    def __init__(
        self,
        pivot: Vec3,
        arm_length: float = DEFAULT_CRANE_ARM_LENGTH,
        min_angle: float = -45.0,
        max_angle: float = 45.0,
    ) -> None:
        """
        Initialize crane.

        Args:
            pivot: Pivot point position
            arm_length: Crane arm length
            min_angle: Minimum angle in degrees (down)
            max_angle: Maximum angle in degrees (up)
        """
        self._pivot = pivot
        self._arm_length = arm_length
        self._min_angle = min_angle
        self._max_angle = max_angle
        self._current_angle = 0.0
        self._target_angle = 0.0
        self._speed = 45.0  # Degrees per second
        self._is_playing = False

    @property
    def pivot(self) -> Vec3:
        """Get pivot position."""
        return self._pivot

    @pivot.setter
    def pivot(self, value: Vec3) -> None:
        """Set pivot position."""
        self._pivot = value

    @property
    def arm_length(self) -> float:
        """Get arm length."""
        return self._arm_length

    @arm_length.setter
    def arm_length(self, value: float) -> None:
        """Set arm length."""
        self._arm_length = max(1.0, value)

    @property
    def current_angle(self) -> float:
        """Get current angle in degrees."""
        return self._current_angle

    @property
    def current_position(self) -> Vec3:
        """Get current camera position on crane arc."""
        angle_rad = self._current_angle * DEG_TO_RAD
        offset = Vec3(
            0.0,
            math.sin(angle_rad) * self._arm_length,
            math.cos(angle_rad) * self._arm_length,
        )
        return self._pivot + offset

    def set_angle(self, angle: float) -> None:
        """Set target angle."""
        self._target_angle = max(self._min_angle, min(self._max_angle, angle))
        self._is_playing = True

    def set_angle_immediate(self, angle: float) -> None:
        """Set angle immediately without interpolation."""
        self._current_angle = max(self._min_angle, min(self._max_angle, angle))
        self._target_angle = self._current_angle
        self._is_playing = False

    def move_up(self, amount: float = 10.0) -> None:
        """Move crane up by amount degrees."""
        self.set_angle(self._target_angle + amount)

    def move_down(self, amount: float = 10.0) -> None:
        """Move crane down by amount degrees."""
        self.set_angle(self._target_angle - amount)

    def update(self, delta_time: float) -> Vec3:
        """
        Update crane position.

        Args:
            delta_time: Time since last update

        Returns:
            Current camera position
        """
        if not self._is_playing:
            return self.current_position

        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        angle_diff = self._target_angle - self._current_angle
        if abs(angle_diff) > CAMERA_EPSILON:
            max_change = self._speed * delta_time
            if abs(angle_diff) <= max_change:
                self._current_angle = self._target_angle
                self._is_playing = False
            else:
                self._current_angle += math.copysign(max_change, angle_diff)
        else:
            self._is_playing = False

        return self.current_position


class TriggerVolumeManager:
    """
    Manages multiple trigger volumes.

    Features:
    - Priority-based activation
    - Efficient spatial queries
    - Active trigger tracking
    """

    __slots__ = (
        "_triggers",
        "_active_triggers",
        "_check_interval",
        "_time_since_check",
    )

    def __init__(self, check_interval: float = TRIGGER_CHECK_INTERVAL) -> None:
        """
        Initialize trigger manager.

        Args:
            check_interval: Time between trigger checks
        """
        self._triggers: List[TriggerVolume] = []
        self._active_triggers: List[TriggerVolume] = []
        self._check_interval = check_interval
        self._time_since_check = 0.0

    def add_trigger(self, trigger: TriggerVolume) -> None:
        """Add a trigger volume."""
        if len(self._triggers) < MAX_ACTIVE_TRIGGERS:
            self._triggers.append(trigger)
            # Keep sorted by priority (highest first)
            self._triggers.sort(key=lambda t: t.priority, reverse=True)

    def remove_trigger(self, trigger: TriggerVolume) -> None:
        """Remove a trigger volume."""
        if trigger in self._triggers:
            self._triggers.remove(trigger)
        if trigger in self._active_triggers:
            self._active_triggers.remove(trigger)

    def get_trigger_by_name(self, name: str) -> Optional[TriggerVolume]:
        """Find trigger by name."""
        for trigger in self._triggers:
            if trigger.name == name:
                return trigger
        return None

    @property
    def active_triggers(self) -> List[TriggerVolume]:
        """Get list of currently active triggers."""
        return self._active_triggers.copy()

    @property
    def highest_priority_trigger(self) -> Optional[TriggerVolume]:
        """Get the highest priority active trigger."""
        if self._active_triggers:
            return self._active_triggers[0]
        return None

    def update(self, position: Vec3, delta_time: float) -> List[TriggerVolume]:
        """
        Update all triggers.

        Args:
            position: Current position to check
            delta_time: Time since last update

        Returns:
            List of active triggers
        """
        self._time_since_check += delta_time

        if self._time_since_check < self._check_interval:
            # Just call stay callbacks
            for trigger in self._active_triggers:
                if trigger.enabled and trigger.is_inside:
                    trigger.check(position, delta_time)
            return self._active_triggers

        self._time_since_check = 0.0

        # Full check
        self._active_triggers.clear()
        for trigger in self._triggers:
            if trigger.check(position, delta_time):
                self._active_triggers.append(trigger)

        return self._active_triggers


__all__ = [
    "LoopMode",
    "SplineType",
    "RailPoint",
    "CameraRail",
    "RailFollower",
    "TriggerBounds",
    "TriggerVolume",
    "BlendRegion",
    "Dolly",
    "Crane",
    "TriggerVolumeManager",
]
