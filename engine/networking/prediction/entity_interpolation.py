"""
Entity interpolation system for smooth rendering of remote entities.

For entities controlled by other players or the server, we receive
periodic state updates but need smooth visual representation. This
module provides:
- Snapshot buffering for consistent interpolation delay
- Linear and hermite interpolation between states
- Extrapolation for brief network gaps
- Proper quaternion interpolation for rotations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
from collections import deque
import math

from engine.networking.config import (
    DEFAULT_INTERPOLATION_BUFFER_SIZE,
    DEFAULT_EXTRAPOLATION_LIMIT,
    DEFAULT_ENTITY_INTERPOLATION_DELAY,
    QUATERNION_LERP_THRESHOLD,
)


# Type aliases
Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # x, y, z, w


class InterpolationMode(Enum):
    """Interpolation method for state blending."""

    LINEAR = auto()
    """Linear interpolation (lerp) - simple but may appear stiff."""

    HERMITE = auto()
    """Cubic hermite interpolation - smoother with velocity consideration."""

    CATMULL_ROM = auto()
    """Catmull-Rom spline - smooth curve through multiple points."""


@dataclass
class Snapshot:
    """
    A single state snapshot received from the network.

    Attributes:
        position: Entity position as (x, y, z).
        rotation: Entity rotation as quaternion (x, y, z, w).
        velocity: Entity velocity as (vx, vy, vz).
        timestamp: Server timestamp when this state was recorded.
        sequence: Optional sequence number for ordering.
    """

    position: Vector3 = field(default_factory=lambda: (0.0, 0.0, 0.0))
    rotation: Optional[Quaternion] = None
    velocity: Vector3 = field(default_factory=lambda: (0.0, 0.0, 0.0))
    timestamp: float = 0.0
    sequence: int = 0

    def copy(self) -> Snapshot:
        """Create a copy of this snapshot."""
        return Snapshot(
            position=self.position,
            rotation=self.rotation,
            velocity=self.velocity,
            timestamp=self.timestamp,
            sequence=self.sequence,
        )


@dataclass
class InterpolatedState:
    """Result of interpolation between snapshots."""

    position: Vector3
    rotation: Optional[Quaternion]
    velocity: Vector3
    timestamp: float
    is_extrapolated: bool = False
    extrapolation_time: float = 0.0


def lerp_position(
    a: Vector3,
    b: Vector3,
    t: float,
) -> Vector3:
    """
    Linear interpolation between two positions.

    Args:
        a: Start position.
        b: End position.
        t: Interpolation factor (0.0 = a, 1.0 = b).

    Returns:
        Interpolated position.
    """
    t = max(0.0, min(1.0, t))  # Clamp to [0, 1]
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def lerp_scalar(a: float, b: float, t: float) -> float:
    """Linear interpolation between two scalars."""
    return a + (b - a) * t


def slerp_rotation(
    a: Quaternion,
    b: Quaternion,
    t: float,
) -> Quaternion:
    """
    Spherical linear interpolation between two quaternions.

    This ensures constant angular velocity and smooth rotation.

    Args:
        a: Start rotation quaternion (x, y, z, w).
        b: End rotation quaternion (x, y, z, w).
        t: Interpolation factor (0.0 = a, 1.0 = b).

    Returns:
        Interpolated rotation quaternion.
    """
    t = max(0.0, min(1.0, t))

    # Compute dot product
    dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]

    # If negative dot, negate one quaternion to take shorter path
    if dot < 0.0:
        b = (-b[0], -b[1], -b[2], -b[3])
        dot = -dot

    # Clamp dot to valid range
    dot = min(1.0, dot)

    # If quaternions are very close, use linear interpolation
    if dot > QUATERNION_LERP_THRESHOLD:
        result = (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        )
        # Normalize (with zero-check to avoid division by zero)
        length = math.sqrt(sum(x*x for x in result))
        if length > 0:
            return tuple(x / length for x in result)
        return result

    # Compute interpolation parameters
    theta_0 = math.acos(dot)
    theta = theta_0 * t

    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    # Prevent division by zero
    if abs(sin_theta_0) < 1e-10:
        # Fall back to linear interpolation
        result = (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        )
        length = math.sqrt(sum(x*x for x in result))
        if length > 0:
            return tuple(x / length for x in result)
        return result

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (
        a[0] * s0 + b[0] * s1,
        a[1] * s0 + b[1] * s1,
        a[2] * s0 + b[2] * s1,
        a[3] * s0 + b[3] * s1,
    )


def hermite_interpolate(
    p0: Vector3,
    p1: Vector3,
    v0: Vector3,
    v1: Vector3,
    t: float,
    duration: float = 1.0,
) -> Vector3:
    """
    Cubic hermite interpolation using position and velocity.

    Provides smoother motion that considers velocity tangents.

    Args:
        p0: Start position.
        p1: End position.
        v0: Velocity at start.
        v1: Velocity at end.
        t: Interpolation factor (0.0 to 1.0).
        duration: Time between snapshots for velocity scaling.

    Returns:
        Interpolated position.
    """
    t = max(0.0, min(1.0, t))

    # Scale velocities by duration
    m0 = (v0[0] * duration, v0[1] * duration, v0[2] * duration)
    m1 = (v1[0] * duration, v1[1] * duration, v1[2] * duration)

    # Hermite basis functions
    t2 = t * t
    t3 = t2 * t

    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2

    return (
        h00*p0[0] + h10*m0[0] + h01*p1[0] + h11*m1[0],
        h00*p0[1] + h10*m0[1] + h01*p1[1] + h11*m1[1],
        h00*p0[2] + h10*m0[2] + h01*p1[2] + h11*m1[2],
    )


class InterpolationBuffer:
    """
    Buffer for storing snapshots and computing interpolated states.

    Maintains a sliding window of recent snapshots and computes
    interpolated states for rendering at any timestamp.

    The buffer introduces intentional latency (interpolation delay)
    to ensure we always have snapshots to interpolate between.

    Example:
        buffer = InterpolationBuffer(buffer_size=3)

        # Add snapshots as they arrive
        buffer.push_snapshot(snapshot)

        # Get interpolated state for rendering
        # render_time is typically server_time - interpolation_delay
        state = buffer.get_interpolated(render_time)
    """

    def __init__(
        self,
        buffer_size: int = DEFAULT_INTERPOLATION_BUFFER_SIZE,
        extrapolation_limit: float = DEFAULT_EXTRAPOLATION_LIMIT,
        interpolation_mode: InterpolationMode = InterpolationMode.LINEAR,
    ) -> None:
        """
        Initialize the interpolation buffer.

        Args:
            buffer_size: Number of snapshots to buffer.
            extrapolation_limit: Maximum time to extrapolate (seconds).
            interpolation_mode: Default interpolation method.
        """
        self._buffer: deque[Snapshot] = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size
        self._extrapolation_limit = extrapolation_limit
        self._interpolation_mode = interpolation_mode
        self._last_extrapolated_time: float = 0.0

    @property
    def buffer_size(self) -> int:
        """Get the buffer size."""
        return self._buffer_size

    @property
    def snapshot_count(self) -> int:
        """Get number of snapshots in buffer."""
        return len(self._buffer)

    @property
    def interpolation_mode(self) -> InterpolationMode:
        """Get the current interpolation mode."""
        return self._interpolation_mode

    @interpolation_mode.setter
    def interpolation_mode(self, mode: InterpolationMode) -> None:
        """Set the interpolation mode."""
        self._interpolation_mode = mode

    @property
    def oldest_timestamp(self) -> Optional[float]:
        """Get the timestamp of the oldest snapshot."""
        if not self._buffer:
            return None
        return self._buffer[0].timestamp

    @property
    def newest_timestamp(self) -> Optional[float]:
        """Get the timestamp of the newest snapshot."""
        if not self._buffer:
            return None
        return self._buffer[-1].timestamp

    def push_snapshot(self, snapshot: Snapshot) -> None:
        """
        Add a new snapshot to the buffer.

        Snapshots should be added in timestamp order. Out-of-order
        snapshots are inserted at the correct position.

        Args:
            snapshot: The snapshot to add.
        """
        # Handle empty buffer
        if not self._buffer:
            self._buffer.append(snapshot)
            return

        # Find insertion point (maintain sorted order by timestamp)
        # Most common case: new snapshot is newest
        if snapshot.timestamp >= self._buffer[-1].timestamp:
            self._buffer.append(snapshot)
        else:
            # Find correct position
            for i, existing in enumerate(self._buffer):
                if snapshot.timestamp < existing.timestamp:
                    # Insert at position i
                    temp = list(self._buffer)
                    temp.insert(i, snapshot)
                    # Maintain max size
                    if len(temp) > self._buffer_size:
                        temp.pop(0)
                    self._buffer = deque(temp, maxlen=self._buffer_size)
                    return
            self._buffer.append(snapshot)

    def get_interpolated(
        self,
        render_time: float,
        mode: Optional[InterpolationMode] = None,
    ) -> Optional[InterpolatedState]:
        """
        Get interpolated state at the given render time.

        Args:
            render_time: The timestamp to interpolate to.
            mode: Optional override for interpolation mode.

        Returns:
            Interpolated state, or None if insufficient data.
        """
        if len(self._buffer) < 2:
            # Not enough snapshots - return latest if available
            if self._buffer:
                snap = self._buffer[-1]
                return InterpolatedState(
                    position=snap.position,
                    rotation=snap.rotation,
                    velocity=snap.velocity,
                    timestamp=render_time,
                    is_extrapolated=True,
                    extrapolation_time=render_time - snap.timestamp,
                )
            return None

        mode = mode or self._interpolation_mode

        # Find the two snapshots to interpolate between
        before: Optional[Snapshot] = None
        after: Optional[Snapshot] = None

        for i, snapshot in enumerate(self._buffer):
            if snapshot.timestamp > render_time:
                after = snapshot
                if i > 0:
                    before = self._buffer[i - 1]
                break
            before = snapshot

        # Handle edge cases
        if before is None:
            # render_time is before all snapshots - use earliest
            snap = self._buffer[0]
            return InterpolatedState(
                position=snap.position,
                rotation=snap.rotation,
                velocity=snap.velocity,
                timestamp=render_time,
                is_extrapolated=True,
                extrapolation_time=snap.timestamp - render_time,
            )

        if after is None:
            # render_time is after all snapshots - extrapolate
            return self._extrapolate(before, render_time)

        # Interpolate between before and after
        return self._interpolate(before, after, render_time, mode)

    def _interpolate(
        self,
        before: Snapshot,
        after: Snapshot,
        render_time: float,
        mode: InterpolationMode,
    ) -> InterpolatedState:
        """Interpolate between two snapshots."""
        # Calculate interpolation factor
        duration = after.timestamp - before.timestamp
        if duration <= 0:
            t = 0.5
        else:
            t = (render_time - before.timestamp) / duration

        t = max(0.0, min(1.0, t))

        # Interpolate position
        if mode == InterpolationMode.HERMITE:
            position = hermite_interpolate(
                before.position,
                after.position,
                before.velocity,
                after.velocity,
                t,
                duration,
            )
        else:  # LINEAR or default
            position = lerp_position(before.position, after.position, t)

        # Interpolate rotation
        rotation = None
        if before.rotation and after.rotation:
            rotation = slerp_rotation(before.rotation, after.rotation, t)
        elif after.rotation:
            rotation = after.rotation
        elif before.rotation:
            rotation = before.rotation

        # Interpolate velocity
        velocity = lerp_position(before.velocity, after.velocity, t)

        return InterpolatedState(
            position=position,
            rotation=rotation,
            velocity=velocity,
            timestamp=render_time,
            is_extrapolated=False,
            extrapolation_time=0.0,
        )

    def _extrapolate(
        self,
        last_snapshot: Snapshot,
        render_time: float,
    ) -> InterpolatedState:
        """Extrapolate beyond the latest snapshot."""
        dt = render_time - last_snapshot.timestamp

        # Clamp extrapolation time
        if dt > self._extrapolation_limit:
            dt = self._extrapolation_limit

        # Simple linear extrapolation using velocity
        position = (
            last_snapshot.position[0] + last_snapshot.velocity[0] * dt,
            last_snapshot.position[1] + last_snapshot.velocity[1] * dt,
            last_snapshot.position[2] + last_snapshot.velocity[2] * dt,
        )

        self._last_extrapolated_time = dt

        return InterpolatedState(
            position=position,
            rotation=last_snapshot.rotation,
            velocity=last_snapshot.velocity,
            timestamp=render_time,
            is_extrapolated=True,
            extrapolation_time=dt,
        )

    def clear(self) -> None:
        """Clear all buffered snapshots."""
        self._buffer.clear()

    def get_latest(self) -> Optional[Snapshot]:
        """Get the most recent snapshot."""
        if self._buffer:
            return self._buffer[-1]
        return None

    def get_buffer_time_range(self) -> Tuple[float, float]:
        """
        Get the time range covered by buffered snapshots.

        Returns:
            Tuple of (oldest_time, newest_time), or (0, 0) if empty.
        """
        if not self._buffer:
            return (0.0, 0.0)
        return (self._buffer[0].timestamp, self._buffer[-1].timestamp)


class EntityInterpolator:
    """
    Per-entity interpolation manager.

    Manages interpolation state for a single networked entity,
    including visual smoothing and interpolation delay calculation.
    """

    def __init__(
        self,
        entity_id: int,
        buffer_size: int = DEFAULT_INTERPOLATION_BUFFER_SIZE,
        interpolation_delay: float = DEFAULT_ENTITY_INTERPOLATION_DELAY,
    ) -> None:
        """
        Initialize the entity interpolator.

        Args:
            entity_id: Unique identifier for this entity.
            buffer_size: Number of snapshots to buffer.
            interpolation_delay: Fixed delay for interpolation (seconds).
        """
        self.entity_id = entity_id
        self._buffer = InterpolationBuffer(buffer_size=buffer_size)
        self._interpolation_delay = interpolation_delay
        self._current_state: Optional[InterpolatedState] = None
        self._server_time_offset: float = 0.0

    @property
    def interpolation_delay(self) -> float:
        """Get the interpolation delay."""
        return self._interpolation_delay

    @interpolation_delay.setter
    def interpolation_delay(self, delay: float) -> None:
        """Set the interpolation delay."""
        self._interpolation_delay = max(0.0, delay)

    def add_snapshot(self, snapshot: Snapshot) -> None:
        """Add a network snapshot for this entity."""
        self._buffer.push_snapshot(snapshot)

    def update(self, server_time: float) -> Optional[InterpolatedState]:
        """
        Update and return the interpolated state.

        Args:
            server_time: Current server time estimate.

        Returns:
            The interpolated state for rendering.
        """
        render_time = server_time - self._interpolation_delay
        self._current_state = self._buffer.get_interpolated(render_time)
        return self._current_state

    def get_position(self) -> Optional[Vector3]:
        """Get current interpolated position."""
        if self._current_state:
            return self._current_state.position
        return None

    def get_rotation(self) -> Optional[Quaternion]:
        """Get current interpolated rotation."""
        if self._current_state:
            return self._current_state.rotation
        return None

    def is_extrapolating(self) -> bool:
        """Check if currently extrapolating."""
        if self._current_state:
            return self._current_state.is_extrapolated
        return False
