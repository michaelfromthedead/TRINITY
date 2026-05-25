"""Root motion extraction and application.

This module provides functionality for:
- Extracting root bone movement from animation clips
- Applying root motion deltas to entity transforms
- Accumulating continuous motion across frames
- Handling looping clips with wrap-around deltas
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.mat import Mat4
from engine.core.math.transform import Transform, RigidTransform
from engine.core.constants import MATH_EPSILON

from engine.animation.skeletal.constants import (
    DEFAULT_ROOT_MOTION_SCALE,
    DEFAULT_ROTATION_SCALE,
    DEFAULT_GROUND_HEIGHT,
)

if TYPE_CHECKING:
    from engine.animation.skeletal.clip import AnimationClip
    from engine.animation.skeletal.pose import Pose


class RootMotionMode(Enum):
    """Root motion extraction modes."""
    IN_PLACE = auto()       # No root motion, animation plays in place
    EXTRACT_XZ = auto()     # Extract horizontal movement (XZ plane)
    EXTRACT_XYZ = auto()    # Extract full 3D movement
    EXTRACT_ROTATION = auto()  # Extract only rotation (yaw)
    EXTRACT_ALL = auto()    # Extract both movement and rotation


@dataclass
class RootMotionData:
    """Per-frame root motion deltas.

    Stores the change in root position and rotation between frames.

    Attributes:
        delta_positions: Position change per frame
        delta_rotations: Rotation change per frame
        frame_times: Time of each frame
        total_delta_position: Total position change over clip
        total_delta_rotation: Total rotation change over clip
    """
    delta_positions: List[Vec3] = field(default_factory=list)
    delta_rotations: List[Quat] = field(default_factory=list)
    frame_times: List[float] = field(default_factory=list)
    total_delta_position: Vec3 = field(default_factory=Vec3.zero)
    total_delta_rotation: Quat = field(default_factory=Quat.identity)

    @property
    def frame_count(self) -> int:
        return len(self.delta_positions)

    @property
    def duration(self) -> float:
        if not self.frame_times:
            return 0.0
        return self.frame_times[-1]

    def get_delta_at_time(self, time: float) -> Tuple[Vec3, Quat]:
        """Get interpolated delta at a specific time.

        Args:
            time: Time in seconds

        Returns:
            Tuple of (position_delta, rotation_delta)
        """
        if not self.frame_times or time <= 0:
            return Vec3.zero(), Quat.identity()

        if time >= self.duration:
            return self.total_delta_position, self.total_delta_rotation

        # Find the frame interval
        frame_idx = 0
        for i, ft in enumerate(self.frame_times):
            if ft > time:
                frame_idx = max(0, i - 1)
                break
        else:
            frame_idx = len(self.frame_times) - 1

        # Accumulate deltas up to frame_idx
        pos = Vec3.zero()
        rot = Quat.identity()

        for i in range(frame_idx):
            pos = pos + self.delta_positions[i]
            rot = self.delta_rotations[i] * rot

        # Interpolate within current frame if needed
        if frame_idx < len(self.frame_times) - 1:
            t0 = self.frame_times[frame_idx]
            t1 = self.frame_times[frame_idx + 1]
            if t1 > t0:
                t = (time - t0) / (t1 - t0)
                partial_pos = self.delta_positions[frame_idx] * t
                pos = pos + partial_pos
                # For rotation, slerp from identity to delta
                partial_rot = Quat.identity().slerp(self.delta_rotations[frame_idx], t)
                rot = partial_rot * rot

        return pos, rot

    def get_delta_between(self, start_time: float, end_time: float) -> Tuple[Vec3, Quat]:
        """Get root motion delta between two times.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Tuple of (position_delta, rotation_delta)
        """
        if end_time <= start_time:
            return Vec3.zero(), Quat.identity()

        # Get cumulative deltas at both times
        start_pos, start_rot = self.get_delta_at_time(start_time)
        end_pos, end_rot = self.get_delta_at_time(end_time)

        # Compute relative delta
        pos_delta = end_pos - start_pos
        rot_delta = start_rot.inverse() * end_rot

        return pos_delta, rot_delta


@dataclass
class RootBoneTransform:
    """Root bone transform for a single frame.

    Attributes:
        position: World position of root bone
        rotation: World rotation of root bone
        time: Frame time
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    rotation: Quat = field(default_factory=Quat.identity)
    time: float = 0.0


def _extract_yaw_rotation(rotation: Quat) -> Quat:
    """Extract only the Y-axis (yaw) rotation from a quaternion."""
    # Get euler angles
    pitch, yaw, roll = rotation.to_euler()
    # Create quaternion from yaw only
    return Quat.from_euler(0.0, yaw, 0.0)


def _project_to_xz(position: Vec3) -> Vec3:
    """Project position to XZ plane (zero out Y)."""
    return Vec3(position.x, 0.0, position.z)


def extract_root_motion(
    root_transforms: List[RootBoneTransform],
    mode: RootMotionMode = RootMotionMode.EXTRACT_XZ
) -> RootMotionData:
    """Extract root motion data from root bone transforms.

    Args:
        root_transforms: List of root bone transforms per frame
        mode: Extraction mode controlling what motion to capture

    Returns:
        RootMotionData containing per-frame deltas
    """
    if not root_transforms:
        return RootMotionData()

    if mode == RootMotionMode.IN_PLACE:
        # No motion extraction - return zeros
        return RootMotionData(
            delta_positions=[Vec3.zero()] * len(root_transforms),
            delta_rotations=[Quat.identity()] * len(root_transforms),
            frame_times=[rt.time for rt in root_transforms],
            total_delta_position=Vec3.zero(),
            total_delta_rotation=Quat.identity()
        )

    delta_positions = []
    delta_rotations = []
    frame_times = []

    total_pos = Vec3.zero()
    total_rot = Quat.identity()

    for i, rt in enumerate(root_transforms):
        frame_times.append(rt.time)

        if i == 0:
            # First frame has no delta
            delta_positions.append(Vec3.zero())
            delta_rotations.append(Quat.identity())
            continue

        prev = root_transforms[i - 1]

        # Compute position delta
        if mode == RootMotionMode.EXTRACT_XZ:
            pos_delta = _project_to_xz(rt.position - prev.position)
        elif mode in (RootMotionMode.EXTRACT_XYZ, RootMotionMode.EXTRACT_ALL):
            pos_delta = rt.position - prev.position
        elif mode == RootMotionMode.EXTRACT_ROTATION:
            pos_delta = Vec3.zero()
        else:
            pos_delta = Vec3.zero()

        # Compute rotation delta
        if mode in (RootMotionMode.EXTRACT_ROTATION, RootMotionMode.EXTRACT_ALL):
            # Only extract yaw rotation
            prev_yaw = _extract_yaw_rotation(prev.rotation)
            curr_yaw = _extract_yaw_rotation(rt.rotation)
            rot_delta = prev_yaw.inverse() * curr_yaw
        else:
            rot_delta = Quat.identity()

        delta_positions.append(pos_delta)
        delta_rotations.append(rot_delta)

        total_pos = total_pos + pos_delta
        total_rot = rot_delta * total_rot

    return RootMotionData(
        delta_positions=delta_positions,
        delta_rotations=delta_rotations,
        frame_times=frame_times,
        total_delta_position=total_pos,
        total_delta_rotation=total_rot
    )


def apply_root_motion(
    entity_transform: Transform,
    delta_position: Vec3,
    delta_rotation: Quat,
    dt: float = 1.0
) -> Transform:
    """Apply root motion delta to an entity transform.

    The delta is applied in the entity's local space - position
    delta is rotated by current facing, rotation is concatenated.

    Args:
        entity_transform: Current entity transform
        delta_position: Position change (in local/root space)
        delta_rotation: Rotation change
        dt: Optional time scale factor

    Returns:
        Updated entity transform
    """
    # Scale deltas by dt if needed (for partial frame updates)
    scaled_pos = delta_position * dt
    scaled_rot = Quat.identity().slerp(delta_rotation, dt) if dt != 1.0 else delta_rotation

    # Transform position delta from local to world space
    world_pos_delta = entity_transform.rotation.rotate_vector(scaled_pos)

    # Apply position
    new_position = entity_transform.translation + world_pos_delta

    # Apply rotation
    new_rotation = entity_transform.rotation * scaled_rot

    return Transform(
        translation=new_position,
        rotation=new_rotation.normalized(),
        scale=entity_transform.scale
    )


class RootMotionAccumulator:
    """Accumulates root motion over time for continuous playback.

    Handles motion accumulation across frames and properly wraps
    deltas when looping clips reach their end.

    Attributes:
        accumulated_position: Total position accumulated
        accumulated_rotation: Total rotation accumulated
        current_time: Current playback time
    """

    def __init__(
        self,
        root_motion_data: Optional[RootMotionData] = None,
        loop: bool = True
    ) -> None:
        self._root_motion_data = root_motion_data
        self._loop = loop
        self._accumulated_position = Vec3.zero()
        self._accumulated_rotation = Quat.identity()
        self._current_time = 0.0
        self._last_time = 0.0
        self._loop_count = 0

    @property
    def accumulated_position(self) -> Vec3:
        return self._accumulated_position

    @property
    def accumulated_rotation(self) -> Quat:
        return self._accumulated_rotation

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def loop_count(self) -> int:
        return self._loop_count

    def set_root_motion_data(self, data: RootMotionData) -> None:
        """Set or update the root motion data source."""
        self._root_motion_data = data

    def reset(self) -> None:
        """Reset accumulator to initial state."""
        self._accumulated_position = Vec3.zero()
        self._accumulated_rotation = Quat.identity()
        self._current_time = 0.0
        self._last_time = 0.0
        self._loop_count = 0

    def update(self, dt: float, playback_rate: float = 1.0) -> Tuple[Vec3, Quat]:
        """Update accumulator and return delta since last update.

        Args:
            dt: Time elapsed since last update
            playback_rate: Animation playback speed multiplier

        Returns:
            Tuple of (position_delta, rotation_delta) for this update
        """
        if self._root_motion_data is None:
            return Vec3.zero(), Quat.identity()

        duration = self._root_motion_data.duration
        if duration <= 0:
            return Vec3.zero(), Quat.identity()

        # Advance time
        time_delta = dt * playback_rate
        new_time = self._current_time + time_delta

        # Handle looping
        if self._loop and new_time >= duration:
            # Calculate motion before wrap
            pos_before, rot_before = self._root_motion_data.get_delta_between(
                self._current_time, duration
            )

            # Handle wrap delta at loop point
            wrap_pos = self._root_motion_data.total_delta_position
            wrap_rot = self._root_motion_data.total_delta_rotation

            # Calculate remaining time after wrap
            excess_time = new_time - duration
            loops = int(excess_time / duration) if duration > 0 else 0
            self._loop_count += 1 + loops

            # Calculate time within new loop
            new_time = excess_time - (loops * duration) if loops > 0 else excess_time

            # Get motion for the new loop portion
            pos_after, rot_after = self._root_motion_data.get_delta_between(0, new_time)

            # Combine all deltas
            # Transform after-loop motion by the accumulated rotation at loop point
            rotated_pos_after = wrap_rot.rotate_vector(pos_after)

            total_pos = pos_before + rotated_pos_after
            total_rot = rot_after * wrap_rot * rot_before

            # Add full loop deltas for any complete loops
            for _ in range(loops):
                total_pos = total_pos + wrap_rot.rotate_vector(wrap_pos)
                total_rot = wrap_rot * total_rot

            # Update accumulators
            self._accumulated_position = self._accumulated_position + total_pos
            self._accumulated_rotation = total_rot * self._accumulated_rotation

            self._last_time = self._current_time
            self._current_time = new_time

            return total_pos, total_rot

        elif not self._loop and new_time >= duration:
            # Clamp to end for non-looping
            pos_delta, rot_delta = self._root_motion_data.get_delta_between(
                self._current_time, duration
            )
            self._accumulated_position = self._accumulated_position + pos_delta
            self._accumulated_rotation = rot_delta * self._accumulated_rotation
            self._last_time = self._current_time
            self._current_time = duration
            return pos_delta, rot_delta

        else:
            # Normal case - within clip duration
            pos_delta, rot_delta = self._root_motion_data.get_delta_between(
                self._current_time, new_time
            )
            self._accumulated_position = self._accumulated_position + pos_delta
            self._accumulated_rotation = rot_delta * self._accumulated_rotation
            self._last_time = self._current_time
            self._current_time = new_time
            return pos_delta, rot_delta

    def seek(self, time: float) -> None:
        """Seek to a specific time, updating accumulator accordingly.

        Args:
            time: Target time to seek to
        """
        if self._root_motion_data is None:
            self._current_time = time
            return

        duration = self._root_motion_data.duration

        if self._loop and duration > 0:
            # Handle seeking in looping mode
            self._loop_count = int(time / duration)
            time = time - (self._loop_count * duration)

        # Calculate accumulated motion from start to seek point
        self._accumulated_position, self._accumulated_rotation = \
            self._root_motion_data.get_delta_at_time(time)

        # Add full loop contributions
        if self._loop_count > 0:
            wrap_pos = self._root_motion_data.total_delta_position
            wrap_rot = self._root_motion_data.total_delta_rotation

            for _ in range(self._loop_count):
                self._accumulated_position = \
                    self._accumulated_position + wrap_rot.rotate_vector(wrap_pos)
                self._accumulated_rotation = wrap_rot * self._accumulated_rotation

        self._current_time = time
        self._last_time = time

    def get_transform_at_time(
        self,
        time: float,
        base_transform: Optional[Transform] = None
    ) -> Transform:
        """Get the accumulated transform at a specific time.

        Args:
            time: Time to evaluate
            base_transform: Optional starting transform

        Returns:
            Transform with accumulated root motion applied
        """
        if base_transform is None:
            base_transform = Transform.identity()

        if self._root_motion_data is None:
            return base_transform

        pos, rot = self._root_motion_data.get_delta_at_time(time)

        return apply_root_motion(base_transform, pos, rot)


@dataclass
class RootMotionConfig:
    """Configuration for root motion extraction and application.

    Attributes:
        mode: Extraction mode
        scale: Scale factor for extracted motion
        rotation_scale: Scale factor for rotation (useful for partial turns)
        ground_height: Y value to clamp to (for grounded characters)
        clamp_to_ground: Whether to zero out vertical motion
    """
    mode: RootMotionMode = RootMotionMode.EXTRACT_XZ
    scale: float = DEFAULT_ROOT_MOTION_SCALE
    rotation_scale: float = DEFAULT_ROTATION_SCALE
    ground_height: float = DEFAULT_GROUND_HEIGHT
    clamp_to_ground: bool = True

    def apply_to_delta(self, pos: Vec3, rot: Quat) -> Tuple[Vec3, Quat]:
        """Apply configuration to a root motion delta.

        Args:
            pos: Position delta
            rot: Rotation delta

        Returns:
            Tuple of (scaled_position, scaled_rotation)
        """
        scaled_pos = pos * self.scale

        if self.clamp_to_ground:
            scaled_pos = Vec3(scaled_pos.x, 0.0, scaled_pos.z)

        # Scale rotation (interpolate toward identity for partial)
        if abs(self.rotation_scale - 1.0) > MATH_EPSILON:
            scaled_rot = Quat.identity().slerp(rot, self.rotation_scale)
        else:
            scaled_rot = rot

        return scaled_pos, scaled_rot


def blend_root_motion(
    motion_a: Tuple[Vec3, Quat],
    motion_b: Tuple[Vec3, Quat],
    weight: float
) -> Tuple[Vec3, Quat]:
    """Blend two root motion deltas.

    Args:
        motion_a: First motion (position, rotation)
        motion_b: Second motion (position, rotation)
        weight: Blend weight (0 = full A, 1 = full B)

    Returns:
        Blended motion tuple
    """
    pos_a, rot_a = motion_a
    pos_b, rot_b = motion_b

    blended_pos = pos_a.lerp(pos_b, weight)
    blended_rot = rot_a.slerp(rot_b, weight)

    return blended_pos, blended_rot


class RootMotionBlender:
    """Blends root motion from multiple animation sources.

    Used when blending between animations (transitions, blend trees)
    to produce coherent root motion.
    """

    def __init__(self) -> None:
        self._sources: List[Tuple[RootMotionAccumulator, float]] = []

    def add_source(self, accumulator: RootMotionAccumulator, weight: float) -> None:
        """Add a root motion source with weight.

        Args:
            accumulator: Root motion accumulator
            weight: Blend weight for this source
        """
        if weight > MATH_EPSILON:
            self._sources.append((accumulator, weight))

    def clear(self) -> None:
        """Clear all sources."""
        self._sources.clear()

    def update(self, dt: float, playback_rate: float = 1.0) -> Tuple[Vec3, Quat]:
        """Update all sources and blend results.

        Args:
            dt: Time delta
            playback_rate: Global playback rate

        Returns:
            Blended root motion delta
        """
        if not self._sources:
            return Vec3.zero(), Quat.identity()

        # Normalize weights
        total_weight = sum(w for _, w in self._sources)
        if total_weight < MATH_EPSILON:
            return Vec3.zero(), Quat.identity()

        blended_pos = Vec3.zero()
        blended_rot = Quat(0, 0, 0, 0)  # For accumulation
        first = True

        for accumulator, weight in self._sources:
            normalized_weight = weight / total_weight
            pos, rot = accumulator.update(dt, playback_rate)

            blended_pos = blended_pos + pos * normalized_weight

            if first:
                blended_rot = Quat(
                    rot.x * normalized_weight,
                    rot.y * normalized_weight,
                    rot.z * normalized_weight,
                    rot.w * normalized_weight
                )
                first = False
            else:
                # Handle quaternion antipodality for blending
                if blended_rot.dot(rot) < 0:
                    rot = Quat(-rot.x, -rot.y, -rot.z, -rot.w)
                blended_rot = Quat(
                    blended_rot.x + rot.x * normalized_weight,
                    blended_rot.y + rot.y * normalized_weight,
                    blended_rot.z + rot.z * normalized_weight,
                    blended_rot.w + rot.w * normalized_weight
                )

        return blended_pos, blended_rot.normalized()
