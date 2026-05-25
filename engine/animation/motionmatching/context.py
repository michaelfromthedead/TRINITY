"""
Motion Matching Context - Runtime motion matching controller.

This module provides the runtime context for motion matching:
- MotionContext: Holds current state, desired trajectory, contacts
- DesiredTrajectory: Future positions and facings from player input
- MotionMatchingController: Main controller for motion matching
- Idle detection and state management

Usage:
    from engine.animation.motionmatching.context import (
        MotionMatchingController, MotionContext, DesiredTrajectory
    )

    # Create controller
    controller = MotionMatchingController(database, config)

    # Update each frame
    pose = controller.update(input_direction, dt)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import numpy as np

from engine.animation.motionmatching.database import (
    DatabaseEntry,
    MotionDatabase,
)
from engine.animation.motionmatching.features import (
    FeatureConfig,
    FeatureExtractor,
    FeatureSet,
    FootContact,
    TrajectoryPoint,
)
from engine.animation.motionmatching.search import (
    MotionSearch,
    SearchConfig,
    SearchMethod,
    SearchResult,
)
from engine.animation.motionmatching.transition import (
    BlendMode,
    MotionTransition,
    Pose,
    TransitionConfig,
)


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================


class ControllerState(Enum):
    """States for the motion matching controller."""
    IDLE = auto()           # Character is stationary
    MOVING = auto()         # Character is in locomotion
    TRANSITIONING = auto()  # Blending between clips
    STOPPED = auto()        # Controller is not running


# Import centralized config
from engine.animation.motionmatching.config import (
    DEFAULT_CONTROLLER_TIMING,
    DEFAULT_SEARCH_PARAMS,
    DEFAULT_IDLE_DETECTION,
    DEFAULT_FEATURE_WEIGHTS,
    DEFAULT_TRAJECTORY_TIMES,
)

# Default timing parameters from config
DEFAULT_MIN_TIME_IN_CLIP = DEFAULT_CONTROLLER_TIMING.min_time_in_clip
DEFAULT_SEARCH_INTERVAL = DEFAULT_CONTROLLER_TIMING.search_interval
DEFAULT_COST_IMPROVEMENT_THRESHOLD = DEFAULT_SEARCH_PARAMS.cost_improvement_threshold
DEFAULT_IDLE_VELOCITY_THRESHOLD = DEFAULT_IDLE_DETECTION.velocity_threshold


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class DesiredTrajectory:
    """Desired future trajectory from player input.

    Attributes:
        points: List of future trajectory points
        desired_speed: Target movement speed
        desired_facing: Target facing direction (radians)
        is_stationary: Whether the character should be stationary
    """
    points: List[TrajectoryPoint] = field(default_factory=list)
    desired_speed: float = 0.0
    desired_facing: float = 0.0
    is_stationary: bool = True

    @classmethod
    def from_input(
        cls,
        direction: np.ndarray,
        speed: float,
        current_position: np.ndarray,
        current_facing: float,
        trajectory_times: List[float],
        turn_rate: float = 10.0,
    ) -> DesiredTrajectory:
        """Create trajectory from input direction.

        Args:
            direction: Input direction (2D or 3D, normalized)
            speed: Desired speed
            current_position: Current world position
            current_facing: Current facing angle (radians)
            trajectory_times: Time points for trajectory
            turn_rate: Turn rate in radians per second

        Returns:
            DesiredTrajectory
        """
        is_stationary = speed < DEFAULT_IDLE_VELOCITY_THRESHOLD

        if is_stationary:
            # Stationary trajectory - stay in place
            points = [
                TrajectoryPoint(
                    time_offset=t,
                    position=current_position.copy(),
                    facing=current_facing,
                    velocity=np.zeros(3, dtype=np.float32),
                )
                for t in trajectory_times
            ]
            return cls(
                points=points,
                desired_speed=0.0,
                desired_facing=current_facing,
                is_stationary=True,
            )

        # Compute target facing from direction
        if len(direction) >= 2 and (direction[0] != 0 or direction[1] != 0):
            target_facing = math.atan2(direction[1], direction[0])
        else:
            target_facing = current_facing

        # Build trajectory points
        points = []
        position = current_position.copy()
        facing = current_facing

        for t in trajectory_times:
            # Interpolate facing toward target
            facing_diff = _normalize_angle(target_facing - facing)
            max_turn = turn_rate * t
            if abs(facing_diff) > max_turn:
                facing = facing + max_turn * np.sign(facing_diff)
            else:
                facing = target_facing

            # Move in facing direction
            velocity = np.array([
                math.cos(facing) * speed,
                0.0,  # Assuming Y is up
                math.sin(facing) * speed,
            ], dtype=np.float32)

            position = current_position + velocity * t

            points.append(TrajectoryPoint(
                time_offset=t,
                position=position,
                facing=facing,
                velocity=velocity,
            ))

        return cls(
            points=points,
            desired_speed=speed,
            desired_facing=target_facing,
            is_stationary=False,
        )


@dataclass
class MotionContext:
    """Runtime context for motion matching.

    Attributes:
        current_clip_index: Index of current animation clip
        current_frame: Current frame in the clip
        current_time: Time into current clip (seconds)
        current_pose: Current output pose
        current_entry: Current database entry
        desired_trajectory: Desired future trajectory
        foot_contacts: Current foot contact states
        state: Current controller state
        last_search_time: Time of last search
        time_since_transition: Time since last transition
        current_search_cost: Cost of current match
    """
    current_clip_index: int = -1
    current_frame: int = 0
    current_time: float = 0.0
    current_pose: Optional[Pose] = None
    current_entry: Optional[DatabaseEntry] = None

    desired_trajectory: DesiredTrajectory = field(default_factory=DesiredTrajectory)
    foot_contacts: FootContact = field(default_factory=FootContact)

    state: ControllerState = ControllerState.STOPPED
    last_search_time: float = 0.0
    time_since_transition: float = 0.0
    current_search_cost: float = float('inf')

    def advance_frame(self, dt: float, frame_rate: float) -> None:
        """Advance to next frame.

        Args:
            dt: Time step in seconds
            frame_rate: Animation frame rate
        """
        self.current_time += dt
        self.current_frame = int(self.current_time * frame_rate)
        self.time_since_transition += dt


@dataclass
class ControllerConfig:
    """Configuration for motion matching controller.

    Attributes:
        min_time_in_clip: Minimum time before allowing transition
        search_interval: Time between database searches
        cost_improvement_threshold: Minimum improvement to trigger transition
        idle_velocity_threshold: Velocity threshold for idle detection
        trajectory_times: Time points for trajectory prediction
        position_weight: Weight for position matching
        velocity_weight: Weight for velocity matching
        trajectory_weight: Weight for trajectory matching
        contact_weight: Weight for foot contact matching
        transition_config: Configuration for transitions
        search_method: Search method to use
    """
    min_time_in_clip: float = DEFAULT_MIN_TIME_IN_CLIP
    search_interval: float = DEFAULT_SEARCH_INTERVAL
    cost_improvement_threshold: float = DEFAULT_COST_IMPROVEMENT_THRESHOLD
    idle_velocity_threshold: float = DEFAULT_IDLE_VELOCITY_THRESHOLD

    trajectory_times: List[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6]
    )

    position_weight: float = 1.0
    velocity_weight: float = 0.5
    trajectory_weight: float = 1.0
    contact_weight: float = 2.0

    transition_config: TransitionConfig = field(default_factory=TransitionConfig)
    search_method: SearchMethod = SearchMethod.BRUTE_FORCE


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi].

    Args:
        angle: Angle in radians

    Returns:
        Normalized angle
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


# =============================================================================
# MOTION MATCHING CONTROLLER
# =============================================================================


class MotionMatchingController:
    """Main controller for motion matching animation.

    Handles database search, transitions, and pose output based on
    player input and current state.
    """

    def __init__(
        self,
        database: MotionDatabase,
        config: Optional[ControllerConfig] = None,
        pose_provider: Optional[Callable[[int, int], Pose]] = None,
    ):
        """Initialize controller.

        Args:
            database: Motion database to search
            config: Controller configuration
            pose_provider: Optional function to get pose from (clip_idx, frame)
        """
        self.database = database
        self.config = config or ControllerConfig()
        self.pose_provider = pose_provider

        # Context
        self.context = MotionContext()

        # Feature extractor
        self._feature_config = FeatureConfig(
            trajectory_times=self.config.trajectory_times,
            position_weight=self.config.position_weight,
            velocity_weight=self.config.velocity_weight,
            trajectory_weight=self.config.trajectory_weight,
            contact_weight=self.config.contact_weight,
        )
        self._feature_extractor = FeatureExtractor(self._feature_config)

        # Search
        self._search = MotionSearch(
            database,
            method=self.config.search_method,
        )

        # Transition
        self._current_transition: Optional[MotionTransition] = None

        # Timing
        self._total_time = 0.0
        self._time_since_search = 0.0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> ControllerState:
        """Current controller state."""
        return self.context.state

    @property
    def is_idle(self) -> bool:
        """Whether character is idle."""
        return self.context.state == ControllerState.IDLE

    @property
    def is_transitioning(self) -> bool:
        """Whether a transition is in progress."""
        return self._current_transition is not None and not self._current_transition.is_complete

    @property
    def current_pose(self) -> Optional[Pose]:
        """Current output pose."""
        return self.context.current_pose

    @property
    def current_entry(self) -> Optional[DatabaseEntry]:
        """Current database entry."""
        return self.context.current_entry

    # -------------------------------------------------------------------------
    # Main Update
    # -------------------------------------------------------------------------

    def start(self, initial_tags: Optional[set] = None) -> None:
        """Start the controller.

        Args:
            initial_tags: Optional tags to filter initial search
        """
        self.context.state = ControllerState.IDLE
        self._total_time = 0.0
        self._time_since_search = 0.0

        # Find initial pose
        if self.database.entry_count > 0:
            # Search for idle pose
            search_config = SearchConfig(
                max_results=1,
                required_tags=initial_tags,
            )

            # Build idle query
            idle_trajectory = DesiredTrajectory.from_input(
                direction=np.zeros(3),
                speed=0.0,
                current_position=np.zeros(3),
                current_facing=0.0,
                trajectory_times=self.config.trajectory_times,
            )

            query = self._build_query(idle_trajectory)
            results = self._search.search(query, search_config)

            if results:
                self._set_current_entry(results[0].entry, results[0].entry_index)
                self.context.current_search_cost = results[0].cost

    def stop(self) -> None:
        """Stop the controller."""
        self.context.state = ControllerState.STOPPED
        self._current_transition = None

    def update(
        self,
        input_direction: Union[np.ndarray, Tuple[float, float], Tuple[float, float, float]],
        dt: float,
        desired_speed: Optional[float] = None,
    ) -> Optional[Pose]:
        """Update controller and get output pose.

        Args:
            input_direction: Movement direction (2D or 3D)
            dt: Time step in seconds
            desired_speed: Optional override for desired speed

        Returns:
            Output pose or None if stopped
        """
        if self.context.state == ControllerState.STOPPED:
            return None

        self._total_time += dt
        self._time_since_search += dt

        # Normalize input direction
        direction = np.asarray(input_direction, dtype=np.float32)
        if len(direction) == 2:
            direction = np.array([direction[0], 0.0, direction[1]], dtype=np.float32)

        dir_length = np.linalg.norm(direction)
        if dir_length > 1e-6:
            direction = direction / dir_length

        # Compute desired speed
        if desired_speed is None:
            desired_speed = dir_length * 5.0  # Default walk/run speed scaling

        # Update desired trajectory
        current_position = self._get_current_position()
        current_facing = self._get_current_facing()

        self.context.desired_trajectory = DesiredTrajectory.from_input(
            direction=direction,
            speed=desired_speed,
            current_position=current_position,
            current_facing=current_facing,
            trajectory_times=self.config.trajectory_times,
        )

        # Detect idle state
        if self.context.desired_trajectory.is_stationary:
            if self.context.state == ControllerState.MOVING:
                self.context.state = ControllerState.IDLE
        else:
            if self.context.state == ControllerState.IDLE:
                self.context.state = ControllerState.MOVING

        # Update transition if active
        if self._current_transition is not None:
            target_pose = self._get_current_target_pose()
            self.context.current_pose = self._current_transition.update(dt, target_pose)

            if self._current_transition.is_complete:
                self._current_transition = None
                self.context.state = (
                    ControllerState.IDLE if self.context.desired_trajectory.is_stationary
                    else ControllerState.MOVING
                )

        # Search for better match
        if self._should_search():
            self._perform_search()

        # Advance current frame
        self._advance_playback(dt)

        # Update output pose
        if self._current_transition is None:
            self.context.current_pose = self._get_current_target_pose()

        return self.context.current_pose

    # -------------------------------------------------------------------------
    # Search Logic
    # -------------------------------------------------------------------------

    def _should_search(self) -> bool:
        """Determine if we should search for a new match."""
        # Don't search during transition
        if self._current_transition is not None:
            return False

        # Check time since last search
        if self._time_since_search < self.config.search_interval:
            return False

        # Check minimum time in current clip
        if self.context.time_since_transition < self.config.min_time_in_clip:
            return False

        return True

    def _perform_search(self) -> None:
        """Search for a better matching frame."""
        self._time_since_search = 0.0

        # Build query from current state
        query = self._build_query(self.context.desired_trajectory)

        # Configure search
        search_config = SearchConfig(
            max_results=3,
            method=self.config.search_method,
            only_transition_candidates=True,
        )

        # Exclude frames too close to current position
        if self.context.current_entry is not None:
            min_frame = self.context.current_frame - 5
            max_frame = self.context.current_frame + 10
            search_config.exclude_frames_range = (
                self.context.current_clip_index,
                min_frame,
                max_frame,
            )

        # Search
        results = self._search.search(query, search_config)

        if not results:
            return

        best_result = results[0]

        # Check if improvement is worth transitioning
        cost_improvement = self.context.current_search_cost - best_result.cost
        relative_improvement = cost_improvement / max(self.context.current_search_cost, 1e-6)

        if relative_improvement > self.config.cost_improvement_threshold:
            self._start_transition(best_result)

    def _build_query(self, trajectory: DesiredTrajectory) -> FeatureSet:
        """Build query features from desired trajectory.

        Args:
            trajectory: Desired trajectory

        Returns:
            FeatureSet for search
        """
        # Get current pose features
        if self.context.current_entry is not None and self.pose_provider:
            # Extract features from current state
            bone_data = self._get_current_bone_data()
            return self._feature_extractor.extract_from_pose(
                bone_data=bone_data,
                trajectory=trajectory.points,
                foot_contacts=self.context.foot_contacts,
            )

        # Build from trajectory alone
        return self._feature_extractor.extract_from_pose(
            bone_data={},
            trajectory=trajectory.points,
            foot_contacts=self.context.foot_contacts,
        )

    def _start_transition(self, result: SearchResult) -> None:
        """Start transition to a new entry.

        Args:
            result: Search result to transition to
        """
        if self.context.current_entry is None:
            # No current entry - just set directly
            self._set_current_entry(result.entry, result.entry_index)
            return

        # Create transition
        self._current_transition = MotionTransition(
            from_entry=self.context.current_entry,
            to_entry=result.entry,
            config=self.config.transition_config,
        )

        # Initialize with poses
        from_pose = self.context.current_pose or Pose()
        to_pose = self._get_pose_for_entry(result.entry)

        self._current_transition.initialize(from_pose, to_pose)

        # Update context
        self._set_current_entry(result.entry, result.entry_index)
        self.context.current_search_cost = result.cost
        self.context.time_since_transition = 0.0

    def _set_current_entry(self, entry: DatabaseEntry, entry_index: int) -> None:
        """Set the current entry.

        Args:
            entry: New current entry
            entry_index: Index in database
        """
        self.context.current_entry = entry
        self.context.current_clip_index = entry.clip_index
        self.context.current_frame = entry.frame
        self.context.current_time = entry.frame / self._get_frame_rate(entry.clip_index)

    # -------------------------------------------------------------------------
    # Playback
    # -------------------------------------------------------------------------

    def _advance_playback(self, dt: float) -> None:
        """Advance playback in current clip.

        Args:
            dt: Time step
        """
        if self.context.current_entry is None:
            return

        clip_metadata = self.database.get_clip_metadata(self.context.current_clip_index)
        if clip_metadata is None:
            return

        frame_rate = clip_metadata.frame_rate
        self.context.current_time += dt
        self.context.current_frame = int(self.context.current_time * frame_rate)
        self.context.time_since_transition += dt

        # Handle clip end
        if self.context.current_frame >= clip_metadata.frame_count - 1:
            if clip_metadata.is_looping:
                self.context.current_time = 0.0
                self.context.current_frame = 0
            else:
                # Force search for new clip
                self.context.current_search_cost = float('inf')

    def _get_frame_rate(self, clip_index: int) -> float:
        """Get frame rate for a clip."""
        clip_metadata = self.database.get_clip_metadata(clip_index)
        return clip_metadata.frame_rate if clip_metadata else 30.0

    # -------------------------------------------------------------------------
    # Pose Access
    # -------------------------------------------------------------------------

    def _get_current_target_pose(self) -> Pose:
        """Get the current target pose without transition blending."""
        if self.context.current_entry is None:
            return Pose()

        return self._get_pose_for_entry(self.context.current_entry)

    def _get_pose_for_entry(self, entry: DatabaseEntry) -> Pose:
        """Get pose for a database entry.

        Args:
            entry: Database entry

        Returns:
            Pose for entry
        """
        if self.pose_provider:
            return self.pose_provider(entry.clip_index, entry.frame)

        # Return empty pose if no provider
        return Pose()

    def _get_current_position(self) -> np.ndarray:
        """Get current character position."""
        if self.context.current_pose:
            return self.context.current_pose.root_position.copy()
        return np.zeros(3, dtype=np.float32)

    def _get_current_facing(self) -> float:
        """Get current character facing angle."""
        if self.context.current_pose:
            # Extract yaw from root rotation
            q = self.context.current_pose.root_rotation
            siny_cosp = 2.0 * (q[3] * q[1] + q[2] * q[0])
            cosy_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
            return math.atan2(siny_cosp, cosy_cosp)
        return 0.0

    def _get_current_bone_data(self) -> Dict[str, Any]:
        """Get current bone data for feature extraction."""
        # This would extract bone positions/velocities from current pose
        # Implementation depends on pose format
        return {}

    # -------------------------------------------------------------------------
    # Debug / State Access
    # -------------------------------------------------------------------------

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about controller state.

        Returns:
            Dictionary of debug values
        """
        return {
            'state': self.context.state.name,
            'clip_index': self.context.current_clip_index,
            'frame': self.context.current_frame,
            'time': self.context.current_time,
            'search_cost': self.context.current_search_cost,
            'time_since_transition': self.context.time_since_transition,
            'is_transitioning': self.is_transitioning,
            'desired_speed': self.context.desired_trajectory.desired_speed,
            'desired_facing': self.context.desired_trajectory.desired_facing,
        }

    def force_transition(self, entry_index: int) -> bool:
        """Force a transition to a specific entry.

        Args:
            entry_index: Index of entry to transition to

        Returns:
            True if transition was started
        """
        entry = self.database.get_entry(entry_index)
        if entry is None:
            return False

        result = SearchResult(
            entry=entry,
            entry_index=entry_index,
            cost=0.0,
        )
        self._start_transition(result)
        return True


# =============================================================================
# IDLE DETECTION
# =============================================================================


class IdleDetector:
    """Detects when character has transitioned to idle state.

    Uses velocity history and input to determine when character
    should enter idle animation.
    """

    def __init__(
        self,
        velocity_threshold: float = DEFAULT_IDLE_VELOCITY_THRESHOLD,
        hold_time: float = 0.1,
    ):
        """Initialize detector.

        Args:
            velocity_threshold: Velocity below this is considered idle
            hold_time: Time velocity must be low before triggering idle
        """
        self.velocity_threshold = velocity_threshold
        self.hold_time = hold_time

        self._time_below_threshold = 0.0
        self._is_idle = True

    def update(
        self,
        current_velocity: float,
        input_velocity: float,
        dt: float,
    ) -> bool:
        """Update idle detection.

        Args:
            current_velocity: Current character velocity
            input_velocity: Desired velocity from input
            dt: Time step

        Returns:
            True if character should be idle
        """
        # Check both current and input velocity
        is_low_velocity = (
            current_velocity < self.velocity_threshold and
            input_velocity < self.velocity_threshold
        )

        if is_low_velocity:
            self._time_below_threshold += dt
            if self._time_below_threshold >= self.hold_time:
                self._is_idle = True
        else:
            self._time_below_threshold = 0.0
            self._is_idle = False

        return self._is_idle

    @property
    def is_idle(self) -> bool:
        """Current idle state."""
        return self._is_idle

    def reset(self) -> None:
        """Reset detector state."""
        self._time_below_threshold = 0.0
        self._is_idle = True


# =============================================================================
# TRAJECTORY BUILDER
# =============================================================================


class TrajectoryBuilder:
    """Builds desired trajectory from various input sources.

    Supports gamepad input, keyboard input, and direct velocity commands.
    """

    def __init__(
        self,
        trajectory_times: List[float],
        max_speed: float = 5.0,
        turn_rate: float = 10.0,
    ):
        """Initialize builder.

        Args:
            trajectory_times: Time points for trajectory
            max_speed: Maximum movement speed
            turn_rate: Turn rate in radians per second
        """
        self.trajectory_times = trajectory_times
        self.max_speed = max_speed
        self.turn_rate = turn_rate

        self._current_position = np.zeros(3, dtype=np.float32)
        self._current_facing = 0.0

    def set_current_state(
        self,
        position: np.ndarray,
        facing: float,
    ) -> None:
        """Set current character state.

        Args:
            position: Current world position
            facing: Current facing angle (radians)
        """
        self._current_position = np.asarray(position, dtype=np.float32)
        self._current_facing = facing

    def build_from_gamepad(
        self,
        stick_x: float,
        stick_y: float,
    ) -> DesiredTrajectory:
        """Build trajectory from gamepad stick input.

        Args:
            stick_x: Left stick X (-1 to 1)
            stick_y: Left stick Y (-1 to 1)

        Returns:
            DesiredTrajectory
        """
        # Compute direction and magnitude
        magnitude = min(1.0, math.sqrt(stick_x * stick_x + stick_y * stick_y))
        speed = magnitude * self.max_speed

        if magnitude > 0.1:
            direction = np.array([stick_x, 0.0, stick_y], dtype=np.float32) / magnitude
        else:
            direction = np.zeros(3, dtype=np.float32)

        return DesiredTrajectory.from_input(
            direction=direction,
            speed=speed,
            current_position=self._current_position,
            current_facing=self._current_facing,
            trajectory_times=self.trajectory_times,
            turn_rate=self.turn_rate,
        )

    def build_from_keyboard(
        self,
        forward: bool,
        backward: bool,
        left: bool,
        right: bool,
        running: bool = False,
    ) -> DesiredTrajectory:
        """Build trajectory from keyboard input.

        Args:
            forward: Forward key pressed
            backward: Backward key pressed
            left: Left key pressed
            right: Right key pressed
            running: Run modifier active

        Returns:
            DesiredTrajectory
        """
        # Build direction vector
        direction = np.zeros(3, dtype=np.float32)

        if forward:
            direction[2] += 1.0
        if backward:
            direction[2] -= 1.0
        if right:
            direction[0] += 1.0
        if left:
            direction[0] -= 1.0

        # Normalize
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length

        # Compute speed
        if length > 0:
            speed = self.max_speed if running else self.max_speed * 0.5
        else:
            speed = 0.0

        return DesiredTrajectory.from_input(
            direction=direction,
            speed=speed,
            current_position=self._current_position,
            current_facing=self._current_facing,
            trajectory_times=self.trajectory_times,
            turn_rate=self.turn_rate,
        )

    def build_from_velocity(
        self,
        velocity: np.ndarray,
    ) -> DesiredTrajectory:
        """Build trajectory from direct velocity command.

        Args:
            velocity: Desired velocity vector (3D)

        Returns:
            DesiredTrajectory
        """
        velocity = np.asarray(velocity, dtype=np.float32)
        speed = float(np.linalg.norm(velocity))

        if speed > 0:
            direction = velocity / speed
        else:
            direction = np.zeros(3, dtype=np.float32)

        return DesiredTrajectory.from_input(
            direction=direction,
            speed=speed,
            current_position=self._current_position,
            current_facing=self._current_facing,
            trajectory_times=self.trajectory_times,
            turn_rate=self.turn_rate,
        )
