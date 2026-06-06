"""Motion Matching System — ECS system for motion matching animation (T-AN-9.7).

This system implements the runtime motion matching pipeline, replacing standard
state machines for MM-enabled characters. Each frame it:
1. Computes the desired trajectory (position, facing at T+0.2, T+0.5, T+1.0s)
2. Extracts pose features using FeatureExtractor (from T-AN-6.2)
3. Searches the database for the best match (from T-AN-6.3)
4. Applies inertialization transition (from T-AN-6.4)
5. Enforces budget_ms limits for performance

Key Features:
- @system(phase="animation") annotation for ECS scheduling
- Budget enforcement (motion_matching_budget_ms)
- Fallback to state machine when budget exceeded or no match found
- Statistics tracking (queries, matches, budget usage)
- Context modifier support (from T-AN-6.5)
- Inertialization blending for smooth transitions

Dependencies:
- engine.animation.motionmatching.database: MotionDatabase (T-AN-6.1)
- engine.animation.motionmatching.features: FeatureExtractor (T-AN-6.2)
- engine.animation.motionmatching.search: MotionSearch (T-AN-6.3)
- engine.animation.motionmatching.transition: InertializationBlender (T-AN-6.4)
- engine.animation.motionmatching.context: MotionMatchingController (T-AN-6.5)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)
import numpy as np

from engine.animation.config import ANIMATION_SYSTEM_CONFIG
from engine.animation.motionmatching.config import (
    DEFAULT_FEATURE_WEIGHTS,
    DEFAULT_SEARCH_PARAMS,
    DEFAULT_TRANSITION_PARAMS,
    DEFAULT_CONTROLLER_TIMING,
    DEFAULT_IDLE_DETECTION,
    DEFAULT_TRAJECTORY_TIMES,
)
from engine.animation.motionmatching.database import (
    ClipMetadata,
    DatabaseEntry,
    MotionDatabase,
)
from engine.animation.motionmatching.features import (
    BoneData,
    FeatureConfig,
    FeatureExtractor,
    FeatureSet,
    FeatureType,
    FootContact,
    TrajectoryPoint,
)
from engine.animation.motionmatching.search import (
    MotionSearch,
    SearchConfig,
    SearchMethod,
    SearchResult,
    compute_cost,
)
from engine.animation.motionmatching.transition import (
    BlendMode,
    BoneTransform,
    InertializationBlender,
    MotionTransition,
    Pose,
    TransitionConfig,
    quaternion_slerp,
)
from engine.animation.motionmatching.context import (
    ControllerConfig,
    ControllerState,
    DesiredTrajectory,
    IdleDetector,
    MotionContext,
    MotionMatchingController,
    TrajectoryBuilder,
)

if TYPE_CHECKING:
    from engine.core.ecs import Entity, World
    from engine.core.tasks.scheduler import TaskScheduler, TaskHandle


# =============================================================================
# SYSTEM DECORATOR
# =============================================================================


def system(
    phase: str = "update",
    order: int = 0,
    reads: Optional[Tuple[str, ...]] = None,
    writes: Optional[Tuple[str, ...]] = None,
) -> Callable:
    """Decorator to mark a class as an ECS system with phase scheduling.

    Args:
        phase: Frame phase for execution ("animation", "update", "render", etc.)
        order: Execution order within phase (lower = earlier)
        reads: Component types this system reads from
        writes: Component types this system writes to

    Returns:
        Decorated class with system metadata.
    """
    def decorator(cls: type) -> type:
        cls._system_phase = phase
        cls._system_order = order
        cls._system_reads = reads or ()
        cls._system_writes = writes or ()
        return cls
    return decorator


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class FallbackReason(Enum):
    """Reasons for falling back to state machine."""
    NONE = auto()                    # No fallback needed
    BUDGET_EXCEEDED = auto()         # Frame budget exhausted
    NO_MATCH_FOUND = auto()          # No suitable match in database
    DATABASE_EMPTY = auto()          # No database assigned
    EXPLICIT_FLAG = auto()           # Explicit fallback flag set
    TRANSITION_FAILURE = auto()      # Inertialization transition failed
    DISABLED = auto()                # Motion matching disabled


class MotionMatchingMode(Enum):
    """Operating mode for motion matching."""
    FULL = auto()           # Full motion matching with search
    CONTINUATION_ONLY = auto()  # Only continue current clip, no search
    FALLBACK = auto()       # Using state machine fallback


# Default budget in milliseconds
DEFAULT_MOTION_MATCHING_BUDGET_MS = 2.0

# Default trajectory time points
DEFAULT_MM_TRAJECTORY_TIMES = [0.2, 0.5, 1.0]

# Minimum cost improvement to trigger transition
DEFAULT_COST_IMPROVEMENT_RATIO = 0.15

# Frame distance for excluding nearby frames from search
DEFAULT_EXCLUDE_FRAMES_BEFORE = 5
DEFAULT_EXCLUDE_FRAMES_AFTER = 10


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class MotionMatchingConfig:
    """Configuration for motion matching system behavior.

    Attributes:
        budget_ms: Maximum time budget per frame for MM processing (milliseconds)
        search_interval: Time between database searches (seconds)
        min_time_in_clip: Minimum time before allowing transition (seconds)
        cost_improvement_threshold: Minimum relative improvement to transition
        trajectory_times: Time points for trajectory prediction (seconds)
        blend_duration: Duration of inertialization blend (seconds)
        enable_foot_locking: Whether to apply foot sliding correction
        fallback_enabled: Whether to allow fallback to state machine
        search_method: Algorithm for database search
        position_weight: Weight for position features
        velocity_weight: Weight for velocity features
        trajectory_weight: Weight for trajectory features
        contact_weight: Weight for foot contact features
    """
    budget_ms: float = DEFAULT_MOTION_MATCHING_BUDGET_MS
    search_interval: float = DEFAULT_CONTROLLER_TIMING.search_interval
    min_time_in_clip: float = DEFAULT_CONTROLLER_TIMING.min_time_in_clip
    cost_improvement_threshold: float = DEFAULT_COST_IMPROVEMENT_RATIO

    trajectory_times: List[float] = field(
        default_factory=lambda: DEFAULT_MM_TRAJECTORY_TIMES.copy()
    )
    blend_duration: float = DEFAULT_TRANSITION_PARAMS.default_blend_duration

    enable_foot_locking: bool = True
    fallback_enabled: bool = True
    search_method: SearchMethod = SearchMethod.BRUTE_FORCE

    position_weight: float = DEFAULT_FEATURE_WEIGHTS.position_weight
    velocity_weight: float = DEFAULT_FEATURE_WEIGHTS.velocity_weight
    trajectory_weight: float = DEFAULT_FEATURE_WEIGHTS.trajectory_weight
    contact_weight: float = DEFAULT_FEATURE_WEIGHTS.contact_weight


@dataclass
class MotionMatchingStatistics:
    """Statistics for motion matching performance tracking.

    Attributes:
        total_queries: Total number of database searches
        successful_matches: Number of searches that found a match
        transitions_triggered: Number of transitions started
        budget_exceeded_count: Times budget was exceeded
        fallback_count: Times fallback was used
        total_search_time_ms: Cumulative search time in milliseconds
        avg_search_time_ms: Average search time per query
        avg_match_cost: Average cost of matched frames
        current_mode: Current operating mode
        last_fallback_reason: Most recent fallback reason
    """
    total_queries: int = 0
    successful_matches: int = 0
    transitions_triggered: int = 0
    budget_exceeded_count: int = 0
    fallback_count: int = 0
    total_search_time_ms: float = 0.0
    avg_search_time_ms: float = 0.0
    avg_match_cost: float = 0.0
    current_mode: MotionMatchingMode = MotionMatchingMode.FULL
    last_fallback_reason: FallbackReason = FallbackReason.NONE

    def record_query(self, search_time_ms: float, cost: float, success: bool) -> None:
        """Record a search query result."""
        self.total_queries += 1
        self.total_search_time_ms += search_time_ms
        self.avg_search_time_ms = self.total_search_time_ms / self.total_queries

        if success:
            self.successful_matches += 1
            # Running average of cost
            n = self.successful_matches
            self.avg_match_cost = ((n - 1) * self.avg_match_cost + cost) / n

    def record_transition(self) -> None:
        """Record a transition was triggered."""
        self.transitions_triggered += 1

    def record_budget_exceeded(self) -> None:
        """Record budget was exceeded."""
        self.budget_exceeded_count += 1

    def record_fallback(self, reason: FallbackReason) -> None:
        """Record fallback was used."""
        self.fallback_count += 1
        self.last_fallback_reason = reason
        self.current_mode = MotionMatchingMode.FALLBACK

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_queries = 0
        self.successful_matches = 0
        self.transitions_triggered = 0
        self.budget_exceeded_count = 0
        self.fallback_count = 0
        self.total_search_time_ms = 0.0
        self.avg_search_time_ms = 0.0
        self.avg_match_cost = 0.0
        self.current_mode = MotionMatchingMode.FULL
        self.last_fallback_reason = FallbackReason.NONE


@dataclass
class TrajectoryState:
    """Current and desired trajectory state.

    Attributes:
        current_position: Current world position
        current_facing: Current facing angle (radians)
        current_velocity: Current velocity vector
        desired_trajectory: Predicted future trajectory
        foot_contacts: Current foot contact states
    """
    current_position: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    current_facing: float = 0.0
    current_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    desired_trajectory: List[TrajectoryPoint] = field(default_factory=list)
    foot_contacts: FootContact = field(default_factory=FootContact)

    def compute_trajectory(
        self,
        input_direction: np.ndarray,
        input_speed: float,
        trajectory_times: List[float],
        turn_rate: float = 10.0,
    ) -> None:
        """Compute desired trajectory from input.

        Args:
            input_direction: Normalized movement direction
            input_speed: Desired movement speed
            trajectory_times: Time points for trajectory (seconds)
            turn_rate: Maximum turn rate (radians/second)
        """
        self.desired_trajectory.clear()

        if input_speed < DEFAULT_IDLE_DETECTION.velocity_threshold:
            # Stationary - stay in place
            for t in trajectory_times:
                self.desired_trajectory.append(TrajectoryPoint(
                    time_offset=t,
                    position=self.current_position.copy(),
                    facing=self.current_facing,
                    velocity=np.zeros(3, dtype=np.float32),
                ))
            return

        # Compute target facing from direction
        if input_direction[0] != 0 or input_direction[2] != 0:
            target_facing = math.atan2(input_direction[2], input_direction[0])
        else:
            target_facing = self.current_facing

        # Build trajectory points
        position = self.current_position.copy()
        facing = self.current_facing

        for t in trajectory_times:
            # Interpolate facing toward target
            facing_diff = _normalize_angle(target_facing - facing)
            max_turn = turn_rate * t
            if abs(facing_diff) > max_turn:
                facing = facing + max_turn * np.sign(facing_diff)
            else:
                facing = target_facing

            # Compute velocity in facing direction
            velocity = np.array([
                math.cos(facing) * input_speed,
                0.0,
                math.sin(facing) * input_speed,
            ], dtype=np.float32)

            # Predict future position
            future_position = self.current_position + velocity * t

            self.desired_trajectory.append(TrajectoryPoint(
                time_offset=t,
                position=future_position,
                facing=facing,
                velocity=velocity,
            ))


@dataclass
class MotionMatchingComponent:
    """Component for entities using motion matching animation.

    Attributes:
        database: Motion database for searching
        config: Motion matching configuration
        enabled: Whether motion matching is active
        use_fallback: Force fallback to state machine
        required_tags: Tags required for search filtering
        excluded_tags: Tags to exclude from search
        context_modifiers: Context-dependent cost modifiers

        current_clip_index: Currently playing clip
        current_frame: Current frame in clip
        current_time: Time into current clip (seconds)
        current_pose: Current output pose
        current_entry_index: Index of current database entry

        trajectory_state: Current trajectory information
        statistics: Performance statistics

        _search: Internal search instance
        _feature_extractor: Internal feature extractor
        _transition: Active inertialization transition
        _time_since_search: Time since last search
        _time_since_transition: Time since last transition
        _current_cost: Cost of current match
        _frame_budget_used_ms: Budget used this frame
    """
    database: Optional[MotionDatabase] = None
    config: MotionMatchingConfig = field(default_factory=MotionMatchingConfig)
    enabled: bool = True
    use_fallback: bool = False
    required_tags: Optional[Set[str]] = None
    excluded_tags: Optional[Set[str]] = None
    context_modifiers: Dict[str, float] = field(default_factory=dict)

    # Playback state
    current_clip_index: int = -1
    current_frame: int = 0
    current_time: float = 0.0
    current_pose: Optional[Pose] = None
    current_entry_index: int = -1

    # Trajectory and contacts
    trajectory_state: TrajectoryState = field(default_factory=TrajectoryState)
    statistics: MotionMatchingStatistics = field(default_factory=MotionMatchingStatistics)

    # Internal state (not serialized)
    _search: Optional[MotionSearch] = field(default=None, repr=False)
    _feature_extractor: Optional[FeatureExtractor] = field(default=None, repr=False)
    _transition: Optional[MotionTransition] = field(default=None, repr=False)
    _blender: Optional[InertializationBlender] = field(default=None, repr=False)
    _time_since_search: float = field(default=0.0, repr=False)
    _time_since_transition: float = field(default=0.0, repr=False)
    _current_cost: float = field(default=float('inf'), repr=False)
    _frame_budget_used_ms: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        """Initialize internal components if database is provided."""
        if self.database is not None:
            self._initialize_search()

    def _initialize_search(self) -> None:
        """Initialize search and feature extractor from database."""
        if self.database is None:
            return

        # Create feature extractor
        feature_config = FeatureConfig(
            trajectory_times=self.config.trajectory_times,
            position_weight=self.config.position_weight,
            velocity_weight=self.config.velocity_weight,
            trajectory_weight=self.config.trajectory_weight,
            contact_weight=self.config.contact_weight,
        )
        self._feature_extractor = FeatureExtractor(feature_config)

        # Create search instance
        self._search = MotionSearch(
            self.database,
            method=self.config.search_method,
        )

    def set_database(self, database: MotionDatabase) -> None:
        """Set or update the motion database.

        Args:
            database: New motion database
        """
        self.database = database
        self._initialize_search()
        # Reset state
        self.current_clip_index = -1
        self.current_frame = 0
        self.current_time = 0.0
        self.current_entry_index = -1
        self._current_cost = float('inf')

    @property
    def is_transitioning(self) -> bool:
        """Whether an inertialization transition is active."""
        return self._blender is not None

    @property
    def is_ready(self) -> bool:
        """Whether the component is ready for motion matching."""
        return (
            self.database is not None and
            self._search is not None and
            self._feature_extractor is not None and
            self.database.entry_count > 0
        )


@dataclass
class MotionMatchingInput:
    """Input state for motion matching update.

    Attributes:
        direction: Movement direction (normalized 3D vector)
        speed: Desired movement speed
        bone_data: Current bone positions/velocities for feature extraction
    """
    direction: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    speed: float = 0.0
    bone_data: Dict[str, BoneData] = field(default_factory=dict)


# Legacy compatibility types
@dataclass
class MotionInput:
    """Legacy input state for motion matching (compatibility).

    Attributes:
        desired_velocity: Desired movement velocity
        desired_direction: Desired facing direction
        trajectory: Future trajectory points (positions)
        trajectory_times: Times for trajectory points
        features: Additional feature values
    """
    desired_velocity: Any = field(default_factory=lambda: np.zeros(3))
    desired_direction: Any = field(default_factory=lambda: np.array([0, 0, 1]))
    trajectory: List[Any] = field(default_factory=list)
    trajectory_times: List[float] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionFeature:
    """Legacy feature definition (compatibility).

    Attributes:
        name: Feature name
        feature_type: Type of feature
        bone_index: Bone to track (-1 for root)
        weight: Feature weight in matching
        trajectory_times: Times for trajectory features (seconds)
    """
    name: str = ""
    feature_type: FeatureType = FeatureType.BONE_POSITION
    bone_index: int = -1
    weight: float = 1.0
    trajectory_times: List[float] = field(default_factory=list)


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


def _extract_facing_from_pose(pose: Pose) -> float:
    """Extract facing angle from pose root rotation.

    Args:
        pose: Pose with root rotation

    Returns:
        Facing angle in radians
    """
    q = pose.root_rotation
    siny_cosp = 2.0 * (q[3] * q[1] + q[2] * q[0])
    cosy_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    return math.atan2(siny_cosp, cosy_cosp)


def _pose_to_bone_data(pose: Pose) -> Dict[str, BoneData]:
    """Convert Pose to bone data dictionary.

    Args:
        pose: Pose to convert

    Returns:
        Dictionary of bone name to BoneData
    """
    bone_data = {}
    for name, transform in pose.bone_transforms.items():
        bone_data[name] = BoneData(
            position=transform.position.copy(),
            velocity=np.zeros(3, dtype=np.float32),  # No velocity info in static pose
            rotation=transform.rotation.copy() if transform.rotation is not None else None,
        )
    return bone_data


# =============================================================================
# POSE PROVIDER PROTOCOL
# =============================================================================


class PoseProvider(Protocol):
    """Protocol for objects that can provide poses from clip/frame indices."""

    def get_pose(self, clip_index: int, frame: int) -> Pose:
        """Get pose for a specific clip and frame.

        Args:
            clip_index: Index of animation clip
            frame: Frame number within clip

        Returns:
            Pose at that frame
        """
        ...


# =============================================================================
# MOTION MATCHING SYSTEM
# =============================================================================


@system(phase="animation", order=0, reads=("MotionMatchingComponent",), writes=("Pose",))
class MotionMatchingSystem:
    """ECS system for motion matching animation.

    Replaces standard state machine for MM-enabled characters. Each frame:
    1. Computes desired trajectory from input
    2. Extracts pose features
    3. Searches database for best match
    4. Applies inertialization transition
    5. Enforces budget limits with fallback

    Attributes:
        pose_provider: Function to get poses from clip/frame indices
        state_machine_fallback: Callback for state machine fallback
        _total_time: Total elapsed time
    """

    def __init__(
        self,
        pose_provider: Optional[Callable[[int, int], Pose]] = None,
        state_machine_fallback: Optional[Callable[[Any, float], Pose]] = None,
    ):
        """Initialize the motion matching system.

        Args:
            pose_provider: Function(clip_index, frame) -> Pose
            state_machine_fallback: Callback for fallback animation
        """
        self.pose_provider = pose_provider
        self.state_machine_fallback = state_machine_fallback
        self._total_time = 0.0

    def set_pose_provider(
        self,
        provider: Callable[[int, int], Pose]
    ) -> None:
        """Set the pose provider function.

        Args:
            provider: Function(clip_index, frame) -> Pose
        """
        self.pose_provider = provider

    def set_state_machine_fallback(
        self,
        fallback: Callable[[Any, float], Pose]
    ) -> None:
        """Set the state machine fallback callback.

        Args:
            fallback: Function(entity, dt) -> Pose for fallback animation
        """
        self.state_machine_fallback = fallback

    def update(
        self,
        world: Any,
        dt: float,
        entity_components: List[Tuple[Any, MotionMatchingComponent]],
    ) -> None:
        """Update all motion matching components.

        Args:
            world: ECS world
            dt: Delta time in seconds
            entity_components: List of (entity, component) tuples
        """
        self._total_time += dt

        for entity, component in entity_components:
            self._update_component(entity, component, dt)

    def update_single(
        self,
        component: MotionMatchingComponent,
        input_state: MotionMatchingInput,
        dt: float,
    ) -> Optional[Pose]:
        """Update a single component with explicit input.

        Args:
            component: Motion matching component
            input_state: Input direction and speed
            dt: Delta time

        Returns:
            Output pose or None
        """
        if not component.enabled:
            return None

        # Reset frame budget
        component._frame_budget_used_ms = 0.0

        # Check for fallback conditions
        fallback_reason = self._check_fallback(component)
        if fallback_reason != FallbackReason.NONE:
            component.statistics.record_fallback(fallback_reason)
            return self._get_fallback_pose(component, dt)

        # Update trajectory from input
        component.trajectory_state.compute_trajectory(
            input_direction=input_state.direction,
            input_speed=input_state.speed,
            trajectory_times=component.config.trajectory_times,
        )

        # Update timers
        component._time_since_search += dt
        component._time_since_transition += dt

        # Update active transition
        if component._blender is not None:
            self._update_transition(component, dt)

        # Perform search if conditions are met
        if self._should_search(component):
            self._perform_search(component, input_state)

        # Advance playback
        self._advance_playback(component, dt)

        # Get output pose
        return self._get_output_pose(component)

    def _update_component(
        self,
        entity: Any,
        component: MotionMatchingComponent,
        dt: float,
    ) -> None:
        """Update a single motion matching component.

        Args:
            entity: ECS entity
            component: Motion matching component
            dt: Delta time
        """
        if not component.enabled:
            return

        # Reset frame budget
        component._frame_budget_used_ms = 0.0

        # Check for fallback conditions
        fallback_reason = self._check_fallback(component)
        if fallback_reason != FallbackReason.NONE:
            component.statistics.record_fallback(fallback_reason)
            component.current_pose = self._get_fallback_pose(component, dt, entity)
            return

        # Update trajectory from current pose
        if component.current_pose is not None:
            component.trajectory_state.current_position = component.current_pose.root_position.copy()
            component.trajectory_state.current_facing = _extract_facing_from_pose(component.current_pose)

        # Update timers
        component._time_since_search += dt
        component._time_since_transition += dt

        # Update active transition
        if component._blender is not None:
            self._update_transition(component, dt)

        # Perform search if conditions are met
        if self._should_search(component):
            input_state = MotionMatchingInput()
            if component.current_pose:
                input_state.bone_data = _pose_to_bone_data(component.current_pose)
            self._perform_search(component, input_state)

        # Advance playback
        self._advance_playback(component, dt)

        # Get output pose
        component.current_pose = self._get_output_pose(component)

    # -------------------------------------------------------------------------
    # Fallback Handling
    # -------------------------------------------------------------------------

    def _check_fallback(self, component: MotionMatchingComponent) -> FallbackReason:
        """Check if fallback is needed.

        Args:
            component: Motion matching component

        Returns:
            Reason for fallback, or NONE if no fallback needed
        """
        if not component.enabled:
            return FallbackReason.DISABLED

        if component.use_fallback:
            return FallbackReason.EXPLICIT_FLAG

        if component.database is None:
            return FallbackReason.DATABASE_EMPTY

        if component.database.entry_count == 0:
            return FallbackReason.DATABASE_EMPTY

        if not component.is_ready:
            return FallbackReason.DATABASE_EMPTY

        # Check budget
        if component._frame_budget_used_ms >= component.config.budget_ms:
            component.statistics.record_budget_exceeded()
            return FallbackReason.BUDGET_EXCEEDED

        return FallbackReason.NONE

    def _get_fallback_pose(
        self,
        component: MotionMatchingComponent,
        dt: float,
        entity: Any = None,
    ) -> Optional[Pose]:
        """Get pose from state machine fallback.

        Args:
            component: Motion matching component
            dt: Delta time
            entity: ECS entity

        Returns:
            Fallback pose or None
        """
        if self.state_machine_fallback is not None and entity is not None:
            return self.state_machine_fallback(entity, dt)

        # Return current pose if no fallback available
        return component.current_pose

    # -------------------------------------------------------------------------
    # Search Logic
    # -------------------------------------------------------------------------

    def _should_search(self, component: MotionMatchingComponent) -> bool:
        """Determine if we should search for a new match.

        Args:
            component: Motion matching component

        Returns:
            True if search should be performed
        """
        # Don't search during active transition
        if component._blender is not None:
            return False

        # Check time since last search
        if component._time_since_search < component.config.search_interval:
            return False

        # Check minimum time in clip
        if component._time_since_transition < component.config.min_time_in_clip:
            return False

        # Check remaining budget
        remaining_budget = component.config.budget_ms - component._frame_budget_used_ms
        if remaining_budget < 0.5:  # Need at least 0.5ms for search
            return False

        return True

    def _perform_search(
        self,
        component: MotionMatchingComponent,
        input_state: MotionMatchingInput,
    ) -> None:
        """Search database for best matching frame.

        Args:
            component: Motion matching component
            input_state: Current input state
        """
        if component._search is None or component._feature_extractor is None:
            return

        start_time = time.perf_counter()

        # Build query features
        query = self._build_query(component, input_state)

        # Configure search
        search_config = SearchConfig(
            max_results=3,
            method=component.config.search_method,
            only_transition_candidates=True,
            required_tags=component.required_tags,
            excluded_tags=component.excluded_tags,
        )

        # Exclude frames near current position
        if component.current_entry_index >= 0 and component.current_clip_index >= 0:
            min_frame = component.current_frame - DEFAULT_EXCLUDE_FRAMES_BEFORE
            max_frame = component.current_frame + DEFAULT_EXCLUDE_FRAMES_AFTER
            search_config.exclude_frames_range = (
                component.current_clip_index,
                max(0, min_frame),
                max_frame,
            )

        # Perform search
        try:
            results = component._search.search(query, search_config)
        except Exception:
            results = []

        # Record timing
        search_time_ms = (time.perf_counter() - start_time) * 1000.0
        component._frame_budget_used_ms += search_time_ms
        component._time_since_search = 0.0

        # Process results
        if not results:
            component.statistics.record_query(search_time_ms, 0.0, False)
            component.statistics.last_fallback_reason = FallbackReason.NO_MATCH_FOUND
            return

        best_result = results[0]

        # Apply context modifiers to cost
        modified_cost = best_result.cost
        for modifier_key, modifier_value in component.context_modifiers.items():
            if modifier_key in best_result.entry.tags:
                modified_cost *= modifier_value

        component.statistics.record_query(search_time_ms, modified_cost, True)

        # Check if improvement is worth transitioning
        cost_improvement = component._current_cost - modified_cost
        relative_improvement = cost_improvement / max(component._current_cost, 1e-6)

        if relative_improvement > component.config.cost_improvement_threshold:
            self._start_transition(component, best_result)
            component.statistics.record_transition()

    def _build_query(
        self,
        component: MotionMatchingComponent,
        input_state: MotionMatchingInput,
    ) -> FeatureSet:
        """Build query features for search.

        Args:
            component: Motion matching component
            input_state: Current input state

        Returns:
            FeatureSet for database search
        """
        if component._feature_extractor is None:
            # Return empty feature set
            return FeatureSet(values=np.zeros(0, dtype=np.float32))

        return component._feature_extractor.extract_from_pose(
            bone_data=input_state.bone_data,
            trajectory=component.trajectory_state.desired_trajectory,
            foot_contacts=component.trajectory_state.foot_contacts,
        )

    # -------------------------------------------------------------------------
    # Transition Handling
    # -------------------------------------------------------------------------

    def _start_transition(
        self,
        component: MotionMatchingComponent,
        result: SearchResult,
    ) -> None:
        """Start inertialization transition to new match.

        Args:
            component: Motion matching component
            result: Search result to transition to
        """
        # Get current and target poses
        from_pose = component.current_pose or Pose()
        to_pose = self._get_pose_for_entry(component, result.entry)

        if to_pose is None:
            return

        # Create inertialization blender
        transition_config = TransitionConfig(
            blend_duration=component.config.blend_duration,
            blend_mode=BlendMode.INERTIALIZATION,
            foot_locking=component.config.enable_foot_locking,
        )
        component._blender = InertializationBlender(transition_config)
        component._blender.compute_offsets(from_pose, to_pose)

        # Update playback state
        component.current_entry_index = result.entry_index
        component.current_clip_index = result.entry.clip_index
        component.current_frame = result.entry.frame
        component.current_time = result.entry.frame / self._get_frame_rate(component)
        component._current_cost = result.cost
        component._time_since_transition = 0.0

        component.statistics.current_mode = MotionMatchingMode.FULL

    def _update_transition(self, component: MotionMatchingComponent, dt: float) -> None:
        """Update active inertialization transition.

        Args:
            component: Motion matching component
            dt: Delta time
        """
        if component._blender is None:
            return

        component._blender.update(dt)

        # Check if transition is complete
        if component._blender.is_complete:
            component._blender = None

    def _get_pose_for_entry(
        self,
        component: MotionMatchingComponent,
        entry: DatabaseEntry,
    ) -> Optional[Pose]:
        """Get pose for a database entry.

        Args:
            component: Motion matching component
            entry: Database entry

        Returns:
            Pose for entry or None
        """
        if self.pose_provider is not None:
            return self.pose_provider(entry.clip_index, entry.frame)

        return None

    # -------------------------------------------------------------------------
    # Playback
    # -------------------------------------------------------------------------

    def _advance_playback(self, component: MotionMatchingComponent, dt: float) -> None:
        """Advance playback in current clip.

        Args:
            component: Motion matching component
            dt: Delta time
        """
        if component.database is None or component.current_clip_index < 0:
            return

        clip_metadata = component.database.get_clip_metadata(component.current_clip_index)
        if clip_metadata is None:
            return

        frame_rate = clip_metadata.frame_rate
        component.current_time += dt
        new_frame = int(component.current_time * frame_rate)

        # Handle clip end
        if new_frame >= clip_metadata.frame_count - 1:
            if clip_metadata.is_looping:
                component.current_time = 0.0
                component.current_frame = 0
            else:
                # Force search for new clip
                component._current_cost = float('inf')
                component.current_frame = clip_metadata.frame_count - 1
        else:
            component.current_frame = new_frame

    def _get_frame_rate(self, component: MotionMatchingComponent) -> float:
        """Get frame rate for current clip.

        Args:
            component: Motion matching component

        Returns:
            Frame rate in FPS
        """
        if component.database is None or component.current_clip_index < 0:
            return 30.0

        clip_metadata = component.database.get_clip_metadata(component.current_clip_index)
        return clip_metadata.frame_rate if clip_metadata else 30.0

    def _get_output_pose(self, component: MotionMatchingComponent) -> Optional[Pose]:
        """Get final output pose with transitions applied.

        Args:
            component: Motion matching component

        Returns:
            Final output pose
        """
        if self.pose_provider is None:
            return component.current_pose

        if component.current_clip_index < 0:
            return component.current_pose

        # Get base pose from provider
        base_pose = self.pose_provider(component.current_clip_index, component.current_frame)

        if base_pose is None:
            return component.current_pose

        # Apply inertialization offset if transitioning
        if component._blender is not None:
            return component._blender.apply(base_pose)

        return base_pose

    # -------------------------------------------------------------------------
    # Statistics and Debug
    # -------------------------------------------------------------------------

    def get_statistics(
        self,
        component: MotionMatchingComponent,
    ) -> MotionMatchingStatistics:
        """Get statistics for a component.

        Args:
            component: Motion matching component

        Returns:
            Statistics snapshot
        """
        return component.statistics

    def reset_statistics(self, component: MotionMatchingComponent) -> None:
        """Reset statistics for a component.

        Args:
            component: Motion matching component
        """
        component.statistics.reset()

    def get_debug_info(
        self,
        component: MotionMatchingComponent,
    ) -> Dict[str, Any]:
        """Get debug information for a component.

        Args:
            component: Motion matching component

        Returns:
            Dictionary of debug values
        """
        return {
            "enabled": component.enabled,
            "is_ready": component.is_ready,
            "mode": component.statistics.current_mode.name,
            "clip_index": component.current_clip_index,
            "frame": component.current_frame,
            "time": component.current_time,
            "current_cost": component._current_cost,
            "is_transitioning": component.is_transitioning,
            "time_since_search": component._time_since_search,
            "time_since_transition": component._time_since_transition,
            "frame_budget_used_ms": component._frame_budget_used_ms,
            "total_queries": component.statistics.total_queries,
            "successful_matches": component.statistics.successful_matches,
            "transitions": component.statistics.transitions_triggered,
            "fallback_count": component.statistics.fallback_count,
            "avg_search_time_ms": component.statistics.avg_search_time_ms,
            "avg_match_cost": component.statistics.avg_match_cost,
            "last_fallback_reason": component.statistics.last_fallback_reason.name,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # System decorator
    "system",
    # Enums
    "FallbackReason",
    "MotionMatchingMode",
    # Configuration
    "MotionMatchingConfig",
    "MotionMatchingStatistics",
    # Components
    "MotionMatchingComponent",
    "MotionMatchingInput",
    "TrajectoryState",
    # Legacy compatibility
    "MotionInput",
    "MotionFeature",
    # System
    "MotionMatchingSystem",
    # Protocols
    "PoseProvider",
]
