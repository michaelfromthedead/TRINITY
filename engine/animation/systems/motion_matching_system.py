"""ECS system for motion matching.

Provides data-driven animation selection using motion matching techniques.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from engine.core.math import Vec3, Quat, Transform
from engine.core.ecs import Entity, World
from engine.animation.config import ANIMATION_SYSTEM_CONFIG

# Import from motionmatching submodule
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
)
from engine.animation.motionmatching.transition import (
    BlendMode,
    BoneTransform,
    InertializationBlender,
    MotionTransition,
    Pose,
    TransitionConfig,
)

# Re-export commonly used types
__all__ = [
    "MotionMatchingSystem",
    "MotionMatchingComponent",
    "MotionMatchingConfig",
    "MotionMatchingStatistics",
    "MotionInput",
    "MotionFeature",
    "MotionFrame",
    "MotionDatabase",
    "MotionMatchingController",
    "FeatureType",
    "FallbackReason",
    "MotionMatchingMode",
    "TrajectoryState",
]


# =============================================================================
# MOTION FRAME
# =============================================================================

@dataclass
class MotionFrame:
    """Single frame of motion data for matching."""
    clip_index: int = 0
    frame_index: int = 0
    time: float = 0.0
    pose: Optional[Pose] = None
    features: Optional[np.ndarray] = None
    velocity: Optional[Vec3] = None
    angular_velocity: float = 0.0
    foot_contacts: int = 0  # Bitmask
    animation_index: int = 0  # Alias for clip_index (legacy compatibility)

    def __post_init__(self):
        # Sync animation_index and clip_index
        if self.animation_index != 0 and self.clip_index == 0:
            self.clip_index = self.animation_index
        elif self.clip_index != 0 and self.animation_index == 0:
            self.animation_index = self.clip_index

    def get_feature_vector(self) -> np.ndarray:
        """Get feature vector for this frame."""
        if self.features is not None:
            return self.features
        return np.zeros(0)


# =============================================================================
# MOTION MATCHING CONTROLLER
# =============================================================================

class MotionMatchingController:
    """High-level controller for motion matching."""

    def __init__(self, system: Optional['MotionMatchingSystem'] = None):
        self.system = system
        self._active_entities: Set[Entity] = set()
        self._paused = False

    def register_entity(self, entity: Entity) -> None:
        """Register an entity for motion matching."""
        self._active_entities.add(entity)

    def unregister_entity(self, entity: Entity) -> None:
        """Unregister an entity from motion matching."""
        self._active_entities.discard(entity)

    def pause(self) -> None:
        """Pause motion matching for all entities."""
        self._paused = True

    def resume(self) -> None:
        """Resume motion matching for all entities."""
        self._paused = False

    def is_paused(self) -> bool:
        """Check if motion matching is paused."""
        return self._paused

    def get_active_entities(self) -> Set[Entity]:
        """Get all active entities."""
        return self._active_entities.copy()

    def update(self, dt: float) -> None:
        """Update motion matching for all active entities."""
        if self._paused or self.system is None:
            return
        # Delegate to system


# =============================================================================
# SYSTEM DECORATOR
# =============================================================================


def system(
    phase: str = "update",
    order: int = 0,
    reads: Optional[List[str]] = None,
    writes: Optional[List[str]] = None,
):
    """Decorator to mark a class as an ECS system with metadata.

    Args:
        phase: System execution phase (e.g., "animation", "physics")
        order: Execution order within phase (lower = earlier)
        reads: List of component types this system reads
        writes: List of component types this system writes
    """
    def decorator(cls):
        cls._system_phase = phase
        cls._system_order = order
        cls._system_reads = reads or []
        cls._system_writes = writes or []
        cls._is_system = True
        return cls
    return decorator


# =============================================================================
# ENUMS
# =============================================================================


class FallbackReason(Enum):
    """Why motion matching fell back to blend tree."""
    NONE = auto()
    DISABLED = auto()
    EXPLICIT_FLAG = auto()
    DATABASE_EMPTY = auto()
    NO_DATABASE = auto()
    SEARCH_FAILED = auto()
    NO_MATCH_FOUND = auto()
    QUALITY_TOO_LOW = auto()
    TRANSITION_LOCKED = auto()
    BUDGET_EXCEEDED = auto()


class MotionMatchingMode(Enum):
    """Motion matching operating modes."""
    FULL = auto()              # Full motion matching
    TRAJECTORY_ONLY = auto()   # Only match trajectory
    POSE_ONLY = auto()         # Only match pose
    FALLBACK = auto()          # Using fallback blend tree
    DISABLED = auto()          # Disabled


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class MotionMatchingConfig:
    """Motion matching configuration."""
    search_interval: float = 0.1    # Seconds between searches
    quality_threshold: float = 0.8  # Min match quality (0-1)
    blend_time: float = 0.2         # Transition blend duration
    trajectory_weight: float = 1.0  # Trajectory feature weight
    pose_weight: float = 1.0        # Pose feature weight
    velocity_weight: float = 0.5    # Velocity feature weight
    max_candidates: int = 10        # Max candidates to evaluate
    budget_ms: float = 2.0          # Frame budget in milliseconds
    min_search_interval: float = 0.05  # Minimum time between searches
    min_transition_interval: float = 0.1  # Minimum time between transitions


# =============================================================================
# STATISTICS
# =============================================================================


@dataclass
class MotionMatchingStatistics:
    """Runtime statistics for motion matching."""
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

    # Internal tracking
    _total_cost: float = 0.0
    _cost_count: int = 0

    def record_query(self, search_time_ms: float, cost: float, success: bool) -> None:
        """Record a search query."""
        self.total_queries += 1
        self.total_search_time_ms += search_time_ms
        self.avg_search_time_ms = self.total_search_time_ms / self.total_queries

        if success:
            self.successful_matches += 1
            self._total_cost += cost
            self._cost_count += 1
            self.avg_match_cost = self._total_cost / self._cost_count

    def record_transition(self) -> None:
        """Record a transition."""
        self.transitions_triggered += 1

    def record_budget_exceeded(self) -> None:
        """Record a budget exceeded event."""
        self.budget_exceeded_count += 1

    def record_fallback(self, reason: FallbackReason) -> None:
        """Record a fallback event."""
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
        self._total_cost = 0.0
        self._cost_count = 0


# =============================================================================
# TRAJECTORY STATE
# =============================================================================


@dataclass
class TrajectoryState:
    """Future trajectory prediction state.

    Attributes:
        current_position: Current world position
        current_facing: Current facing angle (radians)
        current_velocity: Current velocity
        desired_trajectory: List of predicted trajectory points
        foot_contacts: Current foot contact state
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

    def __post_init__(self):
        if not isinstance(self.current_position, np.ndarray):
            self.current_position = np.array(self.current_position, dtype=np.float32)
        if not isinstance(self.current_velocity, np.ndarray):
            self.current_velocity = np.array(self.current_velocity, dtype=np.float32)

    def compute_trajectory(
        self,
        input_direction: np.ndarray,
        input_speed: float,
        trajectory_times: List[float],
        turn_rate: float = 3.0,
    ) -> None:
        """Compute desired trajectory from input.

        Args:
            input_direction: Desired movement direction (normalized)
            input_speed: Desired movement speed
            trajectory_times: Time points to compute trajectory for
            turn_rate: Maximum turn rate in radians per second
        """
        self.desired_trajectory = []

        if not trajectory_times:
            return

        # Compute velocity from direction and speed
        dir_norm = np.linalg.norm(input_direction)
        if dir_norm > 1e-6:
            normalized_dir = input_direction / dir_norm
            desired_velocity = normalized_dir * input_speed
            # Compute target facing from direction
            target_facing = math.atan2(normalized_dir[2], normalized_dir[0])
        else:
            desired_velocity = np.zeros(3, dtype=np.float32)
            target_facing = self.current_facing

        # Compute trajectory points at each time
        for t in trajectory_times:
            # Interpolate facing toward target
            facing_diff = target_facing - self.current_facing
            # Normalize angle to [-pi, pi]
            while facing_diff > math.pi:
                facing_diff -= 2 * math.pi
            while facing_diff < -math.pi:
                facing_diff += 2 * math.pi

            max_turn = turn_rate * t
            if abs(facing_diff) <= max_turn:
                facing = target_facing
            else:
                facing = self.current_facing + max_turn * (1 if facing_diff > 0 else -1)

            # Compute position
            position = self.current_position + desired_velocity * t

            # Create trajectory point
            point = TrajectoryPoint(
                time_offset=t,
                position=position.copy(),
                facing=facing,
                velocity=desired_velocity.copy(),
            )
            self.desired_trajectory.append(point)


# =============================================================================
# INPUT
# =============================================================================


@dataclass
class MotionMatchingInput:
    """Input state for motion matching (new API)."""
    desired_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    desired_facing: np.ndarray = field(
        default_factory=lambda: np.array([1, 0, 0], dtype=np.float32)
    )
    trajectory: Optional[TrajectoryState] = None

    def __post_init__(self):
        if not isinstance(self.desired_velocity, np.ndarray):
            self.desired_velocity = np.array(self.desired_velocity, dtype=np.float32)
        if not isinstance(self.desired_facing, np.ndarray):
            self.desired_facing = np.array(self.desired_facing, dtype=np.float32)


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================


@dataclass
class MotionInput:
    """Input state for motion matching (legacy API).

    Attributes:
        desired_velocity: Desired movement velocity
        desired_direction: Desired facing direction
        trajectory: Future trajectory points (positions)
        trajectory_times: Times for trajectory points
        features: Additional feature values
    """
    desired_velocity: Vec3 = field(default_factory=Vec3.zero)
    desired_direction: Vec3 = field(default_factory=Vec3.forward)
    trajectory: List[Vec3] = field(default_factory=list)
    trajectory_times: List[float] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionFeature:
    """Feature used for motion matching.

    Attributes:
        name: Feature name
        feature_type: Type of feature
        bone_index: Bone to track (-1 for root)
        weight: Feature weight in matching
        trajectory_times: Times for trajectory features (seconds)
        values: Feature vector values (for distance calculations)
    """
    name: str = ""
    feature_type: FeatureType = FeatureType.BONE_POSITION
    bone_index: int = -1
    weight: float = 1.0
    trajectory_times: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def distance(self, other: 'MotionFeature') -> float:
        """Compute squared distance between feature vectors."""
        if len(self.values) != len(other.values):
            return float('inf')
        return sum((a - b) ** 2 for a, b in zip(self.values, other.values))

    def weighted_distance(self, other: 'MotionFeature', weights: List[float]) -> float:
        """Compute weighted squared distance."""
        if len(self.values) != len(other.values) or len(weights) != len(self.values):
            return float('inf')
        return sum(w * (a - b) ** 2 for w, a, b in zip(weights, self.values, other.values))


# =============================================================================
# COMPONENT
# =============================================================================


@dataclass
class MotionMatchingComponent:
    """Component for entities using motion matching.

    Attributes:
        database: Motion database to search
        config: Motion matching configuration
        enabled: Whether motion matching is enabled
        use_fallback: Force fallback to blend tree
        required_tags: Tags required for search results
        context_modifiers: Tag-based cost modifiers
        statistics: Runtime statistics
        current_pose: Current output pose
        current_clip_index: Current clip being played
        current_frame: Current frame in clip
        current_time: Current time in clip
        current_cost: Cost of current match
    """
    database: Optional[MotionDatabase] = None
    config: MotionMatchingConfig = field(default_factory=MotionMatchingConfig)
    enabled: bool = True
    use_fallback: bool = False
    required_tags: Set[str] = field(default_factory=set)
    context_modifiers: Dict[str, float] = field(default_factory=dict)
    statistics: MotionMatchingStatistics = field(default_factory=MotionMatchingStatistics)

    # Output state
    current_pose: Optional[Pose] = None
    current_clip_index: int = -1
    current_frame: int = 0
    current_time: float = 0.0
    current_cost: float = 0.0

    # Internal state
    _search: Optional[MotionSearch] = None
    _blender: Optional[InertializationBlender] = None
    _time_since_search: float = 0.0
    _time_since_transition: float = 0.0
    _frame_budget_used_ms: float = 0.0
    _last_search_time: float = 0.0

    def __post_init__(self):
        if isinstance(self.required_tags, (list, tuple)):
            self.required_tags = set(self.required_tags)
        if self.database is not None:
            self._search = MotionSearch(self.database)

    @property
    def is_ready(self) -> bool:
        """Check if component is ready for motion matching."""
        return (
            self.database is not None
            and self.database.entry_count > 0
            and self.enabled
        )

    @property
    def is_transitioning(self) -> bool:
        """Check if currently in a transition."""
        return self._blender is not None

    def set_database(self, database: MotionDatabase) -> None:
        """Set the motion database.

        Args:
            database: Motion database to use
        """
        self.database = database
        self._search = MotionSearch(database)


# =============================================================================
# SYSTEM
# =============================================================================


@system(
    phase="animation",
    order=0,
    reads=["MotionMatchingComponent", "MotionMatchingInput"],
    writes=["MotionMatchingComponent"],
)
class MotionMatchingSystem:
    """ECS system for motion matching.

    Updates motion matching controllers and produces animation outputs.
    """

    def __init__(
        self,
        pose_provider: Optional[Callable[[int, int], Pose]] = None,
        state_machine_fallback: Optional[Callable[[Any, float], Pose]] = None,
    ):
        """Initialize motion matching system.

        Args:
            pose_provider: Function to get poses from clips
            state_machine_fallback: Fallback callback for blend tree
        """
        self.pose_provider = pose_provider
        self.state_machine_fallback = state_machine_fallback

    def set_pose_provider(
        self,
        provider: Callable[[int, int], Pose]
    ) -> None:
        """Set function that provides poses.

        Args:
            provider: Function(clip_index, frame) -> Pose
        """
        self.pose_provider = provider

    def set_state_machine_fallback(
        self,
        fallback: Callable[[Any, float], Pose]
    ) -> None:
        """Set fallback callback for blend tree.

        Args:
            fallback: Function(entity, dt) -> Pose
        """
        self.state_machine_fallback = fallback

    def update(
        self,
        world: Optional[World],
        dt: float,
        entity_components: List[Tuple[Entity, MotionMatchingComponent]]
    ) -> None:
        """Update all motion matching components.

        Args:
            world: ECS world
            dt: Delta time
            entity_components: List of (entity, component) tuples
        """
        for entity, component in entity_components:
            self._update_component(entity, component, dt)

    def _update_component(
        self,
        entity: Optional[Entity],
        component: MotionMatchingComponent,
        dt: float
    ) -> None:
        """Update single motion matching component."""
        # Reset frame budget
        component._frame_budget_used_ms = 0.0

        # Update timers
        component._time_since_search += dt
        component._time_since_transition += dt

        # Check for fallback conditions
        fallback_reason = self._check_fallback(component)
        if fallback_reason != FallbackReason.NONE:
            component.statistics.record_fallback(fallback_reason)
            pose = self._get_fallback_pose(component, dt)
            if pose:
                component.current_pose = pose
            return

        # Reset mode to FULL if not in fallback
        component.statistics.current_mode = MotionMatchingMode.FULL

        # Get current input
        input_state = MotionMatchingInput()

        # Perform search if needed
        if self._should_search(component):
            self._perform_search(component, input_state)

        # Update transition blending
        if component._blender is not None:
            component._blender.update(dt)
            if component._blender.is_complete:
                component._blender = None

        # Advance playback
        if component.database is not None:
            clip = component.database.get_clip_metadata(component.current_clip_index)
            if clip is not None:
                component.current_time += dt
                frame_time = 1.0 / clip.frame_rate if clip.frame_rate > 0 else 1.0 / 30.0
                frames_to_advance = int(component.current_time / frame_time)
                if frames_to_advance > 0:
                    component.current_time -= frames_to_advance * frame_time
                    component.current_frame += frames_to_advance
                    # Handle looping
                    if clip.is_looping:
                        component.current_frame %= clip.frame_count
                    else:
                        component.current_frame = min(component.current_frame, clip.frame_count - 1)

        # Get output pose
        if self.pose_provider and component.current_clip_index >= 0:
            pose = self.pose_provider(component.current_clip_index, component.current_frame)
            if component._blender is not None:
                pose = component._blender.apply(pose)
            component.current_pose = pose

    def _check_fallback(self, component: MotionMatchingComponent) -> FallbackReason:
        """Check if component should use fallback.

        Args:
            component: Component to check

        Returns:
            Fallback reason or NONE if no fallback needed
        """
        if not component.enabled:
            return FallbackReason.DISABLED

        if component.use_fallback:
            return FallbackReason.EXPLICIT_FLAG

        if component.database is None or component.database.entry_count == 0:
            return FallbackReason.DATABASE_EMPTY

        if component._frame_budget_used_ms >= component.config.budget_ms:
            return FallbackReason.BUDGET_EXCEEDED

        return FallbackReason.NONE

    def _should_search(self, component: MotionMatchingComponent) -> bool:
        """Check if a search should be performed.

        Args:
            component: Component to check

        Returns:
            True if search should be performed
        """
        # Check budget
        if component.config.budget_ms <= 0:
            return False

        if component._frame_budget_used_ms >= component.config.budget_ms:
            return False

        # Check search interval
        if component._time_since_search < component.config.min_search_interval:
            return False

        # Check transition interval
        if component._time_since_transition < component.config.min_transition_interval:
            return False

        return True

    def _perform_search(
        self,
        component: MotionMatchingComponent,
        input_state: MotionMatchingInput
    ) -> None:
        """Perform motion matching search.

        Args:
            component: Component to search for
            input_state: Current input state
        """
        if component._search is None or component.database is None:
            return

        start_time = time.perf_counter()

        # Build query from input
        query = self._build_query(component, input_state)

        # Configure search
        search_config = SearchConfig(
            max_results=component.config.max_candidates,
            required_tags=component.required_tags if component.required_tags else None,
        )

        # Perform search
        try:
            results = component._search.search(query, search_config)
        except Exception:
            results = []

        end_time = time.perf_counter()
        search_time_ms = (end_time - start_time) * 1000.0

        # Update budget
        component._frame_budget_used_ms += search_time_ms
        component._time_since_search = 0.0

        # Process results
        if results:
            best = results[0]
            component.statistics.record_query(search_time_ms, best.cost, True)

            # Check if we should transition
            if self._should_transition(component, best):
                self._start_transition(component, best)
        else:
            component.statistics.record_query(search_time_ms, 0.0, False)

    def _build_query(
        self,
        component: MotionMatchingComponent,
        input_state: MotionMatchingInput
    ) -> np.ndarray:
        """Build query feature vector from input.

        Args:
            component: Component context
            input_state: Current input state

        Returns:
            Query feature vector
        """
        # Create a basic query from the database's feature dimension
        if component.database is None:
            return np.zeros(47, dtype=np.float32)

        dim = component.database.feature_dimension
        query = np.zeros(dim, dtype=np.float32)

        # Fill with velocity if available
        if input_state.desired_velocity is not None:
            query[:min(3, dim)] = input_state.desired_velocity[:3]

        return query

    def _should_transition(
        self,
        component: MotionMatchingComponent,
        result: SearchResult
    ) -> bool:
        """Check if we should transition to a new match.

        Args:
            component: Component context
            result: Search result to check

        Returns:
            True if transition should occur
        """
        # Always transition if no current clip
        if component.current_clip_index < 0:
            return True

        # Check if cost is significantly better
        if result.cost < component.current_cost * 0.8:
            return True

        return False

    def _start_transition(
        self,
        component: MotionMatchingComponent,
        result: SearchResult
    ) -> None:
        """Start a transition to a new match.

        Args:
            component: Component to transition
            result: Search result to transition to
        """
        if self.pose_provider is None:
            return

        # Get poses for transition
        from_pose = component.current_pose or Pose()
        to_pose = self.pose_provider(result.entry.clip_index, result.entry.frame)

        # Create blender
        config = TransitionConfig(
            blend_duration=component.config.blend_time,
            blend_mode=BlendMode.INERTIALIZATION,
        )
        component._blender = InertializationBlender(config)
        component._blender.compute_offsets(from_pose, to_pose)

        # Update state
        component.current_clip_index = result.entry.clip_index
        component.current_frame = result.entry.frame
        component.current_time = 0.0
        component.current_cost = result.cost
        component._time_since_transition = 0.0

        component.statistics.record_transition()

    def _get_fallback_pose(
        self,
        component: MotionMatchingComponent,
        dt: float
    ) -> Optional[Pose]:
        """Get fallback pose from state machine.

        Args:
            component: Component context
            dt: Delta time

        Returns:
            Fallback pose or current pose
        """
        if self.state_machine_fallback is not None:
            return self.state_machine_fallback(component, dt)
        return component.current_pose

    def get_debug_info(self, component: MotionMatchingComponent) -> Dict[str, Any]:
        """Get debug information for a component.

        Args:
            component: Component to get info for

        Returns:
            Dictionary of debug information
        """
        return {
            "enabled": component.enabled,
            "is_ready": component.is_ready,
            "is_transitioning": component.is_transitioning,
            "mode": component.statistics.current_mode.name,
            "clip_index": component.current_clip_index,
            "frame": component.current_frame,
            "time": component.current_time,
            "current_cost": component.current_cost,
            "time_since_search": component._time_since_search,
            "time_since_transition": component._time_since_transition,
            "budget_used_ms": component._frame_budget_used_ms,
            "total_queries": component.statistics.total_queries,
            "successful_matches": component.statistics.successful_matches,
            "transitions": component.statistics.transitions_triggered,
            "fallbacks": component.statistics.fallback_count,
            "avg_search_time_ms": component.statistics.avg_search_time_ms,
            "avg_match_cost": component.statistics.avg_match_cost,
        }

    def reset_statistics(self, component: MotionMatchingComponent) -> None:
        """Reset statistics for a component.

        Args:
            component: Component to reset
        """
        component.statistics.reset()

    def get_statistics(
        self, component: MotionMatchingComponent
    ) -> MotionMatchingStatistics:
        """Get statistics for a component.

        Args:
            component: Component to get stats for

        Returns:
            Component statistics
        """
        return component.statistics
