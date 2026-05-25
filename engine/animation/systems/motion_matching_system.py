"""ECS system for motion matching.

Provides data-driven animation selection using motion matching techniques.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Sequence

from engine.core.math import Vec3, Quat, Transform
from engine.core.ecs import Entity, World
from engine.animation.config import ANIMATION_SYSTEM_CONFIG


class FeatureType(Enum):
    """Type of motion feature."""
    POSITION = auto()  # 3D position
    VELOCITY = auto()  # 3D velocity
    DIRECTION = auto()  # 3D direction
    TRAJECTORY = auto()  # Future/past trajectory points


@dataclass
class MotionFeature:
    """Feature used for motion matching.

    Attributes:
        name: Feature name
        feature_type: Type of feature
        bone_index: Bone to track (-1 for root)
        weight: Feature weight in matching
        trajectory_times: Times for trajectory features (seconds)
    """
    name: str = ""
    feature_type: FeatureType = FeatureType.POSITION
    bone_index: int = -1
    weight: float = 1.0
    trajectory_times: list[float] = field(default_factory=list)


@dataclass
class MotionInput:
    """Input state for motion matching.

    Attributes:
        desired_velocity: Desired movement velocity
        desired_direction: Desired facing direction
        trajectory: Future trajectory points (positions)
        trajectory_times: Times for trajectory points
        features: Additional feature values
    """
    desired_velocity: Vec3 = field(default_factory=Vec3.zero)
    desired_direction: Vec3 = field(default_factory=Vec3.forward)
    trajectory: list[Vec3] = field(default_factory=list)
    trajectory_times: list[float] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionFrame:
    """Single frame of motion data.

    Attributes:
        animation_index: Index of source animation
        frame_index: Frame within animation
        feature_vector: Precomputed feature values
        root_position: Root position at this frame
        root_rotation: Root rotation at this frame
    """
    animation_index: int = 0
    frame_index: int = 0
    feature_vector: list[float] = field(default_factory=list)
    root_position: Vec3 = field(default_factory=Vec3.zero)
    root_rotation: Quat = field(default_factory=Quat.identity)


@dataclass
class MotionDatabase:
    """Database of motion frames for matching.

    Attributes:
        frames: All motion frames
        features: Feature definitions
        animation_names: Names of animations
        frame_rate: Database frame rate
    """
    frames: list[MotionFrame] = field(default_factory=list)
    features: list[MotionFeature] = field(default_factory=list)
    animation_names: list[str] = field(default_factory=list)
    frame_rate: float = 30.0

    @property
    def feature_count(self) -> int:
        """Number of features in feature vector."""
        return sum(self._feature_dimension(f) for f in self.features)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def _feature_dimension(self, feature: MotionFeature) -> int:
        """Get dimension of feature vector component."""
        if feature.feature_type == FeatureType.TRAJECTORY:
            return len(feature.trajectory_times) * 3
        elif feature.feature_type in (FeatureType.POSITION, FeatureType.VELOCITY, FeatureType.DIRECTION):
            return 3
        return 1

    def add_frame(self, frame: MotionFrame) -> int:
        """Add frame to database."""
        self.frames.append(frame)
        return len(self.frames) - 1

    def get_frame(self, index: int) -> MotionFrame | None:
        """Get frame by index."""
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None


@dataclass
class MotionMatchResult:
    """Result of motion matching search."""
    frame_index: int = -1
    cost: float = float('inf')
    animation_index: int = -1
    animation_frame: int = 0


@dataclass
class MotionMatchingController:
    """Controller for motion matching.

    Attributes:
        database: Motion database to search
        current_frame: Current database frame index
        transition_time: Time for blending between matches
        search_interval: Frames between searches
        pose_cost_weight: Weight for pose cost in matching
        trajectory_cost_weight: Weight for trajectory cost
    """
    database: MotionDatabase | None = None
    current_frame: int = 0
    transition_time: float = ANIMATION_SYSTEM_CONFIG.DEFAULT_MOTION_MATCH_TRANSITION
    search_interval: int = ANIMATION_SYSTEM_CONFIG.MOTION_MATCH_SEARCH_INTERVAL
    pose_cost_weight: float = 1.0
    trajectory_cost_weight: float = 1.0

    # Internal state
    _frames_since_search: int = 0
    _transition_progress: float = 1.0
    _previous_frame: int = -1
    _accumulated_time: float = 0.0


@dataclass
class MotionMatchingComponent:
    """Component for entities using motion matching.

    Attributes:
        controller: Motion matching controller
        input_provider: Function to get current input
        enabled: Whether motion matching is enabled
        output_animation: Current animation index
        output_time: Current animation time
    """
    controller: MotionMatchingController = field(default_factory=MotionMatchingController)
    input_provider: Callable[[], MotionInput] | None = None
    enabled: bool = True

    # Output
    output_animation: int = 0
    output_time: float = 0.0
    output_pose: dict[int, Transform] = field(default_factory=dict)


class MotionMatchingSystem:
    """ECS system for motion matching.

    Updates motion matching controllers and produces animation outputs.
    """

    def __init__(self):
        self._pose_provider: Callable[[int, int], dict[int, Transform]] | None = None

    def set_pose_provider(
        self,
        provider: Callable[[int, int], dict[int, Transform]]
    ) -> None:
        """Set function that provides poses.

        Args:
            provider: Function(animation_index, frame) -> pose
        """
        self._pose_provider = provider

    def update(
        self,
        world: World,
        dt: float,
        entity_components: list[tuple[Entity, MotionMatchingComponent]]
    ) -> None:
        """Update all motion matching components.

        Args:
            world: ECS world
            dt: Delta time
            entity_components: List of (entity, component) tuples
        """
        for entity, component in entity_components:
            if not component.enabled:
                continue

            self._update_component(component, dt)

    def _update_component(self, component: MotionMatchingComponent, dt: float) -> None:
        """Update single motion matching component."""
        controller = component.controller

        if not controller.database:
            return

        # Get current input
        input_state = MotionInput()
        if component.input_provider:
            input_state = component.input_provider()

        controller._accumulated_time += dt
        controller._frames_since_search += 1

        # Perform search if needed
        if controller._frames_since_search >= controller.search_interval:
            controller._frames_since_search = 0

            query = self._build_query(input_state, controller.database.features)
            result = self._search_database(query, controller)

            if result.frame_index >= 0 and result.cost < self._get_continuation_cost(controller):
                # Transition to new match
                controller._previous_frame = controller.current_frame
                controller.current_frame = result.frame_index
                controller._transition_progress = 0.0

        # Update transition
        if controller._transition_progress < 1.0:
            controller._transition_progress += dt / controller.transition_time
            controller._transition_progress = min(1.0, controller._transition_progress)

        # Advance current frame
        frame_advance = int(controller._accumulated_time * controller.database.frame_rate)
        if frame_advance > 0:
            controller._accumulated_time -= frame_advance / controller.database.frame_rate
            controller.current_frame = (controller.current_frame + frame_advance) % controller.database.frame_count

        # Update output
        current_frame = controller.database.get_frame(controller.current_frame)
        if current_frame:
            component.output_animation = current_frame.animation_index
            component.output_time = current_frame.frame_index / controller.database.frame_rate

            # Get pose
            if self._pose_provider:
                current_pose = self._pose_provider(current_frame.animation_index, current_frame.frame_index)

                # Blend with previous if transitioning
                if controller._transition_progress < 1.0 and controller._previous_frame >= 0:
                    prev_frame = controller.database.get_frame(controller._previous_frame)
                    if prev_frame:
                        prev_pose = self._pose_provider(prev_frame.animation_index, prev_frame.frame_index)
                        current_pose = self._blend_poses(prev_pose, current_pose, controller._transition_progress)

                component.output_pose = current_pose

    def _build_query(
        self,
        input_state: MotionInput,
        features: list[MotionFeature]
    ) -> list[float]:
        """Build query feature vector from input state."""
        query = []

        for feature in features:
            if feature.feature_type == FeatureType.VELOCITY:
                query.extend([
                    input_state.desired_velocity.x,
                    input_state.desired_velocity.y,
                    input_state.desired_velocity.z,
                ])
            elif feature.feature_type == FeatureType.DIRECTION:
                query.extend([
                    input_state.desired_direction.x,
                    input_state.desired_direction.y,
                    input_state.desired_direction.z,
                ])
            elif feature.feature_type == FeatureType.TRAJECTORY:
                for t_time in feature.trajectory_times:
                    # Find closest trajectory point
                    pos = Vec3.zero()
                    for i, input_time in enumerate(input_state.trajectory_times):
                        if abs(input_time - t_time) < 0.1 and i < len(input_state.trajectory):
                            pos = input_state.trajectory[i]
                            break
                    query.extend([pos.x, pos.y, pos.z])
            elif feature.feature_type == FeatureType.POSITION:
                value = input_state.features.get(feature.name, Vec3.zero())
                if isinstance(value, Vec3):
                    query.extend([value.x, value.y, value.z])
                else:
                    query.extend([0, 0, 0])

        return query

    def _search_database(
        self,
        query: list[float],
        controller: MotionMatchingController
    ) -> MotionMatchResult:
        """Search database for best matching frame."""
        database = controller.database
        if not database or not database.frames:
            return MotionMatchResult()

        best_result = MotionMatchResult()
        feature_weights = [f.weight for f in database.features]

        for i, frame in enumerate(database.frames):
            cost = self._compute_cost(query, frame.feature_vector, feature_weights, database.features)

            if cost < best_result.cost:
                best_result.frame_index = i
                best_result.cost = cost
                best_result.animation_index = frame.animation_index
                best_result.animation_frame = frame.frame_index

        return best_result

    def _compute_cost(
        self,
        query: list[float],
        frame_features: list[float],
        weights: list[float],
        features: list[MotionFeature]
    ) -> float:
        """Compute matching cost between query and frame features."""
        if len(query) != len(frame_features):
            return float('inf')

        total_cost = 0.0
        idx = 0

        for i, feature in enumerate(features):
            weight = weights[i] if i < len(weights) else 1.0
            dim = self._feature_dimension(feature)

            feature_cost = 0.0
            for j in range(dim):
                if idx + j < len(query):
                    diff = query[idx + j] - frame_features[idx + j]
                    feature_cost += diff * diff

            total_cost += feature_cost * weight
            idx += dim

        return total_cost

    def _feature_dimension(self, feature: MotionFeature) -> int:
        """Get dimension of feature."""
        if feature.feature_type == FeatureType.TRAJECTORY:
            return len(feature.trajectory_times) * 3
        return 3

    def _get_continuation_cost(self, controller: MotionMatchingController) -> float:
        """Get cost threshold for continuing current animation.

        Lower values make system more likely to switch.
        """
        return ANIMATION_SYSTEM_CONFIG.MOTION_MATCH_CONTINUATION_COST  # Tunable threshold

    def _blend_poses(
        self,
        pose_a: dict[int, Transform],
        pose_b: dict[int, Transform],
        blend: float
    ) -> dict[int, Transform]:
        """Blend two poses together."""
        result = {}
        all_bones = set(pose_a.keys()) | set(pose_b.keys())

        for bone in all_bones:
            t_a = pose_a.get(bone, Transform.identity())
            t_b = pose_b.get(bone, Transform.identity())
            result[bone] = t_a.lerp(t_b, blend)

        return result

    def build_database(
        self,
        animations: list[tuple[str, list[dict[int, Transform]]]],
        features: list[MotionFeature],
        frame_rate: float = 30.0
    ) -> MotionDatabase:
        """Build motion database from animations.

        Args:
            animations: List of (name, frames) where frames is list of poses
            features: Feature definitions
            frame_rate: Frame rate of animations

        Returns:
            Built motion database
        """
        database = MotionDatabase(
            features=features,
            frame_rate=frame_rate,
        )

        for anim_idx, (name, frames) in enumerate(animations):
            database.animation_names.append(name)

            for frame_idx, pose in enumerate(frames):
                feature_vector = self._extract_features(pose, features, frames, frame_idx, frame_rate)

                root_transform = pose.get(0, Transform.identity())

                motion_frame = MotionFrame(
                    animation_index=anim_idx,
                    frame_index=frame_idx,
                    feature_vector=feature_vector,
                    root_position=root_transform.translation,
                    root_rotation=root_transform.rotation,
                )

                database.add_frame(motion_frame)

        return database

    def _extract_features(
        self,
        pose: dict[int, Transform],
        features: list[MotionFeature],
        all_frames: list[dict[int, Transform]],
        frame_idx: int,
        frame_rate: float
    ) -> list[float]:
        """Extract feature vector from pose."""
        result = []

        for feature in features:
            if feature.feature_type == FeatureType.POSITION:
                transform = pose.get(feature.bone_index, Transform.identity())
                result.extend([transform.translation.x, transform.translation.y, transform.translation.z])

            elif feature.feature_type == FeatureType.VELOCITY:
                # Compute velocity from adjacent frames
                velocity = Vec3.zero()
                if frame_idx > 0:
                    prev_transform = all_frames[frame_idx - 1].get(feature.bone_index, Transform.identity())
                    curr_transform = pose.get(feature.bone_index, Transform.identity())
                    delta = curr_transform.translation - prev_transform.translation
                    velocity = delta * frame_rate

                result.extend([velocity.x, velocity.y, velocity.z])

            elif feature.feature_type == FeatureType.DIRECTION:
                transform = pose.get(feature.bone_index, Transform.identity())
                forward = transform.rotation.forward()
                result.extend([forward.x, forward.y, forward.z])

            elif feature.feature_type == FeatureType.TRAJECTORY:
                for t_time in feature.trajectory_times:
                    future_frame = frame_idx + int(t_time * frame_rate)
                    if 0 <= future_frame < len(all_frames):
                        transform = all_frames[future_frame].get(feature.bone_index, Transform.identity())
                        result.extend([transform.translation.x, transform.translation.y, transform.translation.z])
                    else:
                        # Extrapolate or use current
                        transform = pose.get(feature.bone_index, Transform.identity())
                        result.extend([transform.translation.x, transform.translation.y, transform.translation.z])

        return result
