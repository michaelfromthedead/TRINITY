"""Crowd level of detail (LOD) system.

Provides distance-based LOD selection for crowd characters, including
skeleton simplification and smooth LOD transitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence

from engine.core.math import Vec3, Quat, Transform

from .animation_texture import Skeleton
from engine.animation.config import CROWD_LOD_CONFIG


class LODTransitionMode(Enum):
    """Mode for LOD transition."""
    INSTANT = auto()  # Instant switch
    BLEND = auto()  # Smooth blend between LODs
    DITHER = auto()  # Screen-space dithering


@dataclass
class LODLevel:
    """Definition of a single LOD level.

    Attributes:
        distance: Minimum distance for this LOD
        bone_count: Number of bones to use at this LOD
        update_rate: Animation update rate (1.0 = every frame)
        mesh_reduction: Mesh polygon reduction factor (0.0-1.0)
        shadow_enabled: Whether shadows are enabled
        animation_quality: Animation sampling quality (1.0 = full)
    """
    distance: float = 0.0
    bone_count: int = 0
    update_rate: float = 1.0
    mesh_reduction: float = 0.0
    shadow_enabled: bool = True
    animation_quality: float = 1.0

    def __lt__(self, other: LODLevel) -> bool:
        return self.distance < other.distance

    def should_update(self, frame_counter: int) -> bool:
        """Check if animation should update this frame."""
        if self.update_rate >= 1.0:
            return True
        if self.update_rate <= 0:
            return False

        # Update every N frames based on rate
        update_interval = int(1.0 / self.update_rate)
        return frame_counter % update_interval == 0


@dataclass
class LODTransition:
    """Handles smooth transitions between LOD levels.

    Attributes:
        from_lod: Starting LOD level
        to_lod: Target LOD level
        progress: Transition progress (0-1)
        duration: Transition duration in seconds
        mode: Transition mode
    """
    from_lod: int = 0
    to_lod: int = 0
    progress: float = 0.0
    duration: float = CROWD_LOD_CONFIG.DEFAULT_TRANSITION_DURATION
    mode: LODTransitionMode = LODTransitionMode.BLEND
    active: bool = False

    def start(self, from_lod: int, to_lod: int) -> None:
        """Start a new transition."""
        if from_lod == to_lod:
            self.active = False
            return

        self.from_lod = from_lod
        self.to_lod = to_lod
        self.progress = 0.0
        self.active = True

    def update(self, dt: float) -> bool:
        """Update transition, returns True when complete."""
        if not self.active:
            return True

        if self.duration <= 0:
            self.progress = 1.0
            self.active = False
            return True

        self.progress += dt / self.duration
        if self.progress >= 1.0:
            self.progress = 1.0
            self.active = False
            return True

        return False

    def get_blend_factor(self) -> float:
        """Get smoothed blend factor using smoothstep."""
        t = max(0.0, min(1.0, self.progress))
        # Smoothstep for nicer transition
        return t * t * (3.0 - 2.0 * t)

    def get_current_lod(self) -> int:
        """Get current effective LOD level."""
        if not self.active or self.progress >= 0.5:
            return self.to_lod
        return self.from_lod


class CrowdLOD:
    """LOD manager for crowd characters.

    Manages LOD level definitions and provides distance-based LOD selection
    with support for skeleton simplification and smooth transitions.
    """

    def __init__(self, skeleton: Skeleton | None = None, levels: list[LODLevel] | None = None):
        self._lod_levels: list[LODLevel] = []
        self._skeleton = skeleton
        self._reduced_skeletons: dict[int, Skeleton] = {}
        self._hysteresis = CROWD_LOD_CONFIG.DEFAULT_HYSTERESIS  # Distance hysteresis to prevent LOD flickering
        self._frame_counter = 0
        self._max_lod = 0

        if levels is not None:
            self.set_lod_levels(levels)

    @property
    def lod_count(self) -> int:
        """Number of defined LOD levels."""
        return len(self._lod_levels)

    @property
    def max_lod(self) -> int:
        """Maximum LOD index."""
        return self._max_lod

    def add_lod_level(self, level: LODLevel) -> int:
        """Add a LOD level definition.

        Args:
            level: LOD level to add

        Returns:
            Index of the added LOD level
        """
        self._lod_levels.append(level)
        self._lod_levels.sort()
        self._max_lod = len(self._lod_levels) - 1

        # Pre-generate reduced skeleton if we have a skeleton
        if self._skeleton and level.bone_count < self._skeleton.bone_count:
            self._reduced_skeletons[level.bone_count] = create_reduced_skeleton(
                self._skeleton, level.bone_count
            )

        return len(self._lod_levels) - 1

    def set_lod_levels(self, levels: list[LODLevel]) -> None:
        """Set all LOD levels at once."""
        self._lod_levels = sorted(levels)
        self._max_lod = len(self._lod_levels) - 1

        # Generate reduced skeletons
        if self._skeleton:
            self._reduced_skeletons.clear()
            for level in self._lod_levels:
                if level.bone_count < self._skeleton.bone_count:
                    self._reduced_skeletons[level.bone_count] = create_reduced_skeleton(
                        self._skeleton, level.bone_count
                    )

    def get_lod_level(self, index: int) -> LODLevel | None:
        """Get LOD level by index."""
        if 0 <= index < len(self._lod_levels):
            return self._lod_levels[index]
        return None

    def get_lod_for_distance(self, distance: float, current_lod: int = -1) -> int:
        """Get appropriate LOD level for distance.

        Args:
            distance: Distance to camera (clamped to minimum to avoid issues at distance=0)
            current_lod: Current LOD level for hysteresis

        Returns:
            LOD level index
        """
        if not self._lod_levels:
            return 0

        # Clamp distance to minimum to avoid division by zero or edge cases at distance=0
        distance = max(0.0, distance)

        # Apply hysteresis if we have a current LOD
        hysteresis = self._hysteresis if current_lod >= 0 else 0.0

        for i, level in enumerate(self._lod_levels):
            threshold = level.distance

            # Add hysteresis when switching to higher detail (lower index)
            if current_lod > i:
                threshold -= hysteresis
            # Add hysteresis when switching to lower detail (higher index)
            elif current_lod < i:
                threshold += hysteresis

            # Ensure threshold doesn't go negative
            threshold = max(0.0, threshold)

            if distance < threshold:
                return max(0, i - 1) if i > 0 else 0

        return self._max_lod

    def get_skeleton_for_lod(self, lod_index: int) -> Skeleton | None:
        """Get skeleton for LOD level.

        Args:
            lod_index: LOD level index

        Returns:
            Skeleton for that LOD (may be reduced)
        """
        if not self._skeleton:
            return None

        level = self.get_lod_level(lod_index)
        if level is None:
            return self._skeleton

        if level.bone_count >= self._skeleton.bone_count:
            return self._skeleton

        return self._reduced_skeletons.get(level.bone_count, self._skeleton)

    def should_update_animation(self, lod_index: int) -> bool:
        """Check if animation should update for given LOD."""
        level = self.get_lod_level(lod_index)
        if level is None:
            return True
        return level.should_update(self._frame_counter)

    def advance_frame(self) -> None:
        """Advance frame counter for update rate calculations."""
        self._frame_counter += 1

    def set_hysteresis(self, distance: float) -> None:
        """Set LOD switching hysteresis distance."""
        self._hysteresis = max(0.0, distance)

    def get_bone_count_for_lod(self, lod_index: int) -> int:
        """Get bone count for LOD level."""
        level = self.get_lod_level(lod_index)
        if level is None and self._skeleton:
            return self._skeleton.bone_count
        if level is None:
            return 0
        return level.bone_count

    def create_default_lods(self, max_distance: float, lod_count: int = 4) -> None:
        """Create default LOD levels based on max distance.

        Args:
            max_distance: Maximum render distance
            lod_count: Number of LOD levels to create (clamped to MAX_LOD_LEVELS)
        """
        if not self._skeleton:
            return

        # Clamp to maximum LOD levels
        lod_count = min(lod_count, CROWD_LOD_CONFIG.MAX_LOD_LEVELS)

        # Avoid division by zero
        if lod_count <= 0:
            return

        self._lod_levels.clear()
        bone_count = self._skeleton.bone_count

        for i in range(lod_count):
            distance = max_distance * (i / lod_count)
            # Reduce bones progressively, with minimum from config
            lod_bones = max(
                CROWD_LOD_CONFIG.MIN_BONES_AT_LOWEST_LOD,
                bone_count - (bone_count // lod_count) * i
            )
            # Reduce update rate at distance
            update_rate = 1.0 - (i * 0.25)

            self.add_lod_level(LODLevel(
                distance=distance,
                bone_count=lod_bones,
                update_rate=max(CROWD_LOD_CONFIG.MIN_UPDATE_RATE, update_rate),
                mesh_reduction=i * 0.2,
                shadow_enabled=(i < 2),
                animation_quality=1.0 - (i * 0.2),
            ))


@dataclass
class BoneWeight:
    """Represents bone importance for LOD reduction."""
    bone_index: int
    importance: float  # 0-1, higher = more important

    def __lt__(self, other: BoneWeight) -> bool:
        return self.importance < other.importance


def create_reduced_skeleton(skeleton: Skeleton, target_bone_count: int) -> Skeleton:
    """Create a reduced skeleton with fewer bones.

    Bones are prioritized based on hierarchy depth and name patterns.

    Args:
        skeleton: Original skeleton
        target_bone_count: Target number of bones

    Returns:
        Reduced skeleton
    """
    if target_bone_count >= skeleton.bone_count:
        return skeleton

    if target_bone_count <= 0:
        return Skeleton(
            bone_names=[],
            bone_parents=[],
            bind_poses=[],
        )

    # Calculate bone importance
    weights = []
    for i, name in enumerate(skeleton.bone_names):
        importance = _calculate_bone_importance(name, i, skeleton)
        weights.append(BoneWeight(bone_index=i, importance=importance))

    # Sort by importance (descending)
    weights.sort(reverse=True)

    # Select top N bones
    selected_indices = sorted([w.bone_index for w in weights[:target_bone_count]])

    # Build index mapping
    old_to_new: dict[int, int] = {old: new for new, old in enumerate(selected_indices)}

    # Create reduced skeleton
    new_names = []
    new_parents = []
    new_bind_poses = []

    for new_idx, old_idx in enumerate(selected_indices):
        new_names.append(skeleton.bone_names[old_idx])

        # Find parent in new skeleton
        old_parent = skeleton.bone_parents[old_idx]
        new_parent = -1

        # Walk up hierarchy to find included ancestor
        while old_parent >= 0:
            if old_parent in old_to_new:
                new_parent = old_to_new[old_parent]
                break
            old_parent = skeleton.bone_parents[old_parent]

        new_parents.append(new_parent)
        new_bind_poses.append(skeleton.bind_poses[old_idx] if old_idx < len(skeleton.bind_poses) else Transform.identity())

    return Skeleton(
        bone_names=new_names,
        bone_parents=new_parents,
        bind_poses=new_bind_poses,
    )


def _calculate_bone_importance(name: str, index: int, skeleton: Skeleton) -> float:
    """Calculate importance score for a bone.

    Higher scores for:
    - Root bones
    - Spine/torso bones
    - Head
    - Upper limbs

    Lower scores for:
    - Fingers
    - Toes
    - Auxiliary bones (twist, helper)
    """
    score = 0.5  # Base score

    name_lower = name.lower()

    # Root/core bones are very important
    if index == 0 or "root" in name_lower:
        score += 0.5
    elif "pelvis" in name_lower or "hips" in name_lower:
        score += 0.45
    elif "spine" in name_lower:
        score += 0.4
    elif "chest" in name_lower or "torso" in name_lower:
        score += 0.35

    # Head is important
    elif "head" in name_lower:
        score += 0.4
    elif "neck" in name_lower:
        score += 0.35

    # Major limb bones
    elif "shoulder" in name_lower or "clavicle" in name_lower:
        score += 0.3
    elif "upperarm" in name_lower or "upper_arm" in name_lower:
        score += 0.3
    elif "forearm" in name_lower or "lowerarm" in name_lower:
        score += 0.25
    elif "hand" in name_lower and "finger" not in name_lower:
        score += 0.2

    # Legs
    elif "thigh" in name_lower or "upperleg" in name_lower:
        score += 0.3
    elif "calf" in name_lower or "lowerleg" in name_lower or "shin" in name_lower:
        score += 0.25
    elif "foot" in name_lower and "toe" not in name_lower:
        score += 0.2

    # Less important - fingers
    elif "finger" in name_lower:
        if "index" in name_lower or "thumb" in name_lower:
            score += 0.1
        else:
            score += 0.05

    # Less important - toes
    elif "toe" in name_lower:
        score += 0.02

    # Auxiliary bones
    if "twist" in name_lower or "roll" in name_lower or "helper" in name_lower:
        score -= 0.2

    # Hierarchy depth penalty (deeper bones less important)
    depth = _get_bone_depth(index, skeleton)
    score -= depth * 0.02

    return max(0.0, min(1.0, score))


def _get_bone_depth(bone_index: int, skeleton: Skeleton) -> int:
    """Calculate bone depth in hierarchy."""
    depth = 0
    current = bone_index
    while current >= 0 and skeleton.bone_parents[current] >= 0:
        current = skeleton.bone_parents[current]
        depth += 1
    return depth


def calculate_lod_blend_weights(
    full_skeleton: Skeleton,
    reduced_skeleton: Skeleton,
    bone_map: dict[int, int],  # reduced -> full index
) -> dict[int, list[tuple[int, float]]]:
    """Calculate blend weights for LOD transition.

    Maps bones from reduced skeleton to full skeleton with weights.

    Args:
        full_skeleton: Full detail skeleton
        reduced_skeleton: Reduced LOD skeleton
        bone_map: Mapping from reduced bone indices to full skeleton

    Returns:
        Dictionary mapping reduced bone index to list of (full_bone_idx, weight)
    """
    blend_weights: dict[int, list[tuple[int, float]]] = {}

    for reduced_idx in range(reduced_skeleton.bone_count):
        full_idx = bone_map.get(reduced_idx)
        if full_idx is not None:
            blend_weights[reduced_idx] = [(full_idx, 1.0)]
        else:
            # This shouldn't happen with proper mapping, but handle it
            blend_weights[reduced_idx] = []

    return blend_weights
