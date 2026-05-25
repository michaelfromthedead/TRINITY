"""Animation retargeting between skeletons.

This module provides functionality for:
- Mapping bones between source and target skeletons
- Retargeting poses while preserving proportions
- Handling different skeleton configurations
- Preserving contact points (e.g., feet grounded)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Callable

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.mat import Mat4
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON

from engine.animation.skeletal.constants import (
    RETARGET_POSITION_MATCH_THRESHOLD,
    DEFAULT_UNMAPPED_BLEND_FACTOR,
    DEFAULT_SCALE_FACTOR,
)

if TYPE_CHECKING:
    from engine.animation.skeletal.skeleton import Skeleton
    from engine.animation.skeletal.pose import Pose


class BoneMappingStrategy(Enum):
    """Strategy for automatic bone mapping."""
    BY_NAME = auto()           # Match by exact bone name
    BY_NAME_FUZZY = auto()     # Match by similar name (ignores case, prefixes)
    BY_HIERARCHY = auto()      # Match by hierarchy position
    BY_POSITION = auto()       # Match by bind pose position


@dataclass
class BoneMapping:
    """Mapping between a source and target bone.

    Attributes:
        source_index: Bone index in source skeleton
        target_index: Bone index in target skeleton
        source_name: Bone name in source skeleton
        target_name: Bone name in target skeleton
        rotation_offset: Additional rotation to apply during retarget
        translation_offset: Additional translation offset
        translation_mode: How to handle translation
    """
    source_index: int
    target_index: int
    source_name: str = ""
    target_name: str = ""
    rotation_offset: Quat = field(default_factory=Quat.identity)
    translation_offset: Vec3 = field(default_factory=Vec3.zero)
    translation_mode: str = "proportional"  # "proportional", "direct", "ignore"

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "source_name": self.source_name,
            "target_name": self.target_name,
            "rotation_offset": [
                self.rotation_offset.x, self.rotation_offset.y,
                self.rotation_offset.z, self.rotation_offset.w
            ],
            "translation_offset": [
                self.translation_offset.x,
                self.translation_offset.y,
                self.translation_offset.z
            ],
            "translation_mode": self.translation_mode
        }

    @staticmethod
    def from_dict(data: dict) -> BoneMapping:
        """Deserialize from dictionary."""
        rot = data.get("rotation_offset", [0, 0, 0, 1])
        trans = data.get("translation_offset", [0, 0, 0])
        return BoneMapping(
            source_index=data["source_index"],
            target_index=data["target_index"],
            source_name=data.get("source_name", ""),
            target_name=data.get("target_name", ""),
            rotation_offset=Quat(rot[0], rot[1], rot[2], rot[3]),
            translation_offset=Vec3(trans[0], trans[1], trans[2]),
            translation_mode=data.get("translation_mode", "proportional")
        )


@dataclass
class RetargetMap:
    """Complete mapping between source and target skeletons.

    Attributes:
        mappings: List of bone mappings
        source_bone_count: Number of bones in source skeleton
        target_bone_count: Number of bones in target skeleton
        unmapped_source_bones: Set of source bone indices without mapping
        unmapped_target_bones: Set of target bone indices without mapping
    """
    mappings: List[BoneMapping] = field(default_factory=list)
    source_bone_count: int = 0
    target_bone_count: int = 0
    unmapped_source_bones: Set[int] = field(default_factory=set)
    unmapped_target_bones: Set[int] = field(default_factory=set)

    # Index lookups for fast access
    _source_to_mapping: Dict[int, BoneMapping] = field(default_factory=dict, repr=False)
    _target_to_mapping: Dict[int, BoneMapping] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._build_indices()

    def _build_indices(self) -> None:
        """Build lookup indices for fast access."""
        self._source_to_mapping = {m.source_index: m for m in self.mappings}
        self._target_to_mapping = {m.target_index: m for m in self.mappings}

    def add_mapping(self, mapping: BoneMapping) -> None:
        """Add a bone mapping."""
        self.mappings.append(mapping)
        self._source_to_mapping[mapping.source_index] = mapping
        self._target_to_mapping[mapping.target_index] = mapping
        self.unmapped_source_bones.discard(mapping.source_index)
        self.unmapped_target_bones.discard(mapping.target_index)

    def remove_mapping(self, source_index: int) -> Optional[BoneMapping]:
        """Remove a mapping by source index."""
        if source_index not in self._source_to_mapping:
            return None

        mapping = self._source_to_mapping.pop(source_index)
        self._target_to_mapping.pop(mapping.target_index, None)
        self.mappings.remove(mapping)
        self.unmapped_source_bones.add(source_index)
        self.unmapped_target_bones.add(mapping.target_index)
        return mapping

    def get_mapping_for_source(self, source_index: int) -> Optional[BoneMapping]:
        """Get mapping for a source bone index."""
        return self._source_to_mapping.get(source_index)

    def get_mapping_for_target(self, target_index: int) -> Optional[BoneMapping]:
        """Get mapping for a target bone index."""
        return self._target_to_mapping.get(target_index)

    @property
    def mapped_count(self) -> int:
        return len(self.mappings)

    @property
    def coverage_ratio(self) -> float:
        """Ratio of mapped bones to total target bones."""
        if self.target_bone_count == 0:
            return 0.0
        return len(self.mappings) / self.target_bone_count

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "mappings": [m.to_dict() for m in self.mappings],
            "source_bone_count": self.source_bone_count,
            "target_bone_count": self.target_bone_count,
            "unmapped_source_bones": list(self.unmapped_source_bones),
            "unmapped_target_bones": list(self.unmapped_target_bones)
        }

    @staticmethod
    def from_dict(data: dict) -> RetargetMap:
        """Deserialize from dictionary."""
        rm = RetargetMap(
            mappings=[BoneMapping.from_dict(m) for m in data["mappings"]],
            source_bone_count=data["source_bone_count"],
            target_bone_count=data["target_bone_count"],
            unmapped_source_bones=set(data.get("unmapped_source_bones", [])),
            unmapped_target_bones=set(data.get("unmapped_target_bones", []))
        )
        rm._build_indices()
        return rm


@dataclass
class RetargetConfig:
    """Configuration for retargeting behavior.

    Attributes:
        scale_factor: Global scale factor (target / source)
        per_bone_scale: Optional per-bone scale factors
        rotation_offsets: Per-bone rotation adjustments
        preserve_root_height: Keep root at same relative height
        preserve_foot_contact: Ensure feet stay grounded
        foot_bone_names: Names of foot bones for contact preservation
        ik_enabled: Use IK for foot placement adjustment
        blend_unmapped: Blend unmapped bones toward identity
    """
    scale_factor: float = DEFAULT_SCALE_FACTOR
    per_bone_scale: Optional[Dict[int, float]] = None
    rotation_offsets: Optional[Dict[int, Quat]] = None
    preserve_root_height: bool = True
    preserve_foot_contact: bool = True
    foot_bone_names: List[str] = field(default_factory=lambda: ["foot_l", "foot_r"])
    ik_enabled: bool = False
    blend_unmapped: bool = True
    unmapped_blend_factor: float = DEFAULT_UNMAPPED_BLEND_FACTOR

    def get_scale_for_bone(self, bone_index: int) -> float:
        """Get scale factor for a specific bone."""
        if self.per_bone_scale and bone_index in self.per_bone_scale:
            return self.per_bone_scale[bone_index]
        return self.scale_factor

    def get_rotation_offset(self, bone_index: int) -> Quat:
        """Get rotation offset for a specific bone."""
        if self.rotation_offsets and bone_index in self.rotation_offsets:
            return self.rotation_offsets[bone_index]
        return Quat.identity()


@dataclass
class SkeletonInfo:
    """Minimal skeleton information for retargeting.

    This allows retargeting without requiring the full Skeleton class.

    Attributes:
        bone_names: List of bone names
        bone_parents: List of parent indices (-1 for root)
        bind_translations: Local bind pose translations
        bind_rotations: Local bind pose rotations
        bind_world_positions: World-space bind positions
    """
    bone_names: List[str] = field(default_factory=list)
    bone_parents: List[int] = field(default_factory=list)
    bind_translations: List[Vec3] = field(default_factory=list)
    bind_rotations: List[Quat] = field(default_factory=list)
    bind_world_positions: Optional[List[Vec3]] = None

    @property
    def bone_count(self) -> int:
        return len(self.bone_names)

    def get_bone_index(self, name: str) -> Optional[int]:
        """Get bone index by name."""
        try:
            return self.bone_names.index(name)
        except ValueError:
            return None

    def get_bone_length(self, bone_index: int) -> float:
        """Get bone length from bind pose."""
        if self.bind_world_positions is None or bone_index >= len(self.bind_world_positions):
            return 0.0

        parent = self.bone_parents[bone_index]
        if parent < 0:
            return 0.0

        pos = self.bind_world_positions[bone_index]
        parent_pos = self.bind_world_positions[parent]
        return (pos - parent_pos).length()


def _normalize_bone_name(name: str) -> str:
    """Normalize bone name for fuzzy matching.

    Removes common prefixes, converts to lowercase, removes underscores.
    """
    name = name.lower()
    # Remove common prefixes
    prefixes = ["bip01_", "bip_", "def_", "bn_", "bone_", "j_", "jnt_"]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Remove underscores and common suffixes
    name = name.replace("_", "")
    suffixes = ["_l", "_r", "_left", "_right", "left", "right", "_001", "_002"]
    for suffix in suffixes:
        if name.endswith(suffix.replace("_", "")):
            base = name[:-len(suffix.replace("_", ""))]
            # Keep side indicator
            if "l" in suffix or "left" in suffix:
                name = base + "l"
            elif "r" in suffix or "right" in suffix:
                name = base + "r"
            break
    return name


def _match_bones_by_name(
    source_names: List[str],
    target_names: List[str],
    fuzzy: bool = False
) -> List[Tuple[int, int]]:
    """Match bones by name.

    Args:
        source_names: Source skeleton bone names
        target_names: Target skeleton bone names
        fuzzy: Use fuzzy matching

    Returns:
        List of (source_index, target_index) pairs
    """
    matches = []

    if fuzzy:
        normalized_source = {_normalize_bone_name(n): i for i, n in enumerate(source_names)}
        for target_idx, target_name in enumerate(target_names):
            normalized_target = _normalize_bone_name(target_name)
            if normalized_target in normalized_source:
                source_idx = normalized_source[normalized_target]
                matches.append((source_idx, target_idx))
    else:
        source_name_map = {n: i for i, n in enumerate(source_names)}
        for target_idx, target_name in enumerate(target_names):
            if target_name in source_name_map:
                matches.append((source_name_map[target_name], target_idx))

    return matches


def create_retarget_map(
    source_skeleton: SkeletonInfo,
    target_skeleton: SkeletonInfo,
    strategy: BoneMappingStrategy = BoneMappingStrategy.BY_NAME_FUZZY,
    manual_mappings: Optional[Dict[str, str]] = None
) -> RetargetMap:
    """Create a retarget map between two skeletons.

    Args:
        source_skeleton: Source skeleton info
        target_skeleton: Target skeleton info
        strategy: Automatic mapping strategy
        manual_mappings: Optional dict of manual mappings {target_name: source_name}

    Returns:
        RetargetMap for retargeting poses
    """
    retarget_map = RetargetMap(
        source_bone_count=source_skeleton.bone_count,
        target_bone_count=target_skeleton.bone_count,
        unmapped_source_bones=set(range(source_skeleton.bone_count)),
        unmapped_target_bones=set(range(target_skeleton.bone_count))
    )

    # Apply manual mappings first
    if manual_mappings:
        for target_name, source_name in manual_mappings.items():
            source_idx = source_skeleton.get_bone_index(source_name)
            target_idx = target_skeleton.get_bone_index(target_name)
            if source_idx is not None and target_idx is not None:
                mapping = BoneMapping(
                    source_index=source_idx,
                    target_index=target_idx,
                    source_name=source_name,
                    target_name=target_name
                )
                retarget_map.add_mapping(mapping)

    # Auto-map remaining bones
    if strategy in (BoneMappingStrategy.BY_NAME, BoneMappingStrategy.BY_NAME_FUZZY):
        fuzzy = strategy == BoneMappingStrategy.BY_NAME_FUZZY
        matches = _match_bones_by_name(
            source_skeleton.bone_names,
            target_skeleton.bone_names,
            fuzzy=fuzzy
        )

        for source_idx, target_idx in matches:
            # Skip if already mapped
            if source_idx not in retarget_map.unmapped_source_bones:
                continue
            if target_idx not in retarget_map.unmapped_target_bones:
                continue

            mapping = BoneMapping(
                source_index=source_idx,
                target_index=target_idx,
                source_name=source_skeleton.bone_names[source_idx],
                target_name=target_skeleton.bone_names[target_idx]
            )
            retarget_map.add_mapping(mapping)

    elif strategy == BoneMappingStrategy.BY_POSITION:
        # Match by closest bind pose position
        if (source_skeleton.bind_world_positions is not None and
            target_skeleton.bind_world_positions is not None):

            for target_idx in list(retarget_map.unmapped_target_bones):
                target_pos = target_skeleton.bind_world_positions[target_idx]

                best_source = -1
                best_dist = float('inf')

                for source_idx in retarget_map.unmapped_source_bones:
                    source_pos = source_skeleton.bind_world_positions[source_idx]
                    dist = (source_pos - target_pos).length()
                    if dist < best_dist:
                        best_dist = dist
                        best_source = source_idx

                if best_source >= 0 and best_dist < RETARGET_POSITION_MATCH_THRESHOLD:
                    mapping = BoneMapping(
                        source_index=best_source,
                        target_index=target_idx,
                        source_name=source_skeleton.bone_names[best_source],
                        target_name=target_skeleton.bone_names[target_idx]
                    )
                    retarget_map.add_mapping(mapping)

    return retarget_map


@dataclass
class PoseData:
    """Simple pose representation for retargeting.

    Attributes:
        local_translations: Per-bone local translations
        local_rotations: Per-bone local rotations
        local_scales: Optional per-bone local scales
    """
    local_translations: List[Vec3] = field(default_factory=list)
    local_rotations: List[Quat] = field(default_factory=list)
    local_scales: Optional[List[Vec3]] = None

    @property
    def bone_count(self) -> int:
        return len(self.local_rotations)

    @staticmethod
    def identity(bone_count: int) -> PoseData:
        """Create an identity pose."""
        return PoseData(
            local_translations=[Vec3.zero() for _ in range(bone_count)],
            local_rotations=[Quat.identity() for _ in range(bone_count)]
        )


def compute_scale_factor(
    source_skeleton: SkeletonInfo,
    target_skeleton: SkeletonInfo,
    reference_bone: Optional[str] = None
) -> float:
    """Compute scale factor between skeletons.

    Args:
        source_skeleton: Source skeleton info
        target_skeleton: Target skeleton info
        reference_bone: Optional bone to use for scale calculation

    Returns:
        Scale factor (target / source)
    """
    if reference_bone:
        source_idx = source_skeleton.get_bone_index(reference_bone)
        target_idx = target_skeleton.get_bone_index(reference_bone)
        if source_idx is not None and target_idx is not None:
            source_len = source_skeleton.get_bone_length(source_idx)
            target_len = target_skeleton.get_bone_length(target_idx)
            if source_len > MATH_EPSILON:
                return target_len / source_len

    # Use average bone length as fallback
    source_total = 0.0
    source_count = 0
    for i in range(source_skeleton.bone_count):
        length = source_skeleton.get_bone_length(i)
        if length > MATH_EPSILON:
            source_total += length
            source_count += 1

    target_total = 0.0
    target_count = 0
    for i in range(target_skeleton.bone_count):
        length = target_skeleton.get_bone_length(i)
        if length > MATH_EPSILON:
            target_total += length
            target_count += 1

    if source_count > 0 and target_count > 0:
        source_avg = source_total / source_count
        target_avg = target_total / target_count
        if source_avg > MATH_EPSILON:
            return target_avg / source_avg

    return 1.0


def retarget_pose(
    source_pose: PoseData,
    source_skeleton: SkeletonInfo,
    target_skeleton: SkeletonInfo,
    retarget_map: RetargetMap,
    config: Optional[RetargetConfig] = None
) -> PoseData:
    """Retarget a pose from source to target skeleton.

    Args:
        source_pose: Source skeleton pose
        source_skeleton: Source skeleton info
        target_skeleton: Target skeleton info
        retarget_map: Bone mapping
        config: Retarget configuration

    Returns:
        Retargeted pose for target skeleton
    """
    if config is None:
        config = RetargetConfig()

    # Start with bind pose of target
    result = PoseData(
        local_translations=list(target_skeleton.bind_translations),
        local_rotations=list(target_skeleton.bind_rotations)
    )

    # Process each mapping
    for mapping in retarget_map.mappings:
        source_idx = mapping.source_index
        target_idx = mapping.target_index

        if source_idx >= len(source_pose.local_rotations):
            continue
        if target_idx >= len(result.local_rotations):
            continue

        # Get source rotation
        source_rot = source_pose.local_rotations[source_idx]

        # Apply rotation offset from mapping
        if mapping.rotation_offset != Quat.identity():
            source_rot = mapping.rotation_offset * source_rot * mapping.rotation_offset.inverse()

        # Apply rotation offset from config
        config_offset = config.get_rotation_offset(target_idx)
        if config_offset != Quat.identity():
            source_rot = config_offset * source_rot * config_offset.inverse()

        # Set target rotation
        result.local_rotations[target_idx] = source_rot

        # Handle translation based on mode
        if mapping.translation_mode == "direct":
            # Use source translation directly (scaled)
            scale = config.get_scale_for_bone(target_idx)
            result.local_translations[target_idx] = source_pose.local_translations[source_idx] * scale
        elif mapping.translation_mode == "proportional":
            # Scale translation proportionally to skeleton ratio
            source_trans = source_pose.local_translations[source_idx]
            source_bind = source_skeleton.bind_translations[source_idx]
            target_bind = target_skeleton.bind_translations[target_idx]

            # Compute delta from bind pose
            delta = source_trans - source_bind

            # Scale and apply to target
            scale = config.get_scale_for_bone(target_idx)
            result.local_translations[target_idx] = target_bind + delta * scale
        # "ignore" mode keeps target bind pose translation

        # Apply translation offset
        if mapping.translation_offset != Vec3.zero():
            result.local_translations[target_idx] = \
                result.local_translations[target_idx] + mapping.translation_offset

    # Handle unmapped bones
    if config.blend_unmapped:
        for target_idx in retarget_map.unmapped_target_bones:
            if target_idx < len(result.local_rotations):
                # Blend toward identity
                bind_rot = target_skeleton.bind_rotations[target_idx]
                result.local_rotations[target_idx] = bind_rot.slerp(
                    Quat.identity(),
                    config.unmapped_blend_factor
                )

    return result


def preserve_foot_contact(
    pose: PoseData,
    skeleton: SkeletonInfo,
    foot_bone_indices: List[int],
    ground_height: float = 0.0
) -> PoseData:
    """Adjust pose to keep feet at ground height.

    This is a simplified foot contact preservation. For production use,
    integrate with the IK system for proper foot placement.

    Args:
        pose: Current pose
        skeleton: Skeleton info
        foot_bone_indices: Indices of foot bones
        ground_height: Target ground Y value

    Returns:
        Adjusted pose
    """
    if skeleton.bind_world_positions is None:
        return pose

    # Compute world positions for current pose
    # (simplified - would need full forward kinematics in production)
    result = PoseData(
        local_translations=list(pose.local_translations),
        local_rotations=list(pose.local_rotations)
    )

    # Find lowest foot position
    min_foot_height = float('inf')
    for foot_idx in foot_bone_indices:
        if foot_idx < len(skeleton.bind_world_positions):
            # Approximate world position (very simplified)
            pos = skeleton.bind_world_positions[foot_idx]
            if pos.y < min_foot_height:
                min_foot_height = pos.y

    # Adjust root bone to compensate
    if min_foot_height != float('inf') and min_foot_height < ground_height:
        adjustment = ground_height - min_foot_height
        # Find root bone (parent = -1)
        for i, parent in enumerate(skeleton.bone_parents):
            if parent < 0:
                current_trans = result.local_translations[i]
                result.local_translations[i] = Vec3(
                    current_trans.x,
                    current_trans.y + adjustment,
                    current_trans.z
                )
                break

    return result


class RetargetPipeline:
    """Complete retargeting pipeline for runtime use.

    Caches computed data and provides efficient retargeting for
    continuous animation playback.
    """

    def __init__(
        self,
        source_skeleton: SkeletonInfo,
        target_skeleton: SkeletonInfo,
        config: Optional[RetargetConfig] = None
    ) -> None:
        self._source_skeleton = source_skeleton
        self._target_skeleton = target_skeleton
        self._config = config or RetargetConfig()
        self._retarget_map: Optional[RetargetMap] = None

        # Auto-compute scale factor
        if abs(self._config.scale_factor - 1.0) < MATH_EPSILON:
            self._config.scale_factor = compute_scale_factor(
                source_skeleton, target_skeleton
            )

    def create_mapping(
        self,
        strategy: BoneMappingStrategy = BoneMappingStrategy.BY_NAME_FUZZY,
        manual_mappings: Optional[Dict[str, str]] = None
    ) -> RetargetMap:
        """Create or recreate the bone mapping."""
        self._retarget_map = create_retarget_map(
            self._source_skeleton,
            self._target_skeleton,
            strategy=strategy,
            manual_mappings=manual_mappings
        )
        return self._retarget_map

    def set_mapping(self, retarget_map: RetargetMap) -> None:
        """Set a pre-computed mapping."""
        self._retarget_map = retarget_map

    def retarget(self, source_pose: PoseData) -> PoseData:
        """Retarget a pose."""
        if self._retarget_map is None:
            self.create_mapping()

        result = retarget_pose(
            source_pose,
            self._source_skeleton,
            self._target_skeleton,
            self._retarget_map,
            self._config
        )

        # Optionally preserve foot contact
        if self._config.preserve_foot_contact:
            foot_indices = []
            for name in self._config.foot_bone_names:
                idx = self._target_skeleton.get_bone_index(name)
                if idx is not None:
                    foot_indices.append(idx)

            if foot_indices:
                result = preserve_foot_contact(
                    result,
                    self._target_skeleton,
                    foot_indices,
                    ground_height=0.0
                )

        return result

    @property
    def mapping(self) -> Optional[RetargetMap]:
        return self._retarget_map

    @property
    def config(self) -> RetargetConfig:
        return self._config

    @property
    def scale_factor(self) -> float:
        return self._config.scale_factor


def validate_retarget_map(
    retarget_map: RetargetMap,
    source_skeleton: SkeletonInfo,
    target_skeleton: SkeletonInfo
) -> List[str]:
    """Validate a retarget map against skeletons.

    Args:
        retarget_map: Mapping to validate
        source_skeleton: Source skeleton info
        target_skeleton: Target skeleton info

    Returns:
        List of validation error messages
    """
    errors = []

    if retarget_map.source_bone_count != source_skeleton.bone_count:
        errors.append(
            f"Source bone count mismatch: map has {retarget_map.source_bone_count}, "
            f"skeleton has {source_skeleton.bone_count}"
        )

    if retarget_map.target_bone_count != target_skeleton.bone_count:
        errors.append(
            f"Target bone count mismatch: map has {retarget_map.target_bone_count}, "
            f"skeleton has {target_skeleton.bone_count}"
        )

    for mapping in retarget_map.mappings:
        if mapping.source_index < 0 or mapping.source_index >= source_skeleton.bone_count:
            errors.append(f"Invalid source bone index: {mapping.source_index}")

        if mapping.target_index < 0 or mapping.target_index >= target_skeleton.bone_count:
            errors.append(f"Invalid target bone index: {mapping.target_index}")

        if mapping.source_name and mapping.source_name != source_skeleton.bone_names[mapping.source_index]:
            errors.append(
                f"Source name mismatch at index {mapping.source_index}: "
                f"expected '{source_skeleton.bone_names[mapping.source_index]}', "
                f"got '{mapping.source_name}'"
            )

    if retarget_map.mapped_count == 0:
        errors.append("No bones are mapped")

    return errors
