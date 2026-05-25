"""
Core animation graph implementation.

Provides the foundational classes for animation graphs:
- AnimationGraph: Container for nodes, connections, and evaluation
- GraphParameter: Typed parameters that drive animation blending
- AnimationNode: Base class for all graph nodes
- GraphContext: Evaluation context with parameters and skeleton info

The animation graph is a directed acyclic graph (DAG) where nodes produce
poses and connections route data between nodes.
"""

from __future__ import annotations

import math
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar
from weakref import WeakValueDictionary

from trinity.metaclasses.engine_meta import EngineMeta

from .config import get_config


# =============================================================================
# SLOT TYPE SYSTEM
# =============================================================================


class SlotType(Enum):
    """Types for typed input/output slots on animation nodes."""

    POSE = auto()     # Pose (bone transforms)
    FLOAT = auto()    # Float value
    BOOL = auto()     # Boolean value
    INT = auto()      # Integer value
    TRIGGER = auto()  # Trigger (fire-and-forget)
    ENUM = auto()     # Enum value


@dataclass
class InputSlot:
    """Typed input slot definition for an animation node."""

    name: str
    slot_type: SlotType
    description: str = ""
    optional: bool = False


@dataclass
class OutputSlot:
    """Typed output slot definition for an animation node."""

    name: str
    slot_type: SlotType
    description: str = ""


# =============================================================================
# GRAPH NODE METACLASS (Auto-Registration)
# =============================================================================


class GraphNodeMeta(EngineMeta, ABCMeta):
    """Metaclass for automatic registration of animation node types.

    Inherits from EngineMeta for Trinity engine compatibility (global type
    registry, debug introspection) and ABCMeta so AnimationNode can declare
    abstract methods.
    """

    _registry: Dict[str, Type["AnimationNode"]] = {}

    def __new__(mcs, name: str, bases: tuple, namespace: dict) -> "GraphNodeMeta":
        cls = super().__new__(mcs, name, bases, namespace)
        # Don't register the base class
        if name != "AnimationNode" and not namespace.get("_abstract", False):
            mcs._registry[name] = cls
        return cls

    @classmethod
    def registry(mcs) -> Dict[str, Type["AnimationNode"]]:
        """Get all registered node types (name -> class mapping)."""
        return dict(mcs._registry)

    @classmethod
    def get_node_type(mcs, name: str) -> Optional[Type["AnimationNode"]]:
        """Get a registered node type by name."""
        return mcs._registry.get(name)

    @classmethod
    def all_node_types(mcs) -> Dict[str, Type["AnimationNode"]]:
        """Get all registered node types."""
        return dict(mcs._registry)

    @classmethod
    def clear_registry(mcs) -> None:
        """Clear the node registry (for testing)."""
        mcs._registry.clear()


# =============================================================================
# POSE REPRESENTATION
# =============================================================================


@dataclass
class Transform:
    """A 3D transform with position, rotation (quaternion), and scale."""

    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # x, y, z, w
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def lerp(self, other: "Transform", t: float) -> "Transform":
        """Linear interpolation between transforms."""
        return Transform(
            position=tuple(
                a + (b - a) * t
                for a, b in zip(self.position, other.position)
            ),
            rotation=self._slerp(self.rotation, other.rotation, t),
            scale=tuple(
                a + (b - a) * t
                for a, b in zip(self.scale, other.scale)
            ),
        )

    @staticmethod
    def _slerp(
        q1: Tuple[float, float, float, float],
        q2: Tuple[float, float, float, float],
        t: float
    ) -> Tuple[float, float, float, float]:
        """Spherical linear interpolation for quaternions."""
        config = get_config()

        # Compute dot product
        dot = sum(a * b for a, b in zip(q1, q2))

        # If dot is negative, negate one quaternion to take shorter path
        if dot < 0:
            q2 = tuple(-x for x in q2)
            dot = -dot

        # If quaternions are very close, use linear interpolation
        if dot > config.quaternion.SLERP_DOT_THRESHOLD:
            result = tuple(a + (b - a) * t for a, b in zip(q1, q2))
            # Normalize
            length = math.sqrt(sum(x * x for x in result))
            if length > 0:
                return tuple(x / length for x in result)
            return q1

        # Standard slerp
        theta_0 = math.acos(min(1.0, max(-1.0, dot)))
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)

        if sin_theta_0 < config.quaternion.SLERP_MIN_SIN_THETA:
            return q1

        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return tuple(s0 * a + s1 * b for a, b in zip(q1, q2))

    def __add__(self, other: "Transform") -> "Transform":
        """Additive blend (for additive animations)."""
        return Transform(
            position=tuple(a + b for a, b in zip(self.position, other.position)),
            rotation=self._multiply_quaternion(self.rotation, other.rotation),
            scale=tuple(a * b for a, b in zip(self.scale, other.scale)),
        )

    @staticmethod
    def _multiply_quaternion(
        q1: Tuple[float, float, float, float],
        q2: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Multiply two quaternions."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )

    def blend(self, other: "Transform", t: float) -> "Transform":
        """Blend between two transforms using SLERP for rotation.

        Clamps *t* to [0, 1] for numerical stability then delegates
        to the underlying lerp (which applies SLERP for the rotation
        component).
        """
        if t <= 0.0:
            return self.copy()
        if t >= 1.0:
            return other.copy()
        return self.lerp(other, t)

    def compose(self, other: "Transform") -> "Transform":
        """Hierarchical composition: treat *self* as parent, *other* as child.

        Returns a single Transform equivalent to placing the child in the
        parent's local space:

            pos_out = parent.pos + rotate(parent.rot, parent.scale * child.pos)
            rot_out = parent.rot * child.rot
            scale_out = parent.scale * child.scale
        """
        # Scale child position by parent scale
        scaled_cp = (
            other.position[0] * self.scale[0],
            other.position[1] * self.scale[1],
            other.position[2] * self.scale[2],
        )
        # Rotate scaled child position by parent rotation
        rotated_cp = self._rotate_vector(scaled_cp, self.rotation)
        return Transform(
            position=(
                self.position[0] + rotated_cp[0],
                self.position[1] + rotated_cp[1],
                self.position[2] + rotated_cp[2],
            ),
            rotation=self._multiply_quaternion(self.rotation, other.rotation),
            scale=(
                self.scale[0] * other.scale[0],
                self.scale[1] * other.scale[1],
                self.scale[2] * other.scale[2],
            ),
        )

    @staticmethod
    def _rotate_vector(
        v: Tuple[float, float, float],
        q: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float]:
        """Rotate 3-D vector *v* by quaternion *q* (x, y, z, w)."""
        qx, qy, qz, qw = q
        vx, vy, vz = v
        # uv = cross(qv, v)
        uv_x = qy * vz - qz * vy
        uv_y = qz * vx - qx * vz
        uv_z = qx * vy - qy * vx
        # uuv = cross(qv, uv)
        uuv_x = qy * uv_z - qz * uv_y
        uuv_y = qz * uv_x - qx * uv_z
        uuv_z = qx * uv_y - qy * uv_x
        # v + (uv * qw + uuv) * 2
        return (
            vx + (uv_x * qw + uuv_x) * 2.0,
            vy + (uv_y * qw + uuv_y) * 2.0,
            vz + (uv_z * qw + uuv_z) * 2.0,
        )

    @classmethod
    def identity(cls) -> "Transform":
        """Return an identity transform."""
        return cls()

    def copy(self) -> "Transform":
        """Return a copy of this transform."""
        return Transform(
            position=self.position,
            rotation=self.rotation,
            scale=self.scale,
        )


@dataclass
class Pose:
    """A collection of bone transforms representing an animation pose."""

    transforms: List[Transform] = field(default_factory=list)
    root_motion: Optional[Transform] = None
    skeleton: Optional["Skeleton"] = None

    def bone_count(self) -> int:
        """Return the number of bones in this pose."""
        return len(self.transforms)

    def get_transform(self, bone_index: int) -> Transform:
        """Get the transform for a specific bone."""
        if 0 <= bone_index < len(self.transforms):
            return self.transforms[bone_index]
        return Transform.identity()

    def set_transform(self, bone_index: int, transform: Transform) -> None:
        """Set the transform for a specific bone."""
        while len(self.transforms) <= bone_index:
            self.transforms.append(Transform.identity())
        self.transforms[bone_index] = transform

    def lerp(self, other: "Pose", t: float) -> "Pose":
        """Linear interpolation between poses."""
        max_bones = max(len(self.transforms), len(other.transforms))
        result_transforms = []

        for i in range(max_bones):
            t1 = self.get_transform(i)
            t2 = other.get_transform(i)
            result_transforms.append(t1.lerp(t2, t))

        root_motion = None
        if self.root_motion and other.root_motion:
            root_motion = self.root_motion.lerp(other.root_motion, t)
        elif self.root_motion:
            root_motion = self.root_motion.copy()
        elif other.root_motion:
            root_motion = other.root_motion.copy()

        return Pose(transforms=result_transforms, root_motion=root_motion,
                     skeleton=self.skeleton or other.skeleton)

    def blend(self, other: "Pose", t: float) -> "Pose":
        """Blend between two poses, handling missing bones gracefully.

        Bones present in one pose but not the other default to identity.
        Clamps *t* to [0, 1] for numerical stability.
        """
        if t <= 0.0:
            return self.copy()
        if t >= 1.0:
            return other.copy()
        return self.lerp(other, t)

    def apply_mask(self, mask: "BoneMask", weight_multiplier: float = 1.0) -> "Pose":
        """Apply a bone mask to this pose, returning a new pose."""
        return mask.apply_to_pose(self, weight_multiplier)

    def additive_blend(self, additive: "Pose", weight: float = 1.0) -> "Pose":
        """Apply an additive pose on top of this base pose."""
        max_bones = max(len(self.transforms), len(additive.transforms))
        result_transforms = []

        for i in range(max_bones):
            base = self.get_transform(i)
            add = additive.get_transform(i)

            # Scale the additive contribution by weight
            if weight < 1.0:
                add = Transform.identity().lerp(add, weight)

            result_transforms.append(base + add)

        return Pose(transforms=result_transforms, root_motion=self.root_motion,
                     skeleton=self.skeleton)

    @classmethod
    def identity(cls, bone_count: int) -> "Pose":
        """Create an identity pose with the given bone count."""
        return cls(transforms=[Transform.identity() for _ in range(bone_count)])

    def copy(self) -> "Pose":
        """Return a deep copy of this pose."""
        return Pose(
            transforms=[t.copy() for t in self.transforms],
            root_motion=self.root_motion.copy() if self.root_motion else None,
            skeleton=self.skeleton,
        )


# =============================================================================
# SKELETON
# =============================================================================


@dataclass
class Bone:
    """A bone in a skeleton hierarchy (simple index-based).

    This is the **legacy** Bone dataclass used internally by the
    ``animation_graph`` module.  It stores hierarchy via ``parent_index``
    and identifies bones by a flat index.

    For the production Bone/Skeleton pair (name-based, tree-based
    hierarchy with parent/children references, IK chain queries, and
    validation), see ``engine.animation.graph.skeleton``.  Both are
    re-exported from ``__init__.py`` as ``Bone`` / ``Skeleton`` (from
    this module) and ``SkeletonBone`` / ``SkeletonHierarchy`` (from
    ``skeleton.py``).
    """

    name: str
    index: int
    parent_index: int = -1  # -1 means no parent (root)
    bind_pose: Transform = field(default_factory=Transform.identity)

    def is_root(self) -> bool:
        """Return True if this is a root bone."""
        return self.parent_index < 0


@dataclass
class Skeleton:
    """A skeleton hierarchy for animation (simple index-based).

    This is the **legacy** Skeleton dataclass used internally by the
    ``animation_graph`` module.  It stores bones in a flat list with
    a name-to-index map for lookups.

    For the production Skeleton (name-based, tree-based hierarchy with
    parent/children references, IK chain queries, and structural
    validation), see ``engine.animation.graph.skeleton``.  Both are
    re-exported from ``__init__.py`` as ``Skeleton`` (from this module)
    and ``SkeletonHierarchy`` (from ``skeleton.py``).
    """

    bones: List[Bone] = field(default_factory=list)
    _name_to_index: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._rebuild_name_map()

    def _rebuild_name_map(self) -> None:
        """Rebuild the name to index mapping."""
        self._name_to_index = {bone.name: bone.index for bone in self.bones}

    def bone_count(self) -> int:
        """Return the number of bones."""
        return len(self.bones)

    def get_bone(self, index: int) -> Optional[Bone]:
        """Get a bone by index."""
        if 0 <= index < len(self.bones):
            return self.bones[index]
        return None

    def get_bone_by_name(self, name: str) -> Optional[Bone]:
        """Get a bone by name."""
        index = self._name_to_index.get(name)
        if index is not None:
            return self.bones[index]
        return None

    def get_bone_index(self, name: str) -> int:
        """Get the index of a bone by name, or -1 if not found."""
        return self._name_to_index.get(name, -1)

    def add_bone(self, name: str, parent_index: int = -1,
                 bind_pose: Optional[Transform] = None) -> Bone:
        """Add a bone to the skeleton."""
        index = len(self.bones)
        bone = Bone(
            name=name,
            index=index,
            parent_index=parent_index,
            bind_pose=bind_pose or Transform.identity(),
        )
        self.bones.append(bone)
        self._name_to_index[name] = index
        return bone

    def get_bind_pose(self) -> Pose:
        """Get the bind pose for this skeleton."""
        return Pose(transforms=[bone.bind_pose.copy() for bone in self.bones])

    def get_children(self, bone_index: int) -> List[int]:
        """Get the indices of all children of a bone."""
        return [bone.index for bone in self.bones if bone.parent_index == bone_index]

    def get_chain(self, start: int, end: int) -> List[int]:
        """Get the chain of bone indices from start to end (following parents)."""
        if end < 0 or end >= len(self.bones):
            return []

        chain = [end]
        current = end

        while current != start and self.bones[current].parent_index >= 0:
            current = self.bones[current].parent_index
            chain.append(current)

        if current != start:
            return []  # Not connected

        chain.reverse()
        return chain


# =============================================================================
# BONE MASK
# =============================================================================


@dataclass
class BoneMask:
    """A mask defining bone weights for partial animation."""

    name: str
    weights: Dict[int, float] = field(default_factory=dict)

    def get_weight(self, bone_index: int) -> float:
        """Get the weight for a bone (defaults to 0.0)."""
        return self.weights.get(bone_index, 0.0)

    def set_weight(self, bone_index: int, weight: float) -> None:
        """Set the weight for a bone."""
        self.weights[bone_index] = max(0.0, min(1.0, weight))

    def set_weights(self, indices: List[int], weight: float) -> None:
        """Set the same weight for multiple bones."""
        for index in indices:
            self.set_weight(index, weight)

    @classmethod
    def full(cls, skeleton: Skeleton, name: str = "full") -> "BoneMask":
        """Create a mask with all bones at weight 1.0."""
        mask = cls(name=name)
        for bone in skeleton.bones:
            mask.set_weight(bone.index, 1.0)
        return mask

    @classmethod
    def from_bone_names(cls, skeleton: Skeleton, name: str,
                        bone_names: List[str], weight: float = 1.0,
                        include_children: bool = False) -> "BoneMask":
        """Create a mask from bone names."""
        mask = cls(name=name)

        for bone_name in bone_names:
            bone = skeleton.get_bone_by_name(bone_name)
            if bone:
                mask.set_weight(bone.index, weight)
                if include_children:
                    cls._add_children(skeleton, bone.index, mask, weight)

        return mask

    @classmethod
    def _add_children(cls, skeleton: Skeleton, parent_index: int,
                      mask: "BoneMask", weight: float) -> None:
        """Recursively add children bones to the mask."""
        for child_index in skeleton.get_children(parent_index):
            mask.set_weight(child_index, weight)
            cls._add_children(skeleton, child_index, mask, weight)

    def apply_to_pose(self, pose: Pose, weight_multiplier: float = 1.0) -> Pose:
        """Apply this mask to a pose, scaling transforms by bone weights."""
        result = Pose.identity(pose.bone_count())
        identity = Transform.identity()

        for i in range(pose.bone_count()):
            bone_weight = self.get_weight(i) * weight_multiplier
            if bone_weight > 0:
                result.transforms[i] = identity.lerp(pose.transforms[i], bone_weight)

        return result


# =============================================================================
# GRAPH PARAMETER
# =============================================================================


class ParameterType(Enum):
    """Types of graph parameters."""

    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    TRIGGER = auto()  # Single-fire trigger (auto-resets)
    ENUM = auto()


@dataclass
class GraphParameter:
    """A parameter that can drive animation graph behavior."""

    name: str
    param_type: ParameterType
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    enum_values: Optional[List[str]] = None

    _value: Any = field(default=None, repr=False)
    _was_triggered: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self._value is None:
            self._value = self.default_value

    @property
    def value(self) -> Any:
        """Get the current value."""
        if self.param_type == ParameterType.TRIGGER:
            triggered = self._was_triggered
            self._was_triggered = False
            return triggered
        return self._value

    @value.setter
    def value(self, new_value: Any) -> None:
        """Set the current value with validation."""
        if self.param_type == ParameterType.TRIGGER:
            if new_value:
                self._was_triggered = True
            return

        if self.param_type == ParameterType.FLOAT:
            new_value = float(new_value)
            if self.min_value is not None:
                new_value = max(self.min_value, new_value)
            if self.max_value is not None:
                new_value = min(self.max_value, new_value)
        elif self.param_type == ParameterType.INT:
            new_value = int(new_value)
            if self.min_value is not None:
                new_value = max(int(self.min_value), new_value)
            if self.max_value is not None:
                new_value = min(int(self.max_value), new_value)
        elif self.param_type == ParameterType.BOOL:
            new_value = bool(new_value)
        elif self.param_type == ParameterType.ENUM:
            if self.enum_values and new_value not in self.enum_values:
                raise ValueError(f"Invalid enum value: {new_value}")

        self._value = new_value

    def trigger(self) -> None:
        """Trigger a trigger parameter."""
        if self.param_type == ParameterType.TRIGGER:
            self._was_triggered = True

    def reset(self) -> None:
        """Reset to default value."""
        self._value = self.default_value
        self._was_triggered = False

    @classmethod
    def float_param(cls, name: str, default: float = 0.0,
                    min_val: Optional[float] = None,
                    max_val: Optional[float] = None) -> "GraphParameter":
        """Create a float parameter."""
        return cls(
            name=name,
            param_type=ParameterType.FLOAT,
            default_value=default,
            min_value=min_val,
            max_value=max_val,
        )

    @classmethod
    def int_param(cls, name: str, default: int = 0,
                  min_val: Optional[int] = None,
                  max_val: Optional[int] = None) -> "GraphParameter":
        """Create an integer parameter."""
        return cls(
            name=name,
            param_type=ParameterType.INT,
            default_value=default,
            min_value=min_val,
            max_value=max_val,
        )

    @classmethod
    def bool_param(cls, name: str, default: bool = False) -> "GraphParameter":
        """Create a boolean parameter."""
        return cls(
            name=name,
            param_type=ParameterType.BOOL,
            default_value=default,
        )

    @classmethod
    def trigger_param(cls, name: str) -> "GraphParameter":
        """Create a trigger parameter."""
        return cls(
            name=name,
            param_type=ParameterType.TRIGGER,
            default_value=False,
        )

    @classmethod
    def enum_param(cls, name: str, values: List[str],
                   default: Optional[str] = None) -> "GraphParameter":
        """Create an enum parameter."""
        if not values:
            raise ValueError("Enum parameter must have at least one value")
        return cls(
            name=name,
            param_type=ParameterType.ENUM,
            default_value=default or values[0],
            enum_values=values,
        )


# =============================================================================
# GRAPH CONTEXT
# =============================================================================


@dataclass
class GraphContext:
    """Context passed to nodes during evaluation.

    Carries all per-evaluation state: parameters, skeleton reference, delta
    time, accumulated time, frame counter, sync group, bone masks, and
    debugging metadata.

    The context is intended to be **lightweight** -- it is passed by
    reference (dataclass object) and shallow-copied via ``with_depth()``
    when child-node evaluation requires depth tracking.  For hot loops,
    see ``ContextPool``.
    """

    parameters: Dict[str, GraphParameter] = field(default_factory=dict)
    dt: float = 0.0
    skeleton: Optional[Skeleton] = None
    bone_masks: Dict[str, BoneMask] = field(default_factory=dict)

    # Runtime state
    normalized_time: float = 0.0
    sync_group: Optional[str] = None
    layer_weight: float = 1.0

    # Time tracking
    current_time: float = 0.0
    tick: int = 0

    # Metadata for debugging
    current_node_id: Optional[str] = None
    evaluation_depth: int = 0

    # Optional cache of node results for graph-level topological evaluation.
    # Populated by AnimationGraph.evaluate() and checked by
    # AnimationNode.evaluate_input() to avoid redundant evaluation.
    _node_results: Optional[Dict[str, "Pose"]] = field(default=None, repr=False)

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a parameter value by name, returning *default* when absent."""
        param = self.parameters.get(name)
        if param is not None:
            return param.value
        return default

    def get_parameter_float(self, name: str, default: float = 0.0) -> float:
        """Get a float parameter value."""
        value = self.get_parameter(name)
        if value is not None:
            return float(value)
        return default

    def get_parameter_int(self, name: str, default: int = 0) -> int:
        """Get an int parameter value."""
        value = self.get_parameter(name)
        if value is not None:
            return int(value)
        return default

    def get_parameter_bool(self, name: str, default: bool = False) -> bool:
        """Get a bool parameter value."""
        value = self.get_parameter(name)
        if value is not None:
            return bool(value)
        return default

    def get_bone_mask(self, name: str) -> Optional[BoneMask]:
        """Get a bone mask by name."""
        return self.bone_masks.get(name)

    def advance_time(self, dt: float) -> "GraphContext":
        """Advance time by *dt* seconds and increment tick.

        Returns a new ``GraphContext`` with updated ``current_time``,
        ``tick``, and ``dt``.  All other fields are shared by reference.
        """
        return GraphContext(
            parameters=self.parameters,
            dt=dt,
            skeleton=self.skeleton,
            bone_masks=self.bone_masks,
            normalized_time=self.normalized_time,
            sync_group=self.sync_group,
            layer_weight=self.layer_weight,
            current_time=self.current_time + dt,
            tick=self.tick + 1,
            current_node_id=self.current_node_id,
            evaluation_depth=self.evaluation_depth,
            _node_results=self._node_results,
        )

    def with_depth(self) -> "GraphContext":
        """Create a new context with incremented depth."""
        return GraphContext(
            parameters=self.parameters,
            dt=self.dt,
            skeleton=self.skeleton,
            bone_masks=self.bone_masks,
            normalized_time=self.normalized_time,
            sync_group=self.sync_group,
            layer_weight=self.layer_weight,
            current_time=self.current_time,
            tick=self.tick,
            current_node_id=self.current_node_id,
            evaluation_depth=self.evaluation_depth + 1,
            _node_results=self._node_results,
        )


# =============================================================================
# CONTEXT POOL
# =============================================================================


@dataclass
class ContextPool:
    """Reusable pool of ``GraphContext`` objects.

    Reduces allocation pressure in hot evaluation loops by recycling
    contexts.  The pool is **not** thread-safe.

    Usage::

        pool = ContextPool()
        ctx = pool.acquire(dt=1/60, skeleton=skel)
        try:
            result = graph.evaluate(ctx)
        finally:
            pool.release(ctx)
    """

    _available: List[GraphContext] = field(default_factory=list, repr=False)
    _active_count: int = field(default=0, repr=False)

    # -- pool lifecycle --------------------------------------------------------

    def acquire(
        self,
        parameters: Optional[Dict[str, GraphParameter]] = None,
        dt: float = 0.0,
        skeleton: Optional[Skeleton] = None,
        bone_masks: Optional[Dict[str, BoneMask]] = None,
        normalized_time: float = 0.0,
        sync_group: Optional[str] = None,
        layer_weight: float = 1.0,
        current_time: float = 0.0,
        tick: int = 0,
    ) -> GraphContext:
        """Acquire a ``GraphContext`` from the pool (or create one).

        The returned context is reset to the supplied field values so
        callers do not see stale state from previous use.
        """
        if self._available:
            ctx = self._available.pop()
            ctx.parameters = parameters or {}
            ctx.dt = dt
            ctx.skeleton = skeleton
            ctx.bone_masks = bone_masks or {}
            ctx.normalized_time = normalized_time
            ctx.sync_group = sync_group
            ctx.layer_weight = layer_weight
            ctx.current_time = current_time
            ctx.tick = tick
            ctx.current_node_id = None
            ctx.evaluation_depth = 0
            ctx._node_results = None
        else:
            ctx = GraphContext(
                parameters=parameters or {},
                dt=dt,
                skeleton=skeleton,
                bone_masks=bone_masks or {},
                normalized_time=normalized_time,
                sync_group=sync_group,
                layer_weight=layer_weight,
                current_time=current_time,
                tick=tick,
            )
        self._active_count += 1
        return ctx

    def release(self, ctx: GraphContext) -> None:
        """Return a context to the pool for reuse."""
        self._available.append(ctx)
        self._active_count -= 1

    # -- diagnostics -----------------------------------------------------------

    @property
    def available_count(self) -> int:
        """Number of contexts currently sitting in the pool."""
        return len(self._available)

    @property
    def active_count(self) -> int:
        """Number of contexts currently acquired (not yet released)."""
        return self._active_count

    @property
    def total_created(self) -> int:
        """Total number of contexts created (pool size + active)."""
        return len(self._available) + self._active_count


# =============================================================================
# ANIMATION NODE (Base Class)
# =============================================================================


class AnimationNode(metaclass=GraphNodeMeta):
    """Base class for all animation graph nodes.

    Every animation node has:
    - A unique node_id and optional display_name
    - Typed input and output slot definitions
    - A signature evaluate(context) method that returns a Pose
    """

    _abstract = True

    def __init__(self, node_id: str, display_name: Optional[str] = None) -> None:
        self.node_id = node_id
        self._display_name: str = display_name or node_id
        self._input_slots: Dict[str, InputSlot] = {}
        self._output_slots: Dict[str, OutputSlot] = {}
        self.inputs: Dict[str, Optional["AnimationNode"]] = {}
        self.outputs: Dict[str, Any] = {}
        self._cached_pose: Optional[Pose] = None
        self._cache_valid: bool = False

    # -- Node identification ---------------------------------------------------

    @property
    def display_name(self) -> str:
        """Human-readable display name for this node."""
        return self._display_name

    @display_name.setter
    def display_name(self, value: str) -> None:
        self._display_name = value

    def node_type_name(self) -> str:
        """Returns the registered type name for this node class."""
        return self.__class__.__name__

    # -- Slot system -----------------------------------------------------------

    def define_input_slot(
        self,
        name: str,
        slot_type: SlotType,
        description: str = "",
        optional: bool = False,
    ) -> InputSlot:
        """Define a typed input slot on this node.

        Args:
            name: Slot identifier (used in connect/disconnect).
            slot_type: The type of data this slot accepts.
            description: Human-readable description of the slot.
            optional: Whether this slot may be left disconnected.

        Returns:
            The newly created InputSlot.
        """
        slot = InputSlot(
            name=name,
            slot_type=slot_type,
            description=description,
            optional=optional,
        )
        self._input_slots[name] = slot
        return slot

    def define_output_slot(
        self,
        name: str,
        slot_type: SlotType,
        description: str = "",
    ) -> OutputSlot:
        """Define a typed output slot on this node.

        Args:
            name: Slot identifier.
            slot_type: The type of data this slot produces.
            description: Human-readable description of the slot.

        Returns:
            The newly created OutputSlot.
        """
        slot = OutputSlot(
            name=name,
            slot_type=slot_type,
            description=description,
        )
        self._output_slots[name] = slot
        return slot

    def get_input_slot(self, name: str) -> Optional[InputSlot]:
        """Get an input slot definition by name."""
        return self._input_slots.get(name)

    def get_output_slot(self, name: str) -> Optional[OutputSlot]:
        """Get an output slot definition by name."""
        return self._output_slots.get(name)

    @property
    def input_slots(self) -> Dict[str, InputSlot]:
        """All defined input slots on this node."""
        return dict(self._input_slots)

    @property
    def output_slots(self) -> Dict[str, OutputSlot]:
        """All defined output slots on this node."""
        return dict(self._output_slots)

    @abstractmethod
    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate this node and return a pose.

        Subclasses must override this to implement their specific animation
        logic. The context provides delta time, skeleton reference, parameters,
        and other evaluation state.
        """
        pass

    def invalidate_cache(self) -> None:
        """Invalidate the cached pose."""
        self._cache_valid = False
        self._cached_pose = None

    def get_input(self, name: str) -> Optional["AnimationNode"]:
        """Get an input node by name."""
        return self.inputs.get(name)

    def set_input(self, name: str, node: Optional["AnimationNode"]) -> None:
        """Set an input node."""
        self.inputs[name] = node

    def evaluate_input(self, name: str, context: GraphContext) -> Optional[Pose]:
        """Evaluate an input node and return its pose.

        When the graph is using topological evaluation, this method checks
        the context's node-results cache first to avoid redundant recursive
        evaluation.  Falls back to standard depth-incremented recursive
        evaluation when no cache is available (e.g. standalone node use).
        """
        node = self.inputs.get(name)
        if node is None:
            return None

        # Topological-evaluation cache: return pre-computed result when
        # the graph evaluates nodes in dependency order.
        if context._node_results is not None:
            cached = context._node_results.get(node.node_id)
            if cached is not None:
                return cached

        return node.evaluate(context.with_depth())

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about this node."""
        return {
            "node_id": self.node_id,
            "display_name": self._display_name,
            "type": self.__class__.__name__,
            "input_slots": {k: v.slot_type.name for k, v in self._input_slots.items()},
            "output_slots": {k: v.slot_type.name for k, v in self._output_slots.items()},
            "inputs": list(self.inputs.keys()),
            "outputs": list(self.outputs.keys()),
        }


# =============================================================================
# CONNECTION
# =============================================================================


@dataclass
class Connection:
    """A connection between two nodes in the graph."""

    source_node_id: str
    source_output: str
    target_node_id: str
    target_input: str

    def __hash__(self) -> int:
        return hash((self.source_node_id, self.source_output,
                     self.target_node_id, self.target_input))


# =============================================================================
# ANIMATION GRAPH
# =============================================================================


class AnimationGraph:
    """
    The main animation graph container.

    An animation graph is a DAG of nodes that produces a final output pose.
    It contains:
    - Nodes: Animation processing nodes (state machines, blend trees, etc.)
    - Connections: Links between node inputs and outputs
    - Parameters: Values that drive animation behavior
    - Subgraphs: Reusable nested graphs
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self.nodes: Dict[str, AnimationNode] = {}
        self.connections: Set[Connection] = set()
        self.parameters: Dict[str, GraphParameter] = {}
        self.output_node_id: Optional[str] = None
        self.subgraphs: Dict[str, "AnimationGraph"] = {}

        # Evaluation state
        self._output_pose: Optional[Pose] = None
        self._dirty: bool = True

    def add_node(self, node: AnimationNode) -> None:
        """Add a node to the graph."""
        if node.node_id in self.nodes:
            raise ValueError(f"Node '{node.node_id}' already exists in graph")
        self.nodes[node.node_id] = node
        self._dirty = True

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its connections."""
        if node_id not in self.nodes:
            return False

        # Remove connections involving this node
        self.connections = {
            c for c in self.connections
            if c.source_node_id != node_id and c.target_node_id != node_id
        }

        # Remove from other nodes' inputs
        for node in self.nodes.values():
            for input_name, input_node in list(node.inputs.items()):
                if input_node and input_node.node_id == node_id:
                    node.inputs[input_name] = None

        del self.nodes[node_id]

        if self.output_node_id == node_id:
            self.output_node_id = None

        self._dirty = True
        return True

    def get_node(self, node_id: str) -> Optional[AnimationNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def connect(self, source_node_id: str, source_output: str,
                target_node_id: str, target_input: str) -> bool:
        """Connect two nodes."""
        source_node = self.nodes.get(source_node_id)
        target_node = self.nodes.get(target_node_id)

        if not source_node or not target_node:
            return False

        # Validate slot type compatibility when both slots are defined
        source_slot = source_node.get_output_slot(source_output)
        target_slot = target_node.get_input_slot(target_input)
        if source_slot is not None and target_slot is not None:
            if source_slot.slot_type != target_slot.slot_type:
                raise TypeError(
                    f"Slot type mismatch: source output '{source_output}' "
                    f"has type {source_slot.slot_type.name}, "
                    f"target input '{target_input}' "
                    f"has type {target_slot.slot_type.name}"
                )

        connection = Connection(
            source_node_id=source_node_id,
            source_output=source_output,
            target_node_id=target_node_id,
            target_input=target_input,
        )

        self.connections.add(connection)
        target_node.set_input(target_input, source_node)
        self._dirty = True
        return True

    def disconnect(self, source_node_id: str, source_output: str,
                   target_node_id: str, target_input: str) -> bool:
        """Disconnect two nodes."""
        connection = Connection(
            source_node_id=source_node_id,
            source_output=source_output,
            target_node_id=target_node_id,
            target_input=target_input,
        )

        if connection in self.connections:
            self.connections.remove(connection)
            target_node = self.nodes.get(target_node_id)
            if target_node:
                target_node.set_input(target_input, None)
            self._dirty = True
            return True
        return False

    def add_parameter(self, param: GraphParameter) -> None:
        """Add a parameter to the graph."""
        self.parameters[param.name] = param

    def set_parameter(self, name: str, value: Any) -> bool:
        """Set a parameter value."""
        param = self.parameters.get(name)
        if param:
            param.value = value
            self._dirty = True
            return True
        return False

    def get_parameter(self, name: str) -> Optional[Any]:
        """Get a parameter value."""
        param = self.parameters.get(name)
        if param:
            return param.value
        return None

    def trigger_parameter(self, name: str) -> bool:
        """Trigger a trigger parameter."""
        param = self.parameters.get(name)
        if param and param.param_type == ParameterType.TRIGGER:
            param.trigger()
            self._dirty = True
            return True
        return False

    def set_output_node(self, node_id: str) -> bool:
        """Set the output node of the graph."""
        if node_id in self.nodes:
            self.output_node_id = node_id
            return True
        return False

    def add_subgraph(self, name: str, subgraph: "AnimationGraph") -> None:
        """Add a subgraph for reuse."""
        self.subgraphs[name] = subgraph

    def get_subgraph(self, name: str) -> Optional["AnimationGraph"]:
        """Get a subgraph by name."""
        return self.subgraphs.get(name)

    def evaluate(self, context: Optional[GraphContext] = None) -> Pose:
        """Evaluate the graph and return the output pose.

        Uses topological traversal to evaluate each node exactly once,
        caching intermediate results so nodes with multiple consumers
        are not re-evaluated.  Cycle detection is performed first when
        the config enables it.
        """
        if not self.output_node_id:
            return Pose()

        output_node = self.nodes.get(self.output_node_id)
        if not output_node:
            return Pose()

        # -- Cycle detection ---------------------------------------------------
        cfg = get_config()
        if cfg.graph.CYCLE_DETECTION_ENABLED:
            if self._has_cycle():
                return Pose()

        # -- Build evaluation context ------------------------------------------
        if context is None:
            context = GraphContext()

        merged_params = dict(self.parameters)
        merged_params.update(context.parameters)

        eval_context = GraphContext(
            parameters=merged_params,
            dt=context.dt,
            skeleton=context.skeleton,
            bone_masks=context.bone_masks,
            normalized_time=context.normalized_time,
            sync_group=context.sync_group,
            layer_weight=context.layer_weight,
            current_time=context.current_time,
            tick=context.tick,
        )

        # -- Topological traversal ---------------------------------------------
        topo_order = self.get_topology_order()

        # Prime the context with a results cache so that evaluate_input
        # checks it before recursing.
        node_results: Dict[str, Pose] = {}
        eval_context._node_results = node_results

        for node_id in topo_order:
            node = self.nodes.get(node_id)
            if node is not None and node_id not in node_results:
                result = node.evaluate(eval_context)
                node_results[node_id] = result

        self._output_pose = node_results.get(self.output_node_id, Pose())
        self._dirty = False
        return self._output_pose

    @property
    def output_pose(self) -> Optional[Pose]:
        """Get the last evaluated output pose."""
        return self._output_pose

    def invalidate(self) -> None:
        """Invalidate all cached poses."""
        self._dirty = True
        for node in self.nodes.values():
            node.invalidate_cache()

    def get_topology_order(self) -> List[str]:
        """Get nodes in topological order for evaluation."""
        visited: Set[str] = set()
        order: List[str] = []

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)

            node = self.nodes.get(node_id)
            if node:
                for input_node in node.inputs.values():
                    if input_node:
                        visit(input_node.node_id)
                order.append(node_id)

        if self.output_node_id:
            visit(self.output_node_id)

        return order

    def validate(self) -> List[str]:
        """Validate the graph and return a list of errors."""
        errors = []

        # Check for cycles (report each cycle individually)
        for cycle_desc in detect_cycles(self):
            errors.append(cycle_desc)

        # Check output node
        if not self.output_node_id:
            errors.append("No output node set")
        elif self.output_node_id not in self.nodes:
            errors.append(f"Output node '{self.output_node_id}' not found")

        # Check connections
        for conn in self.connections:
            if conn.source_node_id not in self.nodes:
                errors.append(f"Connection source '{conn.source_node_id}' not found")
            if conn.target_node_id not in self.nodes:
                errors.append(f"Connection target '{conn.target_node_id}' not found")

        return errors

    def _has_cycle(self) -> bool:
        """Check if the graph has a cycle.

        Delegates to the public :func:`detect_cycles` for the actual
        three-color DFS; this method only checks whether any cycle exists.
        """
        return len(detect_cycles(self)) > 0

    def copy(self) -> "AnimationGraph":
        """Create a deep copy of this graph."""
        new_graph = AnimationGraph(name=f"{self.name}_copy")

        # Copy parameters
        for name, param in self.parameters.items():
            new_param = GraphParameter(
                name=param.name,
                param_type=param.param_type,
                default_value=param.default_value,
                min_value=param.min_value,
                max_value=param.max_value,
                enum_values=param.enum_values,
            )
            new_graph.add_parameter(new_param)

        # Note: Nodes need to be copied by the caller since they may have complex state

        return new_graph


# =============================================================================
# CYCLE DETECTION
# =============================================================================


def detect_cycles(graph: AnimationGraph) -> List[str]:
    """Detect all cycles in an animation graph using three-color DFS.

    Uses the WHITE (unvisited) / GRAY (in-progress) / BLACK (finished)
    coloring scheme.  When a back edge is found (neighbour is GRAY), the
    nodes on the current DFS path from the neighbour back to itself form a
    cycle.

    All cycles in the graph are reported, not just the first one found.
    Returns an empty list for acyclic graphs.

    Args:
        graph: The :class:`AnimationGraph` to inspect.

    Returns:
        A list of human-readable cycle descriptions, one per cycle.
        Each description includes the node names forming the cycle::

            Cycle detected: blend_tree -> state_machine -> clip_node -> blend_tree

    Notes:
        The traversal follows *input* edges (node A stores a reference to
        node B when B feeds into A).  The reported node order reflects the
        traversal order; for cycle detection purposes the direction serves
        to unambiguously list the participating nodes.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {}
    path: List[str] = []    # Current DFS recursion stack
    cycles: List[str] = []

    for node_id in graph.nodes:
        color[node_id] = WHITE

    def dfs(node_id: str) -> None:
        color[node_id] = GRAY
        path.append(node_id)

        node = graph.nodes.get(node_id)
        if node:
            for input_node in node.inputs.values():
                if input_node:
                    neighbour_id = input_node.node_id

                    if color[neighbour_id] == GRAY:
                        # Back edge: neighbour is already on the current
                        # DFS stack, which means we have a cycle.
                        idx = path.index(neighbour_id)
                        cycle_nodes = path[idx:] + [neighbour_id]
                        cycles.append(
                            f"Cycle detected: {' -> '.join(cycle_nodes)}"
                        )

                    elif color[neighbour_id] == WHITE:
                        dfs(neighbour_id)

        path.pop()
        color[node_id] = BLACK

    for node_id in graph.nodes:
        if color[node_id] == WHITE:
            dfs(node_id)

    return cycles


# =============================================================================
# SUBGRAPH NODE
# =============================================================================


class SubgraphNode(AnimationNode):
    """A node that evaluates a subgraph."""

    _abstract = False

    def __init__(self, node_id: str, subgraph: AnimationGraph) -> None:
        super().__init__(node_id)
        self.subgraph = subgraph

        # Map inputs to subgraph parameters
        self.parameter_mapping: Dict[str, str] = {}

    def map_parameter(self, input_name: str, subgraph_param: str) -> None:
        """Map an input to a subgraph parameter."""
        self.parameter_mapping[input_name] = subgraph_param

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate the subgraph."""
        # Create subgraph context
        subgraph_context = GraphContext(
            parameters=dict(self.subgraph.parameters),
            dt=context.dt,
            skeleton=context.skeleton,
            bone_masks=context.bone_masks,
            normalized_time=context.normalized_time,
            sync_group=context.sync_group,
            layer_weight=context.layer_weight,
            current_time=context.current_time,
            tick=context.tick,
        )

        for input_name, param_name in self.parameter_mapping.items():
            value = context.get_parameter(input_name)
            if value is not None and param_name in subgraph_context.parameters:
                subgraph_context.parameters[param_name].value = value

        return self.subgraph.evaluate(subgraph_context)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Metaclass
    "GraphNodeMeta",
    # Slot types
    "SlotType",
    "InputSlot",
    "OutputSlot",
    # Transform and Pose
    "Transform",
    "Pose",
    # Skeleton
    "Bone",
    "Skeleton",
    "BoneMask",
    # Parameters
    "ParameterType",
    "GraphParameter",
    # Context
    "GraphContext",
    "ContextPool",
    # Nodes
    "AnimationNode",
    "SubgraphNode",
    # Graph
    "Connection",
    "AnimationGraph",
    # Cycle detection
    "detect_cycles",
]
