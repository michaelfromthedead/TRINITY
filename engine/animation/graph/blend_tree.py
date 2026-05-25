"""
Blend tree implementations for animation.

Provides blend tree functionality for smooth animation blending:
- BlendTree: Base class for blend trees
- BlendTree1D: 1D parameter-driven blending
- BlendTree2D: 2D parameter-driven blending (Cartesian and polar)
- BlendTreeDirect: Direct weight-based blending

Blend trees allow for parametric control of animation blending,
enabling smooth locomotion and directional movement.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from .animation_graph import (
    AnimationNode,
    GraphContext,
    Pose,
    Transform,
)
from .config import get_config


# =============================================================================
# BLEND TREE BASE
# =============================================================================


class BlendTree(AnimationNode, ABC):
    """
    Base class for blend trees.

    Blend trees produce a pose by blending multiple child animations
    based on parameters. Subclasses implement specific blending strategies.
    """

    _abstract = True

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.children: List[AnimationNode] = []

    def add_child(self, node: AnimationNode) -> int:
        """Add a child node and return its index."""
        self.children.append(node)
        return len(self.children) - 1

    def remove_child(self, index: int) -> bool:
        """Remove a child node by index."""
        if 0 <= index < len(self.children):
            self.children.pop(index)
            return True
        return False

    def get_child(self, index: int) -> Optional[AnimationNode]:
        """Get a child node by index."""
        if 0 <= index < len(self.children):
            return self.children[index]
        return None

    def child_count(self) -> int:
        """Get the number of children."""
        return len(self.children)


# =============================================================================
# 1D BLEND TREE
# =============================================================================


@dataclass
class BlendTree1DEntry:
    """An entry in a 1D blend tree."""

    threshold: float
    node: AnimationNode
    speed_multiplier: float = 1.0


class BlendTree1D(BlendTree):
    """
    A 1D blend tree that blends based on a single parameter.

    Children are arranged along a 1D axis with thresholds. The tree
    blends between adjacent entries based on the parameter value.

    Example: Walk/Run blending based on speed parameter
        threshold=0.0 -> Idle
        threshold=2.0 -> Walk
        threshold=5.0 -> Run
    """

    _abstract = False

    def __init__(self, node_id: str, parameter: str) -> None:
        super().__init__(node_id)
        self.parameter = parameter
        self.entries: List[BlendTree1DEntry] = []
        self.use_gradient_bands: bool = True
        config = get_config()
        self.gradient_band_width: float = config.blend_tree.DEFAULT_GRADIENT_BAND_WIDTH

    def add_entry(self, threshold: float, node: AnimationNode,
                  speed_multiplier: float = 1.0) -> int:
        """Add an entry at a threshold value."""
        entry = BlendTree1DEntry(
            threshold=threshold,
            node=node,
            speed_multiplier=speed_multiplier,
        )
        self.entries.append(entry)
        self.entries.sort(key=lambda e: e.threshold)

        # Also track as child
        if node not in self.children:
            self.children.append(node)

        return self.entries.index(entry)

    def remove_entry(self, index: int) -> bool:
        """Remove an entry by index."""
        if 0 <= index < len(self.entries):
            entry = self.entries.pop(index)
            if entry.node in self.children:
                self.children.remove(entry.node)
            return True
        return False

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate the blend tree based on the parameter."""
        if not self.entries:
            return Pose()

        # Get parameter value
        value = context.get_parameter_float(self.parameter, 0.0)

        # Handle edge cases
        if len(self.entries) == 1:
            return self.entries[0].node.evaluate(context)

        # Find bracketing entries
        lower_idx = 0
        upper_idx = 0

        for i, entry in enumerate(self.entries):
            if entry.threshold <= value:
                lower_idx = i
            if entry.threshold >= value:
                upper_idx = i
                break
        else:
            upper_idx = len(self.entries) - 1

        # Clamp to range
        if value <= self.entries[0].threshold:
            return self.entries[0].node.evaluate(context)
        if value >= self.entries[-1].threshold:
            return self.entries[-1].node.evaluate(context)

        # Calculate blend weight
        lower_entry = self.entries[lower_idx]
        upper_entry = self.entries[upper_idx]

        if lower_idx == upper_idx:
            return lower_entry.node.evaluate(context)

        range_size = upper_entry.threshold - lower_entry.threshold
        if range_size <= 0:
            return lower_entry.node.evaluate(context)

        t = (value - lower_entry.threshold) / range_size

        # Apply gradient band interpolation for smoother transitions
        if self.use_gradient_bands and len(self.entries) > 2:
            t = self._apply_gradient_band(t)

        # Evaluate and blend
        lower_pose = lower_entry.node.evaluate(context)
        upper_pose = upper_entry.node.evaluate(context)

        return lower_pose.lerp(upper_pose, t)

    def _apply_gradient_band(self, t: float) -> float:
        """Apply gradient band smoothing for smoother transitions."""
        # Smooth step for gradient bands
        band = self.gradient_band_width
        if t < band:
            # Ease into blend
            return t * t / (2 * band)
        elif t > 1.0 - band:
            # Ease out of blend
            adjusted = t - (1.0 - band)
            return 0.5 + t - band - adjusted * adjusted / (2 * band)
        else:
            # Linear in the middle
            return t
        return t

    def get_weights(self, context: GraphContext) -> Dict[int, float]:
        """Get the blend weights for each entry."""
        if not self.entries:
            return {}

        value = context.get_parameter_float(self.parameter, 0.0)
        weights = {i: 0.0 for i in range(len(self.entries))}

        if len(self.entries) == 1:
            weights[0] = 1.0
            return weights

        # Find and calculate weights
        for i in range(len(self.entries) - 1):
            lower = self.entries[i].threshold
            upper = self.entries[i + 1].threshold

            if lower <= value <= upper:
                range_size = upper - lower
                if range_size > 0:
                    t = (value - lower) / range_size
                    weights[i] = 1.0 - t
                    weights[i + 1] = t
                else:
                    weights[i] = 1.0
                return weights

        # Handle edge cases
        if value <= self.entries[0].threshold:
            weights[0] = 1.0
        else:
            weights[len(self.entries) - 1] = 1.0

        return weights


# =============================================================================
# 2D BLEND TREE
# =============================================================================


class BlendTree2DMode(Enum):
    """Mode for 2D blend tree interpolation."""

    CARTESIAN = auto()  # X-Y grid interpolation
    POLAR = auto()  # Angle-magnitude interpolation
    FREEFORM_DIRECTIONAL = auto()  # Gradient-based
    FREEFORM_CARTESIAN = auto()  # Delaunay triangulation


@dataclass
class BlendTree2DSample:
    """A sample point in a 2D blend tree."""

    position: Tuple[float, float]
    node: AnimationNode
    speed_multiplier: float = 1.0


@dataclass
class Triangle:
    """A triangle for Delaunay triangulation."""

    indices: Tuple[int, int, int]
    vertices: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside this triangle."""
        p = point
        v0, v1, v2 = self.vertices

        # Barycentric coordinate method
        def sign(p1: Tuple[float, float], p2: Tuple[float, float],
                 p3: Tuple[float, float]) -> float:
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(p, v0, v1)
        d2 = sign(p, v1, v2)
        d3 = sign(p, v2, v0)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def get_barycentric(self, point: Tuple[float, float]) -> Tuple[float, float, float]:
        """Get barycentric coordinates for a point in this triangle."""
        v0, v1, v2 = self.vertices
        px, py = point

        # Compute vectors
        v0x, v0y = v2[0] - v0[0], v2[1] - v0[1]
        v1x, v1y = v1[0] - v0[0], v1[1] - v0[1]
        v2x, v2y = px - v0[0], py - v0[1]

        # Compute dot products
        dot00 = v0x * v0x + v0y * v0y
        dot01 = v0x * v1x + v0y * v1y
        dot02 = v0x * v2x + v0y * v2y
        dot11 = v1x * v1x + v1y * v1y
        dot12 = v1x * v2x + v1y * v2y

        # Compute barycentric coordinates
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-10:
            return (1.0, 0.0, 0.0)

        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Clamp to valid range
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        w = 1.0 - u - v

        return (max(0.0, w), max(0.0, v), max(0.0, u))


class BlendTree2D(BlendTree):
    """
    A 2D blend tree that blends based on two parameters.

    Samples are arranged in a 2D space. The tree blends between nearby
    samples based on the parameter values. Supports multiple interpolation
    modes including Cartesian, polar, and Delaunay triangulation.

    Example: Directional movement based on speed and direction
    """

    _abstract = False

    def __init__(self, node_id: str, param_x: str, param_y: str,
                 mode: BlendTree2DMode = BlendTree2DMode.CARTESIAN) -> None:
        super().__init__(node_id)
        self.param_x = param_x
        self.param_y = param_y
        self.mode = mode
        self.samples: List[BlendTree2DSample] = []
        self._triangles: List[Triangle] = []
        self._needs_triangulation: bool = True

    def add_sample(self, x: float, y: float, node: AnimationNode,
                   speed_multiplier: float = 1.0) -> int:
        """Add a sample at the given position."""
        sample = BlendTree2DSample(
            position=(x, y),
            node=node,
            speed_multiplier=speed_multiplier,
        )
        self.samples.append(sample)
        self._needs_triangulation = True

        if node not in self.children:
            self.children.append(node)

        return len(self.samples) - 1

    def remove_sample(self, index: int) -> bool:
        """Remove a sample by index."""
        if 0 <= index < len(self.samples):
            sample = self.samples.pop(index)
            if sample.node in self.children:
                self.children.remove(sample.node)
            self._needs_triangulation = True
            return True
        return False

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate the blend tree based on the parameters."""
        if not self.samples:
            return Pose()

        if len(self.samples) == 1:
            return self.samples[0].node.evaluate(context)

        # Get parameter values
        x = context.get_parameter_float(self.param_x, 0.0)
        y = context.get_parameter_float(self.param_y, 0.0)
        point = (x, y)

        if self.mode == BlendTree2DMode.POLAR:
            return self._evaluate_polar(point, context)
        elif self.mode in (BlendTree2DMode.FREEFORM_CARTESIAN,
                           BlendTree2DMode.FREEFORM_DIRECTIONAL):
            return self._evaluate_triangulated(point, context)
        else:  # CARTESIAN
            return self._evaluate_cartesian(point, context)

    def _evaluate_cartesian(self, point: Tuple[float, float],
                            context: GraphContext) -> Pose:
        """Evaluate using Cartesian (nearest neighbor + bilinear) interpolation."""
        # Find the 4 nearest samples that form a quad
        weights = self._compute_inverse_distance_weights(point)

        if not weights:
            return Pose()

        # Blend poses
        result_pose = None
        total_weight = sum(weights.values())

        if total_weight <= 0:
            return self.samples[0].node.evaluate(context)

        for idx, weight in weights.items():
            if weight <= 0:
                continue

            normalized_weight = weight / total_weight
            pose = self.samples[idx].node.evaluate(context)

            if result_pose is None:
                # Scale first pose by its weight
                result_pose = Pose.identity(pose.bone_count())
                for i in range(pose.bone_count()):
                    result_pose.transforms[i] = Transform.identity().lerp(
                        pose.transforms[i], normalized_weight
                    )
            else:
                # Additive blend
                for i in range(min(result_pose.bone_count(), pose.bone_count())):
                    blended = Transform.identity().lerp(pose.transforms[i], normalized_weight)
                    result_pose.transforms[i] = result_pose.transforms[i] + Transform(
                        position=tuple(
                            b - a for a, b in zip(
                                Transform.identity().position, blended.position
                            )
                        ),
                        rotation=blended.rotation,  # Simplified for additive
                        scale=blended.scale,
                    )

        return result_pose or Pose()

    def _evaluate_polar(self, point: Tuple[float, float],
                        context: GraphContext) -> Pose:
        """Evaluate using polar (angle + magnitude) interpolation."""
        # Compute polar coordinates
        x, y = point
        magnitude = math.sqrt(x * x + y * y)
        angle = math.atan2(y, x)

        # Find samples by angle
        angle_samples = []
        for i, sample in enumerate(self.samples):
            sx, sy = sample.position
            sample_mag = math.sqrt(sx * sx + sy * sy)
            sample_angle = math.atan2(sy, sx)
            angle_samples.append((i, sample_angle, sample_mag))

        # Sort by angle
        angle_samples.sort(key=lambda x: x[1])

        if len(angle_samples) < 2:
            return self.samples[0].node.evaluate(context)

        # Find bracketing angles
        lower_idx = 0
        upper_idx = 0

        for i, (idx, sample_angle, _) in enumerate(angle_samples):
            if sample_angle <= angle:
                lower_idx = i
            if sample_angle >= angle:
                upper_idx = i
                break
        else:
            upper_idx = len(angle_samples) - 1

        # Handle wrap-around
        if lower_idx == upper_idx:
            return self.samples[angle_samples[lower_idx][0]].node.evaluate(context)

        # Blend based on angle
        lower_sample = angle_samples[lower_idx]
        upper_sample = angle_samples[upper_idx]

        angle_range = upper_sample[1] - lower_sample[1]
        if angle_range <= 0:
            angle_range = 2 * math.pi + angle_range

        t = (angle - lower_sample[1]) / angle_range if angle_range > 0 else 0.0
        t = max(0.0, min(1.0, t))

        lower_pose = self.samples[lower_sample[0]].node.evaluate(context)
        upper_pose = self.samples[upper_sample[0]].node.evaluate(context)

        return lower_pose.lerp(upper_pose, t)

    def _evaluate_triangulated(self, point: Tuple[float, float],
                               context: GraphContext) -> Pose:
        """Evaluate using Delaunay triangulation."""
        if self._needs_triangulation:
            self._triangulate()

        # Find containing triangle
        for triangle in self._triangles:
            if triangle.contains_point(point):
                # Get barycentric weights
                w0, w1, w2 = triangle.get_barycentric(point)
                i0, i1, i2 = triangle.indices

                # Evaluate and blend
                pose0 = self.samples[i0].node.evaluate(context)
                pose1 = self.samples[i1].node.evaluate(context)
                pose2 = self.samples[i2].node.evaluate(context)

                # Blend using barycentric weights
                # Normalize weights to ensure they sum to 1.0
                total_weight = w0 + w1 + w2
                if total_weight > 0:
                    w0, w1, w2 = w0 / total_weight, w1 / total_weight, w2 / total_weight
                else:
                    w0, w1, w2 = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
                temp = pose0.lerp(pose1, w1 / (w0 + w1) if (w0 + w1) > 0 else 0.5)
                return temp.lerp(pose2, w2)

        # Fallback: use inverse distance weighting
        return self._evaluate_cartesian(point, context)

    def _compute_inverse_distance_weights(
        self, point: Tuple[float, float], power: Optional[float] = None
    ) -> Dict[int, float]:
        """Compute inverse distance weights for all samples."""
        if power is None:
            config = get_config()
            power = config.blend_tree.INVERSE_DISTANCE_POWER

        weights: Dict[int, float] = {}
        px, py = point

        config = get_config()
        for i, sample in enumerate(self.samples):
            sx, sy = sample.position
            dist = math.sqrt((px - sx) ** 2 + (py - sy) ** 2)

            if dist < config.blend_tree.DISTANCE_EPSILON:
                # Very close to this sample
                return {i: 1.0}

            weights[i] = 1.0 / (dist ** power)

        return weights

    def _triangulate(self) -> None:
        """Perform Delaunay triangulation on samples."""
        self._triangles = []
        self._needs_triangulation = False

        if len(self.samples) < 3:
            return

        # Simple Bowyer-Watson algorithm for Delaunay triangulation
        points = [sample.position for sample in self.samples]

        # Create super-triangle
        min_x = min(p[0] for p in points) - 1
        max_x = max(p[0] for p in points) + 1
        min_y = min(p[1] for p in points) - 1
        max_y = max(p[1] for p in points) + 1

        # Super-triangle vertices (large enough to contain all points)
        margin = max(max_x - min_x, max_y - min_y) * 10
        super_triangle = [
            (min_x - margin, min_y - margin),
            (max_x + margin, min_y - margin),
            ((min_x + max_x) / 2, max_y + margin),
        ]

        # Add super-triangle indices as negative (to be removed later)
        triangles = [(-3, -2, -1)]

        # Extend points with super-triangle
        all_points = points + super_triangle

        # Add points one by one
        for i, point in enumerate(points):
            bad_triangles = []

            # Find triangles whose circumcircle contains the point
            for tri in triangles:
                if self._point_in_circumcircle(point, tri, all_points):
                    bad_triangles.append(tri)

            # Find boundary polygon
            polygon = []
            for tri in bad_triangles:
                for j in range(3):
                    edge = (tri[j], tri[(j + 1) % 3])
                    # Check if edge is shared
                    shared = False
                    for other in bad_triangles:
                        if other == tri:
                            continue
                        if (edge[1], edge[0]) in [(other[k], other[(k + 1) % 3])
                                                   for k in range(3)]:
                            shared = True
                            break
                    if not shared:
                        polygon.append(edge)

            # Remove bad triangles
            for tri in bad_triangles:
                triangles.remove(tri)

            # Create new triangles
            for edge in polygon:
                triangles.append((edge[0], edge[1], i))

        # Remove triangles connected to super-triangle
        triangles = [tri for tri in triangles
                     if all(idx >= 0 for idx in tri)]

        # Convert to Triangle objects
        for tri in triangles:
            self._triangles.append(Triangle(
                indices=tri,
                vertices=(points[tri[0]], points[tri[1]], points[tri[2]]),
            ))

    def _point_in_circumcircle(
        self, point: Tuple[float, float],
        triangle: Tuple[int, int, int],
        all_points: List[Tuple[float, float]]
    ) -> bool:
        """Check if a point is inside the circumcircle of a triangle."""
        def get_point(idx: int) -> Tuple[float, float]:
            if idx < 0:
                return all_points[len(all_points) + idx]
            return all_points[idx]

        ax, ay = get_point(triangle[0])
        bx, by = get_point(triangle[1])
        cx, cy = get_point(triangle[2])
        dx, dy = point

        # Matrix determinant method
        ax_ = ax - dx
        ay_ = ay - dy
        bx_ = bx - dx
        by_ = by - dy
        cx_ = cx - dx
        cy_ = cy - dy

        det = (
            (ax_ * ax_ + ay_ * ay_) * (bx_ * cy_ - cx_ * by_) -
            (bx_ * bx_ + by_ * by_) * (ax_ * cy_ - cx_ * ay_) +
            (cx_ * cx_ + cy_ * cy_) * (ax_ * by_ - bx_ * ay_)
        )

        return det > 0

    def get_weights(self, context: GraphContext) -> Dict[int, float]:
        """Get the blend weights for each sample."""
        if not self.samples:
            return {}

        x = context.get_parameter_float(self.param_x, 0.0)
        y = context.get_parameter_float(self.param_y, 0.0)

        return self._compute_inverse_distance_weights((x, y))


# =============================================================================
# DIRECT BLEND TREE
# =============================================================================


@dataclass
class BlendTreeDirectEntry:
    """An entry in a direct blend tree with explicit weight."""

    node: AnimationNode
    weight_parameter: Optional[str] = None  # If None, uses fixed weight
    fixed_weight: float = 1.0


class BlendTreeDirect(BlendTree):
    """
    A blend tree with explicitly controlled weights.

    Each child has a weight that can be a parameter or a fixed value.
    The tree normalizes weights and blends all children.

    Useful for:
    - Layered blending with parameter-controlled weights
    - Additive animation layers
    - Complex multi-animation blends
    """

    _abstract = False

    def __init__(self, node_id: str, normalize_weights: bool = True) -> None:
        super().__init__(node_id)
        self.entries: List[BlendTreeDirectEntry] = []
        self.normalize_weights = normalize_weights

    def add_entry(self, node: AnimationNode,
                  weight_parameter: Optional[str] = None,
                  fixed_weight: float = 1.0) -> int:
        """Add an entry with explicit weight control."""
        entry = BlendTreeDirectEntry(
            node=node,
            weight_parameter=weight_parameter,
            fixed_weight=fixed_weight,
        )
        self.entries.append(entry)

        if node not in self.children:
            self.children.append(node)

        return len(self.entries) - 1

    def remove_entry(self, index: int) -> bool:
        """Remove an entry by index."""
        if 0 <= index < len(self.entries):
            entry = self.entries.pop(index)
            if entry.node in self.children:
                self.children.remove(entry.node)
            return True
        return False

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate the blend tree with explicit weights."""
        if not self.entries:
            return Pose()

        if len(self.entries) == 1:
            return self.entries[0].node.evaluate(context)

        # Compute weights
        weights = []
        for entry in self.entries:
            if entry.weight_parameter:
                weight = context.get_parameter_float(entry.weight_parameter, 0.0)
            else:
                weight = entry.fixed_weight
            weights.append(max(0.0, weight))

        # Normalize if needed
        total = sum(weights)
        if total <= 0:
            return Pose()
        if self.normalize_weights:
            weights = [w / total for w in weights]

        # Blend poses
        result_pose = None

        for entry, weight in zip(self.entries, weights):
            if weight <= 0:
                continue

            pose = entry.node.evaluate(context)

            if result_pose is None:
                result_pose = Pose.identity(pose.bone_count())

            # Accumulate weighted pose
            for i in range(min(result_pose.bone_count(), pose.bone_count())):
                weighted_transform = Transform.identity().lerp(
                    pose.transforms[i], weight
                )
                # Add to result (simplified accumulation)
                result_pose.transforms[i] = Transform(
                    position=tuple(
                        a + b for a, b in zip(
                            result_pose.transforms[i].position,
                            tuple(c * weight for c in pose.transforms[i].position)
                        )
                    ) if result_pose.transforms[i].position != Transform.identity().position
                    else weighted_transform.position,
                    rotation=weighted_transform.rotation,  # Last wins for now
                    scale=weighted_transform.scale,
                )

        return result_pose or Pose()

    def get_weights(self, context: GraphContext) -> Dict[int, float]:
        """Get the current weights for each entry."""
        weights = {}
        total = 0.0

        for i, entry in enumerate(self.entries):
            if entry.weight_parameter:
                weight = context.get_parameter_float(entry.weight_parameter, 0.0)
            else:
                weight = entry.fixed_weight
            weight = max(0.0, weight)
            weights[i] = weight
            total += weight

        if self.normalize_weights and total > 0:
            weights = {i: w / total for i, w in weights.items()}

        return weights


# =============================================================================
# BLEND TREE DECORATOR
# =============================================================================


def blend_tree(parameter: str, clips: List[str]) -> Callable[[Type], Type]:
    """
    Decorator to define a blend tree class.

    Usage:
        @blend_tree(parameter="speed", clips=["idle", "walk", "run"])
        class LocomotionBlendTree:
            pass
    """
    def decorator(cls: Type) -> Type:
        cls._blend_tree = True
        cls._blend_parameter = parameter
        cls._blend_clips = list(clips)
        return cls
    return decorator


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Base
    "BlendTree",
    # 1D
    "BlendTree1DEntry",
    "BlendTree1D",
    # 2D
    "BlendTree2DMode",
    "BlendTree2DSample",
    "Triangle",
    "BlendTree2D",
    # Direct
    "BlendTreeDirectEntry",
    "BlendTreeDirect",
    # Decorator
    "blend_tree",
]
