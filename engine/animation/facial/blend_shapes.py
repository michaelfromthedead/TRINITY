"""
Blend Shape / Morph Target System.

Provides vertex morphing capabilities for facial animation and other
deformable mesh systems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np


# =============================================================================
# Vector3 Type Alias (for clarity)
# =============================================================================

Vector3 = tuple[float, float, float]


# =============================================================================
# Blend Shape Data Structures
# =============================================================================


@dataclass
class BlendShape:
    """
    A single blend shape / morph target.

    Contains the delta offsets to apply to base mesh vertices
    when this shape is activated.

    Attributes:
        name: Unique name of the blend shape (e.g., "smile", "AU12_left")
        vertex_indices: Indices of affected vertices (sparse representation)
        deltas: Position offsets for each affected vertex
        normal_deltas: Optional normal offsets for each affected vertex
        tangent_deltas: Optional tangent offsets (for normal mapping)
    """
    name: str
    vertex_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    deltas: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32).reshape(0, 3))
    normal_deltas: Optional[np.ndarray] = None
    tangent_deltas: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate and convert data types."""
        if isinstance(self.vertex_indices, (list, tuple)):
            self.vertex_indices = np.array(self.vertex_indices, dtype=np.int32)
        if isinstance(self.deltas, (list, tuple)):
            self.deltas = np.array(self.deltas, dtype=np.float32)

        # Ensure deltas is 2D (N, 3)
        if len(self.deltas.shape) == 1 and len(self.deltas) > 0:
            self.deltas = self.deltas.reshape(-1, 3)

    @property
    def vertex_count(self) -> int:
        """Get number of affected vertices."""
        return len(self.vertex_indices)

    @property
    def is_sparse(self) -> bool:
        """Check if this is a sparse representation."""
        return len(self.vertex_indices) > 0

    def get_delta(self, local_index: int) -> Vector3:
        """
        Get delta for a specific local index.

        Args:
            local_index: Index into the sparse arrays

        Returns:
            Delta vector (x, y, z)
        """
        if local_index < 0 or local_index >= len(self.deltas):
            return (0.0, 0.0, 0.0)
        return tuple(self.deltas[local_index])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "vertex_indices": self.vertex_indices.tolist(),
            "deltas": self.deltas.tolist(),
        }
        if self.normal_deltas is not None:
            result["normal_deltas"] = self.normal_deltas.tolist()
        if self.tangent_deltas is not None:
            result["tangent_deltas"] = self.tangent_deltas.tolist()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlendShape:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            vertex_indices=np.array(data["vertex_indices"], dtype=np.int32),
            deltas=np.array(data["deltas"], dtype=np.float32),
            normal_deltas=np.array(data["normal_deltas"], dtype=np.float32) if "normal_deltas" in data else None,
            tangent_deltas=np.array(data["tangent_deltas"], dtype=np.float32) if "tangent_deltas" in data else None,
        )


@dataclass
class CorrectiveBlendShape:
    """
    A corrective blend shape that activates based on combinations of other shapes.

    Used to fix artifacts when multiple blend shapes are combined.
    For example, a corrective that fixes the corner of the mouth
    when both smile and mouth-open are active.

    Attributes:
        shape: The blend shape data
        driver_shapes: Names of shapes that drive this corrective
        driver_weights: Minimum weights required for each driver
        combination_mode: How drivers are combined ("multiply", "min", "add")
    """
    shape: BlendShape
    driver_shapes: list[str] = field(default_factory=list)
    driver_weights: list[float] = field(default_factory=list)
    combination_mode: str = "multiply"

    def __post_init__(self) -> None:
        """Validate driver configuration."""
        if len(self.driver_weights) == 0:
            self.driver_weights = [0.5] * len(self.driver_shapes)
        elif len(self.driver_weights) != len(self.driver_shapes):
            raise ValueError("driver_weights must match driver_shapes length")

    @property
    def name(self) -> str:
        """Get the name of the underlying shape."""
        return self.shape.name

    def calculate_weight(self, current_weights: dict[str, float]) -> float:
        """
        Calculate the corrective weight based on driver weights.

        Args:
            current_weights: Current weights of all blend shapes

        Returns:
            Calculated corrective weight
        """
        if not self.driver_shapes:
            return 0.0

        driver_values = []
        for shape_name, threshold in zip(self.driver_shapes, self.driver_weights):
            weight = current_weights.get(shape_name, 0.0)
            # Normalize based on threshold
            if weight >= threshold:
                normalized = (weight - threshold) / (1.0 - threshold) if threshold < 1.0 else weight
                driver_values.append(max(0.0, min(1.0, normalized)))
            else:
                driver_values.append(0.0)

        if not driver_values:
            return 0.0

        if self.combination_mode == "multiply":
            result = 1.0
            for v in driver_values:
                result *= v
            return result
        elif self.combination_mode == "min":
            return min(driver_values)
        elif self.combination_mode == "add":
            return min(1.0, sum(driver_values) / len(driver_values))
        else:
            return 0.0


@dataclass
class BlendShapeSet:
    """
    Collection of blend shapes for a single mesh.

    Manages the base mesh and all associated blend shapes.

    Attributes:
        name: Name of this blend shape set (typically mesh name)
        base_vertices: The base mesh vertex positions (N, 3)
        blend_shapes: Dictionary of blend shapes by name
        correctives: List of corrective blend shapes
    """
    name: str
    base_vertices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32).reshape(0, 3))
    blend_shapes: dict[str, BlendShape] = field(default_factory=dict)
    correctives: list[CorrectiveBlendShape] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure base_vertices is properly shaped."""
        if isinstance(self.base_vertices, (list, tuple)):
            self.base_vertices = np.array(self.base_vertices, dtype=np.float32)
        if len(self.base_vertices.shape) == 1 and len(self.base_vertices) > 0:
            self.base_vertices = self.base_vertices.reshape(-1, 3)

    @property
    def vertex_count(self) -> int:
        """Get number of vertices in base mesh."""
        return len(self.base_vertices)

    @property
    def shape_count(self) -> int:
        """Get number of blend shapes."""
        return len(self.blend_shapes)

    @property
    def shape_names(self) -> list[str]:
        """Get list of blend shape names."""
        return list(self.blend_shapes.keys())

    def add_shape(self, shape: BlendShape) -> None:
        """
        Add a blend shape to the set.

        Args:
            shape: The blend shape to add
        """
        self.blend_shapes[shape.name] = shape

    def remove_shape(self, name: str) -> bool:
        """
        Remove a blend shape from the set.

        Args:
            name: Name of shape to remove

        Returns:
            True if shape was removed
        """
        if name in self.blend_shapes:
            del self.blend_shapes[name]
            return True
        return False

    def get_shape(self, name: str) -> Optional[BlendShape]:
        """Get a blend shape by name."""
        return self.blend_shapes.get(name)

    def add_corrective(self, corrective: CorrectiveBlendShape) -> None:
        """Add a corrective blend shape."""
        self.correctives.append(corrective)

    def has_shape(self, name: str) -> bool:
        """Check if a blend shape exists."""
        return name in self.blend_shapes


# =============================================================================
# Blend Shape Application
# =============================================================================


def apply_blend_shapes(
    base_vertices: np.ndarray,
    shapes: dict[str, BlendShape],
    weights: dict[str, float],
    normalize_weights: bool = False,
    clamp_weights: bool = True,
    weight_min: float = 0.0,
    weight_max: float = 1.0,
) -> np.ndarray:
    """
    Apply weighted blend shapes to base vertices.

    Args:
        base_vertices: Base mesh vertices (N, 3)
        shapes: Dictionary of blend shapes by name
        weights: Dictionary of weights by shape name
        normalize_weights: If True, normalize weights to sum to 1
        clamp_weights: If True, clamp weights to [weight_min, weight_max]
        weight_min: Minimum allowed weight (default 0.0)
        weight_max: Maximum allowed weight (default 1.0)

    Returns:
        Morphed vertex positions (N, 3)
    """
    if not shapes or not weights:
        return base_vertices.copy()

    # Filter to active weights and optionally clamp
    active_weights = {}
    for k, v in weights.items():
        if k not in shapes:
            continue
        if clamp_weights:
            v = max(weight_min, min(weight_max, v))
        if v != 0.0:
            active_weights[k] = v

    if not active_weights:
        return base_vertices.copy()

    # Optionally normalize
    if normalize_weights and active_weights:
        total = sum(abs(w) for w in active_weights.values())
        if total > 0:
            active_weights = {k: v / total for k, v in active_weights.items()}

    # Apply blend shapes
    result = base_vertices.copy()

    for name, weight in active_weights.items():
        shape = shapes[name]
        if shape.vertex_count == 0:
            continue

        # Apply deltas with weight
        if shape.is_sparse:
            # Sparse application
            result[shape.vertex_indices] += shape.deltas * weight
        else:
            # Dense application (full vertex count)
            result += shape.deltas * weight

    return result


def apply_blend_shapes_with_correctives(
    base_vertices: np.ndarray,
    shape_set: BlendShapeSet,
    weights: dict[str, float],
) -> np.ndarray:
    """
    Apply blend shapes including correctives.

    Args:
        base_vertices: Base mesh vertices (N, 3)
        shape_set: The blend shape set with correctives
        weights: Dictionary of weights by shape name

    Returns:
        Morphed vertex positions (N, 3)
    """
    # First apply base blend shapes
    result = apply_blend_shapes(base_vertices, shape_set.blend_shapes, weights)

    # Then apply correctives
    for corrective in shape_set.correctives:
        corrective_weight = corrective.calculate_weight(weights)
        if corrective_weight > 0.001:  # Threshold for numerical stability
            shape = corrective.shape
            if shape.is_sparse:
                result[shape.vertex_indices] += shape.deltas * corrective_weight
            else:
                result += shape.deltas * corrective_weight

    return result


# =============================================================================
# Blend Shape Controller
# =============================================================================


class BlendShapeController:
    """
    Controller for managing blend shape weights on a mesh.

    Provides high-level interface for setting/getting weights,
    transitioning between weights, and querying state.
    """

    def __init__(
        self,
        shape_set: BlendShapeSet,
        on_weights_changed: Optional[Callable[[dict[str, float]], None]] = None,
    ) -> None:
        """
        Initialize the controller.

        Args:
            shape_set: The blend shape set to control
            on_weights_changed: Callback when weights change
        """
        self._shape_set = shape_set
        self._weights: dict[str, float] = {name: 0.0 for name in shape_set.shape_names}
        self._target_weights: dict[str, float] = {}
        self._transition_speeds: dict[str, float] = {}
        self._on_weights_changed = on_weights_changed
        self._dirty = False

    @property
    def shape_set(self) -> BlendShapeSet:
        """Get the blend shape set."""
        return self._shape_set

    @property
    def weights(self) -> dict[str, float]:
        """Get current weights (read-only copy)."""
        return self._weights.copy()

    @property
    def dirty(self) -> bool:
        """Check if weights have changed since last query."""
        return self._dirty

    def set_weight(self, name: str, weight: float, clamp: bool = True) -> bool:
        """
        Set weight for a single blend shape.

        Args:
            name: Blend shape name
            weight: Weight value
            clamp: If True, clamp to [0, 1]

        Returns:
            True if weight was set
        """
        if name not in self._weights:
            return False

        if clamp:
            weight = max(0.0, min(1.0, weight))

        if self._weights[name] != weight:
            self._weights[name] = weight
            self._dirty = True
            self._notify_change()

        return True

    def get_weight(self, name: str) -> float:
        """
        Get weight for a blend shape.

        Args:
            name: Blend shape name

        Returns:
            Weight value, or 0.0 if not found
        """
        return self._weights.get(name, 0.0)

    def set_weights(self, weights: dict[str, float], clamp: bool = True) -> None:
        """
        Set multiple weights at once.

        Args:
            weights: Dictionary of weights to set
            clamp: If True, clamp values to [0, 1]
        """
        changed = False
        for name, weight in weights.items():
            if name in self._weights:
                if clamp:
                    weight = max(0.0, min(1.0, weight))
                if self._weights[name] != weight:
                    self._weights[name] = weight
                    changed = True

        if changed:
            self._dirty = True
            self._notify_change()

    def reset_all(self) -> None:
        """Reset all weights to zero."""
        changed = any(w != 0.0 for w in self._weights.values())
        self._weights = {name: 0.0 for name in self._weights}
        self._target_weights.clear()
        self._transition_speeds.clear()

        if changed:
            self._dirty = True
            self._notify_change()

    def reset_weights(self, names: Sequence[str]) -> None:
        """
        Reset specific weights to zero.

        Args:
            names: Names of weights to reset
        """
        changed = False
        for name in names:
            if name in self._weights and self._weights[name] != 0.0:
                self._weights[name] = 0.0
                changed = True
                self._target_weights.pop(name, None)
                self._transition_speeds.pop(name, None)

        if changed:
            self._dirty = True
            self._notify_change()

    def set_target_weight(
        self,
        name: str,
        target: float,
        speed: float = 10.0,
    ) -> bool:
        """
        Set target weight for smooth transition.

        Args:
            name: Blend shape name
            target: Target weight
            speed: Transition speed (units per second)

        Returns:
            True if target was set
        """
        if name not in self._weights:
            return False

        target = max(0.0, min(1.0, target))
        self._target_weights[name] = target
        self._transition_speeds[name] = max(0.001, abs(speed))
        return True

    def cancel_transition(self, name: str) -> None:
        """Cancel an active transition."""
        self._target_weights.pop(name, None)
        self._transition_speeds.pop(name, None)

    def cancel_all_transitions(self) -> None:
        """Cancel all active transitions."""
        self._target_weights.clear()
        self._transition_speeds.clear()

    def update(self, dt: float) -> bool:
        """
        Update transitions.

        Args:
            dt: Delta time in seconds

        Returns:
            True if any weights changed
        """
        if not self._target_weights:
            return False

        changed = False
        completed = []

        for name, target in self._target_weights.items():
            current = self._weights.get(name, 0.0)
            speed = self._transition_speeds.get(name, 10.0)

            if abs(target - current) < 0.0001:
                self._weights[name] = target
                completed.append(name)
                changed = True
            else:
                direction = 1.0 if target > current else -1.0
                delta = speed * dt * direction

                if direction > 0:
                    new_weight = min(target, current + delta)
                else:
                    new_weight = max(target, current + delta)

                if new_weight != current:
                    self._weights[name] = new_weight
                    changed = True

                if abs(new_weight - target) < 0.0001:
                    completed.append(name)

        # Remove completed transitions
        for name in completed:
            self._target_weights.pop(name, None)
            self._transition_speeds.pop(name, None)

        if changed:
            self._dirty = True
            self._notify_change()

        return changed

    def has_active_transitions(self) -> bool:
        """Check if there are active transitions."""
        return len(self._target_weights) > 0

    def get_active_shapes(self, threshold: float = 0.001) -> list[str]:
        """
        Get names of shapes with non-zero weights.

        Args:
            threshold: Minimum weight to consider active

        Returns:
            List of active shape names
        """
        return [name for name, weight in self._weights.items() if abs(weight) >= threshold]

    def apply_to_mesh(
        self,
        base_vertices: Optional[np.ndarray] = None,
        include_correctives: bool = True,
    ) -> np.ndarray:
        """
        Apply current weights to get morphed vertices.

        Args:
            base_vertices: Optional override for base vertices
            include_correctives: Whether to apply corrective shapes

        Returns:
            Morphed vertex positions
        """
        vertices = base_vertices if base_vertices is not None else self._shape_set.base_vertices

        if include_correctives:
            return apply_blend_shapes_with_correctives(vertices, self._shape_set, self._weights)
        else:
            return apply_blend_shapes(vertices, self._shape_set.blend_shapes, self._weights)

    def clear_dirty(self) -> None:
        """Clear the dirty flag."""
        self._dirty = False

    def _notify_change(self) -> None:
        """Notify change callback."""
        if self._on_weights_changed:
            self._on_weights_changed(self._weights.copy())


# =============================================================================
# ARKit 52 Blend Shape Compatibility
# =============================================================================


# Standard ARKit blend shape names for iOS face tracking compatibility
ARKIT_BLEND_SHAPES = [
    # Eye shapes
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",

    # Jaw
    "jawForward", "jawLeft", "jawRight", "jawOpen",

    # Mouth shapes
    "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",

    # Brow shapes
    "browDownLeft", "browDownRight",
    "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",

    # Cheek and nose
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",

    # Tongue (may not be tracked)
    "tongueOut",
]


def create_arkit_compatible_set(name: str, vertex_count: int) -> BlendShapeSet:
    """
    Create a blend shape set with ARKit-compatible shape names.

    Args:
        name: Name for the blend shape set
        vertex_count: Number of vertices in the mesh

    Returns:
        BlendShapeSet with empty ARKit-named shapes
    """
    base_vertices = np.zeros((vertex_count, 3), dtype=np.float32)
    shapes = {
        shape_name: BlendShape(
            name=shape_name,
            vertex_indices=np.array([], dtype=np.int32),
            deltas=np.array([], dtype=np.float32).reshape(0, 3),
        )
        for shape_name in ARKIT_BLEND_SHAPES
    }

    return BlendShapeSet(
        name=name,
        base_vertices=base_vertices,
        blend_shapes=shapes,
    )


def remap_blend_shape_weights(
    weights: dict[str, float],
    mapping: dict[str, str],
    missing_value: float = 0.0,
) -> dict[str, float]:
    """
    Remap blend shape weights from one naming convention to another.

    Args:
        weights: Source weights
        mapping: Mapping from source names to target names
        missing_value: Value for unmapped shapes

    Returns:
        Remapped weights
    """
    result = {}
    for source_name, weight in weights.items():
        target_name = mapping.get(source_name, source_name)
        result[target_name] = weight

    return result
