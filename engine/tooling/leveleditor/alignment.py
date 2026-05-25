"""
Alignment Tools - Align objects relative to each other or reference points.

Provides alignment operations:
- Left, Center, Right (X-axis)
- Top, Middle, Bottom (Y-axis)
- Front, Center, Back (Z-axis)
- Align to selection bounds, active object, or custom reference

All alignment operations integrate with Foundation Tracker for undo/redo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from .placement import Vector3, Quaternion, Transform, editor, track_changes
from foundation.tracker import tracker


# =============================================================================
# Enums
# =============================================================================

class AlignAxis(Enum):
    """Axis for alignment."""
    X = auto()
    Y = auto()
    Z = auto()


class AlignEdge(Enum):
    """Edge/side for alignment along an axis."""
    MIN = auto()  # Left, Bottom, Front
    CENTER = auto()  # Center
    MAX = auto()  # Right, Top, Back


class AlignReference(Enum):
    """Reference point for alignment."""
    SELECTION_BOUNDS = auto()  # Align to combined selection bounds
    ACTIVE_OBJECT = auto()  # Align to the active/primary selected object
    FIRST_SELECTED = auto()  # Align to first selected object
    LAST_SELECTED = auto()  # Align to last selected object
    WORLD_ORIGIN = auto()  # Align to world origin
    CURSOR = auto()  # Align to 3D cursor
    CUSTOM = auto()  # Align to custom position


class SpaceMode(Enum):
    """Coordinate space for alignment."""
    WORLD = auto()
    LOCAL = auto()
    ACTIVE_OBJECT = auto()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(slots=True)
class AlignTarget:
    """Target object for alignment."""
    object_id: str
    position: Vector3
    bounds_min: Vector3
    bounds_max: Vector3
    pivot: Vector3

    @property
    def center(self) -> Vector3:
        return Vector3(
            (self.bounds_min.x + self.bounds_max.x) / 2,
            (self.bounds_min.y + self.bounds_max.y) / 2,
            (self.bounds_min.z + self.bounds_max.z) / 2,
        )

    @property
    def size(self) -> Vector3:
        return Vector3(
            self.bounds_max.x - self.bounds_min.x,
            self.bounds_max.y - self.bounds_min.y,
            self.bounds_max.z - self.bounds_min.z,
        )


@dataclass(slots=True)
class AlignmentSettings:
    """Settings for alignment operations."""
    axis: AlignAxis = AlignAxis.X
    edge: AlignEdge = AlignEdge.MIN
    reference: AlignReference = AlignReference.SELECTION_BOUNDS
    space: SpaceMode = SpaceMode.WORLD
    use_pivot: bool = False  # Align pivot instead of bounds
    preserve_offset: bool = False  # Maintain relative offsets
    custom_position: Optional[Vector3] = None


@dataclass(slots=True)
class AlignmentResult:
    """Result of an alignment operation."""
    success: bool
    objects_aligned: int
    error_message: Optional[str] = None
    original_positions: dict[str, Vector3] = field(default_factory=dict)
    new_positions: dict[str, Vector3] = field(default_factory=dict)


# =============================================================================
# Alignment Tool
# =============================================================================

@editor
class AlignmentTool:
    """
    Tool for aligning objects in the scene.

    Supports multiple alignment modes and reference points.
    All operations integrate with Foundation Tracker for undo/redo.
    """

    __slots__ = (
        "_settings",
        "_targets",
        "_active_target_id",
        "_cursor_position",
        "_callbacks",
        "__weakref__",
    )

    def __init__(self):
        """Initialize alignment tool."""
        self._settings = AlignmentSettings()
        self._targets: list[AlignTarget] = []
        self._active_target_id: Optional[str] = None
        self._cursor_position = Vector3()
        self._callbacks: dict[str, list[Callable]] = {
            "on_align": [],
            "on_preview": [],
        }

    @property
    def settings(self) -> AlignmentSettings:
        return self._settings

    @settings.setter
    def settings(self, value: AlignmentSettings) -> None:
        self._settings = value

    @property
    def cursor_position(self) -> Vector3:
        return self._cursor_position

    @cursor_position.setter
    def cursor_position(self, value: Vector3) -> None:
        self._cursor_position = value

    def on(self, event: str, callback: Callable) -> None:
        """Register callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Unregister callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def set_targets(
        self,
        targets: list[AlignTarget],
        active_id: Optional[str] = None
    ) -> None:
        """
        Set the objects to align.

        Args:
            targets: List of alignment targets
            active_id: ID of the active/reference object
        """
        self._targets = targets
        self._active_target_id = active_id or (targets[0].object_id if targets else None)

    def get_reference_position(self) -> Optional[Vector3]:
        """Get the alignment reference position based on settings."""
        if not self._targets:
            return None

        settings = self._settings

        if settings.reference == AlignReference.WORLD_ORIGIN:
            return Vector3(0, 0, 0)

        if settings.reference == AlignReference.CURSOR:
            return self._cursor_position

        if settings.reference == AlignReference.CUSTOM and settings.custom_position:
            return settings.custom_position

        if settings.reference == AlignReference.ACTIVE_OBJECT:
            for target in self._targets:
                if target.object_id == self._active_target_id:
                    return self._get_target_align_position(target)
            return None

        if settings.reference == AlignReference.FIRST_SELECTED:
            return self._get_target_align_position(self._targets[0])

        if settings.reference == AlignReference.LAST_SELECTED:
            return self._get_target_align_position(self._targets[-1])

        if settings.reference == AlignReference.SELECTION_BOUNDS:
            return self._get_selection_bounds_position()

        return None

    def _get_target_align_position(self, target: AlignTarget) -> Vector3:
        """Get alignment position for a target based on current settings."""
        settings = self._settings

        if settings.use_pivot:
            return target.pivot

        # Get position based on edge
        pos = Vector3()
        if settings.axis == AlignAxis.X:
            if settings.edge == AlignEdge.MIN:
                pos.x = target.bounds_min.x
            elif settings.edge == AlignEdge.CENTER:
                pos.x = (target.bounds_min.x + target.bounds_max.x) / 2
            else:
                pos.x = target.bounds_max.x
            pos.y = target.center.y
            pos.z = target.center.z
        elif settings.axis == AlignAxis.Y:
            pos.x = target.center.x
            if settings.edge == AlignEdge.MIN:
                pos.y = target.bounds_min.y
            elif settings.edge == AlignEdge.CENTER:
                pos.y = (target.bounds_min.y + target.bounds_max.y) / 2
            else:
                pos.y = target.bounds_max.y
            pos.z = target.center.z
        else:  # Z
            pos.x = target.center.x
            pos.y = target.center.y
            if settings.edge == AlignEdge.MIN:
                pos.z = target.bounds_min.z
            elif settings.edge == AlignEdge.CENTER:
                pos.z = (target.bounds_min.z + target.bounds_max.z) / 2
            else:
                pos.z = target.bounds_max.z

        return pos

    def _get_selection_bounds_position(self) -> Vector3:
        """Get alignment position from combined selection bounds."""
        if not self._targets:
            return Vector3()

        settings = self._settings

        # Calculate combined bounds
        min_x = min(t.bounds_min.x for t in self._targets)
        min_y = min(t.bounds_min.y for t in self._targets)
        min_z = min(t.bounds_min.z for t in self._targets)
        max_x = max(t.bounds_max.x for t in self._targets)
        max_y = max(t.bounds_max.y for t in self._targets)
        max_z = max(t.bounds_max.z for t in self._targets)

        pos = Vector3()

        if settings.axis == AlignAxis.X:
            if settings.edge == AlignEdge.MIN:
                pos.x = min_x
            elif settings.edge == AlignEdge.CENTER:
                pos.x = (min_x + max_x) / 2
            else:
                pos.x = max_x
        elif settings.axis == AlignAxis.Y:
            if settings.edge == AlignEdge.MIN:
                pos.y = min_y
            elif settings.edge == AlignEdge.CENTER:
                pos.y = (min_y + max_y) / 2
            else:
                pos.y = max_y
        else:  # Z
            if settings.edge == AlignEdge.MIN:
                pos.z = min_z
            elif settings.edge == AlignEdge.CENTER:
                pos.z = (min_z + max_z) / 2
            else:
                pos.z = max_z

        return pos

    def calculate_new_position(self, target: AlignTarget, reference_pos: Vector3) -> Vector3:
        """
        Calculate new position for a target after alignment.

        Args:
            target: Target to calculate for
            reference_pos: Reference position to align to

        Returns:
            New position for the target
        """
        settings = self._settings
        new_pos = Vector3(target.position.x, target.position.y, target.position.z)

        # Calculate offset from target's align point to its position
        target_align_pos = self._get_target_align_position(target)

        if settings.axis == AlignAxis.X:
            offset = target.position.x - target_align_pos.x
            new_pos.x = reference_pos.x + offset
        elif settings.axis == AlignAxis.Y:
            offset = target.position.y - target_align_pos.y
            new_pos.y = reference_pos.y + offset
        else:  # Z
            offset = target.position.z - target_align_pos.z
            new_pos.z = reference_pos.z + offset

        return new_pos

    def preview_alignment(self) -> dict[str, Vector3]:
        """
        Preview alignment positions without applying.

        Returns:
            Dictionary of object_id to new position
        """
        reference_pos = self.get_reference_position()
        if reference_pos is None:
            return {}

        preview = {}
        for target in self._targets:
            # Skip reference object if aligning to active/first/last
            settings = self._settings
            if settings.reference in [
                AlignReference.ACTIVE_OBJECT,
                AlignReference.FIRST_SELECTED,
                AlignReference.LAST_SELECTED,
            ]:
                if settings.reference == AlignReference.ACTIVE_OBJECT and target.object_id == self._active_target_id:
                    continue
                if settings.reference == AlignReference.FIRST_SELECTED and target == self._targets[0]:
                    continue
                if settings.reference == AlignReference.LAST_SELECTED and target == self._targets[-1]:
                    continue

            new_pos = self.calculate_new_position(target, reference_pos)
            preview[target.object_id] = new_pos

        for callback in self._callbacks["on_preview"]:
            callback(preview)

        return preview

    @track_changes
    def align(self) -> AlignmentResult:
        """
        Perform the alignment operation.

        Returns:
            AlignmentResult with operation details
        """
        if not self._targets:
            return AlignmentResult(
                success=False,
                objects_aligned=0,
                error_message="No targets to align"
            )

        reference_pos = self.get_reference_position()
        if reference_pos is None:
            return AlignmentResult(
                success=False,
                objects_aligned=0,
                error_message="Could not determine reference position"
            )

        original_positions: dict[str, Vector3] = {}
        new_positions: dict[str, Vector3] = {}

        for target in self._targets:
            # Skip reference object based on settings
            settings = self._settings
            skip = False
            if settings.reference == AlignReference.ACTIVE_OBJECT and target.object_id == self._active_target_id:
                skip = True
            if settings.reference == AlignReference.FIRST_SELECTED and target == self._targets[0]:
                skip = True
            if settings.reference == AlignReference.LAST_SELECTED and target == self._targets[-1]:
                skip = True

            if skip:
                continue

            original_positions[target.object_id] = Vector3(
                target.position.x,
                target.position.y,
                target.position.z
            )

            new_pos = self.calculate_new_position(target, reference_pos)
            new_positions[target.object_id] = new_pos

            # Update target position
            target.position.x = new_pos.x
            target.position.y = new_pos.y
            target.position.z = new_pos.z

        # Track the overall alignment operation on self, not individual targets
        if original_positions:
            tracker.mark_dirty(self, "_targets", original_positions, new_positions)

        result = AlignmentResult(
            success=True,
            objects_aligned=len(new_positions),
            original_positions=original_positions,
            new_positions=new_positions,
        )

        for callback in self._callbacks["on_align"]:
            callback(result)

        return result

    # Convenience methods for common alignments
    @track_changes
    def align_left(self) -> AlignmentResult:
        """Align to left (min X)."""
        self._settings.axis = AlignAxis.X
        self._settings.edge = AlignEdge.MIN
        return self.align()

    @track_changes
    def align_center_x(self) -> AlignmentResult:
        """Align to center X."""
        self._settings.axis = AlignAxis.X
        self._settings.edge = AlignEdge.CENTER
        return self.align()

    @track_changes
    def align_right(self) -> AlignmentResult:
        """Align to right (max X)."""
        self._settings.axis = AlignAxis.X
        self._settings.edge = AlignEdge.MAX
        return self.align()

    @track_changes
    def align_bottom(self) -> AlignmentResult:
        """Align to bottom (min Y)."""
        self._settings.axis = AlignAxis.Y
        self._settings.edge = AlignEdge.MIN
        return self.align()

    @track_changes
    def align_center_y(self) -> AlignmentResult:
        """Align to center Y."""
        self._settings.axis = AlignAxis.Y
        self._settings.edge = AlignEdge.CENTER
        return self.align()

    @track_changes
    def align_top(self) -> AlignmentResult:
        """Align to top (max Y)."""
        self._settings.axis = AlignAxis.Y
        self._settings.edge = AlignEdge.MAX
        return self.align()

    @track_changes
    def align_front(self) -> AlignmentResult:
        """Align to front (min Z)."""
        self._settings.axis = AlignAxis.Z
        self._settings.edge = AlignEdge.MIN
        return self.align()

    @track_changes
    def align_center_z(self) -> AlignmentResult:
        """Align to center Z."""
        self._settings.axis = AlignAxis.Z
        self._settings.edge = AlignEdge.CENTER
        return self.align()

    @track_changes
    def align_back(self) -> AlignmentResult:
        """Align to back (max Z)."""
        self._settings.axis = AlignAxis.Z
        self._settings.edge = AlignEdge.MAX
        return self.align()

    @track_changes
    def align_to_ground(self, ground_y: float = 0.0) -> AlignmentResult:
        """
        Align objects to ground plane.

        Args:
            ground_y: Y coordinate of ground

        Returns:
            AlignmentResult
        """
        self._settings.axis = AlignAxis.Y
        self._settings.edge = AlignEdge.MIN
        self._settings.reference = AlignReference.CUSTOM
        self._settings.custom_position = Vector3(0, ground_y, 0)
        return self.align()

    @track_changes
    def stack_vertically(self, spacing: float = 0.0) -> AlignmentResult:
        """
        Stack objects vertically.

        Args:
            spacing: Gap between objects

        Returns:
            AlignmentResult
        """
        if len(self._targets) < 2:
            return AlignmentResult(
                success=False,
                objects_aligned=0,
                error_message="Need at least 2 objects to stack"
            )

        # Sort by Y position
        sorted_targets = sorted(self._targets, key=lambda t: t.position.y)

        original_positions: dict[str, Vector3] = {}
        new_positions: dict[str, Vector3] = {}

        current_y = sorted_targets[0].bounds_max.y

        for i, target in enumerate(sorted_targets):
            original_positions[target.object_id] = Vector3(
                target.position.x,
                target.position.y,
                target.position.z
            )

            if i > 0:
                # Calculate new Y based on stacking
                offset = target.position.y - target.bounds_min.y
                new_y = current_y + spacing + offset
                target.position.y = new_y
                current_y = target.bounds_max.y + (new_y - target.position.y)
            else:
                current_y = target.bounds_max.y

            new_positions[target.object_id] = Vector3(
                target.position.x,
                target.position.y,
                target.position.z
            )

        # Track the overall stack operation on self
        if original_positions:
            tracker.mark_dirty(self, "_targets", original_positions, new_positions)

        return AlignmentResult(
            success=True,
            objects_aligned=len(new_positions),
            original_positions=original_positions,
            new_positions=new_positions,
        )


__all__ = [
    "AlignAxis",
    "AlignEdge",
    "AlignReference",
    "SpaceMode",
    "AlignTarget",
    "AlignmentSettings",
    "AlignmentResult",
    "AlignmentTool",
]
