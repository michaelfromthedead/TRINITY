"""
Distribution Tools - Distribute objects evenly in space.

Provides distribution operations:
- Distribute evenly along axis
- Distribute with equal spacing
- Distribute by bounds or pivots
- Pattern-based distribution (circle, grid, etc.)

All distribution operations integrate with Foundation Tracker for undo/redo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from .placement import Vector3, Quaternion, Transform, editor, track_changes
from .alignment import AlignTarget
from foundation.tracker import tracker


# =============================================================================
# Enums
# =============================================================================

class DistributionMode(Enum):
    """Mode of distribution."""
    EVEN_SPACING = auto()  # Equal spacing between objects
    EQUAL_GAPS = auto()  # Equal gaps between bounds
    BY_CENTERS = auto()  # Distribute centers evenly
    BY_PIVOTS = auto()  # Distribute pivots evenly
    PATTERN = auto()  # Use a pattern


class DistributionAxis(Enum):
    """Axis for distribution."""
    X = auto()
    Y = auto()
    Z = auto()
    XY = auto()  # Distribute in XY plane
    XZ = auto()  # Distribute in XZ plane (floor)
    YZ = auto()  # Distribute in YZ plane


class PatternType(Enum):
    """Pattern types for distribution."""
    LINE = auto()
    CIRCLE = auto()
    ARC = auto()
    GRID = auto()
    SPIRAL = auto()
    RANDOM = auto()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(slots=True)
class SpacingSettings:
    """Settings for spacing distribution."""
    fixed_spacing: Optional[float] = None  # None = auto-calculate
    include_first: bool = True
    include_last: bool = True
    reverse_order: bool = False


@dataclass(slots=True)
class PatternSettings:
    """Settings for pattern distribution."""
    pattern_type: PatternType = PatternType.LINE
    # Circle/Arc settings
    radius: float = 10.0
    start_angle: float = 0.0  # Radians
    end_angle: float = 2 * math.pi  # Radians
    # Grid settings
    columns: int = 3
    rows: int = 3
    column_spacing: float = 2.0
    row_spacing: float = 2.0
    # Spiral settings
    spiral_turns: float = 2.0
    spiral_start_radius: float = 1.0
    spiral_end_radius: float = 10.0
    # General
    center: Vector3 = field(default_factory=Vector3)
    orient_to_path: bool = True


@dataclass(slots=True)
class DistributionSettings:
    """Combined settings for distribution operations."""
    mode: DistributionMode = DistributionMode.EVEN_SPACING
    axis: DistributionAxis = DistributionAxis.X
    spacing: SpacingSettings = field(default_factory=SpacingSettings)
    pattern: PatternSettings = field(default_factory=PatternSettings)
    preserve_dimension: bool = True  # Keep objects' positions on non-distributed axes


@dataclass(slots=True)
class DistributionResult:
    """Result of a distribution operation."""
    success: bool
    objects_distributed: int
    error_message: Optional[str] = None
    original_positions: dict[str, Vector3] = field(default_factory=dict)
    new_positions: dict[str, Vector3] = field(default_factory=dict)


# =============================================================================
# Distribution Tool
# =============================================================================

@editor
class DistributionTool:
    """
    Tool for distributing objects in the scene.

    Supports various distribution modes and patterns.
    All operations integrate with Foundation Tracker for undo/redo.
    """

    __slots__ = (
        "_settings",
        "_targets",
        "_callbacks",
        "__weakref__",
    )

    def __init__(self):
        """Initialize distribution tool."""
        self._settings = DistributionSettings()
        self._targets: list[AlignTarget] = []
        self._callbacks: dict[str, list[Callable]] = {
            "on_distribute": [],
            "on_preview": [],
        }

    @property
    def settings(self) -> DistributionSettings:
        return self._settings

    @settings.setter
    def settings(self, value: DistributionSettings) -> None:
        self._settings = value

    def on(self, event: str, callback: Callable) -> None:
        """Register callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Unregister callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def set_targets(self, targets: list[AlignTarget]) -> None:
        """
        Set the objects to distribute.

        Args:
            targets: List of distribution targets
        """
        self._targets = targets

    def _sort_targets_by_axis(
        self,
        targets: list[AlignTarget],
        axis: DistributionAxis
    ) -> list[AlignTarget]:
        """Sort targets by position along axis."""
        if axis == DistributionAxis.X:
            return sorted(targets, key=lambda t: t.position.x)
        elif axis == DistributionAxis.Y:
            return sorted(targets, key=lambda t: t.position.y)
        elif axis == DistributionAxis.Z:
            return sorted(targets, key=lambda t: t.position.z)
        elif axis == DistributionAxis.XY:
            # Sort by X, then Y
            return sorted(targets, key=lambda t: (t.position.x, t.position.y))
        elif axis == DistributionAxis.XZ:
            # Sort by X, then Z
            return sorted(targets, key=lambda t: (t.position.x, t.position.z))
        elif axis == DistributionAxis.YZ:
            # Sort by Y, then Z
            return sorted(targets, key=lambda t: (t.position.y, t.position.z))
        return targets

    def _get_axis_position(self, position: Vector3, axis: DistributionAxis) -> float:
        """Get position along primary axis."""
        if axis in [DistributionAxis.X, DistributionAxis.XY, DistributionAxis.XZ]:
            return position.x
        elif axis in [DistributionAxis.Y, DistributionAxis.YZ]:
            return position.y
        else:
            return position.z

    def _set_axis_position(
        self,
        position: Vector3,
        value: float,
        axis: DistributionAxis
    ) -> Vector3:
        """Set position along primary axis."""
        new_pos = Vector3(position.x, position.y, position.z)
        if axis in [DistributionAxis.X, DistributionAxis.XY, DistributionAxis.XZ]:
            new_pos.x = value
        elif axis in [DistributionAxis.Y, DistributionAxis.YZ]:
            new_pos.y = value
        else:
            new_pos.z = value
        return new_pos

    def _get_target_extent(self, target: AlignTarget, axis: DistributionAxis) -> float:
        """Get object extent along axis."""
        if axis in [DistributionAxis.X, DistributionAxis.XY, DistributionAxis.XZ]:
            return target.bounds_max.x - target.bounds_min.x
        elif axis in [DistributionAxis.Y, DistributionAxis.YZ]:
            return target.bounds_max.y - target.bounds_min.y
        else:
            return target.bounds_max.z - target.bounds_min.z

    def preview_distribution(self) -> dict[str, Vector3]:
        """
        Preview distribution positions without applying.

        Returns:
            Dictionary of object_id to new position
        """
        settings = self._settings

        if settings.mode == DistributionMode.PATTERN:
            preview = self._calculate_pattern_positions()
        else:
            preview = self._calculate_linear_positions()

        for callback in self._callbacks["on_preview"]:
            callback(preview)

        return preview

    def _calculate_linear_positions(self) -> dict[str, Vector3]:
        """Calculate positions for linear distribution."""
        if len(self._targets) < 2:
            return {}

        settings = self._settings
        sorted_targets = self._sort_targets_by_axis(self._targets, settings.axis)

        if settings.spacing.reverse_order:
            sorted_targets = list(reversed(sorted_targets))

        # Get range
        first = sorted_targets[0]
        last = sorted_targets[-1]

        first_pos = self._get_axis_position(first.position, settings.axis)
        last_pos = self._get_axis_position(last.position, settings.axis)

        positions: dict[str, Vector3] = {}

        if settings.mode == DistributionMode.EVEN_SPACING:
            if settings.spacing.fixed_spacing is not None:
                spacing = settings.spacing.fixed_spacing
            else:
                # Auto-calculate to fit between first and last
                spacing = (last_pos - first_pos) / (len(sorted_targets) - 1)

            for i, target in enumerate(sorted_targets):
                if i == 0 and not settings.spacing.include_first:
                    continue
                if i == len(sorted_targets) - 1 and not settings.spacing.include_last:
                    continue

                new_value = first_pos + spacing * i
                new_pos = self._set_axis_position(target.position, new_value, settings.axis)

                if settings.preserve_dimension:
                    # Keep other dimensions
                    if settings.axis == DistributionAxis.X:
                        new_pos.y = target.position.y
                        new_pos.z = target.position.z
                    elif settings.axis == DistributionAxis.Y:
                        new_pos.x = target.position.x
                        new_pos.z = target.position.z
                    elif settings.axis == DistributionAxis.Z:
                        new_pos.x = target.position.x
                        new_pos.y = target.position.y

                positions[target.object_id] = new_pos

        elif settings.mode == DistributionMode.EQUAL_GAPS:
            # Calculate total extent of all objects
            total_extent = sum(
                self._get_target_extent(t, settings.axis)
                for t in sorted_targets
            )
            total_range = last_pos - first_pos
            total_gap = total_range - total_extent

            if len(sorted_targets) > 1:
                gap = total_gap / (len(sorted_targets) - 1)
            else:
                gap = 0

            current_pos = first_pos
            for i, target in enumerate(sorted_targets):
                extent = self._get_target_extent(target, settings.axis)

                if i == 0:
                    new_value = first_pos
                else:
                    new_value = current_pos + gap

                new_pos = self._set_axis_position(target.position, new_value, settings.axis)
                positions[target.object_id] = new_pos
                current_pos = new_value + extent

        elif settings.mode in [DistributionMode.BY_CENTERS, DistributionMode.BY_PIVOTS]:
            # Simple even distribution of centers/pivots
            spacing = (last_pos - first_pos) / (len(sorted_targets) - 1) if len(sorted_targets) > 1 else 0

            for i, target in enumerate(sorted_targets):
                new_value = first_pos + spacing * i
                new_pos = self._set_axis_position(target.position, new_value, settings.axis)
                positions[target.object_id] = new_pos

        return positions

    def _calculate_pattern_positions(self) -> dict[str, Vector3]:
        """Calculate positions for pattern distribution."""
        pattern = self._settings.pattern
        positions: dict[str, Vector3] = {}

        if pattern.pattern_type == PatternType.CIRCLE:
            positions = self._distribute_circle()
        elif pattern.pattern_type == PatternType.ARC:
            positions = self._distribute_arc()
        elif pattern.pattern_type == PatternType.GRID:
            positions = self._distribute_grid()
        elif pattern.pattern_type == PatternType.SPIRAL:
            positions = self._distribute_spiral()
        elif pattern.pattern_type == PatternType.LINE:
            positions = self._calculate_linear_positions()
        elif pattern.pattern_type == PatternType.RANDOM:
            positions = self._distribute_random()

        return positions

    def _distribute_circle(self) -> dict[str, Vector3]:
        """Distribute objects in a circle."""
        pattern = self._settings.pattern
        positions: dict[str, Vector3] = {}
        count = len(self._targets)

        if count == 0:
            return positions

        angle_step = (2 * math.pi) / count

        for i, target in enumerate(self._targets):
            angle = pattern.start_angle + angle_step * i
            x = pattern.center.x + math.cos(angle) * pattern.radius
            z = pattern.center.z + math.sin(angle) * pattern.radius
            y = pattern.center.y if not self._settings.preserve_dimension else target.position.y

            positions[target.object_id] = Vector3(x, y, z)

        return positions

    def _distribute_arc(self) -> dict[str, Vector3]:
        """Distribute objects along an arc."""
        pattern = self._settings.pattern
        positions: dict[str, Vector3] = {}
        count = len(self._targets)

        if count == 0:
            return positions

        arc_angle = pattern.end_angle - pattern.start_angle
        angle_step = arc_angle / (count - 1) if count > 1 else 0

        for i, target in enumerate(self._targets):
            angle = pattern.start_angle + angle_step * i
            x = pattern.center.x + math.cos(angle) * pattern.radius
            z = pattern.center.z + math.sin(angle) * pattern.radius
            y = pattern.center.y if not self._settings.preserve_dimension else target.position.y

            positions[target.object_id] = Vector3(x, y, z)

        return positions

    def _distribute_grid(self) -> dict[str, Vector3]:
        """Distribute objects in a grid pattern."""
        pattern = self._settings.pattern
        positions: dict[str, Vector3] = {}

        for i, target in enumerate(self._targets):
            col = i % pattern.columns
            row = i // pattern.columns

            if row >= pattern.rows:
                break

            x = pattern.center.x + col * pattern.column_spacing
            z = pattern.center.z + row * pattern.row_spacing
            y = pattern.center.y if not self._settings.preserve_dimension else target.position.y

            positions[target.object_id] = Vector3(x, y, z)

        return positions

    def _distribute_spiral(self) -> dict[str, Vector3]:
        """Distribute objects along a spiral."""
        pattern = self._settings.pattern
        positions: dict[str, Vector3] = {}
        count = len(self._targets)

        if count == 0:
            return positions

        total_angle = pattern.spiral_turns * 2 * math.pi
        angle_step = total_angle / count if count > 0 else 0
        radius_step = (pattern.spiral_end_radius - pattern.spiral_start_radius) / count if count > 0 else 0

        for i, target in enumerate(self._targets):
            angle = angle_step * i
            radius = pattern.spiral_start_radius + radius_step * i
            x = pattern.center.x + math.cos(angle) * radius
            z = pattern.center.z + math.sin(angle) * radius
            y = pattern.center.y if not self._settings.preserve_dimension else target.position.y

            positions[target.object_id] = Vector3(x, y, z)

        return positions

    def _distribute_random(self) -> dict[str, Vector3]:
        """Distribute objects randomly within a radius."""
        import random
        pattern = self._settings.pattern
        positions: dict[str, Vector3] = {}

        for target in self._targets:
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, pattern.radius)
            x = pattern.center.x + math.cos(angle) * distance
            z = pattern.center.z + math.sin(angle) * distance
            y = pattern.center.y if not self._settings.preserve_dimension else target.position.y

            positions[target.object_id] = Vector3(x, y, z)

        return positions

    @track_changes
    def distribute(self) -> DistributionResult:
        """
        Perform the distribution operation.

        Returns:
            DistributionResult with operation details
        """
        if len(self._targets) < 2:
            return DistributionResult(
                success=False,
                objects_distributed=0,
                error_message="Need at least 2 objects to distribute"
            )

        new_positions = self.preview_distribution()

        if not new_positions:
            return DistributionResult(
                success=False,
                objects_distributed=0,
                error_message="Could not calculate distribution positions"
            )

        original_positions: dict[str, Vector3] = {}

        for target in self._targets:
            if target.object_id not in new_positions:
                continue

            original_positions[target.object_id] = Vector3(
                target.position.x,
                target.position.y,
                target.position.z
            )

            new_pos = new_positions[target.object_id]
            target.position.x = new_pos.x
            target.position.y = new_pos.y
            target.position.z = new_pos.z

        # Track the overall distribution operation on self, not individual targets
        tracker.mark_dirty(self, "_targets",
                         original_positions,
                         new_positions)

        result = DistributionResult(
            success=True,
            objects_distributed=len(new_positions),
            original_positions=original_positions,
            new_positions=new_positions,
        )

        for callback in self._callbacks["on_distribute"]:
            callback(result)

        return result

    # Convenience methods
    @track_changes
    def distribute_horizontally(self, spacing: Optional[float] = None) -> DistributionResult:
        """
        Distribute objects horizontally (along X).

        Args:
            spacing: Optional fixed spacing

        Returns:
            DistributionResult
        """
        self._settings.axis = DistributionAxis.X
        self._settings.mode = DistributionMode.EVEN_SPACING
        self._settings.spacing.fixed_spacing = spacing
        return self.distribute()

    @track_changes
    def distribute_vertically(self, spacing: Optional[float] = None) -> DistributionResult:
        """
        Distribute objects vertically (along Y).

        Args:
            spacing: Optional fixed spacing

        Returns:
            DistributionResult
        """
        self._settings.axis = DistributionAxis.Y
        self._settings.mode = DistributionMode.EVEN_SPACING
        self._settings.spacing.fixed_spacing = spacing
        return self.distribute()

    @track_changes
    def distribute_depth(self, spacing: Optional[float] = None) -> DistributionResult:
        """
        Distribute objects along depth (Z).

        Args:
            spacing: Optional fixed spacing

        Returns:
            DistributionResult
        """
        self._settings.axis = DistributionAxis.Z
        self._settings.mode = DistributionMode.EVEN_SPACING
        self._settings.spacing.fixed_spacing = spacing
        return self.distribute()

    @track_changes
    def distribute_in_circle(
        self,
        center: Vector3,
        radius: float
    ) -> DistributionResult:
        """
        Distribute objects in a circle.

        Args:
            center: Center of the circle
            radius: Radius of the circle

        Returns:
            DistributionResult
        """
        self._settings.mode = DistributionMode.PATTERN
        self._settings.pattern.pattern_type = PatternType.CIRCLE
        self._settings.pattern.center = center
        self._settings.pattern.radius = radius
        return self.distribute()

    @track_changes
    def distribute_in_grid(
        self,
        center: Vector3,
        columns: int,
        rows: int,
        spacing: float = 2.0
    ) -> DistributionResult:
        """
        Distribute objects in a grid.

        Args:
            center: Center of the grid
            columns: Number of columns
            rows: Number of rows
            spacing: Spacing between cells

        Returns:
            DistributionResult
        """
        self._settings.mode = DistributionMode.PATTERN
        self._settings.pattern.pattern_type = PatternType.GRID
        self._settings.pattern.center = center
        self._settings.pattern.columns = columns
        self._settings.pattern.rows = rows
        self._settings.pattern.column_spacing = spacing
        self._settings.pattern.row_spacing = spacing
        return self.distribute()

    @track_changes
    def equalize_spacing(self, axis: DistributionAxis = DistributionAxis.X) -> DistributionResult:
        """
        Equalize spacing between objects.

        Args:
            axis: Axis to equalize along

        Returns:
            DistributionResult
        """
        self._settings.axis = axis
        self._settings.mode = DistributionMode.EQUAL_GAPS
        return self.distribute()


__all__ = [
    "DistributionMode",
    "DistributionAxis",
    "PatternType",
    "SpacingSettings",
    "PatternSettings",
    "DistributionSettings",
    "DistributionResult",
    "DistributionTool",
]
