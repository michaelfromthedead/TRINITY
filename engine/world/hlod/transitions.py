"""
HLOD Transition Handling.

Implements smooth LOD transitions using various techniques including
instant switching, dithered transitions, alpha crossfade, and vertex morphing.

References:
- WORLD_CONTEXT.md Section 7 HLOD System
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from engine.world.hlod.constants import (
    TransitionConstantsConfig as _HLODTransitionConstants,
)
from .generator import AABB, Vec3


# =============================================================================
# TRANSITION CONSTANTS
# =============================================================================


class TransitionConstants:
    """Constants for LOD transitions."""
    # Epsilon for floating point comparisons
    EPSILON: float = 1e-6

    # Default transition settings
    DEFAULT_TRANSITION_RANGE: float = 50.0
    DEFAULT_DITHER_SCALE: float = 1.0
    DEFAULT_MORPH_SPEED: float = 5.0

    # Screen space error defaults
    DEFAULT_ERROR_THRESHOLD: float = 2.0  # pixels
    DEFAULT_MIN_SCREEN_SIZE: float = 1.0  # pixels

    # Dither pattern size
    DITHER_PATTERN_SIZE: int = 4


# =============================================================================
# TRANSITION MODE ENUM
# =============================================================================


class TransitionMode(Enum):
    """Transition mode between LOD levels."""
    INSTANT = auto()       # Pop/snap transition (no blending)
    DITHERED = auto()      # Alpha dither pattern
    CROSSFADE = auto()     # Alpha blending between LODs
    MORPHING = auto()      # Vertex interpolation


# =============================================================================
# TRANSITION SETTINGS
# =============================================================================


@dataclass
class TransitionSettings:
    """Settings for LOD transitions."""
    mode: TransitionMode = TransitionMode.DITHERED
    transition_range: float = TransitionConstants.DEFAULT_TRANSITION_RANGE
    dither_scale: float = TransitionConstants.DEFAULT_DITHER_SCALE
    morph_speed: float = TransitionConstants.DEFAULT_MORPH_SPEED
    use_hysteresis: bool = True
    hysteresis_factor: float = 0.1  # 10% hysteresis band

    def __post_init__(self) -> None:
        """Validate settings."""
        if self.transition_range < 0:
            raise ValueError("transition_range must be non-negative")
        if self.dither_scale <= 0:
            raise ValueError("dither_scale must be positive")
        if self.morph_speed <= 0:
            raise ValueError("morph_speed must be positive")
        if not 0.0 <= self.hysteresis_factor <= 1.0:
            raise ValueError("hysteresis_factor must be in [0, 1]")


# =============================================================================
# LOD TRANSITION STATE
# =============================================================================


class TransitionState(Enum):
    """State of a LOD transition."""
    STABLE = auto()        # No transition in progress
    TRANSITIONING = auto()  # Transition in progress
    COMPLETE = auto()       # Transition just completed


@dataclass
class LODTransition:
    """Tracks the state of a single LOD transition."""
    from_lod: int = 0
    to_lod: int = 0
    blend_factor: float = 0.0  # 0.0 = from_lod, 1.0 = to_lod
    state: TransitionState = TransitionState.STABLE
    _start_distance: float = 0.0
    _end_distance: float = 0.0

    @property
    def is_active(self) -> bool:
        """Check if transition is in progress."""
        return self.state == TransitionState.TRANSITIONING

    @property
    def is_complete(self) -> bool:
        """Check if transition just completed."""
        return self.state == TransitionState.COMPLETE

    @property
    def current_lod(self) -> int:
        """Get the current primary LOD (based on blend factor)."""
        return self.to_lod if self.blend_factor > 0.5 else self.from_lod

    def start(
        self,
        from_lod: int,
        to_lod: int,
        start_distance: float,
        end_distance: float,
    ) -> None:
        """Start a new transition."""
        self.from_lod = from_lod
        self.to_lod = to_lod
        self.blend_factor = 0.0
        self.state = TransitionState.TRANSITIONING
        self._start_distance = start_distance
        self._end_distance = end_distance

    def update(
        self,
        camera_distance: float,
        thresholds: List[float],
    ) -> float:
        """
        Update transition based on camera distance.

        Args:
            camera_distance: Current distance from camera
            thresholds: List of LOD distance thresholds

        Returns:
            Current blend factor (0-1)
        """
        if self.state != TransitionState.TRANSITIONING:
            return self.blend_factor

        # Calculate progress through transition range
        if abs(self._end_distance - self._start_distance) < TransitionConstants.EPSILON:
            self.blend_factor = 1.0
        else:
            # Normalize distance to blend factor
            t = (camera_distance - self._start_distance) / (
                self._end_distance - self._start_distance
            )
            self.blend_factor = max(0.0, min(1.0, t))

        # Check for completion
        if self.blend_factor >= 1.0:
            self.complete()
        elif self.blend_factor <= 0.0:
            self.cancel()

        return self.blend_factor

    def complete(self) -> None:
        """Mark transition as complete."""
        self.blend_factor = 1.0
        self.state = TransitionState.COMPLETE

    def cancel(self) -> None:
        """Cancel transition (return to from_lod)."""
        self.blend_factor = 0.0
        self.state = TransitionState.STABLE

    def reset(self) -> None:
        """Reset to stable state at current LOD."""
        self.from_lod = self.to_lod
        self.blend_factor = 0.0
        self.state = TransitionState.STABLE


# =============================================================================
# TRANSITION CALCULATOR
# =============================================================================


class TransitionCalculator:
    """Calculates transition blend factors and dither patterns."""

    def __init__(self, settings: Optional[TransitionSettings] = None) -> None:
        self._settings = settings or TransitionSettings()
        self._dither_pattern = self._generate_dither_pattern()

    @property
    def settings(self) -> TransitionSettings:
        return self._settings

    @settings.setter
    def settings(self, value: TransitionSettings) -> None:
        self._settings = value
        self._dither_pattern = self._generate_dither_pattern()

    def calculate_blend(
        self,
        distance: float,
        near_threshold: float,
        far_threshold: float,
    ) -> float:
        """
        Calculate blend factor for transition between two LODs.

        Args:
            distance: Current camera distance
            near_threshold: Near LOD distance threshold
            far_threshold: Far LOD distance threshold

        Returns:
            Blend factor from 0.0 (near) to 1.0 (far)
        """
        if self._settings.mode == TransitionMode.INSTANT:
            # Instant transition at midpoint
            midpoint = (near_threshold + far_threshold) / 2.0
            return 0.0 if distance < midpoint else 1.0

        # Calculate transition zone
        transition_start = far_threshold - self._settings.transition_range
        transition_end = far_threshold

        if distance <= transition_start:
            return 0.0
        elif distance >= transition_end:
            return 1.0
        else:
            # Linear interpolation through transition zone
            t = (distance - transition_start) / self._settings.transition_range
            return self._smooth_step(t)

    def get_dither_pattern(
        self,
        blend_factor: float,
        pixel_x: int,
        pixel_y: int,
    ) -> bool:
        """
        Get dither pattern value for a pixel.

        Args:
            blend_factor: Current blend factor (0-1)
            pixel_x: Pixel X coordinate
            pixel_y: Pixel Y coordinate

        Returns:
            True if pixel should show higher LOD, False for lower LOD
        """
        if self._settings.mode != TransitionMode.DITHERED:
            return blend_factor > 0.5

        # Get pattern value for this pixel
        px = pixel_x % TransitionConstants.DITHER_PATTERN_SIZE
        py = pixel_y % TransitionConstants.DITHER_PATTERN_SIZE

        threshold = self._dither_pattern[py][px]

        # Scale threshold by dither scale
        scaled_threshold = threshold * self._settings.dither_scale

        return blend_factor > scaled_threshold

    def get_morph_factor(
        self,
        blend_factor: float,
        vertex_index: int,
    ) -> float:
        """
        Get vertex morph factor for morphing transitions.

        Args:
            blend_factor: Current blend factor (0-1)
            vertex_index: Index of the vertex

        Returns:
            Morph factor for vertex interpolation
        """
        if self._settings.mode != TransitionMode.MORPHING:
            return blend_factor

        # Apply smooth step for nicer morphing
        return self._smooth_step(blend_factor)

    def _generate_dither_pattern(self) -> List[List[float]]:
        """Generate Bayer dither pattern."""
        size = TransitionConstants.DITHER_PATTERN_SIZE

        # 4x4 Bayer matrix
        if size == 4:
            pattern = [
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5],
            ]
        else:
            # Generate pattern for other sizes
            pattern = [[0.0] * size for _ in range(size)]
            for y in range(size):
                for x in range(size):
                    pattern[y][x] = float((x + y * 2) % (size * size))

        # Normalize to 0-1
        max_val = size * size
        return [[v / max_val for v in row] for row in pattern]

    def _smooth_step(self, t: float) -> float:
        """Smooth step function for nicer transitions."""
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)


# =============================================================================
# SCREEN SPACE ERROR CALCULATOR
# =============================================================================


class ScreenSpaceError:
    """Calculates screen space error for LOD selection."""

    def __init__(
        self,
        error_threshold: float = TransitionConstants.DEFAULT_ERROR_THRESHOLD,
    ) -> None:
        """
        Initialize screen space error calculator.

        Args:
            error_threshold: Maximum allowed error in pixels
        """
        self._error_threshold = error_threshold

    @property
    def error_threshold(self) -> float:
        return self._error_threshold

    @error_threshold.setter
    def error_threshold(self, value: float) -> None:
        if value <= 0:
            raise ValueError("error_threshold must be positive")
        self._error_threshold = value

    def calculate_error(
        self,
        bounds: AABB,
        camera_position: Vec3,
        fov: float,
        screen_height: int,
    ) -> float:
        """
        Calculate screen space error in pixels.

        Args:
            bounds: Object bounding box
            camera_position: Camera world position
            fov: Vertical field of view in radians
            screen_height: Screen height in pixels

        Returns:
            Estimated screen space error in pixels
        """
        # Calculate distance from camera to bounds center
        center = bounds.center
        distance = camera_position.distance_to(center)

        if distance < TransitionConstants.EPSILON:
            return float("inf")  # Very close, maximum error

        # Calculate screen size based on bounds extent
        extents = bounds.extents
        world_size = max(extents.x, extents.y, extents.z) * 2.0

        # Project to screen space
        half_fov_tan = math.tan(fov * 0.5)
        projected_size = (world_size / distance) / half_fov_tan
        screen_size = projected_size * screen_height * 0.5

        return screen_size

    def calculate_error_from_radius(
        self,
        radius: float,
        distance: float,
        fov: float,
        screen_height: int,
    ) -> float:
        """
        Calculate screen space error from bounding radius.

        Args:
            radius: Bounding sphere radius
            distance: Distance from camera
            fov: Vertical field of view in radians
            screen_height: Screen height in pixels

        Returns:
            Estimated screen space error in pixels
        """
        if distance < TransitionConstants.EPSILON:
            return float("inf")

        half_fov_tan = math.tan(fov * 0.5)
        projected_size = (radius * 2.0 / distance) / half_fov_tan
        screen_size = projected_size * screen_height * 0.5

        return screen_size

    def get_lod_for_error(
        self,
        error: float,
        thresholds: List[float],
    ) -> int:
        """
        Determine LOD level based on screen space error.

        Args:
            error: Screen space error in pixels
            thresholds: Error thresholds for each LOD level

        Returns:
            Recommended LOD index
        """
        for i, threshold in enumerate(thresholds):
            if error >= threshold:
                return i

        return len(thresholds)

    def compute_optimal_distance(
        self,
        bounds: AABB,
        target_error: float,
        fov: float,
        screen_height: int,
    ) -> float:
        """
        Compute distance at which object has target screen space error.

        Args:
            bounds: Object bounding box
            target_error: Target error in pixels
            fov: Vertical field of view in radians
            screen_height: Screen height in pixels

        Returns:
            Distance at which target error is achieved
        """
        if target_error <= 0:
            return float("inf")

        extents = bounds.extents
        world_size = max(extents.x, extents.y, extents.z) * 2.0

        half_fov_tan = math.tan(fov * 0.5)
        target_projected = (target_error * 2.0) / screen_height
        distance = world_size / (target_projected * half_fov_tan)

        return max(0.0, distance)


# =============================================================================
# HLOD TRANSITION MANAGER
# =============================================================================


class HLODTransitionManager:
    """
    Manages LOD transitions for multiple HLOD cells.
    """

    def __init__(self, settings: Optional[TransitionSettings] = None) -> None:
        """
        Initialize transition manager.

        Args:
            settings: Transition settings
        """
        self._settings = settings or TransitionSettings()
        self._calculator = TransitionCalculator(self._settings)
        self._active_transitions: Dict[Tuple[int, int], LODTransition] = {}
        self._current_lods: Dict[Tuple[int, int], int] = {}

    @property
    def settings(self) -> TransitionSettings:
        return self._settings

    @settings.setter
    def settings(self, value: TransitionSettings) -> None:
        self._settings = value
        self._calculator.settings = value

    @property
    def calculator(self) -> TransitionCalculator:
        return self._calculator

    @property
    def active_transitions(self) -> Dict[Tuple[int, int], LODTransition]:
        return self._active_transitions

    def update(
        self,
        camera_position: Vec3,
        cell_bounds: Dict[Tuple[int, int], AABB],
        lod_thresholds: List[float],
    ) -> None:
        """
        Update all transitions based on camera position.

        Args:
            camera_position: Current camera world position
            cell_bounds: Bounds for each cell
            lod_thresholds: Distance thresholds for each LOD level
        """
        for cell_id, bounds in cell_bounds.items():
            distance = camera_position.distance_to(bounds.center)

            # Determine target LOD for this distance
            target_lod = self._get_lod_for_distance(distance, lod_thresholds)

            # Get or create transition state
            if cell_id in self._active_transitions:
                transition = self._active_transitions[cell_id]
            else:
                transition = LODTransition()
                self._active_transitions[cell_id] = transition

            # Get current LOD
            current_lod = self._current_lods.get(cell_id, 0)

            # Check if we need to start a new transition
            if target_lod != current_lod and not transition.is_active:
                # Apply hysteresis
                if self._settings.use_hysteresis:
                    hysteresis = lod_thresholds[min(target_lod, len(lod_thresholds) - 1)]
                    hysteresis *= self._settings.hysteresis_factor

                    # Only transition if we've moved enough
                    if abs(distance - lod_thresholds[current_lod]) < hysteresis:
                        continue

                # Start transition
                start_threshold = lod_thresholds[current_lod] if current_lod < len(lod_thresholds) else 0.0
                end_threshold = lod_thresholds[target_lod] if target_lod < len(lod_thresholds) else float("inf")

                transition.start(current_lod, target_lod, start_threshold, end_threshold)

            # Update active transition
            if transition.is_active:
                transition.update(distance, lod_thresholds)

            # Handle completed transitions
            if transition.is_complete:
                self._current_lods[cell_id] = transition.to_lod
                transition.reset()

    def _get_lod_for_distance(
        self,
        distance: float,
        thresholds: List[float],
    ) -> int:
        """Get LOD level for a given distance."""
        for i, threshold in enumerate(thresholds):
            if distance < threshold:
                return i
        return len(thresholds)

    def get_transition_state(
        self,
        cell_id: Tuple[int, int],
    ) -> Tuple[int, int, float]:
        """
        Get transition state for a cell.

        Args:
            cell_id: Cell identifier

        Returns:
            Tuple of (from_lod, to_lod, blend_factor)
        """
        if cell_id in self._active_transitions:
            transition = self._active_transitions[cell_id]
            return (transition.from_lod, transition.to_lod, transition.blend_factor)

        current_lod = self._current_lods.get(cell_id, 0)
        return (current_lod, current_lod, 0.0)

    def start_transition(
        self,
        cell_id: Tuple[int, int],
        from_lod: int,
        to_lod: int,
        start_distance: float = 0.0,
        end_distance: float = 100.0,
    ) -> None:
        """
        Manually start a transition for a cell.

        Args:
            cell_id: Cell identifier
            from_lod: Starting LOD level
            to_lod: Target LOD level
            start_distance: Distance at which transition starts
            end_distance: Distance at which transition completes
        """
        transition = LODTransition()
        transition.start(from_lod, to_lod, start_distance, end_distance)
        self._active_transitions[cell_id] = transition
        self._current_lods[cell_id] = from_lod

    def complete_transition(self, cell_id: Tuple[int, int]) -> None:
        """
        Force completion of a transition.

        Args:
            cell_id: Cell identifier
        """
        if cell_id in self._active_transitions:
            transition = self._active_transitions[cell_id]
            transition.complete()
            self._current_lods[cell_id] = transition.to_lod
            transition.reset()

    def cancel_transition(self, cell_id: Tuple[int, int]) -> None:
        """
        Cancel a transition and return to starting LOD.

        Args:
            cell_id: Cell identifier
        """
        if cell_id in self._active_transitions:
            transition = self._active_transitions[cell_id]
            transition.cancel()

    def is_transitioning(self, cell_id: Tuple[int, int]) -> bool:
        """Check if a cell is currently transitioning."""
        if cell_id in self._active_transitions:
            return self._active_transitions[cell_id].is_active
        return False

    def get_current_lod(self, cell_id: Tuple[int, int]) -> int:
        """Get the current stable LOD for a cell."""
        return self._current_lods.get(cell_id, 0)

    def clear_transitions(self) -> None:
        """Clear all active transitions."""
        self._active_transitions.clear()

    def remove_cell(self, cell_id: Tuple[int, int]) -> None:
        """Remove a cell from tracking."""
        self._active_transitions.pop(cell_id, None)
        self._current_lods.pop(cell_id, None)


# =============================================================================
# HLOD VISIBILITY SYSTEM
# =============================================================================


@dataclass
class VisibilityResult:
    """Result of visibility determination for a cell."""
    cell_id: Tuple[int, int]
    is_visible: bool
    lod_index: int
    blend_factor: float
    screen_error: float
    distance: float


class HLODVisibilitySystem:
    """
    Determines visibility and LOD for HLOD cells.
    """

    def __init__(
        self,
        error_threshold: float = TransitionConstants.DEFAULT_ERROR_THRESHOLD,
    ) -> None:
        """
        Initialize visibility system.

        Args:
            error_threshold: Screen space error threshold for LOD selection
        """
        self._cells: Dict[Tuple[int, int], AABB] = {}
        self._camera_position: Vec3 = Vec3()
        self._camera_forward: Vec3 = Vec3(0.0, 0.0, -1.0)
        self._fov: float = math.radians(60.0)
        self._screen_height: int = 1080
        self._error_calculator = ScreenSpaceError(error_threshold)
        self._transition_manager = HLODTransitionManager()
        self._lod_thresholds: List[float] = [500.0, 1000.0, 2000.0, 4000.0]
        self._max_distance: float = 10000.0

    @property
    def error_threshold(self) -> float:
        return self._error_calculator.error_threshold

    @error_threshold.setter
    def error_threshold(self, value: float) -> None:
        self._error_calculator.error_threshold = value

    @property
    def lod_thresholds(self) -> List[float]:
        return self._lod_thresholds

    @lod_thresholds.setter
    def lod_thresholds(self, value: List[float]) -> None:
        self._lod_thresholds = sorted(value)

    @property
    def max_distance(self) -> float:
        return self._max_distance

    @max_distance.setter
    def max_distance(self, value: float) -> None:
        self._max_distance = max(0.0, value)

    def add_cell(self, cell_id: Tuple[int, int], bounds: AABB) -> None:
        """Add a cell to track."""
        self._cells[cell_id] = bounds

    def remove_cell(self, cell_id: Tuple[int, int]) -> None:
        """Remove a cell from tracking."""
        self._cells.pop(cell_id, None)
        self._transition_manager.remove_cell(cell_id)

    def update(
        self,
        camera_position: Vec3,
        camera_forward: Optional[Vec3] = None,
        fov: Optional[float] = None,
        screen_height: Optional[int] = None,
    ) -> None:
        """
        Update camera parameters.

        Args:
            camera_position: Camera world position
            camera_forward: Camera forward direction (optional)
            fov: Vertical field of view in radians (optional)
            screen_height: Screen height in pixels (optional)
        """
        self._camera_position = camera_position

        if camera_forward is not None:
            self._camera_forward = camera_forward.normalized()
        if fov is not None:
            self._fov = fov
        if screen_height is not None:
            self._screen_height = screen_height

        # Update transitions
        self._transition_manager.update(
            camera_position,
            self._cells,
            self._lod_thresholds,
        )

    def get_visible_cells(self) -> List[VisibilityResult]:
        """
        Get list of visible cells with LOD assignments.

        Returns:
            List of visibility results for visible cells
        """
        results: List[VisibilityResult] = []

        for cell_id, bounds in self._cells.items():
            # Calculate distance
            distance = self._camera_position.distance_to(bounds.center)

            # Distance culling
            if distance > self._max_distance:
                continue

            # Frustum culling (simplified - just check if in front of camera)
            to_cell = bounds.center - self._camera_position
            forward_dist = to_cell.dot(self._camera_forward)

            if forward_dist < -bounds.extents.length():
                continue  # Behind camera

            # Calculate screen space error
            error = self._error_calculator.calculate_error(
                bounds,
                self._camera_position,
                self._fov,
                self._screen_height,
            )

            # Get LOD and transition state
            from_lod, to_lod, blend = self._transition_manager.get_transition_state(cell_id)

            results.append(
                VisibilityResult(
                    cell_id=cell_id,
                    is_visible=True,
                    lod_index=to_lod if blend > 0.5 else from_lod,
                    blend_factor=blend,
                    screen_error=error,
                    distance=distance,
                )
            )

        # Sort by distance for front-to-back rendering
        results.sort(key=lambda r: r.distance)

        return results

    def get_lod_assignments(self) -> Dict[Tuple[int, int], int]:
        """
        Get LOD assignments for all visible cells.

        Returns:
            Dictionary mapping cell_id to LOD index
        """
        visible = self.get_visible_cells()
        return {r.cell_id: r.lod_index for r in visible}

    def get_cell_visibility(self, cell_id: Tuple[int, int]) -> Optional[VisibilityResult]:
        """Get visibility result for a specific cell."""
        if cell_id not in self._cells:
            return None

        bounds = self._cells[cell_id]
        distance = self._camera_position.distance_to(bounds.center)

        # Check visibility
        if distance > self._max_distance:
            return VisibilityResult(
                cell_id=cell_id,
                is_visible=False,
                lod_index=0,
                blend_factor=0.0,
                screen_error=0.0,
                distance=distance,
            )

        to_cell = bounds.center - self._camera_position
        forward_dist = to_cell.dot(self._camera_forward)

        if forward_dist < -bounds.extents.length():
            return VisibilityResult(
                cell_id=cell_id,
                is_visible=False,
                lod_index=0,
                blend_factor=0.0,
                screen_error=0.0,
                distance=distance,
            )

        error = self._error_calculator.calculate_error(
            bounds,
            self._camera_position,
            self._fov,
            self._screen_height,
        )

        from_lod, to_lod, blend = self._transition_manager.get_transition_state(cell_id)

        return VisibilityResult(
            cell_id=cell_id,
            is_visible=True,
            lod_index=to_lod if blend > 0.5 else from_lod,
            blend_factor=blend,
            screen_error=error,
            distance=distance,
        )

    def configure_transitions(self, settings: TransitionSettings) -> None:
        """Configure transition settings."""
        self._transition_manager.settings = settings


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Constants
    "TransitionConstants",
    # Enums
    "TransitionMode",
    "TransitionState",
    # Settings
    "TransitionSettings",
    # Core classes
    "LODTransition",
    "TransitionCalculator",
    "ScreenSpaceError",
    "HLODTransitionManager",
    # Visibility
    "VisibilityResult",
    "HLODVisibilitySystem",
]
