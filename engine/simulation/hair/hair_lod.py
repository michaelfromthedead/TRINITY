"""
Hair Level of Detail (LOD) system.

Manages hair complexity based on distance from camera:
- Full guide hairs at close range
- Reduced guide count at medium range
- Shell-based rendering at far range
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import (
    LOD_DISTANCE_HIGH,
    LOD_DISTANCE_LOW,
    LOD_DISTANCE_MEDIUM,
    LOD_DISTANCE_SHELL,
    LOD_GUIDE_FACTOR_HIGH,
    LOD_GUIDE_FACTOR_LOW,
    LOD_GUIDE_FACTOR_MEDIUM,
    LOD_GUIDE_FACTOR_SHELL,
    LOD_INTERPOLATION_OFFSET,
    LOD_SEGMENT_FACTOR_HIGH,
    LOD_SEGMENT_FACTOR_LOW,
    LOD_SEGMENT_FACTOR_MEDIUM,
)
from .hair_simulation import GuideHair, HairStrand, InterpolatedHair


class HairLODLevel(Enum):
    """LOD levels for hair rendering."""

    HIGH = auto()  # Full quality
    MEDIUM = auto()  # Reduced guides
    LOW = auto()  # Minimal guides
    SHELL = auto()  # Shell-based fallback


@dataclass
class LODSettings:
    """Configuration for LOD transitions."""

    # Distance thresholds
    distance_high: float = LOD_DISTANCE_HIGH
    distance_medium: float = LOD_DISTANCE_MEDIUM
    distance_low: float = LOD_DISTANCE_LOW
    distance_shell: float = LOD_DISTANCE_SHELL

    # Guide hair reduction factors
    guide_factor_high: float = LOD_GUIDE_FACTOR_HIGH
    guide_factor_medium: float = LOD_GUIDE_FACTOR_MEDIUM
    guide_factor_low: float = LOD_GUIDE_FACTOR_LOW
    guide_factor_shell: float = LOD_GUIDE_FACTOR_SHELL

    # Segment reduction factors
    segment_factor_high: float = LOD_SEGMENT_FACTOR_HIGH
    segment_factor_medium: float = LOD_SEGMENT_FACTOR_MEDIUM
    segment_factor_low: float = LOD_SEGMENT_FACTOR_LOW

    # Hysteresis to prevent LOD popping
    hysteresis: float = 0.1


@dataclass
class LODState:
    """Current LOD state for a hair system."""

    level: HairLODLevel = HairLODLevel.HIGH
    distance: float = 0.0
    guide_factor: float = 1.0
    segment_factor: float = 1.0
    blend_factor: float = 0.0  # For smooth transitions


class HairLODSystem:
    """
    Manages hair LOD transitions based on camera distance.
    """

    def __init__(
        self,
        settings: Optional[LODSettings] = None,
    ) -> None:
        """
        Initialize the LOD system.

        Args:
            settings: LOD configuration settings
        """
        self.settings = settings or LODSettings()
        self._state = LODState()

        # Full guide hair list
        self._all_guides: List[GuideHair] = []

        # Active guides (reduced based on LOD)
        self._active_guides: List[GuideHair] = []

        # LOD-specific hair sets (precomputed)
        self._guides_high: List[GuideHair] = []
        self._guides_medium: List[GuideHair] = []
        self._guides_low: List[GuideHair] = []

        # Shell rendering data
        self._shell_layers: int = 4
        self._shell_data: Optional[NDArray[np.float32]] = None

    @property
    def current_level(self) -> HairLODLevel:
        """Get current LOD level."""
        return self._state.level

    @property
    def active_guides(self) -> List[GuideHair]:
        """Get currently active guide hairs."""
        return self._active_guides

    @property
    def guide_count(self) -> int:
        """Get number of active guides."""
        return len(self._active_guides)

    def initialize(self, guides: List[GuideHair]) -> None:
        """
        Initialize LOD system with guide hairs.

        Pre-computes reduced guide sets for each LOD level.

        Args:
            guides: Full list of guide hairs
        """
        self._all_guides = guides.copy()

        # Create LOD sets
        num_guides = len(guides)

        # High LOD - all guides
        self._guides_high = guides.copy()

        # Medium LOD - reduce by factor
        num_medium = max(1, int(num_guides * self.settings.guide_factor_medium))
        self._guides_medium = self._select_guides(guides, num_medium)

        # Low LOD - minimal guides
        num_low = max(1, int(num_guides * self.settings.guide_factor_low))
        self._guides_low = self._select_guides(guides, num_low)

        # Initialize to high quality
        self._active_guides = self._guides_high
        self._state.level = HairLODLevel.HIGH
        self._state.guide_factor = 1.0
        self._state.segment_factor = 1.0

    def _select_guides(
        self,
        guides: List[GuideHair],
        count: int,
    ) -> List[GuideHair]:
        """
        Select a subset of guides with good spatial distribution.

        Args:
            guides: Full guide list
            count: Number to select

        Returns:
            Selected guides
        """
        if count >= len(guides):
            return guides.copy()

        if count <= 0:
            return []

        # Simple uniform sampling by index
        # A better approach would use spatial clustering
        step = len(guides) / count
        selected = []

        for i in range(count):
            idx = int(i * step)
            if idx < len(guides):
                selected.append(guides[idx])

        return selected

    def update(
        self,
        camera_position: NDArray[np.float32],
        hair_center: NDArray[np.float32],
    ) -> bool:
        """
        Update LOD based on camera distance.

        Args:
            camera_position: Camera world position
            hair_center: Center of hair system (e.g., head position)

        Returns:
            True if LOD level changed
        """
        distance = float(np.linalg.norm(camera_position - hair_center))
        self._state.distance = distance

        # Determine target LOD level
        prev_level = self._state.level
        hysteresis = self.settings.hysteresis

        # Check transitions with hysteresis
        if self._state.level == HairLODLevel.HIGH:
            if distance > self.settings.distance_high + hysteresis:
                self._state.level = HairLODLevel.MEDIUM
        elif self._state.level == HairLODLevel.MEDIUM:
            if distance < self.settings.distance_high - hysteresis:
                self._state.level = HairLODLevel.HIGH
            elif distance > self.settings.distance_medium + hysteresis:
                self._state.level = HairLODLevel.LOW
        elif self._state.level == HairLODLevel.LOW:
            if distance < self.settings.distance_medium - hysteresis:
                self._state.level = HairLODLevel.MEDIUM
            elif distance > self.settings.distance_low + hysteresis:
                self._state.level = HairLODLevel.SHELL
        elif self._state.level == HairLODLevel.SHELL:
            if distance < self.settings.distance_low - hysteresis:
                self._state.level = HairLODLevel.LOW

        # Update active guides if level changed
        if self._state.level != prev_level:
            self._apply_lod_level()
            return True

        return False

    def _apply_lod_level(self) -> None:
        """Apply the current LOD level settings."""
        if self._state.level == HairLODLevel.HIGH:
            self._active_guides = self._guides_high
            self._state.guide_factor = self.settings.guide_factor_high
            self._state.segment_factor = self.settings.segment_factor_high
        elif self._state.level == HairLODLevel.MEDIUM:
            self._active_guides = self._guides_medium
            self._state.guide_factor = self.settings.guide_factor_medium
            self._state.segment_factor = self.settings.segment_factor_medium
        elif self._state.level == HairLODLevel.LOW:
            self._active_guides = self._guides_low
            self._state.guide_factor = self.settings.guide_factor_low
            self._state.segment_factor = self.settings.segment_factor_low
        elif self._state.level == HairLODLevel.SHELL:
            self._active_guides = []
            self._state.guide_factor = 0.0
            self._state.segment_factor = 0.0

    def reduce_guide_count(
        self,
        guides: List[GuideHair],
        target_count: int,
    ) -> List[GuideHair]:
        """
        Reduce guide hair count while maintaining distribution.

        Args:
            guides: Input guide hairs
            target_count: Target number of guides

        Returns:
            Reduced list of guides
        """
        return self._select_guides(guides, target_count)

    def get_interpolation_weights(
        self,
        position: NDArray[np.float32],
        k_nearest: int = 3,
    ) -> Tuple[List[int], NDArray[np.float32]]:
        """
        Get interpolation weights for rendering a hair at position.

        Args:
            position: Position to interpolate at
            k_nearest: Number of nearest guides to use

        Returns:
            Tuple of (guide indices, weights)
        """
        if not self._active_guides:
            return [], np.array([], dtype=np.float32)

        k = min(k_nearest, len(self._active_guides))

        # Find k nearest guides
        distances = []
        for i, guide in enumerate(self._active_guides):
            dist = float(np.linalg.norm(position - guide.root_position))
            distances.append((dist, i))

        distances.sort(key=lambda x: x[0])
        nearest = distances[:k]

        # Compute inverse distance weights
        indices = [n[1] for n in nearest]
        inv_dists = [1.0 / (n[0] + 1e-6) for n in nearest]
        total = sum(inv_dists)

        weights = np.array([d / total for d in inv_dists], dtype=np.float32)

        return indices, weights

    def get_segment_count(self, base_segments: int) -> int:
        """
        Get segment count for current LOD.

        Args:
            base_segments: Full quality segment count

        Returns:
            Reduced segment count based on LOD
        """
        return max(2, int(base_segments * self._state.segment_factor))

    def prepare_shell_data(
        self,
        guides: List[GuideHair],
        num_layers: int = 4,
    ) -> None:
        """
        Prepare shell rendering data for far LOD.

        Args:
            guides: Guide hairs for computing shell
            num_layers: Number of shell layers
        """
        self._shell_layers = num_layers

        # Compute average hair direction and length
        if not guides:
            self._shell_data = None
            return

        # For shell rendering, we need:
        # - Per-vertex shell offset direction
        # - Per-layer thickness
        # This is typically done in the shader, but we prepare parameters here

        avg_length = sum(g.length for g in guides) / len(guides)
        layer_spacing = avg_length / num_layers

        self._shell_data = np.array(
            [layer_spacing * (i + 1) for i in range(num_layers)],
            dtype=np.float32,
        )

    def get_shell_offsets(self) -> Optional[NDArray[np.float32]]:
        """
        Get shell layer offsets for rendering.

        Returns:
            Array of per-layer offsets, or None if not using shell rendering
        """
        if self._state.level != HairLODLevel.SHELL:
            return None
        return self._shell_data

    def is_shell_mode(self) -> bool:
        """Check if currently in shell rendering mode."""
        return self._state.level == HairLODLevel.SHELL


@dataclass
class LODTransition:
    """Handles smooth transitions between LOD levels."""

    duration: float = 0.5  # Transition time in seconds
    _progress: float = 1.0
    _source_level: HairLODLevel = HairLODLevel.HIGH
    _target_level: HairLODLevel = HairLODLevel.HIGH

    def start_transition(
        self,
        source: HairLODLevel,
        target: HairLODLevel,
    ) -> None:
        """Start a LOD transition."""
        self._source_level = source
        self._target_level = target
        self._progress = 0.0

    def update(self, dt: float) -> bool:
        """
        Update the transition.

        Args:
            dt: Delta time

        Returns:
            True if transition is complete
        """
        if self._progress >= 1.0:
            return True

        self._progress = min(1.0, self._progress + dt / self.duration)
        return self._progress >= 1.0

    @property
    def blend_factor(self) -> float:
        """Get current blend factor (0 = source, 1 = target)."""
        # Smooth step for nicer transition
        t = self._progress
        return t * t * (3.0 - 2.0 * t)

    @property
    def is_transitioning(self) -> bool:
        """Check if currently transitioning."""
        return self._progress < 1.0


def create_lod_interpolated_hairs(
    lod_system: HairLODSystem,
    count_per_guide: int = 10,
) -> List[InterpolatedHair]:
    """
    Create interpolated hairs for the current LOD level.

    Args:
        lod_system: The LOD system
        count_per_guide: Interpolated hairs per guide

    Returns:
        List of interpolated hairs
    """
    from .hair_simulation import HairControlPoint, InterpolatedHair

    interpolated = []
    active_guides = lod_system.active_guides

    for guide in active_guides:
        for _ in range(count_per_guide):
            # Random offset from guide (using config constant for tunable spacing)
            offset = np.random.randn(3).astype(np.float32) * LOD_INTERPOLATION_OFFSET

            # Create control points
            cps = []
            for cp in guide.control_points:
                new_cp = HairControlPoint(
                    position=cp.position + offset,
                    prev_position=cp.prev_position + offset,
                    rest_position=cp.rest_position.copy(),
                    inv_mass=cp.inv_mass,
                )
                cps.append(new_cp)

            hair = InterpolatedHair(
                control_points=cps,
                rest_lengths=guide.rest_lengths.copy(),
                root_position=guide.root_position + offset,
                root_normal=guide.root_normal.copy(),
                thickness=guide.thickness,
                is_guide=False,
                guide_hair_indices=[guide.index],
                interpolation_weights=np.array([1.0], dtype=np.float32),
            )
            interpolated.append(hair)

    return interpolated
