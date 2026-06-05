"""Material Level-of-Detail (LOD) selection system.

This module implements distance-based material LOD selection, enabling
automatic quality reduction for distant objects to optimize rendering
performance.

The LOD system provides:
1. LODConfig: Configuration for LOD distance thresholds and quality mapping
2. MaterialLODSelector: Distance-based variant selection with cross-fade support

LOD levels:
- LOD 0: Full quality (HIGH) - closest to camera
- LOD 1: Reduced quality (MEDIUM)
- LOD 2: Low quality (LOW)
- LOD 3: Unlit fallback (LOW with minimal features)

Cross-fade transitions enable smooth LOD switching without visual pops.

Task: T-MAT-5.6 Material LOD System
Gap: S3-G15 (MEDIUM)
Dependency: T-MAT-5.1 (MaterialVariantRegistry)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from trinity.materials.variants import (
    MaterialDomain,
    BlendMode,
    QualityTier,
    VariantConfig,
)
from trinity.materials.variant_registry import (
    CompiledVariant,
    MaterialVariantRegistry,
)

if TYPE_CHECKING:
    pass


# Default LOD distance thresholds (in world units)
DEFAULT_LOD_DISTANCES: List[float] = [0.0, 10.0, 50.0, 100.0]

# Default quality mapping per LOD level
DEFAULT_LOD_QUALITIES: List[QualityTier] = [
    QualityTier.HIGH,    # LOD 0: Full quality
    QualityTier.MEDIUM,  # LOD 1: Reduced quality
    QualityTier.LOW,     # LOD 2: Low quality
    QualityTier.LOW,     # LOD 3: Unlit fallback
]

# Maximum supported LOD levels
MAX_LOD_LEVELS = 8


@dataclass(slots=True)
class LODConfig:
    """Configuration for material LOD system.

    Defines distance thresholds and quality tier mapping for LOD levels.
    The number of thresholds determines the number of LOD levels.

    Attributes:
        distances: Distance thresholds for each LOD level. First value should
            be 0.0 (LOD 0 starts at camera). Must be monotonically increasing.
        qualities: Quality tier for each LOD level. Length must match distances.
        blend_range: Distance range for cross-fade between LOD levels.
            0.0 disables cross-fade (instant switch).
        unlit_fallback_lod: LOD level at which to use unlit fallback.
            -1 disables unlit fallback.

    Example::

        config = LODConfig(
            distances=[0.0, 10.0, 50.0, 100.0],
            qualities=[QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW, QualityTier.LOW],
            blend_range=5.0,  # 5 unit cross-fade
        )
    """

    distances: List[float] = field(default_factory=lambda: DEFAULT_LOD_DISTANCES.copy())
    qualities: List[QualityTier] = field(default_factory=lambda: DEFAULT_LOD_QUALITIES.copy())
    blend_range: float = 0.0  # Distance for cross-fade (0 = instant switch)
    unlit_fallback_lod: int = 3  # LOD level for unlit fallback (-1 to disable)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate LOD configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Check lengths match
        if len(self.distances) != len(self.qualities):
            raise ValueError(
                f"distances length ({len(self.distances)}) must match "
                f"qualities length ({len(self.qualities)})"
            )

        # Check minimum levels
        if len(self.distances) < 1:
            raise ValueError("At least 1 LOD level required")

        # Check maximum levels
        if len(self.distances) > MAX_LOD_LEVELS:
            raise ValueError(f"Maximum {MAX_LOD_LEVELS} LOD levels supported")

        # Check first distance is 0 or positive
        if self.distances[0] < 0:
            raise ValueError("First LOD distance must be >= 0")

        # Check monotonically increasing distances
        for i in range(1, len(self.distances)):
            if self.distances[i] <= self.distances[i - 1]:
                raise ValueError(
                    f"LOD distances must be monotonically increasing: "
                    f"distances[{i}]={self.distances[i]} <= "
                    f"distances[{i-1}]={self.distances[i-1]}"
                )

        # Check blend_range is non-negative
        if self.blend_range < 0:
            raise ValueError("blend_range must be >= 0")

        # Check unlit_fallback_lod is valid
        if self.unlit_fallback_lod >= len(self.distances):
            raise ValueError(
                f"unlit_fallback_lod ({self.unlit_fallback_lod}) must be < "
                f"number of LOD levels ({len(self.distances)})"
            )

    @property
    def lod_count(self) -> int:
        """Get the number of LOD levels."""
        return len(self.distances)

    @property
    def max_distance(self) -> float:
        """Get the maximum LOD distance threshold."""
        return self.distances[-1] if self.distances else 0.0

    @property
    def has_crossfade(self) -> bool:
        """Check if cross-fade is enabled."""
        return self.blend_range > 0

    def get_lod_level(self, distance: float) -> int:
        """Get the LOD level for a given distance.

        Args:
            distance: Distance from camera to object.

        Returns:
            LOD level (0 = highest quality, increases with distance).
        """
        if distance < 0:
            return 0

        for i in range(len(self.distances) - 1, -1, -1):
            if distance >= self.distances[i]:
                return i

        return 0

    def get_quality_for_lod(self, lod: int) -> QualityTier:
        """Get the quality tier for a specific LOD level.

        Args:
            lod: LOD level index.

        Returns:
            Quality tier for the LOD level.

        Raises:
            IndexError: If LOD level is out of range.
        """
        if lod < 0 or lod >= len(self.qualities):
            raise IndexError(f"LOD level {lod} out of range [0, {len(self.qualities)-1}]")
        return self.qualities[lod]

    def is_unlit_lod(self, lod: int) -> bool:
        """Check if the LOD level uses unlit fallback.

        Args:
            lod: LOD level index.

        Returns:
            True if this LOD uses unlit fallback.
        """
        if self.unlit_fallback_lod < 0:
            return False
        return lod >= self.unlit_fallback_lod

    @classmethod
    def create_simple(
        cls,
        near: float = 10.0,
        mid: float = 50.0,
        far: float = 100.0,
        blend_range: float = 0.0,
    ) -> "LODConfig":
        """Create a simple 4-level LOD configuration.

        Args:
            near: Distance for LOD 1 (MEDIUM quality).
            mid: Distance for LOD 2 (LOW quality).
            far: Distance for LOD 3 (unlit fallback).
            blend_range: Cross-fade distance.

        Returns:
            LODConfig with 4 levels.
        """
        return cls(
            distances=[0.0, near, mid, far],
            qualities=[
                QualityTier.HIGH,
                QualityTier.MEDIUM,
                QualityTier.LOW,
                QualityTier.LOW,
            ],
            blend_range=blend_range,
            unlit_fallback_lod=3,
        )

    @classmethod
    def create_aggressive(
        cls,
        near: float = 5.0,
        far: float = 25.0,
        blend_range: float = 2.0,
    ) -> "LODConfig":
        """Create an aggressive LOD configuration for performance.

        Uses shorter distances and fewer levels for maximum performance.

        Args:
            near: Distance for LOD 1 (LOW quality).
            far: Distance for LOD 2 (unlit fallback).
            blend_range: Cross-fade distance.

        Returns:
            LODConfig with 3 levels optimized for performance.
        """
        return cls(
            distances=[0.0, near, far],
            qualities=[
                QualityTier.MEDIUM,  # Even LOD 0 is MEDIUM
                QualityTier.LOW,
                QualityTier.LOW,
            ],
            blend_range=blend_range,
            unlit_fallback_lod=2,
        )

    @classmethod
    def create_quality_focused(
        cls,
        near: float = 25.0,
        mid: float = 75.0,
        far: float = 150.0,
        blend_range: float = 5.0,
    ) -> "LODConfig":
        """Create a quality-focused LOD configuration.

        Uses longer distances to maintain quality for longer.

        Args:
            near: Distance for LOD 1.
            mid: Distance for LOD 2.
            far: Distance for LOD 3.
            blend_range: Cross-fade distance.

        Returns:
            LODConfig optimized for visual quality.
        """
        return cls(
            distances=[0.0, near, mid, far],
            qualities=[
                QualityTier.HIGH,
                QualityTier.HIGH,  # Maintain HIGH longer
                QualityTier.MEDIUM,
                QualityTier.LOW,
            ],
            blend_range=blend_range,
            unlit_fallback_lod=3,
        )


@dataclass(frozen=True, slots=True)
class LODBlendInfo:
    """Information for cross-fade blending between LOD levels.

    Attributes:
        near_lod: The LOD level closer to the camera.
        far_lod: The LOD level farther from the camera.
        blend_factor: Blend factor [0.0, 1.0] where 0 = full near_lod, 1 = full far_lod.
        is_blending: True if currently in a blend transition zone.
    """

    near_lod: int
    far_lod: int
    blend_factor: float
    is_blending: bool

    @property
    def primary_lod(self) -> int:
        """Get the dominant LOD level based on blend factor."""
        return self.far_lod if self.blend_factor > 0.5 else self.near_lod


def compute_blend_factor(
    distance: float,
    lod_near_distance: float,
    lod_far_distance: float,
    blend_range: float,
) -> float:
    """Compute the blend factor for cross-fade between LOD levels.

    The blend factor is computed based on the distance within the transition
    zone. The transition zone starts at (lod_far_distance - blend_range) and
    ends at lod_far_distance.

    Args:
        distance: Current distance from camera.
        lod_near_distance: Distance threshold for the nearer LOD level.
        lod_far_distance: Distance threshold for the farther LOD level.
        blend_range: Width of the transition zone.

    Returns:
        Blend factor in range [0.0, 1.0]:
        - 0.0: Fully near LOD
        - 1.0: Fully far LOD
        - 0.0-1.0: Transitioning

    Example::

        # LOD 0 at 0.0, LOD 1 at 10.0, blend_range = 2.0
        # Transition zone is [8.0, 10.0]
        factor = compute_blend_factor(8.5, 0.0, 10.0, 2.0)
        # factor = 0.25 (25% towards LOD 1)
    """
    if blend_range <= 0:
        # No blending - instant switch at far distance
        return 1.0 if distance >= lod_far_distance else 0.0

    # Calculate transition zone boundaries
    transition_start = max(lod_near_distance, lod_far_distance - blend_range)
    transition_end = lod_far_distance

    if distance <= transition_start:
        return 0.0
    elif distance >= transition_end:
        return 1.0
    else:
        # Linear interpolation in transition zone
        t = (distance - transition_start) / (transition_end - transition_start)
        return max(0.0, min(1.0, t))


def smooth_blend_factor(linear_factor: float) -> float:
    """Apply smoothstep to a linear blend factor for smoother transitions.

    Uses the standard smoothstep function: 3t^2 - 2t^3

    Args:
        linear_factor: Linear blend factor in range [0.0, 1.0].

    Returns:
        Smoothed blend factor with ease-in/ease-out.
    """
    t = max(0.0, min(1.0, linear_factor))
    return t * t * (3.0 - 2.0 * t)


class MaterialLODSelector:
    """Distance-based material LOD selection with registry integration.

    This class provides LOD-aware variant selection from a MaterialVariantRegistry,
    enabling automatic quality reduction based on camera distance.

    Features:
    - Distance-based LOD level determination
    - Quality tier mapping per LOD level
    - Cross-fade support for smooth transitions
    - Integration with MaterialVariantRegistry

    Thread Safety:
        Read operations (select_*) are thread-safe if the underlying registry
        is not being modified. Configuration is immutable after construction.

    Example::

        registry = MaterialVariantRegistry()
        registry.register_material("gold", compiler, GoldMaterial)

        config = LODConfig.create_simple(near=10.0, mid=50.0, far=100.0)
        selector = MaterialLODSelector(registry, config)

        # Get LOD level for an object at distance 25.0
        lod = selector.select_lod("gold", 25.0)  # Returns 1 (MEDIUM)

        # Get the appropriate variant
        variant = selector.select_variant("gold", 25.0)

        # For cross-fade (if enabled)
        near_v, far_v, blend = selector.select_with_blend("gold", 8.5)
    """

    def __init__(
        self,
        registry: MaterialVariantRegistry,
        config: Optional[LODConfig] = None,
    ) -> None:
        """Initialize the LOD selector.

        Args:
            registry: MaterialVariantRegistry containing compiled variants.
            config: LODConfig for distance thresholds. Defaults to standard config.
        """
        self._registry = registry
        self._config = config if config is not None else LODConfig()
        # Cache for LOD->variant mapping per material
        self._lod_cache: Dict[str, Dict[int, CompiledVariant]] = {}

    @property
    def registry(self) -> MaterialVariantRegistry:
        """Get the underlying MaterialVariantRegistry."""
        return self._registry

    @property
    def config(self) -> LODConfig:
        """Get the LOD configuration."""
        return self._config

    @property
    def lod_count(self) -> int:
        """Get the number of LOD levels."""
        return self._config.lod_count

    def select_lod(self, material_name: str, distance: float) -> int:
        """Select the LOD level for a material at a given distance.

        Args:
            material_name: Name of the registered material.
            distance: Distance from camera to object (in world units).

        Returns:
            LOD level (0 = highest quality, increases with distance).
            Returns 0 if distance is negative.
        """
        return self._config.get_lod_level(distance)

    def select_quality(self, material_name: str, distance: float) -> QualityTier:
        """Select the quality tier for a material at a given distance.

        Args:
            material_name: Name of the registered material.
            distance: Distance from camera to object.

        Returns:
            Quality tier for the current LOD level.
        """
        lod = self.select_lod(material_name, distance)
        return self._config.get_quality_for_lod(lod)

    def select_variant(
        self,
        material_name: str,
        distance: float,
        domain: Optional[MaterialDomain] = None,
        blend: Optional[BlendMode] = None,
    ) -> Optional[CompiledVariant]:
        """Select the appropriate variant for a material at a given distance.

        Args:
            material_name: Name of the registered material.
            distance: Distance from camera to object.
            domain: Optional domain filter. Defaults to SURFACE.
            blend: Optional blend mode filter. Defaults to OPAQUE.

        Returns:
            CompiledVariant for the appropriate LOD level, or None if not found.
        """
        quality = self.select_quality(material_name, distance)
        return self._registry.select_variant(material_name, quality, domain, blend)

    def select_with_blend(
        self,
        material_name: str,
        distance: float,
        domain: Optional[MaterialDomain] = None,
        blend: Optional[BlendMode] = None,
        use_smooth_blend: bool = True,
    ) -> Tuple[Optional[CompiledVariant], Optional[CompiledVariant], float]:
        """Select variants and blend factor for cross-fade transition.

        When the object is within a transition zone (as defined by config.blend_range),
        this method returns both the near and far LOD variants along with a blend
        factor for cross-fade rendering.

        Args:
            material_name: Name of the registered material.
            distance: Distance from camera to object.
            domain: Optional domain filter.
            blend: Optional blend mode filter.
            use_smooth_blend: Apply smoothstep to blend factor (default: True).

        Returns:
            Tuple of (near_variant, far_variant, blend_factor):
            - near_variant: Variant for the closer LOD level
            - far_variant: Variant for the farther LOD level
            - blend_factor: [0.0, 1.0] where 0 = full near, 1 = full far

            If not in a transition zone, near_variant == far_variant and
            blend_factor is 0.0 or 1.0.
        """
        lod = self.select_lod(material_name, distance)

        # If no blending or at last LOD, return single variant
        if not self._config.has_crossfade or lod >= self._config.lod_count - 1:
            variant = self.select_variant(material_name, distance, domain, blend)
            return (variant, variant, 0.0)

        # Compute blend info
        blend_info = self.get_blend_info(material_name, distance)

        if not blend_info.is_blending:
            # Not in transition zone
            variant = self.select_variant(material_name, distance, domain, blend)
            return (variant, variant, 0.0)

        # Get both variants for cross-fade
        near_quality = self._config.get_quality_for_lod(blend_info.near_lod)
        far_quality = self._config.get_quality_for_lod(blend_info.far_lod)

        near_variant = self._registry.select_variant(
            material_name, near_quality, domain, blend
        )
        far_variant = self._registry.select_variant(
            material_name, far_quality, domain, blend
        )

        factor = blend_info.blend_factor
        if use_smooth_blend:
            factor = smooth_blend_factor(factor)

        return (near_variant, far_variant, factor)

    def get_blend_info(self, material_name: str, distance: float) -> LODBlendInfo:
        """Get detailed LOD blend information for a distance.

        Args:
            material_name: Name of the registered material.
            distance: Distance from camera to object.

        Returns:
            LODBlendInfo with LOD levels and blend factor.
        """
        lod = self.select_lod(material_name, distance)

        # Check if we're at the last LOD or no blending enabled
        if not self._config.has_crossfade or lod >= self._config.lod_count - 1:
            return LODBlendInfo(
                near_lod=lod,
                far_lod=lod,
                blend_factor=0.0,
                is_blending=False,
            )

        # Get distance thresholds for transition
        near_distance = self._config.distances[lod]
        far_distance = self._config.distances[lod + 1]

        # Compute blend factor
        factor = compute_blend_factor(
            distance,
            near_distance,
            far_distance,
            self._config.blend_range,
        )

        # Determine if we're in the transition zone
        is_blending = 0.0 < factor < 1.0

        return LODBlendInfo(
            near_lod=lod,
            far_lod=lod + 1 if factor > 0 else lod,
            blend_factor=factor,
            is_blending=is_blending,
        )

    def is_unlit(self, material_name: str, distance: float) -> bool:
        """Check if the material should use unlit fallback at this distance.

        Args:
            material_name: Name of the registered material.
            distance: Distance from camera to object.

        Returns:
            True if the current LOD level uses unlit fallback.
        """
        lod = self.select_lod(material_name, distance)
        return self._config.is_unlit_lod(lod)

    def get_lod_distances(self) -> List[float]:
        """Get the LOD distance thresholds."""
        return self._config.distances.copy()

    def get_lod_qualities(self) -> List[QualityTier]:
        """Get the quality tier for each LOD level."""
        return self._config.qualities.copy()

    def precompute_lod_variants(
        self,
        material_name: str,
        domain: Optional[MaterialDomain] = None,
        blend: Optional[BlendMode] = None,
    ) -> Dict[int, CompiledVariant]:
        """Precompute and cache variants for all LOD levels.

        Args:
            material_name: Name of the registered material.
            domain: Optional domain filter.
            blend: Optional blend mode filter.

        Returns:
            Dict mapping LOD level to CompiledVariant.
        """
        cache_key = material_name
        if cache_key in self._lod_cache:
            return self._lod_cache[cache_key]

        result: Dict[int, CompiledVariant] = {}
        for lod in range(self._config.lod_count):
            quality = self._config.get_quality_for_lod(lod)
            variant = self._registry.select_variant(material_name, quality, domain, blend)
            if variant is not None:
                result[lod] = variant

        self._lod_cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        """Clear the LOD variant cache."""
        self._lod_cache.clear()


def create_lod_selector_with_defaults(
    registry: MaterialVariantRegistry,
) -> MaterialLODSelector:
    """Create a MaterialLODSelector with default configuration.

    Args:
        registry: MaterialVariantRegistry containing compiled variants.

    Returns:
        MaterialLODSelector with default LOD distances.
    """
    return MaterialLODSelector(registry, LODConfig())


def create_lod_selector_for_quality(
    registry: MaterialVariantRegistry,
    quality_preset: str,
) -> MaterialLODSelector:
    """Create a MaterialLODSelector optimized for a quality preset.

    Args:
        registry: MaterialVariantRegistry containing compiled variants.
        quality_preset: One of "performance", "balanced", "quality".

    Returns:
        MaterialLODSelector with configuration for the preset.

    Raises:
        ValueError: If quality_preset is not recognized.
    """
    if quality_preset == "performance":
        config = LODConfig.create_aggressive()
    elif quality_preset == "balanced":
        config = LODConfig.create_simple(blend_range=2.0)
    elif quality_preset == "quality":
        config = LODConfig.create_quality_focused()
    else:
        raise ValueError(
            f"Unknown quality preset: {quality_preset}. "
            "Use 'performance', 'balanced', or 'quality'."
        )

    return MaterialLODSelector(registry, config)


__all__ = [
    # Configuration
    "LODConfig",
    "LODBlendInfo",
    "DEFAULT_LOD_DISTANCES",
    "DEFAULT_LOD_QUALITIES",
    "MAX_LOD_LEVELS",
    # Core class
    "MaterialLODSelector",
    # Helper functions
    "compute_blend_factor",
    "smooth_blend_factor",
    "create_lod_selector_with_defaults",
    "create_lod_selector_for_quality",
]
