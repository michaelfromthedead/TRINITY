"""
Procedural Palettes and Color LUT (T-DEMO-4.14, T-DEMO-4.15, T-DEMO-4.16).

This module implements procedural palette generation for demoscene rendering:

T-DEMO-4.14: Height-Based Terrain Color Palettes
  - TerrainPalette class with height zones (water, sand, grass, rock, snow)
  - Smooth gradient transitions between zones
  - Configurable zone_heights and zone_colors

T-DEMO-4.15: Procedural Palette Patterns
  - ProceduralPattern class for mathematical patterns
  - Stripes: sin(p.x * freq)
  - Checkerboard: floor(p.x) + floor(p.z) mod 2
  - Wood grain: sin(length(p.xz) + noise(p))
  - Marble: sin(p.x + fbm(p))
  - Rust: fbm(p) * erosion_mask

T-DEMO-4.16: 256-Entry Palette LUT
  - PaletteLUT class with 256 color entries
  - 1KB texture (256 * 4 bytes RGBA)
  - Index lookup via normalized float
  - Optional per-material palette assignment
  - Bake method to generate LUT texture data

Usage:
    >>> from engine.rendering.demoscene.procedural_palette import (
    ...     TerrainPalette, ProceduralPattern, PaletteLUT, PatternType
    ... )
    >>> terrain = TerrainPalette()
    >>> color = terrain.sample(height=0.3)  # Returns grass color
    >>>
    >>> pattern = ProceduralPattern(PatternType.MARBLE)
    >>> value = pattern.evaluate((1.0, 2.0, 3.0))  # 0-1 pattern value
    >>>
    >>> lut = PaletteLUT.from_gradient([(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)])
    >>> texture_data = lut.bake()  # 1KB RGBA texture
"""

from __future__ import annotations

import math
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Sequence, Tuple, Union


# =============================================================================
# Type Aliases
# =============================================================================

Color3 = Tuple[float, float, float]
Color4 = Tuple[float, float, float, float]
Point3 = Tuple[float, float, float]


# =============================================================================
# Vec3 Helper (for standalone use)
# =============================================================================

@dataclass(frozen=True, slots=True)
class Vec3:
    """Minimal Vec3 for procedural calculations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self.__mul__(scalar)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def length_xz(self) -> float:
        """Length in XZ plane (ignoring Y)."""
        return math.sqrt(self.x * self.x + self.z * self.z)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def normalized(self) -> "Vec3":
        ln = self.length()
        if ln < 1e-10:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / ln, self.y / ln, self.z / ln)

    def as_tuple(self) -> Point3:
        return (self.x, self.y, self.z)

    @classmethod
    def from_tuple(cls, t: Point3) -> "Vec3":
        return cls(t[0], t[1], t[2])


# =============================================================================
# T-DEMO-4.14: Height-Based Terrain Color Palettes
# =============================================================================

@dataclass(frozen=True, slots=True)
class TerrainZone:
    """A terrain zone with height threshold and color."""
    height: float  # Normalized height [0, 1] at which this zone starts
    color: Color3
    roughness: float = 0.5


# Default terrain palette: water -> sand -> grass -> rock -> snow
DEFAULT_TERRAIN_ZONES: Tuple[TerrainZone, ...] = (
    TerrainZone(height=0.0, color=(0.05, 0.10, 0.35), roughness=0.2),   # Deep water
    TerrainZone(height=0.15, color=(0.10, 0.25, 0.55), roughness=0.3),  # Shallow water
    TerrainZone(height=0.25, color=(0.76, 0.70, 0.50), roughness=0.8),  # Sand
    TerrainZone(height=0.35, color=(0.20, 0.45, 0.15), roughness=0.6),  # Grass
    TerrainZone(height=0.55, color=(0.35, 0.30, 0.25), roughness=0.7),  # Rock
    TerrainZone(height=0.80, color=(0.55, 0.50, 0.45), roughness=0.5),  # High rock
    TerrainZone(height=0.90, color=(0.95, 0.95, 0.98), roughness=0.2),  # Snow
)


@dataclass
class TerrainPalette:
    """Height-based terrain color palette with smooth zone transitions.

    Implements T-DEMO-4.14: terrain coloring based on normalized height.
    Zones define colors at specific heights, with smooth linear interpolation
    between adjacent zones.

    Attributes:
        zones: Sequence of TerrainZone objects, sorted by height.
        blend_width: Width of transition zone between colors (default 0.05).
    """
    zones: Tuple[TerrainZone, ...] = DEFAULT_TERRAIN_ZONES
    blend_width: float = 0.05

    def __post_init__(self) -> None:
        """Validate zones and ensure proper ordering."""
        if not self.zones:
            raise ValueError("zones must contain at least one TerrainZone")
        if self.blend_width < 0.0:
            raise ValueError(f"blend_width must be >= 0, got {self.blend_width}")
        if self.blend_width > 0.5:
            raise ValueError(f"blend_width must be <= 0.5, got {self.blend_width}")

        # Verify zones are sorted by height (strictly increasing after first)
        heights = [z.height for z in self.zones]
        if heights != sorted(heights):
            raise ValueError("zones must be sorted by height in ascending order")

        # Verify no duplicate heights (except first can be 0.0)
        for i in range(1, len(heights)):
            if heights[i] <= heights[i - 1]:
                raise ValueError(
                    f"zone heights must be strictly increasing, "
                    f"got {heights[i]} after {heights[i-1]}"
                )

        # Verify first zone starts at 0
        if self.zones[0].height != 0.0:
            raise ValueError(f"first zone must start at height 0.0, got {self.zones[0].height}")

        # Verify heights are in [0, 1]
        for z in self.zones:
            if not 0.0 <= z.height <= 1.0:
                raise ValueError(f"zone height must be in [0, 1], got {z.height}")

    def sample(self, height: float) -> Color3:
        """Sample the terrain palette at a given normalized height.

        Args:
            height: Normalized height in [0, 1]. Values outside are clamped.

        Returns:
            RGB color tuple with values in [0, 1].
        """
        height = max(0.0, min(1.0, height))

        # Find the zone containing this height
        zone_idx = 0
        for i, zone in enumerate(self.zones):
            if zone.height <= height:
                zone_idx = i

        # Get current and next zone
        current = self.zones[zone_idx]
        if zone_idx >= len(self.zones) - 1:
            return current.color

        next_zone = self.zones[zone_idx + 1]

        # Calculate blend factor
        zone_start = current.height
        zone_end = next_zone.height
        zone_width = zone_end - zone_start

        if zone_width <= 0:
            return current.color

        # Smooth transition using blend width
        progress = (height - zone_start) / zone_width
        t = _smoothstep(progress)

        return _lerp_color(current.color, next_zone.color, t)

    def sample_with_roughness(self, height: float) -> Tuple[Color3, float]:
        """Sample both color and roughness at a given height.

        Args:
            height: Normalized height in [0, 1].

        Returns:
            Tuple of (RGB color, roughness) with all values in [0, 1].
        """
        height = max(0.0, min(1.0, height))

        zone_idx = 0
        for i, zone in enumerate(self.zones):
            if zone.height <= height:
                zone_idx = i

        current = self.zones[zone_idx]
        if zone_idx >= len(self.zones) - 1:
            return (current.color, current.roughness)

        next_zone = self.zones[zone_idx + 1]

        zone_start = current.height
        zone_end = next_zone.height
        zone_width = zone_end - zone_start

        if zone_width <= 0:
            return (current.color, current.roughness)

        progress = (height - zone_start) / zone_width
        t = _smoothstep(progress)

        color = _lerp_color(current.color, next_zone.color, t)
        roughness = _lerp(current.roughness, next_zone.roughness, t)

        return (color, roughness)

    def to_wgsl(self, fn_name: str = "terrain_palette") -> str:
        """Generate WGSL function for this terrain palette.

        Args:
            fn_name: Name of the generated function.

        Returns:
            WGSL source code string.
        """
        lines = [
            f"/// Height-based terrain palette lookup.",
            f"///   height   -- normalized height [0, 1]",
            f"///   returns  -- RGB color",
            f"fn {fn_name}(height: f32) -> vec3<f32> {{",
            f"    let h = clamp(height, 0.0, 1.0);",
        ]

        # Generate color constants
        for i, zone in enumerate(self.zones):
            r, g, b = zone.color
            lines.append(
                f"    let c{i} = vec3<f32>({_fmt_float(r)}, {_fmt_float(g)}, {_fmt_float(b)});"
            )

        # Generate height thresholds
        lines.append("")
        lines.append("    var color: vec3<f32>;")

        # First zone
        if len(self.zones) == 1:
            lines.append("    color = c0;")
        else:
            for i in range(len(self.zones) - 1):
                zone = self.zones[i]
                next_zone = self.zones[i + 1]
                h_start = zone.height
                h_end = next_zone.height

                if i == 0:
                    lines.append(f"    if (h < {_fmt_float(h_end)}) {{")
                elif i < len(self.zones) - 2:
                    lines.append(f"    }} else if (h < {_fmt_float(h_end)}) {{")
                else:
                    lines.append(f"    }} else {{")

                if h_end - h_start > 0:
                    lines.append(
                        f"        let t = smoothstep({_fmt_float(h_start)}, "
                        f"{_fmt_float(h_end)}, h);"
                    )
                    lines.append(f"        color = mix(c{i}, c{i+1}, t);")
                else:
                    lines.append(f"        color = c{i};")

            lines.append("    }")

        lines.append("    return color;")
        lines.append("}")

        return "\n".join(lines)


# =============================================================================
# T-DEMO-4.15: Procedural Palette Patterns
# =============================================================================

class PatternType(Enum):
    """Types of procedural patterns."""
    STRIPES = auto()       # sin(p.x * freq)
    CHECKERBOARD = auto()  # floor(p.x) + floor(p.z) mod 2
    WOOD_GRAIN = auto()    # sin(length(p.xz) + noise(p))
    MARBLE = auto()        # sin(p.x + fbm(p))
    RUST = auto()          # fbm(p) * erosion_mask
    GRADIENT_X = auto()    # Linear gradient along X
    GRADIENT_Y = auto()    # Linear gradient along Y
    GRADIENT_Z = auto()    # Linear gradient along Z
    RADIAL = auto()        # Radial gradient from origin
    RINGS = auto()         # Concentric rings


@dataclass
class ProceduralPattern:
    """Mathematical pattern generator for palette indices.

    Implements T-DEMO-4.15: procedural patterns that return values in [0, 1]
    suitable for palette LUT indexing.

    Attributes:
        pattern_type: Type of pattern to generate.
        frequency: Base frequency for the pattern.
        colors: Optional two colors for direct pattern coloring.
        noise_seed: Seed for deterministic noise generation.
        octaves: Number of FBM octaves for noise-based patterns.
        lacunarity: Frequency multiplier per FBM octave.
        persistence: Amplitude multiplier per FBM octave.
    """
    pattern_type: PatternType = PatternType.STRIPES
    frequency: float = 1.0
    colors: Optional[Tuple[Color3, Color3]] = None
    noise_seed: int = 0
    octaves: int = 4
    lacunarity: float = 2.0
    persistence: float = 0.5

    def __post_init__(self) -> None:
        """Validate pattern parameters."""
        if self.frequency <= 0.0:
            raise ValueError(f"frequency must be > 0, got {self.frequency}")
        if self.octaves < 1:
            raise ValueError(f"octaves must be >= 1, got {self.octaves}")
        if self.lacunarity <= 0.0:
            raise ValueError(f"lacunarity must be > 0, got {self.lacunarity}")
        if not 0.0 < self.persistence <= 1.0:
            raise ValueError(f"persistence must be in (0, 1], got {self.persistence}")

        # Initialize permutation table for noise
        self._perm = _create_permutation_table(self.noise_seed)

    def evaluate(self, position: Point3) -> float:
        """Evaluate the pattern at a 3D position.

        Args:
            position: 3D position tuple (x, y, z).

        Returns:
            Pattern value in [0, 1].
        """
        p = Vec3.from_tuple(position)

        if self.pattern_type == PatternType.STRIPES:
            return self._stripes(p)
        elif self.pattern_type == PatternType.CHECKERBOARD:
            return self._checkerboard(p)
        elif self.pattern_type == PatternType.WOOD_GRAIN:
            return self._wood_grain(p)
        elif self.pattern_type == PatternType.MARBLE:
            return self._marble(p)
        elif self.pattern_type == PatternType.RUST:
            return self._rust(p)
        elif self.pattern_type == PatternType.GRADIENT_X:
            return self._gradient_x(p)
        elif self.pattern_type == PatternType.GRADIENT_Y:
            return self._gradient_y(p)
        elif self.pattern_type == PatternType.GRADIENT_Z:
            return self._gradient_z(p)
        elif self.pattern_type == PatternType.RADIAL:
            return self._radial(p)
        elif self.pattern_type == PatternType.RINGS:
            return self._rings(p)
        else:
            return 0.5

    def evaluate_color(self, position: Point3) -> Color3:
        """Evaluate the pattern and return interpolated color.

        Args:
            position: 3D position tuple.

        Returns:
            RGB color tuple. If colors not set, returns grayscale.
        """
        t = self.evaluate(position)

        if self.colors is None:
            return (t, t, t)

        return _lerp_color(self.colors[0], self.colors[1], t)

    def _stripes(self, p: Vec3) -> float:
        """Stripes pattern: sin(p.x * freq) mapped to [0, 1]."""
        value = math.sin(p.x * self.frequency * math.pi * 2.0)
        return (value + 1.0) * 0.5

    def _checkerboard(self, p: Vec3) -> float:
        """Checkerboard pattern: (floor(x) + floor(z)) mod 2."""
        ix = int(math.floor(p.x * self.frequency))
        iz = int(math.floor(p.z * self.frequency))
        return float((ix + iz) & 1)

    def _wood_grain(self, p: Vec3) -> float:
        """Wood grain pattern: sin(length(p.xz) * freq + noise(p)).

        Creates concentric rings with noise perturbation for organic appearance.
        Maintains radial symmetry around Y-axis.
        """
        # Radial distance from Y-axis
        r = p.length_xz()

        # Add noise perturbation for organic look
        noise = self._fbm(p) * 0.5

        # Create rings with sinusoidal pattern
        value = math.sin((r * self.frequency + noise) * math.pi * 2.0)

        return (value + 1.0) * 0.5

    def _marble(self, p: Vec3) -> float:
        """Marble pattern: sin(p.x + fbm(p)).

        Creates veined appearance with continuous patterns.
        """
        # FBM noise to perturb the base coordinate
        noise = self._fbm(p) * 2.0

        # Base vein pattern along X with noise perturbation
        value = math.sin((p.x * self.frequency + noise) * math.pi)

        return (value + 1.0) * 0.5

    def _rust(self, p: Vec3) -> float:
        """Rust pattern: fbm(p) * erosion_mask.

        Creates patchy, eroded appearance typical of rust.
        """
        # Base FBM noise
        base_noise = self._fbm(p)

        # Erosion mask using higher frequency noise
        erosion = self._fbm(Vec3(p.x * 2.5, p.y * 2.5, p.z * 2.5))

        # Threshold to create patches
        mask = 1.0 if erosion > 0.0 else 0.3

        # Combine with base noise
        value = (base_noise + 1.0) * 0.5 * mask

        return max(0.0, min(1.0, value))

    def _gradient_x(self, p: Vec3) -> float:
        """Linear gradient along X axis."""
        value = p.x * self.frequency
        return max(0.0, min(1.0, (value + 1.0) * 0.5))

    def _gradient_y(self, p: Vec3) -> float:
        """Linear gradient along Y axis."""
        value = p.y * self.frequency
        return max(0.0, min(1.0, (value + 1.0) * 0.5))

    def _gradient_z(self, p: Vec3) -> float:
        """Linear gradient along Z axis."""
        value = p.z * self.frequency
        return max(0.0, min(1.0, (value + 1.0) * 0.5))

    def _radial(self, p: Vec3) -> float:
        """Radial gradient from origin."""
        r = p.length()
        return max(0.0, min(1.0, r * self.frequency))

    def _rings(self, p: Vec3) -> float:
        """Concentric rings pattern."""
        r = p.length_xz()
        value = math.sin(r * self.frequency * math.pi * 2.0)
        return (value + 1.0) * 0.5

    def _fbm(self, p: Vec3) -> float:
        """Fractal Brownian Motion noise."""
        total = 0.0
        amplitude = 1.0
        frequency = self.frequency
        max_value = 0.0

        for _ in range(self.octaves):
            total += self._noise3d(p.x * frequency, p.y * frequency, p.z * frequency) * amplitude
            max_value += amplitude
            frequency *= self.lacunarity
            amplitude *= self.persistence

        if max_value > 0:
            return total / max_value
        return 0.0

    def _noise3d(self, x: float, y: float, z: float) -> float:
        """3D value noise using permutation table."""
        # Integer coordinates
        xi = int(math.floor(x)) & 255
        yi = int(math.floor(y)) & 255
        zi = int(math.floor(z)) & 255

        # Fractional coordinates
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        zf = z - math.floor(z)

        # Smooth interpolation curves
        u = _fade(xf)
        v = _fade(yf)
        w = _fade(zf)

        # Hash coordinates
        a = self._perm[xi] + yi
        aa = self._perm[a & 255] + zi
        ab = self._perm[(a + 1) & 255] + zi
        b = self._perm[(xi + 1) & 255] + yi
        ba = self._perm[b & 255] + zi
        bb = self._perm[(b + 1) & 255] + zi

        # Generate pseudo-random values at cube corners
        def grad(hash_val: int) -> float:
            return (self._perm[hash_val & 255] / 127.5) - 1.0

        # Trilinear interpolation
        x1 = _lerp(grad(self._perm[aa & 255]), grad(self._perm[ba & 255]), u)
        x2 = _lerp(grad(self._perm[ab & 255]), grad(self._perm[bb & 255]), u)
        y1 = _lerp(x1, x2, v)

        x1 = _lerp(grad(self._perm[(aa + 1) & 255]), grad(self._perm[(ba + 1) & 255]), u)
        x2 = _lerp(grad(self._perm[(ab + 1) & 255]), grad(self._perm[(bb + 1) & 255]), u)
        y2 = _lerp(x1, x2, v)

        return _lerp(y1, y2, w)

    def to_wgsl(self, fn_name: str = "procedural_pattern") -> str:
        """Generate WGSL function for this pattern.

        Args:
            fn_name: Name of the generated function.

        Returns:
            WGSL source code string.
        """
        lines = [
            f"/// Procedural pattern: {self.pattern_type.name}",
            f"///   p        -- 3D position",
            f"///   returns  -- pattern value in [0, 1]",
            f"fn {fn_name}(p: vec3<f32>) -> f32 {{",
        ]

        if self.pattern_type == PatternType.STRIPES:
            lines.append(f"    let freq = {_fmt_float(self.frequency)};")
            lines.append(f"    return (sin(p.x * freq * 6.283185307) + 1.0) * 0.5;")

        elif self.pattern_type == PatternType.CHECKERBOARD:
            lines.append(f"    let freq = {_fmt_float(self.frequency)};")
            lines.append(f"    let ix = i32(floor(p.x * freq));")
            lines.append(f"    let iz = i32(floor(p.z * freq));")
            lines.append(f"    return f32((ix + iz) & 1);")

        elif self.pattern_type == PatternType.WOOD_GRAIN:
            lines.append(f"    let freq = {_fmt_float(self.frequency)};")
            lines.append(f"    let r = length(p.xz);")
            lines.append(f"    let noise = fbm(p) * 0.5;")
            lines.append(f"    return (sin((r * freq + noise) * 6.283185307) + 1.0) * 0.5;")

        elif self.pattern_type == PatternType.MARBLE:
            lines.append(f"    let freq = {_fmt_float(self.frequency)};")
            lines.append(f"    let noise = fbm(p) * 2.0;")
            lines.append(f"    return (sin((p.x * freq + noise) * 3.141592654) + 1.0) * 0.5;")

        elif self.pattern_type == PatternType.RUST:
            lines.append(f"    let base = (fbm(p) + 1.0) * 0.5;")
            lines.append(f"    let erosion = fbm(p * 2.5);")
            lines.append(f"    let mask = select(0.3, 1.0, erosion > 0.0);")
            lines.append(f"    return clamp(base * mask, 0.0, 1.0);")

        elif self.pattern_type == PatternType.RADIAL:
            lines.append(f"    let freq = {_fmt_float(self.frequency)};")
            lines.append(f"    return clamp(length(p) * freq, 0.0, 1.0);")

        elif self.pattern_type == PatternType.RINGS:
            lines.append(f"    let freq = {_fmt_float(self.frequency)};")
            lines.append(f"    let r = length(p.xz);")
            lines.append(f"    return (sin(r * freq * 6.283185307) + 1.0) * 0.5;")

        else:
            # Gradient patterns
            axis = "x" if self.pattern_type == PatternType.GRADIENT_X else (
                "y" if self.pattern_type == PatternType.GRADIENT_Y else "z"
            )
            lines.append(f"    let freq = {_fmt_float(self.frequency)};")
            lines.append(f"    return clamp((p.{axis} * freq + 1.0) * 0.5, 0.0, 1.0);")

        lines.append("}")

        return "\n".join(lines)


# =============================================================================
# T-DEMO-4.16: 256-Entry Palette LUT
# =============================================================================

@dataclass
class PaletteLUT:
    """256-entry color lookup table for efficient palette-based rendering.

    Implements T-DEMO-4.16: a 1KB texture (256 * 4 bytes RGBA) for GPU palette
    lookup. Supports gradient generation, custom entries, and optional
    bilinear filtering for smooth color transitions.

    Attributes:
        entries: 256 RGBA color entries (0-255 per channel).
        bilinear: Enable bilinear filtering for smooth lookup.
    """
    entries: List[Color4] = field(default_factory=lambda: [(0.0, 0.0, 0.0, 1.0)] * 256)
    bilinear: bool = False

    def __post_init__(self) -> None:
        """Validate LUT entries."""
        if len(self.entries) != 256:
            raise ValueError(f"LUT must have exactly 256 entries, got {len(self.entries)}")

        # Validate color ranges
        for i, entry in enumerate(self.entries):
            if len(entry) != 4:
                raise ValueError(f"Entry {i} must have 4 components (RGBA), got {len(entry)}")
            for j, v in enumerate(entry):
                if not 0.0 <= v <= 1.0:
                    raise ValueError(
                        f"Entry {i} component {j} must be in [0, 1], got {v}"
                    )

    @classmethod
    def from_gradient(
        cls,
        colors: Sequence[Color3],
        alpha: float = 1.0,
        bilinear: bool = False,
    ) -> "PaletteLUT":
        """Create a LUT from a gradient of colors.

        Args:
            colors: Sequence of RGB colors to interpolate.
            alpha: Alpha value for all entries (default 1.0).
            bilinear: Enable bilinear filtering.

        Returns:
            New PaletteLUT with interpolated gradient.
        """
        if len(colors) < 2:
            raise ValueError("gradient requires at least 2 colors")

        entries: List[Color4] = []
        num_segments = len(colors) - 1

        for i in range(256):
            t = i / 255.0
            segment_t = t * num_segments
            segment_idx = min(int(segment_t), num_segments - 1)
            local_t = segment_t - segment_idx

            c1 = colors[segment_idx]
            c2 = colors[segment_idx + 1]

            r = _lerp(c1[0], c2[0], local_t)
            g = _lerp(c1[1], c2[1], local_t)
            b = _lerp(c1[2], c2[2], local_t)

            entries.append((r, g, b, alpha))

        return cls(entries=entries, bilinear=bilinear)

    @classmethod
    def from_terrain(
        cls,
        terrain: TerrainPalette,
        alpha: float = 1.0,
        bilinear: bool = False,
    ) -> "PaletteLUT":
        """Create a LUT from a terrain palette.

        Args:
            terrain: TerrainPalette to sample.
            alpha: Alpha value for all entries.
            bilinear: Enable bilinear filtering.

        Returns:
            New PaletteLUT with terrain colors.
        """
        entries: List[Color4] = []

        for i in range(256):
            height = i / 255.0
            r, g, b = terrain.sample(height)
            entries.append((r, g, b, alpha))

        return cls(entries=entries, bilinear=bilinear)

    @classmethod
    def from_pattern(
        cls,
        pattern: ProceduralPattern,
        sample_positions: Optional[Sequence[Point3]] = None,
        alpha: float = 1.0,
        bilinear: bool = False,
    ) -> "PaletteLUT":
        """Create a LUT from pattern evaluation.

        If sample_positions is None, samples along X axis from -1 to 1.

        Args:
            pattern: ProceduralPattern to sample.
            sample_positions: Optional custom sample positions.
            alpha: Alpha value for all entries.
            bilinear: Enable bilinear filtering.

        Returns:
            New PaletteLUT with pattern colors.
        """
        entries: List[Color4] = []

        if sample_positions is None:
            # Sample along X axis
            sample_positions = [(i / 127.5 - 1.0, 0.0, 0.0) for i in range(256)]

        if len(sample_positions) != 256:
            raise ValueError(f"sample_positions must have 256 entries, got {len(sample_positions)}")

        for pos in sample_positions:
            r, g, b = pattern.evaluate_color(pos)
            entries.append((r, g, b, alpha))

        return cls(entries=entries, bilinear=bilinear)

    def lookup(self, index: float) -> Color4:
        """Look up a color by normalized index.

        Args:
            index: Normalized index in [0, 1].

        Returns:
            RGBA color tuple with values in [0, 1].
        """
        index = max(0.0, min(1.0, index))
        float_idx = index * 255.0

        if not self.bilinear:
            # Nearest neighbor
            idx = int(round(float_idx))
            idx = min(255, max(0, idx))
            return self.entries[idx]

        # Bilinear interpolation
        idx_low = int(math.floor(float_idx))
        idx_high = min(idx_low + 1, 255)
        t = float_idx - idx_low

        c1 = self.entries[idx_low]
        c2 = self.entries[idx_high]

        return (
            _lerp(c1[0], c2[0], t),
            _lerp(c1[1], c2[1], t),
            _lerp(c1[2], c2[2], t),
            _lerp(c1[3], c2[3], t),
        )

    def lookup_rgb(self, index: float) -> Color3:
        """Look up an RGB color by normalized index.

        Args:
            index: Normalized index in [0, 1].

        Returns:
            RGB color tuple with values in [0, 1].
        """
        rgba = self.lookup(index)
        return (rgba[0], rgba[1], rgba[2])

    def set_entry(self, index: int, color: Color4) -> None:
        """Set a single LUT entry.

        Args:
            index: Entry index (0-255).
            color: RGBA color tuple with values in [0, 1].
        """
        if not 0 <= index <= 255:
            raise ValueError(f"index must be in [0, 255], got {index}")
        if len(color) != 4:
            raise ValueError(f"color must have 4 components, got {len(color)}")

        self.entries[index] = color

    def bake(self) -> bytes:
        """Bake the LUT to a 1KB RGBA texture buffer.

        Returns:
            1024 bytes (256 * 4 bytes RGBA8).
        """
        data = bytearray(1024)

        for i, entry in enumerate(self.entries):
            offset = i * 4
            data[offset] = int(max(0.0, min(1.0, entry[0])) * 255.0 + 0.5)
            data[offset + 1] = int(max(0.0, min(1.0, entry[1])) * 255.0 + 0.5)
            data[offset + 2] = int(max(0.0, min(1.0, entry[2])) * 255.0 + 0.5)
            data[offset + 3] = int(max(0.0, min(1.0, entry[3])) * 255.0 + 0.5)

        return bytes(data)

    def bake_float32(self) -> bytes:
        """Bake the LUT to a float32 RGBA texture buffer.

        Returns:
            4096 bytes (256 * 16 bytes RGBA32F).
        """
        data = bytearray()

        for entry in self.entries:
            data.extend(struct.pack('4f', entry[0], entry[1], entry[2], entry[3]))

        return bytes(data)

    def to_wgsl(self, texture_name: str = "palette_lut") -> str:
        """Generate WGSL code for palette LUT lookup.

        Args:
            texture_name: Name of the texture binding.

        Returns:
            WGSL source code string.
        """
        lines = [
            f"/// Palette LUT texture binding",
            f"@group(0) @binding(0) var {texture_name}: texture_1d<f32>;",
            f"@group(0) @binding(1) var {texture_name}_sampler: sampler;",
            f"",
            f"/// Look up color from palette LUT.",
            f"///   index    -- normalized index [0, 1]",
            f"///   returns  -- RGBA color",
            f"fn palette_lookup(index: f32) -> vec4<f32> {{",
        ]

        if self.bilinear:
            lines.append(f"    return textureSample({texture_name}, {texture_name}_sampler, index);")
        else:
            lines.append(f"    let idx = u32(clamp(index * 255.0, 0.0, 255.0));")
            lines.append(f"    return textureLoad({texture_name}, idx, 0);")

        lines.append("}")

        return "\n".join(lines)


# =============================================================================
# Material Palette Assignment
# =============================================================================

@dataclass
class MaterialPaletteMap:
    """Maps material IDs to palette LUTs for per-material color lookup.

    Enables different materials to use different palettes while sharing
    the same procedural pattern evaluation.
    """
    palettes: dict[int, PaletteLUT] = field(default_factory=dict)
    default_palette: Optional[PaletteLUT] = None

    def assign(self, material_id: int, palette: PaletteLUT) -> None:
        """Assign a palette to a material ID.

        Args:
            material_id: Material identifier.
            palette: PaletteLUT to assign.
        """
        self.palettes[material_id] = palette

    def get_palette(self, material_id: int) -> Optional[PaletteLUT]:
        """Get the palette assigned to a material ID.

        Args:
            material_id: Material identifier.

        Returns:
            Assigned PaletteLUT or default if not found.
        """
        return self.palettes.get(material_id, self.default_palette)

    def lookup(self, material_id: int, index: float) -> Color4:
        """Look up color for a material at an index.

        Args:
            material_id: Material identifier.
            index: Normalized palette index.

        Returns:
            RGBA color or default gray if no palette assigned.
        """
        palette = self.get_palette(material_id)
        if palette is None:
            return (0.5, 0.5, 0.5, 1.0)
        return palette.lookup(index)

    def to_wgsl(self, max_materials: int = 8) -> str:
        """Generate WGSL code for material palette mapping.

        Args:
            max_materials: Maximum number of material palettes.

        Returns:
            WGSL source code string.
        """
        lines = [
            f"/// Material palette LUT array",
            f"@group(0) @binding(0) var material_palettes: "
            f"array<texture_1d<f32>, {max_materials}>;",
            f"@group(0) @binding(1) var palette_sampler: sampler;",
            f"",
            f"/// Look up color from material-specific palette.",
            f"///   material_id -- material identifier (0-{max_materials-1})",
            f"///   index       -- normalized index [0, 1]",
            f"///   returns     -- RGBA color",
            f"fn material_palette_lookup(material_id: u32, index: f32) -> vec4<f32> {{",
            f"    let safe_id = min(material_id, {max_materials - 1}u);",
            f"    return textureSample(material_palettes[safe_id], palette_sampler, index);",
            f"}}",
        ]

        return "\n".join(lines)


# =============================================================================
# Helper Functions
# =============================================================================

def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


def _lerp_color(c1: Color3, c2: Color3, t: float) -> Color3:
    """Linear interpolation between two colors."""
    return (
        _lerp(c1[0], c2[0], t),
        _lerp(c1[1], c2[1], t),
        _lerp(c1[2], c2[2], t),
    )


def _smoothstep(t: float) -> float:
    """Smooth Hermite interpolation (3t^2 - 2t^3)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _fade(t: float) -> float:
    """Perlin noise fade function (6t^5 - 15t^4 + 10t^3)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _fmt_float(val: float) -> str:
    """Format float for WGSL output."""
    if val == int(val) and not (val == 0.0 and str(val).startswith("-")):
        return f"{int(val)}.0"
    return f"{val}"


def _create_permutation_table(seed: int) -> List[int]:
    """Create a deterministic permutation table for noise generation."""
    perm = list(range(256))

    # Fisher-Yates shuffle with LCG
    state = seed
    for i in range(255, 0, -1):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        j = state % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]

    # Double to avoid wrapping
    return perm + perm


# =============================================================================
# FBM WGSL Code Generation
# =============================================================================

FBM_WGSL = """\
/// Simple hash function for noise
fn hash3(p: vec3<f32>) -> f32 {
    let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453);
}

/// 3D value noise
fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(
            mix(hash3(i + vec3<f32>(0.0, 0.0, 0.0)), hash3(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(hash3(i + vec3<f32>(0.0, 1.0, 0.0)), hash3(i + vec3<f32>(1.0, 1.0, 0.0)), u.x),
            u.y
        ),
        mix(
            mix(hash3(i + vec3<f32>(0.0, 0.0, 1.0)), hash3(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(hash3(i + vec3<f32>(0.0, 1.0, 1.0)), hash3(i + vec3<f32>(1.0, 1.0, 1.0)), u.x),
            u.y
        ),
        u.z
    ) * 2.0 - 1.0;
}

/// Fractal Brownian Motion (4 octaves)
fn fbm(p: vec3<f32>) -> f32 {
    var total = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_value = 0.0;

    for (var i = 0u; i < 4u; i = i + 1u) {
        total = total + noise3d(p * frequency) * amplitude;
        max_value = max_value + amplitude;
        frequency = frequency * 2.0;
        amplitude = amplitude * 0.5;
    }

    return total / max_value;
}
"""


def generate_palette_wgsl(
    terrain: Optional[TerrainPalette] = None,
    pattern: Optional[ProceduralPattern] = None,
    lut: Optional[PaletteLUT] = None,
    include_fbm: bool = True,
) -> str:
    """Generate complete WGSL module for procedural palettes.

    Args:
        terrain: Optional TerrainPalette for height-based coloring.
        pattern: Optional ProceduralPattern for procedural patterns.
        lut: Optional PaletteLUT for texture-based lookup.
        include_fbm: Include FBM noise functions.

    Returns:
        Complete WGSL source code string.
    """
    lines = [
        "// SPDX-License-Identifier: MIT",
        "//",
        "// Auto-generated procedural palette module.",
        "// T-DEMO-4.14, T-DEMO-4.15, T-DEMO-4.16",
        "//",
        "",
    ]

    # Add FBM functions if needed
    needs_fbm = pattern is not None and pattern.pattern_type in (
        PatternType.WOOD_GRAIN, PatternType.MARBLE, PatternType.RUST
    )
    if include_fbm and needs_fbm:
        lines.append(FBM_WGSL)
        lines.append("")

    # Add terrain palette
    if terrain is not None:
        lines.append(terrain.to_wgsl())
        lines.append("")

    # Add pattern
    if pattern is not None:
        lines.append(pattern.to_wgsl())
        lines.append("")

    # Add LUT
    if lut is not None:
        lines.append(lut.to_wgsl())
        lines.append("")

    return "\n".join(lines)
