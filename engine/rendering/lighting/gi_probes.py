"""Probe-based Global Illumination.

Implements probe-based GI from Section 6.4 of RENDERING_CONTEXT.md:
- Light Probes with Spherical Harmonics (SH) coefficients
- 3D Probe Grids
- Irradiance Volumes
- Baked Lightmaps
- Reflection Probes (baked/realtime cubemaps)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Optional

from engine.core.math.geometry import AABB, Sphere
from engine.core.math.mat import Mat4
from engine.core.math.vec import Vec2, Vec3, Vec4

if TYPE_CHECKING:
    pass


class ProbeType(Enum):
    """Types of GI probes."""
    LIGHT_PROBE = auto()      # SH-based diffuse
    REFLECTION_PROBE = auto()  # Cubemap-based specular
    IRRADIANCE_VOLUME = auto() # Volume of probes


class CaptureMode(Enum):
    """Probe capture modes from @reflection_probe decorator."""
    BAKED = "baked"
    REALTIME = "realtime"
    MIXED = "mixed"


@dataclass
class SphericalHarmonics:
    """Spherical Harmonics coefficients for irradiance encoding.

    Uses L2 (9 coefficients per color channel) for diffuse irradiance.
    Order: L0 (1), L1 (3), L2 (5) = 9 total per channel.

    Attributes:
        coefficients: 27 coefficients (9 per RGB channel)
    """
    coefficients: list[float] = field(
        default_factory=lambda: [0.0] * 27
    )

    @staticmethod
    def num_coefficients() -> int:
        """Number of coefficients for L2 SH."""
        return 27  # 9 per channel * 3 channels

    def evaluate(self, direction: Vec3) -> Vec3:
        """Evaluate SH in a given direction.

        Args:
            direction: Normalized direction vector

        Returns:
            Irradiance color
        """
        d = direction.normalized()
        c = self.coefficients

        # SH basis functions for L2
        # L0
        y0 = 0.282095  # 1/(2*sqrt(pi))

        # L1
        y1 = 0.488603 * d.y    # sqrt(3/(4*pi)) * y
        y2 = 0.488603 * d.z    # sqrt(3/(4*pi)) * z
        y3 = 0.488603 * d.x    # sqrt(3/(4*pi)) * x

        # L2
        y4 = 1.092548 * d.x * d.y        # sqrt(15/(4*pi)) * xy
        y5 = 1.092548 * d.y * d.z        # sqrt(15/(4*pi)) * yz
        y6 = 0.315392 * (3 * d.z * d.z - 1)  # sqrt(5/(16*pi)) * (3z^2 - 1)
        y7 = 1.092548 * d.x * d.z        # sqrt(15/(4*pi)) * xz
        y8 = 0.546274 * (d.x * d.x - d.y * d.y)  # sqrt(15/(16*pi)) * (x^2 - y^2)

        # Evaluate for each channel
        basis = [y0, y1, y2, y3, y4, y5, y6, y7, y8]

        r = sum(c[i] * basis[i] for i in range(9))
        g = sum(c[9 + i] * basis[i] for i in range(9))
        b = sum(c[18 + i] * basis[i] for i in range(9))

        return Vec3(max(0, r), max(0, g), max(0, b))

    def add_sample(self, direction: Vec3, color: Vec3, weight: float = 1.0) -> None:
        """Add a directional sample to the SH representation.

        Args:
            direction: Sample direction
            color: Sample color/radiance
            weight: Sample weight
        """
        d = direction.normalized()

        # Compute basis function values
        y0 = 0.282095
        y1 = 0.488603 * d.y
        y2 = 0.488603 * d.z
        y3 = 0.488603 * d.x
        y4 = 1.092548 * d.x * d.y
        y5 = 1.092548 * d.y * d.z
        y6 = 0.315392 * (3 * d.z * d.z - 1)
        y7 = 1.092548 * d.x * d.z
        y8 = 0.546274 * (d.x * d.x - d.y * d.y)

        basis = [y0, y1, y2, y3, y4, y5, y6, y7, y8]

        # Project sample onto SH basis for each channel
        for i in range(9):
            self.coefficients[i] += color.x * basis[i] * weight
            self.coefficients[9 + i] += color.y * basis[i] * weight
            self.coefficients[18 + i] += color.z * basis[i] * weight

    def scale(self, factor: float) -> None:
        """Scale all coefficients by a factor.

        Args:
            factor: Scale factor
        """
        for i in range(len(self.coefficients)):
            self.coefficients[i] *= factor

    def add(self, other: SphericalHarmonics) -> SphericalHarmonics:
        """Add another SH to this one.

        Args:
            other: SH to add

        Returns:
            New SH with summed coefficients
        """
        result = SphericalHarmonics()
        for i in range(len(self.coefficients)):
            result.coefficients[i] = self.coefficients[i] + other.coefficients[i]
        return result

    def lerp(self, other: SphericalHarmonics, t: float) -> SphericalHarmonics:
        """Linearly interpolate between two SH.

        Args:
            other: Target SH
            t: Interpolation factor [0, 1]

        Returns:
            Interpolated SH
        """
        result = SphericalHarmonics()
        for i in range(len(self.coefficients)):
            result.coefficients[i] = (
                self.coefficients[i] * (1 - t) + other.coefficients[i] * t
            )
        return result


@dataclass
class LightProbe:
    """Light probe storing SH coefficients for diffuse irradiance.

    Attributes:
        position: World position of the probe
        sh: Spherical harmonics coefficients
        radius: Influence radius
        priority: Priority for overlapping probes (higher = more important)
        valid: Whether the probe has been baked
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    sh: SphericalHarmonics = field(default_factory=SphericalHarmonics)
    radius: float = 10.0
    priority: int = 0
    valid: bool = False

    # Unique ID for the probe
    _probe_id: int = 0
    _id_counter: int = 0

    def __post_init__(self) -> None:
        LightProbe._id_counter += 1
        self._probe_id = LightProbe._id_counter

    def sample(self, direction: Vec3) -> Vec3:
        """Sample irradiance in a direction.

        Args:
            direction: Sample direction

        Returns:
            Irradiance color
        """
        return self.sh.evaluate(direction)

    def get_influence(self, point: Vec3) -> float:
        """Get influence weight at a point.

        Args:
            point: World position

        Returns:
            Influence weight [0, 1]
        """
        distance = self.position.distance(point)
        if distance >= self.radius:
            return 0.0

        # Smooth falloff
        t = distance / self.radius
        return 1.0 - t * t * (3.0 - 2.0 * t)

    def bake(
        self,
        sample_func: Callable[[Vec3], Vec3],
        sample_count: int = 64,
    ) -> None:
        """Bake the probe by sampling the environment.

        Args:
            sample_func: Function to sample radiance in a direction
            sample_count: Number of samples to take
        """
        self.sh = SphericalHarmonics()

        # Generate quasi-random directions using Fibonacci spiral
        golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0

        for i in range(sample_count):
            # Fibonacci spiral for uniform sphere sampling
            t = i / sample_count
            theta = 2.0 * math.pi * i / golden_ratio
            phi = math.acos(1.0 - 2.0 * t)

            direction = Vec3(
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi),
            )

            # Sample radiance and add to SH
            radiance = sample_func(direction)
            self.sh.add_sample(direction, radiance, 4.0 * math.pi / sample_count)

        self.valid = True


@dataclass
class ProbeGridConfig:
    """Configuration for a 3D probe grid.

    Attributes:
        resolution: Grid resolution (x, y, z)
        bounds: World-space bounds of the grid
        spacing: Distance between probes (computed from bounds/resolution)
    """
    resolution: tuple[int, int, int] = (8, 4, 8)
    bounds: AABB = field(
        default_factory=lambda: AABB(Vec3(-10, 0, -10), Vec3(10, 5, 10))
    )

    @property
    def spacing(self) -> Vec3:
        """Compute spacing between probes."""
        extent = self.bounds.max - self.bounds.min
        return Vec3(
            extent.x / max(1, self.resolution[0] - 1),
            extent.y / max(1, self.resolution[1] - 1),
            extent.z / max(1, self.resolution[2] - 1),
        )

    @property
    def probe_count(self) -> int:
        """Total number of probes in the grid."""
        return self.resolution[0] * self.resolution[1] * self.resolution[2]


class ProbeGrid:
    """3D grid of light probes for irradiance sampling.

    The grid provides efficient lookup and trilinear interpolation
    of irradiance at any point within the volume.
    """

    def __init__(self, config: ProbeGridConfig) -> None:
        """Initialize the probe grid.

        Args:
            config: Grid configuration
        """
        self.config = config
        self._probes: list[list[list[LightProbe]]] = []
        self._build_grid()

    def _build_grid(self) -> None:
        """Create probes at grid positions."""
        rx, ry, rz = self.config.resolution
        spacing = self.config.spacing
        origin = self.config.bounds.min

        self._probes = []
        for z in range(rz):
            z_slice = []
            for y in range(ry):
                y_row = []
                for x in range(rx):
                    position = Vec3(
                        origin.x + x * spacing.x,
                        origin.y + y * spacing.y,
                        origin.z + z * spacing.z,
                    )
                    probe = LightProbe(position=position)
                    y_row.append(probe)
                z_slice.append(y_row)
            self._probes.append(z_slice)

    def get_probe(self, x: int, y: int, z: int) -> Optional[LightProbe]:
        """Get a probe by grid index.

        Args:
            x: X index
            y: Y index
            z: Z index

        Returns:
            Probe at the index, or None if out of bounds
        """
        rx, ry, rz = self.config.resolution
        if 0 <= x < rx and 0 <= y < ry and 0 <= z < rz:
            return self._probes[z][y][x]
        return None

    def world_to_grid(self, point: Vec3) -> tuple[float, float, float]:
        """Convert world position to grid coordinates.

        Args:
            point: World position

        Returns:
            Grid coordinates (may be fractional)
        """
        local = point - self.config.bounds.min
        spacing = self.config.spacing

        return (
            local.x / spacing.x if spacing.x > 0 else 0,
            local.y / spacing.y if spacing.y > 0 else 0,
            local.z / spacing.z if spacing.z > 0 else 0,
        )

    def sample(self, point: Vec3, direction: Vec3) -> Vec3:
        """Sample irradiance at a point using trilinear interpolation.

        Args:
            point: World position
            direction: Normal direction for irradiance

        Returns:
            Interpolated irradiance
        """
        if not self.config.bounds.contains(point):
            return Vec3.zero()

        # Get grid coordinates
        gx, gy, gz = self.world_to_grid(point)
        rx, ry, rz = self.config.resolution

        # Clamp to valid range
        gx = max(0, min(gx, rx - 1.001))
        gy = max(0, min(gy, ry - 1.001))
        gz = max(0, min(gz, rz - 1.001))

        # Get integer indices and fractions
        x0, y0, z0 = int(gx), int(gy), int(gz)
        x1 = min(x0 + 1, rx - 1)
        y1 = min(y0 + 1, ry - 1)
        z1 = min(z0 + 1, rz - 1)

        fx = gx - x0
        fy = gy - y0
        fz = gz - z0

        # Sample 8 corner probes
        samples = []
        weights = [
            (1 - fx) * (1 - fy) * (1 - fz),
            fx * (1 - fy) * (1 - fz),
            (1 - fx) * fy * (1 - fz),
            fx * fy * (1 - fz),
            (1 - fx) * (1 - fy) * fz,
            fx * (1 - fy) * fz,
            (1 - fx) * fy * fz,
            fx * fy * fz,
        ]
        corners = [
            (x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (x1, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (x1, y1, z1),
        ]

        result = Vec3.zero()
        total_weight = 0.0

        for (cx, cy, cz), w in zip(corners, weights):
            probe = self.get_probe(cx, cy, cz)
            if probe and probe.valid and w > 0:
                irradiance = probe.sample(direction)
                result = result + irradiance * w
                total_weight += w

        if total_weight > 0:
            return result * (1.0 / total_weight)
        return Vec3.zero()

    def bake_all(
        self,
        sample_func: Callable[[Vec3, Vec3], Vec3],
        samples_per_probe: int = 64,
    ) -> None:
        """Bake all probes in the grid.

        Args:
            sample_func: Function(position, direction) -> radiance
            samples_per_probe: Samples per probe
        """
        for z_slice in self._probes:
            for y_row in z_slice:
                for probe in y_row:
                    probe.bake(
                        lambda d, p=probe.position: sample_func(p, d),
                        samples_per_probe,
                    )

    def iterate_probes(self):
        """Iterate over all probes in the grid.

        Yields:
            Each probe in the grid
        """
        for z_slice in self._probes:
            for y_row in z_slice:
                for probe in y_row:
                    yield probe


@dataclass
class IrradianceVolume:
    """Volume for storing and interpolating irradiance data.

    Wraps a ProbeGrid with additional blending and falloff options.

    Attributes:
        grid: Underlying probe grid
        blend_distance: Distance over which to blend at edges
        falloff_mode: How irradiance falls off ("none", "linear", "smooth")
    """
    grid: ProbeGrid = field(default_factory=lambda: ProbeGrid(ProbeGridConfig()))
    blend_distance: float = 1.0
    falloff_mode: str = "smooth"

    def sample(self, point: Vec3, normal: Vec3) -> Vec3:
        """Sample irradiance with edge blending.

        Args:
            point: World position
            normal: Surface normal

        Returns:
            Blended irradiance
        """
        bounds = self.grid.config.bounds

        # Compute distance to volume boundary
        edge_distance = min(
            point.x - bounds.min.x, bounds.max.x - point.x,
            point.y - bounds.min.y, bounds.max.y - point.y,
            point.z - bounds.min.z, bounds.max.z - point.z,
        )

        # Sample grid
        irradiance = self.grid.sample(point, normal)

        # Apply edge falloff
        # Guard against division by zero with minimum blend distance
        min_blend_distance = 0.001  # GIProbeConstants.MIN_BLEND_DISTANCE
        safe_blend_distance = max(self.blend_distance, min_blend_distance)
        if edge_distance < safe_blend_distance:
            t = edge_distance / safe_blend_distance
            if self.falloff_mode == "smooth":
                t = t * t * (3.0 - 2.0 * t)
            elif self.falloff_mode == "linear":
                pass  # t is already linear
            irradiance = irradiance * t

        return irradiance


@dataclass
class LightmapTexel:
    """A single texel in a baked lightmap.

    Attributes:
        irradiance: Baked irradiance color
        direction: Dominant light direction (for directional lightmaps)
        validity: Validity mask (for filtering invalid texels)
    """
    irradiance: Vec3 = field(default_factory=Vec3.zero)
    direction: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    validity: float = 0.0


@dataclass
class BakedLightmap:
    """Per-texel irradiance stored as a texture.

    Attributes:
        width: Lightmap width in texels
        height: Lightmap height in texels
        texels: 2D array of lightmap texels
        directional: Whether to store directional information
    """
    width: int = 256
    height: int = 256
    texels: list[list[LightmapTexel]] = field(default_factory=list)
    directional: bool = False

    def __post_init__(self) -> None:
        if not self.texels:
            self.texels = [
                [LightmapTexel() for _ in range(self.width)]
                for _ in range(self.height)
            ]

    def sample(self, uv: Vec2) -> Vec3:
        """Sample the lightmap at UV coordinates.

        Args:
            uv: Texture coordinates [0, 1]

        Returns:
            Irradiance at the sample location
        """
        # Bilinear sampling
        x = uv.x * (self.width - 1)
        y = uv.y * (self.height - 1)

        x0, y0 = int(x), int(y)
        x1 = min(x0 + 1, self.width - 1)
        y1 = min(y0 + 1, self.height - 1)

        fx = x - x0
        fy = y - y0

        t00 = self.texels[y0][x0]
        t10 = self.texels[y0][x1]
        t01 = self.texels[y1][x0]
        t11 = self.texels[y1][x1]

        # Bilinear interpolation
        result = (
            t00.irradiance * (1 - fx) * (1 - fy) +
            t10.irradiance * fx * (1 - fy) +
            t01.irradiance * (1 - fx) * fy +
            t11.irradiance * fx * fy
        )

        return result

    def set_texel(self, x: int, y: int, irradiance: Vec3, direction: Vec3 = None) -> None:
        """Set a texel's irradiance.

        Args:
            x: X coordinate
            y: Y coordinate
            irradiance: Irradiance color
            direction: Optional dominant direction
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.texels[y][x].irradiance = irradiance
            self.texels[y][x].validity = 1.0
            if direction:
                self.texels[y][x].direction = direction


@dataclass
class ReflectionProbeConfig:
    """Configuration from @reflection_probe decorator.

    Attributes:
        capture_mode: Capture mode (baked/realtime/mixed)
        resolution: Cubemap resolution
        update_rate: Update rate in Hz for realtime probes
        importance: Priority weight for blending (higher = more influence)
        box_extents: Influence box half-extents (full size is 2x this)
        inner_radius: Inner blend zone radius (full intensity inside)
        outer_radius: Outer influence radius (zero intensity outside)
        roughness_levels: Number of pre-filtered mip levels for roughness
        capture_lod_bias: LOD bias applied during cubemap capture
        include_layers: Bitmask of layers to include in capture
        exclude_actors: List of actor tags to exclude from capture
    """
    capture_mode: CaptureMode = CaptureMode.BAKED
    resolution: int = 256
    update_rate: float = 0.0
    importance: float = 1.0
    box_extents: Vec3 = field(default_factory=lambda: Vec3(10.0, 10.0, 10.0))
    inner_radius: float = 0.0
    outer_radius: float = 10.0
    roughness_levels: int = 8
    capture_lod_bias: float = 0.0
    include_layers: int = 0xFFFFFFFF
    exclude_actors: list = field(default_factory=list)


@dataclass
class ReflectionProbe:
    """Reflection probe using environment cubemap.

    Stores a cubemap for specular reflections with parallax correction.

    Attributes:
        position: World position of the probe
        config: Probe configuration
        bounds: Influence bounds (for parallax correction)
        blend_distance: Blending distance at edges
        cubemap_data: Cubemap face data (6 faces x resolution x resolution)
        box_min: Minimum corner of parallax correction box (computed)
        box_max: Maximum corner of parallax correction box (computed)
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    config: ReflectionProbeConfig = field(default_factory=ReflectionProbeConfig)
    bounds: AABB = field(
        default_factory=lambda: AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5))
    )
    blend_distance: float = 1.0

    # Cubemap data placeholder (6 faces)
    cubemap_data: list = field(default_factory=list)

    # Parallax correction bounding box (computed from config.box_extents)
    box_min: Vec3 = field(default_factory=Vec3.zero)
    box_max: Vec3 = field(default_factory=Vec3.zero)

    # Probe state
    _dirty: bool = True
    _last_update_time: float = 0.0
    _probe_id: int = 0
    _id_counter: int = 0

    def __post_init__(self) -> None:
        ReflectionProbe._id_counter += 1
        self._probe_id = ReflectionProbe._id_counter
        # Compute parallax correction box from config.box_extents
        extents = self.config.box_extents
        self.box_min = Vec3(
            self.position.x - extents.x,
            self.position.y - extents.y,
            self.position.z - extents.z,
        )
        self.box_max = Vec3(
            self.position.x + extents.x,
            self.position.y + extents.y,
            self.position.z + extents.z,
        )

    @property
    def needs_update(self) -> bool:
        """Check if the probe needs to be updated."""
        if self.config.capture_mode == CaptureMode.BAKED:
            return self._dirty
        return True  # Realtime probes always need updates

    def sample(
        self,
        world_pos: Vec3,
        reflection_dir: Vec3,
        roughness: float = 0.0,
    ) -> Vec3:
        """Sample the reflection probe with parallax correction.

        Args:
            world_pos: World position of the shading point
            reflection_dir: Reflection direction
            roughness: Surface roughness (for mip level selection)

        Returns:
            Reflected color
        """
        # Apply box parallax correction
        corrected_dir = self._parallax_correct(world_pos, reflection_dir)

        # In production: sample cubemap at corrected direction
        # with mip level based on roughness
        # mip_level = roughness * max_mip_levels

        # Placeholder: return a default color
        # Real implementation would sample the cubemap texture
        return Vec3(0.5, 0.5, 0.5)

    def _parallax_correct(self, world_pos: Vec3, direction: Vec3) -> Vec3:
        """Apply box parallax correction.

        Args:
            world_pos: World position
            direction: Reflection direction

        Returns:
            Corrected reflection direction
        """
        # Ray-box intersection for parallax correction
        inv_dir = Vec3(
            1.0 / direction.x if abs(direction.x) > 1e-6 else 1e6,
            1.0 / direction.y if abs(direction.y) > 1e-6 else 1e6,
            1.0 / direction.z if abs(direction.z) > 1e-6 else 1e6,
        )

        t_min = Vec3(
            (self.bounds.min.x - world_pos.x) * inv_dir.x,
            (self.bounds.min.y - world_pos.y) * inv_dir.y,
            (self.bounds.min.z - world_pos.z) * inv_dir.z,
        )
        t_max = Vec3(
            (self.bounds.max.x - world_pos.x) * inv_dir.x,
            (self.bounds.max.y - world_pos.y) * inv_dir.y,
            (self.bounds.max.z - world_pos.z) * inv_dir.z,
        )

        t1 = Vec3(min(t_min.x, t_max.x), min(t_min.y, t_max.y), min(t_min.z, t_max.z))
        t2 = Vec3(max(t_min.x, t_max.x), max(t_min.y, t_max.y), max(t_min.z, t_max.z))

        t_near = max(t1.x, max(t1.y, t1.z))
        t_far = min(t2.x, min(t2.y, t2.z))

        if t_far < t_near or t_far < 0:
            return direction

        # Use far intersection point
        intersection = world_pos + direction * t_far

        # Corrected direction from probe center to intersection
        return (intersection - self.position).normalized()

    def get_blend_factor(self, point: Vec3) -> float:
        """Get blending factor for a point.

        Args:
            point: World position

        Returns:
            Blend factor [0, 1]
        """
        if not self.bounds.contains(point):
            return 0.0

        # Distance to boundary
        edge_dist = min(
            point.x - self.bounds.min.x, self.bounds.max.x - point.x,
            point.y - self.bounds.min.y, self.bounds.max.y - point.y,
            point.z - self.bounds.min.z, self.bounds.max.z - point.z,
        )

        # Guard against division by zero
        min_blend_distance = 0.001  # GIProbeConstants.MIN_BLEND_DISTANCE
        safe_blend_distance = max(self.blend_distance, min_blend_distance)

        if edge_dist >= safe_blend_distance:
            return 1.0

        return edge_dist / safe_blend_distance

    def mark_dirty(self) -> None:
        """Mark the probe as needing re-capture."""
        self._dirty = True

    def clear_dirty(self) -> None:
        """Clear the dirty flag."""
        self._dirty = False


def reflection_probe(
    capture_mode: str = "baked",
    resolution: int = 256,
    update_rate: float = 0.0,
    importance: float = 1.0,
    box_extents: Optional[Vec3] = None,
    inner_radius: float = 0.0,
    outer_radius: float = 10.0,
    roughness_levels: int = 8,
    capture_lod_bias: float = 0.0,
    include_layers: int = 0xFFFFFFFF,
    exclude_actors: Optional[list] = None,
):
    """Decorator to configure reflection probes.

    Args:
        capture_mode: Capture mode ("baked", "realtime", "mixed")
        resolution: Cubemap resolution
        update_rate: Update rate for realtime probes
        importance: Priority weight for blending (higher = more influence)
        box_extents: Influence box half-extents (defaults to 10x10x10)
        inner_radius: Inner blend zone radius (full intensity inside)
        outer_radius: Outer influence radius (zero intensity outside)
        roughness_levels: Number of pre-filtered mip levels for roughness
        capture_lod_bias: LOD bias applied during cubemap capture
        include_layers: Bitmask of layers to include in capture
        exclude_actors: List of actor tags to exclude from capture

    Returns:
        Decorated class
    """
    def decorator(cls):
        config = ReflectionProbeConfig(
            capture_mode=CaptureMode(capture_mode),
            resolution=resolution,
            update_rate=update_rate,
            importance=importance,
            box_extents=box_extents if box_extents is not None else Vec3(10.0, 10.0, 10.0),
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            roughness_levels=roughness_levels,
            capture_lod_bias=capture_lod_bias,
            include_layers=include_layers,
            exclude_actors=exclude_actors if exclude_actors is not None else [],
        )
        cls._reflection_probe = True
        cls._reflection_capture_mode = CaptureMode(capture_mode)
        cls._reflection_resolution = resolution
        cls._reflection_update_rate = update_rate
        cls._reflection_importance = importance
        cls._reflection_box_extents = config.box_extents
        cls._reflection_inner_radius = inner_radius
        cls._reflection_outer_radius = outer_radius
        cls._reflection_roughness_levels = roughness_levels
        cls._reflection_capture_lod_bias = capture_lod_bias
        cls._reflection_include_layers = include_layers
        cls._reflection_exclude_actors = config.exclude_actors
        cls._reflection_config = config
        return cls
    return decorator
