"""Offline baker for static light probes into SH coefficients.

This module provides a complete pipeline for baking static light probes:
1. Generate stratified ray directions (Fibonacci spiral)
2. Trace rays against scene geometry
3. Project gathered radiance into SH coefficients
4. Export to KTX2 or TRINITY probe format

The baker integrates with the SH library from T-GIR-P1.1.

Usage:
    uv run python -m engine.tools.probe_baker \\
        --scene scene.gltf \\
        --output probes.ktx2 \\
        --rays 1024 \\
        --bounces 3

References:
    - Ramamoorthi & Hanrahan, "An Efficient Representation for Irradiance
      Environment Maps", SIGGRAPH 2001
    - Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from engine.rendering.gi.sh_math import (
    SHCoefficientsL2,
    fibonacci_sphere_directions,
    sh_convolve_irradiance,
    sh_project_l2,
)

# ============================================================================
# Constants
# ============================================================================

# TRINITY probe format magic number: "TPRB" in little-endian
TPROBE_MAGIC = 0x42525054  # 'TPRB'
TPROBE_VERSION = 1

# LZ4 compression block size
LZ4_BLOCK_SIZE = 65536

# KTX2 format identifiers
KTX2_MAGIC = bytes([0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A])
KTX2_VK_FORMAT_R32G32B32A32_SFLOAT = 109


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class BakeConfig:
    """Configuration for probe baking.

    Attributes:
        rays_per_probe: Number of Monte Carlo samples per probe.
        max_bounces: Maximum number of light bounces for indirect lighting.
        output_format: Output format ("ktx2" or "trinity_probe").
        compress: Whether to apply LZ4 compression to output.
        sky_color: Default sky/environment color for rays that don't hit geometry.
        min_distance: Minimum ray distance to prevent self-intersection.
        max_distance: Maximum ray trace distance.
    """

    rays_per_probe: int = 512
    max_bounces: int = 3
    output_format: str = "ktx2"
    compress: bool = True
    sky_color: tuple[float, float, float] = (0.5, 0.7, 1.0)
    min_distance: float = 0.001
    max_distance: float = 1000.0


@dataclass
class ProbeLocation:
    """A probe placement in the scene.

    Attributes:
        position: World-space position (x, y, z).
        name: Optional identifier for the probe.
    """

    position: tuple[float, float, float]
    name: str = ""


@dataclass
class HitInfo:
    """Ray hit information from scene tracing.

    Attributes:
        position: World-space hit position.
        normal: Surface normal at hit point.
        distance: Ray travel distance.
        albedo: Surface albedo/color at hit point.
        emissive: Emissive contribution at hit point.
    """

    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    distance: float
    albedo: tuple[float, float, float] = (0.5, 0.5, 0.5)
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class BakedProbe:
    """Result of baking a single probe.

    Attributes:
        location: Original probe placement.
        irradiance: SH coefficients representing irradiance.
        distance_mean: Mean hit distance (for visibility encoding).
        distance_variance: Variance of hit distances.
        visibility: SH coefficients for visibility (optional).
    """

    location: ProbeLocation
    irradiance: SHCoefficientsL2
    distance_mean: float
    distance_variance: float
    visibility: Optional[SHCoefficientsL2] = None


@dataclass
class BakeResult:
    """Result of a complete bake operation.

    Attributes:
        probes: List of baked probes.
        total_rays: Total number of rays traced.
        elapsed_seconds: Time spent baking.
        config: Configuration used for baking.
    """

    probes: list[BakedProbe] = field(default_factory=list)
    total_rays: int = 0
    elapsed_seconds: float = 0.0
    config: Optional[BakeConfig] = None


# ============================================================================
# Scene Protocol
# ============================================================================


@runtime_checkable
class SceneProxy(Protocol):
    """Protocol for scene ray tracing.

    Implementations provide ray-scene intersection for probe baking.
    """

    def trace_ray(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
    ) -> Optional[HitInfo]:
        """Trace a ray against the scene.

        Args:
            origin: Ray origin in world space.
            direction: Normalized ray direction.

        Returns:
            HitInfo if ray intersects geometry, None otherwise.
        """
        ...


class EmptyScene:
    """Empty scene that returns only sky color.

    Useful for testing and as a fallback.
    """

    def __init__(self, sky_color: tuple[float, float, float] = (0.5, 0.7, 1.0)) -> None:
        self.sky_color = sky_color

    def trace_ray(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
    ) -> Optional[HitInfo]:
        """Always returns None (no geometry)."""
        return None


class GroundPlaneScene:
    """Simple scene with an infinite ground plane for testing.

    The ground plane is at y=0 with configurable albedo.
    """

    def __init__(
        self,
        ground_albedo: tuple[float, float, float] = (0.3, 0.5, 0.2),
        sky_color: tuple[float, float, float] = (0.5, 0.7, 1.0),
    ) -> None:
        self.ground_albedo = ground_albedo
        self.sky_color = sky_color

    def trace_ray(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
    ) -> Optional[HitInfo]:
        """Trace ray against y=0 ground plane."""
        # Ray: p = origin + t * direction
        # Plane: y = 0
        # Solve: origin.y + t * direction.y = 0
        ox, oy, oz = origin
        dx, dy, dz = direction

        if abs(dy) < 1e-8:
            return None  # Ray parallel to plane

        t = -oy / dy
        if t < 0.001:
            return None  # Behind ray origin

        hit_x = ox + t * dx
        hit_z = oz + t * dz

        return HitInfo(
            position=(hit_x, 0.0, hit_z),
            normal=(0.0, 1.0, 0.0),
            distance=t,
            albedo=self.ground_albedo,
            emissive=(0.0, 0.0, 0.0),
        )


# ============================================================================
# Probe Baker
# ============================================================================


class ProbeBaker:
    """Offline baker for static light probes.

    Traces rays from probe positions and projects gathered radiance
    into spherical harmonic coefficients for efficient runtime evaluation.
    """

    def __init__(self, config: BakeConfig) -> None:
        """Initialize the probe baker.

        Args:
            config: Bake configuration settings.
        """
        self.config = config
        self._cached_directions: Optional[NDArray[np.float32]] = None
        self._cached_ray_count: int = 0

    def generate_ray_directions(self, n: int) -> NDArray[np.float32]:
        """Generate stratified spherical samples using Fibonacci spiral.

        The Fibonacci lattice provides approximately uniform coverage
        of the sphere with low discrepancy, making it ideal for
        Monte Carlo integration.

        Args:
            n: Number of ray directions to generate.

        Returns:
            Array of normalized directions, shape (n, 3).
        """
        # Use cached directions if available
        if self._cached_directions is not None and self._cached_ray_count == n:
            return self._cached_directions

        self._cached_directions = fibonacci_sphere_directions(n)
        self._cached_ray_count = n
        return self._cached_directions

    def trace_probe(
        self,
        position: tuple[float, float, float],
        scene: SceneProxy,
    ) -> BakedProbe:
        """Trace rays from probe position and project into SH.

        Performs Monte Carlo integration by:
        1. Generating stratified ray directions
        2. Tracing each ray against the scene
        3. Accumulating radiance contributions
        4. Projecting into SH coefficients
        5. Computing visibility statistics

        Args:
            position: World-space probe position.
            scene: Scene proxy for ray tracing.

        Returns:
            BakedProbe with irradiance SH and visibility data.
        """
        directions = self.generate_ray_directions(self.config.rays_per_probe)
        num_rays = len(directions)

        # Accumulators
        irradiance = SHCoefficientsL2.zero()
        distances: list[float] = []

        sky_color = np.array(self.config.sky_color, dtype=np.float32)

        for direction in directions:
            dir_tuple = (float(direction[0]), float(direction[1]), float(direction[2]))

            # Trace ray with bounce accumulation
            radiance = self._trace_path(
                position, dir_tuple, scene, self.config.max_bounces
            )

            # Project sample into SH
            sample_sh = sh_project_l2(direction, radiance)
            irradiance.add(sample_sh)

            # Record distance for visibility
            hit = scene.trace_ray(position, dir_tuple)
            if hit is not None:
                distances.append(hit.distance)
            else:
                distances.append(self.config.max_distance)

        # Scale by solid angle: 4*PI / num_samples
        irradiance.scale(4.0 * np.pi / num_rays)

        # Convolve with cosine lobe for irradiance
        irradiance = sh_convolve_irradiance(irradiance)

        # Compute distance statistics
        distances_arr = np.array(distances, dtype=np.float32)
        distance_mean = float(np.mean(distances_arr))
        distance_variance = float(np.var(distances_arr))

        return BakedProbe(
            location=ProbeLocation(position=position),
            irradiance=irradiance,
            distance_mean=distance_mean,
            distance_variance=distance_variance,
        )

    def _trace_path(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        scene: SceneProxy,
        bounces_remaining: int,
    ) -> NDArray[np.float32]:
        """Trace a path through the scene with multiple bounces.

        Args:
            origin: Ray origin.
            direction: Ray direction.
            scene: Scene proxy.
            bounces_remaining: Number of bounces left.

        Returns:
            Accumulated radiance along the path.
        """
        hit = scene.trace_ray(origin, direction)

        if hit is None:
            # Return sky color
            return np.array(self.config.sky_color, dtype=np.float32)

        # Start with emissive contribution
        radiance = np.array(hit.emissive, dtype=np.float32)

        if bounces_remaining > 0:
            # Generate bounce direction (cosine-weighted hemisphere)
            bounce_dir = self._cosine_hemisphere_sample(hit.normal)

            # Offset origin to prevent self-intersection
            bounce_origin = (
                hit.position[0] + hit.normal[0] * self.config.min_distance,
                hit.position[1] + hit.normal[1] * self.config.min_distance,
                hit.position[2] + hit.normal[2] * self.config.min_distance,
            )

            # Recursive trace
            indirect = self._trace_path(
                bounce_origin, bounce_dir, scene, bounces_remaining - 1
            )

            # Apply BRDF (Lambertian: albedo / PI, but cosine term cancels with PDF)
            albedo = np.array(hit.albedo, dtype=np.float32)
            radiance += albedo * indirect

        return radiance

    def _cosine_hemisphere_sample(
        self, normal: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Generate cosine-weighted random direction in hemisphere.

        Args:
            normal: Surface normal defining the hemisphere.

        Returns:
            Random direction in the hemisphere.
        """
        # Generate random numbers
        u1 = np.random.random()
        u2 = np.random.random()

        # Cosine-weighted hemisphere sampling
        r = np.sqrt(u1)
        theta = 2.0 * np.pi * u2

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.sqrt(max(0.0, 1.0 - u1))

        # Build tangent frame
        n = np.array(normal, dtype=np.float32)
        n = n / np.linalg.norm(n)

        # Choose tangent not parallel to normal
        if abs(n[0]) < 0.9:
            tangent = np.cross(n, np.array([1, 0, 0], dtype=np.float32))
        else:
            tangent = np.cross(n, np.array([0, 1, 0], dtype=np.float32))
        tangent = tangent / np.linalg.norm(tangent)
        bitangent = np.cross(n, tangent)

        # Transform to world space
        world_dir = x * tangent + y * bitangent + z * n
        world_dir = world_dir / np.linalg.norm(world_dir)

        return (float(world_dir[0]), float(world_dir[1]), float(world_dir[2]))

    def bake_probes(
        self,
        locations: list[ProbeLocation],
        scene: SceneProxy,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[BakedProbe]:
        """Bake all probes with progress reporting.

        Args:
            locations: List of probe placements.
            scene: Scene proxy for ray tracing.
            progress_callback: Optional callback(current, total, message).

        Returns:
            List of baked probes.
        """
        results: list[BakedProbe] = []
        total = len(locations)

        for i, location in enumerate(locations):
            if progress_callback:
                name = location.name or f"probe_{i}"
                progress_callback(i, total, f"Baking {name}")

            probe = self.trace_probe(location.position, scene)
            probe.location = location
            results.append(probe)

        if progress_callback:
            progress_callback(total, total, "Complete")

        return results

    def bake_probes_iter(
        self,
        locations: list[ProbeLocation],
        scene: SceneProxy,
    ) -> Iterator[BakedProbe]:
        """Bake probes with iterator interface for streaming.

        Args:
            locations: List of probe placements.
            scene: Scene proxy for ray tracing.

        Yields:
            BakedProbe for each location.
        """
        for location in locations:
            probe = self.trace_probe(location.position, scene)
            probe.location = location
            yield probe

    # ========================================================================
    # Export Methods
    # ========================================================================

    def export_ktx2(self, probes: list[BakedProbe], path: Path) -> None:
        """Export probes to KTX2 format (cubemap atlas).

        Creates a KTX2 file containing probe SH coefficients as a
        1D texture array, where each texel holds one RGB coefficient.

        Format:
            - Width: 9 (one per SH coefficient)
            - Height: 1
            - Depth: len(probes) (array layers)
            - Format: R32G32B32A32_SFLOAT

        Args:
            probes: List of baked probes.
            path: Output file path.
        """
        if not probes:
            raise ValueError("No probes to export")

        # Build coefficient data: shape (num_probes, 9, 4)
        num_probes = len(probes)
        data = np.zeros((num_probes, 9, 4), dtype=np.float32)

        for i, probe in enumerate(probes):
            for j in range(9):
                rgb = probe.irradiance.coeffs[j]
                data[i, j, :3] = rgb
                data[i, j, 3] = 0.0  # Padding

        # Write KTX2 file
        with open(path, "wb") as f:
            # KTX2 header
            f.write(KTX2_MAGIC)

            # Format (VK_FORMAT_R32G32B32A32_SFLOAT)
            f.write(struct.pack("<I", KTX2_VK_FORMAT_R32G32B32A32_SFLOAT))

            # Type size (16 bytes per texel)
            f.write(struct.pack("<I", 16))

            # Dimensions: width=9, height=1, depth=0 (2D array)
            f.write(struct.pack("<III", 9, 1, 0))

            # Layer count, face count, level count
            f.write(struct.pack("<III", num_probes, 1, 1))

            # Supercompression scheme (0 = none)
            f.write(struct.pack("<I", 0))

            # DFD byte offset and length (simplified: no DFD)
            f.write(struct.pack("<II", 0, 0))

            # KVD byte offset and length (simplified: no KVD)
            f.write(struct.pack("<II", 0, 0))

            # SGD byte offset and length (simplified: no SGD)
            f.write(struct.pack("<QQ", 0, 0))

            # Level index (single level)
            level_offset = f.tell() + 24  # After this entry
            level_size = data.nbytes
            f.write(struct.pack("<QQQ", level_offset, level_size, level_size))

            # Pixel data
            f.write(data.tobytes())

    def export_trinity(self, probes: list[BakedProbe], path: Path) -> None:
        """Export probes to TRINITY custom format (.tprobe).

        Binary format:
            Header (16 bytes):
                - magic: u32 = 0x42525054 ('TPRB')
                - version: u32 = 1
                - probe_count: u32
                - sh_order: u32 = 2 (L2)

            Per probe (192 bytes):
                - position: f32[3] (12 bytes)
                - irradiance: f32[9][3] (108 bytes) - RGB SH coefficients
                - visibility: f32[9] (36 bytes) - Mono SH for distance
                - distance_mean: f32 (4 bytes)
                - distance_variance: f32 (4 bytes)
                - padding: f32[7] (28 bytes) - Align to 192

        Optional LZ4 compression applied to probe data.

        Args:
            probes: List of baked probes.
            path: Output file path.
        """
        if not probes:
            raise ValueError("No probes to export")

        # Build binary data
        header = struct.pack(
            "<IIII",
            TPROBE_MAGIC,
            TPROBE_VERSION,
            len(probes),
            2,  # SH order (L2)
        )

        probe_data = bytearray()
        for probe in probes:
            # Position (12 bytes)
            px, py, pz = probe.location.position
            probe_data.extend(struct.pack("<fff", px, py, pz))

            # Irradiance SH (108 bytes = 9 * 3 * 4)
            for i in range(9):
                r, g, b = probe.irradiance.coeffs[i]
                probe_data.extend(struct.pack("<fff", r, g, b))

            # Visibility SH (36 bytes = 9 * 4) - use irradiance luminance for now
            for i in range(9):
                r, g, b = probe.irradiance.coeffs[i]
                # Luminance approximation
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                probe_data.extend(struct.pack("<f", lum))

            # Distance stats (8 bytes)
            probe_data.extend(struct.pack("<ff", probe.distance_mean, probe.distance_variance))

            # Padding to 192 bytes (28 bytes)
            probe_data.extend(b"\x00" * 28)

        # Apply compression if requested
        if self.config.compress:
            try:
                import lz4.frame

                compressed = lz4.frame.compress(bytes(probe_data))
                # Update header to indicate compression
                header = struct.pack(
                    "<IIII",
                    TPROBE_MAGIC | 0x80000000,  # High bit = compressed
                    TPROBE_VERSION,
                    len(probes),
                    2,
                )
                probe_data = bytearray(compressed)
            except ImportError:
                pass  # Fall back to uncompressed

        # Write file
        with open(path, "wb") as f:
            f.write(header)
            f.write(probe_data)

    @staticmethod
    def load_trinity(path: Path) -> list[BakedProbe]:
        """Load probes from TRINITY custom format (.tprobe).

        Args:
            path: Input file path.

        Returns:
            List of loaded probes.
        """
        with open(path, "rb") as f:
            # Read header
            header = f.read(16)
            magic, version, probe_count, sh_order = struct.unpack("<IIII", header)

            # Check magic (with or without compression bit)
            is_compressed = (magic & 0x80000000) != 0
            base_magic = magic & 0x7FFFFFFF
            if base_magic != TPROBE_MAGIC:
                raise ValueError(f"Invalid TPROBE magic: {hex(magic)}")

            if version != TPROBE_VERSION:
                raise ValueError(f"Unsupported TPROBE version: {version}")

            if sh_order != 2:
                raise ValueError(f"Only L2 SH supported, got order {sh_order}")

            # Read probe data
            probe_data = f.read()

            if is_compressed:
                try:
                    import lz4.frame

                    probe_data = lz4.frame.decompress(probe_data)
                except ImportError as e:
                    raise ImportError("LZ4 required for compressed TPROBE") from e

            # Parse probes
            probes: list[BakedProbe] = []
            offset = 0
            probe_size = 192

            for _ in range(probe_count):
                if offset + probe_size > len(probe_data):
                    raise ValueError("Truncated probe data")

                # Position
                px, py, pz = struct.unpack_from("<fff", probe_data, offset)
                offset += 12

                # Irradiance
                irradiance = SHCoefficientsL2.zero()
                for i in range(9):
                    r, g, b = struct.unpack_from("<fff", probe_data, offset)
                    irradiance.coeffs[i] = [r, g, b]
                    offset += 12

                # Skip visibility (36 bytes)
                offset += 36

                # Distance stats
                distance_mean, distance_variance = struct.unpack_from("<ff", probe_data, offset)
                offset += 8

                # Skip padding
                offset += 28

                probes.append(
                    BakedProbe(
                        location=ProbeLocation(position=(px, py, pz)),
                        irradiance=irradiance,
                        distance_mean=distance_mean,
                        distance_variance=distance_variance,
                    )
                )

            return probes


# ============================================================================
# Probe Grid Generation
# ============================================================================


def generate_probe_grid(
    min_bounds: tuple[float, float, float],
    max_bounds: tuple[float, float, float],
    spacing: float,
) -> list[ProbeLocation]:
    """Generate a uniform grid of probe locations.

    Args:
        min_bounds: Minimum corner of the grid volume.
        max_bounds: Maximum corner of the grid volume.
        spacing: Distance between probes.

    Returns:
        List of probe locations.
    """
    probes: list[ProbeLocation] = []
    idx = 0

    x = min_bounds[0]
    while x <= max_bounds[0]:
        y = min_bounds[1]
        while y <= max_bounds[1]:
            z = min_bounds[2]
            while z <= max_bounds[2]:
                probes.append(ProbeLocation(position=(x, y, z), name=f"probe_{idx}"))
                idx += 1
                z += spacing
            y += spacing
        x += spacing

    return probes


# ============================================================================
# CLI Interface
# ============================================================================


def main(args: Optional[list[str]] = None) -> int:
    """Command-line interface for probe baking.

    Args:
        args: Command-line arguments (default: sys.argv[1:]).

    Returns:
        Exit code (0 = success).
    """
    parser = argparse.ArgumentParser(
        description="Bake static light probes into SH coefficients.",
        prog="python -m engine.tools.probe_baker",
    )

    parser.add_argument(
        "--scene",
        type=Path,
        help="Input scene file (GLTF/GLB)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output file path (.ktx2 or .tprobe)",
    )

    parser.add_argument(
        "--rays",
        "-r",
        type=int,
        default=512,
        help="Rays per probe (default: 512)",
    )

    parser.add_argument(
        "--bounces",
        "-b",
        type=int,
        default=3,
        help="Maximum light bounces (default: 3)",
    )

    parser.add_argument(
        "--grid-min",
        type=float,
        nargs=3,
        default=[-5, 0.5, -5],
        metavar=("X", "Y", "Z"),
        help="Minimum corner of probe grid",
    )

    parser.add_argument(
        "--grid-max",
        type=float,
        nargs=3,
        default=[5, 3, 5],
        metavar=("X", "Y", "Z"),
        help="Maximum corner of probe grid",
    )

    parser.add_argument(
        "--spacing",
        "-s",
        type=float,
        default=2.0,
        help="Probe grid spacing (default: 2.0)",
    )

    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable LZ4 compression",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parsed = parser.parse_args(args)

    # Determine output format
    output_format = "ktx2"
    if parsed.output.suffix.lower() == ".tprobe":
        output_format = "trinity_probe"

    # Create config
    config = BakeConfig(
        rays_per_probe=parsed.rays,
        max_bounces=parsed.bounces,
        output_format=output_format,
        compress=not parsed.no_compress,
    )

    # Create baker
    baker = ProbeBaker(config)

    # Generate probe grid
    locations = generate_probe_grid(
        tuple(parsed.grid_min),
        tuple(parsed.grid_max),
        parsed.spacing,
    )

    if parsed.verbose:
        print(f"Generated {len(locations)} probe locations")
        print(f"Rays per probe: {config.rays_per_probe}")
        print(f"Max bounces: {config.max_bounces}")

    # Use test scene for now (real scene loading would go here)
    scene = GroundPlaneScene()

    # Progress callback
    def progress(current: int, total: int, message: str) -> None:
        if parsed.verbose:
            print(f"[{current}/{total}] {message}")

    # Bake probes
    probes = baker.bake_probes(locations, scene, progress)

    # Export
    if output_format == "ktx2":
        baker.export_ktx2(probes, parsed.output)
    else:
        baker.export_trinity(probes, parsed.output)

    if parsed.verbose:
        print(f"Wrote {len(probes)} probes to {parsed.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
