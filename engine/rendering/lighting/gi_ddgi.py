"""Dynamic Diffuse Global Illumination (DDGI).

Implements DDGI from Section 6.4 of RENDERING_CONTEXT.md:
- DDGIProbe: Ray-traced probe with irradiance and visibility
- DDGIProbeGrid: Grid of DDGI probes
- DDGIUpdatePass: Ray tracing and irradiance update
- DDGILookup: Trilinear interpolation at shading points
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Optional

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec2, Vec3, Vec4

if TYPE_CHECKING:
    pass


class DDGIProbeState(Enum):
    """State of a DDGI probe."""
    INACTIVE = auto()      # Probe is not being updated
    ACTIVE = auto()        # Probe is actively being updated
    SLEEPING = auto()      # Probe is stable, update less frequently
    NEWLY_PLACED = auto()  # Probe was just placed, needs full update


@dataclass
class DDGIProbeConfig:
    """Configuration for DDGI probes.

    Attributes:
        rays_per_probe: Number of rays traced per probe per update
        irradiance_resolution: Resolution of irradiance texture (octahedral)
        visibility_resolution: Resolution of visibility/distance texture
        hysteresis: Temporal blending factor (higher = more stable)
        depth_sharpness: Sharpness of visibility weights
        max_ray_distance: Maximum ray trace distance
        normal_bias: Bias along surface normal
        view_bias: Bias towards viewer
    """
    rays_per_probe: int = 256
    irradiance_resolution: int = 8
    visibility_resolution: int = 16
    hysteresis: float = 0.97
    depth_sharpness: float = 50.0
    max_ray_distance: float = 100.0
    normal_bias: float = 0.25
    view_bias: float = 0.25


@dataclass
class DDGIProbe:
    """DDGI probe storing irradiance and visibility.

    Each probe stores:
    - Irradiance: Low-frequency indirect lighting (octahedral encoding)
    - Visibility: Mean distance and variance (for soft occlusion)

    Attributes:
        position: World position of the probe
        state: Current probe state
        irradiance_data: Octahedrally-encoded irradiance
        visibility_data: Octahedrally-encoded depth/variance
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    state: DDGIProbeState = DDGIProbeState.ACTIVE

    # Octahedral map data (flattened from 2D)
    irradiance_data: list[Vec3] = field(default_factory=list)
    visibility_data: list[Vec2] = field(default_factory=list)  # (mean_dist, variance)

    # Configuration
    config: DDGIProbeConfig = field(default_factory=DDGIProbeConfig)

    # Update tracking
    _frames_since_update: int = 0
    _total_updates: int = 0

    def __post_init__(self) -> None:
        # Initialize octahedral maps
        ir_size = self.config.irradiance_resolution ** 2
        vis_size = self.config.visibility_resolution ** 2

        if not self.irradiance_data:
            self.irradiance_data = [Vec3.zero() for _ in range(ir_size)]
        if not self.visibility_data:
            self.visibility_data = [Vec2(self.config.max_ray_distance, 0.0)
                                    for _ in range(vis_size)]

    def sample_irradiance(self, direction: Vec3) -> Vec3:
        """Sample irradiance in a direction.

        Args:
            direction: Sample direction (normalized)

        Returns:
            Irradiance color
        """
        # Convert direction to octahedral coordinates
        oct = self._direction_to_octahedral(direction)

        # Sample with bilinear interpolation
        return self._sample_octahedral(
            oct,
            self.irradiance_data,
            self.config.irradiance_resolution,
        )

    def sample_visibility(self, direction: Vec3) -> Vec2:
        """Sample visibility (distance, variance) in a direction.

        Args:
            direction: Sample direction

        Returns:
            Vec2(mean_distance, variance)
        """
        oct = self._direction_to_octahedral(direction)

        return self._sample_octahedral_vec2(
            oct,
            self.visibility_data,
            self.config.visibility_resolution,
        )

    def _direction_to_octahedral(self, direction: Vec3) -> Vec2:
        """Convert direction to octahedral coordinates.

        Args:
            direction: Normalized direction

        Returns:
            Octahedral coordinates in [0, 1]
        """
        d = direction.normalized()

        # Project onto octahedron
        inv_l1 = 1.0 / (abs(d.x) + abs(d.y) + abs(d.z))
        ox = d.x * inv_l1
        oy = d.y * inv_l1

        # Wrap negative hemisphere
        if d.z < 0:
            ox = (1.0 - abs(oy)) * (1.0 if ox >= 0 else -1.0)
            oy = (1.0 - abs(ox)) * (1.0 if oy >= 0 else -1.0)

        # Convert to [0, 1] range
        return Vec2(ox * 0.5 + 0.5, oy * 0.5 + 0.5)

    def _octahedral_to_direction(self, oct: Vec2) -> Vec3:
        """Convert octahedral coordinates to direction.

        Args:
            oct: Octahedral coordinates in [0, 1]

        Returns:
            Normalized direction
        """
        # Convert from [0, 1] to [-1, 1]
        ox = oct.x * 2.0 - 1.0
        oy = oct.y * 2.0 - 1.0

        # Unfold octahedron
        oz = 1.0 - abs(ox) - abs(oy)

        if oz < 0:
            # Lower hemisphere
            new_ox = (1.0 - abs(oy)) * (1.0 if ox >= 0 else -1.0)
            new_oy = (1.0 - abs(ox)) * (1.0 if oy >= 0 else -1.0)
            ox, oy = new_ox, new_oy

        return Vec3(ox, oy, oz).normalized()

    def _sample_octahedral(
        self,
        oct: Vec2,
        data: list[Vec3],
        resolution: int,
    ) -> Vec3:
        """Sample octahedral map with bilinear interpolation.

        Args:
            oct: Octahedral coordinates
            data: Map data
            resolution: Map resolution

        Returns:
            Sampled value
        """
        # Add border texel offset
        border = 1.0 / resolution
        oct = Vec2(
            oct.x * (1.0 - 2.0 * border) + border,
            oct.y * (1.0 - 2.0 * border) + border,
        )

        x = oct.x * (resolution - 1)
        y = oct.y * (resolution - 1)

        x0 = max(0, min(int(x), resolution - 1))
        y0 = max(0, min(int(y), resolution - 1))
        x1 = min(x0 + 1, resolution - 1)
        y1 = min(y0 + 1, resolution - 1)

        fx = x - x0
        fy = y - y0

        def get(xi: int, yi: int) -> Vec3:
            return data[yi * resolution + xi]

        # Bilinear interpolation
        v00 = get(x0, y0)
        v10 = get(x1, y0)
        v01 = get(x0, y1)
        v11 = get(x1, y1)

        return (
            v00 * (1 - fx) * (1 - fy) +
            v10 * fx * (1 - fy) +
            v01 * (1 - fx) * fy +
            v11 * fx * fy
        )

    def _sample_octahedral_vec2(
        self,
        oct: Vec2,
        data: list[Vec2],
        resolution: int,
    ) -> Vec2:
        """Sample octahedral Vec2 map with bilinear interpolation."""
        border = 1.0 / resolution
        oct = Vec2(
            oct.x * (1.0 - 2.0 * border) + border,
            oct.y * (1.0 - 2.0 * border) + border,
        )

        x = oct.x * (resolution - 1)
        y = oct.y * (resolution - 1)

        x0 = max(0, min(int(x), resolution - 1))
        y0 = max(0, min(int(y), resolution - 1))
        x1 = min(x0 + 1, resolution - 1)
        y1 = min(y0 + 1, resolution - 1)

        fx = x - x0
        fy = y - y0

        def get(xi: int, yi: int) -> Vec2:
            return data[yi * resolution + xi]

        v00 = get(x0, y0)
        v10 = get(x1, y0)
        v01 = get(x0, y1)
        v11 = get(x1, y1)

        return Vec2(
            v00.x * (1 - fx) * (1 - fy) + v10.x * fx * (1 - fy) +
            v01.x * (1 - fx) * fy + v11.x * fx * fy,
            v00.y * (1 - fx) * (1 - fy) + v10.y * fx * (1 - fy) +
            v01.y * (1 - fx) * fy + v11.y * fx * fy,
        )

    def update_texel(
        self,
        oct_coord: Vec2,
        new_irradiance: Vec3,
        new_depth: float,
        is_irradiance: bool = True,
    ) -> None:
        """Update a texel with temporal blending.

        Args:
            oct_coord: Octahedral coordinate
            new_irradiance: New irradiance value
            new_depth: New depth value
            is_irradiance: Whether updating irradiance or visibility
        """
        hysteresis = self.config.hysteresis

        if is_irradiance:
            resolution = self.config.irradiance_resolution
            x = int(oct_coord.x * (resolution - 1))
            y = int(oct_coord.y * (resolution - 1))
            idx = y * resolution + x

            if 0 <= idx < len(self.irradiance_data):
                old = self.irradiance_data[idx]
                self.irradiance_data[idx] = Vec3(
                    old.x * hysteresis + new_irradiance.x * (1 - hysteresis),
                    old.y * hysteresis + new_irradiance.y * (1 - hysteresis),
                    old.z * hysteresis + new_irradiance.z * (1 - hysteresis),
                )
        else:
            resolution = self.config.visibility_resolution
            x = int(oct_coord.x * (resolution - 1))
            y = int(oct_coord.y * (resolution - 1))
            idx = y * resolution + x

            if 0 <= idx < len(self.visibility_data):
                old = self.visibility_data[idx]
                new_var = new_depth * new_depth
                self.visibility_data[idx] = Vec2(
                    old.x * hysteresis + new_depth * (1 - hysteresis),
                    old.y * hysteresis + new_var * (1 - hysteresis),
                )


@dataclass
class DDGIGridConfig:
    """Configuration for DDGI probe grid.

    Attributes:
        resolution: Grid resolution (x, y, z)
        bounds: World-space bounds
        probe_config: Configuration for individual probes
        scroll_with_camera: Whether the grid scrolls with camera
        infinite_scrolling: Enable infinite scrolling grid
    """
    resolution: tuple[int, int, int] = (8, 4, 8)
    bounds: AABB = field(
        default_factory=lambda: AABB(Vec3(-20, 0, -20), Vec3(20, 10, 20))
    )
    probe_config: DDGIProbeConfig = field(default_factory=DDGIProbeConfig)
    scroll_with_camera: bool = False
    infinite_scrolling: bool = False

    @property
    def probe_spacing(self) -> Vec3:
        """Spacing between probes."""
        extent = self.bounds.max - self.bounds.min
        return Vec3(
            extent.x / max(1, self.resolution[0] - 1),
            extent.y / max(1, self.resolution[1] - 1),
            extent.z / max(1, self.resolution[2] - 1),
        )

    @property
    def total_probes(self) -> int:
        """Total number of probes."""
        return self.resolution[0] * self.resolution[1] * self.resolution[2]


class DDGIProbeGrid:
    """Grid of DDGI probes for dynamic diffuse GI.

    The grid manages probe placement, updates, and provides
    efficient lookup for shading.
    """

    def __init__(self, config: DDGIGridConfig) -> None:
        """Initialize the DDGI grid.

        Args:
            config: Grid configuration
        """
        self.config = config
        self._probes: list[DDGIProbe] = []
        self._grid_offset: Vec3 = Vec3.zero()  # For scrolling
        self._build_grid()

    def _build_grid(self) -> None:
        """Create probes at grid positions."""
        rx, ry, rz = self.config.resolution
        spacing = self.config.probe_spacing
        origin = self.config.bounds.min

        self._probes = []
        for z in range(rz):
            for y in range(ry):
                for x in range(rx):
                    position = Vec3(
                        origin.x + x * spacing.x,
                        origin.y + y * spacing.y,
                        origin.z + z * spacing.z,
                    )
                    probe = DDGIProbe(
                        position=position,
                        config=self.config.probe_config,
                    )
                    self._probes.append(probe)

    def get_probe_index(self, x: int, y: int, z: int) -> int:
        """Get linear index for a probe.

        Args:
            x: X index
            y: Y index
            z: Z index

        Returns:
            Linear index
        """
        rx, ry, _ = self.config.resolution
        return z * (rx * ry) + y * rx + x

    def get_probe(self, x: int, y: int, z: int) -> Optional[DDGIProbe]:
        """Get a probe by grid index.

        Args:
            x: X index
            y: Y index
            z: Z index

        Returns:
            Probe at the index
        """
        rx, ry, rz = self.config.resolution
        if 0 <= x < rx and 0 <= y < ry and 0 <= z < rz:
            return self._probes[self.get_probe_index(x, y, z)]
        return None

    def world_to_grid(self, point: Vec3) -> tuple[float, float, float]:
        """Convert world position to grid coordinates.

        Args:
            point: World position

        Returns:
            Grid coordinates (may be fractional)
        """
        local = point - self.config.bounds.min - self._grid_offset
        spacing = self.config.probe_spacing

        return (
            local.x / spacing.x if spacing.x > 0 else 0,
            local.y / spacing.y if spacing.y > 0 else 0,
            local.z / spacing.z if spacing.z > 0 else 0,
        )

    def get_probes_for_update(self, budget: int) -> list[DDGIProbe]:
        """Get probes that need updating within budget.

        Args:
            budget: Maximum number of probes to return

        Returns:
            List of probes to update
        """
        # Prioritize:
        # 1. Newly placed probes
        # 2. Active probes
        # 3. Sleeping probes (occasional updates)

        update_list = []

        # First pass: newly placed
        for probe in self._probes:
            if len(update_list) >= budget:
                break
            if probe.state == DDGIProbeState.NEWLY_PLACED:
                update_list.append(probe)
                probe.state = DDGIProbeState.ACTIVE

        # Second pass: active probes
        for probe in self._probes:
            if len(update_list) >= budget:
                break
            if probe.state == DDGIProbeState.ACTIVE and probe not in update_list:
                update_list.append(probe)

        # Third pass: sleeping probes (update occasionally)
        for probe in self._probes:
            if len(update_list) >= budget:
                break
            if probe.state == DDGIProbeState.SLEEPING:
                probe._frames_since_update += 1
                # Update sleeping probes every 60 frames
                if probe._frames_since_update > 60:
                    update_list.append(probe)
                    probe._frames_since_update = 0

        return update_list

    def scroll_grid(self, camera_position: Vec3) -> None:
        """Scroll the grid to follow the camera.

        Args:
            camera_position: Current camera position
        """
        if not self.config.scroll_with_camera:
            return

        spacing = self.config.probe_spacing
        bounds = self.config.bounds
        center = bounds.center

        # Check if camera has moved enough to scroll
        offset_needed = Vec3(
            (camera_position.x - center.x - self._grid_offset.x),
            0,  # Usually don't scroll vertically
            (camera_position.z - center.z - self._grid_offset.z),
        )

        # Scroll in whole probe increments
        scroll_x = int(offset_needed.x / spacing.x) if spacing.x > 0 else 0
        scroll_z = int(offset_needed.z / spacing.z) if spacing.z > 0 else 0

        if scroll_x != 0 or scroll_z != 0:
            self._scroll(scroll_x, 0, scroll_z)

    def _scroll(self, dx: int, dy: int, dz: int) -> None:
        """Scroll the grid by probe increments.

        Args:
            dx: X scroll amount (in probes)
            dy: Y scroll amount (in probes)
            dz: Z scroll amount (in probes)
        """
        spacing = self.config.probe_spacing
        self._grid_offset = self._grid_offset + Vec3(
            dx * spacing.x,
            dy * spacing.y,
            dz * spacing.z,
        )

        # Update probe positions and mark edge probes as newly placed
        rx, ry, rz = self.config.resolution

        for z in range(rz):
            for y in range(ry):
                for x in range(rx):
                    probe = self.get_probe(x, y, z)
                    if probe:
                        # Update position
                        base_pos = Vec3(
                            self.config.bounds.min.x + x * spacing.x,
                            self.config.bounds.min.y + y * spacing.y,
                            self.config.bounds.min.z + z * spacing.z,
                        )
                        probe.position = base_pos + self._grid_offset

                        # Mark edge probes as needing updates
                        if self.config.infinite_scrolling:
                            if (dx != 0 and (x == 0 or x == rx - 1)) or \
                               (dz != 0 and (z == 0 or z == rz - 1)):
                                probe.state = DDGIProbeState.NEWLY_PLACED


@dataclass
class RayResult:
    """Result of a traced ray.

    Attributes:
        hit: Whether the ray hit geometry
        hit_position: World position of hit
        hit_normal: Normal at hit point
        hit_distance: Distance to hit
        radiance: Radiance at hit point
    """
    hit: bool = False
    hit_position: Vec3 = field(default_factory=Vec3.zero)
    hit_normal: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    hit_distance: float = 0.0
    radiance: Vec3 = field(default_factory=Vec3.zero)


class DDGIUpdatePass:
    """Update pass for DDGI probes using ray tracing.

    Traces rays from each probe and updates irradiance/visibility textures.
    """

    def __init__(self, grid: DDGIProbeGrid) -> None:
        """Initialize the update pass.

        Args:
            grid: DDGI probe grid to update
        """
        self.grid = grid
        self._ray_directions: list[Vec3] = []
        self._regenerate_ray_directions()

    def _regenerate_ray_directions(self) -> None:
        """Generate ray directions using Fibonacci spiral."""
        n = self.grid.config.probe_config.rays_per_probe
        golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0

        self._ray_directions = []
        for i in range(n):
            theta = 2.0 * math.pi * i / golden_ratio
            phi = math.acos(1.0 - 2.0 * (i + 0.5) / n)

            self._ray_directions.append(Vec3(
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi),
            ))

    def update(
        self,
        probes: list[DDGIProbe],
        trace_func: Callable[[Vec3, Vec3], RayResult],
        frame_random_rotation: float = 0.0,
    ) -> None:
        """Update a batch of probes.

        Args:
            probes: List of probes to update
            trace_func: Function to trace rays (origin, direction) -> RayResult
            frame_random_rotation: Random rotation for ray jittering
        """
        # Apply random rotation to ray directions for temporal stability
        rotation_matrix = self._get_rotation_matrix(frame_random_rotation)

        for probe in probes:
            self._update_probe(probe, trace_func, rotation_matrix)
            probe._total_updates += 1

    def _get_rotation_matrix(self, angle: float) -> list[Vec3]:
        """Get rotation matrix rows for Y-axis rotation.

        Args:
            angle: Rotation angle in radians

        Returns:
            3x3 rotation matrix as list of row vectors
        """
        c = math.cos(angle)
        s = math.sin(angle)
        return [
            Vec3(c, 0, s),
            Vec3(0, 1, 0),
            Vec3(-s, 0, c),
        ]

    def _rotate_direction(self, d: Vec3, matrix: list[Vec3]) -> Vec3:
        """Apply rotation matrix to direction."""
        return Vec3(
            d.x * matrix[0].x + d.y * matrix[0].y + d.z * matrix[0].z,
            d.x * matrix[1].x + d.y * matrix[1].y + d.z * matrix[1].z,
            d.x * matrix[2].x + d.y * matrix[2].y + d.z * matrix[2].z,
        )

    def _update_probe(
        self,
        probe: DDGIProbe,
        trace_func: Callable[[Vec3, Vec3], RayResult],
        rotation: list[Vec3],
    ) -> None:
        """Update a single probe.

        Args:
            probe: Probe to update
            trace_func: Ray trace function
            rotation: Rotation matrix for ray jittering
        """
        config = probe.config

        # Accumulate results per octahedral texel
        ir_res = config.irradiance_resolution
        vis_res = config.visibility_resolution

        ir_accum = [[Vec3.zero(), 0.0] for _ in range(ir_res * ir_res)]
        vis_accum = [[0.0, 0.0, 0] for _ in range(vis_res * vis_res)]  # sum, sum_sq, count

        for ray_dir in self._ray_directions:
            # Apply rotation
            rotated_dir = self._rotate_direction(ray_dir, rotation)

            # Trace ray
            result = trace_func(probe.position, rotated_dir)

            # Get octahedral coordinates
            oct = probe._direction_to_octahedral(rotated_dir)

            # Accumulate irradiance
            ir_x = int(oct.x * (ir_res - 1))
            ir_y = int(oct.y * (ir_res - 1))
            ir_idx = ir_y * ir_res + ir_x
            if 0 <= ir_idx < len(ir_accum):
                ir_accum[ir_idx][0] = ir_accum[ir_idx][0] + result.radiance
                ir_accum[ir_idx][1] += 1.0

            # Accumulate visibility
            vis_x = int(oct.x * (vis_res - 1))
            vis_y = int(oct.y * (vis_res - 1))
            vis_idx = vis_y * vis_res + vis_x
            if 0 <= vis_idx < len(vis_accum):
                depth = result.hit_distance if result.hit else config.max_ray_distance
                vis_accum[vis_idx][0] += depth
                vis_accum[vis_idx][1] += depth * depth
                vis_accum[vis_idx][2] += 1

        # Apply accumulated values with temporal blending
        hysteresis = config.hysteresis

        for idx, (accum_color, count) in enumerate(ir_accum):
            if count > 0:
                new_ir = accum_color * (1.0 / count)
                old_ir = probe.irradiance_data[idx]
                probe.irradiance_data[idx] = Vec3(
                    old_ir.x * hysteresis + new_ir.x * (1 - hysteresis),
                    old_ir.y * hysteresis + new_ir.y * (1 - hysteresis),
                    old_ir.z * hysteresis + new_ir.z * (1 - hysteresis),
                )

        for idx, (sum_d, sum_sq, count) in enumerate(vis_accum):
            if count > 0:
                new_mean = sum_d / count
                new_var = sum_sq / count - new_mean * new_mean
                old_vis = probe.visibility_data[idx]
                probe.visibility_data[idx] = Vec2(
                    old_vis.x * hysteresis + new_mean * (1 - hysteresis),
                    old_vis.y * hysteresis + max(0, new_var) * (1 - hysteresis),
                )


class DDGILookup:
    """Lookup utility for sampling DDGI at shading points."""

    def __init__(self, grid: DDGIProbeGrid) -> None:
        """Initialize the lookup utility.

        Args:
            grid: DDGI probe grid to sample from
        """
        self.grid = grid

    def sample_irradiance(
        self,
        world_pos: Vec3,
        normal: Vec3,
        view_dir: Vec3,
    ) -> Vec3:
        """Sample irradiance at a shading point.

        Uses trilinear interpolation with visibility-based weighting.

        Args:
            world_pos: World position of shading point
            normal: Surface normal
            view_dir: View direction (towards camera)

        Returns:
            Irradiance color
        """
        config = self.grid.config
        bounds = config.bounds

        # Check if point is within bounds
        if not bounds.contains(world_pos):
            return Vec3.zero()

        # Get grid coordinates
        gx, gy, gz = self.grid.world_to_grid(world_pos)
        rx, ry, rz = config.resolution

        # Clamp to valid range
        gx = max(0, min(gx, rx - 1.001))
        gy = max(0, min(gy, ry - 1.001))
        gz = max(0, min(gz, rz - 1.001))

        # Get surrounding probe indices
        x0, y0, z0 = int(gx), int(gy), int(gz)
        x1 = min(x0 + 1, rx - 1)
        y1 = min(y0 + 1, ry - 1)
        z1 = min(z0 + 1, rz - 1)

        # Biases for stable sampling
        bias_config = config.probe_config
        biased_pos = world_pos + normal * bias_config.normal_bias + view_dir * bias_config.view_bias

        # Sample 8 surrounding probes with visibility weighting
        total_weight = 0.0
        result = Vec3.zero()

        for corner in [(x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (x1, y1, z0),
                       (x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (x1, y1, z1)]:
            probe = self.grid.get_probe(corner[0], corner[1], corner[2])
            if not probe:
                continue

            # Trilinear weight
            trilinear = (
                (1 - abs(gx - corner[0])) *
                (1 - abs(gy - corner[1])) *
                (1 - abs(gz - corner[2]))
            )

            # Direction from probe to shading point
            to_surface = (biased_pos - probe.position).normalized()

            # Visibility weight using Chebyshev test
            visibility = probe.sample_visibility(to_surface)
            dist_to_probe = (world_pos - probe.position).length()

            visibility_weight = self._chebyshev_weight(
                dist_to_probe,
                visibility.x,  # mean
                visibility.y,  # variance
                bias_config.depth_sharpness,
            )

            # Normal weight (backface rejection)
            normal_weight = max(0.0001, normal.dot(-to_surface))

            # Combined weight
            weight = trilinear * visibility_weight * normal_weight

            if weight > 0:
                irradiance = probe.sample_irradiance(normal)
                result = result + irradiance * weight
                total_weight += weight

        if total_weight > 0:
            return result * (1.0 / total_weight)
        return Vec3.zero()

    def _chebyshev_weight(
        self,
        distance: float,
        mean: float,
        variance: float,
        sharpness: float,
    ) -> float:
        """Compute Chebyshev visibility weight.

        Args:
            distance: Distance to test
            mean: Mean distance from probe
            variance: Variance of distances
            sharpness: Weight sharpness

        Returns:
            Visibility weight [0, 1]
        """
        if distance <= mean:
            return 1.0

        # Chebyshev's inequality
        min_variance = 0.0001
        variance = max(variance, min_variance)

        d = distance - mean
        p_max = variance / (variance + d * d)

        # Apply sharpness
        return pow(p_max, sharpness)
