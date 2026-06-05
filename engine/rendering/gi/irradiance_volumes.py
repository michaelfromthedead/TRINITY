"""Irradiance Volume System for TRINITY (T-GIR-P2.8).

This module implements support for multiple independent probe grids with
cross-fade blending. It builds upon the single-grid DDGI implementation
from T-GIR-P2.1.

Features:
    - Multiple independent volume types (GLOBAL, LOCAL, INTERIOR, EXTERIOR)
    - Priority-based volume layering
    - Soft boundary cross-fade blending
    - Smooth transitions when entering/exiting volumes
    - Confidence output for fallback decisions
    - GPU buffer management for active volumes

Volume Types:
    - GLOBAL: Camera-relative, infinite extent, lowest priority
    - LOCAL: Fixed bounds, higher detail for specific areas
    - INTERIOR: Indoor volumes with contained lighting
    - EXTERIOR: Outdoor/sky volumes with environmental lighting

References:
    - DDGI paper (2019) - probe-based GI
    - UE5 Lumen - multi-volume GI approach
    - Frostbite Enlighten - irradiance volume blending
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Iterator, Optional

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3

if TYPE_CHECKING:
    from engine.rendering.lighting.gi_ddgi import DDGICameraRelativeGrid


# ============================================================================
# Volume Types
# ============================================================================


class VolumeType(Enum):
    """Classification of irradiance volume behavior.

    Volume types determine default priority and blending behavior:

    | Type     | Priority | Behavior                           |
    |----------|----------|-------------------------------------|
    | GLOBAL   | 0        | Camera-relative, covers everything |
    | EXTERIOR | 1        | Outdoor areas, sky lighting        |
    | INTERIOR | 2        | Indoor areas, contained lighting   |
    | LOCAL    | 3        | High-detail specific regions       |
    """

    GLOBAL = auto()      # Camera-relative, infinite
    LOCAL = auto()       # Fixed bounds, higher detail
    INTERIOR = auto()    # Indoor volumes
    EXTERIOR = auto()    # Outdoor/sky volumes

    @property
    def default_priority(self) -> int:
        """Get the default priority for this volume type."""
        priorities = {
            VolumeType.GLOBAL: 0,
            VolumeType.EXTERIOR: 1,
            VolumeType.INTERIOR: 2,
            VolumeType.LOCAL: 3,
        }
        return priorities.get(self, 0)

    @property
    def default_blend_distance(self) -> float:
        """Get the default blend distance for this volume type."""
        distances = {
            VolumeType.GLOBAL: 0.0,   # No blend for global
            VolumeType.EXTERIOR: 4.0,  # Wide exterior transitions
            VolumeType.INTERIOR: 2.0,  # Medium interior transitions
            VolumeType.LOCAL: 1.0,     # Tight local blends
        }
        return distances.get(self, 2.0)


# ============================================================================
# Volume State
# ============================================================================


class VolumeState(Enum):
    """State of an irradiance volume for transition management."""

    INACTIVE = auto()     # Volume is disabled
    ENTERING = auto()     # Camera entering volume (blend in)
    ACTIVE = auto()       # Volume is fully active
    EXITING = auto()      # Camera exiting volume (blend out)


@dataclass
class VolumeTransition:
    """Tracks smooth transitions between volume states.

    Attributes:
        state: Current transition state
        progress: Transition progress (0.0 to 1.0)
        rate: Transition rate per second
    """

    state: VolumeState = VolumeState.INACTIVE
    progress: float = 0.0
    rate: float = 2.0  # Transition speed (1/seconds)

    def update(self, dt: float, should_be_active: bool) -> None:
        """Update transition state based on whether volume should be active.

        Args:
            dt: Delta time in seconds
            should_be_active: Whether the volume should be active
        """
        if should_be_active:
            if self.state in (VolumeState.INACTIVE, VolumeState.EXITING):
                self.state = VolumeState.ENTERING
            if self.state == VolumeState.ENTERING:
                self.progress += dt * self.rate
                if self.progress >= 1.0:
                    self.progress = 1.0
                    self.state = VolumeState.ACTIVE
        else:
            if self.state in (VolumeState.ACTIVE, VolumeState.ENTERING):
                self.state = VolumeState.EXITING
            if self.state == VolumeState.EXITING:
                self.progress -= dt * self.rate
                if self.progress <= 0.0:
                    self.progress = 0.0
                    self.state = VolumeState.INACTIVE

    def get_blend_factor(self) -> float:
        """Get the current blend factor for smooth transitions."""
        # Use smoothstep for eased transitions
        t = self.progress
        return t * t * (3.0 - 2.0 * t)

    def is_contributing(self) -> bool:
        """Check if volume is contributing to lighting."""
        return self.state != VolumeState.INACTIVE


# ============================================================================
# Irradiance Volume
# ============================================================================


@dataclass
class IrradianceVolume:
    """An irradiance probe volume for indirect lighting.

    Each volume contains a probe grid and defines how its lighting
    contributes to the scene. Volumes can overlap with priority-based
    blending at boundaries.

    Attributes:
        id: Unique volume identifier
        volume_type: Type classification (GLOBAL, LOCAL, etc.)
        bounds: World-space axis-aligned bounding box
        priority: Blending priority (higher overrides lower)
        blend_distance: Distance over which boundaries cross-fade
        grid: Associated DDGI probe grid (may be None if not initialized)
        enabled: Whether volume is actively contributing
        transition: State for smooth enter/exit transitions
    """

    id: int
    volume_type: VolumeType
    bounds: AABB
    priority: int = 0
    blend_distance: float = 2.0
    grid: Optional[DDGICameraRelativeGrid] = None
    enabled: bool = True
    transition: VolumeTransition = field(default_factory=VolumeTransition)

    # Internal state
    _last_sample_weight: float = 0.0

    def __post_init__(self) -> None:
        """Initialize defaults based on volume type if not specified.

        Note: This only applies defaults for priority=0 (non-GLOBAL) and
        blend_distance=2.0 (the dataclass default). If you explicitly pass
        these values and want to keep them, use priority=-1 to indicate
        explicit zero priority, or pass any non-default blend_distance.
        """
        # Only apply default priority if priority=0 and not GLOBAL
        # (GLOBAL has default_priority=0, so this is correct)
        if self.priority == 0 and self.volume_type != VolumeType.GLOBAL:
            self.priority = self.volume_type.default_priority
        # Note: blend_distance is NOT overridden anymore to preserve explicit values
        # The dataclass default of 2.0 is a reasonable middle-ground

    def contains(self, point: Vec3) -> bool:
        """Check if a point is inside the volume bounds.

        Args:
            point: World-space position to test

        Returns:
            True if point is within volume bounds
        """
        return self.bounds.contains(point)

    def signed_distance(self, point: Vec3) -> float:
        """Compute signed distance from point to volume boundary.

        Negative = inside volume
        Zero = on boundary
        Positive = outside volume

        Args:
            point: World-space position

        Returns:
            Signed distance to nearest boundary
        """
        center = self.bounds.center
        extents = self.bounds.extents

        # Distance to each face
        dx = abs(point.x - center.x) - extents.x
        dy = abs(point.y - center.y) - extents.y
        dz = abs(point.z - center.z) - extents.z

        # Outside: Euclidean distance to corner
        outside_dist = Vec3(max(dx, 0.0), max(dy, 0.0), max(dz, 0.0)).length()

        # Inside: distance to nearest face (negative)
        inside_dist = max(dx, max(dy, dz))

        if inside_dist < 0:
            return inside_dist  # Inside volume (negative)
        return outside_dist  # Outside volume (positive)

    def get_blend_weight(self, point: Vec3) -> float:
        """Compute blend weight for a point based on distance to boundary.

        The blend weight is:
        - 1.0 when deep inside the volume (beyond blend_distance from edge)
        - 0.0 when outside the volume
        - Interpolated smoothly in the blend region

        Args:
            point: World-space position

        Returns:
            Blend weight in range [0.0, 1.0]
        """
        if not self.enabled:
            return 0.0

        if self.blend_distance <= 0.0:
            # No blending - binary in/out
            return 1.0 if self.contains(point) else 0.0

        signed_dist = self.signed_distance(point)

        if signed_dist >= 0:
            # Outside volume
            return 0.0
        elif signed_dist <= -self.blend_distance:
            # Deep inside volume
            return 1.0
        else:
            # In blend region: smoothstep from 0 to 1
            t = -signed_dist / self.blend_distance
            # Smoothstep for smooth derivative at boundaries
            return t * t * (3.0 - 2.0 * t)

    def get_effective_weight(self, point: Vec3) -> float:
        """Get weight including transition state.

        Args:
            point: World-space position

        Returns:
            Effective blend weight including transitions
        """
        base_weight = self.get_blend_weight(point)
        transition_factor = self.transition.get_blend_factor()
        return base_weight * transition_factor

    def sample_irradiance(
        self,
        world_pos: Vec3,
        normal: Vec3,
    ) -> tuple[Vec3, float]:
        """Sample irradiance from this volume at a world position.

        Args:
            world_pos: World-space position to sample
            normal: Surface normal for hemisphere sampling

        Returns:
            Tuple of (irradiance color, confidence)
            Confidence is 0.0 if volume has no grid or point is outside
        """
        if self.grid is None:
            return (Vec3.zero(), 0.0)

        weight = self.get_effective_weight(world_pos)
        if weight <= 0.0:
            return (Vec3.zero(), 0.0)

        # Sample from probe grid
        # Note: In production, this would call grid.sample_irradiance()
        # For now, return a placeholder until grid integration is complete
        irradiance = self._sample_grid(world_pos, normal)
        confidence = weight * (1.0 if self.grid is not None else 0.0)

        self._last_sample_weight = weight
        return (irradiance, confidence)

    def _sample_grid(self, world_pos: Vec3, normal: Vec3) -> Vec3:
        """Internal grid sampling implementation.

        This method interfaces with the DDGICameraRelativeGrid for
        actual probe lookup and interpolation.

        Args:
            world_pos: World-space sample position
            normal: Surface normal

        Returns:
            Sampled irradiance color
        """
        if self.grid is None:
            return Vec3.zero()

        # Check if within grid bounds
        grid_bounds = self.grid.get_bounds()
        if not grid_bounds.contains(world_pos):
            return Vec3.zero()

        # Get probe indices for trilinear interpolation
        # This is a simplified implementation - full version would
        # use 8-probe trilinear with visibility weighting
        ix, iy, iz = self.grid.world_to_probe_index(world_pos)
        probe_pos = self.grid.get_probe_world_position(ix, iy, iz)

        # Placeholder: return attenuated ambient based on distance
        dist = (world_pos - probe_pos).length()
        spacing = self.grid.config.get_spacing()
        atten = max(0.0, 1.0 - dist / (spacing * 2.0))

        # Approximate ambient contribution
        return Vec3(0.1, 0.1, 0.15) * atten

    def update(self, camera_pos: Vec3, dt: float) -> None:
        """Update volume state for a frame.

        Args:
            camera_pos: Current camera position
            dt: Delta time in seconds
        """
        # Update transition based on whether camera is in volume
        should_be_active = self.enabled and self.contains(camera_pos)
        self.transition.update(dt, should_be_active)

        # Update underlying grid if present
        if self.grid is not None:
            self.grid.update_for_camera(camera_pos)
            self.grid.advance_frame()


# ============================================================================
# GPU Buffer Structures
# ============================================================================


@dataclass
class VolumeGpuData:
    """GPU-uploadable volume metadata.

    Size: 80 bytes (padded for alignment)

    Layout matches WGSL struct:
        struct VolumeGpu {
            bounds_min: vec3<f32>,
            priority: i32,
            bounds_max: vec3<f32>,
            blend_distance: f32,
            grid_index: u32,      // Index into probe grid array
            volume_type: u32,
            transition_factor: f32,
            _pad: f32,
        }
    """

    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    priority: int
    blend_distance: float
    grid_index: int
    volume_type: int
    transition_factor: float

    def to_bytes(self) -> bytes:
        """Pack to GPU buffer format (48 bytes).

        Returns:
            Packed bytes for GPU upload
        """
        return struct.pack(
            "<3fi3ffiIff",
            # bounds_min (vec3<f32>)
            self.bounds_min[0], self.bounds_min[1], self.bounds_min[2],
            # priority (i32)
            self.priority,
            # bounds_max (vec3<f32>)
            self.bounds_max[0], self.bounds_max[1], self.bounds_max[2],
            # blend_distance (f32)
            self.blend_distance,
            # grid_index (u32)
            self.grid_index,
            # volume_type (u32)
            self.volume_type,
            # transition_factor (f32)
            self.transition_factor,
            # _pad (f32)
            0.0,
        )

    @staticmethod
    def from_volume(volume: IrradianceVolume, grid_index: int) -> VolumeGpuData:
        """Create GPU data from a volume.

        Args:
            volume: Source irradiance volume
            grid_index: Index of volume's grid in GPU grid array

        Returns:
            GPU-uploadable volume data
        """
        return VolumeGpuData(
            bounds_min=(volume.bounds.min.x, volume.bounds.min.y, volume.bounds.min.z),
            bounds_max=(volume.bounds.max.x, volume.bounds.max.y, volume.bounds.max.z),
            priority=volume.priority,
            blend_distance=volume.blend_distance,
            grid_index=grid_index,
            volume_type=volume.volume_type.value,
            transition_factor=volume.transition.get_blend_factor(),
        )


# ============================================================================
# Volume Manager
# ============================================================================


@dataclass
class IrradianceVolumeManager:
    """Manages multiple irradiance volumes with blending.

    The manager handles:
    - Volume registration and lifecycle
    - Active volume selection based on camera position
    - Cross-fade blending at volume boundaries
    - GPU buffer packing for shader access

    Attributes:
        volumes: All registered volumes
        max_active_volumes: Maximum volumes to sample simultaneously
        _next_id: Counter for unique volume IDs
        _active_cache: Cached list of active volumes
        _gpu_buffer_dirty: Whether GPU buffers need update
    """

    volumes: list[IrradianceVolume] = field(default_factory=list)
    max_active_volumes: int = 4

    _next_id: int = 0
    _active_cache: list[IrradianceVolume] = field(default_factory=list)
    _gpu_buffer_dirty: bool = True

    def add_volume(self, volume: IrradianceVolume) -> int:
        """Add a volume to the manager.

        Args:
            volume: Volume to add

        Returns:
            Assigned volume ID
        """
        volume.id = self._next_id
        self._next_id += 1
        self.volumes.append(volume)
        self._gpu_buffer_dirty = True
        return volume.id

    def create_volume(
        self,
        volume_type: VolumeType,
        bounds: AABB,
        priority: Optional[int] = None,
        blend_distance: Optional[float] = None,
        grid: Optional[DDGICameraRelativeGrid] = None,
    ) -> IrradianceVolume:
        """Create and register a new volume.

        Args:
            volume_type: Type classification
            bounds: World-space bounds
            priority: Optional priority override
            blend_distance: Optional blend distance override
            grid: Optional probe grid

        Returns:
            The created and registered volume
        """
        volume = IrradianceVolume(
            id=0,  # Will be assigned by add_volume
            volume_type=volume_type,
            bounds=bounds,
            priority=priority if priority is not None else volume_type.default_priority,
            blend_distance=(
                blend_distance
                if blend_distance is not None
                else volume_type.default_blend_distance
            ),
            grid=grid,
        )
        self.add_volume(volume)
        return volume

    def remove_volume(self, volume_id: int) -> bool:
        """Remove a volume by ID.

        Args:
            volume_id: ID of volume to remove

        Returns:
            True if volume was found and removed
        """
        for i, vol in enumerate(self.volumes):
            if vol.id == volume_id:
                self.volumes.pop(i)
                self._gpu_buffer_dirty = True
                return True
        return False

    def get_volume(self, volume_id: int) -> Optional[IrradianceVolume]:
        """Get a volume by ID.

        Args:
            volume_id: Volume ID to find

        Returns:
            Volume if found, None otherwise
        """
        for vol in self.volumes:
            if vol.id == volume_id:
                return vol
        return None

    def get_active_volumes(self, camera_pos: Vec3) -> list[IrradianceVolume]:
        """Get volumes affecting the camera position, sorted by priority.

        Returns up to max_active_volumes volumes that either:
        - Contain the camera position, or
        - Are within blend distance of the camera

        Args:
            camera_pos: Current camera position

        Returns:
            List of active volumes, sorted by priority (highest first)
        """
        active = []

        for volume in self.volumes:
            if not volume.enabled:
                continue

            # Check if volume affects this position
            signed_dist = volume.signed_distance(camera_pos)

            # Include if inside or within blend distance
            if signed_dist < volume.blend_distance:
                active.append(volume)

        # Sort by priority (highest first)
        active.sort(key=lambda v: v.priority, reverse=True)

        # Limit to max active
        return active[: self.max_active_volumes]

    def iter_contributing_volumes(
        self, camera_pos: Vec3
    ) -> Iterator[tuple[IrradianceVolume, float]]:
        """Iterate over volumes contributing to lighting with their weights.

        Args:
            camera_pos: Current camera position

        Yields:
            Tuples of (volume, weight) for contributing volumes
        """
        for volume in self.get_active_volumes(camera_pos):
            weight = volume.get_effective_weight(camera_pos)
            if weight > 0.0:
                yield (volume, weight)

    def sample_irradiance(
        self,
        world_pos: Vec3,
        normal: Vec3,
    ) -> tuple[Vec3, float]:
        """Sample and blend irradiance across all active volumes.

        Uses priority-based blending:
        1. Higher priority volumes override lower priority
        2. Within same priority, blend by spatial weight
        3. Returns confidence based on total contributing weight

        Args:
            world_pos: World-space sample position
            normal: Surface normal for hemisphere sampling

        Returns:
            Tuple of (blended irradiance, confidence)
        """
        total_irradiance = Vec3.zero()
        total_weight = 0.0
        max_confidence = 0.0

        # Group by priority for proper layering
        priority_groups: dict[int, list[tuple[IrradianceVolume, float]]] = {}

        for volume in self.volumes:
            if not volume.enabled:
                continue

            weight = volume.get_effective_weight(world_pos)
            if weight > 0.0:
                priority = volume.priority
                if priority not in priority_groups:
                    priority_groups[priority] = []
                priority_groups[priority].append((volume, weight))

        # Process from lowest to highest priority
        # Higher priority volumes override (blend over) lower ones
        for priority in sorted(priority_groups.keys()):
            group = priority_groups[priority]

            group_irradiance = Vec3.zero()
            group_weight = 0.0

            for volume, weight in group:
                irradiance, confidence = volume.sample_irradiance(world_pos, normal)
                group_irradiance = group_irradiance + irradiance * weight
                group_weight += weight
                max_confidence = max(max_confidence, confidence)

            if group_weight > 0.0:
                # Normalize group contribution
                group_irradiance = group_irradiance * (1.0 / group_weight)

                # Blend with accumulated result
                # Higher priority blends over (not additive)
                blend_factor = min(group_weight, 1.0)
                total_irradiance = total_irradiance.lerp(
                    group_irradiance, blend_factor
                )
                total_weight = max(total_weight, group_weight)

        return (total_irradiance, max_confidence)

    def update(self, camera_pos: Vec3, dt: float) -> None:
        """Update all volumes for a frame.

        Args:
            camera_pos: Current camera position
            dt: Delta time in seconds
        """
        self._active_cache = self.get_active_volumes(camera_pos)

        for volume in self.volumes:
            volume.update(camera_pos, dt)

        self._gpu_buffer_dirty = True

    def build_gpu_buffer(self) -> bytes:
        """Build GPU buffer containing active volume metadata.

        The buffer contains:
        - Header: u32 volume_count, u32 max_volumes, u64 _pad
        - Array of VolumeGpuData structs

        Returns:
            Packed bytes for GPU upload
        """
        # Header (16 bytes)
        header = struct.pack(
            "<IIII",
            len(self._active_cache),  # volume_count
            self.max_active_volumes,  # max_volumes
            0,  # _pad0
            0,  # _pad1
        )

        # Volume data
        volume_data = b""
        for i, volume in enumerate(self._active_cache):
            gpu_data = VolumeGpuData.from_volume(volume, i)
            volume_data += gpu_data.to_bytes()

        # Pad to max volumes for consistent buffer size
        empty_volume = bytes(48)  # Size of VolumeGpuData
        while len(volume_data) < self.max_active_volumes * 48:
            volume_data += empty_volume

        self._gpu_buffer_dirty = False
        return header + volume_data

    def needs_gpu_upload(self) -> bool:
        """Check if GPU buffers need to be re-uploaded."""
        return self._gpu_buffer_dirty

    def mark_uploaded(self) -> None:
        """Mark GPU buffers as uploaded."""
        self._gpu_buffer_dirty = False

    def get_statistics(self) -> dict:
        """Get volume system statistics.

        Returns:
            Dictionary with statistics about volumes
        """
        type_counts = {vt: 0 for vt in VolumeType}
        active_count = len(self._active_cache)

        for volume in self.volumes:
            type_counts[volume.volume_type] += 1

        return {
            "total_volumes": len(self.volumes),
            "active_volumes": active_count,
            "max_active_volumes": self.max_active_volumes,
            "by_type": {vt.name: count for vt, count in type_counts.items()},
            "gpu_buffer_dirty": self._gpu_buffer_dirty,
        }


# ============================================================================
# Shader Helper Code Generator
# ============================================================================


def generate_volume_sampling_wgsl() -> str:
    """Generate WGSL shader code for volume sampling.

    Returns:
        WGSL code string for volume sampling helper functions
    """
    return """
// Irradiance Volume GPU Structures
struct VolumeGpu {
    bounds_min: vec3<f32>,
    priority: i32,
    bounds_max: vec3<f32>,
    blend_distance: f32,
    grid_index: u32,
    volume_type: u32,
    transition_factor: f32,
    _pad: f32,
}

struct VolumeBuffer {
    volume_count: u32,
    max_volumes: u32,
    _pad0: u32,
    _pad1: u32,
    volumes: array<VolumeGpu>,
}

@group(0) @binding(0) var<storage, read> volume_buffer: VolumeBuffer;

// Compute signed distance to volume bounds
fn volume_signed_distance(volume: VolumeGpu, pos: vec3<f32>) -> f32 {
    let center = (volume.bounds_min + volume.bounds_max) * 0.5;
    let extents = (volume.bounds_max - volume.bounds_min) * 0.5;

    let d = abs(pos - center) - extents;
    let outside_dist = length(max(d, vec3<f32>(0.0)));
    let inside_dist = max(d.x, max(d.y, d.z));

    return select(outside_dist, inside_dist, inside_dist < 0.0);
}

// Compute blend weight for a position within volume
fn volume_blend_weight(volume: VolumeGpu, pos: vec3<f32>) -> f32 {
    let signed_dist = volume_signed_distance(volume, pos);

    if (signed_dist >= 0.0) {
        return 0.0;
    }
    if (volume.blend_distance <= 0.0) {
        return 1.0;
    }
    if (signed_dist <= -volume.blend_distance) {
        return 1.0;
    }

    // Smoothstep in blend region
    let t = -signed_dist / volume.blend_distance;
    return t * t * (3.0 - 2.0 * t);
}

// Sample blended irradiance from all active volumes
fn sample_volume_irradiance(
    pos: vec3<f32>,
    normal: vec3<f32>,
    probe_grids: array<ProbeGridGpu, 4>,
) -> vec3<f32> {
    var result = vec3<f32>(0.0);
    var prev_priority = -1;
    var group_irradiance = vec3<f32>(0.0);
    var group_weight = 0.0;

    for (var i = 0u; i < volume_buffer.volume_count; i++) {
        let volume = volume_buffer.volumes[i];
        let weight = volume_blend_weight(volume, pos) * volume.transition_factor;

        if (weight > 0.0) {
            // Sample from corresponding probe grid
            let grid = probe_grids[volume.grid_index];
            let irradiance = sample_probe_grid(grid, pos, normal);

            // Priority-based blending
            if (volume.priority != prev_priority && group_weight > 0.0) {
                let blend = min(group_weight, 1.0);
                result = mix(result, group_irradiance / group_weight, blend);
                group_irradiance = vec3<f32>(0.0);
                group_weight = 0.0;
            }

            group_irradiance += irradiance * weight;
            group_weight += weight;
            prev_priority = volume.priority;
        }
    }

    // Final group blend
    if (group_weight > 0.0) {
        let blend = min(group_weight, 1.0);
        result = mix(result, group_irradiance / group_weight, blend);
    }

    return result;
}
"""


# ============================================================================
# Utility Functions
# ============================================================================


def recommend_max_volumes(gpu_tier: str = "high") -> int:
    """Recommend maximum active volumes based on GPU tier.

    Args:
        gpu_tier: One of "low", "medium", "high", "ultra"

    Returns:
        Recommended max_active_volumes setting

    Performance guidance:
        - low: 2 volumes (mobile/integrated)
        - medium: 4 volumes (laptops)
        - high: 6 volumes (desktop)
        - ultra: 8 volumes (enthusiast)
    """
    recommendations = {
        "low": 2,
        "medium": 4,
        "high": 6,
        "ultra": 8,
    }
    return recommendations.get(gpu_tier.lower(), 4)


def estimate_volume_memory(
    volume_count: int,
    probes_per_volume: int = 8192,
) -> int:
    """Estimate GPU memory usage for volume system.

    Args:
        volume_count: Number of volumes
        probes_per_volume: Average probes per volume grid

    Returns:
        Estimated memory in bytes
    """
    # Per-probe storage
    probe_sh_size = 192  # ProbeSH struct
    probe_vis_size = 16  # ProbeVis struct
    per_probe = probe_sh_size + probe_vis_size

    # Per-volume overhead
    grid_uniform_size = 64  # ProbeGridGpu
    volume_metadata_size = 48  # VolumeGpuData

    per_volume = probes_per_volume * per_probe + grid_uniform_size + volume_metadata_size

    # Buffer header
    header_size = 16

    return header_size + volume_count * per_volume
