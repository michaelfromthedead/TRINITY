"""GPU crowd rendering system.

Provides efficient instanced rendering for large crowds of animated characters,
using animation textures for GPU-based skeletal animation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterator, Sequence

from engine.core.math import Vec3, Vec4, Quat, Mat4, Transform

from .animation_texture import AnimationTexture, AnimationTextureAtlas
from .crowd_lod import CrowdLOD
from engine.animation.config import CROWD_RENDERER_CONFIG


class InstanceBufferOverflowError(Exception):
    """Raised when instance buffer exceeds maximum capacity."""
    pass


class RenderPriority(Enum):
    """Rendering priority for crowd instances."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class CrowdInstance:
    """Single instance in a crowd for rendering.

    Attributes:
        position: World position
        rotation: World rotation
        scale: Uniform scale factor
        animation_index: Index of animation in the atlas
        animation_time: Current playback time
        animation_speed: Playback speed multiplier
        tint_color: Instance color tint (RGBA)
        lod_level: Current LOD level (0 = highest detail)
        visible: Whether instance is visible
        instance_id: Unique identifier
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    rotation: Quat = field(default_factory=Quat.identity)
    scale: float = 1.0
    animation_index: int = 0
    animation_time: float = 0.0
    animation_speed: float = 1.0
    tint_color: Vec4 = field(default_factory=lambda: Vec4(1.0, 1.0, 1.0, 1.0))
    lod_level: int = 0
    visible: bool = True
    instance_id: int = 0

    _next_id: int = 0

    def __post_init__(self):
        if self.instance_id == 0:
            CrowdInstance._next_id += 1
            self.instance_id = CrowdInstance._next_id

    def get_transform_matrix(self) -> Mat4:
        """Get world transform matrix for this instance."""
        transform = Transform(
            translation=self.position,
            rotation=self.rotation,
            scale=Vec3(self.scale, self.scale, self.scale),
        )
        return transform.to_matrix()

    def advance_time(self, dt: float) -> None:
        """Advance animation time."""
        self.animation_time += dt * self.animation_speed

    def set_animation(self, index: int, reset_time: bool = True) -> None:
        """Set animation index, optionally resetting time."""
        self.animation_index = index
        if reset_time:
            self.animation_time = 0.0

    def distance_to(self, point: Vec3) -> float:
        """Calculate distance to a point."""
        return self.position.distance(point)


@dataclass
class InstanceBuffer:
    """GPU instance buffer for crowd rendering.

    Contains packed instance data ready for GPU upload.
    Uses configurable sizes and validates against maximum capacity.
    """
    # Per-instance data packed for GPU
    transform_data: list[float] = field(default_factory=list)  # 4x4 matrices
    animation_data: list[float] = field(default_factory=list)  # (anim_index, time, speed, lod)
    color_data: list[float] = field(default_factory=list)  # RGBA tints

    instance_count: int = 0
    capacity: int = 0
    max_capacity: int = field(default_factory=lambda: CROWD_RENDERER_CONFIG.MAX_INSTANCES_PER_BATCH * 10)
    dirty: bool = True

    def clear(self) -> None:
        """Clear all instance data."""
        self.transform_data.clear()
        self.animation_data.clear()
        self.color_data.clear()
        self.instance_count = 0
        self.dirty = True

    def reserve(self, count: int) -> None:
        """Reserve capacity for instances.

        Args:
            count: Number of instances to reserve space for

        Raises:
            InstanceBufferOverflowError: If count exceeds maximum capacity
        """
        if count > self.max_capacity:
            raise InstanceBufferOverflowError(
                f"Cannot reserve {count} instances, maximum is {self.max_capacity}"
            )
        self.capacity = count
        # Pre-allocate arrays with GPU alignment consideration
        # Sizes based on config constants
        self.transform_data = [0.0] * (count * CROWD_RENDERER_CONFIG.TRANSFORM_FLOATS)
        self.animation_data = [0.0] * (count * CROWD_RENDERER_CONFIG.ANIMATION_FLOATS)
        self.color_data = [0.0] * (count * CROWD_RENDERER_CONFIG.COLOR_FLOATS)

    def add_instance(self, instance: CrowdInstance) -> int:
        """Add instance to buffer, returns index.

        Raises:
            InstanceBufferOverflowError: If buffer at maximum capacity
        """
        if self.instance_count >= self.capacity > 0:
            # Need to grow
            if self.capacity >= self.max_capacity:
                raise InstanceBufferOverflowError(
                    f"Instance buffer at maximum capacity ({self.max_capacity})"
                )
            self._grow()

        idx = self.instance_count
        self.instance_count += 1

        # Pack transform matrix
        matrix = instance.get_transform_matrix()
        mat_offset = idx * 16
        if mat_offset + 16 > len(self.transform_data):
            self.transform_data.extend([0.0] * 16)
        self.transform_data[mat_offset:mat_offset + 16] = matrix.m

        # Pack animation data
        anim_offset = idx * 4
        if anim_offset + 4 > len(self.animation_data):
            self.animation_data.extend([0.0] * 4)
        self.animation_data[anim_offset:anim_offset + 4] = [
            float(instance.animation_index),
            instance.animation_time,
            instance.animation_speed,
            float(instance.lod_level),
        ]

        # Pack color data
        color_offset = idx * 4
        if color_offset + 4 > len(self.color_data):
            self.color_data.extend([0.0] * 4)
        self.color_data[color_offset:color_offset + 4] = [
            instance.tint_color.x,
            instance.tint_color.y,
            instance.tint_color.z,
            instance.tint_color.w,
        ]

        self.dirty = True
        return idx

    def update_instance(self, index: int, instance: CrowdInstance) -> None:
        """Update existing instance in buffer."""
        if index < 0 or index >= self.instance_count:
            return

        # Update transform
        matrix = instance.get_transform_matrix()
        mat_offset = index * 16
        self.transform_data[mat_offset:mat_offset + 16] = matrix.m

        # Update animation
        anim_offset = index * 4
        self.animation_data[anim_offset:anim_offset + 4] = [
            float(instance.animation_index),
            instance.animation_time,
            instance.animation_speed,
            float(instance.lod_level),
        ]

        # Update color
        color_offset = index * 4
        self.color_data[color_offset:color_offset + 4] = [
            instance.tint_color.x,
            instance.tint_color.y,
            instance.tint_color.z,
            instance.tint_color.w,
        ]

        self.dirty = True

    def _grow(self) -> None:
        """Grow buffer capacity using configured growth factor."""
        new_capacity = max(
            self.capacity * CROWD_RENDERER_CONFIG.BUFFER_GROWTH_FACTOR,
            CROWD_RENDERER_CONFIG.DEFAULT_BUFFER_CAPACITY
        )
        # Clamp to maximum
        new_capacity = min(new_capacity, self.max_capacity)

        growth = new_capacity - self.capacity
        if growth > 0:
            self.transform_data.extend([0.0] * (growth * CROWD_RENDERER_CONFIG.TRANSFORM_FLOATS))
            self.animation_data.extend([0.0] * (growth * CROWD_RENDERER_CONFIG.ANIMATION_FLOATS))
            self.color_data.extend([0.0] * (growth * CROWD_RENDERER_CONFIG.COLOR_FLOATS))
            self.capacity = new_capacity

    def get_memory_size_bytes(self) -> int:
        """Calculate memory size in bytes."""
        # Assuming float32 (4 bytes per float)
        return (len(self.transform_data) + len(self.animation_data) + len(self.color_data)) * 4


@dataclass
class CrowdRenderBatch:
    """Batch of crowd instances sharing same mesh/material.

    Instances are grouped by mesh and material for efficient rendering.
    """
    mesh_id: int = 0
    material_id: int = 0
    animation_atlas: AnimationTextureAtlas | None = None
    instance_buffer: InstanceBuffer = field(default_factory=InstanceBuffer)
    instances: list[CrowdInstance] = field(default_factory=list)

    visible: bool = True
    priority: RenderPriority = RenderPriority.NORMAL

    def add_instance(self, instance: CrowdInstance) -> int:
        """Add instance to batch."""
        self.instances.append(instance)
        return self.instance_buffer.add_instance(instance)

    def remove_instance(self, instance_id: int) -> bool:
        """Remove instance by ID."""
        for i, inst in enumerate(self.instances):
            if inst.instance_id == instance_id:
                self.instances.pop(i)
                self._rebuild_buffer()
                return True
        return False

    def update(self, dt: float) -> None:
        """Update all instances."""
        for i, instance in enumerate(self.instances):
            if instance.visible:
                instance.advance_time(dt)
                self.instance_buffer.update_instance(i, instance)

    def _rebuild_buffer(self) -> None:
        """Rebuild instance buffer from instances list."""
        self.instance_buffer.clear()
        for instance in self.instances:
            if instance.visible:
                self.instance_buffer.add_instance(instance)

    def get_visible_count(self) -> int:
        """Get count of visible instances."""
        return sum(1 for inst in self.instances if inst.visible)

    def sort_by_distance(self, camera_pos: Vec3, front_to_back: bool = True) -> None:
        """Sort instances by distance to camera."""
        self.instances.sort(
            key=lambda inst: inst.distance_to(camera_pos),
            reverse=not front_to_back,
        )
        self._rebuild_buffer()


class CrowdRenderer:
    """GPU crowd rendering system.

    Manages multiple batches of crowd instances and prepares them
    for efficient instanced rendering.
    """

    def __init__(self, max_instances_per_batch: int = CROWD_RENDERER_CONFIG.MAX_INSTANCES_PER_BATCH):
        self._batches: dict[tuple[int, int], CrowdRenderBatch] = {}
        self._animation_atlases: dict[str, AnimationTextureAtlas] = {}
        self._max_instances_per_batch = max_instances_per_batch
        self._total_instance_count = 0
        self._frame_count = 0

    @property
    def total_instance_count(self) -> int:
        """Total number of instances across all batches."""
        return self._total_instance_count

    @property
    def batch_count(self) -> int:
        """Number of render batches."""
        return len(self._batches)

    def register_animation_atlas(self, name: str, atlas: AnimationTextureAtlas) -> None:
        """Register an animation texture atlas."""
        self._animation_atlases[name] = atlas

    def get_animation_atlas(self, name: str) -> AnimationTextureAtlas | None:
        """Get registered animation atlas by name."""
        return self._animation_atlases.get(name)

    def add_instance(
        self,
        instance: CrowdInstance,
        mesh_id: int,
        material_id: int,
        atlas_name: str | None = None,
    ) -> int:
        """Add instance to appropriate batch.

        Args:
            instance: Instance to add
            mesh_id: Mesh identifier
            material_id: Material identifier
            atlas_name: Name of animation atlas to use

        Returns:
            Instance ID
        """
        batch_key = (mesh_id, material_id)

        if batch_key not in self._batches:
            batch = CrowdRenderBatch(
                mesh_id=mesh_id,
                material_id=material_id,
            )
            if atlas_name:
                batch.animation_atlas = self._animation_atlases.get(atlas_name)
            self._batches[batch_key] = batch

        batch = self._batches[batch_key]
        batch.add_instance(instance)
        self._total_instance_count += 1

        return instance.instance_id

    def remove_instance(self, instance_id: int) -> bool:
        """Remove instance by ID from all batches."""
        for batch in self._batches.values():
            if batch.remove_instance(instance_id):
                self._total_instance_count -= 1
                return True
        return False

    def update(self, dt: float) -> None:
        """Update all crowd instances.

        Args:
            dt: Delta time in seconds
        """
        self._frame_count += 1
        for batch in self._batches.values():
            batch.update(dt)

    def cull_instances(self, camera_pos: Vec3, max_distance: float) -> int:
        """Cull instances beyond max distance.

        Args:
            camera_pos: Camera world position
            max_distance: Maximum render distance

        Returns:
            Number of culled instances
        """
        culled = 0
        for batch in self._batches.values():
            for instance in batch.instances:
                was_visible = instance.visible
                instance.visible = instance.distance_to(camera_pos) <= max_distance
                if was_visible and not instance.visible:
                    culled += 1
        return culled

    def update_lod_levels(self, camera_pos: Vec3, lod_distances: list[float]) -> None:
        """Update LOD levels based on distance to camera.

        Args:
            camera_pos: Camera world position
            lod_distances: List of distance thresholds for each LOD level
        """
        for batch in self._batches.values():
            for instance in batch.instances:
                dist = instance.distance_to(camera_pos)
                lod = 0
                for i, threshold in enumerate(lod_distances):
                    if dist > threshold:
                        lod = i + 1
                    else:
                        break
                instance.lod_level = min(lod, len(lod_distances))

    def update_lod_levels_from_system(self, camera_pos: Vec3, lod_system: CrowdLOD) -> None:
        """Update LOD levels using a CrowdLOD system with hysteresis.

        Uses CrowdLOD.get_lod_for_distance which applies hysteresis to prevent
        LOD flickering when an instance is near a distance threshold.

        Args:
            camera_pos: Camera world position
            lod_system: CrowdLOD instance managing LOD level definitions
        """
        for batch in self._batches.values():
            for instance in batch.instances:
                dist = instance.distance_to(camera_pos)
                instance.lod_level = lod_system.get_lod_for_distance(dist, instance.lod_level)

    def prepare_render_data(self) -> list[tuple[CrowdRenderBatch, InstanceBuffer]]:
        """Prepare render data for all batches.

        Returns:
            List of (batch, instance_buffer) tuples ready for rendering
        """
        result = []
        for batch in self._batches.values():
            if batch.visible and batch.get_visible_count() > 0:
                result.append((batch, batch.instance_buffer))

        # Sort by priority
        result.sort(key=lambda x: x[0].priority.value, reverse=True)
        return result

    def get_batches(self) -> Iterator[CrowdRenderBatch]:
        """Iterate over all batches."""
        yield from self._batches.values()

    def get_batch(self, mesh_id: int, material_id: int) -> CrowdRenderBatch | None:
        """Get specific batch by mesh and material IDs."""
        return self._batches.get((mesh_id, material_id))

    def clear(self) -> None:
        """Clear all batches and instances."""
        self._batches.clear()
        self._total_instance_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get rendering statistics."""
        visible_count = sum(batch.get_visible_count() for batch in self._batches.values())
        total_memory = sum(
            batch.instance_buffer.get_memory_size_bytes()
            for batch in self._batches.values()
        )

        return {
            "total_instances": self._total_instance_count,
            "visible_instances": visible_count,
            "batch_count": len(self._batches),
            "frame_count": self._frame_count,
            "total_memory_bytes": total_memory,
            "atlas_count": len(self._animation_atlases),
        }
