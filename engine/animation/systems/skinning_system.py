"""GPU Skinning System Integration (T-AN-9.6).

This module implements the skinning system that dispatches GPU skinning compute
shaders with LOD-based influence reduction and CPU fallback support.

Key Features:
- @system(phase="animation", order=3) annotation for ECS scheduling
- GPU compute shader dispatch for skinning (LBS and DQS)
- LOD-based influence reduction: 4 -> 2 -> 1 bones per vertex
- CPU SIMD fallback for platforms without compute shader support
- Async compute overlap with render (when hardware supports)
- Batched skinning dispatch for efficiency

Dependencies:
- Phase 3 skinning: T-AN-3.1 (orchestrator), T-AN-3.2-3.4 (shaders)
- engine.animation.skeletal.skinning: LBS, DQS implementations
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from engine.core.math import Vec3, Vec4, Quat, Mat4, Transform
from engine.animation.config import SKINNING_CONFIG, ANIMATION_SYSTEM_CONFIG
from engine.animation.skeletal.skinning import (
    SkinningMethod as BaseSkinningMethod,
    SkinningData as BaseSkinningData,
    VertexWeight,
    LinearBlendSkinning,
    DualQuaternionSkinning,
    DualQuaternion,
    GPUSkinningData,
    SkinningCache,
    prepare_gpu_skinning_data,
    skin_mesh,
)

if TYPE_CHECKING:
    from engine.core.ecs import Entity, World


# =============================================================================
# SYSTEM DECORATOR
# =============================================================================


def system(
    phase: str = "update",
    order: int = 0,
    priority: int = 0,
    reads: Optional[Tuple[str, ...]] = None,
    writes: Optional[Tuple[str, ...]] = None,
) -> Callable:
    """Decorator to mark a class as an ECS system with phase scheduling.

    Args:
        phase: Frame phase for execution ("animation", "update", "render", etc.)
        order: Execution order within phase (lower = earlier).
               0 = animation graph, 1 = IK, 2 = procedural, 3 = skinning
        priority: Legacy priority field (deprecated, use order instead)
        reads: Component types this system reads from
        writes: Component types this system writes to

    Returns:
        Decorated class with system metadata.
    """
    def decorator(cls: type) -> type:
        cls._system_phase = phase
        cls._system_order = order
        cls._system_priority = priority if priority else order
        cls._system_reads = reads or ()
        cls._system_writes = writes or ()
        return cls
    return decorator


# =============================================================================
# ENUMS
# =============================================================================


class SkinningMethod(Enum):
    """Skinning algorithm selection."""
    LBS = auto()  # Linear Blend Skinning - fast, standard
    DQS = auto()  # Dual Quaternion Skinning - volume preserving
    AUTO = auto()  # Automatically select based on joint angles


class SkinningBackend(Enum):
    """Skinning compute backend selection."""
    GPU_COMPUTE = auto()  # GPU compute shader (default)
    GPU_VERTEX = auto()   # GPU vertex shader skinning
    CPU_SIMD = auto()     # CPU with SIMD optimization
    CPU_SCALAR = auto()   # CPU scalar fallback


class LODInfluenceLevel(Enum):
    """LOD-based bone influence count."""
    HIGH = 4      # LOD 0: 4 influences per vertex
    MEDIUM = 2    # LOD 1-2: 2 influences per vertex
    LOW = 1       # LOD 3+: 1 influence per vertex


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class BoneInfluence:
    """Bone influence on a vertex.

    Attributes:
        bone_index: Index of influencing bone
        weight: Influence weight (0-1)
    """
    bone_index: int = 0
    weight: float = 0.0


@dataclass
class VertexSkinData:
    """Skinning data for a single vertex.

    Typically 4 bone influences per vertex for compatibility.
    """
    influences: List[BoneInfluence] = field(default_factory=list)

    @property
    def bone_count(self) -> int:
        return len(self.influences)

    def normalize(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(inf.weight for inf in self.influences)
        if total > 0:
            for inf in self.influences:
                inf.weight /= total


@dataclass
class SkinningData:
    """Complete skinning data for a mesh.

    Attributes:
        vertex_data: Per-vertex skinning information
        bind_poses: Bind pose (inverse of rest pose) per bone
        bone_names: Names of bones
        max_influences: Maximum influences per vertex
    """
    vertex_data: List[VertexSkinData] = field(default_factory=list)
    bind_poses: List[Mat4] = field(default_factory=list)
    bone_names: List[str] = field(default_factory=list)
    max_influences: int = SKINNING_CONFIG.DEFAULT_MAX_INFLUENCES

    @property
    def vertex_count(self) -> int:
        return len(self.vertex_data)

    @property
    def bone_count(self) -> int:
        return len(self.bind_poses)


@dataclass
class MeshData:
    """Simple mesh data representation."""
    positions: List[Vec3] = field(default_factory=list)
    normals: List[Vec3] = field(default_factory=list)
    tangents: List[Vec4] = field(default_factory=list)

    @property
    def vertex_count(self) -> int:
        return len(self.positions)


@dataclass
class GPUDispatchConfig:
    """Configuration for GPU compute dispatch.

    Attributes:
        workgroup_size_x: Compute shader workgroup size X
        workgroup_size_y: Compute shader workgroup size Y (usually 1 for skinning)
        workgroup_size_z: Compute shader workgroup size Z (usually 1)
        max_vertices_per_dispatch: Max vertices per dispatch
        use_async_compute: Enable async compute overlap
        enable_debug_markers: Add GPU debug markers
    """
    workgroup_size_x: int = 64
    workgroup_size_y: int = 1
    workgroup_size_z: int = 1
    max_vertices_per_dispatch: int = 65536
    use_async_compute: bool = True
    enable_debug_markers: bool = False


@dataclass
class SkinningDispatch:
    """Represents a single skinning dispatch operation.

    Attributes:
        entity: Entity being skinned
        vertex_count: Number of vertices to skin
        bone_count: Number of bones
        influence_count: Influences per vertex (1, 2, or 4)
        method: Skinning method (LBS or DQS)
        backend: Compute backend to use
        priority: Dispatch priority (for batching)
    """
    entity: Any  # Entity type
    vertex_count: int = 0
    bone_count: int = 0
    influence_count: int = 4
    method: SkinningMethod = SkinningMethod.LBS
    backend: SkinningBackend = SkinningBackend.GPU_COMPUTE
    priority: int = 0


@dataclass
class SkinningBatch:
    """Batch of skinning dispatches for efficient processing.

    Groups dispatches by method and influence count for optimal
    GPU dispatch batching.

    Attributes:
        dispatches: List of skinning dispatches in this batch
        total_vertices: Total vertex count across all dispatches
        method: Shared skinning method for the batch
        influence_count: Shared influence count for the batch
    """
    dispatches: List[SkinningDispatch] = field(default_factory=list)
    total_vertices: int = 0
    method: SkinningMethod = SkinningMethod.LBS
    influence_count: int = 4

    def add_dispatch(self, dispatch: SkinningDispatch) -> bool:
        """Add dispatch to batch if compatible.

        Returns:
            True if added, False if incompatible
        """
        if self.dispatches:
            # Must match method and influence count for batching
            if (dispatch.method != self.method or
                dispatch.influence_count != self.influence_count):
                return False

        self.dispatches.append(dispatch)
        self.total_vertices += dispatch.vertex_count
        if not self.dispatches[:-1]:  # First dispatch sets batch params
            self.method = dispatch.method
            self.influence_count = dispatch.influence_count
        return True


@dataclass
class SkinningStats:
    """Runtime statistics for skinning system.

    Attributes:
        entities_skinned: Number of entities processed
        vertices_skinned: Total vertices skinned
        gpu_dispatches: Number of GPU compute dispatches
        cpu_fallback_count: Times CPU fallback was used
        async_overlaps: Async compute overlaps achieved
        batches_processed: Number of batches processed
        total_time_ms: Total processing time in milliseconds
        gpu_time_ms: GPU compute time in milliseconds
        cpu_time_ms: CPU fallback time in milliseconds
    """
    entities_skinned: int = 0
    vertices_skinned: int = 0
    gpu_dispatches: int = 0
    cpu_fallback_count: int = 0
    async_overlaps: int = 0
    batches_processed: int = 0
    total_time_ms: float = 0.0
    gpu_time_ms: float = 0.0
    cpu_time_ms: float = 0.0

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.entities_skinned = 0
        self.vertices_skinned = 0
        self.gpu_dispatches = 0
        self.cpu_fallback_count = 0
        self.async_overlaps = 0
        self.batches_processed = 0
        self.total_time_ms = 0.0
        self.gpu_time_ms = 0.0
        self.cpu_time_ms = 0.0


@dataclass
class LODComponent:
    """LOD component for distance-based level of detail.

    Attributes:
        current_lod: Current LOD level (0 = highest detail)
        distance: Distance from camera
        transition_factor: Blend factor during transitions (0-1)
    """
    current_lod: int = 0
    distance: float = 0.0
    transition_factor: float = 1.0


@dataclass
class GPUBufferHandle:
    """Handle to a GPU buffer resource.

    Placeholder for actual GPU buffer management integration.
    """
    buffer_id: int = 0
    size_bytes: int = 0
    is_valid: bool = False
    is_mapped: bool = False


@dataclass
class SkinnedMeshComponent:
    """Component for skinned mesh entities.

    Attributes:
        mesh: Source mesh data
        skinning_data: Skinning weights and bind poses
        method: Skinning method to use (LBS, DQS, AUTO)
        enabled: Whether skinning is enabled
        lod_level: Current LOD level for influence reduction
        force_cpu: Force CPU skinning (for debugging)
        skinned_positions: Output skinned positions
        skinned_normals: Output skinned normals
        skinning_matrices: Computed skinning matrices
        bone_matrices_buffer: Flattened 4x4 matrices for GPU
        gpu_vertex_buffer: Handle to GPU vertex buffer
        gpu_bone_buffer: Handle to GPU bone matrix buffer
        cache: Skinning matrix cache
        last_update_frame: Last frame this was updated
    """
    mesh: Optional[MeshData] = None
    skinning_data: Optional[SkinningData] = None
    method: SkinningMethod = SkinningMethod.LBS
    enabled: bool = True
    lod_level: int = 0
    force_cpu: bool = False

    # Output data
    skinned_positions: List[Vec3] = field(default_factory=list)
    skinned_normals: List[Vec3] = field(default_factory=list)
    skinning_matrices: List[Mat4] = field(default_factory=list)

    # For GPU skinning
    bone_matrices_buffer: List[float] = field(default_factory=list)
    gpu_vertex_buffer: Optional[GPUBufferHandle] = None
    gpu_bone_buffer: Optional[GPUBufferHandle] = None

    # Cache and tracking
    cache: Optional[SkinningCache] = None
    last_update_frame: int = -1

    def __post_init__(self) -> None:
        if self.cache is None:
            self.cache = SkinningCache()

    def prepare_gpu_buffer(self) -> None:
        """Prepare flattened bone matrix buffer for GPU upload."""
        self.bone_matrices_buffer = []
        for mat in self.skinning_matrices:
            self.bone_matrices_buffer.extend(mat.m)

    def get_memory_size_bytes(self) -> int:
        """Calculate memory size of skinning data."""
        size = 0
        if self.mesh:
            size += self.mesh.vertex_count * (3 * 4 + 3 * 4)  # positions + normals
        if self.skinning_data:
            size += self.skinning_data.vertex_count * self.skinning_data.max_influences * (4 + 4)
            size += self.skinning_data.bone_count * 16 * 4  # bind poses
        return size

    def get_influence_count_for_lod(self) -> int:
        """Get bone influence count based on LOD level.

        Returns:
            4 for LOD 0, 2 for LOD 1-2, 1 for LOD 3+
        """
        if self.lod_level <= 0:
            return 4  # HIGH: 4 influences
        elif self.lod_level <= 2:
            return 2  # MEDIUM: 2 influences
        else:
            return 1  # LOW: 1 influence

    def invalidate_cache(self) -> None:
        """Force cache invalidation."""
        if self.cache:
            self.cache.invalidate()


@dataclass
class GPUCapabilities:
    """GPU compute capability detection.

    Attributes:
        has_compute_shader: GPU supports compute shaders
        has_async_compute: GPU supports async compute
        max_workgroup_size: Maximum workgroup size
        max_shared_memory: Maximum shared memory per workgroup
        compute_queue_count: Number of compute queues
    """
    has_compute_shader: bool = True
    has_async_compute: bool = True
    max_workgroup_size: int = 1024
    max_shared_memory: int = 32768
    compute_queue_count: int = 1


# =============================================================================
# GPU SKINNING DISPATCHER
# =============================================================================


class GPUSkinningDispatcher:
    """Handles GPU compute shader dispatch for skinning.

    Manages shader selection, buffer preparation, and async compute
    coordination for efficient GPU skinning.
    """

    # Compute shader identifiers
    SHADER_LBS_4BONES = "skinning_lbs_4"
    SHADER_LBS_2BONES = "skinning_lbs_2"
    SHADER_LBS_1BONE = "skinning_lbs_1"
    SHADER_DQS_4BONES = "skinning_dqs_4"
    SHADER_DQS_2BONES = "skinning_dqs_2"
    SHADER_DQS_1BONE = "skinning_dqs_1"

    def __init__(
        self,
        config: Optional[GPUDispatchConfig] = None,
        capabilities: Optional[GPUCapabilities] = None,
    ) -> None:
        """Initialize GPU skinning dispatcher.

        Args:
            config: Dispatch configuration
            capabilities: GPU capability information
        """
        self._config = config or GPUDispatchConfig()
        self._capabilities = capabilities or GPUCapabilities()
        self._pending_dispatches: List[SkinningDispatch] = []
        self._active_async_count: int = 0
        self._shader_cache: Dict[str, Any] = {}
        self._is_initialized: bool = False

    @property
    def is_available(self) -> bool:
        """Check if GPU compute is available."""
        return self._capabilities.has_compute_shader

    @property
    def supports_async_compute(self) -> bool:
        """Check if async compute is supported."""
        return (self._capabilities.has_async_compute and
                self._config.use_async_compute)

    def initialize(self) -> bool:
        """Initialize GPU resources for skinning.

        Returns:
            True if initialization succeeded
        """
        if self._is_initialized:
            return True

        if not self._capabilities.has_compute_shader:
            return False

        # Load/compile skinning compute shaders
        self._shader_cache = {
            self.SHADER_LBS_4BONES: self._create_shader_handle("lbs_4"),
            self.SHADER_LBS_2BONES: self._create_shader_handle("lbs_2"),
            self.SHADER_LBS_1BONE: self._create_shader_handle("lbs_1"),
            self.SHADER_DQS_4BONES: self._create_shader_handle("dqs_4"),
            self.SHADER_DQS_2BONES: self._create_shader_handle("dqs_2"),
            self.SHADER_DQS_1BONE: self._create_shader_handle("dqs_1"),
        }

        self._is_initialized = True
        return True

    def _create_shader_handle(self, shader_name: str) -> Dict[str, Any]:
        """Create a shader handle (placeholder for actual GPU API).

        Args:
            shader_name: Name of the shader variant

        Returns:
            Shader handle dictionary
        """
        return {
            "name": shader_name,
            "handle": hash(shader_name),  # Placeholder handle
            "is_loaded": True,
        }

    def select_shader(
        self,
        method: SkinningMethod,
        influence_count: int,
    ) -> str:
        """Select appropriate shader based on method and influence count.

        Args:
            method: LBS or DQS skinning method
            influence_count: Number of bone influences (1, 2, or 4)

        Returns:
            Shader identifier string
        """
        if method == SkinningMethod.DQS:
            if influence_count >= 4:
                return self.SHADER_DQS_4BONES
            elif influence_count >= 2:
                return self.SHADER_DQS_2BONES
            else:
                return self.SHADER_DQS_1BONE
        else:  # LBS or AUTO defaults to LBS
            if influence_count >= 4:
                return self.SHADER_LBS_4BONES
            elif influence_count >= 2:
                return self.SHADER_LBS_2BONES
            else:
                return self.SHADER_LBS_1BONE

    def calculate_workgroups(self, vertex_count: int) -> Tuple[int, int, int]:
        """Calculate compute workgroup dispatch dimensions.

        Args:
            vertex_count: Total vertices to process

        Returns:
            Tuple of (x, y, z) workgroup counts
        """
        workgroup_size = self._config.workgroup_size_x
        workgroups_x = (vertex_count + workgroup_size - 1) // workgroup_size
        return (workgroups_x, 1, 1)

    def dispatch_skinning(
        self,
        component: SkinnedMeshComponent,
        bone_transforms: Dict[int, Transform],
        use_async: bool = False,
    ) -> bool:
        """Dispatch GPU skinning for a single mesh.

        Args:
            component: Skinned mesh component
            bone_transforms: Current bone world transforms
            use_async: Use async compute if available

        Returns:
            True if dispatch succeeded
        """
        if not self._is_initialized:
            return False

        if not component.mesh or not component.skinning_data:
            return False

        # Select shader based on method and LOD
        influence_count = component.get_influence_count_for_lod()
        shader_id = self.select_shader(component.method, influence_count)

        if shader_id not in self._shader_cache:
            return False

        # Compute skinning matrices
        self._compute_skinning_matrices(component, bone_transforms)

        # Prepare GPU buffers
        component.prepare_gpu_buffer()

        # Calculate dispatch dimensions
        vertex_count = component.mesh.vertex_count
        workgroups = self.calculate_workgroups(vertex_count)

        # Create dispatch record
        dispatch = SkinningDispatch(
            entity=None,  # Set externally
            vertex_count=vertex_count,
            bone_count=component.skinning_data.bone_count,
            influence_count=influence_count,
            method=component.method,
            backend=(SkinningBackend.GPU_COMPUTE if not use_async
                     else SkinningBackend.GPU_COMPUTE),
        )

        # Perform dispatch (placeholder for actual GPU API call)
        self._execute_dispatch(shader_id, workgroups, component, use_async)

        return True

    def _compute_skinning_matrices(
        self,
        component: SkinnedMeshComponent,
        bone_transforms: Dict[int, Transform],
    ) -> None:
        """Compute skinning matrices from bone transforms.

        Args:
            component: Skinned mesh component
            bone_transforms: Current bone transforms
        """
        skinning_data = component.skinning_data
        if not skinning_data:
            return

        component.skinning_matrices = []

        for i, bind_pose in enumerate(skinning_data.bind_poses):
            # Get current bone transform
            bone_transform = bone_transforms.get(i, Transform.identity())
            world_matrix = bone_transform.to_matrix()

            # Skinning matrix = world * bind_pose_inverse
            skinning_matrix = world_matrix @ bind_pose
            component.skinning_matrices.append(skinning_matrix)

    def _execute_dispatch(
        self,
        shader_id: str,
        workgroups: Tuple[int, int, int],
        component: SkinnedMeshComponent,
        use_async: bool,
    ) -> None:
        """Execute GPU compute dispatch.

        This is a placeholder for actual GPU API integration.
        In production, this would:
        1. Bind the compute shader
        2. Bind vertex/bone buffers
        3. Set uniform data
        4. Dispatch compute
        5. Insert memory barrier

        Args:
            shader_id: Shader to use
            workgroups: Dispatch dimensions
            component: Skinned mesh component
            use_async: Use async compute queue
        """
        # Placeholder: In real implementation, this would dispatch to GPU
        # gpu.bind_shader(self._shader_cache[shader_id])
        # gpu.bind_buffer(0, component.gpu_vertex_buffer)
        # gpu.bind_buffer(1, component.gpu_bone_buffer)
        # gpu.dispatch(workgroups)
        pass

    def dispatch_batch(
        self,
        batch: SkinningBatch,
        components: Dict[Any, SkinnedMeshComponent],
        bone_transforms: Dict[Any, Dict[int, Transform]],
    ) -> int:
        """Dispatch a batch of skinning operations.

        Batches dispatches with same method/influence count for efficiency.

        Args:
            batch: Batch of dispatches
            components: Entity -> Component mapping
            bone_transforms: Entity -> BoneTransforms mapping

        Returns:
            Number of successful dispatches
        """
        if not self._is_initialized or not batch.dispatches:
            return 0

        success_count = 0

        # Select shader for batch (all use same method/influence count)
        shader_id = self.select_shader(batch.method, batch.influence_count)

        for dispatch in batch.dispatches:
            entity = dispatch.entity
            if entity not in components:
                continue

            component = components[entity]
            entity_bone_transforms = bone_transforms.get(entity, {})

            if self.dispatch_skinning(
                component,
                entity_bone_transforms,
                use_async=self.supports_async_compute,
            ):
                success_count += 1

        return success_count

    def begin_async_region(self) -> bool:
        """Begin async compute region for render overlap.

        Returns:
            True if async compute started successfully
        """
        if not self.supports_async_compute:
            return False

        # Placeholder: Begin async compute queue
        # gpu.begin_async_compute()
        self._active_async_count += 1
        return True

    def end_async_region(self) -> None:
        """End async compute region and synchronize."""
        if self._active_async_count > 0:
            # Placeholder: End async compute and barrier
            # gpu.end_async_compute()
            # gpu.memory_barrier()
            self._active_async_count -= 1

    def shutdown(self) -> None:
        """Release GPU resources."""
        self._shader_cache.clear()
        self._pending_dispatches.clear()
        self._is_initialized = False


# =============================================================================
# CPU SKINNING FALLBACK
# =============================================================================


class CPUSkinningFallback:
    """CPU fallback for skinning when GPU compute unavailable.

    Provides SIMD-optimized and scalar CPU skinning implementations.
    """

    def __init__(self, use_simd: bool = True) -> None:
        """Initialize CPU skinning fallback.

        Args:
            use_simd: Try to use SIMD instructions
        """
        self._use_simd = use_simd
        self._simd_available = self._detect_simd()

    def _detect_simd(self) -> bool:
        """Detect SIMD availability.

        Returns:
            True if SIMD (SSE/AVX/NEON) is available
        """
        # Placeholder: Would check for numpy/scipy with SIMD
        # or native SIMD intrinsics
        try:
            import numpy as np
            return True
        except ImportError:
            return False

    @property
    def backend(self) -> SkinningBackend:
        """Get the active CPU backend."""
        if self._use_simd and self._simd_available:
            return SkinningBackend.CPU_SIMD
        return SkinningBackend.CPU_SCALAR

    def skin_mesh_lbs(
        self,
        component: SkinnedMeshComponent,
        influence_count: int = 4,
    ) -> None:
        """Perform LBS skinning on CPU.

        Args:
            component: Skinned mesh component with matrices computed
            influence_count: Number of influences to use (LOD reduction)
        """
        mesh = component.mesh
        skinning_data = component.skinning_data

        if not mesh or not skinning_data:
            return

        component.skinned_positions = []
        component.skinned_normals = []

        for i, vertex_data in enumerate(skinning_data.vertex_data):
            if i >= mesh.vertex_count:
                break

            pos = mesh.positions[i]
            normal = mesh.normals[i] if i < len(mesh.normals) else Vec3(0, 1, 0)

            skinned_pos = Vec3.zero()
            skinned_normal = Vec3.zero()

            # Process only up to influence_count bones (LOD reduction)
            influences_used = 0
            total_weight = 0.0

            for influence in vertex_data.influences:
                if influences_used >= influence_count:
                    break
                if influence.weight <= SKINNING_CONFIG.MIN_WEIGHT_THRESHOLD:
                    continue

                bone_idx = influence.bone_index
                if bone_idx >= len(component.skinning_matrices):
                    continue

                mat = component.skinning_matrices[bone_idx]
                weight = influence.weight

                # Transform position
                transformed_pos = mat.transform_point(pos)
                skinned_pos = skinned_pos + transformed_pos * weight

                # Transform normal (using upper-left 3x3)
                transformed_normal = mat.transform_direction(normal)
                skinned_normal = skinned_normal + transformed_normal * weight

                total_weight += weight
                influences_used += 1

            # Normalize if we didn't use all influences
            if total_weight > 0 and total_weight < 0.99:
                inv_weight = 1.0 / total_weight
                skinned_pos = skinned_pos * inv_weight
                skinned_normal = skinned_normal * inv_weight

            component.skinned_positions.append(skinned_pos)
            component.skinned_normals.append(skinned_normal.normalized())

    def skin_mesh_dqs(
        self,
        component: SkinnedMeshComponent,
        influence_count: int = 4,
    ) -> None:
        """Perform DQS skinning on CPU.

        Args:
            component: Skinned mesh component with matrices computed
            influence_count: Number of influences to use (LOD reduction)
        """
        mesh = component.mesh
        skinning_data = component.skinning_data

        if not mesh or not skinning_data:
            return

        # Convert skinning matrices to dual quaternions
        dual_quats = [self._mat4_to_dual_quat(mat) for mat in component.skinning_matrices]

        component.skinned_positions = []
        component.skinned_normals = []

        for i, vertex_data in enumerate(skinning_data.vertex_data):
            if i >= mesh.vertex_count:
                break

            pos = mesh.positions[i]
            normal = mesh.normals[i] if i < len(mesh.normals) else Vec3(0, 1, 0)

            # Blend dual quaternions with LOD reduction
            blended_real = Quat(0, 0, 0, 0)
            blended_dual = Quat(0, 0, 0, 0)
            first_real = None
            influences_used = 0
            total_weight = 0.0

            for influence in vertex_data.influences:
                if influences_used >= influence_count:
                    break
                if influence.weight <= SKINNING_CONFIG.MIN_WEIGHT_THRESHOLD:
                    continue

                bone_idx = influence.bone_index
                if bone_idx >= len(dual_quats):
                    continue

                real, dual = dual_quats[bone_idx]
                weight = influence.weight

                # Ensure consistent hemisphere
                if first_real is None:
                    first_real = real
                elif real.dot(first_real) < 0:
                    real = Quat(-real.x, -real.y, -real.z, -real.w)
                    dual = Quat(-dual.x, -dual.y, -dual.z, -dual.w)

                blended_real = Quat(
                    blended_real.x + real.x * weight,
                    blended_real.y + real.y * weight,
                    blended_real.z + real.z * weight,
                    blended_real.w + real.w * weight,
                )
                blended_dual = Quat(
                    blended_dual.x + dual.x * weight,
                    blended_dual.y + dual.y * weight,
                    blended_dual.z + dual.z * weight,
                    blended_dual.w + dual.w * weight,
                )

                total_weight += weight
                influences_used += 1

            # Normalize
            length = blended_real.length()
            if length > SKINNING_CONFIG.MIN_QUATERNION_LENGTH:
                blended_real = Quat(
                    blended_real.x / length,
                    blended_real.y / length,
                    blended_real.z / length,
                    blended_real.w / length,
                )
                blended_dual = Quat(
                    blended_dual.x / length,
                    blended_dual.y / length,
                    blended_dual.z / length,
                    blended_dual.w / length,
                )

            # Transform position using dual quaternion
            skinned_pos = self._transform_point_dual_quat(pos, blended_real, blended_dual)

            # Transform normal using just rotation part
            skinned_normal = blended_real.rotate_vector(normal).normalized()

            component.skinned_positions.append(skinned_pos)
            component.skinned_normals.append(skinned_normal)

    def _mat4_to_dual_quat(self, mat: Mat4) -> Tuple[Quat, Quat]:
        """Convert matrix to dual quaternion (rotation + translation)."""
        transform = Transform.from_matrix(mat)
        rot = transform.rotation.normalized()

        t = transform.translation
        dual = Quat(
            0.5 * (t.x * rot.w + t.y * rot.z - t.z * rot.y),
            0.5 * (-t.x * rot.z + t.y * rot.w + t.z * rot.x),
            0.5 * (t.x * rot.y - t.y * rot.x + t.z * rot.w),
            -0.5 * (t.x * rot.x + t.y * rot.y + t.z * rot.z),
        )

        return (rot, dual)

    def _transform_point_dual_quat(
        self,
        point: Vec3,
        real: Quat,
        dual: Quat,
    ) -> Vec3:
        """Transform point using dual quaternion."""
        # Rotate point
        rotated = real.rotate_vector(point)

        # Extract translation from dual quaternion
        t = Vec3(
            2.0 * (-dual.w * real.x + dual.x * real.w - dual.y * real.z + dual.z * real.y),
            2.0 * (-dual.w * real.y + dual.x * real.z + dual.y * real.w - dual.z * real.x),
            2.0 * (-dual.w * real.z - dual.x * real.y + dual.y * real.x + dual.z * real.w),
        )

        return rotated + t


# =============================================================================
# SKINNING SYSTEM
# =============================================================================


@system(
    phase="animation",
    order=3,
    reads=("SkinnedMeshComponent", "LODComponent"),
    writes=("SkinnedMeshComponent",),
)
class SkinningSystem:
    """ECS system for mesh skinning dispatch.

    Dispatches GPU skinning compute shaders with LOD-based influence
    reduction and CPU fallback support. Runs after animation graph (0),
    IK (1), and procedural systems (2).

    Features:
    - GPU compute shader dispatch for LBS and DQS skinning
    - LOD-based influence reduction (4->2->1 bones per vertex)
    - CPU SIMD fallback for platforms without compute support
    - Async compute overlap with render when supported
    - Batched dispatches for efficiency
    """

    def __init__(
        self,
        gpu_config: Optional[GPUDispatchConfig] = None,
        gpu_capabilities: Optional[GPUCapabilities] = None,
        enable_async_compute: bool = True,
        enable_batching: bool = True,
    ) -> None:
        """Initialize skinning system.

        Args:
            gpu_config: GPU dispatch configuration
            gpu_capabilities: GPU capability information
            enable_async_compute: Allow async compute overlap
            enable_batching: Enable batch optimization
        """
        self._gpu_dispatcher = GPUSkinningDispatcher(
            config=gpu_config,
            capabilities=gpu_capabilities,
        )
        self._cpu_fallback = CPUSkinningFallback(use_simd=True)
        self._enable_async_compute = enable_async_compute
        self._enable_batching = enable_batching
        self._stats = SkinningStats()
        self._current_frame = 0
        self._dq_blend_threshold = SKINNING_CONFIG.DQ_BLEND_THRESHOLD

        # Initialize GPU resources
        self._gpu_available = self._gpu_dispatcher.initialize()

    @property
    def stats(self) -> SkinningStats:
        """Get current frame statistics."""
        return self._stats

    @property
    def gpu_available(self) -> bool:
        """Check if GPU compute is available."""
        return self._gpu_available

    def update(
        self,
        world: "World",
        entity_components: List[Tuple["Entity", SkinnedMeshComponent]],
        pose_data: Dict["Entity", Dict[int, Transform]],
        lod_data: Optional[Dict["Entity", LODComponent]] = None,
        delta_time: float = 0.0,
    ) -> None:
        """Update skinning for all entities.

        Main system entry point called each frame by ECS scheduler.

        Args:
            world: ECS world
            entity_components: List of (entity, component) tuples
            pose_data: Bone poses per entity
            lod_data: LOD components per entity (optional)
            delta_time: Time since last frame
        """
        start_time = time.perf_counter()
        self._stats.reset()
        self._current_frame += 1

        # Filter enabled components
        active_components = [
            (entity, comp) for entity, comp in entity_components
            if comp.enabled and comp.skinning_data is not None
        ]

        if not active_components:
            return

        # Update LOD levels
        if lod_data:
            self._update_lod_levels(active_components, lod_data)

        # Create dispatch list
        dispatches = self._create_dispatches(active_components, pose_data)

        # Sort and batch dispatches
        if self._enable_batching:
            batches = self._batch_dispatches(dispatches)
        else:
            batches = [self._create_single_batch(d) for d in dispatches]

        # Begin async compute region if supported
        async_started = False
        if (self._enable_async_compute and
            self._gpu_available and
            self._gpu_dispatcher.supports_async_compute):
            async_started = self._gpu_dispatcher.begin_async_region()
            if async_started:
                self._stats.async_overlaps += 1

        # Process batches
        for batch in batches:
            self._process_batch(
                batch,
                {e: c for e, c in active_components},
                pose_data,
            )
            self._stats.batches_processed += 1

        # End async compute region
        if async_started:
            self._gpu_dispatcher.end_async_region()

        # Update statistics
        self._stats.entities_skinned = len(active_components)
        end_time = time.perf_counter()
        self._stats.total_time_ms = (end_time - start_time) * 1000.0

    def _update_lod_levels(
        self,
        entity_components: List[Tuple["Entity", SkinnedMeshComponent]],
        lod_data: Dict["Entity", LODComponent],
    ) -> None:
        """Update LOD levels on skinned mesh components.

        Args:
            entity_components: Active entity/component pairs
            lod_data: LOD components per entity
        """
        for entity, component in entity_components:
            lod = lod_data.get(entity)
            if lod:
                component.lod_level = lod.current_lod

    def _create_dispatches(
        self,
        entity_components: List[Tuple["Entity", SkinnedMeshComponent]],
        pose_data: Dict["Entity", Dict[int, Transform]],
    ) -> List[SkinningDispatch]:
        """Create dispatch records for all entities.

        Args:
            entity_components: Active entity/component pairs
            pose_data: Bone poses per entity

        Returns:
            List of skinning dispatch records
        """
        dispatches = []

        for entity, component in entity_components:
            if not component.mesh or not component.skinning_data:
                continue

            # Determine skinning method
            method = component.method
            if method == SkinningMethod.AUTO:
                method = self._select_method_auto(component, pose_data.get(entity, {}))

            # Get influence count from LOD
            influence_count = component.get_influence_count_for_lod()

            # Determine backend
            if component.force_cpu or not self._gpu_available:
                backend = self._cpu_fallback.backend
            else:
                backend = SkinningBackend.GPU_COMPUTE

            dispatch = SkinningDispatch(
                entity=entity,
                vertex_count=component.mesh.vertex_count,
                bone_count=component.skinning_data.bone_count,
                influence_count=influence_count,
                method=method,
                backend=backend,
            )
            dispatches.append(dispatch)

        return dispatches

    def _select_method_auto(
        self,
        component: SkinnedMeshComponent,
        bone_transforms: Dict[int, Transform],
    ) -> SkinningMethod:
        """Automatically select LBS or DQS based on joint angles.

        DQS is preferred when joint angles are large to avoid
        candy-wrapper artifacts.

        Args:
            component: Skinned mesh component
            bone_transforms: Current bone transforms

        Returns:
            Selected skinning method
        """
        if not bone_transforms:
            return SkinningMethod.LBS

        # Check for large joint rotations that would benefit from DQS
        max_angle = 0.0
        for bone_idx, transform in bone_transforms.items():
            # Estimate rotation angle from quaternion
            quat = transform.rotation
            angle = 2.0 * math.acos(min(abs(quat.w), 1.0))
            max_angle = max(max_angle, angle)

        # Use DQS if any joint exceeds threshold
        if max_angle > self._dq_blend_threshold * math.pi:
            return SkinningMethod.DQS

        return SkinningMethod.LBS

    def _batch_dispatches(
        self,
        dispatches: List[SkinningDispatch],
    ) -> List[SkinningBatch]:
        """Batch dispatches by method and influence count.

        Args:
            dispatches: List of dispatch records

        Returns:
            List of optimized batches
        """
        # Group by (method, influence_count, backend)
        groups: Dict[Tuple[SkinningMethod, int, SkinningBackend], List[SkinningDispatch]] = {}

        for dispatch in dispatches:
            key = (dispatch.method, dispatch.influence_count, dispatch.backend)
            if key not in groups:
                groups[key] = []
            groups[key].append(dispatch)

        # Create batches from groups
        batches = []
        for (method, influence, backend), group_dispatches in groups.items():
            batch = SkinningBatch(
                dispatches=group_dispatches,
                total_vertices=sum(d.vertex_count for d in group_dispatches),
                method=method,
                influence_count=influence,
            )
            batches.append(batch)

        return batches

    def _create_single_batch(self, dispatch: SkinningDispatch) -> SkinningBatch:
        """Create a batch from a single dispatch.

        Args:
            dispatch: Single dispatch record

        Returns:
            Batch containing the dispatch
        """
        return SkinningBatch(
            dispatches=[dispatch],
            total_vertices=dispatch.vertex_count,
            method=dispatch.method,
            influence_count=dispatch.influence_count,
        )

    def _process_batch(
        self,
        batch: SkinningBatch,
        components: Dict["Entity", SkinnedMeshComponent],
        pose_data: Dict["Entity", Dict[int, Transform]],
    ) -> None:
        """Process a batch of skinning dispatches.

        Args:
            batch: Batch to process
            components: Entity -> Component mapping
            pose_data: Entity -> BoneTransforms mapping
        """
        for dispatch in batch.dispatches:
            entity = dispatch.entity
            if entity not in components:
                continue

            component = components[entity]
            bone_transforms = pose_data.get(entity, {})

            # Compute skinning matrices
            self._compute_skinning_matrices(component, bone_transforms)

            if dispatch.backend in (SkinningBackend.CPU_SIMD, SkinningBackend.CPU_SCALAR):
                # CPU fallback path
                self._skin_cpu(component, dispatch)
                self._stats.cpu_fallback_count += 1
                cpu_start = time.perf_counter()
                # Time already spent in _skin_cpu
                cpu_end = time.perf_counter()
            else:
                # GPU compute path
                success = self._gpu_dispatcher.dispatch_skinning(
                    component,
                    bone_transforms,
                    use_async=self._enable_async_compute,
                )
                if success:
                    self._stats.gpu_dispatches += 1
                else:
                    # Fall back to CPU on GPU failure
                    self._skin_cpu(component, dispatch)
                    self._stats.cpu_fallback_count += 1

            self._stats.vertices_skinned += dispatch.vertex_count
            component.last_update_frame = self._current_frame

    def _compute_skinning_matrices(
        self,
        component: SkinnedMeshComponent,
        bone_transforms: Dict[int, Transform],
    ) -> None:
        """Compute skinning matrices from bone transforms and bind poses.

        Args:
            component: Skinned mesh component
            bone_transforms: Current bone world transforms
        """
        skinning_data = component.skinning_data
        if not skinning_data:
            return

        component.skinning_matrices = []

        for i, bind_pose in enumerate(skinning_data.bind_poses):
            # Get current bone transform
            bone_transform = bone_transforms.get(i, Transform.identity())
            world_matrix = bone_transform.to_matrix()

            # Skinning matrix = world * bind_pose_inverse
            skinning_matrix = world_matrix @ bind_pose
            component.skinning_matrices.append(skinning_matrix)

    def _skin_cpu(
        self,
        component: SkinnedMeshComponent,
        dispatch: SkinningDispatch,
    ) -> None:
        """Perform CPU skinning fallback.

        Args:
            component: Skinned mesh component
            dispatch: Dispatch information
        """
        influence_count = dispatch.influence_count

        if dispatch.method == SkinningMethod.DQS:
            self._cpu_fallback.skin_mesh_dqs(component, influence_count)
        else:
            self._cpu_fallback.skin_mesh_lbs(component, influence_count)

    def compute_bounding_box(
        self,
        component: SkinnedMeshComponent,
    ) -> Optional[Tuple[Vec3, Vec3]]:
        """Compute bounding box of skinned mesh.

        Args:
            component: Skinned mesh component

        Returns:
            Tuple of (min, max) corners, or None if no data
        """
        positions = component.skinned_positions or (
            component.mesh.positions if component.mesh else None
        )
        if not positions:
            return None

        min_corner = Vec3(float('inf'), float('inf'), float('inf'))
        max_corner = Vec3(float('-inf'), float('-inf'), float('-inf'))

        for pos in positions:
            min_corner = Vec3(
                min(min_corner.x, pos.x),
                min(min_corner.y, pos.y),
                min(min_corner.z, pos.z),
            )
            max_corner = Vec3(
                max(max_corner.x, pos.x),
                max(max_corner.y, pos.y),
                max(max_corner.z, pos.z),
            )

        return (min_corner, max_corner)

    def shutdown(self) -> None:
        """Release resources and shutdown system."""
        self._gpu_dispatcher.shutdown()
