"""ECS system for mesh skinning.

Computes skinning matrices and transforms vertices for rendering.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Any, Sequence, Tuple, Optional, List, Dict

from engine.core.math import Vec3, Vec4, Quat, Mat4, Transform
from engine.core.ecs import Entity, World
from engine.animation.config import SKINNING_CONFIG


class SkinningMethod(Enum):
    """Skinning computation method."""
    LINEAR = auto()  # Linear blend skinning (LBS)
    DUAL_QUATERNION = auto()  # Dual quaternion skinning
    GPU = auto()  # GPU-based skinning
    # Aliases for compatibility
    LBS = LINEAR
    DQS = DUAL_QUATERNION
    AUTO = auto()  # Auto-select based on joint angles


class SkinningBackend(Enum):
    """Where skinning computation runs."""
    CPU = auto()
    CPU_SCALAR = auto()
    CPU_SIMD = auto()
    GPU_COMPUTE = auto()
    GPU_VERTEX = auto()


class LODInfluenceLevel(Enum):
    """Bone influence count per LOD."""
    LOD0 = 4  # 4 bones per vertex
    LOD1 = 2  # 2 bones per vertex
    LOD2 = 1  # 1 bone per vertex


@dataclass
class GPUBufferHandle:
    """Handle to GPU buffer."""
    id: int = 0
    size: int = 0


@dataclass
class GPUCapabilities:
    """GPU feature support."""
    has_compute_shader: bool = True
    compute_shaders: bool = True  # Alias
    has_async_compute: bool = True
    max_workgroup_size: int = 1024
    max_shared_memory: int = 32768
    compute_queue_count: int = 2


@dataclass
class GPUDispatchConfig:
    """Compute dispatch parameters."""
    workgroup_size_x: int = 64
    workgroup_size_y: int = 1
    workgroup_size_z: int = 1
    workgroup_size: Tuple[int, int, int] = (64, 1, 1)
    workgroup_count: Tuple[int, int, int] = (1, 1, 1)
    max_vertices_per_dispatch: int = 65536
    use_async_compute: bool = True
    enable_debug_markers: bool = False


@dataclass
class SkinningDispatch:
    """Single skinning dispatch."""
    entity: Any = None
    mesh: Any = None
    bone_matrices: Optional[GPUBufferHandle] = None
    output_buffer: Optional[GPUBufferHandle] = None
    vertex_count: int = 0
    method: SkinningMethod = SkinningMethod.LINEAR
    influence_count: int = 4


@dataclass
class SkinningBatch:
    """Batched skinning operations."""
    dispatches: List[SkinningDispatch] = field(default_factory=list)
    method: Optional[SkinningMethod] = None
    influence_count: Optional[int] = None

    def add_dispatch(self, dispatch: SkinningDispatch) -> bool:
        """Add dispatch to batch. Returns False if incompatible."""
        if len(self.dispatches) == 0:
            self.method = dispatch.method
            self.influence_count = dispatch.influence_count
            self.dispatches.append(dispatch)
            return True

        # Check compatibility
        if dispatch.method != self.method or dispatch.influence_count != self.influence_count:
            return False

        self.dispatches.append(dispatch)
        return True

    def add(self, dispatch: SkinningDispatch) -> None:
        """Add dispatch unconditionally."""
        self.dispatches.append(dispatch)

    @property
    def total_vertices(self) -> int:
        return sum(d.vertex_count for d in self.dispatches)


@dataclass
class SkinningStats:
    """Performance statistics."""
    vertices_skinned: int = 0
    entities_skinned: int = 0
    dispatches: int = 0
    gpu_dispatches: int = 0
    cpu_fallback_count: int = 0
    batches_processed: int = 0
    async_overlaps: int = 0
    time_ms: float = 0.0
    total_time_ms: float = 0.0

    def reset(self) -> None:
        """Reset all statistics."""
        self.vertices_skinned = 0
        self.entities_skinned = 0
        self.dispatches = 0
        self.gpu_dispatches = 0
        self.cpu_fallback_count = 0
        self.batches_processed = 0
        self.async_overlaps = 0
        self.time_ms = 0.0
        self.total_time_ms = 0.0


@dataclass
class LODComponent:
    """LOD tracking for skinning."""
    current_lod: int = 0
    distance: float = 0.0
    influence_level: LODInfluenceLevel = LODInfluenceLevel.LOD0


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
    influences: list[BoneInfluence] = field(default_factory=list)

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
    vertex_data: list[VertexSkinData] = field(default_factory=list)
    bind_poses: list[Mat4] = field(default_factory=list)
    bone_names: list[str] = field(default_factory=list)
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
    positions: list[Vec3] = field(default_factory=list)
    normals: list[Vec3] = field(default_factory=list)
    tangents: list[Vec4] = field(default_factory=list)

    @property
    def vertex_count(self) -> int:
        return len(self.positions)


@dataclass
class SkinnedMeshComponent:
    """Component for skinned mesh entities.

    Attributes:
        mesh: Source mesh data
        skinning_data: Skinning weights and bind poses
        method: Skinning method to use
        backend: Skinning backend to use
        enabled: Whether skinning is enabled
        skinned_positions: Output skinned positions
        skinned_normals: Output skinned normals
        skinning_matrices: Computed skinning matrices
        lod_level: Current LOD level (0 = highest quality)
        force_cpu: Force CPU fallback even if GPU available
    """
    mesh: MeshData | None = None
    skinning_data: SkinningData | None = None
    method: SkinningMethod = SkinningMethod.LINEAR
    backend: SkinningBackend = SkinningBackend.GPU_COMPUTE
    enabled: bool = True
    lod_level: int = 0
    force_cpu: bool = False
    cache: Optional[Dict[str, Any]] = field(default_factory=dict)

    # Output data
    skinned_positions: list[Vec3] = field(default_factory=list)
    skinned_normals: list[Vec3] = field(default_factory=list)
    skinning_matrices: list[Mat4] = field(default_factory=list)

    # For GPU skinning
    bone_matrices_buffer: list[float] = field(default_factory=list)  # Flattened 4x4 matrices

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
            size += self.skinning_data.vertex_count * self.skinning_data.max_influences * (4 + 4)  # bone idx + weight
            size += self.skinning_data.bone_count * 16 * 4  # bind poses
        return size

    def get_influence_count_for_lod(self) -> int:
        """Get bone influence count for current LOD level."""
        if self.lod_level == 0:
            return 4
        elif self.lod_level in (1, 2):
            return 2
        else:
            return 1

    def invalidate_cache(self) -> None:
        """Invalidate cached skinning data."""
        if self.cache is not None:
            self.cache.clear()


class GPUSkinningDispatcher:
    """Dispatches GPU skinning work via compute shaders."""

    # Shader identifiers
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
    ):
        self._config = config or GPUDispatchConfig()
        self._capabilities = capabilities or GPUCapabilities()
        self._initialized = False
        self._pipeline = None

    @property
    def is_available(self) -> bool:
        return self._capabilities.has_compute_shader

    @property
    def supports_async_compute(self) -> bool:
        return (
            self._capabilities.has_async_compute and
            self._config.use_async_compute
        )

    def initialize(self) -> bool:
        """Initialize GPU dispatcher."""
        if not self.is_available:
            return False
        self._initialized = True
        return True

    def select_shader(self, method: SkinningMethod, bone_count: int) -> str:
        """Select appropriate shader based on method and bone count."""
        is_lbs = method in (SkinningMethod.LINEAR, SkinningMethod.LBS, SkinningMethod.GPU)
        if bone_count >= 4:
            return self.SHADER_LBS_4BONES if is_lbs else self.SHADER_DQS_4BONES
        elif bone_count >= 2:
            return self.SHADER_LBS_2BONES if is_lbs else self.SHADER_DQS_2BONES
        else:
            return self.SHADER_LBS_1BONE if is_lbs else self.SHADER_DQS_1BONE

    def calculate_workgroups(self, vertex_count: int) -> Tuple[int, int, int]:
        """Calculate workgroup dispatch dimensions."""
        workgroup_size = self._config.workgroup_size_x
        x = (vertex_count + workgroup_size - 1) // workgroup_size
        return (x, 1, 1)

    def dispatch_skinning(
        self,
        component: SkinnedMeshComponent,
        bone_transforms: Dict[int, Any],
    ) -> bool:
        """Dispatch skinning computation on GPU."""
        if not self._initialized:
            return False
        if not component.mesh:
            return False
        # GPU dispatch would happen here in real implementation
        return True

    def begin_async_region(self) -> bool:
        """Begin async compute region."""
        if not self.supports_async_compute:
            return False
        return True

    def end_async_region(self) -> None:
        """End async compute region."""
        pass

    def dispatch(self, batch: SkinningBatch) -> None:
        """Dispatch a batch of skinning operations."""
        pass


class CPUSkinningFallback:
    """CPU fallback when GPU unavailable."""

    def __init__(self, use_simd: bool = True):
        self._use_simd = use_simd
        self._backend = SkinningBackend.CPU_SIMD if use_simd else SkinningBackend.CPU_SCALAR

    @property
    def backend(self) -> SkinningBackend:
        return self._backend

    def skin_mesh(self, mesh: MeshData, bones: List[Any]) -> None:
        """CPU skinning of mesh."""
        pass

    def skin_mesh_lbs(self, component: SkinnedMeshComponent, max_influences: int) -> None:
        """Linear blend skinning on CPU with limited influences."""
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
            normal = mesh.normals[i] if i < len(mesh.normals) else Vec3.up()

            skinned_pos = Vec3.zero()
            skinned_normal = Vec3.zero()

            # Limit influences
            influences = vertex_data.influences[:max_influences]

            for influence in influences:
                if influence.weight <= 0:
                    continue

                bone_idx = influence.bone_index
                if bone_idx >= len(component.skinning_matrices):
                    continue

                mat = component.skinning_matrices[bone_idx]
                weight = influence.weight

                transformed_pos = mat.transform_point(pos)
                skinned_pos = skinned_pos + transformed_pos * weight

                transformed_normal = mat.transform_direction(normal)
                skinned_normal = skinned_normal + transformed_normal * weight

            component.skinned_positions.append(skinned_pos)
            component.skinned_normals.append(skinned_normal.normalized())

    def skin_mesh_dqs(self, component: SkinnedMeshComponent, max_influences: int) -> None:
        """Dual quaternion skinning on CPU with limited influences."""
        mesh = component.mesh
        skinning_data = component.skinning_data

        if not mesh or not skinning_data:
            return

        # Convert skinning matrices to dual quaternions
        dual_quats = []
        for mat in component.skinning_matrices:
            transform = Transform.from_matrix(mat)
            rot = transform.rotation.normalized()
            t = transform.translation
            dual = Quat(
                0.5 * (t.x * rot.w + t.y * rot.z - t.z * rot.y),
                0.5 * (-t.x * rot.z + t.y * rot.w + t.z * rot.x),
                0.5 * (t.x * rot.y - t.y * rot.x + t.z * rot.w),
                -0.5 * (t.x * rot.x + t.y * rot.y + t.z * rot.z),
            )
            dual_quats.append((rot, dual))

        component.skinned_positions = []
        component.skinned_normals = []

        for i, vertex_data in enumerate(skinning_data.vertex_data):
            if i >= mesh.vertex_count:
                break

            pos = mesh.positions[i]
            normal = mesh.normals[i] if i < len(mesh.normals) else Vec3.up()

            blended_real = Quat(0, 0, 0, 0)
            blended_dual = Quat(0, 0, 0, 0)

            influences = vertex_data.influences[:max_influences]
            first_real = None

            for influence in influences:
                if influence.weight <= 0:
                    continue

                bone_idx = influence.bone_index
                if bone_idx >= len(dual_quats):
                    continue

                real, dual = dual_quats[bone_idx]
                weight = influence.weight

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
            rotated = blended_real.rotate_vector(pos)
            t = Vec3(
                2.0 * (-blended_dual.w * blended_real.x + blended_dual.x * blended_real.w - blended_dual.y * blended_real.z + blended_dual.z * blended_real.y),
                2.0 * (-blended_dual.w * blended_real.y + blended_dual.x * blended_real.z + blended_dual.y * blended_real.w - blended_dual.z * blended_real.x),
                2.0 * (-blended_dual.w * blended_real.z - blended_dual.x * blended_real.y + blended_dual.y * blended_real.x + blended_dual.z * blended_real.w),
            )
            skinned_pos = rotated + t
            skinned_normal = blended_real.rotate_vector(normal).normalized()

            component.skinned_positions.append(skinned_pos)
            component.skinned_normals.append(skinned_normal)


class SkinningSystem:
    """ECS system for mesh skinning.

    Computes skinning matrices and optionally transforms vertices.
    """

    # System annotations for ECS registration
    _system_phase = "animation"
    _system_order = 3
    _system_reads = ["SkinnedMeshComponent", "LODComponent"]
    _system_writes = ["SkinnedMeshComponent"]

    def __init__(
        self,
        gpu_config: Optional[GPUDispatchConfig] = None,
        gpu_capabilities: Optional[GPUCapabilities] = None,
        enable_async_compute: bool = True,
        enable_batching: bool = True,
    ):
        self._use_dual_quaternion_threshold = SKINNING_CONFIG.DQ_BLEND_THRESHOLD
        self._gpu_config = gpu_config or GPUDispatchConfig()
        self._gpu_capabilities = gpu_capabilities or GPUCapabilities()
        self._enable_async_compute = enable_async_compute
        self._enable_batching = enable_batching
        self._stats = SkinningStats()
        self._cpu_fallback = CPUSkinningFallback()
        self._dispatcher: Optional[GPUSkinningDispatcher] = None

        # Check GPU availability
        self._gpu_available = self._gpu_capabilities.has_compute_shader
        if self._gpu_available and gpu_config:
            self._dispatcher = GPUSkinningDispatcher(gpu_config, self._gpu_capabilities)

    @property
    def stats(self) -> SkinningStats:
        return self._stats

    @property
    def gpu_available(self) -> bool:
        return self._gpu_available

    def update(
        self,
        world: World,
        entity_components: list[tuple[Entity, SkinnedMeshComponent]],
        pose_data: dict[Entity, dict[int, Transform]],
        lod_data: Optional[dict[Entity, LODComponent]] = None,
    ) -> None:
        """Update skinning for all entities.

        Args:
            world: ECS world
            entity_components: List of (entity, component) tuples
            pose_data: Bone poses per entity
            lod_data: Optional LOD components per entity
        """
        start_time = time.perf_counter()
        self._stats.reset()

        # Apply LOD data to components
        if lod_data:
            for entity, component in entity_components:
                if entity in lod_data:
                    lod = lod_data[entity]
                    component.lod_level = lod.current_lod

        # Track unique batches (method + influence_count combinations)
        seen_batches: set[tuple] = set()

        # Process each entity
        for entity, component in entity_components:
            if not component.enabled or not component.skinning_data:
                continue

            bone_transforms = pose_data.get(entity, {})

            # Compute skinning matrices
            self._compute_skinning_matrices(component, bone_transforms)

            # Determine if we should use CPU fallback
            use_cpu = component.force_cpu or not self._gpu_available

            # Transform vertices based on method and backend
            method = component.method
            if method == SkinningMethod.AUTO:
                # Auto-select based on joint angles - for now default to LBS
                method = SkinningMethod.LINEAR

            influence_count = component.get_influence_count_for_lod()

            # Track batch for this method/influence combo
            batch_key = (method, influence_count)
            if batch_key not in seen_batches:
                seen_batches.add(batch_key)
                self._stats.batches_processed += 1

            if use_cpu:
                self._stats.cpu_fallback_count += 1
                if method in (SkinningMethod.LINEAR, SkinningMethod.LBS):
                    self._cpu_fallback.skin_mesh_lbs(component, influence_count)
                elif method in (SkinningMethod.DUAL_QUATERNION, SkinningMethod.DQS):
                    self._cpu_fallback.skin_mesh_dqs(component, influence_count)
            elif component.method == SkinningMethod.GPU:
                component.prepare_gpu_buffer()
                self._stats.gpu_dispatches += 1
            elif method in (SkinningMethod.LINEAR, SkinningMethod.LBS):
                self._compute_linear_skinning(component)
            elif method in (SkinningMethod.DUAL_QUATERNION, SkinningMethod.DQS):
                self._compute_dual_quaternion_skinning(component)

            self._stats.entities_skinned += 1
            if component.mesh:
                self._stats.vertices_skinned += component.mesh.vertex_count

        # Track async overlaps if dispatcher is active
        if self._dispatcher and self._enable_async_compute and self._gpu_available:
            self._stats.async_overlaps = 1  # Simplified

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._stats.time_ms = elapsed_ms
        self._stats.total_time_ms = elapsed_ms

    def shutdown(self) -> None:
        """Release all resources."""
        if self._dispatcher:
            self._dispatcher = None

    def _compute_skinning_matrices(
        self,
        component: SkinnedMeshComponent,
        bone_transforms: dict[int, Transform]
    ) -> None:
        """Compute skinning matrices from bone transforms and bind poses."""
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

    def _compute_linear_skinning(self, component: SkinnedMeshComponent) -> None:
        """Compute linear blend skinning on CPU."""
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
            normal = mesh.normals[i] if i < len(mesh.normals) else Vec3.up()

            skinned_pos = Vec3.zero()
            skinned_normal = Vec3.zero()

            for influence in vertex_data.influences:
                if influence.weight <= 0:
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

            component.skinned_positions.append(skinned_pos)
            component.skinned_normals.append(skinned_normal.normalized())

    def _compute_dual_quaternion_skinning(self, component: SkinnedMeshComponent) -> None:
        """Compute dual quaternion skinning on CPU.

        Reduces artifacts at joints compared to linear blend skinning.
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
            normal = mesh.normals[i] if i < len(mesh.normals) else Vec3.up()

            # Blend dual quaternions
            blended_real = Quat(0, 0, 0, 0)
            blended_dual = Quat(0, 0, 0, 0)

            first_real = None
            for influence in vertex_data.influences:
                if influence.weight <= 0:
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

    def _mat4_to_dual_quat(self, mat: Mat4) -> tuple[Quat, Quat]:
        """Convert matrix to dual quaternion (rotation + translation)."""
        # Extract rotation
        transform = Transform.from_matrix(mat)
        rot = transform.rotation.normalized()

        # Dual part encodes translation
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
        dual: Quat
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

    def compute_bounding_box(
        self,
        component: SkinnedMeshComponent
    ) -> tuple[Vec3, Vec3] | None:
        """Compute bounding box of skinned mesh.

        Returns:
            Tuple of (min, max) corners, or None if no data
        """
        positions = component.skinned_positions or (component.mesh.positions if component.mesh else None)
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


def system(func: Callable) -> Callable:
    """Decorator to mark a function as an ECS system."""
    func._is_system = True
    return func
