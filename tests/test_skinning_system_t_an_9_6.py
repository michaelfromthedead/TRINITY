"""Skinning System Integration Tests (T-AN-9.6).

Comprehensive tests for the GPU skinning dispatch system including:
- GPU compute dispatch correctness
- LBS vs DQS selection
- LOD influence reduction (4->2->1)
- CPU fallback activation
- Async compute flag handling
- Batch optimization
- Error handling (missing bones, invalid weights)

Total: 60+ tests
"""

from __future__ import annotations

import math
import time
import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

from engine.core.math.vec import Vec3, Vec4
from engine.core.math.quat import Quat
from engine.core.math.mat import Mat4
from engine.core.math.transform import Transform

from engine.animation.systems.skinning_system import (
    # System decorator
    system,
    # Enums
    SkinningMethod,
    SkinningBackend,
    LODInfluenceLevel,
    # Data structures
    BoneInfluence,
    VertexSkinData,
    SkinningData,
    MeshData,
    GPUDispatchConfig,
    SkinningDispatch,
    SkinningBatch,
    SkinningStats,
    LODComponent,
    GPUBufferHandle,
    GPUCapabilities,
    # Components
    SkinnedMeshComponent,
    # Dispatcher and fallback
    GPUSkinningDispatcher,
    CPUSkinningFallback,
    # System
    SkinningSystem,
)
from engine.animation.config import SKINNING_CONFIG


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass(frozen=True)
class MockEntity:
    """Mock entity for testing."""
    id: int

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class MockWorld:
    """Mock ECS world for testing."""
    pass


@pytest.fixture
def simple_mesh() -> MeshData:
    """Simple cube mesh for testing."""
    positions = [
        Vec3(-1, -1, -1), Vec3(1, -1, -1), Vec3(1, 1, -1), Vec3(-1, 1, -1),
        Vec3(-1, -1, 1), Vec3(1, -1, 1), Vec3(1, 1, 1), Vec3(-1, 1, 1),
    ]
    normals = [
        Vec3(0, 0, -1), Vec3(0, 0, -1), Vec3(0, 0, -1), Vec3(0, 0, -1),
        Vec3(0, 0, 1), Vec3(0, 0, 1), Vec3(0, 0, 1), Vec3(0, 0, 1),
    ]
    return MeshData(positions=positions, normals=normals)


@pytest.fixture
def skinning_data_4_influences() -> SkinningData:
    """Skinning data with 4 influences per vertex."""
    vertex_data = []
    for i in range(8):
        influences = [
            BoneInfluence(0, 0.4),
            BoneInfluence(1, 0.3),
            BoneInfluence(2, 0.2),
            BoneInfluence(3, 0.1),
        ]
        vertex_data.append(VertexSkinData(influences=influences))

    bind_poses = [Mat4.identity() for _ in range(4)]
    bone_names = ["bone0", "bone1", "bone2", "bone3"]

    return SkinningData(
        vertex_data=vertex_data,
        bind_poses=bind_poses,
        bone_names=bone_names,
        max_influences=4,
    )


@pytest.fixture
def skinning_data_single_influence() -> SkinningData:
    """Skinning data with 1 influence per vertex."""
    vertex_data = []
    for i in range(8):
        influences = [BoneInfluence(i % 4, 1.0)]
        vertex_data.append(VertexSkinData(influences=influences))

    bind_poses = [Mat4.identity() for _ in range(4)]
    bone_names = ["bone0", "bone1", "bone2", "bone3"]

    return SkinningData(
        vertex_data=vertex_data,
        bind_poses=bind_poses,
        bone_names=bone_names,
        max_influences=1,
    )


@pytest.fixture
def bone_transforms() -> Dict[int, Transform]:
    """Simple bone transforms."""
    return {
        0: Transform(Vec3(0, 0, 0), Quat.identity(), Vec3.one()),
        1: Transform(Vec3(0, 1, 0), Quat.identity(), Vec3.one()),
        2: Transform(Vec3(0, 2, 0), Quat.identity(), Vec3.one()),
        3: Transform(Vec3(0, 3, 0), Quat.identity(), Vec3.one()),
    }


@pytest.fixture
def rotated_bone_transforms() -> Dict[int, Transform]:
    """Bone transforms with rotation for DQS testing."""
    return {
        0: Transform(Vec3(0, 0, 0), Quat.identity(), Vec3.one()),
        1: Transform(
            Vec3(0, 1, 0),
            Quat.from_axis_angle(Vec3(0, 0, 1), math.pi / 2),  # 90 degree rotation
            Vec3.one()
        ),
        2: Transform(
            Vec3(0, 2, 0),
            Quat.from_axis_angle(Vec3(0, 0, 1), math.pi),  # 180 degree rotation
            Vec3.one()
        ),
        3: Transform(Vec3(0, 3, 0), Quat.identity(), Vec3.one()),
    }


@pytest.fixture
def skinned_mesh_component(simple_mesh, skinning_data_4_influences) -> SkinnedMeshComponent:
    """Configured skinned mesh component."""
    return SkinnedMeshComponent(
        mesh=simple_mesh,
        skinning_data=skinning_data_4_influences,
        method=SkinningMethod.LBS,
        enabled=True,
    )


@pytest.fixture
def gpu_capabilities_full() -> GPUCapabilities:
    """Full GPU capabilities."""
    return GPUCapabilities(
        has_compute_shader=True,
        has_async_compute=True,
        max_workgroup_size=1024,
        max_shared_memory=32768,
        compute_queue_count=2,
    )


@pytest.fixture
def gpu_capabilities_no_compute() -> GPUCapabilities:
    """No compute shader support."""
    return GPUCapabilities(
        has_compute_shader=False,
        has_async_compute=False,
        max_workgroup_size=0,
        max_shared_memory=0,
        compute_queue_count=0,
    )


@pytest.fixture
def gpu_capabilities_no_async() -> GPUCapabilities:
    """Compute but no async."""
    return GPUCapabilities(
        has_compute_shader=True,
        has_async_compute=False,
        max_workgroup_size=1024,
        max_shared_memory=32768,
        compute_queue_count=1,
    )


@pytest.fixture
def gpu_dispatch_config() -> GPUDispatchConfig:
    """Standard dispatch configuration."""
    return GPUDispatchConfig(
        workgroup_size_x=64,
        workgroup_size_y=1,
        workgroup_size_z=1,
        max_vertices_per_dispatch=65536,
        use_async_compute=True,
        enable_debug_markers=False,
    )


@pytest.fixture
def skinning_system(gpu_capabilities_full, gpu_dispatch_config) -> SkinningSystem:
    """Configured skinning system."""
    return SkinningSystem(
        gpu_config=gpu_dispatch_config,
        gpu_capabilities=gpu_capabilities_full,
        enable_async_compute=True,
        enable_batching=True,
    )


@pytest.fixture
def skinning_system_cpu_only(gpu_capabilities_no_compute) -> SkinningSystem:
    """Skinning system with CPU fallback only."""
    return SkinningSystem(
        gpu_capabilities=gpu_capabilities_no_compute,
        enable_async_compute=False,
        enable_batching=True,
    )


# =============================================================================
# System Decorator Tests
# =============================================================================


class TestSystemDecorator:
    """Tests for @system decorator on SkinningSystem."""

    def test_system_phase_annotation(self):
        """Verify system has correct phase annotation."""
        assert hasattr(SkinningSystem, "_system_phase")
        assert SkinningSystem._system_phase == "animation"

    def test_system_order_annotation(self):
        """Verify system has correct order (3 for skinning)."""
        assert hasattr(SkinningSystem, "_system_order")
        assert SkinningSystem._system_order == 3

    def test_system_reads_annotation(self):
        """Verify system declares reads."""
        assert hasattr(SkinningSystem, "_system_reads")
        assert "SkinnedMeshComponent" in SkinningSystem._system_reads
        assert "LODComponent" in SkinningSystem._system_reads

    def test_system_writes_annotation(self):
        """Verify system declares writes."""
        assert hasattr(SkinningSystem, "_system_writes")
        assert "SkinnedMeshComponent" in SkinningSystem._system_writes


# =============================================================================
# GPU Compute Dispatch Tests
# =============================================================================


class TestGPUComputeDispatch:
    """Tests for GPU compute shader dispatch."""

    def test_dispatcher_initialization(self, gpu_capabilities_full, gpu_dispatch_config):
        """Test GPU dispatcher initializes correctly."""
        dispatcher = GPUSkinningDispatcher(gpu_dispatch_config, gpu_capabilities_full)
        assert dispatcher.is_available
        assert dispatcher.supports_async_compute
        assert dispatcher.initialize()

    def test_dispatcher_not_available_without_compute(self, gpu_capabilities_no_compute):
        """Test dispatcher unavailable without compute shaders."""
        dispatcher = GPUSkinningDispatcher(capabilities=gpu_capabilities_no_compute)
        assert not dispatcher.is_available
        assert not dispatcher.initialize()

    def test_dispatcher_no_async_without_support(self, gpu_capabilities_no_async, gpu_dispatch_config):
        """Test async disabled without hardware support."""
        config = GPUDispatchConfig(use_async_compute=True)
        dispatcher = GPUSkinningDispatcher(config, gpu_capabilities_no_async)
        assert dispatcher.is_available
        assert not dispatcher.supports_async_compute

    def test_shader_selection_lbs_4bones(self, gpu_capabilities_full, gpu_dispatch_config):
        """Test LBS 4-bone shader selection."""
        dispatcher = GPUSkinningDispatcher(gpu_dispatch_config, gpu_capabilities_full)
        shader = dispatcher.select_shader(SkinningMethod.LBS, 4)
        assert shader == GPUSkinningDispatcher.SHADER_LBS_4BONES

    def test_shader_selection_lbs_2bones(self, gpu_capabilities_full, gpu_dispatch_config):
        """Test LBS 2-bone shader selection."""
        dispatcher = GPUSkinningDispatcher(gpu_dispatch_config, gpu_capabilities_full)
        shader = dispatcher.select_shader(SkinningMethod.LBS, 2)
        assert shader == GPUSkinningDispatcher.SHADER_LBS_2BONES

    def test_shader_selection_lbs_1bone(self, gpu_capabilities_full, gpu_dispatch_config):
        """Test LBS 1-bone shader selection."""
        dispatcher = GPUSkinningDispatcher(gpu_dispatch_config, gpu_capabilities_full)
        shader = dispatcher.select_shader(SkinningMethod.LBS, 1)
        assert shader == GPUSkinningDispatcher.SHADER_LBS_1BONE

    def test_shader_selection_dqs_4bones(self, gpu_capabilities_full, gpu_dispatch_config):
        """Test DQS 4-bone shader selection."""
        dispatcher = GPUSkinningDispatcher(gpu_dispatch_config, gpu_capabilities_full)
        shader = dispatcher.select_shader(SkinningMethod.DQS, 4)
        assert shader == GPUSkinningDispatcher.SHADER_DQS_4BONES

    def test_shader_selection_dqs_2bones(self, gpu_capabilities_full, gpu_dispatch_config):
        """Test DQS 2-bone shader selection."""
        dispatcher = GPUSkinningDispatcher(gpu_dispatch_config, gpu_capabilities_full)
        shader = dispatcher.select_shader(SkinningMethod.DQS, 2)
        assert shader == GPUSkinningDispatcher.SHADER_DQS_2BONES

    def test_shader_selection_dqs_1bone(self, gpu_capabilities_full, gpu_dispatch_config):
        """Test DQS 1-bone shader selection."""
        dispatcher = GPUSkinningDispatcher(gpu_dispatch_config, gpu_capabilities_full)
        shader = dispatcher.select_shader(SkinningMethod.DQS, 1)
        assert shader == GPUSkinningDispatcher.SHADER_DQS_1BONE

    def test_workgroup_calculation_small(self, gpu_capabilities_full, gpu_dispatch_config):
        """Test workgroup calculation for small mesh."""
        dispatcher = GPUSkinningDispatcher(gpu_dispatch_config, gpu_capabilities_full)
        dispatcher.initialize()
        wg = dispatcher.calculate_workgroups(100)
        assert wg[0] == 2  # ceil(100/64) = 2
        assert wg[1] == 1
        assert wg[2] == 1

    def test_workgroup_calculation_exact(self, gpu_capabilities_full, gpu_dispatch_config):
        """Test workgroup calculation for exact multiple."""
        dispatcher = GPUSkinningDispatcher(gpu_dispatch_config, gpu_capabilities_full)
        dispatcher.initialize()
        wg = dispatcher.calculate_workgroups(128)
        assert wg[0] == 2  # 128/64 = 2
        assert wg[1] == 1
        assert wg[2] == 1

    def test_workgroup_calculation_large(self, gpu_capabilities_full, gpu_dispatch_config):
        """Test workgroup calculation for large mesh."""
        dispatcher = GPUSkinningDispatcher(gpu_dispatch_config, gpu_capabilities_full)
        dispatcher.initialize()
        wg = dispatcher.calculate_workgroups(10000)
        assert wg[0] == 157  # ceil(10000/64) = 157
        assert wg[1] == 1
        assert wg[2] == 1

    def test_dispatch_skinning_success(
        self,
        gpu_capabilities_full,
        gpu_dispatch_config,
        skinned_mesh_component,
        bone_transforms,
    ):
        """Test successful GPU skinning dispatch."""
        dispatcher = GPUSkinningDispatcher(gpu_dispatch_config, gpu_capabilities_full)
        dispatcher.initialize()
        result = dispatcher.dispatch_skinning(skinned_mesh_component, bone_transforms)
        assert result is True

    def test_dispatch_skinning_no_init(
        self,
        gpu_capabilities_full,
        skinned_mesh_component,
        bone_transforms,
    ):
        """Test dispatch fails without initialization."""
        dispatcher = GPUSkinningDispatcher(capabilities=gpu_capabilities_full)
        # Don't call initialize()
        result = dispatcher.dispatch_skinning(skinned_mesh_component, bone_transforms)
        assert result is False

    def test_dispatch_skinning_no_mesh(
        self,
        gpu_capabilities_full,
        gpu_dispatch_config,
        skinning_data_4_influences,
        bone_transforms,
    ):
        """Test dispatch fails with no mesh data."""
        dispatcher = GPUSkinningDispatcher(gpu_dispatch_config, gpu_capabilities_full)
        dispatcher.initialize()
        component = SkinnedMeshComponent(
            mesh=None,
            skinning_data=skinning_data_4_influences,
        )
        result = dispatcher.dispatch_skinning(component, bone_transforms)
        assert result is False


# =============================================================================
# LBS vs DQS Selection Tests
# =============================================================================


class TestSkinningMethodSelection:
    """Tests for LBS vs DQS selection."""

    def test_lbs_method_explicit(self, skinning_system, skinned_mesh_component, bone_transforms):
        """Test explicit LBS method selection."""
        skinned_mesh_component.method = SkinningMethod.LBS
        entity = MockEntity(1)

        skinning_system.update(
            MockWorld(),
            [(entity, skinned_mesh_component)],
            {entity: bone_transforms},
        )

        assert skinning_system.stats.entities_skinned == 1

    def test_dqs_method_explicit(self, skinning_system, skinned_mesh_component, bone_transforms):
        """Test explicit DQS method selection."""
        skinned_mesh_component.method = SkinningMethod.DQS
        entity = MockEntity(1)

        skinning_system.update(
            MockWorld(),
            [(entity, skinned_mesh_component)],
            {entity: bone_transforms},
        )

        assert skinning_system.stats.entities_skinned == 1

    def test_auto_method_selects_lbs_for_small_rotations(
        self, skinning_system, skinned_mesh_component, bone_transforms
    ):
        """Test AUTO selects LBS for small joint angles."""
        skinned_mesh_component.method = SkinningMethod.AUTO
        entity = MockEntity(1)

        # bone_transforms has identity rotations -> small angles -> LBS
        skinning_system.update(
            MockWorld(),
            [(entity, skinned_mesh_component)],
            {entity: bone_transforms},
        )

        assert skinning_system.stats.entities_skinned == 1

    def test_auto_method_selects_dqs_for_large_rotations(
        self, skinning_system, skinned_mesh_component, rotated_bone_transforms
    ):
        """Test AUTO selects DQS for large joint angles."""
        skinned_mesh_component.method = SkinningMethod.AUTO
        entity = MockEntity(1)

        # rotated_bone_transforms has 90+ degree rotations -> DQS
        skinning_system.update(
            MockWorld(),
            [(entity, skinned_mesh_component)],
            {entity: rotated_bone_transforms},
        )

        assert skinning_system.stats.entities_skinned == 1


# =============================================================================
# LOD Influence Reduction Tests
# =============================================================================


class TestLODInfluenceReduction:
    """Tests for LOD-based influence reduction (4->2->1)."""

    def test_lod_0_uses_4_influences(self, skinned_mesh_component):
        """Test LOD 0 uses 4 bone influences."""
        skinned_mesh_component.lod_level = 0
        assert skinned_mesh_component.get_influence_count_for_lod() == 4

    def test_lod_1_uses_2_influences(self, skinned_mesh_component):
        """Test LOD 1 uses 2 bone influences."""
        skinned_mesh_component.lod_level = 1
        assert skinned_mesh_component.get_influence_count_for_lod() == 2

    def test_lod_2_uses_2_influences(self, skinned_mesh_component):
        """Test LOD 2 uses 2 bone influences."""
        skinned_mesh_component.lod_level = 2
        assert skinned_mesh_component.get_influence_count_for_lod() == 2

    def test_lod_3_uses_1_influence(self, skinned_mesh_component):
        """Test LOD 3 uses 1 bone influence."""
        skinned_mesh_component.lod_level = 3
        assert skinned_mesh_component.get_influence_count_for_lod() == 1

    def test_lod_4_uses_1_influence(self, skinned_mesh_component):
        """Test LOD 4+ uses 1 bone influence."""
        skinned_mesh_component.lod_level = 4
        assert skinned_mesh_component.get_influence_count_for_lod() == 1

    def test_lod_component_updates_skinning(
        self, skinning_system, skinned_mesh_component, bone_transforms
    ):
        """Test LOD component updates skinning influence count."""
        entity = MockEntity(1)
        lod_component = LODComponent(current_lod=2, distance=50.0)

        skinning_system.update(
            MockWorld(),
            [(entity, skinned_mesh_component)],
            {entity: bone_transforms},
            lod_data={entity: lod_component},
        )

        # Component should now have LOD level 2
        assert skinned_mesh_component.lod_level == 2
        assert skinned_mesh_component.get_influence_count_for_lod() == 2

    def test_lod_influence_reduction_cpu_fallback(
        self, skinning_system_cpu_only, simple_mesh, skinning_data_4_influences, bone_transforms
    ):
        """Test LOD influence reduction works with CPU fallback."""
        component = SkinnedMeshComponent(
            mesh=simple_mesh,
            skinning_data=skinning_data_4_influences,
            method=SkinningMethod.LBS,
            lod_level=3,  # Should use 1 influence
            force_cpu=True,
        )
        entity = MockEntity(1)

        skinning_system_cpu_only.update(
            MockWorld(),
            [(entity, component)],
            {entity: bone_transforms},
        )

        # Should have skinned with 1 influence
        assert component.get_influence_count_for_lod() == 1
        assert len(component.skinned_positions) > 0


# =============================================================================
# CPU Fallback Tests
# =============================================================================


class TestCPUFallback:
    """Tests for CPU skinning fallback."""

    def test_cpu_fallback_initialization(self):
        """Test CPU fallback initializes correctly."""
        fallback = CPUSkinningFallback(use_simd=True)
        assert fallback.backend in (SkinningBackend.CPU_SIMD, SkinningBackend.CPU_SCALAR)

    def test_cpu_fallback_lbs_skinning(
        self, simple_mesh, skinning_data_4_influences, bone_transforms
    ):
        """Test CPU LBS skinning produces valid results."""
        fallback = CPUSkinningFallback()
        component = SkinnedMeshComponent(
            mesh=simple_mesh,
            skinning_data=skinning_data_4_influences,
            method=SkinningMethod.LBS,
        )

        # Compute skinning matrices
        component.skinning_matrices = [
            bone_transforms[i].to_matrix() @ skinning_data_4_influences.bind_poses[i]
            for i in range(4)
        ]

        fallback.skin_mesh_lbs(component, 4)

        assert len(component.skinned_positions) == simple_mesh.vertex_count
        assert len(component.skinned_normals) == simple_mesh.vertex_count

    def test_cpu_fallback_dqs_skinning(
        self, simple_mesh, skinning_data_4_influences, bone_transforms
    ):
        """Test CPU DQS skinning produces valid results."""
        fallback = CPUSkinningFallback()
        component = SkinnedMeshComponent(
            mesh=simple_mesh,
            skinning_data=skinning_data_4_influences,
            method=SkinningMethod.DQS,
        )

        # Compute skinning matrices
        component.skinning_matrices = [
            bone_transforms[i].to_matrix() @ skinning_data_4_influences.bind_poses[i]
            for i in range(4)
        ]

        fallback.skin_mesh_dqs(component, 4)

        assert len(component.skinned_positions) == simple_mesh.vertex_count
        assert len(component.skinned_normals) == simple_mesh.vertex_count

    def test_cpu_fallback_lbs_reduced_influences(
        self, simple_mesh, skinning_data_4_influences, bone_transforms
    ):
        """Test CPU LBS with reduced influence count."""
        fallback = CPUSkinningFallback()
        component = SkinnedMeshComponent(
            mesh=simple_mesh,
            skinning_data=skinning_data_4_influences,
            method=SkinningMethod.LBS,
        )

        component.skinning_matrices = [
            bone_transforms[i].to_matrix() @ skinning_data_4_influences.bind_poses[i]
            for i in range(4)
        ]

        # Use only 2 influences
        fallback.skin_mesh_lbs(component, 2)

        assert len(component.skinned_positions) == simple_mesh.vertex_count

    def test_cpu_fallback_triggered_without_gpu(
        self, skinning_system_cpu_only, skinned_mesh_component, bone_transforms
    ):
        """Test CPU fallback triggered when GPU unavailable."""
        entity = MockEntity(1)

        skinning_system_cpu_only.update(
            MockWorld(),
            [(entity, skinned_mesh_component)],
            {entity: bone_transforms},
        )

        # Should have used CPU fallback
        assert skinning_system_cpu_only.stats.cpu_fallback_count > 0
        assert skinning_system_cpu_only.stats.gpu_dispatches == 0

    def test_force_cpu_flag(
        self, skinning_system, skinned_mesh_component, bone_transforms
    ):
        """Test force_cpu flag forces CPU skinning."""
        skinned_mesh_component.force_cpu = True
        entity = MockEntity(1)

        skinning_system.update(
            MockWorld(),
            [(entity, skinned_mesh_component)],
            {entity: bone_transforms},
        )

        assert skinning_system.stats.cpu_fallback_count > 0


# =============================================================================
# Async Compute Tests
# =============================================================================


class TestAsyncCompute:
    """Tests for async compute overlap."""

    def test_async_compute_enabled_when_supported(
        self, gpu_capabilities_full, gpu_dispatch_config
    ):
        """Test async compute enabled with hardware support."""
        dispatcher = GPUSkinningDispatcher(gpu_dispatch_config, gpu_capabilities_full)
        assert dispatcher.supports_async_compute

    def test_async_compute_disabled_without_support(
        self, gpu_capabilities_no_async
    ):
        """Test async compute disabled without hardware support."""
        config = GPUDispatchConfig(use_async_compute=True)
        dispatcher = GPUSkinningDispatcher(config, gpu_capabilities_no_async)
        assert not dispatcher.supports_async_compute

    def test_async_compute_disabled_by_config(
        self, gpu_capabilities_full
    ):
        """Test async compute disabled by config."""
        config = GPUDispatchConfig(use_async_compute=False)
        dispatcher = GPUSkinningDispatcher(config, gpu_capabilities_full)
        assert not dispatcher.supports_async_compute

    def test_async_region_begin_end(
        self, gpu_capabilities_full, gpu_dispatch_config
    ):
        """Test async compute region begin/end."""
        dispatcher = GPUSkinningDispatcher(gpu_dispatch_config, gpu_capabilities_full)
        dispatcher.initialize()

        assert dispatcher.begin_async_region()
        dispatcher.end_async_region()

    def test_async_overlap_counted(
        self, skinning_system, skinned_mesh_component, bone_transforms
    ):
        """Test async overlaps are counted in stats."""
        entity = MockEntity(1)

        skinning_system.update(
            MockWorld(),
            [(entity, skinned_mesh_component)],
            {entity: bone_transforms},
        )

        # With async enabled and supported, should have overlap
        if skinning_system.gpu_available:
            assert skinning_system.stats.async_overlaps > 0


# =============================================================================
# Batch Optimization Tests
# =============================================================================


class TestBatchOptimization:
    """Tests for batch optimization."""

    def test_batch_creation_same_params(self):
        """Test dispatches with same params batch together."""
        dispatch1 = SkinningDispatch(
            entity=MockEntity(1),
            vertex_count=1000,
            method=SkinningMethod.LBS,
            influence_count=4,
        )
        dispatch2 = SkinningDispatch(
            entity=MockEntity(2),
            vertex_count=2000,
            method=SkinningMethod.LBS,
            influence_count=4,
        )

        batch = SkinningBatch()
        assert batch.add_dispatch(dispatch1)
        assert batch.add_dispatch(dispatch2)
        assert len(batch.dispatches) == 2
        assert batch.total_vertices == 3000

    def test_batch_rejects_different_method(self):
        """Test batch rejects different skinning method."""
        dispatch1 = SkinningDispatch(
            entity=MockEntity(1),
            method=SkinningMethod.LBS,
            influence_count=4,
        )
        dispatch2 = SkinningDispatch(
            entity=MockEntity(2),
            method=SkinningMethod.DQS,
            influence_count=4,
        )

        batch = SkinningBatch()
        assert batch.add_dispatch(dispatch1)
        assert not batch.add_dispatch(dispatch2)  # Should reject

    def test_batch_rejects_different_influence_count(self):
        """Test batch rejects different influence count."""
        dispatch1 = SkinningDispatch(
            entity=MockEntity(1),
            method=SkinningMethod.LBS,
            influence_count=4,
        )
        dispatch2 = SkinningDispatch(
            entity=MockEntity(2),
            method=SkinningMethod.LBS,
            influence_count=2,
        )

        batch = SkinningBatch()
        assert batch.add_dispatch(dispatch1)
        assert not batch.add_dispatch(dispatch2)  # Should reject

    def test_multiple_batches_created(
        self, skinning_system, simple_mesh, skinning_data_4_influences, bone_transforms
    ):
        """Test multiple batches created for different params."""
        # Create components with different methods
        component1 = SkinnedMeshComponent(
            mesh=simple_mesh,
            skinning_data=skinning_data_4_influences,
            method=SkinningMethod.LBS,
        )
        component2 = SkinnedMeshComponent(
            mesh=simple_mesh,
            skinning_data=skinning_data_4_influences,
            method=SkinningMethod.DQS,
        )

        entity1 = MockEntity(1)
        entity2 = MockEntity(2)

        skinning_system.update(
            MockWorld(),
            [(entity1, component1), (entity2, component2)],
            {entity1: bone_transforms, entity2: bone_transforms},
        )

        # Should have created 2 batches (one for LBS, one for DQS)
        assert skinning_system.stats.batches_processed >= 2

    def test_batching_disabled(
        self, gpu_capabilities_full, gpu_dispatch_config, simple_mesh,
        skinning_data_4_influences, bone_transforms
    ):
        """Test system works with batching disabled."""
        system = SkinningSystem(
            gpu_config=gpu_dispatch_config,
            gpu_capabilities=gpu_capabilities_full,
            enable_batching=False,
        )

        component = SkinnedMeshComponent(
            mesh=simple_mesh,
            skinning_data=skinning_data_4_influences,
            method=SkinningMethod.LBS,
        )
        entity = MockEntity(1)

        system.update(
            MockWorld(),
            [(entity, component)],
            {entity: bone_transforms},
        )

        assert system.stats.entities_skinned == 1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling with missing bones and invalid weights."""

    def test_missing_bones_handled(
        self, skinning_system_cpu_only, simple_mesh, bone_transforms
    ):
        """Test missing bones in pose data handled gracefully."""
        # Create skinning data referencing bone index 5 (doesn't exist)
        vertex_data = [
            VertexSkinData(influences=[BoneInfluence(5, 1.0)])
            for _ in range(8)
        ]
        skinning_data = SkinningData(
            vertex_data=vertex_data,
            bind_poses=[Mat4.identity() for _ in range(6)],  # 6 bones
            bone_names=["bone" + str(i) for i in range(6)],
        )

        component = SkinnedMeshComponent(
            mesh=simple_mesh,
            skinning_data=skinning_data,
            method=SkinningMethod.LBS,
            force_cpu=True,
        )
        entity = MockEntity(1)

        # Should not crash
        skinning_system_cpu_only.update(
            MockWorld(),
            [(entity, component)],
            {entity: bone_transforms},  # Only has bones 0-3
        )

    def test_invalid_bone_index_handled(self, simple_mesh):
        """Test invalid bone indices handled gracefully."""
        fallback = CPUSkinningFallback()

        vertex_data = [
            VertexSkinData(influences=[BoneInfluence(99, 1.0)])  # Invalid index
            for _ in range(8)
        ]
        skinning_data = SkinningData(
            vertex_data=vertex_data,
            bind_poses=[Mat4.identity() for _ in range(4)],
        )

        component = SkinnedMeshComponent(
            mesh=simple_mesh,
            skinning_data=skinning_data,
        )
        component.skinning_matrices = [Mat4.identity() for _ in range(4)]

        # Should not crash, just produce zero/identity output
        fallback.skin_mesh_lbs(component, 4)

    def test_zero_weight_handled(self, simple_mesh):
        """Test zero weights handled correctly."""
        fallback = CPUSkinningFallback()

        vertex_data = [
            VertexSkinData(influences=[BoneInfluence(0, 0.0)])  # Zero weight
            for _ in range(8)
        ]
        skinning_data = SkinningData(
            vertex_data=vertex_data,
            bind_poses=[Mat4.identity()],
        )

        component = SkinnedMeshComponent(
            mesh=simple_mesh,
            skinning_data=skinning_data,
        )
        component.skinning_matrices = [Mat4.identity()]

        # Should not crash
        fallback.skin_mesh_lbs(component, 4)

    def test_empty_influences_handled(self, simple_mesh):
        """Test empty influences handled correctly."""
        fallback = CPUSkinningFallback()

        vertex_data = [
            VertexSkinData(influences=[])  # No influences
            for _ in range(8)
        ]
        skinning_data = SkinningData(
            vertex_data=vertex_data,
            bind_poses=[Mat4.identity()],
        )

        component = SkinnedMeshComponent(
            mesh=simple_mesh,
            skinning_data=skinning_data,
        )
        component.skinning_matrices = [Mat4.identity()]

        # Should not crash
        fallback.skin_mesh_lbs(component, 4)

    def test_disabled_component_skipped(
        self, skinning_system, skinned_mesh_component, bone_transforms
    ):
        """Test disabled components are skipped."""
        skinned_mesh_component.enabled = False
        entity = MockEntity(1)

        skinning_system.update(
            MockWorld(),
            [(entity, skinned_mesh_component)],
            {entity: bone_transforms},
        )

        assert skinning_system.stats.entities_skinned == 0

    def test_no_skinning_data_skipped(
        self, skinning_system, simple_mesh, bone_transforms
    ):
        """Test components without skinning data are skipped."""
        component = SkinnedMeshComponent(
            mesh=simple_mesh,
            skinning_data=None,
        )
        entity = MockEntity(1)

        skinning_system.update(
            MockWorld(),
            [(entity, component)],
            {entity: bone_transforms},
        )

        assert skinning_system.stats.entities_skinned == 0


# =============================================================================
# Statistics Tests
# =============================================================================


class TestSkinningStats:
    """Tests for skinning statistics."""

    def test_stats_reset(self):
        """Test stats reset clears all values."""
        stats = SkinningStats()
        stats.entities_skinned = 10
        stats.vertices_skinned = 1000
        stats.gpu_dispatches = 5

        stats.reset()

        assert stats.entities_skinned == 0
        assert stats.vertices_skinned == 0
        assert stats.gpu_dispatches == 0

    def test_stats_vertices_counted(
        self, skinning_system, skinned_mesh_component, bone_transforms
    ):
        """Test vertex count tracked correctly."""
        entity = MockEntity(1)

        skinning_system.update(
            MockWorld(),
            [(entity, skinned_mesh_component)],
            {entity: bone_transforms},
        )

        expected_vertices = skinned_mesh_component.mesh.vertex_count
        assert skinning_system.stats.vertices_skinned == expected_vertices

    def test_stats_time_tracked(
        self, skinning_system, skinned_mesh_component, bone_transforms
    ):
        """Test processing time tracked."""
        entity = MockEntity(1)

        skinning_system.update(
            MockWorld(),
            [(entity, skinned_mesh_component)],
            {entity: bone_transforms},
        )

        assert skinning_system.stats.total_time_ms >= 0


# =============================================================================
# Component Tests
# =============================================================================


class TestSkinnedMeshComponent:
    """Tests for SkinnedMeshComponent."""

    def test_component_initialization(self, simple_mesh, skinning_data_4_influences):
        """Test component initializes correctly."""
        component = SkinnedMeshComponent(
            mesh=simple_mesh,
            skinning_data=skinning_data_4_influences,
        )

        assert component.enabled
        assert component.method == SkinningMethod.LBS
        assert component.lod_level == 0
        assert component.cache is not None

    def test_component_memory_size(self, skinned_mesh_component):
        """Test memory size calculation."""
        size = skinned_mesh_component.get_memory_size_bytes()
        assert size > 0

    def test_component_prepare_gpu_buffer(self, skinned_mesh_component, bone_transforms):
        """Test GPU buffer preparation."""
        # Need skinning matrices
        skinned_mesh_component.skinning_matrices = [
            bone_transforms[i].to_matrix()
            for i in range(4)
        ]

        skinned_mesh_component.prepare_gpu_buffer()

        # Should have flattened matrices
        expected_floats = 4 * 16  # 4 bones * 16 floats per matrix
        assert len(skinned_mesh_component.bone_matrices_buffer) == expected_floats

    def test_component_cache_invalidation(self, skinned_mesh_component):
        """Test cache invalidation."""
        skinned_mesh_component.invalidate_cache()
        # Should not crash


# =============================================================================
# Bounding Box Tests
# =============================================================================


class TestBoundingBox:
    """Tests for bounding box computation."""

    def test_bounding_box_from_skinned_positions(
        self, skinning_system, skinned_mesh_component, bone_transforms
    ):
        """Test bounding box computed from skinned positions."""
        entity = MockEntity(1)
        skinned_mesh_component.force_cpu = True

        skinning_system.update(
            MockWorld(),
            [(entity, skinned_mesh_component)],
            {entity: bone_transforms},
        )

        bbox = skinning_system.compute_bounding_box(skinned_mesh_component)
        assert bbox is not None
        min_corner, max_corner = bbox
        assert max_corner.x >= min_corner.x
        assert max_corner.y >= min_corner.y
        assert max_corner.z >= min_corner.z

    def test_bounding_box_no_data(self, skinning_system):
        """Test bounding box returns None with no data."""
        component = SkinnedMeshComponent()
        bbox = skinning_system.compute_bounding_box(component)
        assert bbox is None


# =============================================================================
# Normalization Tests
# =============================================================================


class TestWeightNormalization:
    """Tests for weight normalization."""

    def test_vertex_skin_data_normalize(self):
        """Test vertex skin data weight normalization."""
        influences = [
            BoneInfluence(0, 0.5),
            BoneInfluence(1, 0.5),
            BoneInfluence(2, 0.5),
        ]
        vertex_data = VertexSkinData(influences=influences)
        vertex_data.normalize()

        total = sum(inf.weight for inf in vertex_data.influences)
        assert abs(total - 1.0) < 0.001

    def test_bone_influence_count(self):
        """Test bone influence counting."""
        influences = [
            BoneInfluence(0, 0.5),
            BoneInfluence(1, 0.3),
        ]
        vertex_data = VertexSkinData(influences=influences)
        assert vertex_data.bone_count == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete skinning pipeline."""

    def test_full_pipeline_lbs_cpu(
        self, simple_mesh, skinning_data_4_influences, bone_transforms
    ):
        """Test complete LBS pipeline with CPU."""
        system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False),
        )

        component = SkinnedMeshComponent(
            mesh=simple_mesh,
            skinning_data=skinning_data_4_influences,
            method=SkinningMethod.LBS,
        )
        entity = MockEntity(1)

        system.update(
            MockWorld(),
            [(entity, component)],
            {entity: bone_transforms},
        )

        assert system.stats.entities_skinned == 1
        assert len(component.skinned_positions) == simple_mesh.vertex_count
        assert len(component.skinned_normals) == simple_mesh.vertex_count

    def test_full_pipeline_dqs_cpu(
        self, simple_mesh, skinning_data_4_influences, bone_transforms
    ):
        """Test complete DQS pipeline with CPU."""
        system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False),
        )

        component = SkinnedMeshComponent(
            mesh=simple_mesh,
            skinning_data=skinning_data_4_influences,
            method=SkinningMethod.DQS,
        )
        entity = MockEntity(1)

        system.update(
            MockWorld(),
            [(entity, component)],
            {entity: bone_transforms},
        )

        assert system.stats.entities_skinned == 1
        assert len(component.skinned_positions) == simple_mesh.vertex_count

    def test_multiple_entities(
        self, skinning_system, simple_mesh, skinning_data_4_influences, bone_transforms
    ):
        """Test skinning multiple entities."""
        components = [
            SkinnedMeshComponent(
                mesh=simple_mesh,
                skinning_data=skinning_data_4_influences,
                method=SkinningMethod.LBS,
            )
            for _ in range(5)
        ]
        entities = [MockEntity(i) for i in range(5)]

        entity_components = list(zip(entities, components))
        pose_data = {e: bone_transforms for e in entities}

        skinning_system.update(MockWorld(), entity_components, pose_data)

        assert skinning_system.stats.entities_skinned == 5
        assert skinning_system.stats.vertices_skinned == 5 * simple_mesh.vertex_count

    def test_system_shutdown(self, skinning_system):
        """Test system shutdown releases resources."""
        skinning_system.shutdown()
        # Should not crash


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance-related tests."""

    def test_large_mesh_performance(self, skinning_system_cpu_only, bone_transforms):
        """Test performance with larger mesh."""
        # Create a larger mesh (1000 vertices)
        positions = [Vec3(i * 0.1, 0, 0) for i in range(1000)]
        normals = [Vec3(0, 1, 0) for _ in range(1000)]
        mesh = MeshData(positions=positions, normals=normals)

        vertex_data = [
            VertexSkinData(influences=[
                BoneInfluence(0, 0.5),
                BoneInfluence(1, 0.5),
            ])
            for _ in range(1000)
        ]
        skinning_data = SkinningData(
            vertex_data=vertex_data,
            bind_poses=[Mat4.identity() for _ in range(4)],
        )

        component = SkinnedMeshComponent(
            mesh=mesh,
            skinning_data=skinning_data,
            method=SkinningMethod.LBS,
        )
        entity = MockEntity(1)

        start = time.perf_counter()
        skinning_system_cpu_only.update(
            MockWorld(),
            [(entity, component)],
            {entity: bone_transforms},
        )
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (< 1 second for CPU)
        assert elapsed < 1.0
        assert len(component.skinned_positions) == 1000

    def test_many_entities_performance(
        self, skinning_system_cpu_only, simple_mesh, skinning_data_4_influences, bone_transforms
    ):
        """Test performance with many entities."""
        components = [
            SkinnedMeshComponent(
                mesh=simple_mesh,
                skinning_data=skinning_data_4_influences,
            )
            for _ in range(100)
        ]
        entities = [MockEntity(i) for i in range(100)]

        entity_components = list(zip(entities, components))
        pose_data = {e: bone_transforms for e in entities}

        start = time.perf_counter()
        skinning_system_cpu_only.update(MockWorld(), entity_components, pose_data)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time
        assert elapsed < 5.0
        assert skinning_system_cpu_only.stats.entities_skinned == 100
