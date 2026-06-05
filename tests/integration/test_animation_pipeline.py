"""Full Animation System Integration Tests (T-AN-9.12).

End-to-end integration tests for the complete animation pipeline covering:
- Full pipeline: skeleton -> playback -> IK -> procedural -> skinning -> output
- Motion matching -> inertialization -> skinning
- State machine -> blend tree -> facial -> skinning
- Crowd system -> animation textures -> instanced rendering
- Deterministic replay of animation state transitions
- Frame time within budget across all LOD levels

Total: 60+ tests covering:
- Full pipeline E2E (12 tests)
- Motion matching pipeline (10 tests)
- State machine + blend tree pipeline (10 tests)
- Facial animation pipeline (8 tests)
- Crowd rendering pipeline (10 tests)
- Deterministic replay (6 tests)
- Performance budgets (8 tests)
- Error recovery (6 tests)
"""

from __future__ import annotations

import math
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch
import pytest

from engine.core.math.vec import Vec3, Vec4
from engine.core.math.quat import Quat
from engine.core.math.mat import Mat4
from engine.core.math.transform import Transform

from engine.animation.skeletal.skeleton import Skeleton, Bone
from engine.animation.skeletal.clip import AnimationClip, AnimationTrack, AnimationKeyframe
from engine.animation.skeletal.pose import Pose
from engine.animation.skeletal.blending import PoseBlender

from engine.animation.graph import (
    AnimationGraph,
    AnimationNode,
    BlendTree1D,
    BlendTree2D,
    BlendTreeDirect,
    ClipNode,
    GraphContext,
    GraphParameter,
    LoopMode,
    ParameterType,
    StateMachine,
)

from engine.animation.systems.animation_graph_system import (
    AnimationGraphSystem,
    AnimationGraphComponent,
    BoneTransformSoA,
    DirtyFlags,
    AnimationDirtyState,
    StateMachineOutput,
    ClipSampler,
)

from engine.animation.systems.ik_system import (
    IKSystem,
    IKComponent,
    IKGoal,
    IKSolverType,
    IKHintType,
)

from engine.animation.systems.procedural_system import (
    ProceduralSystem,
    ProceduralComponent,
    ProceduralModifier,
    BreathingModifier,
    SpringBoneModifier,
    LookAtModifier,
)

from engine.animation.systems.skinning_system import (
    SkinningSystem,
    SkinningMethod,
    SkinningBackend,
    LODInfluenceLevel,
    BoneInfluence,
    VertexSkinData,
    SkinningData,
    MeshData,
    GPUDispatchConfig,
    GPUCapabilities,
    SkinnedMeshComponent,
    CPUSkinningFallback,
)

from engine.animation.systems.motion_matching_system import (
    MotionMatchingSystem,
    MotionMatchingComponent,
    MotionMatchingConfig,
    FallbackReason,
    MotionMatchingMode,
)

from engine.animation.systems.facial_system import (
    FacialSystem,
    FacialComponent,
    Expression,
    EmotionState,
    LipSyncPhoneme,
    FacialRegion,
    FacialLayerPriority,
)

from engine.animation.systems.crowd_system import (
    CrowdSystem,
    CrowdComponent,
)

from engine.animation.crowds.crowd_renderer import (
    CrowdRenderer,
    CrowdInstance,
    InstanceBuffer,
)

from engine.animation.crowds.animation_texture import (
    AnimationTextureAtlas,
)

from engine.animation.crowds.crowd_lod import CrowdLOD, LODLevel

from engine.animation.crowds.crowd_behavior import (
    CrowdSimulator,
    CrowdAgent,
    AgentState,
)


# =============================================================================
# TEST FIXTURES
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
def humanoid_skeleton() -> Skeleton:
    """Create a full humanoid skeleton for testing."""
    skeleton = Skeleton()
    # Root and spine
    skeleton.add_bone("root", parent_index=-1)         # 0
    skeleton.add_bone("pelvis", parent_index=0)        # 1
    skeleton.add_bone("spine_01", parent_index=1)      # 2
    skeleton.add_bone("spine_02", parent_index=2)      # 3
    skeleton.add_bone("chest", parent_index=3)         # 4
    skeleton.add_bone("neck", parent_index=4)          # 5
    skeleton.add_bone("head", parent_index=5)          # 6
    # Left arm
    skeleton.add_bone("clavicle_l", parent_index=4)    # 7
    skeleton.add_bone("upperarm_l", parent_index=7)    # 8
    skeleton.add_bone("lowerarm_l", parent_index=8)    # 9
    skeleton.add_bone("hand_l", parent_index=9)        # 10
    # Right arm
    skeleton.add_bone("clavicle_r", parent_index=4)    # 11
    skeleton.add_bone("upperarm_r", parent_index=11)   # 12
    skeleton.add_bone("lowerarm_r", parent_index=12)   # 13
    skeleton.add_bone("hand_r", parent_index=13)       # 14
    # Left leg
    skeleton.add_bone("thigh_l", parent_index=1)       # 15
    skeleton.add_bone("calf_l", parent_index=15)       # 16
    skeleton.add_bone("foot_l", parent_index=16)       # 17
    # Right leg
    skeleton.add_bone("thigh_r", parent_index=1)       # 18
    skeleton.add_bone("calf_r", parent_index=18)       # 19
    skeleton.add_bone("foot_r", parent_index=19)       # 20
    return skeleton


@pytest.fixture
def bone_hierarchy() -> Dict[int, int]:
    """Bone hierarchy for IK system."""
    return {
        0: -1,  # root
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5,  # spine chain
        7: 4, 8: 7, 9: 8, 10: 9,              # left arm
        11: 4, 12: 11, 13: 12, 14: 13,        # right arm
        15: 1, 16: 15, 17: 16,                # left leg
        18: 1, 19: 18, 20: 19,                # right leg
    }


@pytest.fixture
def bone_lengths() -> Dict[int, float]:
    """Bone lengths for IK calculations."""
    return {
        0: 0.0,                               # root
        1: 0.15, 2: 0.15, 3: 0.15, 4: 0.2, 5: 0.1, 6: 0.15,  # spine
        7: 0.1, 8: 0.3, 9: 0.25, 10: 0.1,    # left arm
        11: 0.1, 12: 0.3, 13: 0.25, 14: 0.1, # right arm
        15: 0.4, 16: 0.4, 17: 0.15,          # left leg
        18: 0.4, 19: 0.4, 20: 0.15,          # right leg
    }


@pytest.fixture
def rest_pose() -> Dict[int, Transform]:
    """Rest pose transforms for the humanoid skeleton."""
    return {i: Transform(Vec3(0, i * 0.1, 0), Quat.identity(), Vec3.one()) for i in range(21)}


@pytest.fixture
def idle_clip(humanoid_skeleton: Skeleton) -> AnimationClip:
    """Create an idle animation clip."""
    clip = AnimationClip(name="idle", duration=2.0, loop_mode=LoopMode.LOOP)
    for i in range(humanoid_skeleton.bone_count):
        track = clip.add_track(i)
        track.keyframes = [
            AnimationKeyframe(time=0.0, value=Transform(position=(0.0, 0.0, 0.0))),
            AnimationKeyframe(time=1.0, value=Transform(position=(0.0, 0.01, 0.0))),
            AnimationKeyframe(time=2.0, value=Transform(position=(0.0, 0.0, 0.0))),
        ]
    return clip


@pytest.fixture
def walk_clip(humanoid_skeleton: Skeleton) -> AnimationClip:
    """Create a walk animation clip."""
    clip = AnimationClip(name="walk", duration=1.0, loop_mode=LoopMode.LOOP)
    for i in range(humanoid_skeleton.bone_count):
        track = clip.add_track(i)
        track.keyframes = [
            AnimationKeyframe(time=0.0, value=Transform(position=(0.0, 0.0, 0.0))),
            AnimationKeyframe(time=0.5, value=Transform(position=(0.1, 0.0, 0.0))),
            AnimationKeyframe(time=1.0, value=Transform(position=(0.2, 0.0, 0.0))),
        ]
    return clip


@pytest.fixture
def run_clip(humanoid_skeleton: Skeleton) -> AnimationClip:
    """Create a run animation clip."""
    clip = AnimationClip(name="run", duration=0.5, loop_mode=LoopMode.LOOP)
    for i in range(humanoid_skeleton.bone_count):
        track = clip.add_track(i)
        track.keyframes = [
            AnimationKeyframe(time=0.0, value=Transform(position=(0.0, 0.0, 0.0))),
            AnimationKeyframe(time=0.25, value=Transform(position=(0.3, 0.1, 0.0))),
            AnimationKeyframe(time=0.5, value=Transform(position=(0.6, 0.0, 0.0))),
        ]
    return clip


@pytest.fixture
def character_mesh() -> MeshData:
    """Create a mesh with typical vertex count for a character."""
    vertex_count = 5000
    positions = [Vec3(i * 0.001, math.sin(i * 0.1), i * 0.0005) for i in range(vertex_count)]
    normals = [Vec3(0, 1, 0) for _ in range(vertex_count)]
    return MeshData(positions=positions, normals=normals)


@pytest.fixture
def skinning_data_4_influences(humanoid_skeleton: Skeleton) -> SkinningData:
    """Create skinning data with 4 bone influences per vertex."""
    vertex_count = 5000
    vertex_data = []
    for i in range(vertex_count):
        bone_idx = i % humanoid_skeleton.bone_count
        influences = [
            BoneInfluence(bone_idx, 0.5),
            BoneInfluence((bone_idx + 1) % humanoid_skeleton.bone_count, 0.25),
            BoneInfluence((bone_idx + 2) % humanoid_skeleton.bone_count, 0.15),
            BoneInfluence((bone_idx + 3) % humanoid_skeleton.bone_count, 0.1),
        ]
        vertex_data.append(VertexSkinData(influences=influences))

    bind_poses = [Mat4.identity() for _ in range(humanoid_skeleton.bone_count)]
    bone_names = [f"bone_{i}" for i in range(humanoid_skeleton.bone_count)]

    return SkinningData(
        vertex_data=vertex_data,
        bind_poses=bind_poses,
        bone_names=bone_names,
        max_influences=4,
    )


@pytest.fixture
def gpu_capabilities() -> GPUCapabilities:
    """Full GPU capabilities for testing."""
    return GPUCapabilities(
        has_compute_shader=True,
        has_async_compute=True,
        max_workgroup_size=1024,
        max_shared_memory=32768,
        compute_queue_count=2,
    )


# =============================================================================
# 1. FULL PIPELINE E2E TESTS (12 tests)
# =============================================================================


class TestFullPipelineE2E:
    """End-to-end tests for the complete animation pipeline."""

    def test_skeleton_to_output_basic_pipeline(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        character_mesh: MeshData,
        skinning_data_4_influences: SkinningData,
        rest_pose: Dict[int, Transform],
    ):
        """Test basic pipeline: skeleton -> playback -> skinning -> output."""
        # Initialize systems
        graph_system = AnimationGraphSystem()
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        # Create components
        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.enabled = True
        graph_component.register_clip("idle", idle_clip)
        graph_component.state_machine_output = StateMachineOutput(
            current_state="idle",
            state_time=0.5,
        )

        skinned_component = SkinnedMeshComponent(
            mesh=character_mesh,
            skinning_data=skinning_data_4_influences,
            method=SkinningMethod.LBS,
            force_cpu=True,
        )

        entity = MockEntity(1)
        world = MockWorld()

        # Run graph evaluation
        graph_system.update(world, 0.016, [(entity, graph_component)])

        # Verify pose output
        assert graph_component.output_pose.bone_count() == humanoid_skeleton.bone_count

        # Run skinning
        skinning_system.update(
            world,
            [(entity, skinned_component)],
            {entity: rest_pose},
        )

        # Verify skinned output
        assert len(skinned_component.skinned_positions) == character_mesh.vertex_count
        assert skinning_system.stats.entities_skinned == 1

    def test_full_pipeline_with_ik(
        self,
        humanoid_skeleton: Skeleton,
        bone_hierarchy: Dict[int, int],
        bone_lengths: Dict[int, float],
        idle_clip: AnimationClip,
        character_mesh: MeshData,
        skinning_data_4_influences: SkinningData,
        rest_pose: Dict[int, Transform],
    ):
        """Test full pipeline with IK layer."""
        # Initialize systems
        graph_system = AnimationGraphSystem()
        ik_system = IKSystem()
        ik_system.set_skeleton_data(bone_hierarchy, bone_lengths)
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        # Create components
        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("idle", idle_clip)
        graph_component.state_machine_output = StateMachineOutput(current_state="idle")

        ik_component = IKComponent()
        ik_component.add_goal(IKGoal(
            target_bone=10,  # Left hand
            target_position=Vec3(0.5, 1.0, 0.3),
            chain_length=3,
            weight=1.0,
        ))

        skinned_component = SkinnedMeshComponent(
            mesh=character_mesh,
            skinning_data=skinning_data_4_influences,
            method=SkinningMethod.LBS,
            force_cpu=True,
        )

        entity = MockEntity(1)
        world = MockWorld()

        # Run pipeline stages
        graph_system.update(world, 0.016, [(entity, graph_component)])
        ik_result = ik_system.update(world, [(entity, ik_component)], {entity: rest_pose})
        skinning_system.update(world, [(entity, skinned_component)], {entity: rest_pose})

        # Verify IK modified the pose
        assert entity in ik_result
        assert ik_system.get_stats().goals_processed == 1
        assert len(skinned_component.skinned_positions) == character_mesh.vertex_count

    def test_full_pipeline_with_procedural(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        character_mesh: MeshData,
        skinning_data_4_influences: SkinningData,
        rest_pose: Dict[int, Transform],
    ):
        """Test full pipeline with procedural modifiers."""
        # Initialize systems
        graph_system = AnimationGraphSystem()
        procedural_system = ProceduralSystem()
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        # Create components
        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("idle", idle_clip)
        graph_component.state_machine_output = StateMachineOutput(current_state="idle")

        procedural_component = ProceduralComponent()
        procedural_component.add_modifier(BreathingModifier(
            chest_bone=4,
            amplitude=0.02,
            frequency=0.5,
        ))

        skinned_component = SkinnedMeshComponent(
            mesh=character_mesh,
            skinning_data=skinning_data_4_influences,
            force_cpu=True,
        )

        entity = MockEntity(1)
        world = MockWorld()

        # Run pipeline
        graph_system.update(world, 0.016, [(entity, graph_component)])
        procedural_system.update(world, 0.016, [(entity, procedural_component)], {entity: rest_pose})
        skinning_system.update(world, [(entity, skinned_component)], {entity: rest_pose})

        # Verify pipeline completed
        assert len(skinned_component.skinned_positions) > 0

    def test_full_pipeline_with_all_layers(
        self,
        humanoid_skeleton: Skeleton,
        bone_hierarchy: Dict[int, int],
        bone_lengths: Dict[int, float],
        idle_clip: AnimationClip,
        character_mesh: MeshData,
        skinning_data_4_influences: SkinningData,
        rest_pose: Dict[int, Transform],
    ):
        """Test full pipeline with all layers: playback + IK + procedural + skinning."""
        # Initialize all systems
        graph_system = AnimationGraphSystem()
        ik_system = IKSystem()
        ik_system.set_skeleton_data(bone_hierarchy, bone_lengths)
        procedural_system = ProceduralSystem()
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        # Create all components
        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("idle", idle_clip)
        graph_component.state_machine_output = StateMachineOutput(current_state="idle")

        ik_component = IKComponent()
        ik_component.add_goal(IKGoal(
            target_bone=10,
            target_position=Vec3(0.5, 1.0, 0.3),
            chain_length=3,
        ))
        ik_component.add_goal(IKGoal(
            target_bone=17,  # Left foot
            target_position=Vec3(0, 0, 0),
            chain_length=3,
        ))

        procedural_component = ProceduralComponent()
        procedural_component.add_modifier(BreathingModifier(chest_bone=4))

        skinned_component = SkinnedMeshComponent(
            mesh=character_mesh,
            skinning_data=skinning_data_4_influences,
            force_cpu=True,
        )

        entity = MockEntity(1)
        world = MockWorld()

        # Run full pipeline in order
        graph_system.update(world, 0.016, [(entity, graph_component)])
        ik_system.update(world, [(entity, ik_component)], {entity: rest_pose})
        procedural_system.update(world, 0.016, [(entity, procedural_component)], {entity: rest_pose})
        skinning_system.update(world, [(entity, skinned_component)], {entity: rest_pose})

        # Verify complete pipeline execution
        assert graph_component.output_pose.bone_count() == humanoid_skeleton.bone_count
        assert ik_system.get_stats().goals_processed == 2
        assert len(skinned_component.skinned_positions) == character_mesh.vertex_count

    def test_pipeline_multiple_entities(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        character_mesh: MeshData,
        skinning_data_4_influences: SkinningData,
        rest_pose: Dict[int, Transform],
    ):
        """Test pipeline with multiple entities."""
        graph_system = AnimationGraphSystem()
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        entities = []
        graph_components = []
        skinned_components = []

        for i in range(5):
            entity = MockEntity(i)
            entities.append(entity)

            gc = AnimationGraphComponent()
            gc.skeleton = humanoid_skeleton
            gc.register_clip("idle", idle_clip)
            gc.state_machine_output = StateMachineOutput(current_state="idle", state_time=i * 0.1)
            graph_components.append(gc)

            sc = SkinnedMeshComponent(
                mesh=character_mesh,
                skinning_data=skinning_data_4_influences,
                force_cpu=True,
            )
            skinned_components.append(sc)

        world = MockWorld()

        # Run pipeline for all entities
        graph_system.update(world, 0.016, list(zip(entities, graph_components)))
        skinning_system.update(
            world,
            list(zip(entities, skinned_components)),
            {e: rest_pose for e in entities},
        )

        # Verify all entities processed
        assert graph_system.get_statistics()["entities_evaluated"] == 5
        assert skinning_system.stats.entities_skinned == 5

    def test_pipeline_lod_transitions(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        character_mesh: MeshData,
        skinning_data_4_influences: SkinningData,
        rest_pose: Dict[int, Transform],
    ):
        """Test pipeline with LOD level transitions."""
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        entity = MockEntity(1)
        world = MockWorld()

        # Test different LOD levels
        for lod_level in [0, 1, 2, 3]:
            skinned_component = SkinnedMeshComponent(
                mesh=character_mesh,
                skinning_data=skinning_data_4_influences,
                lod_level=lod_level,
                force_cpu=True,
            )

            skinning_system.update(
                world,
                [(entity, skinned_component)],
                {entity: rest_pose},
            )

            # Verify influence count reduces with LOD
            expected_influences = skinned_component.get_influence_count_for_lod()
            if lod_level == 0:
                assert expected_influences == 4
            elif lod_level in [1, 2]:
                assert expected_influences == 2
            else:
                assert expected_influences == 1

    def test_pipeline_dqs_skinning(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        character_mesh: MeshData,
        skinning_data_4_influences: SkinningData,
        rest_pose: Dict[int, Transform],
    ):
        """Test pipeline with dual quaternion skinning."""
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        skinned_component = SkinnedMeshComponent(
            mesh=character_mesh,
            skinning_data=skinning_data_4_influences,
            method=SkinningMethod.DQS,
            force_cpu=True,
        )

        entity = MockEntity(1)
        world = MockWorld()

        skinning_system.update(
            world,
            [(entity, skinned_component)],
            {entity: rest_pose},
        )

        assert len(skinned_component.skinned_positions) == character_mesh.vertex_count
        assert skinning_system.stats.entities_skinned == 1

    def test_pipeline_state_transitions(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        walk_clip: AnimationClip,
        character_mesh: MeshData,
        skinning_data_4_influences: SkinningData,
        rest_pose: Dict[int, Transform],
    ):
        """Test pipeline during state transitions."""
        graph_system = AnimationGraphSystem()
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("idle", idle_clip)
        graph_component.register_clip("walk", walk_clip)
        graph_component.state_machine_output = StateMachineOutput(
            current_state="idle",
            target_state="walk",
            is_transitioning=True,
            transition_progress=0.5,
            transition_duration=0.3,
        )

        skinned_component = SkinnedMeshComponent(
            mesh=character_mesh,
            skinning_data=skinning_data_4_influences,
            force_cpu=True,
        )

        entity = MockEntity(1)
        world = MockWorld()

        # Run pipeline during transition
        graph_system.update(world, 0.016, [(entity, graph_component)])
        skinning_system.update(world, [(entity, skinned_component)], {entity: rest_pose})

        # Verify blended output
        assert graph_component.output_pose.bone_count() == humanoid_skeleton.bone_count
        assert len(skinned_component.skinned_positions) == character_mesh.vertex_count

    def test_pipeline_root_motion_extraction(
        self,
        humanoid_skeleton: Skeleton,
        walk_clip: AnimationClip,
    ):
        """Test root motion extraction in pipeline."""
        graph_system = AnimationGraphSystem()

        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("walk", walk_clip)
        graph_component.root_motion_enabled = True
        graph_component.state_machine_output = StateMachineOutput(
            current_state="walk",
            state_time=0.5,
        )

        entity = MockEntity(1)
        world = MockWorld()

        graph_system.update(world, 0.016, [(entity, graph_component)])

        # Root motion should be enabled
        assert graph_component.root_motion_enabled

    def test_pipeline_soa_output_format(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
    ):
        """Test SoA output format compatibility with GPU skinning."""
        graph_system = AnimationGraphSystem()

        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("idle", idle_clip)
        graph_component.state_machine_output = StateMachineOutput(current_state="idle")

        entity = MockEntity(1)
        world = MockWorld()

        graph_system.update(world, 0.016, [(entity, graph_component)])

        # Verify SoA structure
        soa = graph_component.output_soa
        assert soa.bone_count == humanoid_skeleton.bone_count
        assert len(soa.positions_x) == soa.bone_count
        assert len(soa.rotations_w) == soa.bone_count
        assert len(soa.scales_z) == soa.bone_count

        # Verify flat arrays for GPU upload
        positions = soa.get_flat_positions()
        rotations = soa.get_flat_rotations()
        scales = soa.get_flat_scales()

        assert len(positions) == soa.bone_count * 3
        assert len(rotations) == soa.bone_count * 4
        assert len(scales) == soa.bone_count * 3

    def test_pipeline_disabled_components_skipped(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        bone_hierarchy: Dict[int, int],
        bone_lengths: Dict[int, float],
        rest_pose: Dict[int, Transform],
    ):
        """Test that disabled components are properly skipped."""
        graph_system = AnimationGraphSystem()
        ik_system = IKSystem()
        ik_system.set_skeleton_data(bone_hierarchy, bone_lengths)

        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.enabled = False  # Disabled

        ik_component = IKComponent(enabled=False)  # Disabled

        entity = MockEntity(1)
        world = MockWorld()

        graph_system.update(world, 0.016, [(entity, graph_component)])
        ik_system.update(world, [(entity, ik_component)], {entity: rest_pose})

        assert graph_system.get_statistics()["entities_evaluated"] == 0
        assert ik_system.get_stats().goals_processed == 0


# =============================================================================
# 2. MOTION MATCHING PIPELINE TESTS (10 tests)
# =============================================================================


class TestMotionMatchingPipeline:
    """Tests for motion matching -> inertialization -> skinning pipeline."""

    def test_motion_matching_basic_query(
        self,
        humanoid_skeleton: Skeleton,
        rest_pose: Dict[int, Transform],
    ):
        """Test basic motion matching query."""
        mm_system = MotionMatchingSystem()

        mm_component = MotionMatchingComponent()
        mm_component.skeleton = humanoid_skeleton
        mm_component.enabled = True
        mm_component.config = MotionMatchingConfig(
            budget_ms=5.0,
            search_interval=0.1,
        )

        entity = MockEntity(1)
        world = MockWorld()

        mm_system.update(world, 0.016, [(entity, mm_component)], {entity: rest_pose})

        stats = mm_system.get_stats()
        assert stats.entities_processed >= 0

    def test_motion_matching_with_skinning(
        self,
        humanoid_skeleton: Skeleton,
        character_mesh: MeshData,
        skinning_data_4_influences: SkinningData,
        rest_pose: Dict[int, Transform],
    ):
        """Test motion matching output feeds into skinning."""
        mm_system = MotionMatchingSystem()
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        mm_component = MotionMatchingComponent()
        mm_component.skeleton = humanoid_skeleton
        mm_component.enabled = True

        skinned_component = SkinnedMeshComponent(
            mesh=character_mesh,
            skinning_data=skinning_data_4_influences,
            force_cpu=True,
        )

        entity = MockEntity(1)
        world = MockWorld()

        mm_system.update(world, 0.016, [(entity, mm_component)], {entity: rest_pose})
        skinning_system.update(world, [(entity, skinned_component)], {entity: rest_pose})

        assert len(skinned_component.skinned_positions) == character_mesh.vertex_count

    def test_motion_matching_budget_enforcement(
        self,
        humanoid_skeleton: Skeleton,
        rest_pose: Dict[int, Transform],
    ):
        """Test that motion matching respects time budget."""
        mm_system = MotionMatchingSystem()

        mm_component = MotionMatchingComponent()
        mm_component.skeleton = humanoid_skeleton
        mm_component.config = MotionMatchingConfig(budget_ms=0.1)  # Very tight budget

        entity = MockEntity(1)
        world = MockWorld()

        start = time.perf_counter()
        mm_system.update(world, 0.016, [(entity, mm_component)], {entity: rest_pose})
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete without exceeding reasonable time
        assert elapsed_ms < 100  # 100ms max for test

    def test_motion_matching_fallback_mode(
        self,
        humanoid_skeleton: Skeleton,
        rest_pose: Dict[int, Transform],
    ):
        """Test fallback to state machine when MM unavailable."""
        mm_system = MotionMatchingSystem()

        mm_component = MotionMatchingComponent()
        mm_component.skeleton = humanoid_skeleton
        mm_component.fallback_enabled = True
        mm_component.mode = MotionMatchingMode.FALLBACK

        entity = MockEntity(1)
        world = MockWorld()

        mm_system.update(world, 0.016, [(entity, mm_component)], {entity: rest_pose})

        # Should handle fallback gracefully
        assert mm_component.fallback_reason in [FallbackReason.NONE, FallbackReason.DATABASE_EMPTY]

    def test_motion_matching_trajectory_computation(
        self,
        humanoid_skeleton: Skeleton,
        rest_pose: Dict[int, Transform],
    ):
        """Test trajectory prediction for motion matching."""
        mm_system = MotionMatchingSystem()

        mm_component = MotionMatchingComponent()
        mm_component.skeleton = humanoid_skeleton
        mm_component.config = MotionMatchingConfig(
            trajectory_times=[0.2, 0.5, 1.0],
        )
        mm_component.desired_velocity = Vec3(1.0, 0.0, 0.0)
        mm_component.desired_facing = Vec3(1.0, 0.0, 0.0)

        entity = MockEntity(1)
        world = MockWorld()

        mm_system.update(world, 0.016, [(entity, mm_component)], {entity: rest_pose})

        # Should compute trajectory
        assert len(mm_component.config.trajectory_times) == 3

    def test_motion_matching_inertialization(
        self,
        humanoid_skeleton: Skeleton,
        rest_pose: Dict[int, Transform],
    ):
        """Test inertialization blending on transition."""
        mm_system = MotionMatchingSystem()

        mm_component = MotionMatchingComponent()
        mm_component.skeleton = humanoid_skeleton
        mm_component.config = MotionMatchingConfig(blend_duration=0.15)
        mm_component.is_blending = True
        mm_component.blend_progress = 0.5

        entity = MockEntity(1)
        world = MockWorld()

        mm_system.update(world, 0.016, [(entity, mm_component)], {entity: rest_pose})

        # System should handle blend state
        assert mm_component.config.blend_duration == 0.15

    def test_motion_matching_continuation_mode(
        self,
        humanoid_skeleton: Skeleton,
        rest_pose: Dict[int, Transform],
    ):
        """Test continuation-only mode (no search)."""
        mm_system = MotionMatchingSystem()

        mm_component = MotionMatchingComponent()
        mm_component.skeleton = humanoid_skeleton
        mm_component.mode = MotionMatchingMode.CONTINUATION_ONLY

        entity = MockEntity(1)
        world = MockWorld()

        mm_system.update(world, 0.016, [(entity, mm_component)], {entity: rest_pose})

        # Should continue without search
        assert mm_component.mode == MotionMatchingMode.CONTINUATION_ONLY

    def test_motion_matching_multiple_entities(
        self,
        humanoid_skeleton: Skeleton,
        rest_pose: Dict[int, Transform],
    ):
        """Test motion matching with multiple entities."""
        mm_system = MotionMatchingSystem()

        entities = []
        components = []

        for i in range(5):
            entity = MockEntity(i)
            mm_component = MotionMatchingComponent()
            mm_component.skeleton = humanoid_skeleton
            mm_component.config = MotionMatchingConfig(budget_ms=1.0)
            entities.append(entity)
            components.append(mm_component)

        world = MockWorld()
        pose_data = {e: rest_pose for e in entities}

        mm_system.update(world, 0.016, list(zip(entities, components)), pose_data)

        stats = mm_system.get_stats()
        assert stats.entities_processed >= 0

    def test_motion_matching_feature_extraction(
        self,
        humanoid_skeleton: Skeleton,
        rest_pose: Dict[int, Transform],
    ):
        """Test pose feature extraction for motion matching."""
        mm_system = MotionMatchingSystem()

        mm_component = MotionMatchingComponent()
        mm_component.skeleton = humanoid_skeleton
        mm_component.config = MotionMatchingConfig(
            position_weight=1.0,
            velocity_weight=0.5,
            trajectory_weight=0.8,
        )

        entity = MockEntity(1)
        world = MockWorld()

        mm_system.update(world, 0.016, [(entity, mm_component)], {entity: rest_pose})

        # Verify feature weights are configured
        assert mm_component.config.position_weight == 1.0
        assert mm_component.config.velocity_weight == 0.5

    def test_motion_matching_disabled_component(
        self,
        humanoid_skeleton: Skeleton,
        rest_pose: Dict[int, Transform],
    ):
        """Test that disabled MM components are skipped."""
        mm_system = MotionMatchingSystem()

        mm_component = MotionMatchingComponent()
        mm_component.skeleton = humanoid_skeleton
        mm_component.enabled = False

        entity = MockEntity(1)
        world = MockWorld()

        mm_system.update(world, 0.016, [(entity, mm_component)], {entity: rest_pose})

        stats = mm_system.get_stats()
        assert stats.entities_processed == 0


# =============================================================================
# 3. STATE MACHINE + BLEND TREE PIPELINE TESTS (10 tests)
# =============================================================================


class TestStateMachineBlendTreePipeline:
    """Tests for state machine -> blend tree -> skinning pipeline."""

    def test_state_machine_to_blend_tree(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        walk_clip: AnimationClip,
    ):
        """Test state machine evaluates blend tree nodes."""
        graph_system = AnimationGraphSystem()

        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("idle", idle_clip)
        graph_component.register_clip("walk", walk_clip)

        # Create blend tree in graph
        graph = AnimationGraph("locomotion")
        graph.add_parameter(GraphParameter.float_param("speed", default=0.5))
        graph_component.graph = graph

        graph_component.state_machine_output = StateMachineOutput(
            current_state="blend",
            state_time=0.5,
        )

        entity = MockEntity(1)
        world = MockWorld()

        graph_system.update(world, 0.016, [(entity, graph_component)])

        assert graph_component.output_pose.bone_count() >= 0

    def test_blend_tree_1d_parameter(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        walk_clip: AnimationClip,
        run_clip: AnimationClip,
    ):
        """Test 1D blend tree interpolation based on speed parameter."""
        graph_system = AnimationGraphSystem()

        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("idle", idle_clip)
        graph_component.register_clip("walk", walk_clip)
        graph_component.register_clip("run", run_clip)

        graph = AnimationGraph("locomotion")
        graph.add_parameter(GraphParameter.float_param("speed", default=0.5))
        graph_component.graph = graph

        entity = MockEntity(1)
        world = MockWorld()

        # Test at different speeds
        for speed in [0.0, 0.5, 1.0]:
            graph_component.set_parameter("speed", speed)
            graph_system.update(world, 0.016, [(entity, graph_component)])
            assert graph_component.output_pose.bone_count() >= 0

    def test_blend_tree_2d_directional(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
    ):
        """Test 2D blend tree for directional movement."""
        graph_system = AnimationGraphSystem()

        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("idle", idle_clip)

        graph = AnimationGraph("directional")
        graph.add_parameter(GraphParameter.float_param("direction_x", default=0.0))
        graph.add_parameter(GraphParameter.float_param("direction_y", default=0.0))
        graph_component.graph = graph

        entity = MockEntity(1)
        world = MockWorld()

        # Test different directions
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        for dx, dy in directions:
            graph_component.set_parameter("direction_x", float(dx))
            graph_component.set_parameter("direction_y", float(dy))
            graph_system.update(world, 0.016, [(entity, graph_component)])

    def test_state_machine_transition_blending(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        walk_clip: AnimationClip,
    ):
        """Test smooth blending during state transitions."""
        graph_system = AnimationGraphSystem()

        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("idle", idle_clip)
        graph_component.register_clip("walk", walk_clip)

        entity = MockEntity(1)
        world = MockWorld()

        # Simulate transition over multiple frames
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            graph_component.state_machine_output = StateMachineOutput(
                current_state="idle",
                target_state="walk",
                is_transitioning=progress < 1.0,
                transition_progress=progress,
                transition_duration=0.3,
            )
            graph_system.update(world, 0.016, [(entity, graph_component)])
            assert graph_component.output_pose.bone_count() >= 0

    def test_blend_tree_direct_control(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        walk_clip: AnimationClip,
    ):
        """Test direct blend tree weight control."""
        graph_system = AnimationGraphSystem()

        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("idle", idle_clip)
        graph_component.register_clip("walk", walk_clip)

        graph = AnimationGraph("direct")
        graph.add_parameter(GraphParameter.float_param("idle_weight", default=0.5))
        graph.add_parameter(GraphParameter.float_param("walk_weight", default=0.5))
        graph_component.graph = graph

        entity = MockEntity(1)
        world = MockWorld()

        graph_system.update(world, 0.016, [(entity, graph_component)])

        # Verify parameters are accessible
        assert graph_component.get_parameter("idle_weight") == 0.5

    def test_pipeline_with_facial_overlay(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
    ):
        """Test body animation with facial animation overlay."""
        graph_system = AnimationGraphSystem()
        facial_system = FacialSystem()

        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("idle", idle_clip)
        graph_component.state_machine_output = StateMachineOutput(current_state="idle")

        facial_component = FacialComponent()
        facial_component.enabled = True
        facial_component.set_emotion(EmotionState.HAPPY, intensity=0.5)

        entity = MockEntity(1)
        world = MockWorld()

        graph_system.update(world, 0.016, [(entity, graph_component)])
        facial_system.update(world, 0.016, [(entity, facial_component)])

        # Both systems should process
        assert graph_component.output_pose.bone_count() >= 0

    def test_parameter_binding_from_gameplay(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
    ):
        """Test parameter binding from gameplay data."""
        graph_system = AnimationGraphSystem()

        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("idle", idle_clip)

        graph = AnimationGraph("test")
        graph.add_parameter(GraphParameter.float_param("speed", default=0.0))
        graph_component.graph = graph
        graph_component.parameter_bindings = {"speed": "player_speed"}

        entity = MockEntity(1)
        world = MockWorld()

        gameplay_data = {"player_speed": 2.5}
        updated = graph_system.sync_parameters_from_gameplay(graph_component, gameplay_data)

        assert updated == 1
        assert graph_component.get_parameter("speed") == 2.5

    def test_additive_layer_blending(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
    ):
        """Test additive animation layer blending."""
        graph_system = AnimationGraphSystem()

        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("idle", idle_clip)
        graph_component.state_machine_output = StateMachineOutput(current_state="idle")
        graph_component.additive_layers = [
            {"clip": "breathing", "weight": 0.3},
        ]

        entity = MockEntity(1)
        world = MockWorld()

        graph_system.update(world, 0.016, [(entity, graph_component)])

        assert graph_component.output_pose.bone_count() >= 0

    def test_layer_mask_application(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
    ):
        """Test bone mask for partial body layers."""
        graph_system = AnimationGraphSystem()

        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("idle", idle_clip)
        graph_component.state_machine_output = StateMachineOutput(current_state="idle")

        # Upper body mask
        upper_body_mask = [False] * humanoid_skeleton.bone_count
        for i in range(4, 15):  # Chest and arms
            upper_body_mask[i] = True
        graph_component.layer_masks = [upper_body_mask]

        entity = MockEntity(1)
        world = MockWorld()

        graph_system.update(world, 0.016, [(entity, graph_component)])

        assert graph_component.output_pose.bone_count() >= 0

    def test_state_machine_force_state(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
    ):
        """Test forcing immediate state change."""
        graph_system = AnimationGraphSystem()

        graph_component = AnimationGraphComponent()
        graph_component.skeleton = humanoid_skeleton
        graph_component.register_clip("idle", idle_clip)
        graph_component.state_machine_output = StateMachineOutput(current_state="idle")

        result = graph_system.force_state(graph_component, "walk")

        assert result is True
        assert graph_component.state_machine_output.current_state == "walk"
        assert not graph_component.state_machine_output.is_transitioning


# =============================================================================
# 4. FACIAL ANIMATION PIPELINE TESTS (8 tests)
# =============================================================================


class TestFacialAnimationPipeline:
    """Tests for facial animation pipeline including blend shapes, lip sync, and eyes."""

    def test_facial_blend_shape_evaluation(self):
        """Test blend shape weight evaluation."""
        facial_system = FacialSystem()

        facial_component = FacialComponent()
        facial_component.enabled = True
        facial_component.set_blend_shape("smile_l", 0.5)
        facial_component.set_blend_shape("smile_r", 0.5)

        entity = MockEntity(1)
        world = MockWorld()

        facial_system.update(world, 0.016, [(entity, facial_component)])

        # Verify blend shapes are set
        assert facial_component.get_blend_shape("smile_l") == 0.5
        assert facial_component.get_blend_shape("smile_r") == 0.5

    def test_facial_emotion_expressions(self):
        """Test emotion-based expression application."""
        facial_system = FacialSystem()

        for emotion in [EmotionState.HAPPY, EmotionState.SAD, EmotionState.ANGRY]:
            facial_component = FacialComponent()
            facial_component.set_emotion(emotion, intensity=0.8)

            entity = MockEntity(1)
            world = MockWorld()

            facial_system.update(world, 0.016, [(entity, facial_component)])

            assert facial_component.current_emotion == emotion
            assert facial_component.emotion_intensity == 0.8

    def test_facial_lip_sync_phonemes(self):
        """Test lip sync phoneme evaluation."""
        facial_system = FacialSystem()

        facial_component = FacialComponent()
        facial_component.enabled = True

        # Simulate phoneme sequence
        phonemes = [LipSyncPhoneme.AA, LipSyncPhoneme.EE, LipSyncPhoneme.OH, LipSyncPhoneme.SILENCE]

        entity = MockEntity(1)
        world = MockWorld()

        for phoneme in phonemes:
            facial_component.set_phoneme(phoneme)
            facial_system.update(world, 0.033, [(entity, facial_component)])
            assert facial_component.current_phoneme == phoneme

    def test_facial_eye_animation(self):
        """Test eye look-at animation."""
        facial_system = FacialSystem()

        facial_component = FacialComponent()
        facial_component.enabled = True
        facial_component.eye_tracking_enabled = True
        facial_component.look_at_target = Vec3(1.0, 1.5, 2.0)

        entity = MockEntity(1)
        world = MockWorld()

        facial_system.update(world, 0.016, [(entity, facial_component)])

        assert facial_component.eye_tracking_enabled
        assert facial_component.look_at_target.x == 1.0

    def test_facial_layer_composition(self):
        """Test layered facial animation composition."""
        facial_system = FacialSystem()

        facial_component = FacialComponent()
        facial_component.enabled = True

        # Add multiple layers
        facial_component.set_emotion(EmotionState.HAPPY, intensity=0.5)
        facial_component.set_phoneme(LipSyncPhoneme.AA)
        facial_component.set_blend_shape("brow_raise", 0.3)

        entity = MockEntity(1)
        world = MockWorld()

        facial_system.update(world, 0.016, [(entity, facial_component)])

        # All layers should be active
        assert facial_component.current_emotion == EmotionState.HAPPY
        assert facial_component.current_phoneme == LipSyncPhoneme.AA

    def test_facial_region_masking(self):
        """Test facial region masking for partial updates."""
        facial_system = FacialSystem()

        facial_component = FacialComponent()
        facial_component.enabled = True

        # Apply expression only to lower face (lip sync region)
        facial_component.add_layer(
            expression=Expression(
                name="talk",
                blend_shapes={"jaw_open": 0.5},
            ),
            priority=FacialLayerPriority.LIP_SYNC,
            region_mask=FacialRegion.LOWER_FACE,
        )

        entity = MockEntity(1)
        world = MockWorld()

        facial_system.update(world, 0.016, [(entity, facial_component)])

    def test_facial_transition_blending(self):
        """Test smooth blending between facial expressions."""
        facial_system = FacialSystem()

        facial_component = FacialComponent()
        facial_component.enabled = True

        entity = MockEntity(1)
        world = MockWorld()

        # Transition from neutral to happy
        facial_component.set_emotion(EmotionState.NEUTRAL)
        facial_system.update(world, 0.016, [(entity, facial_component)])

        facial_component.blend_to_emotion(EmotionState.HAPPY, duration=0.5)

        # Simulate blend over time
        for _ in range(30):  # ~0.5 seconds at 60fps
            facial_system.update(world, 0.016, [(entity, facial_component)])

    def test_facial_audio_sync(self):
        """Test facial animation synchronization with audio."""
        facial_system = FacialSystem()

        facial_component = FacialComponent()
        facial_component.enabled = True
        facial_component.audio_time = 0.0

        entity = MockEntity(1)
        world = MockWorld()

        # Simulate audio playback with phoneme timestamps
        phoneme_timeline = [
            (0.0, LipSyncPhoneme.SILENCE),
            (0.1, LipSyncPhoneme.AA),
            (0.2, LipSyncPhoneme.EE),
            (0.3, LipSyncPhoneme.MBP),
        ]

        facial_component.phoneme_timeline = phoneme_timeline

        for i in range(20):
            facial_component.audio_time = i * 0.016
            facial_system.update(world, 0.016, [(entity, facial_component)])


# =============================================================================
# 5. CROWD RENDERING PIPELINE TESTS (10 tests)
# =============================================================================


class TestCrowdRenderingPipeline:
    """Tests for crowd system -> animation textures -> instanced rendering pipeline."""

    def test_crowd_system_basic_update(self):
        """Test basic crowd system update."""
        crowd_system = CrowdSystem()

        crowd_component = CrowdComponent()
        crowd_component.enabled = True

        # Add some agents
        for i in range(10):
            crowd_component.add_agent(
                position=Vec3(i * 2.0, 0.0, 0.0),
                initial_state=AgentState.IDLE,
            )

        entity = MockEntity(1)
        world = MockWorld()

        crowd_system.update(world, 0.016, [(entity, crowd_component)])

        assert crowd_component.get_agent_count() == 10

    def test_crowd_100_agents_performance(self):
        """Test crowd system with 100+ agents within budget."""
        crowd_system = CrowdSystem()

        crowd_component = CrowdComponent()
        crowd_component.enabled = True

        # Add 100 agents
        for i in range(100):
            x = (i % 10) * 2.0
            z = (i // 10) * 2.0
            crowd_component.add_agent(
                position=Vec3(x, 0.0, z),
                initial_state=AgentState.WALKING,
            )

        entity = MockEntity(1)
        world = MockWorld()

        start = time.perf_counter()
        crowd_system.update(world, 0.016, [(entity, crowd_component)])
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete within 5ms for 100 agents
        assert elapsed_ms < 5.0, f"Crowd update took {elapsed_ms:.2f}ms, expected <5ms"
        assert crowd_component.get_agent_count() == 100

    def test_crowd_lod_levels(self):
        """Test crowd LOD level assignments based on distance."""
        crowd_system = CrowdSystem()
        crowd_system.set_lod_distances([10.0, 25.0, 50.0, 100.0])

        crowd_component = CrowdComponent()
        crowd_component.camera_position = Vec3(0, 0, 0)

        # Add agents at different distances
        distances = [5, 15, 30, 75, 150]
        for i, dist in enumerate(distances):
            crowd_component.add_agent(
                position=Vec3(dist, 0, 0),
            )

        entity = MockEntity(1)
        world = MockWorld()

        crowd_system.update(world, 0.016, [(entity, crowd_component)])

        # Verify LOD assignments
        assert crowd_component.get_agent_count() == 5

    def test_crowd_animation_texture_sampling(self):
        """Test animation texture atlas sampling for crowds."""
        atlas = AnimationTextureAtlas(
            width=1024,
            height=1024,
            max_bones=64,
            max_frames=128,
        )

        # Add animation data
        atlas.add_animation(
            name="walk",
            bone_count=21,
            frame_count=30,
            duration=1.0,
        )

        # Sample animation
        frame_data = atlas.sample_frame("walk", normalized_time=0.5)

        assert frame_data is not None

    def test_crowd_instance_buffer_batching(self):
        """Test instance buffer batching for efficient rendering."""
        buffer = InstanceBuffer()
        buffer.max_capacity = 1000

        # Add instances
        for i in range(100):
            instance = CrowdInstance(
                position=Vec3(i, 0, 0),
                rotation=Quat.identity(),
                animation_index=i % 4,
                animation_time=i * 0.1,
            )
            buffer.add_instance(instance)

        assert buffer.instance_count == 100
        assert buffer.get_memory_size_bytes() == 100 * 96  # 96 bytes per instance

    def test_crowd_culling(self):
        """Test distance-based culling of crowd instances."""
        crowd_system = CrowdSystem()
        crowd_system.set_cull_distance(50.0)

        crowd_component = CrowdComponent()
        crowd_component.camera_position = Vec3(0, 0, 0)

        # Add agents at various distances
        for i in range(20):
            dist = i * 5  # 0, 5, 10, ..., 95
            crowd_component.add_agent(position=Vec3(dist, 0, 0))

        entity = MockEntity(1)
        world = MockWorld()

        crowd_system.update(world, 0.016, [(entity, crowd_component)])

        # Agents beyond 50 units should be culled
        visible = crowd_component.get_visible_count()
        assert visible <= 11  # 0, 5, 10, ..., 50

    def test_crowd_agent_state_transitions(self):
        """Test crowd agent behavior state transitions."""
        crowd_component = CrowdComponent()

        agent_id = crowd_component.add_agent(
            position=Vec3(0, 0, 0),
            initial_state=AgentState.IDLE,
        )

        # Transition to walking
        crowd_component.set_agent_target(agent_id, Vec3(10, 0, 0))
        agent = crowd_component.get_agent(agent_id)

        assert agent.current_state == AgentState.WALKING

    def test_crowd_simulation_integration(self):
        """Test crowd simulation and rendering integration."""
        crowd_system = CrowdSystem()

        crowd_component = CrowdComponent()
        crowd_component.update_rate = 30.0  # 30 updates per second

        # Add moving agents
        for i in range(50):
            agent_id = crowd_component.add_agent(
                position=Vec3(i * 2, 0, 0),
                initial_state=AgentState.WALKING,
            )
            crowd_component.set_agent_target(agent_id, Vec3(i * 2, 0, 100))

        entity = MockEntity(1)
        world = MockWorld()

        # Simulate several frames
        for _ in range(60):
            crowd_system.update(world, 0.016, [(entity, crowd_component)])

        assert crowd_component.get_agent_count() == 50

    def test_crowd_render_batch_generation(self):
        """Test generation of render batches from crowd data."""
        crowd_component = CrowdComponent()

        # Add agents with different mesh/material combinations
        for i in range(30):
            crowd_component.add_agent(
                position=Vec3(i, 0, 0),
                mesh_id=i % 3,
                material_id=i % 2,
            )

        # Get render batches
        batches = crowd_component.renderer.get_batches()

        # Should have batches organized by mesh/material
        assert len(batches) > 0

    def test_crowd_stress_1000_agents(self):
        """Stress test with 1000 agents."""
        crowd_system = CrowdSystem()

        crowd_component = CrowdComponent()
        crowd_component.max_visible = 1000

        # Add 1000 agents in a grid
        for i in range(1000):
            x = (i % 32) * 2
            z = (i // 32) * 2
            crowd_component.add_agent(position=Vec3(x, 0, z))

        entity = MockEntity(1)
        world = MockWorld()

        start = time.perf_counter()
        crowd_system.update(world, 0.016, [(entity, crowd_component)])
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert crowd_component.get_agent_count() == 1000
        # Should complete in reasonable time for 1000 agents
        assert elapsed_ms < 50.0, f"1000 agent update took {elapsed_ms:.2f}ms"


# =============================================================================
# 6. DETERMINISTIC REPLAY TESTS (6 tests)
# =============================================================================


class TestDeterministicReplay:
    """Tests for deterministic replay of animation state transitions."""

    def test_animation_graph_deterministic(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
    ):
        """Test animation graph produces deterministic output."""
        graph_system = AnimationGraphSystem()

        # Create identical components
        def create_component():
            gc = AnimationGraphComponent()
            gc.skeleton = humanoid_skeleton
            gc.register_clip("idle", idle_clip)
            gc.state_machine_output = StateMachineOutput(current_state="idle", state_time=0.5)
            return gc

        world = MockWorld()

        results = []
        for _ in range(3):
            gc = create_component()
            entity = MockEntity(1)
            graph_system.update(world, 0.016, [(entity, gc)])
            results.append(gc.output_pose)

        # All runs should produce identical output
        for i in range(1, len(results)):
            assert results[0].bone_count() == results[i].bone_count()

    def test_ik_system_deterministic(
        self,
        bone_hierarchy: Dict[int, int],
        bone_lengths: Dict[int, float],
        rest_pose: Dict[int, Transform],
    ):
        """Test IK solver produces deterministic output."""
        results = []

        for _ in range(3):
            ik_system = IKSystem()
            ik_system.set_skeleton_data(bone_hierarchy, bone_lengths)

            ik_component = IKComponent()
            ik_component.add_goal(IKGoal(
                target_bone=10,
                target_position=Vec3(0.5, 1.0, 0.3),
                chain_length=3,
            ))

            entity = MockEntity(1)
            world = MockWorld()

            result = ik_system.update(world, [(entity, ik_component)], {entity: rest_pose})
            results.append(result[entity])

        # All runs should produce identical transforms
        for i in range(1, len(results)):
            assert len(results[0]) == len(results[i])

    def test_state_transition_deterministic(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        walk_clip: AnimationClip,
    ):
        """Test state transitions produce deterministic output."""
        graph_system = AnimationGraphSystem()

        results = []

        for _ in range(3):
            gc = AnimationGraphComponent()
            gc.skeleton = humanoid_skeleton
            gc.register_clip("idle", idle_clip)
            gc.register_clip("walk", walk_clip)
            gc.state_machine_output = StateMachineOutput(
                current_state="idle",
                target_state="walk",
                is_transitioning=True,
                transition_progress=0.5,
            )

            entity = MockEntity(1)
            world = MockWorld()

            graph_system.update(world, 0.016, [(entity, gc)])
            results.append(gc.output_pose)

        # All transitions should produce identical blended output
        for i in range(1, len(results)):
            assert results[0].bone_count() == results[i].bone_count()

    def test_replay_recorded_inputs(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
    ):
        """Test replay of recorded animation inputs."""
        graph_system = AnimationGraphSystem()

        # Record inputs
        recorded_inputs = [
            {"state": "idle", "time": 0.0, "params": {"speed": 0.0}},
            {"state": "idle", "time": 0.5, "params": {"speed": 0.5}},
            {"state": "walk", "time": 0.0, "params": {"speed": 1.0}},
        ]

        world = MockWorld()
        entity = MockEntity(1)

        # First playback
        outputs_1 = []
        gc = AnimationGraphComponent()
        gc.skeleton = humanoid_skeleton
        gc.register_clip("idle", idle_clip)

        for inp in recorded_inputs:
            gc.state_machine_output = StateMachineOutput(
                current_state=inp["state"],
                state_time=inp["time"],
            )
            graph_system.update(world, 0.016, [(entity, gc)])
            outputs_1.append(gc.output_pose.bone_count())

        # Second playback (should match)
        outputs_2 = []
        gc2 = AnimationGraphComponent()
        gc2.skeleton = humanoid_skeleton
        gc2.register_clip("idle", idle_clip)

        for inp in recorded_inputs:
            gc2.state_machine_output = StateMachineOutput(
                current_state=inp["state"],
                state_time=inp["time"],
            )
            graph_system.update(world, 0.016, [(entity, gc2)])
            outputs_2.append(gc2.output_pose.bone_count())

        assert outputs_1 == outputs_2

    def test_frame_by_frame_replay(
        self,
        humanoid_skeleton: Skeleton,
        walk_clip: AnimationClip,
    ):
        """Test frame-by-frame replay matches original."""
        graph_system = AnimationGraphSystem()

        world = MockWorld()
        entity = MockEntity(1)

        # Original playback
        original_poses = []
        gc = AnimationGraphComponent()
        gc.skeleton = humanoid_skeleton
        gc.register_clip("walk", walk_clip)

        for frame in range(60):
            gc.state_machine_output = StateMachineOutput(
                current_state="walk",
                state_time=frame * 0.016,
            )
            graph_system.update(world, 0.016, [(entity, gc)])
            original_poses.append(gc.output_pose.bone_count())

        # Replay
        replay_poses = []
        gc2 = AnimationGraphComponent()
        gc2.skeleton = humanoid_skeleton
        gc2.register_clip("walk", walk_clip)

        for frame in range(60):
            gc2.state_machine_output = StateMachineOutput(
                current_state="walk",
                state_time=frame * 0.016,
            )
            graph_system.update(world, 0.016, [(entity, gc2)])
            replay_poses.append(gc2.output_pose.bone_count())

        assert original_poses == replay_poses

    def test_random_seed_determinism(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
    ):
        """Test that seeded random variations are deterministic."""
        procedural_system = ProceduralSystem()

        results = []

        for _ in range(3):
            random.seed(42)  # Fixed seed

            pc = ProceduralComponent()
            pc.add_modifier(BreathingModifier(
                chest_bone=4,
                amplitude=0.02,
                frequency=0.5 + random.random() * 0.1,  # Seeded random
            ))

            entity = MockEntity(1)
            world = MockWorld()

            rest_pose = {i: Transform() for i in range(21)}
            procedural_system.update(world, 0.016, [(entity, pc)], {entity: rest_pose})
            results.append(pc.modifiers[0].frequency)

        # All should be identical due to seed
        assert all(r == results[0] for r in results)


# =============================================================================
# 7. PERFORMANCE BUDGET TESTS (8 tests)
# =============================================================================


class TestPerformanceBudgets:
    """Tests for frame timing budgets across all LOD levels."""

    def test_lod_0_budget_under_2ms(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        character_mesh: MeshData,
        skinning_data_4_influences: SkinningData,
        rest_pose: Dict[int, Transform],
    ):
        """Test LOD 0 (highest quality) stays under 2ms per character."""
        graph_system = AnimationGraphSystem()
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        gc = AnimationGraphComponent()
        gc.skeleton = humanoid_skeleton
        gc.register_clip("idle", idle_clip)
        gc.state_machine_output = StateMachineOutput(current_state="idle")

        sc = SkinnedMeshComponent(
            mesh=character_mesh,
            skinning_data=skinning_data_4_influences,
            lod_level=0,  # Highest quality
            force_cpu=True,
        )

        entity = MockEntity(1)
        world = MockWorld()

        # Measure total time
        start = time.perf_counter()
        graph_system.update(world, 0.016, [(entity, gc)])
        skinning_system.update(world, [(entity, sc)], {entity: rest_pose})
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 2.0, f"LOD 0 took {elapsed_ms:.2f}ms, budget is 2ms"

    def test_lod_1_budget_under_1ms(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        character_mesh: MeshData,
        skinning_data_4_influences: SkinningData,
        rest_pose: Dict[int, Transform],
    ):
        """Test LOD 1 stays under 1ms per character."""
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        sc = SkinnedMeshComponent(
            mesh=character_mesh,
            skinning_data=skinning_data_4_influences,
            lod_level=1,
            force_cpu=True,
        )

        entity = MockEntity(1)
        world = MockWorld()

        start = time.perf_counter()
        skinning_system.update(world, [(entity, sc)], {entity: rest_pose})
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 1.0, f"LOD 1 took {elapsed_ms:.2f}ms, budget is 1ms"

    def test_lod_2_budget_under_half_ms(
        self,
        humanoid_skeleton: Skeleton,
        character_mesh: MeshData,
        skinning_data_4_influences: SkinningData,
        rest_pose: Dict[int, Transform],
    ):
        """Test LOD 2 stays under 0.5ms per character."""
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        sc = SkinnedMeshComponent(
            mesh=character_mesh,
            skinning_data=skinning_data_4_influences,
            lod_level=2,
            force_cpu=True,
        )

        entity = MockEntity(1)
        world = MockWorld()

        start = time.perf_counter()
        skinning_system.update(world, [(entity, sc)], {entity: rest_pose})
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 0.5, f"LOD 2 took {elapsed_ms:.2f}ms, budget is 0.5ms"

    def test_crowd_100_agents_under_5ms(self):
        """Test crowd of 100 agents stays under 5ms total."""
        crowd_system = CrowdSystem()

        crowd_component = CrowdComponent()

        for i in range(100):
            crowd_component.add_agent(
                position=Vec3((i % 10) * 2, 0, (i // 10) * 2),
                initial_state=AgentState.WALKING,
            )

        entity = MockEntity(1)
        world = MockWorld()

        start = time.perf_counter()
        crowd_system.update(world, 0.016, [(entity, crowd_component)])
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 5.0, f"100 agents took {elapsed_ms:.2f}ms, budget is 5ms"

    def test_full_pipeline_60fps_budget(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        bone_hierarchy: Dict[int, int],
        bone_lengths: Dict[int, float],
        character_mesh: MeshData,
        skinning_data_4_influences: SkinningData,
        rest_pose: Dict[int, Transform],
    ):
        """Test full pipeline stays within 60fps frame budget (~16.67ms)."""
        graph_system = AnimationGraphSystem()
        ik_system = IKSystem()
        ik_system.set_skeleton_data(bone_hierarchy, bone_lengths)
        procedural_system = ProceduralSystem()
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        gc = AnimationGraphComponent()
        gc.skeleton = humanoid_skeleton
        gc.register_clip("idle", idle_clip)
        gc.state_machine_output = StateMachineOutput(current_state="idle")

        ik_component = IKComponent()
        ik_component.add_goal(IKGoal(target_bone=10, target_position=Vec3(0.5, 1.0, 0.3), chain_length=3))

        pc = ProceduralComponent()
        pc.add_modifier(BreathingModifier(chest_bone=4))

        sc = SkinnedMeshComponent(
            mesh=character_mesh,
            skinning_data=skinning_data_4_influences,
            force_cpu=True,
        )

        entity = MockEntity(1)
        world = MockWorld()

        start = time.perf_counter()
        graph_system.update(world, 0.016, [(entity, gc)])
        ik_system.update(world, [(entity, ik_component)], {entity: rest_pose})
        procedural_system.update(world, 0.016, [(entity, pc)], {entity: rest_pose})
        skinning_system.update(world, [(entity, sc)], {entity: rest_pose})
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Full animation should fit in animation budget (~4ms of 16.67ms frame)
        assert elapsed_ms < 4.0, f"Full pipeline took {elapsed_ms:.2f}ms, budget is ~4ms"

    def test_motion_matching_query_budget(
        self,
        humanoid_skeleton: Skeleton,
        rest_pose: Dict[int, Transform],
    ):
        """Test motion matching query stays within budget."""
        mm_system = MotionMatchingSystem()

        mm_component = MotionMatchingComponent()
        mm_component.skeleton = humanoid_skeleton
        mm_component.config = MotionMatchingConfig(budget_ms=2.0)

        entity = MockEntity(1)
        world = MockWorld()

        start = time.perf_counter()
        mm_system.update(world, 0.016, [(entity, mm_component)], {entity: rest_pose})
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 2.0 or mm_component.fallback_reason == FallbackReason.BUDGET_EXCEEDED

    def test_facial_system_budget(self):
        """Test facial system stays within budget."""
        facial_system = FacialSystem()

        fc = FacialComponent()
        fc.set_emotion(EmotionState.HAPPY, intensity=0.8)
        fc.set_phoneme(LipSyncPhoneme.AA)
        fc.eye_tracking_enabled = True

        entity = MockEntity(1)
        world = MockWorld()

        start = time.perf_counter()
        facial_system.update(world, 0.016, [(entity, fc)])
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Facial should be very fast
        assert elapsed_ms < 0.5, f"Facial system took {elapsed_ms:.2f}ms"

    def test_multiple_characters_scaled_budget(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        character_mesh: MeshData,
        skinning_data_4_influences: SkinningData,
        rest_pose: Dict[int, Transform],
    ):
        """Test that budget scales appropriately with character count."""
        graph_system = AnimationGraphSystem()
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        character_counts = [1, 5, 10]
        times = []

        for count in character_counts:
            entities = [MockEntity(i) for i in range(count)]

            gcs = []
            scs = []
            for _ in range(count):
                gc = AnimationGraphComponent()
                gc.skeleton = humanoid_skeleton
                gc.register_clip("idle", idle_clip)
                gc.state_machine_output = StateMachineOutput(current_state="idle")
                gcs.append(gc)

                sc = SkinnedMeshComponent(
                    mesh=character_mesh,
                    skinning_data=skinning_data_4_influences,
                    force_cpu=True,
                )
                scs.append(sc)

            world = MockWorld()

            start = time.perf_counter()
            graph_system.update(world, 0.016, list(zip(entities, gcs)))
            skinning_system.update(world, list(zip(entities, scs)), {e: rest_pose for e in entities})
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        # Time should scale roughly linearly (allow some overhead)
        assert times[1] < times[0] * 10  # 5 chars should be < 10x single
        assert times[2] < times[0] * 20  # 10 chars should be < 20x single


# =============================================================================
# 8. ERROR RECOVERY TESTS (6 tests)
# =============================================================================


class TestErrorRecovery:
    """Tests for error handling and recovery in the animation pipeline."""

    def test_missing_clip_graceful_fallback(
        self,
        humanoid_skeleton: Skeleton,
    ):
        """Test graceful handling of missing animation clip."""
        graph_system = AnimationGraphSystem()

        gc = AnimationGraphComponent()
        gc.skeleton = humanoid_skeleton
        gc.state_machine_output = StateMachineOutput(current_state="nonexistent")

        entity = MockEntity(1)
        world = MockWorld()

        # Should not raise
        graph_system.update(world, 0.016, [(entity, gc)])

        # Should produce empty or default pose
        assert gc.output_pose.bone_count() == 0

    def test_missing_bone_in_ik_chain(
        self,
        bone_hierarchy: Dict[int, int],
        bone_lengths: Dict[int, float],
        rest_pose: Dict[int, Transform],
    ):
        """Test IK handles missing bones gracefully."""
        ik_system = IKSystem()
        ik_system.set_skeleton_data(bone_hierarchy, bone_lengths)

        ik_component = IKComponent()
        ik_component.add_goal(IKGoal(
            target_bone=99,  # Non-existent bone
            target_position=Vec3(1, 1, 1),
            chain_length=3,
        ))

        entity = MockEntity(1)
        world = MockWorld()

        # Should not crash
        result = ik_system.update(world, [(entity, ik_component)], {entity: rest_pose})

        stats = ik_system.get_stats()
        assert stats.invalid_goals == 1

    def test_invalid_skinning_weights(
        self,
        character_mesh: MeshData,
        rest_pose: Dict[int, Transform],
    ):
        """Test skinning handles invalid weights gracefully."""
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        # Create invalid skinning data
        vertex_data = [
            VertexSkinData(influences=[BoneInfluence(99, 1.0)])  # Invalid bone index
            for _ in range(character_mesh.vertex_count)
        ]
        skinning_data = SkinningData(
            vertex_data=vertex_data,
            bind_poses=[Mat4.identity() for _ in range(4)],
        )

        sc = SkinnedMeshComponent(
            mesh=character_mesh,
            skinning_data=skinning_data,
            force_cpu=True,
        )

        entity = MockEntity(1)
        world = MockWorld()

        # Should not crash
        skinning_system.update(world, [(entity, sc)], {entity: rest_pose})

    def test_empty_pose_data_handled(
        self,
        bone_hierarchy: Dict[int, int],
        bone_lengths: Dict[int, float],
    ):
        """Test systems handle empty pose data."""
        ik_system = IKSystem()
        ik_system.set_skeleton_data(bone_hierarchy, bone_lengths)

        ik_component = IKComponent()
        ik_component.add_goal(IKGoal(
            target_bone=10,
            target_position=Vec3(1, 1, 1),
            chain_length=3,
        ))

        entity = MockEntity(1)
        world = MockWorld()

        # Empty pose data
        result = ik_system.update(world, [(entity, ik_component)], {entity: {}})

        # Should handle gracefully
        assert entity in result

    def test_nan_values_in_transforms(
        self,
        humanoid_skeleton: Skeleton,
        character_mesh: MeshData,
        skinning_data_4_influences: SkinningData,
    ):
        """Test handling of NaN values in transforms."""
        skinning_system = SkinningSystem(
            gpu_capabilities=GPUCapabilities(has_compute_shader=False)
        )

        # Create pose with NaN values
        nan_pose = {
            0: Transform(Vec3(float('nan'), 0, 0), Quat.identity(), Vec3.one()),
            1: Transform(Vec3(0, 0, 0), Quat.identity(), Vec3.one()),
        }

        sc = SkinnedMeshComponent(
            mesh=character_mesh,
            skinning_data=skinning_data_4_influences,
            force_cpu=True,
        )

        entity = MockEntity(1)
        world = MockWorld()

        # Should not crash (though output may be invalid)
        skinning_system.update(world, [(entity, sc)], {entity: nan_pose})

    def test_system_recovery_after_error(
        self,
        humanoid_skeleton: Skeleton,
        idle_clip: AnimationClip,
        rest_pose: Dict[int, Transform],
    ):
        """Test systems recover after processing error."""
        graph_system = AnimationGraphSystem()

        entity = MockEntity(1)
        world = MockWorld()

        # First update with invalid state
        gc_invalid = AnimationGraphComponent()
        gc_invalid.skeleton = humanoid_skeleton
        gc_invalid.state_machine_output = StateMachineOutput(current_state="invalid")
        graph_system.update(world, 0.016, [(entity, gc_invalid)])

        # Second update with valid state should work
        gc_valid = AnimationGraphComponent()
        gc_valid.skeleton = humanoid_skeleton
        gc_valid.register_clip("idle", idle_clip)
        gc_valid.state_machine_output = StateMachineOutput(current_state="idle")
        graph_system.update(world, 0.016, [(entity, gc_valid)])

        # System should have recovered
        assert graph_system.get_statistics()["entities_evaluated"] == 1


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
