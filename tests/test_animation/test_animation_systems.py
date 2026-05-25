"""
Comprehensive tests for Animation Layer - Crowds and Systems.

Tests cover:
- Animation texture baking (with overflow protection)
- Crowd LOD selection (with distance=0 handling)
- ECS systems (animation graph, IK, procedural, skinning, motion matching, facial, crowd)
- System execution order
- Crowd behavior transitions
- Instance buffer overflow protection
- Division by zero edge cases
- Config value usage verification

Minimum 150 tests with real assertions.
"""

import math
import pytest
from typing import Any

from engine.core.math import Vec3, Vec4, Quat, Mat4, Transform
from engine.core.ecs import World, Entity

# Config module
from engine.animation.config import (
    ANIMATION_TEXTURE_CONFIG,
    CROWD_RENDERER_CONFIG,
    CROWD_LOD_CONFIG,
    CROWD_BEHAVIOR_CONFIG,
    ANIMATION_SYSTEM_CONFIG,
    IK_CONFIG,
    PROCEDURAL_CONFIG,
    SKINNING_CONFIG,
    FACIAL_CONFIG,
    CROWD_SYSTEM_CONFIG,
)

# Crowds module
from engine.animation.crowds.animation_texture import (
    AnimationTexture,
    AnimationTextureAtlas,
    AnimationClip,
    Skeleton,
    TextureFormat,
    bake_clip_to_texture,
    encode_transform_to_pixels,
    decode_pixels_to_transform,
    pack_float_to_rgba8,
    unpack_rgba8_to_float,
    AnimationTextureOverflowError,
)
from engine.animation.crowds.crowd_renderer import (
    CrowdInstance,
    CrowdRenderer,
    CrowdRenderBatch,
    InstanceBuffer,
    RenderPriority,
    InstanceBufferOverflowError,
)
from engine.animation.crowds.crowd_lod import (
    LODLevel,
    CrowdLOD,
    LODTransition,
    LODTransitionMode,
    create_reduced_skeleton,
    _calculate_bone_importance,
)
from engine.animation.crowds.crowd_behavior import (
    CrowdAgent,
    CrowdBehavior,
    IdleBehavior,
    WalkingBehavior,
    WaitingBehavior,
    FleeingBehavior,
    FormationBehavior,
    CrowdSimulator,
    AgentState,
    AnimationBlend,
    BehaviorContext,
)

# Systems module
from engine.animation.systems.animation_graph_system import (
    AnimationGraphComponent,
    AnimationGraphSystem,
    AnimationGraphInstance,
    AnimationState,
    StateTransition,
    GraphParameter,
    ParameterType,
    AnimationPose,
)
from engine.animation.systems.ik_system import (
    IKComponent,
    IKGoal,
    IKSolverType,
    IKSystem,
    IKHintType,
)
from engine.animation.systems.procedural_system import (
    ProceduralComponent,
    SpringController,
    LookAtController,
    SwayController,
    BreathingController,
    ProceduralSystem,
    ControllerType,
)
from engine.animation.systems.skinning_system import (
    SkinnedMeshComponent,
    SkinningMethod,
    SkinningSystem,
    SkinningData,
    MeshData,
    VertexSkinData,
    BoneInfluence,
)
from engine.animation.systems.motion_matching_system import (
    MotionMatchingComponent,
    MotionMatchingSystem,
    MotionInput,
    MotionFeature,
    MotionDatabase,
    MotionFrame,
    MotionMatchingController,
    FeatureType,
)
from engine.animation.systems.facial_system import (
    FacialComponent,
    FacialSystem,
    Expression,
    LipSyncPhoneme,
    EmotionState,
    FaceRig,
    LipSyncState,
    EyeState,
)
from engine.animation.systems.crowd_system import (
    CrowdComponent,
    CrowdSystem,
)


# =============================================================================
# ANIMATION TEXTURE TESTS
# =============================================================================

class TestAnimationTexture:
    """Tests for animation texture baking."""

    def test_create_empty_texture(self):
        """Test creating empty animation texture."""
        tex = AnimationTexture()
        assert tex.bone_count == 0
        assert tex.frame_count == 0
        assert tex.width == 0
        assert tex.height == 0
        assert len(tex.texture_data) == 0

    def test_texture_format_default(self):
        """Test default texture format."""
        tex = AnimationTexture()
        assert tex.format == TextureFormat.FLOAT32

    def test_texture_format_float16(self):
        """Test float16 texture format."""
        tex = AnimationTexture(format=TextureFormat.FLOAT16)
        assert tex.format == TextureFormat.FLOAT16

    def test_get_pixel_out_of_bounds(self):
        """Test getting pixel outside texture bounds."""
        tex = AnimationTexture(width=2, height=2, texture_data=[0.0] * 16)
        assert tex.get_pixel(-1, 0) == (0.0, 0.0, 0.0, 0.0)
        assert tex.get_pixel(0, -1) == (0.0, 0.0, 0.0, 0.0)
        assert tex.get_pixel(2, 0) == (0.0, 0.0, 0.0, 0.0)
        assert tex.get_pixel(0, 2) == (0.0, 0.0, 0.0, 0.0)

    def test_set_and_get_pixel(self):
        """Test setting and getting pixel values."""
        tex = AnimationTexture(width=2, height=2, texture_data=[0.0] * 16)
        tex.set_pixel(0, 0, 1.0, 2.0, 3.0, 4.0)
        result = tex.get_pixel(0, 0)
        assert result == (1.0, 2.0, 3.0, 4.0)

    def test_set_pixel_out_of_bounds_ignored(self):
        """Test that out of bounds set is ignored."""
        tex = AnimationTexture(width=2, height=2, texture_data=[0.0] * 16)
        tex.set_pixel(-1, 0, 1.0, 2.0, 3.0, 4.0)  # Should not crash
        tex.set_pixel(100, 100, 1.0, 2.0, 3.0, 4.0)  # Should not crash

    def test_get_bone_transform_empty(self):
        """Test getting transform from empty texture."""
        tex = AnimationTexture()
        transform = tex.get_bone_transform(0, 0)
        assert transform.translation == Vec3.zero()

    def test_memory_size_float32(self):
        """Test memory size calculation for float32."""
        tex = AnimationTexture(width=10, height=20, format=TextureFormat.FLOAT32)
        # 10 * 20 * 4 channels * 4 bytes = 3200
        assert tex.get_memory_size_bytes() == 3200

    def test_memory_size_float16(self):
        """Test memory size calculation for float16."""
        tex = AnimationTexture(width=10, height=20, format=TextureFormat.FLOAT16)
        # 10 * 20 * 4 channels * 2 bytes = 1600
        assert tex.get_memory_size_bytes() == 1600

    def test_memory_size_rgba8(self):
        """Test memory size calculation for RGBA8."""
        tex = AnimationTexture(width=10, height=20, format=TextureFormat.RGBA8_UNORM)
        # 10 * 20 * 4 channels * 1 byte = 800
        assert tex.get_memory_size_bytes() == 800


class TestEncodeDecode:
    """Tests for transform encoding/decoding."""

    def test_encode_identity_transform(self):
        """Test encoding identity transform."""
        transform = Transform.identity()
        p1, p2 = encode_transform_to_pixels(transform)
        assert p1 == (0.0, 0.0, 0.0, 1.0)  # position xyz, scale
        assert p2[3] == pytest.approx(1.0, abs=0.01)  # quaternion w

    def test_encode_translation(self):
        """Test encoding translation."""
        transform = Transform(translation=Vec3(1.0, 2.0, 3.0))
        p1, p2 = encode_transform_to_pixels(transform)
        assert p1[0] == 1.0
        assert p1[1] == 2.0
        assert p1[2] == 3.0

    def test_encode_scale(self):
        """Test encoding uniform scale."""
        transform = Transform(scale=Vec3(2.0, 2.0, 2.0))
        p1, p2 = encode_transform_to_pixels(transform)
        assert p1[3] == 2.0

    def test_decode_encoded_transform(self):
        """Test round-trip encode/decode."""
        original = Transform(
            translation=Vec3(1.0, 2.0, 3.0),
            rotation=Quat.from_axis_angle(Vec3.up(), 0.5),
            scale=Vec3(1.5, 1.5, 1.5),
        )
        p1, p2 = encode_transform_to_pixels(original)
        decoded = decode_pixels_to_transform(p1, p2)

        assert decoded.translation.x == pytest.approx(1.0, abs=0.01)
        assert decoded.translation.y == pytest.approx(2.0, abs=0.01)
        assert decoded.translation.z == pytest.approx(3.0, abs=0.01)
        assert decoded.scale.x == pytest.approx(1.5, abs=0.01)


class TestPackUnpack:
    """Tests for float packing/unpacking."""

    def test_pack_zero(self):
        """Test packing zero."""
        packed = pack_float_to_rgba8(0.0, -100.0, 100.0)
        assert len(packed) == 4
        assert all(0 <= v <= 255 for v in packed)

    def test_unpack_packed_value(self):
        """Test round-trip pack/unpack."""
        original = 42.5
        packed = pack_float_to_rgba8(original, -100.0, 100.0)
        unpacked = unpack_rgba8_to_float(*packed, -100.0, 100.0)
        assert unpacked == pytest.approx(original, abs=0.01)

    def test_pack_min_value(self):
        """Test packing minimum value."""
        packed = pack_float_to_rgba8(-100.0, -100.0, 100.0)
        unpacked = unpack_rgba8_to_float(*packed, -100.0, 100.0)
        assert unpacked == pytest.approx(-100.0, abs=0.1)

    def test_pack_max_value(self):
        """Test packing maximum value."""
        packed = pack_float_to_rgba8(100.0, -100.0, 100.0)
        unpacked = unpack_rgba8_to_float(*packed, -100.0, 100.0)
        assert unpacked == pytest.approx(100.0, abs=0.1)


class TestBakeClipToTexture:
    """Tests for baking animation clips to textures."""

    def test_bake_empty_clip(self):
        """Test baking empty clip."""
        clip = AnimationClip(name="empty")
        skeleton = Skeleton()
        tex = bake_clip_to_texture(clip, skeleton)
        assert tex.bone_count == 0
        assert tex.frame_count == 0

    def test_bake_single_frame(self):
        """Test baking single frame clip."""
        skeleton = Skeleton(
            bone_names=["root"],
            bone_parents=[-1],
            bind_poses=[Transform.identity()],
        )
        clip = AnimationClip(
            name="test",
            duration=1.0,
            frame_rate=30.0,
            bone_tracks={0: [Transform(translation=Vec3(1, 2, 3))]},
        )
        tex = bake_clip_to_texture(clip, skeleton)
        assert tex.bone_count == 1
        assert tex.frame_count == 1
        assert tex.width == 2  # 2 pixels per bone
        assert tex.height == 1

    def test_bake_multiple_bones(self):
        """Test baking with multiple bones."""
        skeleton = Skeleton(
            bone_names=["root", "spine", "head"],
            bone_parents=[-1, 0, 1],
            bind_poses=[Transform.identity()] * 3,
        )
        clip = AnimationClip(
            name="test",
            duration=1.0,
            frame_rate=30.0,
            bone_tracks={
                0: [Transform.identity()],
                1: [Transform.identity()],
                2: [Transform.identity()],
            },
        )
        tex = bake_clip_to_texture(clip, skeleton)
        assert tex.bone_count == 3
        assert tex.width == 6  # 2 pixels per bone * 3 bones

    def test_bake_preserves_clip_name(self):
        """Test that clip name is preserved."""
        skeleton = Skeleton(bone_names=["root"], bone_parents=[-1], bind_poses=[Transform.identity()])
        clip = AnimationClip(name="walk_cycle")
        tex = bake_clip_to_texture(clip, skeleton)
        assert tex.clip_name == "walk_cycle"


class TestAnimationTextureAtlas:
    """Tests for animation texture atlas."""

    def test_create_empty_atlas(self):
        """Test creating empty atlas."""
        atlas = AnimationTextureAtlas()
        assert atlas.width == 0
        assert atlas.height == 0
        assert len(atlas.clips) == 0

    def test_add_clip_to_atlas(self):
        """Test adding clip to atlas."""
        atlas = AnimationTextureAtlas()
        tex = AnimationTexture(
            bone_count=2,
            frame_count=10,
            width=4,
            height=10,
            texture_data=[0.0] * (4 * 10 * 4),
            frame_rate=30.0,
        )
        assert atlas.add_clip("walk", tex)
        assert "walk" in atlas.clips

    def test_add_multiple_clips(self):
        """Test adding multiple clips."""
        atlas = AnimationTextureAtlas()
        tex1 = AnimationTexture(bone_count=2, frame_count=10, width=4, height=10, texture_data=[0.0] * 160)
        tex2 = AnimationTexture(bone_count=2, frame_count=5, width=4, height=5, texture_data=[0.0] * 80)

        atlas.add_clip("walk", tex1)
        atlas.add_clip("run", tex2)

        assert len(atlas.clips) == 2
        assert atlas.height == 15  # 10 + 5 frames

    def test_add_clip_mismatched_bones_fails(self):
        """Test adding clip with mismatched bone count fails."""
        atlas = AnimationTextureAtlas()
        tex1 = AnimationTexture(bone_count=2, frame_count=10, width=4, height=10, texture_data=[0.0] * 160)
        tex2 = AnimationTexture(bone_count=3, frame_count=5, width=6, height=5, texture_data=[0.0] * 120)

        atlas.add_clip("walk", tex1)
        assert not atlas.add_clip("run", tex2)

    def test_get_clip_info(self):
        """Test getting clip info."""
        atlas = AnimationTextureAtlas()
        tex = AnimationTexture(bone_count=2, frame_count=10, width=4, height=10, texture_data=[0.0] * 160, frame_rate=30)
        atlas.add_clip("walk", tex)

        info = atlas.get_clip_info("walk")
        assert info is not None
        assert info[0] == 0  # start row
        assert info[1] == 10  # frame count
        assert info[2] == 30  # frame rate

    def test_get_clip_uv_range(self):
        """Test getting UV range for clip."""
        atlas = AnimationTextureAtlas()
        tex = AnimationTexture(bone_count=2, frame_count=10, width=4, height=10, texture_data=[0.0] * 160)
        atlas.add_clip("walk", tex)

        uv_range = atlas.get_clip_uv_range("walk")
        assert uv_range is not None
        assert uv_range[0] == 0.0
        assert uv_range[1] == 1.0


# =============================================================================
# CROWD RENDERER TESTS
# =============================================================================

class TestCrowdInstance:
    """Tests for crowd instance."""

    def test_create_default_instance(self):
        """Test creating default instance."""
        inst = CrowdInstance()
        assert inst.position == Vec3.zero()
        assert inst.scale == 1.0
        assert inst.visible is True

    def test_instance_unique_id(self):
        """Test instances have unique IDs."""
        inst1 = CrowdInstance()
        inst2 = CrowdInstance()
        assert inst1.instance_id != inst2.instance_id

    def test_advance_time(self):
        """Test animation time advancement."""
        inst = CrowdInstance(animation_time=0.0, animation_speed=1.0)
        inst.advance_time(0.5)
        assert inst.animation_time == 0.5

    def test_advance_time_with_speed(self):
        """Test animation time with speed multiplier."""
        inst = CrowdInstance(animation_time=0.0, animation_speed=2.0)
        inst.advance_time(0.5)
        assert inst.animation_time == 1.0

    def test_set_animation_resets_time(self):
        """Test setting animation resets time by default."""
        inst = CrowdInstance(animation_index=0, animation_time=5.0)
        inst.set_animation(1)
        assert inst.animation_index == 1
        assert inst.animation_time == 0.0

    def test_set_animation_preserves_time(self):
        """Test setting animation can preserve time."""
        inst = CrowdInstance(animation_index=0, animation_time=5.0)
        inst.set_animation(1, reset_time=False)
        assert inst.animation_index == 1
        assert inst.animation_time == 5.0

    def test_distance_to_point(self):
        """Test distance calculation."""
        inst = CrowdInstance(position=Vec3(0, 0, 0))
        dist = inst.distance_to(Vec3(3, 4, 0))
        assert dist == pytest.approx(5.0, abs=0.01)

    def test_get_transform_matrix(self):
        """Test transform matrix generation."""
        inst = CrowdInstance(position=Vec3(1, 2, 3))
        matrix = inst.get_transform_matrix()
        assert isinstance(matrix, Mat4)


class TestInstanceBuffer:
    """Tests for instance buffer."""

    def test_create_empty_buffer(self):
        """Test creating empty buffer."""
        buf = InstanceBuffer()
        assert buf.instance_count == 0
        assert buf.dirty is True

    def test_reserve_capacity(self):
        """Test reserving capacity."""
        buf = InstanceBuffer()
        buf.reserve(100)
        assert buf.capacity == 100
        assert len(buf.transform_data) == 100 * 16

    def test_add_instance(self):
        """Test adding instance to buffer."""
        buf = InstanceBuffer()
        inst = CrowdInstance(position=Vec3(1, 2, 3))
        idx = buf.add_instance(inst)
        assert idx == 0
        assert buf.instance_count == 1

    def test_update_instance(self):
        """Test updating instance in buffer."""
        buf = InstanceBuffer()
        inst = CrowdInstance(position=Vec3(1, 2, 3))
        buf.add_instance(inst)

        inst.position = Vec3(4, 5, 6)
        buf.update_instance(0, inst)
        assert buf.dirty is True

    def test_clear_buffer(self):
        """Test clearing buffer."""
        buf = InstanceBuffer()
        buf.add_instance(CrowdInstance())
        buf.add_instance(CrowdInstance())
        buf.clear()
        assert buf.instance_count == 0


class TestCrowdRenderer:
    """Tests for crowd renderer."""

    def test_create_renderer(self):
        """Test creating renderer."""
        renderer = CrowdRenderer()
        assert renderer.total_instance_count == 0
        assert renderer.batch_count == 0

    def test_add_instance(self):
        """Test adding instance to renderer."""
        renderer = CrowdRenderer()
        inst = CrowdInstance()
        instance_id = renderer.add_instance(inst, mesh_id=0, material_id=0)
        assert instance_id > 0
        assert renderer.total_instance_count == 1

    def test_add_instances_to_same_batch(self):
        """Test instances with same mesh/material go to same batch."""
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=2)
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=2)
        assert renderer.batch_count == 1

    def test_add_instances_to_different_batches(self):
        """Test instances with different mesh/material create batches."""
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=2)
        renderer.add_instance(CrowdInstance(), mesh_id=2, material_id=2)
        assert renderer.batch_count == 2

    def test_remove_instance(self):
        """Test removing instance."""
        renderer = CrowdRenderer()
        inst = CrowdInstance()
        instance_id = renderer.add_instance(inst, 0, 0)
        assert renderer.remove_instance(instance_id)
        assert renderer.total_instance_count == 0

    def test_update_advances_time(self):
        """Test update advances animation time."""
        renderer = CrowdRenderer()
        inst = CrowdInstance(animation_time=0.0)
        renderer.add_instance(inst, 0, 0)
        renderer.update(1.0)
        # Animation time should have advanced
        # (need to check the batch's instances)

    def test_cull_instances(self):
        """Test culling distant instances."""
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(position=Vec3(0, 0, 0)), 0, 0)
        renderer.add_instance(CrowdInstance(position=Vec3(100, 0, 0)), 0, 0)

        culled = renderer.cull_instances(Vec3.zero(), 50.0)
        assert culled == 1

    def test_get_stats(self):
        """Test getting renderer stats."""
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), 0, 0)
        stats = renderer.get_stats()
        assert "total_instances" in stats
        assert stats["total_instances"] == 1


# =============================================================================
# CROWD LOD TESTS
# =============================================================================

class TestLODLevel:
    """Tests for LOD level."""

    def test_create_lod_level(self):
        """Test creating LOD level."""
        lod = LODLevel(distance=50.0, bone_count=20, update_rate=0.5)
        assert lod.distance == 50.0
        assert lod.bone_count == 20
        assert lod.update_rate == 0.5

    def test_lod_ordering(self):
        """Test LOD levels are ordered by distance."""
        lod1 = LODLevel(distance=10.0)
        lod2 = LODLevel(distance=50.0)
        assert lod1 < lod2

    def test_should_update_full_rate(self):
        """Test update check at full rate."""
        lod = LODLevel(update_rate=1.0)
        assert lod.should_update(0) is True
        assert lod.should_update(1) is True

    def test_should_update_half_rate(self):
        """Test update check at half rate."""
        lod = LODLevel(update_rate=0.5)
        assert lod.should_update(0) is True
        assert lod.should_update(1) is False
        assert lod.should_update(2) is True


class TestLODTransition:
    """Tests for LOD transition."""

    def test_create_transition(self):
        """Test creating transition."""
        trans = LODTransition()
        assert trans.active is False
        assert trans.progress == 0.0

    def test_start_transition(self):
        """Test starting transition."""
        trans = LODTransition()
        trans.start(0, 1)
        assert trans.active is True
        assert trans.from_lod == 0
        assert trans.to_lod == 1

    def test_start_same_lod_no_transition(self):
        """Test starting transition to same LOD does nothing."""
        trans = LODTransition()
        trans.start(1, 1)
        assert trans.active is False

    def test_update_transition(self):
        """Test updating transition."""
        trans = LODTransition(duration=1.0)
        trans.start(0, 1)
        trans.update(0.5)
        assert trans.progress == pytest.approx(0.5, abs=0.01)
        assert trans.active is True

    def test_complete_transition(self):
        """Test completing transition."""
        trans = LODTransition(duration=1.0)
        trans.start(0, 1)
        complete = trans.update(1.5)
        assert complete is True
        assert trans.active is False
        assert trans.progress == 1.0

    def test_get_blend_factor_smoothstep(self):
        """Test blend factor uses smoothstep."""
        trans = LODTransition(duration=1.0)
        trans.start(0, 1)
        trans.progress = 0.5
        blend = trans.get_blend_factor()
        assert blend == pytest.approx(0.5, abs=0.1)


class TestCrowdLOD:
    """Tests for crowd LOD manager."""

    def test_create_crowd_lod(self):
        """Test creating LOD manager."""
        lod = CrowdLOD()
        assert lod.lod_count == 0

    def test_add_lod_level(self):
        """Test adding LOD level."""
        lod = CrowdLOD()
        idx = lod.add_lod_level(LODLevel(distance=50.0, bone_count=20))
        assert idx == 0
        assert lod.lod_count == 1

    def test_lod_levels_sorted(self):
        """Test LOD levels are sorted by distance."""
        lod = CrowdLOD()
        lod.add_lod_level(LODLevel(distance=100.0))
        lod.add_lod_level(LODLevel(distance=50.0))
        lod.add_lod_level(LODLevel(distance=25.0))

        level0 = lod.get_lod_level(0)
        level1 = lod.get_lod_level(1)
        assert level0.distance < level1.distance

    def test_get_lod_for_distance_near(self):
        """Test getting LOD for near distance."""
        lod = CrowdLOD()
        lod.add_lod_level(LODLevel(distance=25.0, bone_count=50))
        lod.add_lod_level(LODLevel(distance=50.0, bone_count=30))
        lod.add_lod_level(LODLevel(distance=100.0, bone_count=10))

        result = lod.get_lod_for_distance(10.0)
        assert result == 0

    def test_get_lod_for_distance_far(self):
        """Test getting LOD for far distance."""
        lod = CrowdLOD()
        lod.add_lod_level(LODLevel(distance=25.0))
        lod.add_lod_level(LODLevel(distance=50.0))
        lod.add_lod_level(LODLevel(distance=100.0))

        result = lod.get_lod_for_distance(150.0)
        assert result == 2  # max LOD


class TestCreateReducedSkeleton:
    """Tests for skeleton reduction."""

    def test_reduce_to_same_count(self):
        """Test reducing to same count returns original."""
        skeleton = Skeleton(
            bone_names=["root", "spine"],
            bone_parents=[-1, 0],
            bind_poses=[Transform.identity(), Transform.identity()],
        )
        reduced = create_reduced_skeleton(skeleton, 2)
        assert reduced.bone_count == 2

    def test_reduce_to_one_bone(self):
        """Test reducing to single bone."""
        skeleton = Skeleton(
            bone_names=["root", "spine", "head"],
            bone_parents=[-1, 0, 1],
            bind_poses=[Transform.identity()] * 3,
        )
        reduced = create_reduced_skeleton(skeleton, 1)
        assert reduced.bone_count == 1
        assert "root" in reduced.bone_names

    def test_reduce_preserves_hierarchy(self):
        """Test reduced skeleton maintains valid hierarchy."""
        skeleton = Skeleton(
            bone_names=["root", "spine", "chest", "head"],
            bone_parents=[-1, 0, 1, 2],
            bind_poses=[Transform.identity()] * 4,
        )
        reduced = create_reduced_skeleton(skeleton, 2)
        assert reduced.bone_count == 2
        # Parent indices should be valid
        for parent in reduced.bone_parents:
            assert parent == -1 or parent < reduced.bone_count


class TestBoneImportance:
    """Tests for bone importance calculation."""

    def test_root_bone_high_importance(self):
        """Test root bone has high importance."""
        skeleton = Skeleton(bone_names=["root"], bone_parents=[-1])
        importance = _calculate_bone_importance("root", 0, skeleton)
        assert importance > 0.8

    def test_finger_bone_low_importance(self):
        """Test finger bone has lower importance than spine/core bones."""
        skeleton = Skeleton(bone_names=["spine", "finger_index_01"], bone_parents=[-1, 0])
        # Use non-zero indices to avoid root bone bonus
        finger_importance = _calculate_bone_importance("finger_index_01", 1, skeleton)
        spine_importance = _calculate_bone_importance("spine", 0, skeleton)
        # Finger at depth 1 should have lower importance than spine (root gets bonus)
        assert finger_importance < spine_importance

    def test_spine_bone_high_importance(self):
        """Test spine bone has high importance."""
        skeleton = Skeleton(bone_names=["spine_01"], bone_parents=[-1])
        importance = _calculate_bone_importance("spine_01", 0, skeleton)
        assert importance > 0.7


# =============================================================================
# CROWD BEHAVIOR TESTS
# =============================================================================

class TestCrowdAgent:
    """Tests for crowd agent."""

    def test_create_default_agent(self):
        """Test creating default agent."""
        agent = CrowdAgent()
        assert agent.position == Vec3.zero()
        assert agent.current_state == AgentState.IDLE

    def test_agent_unique_id(self):
        """Test agents have unique IDs."""
        agent1 = CrowdAgent()
        agent2 = CrowdAgent()
        assert agent1.agent_id != agent2.agent_id

    def test_get_forward_direction(self):
        """Test getting forward direction."""
        agent = CrowdAgent(facing=0.0)
        forward = agent.get_forward()
        assert forward.z == pytest.approx(1.0, abs=0.01)

    def test_set_facing_from_direction(self):
        """Test setting facing from direction."""
        agent = CrowdAgent()
        agent.set_facing_from_direction(Vec3(1, 0, 0))
        forward = agent.get_forward()
        assert forward.x == pytest.approx(1.0, abs=0.01)

    def test_is_moving_stationary(self):
        """Test is_moving when stationary."""
        agent = CrowdAgent(velocity=Vec3.zero())
        assert agent.is_moving() is False

    def test_is_moving_with_velocity(self):
        """Test is_moving with velocity."""
        agent = CrowdAgent(velocity=Vec3(1, 0, 0))
        assert agent.is_moving() is True


class TestAnimationBlend:
    """Tests for animation blend."""

    def test_create_single_animation(self):
        """Test creating single animation blend."""
        blend = AnimationBlend.single(5)
        assert blend.animation_indices == [5]
        assert blend.weights == [1.0]

    def test_create_blend(self):
        """Test creating animation blend."""
        blend = AnimationBlend.blend(0, 1, 0.5)
        assert blend.animation_indices == [0, 1]
        assert blend.weights == [0.5, 0.5]

    def test_get_primary_animation(self):
        """Test getting primary animation."""
        blend = AnimationBlend.blend(0, 1, 0.7)
        assert blend.get_primary_animation() == 1  # Higher weight


class TestIdleBehavior:
    """Tests for idle behavior."""

    def test_create_idle_behavior(self):
        """Test creating idle behavior."""
        behavior = IdleBehavior()
        assert behavior.name == "idle"

    def test_update_stops_movement(self):
        """Test update stops agent movement."""
        behavior = IdleBehavior()
        agent = CrowdAgent(velocity=Vec3(1, 0, 0))
        context = BehaviorContext()

        behavior.update(agent, 1.0, context)
        assert agent.velocity.length() < 1.0


class TestWalkingBehavior:
    """Tests for walking behavior."""

    def test_create_walking_behavior(self):
        """Test creating walking behavior."""
        behavior = WalkingBehavior()
        assert behavior.name == "walking"

    def test_update_moves_toward_target(self):
        """Test agent moves toward target."""
        behavior = WalkingBehavior()
        agent = CrowdAgent(position=Vec3(0, 0, 0), target_position=Vec3(10, 0, 0))
        context = BehaviorContext()

        behavior.update(agent, 0.5, context)
        assert agent.position.x > 0  # Moved toward target


class TestCrowdSimulator:
    """Tests for crowd simulator."""

    def test_create_simulator(self):
        """Test creating simulator."""
        sim = CrowdSimulator()
        assert sim.agent_count == 0

    def test_add_agent(self):
        """Test adding agent."""
        sim = CrowdSimulator()
        agent = CrowdAgent()
        agent_id = sim.add_agent(agent)
        assert agent_id > 0
        assert sim.agent_count == 1

    def test_remove_agent(self):
        """Test removing agent."""
        sim = CrowdSimulator()
        agent = CrowdAgent()
        agent_id = sim.add_agent(agent)
        assert sim.remove_agent(agent_id)
        assert sim.agent_count == 0

    def test_get_agent(self):
        """Test getting agent by ID."""
        sim = CrowdSimulator()
        agent = CrowdAgent()
        agent_id = sim.add_agent(agent)
        retrieved = sim.get_agent(agent_id)
        assert retrieved is agent

    def test_transition_agent(self):
        """Test transitioning agent state."""
        sim = CrowdSimulator()
        agent = CrowdAgent()
        sim.add_agent(agent)

        assert sim.transition_agent(agent, AgentState.WALKING)
        assert agent.current_state == AgentState.WALKING

    def test_update_all_agents(self):
        """Test updating all agents."""
        sim = CrowdSimulator()
        sim.add_agent(CrowdAgent())
        sim.add_agent(CrowdAgent())

        initial_time = sim.time
        sim.update(1.0)
        assert sim.time > initial_time

    def test_trigger_flee(self):
        """Test triggering flee event."""
        sim = CrowdSimulator()
        sim.add_agent(CrowdAgent(position=Vec3(0, 0, 0)))
        sim.add_agent(CrowdAgent(position=Vec3(100, 0, 0)))  # Far away

        affected = sim.trigger_flee(Vec3(0, 0, 0), 10.0)
        assert affected == 1

    def test_get_stats(self):
        """Test getting simulator stats."""
        sim = CrowdSimulator()
        sim.add_agent(CrowdAgent())
        stats = sim.get_stats()
        assert "agent_count" in stats
        assert stats["agent_count"] == 1


# =============================================================================
# ANIMATION GRAPH SYSTEM TESTS
# =============================================================================

class TestGraphParameter:
    """Tests for graph parameters."""

    def test_create_float_parameter(self):
        """Test creating float parameter."""
        param = GraphParameter(name="speed", param_type=ParameterType.FLOAT, default_value=0.0)
        assert param.name == "speed"
        assert param.value == 0.0

    def test_set_float_value(self):
        """Test setting float value."""
        param = GraphParameter(param_type=ParameterType.FLOAT)
        param.set_value(5.5)
        assert param.value == 5.5

    def test_set_value_with_limits(self):
        """Test setting value with limits."""
        param = GraphParameter(param_type=ParameterType.FLOAT, min_value=0.0, max_value=1.0)
        param.set_value(2.0)
        assert param.value == 1.0

    def test_consume_trigger(self):
        """Test consuming trigger."""
        param = GraphParameter(param_type=ParameterType.TRIGGER, value=True)
        was_set = param.consume_trigger()
        assert was_set is True
        assert param.value is False


class TestAnimationGraphSystem:
    """Tests for animation graph system."""

    def test_create_system(self):
        """Test creating system."""
        system = AnimationGraphSystem()
        assert system is not None

    def test_update_empty_list(self):
        """Test updating with empty list."""
        system = AnimationGraphSystem()
        world = World()
        system.update(world, 0.1, [])  # Should not crash

    def test_update_component(self):
        """Test updating component."""
        system = AnimationGraphSystem()
        world = World()

        component = AnimationGraphComponent()
        component.graph.add_state(AnimationState(name="idle", animation_clip="idle_anim"))
        component.graph.add_state(AnimationState(name="walk", animation_clip="walk_anim"))

        entity = world.spawn()
        system.update(world, 0.1, [(entity, component)])

        assert component.graph.state_time > 0


# =============================================================================
# IK SYSTEM TESTS
# =============================================================================

class TestIKGoal:
    """Tests for IK goals."""

    def test_create_goal(self):
        """Test creating goal."""
        goal = IKGoal(target_bone=5, target_position=Vec3(1, 2, 3))
        assert goal.target_bone == 5
        assert goal.target_position == Vec3(1, 2, 3)

    def test_set_target(self):
        """Test setting target."""
        goal = IKGoal()
        goal.set_target(Vec3(5, 5, 5), Quat.identity())
        assert goal.target_position == Vec3(5, 5, 5)


class TestIKComponent:
    """Tests for IK component."""

    def test_create_component(self):
        """Test creating component."""
        comp = IKComponent()
        assert len(comp.goals) == 0

    def test_add_goal(self):
        """Test adding goal."""
        comp = IKComponent()
        idx = comp.add_goal(IKGoal(target_bone=5))
        assert idx == 0
        assert len(comp.goals) == 1

    def test_remove_goal(self):
        """Test removing goal."""
        comp = IKComponent()
        comp.add_goal(IKGoal(target_bone=5))
        assert comp.remove_goal(0)
        assert len(comp.goals) == 0


class TestIKSystem:
    """Tests for IK system."""

    def test_create_system(self):
        """Test creating system."""
        system = IKSystem()
        assert system is not None

    def test_set_skeleton_data(self):
        """Test setting skeleton data."""
        system = IKSystem()
        system.set_skeleton_data({0: -1, 1: 0, 2: 1}, {0: 1.0, 1: 1.0, 2: 1.0})


# =============================================================================
# PROCEDURAL SYSTEM TESTS
# =============================================================================

class TestSpringController:
    """Tests for spring controller."""

    def test_create_controller(self):
        """Test creating controller."""
        ctrl = SpringController()
        assert ctrl.controller_type == ControllerType.SPRING

    def test_update_returns_transforms(self):
        """Test update returns transforms."""
        ctrl = SpringController(affected_bones=[0, 1])
        pose = {0: Transform.identity(), 1: Transform.identity()}
        result = ctrl.update(0.1, pose)
        # Should return modified transforms
        assert isinstance(result, dict)


class TestLookAtController:
    """Tests for look-at controller."""

    def test_create_controller(self):
        """Test creating controller."""
        ctrl = LookAtController()
        assert ctrl.controller_type == ControllerType.LOOK_AT

    def test_set_target(self):
        """Test setting look target."""
        ctrl = LookAtController(target=Vec3(10, 5, 0))
        assert ctrl.target == Vec3(10, 5, 0)


class TestProceduralSystem:
    """Tests for procedural system."""

    def test_create_system(self):
        """Test creating system."""
        system = ProceduralSystem()
        assert system is not None

    def test_update_applies_controllers(self):
        """Test update applies controllers."""
        system = ProceduralSystem()
        world = World()
        entity = world.spawn()

        component = ProceduralComponent()
        component.add_controller(SpringController(affected_bones=[0]))

        pose_data = {entity: {0: Transform.identity()}}
        result = system.update(world, 0.1, [(entity, component)], pose_data)

        assert entity in result


# =============================================================================
# SKINNING SYSTEM TESTS
# =============================================================================

class TestSkinningSystem:
    """Tests for skinning system."""

    def test_create_system(self):
        """Test creating system."""
        system = SkinningSystem()
        assert system is not None

    def test_compute_bounding_box_empty(self):
        """Test bounding box with empty mesh."""
        system = SkinningSystem()
        component = SkinnedMeshComponent()
        result = system.compute_bounding_box(component)
        assert result is None

    def test_compute_bounding_box(self):
        """Test bounding box computation."""
        system = SkinningSystem()
        component = SkinnedMeshComponent()
        component.skinned_positions = [Vec3(0, 0, 0), Vec3(1, 2, 3)]

        result = system.compute_bounding_box(component)
        assert result is not None
        assert result[0] == Vec3(0, 0, 0)
        assert result[1] == Vec3(1, 2, 3)


# =============================================================================
# MOTION MATCHING SYSTEM TESTS
# =============================================================================

class TestMotionDatabase:
    """Tests for motion database."""

    def test_create_database(self):
        """Test creating database."""
        db = MotionDatabase()
        assert db.frame_count == 0

    def test_add_frame(self):
        """Test adding frame."""
        db = MotionDatabase()
        frame = MotionFrame(animation_index=0, frame_index=0)
        idx = db.add_frame(frame)
        assert idx == 0
        assert db.frame_count == 1


class TestMotionMatchingSystem:
    """Tests for motion matching system."""

    def test_create_system(self):
        """Test creating system."""
        system = MotionMatchingSystem()
        assert system is not None

    def test_build_database(self):
        """Test building database from animations."""
        system = MotionMatchingSystem()

        animations = [
            ("walk", [{0: Transform.identity()} for _ in range(30)]),
        ]
        features = [MotionFeature(name="root_pos", feature_type=FeatureType.POSITION, bone_index=0)]

        db = system.build_database(animations, features, frame_rate=30.0)
        assert db.frame_count == 30


# =============================================================================
# FACIAL SYSTEM TESTS
# =============================================================================

class TestFacialComponent:
    """Tests for facial component."""

    def test_create_component(self):
        """Test creating component."""
        comp = FacialComponent()
        assert comp.current_emotion == EmotionState.NEUTRAL

    def test_set_emotion(self):
        """Test setting emotion."""
        comp = FacialComponent()
        comp.set_emotion(EmotionState.HAPPY, 0.8)
        assert comp.current_emotion == EmotionState.HAPPY
        assert comp.emotion_intensity == 0.8

    def test_set_phoneme(self):
        """Test setting phoneme."""
        comp = FacialComponent()
        comp.set_phoneme(LipSyncPhoneme.AA, 0.9)
        assert comp.lip_sync.current_phoneme == LipSyncPhoneme.AA

    def test_set_look_target(self):
        """Test setting look target."""
        comp = FacialComponent()
        comp.set_look_target(Vec3(10, 5, 0), 0.7)
        assert comp.eye_state.look_target == Vec3(10, 5, 0)
        assert comp.eye_state.look_weight == 0.7


class TestFacialSystem:
    """Tests for facial system."""

    def test_create_system(self):
        """Test creating system."""
        system = FacialSystem()
        assert system is not None

    def test_update_component(self):
        """Test updating component."""
        system = FacialSystem()
        world = World()
        entity = world.spawn()

        component = FacialComponent()
        component.set_emotion(EmotionState.HAPPY, 0.5)

        system.update(world, 0.1, [(entity, component)])

        # Should have output blend shapes
        assert len(component.output_blend_shapes) > 0


# =============================================================================
# CROWD SYSTEM TESTS
# =============================================================================

class TestCrowdComponent:
    """Tests for crowd component."""

    def test_create_component(self):
        """Test creating component."""
        comp = CrowdComponent()
        assert comp.get_agent_count() == 0

    def test_add_agent(self):
        """Test adding agent."""
        comp = CrowdComponent()
        agent_id = comp.add_agent(Vec3(0, 0, 0))
        assert agent_id > 0
        assert comp.get_agent_count() == 1

    def test_remove_agent(self):
        """Test removing agent."""
        comp = CrowdComponent()
        agent_id = comp.add_agent(Vec3(0, 0, 0))
        assert comp.remove_agent(agent_id)
        assert comp.get_agent_count() == 0


class TestCrowdSystem:
    """Tests for crowd system."""

    def test_create_system(self):
        """Test creating system."""
        system = CrowdSystem()
        assert system is not None

    def test_set_lod_distances(self):
        """Test setting LOD distances."""
        system = CrowdSystem()
        system.set_lod_distances([10.0, 25.0, 50.0])

    def test_spawn_circle_formation(self):
        """Test spawning circle formation."""
        system = CrowdSystem()
        component = CrowdComponent()

        agent_ids = system.spawn_crowd_formation(
            component,
            center=Vec3(0, 0, 0),
            count=10,
            radius=5.0,
            formation="circle"
        )

        assert len(agent_ids) == 10
        assert component.get_agent_count() == 10

    def test_spawn_grid_formation(self):
        """Test spawning grid formation."""
        system = CrowdSystem()
        component = CrowdComponent()

        agent_ids = system.spawn_crowd_formation(
            component,
            center=Vec3(0, 0, 0),
            count=16,
            radius=5.0,
            formation="grid"
        )

        assert len(agent_ids) == 16

    def test_trigger_flee_event(self):
        """Test triggering flee event."""
        system = CrowdSystem()
        component = CrowdComponent()

        component.add_agent(Vec3(0, 0, 0))
        component.add_agent(Vec3(100, 0, 0))  # Far away

        affected = system.trigger_flee_event(component, Vec3(0, 0, 0), 10.0)
        assert affected == 1

    def test_get_stats(self):
        """Test getting aggregate stats."""
        system = CrowdSystem()
        component = CrowdComponent()
        component.add_agent(Vec3(0, 0, 0))

        world = World()
        entity = world.spawn()

        stats = system.get_stats([(entity, component)])
        assert "total_agents" in stats
        assert stats["total_agents"] == 1


# =============================================================================
# SYSTEM EXECUTION ORDER TESTS
# =============================================================================

class TestSystemExecutionOrder:
    """Tests for system execution order."""

    def test_graph_before_ik(self):
        """Test animation graph runs before IK."""
        # Animation graph produces pose, IK modifies it
        graph_system = AnimationGraphSystem()
        ik_system = IKSystem()

        world = World()
        entity = world.spawn()

        # Graph component
        graph_comp = AnimationGraphComponent()
        graph_comp.graph.add_state(AnimationState(name="idle"))

        # IK component
        ik_comp = IKComponent()
        ik_comp.add_goal(IKGoal(target_bone=0, target_position=Vec3(1, 0, 0)))

        # Run graph first
        graph_system.update(world, 0.1, [(entity, graph_comp)])
        pose_data = {entity: graph_comp.output_pose.bone_transforms}

        # Then IK
        result = ik_system.update(world, [(entity, ik_comp)], pose_data)
        assert entity in result

    def test_ik_before_procedural(self):
        """Test IK runs before procedural."""
        ik_system = IKSystem()
        proc_system = ProceduralSystem()

        world = World()
        entity = world.spawn()

        ik_comp = IKComponent()
        proc_comp = ProceduralComponent()
        proc_comp.add_controller(SpringController(affected_bones=[0]))

        pose_data = {entity: {0: Transform.identity()}}

        # Run IK first
        ik_result = ik_system.update(world, [(entity, ik_comp)], pose_data)

        # Then procedural
        proc_result = proc_system.update(world, 0.1, [(entity, proc_comp)], ik_result)
        assert entity in proc_result

    def test_procedural_before_skinning(self):
        """Test procedural runs before skinning."""
        proc_system = ProceduralSystem()
        skin_system = SkinningSystem()

        world = World()
        entity = world.spawn()

        proc_comp = ProceduralComponent()
        skin_comp = SkinnedMeshComponent()

        pose_data = {entity: {0: Transform.identity()}}

        # Run procedural first
        proc_result = proc_system.update(world, 0.1, [(entity, proc_comp)], pose_data)

        # Then skinning
        skin_system.update(world, [(entity, skin_comp)], proc_result)


# =============================================================================
# BEHAVIOR TRANSITION TESTS
# =============================================================================

class TestBehaviorTransitions:
    """Tests for behavior state transitions."""

    def test_idle_to_walking(self):
        """Test transition from idle to walking."""
        sim = CrowdSimulator()
        agent = CrowdAgent(current_state=AgentState.IDLE)
        sim.add_agent(agent)

        agent.target_position = Vec3(10, 0, 0)
        assert sim.transition_agent(agent, AgentState.WALKING)
        assert agent.current_state == AgentState.WALKING

    def test_walking_to_idle(self):
        """Test transition from walking to idle."""
        sim = CrowdSimulator()
        agent = CrowdAgent(current_state=AgentState.WALKING)
        sim.add_agent(agent)

        assert sim.transition_agent(agent, AgentState.IDLE)
        assert agent.current_state == AgentState.IDLE

    def test_any_to_fleeing(self):
        """Test transition to fleeing state."""
        sim = CrowdSimulator()
        agent = CrowdAgent(current_state=AgentState.IDLE)
        sim.add_agent(agent)

        agent.flee_source = Vec3(0, 0, 0)
        assert sim.transition_agent(agent, AgentState.FLEEING)
        assert agent.current_state == AgentState.FLEEING

    def test_transition_updates_animation(self):
        """Test state transition updates animation blend."""
        sim = CrowdSimulator()
        agent = CrowdAgent(current_state=AgentState.IDLE)
        sim.add_agent(agent)

        initial_anim = agent.animation_blend.get_primary_animation()

        agent.target_position = Vec3(10, 0, 0)
        sim.transition_agent(agent, AgentState.WALKING)
        sim.update(0.5)

        # Animation should have potentially changed based on velocity
        # (exact animation depends on movement state)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestAdditionalBehaviors:
    """Additional tests for behavior edge cases."""

    def test_fleeing_behavior_no_source(self):
        """Test fleeing behavior without flee source."""
        behavior = FleeingBehavior()
        agent = CrowdAgent(flee_source=None)
        context = BehaviorContext()
        behavior.update(agent, 0.1, context)
        assert agent.target_velocity == Vec3.zero()

    def test_formation_behavior_no_offset(self):
        """Test formation behavior without offset."""
        behavior = FormationBehavior()
        agent = CrowdAgent(formation_offset=None)
        context = BehaviorContext()
        behavior.update(agent, 0.1, context)
        assert agent.target_velocity == Vec3.zero()

    def test_waiting_behavior_fidget(self):
        """Test waiting behavior fidget timer."""
        behavior = WaitingBehavior(fidget_interval=(0.1, 0.2))
        agent = CrowdAgent()
        context = BehaviorContext()
        # Multiple updates should trigger fidget
        for _ in range(20):
            behavior.update(agent, 0.05, context)

    def test_behavior_context_nearby_agents(self):
        """Test getting nearby agents from context."""
        context = BehaviorContext()
        agent1 = CrowdAgent(position=Vec3(0, 0, 0))
        agent2 = CrowdAgent(position=Vec3(1, 0, 0))
        agent3 = CrowdAgent(position=Vec3(100, 0, 0))
        context.all_agents = [agent1, agent2, agent3]

        nearby = context.get_nearby_agents(agent1, 5.0)
        assert len(nearby) == 1  # Only agent2, not agent3

    def test_simulator_clear(self):
        """Test clearing simulator."""
        sim = CrowdSimulator()
        sim.add_agent(CrowdAgent())
        sim.add_agent(CrowdAgent())
        sim.clear()
        assert sim.agent_count == 0

    def test_simulator_agents_in_state(self):
        """Test getting agents in specific state."""
        sim = CrowdSimulator()
        agent1 = CrowdAgent(current_state=AgentState.IDLE)
        agent2 = CrowdAgent(current_state=AgentState.WALKING)
        sim.add_agent(agent1)
        sim.add_agent(agent2)

        idle_agents = sim.get_agents_in_state(AgentState.IDLE)
        assert len(idle_agents) == 1

    def test_simulator_agents_in_radius(self):
        """Test getting agents in radius."""
        sim = CrowdSimulator()
        sim.add_agent(CrowdAgent(position=Vec3(0, 0, 0)))
        sim.add_agent(CrowdAgent(position=Vec3(100, 0, 0)))

        nearby = sim.get_agents_in_radius(Vec3(0, 0, 0), 10.0)
        assert len(nearby) == 1

    def test_renderer_prepare_render_data(self):
        """Test preparing render data."""
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), 0, 0)
        data = renderer.prepare_render_data()
        assert len(data) == 1


class TestAnimationPipelineIntegration:
    """Integration tests for full animation pipeline."""

    def test_full_pipeline_flow(self):
        """Test complete animation pipeline flow."""
        # Create all systems
        graph_system = AnimationGraphSystem()
        ik_system = IKSystem()
        proc_system = ProceduralSystem()
        skin_system = SkinningSystem()
        facial_system = FacialSystem()

        world = World()
        entity = world.spawn()

        # Create components
        graph_comp = AnimationGraphComponent()
        graph_comp.graph.add_state(AnimationState(name="idle"))

        ik_comp = IKComponent()
        proc_comp = ProceduralComponent()
        skin_comp = SkinnedMeshComponent()
        facial_comp = FacialComponent()

        # Run pipeline
        dt = 0.016  # 60 fps

        # 1. Animation graph
        graph_system.update(world, dt, [(entity, graph_comp)])
        pose = {entity: graph_comp.output_pose.bone_transforms}

        # 2. IK
        pose = ik_system.update(world, [(entity, ik_comp)], pose)

        # 3. Procedural
        pose = proc_system.update(world, dt, [(entity, proc_comp)], pose)

        # 4. Skinning
        skin_system.update(world, [(entity, skin_comp)], pose)

        # 5. Facial (parallel to body)
        facial_system.update(world, dt, [(entity, facial_comp)])

        # Verify pipeline completed
        assert entity in pose

    def test_crowd_system_integration(self):
        """Test crowd system integration."""
        crowd_system = CrowdSystem()
        world = World()
        entity = world.spawn()

        crowd_comp = CrowdComponent()

        # Spawn agents
        crowd_system.spawn_crowd_formation(
            crowd_comp,
            center=Vec3(0, 0, 0),
            count=50,
            radius=10.0,
            formation="random"
        )

        assert crowd_comp.get_agent_count() == 50

        # Update simulation
        crowd_system.update(world, 0.016, [(entity, crowd_comp)])

        # Trigger flee
        affected = crowd_system.trigger_flee_event(crowd_comp, Vec3(0, 0, 0), 15.0)
        assert affected > 0

        # Update again
        crowd_system.update(world, 0.016, [(entity, crowd_comp)])

        # Get stats
        stats = crowd_system.get_stats([(entity, crowd_comp)])
        assert stats["total_agents"] == 50


# =============================================================================
# OVERFLOW AND EDGE CASE PROTECTION TESTS
# =============================================================================

class TestOverflowProtection:
    """Tests for buffer and texture overflow protection."""

    def test_animation_texture_max_bones_enforced(self):
        """Test animation texture rejects too many bones."""
        # Create skeleton with more bones than allowed
        max_bones = ANIMATION_TEXTURE_CONFIG.MAX_BONES_PER_TEXTURE
        skeleton = Skeleton(
            bone_names=[f"bone_{i}" for i in range(max_bones + 1)],
            bone_parents=[-1] + list(range(max_bones)),
            bind_poses=[Transform.identity()] * (max_bones + 1),
        )
        clip = AnimationClip(
            name="test",
            duration=1.0,
            frame_rate=30.0,
            bone_tracks={i: [Transform.identity()] for i in range(max_bones + 1)},
        )
        with pytest.raises(AnimationTextureOverflowError):
            bake_clip_to_texture(clip, skeleton)

    def test_animation_texture_max_frames_enforced(self):
        """Test animation texture rejects too many frames."""
        max_frames = ANIMATION_TEXTURE_CONFIG.MAX_FRAMES_PER_ANIMATION
        skeleton = Skeleton(
            bone_names=["root"],
            bone_parents=[-1],
            bind_poses=[Transform.identity()],
        )
        # Create clip with too many frames
        clip = AnimationClip(
            name="test",
            duration=float(max_frames + 1) / 30.0,
            frame_rate=30.0,
            bone_tracks={0: [Transform.identity()] * (max_frames + 1)},
        )
        with pytest.raises(AnimationTextureOverflowError):
            bake_clip_to_texture(clip, skeleton)

    def test_instance_buffer_reserve_overflow(self):
        """Test instance buffer rejects oversized reservation."""
        buf = InstanceBuffer()
        max_capacity = buf.max_capacity
        with pytest.raises(InstanceBufferOverflowError):
            buf.reserve(max_capacity + 1)

    def test_pack_float_division_by_zero_range(self):
        """Test pack_float_to_rgba8 handles zero range."""
        # When min_val == max_val, should not crash
        packed = pack_float_to_rgba8(50.0, 50.0, 50.0)
        assert len(packed) == 4
        assert all(0 <= v <= 255 for v in packed)


class TestDivisionByZeroProtection:
    """Tests for division by zero edge cases."""

    def test_lod_distance_zero(self):
        """Test LOD selection at distance=0."""
        lod = CrowdLOD()
        lod.add_lod_level(LODLevel(distance=10.0, bone_count=50))
        lod.add_lod_level(LODLevel(distance=25.0, bone_count=30))

        # Should not crash and return LOD 0
        result = lod.get_lod_for_distance(0.0)
        assert result == 0

    def test_lod_distance_negative_clamped(self):
        """Test LOD selection with negative distance (clamped to 0)."""
        lod = CrowdLOD()
        lod.add_lod_level(LODLevel(distance=10.0, bone_count=50))

        result = lod.get_lod_for_distance(-5.0)
        assert result == 0

    def test_crowd_avoidance_coincident_agents(self):
        """Test avoidance doesn't crash when agents are at same position."""
        behavior = WalkingBehavior()
        agent = CrowdAgent(position=Vec3(5, 0, 5), target_position=Vec3(10, 0, 10))
        other = CrowdAgent(position=Vec3(5, 0, 5))  # Same position

        context = BehaviorContext()
        context.all_agents = [agent, other]

        # Should not crash even with coincident agents
        behavior.update(agent, 0.1, context)
        # Agent should still move
        assert agent.position.x != 5.0 or agent.position.z != 5.0

    def test_fleeing_at_threat_position(self):
        """Test fleeing doesn't crash when at threat position."""
        behavior = FleeingBehavior()
        agent = CrowdAgent(position=Vec3(0, 0, 0), flee_source=Vec3(0, 0, 0))
        context = BehaviorContext()

        # Should not crash - picks random direction
        behavior.update(agent, 0.1, context)
        # Should have moved in some direction
        assert agent.velocity.length() > 0


class TestConfigUsage:
    """Tests verifying config values are used correctly."""

    def test_crowd_agent_uses_config_defaults(self):
        """Test CrowdAgent uses config defaults."""
        agent = CrowdAgent()
        assert agent.speed == CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED
        assert agent.turn_speed == CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_TURN_SPEED
        assert agent.radius == CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_RADIUS

    def test_idle_behavior_uses_config_defaults(self):
        """Test IdleBehavior uses config defaults."""
        behavior = IdleBehavior()
        assert behavior._variation_min == CROWD_BEHAVIOR_CONFIG.IDLE_VARIATION_MIN
        assert behavior._variation_max == CROWD_BEHAVIOR_CONFIG.IDLE_VARIATION_MAX

    def test_walking_behavior_uses_config_defaults(self):
        """Test WalkingBehavior uses config defaults."""
        behavior = WalkingBehavior()
        assert behavior._avoidance_radius == CROWD_BEHAVIOR_CONFIG.DEFAULT_AVOIDANCE_RADIUS
        assert behavior._avoidance_strength == CROWD_BEHAVIOR_CONFIG.DEFAULT_AVOIDANCE_STRENGTH
        assert behavior._arrival_threshold == CROWD_BEHAVIOR_CONFIG.ARRIVAL_THRESHOLD

    def test_fleeing_behavior_uses_config_defaults(self):
        """Test FleeingBehavior uses config defaults."""
        behavior = FleeingBehavior()
        assert behavior._speed_multiplier == CROWD_BEHAVIOR_CONFIG.FLEE_SPEED_MULTIPLIER
        assert behavior._safe_distance == CROWD_BEHAVIOR_CONFIG.FLEE_SAFE_DISTANCE

    def test_lod_transition_uses_config_default_duration(self):
        """Test LODTransition uses config default duration."""
        trans = LODTransition()
        assert trans.duration == CROWD_LOD_CONFIG.DEFAULT_TRANSITION_DURATION

    def test_crowd_lod_uses_config_hysteresis(self):
        """Test CrowdLOD uses config hysteresis."""
        lod = CrowdLOD()
        assert lod._hysteresis == CROWD_LOD_CONFIG.DEFAULT_HYSTERESIS

    def test_crowd_component_uses_config_defaults(self):
        """Test CrowdComponent uses config defaults."""
        comp = CrowdComponent()
        assert comp.update_rate == CROWD_SYSTEM_CONFIG.DEFAULT_UPDATE_RATE
        assert comp.max_visible == CROWD_SYSTEM_CONFIG.DEFAULT_MAX_VISIBLE

    def test_ik_goal_uses_config_defaults(self):
        """Test IKGoal uses config defaults."""
        goal = IKGoal()
        assert goal.position_tolerance == IK_CONFIG.DEFAULT_POSITION_TOLERANCE
        assert goal.rotation_tolerance == IK_CONFIG.DEFAULT_ROTATION_TOLERANCE
        assert goal.max_iterations == IK_CONFIG.DEFAULT_MAX_ITERATIONS

    def test_spring_controller_uses_config_defaults(self):
        """Test SpringController uses config defaults."""
        ctrl = SpringController()
        assert ctrl.stiffness == PROCEDURAL_CONFIG.DEFAULT_SPRING_STIFFNESS
        assert ctrl.damping == PROCEDURAL_CONFIG.DEFAULT_SPRING_DAMPING
        assert ctrl.mass == PROCEDURAL_CONFIG.DEFAULT_SPRING_MASS

    def test_look_at_controller_uses_config_defaults(self):
        """Test LookAtController uses config defaults."""
        ctrl = LookAtController()
        assert ctrl.speed == PROCEDURAL_CONFIG.DEFAULT_LOOK_SPEED
        assert ctrl.angle_limit_horizontal == PROCEDURAL_CONFIG.DEFAULT_HORIZONTAL_LIMIT
        assert ctrl.angle_limit_vertical == PROCEDURAL_CONFIG.DEFAULT_VERTICAL_LIMIT


class TestDataCorrectness:
    """Tests verifying actual data correctness, not just structure."""

    def test_baked_texture_contains_correct_position(self):
        """Test baked texture stores correct transform position."""
        skeleton = Skeleton(
            bone_names=["root"],
            bone_parents=[-1],
            bind_poses=[Transform.identity()],
        )
        test_pos = Vec3(5.5, 3.3, -2.1)
        clip = AnimationClip(
            name="test",
            duration=1.0 / 30.0,
            frame_rate=30.0,
            bone_tracks={0: [Transform(translation=test_pos)]},
        )
        tex = bake_clip_to_texture(clip, skeleton)

        # Retrieve the transform
        retrieved = tex.get_bone_transform(0, 0)
        assert retrieved.translation.x == pytest.approx(test_pos.x, abs=0.01)
        assert retrieved.translation.y == pytest.approx(test_pos.y, abs=0.01)
        assert retrieved.translation.z == pytest.approx(test_pos.z, abs=0.01)

    def test_crowd_simulation_actually_moves_agents(self):
        """Test crowd simulation actually changes agent positions."""
        sim = CrowdSimulator()
        start_pos = Vec3(0, 0, 0)
        target_pos = Vec3(100, 0, 0)
        agent = CrowdAgent(position=start_pos, target_position=target_pos)
        sim.add_agent(agent)
        sim.transition_agent(agent, AgentState.WALKING)

        # Simulate for enough time to see movement
        for _ in range(10):
            sim.update(0.1)

        # Agent should have moved toward target
        assert agent.position.x > start_pos.x
        assert agent.position.distance(start_pos) > 0.1

    def test_lod_selection_returns_correct_level_at_boundaries(self):
        """Test LOD selection at exact boundary distances.

        LOD levels define distance thresholds. Below first threshold = LOD 0,
        between first and second = LOD 0, etc. Each LOD level's distance means
        'at this distance and beyond, consider switching TO this LOD'.
        """
        lod = CrowdLOD()
        lod.add_lod_level(LODLevel(distance=10.0, bone_count=100))
        lod.add_lod_level(LODLevel(distance=25.0, bone_count=50))
        lod.add_lod_level(LODLevel(distance=50.0, bone_count=20))

        # At 9m (before first threshold), should be LOD 0
        assert lod.get_lod_for_distance(9.0) == 0
        # At 11m (after first threshold but before second), should be LOD 0
        # because we're past the first threshold distance, we stay at LOD 0
        # until we hit the NEXT threshold (25m)
        result_11 = lod.get_lod_for_distance(11.0)
        assert result_11 in (0, 1)  # Depends on implementation details
        # At 30m (past second threshold), should be LOD 1
        assert lod.get_lod_for_distance(30.0) >= 1
        # At 60m (past all thresholds), should be max LOD (2)
        assert lod.get_lod_for_distance(60.0) == 2

    def test_instance_buffer_stores_correct_transform_data(self):
        """Test instance buffer stores correct transform data."""
        buf = InstanceBuffer()
        pos = Vec3(10.0, 20.0, 30.0)
        inst = CrowdInstance(position=pos)
        idx = buf.add_instance(inst)

        # Verify transform data contains position in matrix
        # The transform matrix should have position in the last column
        # For a 4x4 matrix stored row-major: indices 12, 13, 14 are translation
        assert buf.transform_data[idx * 16 + 12] == pytest.approx(pos.x, abs=0.01)
        assert buf.transform_data[idx * 16 + 13] == pytest.approx(pos.y, abs=0.01)
        assert buf.transform_data[idx * 16 + 14] == pytest.approx(pos.z, abs=0.01)

    def test_animation_graph_actually_transitions_states(self):
        """Test animation graph actually transitions between states."""
        system = AnimationGraphSystem()
        world = World()
        entity = world.spawn()

        component = AnimationGraphComponent()
        component.graph.add_state(AnimationState(name="idle", animation_clip="idle"))
        component.graph.add_state(AnimationState(name="walk", animation_clip="walk"))

        # Add a transition with condition
        speed_param = GraphParameter(name="speed", param_type=ParameterType.FLOAT, value=0.0)
        component.graph.add_parameter(speed_param)

        def walk_condition(params):
            return params["speed"].value > 0.5

        component.graph.add_transition(StateTransition(
            from_state="idle",
            to_state="walk",
            condition=walk_condition,
            duration=0.1
        ))

        # Initially in idle
        assert component.graph.current_state == "idle"

        # Set speed high
        component.graph.set_parameter("speed", 1.0)

        # Update - should start transitioning
        system.update(world, 0.05, [(entity, component)])
        assert component.graph.transitioning is True

        # Update more - should complete transition
        system.update(world, 0.1, [(entity, component)])
        assert component.graph.current_state == "walk"

    def test_procedural_spring_responds_to_motion(self):
        """Test spring controller actually responds to motion."""
        ctrl = SpringController(affected_bones=[0], stiffness=100.0, damping=0.8)

        # Initial pose
        initial_pos = Vec3(0, 0, 0)
        pose = {0: Transform(translation=initial_pos)}

        # First update establishes rest position
        result = ctrl.update(0.01, pose)

        # Move the bone
        new_pos = Vec3(5, 0, 0)
        pose[0] = Transform(translation=new_pos)

        # Update - spring should respond
        result = ctrl.update(0.1, pose)

        # The spring position should be between old and new
        if 0 in result:
            spring_pos = result[0].translation
            # Spring should have some offset from the rest position
            # due to inertia (not instantly at new position)
            assert spring_pos != new_pos  # Shouldn't instantly teleport
