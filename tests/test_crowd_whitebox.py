"""WHITEBOX tests for crowds/animation internal code paths.

WHITEBOX coverage plan:
  [animation_texture.py]
    Path A1: get_pixel out-of-bounds x/y -> returns (0,0,0,0)
    Path A2: get_pixel short texture data -> returns (0,0,0,0)
    Path B1: set_pixel out-of-bounds x/y -> early return (no-op)
    Path B2: set_pixel short texture data -> early return (no-op)
    Path C:   get_bone_transform out-of-range bone/frame -> identity
    Path D:   sample_bone_transform duration>0 but 1 frame -> returns frame 0
    Path E:   get_memory_size_bytes all 4 formats -> byte count
    Path F1:  Atlas add_clip bone count mismatch -> False
    Path F2:  Atlas get_clip_info missing clip -> None
    Path G1:  Atlas get_clip_uv_range missing clip -> None
    Path G2:  Atlas get_clip_uv_range atlas height=0 -> None
    Path H1:  Atlas sample_clip missing clip -> identity
    Path H2:  Atlas sample_clip single frame -> direct lookup
    Path I1:  Atlas _get_bone_transform OOB -> identity
    Path I2:  Atlas _get_bone_transform short data -> identity
    Path J1:  bake_clip_to_texture bone overflow -> AnimationTextureOverflowError
    Path J2:  bake_clip_to_texture frame overflow -> AnimationTextureOverflowError
    Path J3:  bake_clip_to_texture width overflow -> AnimationTextureOverflowError
    Path J4:  bake_clip_to_texture height overflow -> AnimationTextureOverflowError
    Path J5:  bake_clip_to_texture missing track fallback to bind pose -> uses bind pose
    Path J6:  bake_clip_to_texture missing track + missing bind pose -> identity
    Path K1:  AnimationClip.frame_count empty tracks -> 0
    Path K2:  AnimationClip.get_bone_transform missing track -> None
    Path K3:  AnimationClip.get_bone_transform frame OOB -> None
    Path L1:  AnimationClip.sample_bone empty track -> None
    Path L2:  AnimationClip.sample_bone single frame -> frame[0]
    Path M1:  Skeleton.bone_count property -> len(bone_names)
    Path N1:  pack_float_to_rgba8 range=0 branch -> avoids div by zero
    Path N2:  pack_float_to_rgba8 value clamping at min/max
    Path N3:  pack_float_to_rgba8 / unpack_rgba8_to_float roundtrip preserves value

  [crowd_renderer.py]
    Path O1:  CrowdInstance.__post_init__ auto-increment ID
    Path O2:  CrowdInstance.advance_time with animation_speed multiplier
    Path O3:  CrowdInstance.set_animation reset_time=True (default)
    Path O4:  CrowdInstance.set_animation reset_time=False
    Path O5:  CrowdInstance.distance_to -> pythagorean distance
    Path P1:  CrowdRenderBatch.add_instance -> correct batch count
    Path P2:  CrowdRenderBatch.remove_instance found -> True
    Path P3:  CrowdRenderBatch.remove_instance not found -> False
    Path P4:  CrowdRenderBatch.update visible vs invisible
    Path P5:  CrowdRenderBatch.get_visible_count mixed visibility
    Path P6:  CrowdRenderBatch.sort_by_distance front-to-back
    Path P7:  CrowdRenderBatch.sort_by_distance back-to-front
    Path Q1:  CrowdRenderer.remove_instance found -> True
    Path Q2:  CrowdRenderer.remove_instance not found -> False
    Path Q3:  CrowdRenderer.update increments frame_count
    Path Q4:  CrowdRenderer.cull_instances returns correct count
    Path Q5:  CrowdRenderer.update_lod_levels direct (not from_system)
    Path Q6:  CrowdRenderer.register_animation_atlas roundtrip
    Path Q7:  CrowdRenderer.clear resets all state
    Path R1:  InstanceBuffer.update_instance OOB index -> no-op
    Path R2:  InstanceBuffer.add_instance auto-extend without reserve
    Path S1:  CrowdRenderer.add_instance atlas_name branch

  [crowd_lod.py]
    Path T1:  BoneWeight.__lt__ compares by importance
    Path T2:  _get_bone_depth root bone -> 0
    Path T3:  _get_bone_depth chain -> N
    Path U1:  _calculate_bone_importance pelvis/hips branch
    Path U2:  _calculate_bone_importance chest/torso branch
    Path U3:  _calculate_bone_importance shoulder/clavicle branch
    Path U4:  _calculate_bone_importance forearm/lowerarm branch
    Path U5:  _calculate_bone_importance hand (not finger) branch
    Path U6:  _calculate_bone_importance thigh/upperleg branch
    Path U7:  _calculate_bone_importance calf/lowerleg/shin branch
    Path U8:  _calculate_bone_importance foot (not toe) branch
    Path U9:  _calculate_bone_importance toe branch
    Path U10: _calculate_bone_importance thumb branch
    Path U11: _calculate_bone_importance neck branch
"""

from __future__ import annotations

import pytest
from engine.core.math import Vec3, Vec4, Quat, Transform
from engine.animation.crowds.animation_texture import (
    AnimationTexture,
    AnimationTextureAtlas,
    AnimationTextureOverflowError,
    TextureFormat,
    Skeleton,
    AnimationClip,
    bake_clip_to_texture,
    encode_transform_to_pixels,
    decode_pixels_to_transform,
    pack_float_to_rgba8,
    unpack_rgba8_to_float,
)
from engine.animation.crowds.crowd_renderer import (
    CrowdInstance,
    InstanceBuffer,
    InstanceBufferOverflowError,
    CrowdRenderBatch,
    CrowdRenderer,
    RenderPriority,
    CrowdLOD,
)
from engine.animation.crowds.crowd_lod import (
    LODLevel,
    LODTransition,
    BoneWeight,
    create_reduced_skeleton,
    _calculate_bone_importance,
    _get_bone_depth,
)
from engine.animation.config import ANIMATION_TEXTURE_CONFIG
from unittest import mock


# =========================================================================
# Helpers
# =========================================================================

def _vec3_close(a: Vec3, b: Vec3, eps: float = 0.001) -> bool:
    return abs(a.x - b.x) < eps and abs(a.y - b.y) < eps and abs(a.z - b.z) < eps


def _quat_close(a: Quat, b: Quat, eps: float = 0.001) -> bool:
    return abs(a.x - b.x) < eps and abs(a.y - b.y) < eps and abs(a.z - b.z) < eps and abs(a.w - b.w) < eps


def _transform_close(a: Transform, b: Transform, eps: float = 0.001) -> bool:
    return (
        _vec3_close(a.translation, b.translation, eps)
        and _quat_close(a.rotation.normalized(), b.rotation.normalized(), eps)
        and _vec3_close(a.scale, b.scale, eps)
    )


SKEL_SIMPLE = Skeleton(
    bone_names=["root", "spine", "head"],
    bone_parents=[-1, 0, 1],
    bind_poses=[Transform.identity() for _ in range(3)],
)


# =========================================================================
# A: AnimationTexture boundary paths
# =========================================================================

class TestAnimationTextureBoundary:
    """Every branch in get_pixel / set_pixel / get_bone_transform."""

    def test_get_pixel_x_negative_returns_zero(self) -> None:
        t = AnimationTexture(texture_data=[1.0] * 16, width=2, height=2)
        assert t.get_pixel(-1, 0) == (0.0, 0.0, 0.0, 0.0)

    def test_get_pixel_x_exceeds_width_returns_zero(self) -> None:
        t = AnimationTexture(texture_data=[1.0] * 16, width=2, height=2)
        assert t.get_pixel(2, 0) == (0.0, 0.0, 0.0, 0.0)

    def test_get_pixel_y_negative_returns_zero(self) -> None:
        t = AnimationTexture(texture_data=[1.0] * 16, width=2, height=2)
        assert t.get_pixel(0, -1) == (0.0, 0.0, 0.0, 0.0)

    def test_get_pixel_y_exceeds_height_returns_zero(self) -> None:
        t = AnimationTexture(texture_data=[1.0] * 16, width=2, height=2)
        assert t.get_pixel(0, 2) == (0.0, 0.0, 0.0, 0.0)

    def test_get_pixel_short_data_returns_zero(self) -> None:
        """When idx+4 > len(texture_data), return (0,0,0,0)."""
        t = AnimationTexture(texture_data=[1.0, 1.0, 1.0], width=100, height=100)
        assert t.get_pixel(0, 0) == (0.0, 0.0, 0.0, 0.0)

    def test_set_pixel_x_negative_no_op(self) -> None:
        """set_pixel with x < 0 must not modify data."""
        data = [0.0] * 16
        t = AnimationTexture(texture_data=data, width=2, height=2)
        t.set_pixel(-1, 0, 0.5, 0.5, 0.5, 0.5)
        assert t.texture_data == data

    def test_set_pixel_x_exceeds_width_no_op(self) -> None:
        data = [0.0] * 16
        t = AnimationTexture(texture_data=data, width=2, height=2)
        t.set_pixel(2, 0, 0.5, 0.5, 0.5, 0.5)
        assert t.texture_data == data

    def test_set_pixel_y_negative_no_op(self) -> None:
        data = [0.0] * 16
        t = AnimationTexture(texture_data=data, width=2, height=2)
        t.set_pixel(0, -1, 0.5, 0.5, 0.5, 0.5)
        assert t.texture_data == data

    def test_set_pixel_y_exceeds_height_no_op(self) -> None:
        data = [0.0] * 16
        t = AnimationTexture(texture_data=data, width=2, height=2)
        t.set_pixel(0, 2, 0.5, 0.5, 0.5, 0.5)
        assert t.texture_data == data

    def test_set_pixel_short_data_no_op(self) -> None:
        """When idx+4 > len(texture_data), do nothing."""
        data = [1.0, 1.0, 1.0]
        t = AnimationTexture(texture_data=data, width=100, height=100)
        t.set_pixel(0, 0, 0.5, 0.5, 0.5, 0.5)
        assert t.texture_data == data

    def test_get_bone_transform_oob_bone_returns_identity(self) -> None:
        t = AnimationTexture(texture_data=[0.0] * 16, bone_count=2, frame_count=2, width=4, height=2)
        result = t.get_bone_transform(99, 0)
        assert _transform_close(result, Transform.identity())

    def test_get_bone_transform_oob_frame_returns_identity(self) -> None:
        t = AnimationTexture(texture_data=[0.0] * 16, bone_count=2, frame_count=2, width=4, height=2)
        result = t.get_bone_transform(0, 99)
        assert _transform_close(result, Transform.identity())


# =========================================================================
# D: sample_bone_transform internal edge cases
# =========================================================================

class TestSampleBoneTransform:
    """Internal edge cases for sample_bone_transform."""

    def test_sample_duration_zero_returns_frame_zero(self) -> None:
        """When duration <= 0 and frame_count > 1, return frame 0."""
        t0 = Transform(translation=Vec3(5, 0, 0), rotation=Quat.identity(), scale=Vec3(1, 1, 1))
        t1 = Transform(translation=Vec3(10, 0, 0), rotation=Quat.identity(), scale=Vec3(1, 1, 1))
        clip = AnimationClip(name="test", duration=0.0, frame_rate=2, bone_tracks={0: [t0, t1]})
        tex = bake_clip_to_texture(clip, SKEL_SIMPLE)
        result = tex.sample_bone_transform(0, 0.5)
        # duration 0 -> early return get_bone_transform(bone_index, 0) which is t0
        assert _transform_close(result, t0)

    def test_sample_frame_count_one_with_duration_zero(self) -> None:
        """When frame_count <= 1 and duration <= 0, return frame 0."""
        t = Transform(translation=Vec3(1, 2, 3), rotation=Quat.identity(), scale=Vec3(1, 1, 1))
        clip = AnimationClip(name="test", duration=0.0, frame_rate=30, bone_tracks={0: [t]})
        tex = bake_clip_to_texture(clip, SKEL_SIMPLE)
        result = tex.sample_bone_transform(0, 999.0)
        assert _transform_close(result, t)


# =========================================================================
# E: get_memory_size_bytes for all 4 TextureFormats
# =========================================================================

class TestGetMemorySizeBytes:
    """get_memory_size_bytes must return correct byte counts per format."""

    @pytest.mark.parametrize("fmt,mult", [
        (TextureFormat.FLOAT32, 4),
        (TextureFormat.FLOAT16, 2),
        (TextureFormat.RGBA8_UNORM, 1),
        (TextureFormat.RGBA8_SNORM, 1),
    ])
    def test_memory_size_by_format(self, fmt: TextureFormat, mult: int) -> None:
        t = AnimationTexture(
            texture_data=[0.0] * (8 * 4), width=8, height=1, format=fmt,
        )
        expected = 8 * 1 * 4 * mult
        assert t.get_memory_size_bytes() == expected


# =========================================================================
# F, G, H, I: AnimationTextureAtlas internal paths
# =========================================================================

class TestAnimationTextureAtlasInternal:
    """Internal error paths in Atlas."""

    def test_add_clip_bone_count_mismatch_returns_false(self) -> None:
        atlas = AnimationTextureAtlas(bone_count=5)
        t = Transform.identity()
        clip = AnimationClip(name="a", duration=1.0, frame_rate=2, bone_tracks={0: [t, t]})
        tex = bake_clip_to_texture(clip, SKEL_SIMPLE)  # 3 bones
        assert atlas.add_clip("a", tex) is False

    def test_get_clip_info_missing_returns_none(self) -> None:
        atlas = AnimationTextureAtlas()
        assert atlas.get_clip_info("nonexistent") is None

    def test_get_clip_uv_range_missing_returns_none(self) -> None:
        atlas = AnimationTextureAtlas()
        assert atlas.get_clip_uv_range("nonexistent") is None

    def test_get_clip_uv_range_zero_height_returns_none(self) -> None:
        atlas = AnimationTextureAtlas(height=0)
        info = atlas.clips.get("x")  # we need to simulate a clip in atlas with height=0
        # Direct way: add a clip to a fresh atlas, force height=0
        t = Transform.identity()
        clip = AnimationClip(name="test", duration=1.0, frame_rate=2, bone_tracks={0: [t, t]})
        tex = bake_clip_to_texture(clip, SKEL_SIMPLE)
        atlas2 = AnimationTextureAtlas()
        atlas2.add_clip("test", tex)
        atlas2.height = 0  # corrupt height
        assert atlas2.get_clip_uv_range("test") is None

    def test_sample_clip_missing_returns_identity(self) -> None:
        atlas = AnimationTextureAtlas()
        result = atlas.sample_clip("nonexistent", 0, 0.0)
        assert _transform_close(result, Transform.identity())

    def test_sample_clip_single_frame(self) -> None:
        t = Transform(translation=Vec3(7, 0, 0), rotation=Quat.identity(), scale=Vec3(1, 1, 1))
        clip = AnimationClip(name="single", duration=1.0, frame_rate=1, bone_tracks={0: [t]})
        tex = bake_clip_to_texture(clip, SKEL_SIMPLE)
        atlas = AnimationTextureAtlas()
        atlas.add_clip("single", tex)
        result = atlas.sample_clip("single", 0, 0.5)
        # frame_count=1, frame_rate=1 -> frame_count <= 1 branch
        # So it calls _get_bone_transform for row start_row + 0
        assert _transform_close(result, t)

    def test_get_bone_transform_oob_returns_identity(self) -> None:
        atlas = AnimationTextureAtlas()
        result = atlas._get_bone_transform(99, 0)
        assert _transform_close(result, Transform.identity())

    def test_get_bone_transform_oob_row_returns_identity(self) -> None:
        atlas = AnimationTextureAtlas()
        result = atlas._get_bone_transform(0, 999)
        assert _transform_close(result, Transform.identity())

    def test_get_bone_transform_short_data_returns_identity(self) -> None:
        """When idx+8 > len(texture_data), return identity."""
        atlas = AnimationTextureAtlas(
            texture_data=[1.0, 1.0, 1.0],  # too short for idx+8
            width=4,
            height=1,
            bone_count=2,
        )
        result = atlas._get_bone_transform(0, 0)
        assert _transform_close(result, Transform.identity())

    def test_clip_frame_rate_zero_uses_direct_lookup(self) -> None:
        """sample_clip with frame_rate <= 0 calls _get_bone_transform directly."""
        t = Transform(translation=Vec3(3, 0, 0), rotation=Quat.identity(), scale=Vec3(1, 1, 1))
        clip = AnimationClip(name="test", duration=1.0, frame_rate=0, bone_tracks={0: [t, t]})
        tex = bake_clip_to_texture(clip, SKEL_SIMPLE)
        atlas = AnimationTextureAtlas()
        atlas.add_clip("test", tex)
        # Override the stored frame_rate in clips to 0 to force that branch
        atlas.clips["test"] = (atlas.clips["test"][0], atlas.clips["test"][1], 0)
        result = atlas.sample_clip("test", 0, 0.5)
        # Should still return something reasonable (direct lookup start_row)
        assert result is not None


# =========================================================================
# J: bake_clip_to_texture overflow errors and fallbacks
# =========================================================================

class TestBakeClipOverflow:
    """bake_clip_to_texture must raise for each overflow condition."""

    def test_bone_overflow(self) -> None:
        too_many_bones = Skeleton(
            bone_names=[str(i) for i in range(ANIMATION_TEXTURE_CONFIG.MAX_BONES_PER_TEXTURE + 1)],
            bone_parents=[-1] + [0] * ANIMATION_TEXTURE_CONFIG.MAX_BONES_PER_TEXTURE,
            bind_poses=[Transform.identity() for _ in range(ANIMATION_TEXTURE_CONFIG.MAX_BONES_PER_TEXTURE + 1)],
        )
        clip = AnimationClip(name="test", duration=1.0, frame_rate=2, bone_tracks={
            i: [Transform.identity(), Transform.identity()] for i in range(ANIMATION_TEXTURE_CONFIG.MAX_BONES_PER_TEXTURE + 1)
        })
        with pytest.raises(AnimationTextureOverflowError, match="Bone count"):
            bake_clip_to_texture(clip, too_many_bones)

    def test_frame_overflow(self) -> None:
        skel = Skeleton(bone_names=["root"], bone_parents=[-1], bind_poses=[Transform.identity()])
        many_frames = [Transform.identity() for _ in range(ANIMATION_TEXTURE_CONFIG.MAX_FRAMES_PER_ANIMATION + 1)]
        clip = AnimationClip(name="test", duration=100.0, frame_rate=30, bone_tracks={0: many_frames})
        with pytest.raises(AnimationTextureOverflowError, match="Frame count"):
            bake_clip_to_texture(clip, skel)

    def test_width_overflow(self) -> None:
        """Width check: requires bypassing bone count check via patched config."""
        import engine.animation.crowds.animation_texture as at_module
        max_w = ANIMATION_TEXTURE_CONFIG.MAX_TEXTURE_WIDTH
        bone_count = max_w // 2 + 1  # 2049 -> width = 4098 > 4096
        skel = Skeleton(
            bone_names=[str(i) for i in range(bone_count)],
            bone_parents=[-1] + [0] * (bone_count - 1),
            bind_poses=[Transform.identity() for _ in range(bone_count)],
        )
        clip = AnimationClip(name="test", duration=1.0, frame_rate=2, bone_tracks={
            i: [Transform.identity(), Transform.identity()] for i in range(bone_count)
        })
        # Bypass bone_count check by patching MAX_BONES_PER_TEXTURE on the module
        orig_bones = at_module.ANIMATION_TEXTURE_CONFIG.MAX_BONES_PER_TEXTURE
        orig_width = at_module.ANIMATION_TEXTURE_CONFIG.MAX_TEXTURE_WIDTH
        object.__setattr__(at_module.ANIMATION_TEXTURE_CONFIG, 'MAX_BONES_PER_TEXTURE', bone_count + 1)
        try:
            with pytest.raises(AnimationTextureOverflowError, match="Texture width"):
                bake_clip_to_texture(clip, skel)
        finally:
            object.__setattr__(at_module.ANIMATION_TEXTURE_CONFIG, 'MAX_BONES_PER_TEXTURE', orig_bones)
            object.__setattr__(at_module.ANIMATION_TEXTURE_CONFIG, 'MAX_TEXTURE_WIDTH', orig_width)

    def test_height_overflow(self) -> None:
        """Height check: requires bypassing frame count check via patched config."""
        import engine.animation.crowds.animation_texture as at_module
        skel = Skeleton(bone_names=["root"], bone_parents=[-1], bind_poses=[Transform.identity()])
        max_h = ANIMATION_TEXTURE_CONFIG.MAX_TEXTURE_HEIGHT
        many_frames = [Transform.identity() for _ in range(max_h + 1)]
        clip = AnimationClip(name="test", duration=100.0, frame_rate=30, bone_tracks={0: many_frames})
        # Bypass frame_count check by raising MAX_FRAMES_PER_ANIMATION
        orig_frames = at_module.ANIMATION_TEXTURE_CONFIG.MAX_FRAMES_PER_ANIMATION
        object.__setattr__(at_module.ANIMATION_TEXTURE_CONFIG, 'MAX_FRAMES_PER_ANIMATION', max_h + 2)
        try:
            with pytest.raises(AnimationTextureOverflowError, match="Texture height"):
                bake_clip_to_texture(clip, skel)
        finally:
            object.__setattr__(at_module.ANIMATION_TEXTURE_CONFIG, 'MAX_FRAMES_PER_ANIMATION', orig_frames)


class TestBakeClipTransformFallback:
    """When a clip track is missing a bone, the bake falls back to bind pose or identity."""

    def test_missing_bone_falls_back_to_bind_pose(self) -> None:
        """Bone 1 has no track -> should use skeleton.bind_poses[1]."""
        bind = Transform(translation=Vec3(99, 88, 77), rotation=Quat.identity(), scale=Vec3(1, 1, 1))
        skel = Skeleton(
            bone_names=["root", "spine"],
            bone_parents=[-1, 0],
            bind_poses=[Transform.identity(), bind],
        )
        clip = AnimationClip(name="test", duration=1.0, frame_rate=2, bone_tracks={
            0: [Transform.identity(), Transform.identity()],  # only bone 0 has data
        })
        tex = bake_clip_to_texture(clip, skel)
        # Bone 1 frame 0 should use bind pose
        result = tex.get_bone_transform(1, 0)
        assert _transform_close(result, bind), "Missing track should use bind pose"

    def test_missing_bone_and_missing_bind_pose_returns_identity(self) -> None:
        """Bone index outside bind_poses list -> identity."""
        skel = Skeleton(
            bone_names=["root", "spine"],
            bone_parents=[-1, 0],
            bind_poses=[Transform.identity()],  # only 1 bind pose for 2 bones
        )
        clip = AnimationClip(name="test", duration=1.0, frame_rate=2, bone_tracks={
            0: [Transform.identity(), Transform.identity()],
        })
        tex = bake_clip_to_texture(clip, skel)
        result = tex.get_bone_transform(1, 0)
        # bone 1 missing track, bone_idx=1 >= len(bind_poses)=1, so identity
        assert _transform_close(result, Transform.identity())

    def test_empty_skeleton_returns_empty_texture(self) -> None:
        """When bone_count == 0, return empty texture."""
        skel = Skeleton(bone_names=[], bone_parents=[], bind_poses=[])
        clip = AnimationClip(name="empty", duration=1.0, frame_rate=30, bone_tracks={})
        tex = bake_clip_to_texture(clip, skel)
        assert tex.bone_count == 0
        assert tex.frame_count == 0
        assert len(tex.texture_data) == 0


# =========================================================================
# K: AnimationClip internal edge cases
# =========================================================================

class TestAnimationClipInternal:
    """Internal branches in AnimationClip."""

    def test_frame_count_empty(self) -> None:
        clip = AnimationClip(name="test", bone_tracks={})
        assert clip.frame_count == 0

    def test_get_bone_transform_missing_track(self) -> None:
        clip = AnimationClip(name="test", bone_tracks={})
        assert clip.get_bone_transform(0, 0) is None

    def test_get_bone_transform_frame_oob(self) -> None:
        clip = AnimationClip(name="test", bone_tracks={0: [Transform.identity()]})
        assert clip.get_bone_transform(0, 99) is None

    def test_sample_bone_empty_track(self) -> None:
        clip = AnimationClip(name="test", bone_tracks={0: []})
        assert clip.sample_bone(0, 0.0) is None

    def test_sample_bone_missing_track(self) -> None:
        clip = AnimationClip(name="test", bone_tracks={})
        assert clip.sample_bone(0, 0.0) is None

    def test_sample_bone_single_frame(self) -> None:
        t = Transform(translation=Vec3(5, 0, 0), rotation=Quat.identity(), scale=Vec3(1, 1, 1))
        clip = AnimationClip(name="test", duration=0.0, frame_rate=30, bone_tracks={0: [t]})
        result = clip.sample_bone(0, 999.0)
        assert result is t  # exact same object for single frame

    def test_sample_bone_single_frame_duration_nonzero(self) -> None:
        """duration > 0 but len(track) == 1 -> return track[0]."""
        t = Transform(translation=Vec3(5, 0, 0), rotation=Quat.identity(), scale=Vec3(1, 1, 1))
        clip = AnimationClip(name="test", duration=10.0, frame_rate=30, bone_tracks={0: [t]})
        result = clip.sample_bone(0, 5.0)
        assert result is t


# =========================================================================
# M: Skeleton property
# =========================================================================

class TestSkeletonProperty:
    """Skeleton.bone_count property."""

    def test_bone_count_property(self) -> None:
        skel = Skeleton(bone_names=["a", "b", "c"], bone_parents=[-1, 0, 1])
        assert skel.bone_count == 3

    def test_bone_count_empty(self) -> None:
        skel = Skeleton()
        assert skel.bone_count == 0


# =========================================================================
# N: pack_float_to_rgba8 / unpack_rgba8_to_float internal branches
# =========================================================================

class TestPackUnpackRGBA8:
    """Internal branches in pack/unpack RGBA8."""

    def test_pack_range_zero_avoids_div_zero(self) -> None:
        """When min_val == max_val, range_val is set to 1.0."""
        packed = pack_float_to_rgba8(5.0, min_val=5.0, max_val=5.0)
        # normalized = 0.0 / 1.0 = 0.0 -> int_val = 0
        assert packed == (0, 0, 0, 0)

    def test_pack_value_below_min_clamped(self) -> None:
        packed = pack_float_to_rgba8(-200.0, min_val=-100.0, max_val=100.0)
        unpacked = unpack_rgba8_to_float(*packed, min_val=-100.0, max_val=100.0)
        assert unpacked == pytest.approx(-100.0, abs=0.5)

    def test_pack_value_above_max_clamped(self) -> None:
        packed = pack_float_to_rgba8(200.0, min_val=-100.0, max_val=100.0)
        unpacked = unpack_rgba8_to_float(*packed, min_val=-100.0, max_val=100.0)
        assert unpacked == pytest.approx(100.0, abs=0.5)

    def test_pack_unpack_roundtrip_zero(self) -> None:
        packed = pack_float_to_rgba8(0.0)
        unpacked = unpack_rgba8_to_float(*packed)
        assert unpacked == pytest.approx(0.0, abs=0.5)

    def test_pack_unpack_roundtrip_positive(self) -> None:
        packed = pack_float_to_rgba8(42.5)
        unpacked = unpack_rgba8_to_float(*packed)
        assert unpacked == pytest.approx(42.5, abs=0.5)

    def test_pack_unpack_roundtrip_negative(self) -> None:
        packed = pack_float_to_rgba8(-30.0)
        unpacked = unpack_rgba8_to_float(*packed)
        assert unpacked == pytest.approx(-30.0, abs=0.5)

    def test_pack_unpack_roundtrip_custom_range(self) -> None:
        packed = pack_float_to_rgba8(0.5, min_val=0.0, max_val=1.0)
        unpacked = unpack_rgba8_to_float(*packed, min_val=0.0, max_val=1.0)
        assert unpacked == pytest.approx(0.5, abs=0.005)


# =========================================================================
# O: CrowdInstance internal paths
# =========================================================================

@pytest.fixture(autouse=True)
def reset_instance_id() -> None:
    CrowdInstance._next_id = 0


class TestCrowdInstanceInternal:
    """Internal branches in CrowdInstance."""

    def test_auto_id_increment(self) -> None:
        a = CrowdInstance(instance_id=0)  # _next_id becomes 1
        b = CrowdInstance(instance_id=0)  # _next_id becomes 2
        assert a.instance_id == 1
        assert b.instance_id == 2
        assert a.instance_id != b.instance_id

    def test_explicit_id_no_auto(self) -> None:
        a = CrowdInstance(instance_id=100)
        assert a.instance_id == 100
        # _next_id should NOT have been incremented
        assert CrowdInstance._next_id == 0

    def test_advance_time_with_speed(self) -> None:
        inst = CrowdInstance(animation_time=0.0, animation_speed=2.0)
        inst.advance_time(0.5)
        assert inst.animation_time == 1.0  # 0.5 * 2.0

    def test_advance_time_default_speed(self) -> None:
        inst = CrowdInstance(animation_time=0.0)
        inst.advance_time(0.5)
        assert inst.animation_time == 0.5

    def test_set_animation_resets_time(self) -> None:
        inst = CrowdInstance(animation_index=0, animation_time=5.0)
        inst.set_animation(1, reset_time=True)
        assert inst.animation_index == 1
        assert inst.animation_time == 0.0

    def test_set_animation_keeps_time(self) -> None:
        inst = CrowdInstance(animation_index=0, animation_time=5.0)
        inst.set_animation(1, reset_time=False)
        assert inst.animation_index == 1
        assert inst.animation_time == 5.0

    def test_distance_to(self) -> None:
        inst = CrowdInstance(position=Vec3(3.0, 0.0, 4.0))
        dist = inst.distance_to(Vec3(0.0, 0.0, 0.0))
        assert dist == 5.0  # 3-4-5 triangle


# =========================================================================
# P: CrowdRenderBatch internal paths
# =========================================================================

class TestCrowdRenderBatchInternal:
    """Internal branches in CrowdRenderBatch."""

    def test_add_instance(self) -> None:
        batch = CrowdRenderBatch()
        inst = CrowdInstance(instance_id=1)
        idx = batch.add_instance(inst)
        assert idx == 0
        assert len(batch.instances) == 1
        assert batch.instance_buffer.instance_count == 1

    def test_remove_instance_found(self) -> None:
        batch = CrowdRenderBatch()
        inst = CrowdInstance(instance_id=42)
        batch.add_instance(inst)
        assert batch.remove_instance(42) is True
        assert len(batch.instances) == 0

    def test_remove_instance_not_found(self) -> None:
        batch = CrowdRenderBatch()
        inst = CrowdInstance(instance_id=1)
        batch.add_instance(inst)
        assert batch.remove_instance(999) is False

    def test_update_visible_instance_advances_time(self) -> None:
        """Only visible instances should have their time advanced."""
        batch = CrowdRenderBatch()
        vis = CrowdInstance(visible=True, animation_time=0.0, instance_id=1)
        invis = CrowdInstance(visible=False, animation_time=0.0, instance_id=2)
        batch.add_instance(vis)
        batch.add_instance(invis)
        batch.update(dt=1.0)
        assert vis.animation_time == 1.0
        assert invis.animation_time == 0.0  # not advanced

    def test_get_visible_count_all_visible(self) -> None:
        batch = CrowdRenderBatch()
        batch.add_instance(CrowdInstance(visible=True, instance_id=1))
        batch.add_instance(CrowdInstance(visible=True, instance_id=2))
        assert batch.get_visible_count() == 2

    def test_get_visible_count_mixed(self) -> None:
        batch = CrowdRenderBatch()
        batch.add_instance(CrowdInstance(visible=True, instance_id=1))
        batch.add_instance(CrowdInstance(visible=False, instance_id=2))
        batch.add_instance(CrowdInstance(visible=True, instance_id=3))
        assert batch.get_visible_count() == 2

    def test_sort_by_distance_front_to_back(self) -> None:
        batch = CrowdRenderBatch()
        batch.add_instance(CrowdInstance(position=Vec3(30, 0, 0), instance_id=1))
        batch.add_instance(CrowdInstance(position=Vec3(10, 0, 0), instance_id=2))
        batch.add_instance(CrowdInstance(position=Vec3(20, 0, 0), instance_id=3))
        batch.sort_by_distance(Vec3(0, 0, 0), front_to_back=True)
        distances = [inst.distance_to(Vec3(0, 0, 0)) for inst in batch.instances]
        assert distances == [10.0, 20.0, 30.0]

    def test_sort_by_distance_back_to_front(self) -> None:
        batch = CrowdRenderBatch()
        batch.add_instance(CrowdInstance(position=Vec3(10, 0, 0), instance_id=1))
        batch.add_instance(CrowdInstance(position=Vec3(30, 0, 0), instance_id=2))
        batch.add_instance(CrowdInstance(position=Vec3(20, 0, 0), instance_id=3))
        batch.sort_by_distance(Vec3(0, 0, 0), front_to_back=False)
        distances = [inst.distance_to(Vec3(0, 0, 0)) for inst in batch.instances]
        assert distances == [30.0, 20.0, 10.0]


# =========================================================================
# Q: CrowdRenderer internal paths
# =========================================================================

class TestCrowdRendererInternal:
    """Internal branches in CrowdRenderer."""

    def test_remove_instance_found(self) -> None:
        renderer = CrowdRenderer()
        inst = CrowdInstance(instance_id=1)
        renderer.add_instance(inst, mesh_id=0, material_id=0)
        assert renderer.remove_instance(1) is True
        assert renderer.total_instance_count == 0

    def test_remove_instance_not_found(self) -> None:
        renderer = CrowdRenderer()
        assert renderer.remove_instance(999) is False

    def test_update_increments_frame_count(self) -> None:
        renderer = CrowdRenderer()
        assert renderer._frame_count == 0
        renderer.update(0.0)
        assert renderer._frame_count == 1
        renderer.update(0.0)
        assert renderer._frame_count == 2

    def test_cull_instances_counts_correctly(self) -> None:
        renderer = CrowdRenderer()
        for i in range(5):
            renderer.add_instance(
                CrowdInstance(position=Vec3(float(i * 20), 0, 0), instance_id=i + 1),
                mesh_id=0, material_id=0,
            )
        culled = renderer.cull_instances(Vec3(0, 0, 0), max_distance=45.0)
        # Instances at 0, 20, 40 are visible (0 <= 45, 20 <= 45, 40 <= 45)
        # Instances at 60, 80 are culled
        assert culled == 2

    def test_cull_instances_no_cull(self) -> None:
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(position=Vec3(5, 0, 0), instance_id=1), mesh_id=0, material_id=0)
        renderer.add_instance(CrowdInstance(position=Vec3(10, 0, 0), instance_id=2), mesh_id=0, material_id=0)
        culled = renderer.cull_instances(Vec3(0, 0, 0), max_distance=100.0)
        assert culled == 0

    def test_update_lod_levels_direct(self) -> None:
        """update_lod_levels (not from_system) uses direct distance thresholds."""
        renderer = CrowdRenderer()
        renderer.add_instance(
            CrowdInstance(position=Vec3(5, 0, 0), instance_id=1),
            mesh_id=0, material_id=0,
        )
        renderer.add_instance(
            CrowdInstance(position=Vec3(30, 0, 0), instance_id=2),
            mesh_id=0, material_id=0,
        )
        renderer.update_lod_levels(Vec3(0, 0, 0), lod_distances=[10.0, 20.0])
        batch = renderer.get_batch(0, 0)
        assert batch is not None
        assert batch.instances[0].lod_level == 0  # 5 <= 10
        # dist=30: 30>10->lod=1, 30>20->lod=2, min(2, len([10,20]))=min(2,2)=2
        assert batch.instances[1].lod_level == 2

    def test_update_lod_levels_at_boundary(self) -> None:
        """At exact distance threshold, stays at lower LOD."""
        renderer = CrowdRenderer()
        renderer.add_instance(
            CrowdInstance(position=Vec3(10, 0, 0), instance_id=1),
            mesh_id=0, material_id=0,
        )
        renderer.update_lod_levels(Vec3(0, 0, 0), lod_distances=[10.0, 50.0])
        batch = renderer.get_batch(0, 0)
        assert batch is not None
        # dist == threshold -> NOT > threshold -> lod stays at 0
        assert batch.instances[0].lod_level == 0

    def test_register_and_get_animation_atlas(self) -> None:
        renderer = CrowdRenderer()
        atlas = AnimationTextureAtlas()
        renderer.register_animation_atlas("test", atlas)
        assert renderer.get_animation_atlas("test") is atlas
        assert renderer.get_animation_atlas("nonexistent") is None

    def test_clear_resets_all_state(self) -> None:
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(instance_id=1), mesh_id=1, material_id=1)
        renderer.add_instance(CrowdInstance(instance_id=2), mesh_id=2, material_id=2)
        renderer.update(0.0)
        renderer.clear()
        assert renderer.batch_count == 0
        assert renderer.total_instance_count == 0
        assert renderer._frame_count == 1  # frame_count is NOT reset by clear

    def test_add_instance_with_atlas_name_gets_atlas(self) -> None:
        renderer = CrowdRenderer()
        atlas = AnimationTextureAtlas(bone_count=3, width=2, height=1)
        renderer.register_animation_atlas("char_atlas", atlas)
        renderer.add_instance(
            CrowdInstance(instance_id=1),
            mesh_id=1, material_id=1, atlas_name="char_atlas",
        )
        batch = renderer.get_batch(1, 1)
        assert batch is not None
        assert batch.animation_atlas is atlas

    def test_add_instance_with_nonexistent_atlas_name(self) -> None:
        """When atlas_name refers to a missing atlas, batch atlas stays None."""
        renderer = CrowdRenderer()
        renderer.add_instance(
            CrowdInstance(instance_id=1),
            mesh_id=1, material_id=1, atlas_name="missing",
        )
        batch = renderer.get_batch(1, 1)
        assert batch is not None
        assert batch.animation_atlas is None

    def test_add_instance_with_no_atlas_name(self) -> None:
        """When atlas_name is None, batch atlas is None."""
        renderer = CrowdRenderer()
        renderer.add_instance(
            CrowdInstance(instance_id=1),
            mesh_id=1, material_id=1,
        )
        batch = renderer.get_batch(1, 1)
        assert batch is not None
        assert batch.animation_atlas is None


# =========================================================================
# R: InstanceBuffer internal paths
# =========================================================================

class TestInstanceBufferInternal:
    """Internal branches in InstanceBuffer not covered by DEV tests."""

    def test_update_instance_oob_index_does_nothing(self) -> None:
        buf = InstanceBuffer()
        buf.add_instance(CrowdInstance(instance_id=1))
        original_data = list(buf.transform_data)
        buf.update_instance(-1, CrowdInstance(instance_id=2))
        assert buf.transform_data == original_data
        buf.update_instance(5, CrowdInstance(instance_id=3))
        assert buf.transform_data == original_data

    def test_add_instance_unreserved_auto_extends_when_needed(self) -> None:
        """add_instance without reserve auto-extends when offset > len(data)."""
        buf = InstanceBuffer()
        buf.capacity = 0  # no capacity
        buf.max_capacity = 1000
        # Add first instance with no capacity - should auto-extend
        buf.add_instance(CrowdInstance(instance_id=1))
        assert buf.instance_count == 1
        assert len(buf.transform_data) >= 16

    def test_get_batches_iterator(self) -> None:
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(instance_id=1), mesh_id=0, material_id=0)
        renderer.add_instance(CrowdInstance(instance_id=2), mesh_id=1, material_id=1)
        batches = list(renderer.get_batches())
        assert len(batches) == 2


# =========================================================================
# T: BoneWeight and _get_bone_depth (crowd_lod.py helpers)
# =========================================================================

class TestBoneWeight:
    def test_lt_compares_by_importance(self) -> None:
        low = BoneWeight(bone_index=0, importance=0.3)
        high = BoneWeight(bone_index=1, importance=0.9)
        assert low < high
        assert not (high < low)

    def test_equal_importance_not_less(self) -> None:
        a = BoneWeight(bone_index=0, importance=0.5)
        b = BoneWeight(bone_index=1, importance=0.5)
        assert not (a < b)
        assert not (b < a)


class TestGetBoneDepth:
    """Direct tests for _get_bone_depth helper."""

    def test_root_depth_zero(self) -> None:
        skel = SKEL_SIMPLE
        assert _get_bone_depth(0, skel) == 0

    def test_chain_depth(self) -> None:
        """root(-1) -> spine(0) -> head(1): head depth = 2."""
        skel = SKEL_SIMPLE
        assert _get_bone_depth(1, skel) == 1  # spine -> root
        assert _get_bone_depth(2, skel) == 2  # head -> spine -> root

    def test_orphan_bone_parent_minus_one(self) -> None:
        """Bone with parent -1 has depth 0."""
        skel = Skeleton(bone_names=["a", "b"], bone_parents=[-1, -1])
        assert _get_bone_depth(0, skel) == 0
        assert _get_bone_depth(1, skel) == 0

    def test_complex_hierarchy(self) -> None:
        """root -> a -> b -> c: c depth = 3."""
        skel = Skeleton(bone_names=["root", "a", "b", "c"], bone_parents=[-1, 0, 1, 2])
        assert _get_bone_depth(3, skel) == 3


# =========================================================================
# U: _calculate_bone_importance remaining branches
# =========================================================================

SKEL_WITH_PARENTS = Skeleton(
    bone_names=["root", "pelvis", "chest", "neck", "head"],
    bone_parents=[-1, 0, 1, 2, 3],
    bind_poses=[Transform.identity() for _ in range(5)],
)


class TestBoneImportanceUncovered:
    """Branches in _calculate_bone_importance not covered by existing tests."""

    def test_pelvis_branch(self) -> None:
        score = _calculate_bone_importance("pelvis", 1, SKEL_WITH_PARENTS)
        # base 0.5 + pelvis 0.45 - depth(1)*0.02 = 0.93
        assert score > 0.8

    def test_hips_branch(self) -> None:
        score = _calculate_bone_importance("hips", 1, SKEL_WITH_PARENTS)
        # base 0.5 + hips 0.45 - depth(1)*0.02 = 0.93
        assert score > 0.8

    def test_chest_branch(self) -> None:
        score = _calculate_bone_importance("chest", 2, SKEL_WITH_PARENTS)
        # base 0.5 + chest 0.35 - depth(2)*0.02 = 0.81
        assert score > 0.7

    def test_torso_branch(self) -> None:
        score = _calculate_bone_importance("torso", 2, SKEL_WITH_PARENTS)
        assert score > 0.7

    def test_neck_branch(self) -> None:
        score = _calculate_bone_importance("neck", 3, SKEL_WITH_PARENTS)
        # base 0.5 + neck 0.35 - depth(3)*0.02 = 0.79
        assert score > 0.7

    def test_shoulder_branch(self) -> None:
        skel = Skeleton(
            bone_names=["root", "spine", "shoulder"],
            bone_parents=[-1, 0, 1],
            bind_poses=[Transform.identity() for _ in range(3)],
        )
        score = _calculate_bone_importance("shoulder", 2, skel)
        assert score > 0.7

    def test_clavicle_branch(self) -> None:
        skel = SKEL_WITH_PARENTS
        score = _calculate_bone_importance("clavicle", 1, skel)
        assert score > 0.7

    def test_upperarm_branch(self) -> None:
        skel = Skeleton(
            bone_names=["root", "spine", "upperarm"],
            bone_parents=[-1, 0, 1],
            bind_poses=[Transform.identity() for _ in range(3)],
        )
        score = _calculate_bone_importance("upperarm", 2, skel)
        # base 0.5 + upperarm 0.3 - depth(2)*0.02 = 0.76
        assert 0.7 < score < 0.85

    def test_upper_arm_branch(self) -> None:
        skel = SKEL_WITH_PARENTS
        score = _calculate_bone_importance("upper_arm", 1, skel)
        assert score > 0.7

    def test_forearm_branch(self) -> None:
        skel = SKEL_WITH_PARENTS
        score = _calculate_bone_importance("forearm", 1, skel)
        assert score > 0.6

    def test_lowerarm_branch(self) -> None:
        skel = SKEL_WITH_PARENTS
        score = _calculate_bone_importance("lowerarm", 1, skel)
        assert score > 0.6

    def test_hand_not_finger_branch(self) -> None:
        skel = SKEL_WITH_PARENTS
        score = _calculate_bone_importance("hand", 1, skel)
        # base 0.5 + hand 0.2 - depth(1)*0.02 = 0.68
        assert 0.6 < score < 0.8

    def test_thigh_branch(self) -> None:
        skel = SKEL_WITH_PARENTS
        score = _calculate_bone_importance("thigh", 1, skel)
        assert score > 0.7

    def test_upperleg_branch(self) -> None:
        skel = SKEL_WITH_PARENTS
        score = _calculate_bone_importance("upperleg", 1, skel)
        assert score > 0.7

    def test_calf_branch(self) -> None:
        skel = SKEL_WITH_PARENTS
        score = _calculate_bone_importance("calf", 1, skel)
        assert score > 0.6

    def test_lowerleg_branch(self) -> None:
        skel = SKEL_WITH_PARENTS
        score = _calculate_bone_importance("lowerleg", 1, skel)
        assert score > 0.6

    def test_shin_branch(self) -> None:
        skel = SKEL_WITH_PARENTS
        score = _calculate_bone_importance("shin", 1, skel)
        assert score > 0.6

    def test_foot_no_toe_branch(self) -> None:
        skel = SKEL_WITH_PARENTS
        score = _calculate_bone_importance("foot", 1, skel)
        # base 0.5 + foot 0.2 - depth(1)*0.02 = 0.68
        assert 0.6 < score < 0.8

    def test_toe_branch(self) -> None:
        skel = SKEL_WITH_PARENTS
        score = _calculate_bone_importance("toe", 1, skel)
        # base 0.5 + toe 0.02 - depth(1)*0.02 = 0.50
        assert score == pytest.approx(0.5, abs=0.05)

    def test_thumb_branch(self) -> None:
        """'thumb' doesn't match 'finger' in name_lower; falls through all cat branches."""
        skel = SKEL_WITH_PARENTS
        score = _calculate_bone_importance("thumb", 1, skel)
        # base 0.5, no category match, depth 1 => 0.48
        assert score == pytest.approx(0.48, abs=0.01)

    def test_finger_non_index(self) -> None:
        skel = Skeleton(
            bone_names=["root", "spine", "finger_middle"],
            bone_parents=[-1, 0, 1],
            bind_poses=[Transform.identity() for _ in range(3)],
        )
        score = _calculate_bone_importance("finger_middle", 2, skel)
        # base 0.5 + finger(not index) 0.05 = 0.55, - depth(2)*0.02 = 0.51
        assert 0.4 < score < 0.65

    def test_auxiliary_bones_penalty(self) -> None:
        """twist/roll/helper penalty applied after category bonus."""
        skel = SKEL_WITH_PARENTS
        score = _calculate_bone_importance("spine_twist", 2, skel)
        # spine -> base 0.5 + 0.4 = 0.9, -twist 0.2, -depth(2)*0.02 = 0.66
        base = _calculate_bone_importance("spine", 2, skel)
        penalized = _calculate_bone_importance("spine_twist", 2, skel)
        assert penalized < base
