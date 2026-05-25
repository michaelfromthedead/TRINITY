"""T1.1 — Animation Texture Verification tests."""
from __future__ import annotations
import math
import pytest
from engine.core.math import Vec3, Vec4, Quat, Transform
from engine.animation.crowds.animation_texture import (
    encode_transform_to_pixels,
    decode_pixels_to_transform,
    AnimationTexture,
    AnimationTextureAtlas,
    TextureFormat,
    Skeleton,
    AnimationClip,
    bake_clip_to_texture,
)


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


# ---- Round-trip encode/decode ----

def test_transform_roundtrip_identity():
    original = Transform.identity()
    p1, p2 = encode_transform_to_pixels(original)
    decoded = decode_pixels_to_transform(p1, p2)
    assert _transform_close(original, decoded, eps=0.001)


def test_transform_roundtrip_with_translation():
    original = Transform(translation=Vec3(1.5, -2.3, 7.8), rotation=Quat.identity(), scale=Vec3(1, 1, 1))
    p1, p2 = encode_transform_to_pixels(original)
    decoded = decode_pixels_to_transform(p1, p2)
    assert _vec3_close(decoded.translation, original.translation, eps=0.001)


def test_transform_roundtrip_with_rotation():
    rot = Quat(0.0, 0.7071, 0.0, 0.7071).normalized()  # 90-degree Y rotation
    original = Transform(translation=Vec3(0, 0, 0), rotation=rot, scale=Vec3(1, 1, 1))
    p1, p2 = encode_transform_to_pixels(original)
    decoded = decode_pixels_to_transform(p1, p2)
    assert _quat_close(decoded.rotation, rot, eps=0.001)


def test_transform_roundtrip_nonuniform_scale_gets_averaged():
    """Non-uniform scale is averaged during encode — verify roundtrip preserves the average."""
    original = Transform(translation=Vec3(0, 0, 0), rotation=Quat.identity(), scale=Vec3(2.0, 3.0, 4.0))
    p1, p2 = encode_transform_to_pixels(original)
    decoded = decode_pixels_to_transform(p1, p2)
    avg = (2.0 + 3.0 + 4.0) / 3.0
    assert abs(decoded.scale.x - avg) < 0.001


# ---- Interpolation ----

def test_interpolation_identity():
    """Two identical identity frames should produce identity at any blend."""
    t = Transform.identity()
    clip = AnimationClip(name="test", duration=1.0, frame_rate=2, bone_tracks={0: [t, t]})
    skeleton = Skeleton(bone_names=["root"], bone_parents=[-1], bind_poses=[Transform.identity()])
    tex = bake_clip_to_texture(clip, skeleton)
    result = tex.sample_bone_transform(0, 0.5)
    assert _transform_close(result, Transform.identity())


def test_interpolation_midpoint():
    """At t=0.5 of a 2-frame animation, should get the lerp midpoint."""
    t0 = Transform(translation=Vec3(0, 0, 0), rotation=Quat.identity(), scale=Vec3(1, 1, 1))
    t1 = Transform(translation=Vec3(10, 0, 0), rotation=Quat.identity(), scale=Vec3(1, 1, 1))
    clip = AnimationClip(name="test", duration=1.0, frame_rate=2, bone_tracks={0: [t0, t1]})
    skeleton = Skeleton(bone_names=["root"], bone_parents=[-1], bind_poses=[Transform.identity()])
    tex = bake_clip_to_texture(clip, skeleton)
    result = tex.sample_bone_transform(0, 0.5)
    assert abs(result.translation.x - 5.0) < 0.01


def test_interpolation_monotonic():
    """Interpolation between monotonic keyframes should produce monotonic results."""
    positions = [Vec3(float(i), 0, 0) for i in range(5)]
    t0 = Transform(translation=Vec3(0, 0, 0), rotation=Quat.identity(), scale=Vec3(1, 1, 1))
    clip = AnimationClip(name="test", duration=2.0, frame_rate=2.5, bone_tracks={0: [
        Transform(translation=p, rotation=Quat.identity(), scale=Vec3(1, 1, 1))
        for p in positions
    ]})
    skeleton = Skeleton(bone_names=["root"], bone_parents=[-1], bind_poses=[Transform.identity()])
    tex = bake_clip_to_texture(clip, skeleton)
    prev_x = -1.0
    for t in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.99]:
        result = tex.sample_bone_transform(0, t)
        assert result.translation.x >= prev_x - 0.001, f"non-monotonic at t={t}"
        prev_x = result.translation.x


# ---- Atlas UV ranges are non-overlapping ----

def test_atlas_uv_non_overlapping():
    atlas = AnimationTextureAtlas(format=TextureFormat.FLOAT32)
    t = Transform.identity()
    clip1 = AnimationClip(name="clip1", duration=1.0, frame_rate=2, bone_tracks={0: [t, t]})
    clip2 = AnimationClip(name="clip2", duration=1.0, frame_rate=3, bone_tracks={0: [t, t, t]})
    skeleton = Skeleton(bone_names=["root"], bone_parents=[-1], bind_poses=[Transform.identity()])
    tex1 = bake_clip_to_texture(clip1, skeleton)
    tex2 = bake_clip_to_texture(clip2, skeleton)
    assert atlas.add_clip("clip1", tex1)
    assert atlas.add_clip("clip2", tex2)
    uv1 = atlas.get_clip_uv_range("clip1")
    uv2 = atlas.get_clip_uv_range("clip2")
    assert uv1 is not None
    assert uv2 is not None
    # UV ranges should not overlap
    assert uv1[1] <= uv2[0] or uv2[1] <= uv1[0]


def test_atlas_uv_range_is_valid():
    atlas = AnimationTextureAtlas(format=TextureFormat.FLOAT32)
    t = Transform.identity()
    clip = AnimationClip(name="test", duration=1.0, frame_rate=2, bone_tracks={0: [t, t]})
    skeleton = Skeleton(bone_names=["root"], bone_parents=[-1], bind_poses=[Transform.identity()])
    tex = bake_clip_to_texture(clip, skeleton)
    atlas.add_clip("test", tex)
    uv = atlas.get_clip_uv_range("test")
    assert uv is not None
    assert 0.0 <= uv[0] <= 1.0
    assert 0.0 <= uv[1] <= 1.0
    assert uv[0] < uv[1]


# ---- Edge case: single-frame animation ----

def test_single_frame_animation():
    t = Transform(translation=Vec3(1, 2, 3), rotation=Quat.identity(), scale=Vec3(1, 1, 1))
    clip = AnimationClip(name="single", duration=0.0, frame_rate=30, bone_tracks={0: [t]})
    skeleton = Skeleton(bone_names=["root"], bone_parents=[-1], bind_poses=[Transform.identity()])
    tex = bake_clip_to_texture(clip, skeleton)
    assert tex.frame_count == 1
    # Sampling at any time should return the single frame
    result = tex.sample_bone_transform(0, 0.0)
    assert _transform_close(result, t)
    result = tex.sample_bone_transform(0, 999.0)
    assert _transform_close(result, t)


def test_single_frame_sample_bone_transform_duration_zero():
    """When duration is 0, sample_bone_transform returns frame 0."""
    t = Transform(translation=Vec3(5, 0, 0), rotation=Quat.identity(), scale=Vec3(1, 1, 1))
    clip = AnimationClip(name="instant", duration=0.0, frame_rate=30, bone_tracks={0: [t]})
    skeleton = Skeleton(bone_names=["root"], bone_parents=[-1], bind_poses=[Transform.identity()])
    tex = bake_clip_to_texture(clip, skeleton)
    result = tex.sample_bone_transform(0, 0.5)
    assert _transform_close(result, t)
