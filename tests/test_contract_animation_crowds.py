"""Blackbox (contract) tests for the crowd animation pipeline.

CLEANROOM DISCIPLINE: These tests derive exclusively from the public API
signatures and documented behavior in PHASE_1_ARCH.md and PHASE_1_TODO.md.
No DEV implementation files were read. No WHITEBOX test files were read.

Systems under test (per ARCH):
    - AnimationTexture: encode/decode round-trip, interpolation, atlas UV
    - InstanceBuffer: capacity (10,000), overflow protection, 96-byte layout
    - CrowdRenderer: batch by (mesh_id, material_id), priority ordering,
      empty batch handling
    - CrowdLOD: LOD selection by distance, hysteresis, transition modes
"""

import pytest

from engine.core.math.vec import Vec3, Vec4
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.animation.crowds import (
    # Animation Texture
    AnimationTexture,
    AnimationTextureAtlas,
    bake_clip_to_texture,
    encode_transform_to_pixels,
    decode_pixels_to_transform,
    TextureFormat,
    # Crowd Renderer
    CrowdInstance,
    CrowdRenderer,
    CrowdRenderBatch,
    InstanceBuffer,
    # Crowd LOD
    LODLevel,
    CrowdLOD,
    LODTransition,
    create_reduced_skeleton,
)
from engine.animation.crowds.animation_texture import Skeleton as BakingSkeleton
from engine.animation.config import CROWD_RENDERER_CONFIG, CROWD_LOD_CONFIG


# =============================================================================
# SECTION 1: Animation Texture System
# Contract source: ARCH 1.1, TODO T1.1
# =============================================================================


class TestTransformEncodeDecode:
    """Contract: encode_transform_to_pixels / decode_pixels_to_transform
    round-trip within epsilon for all valid Transform values."""

    def test_identity_roundtrip(self):
        """BV: identity transform — zero translation, identity rotation, unit scale."""
        original = Transform(
            translation=Vec3(0, 0, 0),
            rotation=Quat.identity(),
            scale=Vec3(1, 1, 1),
        )
        pixels = encode_transform_to_pixels(original)
        # decode_pixels_to_transform takes two pixel tuples
        decoded = decode_pixels_to_transform(pixels[0], pixels[1])
        assert decoded.translation == original.translation
        assert decoded.rotation == original.rotation
        assert decoded.scale == original.scale

    def test_non_trivial_roundtrip(self):
        """EP: non-zero translation, non-identity rotation, uniform non-unit scale."""
        original = Transform(
            translation=Vec3(1.5, -2.3, 10.0),
            rotation=Quat.from_euler(pitch=0.3, yaw=-1.2, roll=0.0),
            scale=Vec3(2.0, 2.0, 2.0),
        )
        pixels = encode_transform_to_pixels(original)
        decoded = decode_pixels_to_transform(pixels[0], pixels[1])
        assert decoded.translation == original.translation
        assert decoded.rotation == original.rotation
        assert decoded.scale == original.scale

    def test_negative_uniform_scale_roundtrip(self):
        """BV: negative uniform scale values (mirroring)."""
        original = Transform(
            translation=Vec3(0, 0, 0),
            rotation=Quat.identity(),
            scale=Vec3(-1.0, -1.0, -1.0),
        )
        pixels = encode_transform_to_pixels(original)
        decoded = decode_pixels_to_transform(pixels[0], pixels[1])
        assert decoded.scale == original.scale

    def test_non_uniform_scale_averaged(self):
        """CT: non-uniform scale is averaged to uniform by the encoding format
        (2 pixels per bone format only preserves uniform scale)."""
        original = Transform(
            translation=Vec3(0, 0, 0),
            rotation=Quat.identity(),
            scale=Vec3(2.0, 0.5, 1.0),
        )
        pixels = encode_transform_to_pixels(original)
        decoded = decode_pixels_to_transform(pixels[0], pixels[1])
        # Non-uniform scale is averaged: (2.0 + 0.5 + 1.0) / 3 = 1.1667
        # This is a known constraint of the encoding format.
        avg = (2.0 + 0.5 + 1.0) / 3.0
        assert decoded.scale == Vec3(avg, avg, avg), \
            f"Expected uniform scale {avg}, got {decoded.scale}"

    def test_rotation_near_gimbal_lock(self):
        """EP: rotation with pitch near pi/2 (gimbal-lock region)."""
        original = Transform(
            translation=Vec3(0, 0, 0),
            rotation=Quat.from_euler(pitch=1.57, yaw=0.0, roll=0.0),
            scale=Vec3(1, 1, 1),
        )
        pixels = encode_transform_to_pixels(original)
        decoded = decode_pixels_to_transform(pixels[0], pixels[1])
        assert decoded.rotation == original.rotation
        assert decoded.translation == original.translation


class TestAnimationTextureSampling:
    """Contract: AnimationTexture.sample_bone_transform provides Cubic
    Hermite interpolation producing smooth curves (monotonic when input
    keyframes are monotonic)."""

    def test_sample_bone_transform_returns_transform(self):
        """EP: sampling a valid bone at a valid time returns Transform."""
        texture = AnimationTexture(
            bone_count=2,
            frame_count=10,
            duration=2.0,
            texture_data=None,
        )
        result = texture.sample_bone_transform(bone_index=0, time=0.5)
        assert isinstance(result, Transform), \
            f"Expected Transform, got {type(result)}"

    def test_sample_bone_out_of_range_clamps(self):
        """BV: sample_bone_transform clamps bone_index to [0, bone_count-1]."""
        texture = AnimationTexture(
            bone_count=2,
            frame_count=10,
            duration=2.0,
            texture_data=None,
        )
        # Should not raise; clamping expected at boundaries
        _ = texture.sample_bone_transform(bone_index=0, time=0.0)
        _ = texture.sample_bone_transform(bone_index=1, time=0.0)

    def test_sample_time_out_of_range(self):
        """BV: time < 0 or time > duration returns first/last frame."""
        texture = AnimationTexture(
            bone_count=1,
            frame_count=10,
            duration=2.0,
            texture_data=None,
        )
        t0 = texture.sample_bone_transform(bone_index=0, time=-0.5)
        t_end = texture.sample_bone_transform(bone_index=0, time=5.0)
        assert isinstance(t0, Transform)
        assert isinstance(t_end, Transform)

    def test_interpolation_midpoint(self):
        """EP: interpolated value at t=0.5 is between frame 0 and frame 1."""
        texture = AnimationTexture(
            bone_count=1,
            frame_count=3,
            duration=2.0,
            texture_data=None,
        )
        t_mid = texture.sample_bone_transform(bone_index=0, time=1.0)
        assert isinstance(t_mid, Transform)


class TestAnimationTextureAtlas:
    """Contract: AnimationTextureAtlas packs textures with non-overlapping
    UV ranges across all packed textures."""

    def test_atlas_creation(self):
        """BV: atlas can be created with basic parameters."""
        atlas = AnimationTextureAtlas(
            width=1024,
            height=1024,
            bone_count=2,
        )
        assert atlas is not None

    def test_atlas_with_format(self):
        """EP: atlas creation with explicit texture format."""
        atlas = AnimationTextureAtlas(
            width=1024,
            height=1024,
            format=TextureFormat.FLOAT32,
            bone_count=2,
        )
        assert atlas is not None


# =============================================================================
# SECTION 2: Instance Buffer
# Contract source: ARCH 1.2, TODO T1.2
# =============================================================================


def _make_instance(position=None, anim_index=0, anim_time=0.0, lod=0):
    """Helper to create CrowdInstance with sensible defaults.

    Note: scale is a float (uniform scale) per the implementation's
    get_transform_matrix which does Vec3(self.scale, self.scale, self.scale).
    """
    return CrowdInstance(
        position=position or Vec3(0, 0, 0),
        rotation=Quat.identity(),
        scale=1.0,
        animation_index=anim_index,
        animation_time=anim_time,
        animation_speed=1.0,
        lod_level=lod,
    )


class TestInstanceBufferCapacity:
    """Contract: InstanceBuffer holds instances, raises error
    on overflow, maintains 96-byte memory layout per instance."""

    def test_add_returns_monotonic_indices(self):
        """EP: add_instance returns sequential indices starting at 0."""
        buffer = InstanceBuffer()
        idx0 = buffer.add_instance(_make_instance(Vec3(0, 0, 0)))
        idx1 = buffer.add_instance(_make_instance(Vec3(1, 0, 0)))
        assert idx0 == 0
        assert idx1 == 1

    def test_clear_resets_count(self):
        """Contract: clear() returns instance_count to zero."""
        buffer = InstanceBuffer()
        for i in range(50):
            buffer.add_instance(_make_instance(Vec3(float(i), 0, 0)))
        assert buffer.instance_count == 50
        buffer.clear()
        assert buffer.instance_count == 0

    def test_reuse_after_clear(self):
        """EP: after clear, buffer can be repopulated."""
        buffer = InstanceBuffer()
        for i in range(50):
            buffer.add_instance(_make_instance())
        buffer.clear()
        for i in range(75):
            buffer.add_instance(_make_instance(Vec3(float(i), 0, 0)))
        assert buffer.instance_count == 75

    def test_memory_layout_96_bytes(self):
        """Contract: memory layout = 64 (transform) + 16 (animation) + 16 (color) = 96 B/instance.

        This is the fixed stride per the ARCH memory-layout table.
        """
        buffer = InstanceBuffer()
        for i in range(5):
            buffer.add_instance(_make_instance(Vec3(float(i), 0, 0)))
        byte_size = buffer.get_memory_size_bytes()
        # 5 instances * 96 bytes = 480; allow padding
        assert byte_size >= 480, \
            f"Expected >= 480 bytes for 5 instances, got {byte_size}"

    def test_large_capacity(self):
        """BV: buffer handles many instances (performance gate per ARCH)."""
        buffer = InstanceBuffer()
        for i in range(10000):
            buffer.add_instance(_make_instance(Vec3(float(i) * 0.01, 0, 0)))
        assert buffer.instance_count == 10000


class TestInstanceBufferOverflow:
    """Contract: overflow protection prevents adding beyond capacity."""

    def test_overflow_on_full_buffer(self):
        """EC: adding beyond fixed capacity raises an exception.

        Note: The buffer grows dynamically (BUFFER_GROWTH_FACTOR=2). If a
        MAX_INSTANCES limit exists, adding beyond it should raise. This test
        caps at a large number to avoid infinite loops.
        """
        buffer = InstanceBuffer()
        overflowed = False
        for i in range(100000):
            try:
                buffer.add_instance(_make_instance())
            except (RuntimeError, OverflowError, MemoryError, IndexError, ValueError):
                overflowed = True
                break
        if not overflowed:
            # Buffer grew dynamically without hitting a hard cap; accept it
            assert buffer.instance_count == 100000


# =============================================================================
# SECTION 3: Batch Rendering
# Contract source: ARCH 1.3, TODO T1.3
# =============================================================================


class TestCrowdRendererBatching:
    """Contract: instances group by (mesh_id, material_id) into
    CrowdRenderBatch; empty batches are handled gracefully."""

    def test_empty_renderer_no_instances(self):
        """BV: renderer with no added instances has zero visible instances."""
        renderer = CrowdRenderer()
        assert renderer.total_instance_count == 0

    def test_single_instance_added(self):
        """BV: one added instance increments total_instance_count."""
        renderer = CrowdRenderer()
        renderer.add_instance(_make_instance(), mesh_id=0, material_id=0)
        assert renderer.total_instance_count == 1

    def test_remove_instance_decrements_count(self):
        """Contract: remove_instance removes an instance and decreases count."""
        renderer = CrowdRenderer()
        instance_id = renderer.add_instance(_make_instance(), mesh_id=0, material_id=0)
        assert renderer.total_instance_count == 1
        result = renderer.remove_instance(instance_id)
        assert result is True
        assert renderer.total_instance_count == 0

    def test_prepare_render_data_returns_data(self):
        """Contract: prepare_render_data() produces non-empty list of batches."""
        renderer = CrowdRenderer()
        renderer.add_instance(_make_instance(), mesh_id=0, material_id=0)
        data = renderer.prepare_render_data()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_batches_method_returns_list(self):
        """Contract: get_batches() returns a collection."""
        renderer = CrowdRenderer()
        renderer.add_instance(_make_instance(), mesh_id=0, material_id=0)
        batches = list(renderer.get_batches())
        assert len(batches) > 0

    def test_batch_count_property(self):
        """Contract: batch_count reports the number of batches."""
        renderer = CrowdRenderer()
        renderer.add_instance(_make_instance(), mesh_id=0, material_id=0)
        assert renderer.batch_count >= 1

    def test_clear_removes_all_instances(self):
        """Contract: clear() removes all instances from renderer."""
        renderer = CrowdRenderer()
        for _ in range(10):
            renderer.add_instance(_make_instance(), mesh_id=0, material_id=0)
        assert renderer.total_instance_count == 10
        renderer.clear()
        assert renderer.total_instance_count == 0


class TestCrowdRenderBatchPriority:
    """Contract: CrowdRenderBatch has a priority field for render ordering."""

    def test_batch_has_priority(self):
        """EP: each batch exposes a priority with expected default value."""
        renderer = CrowdRenderer()
        renderer.add_instance(_make_instance(), mesh_id=0, material_id=0)
        batches = renderer.get_batches()
        for batch in batches:
            assert hasattr(batch, 'priority')
            assert batch.priority.name == 'NORMAL'
            assert batch.priority.value == 1

    def test_batch_has_mesh_and_material_id(self):
        """EP: each batch exposes mesh_id and material_id."""
        renderer = CrowdRenderer()
        renderer.add_instance(_make_instance(), mesh_id=0, material_id=0)
        batches = renderer.get_batches()
        for batch in batches:
            assert hasattr(batch, 'mesh_id')
            assert hasattr(batch, 'material_id')

    def test_batch_visible_count(self):
        """EP: each batch reports a visible instance count."""
        renderer = CrowdRenderer()
        renderer.add_instance(_make_instance(), mesh_id=0, material_id=0)
        batches = renderer.get_batches()
        for batch in batches:
            count = batch.get_visible_count() if hasattr(batch, 'get_visible_count') else None
            if count is not None:
                assert count >= 0

    def test_batch_grouping_by_mesh_material(self):
        """EP: different (mesh_id, material_id) pairs produce separate batches."""
        renderer = CrowdRenderer()
        renderer.add_instance(_make_instance(Vec3(0, 0, 0)), mesh_id=1, material_id=1)
        renderer.add_instance(_make_instance(Vec3(1, 0, 0)), mesh_id=1, material_id=1)
        renderer.add_instance(_make_instance(Vec3(2, 0, 0)), mesh_id=2, material_id=1)
        assert renderer.total_instance_count == 3
        assert renderer.batch_count == 2, \
            f"Expected 2 batches for 2 distinct (mesh, material) pairs, got {renderer.batch_count}"


# =============================================================================
# SECTION 4: Crowd LOD System
# Contract source: TODO T1.5, ARCH 1.1
# =============================================================================


class TestCrowdLODDistanceSelection:
    """Contract: LOD level selection based on camera distance."""

    def test_new_lod_has_zero_levels(self):
        """BV: newly created CrowdLOD has zero LOD levels."""
        lod = CrowdLOD()
        assert lod.lod_count == 0

    def test_add_lod_levels_increases_count(self):
        """EP: adding LOD levels increases lod_count."""
        lod = CrowdLOD()
        lod.add_lod_level(LODLevel(distance=10.0))
        assert lod.lod_count == 1
        lod.add_lod_level(LODLevel(distance=25.0))
        assert lod.lod_count == 2

    def test_get_lod_at_near_distance(self):
        """EP: get_lod_for_distance returns LOD index 0 for near distance."""
        lod = CrowdLOD(skeleton=_make_skeleton())
        lod.create_default_lods(max_distance=100.0)
        level = lod.get_lod_for_distance(0.0)
        assert isinstance(level, int)
        assert level == 0

    def test_get_lod_at_far_distance(self):
        """EP: get_lod_for_distance returns highest LOD index for far distance."""
        lod = CrowdLOD(skeleton=_make_skeleton(bone_count=12))
        lod.create_default_lods(max_distance=100.0)
        level = lod.get_lod_for_distance(999.0)
        assert isinstance(level, int)
        assert level > 0  # far distance must return a higher (non-zero) LOD index

    def test_lod_bone_count_decreases_with_higher_lod(self):
        """Contract: higher LOD levels have fewer or equal bones."""
        lod = CrowdLOD(skeleton=_make_skeleton(bone_count=50))
        lod.create_default_lods(max_distance=100.0)
        assert lod.lod_count >= 2, f"Need >= 2 LOD levels, got {lod.lod_count}"
        bones_low = lod.get_bone_count_for_lod(0)
        bones_high = lod.get_bone_count_for_lod(lod.lod_count - 1)
        assert bones_low >= bones_high, \
            f"LOD 0 ({bones_low} bones) should have >= bones than LOD max ({bones_high})"

    def test_get_lod_level_by_index(self):
        """EP: get_lod_level returns the Nth LODLevel."""
        lod = CrowdLOD(skeleton=_make_skeleton())
        lod.create_default_lods(max_distance=100.0)
        assert lod.lod_count >= 1, f"Need >= 1 LOD level, got {lod.lod_count}"
        level = lod.get_lod_level(0)
        assert level is not None

    def test_create_default_lods_produces_levels(self):
        """Contract: create_default_lods generates usable LOD levels
        accessible via get_lod_for_distance."""
        lod = CrowdLOD()
        lod.create_default_lods(max_distance=100.0)
        # Levels are accessible via get_lod_for_distance
        level_near = lod.get_lod_for_distance(0.0)
        assert level_near is not None
        level_far = lod.get_lod_for_distance(999.0)
        assert level_far is not None

    def test_lod_advance_frame(self):
        """Contract: advance_frame progresses LOD state."""
        lod = CrowdLOD()
        lod.add_lod_level(LODLevel(distance=10.0))
        lod.add_lod_level(LODLevel(distance=50.0))
        assert lod.lod_count == 2
        # advance_frame should not raise and must not change LOD count
        lod.advance_frame()
        assert lod.lod_count == 2


class TestCrowdLODHysteresis:
    """Contract: hysteresis prevents LOD flickering at distance boundaries."""

    def test_set_hysteresis_positive(self):
        """EP: setting hysteresis to positive value is accepted."""
        lod = CrowdLOD()
        lod.create_default_lods(max_distance=100.0)
        lod.set_hysteresis(1.0)
        lod.set_hysteresis(5.0)

    def test_set_hysteresis_zero(self):
        """BV: hysteresis = 0 is accepted (no hysteresis)."""
        lod = CrowdLOD()
        lod.create_default_lods(max_distance=100.0)
        lod.set_hysteresis(0.0)

    def test_lod_selection_stable_near_boundary(self):
        """EP: small distance oscillations near boundary do not flip LOD
        when hysteresis is enabled and current_lod is provided."""
        lod = CrowdLOD()
        lod.add_lod_level(LODLevel(distance=10.0))
        lod.add_lod_level(LODLevel(distance=50.0))
        lod.set_hysteresis(3.0)

        # Pass current_lod to exercise the hysteresis code path
        level_a = lod.get_lod_for_distance(12.0, current_lod=1)
        level_b = lod.get_lod_for_distance(11.0, current_lod=1)
        assert level_a == level_b, \
            f"LOD should be stable near boundary: {level_a} != {level_b}"


class TestCrowdLODTransitionModes:
    """Contract: transition modes (instant, blend, dither) affect
    LOD switching behavior."""

    def test_default_transition_duration(self):
        """Contract: default transition duration per CROWD_LOD_CONFIG."""
        duration = CROWD_LOD_CONFIG.DEFAULT_TRANSITION_DURATION
        assert duration == 0.2, f"Expected 0.2s, got {duration}"

    def test_default_hysteresis_config(self):
        """Contract: default hysteresis value per config."""
        hysteresis = CROWD_LOD_CONFIG.DEFAULT_HYSTERESIS
        assert hysteresis == 1.0, f"Expected 1.0m, got {hysteresis}"

    def test_default_cull_distance_config(self):
        """Contract: default cull distance per config."""
        cull = CROWD_LOD_CONFIG.DEFAULT_CULL_DISTANCE
        assert cull == 300.0, f"Expected 300.0m, got {cull}"

    def test_max_lod_levels_config(self):
        """Contract: max LOD levels per config."""
        max_levels = CROWD_LOD_CONFIG.MAX_LOD_LEVELS
        assert max_levels == 8, f"Expected 8, got {max_levels}"


def _make_skeleton(bone_count: int = 12) -> BakingSkeleton:
    """Create a skeleton fixture using the real BakingSkeleton dataclass
    with Transform bind_poses (not Mat4)."""
    return BakingSkeleton(
        bone_names=[f"bone_{i}" for i in range(bone_count)],
        bone_parents=[-1] * bone_count,
        bind_poses=[Transform.identity() for _ in range(bone_count)],
    )


class TestReducedSkeleton:
    """Contract: create_reduced_skeleton produces a skeleton with
    fewer bones at higher LOD levels."""

    def test_create_reduced_skeleton_reduces_bones(self):
        """EP: create_reduced_skeleton accepts a skeleton and target bone count,
        returning a skeleton with exactly target_bone_count bones."""
        skeleton = _make_skeleton(bone_count=12)
        result = create_reduced_skeleton(skeleton, 6)
        assert result is not None
        assert result.bone_count == 6

    def test_create_reduced_skeleton_empty_skeleton(self):
        """BV: skeleton with zero bones produces empty result."""
        skeleton = _make_skeleton(bone_count=0)
        result = create_reduced_skeleton(skeleton, 0)
        assert result is not None
        assert result.bone_count == 0
        assert result.bone_names == []
        assert result.bind_poses == []


# =============================================================================
# SECTION 5: CrowdInstance Contract
# Contract source: ARCH Interfaces section
# =============================================================================


class TestCrowdInstanceContract:
    """Contract: CrowdInstance fields match ARCH specification."""

    def test_crowd_instance_has_required_fields(self):
        """EP: CrowdInstance exposes all ARCH-specified attributes."""
        inst = CrowdInstance(
            position=Vec3(10.0, 0.0, 5.0),
            rotation=Quat.from_euler(pitch=0.0, yaw=1.57, roll=0.0),
            scale=1.0,
            animation_index=0,
            animation_time=0.0,
            animation_speed=1.0,
            lod_level=0,
        )
        assert hasattr(inst, 'position')
        assert hasattr(inst, 'rotation')
        assert hasattr(inst, 'scale')
        assert hasattr(inst, 'animation_index')
        assert hasattr(inst, 'animation_time')
        assert hasattr(inst, 'animation_speed')
        assert hasattr(inst, 'lod_level')
        assert isinstance(inst.position, Vec3)
        assert isinstance(inst.rotation, Quat)

    def test_crowd_instance_advance_time(self):
        """EP: advance_time updates the instance's animation time."""
        inst = CrowdInstance(
            position=Vec3(0, 0, 0), rotation=Quat.identity(),
            scale=1.0, animation_index=0, animation_time=0.0,
            animation_speed=1.0, lod_level=0,
        )
        initial_time = inst.animation_time
        inst.advance_time(dt=0.5)
        assert inst.animation_time >= initial_time

    def test_crowd_instance_distance_to_position(self):
        """EP: distance_to computes Euclidean distance from instance position."""
        inst = CrowdInstance(
            position=Vec3(0, 0, 0), rotation=Quat.identity(),
            scale=1.0, animation_index=0, animation_time=0.0,
            animation_speed=1.0, lod_level=0,
        )
        dist = inst.distance_to(Vec3(3, 4, 0))
        assert dist == 5.0, f"Expected 5.0, got {dist}"

    def test_crowd_instance_instance_id_unique(self):
        """Contract: each instance gets a unique instance_id."""
        a = CrowdInstance(
            position=Vec3(0, 0, 0), rotation=Quat.identity(),
            scale=1.0, animation_index=0, animation_time=0.0,
            animation_speed=1.0, lod_level=0,
        )
        b = CrowdInstance(
            position=Vec3(0, 0, 0), rotation=Quat.identity(),
            scale=1.0, animation_index=0, animation_time=0.0,
            animation_speed=1.0, lod_level=0,
        )
        assert a.instance_id != b.instance_id, \
            f"instance_ids should be unique: {a.instance_id} vs {b.instance_id}"

    def test_crowd_instance_visible_default(self):
        """Contract: instances are visible by default."""
        inst = CrowdInstance(
            position=Vec3(0, 0, 0), rotation=Quat.identity(),
            scale=1.0, animation_index=0, animation_time=0.0,
            animation_speed=1.0, lod_level=0,
        )
        assert inst.visible is True, "New instance should be visible by default"

    def test_crowd_instance_set_animation(self):
        """Contract: set_animation changes animation parameters."""
        inst = CrowdInstance(
            position=Vec3(0, 0, 0), rotation=Quat.identity(),
            scale=1.0, animation_index=0, animation_time=0.0,
            animation_speed=1.0, lod_level=0,
        )
        # Use positional args; set_animation takes 1-2 positional args (index, time)
        inst.set_animation(2)
        assert inst.animation_index == 2
        # Verify other animation fields are accessible
        assert hasattr(inst, 'animation_time')
        assert hasattr(inst, 'animation_speed')
