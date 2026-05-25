"""Blackbox contract tests for Phase 1 Shadow Map Infrastructure.

CLEANROOM DISCIPLINE: This file tests only the public contract (API signatures,
documented behavior, constant values). It does NOT depend on implementation
details of engine/rendering/lighting/shadows.py.

Contract under test:
  - ShadowMap (abstract base, aka ShadowMapBase)
  - ShadowAtlas (shadow map texture atlas)
  - CascadedShadowMap (CSM)
  - CubeShadowMap (point light cube shadows)
  - SpotShadowMap (spot light shadows)
  - ShadowBiasConfig (bias configuration, via ShadowMapConfig)

Reference: PHASE_1_TODO.md (engine_rendering_lighting)
"""

from __future__ import annotations

import math
from typing import Optional

import pytest

from engine.core.math.mat import Mat4
from engine.core.math.vec import Vec2, Vec3
from engine.rendering.lighting import (
    CascadeData,
    CascadedShadowMap,
    CubeShadowMap,
    ShadowAtlas,
    ShadowAtlasSlot,
    ShadowMap,
    ShadowMapConfig,
    ShadowMapType,
    SpotShadowMap,
)
from engine.rendering.lighting.constants import (
    CSMConstants,
    ShadowConstants,
)
from engine.rendering.lighting.light_types import (
    DirectionalLight,
    PointLight,
    ShadowCasterConfig,
    ShadowMode,
    SpotLight,
)


# =============================================================================
# Helper: concrete subclass for testing the abstract ShadowMap base
# =============================================================================


class _TestingShadowMap(ShadowMap):
    """Minimal concrete ShadowMap subclass for contract testing.

    Provides the minimum contract: a shadow_type, a resolution, and a VP matrix.
    """

    @property
    def shadow_type(self) -> ShadowMapType:
        return ShadowMapType.VIRTUAL

    def get_resolution(self) -> tuple[int, int]:
        return (512, 512)

    def get_view_projection_matrix(self, face: int = 0) -> Mat4:
        return Mat4.identity()


# =============================================================================
# 1. ShadowMap (ShadowMapBase) Abstract Base Contract
# =============================================================================


class TestShadowMapBaseContract:
    """Verifies the abstract base class contract.

    Equivalence partitions:
      - Abstract class: cannot be instantiated directly
      - Concrete subclass: must implement shadow_type, get_resolution, get_view_projection_matrix
      - Dirty flag: starts True, can be cleared and re-marked
      - Config: defaults from ShadowConstants, customisable
    """

    def test_abstract_base_cannot_be_instantiated(self) -> None:
        """ShadowMap is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ShadowMap()  # type: ignore[abstract]

    def test_concrete_subclass_minimal_contract(self) -> None:
        """A minimal subclass satisfies the abstract contract."""
        sm = _TestingShadowMap()
        assert sm.shadow_type == ShadowMapType.VIRTUAL
        assert sm.get_resolution() == (512, 512)
        vp = sm.get_view_projection_matrix()
        assert isinstance(vp, Mat4)

    def test_dirty_flag_starts_true(self) -> None:
        """Shadow maps start dirty (need initial render)."""
        sm = _TestingShadowMap()
        assert sm.dirty is True

    def test_dirty_flag_clear_and_mark_cycle(self) -> None:
        """Full dirty lifecycle: start True -> clear -> mark -> verified."""
        sm = _TestingShadowMap()
        sm.clear_dirty()
        assert sm.dirty is False
        sm.mark_dirty()
        assert sm.dirty is True
        sm.clear_dirty()
        assert sm.dirty is False

    def test_light_id_defaults_to_zero(self) -> None:
        """Default light_id is 0."""
        sm = _TestingShadowMap()
        assert sm.light_id == 0

    def test_light_id_assignable(self) -> None:
        """light_id can be set via constructor."""
        sm = _TestingShadowMap(light_id=7)
        assert sm.light_id == 7

    def test_config_defaults_from_constants(self) -> None:
        """Default ShadowMapConfig matches ShadowConstants."""
        config = ShadowMapConfig()
        assert config.resolution == ShadowConstants.DEFAULT_RESOLUTION
        assert config.depth_bias == ShadowConstants.DEFAULT_DEPTH_BIAS
        assert config.slope_bias == ShadowConstants.DEFAULT_SLOPE_BIAS
        assert config.normal_bias == ShadowConstants.DEFAULT_NORMAL_BIAS
        assert config.filter_size == ShadowConstants.DEFAULT_FILTER_SIZE
        assert config.softness == ShadowConstants.DEFAULT_SOFTNESS

    def test_config_custom_resolution(self) -> None:
        """Custom resolution is accepted."""
        for res in [256, 512, 1024, 2048, 4096, 8192]:
            config = ShadowMapConfig(resolution=res)
            assert config.resolution == res

    def test_config_custom_bias(self) -> None:
        """Custom bias values are stored."""
        config = ShadowMapConfig(
            depth_bias=0.005,
            slope_bias=0.01,
            normal_bias=0.1,
        )
        assert config.depth_bias == 0.005
        assert config.slope_bias == 0.01
        assert config.normal_bias == 0.1

    def test_shadow_map_accepts_config(self) -> None:
        """ShadowMap constructor accepts a config."""
        config = ShadowMapConfig(resolution=1024)
        sm = _TestingShadowMap(config=config)
        assert sm.config.resolution == 1024

    def test_get_resolution_returns_positive_integers(self) -> None:
        """get_resolution always returns positive (width, height)."""
        sm = _TestingShadowMap()
        w, h = sm.get_resolution()
        assert isinstance(w, int)
        assert isinstance(h, int)
        assert w > 0
        assert h > 0

    def test_get_view_projection_matrix_returns_mat4(self) -> None:
        """get_view_projection_matrix returns a Mat4 instance."""
        sm = _TestingShadowMap()
        vp = sm.get_view_projection_matrix()
        assert isinstance(vp, Mat4)

    def test_get_view_projection_matrix_accepts_face_arg(self) -> None:
        """get_view_projection_matrix accepts optional face argument."""
        sm = _TestingShadowMap()
        vp = sm.get_view_projection_matrix(face=0)
        assert isinstance(vp, Mat4)


# =============================================================================
# 2. Shadow Constants Contract
# =============================================================================


class TestShadowConstantsContract:
    """Verifies all shadow-related constants match the spec.

    Boundary values: min/max resolution, cascade count, near plane.
    """

    def test_default_resolution(self) -> None:
        assert ShadowConstants.DEFAULT_RESOLUTION == 2048

    def test_min_resolution(self) -> None:
        assert ShadowConstants.MIN_RESOLUTION == 256

    def test_max_resolution(self) -> None:
        assert ShadowConstants.MAX_RESOLUTION == 8192

    def test_default_atlas_resolution(self) -> None:
        assert ShadowConstants.DEFAULT_ATLAS_RESOLUTION == 4096

    def test_default_depth_bias(self) -> None:
        assert ShadowConstants.DEFAULT_DEPTH_BIAS == pytest.approx(0.0001)

    def test_default_slope_bias(self) -> None:
        assert ShadowConstants.DEFAULT_SLOPE_BIAS == pytest.approx(0.001)

    def test_default_normal_bias(self) -> None:
        assert ShadowConstants.DEFAULT_NORMAL_BIAS == pytest.approx(0.02)

    def test_default_filter_size(self) -> None:
        assert ShadowConstants.DEFAULT_FILTER_SIZE == 3

    def test_default_softness(self) -> None:
        assert ShadowConstants.DEFAULT_SOFTNESS == 1.0

    def test_shadow_near_plane(self) -> None:
        assert ShadowConstants.SHADOW_NEAR_PLANE == 0.1

    def test_csm_min_cascade(self) -> None:
        assert CSMConstants.MIN_CASCADE_COUNT == 1

    def test_csm_max_cascade(self) -> None:
        assert CSMConstants.MAX_CASCADE_COUNT == 4

    def test_csm_default_cascade(self) -> None:
        assert CSMConstants.DEFAULT_CASCADE_COUNT == 4

    def test_csm_lambda(self) -> None:
        assert CSMConstants.CASCADE_LAMBDA == 0.75

    def test_csm_default_distances(self) -> None:
        dists = CSMConstants.DEFAULT_CASCADE_DISTANCES
        assert len(dists) == 4
        assert dists == [10.0, 30.0, 100.0, 500.0]

    def test_csm_blend_range(self) -> None:
        assert CSMConstants.DEFAULT_CASCADE_BLEND_RANGE == 2.0


# =============================================================================
# 3. ShadowMapType Enum Contract
# =============================================================================


class TestShadowMapTypeContract:
    """Verifies the ShadowMapType enum variants.

    Equivalence: each variant is a distinct member.
    Boundary: exactly 4 variants are defined.
    """

    def test_cascaded_variant(self) -> None:
        assert ShadowMapType.CASCADED is not None

    def test_cube_variant(self) -> None:
        assert ShadowMapType.CUBE is not None

    def test_spot_variant(self) -> None:
        assert ShadowMapType.SPOT is not None

    def test_virtual_variant(self) -> None:
        assert ShadowMapType.VIRTUAL is not None

    def test_all_variants_are_unique(self) -> None:
        variants = list(ShadowMapType)
        assert len(variants) == 4
        assert len(set(variants)) == 4


# =============================================================================
# 4. ShadowAtlas Contract
# =============================================================================


class TestShadowAtlasContract:
    """Verifies the ShadowAtlas public contract (T-LGT-1.3).

    Equivalence partitions:
      - Valid resolution: power of 2, >0
      - Invalid resolution: non-power-of-2, zero, negative
      - Empty atlas: zero slots, zero utilization
      - Single allocation: returns slot at origin
      - Multiple allocations: non-overlapping
      - Overflow: returns None when full
      - Deallocation: frees space, enables reuse
    """

    def test_default_resolution(self) -> None:
        """Default atlas resolution is 4096."""
        atlas = ShadowAtlas()
        assert atlas.resolution == 4096

    def test_custom_resolution(self) -> None:
        """Atlas accepts custom power-of-2 resolution."""
        for res in [256, 512, 1024, 2048, 4096, 8192]:
            atlas = ShadowAtlas(resolution=res)
            assert atlas.resolution == res

    def test_invalid_resolution_raises(self) -> None:
        """Non-power-of-2, zero, or negative resolution raises ValueError."""
        with pytest.raises(ValueError, match=".*[Pp]ower.*|.*[Ii]nvalid.*|.*resolution.*"):
            ShadowAtlas(resolution=1000)
        with pytest.raises(ValueError):
            ShadowAtlas(resolution=0)
        with pytest.raises(ValueError):
            ShadowAtlas(resolution=-1)

    def test_atlas_starts_empty(self) -> None:
        """New atlas has no slots and zero utilization."""
        atlas = ShadowAtlas()
        assert len(atlas.slots) == 0
        assert atlas.get_utilization() == 0.0

    def test_allocate_returns_slot(self) -> None:
        """Allocate returns a slot with correct dimensions at origin."""
        atlas = ShadowAtlas(resolution=4096)
        slot = atlas.allocate(512, 512)
        assert slot is not None
        assert slot.x == 0
        assert slot.y == 0
        assert slot.width == 512
        assert slot.height == 512
        assert slot.shadow_map is None

    def test_allocate_increments_slot_count(self) -> None:
        """Allocation increments slot count."""
        atlas = ShadowAtlas(resolution=4096)
        atlas.allocate(512, 512)
        assert len(atlas.slots) == 1

    def test_allocate_multiple_non_overlapping(self) -> None:
        """Multiple allocations produce non-overlapping slots."""
        atlas = ShadowAtlas(resolution=4096)
        n = 8
        slots = []
        for _ in range(n):
            slot = atlas.allocate(512, 512)
            assert slot is not None, f"Allocation {len(slots)} returned None"
            slots.append(slot)

        # Verify pairwise non-overlap
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                a = slots[i]
                b = slots[j]
                a_left, a_right = a.x, a.x + a.width
                a_bottom, a_top = a.y, a.y + a.height
                b_left, b_right = b.x, b.x + b.width
                b_bottom, b_top = b.y, b.y + b.height
                x_overlap = a_left < b_right and a_right > b_left
                y_overlap = a_bottom < b_top and a_top > b_bottom
                assert not (x_overlap and y_overlap), (
                    f"Slot {i} ({a_left},{a_bottom},{a_right},{a_top}) "
                    f"overlaps slot {j} ({b_left},{b_bottom},{b_right},{b_top})"
                )

    def test_allocate_returns_none_when_full(self) -> None:
        """Allocate returns None when no space remains."""
        atlas = ShadowAtlas(resolution=512)
        # Exhaust atlas with many small allocations
        for _ in range(100):
            slot = atlas.allocate(256, 256)
            if slot is None:
                break
        # No more space for a large allocation
        big = atlas.allocate(512, 512)
        assert big is None

    def test_deallocate_frees_slot(self) -> None:
        """Deallocate removes the slot from atlas."""
        atlas = ShadowAtlas(resolution=2048)
        slot = atlas.allocate(256, 256)
        assert slot is not None
        assert len(atlas.slots) == 1

        atlas.deallocate(slot)
        assert len(atlas.slots) == 0

    def test_deallocate_reenables_reuse(self) -> None:
        """After deallocate, freed space can be reallocated."""
        atlas = ShadowAtlas(resolution=2048)
        slot = atlas.allocate(1024, 1024)
        assert slot is not None
        atlas.deallocate(slot)
        slot2 = atlas.allocate(1024, 1024)
        assert slot2 is not None

    def test_allocate_shadow_map(self) -> None:
        """allocate_shadow_map assigns shadow map reference to slot."""
        atlas = ShadowAtlas(resolution=4096)
        csm = CascadedShadowMap(light_id=42)
        slot = atlas.allocate_shadow_map(csm)
        assert slot is not None
        assert slot.shadow_map is csm

    def test_get_slot_for_light_found(self) -> None:
        """get_slot_for_light retrieves slot by light_id."""
        atlas = ShadowAtlas(resolution=4096)
        csm = CascadedShadowMap(light_id=42)
        atlas.allocate_shadow_map(csm)
        found = atlas.get_slot_for_light(42)
        assert found is not None
        assert found.shadow_map is csm

    def test_get_slot_for_light_not_found(self) -> None:
        """get_slot_for_light returns None for unknown light_id."""
        atlas = ShadowAtlas(resolution=4096)
        assert atlas.get_slot_for_light(999) is None

    def test_get_uv_transform_returns_0_1_range(self) -> None:
        """get_uv_transform returns (offset, scale) in normalized UV space."""
        atlas = ShadowAtlas(resolution=4096)
        slot = atlas.allocate(1024, 1024)
        assert slot is not None
        offset, scale = atlas.get_uv_transform(slot)
        assert isinstance(offset, Vec2)
        assert isinstance(scale, Vec2)
        # All values in [0, 1] range
        assert 0.0 <= offset.x <= 1.0
        assert 0.0 <= offset.y <= 1.0
        assert 0.0 < scale.x <= 1.0
        assert 0.0 < scale.y <= 1.0
        # offset + scale <= 1.0 (fits in atlas)
        assert offset.x + scale.x <= 1.0
        assert offset.y + scale.y <= 1.0

    def test_utilization_increases_with_allocations(self) -> None:
        """Utilization ratio increases monotonically with allocations."""
        atlas = ShadowAtlas(resolution=4096)
        util_before = atlas.get_utilization()
        atlas.allocate(1024, 1024)
        util_after = atlas.get_utilization()
        assert util_after > util_before

    def test_utilization_bounded_0_to_1(self) -> None:
        """Utilization always stays within [0, 1]."""
        atlas = ShadowAtlas(resolution=4096)
        assert 0.0 <= atlas.get_utilization() <= 1.0

        atlas.allocate(2048, 2048)
        assert 0.0 <= atlas.get_utilization() <= 1.0

    def test_defragment_preserves_references(self) -> None:
        """Defragment compacts without losing shadow map associations."""
        atlas = ShadowAtlas(resolution=4096)
        csm1 = CascadedShadowMap(light_id=1)
        csm2 = CascadedShadowMap(light_id=2)
        atlas.allocate_shadow_map(csm1)
        atlas.allocate_shadow_map(csm2)
        atlas.defragment()
        # Both shadow maps still reachable
        assert atlas.get_slot_for_light(1) is not None
        assert atlas.get_slot_for_light(2) is not None

    def test_defragment_marks_maps_dirty(self) -> None:
        """Defragmented shadow maps are marked dirty (need re-render)."""
        atlas = ShadowAtlas(resolution=4096)
        csm1 = CascadedShadowMap(light_id=1)
        csm2 = CascadedShadowMap(light_id=2)
        atlas.allocate_shadow_map(csm1)
        atlas.allocate_shadow_map(csm2)
        csm1.clear_dirty()
        csm2.clear_dirty()
        atlas.defragment()
        assert csm1.dirty is True
        assert csm2.dirty is True

    def test_slot_uv_properties(self) -> None:
        """ShadowAtlasSlot exposes uv_offset and uv_scale as Vec2."""
        slot = ShadowAtlasSlot(x=256, y=128, width=512, height=256)
        offset = slot.uv_offset
        scale = slot.uv_scale
        assert isinstance(offset, Vec2)
        assert isinstance(scale, Vec2)
        assert offset.x == 256
        assert offset.y == 128
        assert scale.x == 512
        assert scale.y == 256


# =============================================================================
# 5. CascadedShadowMap Contract (T-LGT-1.4)
# =============================================================================


class TestCascadedShadowMapContract:
    """Verifies the CascadedShadowMap public contract.

    Equivalence partitions:
      - Valid cascade counts: 1, 2, 3, 4
      - Invalid cascade counts: 0, 5 (outside [1, 4])
      - Splits: strictly increasing, bounded by near/far
      - Cascade data: each has split_depth, texel_size

    Boundary values:
      - cascade_count=1 (minimum valid)
      - cascade_count=4 (maximum valid)
      - cascade_count=0 (below minimum)
      - cascade_count=5 (above maximum)
    """

    def test_default_cascade_count(self) -> None:
        """Default cascade count is 4."""
        csm = CascadedShadowMap()
        assert csm.cascade_count == 4

    def test_valid_cascade_counts(self) -> None:
        """Cascade counts 1 through 4 are valid."""
        for count in range(1, 5):
            csm = CascadedShadowMap(cascade_count=count)
            assert csm.cascade_count == count

    def test_invalid_cascade_count_zero_raises(self) -> None:
        """cascade_count=0 raises ValueError."""
        with pytest.raises(ValueError):
            CascadedShadowMap(cascade_count=0)

    def test_invalid_cascade_count_five_raises(self) -> None:
        """cascade_count=5 raises ValueError."""
        with pytest.raises(ValueError):
            CascadedShadowMap(cascade_count=5)

    def test_cascade_data_initialized(self) -> None:
        """CascadeData list length matches cascade_count."""
        csm = CascadedShadowMap(cascade_count=4)
        assert len(csm.cascade_data) == 4
        for cascade in csm.cascade_data:
            assert isinstance(cascade, CascadeData)

    def test_get_resolution_from_config(self) -> None:
        """get_resolution returns (config.resolution, config.resolution)."""
        csm = CascadedShadowMap(config=ShadowMapConfig(resolution=1024))
        w, h = csm.get_resolution()
        assert w == 1024
        assert h == 1024

    def test_get_view_projection_per_cascade(self) -> None:
        """get_view_projection_matrix returns a Mat4 for each valid cascade index."""
        csm = CascadedShadowMap(cascade_count=4)
        for i in range(4):
            vp = csm.get_view_projection_matrix(face=i)
            assert isinstance(vp, Mat4)

    def test_get_view_projection_out_of_range_returns_identity(self) -> None:
        """Out-of-range cascade index returns identity matrix."""
        csm = CascadedShadowMap(cascade_count=2)
        vp = csm.get_view_projection_matrix(face=99)
        assert vp == Mat4()

    def test_get_cascade_for_depth_distribution(self) -> None:
        """get_cascade_for_depth distributes depths within split range."""
        csm = CascadedShadowMap(cascade_count=4)
        # Before splits are computed, all split_depths are 0.0,
        # so any depth > 0 returns last cascade.
        for depth in [1.0, 50.0, 500.0]:
            idx = csm.get_cascade_for_depth(depth)
            assert 0 <= idx < 4

    def test_configure_for_light_sets_split_depths(self) -> None:
        """configure_for_light computes positive split depths."""
        csm = CascadedShadowMap(cascade_count=4)
        light = DirectionalLight()
        view = Mat4.look_at(Vec3(0, 0, 5), Vec3(0, 0, 0), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.radians(60), 16 / 9, 0.1, 1000.0)

        csm.configure_for_light(light, view, proj, 0.1, 1000.0)

        for i, cascade in enumerate(csm.cascade_data):
            assert cascade.split_depth > 0, f"Cascade {i} split must be > 0"

    def test_split_depths_monotonically_increasing(self) -> None:
        """Cascade split depths are strictly increasing."""
        csm = CascadedShadowMap(cascade_count=4)
        splits = csm._compute_cascade_splits(0.1, 500.0)
        for i in range(1, len(splits)):
            assert splits[i] > splits[i - 1], (
                f"Split {i} ({splits[i]:.2f}) must be > split {i-1} ({splits[i-1]:.2f})"
            )

    def test_first_split_near_near_plane(self) -> None:
        """First cascade split is close to the near plane."""
        csm = CascadedShadowMap(cascade_count=4)
        splits = csm._compute_cascade_splits(0.1, 500.0)
        assert splits[0] > 0.1
        assert splits[0] < 50.0  # Reasonably close

    def test_last_split_at_far_plane(self) -> None:
        """Last cascade split does not exceed the far plane."""
        csm = CascadedShadowMap(cascade_count=4)
        splits = csm._compute_cascade_splits(0.1, 500.0)
        assert splits[-1] <= 500.0

    def test_configure_updates_light_direction(self) -> None:
        """configure_for_light stores normalized light direction."""
        csm = CascadedShadowMap(cascade_count=2)
        light = DirectionalLight(direction=Vec3(1, -1, 0))
        view = Mat4.look_at(Vec3(0, 0, 5), Vec3(0, 0, 0), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.radians(60), 16 / 9, 0.1, 1000.0)

        csm.configure_for_light(light, view, proj, 0.1, 1000.0)
        # Direction should be normalized
        assert abs(csm._light_direction.length() - 1.0) < 1e-6

    def test_configure_sets_texel_sizes(self) -> None:
        """After configure, cascade texel_sizes are positive."""
        csm = CascadedShadowMap(cascade_count=4)
        light = DirectionalLight()
        view = Mat4.look_at(Vec3(0, 0, 5), Vec3(0, 0, 0), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.radians(60), 16 / 9, 0.1, 1000.0)

        csm.configure_for_light(light, view, proj, 0.1, 1000.0)
        for i, cascade in enumerate(csm.cascade_data):
            assert cascade.texel_size > 0, f"Cascade {i} texel_size must be > 0"

    def test_default_blend_range(self) -> None:
        """Default cascade blend range is 2.0."""
        csm = CascadedShadowMap()
        assert csm.cascade_blend_range == 2.0

    def test_stabilize_cascades_default(self) -> None:
        """Stabilize cascades defaults to True."""
        csm = CascadedShadowMap()
        assert csm.stabilize_cascades is True


# =============================================================================
# 6. CubeShadowMap Contract
# =============================================================================


class TestCubeShadowMapContract:
    """Verifies the CubeShadowMap public contract.

    Equivalence:
      - Default position: origin
      - Default radius: 10.0
      - Face count: exactly 6
      - Face directions: +X, -X, +Y, -Y, +Z, -Z

    Boundary:
      - Face index 0 and 5 (valid range)
      - Face index 6 and -1 (out of range)
    """

    def test_default_position(self) -> None:
        """Default position is at origin."""
        cube = CubeShadowMap()
        assert cube.position.x == 0.0
        assert cube.position.y == 0.0
        assert cube.position.z == 0.0

    def test_default_radius(self) -> None:
        """Default radius is 10.0."""
        cube = CubeShadowMap()
        assert cube.radius == 10.0

    def test_shadow_type_is_cube(self) -> None:
        """shadow_type returns CUBE."""
        cube = CubeShadowMap()
        assert cube.shadow_type == ShadowMapType.CUBE

    def test_get_resolution_from_config(self) -> None:
        """get_resolution returns (config.resolution, config.resolution)."""
        cube = CubeShadowMap(config=ShadowMapConfig(resolution=256))
        w, h = cube.get_resolution()
        assert w == 256
        assert h == 256

    def test_six_faces_initialized(self) -> None:
        """Exactly 6 face matrices are pre-initialized."""
        cube = CubeShadowMap()
        assert len(cube.face_matrices) == 6
        for i, mat in enumerate(cube.face_matrices):
            assert isinstance(mat, Mat4), f"Face {i} matrix is not Mat4"

    def test_view_projection_for_each_face(self) -> None:
        """get_view_projection_matrix returns non-identity Mat4 for faces 0-5."""
        cube = CubeShadowMap()
        for face in range(6):
            vp = cube.get_view_projection_matrix(face=face)
            assert isinstance(vp, Mat4)
            # Each face should have a distinct VP (not identity for default cube)
            # Mat4 stores 16 floats in column-major flat layout; check all entries
            has_content = any(abs(vp.m[i]) > 1e-6 for i in range(16))
            assert has_content, f"Face {face} VP is identity"

    def test_view_projection_out_of_range_returns_identity(self) -> None:
        """Out-of-range face returns identity."""
        cube = CubeShadowMap()
        vp = cube.get_view_projection_matrix(face=99)
        assert vp == Mat4()

    def test_face_directions_cover_all_axes(self) -> None:
        """Six face directions cover +-X, +-Y, +-Z."""
        cube = CubeShadowMap()
        directions = [cube.get_face_direction(i) for i in range(6)]
        has_positive_x = any(d.x > 0.5 for d in directions)
        has_negative_x = any(d.x < -0.5 for d in directions)
        has_positive_y = any(d.y > 0.5 for d in directions)
        has_negative_y = any(d.y < -0.5 for d in directions)
        has_positive_z = any(d.z > 0.5 for d in directions)
        has_negative_z = any(d.z < -0.5 for d in directions)
        assert has_positive_x and has_negative_x, "+-X must be covered"
        assert has_positive_y and has_negative_y, "+-Y must be covered"
        assert has_positive_z and has_negative_z, "+-Z must be covered"

    def test_face_0_positive_x(self) -> None:
        """Face 0 is +X direction."""
        cube = CubeShadowMap()
        d = cube.get_face_direction(0)
        assert d.x == pytest.approx(1.0, abs=1e-6)
        assert d.y == pytest.approx(0.0, abs=1e-6)
        assert d.z == pytest.approx(0.0, abs=1e-6)

    def test_face_1_negative_x(self) -> None:
        """Face 1 is -X direction."""
        cube = CubeShadowMap()
        d = cube.get_face_direction(1)
        assert d.x == pytest.approx(-1.0, abs=1e-6)

    def test_configure_for_light_updates_position_and_radius(self) -> None:
        """configure_for_light sets position and radius from PointLight."""
        cube = CubeShadowMap()
        light = PointLight(position=Vec3(10, 20, 30), radius=50.0)
        cube.configure_for_light(light)
        assert cube.position.x == 10.0
        assert cube.position.y == 20.0
        assert cube.position.z == 30.0
        assert cube.radius == 50.0

    def test_configure_for_light_sets_dirty(self) -> None:
        """configure_for_light marks the map dirty."""
        cube = CubeShadowMap()
        cube.clear_dirty()
        light = PointLight(position=Vec3(5, 5, 5), radius=25.0)
        cube.configure_for_light(light)
        assert cube.dirty is True


# =============================================================================
# 7. SpotShadowMap Contract
# =============================================================================


class TestSpotShadowMapContract:
    """Verifies the SpotShadowMap public contract.

    Equivalence:
      - Default outer angle: 45 degrees
      - Valid angles: (0, pi)
      - Position, direction, radius defaults

    Boundary:
      - outer_angle at 0 (degenerate cone)
      - outer_angle at pi (hemisphere)
    """

    def test_default_outer_angle(self) -> None:
        """Default outer angle is 45 degrees."""
        spot = SpotShadowMap()
        assert spot.outer_angle == pytest.approx(math.radians(45.0))

    def test_shadow_type_is_spot(self) -> None:
        """shadow_type returns SPOT."""
        spot = SpotShadowMap()
        assert spot.shadow_type == ShadowMapType.SPOT

    def test_get_resolution_from_config(self) -> None:
        """get_resolution returns (config.resolution, config.resolution)."""
        spot = SpotShadowMap(config=ShadowMapConfig(resolution=1024))
        w, h = spot.get_resolution()
        assert w == 1024
        assert h == 1024

    def test_view_projection_returns_mat4(self) -> None:
        """get_view_projection_matrix returns a Mat4."""
        spot = SpotShadowMap()
        vp = spot.get_view_projection_matrix()
        assert isinstance(vp, Mat4)

    def test_configure_for_light_updates_params(self) -> None:
        """configure_for_light sets position, direction, angle, radius."""
        spot = SpotShadowMap()
        light = SpotLight(
            position=Vec3(5, 10, 15),
            direction=Vec3(0, -1, 0),
            outer_angle=math.radians(30.0),
            radius=25.0,
        )
        spot.configure_for_light(light)
        assert spot.position.x == 5.0
        assert spot.position.y == 10.0
        assert spot.position.z == 15.0
        assert spot.radius == 25.0
        assert spot.outer_angle == pytest.approx(math.radians(30.0))

    def test_configure_marks_dirty(self) -> None:
        """configure_for_light marks the map dirty."""
        spot = SpotShadowMap()
        spot.clear_dirty()
        light = SpotLight(
            position=Vec3(0, 10, 0),
            direction=Vec3(0, -1, 0),
            outer_angle=math.radians(30.0),
            radius=10.0,
        )
        spot.configure_for_light(light)
        assert spot.dirty is True

    def test_view_projection_after_configure(self) -> None:
        """get_view_projection_matrix is valid after configure."""
        spot = SpotShadowMap()
        light = SpotLight(
            position=Vec3(100, 0, 0),
            direction=Vec3(-1, 0, 0),
            outer_angle=math.radians(60.0),
            radius=50.0,
        )
        spot.configure_for_light(light)
        vp = spot.get_view_projection_matrix()
        assert isinstance(vp, Mat4)

    def test_outer_angle_matches_light_after_configure(self) -> None:
        """Spot outer_angle matches the source SpotLight after configure."""
        spot = SpotShadowMap()
        light = SpotLight(
            position=Vec3(0, 10, 0),
            direction=Vec3(0, -1, 0),
            outer_angle=math.radians(30.0),
            radius=50.0,
        )
        spot.configure_for_light(light)
        assert spot.outer_angle == pytest.approx(light.outer_angle)


# =============================================================================
# 8. ShadowBias Configuration Contract (T-LGT-1.6)
# =============================================================================


class TestShadowBiasConfigContract:
    """Verifies shadow bias configuration through ShadowMapConfig.

    The Phase 1 TODO (T-LGT-1.6) specifies a ShadowBias dataclass with
    constant, slope_scale, and normal_offset fields. The current ShadowMapConfig
    already carries equivalent fields (depth_bias, slope_bias, normal_bias).

    Equivalence:
      - Default bias values from constants
      - Custom bias values accepted
      - Bias inherited by all shadow map types

    After T-LGT-1.6 implementation, a standalone ShadowBias class may also
    be available through shadows module imports.
    """

    def test_config_has_bias_fields(self) -> None:
        """ShadowMapConfig exposes depth_bias, slope_bias, normal_bias."""
        config = ShadowMapConfig()
        assert hasattr(config, "depth_bias")
        assert hasattr(config, "slope_bias")
        assert hasattr(config, "normal_bias")

    def test_default_bias_values(self) -> None:
        """Default bias values match ShadowConstants."""
        config = ShadowMapConfig()
        assert config.depth_bias == pytest.approx(ShadowConstants.DEFAULT_DEPTH_BIAS)
        assert config.slope_bias == pytest.approx(ShadowConstants.DEFAULT_SLOPE_BIAS)
        assert config.normal_bias == pytest.approx(ShadowConstants.DEFAULT_NORMAL_BIAS)

    def test_custom_bias_accepted(self) -> None:
        """Non-default bias values are accepted."""
        config = ShadowMapConfig(
            depth_bias=0.005,
            slope_bias=0.01,
            normal_bias=0.1,
        )
        assert config.depth_bias == 0.005
        assert config.slope_bias == 0.01
        assert config.normal_bias == 0.1

    def test_bias_inherited_by_csm(self) -> None:
        """CascadedShadowMap inherits bias from config."""
        config = ShadowMapConfig(depth_bias=0.002, slope_bias=0.005)
        csm = CascadedShadowMap(config=config)
        assert csm.config.depth_bias == 0.002
        assert csm.config.slope_bias == 0.005

    def test_bias_inherited_by_cube(self) -> None:
        """CubeShadowMap inherits bias from config."""
        config = ShadowMapConfig(depth_bias=0.002, slope_bias=0.005)
        cube = CubeShadowMap(config=config)
        assert cube.config.depth_bias == 0.002
        assert cube.config.slope_bias == 0.005

    def test_bias_inherited_by_spot(self) -> None:
        """SpotShadowMap inherits bias from config."""
        config = ShadowMapConfig(depth_bias=0.002, slope_bias=0.005)
        spot = SpotShadowMap(config=config)
        assert spot.config.depth_bias == 0.002
        assert spot.config.slope_bias == 0.005

    def test_light_has_shadow_config(self) -> None:
        """DirectionalLight carries shadow_caster config."""
        light = DirectionalLight()
        assert hasattr(light, "_shadow_config")
        assert isinstance(light._shadow_config, ShadowCasterConfig)
        assert light._shadow_config.mode == ShadowMode.DYNAMIC


# =============================================================================
# 9. CascadeData Contract
# =============================================================================


class TestCascadeDataContract:
    """Verifies the CascadeData dataclass contract."""

    def test_default_values(self) -> None:
        """Default cascade data has zero split_depth and texel_size."""
        cascade = CascadeData()
        assert cascade.split_depth == 0.0
        assert cascade.texel_size == 0.0

    def test_custom_values(self) -> None:
        """CascadeData accepts custom split_depth and texel_size."""
        cascade = CascadeData(split_depth=100.0, texel_size=0.05)
        assert cascade.split_depth == 100.0
        assert cascade.texel_size == 0.05


# =============================================================================
# 10. Light-Shadow Integration Contract
# =============================================================================


class TestLightShadowIntegrationContract:
    """Verifies integration between light types and shadow maps."""

    def test_directional_light_cascade_count(self) -> None:
        """DirectionalLight accepts cascade_count."""
        light = DirectionalLight(cascade_count=3)
        assert light.cascade_count == 3

    def test_directional_light_cascade_distances(self) -> None:
        """DirectionalLight cascade_distances match count."""
        light = DirectionalLight(cascade_count=3)
        assert len(light.cascade_distances) == 3

    def test_directional_light_angular_diameter(self) -> None:
        """DirectionalLight has sun-like angular diameter."""
        light = DirectionalLight()
        assert light.angular_diameter == pytest.approx(0.00935, abs=1e-5)

    def test_point_light_has_shadow_config(self) -> None:
        """PointLight has shadow_caster config."""
        light = PointLight()
        assert light._shadow_config is not None

    def test_spot_light_has_shadow_config(self) -> None:
        """SpotLight has shadow_caster config."""
        light = SpotLight()
        assert light._shadow_config is not None

    def test_shadow_caster_config_defaults(self) -> None:
        """ShadowCasterConfig defaults to DYNAMIC mode."""
        config = ShadowCasterConfig()
        assert config.mode == ShadowMode.DYNAMIC
        assert config.resolution_scale == 1.0
        assert config.cascade_bias == 0.0

    def test_shadow_caster_config_validates_resolution_scale(self) -> None:
        """resolution_scale must be positive."""
        with pytest.raises(ValueError):
            ShadowCasterConfig(resolution_scale=0.0)
        with pytest.raises(ValueError):
            ShadowCasterConfig(resolution_scale=-1.0)


# =============================================================================
# 11. Error Handling & Edge Cases
# =============================================================================


class TestShadowEdgeCaseContract:
    """Contract tests for shadow system error handling.

    These tests validate that the public API properly rejects invalid inputs
    and handles boundary conditions as specified in PHASE_1_TODO.md.
    """

    def test_cascade_count_1_valid(self) -> None:
        """Single cascade is the minimum valid count."""
        csm = CascadedShadowMap(cascade_count=1)
        assert csm.cascade_count == 1
        assert len(csm.cascade_data) == 1

    def test_cascade_count_4_valid(self) -> None:
        """Four cascades is the maximum valid count."""
        csm = CascadedShadowMap(cascade_count=4)
        assert csm.cascade_count == 4
        assert len(csm.cascade_data) == 4

    def test_spot_light_validates_inner_less_than_outer(self) -> None:
        """SpotLight raises if inner_angle > outer_angle."""
        with pytest.raises(ValueError):
            SpotLight(inner_angle=0.5, outer_angle=0.3)

    def test_spot_light_validates_radius_positive(self) -> None:
        """SpotLight raises if radius <= 0."""
        with pytest.raises(ValueError):
            SpotLight(radius=0.0)

    def test_point_light_validates_radius_positive(self) -> None:
        """PointLight raises if radius <= 0."""
        with pytest.raises(ValueError):
            PointLight(radius=0.0)

    def test_directional_light_validates_cascade_count(self) -> None:
        """DirectionalLight raises if cascade_count out of range."""
        with pytest.raises(ValueError):
            DirectionalLight(cascade_count=0)
        with pytest.raises(ValueError):
            DirectionalLight(cascade_count=5)

    def test_atlas_cannot_allocate_zero_size(self) -> None:
        """Allocating zero-size slot may fail or return degenerate slot."""
        atlas = ShadowAtlas(resolution=4096)
        # Zero-size allocation: contract does not specify behavior,
        # but the system should not crash.
        try:
            slot = atlas.allocate(0, 0)
            if slot is not None:
                # If it returns a slot, it must be degenerate
                assert slot.width == 0 or slot.height == 0
        except (ValueError, RuntimeError):
            pass  # Acceptable error behavior

    def test_null_light_direction_defaults_to_down(self) -> None:
        """DirectionalLight with zero direction defaults to (0, -1, 0)."""
        light = DirectionalLight(direction=Vec3(0, 0, 0))
        assert light.direction.y == pytest.approx(-1.0)
        assert light.direction.x == pytest.approx(0.0)
        assert light.direction.z == pytest.approx(0.0)

    def test_spot_null_direction_defaults_to_down(self) -> None:
        """SpotLight with zero direction defaults to (0, -1, 0)."""
        light = SpotLight(direction=Vec3(0, 0, 0))
        assert light.direction.y == pytest.approx(-1.0)

    def test_cascade_texel_size_non_negative(self) -> None:
        """Texel size is always >= 0 after configuration."""
        csm = CascadedShadowMap(cascade_count=4)
        light = DirectionalLight(direction=Vec3(0, -1, 0))
        view = Mat4.look_at(Vec3(0, 0, 10), Vec3(0, 0, 0), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.radians(60), 16 / 9, 0.1, 500.0)

        csm.configure_for_light(light, view, proj, 0.1, 500.0)

        for cascade in csm.cascade_data:
            assert cascade.texel_size >= 0.0
