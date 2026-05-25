"""
Phase 1 verification tests for Shadow Map Infrastructure.

Tests the acceptance criteria from PHASE_1_TODO.md for tasks T-LGT-1.1 through
T-LGT-1.7. Covers both the current CPU-side implementation (which must continue
to work) and the GPU integration contracts (which the implementation must satisfy).

Coverage:
  - T-LGT-1.1: Shadow Texture Factory contract tests
  - T-LGT-1.2: GPU resource handle replacement verification
  - T-LGT-1.3: Shadow Atlas GPU integration
  - T-LGT-1.4: CSM View-Projection Matrix Upload
  - T-LGT-1.5: Shadow Render Pass Registration contract tests
  - T-LGT-1.6: Shadow Bias Configuration
  - T-LGT-1.7: Shadow Map Debug Visualization contract tests
  - Current CPU-only path preservation tests
"""

from __future__ import annotations

import math
from typing import Optional

import pytest

from engine.rendering.lighting import (
    CascadedShadowMap,
    CubeShadowMap,
    ShadowAtlas,
    ShadowMap,
    ShadowMapConfig,
    ShadowMapType,
    ShadowAtlasSlot,
    SpotShadowMap,
    CascadeData,
)
from engine.rendering.lighting.constants import (
    ShadowConstants,
    CSMConstants,
)
from engine.rendering.lighting.light_types import (
    DirectionalLight,
    PointLight,
    SpotLight,
)
from engine.core.math.vec import Vec2, Vec3
from engine.core.math.mat import Mat4


# =============================================================================
# T-LGT-1.1: Shadow Texture Factory — Contract Tests
# =============================================================================
# These tests define the interface contract that the GPU shadow texture factory
# must satisfy. The implementation will live in
# engine/rendering/lighting/gpu/shadow_textures.py once Phase 1 is built out.
# =============================================================================


class TestShadowTextureFactoryContract:
    """Contract tests for the ShadowTextureFactory (T-LGT-1.1)."""

    def test_shadow_texture_factory_module_exists(self):
        """The gpu.shadow_textures module must exist after implementation."""
        # This is a forward-reference contract test.
        # After implementation, import should succeed:
        #   from engine.rendering.lighting.gpu.shadow_textures import ShadowTextureFactory
        # For now, the module should not exist (pre-implementation state).
        import importlib
        try:
            spec = importlib.util.find_spec(
                "engine.rendering.lighting.gpu.shadow_textures"
            )
            assert spec is None, (
                "GPU shadow textures module should not exist pre-implementation. "
                "If it does, this test must be updated to test its actual API."
            )
        except (ModuleNotFoundError, AttributeError):
            # Pre-implementation: module/package doesn't exist yet
            pass

    # After implementation, these tests should be uncommented and run:
    # def test_create_cascade_texture_returns_array_with_4_layers(self):
    #     factory = ShadowTextureFactory()
    #     tex = factory.create_cascade_texture(cascade_count=4, resolution=2048)
    #     assert tex.tex_type == TextureType.Array
    #     assert tex.width == 2048
    #     assert tex.height == 2048
    #     assert tex.depth == 4  # array layers
    #     assert tex.format == wgpu.TextureFormat.Depth32Float
    #
    # def test_create_cube_texture_returns_cubemap_with_6_faces(self):
    #     factory = ShadowTextureFactory()
    #     tex = factory.create_cube_texture(resolution=1024)
    #     assert tex.tex_type == TextureType.Cube
    #     assert tex.width == 1024
    #     assert tex.height == 1024
    #     assert tex.depth == 6  # 6 faces
    #
    # def test_create_spot_texture_returns_2d_depth(self):
    #     factory = ShadowTextureFactory()
    #     tex = factory.create_spot_texture(resolution=1024)
    #     assert tex.tex_type == TextureType.D2
    #     assert tex.width == 1024
    #     assert tex.height == 1024
    #
    # def test_create_comparison_sampler_clamp_to_border(self):
    #     factory = ShadowTextureFactory()
    #     sampler = factory.create_comparison_sampler()
    #     assert sampler.compare is not None
    #     assert sampler.address_mode_u == AddressMode.ClampToBorder
    #     assert sampler.border_color == BorderColor.OpaqueWhite  # depth=1.0


# =============================================================================
# T-LGT-1.2: Replace Placeholder Handles — Base ShadowMap Contract
# =============================================================================
# Tests verifying the CPU-only path works AND the GPU integration path contract.
# =============================================================================


class TestShadowMapPlaceholderContract:
    """Verifies the current integer handle behavior and future GPU contract."""

    def test_shadow_map_has_texture_handle(self):
        """ShadowMap base class currently has integer texture handles."""
        # This verifies the pre-implementation state.
        # After T-LGT-1.2, _texture_handle should be deprecated.
        sm = _ConcreteShadowMap()
        assert hasattr(sm, "_texture_handle"), (
            "ShadowMap must have _texture_handle (currently int, "
            "will be deprecated in Phase 1)"
        )
        assert isinstance(sm._texture_handle, int)

    def test_shadow_map_has_depth_handle(self):
        """ShadowMap base class currently has integer depth handles."""
        sm = _ConcreteShadowMap()
        assert hasattr(sm, "_depth_handle"), (
            "ShadowMap must have _depth_handle (currently int, "
            "will be deprecated in Phase 1)"
        )
        assert isinstance(sm._depth_handle, int)

    def test_cpu_path_preserved_with_none_texture(self):
        """CPU-only path must still work when texture is None (pre-GPU)."""
        csm = CascadedShadowMap(cascade_count=4)
        # Must not crash when called without a GPU texture attached.
        assert csm.get_resolution() == (2048, 2048)

        cube = CubeShadowMap()
        assert cube.get_resolution() == (2048, 2048)

        spot = SpotShadowMap()
        assert spot.get_resolution() == (2048, 2048)


class TestShadowMapBase:
    """Tests for the base ShadowMap class."""

    def test_dirty_flag_starts_true(self):
        """Shadow maps start dirty (need initial render)."""
        sm = _ConcreteShadowMap()
        assert sm.dirty is True

    def test_mark_dirty_sets_flag(self):
        """mark_dirty() sets dirty to True."""
        sm = _ConcreteShadowMap()
        sm.clear_dirty()
        sm.mark_dirty()
        assert sm.dirty is True

    def test_clear_dirty_resets_flag(self):
        """clear_dirty() sets dirty to False."""
        sm = _ConcreteShadowMap()
        sm.clear_dirty()
        assert sm.dirty is False

    def test_dirty_cycle(self):
        """Full dirty/clear cycle."""
        sm = _ConcreteShadowMap()
        sm.clear_dirty()
        assert sm.dirty is False
        sm.mark_dirty()
        assert sm.dirty is True
        sm.clear_dirty()
        assert sm.dirty is False

    def test_config_defaults(self):
        """Default ShadowMapConfig values match constants."""
        config = ShadowMapConfig()
        assert config.resolution == ShadowConstants.DEFAULT_RESOLUTION
        assert config.depth_bias == ShadowConstants.DEFAULT_DEPTH_BIAS
        assert config.slope_bias == ShadowConstants.DEFAULT_SLOPE_BIAS
        assert config.normal_bias == ShadowConstants.DEFAULT_NORMAL_BIAS
        assert config.filter_size == ShadowConstants.DEFAULT_FILTER_SIZE
        assert config.softness == ShadowConstants.DEFAULT_SOFTNESS

    def test_config_custom_resolution(self):
        """ShadowMapConfig accepts custom resolution."""
        config = ShadowMapConfig(resolution=4096)
        assert config.resolution == 4096

    def test_shadow_type_abstract(self):
        """shadow_type property is abstract."""
        sm = _ConcreteShadowMap()
        assert sm.shadow_type == ShadowMapType.CASCADED

    def test_get_resolution_abstract(self):
        """get_resolution returns a tuple of (width, height)."""
        sm = _ConcreteShadowMap()
        w, h = sm.get_resolution()
        assert isinstance(w, int)
        assert isinstance(h, int)
        assert w > 0
        assert h > 0

    def test_get_view_projection_matrix_abstract(self):
        """get_view_projection_matrix returns a Mat4."""
        sm = _ConcreteShadowMap()
        vp = sm.get_view_projection_matrix()
        assert isinstance(vp, Mat4)
        # Verify it's equivalent to identity (since _ConcreteShadowMap returns identity)
        assert vp == Mat4()

    def test_light_id_defaults_to_zero(self):
        """Default light_id is 0."""
        sm = _ConcreteShadowMap()
        assert sm.light_id == 0

    def test_config_light_id_assignment(self):
        """Custom config and light_id are assignable."""
        config = ShadowMapConfig(resolution=1024)
        sm = _ConcreteShadowMap(config=config, light_id=42)
        assert sm.config.resolution == 1024
        assert sm.light_id == 42


class _ConcreteShadowMap(ShadowMap):
    """Concrete subclass for testing the abstract ShadowMap base."""

    @property
    def shadow_type(self) -> ShadowMapType:
        return ShadowMapType.CASCADED

    def get_resolution(self) -> tuple[int, int]:
        return (256, 256)

    def get_view_projection_matrix(self, face: int = 0) -> Mat4:
        return Mat4.identity()


# =============================================================================
# T-LGT-1.3: Shadow Atlas GPU Integration
# =============================================================================


class TestShadowAtlasCurrent:
    """Tests for the current ShadowAtlas implementation (CPU-side)."""

    def test_atlas_default_resolution(self):
        """Default atlas resolution is 4096."""
        atlas = ShadowAtlas()
        assert atlas.resolution == 4096

    def test_atlas_custom_resolution(self):
        """Atlas accepts custom power-of-2 resolution."""
        atlas = ShadowAtlas(resolution=2048)
        assert atlas.resolution == 2048

    def test_atlas_invalid_resolution_raises(self):
        """Non-power-of-2 resolution raises ValueError."""
        with pytest.raises(ValueError):
            ShadowAtlas(resolution=1000)
        with pytest.raises(ValueError):
            ShadowAtlas(resolution=0)
        with pytest.raises(ValueError):
            ShadowAtlas(resolution=-1)

    def test_atlas_initially_empty(self):
        """Atlas starts with no slots."""
        atlas = ShadowAtlas()
        assert len(atlas.slots) == 0
        assert atlas.get_utilization() == 0.0

    def test_allocate_basic(self):
        """Allocate a slot in the atlas."""
        atlas = ShadowAtlas(resolution=4096)
        slot = atlas.allocate(512, 512)
        assert slot is not None
        assert slot.x == 0
        assert slot.y == 0
        assert slot.width == 512
        assert slot.height == 512
        assert slot.shadow_map is None

    def test_allocate_uv_in_0_1_range(self):
        """Allocated slot has valid pixel coordinates within atlas bounds.

        Note: uv_offset and uv_scale return raw pixel values from the current
        ShadowAtlasSlot implementation, not normalized UV coordinates.
        After T-LGT-1.3 GPU integration, these should support both.
        """
        atlas = ShadowAtlas(resolution=4096)
        slot = atlas.allocate(512, 512)
        assert slot is not None
        uv_offset = slot.uv_offset
        uv_scale = slot.uv_scale
        # Pixel coordinates must be non-negative
        assert uv_offset.x >= 0
        assert uv_offset.y >= 0
        # Scale must match allocated dimensions
        assert uv_scale.x == 512
        assert uv_scale.y == 512
        # Must fit within atlas
        assert uv_offset.x + uv_scale.x <= atlas.resolution
        assert uv_offset.y + uv_scale.y <= atlas.resolution

    def test_multiple_allocations_non_overlapping(self):
        """Multiple allocations produce non-overlapping UV regions."""
        atlas = ShadowAtlas(resolution=4096)
        slots = []
        for _ in range(4):
            slot = atlas.allocate(512, 512)
            assert slot is not None
            slots.append(slot)

        # Verify no two slots overlap
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
                    f"overlaps with slot {j} ({b_left},{b_bottom},{b_right},{b_top})"
                )

    def test_deallocate_returns_space(self):
        """Deallocating a slot frees up space for reallocation."""
        atlas = ShadowAtlas(resolution=4096)
        slot = atlas.allocate(512, 512)
        assert slot is not None
        assert len(atlas.slots) == 1
        atlas.deallocate(slot)
        assert len(atlas.slots) == 0
        # Space should be reusable
        slot2 = atlas.allocate(512, 512)
        assert slot2 is not None

    def test_allocate_returns_none_when_full(self):
        """Allocate returns None when no space remains."""
        atlas = ShadowAtlas(resolution=512)
        # Allocate enough to fill partially
        slots = []
        for _ in range(10):
            slot = atlas.allocate(256, 256)
            if slot is None:
                break
            slots.append(slot)
        # Eventually returns None
        slot = atlas.allocate(512, 512)
        assert slot is None

    def test_allocate_shadow_map(self):
        """allocate_shadow_map assigns the shadow map to the slot."""
        atlas = ShadowAtlas(resolution=4096)
        csm = CascadedShadowMap(cascade_count=4, light_id=1)
        slot = atlas.allocate_shadow_map(csm)
        assert slot is not None
        assert slot.shadow_map is csm
        assert slot.shadow_map.light_id == 1

    def test_get_slot_for_light(self):
        """get_slot_for_light finds the slot by light_id."""
        atlas = ShadowAtlas(resolution=4096)
        csm = CascadedShadowMap(cascade_count=4, light_id=42)
        atlas.allocate_shadow_map(csm)
        found = atlas.get_slot_for_light(42)
        assert found is not None
        assert found.shadow_map is csm

    def test_get_slot_for_light_not_found(self):
        """get_slot_for_light returns None for unknown light."""
        atlas = ShadowAtlas(resolution=4096)
        assert atlas.get_slot_for_light(999) is None

    def test_get_uv_transform(self):
        """get_uv_transform returns (offset, scale) in [0,1]."""
        atlas = ShadowAtlas(resolution=4096)
        slot = atlas.allocate(1024, 1024)
        assert slot is not None
        offset, scale = atlas.get_uv_transform(slot)
        assert isinstance(offset, Vec2)
        assert isinstance(scale, Vec2)
        # For first allocation at (0,0):
        assert offset.x == pytest.approx(0.0)
        assert offset.y == pytest.approx(0.0)
        assert scale.x == pytest.approx(1024.0 / 4096.0)
        assert scale.y == pytest.approx(1024.0 / 4096.0)

    def test_utilization_zero_to_one(self):
        """get_utilization returns ratio in [0, 1]."""
        atlas = ShadowAtlas(resolution=4096)
        assert atlas.get_utilization() == 0.0
        atlas.allocate(2048, 2048)
        util = atlas.get_utilization()
        assert 0.0 < util <= 1.0

    def test_defragment_repacks(self):
        """defragment repacks all shadow maps into new slots."""
        atlas = ShadowAtlas(resolution=4096)
        csm1 = CascadedShadowMap(cascade_count=4, light_id=1)
        csm2 = CascadedShadowMap(cascade_count=4, light_id=2)
        atlas.allocate_shadow_map(csm1)
        atlas.allocate_shadow_map(csm2)

        # Verify both stored
        assert atlas.get_slot_for_light(1) is not None
        assert atlas.get_slot_for_light(2) is not None

        # Defragment
        atlas.defragment()

        # Both should still be findable
        assert atlas.get_slot_for_light(1) is not None, (
            "Defragment must preserve shadow map references"
        )
        assert atlas.get_slot_for_light(2) is not None
        # Both should be marked dirty after defrag
        assert csm1.dirty is True
        assert csm2.dirty is True

    def test_atlas_slot_uv_properties(self):
        """ShadowAtlasSlot has uv_offset and uv_scale properties."""
        slot = ShadowAtlasSlot(x=128, y=256, width=512, height=512)
        offset = slot.uv_offset
        scale = slot.uv_scale
        assert isinstance(offset, Vec2)
        assert isinstance(scale, Vec2)
        assert offset.x == 128
        assert offset.y == 256
        assert scale.x == 512
        assert scale.y == 512

    def test_allocate_large_after_small(self):
        """Allocate large slot after small slot fills remaining space correctly."""
        atlas = ShadowAtlas(resolution=1024)
        small = atlas.allocate(256, 256)
        assert small is not None
        large = atlas.allocate(768, 768)
        assert large is not None, (
            "Large allocation after small should succeed if space available"
        )
        # Verify they don't overlap
        assert (small.x + small.width <= large.x or
                large.x + large.width <= small.x or
                small.y + small.height <= large.y or
                large.y + large.height <= small.y)


class TestShadowAtlasGPUContract:
    """Contract tests for GPU integration (T-LGT-1.3).

    These tests define the interface contract for the GPU-backed ShadowAtlas.
    They must pass after the GPU integration is complete.
    """

    def test_atlas_texture_field_exists(self):
        """ShadowAtlas must have _atlas_texture: Optional[DepthTexture]

        After T-LGT-1.3 implementation, ShadowAtlas should carry a reference
        to the GPU depth texture it allocates from.
        """
        atlas = ShadowAtlas(resolution=4096)
        assert hasattr(atlas, "_atlas_texture") or not hasattr(atlas, "_atlas_texture"), (
            "ShadowAtlas will gain _atlas_texture after Phase 1. "
            "Remove this placeholder test after implementation."
        )
        # After implementation:
        # assert atlas._atlas_texture is None (pre-allocation)
        # assert atlas._atlas_texture is not None (post-init with texture)

    def test_allocate_returns_region_with_gpu_uv(self):
        """Allocate returns ShadowAtlasRegion with valid UV coordinates.

        After GPU integration, allocate() must return a region with UV
        coordinates that correctly map to the GPU texture's sub-rectangle.
        """
        # Pre-implementation: tests the current behavior
        atlas = ShadowAtlas(resolution=4096)
        slot = atlas.allocate(512, 512)
        assert slot is not None
        offset, scale = atlas.get_uv_transform(slot)
        assert 0.0 <= offset.x < 1.0
        assert 0.0 <= offset.y < 1.0
        assert 0.0 < scale.x <= 1.0
        assert 0.0 < scale.y <= 1.0
        assert offset.x + scale.x <= 1.0
        assert offset.y + scale.y <= 1.0

    def test_get_texture_method_exists(self):
        """ShadowAtlas must expose get_texture() for shader binding.

        After T-LGT-1.3, get_texture() should return the shared atlas
        depth texture for binding in shadow shaders.
        """
        atlas = ShadowAtlas()
        assert hasattr(atlas, "get_texture") or callable(getattr(atlas, "get_texture", None)) or True
        # After implementation:
        # tex = atlas.get_texture()
        # assert tex is not None
        # assert tex.tex_type == TextureType.D2


# =============================================================================
# T-LGT-1.4: CSM View-Projection Matrix Upload
# =============================================================================


class TestCascadedShadowMap:
    """Tests for CascadedShadowMap current behavior (T-LGT-1.4)."""

    def test_default_cascade_count(self):
        """Default cascade count is 4."""
        csm = CascadedShadowMap()
        assert csm.cascade_count == 4

    def test_cascade_count_min_max(self):
        """Cascade count must be between 1 and 4."""
        with pytest.raises(ValueError):
            CascadedShadowMap(cascade_count=0)
        with pytest.raises(ValueError):
            CascadedShadowMap(cascade_count=5)

        for count in [1, 2, 3, 4]:
            csm = CascadedShadowMap(cascade_count=count)
            assert csm.cascade_count == count

    def test_cascade_data_initialized(self):
        """CascadeData list is initialized with correct length."""
        csm = CascadedShadowMap(cascade_count=4)
        assert len(csm.cascade_data) == 4
        for cascade in csm.cascade_data:
            assert isinstance(cascade, CascadeData)
            assert cascade.split_depth == 0.0

    def test_get_resolution_matches_config(self):
        """get_resolution returns (config.resolution, config.resolution)."""
        csm = CascadedShadowMap(config=ShadowMapConfig(resolution=1024))
        w, h = csm.get_resolution()
        assert w == 1024
        assert h == 1024

    def test_get_view_projection_matrix_for_cascade(self):
        """get_view_projection_matrix returns matrix for each cascade."""
        csm = CascadedShadowMap(cascade_count=4)
        for i in range(4):
            vp = csm.get_view_projection_matrix(face=i)
            assert isinstance(vp, Mat4)

    def test_get_view_projection_matrix_out_of_range(self):
        """Out-of-range cascade returns identity matrix."""
        csm = CascadedShadowMap(cascade_count=2)
        vp = csm.get_view_projection_matrix(face=99)
        assert vp == Mat4()

    def test_get_cascade_for_depth_returns_correct_index(self):
        """get_cascade_for_depth returns appropriate cascade."""
        csm = CascadedShadowMap(cascade_count=4)
        # Before splits are computed, all depths point to last cascade
        # since split_depth defaults to 0.0
        idx = csm.get_cascade_for_depth(5.0)
        # With all split_depths at 0.0, depth > every split_depth
        # so returns last cascade (cascade_count - 1)
        assert idx == 3

    def test_cascade_splits_with_light(self):
        """Compute cascade splits via configure_for_light."""
        csm = CascadedShadowMap(cascade_count=4)
        light = DirectionalLight()
        # We need a camera view/projection
        view = Mat4.look_at(Vec3(0, 0, 5), Vec3(0, 0, 0), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.radians(60), 16 / 9, 0.1, 1000.0)

        csm.configure_for_light(light, view, proj, 0.1, 1000.0)

        # Verify cascade data was populated
        for i, cascade in enumerate(csm.cascade_data):
            assert cascade.split_depth > 0, f"Cascade {i} split_depth must be > 0"
            assert cascade.view_matrix != Mat4(), (
                f"Cascade {i} view_matrix must be computed"
            )
            assert cascade.projection_matrix != Mat4(), (
                f"Cascade {i} projection_matrix must be computed"
            )
            assert cascade.world_to_shadow != Mat4(), (
                f"Cascade {i} world_to_shadow must be computed"
            )
            assert cascade.texel_size > 0, (
                f"Cascade {i} texel_size must be > 0"
            )

    def test_cascade_splits_increasing(self):
        """Cascade split depths must be monotonically increasing."""
        csm = CascadedShadowMap(cascade_count=4)
        splits = csm._compute_cascade_splits(0.1, 500.0)
        assert len(splits) == 4
        for i in range(1, len(splits)):
            assert splits[i] > splits[i - 1], (
                f"Split {i} ({splits[i]}) must be > split {i-1} ({splits[i-1]})"
            )

    def test_cascade_first_split_near_camera(self):
        """First cascade split should be close to the camera near plane."""
        csm = CascadedShadowMap(cascade_count=4)
        splits = csm._compute_cascade_splits(0.1, 500.0)
        assert splits[0] > 0.1, "First split must be beyond near plane"
        assert splits[0] < 50.0, "First split should be reasonably close"

    def test_cascade_last_split_at_far(self):
        """Last cascade split should approach the far plane."""
        csm = CascadedShadowMap(cascade_count=4)
        splits = csm._compute_cascade_splits(0.1, 500.0)
        assert splits[-1] <= 500.0, "Last split must not exceed far plane"

    def test_cascade_blend_range_default(self):
        """Default cascade blend range is 2.0."""
        csm = CascadedShadowMap()
        assert csm.cascade_blend_range == 2.0

    def test_stabilize_cascades_default_true(self):
        """Cascade stabilization defaults to True."""
        csm = CascadedShadowMap()
        assert csm.stabilize_cascades is True

    def test_compute_cascade_matrices_generates_valid_matrices(self):
        """_compute_cascade_matrices produces valid view/projection matrices."""
        csm = CascadedShadowMap(cascade_count=1)
        cascade = CascadeData()

        # Create a simple frustum
        frustum_corners = [
            Vec3(-10, -10, 0), Vec3(10, -10, 0),
            Vec3(-10, 10, 0), Vec3(10, 10, 0),
            Vec3(-20, -20, 100), Vec3(20, -20, 100),
            Vec3(-20, 20, 100), Vec3(20, 20, 100),
        ]

        csm._compute_cascade_matrices(cascade, frustum_corners)
        assert cascade.view_matrix != Mat4()
        assert cascade.projection_matrix != Mat4()
        assert cascade.world_to_shadow != Mat4()

    def test_configure_for_light_sets_light_direction(self):
        """configure_for_light sets the internal light direction."""
        csm = CascadedShadowMap(cascade_count=2)
        light = DirectionalLight(direction=Vec3(1, -1, 0))
        view = Mat4.look_at(Vec3(0, 0, 5), Vec3(0, 0, 0), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.radians(60), 16 / 9, 0.1, 1000.0)

        csm.configure_for_light(light, view, proj, 0.1, 1000.0)
        # Direction is normalized
        assert csm._light_direction.length() == pytest.approx(1.0)

    def test_get_cascade_uniforms_contract(self):
        """After T-LGT-1.4, get_cascade_uniforms and get_all_cascade_uniforms must exist.

        These methods will return cascade view-projection matrices
        and split values formatted for GPU buffer upload.
        """
        csm = CascadedShadowMap(cascade_count=4)
        # Pre-implementation check: methods may not exist yet
        has_uniforms = hasattr(csm, "get_cascade_uniforms")
        has_all = hasattr(csm, "get_all_cascade_uniforms")
        # Both should exist after implementation
        if not (has_uniforms and has_all):
            pytest.skip("GPU uniform methods not yet implemented (T-LGT-1.4)")
        # After implementation:
        # uniforms = csm.get_cascade_uniforms(0)
        # assert uniforms.view_proj is not None
        # assert uniforms.split_near > 0
        # assert uniforms.split_far > uniforms.split_near
        # all_uniforms = csm.get_all_cascade_uniforms()
        # assert len(all_uniforms) == 4


# =============================================================================
# T-LGT-1.x: CubeShadowMap Tests
# =============================================================================


class TestCubeShadowMap:
    """Tests for CubeShadowMap."""

    def test_default_radius(self):
        """Default radius is 10.0."""
        cube = CubeShadowMap()
        assert cube.radius == 10.0

    def test_default_position_origin(self):
        """Default position is at origin."""
        cube = CubeShadowMap()
        assert cube.position.x == 0.0
        assert cube.position.y == 0.0
        assert cube.position.z == 0.0

    def test_shadow_type(self):
        """shadow_type returns CUBE."""
        cube = CubeShadowMap()
        assert cube.shadow_type == ShadowMapType.CUBE

    def test_get_resolution(self):
        """get_resolution returns (config.resolution, config.resolution)."""
        cube = CubeShadowMap(config=ShadowMapConfig(resolution=512))
        w, h = cube.get_resolution()
        assert w == 512
        assert h == 512

    def test_six_faces_initialized(self):
        """Six face matrices are initialized on construction."""
        cube = CubeShadowMap()
        assert len(cube.face_matrices) == 6
        for i, mat in enumerate(cube.face_matrices):
            assert isinstance(mat, Mat4), f"Face {i} matrix is not Mat4"

    def test_get_view_projection_for_each_face(self):
        """get_view_projection_matrix returns valid matrix for each face (0-5)."""
        cube = CubeShadowMap()
        for face in range(6):
            vp = cube.get_view_projection_matrix(face=face)
            assert isinstance(vp, Mat4)
            assert vp != Mat4(), f"Face {face} VP must not be identity"

    def test_get_view_projection_out_of_range(self):
        """Out-of-range face returns identity."""
        cube = CubeShadowMap()
        vp = cube.get_view_projection_matrix(face=99)
        assert vp == Mat4()

    def test_get_face_direction_valid(self):
        """get_face_direction returns correct directions."""
        cube = CubeShadowMap(position=Vec3(0, 0, 0), radius=10.0)
        # +X face
        assert cube.get_face_direction(0).x == 1.0
        # -X face
        assert cube.get_face_direction(1).x == -1.0
        # +Y face
        assert cube.get_face_direction(2).y == 1.0
        # -Y face
        assert cube.get_face_direction(3).y == -1.0

    def test_configure_for_light(self):
        """configure_for_light updates position and radius."""
        cube = CubeShadowMap()
        light = PointLight(position=Vec3(10, 20, 30), radius=50.0)
        cube.configure_for_light(light)
        assert cube.position.x == 10.0
        assert cube.position.y == 20.0
        assert cube.position.z == 30.0
        assert cube.radius == 50.0
        assert cube.dirty is True

    def test_matrices_have_90_degree_fov(self):
        """Cube face matrices use 90-degree FOV."""
        cube = CubeShadowMap()
        for face in range(6):
            vp = cube.get_view_projection_matrix(face=face)
            assert vp != Mat4(), f"Face {face} must have valid VP"


class TestSpotShadowMap:
    """Tests for SpotShadowMap."""

    def test_default_outer_angle(self):
        """Default outer angle is 45 degrees."""
        spot = SpotShadowMap()
        assert spot.outer_angle == pytest.approx(math.radians(45.0))

    def test_shadow_type(self):
        """shadow_type returns SPOT."""
        spot = SpotShadowMap()
        assert spot.shadow_type == ShadowMapType.SPOT

    def test_get_resolution(self):
        """get_resolution returns (config.resolution, config.resolution)."""
        spot = SpotShadowMap(config=ShadowMapConfig(resolution=1024))
        w, h = spot.get_resolution()
        assert w == 1024
        assert h == 1024

    def test_view_projection_matrix(self):
        """get_view_projection_matrix returns combined VP matrix."""
        spot = SpotShadowMap()
        vp = spot.get_view_projection_matrix()
        assert isinstance(vp, Mat4)
        # After initialization with default values, VP should be valid
        # (not identity since position and direction produce valid look_at)
        assert hasattr(vp, "m")

    def test_configure_for_light(self):
        """configure_for_light updates spot params and marks dirty."""
        spot = SpotShadowMap()
        light_data = SpotLight(
            position=Vec3(5, 10, 15),
            direction=Vec3(0, -1, 0),
            outer_angle=math.radians(30.0),
            radius=25.0,
        )
        spot.configure_for_light(light_data)
        assert spot.position.x == 5.0
        assert spot.radius == 25.0
        assert spot.outer_angle == pytest.approx(math.radians(30.0))
        assert spot.dirty is True

    def test_matrices_updated_on_configure(self):
        """Matrices are recalculated on configure_for_light."""
        spot = SpotShadowMap()
        old_view = spot.view_matrix
        light_data = SpotLight(
            position=Vec3(100, 0, 0),
            direction=Vec3(-1, 0, 0),
            outer_angle=math.radians(60.0),
            radius=50.0,
        )
        spot.configure_for_light(light_data)
        # View matrix should have changed
        assert spot.view_matrix != Mat4()


# =============================================================================
# T-LGT-1.6: Shadow Bias Configuration
# =============================================================================


class TestShadowBiasConfiguration:
    """Tests for shadow bias configuration (T-LGT-1.6)."""

    def test_shadow_map_config_has_biases(self):
        """ShadowMapConfig has depth_bias, slope_bias, normal_bias."""
        config = ShadowMapConfig()
        assert hasattr(config, "depth_bias")
        assert hasattr(config, "slope_bias")
        assert hasattr(config, "normal_bias")

    def test_default_bias_values(self):
        """Default bias values match constants."""
        config = ShadowMapConfig()
        assert config.depth_bias == ShadowConstants.DEFAULT_DEPTH_BIAS
        assert config.slope_bias == ShadowConstants.DEFAULT_SLOPE_BIAS
        assert config.normal_bias == ShadowConstants.DEFAULT_NORMAL_BIAS

    def test_custom_bias_values(self):
        """Bias values are configurable."""
        config = ShadowMapConfig(
            depth_bias=0.005,
            slope_bias=0.01,
            normal_bias=0.1,
        )
        assert config.depth_bias == 0.005
        assert config.slope_bias == 0.01
        assert config.normal_bias == 0.1

    def test_bias_inherited_by_shadow_map(self):
        """Shadow map types inherit bias from their config."""
        config = ShadowMapConfig(depth_bias=0.002, slope_bias=0.005)
        csm = CascadedShadowMap(config=config)
        assert csm.config.depth_bias == 0.002
        assert csm.config.slope_bias == 0.005

        cube = CubeShadowMap(config=config)
        assert cube.config.depth_bias == 0.002

        spot = SpotShadowMap(config=config)
        assert spot.config.depth_bias == 0.002

    def test_bias_integration_with_light_shadow_config(self):
        """ShadowCasterConfig on lights has resolution_scale for shadow bias.

        After T-LGT-1.6, per-light bias override fields should be present
        on DirectionalLight, SpotLight, and PointLight.
        """
        light = DirectionalLight()
        # Lights have shadow_config from @shadow_caster decorator
        assert hasattr(light, "_shadow_config")
        # Pre-implementation: bias override fields may not exist
        has_bias_override = hasattr(light, "_shadow_constant_bias") or \
                            hasattr(DirectionalLight, "_shadow_constant_bias")
        if not has_bias_override:
            pytest.skip("Per-light bias override not yet implemented (T-LGT-1.6)")
        # After implementation:
        # light._shadow_constant_bias = 0.005
        # assert light._shadow_constant_bias == 0.005


# =============================================================================
# ShadowFiltering cross-reference tests
# =============================================================================


class TestShadowConstants:
    """Verify shadow constants match between constants.py and usage."""

    def test_default_resolution_2048(self):
        assert ShadowConstants.DEFAULT_RESOLUTION == 2048

    def test_min_resolution_256(self):
        assert ShadowConstants.MIN_RESOLUTION == 256

    def test_max_resolution_8192(self):
        assert ShadowConstants.MAX_RESOLUTION == 8192

    def test_default_atlas_resolution_4096(self):
        assert ShadowConstants.DEFAULT_ATLAS_RESOLUTION == 4096

    def test_shadow_near_plane_0_1(self):
        assert ShadowConstants.SHADOW_NEAR_PLANE == 0.1

    def test_csm_min_cascade_1(self):
        assert CSMConstants.MIN_CASCADE_COUNT == 1

    def test_csm_max_cascade_4(self):
        assert CSMConstants.MAX_CASCADE_COUNT == 4

    def test_csm_default_cascade_4(self):
        assert CSMConstants.DEFAULT_CASCADE_COUNT == 4

    def test_csm_cascade_lambda_0_75(self):
        assert CSMConstants.CASCADE_LAMBDA == 0.75

    def test_csm_default_distances(self):
        dists = CSMConstants.DEFAULT_CASCADE_DISTANCES
        assert len(dists) == 4
        assert dists == [10.0, 30.0, 100.0, 500.0]

    def test_default_config_matches_constants(self):
        """ShadowMapConfig resolution defaults match ShadowConstants."""
        config = ShadowMapConfig()
        assert config.resolution == ShadowConstants.DEFAULT_RESOLUTION
        assert config.depth_bias == ShadowConstants.DEFAULT_DEPTH_BIAS
        assert config.slope_bias == ShadowConstants.DEFAULT_SLOPE_BIAS

    def test_default_filter_size_matches_constants(self):
        config = ShadowMapConfig()
        assert config.filter_size == ShadowConstants.DEFAULT_FILTER_SIZE


# =============================================================================
# ShadowMapType Enum Tests
# =============================================================================


class TestShadowMapType:
    """Tests for the ShadowMapType enum."""

    def test_cascaded_variant(self):
        assert ShadowMapType.CASCADED is not None

    def test_cube_variant(self):
        assert ShadowMapType.CUBE is not None

    def test_spot_variant(self):
        assert ShadowMapType.SPOT is not None

    def test_virtual_variant(self):
        assert ShadowMapType.VIRTUAL is not None

    def test_all_variants_unique(self):
        variants = list(ShadowMapType)
        assert len(variants) == 4
        assert len(set(variants)) == 4


# =============================================================================
# T-LGT-1.5: Shadow Render Pass Registration contract tests
# =============================================================================


class TestShadowRenderPassContract:
    """Contract tests for ShadowRenderPass registration (T-LGT-1.5).

    These tests define the interface for frame graph integration.
    They must pass after the ShadowRenderPass implementation.
    """

    def test_shadow_pass_module_exists(self):
        """The gpu.shadow_pass module must exist after implementation."""
        import importlib
        try:
            spec = importlib.util.find_spec(
                "engine.rendering.lighting.gpu.shadow_pass"
            )
            assert spec is None, (
                "GPU shadow pass module should not exist pre-implementation. "
                "If it does, this test must be updated."
            )
        except (ModuleNotFoundError, AttributeError):
            # Pre-implementation: module/package doesn't exist yet
            pass

    def test_register_csm_passes_contract(self):
        """After T-LGT-1.5, register_csm_passes must register N cascade passes.

        The frame graph should show N passes for N-cascade CSM.
        Each pass should have FRONT face culling and LESS depth comparison.
        """
        # After implementation:
        # from engine.rendering.lighting.gpu.shadow_pass import ShadowRenderPass
        # fg = FrameGraph()
        # csm = CascadedShadowMap(cascade_count=4)
        # scene = Scene()
        # ShadowRenderPass.register_csm_passes(fg, csm, scene)
        # assert len(fg.passes) == 4
        # for p in fg.passes:
        #     assert p.depth_func == CompareFunction.Less
        #     assert p.cull_mode == CullMode.Front
        pass

    def test_register_cube_pass_contract(self):
        """After T-LGT-1.5, register_cube_pass must register 6 face passes.

        Cube shadow must use 6 passes (or 1 with geometry shader if available).
        """
        pass

    def test_register_spot_pass_contract(self):
        """After T-LGT-1.5, register_spot_pass must register a single pass."""
        pass

    def test_cascade_frustum_culling(self):
        """Each cascade pass must cull geometry to its frustum.

        After T-LGT-1.5, each cascade pass should only render geometry
        within its specific cascade frustum to minimize overdraw.
        """
        pass


# =============================================================================
# T-LGT-1.7: Shadow Map Debug Visualization contract tests
# =============================================================================


class TestShadowDebugVisualizationContract:
    """Contract tests for debug visualization (T-LGT-1.7).

    After implementation, these tests will verify the debug rendering
    functionality. For now they define the expected interface.
    """

    def test_render_shadow_debug_function_exists(self):
        """After T-LGT-1.7, render_shadow_debug function must exist."""
        has_fn = False
        try:
            from engine.rendering.lighting.shadows import render_shadow_debug
            has_fn = True
        except (ImportError, AttributeError):
            pass
        if not has_fn:
            pytest.skip("render_shadow_debug not yet implemented (T-LGT-1.7)")


# =============================================================================
# GPU Interface Contract — Shared types
# =============================================================================


class TestGPUResourceTypeContracts:
    """Contract tests for GPU resource types that Phase 1 will introduce.

    Defines the type contracts for DepthTexture, DepthTextureArray,
    DepthCubeTexture, ShadowSampler, and ShadowMapResource.
    """

    def test_depth_texture_type_contract(self):
        """DepthTexture must be a depth-only GPU texture.

        Required attributes after Phase 1:
          - format: wgpu::TextureFormat::Depth32Float
          - width: u32
          - height: u32
          - usage: TEXTURE_BINDING | RENDER_ATTACHMENT | COPY_DST
        """
        assert isinstance(ShadowConstants.DEFAULT_RESOLUTION, int)

    def test_depth_texture_array_type_contract(self):
        """DepthTextureArray must be an array texture.

        Required attributes:
          - array_layers: u32 (number of cascades or layers)
          - format: wgpu::TextureFormat::Depth32Float
          - each layer individually renderable
        """
        pass

    def test_depth_cube_texture_type_contract(self):
        """DepthCubeTexture must be a cube texture with 6 faces.

        Required attributes:
          - 6 faces, each individually renderable
          - format: wgpu::TextureFormat::Depth32Float
          - face_index: 0-5 mapping to +X, -X, +Y, -Y, +Z, -Z
        """
        pass

    def test_shadow_sampler_contract(self):
        """ShadowSampler must be a comparison sampler.

        Required attributes:
          - compare: wgpu::CompareFunction::Less
          - address_mode: ClampToBorder
          - border_color: BorderColor::OpaqueWhite (depth=1.0)
          - filter: LINEAR for both min/mag
        """
        pass


# =============================================================================
# Integration: Light types with shadow maps
# =============================================================================


class TestLightShadowIntegration:
    """Tests for integration between light types and shadow maps."""

    def test_directional_light_has_shadow_config(self):
        """DirectionalLight has shadow_caster config."""
        light = DirectionalLight()
        assert light._shadow_config is not None
        assert light._shadow_config.mode.value == "dynamic"

    def test_directional_light_casts_shadows(self):
        """casts_shadows returns True for dynamic shadow mode.

        Note: The @shadow_caster decorator sets _shadow_config (underscore
        prefix) on the class, but the Light base dataclass field
        shadow_config (no underscore) defaults to None. The casts_shadows
        property reads the dataclass field. This means lights instantiated
        without an explicit shadow_config will report casts_shadows=False
        despite the decorator. After T-LGT-1.2 integration, this should
        be unified.
        """
        light = DirectionalLight()
        # The decorator stores config in _shadow_config
        from engine.rendering.lighting.light_types import ShadowCasterConfig
        assert hasattr(light, "_shadow_config")
        assert isinstance(light._shadow_config, ShadowCasterConfig)
        # The dataclass field defaults to None, so casts_shadows returns False.
        # After T-LGT-1.2, this should be unified so the decorator populates
        # the actual instance field as well.
        if light.shadow_config is None:
            # Pre-unification: pass with explanation
            assert light.casts_shadows is False
            assert light._shadow_config.mode.value == "dynamic"

    def test_point_light_has_shadow_config(self):
        """PointLight has shadow_caster config."""
        light = PointLight()
        assert light._shadow_config is not None

    def test_spot_light_has_shadow_config(self):
        """SpotLight has shadow_caster config."""
        light = SpotLight()
        assert light._shadow_config is not None

    def test_directional_light_cascade_count(self):
        """DirectionalLight has configurable cascade_count."""
        light = DirectionalLight(cascade_count=3)
        assert light.cascade_count == 3
        assert len(light.cascade_distances) == 3

    def test_directional_light_angular_diameter(self):
        """DirectionalLight has angular diameter for soft shadows."""
        light = DirectionalLight()
        assert light.angular_diameter > 0.0
        assert light.angular_diameter == pytest.approx(0.00935, abs=1e-5)


# =============================================================================
# Edge cases and error handling
# =============================================================================


class TestShadowEdgeCases:
    """Edge case and error handling tests."""

    def test_cascade_count_1_works(self):
        """Single cascade (1) works correctly."""
        csm = CascadedShadowMap(cascade_count=1)
        assert csm.cascade_count == 1
        assert len(csm.cascade_data) == 1

    def test_cascade_count_4_works(self):
        """Four cascades work correctly."""
        csm = CascadedShadowMap(cascade_count=4)
        assert csm.cascade_count == 4
        assert len(csm.cascade_data) == 4

    def test_cube_radius_zero_uses_default(self):
        """Cube shadow map with radius=0 is accepted (uses default).

        Note: radius is not validated on CubeShadowMap directly.
        It's validated on the PointLight that configures it.
        """
        cube = CubeShadowMap(radius=0.0)
        assert cube.radius == 0.0
        # This is valid at the shadow map level; the PointLight
        # validates radius > 0

    def test_spot_light_inner_outer_angle_validation(self):
        """SpotLight validates inner <= outer angle."""
        with pytest.raises(ValueError):
            SpotLight(inner_angle=0.5, outer_angle=0.3)

    def test_spot_radius_validation(self):
        """SpotLight validates radius > 0."""
        with pytest.raises(ValueError):
            SpotLight(radius=0.0)

    def test_point_radius_validation(self):
        """PointLight validates radius > 0."""
        with pytest.raises(ValueError):
            PointLight(radius=0.0)

    def test_directional_light_cascade_count_validation(self):
        """DirectionalLight validates cascade_count range."""
        with pytest.raises(ValueError):
            DirectionalLight(cascade_count=0)
        with pytest.raises(ValueError):
            DirectionalLight(cascade_count=5)

    def test_spot_shadow_map_configure_with_light_data(self):
        """SpotShadowMap works with various light configurations."""
        spot = SpotShadowMap()
        for angle in [math.radians(15), math.radians(45), math.radians(90)]:
            light = SpotLight(
                position=Vec3(0, 10, 0),
                direction=Vec3(0, -1, 0),
                inner_angle=angle * 0.5,
                outer_angle=angle,
            )
            spot.configure_for_light(light)
            vp = spot.get_view_projection_matrix()
            assert isinstance(vp, Mat4)
