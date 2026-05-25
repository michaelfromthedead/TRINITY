"""Whitebox tests for the materials subsystem.

WHITEBOX coverage plan:
  - material_system.py:
    - MaterialParameter.clamp(): non-numeric passthrough, min-only, max-only, both bounds
    - MaterialTemplate.add_parameter(): duplicate name raises ValueError
    - MaterialTemplate.remove_parameter(): missing key raises KeyError
    - MaterialTemplate.add_function(): duplicate func is a no-op (no version bump)
    - MaterialTemplate.compute_permutation_key(): sorted features produce same key
    - MaterialInstance.set_parameter(): no-change skips dirty, clamp=False/validate=False
    - MaterialInstance.set_parameter(): TEXTURE_2D sets textures dirty, FLOAT sets parameters dirty
    - MaterialInstance.clear_override(): non-existent override is no-op
    - MaterialInstance.pop_layer(): empty stack returns None
    - MaterialInstance.clone(): overrides are independent copies
    - MaterialFunction.get_dependencies(): transitive with visited-set cycle avoidance
    - MaterialFunction.get_full_code(): dependency ordering
    - MaterialLayer.enabled setter: toggling
    - MaterialSystem.unregister_template(): missing template raises KeyError
    - MaterialSystem.get_template_by_name(): not found returns None
    - MaterialSystem.get_dirty_instances(): instance removed mid-iteration, non-dirty cleanup
    - MaterialSystem.clear_dirty_flags(): instance removed during cleanup
    - MaterialSystem.create_instance(): string template with fallback to get_template_by_name
  - pbr_model.py:
    - PBRParameters.validate(): multiple errors collected; emissive below-zero rejected
    - PBRParameters.clamp(): all fields clamped
    - PBRParameters.lerp(): t clamped to [0,1]
    - PBRTextureSet.has_any_texture(): all None returns False
    - PBRTextureSet.get_texture_paths(): all None returns empty list
    - PBRDirtyFlags: each property setter marks _all_dirty
    - PBRMaterial: each setter no-change optimization, callback notification
    - validate_pbr_parameter(): unknown param, Vec4/Vec3/float type branches
    - clamp_pbr_parameter(): unknown param, Vec4/Vec3/float/other isinstance branches
  - material_functions.py:
    - MaterialFunctionLibrary singleton: __new__ returns same instance
    - MaterialFunctionLibrary.get_by_category(): case-insensitive partial match
    - Each create_*_function(): verify input/output port counts and types
  - material_graph.py:
    - NodePort.is_compatible_with(): 4 branches
    - ConstantNode.generate_code(): FLOAT, VEC2, VEC3, VEC4, unknown branches
    - ComponentMask._define_ports(): 1,2,3,4 component branches
    - MaterialGraph.add_node(): duplicate OutputNode raises GraphValidationError
    - MaterialGraph.remove_node(): not found is no-op, connection cleanup
    - MaterialGraph.connect(): all 6 error paths
    - MaterialGraph._has_cycle(): cycle and no-cycle detection
    - MaterialGraph.get_topological_order(): with dependencies
    - GraphCompiler.compile(): validation failure raises GraphValidationError
    - GraphCompiler._default_value(): all DataType branches
    - GraphCompiler._format_value(): Vec2/Vec3/Vec4/bool/float/int/other branches
  - shader_compiler.py:
    - ShaderDefine.to_string(): with and without value
    - ShaderSource.from_file(): FileNotFoundError, language auto-detect
    - ShaderSource.has_changed(): no-path returns False, OSError returns False
    - ShaderSource.reload(): unchanged returns False
    - ShaderSource.get_content_hash(): caching behavior
    - PermutationKey.to_defines(): sorting
    - ShaderPermutation.validate_key(): missing required, unknown, conflicts
    - ShaderPermutation.get_valid_keys(): with conflict filtering
    - PSODescriptor.is_compute_pipeline(): both branches
    - PSOCache.get(): LRU order update
    - PSOCache.put(): eviction of oldest
    - HotReloadWatcher.watch(): OSError handling
    - HotReloadWatcher.check_changes(): poll interval gate, OSError handling
    - ShaderCompiler.compile(): cache hit with matching hash, permutation defines
    - ShaderCompiler.invalidate(): all vs specific permutation
  - advanced_models.py:
    - SubsurfaceProfile.get_diffusion_profile(): zero total normalization edge case
    - SubsurfaceScattering: opacity clamping, dirty flag on each setter
    - ClearCoat.validate(): error collection
    - ClearCoat.to_shader_data(): F0 calculation
    - Anisotropy.get_anisotropic_roughness(): strength >=0 and <0 branches
    - Iridescence.__post_init__(): thickness_max clamped >= thickness_min
    - Iridescence.get_interference_color(): verify output
    - Transmission.get_fresnel_mix(): Schlick approximation
    - Transmission.get_attenuation(): infinite distance early-return
    - AdvancedShadingModel.get_active_models(): all None, mixed activation
"""
from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from engine.core.math.vec import Vec2, Vec3, Vec4
from engine.rendering.materials.advanced_models import (
    AdvancedShadingModel,
    Anisotropy,
    ClearCoat,
    Iridescence,
    ShadingModelType,
    Sheen,
    SubsurfaceProfile,
    SubsurfaceScattering,
    Transmission,
)
from engine.rendering.materials.constants import (
    PBRParameterRange,
)
from engine.rendering.materials.material_functions import (
    MaterialFunctionLibrary,
    create_fresnel_function,
    create_normal_blend_function,
    create_parallax_function,
    create_triplanar_function,
    create_detail_normal_function,
    create_height_blend_function,
    create_srgb_to_linear_function,
    create_linear_to_srgb_function,
    create_luminance_function,
    create_saturation_function,
    create_contrast_function,
    create_noise_function,
    create_voronoi_function,
    create_gradient_noise_function,
    create_checkerboard_function,
    create_radial_gradient_function,
    create_box_mask_function,
    create_sphere_mask_function,
    create_blend_overlay_function,
    create_blend_soft_light_function,
    create_fresnel_schlick_function,
    create_normal_blend_rnm_function,
    create_parallax_occlusion_function,
)
from engine.rendering.materials.material_graph import (
    AddNode,
    AppendNode,
    CeilNode,
    ClampNode,
    ComponentMask,
    ConstantNode,
    DataType,
    DivideNode,
    DotNode,
    FracNode,
    GraphCompiler,
    GraphValidationError,
    LerpNode,
    MaterialGraph,
    MaterialNode,
    MaxNode,
    MinNode,
    MultiplyNode,
    NodeConnection,
    NodePort,
    NormalizeNode,
    OneMinus,
    OutputNode,
    ParameterNode,
    PowerNode,
    SinNode,
    CosNode,
    SqrtNode,
    AbsNode,
    FloorNode,
    SubtractNode,
    TextureSampleNode,
    UVNode,
)
from engine.rendering.materials.material_system import (
    BlendMode,
    DirtyFlags,
    LayerBlendSettings,
    MaterialDomain,
    MaterialFunction,
    MaterialInstance,
    MaterialLayer,
    MaterialParameter,
    MaterialSystem,
    MaterialTemplate,
    ParameterType,
    ShadingModel,
)
from engine.rendering.materials.pbr_model import (
    PBRDirtyFlags,
    PBRMaterial,
    PBRParameters,
    PBRTextureSet,
    PBRWorkflow,
    TextureChannel,
    clamp_pbr_parameter,
    validate_pbr_parameter,
)
from engine.rendering.materials.shader_compiler import (
    CompiledShader,
    CompilationError,
    HotReloadWatcher,
    PermutationKey,
    PSOCache,
    PSODescriptor,
    ShaderCompiler,
    ShaderDefine,
    ShaderLanguage,
    ShaderPermutation,
    ShaderSource,
    ShaderStage,
)

# =============================================================================
# material_system.py — Whitebox
# =============================================================================


class TestMaterialParameterWhitebox:
    """Whitebox: internal branches of MaterialParameter."""

    def test_clamp_non_numeric_type_passthrough(self) -> None:
        """clamp() returns value unchanged for non-numeric types."""
        param = MaterialParameter(
            name="tex", param_type=ParameterType.TEXTURE_2D,
            default_value=None,
        )
        # Vec3 is not int/float — clamp should return it as-is
        val = Vec3(9.9, 9.9, 9.9)
        assert param.clamp(val) is val

    def test_clamp_min_only(self) -> None:
        """clamp() clamps below min when max is None."""
        param = MaterialParameter(
            name="e", param_type=ParameterType.FLOAT,
            default_value=0.0, min_value=0.0, max_value=None,
        )
        assert param.clamp(-5.0) == 0.0

    def test_clamp_max_only(self) -> None:
        """clamp() clamps above max when min is None."""
        param = MaterialParameter(
            name="e", param_type=ParameterType.FLOAT,
            default_value=0.0, min_value=None, max_value=100.0,
        )
        assert param.clamp(999.0) == 100.0

    def test_clamp_no_bounds(self) -> None:
        """clamp() returns value unchanged when both bounds are None."""
        param = MaterialParameter(
            name="e", param_type=ParameterType.FLOAT,
            default_value=0.0, min_value=None, max_value=None,
        )
        assert param.clamp(-5.0) == -5.0


class TestMaterialTemplateWhitebox:
    """Whitebox: internal branches of MaterialTemplate."""

    def test_add_parameter_raises_on_duplicate(self) -> None:
        """add_parameter raises ValueError for duplicate name."""
        tmpl = MaterialTemplate("test")
        tmpl.add_parameter(MaterialParameter(
            name="p", param_type=ParameterType.FLOAT, default_value=0.0,
        ))
        with pytest.raises(ValueError, match="already exists"):
            tmpl.add_parameter(MaterialParameter(
                name="p", param_type=ParameterType.FLOAT, default_value=1.0,
            ))

    def test_remove_parameter_raises_on_missing(self) -> None:
        """remove_parameter raises KeyError for unknown name."""
        tmpl = MaterialTemplate("test")
        with pytest.raises(KeyError, match="not found"):
            tmpl.remove_parameter("nonexistent")

    def test_add_function_duplicate_noop(self) -> None:
        """add_function with already-present func does not bump version."""
        tmpl = MaterialTemplate("test")
        v0 = tmpl.version
        func = MaterialFunction(name="F", code="//nop")
        tmpl.add_function(func)
        v1 = tmpl.version
        tmpl.add_function(func)  # same object again
        assert tmpl.version == v1, "duplicate add_function bumped version"

    def test_compute_permutation_key_sorted_features(self) -> None:
        """compute_permutation_key produces same hash regardless of insertion order."""
        tmpl = MaterialTemplate("test")
        k1 = tmpl.compute_permutation_key({"a", "b", "c"})
        k2 = tmpl.compute_permutation_key({"c", "a", "b"})
        assert k1 == k2


class TestMaterialInstanceWhitebox:
    """Whitebox: internal branches of MaterialInstance."""

    def test_set_parameter_no_change_skips_dirty(self) -> None:
        """set_parameter with same value (already an override) does not mark dirty."""
        tmpl = MaterialTemplate("test")
        tmpl.add_parameter(MaterialParameter(
            name="p", param_type=ParameterType.FLOAT, default_value=0.0,
        ))
        inst = tmpl.create_instance()
        inst.set_parameter("p", 1.0)  # first override
        inst._dirty.clear_all()
        # Set same override again — should be a no-op
        inst.set_parameter("p", 1.0)
        assert not inst.dirty.any_dirty()

    def test_set_parameter_clamp_false_preserves_value(self) -> None:
        """set_parameter(clamp=False) does not clamp out-of-range value."""
        tmpl = MaterialTemplate("test")
        tmpl.add_parameter(MaterialParameter(
            name="p", param_type=ParameterType.FLOAT,
            default_value=0.5, min_value=0.0, max_value=1.0,
        ))
        inst = tmpl.create_instance()
        inst.set_parameter("p", 5.0, clamp=False, validate=False)
        assert inst.get_parameter("p") == 5.0

    def test_set_parameter_texture_sets_textures_dirty(self) -> None:
        """TEXTURE_2D parameter sets dirty.textures not dirty.parameters."""
        tmpl = MaterialTemplate("test")
        tmpl.add_parameter(MaterialParameter(
            name="tex", param_type=ParameterType.TEXTURE_2D,
            default_value=None,
        ))
        inst = tmpl.create_instance()
        inst._dirty.clear_all()
        inst.set_parameter("tex", "path.dds")
        assert inst.dirty.textures
        assert not inst.dirty.parameters

    def test_set_parameter_float_sets_parameters_dirty(self) -> None:
        """FLOAT parameter sets dirty.parameters not dirty.textures."""
        tmpl = MaterialTemplate("test")
        tmpl.add_parameter(MaterialParameter(
            name="p", param_type=ParameterType.FLOAT, default_value=0.0,
        ))
        inst = tmpl.create_instance()
        inst._dirty.clear_all()
        inst.set_parameter("p", 0.5)
        assert inst.dirty.parameters
        assert not inst.dirty.textures

    def test_clear_override_nonexistent_noop(self) -> None:
        """clear_override for non-existent name does not mark dirty."""
        tmpl = MaterialTemplate("test")
        inst = tmpl.create_instance()
        inst._dirty.clear_all()
        inst.clear_override("nonexistent")
        assert not inst.dirty.any_dirty()

    def test_pop_layer_empty_returns_none(self) -> None:
        """pop_layer on empty stack returns None."""
        tmpl = MaterialTemplate("test")
        inst = tmpl.create_instance()
        assert inst.pop_layer() is None

    def test_clone_independent_overrides(self) -> None:
        """clone produces an independent copy; mutations do not cross."""
        tmpl = MaterialTemplate("test")
        tmpl.add_parameter(MaterialParameter(
            name="p", param_type=ParameterType.FLOAT, default_value=0.0,
        ))
        inst = tmpl.create_instance()
        inst.set_parameter("p", 1.0)
        clone = inst.clone()
        clone.set_parameter("p", 2.0)
        assert inst.get_parameter("p") == 1.0
        assert clone.get_parameter("p") == 2.0


class TestMaterialFunctionWhitebox:
    """Whitebox: internal branches of MaterialFunction."""

    def test_get_dependencies_transitive_with_visited(self) -> None:
        """get_dependencies collects transitive deps with visited-set dedup."""
        a = MaterialFunction(name="A", code="// A")
        b = MaterialFunction(name="B", code="// B")
        c = MaterialFunction(name="C", code="// C")
        # A -> B -> C, and B -> C again (duplicate edge)
        b.add_dependency(c)
        a.add_dependency(b)
        a.add_dependency(c)  # direct ref to C as well
        deps = a.get_dependencies()
        names = [d.name for d in deps]
        # C should appear before B (post-order), and only once
        assert names == ["C", "B"]
        assert len(names) == len(set(names))

    def test_get_full_code_dependency_ordering(self) -> None:
        """get_full_code emits dependency code before own code."""
        a = MaterialFunction(name="A", code="// code A")
        b = MaterialFunction(name="B", code="// code B")
        b.add_dependency(a)
        full = b.get_full_code()
        assert full.index("// code A") < full.index("// code B")


class TestMaterialLayerWhitebox:
    """Whitebox: internal branches of MaterialLayer."""

    def test_enabled_setter_toggle(self) -> None:
        """enabled setter toggles the flag."""
        layer = MaterialLayer("test")
        assert layer.enabled
        layer.enabled = False
        assert not layer.enabled

    def test_get_parameter_default(self) -> None:
        """get_parameter returns default when key not set."""
        layer = MaterialLayer("test")
        assert layer.get_parameter("missing", 42) == 42


class TestMaterialSystemWhitebox:
    """Whitebox: internal branches of MaterialSystem."""

    def test_unregister_template_missing_raises(self) -> None:
        """unregister_template raises KeyError for unknown id."""
        sys = MaterialSystem()
        with pytest.raises(KeyError, match="not found"):
            sys.unregister_template("nonexistent")

    def test_get_template_by_name_missing_returns_none(self) -> None:
        """get_template_by_name returns None when no match."""
        sys = MaterialSystem()
        assert sys.get_template_by_name("nonexistent") is None

    def test_get_dirty_instances_removes_stale(self) -> None:
        """get_dirty_instances discards ids whose instance was removed."""
        sys = MaterialSystem()
        tmpl = sys.create_template("T")
        inst = tmpl.create_instance()
        sys.register_instance(inst)
        # Manually inject a stale id
        sys._dirty_instances.add("stale-id")
        dirty = sys.get_dirty_instances()
        assert "stale-id" not in sys._dirty_instances

    def test_get_dirty_instances_clean_instance_not_returned(self) -> None:
        """get_dirty_instances does not return instance with no dirty flags."""
        sys = MaterialSystem()
        tmpl = sys.create_template("T")
        inst = tmpl.create_instance()
        sys.register_instance(inst)
        inst._dirty.clear_all()
        dirty = sys.get_dirty_instances()
        assert len(dirty) == 0

    def test_clear_dirty_flags_handles_missing_instance(self) -> None:
        """clear_dirty_flags tolerates instance removed mid-clear."""
        sys = MaterialSystem()
        tmpl = sys.create_template("T")
        inst = tmpl.create_instance()
        sys.register_instance(inst)
        sys._dirty_instances.add("orphan")
        sys.clear_dirty_flags()  # should not raise

    def test_create_instance_string_template_with_fallback(self) -> None:
        """create_instance with string looks up by id then by name."""
        sys = MaterialSystem()
        tmpl = sys.create_template("MyMaterial")
        # Use name as lookup (id lookup will fail, name lookup should succeed)
        inst = sys.create_instance("MyMaterial")
        assert inst.template is tmpl

    def test_create_instance_string_template_not_found(self) -> None:
        """create_instance with unresolvable string raises KeyError."""
        sys = MaterialSystem()
        with pytest.raises(KeyError, match="not found"):
            sys.create_instance("nonexistent")


# =============================================================================
# pbr_model.py — Whitebox
# =============================================================================


class TestPBRParametersWhitebox:
    """Whitebox: internal branches of PBRParameters."""

    def test_validate_collects_multiple_errors(self) -> None:
        """validate() collects multiple parameter errors at once."""
        params = PBRParameters(
            base_color=Vec4(2.0, 1.0, 1.0, 1.0),
            metallic=1.5,
            roughness=-0.1,
            emissive=Vec3(0.0, -1.0, 0.0),
        )
        is_valid, errors = params.validate()
        assert not is_valid
        assert len(errors) >= 3

    def test_validate_emissive_below_zero(self) -> None:
        """validate() rejects negative emissive channels."""
        params = PBRParameters(emissive=Vec3(-0.5, 0.0, 0.0))
        is_valid, errors = params.validate()
        assert not is_valid
        assert any("emissive.x" in e for e in errors)

    def test_clamp_bounds_all_fields(self) -> None:
        """clamp() brings all out-of-range fields into range."""
        params = PBRParameters(
            base_color=Vec4(-0.1, 1.5, 2.0, -1.0),
            metallic=-0.5,
            roughness=2.0,
            normal_scale=-1.0,
            ao=-1.0,
            emissive=Vec3(-5.0, 0.0, 0.0),
        )
        clamped = params.clamp()
        assert clamped.base_color == Vec4(0.0, 1.0, 1.0, 0.0)
        assert clamped.metallic == 0.0
        assert clamped.roughness == 1.0
        assert clamped.normal_scale == 0.0
        assert clamped.ao == 0.0
        assert clamped.emissive == Vec3(0.0, 0.0, 0.0)

    def test_lerp_t_clamped(self) -> None:
        """lerp() clamps t to [0, 1]."""
        a = PBRParameters(metallic=0.0)
        b = PBRParameters(metallic=1.0)
        r1 = a.lerp(b, -0.5)
        assert r1.metallic == 0.0
        r2 = a.lerp(b, 1.5)
        assert r2.metallic == 1.0


class TestPBRTextureSetWhitebox:
    """Whitebox: internal branches of PBRTextureSet."""

    def test_has_any_texture_all_none(self) -> None:
        """has_any_texture returns False when all are None."""
        ts = PBRTextureSet()
        assert not ts.has_any_texture()

    def test_get_texture_paths_all_none(self) -> None:
        """get_texture_paths returns empty list when all are None."""
        ts = PBRTextureSet()
        assert ts.get_texture_paths() == []


class TestPBRDirtyFlagsWhitebox:
    """Whitebox: every setter marks _all_dirty."""

    def _check_all_dirty(self, flag_name: str, value: bool = True) -> None:
        flags = PBRDirtyFlags()
        flags.clear_all()
        setattr(flags, flag_name, value)
        assert flags._all_dirty == value

    def test_base_color_sets_all_dirty(self) -> None:
        self._check_all_dirty("base_color")

    def test_metallic_sets_all_dirty(self) -> None:
        self._check_all_dirty("metallic")

    def test_roughness_sets_all_dirty(self) -> None:
        self._check_all_dirty("roughness")

    def test_normal_scale_sets_all_dirty(self) -> None:
        self._check_all_dirty("normal_scale")

    def test_ao_sets_all_dirty(self) -> None:
        self._check_all_dirty("ao")

    def test_emissive_sets_all_dirty(self) -> None:
        self._check_all_dirty("emissive")

    def test_textures_sets_all_dirty(self) -> None:
        self._check_all_dirty("textures")


class TestPBRMaterialWhitebox:
    """Whitebox: internal branches of PBRMaterial setters and callbacks."""

    def test_same_value_no_change_no_dirty(self) -> None:
        """Setting same value does not mark dirty (no-change optimization)."""
        mat = PBRMaterial()
        mat.dirty.clear_all()
        mat.metallic = 0.0  # already default
        assert not mat.dirty.metallic
        mat.roughness = 0.5  # already default
        assert not mat.dirty.roughness

    def test_callback_notification(self) -> None:
        """on_change callback fires on value change."""
        mat = PBRMaterial()
        calls: List[Tuple[str, Any]] = []
        mat.on_change(lambda name, val: calls.append((name, val)))
        mat.metallic = 0.5
        assert len(calls) == 1
        assert calls[0] == ("metallic", 0.5)

    def test_callback_not_called_on_no_change(self) -> None:
        """on_change callback does NOT fire when value unchanged."""
        mat = PBRMaterial()
        calls: List[Tuple[str, Any]] = []
        mat.on_change(lambda name, val: calls.append((name, val)))
        mat.metallic = 0.0  # already 0.0
        assert len(calls) == 0

    def test_clone_preserves_workflow(self) -> None:
        """clone preserves workflow type."""
        mat = PBRMaterial(workflow=PBRWorkflow.SPECULAR_GLOSSINESS)
        clone = mat.clone()
        assert clone.workflow == PBRWorkflow.SPECULAR_GLOSSINESS

    def test_set_textures_marks_dirty(self) -> None:
        """Setting textures property marks dirty."""
        mat = PBRMaterial()
        mat.dirty.clear_all()
        mat.textures = PBRTextureSet(base_color_map="test.dds")
        assert mat.dirty.textures


class TestValidatePBRParameterWhitebox:
    """Whitebox: branches in validate_pbr_parameter."""

    def test_unknown_parameter(self) -> None:
        """validate_pbr_parameter returns False for unknown name."""
        is_valid, msg = validate_pbr_parameter("bogus", 0.0)
        assert not is_valid
        assert "Unknown" in msg

    def test_base_color_wrong_type(self) -> None:
        """validate_pbr_parameter rejects non-Vec4 for base_color."""
        is_valid, msg = validate_pbr_parameter("base_color", 1.0)
        assert not is_valid
        assert "Vec4" in msg

    def test_emissive_negative_channel(self) -> None:
        """validate_pbr_parameter rejects Vec3 with negative channel."""
        is_valid, msg = validate_pbr_parameter("emissive", Vec3(-1.0, 0.0, 0.0))
        assert not is_valid
        assert ">=" in msg

    def test_metallic_wrong_type(self) -> None:
        """validate_pbr_parameter rejects non-numeric for metallic."""
        is_valid, msg = validate_pbr_parameter("metallic", "high")
        assert not is_valid
        assert "number" in msg


class TestClampPBRParameterWhitebox:
    """Whitebox: isinstance branches in clamp_pbr_parameter."""

    def test_unknown_name_returns_value(self) -> None:
        """clamp_pbr_parameter returns value unchanged for unknown name."""
        val = 42
        assert clamp_pbr_parameter("bogus", val) is val

    def test_clamp_vec4(self) -> None:
        """clamp_pbr_parameter clamps Vec4 channels."""
        result = clamp_pbr_parameter("base_color", Vec4(-1.0, 2.0, 0.5, 1.5))
        assert result == Vec4(0.0, 1.0, 0.5, 1.0)

    def test_clamp_vec3(self) -> None:
        """clamp_pbr_parameter clamps Vec3 channels."""
        result = clamp_pbr_parameter("emissive", Vec3(-1.0, 5.0, 0.0))
        assert result == Vec3(0.0, 5.0, 0.0)

    def test_clamp_float(self) -> None:
        """clamp_pbr_parameter clamps float."""
        result = clamp_pbr_parameter("metallic", 5.0)
        assert result == 1.0

    def test_clamp_other_type(self) -> None:
        """clamp_pbr_parameter returns non-Vec4/Vec3/float unchanged."""
        result = clamp_pbr_parameter("base_color", [1, 2, 3, 4])
        assert result == [1, 2, 3, 4]


# =============================================================================
# material_functions.py — Whitebox
# =============================================================================


class TestMaterialFunctionLibraryWhitebox:
    """Whitebox: singleton and category filtering."""

    def test_singleton(self) -> None:
        """MaterialFunctionLibrary() returns the same instance."""
        a = MaterialFunctionLibrary()
        b = MaterialFunctionLibrary()
        assert a is b

    def test_get_by_category_case_insensitive(self) -> None:
        """get_by_category matches case-insensitively on description."""
        lib = MaterialFunctionLibrary()
        # At least "Lighting" should match
        lighting_funcs = lib.get_by_category("lighting")
        assert len(lighting_funcs) > 0
        for f in lighting_funcs:
            assert "lighting" in f.description.lower()

    def test_get_all_returns_all(self) -> None:
        """get_all returns all registered functions."""
        lib = MaterialFunctionLibrary()
        all_funcs = lib.get_all()
        assert len(all_funcs) > 20  # should have 24 builtins


class TestCreateFunctionsWhitebox:
    """Whitebox: verify input/output port counts for each create_*_function."""

    def _check_ports(
        self, func, num_inputs: int, num_outputs: int,
    ) -> None:
        assert len(func.inputs) == num_inputs, (
            f"{func.name}: expected {num_inputs} inputs, got {len(func.inputs)}"
        )
        assert len(func.outputs) == num_outputs, (
            "{func.name}: expected {num_outputs} outputs, got {len(func.outputs)}"
        )

    def test_fresnel(self) -> None:
        self._check_ports(create_fresnel_function(), 3, 1)

    def test_fresnel_schlick(self) -> None:
        self._check_ports(create_fresnel_schlick_function(), 2, 1)

    def test_normal_blend(self) -> None:
        self._check_ports(create_normal_blend_function(), 2, 1)

    def test_normal_blend_rnm(self) -> None:
        self._check_ports(create_normal_blend_rnm_function(), 2, 1)

    def test_parallax(self) -> None:
        self._check_ports(create_parallax_function(), 4, 1)

    def test_parallax_occlusion(self) -> None:
        self._check_ports(create_parallax_occlusion_function(), 3, 1)

    def test_triplanar(self) -> None:
        self._check_ports(create_triplanar_function(), 4, 1)

    def test_detail_normal(self) -> None:
        self._check_ports(create_detail_normal_function(), 3, 1)

    def test_height_blend(self) -> None:
        self._check_ports(create_height_blend_function(), 4, 1)

    def test_srgb_to_linear(self) -> None:
        self._check_ports(create_srgb_to_linear_function(), 1, 1)

    def test_linear_to_srgb(self) -> None:
        self._check_ports(create_linear_to_srgb_function(), 1, 1)

    def test_luminance(self) -> None:
        self._check_ports(create_luminance_function(), 1, 1)

    def test_saturation(self) -> None:
        self._check_ports(create_saturation_function(), 2, 1)

    def test_contrast(self) -> None:
        self._check_ports(create_contrast_function(), 2, 1)

    def test_noise(self) -> None:
        self._check_ports(create_noise_function(), 1, 1)

    def test_voronoi(self) -> None:
        self._check_ports(create_voronoi_function(), 1, 1)

    def test_gradient_noise(self) -> None:
        self._check_ports(create_gradient_noise_function(), 1, 1)

    def test_checkerboard(self) -> None:
        self._check_ports(create_checkerboard_function(), 2, 1)

    def test_radial_gradient(self) -> None:
        self._check_ports(create_radial_gradient_function(), 3, 1)

    def test_box_mask(self) -> None:
        self._check_ports(create_box_mask_function(), 4, 1)

    def test_sphere_mask(self) -> None:
        self._check_ports(create_sphere_mask_function(), 4, 1)

    def test_blend_overlay(self) -> None:
        self._check_ports(create_blend_overlay_function(), 2, 1)

    def test_blend_soft_light(self) -> None:
        self._check_ports(create_blend_soft_light_function(), 2, 1)


# =============================================================================
# material_graph.py — Whitebox
# =============================================================================


class TestNodePortWhitebox:
    """Whitebox: is_compatible_with branches."""

    def test_same_type_connected(self) -> None:
        """Same-type ports are compatible."""
        a = NodePort("a", DataType.FLOAT, is_output=True)
        b = NodePort("b", DataType.FLOAT, is_output=False)
        assert a.is_compatible_with(b)

    def test_same_direction_incompatible(self) -> None:
        """Two outputs are incompatible."""
        a = NodePort("a", DataType.FLOAT, is_output=True)
        b = NodePort("b", DataType.FLOAT, is_output=True)
        assert not a.is_compatible_with(b)

    def test_float_promotion_to_vec3(self) -> None:
        """FLOAT output can connect to VEC3 input."""
        a = NodePort("a", DataType.FLOAT, is_output=True)
        b = NodePort("b", DataType.VEC3, is_output=False)
        assert a.is_compatible_with(b)

    def test_vec3_promotion_to_float(self) -> None:
        """VEC3 output can connect to FLOAT input."""
        a = NodePort("a", DataType.VEC3, is_output=True)
        b = NodePort("b", DataType.FLOAT, is_output=False)
        assert a.is_compatible_with(b)

    def test_incompatible_types(self) -> None:
        """Unrelated types are incompatible."""
        a = NodePort("a", DataType.TEXTURE2D, is_output=True)
        b = NodePort("b", DataType.BOOL, is_output=False)
        assert not a.is_compatible_with(b)


class TestConstantNodeWhitebox:
    """Whitebox: generate_code for each DataType."""

    def test_generate_code_float(self) -> None:
        node = ConstantNode(3.14, DataType.FLOAT)
        code = node.generate_code({}, "out")
        assert "float out = 3.14" in code

    def test_generate_code_vec2(self) -> None:
        node = ConstantNode(Vec2(1, 2), DataType.VEC2)
        code = node.generate_code({}, "out")
        assert "vec2 out = vec2(1.0, 2.0)" in code

    def test_generate_code_vec3(self) -> None:
        node = ConstantNode(Vec3(1, 2, 3), DataType.VEC3)
        code = node.generate_code({}, "out")
        assert "vec3 out = vec3(1.0, 2.0, 3.0)" in code

    def test_generate_code_vec4(self) -> None:
        node = ConstantNode(Vec4(1, 2, 3, 4), DataType.VEC4)
        code = node.generate_code({}, "out")
        assert "vec4 out = vec4(1.0, 2.0, 3.0, 4.0)" in code

    def test_generate_code_unknown(self) -> None:
        node = ConstantNode(True, DataType.BOOL)
        code = node.generate_code({}, "out")
        assert "Unknown type" in code


class TestComponentMaskWhitebox:
    """Whitebox: output type depends on component count."""

    def test_one_component(self) -> None:
        node = ComponentMask("x")
        assert "result" in node.outputs
        assert node.outputs["result"].data_type == DataType.FLOAT

    def test_two_components(self) -> None:
        node = ComponentMask("xy")
        assert node.outputs["result"].data_type == DataType.VEC2

    def test_three_components(self) -> None:
        node = ComponentMask("xyz")
        assert node.outputs["result"].data_type == DataType.VEC3

    def test_four_components(self) -> None:
        node = ComponentMask("xyzw")
        assert node.outputs["result"].data_type == DataType.VEC4


class TestMaterialGraphWhitebox:
    """Whitebox: graph mutation error paths."""

    def test_add_node_duplicate_output(self) -> None:
        """add_node raises GraphValidationError on second OutputNode."""
        graph = MaterialGraph("test")
        graph.add_node(OutputNode())
        with pytest.raises(GraphValidationError, match="one output node"):
            graph.add_node(OutputNode())

    def test_remove_node_not_found_noop(self) -> None:
        """remove_node on non-existent id does nothing."""
        graph = MaterialGraph("test")
        graph.remove_node("nonexistent")  # should not raise

    def test_remove_node_cleans_connections(self) -> None:
        """remove_node removes all connections to/from the node."""
        graph = MaterialGraph("test")
        a = ConstantNode(1.0)
        b = ConstantNode(2.0)
        out = OutputNode()
        graph.add_node(a)
        graph.add_node(b)
        graph.add_node(out)
        graph.connect(a, "value", out, "base_color")
        graph.connect(b, "value", out, "roughness")
        assert len(graph.connections) == 2
        graph.remove_node(a.node_id)
        assert len(graph.connections) == 1

    def test_connect_source_not_found(self) -> None:
        """connect raises GraphValidationError when source node missing."""
        graph = MaterialGraph("test")
        out = OutputNode()
        graph.add_node(out)
        with pytest.raises(GraphValidationError, match="Source node not found"):
            graph.connect("bogus", "value", out, "base_color")

    def test_connect_target_not_found(self) -> None:
        """connect raises GraphValidationError when target node missing."""
        graph = MaterialGraph("test")
        a = ConstantNode(1.0)
        graph.add_node(a)
        with pytest.raises(GraphValidationError, match="Target node not found"):
            graph.connect(a, "value", "bogus", "base_color")

    def test_connect_output_port_missing(self) -> None:
        """connect raises GraphValidationError when output port missing."""
        graph = MaterialGraph("test")
        a = ConstantNode(1.0)
        out = OutputNode()
        graph.add_node(a)
        graph.add_node(out)
        with pytest.raises(GraphValidationError, match="Output port not found"):
            graph.connect(a, "bogus_port", out, "base_color")

    def test_connect_input_port_missing(self) -> None:
        """connect raises GraphValidationError when input port missing."""
        graph = MaterialGraph("test")
        a = ConstantNode(1.0)
        out = OutputNode()
        graph.add_node(a)
        graph.add_node(out)
        with pytest.raises(GraphValidationError, match="Input port not found"):
            graph.connect(a, "value", out, "bogus_port")

    def test_connect_incompatible_types(self) -> None:
        """connect raises on incompatible type connection."""
        graph = MaterialGraph("test")
        # VEC2 output from UVNode -> VEC3 input on OutputNode IS incompatible
        # because only FLOAT <-> vector promotion is supported, not vec2 <-> vec3
        uv_node = UVNode()
        out = OutputNode()
        graph.add_node(uv_node)
        graph.add_node(out)
        with pytest.raises(GraphValidationError, match="Incompatible types"):
            graph.connect(uv_node, "uv", out, "base_color")

    def test_connect_already_connected(self) -> None:
        """connect raises when input port already occupied."""
        graph = MaterialGraph("test")
        a = ConstantNode(1.0)
        b = ConstantNode(2.0)
        out = OutputNode()
        graph.add_node(a)
        graph.add_node(b)
        graph.add_node(out)
        graph.connect(a, "value", out, "roughness")
        with pytest.raises(GraphValidationError, match="already connected"):
            graph.connect(b, "value", out, "roughness")

    def test_disconnect_removes_exact_connection(self) -> None:
        """disconnect returns True only for exact match."""
        graph = MaterialGraph("test")
        a = ConstantNode(1.0)
        out = OutputNode()
        graph.add_node(a)
        graph.add_node(out)
        graph.connect(a, "value", out, "roughness")
        assert graph.disconnect(out, "roughness")
        assert not graph.disconnect(out, "roughness")

    def test_get_input_connection(self) -> None:
        """get_input_connection finds connection to a specific port."""
        graph = MaterialGraph("test")
        a = ConstantNode(1.0)
        out = OutputNode()
        graph.add_node(a)
        graph.add_node(out)
        graph.connect(a, "value", out, "roughness")
        conn = graph.get_input_connection(out.node_id, "roughness")
        assert conn is not None
        assert conn.source_node == a.node_id

    def test_get_input_connection_none(self) -> None:
        """get_input_connection returns None for unconnected port."""
        graph = MaterialGraph("test")
        out = OutputNode()
        graph.add_node(out)
        conn = graph.get_input_connection(out.node_id, "roughness")
        assert conn is None

    def test_get_output_connections(self) -> None:
        """get_output_connections returns all downstream connections."""
        graph = MaterialGraph("test")
        a = ConstantNode(1.0)
        out = OutputNode()
        graph.add_node(a)
        graph.add_node(out)
        graph.connect(a, "value", out, "roughness")
        conns = graph.get_output_connections(a.node_id, "value")
        assert len(conns) == 1

    def test_has_cycle_detection(self) -> None:
        """_has_cycle detects a cycle."""
        graph = MaterialGraph("test")
        a = ConstantNode(1.0)
        b = ConstantNode(2.0)
        graph.add_node(a)
        graph.add_node(b)
        # Manually wire a -> b -> a (cycle)
        graph._connections = [
            NodeConnection(source_node=a.node_id, source_port="value",
                           target_node=b.node_id, target_port="value"),
            NodeConnection(source_node=b.node_id, source_port="value",
                           target_node=a.node_id, target_port="value"),
        ]
        assert graph._has_cycle()

    def test_has_no_cycle(self) -> None:
        """_has_cycle returns False for acyclic graph."""
        graph = MaterialGraph("test")
        assert not graph._has_cycle()

    def test_topological_order(self) -> None:
        """get_topological_order returns nodes in dependency order."""
        graph = MaterialGraph("test")
        a = ConstantNode(1.0)
        b = ConstantNode(2.0)
        out = OutputNode()
        graph.add_node(a)
        graph.add_node(b)
        graph.add_node(out)
        graph.connect(a, "value", out, "roughness")
        graph.connect(b, "value", out, "metallic")
        order = graph.get_topological_order()
        # a and b should come before out
        assert order.index(a.node_id) < order.index(out.node_id)
        assert order.index(b.node_id) < order.index(out.node_id)

    def test_validate_no_output_node(self) -> None:
        """validate() reports missing output node."""
        graph = MaterialGraph("test")
        is_valid, errors = graph.validate()
        assert not is_valid
        assert any("output" in e.lower() for e in errors)

    def test_validate_invalid_connections(self) -> None:
        """validate() reports dangling connection targets."""
        graph = MaterialGraph("test")
        out = OutputNode()
        graph.add_node(out)
        graph._connections.append(
            NodeConnection("bogus", "value", out.node_id, "roughness")
        )
        is_valid, errors = graph.validate()
        assert not is_valid
        assert any("Invalid connection" in e for e in errors)


class TestGraphCompilerWhitebox:
    """Whitebox: internal code generation paths."""

    def test_compile_invalid_graph_raises(self) -> None:
        """compile() raises GraphValidationError on invalid graph."""
        compiler = GraphCompiler()
        graph = MaterialGraph("empty")
        with pytest.raises(GraphValidationError, match="Invalid graph"):
            compiler.compile(graph)

    def test_default_value_all_types(self) -> None:
        """_default_value returns correct strings."""
        compiler = GraphCompiler()
        assert compiler._default_value(DataType.FLOAT) == "0.0"
        assert compiler._default_value(DataType.VEC2) == "vec2(0.0)"
        assert compiler._default_value(DataType.VEC3) == "vec3(0.0)"
        assert compiler._default_value(DataType.VEC4) == "vec4(0.0)"
        assert compiler._default_value(DataType.INT) == "0"
        assert compiler._default_value(DataType.BOOL) == "false"
        assert compiler._default_value(DataType.TEXTURE2D) == "0.0"

    def test_format_value_vec2(self) -> None:
        """_format_value handles Vec2."""
        compiler = GraphCompiler()
        assert "vec2(1.0, 2.0)" in compiler._format_value(Vec2(1, 2))

    def test_format_value_vec3(self) -> None:
        """_format_value handles Vec3."""
        compiler = GraphCompiler()
        assert "vec3" in compiler._format_value(Vec3(1, 2, 3))

    def test_format_value_vec4(self) -> None:
        """_format_value handles Vec4."""
        compiler = GraphCompiler()
        assert "vec4" in compiler._format_value(Vec4(1, 2, 3, 4))

    def test_format_value_bool(self) -> None:
        """_format_value handles bool."""
        compiler = GraphCompiler()
        assert compiler._format_value(True) == "true"
        assert compiler._format_value(False) == "false"

    def test_format_value_float_int(self) -> None:
        """_format_value handles float and int."""
        compiler = GraphCompiler()
        assert compiler._format_value(3.14) == "3.14"
        assert compiler._format_value(42) == "42"

    def test_format_value_other(self) -> None:
        """_format_value falls back to str()."""
        compiler = GraphCompiler()
        assert compiler._format_value("hello") == "hello"


# =============================================================================
# shader_compiler.py — Whitebox
# =============================================================================


class TestShaderDefineWhitebox:
    """Whitebox: to_string branches."""

    def test_with_value(self) -> None:
        d = ShaderDefine("QUALITY", "high")
        assert d.to_string() == "#define QUALITY high"

    def test_without_value(self) -> None:
        d = ShaderDefine("NORMAL_MAPPING")
        assert d.to_string() == "#define NORMAL_MAPPING"


class TestShaderSourceWhitebox:
    """Whitebox: file load, hash, change detection."""

    def test_from_file_not_found(self) -> None:
        """from_file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            ShaderSource.from_file("/nonexistent/file.glsl", ShaderStage.VERTEX)

    def test_from_file_auto_detect_language(self) -> None:
        """from_file auto-detects language from extension."""
        with tempfile.NamedTemporaryFile(
            suffix=".wgsl", mode="w", delete=False,
        ) as f:
            f.write("// test")
            path = f.name
        try:
            source = ShaderSource.from_file(path, ShaderStage.COMPUTE)
            assert source.language == ShaderLanguage.WGSL
        finally:
            os.unlink(path)

    def test_from_file_unknown_extension_defaults_glsl(self) -> None:
        """from_file defaults to GLSL for unknown extension."""
        with tempfile.NamedTemporaryFile(
            suffix=".custom", mode="w", delete=False,
        ) as f:
            f.write("// test")
            path = f.name
        try:
            source = ShaderSource.from_file(path, ShaderStage.FRAGMENT)
            assert source.language == ShaderLanguage.GLSL
        finally:
            os.unlink(path)

    def test_has_changed_no_path(self) -> None:
        """has_changed returns False for inline source."""
        source = ShaderSource.from_string(
            "// test", ShaderStage.FRAGMENT, ShaderLanguage.GLSL,
        )
        assert not source.has_changed()

    def test_has_changed_os_error(self) -> None:
        """has_changed returns False on OSError."""
        source = ShaderSource.from_string(
            "// test", ShaderStage.FRAGMENT, ShaderLanguage.GLSL,
        )
        source.path = "/tmp/nonexistent_shader_file.glsl"
        assert not source.has_changed()

    def test_reload_unchanged(self) -> None:
        """reload returns False when file hasn't changed."""
        source = ShaderSource.from_string(
            "// test", ShaderStage.FRAGMENT, ShaderLanguage.GLSL,
        )
        assert not source.reload()

    def test_get_content_hash_caching(self) -> None:
        """get_content_hash caches and returns the same value on re-call."""
        source = ShaderSource.from_string(
            "// test", ShaderStage.FRAGMENT, ShaderLanguage.GLSL,
        )
        h1 = source.get_content_hash()
        h2 = source.get_content_hash()
        assert h1 == h2
        assert len(h1) == 16  # SHADER_HASH_LENGTH

    def test_get_content_hash_includes_defines(self) -> None:
        """get_content_hash changes when defines change."""
        source = ShaderSource.from_string(
            "// test", ShaderStage.FRAGMENT, ShaderLanguage.GLSL,
        )
        h1 = source.get_content_hash()
        source.add_define("TEST_FEATURE")
        h2 = source.get_content_hash()
        assert h1 != h2


class TestPermutationKeyWhitebox:
    """Whitebox: feature management on PermutationKey."""

    def test_to_defines_sorted(self) -> None:
        """to_defines returns defines in sorted order."""
        key = PermutationKey.from_set({"z", "a", "m"})
        defines = key.to_defines()
        names = [d.name for d in defines]
        assert names == sorted(names)

    def test_with_feature_creates_new_key(self) -> None:
        """with_feature returns a new key without mutating original."""
        key = PermutationKey.empty()
        key2 = key.with_feature("A")
        assert not key.has_feature("A")
        assert key2.has_feature("A")

    def test_without_feature(self) -> None:
        key = PermutationKey.from_set({"A", "B"})
        key2 = key.without_feature("A")
        assert not key2.has_feature("A")
        assert key2.has_feature("B")

    def test_from_list(self) -> None:
        key = PermutationKey.from_list(["x", "y"])
        assert key.features == frozenset({"x", "y"})


class TestShaderPermutationWhitebox:
    """Whitebox: validation branches."""

    def test_validate_missing_required(self) -> None:
        perm = ShaderPermutation("test", features={"A", "B"}, required={"A"})
        key = PermutationKey.from_set({"B"})
        is_valid, msg = perm.validate_key(key)
        assert not is_valid
        assert "Required" in msg

    def test_validate_unknown_feature(self) -> None:
        perm = ShaderPermutation("test", features={"A"})
        key = PermutationKey.from_set({"ZOMBIE"})
        is_valid, msg = perm.validate_key(key)
        assert not is_valid
        assert "Unknown" in msg

    def test_validate_conflict(self) -> None:
        perm = ShaderPermutation("test", features={"A", "B"})
        perm.add_conflict("A", "B")
        key = PermutationKey.from_set({"A", "B"})
        is_valid, msg = perm.validate_key(key)
        assert not is_valid
        assert "Conflicting" in msg

    def test_get_valid_keys_excludes_conflicts(self) -> None:
        """get_valid_keys omits keys that trigger conflicts."""
        perm = ShaderPermutation("test", features={"A", "B", "C"})
        perm.add_conflict("A", "B")
        keys = perm.get_valid_keys()
        for key in keys:
            assert not ("A" in key.features and "B" in key.features)


class TestPSODescriptorWhitebox:
    """Whitebox: compute pipeline detection."""

    def test_is_compute_pipeline_true(self) -> None:
        desc = PSODescriptor(compute_shader_hash="abc123")
        assert desc.is_compute_pipeline()

    def test_is_compute_pipeline_false(self) -> None:
        desc = PSODescriptor(vertex_shader_hash="xyz")
        assert not desc.is_compute_pipeline()


class TestPSOCacheWhitebox:
    """Whitebox: LRU behavior."""

    def test_get_updates_lru_order(self) -> None:
        """get moves accessed entry to end of LRU order."""
        cache = PSOCache(max_size=10)
        desc_a = PSODescriptor(vertex_shader_hash="aaa")
        desc_b = PSODescriptor(vertex_shader_hash="bbb")
        cache.put(desc_a, "pso_a")
        cache.put(desc_b, "pso_b")
        # Access a — it should move to end
        cache.get(desc_a)
        assert cache._lru_order[-1] == desc_a.get_hash()

    def test_put_evicts_oldest(self) -> None:
        """put evicts oldest entry when at capacity."""
        cache = PSOCache(max_size=2)
        desc_a = PSODescriptor(vertex_shader_hash="aaa")
        desc_b = PSODescriptor(vertex_shader_hash="bbb")
        desc_c = PSODescriptor(vertex_shader_hash="ccc")
        cache.put(desc_a, "pso_a")
        cache.put(desc_b, "pso_b")
        cache.put(desc_c, "pso_c")  # should evict a
        assert cache.get(desc_a) is None
        assert cache.get(desc_b) is not None

    def test_hit_rate_zero_when_empty(self) -> None:
        cache = PSOCache()
        assert cache.hit_rate == 0.0

    def test_invalidate_missing_key(self) -> None:
        """invalidate does nothing for non-existent key."""
        cache = PSOCache()
        desc = PSODescriptor(vertex_shader_hash="nope")
        cache.invalidate(desc)  # should not raise


class TestHotReloadWatcherWhitebox:
    """Whitebox: poll interval and error handling."""

    def test_check_changes_poll_interval_gate(self) -> None:
        """check_changes returns [] before poll interval elapses."""
        watcher = HotReloadWatcher(poll_interval=10.0)
        result = watcher.check_changes()
        assert result == []

    def test_watch_os_error_handled(self) -> None:
        """watch handles OSError without crashing."""
        watcher = HotReloadWatcher()
        # Non-existent path should raise OSError internally and be caught
        watcher.watch("/nonexistent/path/file.glsl")
        assert "/nonexistent/path/file.glsl" not in watcher._watched

    def test_check_changes_os_error_handled(self) -> None:
        """check_changes handles OSError when a file disappears."""
        watcher = HotReloadWatcher()
        watcher._poll_interval = 0.0
        watcher._last_check = 0.0
        watcher._watched["/nonexistent/path/file.glsl"] = 0.0
        result = watcher.check_changes()  # should not raise
        assert isinstance(result, list)

    def test_on_change_callback_fired(self) -> None:
        """on_change callback fires for changed files."""
        with tempfile.NamedTemporaryFile(suffix=".glsl", mode="w", delete=False) as f:
            f.write("// v1")
            path = f.name

        try:
            watcher = HotReloadWatcher(poll_interval=0.0)
            watcher._last_check = 0.0
            watcher.watch(path)

            # Modify the file
            with open(path, "w") as f:
                f.write("// v2")

            calls: List[str] = []
            watcher.on_change(lambda p: calls.append(p))
            changed = watcher.check_changes()
            assert path in changed
            assert path in calls
        finally:
            os.unlink(path)


# =============================================================================
# advanced_models.py — Whitebox
# =============================================================================


class TestSubsurfaceProfileWhitebox:
    """Whitebox: diffusion profile normalization edge case."""

    def test_get_diffusion_profile_normalizes(self) -> None:
        """get_diffusion_profile returns weights that sum to ~1."""
        profile = SubsurfaceProfile(scatter_radius=1.0)
        samples = profile.get_diffusion_profile(num_samples=8)
        assert len(samples) == 8
        assert abs(sum(samples) - 1.0) < 1e-6

    def test_get_diffusion_profile_zero_total_handled(self) -> None:
        """get_diffusion_profile handles very small radius gracefully."""
        profile = SubsurfaceProfile(scatter_radius=1e-6)
        samples = profile.get_diffusion_profile(num_samples=4)
        # With tiny radius, weights should still normalize to 1
        assert abs(sum(samples) - 1.0) < 1e-6


class TestSubsurfaceScatteringWhitebox:
    """Whitebox: setter clamping and dirty flag."""

    def test_opacity_clamped(self) -> None:
        """opacity setter clamps to [0, 1]."""
        sss = SubsurfaceScattering(opacity=5.0)
        assert sss.opacity == 1.0
        sss.opacity = -1.0
        assert sss.opacity == 0.0

    def test_setter_marks_dirty(self) -> None:
        """Each setter marks the dirty flag."""
        sss = SubsurfaceScattering()
        sss._dirty = False
        sss.profile = SubsurfaceProfile(name="Custom")
        assert sss._dirty
        sss._dirty = False
        sss.subsurface_color = Vec3(1, 0, 0)
        assert sss._dirty
        sss._dirty = False
        sss.enable_transmission = True
        assert sss._dirty

    def test_get_shader_defines_with_transmission(self) -> None:
        """get_shader_defines includes transmission when enabled."""
        sss = SubsurfaceScattering(enable_transmission=True)
        defines = sss.get_shader_defines()
        assert "HAS_SUBSURFACE_SCATTERING" in defines
        assert "HAS_SSS_TRANSMISSION" in defines


class TestClearCoatWhitebox:
    """Whitebox: validate and F0 calculation."""

    def test_validate_collects_errors(self) -> None:
        """validate() collects multiple errors with out-of-range values."""
        cc = ClearCoat()
        # Bypass __post_init__ clamping to set invalid values directly
        object.__setattr__(cc, "intensity", 5.0)
        object.__setattr__(cc, "roughness", -1.0)
        object.__setattr__(cc, "ior", 0.5)
        is_valid, errors = cc.validate()
        assert not is_valid
        assert len(errors) == 3

    def test_validate_pass(self) -> None:
        cc = ClearCoat(intensity=0.5, roughness=0.1, ior=1.5)
        is_valid, errors = cc.validate()
        assert is_valid
        assert errors == []

    def test_to_shader_data_f0(self) -> None:
        """F0 is calculated from IOR: ((n-1)/(n+1))^2."""
        cc = ClearCoat(ior=1.5)
        data = cc.to_shader_data()
        expected_f0 = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2
        assert abs(data["clearCoatF0"] - expected_f0) < 1e-6

    def test_get_shader_defines_with_normal(self) -> None:
        cc = ClearCoat(normal_map="coat_n.dds")
        defines = cc.get_shader_defines()
        assert "HAS_CLEAR_COAT_NORMAL" in defines


class TestAnisotropyWhitebox:
    """Whitebox: roughness calculation branches."""

    def test_positive_strength(self) -> None:
        """strength >= 0: roughness_t = base/aspect, roughness_b = base*aspect."""
        a = Anisotropy(strength=0.5)
        rt, rb = a.get_anisotropic_roughness(0.3)
        assert rt > rb  # stretched along tangent

    def test_negative_strength(self) -> None:
        """strength < 0: roughness_t = base*aspect, roughness_b = base/aspect."""
        a = Anisotropy(strength=-0.5)
        rt, rb = a.get_anisotropic_roughness(0.3)
        assert rt < rb  # stretched along bitangent

    def test_rotation_wrapped(self) -> None:
        """__post_init__ wraps rotation to [0, 2*pi)."""
        a = Anisotropy(rotation=7.0)  # > 2*pi
        assert 0.0 <= a.rotation < 2.0 * math.pi


class TestIridescenceWhitebox:
    """Whitebox: thickness clamping and interference color."""

    def test_thickness_max_clamped(self) -> None:
        """__post_init__ ensures thickness_max >= thickness_min."""
        iri = Iridescence(thickness_min=500.0, thickness_max=100.0)
        assert iri.thickness_max >= iri.thickness_min

    def test_get_interference_color_output(self) -> None:
        """get_interference_color returns Vec3 in [0,1]."""
        iri = Iridescence()
        color = iri.get_interference_color(250.0, 0.8)
        assert isinstance(color, Vec3)
        assert all(0.0 <= c <= 1.0 for c in (color.x, color.y, color.z))


class TestSheenWhitebox:
    """Whitebox: post-init clamping."""

    def test_roughness_clamped(self) -> None:
        sheen = Sheen(roughness=5.0)
        assert sheen.roughness == 1.0

    def test_intensity_clamped(self) -> None:
        sheen = Sheen(intensity=5.0)
        assert sheen.intensity == 2.0


class TestTransmissionWhitebox:
    """Whitebox: Fresnel mix and attenuation."""

    def test_get_fresnel_mix_schlick(self) -> None:
        """get_fresnel_mix follows Schlick approximation for IOR=1.0."""
        t = Transmission(ior=1.0)
        # When IOR == 1, F0 == 0, so reflection should be 0
        ref = t.get_fresnel_mix(1.0)
        assert ref == 0.0

    def test_get_fresnel_mix_graze(self) -> None:
        """At grazing angle (cos=0), reflection approaches 1."""
        t = Transmission(ior=1.5)
        ref = t.get_fresnel_mix(0.0)
        assert abs(ref - 1.0) < 1e-6

    def test_get_attenuation_infinite_distance(self) -> None:
        """With infinite attenuation_distance, returns (1,1,1)."""
        t = Transmission(attenuation_distance=float("inf"))
        att = t.get_attenuation(100.0)
        assert att == Vec3(1.0, 1.0, 1.0)

    def test_get_attenuation_finite(self) -> None:
        """With finite distance, follows beer-lambert."""
        t = Transmission(
            attenuation_distance=10.0,
            attenuation_color=Vec3(0.5, 0.5, 0.5),
        )
        att = t.get_attenuation(5.0)
        assert all(c < 1.0 for c in (att.x, att.y, att.z))

    def test_post_init_clamping(self) -> None:
        t = Transmission(factor=5.0, ior=0.5, roughness=-1.0, thickness=-5.0)
        assert t.factor == 1.0
        assert t.ior == 1.0
        assert t.roughness == 0.0
        assert t.thickness == 0.0


class TestAdvancedShadingModelWhitebox:
    """Whitebox: active model enumeration."""

    def test_all_none(self) -> None:
        model = AdvancedShadingModel()
        assert model.get_active_models() == []

    def test_mixed_activation(self) -> None:
        model = AdvancedShadingModel(
            subsurface=SubsurfaceScattering(),
            clear_coat=ClearCoat(),
            sheen=Sheen(),
        )
        active = model.get_active_models()
        assert ShadingModelType.SUBSURFACE in active
        assert ShadingModelType.CLEAR_COAT in active
        assert ShadingModelType.SHEEN in active
        assert ShadingModelType.ANISOTROPY not in active
        assert ShadingModelType.IRIDESCENCE not in active
        assert ShadingModelType.TRANSMISSION not in active


# =============================================================================
# constants.py — Whitebox
# =============================================================================


class TestPBRParameterRangeWhitebox:
    """Whitebox: frozen dataclass."""

    def test_frozen(self) -> None:
        r = PBRParameterRange(min_value=0.0, max_value=1.0, default_value=0.5)
        with pytest.raises(AttributeError):
            r.min_value = 2.0  # type: ignore[misc]
