"""Tests for the core material system.

Tests MaterialTemplate, MaterialInstance, MaterialFunction, MaterialLayer,
and MaterialSystem classes.
"""
import pytest

from engine.core.math.vec import Vec3, Vec4
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


class TestDirtyFlags:
    """Test DirtyFlags tracking."""

    def test_initial_state(self):
        """Test initial dirty flag state."""
        flags = DirtyFlags()
        assert not flags.parameters
        assert not flags.textures
        assert not flags.shader

    def test_mark_all(self):
        """Test marking all flags dirty."""
        flags = DirtyFlags()
        flags.mark_all()
        assert flags.parameters
        assert flags.textures
        assert flags.shader
        assert flags.any_dirty()

    def test_clear_all(self):
        """Test clearing all flags."""
        flags = DirtyFlags()
        flags.mark_all()
        flags.clear_all()
        assert not flags.parameters
        assert not flags.textures
        assert not flags.shader
        assert not flags.any_dirty()


class TestMaterialParameter:
    """Test MaterialParameter definition and validation."""

    def test_basic_parameter(self):
        """Test basic parameter creation."""
        param = MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
            min_value=0.0,
            max_value=1.0,
        )
        assert param.name == "roughness"
        assert param.param_type == ParameterType.FLOAT
        assert param.default_value == 0.5

    def test_validate_float_in_range(self):
        """Test float parameter validation."""
        param = MaterialParameter(
            name="test",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
            min_value=0.0,
            max_value=1.0,
        )
        is_valid, _ = param.validate(0.5)
        assert is_valid

    def test_validate_float_out_of_range(self):
        """Test float parameter validation for out-of-range value."""
        param = MaterialParameter(
            name="test",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
            min_value=0.0,
            max_value=1.0,
        )
        is_valid, error = param.validate(1.5)
        assert not is_valid
        assert "above maximum" in error

    def test_validate_none_value(self):
        """Test validation rejects None."""
        param = MaterialParameter(
            name="test",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
        )
        is_valid, error = param.validate(None)
        assert not is_valid
        assert "cannot be None" in error

    def test_clamp_value(self):
        """Test value clamping."""
        param = MaterialParameter(
            name="test",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
            min_value=0.0,
            max_value=1.0,
        )
        assert param.clamp(-0.5) == 0.0
        assert param.clamp(1.5) == 1.0
        assert param.clamp(0.5) == 0.5


class TestMaterialTemplate:
    """Test MaterialTemplate creation and management."""

    def test_create_template(self):
        """Test basic template creation."""
        template = MaterialTemplate(
            name="TestMaterial",
            domain=MaterialDomain.SURFACE,
            blend_mode=BlendMode.OPAQUE,
        )
        assert template.name == "TestMaterial"
        assert template.domain == MaterialDomain.SURFACE
        assert template.blend_mode == BlendMode.OPAQUE
        assert template.template_id is not None

    def test_add_parameter(self):
        """Test adding parameters to template."""
        template = MaterialTemplate(name="Test")
        param = MaterialParameter(
            name="metallic",
            param_type=ParameterType.FLOAT,
            default_value=0.0,
        )
        template.add_parameter(param)
        assert "metallic" in template.parameters
        assert template.version == 1

    def test_add_duplicate_parameter_fails(self):
        """Test that duplicate parameters are rejected."""
        template = MaterialTemplate(name="Test")
        param = MaterialParameter(
            name="metallic",
            param_type=ParameterType.FLOAT,
            default_value=0.0,
        )
        template.add_parameter(param)
        with pytest.raises(ValueError, match="already exists"):
            template.add_parameter(param)

    def test_remove_parameter(self):
        """Test removing parameters."""
        template = MaterialTemplate(name="Test")
        param = MaterialParameter(
            name="metallic",
            param_type=ParameterType.FLOAT,
            default_value=0.0,
        )
        template.add_parameter(param)
        template.remove_parameter("metallic")
        assert "metallic" not in template.parameters

    def test_create_instance(self):
        """Test creating instance from template."""
        template = MaterialTemplate(name="Test")
        instance = template.create_instance("MyInstance")
        assert instance.template == template
        assert instance.name == "MyInstance"
        assert instance in template.get_instances()

    def test_get_default_values(self):
        """Test getting default values."""
        template = MaterialTemplate(name="Test")
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
        ))
        defaults = template.get_default_values()
        assert defaults["roughness"] == 0.5


class TestMaterialInstance:
    """Test MaterialInstance parameter overrides."""

    def test_create_instance(self):
        """Test instance creation."""
        template = MaterialTemplate(name="Test")
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
        ))
        instance = template.create_instance()
        assert instance.template == template
        assert instance.dirty.any_dirty()  # New instances start dirty

    def test_get_default_parameter(self):
        """Test getting parameter default from template."""
        template = MaterialTemplate(name="Test")
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
        ))
        instance = template.create_instance()
        assert instance.get_parameter("roughness") == 0.5

    def test_set_parameter_override(self):
        """Test setting parameter override."""
        template = MaterialTemplate(name="Test")
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
            min_value=0.0,
            max_value=1.0,
        ))
        instance = template.create_instance()
        instance.dirty.clear_all()

        instance.set_parameter("roughness", 0.8)
        assert instance.get_parameter("roughness") == 0.8
        assert instance.dirty.parameters

    def test_set_invalid_parameter_fails(self):
        """Test that invalid parameter values are rejected."""
        template = MaterialTemplate(name="Test")
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
            min_value=0.0,
            max_value=1.0,
        ))
        instance = template.create_instance()
        with pytest.raises(ValueError):
            instance.set_parameter("roughness", 2.0, clamp=False)

    def test_set_unknown_parameter_fails(self):
        """Test that unknown parameters are rejected."""
        template = MaterialTemplate(name="Test")
        instance = template.create_instance()
        with pytest.raises(KeyError, match="Unknown parameter"):
            instance.set_parameter("nonexistent", 1.0)

    def test_clear_override(self):
        """Test clearing parameter override."""
        template = MaterialTemplate(name="Test")
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
        ))
        instance = template.create_instance()
        instance.set_parameter("roughness", 0.8)
        instance.clear_override("roughness")
        assert instance.get_parameter("roughness") == 0.5

    def test_enable_feature(self):
        """Test enabling shader features."""
        template = MaterialTemplate(name="Test")
        instance = template.create_instance()
        instance.dirty.clear_all()

        instance.enable_feature("NORMAL_MAPPING")
        assert "NORMAL_MAPPING" in instance.features
        assert instance.dirty.shader

    def test_clone_instance(self):
        """Test cloning an instance."""
        template = MaterialTemplate(name="Test")
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
        ))
        instance = template.create_instance()
        instance.set_parameter("roughness", 0.8)
        instance.enable_feature("TEST")

        clone = instance.clone("ClonedInstance")
        assert clone.get_parameter("roughness") == 0.8
        assert "TEST" in clone.features
        assert clone.name == "ClonedInstance"
        assert clone.instance_id != instance.instance_id


class TestMaterialFunction:
    """Test MaterialFunction shader snippets."""

    def test_create_function(self):
        """Test function creation."""
        func = MaterialFunction(
            name="Fresnel",
            code="float fresnel = pow(1.0 - NdotV, 5.0);",
            description="Basic Fresnel effect",
        )
        assert func.name == "Fresnel"
        assert "fresnel" in func.code
        assert func.function_id is not None

    def test_add_inputs_outputs(self):
        """Test adding inputs and outputs."""
        func = MaterialFunction(name="Test", code="")
        func.add_input(MaterialParameter(
            name="normal",
            param_type=ParameterType.VEC3,
            default_value=None,
        ))
        func.add_output(MaterialParameter(
            name="result",
            param_type=ParameterType.FLOAT,
            default_value=None,
        ))
        assert len(func.inputs) == 1
        assert len(func.outputs) == 1

    def test_dependency_resolution(self):
        """Test transitive dependency resolution."""
        func_a = MaterialFunction(name="A", code="// A")
        func_b = MaterialFunction(name="B", code="// B")
        func_c = MaterialFunction(name="C", code="// C")

        func_b.add_dependency(func_a)
        func_c.add_dependency(func_b)

        deps = func_c.get_dependencies()
        assert len(deps) == 2
        assert func_a in deps
        assert func_b in deps


class TestMaterialLayer:
    """Test MaterialLayer compositing."""

    def test_create_layer(self):
        """Test layer creation."""
        layer = MaterialLayer(name="Detail")
        assert layer.name == "Detail"
        assert layer.enabled
        assert layer.blend_settings.blend_weight == 1.0

    def test_set_layer_parameter(self):
        """Test setting layer parameters."""
        layer = MaterialLayer(name="Detail")
        layer.set_parameter("tiling", 4.0)
        assert layer.get_parameter("tiling") == 4.0

    def test_disable_layer(self):
        """Test disabling layer."""
        layer = MaterialLayer(name="Detail")
        layer.enabled = False
        assert not layer.enabled


class TestMaterialSystem:
    """Test MaterialSystem resource management."""

    def test_create_system(self):
        """Test system creation."""
        system = MaterialSystem()
        assert len(system.get_all_templates()) == 0
        assert len(system.get_all_instances()) == 0

    def test_register_template(self):
        """Test registering templates."""
        system = MaterialSystem()
        template = MaterialTemplate(name="Test")
        system.register_template(template)
        assert system.get_template(template.template_id) == template

    def test_get_template_by_name(self):
        """Test finding template by name."""
        system = MaterialSystem()
        template = MaterialTemplate(name="TestMaterial")
        system.register_template(template)
        found = system.get_template_by_name("TestMaterial")
        assert found == template

    def test_create_template_via_system(self):
        """Test creating template through system."""
        system = MaterialSystem()
        template = system.create_template(
            name="Metal",
            domain=MaterialDomain.SURFACE,
            blend_mode=BlendMode.OPAQUE,
        )
        assert template in system.get_all_templates()

    def test_create_instance_via_system(self):
        """Test creating instance through system."""
        system = MaterialSystem()
        template = system.create_template(name="Metal")
        instance = system.create_instance(template, "MetalInstance")
        assert instance in system.get_all_instances()
        assert instance in system.get_dirty_instances()

    def test_dirty_tracking(self):
        """Test dirty instance tracking."""
        system = MaterialSystem()
        template = system.create_template(name="Test")
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
        ))
        instance = system.create_instance(template)

        # New instance should be dirty
        dirty = system.get_dirty_instances()
        assert instance in dirty

        # Clear and verify
        system.clear_dirty_flags()
        dirty = system.get_dirty_instances()
        assert len(dirty) == 0

    def test_register_function(self):
        """Test registering material functions."""
        system = MaterialSystem()
        func = MaterialFunction(name="Fresnel", code="// fresnel")
        system.register_function(func)
        assert system.get_function("Fresnel") == func

    def test_hot_reload_callbacks(self):
        """Test hot-reload notification."""
        system = MaterialSystem()
        template = system.create_template(name="Test")
        instance = system.create_instance(template)
        instance.dirty.clear_all()

        callback_called = []

        def on_change(t):
            callback_called.append(t)

        system.on_template_changed(on_change)
        system.notify_template_changed(template)

        assert len(callback_called) == 1
        assert callback_called[0] == template
        assert instance.dirty.shader  # Instance should be marked dirty

    def test_enable_disable_hot_reload(self):
        """Test enabling and disabling hot-reload."""
        system = MaterialSystem()
        system.enable_hot_reload()
        # Internal state toggle (no public getter, but no error is success)
        system.disable_hot_reload()


class TestMaterialParameterExtended:
    """Extended tests for MaterialParameter type coverage."""

    def test_float_type_validation(self):
        """Test float parameter accepts int and float."""
        param = MaterialParameter(
            name="test",
            param_type=ParameterType.FLOAT,
            default_value=0.0,
        )
        is_valid, _ = param.validate(0.5)
        assert is_valid
        # int should also be valid for float params
        is_valid, _ = param.validate(42)
        assert is_valid

    def test_float_type_rejects_string(self):
        """Test float parameter rejects string."""
        param = MaterialParameter(
            name="test",
            param_type=ParameterType.FLOAT,
            default_value=0.0,
        )
        is_valid, error = param.validate("not_a_number")
        assert not is_valid

    def test_int_parameter(self):
        """Test int parameter type."""
        param = MaterialParameter(
            name="count",
            param_type=ParameterType.INT,
            default_value=0,
            min_value=0,
            max_value=100,
        )
        assert param.name == "count"
        assert param.param_type == ParameterType.INT
        assert param.default_value == 0

        is_valid, _ = param.validate(50)
        assert is_valid

        is_valid, error = param.validate(150)
        assert not is_valid
        assert "above maximum" in error

    def test_int_rejects_float(self):
        """Test int parameter rejects float values."""
        param = MaterialParameter(
            name="count",
            param_type=ParameterType.INT,
            default_value=0,
        )
        is_valid, error = param.validate(1.5)
        assert not is_valid
        assert "expected" in error

    def test_bool_parameter(self):
        """Test bool parameter type."""
        param = MaterialParameter(
            name="enable",
            param_type=ParameterType.BOOL,
            default_value=True,
        )
        assert param.param_type == ParameterType.BOOL
        assert param.default_value is True

        is_valid, _ = param.validate(True)
        assert is_valid

        is_valid, _ = param.validate(False)
        assert is_valid

    def test_bool_rejects_non_bool(self):
        """Test bool parameter rejects non-bool."""
        param = MaterialParameter(
            name="enable",
            param_type=ParameterType.BOOL,
            default_value=False,
        )
        is_valid, error = param.validate(1)
        assert not is_valid

    def test_vec2_type_validation(self):
        """Test vec2 parameter type access."""
        from engine.core.math.vec import Vec2
        param = MaterialParameter(
            name="uv_scale",
            param_type=ParameterType.VEC2,
            default_value=None,
        )
        assert param.param_type == ParameterType.VEC2

    def test_vec3_type_validation(self):
        """Test vec3 parameter type access."""
        from engine.core.math.vec import Vec3
        param = MaterialParameter(
            name="normal",
            param_type=ParameterType.VEC3,
            default_value=None,
        )
        assert param.param_type == ParameterType.VEC3

    def test_vec4_type_validation(self):
        """Test vec4 parameter type access."""
        from engine.core.math.vec import Vec4
        param = MaterialParameter(
            name="base_color",
            param_type=ParameterType.VEC4,
            default_value=None,
        )
        assert param.param_type == ParameterType.VEC4

    def test_clamp_no_op_for_non_numeric(self):
        """Test that clamp is no-op for non-numeric types."""
        param = MaterialParameter(
            name="flag",
            param_type=ParameterType.BOOL,
            default_value=False,
        )
        result = param.clamp(True)
        assert result is True  # Unchanged


class TestMaterialTemplateExtended:
    """Extended tests for MaterialTemplate edge cases."""

    def test_add_function_to_template(self):
        """Test adding material functions to template."""
        template = MaterialTemplate(name="Test")
        func = MaterialFunction(name="Fresnel", code="// fresnel")
        template.add_function(func)
        assert template.version == 1  # add_function increments version

    def test_remove_nonexistent_parameter(self):
        """Test removing nonexistent parameter raises."""
        template = MaterialTemplate(name="Test")
        with pytest.raises(KeyError, match="not found"):
            template.remove_parameter("nonexistent")

    def test_compute_permutation_key(self):
        """Test permutation key computation."""
        template = MaterialTemplate(name="Test")
        key1 = template.compute_permutation_key({"NORMAL_MAP", "AO"})
        key2 = template.compute_permutation_key({"AO", "NORMAL_MAP"})
        # Same features should produce same key regardless of order
        assert key1 == key2

    def test_compute_permutation_key_different_features(self):
        """Test different features produce different keys."""
        template = MaterialTemplate(name="Test")
        key1 = template.compute_permutation_key({"NORMAL_MAP"})
        key2 = template.compute_permutation_key({"AO"})
        assert key1 != key2

    def test_compute_permutation_key_empty(self):
        """Test empty permutation key."""
        template = MaterialTemplate(name="Test")
        key = template.compute_permutation_key(set())
        assert isinstance(key, int)

    def test_get_instances_empty(self):
        """Test getting instances when none exist."""
        template = MaterialTemplate(name="Test")
        assert len(template.get_instances()) == 0

    def test_repr(self):
        """Test template string representation."""
        template = MaterialTemplate(
            name="Metal",
            domain=MaterialDomain.SURFACE,
            blend_mode=BlendMode.OPAQUE,
        )
        rep = repr(template)
        assert "MaterialTemplate" in rep
        assert "Metal" in rep
        assert MaterialDomain.SURFACE.value in rep


class TestMaterialInstanceExtended:
    """Extended tests for MaterialInstance override semantics."""

    def test_texture_parameter_sets_textures_dirty(self):
        """Test texture parameter sets TEXTURES flag."""
        template = MaterialTemplate(name="Test")
        template.add_parameter(MaterialParameter(
            name="albedo",
            param_type=ParameterType.TEXTURE_2D,
            default_value=None,
        ))
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
        ))
        instance = template.create_instance()
        instance.dirty.clear_all()

        # Texture parameter should set textures flag
        instance.set_parameter("albedo", "textures/albedo.png")
        assert instance.dirty.textures
        assert not instance.dirty.parameters  # Not a scalar param

        # Float parameter should set parameters flag
        instance.dirty.clear_all()
        instance.set_parameter("roughness", 0.8)
        assert instance.dirty.parameters
        assert not instance.dirty.textures

    def test_get_all_parameters_merges_overrides(self):
        """Test get_all_parameters merges overrides with defaults."""
        template = MaterialTemplate(name="Test")
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
        ))
        template.add_parameter(MaterialParameter(
            name="metallic",
            param_type=ParameterType.FLOAT,
            default_value=0.0,
        ))
        instance = template.create_instance()
        instance.set_parameter("metallic", 0.8)

        all_params = instance.get_all_parameters()
        assert all_params["roughness"] == 0.5  # Default
        assert all_params["metallic"] == 0.8    # Override

    def test_get_all_parameters_no_overrides(self):
        """Test get_all_parameters without overrides."""
        template = MaterialTemplate(name="Test")
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
        ))
        instance = template.create_instance()
        all_params = instance.get_all_parameters()
        assert all_params["roughness"] == 0.5

    def test_clear_all_overrides(self):
        """Test clearing all overrides."""
        template = MaterialTemplate(name="Test")
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
        ))
        template.add_parameter(MaterialParameter(
            name="metallic",
            param_type=ParameterType.FLOAT,
            default_value=0.0,
        ))
        instance = template.create_instance()
        instance.set_parameter("roughness", 0.8)
        instance.set_parameter("metallic", 0.5)
        instance.dirty.clear_all()

        instance.clear_all_overrides()
        assert instance.get_parameter("roughness") == 0.5
        assert instance.get_parameter("metallic") == 0.0
        assert instance.dirty.any_dirty()

    def test_disable_feature_sets_shader_dirty(self):
        """Test disabling feature sets shader dirty."""
        template = MaterialTemplate(name="Test")
        instance = template.create_instance()
        instance.enable_feature("NORMAL_MAP")
        instance.dirty.clear_all()

        instance.disable_feature("NORMAL_MAP")
        assert "NORMAL_MAP" not in instance.features
        assert instance.dirty.shader

    def test_layer_push_pop(self):
        """Test layer stack operations on instance."""
        template = MaterialTemplate(name="Test")
        instance = template.create_instance()

        layer1 = MaterialLayer(name="Base")
        layer2 = MaterialLayer(name="Detail")
        assert len(instance.get_layers()) == 0

        instance.push_layer(layer1)
        assert len(instance.get_layers()) == 1
        assert instance.get_layers()[0].name == "Base"

        instance.push_layer(layer2)
        assert len(instance.get_layers()) == 2

        # Pop returns in LIFO order
        popped = instance.pop_layer()
        assert popped is layer2
        assert len(instance.get_layers()) == 1

    def test_layer_push_sets_dirty(self):
        """Test that pushing a layer marks all dirty."""
        template = MaterialTemplate(name="Test")
        instance = template.create_instance()
        instance.dirty.clear_all()

        layer = MaterialLayer(name="Test")
        instance.push_layer(layer)
        assert instance.dirty.any_dirty()

    def test_layer_pop_empty_returns_none(self):
        """Test popping from empty layer stack returns None."""
        template = MaterialTemplate(name="Test")
        instance = template.create_instance()
        result = instance.pop_layer()
        assert result is None

    def test_get_unknown_parameter_raises(self):
        """Test getting unknown parameter raises KeyError."""
        template = MaterialTemplate(name="Test")
        instance = template.create_instance()
        with pytest.raises(KeyError, match="Unknown parameter"):
            instance.get_parameter("nonexistent")

    def test_get_permutation_key(self):
        """Test instance permutation key computation."""
        template = MaterialTemplate(name="Test")
        instance = template.create_instance()
        instance.enable_feature("FEATURE_A")
        key = instance.get_permutation_key()
        assert isinstance(key, int)


class TestMaterialFunctionExtended:
    """Extended tests for MaterialFunction dependencies and code."""

    def test_get_full_code_no_deps(self):
        """Test get_full_code without dependencies."""
        func = MaterialFunction(
            name="Simple",
            code="float simple() { return 1.0; }",
        )
        full = func.get_full_code()
        assert "simple" in full
        assert full == func.code  # No deps, should be same

    def test_get_full_code_with_deps(self):
        """Test get_full_code includes dependency code."""
        func_a = MaterialFunction(
            name="A",
            code="float funcA() { return 1.0; }",
        )
        func_b = MaterialFunction(
            name="B",
            code="float funcB() { return funcA(); }",
        )
        func_b.add_dependency(func_a)

        full = func_b.get_full_code()
        assert "funcA" in full
        assert "funcB" in full

    def test_circular_dependency_handling(self):
        """Test circular dependency doesn't cause infinite recursion."""
        func_a = MaterialFunction(name="A", code="// A")
        func_b = MaterialFunction(name="B", code="// B")

        # Create cycle: A -> B and B -> A
        func_a.add_dependency(func_b)
        func_b.add_dependency(func_a)

        # Should not infinite loop; get_dependencies should terminate
        deps_a = func_a.get_dependencies()
        deps_b = func_b.get_dependencies()
        # Both should terminate and return results
        assert isinstance(deps_a, list)
        assert isinstance(deps_b, list)

    def test_diamond_dependency(self):
        """Test diamond dependency pattern."""
        func_base = MaterialFunction(name="Base", code="// Base")
        func_left = MaterialFunction(name="Left", code="// Left")
        func_right = MaterialFunction(name="Right", code="// Right")
        func_top = MaterialFunction(name="Top", code="// Top")

        # Diamond: Top -> Left -> Base, Top -> Right -> Base
        func_left.add_dependency(func_base)
        func_right.add_dependency(func_base)
        func_top.add_dependency(func_left)
        func_top.add_dependency(func_right)

        deps = func_top.get_dependencies()
        # Base should appear only once despite being in both branches
        assert len(deps) == 3  # Base, Left, Right (order may vary)
        assert func_base in deps

    def test_function_id_uniqueness(self):
        """Test each function gets a unique ID."""
        func1 = MaterialFunction(name="A", code="// A")
        func2 = MaterialFunction(name="A", code="// A")
        assert func1.function_id != func2.function_id

    def test_add_same_dependency_twice(self):
        """Test adding same dependency twice is idempotent."""
        func_a = MaterialFunction(name="A", code="// A")
        func_b = MaterialFunction(name="B", code="// B")
        func_a.add_dependency(func_b)
        func_a.add_dependency(func_b)  # Should be no-op

        assert len(func_a.get_dependencies()) == 1


class TestMaterialLayerExtended:
    """Extended tests for MaterialLayer composition."""

    def test_layer_blend_settings_default(self):
        """Test default blend settings."""
        layer = MaterialLayer(name="Base")
        assert layer.blend_settings.blend_weight == 1.0
        assert layer.blend_settings.blend_mode == "lerp"
        assert layer.blend_settings.mask_channel is None

    def test_layer_custom_blend_settings(self):
        """Test custom blend settings."""
        settings = LayerBlendSettings(
            blend_weight=0.7,
            blend_mode="add",
            mask_channel="r",
        )
        layer = MaterialLayer(name="Detail", blend_settings=settings)
        assert layer.blend_settings.blend_weight == 0.7
        assert layer.blend_settings.blend_mode == "add"
        assert layer.blend_settings.mask_channel == "r"

    def test_layer_mask_texture(self):
        """Test layer mask texture assignment."""
        layer = MaterialLayer(name="Detail")
        layer.set_mask_texture("textures/layer_mask.png")
        # Internal state set; no public getter but no error expected
        layer.set_mask_texture(None)

    def test_layer_get_parameter_default(self):
        """Test getting layer parameter with default."""
        layer = MaterialLayer(name="Test")
        result = layer.get_parameter("nonexistent", 42)
        assert result == 42

    def test_layer_repr(self):
        """Test layer string representation."""
        layer = MaterialLayer(name="Detail")
        layer.enabled = False
        rep = repr(layer)
        assert "MaterialLayer" in rep
        assert "Detail" in rep
        assert "False" in rep or "false" in rep


class TestMaterialSystemExtended:
    """Extended tests for MaterialSystem resource management."""

    def test_duplicate_template_registration(self):
        """Test duplicate template registration raises."""
        system = MaterialSystem()
        template = MaterialTemplate(name="Test")
        system.register_template(template)
        with pytest.raises(ValueError, match="already registered"):
            system.register_template(template)

    def test_unregister_template(self):
        """Test unregistering a template."""
        system = MaterialSystem()
        template = MaterialTemplate(name="Test")
        system.register_template(template)
        assert system.get_template(template.template_id) is not None

        system.unregister_template(template.template_id)
        assert system.get_template(template.template_id) is None
        assert len(system.get_all_templates()) == 0

    def test_unregister_nonexistent_template(self):
        """Test unregistering nonexistent template raises."""
        system = MaterialSystem()
        with pytest.raises(KeyError, match="not found"):
            system.unregister_template("nonexistent")

    def test_unregister_template_removes_instances(self):
        """Test unregistering template also removes its instances."""
        system = MaterialSystem()
        template = MaterialTemplate(name="Test")
        system.register_template(template)
        instance = system.create_instance(template)
        assert instance.instance_id in [
            inst.instance_id for inst in system.get_all_instances()
        ]

        system.unregister_template(template.template_id)
        assert instance.instance_id not in [
            inst.instance_id for inst in system.get_all_instances()
        ]

    def test_get_instance(self):
        """Test retrieving instance by ID."""
        system = MaterialSystem()
        template = MaterialTemplate(name="Test")
        system.register_template(template)
        instance = system.create_instance(template)

        retrieved = system.get_instance(instance.instance_id)
        assert retrieved is instance

    def test_get_instance_invalid(self):
        """Test retrieving invalid instance returns None."""
        system = MaterialSystem()
        result = system.get_instance("nonexistent")
        assert result is None

    def test_unregister_instance(self):
        """Test unregistering an instance."""
        system = MaterialSystem()
        template = MaterialTemplate(name="Test")
        system.register_template(template)
        instance = system.create_instance(template)

        system.unregister_instance(instance.instance_id)
        assert system.get_instance(instance.instance_id) is None

    def test_mark_instance_dirty(self):
        """Test marking instance dirty after flags cleared."""
        system = MaterialSystem()
        template = MaterialTemplate(name="Test")
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
        ))
        instance = system.create_instance(template)

        system.clear_dirty_flags()
        assert len(system.get_dirty_instances()) == 0

        # Make the instance actually dirty, then re-add to tracking
        instance.set_parameter("roughness", 0.3)
        system.mark_instance_dirty(instance.instance_id)
        dirty = system.get_dirty_instances()
        assert instance in dirty

    def test_notify_template_changed_propagates_to_instances(self):
        """Test template change notification propagates dirty flag."""
        system = MaterialSystem()
        template = system.create_template(name="Test")
        instance1 = system.create_instance(template)
        instance2 = system.create_instance(template)

        instance1.dirty.clear_all()
        instance2.dirty.clear_all()

        system.notify_template_changed(template)

        assert instance1.dirty.shader
        assert instance2.dirty.shader

    def test_get_template_by_name_not_found(self):
        """Test getting template by name that doesn't exist."""
        system = MaterialSystem()
        result = system.get_template_by_name("NonExistent")
        assert result is None

    def test_get_dirty_instances_removes_clean(self):
        """Test that get_dirty_instances removes clean instances."""
        system = MaterialSystem()
        template = MaterialTemplate(name="Test")
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
        ))
        instance = system.create_instance(template)

        # Make instance dirty, add to tracking
        instance.set_parameter("roughness", 0.3)
        system.mark_instance_dirty(instance.instance_id)
        dirty = system.get_dirty_instances()
        assert len(dirty) == 1

        # After clearing, dirty set should be empty
        system.clear_dirty_flags()
        assert len(system.get_dirty_instances()) == 0

    def test_create_instance_by_name(self):
        """Test creating instance from template by name."""
        system = MaterialSystem()
        system.create_template(name="Metal")
        instance = system.create_instance("Metal", "MyMetal")
        assert instance.name == "MyMetal"
        assert instance.template.name == "Metal"

    def test_create_instance_by_id(self):
        """Test creating instance from template by ID."""
        system = MaterialSystem()
        template = system.create_template(name="Metal")
        instance = system.create_instance(
            template.template_id,
            "ByIdInstance",
        )
        assert instance.name == "ByIdInstance"

    def test_create_instance_template_not_found(self):
        """Test creating instance with nonexistent template raises."""
        system = MaterialSystem()
        with pytest.raises(KeyError, match="Template not found"):
            system.create_instance("nonexistent_template_id")


class TestDirtyFlagsExtended:
    """Extended tests for DirtyFlags integration and propagation."""

    def test_dirty_flags_initial_clean(self):
        """Test DirtyFlags initial clean state."""
        flags = DirtyFlags()
        assert not flags.parameters
        assert not flags.textures
        assert not flags.shader

    def test_dirty_flags_mark_all_via_instance(self):
        """Test that template invalidation marks instance dirty."""
        template = MaterialTemplate(name="Test")
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
        ))
        instance = template.create_instance()
        instance.dirty.clear_all()

        # Template parameter change triggers variant invalidation
        template.add_parameter(MaterialParameter(
            name="metallic",
            param_type=ParameterType.FLOAT,
            default_value=0.0,
        ))

        # Instances should have shader dirty flag set
        assert instance.dirty.shader

    def test_clear_flags_atomic(self):
        """Test atomic clearing of all flags."""
        flags = DirtyFlags()
        flags.parameters = True
        flags.textures = True

        flags.clear_all()
        assert not flags.any_dirty()
        assert not flags.parameters
        assert not flags.textures
        assert not flags.shader

    def test_dirty_flags_repr_clean(self):
        """Test clean state is not dirty."""
        flags = DirtyFlags()
        flags.parameters = False
        flags.textures = False
        flags.shader = False
        assert not flags.any_dirty()

    def test_propagation_template_remove_parameter(self):
        """Test template parameter removal propagates to instances."""
        template = MaterialTemplate(name="Test")
        template.add_parameter(MaterialParameter(
            name="roughness",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
        ))
        instance = template.create_instance()
        instance.dirty.clear_all()

        template.remove_parameter("roughness")
        assert instance.dirty.shader
