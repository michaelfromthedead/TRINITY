"""Contract tests for the Core Material System (CLEANROOM BLACKBOX).

Tests the public API contract of MaterialTemplate, MaterialInstance,
MaterialFunction, MaterialLayer, MaterialSystem, and DirtyFlags.

Designed without knowledge of implementation internals.
Contract sources:
  - engine/rendering/materials/__init__.py (public API exports)
  - PHASE_1_TODO.md (T-MAT-1.1 through T-MAT-1.6 acceptance criteria)
  - GAPSET_4_MATERIALS/PHASE_N_TODO.md (architecture context)
"""

import pytest
from engine.rendering.materials import (
    MaterialTemplate,
    MaterialInstance,
    MaterialParameter,
    MaterialFunction,
    MaterialLayer,
    MaterialSystem,
    ParameterType,
    MaterialDomain,
    BlendMode,
    ShadingModel,
    DirtyFlags,
)
from engine.rendering.materials.material_system import LayerBlendSettings


# =============================================================================
# T-MAT-1.1: MaterialTemplate API
# =============================================================================

class TestMaterialTemplateCreation:
    """MaterialTemplate correctly defines parameter schemas and default values."""

    def test_create_default_template(self):
        """A template can be created with just a name, using defaults."""
        t = MaterialTemplate("test")
        assert t.name == "test"
        assert t.template_id is not None
        assert isinstance(t.template_id, str)
        assert len(t.template_id) > 0
        assert t.domain == MaterialDomain.SURFACE
        assert t.blend_mode == BlendMode.OPAQUE
        assert t.shading_model == ShadingModel.DEFAULT_LIT
        assert not t.two_sided
        assert not t.wireframe

    def test_create_template_with_all_params(self):
        """A template accepts all documented construction parameters."""
        t = MaterialTemplate(
            "full",
            domain=MaterialDomain.POST_PROCESS,
            blend_mode=BlendMode.TRANSLUCENT,
            shading_model=ShadingModel.UNLIT,
            two_sided=True,
            wireframe=False,
        )
        assert t.name == "full"
        assert t.domain == MaterialDomain.POST_PROCESS
        assert t.blend_mode == BlendMode.TRANSLUCENT
        assert t.shading_model == ShadingModel.UNLIT
        assert t.two_sided
        assert not t.wireframe

    def test_create_template_with_vertex_and_fragment_shaders(self):
        """Template accepts optional vertex and fragment shader source."""
        v_shader = "// vertex shader"
        f_shader = "// fragment shader"
        t = MaterialTemplate("shader", vertex_shader=v_shader, fragment_shader=f_shader)
        # Shaders are accepted at construction without raising
        assert t.name == "shader"

    def test_each_template_has_unique_id(self):
        """Each template receives a unique identifier."""
        t1 = MaterialTemplate("a")
        t2 = MaterialTemplate("b")
        assert t1.template_id != t2.template_id

    def test_template_ids_stable_per_instance(self):
        """The template_id is stable for the lifetime of the object."""
        t = MaterialTemplate("stable")
        tid = t.template_id
        _ = t.name
        assert t.template_id == tid


class TestMaterialParameterLifecycle:
    """Parameter schema management on MaterialTemplate."""

    def test_add_parameter(self):
        """A parameter can be added and is accessible."""
        t = MaterialTemplate("mat")
        p = MaterialParameter("roughness", ParameterType.FLOAT, 0.5, 0.0, 1.0)
        t.add_parameter(p)
        # Parameter is stored and its default is retrievable
        defaults = t.get_default_values()
        assert "roughness" in defaults
        assert defaults["roughness"] == 0.5

    def test_add_multiple_parameters(self):
        """Multiple parameters of different types can be added."""
        t = MaterialTemplate("multi")
        params = [
            MaterialParameter("a", ParameterType.FLOAT, 1.0, 0.0, 10.0, group="Numbers"),
            MaterialParameter("b", ParameterType.INT, 5, 0, 100, group="Numbers"),
            MaterialParameter("c", ParameterType.BOOL, False, group="Flags"),
            MaterialParameter("d", ParameterType.VEC3, (1.0, 1.0, 1.0), group="Colors"),
        ]
        for p in params:
            t.add_parameter(p)
        defaults = t.get_default_values()
        assert len(defaults) == 4
        assert defaults["a"] == 1.0
        assert defaults["b"] == 5
        assert not defaults["c"]

    def test_remove_parameter_by_name(self):
        """A parameter can be removed by its name."""
        t = MaterialTemplate("mat")
        p = MaterialParameter("roughness", ParameterType.FLOAT, 0.5)
        t.add_parameter(p)
        t.remove_parameter("roughness")
        defaults = t.get_default_values()
        assert "roughness" not in defaults

    def test_remove_nonexistent_parameter_raises(self):
        """Removing a parameter that doesn't exist raises an error."""
        t = MaterialTemplate("mat")
        with pytest.raises((KeyError, ValueError, LookupError)):
            t.remove_parameter("does_not_exist")

    def test_parameter_fields_are_preserved(self):
        """All MaterialParameter fields are stored and retrievable."""
        p = MaterialParameter(
            name="test_param",
            param_type=ParameterType.FLOAT,
            default_value=0.5,
            min_value=0.0,
            max_value=1.0,
            description="A test parameter",
            group="TestGroup",
            hidden=True,
        )
        assert p.name == "test_param"
        assert p.param_type == ParameterType.FLOAT
        assert p.default_value == 0.5
        assert p.min_value == 0.0
        assert p.max_value == 1.0
        assert p.description == "A test parameter"
        assert p.group == "TestGroup"
        assert p.hidden

    def test_parameter_optional_fields_default(self):
        """Optional MaterialParameter fields have sensible defaults."""
        p = MaterialParameter("simple", ParameterType.FLOAT, 0.0)
        assert p.description == ""
        assert p.group == "Default"
        assert not p.hidden


class TestParameterValidation:
    """Parameter value validation and clamping."""

    def test_validate_valid_value(self):
        """validate() returns (True, '') for in-range values."""
        p = MaterialParameter("t", ParameterType.FLOAT, 0.5, 0.0, 1.0)
        valid, msg = p.validate(0.5)
        assert valid
        assert msg == ""

    def test_validate_out_of_range_high(self):
        """validate() returns with message for values above max."""
        p = MaterialParameter("t", ParameterType.FLOAT, 0.5, 0.0, 1.0)
        valid, msg = p.validate(2.0)
        assert not valid
        assert msg is not None and len(msg) > 0

    def test_validate_out_of_range_low(self):
        """validate() returns with message for values below min."""
        p = MaterialParameter("t", ParameterType.FLOAT, 0.5, 0.0, 1.0)
        valid, msg = p.validate(-1.0)
        assert not valid
        assert msg is not None and len(msg) > 0

    def test_clamp_high_value(self):
        """clamp() brings out-of-range-high values to the maximum."""
        p = MaterialParameter("t", ParameterType.FLOAT, 0.5, 0.0, 1.0)
        assert p.clamp(2.0) == 1.0

    def test_clamp_low_value(self):
        """clamp() brings out-of-range-low values to the minimum."""
        p = MaterialParameter("t", ParameterType.FLOAT, 0.5, 0.0, 1.0)
        assert p.clamp(-1.0) == 0.0

    def test_clamp_in_range_value(self):
        """clamp() returns the value unchanged if within range."""
        p = MaterialParameter("t", ParameterType.FLOAT, 0.5, 0.0, 1.0)
        assert p.clamp(0.5) == 0.5

    def test_clamp_int_parameter(self):
        """clamp() works for integer parameters."""
        p = MaterialParameter("t", ParameterType.INT, 5, 0, 10)
        assert p.clamp(15) == 10
        assert p.clamp(-5) == 0
        assert p.clamp(7) == 7


class TestDefaultValues:
    """Template default value management."""

    def test_get_default_values_empty(self):
        """get_default_values() returns empty dict when no parameters."""
        t = MaterialTemplate("empty")
        assert t.get_default_values() == {}

    def test_get_default_values_with_params(self):
        """get_default_values() returns all parameter defaults."""
        t = MaterialTemplate("mat")
        t.add_parameter(MaterialParameter("a", ParameterType.FLOAT, 1.0))
        t.add_parameter(MaterialParameter("b", ParameterType.INT, 42))
        t.add_parameter(MaterialParameter("c", ParameterType.BOOL, True))
        defaults = t.get_default_values()
        assert defaults["a"] == 1.0
        assert defaults["b"] == 42
        assert defaults["c"]

    def test_get_default_values_after_remove(self):
        """Removing a parameter removes it from default values."""
        t = MaterialTemplate("mat")
        t.add_parameter(MaterialParameter("a", ParameterType.FLOAT, 1.0))
        t.add_parameter(MaterialParameter("b", ParameterType.FLOAT, 2.0))
        t.remove_parameter("a")
        defaults = t.get_default_values()
        assert "a" not in defaults
        assert defaults["b"] == 2.0


# =============================================================================
# T-MAT-1.2: MaterialInstance Override Semantics
# =============================================================================

class TestMaterialInstanceCreation:
    """MaterialInstance correctly inherits from template and supports overrides."""

    def test_create_instance_from_template(self):
        """An instance can be created from a template."""
        t = MaterialTemplate("base")
        inst = t.create_instance("my_instance")
        assert inst.name == "my_instance"
        assert inst.instance_id is not None
        assert isinstance(inst.instance_id, str)
        assert len(inst.instance_id) > 0
        assert inst.template is t

    def test_create_instance_auto_name(self):
        """An instance can be created without an explicit name."""
        t = MaterialTemplate("base")
        inst = t.create_instance()
        assert inst.name is not None

    def test_instance_inherits_template_parameters(self):
        """Instance inherits all template parameters by default."""
        t = MaterialTemplate("base")
        t.add_parameter(MaterialParameter("roughness", ParameterType.FLOAT, 0.5, 0.0, 1.0))
        t.add_parameter(MaterialParameter("metallic", ParameterType.FLOAT, 0.0, 0.0, 1.0))
        inst = t.create_instance("inst")
        params = inst.get_all_parameters()
        assert params["roughness"] == 0.5
        assert params["metallic"] == 0.0

    def test_each_instance_has_unique_id(self):
        """Each instance receives a unique identifier."""
        t = MaterialTemplate("base")
        i1 = t.create_instance("a")
        i2 = t.create_instance("b")
        assert i1.instance_id != i2.instance_id


class TestInstanceParameterOverrides:
    """Instance parameter override behavior."""

    def test_set_parameter_override(self):
        """Setting a parameter on the instance overrides the template default."""
        t = MaterialTemplate("base")
        t.add_parameter(MaterialParameter("roughness", ParameterType.FLOAT, 0.5, 0.0, 1.0))
        inst = t.create_instance("inst")
        inst.set_parameter("roughness", 0.8)
        assert inst.get_parameter("roughness") == 0.8

    def test_set_parameter_does_not_affect_template(self):
        """Instance overrides do not change the template's default values."""
        t = MaterialTemplate("base")
        t.add_parameter(MaterialParameter("roughness", ParameterType.FLOAT, 0.5, 0.0, 1.0))
        inst1 = t.create_instance("a")
        inst2 = t.create_instance("b")
        inst1.set_parameter("roughness", 0.8)
        assert t.get_default_values()["roughness"] == 0.5
        assert inst2.get_parameter("roughness") == 0.5

    def test_get_parameter_returns_template_default_when_not_overridden(self):
        """Instances return the template default for un-overridden parameters."""
        t = MaterialTemplate("base")
        t.add_parameter(MaterialParameter("a", ParameterType.FLOAT, 1.0))
        t.add_parameter(MaterialParameter("b", ParameterType.FLOAT, 2.0))
        inst = t.create_instance("inst")
        inst.set_parameter("a", 99.0)
        assert inst.get_parameter("a") == 99.0  # overridden
        assert inst.get_parameter("b") == 2.0   # inherited

    def test_clear_override_reverts_to_template_default(self):
        """Clearing an override restores the template default."""
        t = MaterialTemplate("base")
        t.add_parameter(MaterialParameter("roughness", ParameterType.FLOAT, 0.5, 0.0, 1.0))
        inst = t.create_instance("inst")
        inst.set_parameter("roughness", 0.8)
        inst.clear_override("roughness")
        assert inst.get_parameter("roughness") == 0.5

    def test_clear_all_overrides(self):
        """Clearing all overrides reverts every parameter to template defaults."""
        t = MaterialTemplate("base")
        t.add_parameter(MaterialParameter("a", ParameterType.FLOAT, 1.0))
        t.add_parameter(MaterialParameter("b", ParameterType.FLOAT, 2.0))
        inst = t.create_instance("inst")
        inst.set_parameter("a", 10.0)
        inst.set_parameter("b", 20.0)
        inst.clear_all_overrides()
        params = inst.get_all_parameters()
        assert params["a"] == 1.0
        assert params["b"] == 2.0

    def test_get_all_parameters_merged(self):
        """get_all_parameters returns merged template+override values."""
        t = MaterialTemplate("base")
        t.add_parameter(MaterialParameter("a", ParameterType.FLOAT, 1.0))
        t.add_parameter(MaterialParameter("b", ParameterType.FLOAT, 2.0))
        inst = t.create_instance("inst")
        inst.set_parameter("a", 99.0)
        all_params = inst.get_all_parameters()
        assert all_params["a"] == 99.0
        assert all_params["b"] == 2.0

    def test_clone_preserves_overrides(self):
        """Cloning an instance preserves the current parameter overrides."""
        t = MaterialTemplate("base")
        t.add_parameter(MaterialParameter("a", ParameterType.FLOAT, 1.0))
        inst = t.create_instance("original")
        inst.set_parameter("a", 42.0)
        cloned = inst.clone("copy")
        assert cloned.name == "copy"
        assert cloned.get_parameter("a") == 42.0

    def test_clone_inherits_new_name(self):
        """clone() assigns the given name to the new instance."""
        t = MaterialTemplate("base")
        inst = t.create_instance("original")
        cloned = inst.clone("copy")
        assert cloned.name == "copy"
        assert cloned.instance_id != inst.instance_id


class TestInstanceFeatures:
    """Instance feature toggle management."""

    def test_features_initial_empty(self):
        """Features start as an empty collection."""
        t = MaterialTemplate("base")
        inst = t.create_instance("inst")
        assert len(inst.features) == 0

    def test_enable_feature(self):
        """A feature can be enabled."""
        t = MaterialTemplate("base")
        inst = t.create_instance("inst")
        inst.enable_feature("DOUBLE_SIDED")
        assert "DOUBLE_SIDED" in inst.features

    def test_disable_feature(self):
        """A feature can be disabled."""
        t = MaterialTemplate("base")
        inst = t.create_instance("inst")
        inst.enable_feature("FEATURE_X")
        inst.disable_feature("FEATURE_X")
        assert "FEATURE_X" not in inst.features

    def test_disable_nonexistent_feature_does_not_raise(self):
        """Disabling a feature not in the set does not raise."""
        t = MaterialTemplate("base")
        inst = t.create_instance("inst")
        inst.disable_feature("NONEXISTENT")  # Should not raise

    def test_multiple_features(self):
        """Multiple independent features can be active simultaneously."""
        t = MaterialTemplate("base")
        inst = t.create_instance("inst")
        inst.enable_feature("A")
        inst.enable_feature("B")
        inst.enable_feature("C")
        assert "A" in inst.features
        assert "B" in inst.features
        assert "C" in inst.features


class TestPermutationKey:
    """Permutation key computation."""

    def test_template_permutation_key(self):
        """Template computes a permutation key from a set of features."""
        t = MaterialTemplate("base")
        key = t.compute_permutation_key({"FEATURE_A", "FEATURE_B"})
        assert isinstance(key, int)

    def test_template_permutation_key_differentiates_feature_sets(self):
        """Different feature sets produce different permutation keys."""
        t = MaterialTemplate("base")
        k1 = t.compute_permutation_key({"A"})
        k2 = t.compute_permutation_key({"B"})
        assert k1 != k2

    def test_instance_permutation_key(self):
        """Instance computes a permutation key from its enabled features."""
        t = MaterialTemplate("base")
        inst = t.create_instance("inst")
        inst.enable_feature("FEATURE_A")
        key = inst.get_permutation_key()
        assert isinstance(key, int)

    def test_permutation_key_changes_with_features(self):
        """Instance permutation key changes when features change."""
        t = MaterialTemplate("base")
        inst = t.create_instance("inst")
        k1 = inst.get_permutation_key()
        inst.enable_feature("FEATURE_X")
        k2 = inst.get_permutation_key()
        assert k1 != k2


# =============================================================================
# T-MAT-1.3: MaterialFunction Dependencies
# =============================================================================

class TestMaterialFunction:
    """MaterialFunction dependency tracking."""

    def test_create_function(self):
        """A function can be created with name, code, and description."""
        fn = MaterialFunction("my_func", "// glsl code", "test function")
        assert fn.name == "my_func"
        assert fn.function_id is not None
        assert isinstance(fn.function_id, str)
        assert fn.description == "test function"
        assert fn.code == "// glsl code"
        assert fn.inputs == []
        assert fn.outputs == []

    def test_create_function_default_description(self):
        """Function description defaults to empty string."""
        fn = MaterialFunction("bare", "// code")
        assert fn.description == ""

    def test_add_dependency(self):
        """A dependency on another function can be declared."""
        fn1 = MaterialFunction("base", "// base")
        fn2 = MaterialFunction("dep", "// dep")
        fn1.add_dependency(fn2)
        deps = fn1.get_dependencies()
        assert len(deps) == 1
        assert deps[0] is fn2

    def test_multiple_dependencies(self):
        """Multiple dependencies can be declared."""
        fn1 = MaterialFunction("main", "// main")
        fn2 = MaterialFunction("a", "// a")
        fn3 = MaterialFunction("b", "// b")
        fn1.add_dependency(fn2)
        fn1.add_dependency(fn3)
        assert len(fn1.get_dependencies()) == 2

    def test_get_full_code_includes_code(self):
        """get_full_code() returns the function body."""
        fn = MaterialFunction("test", "// body")
        full = fn.get_full_code()
        assert isinstance(full, str)
        assert len(full) > 0

    def test_input_and_output_ports(self):
        """Functions can declare input and output ports."""
        fn = MaterialFunction("fn", "// code")
        inp = MaterialParameter("input_val", ParameterType.FLOAT, 0.0)
        out = MaterialParameter("output_val", ParameterType.FLOAT, 0.0)
        fn.add_input(inp)
        fn.add_output(out)
        # After adding, get_full_code should not raise
        fn.get_full_code()

    def test_each_function_has_unique_id(self):
        """Each function receives a unique identifier."""
        fn1 = MaterialFunction("a", "// a")
        fn2 = MaterialFunction("b", "// b")
        assert fn1.function_id != fn2.function_id


class TestFunctionOnTemplate:
    """Function attachment to templates."""

    def test_add_function_to_template(self):
        """A function can be added to a template."""
        t = MaterialTemplate("mat")
        fn = MaterialFunction("fn", "// code")
        t.add_function(fn)  # Should not raise

    def test_functions_in_permutation_key(self):
        """Permutation key computation succeeds when functions are attached."""
        t = MaterialTemplate("mat")
        fn = MaterialFunction("fn", "// code")
        t.add_function(fn)
        key = t.compute_permutation_key({"A"})
        assert isinstance(key, int)


# =============================================================================
# T-MAT-1.4: MaterialLayer Composition
# =============================================================================

class TestMaterialLayer:
    """MaterialLayer stacking and blend modes."""

    def test_create_layer(self):
        """A layer can be created with a name."""
        layer = MaterialLayer("detail_layer")
        assert layer.name == "detail_layer"
        assert layer.layer_id is not None
        assert isinstance(layer.layer_id, str)
        assert layer.layer_id != ""

    def test_create_layer_with_blend_settings(self):
        """A layer accepts LayerBlendSettings."""
        blend = LayerBlendSettings(blend_weight=0.5, blend_mode="lerp")
        layer = MaterialLayer("detail", blend_settings=blend)
        assert layer.blend_settings.blend_weight == 0.5
        assert layer.blend_settings.blend_mode == "lerp"

    def test_layer_default_blend_settings(self):
        """A layer created without blend settings gets defaults."""
        layer = MaterialLayer("simple")
        assert layer.blend_settings is not None

    def test_layer_enabled_by_default(self):
        """New layers are enabled by default."""
        layer = MaterialLayer("default")
        assert layer.enabled

    def test_layer_can_be_disabled(self):
        """A layer can be disabled."""
        layer = MaterialLayer("toggle")
        layer.enabled = False
        assert not layer.enabled

    def test_layer_can_be_re_enabled(self):
        """A disabled layer can be re-enabled."""
        layer = MaterialLayer("toggle")
        layer.enabled = False
        layer.enabled = True
        assert layer.enabled

    def test_layer_set_parameter(self):
        """A layer can store arbitrary parameters."""
        layer = MaterialLayer("param_test")
        layer.set_parameter("intensity", 0.8)
        assert layer.get_parameter("intensity") == 0.8

    def test_layer_get_parameter_default(self):
        """get_parameter returns the default for unset parameters."""
        layer = MaterialLayer("default_test")
        assert layer.get_parameter("nonexistent") is None
        assert layer.get_parameter("missing", 42) == 42

    def test_layer_mask_texture(self):
        """A layer can have a mask texture assigned."""
        layer = MaterialLayer("masked")
        layer.set_mask_texture("textures/detail_mask.png")
        layer.set_mask_texture(None)  # Should also work


class TestLayerBlendSettings:
    """LayerBlendSettings configuration."""

    def test_create_blend_settings(self):
        """LayerBlendSettings can be created with defaults."""
        b = LayerBlendSettings()
        assert b.blend_weight == 1.0
        assert b.blend_mode == "lerp"
        assert b.mask_channel is None

    def test_create_blend_settings_custom(self):
        """LayerBlendSettings accepts custom values."""
        b = LayerBlendSettings(blend_weight=0.3, blend_mode="add", mask_channel="r")
        assert b.blend_weight == 0.3
        assert b.blend_mode == "add"
        assert b.mask_channel == "r"

    def test_blend_settings_fields_are_mutable(self):
        """LayerBlendSettings fields can be mutated after creation."""
        b = LayerBlendSettings()
        b.blend_weight = 0.75
        b.blend_mode = "multiply"
        assert b.blend_weight == 0.75
        assert b.blend_mode == "multiply"


class TestInstanceLayerStack:
    """Layer push/pop on MaterialInstance."""

    def test_push_layer(self):
        """A layer can be pushed onto an instance's layer stack."""
        t = MaterialTemplate("base")
        inst = t.create_instance("inst")
        layer = MaterialLayer("detail")
        inst.push_layer(layer)
        layers = inst.get_layers()
        assert len(layers) == 1
        assert layers[0] is layer

    def test_pop_layer(self):
        """A layer can be popped from the stack."""
        t = MaterialTemplate("base")
        inst = t.create_instance("inst")
        layer = MaterialLayer("detail")
        inst.push_layer(layer)
        popped = inst.pop_layer()
        assert popped is layer
        assert inst.get_layers() == []

    def test_pop_empty_stack_returns_none(self):
        """Popping from an empty stack returns None."""
        t = MaterialTemplate("base")
        inst = t.create_instance("inst")
        result = inst.pop_layer()
        assert result is None

    def test_layers_ordered_by_push(self):
        """Layers are composited in the order they were pushed."""
        t = MaterialTemplate("base")
        inst = t.create_instance("inst")
        l1 = MaterialLayer("first")
        l2 = MaterialLayer("second")
        l3 = MaterialLayer("third")
        inst.push_layer(l1)
        inst.push_layer(l2)
        inst.push_layer(l3)
        layers = inst.get_layers()
        assert layers[0] is l1
        assert layers[1] is l2
        assert layers[2] is l3


# =============================================================================
# T-MAT-1.5: MaterialSystem Registry
# =============================================================================

class TestMaterialSystemRegistry:
    """MaterialSystem correctly manages templates, instances, and functions."""

    def test_create_system(self):
        """A MaterialSystem can be instantiated."""
        system = MaterialSystem()
        assert system is not None

    def test_register_and_get_template(self):
        """A template can be registered and retrieved by ID."""
        system = MaterialSystem()
        t = MaterialTemplate("test")
        system.register_template(t)
        retrieved = system.get_template(t.template_id)
        assert retrieved is t

    def test_get_template_by_name(self):
        """A template can be retrieved by name."""
        system = MaterialSystem()
        t = MaterialTemplate("test")
        system.register_template(t)
        retrieved = system.get_template_by_name("test")
        assert retrieved is t

    def test_get_template_by_name_no_match(self):
        """get_template_by_name returns None when name not found."""
        system = MaterialSystem()
        t = MaterialTemplate("exists")
        system.register_template(t)
        assert system.get_template_by_name("nonexistent") is None

    def test_unregister_template(self):
        """A registered template can be unregistered."""
        system = MaterialSystem()
        t = MaterialTemplate("test")
        system.register_template(t)
        system.unregister_template(t.template_id)
        assert system.get_template(t.template_id) is None

    def test_register_and_get_instance(self):
        """An instance can be registered and retrieved by ID."""
        system = MaterialSystem()
        t = MaterialTemplate("base")
        system.register_template(t)
        inst = t.create_instance("inst")
        system.register_instance(inst)
        retrieved = system.get_instance(inst.instance_id)
        assert retrieved is inst

    def test_unregister_instance(self):
        """A registered instance can be unregistered."""
        system = MaterialSystem()
        t = MaterialTemplate("base")
        system.register_template(t)
        inst = t.create_instance("inst")
        system.register_instance(inst)
        system.unregister_instance(inst.instance_id)
        assert system.get_instance(inst.instance_id) is None

    def test_register_and_get_function(self):
        """A function can be registered and retrieved by name."""
        system = MaterialSystem()
        fn = MaterialFunction("my_fn", "// code")
        system.register_function(fn)
        retrieved = system.get_function("my_fn")
        assert retrieved is fn

    def test_get_function_no_match(self):
        """get_function returns None when name not found."""
        system = MaterialSystem()
        assert system.get_function("nonexistent") is None

    def test_get_all_templates(self):
        """get_all_templates returns all registered templates."""
        system = MaterialSystem()
        t1 = MaterialTemplate("a")
        t2 = MaterialTemplate("b")
        system.register_template(t1)
        system.register_template(t2)
        all_t = system.get_all_templates()
        assert t1 in all_t
        assert t2 in all_t

    def test_get_all_instances(self):
        """get_all_instances returns all registered instances."""
        system = MaterialSystem()
        t = MaterialTemplate("base")
        system.register_template(t)
        i1 = t.create_instance("a")
        i2 = t.create_instance("b")
        system.register_instance(i1)
        system.register_instance(i2)
        all_i = system.get_all_instances()
        assert i1 in all_i
        assert i2 in all_i


class TestSystemConvenienceMethods:
    """MaterialSystem convenience factory methods."""

    def test_create_template(self):
        """create_template creates and registers a template."""
        system = MaterialSystem()
        t = system.create_template("convenient")
        assert t.name == "convenient"
        assert system.get_template(t.template_id) is t

    def test_create_template_with_kwargs(self):
        """create_template forwards keyword arguments to MaterialTemplate."""
        system = MaterialSystem()
        t = system.create_template("custom", domain=MaterialDomain.UI, blend_mode=BlendMode.TRANSLUCENT)
        assert t.domain == MaterialDomain.UI
        assert t.blend_mode == BlendMode.TRANSLUCENT

    def test_create_instance(self):
        """create_instance creates and registers an instance."""
        system = MaterialSystem()
        t = system.create_template("base")
        inst = system.create_instance(t, "my_inst")
        assert inst.name == "my_inst"
        assert system.get_instance(inst.instance_id) is inst

    def test_create_instance_by_template_id(self):
        """create_instance accepts a template ID string."""
        system = MaterialSystem()
        t = system.create_template("base")
        inst = system.create_instance(t.template_id, "by_id")
        assert inst.name == "by_id"
        assert inst.template is t


class TestSystemDirtyTracking:
    """MaterialSystem dirty instance tracking."""

    def test_mark_instance_dirty(self):
        """An instance can be marked dirty and appears in the dirty list."""
        system = MaterialSystem()
        t = MaterialTemplate("base")
        system.register_template(t)
        inst = t.create_instance("inst")
        system.register_instance(inst)
        system.mark_instance_dirty(inst.instance_id)
        assert inst in system.get_dirty_instances()

    def test_mark_dirty_unknown_id_does_not_raise(self):
        """Marking an unknown instance as dirty does not raise."""
        system = MaterialSystem()
        system.mark_instance_dirty("nonexistent")

    def test_clear_dirty_flags(self):
        """Clearing dirty flags empties the dirty set."""
        system = MaterialSystem()
        t = MaterialTemplate("base")
        system.register_template(t)
        inst = t.create_instance("inst")
        system.register_instance(inst)
        system.mark_instance_dirty(inst.instance_id)
        system.clear_dirty_flags()
        assert system.get_dirty_instances() == []

    def test_mark_dirty_then_clear(self):
        """Marking an instance dirty and then clearing flags works correctly."""
        system = MaterialSystem()
        t = MaterialTemplate("base")
        system.register_template(t)
        inst = t.create_instance("inst")
        system.register_instance(inst)
        system.mark_instance_dirty(inst.instance_id)
        dirty_before = system.get_dirty_instances()
        assert inst in dirty_before
        system.clear_dirty_flags()
        dirty_after = system.get_dirty_instances()
        assert inst not in dirty_after


class TestSystemHotReload:
    """MaterialSystem hot-reload callback mechanism."""

    def test_on_template_changed(self):
        """A callback can be registered for template change notifications."""
        system = MaterialSystem()
        notifications = []
        system.on_template_changed(lambda t: notifications.append(t.name))
        t = system.create_template("hot")
        system.notify_template_changed(t)
        assert notifications == ["hot"]

    def test_multiple_callbacks(self):
        """Multiple callbacks can be registered."""
        system = MaterialSystem()
        log1 = []
        log2 = []
        system.on_template_changed(lambda t: log1.append(t.name))
        system.on_template_changed(lambda t: log2.append(t.name))
        t = system.create_template("multi")
        system.notify_template_changed(t)
        assert log1 == ["multi"]
        assert log2 == ["multi"]

    def test_enable_hot_reload(self):
        """enable_hot_reload() does not raise."""
        system = MaterialSystem()
        system.enable_hot_reload()

    def test_disable_hot_reload(self):
        """disable_hot_reload() does not raise."""
        system = MaterialSystem()
        system.disable_hot_reload()

    def test_hot_reload_toggle(self):
        """Hot reload can be toggled on and off without error."""
        system = MaterialSystem()
        system.enable_hot_reload()
        system.disable_hot_reload()
        system.enable_hot_reload()


# =============================================================================
# T-MAT-1.6: DirtyFlags Integration
# =============================================================================

class TestDirtyFlags:
    """Dirty flags system tracks changes across the material hierarchy."""

    def test_dirty_flags_type(self):
        """DirtyFlags is a class that can be instantiated."""
        df = DirtyFlags()
        assert df is not None

    def test_dirty_flags_initial_clean(self):
        """A new DirtyFlags instance is initially clean."""
        df = DirtyFlags()
        assert not df.any_dirty() or not bool(df.any_dirty())

    def test_mark_all_sets_flags(self):
        """mark_all() sets all dirty flags."""
        df = DirtyFlags()
        df.mark_all()
        assert bool(df.parameters) is True
        assert bool(df.textures) is True
        assert bool(df.shader) is True

    def test_clear_all_clears_flags(self):
        """clear_all() clears all dirty flags."""
        df = DirtyFlags()
        df.mark_all()
        df.clear_all()
        assert bool(df.parameters) is False
        assert bool(df.textures) is False
        assert bool(df.shader) is False

    def test_parameters_flag_independent(self):
        """PARAMETERS flag can be set independently."""
        df = DirtyFlags()
        df.parameters = True
        assert bool(df.parameters) is True
        assert bool(df.textures) is False
        assert bool(df.shader) is False

    def test_textures_flag_independent(self):
        """TEXTURES flag can be set independently."""
        df = DirtyFlags()
        df.textures = True
        assert bool(df.textures) is True
        assert bool(df.parameters) is False
        assert bool(df.shader) is False

    def test_shader_flag_independent(self):
        """SHADER flag can be set independently."""
        df = DirtyFlags()
        df.shader = True
        assert bool(df.shader) is True
        assert bool(df.parameters) is False
        assert bool(df.textures) is False


# =============================================================================
# Cross-cutting: Instance dirty tracking through parameter changes
# =============================================================================

class TestInstanceDirtyViaParameters:
    """Dirty flag propagation on parameter change (T-MAT-1.2 / T-MAT-1.6)."""

    def test_set_parameter_triggers_dirty(self):
        """Setting a parameter value affects the dirty flag after clearing."""
        t = MaterialTemplate("base")
        t.add_parameter(MaterialParameter("a", ParameterType.FLOAT, 0.0))
        inst = t.create_instance("inst")
        # Clear initial dirty state
        inst.dirty.clear_all()
        assert bool(inst.dirty.parameters) is False
        inst.set_parameter("a", 1.0)
        # After parameter set, the dirty flags should indicate a change
        any_flag_set = (
            bool(inst.dirty.parameters)
            or bool(inst.dirty.textures)
            or bool(inst.dirty.shader)
        )
        assert any_flag_set


# =============================================================================
# Enums contract: All expected enum members exist
# =============================================================================

class TestEnumContracts:
    """Enum types have the expected members (contract-level check)."""

    def test_material_domain_members(self):
        """MaterialDomain has the five expected domains."""
        assert hasattr(MaterialDomain, "SURFACE")
        assert hasattr(MaterialDomain, "DEFERRED_DECAL")
        assert hasattr(MaterialDomain, "POST_PROCESS")
        assert hasattr(MaterialDomain, "UI")
        assert hasattr(MaterialDomain, "VOLUME")

    def test_blend_mode_members(self):
        """BlendMode has the five expected modes."""
        assert hasattr(BlendMode, "OPAQUE")
        assert hasattr(BlendMode, "MASKED")
        assert hasattr(BlendMode, "TRANSLUCENT")
        assert hasattr(BlendMode, "ADDITIVE")
        assert hasattr(BlendMode, "MODULATE")

    def test_shading_model_members(self):
        """ShadingModel has the eight expected models."""
        assert hasattr(ShadingModel, "DEFAULT_LIT")
        assert hasattr(ShadingModel, "UNLIT")
        assert hasattr(ShadingModel, "SUBSURFACE")
        assert hasattr(ShadingModel, "CLEAR_COAT")
        assert hasattr(ShadingModel, "CLOTH")
        assert hasattr(ShadingModel, "FOLIAGE")
        assert hasattr(ShadingModel, "HAIR")
        assert hasattr(ShadingModel, "EYE")

    def test_parameter_type_members(self):
        """ParameterType has the nine expected types."""
        assert hasattr(ParameterType, "FLOAT")
        assert hasattr(ParameterType, "INT")
        assert hasattr(ParameterType, "BOOL")
        assert hasattr(ParameterType, "VEC2")
        assert hasattr(ParameterType, "VEC3")
        assert hasattr(ParameterType, "VEC4")
        assert hasattr(ParameterType, "TEXTURE_2D")
        assert hasattr(ParameterType, "TEXTURE_CUBE")
        assert hasattr(ParameterType, "SAMPLER")
