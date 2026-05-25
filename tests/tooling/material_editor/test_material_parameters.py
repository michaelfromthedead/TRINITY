"""Tests for material parameters."""
import pytest
from engine.tooling.material_editor.material_parameters import (
    ParameterType, ParameterSemantics, ParameterRange, TextureSettings,
    MaterialParameter, ScalarParameter, VectorParameter, ColorParameter,
    TextureParameter, BooleanParameter, IntegerParameter, ParameterCollection
)


class TestParameterRange:
    """Tests for ParameterRange."""

    def test_clamp_within_range(self):
        """Test clamping value within range."""
        range = ParameterRange(min_value=0.0, max_value=1.0)
        assert range.clamp(0.5) == 0.5

    def test_clamp_below_min(self):
        """Test clamping value below minimum."""
        range = ParameterRange(min_value=0.0, max_value=1.0)
        assert range.clamp(-0.5) == 0.0

    def test_clamp_above_max(self):
        """Test clamping value above maximum."""
        range = ParameterRange(min_value=0.0, max_value=1.0)
        assert range.clamp(1.5) == 1.0

    def test_clamp_no_limits(self):
        """Test clamping with no limits."""
        range = ParameterRange()
        assert range.clamp(100.0) == 100.0
        assert range.clamp(-100.0) == -100.0

    def test_is_valid_within_range(self):
        """Test validity check within range."""
        range = ParameterRange(min_value=0.0, max_value=1.0)
        assert range.is_valid(0.5) is True

    def test_is_valid_outside_range(self):
        """Test validity check outside range."""
        range = ParameterRange(min_value=0.0, max_value=1.0)
        assert range.is_valid(-0.5) is False
        assert range.is_valid(1.5) is False


class TestScalarParameter:
    """Tests for ScalarParameter."""

    def test_create_with_default(self):
        """Test creating scalar parameter with default."""
        param = ScalarParameter("roughness", default=0.5)
        assert param.name == "roughness"
        assert param.value == 0.5
        assert param.default_value == 0.5
        assert param.parameter_type == ParameterType.SCALAR

    def test_set_value(self):
        """Test setting parameter value."""
        param = ScalarParameter("roughness", default=0.5)
        param.value = 0.8
        assert param.value == 0.8

    def test_reset(self):
        """Test resetting to default."""
        param = ScalarParameter("roughness", default=0.5)
        param.value = 0.8
        param.reset()
        assert param.value == 0.5

    def test_clone(self):
        """Test cloning parameter."""
        param = ScalarParameter("roughness", default=0.5, group="PBR")
        param.value = 0.8
        cloned = param.clone()
        assert cloned.name == param.name
        assert cloned.value == param.value
        assert cloned.group == param.group
        assert cloned is not param

    def test_clamping_with_range(self):
        """Test value clamping with range."""
        range = ParameterRange(min_value=0.0, max_value=1.0)
        param = ScalarParameter("roughness", default=0.5, range=range)
        param.value = 1.5
        assert param.value == 1.0

    def test_to_dict(self):
        """Test serialization to dict."""
        param = ScalarParameter("roughness", default=0.5, group="PBR")
        data = param.to_dict()
        assert data["name"] == "roughness"
        assert data["type"] == "SCALAR"
        assert data["default_value"] == 0.5


class TestVectorParameter:
    """Tests for VectorParameter."""

    def test_create_vector2(self):
        """Test creating float2 parameter."""
        param = VectorParameter("tiling", components=2, default=(1.0, 1.0))
        assert param.parameter_type == ParameterType.VECTOR2
        assert param.value == (1.0, 1.0)

    def test_create_vector3(self):
        """Test creating float3 parameter."""
        param = VectorParameter("offset", components=3, default=(0.0, 0.0, 0.0))
        assert param.parameter_type == ParameterType.VECTOR3
        assert param.value == (0.0, 0.0, 0.0)

    def test_create_vector4(self):
        """Test creating float4 parameter."""
        param = VectorParameter("rect", components=4, default=(0.0, 0.0, 1.0, 1.0))
        assert param.parameter_type == ParameterType.VECTOR4

    def test_invalid_components(self):
        """Test invalid component count."""
        with pytest.raises(ValueError):
            VectorParameter("bad", components=5)

    def test_set_value(self):
        """Test setting vector value."""
        param = VectorParameter("tiling", components=2, default=(1.0, 1.0))
        param.value = (2.0, 2.0)
        assert param.value == (2.0, 2.0)

    def test_wrong_component_count(self):
        """Test setting value with wrong component count."""
        param = VectorParameter("tiling", components=2, default=(1.0, 1.0))
        with pytest.raises(ValueError):
            param.value = (1.0, 2.0, 3.0)

    def test_clone(self):
        """Test cloning vector parameter."""
        param = VectorParameter("tiling", components=2, default=(1.0, 1.0))
        param.value = (2.0, 3.0)
        cloned = param.clone()
        assert cloned.value == (2.0, 3.0)
        assert cloned is not param


class TestColorParameter:
    """Tests for ColorParameter."""

    def test_create_rgba(self):
        """Test creating RGBA color parameter."""
        param = ColorParameter("baseColor", default=(1.0, 0.5, 0.0, 1.0))
        assert param.parameter_type == ParameterType.COLOR
        assert param.value == (1.0, 0.5, 0.0, 1.0)
        assert param.has_alpha is True

    def test_create_rgb(self):
        """Test creating RGB color parameter."""
        param = ColorParameter("baseColor", has_alpha=False, default=(1.0, 0.5, 0.0))
        assert len(param.value) == 3

    def test_clamp_values(self):
        """Test color values are clamped to [0,1]."""
        param = ColorParameter("color")
        param.value = (1.5, -0.5, 0.5, 2.0)
        assert param.value == (1.0, 0.0, 0.5, 1.0)

    def test_hdr_values(self):
        """Test HDR colors allow values > 1."""
        param = ColorParameter("emissive", hdr=True)
        param.value = (2.0, 3.0, 4.0, 1.0)
        assert param.value == (2.0, 3.0, 4.0, 1.0)

    def test_reset(self):
        """Test resetting color to default."""
        param = ColorParameter("color", default=(1.0, 0.0, 0.0, 1.0))
        param.value = (0.0, 1.0, 0.0, 1.0)
        param.reset()
        assert param.value == (1.0, 0.0, 0.0, 1.0)


class TestTextureParameter:
    """Tests for TextureParameter."""

    def test_create(self):
        """Test creating texture parameter."""
        param = TextureParameter("albedoMap", default_path="textures/white.png")
        assert param.parameter_type == ParameterType.TEXTURE
        assert param.value == "textures/white.png"

    def test_set_path(self):
        """Test setting texture path."""
        param = TextureParameter("albedoMap")
        param.value = "textures/brick.png"
        assert param.value == "textures/brick.png"

    def test_texture_handle(self):
        """Test texture handle property."""
        param = TextureParameter("albedoMap")
        assert param.texture_handle is None
        param.texture_handle = 42
        assert param.texture_handle == 42

    def test_handle_invalidated_on_path_change(self):
        """Test handle is invalidated when path changes."""
        param = TextureParameter("albedoMap")
        param.texture_handle = 42
        param.value = "new/path.png"
        assert param.texture_handle is None

    def test_settings(self):
        """Test texture settings."""
        settings = TextureSettings(filter_mode="nearest", srgb=False)
        param = TextureParameter("normalMap", settings=settings)
        assert param.settings.filter_mode == "nearest"
        assert param.settings.srgb is False


class TestBooleanParameter:
    """Tests for BooleanParameter."""

    def test_create(self):
        """Test creating boolean parameter."""
        param = BooleanParameter("useNormalMap", default=True)
        assert param.parameter_type == ParameterType.BOOLEAN
        assert param.value is True

    def test_set_value(self):
        """Test setting boolean value."""
        param = BooleanParameter("flag", default=False)
        param.value = True
        assert param.value is True

    def test_coerce_to_bool(self):
        """Test values are coerced to bool."""
        param = BooleanParameter("flag")
        param.value = 1
        assert param.value is True
        param.value = 0
        assert param.value is False


class TestIntegerParameter:
    """Tests for IntegerParameter."""

    def test_create(self):
        """Test creating integer parameter."""
        param = IntegerParameter("iterations", default=10)
        assert param.parameter_type == ParameterType.INTEGER
        assert param.value == 10

    def test_set_value(self):
        """Test setting integer value."""
        param = IntegerParameter("count", default=5)
        param.value = 15
        assert param.value == 15

    def test_min_max_clamping(self):
        """Test value clamping with min/max."""
        param = IntegerParameter("count", default=5, min_value=0, max_value=10)
        param.value = -5
        assert param.value == 0
        param.value = 15
        assert param.value == 10

    def test_float_converted_to_int(self):
        """Test float values are converted to int."""
        param = IntegerParameter("count", default=5)
        param.value = 7.8
        assert param.value == 7
        assert isinstance(param.value, int)


class TestParameterCollection:
    """Tests for ParameterCollection."""

    def test_add_and_get(self):
        """Test adding and getting parameters."""
        collection = ParameterCollection()
        param = ScalarParameter("roughness", default=0.5)
        collection.add(param)
        assert collection.get("roughness") == param

    def test_remove(self):
        """Test removing parameters."""
        collection = ParameterCollection()
        param = ScalarParameter("roughness", default=0.5)
        collection.add(param)
        removed = collection.remove("roughness")
        assert removed == param
        assert collection.get("roughness") is None

    def test_contains(self):
        """Test containment check."""
        collection = ParameterCollection()
        param = ScalarParameter("roughness", default=0.5)
        collection.add(param)
        assert "roughness" in collection
        assert "metallic" not in collection

    def test_len(self):
        """Test length."""
        collection = ParameterCollection()
        collection.add(ScalarParameter("a", default=0.0))
        collection.add(ScalarParameter("b", default=0.0))
        assert len(collection) == 2

    def test_iter(self):
        """Test iteration."""
        collection = ParameterCollection()
        collection.add(ScalarParameter("a", default=0.0))
        collection.add(ScalarParameter("b", default=0.0))
        names = [p.name for p in collection]
        assert "a" in names
        assert "b" in names

    def test_get_by_group(self):
        """Test getting parameters by group."""
        collection = ParameterCollection()
        collection.add(ScalarParameter("roughness", group="PBR"))
        collection.add(ScalarParameter("metallic", group="PBR"))
        collection.add(ScalarParameter("emissive", group="Lighting"))
        pbr_params = collection.get_by_group("PBR")
        assert len(pbr_params) == 2

    def test_get_by_type(self):
        """Test getting parameters by type."""
        collection = ParameterCollection()
        collection.add(ScalarParameter("roughness"))
        collection.add(VectorParameter("tiling", components=2))
        collection.add(TextureParameter("albedoMap"))
        scalars = collection.get_by_type(ParameterType.SCALAR)
        assert len(scalars) == 1

    def test_reset_all(self):
        """Test resetting all parameters."""
        collection = ParameterCollection()
        p1 = ScalarParameter("a", default=0.5)
        p2 = ScalarParameter("b", default=0.3)
        p1.value = 0.8
        p2.value = 0.9
        collection.add(p1)
        collection.add(p2)
        collection.reset_all()
        assert p1.value == 0.5
        assert p2.value == 0.3

    def test_clone(self):
        """Test cloning collection."""
        collection = ParameterCollection()
        collection.add(ScalarParameter("a", default=0.5))
        cloned = collection.clone()
        assert len(cloned) == 1
        assert cloned.get("a") is not collection.get("a")

    def test_serialization(self):
        """Test to_dict and from_dict."""
        collection = ParameterCollection()
        collection.add(ScalarParameter("roughness", default=0.5, group="PBR"))
        collection.add(ColorParameter("color", default=(1.0, 0.0, 0.0, 1.0)))

        data = collection.to_dict()
        restored = ParameterCollection.from_dict(data)

        assert len(restored) == 2
        assert restored.get("roughness").default_value == 0.5
        assert restored.get("color").parameter_type == ParameterType.COLOR
