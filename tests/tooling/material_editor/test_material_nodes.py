"""Tests for material nodes."""
import pytest
import math
from engine.tooling.material_editor.material_nodes import (
    NodeCategory, DataType, NodePin, MaterialNode,
    ConstantNode, Constant2Node, Constant3Node, Constant4Node, ParameterNode,
    AddNode, SubtractNode, MultiplyNode, DivideNode, LerpNode, ClampNode,
    SaturateNode, PowerNode, DotNode, CrossNode, NormalizeNode,
    AbsNode, FloorNode, CeilNode, FracNode, SinNode, CosNode, OneMinusNode,
    TextureSampleNode, UVNode, TilingOffsetNode, NormalMapNode, ParallaxNode,
    TimeNode, WorldPositionNode, ViewDirectionNode, ScreenPositionNode,
    VertexColorNode, SplitNode, CombineNode,
    FresnelNode, GGXNode, LambertNode, BRDFNode,
    PBROutputNode, UnlitOutputNode, CustomCodeNode,
    NODE_REGISTRY
)


class TestNodePin:
    """Tests for NodePin."""

    def test_create_input_pin(self):
        """Test creating input pin."""
        pin = NodePin("Value", DataType.FLOAT, is_output=False, default_value=0.5)
        assert pin.name == "Value"
        assert pin.data_type == DataType.FLOAT
        assert pin.is_output is False
        assert pin.default_value == 0.5

    def test_create_output_pin(self):
        """Test creating output pin."""
        pin = NodePin("Result", DataType.FLOAT3, is_output=True)
        assert pin.is_output is True

    def test_compatibility_same_type(self):
        """Test pins of same type are compatible."""
        output = NodePin("Out", DataType.FLOAT, is_output=True)
        input = NodePin("In", DataType.FLOAT, is_output=False)
        assert output.is_compatible(input) is True

    def test_compatibility_different_direction(self):
        """Test pins must have different directions."""
        out1 = NodePin("Out1", DataType.FLOAT, is_output=True)
        out2 = NodePin("Out2", DataType.FLOAT, is_output=True)
        assert out1.is_compatible(out2) is False

    def test_compatibility_float_types(self):
        """Test float types are compatible."""
        output = NodePin("Out", DataType.FLOAT, is_output=True)
        input = NodePin("In", DataType.FLOAT3, is_output=False)
        assert output.is_compatible(input) is True

    def test_compatibility_any_type(self):
        """Test ANY type is compatible with everything."""
        any_pin = NodePin("Any", DataType.ANY, is_output=True)
        float_pin = NodePin("Float", DataType.FLOAT, is_output=False)
        tex_pin = NodePin("Tex", DataType.TEXTURE2D, is_output=False)
        assert any_pin.is_compatible(float_pin) is True
        assert any_pin.is_compatible(tex_pin) is True


class TestConstantNodes:
    """Tests for constant input nodes."""

    def test_constant_node(self):
        """Test scalar constant node."""
        node = ConstantNode(value=0.5)
        assert node.category == NodeCategory.INPUT
        assert node.value == 0.5
        result = node.evaluate({})
        assert result["Value"] == 0.5

    def test_constant2_node(self):
        """Test float2 constant node."""
        node = Constant2Node(value=(1.0, 2.0))
        assert node.value == (1.0, 2.0)
        result = node.evaluate({})
        assert result["Value"] == (1.0, 2.0)

    def test_constant3_node(self):
        """Test float3 constant node."""
        node = Constant3Node(value=(1.0, 2.0, 3.0))
        result = node.evaluate({})
        assert result["Value"] == (1.0, 2.0, 3.0)

    def test_constant4_node(self):
        """Test float4 constant node."""
        node = Constant4Node(value=(1.0, 2.0, 3.0, 4.0))
        result = node.evaluate({})
        assert result["Value"] == (1.0, 2.0, 3.0, 4.0)

    def test_constant_code_generation(self):
        """Test constant node code generation."""
        node = ConstantNode(value=0.5)
        code = node.generate_code({}, {"Value": "myVar"})
        assert "myVar" in code
        assert "0.5" in code

    def test_parameter_node(self):
        """Test parameter node."""
        node = ParameterNode(param_name="Roughness", param_type=DataType.FLOAT)
        assert node.param_name == "Roughness"
        assert node.category == NodeCategory.INPUT


class TestMathNodes:
    """Tests for math operation nodes."""

    def test_add_scalars(self):
        """Test adding scalar values."""
        node = AddNode()
        result = node.evaluate({"A": 2.0, "B": 3.0})
        assert result["Result"] == 5.0

    def test_add_vectors(self):
        """Test adding vector values."""
        node = AddNode()
        result = node.evaluate({"A": (1.0, 2.0), "B": (3.0, 4.0)})
        assert result["Result"] == (4.0, 6.0)

    def test_subtract(self):
        """Test subtraction."""
        node = SubtractNode()
        result = node.evaluate({"A": 5.0, "B": 3.0})
        assert result["Result"] == 2.0

    def test_multiply_scalars(self):
        """Test multiplying scalars."""
        node = MultiplyNode()
        result = node.evaluate({"A": 2.0, "B": 3.0})
        assert result["Result"] == 6.0

    def test_multiply_vector_scalar(self):
        """Test multiplying vector by scalar."""
        node = MultiplyNode()
        result = node.evaluate({"A": (1.0, 2.0, 3.0), "B": 2.0})
        assert result["Result"] == (2.0, 4.0, 6.0)

    def test_divide(self):
        """Test division."""
        node = DivideNode()
        result = node.evaluate({"A": 6.0, "B": 2.0})
        assert result["Result"] == 3.0

    def test_divide_by_zero(self):
        """Test division by zero returns zero."""
        node = DivideNode()
        result = node.evaluate({"A": 6.0, "B": 0.0})
        assert result["Result"] == 0.0

    def test_lerp(self):
        """Test linear interpolation."""
        node = LerpNode()
        result = node.evaluate({"A": 0.0, "B": 10.0, "Alpha": 0.5})
        assert result["Result"] == 5.0

    def test_clamp(self):
        """Test clamping."""
        node = ClampNode()
        result = node.evaluate({"Value": 1.5, "Min": 0.0, "Max": 1.0})
        assert result["Result"] == 1.0
        result = node.evaluate({"Value": -0.5, "Min": 0.0, "Max": 1.0})
        assert result["Result"] == 0.0

    def test_saturate(self):
        """Test saturate (clamp 0-1)."""
        node = SaturateNode()
        result = node.evaluate({"Value": 1.5})
        assert result["Result"] == 1.0
        result = node.evaluate({"Value": 0.5})
        assert result["Result"] == 0.5

    def test_power(self):
        """Test power function."""
        node = PowerNode()
        result = node.evaluate({"Base": 2.0, "Exponent": 3.0})
        assert result["Result"] == 8.0

    def test_dot_product(self):
        """Test dot product."""
        node = DotNode()
        result = node.evaluate({
            "A": (1.0, 0.0, 0.0),
            "B": (1.0, 0.0, 0.0)
        })
        assert result["Result"] == 1.0

    def test_cross_product(self):
        """Test cross product."""
        node = CrossNode()
        result = node.evaluate({
            "A": (1.0, 0.0, 0.0),
            "B": (0.0, 1.0, 0.0)
        })
        assert result["Result"] == (0.0, 0.0, 1.0)

    def test_normalize(self):
        """Test vector normalization."""
        node = NormalizeNode()
        result = node.evaluate({"Vector": (3.0, 0.0, 0.0)})
        assert abs(result["Result"][0] - 1.0) < 0.0001
        assert result["Result"][1] == 0.0

    def test_abs(self):
        """Test absolute value."""
        node = AbsNode()
        result = node.evaluate({"Value": -5.0})
        assert result["Result"] == 5.0

    def test_floor(self):
        """Test floor function."""
        node = FloorNode()
        result = node.evaluate({"Value": 3.7})
        assert result["Result"] == 3.0

    def test_ceil(self):
        """Test ceiling function."""
        node = CeilNode()
        result = node.evaluate({"Value": 3.2})
        assert result["Result"] == 4.0

    def test_frac(self):
        """Test fractional part."""
        node = FracNode()
        result = node.evaluate({"Value": 3.7})
        assert abs(result["Result"] - 0.7) < 0.0001

    def test_sin(self):
        """Test sine function."""
        node = SinNode()
        result = node.evaluate({"Value": 0.0})
        assert result["Result"] == 0.0
        result = node.evaluate({"Value": math.pi / 2})
        assert abs(result["Result"] - 1.0) < 0.0001

    def test_cos(self):
        """Test cosine function."""
        node = CosNode()
        result = node.evaluate({"Value": 0.0})
        assert result["Result"] == 1.0

    def test_one_minus(self):
        """Test 1 - x."""
        node = OneMinusNode()
        result = node.evaluate({"Value": 0.3})
        assert abs(result["Result"] - 0.7) < 0.0001


class TestTextureNodes:
    """Tests for texture nodes."""

    def test_texture_sample_node(self):
        """Test texture sample node."""
        node = TextureSampleNode(texture_path="textures/albedo.png")
        assert node.category == NodeCategory.TEXTURE
        assert node.texture_path == "textures/albedo.png"
        result = node.evaluate({"UV": (0.5, 0.5)})
        assert "RGBA" in result
        assert "R" in result

    def test_uv_node(self):
        """Test UV coordinate node."""
        node = UVNode(uv_channel=0)
        assert node.uv_channel == 0
        result = node.evaluate({})
        assert "UV" in result
        assert "U" in result
        assert "V" in result

    def test_tiling_offset_node(self):
        """Test tiling and offset node."""
        node = TilingOffsetNode()
        result = node.evaluate({
            "UV": (0.5, 0.5),
            "Tiling": (2.0, 2.0),
            "Offset": (0.1, 0.1)
        })
        assert result["Result"] == (1.1, 1.1)

    def test_normal_map_node(self):
        """Test normal map unpacking."""
        node = NormalMapNode()
        result = node.evaluate({
            "Normal": (0.5, 0.5, 1.0),
            "Strength": 1.0
        })
        # Should unpack from [0,1] to [-1,1]
        assert abs(result["Result"][0]) < 0.0001
        assert abs(result["Result"][1]) < 0.0001

    def test_parallax_node(self):
        """Test parallax offset."""
        node = ParallaxNode()
        result = node.evaluate({
            "UV": (0.5, 0.5),
            "Height": 0.5,
            "Scale": 0.1,
            "ViewDir": (0.0, 0.0, 1.0)
        })
        assert "Result" in result


class TestUtilityNodes:
    """Tests for utility nodes."""

    def test_time_node(self):
        """Test time node outputs."""
        node = TimeNode()
        assert node.category == NodeCategory.UTILITY
        result = node.evaluate({})
        assert "Time" in result
        assert "SinTime" in result
        assert "DeltaTime" in result

    def test_world_position_node(self):
        """Test world position node."""
        node = WorldPositionNode()
        result = node.evaluate({})
        assert "Position" in result
        assert "X" in result
        assert "Y" in result
        assert "Z" in result

    def test_view_direction_node(self):
        """Test view direction node."""
        node = ViewDirectionNode()
        result = node.evaluate({})
        assert "ViewDir" in result

    def test_screen_position_node(self):
        """Test screen position node."""
        node = ScreenPositionNode()
        result = node.evaluate({})
        assert "Position" in result
        assert "UV" in result

    def test_vertex_color_node(self):
        """Test vertex color node."""
        node = VertexColorNode()
        result = node.evaluate({})
        assert "Color" in result
        assert "RGB" in result
        assert "R" in result

    def test_split_node(self):
        """Test vector split node."""
        node = SplitNode()
        result = node.evaluate({"Vector": (1.0, 2.0, 3.0, 4.0)})
        assert result["X"] == 1.0
        assert result["Y"] == 2.0
        assert result["Z"] == 3.0
        assert result["W"] == 4.0

    def test_combine_node(self):
        """Test vector combine node."""
        node = CombineNode()
        result = node.evaluate({"X": 1.0, "Y": 2.0, "Z": 3.0, "W": 4.0})
        assert result["XYZW"] == (1.0, 2.0, 3.0, 4.0)
        assert result["XYZ"] == (1.0, 2.0, 3.0)
        assert result["XY"] == (1.0, 2.0)


class TestPBRNodes:
    """Tests for PBR nodes."""

    def test_fresnel_node(self):
        """Test Fresnel effect calculation."""
        node = FresnelNode()
        assert node.category == NodeCategory.PBR
        result = node.evaluate({
            "Normal": (0.0, 0.0, 1.0),
            "ViewDir": (0.0, 0.0, 1.0),
            "Power": 5.0
        })
        # When normal and view are aligned, fresnel should be 0
        assert result["Result"] == 0.0

    def test_ggx_node(self):
        """Test GGX normal distribution."""
        node = GGXNode()
        result = node.evaluate({
            "NdotH": 1.0,
            "Roughness": 0.5
        })
        assert "Result" in result

    def test_lambert_node(self):
        """Test Lambert diffuse."""
        node = LambertNode()
        result = node.evaluate({
            "Normal": (0.0, 1.0, 0.0),
            "LightDir": (0.0, 1.0, 0.0)
        })
        assert result["Result"] == 1.0  # Aligned normal and light

    def test_brdf_node(self):
        """Test complete BRDF calculation."""
        node = BRDFNode()
        result = node.evaluate({
            "Albedo": (0.5, 0.5, 0.5),
            "Normal": (0.0, 0.0, 1.0),
            "Metallic": 0.0,
            "Roughness": 0.5,
            "AO": 1.0
        })
        assert "Color" in result


class TestOutputNodes:
    """Tests for output nodes."""

    def test_pbr_output_node(self):
        """Test PBR output node."""
        node = PBROutputNode()
        assert node.category == NodeCategory.OUTPUT
        # Check all PBR inputs exist
        assert "Albedo" in node.inputs
        assert "Normal" in node.inputs
        assert "Metallic" in node.inputs
        assert "Roughness" in node.inputs
        assert "AO" in node.inputs
        assert "Emissive" in node.inputs
        assert "Alpha" in node.inputs

    def test_unlit_output_node(self):
        """Test unlit output node."""
        node = UnlitOutputNode()
        assert node.category == NodeCategory.OUTPUT
        assert "Color" in node.inputs
        assert "Alpha" in node.inputs


class TestCustomCodeNode:
    """Tests for custom code node."""

    def test_create_custom_node(self):
        """Test creating custom code node."""
        node = CustomCodeNode(code="float ${Result} = ${A} + ${B};")
        assert node.category == NodeCategory.CUSTOM
        assert node.code == "float ${Result} = ${A} + ${B};"

    def test_add_custom_pins(self):
        """Test adding custom pins."""
        node = CustomCodeNode()
        node.add_custom_input("A", DataType.FLOAT)
        node.add_custom_output("Result", DataType.FLOAT)
        assert "A" in node.inputs
        assert "Result" in node.outputs


class TestNodeProperties:
    """Tests for node common properties."""

    def test_node_id_unique(self):
        """Test each node gets unique ID."""
        node1 = ConstantNode()
        node2 = ConstantNode()
        assert node1.id != node2.id

    def test_node_position(self):
        """Test node position property."""
        node = ConstantNode()
        node.position = (100.0, 200.0)
        assert node.position == (100.0, 200.0)

    def test_node_name(self):
        """Test node name property."""
        node = ConstantNode(name="MyConstant")
        assert node.name == "MyConstant"
        node.name = "Renamed"
        assert node.name == "Renamed"

    def test_node_preview_enabled(self):
        """Test preview enabled property."""
        node = ConstantNode()
        assert node.preview_enabled is True
        node.preview_enabled = False
        assert node.preview_enabled is False

    def test_node_collapsed(self):
        """Test collapsed property."""
        node = ConstantNode()
        assert node.collapsed is False
        node.collapsed = True
        assert node.collapsed is True

    def test_node_comment(self):
        """Test comment property."""
        node = ConstantNode()
        node.comment = "This is a test"
        assert node.comment == "This is a test"

    def test_node_color(self):
        """Test custom color property."""
        node = ConstantNode()
        node.color = (255, 128, 0)
        assert node.color == (255, 128, 0)

    def test_node_to_dict(self):
        """Test node serialization."""
        node = ConstantNode(value=0.5, name="Test")
        node.position = (100, 200)
        data = node.to_dict()
        # Type uses registry key for proper serialization/deserialization
        assert data["type"] == "Constant"
        assert data["name"] == "Test"
        assert data["position"] == [100, 200]


class TestNodeRegistry:
    """Tests for node registry."""

    def test_registry_contains_basic_nodes(self):
        """Test registry contains expected nodes."""
        assert "Constant" in NODE_REGISTRY
        assert "Add" in NODE_REGISTRY
        assert "Multiply" in NODE_REGISTRY
        assert "TextureSample" in NODE_REGISTRY
        assert "PBROutput" in NODE_REGISTRY

    def test_can_instantiate_from_registry(self):
        """Test nodes can be created from registry."""
        node_class = NODE_REGISTRY["Add"]
        node = node_class()
        assert isinstance(node, AddNode)
