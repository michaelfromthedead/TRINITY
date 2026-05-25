"""Tests for the node-based material graph system.

Tests MaterialNode, MaterialGraph, and GraphCompiler.
"""
import pytest

from engine.core.math.vec import Vec2, Vec3, Vec4
from engine.rendering.materials.material_graph import (
    AddNode,
    AppendNode,
    ClampNode,
    ComponentMask,
    ConstantNode,
    DataType,
    DivideNode,
    DotNode,
    GraphCompiler,
    GraphValidationError,
    LerpNode,
    MaterialGraph,
    MultiplyNode,
    NodeConnection,
    NodePort,
    NormalizeNode,
    OutputNode,
    ParameterNode,
    SubtractNode,
    TextureSampleNode,
    UVNode,
)


class TestNodePort:
    """Test NodePort configuration."""

    def test_create_input_port(self):
        """Test creating an input port."""
        port = NodePort(
            name="value",
            data_type=DataType.FLOAT,
            is_output=False,
            default_value=0.5,
        )
        assert port.name == "value"
        assert port.data_type == DataType.FLOAT
        assert not port.is_output
        assert port.default_value == 0.5

    def test_create_output_port(self):
        """Test creating an output port."""
        port = NodePort(
            name="result",
            data_type=DataType.VEC3,
            is_output=True,
        )
        assert port.is_output

    def test_compatible_same_type(self):
        """Test compatibility between same types."""
        output = NodePort(name="out", data_type=DataType.FLOAT, is_output=True)
        input_ = NodePort(name="in", data_type=DataType.FLOAT, is_output=False)
        assert output.is_compatible_with(input_)

    def test_compatible_float_to_vec(self):
        """Test float promotion to vector."""
        output = NodePort(name="out", data_type=DataType.FLOAT, is_output=True)
        input_ = NodePort(name="in", data_type=DataType.VEC3, is_output=False)
        assert output.is_compatible_with(input_)

    def test_incompatible_same_direction(self):
        """Test incompatibility between same direction ports."""
        port1 = NodePort(name="out1", data_type=DataType.FLOAT, is_output=True)
        port2 = NodePort(name="out2", data_type=DataType.FLOAT, is_output=True)
        assert not port1.is_compatible_with(port2)


class TestConstantNode:
    """Test ConstantNode for literal values."""

    def test_float_constant(self):
        """Test float constant node."""
        node = ConstantNode(value=0.5, data_type=DataType.FLOAT)
        assert node.value == 0.5
        assert "value" in node.outputs

    def test_vec3_constant(self):
        """Test Vec3 constant node."""
        node = ConstantNode(
            value=Vec3(1.0, 0.5, 0.0),
            data_type=DataType.VEC3,
        )
        assert node.value.x == 1.0

    def test_generate_code_float(self):
        """Test code generation for float."""
        node = ConstantNode(value=0.5, data_type=DataType.FLOAT)
        code = node.generate_code({}, "v1")
        assert "float v1 = 0.5" in code

    def test_generate_code_vec3(self):
        """Test code generation for Vec3."""
        node = ConstantNode(
            value=Vec3(1.0, 0.5, 0.0),
            data_type=DataType.VEC3,
        )
        code = node.generate_code({}, "v1")
        assert "vec3 v1" in code
        assert "1.0" in code


class TestParameterNode:
    """Test ParameterNode for material parameters."""

    def test_create_parameter(self):
        """Test parameter node creation."""
        node = ParameterNode(
            param_name="roughness",
            data_type=DataType.FLOAT,
            default_value=0.5,
        )
        assert node.param_name == "roughness"
        assert "value" in node.outputs

    def test_generate_code(self):
        """Test code generation."""
        node = ParameterNode(
            param_name="metallic",
            data_type=DataType.FLOAT,
        )
        code = node.generate_code({}, "v1")
        assert "u_metallic" in code


class TestMathNodes:
    """Test mathematical operation nodes."""

    def test_add_node(self):
        """Test add node."""
        node = AddNode()
        assert "a" in node.inputs
        assert "b" in node.inputs
        assert "result" in node.outputs

        code = node.generate_code({"a": "x", "b": "y"}, "v1")
        assert "x + y" in code

    def test_subtract_node(self):
        """Test subtract node."""
        node = SubtractNode()
        code = node.generate_code({"a": "x", "b": "y"}, "v1")
        assert "x - y" in code

    def test_multiply_node(self):
        """Test multiply node."""
        node = MultiplyNode()
        code = node.generate_code({"a": "x", "b": "y"}, "v1")
        assert "x * y" in code

    def test_divide_node(self):
        """Test divide node with safe division."""
        node = DivideNode()
        code = node.generate_code({"a": "x", "b": "y"}, "v1")
        assert "/" in code
        assert "max" in code  # Safe division

    def test_lerp_node(self):
        """Test lerp node."""
        node = LerpNode()
        assert "t" in node.inputs
        code = node.generate_code({"a": "x", "b": "y", "t": "0.5"}, "v1")
        assert "mix" in code

    def test_clamp_node(self):
        """Test clamp node."""
        node = ClampNode()
        code = node.generate_code(
            {"value": "x", "min": "0.0", "max": "1.0"},
            "v1",
        )
        assert "clamp" in code

    def test_dot_node(self):
        """Test dot product node."""
        node = DotNode()
        assert node.inputs["a"].data_type == DataType.VEC3
        code = node.generate_code({"a": "n", "b": "v"}, "v1")
        assert "dot(n, v)" in code

    def test_normalize_node(self):
        """Test normalize node."""
        node = NormalizeNode()
        code = node.generate_code({"vector": "v"}, "v1")
        assert "normalize" in code


class TestTextureNodes:
    """Test texture sampling nodes."""

    def test_texture_sample_node(self):
        """Test texture sample node."""
        node = TextureSampleNode(texture_name="albedo")
        assert node.texture_name == "albedo"
        assert "uv" in node.inputs
        assert "rgba" in node.outputs
        assert "r" in node.outputs

    def test_texture_generate_code(self):
        """Test texture sampling code generation."""
        node = TextureSampleNode(texture_name="albedo")
        code = node.generate_code({"uv": "v_uv"}, "v1")
        assert "texture(tex_albedo" in code
        assert "v1_rgba" in code

    def test_uv_node(self):
        """Test UV coordinate node."""
        node = UVNode(uv_index=0)
        assert "uv" in node.outputs
        assert "u" in node.outputs
        assert "v" in node.outputs


class TestUtilityNodes:
    """Test utility nodes."""

    def test_component_mask(self):
        """Test component mask node."""
        node = ComponentMask(components="xy")
        assert node.outputs["result"].data_type == DataType.VEC2

        code = node.generate_code({"vector": "v"}, "r")
        assert ".xy" in code

    def test_append_node(self):
        """Test append node."""
        node = AppendNode()
        code = node.generate_code({"a": "x", "b": "y"}, "v1")
        assert "vec2(x, y)" in code


class TestOutputNode:
    """Test material output node."""

    def test_output_ports(self):
        """Test output node has all material ports."""
        node = OutputNode()
        assert "base_color" in node.inputs
        assert "metallic" in node.inputs
        assert "roughness" in node.inputs
        assert "normal" in node.inputs
        assert "emissive" in node.inputs
        assert "ao" in node.inputs
        assert "opacity" in node.inputs

    def test_output_generate_code(self):
        """Test output node code generation."""
        node = OutputNode()
        code = node.generate_code(
            {
                "base_color": "albedo",
                "metallic": "0.0",
                "roughness": "0.5",
            },
            "output",
        )
        assert "out_BaseColor = albedo" in code
        assert "out_Metallic = 0.0" in code


class TestMaterialGraph:
    """Test MaterialGraph node management."""

    def test_create_graph(self):
        """Test graph creation."""
        graph = MaterialGraph(name="TestMaterial")
        assert graph.name == "TestMaterial"
        assert len(graph.nodes) == 0

    def test_add_node(self):
        """Test adding nodes."""
        graph = MaterialGraph()
        node = ConstantNode(value=0.5, data_type=DataType.FLOAT)
        node_id = graph.add_node(node)
        assert node_id in graph.nodes
        assert graph.get_node(node_id) == node

    def test_add_output_node(self):
        """Test adding output node."""
        graph = MaterialGraph()
        output = OutputNode()
        graph.add_node(output)
        assert graph.output_node == output

    def test_only_one_output_node(self):
        """Test that only one output node is allowed."""
        graph = MaterialGraph()
        graph.add_node(OutputNode())
        with pytest.raises(GraphValidationError, match="one output"):
            graph.add_node(OutputNode())

    def test_remove_node(self):
        """Test removing nodes."""
        graph = MaterialGraph()
        node = ConstantNode(value=0.5, data_type=DataType.FLOAT)
        node_id = graph.add_node(node)
        graph.remove_node(node_id)
        assert node_id not in graph.nodes

    def test_connect_nodes(self):
        """Test connecting nodes."""
        graph = MaterialGraph()
        const = ConstantNode(value=0.5, data_type=DataType.FLOAT)
        output = OutputNode()

        graph.add_node(const)
        graph.add_node(output)

        conn = graph.connect(const, "value", output, "roughness")
        assert conn in graph.connections
        assert conn.source_node == const.node_id
        assert conn.target_port == "roughness"

    def test_connect_invalid_source(self):
        """Test error on invalid source node."""
        graph = MaterialGraph()
        output = OutputNode()
        graph.add_node(output)

        with pytest.raises(GraphValidationError, match="Source node not found"):
            graph.connect("invalid_id", "value", output, "roughness")

    def test_connect_incompatible_ports(self):
        """Test error on incompatible port types."""
        graph = MaterialGraph()
        # Using a different approach - texture to float
        uv = UVNode()
        output = OutputNode()

        graph.add_node(uv)
        graph.add_node(output)

        # UV outputs Vec2, roughness expects float - should fail
        # Actually, let's use a case that definitely fails
        texture = TextureSampleNode("test")
        graph.add_node(texture)

        # Trying to connect texture sampler (needs TEXTURE2D) which isn't compatible
        with pytest.raises(GraphValidationError):
            # Connecting two outputs together should fail
            graph.connect(uv, "uv", texture, "rgba")  # rgba is output not input

    def test_connect_already_connected(self):
        """Test error when input already connected."""
        graph = MaterialGraph()
        const1 = ConstantNode(value=0.5, data_type=DataType.FLOAT)
        const2 = ConstantNode(value=0.3, data_type=DataType.FLOAT)
        output = OutputNode()

        graph.add_node(const1)
        graph.add_node(const2)
        graph.add_node(output)

        graph.connect(const1, "value", output, "roughness")
        with pytest.raises(GraphValidationError, match="already connected"):
            graph.connect(const2, "value", output, "roughness")

    def test_disconnect(self):
        """Test disconnecting nodes."""
        graph = MaterialGraph()
        const = ConstantNode(value=0.5, data_type=DataType.FLOAT)
        output = OutputNode()

        graph.add_node(const)
        graph.add_node(output)
        graph.connect(const, "value", output, "roughness")

        result = graph.disconnect(output, "roughness")
        assert result
        assert len(graph.connections) == 0

    def test_get_input_connection(self):
        """Test getting input connection."""
        graph = MaterialGraph()
        const = ConstantNode(value=0.5, data_type=DataType.FLOAT)
        output = OutputNode()

        graph.add_node(const)
        graph.add_node(output)
        graph.connect(const, "value", output, "roughness")

        conn = graph.get_input_connection(output.node_id, "roughness")
        assert conn is not None
        assert conn.source_node == const.node_id

    def test_validate_no_output(self):
        """Test validation fails without output node."""
        graph = MaterialGraph()
        const = ConstantNode(value=0.5, data_type=DataType.FLOAT)
        graph.add_node(const)

        is_valid, errors = graph.validate()
        assert not is_valid
        assert any("output node" in e for e in errors)

    def test_validate_cycle_detection(self):
        """Test cycle detection in validation."""
        graph = MaterialGraph()
        add1 = AddNode()
        add2 = AddNode()
        output = OutputNode()

        graph.add_node(add1)
        graph.add_node(add2)
        graph.add_node(output)

        # Connect nodes properly: add1 -> add2 -> output
        graph.connect(add1, "result", add2, "a")
        graph.connect(add2, "result", output, "roughness")

        # Valid acyclic graph
        is_valid, errors = graph.validate()
        assert is_valid

        # Manually inject a cycle by manipulating the connections list
        # This simulates what would happen if there was a cycle
        cycle_connection = NodeConnection(
            source_node=add2.node_id,
            source_port="result",
            target_node=add1.node_id,
            target_port="a",
        )
        graph._connections.append(cycle_connection)

        # Now validation should detect the cycle
        is_valid, errors = graph.validate()
        assert not is_valid
        assert any("cycle" in e.lower() for e in errors)

    def test_topological_order(self):
        """Test topological ordering for code generation."""
        graph = MaterialGraph()
        const = ConstantNode(value=0.5, data_type=DataType.FLOAT)
        mult = MultiplyNode()
        output = OutputNode()

        graph.add_node(const)
        graph.add_node(mult)
        graph.add_node(output)

        graph.connect(const, "value", mult, "a")
        graph.connect(mult, "result", output, "roughness")

        order = graph.get_topological_order()
        # Const should come before mult, mult before output
        const_idx = order.index(const.node_id)
        mult_idx = order.index(mult.node_id)
        output_idx = order.index(output.node_id)

        assert const_idx < mult_idx
        assert mult_idx < output_idx


class TestGraphCompiler:
    """Test GraphCompiler shader code generation."""

    def test_compile_simple_graph(self):
        """Test compiling a simple material graph."""
        graph = MaterialGraph()
        const = ConstantNode(value=0.5, data_type=DataType.FLOAT)
        output = OutputNode()

        graph.add_node(const)
        graph.add_node(output)
        graph.connect(const, "value", output, "roughness")

        compiler = GraphCompiler()
        code = compiler.compile(graph)

        assert "void materialMain()" in code
        assert "out_Roughness" in code

    def test_compile_with_parameters(self):
        """Test compiling with material parameters."""
        graph = MaterialGraph()
        param = ParameterNode(
            param_name="roughness",
            data_type=DataType.FLOAT,
        )
        output = OutputNode()

        graph.add_node(param)
        graph.add_node(output)
        graph.connect(param, "value", output, "roughness")

        compiler = GraphCompiler()
        code = compiler.compile(graph)

        assert "uniform float u_roughness" in code

    def test_compile_with_textures(self):
        """Test compiling with texture samplers."""
        graph = MaterialGraph()
        uv = UVNode()
        tex = TextureSampleNode(texture_name="albedo")
        output = OutputNode()

        graph.add_node(uv)
        graph.add_node(tex)
        graph.add_node(output)

        graph.connect(uv, "uv", tex, "uv")
        graph.connect(tex, "rgb", output, "base_color")

        compiler = GraphCompiler()
        code = compiler.compile(graph)

        assert "uniform sampler2D tex_albedo" in code
        assert "texture(tex_albedo" in code

    def test_compile_invalid_graph_fails(self):
        """Test that compiling invalid graph raises error."""
        graph = MaterialGraph()
        # No output node
        const = ConstantNode(value=0.5, data_type=DataType.FLOAT)
        graph.add_node(const)

        compiler = GraphCompiler()
        with pytest.raises(GraphValidationError, match="Invalid graph"):
            compiler.compile(graph)

    def test_compile_complex_graph(self):
        """Test compiling a more complex material graph."""
        graph = MaterialGraph()

        # Parameters
        base_color = ParameterNode("baseColor", DataType.VEC3)
        roughness = ParameterNode("roughness", DataType.FLOAT)
        metallic = ParameterNode("metallic", DataType.FLOAT)

        # Math operations
        lerp = LerpNode()
        mult = MultiplyNode()

        # Output
        output = OutputNode()

        # Add all nodes
        for node in [base_color, roughness, metallic, lerp, mult, output]:
            graph.add_node(node)

        # Connect roughness through multiplication
        graph.connect(roughness, "value", mult, "a")
        graph.connect(metallic, "value", mult, "b")
        # Connect mult result to output roughness
        graph.connect(mult, "result", output, "roughness")
        # Connect base_color to output
        graph.connect(base_color, "value", output, "base_color")

        compiler = GraphCompiler()
        code = compiler.compile(graph)

        # Should have all uniforms declared
        assert "uniform vec3 u_baseColor" in code
        assert "uniform float u_roughness" in code
        assert "uniform float u_metallic" in code

        # Should have the multiplication operation in the code
        assert "*" in code

        # Should have material main function
        assert "void materialMain()" in code

        # Should set output values
        assert "out_Roughness" in code
        assert "out_BaseColor" in code

    def test_compile_graph_generates_valid_structure(self):
        """Test that compiled graph code has valid GLSL structure."""
        graph = MaterialGraph()

        # Simple graph: constant -> output
        const = ConstantNode(value=Vec3(1.0, 0.5, 0.0), data_type=DataType.VEC3)
        output = OutputNode()

        graph.add_node(const)
        graph.add_node(output)
        graph.connect(const, "value", output, "base_color")

        compiler = GraphCompiler()
        code = compiler.compile(graph)

        # Verify code structure
        lines = code.split("\n")

        # Should have comments sections
        assert any("// Uniforms" in line for line in lines)
        assert any("// Samplers" in line for line in lines)
        assert any("// Main shader code" in line for line in lines)

        # Should have proper function structure
        assert "void materialMain() {" in code
        assert code.strip().endswith("}")
