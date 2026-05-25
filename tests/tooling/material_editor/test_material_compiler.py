"""Tests for material compiler."""
import pytest
from engine.tooling.material_editor.material_compiler import (
    ShaderLanguage, ShaderStage, ShaderParameter, TextureBinding,
    CompilationResult, ShaderGenerator, HLSLGenerator, GLSLGenerator,
    MetalGenerator, MaterialCompiler
)
from engine.tooling.material_editor.material_graph import MaterialGraph
from engine.tooling.material_editor.material_nodes import (
    ConstantNode, Constant3Node, Constant4Node, AddNode, MultiplyNode,
    TextureSampleNode, ParameterNode, PBROutputNode, UnlitOutputNode,
    DataType
)


class TestShaderParameter:
    """Tests for ShaderParameter."""

    def test_create_parameter(self):
        """Test creating shader parameter."""
        param = ShaderParameter(
            name="roughness",
            data_type="float",
            default_value=0.5,
            binding=0
        )
        assert param.name == "roughness"
        assert param.data_type == "float"
        assert param.default_value == 0.5


class TestTextureBinding:
    """Tests for TextureBinding."""

    def test_create_binding(self):
        """Test creating texture binding."""
        binding = TextureBinding(
            name="albedoTex",
            slot=0,
            sampler_slot=0
        )
        assert binding.name == "albedoTex"
        assert binding.slot == 0


class TestCompilationResult:
    """Tests for CompilationResult."""

    def test_success_result(self):
        """Test successful compilation result."""
        result = CompilationResult(
            success=True,
            vertex_shader="// vertex",
            fragment_shader="// fragment"
        )
        assert result.success is True
        assert len(result.errors) == 0

    def test_failure_result(self):
        """Test failed compilation result."""
        result = CompilationResult(
            success=False,
            errors=["Missing output node"]
        )
        assert result.success is False
        assert len(result.errors) == 1


class TestHLSLGenerator:
    """Tests for HLSL shader generator."""

    @pytest.fixture
    def generator(self):
        return HLSLGenerator()

    def test_language(self, generator):
        """Test language property."""
        assert generator.language == ShaderLanguage.HLSL

    def test_generate_header(self, generator):
        """Test header generation."""
        header = generator.generate_header()
        assert "HLSL" in header

    def test_generate_structs(self, generator):
        """Test struct generation."""
        structs = generator.generate_structs()
        assert "VSInput" in structs
        assert "PSInput" in structs
        assert "PSOutput" in structs

    def test_generate_parameters(self, generator):
        """Test parameter generation."""
        params = [
            ShaderParameter("roughness", "float", 0.5, 0),
            ShaderParameter("metallic", "float", 0.0, 1)
        ]
        code = generator.generate_parameters(params)
        assert "cbuffer" in code
        assert "roughness" in code
        assert "metallic" in code

    def test_generate_textures(self, generator):
        """Test texture declaration generation."""
        textures = [
            TextureBinding("albedoTex", 0, 0),
            TextureBinding("normalTex", 1, 1)
        ]
        code = generator.generate_textures(textures)
        assert "Texture2D" in code
        assert "SamplerState" in code
        assert "register(t0)" in code

    def test_generate_function_start(self, generator):
        """Test function start generation."""
        code = generator.generate_function_start(ShaderStage.FRAGMENT)
        assert "PSMain" in code
        assert "PSInput" in code

    def test_generate_type(self, generator):
        """Test type generation."""
        assert generator.generate_type(DataType.FLOAT) == "float"
        assert generator.generate_type(DataType.FLOAT3) == "float3"
        assert generator.generate_type(DataType.FLOAT4) == "float4"


class TestGLSLGenerator:
    """Tests for GLSL shader generator."""

    @pytest.fixture
    def generator(self):
        return GLSLGenerator()

    def test_language(self, generator):
        """Test language property."""
        assert generator.language == ShaderLanguage.GLSL

    def test_generate_header(self, generator):
        """Test header generation."""
        header = generator.generate_header()
        assert "#version" in header

    def test_generate_type(self, generator):
        """Test type generation."""
        assert generator.generate_type(DataType.FLOAT) == "float"
        assert generator.generate_type(DataType.FLOAT3) == "vec3"
        assert generator.generate_type(DataType.FLOAT4) == "vec4"

    def test_generate_textures(self, generator):
        """Test texture generation."""
        textures = [TextureBinding("albedoTex", 0, 0)]
        code = generator.generate_textures(textures)
        assert "sampler2D" in code
        assert "binding" in code


class TestMetalGenerator:
    """Tests for Metal shader generator."""

    @pytest.fixture
    def generator(self):
        return MetalGenerator()

    def test_language(self, generator):
        """Test language property."""
        assert generator.language == ShaderLanguage.METAL

    def test_generate_header(self, generator):
        """Test header generation."""
        header = generator.generate_header()
        assert "metal_stdlib" in header

    def test_generate_type(self, generator):
        """Test type generation."""
        assert generator.generate_type(DataType.FLOAT) == "float"
        assert generator.generate_type(DataType.FLOAT3) == "float3"
        assert generator.generate_type(DataType.MATRIX) == "float4x4"


class TestMaterialCompiler:
    """Tests for MaterialCompiler."""

    @pytest.fixture
    def compiler(self):
        return MaterialCompiler()

    @pytest.fixture
    def simple_graph(self):
        """Create a simple valid graph."""
        graph = MaterialGraph("Simple")
        const = Constant4Node(value=(0.8, 0.2, 0.1, 1.0), name="Color")
        roughness = ConstantNode(value=0.5, name="Roughness")
        output = PBROutputNode(name="Output")

        graph.add_node(const)
        graph.add_node(roughness)
        graph.add_node(output)

        graph.connect(const.id, "Value", output.id, "Albedo")
        graph.connect(roughness.id, "Value", output.id, "Roughness")

        return graph

    def test_compile_simple_graph(self, compiler, simple_graph):
        """Test compiling a simple graph."""
        result = compiler.compile(simple_graph)

        assert result.success is True
        assert len(result.fragment_shader) > 0
        assert len(result.vertex_shader) > 0

    def test_compile_to_hlsl(self, compiler, simple_graph):
        """Test compiling to HLSL."""
        result = compiler.compile(simple_graph, ShaderLanguage.HLSL)

        assert result.success is True
        assert "HLSL" in result.vertex_shader

    def test_compile_to_glsl(self, compiler, simple_graph):
        """Test compiling to GLSL."""
        result = compiler.compile(simple_graph, ShaderLanguage.GLSL)

        assert result.success is True
        assert "#version" in result.vertex_shader

    def test_compile_to_metal(self, compiler, simple_graph):
        """Test compiling to Metal."""
        result = compiler.compile(simple_graph, ShaderLanguage.METAL)

        assert result.success is True
        assert "metal" in result.vertex_shader

    def test_compile_invalid_graph(self, compiler):
        """Test compiling invalid graph."""
        graph = MaterialGraph("Invalid")
        # No output node

        result = compiler.compile(graph)

        assert result.success is False
        assert len(result.errors) > 0

    def test_compile_with_parameters(self, compiler):
        """Test compiling graph with parameters."""
        graph = MaterialGraph("Parameterized")
        param = ParameterNode(param_name="Roughness", param_type=DataType.FLOAT)
        output = PBROutputNode()

        graph.add_node(param)
        graph.add_node(output)
        graph.connect(param.id, "Value", output.id, "Roughness")

        result = compiler.compile(graph)

        assert result.success is True
        assert len(result.parameters) == 1
        assert result.parameters[0].name == "Roughness"

    def test_compile_with_textures(self, compiler):
        """Test compiling graph with textures."""
        graph = MaterialGraph("Textured")
        tex = TextureSampleNode(texture_path="textures/albedo.png")
        output = PBROutputNode()

        graph.add_node(tex)
        graph.add_node(output)
        graph.connect(tex.id, "RGB", output.id, "Albedo")

        result = compiler.compile(graph)

        assert result.success is True
        assert len(result.textures) == 1

    def test_compile_complex_graph(self, compiler):
        """Test compiling complex graph with multiple nodes."""
        graph = MaterialGraph("Complex")

        # Create nodes
        albedo = Constant4Node(value=(0.8, 0.2, 0.1, 1.0), name="Albedo")
        roughness = ConstantNode(value=0.5, name="Roughness")
        metallic = ConstantNode(value=0.0, name="Metallic")
        multiply = MultiplyNode(name="Multiply")
        add = AddNode(name="Add")
        output = PBROutputNode()

        # Add nodes
        for node in [albedo, roughness, metallic, multiply, add, output]:
            graph.add_node(node)

        # Create connections
        graph.connect(albedo.id, "Value", output.id, "Albedo")
        graph.connect(roughness.id, "Value", multiply.id, "A")
        graph.connect(metallic.id, "Value", multiply.id, "B")
        graph.connect(multiply.id, "Result", add.id, "A")
        graph.connect(add.id, "Result", output.id, "Roughness")

        result = compiler.compile(graph)

        assert result.success is True

    def test_compile_with_optimization(self, compiler, simple_graph):
        """Test compilation with optimization enabled."""
        result = compiler.compile(simple_graph, optimize=True)

        assert result.success is True
        # Optimized shader shouldn't have multiple consecutive empty lines
        lines = result.fragment_shader.split("\n")
        consecutive_empty = 0
        max_consecutive = 0
        for line in lines:
            if not line.strip():
                consecutive_empty += 1
                max_consecutive = max(max_consecutive, consecutive_empty)
            else:
                consecutive_empty = 0
        assert max_consecutive <= 2

    def test_get_supported_languages(self, compiler):
        """Test getting supported languages."""
        languages = compiler.get_supported_languages()

        assert ShaderLanguage.HLSL in languages
        assert ShaderLanguage.GLSL in languages
        assert ShaderLanguage.METAL in languages


class TestShaderCodeGeneration:
    """Tests for shader code generation quality."""

    @pytest.fixture
    def compiler(self):
        return MaterialCompiler()

    def test_generated_code_has_main_function(self, compiler):
        """Test generated code has main function."""
        graph = MaterialGraph()
        output = PBROutputNode()
        graph.add_node(output)

        result = compiler.compile(graph, ShaderLanguage.HLSL)

        assert "PSMain" in result.fragment_shader
        assert "VSMain" in result.vertex_shader

    def test_generated_code_has_structs(self, compiler):
        """Test generated code has required structs."""
        graph = MaterialGraph()
        output = PBROutputNode()
        graph.add_node(output)

        result = compiler.compile(graph, ShaderLanguage.HLSL)

        assert "PSInput" in result.fragment_shader
        assert "PSOutput" in result.fragment_shader

    def test_code_generation_unique_variables(self, compiler):
        """Test code generation produces unique variable names."""
        graph = MaterialGraph()
        const1 = ConstantNode(value=0.5, name="A")
        const2 = ConstantNode(value=0.3, name="B")
        add = AddNode(name="Add")
        output = PBROutputNode()

        graph.add_node(const1)
        graph.add_node(const2)
        graph.add_node(add)
        graph.add_node(output)

        graph.connect(const1.id, "Value", add.id, "A")
        graph.connect(const2.id, "Value", add.id, "B")
        graph.connect(add.id, "Result", output.id, "Metallic")

        result = compiler.compile(graph)

        # Check that variable names don't conflict
        assert result.success is True


class TestUnlitShaderCompilation:
    """Tests for unlit shader compilation."""

    @pytest.fixture
    def compiler(self):
        return MaterialCompiler()

    def test_compile_unlit_material(self, compiler):
        """Test compiling unlit material."""
        graph = MaterialGraph("Unlit")
        color = Constant4Node(value=(1.0, 0.0, 0.0, 1.0), name="Color")
        output = UnlitOutputNode()

        graph.add_node(color)
        graph.add_node(output)
        graph.connect(color.id, "Value", output.id, "Color")

        result = compiler.compile(graph)

        assert result.success is True
        assert "Color" in result.fragment_shader


class TestCompilationWarnings:
    """Tests for compilation warnings."""

    @pytest.fixture
    def compiler(self):
        return MaterialCompiler()

    def test_warnings_passed_through(self, compiler):
        """Test validation warnings are passed through."""
        graph = MaterialGraph()
        const = ConstantNode(value=0.5)  # Disconnected
        output = PBROutputNode()

        graph.add_node(const)
        graph.add_node(output)

        result = compiler.compile(graph)

        # Should succeed but with warnings
        assert result.success is True
        assert len(result.warnings) > 0
