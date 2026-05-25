"""Tests for node factory."""
import pytest
from engine.tooling.material_editor.node_factory import (
    NodePreset, NodeFactory, NodeTemplates, get_default_factory
)
from engine.tooling.material_editor.material_nodes import (
    NodeCategory, DataType, ConstantNode, Constant2Node, Constant3Node,
    Constant4Node, ParameterNode, AddNode, TextureSampleNode, PBROutputNode,
    UnlitOutputNode, CustomCodeNode
)


class TestNodePreset:
    """Tests for NodePreset."""

    def test_create_preset(self):
        """Test creating a node preset."""
        preset = NodePreset(
            name="Test Preset",
            description="A test preset",
            node_type="Constant",
            custom_config={"value": 0.5},
            category="Test",
            tags=["test", "preset"]
        )
        assert preset.name == "Test Preset"
        assert preset.node_type == "Constant"
        assert preset.custom_config["value"] == 0.5


class TestNodeFactory:
    """Tests for NodeFactory."""

    @pytest.fixture
    def factory(self):
        """Create a fresh factory."""
        return NodeFactory()

    def test_get_node_types(self, factory):
        """Test getting available node types."""
        types = factory.get_node_types()
        assert "Constant" in types
        assert "Add" in types
        assert "Multiply" in types
        assert "PBROutput" in types

    def test_get_node_types_by_category(self, factory):
        """Test getting node types by category."""
        math_types = factory.get_node_types_by_category(NodeCategory.MATH)
        assert "Add" in math_types
        assert "Multiply" in math_types
        assert "Constant" not in math_types  # INPUT category

    def test_create_node(self, factory):
        """Test creating a node by type."""
        node = factory.create_node("Constant")
        assert isinstance(node, ConstantNode)

    def test_create_node_with_name(self, factory):
        """Test creating a node with custom name."""
        node = factory.create_node("Constant", name="MyConstant")
        assert node.name == "MyConstant"

    def test_create_invalid_node(self, factory):
        """Test creating invalid node type returns None."""
        node = factory.create_node("NonExistent")
        assert node is None

    def test_create_constant(self, factory):
        """Test creating constant node helper."""
        node = factory.create_constant(0.5, "Half")
        assert isinstance(node, ConstantNode)
        assert node.value == 0.5
        assert node.name == "Half"

    def test_create_constant2(self, factory):
        """Test creating float2 constant."""
        node = factory.create_constant2(1.0, 2.0)
        assert isinstance(node, Constant2Node)
        assert node.value == (1.0, 2.0)

    def test_create_constant3(self, factory):
        """Test creating float3 constant."""
        node = factory.create_constant3(1.0, 2.0, 3.0)
        assert isinstance(node, Constant3Node)
        assert node.value == (1.0, 2.0, 3.0)

    def test_create_constant4(self, factory):
        """Test creating float4 constant."""
        node = factory.create_constant4(1.0, 2.0, 3.0, 4.0)
        assert isinstance(node, Constant4Node)
        assert node.value == (1.0, 2.0, 3.0, 4.0)

    def test_create_color(self, factory):
        """Test creating color constant."""
        node = factory.create_color(0.8, 0.2, 0.1, 1.0, "Red")
        assert isinstance(node, Constant4Node)
        assert node.value == (0.8, 0.2, 0.1, 1.0)

    def test_create_parameter(self, factory):
        """Test creating parameter node."""
        node = factory.create_parameter("Roughness", DataType.FLOAT)
        assert isinstance(node, ParameterNode)
        assert node.param_name == "Roughness"

    def test_create_texture_sample(self, factory):
        """Test creating texture sample node."""
        node = factory.create_texture_sample("textures/albedo.png")
        assert isinstance(node, TextureSampleNode)
        assert node.texture_path == "textures/albedo.png"

    def test_create_pbr_output(self, factory):
        """Test creating PBR output node."""
        node = factory.create_pbr_output()
        assert isinstance(node, PBROutputNode)

    def test_create_unlit_output(self, factory):
        """Test creating unlit output node."""
        node = factory.create_unlit_output()
        assert isinstance(node, UnlitOutputNode)

    def test_create_custom(self, factory):
        """Test creating custom code node."""
        node = factory.create_custom("float x = 1.0;")
        assert isinstance(node, CustomCodeNode)
        assert node.code == "float x = 1.0;"


class TestNodePresets:
    """Tests for node presets."""

    @pytest.fixture
    def factory(self):
        return NodeFactory()

    def test_get_presets(self, factory):
        """Test getting available presets."""
        presets = factory.get_presets()
        assert len(presets) > 0

    def test_default_presets_exist(self, factory):
        """Test default presets are registered."""
        preset_names = [p.name for p in factory.get_presets()]
        assert "White" in preset_names
        assert "Black" in preset_names
        assert "Gray 50%" in preset_names

    def test_get_presets_by_category(self, factory):
        """Test getting presets by category."""
        color_presets = factory.get_presets_by_category("Colors")
        assert len(color_presets) > 0
        assert all(p.category == "Colors" for p in color_presets)

    def test_get_preset_categories(self, factory):
        """Test getting preset categories."""
        categories = factory.get_preset_categories()
        assert "Colors" in categories
        assert "Values" in categories

    def test_search_presets(self, factory):
        """Test searching presets."""
        results = factory.search_presets("white")
        assert len(results) > 0
        assert any("White" in p.name for p in results)

    def test_search_presets_by_tag(self, factory):
        """Test searching presets by tag."""
        results = factory.search_presets("color")
        assert len(results) > 0

    def test_create_from_preset(self, factory):
        """Test creating node from preset."""
        node = factory.create_from_preset("White")
        assert isinstance(node, Constant4Node)
        assert node.value == (1.0, 1.0, 1.0, 1.0)

    def test_create_from_invalid_preset(self, factory):
        """Test creating from invalid preset returns None."""
        node = factory.create_from_preset("NonExistent")
        assert node is None

    def test_register_custom_preset(self, factory):
        """Test registering custom preset."""
        preset = NodePreset(
            name="Custom Orange",
            description="Orange color",
            node_type="Constant4",
            custom_config={"value": (1.0, 0.5, 0.0, 1.0)},
            category="Custom",
            tags=["custom", "orange"]
        )
        factory.register_preset(preset)

        node = factory.create_from_preset("Custom Orange")
        assert node is not None
        assert node.value == (1.0, 0.5, 0.0, 1.0)

    def test_unregister_preset(self, factory):
        """Test unregistering preset."""
        preset = NodePreset(name="ToRemove", description="", node_type="Constant")
        factory.register_preset(preset)
        factory.unregister_preset("ToRemove")

        node = factory.create_from_preset("ToRemove")
        assert node is None

    def test_register_preset_creator(self, factory):
        """Test registering custom preset creator."""
        def create_special():
            node = ConstantNode(value=42.0, name="Special")
            node.position = (100, 100)
            return node

        factory.register_preset_creator("Special", create_special)
        node = factory.create_from_preset("Special")

        assert node.value == 42.0
        assert node.position == (100, 100)


class TestNodeTemplates:
    """Tests for NodeTemplates."""

    @pytest.fixture
    def factory(self):
        return NodeFactory()

    def test_create_pbr_material_nodes(self, factory):
        """Test creating PBR material node setup."""
        nodes = NodeTemplates.create_pbr_material_nodes(factory)

        assert len(nodes) == 4  # Albedo, Metallic, Roughness, Output

        # Check types
        node_types = [type(n).__name__ for n in nodes]
        assert "Constant4Node" in node_types
        assert "ConstantNode" in node_types
        assert "PBROutputNode" in node_types

    def test_create_textured_pbr_nodes(self, factory):
        """Test creating textured PBR node setup."""
        nodes = NodeTemplates.create_textured_pbr_nodes(factory)

        # Check expected nodes
        node_types = [type(n).__name__ for n in nodes]
        assert "TextureSampleNode" in node_types
        assert "PBROutputNode" in node_types

    def test_create_unlit_material_nodes(self, factory):
        """Test creating unlit material node setup."""
        nodes = NodeTemplates.create_unlit_material_nodes(factory)

        node_types = [type(n).__name__ for n in nodes]
        assert "UnlitOutputNode" in node_types
        assert "Constant4Node" in node_types

    def test_create_fresnel_effect_nodes(self, factory):
        """Test creating fresnel effect nodes."""
        nodes = NodeTemplates.create_fresnel_effect_nodes(factory)

        node_names = [n.name for n in nodes]
        assert "Fresnel" in node_names

    def test_create_animated_uv_nodes(self, factory):
        """Test creating animated UV nodes."""
        nodes = NodeTemplates.create_animated_uv_nodes(factory)

        node_names = [n.name for n in nodes]
        assert "Time" in node_names

    def test_nodes_have_positions(self, factory):
        """Test template nodes have positions set."""
        nodes = NodeTemplates.create_pbr_material_nodes(factory)
        for node in nodes:
            assert node.position is not None


class TestDefaultFactory:
    """Tests for default factory."""

    def test_get_default_factory(self):
        """Test getting default factory."""
        factory = get_default_factory()
        assert factory is not None
        assert isinstance(factory, NodeFactory)

    def test_default_factory_is_singleton(self):
        """Test default factory returns same instance."""
        factory1 = get_default_factory()
        factory2 = get_default_factory()
        assert factory1 is factory2


class TestRegisterCustomNodeType:
    """Tests for registering custom node types."""

    def test_register_node_type(self):
        """Test registering custom node type."""
        factory = NodeFactory()

        # Create a simple custom node class
        class MyCustomNode(ConstantNode):
            pass

        factory.register_node_type("MyCustom", MyCustomNode)
        node = factory.create_node("MyCustom")

        assert isinstance(node, MyCustomNode)

    def test_unregister_node_type(self):
        """Test unregistering node type."""
        factory = NodeFactory()
        factory.register_node_type("ToRemove", ConstantNode)
        factory.unregister_node_type("ToRemove")

        node = factory.create_node("ToRemove")
        assert node is None
