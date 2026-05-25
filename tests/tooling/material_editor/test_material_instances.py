"""Tests for material instances."""
import pytest
from engine.tooling.material_editor.material_instances import (
    InstanceState, ParameterOverride, MaterialDefinition, MaterialInstance,
    MaterialInstanceManager
)
from engine.tooling.material_editor.material_parameters import (
    ScalarParameter, ColorParameter, TextureParameter, ParameterType
)


class TestParameterOverride:
    """Tests for ParameterOverride."""

    def test_create_override(self):
        """Test creating parameter override."""
        override = ParameterOverride(
            parameter_name="roughness",
            value=0.8,
            enabled=True
        )
        assert override.parameter_name == "roughness"
        assert override.value == 0.8
        assert override.enabled is True

    def test_to_dict(self):
        """Test serialization to dict."""
        override = ParameterOverride("roughness", 0.8)
        data = override.to_dict()
        assert data["parameter_name"] == "roughness"
        assert data["value"] == 0.8

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {"parameter_name": "roughness", "value": 0.8, "enabled": True}
        override = ParameterOverride.from_dict(data)
        assert override.parameter_name == "roughness"
        assert override.value == 0.8


class TestMaterialDefinition:
    """Tests for MaterialDefinition."""

    @pytest.fixture
    def definition(self):
        """Create a test definition."""
        defn = MaterialDefinition("TestMaterial", "shaders/test.shader")
        defn.add_parameter(ScalarParameter("roughness", default=0.5))
        defn.add_parameter(ScalarParameter("metallic", default=0.0))
        defn.add_parameter(ColorParameter("baseColor", default=(1.0, 1.0, 1.0, 1.0)))
        return defn

    def test_create_definition(self, definition):
        """Test creating material definition."""
        assert definition.name == "TestMaterial"
        assert definition.shader_path == "shaders/test.shader"

    def test_add_parameter(self, definition):
        """Test adding parameters."""
        assert "roughness" in definition.parameters.names
        assert "metallic" in definition.parameters.names

    def test_remove_parameter(self, definition):
        """Test removing parameter."""
        param = definition.remove_parameter("metallic")
        assert param is not None
        assert "metallic" not in definition.parameters.names

    def test_get_parameter(self, definition):
        """Test getting parameter."""
        param = definition.get_parameter("roughness")
        assert param is not None
        assert param.default_value == 0.5

    def test_version_increments_on_change(self, definition):
        """Test version increments when parameters change."""
        initial_version = definition.version
        definition.add_parameter(ScalarParameter("ao", default=1.0))
        assert definition.version > initial_version

    def test_serialization(self, definition):
        """Test serialization and deserialization."""
        data = definition.to_dict()
        restored = MaterialDefinition.from_dict(data)

        assert restored.name == definition.name
        assert restored.shader_path == definition.shader_path
        assert "roughness" in restored.parameters.names


class TestMaterialInstance:
    """Tests for MaterialInstance."""

    @pytest.fixture
    def definition(self):
        """Create a test definition."""
        defn = MaterialDefinition("TestMaterial")
        defn.add_parameter(ScalarParameter("roughness", default=0.5))
        defn.add_parameter(ScalarParameter("metallic", default=0.0))
        defn.add_parameter(ColorParameter("baseColor", default=(1.0, 1.0, 1.0, 1.0)))
        return defn

    @pytest.fixture
    def instance(self, definition):
        """Create a test instance."""
        return MaterialInstance("TestInstance", definition)

    def test_create_instance(self, instance, definition):
        """Test creating material instance."""
        assert instance.name == "TestInstance"
        assert instance.parent == definition

    def test_set_override(self, instance):
        """Test setting parameter override."""
        result = instance.set_override("roughness", 0.8)
        assert result is True
        assert instance.has_override("roughness") is True

    def test_set_invalid_override(self, instance):
        """Test setting override for non-existent parameter."""
        result = instance.set_override("nonexistent", 0.5)
        assert result is False

    def test_clear_override(self, instance):
        """Test clearing override."""
        instance.set_override("roughness", 0.8)
        result = instance.clear_override("roughness")
        assert result is True
        assert instance.has_override("roughness") is False

    def test_clear_all_overrides(self, instance):
        """Test clearing all overrides."""
        instance.set_override("roughness", 0.8)
        instance.set_override("metallic", 1.0)
        instance.clear_all_overrides()
        assert instance.override_count == 0

    def test_get_effective_value_with_override(self, instance):
        """Test effective value returns override."""
        instance.set_override("roughness", 0.8)
        value = instance.get_effective_value("roughness")
        assert value == 0.8

    def test_get_effective_value_without_override(self, instance):
        """Test effective value returns default."""
        value = instance.get_effective_value("roughness")
        assert value == 0.5  # Default from definition

    def test_get_all_effective_values(self, instance):
        """Test getting all effective values."""
        instance.set_override("roughness", 0.8)
        values = instance.get_all_effective_values()
        assert values["roughness"] == 0.8
        assert values["metallic"] == 0.0

    def test_enable_disable_override(self, instance):
        """Test enabling and disabling override."""
        instance.set_override("roughness", 0.8)
        instance.disable_override("roughness")

        # Disabled override should return default
        assert instance.has_override("roughness") is False
        assert instance.get_effective_value("roughness") == 0.5

        instance.enable_override("roughness")
        assert instance.has_override("roughness") is True
        assert instance.get_effective_value("roughness") == 0.8

    def test_tags(self, instance):
        """Test instance tags."""
        instance.add_tag("wood")
        instance.add_tag("exterior")

        assert instance.has_tag("wood") is True
        assert "wood" in instance.tags

        instance.remove_tag("wood")
        assert instance.has_tag("wood") is False

    def test_metadata(self, instance):
        """Test instance metadata."""
        instance.set_metadata("author", "Test")
        assert instance.get_metadata("author") == "Test"

    def test_clone(self, instance):
        """Test cloning instance."""
        instance.set_override("roughness", 0.8)
        instance.add_tag("test")

        clone = instance.clone("ClonedInstance")

        assert clone.name == "ClonedInstance"
        assert clone.get_effective_value("roughness") == 0.8
        assert clone.has_tag("test") is True
        assert clone.id != instance.id

    def test_state_tracking(self, instance, definition):
        """Test instance state tracking."""
        assert instance.state == InstanceState.VALID

        # Modify definition
        definition.add_parameter(ScalarParameter("ao", default=1.0))

        assert instance.state == InstanceState.DIRTY

    def test_refresh(self, instance, definition):
        """Test refreshing instance after parent change."""
        definition.add_parameter(ScalarParameter("ao", default=1.0))
        instance.refresh()
        assert instance.state == InstanceState.VALID

    def test_callback_on_changed(self, instance):
        """Test callback when instance changes."""
        changed = [False]

        def on_change():
            changed[0] = True

        instance.on_changed(on_change)
        instance.set_override("roughness", 0.8)

        assert changed[0] is True

    def test_serialization(self, instance, definition):
        """Test serialization and deserialization."""
        instance.set_override("roughness", 0.8)
        instance.add_tag("test")

        data = instance.to_dict()
        restored = MaterialInstance.from_dict(data, definition)

        assert restored.name == instance.name
        assert restored.get_effective_value("roughness") == 0.8
        assert restored.has_tag("test") is True


class TestMaterialInstanceManager:
    """Tests for MaterialInstanceManager."""

    @pytest.fixture
    def manager(self):
        """Create a test manager."""
        return MaterialInstanceManager()

    def test_create_definition(self, manager):
        """Test creating definition."""
        defn = manager.create_definition("TestMaterial", "shaders/test.shader")
        assert defn is not None
        assert manager.definition_count == 1

    def test_get_definition(self, manager):
        """Test getting definition by ID."""
        defn = manager.create_definition("TestMaterial")
        retrieved = manager.get_definition(defn.id)
        assert retrieved == defn

    def test_get_definition_by_name(self, manager):
        """Test getting definition by name."""
        defn = manager.create_definition("TestMaterial")
        retrieved = manager.get_definition_by_name("TestMaterial")
        assert retrieved == defn

    def test_remove_definition(self, manager):
        """Test removing definition."""
        defn = manager.create_definition("TestMaterial")
        result = manager.remove_definition(defn.id)
        assert result is True
        assert manager.definition_count == 0

    def test_remove_definition_removes_instances(self, manager):
        """Test removing definition also removes its instances."""
        defn = manager.create_definition("TestMaterial")
        defn.add_parameter(ScalarParameter("roughness", default=0.5))
        manager.create_instance("Instance1", defn.id)
        manager.create_instance("Instance2", defn.id)

        manager.remove_definition(defn.id)

        assert manager.instance_count == 0

    def test_create_instance(self, manager):
        """Test creating instance."""
        defn = manager.create_definition("TestMaterial")
        defn.add_parameter(ScalarParameter("roughness", default=0.5))

        instance = manager.create_instance("TestInstance", defn.id)
        assert instance is not None
        assert manager.instance_count == 1

    def test_create_instance_invalid_definition(self, manager):
        """Test creating instance with invalid definition."""
        instance = manager.create_instance("TestInstance", "invalid")
        assert instance is None

    def test_get_instance(self, manager):
        """Test getting instance by ID."""
        defn = manager.create_definition("TestMaterial")
        instance = manager.create_instance("TestInstance", defn.id)
        retrieved = manager.get_instance(instance.id)
        assert retrieved == instance

    def test_get_instance_by_name(self, manager):
        """Test getting instance by name."""
        defn = manager.create_definition("TestMaterial")
        instance = manager.create_instance("TestInstance", defn.id)
        retrieved = manager.get_instance_by_name("TestInstance")
        assert retrieved == instance

    def test_remove_instance(self, manager):
        """Test removing instance."""
        defn = manager.create_definition("TestMaterial")
        instance = manager.create_instance("TestInstance", defn.id)
        result = manager.remove_instance(instance.id)
        assert result is True
        assert manager.instance_count == 0

    def test_get_instances_of_definition(self, manager):
        """Test getting all instances of a definition."""
        defn = manager.create_definition("TestMaterial")
        manager.create_instance("Instance1", defn.id)
        manager.create_instance("Instance2", defn.id)

        instances = manager.get_instances_of_definition(defn.id)
        assert len(instances) == 2

    def test_get_instances_by_tag(self, manager):
        """Test getting instances by tag."""
        defn = manager.create_definition("TestMaterial")
        inst1 = manager.create_instance("Instance1", defn.id)
        inst2 = manager.create_instance("Instance2", defn.id)
        inst1.add_tag("wood")

        instances = manager.get_instances_by_tag("wood")
        assert len(instances) == 1
        assert instances[0] == inst1

    def test_get_all_definitions(self, manager):
        """Test getting all definitions."""
        manager.create_definition("Material1")
        manager.create_definition("Material2")

        definitions = manager.get_all_definitions()
        assert len(definitions) == 2

    def test_get_all_instances(self, manager):
        """Test getting all instances."""
        defn = manager.create_definition("TestMaterial")
        manager.create_instance("Instance1", defn.id)
        manager.create_instance("Instance2", defn.id)

        instances = manager.get_all_instances()
        assert len(instances) == 2

    def test_serialization(self, manager):
        """Test manager serialization."""
        defn = manager.create_definition("TestMaterial")
        defn.add_parameter(ScalarParameter("roughness", default=0.5))
        inst = manager.create_instance("TestInstance", defn.id)
        inst.set_override("roughness", 0.8)

        data = manager.to_dict()
        restored = MaterialInstanceManager.from_dict(data)

        assert restored.definition_count == 1
        assert restored.instance_count == 1

        restored_inst = restored.get_instance_by_name("TestInstance")
        assert restored_inst.get_effective_value("roughness") == 0.8

    def test_clear(self, manager):
        """Test clearing manager."""
        defn = manager.create_definition("TestMaterial")
        manager.create_instance("TestInstance", defn.id)

        manager.clear()

        assert manager.definition_count == 0
        assert manager.instance_count == 0


class TestMaterialInstanceManagerFromDefinition:
    """Tests for creating instances directly from definition objects."""

    def test_create_instance_from_definition(self):
        """Test creating instance from definition object."""
        manager = MaterialInstanceManager()
        defn = MaterialDefinition("TestMaterial")
        defn.add_parameter(ScalarParameter("roughness", default=0.5))

        instance = manager.create_instance_from_definition("TestInstance", defn)

        assert instance is not None
        assert manager.definition_count == 1  # Definition auto-registered
        assert manager.instance_count == 1
