"""Tests for build configuration management."""
import pytest
from engine.tooling.build.build_config import (
    BuildConfiguration,
    BuildType,
    ConfigurationPreset,
    OptimizationLevel,
    DebugLevel,
    CompilerSettings,
    LinkerSettings,
    AssetSettings,
    ConfigurationManager,
    create_debug_config,
    create_development_config,
    create_shipping_config,
    create_test_config,
)


class TestBuildType:
    """Tests for BuildType enum."""

    def test_build_type_values(self):
        """Test all build type values exist."""
        assert BuildType.FULL
        assert BuildType.INCREMENTAL
        assert BuildType.DISTRIBUTION
        assert BuildType.DEBUG_SYMBOLS

    def test_build_type_unique(self):
        """Test build types are unique."""
        types = [bt.value for bt in BuildType]
        assert len(types) == len(set(types))


class TestOptimizationLevel:
    """Tests for OptimizationLevel enum."""

    def test_optimization_levels(self):
        """Test all optimization levels exist."""
        assert OptimizationLevel.NONE.value == 0
        assert OptimizationLevel.MINIMAL.value == 1
        assert OptimizationLevel.STANDARD.value == 2
        assert OptimizationLevel.AGGRESSIVE.value == 3
        assert OptimizationLevel.SIZE.value == 4
        assert OptimizationLevel.SPEED.value == 5

    def test_optimization_ordering(self):
        """Test optimization levels are ordered."""
        assert OptimizationLevel.NONE.value < OptimizationLevel.AGGRESSIVE.value


class TestDebugLevel:
    """Tests for DebugLevel enum."""

    def test_debug_levels_exist(self):
        """Test all debug levels exist."""
        assert DebugLevel.NONE
        assert DebugLevel.MINIMAL
        assert DebugLevel.STANDARD
        assert DebugLevel.FULL
        assert DebugLevel.PROFILING


class TestConfigurationPreset:
    """Tests for ConfigurationPreset enum."""

    def test_presets_exist(self):
        """Test all presets exist."""
        assert ConfigurationPreset.DEBUG
        assert ConfigurationPreset.DEVELOPMENT
        assert ConfigurationPreset.SHIPPING
        assert ConfigurationPreset.TEST
        assert ConfigurationPreset.PROFILE
        assert ConfigurationPreset.DEMO


class TestCompilerSettings:
    """Tests for CompilerSettings dataclass."""

    def test_default_values(self):
        """Test default compiler settings."""
        settings = CompilerSettings()
        assert settings.optimization == OptimizationLevel.STANDARD
        assert settings.debug_level == DebugLevel.STANDARD
        assert settings.defines == {}
        assert settings.flags == []
        assert settings.warnings_as_errors is False
        assert settings.enable_exceptions is True
        assert settings.enable_rtti is True

    def test_custom_values(self):
        """Test custom compiler settings."""
        settings = CompilerSettings(
            optimization=OptimizationLevel.AGGRESSIVE,
            debug_level=DebugLevel.NONE,
            defines={"NDEBUG": "1"},
            warnings_as_errors=True,
        )
        assert settings.optimization == OptimizationLevel.AGGRESSIVE
        assert settings.defines == {"NDEBUG": "1"}
        assert settings.warnings_as_errors is True

    def test_merge_with(self):
        """Test merging compiler settings."""
        base = CompilerSettings(defines={"A": "1"}, flags=["-O2"])
        override = CompilerSettings(
            optimization=OptimizationLevel.AGGRESSIVE,
            defines={"B": "2"},
            flags=["-O3"],
        )
        merged = base.merge_with(override)

        assert merged.optimization == OptimizationLevel.AGGRESSIVE
        assert "A" in merged.defines
        assert "B" in merged.defines
        assert "-O2" in merged.flags
        assert "-O3" in merged.flags


class TestLinkerSettings:
    """Tests for LinkerSettings dataclass."""

    def test_default_values(self):
        """Test default linker settings."""
        settings = LinkerSettings()
        assert settings.static_linking is False
        assert settings.strip_symbols is False
        assert settings.link_time_optimization is False
        assert settings.dead_code_elimination is True

    def test_merge_with(self):
        """Test merging linker settings."""
        base = LinkerSettings(library_search_paths=["/usr/lib"])
        override = LinkerSettings(
            static_linking=True,
            library_search_paths=["/opt/lib"],
        )
        merged = base.merge_with(override)

        assert merged.static_linking is True
        assert "/usr/lib" in merged.library_search_paths
        assert "/opt/lib" in merged.library_search_paths


class TestAssetSettings:
    """Tests for AssetSettings dataclass."""

    def test_default_values(self):
        """Test default asset settings."""
        settings = AssetSettings()
        assert settings.compress_textures is True
        assert settings.texture_format == "DXT"
        assert settings.max_texture_size == 4096
        assert settings.generate_mipmaps is True

    def test_merge_with(self):
        """Test merging asset settings."""
        base = AssetSettings()
        override = AssetSettings(
            max_texture_size=8192,
            compress_audio=False,
        )
        merged = base.merge_with(override)

        assert merged.max_texture_size == 8192
        assert merged.compress_audio is False


class TestBuildConfiguration:
    """Tests for BuildConfiguration dataclass."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = BuildConfiguration(
            name="Test",
            preset=ConfigurationPreset.DEBUG,
        )
        assert config.name == "Test"
        assert config.preset == ConfigurationPreset.DEBUG
        assert config.build_type == BuildType.INCREMENTAL
        assert config.enable_logging is True

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = BuildConfiguration(
            name="Debug",
            preset=ConfigurationPreset.DEBUG,
        )
        issues = config.validate()
        assert len(issues) == 0

    def test_validate_shipping_with_cheats(self):
        """Test validation catches cheats in shipping."""
        config = BuildConfiguration(
            name="Shipping",
            preset=ConfigurationPreset.SHIPPING,
            enable_cheats=True,
        )
        issues = config.validate()
        assert any("Cheats" in issue for issue in issues)

    def test_validate_shipping_with_dev_tools(self):
        """Test validation catches dev tools in shipping."""
        config = BuildConfiguration(
            name="Shipping",
            preset=ConfigurationPreset.SHIPPING,
            enable_developer_tools=True,
        )
        issues = config.validate()
        assert any("Developer tools" in issue for issue in issues)

    def test_validate_missing_name(self):
        """Test validation catches missing name."""
        config = BuildConfiguration(name="", preset=ConfigurationPreset.DEBUG)
        issues = config.validate()
        assert any("name" in issue.lower() for issue in issues)

    def test_clone(self):
        """Test cloning configuration."""
        original = BuildConfiguration(
            name="Original",
            preset=ConfigurationPreset.DEBUG,
        )
        original.compiler.defines["TEST"] = "1"

        cloned = original.clone("Cloned")
        assert cloned.name == "Cloned"
        assert cloned.preset == ConfigurationPreset.DEBUG
        assert "TEST" in cloned.compiler.defines

        # Verify independence
        cloned.compiler.defines["NEW"] = "2"
        assert "NEW" not in original.compiler.defines

    def test_apply_preset_debug(self):
        """Test applying debug preset."""
        config = BuildConfiguration(name="Test", preset=ConfigurationPreset.DEBUG)
        config.apply_preset(ConfigurationPreset.DEBUG)

        assert config.compiler.optimization == OptimizationLevel.NONE
        assert config.compiler.debug_level == DebugLevel.FULL
        assert config.enable_cheats is True
        assert config.enable_developer_tools is True

    def test_apply_preset_shipping(self):
        """Test applying shipping preset."""
        config = BuildConfiguration(name="Test", preset=ConfigurationPreset.SHIPPING)
        config.apply_preset(ConfigurationPreset.SHIPPING)

        assert config.compiler.optimization == OptimizationLevel.AGGRESSIVE
        assert config.compiler.debug_level == DebugLevel.NONE
        assert config.linker.strip_symbols is True
        assert config.enable_cheats is False
        assert config.enable_logging is False


class TestConfigurationManager:
    """Tests for ConfigurationManager."""

    def test_register_configuration(self):
        """Test registering a configuration."""
        manager = ConfigurationManager()
        config = create_debug_config()
        manager.register(config)

        assert "Debug" in manager.list_names()

    def test_get_configuration(self):
        """Test getting a configuration."""
        manager = ConfigurationManager()
        config = create_debug_config()
        manager.register(config)

        retrieved = manager.get("Debug")
        assert retrieved is not None
        assert retrieved.name == "Debug"

    def test_get_nonexistent(self):
        """Test getting nonexistent configuration."""
        manager = ConfigurationManager()
        assert manager.get("NonExistent") is None

    def test_unregister(self):
        """Test unregistering a configuration."""
        manager = ConfigurationManager()
        config = create_debug_config()
        manager.register(config)

        result = manager.unregister("Debug")
        assert result is True
        assert "Debug" not in manager.list_names()

    def test_set_active(self):
        """Test setting active configuration."""
        manager = ConfigurationManager()
        manager.register(create_debug_config())
        manager.register(create_shipping_config())

        manager.set_active("Debug")
        active = manager.get_active()
        assert active is not None
        assert active.name == "Debug"

    def test_list_names(self):
        """Test listing configuration names."""
        manager = ConfigurationManager()
        manager.register(create_debug_config())
        manager.register(create_development_config())

        names = manager.list_names()
        assert "Debug" in names
        assert "Development" in names


class TestConfigurationFactories:
    """Tests for configuration factory functions."""

    def test_create_debug_config(self):
        """Test creating debug configuration."""
        config = create_debug_config()
        assert config.name == "Debug"
        assert config.preset == ConfigurationPreset.DEBUG
        assert "DEBUG" in config.compiler.defines

    def test_create_debug_config_custom_name(self):
        """Test creating debug configuration with custom name."""
        config = create_debug_config("MyDebug")
        assert config.name == "MyDebug"

    def test_create_development_config(self):
        """Test creating development configuration."""
        config = create_development_config()
        assert config.name == "Development"
        assert config.preset == ConfigurationPreset.DEVELOPMENT
        assert "DEVELOPMENT" in config.compiler.defines

    def test_create_shipping_config(self):
        """Test creating shipping configuration."""
        config = create_shipping_config()
        assert config.name == "Shipping"
        assert config.preset == ConfigurationPreset.SHIPPING
        assert config.build_type == BuildType.DISTRIBUTION
        assert "NDEBUG" in config.compiler.defines
        assert "SHIPPING" in config.compiler.defines

    def test_create_test_config(self):
        """Test creating test configuration."""
        config = create_test_config()
        assert config.name == "Test"
        assert config.preset == ConfigurationPreset.TEST
        assert "TEST" in config.compiler.defines
        assert "UNIT_TESTING" in config.compiler.defines
