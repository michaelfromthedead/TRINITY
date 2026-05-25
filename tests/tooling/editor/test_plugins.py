"""
Comprehensive tests for the Plugin system.

Tests cover:
- Plugin lifecycle (load, init, enable, disable, unload)
- Plugin dependencies and version checking
- Extension points and extension registration
- Plugin discovery
- Hot-reload functionality
- Plugin manager operations
"""
import pytest
import sys
import tempfile
import os
import json
from pathlib import Path

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.editor.plugins import (
    PluginManager,
    Plugin,
    PluginManifest,
    PluginState,
    PluginDependency,
    PluginExtensionPoint,
)


class TestPluginDependency:
    """Tests for PluginDependency class."""

    def test_dependency_creation(self):
        """PluginDependency should be created with defaults."""
        dep = PluginDependency("other-plugin")
        assert dep.plugin_id == "other-plugin"
        assert dep.min_version is None
        assert dep.max_version is None
        assert dep.optional is False

    def test_dependency_with_versions(self):
        """PluginDependency can have version constraints."""
        dep = PluginDependency("other-plugin", min_version="1.0.0", max_version="2.0.0")
        assert dep.min_version == "1.0.0"
        assert dep.max_version == "2.0.0"

    def test_dependency_optional(self):
        """PluginDependency can be optional."""
        dep = PluginDependency("other-plugin", optional=True)
        assert dep.optional is True

    def test_dependency_satisfied_no_constraints(self):
        """Dependency without constraints is always satisfied."""
        dep = PluginDependency("other-plugin")
        assert dep.is_satisfied_by("1.0.0") is True
        assert dep.is_satisfied_by("5.0.0") is True
        assert dep.is_satisfied_by("0.0.1") is True

    def test_dependency_satisfied_min_version(self):
        """Dependency checks minimum version."""
        dep = PluginDependency("other-plugin", min_version="2.0.0")
        assert dep.is_satisfied_by("2.0.0") is True
        assert dep.is_satisfied_by("2.5.0") is True
        assert dep.is_satisfied_by("3.0.0") is True
        assert dep.is_satisfied_by("1.9.9") is False
        assert dep.is_satisfied_by("1.0.0") is False

    def test_dependency_satisfied_max_version(self):
        """Dependency checks maximum version."""
        dep = PluginDependency("other-plugin", max_version="3.0.0")
        assert dep.is_satisfied_by("2.0.0") is True
        assert dep.is_satisfied_by("3.0.0") is True
        assert dep.is_satisfied_by("3.0.1") is False
        assert dep.is_satisfied_by("4.0.0") is False

    def test_dependency_satisfied_version_range(self):
        """Dependency checks version range."""
        dep = PluginDependency("other-plugin", min_version="1.5.0", max_version="2.5.0")
        assert dep.is_satisfied_by("1.5.0") is True
        assert dep.is_satisfied_by("2.0.0") is True
        assert dep.is_satisfied_by("2.5.0") is True
        assert dep.is_satisfied_by("1.4.9") is False
        assert dep.is_satisfied_by("2.5.1") is False

    def test_version_comparison_equal(self):
        """Version comparison handles equal versions."""
        result = PluginDependency._compare_versions("1.0.0", "1.0.0")
        assert result == 0

    def test_version_comparison_less(self):
        """Version comparison handles less than."""
        assert PluginDependency._compare_versions("1.0.0", "2.0.0") == -1
        assert PluginDependency._compare_versions("1.0.0", "1.1.0") == -1
        assert PluginDependency._compare_versions("1.0.0", "1.0.1") == -1

    def test_version_comparison_greater(self):
        """Version comparison handles greater than."""
        assert PluginDependency._compare_versions("2.0.0", "1.0.0") == 1
        assert PluginDependency._compare_versions("1.1.0", "1.0.0") == 1
        assert PluginDependency._compare_versions("1.0.1", "1.0.0") == 1

    def test_version_comparison_different_length(self):
        """Version comparison handles different lengths."""
        assert PluginDependency._compare_versions("1.0", "1.0.0") == 0
        assert PluginDependency._compare_versions("1.0.0.1", "1.0.0") == 1
        assert PluginDependency._compare_versions("1.0", "1.0.1") == -1


class TestPluginExtensionPoint:
    """Tests for PluginExtensionPoint class."""

    def test_extension_point_creation(self):
        """PluginExtensionPoint should be created properly."""
        point = PluginExtensionPoint("tools", "Editor Tools", "Add tools")
        assert point.id == "tools"
        assert point.name == "Editor Tools"
        assert point.description == "Add tools"

    def test_extension_point_with_interface(self):
        """PluginExtensionPoint can specify interface."""
        class IToolInterface:
            pass

        point = PluginExtensionPoint("tools", "Tools", interface=IToolInterface)
        assert point.interface == IToolInterface

    def test_register_extension(self):
        """Extensions can be registered."""
        point = PluginExtensionPoint("tools", "Tools")

        class MyTool:
            pass

        tool = MyTool()
        assert point.register_extension("my-plugin", tool) is True
        assert tool in point.get_extensions()

    def test_register_extension_type_check(self):
        """Extension registration checks interface type."""
        class IToolInterface:
            pass

        point = PluginExtensionPoint("tools", "Tools", interface=IToolInterface)

        class WrongType:
            pass

        wrong = WrongType()
        assert point.register_extension("my-plugin", wrong) is False

    def test_register_extension_interface_satisfied(self):
        """Extension registration accepts correct interface."""
        class IToolInterface:
            pass

        class MyTool(IToolInterface):
            pass

        point = PluginExtensionPoint("tools", "Tools", interface=IToolInterface)
        tool = MyTool()
        assert point.register_extension("my-plugin", tool) is True

    def test_unregister_extensions(self):
        """Extensions can be unregistered by plugin ID."""
        point = PluginExtensionPoint("tools", "Tools")

        point.register_extension("plugin1", "ext1")
        point.register_extension("plugin1", "ext2")
        point.register_extension("plugin2", "ext3")

        count = point.unregister_extensions("plugin1")
        assert count == 2
        assert len(point.get_extensions()) == 1
        assert "ext3" in point.get_extensions()

    def test_get_extensions(self):
        """Can get all extensions."""
        point = PluginExtensionPoint("tools", "Tools")

        point.register_extension("p1", "ext1")
        point.register_extension("p2", "ext2")

        extensions = point.get_extensions()
        assert len(extensions) == 2
        assert "ext1" in extensions
        assert "ext2" in extensions

    def test_get_extensions_with_plugin(self):
        """Can get extensions with plugin IDs."""
        point = PluginExtensionPoint("tools", "Tools")

        point.register_extension("plugin1", "ext1")
        point.register_extension("plugin2", "ext2")

        ext_with_plugins = point.get_extensions_with_plugin()
        assert len(ext_with_plugins) == 2
        assert ("plugin1", "ext1") in ext_with_plugins
        assert ("plugin2", "ext2") in ext_with_plugins


class TestPluginManifest:
    """Tests for PluginManifest class."""

    def test_manifest_creation(self):
        """PluginManifest should be created with defaults."""
        manifest = PluginManifest("my-plugin", "My Plugin")
        assert manifest.id == "my-plugin"
        assert manifest.name == "My Plugin"
        assert manifest.version == "1.0.0"

    def test_manifest_with_metadata(self):
        """PluginManifest can have full metadata."""
        manifest = PluginManifest(
            "my-plugin", "My Plugin",
            version="2.0.0",
            author="Test Author",
            description="Test plugin",
            license="MIT",
            homepage="https://example.com"
        )
        assert manifest.version == "2.0.0"
        assert manifest.author == "Test Author"
        assert manifest.description == "Test plugin"
        assert manifest.license == "MIT"
        assert manifest.homepage == "https://example.com"

    def test_manifest_with_dependencies(self):
        """PluginManifest can have dependencies."""
        dep = PluginDependency("other-plugin", min_version="1.0.0")
        manifest = PluginManifest("my-plugin", "My Plugin", dependencies=[dep])
        assert len(manifest.dependencies) == 1
        assert manifest.dependencies[0].plugin_id == "other-plugin"

    def test_manifest_from_dict(self):
        """PluginManifest can be created from dict."""
        data = {
            "id": "my-plugin",
            "name": "My Plugin",
            "version": "1.5.0",
            "author": "Test",
            "dependencies": [
                {"plugin_id": "dep1", "min_version": "1.0.0"},
                "dep2"  # String shorthand
            ]
        }

        manifest = PluginManifest.from_dict(data)
        assert manifest.id == "my-plugin"
        assert manifest.version == "1.5.0"
        assert len(manifest.dependencies) == 2
        assert manifest.dependencies[0].min_version == "1.0.0"
        assert manifest.dependencies[1].plugin_id == "dep2"

    def test_manifest_to_dict(self):
        """PluginManifest can be converted to dict."""
        dep = PluginDependency("dep1", min_version="1.0.0")
        manifest = PluginManifest(
            "my-plugin", "My Plugin",
            version="2.0.0",
            dependencies=[dep],
            supports_hot_reload=True
        )

        data = manifest.to_dict()
        assert data["id"] == "my-plugin"
        assert data["version"] == "2.0.0"
        assert len(data["dependencies"]) == 1
        assert data["dependencies"][0]["plugin_id"] == "dep1"
        assert data["supports_hot_reload"] is True

    def test_manifest_extension_points(self):
        """PluginManifest can specify extension points."""
        manifest = PluginManifest(
            "my-plugin", "My Plugin",
            extension_points=["tools", "panels"]
        )
        assert "tools" in manifest.extension_points
        assert "panels" in manifest.extension_points

    def test_manifest_min_editor_version(self):
        """PluginManifest can specify minimum editor version."""
        manifest = PluginManifest(
            "my-plugin", "My Plugin",
            min_editor_version="2.5.0"
        )
        assert manifest.min_editor_version == "2.5.0"


class TestPlugin:
    """Tests for Plugin class."""

    def test_plugin_creation(self):
        """Plugin should be created properly."""
        manifest = PluginManifest("test-plugin", "Test Plugin")
        plugin = Plugin(manifest)

        assert plugin.id == "test-plugin"
        assert plugin.name == "Test Plugin"
        assert plugin.state == PluginState.UNLOADED
        assert plugin.is_loaded is False
        assert plugin.is_enabled is False

    def test_plugin_state_unloaded(self):
        """Plugin starts in UNLOADED state."""
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)

        assert plugin.state == PluginState.UNLOADED
        assert plugin.is_loaded is False

    def test_plugin_state_change_callback(self):
        """Plugin triggers callback on state change."""
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)

        states = []
        plugin.on_state_changed = lambda s: states.append(s)

        plugin._set_state(PluginState.LOADED)
        assert len(states) == 1
        assert states[0] == PluginState.LOADED

    def test_plugin_is_loaded_states(self):
        """is_loaded returns True for loaded states."""
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)

        plugin._set_state(PluginState.LOADED)
        assert plugin.is_loaded is True

        plugin._set_state(PluginState.INITIALIZED)
        assert plugin.is_loaded is True

        plugin._set_state(PluginState.ENABLED)
        assert plugin.is_loaded is True

        plugin._set_state(PluginState.DISABLED)
        assert plugin.is_loaded is True

        plugin._set_state(PluginState.ERROR)
        assert plugin.is_loaded is False

    def test_plugin_is_enabled(self):
        """is_enabled returns True only for ENABLED state."""
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)

        plugin._set_state(PluginState.LOADED)
        assert plugin.is_enabled is False

        plugin._set_state(PluginState.ENABLED)
        assert plugin.is_enabled is True

    def test_plugin_version(self):
        """Plugin version comes from manifest."""
        manifest = PluginManifest("test", "Test", version="3.2.1")
        plugin = Plugin(manifest)
        assert plugin.version == "3.2.1"

    def test_plugin_error(self):
        """Plugin error is accessible."""
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)

        assert plugin.error is None

        plugin._error = "Something went wrong"
        assert plugin.error == "Something went wrong"

    def test_plugin_unload_from_unloaded(self):
        """Unloading already unloaded plugin returns True."""
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)
        assert plugin.unload() is True

    def test_plugin_load_invalid_already_loaded(self):
        """Loading already loaded plugin returns False."""
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)
        plugin._set_state(PluginState.LOADED)
        assert plugin.load() is False

    def test_plugin_initialize_requires_loaded(self):
        """Initialize requires LOADED state."""
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)

        # Not loaded yet
        assert plugin.initialize() is False

        plugin._set_state(PluginState.INITIALIZED)
        assert plugin.initialize() is False  # Already initialized

    def test_plugin_enable_requires_initialized(self):
        """Enable requires INITIALIZED or DISABLED state."""
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)

        plugin._set_state(PluginState.LOADED)
        assert plugin.enable() is False

        plugin._set_state(PluginState.INITIALIZED)
        assert plugin.enable() is True

    def test_plugin_enable_from_disabled(self):
        """Plugin can be re-enabled from DISABLED."""
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)

        plugin._set_state(PluginState.DISABLED)
        assert plugin.enable() is True
        assert plugin.state == PluginState.ENABLED

    def test_plugin_disable_requires_enabled(self):
        """Disable requires ENABLED state."""
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)

        plugin._set_state(PluginState.INITIALIZED)
        assert plugin.disable() is False

        plugin._set_state(PluginState.ENABLED)
        assert plugin.disable() is True

    def test_plugin_reload_requires_hot_reload_support(self):
        """Reload requires supports_hot_reload flag."""
        manifest = PluginManifest("test", "Test", supports_hot_reload=False)
        plugin = Plugin(manifest)

        plugin._set_state(PluginState.ENABLED)
        assert plugin.reload() is False

    def test_plugin_reload_requires_loaded(self):
        """Reload requires plugin to be loaded."""
        manifest = PluginManifest("test", "Test", supports_hot_reload=True)
        plugin = Plugin(manifest)

        # Unloaded
        assert plugin.reload() is False


class TestPluginManager:
    """Tests for PluginManager class."""

    def test_manager_creation(self):
        """PluginManager should be created with defaults."""
        manager = PluginManager()
        assert len(manager.plugins) == 0
        assert len(manager.extension_points) > 0  # Has default points

    def test_manager_default_extension_points(self):
        """PluginManager has default extension points."""
        manager = PluginManager()

        point_ids = [p.id for p in manager.extension_points]
        assert "menu_items" in point_ids
        assert "toolbar_buttons" in point_ids
        assert "panels" in point_ids
        assert "importers" in point_ids
        assert "exporters" in point_ids
        assert "tools" in point_ids
        assert "modes" in point_ids
        assert "commands" in point_ids
        assert "preferences" in point_ids

    def test_manager_register_plugin(self):
        """PluginManager can register plugins."""
        manager = PluginManager()
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)

        assert manager.register_plugin(plugin) is True
        assert manager.get_plugin("test") == plugin
        assert plugin in manager.plugins

    def test_manager_register_duplicate(self):
        """PluginManager rejects duplicate plugins."""
        manager = PluginManager()
        manifest = PluginManifest("test", "Test")
        plugin1 = Plugin(manifest)
        plugin2 = Plugin(manifest)

        assert manager.register_plugin(plugin1) is True
        assert manager.register_plugin(plugin2) is False

    def test_manager_unregister_plugin(self):
        """PluginManager can unregister plugins."""
        manager = PluginManager()
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)

        manager.register_plugin(plugin)
        removed = manager.unregister_plugin("test")

        assert removed == plugin
        assert manager.get_plugin("test") is None

    def test_manager_unregister_clears_extensions(self):
        """Unregistering plugin clears its extensions."""
        manager = PluginManager()
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)
        manager.register_plugin(plugin)

        # Register an extension
        manager.register_extension("test", "tools", "my_tool")

        # Unregister plugin
        manager.unregister_plugin("test")

        # Extension should be gone
        assert len(manager.get_extensions("tools")) == 0

    def test_manager_get_plugin(self):
        """PluginManager can get plugins by ID."""
        manager = PluginManager()
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)
        manager.register_plugin(plugin)

        assert manager.get_plugin("test") == plugin
        assert manager.get_plugin("nonexistent") is None

    def test_manager_enabled_plugins(self):
        """PluginManager tracks enabled plugins."""
        manager = PluginManager()

        m1 = PluginManifest("p1", "P1")
        m2 = PluginManifest("p2", "P2")
        p1 = Plugin(m1)
        p2 = Plugin(m2)

        manager.register_plugin(p1)
        manager.register_plugin(p2)

        p1._set_state(PluginState.ENABLED)
        p2._set_state(PluginState.DISABLED)

        enabled = manager.enabled_plugins
        assert p1 in enabled
        assert p2 not in enabled

    def test_manager_register_extension_point(self):
        """PluginManager can register extension points."""
        manager = PluginManager()
        point = PluginExtensionPoint("custom", "Custom Point")

        manager.register_extension_point(point)
        assert manager.get_extension_point("custom") == point

    def test_manager_get_extension_point(self):
        """PluginManager can get extension points."""
        manager = PluginManager()
        point = manager.get_extension_point("tools")
        assert point is not None
        assert point.id == "tools"

    def test_manager_editor_version_check(self):
        """PluginManager checks editor version compatibility."""
        manager = PluginManager(editor_version="2.0.0")

        manifest = PluginManifest("test", "Test", min_editor_version="3.0.0")
        plugin = Plugin(manifest)

        # Should reject - editor version too low
        assert manager.register_plugin(plugin) is False

    def test_manager_editor_version_compatible(self):
        """PluginManager accepts compatible plugins."""
        manager = PluginManager(editor_version="3.0.0")

        manifest = PluginManifest("test", "Test", min_editor_version="2.0.0")
        plugin = Plugin(manifest)

        assert manager.register_plugin(plugin) is True

    def test_manager_add_plugin_path(self):
        """PluginManager can add plugin search paths."""
        manager = PluginManager()
        manager.add_plugin_path("/path/to/plugins")

        assert "/path/to/plugins" in manager._plugin_paths

    def test_manager_add_plugin_path_no_duplicates(self):
        """Adding same path twice doesn't duplicate."""
        manager = PluginManager()
        manager.add_plugin_path("/path/to/plugins")
        manager.add_plugin_path("/path/to/plugins")

        assert manager._plugin_paths.count("/path/to/plugins") == 1

    def test_manager_register_extension(self):
        """PluginManager can register extensions."""
        manager = PluginManager()
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)
        manager.register_plugin(plugin)

        result = manager.register_extension("test", "tools", "my_tool")
        assert result is True
        assert "my_tool" in manager.get_extensions("tools")

    def test_manager_register_extension_unknown_plugin(self):
        """Cannot register extension for unknown plugin."""
        manager = PluginManager()
        result = manager.register_extension("unknown", "tools", "my_tool")
        assert result is False

    def test_manager_register_extension_unknown_point(self):
        """Cannot register extension for unknown point."""
        manager = PluginManager()
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)
        manager.register_plugin(plugin)

        result = manager.register_extension("test", "unknown_point", "ext")
        assert result is False

    def test_manager_get_extensions(self):
        """PluginManager can get extensions by point."""
        manager = PluginManager()
        m1 = PluginManifest("p1", "P1")
        m2 = PluginManifest("p2", "P2")
        manager.register_plugin(Plugin(m1))
        manager.register_plugin(Plugin(m2))

        manager.register_extension("p1", "tools", "tool1")
        manager.register_extension("p2", "tools", "tool2")

        extensions = manager.get_extensions("tools")
        assert len(extensions) == 2
        assert "tool1" in extensions
        assert "tool2" in extensions

    def test_manager_enable_plugin_auto_loads(self):
        """Enabling unloaded plugin auto-loads it."""
        manager = PluginManager()
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)
        manager.register_plugin(plugin)

        # Plugin is unloaded, enabling should try to load first
        # This will fail without actual module, but shows the flow
        enabled = manager.enable_plugin("test")
        # Will be False because no actual module to load
        # But demonstrates the auto-load attempt

    def test_manager_disable_plugin_clears_extensions(self):
        """Disabling plugin clears its extensions."""
        manager = PluginManager()
        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)
        manager.register_plugin(plugin)

        plugin._set_state(PluginState.ENABLED)
        manager.register_extension("test", "tools", "my_tool")

        manager.disable_plugin("test")
        assert len(manager.get_extensions("tools")) == 0

    def test_manager_callbacks_error(self):
        """PluginManager triggers error callback."""
        manager = PluginManager()
        errors = []
        manager.on_plugin_error = lambda p, e: errors.append((p.id, e))

        manifest = PluginManifest("test", "Test")
        plugin = Plugin(manifest)
        manager.register_plugin(plugin)

        plugin._error = "Test error"
        plugin._set_state(PluginState.ERROR)

        assert len(errors) == 1
        assert errors[0][0] == "test"
        assert "Test error" in errors[0][1]

    def test_manager_load_all(self):
        """PluginManager can load all plugins."""
        manager = PluginManager()

        # Register multiple plugins
        for i in range(3):
            m = PluginManifest(f"p{i}", f"P{i}")
            manager.register_plugin(Plugin(m))

        # load_all will try to load - may fail without modules
        # but tests the function runs
        count = manager.load_all()
        # Count may be 0 if no actual modules

    def test_manager_disable_all(self):
        """PluginManager can disable all plugins."""
        manager = PluginManager()

        m1 = PluginManifest("p1", "P1")
        m2 = PluginManifest("p2", "P2")
        p1 = Plugin(m1)
        p2 = Plugin(m2)

        manager.register_plugin(p1)
        manager.register_plugin(p2)

        p1._set_state(PluginState.ENABLED)
        p2._set_state(PluginState.ENABLED)

        count = manager.disable_all()
        assert count == 2
        assert p1.state == PluginState.DISABLED
        assert p2.state == PluginState.DISABLED

    def test_manager_unload_all(self):
        """PluginManager can unload all plugins."""
        manager = PluginManager()

        m1 = PluginManifest("p1", "P1")
        m2 = PluginManifest("p2", "P2")
        p1 = Plugin(m1)
        p2 = Plugin(m2)

        manager.register_plugin(p1)
        manager.register_plugin(p2)

        p1._set_state(PluginState.LOADED)
        p2._set_state(PluginState.LOADED)

        count = manager.unload_all()
        assert count == 2
        assert p1.state == PluginState.UNLOADED
        assert p2.state == PluginState.UNLOADED

    def test_manager_dependency_order(self):
        """PluginManager loads in dependency order."""
        manager = PluginManager()

        # p2 depends on p1
        m1 = PluginManifest("p1", "P1", version="1.0.0")
        m2 = PluginManifest("p2", "P2", dependencies=[
            PluginDependency("p1", min_version="1.0.0")
        ])

        manager.register_plugin(Plugin(m1))
        manager.register_plugin(Plugin(m2))

        order = manager._get_load_order()
        assert order.index("p1") < order.index("p2")

    def test_manager_check_dependencies_satisfied(self):
        """PluginManager checks dependencies."""
        manager = PluginManager()

        m1 = PluginManifest("p1", "P1", version="2.0.0")
        m2 = PluginManifest("p2", "P2", dependencies=[
            PluginDependency("p1", min_version="1.0.0")
        ])

        p1 = Plugin(m1)
        p2 = Plugin(m2)

        manager.register_plugin(p1)
        manager.register_plugin(p2)

        satisfied, missing = manager._check_dependencies(p2)
        assert satisfied is True
        assert len(missing) == 0

    def test_manager_check_dependencies_missing(self):
        """PluginManager detects missing dependencies."""
        manager = PluginManager()

        m = PluginManifest("p1", "P1", dependencies=[
            PluginDependency("nonexistent")
        ])
        p = Plugin(m)
        manager.register_plugin(p)

        satisfied, missing = manager._check_dependencies(p)
        assert satisfied is False
        assert len(missing) == 1

    def test_manager_check_dependencies_version_mismatch(self):
        """PluginManager detects version mismatch."""
        manager = PluginManager()

        m1 = PluginManifest("p1", "P1", version="1.0.0")
        m2 = PluginManifest("p2", "P2", dependencies=[
            PluginDependency("p1", min_version="2.0.0")
        ])

        manager.register_plugin(Plugin(m1))
        p2 = Plugin(m2)
        manager.register_plugin(p2)

        satisfied, missing = manager._check_dependencies(p2)
        assert satisfied is False
        assert len(missing) == 1

    def test_manager_check_dependencies_optional(self):
        """PluginManager ignores missing optional dependencies."""
        manager = PluginManager()

        m = PluginManifest("p1", "P1", dependencies=[
            PluginDependency("optional-plugin", optional=True)
        ])
        p = Plugin(m)
        manager.register_plugin(p)

        satisfied, missing = manager._check_dependencies(p)
        assert satisfied is True

    def test_manager_set_editor_api(self):
        """PluginManager can set editor API."""
        manager = PluginManager()

        class MockAPI:
            pass

        api = MockAPI()
        manager.set_editor_api(api)
        assert manager._editor_api == api

    def test_manager_discover_plugins(self):
        """PluginManager can discover plugins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a plugin directory with manifest
            plugin_dir = Path(tmpdir) / "my-plugin"
            plugin_dir.mkdir()

            manifest_file = plugin_dir / "plugin.json"
            manifest_data = {
                "id": "my-plugin",
                "name": "My Plugin",
                "version": "1.0.0"
            }
            with open(manifest_file, 'w') as f:
                json.dump(manifest_data, f)

            manager = PluginManager()
            manifests = manager.discover_plugins(tmpdir)

            assert len(manifests) == 1
            assert manifests[0].id == "my-plugin"

    def test_manager_discover_plugins_from_paths(self):
        """PluginManager discovers from registered paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "test-plugin"
            plugin_dir.mkdir()

            manifest_file = plugin_dir / "plugin.json"
            manifest_data = {"id": "test-plugin", "name": "Test"}
            with open(manifest_file, 'w') as f:
                json.dump(manifest_data, f)

            manager = PluginManager()
            manager.add_plugin_path(tmpdir)

            manifests = manager.discover_plugins()
            assert len(manifests) == 1

    def test_manager_reload_plugin(self):
        """PluginManager can reload plugins."""
        manager = PluginManager()
        manifest = PluginManifest("test", "Test", supports_hot_reload=True)
        plugin = Plugin(manifest)
        manager.register_plugin(plugin)

        # Reload requires loaded state
        result = manager.reload_plugin("test")
        assert result is False  # Not loaded

    def test_manager_reload_nonexistent(self):
        """Reloading nonexistent plugin returns False."""
        manager = PluginManager()
        result = manager.reload_plugin("nonexistent")
        assert result is False


class TestPluginState:
    """Tests for PluginState enum."""

    def test_states_exist(self):
        """All plugin states should exist."""
        assert PluginState.UNLOADED is not None
        assert PluginState.LOADED is not None
        assert PluginState.INITIALIZED is not None
        assert PluginState.ENABLED is not None
        assert PluginState.DISABLED is not None
        assert PluginState.ERROR is not None

    def test_states_distinct(self):
        """All states should be distinct."""
        states = list(PluginState)
        assert len(states) == len(set(states))


class TestPluginIntegration:
    """Integration tests for plugin system."""

    def test_full_plugin_lifecycle(self):
        """Test complete plugin lifecycle."""
        manager = PluginManager()

        # Create and register plugin
        manifest = PluginManifest("test", "Test Plugin", version="1.0.0")
        plugin = Plugin(manifest)
        assert manager.register_plugin(plugin) is True

        # Track state changes
        states = []
        plugin.on_state_changed = lambda s: states.append(s)

        # Cannot enable without loading
        plugin._set_state(PluginState.LOADED)
        plugin._set_state(PluginState.INITIALIZED)

        # Enable
        assert plugin.enable() is True
        assert plugin.is_enabled is True
        assert PluginState.ENABLED in states

        # Disable
        assert plugin.disable() is True
        assert plugin.is_enabled is False
        assert PluginState.DISABLED in states

        # Unload
        assert plugin.unload() is True
        assert plugin.state == PluginState.UNLOADED

    def test_dependency_chain(self):
        """Test plugin dependency chain resolution."""
        manager = PluginManager()

        # Create chain: p3 -> p2 -> p1
        m1 = PluginManifest("p1", "P1", version="1.0.0")
        m2 = PluginManifest("p2", "P2", version="1.0.0", dependencies=[
            PluginDependency("p1")
        ])
        m3 = PluginManifest("p3", "P3", version="1.0.0", dependencies=[
            PluginDependency("p2")
        ])

        manager.register_plugin(Plugin(m1))
        manager.register_plugin(Plugin(m2))
        manager.register_plugin(Plugin(m3))

        order = manager._get_load_order()

        # Verify order
        assert order.index("p1") < order.index("p2")
        assert order.index("p2") < order.index("p3")

    def test_extension_point_workflow(self):
        """Test extension point registration and lookup."""
        manager = PluginManager()

        # Register plugin
        manifest = PluginManifest("my-plugin", "My Plugin")
        plugin = Plugin(manifest)
        manager.register_plugin(plugin)
        plugin._set_state(PluginState.ENABLED)

        # Register extensions
        manager.register_extension("my-plugin", "tools", {"name": "Tool1"})
        manager.register_extension("my-plugin", "tools", {"name": "Tool2"})
        manager.register_extension("my-plugin", "panels", {"name": "Panel1"})

        # Check extensions
        tools = manager.get_extensions("tools")
        assert len(tools) == 2

        panels = manager.get_extensions("panels")
        assert len(panels) == 1

        # Disable plugin clears extensions
        manager.disable_plugin("my-plugin")
        assert len(manager.get_extensions("tools")) == 0
        assert len(manager.get_extensions("panels")) == 0

    def test_plugin_error_handling(self):
        """Test plugin error state handling."""
        manager = PluginManager()
        errors = []
        manager.on_plugin_error = lambda p, e: errors.append((p.id, e))

        manifest = PluginManifest("failing", "Failing Plugin")
        plugin = Plugin(manifest)
        manager.register_plugin(plugin)

        # Simulate error
        plugin._error = "Module not found"
        plugin._set_state(PluginState.ERROR)

        assert len(errors) == 1
        assert "failing" in errors[0][0]
        assert "Module not found" in errors[0][1]

    def test_multiple_plugins_extensions(self):
        """Test multiple plugins contributing to same extension point."""
        manager = PluginManager()

        # Register multiple plugins
        for i in range(3):
            m = PluginManifest(f"plugin{i}", f"Plugin {i}")
            p = Plugin(m)
            manager.register_plugin(p)
            p._set_state(PluginState.ENABLED)
            manager.register_extension(f"plugin{i}", "tools", f"tool_{i}")

        # All extensions should be present
        tools = manager.get_extensions("tools")
        assert len(tools) == 3
        for i in range(3):
            assert f"tool_{i}" in tools

    def test_version_compatibility_matrix(self):
        """Test various version compatibility scenarios."""
        manager = PluginManager(editor_version="2.5.0")

        test_cases = [
            ("1.0.0", True),   # Older requirement
            ("2.5.0", True),   # Exact match
            ("2.4.9", True),   # Just under
            ("2.5.1", False),  # Just over
            ("3.0.0", False),  # Major version higher
        ]

        for min_ver, expected in test_cases:
            manifest = PluginManifest(
                f"test-{min_ver}", "Test",
                min_editor_version=min_ver
            )
            plugin = Plugin(manifest)
            result = manager.register_plugin(plugin)
            assert result == expected, f"Failed for min_version={min_ver}"
            if result:
                manager.unregister_plugin(f"test-{min_ver}")

