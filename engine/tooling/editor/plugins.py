"""
Plugins - Plugin manager with hot-loading capability.

Provides:
- Plugin lifecycle management (load, init, enable, disable, unload)
- Plugin dependency resolution
- Extension points for UI, menus, tools, importers
- Hot-reload support via @reloadable decorator
- Plugin manifest parsing and validation
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional, Type

from engine.tooling.editor.app_shell import editor, reloadable


class PluginState(Enum):
    """Plugin lifecycle states."""
    UNLOADED = auto()
    LOADED = auto()
    INITIALIZED = auto()
    ENABLED = auto()
    DISABLED = auto()
    ERROR = auto()


@editor(category="Plugins")
@reloadable()
class PluginDependency:
    """A plugin dependency specification."""
    __slots__ = ("plugin_id", "min_version", "max_version", "optional")

    def __init__(self, plugin_id: str, min_version: Optional[str] = None,
                 max_version: Optional[str] = None, optional: bool = False):
        self.plugin_id = plugin_id
        self.min_version = min_version
        self.max_version = max_version
        self.optional = optional

    def is_satisfied_by(self, version: str) -> bool:
        """Check if a version satisfies this dependency."""
        if self.min_version:
            if not self._compare_versions(version, self.min_version) >= 0:
                return False
        if self.max_version:
            if not self._compare_versions(version, self.max_version) <= 0:
                return False
        return True

    @staticmethod
    def _compare_versions(v1: str, v2: str) -> int:
        """Compare version strings. Returns -1, 0, or 1."""
        def parse_version(v: str) -> list[int]:
            parts = []
            for part in v.split('.'):
                try:
                    parts.append(int(part))
                except ValueError:
                    parts.append(0)
            return parts

        p1 = parse_version(v1)
        p2 = parse_version(v2)

        # Pad to same length
        max_len = max(len(p1), len(p2))
        p1.extend([0] * (max_len - len(p1)))
        p2.extend([0] * (max_len - len(p2)))

        for a, b in zip(p1, p2):
            if a < b:
                return -1
            if a > b:
                return 1
        return 0


@editor(category="Plugins")
@reloadable()
class PluginExtensionPoint:
    """An extension point that plugins can extend."""
    __slots__ = ("id", "name", "description", "interface", "_extensions")

    def __init__(self, id: str, name: str, description: str = "",
                 interface: Optional[Type] = None):
        self.id = id
        self.name = name
        self.description = description
        self.interface = interface
        self._extensions: list[tuple[str, Any]] = []  # (plugin_id, extension)

    def register_extension(self, plugin_id: str, extension: Any) -> bool:
        """Register an extension from a plugin."""
        if self.interface and not isinstance(extension, self.interface):
            return False
        self._extensions.append((plugin_id, extension))
        return True

    def unregister_extensions(self, plugin_id: str) -> int:
        """Unregister all extensions from a plugin. Returns count removed."""
        before = len(self._extensions)
        self._extensions = [(pid, ext) for pid, ext in self._extensions
                           if pid != plugin_id]
        return before - len(self._extensions)

    def get_extensions(self) -> list[Any]:
        """Get all registered extensions."""
        return [ext for _, ext in self._extensions]

    def get_extensions_with_plugin(self) -> list[tuple[str, Any]]:
        """Get extensions with their plugin IDs."""
        return list(self._extensions)


@editor(category="Plugins")
@reloadable()
class PluginManifest:
    """Plugin metadata manifest."""
    __slots__ = ("id", "name", "version", "author", "description",
                 "license", "homepage", "dependencies", "entry_point",
                 "extension_points", "supports_hot_reload", "min_editor_version")

    def __init__(self, id: str, name: str, version: str = "1.0.0", **kwargs):
        self.id = id
        self.name = name
        self.version = version
        self.author = kwargs.get("author", "")
        self.description = kwargs.get("description", "")
        self.license = kwargs.get("license", "")
        self.homepage = kwargs.get("homepage", "")
        self.dependencies = kwargs.get("dependencies", [])
        self.entry_point = kwargs.get("entry_point", "")
        self.extension_points = kwargs.get("extension_points", [])
        self.supports_hot_reload = kwargs.get("supports_hot_reload", False)
        self.min_editor_version = kwargs.get("min_editor_version")

    @classmethod
    def from_dict(cls, data: dict) -> "PluginManifest":
        """Create manifest from dictionary."""
        deps = []
        for dep_data in data.get("dependencies", []):
            if isinstance(dep_data, str):
                deps.append(PluginDependency(dep_data))
            elif isinstance(dep_data, dict):
                deps.append(PluginDependency(
                    dep_data["plugin_id"],
                    dep_data.get("min_version"),
                    dep_data.get("max_version"),
                    dep_data.get("optional", False)
                ))

        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            version=data.get("version", "1.0.0"),
            author=data.get("author", ""),
            description=data.get("description", ""),
            license=data.get("license", ""),
            homepage=data.get("homepage", ""),
            dependencies=deps,
            entry_point=data.get("entry_point", ""),
            extension_points=data.get("extension_points", []),
            supports_hot_reload=data.get("supports_hot_reload", False),
            min_editor_version=data.get("min_editor_version"),
        )

    def to_dict(self) -> dict:
        """Convert manifest to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "license": self.license,
            "homepage": self.homepage,
            "dependencies": [
                {
                    "plugin_id": d.plugin_id,
                    "min_version": d.min_version,
                    "max_version": d.max_version,
                    "optional": d.optional,
                }
                for d in self.dependencies
            ],
            "entry_point": self.entry_point,
            "extension_points": self.extension_points,
            "supports_hot_reload": self.supports_hot_reload,
            "min_editor_version": self.min_editor_version,
        }


@editor(category="Plugins")
@reloadable(preserve=["id", "manifest"])
class Plugin:
    """Base class for editor plugins."""
    __slots__ = ("id", "manifest", "state", "path", "_module", "_instance",
                 "_error", "_registered_extensions", "on_state_changed")

    def __init__(self, manifest: PluginManifest, path: Optional[str] = None):
        self.id = manifest.id
        self.manifest = manifest
        self.state = PluginState.UNLOADED
        self.path = path
        self._module: Any = None
        self._instance: Any = None
        self._error: Optional[str] = None
        self._registered_extensions: list[tuple[str, Any]] = []
        self.on_state_changed: Optional[Callable[[PluginState], None]] = None

    @property
    def name(self) -> str:
        """Get plugin name."""
        return self.manifest.name

    @property
    def version(self) -> str:
        """Get plugin version."""
        return self.manifest.version

    @property
    def error(self) -> Optional[str]:
        """Get error message if in error state."""
        return self._error

    @property
    def is_loaded(self) -> bool:
        """Check if plugin is loaded."""
        return self.state in (PluginState.LOADED, PluginState.INITIALIZED,
                              PluginState.ENABLED, PluginState.DISABLED)

    @property
    def is_enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self.state == PluginState.ENABLED

    def _set_state(self, state: PluginState) -> None:
        """Set state and notify listeners."""
        self.state = state
        if self.on_state_changed:
            self.on_state_changed(state)

    def load(self) -> bool:
        """Load the plugin module. Returns True if successful."""
        if self.state != PluginState.UNLOADED:
            return False

        try:
            if self.path and Path(self.path).exists():
                # Load from file path
                spec = importlib.util.spec_from_file_location(
                    self.manifest.entry_point or self.id,
                    self.path
                )
                if spec and spec.loader:
                    self._module = importlib.util.module_from_spec(spec)
                    sys.modules[self.id] = self._module
                    spec.loader.exec_module(self._module)
            elif self.manifest.entry_point:
                # Load as module
                self._module = importlib.import_module(self.manifest.entry_point)

            self._set_state(PluginState.LOADED)
            return True

        except Exception as e:
            self._error = str(e)
            self._set_state(PluginState.ERROR)
            return False

    def unload(self) -> bool:
        """Unload the plugin module. Returns True if successful."""
        if self.state == PluginState.UNLOADED:
            return True

        if self.state == PluginState.ENABLED:
            self.disable()

        try:
            # Remove from sys.modules
            if self.id in sys.modules:
                del sys.modules[self.id]

            self._module = None
            self._instance = None
            self._registered_extensions.clear()
            self._set_state(PluginState.UNLOADED)
            return True

        except Exception as e:
            self._error = str(e)
            self._set_state(PluginState.ERROR)
            return False

    def initialize(self, editor_api: Any = None) -> bool:
        """Initialize the plugin. Returns True if successful."""
        if self.state != PluginState.LOADED:
            return False

        try:
            # Look for plugin class or initialize function
            if self._module:
                if hasattr(self._module, "Plugin"):
                    plugin_cls = self._module.Plugin
                    self._instance = plugin_cls()
                elif hasattr(self._module, "initialize"):
                    self._module.initialize(editor_api)

                if self._instance and hasattr(self._instance, "initialize"):
                    self._instance.initialize(editor_api)

            self._set_state(PluginState.INITIALIZED)
            return True

        except Exception as e:
            self._error = str(e)
            self._set_state(PluginState.ERROR)
            return False

    def enable(self) -> bool:
        """Enable the plugin. Returns True if successful."""
        if self.state not in (PluginState.INITIALIZED, PluginState.DISABLED):
            return False

        try:
            if self._instance and hasattr(self._instance, "enable"):
                self._instance.enable()
            elif self._module and hasattr(self._module, "enable"):
                self._module.enable()

            self._set_state(PluginState.ENABLED)
            return True

        except Exception as e:
            self._error = str(e)
            self._set_state(PluginState.ERROR)
            return False

    def disable(self) -> bool:
        """Disable the plugin. Returns True if successful."""
        if self.state != PluginState.ENABLED:
            return False

        try:
            if self._instance and hasattr(self._instance, "disable"):
                self._instance.disable()
            elif self._module and hasattr(self._module, "disable"):
                self._module.disable()

            self._set_state(PluginState.DISABLED)
            return True

        except Exception as e:
            self._error = str(e)
            self._set_state(PluginState.ERROR)
            return False

    def reload(self) -> bool:
        """Hot-reload the plugin. Returns True if successful."""
        if not self.manifest.supports_hot_reload:
            return False

        if self.state == PluginState.UNLOADED:
            return False

        was_enabled = self.state == PluginState.ENABLED

        try:
            # Preserve state if plugin supports it
            preserved_state = None
            if self._instance and hasattr(self._instance, "get_state_for_reload"):
                preserved_state = self._instance.get_state_for_reload()

            # Disable and unload
            if was_enabled:
                self.disable()
            self.unload()

            # Reload
            if not self.load():
                return False

            if not self.initialize():
                return False

            # Restore state
            if preserved_state and self._instance:
                if hasattr(self._instance, "restore_state_after_reload"):
                    self._instance.restore_state_after_reload(preserved_state)

            # Re-enable if was enabled
            if was_enabled:
                if not self.enable():
                    return False

            return True

        except Exception as e:
            self._error = str(e)
            self._set_state(PluginState.ERROR)
            return False


@editor(category="Plugins")
@reloadable(preserve=["_plugins", "_extension_points"])
class PluginManager:
    """Manages editor plugins."""
    __slots__ = ("_plugins", "_extension_points", "_plugin_paths",
                 "_editor_api", "_editor_version", "on_plugin_loaded",
                 "on_plugin_unloaded", "on_plugin_enabled",
                 "on_plugin_disabled", "on_plugin_error")

    def __init__(self, editor_version: str = "1.0.0"):
        self._plugins: dict[str, Plugin] = {}
        self._extension_points: dict[str, PluginExtensionPoint] = {}
        self._plugin_paths: list[str] = []
        self._editor_api: Any = None
        self._editor_version = editor_version

        # Callbacks
        self.on_plugin_loaded: Optional[Callable[[Plugin], None]] = None
        self.on_plugin_unloaded: Optional[Callable[[Plugin], None]] = None
        self.on_plugin_enabled: Optional[Callable[[Plugin], None]] = None
        self.on_plugin_disabled: Optional[Callable[[Plugin], None]] = None
        self.on_plugin_error: Optional[Callable[[Plugin, str], None]] = None

        # Register default extension points
        self._create_default_extension_points()

    def _create_default_extension_points(self) -> None:
        """Create default extension points."""
        points = [
            PluginExtensionPoint("menu_items", "Menu Items",
                                 "Add items to editor menus"),
            PluginExtensionPoint("toolbar_buttons", "Toolbar Buttons",
                                 "Add buttons to editor toolbars"),
            PluginExtensionPoint("panels", "Panels",
                                 "Add dockable panels to the editor"),
            PluginExtensionPoint("importers", "Asset Importers",
                                 "Import new asset types"),
            PluginExtensionPoint("exporters", "Asset Exporters",
                                 "Export to new formats"),
            PluginExtensionPoint("tools", "Editor Tools",
                                 "Add new editor tools"),
            PluginExtensionPoint("modes", "Editor Modes",
                                 "Add new editor modes"),
            PluginExtensionPoint("commands", "Commands",
                                 "Add new editor commands"),
            PluginExtensionPoint("preferences", "Preferences",
                                 "Add preferences pages"),
        ]
        for point in points:
            self._extension_points[point.id] = point

    def set_editor_api(self, api: Any) -> None:
        """Set the editor API for plugin access."""
        self._editor_api = api

    def add_plugin_path(self, path: str) -> None:
        """Add a path to search for plugins."""
        if path not in self._plugin_paths:
            self._plugin_paths.append(path)

    @property
    def plugins(self) -> list[Plugin]:
        """Get all registered plugins."""
        return list(self._plugins.values())

    @property
    def enabled_plugins(self) -> list[Plugin]:
        """Get all enabled plugins."""
        return [p for p in self._plugins.values() if p.is_enabled]

    @property
    def extension_points(self) -> list[PluginExtensionPoint]:
        """Get all extension points."""
        return list(self._extension_points.values())

    def register_extension_point(self, point: PluginExtensionPoint) -> None:
        """Register an extension point."""
        self._extension_points[point.id] = point

    def get_extension_point(self, point_id: str) -> Optional[PluginExtensionPoint]:
        """Get an extension point by ID."""
        return self._extension_points.get(point_id)

    def register_plugin(self, plugin: Plugin) -> bool:
        """Register a plugin. Returns True if successful."""
        if plugin.id in self._plugins:
            return False

        # Check editor version compatibility
        if plugin.manifest.min_editor_version:
            if not self._is_version_compatible(plugin.manifest.min_editor_version):
                return False

        self._plugins[plugin.id] = plugin

        # Wire up state change callback
        def on_state_change(state: PluginState) -> None:
            if state == PluginState.ERROR and self.on_plugin_error:
                self.on_plugin_error(plugin, plugin.error or "Unknown error")

        plugin.on_state_changed = on_state_change

        return True

    def unregister_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Unregister a plugin."""
        plugin = self._plugins.pop(plugin_id, None)
        if plugin:
            # Unregister all extensions
            for point in self._extension_points.values():
                point.unregister_extensions(plugin_id)
        return plugin

    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get a plugin by ID."""
        return self._plugins.get(plugin_id)

    def _is_version_compatible(self, min_version: str) -> bool:
        """Check if editor version is compatible."""
        return PluginDependency._compare_versions(
            self._editor_version, min_version
        ) >= 0

    def _check_dependencies(self, plugin: Plugin) -> tuple[bool, list[str]]:
        """Check if plugin dependencies are satisfied."""
        missing = []
        for dep in plugin.manifest.dependencies:
            other = self._plugins.get(dep.plugin_id)
            if other is None:
                if not dep.optional:
                    missing.append(f"Plugin '{dep.plugin_id}' not found")
            elif not dep.is_satisfied_by(other.version):
                missing.append(
                    f"Plugin '{dep.plugin_id}' version {other.version} "
                    f"does not satisfy {dep.min_version}-{dep.max_version}"
                )
        return len(missing) == 0, missing

    def _get_load_order(self) -> list[str]:
        """Get plugins in dependency-sorted order."""
        # Topological sort
        visited: set[str] = set()
        order: list[str] = []

        def visit(plugin_id: str) -> None:
            if plugin_id in visited:
                return
            visited.add(plugin_id)

            plugin = self._plugins.get(plugin_id)
            if plugin:
                for dep in plugin.manifest.dependencies:
                    if dep.plugin_id in self._plugins:
                        visit(dep.plugin_id)
                order.append(plugin_id)

        for plugin_id in self._plugins:
            visit(plugin_id)

        return order

    def load_plugin(self, plugin_id: str) -> bool:
        """Load a plugin. Returns True if successful."""
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return False

        # Check dependencies
        satisfied, missing = self._check_dependencies(plugin)
        if not satisfied:
            plugin._error = "; ".join(missing)
            plugin._set_state(PluginState.ERROR)
            return False

        if plugin.load():
            if self.on_plugin_loaded:
                self.on_plugin_loaded(plugin)
            return True
        return False

    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin. Returns True if successful."""
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return False

        # Unregister extensions first
        for point in self._extension_points.values():
            point.unregister_extensions(plugin_id)

        if plugin.unload():
            if self.on_plugin_unloaded:
                self.on_plugin_unloaded(plugin)
            return True
        return False

    def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin. Returns True if successful."""
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return False

        # Load and initialize if needed
        if plugin.state == PluginState.UNLOADED:
            if not self.load_plugin(plugin_id):
                return False

        if plugin.state == PluginState.LOADED:
            if not plugin.initialize(self._editor_api):
                return False

        if plugin.enable():
            if self.on_plugin_enabled:
                self.on_plugin_enabled(plugin)
            return True
        return False

    def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin. Returns True if successful."""
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return False

        # Unregister extensions
        for point in self._extension_points.values():
            point.unregister_extensions(plugin_id)

        if plugin.disable():
            if self.on_plugin_disabled:
                self.on_plugin_disabled(plugin)
            return True
        return False

    def reload_plugin(self, plugin_id: str) -> bool:
        """Hot-reload a plugin. Returns True if successful."""
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return False

        return plugin.reload()

    def load_all(self) -> int:
        """Load all plugins in dependency order. Returns count loaded."""
        count = 0
        for plugin_id in self._get_load_order():
            if self.load_plugin(plugin_id):
                count += 1
        return count

    def enable_all(self) -> int:
        """Enable all loaded plugins. Returns count enabled."""
        count = 0
        for plugin_id in self._get_load_order():
            if self.enable_plugin(plugin_id):
                count += 1
        return count

    def disable_all(self) -> int:
        """Disable all plugins. Returns count disabled."""
        count = 0
        for plugin in self._plugins.values():
            if plugin.is_enabled:
                if plugin.disable():
                    count += 1
        return count

    def unload_all(self) -> int:
        """Unload all plugins. Returns count unloaded."""
        count = 0
        # Unload in reverse dependency order
        for plugin_id in reversed(self._get_load_order()):
            if self.unload_plugin(plugin_id):
                count += 1
        return count

    def discover_plugins(self, path: Optional[str] = None) -> list[PluginManifest]:
        """
        Discover plugins in paths. Returns list of manifests found.

        Looks for plugin.json manifest files in plugin directories.
        """
        import json

        manifests = []
        paths = [path] if path else self._plugin_paths

        for search_path in paths:
            p = Path(search_path)
            if not p.exists():
                continue

            # Look for plugin directories
            for item in p.iterdir():
                if item.is_dir():
                    manifest_file = item / "plugin.json"
                    if manifest_file.exists():
                        try:
                            with open(manifest_file) as f:
                                data = json.load(f)
                            manifest = PluginManifest.from_dict(data)
                            manifests.append(manifest)
                        except Exception:
                            pass

        return manifests

    def register_extension(self, plugin_id: str, point_id: str,
                          extension: Any) -> bool:
        """Register an extension from a plugin."""
        plugin = self._plugins.get(plugin_id)
        point = self._extension_points.get(point_id)

        if not plugin or not point:
            return False

        if point.register_extension(plugin_id, extension):
            plugin._registered_extensions.append((point_id, extension))
            return True
        return False

    def get_extensions(self, point_id: str) -> list[Any]:
        """Get all extensions for an extension point."""
        point = self._extension_points.get(point_id)
        return point.get_extensions() if point else []
