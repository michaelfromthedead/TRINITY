"""Build configuration management.

Provides build configuration types, presets, and settings management
for different build scenarios (Debug, Development, Shipping, Test).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set
import copy


class BuildType(Enum):
    """Type of build to perform."""
    FULL = auto()           # Complete rebuild from scratch
    INCREMENTAL = auto()    # Only rebuild changed files
    DISTRIBUTION = auto()   # Full build optimized for distribution
    DEBUG_SYMBOLS = auto()  # Build with debug symbols only


class OptimizationLevel(Enum):
    """Compiler optimization level."""
    NONE = 0        # No optimization (-O0)
    MINIMAL = 1     # Minimal optimization (-O1)
    STANDARD = 2    # Standard optimization (-O2)
    AGGRESSIVE = 3  # Aggressive optimization (-O3)
    SIZE = 4        # Optimize for size (-Os)
    SPEED = 5       # Optimize for speed (-Ofast)


class DebugLevel(Enum):
    """Debug information level."""
    NONE = auto()           # No debug info
    MINIMAL = auto()        # Minimal debug info
    STANDARD = auto()       # Standard debug info
    FULL = auto()           # Full debug info with all symbols
    PROFILING = auto()      # Debug info optimized for profiling


class ConfigurationPreset(Enum):
    """Predefined build configuration presets."""
    DEBUG = auto()
    DEVELOPMENT = auto()
    SHIPPING = auto()
    TEST = auto()
    PROFILE = auto()
    DEMO = auto()


@dataclass
class CompilerSettings:
    """Compiler-specific settings."""
    optimization: OptimizationLevel = OptimizationLevel.STANDARD
    debug_level: DebugLevel = DebugLevel.STANDARD
    defines: Dict[str, str] = field(default_factory=dict)
    include_paths: List[str] = field(default_factory=list)
    library_paths: List[str] = field(default_factory=list)
    libraries: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    warnings_as_errors: bool = False
    enable_exceptions: bool = True
    enable_rtti: bool = True

    def merge_with(self, other: CompilerSettings) -> CompilerSettings:
        """Merge with another settings object, other takes precedence."""
        merged = copy.deepcopy(self)
        merged.optimization = other.optimization
        merged.debug_level = other.debug_level
        merged.defines.update(other.defines)
        merged.include_paths.extend(other.include_paths)
        merged.library_paths.extend(other.library_paths)
        merged.libraries.extend(other.libraries)
        merged.flags.extend(other.flags)
        merged.warnings_as_errors = other.warnings_as_errors
        merged.enable_exceptions = other.enable_exceptions
        merged.enable_rtti = other.enable_rtti
        return merged


@dataclass
class LinkerSettings:
    """Linker-specific settings."""
    static_linking: bool = False
    strip_symbols: bool = False
    link_time_optimization: bool = False
    dead_code_elimination: bool = True
    library_search_paths: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def merge_with(self, other: LinkerSettings) -> LinkerSettings:
        """Merge with another settings object."""
        merged = copy.deepcopy(self)
        merged.static_linking = other.static_linking
        merged.strip_symbols = other.strip_symbols
        merged.link_time_optimization = other.link_time_optimization
        merged.dead_code_elimination = other.dead_code_elimination
        merged.library_search_paths.extend(other.library_search_paths)
        merged.frameworks.extend(other.frameworks)
        merged.flags.extend(other.flags)
        return merged


@dataclass
class AssetSettings:
    """Asset processing settings."""
    compress_textures: bool = True
    texture_format: str = "DXT"
    max_texture_size: int = 4096
    compress_meshes: bool = True
    compress_audio: bool = True
    audio_quality: float = 0.8
    strip_editor_data: bool = False
    generate_mipmaps: bool = True

    def merge_with(self, other: AssetSettings) -> AssetSettings:
        """Merge with another settings object."""
        merged = copy.deepcopy(self)
        merged.compress_textures = other.compress_textures
        merged.texture_format = other.texture_format
        merged.max_texture_size = other.max_texture_size
        merged.compress_meshes = other.compress_meshes
        merged.compress_audio = other.compress_audio
        merged.audio_quality = other.audio_quality
        merged.strip_editor_data = other.strip_editor_data
        merged.generate_mipmaps = other.generate_mipmaps
        return merged


@dataclass
class BuildConfiguration:
    """Complete build configuration."""
    name: str
    preset: ConfigurationPreset
    build_type: BuildType = BuildType.INCREMENTAL
    compiler: CompilerSettings = field(default_factory=CompilerSettings)
    linker: LinkerSettings = field(default_factory=LinkerSettings)
    assets: AssetSettings = field(default_factory=AssetSettings)
    output_dir: str = "Build"
    intermediate_dir: str = "Intermediate"
    enable_logging: bool = True
    enable_assertions: bool = True
    enable_profiling: bool = False
    enable_cheats: bool = False
    enable_developer_tools: bool = True
    package_debug_files: bool = False
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if not self.name:
            issues.append("Configuration name is required")

        if self.preset == ConfigurationPreset.SHIPPING:
            if self.enable_cheats:
                issues.append("Cheats should be disabled in Shipping builds")
            if self.compiler.debug_level == DebugLevel.FULL:
                issues.append("Full debug info not recommended for Shipping")
            if self.enable_developer_tools:
                issues.append("Developer tools should be disabled in Shipping")

        if self.preset == ConfigurationPreset.DEBUG:
            if self.compiler.optimization == OptimizationLevel.AGGRESSIVE:
                issues.append("Aggressive optimization may hinder debugging")
            if self.linker.strip_symbols:
                issues.append("Stripping symbols will hinder debugging")

        return issues

    def clone(self, name: Optional[str] = None) -> BuildConfiguration:
        """Create a deep copy of this configuration."""
        cloned = copy.deepcopy(self)
        if name:
            cloned.name = name
        return cloned

    def apply_preset(self, preset: ConfigurationPreset) -> None:
        """Apply settings from a preset."""
        self.preset = preset

        if preset == ConfigurationPreset.DEBUG:
            self.compiler.optimization = OptimizationLevel.NONE
            self.compiler.debug_level = DebugLevel.FULL
            self.linker.strip_symbols = False
            self.enable_assertions = True
            self.enable_developer_tools = True
            self.enable_cheats = True
            self.assets.strip_editor_data = False

        elif preset == ConfigurationPreset.DEVELOPMENT:
            self.compiler.optimization = OptimizationLevel.STANDARD
            self.compiler.debug_level = DebugLevel.STANDARD
            self.linker.strip_symbols = False
            self.enable_assertions = True
            self.enable_developer_tools = True
            self.enable_cheats = True
            self.assets.strip_editor_data = False

        elif preset == ConfigurationPreset.SHIPPING:
            self.compiler.optimization = OptimizationLevel.AGGRESSIVE
            self.compiler.debug_level = DebugLevel.NONE
            self.linker.strip_symbols = True
            self.linker.link_time_optimization = True
            self.enable_assertions = False
            self.enable_developer_tools = False
            self.enable_cheats = False
            self.enable_logging = False
            self.assets.strip_editor_data = True

        elif preset == ConfigurationPreset.TEST:
            self.compiler.optimization = OptimizationLevel.MINIMAL
            self.compiler.debug_level = DebugLevel.STANDARD
            self.enable_assertions = True
            self.enable_developer_tools = True

        elif preset == ConfigurationPreset.PROFILE:
            self.compiler.optimization = OptimizationLevel.STANDARD
            self.compiler.debug_level = DebugLevel.PROFILING
            self.enable_profiling = True
            self.linker.strip_symbols = False

        elif preset == ConfigurationPreset.DEMO:
            self.compiler.optimization = OptimizationLevel.STANDARD
            self.compiler.debug_level = DebugLevel.MINIMAL
            self.enable_developer_tools = False
            self.enable_cheats = False


class ConfigurationManager:
    """Manages build configurations."""

    def __init__(self):
        self._configurations: Dict[str, BuildConfiguration] = {}
        self._active_config: Optional[str] = None

    def register(self, config: BuildConfiguration) -> None:
        """Register a build configuration."""
        self._configurations[config.name] = config

    def unregister(self, name: str) -> bool:
        """Unregister a configuration by name."""
        if name in self._configurations:
            del self._configurations[name]
            if self._active_config == name:
                self._active_config = None
            return True
        return False

    def get(self, name: str) -> Optional[BuildConfiguration]:
        """Get a configuration by name."""
        return self._configurations.get(name)

    def get_all(self) -> Dict[str, BuildConfiguration]:
        """Get all registered configurations."""
        return dict(self._configurations)

    def set_active(self, name: str) -> bool:
        """Set the active configuration."""
        if name in self._configurations:
            self._active_config = name
            return True
        return False

    def get_active(self) -> Optional[BuildConfiguration]:
        """Get the active configuration."""
        if self._active_config:
            return self._configurations.get(self._active_config)
        return None

    def list_names(self) -> List[str]:
        """List all configuration names."""
        return list(self._configurations.keys())


def create_debug_config(name: str = "Debug") -> BuildConfiguration:
    """Create a debug build configuration."""
    config = BuildConfiguration(
        name=name,
        preset=ConfigurationPreset.DEBUG,
        build_type=BuildType.INCREMENTAL,
    )
    config.apply_preset(ConfigurationPreset.DEBUG)
    config.compiler.defines["DEBUG"] = "1"
    config.compiler.defines["_DEBUG"] = "1"
    return config


def create_development_config(name: str = "Development") -> BuildConfiguration:
    """Create a development build configuration."""
    config = BuildConfiguration(
        name=name,
        preset=ConfigurationPreset.DEVELOPMENT,
        build_type=BuildType.INCREMENTAL,
    )
    config.apply_preset(ConfigurationPreset.DEVELOPMENT)
    config.compiler.defines["DEVELOPMENT"] = "1"
    return config


def create_shipping_config(name: str = "Shipping") -> BuildConfiguration:
    """Create a shipping build configuration."""
    config = BuildConfiguration(
        name=name,
        preset=ConfigurationPreset.SHIPPING,
        build_type=BuildType.DISTRIBUTION,
    )
    config.apply_preset(ConfigurationPreset.SHIPPING)
    config.compiler.defines["NDEBUG"] = "1"
    config.compiler.defines["SHIPPING"] = "1"
    return config


def create_test_config(name: str = "Test") -> BuildConfiguration:
    """Create a test build configuration."""
    config = BuildConfiguration(
        name=name,
        preset=ConfigurationPreset.TEST,
        build_type=BuildType.INCREMENTAL,
    )
    config.apply_preset(ConfigurationPreset.TEST)
    config.compiler.defines["TEST"] = "1"
    config.compiler.defines["UNIT_TESTING"] = "1"
    return config
