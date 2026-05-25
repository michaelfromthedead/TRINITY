"""Build & Cook subsystem for AI Game Engine.

This module provides build orchestration, asset cooking, and packaging
functionality for game development workflows.

Components:
- BuildPipeline: Orchestrates build stages with dependencies and parallelism
- CookSystem: Platform-specific asset processing and optimization
- Packaging: Game packaging with compression, encryption, DLC support
- BuildConfig: Build configurations (Debug, Development, Shipping)
- BuildTargets: Platform targets (Windows, Linux, Mac, Console, Mobile)
- BuildCache: Incremental builds with content hash caching
- BuildReport: Build reports with timing, warnings, errors
"""
from __future__ import annotations

from .build_config import (
    BuildConfiguration,
    BuildType,
    ConfigurationPreset,
    OptimizationLevel,
    DebugLevel,
    create_debug_config,
    create_development_config,
    create_shipping_config,
    create_test_config,
)
from .build_targets import (
    Platform,
    Architecture,
    PlatformTarget,
    PlatformCapabilities,
    WindowsTarget,
    LinuxTarget,
    MacTarget,
    AndroidTarget,
    iOSTarget,
    PS5Target,
    XboxSeriesTarget,
    SwitchTarget,
    get_current_platform,
    get_supported_platforms,
)
from .build_pipeline import (
    BuildStage,
    BuildStageStatus,
    BuildDependency,
    BuildGraph,
    BuildExecutor,
    BuildPipeline,
    ParallelBuildExecutor,
)
from .cook_system import (
    AssetCookState,
    CookResult,
    AssetCooker,
    TextureCooker,
    MeshCooker,
    AudioCooker,
    ShaderCooker,
    CookPipeline,
    CookRegistry,
)
from .packaging import (
    CompressionMethod,
    EncryptionMethod,
    PackageType,
    DLCInfo,
    PackageManifest,
    PackageBuilder,
    DLCManager,
    PackageEncryption,
)
from .build_cache import (
    ContentHash,
    CacheEntry,
    BuildCacheBackend,
    FilesystemCache,
    BuildCache,
    IncrementalBuilder,
)
from .build_report import (
    BuildSeverity,
    BuildMessage,
    BuildTiming,
    BuildStatistics,
    BuildReport,
    ReportFormatter,
    HTMLReportFormatter,
    JSONReportFormatter,
)

__all__ = [
    # Build Config
    "BuildConfiguration",
    "BuildType",
    "ConfigurationPreset",
    "OptimizationLevel",
    "DebugLevel",
    "create_debug_config",
    "create_development_config",
    "create_shipping_config",
    "create_test_config",
    # Build Targets
    "Platform",
    "Architecture",
    "PlatformTarget",
    "PlatformCapabilities",
    "WindowsTarget",
    "LinuxTarget",
    "MacTarget",
    "AndroidTarget",
    "iOSTarget",
    "PS5Target",
    "XboxSeriesTarget",
    "SwitchTarget",
    "get_current_platform",
    "get_supported_platforms",
    # Build Pipeline
    "BuildStage",
    "BuildStageStatus",
    "BuildDependency",
    "BuildGraph",
    "BuildExecutor",
    "BuildPipeline",
    "ParallelBuildExecutor",
    # Cook System
    "AssetCookState",
    "CookResult",
    "AssetCooker",
    "TextureCooker",
    "MeshCooker",
    "AudioCooker",
    "ShaderCooker",
    "CookPipeline",
    "CookRegistry",
    # Packaging
    "CompressionMethod",
    "EncryptionMethod",
    "PackageType",
    "DLCInfo",
    "PackageManifest",
    "PackageBuilder",
    "DLCManager",
    "PackageEncryption",
    # Build Cache
    "ContentHash",
    "CacheEntry",
    "BuildCacheBackend",
    "FilesystemCache",
    "BuildCache",
    "IncrementalBuilder",
    # Build Report
    "BuildSeverity",
    "BuildMessage",
    "BuildTiming",
    "BuildStatistics",
    "BuildReport",
    "ReportFormatter",
    "HTMLReportFormatter",
    "JSONReportFormatter",
]
