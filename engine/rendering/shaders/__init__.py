"""Shader management and hot-reload system.

This package provides:
- ShaderReloader: File-based shader hot-reload with dependency cascade
- ShaderDependencyGraph: Tracks #include dependencies between shaders
- PSOHotSwap: Pipeline State Object hot-swapping without render stalls
"""
from engine.rendering.shaders.hot_reload import (
    ShaderReloader,
    ShaderDependencyGraph,
    PSOHotSwap,
    ShaderHotReloadEvent,
    ShaderReloadCallback,
    ShaderCompileResult,
    DependencyNode,
    CascadeResult,
    ReloadStats,
    ShaderReloadError,
    IncludeParseError,
)

__all__ = [
    "ShaderReloader",
    "ShaderDependencyGraph",
    "PSOHotSwap",
    "ShaderHotReloadEvent",
    "ShaderReloadCallback",
    "ShaderCompileResult",
    "DependencyNode",
    "CascadeResult",
    "ReloadStats",
    "ShaderReloadError",
    "IncludeParseError",
]
