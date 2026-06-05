"""Shader hot-reload with dependency cascade (T-CC-3.2).

Provides:
- ShaderReloader: Watches shader source files and triggers recompilation
- ShaderDependencyGraph: Tracks #include dependencies for cascade reloads
- PSOHotSwap: Hot-swaps PSOs without stalling the render pipeline

Supports .wgsl and .glsl shader files with automatic dependency detection.
"""
from __future__ import annotations

import enum
import hashlib
import logging
import os
import re
import threading
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    Union,
)

from engine.core.file_watcher import FileChangeEvent, FileChangeType, FileWatcher
from engine.rendering.materials.constants import (
    HOT_RELOAD_POLL_INTERVAL_SECONDS,
    PSO_CACHE_DEFAULT_MAX_SIZE,
    SHADER_HASH_LENGTH,
)
from engine.rendering.materials.shader_compiler import (
    CompiledShader,
    CompilationError,
    PermutationKey,
    PSOCache,
    PSODescriptor,
    ShaderLanguage,
    ShaderSource,
    ShaderStage,
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

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class ShaderReloadError(Exception):
    """Error during shader hot-reload."""

    def __init__(
        self,
        shader_path: str,
        message: str,
        cause: Optional[Exception] = None,
    ) -> None:
        self.shader_path = shader_path
        self.message = message
        self.cause = cause
        super().__init__(f"Shader reload failed for {shader_path}: {message}")


class IncludeParseError(Exception):
    """Error parsing shader include directives."""

    def __init__(self, path: str, line: int, message: str) -> None:
        self.path = path
        self.line = line
        self.message = message
        super().__init__(f"{path}:{line}: {message}")


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


class ReloadState(enum.Enum):
    """State of a shader during reload."""

    IDLE = 0
    PENDING = 1
    COMPILING = 2
    READY = 3
    FAILED = 4


@dataclass(slots=True)
class DependencyNode:
    """Node in the shader dependency graph.

    Attributes:
        path: Absolute path to the shader file
        includes: Set of paths this shader includes
        included_by: Set of paths that include this shader
        content_hash: Hash of file contents
        last_modified: Last modification timestamp
        is_header: Whether this is a header-only file (no entry point)
    """

    path: str
    includes: Set[str] = field(default_factory=set)
    included_by: Set[str] = field(default_factory=set)
    content_hash: str = ""
    last_modified: float = 0.0
    is_header: bool = False

    def add_include(self, include_path: str) -> None:
        """Add an include dependency."""
        self.includes.add(include_path)

    def add_included_by(self, parent_path: str) -> None:
        """Track that this file is included by parent."""
        self.included_by.add(parent_path)


@dataclass(slots=True)
class ShaderCompileResult:
    """Result of compiling a single shader.

    Attributes:
        path: Shader file path
        success: Whether compilation succeeded
        compiled_shader: The compiled shader (if success)
        error: Error message (if failed)
        compile_time_ms: Time taken to compile
        permutation_key: Permutation used
    """

    path: str
    success: bool
    compiled_shader: Optional[CompiledShader] = None
    error: Optional[str] = None
    compile_time_ms: float = 0.0
    permutation_key: Optional[PermutationKey] = None


@dataclass(slots=True)
class CascadeResult:
    """Result of cascading shader recompilation.

    Attributes:
        root_path: The shader that triggered the cascade
        affected_paths: All paths that were recompiled
        compile_results: Results for each shader
        total_compile_time_ms: Total time for all compilations
        cascade_depth: Maximum depth of the cascade
    """

    root_path: str
    affected_paths: List[str] = field(default_factory=list)
    compile_results: List[ShaderCompileResult] = field(default_factory=list)
    total_compile_time_ms: float = 0.0
    cascade_depth: int = 0

    @property
    def success(self) -> bool:
        """True if all compilations succeeded."""
        return all(r.success for r in self.compile_results)

    @property
    def failed_count(self) -> int:
        """Number of failed compilations."""
        return sum(1 for r in self.compile_results if not r.success)


@dataclass(slots=True)
class ShaderHotReloadEvent:
    """Event emitted when shaders are hot-reloaded.

    Attributes:
        source_path: Path that triggered the reload
        change_type: Type of file change
        cascade_result: Result of cascade recompilation
        affected_materials: Material IDs that need PSO update
        timestamp: When the event occurred
    """

    source_path: str
    change_type: FileChangeType
    cascade_result: CascadeResult
    affected_materials: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def success(self) -> bool:
        """True if reload fully succeeded."""
        return self.cascade_result.success


@dataclass(slots=True)
class ReloadStats:
    """Statistics for shader hot-reload.

    Attributes:
        reloads_triggered: Total reload events
        shaders_recompiled: Total shaders recompiled
        cascade_recompiles: Recompiles due to dependency cascade
        pso_swaps: Number of PSO hot-swaps
        total_compile_time_ms: Total compilation time
        failed_reloads: Failed reload attempts
        average_cascade_depth: Average dependency cascade depth
    """

    reloads_triggered: int = 0
    shaders_recompiled: int = 0
    cascade_recompiles: int = 0
    pso_swaps: int = 0
    total_compile_time_ms: float = 0.0
    failed_reloads: int = 0
    average_cascade_depth: float = 0.0

    def record_cascade(self, result: CascadeResult) -> None:
        """Record stats from a cascade result."""
        self.reloads_triggered += 1
        self.shaders_recompiled += len(result.compile_results)
        self.cascade_recompiles += max(0, len(result.compile_results) - 1)
        self.total_compile_time_ms += result.total_compile_time_ms
        if not result.success:
            self.failed_reloads += 1
        # Update running average of cascade depth
        n = self.reloads_triggered
        self.average_cascade_depth = (
            (self.average_cascade_depth * (n - 1) + result.cascade_depth) / n
        )


ShaderReloadCallback = Callable[[ShaderHotReloadEvent], None]


# -----------------------------------------------------------------------------
# Shader Dependency Graph
# -----------------------------------------------------------------------------


class ShaderDependencyGraph:
    """Tracks #include dependencies between shader files.

    Builds a dependency graph from shader source files, detecting include
    directives in both WGSL and GLSL formats. When a file changes, computes
    all affected shaders that need recompilation.

    Supported include formats:
    - WGSL: #include "path/to/file.wgsl"
    - GLSL: #include "path/to/file.glsl"
    - GLSL: #include <path/to/file.glsl>

    Attributes:
        nodes: All dependency nodes by path
        include_dirs: Search directories for includes
    """

    __slots__ = (
        "_nodes",
        "_include_dirs",
        "_lock",
        "_include_pattern_quotes",
        "_include_pattern_brackets",
    )

    def __init__(self, include_dirs: Optional[List[str]] = None) -> None:
        self._nodes: Dict[str, DependencyNode] = {}
        self._include_dirs: List[str] = include_dirs or []
        self._lock = threading.RLock()
        # Regex for include directives
        # Matches: #include "file.wgsl" or #include "path/to/file.glsl"
        self._include_pattern_quotes = re.compile(
            r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE
        )
        # Matches: #include <file.glsl>
        self._include_pattern_brackets = re.compile(
            r"^\s*#\s*include\s+<([^>]+)>", re.MULTILINE
        )

    def add_include_dir(self, directory: str) -> None:
        """Add a directory to search for includes."""
        abs_dir = os.path.abspath(directory)
        with self._lock:
            if abs_dir not in self._include_dirs:
                self._include_dirs.append(abs_dir)

    def remove_include_dir(self, directory: str) -> bool:
        """Remove an include directory."""
        abs_dir = os.path.abspath(directory)
        with self._lock:
            if abs_dir in self._include_dirs:
                self._include_dirs.remove(abs_dir)
                return True
            return False

    def get_include_dirs(self) -> List[str]:
        """Get all include directories."""
        with self._lock:
            return list(self._include_dirs)

    def register_shader(self, path: str) -> DependencyNode:
        """Register a shader file and parse its dependencies.

        Args:
            path: Path to the shader file

        Returns:
            The created/updated dependency node

        Raises:
            FileNotFoundError: If shader file doesn't exist
            IncludeParseError: If include parsing fails
        """
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Shader file not found: {abs_path}")

        content = Path(abs_path).read_text(encoding="utf-8")
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:SHADER_HASH_LENGTH]
        mtime = os.path.getmtime(abs_path)

        # Detect if this is a header (no main function)
        is_header = "fn main" not in content and "void main" not in content

        with self._lock:
            # Create or update node
            if abs_path in self._nodes:
                node = self._nodes[abs_path]
                # Clear old forward dependencies
                for old_include in node.includes:
                    if old_include in self._nodes:
                        self._nodes[old_include].included_by.discard(abs_path)
                node.includes.clear()
            else:
                node = DependencyNode(path=abs_path)
                self._nodes[abs_path] = node

            node.content_hash = content_hash
            node.last_modified = mtime
            node.is_header = is_header

            # Parse includes
            includes = self._parse_includes(abs_path, content)
            for include_path in includes:
                resolved = self._resolve_include(abs_path, include_path)
                if resolved:
                    node.add_include(resolved)
                    # Create node for included file if not exists
                    if resolved not in self._nodes:
                        self._nodes[resolved] = DependencyNode(
                            path=resolved, is_header=True
                        )
                    self._nodes[resolved].add_included_by(abs_path)

            return node

    def unregister_shader(self, path: str) -> bool:
        """Remove a shader from the dependency graph."""
        abs_path = os.path.abspath(path)
        with self._lock:
            if abs_path not in self._nodes:
                return False

            node = self._nodes[abs_path]
            # Remove from dependents
            for include in node.includes:
                if include in self._nodes:
                    self._nodes[include].included_by.discard(abs_path)
            # Remove from parents
            for parent in node.included_by:
                if parent in self._nodes:
                    self._nodes[parent].includes.discard(abs_path)

            del self._nodes[abs_path]
            return True

    def get_node(self, path: str) -> Optional[DependencyNode]:
        """Get a dependency node by path."""
        abs_path = os.path.abspath(path)
        with self._lock:
            return self._nodes.get(abs_path)

    def get_all_nodes(self) -> List[DependencyNode]:
        """Get all nodes in the graph."""
        with self._lock:
            return list(self._nodes.values())

    def get_dependents(self, path: str) -> List[str]:
        """Get all shaders that include this file (direct only)."""
        abs_path = os.path.abspath(path)
        with self._lock:
            node = self._nodes.get(abs_path)
            if node is None:
                return []
            return list(node.included_by)

    def get_all_dependents(self, path: str) -> List[str]:
        """Get all shaders that depend on this file (transitive).

        Uses BFS to find all files that directly or indirectly include
        the given file. Returns paths in breadth-first order.
        """
        abs_path = os.path.abspath(path)
        with self._lock:
            if abs_path not in self._nodes:
                return []

            visited: Set[str] = set()
            result: List[str] = []
            queue = [abs_path]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)

                node = self._nodes.get(current)
                if node is None:
                    continue

                for parent in node.included_by:
                    if parent not in visited:
                        result.append(parent)
                        queue.append(parent)

            return result

    def get_affected_shaders(self, changed_path: str) -> Tuple[List[str], int]:
        """Get all shaders that need recompilation when a file changes.

        Returns the changed file plus all its transitive dependents.
        Also returns the cascade depth.

        Args:
            changed_path: Path to the changed file

        Returns:
            Tuple of (affected paths, cascade depth)
        """
        abs_path = os.path.abspath(changed_path)
        with self._lock:
            if abs_path not in self._nodes:
                return ([abs_path] if os.path.exists(abs_path) else [], 0)

            affected = [abs_path]
            visited: Set[str] = {abs_path}
            current_level = [abs_path]
            depth = 0

            while current_level:
                next_level: List[str] = []
                for path in current_level:
                    node = self._nodes.get(path)
                    if node is None:
                        continue
                    for parent in node.included_by:
                        if parent not in visited:
                            visited.add(parent)
                            affected.append(parent)
                            next_level.append(parent)
                if next_level:
                    depth += 1
                current_level = next_level

            return (affected, depth)

    def has_circular_dependency(self, path: str) -> bool:
        """Check if adding a shader would create a circular dependency."""
        abs_path = os.path.abspath(path)
        with self._lock:
            node = self._nodes.get(abs_path)
            if node is None:
                return False

            visited: Set[str] = set()
            stack: Set[str] = set()

            def dfs(current: str) -> bool:
                if current in stack:
                    return True
                if current in visited:
                    return False
                visited.add(current)
                stack.add(current)

                node = self._nodes.get(current)
                if node:
                    for include in node.includes:
                        if dfs(include):
                            return True

                stack.remove(current)
                return False

            return dfs(abs_path)

    def _parse_includes(self, file_path: str, content: str) -> List[str]:
        """Parse include directives from shader source."""
        includes: List[str] = []

        # Find quoted includes
        for match in self._include_pattern_quotes.finditer(content):
            includes.append(match.group(1))

        # Find bracket includes
        for match in self._include_pattern_brackets.finditer(content):
            includes.append(match.group(1))

        return includes

    def _resolve_include(self, source_path: str, include_path: str) -> Optional[str]:
        """Resolve an include path to an absolute path.

        Search order:
        1. Relative to source file
        2. Include directories in order
        """
        # Try relative to source
        source_dir = os.path.dirname(source_path)
        relative = os.path.join(source_dir, include_path)
        if os.path.exists(relative):
            return os.path.abspath(relative)

        # Try include directories
        for include_dir in self._include_dirs:
            candidate = os.path.join(include_dir, include_path)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        logger.warning(
            "Could not resolve include '%s' from '%s'", include_path, source_path
        )
        return None

    def rebuild_graph(self) -> int:
        """Re-parse all registered shaders and rebuild dependencies.

        Returns:
            Number of shaders re-parsed
        """
        with self._lock:
            paths = list(self._nodes.keys())

        count = 0
        for path in paths:
            if os.path.exists(path):
                try:
                    self.register_shader(path)
                    count += 1
                except Exception as e:
                    logger.warning("Failed to re-parse shader %s: %s", path, e)
        return count

    def clear(self) -> None:
        """Clear all nodes from the graph."""
        with self._lock:
            self._nodes.clear()

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        with self._lock:
            return len(self._nodes)


# -----------------------------------------------------------------------------
# PSO Hot-Swap
# -----------------------------------------------------------------------------


class PSOHotSwap:
    """Manages hot-swapping of Pipeline State Objects.

    When shaders are recompiled, PSOs using those shaders need to be
    updated. This class provides a double-buffered approach that allows
    new PSOs to be created without blocking the render thread.

    The swap is atomic: all affected PSOs are updated at once between
    frames to maintain visual consistency.

    Attributes:
        pso_cache: The PSO cache to update
        pending_swaps: PSOs waiting to be swapped
    """

    __slots__ = (
        "_pso_cache",
        "_pending_swaps",
        "_shader_to_psos",
        "_lock",
        "_swap_callbacks",
        "_swap_in_progress",
    )

    def __init__(self, pso_cache: Optional[PSOCache] = None) -> None:
        self._pso_cache = pso_cache or PSOCache()
        self._pending_swaps: Dict[str, Any] = {}  # PSO hash -> new PSO
        self._shader_to_psos: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        self._swap_callbacks: List[Callable[[List[str]], None]] = []
        self._swap_in_progress = False

    @property
    def pso_cache(self) -> PSOCache:
        """Get the underlying PSO cache."""
        return self._pso_cache

    def register_pso(
        self,
        descriptor: PSODescriptor,
        pso: Any,
        shader_paths: List[str],
    ) -> None:
        """Register a PSO with the shaders it uses.

        Args:
            descriptor: PSO descriptor for cache key
            pso: The pipeline state object
            shader_paths: Paths to shaders used by this PSO
        """
        pso_hash = descriptor.get_hash()
        with self._lock:
            self._pso_cache.put(descriptor, pso)
            for path in shader_paths:
                abs_path = os.path.abspath(path)
                self._shader_to_psos[abs_path].add(pso_hash)

    def unregister_pso(self, descriptor: PSODescriptor) -> None:
        """Unregister a PSO."""
        pso_hash = descriptor.get_hash()
        with self._lock:
            self._pso_cache.invalidate(descriptor)
            for pso_set in self._shader_to_psos.values():
                pso_set.discard(pso_hash)

    def get_affected_psos(self, shader_path: str) -> Set[str]:
        """Get PSO hashes affected by a shader change."""
        abs_path = os.path.abspath(shader_path)
        with self._lock:
            return self._shader_to_psos.get(abs_path, set()).copy()

    def queue_swap(self, pso_hash: str, new_pso: Any) -> None:
        """Queue a PSO for swap at next opportunity.

        Args:
            pso_hash: Hash of PSO to replace
            new_pso: New PSO to swap in
        """
        with self._lock:
            self._pending_swaps[pso_hash] = new_pso

    def has_pending_swaps(self) -> bool:
        """Check if there are PSOs waiting to be swapped."""
        with self._lock:
            return len(self._pending_swaps) > 0

    def get_pending_count(self) -> int:
        """Get number of pending PSO swaps."""
        with self._lock:
            return len(self._pending_swaps)

    def execute_swaps(self) -> List[str]:
        """Execute all pending PSO swaps.

        This should be called between frames when no rendering is active.
        Returns the list of PSO hashes that were swapped.
        """
        with self._lock:
            if not self._pending_swaps or self._swap_in_progress:
                return []

            self._swap_in_progress = True
            try:
                swapped: List[str] = []
                for pso_hash, new_pso in self._pending_swaps.items():
                    # Create a dummy descriptor for cache update
                    # In practice, we'd need the real descriptor
                    # For now, directly update the cache's internal dict
                    if hasattr(self._pso_cache, "_cache"):
                        self._pso_cache._cache[pso_hash] = new_pso
                        swapped.append(pso_hash)

                self._pending_swaps.clear()

                # Notify callbacks
                for callback in self._swap_callbacks:
                    try:
                        callback(swapped)
                    except Exception as e:
                        logger.warning("PSO swap callback error: %s", e)

                return swapped
            finally:
                self._swap_in_progress = False

    def add_swap_callback(self, callback: Callable[[List[str]], None]) -> None:
        """Register callback for when swaps complete."""
        with self._lock:
            if callback not in self._swap_callbacks:
                self._swap_callbacks.append(callback)

    def remove_swap_callback(self, callback: Callable[[List[str]], None]) -> bool:
        """Remove a swap callback."""
        with self._lock:
            if callback in self._swap_callbacks:
                self._swap_callbacks.remove(callback)
                return True
            return False

    def clear_pending(self) -> int:
        """Clear all pending swaps without executing.

        Returns:
            Number of swaps cleared
        """
        with self._lock:
            count = len(self._pending_swaps)
            self._pending_swaps.clear()
            return count

    def clear(self) -> None:
        """Clear all state."""
        with self._lock:
            self._pending_swaps.clear()
            self._shader_to_psos.clear()
            self._swap_callbacks.clear()


# -----------------------------------------------------------------------------
# Material Binding Protocol
# -----------------------------------------------------------------------------


class MaterialBindingProtocol(Protocol):
    """Protocol for objects that can receive shader rebinding notifications."""

    def on_shader_reloaded(self, shader_path: str, new_shader: CompiledShader) -> None:
        """Called when a shader used by this material is reloaded."""
        ...


# -----------------------------------------------------------------------------
# Shader Reloader
# -----------------------------------------------------------------------------


class ShaderReloader:
    """Watches shader source files and triggers recompilation with cascade.

    Integrates file watching, dependency tracking, compilation, PSO update,
    and material rebinding into a single coordinated system.

    Features:
    - Watches .wgsl and .glsl files for changes
    - Tracks #include dependencies
    - Cascades recompilation to dependent shaders
    - Hot-swaps PSOs without rendering stalls
    - Notifies materials of shader updates

    Attributes:
        dependency_graph: Shader dependency graph
        pso_swap: PSO hot-swap manager
        stats: Reload statistics
    """

    __slots__ = (
        "_file_watcher",
        "_dependency_graph",
        "_pso_swap",
        "_compiled_shaders",
        "_material_bindings",
        "_shader_to_materials",
        "_reload_callbacks",
        "_compile_func",
        "_running",
        "_lock",
        "_stats",
        "_watched_dirs",
        "_pending_reloads",
    )

    SUPPORTED_EXTENSIONS = {".wgsl", ".glsl", ".vert", ".frag", ".comp"}

    def __init__(
        self,
        pso_cache: Optional[PSOCache] = None,
        include_dirs: Optional[List[str]] = None,
        compile_func: Optional[
            Callable[[str, Optional[PermutationKey]], CompiledShader]
        ] = None,
    ) -> None:
        self._file_watcher = FileWatcher(
            poll_interval_ms=int(HOT_RELOAD_POLL_INTERVAL_SECONDS * 1000)
        )
        self._dependency_graph = ShaderDependencyGraph(include_dirs)
        self._pso_swap = PSOHotSwap(pso_cache)
        self._compiled_shaders: Dict[str, CompiledShader] = {}
        self._material_bindings: weakref.WeakSet[MaterialBindingProtocol] = (
            weakref.WeakSet()
        )
        self._shader_to_materials: Dict[str, Set[str]] = defaultdict(set)
        self._reload_callbacks: List[ShaderReloadCallback] = []
        self._compile_func = compile_func or self._default_compile
        self._running = False
        self._lock = threading.RLock()
        self._stats = ReloadStats()
        self._watched_dirs: Set[str] = set()
        self._pending_reloads: List[Tuple[str, FileChangeType]] = []

        # Set up file watcher callback
        self._file_watcher.registry.register_global(self._on_file_change)

    @property
    def dependency_graph(self) -> ShaderDependencyGraph:
        """Get the dependency graph."""
        return self._dependency_graph

    @property
    def pso_swap(self) -> PSOHotSwap:
        """Get the PSO hot-swap manager."""
        return self._pso_swap

    @property
    def stats(self) -> ReloadStats:
        """Get reload statistics."""
        return self._stats

    @property
    def is_running(self) -> bool:
        """Whether the reloader is running."""
        return self._running

    def watch_directory(
        self,
        directory: str,
        recursive: bool = True,
    ) -> bool:
        """Add a directory to watch for shader changes.

        Args:
            directory: Directory path
            recursive: Whether to watch subdirectories

        Returns:
            True if watch was added
        """
        abs_dir = os.path.abspath(directory)
        if not os.path.isdir(abs_dir):
            return False

        patterns = {f"*{ext}" for ext in self.SUPPORTED_EXTENSIONS}
        success = self._file_watcher.watch(
            abs_dir, recursive=recursive, patterns=patterns
        )
        if success:
            with self._lock:
                self._watched_dirs.add(abs_dir)
                # Also add as include directory
                self._dependency_graph.add_include_dir(abs_dir)

            # Scan and register existing shaders
            self._scan_directory(abs_dir, recursive)

        return success

    def unwatch_directory(self, directory: str) -> bool:
        """Remove a directory from watching."""
        abs_dir = os.path.abspath(directory)
        success = self._file_watcher.unwatch(abs_dir)
        if success:
            with self._lock:
                self._watched_dirs.discard(abs_dir)
        return success

    def register_shader(self, path: str) -> Optional[DependencyNode]:
        """Manually register a shader file.

        Args:
            path: Path to shader file

        Returns:
            The dependency node, or None on failure
        """
        abs_path = os.path.abspath(path)
        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.warning("Unsupported shader extension: %s", ext)
            return None

        try:
            node = self._dependency_graph.register_shader(abs_path)
            # Also watch the file directly if not in a watched dir
            self._file_watcher.watch(abs_path)
            return node
        except Exception as e:
            logger.error("Failed to register shader %s: %s", abs_path, e)
            return None

    def unregister_shader(self, path: str) -> bool:
        """Unregister a shader from tracking."""
        abs_path = os.path.abspath(path)
        with self._lock:
            self._compiled_shaders.pop(abs_path, None)
            self._shader_to_materials.pop(abs_path, None)
        return self._dependency_graph.unregister_shader(abs_path)

    def compile_shader(
        self,
        path: str,
        permutation: Optional[PermutationKey] = None,
    ) -> ShaderCompileResult:
        """Compile a single shader.

        Args:
            path: Shader file path
            permutation: Optional permutation key

        Returns:
            Compilation result
        """
        abs_path = os.path.abspath(path)
        start_time = time.perf_counter()

        try:
            compiled = self._compile_func(abs_path, permutation)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            with self._lock:
                self._compiled_shaders[abs_path] = compiled

            return ShaderCompileResult(
                path=abs_path,
                success=True,
                compiled_shader=compiled,
                compile_time_ms=elapsed_ms,
                permutation_key=permutation,
            )
        except CompilationError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ShaderCompileResult(
                path=abs_path,
                success=False,
                error=str(e),
                compile_time_ms=elapsed_ms,
                permutation_key=permutation,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ShaderCompileResult(
                path=abs_path,
                success=False,
                error=f"Unexpected error: {e}",
                compile_time_ms=elapsed_ms,
                permutation_key=permutation,
            )

    def reload_shader(self, path: str) -> CascadeResult:
        """Reload a shader and cascade to dependents.

        Args:
            path: Path to shader that changed

        Returns:
            Cascade result with all compilation results
        """
        abs_path = os.path.abspath(path)

        # Re-parse the shader for updated dependencies
        try:
            self._dependency_graph.register_shader(abs_path)
        except Exception as e:
            logger.warning("Failed to re-parse shader %s: %s", abs_path, e)

        # Get all affected shaders
        affected, depth = self._dependency_graph.get_affected_shaders(abs_path)

        result = CascadeResult(
            root_path=abs_path,
            affected_paths=affected,
            cascade_depth=depth,
        )

        start_time = time.perf_counter()

        # Compile each affected shader (deepest first for includes)
        for shader_path in reversed(affected):
            compile_result = self.compile_shader(shader_path)
            result.compile_results.append(compile_result)

        result.total_compile_time_ms = (time.perf_counter() - start_time) * 1000

        # Update stats
        self._stats.record_cascade(result)

        # Queue PSO updates for successful compilations
        if result.success:
            self._queue_pso_updates(result)

        return result

    def bind_material(
        self,
        shader_path: str,
        material_id: str,
        material: Optional[MaterialBindingProtocol] = None,
    ) -> None:
        """Bind a material to a shader for reload notifications.

        Args:
            shader_path: Path to shader
            material_id: Unique material identifier
            material: Optional material object for callbacks
        """
        abs_path = os.path.abspath(shader_path)
        with self._lock:
            self._shader_to_materials[abs_path].add(material_id)
            if material is not None:
                self._material_bindings.add(material)

    def unbind_material(self, shader_path: str, material_id: str) -> bool:
        """Unbind a material from a shader."""
        abs_path = os.path.abspath(shader_path)
        with self._lock:
            if abs_path in self._shader_to_materials:
                self._shader_to_materials[abs_path].discard(material_id)
                return True
            return False

    def get_bound_materials(self, shader_path: str) -> Set[str]:
        """Get material IDs bound to a shader."""
        abs_path = os.path.abspath(shader_path)
        with self._lock:
            return self._shader_to_materials.get(abs_path, set()).copy()

    def get_compiled_shader(self, path: str) -> Optional[CompiledShader]:
        """Get a compiled shader by path."""
        abs_path = os.path.abspath(path)
        with self._lock:
            return self._compiled_shaders.get(abs_path)

    def add_reload_callback(self, callback: ShaderReloadCallback) -> None:
        """Register callback for reload events."""
        with self._lock:
            if callback not in self._reload_callbacks:
                self._reload_callbacks.append(callback)

    def remove_reload_callback(self, callback: ShaderReloadCallback) -> bool:
        """Remove a reload callback."""
        with self._lock:
            if callback in self._reload_callbacks:
                self._reload_callbacks.remove(callback)
                return True
            return False

    def set_compile_function(
        self,
        func: Callable[[str, Optional[PermutationKey]], CompiledShader],
    ) -> None:
        """Set custom compile function."""
        self._compile_func = func

    def start(self) -> None:
        """Start the file watcher."""
        if self._running:
            return
        self._running = True
        self._file_watcher.start()

    def stop(self) -> None:
        """Stop the file watcher."""
        self._running = False
        self._file_watcher.stop()

    def poll(self) -> List[FileChangeEvent]:
        """Manually poll for changes."""
        return self._file_watcher.poll_once()

    def process_pending(self) -> List[CascadeResult]:
        """Process any pending reloads.

        Returns:
            List of cascade results
        """
        with self._lock:
            pending = list(self._pending_reloads)
            self._pending_reloads.clear()

        results: List[CascadeResult] = []
        for path, change_type in pending:
            if change_type == FileChangeType.DELETED:
                self.unregister_shader(path)
            else:
                result = self.reload_shader(path)
                results.append(result)

                # Emit reload event
                affected_materials: List[str] = []
                for affected_path in result.affected_paths:
                    affected_materials.extend(self.get_bound_materials(affected_path))

                event = ShaderHotReloadEvent(
                    source_path=path,
                    change_type=change_type,
                    cascade_result=result,
                    affected_materials=affected_materials,
                )

                self._emit_reload_event(event)

        return results

    def execute_pso_swaps(self) -> int:
        """Execute pending PSO swaps.

        Should be called between frames.

        Returns:
            Number of PSOs swapped
        """
        swapped = self._pso_swap.execute_swaps()
        self._stats.pso_swaps += len(swapped)
        return len(swapped)

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        return {
            "reloads_triggered": self._stats.reloads_triggered,
            "shaders_recompiled": self._stats.shaders_recompiled,
            "cascade_recompiles": self._stats.cascade_recompiles,
            "pso_swaps": self._stats.pso_swaps,
            "total_compile_time_ms": self._stats.total_compile_time_ms,
            "failed_reloads": self._stats.failed_reloads,
            "average_cascade_depth": self._stats.average_cascade_depth,
            "watched_directories": len(self._watched_dirs),
            "tracked_shaders": self._dependency_graph.node_count,
            "pending_pso_swaps": self._pso_swap.get_pending_count(),
        }

    def dispose(self) -> None:
        """Clean up all resources."""
        self.stop()
        self._file_watcher.clear()
        self._dependency_graph.clear()
        self._pso_swap.clear()
        with self._lock:
            self._compiled_shaders.clear()
            self._shader_to_materials.clear()
            self._reload_callbacks.clear()
            self._watched_dirs.clear()
            self._pending_reloads.clear()

    def _on_file_change(self, event: FileChangeEvent) -> None:
        """Handle file change events from file watcher."""
        path = str(event.path)
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return

        with self._lock:
            self._pending_reloads.append((path, event.change_type))

    def _scan_directory(self, directory: str, recursive: bool) -> None:
        """Scan directory and register all shader files."""
        path = Path(directory)
        pattern = "**/*" if recursive else "*"

        for ext in self.SUPPORTED_EXTENSIONS:
            for shader_file in path.glob(f"{pattern}{ext}"):
                try:
                    self._dependency_graph.register_shader(str(shader_file))
                except Exception as e:
                    logger.warning(
                        "Failed to register shader %s: %s", shader_file, e
                    )

    def _default_compile(
        self,
        path: str,
        permutation: Optional[PermutationKey],
    ) -> CompiledShader:
        """Default shader compilation (placeholder).

        Real implementation would invoke actual shader compiler.
        """
        content = Path(path).read_text(encoding="utf-8")
        content_hash = hashlib.sha256(content.encode()).hexdigest()[
            :SHADER_HASH_LENGTH
        ]

        # Detect stage from extension
        ext = os.path.splitext(path)[1].lower()
        stage_map = {
            ".vert": ShaderStage.VERTEX,
            ".frag": ShaderStage.FRAGMENT,
            ".comp": ShaderStage.COMPUTE,
            ".wgsl": ShaderStage.FRAGMENT,  # Default for WGSL
            ".glsl": ShaderStage.FRAGMENT,  # Default for GLSL
        }
        stage = stage_map.get(ext, ShaderStage.FRAGMENT)

        return CompiledShader(
            source_hash=content_hash,
            bytecode=hashlib.sha256(content.encode()).digest(),
            stage=stage,
            entry_point="main",
            permutation_key=permutation or PermutationKey.empty(),
        )

    def _queue_pso_updates(self, result: CascadeResult) -> None:
        """Queue PSO updates for affected shaders."""
        for compile_result in result.compile_results:
            if not compile_result.success:
                continue

            affected_psos = self._pso_swap.get_affected_psos(compile_result.path)
            for pso_hash in affected_psos:
                # In real implementation, we'd create the new PSO here
                # For now, we just queue the compiled shader as placeholder
                self._pso_swap.queue_swap(pso_hash, compile_result.compiled_shader)

    def _emit_reload_event(self, event: ShaderHotReloadEvent) -> None:
        """Emit reload event to callbacks and materials."""
        # Notify registered callbacks
        with self._lock:
            callbacks = list(self._reload_callbacks)

        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning("Reload callback error: %s", e)

        # Notify materials with protocol
        if event.success:
            for compile_result in event.cascade_result.compile_results:
                if compile_result.compiled_shader is None:
                    continue
                for material in list(self._material_bindings):
                    try:
                        material.on_shader_reloaded(
                            compile_result.path, compile_result.compiled_shader
                        )
                    except Exception as e:
                        logger.warning("Material binding callback error: %s", e)
