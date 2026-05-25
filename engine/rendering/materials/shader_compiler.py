"""Shader variant compilation and PSO caching.

This module provides shader compilation infrastructure:
- ShaderSource: HLSL/GLSL/Metal source management
- ShaderPermutation: Static permutations (compile-time variants)
- PSOCache: Pipeline State Object caching
- ShaderCompiler: Compile variants, hot-reload support
- PermutationKey: Variant selection
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from engine.rendering.materials.constants import (
    HOT_RELOAD_POLL_INTERVAL_SECONDS,
    PSO_CACHE_DEFAULT_MAX_SIZE,
    SHADER_HASH_LENGTH,
)

__all__ = [
    "ShaderStage",
    "ShaderLanguage",
    "ShaderSource",
    "ShaderDefine",
    "PermutationKey",
    "ShaderPermutation",
    "CompiledShader",
    "PSODescriptor",
    "PSOCache",
    "ShaderCompiler",
    "CompilationError",
    "HotReloadWatcher",
]


class ShaderStage(Enum):
    """Shader pipeline stages."""
    VERTEX = "vertex"
    FRAGMENT = "fragment"
    COMPUTE = "compute"
    GEOMETRY = "geometry"
    TESSELLATION_CONTROL = "tess_control"
    TESSELLATION_EVAL = "tess_eval"
    MESH = "mesh"
    TASK = "task"
    RAY_GENERATION = "ray_gen"
    RAY_CLOSEST_HIT = "ray_closest"
    RAY_ANY_HIT = "ray_any"
    RAY_MISS = "ray_miss"
    RAY_INTERSECTION = "ray_intersect"


class ShaderLanguage(Enum):
    """Shader source language."""
    HLSL = "hlsl"
    GLSL = "glsl"
    METAL = "metal"
    SPIRV = "spirv"
    WGSL = "wgsl"


class CompilationError(Exception):
    """Raised when shader compilation fails."""

    def __init__(
        self,
        message: str,
        source_file: Optional[str] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.source_file = source_file
        self.line = line
        self.column = column


@dataclass(slots=True, frozen=True)
class ShaderDefine:
    """A preprocessor define for shader compilation.

    Attributes:
        name: Define name
        value: Optional value (if None, just defined without value)
    """
    name: str
    value: Optional[str] = None

    def to_string(self) -> str:
        """Convert to preprocessor string format."""
        if self.value is None:
            return f"#define {self.name}"
        return f"#define {self.name} {self.value}"


@dataclass(slots=True)
class ShaderSource:
    """Shader source code management.

    Handles loading, preprocessing, and caching of shader source files.

    Attributes:
        path: Source file path (or None for inline code)
        code: Source code string
        language: Shader language
        stage: Shader stage
        entry_point: Entry point function name
        includes: List of include paths
        defines: Preprocessor defines
    """
    path: Optional[str]
    code: str
    language: ShaderLanguage
    stage: ShaderStage
    entry_point: str = "main"
    includes: List[str] = field(default_factory=list)
    defines: List[ShaderDefine] = field(default_factory=list)

    _content_hash: Optional[str] = field(default=None, repr=False)
    _last_modified: float = field(default=0.0, repr=False)

    @classmethod
    def from_file(
        cls,
        path: str,
        stage: ShaderStage,
        language: Optional[ShaderLanguage] = None,
        entry_point: str = "main",
    ) -> ShaderSource:
        """Load shader source from file.

        Args:
            path: File path
            stage: Shader stage
            language: Shader language (auto-detected from extension if None)
            entry_point: Entry point function name

        Returns:
            ShaderSource instance
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Shader file not found: {path}")

        code = path_obj.read_text(encoding="utf-8")

        # Auto-detect language from extension
        if language is None:
            ext_map = {
                ".hlsl": ShaderLanguage.HLSL,
                ".glsl": ShaderLanguage.GLSL,
                ".vert": ShaderLanguage.GLSL,
                ".frag": ShaderLanguage.GLSL,
                ".comp": ShaderLanguage.GLSL,
                ".metal": ShaderLanguage.METAL,
                ".spv": ShaderLanguage.SPIRV,
                ".wgsl": ShaderLanguage.WGSL,
            }
            language = ext_map.get(path_obj.suffix.lower(), ShaderLanguage.GLSL)

        source = cls(
            path=str(path_obj.absolute()),
            code=code,
            language=language,
            stage=stage,
            entry_point=entry_point,
        )
        source._last_modified = path_obj.stat().st_mtime
        return source

    @classmethod
    def from_string(
        cls,
        code: str,
        stage: ShaderStage,
        language: ShaderLanguage,
        entry_point: str = "main",
    ) -> ShaderSource:
        """Create shader source from inline code.

        Args:
            code: Shader source code
            stage: Shader stage
            language: Shader language
            entry_point: Entry point function name

        Returns:
            ShaderSource instance
        """
        return cls(
            path=None,
            code=code,
            language=language,
            stage=stage,
            entry_point=entry_point,
        )

    def get_content_hash(self) -> str:
        """Get hash of shader content including defines."""
        if self._content_hash is None:
            content = self.code
            for define in sorted(self.defines, key=lambda d: d.name):
                content += define.to_string()
            self._content_hash = hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest()[:SHADER_HASH_LENGTH]
        return self._content_hash

    def has_changed(self) -> bool:
        """Check if source file has been modified since loading."""
        if self.path is None:
            return False
        try:
            current_mtime = Path(self.path).stat().st_mtime
            return current_mtime > self._last_modified
        except OSError:
            return False

    def reload(self) -> bool:
        """Reload source from file if changed.

        Returns:
            True if reloaded, False otherwise
        """
        if self.path is None or not self.has_changed():
            return False

        path_obj = Path(self.path)
        self.code = path_obj.read_text(encoding="utf-8")
        self._last_modified = path_obj.stat().st_mtime
        self._content_hash = None
        return True

    def add_define(self, name: str, value: Optional[str] = None) -> None:
        """Add a preprocessor define."""
        self.defines.append(ShaderDefine(name, value))
        self._content_hash = None

    def get_preprocessed_code(self) -> str:
        """Get code with defines prepended."""
        define_block = "\n".join(d.to_string() for d in self.defines)
        if define_block:
            return f"{define_block}\n\n{self.code}"
        return self.code


@dataclass(slots=True, frozen=True)
class PermutationKey:
    """Key identifying a specific shader permutation.

    Permutations are compile-time variants controlled by defines.

    Attributes:
        features: Frozenset of enabled feature names
    """
    features: FrozenSet[str]

    @classmethod
    def from_set(cls, features: Set[str]) -> PermutationKey:
        """Create from a mutable set."""
        return cls(frozenset(features))

    @classmethod
    def from_list(cls, features: List[str]) -> PermutationKey:
        """Create from a list."""
        return cls(frozenset(features))

    @classmethod
    def empty(cls) -> PermutationKey:
        """Create empty permutation key."""
        return cls(frozenset())

    def with_feature(self, feature: str) -> PermutationKey:
        """Return new key with feature added."""
        return PermutationKey(self.features | {feature})

    def without_feature(self, feature: str) -> PermutationKey:
        """Return new key with feature removed."""
        return PermutationKey(self.features - {feature})

    def has_feature(self, feature: str) -> bool:
        """Check if feature is enabled."""
        return feature in self.features

    def to_defines(self) -> List[ShaderDefine]:
        """Convert to list of shader defines."""
        return [ShaderDefine(f"HAS_{f.upper()}") for f in sorted(self.features)]

    def __hash__(self) -> int:
        return hash(self.features)


@dataclass(slots=True)
class ShaderPermutation:
    """Static shader permutation configuration.

    Defines which features are available and their combinations.

    Attributes:
        name: Permutation set name
        features: Available feature names
        required: Features that are always enabled
        conflicts: Feature pairs that cannot be enabled together
    """
    name: str
    features: Set[str] = field(default_factory=set)
    required: Set[str] = field(default_factory=set)
    conflicts: Set[FrozenSet[str]] = field(default_factory=set)

    def add_feature(self, name: str, required: bool = False) -> None:
        """Add a permutation feature."""
        self.features.add(name)
        if required:
            self.required.add(name)

    def add_conflict(self, feature_a: str, feature_b: str) -> None:
        """Add a conflict between two features."""
        self.conflicts.add(frozenset({feature_a, feature_b}))

    def validate_key(self, key: PermutationKey) -> Tuple[bool, str]:
        """Validate a permutation key against this configuration.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required features
        for req in self.required:
            if req not in key.features:
                return False, f"Required feature missing: {req}"

        # Check unknown features
        for feature in key.features:
            if feature not in self.features:
                return False, f"Unknown feature: {feature}"

        # Check conflicts
        for conflict_set in self.conflicts:
            if conflict_set <= key.features:
                features = ", ".join(conflict_set)
                return False, f"Conflicting features: {features}"

        return True, ""

    def get_valid_keys(self) -> List[PermutationKey]:
        """Generate all valid permutation keys."""
        import itertools

        optional = self.features - self.required
        valid_keys = []

        # Generate all combinations of optional features
        for r in range(len(optional) + 1):
            for combo in itertools.combinations(optional, r):
                key = PermutationKey.from_set(self.required | set(combo))
                is_valid, _ = self.validate_key(key)
                if is_valid:
                    valid_keys.append(key)

        return valid_keys

    def count_permutations(self) -> int:
        """Count total number of valid permutations."""
        return len(self.get_valid_keys())


@dataclass(slots=True)
class CompiledShader:
    """Compiled shader binary.

    Attributes:
        source_hash: Hash of source used for compilation
        bytecode: Compiled shader bytecode
        stage: Shader stage
        entry_point: Entry point name
        permutation_key: Permutation used
        compile_time_ms: Time taken to compile in milliseconds
        reflection_data: Shader reflection data (uniforms, attributes, etc.)
    """
    source_hash: str
    bytecode: bytes
    stage: ShaderStage
    entry_point: str
    permutation_key: PermutationKey
    compile_time_ms: float = 0.0
    reflection_data: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if shader bytecode is valid."""
        return len(self.bytecode) > 0


@dataclass(slots=True, frozen=True)
class PSODescriptor:
    """Pipeline State Object descriptor.

    Describes the complete pipeline state for PSO creation/caching.

    Attributes:
        vertex_shader_hash: Hash of vertex shader
        fragment_shader_hash: Hash of fragment shader
        compute_shader_hash: Hash of compute shader (for compute pipelines)
        vertex_format_hash: Hash of vertex input format
        blend_state_hash: Hash of blend state configuration
        depth_state_hash: Hash of depth/stencil state
        rasterizer_state_hash: Hash of rasterizer state
        render_target_formats: Tuple of render target format hashes
    """
    vertex_shader_hash: str = ""
    fragment_shader_hash: str = ""
    compute_shader_hash: str = ""
    vertex_format_hash: str = ""
    blend_state_hash: str = ""
    depth_state_hash: str = ""
    rasterizer_state_hash: str = ""
    render_target_formats: Tuple[str, ...] = ()

    def get_hash(self) -> str:
        """Get combined hash for PSO lookup."""
        combined = (
            f"{self.vertex_shader_hash}|{self.fragment_shader_hash}|"
            f"{self.compute_shader_hash}|{self.vertex_format_hash}|"
            f"{self.blend_state_hash}|{self.depth_state_hash}|"
            f"{self.rasterizer_state_hash}|{':'.join(self.render_target_formats)}"
        )
        return hashlib.sha256(combined.encode()).hexdigest()[:SHADER_HASH_LENGTH]

    def is_compute_pipeline(self) -> bool:
        """Check if this is a compute pipeline."""
        return bool(self.compute_shader_hash) and not self.vertex_shader_hash


class PSOCache:
    """Pipeline State Object cache.

    Caches compiled PSOs to avoid redundant creation.
    Supports disk serialization for faster subsequent loads.

    Attributes:
        max_size: Maximum number of cached PSOs
        hit_count: Number of cache hits
        miss_count: Number of cache misses
    """

    __slots__ = (
        "_cache",
        "_max_size",
        "_hit_count",
        "_miss_count",
        "_lru_order",
        "_cache_path",
    )

    def __init__(
        self,
        max_size: int = PSO_CACHE_DEFAULT_MAX_SIZE,
        cache_path: Optional[str] = None,
    ) -> None:
        self._cache: Dict[str, Any] = {}
        self._max_size = max_size
        self._hit_count = 0
        self._miss_count = 0
        self._lru_order: List[str] = []
        self._cache_path = cache_path

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate as a ratio."""
        total = self._hit_count + self._miss_count
        if total == 0:
            return 0.0
        return self._hit_count / total

    def get(self, descriptor: PSODescriptor) -> Optional[Any]:
        """Get cached PSO for descriptor.

        Args:
            descriptor: PSO descriptor

        Returns:
            Cached PSO or None if not found
        """
        key = descriptor.get_hash()
        if key in self._cache:
            self._hit_count += 1
            # Update LRU order
            self._lru_order.remove(key)
            self._lru_order.append(key)
            return self._cache[key]

        self._miss_count += 1
        return None

    def put(self, descriptor: PSODescriptor, pso: Any) -> None:
        """Cache a PSO.

        Args:
            descriptor: PSO descriptor
            pso: Pipeline state object to cache
        """
        key = descriptor.get_hash()

        # Evict if at capacity
        while len(self._cache) >= self._max_size and self._lru_order:
            oldest = self._lru_order.pop(0)
            self._cache.pop(oldest, None)

        self._cache[key] = pso
        if key in self._lru_order:
            self._lru_order.remove(key)
        self._lru_order.append(key)

    def invalidate(self, descriptor: PSODescriptor) -> None:
        """Invalidate a cached PSO."""
        key = descriptor.get_hash()
        self._cache.pop(key, None)
        if key in self._lru_order:
            self._lru_order.remove(key)

    def clear(self) -> None:
        """Clear all cached PSOs."""
        self._cache.clear()
        self._lru_order.clear()
        self._hit_count = 0
        self._miss_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self.hit_rate,
        }

    def __len__(self) -> int:
        return len(self._cache)


class HotReloadWatcher:
    """Watches shader files for changes and triggers recompilation.

    Attributes:
        watched_paths: Set of watched file paths
        poll_interval: How often to check for changes (seconds)
    """

    __slots__ = (
        "_watched",
        "_poll_interval",
        "_last_check",
        "_on_change",
    )

    def __init__(
        self,
        poll_interval: float = HOT_RELOAD_POLL_INTERVAL_SECONDS,
    ) -> None:
        self._watched: Dict[str, float] = {}  # path -> last_mtime
        self._poll_interval = poll_interval
        self._last_check = 0.0
        self._on_change: List[Callable[[str], None]] = []

    def watch(self, path: str) -> None:
        """Start watching a shader file."""
        path = os.path.abspath(path)
        if path not in self._watched:
            try:
                mtime = os.path.getmtime(path)
                self._watched[path] = mtime
            except OSError:
                pass

    def unwatch(self, path: str) -> None:
        """Stop watching a shader file."""
        path = os.path.abspath(path)
        self._watched.pop(path, None)

    def on_change(self, callback: Callable[[str], None]) -> None:
        """Register callback for file changes."""
        self._on_change.append(callback)

    def check_changes(self) -> List[str]:
        """Check for changed files.

        Returns:
            List of paths that have changed since last check
        """
        now = time.time()
        if now - self._last_check < self._poll_interval:
            return []

        self._last_check = now
        changed = []

        for path, last_mtime in list(self._watched.items()):
            try:
                current_mtime = os.path.getmtime(path)
                if current_mtime > last_mtime:
                    self._watched[path] = current_mtime
                    changed.append(path)
                    for callback in self._on_change:
                        callback(path)
            except OSError:
                pass

        return changed


class ShaderCompiler:
    """Shader variant compilation manager.

    Handles compilation of shader sources into GPU bytecode with:
    - Permutation variant generation
    - Hot-reload support
    - PSO caching
    - Cross-compilation between shader languages

    Attributes:
        pso_cache: Pipeline state object cache
        hot_reload: Hot-reload watcher
    """

    __slots__ = (
        "_pso_cache",
        "_hot_reload",
        "_compiled_shaders",
        "_compile_callbacks",
        "_error_callbacks",
        "_target_language",
    )

    def __init__(
        self,
        pso_cache: Optional[PSOCache] = None,
        enable_hot_reload: bool = False,
        target_language: ShaderLanguage = ShaderLanguage.SPIRV,
    ) -> None:
        self._pso_cache = pso_cache or PSOCache()
        self._hot_reload = HotReloadWatcher() if enable_hot_reload else None
        self._compiled_shaders: Dict[str, CompiledShader] = {}
        self._compile_callbacks: List[
            Callable[[CompiledShader], None]
        ] = []
        self._error_callbacks: List[
            Callable[[CompilationError], None]
        ] = []
        self._target_language = target_language

        if self._hot_reload:
            self._hot_reload.on_change(self._on_source_changed)

    @property
    def pso_cache(self) -> PSOCache:
        return self._pso_cache

    def compile(
        self,
        source: ShaderSource,
        permutation: Optional[PermutationKey] = None,
        optimize: bool = True,
    ) -> CompiledShader:
        """Compile a shader source.

        Args:
            source: Shader source to compile
            permutation: Permutation key for variants
            optimize: Whether to enable optimizations

        Returns:
            Compiled shader

        Raises:
            CompilationError: If compilation fails
        """
        permutation = permutation or PermutationKey.empty()

        # Check cache first
        cache_key = self._get_cache_key(source, permutation)
        if cache_key in self._compiled_shaders:
            cached = self._compiled_shaders[cache_key]
            if cached.source_hash == source.get_content_hash():
                return cached

        # Add permutation defines
        source_with_defines = ShaderSource(
            path=source.path,
            code=source.code,
            language=source.language,
            stage=source.stage,
            entry_point=source.entry_point,
            includes=source.includes,
            defines=source.defines + permutation.to_defines(),
        )

        start_time = time.time()

        try:
            # Perform compilation (placeholder - actual implementation
            # would call platform-specific compiler)
            bytecode = self._compile_internal(source_with_defines, optimize)

            compile_time_ms = (time.time() - start_time) * 1000

            compiled = CompiledShader(
                source_hash=source.get_content_hash(),
                bytecode=bytecode,
                stage=source.stage,
                entry_point=source.entry_point,
                permutation_key=permutation,
                compile_time_ms=compile_time_ms,
                reflection_data=self._extract_reflection(bytecode),
            )

            self._compiled_shaders[cache_key] = compiled

            # Register for hot-reload if enabled
            if self._hot_reload and source.path:
                self._hot_reload.watch(source.path)

            # Notify callbacks
            for callback in self._compile_callbacks:
                callback(compiled)

            return compiled

        except Exception as e:
            error = CompilationError(
                str(e),
                source_file=source.path,
            )
            for callback in self._error_callbacks:
                callback(error)
            raise error

    def compile_permutations(
        self,
        source: ShaderSource,
        permutation_config: ShaderPermutation,
        optimize: bool = True,
    ) -> List[CompiledShader]:
        """Compile all valid permutations of a shader.

        Args:
            source: Shader source
            permutation_config: Permutation configuration
            optimize: Whether to enable optimizations

        Returns:
            List of compiled shaders for each permutation
        """
        compiled = []
        for key in permutation_config.get_valid_keys():
            try:
                shader = self.compile(source, key, optimize)
                compiled.append(shader)
            except CompilationError:
                # Continue with other permutations on error
                pass
        return compiled

    def get_cached(
        self,
        source: ShaderSource,
        permutation: Optional[PermutationKey] = None,
    ) -> Optional[CompiledShader]:
        """Get compiled shader from cache without recompiling.

        Args:
            source: Shader source
            permutation: Permutation key

        Returns:
            Cached compiled shader or None
        """
        permutation = permutation or PermutationKey.empty()
        cache_key = self._get_cache_key(source, permutation)
        return self._compiled_shaders.get(cache_key)

    def invalidate(
        self,
        source: ShaderSource,
        permutation: Optional[PermutationKey] = None,
    ) -> None:
        """Invalidate cached shader.

        Args:
            source: Shader source
            permutation: Permutation key (if None, invalidates all)
        """
        if permutation is None:
            # Invalidate all permutations for this source
            prefix = f"{source.path or id(source)}|"
            keys_to_remove = [
                k for k in self._compiled_shaders if k.startswith(prefix)
            ]
            for key in keys_to_remove:
                del self._compiled_shaders[key]
        else:
            cache_key = self._get_cache_key(source, permutation)
            self._compiled_shaders.pop(cache_key, None)

    def check_hot_reload(self) -> List[str]:
        """Check for shader file changes and trigger recompilation.

        Returns:
            List of changed file paths
        """
        if self._hot_reload is None:
            return []
        return self._hot_reload.check_changes()

    def on_compile(
        self,
        callback: Callable[[CompiledShader], None],
    ) -> None:
        """Register callback for successful compilation."""
        self._compile_callbacks.append(callback)

    def on_error(
        self,
        callback: Callable[[CompilationError], None],
    ) -> None:
        """Register callback for compilation errors."""
        self._error_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get compiler statistics."""
        return {
            "cached_shaders": len(self._compiled_shaders),
            "pso_cache": self._pso_cache.get_stats(),
        }

    def _get_cache_key(
        self,
        source: ShaderSource,
        permutation: PermutationKey,
    ) -> str:
        """Generate cache key for source + permutation."""
        source_id = source.path or str(id(source))
        perm_str = "|".join(sorted(permutation.features))
        return f"{source_id}|{perm_str}"

    def _compile_internal(
        self,
        source: ShaderSource,
        optimize: bool,
    ) -> bytes:
        """Internal compilation implementation.

        This is a placeholder that should be overridden or extended
        with actual compiler integration (glslang, dxc, etc.).
        """
        # Placeholder: return hash of source as "bytecode"
        code = source.get_preprocessed_code()
        return hashlib.sha256(code.encode()).digest()

    def _extract_reflection(self, bytecode: bytes) -> Dict[str, Any]:
        """Extract reflection data from compiled shader.

        This is a placeholder for SPIRV-Cross or similar reflection.
        """
        return {
            "uniforms": [],
            "samplers": [],
            "inputs": [],
            "outputs": [],
        }

    def _on_source_changed(self, path: str) -> None:
        """Handle shader source file change."""
        # Find and recompile affected shaders
        prefix = f"{path}|"
        for key in list(self._compiled_shaders.keys()):
            if key.startswith(prefix):
                del self._compiled_shaders[key]
