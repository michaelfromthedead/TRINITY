"""Tests for the shader compilation system.

Tests ShaderSource, ShaderPermutation, PSOCache, and ShaderCompiler.
"""
import tempfile
import os
import pytest

from engine.rendering.materials.shader_compiler import (
    CompiledShader,
    CompilationError,
    HotReloadWatcher,
    PermutationKey,
    PSOCache,
    PSODescriptor,
    ShaderCompiler,
    ShaderDefine,
    ShaderLanguage,
    ShaderPermutation,
    ShaderSource,
    ShaderStage,
)


class TestShaderDefine:
    """Test ShaderDefine preprocessor defines."""

    def test_define_without_value(self):
        """Test define without value."""
        define = ShaderDefine(name="NORMAL_MAPPING")
        assert define.to_string() == "#define NORMAL_MAPPING"

    def test_define_with_value(self):
        """Test define with value."""
        define = ShaderDefine(name="MAX_LIGHTS", value="16")
        assert define.to_string() == "#define MAX_LIGHTS 16"

    def test_define_hashable(self):
        """Test that defines are hashable (frozen)."""
        define = ShaderDefine(name="TEST")
        hash(define)  # Should not raise


class TestShaderSource:
    """Test ShaderSource management."""

    def test_from_string(self):
        """Test creating from inline code."""
        code = "void main() { }"
        source = ShaderSource.from_string(
            code=code,
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )
        assert source.code == code
        assert source.stage == ShaderStage.FRAGMENT
        assert source.language == ShaderLanguage.GLSL
        assert source.path is None

    def test_from_file(self):
        """Test loading from file."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".glsl",
            delete=False,
        ) as f:
            f.write("void main() { gl_FragColor = vec4(1.0); }")
            temp_path = f.name

        try:
            source = ShaderSource.from_file(
                path=temp_path,
                stage=ShaderStage.FRAGMENT,
            )
            assert "gl_FragColor" in source.code
            assert source.language == ShaderLanguage.GLSL
        finally:
            os.unlink(temp_path)

    def test_from_file_auto_detect_hlsl(self):
        """Test auto-detecting HLSL language."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".hlsl",
            delete=False,
        ) as f:
            f.write("float4 main() : SV_Target { return 1; }")
            temp_path = f.name

        try:
            source = ShaderSource.from_file(
                path=temp_path,
                stage=ShaderStage.FRAGMENT,
            )
            assert source.language == ShaderLanguage.HLSL
        finally:
            os.unlink(temp_path)

    def test_from_file_not_found(self):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            ShaderSource.from_file(
                path="/nonexistent/shader.glsl",
                stage=ShaderStage.VERTEX,
            )

    def test_content_hash(self):
        """Test content hashing."""
        source = ShaderSource.from_string(
            code="void main() { }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )
        hash1 = source.get_content_hash()
        assert len(hash1) == 16  # Truncated SHA256

        # Same code should produce same hash
        source2 = ShaderSource.from_string(
            code="void main() { }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )
        assert source2.get_content_hash() == hash1

    def test_add_define(self):
        """Test adding defines."""
        source = ShaderSource.from_string(
            code="// shader",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )
        source.add_define("USE_NORMAL_MAP")
        assert len(source.defines) == 1

    def test_get_preprocessed_code(self):
        """Test getting code with defines."""
        source = ShaderSource.from_string(
            code="void main() { }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )
        source.add_define("FEATURE_A")
        source.add_define("MAX_LIGHTS", "8")

        preprocessed = source.get_preprocessed_code()
        assert "#define FEATURE_A" in preprocessed
        assert "#define MAX_LIGHTS 8" in preprocessed
        assert "void main()" in preprocessed


class TestPermutationKey:
    """Test PermutationKey variant selection."""

    def test_empty_key(self):
        """Test empty permutation key."""
        key = PermutationKey.empty()
        assert len(key.features) == 0

    def test_from_set(self):
        """Test creating from set."""
        key = PermutationKey.from_set({"FEATURE_A", "FEATURE_B"})
        assert len(key.features) == 2
        assert key.has_feature("FEATURE_A")

    def test_with_feature(self):
        """Test adding feature."""
        key = PermutationKey.empty()
        new_key = key.with_feature("NORMAL_MAP")
        assert key.features != new_key.features
        assert new_key.has_feature("NORMAL_MAP")

    def test_without_feature(self):
        """Test removing feature."""
        key = PermutationKey.from_set({"A", "B"})
        new_key = key.without_feature("A")
        assert not new_key.has_feature("A")
        assert new_key.has_feature("B")

    def test_to_defines(self):
        """Test converting to shader defines."""
        key = PermutationKey.from_set({"normal_map", "ao"})
        defines = key.to_defines()
        assert len(defines) == 2
        names = [d.name for d in defines]
        assert "HAS_NORMAL_MAP" in names
        assert "HAS_AO" in names

    def test_hashable(self):
        """Test that keys are hashable."""
        key1 = PermutationKey.from_set({"A", "B"})
        key2 = PermutationKey.from_set({"A", "B"})
        assert hash(key1) == hash(key2)


class TestShaderPermutation:
    """Test ShaderPermutation configuration."""

    def test_basic_permutation(self):
        """Test basic permutation setup."""
        perm = ShaderPermutation(name="PBR")
        perm.add_feature("NORMAL_MAP")
        perm.add_feature("METALLIC_MAP")
        assert len(perm.features) == 2

    def test_required_feature(self):
        """Test required features."""
        perm = ShaderPermutation(name="PBR")
        perm.add_feature("BASE_PASS", required=True)
        perm.add_feature("NORMAL_MAP")

        # Valid key with required feature
        key = PermutationKey.from_set({"BASE_PASS", "NORMAL_MAP"})
        is_valid, _ = perm.validate_key(key)
        assert is_valid

        # Invalid key missing required feature
        key = PermutationKey.from_set({"NORMAL_MAP"})
        is_valid, error = perm.validate_key(key)
        assert not is_valid
        assert "Required feature missing" in error

    def test_conflicting_features(self):
        """Test feature conflicts."""
        perm = ShaderPermutation(name="Blend")
        perm.add_feature("OPAQUE")
        perm.add_feature("TRANSLUCENT")
        perm.add_conflict("OPAQUE", "TRANSLUCENT")

        key = PermutationKey.from_set({"OPAQUE", "TRANSLUCENT"})
        is_valid, error = perm.validate_key(key)
        assert not is_valid
        assert "Conflicting features" in error

    def test_unknown_feature(self):
        """Test unknown feature rejection."""
        perm = ShaderPermutation(name="Test")
        perm.add_feature("KNOWN")

        key = PermutationKey.from_set({"UNKNOWN"})
        is_valid, error = perm.validate_key(key)
        assert not is_valid
        assert "Unknown feature" in error

    def test_get_valid_keys(self):
        """Test generating valid permutation keys."""
        perm = ShaderPermutation(name="Test")
        perm.add_feature("A")
        perm.add_feature("B")

        keys = perm.get_valid_keys()
        # Should have: {}, {A}, {B}, {A,B}
        assert len(keys) == 4

    def test_count_permutations(self):
        """Test permutation counting."""
        perm = ShaderPermutation(name="Test")
        perm.add_feature("A")
        perm.add_feature("B")
        perm.add_feature("C")

        # 2^3 = 8 combinations
        assert perm.count_permutations() == 8


class TestPSOCache:
    """Test Pipeline State Object caching."""

    def test_empty_cache(self):
        """Test empty cache state."""
        cache = PSOCache()
        assert len(cache) == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0

    def test_put_and_get(self):
        """Test putting and retrieving PSOs."""
        cache = PSOCache()
        desc = PSODescriptor(
            vertex_shader_hash="vs_hash",
            fragment_shader_hash="fs_hash",
        )
        pso = {"pipeline": "mock_pso"}

        cache.put(desc, pso)
        assert len(cache) == 1

        result = cache.get(desc)
        assert result == pso
        assert cache.hit_count == 1

    def test_cache_miss(self):
        """Test cache miss tracking."""
        cache = PSOCache()
        desc = PSODescriptor(vertex_shader_hash="missing")

        result = cache.get(desc)
        assert result is None
        assert cache.miss_count == 1

    def test_hit_rate(self):
        """Test hit rate calculation."""
        cache = PSOCache()
        desc = PSODescriptor(vertex_shader_hash="test")
        cache.put(desc, "pso")

        cache.get(desc)  # Hit
        cache.get(desc)  # Hit
        cache.get(PSODescriptor(vertex_shader_hash="miss"))  # Miss

        assert cache.hit_rate == 2 / 3

    def test_lru_eviction(self):
        """Test LRU eviction when at capacity."""
        cache = PSOCache(max_size=2)

        desc1 = PSODescriptor(vertex_shader_hash="1")
        desc2 = PSODescriptor(vertex_shader_hash="2")
        desc3 = PSODescriptor(vertex_shader_hash="3")

        cache.put(desc1, "pso1")
        cache.put(desc2, "pso2")
        cache.put(desc3, "pso3")

        assert len(cache) == 2
        # desc1 should have been evicted (LRU)
        assert cache.get(desc1) is None
        assert cache.get(desc2) is not None

    def test_invalidate(self):
        """Test PSO invalidation."""
        cache = PSOCache()
        desc = PSODescriptor(vertex_shader_hash="test")
        cache.put(desc, "pso")

        cache.invalidate(desc)
        assert cache.get(desc) is None

    def test_clear(self):
        """Test clearing cache."""
        cache = PSOCache()
        cache.put(PSODescriptor(vertex_shader_hash="1"), "pso1")
        cache.put(PSODescriptor(vertex_shader_hash="2"), "pso2")

        cache.clear()
        assert len(cache) == 0
        assert cache.hit_count == 0

    def test_get_stats(self):
        """Test stats retrieval."""
        cache = PSOCache(max_size=100)
        cache.put(PSODescriptor(vertex_shader_hash="1"), "pso")
        cache.get(PSODescriptor(vertex_shader_hash="1"))

        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["hit_count"] == 1


class TestPSODescriptor:
    """Test PSODescriptor hashing."""

    def test_get_hash(self):
        """Test descriptor hashing."""
        desc = PSODescriptor(
            vertex_shader_hash="vs",
            fragment_shader_hash="fs",
            blend_state_hash="blend",
        )
        h = desc.get_hash()
        assert len(h) == 16

    def test_same_descriptors_same_hash(self):
        """Test identical descriptors produce same hash."""
        desc1 = PSODescriptor(
            vertex_shader_hash="vs",
            fragment_shader_hash="fs",
        )
        desc2 = PSODescriptor(
            vertex_shader_hash="vs",
            fragment_shader_hash="fs",
        )
        assert desc1.get_hash() == desc2.get_hash()

    def test_is_compute_pipeline(self):
        """Test compute pipeline detection."""
        graphics = PSODescriptor(
            vertex_shader_hash="vs",
            fragment_shader_hash="fs",
        )
        compute = PSODescriptor(
            compute_shader_hash="cs",
        )
        assert not graphics.is_compute_pipeline()
        assert compute.is_compute_pipeline()


class TestShaderCompiler:
    """Test ShaderCompiler compilation."""

    def test_compile_basic(self):
        """Test basic shader compilation."""
        compiler = ShaderCompiler()
        source = ShaderSource.from_string(
            code="void main() { }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )

        compiled = compiler.compile(source)
        assert compiled.is_valid()
        assert compiled.stage == ShaderStage.FRAGMENT
        assert compiled.entry_point == "main"

    def test_compile_with_permutation(self):
        """Test compilation with permutation key."""
        compiler = ShaderCompiler()
        source = ShaderSource.from_string(
            code="void main() { }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )
        key = PermutationKey.from_set({"NORMAL_MAP"})

        compiled = compiler.compile(source, key)
        assert compiled.permutation_key == key

    def test_compile_cached(self):
        """Test that repeated compilation uses cache."""
        compiler = ShaderCompiler()
        source = ShaderSource.from_string(
            code="void main() { }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )

        compiled1 = compiler.compile(source)
        compiled2 = compiler.compile(source)

        # Should return same compiled shader from cache
        assert compiled1.source_hash == compiled2.source_hash

    def test_get_cached(self):
        """Test getting cached shader without recompiling."""
        compiler = ShaderCompiler()
        source = ShaderSource.from_string(
            code="void main() { }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )

        # Not compiled yet
        assert compiler.get_cached(source) is None

        compiler.compile(source)
        cached = compiler.get_cached(source)
        assert cached is not None

    def test_invalidate(self):
        """Test cache invalidation."""
        compiler = ShaderCompiler()
        source = ShaderSource.from_string(
            code="void main() { }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )

        compiler.compile(source)
        compiler.invalidate(source)
        assert compiler.get_cached(source) is None

    def test_compile_permutations(self):
        """Test compiling all permutations."""
        compiler = ShaderCompiler()
        source = ShaderSource.from_string(
            code="void main() { }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )

        perm = ShaderPermutation(name="Test")
        perm.add_feature("A")
        perm.add_feature("B")

        compiled = compiler.compile_permutations(source, perm)
        assert len(compiled) == 4  # 2^2 permutations

    def test_on_compile_callback(self):
        """Test compile success callback."""
        compiler = ShaderCompiler()
        source = ShaderSource.from_string(
            code="void main() { }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )

        callbacks = []
        compiler.on_compile(lambda c: callbacks.append(c))

        compiler.compile(source)
        assert len(callbacks) == 1

    def test_get_stats(self):
        """Test compiler statistics."""
        compiler = ShaderCompiler()
        source = ShaderSource.from_string(
            code="void main() { }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )

        compiler.compile(source)
        stats = compiler.get_stats()
        assert stats["cached_shaders"] == 1

    def test_compiled_shader_has_bytecode(self):
        """Test that compiled shader contains non-empty bytecode."""
        compiler = ShaderCompiler()
        source = ShaderSource.from_string(
            code="void main() { gl_FragColor = vec4(1.0); }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )

        compiled = compiler.compile(source)

        # Verify bytecode is produced
        assert compiled.bytecode is not None
        assert len(compiled.bytecode) > 0
        assert compiled.is_valid()

    def test_compiled_shader_has_reflection_data(self):
        """Test that compiled shader includes reflection data structure."""
        compiler = ShaderCompiler()
        source = ShaderSource.from_string(
            code="uniform float u_time; void main() { }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )

        compiled = compiler.compile(source)

        # Verify reflection data structure exists
        assert "uniforms" in compiled.reflection_data
        assert "samplers" in compiled.reflection_data
        assert "inputs" in compiled.reflection_data
        assert "outputs" in compiled.reflection_data

    def test_compile_tracks_time(self):
        """Test that compilation time is tracked."""
        compiler = ShaderCompiler()
        source = ShaderSource.from_string(
            code="void main() { }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )

        compiled = compiler.compile(source)

        # Compilation time should be recorded (may be very small)
        assert compiled.compile_time_ms >= 0.0

    def test_different_sources_different_bytecode(self):
        """Test that different source code produces different bytecode."""
        compiler = ShaderCompiler()

        source1 = ShaderSource.from_string(
            code="void main() { gl_FragColor = vec4(1.0); }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )
        source2 = ShaderSource.from_string(
            code="void main() { gl_FragColor = vec4(0.5); }",
            stage=ShaderStage.FRAGMENT,
            language=ShaderLanguage.GLSL,
        )

        compiled1 = compiler.compile(source1)
        compiled2 = compiler.compile(source2)

        # Different source should produce different bytecode
        assert compiled1.bytecode != compiled2.bytecode
        assert compiled1.source_hash != compiled2.source_hash


class TestHotReloadWatcher:
    """Test HotReloadWatcher file monitoring."""

    def test_watch_file(self):
        """Test watching a file."""
        watcher = HotReloadWatcher()

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".glsl",
            delete=False,
        ) as f:
            f.write("// shader")
            temp_path = f.name

        try:
            watcher.watch(temp_path)
            assert temp_path in [os.path.abspath(p) for p in watcher._watched]
        finally:
            os.unlink(temp_path)

    def test_unwatch_file(self):
        """Test unwatching a file."""
        watcher = HotReloadWatcher()

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".glsl",
            delete=False,
        ) as f:
            f.write("// shader")
            temp_path = f.name

        try:
            watcher.watch(temp_path)
            watcher.unwatch(temp_path)
            assert temp_path not in watcher._watched
        finally:
            os.unlink(temp_path)
