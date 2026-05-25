"""
Tests for compilation decorators (compilation.py).

Tests the 7 compilation decorators built on Ops:
    @native, @ffi, @target, @unsafe, @backend, @capability, @platform

Each test verifies:
1. Steps are applied (decompose works, _applied_steps populated)
2. Domain attributes are set correctly
3. Validation rejects invalid params
4. Introspection works
"""

import pytest

from trinity.decorators.base import get_current_arch, get_current_platform
from trinity.decorators.compilation import (
    VALID_ARCHITECTURES,
    VALID_BACKEND_TYPES,
    VALID_CAPABILITIES,
    VALID_FFI_ABIS,
    VALID_NATIVE_BACKENDS,
    VALID_OPERATING_SYSTEMS,
    BackendConfig,
    CapabilityConfig,
    FFIConfig,
    NativeConfig,
    PlatformConfig,
    TargetConfig,
    backend,
    capability,
    ffi,
    native,
    platform,
    target,
    unsafe,
)
from trinity.decorators.ops import Op, decompose, expand

# =============================================================================
# @native
# =============================================================================


class TestNative:
    def test_default_params(self):
        @native()
        class Foo:
            pass

        assert Foo._native is True
        assert isinstance(Foo._native_config, NativeConfig)
        assert Foo._native_config.backend == "cython"

    def test_custom_backend(self):
        @native(backend="numba", nogil=True)
        class Bar:
            pass

        assert Bar._native_config.backend == "numba"
        assert Bar._native_config.nogil is True

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="invalid backend"):

            @native(backend="not_real")
            class Bad:
                pass

    def test_applied_decorators(self):
        @native()
        class C:
            pass

        assert "native" in C._applied_decorators

    def test_steps_recorded(self):
        @native()
        class C:
            pass

        assert hasattr(C, "_applied_steps")
        ops_used = {s.op for s in C._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used

    def test_decompose(self):
        steps = decompose(native)
        assert len(steps) > 0

    def test_no_parens(self):
        @native
        class C:
            pass

        assert C._native is True

    def test_on_function(self):
        @native()
        def fn():
            pass

        assert fn._native is True


# =============================================================================
# @ffi
# =============================================================================


class TestFFI:
    def test_basic(self):
        @ffi(lib="physics")
        def create_body():
            pass

        assert create_body._ffi is True
        assert create_body._ffi_config.lib == "physics"
        assert create_body._ffi_config.abi == "c"

    def test_custom_abi(self):
        @ffi(lib="win32", abi="stdcall")
        def win_fn():
            pass

        assert win_fn._ffi_config.abi == "stdcall"

    def test_missing_lib(self):
        with pytest.raises(ValueError, match="'lib' parameter is required"):

            @ffi(lib="")
            def bad():
                pass

    def test_invalid_abi(self):
        with pytest.raises(ValueError, match="invalid abi"):

            @ffi(lib="x", abi="invalid")
            def bad():
                pass

    def test_applied_decorators(self):
        @ffi(lib="test")
        def fn():
            pass

        assert "ffi" in fn._applied_decorators

    def test_steps_recorded(self):
        @ffi(lib="test")
        def fn():
            pass

        ops_used = {s.op for s in fn._applied_steps}
        assert Op.TAG in ops_used


# =============================================================================
# @target
# =============================================================================


class TestTarget:
    def test_default(self):
        @target()
        class C:
            pass

        assert C._target is True
        assert isinstance(C._target_config, TargetConfig)
        assert "x86_64" in C._target_config.arch

    def test_availability_on_current_platform(self):
        current_arch = get_current_arch()

        @target(arch={current_arch})
        class C:
            pass

        assert C._target_available is True

    def test_unavailable_arch(self):
        @target(arch={"wasm32"})
        def fn():
            pass

        # Should be stubbed on non-wasm
        current = get_current_arch()
        if current != "wasm32":
            assert fn._target_available is False

    def test_invalid_arch(self):
        with pytest.raises(ValueError, match="invalid architecture"):

            @target(arch={"powerpc9000"})
            class Bad:
                pass

    def test_invalid_os(self):
        with pytest.raises(ValueError, match="invalid OS"):

            @target(os={"templeOS"})
            class Bad:
                pass

    def test_applied_decorators(self):
        @target()
        class C:
            pass

        assert "target" in C._applied_decorators


# =============================================================================
# @unsafe
# =============================================================================


class TestUnsafe:
    def test_marker(self):
        @unsafe
        class C:
            pass

        assert C._unsafe is True

    def test_on_function(self):
        @unsafe
        def fn():
            pass

        assert fn._unsafe is True

    def test_applied_decorators(self):
        @unsafe
        class C:
            pass

        assert "unsafe" in C._applied_decorators

    def test_steps_recorded(self):
        @unsafe
        class C:
            pass

        assert hasattr(C, "_applied_steps")
        ops_used = {s.op for s in C._applied_steps}
        assert Op.TAG in ops_used


# =============================================================================
# @backend
# =============================================================================


class TestBackend:
    def test_graphics_vulkan(self):
        @backend(type="graphics", api="vulkan")
        class C:
            pass

        assert C._backend is True
        assert C._backend_config.type == "graphics"
        assert C._backend_config.api == "vulkan"

    def test_audio(self):
        @backend(type="audio", api="wasapi")
        class C:
            pass

        assert C._backend_config.type == "audio"

    def test_fallback(self):
        @backend(type="graphics", api="vulkan", fallback="d3d12")
        class C:
            pass

        assert C._backend_config.fallback == "d3d12"

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="invalid type"):

            @backend(type="quantum", api="q")
            class Bad:
                pass

    def test_invalid_graphics_api(self):
        with pytest.raises(ValueError, match="invalid API"):

            @backend(type="graphics", api="directx_99")
            class Bad:
                pass

    def test_invalid_fallback(self):
        with pytest.raises(ValueError, match="invalid fallback"):

            @backend(type="graphics", api="vulkan", fallback="directx_99")
            class Bad:
                pass

    def test_applied_decorators(self):
        @backend(type="graphics", api="vulkan")
        class C:
            pass

        assert "backend" in C._applied_decorators


# =============================================================================
# @capability
# =============================================================================


class TestCapability:
    def test_basic(self):
        @capability(requires={"ray_tracing"})
        class C:
            pass

        assert C._capability is True
        assert "ray_tracing" in C._capability_config.requires

    def test_with_fallback(self):
        fb = lambda: None

        @capability(requires={"avx2"}, fallback=fb)
        class C:
            pass

        assert C._capability_config.fallback is fb

    def test_invalid_capability(self):
        with pytest.raises(ValueError, match="invalid capability"):

            @capability(requires={"quantum_entanglement"})
            class Bad:
                pass

    def test_applied_decorators(self):
        @capability(requires={"compute_shader"})
        class C:
            pass

        assert "capability" in C._applied_decorators


# =============================================================================
# @platform
# =============================================================================


class TestPlatform:
    def test_include(self):
        current = get_current_platform()

        @platform(include={current})
        class C:
            pass

        assert C._platform is True
        assert C._platform_available is True

    def test_exclude_current(self):
        current = get_current_platform()

        @platform(exclude={current})
        def fn():
            pass

        assert fn._platform_available is False

    def test_include_unavailable(self):
        @platform(include={"switch"})
        def fn():
            pass

        current = get_current_platform()
        if current != "switch":
            assert fn._platform_available is False

    def test_both_include_exclude_fails(self):
        with pytest.raises(ValueError, match="cannot specify both"):

            @platform(include={"linux"}, exclude={"windows"})
            class Bad:
                pass

    def test_neither_include_exclude_fails(self):
        with pytest.raises(ValueError, match="must specify either"):

            @platform()
            class Bad:
                pass

    def test_invalid_platform(self):
        with pytest.raises(ValueError, match="invalid platform"):

            @platform(include={"commodore64"})
            class Bad:
                pass

    def test_applied_decorators(self):
        current = get_current_platform()

        @platform(include={current})
        class C:
            pass

        assert "platform" in C._applied_decorators


# =============================================================================
# INTROSPECTION (all decorators decompose)
# =============================================================================


class TestCompilationIntrospection:
    @pytest.mark.parametrize(
        "dec", [native, ffi, target, unsafe, backend, capability, platform]
    )
    def test_decompose_returns_steps(self, dec):
        steps = decompose(dec)
        assert isinstance(steps, list)

    @pytest.mark.parametrize(
        "dec", [native, ffi, target, unsafe, backend, capability, platform]
    )
    def test_expand_returns_string(self, dec):
        result = expand(dec)
        assert isinstance(result, str)

    def test_all_register_compilation(self):
        """Every compilation decorator should have a REGISTER step for 'compilation'."""
        for dec in [native, ffi, target, unsafe, backend, capability, platform]:
            steps = decompose(dec)
            reg_steps = [s for s in steps if s.op is Op.REGISTER]
            assert any(s.args.get("registry") == "compilation" for s in reg_steps), (
                f"{dec.__name__} missing REGISTER(compilation) step"
            )
