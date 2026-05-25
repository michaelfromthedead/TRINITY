"""
Compilation decorators — built from Ops.

These decorators mark code for transformation: native compilation,
FFI binding, platform targeting, backend selection, etc.

Every decorator here is a named list of Steps, created by make_decorator.
The Steps do the real work. The decorators are just configuration.

Decorators:
    @native     - AOT compilation marker
    @ffi        - Foreign function interface
    @target     - Cross-platform targeting
    @unsafe     - Dangerous operations marker
    @backend    - Backend/API selection
    @capability - Hardware capability requirements
    @platform   - Platform inclusion/exclusion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar, Union

from trinity.decorators.base import (
    PlatformUnavailableError,
    create_unavailable_stub,
    get_current_arch,
    get_current_platform,
    validate_target_type,
)
from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import Tier, registry

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# VALID VALUES
# =============================================================================

VALID_NATIVE_BACKENDS = frozenset({"cython", "mypyc", "numba", "nuitka", "pypy"})
VALID_FFI_ABIS = frozenset({"c", "stdcall", "fastcall", "thiscall", "vectorcall"})
VALID_ARCHITECTURES = frozenset({"x86_64", "arm64", "x86", "arm", "wasm32", "riscv64"})
VALID_OPERATING_SYSTEMS = frozenset(
    {
        "windows",
        "linux",
        "macos",
        "ios",
        "android",
        "web",
        "playstation",
        "xbox",
        "switch",
    }
)
VALID_BACKEND_TYPES = frozenset(
    {"graphics", "audio", "input", "network", "physics", "compute"}
)
VALID_GRAPHICS_APIS = frozenset(
    {"vulkan", "d3d12", "d3d11", "metal", "opengl", "webgpu", "platform_native"}
)
VALID_AUDIO_APIS = frozenset(
    {"wasapi", "coreaudio", "alsa", "pulseaudio", "aaudio", "webaudio"}
)
VALID_CAPABILITIES = frozenset(
    {
        # GPU Features
        "ray_tracing",
        "mesh_shaders",
        "bindless",
        "variable_rate_shading",
        "64bit_atomics",
        "16bit_storage",
        "compute_shader",
        "geometry_shader",
        "tessellation",
        "indirect_draw",
        "multi_draw_indirect",
        # CPU Features
        "avx",
        "avx2",
        "avx512",
        "sse4",
        "neon",
        "simd128",
        # Memory Features
        "shared_memory",
        "unified_memory",
        "resizable_bar",
    }
)


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class NativeConfig:
    """Configuration for @native decorator."""

    backend: str = "cython"
    nogil: bool = False
    boundscheck: bool = False
    wraparound: bool = False
    cdivision: bool = True
    infer_types: bool = True


@dataclass(frozen=True)
class FFIConfig:
    """Configuration for @ffi decorator."""

    lib: str = ""
    abi: str = "c"
    header: Optional[str] = None
    link_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetConfig:
    """Configuration for @target decorator."""

    arch: frozenset[str] = frozenset({"x86_64", "arm64"})
    os: Optional[frozenset[str]] = None


@dataclass(frozen=True)
class BackendConfig:
    """Configuration for @backend decorator."""

    type: str = "graphics"
    api: str = "vulkan"
    fallback: Optional[str] = None


@dataclass(frozen=True)
class CapabilityConfig:
    """Configuration for @capability decorator."""

    requires: frozenset[str] = frozenset()
    fallback: Optional[Callable[..., Any]] = None


@dataclass(frozen=True)
class PlatformConfig:
    """Configuration for @platform decorator."""

    include: Optional[frozenset[str]] = None
    exclude: Optional[frozenset[str]] = None


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_native(backend: str = "cython", **_: Any) -> None:
    if backend not in VALID_NATIVE_BACKENDS:
        raise ValueError(
            f"@native: invalid backend '{backend}'. "
            f"Valid backends: {sorted(VALID_NATIVE_BACKENDS)}"
        )


def _validate_ffi(lib: str = "", abi: str = "c", **_: Any) -> None:
    if not lib:
        raise ValueError("@ffi: 'lib' parameter is required")
    if abi not in VALID_FFI_ABIS:
        raise ValueError(
            f"@ffi: invalid abi '{abi}'. Valid ABIs: {sorted(VALID_FFI_ABIS)}"
        )


def _validate_target(
    arch: Any = frozenset({"x86_64", "arm64"}),
    os: Any = None,
    **_: Any,
) -> None:
    arch_set = frozenset(arch) if arch else frozenset()
    invalid_arch = arch_set - VALID_ARCHITECTURES
    if invalid_arch:
        raise ValueError(
            f"@target: invalid architecture(s): {sorted(invalid_arch)}. "
            f"Valid architectures: {sorted(VALID_ARCHITECTURES)}"
        )
    if os is not None:
        os_set = frozenset(os)
        invalid_os = os_set - VALID_OPERATING_SYSTEMS
        if invalid_os:
            raise ValueError(
                f"@target: invalid OS(s): {sorted(invalid_os)}. "
                f"Valid OSs: {sorted(VALID_OPERATING_SYSTEMS)}"
            )


def _validate_backend(
    type: str = "graphics",
    api: str = "vulkan",
    fallback: Optional[str] = None,
    **_: Any,
) -> None:
    if type not in VALID_BACKEND_TYPES:
        raise ValueError(
            f"@backend: invalid type '{type}'. "
            f"Valid types: {sorted(VALID_BACKEND_TYPES)}"
        )
    if type == "graphics":
        valid_apis = VALID_GRAPHICS_APIS
    elif type == "audio":
        valid_apis = VALID_AUDIO_APIS
    else:
        valid_apis = frozenset({api})

    if api not in valid_apis and type in ("graphics", "audio"):
        raise ValueError(
            f"@backend: invalid API '{api}' for type '{type}'. "
            f"Valid APIs: {sorted(valid_apis)}"
        )
    if fallback and type in ("graphics", "audio") and fallback not in valid_apis:
        raise ValueError(
            f"@backend: invalid fallback '{fallback}' for type '{type}'. "
            f"Valid APIs: {sorted(valid_apis)}"
        )


def _validate_capability(requires: Any = frozenset(), **_: Any) -> None:
    requires_set = frozenset(requires)
    invalid_caps = requires_set - VALID_CAPABILITIES
    if invalid_caps:
        raise ValueError(
            f"@capability: invalid capability(ies): {sorted(invalid_caps)}. "
            f"Valid capabilities: {sorted(VALID_CAPABILITIES)}"
        )


def _validate_platform(include: Any = None, exclude: Any = None, **_: Any) -> None:
    if include is not None and exclude is not None:
        raise ValueError(
            "@platform: cannot specify both 'include' and 'exclude'. Choose one."
        )
    if include is None and exclude is None:
        raise ValueError("@platform: must specify either 'include' or 'exclude'.")
    all_platforms = frozenset(include or exclude or ())
    invalid = all_platforms - VALID_OPERATING_SYSTEMS
    if invalid:
        raise ValueError(
            f"@platform: invalid platform(s): {sorted(invalid)}. "
            f"Valid platforms: {sorted(VALID_OPERATING_SYSTEMS)}"
        )


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _native_steps(params: dict[str, Any]) -> list[Step]:
    backend = params.get("backend", "cython")
    return [
        Step(Op.TAG, {"key": "native", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "native_config",
                "value": NativeConfig(
                    backend=backend,
                    nogil=params.get("nogil", False),
                    boundscheck=params.get("boundscheck", False),
                    wraparound=params.get("wraparound", False),
                    cdivision=params.get("cdivision", True),
                    infer_types=params.get("infer_types", True),
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "compilation"}),
    ]


def _ffi_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "ffi", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "ffi_config",
                "value": FFIConfig(
                    lib=params.get("lib", ""),
                    abi=params.get("abi", "c"),
                    header=params.get("header"),
                    link_args=params.get("link_args", ()),
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "compilation"}),
    ]


def _target_steps(params: dict[str, Any]) -> list[Step]:
    arch = frozenset(params.get("arch", {"x86_64", "arm64"}))
    os_val = params.get("os")
    os_set = frozenset(os_val) if os_val else None

    current_arch = get_current_arch()
    current_os = get_current_platform()
    arch_available = current_arch in arch
    os_available = os_set is None or current_os in os_set
    available = arch_available and os_available

    return [
        Step(Op.TAG, {"key": "target", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "target_config",
                "value": TargetConfig(arch=arch, os=os_set),
            },
        ),
        Step(Op.TAG, {"key": "target_available", "value": available}),
        Step(Op.REGISTER, {"registry": "compilation"}),
    ]


def _unsafe_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "unsafe", "value": True}),
        Step(Op.REGISTER, {"registry": "compilation"}),
    ]


def _backend_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "backend", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "backend_config",
                "value": BackendConfig(
                    type=params.get("type", "graphics"),
                    api=params.get("api", "vulkan"),
                    fallback=params.get("fallback"),
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "compilation"}),
    ]


def _capability_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "capability", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "capability_config",
                "value": CapabilityConfig(
                    requires=frozenset(params.get("requires", ())),
                    fallback=params.get("fallback"),
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "compilation"}),
    ]


def _platform_steps(params: dict[str, Any]) -> list[Step]:
    include = params.get("include")
    exclude = params.get("exclude")
    include_set = frozenset(include) if include else None
    exclude_set = frozenset(exclude) if exclude else None

    current = get_current_platform()
    if include_set:
        available = current in include_set
    elif exclude_set:
        available = current not in exclude_set
    else:
        available = True

    return [
        Step(Op.TAG, {"key": "platform", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "platform_config",
                "value": PlatformConfig(include=include_set, exclude=exclude_set),
            },
        ),
        Step(Op.TAG, {"key": "platform_available", "value": available}),
        Step(Op.REGISTER, {"registry": "compilation"}),
    ]


# =============================================================================
# AFTER-STEPS (domain behavior that isn't an Op)
# =============================================================================


def _after_native(target: Any, params: dict[str, Any]) -> Any:
    """Attach underscore attributes for backward compat."""
    validate_target_type(target, "native", ("function", "class"))
    config = target._tags.get("native_config")
    target._native = True
    target._native_config = config
    return None


def _after_ffi(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "ffi", ("function",))
    config = target._tags.get("ffi_config")
    target._ffi = True
    target._ffi_config = config
    return None


def _after_target(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "target", ("function", "class"))
    config = target._tags.get("target_config")
    available = target._tags.get("target_available", True)
    target._target = True
    target._target_config = config
    target._target_available = available

    # If not available, replace function with stub
    if not available and callable(target) and not isinstance(target, type):
        arch_set = config.arch
        os_set = config.os
        current_arch = get_current_arch()
        current_os = get_current_platform()

        reason_parts = []
        if current_arch not in arch_set:
            reason_parts.append(
                f"requires arch {sorted(arch_set)}, current is {current_arch}"
            )
        if os_set and current_os not in os_set:
            reason_parts.append(
                f"requires OS {sorted(os_set)}, current is {current_os}"
            )

        stub = create_unavailable_stub(
            target_name=getattr(target, "__name__", str(target)),
            decorator_name="target",
            reason="; ".join(reason_parts),
        )
        # Preserve attributes on stub
        for attr in (
            "_tags",
            "_applied_steps",
            "_applied_decorators",
            "_target",
            "_target_config",
            "_target_available",
            "_registries",
        ):
            val = getattr(target, attr, None)
            if val is not None:
                setattr(stub, attr, val)
        stub._target_available = False
        return stub

    return None


def _after_unsafe(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "unsafe", ("function", "class"))
    target._unsafe = True
    return None


def _after_backend(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "backend", ("function", "class"))
    config = target._tags.get("backend_config")
    target._backend = True
    target._backend_config = config
    return None


def _after_capability(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "capability", ("function", "class"))
    config = target._tags.get("capability_config")
    target._capability = True
    target._capability_config = config
    return None


def _after_platform(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "platform", ("function", "class"))
    config = target._tags.get("platform_config")
    available = target._tags.get("platform_available", True)
    target._platform = True
    target._platform_config = config
    target._platform_available = available

    # If not available, replace function with stub
    if not available and callable(target) and not isinstance(target, type):
        include_set = config.include
        current = get_current_platform()
        if include_set:
            reason = f"requires platform {sorted(include_set)}, current is '{current}'"
        else:
            reason = f"excluded on platform '{current}'"

        stub = create_unavailable_stub(
            target_name=getattr(target, "__name__", str(target)),
            decorator_name="platform",
            reason=reason,
        )
        for attr in (
            "_tags",
            "_applied_steps",
            "_applied_decorators",
            "_platform",
            "_platform_config",
            "_platform_available",
            "_registries",
        ):
            val = getattr(target, attr, None)
            if val is not None:
                setattr(stub, attr, val)
        stub._platform_available = False
        return stub

    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================


native = make_decorator(
    name="native",
    steps=_native_steps,
    doc="Mark function/class for ahead-of-time compilation to native code.",
    validate=_validate_native,
    after_steps=_after_native,
)

ffi = make_decorator(
    name="ffi",
    steps=_ffi_steps,
    doc="Mark function as FFI binding to native library.",
    validate=_validate_ffi,
    after_steps=_after_ffi,
)

target = make_decorator(
    name="target",
    steps=_target_steps,
    doc="Specify target architectures and operating systems.",
    validate=_validate_target,
    after_steps=_after_target,
)

unsafe = make_decorator(
    name="unsafe",
    steps=_unsafe_steps,
    doc="Mark code as containing unsafe operations.",
    after_steps=_after_unsafe,
)

backend = make_decorator(
    name="backend",
    steps=_backend_steps,
    doc="Specify backend type and API for a subsystem implementation.",
    validate=_validate_backend,
    after_steps=_after_backend,
)

capability = make_decorator(
    name="capability",
    steps=_capability_steps,
    doc="Specify hardware capability requirements with optional fallback.",
    validate=_validate_capability,
    after_steps=_after_capability,
)

platform = make_decorator(
    name="platform",
    steps=_platform_steps,
    doc="Specify platform inclusion or exclusion for code availability.",
    validate=_validate_platform,
    after_steps=_after_platform,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================
# Register decorator *definitions* with the registry for discoverability.
# This is separate from the Op-based REGISTER step which tracks applied instances.

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("native", native, ("function", "class")),
    ("ffi", ffi, ("function",)),
    ("target", target, ("function", "class")),
    ("unsafe", unsafe, ("function", "class")),
    ("backend", backend, ("function", "class")),
    ("capability", capability, ("function", "class")),
    ("platform", platform, ("function", "class")),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.COMPILATION,
            func=_func,
            unique=True,
            foundation=True,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.COMPILATION].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Decorators
    "native",
    "ffi",
    "target",
    "unsafe",
    "backend",
    "capability",
    "platform",
    # Configuration classes
    "NativeConfig",
    "FFIConfig",
    "TargetConfig",
    "BackendConfig",
    "CapabilityConfig",
    "PlatformConfig",
    # Valid values
    "VALID_NATIVE_BACKENDS",
    "VALID_FFI_ABIS",
    "VALID_ARCHITECTURES",
    "VALID_OPERATING_SYSTEMS",
    "VALID_BACKEND_TYPES",
    "VALID_GRAPHICS_APIS",
    "VALID_AUDIO_APIS",
    "VALID_CAPABILITIES",
]
