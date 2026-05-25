"""
GPU decorators — built from Ops.

These decorators mark classes and functions for GPU execution: buffers,
kernels, shaders, render passes, compute dispatches, etc.

Every decorator here is a named list of Steps, created by make_decorator.
The Steps do the real work. The decorators are just configuration.

Decorators:
    @gpu_buffer     - Mark class as GPU buffer with usage flags
    @gpu_kernel     - Mark class as GPU compute kernel
    @gpu_struct     - Mark class as GPU-compatible struct
    @bind_group     - Bind group index for shader resources
    @dispatch       - Mark function as compute dispatch
    @shader         - Mark class/function as shader stage
    @render_pass    - Mark class as render pass configuration
    @async_compute  - Mark for asynchronous compute execution
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.base import validate_target_type
from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

T = TypeVar("T")


# =============================================================================
# VALID VALUES
# =============================================================================

VALID_BUFFER_USAGE = frozenset({
    "vertex", "index", "uniform", "storage", "indirect",
    "map_read", "map_write", "copy_src", "copy_dst",
})
VALID_GPU_BACKENDS = frozenset({"wgpu", "cuda", "metal"})
VALID_SHADER_STAGES = frozenset({"vertex", "fragment", "compute"})
VALID_MSAA_SAMPLES = frozenset({1, 2, 4, 8, 16})


# =============================================================================
# WGPU USAGE FLAG MAPPING
# =============================================================================

# Abstract usage flag names -> wgpu-native BufferUsage bit values.
# These match wgpu-py (the Python wgpu library) and the WebGPU spec.
_WGPU_USAGE_FLAGS: dict[str, int] = {
    "vertex": 0x0020,
    "index": 0x0010,
    "uniform": 0x0040,
    "storage": 0x0080,
    "indirect": 0x0100,
    "map_read": 0x0001,
    "map_write": 0x0002,
    "copy_src": 0x0004,
    "copy_dst": 0x0008,
}


def _resolve_wgpu_usage_flags(usage: frozenset[str]) -> int:
    """Convert a set of abstract usage names to a combined wgpu usage bitmask.

    Args:
        usage: Set of usage flag names (e.g. ``{"vertex", "storage"}``).

    Returns:
        Integer bitmask of OR'd wgpu-native flag values.
    """
    flags = 0
    for name in usage:
        flag = _WGPU_USAGE_FLAGS.get(name)
        if flag is not None:
            flags |= flag
        else:
            warnings.warn(
                f"Unknown wgpu usage flag '{name}' — ignoring it. "
                f"Valid flags: {sorted(_WGPU_USAGE_FLAGS)}",
                stacklevel=2,
            )
    return flags


# =============================================================================
# WGSL TYPE MARKERS
# =============================================================================


class Vec2:
    """WGSL vec2<f32>: 2 floats, align=8, size=8."""
    _wgsl_name = "vec2<f32>"
    _size = 8
    _alignment = 8


class Vec3:
    """WGSL vec3<f32>: 3 floats, align=16, size=12 (stride=16 in arrays)."""
    _wgsl_name = "vec3<f32>"
    _size = 12
    _alignment = 16


class Vec4:
    """WGSL vec4<f32>: 4 floats, align=16, size=16."""
    _wgsl_name = "vec4<f32>"
    _size = 16
    _alignment = 16


class Mat4:
    """WGSL mat4x4<f32>: 4 vec4 columns, align=16, size=64."""
    _wgsl_name = "mat4x4<f32>"
    _size = 64
    _alignment = 16





class f32:
    """WGSL f32: 4-byte float, size=4, align=4.

    Supports subscript for fixed-size arrays: f32[4] -> Annotated[float, 4].
    """
    _size = 4
    _alignment = 4

    def __class_getitem__(cls, count: int) -> Any:
        from typing import Annotated
        if not isinstance(count, int):
            raise TypeError(
                f"f32[N] array size must be an int, got {type(count).__name__}"
            )
        if count <= 0:
            raise ValueError(
                f"f32[N] array size must be > 0, got {count}"
            )
        return Annotated[float, count]

# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class GpuBufferConfig:
    """Configuration for @gpu_buffer decorator."""

    usage: frozenset[str] = frozenset({"storage"})
    mapped: bool = False


@dataclass(frozen=True)
class GpuKernelConfig:
    """Configuration for @gpu_kernel decorator."""

    workgroup_size: tuple[int, int, int] = (64, 1, 1)
    backend: str = "wgpu"


@dataclass(frozen=True)
class ShaderConfig:
    """Configuration for @shader decorator."""

    stage: str = "compute"
    entry: str = "main"


@dataclass(frozen=True)
class RenderPassConfig:
    """Configuration for @render_pass decorator."""

    color_attachments: int = 1
    depth: bool = True
    msaa: int = 1


@dataclass(frozen=True)
class WgpuBufferAllocation:
    """Describes a wgpu buffer allocation derived from a ``@gpu_buffer`` class.

    Attributes:
        size: Byte size of the buffer (computed from struct layout).
        usage: Combined wgpu usage bitmask.
        mapped: Whether the buffer should be created as mapped.
        label: Optional debug label.
    """

    size: int
    usage: int
    mapped: bool = False
    label: str = ""


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_gpu_buffer(usage: Any = {"storage"}, **_: Any) -> None:
    usage_set = frozenset(usage) if isinstance(usage, (set, frozenset, list, tuple)) else frozenset({usage})
    invalid_usage = usage_set - VALID_BUFFER_USAGE
    if invalid_usage:
        raise ValueError(
            f"@gpu_buffer: invalid usage flag(s): {sorted(invalid_usage)}. "
            f"Valid usage flags: {sorted(VALID_BUFFER_USAGE)}"
        )


def _validate_gpu_kernel(backend: str = "wgpu", **_: Any) -> None:
    if backend not in VALID_GPU_BACKENDS:
        raise ValueError(
            f"@gpu_kernel: invalid backend '{backend}'. "
            f"Valid backends: {sorted(VALID_GPU_BACKENDS)}"
        )


def _validate_bind_group(index: int = 0, **_: Any) -> None:
    if index < 0:
        raise ValueError(f"@bind_group: index must be >= 0, got {index}")


def _validate_shader(stage: str = "compute", **_: Any) -> None:
    if stage not in VALID_SHADER_STAGES:
        raise ValueError(
            f"@shader: invalid stage '{stage}'. "
            f"Valid stages: {sorted(VALID_SHADER_STAGES)}"
        )


def _validate_render_pass(
    color_attachments: int = 1, msaa: int = 1, **_: Any
) -> None:
    if color_attachments < 1:
        raise ValueError(
            f"@render_pass: color_attachments must be >= 1, got {color_attachments}"
        )
    if msaa not in VALID_MSAA_SAMPLES:
        raise ValueError(
            f"@render_pass: msaa must be power of 2 (1, 2, 4, 8, 16), got {msaa}"
        )


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _gpu_buffer_steps(params: dict[str, Any]) -> list[Step]:
    usage = params.get("usage", {"storage"})
    usage_set = frozenset(usage) if isinstance(usage, (set, frozenset, list, tuple)) else frozenset({usage})
    mapped = params.get("mapped", False)

    return [
        Step(Op.TAG, {"key": "gpu_buffer", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "gpu_buffer_config",
                "value": GpuBufferConfig(usage=usage_set, mapped=mapped),
            },
        ),
        Step(Op.REGISTER, {"registry": "gpu"}),
        Step(Op.DESCRIBE, {}),
    ]


def _gpu_kernel_steps(params: dict[str, Any]) -> list[Step]:
    workgroup_size = params.get("workgroup_size", (64, 1, 1))
    backend = params.get("backend", "wgpu")

    return [
        Step(Op.TAG, {"key": "gpu_kernel", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "gpu_kernel_config",
                "value": GpuKernelConfig(
                    workgroup_size=workgroup_size, backend=backend
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "gpu"}),
    ]


def _gpu_struct_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "gpu_struct", "value": True}),
        Step(Op.REGISTER, {"registry": "gpu"}),
        Step(Op.DESCRIBE, {}),
    ]


def _bind_group_steps(params: dict[str, Any]) -> list[Step]:
    index = params.get("index", 0)

    return [
        Step(Op.TAG, {"key": "bind_group", "value": True}),
        Step(Op.TAG, {"key": "bind_group_index", "value": index}),
        Step(Op.REGISTER, {"registry": "gpu"}),
    ]


def _dispatch_steps(params: dict[str, Any]) -> list[Step]:
    indirect = params.get("indirect", False)

    return [
        Step(Op.TAG, {"key": "dispatch", "value": True}),
        Step(Op.TAG, {"key": "dispatch_indirect", "value": indirect}),
        Step(Op.REGISTER, {"registry": "gpu"}),
    ]


def _shader_steps(params: dict[str, Any]) -> list[Step]:
    stage = params.get("stage", "compute")
    entry = params.get("entry", "main")

    return [
        Step(Op.TAG, {"key": "shader", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "shader_config",
                "value": ShaderConfig(stage=stage, entry=entry),
            },
        ),
        Step(Op.REGISTER, {"registry": "gpu"}),
    ]


def _render_pass_steps(params: dict[str, Any]) -> list[Step]:
    color_attachments = params.get("color_attachments", 1)
    depth = params.get("depth", True)
    msaa = params.get("msaa", 1)

    return [
        Step(Op.TAG, {"key": "render_pass", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "render_pass_config",
                "value": RenderPassConfig(
                    color_attachments=color_attachments, depth=depth, msaa=msaa
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "gpu"}),
    ]


def _async_compute_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "async_compute", "value": True}),
        Step(Op.REGISTER, {"registry": "gpu"}),
    ]


# =============================================================================
# AFTER-STEPS (domain behavior that isn't an Op)
# =============================================================================


def _after_gpu_buffer(target: Any, params: dict[str, Any]) -> Any:
    """Attach underscore attributes, compute buffer size, and wire wgpu flags."""
    validate_target_type(target, "gpu_buffer", ("class",))
    tags = getattr(target, "_tags", {})
    config = tags.get("gpu_buffer_config")
    target._gpu_buffer = True

    # H1: auto-add COPY_DST when indirect is used — indirect draw/dispatch
    # buffers must be writable by the CPU to populate commands.
    usage = config.usage
    if "indirect" in usage:
        usage = frozenset(usage | {"copy_dst"})
    target._gpu_usage = usage
    target._gpu_mapped = config.mapped

    # Extract buffer fields from annotations
    schema = getattr(target, "_schema", {})
    target._gpu_buffer_fields = list(schema.keys())

    # Compute buffer layout (reuse struct layout logic) — same WGSL rules
    layout = _compute_gpu_struct_layout(
        getattr(target, "__annotations__", {})
    )
    target._gpu_buffer_size = layout["size"]
    target._gpu_buffer_alignment = layout["align"]
    target._gpu_buffer_layout = layout["fields"]

    # Resolve wgpu usage bitmask
    target._gpu_wgpu_usage = _resolve_wgpu_usage_flags(usage)

    return None


def _after_gpu_kernel(target: Any, params: dict[str, Any]) -> Any:
    """Attach underscore attributes for kernel configuration."""
    validate_target_type(target, "gpu_kernel", ("class",))
    config = target._tags.get("gpu_kernel_config")
    target._gpu_kernel = True
    target._workgroup_size = config.workgroup_size
    target._gpu_backend = config.backend
    return None


def _round_up(alignment: int, value: int) -> int:
    """Round *value* up to the next multiple of *alignment*."""
    return ((value + alignment - 1) // alignment) * alignment


_BASE_TYPE_MAP: dict[Any, dict[str, Any]] = {
    float: {"name": "f32", "size": 4, "align": 4},
    "float": {"name": "f32", "size": 4, "align": 4},
    int: {"name": "i32", "size": 4, "align": 4},
    "int": {"name": "i32", "size": 4, "align": 4},
    bool: {"name": "bool", "size": 4, "align": 4},
    "bool": {"name": "bool", "size": 4, "align": 4},
    bytes: {"name": "u32", "size": 4, "align": 4},
    str: {"name": "u32", "size": 4, "align": 4},
}


def _get_gpu_type_info(field_type: Any) -> dict[str, Any]:
    """Resolve a Python type annotation to WGSL type info.

    Returns dict with keys: ``name``, ``size``, ``align``.
    Resolution order:

    1. Direct base types (float, int, bool) via ``_BASE_TYPE_MAP``.
    2. ``typing.Annotated[T, N]`` — fixed-size arrays (checked early
       because Python 3.14+ copies origin attributes like ``_size``
       onto the ``_AnnotatedAlias`` wrapper, which would falsely match
       duck-typed checks below).
    3. Duck-typed WGSL classes (Vec2, Vec3, Vec4, Mat4, custom).
    4. Nested ``@gpu_struct`` classes.
    5. Fallback: treat as 4-byte scalar.
    """
    # --- 1. Direct base-type lookup ---
    if field_type in _BASE_TYPE_MAP:
        return dict(_BASE_TYPE_MAP[field_type])

    type_name = getattr(field_type, "__name__", str(field_type))
    if type_name in _BASE_TYPE_MAP:
        return dict(_BASE_TYPE_MAP[type_name])

    # --- 2. typing.Annotated[T, N] (before duck-type check) ---
    metadata = getattr(field_type, "__metadata__", None)
    origin = getattr(field_type, "__origin__", None)
    if metadata is not None and origin is not None:
        for meta in metadata:
            if isinstance(meta, int):
                base_info = _get_gpu_type_info(origin)
                return _array_type_info(base_info, meta)

    args = getattr(field_type, "__args__", None)
    if args is not None and len(args) >= 2:
        base_info = _get_gpu_type_info(args[0])
        count = None
        for candidate in args[1:]:
            if isinstance(candidate, int):
                count = candidate
                break
        if count is not None:
            return _array_type_info(base_info, count)

    # --- 3. Duck-typed WGSL classes ---
    wgsl_size = getattr(field_type, "_size", None)
    wgsl_align = getattr(field_type, "_alignment", None)
    if wgsl_size is not None and wgsl_align is not None:
        wgsl_name = getattr(field_type, "_wgsl_name", type_name)
        return {"name": wgsl_name, "size": wgsl_size, "align": wgsl_align}

    # --- 4. Nested @gpu_struct ---
    if getattr(field_type, "_gpu_struct", False):
        return {
            "name": type_name,
            "size": field_type._gpu_struct_size,
            "align": field_type._gpu_struct_alignment,
        }

    # --- 5. Fallback ---
    return {"name": type_name, "size": 4, "align": 4}


def _array_type_info(base_info: dict[str, Any], count: int) -> dict[str, Any]:
    """Compute WGSL array<T,N> layout info.

    WGSL storage-buffer rules:
      - element_stride = roundUp(alignof(T), sizeof(T))
      - array alignment = max(16, alignof(T))
      - total size = element_stride * count
    """
    if count <= 0:
        raise ValueError(
            f"array<T,N> count must be > 0, got {count}"
        )
    elem_size = base_info["size"]
    elem_align = base_info["align"]
    elem_stride = _round_up(elem_align, elem_size)
    array_align = max(16, elem_align)
    total_size = elem_stride * count

    return {
        "name": f"array<{base_info['name']},{count}>",
        "size": total_size,
        "align": array_align,
        "stride": elem_stride,
    }


def _compute_gpu_struct_layout(schema: dict[str, Any]) -> dict[str, Any]:
    """Compute full WGSL-compatible struct layout from annotations.

    Returns::
        {
            "size": <total byte size>,
            "align": <struct alignment>,
            "fields": [
                {"name": str, "type": str, "offset": int, "size": int, "align": int},
                ...
            ]
        }
    """
    fields: list[dict[str, Any]] = []
    offset = 0
    struct_align = 4

    for field_name, field_type in schema.items():
        info = _get_gpu_type_info(field_type)
        field_align = info["align"]
        field_size = info["size"]

        aligned_offset = _round_up(field_align, offset)

        entry = {
            "name": field_name,
            "type": info["name"],
            "offset": aligned_offset,
            "size": field_size,
            "align": field_align,
        }
        if "stride" in info:
            entry["stride"] = info["stride"]
        fields.append(entry)

        offset = aligned_offset + field_size
        if field_align > struct_align:
            struct_align = field_align

    # Single-field non-array struct: use content size (no trailing pad).
    # Arrays always round up to alignment (WGSL storage-buffer rule).
    # Multi-field struct: round up to alignment per WGSL rules.
    if len(fields) == 0:
        total_size = offset
    elif len(fields) == 1 and "stride" not in fields[0]:
        total_size = offset
    else:
        total_size = _round_up(struct_align, offset)

    return {"size": total_size, "align": struct_align, "fields": fields}


def _after_gpu_struct(target: Any, params: dict[str, Any]) -> Any:
    """Compute struct size and alignment from annotations.

    Handles: float, int, bool, Vec2, Vec3, Vec4, Mat4,
    nested ``@gpu_struct`` classes, and ``Annotated[T, N]`` arrays.

    Reads ``__annotations__`` directly (not ``_schema``) because
    ``get_type_hints()`` *strips* ``Annotated`` metadata on Python 3.14+,
    which would make fixed-size arrays undetectable.
    """
    validate_target_type(target, "gpu_struct", ("class",))
    target._gpu_struct = True

    # Use __annotations__ directly -- get_type_hints strips Annotated
    schema = getattr(target, "__annotations__", {})
    layout = _compute_gpu_struct_layout(schema)

    target._gpu_struct_fields = layout["fields"]
    target._gpu_struct_size = layout["size"]
    target._gpu_struct_alignment = layout["align"]
    return None


def _after_bind_group(target: Any, params: dict[str, Any]) -> Any:
    """Attach bind group index."""
    validate_target_type(target, "bind_group", ("class",))
    index = target._tags.get("bind_group_index", 0)
    target._bind_group = True
    target._bind_group_index = index
    return None


def _after_dispatch(target: Any, params: dict[str, Any]) -> Any:
    """Attach dispatch configuration."""
    validate_target_type(target, "dispatch", ("function", "class"))
    indirect = target._tags.get("dispatch_indirect", False)
    target._dispatch = True
    target._dispatch_indirect = indirect
    return None


def _after_shader(target: Any, params: dict[str, Any]) -> Any:
    """Attach shader configuration."""
    validate_target_type(target, "shader", ("function", "class"))
    config = target._tags.get("shader_config")
    target._shader = True
    target._shader_stage = config.stage
    target._shader_entry = config.entry
    return None


def _after_render_pass(target: Any, params: dict[str, Any]) -> Any:
    """Attach render pass configuration."""
    validate_target_type(target, "render_pass", ("class",))
    config = target._tags.get("render_pass_config")
    target._render_pass = True
    target._render_pass_colors = config.color_attachments
    target._render_pass_depth = config.depth
    target._render_pass_msaa = config.msaa
    return None


def _after_async_compute(target: Any, params: dict[str, Any]) -> Any:
    """Mark for async compute execution."""
    validate_target_type(target, "async_compute", ("class",))
    target._async_compute = True
    return None


# =============================================================================
# WGPU BUFFER ALLOCATION
# =============================================================================


def allocate_wgpu_buffer(
    target: type,
    device: Any,
    label: str | None = None,
) -> Any:
    """Allocate a wgpu buffer from a ``@gpu_buffer`` decorated class.

    Args:
        target: A class decorated with ``@gpu_buffer``.
        device: A wgpu device (must have a ``create_buffer`` method).
            Accepted for API consistency with ``create_wgpu_buffer``, which
            forwards it here before calling ``device.create_buffer``.  This
            function itself does not interact with the device.
        label: Optional debug label for the buffer.

    Returns:
        A ``WgpuBufferAllocation`` describing the allocation (``device.create_buffer``
        must be called separately, or the caller can use the returned descriptor
        to create the buffer manually).

    Raises:
        TypeError: If *target* is not ``@gpu_buffer`` decorated.
        RuntimeError: If the target has no annotated fields (zero-size buffer).

    Example::

        @gpu_buffer(usage={"storage", "copy_src"})
        class ParticleData:
            position: Vec3
            velocity: Vec3
            mass: float

        # Get allocation descriptor
        alloc = allocate_wgpu_buffer(ParticleData, device)

        # Create the actual wgpu buffer
        buffer = device.create_buffer(
            size=alloc.size,
            usage=alloc.usage,
            mapped_at_creation=alloc.mapped,
            label=alloc.label,
        )
    """
    if not getattr(target, "_gpu_buffer", False):
        raise TypeError(
            f"allocate_wgpu_buffer: {target.__name__} is not decorated with @gpu_buffer"
        )

    size = getattr(target, "_gpu_buffer_size", 0)
    if size <= 0:
        raise RuntimeError(
            f"allocate_wgpu_buffer: {target.__name__} has zero buffer size "
            "(no annotated fields or empty struct)"
        )

    usage = getattr(target, "_gpu_wgpu_usage", 0)
    mapped = getattr(target, "_gpu_mapped", False)
    label = label or target.__name__

    return WgpuBufferAllocation(
        size=size,
        usage=usage,
        mapped=mapped,
        label=label,
    )


def create_wgpu_buffer(
    target: type,
    device: Any,
    label: str | None = None,
) -> Any:
    """Create an actual wgpu buffer from a ``@gpu_buffer`` decorated class.

    This is a convenience wrapper around ``allocate_wgpu_buffer`` that also
    calls ``device.create_buffer`` with the resolved parameters.

    Args:
        target: A class decorated with ``@gpu_buffer``.
        device: A wgpu device (must have a ``create_buffer`` method).
        label: Optional debug label for the buffer.

    Returns:
        The wgpu buffer object returned by ``device.create_buffer``.

    Raises:
        TypeError: If *target* is not ``@gpu_buffer`` decorated.
        RuntimeError: If the target has no annotated fields.

    Example::

        @gpu_buffer(usage={"uniform"})
        class UniformData:
            color: Vec4
            time: float

        buffer = create_wgpu_buffer(UniformData, device)
    """
    alloc = allocate_wgpu_buffer(target, device, label=label)
    return device.create_buffer(
        size=alloc.size,
        usage=alloc.usage,
        mapped_at_creation=alloc.mapped,
        label=alloc.label,
    )


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================


gpu_buffer = make_decorator(
    name="gpu_buffer",
    steps=_gpu_buffer_steps,
    doc="Mark class as GPU buffer with specified usage flags.",
    validate=_validate_gpu_buffer,
    after_steps=_after_gpu_buffer,
)

gpu_kernel = make_decorator(
    name="gpu_kernel",
    steps=_gpu_kernel_steps,
    doc="Mark class as GPU compute kernel with workgroup size and backend.",
    validate=_validate_gpu_kernel,
    after_steps=_after_gpu_kernel,
)

gpu_struct = make_decorator(
    name="gpu_struct",
    steps=_gpu_struct_steps,
    doc="Mark class as GPU-compatible struct with automatic size/alignment computation.",
    after_steps=_after_gpu_struct,
)

bind_group = make_decorator(
    name="bind_group",
    steps=_bind_group_steps,
    doc="Specify bind group index for shader resource binding.",
    validate=_validate_bind_group,
    after_steps=_after_bind_group,
)

dispatch = make_decorator(
    name="dispatch",
    steps=_dispatch_steps,
    doc="Mark function/class as compute dispatch with optional indirect dispatch.",
    after_steps=_after_dispatch,
)

shader = make_decorator(
    name="shader",
    steps=_shader_steps,
    doc="Mark function/class as shader with specified stage and entry point.",
    validate=_validate_shader,
    after_steps=_after_shader,
)

render_pass = make_decorator(
    name="render_pass",
    steps=_render_pass_steps,
    doc="Mark class as render pass with color attachments, depth, and MSAA configuration.",
    validate=_validate_render_pass,
    after_steps=_after_render_pass,
)

async_compute = make_decorator(
    name="async_compute",
    steps=_async_compute_steps,
    doc="Mark class for asynchronous compute execution.",
    after_steps=_after_async_compute,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("gpu_buffer", gpu_buffer, ("class",)),
    ("gpu_kernel", gpu_kernel, ("class",)),
    ("gpu_struct", gpu_struct, ("class",)),
    ("bind_group", bind_group, ("class",)),
    ("dispatch", dispatch, ("function", "class")),
    ("shader", shader, ("function", "class")),
    ("render_pass", render_pass, ("class",)),
    ("async_compute", async_compute, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.GPU,
            func=_func,
            unique=True,
            foundation=True,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.GPU].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Decorators
    "gpu_buffer",
    "gpu_kernel",
    "gpu_struct",
    "bind_group",
    "dispatch",
    "shader",
    "render_pass",
    "async_compute",
    # Configuration classes
    "GpuBufferConfig",
    "GpuKernelConfig",
    "ShaderConfig",
    "RenderPassConfig",
    "WgpuBufferAllocation",
    # WGSL type markers
    "Vec2",
    "Vec3",
    "Vec4",
    "Mat4",
    "f32",
    # Helpers
    "_round_up",
    "_get_gpu_type_info",
    "_compute_gpu_struct_layout",
    "_array_type_info",
    "_resolve_wgpu_usage_flags",
    # WGPU buffer allocation
    "allocate_wgpu_buffer",
    "create_wgpu_buffer",
    # Valid values
    "VALID_BUFFER_USAGE",
    "VALID_GPU_BACKENDS",
    "VALID_SHADER_STAGES",
    "VALID_MSAA_SAMPLES",
]

class f32:
    """WGSL f32: 4-byte float, size=4, align=4.

    Supports subscript for fixed-size arrays: f32[4] -> Annotated[float, 4].
    """
    _size = 4
    _alignment = 4

    def __class_getitem__(cls, count: int) -> Any:
        from typing import Annotated
        return Annotated[float, count]

