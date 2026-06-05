"""
Core Frame Graph implementation.

This module implements the main FrameGraph class as specified in
RENDERING_CONTEXT.md Section 6.1.

Frame Graph (from spec):
"Render pass declaration, resource aliasing, automatic barrier insertion,
dependency scheduling, async compute scheduling, unused pass culling"

Pass Scheduling (from spec):
"Declare passes -> Build dependency graph -> Cull unused passes
 -> Schedule async compute -> Insert barriers -> Execute"

Frame Graph API (from spec example):
    fg = FrameGraph()
    gbuffer = fg.add_pass("GBuffer", type="graphics")
    gbuffer.write(albedo_rt, normal_rt, depth_rt)
    ...
    fg.compile()  # Dependency analysis, barrier insertion, resource aliasing
    fg.execute()  # Run all passes in order
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import json

from .async_scheduler import AsyncScheduler, ScheduledPass
from .barrier_manager import Barrier, BarrierBatch, BarrierManager
from .context import RHIContext
from .pass_node import (
    ComputePass,
    CopyPass,
    GraphicsPass,
    PassFlags,
    PassNode,
    PassType,
    RayTracingPass,
    create_pass,
)
from .resource_manager import (
    ResourceFormat,
    ResourceHandle,
    ResourceManager,
    ResourceState,
    ResourceType,
)

# ---------------------------------------------------------------------------
# Module-level constant mappings for IR serialization / Rust bridge
# ---------------------------------------------------------------------------

_PASS_TYPE_TO_STR: dict[PassType, str] = {
    PassType.GRAPHICS: "Graphics",
    PassType.COMPUTE: "Compute",
    PassType.COPY: "Copy",
    PassType.RAY_TRACING: "RayTracing",
}

_FORMAT_TO_WGPU: dict[ResourceFormat, str] = {
    ResourceFormat.R8_UNORM: "R8_UNORM",
    ResourceFormat.R8G8B8A8_UNORM: "R8G8B8A8_UNORM",
    ResourceFormat.R8G8B8A8_SRGB: "R8G8B8A8_SRGB",
    ResourceFormat.R11G11B10_FLOAT: "R11G11B10_FLOAT",
    ResourceFormat.R16G16B16A16_FLOAT: "R16G16B16A16_FLOAT",
    ResourceFormat.R32G32B32A32_FLOAT: "R32G32B32A32_FLOAT",
    ResourceFormat.R32_FLOAT: "R32_FLOAT",
    ResourceFormat.R32G32_FLOAT: "R32G32_FLOAT",
    ResourceFormat.D24_UNORM_S8_UINT: "D24_UNORM_S8_UINT",
    ResourceFormat.D32_FLOAT: "D32_FLOAT",
    ResourceFormat.D32_FLOAT_S8_UINT: "D32_FLOAT_S8_UINT",
    ResourceFormat.BC1_UNORM: "BC1_UNORM",
    ResourceFormat.BC3_UNORM: "BC3_UNORM",
    ResourceFormat.BC5_UNORM: "BC5_UNORM",
    ResourceFormat.BC6H_FLOAT: "BC6H_FLOAT",
    ResourceFormat.BC7_UNORM: "BC7_UNORM",
}


@dataclass
class CompileError:
    """A non-fatal compilation error or warning produced during frame graph
    compilation.

    Unlike exceptions (which halt compilation), ``CompileError`` records issues
    that the compiler can recover from -- for example, a pass that references
    a non-existent resource handle, or a barrier that violates the expected
    resource state machine.  The caller can inspect these after compilation
    and decide whether to abort execution.
    """

    pass_name: str = ""
    """Name of the pass that triggered the error."""

    phase: str = ""
    """Compiler phase or validation step that produced the error."""

    message: str = ""
    """Human-readable error message."""


@dataclass
class CompilationResult:
    """Results from frame graph compilation."""

    success: bool = True
    """Whether compilation succeeded."""

    error_message: Optional[str] = None
    """Error message if compilation failed."""

    execution_order: list[str] = field(default_factory=list)
    """Pass names in execution order."""

    culled_passes: list[str] = field(default_factory=list)
    """Names of passes that were culled."""

    barrier_count: int = 0
    """Total number of barriers generated."""

    alias_group_count: int = 0
    """Number of resource alias groups."""

    async_pass_count: int = 0
    """Number of passes scheduled for async compute."""

    # ------------------------------------------------------------------
    # T-FG-7.7 fields -- enriched statistics from Rust bridge
    # ------------------------------------------------------------------

    pass_count: int = 0
    """Total number of passes declared before culling."""

    culled_count: int = 0
    """Number of passes eliminated as dead (mirrors ``len(culled_passes)``
    for the Python-only path; populated from Rust ``cull_stats`` when
    the bridge is active)."""

    memory_savings_percent: float = 0.0
    """Memory savings as a percentage of total resource footprint.

    Computed by the Rust side as
    ``(bytes_saved_by_culling + alias_bytes_saved) / total_resource_bytes * 100``.
    Zero when no resources are present or when savings cannot be computed.
    """

    errors: list[CompileError] = field(default_factory=list)
    """Non-fatal compilation errors or warnings produced by the Rust
    compiler phases (T-FG-7.7).  Empty on successful compilation.
    Populated only when the Rust PyO3 bridge is active; the Python-only
    fallback path always produces an empty list."""

    # ------------------------------------------------------------------
    # Bridge deserialization
    # ------------------------------------------------------------------

    @classmethod
    def from_bridge_json(cls, data: dict) -> CompilationResult:
        """Create a ``CompilationResult`` from Rust ``emit_bridge_json()`` output.

        Parses the JSON object returned by the Rust PyO3 bridge
        (``_omega.frame_graph_execute``) and maps its fields to the
        corresponding ``CompilationResult`` attributes.  Performs
        forward-compatible reads so that older bridge outputs (without
        T-FG-7.7 fields) are accepted with default values.

        Args:
            data: Parsed JSON dict from the Rust bridge.  Expected keys:

                - ``passes`` -- list of pass dicts with at least ``"name"``
                  (surviving passes in execution order).
                - ``barriers`` -- list of barrier dicts.
                - ``async_passes`` -- list of async pass entries.
                - ``cull_stats`` -- dict with ``passes_total``,
                  ``culled_pass_count`` /  ``passes_eliminated``, and
                  optionally ``memory_savings_percent``.
                - ``validation`` -- dict with ``"valid"`` bool.
                - ``errors`` -- optional list of ``CompileError`` dicts
                  (T-FG-7.7), each with ``pass_name``, ``phase``,
                  ``message``.

        Returns:
            A new ``CompilationResult`` instance.
        """
        cull_stats = data.get("cull_stats", {})
        passes = data.get("passes", [])
        barriers = data.get("barriers", [])
        async_passes = data.get("async_passes", [])

        # Basic status from the validation sub-object
        validation = data.get("validation", {})
        if isinstance(validation, dict):
            success = validation.get("valid", True)
        else:
            success = True

        # Pass names in execution order (surviving passes only)
        execution_order = [
            p["name"]
            for p in passes
            if isinstance(p, dict) and "name" in p
        ]

        # Aggregate statistics from cull_stats
        pass_count = cull_stats.get("passes_total", len(passes))
        culled_count = cull_stats.get(
            "culled_pass_count",
            cull_stats.get("passes_eliminated", 0),
        )
        async_pass_count = len(async_passes)
        barrier_count = len(barriers)

        # Memory savings percentage (T-FG-7.7)
        memory_savings_percent = cull_stats.get(
            "memory_savings_percent", 0.0
        )

        # Non-fatal compilation errors / warnings (T-FG-7.7)
        errors: list[CompileError] = []
        for err_data in data.get("errors", []):
            if isinstance(err_data, dict):
                errors.append(CompileError(
                    pass_name=err_data.get("pass_name", ""),
                    phase=err_data.get("phase", ""),
                    message=err_data.get("message", ""),
                ))

        return cls(
            success=success,
            execution_order=execution_order,
            pass_count=pass_count,
            culled_count=culled_count,
            async_pass_count=async_pass_count,
            barrier_count=barrier_count,
            memory_savings_percent=memory_savings_percent,
            errors=errors,
        )


class FrameGraph:
    """Core frame graph for render pass management.

    The FrameGraph is the central orchestrator for all rendering work.
    It manages:
    - Pass declaration and dependencies
    - Resource allocation and aliasing
    - Automatic barrier insertion
    - Async compute scheduling
    - Dead code elimination (unused pass culling)

    Per RENDERING_CONTEXT.md Section 6.1:
    "Frame Graph - Render pass declaration, resource aliasing,
     automatic barrier insertion, dependency scheduling,
     async compute scheduling, unused pass culling"

    Usage:
        fg = FrameGraph()

        # Declare resources
        gbuffer_albedo = fg.create_texture("gbuffer_albedo", ...)
        gbuffer_depth = fg.create_texture("gbuffer_depth", ...)

        # Declare passes
        gbuffer_pass = fg.add_graphics_pass("GBuffer")
        gbuffer_pass.add_color_attachment(gbuffer_albedo)
        gbuffer_pass.set_depth_stencil(gbuffer_depth)

        lighting_pass = fg.add_compute_pass("Lighting")
        lighting_pass.read_texture(gbuffer_albedo)
        lighting_pass.read_texture(gbuffer_depth)
        lighting_pass.write_texture(hdr_target)

        # Compile and execute
        fg.compile()
        fg.execute(render_context)
    """

    def __init__(self) -> None:
        """Initialize the frame graph."""
        self._resource_manager = ResourceManager()
        self._barrier_manager = BarrierManager(self._resource_manager)
        self._async_scheduler = AsyncScheduler()

        self._passes: dict[str, PassNode] = {}
        self._pass_order: list[str] = []
        self._compiled = False
        self._compilation_result: Optional[CompilationResult] = None

        self._execution_order: list[PassNode] = []
        self._barrier_batches: list[BarrierBatch] = []
        self._scheduled_passes: list[ScheduledPass] = []

        self._enable_async_compute = True
        self._enable_pass_culling = True
        self._enable_resource_aliasing = True

        # Pass dependency graph (built during compilation)
        self._pass_dependencies: dict[str, list[str]] = {}

    # =========================================================================
    # Resource Creation
    # =========================================================================

    def create_texture(
        self,
        name: str,
        format: ResourceFormat = ResourceFormat.R8G8B8A8_UNORM,
        width: int = 0,
        height: int = 0,
        depth: int = 1,
        mip_levels: int = 1,
        sample_count: int = 1,
        clear_value: Optional[tuple] = None,
    ) -> ResourceHandle:
        """Create a transient texture resource.

        Transient textures are allocated per-frame and can be aliased with
        other transients that don't have overlapping lifetimes.

        Args:
            name: Unique name for this resource.
            format: Pixel format.
            width: Width in pixels (0 = derive from render target).
            height: Height in pixels (0 = derive from render target).
            depth: Depth for 3D textures.
            mip_levels: Number of mipmap levels.
            sample_count: MSAA sample count.
            clear_value: Optional clear value.

        Returns:
            A ResourceHandle for referencing this resource.
        """
        self._invalidate_compilation()
        return self._resource_manager.create_transient(
            name=name,
            format=format,
            width=width,
            height=height,
            depth=depth,
            mip_levels=mip_levels,
            sample_count=sample_count,
            clear_value=clear_value,
        )

    def create_buffer(
        self,
        name: str,
        size_bytes: int,
    ) -> ResourceHandle:
        """Create a transient buffer resource.

        Args:
            name: Unique name for this resource.
            size_bytes: Size of the buffer in bytes.

        Returns:
            A ResourceHandle for referencing this resource.
        """
        self._invalidate_compilation()
        return self._resource_manager.create_buffer(
            name=name,
            size_bytes=size_bytes,
            resource_type=ResourceType.TRANSIENT,
        )

    def create_history_texture(
        self,
        name: str,
        format: ResourceFormat = ResourceFormat.R8G8B8A8_UNORM,
        width: int = 0,
        height: int = 0,
        double_buffered: bool = True,
    ) -> ResourceHandle:
        """Create a history texture persisted across frames.

        History textures are used for temporal effects like TAA,
        motion blur, and GI accumulators.

        Args:
            name: Unique name for this resource.
            format: Pixel format.
            width: Width in pixels.
            height: Height in pixels.
            double_buffered: Whether to maintain two copies.

        Returns:
            A ResourceHandle for referencing this resource.
        """
        self._invalidate_compilation()
        return self._resource_manager.create_history(
            name=name,
            format=format,
            width=width,
            height=height,
            double_buffered=double_buffered,
        )

    def import_external(
        self,
        name: str,
        gpu_resource: Any,
        format: ResourceFormat = ResourceFormat.R8G8B8A8_UNORM,
        width: int = 0,
        height: int = 0,
        is_backbuffer: bool = False,
        initial_state: ResourceState = ResourceState.UNDEFINED,
    ) -> ResourceHandle:
        """Import an external resource (backbuffer, imported texture).

        Args:
            name: Unique name for this resource.
            gpu_resource: The actual GPU resource (platform-specific).
            format: Pixel format.
            width: Width in pixels.
            height: Height in pixels.
            is_backbuffer: True if this is the swap chain backbuffer.
            initial_state: The resource's current state.

        Returns:
            A ResourceHandle for referencing this resource.
        """
        self._invalidate_compilation()
        handle = self._resource_manager.register_external(
            name=name,
            gpu_resource=gpu_resource,
            format=format,
            width=width,
            height=height,
            is_backbuffer=is_backbuffer,
        )

        # Set initial state
        external = self._resource_manager.get_external(name)
        if external:
            external.current_state = initial_state

        return handle

    def get_resource(self, name: str) -> Optional[ResourceHandle]:
        """Get a resource handle by name.

        Args:
            name: The resource name.

        Returns:
            The ResourceHandle, or None if not found.
        """
        return self._resource_manager.get_handle(name)

    # =========================================================================
    # Pass Creation
    # =========================================================================

    def add_pass(
        self,
        name: str,
        pass_type: str = "graphics",
    ) -> PassNode:
        """Add a pass to the frame graph.

        This is the general method for adding any type of pass.
        For convenience, use the type-specific methods instead.

        Args:
            name: Unique name for this pass.
            pass_type: Type of pass ('graphics', 'compute', 'copy', 'raytracing').

        Returns:
            The created pass node.

        Raises:
            ValueError: If the pass name is already used or type is invalid.
        """
        type_map = {
            "graphics": PassType.GRAPHICS,
            "compute": PassType.COMPUTE,
            "copy": PassType.COPY,
            "raytracing": PassType.RAY_TRACING,
            "ray_tracing": PassType.RAY_TRACING,
        }

        if pass_type.lower() not in type_map:
            raise ValueError(f"Unknown pass type: {pass_type}")

        pass_enum = type_map[pass_type.lower()]
        return self._add_pass_internal(name, pass_enum)

    def add_graphics_pass(self, name: str) -> GraphicsPass:
        """Add a graphics (rasterization) pass.

        Args:
            name: Unique name for this pass.

        Returns:
            The created GraphicsPass.
        """
        return self._add_pass_internal(name, PassType.GRAPHICS)

    def add_compute_pass(self, name: str) -> ComputePass:
        """Add a compute dispatch pass.

        Args:
            name: Unique name for this pass.

        Returns:
            The created ComputePass.
        """
        return self._add_pass_internal(name, PassType.COMPUTE)

    def add_copy_pass(self, name: str) -> CopyPass:
        """Add a copy/transfer pass.

        Args:
            name: Unique name for this pass.

        Returns:
            The created CopyPass.
        """
        return self._add_pass_internal(name, PassType.COPY)

    def add_raytracing_pass(self, name: str) -> RayTracingPass:
        """Add a ray tracing pass.

        Args:
            name: Unique name for this pass.

        Returns:
            The created RayTracingPass.
        """
        return self._add_pass_internal(name, PassType.RAY_TRACING)

    def _add_pass_internal(
        self,
        name: str,
        pass_type: PassType,
    ) -> PassNode:
        """Internal method to create and register a pass.

        Args:
            name: The pass name.
            pass_type: The type of pass.

        Returns:
            The created pass node.

        Raises:
            ValueError: If the pass name is already used.
        """
        if name in self._passes:
            raise ValueError(f"Pass '{name}' already exists")

        self._invalidate_compilation()

        pass_node = create_pass(name, pass_type)
        self._passes[name] = pass_node
        self._pass_order.append(name)

        return pass_node

    def get_pass(self, name: str) -> Optional[PassNode]:
        """Get a pass by name.

        Args:
            name: The pass name.

        Returns:
            The PassNode, or None if not found.
        """
        return self._passes.get(name)

    def remove_pass(self, name: str) -> bool:
        """Remove a pass from the frame graph.

        Args:
            name: The pass name.

        Returns:
            True if the pass was removed, False if not found.
        """
        if name not in self._passes:
            return False

        self._invalidate_compilation()
        del self._passes[name]
        self._pass_order.remove(name)
        return True

    # =========================================================================
    # IR Serialization (T-FG-1.6) -- Python -> Rust bridge
    # =========================================================================

    def _format_to_wgpu_str(self, fmt: ResourceFormat) -> str:
        """Map a Python ResourceFormat enum to a wgpu format string.

        Args:
            fmt: The Python ResourceFormat value.

        Returns:
            wgpu-format string as required by the Rust PyResourceDesc schema.
        """
        return _FORMAT_TO_WGPU.get(fmt, "R8G8B8A8_UNORM")

    def _collect_py_pass_nodes(self) -> list[dict]:
        """Collect all passes as a list of dicts matching the PyPassNode JSON schema.

        Each dict contains:
          - name: pass name
          - pass_type: "Graphics" | "Compute" | "Copy" | "RayTracing"
          - reads: list of resource names read
          - writes: list of resource names written
          - color_attachments: list of color attachment resource names (Graphics only)
          - depth_attachment: depth/stencil resource name (Graphics only)
          - workgroup_size: [x, y, z] for compute passes

        Returns:
            List of pass dicts in declaration order (self._pass_order).
        """
        nodes: list[dict] = []
        for pass_name in self._pass_order:
            pn = self._passes[pass_name]

            node: dict = {
                "name": pass_name,
                "pass_type": _PASS_TYPE_TO_STR.get(pn.pass_type, "Graphics"),
                "reads": [a.handle.name for a in pn.reads],
                "writes": [a.handle.name for a in pn.writes],
            }

            # Graphics-specific fields
            if isinstance(pn, GraphicsPass):
                node["color_attachments"] = [
                    ca.handle.name for ca in pn.color_attachments
                ]
                node["depth_attachment"] = (
                    pn.depth_stencil.handle.name if pn.depth_stencil else None
                )
                node["instance_source"] = {
                    "type": "direct",
                    "index_count": 0,
                    "instance_count": 1,
                    "base_vertex": 0,
                    "first_index": 0,
                    "first_instance": 0,
                }
                node["view_type"] = "Texture2D"

            # Compute-specific fields
            if isinstance(pn, ComputePass):
                node["workgroup_size"] = list(pn.dispatch_size)

            nodes.append(node)

        return nodes

    def _collect_py_resource_descs(self) -> list[dict]:
        """Collect all resources as a list of dicts matching the PyResourceDesc schema.

        Iterates transient, history, and external resources, building a
        serializable dict for each that the Rust-side deserialize_from_json
        can parse.

        Returns:
            List of resource dicts.
        """
        descs: list[dict] = []
        seen: set[str] = set()

        # Transient resources
        for name, transient in self._resource_manager._transients.items():
            if name in seen:
                continue
            seen.add(name)
            desc = transient.handle.descriptor
            descs.append({
                "name": name,
                "resource_type": "Texture2D" if desc and desc.is_texture else "Buffer",
                "width": desc.width if desc else 0,
                "height": desc.height if desc else 0,
                "depth": desc.depth if desc else 1,
                "format": self._format_to_wgpu_str(desc.format) if desc else "R8G8B8A8_UNORM",
                "is_transient": True,
            })

        # History resources
        for name, history in self._resource_manager._history.items():
            if name in seen:
                continue
            seen.add(name)
            desc = history.handle.descriptor
            descs.append({
                "name": name,
                "resource_type": "Texture2D" if desc and desc.is_texture else "Buffer",
                "width": desc.width if desc else 0,
                "height": desc.height if desc else 0,
                "depth": desc.depth if desc else 1,
                "format": self._format_to_wgpu_str(desc.format) if desc else "R8G8B8A8_UNORM",
                "is_transient": False,
            })

        # External resources
        for name, external in self._resource_manager._externals.items():
            if name in seen:
                continue
            seen.add(name)
            desc = external.handle.descriptor
            descs.append({
                "name": name,
                "resource_type": "Texture2D" if desc and desc.is_texture else "Buffer",
                "width": desc.width if desc else 0,
                "height": desc.height if desc else 0,
                "depth": desc.depth if desc else 1,
                "format": self._format_to_wgpu_str(desc.format) if desc else "R8G8B8A8_UNORM",
                "is_transient": False,
            })

        return descs

    def _serialize_ir(self) -> str:
        """Serialize the current frame graph state to a JSON string.

        The JSON schema matches what the Rust-side ``deserialize_from_json()``
        expects:

        .. code-block:: json

            {
              "passes": [ { "name": "...", "pass_type": "...", ... } ],
              "resources": [ { "name": "...", "resource_type": "...", ... } ]
            }

        Returns:
            JSON string ready for the PyO3 bridge.
        """
        payload = {
            "passes": self._collect_py_pass_nodes(),
            "resources": self._collect_py_resource_descs(),
        }
        return json.dumps(payload, indent=2)

    def _try_compile_via_rust(self, json_ir: str) -> Optional[CompilationResult]:
        """Attempt to compile via the Rust PyO3 bridge.

        Tries to import the ``_omega`` module and call its
        ``frame_graph_execute`` function. If the bridge is not available
        (e.g. running in a pure-Python environment) returns ``None`` so the
        caller can fall back to Python compilation.

        Args:
            json_ir: Serialised IR JSON string from ``_serialize_ir()``.

        Returns:
            A ``CompilationResult`` if the bridge succeeds, or ``None`` when
            the bridge is unavailable.
        """
        try:
            import _omega  # type: ignore[import-untyped]
        except ImportError:
            # Rust PyO3 bridge not available -- fall back to Python compilation
            return None

        try:
            result_json = _omega.frame_graph_execute(json_ir)
            data = json.loads(result_json)
            bridge_result = CompilationResult.from_bridge_json(data)

            # If the bridge returns success but no execution order and we have
            # passes, fall back to Python compilation to ensure passes are
            # properly included in the execution order.
            if (bridge_result.success
                and not bridge_result.execution_order
                and bridge_result.pass_count > 0):
                import sys
                print("[frame_graph] Rust bridge returned empty execution order; "
                      "falling back to Python compilation", file=sys.stderr)
                return None

            return bridge_result
        except Exception as exc:
            # Bridge call failed, log and fall through to Python path
            import sys
            print(f"[frame_graph] Rust bridge compile failed: {exc}; "
                  f"falling back to Python compilation", file=sys.stderr)
            return None


    # =========================================================================
    # Compilation
    # =========================================================================

    def compile(self) -> CompilationResult:
        """Compile the frame graph.

        Compilation performs:
        1. Serialize IR to JSON (T-FG-1.6)
        2. Try Rust PyO3 bridge (``_omega.frame_graph_execute``)
        3. If bridge unavailable, fall back to pure-Python compilation:
           a. Dependency analysis
           b. Dead pass elimination (culling)
           c. Execution order (topological sort)
           d. Resource lifetime tracking
           e. Memory aliasing
           f. Async compute scheduling
           g. Barrier insertion

        Returns:
            Compilation result with status and statistics.
        """
        # Step 1: Serialize IR for potential Rust bridge compilation
        json_ir = self._serialize_ir()

        # Step 2: Try Rust PyO3 bridge
        bridge_result = self._try_compile_via_rust(json_ir)
        if bridge_result is not None:
            self._compiled = bridge_result.success
            self._compilation_result = bridge_result
            return bridge_result

        # Step 3: Fall back to pure-Python compilation
        result = CompilationResult()

        try:
            # Build dependency graph
            self._build_dependency_graph()

            # Cull unused passes
            if self._enable_pass_culling:
                culled = self._cull_unused_passes()
                result.culled_passes = culled
                result.pass_count = len(self._passes)
                result.culled_count = len(culled)
            else:
                result.pass_count = len(self._passes)
                result.culled_count = 0

            # Determine execution order
            self._compute_execution_order()
            result.execution_order = [p.name for p in self._execution_order]

            # Update resource lifetimes for aliasing
            self._update_resource_lifetimes()

            # Compute resource aliasing
            if self._enable_resource_aliasing:
                self._resource_manager.compute_aliasing()
                result.alias_group_count = self._resource_manager.get_alias_group_count()

            # Schedule async compute
            if self._enable_async_compute:
                self._scheduled_passes = self._async_scheduler.schedule(
                    self._execution_order,
                    enable_async_compute=True,
                )
                result.async_pass_count = len(
                    self._async_scheduler.get_compute_passes()
                )
            else:
                self._scheduled_passes = self._async_scheduler.schedule(
                    self._execution_order,
                    enable_async_compute=False,
                )

            # Generate barriers
            self._barrier_batches = self._barrier_manager.analyze_passes(
                self._execution_order
            )
            result.barrier_count = sum(
                len(batch.barriers) for batch in self._barrier_batches
            )

            self._compiled = True
            result.success = True

            # T-FG-7.7: memory savings and errors are populated from the Rust
            # bridge in ``_try_compile_via_rust``.  The Python fallback path
            # has no byte-level resource tracking, so memory_savings_percent
            # is left at 0.0 and errors at [].

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        self._compilation_result = result
        return result

    def _build_dependency_graph(self) -> None:
        """Build the dependency graph from pass declarations.

        Dependencies are inferred from resource read/write declarations:
        - If pass B reads a resource written by pass A, B depends on A
        """
        # Map resource -> producer pass
        producers: dict[str, str] = {}
        # Map pass -> list of passes it depends on
        dependencies: dict[str, list[str]] = {name: [] for name in self._pass_order}

        for pass_name in self._pass_order:
            pass_node = self._passes[pass_name]

            # Check read dependencies - if we read a resource,
            # we depend on the pass that produced it
            for access in pass_node.reads:
                resource_name = access.handle.name
                if resource_name in producers:
                    producer_pass = producers[resource_name]
                    if producer_pass not in dependencies[pass_name]:
                        dependencies[pass_name].append(producer_pass)

            # Record writes - this pass becomes the producer
            for access in pass_node.writes:
                resource_name = access.handle.name
                producers[resource_name] = pass_name

        # Store dependencies for potential future use in topological sort
        self._pass_dependencies = dependencies

    def _cull_unused_passes(self) -> list[str]:
        """Cull passes whose outputs are never used.

        Dead code elimination for the frame graph. A pass is culled if:
        - None of its outputs are read by any subsequent pass
        - None of its outputs are external/backbuffer
        - It doesn't have the NO_CULL or SIDE_EFFECTS flags

        Returns:
            List of culled pass names.
        """
        culled: list[str] = []

        # Collect all read resources
        all_reads: set[str] = set()
        for pass_node in self._passes.values():
            for access in pass_node.reads:
                all_reads.add(access.handle.name)

        # Identify passes to cull
        for pass_name in self._pass_order:
            pass_node = self._passes[pass_name]

            # Don't cull passes with NO_CULL or SIDE_EFFECTS flags
            if pass_node.has_flag(PassFlags.NO_CULL):
                continue
            if pass_node.has_flag(PassFlags.SIDE_EFFECTS):
                continue

            # Check if any output is used
            outputs_used = False
            for access in pass_node.writes:
                resource_name = access.handle.name

                # Check if read by another pass
                if resource_name in all_reads:
                    outputs_used = True
                    break

                # Check if it's an external resource (backbuffer)
                external = self._resource_manager.get_external(resource_name)
                if external and external.is_backbuffer:
                    outputs_used = True
                    break

            if not outputs_used:
                pass_node._culled = True
                culled.append(pass_name)

        return culled

    def _compute_execution_order(self) -> None:
        """Compute execution order via topological sort.

        The execution order respects dependencies: if pass B depends on
        pass A, A comes before B in the order.
        """
        self._execution_order.clear()

        # Simple approach: use declaration order with dependency validation
        # A full implementation would do proper topological sort

        for pass_name in self._pass_order:
            pass_node = self._passes[pass_name]
            if not pass_node._culled:
                pass_node._execution_index = len(self._execution_order)
                self._execution_order.append(pass_node)

    def _update_resource_lifetimes(self) -> None:
        """Update resource lifetime information for aliasing."""
        for idx, pass_node in enumerate(self._execution_order):
            for access in pass_node.reads:
                self._resource_manager.update_lifetime(access.handle, idx)
            for access in pass_node.writes:
                self._resource_manager.update_lifetime(access.handle, idx)

    def _invalidate_compilation(self) -> None:
        """Mark the frame graph as needing recompilation."""
        self._compiled = False
        self._compilation_result = None
        self._execution_order.clear()
        self._barrier_batches.clear()
        self._scheduled_passes.clear()

        # Reset pass culling state
        for pass_node in self._passes.values():
            pass_node._culled = False
            pass_node._execution_index = -1

    # =========================================================================
    # Execution
    # =========================================================================

    def execute(self, context: RHIContext) -> None:
        """Execute the compiled frame graph.

        Args:
            context: Platform-specific rendering context.

        Raises:
            RuntimeError: If the frame graph hasn't been compiled.
        """
        if not self._compiled:
            raise RuntimeError("Frame graph must be compiled before execution")

        # Begin frame
        self._resource_manager.begin_frame()

        # Execute each pass with its barriers
        for idx, pass_node in enumerate(self._execution_order):
            # Execute barriers before this pass
            if idx < len(self._barrier_batches):
                batch = self._barrier_batches[idx]
                self._execute_barriers(batch, context)

            # Execute the pass
            pass_node.execute(context)

        # Prepare backbuffer for present
        self._prepare_for_present(context)

    def _execute_barriers(
        self,
        batch: BarrierBatch,
        context: RHIContext,
    ) -> None:
        """Execute a batch of barriers.

        Args:
            batch: The barrier batch.
            context: Rendering context.
        """
        if batch.is_empty():
            return

        # In a real implementation, this would call into the RHI
        # to execute the actual GPU barriers.
        # The context object should provide a method like:
        #   context.execute_barriers(batch.barriers)
        # For now, we log/track the barriers for debugging purposes.
        if hasattr(context, 'execute_barriers'):
            context.execute_barriers(batch.barriers)

    def _prepare_for_present(self, context: RHIContext) -> None:
        """Prepare backbuffer for presentation.

        Args:
            context: Rendering context.
        """
        # Find the backbuffer and transition it to PRESENT state
        for name, external in self._resource_manager._externals.items():
            if external.is_backbuffer:
                barrier = self._barrier_manager.prepare_for_present(
                    external.handle
                )
                if barrier:
                    # Execute the presentation barrier through the context
                    if hasattr(context, 'execute_barriers'):
                        context.execute_barriers([barrier])
                break

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_async_compute_enabled(self, enabled: bool) -> None:
        """Enable or disable async compute scheduling.

        Args:
            enabled: Whether to enable async compute.
        """
        if self._enable_async_compute != enabled:
            self._enable_async_compute = enabled
            self._invalidate_compilation()

    def set_pass_culling_enabled(self, enabled: bool) -> None:
        """Enable or disable unused pass culling.

        Args:
            enabled: Whether to enable pass culling.
        """
        if self._enable_pass_culling != enabled:
            self._enable_pass_culling = enabled
            self._invalidate_compilation()

    def set_resource_aliasing_enabled(self, enabled: bool) -> None:
        """Enable or disable resource memory aliasing.

        Args:
            enabled: Whether to enable aliasing.
        """
        if self._enable_resource_aliasing != enabled:
            self._enable_resource_aliasing = enabled
            self._invalidate_compilation()

    # =========================================================================
    # Introspection
    # =========================================================================

    @property
    def is_compiled(self) -> bool:
        """Check if the frame graph is compiled."""
        return self._compiled

    @property
    def pass_count(self) -> int:
        """Get the number of passes."""
        return len(self._passes)

    @property
    def resource_count(self) -> int:
        """Get the number of resources."""
        return len(self._resource_manager._handles)

    def get_pass_names(self) -> list[str]:
        """Get all pass names in declaration order."""
        return list(self._pass_order)

    def get_execution_order(self) -> list[str]:
        """Get pass names in execution order.

        Returns:
            List of pass names, or empty if not compiled.
        """
        return [p.name for p in self._execution_order]

    def get_compilation_result(self) -> Optional[CompilationResult]:
        """Get the last compilation result."""
        return self._compilation_result

    def get_barriers_for_pass(self, pass_name: str) -> list[Barrier]:
        """Get barriers inserted before a specific pass.

        Args:
            pass_name: The pass name.

        Returns:
            List of barriers, or empty if pass not found.
        """
        for batch in self._barrier_batches:
            if batch.before_pass == pass_name:
                return batch.barriers
        return []

    def clear(self) -> None:
        """Clear all passes and resources."""
        self._passes.clear()
        self._pass_order.clear()
        self._resource_manager.clear()
        self._barrier_manager.reset()
        self._invalidate_compilation()
