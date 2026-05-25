"""FlowForge Backend entry point.

Runs the IPC loop, reading line-delimited JSON from stdin,
processing requests, and writing responses to stdout.
Logging goes to stderr to avoid interfering with the protocol.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from . import __version__
from .config import PROTOCOL_VERSION, MILLISECONDS_PER_SECOND
from .ipc import Handler, IPCRequest, IPCResponse, IPCError


def create_handler() -> Handler:
    """Create and configure the IPC handler with built-in methods."""
    handler = Handler()

    @handler.register("ping")
    def handle_ping(params: dict[str, Any]) -> dict[str, Any]:
        """Simple ping handler for connection testing.

        Args:
            params: Optional parameters (ignored)

        Returns:
            Dict with pong=True and timestamp
        """
        import time
        return {
            "pong": True,
            "timestamp": int(time.time() * MILLISECONDS_PER_SECOND),
        }

    @handler.register("get_version")
    def handle_get_version(params: dict[str, Any]) -> dict[str, Any]:
        """Return version information.

        Args:
            params: Optional parameters (ignored)

        Returns:
            Dict with version info
        """
        return {
            "version": __version__,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "protocol_version": PROTOCOL_VERSION,
        }

    @handler.register("get_object_info")
    def handle_get_object_info(params: dict[str, Any]) -> dict[str, Any]:
        """Get information about available Trinity node types.

        This is a stub that returns hardcoded Trinity game engine node types.
        Will be replaced with actual Trinity introspection in Project 3.

        Args:
            params: Optional filter parameters
                - category: Filter by category (optional)
                - search: Search term (optional)

        Returns:
            Dict with node type information
        """
        # Stub data - Trinity game engine node types
        # This will be replaced with actual introspection in Project 3
        node_types = [
            {
                "name": "Transform",
                "category": "Core",
                "description": "Represents position, rotation, and scale in 3D space",
                "properties": [
                    {"name": "position", "type": "Vector3", "default": [0, 0, 0]},
                    {"name": "rotation", "type": "Quaternion", "default": [0, 0, 0, 1]},
                    {"name": "scale", "type": "Vector3", "default": [1, 1, 1]},
                ],
                "methods": [
                    {"name": "translate", "params": [{"name": "delta", "type": "Vector3"}]},
                    {"name": "rotate", "params": [{"name": "euler", "type": "Vector3"}]},
                    {"name": "look_at", "params": [{"name": "target", "type": "Vector3"}]},
                ],
            },
            {
                "name": "Mesh",
                "category": "Rendering",
                "description": "3D mesh for rendering",
                "properties": [
                    {"name": "vertices", "type": "Array<Vector3>", "default": []},
                    {"name": "indices", "type": "Array<int>", "default": []},
                    {"name": "visible", "type": "bool", "default": True},
                ],
                "methods": [
                    {"name": "set_material", "params": [{"name": "material", "type": "Material"}]},
                    {"name": "recalculate_normals", "params": []},
                ],
            },
            {
                "name": "RigidBody",
                "category": "Physics",
                "description": "Physics body with mass and forces",
                "properties": [
                    {"name": "mass", "type": "float", "default": 1.0},
                    {"name": "velocity", "type": "Vector3", "default": [0, 0, 0]},
                    {"name": "angular_velocity", "type": "Vector3", "default": [0, 0, 0]},
                    {"name": "is_kinematic", "type": "bool", "default": False},
                ],
                "methods": [
                    {"name": "apply_force", "params": [{"name": "force", "type": "Vector3"}]},
                    {"name": "apply_impulse", "params": [{"name": "impulse", "type": "Vector3"}]},
                    {"name": "set_velocity", "params": [{"name": "velocity", "type": "Vector3"}]},
                ],
            },
            {
                "name": "AudioSource",
                "category": "Audio",
                "description": "Plays audio clips in 3D space",
                "properties": [
                    {"name": "clip", "type": "AudioClip", "default": None},
                    {"name": "volume", "type": "float", "default": 1.0},
                    {"name": "pitch", "type": "float", "default": 1.0},
                    {"name": "loop", "type": "bool", "default": False},
                    {"name": "spatial_blend", "type": "float", "default": 1.0},
                ],
                "methods": [
                    {"name": "play", "params": []},
                    {"name": "stop", "params": []},
                    {"name": "pause", "params": []},
                ],
            },
            {
                "name": "Script",
                "category": "Logic",
                "description": "Custom behavior script",
                "properties": [
                    {"name": "enabled", "type": "bool", "default": True},
                ],
                "methods": [
                    {"name": "on_start", "params": []},
                    {"name": "on_update", "params": [{"name": "delta_time", "type": "float"}]},
                    {"name": "on_fixed_update", "params": [{"name": "delta_time", "type": "float"}]},
                ],
            },
            {
                "name": "Camera",
                "category": "Rendering",
                "description": "Scene camera for rendering",
                "properties": [
                    {"name": "fov", "type": "float", "default": 60.0},
                    {"name": "near_clip", "type": "float", "default": 0.1},
                    {"name": "far_clip", "type": "float", "default": 1000.0},
                    {"name": "is_orthographic", "type": "bool", "default": False},
                ],
                "methods": [
                    {"name": "world_to_screen", "params": [{"name": "world_pos", "type": "Vector3"}]},
                    {"name": "screen_to_world", "params": [{"name": "screen_pos", "type": "Vector2"}]},
                ],
            },
            {
                "name": "Light",
                "category": "Rendering",
                "description": "Scene lighting",
                "properties": [
                    {"name": "color", "type": "Color", "default": [1, 1, 1, 1]},
                    {"name": "intensity", "type": "float", "default": 1.0},
                    {"name": "range", "type": "float", "default": 10.0},
                    {"name": "type", "type": "LightType", "default": "point"},
                ],
                "methods": [],
            },
            {
                "name": "Collider",
                "category": "Physics",
                "description": "Collision detection shape",
                "properties": [
                    {"name": "is_trigger", "type": "bool", "default": False},
                    {"name": "center", "type": "Vector3", "default": [0, 0, 0]},
                ],
                "methods": [
                    {"name": "on_collision_enter", "params": [{"name": "other", "type": "Collider"}]},
                    {"name": "on_collision_exit", "params": [{"name": "other", "type": "Collider"}]},
                    {"name": "on_trigger_enter", "params": [{"name": "other", "type": "Collider"}]},
                ],
            },
        ]

        # Apply filters if provided
        category = params.get("category")
        search = params.get("search", "").lower()

        filtered = node_types

        if category:
            filtered = [n for n in filtered if n["category"] == category]

        if search:
            filtered = [
                n for n in filtered
                if search in n["name"].lower() or search in n["description"].lower()
            ]

        # Get unique categories
        categories = sorted(set(n["category"] for n in node_types))

        return {
            "nodes": filtered,
            "categories": categories,
            "total_count": len(filtered),
        }

    @handler.register("list_methods")
    def handle_list_methods(params: dict[str, Any]) -> dict[str, Any]:
        """List all available IPC methods.

        Args:
            params: Optional parameters (ignored)

        Returns:
            Dict with list of available methods
        """
        return {
            "methods": handler.list_methods(),
        }

    @handler.register("parse_python_file")
    def handle_parse_python_file(params: dict[str, Any]) -> dict[str, Any]:
        """Parse a Python file and return a node graph.

        Parses the specified Python file for Trinity ECS definitions
        (@component, @system, @resource, @event) and returns a visual
        node graph representation.

        Args:
            params: Dictionary with required parameters:
                - path: str - Absolute path to the Python file to parse

        Returns:
            Dict containing:
                - success: bool - Whether parsing succeeded
                - errors: list[str] - List of error messages (may be non-empty even on success)
                - graph: dict | None - The node graph (nodes, edges) if successful

        Raises:
            ValueError: If path parameter is missing
        """
        import os
        from .ast_parser import parse_file, build_node_graph

        path = params.get("path")
        if not path:
            raise ValueError("Missing required parameter: path")

        if not isinstance(path, str):
            raise ValueError("Parameter 'path' must be a string")

        # Check if file exists before parsing
        if not os.path.isfile(path):
            return {
                "success": False,
                "errors": [f"File not found: {path}"],
                "graph": None,
            }

        try:
            # Parse the file for Trinity definitions
            result = parse_file(path)

            # Check for parsing errors (syntax errors, etc.)
            if result.has_errors:
                return {
                    "success": False,
                    "errors": result.errors,
                    "graph": None,
                }

            # Build the node graph from parse result
            graph = build_node_graph(result)

            return {
                "success": True,
                "errors": [],
                "graph": graph.to_dict(),
            }

        except SyntaxError as e:
            # Handle Python syntax errors with line number info
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}" if e.lineno else str(e)
            return {
                "success": False,
                "errors": [error_msg],
                "graph": None,
            }
        except Exception as e:
            # Handle other unexpected errors
            return {
                "success": False,
                "errors": [f"Error parsing file: {str(e)}"],
                "graph": None,
            }

    # =========================================================================
    # Trinity Runtime Introspection Handlers (Phase 3.3.1)
    # =========================================================================

    @handler.register("trinity.status")
    def handle_trinity_status(params: dict[str, Any]) -> dict[str, Any]:
        """Check if Trinity ECS runtime is available.

        Args:
            params: Optional parameters:
                - initialize: bool - Whether to initialize Trinity (default: False)
                - debug_mode: bool - Enable debug mode if initializing (default: True)

        Returns:
            Dict with Trinity availability status and features.
        """
        from .trinity_introspection import check_trinity_status, initialize_trinity

        initialize = params.get("initialize", False)
        debug_mode = params.get("debug_mode", True)

        status = check_trinity_status()
        result = status.to_dict()

        if initialize and status.available:
            init_result = initialize_trinity(debug_mode=debug_mode)
            result["initialization"] = init_result

        return result

    @handler.register("trinity.registry.list")
    def handle_trinity_registry_list(params: dict[str, Any]) -> dict[str, Any]:
        """List all registered Trinity types.

        Args:
            params: Optional filter parameters:
                - category: str - Filter by category ("component", "system", etc.)
                - search: str - Search term to filter by name

        Returns:
            Dict with list of registered types and metadata.
        """
        from .trinity_introspection import list_registered_types

        category = params.get("category")
        search = params.get("search")

        return list_registered_types(category=category, search=search)

    @handler.register("trinity.registry.info")
    def handle_trinity_registry_info(params: dict[str, Any]) -> dict[str, Any]:
        """Get detailed information about a specific registered type.

        Args:
            params: Dictionary with required parameters:
                - qualified_name: str - The fully qualified type name

        Returns:
            Dict with detailed type information.
        """
        from .trinity_introspection import get_type_info

        qualified_name = params.get("qualified_name")
        if not qualified_name:
            raise ValueError("Missing required parameter: qualified_name")

        return get_type_info(qualified_name)

    @handler.register("trinity.instances.query")
    def handle_trinity_instances_query(params: dict[str, Any]) -> dict[str, Any]:
        """Query active instances in the Trinity runtime.

        Requires Foundation to be available for instance tracking.

        Args:
            params: Optional filter parameters:
                - type_name: str - Filter by type name
                - limit: int - Maximum number of instances (default: 100)

        Returns:
            Dict with list of active instances.
        """
        from .trinity_introspection import query_active_instances

        type_name = params.get("type_name")
        limit = params.get("limit", 100)

        return query_active_instances(type_name=type_name, limit=limit)

    @handler.register("trinity.instances.mirror")
    def handle_trinity_instances_mirror(params: dict[str, Any]) -> dict[str, Any]:
        """Get detailed mirror information for a specific instance.

        Args:
            params: Dictionary with required parameters:
                - type_name: str - The type name of the instance
                - instance_id: int - The instance ID

        Returns:
            Dict with detailed instance information via mirror.
        """
        from .trinity_introspection import get_instance_mirror

        type_name = params.get("type_name")
        instance_id = params.get("instance_id")

        if not type_name:
            raise ValueError("Missing required parameter: type_name")
        if instance_id is None:
            raise ValueError("Missing required parameter: instance_id")

        return get_instance_mirror(type_name, instance_id)

    @handler.register("trinity.events.recent")
    def handle_trinity_events_recent(params: dict[str, Any]) -> dict[str, Any]:
        """Get recent events from the Trinity EventLog.

        Requires Foundation to be available for the EventLog.

        Args:
            params: Optional filter parameters:
                - limit: int - Maximum number of events (default: 50)
                - entity: int - Filter by entity ID
                - operation: str - Filter by operation name
                - tick: int - Filter by specific tick

        Returns:
            Dict with list of recent events.
        """
        from .trinity_introspection import get_recent_events

        limit = params.get("limit", 50)
        entity = params.get("entity")
        operation = params.get("operation")
        tick = params.get("tick")

        return get_recent_events(
            limit=limit,
            entity=entity,
            operation=operation,
            tick=tick,
        )

    @handler.register("trinity.events.stats")
    def handle_trinity_events_stats(params: dict[str, Any]) -> dict[str, Any]:
        """Get statistics about the Trinity EventLog.

        Args:
            params: Optional parameters (ignored)

        Returns:
            Dict with event statistics.
        """
        from .trinity_introspection import get_event_statistics

        return get_event_statistics()

    @handler.register("trinity.decorators.list")
    def handle_trinity_decorators_list(params: dict[str, Any]) -> dict[str, Any]:
        """List all registered Trinity decorators.

        Args:
            params: Optional filter parameters:
                - tier: int - Filter by tier number

        Returns:
            Dict with list of decorators.
        """
        from .trinity_introspection import list_decorators

        tier = params.get("tier")

        return list_decorators(tier=tier)

    @handler.register("trinity.connect")
    def handle_trinity_connect(params: dict[str, Any]) -> dict[str, Any]:
        """Connect to Trinity runtime for live introspection.

        Initializes a connection to the Trinity ECS runtime, enabling
        live introspection of types, instances, and events.

        Args:
            params: Optional parameters:
                - debug_mode: bool - Enable debug mode (default: True)

        Returns:
            Dict with connection status and session information.
        """
        from .trinity_introspection import connect_trinity

        debug_mode = params.get("debug_mode", True)

        return connect_trinity(debug_mode=debug_mode)

    @handler.register("trinity.disconnect")
    def handle_trinity_disconnect(params: dict[str, Any]) -> dict[str, Any]:
        """Disconnect from Trinity runtime.

        Closes the connection to the Trinity ECS runtime and cleans up
        any associated resources.

        Args:
            params: Optional parameters (ignored)

        Returns:
            Dict with disconnection status.
        """
        from .trinity_introspection import disconnect_trinity

        return disconnect_trinity()

    @handler.register("trinity.poll")
    def handle_trinity_poll(params: dict[str, Any]) -> dict[str, Any]:
        """Poll Trinity runtime for updates.

        Returns current state and any pending events since last poll.
        This is the primary method for live introspection updates.

        Args:
            params: Optional parameters:
                - include_events: bool - Include recent events (default: True)
                - include_instances: bool - Include instance snapshots (default: False)
                - event_limit: int - Maximum events to return (default: 50)

        Returns:
            Dict with current state and any updates.
        """
        from .trinity_introspection import poll_trinity

        include_events = params.get("include_events", True)
        include_instances = params.get("include_instances", False)
        event_limit = params.get("event_limit", 50)

        return poll_trinity(
            include_events=include_events,
            include_instances=include_instances,
            event_limit=event_limit,
        )

    @handler.register("trinity.inspector.get")
    def handle_trinity_inspector_get(params: dict[str, Any]) -> dict[str, Any]:
        """Get detailed inspector information for a target.

        Unified inspector API that can get information about types,
        instances, or decorators.

        Args:
            params: Dictionary with required parameters:
                - target_type: str - Type of target ("type", "instance", "decorator")
                - target_id: int - Instance ID (required for "instance" target_type)
                - qualified_name: str - Qualified name for type or decorator lookup

        Returns:
            Dict with detailed inspector information.
        """
        from .trinity_introspection import inspector_get

        target_type = params.get("target_type")
        if not target_type:
            raise ValueError("Missing required parameter: target_type")

        target_id = params.get("target_id")
        qualified_name = params.get("qualified_name")

        return inspector_get(
            target_type=target_type,
            target_id=target_id,
            qualified_name=qualified_name,
        )

    @handler.register("trinity.inspect")
    def handle_trinity_inspect(params: dict[str, Any]) -> dict[str, Any]:
        """Inspect a Trinity type by name.

        Simple inspection API that retrieves detailed information about
        a registered Trinity type.

        Args:
            params: Dictionary with required parameters:
                - type_name: str - The type name to inspect

        Returns:
            Dict with type inspection information.
        """
        from .trinity_introspection import inspector_get

        type_name = params.get("type_name")
        if not type_name:
            raise ValueError("Missing required parameter: type_name")

        return inspector_get(
            target_type="type",
            qualified_name=type_name,
        )

    # =========================================================================
    # Code Validation Handlers (Phase 4.2.4)
    # =========================================================================

    @handler.register("validate_python")
    def handle_validate_python(params: dict[str, Any]) -> dict[str, Any]:
        """Validate Python source code.

        Parses the source with ast.parse() to check syntax and optionally
        performs basic semantic checks like undefined name detection.

        Args:
            params: Dictionary with parameters:
                - source: str - The Python source code to validate (required)
                - check_semantics: bool - If True, perform semantic checks (default: False)
                - filename: str - Filename to use in error messages (default: "<generated>")

        Returns:
            Dict containing:
                - success: bool - True if the code is valid (no errors)
                - errors: list - List of validation errors
                - warnings: list - List of validation warnings
                - source_hash: str - Hash of the validated source
        """
        from .codegen import validate_python

        source = params.get("source")
        if source is None:
            raise ValueError("Missing required parameter: source")

        if not isinstance(source, str):
            raise ValueError("Parameter 'source' must be a string")

        check_semantics = params.get("check_semantics", False)
        filename = params.get("filename", "<generated>")

        result = validate_python(
            source,
            check_semantics=check_semantics,
            filename=filename,
        )

        return result.to_dict()

    @handler.register("generate_python")
    def handle_generate_python(params: dict[str, Any]) -> dict[str, Any]:
        """Generate Python code from a node graph.

        Takes a visual node graph (representing Trinity ECS definitions)
        and generates the corresponding Python source code using AST-based
        code generation.

        Args:
            params: Dictionary with parameters:
                - graph: dict - The node graph to generate code from (required)
                - format_with_black: bool - Format output with black (default: True)
                - line_length: int - Maximum line length (default: 88)
                - add_header: bool - Add generated code header (default: True)
                - check_semantics: bool - Perform semantic validation (default: False)

        Returns:
            Dict containing:
                - source: str - The generated Python source code
                - validation: dict - Validation result for the generated code
                - imports: list - List of imports used
                - node_count: int - Number of nodes processed
                - metadata: dict - Additional generation metadata
        """
        from .codegen import generate_python_with_validation, GenerationResult

        graph_data = params.get("graph")
        if graph_data is None:
            raise ValueError("Missing required parameter: graph")

        if not isinstance(graph_data, dict):
            raise ValueError("Parameter 'graph' must be a dictionary")

        # Extract optional parameters
        format_with_black = params.get("format_with_black", True)
        line_length = params.get("line_length", 88)
        add_header = params.get("add_header", True)
        check_semantics = params.get("check_semantics", False)

        try:
            # Generate Python code from the graph with validation
            result = generate_python_with_validation(
                graph=graph_data,
                format_with_black=format_with_black,
                line_length=line_length,
                add_header=add_header,
                check_semantics=check_semantics,
            )

            return result.to_dict()

        except Exception as e:
            # Return an error result
            result = GenerationResult.error(str(e))
            return result.to_dict()

    # =========================================================================
    # Code Generation Handlers (Phase 4.3)
    # =========================================================================

    @handler.register("generate_diff")
    def handle_generate_diff(params: dict[str, Any]) -> dict[str, Any]:
        """Generate a unified diff between original and modified source.

        Args:
            params: Dictionary with required parameters:
                - original: str - Original source code
                - modified: str - Modified source code
            Optional parameters:
                - filename: str - Name of the file (for display)
                - original_path: str - Path to the original file
                - context_lines: int - Number of context lines (default: 3)
                - side_by_side: bool - Return side-by-side format (default: False)

        Returns:
            Dict containing the diff result with structured data.
        """
        from .codegen import generate_diff, generate_side_by_side_diff

        original = params.get("original")
        modified = params.get("modified")

        if original is None:
            raise ValueError("Missing required parameter: original")
        if modified is None:
            raise ValueError("Missing required parameter: modified")

        if not isinstance(original, str):
            raise ValueError("Parameter 'original' must be a string")
        if not isinstance(modified, str):
            raise ValueError("Parameter 'modified' must be a string")

        filename = params.get("filename", "")
        original_path = params.get("original_path")
        context_lines = params.get("context_lines", 3)
        side_by_side = params.get("side_by_side", False)

        if side_by_side:
            return generate_side_by_side_diff(original, modified, filename)

        diff_result = generate_diff(
            original=original,
            modified=modified,
            filename=filename,
            original_path=original_path,
            context_lines=context_lines,
        )

        return diff_result.to_dict()

    return handler


def log(message: str, level: str = "INFO") -> None:
    """Log a message to stderr.

    Args:
        message: The message to log
        level: Log level (INFO, WARN, ERROR, DEBUG)
    """
    print(f"[{level}] flowforge-backend: {message}", file=sys.stderr)


def process_line(line: str, handler: Handler) -> str:
    """Process a single input line and return the response JSON.

    Args:
        line: The input line (should be JSON)
        handler: The IPC handler

    Returns:
        JSON response string
    """
    # Parse the request
    try:
        request = IPCRequest.from_json(line)
    except ValueError as e:
        # Can't determine request ID if parsing failed
        error = IPCError.parse_error(str(e))
        response = IPCResponse(id="unknown", error=error)
        return response.to_json()

    # Handle the request
    response = handler.handle(request)
    return response.to_json()


def main() -> None:
    """Main entry point - runs the IPC loop."""
    log("Starting FlowForge Backend...")
    log(f"Version: {__version__}")
    log(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    handler = create_handler()
    log(f"Registered methods: {', '.join(handler.list_methods())}")
    log("Ready for requests")

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            log(f"Received: {line[:100]}{'...' if len(line) > 100 else ''}", level="DEBUG")

            response = process_line(line, handler)

            # Write response to stdout (the protocol channel)
            print(response, flush=True)

            log(f"Sent: {response[:100]}{'...' if len(response) > 100 else ''}", level="DEBUG")

    except KeyboardInterrupt:
        log("Shutting down (keyboard interrupt)")
    except BrokenPipeError:
        log("Shutting down (broken pipe - parent process closed)")
    except Exception as e:
        log(f"Fatal error: {e}", level="ERROR")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    log("Shutdown complete")


if __name__ == "__main__":
    main()
