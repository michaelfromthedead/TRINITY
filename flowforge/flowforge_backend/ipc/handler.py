"""IPC Handler for FlowForge Backend.

Provides method routing and dispatch for incoming IPC requests.
"""

from __future__ import annotations

import os
import shutil
import sys
import traceback
from datetime import datetime
from typing import Any, Callable, Awaitable, Optional, Union
from typing_extensions import TypeAlias

from .protocol import IPCRequest, IPCResponse, IPCError

# Code generation imports
from ..codegen.graph_to_ast import graph_to_ast
from ..codegen.emitter import emit_python
from ..codegen.validator import validate_python
from ..codegen.diff import generate_diff, generate_side_by_side_diff
from ..codegen.file_lock import FileLock


# Handler function types
SyncHandler: TypeAlias = Callable[[dict[str, Any]], Any]
AsyncHandler: TypeAlias = Callable[[dict[str, Any]], Awaitable[Any]]
HandlerFunc: TypeAlias = Union[SyncHandler, AsyncHandler]


class Handler:
    """Routes IPC requests to registered handler functions.

    This class maintains a registry of method handlers and dispatches
    incoming requests to the appropriate handler based on the method name.

    Example:
        handler = Handler()

        @handler.register("ping")
        def handle_ping(params):
            return {"pong": True}

        response = handler.handle(request)
    """

    def __init__(self) -> None:
        """Initialize a new Handler with an empty method registry."""
        self._handlers: dict[str, HandlerFunc] = {}

    def register_handler(self, method: str, func: HandlerFunc) -> None:
        """Register a handler function for a method.

        Args:
            method: The method name to handle
            func: The handler function (sync or async)
        """
        self._handlers[method] = func

    def register(self, method: str) -> Callable[[HandlerFunc], HandlerFunc]:
        """Decorator to register a handler function.

        Args:
            method: The method name to handle

        Returns:
            Decorator function

        Example:
            @handler.register("my_method")
            def handle_my_method(params):
                return {"result": "success"}
        """
        def decorator(func: HandlerFunc) -> HandlerFunc:
            self.register_handler(method, func)
            return func
        return decorator

    def has_handler(self, method: str) -> bool:
        """Check if a handler is registered for a method.

        Args:
            method: The method name to check

        Returns:
            True if a handler is registered, False otherwise
        """
        return method in self._handlers

    def list_methods(self) -> list[str]:
        """List all registered method names.

        Returns:
            List of registered method names
        """
        return list(self._handlers.keys())

    def handle(self, request: IPCRequest) -> IPCResponse:
        """Handle an IPC request synchronously.

        Routes the request to the appropriate handler and returns the response.
        If no handler is registered for the method, returns a method not found error.
        If the handler raises an exception, returns an internal error.

        Args:
            request: The incoming IPC request

        Returns:
            The IPC response (success or error)
        """
        method = request.method

        if not self.has_handler(method):
            return IPCResponse.failure(
                request.id,
                IPCError.method_not_found(method)
            )

        handler = self._handlers[method]

        try:
            result = handler(request.params)

            # Check if result is a coroutine (async handler called synchronously)
            if hasattr(result, "__await__"):
                # Log warning to stderr - async handlers need async context
                print(
                    f"Warning: Async handler '{method}' called synchronously. "
                    "Use handle_async() for async handlers.",
                    file=sys.stderr
                )
                return IPCResponse.failure(
                    request.id,
                    IPCError.internal_error(
                        f"Handler '{method}' is async but was called synchronously"
                    )
                )

            return IPCResponse.success(request.id, result)

        except ValueError as e:
            # Treat ValueError as invalid params
            return IPCResponse.failure(
                request.id,
                IPCError.invalid_params(str(e))
            )
        except Exception as e:
            # Log the full traceback to stderr for debugging
            traceback.print_exc(file=sys.stderr)
            return IPCResponse.failure(
                request.id,
                IPCError.internal_error(str(e))
            )

    async def handle_async(self, request: IPCRequest) -> IPCResponse:
        """Handle an IPC request asynchronously.

        Similar to handle() but properly awaits async handlers.

        Args:
            request: The incoming IPC request

        Returns:
            The IPC response (success or error)
        """
        method = request.method

        if not self.has_handler(method):
            return IPCResponse.failure(
                request.id,
                IPCError.method_not_found(method)
            )

        handler = self._handlers[method]

        try:
            result = handler(request.params)

            # Await if it's a coroutine
            if hasattr(result, "__await__"):
                result = await result

            return IPCResponse.success(request.id, result)

        except ValueError as e:
            return IPCResponse.failure(
                request.id,
                IPCError.invalid_params(str(e))
            )
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return IPCResponse.failure(
                request.id,
                IPCError.internal_error(str(e))
            )


# =============================================================================
# Code Generation Handlers
# =============================================================================


def handle_generate_code(request: dict[str, Any]) -> dict[str, Any]:
    """Generate Python source code from node graph.

    Args:
        request: Dictionary with:
            - graph: NodeGraph dict with nodes and edges
            - format_with_black: bool (default True)
            - add_header: bool (default True)

    Returns:
        Dictionary with:
            - source: str - Generated Python code
            - validation: ValidationResult dict
            - node_count: int

    Raises:
        ValueError: If graph is missing or invalid.
    """
    # Validate required parameters
    graph = request.get("graph")
    if graph is None:
        raise ValueError("Missing required parameter: 'graph'")

    if not isinstance(graph, dict):
        raise ValueError("Parameter 'graph' must be a dictionary")

    # Extract optional parameters with defaults
    format_with_black = request.get("format_with_black", True)
    add_header = request.get("add_header", True)

    # Convert graph to AST
    try:
        module = graph_to_ast(graph)
    except Exception as e:
        raise ValueError(f"Failed to convert graph to AST: {e}") from e

    # Emit Python source code
    try:
        source = emit_python(
            module,
            format_with_black=format_with_black,
            add_header=add_header,
        )
    except Exception as e:
        raise ValueError(f"Failed to emit Python code: {e}") from e

    # Validate the generated code
    validation = validate_python(source)

    # Count nodes
    nodes = graph.get("nodes", [])
    node_count = len(nodes) if isinstance(nodes, list) else 0

    return {
        "source": source,
        "validation": validation.to_dict(),
        "node_count": node_count,
    }


def handle_validate_code(request: dict[str, Any]) -> dict[str, Any]:
    """Validate Python source code.

    Args:
        request: Dictionary with:
            - source: str - Python source code to validate
            - check_semantics: bool (default False)

    Returns:
        ValidationResult dict with:
            - success: bool
            - errors: list of error dicts
            - warnings: list of warning dicts
            - source_hash: optional str

    Raises:
        ValueError: If source is missing.
    """
    # Validate required parameters
    source = request.get("source")
    if source is None:
        raise ValueError("Missing required parameter: 'source'")

    if not isinstance(source, str):
        raise ValueError("Parameter 'source' must be a string")

    # Extract optional parameters
    check_semantics = request.get("check_semantics", False)

    # Validate the source code
    result = validate_python(source, check_semantics=check_semantics)

    return result.to_dict()


def handle_generate_diff(request: dict[str, Any]) -> dict[str, Any]:
    """Generate diff between original and modified source.

    Args:
        request: Dictionary with:
            - original: str - Original source code
            - modified: str - Modified source code
            - filename: str (optional) - Filename for display
            - original_path: str (optional) - Path to original file
            - context_lines: int (default 3) - Context lines around changes
            - side_by_side: bool (default False) - Generate side-by-side diff

    Returns:
        DiffResult dict or SideBySideDiff dict depending on side_by_side param.

    Raises:
        ValueError: If original or modified is missing.
    """
    # Validate required parameters
    original = request.get("original")
    if original is None:
        raise ValueError("Missing required parameter: 'original'")

    if not isinstance(original, str):
        raise ValueError("Parameter 'original' must be a string")

    modified = request.get("modified")
    if modified is None:
        raise ValueError("Missing required parameter: 'modified'")

    if not isinstance(modified, str):
        raise ValueError("Parameter 'modified' must be a string")

    # Extract optional parameters
    filename = request.get("filename", "")
    original_path = request.get("original_path")
    context_lines = request.get("context_lines", 3)
    side_by_side = request.get("side_by_side", False)

    # Generate the appropriate diff format
    if side_by_side:
        result = generate_side_by_side_diff(
            original=original,
            modified=modified,
            filename=filename,
        )
        return result
    else:
        result = generate_diff(
            original=original,
            modified=modified,
            filename=filename,
            original_path=original_path,
            context_lines=context_lines,
        )
        return result.to_dict()


def handle_apply_changes(request: dict[str, Any]) -> dict[str, Any]:
    """Apply generated code to file.

    Args:
        request: Dictionary with:
            - file_path: str - Path to the file to write
            - content: str - Content to write to the file
            - create_backup: bool (default True) - Create backup before writing

    Returns:
        Dictionary with:
            - success: bool
            - backup_path: str or None
            - error: str or None

    Raises:
        ValueError: If file_path or content is missing.
    """
    # Validate required parameters
    file_path = request.get("file_path")
    if file_path is None:
        raise ValueError("Missing required parameter: 'file_path'")

    if not isinstance(file_path, str):
        raise ValueError("Parameter 'file_path' must be a string")

    content = request.get("content")
    if content is None:
        raise ValueError("Missing required parameter: 'content'")

    if not isinstance(content, str):
        raise ValueError("Parameter 'content' must be a string")

    # Extract optional parameters
    create_backup = request.get("create_backup", True)

    backup_path: Optional[str] = None
    error: Optional[str] = None

    lock = FileLock()

    try:
        # Normalize the path
        file_path = os.path.normpath(os.path.expanduser(file_path))

        # Acquire file lock
        if not lock.acquire(file_path):
            lock_info = lock.is_locked(file_path)
            owner = f"PID {lock_info.pid}" if lock_info else "unknown"
            return {
                "success": False,
                "backup_path": None,
                "error": f"File is locked by another process ({owner})",
            }

        try:
            # Create backup if requested and file exists
            if create_backup and os.path.exists(file_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{file_path}.backup_{timestamp}"
                shutil.copy2(file_path, backup_path)

            # Ensure the parent directory exists
            parent_dir = os.path.dirname(file_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            # Write the content to the file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return {
                "success": True,
                "backup_path": backup_path,
                "error": None,
            }

        finally:
            lock.release(file_path)

    except PermissionError as e:
        error = f"Permission denied: {e}"
    except OSError as e:
        error = f"OS error: {e}"
    except Exception as e:
        error = f"Unexpected error: {e}"

    return {
        "success": False,
        "backup_path": backup_path,
        "error": error,
    }


def handle_check_file_lock(request: dict[str, Any]) -> dict[str, Any]:
    """Check if a file is locked.

    Args:
        request: Dictionary with:
            - file_path: str - Path to the file to check

    Returns:
        Dictionary with:
            - locked: bool
            - lock_info: dict or None (pid, timestamp, hostname)
            - stale: bool - True if lock exists but owning process is dead

    Raises:
        ValueError: If file_path is missing.
    """
    file_path = request.get("file_path")
    if file_path is None:
        raise ValueError("Missing required parameter: 'file_path'")
    if not isinstance(file_path, str):
        raise ValueError("Parameter 'file_path' must be a string")

    lock = FileLock()
    info = lock.is_locked(file_path)

    if info is None:
        return {"locked": False, "lock_info": None, "stale": False}

    from ..codegen.file_lock import _pid_exists
    stale = not _pid_exists(info.pid)

    return {
        "locked": True,
        "lock_info": info.to_dict(),
        "stale": stale,
    }


def handle_release_file_lock(request: dict[str, Any]) -> dict[str, Any]:
    """Release a file lock (force-release stale locks).

    Args:
        request: Dictionary with:
            - file_path: str - Path to the file to unlock
            - force: bool (default False) - Force release even if owned by another process

    Returns:
        Dictionary with:
            - success: bool
            - error: str or None

    Raises:
        ValueError: If file_path is missing.
    """
    file_path = request.get("file_path")
    if file_path is None:
        raise ValueError("Missing required parameter: 'file_path'")
    if not isinstance(file_path, str):
        raise ValueError("Parameter 'file_path' must be a string")

    force = request.get("force", False)
    lock = FileLock()

    try:
        if force:
            lock.force_release(file_path)
        else:
            lock.release(file_path)
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Incremental Parsing Handler
# =============================================================================


def handle_incremental_parse(request: dict[str, Any]) -> dict[str, Any]:
    """Incrementally parse a project directory.

    Only re-parses files whose modification time has changed since the
    previous graph was built.  Unchanged node positions are preserved.

    Args:
        request: Dictionary with:
            - directory: str - Path to project directory
            - previous_graph: dict (optional) - Previous NodeGraph as dict

    Returns:
        NodeGraph dict with nodes and edges.

    Raises:
        ValueError: If directory is missing.
    """
    directory = request.get("directory")
    if directory is None:
        raise ValueError("Missing required parameter: 'directory'")
    if not isinstance(directory, str):
        raise ValueError("Parameter 'directory' must be a string")

    from ..ast_parser.graph_types import NodeGraph
    from ..ast_parser.incremental import incremental_parse_directory

    previous_graph = None
    raw_prev = request.get("previous_graph")
    if raw_prev is not None and isinstance(raw_prev, dict):
        previous_graph = NodeGraph.from_dict(raw_prev)

    graph = incremental_parse_directory(directory, previous_graph=previous_graph)
    return graph.to_dict()


# =============================================================================
# Default Handler Registry
# =============================================================================


def create_default_handler() -> Handler:
    """Create a Handler with all default code generation handlers registered.

    Returns:
        Handler instance with generate_code, validate_code, generate_diff,
        and apply_changes handlers registered.

    Example:
        handler = create_default_handler()
        response = handler.handle(request)
    """
    handler = Handler()

    # Register code generation handlers
    handler.register_handler("generate_code", handle_generate_code)
    handler.register_handler("validate_code", handle_validate_code)
    handler.register_handler("generate_diff", handle_generate_diff)
    handler.register_handler("apply_changes", handle_apply_changes)
    handler.register_handler("check_file_lock", handle_check_file_lock)
    handler.register_handler("release_file_lock", handle_release_file_lock)
    handler.register_handler("incremental_parse", handle_incremental_parse)

    return handler
