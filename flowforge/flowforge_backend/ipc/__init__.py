"""IPC module for FlowForge Backend.

Provides the communication protocol between TypeScript frontend and Python backend.
Uses line-delimited JSON over stdin/stdout for cross-process communication.
"""

from .protocol import IPCRequest, IPCResponse, IPCError
from .handler import (
    Handler,
    create_default_handler,
    handle_generate_code,
    handle_validate_code,
    handle_generate_diff,
    handle_apply_changes,
)

__all__ = [
    # Protocol types
    "IPCRequest",
    "IPCResponse",
    "IPCError",
    # Handler class
    "Handler",
    # Factory function
    "create_default_handler",
    # Code generation handlers
    "handle_generate_code",
    "handle_validate_code",
    "handle_generate_diff",
    "handle_apply_changes",
]
