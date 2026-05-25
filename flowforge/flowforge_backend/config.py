"""FlowForge Backend Configuration.

Centralized configuration and constants for the backend.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os

# =============================================================================
# Protocol Constants
# =============================================================================

PROTOCOL_VERSION = "1.0.0"
JSONRPC_VERSION = "2.0"

# =============================================================================
# Time Constants
# =============================================================================

MILLISECONDS_PER_SECOND = 1000

# =============================================================================
# Logging
# =============================================================================

LOG_LEVELS = ("DEBUG", "INFO", "WARN", "ERROR")
DEFAULT_LOG_LEVEL = "INFO"

# =============================================================================
# IPC Error Codes (JSON-RPC 2.0 standard)
# =============================================================================

class ErrorCodes:
    """Standard JSON-RPC 2.0 error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Custom error codes (use range -32000 to -32099)
    FILE_NOT_FOUND = -32001
    PERMISSION_DENIED = -32002
    INVALID_PYTHON_SYNTAX = -32003

# =============================================================================
# Environment Configuration
# =============================================================================

@dataclass
class Config:
    """Runtime configuration loaded from environment."""
    debug: bool = False
    log_level: str = DEFAULT_LOG_LEVEL

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            debug=os.environ.get("FLOWFORGE_DEBUG", "").lower() in ("1", "true", "yes"),
            log_level=os.environ.get("FLOWFORGE_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        )

# Global config instance
config = Config.from_env()
