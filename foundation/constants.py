"""
Constants for the Core Foundation layer.

Defines system-wide constants for bounds checking, limits, and configuration.
"""
from __future__ import annotations

# =============================================================================
# UNDO/REDO LIMITS (Memory Safety)
# =============================================================================
MAX_UNDO_STACK_SIZE: int = 1000  # Prevent memory exhaustion
MAX_REDO_STACK_SIZE: int = 1000

# =============================================================================
# POOL SIZES
# =============================================================================
DEFAULT_POOL_SIZE: int = 1024
MAX_POOL_SIZE: int = 1_000_000

# =============================================================================
# TRANSACTION LIMITS
# =============================================================================
MAX_TRANSACTION_CHANGES: int = 10_000

# =============================================================================
# TRACKER LIMITS
# =============================================================================
MAX_CALLBACKS_PER_OBJECT: int = 100
MAX_DIRTY_OBJECTS: int = 100_000

# =============================================================================
# UI/INSPECTOR
# =============================================================================
INDENT_SPACES: int = 2  # Indentation for nested output
HISTORY_FIELD_WIDTH: int = 12  # Width of field name column in history view
HISTORY_TICK_WIDTH: int = 5  # Width of tick number in history view

# =============================================================================
# HASHING CONSTANTS
# =============================================================================
# Length of truncated hash strings (hex characters)
# 16 hex chars = 64 bits of entropy, sufficient for cache keys
HASH_LENGTH: int = 16

# Length of short hash for display purposes (hex characters)
# 8 hex chars = 32 bits, good for human-readable IDs
SHORT_HASH_LENGTH: int = 8

# Full SHA-256 hash length (hex characters)
FULL_HASH_LENGTH: int = 64

# Schema hash length for serialization
SCHEMA_HASH_LENGTH: int = 16

# =============================================================================
# CACHING CONSTANTS
# =============================================================================
# Default maximum entries in query cache
DEFAULT_QUERY_CACHE_SIZE: int = 1000

# Default maximum entries in content store cache
DEFAULT_CONTENT_CACHE_SIZE: int = 10000

# =============================================================================
# FILE BACKEND CONSTANTS
# =============================================================================
# Length of prefix directory for Git-style object storage
# e.g., hash "abcdef..." -> "ab/cdef..."
FILE_BACKEND_PREFIX_LENGTH: int = 2

# =============================================================================
# EVENTLOG CONSTANTS
# =============================================================================
# Default max depth for causal chains
MAX_CAUSAL_DEPTH: int = 100

__all__ = [
    # Undo/Redo
    "MAX_UNDO_STACK_SIZE",
    "MAX_REDO_STACK_SIZE",
    # Pool sizes
    "DEFAULT_POOL_SIZE",
    "MAX_POOL_SIZE",
    # Transaction limits
    "MAX_TRANSACTION_CHANGES",
    # Tracker limits
    "MAX_CALLBACKS_PER_OBJECT",
    "MAX_DIRTY_OBJECTS",
    # UI/Inspector
    "INDENT_SPACES",
    "HISTORY_FIELD_WIDTH",
    "HISTORY_TICK_WIDTH",
    # Hashing
    "HASH_LENGTH",
    "SHORT_HASH_LENGTH",
    "FULL_HASH_LENGTH",
    "SCHEMA_HASH_LENGTH",
    # Caching
    "DEFAULT_QUERY_CACHE_SIZE",
    "DEFAULT_CONTENT_CACHE_SIZE",
    # File backend
    "FILE_BACKEND_PREFIX_LENGTH",
    # EventLog
    "MAX_CAUSAL_DEPTH",
]
