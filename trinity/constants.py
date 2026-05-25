"""
Trinity Pattern - Centralized Constants & Configuration
========================================================
All magic numbers and configuration values in one place.
"""
from __future__ import annotations

# =============================================================================
# FIXED-POINT ARITHMETIC (Q-format numbers)
# =============================================================================
# Q8.8 format: 8 bits integer, 8 bits fraction
FIXED16_SCALE: int = 256  # 2^8 = 256 fractional positions
FIXED16_SHIFT: int = 8  # Bit shift for Q8.8

# Q16.16 format: 16 bits integer, 16 bits fraction
FIXED32_SCALE: int = 65536  # 2^16 = 65536 fractional positions
FIXED32_SHIFT: int = 16  # Bit shift for Q16.16

# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================
DEFAULT_POOL_SIZE: int = 1024  # Initial pool allocation (typical component count)
DEFAULT_POOL_MAX_SIZE: int = 65536  # Maximum pool growth
DEFAULT_POOL_GROW_FACTOR: float = 2.0  # Pool growth multiplier

CACHE_LINE_BYTES: int = 64  # Modern CPU cache line size for alignment
MEMORY_WARN_THRESHOLD: float = 0.8  # 80% - warn when approaching limit

# =============================================================================
# SCHEDULING & PARALLELIZATION
# =============================================================================
DEFAULT_PHYSICS_HZ: int = 60  # Standard physics update rate
DEFAULT_CHUNK_SIZE: int = 64  # SIMD-friendly batch size
DEFAULT_MIN_BATCH: int = 256  # Minimum items for parallelization overhead breakeven
DEFAULT_STACK_SIZE: int = 65536  # 64KB - sufficient for most game systems

# =============================================================================
# ASSET MANAGEMENT
# =============================================================================
ASSET_TYPE_CODE_LENGTH: int = 8  # Truncated type code for compact serialization

# =============================================================================
# RESOURCE MANAGEMENT
# =============================================================================
DEFAULT_RESOURCE_PRIORITY: int = 100  # Middle priority (scale: 0-200)
RESOURCE_PRIORITY_MIN: int = 0
RESOURCE_PRIORITY_MAX: int = 200

# =============================================================================
# SYSTEM MANAGEMENT
# =============================================================================
DEFAULT_SYSTEM_PRIORITY: int = 0  # Default system execution priority

# =============================================================================
# NETWORKING DEFAULTS
# =============================================================================
DEFAULT_NETWORK_PRIORITY: int = 1  # Default network update priority
DEFAULT_UPDATE_FREQUENCY: int = 0  # 0 = send every change
DEFAULT_MAX_UPDATES_PER_SECOND: float = 20.0  # Rate limit for throttled updates
DEFAULT_PREDICTION_HISTORY: int = 30  # Frames to keep for rollback
INTERPOLATION_BUFFER_SIZE: int = 3  # Snapshots for interpolation

# Valid network authority types
NETWORK_AUTHORITY_SERVER: str = "server"
NETWORK_AUTHORITY_CLIENT: str = "client"
NETWORK_AUTHORITY_OWNER: str = "owner"
VALID_NETWORK_AUTHORITIES: frozenset[str] = frozenset({
    NETWORK_AUTHORITY_SERVER,
    NETWORK_AUTHORITY_CLIENT,
    NETWORK_AUTHORITY_OWNER
})

# =============================================================================
# TYPES MODULE DEFAULTS
# =============================================================================
# PoolConfig defaults in types.py
TYPES_POOL_INITIAL_SIZE: int = 64
TYPES_POOL_MAX_SIZE: int = 1024

# BudgetConfig defaults
DEFAULT_MAX_INSTANCES: int = 10000

# ParallelConfig defaults
PARALLEL_BATCH_SIZE: int = 64
PARALLEL_MIN_ENTITIES: int = 100  # Don't parallelize below this

# =============================================================================
# PERSISTENCE & SERIALIZATION
# =============================================================================
DEFAULT_SERIALIZATION_FORMAT: str = "binary"
DEFAULT_SCHEMA_VERSION: int = 1

# =============================================================================
# METACLASSES
# =============================================================================
# State machine history tracking
DEFAULT_STATE_HISTORY_SIZE: int = 10  # Max transitions to keep in history

# Protocol version management
DEFAULT_VERSION_HISTORY_LIMIT: int = 10  # Max version entries in history

# Event system pooling
EVENT_POOL_MAX_SIZE: int = 64  # Maximum pooled instances per event type

# Asset loading queue
ASSET_QUEUE_PROCESS_BATCH: int = 10  # Max assets to process per queue tick

# Component pool & budget defaults
DEFAULT_COMPONENT_POOL_INITIAL_SIZE: int = 64  # Initial capacity for component pools
DEFAULT_COMPONENT_POOL_MAX_SIZE: int = 1024  # Maximum pool size before refusing returns
DEFAULT_COMPONENT_INSTANCE_COUNT_INITIAL: int = 0  # Starting instance count for budgeted components

__all__ = [
    # Fixed-point arithmetic
    "FIXED16_SCALE",
    "FIXED16_SHIFT",
    "FIXED32_SCALE",
    "FIXED32_SHIFT",
    # Memory management
    "DEFAULT_POOL_SIZE",
    "DEFAULT_POOL_MAX_SIZE",
    "DEFAULT_POOL_GROW_FACTOR",
    "CACHE_LINE_BYTES",
    "MEMORY_WARN_THRESHOLD",
    # Scheduling & parallelization
    "DEFAULT_PHYSICS_HZ",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_MIN_BATCH",
    "DEFAULT_STACK_SIZE",
    # Asset management
    "ASSET_TYPE_CODE_LENGTH",
    # Resource management
    "DEFAULT_RESOURCE_PRIORITY",
    "RESOURCE_PRIORITY_MIN",
    "RESOURCE_PRIORITY_MAX",
    # System management
    "DEFAULT_SYSTEM_PRIORITY",
    # Networking
    "DEFAULT_NETWORK_PRIORITY",
    "DEFAULT_UPDATE_FREQUENCY",
    "DEFAULT_MAX_UPDATES_PER_SECOND",
    "DEFAULT_PREDICTION_HISTORY",
    "INTERPOLATION_BUFFER_SIZE",
    "NETWORK_AUTHORITY_SERVER",
    "NETWORK_AUTHORITY_CLIENT",
    "NETWORK_AUTHORITY_OWNER",
    "VALID_NETWORK_AUTHORITIES",
    # Types module defaults
    "TYPES_POOL_INITIAL_SIZE",
    "TYPES_POOL_MAX_SIZE",
    "DEFAULT_MAX_INSTANCES",
    "PARALLEL_BATCH_SIZE",
    "PARALLEL_MIN_ENTITIES",
    # Persistence & serialization
    "DEFAULT_SERIALIZATION_FORMAT",
    "DEFAULT_SCHEMA_VERSION",
    # Metaclasses
    "DEFAULT_STATE_HISTORY_SIZE",
    "DEFAULT_VERSION_HISTORY_LIMIT",
    "EVENT_POOL_MAX_SIZE",
    "ASSET_QUEUE_PROCESS_BATCH",
    "DEFAULT_COMPONENT_POOL_INITIAL_SIZE",
    "DEFAULT_COMPONENT_POOL_MAX_SIZE",
    "DEFAULT_COMPONENT_INSTANCE_COUNT_INITIAL",
]
