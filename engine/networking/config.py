"""
Network configuration constants.

Centralizes all magic numbers and configuration values for the networking module.
This ensures consistency and makes tuning easier.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class NetworkConfig:
    """
    Central configuration for all networking constants.

    Frozen dataclass to prevent accidental modification.
    """

    # ==========================================================================
    # Packet / MTU Settings
    # ==========================================================================

    # Maximum Transmission Unit - safe UDP size for Internet
    MTU: int = 1200

    # Header size in bytes for packets
    PACKET_HEADER_SIZE: int = 12

    # Maximum payload per packet (MTU - header)
    MAX_PAYLOAD_SIZE: int = 1188  # 1200 - 12

    # Fragment header size
    FRAGMENT_HEADER_SIZE: int = 4

    # Maximum sequence number (16-bit)
    MAX_SEQUENCE: int = 65535

    # ==========================================================================
    # Buffer Sizes
    # ==========================================================================

    # Default initial buffer capacity for BitWriter (bytes)
    BIT_WRITER_INITIAL_CAPACITY: int = 256

    # Socket receive buffer size
    SOCKET_RECEIVE_BUFFER_SIZE: int = 65536

    # Socket send buffer size
    SOCKET_SEND_BUFFER_SIZE: int = 65536

    # Maximum received sequences to track
    MAX_RECEIVED_SEQUENCES: int = 256

    # ==========================================================================
    # Serialization Settings
    # ==========================================================================

    # Default protocol version
    PROTOCOL_VERSION: int = 1

    # Compression threshold (compress payloads larger than this)
    COMPRESS_THRESHOLD: int = 128

    # Delta compression threshold
    DELTA_COMPRESS_THRESHOLD: int = 64

    # Default compression level (zlib)
    COMPRESSION_LEVEL: int = 6

    # Maximum string length for serialization
    MAX_STRING_LENGTH: int = 255

    # Message header size
    MESSAGE_HEADER_SIZE: int = 20

    # ==========================================================================
    # Quantization Settings
    # ==========================================================================

    # Default vector quantization precision (bits per component)
    VECTOR_PRECISION_DEFAULT: int = 16

    # Valid vector precisions
    VECTOR_PRECISIONS: tuple = (8, 12, 16, 24)

    # Default range for position vectors
    VECTOR_RANGE_MIN: float = -1000.0
    VECTOR_RANGE_MAX: float = 1000.0

    # Quaternion smallest-three encoding bounds
    QUATERNION_COMPONENT_MIN: float = -0.7071068
    QUATERNION_COMPONENT_MAX: float = 0.7071068

    # Minimum magnitude for normalization
    NORMALIZATION_EPSILON: float = 1e-10

    # ==========================================================================
    # Delta Encoding Settings
    # ==========================================================================

    # Maximum baselines to track
    MAX_BASELINES: int = 64

    # Hash truncation size for baselines
    BASELINE_HASH_SIZE: int = 8

    # ==========================================================================
    # Channel Settings
    # ==========================================================================

    # Maximum pending packets per channel
    CHANNEL_MAX_PENDING: int = 256

    # Initial RTT estimate (seconds)
    CHANNEL_INITIAL_RTT: float = 0.1

    # Maximum retry attempts for reliable packets
    CHANNEL_MAX_RETRIES: int = 10

    # ACK timeout (seconds)
    CHANNEL_ACK_TIMEOUT: float = 0.5

    # Ordering buffer size
    CHANNEL_ORDERING_BUFFER_SIZE: int = 256

    # RTT smoothing factor (EWMA alpha)
    RTT_SMOOTHING_ALPHA: float = 0.125

    # RTT variance smoothing factor (EWMA beta)
    RTT_SMOOTHING_BETA: float = 0.25

    # ACK bits field size
    ACK_BITS_COUNT: int = 32

    # ==========================================================================
    # Connection Settings
    # ==========================================================================

    # Connection timeout (seconds)
    CONNECT_TIMEOUT: float = 10.0

    # Disconnect timeout (seconds)
    DISCONNECT_TIMEOUT: float = 5.0

    # Idle timeout before disconnect (seconds)
    IDLE_TIMEOUT: float = 30.0

    # Heartbeat interval (seconds)
    HEARTBEAT_INTERVAL: float = 1.0

    # Heartbeat timeout (seconds)
    HEARTBEAT_TIMEOUT: float = 5.0

    # Maximum pending packets for connection
    CONNECTION_MAX_PENDING: int = 256

    # Maximum RTT samples to store
    MAX_RTT_SAMPLES: int = 20

    # ==========================================================================
    # Transport Settings
    # ==========================================================================

    # Maximum concurrent connections
    MAX_CONNECTIONS: int = 64

    # Maximum packets per second (rate limiting)
    MAX_PACKETS_PER_SECOND: int = 1000

    # Maximum bytes per second (1 MB/s default)
    MAX_BYTES_PER_SECOND: int = 1024 * 1024

    # Rate limit reset interval (seconds)
    RATE_LIMIT_RESET_INTERVAL: float = 1.0

    # ==========================================================================
    # Quality Settings
    # ==========================================================================

    # Quality level thresholds
    QUALITY_RTT_EXCELLENT: float = 0.050   # < 50ms
    QUALITY_RTT_GOOD: float = 0.100        # < 100ms
    QUALITY_RTT_FAIR: float = 0.200        # < 200ms
    QUALITY_RTT_POOR: float = 0.400        # < 400ms

    QUALITY_LOSS_EXCELLENT: float = 0.01   # < 1%
    QUALITY_LOSS_GOOD: float = 0.02        # < 2%
    QUALITY_LOSS_FAIR: float = 0.05        # < 5%
    QUALITY_LOSS_POOR: float = 0.10        # < 10%

    # Quality monitor window size
    QUALITY_WINDOW_SIZE: int = 100

    # Quality smoothing factor
    QUALITY_SMOOTHING_FACTOR: float = 0.1

    # Bandwidth measurement window (seconds)
    BANDWIDTH_WINDOW: float = 1.0

    # Loss window reset threshold
    LOSS_WINDOW_RESET_THRESHOLD: int = 100

    # ==========================================================================
    # Adaptive Settings
    # ==========================================================================

    # Update rates by quality level (updates per second)
    UPDATE_RATE_EXCELLENT: float = 60.0
    UPDATE_RATE_GOOD: float = 30.0
    UPDATE_RATE_FAIR: float = 20.0
    UPDATE_RATE_POOR: float = 10.0
    UPDATE_RATE_CRITICAL: float = 5.0

    # Minimum/maximum update rates
    MIN_UPDATE_RATE: float = 5.0
    MAX_UPDATE_RATE: float = 60.0

    # Interpolation delays by quality (seconds)
    INTERPOLATION_DELAY_EXCELLENT: float = 0.05
    INTERPOLATION_DELAY_GOOD: float = 0.08
    INTERPOLATION_DELAY_FAIR: float = 0.12
    INTERPOLATION_DELAY_POOR: float = 0.20
    INTERPOLATION_DELAY_CRITICAL: float = 0.30

    # Extrapolation limits (seconds)
    EXTRAPOLATION_LIMIT_EXCELLENT: float = 0.10
    EXTRAPOLATION_LIMIT_GOOD: float = 0.15
    EXTRAPOLATION_LIMIT_FAIR: float = 0.25
    EXTRAPOLATION_LIMIT_POOR: float = 0.40
    EXTRAPOLATION_LIMIT_CRITICAL: float = 0.50

    # Compression levels by quality
    COMPRESSION_LEVEL_EXCELLENT: int = 1
    COMPRESSION_LEVEL_GOOD: int = 4
    COMPRESSION_LEVEL_FAIR: int = 6
    COMPRESSION_LEVEL_POOR: int = 9
    COMPRESSION_LEVEL_CRITICAL: int = 9

    # Adapter hysteresis threshold (seconds)
    ADAPTER_HYSTERESIS_THRESHOLD: float = 2.0

    # Adapter adaptation delay (seconds)
    ADAPTER_ADAPTATION_DELAY: float = 1.0

    # Assumed bytes per update for bandwidth calculation
    ASSUMED_BYTES_PER_UPDATE: int = 100

    # Min/max interpolation delay bounds
    MIN_INTERPOLATION_DELAY: float = 0.05
    MAX_INTERPOLATION_DELAY: float = 0.50

    # ==========================================================================
    # GUID Constants
    # ==========================================================================

    # Reserved GUID values
    INVALID_GUID: int = 0xFFFFFFFF
    NULL_GUID: int = 0x00000000

    # GUID ranges
    SERVER_GUID_START: int = 0x00000001
    SERVER_GUID_MAX: int = 0x7FFFFFFF
    CLIENT_GUID_START: int = 0x80000001
    CLIENT_GUID_MAX: int = 0xFFFFFFFE

    # Client ID encoding (bits 16-30 for client ID in client-allocated GUIDs)
    CLIENT_ID_SHIFT: int = 16
    CLIENT_ID_MASK: int = 0x7FFF0000

    # Maximum valid client ID (15 bits = 0-32767)
    MAX_CLIENT_ID: int = 0x7FFF

    # 32-bit GUID mask for validation
    GUID_32BIT_MAX: int = 0xFFFFFFFF

    # Authority bit mask (bit 31)
    GUID_AUTHORITY_BIT: int = 0x80000000

    # ==========================================================================
    # Relevancy Constants
    # ==========================================================================

    # Default relevancy radius in world units
    DEFAULT_RELEVANCY_RADIUS: float = 5000.0

    # Grid cell size for spatial hashing
    DEFAULT_GRID_CELL_SIZE: float = 500.0

    # Default grid view distance (cells in each direction)
    DEFAULT_GRID_VIEW_DISTANCE: int = 5

    # Minimum relevancy priority (to prevent zero-priority items)
    MIN_RELEVANCY_PRIORITY: float = 0.1

    # Falloff multiplier for default falloff_start (percentage of radius)
    DEFAULT_FALLOFF_START_MULTIPLIER: float = 0.5

    # Default priority for entities without position data
    NO_POSITION_DEFAULT_PRIORITY: float = 0.5

    # ==========================================================================
    # Bandwidth / Priority Constants
    # ==========================================================================

    # Priority levels for replicated entities
    PRIORITY_CRITICAL: int = 100   # Must always replicate (player health, death events)
    PRIORITY_HIGH: int = 75        # Important updates (player position, weapons)
    PRIORITY_NORMAL: int = 50      # Standard updates (NPC state, objects)
    PRIORITY_LOW: int = 25         # Low-priority updates (cosmetics, distant objects)
    PRIORITY_MINIMAL: int = 10     # Lowest priority (ambient effects)

    # Default bandwidth limits (bytes per second)
    DEFAULT_MAX_BANDWIDTH_BPS: int = 128 * 1024  # 128 KB/s per connection
    DEFAULT_BURST_BANDWIDTH_BPS: int = 256 * 1024  # 256 KB/s burst allowance

    # Update rate
    DEFAULT_UPDATE_INTERVAL: float = 1.0 / 60  # 60 Hz update rate

    # Anti-starvation settings
    MAX_STARVATION_TIME_SECONDS: float = 1.0  # Max seconds before forcing an update
    STARVATION_PRIORITY_BOOST: int = 50       # Priority boost for starved entities

    # Maximum priority after starvation boost
    MAX_PRIORITY_WITH_BOOST: int = 150  # PRIORITY_CRITICAL + 50

    # Minimum packet size for updates (bytes)
    MIN_UPDATE_PACKET_SIZE: int = 32

    # Default estimated entity size for bandwidth allocation (bytes)
    DEFAULT_ENTITY_SERIALIZATION_SIZE: int = 64

    # Estimated update packet size (bytes)
    ESTIMATED_UPDATE_SIZE: int = 128

    # ==========================================================================
    # Replication Packet Constants
    # ==========================================================================

    # Packet type identifiers
    PACKET_TYPE_SPAWN: int = 0x01
    PACKET_TYPE_UPDATE: int = 0x02
    PACKET_TYPE_DESTROY: int = 0x03
    PACKET_TYPE_BATCH: int = 0x04

    # Default replication priority
    DEFAULT_REPLICATION_PRIORITY: int = 1

    # ==========================================================================
    # Actor Channel Constants
    # ==========================================================================

    # Message types for actor channel
    ACTOR_MSG_SPAWN: int = 0x01
    ACTOR_MSG_INITIAL_STATE: int = 0x02
    ACTOR_MSG_DELTA_UPDATE: int = 0x03
    ACTOR_MSG_RPC: int = 0x04
    ACTOR_MSG_CLOSE: int = 0x05
    ACTOR_MSG_ACK: int = 0x06

    # Default retransmission timeout (seconds)
    DEFAULT_RETRANSMIT_TIMEOUT: float = 0.5

    # ==========================================================================
    # RPC Constants
    # ==========================================================================

    # RPC message types
    RPC_MSG_CALL: int = 0x01
    RPC_MSG_ACK: int = 0x02
    RPC_MSG_NACK: int = 0x03
    RPC_MSG_BATCH: int = 0x04

    # Default RPC batch size
    DEFAULT_RPC_BATCH_SIZE: int = 10

    # Default max outgoing data size (bytes)
    DEFAULT_MAX_OUTGOING_DATA_SIZE: int = 4096

    # Maximum NACK reason length (bytes)
    MAX_NACK_REASON_LENGTH: int = 255

    # ==========================================================================
    # Rate Limiting Constants
    # ==========================================================================

    # Default rate limit configuration
    DEFAULT_RATE_LIMIT_MAX_CALLS: int = 10
    DEFAULT_RATE_LIMIT_WINDOW_SECONDS: float = 1.0
    DEFAULT_RATE_LIMIT_BURST_ALLOWANCE: int = 5

    # Rate limiter cleanup settings (seconds)
    RATE_LIMITER_MAX_AGE: float = 60.0

    # Call history deduplication cleanup (seconds)
    CALL_HISTORY_MAX_AGE: float = 30.0


# Global default configuration instance
DEFAULT_CONFIG = NetworkConfig()


def get_config() -> NetworkConfig:
    """Get the default network configuration."""
    return DEFAULT_CONFIG


# =============================================================================
# PREDICTION & LAG COMPENSATION CONSTANTS
# =============================================================================

import math
from typing import Final

# Mathematical constants - use these instead of hardcoding
PI: Final[float] = math.pi
TWO_PI: Final[float] = 2.0 * math.pi
HALF_PI: Final[float] = math.pi / 2.0

# -----------------------------------------------------------------------------
# Client Prediction
# -----------------------------------------------------------------------------

DEFAULT_INPUT_BUFFER_SIZE: Final[int] = 64
"""Maximum number of unconfirmed inputs to buffer."""

DEFAULT_PREDICTION_HISTORY_SIZE: Final[int] = 30
"""Number of prediction states to keep for comparison (frames)."""

DEFAULT_DELTA_TIME: Final[float] = 0.016
"""Default frame time for physics integration (~60 FPS)."""

DEFAULT_MOVE_SPEED: Final[float] = 5.0
"""Default movement acceleration (units/second^2)."""

DEFAULT_FRICTION: Final[float] = 0.9
"""Default velocity damping factor (0-1)."""

DEFAULT_JUMP_VELOCITY: Final[float] = 5.0
"""Default jump velocity (units/second)."""

DEFAULT_GRAVITY: Final[float] = -9.81
"""Default gravity acceleration (units/second^2)."""

GROUND_CHECK_TOLERANCE: Final[float] = 0.1
"""Height tolerance for ground detection (units)."""

MISPREDICTION_THRESHOLD: Final[float] = 0.01
"""Error threshold to count as misprediction (units)."""

# -----------------------------------------------------------------------------
# Server Reconciliation
# -----------------------------------------------------------------------------

DEFAULT_RECONCILIATION_SNAP_THRESHOLD: Final[float] = 0.5
"""Position error threshold for snapping vs interpolating (units)."""

DEFAULT_MATCH_THRESHOLD: Final[float] = 0.01
"""Error below this is considered a match (units)."""

DEFAULT_MAX_RECONCILE_FRAMES: Final[int] = 10
"""Maximum frames of inputs to replay during reconciliation."""

DEFAULT_VELOCITY_WEIGHT: Final[float] = 0.3
"""Weight for velocity error in comparison (0-1)."""

DEFAULT_ROTATION_WEIGHT: Final[float] = 0.2
"""Weight for rotation error in comparison (0-1)."""

DEFAULT_RECONCILIATION_HISTORY_SIZE: Final[int] = 100
"""Maximum reconciliation history frames to keep."""

# -----------------------------------------------------------------------------
# Entity Interpolation
# -----------------------------------------------------------------------------

DEFAULT_INTERPOLATION_BUFFER_SIZE: Final[int] = 3
"""Number of snapshots to buffer for interpolation."""

DEFAULT_EXTRAPOLATION_LIMIT: Final[float] = 0.25
"""Maximum time to extrapolate beyond last snapshot (seconds)."""

DEFAULT_ENTITY_INTERPOLATION_DELAY: Final[float] = 0.1
"""Fixed delay for entity interpolation (seconds)."""

QUATERNION_LERP_THRESHOLD: Final[float] = 0.9995
"""Dot product threshold above which to use lerp instead of slerp."""

# -----------------------------------------------------------------------------
# Smoothing
# -----------------------------------------------------------------------------

DEFAULT_BLEND_TIME: Final[float] = 0.1
"""Time to blend from current to target position (seconds)."""

DEFAULT_SMOOTHING_SNAP_THRESHOLD: Final[float] = 2.0
"""Error threshold above which to snap instead of interpolate (units)."""

DEFAULT_EXPONENTIAL_FACTOR: Final[float] = 10.0
"""Exponential smoothing factor (higher = faster)."""

DEFAULT_MIN_BLEND_SPEED: Final[float] = 0.5
"""Minimum blend speed (units/second)."""

DEFAULT_MAX_BLEND_SPEED: Final[float] = 50.0
"""Maximum blend speed (units/second)."""

DEFAULT_ROTATION_SNAP_THRESHOLD: Final[float] = 0.5
"""Rotation snap threshold (radians)."""

EXPONENTIAL_CONVERGENCE_THRESHOLD: Final[float] = 0.001
"""Distance threshold to consider correction complete (units)."""

# -----------------------------------------------------------------------------
# Lag Compensation - Rewind Manager
# -----------------------------------------------------------------------------

DEFAULT_MAX_REWIND_TIME_MS: Final[float] = 200.0
"""Maximum rewind history duration (milliseconds)."""

DEFAULT_TICK_RATE: Final[float] = 60.0
"""Default server tick rate (ticks/second)."""

# -----------------------------------------------------------------------------
# Lag Compensation - Hitbox History
# -----------------------------------------------------------------------------

DEFAULT_HITBOX_HISTORY_FRAMES: Final[int] = 60
"""Maximum frames of hitbox history per entity."""

HITBOX_CACHE_MULTIPLIER: Final[int] = 2
"""Frame cache size multiplier for hitbox history."""

# -----------------------------------------------------------------------------
# Lag Compensation - View Time
# -----------------------------------------------------------------------------

DEFAULT_MAX_LAG_COMPENSATION_MS: Final[float] = 200.0
"""Maximum lag compensation allowed (milliseconds)."""

DEFAULT_CLIENT_INTERPOLATION_DELAY_MS: Final[float] = 100.0
"""Client-side interpolation delay assumption (milliseconds)."""

DEFAULT_JITTER_BUFFER_MS: Final[float] = 20.0
"""Additional buffer for network jitter (milliseconds)."""

DEFAULT_MIN_RTT_SAMPLES: Final[int] = 5
"""Minimum RTT samples before using smoothed average."""

DEFAULT_RTT_HISTORY_SIZE: Final[int] = 30
"""Number of RTT samples to keep for statistics."""

JITTER_STANDARD_DEVIATIONS: Final[float] = 2.0
"""Number of standard deviations to use for jitter buffer."""

# -----------------------------------------------------------------------------
# Lag Compensation - Validator
# -----------------------------------------------------------------------------

DEFAULT_MAX_VIEW_TIME_DEVIATION_MS: Final[float] = 50.0
"""Maximum allowed deviation from expected view time (milliseconds)."""

DEFAULT_SUSPICIOUS_THRESHOLD: Final[int] = 3
"""Number of violations before flagging client as suspicious."""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ms_to_seconds(ms: float) -> float:
    """Convert milliseconds to seconds."""
    return ms / 1000.0


def seconds_to_ms(seconds: float) -> float:
    """Convert seconds to milliseconds."""
    return seconds * 1000.0


def ticks_to_ms(ticks: int, tick_rate: float = DEFAULT_TICK_RATE) -> float:
    """Convert tick count to milliseconds."""
    if tick_rate <= 0:
        return 0.0
    return (ticks / tick_rate) * 1000.0


def ms_to_ticks(ms: float, tick_rate: float = DEFAULT_TICK_RATE) -> int:
    """Convert milliseconds to tick count."""
    if tick_rate <= 0:
        return 0
    return int((ms / 1000.0) * tick_rate)


def calculate_max_history_frames(
    max_history_ms: float,
    tick_rate: float = DEFAULT_TICK_RATE,
) -> int:
    """Calculate the number of frames to store for a given history duration."""
    if tick_rate <= 0:
        return 1
    return int((max_history_ms / 1000.0) * tick_rate) + 1
