"""Core layer constants. All engine-level magic numbers live here."""

# === Frame Timing ===
DEFAULT_TARGET_FPS: int = 60
DEFAULT_FIXED_TIMESTEP: float = 1.0 / 60.0
MAX_DELTA_TIME: float = 0.25  # Prevents spiral of death
DEFAULT_TIME_SCALE: float = 1.0
DEFAULT_FRAME_ALLOCATOR_SIZE: int = 1024 * 1024  # 1MB

# === Entity ===
ENTITY_INDEX_BITS: int = 24
ENTITY_GENERATION_BITS: int = 8
MAX_ENTITIES: int = (1 << ENTITY_INDEX_BITS) - 1

# === Archetype ===
DEFAULT_ARCHETYPE_CHUNK_SIZE: int = 16384
MAX_COMPONENTS: int = 256

# === Task Scheduler ===
DEFAULT_WORKER_COUNT: int = 0  # 0 = auto-detect
TASK_PRIORITY_CRITICAL: int = 0
TASK_PRIORITY_HIGH: int = 1
TASK_PRIORITY_NORMAL: int = 2
TASK_PRIORITY_LOW: int = 3
TASK_PRIORITY_IDLE: int = 4
DEFAULT_TASK_STACK_SIZE: int = 65536

# === Math Epsilon Values ===
MATH_EPSILON: float = 1e-9
MATH_EPSILON_TIGHT: float = 1e-12
SLERP_THRESHOLD: float = 0.9995

# === Component ID ===
COMPONENT_ID_MASK: int = 0xFFFFFFFFFFFFFFFF

# === Memory Allocator Defaults ===
DEFAULT_SLAB_SIZE_CLASSES: tuple = (16, 32, 64, 128, 256, 512, 1024)
DEFAULT_SLAB_SLOTS_PER_CLASS: int = 64
TLSF_SL_COUNT: int = 4
TLSF_SL_BITS: int = 2
TLSF_MIN_BLOCK: int = 16

# === Task/Worker Defaults ===
WORKER_IDLE_POLL_INTERVAL: float = 0.005
FIBER_JOIN_TIMEOUT: float = 2.0

# === Session ===
SESSION_VERSION: int = 1
MAX_CHECKPOINTS: int = 10
CHECKPOINT_ID_LENGTH: int = 12

# === Logging ===
# Log level values (lower = more verbose)
LOG_LEVEL_VERBOSE: int = 0
LOG_LEVEL_DEBUG: int = 10
LOG_LEVEL_INFO: int = 20
LOG_LEVEL_WARNING: int = 30
LOG_LEVEL_ERROR: int = 40
LOG_LEVEL_FATAL: int = 50

# File sink defaults
LOG_FILE_MAX_SIZE: int = 10 * 1024 * 1024  # 10 MB
LOG_FILE_MAX_BACKUPS: int = 5
LOG_FILE_ENCODING: str = "utf-8"

# Network sink defaults
LOG_NETWORK_TIMEOUT: float = 5.0
LOG_NETWORK_BATCH_SIZE: int = 100
LOG_NETWORK_FLUSH_INTERVAL: float = 1.0
LOG_NETWORK_RECONNECT_DELAY: float = 5.0

# Buffered sink defaults
LOG_BUFFER_SIZE: int = 100
LOG_BUFFER_FLUSH_INTERVAL: float = 1.0

# Timed rotation defaults
LOG_ROTATION_DAILY_BACKUPS: int = 7
LOG_ARCHIVER_DEFAULT_DAYS: int = 30
LOG_CLEANUP_DEFAULT_DAYS: int = 90

# Time constants (in seconds)
SECONDS_PER_MINUTE: int = 60
SECONDS_PER_HOUR: int = 3600
SECONDS_PER_DAY: int = 86400
SECONDS_PER_WEEK: int = 604800
