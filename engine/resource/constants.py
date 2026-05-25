"""Resource layer constants."""

# === Asset Handle ===
ASSET_INDEX_BITS: int = 24
ASSET_GENERATION_BITS: int = 8
MAX_ASSETS: int = (1 << ASSET_INDEX_BITS) - 1
ASSET_INDEX_MASK: int = (1 << ASSET_INDEX_BITS) - 1
ASSET_GENERATION_MASK: int = (1 << ASSET_GENERATION_BITS) - 1
NULL_ASSET_INDEX: int = ASSET_INDEX_MASK

# === Asset Loader ===
DEFAULT_LOADER_WORKERS: int = 4
DEFAULT_LOAD_PRIORITY: int = 0

# === Hot Reload ===
HOT_RELOAD_POLL_INTERVAL: float = 0.5  # seconds between polls
THREAD_JOIN_TIMEOUT_MULTIPLIER: int = 3

# === Material Render Queues ===
RENDER_QUEUE_OPAQUE: int = 2000
RENDER_QUEUE_ALPHA_TEST: int = 2450
RENDER_QUEUE_ALPHA_BLEND: int = 3000
RENDER_QUEUE_ADDITIVE: int = 3500

# === Physics Defaults ===
DEFAULT_FRICTION: float = 0.5
DEFAULT_RESTITUTION: float = 0.3

# === Mesh ===
BYTES_PER_FLOAT: int = 4
BYTES_PER_INDEX: int = 4  # 32-bit indices

# === Streaming ===
MAX_CONCURRENT_STREAMS: int = 8
AUDIO_CHUNK_SIZE: int = 4096  # samples per chunk
CHUNK_SIZE: int = 64
DEFAULT_LOAD_RADIUS: int = 3

# === Priority System ===
CRITICAL_PRIORITY_THRESHOLD: float = 0.8
HIGH_PRIORITY_THRESHOLD: float = 0.6
NORMAL_PRIORITY_THRESHOLD: float = 0.4
LOW_PRIORITY_THRESHOLD: float = 0.2
DEFAULT_DISTANCE_WEIGHT: float = 0.4
DEFAULT_SCREEN_SIZE_WEIGHT: float = 0.35
DEFAULT_FREQUENCY_WEIGHT: float = 0.25

# === Virtual Texturing ===
PAGE_SIZE: int = 128
PHYSICAL_POOL_TILES: int = 1024

# === Virtual Geometry ===
LOD_DISTANCES: tuple[float, ...] = (50.0, 100.0, 200.0, 400.0)

# === Virtual Shadow Maps ===
NUM_CLIPMAP_LEVELS: int = 6
SHADOW_PAGE_SIZE: int = 256
SHADOW_BASE_WORLD_SIZE: float = 50.0
SHADOW_RESOLUTION_MULTIPLIER: int = 4

# === Asset Pool ===
DEFAULT_POOL_CAPACITY: int = 256

# === Budget Manager ===
_BYTES_PER_MB: int = 1024 * 1024
DEFAULT_TEXTURE_BUDGET: int = 512 * _BYTES_PER_MB   # 512 MB
DEFAULT_MESH_BUDGET: int = 256 * _BYTES_PER_MB       # 256 MB
DEFAULT_AUDIO_BUDGET: int = 128 * _BYTES_PER_MB      # 128 MB

# === Package Pipeline ===
CRC32_MASK: int = 0xFFFFFFFF

# === Import Pipeline ===
DEFAULT_IMPORT_OUTPUT_PATH: str = "build/imported"
