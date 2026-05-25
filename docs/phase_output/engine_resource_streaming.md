# Engine Resource Streaming Investigation

## Overview

**Location:** `engine/resource/streaming/`
**Total Files:** 7
**Total Lines:** 545
**Classification:** PARTIAL - Architecture is production-ready, but I/O is synchronous simulation

The streaming subsystem provides a comprehensive resource streaming framework with priority queue management, specialized managers for different asset types (texture mip-levels, mesh LODs, audio chunks, world chunks), and a configurable priority calculation system. The data structures and algorithms are real; the actual I/O is stub-level.

## Architecture Summary

```
StreamManager (Central Coordinator)
    |
    +-- Priority Queue (heapq-based)
    |       |
    |       +-- StreamRequest (PENDING -> ACTIVE -> COMPLETE)
    |
    +-- StreamPriorityCalculator
    |       |
    |       +-- Distance-based scoring
    |       +-- Screen-size scoring
    |       +-- Frequency scoring
    |       +-- Bucket classification (CRITICAL/HIGH/NORMAL/LOW/BACKGROUND)
    |
    +-- Specialized Managers
            |
            +-- TextureStreamManager (mip-level streaming)
            +-- MeshStreamManager (LOD streaming)
            +-- AudioStreamManager (chunk ring buffer)
            +-- WorldStreamManager (chunk loading by camera position)
```

## File Analysis

### 1. stream_manager.py (141 lines) - PARTIAL

**Purpose:** Central streaming coordinator with priority queue management.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `StreamType` | Enum | Asset categories: TEXTURE_MIP, MESH_LOD, AUDIO_CHUNK, WORLD_CHUNK |
| `StreamState` | Enum | Lifecycle: PENDING, ACTIVE, COMPLETE, CANCELLED, FAILED |
| `StreamPriority` | IntEnum | Priority levels: CRITICAL(0) to BACKGROUND(4) |
| `StreamRequest` | Dataclass | Request with auto-incrementing ID, priority-based ordering |
| `StreamManager` | Class | Priority queue coordinator with concurrent stream limiting |

**Implementation Details:**
- Uses `heapq` for O(log n) priority queue operations
- Thread-safe auto-incrementing request IDs via `itertools.count()`
- `__slots__` optimization on all classes
- Supports concurrent stream limiting via `MAX_CONCURRENT_STREAMS` (default: 8)
- Progress simulation advances `bytes_loaded` toward `bytes_total` (STUB: no real I/O)

**Key Methods:**
```python
request_stream(asset_id, priority, stream_type) -> StreamRequest
cancel(request_id) -> bool
update() -> None  # Processes pending, advances active (synchronous simulation)
get_active_streams() -> list[StreamRequest]
get_pending_count() -> int
```

**Evidence of Stub I/O:**
```python
# stream_manager.py:115-129 - "Simulate progress"
for rid, req in self._active.items():
    if req.bytes_total > 0:
        req.bytes_loaded = min(req.bytes_loaded + req.bytes_total, req.bytes_total)
        if req.bytes_loaded >= req.bytes_total:
            req.state = StreamState.COMPLETE
    else:
        # Zero-size requests complete immediately.
        req.state = StreamState.COMPLETE
```

### 2. priority_system.py (96 lines) - REAL

**Purpose:** Priority calculation using distance, screen coverage, and usage frequency.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `PriorityBucket` | IntEnum | CRITICAL(0) to BACKGROUND(4) |
| `PriorityWeights` | Dataclass | Configurable weights (distance=0.4, screen_size=0.35, frequency=0.25) |
| `StreamPriorityCalculator` | Class | Weighted scoring with bucket classification |

**Priority Formula:**
```python
distance_score = 1.0 / (1.0 + distance)
priority = (
    weight.distance * distance_score +
    weight.screen_size * screen_size +
    weight.frequency * frequency +
    base_priority
)
# Clamped to [0, 1]
```

**Bucket Thresholds:**
| Score | Bucket |
|-------|--------|
| >= 0.8 | CRITICAL |
| >= 0.6 | HIGH |
| >= 0.4 | NORMAL |
| >= 0.2 | LOW |
| < 0.2 | BACKGROUND |

### 3. texture_streaming.py (59 lines) - STUB

**Purpose:** Mip-level streaming for textures.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `MipStreamRequest` | Dataclass | texture_id, target_mip_level, current_mip_level |
| `TextureStreamManager` | Class | Mip request queue with resident mip tracking |

**Implementation Details:**
- Lower mip level = higher priority (more detail)
- Maintains `_resident_mips` dict tracking loaded mip per texture
- `update()` keeps best (lowest) mip level per texture from pending requests
- **STUB:** No actual texture loading, just state tracking

### 4. mesh_streaming.py (43 lines) - STUB

**Purpose:** LOD streaming for meshes.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `LODStreamRequest` | Dataclass | mesh_id, target_lod, current_lod |
| `MeshStreamManager` | Class | LOD request queue with resident LOD tracking |

**Implementation Details:**
- Similar pattern to TextureStreamManager
- Maintains `_resident_lods` dict tracking current LOD per mesh
- `update()` applies all pending LOD requests directly
- **STUB:** No actual mesh loading, just state tracking

### 5. audio_streaming.py (59 lines) - STUB

**Purpose:** Audio chunk streaming with ring buffer.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `AudioChunk` | Dataclass | chunk_index, sample_offset, sample_count, data (bytes) |
| `AudioStreamManager` | Class | Ring buffer per audio asset |

**Implementation Details:**
- Uses `AUDIO_CHUNK_SIZE` (4096 samples) per chunk
- `_buffers`: `dict[str, dict[int, AudioChunk]]` - per-audio ring buffer
- `request_chunks(audio_id, start_chunk, count)` queues range requests
- `get_buffered_range(audio_id)` returns (min_chunk, max_chunk+1)
- **STUB:** Fills chunks with placeholder data (`b"\x00" * AUDIO_CHUNK_SIZE`)

**Evidence:**
```python
# audio_streaming.py:56-57
data=b"\x00" * AUDIO_CHUNK_SIZE,  # Zero-filled placeholder
```

### 6. world_streaming.py (100 lines) - PARTIAL

**Purpose:** World chunk streaming based on camera position.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `ChunkState` | Enum | UNLOADED, LOADING, LOADED, UNLOADING |
| `WorldChunk` | Dataclass | chunk_x, chunk_y, state |
| `WorldStreamManager` | Class | Camera-driven chunk loading/unloading |

**Implementation Details:**
- Uses `CHUNK_SIZE` (64 units) for world grid
- Default load radius: 3 chunks
- Camera position converted to chunk coordinates
- Desired chunks computed as square region around camera
- State machine: UNLOADED -> LOADING -> LOADED -> UNLOADING -> (removed)
- **PARTIAL:** State machine works, but LOADING -> LOADED is instant (no async)

**Key Methods:**
```python
update_camera(x, y) -> None  # Triggers chunk load/unload
get_loaded_chunks() -> list[WorldChunk]
get_loading_radius() -> int
set_loading_radius(r) -> None
```

### 7. __init__.py (47 lines) - REAL

**Purpose:** Module exports aggregating all streaming components.

**Exports:** 18 public symbols including all managers, enums, dataclasses, and constants.

## Configuration Constants

From `engine/resource/constants.py`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_CONCURRENT_STREAMS` | 8 | Maximum active streams |
| `AUDIO_CHUNK_SIZE` | 4096 | Samples per audio chunk |
| `CHUNK_SIZE` | 64 | World chunk grid size (units) |
| `DEFAULT_LOAD_RADIUS` | 3 | Chunks to load around camera |
| `CRITICAL_PRIORITY_THRESHOLD` | 0.8 | Priority bucket threshold |
| `HIGH_PRIORITY_THRESHOLD` | 0.6 | Priority bucket threshold |
| `NORMAL_PRIORITY_THRESHOLD` | 0.4 | Priority bucket threshold |
| `LOW_PRIORITY_THRESHOLD` | 0.2 | Priority bucket threshold |
| `DEFAULT_DISTANCE_WEIGHT` | 0.4 | Priority calculation weight |
| `DEFAULT_SCREEN_SIZE_WEIGHT` | 0.35 | Priority calculation weight |
| `DEFAULT_FREQUENCY_WEIGHT` | 0.25 | Priority calculation weight |

**Unused Budget Constants:**
```python
DEFAULT_TEXTURE_BUDGET: int = 512 * _BYTES_PER_MB   # 512 MB
DEFAULT_MESH_BUDGET: int = 256 * _BYTES_PER_MB       # 256 MB
DEFAULT_AUDIO_BUDGET: int = 128 * _BYTES_PER_MB      # 128 MB
# These are never referenced in the streaming code
```

## Classification Summary

| File | Lines | Status | Rationale |
|------|-------|--------|-----------|
| stream_manager.py | 141 | PARTIAL | Heapq priority queue real, I/O simulation |
| priority_system.py | 96 | REAL | Complete weighted scoring with bucket classification |
| texture_streaming.py | 59 | STUB | State tracking only, no texture loading |
| mesh_streaming.py | 43 | STUB | State tracking only, no mesh loading |
| audio_streaming.py | 59 | STUB | Zero-filled placeholder data |
| world_streaming.py | 100 | PARTIAL | State machine works, loading is instant |
| __init__.py | 47 | REAL | Clean module exports |

**Overall: PARTIAL IMPLEMENTATION**

## What Works (REAL)

1. **Priority Queue** - heapq-based StreamManager correctly orders requests
2. **Priority Calculation** - Weighted scoring with distance/screen_size/frequency
3. **Bucket Classification** - Maps priority scores to CRITICAL/HIGH/NORMAL/LOW/BACKGROUND
4. **State Machines** - StreamRequest and WorldChunk lifecycle states
5. **Concurrent Limits** - MAX_CONCURRENT_STREAMS enforced
6. **Cancellation** - Request cancellation with proper state updates

## What Is Missing (STUB)

1. **Async I/O** - No asyncio, threading, or file reads
2. **Budget Enforcement** - Memory limits defined but never checked
3. **Asset Integration** - No connection to asset loader/deserializer
4. **GPU Upload** - No texture/mesh GPU memory management
5. **Archive Streaming** - No packaged asset support
6. **Prefetching** - No velocity-based prediction

## Design Patterns

1. **Priority Queue Pattern** - StreamManager uses heapq for efficient request ordering
2. **State Machine Pattern** - StreamRequest and WorldChunk use explicit state enums
3. **Strategy Pattern** - PriorityWeights allows configurable scoring
4. **Ring Buffer Pattern** - AudioStreamManager maintains per-asset chunk buffers
5. **Slots Optimization** - All dataclasses and classes use `__slots__` for memory efficiency

## Dependencies

**Internal:**
- `engine.resource.constants` - Configuration values

**External:**
- `heapq` - Priority queue operations
- `itertools` - Request ID generation
- `dataclasses` - Data structure definitions
- `enum` - State and type enums

## Integration Points

1. **ResourceManager** - Central streaming requests via `StreamManager.request_stream()`
2. **Renderer** - Texture mip and mesh LOD requests
3. **Audio System** - Audio chunk streaming
4. **World System** - Chunk loading based on camera position
5. **Budget Manager** - Memory budget tracking (defined but not connected)

## Quality Assessment

**Strengths:**
- Clean separation of concerns between asset types
- Efficient heapq-based priority queue
- Memory-optimized with `__slots__`
- Well-defined state machines
- Configurable priority weights
- Production-ready data structures

**Missing for Production:**
- `asyncio` or thread-pool based actual file I/O
- Budget enforcement (eviction when over memory limit)
- Integration with asset loader for actual data deserialization
- GPU upload handling for textures/meshes
- Streaming from packaged archives
- Prefetch/prediction based on player velocity

## Usage Example

```python
from engine.resource.streaming import (
    StreamManager, StreamPriority, StreamType,
    StreamPriorityCalculator, PriorityWeights,
    TextureStreamManager, MeshStreamManager,
    AudioStreamManager, WorldStreamManager
)

# Central coordinator
stream_mgr = StreamManager()

# Request texture stream
req = stream_mgr.request_stream(
    "texture_001",
    priority=StreamPriority.HIGH,
    stream_type=StreamType.TEXTURE_MIP
)

# Calculate dynamic priority
calc = StreamPriorityCalculator()
priority = calc.calculate_priority(
    distance=10.0,
    screen_size=0.5,
    frequency=0.8
)
bucket = calc.classify(priority)  # PriorityBucket.HIGH

# World streaming
world_mgr = WorldStreamManager()
world_mgr.update_camera(150.5, 200.3)
loaded = world_mgr.get_loaded_chunks()
```
