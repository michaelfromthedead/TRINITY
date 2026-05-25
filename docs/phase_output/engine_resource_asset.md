# Engine Resource Asset Investigation

**Date**: 2026-05-22
**Module**: `engine/resource/asset/`
**Classification**: REAL (Fully Functional)
**Total Lines**: 606

## Summary

The asset management subsystem provides a complete, production-ready implementation for loading, tracking, and hot-reloading assets. All modules contain real working code with proper error handling, threading support, and memory-efficient data structures. This is a high-quality implementation following game engine best practices.

## Classification

| File | Lines | Status | Description |
|------|-------|--------|-------------|
| `asset_manager.py` | 145 | REAL | Central coordinator with slot allocation, ref counting, load queue |
| `dependency_graph.py` | 107 | REAL | DAG with cycle detection, topological sort (Kahn's algorithm) |
| `asset_loader.py` | 105 | REAL | Sync and async loaders with ThreadPoolExecutor |
| `asset_handle.py` | 86 | REAL | Generational index handles with bit packing |
| `hot_reload.py` | 68 | REAL | mtime-based file watcher with background thread |
| `asset_registry.py` | 65 | REAL | Singleton registry mapping extensions to asset types |
| `__init__.py` | 30 | REAL | Module exports |

## Architecture

### AssetHandle (Generational Index Pattern)

```
+--------------------------------+
|        AssetHandle[T]          |
+--------------------------------+
| _packed: int (32 bits)         |
|   - index: 24 bits             |
|   - generation: 8 bits         |
| _asset_type: type[T] | None    |
+--------------------------------+
| is_valid() -> bool             |
| index -> int                   |
| generation -> int              |
| id -> AssetId                  |
+--------------------------------+
```

The handle system uses bit-packed generational indices:
- **Index (24 bits)**: Slot in the asset array (up to 16M assets)
- **Generation (8 bits)**: Version counter to detect stale handles (0-255 wrap)

This design prevents use-after-free bugs: when an asset is unloaded and its slot reused, the generation increments, invalidating old handles.

### AssetState Lifecycle

```
REQUESTED -> QUEUED -> LOADING -> READY
                              \-> FAILED
                              
READY -> UNLOADING -> UNLOADED (slot recycled)
```

States defined in `AssetState` enum:
- `REQUESTED` (0), `QUEUED` (1), `LOADING` (2), `LOADED` (3)
- `READY` (4), `FAILED` (5), `UNLOADING` (6), `UNLOADED` (7)

### AssetManager Flow

```
                    load(path, type)
                          |
                          v
              +------------------------+
              |  Deduplicate by path?  |
              +------------------------+
                   |             |
                  YES            NO
                   |             |
                   v             v
              ref_count++    allocate_slot()
                   |             |
                   v             v
              return handle  queue LoadRequest
                                 |
                                 v
                          update() [per frame]
                                 |
                                 v
                      loader.load(path, type)
                                 |
                        +--------+--------+
                        |                 |
                     SUCCESS           FAILURE
                        |                 |
                        v                 v
                 state = READY     state = FAILED
                 data = result     log error
```

**Key features:**
- **Path deduplication**: Same path returns existing handle with incremented ref_count
- **Free list recycling**: Freed slots are reused with incremented generation
- **Deferred loading**: `load()` returns immediately; `update()` processes queue

### AssetLoader Hierarchy

```
AssetLoader (ABC)
    |
    +-- SyncLoader
    |       - Reads file as raw bytes synchronously
    |       - No resource cleanup on unload
    |
    +-- AsyncLoader
            - Wraps inner loader with ThreadPoolExecutor
            - load_async() returns Future[LoadResult]
            - Supports callbacks via LoadRequest.callback
```

**LoadResult** dataclass:
```python
@dataclass(slots=True)
class LoadResult:
    success: bool
    data: Any = None
    error: str | None = None
```

### DependencyGraph (DAG)

```
    _deps:  asset_id -> {depends_on_ids}
    _rdeps: asset_id -> {dependent_ids}  (reverse)
    
    Example: Material depends on Texture
    
    add_dependency(material_id, texture_id)
    
    _deps[material_id] = {texture_id}
    _rdeps[texture_id] = {material_id}
```

**Key operations:**
1. **add_dependency()**: BFS cycle detection before adding edge
2. **get_load_order()**: Kahn's algorithm for topological sort
3. **get_dependents()**: Get assets that depend on a given asset (for hot reload propagation)
4. **remove()**: Clean removal of all forward and reverse edges

**Cycle detection** via BFS path search before edge insertion.

### HotReloadWatcher

```
+---------------------------+
|    HotReloadWatcher       |
+---------------------------+
| _watches: dict            |
|   path -> (mtime, callback)|
| _interval: float          |
| _thread: Thread (daemon)  |
+---------------------------+
| register(path, callback)  |
| unregister(path)          |
| poll() -> list[str]       |
| start() / stop()          |
+---------------------------+
```

**Implementation details:**
- Uses `os.path.getmtime()` polling (not inotify/FSEvents)
- Background daemon thread with configurable interval
- Thread-safe via `threading.Lock`
- Returns list of changed paths from `poll()`

### AssetRegistry

```
Extension Mapping (Singleton):
    .png, .jpg, .jpeg -> AssetType.TEXTURE
    .obj, .fbx        -> AssetType.MESH
    .wav, .mp3        -> AssetType.AUDIO
    .json             -> AssetType.DATA_TABLE
    .glsl, .hlsl      -> AssetType.SHADER
```

Uses `ClassVar[AssetRegistry | None]` for singleton pattern with `reset()` for testing.

## Dependencies

### Internal
- `engine.resource.constants`: Bit masks, default values

### External (Standard Library)
- `collections.deque`: Load queue
- `concurrent.futures`: ThreadPoolExecutor, Future
- `threading`: Lock, Thread
- `os`: File mtime
- `logging`: Error reporting
- `enum`: AssetState, AssetType
- `abc`: AssetLoader interface
- `dataclasses`: LoadResult, LoadRequest
- `typing`: Generic, TypeVar

## Patterns Identified

### 1. Generational Index Pattern
- Prevents dangling references to freed assets
- Compact memory layout (single packed int)
- O(1) validity check

### 2. Slot Allocator with Free List
- Reuses freed slots to avoid array fragmentation
- Generation counter prevents use-after-free
- O(1) allocation and deallocation

### 3. Reference Counting
- Automatic unload when ref_count hits zero
- Deduplication via path-to-index map

### 4. Deferred Loading Queue
- `load()` is non-blocking
- `update()` processes queue (call per frame)
- Clean separation of request and execution

### 5. Decorator Pattern (AsyncLoader)
- Wraps any AssetLoader with thread pool
- Preserves sync `load()` interface
- Adds `load_async()` for non-blocking use

## Memory Management

| Structure | Growth Pattern | Notes |
|-----------|---------------|-------|
| `_entries` | Append-only + free list | Slots never removed, only recycled |
| `_path_to_index` | Grows with unique paths | Cleaned on unload |
| `_free_list` | Bounded by max concurrent | Recycled slots |
| `_load_queue` | Transient | Drained each frame |

## Thread Safety Analysis

| Component | Thread Safety | Notes |
|-----------|--------------|-------|
| `AssetManager` | NOT thread-safe | Single-threaded update assumed |
| `AsyncLoader` | Partial | Future callbacks run on worker threads |
| `HotReloadWatcher` | Thread-safe | Lock protects _watches dict |
| `DependencyGraph` | NOT thread-safe | Caller must synchronize |
| `AssetRegistry` | NOT thread-safe | Singleton init race possible |

## API Reference

### AssetManager
```python
manager = AssetManager(loader=AsyncLoader())
handle = manager.load("textures/diffuse.png", TextureAsset)
manager.update()  # Process load queue
data = manager.get(handle)  # Returns TextureAsset | None
state = manager.get_state(handle)  # AssetState
manager.unload(handle)  # Decrement ref count
```

### DependencyGraph
```python
graph = DependencyGraph()
graph.add_dependency(material_id, texture_id)  # material depends on texture
order = graph.get_load_order([material_id])  # [texture_id, material_id]
dependents = graph.get_dependents(texture_id)  # {material_id}
```

### HotReloadWatcher
```python
watcher = HotReloadWatcher(interval=0.5)
watcher.register("assets/shader.glsl", on_shader_changed)
watcher.start()
# ...
watcher.stop()
```

## Integration Points

1. **Rendering**: Materials, shaders, textures use AssetHandle
2. **Audio**: Audio sources reference assets via handles
3. **Scene loading**: DependencyGraph orders asset loads
4. **Editor**: HotReloadWatcher enables live editing

## Gaps and Recommendations

### Current Gaps

1. **No async completion notification**: `update()` is synchronous drain
2. **No priority queue**: All loads are FIFO
3. **No memory budget**: Unlimited loading
4. **No streaming support**: Full file load only
5. **Registry not integrated**: AssetManager ignores registry for type inference

### Recommendations

1. **Add asset priority**: Use `heapq` instead of `deque` for load queue
2. **Memory pressure callbacks**: Evict low-priority assets when budget exceeded
3. **Streaming API**: Chunked loading for large assets (meshes, textures)
4. **Type inference**: Use AssetRegistry in load() for automatic type detection
5. **Thread-safe singleton**: Use `threading.Lock` in AssetRegistry.instance()

## Quality Assessment

| Aspect | Score | Notes |
|--------|-------|-------|
| **Completeness** | 9/10 | Full pipeline, minor gaps noted |
| **Correctness** | 10/10 | Proper algorithms (Kahn's, BFS cycle detection) |
| **Performance** | 8/10 | Efficient data structures, could add priority |
| **Maintainability** | 9/10 | Clean separation, good docstrings |
| **Thread Safety** | 6/10 | Partial coverage, needs documentation |

**Overall**: Production-ready with minor enhancements needed for advanced scenarios.
