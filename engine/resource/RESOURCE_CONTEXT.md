# RESOURCE_CONTEXT.md — Resource & Asset Management Layer

> **Purpose**: Complete implementation reference for the engine/resource/ layer.
> Read this file and ONLY this file when implementing resource management systems.

**Implementation Status:**
| Layer | Status | Notes |
|-------|--------|-------|
| Python | ✅ COMPLETE | Asset pipeline, handles, streaming |
| Rust | ❌ 0% | GAPSET_12_ASSETS — BUILD TARGET |
| Wired | ❌ No | Blocked on Rust asset pipeline |

*See `docs/STATUS.md` for current progress.*

---

## 1. Architecture Summary

The resource layer manages the complete lifecycle of all game assets -- from source files through import, processing, cooking, and packaging to runtime loading, streaming, and unloading. It sits between Core Systems (below) and all consuming layers above (Rendering, Simulation, Animation, Audio, Gameplay).

**Core Subsystems (6):**
1. **Asset Pipeline** -- Import, process, cook, package. Transforms source assets into optimized runtime formats with deterministic builds and incremental processing.
2. **Asset Handle System** -- Typed references with reference counting, async loading, priority queues, and dependency-ordered loading.
3. **Streaming** -- World streaming (level chunks), texture mip streaming, mesh LOD streaming, audio streaming. Distance-based with memory budgets and priority systems.
4. **Virtualization** -- Virtual texturing (page tables, feedback buffers, physical page pools), virtual geometry (Nanite-style cluster DAG), virtual shadow maps. Enables effectively unlimited resolution/detail.
5. **Hot-Reload** -- File watching, change detection, reload notification. Development-time iteration with live updates.
6. **Build System** -- Dependency graph, incremental builds, platform cooking, distributed builds, packaging with manifests.

**Pipeline Models:**
- **Import** -- Source format decode (FBX, glTF, USD, PNG, WAV, etc.)
- **Process** -- Optimization (vertex cache, LOD generation, mip generation, compression)
- **Cook** -- Platform-specific format conversion (BC1-7/ASTC/ETC2, DXIL/SPIR-V/Metal)
- **Package** -- Archive creation (PAK files, manifests, alignment, patch-friendly layout)

**Asset Identification:**
- Virtual paths (`/Game/Characters/Hero`, `/Engine/Materials/Default`)
- Content hashes (SHA-256, content-addressable via Foundation ContentStore)
- GUIDs (persistent across renames via redirectors)
- Mount points (`/Game`, `/Engine`, `/Plugin`)

**Reference Types:**
| Type | Description | Lifetime |
|------|-------------|----------|
| Hard Reference | Always loaded together | Owner lifetime |
| Soft Reference | Path only, lazy load | Independent |
| Weak Reference | No ownership, may invalidate | None |
| Async Reference | Streamed on demand | Budget-managed |

**Resource Layer Position in Engine Stack:**
```
Rendering / Simulation / Animation / Audio / Gameplay
    |
RESOURCE LAYER (this layer)
    Asset Pipeline, Build System, Runtime Loading, Streaming, Virtualization
    |
CORE SYSTEMS
    Memory, Math, Task System, ECS, I/O
    |
PLATFORM LAYER
    OS, Graphics API, File System
```

**Dependency Chain:**
```
Platform (OS/FileSystem/GPU)
    -> Core (Memory/Math/Tasks/ECS)
        -> RESOURCE <- THIS LAYER
            -> Rendering (textures, meshes, shaders, materials)
            -> Simulation (physics assets, collision meshes)
            -> Animation (skeletons, clips, blend spaces)
            -> Audio (sound waves, cues, mixes)
            -> Gameplay (prefabs, data tables, levels)
```

---

## 2. Decorators (Complete Reference)

### 2.1 Asset Pipeline Decorators (Tier 8 -- assets.py)

#### @asset
```python
@asset(
    extensions: tuple[str, ...],       # REQUIRED, non-empty. File extensions (e.g., (".png", ".jpg"))
    loader: Optional[Callable] = None, # Custom loader callable
)
```
- **Config:** `AssetConfig(extensions: tuple[str, ...], loader: Optional[Callable])`
- **Steps:** TAG(asset=True), TAG(asset_config), REGISTER(assets), DESCRIBE()
- **After-Apply:** `_asset`, `_asset_extensions`, `_asset_loader`, `_asset_config`
- **Validation:** `extensions` must be non-empty tuple or list

#### @cook
```python
@cook(
    platform: Optional[str] = None,  # Target platform (None = all)
    compression: str = "lz4",        # "none" | "lz4" | "zstd"
    strip_debug: bool = True,        # Strip debug info in release
)
```
- **Config:** `CookConfig(platform: Optional[str], compression: str, strip_debug: bool)`
- **Steps:** TAG(cook=True), TAG(cook_config), REGISTER(assets)
- **After-Apply:** `_cook`, `_cook_platform`, `_cook_compression`, `_cook_strip_debug`, `_cook_config`
- **Validation:** `compression` must be in `{"none", "lz4", "zstd"}`

#### @preload
```python
@preload(
    priority: int = 0,  # Higher = load earlier
)
```
- **Steps:** TAG(preload=True), TAG(preload_priority), REGISTER(assets)
- **After-Apply:** `_preload`, `_preload_priority`

#### @residency
```python
@residency(
    priority: str = "normal",  # "critical" | "high" | "normal" | "low" | "evictable"
    min_mip: int = 0,          # Minimum mip level always resident (>= 0)
)
```
- **Config:** `ResidencyConfig(priority: str, min_mip: int)`
- **Steps:** TAG(residency=True), TAG(residency_config), REGISTER(assets), VALIDATE(valid_priority)
- **After-Apply:** `_residency`, `_residency_priority`, `_residency_min_mip`, `_residency_config`
- **Validation:** `priority` must be in `{"critical", "high", "normal", "low", "evictable"}`; `min_mip` >= 0

#### @import_settings
```python
@import_settings(
    scale: float = 1.0,                                # Import scale factor
    axis_conversion: tuple[str, str, str] = ("X", "Y", "Z"),  # Axis mapping
    merge_meshes: bool = False,                        # Merge sub-meshes
)
```
- **Config:** `ImportSettingsConfig(scale: float, axis_conversion: tuple[str, str, str], merge_meshes: bool)`
- **Steps:** TAG(import_settings=True), TAG(import_settings_config), REGISTER(assets)
- **After-Apply:** `_import_settings`, `_import_scale`, `_import_axis_conversion`, `_import_merge_meshes`, `_import_settings_config`

### 2.2 LOD & Streaming Decorators (Tier 31 -- lod_streaming.py)

#### @lod
```python
@lod(
    levels: int = 4,                    # Number of LOD levels (> 0)
    distances: Optional[list[float]] = None,  # Transition distances (len == levels, strictly ascending, all > 0)
    bias: float = 0.0,                  # LOD bias offset
)
```
- **Steps:** TAG(lod=True), TAG(lod_levels), TAG(lod_distances), TAG(lod_bias), REGISTER(lod_streaming)
- **After-Apply:** `_lod`, `_lod_levels`, `_lod_distances`, `_lod_bias`
- **Validation:** `levels` must be int > 0; if `distances` provided, length must equal `levels`, all > 0, strictly ascending

#### @streamable
```python
@streamable(
    priority: str = "normal",  # "critical" | "high" | "normal" | "low"
    keep_loaded: bool = False, # Never unload once loaded
)
```
- **Steps:** TAG(streamable=True), TAG(stream_priority), TAG(stream_keep_loaded), REGISTER(lod_streaming)
- **After-Apply:** `_streamable`, `_stream_priority`, `_stream_keep_loaded`
- **Validation:** `priority` must be in `{"critical", "high", "normal", "low"}`

#### @chunk
```python
@chunk(
    size: tuple[float, float, float],  # REQUIRED, all > 0. World chunk dimensions
    overlap: float = 0.0,              # Overlap between chunks (>= 0)
)
```
- **Steps:** TAG(chunk=True), TAG(chunk_size), TAG(chunk_overlap), REGISTER(lod_streaming)
- **After-Apply:** `_chunk`, `_chunk_size`, `_chunk_overlap`
- **Validation:** `size` required, must be 3-tuple, all values > 0; `overlap` >= 0

#### @loading_priority
```python
@loading_priority(
    visibility_weight: float = 1.0,        # Weight for screen visibility (>= 0)
    player_velocity_weight: float = 1.0,   # Weight for player movement direction (>= 0)
)
```
- **Steps:** TAG(loading_priority=True), TAG(loading_priority_visibility_weight), TAG(loading_priority_player_velocity_weight), REGISTER(lod_streaming)
- **After-Apply:** `_loading_priority`, `_loading_priority_visibility_weight`, `_loading_priority_player_velocity_weight`
- **Validation:** Both weights must be >= 0

#### @unloadable
```python
@unloadable(
    min_age: float = 60.0,    # Minimum seconds before eligible for unload (> 0)
    save_state: bool = True,  # Save state before unloading
)
```
- **Steps:** TAG(unloadable=True), TAG(unloadable_min_age), TAG(unloadable_save_state), REGISTER(lod_streaming)
- **After-Apply:** `_unloadable`, `_unloadable_min_age`, `_unloadable_save_state`
- **Validation:** `min_age` must be > 0

### 2.3 Data Flow Decorators (Tier 4 -- data_flow.py)

#### @serializable
```python
@serializable(
    format: str = "binary",  # "binary" | "json" | "msgpack"
    version: int = 1,        # Schema version (positive integer)
)
```
- **Config:** `SerializableConfig(format: str, version: int)`
- **Steps:** TAG(serializable=True), TAG(serializable_config), REGISTER(data_flow), DESCRIBE()
- **After-Apply:** `_serializable`, `_serializable_format`, `_serializable_version`, `_serializable_fields`, plus `serialize()`/`deserialize()` classmethods

#### @versioned
```python
@versioned(
    version: int = 1,                # Schema version (positive integer)
    migrations: Optional[dict] = None,  # Migration functions by version
)
```
- **Config:** `VersionedConfig(version: int, migrations: dict)`
- **Steps:** TAG(versioned=True), TAG(versioned_config), REGISTER(data_flow), VALIDATE(requires_serializable)
- **After-Apply:** `_versioned`, `_versioned_version`, `_versioned_migrations`
- **Requires:** `@serializable` must be applied first

#### @snapshot
```python
@snapshot(
    history_frames: int = 60,  # Number of frames to keep in history (positive integer)
)
```
- **Config:** `SnapshotConfig(history_frames: int)`
- **Steps:** TAG(snapshot=True), TAG(snapshot_config), REGISTER(data_flow)
- **After-Apply:** `_snapshot`, `_snapshot_history_frames`, plus `snapshot_save()`/`snapshot_restore()` methods
- **Requires:** `@serializable` must be applied first

### 2.4 Memory Decorators (Tier 2 -- memory.py)

#### @pooled
```python
@pooled(
    initial_size: int = 1024,          # Initial pool capacity (DEFAULT_POOL_SIZE)
    grow_factor: float = 2.0,          # Growth multiplier (DEFAULT_POOL_GROW_FACTOR)
    max_size: Optional[int] = None,    # Maximum pool size (None = unlimited)
)
```
- **Config:** `PoolConfig(initial_size: int, grow_factor: float, max_size: Optional[int])`
- **Steps:** TAG(pool={initial_size, grow_factor, max_size}), HOOK(on_create), HOOK(on_destroy), REGISTER(PoolManager)
- **After-Apply:** `_pooled`, `_pool_config`, `release()` method

#### @packed
```python
@packed(
    layout: str = "soa",  # "aos" | "soa" | "hybrid"
)
```
- **Config:** `PackedConfig(layout: str)`
- **Steps:** TAG(memory={layout})
- **After-Apply:** `_packed`, `_packed_layout`, `_packed_config`

#### @aligned
```python
@aligned(
    bytes: int = 64,  # Alignment in bytes (must be positive power of 2; default CACHE_LINE_BYTES=64)
)
```
- **Config:** `AlignedConfig(bytes: int)`
- **Steps:** TAG(memory={alignment})
- **After-Apply:** `_aligned`, `_aligned_bytes`, `_aligned_config`
- **Validation:** Must be a positive power of 2

#### @arena
```python
@arena(
    name: str = "default",  # Arena name for grouped allocation
)
```
- **Config:** `ArenaConfig(name: str)`
- **Steps:** TAG(memory={allocator: "arena"}), HOOK(on_create)
- **After-Apply:** `_arena`, `_arena_name`, `_arena_config`

#### @budget
```python
@budget(
    category: str,                     # REQUIRED. Budget category name (e.g., "gpu", "texture", "mesh")
    max_bytes: Optional[int] = None,   # Maximum bytes (None = unlimited)
    warn_at: float = 0.8,             # Warning threshold (0.0-1.0; default MEMORY_WARN_THRESHOLD)
)
```
- **Config:** `BudgetConfig(category: str, max_bytes: Optional[int], warn_at: float)`
- **Steps:** TAG(resource={budget, max_bytes}), VALIDATE(budget_limit)
- **After-Apply:** `_budget`, `_budget_category`, `_budget_max_bytes`, `_budget_warn_at`, `_budget_config`
- **Validation:** `warn_at` must be 0.0-1.0

#### @allocator
```python
@allocator(
    type: str,             # REQUIRED. "linear" | "pool" | "stack" | "buddy" | "tlsf"
    size: int,             # REQUIRED. Allocator size in bytes (must be > 0)
    thread_safe: bool = False,
)
```
- **Config:** `AllocatorConfig(type: str, size: int, thread_safe: bool)`
- **Steps:** TAG(memory={allocator: type})
- **After-Apply:** `_allocator`, `_allocator_type`, `_allocator_size`, `_allocator_thread_safe`, `_allocator_config`
- **Validation:** `size` must be > 0

#### @flyweight
```python
@flyweight
```
- **Steps:** TAG(memory={shared: True}), INTERCEPT(get=lookup), REGISTER(FlyweightCache)
- **After-Apply:** `_flyweight`, `_flyweight_registry`, `_flyweight_next_id`, `get_by_id()` classmethod, `unregister()` method

#### @intern
```python
@intern
```
- **Steps:** TAG(memory={interned: True}), INTERCEPT(get=intern_lookup)
- **After-Apply:** `_intern`, `_intern_table`, `intern_string()` classmethod

#### @generations
```python
@generations
```
- **Steps:** TAG(memory={generational: True}), TRACK()
- **After-Apply:** `_generations`, `_generation_counters`, `is_generation_valid()` method

#### @copy_on_write
```python
@copy_on_write
```
- **Steps:** TAG(memory={cow: True}), INTERCEPT(set=copy_then_write)
- **After-Apply:** `_copy_on_write`, custom `__setattr__`, `cow_clone()` method

#### @inline_array
```python
@inline_array(
    size: int,  # REQUIRED. Fixed array size (must be > 0)
)
```
- **Config:** `InlineArrayConfig(size: int)`
- **Steps:** TAG(memory={inline: True, size})
- **After-Apply:** `_inline_array`, `_inline_array_size`, `_inline_array_config`
- **Validation:** `size` must be > 0

#### @atomic
```python
@atomic
```
- **Steps:** INTERCEPT(set=atomic, get=atomic), TAG(thread_safe=True)
- **After-Apply:** `_atomic`, `_atomic_lock`, `atomic_load()`, `atomic_store()`, `atomic_exchange()`, `compare_exchange()`, `fetch_add()`, `fetch_sub()` methods

---

## 3. Metaclasses

### 3.1 AssetMeta (PRIMARY for asset types)

```python
class AssetMeta(EngineMeta):
    _registry: ClassVar[dict[int, type]] = {}
    _extension_map: ClassVar[dict[str, type]] = {}
    _next_id: ClassVar[int] = 1
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _load_queue: ClassVar[list] = []       # min-heap: (-priority, counter, cls, path, cb)
    _load_counter: ClassVar[int] = 0
    _watched_paths: ClassVar[dict[str, tuple[type, float]]] = {}
```

**`__new__`** processing (6 step groups, all recorded as `cls._metaclass_steps`):

1. **Generate unique ID** -- assigns `_asset_id` (int), `_asset_name` (str, `module.ClassName`), `_asset_type_code` (str, first 8 chars uppercased).
   - Emits: `TAG(asset_id)`, `TAG(asset_type_code)`

2. **Validate extensions** -- reads `_asset_extensions` (REQUIRED). Normalizes to lowercase with leading dot. Raises `TypeError` if missing.
   - Emits: `VALIDATE(asset_extensions_required)`, `TAG(extensions)`

3. **Check extension conflicts** -- ensures no extension is registered to multiple asset types. Raises `TypeError` on conflict.
   - Emits: `VALIDATE(extension_uniqueness)`

4. **Set defaults** -- `_asset_loader` (None), `_asset_cache_policy` (CachePolicy()), `_asset_dependencies` (()), `_asset_hot_reload` (False), `_asset_priority` (0).
   - Emits: `TAG(cache_policy)`, `TAG(hot_reload)`, `TAG(asset_priority)`

5. **Register extensions** -- maps each extension to this type in `_extension_map`.
   - Emits: `REGISTER(asset_extension_map)` per extension

6. **Register** -- stores type in `_registry[_asset_id]`.
   - Emits: `REGISTER(asset_registry)`

**Registry Access Methods:**

| Method | Signature | Returns |
|--------|-----------|---------|
| `get_by_id` | `(asset_id: int) -> Optional[type]` | Asset class by ID |
| `get_by_name` | `(name: str) -> Optional[type]` | Asset class by qualified name |
| `all_assets` | `() -> list[type]` | All registered asset classes |
| `get_for_extension` | `(extension: str) -> Optional[type]` | Asset type for file extension (e.g., ".png" -> TextureAsset) |
| `get_for_path` | `(path: str) -> Optional[type]` | Asset type for file path (extracts extension, looks up) |
| `get_loader` | `(asset_type: type) -> Optional[type]` | Custom loader class for asset type |
| `get_supported_extensions` | `() -> list[str]` | All registered file extensions |
| `get_hot_reloadable` | `() -> list[type]` | All asset types with `_asset_hot_reload=True` |
| `clear_registry` | `() -> None` | Clear all registrations (testing) |

**Async Loading Pipeline:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `queue_load` | `(asset_cls: type, path: str, priority: int = 0, callback: Optional[Callable] = None) -> None` | Queue asset for async load. Higher priority = earlier. Uses min-heap with negative priority for max-heap behavior. |
| `process_queue` | `(max_items: int = 10) -> int` | Process up to N queued loads. Returns count processed. Default batch size from `ASSET_QUEUE_PROCESS_BATCH`. |
| `get_queue_status` | `() -> dict[str, Any]` | Returns `{"pending": int, "total_queued": int}` |

**Hot-Reload Watcher:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `watch` | `(asset_cls: type, path: str) -> None` | Register path for file watching. Records initial mtime. |
| `unwatch` | `(path: str) -> None` | Stop watching a path. |
| `check_changes` | `() -> list[tuple[type, str]]` | Check all watched paths. Returns list of `(asset_cls, path)` for changed files. Auto-unwatches deleted files. |

**Dependency-Ordered Loading:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_load_order` | `(asset_cls: type) -> list[type]` | Topological sort of `_asset_dependencies`. Dependencies first. Raises `ValueError` on circular dependencies. |

### 3.2 ResourceMeta (for singleton resources)

```python
class ResourceMeta(EngineMeta):
    _registry: ClassVar[dict[int, type]] = {}
    _instances: ClassVar[dict[int, Any]] = {}
    _next_id: ClassVar[int] = 1
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _initialization_lock: ClassVar[threading.RLock] = threading.RLock()
```

**`__new__`** processing (recorded as `cls._metaclass_steps`):

1. **Generate unique ID** -- assigns `_resource_id` (int), `_resource_name` (str).
2. **Set defaults** -- `_resource_priority` (100, `DEFAULT_RESOURCE_PRIORITY`), `_resource_dependencies` (()), `_resource_lazy` (False).
   - Emits: `TAG(resource_id)`, `TAG(resource_name)`, `TAG(resource_priority)`, `TAG(resource_lazy)`
3. **Register** -- stores in `_registry`.
   - Emits: `REGISTER(resource_registry)`
4. **Singleton enforcement** -- `__call__` returns existing instance or creates new one.
   - Emits: `HOOK(on_create, singleton_enforce)`

**Key Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_by_id` | `(resource_id: int) -> Optional[type]` | Get resource class by ID |
| `get_by_name` | `(name: str) -> Optional[type]` | Get resource class by qualified name |
| `all_resources` | `() -> list[type]` | All registered resource classes |
| `get_instance` | `(resource_cls: type) -> Optional[Any]` | Get singleton instance |
| `has_instance` | `(resource_cls: type) -> bool` | Check if instantiated |
| `get_or_create` | `(resource_cls: type) -> Any` | Get or create with dependency validation |
| `initialize_all` | `() -> None` | Initialize all non-lazy resources in dependency/priority order. Topological sort on `_resource_dependencies`, then sort by `_resource_priority`. Raises `RuntimeError` on circular deps. |
| `shutdown_all` | `() -> None` | Shutdown in reverse order. Calls `shutdown()` on each instance if present. |
| `reset_instance` | `(resource_cls: type) -> None` | Remove instance (calls `shutdown()` first), allows re-creation |
| `is_lazy` | `(resource_cls: type) -> bool` | Check if resource is lazy-init |
| `clear_registry` | `() -> None` | Clear all registrations and instances (testing) |

---

## 4. Descriptors

### TrackedDescriptor (tracking.py) -- Asset Load State & Ref Count Tracking
```python
TrackedDescriptor(
    field_type: type = object,
    inner: Optional[BaseDescriptor] = None,
    field_offset: int = 0,
    use_bitmask: bool = False,
)
```
- **descriptor_id:** `"tracked"`
- **accepts_inner:** `("storage", "validated", "range")`
- **accepts_outer:** `("networked", "observable", "cached")`
- **excludes:** `("computed",)`
- **post_set():** If value changed: adds field to `_dirty_fields` set (or sets bit in `_dirty_mask`), calls `_notify_foundation_tracker(obj, old, new)` -> `tracker.mark_dirty()`, calls `_notify_eventlog()` -> creates `Change` and calls `add_change_to_current_event()`.
- **Resource usage:** Track asset loading state transitions (Requested -> Loading -> Ready -> Unloading), reference count changes, residency state changes.

### ValidatedDescriptor (validation.py) -- Asset Parameter Bounds
```python
ValidatedDescriptor(validators: list)
```
- **descriptor_id:** `"validated"`
- **pre_set():** Runs all validators before write. Rejects or transforms invalid values.
- **Resource usage:** Validate LOD distances (> 0, ascending), budget limits (>= 0), pool sizes (> 0), compression settings.

### RangeDescriptor (validation.py) -- Clamped Resource Values
```python
RangeDescriptor(min_val, max_val, clamp=True)
```
- **descriptor_id:** `"range"`
- **pre_set():** Clamp or reject out-of-range values.
- **Resource usage:** Mip levels (0-N), priority values, memory budgets, streaming distances.

### ObservableDescriptor (observable.py) -- Asset State Subscriptions
```python
ObservableDescriptor()
```
- **descriptor_id:** `"observable"`
- **post_set():** Notifies all subscribers of value change.
- **Resource usage:** Asset loaded/unloaded notifications -> update dependent systems. Streaming state changes -> update UI. Hot-reload events -> notify consumers.

### VersionedDescriptor (tracking.py) -- Asset Version Tracking
```python
VersionedDescriptor(
    field_type: type = object,
    inner: Optional[BaseDescriptor] = None,
)
```
- **descriptor_id:** `"versioned"`
- **post_set():** Increments per-field version counter in `_version_{name}` attribute.
- **Resource usage:** Track asset revision for cache invalidation, hot-reload versioning, network asset sync.

### SerializableDescriptor (persistence.py) -- Asset Persistence
- **descriptor_id:** `"serializable"`
- **Resource usage:** Configure serialize/deserialize format for asset metadata, save/load asset references.

### LazyDescriptor (async_descriptors.py) -- Deferred Asset Init
- **descriptor_id:** `"lazy"`
- **Resource usage:** Defer heavy asset initialization until first access. Streaming assets start as lazy references.

### AsyncLoadDescriptor (async_descriptors.py) -- Async Asset Loading
- **descriptor_id:** `"async_load"`
- **Resource usage:** Return fallback value while asset loads asynchronously. Transition to real value when load completes.

### Descriptor Chain for Tracked Streamed Asset:
```
ObservableDescriptor        (outermost -- notify consumers of state changes)
  -> TrackedDescriptor      (mark dirty, notify Foundation Tracker)
    -> ValidatedDescriptor  (validate state transitions)
      -> StorageDescriptor  (innermost -- store in __dict__)
```

---

## 5. Foundation Integration

### Foundation Registry -- Asset Type Discovery
```python
# AssetMeta.__new__ calls:
registry.register(cls, name=cls._asset_name, track_instances=False)

# Runtime discovery:
registry.subclasses(Asset)                     # All asset types
registry.types_with_decorator("asset")         # All @asset-decorated types
AssetMeta.get_for_extension(".png")            # Extension -> TextureAsset
AssetMeta.get_for_path("hero/diffuse.png")     # Path -> TextureAsset
AssetMeta.get_supported_extensions()           # [".png", ".jpg", ".fbx", ...]
```

### Foundation Tracker -- Load State & Ref Count Tracking
```python
# Asset state change flow:
asset_handle.state = AssetState.LOADING
    -> TrackedDescriptor.post_set()
    -> tracker.mark_dirty(asset_handle, "state", REQUESTED, LOADING)

# Per-frame: collect all assets that changed state
dirty_assets = tracker.all_dirty()
for asset in dirty_assets:
    fields = tracker.dirty_fields(asset)
    if "state" in fields:
        process_state_transition(asset)
    if "ref_count" in fields:
        check_unload_eligibility(asset)
tracker.mark_clean(asset)

# Subscription: react to asset state changes
tracker.on_change(TextureAsset, callback=on_texture_state_change)
```

### Foundation EventLog -- Asset Lifecycle Audit
```python
@traced
def load_asset(self, path: str, priority: int = 0):
    # EventLog.record("AssetManager.load_asset", tick=N)
    asset_type = AssetMeta.get_for_path(path)
    handle = self.create_handle(asset_type, path)
    self.submit_io_request(handle, priority)
    # EventLog records: operation, entity, changes, causal chain

# Query: "What assets were loaded this frame?"
events = event_log.events_for_operation("AssetManager.load_asset")
# Query: "Why was this texture loaded?"
events = event_log.events_caused_by(texture_entity_id)
```

### Foundation Mirror -- Asset Introspection
```python
m = mirror(texture_asset)
m.get("state")              # Current loading state
m.get("ref_count")          # Active references
m.get("memory_size")        # GPU memory usage
m.to_dict()                 # Export all asset properties

m = mirror(TextureAsset)    # Class-level: field types and defaults
m.describe()                # Human-readable schema
schema_hash(TextureAsset)   # 16-char hex hash for versioning
```

### Foundation Bridge -- Cross-Layer Asset Sharing
```python
# ShellLang debugging:
# > QUERY TextureAsset WHERE state == "loaded" AND memory_size > 1048576
#   -> "Found 42 textures > 1MB loaded"
# > QUERY MeshAsset WHERE ref_count == 0
#   -> "Found 15 unreferenced meshes (candidates for unload)"
# > MUTATE asset=42 SET priority="critical"
#   -> Force high-priority loading
```

### Foundation ContentStore -- Content-Addressable Asset Storage
```python
store = ContentStore(backend=FileBackend("./asset_cache"))

# Store asset data by content hash (deduplication)
hash = store.put(texture_data)           # SHA-256 hash
same_hash = store.put(texture_data)      # Same content = same hash = no duplicate

# Tree storage for asset bundles
bundle_hash = store.put_tree({
    "mesh": mesh_data,
    "material": material_data,
    "textures": [diffuse_data, normal_data],
})
# Structural sharing: unchanged subtrees stored once

# Diff between asset versions
differ = ContentDiffer(store)
diffs = differ.diff(old_hash, new_hash)
# Returns: [Difference(~textures[0], changed), Difference(+textures[2], added)]
```

### Foundation DeltaSync -- Incremental Asset Updates
```python
sync = DeltaSync()

# Compute minimal patch between asset states
old_state = {"format": "bc7", "mips": 10, "size": 4096}
new_state = {"format": "bc7", "mips": 12, "size": 4096}
delta = sync.compute_delta(old_state, new_state)
# delta.changes = [("mips", 12)]  -- only what changed

# Apply patch to another copy
sync.apply_delta(target_state, delta)

# Use cases: hot-reload (send only changed fields), network asset sync, incremental builds
```

### Foundation Provenance -- Asset Build Lineage
```python
@track_provenance
def cook_texture(self, source_path: str) -> CookedTexture:
    raw = self.import_raw(source_path)
    record_input("source", source_path)
    record_input("format", "bc7")
    compressed = self.compress(raw, format="bc7")
    record_input("mip_count", compressed.mip_count)
    return compressed

# Query: "How was this cooked texture produced?"
prov = provenance(cooked_texture, "data")
# -> ComputedProvenance(computed_by="CookSystem.cook_texture", inputs={"source": "hero.png", "format": "bc7"})

# Derivation tree: full build lineage
tree = derivation_tree(cooked_texture, "data")
# -> source.png -> import -> compress(bc7) -> mip_gen -> cooked_texture
```

### Foundation Serializer -- Asset Metadata Persistence
```python
# Serialize asset metadata
metadata = to_dict(asset_handle, include_schema_hash=True)
# {"__type__": "TextureAsset", "__schema__": "a1b2c3d4...", "path": "...", "format": "bc7"}

# Deserialize with schema migration
handle = from_dict(metadata)
# If schema changed, auto-migrates via MigrationRegistry

# Binary format for cooked assets
data = to_bytes(cooked_asset)
asset = from_bytes(data)

# File I/O
to_file(asset_manifest, "assets.json")
manifest = from_file("assets.json")
```

---

## 6. Architecture Spec Details

### 6.1 Asset Pipeline Stages

```
SOURCE FILE                     COOKED ASSET                    RUNTIME OBJECT
-----------                     ------------                    --------------

Import                          Process                         Load
  Source decode                    Optimize                       Request
  Axis/scale convert               LOD generate                  I/O read
  Validation                        Mip generate                 Decompress
                                    Compress                      Deserialize
  -> Raw Asset Data                 Derive (bounds, SDF)         PostLoad
                                                                  -> Ready
                           Cook
                             Platform format
                             Strip debug
                             Validate

                                    Package
                                      Archive (PAK)
                                      Align sectors
                                      Manifest
```

**Import Stage:**
- Source format decode: FBX, glTF/GLB, USD, OBJ, Alembic (mesh); PNG, TGA, PSD, EXR, HDR, TIFF (texture); WAV, FLAC, OGG, MP3 (audio); JSON, XML, CSV, YAML (data)
- Axis conversion, scale normalization, vertex welding, triangulation
- Wire `@import_settings` decorator -> import configuration

**Process Stage:**
- Mesh: vertex cache optimization, overdraw ordering, LOD simplification, meshlet generation, tangent generation (MikkTSpace), bounds (AABB, sphere), convex decomposition, SDF generation
- Texture: format decode, color space (sRGB/linear), channel packing, resize, mipmap generation (box/Kaiser/alpha preservation), normal map processing, cubemap convolution
- Animation: resample, root motion extraction, curve reduction, quantization, ACL compression, sync marker extraction
- Audio: PCM decode, normalize, trim silence, sample rate conversion, codec compression (ADPCM 4:1, Vorbis variable, Opus low-latency)
- Shader: material graph -> HLSL/GLSL generation -> permutation expansion -> compilation (DXC -> DXIL/SPIR-V/Metal) -> debug strip -> optimization

**Cook Stage:**
- Platform-specific format conversion
- Wire `@cook` decorator -> compression selection, debug stripping
- Texture compression: BC1-BC7 (PC/Console), ASTC (Mobile), ETC2 (Mobile)
- Shader bytecode: DXIL (D3D12), SPIR-V (Vulkan), Metal SL, DXBC (D3D11)
- Unused asset stripping, deduplication, merging

**Package Stage:**
- Archive creation (PAK files): compressed, encrypted, file manifest
- Disk layout optimization: access-pattern ordering, sector alignment, patch-friendly
- Manifests: asset manifest, chunk manifest (streaming install), version manifest

### 6.2 Asset Handle System

**Handle Structure:**
```python
class AssetHandle(Generic[T]):
    _asset_id: int           # Unique handle ID
    _asset_type: type        # AssetMeta-registered type
    _path: str               # Virtual path
    _state: AssetState       # Current loading state
    _ref_count: int          # Active references
    _data: Optional[T]       # Loaded asset data (None until Ready)
    _priority: int           # Loading priority
    _dependencies: list      # Other handles this depends on
```

**Reference Counting:**
```
acquire(handle) -> ref_count += 1
release(handle) -> ref_count -= 1
    if ref_count == 0:
        if handle.keep_loaded: pass  # @streamable(keep_loaded=True)
        elif handle.unloadable:
            mark_for_unload(handle)  # @unloadable(min_age=60)
        else:
            unload_immediate(handle)
```

### 6.3 Loading Stages

```
Request -> Queued -> I/O -> Decompress -> Deserialize -> PostLoad -> Ready
   |                                                                   |
   v                                                                   v
 Failed                                                            Unloading
                                                                      |
                                                                      v
                                                                   Unloaded
```

| Stage | Thread | Description |
|-------|--------|-------------|
| Request | Main | `AssetMeta.queue_load()`. Priority-sorted min-heap. |
| Queued | Main | `AssetMeta.process_queue()` dequeues. |
| I/O | I/O Thread | Async file read from disk/network. |
| Decompress | Worker Thread | LZ4/ZSTD decompression. |
| Deserialize | Worker Thread | Format-specific deserialization. |
| PostLoad | Main/Worker | GPU upload, dependency resolution, validation. |
| Ready | Main | Asset available for use. Callbacks invoked. |
| Unloading | Main | Ref count = 0 + min_age elapsed. State saved if `@unloadable(save_state=True)`. |
| Unloaded | Main | Memory freed. Handle remains valid (can re-request). |
| Failed | Main | Error state. Logged via EventLog. |

**Async Loading:**
```
Main Thread          I/O Thread          Worker Threads
-----------          ----------          --------------
queue_load()  --->
process_queue() -->  read_file()  --->
                                        decompress()
                                        deserialize()
                     <--- complete ---
on_loaded()   <---
```

### 6.4 Streaming Systems

**Level Streaming (World Chunks):**
```
World partitioned into grid cells, each a @chunk(size=(X,Y,Z))
Player position -> compute visible cells -> priority sort
    -> stream in (high priority near player)
    -> stream out (low priority, far from player, @unloadable(min_age=60))
```

Cell states: `Unloaded -> Loading -> Loaded -> Activated -> Deactivating -> Unloading -> Unloaded`

**Texture Streaming:**
- Mip streaming: progressive quality, lowest mip always resident (`@residency(min_mip=2)`)
- Feedback analysis: GPU writes requested mip levels to feedback buffer
- Memory budget: `@budget(category="texture", max_bytes=512*1024*1024)`
- Eviction: LRU when pool full

**Mesh Streaming:**
- LOD streaming: distance-based quality via `@lod(levels=4, distances=[10, 50, 200, 1000])`
- Cluster streaming: Nanite-style visibility-driven cluster requests
- Instance streaming: foliage, props via `@streamable(priority="low")`

**Audio Streaming:**
- Prefetch buffer (ahead of playback)
- Double buffering (seamless)
- Decode streaming (on-the-fly decompression)

### 6.5 Virtualization

**Virtual Texturing:**
```
VIRTUAL ADDRESS SPACE                PHYSICAL MEMORY
  Potentially terabytes              Physical Atlas (limited GPU)
  of texture data                    +---------+
                     Page Table      | P P P P |
                     Indirection  -> | P P P P |
                     Lookup          +---------+

  FEEDBACK (GPU) -> REQUEST (CPU) -> LOAD (I/O) -> UPLOAD (GPU) -> COMMIT (page table)
  Eviction: LRU when physical pool full
```

- Page tables map virtual pages to physical atlas locations
- Feedback buffer: GPU records which virtual pages are needed (mip + UV)
- CPU reads feedback, prioritizes page loads, streams from disk
- Physical page pool: fixed-size GPU memory, LRU eviction
- Types: Streaming VT (disk-backed), Runtime VT (GPU-generated), UDIM VT (multi-tile UV)

**Virtual Geometry (Nanite-style):**
```
ORIGINAL MESH (1M triangles)
    |
    v
MESHLET DECOMPOSITION (~64 verts, ~124 tris per meshlet)
    |
    v
CLUSTER HIERARCHY (DAG)
    Root (coarsest) -> Groups -> Clusters (finest detail)
    |
    v
RUNTIME: DAG cut based on screen-space error
    - Visibility-driven cluster requests
    - GPU residency management
    - LRU cluster eviction
    - Seamless transitions (no popping)
```

**Virtual Shadow Maps:**
```
Shadow Page Pool (cached tiles)
    Clipmaps (sun/directional shadows)
    Cube pages (point light shadows)
    Per-page culling (only render changed pages)
    Invalidation (dynamic object movement)
```

### 6.6 Hot-Reload System

```
1. AssetMeta.watch(ShaderAsset, "shaders/pbr.hlsl")
    -> Records path + mtime in _watched_paths

2. Per-frame or on-demand: AssetMeta.check_changes()
    -> Compares current mtime vs recorded mtime
    -> Returns [(ShaderAsset, "shaders/pbr.hlsl")] for changed files

3. Reload handler:
    -> Re-import source file
    -> Re-process (recompile shader variants)
    -> Invalidate PSO cache
    -> Notify consumers via ObservableDescriptor
    -> Consumers rebind (materials keep instances, only rebind PSOs)

4. @reloadable decorator -> marks asset as supporting hot-reload
    -> Sets _asset_hot_reload = True
    -> AssetMeta.get_hot_reloadable() returns these types
```

### 6.7 Build System

**Dependency Graph:**
```
Source Assets              Derived Assets             Final Assets
-------------              --------------             ------------
Hero.fbx  ---------> Hero.mesh  ------+
                                       |
Hero_Diffuse.png -+-> Hero.material --+-> Hero.pak
Hero_Normal.png  -+                    |
                                       |
Hero_Skeleton.fbx -> Hero.skeleton ---+
Hero_Walk.fbx -----> Hero_Walk.anim --+
```

**Change Tracking:**
| Method | Description |
|--------|-------------|
| Timestamps | File modification time |
| Content Hash | SHA-256 of data (via ContentStore) |
| Settings Hash | Import configuration hash |
| Tool Version | Processor version changes |

**Incremental Builds:**
- Dirty detection (content hash changed)
- Cascade detection (downstream dependency graph traversal)
- Minimal rebuild (only necessary targets)
- Build cache: local (developer), shared (team CI), DDC (derived data), content-addressable (Foundation ContentStore)

**Distributed Builds:**
- Build coordinator distributes tasks to worker machines
- Asset transfer: input/output shipping with bandwidth compression
- Shared cache across workers
- Wire Foundation Provenance -> track which worker produced what

---

## 7. Decorator Stacks

### @streaming_chunk -- World Streaming
```python
from trinity.decorators.builtin_stacks.streaming import streaming_chunk

@streaming_chunk(chunk_size=(100, 100, 100), overlap=10, min_age=60.0)
class TerrainChunk(Component): ...
```
**Expands to:**
```python
@chunk(size=(100, 100, 100), overlap=10)
@streamable(priority="normal")
@loading_priority(visibility_weight=3.0, player_velocity_weight=1.5)
@unloadable(min_age=60.0, save_state=True)
@serializable(format="binary")
@track_changes
@async_load(priority=0, fallback=None)
@lazy(init_on="first_access")
```
Steps: TAG(chunk) + TAG(streamable) + TAG(loading_priority) + TAG(unloadable) + TAG(serializable) + TRACK + INTERCEPT(async_load) + INTERCEPT(lazy)

### @lod_scalable -- LOD + Streaming
```python
from trinity.decorators.builtin_stacks.streaming import lod_scalable

@lod_scalable(levels=4, distances=[10, 50, 200, 1000])
class ScalableMesh(Component): ...
```
**Expands to:**
```python
@lod(levels=4, distances=[10, 50, 200, 1000])
@streamable(priority="normal")
@residency(priority="normal", min_mip=2)
```
Steps: TAG(lod) + TAG(streamable) + TAG(residency) + REGISTER(lod_streaming) + REGISTER(assets) + VALIDATE(valid_priority)

### @open_world_entity -- Full Open World
```python
from trinity.decorators.builtin_stacks.composite import open_world_entity

@open_world_entity(pool_size=10000, chunk_size=(100, 100, 100))
class WorldObject(Component): ...
```
**Expands to:** `production_component` + `streaming_chunk` + `lod_scalable` + `versioned_saveable`
```python
@pooled(initial_size=10000)
@packed(layout="soa")
@track_changes
@serializable(format="binary")
@chunk(size=(100, 100, 100), overlap=10)
@streamable(priority="normal")
@loading_priority(visibility_weight=3.0, player_velocity_weight=1.5)
@unloadable(min_age=60.0, save_state=True)
@lod(levels=4, distances=[10, 50, 200, 1000])
@residency(priority="normal", min_mip=2)
@versioned(version=1)
```

### @production_component -- Standard Managed Component
```python
from trinity.decorators.builtin_stacks.composite import production_component

@production_component(pool_size=4096)
class ManagedAsset(Component): ...
```
**Expands to:**
```python
@pooled(initial_size=4096)
@packed(layout="soa")
@track_changes
@serializable(format="binary")
```
Steps: TAG(pool) + HOOK(on_create) + HOOK(on_destroy) + REGISTER(PoolManager) + TAG(memory) + TRACK + TAG(serializable) + REGISTER(data_flow) + DESCRIBE

### Proposed: @budgeted_asset_pool -- Memory-Budgeted Asset Pool
```python
# Proposed:
@pooled(initial_size=1024)
@budget(category="gpu_texture", max_bytes=512*1024*1024, warn_at=0.8)
@aligned(bytes=256)        # GPU buffer alignment
@serializable(format="binary")
@track_changes
class TexturePool(Component): ...
```
**Combines:** @pooled + @budget + @aligned + @serializable + @track_changes

---

## 8. TODO Checklist (from GAME_ENGINE_INTEGRATION_TODO.md Section 4)

### 4.1 Asset Pipeline
- [ ] Implement Asset handle system (typed references with ref-counting)
- [ ] Implement async asset loading with priority queues
- [ ] Implement asset hot-reload (file watcher -> reload -> notify)
- [ ] Wire `AssetMeta`-registered types to asset loader dispatch
- [ ] Wire `@serializable` descriptor chain -> asset serialization format
- [ ] Wire Foundation ContentStore -> content-addressable asset storage
- [ ] Integrate Foundation EventLog -- log asset load/unload events

### 4.2 Asset Types
- [ ] Implement Texture asset (formats, mip generation, compression)
- [ ] Implement Mesh asset (vertex formats, index buffers, LOD levels)
- [ ] Implement Material asset (shader parameters, texture references)
- [ ] Implement Shader asset (compilation, reflection, variants)
- [ ] Implement Animation asset (clips, curves, events)
- [ ] Implement Audio asset (streaming, compression formats)
- [ ] Implement Prefab asset (entity templates)
- [ ] Register all asset types via AssetMeta -> Foundation Registry

### 4.3 Virtualization
- [ ] Implement virtual texturing (tile-based streaming, feedback buffer)
- [ ] Implement virtual geometry (Nanite-style, cluster-based LOD)
- [ ] Wire `@lod` decorator -> LOD level definitions
- [ ] Wire `@streamable` decorator -> streaming priority/distance rules

---

## 9. Directory Structure

```
engine/resource/
|-- __init__.py                        # Public API exports
|-- RESOURCE_CONTEXT.md                # THIS FILE
|-- asset/
|   |-- __init__.py
|   |-- asset_handle.py                # AssetHandle<T> typed reference, ref counting, state machine
|   |-- asset_manager.py               # Central asset manager: load, unload, query, budgets
|   |-- asset_loader.py                # Format-specific loader dispatch, async pipeline
|   |-- asset_registry.py              # Runtime asset catalog: scan, index, resolve paths
|   |-- hot_reload.py                  # File watcher integration, reload coordinator
|   +-- dependency_graph.py            # Asset dependency tracking, topological load order
|-- types/
|   |-- __init__.py
|   |-- texture_asset.py               # TextureAsset: formats, mips, compression, VT pages
|   |-- mesh_asset.py                  # MeshAsset: vertices, indices, LODs, meshlets, bounds
|   |-- material_asset.py              # MaterialAsset: shader params, texture refs, variants
|   |-- shader_asset.py                # ShaderAsset: compilation, reflection, PSO cache
|   |-- animation_asset.py             # AnimationAsset: clips, curves, events, skeleton ref
|   |-- audio_asset.py                 # AudioAsset: streaming, compression codecs, cues
|   |-- prefab_asset.py                # PrefabAsset: entity templates, component snapshots
|   |-- data_table_asset.py            # DataTableAsset: structured game data (CSV/JSON)
|   +-- physics_asset.py               # PhysicsAsset: collision meshes, materials, ragdoll
|-- streaming/
|   |-- __init__.py
|   |-- stream_manager.py              # Central streaming coordinator, budget enforcement
|   |-- texture_streaming.py           # Mip streaming, feedback analysis, residency
|   |-- mesh_streaming.py              # LOD streaming, cluster streaming
|   |-- audio_streaming.py             # Prefetch, double buffer, decode streaming
|   |-- world_streaming.py             # Level/chunk streaming, cell state machine
|   +-- priority_system.py             # Smart priority: visibility, velocity, distance weights
|-- virtualization/
|   |-- __init__.py
|   |-- virtual_texturing.py           # Page tables, feedback buffer, physical page pool
|   |-- virtual_geometry.py            # Nanite-style cluster DAG, screen-error LOD selection
|   +-- virtual_shadow_maps.py         # Shadow page pool, clipmaps, per-page culling
|-- build/
|   |-- __init__.py
|   |-- import_pipeline.py             # Source format importers (FBX, glTF, PNG, WAV, ...)
|   |-- process_pipeline.py            # Optimization passes (LOD gen, mip gen, compression)
|   |-- cook_pipeline.py               # Platform-specific cooking (format conversion)
|   |-- package_pipeline.py            # Archive creation, manifests, alignment
|   |-- dependency_tracker.py          # Build dependency graph, change detection
|   +-- distributed_build.py           # Coordinator, workers, shared cache
+-- memory/
    |-- __init__.py
    |-- asset_pool.py                  # Per-type asset memory pools (@pooled integration)
    |-- budget_manager.py              # Per-category memory budgets (@budget integration)
    |-- eviction.py                    # LRU, priority-weighted, size-based eviction policies
    +-- residency_manager.py           # GPU residency tracking (@residency integration)
```

---

## 10. Canonical Usage Examples

### Example 1: Defining an Asset Type
```python
from trinity.decorators.assets import asset, cook, residency
from trinity.decorators.data_flow import serializable
from trinity.decorators.lod_streaming import streamable

@asset(extensions=(".png", ".jpg", ".tga", ".exr", ".hdr"))
@cook(platform=None, compression="lz4", strip_debug=True)
@residency(priority="normal", min_mip=2)
@serializable(format="binary", version=1)
class TextureAsset:
    """Texture asset with mip streaming and GPU residency."""
    _asset_extensions = (".png", ".jpg", ".tga", ".exr", ".hdr")
    _asset_hot_reload = True
    _asset_cache_policy = CachePolicy(max_memory_bytes=256*1024*1024, preload=False)

    path: str = ""
    width: int = 0
    height: int = 0
    format: str = "rgba8"
    mip_count: int = 1
    srgb: bool = True
    compression: str = "bc7"
```

### Example 2: Loading Assets via AssetMeta
```python
from trinity.metaclasses.asset_meta import AssetMeta

# Discover asset type from file path
asset_cls = AssetMeta.get_for_path("characters/hero/diffuse.png")
# -> TextureAsset

# Queue async load with priority
def on_loaded(cls, path):
    print(f"Loaded {cls.__name__} from {path}")

AssetMeta.queue_load(TextureAsset, "characters/hero/diffuse.png", priority=5, callback=on_loaded)

# Process queue (call per-frame or on I/O thread)
processed = AssetMeta.process_queue(max_items=10)

# Check queue status
status = AssetMeta.get_queue_status()
# -> {"pending": 3, "total_queued": 15}

# Dependency-ordered loading
load_order = AssetMeta.get_load_order(MaterialAsset)
# -> [TextureAsset, ShaderAsset, MaterialAsset]
```

### Example 3: Streaming Setup with Decorator Stacks
```python
from trinity.decorators.builtin_stacks.streaming import streaming_chunk, lod_scalable

@streaming_chunk(chunk_size=(256, 256, 256), overlap=16, min_age=120.0)
class TerrainChunk(Component):
    """Streamable terrain chunk with auto-unload after 2 minutes."""
    heightmap_path: str = ""
    splat_path: str = ""
    foliage_density: float = 1.0

@lod_scalable(levels=5, distances=[5, 25, 100, 500, 2000])
class PropMesh(Component):
    """Scalable prop with 5 LOD levels and residency control."""
    mesh_path: str = ""
    cast_shadow: bool = True
```

### Example 4: Hot-Reload Handler
```python
from trinity.metaclasses.asset_meta import AssetMeta

# Register paths for watching
AssetMeta.watch(ShaderAsset, "shaders/pbr.hlsl")
AssetMeta.watch(ShaderAsset, "shaders/common.hlsli")
AssetMeta.watch(TextureAsset, "textures/hero_diffuse.png")

# Per-frame check (development builds only)
changes = AssetMeta.check_changes()
for asset_cls, path in changes:
    if asset_cls == ShaderAsset:
        # Re-compile shader, invalidate PSO cache
        shader_system.reload(path)
    elif asset_cls == TextureAsset:
        # Re-import texture, re-upload to GPU
        texture_manager.reimport(path)

# Stop watching
AssetMeta.unwatch("shaders/pbr.hlsl")
```

### Example 5: Virtual Texture Setup
```python
from typing import Annotated
from trinity.decorators.assets import asset, residency
from trinity.decorators.memory import budget, aligned
from trinity.descriptors.tracking import TrackedDescriptor
from trinity.descriptors.validation import RangeDescriptor

@asset(extensions=(".vtex",))
@residency(priority="normal", min_mip=0)
@budget(category="virtual_texture", max_bytes=1024*1024*1024, warn_at=0.85)
@aligned(bytes=4096)  # Page-aligned for virtual memory
class VirtualTextureAsset:
    """Virtual texture with page-based streaming."""
    page_size: int = 128                  # Pixels per page edge
    physical_pool_pages: int = 4096       # Max physical pages in GPU
    feedback_buffer_scale: float = 0.25   # Feedback at 1/4 resolution

    resident_pages: Annotated[int,
        TrackedDescriptor(),
        RangeDescriptor(min_val=0, max_val=65536),
    ] = 0

    eviction_policy: str = "lru"          # "lru" | "priority" | "hybrid"
```

### Example 6: Memory-Budgeted Asset Pool
```python
from trinity.decorators.memory import pooled, budget, aligned
from trinity.decorators.data_flow import serializable

@pooled(initial_size=2048, grow_factor=1.5, max_size=8192)
@budget(category="gpu_mesh", max_bytes=256*1024*1024, warn_at=0.75)
@aligned(bytes=64)
@serializable(format="binary")
class MeshPool:
    """Pre-allocated pool for mesh asset GPU data."""
    vertex_count: int = 0
    index_count: int = 0
    lod_count: int = 1
    has_meshlets: bool = False
    bounds_min: tuple = (0.0, 0.0, 0.0)
    bounds_max: tuple = (0.0, 0.0, 0.0)
```

### Example 7: Build Pipeline Configuration
```python
from trinity.decorators.assets import asset, cook, import_settings

@asset(extensions=(".fbx", ".gltf", ".glb"))
@import_settings(scale=0.01, axis_conversion=("X", "Z", "Y"), merge_meshes=False)
@cook(platform=None, compression="zstd", strip_debug=True)
class MeshAsset:
    """Mesh asset with full build pipeline configuration."""
    _asset_extensions = (".fbx", ".gltf", ".glb")
    _asset_hot_reload = True
    _asset_dependencies = (TextureAsset,)  # Materials reference textures
    _asset_priority = 10                    # Higher = load earlier

    path: str = ""
    vertex_format: str = "pntbu"  # position, normal, tangent, binormal, uv
    lod_count: int = 4
    meshlet_count: int = 0
    triangle_count: int = 0
```

---

## 11. Integration Patterns

### Pattern 1: Resource -> Rendering (Textures, Meshes, Shaders)
```python
# Resource layer loads and streams asset data.
# Rendering layer consumes it.
#
# Flow:
# 1. Rendering requests texture via AssetMeta.queue_load(TextureAsset, path, priority)
# 2. Resource layer async loads: I/O -> Decompress -> Deserialize
# 3. PostLoad: GPU upload (resource layer manages GPU memory via @budget + @residency)
# 4. Ready: Rendering binds texture to material
# 5. Streaming: @lod + @streamable -> resource layer manages mip/LOD transitions
# 6. Hot-reload: resource layer detects change -> re-imports -> notifies rendering
#    -> rendering rebinds without losing material instances
```

### Pattern 2: Resource -> Networking (Asset Replication)
```python
# Multiplayer: server decides which assets clients need.
# Resource layer handles the loading; networking handles the transport.
#
# Flow:
# 1. Server determines relevant assets per client (interest management)
# 2. Client receives asset references (soft references / paths)
# 3. Client's resource layer queues loads via AssetMeta.queue_load()
# 4. DeltaSync computes minimal patches for asset metadata updates
# 5. ContentStore deduplicates common assets across clients
```

### Pattern 3: Resource -> Platform (I/O Abstraction)
```python
# Resource layer uses platform I/O for all file access.
# Platform provides: async file reads, path resolution, memory-mapped I/O.
#
# Flow:
# 1. Resource layer calls platform.async_read(path, callback)
# 2. Platform dispatches to appropriate backend (local disk, network, packed archive)
# 3. Platform returns raw bytes
# 4. Resource layer decompresses and deserializes
# 5. @cook(platform="windows") -> resource layer selects correct cooked format
```

### Pattern 4: Resource -> Gameplay (Prefabs, Data Assets)
```python
# Prefabs are templates: entity + component snapshot.
# Data tables are structured data (stats, recipes, dialog).
#
# Flow:
# 1. Gameplay requests prefab: AssetMeta.queue_load(PrefabAsset, "hero.prefab")
# 2. Resource layer loads, deserializes component data via Foundation Serializer
# 3. Gameplay instantiates: creates entity, attaches components from prefab
# 4. Data tables loaded as DataTableAsset -> runtime lookup via keys
# 5. Hot-reload: data table changes -> gameplay re-reads without restart
```

### Pattern 5: Asset Handle Lifecycle
```python
# Complete lifecycle of an asset from request to unload:
#
# 1. handle = AssetHandle(TextureAsset, "hero.png")     # State: Request
# 2. AssetMeta.queue_load(TextureAsset, "hero.png", 5)  # State: Queued
# 3. I/O thread reads file                               # State: I/O
# 4. Worker decompresses                                  # State: Decompress
# 5. Worker deserializes                                  # State: Deserialize
# 6. GPU upload + validation                              # State: PostLoad
# 7. handle.state = Ready                                 # State: Ready
#    -> TrackedDescriptor fires
#    -> tracker.mark_dirty(handle, "state")
#    -> EventLog records state transition
#    -> Consumers notified via ObservableDescriptor
# 8. Consumer calls handle.acquire()                      # ref_count++
# 9. Consumer done, calls handle.release()                # ref_count--
# 10. ref_count == 0, min_age elapsed                     # State: Unloading
# 11. State saved if @unloadable(save_state=True)         # State: Unloaded
# 12. Memory freed, handle remains valid for re-request
```

### Pattern 6: Foundation ContentStore for Build Cache
```python
# Build system uses ContentStore for incremental builds:
#
# 1. Compute content hash of source file: hash_src = store.put(source_bytes)
# 2. Check if cooked result exists: store.has(hash_cooked)?
# 3. If yes -> skip cook, use cached result
# 4. If no -> cook asset, store result: hash_cooked = store.put(cooked_bytes)
# 5. Record dependency: Provenance tracks source -> cooked lineage
# 6. Distributed build: workers share ContentStore backend (FileBackend on NFS)
# 7. Diff between builds: ContentDiffer.diff(old_manifest_hash, new_manifest_hash)
```

---

## 12. Quick Reference Tables

### All Resource-Relevant Decorators
| Decorator | Module | Tier | Key Params | Purpose |
|-----------|--------|------|------------|---------|
| @asset | assets | 8 | extensions, loader | Define asset type with file extensions |
| @cook | assets | 8 | platform, compression, strip_debug | Platform-specific cooking config |
| @preload | assets | 8 | priority | Mark asset for pre-loading |
| @residency | assets | 8 | priority, min_mip | GPU memory residency policy |
| @import_settings | assets | 8 | scale, axis_conversion, merge_meshes | Source import configuration |
| @lod | lod_streaming | 31 | levels, distances, bias | Level of detail management |
| @streamable | lod_streaming | 31 | priority, keep_loaded | Asset streaming config |
| @chunk | lod_streaming | 31 | size, overlap | World chunk definition |
| @loading_priority | lod_streaming | 31 | visibility_weight, player_velocity_weight | Smart load ordering |
| @unloadable | lod_streaming | 31 | min_age, save_state | Unload policy |
| @serializable | data_flow | 4 | format, version | Serialization format |
| @versioned | data_flow | 4 | version, migrations | Schema versioning |
| @snapshot | data_flow | 4 | history_frames | State snapshot history |
| @pooled | memory | 2 | initial_size, grow_factor, max_size | Pre-allocated memory pool |
| @packed | memory | 2 | layout | Memory layout (AoS/SoA/hybrid) |
| @aligned | memory | 2 | bytes | Memory alignment |
| @arena | memory | 2 | name | Arena allocator scope |
| @budget | memory | 2 | category, max_bytes, warn_at | Memory budget tracking |
| @allocator | memory | 2 | type, size, thread_safe | Custom allocator |
| @flyweight | memory | 2 | -- | Shared immutable data |
| @intern | memory | 2 | -- | String interning |
| @generations | memory | 2 | -- | Generational indices |
| @copy_on_write | memory | 2 | -- | Lazy copy on mutation |
| @inline_array | memory | 2 | size | Fixed-size inline array |
| @atomic | memory | 2 | -- | Thread-safe atomic ops |

### Loading Stages
| Stage | Thread | Input | Output | Foundation |
|-------|--------|-------|--------|------------|
| Request | Main | Path + priority | Queue entry | EventLog |
| Queued | Main | Queue pop | I/O request | -- |
| I/O | I/O Thread | File path | Raw bytes | -- |
| Decompress | Worker | Compressed bytes | Raw data | -- |
| Deserialize | Worker | Raw data | Object | Serializer |
| PostLoad | Main/Worker | Object | GPU resource | Tracker |
| Ready | Main | Complete asset | Available handle | EventLog, Tracker |
| Unloading | Main | Zero-ref handle | Saved state | ContentStore |

### Asset Types
| Asset Type | Extensions | LOD | Streamable | Hot-Reload | Compression |
|-----------|-----------|-----|------------|------------|-------------|
| Texture | .png .jpg .tga .exr .hdr | Mips | Yes (mip stream) | Yes | BC1-7/ASTC/ETC2 |
| Mesh | .fbx .gltf .glb .obj | Discrete/Nanite | Yes (LOD stream) | Yes | Meshopt/Quantized |
| Material | .mat | -- | No (small) | Yes | Binary |
| Shader | .hlsl .glsl | -- | No (small) | Yes | Platform bytecode |
| Animation | .fbx .anim | -- | No | Yes | ACL/Quantized |
| Audio | .wav .ogg .flac | -- | Yes (decode) | No | ADPCM/Vorbis/Opus |
| Prefab | .prefab | -- | No | Yes | Binary/JSON |
| Data Table | .csv .json | -- | No | Yes | None |
| Physics | .phys | -- | No | No | Binary |
| Navigation | .navmesh | -- | Yes (chunk) | No | Binary |

### Memory Management Strategies
| Strategy | Decorator | Use Case | Eviction |
|----------|-----------|----------|----------|
| Object Pool | @pooled | Frequent alloc/dealloc (handles, draw calls) | Return to pool |
| Memory Budget | @budget | Category-limited (GPU tex, mesh, audio) | LRU/Priority |
| Arena | @arena | Scoped lifetime (frame, level) | Bulk free |
| Aligned | @aligned | GPU buffers, SIMD data | N/A |
| Flyweight | @flyweight | Shared immutable (material params) | N/A |
| Copy-on-Write | @copy_on_write | Clone-heavy (prefab instances) | N/A |
| Generational | @generations | Entity ID safety | Invalidate |

### Pipeline Stages
| Stage | Input | Output | Tooling |
|-------|-------|--------|---------|
| Import | Source files (FBX, PNG, WAV) | Raw asset data | @import_settings |
| Process | Raw data | Optimized data (LODs, mips, compressed) | @lod, @residency |
| Cook | Optimized data | Platform-native format | @cook |
| Package | Cooked assets | PAK archives + manifests | Build system |

### Reference Type Selection Guide
| Scenario | Reference Type | Decorator |
|----------|---------------|-----------|
| Character mesh + textures always together | Hard | -- |
| Optional VFX asset | Soft | @streamable(priority="low") |
| Cached UI texture (may evict) | Weak | @residency(priority="evictable") |
| Open world terrain chunk | Async | @streaming_chunk stack |
| Always-loaded UI atlas | Hard + keep | @streamable(keep_loaded=True) |
| Background music | Async stream | @streamable(priority="normal") |

---

*End of RESOURCE_CONTEXT.md -- This file is the sole reference for implementing engine/resource/.*
