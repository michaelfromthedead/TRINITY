# Engine Resource Virtualization Investigation

## Overview

Investigation of `engine/resource/virtualization/` - virtual resource systems for texturing, geometry, and shadow maps.

**Classification: REAL**

All three systems contain complete, functional implementations with proper data structures, algorithms, and state management. No placeholder/stub code detected.

---

## File Analysis

### 1. `virtual_texturing.py` (176 lines) - REAL

**Purpose:** Virtual texture system with page table management and LRU eviction.

**Key Classes:**

| Class | Description |
|-------|-------------|
| `Page` | Dataclass representing a virtual texture page mapped to physical tile (page_x, page_y, mip_level, physical coords, residency) |
| `PhysicalTexturePool` | Fixed-size pool of physical texture tiles with free-list allocation |
| `PageTable` | Maps virtual page coordinates to physical locations with LRU tracking |
| `VirtualTextureSystem` | High-level facade managing page table and physical pool |

**Implementation Details:**

- **Page Table Architecture:** Uses `OrderedDict` for LRU tracking with O(1) move-to-end operations
- **Physical Pool:** Free-list allocator that computes 2D tile coordinates from linear index
- **Eviction Strategy:** LRU eviction when pool is full (`_evict_lru` pops oldest from OrderedDict)
- **Request Flow:** `request_page()` -> adds to pending -> `update()` -> `_make_resident()` -> allocate or evict+allocate
- **Feedback System:** `get_feedback()` returns non-resident requested pages for streaming priority

**API:**
```python
VirtualTextureSystem(pool_tiles: int)
  .request_page(page_x, page_y, mip_level) -> Page
  .evict_page(page_x, page_y, mip_level) -> None
  .get_resident_pages() -> list[Page]
  .get_feedback() -> list[PageKey]  # Non-resident requests for streaming
  .update() -> None  # Process pending requests
```

---

### 2. `virtual_shadow_maps.py` (150 lines) - REAL

**Purpose:** Clipmap-based virtual shadow map system.

**Key Classes:**

| Class | Description |
|-------|-------------|
| `ShadowPage` | Single shadow map page in a clipmap level (page coords, level, cached/dirty flags) |
| `ShadowClipmapLevel` | One level of shadow clipmap hierarchy (level index, resolution, world_size) |
| `VirtualShadowMapSystem` | Clipmap-based virtual shadow map manager |

**Implementation Details:**

- **Clipmap Hierarchy:** `NUM_CLIPMAP_LEVELS` levels, each doubling world coverage (`world_size * 2^level`)
- **Page Coverage:** Dynamically creates pages around camera position per clipmap level
- **Dirty Tracking:** Light direction changes mark all pages dirty; region invalidation marks overlapping pages
- **Cache Stats:** Provides total/cached/dirty page counts and cache hit rate

**Clipmap Structure:**
```
Level 0: base_resolution, base_world_size (finest detail near camera)
Level 1: base_resolution, base_world_size * 2
Level 2: base_resolution, base_world_size * 4
...
Level N: base_resolution, base_world_size * 2^N (coarsest, covers distant areas)
```

**API:**
```python
VirtualShadowMapSystem()
  .update_light(light_dir, camera_pos) -> None  # Marks dirty, updates coverage
  .get_dirty_pages() -> list[ShadowPage]  # Pages needing re-render
  .mark_rendered(page_x, page_y, level) -> None  # Clear dirty flag
  .invalidate_region(x, y, radius) -> None  # Dynamic shadow casters
  .get_cache_stats() -> dict  # Performance monitoring
```

---

### 3. `virtual_geometry.py` (95 lines) - REAL

**Purpose:** Nanite-style cluster-based mesh rendering system.

**Key Classes:**

| Class | Description |
|-------|-------------|
| `Cluster` | Mesh cluster at specific LOD (cluster_id, lod_level, vertex/triangle counts, bounding sphere, visibility/residency) |
| `ClusterGroup` | Groups clusters sharing LOD hierarchy |
| `VirtualGeometrySystem` | Manages cluster-based virtual geometry with LOD selection and culling |

**Implementation Details:**

- **LOD Selection:** Distance-based using `LOD_DISTANCES` thresholds
- **Bounding Spheres:** Each cluster has (cx, cy, cz, radius) for culling
- **Culling Strategy:** Distance-based LOD matching + simple frustum approximation (max range check)
- **Residency Tracking:** Clusters marked resident when submitted

**LOD Selection Algorithm:**
```python
def _select_lod(dist: float) -> int:
    for level, threshold in enumerate(LOD_DISTANCES):
        if dist < threshold:
            return level
    return len(LOD_DISTANCES)  # Coarsest LOD beyond all thresholds
```

**API:**
```python
VirtualGeometrySystem()
  .submit_clusters(clusters: list[Cluster]) -> None  # Register clusters
  .cull(camera_pos, fov) -> list[Cluster]  # Returns visible clusters at correct LOD
  .get_visible_count() -> int
  .get_resident_count() -> int
```

---

### 4. `__init__.py` (43 lines) - REAL

**Purpose:** Module exports aggregating all virtualization subsystems.

**Exports:**
- Constants: `PAGE_SIZE`, `PHYSICAL_POOL_TILES`, `LOD_DISTANCES`, `NUM_CLIPMAP_LEVELS`, `SHADOW_PAGE_SIZE`
- Virtual Texturing: `Page`, `PageTable`, `PhysicalTexturePool`, `VirtualTextureSystem`
- Virtual Geometry: `Cluster`, `ClusterGroup`, `VirtualGeometrySystem`
- Virtual Shadow Maps: `ShadowClipmapLevel`, `ShadowPage`, `VirtualShadowMapSystem`

---

## Architecture Summary

```
engine/resource/virtualization/
├── __init__.py              # Module exports
├── virtual_texturing.py     # Page-based virtual textures with LRU eviction
├── virtual_geometry.py      # Nanite-style cluster LOD system
└── virtual_shadow_maps.py   # Clipmap-based shadow pages
```

**Common Patterns:**

1. **Page/Tile Abstraction:** All systems use page/cluster keys for sparse lookup
2. **Dirty/Cached Flags:** Lazy evaluation with dirty tracking
3. **Spatial Indexing:** Distance-based LOD, region-based invalidation
4. **Residency Management:** Explicit resident tracking for streaming

---

## Dependencies

| Module | External Import |
|--------|-----------------|
| All | `engine.resource.constants` (PAGE_SIZE, PHYSICAL_POOL_TILES, LOD_DISTANCES, NUM_CLIPMAP_LEVELS, SHADOW_PAGE_SIZE, SHADOW_BASE_WORLD_SIZE, SHADOW_RESOLUTION_MULTIPLIER) |

---

## Quality Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Implementation completeness | High | All core algorithms implemented |
| Type annotations | High | Full typing with slots-optimized dataclasses |
| Documentation | Medium | Module/class docstrings present; method docs sparse |
| Error handling | Low | No explicit error handling for invalid inputs |
| Test coverage | Unknown | No test files in virtualization directory |

---

## Recommendations

1. **Add input validation:** Bounds checking for page coordinates, mip levels
2. **Add unit tests:** Coverage for eviction, dirty tracking, LOD selection
3. **Consider GPU integration:** These are CPU-side managers; need wgpu integration for actual rendering
4. **Streaming system:** `get_feedback()` returns requests but no streaming implementation visible
