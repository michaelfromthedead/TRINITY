# Engine Resource Types Investigation

## Summary

**Classification: REAL (Production-Ready)**

The `engine/resource/types/` module contains 10 well-designed asset type definitions with comprehensive metadata, proper inheritance hierarchies, and complete loading specifications. All classes inherit from `BaseAsset` and implement the required abstract interface (load/unload/is_loaded). The implementation uses modern Python idioms (slots, dataclasses, enums, type hints).

## File Analysis

### Base Asset (base_asset.py - 74 lines)

**Classification: REAL**

Abstract base class defining the contract for all loadable assets.

| Attribute | Type | Description |
|-----------|------|-------------|
| `_asset_id` | int | Unique identifier |
| `_name` | str | Human-readable name |
| `_path` | str | File system path |
| `_size_bytes` | int | On-disk size |
| `_version` | int | Asset version (default 1) |

**Abstract Methods:**
- `load(data: bytes) -> None` - Load asset from raw bytes
- `unload() -> None` - Release loaded data
- `is_loaded() -> bool` - Query load state

**Properties:**
- `memory_footprint` - Returns `_size_bytes` by default (subclasses override)

---

### Texture Asset (texture_asset.py - 130 lines)

**Classification: REAL**

GPU texture asset with mip-level support and block-compressed format awareness.

| Attribute | Type | Description |
|-----------|------|-------------|
| `_width` | int | Texture width in pixels |
| `_height` | int | Texture height in pixels |
| `_channels` | int | Number of channels |
| `_mip_levels` | int | Mip chain length |
| `_format` | TextureFormat | GPU pixel format |
| `_pixel_data` | bytes or None | Loaded pixel data |

**TextureFormat Enum:**
- `RGBA8`, `RGB8` - Uncompressed
- `BC1`, `BC3`, `BC5`, `BC7` - Block-compressed (DXT)
- `R16F`, `RGBA16F`, `RGBA32F` - Floating-point

**Bytes Per Pixel Mapping:**
```python
RGBA8: 4.0, RGB8: 3.0, BC1: 0.5, BC3: 1.0, BC5: 1.0, BC7: 1.0,
R16F: 2.0, RGBA16F: 8.0, RGBA32F: 16.0
```

**Key Methods:**
- `get_mip_size(level: int) -> tuple[int, int]` - Returns (width, height) at mip level
- `max_mip_levels` property - Calculates maximum possible mip count

**Validation:** Mip-level bounds checked in constructor.

---

### Mesh Asset (mesh_asset.py - 113 lines)

**Classification: REAL**

3D mesh asset with vertex formats, submeshes, and LOD support.

| Attribute | Type | Description |
|-----------|------|-------------|
| `_vertex_count` | int | Number of vertices |
| `_index_count` | int | Number of indices |
| `_vertex_format` | VertexFormat | Vertex attribute layout |
| `_submeshes` | list[SubMesh] | Material index ranges |
| `_lod_levels` | list[MeshAsset] | LOD chain |
| `_vertex_data` | bytes or None | Vertex buffer |
| `_index_data` | bytes or None | Index buffer |

**VertexFormat Enum:**
- `P3` - Position only (3 floats)
- `P3N3` - Position + normal (6 floats)
- `P3N3T2` - Position + normal + texcoord (8 floats)
- `P3N3T2T3` - Position + normal + texcoord + tangent (11 floats)

**SubMesh Dataclass:**
```python
@dataclass(frozen=True, slots=True)
class SubMesh:
    start_index: int
    index_count: int
    material_index: int
```

**External Dependencies:**
- `engine.resource.constants.BYTES_PER_FLOAT`
- `engine.resource.constants.BYTES_PER_INDEX`

**Key Methods:**
- `add_lod(lod: MeshAsset) -> None` - Add LOD level
- `set_index_data(data: bytes) -> None` - Separate index loading

**Memory Footprint Calculation:**
```python
vb = vertex_count * floats_per_vertex * BYTES_PER_FLOAT
ib = index_count * BYTES_PER_INDEX
```

---

### Material Asset (material_asset.py - 101 lines)

**Classification: REAL**

Material asset with blend modes, render queue, texture slots, and shader parameters.

| Attribute | Type | Description |
|-----------|------|-------------|
| `_shader_id` | int | Shader asset reference |
| `_textures` | dict[str, int] | Texture slot to asset ID |
| `_parameters` | dict[str, ParameterValue] | Shader parameters |
| `_blend_mode` | BlendMode | Transparency mode |
| `_render_queue` | int | Draw order priority |
| `_loaded` | bool | Load state flag |

**BlendMode Enum:**
- `OPAQUE` -> `RENDER_QUEUE_OPAQUE`
- `ALPHA_TEST` -> `RENDER_QUEUE_ALPHA_TEST`
- `ALPHA_BLEND` -> `RENDER_QUEUE_ALPHA_BLEND`
- `ADDITIVE` -> `RENDER_QUEUE_ADDITIVE`

**ParameterValue Type:** `Union[float, tuple]`

**External Dependencies:**
- `engine.resource.constants.RENDER_QUEUE_*`

**Key Methods:**
- `set_texture(slot: str, texture_asset_id: int) -> None`
- `set_parameter(key: str, value: ParameterValue) -> None`

---

### Shader Asset (shader_asset.py - 82 lines)

**Classification: REAL (with stub compile)**

GPU shader asset with pipeline stage support and uniform extraction.

| Attribute | Type | Description |
|-----------|------|-------------|
| `_stage` | ShaderStage | Pipeline stage |
| `_source_code` | str or None | GLSL-like source |
| `_compiled_binary` | bytes or None | Compiled SPIR-V/bytecode |

**ShaderStage Enum:**
- `VERTEX`, `FRAGMENT`, `COMPUTE`, `GEOMETRY`, `TESSELLATION`

**Key Methods:**
- `compile() -> bytes` - **STUB** - Returns UTF-8 encoded source (placeholder)
- `get_uniforms() -> list[str]` - Extracts uniform names via regex

**Uniform Extraction Pattern:**
```python
_UNIFORM_PATTERN = re.compile(r"uniform\s+\w+\s+(\w+)")
```

---

### Audio Asset (audio_asset.py - 88 lines)

**Classification: REAL**

Audio clip asset with sample metadata.

| Attribute | Type | Description |
|-----------|------|-------------|
| `_sample_rate` | int | Samples per second (Hz) |
| `_channels` | int | Channel count |
| `_bit_depth` | int | Bits per sample |
| `_duration_seconds` | float | Clip duration |
| `_format` | AudioFormat | Encoding format |
| `_audio_data` | bytes or None | Loaded PCM/compressed data |

**AudioFormat Enum:**
- `PCM16`, `PCM24`, `FLOAT32` - Uncompressed
- `VORBIS`, `OPUS` - Compressed

---

### Physics Asset (physics_asset.py - 85 lines)

**Classification: REAL**

Physics collider definition asset.

| Attribute | Type | Description |
|-----------|------|-------------|
| `_collider_type` | ColliderType | Shape type |
| `_dimensions` | tuple | Shape-specific size |
| `_mass` | float | Mass in kg |
| `_is_static` | bool | Static collider flag |
| `_friction` | float | Friction coefficient |
| `_restitution` | float | Bounce coefficient |
| `_loaded` | bool | Load state |

**ColliderType Enum:**
- `BOX`, `SPHERE`, `CAPSULE` - Primitive shapes
- `MESH` - Triangle mesh
- `CONVEX` - Convex hull

**External Dependencies:**
- `engine.resource.constants.DEFAULT_FRICTION`
- `engine.resource.constants.DEFAULT_RESTITUTION`

---

### Animation Asset (animation_asset.py - 79 lines)

**Classification: REAL**

Animation clip with channels and keyframes.

| Attribute | Type | Description |
|-----------|------|-------------|
| `_duration_seconds` | float | Clip duration |
| `_frame_count` | int | Total frames |
| `_channels` | list[AnimChannel] | Property channels |
| `_loaded` | bool | Load state |

**InterpolationMode Enum:**
- `STEP`, `LINEAR`, `CUBIC`

**Keyframe Dataclass:**
```python
@dataclass(frozen=True, slots=True)
class Keyframe:
    time: float
    value: tuple
    interpolation: InterpolationMode = InterpolationMode.LINEAR
```

**AnimChannel Dataclass:**
```python
@dataclass(slots=True)
class AnimChannel:
    target_path: str
    keyframes: list[Keyframe] = field(default_factory=list)
```

---

### Data Table Asset (data_table_asset.py - 72 lines)

**Classification: REAL**

Structured tabular data for game databases.

| Attribute | Type | Description |
|-----------|------|-------------|
| `_columns` | list[str] | Column names |
| `_rows` | list[dict] | Row data |
| `_row_type` | str | Schema identifier |
| `_loaded` | bool | Load state |

**Key Methods:**
- `get_row(index: int) -> dict` - Returns row copy (bounds-checked)
- `get_column_values(col: str) -> list` - Extract column values
- `find_rows(predicate: Callable[[dict], bool]) -> list[dict]` - Filter rows

**Use Cases:** Item databases, loot tables, stat tables, localization.

---

### Prefab Asset (prefab_asset.py - 60 lines)

**Classification: REAL**

Entity template with hierarchy support.

| Attribute | Type | Description |
|-----------|------|-------------|
| `_components` | list[dict] | Component configurations |
| `_children` | list[PrefabAsset] | Child prefabs |
| `_loaded` | bool | Load state |

**Key Methods:**
- `add_child(child: PrefabAsset) -> None` - Attach child prefab
- `instantiate() -> dict` - Create entity tree dictionary

**Instantiation Output:**
```python
{
    "asset_id": int,
    "name": str,
    "components": [dict, ...],
    "children": [<recursive instantiate()>, ...]
}
```

---

## Module Exports (__init__.py - 27 lines)

All types are properly exported:
```python
__all__ = [
    "BaseAsset",
    "TextureAsset", "TextureFormat",
    "MeshAsset", "VertexFormat", "SubMesh",
    "MaterialAsset", "BlendMode",
    "ShaderAsset", "ShaderStage",
    "AnimationAsset", "AnimChannel", "Keyframe", "InterpolationMode",
    "AudioAsset", "AudioFormat",
    "PrefabAsset",
    "DataTableAsset",
    "PhysicsAsset", "ColliderType",
]
```

---

## Architecture Patterns

### Consistent Design Patterns

1. **Slots for Memory Efficiency:** All classes use `__slots__` to reduce memory overhead
2. **Frozen Dataclasses:** Value types (SubMesh, Keyframe, AnimChannel) use `@dataclass(frozen=True, slots=True)`
3. **Enum for Type Safety:** All categorical values use `Enum` subclasses
4. **Property-Based Access:** All attributes exposed via read-only properties
5. **Defensive Copies:** Collections returned as copies (`list(self._items)`, `dict(self._map)`)
6. **Type Hints Throughout:** Full type annotations including `tuple`, `list`, `dict` generics

### Load/Unload State Management

| Asset Type | Load Mechanism | State Tracking |
|------------|----------------|----------------|
| TextureAsset | `_pixel_data = data` | `_pixel_data is not None` |
| MeshAsset | `_vertex_data = data` | `_vertex_data is not None` |
| AudioAsset | `_audio_data = data` | `_audio_data is not None` |
| ShaderAsset | `_compiled_binary = data` | `_compiled_binary is not None` |
| MaterialAsset | Flag-based | `_loaded` bool |
| PhysicsAsset | Flag-based | `_loaded` bool |
| AnimationAsset | Flag-based | `_loaded` bool |
| DataTableAsset | Flag-based | `_loaded` bool |
| PrefabAsset | Flag-based | `_loaded` bool |

### Memory Footprint Overrides

- **TextureAsset:** Returns `len(_pixel_data)` when loaded
- **MeshAsset:** Calculates `vertex_buffer_size + index_buffer_size`
- **AudioAsset:** Returns `len(_audio_data)` when loaded
- Others use `BaseAsset.memory_footprint` (returns `_size_bytes`)

---

## External Dependencies

| Dependency | Used By |
|------------|---------|
| `engine.resource.constants.BYTES_PER_FLOAT` | MeshAsset |
| `engine.resource.constants.BYTES_PER_INDEX` | MeshAsset |
| `engine.resource.constants.RENDER_QUEUE_*` | MaterialAsset |
| `engine.resource.constants.DEFAULT_FRICTION` | PhysicsAsset |
| `engine.resource.constants.DEFAULT_RESTITUTION` | PhysicsAsset |

---

## Classification Summary

| File | Lines | Classification | Notes |
|------|-------|----------------|-------|
| `__init__.py` | 27 | REAL | Clean module exports |
| `base_asset.py` | 74 | REAL | Well-designed abstract base |
| `texture_asset.py` | 130 | REAL | Complete with BC formats, mip validation |
| `mesh_asset.py` | 113 | REAL | Full LOD/submesh support |
| `material_asset.py` | 101 | REAL | Render queue integration |
| `shader_asset.py` | 82 | REAL (partial stub) | compile() is placeholder |
| `audio_asset.py` | 88 | REAL | Full sample metadata |
| `physics_asset.py` | 85 | REAL | Primitive + mesh colliders |
| `animation_asset.py` | 79 | REAL | Keyframe/channel structure |
| `data_table_asset.py` | 72 | REAL | Query helpers included |
| `prefab_asset.py` | 60 | REAL | Recursive instantiation |

**Total Lines:** 911 (reported 1,011 in task, actual 911 per code review)

**Overall: REAL (Production-Ready)** - Only `ShaderAsset.compile()` is a stub; all other functionality is complete and well-architected.
