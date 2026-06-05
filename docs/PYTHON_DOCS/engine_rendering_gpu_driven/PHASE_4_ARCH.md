# PHASE 4 ARCHITECTURE: Indirect Draw and Bindless Resources

## Overview

Phase 4 covers the indirect drawing system (`indirect_draw.py`, 661 lines), bindless resource management (`bindless.py`, 786 lines), and instance batching (`instancing.py`, 736 lines). Together these enable fully GPU-driven command generation and resource access.

## Components

### Indirect Draw (`indirect_draw.py`)

| Type | Purpose |
|------|---------|
| `DrawIndexedIndirectArgs` | Standard indirect draw arguments |
| `DrawCommand` | Draw with material/instance data |
| `IndirectDrawBuffer` | GPU buffer for indirect commands |
| `DrawCommandGenerator` | Generates draw commands from batches |
| `DrawCommandCompactor` | Merges contiguous draw ranges |
| `MultiDrawIndirectBuffer` | Multi-draw indirect wrapper |

### Bindless Resources (`bindless.py`)

| Type | Purpose |
|------|---------|
| `ResourceHandle` | Generational handle (index + generation) |
| `TextureDescriptor` | Texture bind info (view, sampler, format) |
| `BufferDescriptor` | Buffer bind info (offset, size, BDA) |
| `BindlessTextureManager` | Texture descriptor heap |
| `BindlessBufferManager` | Buffer descriptor heap (BDA) |
| `MaterialResourceTable` | PBR material bindings |
| `BindlessResourceSystem` | Unified resource system |

### Instance Batching (`instancing.py`)

| Type | Purpose |
|------|---------|
| `Mat4x4` | 4x4 transform matrix (TRS from quaternion) |
| `InstanceData` | Per-instance transform + material |
| `BatchKey` | mesh_id + material_id grouping key |
| `InstanceBatch` | Batch of instances sharing key |
| `InstanceBatcher` | Groups instances by batch key |
| `CulledInstanceBatcher` | Batching with culling integration |

## Rendering Pipeline

```
Input: Instance list + Visibility mask (from culling)
    |
    v
[InstanceBatcher] -- Group by BatchKey (mesh + material)
    |                Sort for coherent access
    v
[DrawCommandGenerator] -- Generate DrawIndexedIndirectArgs
    |                     Per-batch command
    v
[DrawCommandCompactor] -- Merge contiguous ranges
    |                     Reduce command count
    v
[MultiDrawIndirectBuffer] -- GPU buffer upload
    |
    v
[BindlessResourceSystem] -- Descriptor heap management
    |                       Material table binding
    v
Output: vkCmdDrawIndexedIndirect(buffer, count)
```

## Architecture Decisions

### ADR-INDIR-001: DrawIndexedIndirectArgs Layout

**Context**: Need indirect draw command format compatible with GPU APIs.

**Decision**: Use standard Vulkan/D3D12 indirect draw layout.

**Rationale**:
- Matches `VkDrawIndexedIndirectCommand` exactly
- Compatible with `wgpu::RenderPassEncoder::draw_indexed_indirect`
- 20 bytes per command (5 x u32)

**Implementation**:
```python
class DrawIndexedIndirectArgs:
    index_count: int      # Number of indices to draw
    instance_count: int   # Number of instances
    first_index: int      # Offset into index buffer
    vertex_offset: int    # Added to vertex index
    first_instance: int   # Instance ID offset
```

### ADR-INDIR-002: Draw Command Compaction

**Context**: Many small draws are inefficient; need to merge where possible.

**Decision**: Compact contiguous draw ranges into single commands.

**Rationale**:
- Reduces draw call count (GPU command overhead)
- Merges batches with same mesh+material and contiguous indices
- Sort by sort_key before compaction for optimal merging

**Implementation** (lines 549-636):
```python
def compact(self, commands: Sequence[DrawCommand]) -> list[DrawCommand]:
    sorted_commands = sorted(commands, key=lambda c: c.sort_key)
    compacted = []
    current = sorted_commands[0]
    for next_cmd in sorted_commands[1:]:
        if self._can_merge(current, next_cmd):
            current = self._merge(current, next_cmd)
        else:
            compacted.append(current)
            current = next_cmd
    compacted.append(current)
    return compacted
```

### ADR-BIND-001: Generational Resource Handles

**Context**: Need safe resource handles that detect use-after-free.

**Decision**: Use generational handles (index + generation counter).

**Rationale**:
- Index provides O(1) lookup
- Generation detects stale references
- Free list reuses indices while incrementing generation
- Validation: `handle.generation == heap[handle.index].generation`

**Implementation**:
```python
class ResourceHandle:
    index: int       # Heap slot index
    generation: int  # Generation counter
    
class DescriptorHeap:
    def allocate(self) -> ResourceHandle:
        index = self.free_list.pop()
        self.generations[index] += 1
        return ResourceHandle(index, self.generations[index])
    
    def is_valid(self, handle: ResourceHandle) -> bool:
        return self.generations[handle.index] == handle.generation
```

### ADR-BIND-002: Buffer Device Address (BDA)

**Context**: Bindless buffers need GPU-accessible pointers.

**Decision**: Store buffer device addresses (64-bit GPU pointers).

**Rationale**:
- Direct GPU pointer avoids descriptor indirection
- Required for Vulkan ray tracing (acceleration structures)
- Compatible with DX12 GPU virtual addresses
- Enables pointer arithmetic in shaders

**Implementation**:
```python
class BufferDescriptor:
    device_address: int  # VkDeviceAddress (64-bit)
    size: int
    offset: int
```

### ADR-BIND-003: PBR Material Resource Table

**Context**: Materials need multiple textures (albedo, normal, roughness, etc.).

**Decision**: MaterialResourceTable maps material ID to texture handle set.

**Rationale**:
- Single material ID indexes into resource table
- Table stores handles to all PBR textures
- Shader fetches handles, then samples textures
- Supports dynamic material updates

**Implementation**:
```python
class MaterialResources:
    albedo_handle: ResourceHandle
    normal_handle: ResourceHandle
    roughness_metallic_handle: ResourceHandle
    emissive_handle: ResourceHandle
    ao_handle: ResourceHandle

class MaterialResourceTable:
    materials: dict[int, MaterialResources]
```

### ADR-INST-001: Instance Batching by BatchKey

**Context**: Need to group instances for efficient draw command generation.

**Decision**: Batch by `BatchKey(mesh_id, material_id)`.

**Rationale**:
- Same mesh + same material = single draw call
- Multiple instances share indirect draw command
- Instance count in command enables GPU instancing
- Sort key enables optimal command ordering

**Implementation**:
```python
class BatchKey:
    mesh_id: int
    material_id: int
    
    def __hash__(self) -> int:
        return hash((self.mesh_id, self.material_id))

class InstanceBatcher:
    def batch(self, instances: list[InstanceData]) -> dict[BatchKey, InstanceBatch]:
        batches: dict[BatchKey, list] = {}
        for instance in instances:
            key = BatchKey(instance.mesh_id, instance.material_id)
            batches.setdefault(key, []).append(instance)
        return {k: InstanceBatch(k, v) for k, v in batches.items()}
```

## Data Flow

### Input Data

```python
class InstanceData:
    mesh_id: int
    material_id: int
    transform: Mat4x4
    bounding_sphere: (Vec3, float)
```

### Draw Commands

```python
class DrawCommand:
    args: DrawIndexedIndirectArgs
    material_id: int
    mesh_id: int
    sort_key: int  # For compaction ordering
```

### Resource Handles

```python
# Shader-visible material data
struct MaterialData {
    albedo_index: u32,      # Texture descriptor index
    normal_index: u32,
    roughness_metallic_index: u32,
    emissive_index: u32,
}
```

## GPU Port Considerations

### Buffer Layouts

```wgsl
struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
}

struct InstanceData {
    transform: mat4x4<f32>,
    material_id: u32,
    _pad: vec3<u32>,
}

struct MaterialData {
    albedo_index: u32,
    normal_index: u32,
    roughness_metallic_index: u32,
    emissive_index: u32,
}
```

### Bindless Texture Access

```wgsl
@group(0) @binding(0) var textures: binding_array<texture_2d<f32>>;
@group(0) @binding(1) var samplers: binding_array<sampler>;
@group(0) @binding(2) var<storage> materials: array<MaterialData>;

fn sample_material(material_id: u32, uv: vec2<f32>) -> vec4<f32> {
    let mat = materials[material_id];
    return textureSample(textures[mat.albedo_index], samplers[0], uv);
}
```

### Indirect Draw Execution

```wgsl
// CPU uploads indirect buffer
// Single vkCmdDrawIndexedIndirect call

encoder.draw_indexed_indirect(
    &indirect_buffer,
    0,  // offset
    command_count,  // number of commands
);
```

## Dependencies

- Requires culling output (visibility mask) from Phase 1
- Requires meshlet data from Phase 2
- Requires visibility buffer from Phase 3 (for material IDs)
- Provides final GPU draw commands
