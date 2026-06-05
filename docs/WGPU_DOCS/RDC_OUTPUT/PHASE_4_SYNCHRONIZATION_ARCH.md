# PHASE 4: SYNCHRONIZATION - Architecture

**Scope:** Command encoding, synchronization, barrier resolution, frame pacing
**Duration:** 2-3 weeks
**Dependencies:** Phase 3 (PIPELINES)
**Produces:** Complete command submission and synchronization layer

---

## Overview

Phase 4 implements the command encoding layer, including command buffers, copy operations, query commands, and synchronization mechanisms. This phase handles CPU-GPU coordination and automatic barrier resolution.

### Covered Content (from MASTER.md Part VI)

- Chapter 9: Command Encoding
  - 9.1 Command encoder (creation, scope, lifetime, finalization)
  - 9.2 Copy commands (buffer-to-buffer, buffer-to-texture, texture-to-texture)
  - 9.3 Clear commands (buffer, texture via pass)
  - 9.4 Query commands (timestamp, occlusion, statistics, resolve)
  - 9.5 Debug commands (groups, markers, labels)

- Chapter 10: Synchronization
  - 10.1 Implicit synchronization (auto barriers, usage tracking, pass ordering)
  - 10.2 Explicit synchronization (workgroupBarrier, storageBarrier, textureBarrier)
  - 10.3 CPU-GPU synchronization (mapping, poll, fences, frame pacing)
  - 10.4 Resource state tracking (states, barriers, split barriers, resolver)

---

## Architectural Decisions

### ADR-013: Command Encoder Lifecycle

**Context:** Command encoders have strict lifetime rules.

**Decision:** Implement TrinityCommandEncoder with:
- RAII wrapper for automatic finish()
- Pass encoder tracking
- Validation in debug builds

**Rationale:** Prevents invalid encoder usage.

**Consequences:**
- Encoder consumed on finish()
- Pass must end before new pass
- Clear error messages for violations

---

### ADR-014: Frame Synchronization Strategy

**Context:** Multi-buffered rendering requires CPU-GPU coordination.

**Decision:** Implement FrameSynchronizer with:
- N-buffered frame pacing (default: 3)
- Per-frame fence tracking
- Adaptive frame latency

**Rationale:** Maximizes GPU utilization while controlling latency.

**Consequences:**
- Frame N writes to buffer (N mod 3)
- CPU waits on frame (N - 2) fence
- Configurable buffer count

---

### ADR-015: Automatic Barrier Resolution

**Context:** wgpu handles most barriers automatically, but TRINITY frame graph needs manual control.

**Decision:** Implement BarrierResolver for frame graph:
- Track resource states per pass
- Compute required barriers
- Batch barriers by stage

**Rationale:** Enables frame graph pass scheduling.

**Consequences:**
- State tracking overhead
- Barrier batching optimization
- Explicit for frame graph, implicit elsewhere

---

### ADR-016: Timestamp Query Strategy

**Context:** GPU timing requires timestamp queries with async readback.

**Decision:** Implement GPUProfiler with:
- Query set per frame
- Resolve to staging buffer
- Async readback with double-buffering

**Rationale:** Non-blocking profiling.

**Consequences:**
- Results delayed by 1-2 frames
- Optional feature (TIMESTAMP_QUERY)
- Fallback to estimation on unsupported

---

## Component Breakdown

### 1. Command Encoder

```
TrinityCommandEncoder
├── encoder: wgpu::CommandEncoder
├── device: Arc<wgpu::Device>
├── frame_number: u64
├── active_pass: Option<PassType>
└── debug_enabled: bool
```

**Encoder States:**
1. Created - can record commands
2. InRenderPass - render pass active
3. InComputePass - compute pass active
4. Finished - cannot record

**Methods:**
- `copy_buffer_to_buffer()`
- `copy_buffer_to_texture()`
- `copy_texture_to_buffer()`
- `copy_texture_to_texture()`
- `clear_buffer()`
- `resolve_query_set()`
- `push_debug_group()`
- `insert_debug_marker()`
- `pop_debug_group()`
- `begin_render_pass()` -> RenderPass
- `begin_compute_pass()` -> ComputePass
- `finish()` -> CommandBuffer

### 2. Copy Operations

```
CopyAlignmentCalculator
├── buffer_offset_alignment: u64      // 4 bytes
├── bytes_per_row_alignment: u32      // 256 bytes
├── copy_size_alignment: u64          // 4 bytes
└── calculate_aligned_bytes_per_row()
```

**Copy Commands:**
- Buffer-to-buffer: offset, size validation
- Buffer-to-texture: bytes_per_row, rows_per_image
- Texture-to-buffer: readback pattern
- Texture-to-texture: format compatibility

### 3. Query System

```
TimestampQueryPool
├── query_set: wgpu::QuerySet
├── resolve_buffer: wgpu::Buffer
├── readback_buffer: wgpu::Buffer
├── capacity: u32
├── next_index: u32
└── timestamp_period: f32
```

**Methods:**
- `write_timestamp()` - in pass
- `resolve()` - after passes
- `async read_results()` - on next frame

**OcclusionQuerySystem:**
- `begin_occlusion_query()`
- `end_occlusion_query()`
- `is_visible()` -> async result

**PipelineStatisticsTypes:**
- VERTEX_SHADER_INVOCATIONS
- CLIPPER_INVOCATIONS
- CLIPPER_PRIMITIVES_OUT
- FRAGMENT_SHADER_INVOCATIONS
- COMPUTE_SHADER_INVOCATIONS

### 4. Debug Commands

```
DebugScope
├── encoder: &mut CommandEncoder
└── Drop -> pop_debug_group()
```

**RAII pattern:**
```rust
{
    let _scope = DebugScope::new(&mut encoder, "Shadow Pass");
    // Commands here are in "Shadow Pass" group
} // Automatically pops on drop
```

**Debug Markers:**
- `push_debug_group(label)` - Begin region
- `pop_debug_group()` - End region
- `insert_debug_marker(label)` - Single point

### 5. Synchronization

```
TrinityFrameSynchronizer
├── frame_count: u64
├── buffer_count: usize            // 2 or 3
├── fences: Vec<FrameFence>
├── current_buffer: usize
└── target_frame_time: Duration
```

**FrameFence:**
- submission_index: Option<SubmissionIndex>
- `wait()` - Block until complete

**Frame Pacing Strategies:**
| Strategy | Buffers | Latency | Throughput |
|----------|---------|---------|------------|
| Double | 2 | Low | Medium |
| Triple | 3 | Medium | High |
| Uncapped | 1 | Lowest | Variable |

### 6. Resource State Tracking

```
BarrierResolver
├── resource_states: HashMap<ResourceId, ResourceState>
├── pending_barriers: Vec<Barrier>
└── access_history: Vec<AccessRecord>
```

**ResourceState:**
- stage: PipelineStage (VERTEX, FRAGMENT, COMPUTE, etc.)
- access: AccessFlags (READ, WRITE)
- layout: TextureLayout (8 states)

**TextureLayout:**
1. Undefined
2. General
3. ColorAttachment
4. DepthAttachment
5. ShaderReadOnly
6. CopySrc
7. CopyDst
8. Present

**Barrier Types:**
- RAW (Read-After-Write)
- WAR (Write-After-Read)
- WAW (Write-After-Write)
- Layout transition

**needs_barrier():**
```rust
fn needs_barrier(old: &ResourceState, new: &ResourceState) -> bool {
    // WAW: write followed by write
    if old.access.contains(WRITE) && new.access.contains(WRITE) { return true; }
    // RAW: read followed by write
    if old.access.contains(READ) && new.access.contains(WRITE) { return true; }
    // WAR: write followed by read (some cases)
    if old.access.contains(WRITE) && new.access.contains(READ) {
        // Depends on pipeline stages
    }
    // Layout transition always needs barrier
    if old.layout != new.layout { return true; }
    false
}
```

---

## Module Structure

```
crates/renderer-backend/src/
├── commands/
│   ├── mod.rs              # Module exports
│   ├── encoder.rs          # TrinityCommandEncoder
│   ├── copy.rs             # Copy operations, alignment
│   ├── clear.rs            # Clear operations
│   └── debug.rs            # DebugScope, markers
│
├── sync/
│   ├── mod.rs              # Module exports
│   ├── frame_sync.rs       # TrinityFrameSynchronizer
│   ├── barriers.rs         # BarrierResolver, ResourceState
│   ├── mapping.rs          # BufferReadback, mapping helpers
│   └── fences.rs           # FrameFence
│
├── queries/
│   ├── mod.rs              # Module exports
│   ├── timestamp.rs        # TimestampQueryPool, GPUProfiler
│   ├── occlusion.rs        # OcclusionQuerySystem
│   └── statistics.rs       # PipelineStatistics
```

---

## Testing Strategy

### Unit Tests

1. **Encoder lifecycle** - State transitions
2. **Copy alignment** - Bytes per row calculation
3. **Barrier detection** - RAW, WAR, WAW scenarios
4. **Layout transitions** - All 8 states
5. **Frame fence** - Wait behavior

### Integration Tests

1. **Buffer copy** - Round-trip verification
2. **Texture copy** - Format preservation
3. **Query readback** - Timestamp values
4. **Frame pacing** - Latency measurement
5. **Occlusion query** - Visibility result

### Blackbox Tests

1. **Profiler output** - Region timings
2. **Debug capture** - RenderDoc integration
3. **Barrier dump** - State transitions logged

---

## Performance Considerations

1. **Command Buffer Reuse** - Pool encoders where possible
2. **Barrier Batching** - Group by pipeline stage
3. **Query Readback** - Async, double-buffered
4. **Frame Pacing** - Match display refresh
5. **Copy Staging** - Persistent mapped buffers

---

## Dependencies

### External Crates

- `wgpu` - Core GPU abstraction
- `tokio` - Async mapping (optional)

### Internal Dependencies

- Phase 1: TrinityDevice, TrinityQueue
- Phase 2: TrinityBufferSystem
- Phase 3: TrinityRenderPass, TrinityComputePass

---

## Deliverables Checklist

- [ ] TrinityCommandEncoder with lifecycle management
- [ ] Copy operations with alignment validation
- [ ] Clear buffer operation
- [ ] TimestampQueryPool with async readback
- [ ] OcclusionQuerySystem
- [ ] PipelineStatistics query
- [ ] DebugScope RAII wrapper
- [ ] TrinityFrameSynchronizer (2 and 3 buffer)
- [ ] BarrierResolver with state tracking
- [ ] BufferReadback utility
- [ ] Unit tests (80%+ coverage)
- [ ] Integration tests
- [ ] Documentation

---

*End of PHASE_4_SYNCHRONIZATION_ARCH.md*
