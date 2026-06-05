# PHASE 4: SYNCHRONIZATION - Task List

**Phase:** 4 - SYNCHRONIZATION
**Estimated Duration:** 2-3 weeks
**Task ID Prefix:** T-WGPU-P4

---

## Task Summary

| ID | Task | Est. Hours | Status |
|----|------|------------|--------|
| T-WGPU-P4.1.1 | Command encoder creation | 4 | - |
| T-WGPU-P4.1.2 | Encoder lifecycle management | 4 | - |
| T-WGPU-P4.1.3 | Pass encoder tracking | 3 | - |
| T-WGPU-P4.1.4 | Command buffer finalization | 2 | - |
| T-WGPU-P4.2.1 | Buffer-to-buffer copy | 3 | - |
| T-WGPU-P4.2.2 | Buffer-to-texture copy | 4 | - |
| T-WGPU-P4.2.3 | Texture-to-buffer copy | 4 | - |
| T-WGPU-P4.2.4 | Texture-to-texture copy | 3 | - |
| T-WGPU-P4.2.5 | Copy alignment calculator | 3 | - |
| T-WGPU-P4.3.1 | Buffer clear | 2 | - |
| T-WGPU-P4.3.2 | Texture clear via pass | 2 | - |
| T-WGPU-P4.4.1 | Timestamp query pool | 6 | - |
| T-WGPU-P4.4.2 | Query resolve | 3 | - |
| T-WGPU-P4.4.3 | Async query readback | 6 | - |
| T-WGPU-P4.4.4 | Occlusion queries | 4 | - |
| T-WGPU-P4.4.5 | Pipeline statistics | 4 | - |
| T-WGPU-P4.5.1 | Debug group RAII | 3 | - |
| T-WGPU-P4.5.2 | Debug markers | 2 | - |
| T-WGPU-P4.5.3 | Resource labels | 2 | - |
| T-WGPU-P4.6.1 | Frame fence | 4 | - |
| T-WGPU-P4.6.2 | Double buffering | 4 | - |
| T-WGPU-P4.6.3 | Triple buffering | 3 | - |
| T-WGPU-P4.6.4 | Frame pacing | 4 | - |
| T-WGPU-P4.7.1 | Resource state tracking | 6 | - |
| T-WGPU-P4.7.2 | Barrier detection | 4 | - |
| T-WGPU-P4.7.3 | Layout transitions | 4 | - |
| T-WGPU-P4.7.4 | Barrier batching | 4 | - |
| T-WGPU-P4.8.1 | Buffer mapping async | 4 | - |
| T-WGPU-P4.8.2 | Buffer readback utility | 4 | - |
| T-WGPU-P4.9.1 | Unit tests | 6 | - |
| T-WGPU-P4.9.2 | Integration tests | 6 | - |

**Total Estimated Hours:** 116 hours

---

## Detailed Tasks

### T-WGPU-P4.1.1 - Command Encoder Creation

**Description:** Implement TrinityCommandEncoder creation.

**Prerequisites:** Phase 3 complete

**Deliverable:** `create_command_encoder()` in commands/encoder.rs

**Acceptance Criteria:**
- [ ] CommandEncoderDescriptor with label
- [ ] TrinityCommandEncoder wrapper struct
- [ ] Frame number tracking
- [ ] Device reference

**Estimate:** 4 hours

---

### T-WGPU-P4.1.2 - Encoder Lifecycle Management

**Description:** Implement encoder state machine.

**Prerequisites:** T-WGPU-P4.1.1

**Deliverable:** State tracking in TrinityCommandEncoder

**Acceptance Criteria:**
- [ ] States: Created, InRenderPass, InComputePass, Finished
- [ ] Transition validation
- [ ] Error on invalid transition (debug builds)
- [ ] State query methods

**Estimate:** 4 hours

---

### T-WGPU-P4.1.3 - Pass Encoder Tracking

**Description:** Track active pass encoder.

**Prerequisites:** T-WGPU-P4.1.2

**Deliverable:** Pass tracking in encoder

**Acceptance Criteria:**
- [ ] Track active pass type
- [ ] Prevent new pass while pass active
- [ ] Automatic pass end on encoder finish
- [ ] Debug warning on implicit end

**Estimate:** 3 hours

---

### T-WGPU-P4.1.4 - Command Buffer Finalization

**Description:** Implement encoder finish and submission.

**Prerequisites:** T-WGPU-P4.1.3

**Deliverable:** `finish()` method

**Acceptance Criteria:**
- [ ] Consumes encoder
- [ ] Returns wgpu::CommandBuffer
- [ ] Validation before finish
- [ ] Submits to queue

**Estimate:** 2 hours

---

### T-WGPU-P4.2.1 - Buffer-to-Buffer Copy

**Description:** Implement buffer copy command.

**Prerequisites:** T-WGPU-P4.1.1

**Deliverable:** `copy_buffer_to_buffer()` wrapper

**Acceptance Criteria:**
- [ ] Source offset, dest offset, size
- [ ] Alignment validation (4 bytes)
- [ ] Size validation
- [ ] COPY_SRC and COPY_DST usage check

**Estimate:** 3 hours

---

### T-WGPU-P4.2.2 - Buffer-to-Texture Copy

**Description:** Implement buffer to texture copy.

**Prerequisites:** T-WGPU-P4.2.5

**Deliverable:** `copy_buffer_to_texture()` wrapper

**Acceptance Criteria:**
- [ ] ImageCopyBuffer struct
- [ ] ImageCopyTexture struct
- [ ] Extent3d specification
- [ ] bytes_per_row alignment (256)
- [ ] rows_per_image for 3D/array

**Estimate:** 4 hours

---

### T-WGPU-P4.2.3 - Texture-to-Buffer Copy

**Description:** Implement texture to buffer copy (readback).

**Prerequisites:** T-WGPU-P4.2.5

**Deliverable:** `copy_texture_to_buffer()` wrapper

**Acceptance Criteria:**
- [ ] Same parameters as buffer-to-texture
- [ ] MAP_READ usage on destination
- [ ] Staging buffer pattern
- [ ] Async readback support

**Estimate:** 4 hours

---

### T-WGPU-P4.2.4 - Texture-to-Texture Copy

**Description:** Implement texture to texture copy.

**Prerequisites:** T-WGPU-P4.2.1

**Deliverable:** `copy_texture_to_texture()` wrapper

**Acceptance Criteria:**
- [ ] Source and dest ImageCopyTexture
- [ ] Format compatibility check
- [ ] Mip level selection
- [ ] Array layer selection

**Estimate:** 3 hours

---

### T-WGPU-P4.2.5 - Copy Alignment Calculator

**Description:** Implement alignment calculation utility.

**Prerequisites:** T-WGPU-P4.2.1

**Deliverable:** CopyAlignmentCalculator struct

**Acceptance Criteria:**
- [ ] BUFFER_OFFSET_ALIGNMENT = 4
- [ ] BYTES_PER_ROW_ALIGNMENT = 256
- [ ] COPY_SIZE_ALIGNMENT = 4
- [ ] calculate_aligned_bytes_per_row()
- [ ] calculate_rows_per_image()

**Estimate:** 3 hours

---

### T-WGPU-P4.3.1 - Buffer Clear

**Description:** Implement buffer clear command.

**Prerequisites:** T-WGPU-P4.1.1

**Deliverable:** `clear_buffer()` wrapper

**Acceptance Criteria:**
- [ ] Clears to zero
- [ ] Offset and size parameters
- [ ] Alignment validation
- [ ] COPY_DST usage required

**Estimate:** 2 hours

---

### T-WGPU-P4.3.2 - Texture Clear via Pass

**Description:** Implement texture clear via render pass.

**Prerequisites:** Phase 3 render pass

**Deliverable:** Clear texture helper

**Acceptance Criteria:**
- [ ] LoadOp::Clear with value
- [ ] StoreOp::Store
- [ ] RENDER_ATTACHMENT usage
- [ ] Depth and color clear

**Estimate:** 2 hours

---

### T-WGPU-P4.4.1 - Timestamp Query Pool

**Description:** Implement timestamp query set and pool.

**Prerequisites:** T-WGPU-P4.1.1

**Deliverable:** TimestampQueryPool struct

**Acceptance Criteria:**
- [ ] QuerySetDescriptor with QueryType::Timestamp
- [ ] Feature check (TIMESTAMP_QUERY)
- [ ] Capacity management
- [ ] Index allocation
- [ ] Resolve buffer creation

**Estimate:** 6 hours

---

### T-WGPU-P4.4.2 - Query Resolve

**Description:** Implement query result resolution.

**Prerequisites:** T-WGPU-P4.4.1

**Deliverable:** `resolve()` method

**Acceptance Criteria:**
- [ ] resolve_query_set() command
- [ ] Range specification
- [ ] Destination buffer offset
- [ ] Timing: after passes, before submit

**Estimate:** 3 hours

---

### T-WGPU-P4.4.3 - Async Query Readback

**Description:** Implement async query result readback.

**Prerequisites:** T-WGPU-P4.4.2

**Deliverable:** `async read_results()` method

**Acceptance Criteria:**
- [ ] Map readback buffer async
- [ ] Parse timestamp values
- [ ] Calculate duration from period
- [ ] Return ProfileResult vec
- [ ] Double-buffered readback

**Estimate:** 6 hours

---

### T-WGPU-P4.4.4 - Occlusion Queries

**Description:** Implement occlusion query system.

**Prerequisites:** T-WGPU-P4.4.1

**Deliverable:** OcclusionQuerySystem struct

**Acceptance Criteria:**
- [ ] QueryType::Occlusion
- [ ] begin_occlusion_query() in render pass
- [ ] end_occlusion_query()
- [ ] is_visible() result query
- [ ] Binary vs sample count modes

**Estimate:** 4 hours

---

### T-WGPU-P4.4.5 - Pipeline Statistics

**Description:** Implement pipeline statistics queries.

**Prerequisites:** T-WGPU-P4.4.1

**Deliverable:** PipelineStatistics struct

**Acceptance Criteria:**
- [ ] Feature check (PIPELINE_STATISTICS_QUERY)
- [ ] QueryType::PipelineStatistics
- [ ] 5 statistic types
- [ ] overdraw_estimate()
- [ ] culling_efficiency()

**Estimate:** 4 hours

---

### T-WGPU-P4.5.1 - Debug Group RAII

**Description:** Implement DebugScope RAII wrapper.

**Prerequisites:** T-WGPU-P4.1.1

**Deliverable:** DebugScope struct

**Acceptance Criteria:**
- [ ] push_debug_group() on creation
- [ ] pop_debug_group() on Drop
- [ ] Works with encoder and passes
- [ ] Nested scopes supported

**Estimate:** 3 hours

---

### T-WGPU-P4.5.2 - Debug Markers

**Description:** Implement debug marker insertion.

**Prerequisites:** T-WGPU-P4.5.1

**Deliverable:** `insert_debug_marker()` wrapper

**Acceptance Criteria:**
- [ ] Single point marker
- [ ] Visible in RenderDoc/PIX
- [ ] Works in encoder and passes

**Estimate:** 2 hours

---

### T-WGPU-P4.5.3 - Resource Labels

**Description:** Document resource labeling conventions.

**Prerequisites:** None

**Deliverable:** Labeling guidelines and helpers

**Acceptance Criteria:**
- [ ] Naming conventions documented
- [ ] label_resource() helper
- [ ] Hierarchical naming (e.g., "Frame/Shadow/Cascade0")
- [ ] Truncation for long names

**Estimate:** 2 hours

---

### T-WGPU-P4.6.1 - Frame Fence

**Description:** Implement frame fence tracking.

**Prerequisites:** Phase 1 queue

**Deliverable:** FrameFence struct

**Acceptance Criteria:**
- [ ] submission_index tracking
- [ ] wait() method
- [ ] is_complete() non-blocking query
- [ ] Reset on new submission

**Estimate:** 4 hours

---

### T-WGPU-P4.6.2 - Double Buffering

**Description:** Implement double-buffered frame sync.

**Prerequisites:** T-WGPU-P4.6.1

**Deliverable:** DoubleBufferedRenderer struct

**Acceptance Criteria:**
- [ ] 2 frame fences
- [ ] Ping-pong buffer index
- [ ] Wait on frame N-1 before writing N
- [ ] Present after render

**Estimate:** 4 hours

---

### T-WGPU-P4.6.3 - Triple Buffering

**Description:** Implement triple-buffered frame sync.

**Prerequisites:** T-WGPU-P4.6.2

**Deliverable:** TrinityFrameSynchronizer struct

**Acceptance Criteria:**
- [ ] Configurable buffer count (2 or 3)
- [ ] N fences array
- [ ] Wait on frame N-2 for triple
- [ ] Lower latency mode option

**Estimate:** 3 hours

---

### T-WGPU-P4.6.4 - Frame Pacing

**Description:** Implement frame pacing logic.

**Prerequisites:** T-WGPU-P4.6.3

**Deliverable:** FramePacer struct

**Acceptance Criteria:**
- [ ] Target frame time
- [ ] Actual frame time tracking
- [ ] Variance calculation
- [ ] FPS reporting
- [ ] Adaptive pacing (optional)

**Estimate:** 4 hours

---

### T-WGPU-P4.7.1 - Resource State Tracking

**Description:** Implement resource state tracking.

**Prerequisites:** None

**Deliverable:** ResourceState struct, state map

**Acceptance Criteria:**
- [ ] PipelineStage enum
- [ ] AccessFlags bitflags
- [ ] TextureLayout enum (8 states)
- [ ] HashMap<ResourceId, ResourceState>
- [ ] State query method

**Estimate:** 6 hours

---

### T-WGPU-P4.7.2 - Barrier Detection

**Description:** Implement barrier need detection.

**Prerequisites:** T-WGPU-P4.7.1

**Deliverable:** `needs_barrier()` function

**Acceptance Criteria:**
- [ ] RAW (read-after-write) detection
- [ ] WAR (write-after-read) detection
- [ ] WAW (write-after-write) detection
- [ ] Layout transition detection
- [ ] BarrierType enum

**Estimate:** 4 hours

---

### T-WGPU-P4.7.3 - Layout Transitions

**Description:** Implement texture layout transitions.

**Prerequisites:** T-WGPU-P4.7.2

**Deliverable:** Layout transition helpers

**Acceptance Criteria:**
- [ ] All 8 TextureLayout states
- [ ] valid_transition() check
- [ ] transition_layout() helper
- [ ] Implicit transitions documented

**Estimate:** 4 hours

---

### T-WGPU-P4.7.4 - Barrier Batching

**Description:** Implement barrier batching by stage.

**Prerequisites:** T-WGPU-P4.7.3

**Deliverable:** BarrierResolver.batch_barriers()

**Acceptance Criteria:**
- [ ] Group barriers by source stage
- [ ] Group barriers by dest stage
- [ ] Minimize barrier calls
- [ ] PassBarriers struct

**Estimate:** 4 hours

---

### T-WGPU-P4.8.1 - Buffer Mapping Async

**Description:** Implement async buffer mapping.

**Prerequisites:** Phase 2 buffers

**Deliverable:** Async mapping helpers

**Acceptance Criteria:**
- [ ] map_async() wrapper
- [ ] Maintain::Poll/Wait options
- [ ] tokio async integration
- [ ] Callback pattern (non-async)

**Estimate:** 4 hours

---

### T-WGPU-P4.8.2 - Buffer Readback Utility

**Description:** Implement BufferReadback utility struct.

**Prerequisites:** T-WGPU-P4.8.1

**Deliverable:** BufferReadback struct

**Acceptance Criteria:**
- [ ] Creates staging buffer
- [ ] Issues copy
- [ ] Maps async
- [ ] Returns data slice
- [ ] Cleanup on drop

**Estimate:** 4 hours

---

### T-WGPU-P4.9.1 - Unit Tests

**Description:** Write unit tests for Phase 4 components.

**Prerequisites:** All T-WGPU-P4.1-8 tasks

**Deliverable:** Tests in commands/tests/, sync/tests/

**Acceptance Criteria:**
- [ ] Encoder state machine tests
- [ ] Copy alignment tests
- [ ] Barrier detection tests
- [ ] Layout transition tests
- [ ] Frame fence tests
- [ ] 80%+ coverage

**Estimate:** 6 hours

---

### T-WGPU-P4.9.2 - Integration Tests

**Description:** Write integration tests for sync operations.

**Prerequisites:** T-WGPU-P4.9.1

**Deliverable:** Integration tests

**Acceptance Criteria:**
- [ ] Buffer copy round-trip test
- [ ] Texture copy test
- [ ] Timestamp query test
- [ ] Frame pacing test
- [ ] Readback test

**Estimate:** 6 hours

---

## Task Dependencies

```
Phase 3 Complete
    |
    +---> T-WGPU-P4.1.1 (Encoder creation)
              |
              +---> T-WGPU-P4.1.2 (Lifecycle)
                        |
                        +---> T-WGPU-P4.1.3 (Pass tracking)
                        +---> T-WGPU-P4.1.4 (Finalization)
    |
    +---> T-WGPU-P4.2.1 (Buffer copy)
              |
              +---> T-WGPU-P4.2.5 (Alignment)
                        |
                        +---> T-WGPU-P4.2.2, P4.2.3, P4.2.4
    |
    +---> T-WGPU-P4.3.1, P4.3.2 (Clear)
    |
    +---> T-WGPU-P4.4.1 (Timestamp pool)
              |
              +---> T-WGPU-P4.4.2 (Resolve)
                        |
                        +---> T-WGPU-P4.4.3 (Async readback)
              +---> T-WGPU-P4.4.4, P4.4.5 (Occlusion, statistics)
    |
    +---> T-WGPU-P4.5.1 (Debug RAII) --> P4.5.2, P4.5.3
    |
    +---> T-WGPU-P4.6.1 (Frame fence)
              |
              +---> T-WGPU-P4.6.2 (Double buffer)
                        |
                        +---> T-WGPU-P4.6.3 (Triple buffer)
                                  |
                                  +---> T-WGPU-P4.6.4 (Pacing)
    |
    +---> T-WGPU-P4.7.1 (State tracking)
              |
              +---> T-WGPU-P4.7.2 (Barrier detection)
                        |
                        +---> T-WGPU-P4.7.3 (Layouts)
                        +---> T-WGPU-P4.7.4 (Batching)
    |
    +---> T-WGPU-P4.8.1 (Mapping async) --> P4.8.2 (Readback)

All --> T-WGPU-P4.9.1 --> T-WGPU-P4.9.2
```

---

*End of PHASE_4_SYNCHRONIZATION_TODO.md*
