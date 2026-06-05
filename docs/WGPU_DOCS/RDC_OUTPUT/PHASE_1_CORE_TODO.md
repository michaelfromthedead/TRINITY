# PHASE 1: CORE - Task List

**Phase:** 1 - CORE
**Estimated Duration:** 2-3 weeks
**Task ID Prefix:** T-WGPU-P1

---

## Task Summary

| ID | Task | Est. Hours | Status |
|----|------|------------|--------|
| T-WGPU-P1.1.1 | Instance creation | 4 | - |
| T-WGPU-P1.1.2 | Backend selection | 4 | - |
| T-WGPU-P1.1.3 | Instance flags | 2 | - |
| T-WGPU-P1.2.1 | Adapter enumeration | 4 | - |
| T-WGPU-P1.2.2 | Adapter properties | 3 | - |
| T-WGPU-P1.2.3 | Adapter limits query | 3 | - |
| T-WGPU-P1.2.4 | Feature detection | 4 | - |
| T-WGPU-P1.2.5 | Adapter selection algorithm | 6 | - |
| T-WGPU-P1.3.1 | Device creation | 4 | - |
| T-WGPU-P1.3.2 | Feature negotiation | 4 | - |
| T-WGPU-P1.3.3 | Limit negotiation | 3 | - |
| T-WGPU-P1.3.4 | Device lost handling | 6 | - |
| T-WGPU-P1.3.5 | Error scopes | 4 | - |
| T-WGPU-P1.4.1 | Queue submission | 4 | - |
| T-WGPU-P1.4.2 | Queue writes | 3 | - |
| T-WGPU-P1.4.3 | Submission batching | 6 | - |
| T-WGPU-P1.5.1 | Capability tier detection | 4 | - |
| T-WGPU-P1.5.2 | CapabilityManager | 6 | - |
| T-WGPU-P1.6.1 | Unit tests | 8 | - |
| T-WGPU-P1.6.2 | Integration tests | 6 | - |

**Total Estimated Hours:** 88 hours

---

## Detailed Tasks

### T-WGPU-P1.1.1 - Instance Creation

**Description:** Implement TrinityInstance struct wrapping wgpu::Instance with multi-backend support.

**Prerequisites:** None

**Deliverable:** `device/instance.rs` with TrinityInstance struct and new() method

**Acceptance Criteria:**
- [ ] Instance creates successfully with Backends::PRIMARY on desktop
- [ ] Instance creates successfully with Backends::BROWSER_WEBGPU on WASM
- [ ] Instance logs backend selection
- [ ] Unit test verifies instance creation

**Estimate:** 4 hours

---

### T-WGPU-P1.1.2 - Backend Selection

**Description:** Implement platform-aware backend selection logic.

**Prerequisites:** T-WGPU-P1.1.1

**Deliverable:** `select_backends()` function in instance.rs

**Acceptance Criteria:**
- [ ] Windows: Vulkan > DX12 > OpenGL
- [ ] macOS: Metal only
- [ ] Linux: Vulkan > OpenGL
- [ ] WASM: BROWSER_WEBGPU only
- [ ] Configurable override via environment variable

**Estimate:** 4 hours

---

### T-WGPU-P1.1.3 - Instance Flags

**Description:** Implement instance flag configuration for debug/validation modes.

**Prerequisites:** T-WGPU-P1.1.1

**Deliverable:** Instance flag configuration in TrinityInstance::new()

**Acceptance Criteria:**
- [ ] VALIDATION flag enabled in debug builds
- [ ] DEBUG flag enabled when WGPU_DEBUG=1
- [ ] Performance impact documented
- [ ] Validation catches logged via error callback

**Estimate:** 2 hours

---

### T-WGPU-P1.2.1 - Adapter Enumeration

**Description:** Implement adapter enumeration and listing.

**Prerequisites:** T-WGPU-P1.1.1

**Deliverable:** `enumerate_adapters()` method in adapter.rs

**Acceptance Criteria:**
- [ ] Returns all available adapters
- [ ] Filters by requested backend
- [ ] Logs adapter info for debugging
- [ ] Handles zero adapters gracefully

**Estimate:** 4 hours

---

### T-WGPU-P1.2.2 - Adapter Properties

**Description:** Extract and expose adapter properties (vendor, device type, driver).

**Prerequisites:** T-WGPU-P1.2.1

**Deliverable:** AdapterInfo wrapper struct

**Acceptance Criteria:**
- [ ] Vendor ID extraction (NVIDIA, AMD, Intel, Apple, ARM)
- [ ] DeviceType detection (DiscreteGpu, IntegratedGpu, Cpu)
- [ ] Driver version when available
- [ ] Human-readable adapter description

**Estimate:** 3 hours

---

### T-WGPU-P1.2.3 - Adapter Limits Query

**Description:** Query and expose adapter limits.

**Prerequisites:** T-WGPU-P1.2.1

**Deliverable:** `inspect_limits()` function

**Acceptance Criteria:**
- [ ] All texture limits exposed
- [ ] All buffer limits exposed
- [ ] All bind group limits exposed
- [ ] All compute limits exposed
- [ ] Formatted output for debugging

**Estimate:** 3 hours

---

### T-WGPU-P1.2.4 - Feature Detection

**Description:** Query and expose available features.

**Prerequisites:** T-WGPU-P1.2.1

**Deliverable:** OptionalFeatures struct with detection logic

**Acceptance Criteria:**
- [ ] All 16+ optional features queried
- [ ] Feature dependencies expanded
- [ ] Feature tier assignment (Minimal/Standard/Advanced/Full)
- [ ] Platform-specific feature availability documented

**Estimate:** 4 hours

---

### T-WGPU-P1.2.5 - Adapter Selection Algorithm

**Description:** Implement scoring-based adapter selection.

**Prerequisites:** T-WGPU-P1.2.2, T-WGPU-P1.2.3, T-WGPU-P1.2.4

**Deliverable:** AdapterSelector struct with select() method

**Acceptance Criteria:**
- [ ] Score by device type (discrete > integrated > software)
- [ ] Score by feature availability
- [ ] Score by limits
- [ ] Blacklist support (for known-broken drivers)
- [ ] Vendor preference support
- [ ] Falls back to first available if no preference match

**Estimate:** 6 hours

---

### T-WGPU-P1.3.1 - Device Creation

**Description:** Implement device creation from selected adapter.

**Prerequisites:** T-WGPU-P1.2.5

**Deliverable:** TrinityDevice struct with new() method

**Acceptance Criteria:**
- [ ] Creates device with requested features
- [ ] Creates device with requested limits
- [ ] Handles request failure gracefully
- [ ] Logs device creation details

**Estimate:** 4 hours

---

### T-WGPU-P1.3.2 - Feature Negotiation

**Description:** Implement feature negotiation (required vs optional).

**Prerequisites:** T-WGPU-P1.3.1

**Deliverable:** DeviceRequirements struct, negotiate_features()

**Acceptance Criteria:**
- [ ] Required features cause failure if unavailable
- [ ] Optional features degraded gracefully
- [ ] Feature dependencies automatically added
- [ ] Final feature set logged

**Estimate:** 4 hours

---

### T-WGPU-P1.3.3 - Limit Negotiation

**Description:** Implement limit negotiation (request capped to adapter).

**Prerequisites:** T-WGPU-P1.3.1

**Deliverable:** LimitRequirements struct, negotiate_limits()

**Acceptance Criteria:**
- [ ] TRINITY minimums enforced (64KB uniform, 128MB storage, 8K texture)
- [ ] Requests capped to adapter limits
- [ ] Shortfall logged with warning
- [ ] Final limits accessible

**Estimate:** 3 hours

---

### T-WGPU-P1.3.4 - Device Lost Handling

**Description:** Implement device lost callback and recovery.

**Prerequisites:** T-WGPU-P1.3.1

**Deliverable:** DeviceManager with lost callback, recovery logic

**Acceptance Criteria:**
- [ ] Lost callback invoked on device loss
- [ ] Recovery attempts device recreation
- [ ] Resource tracking for rebuild
- [ ] Maximum retry limit with backoff
- [ ] Fatal error if recovery fails

**Estimate:** 6 hours

---

### T-WGPU-P1.3.5 - Error Scopes

**Description:** Implement error scope wrapper for fine-grained error handling.

**Prerequisites:** T-WGPU-P1.3.1

**Deliverable:** ErrorScope struct with RAII pattern

**Acceptance Criteria:**
- [ ] push_error_scope() on creation
- [ ] pop_error_scope() on drop
- [ ] Validation and OutOfMemory filters
- [ ] Async error retrieval
- [ ] Error logging and propagation

**Estimate:** 4 hours

---

### T-WGPU-P1.4.1 - Queue Submission

**Description:** Implement command buffer submission.

**Prerequisites:** T-WGPU-P1.3.1

**Deliverable:** TrinityQueue struct with submit() method

**Acceptance Criteria:**
- [ ] Accepts single or multiple command buffers
- [ ] Returns SubmissionIndex
- [ ] Tracks pending submissions
- [ ] Works with on_submitted_work_done callback

**Estimate:** 4 hours

---

### T-WGPU-P1.4.2 - Queue Writes

**Description:** Implement direct buffer and texture writes.

**Prerequisites:** T-WGPU-P1.4.1

**Deliverable:** write_buffer(), write_texture() methods

**Acceptance Criteria:**
- [ ] Buffer write with offset and data
- [ ] Texture write with layout specification
- [ ] Alignment validation
- [ ] Size validation

**Estimate:** 3 hours

---

### T-WGPU-P1.4.3 - Submission Batching

**Description:** Implement submission batching for performance.

**Prerequisites:** T-WGPU-P1.4.1

**Deliverable:** SubmissionBatcher struct

**Acceptance Criteria:**
- [ ] Batch by command buffer count (threshold: 8)
- [ ] Batch by time (threshold: 2ms)
- [ ] Flush on frame end
- [ ] Force flush API for synchronous operations
- [ ] Metrics for batch effectiveness

**Estimate:** 6 hours

---

### T-WGPU-P1.5.1 - Capability Tier Detection

**Description:** Implement capability tier detection algorithm.

**Prerequisites:** T-WGPU-P1.2.4

**Deliverable:** CapabilityTier enum, from_adapter() method

**Acceptance Criteria:**
- [ ] Full tier: RT features present
- [ ] Advanced tier: bindless + multi-draw + large workgroup
- [ ] Standard tier: 8K textures + compute
- [ ] Minimal tier: fallback
- [ ] Tier ordering (Full > Advanced > Standard > Minimal)

**Estimate:** 4 hours

---

### T-WGPU-P1.5.2 - CapabilityManager

**Description:** Implement CapabilityManager for runtime capability queries.

**Prerequisites:** T-WGPU-P1.5.1, T-WGPU-P1.3.1

**Deliverable:** CapabilityManager struct with query methods

**Acceptance Criteria:**
- [ ] supports_ray_tracing()
- [ ] supports_bindless()
- [ ] supports_gpu_culling()
- [ ] supports_timestamp_queries()
- [ ] select_render_path() based on tier
- [ ] select_texture_compression()
- [ ] max_bindless_textures()
- [ ] report() -> CapabilityReport

**Estimate:** 6 hours

---

### T-WGPU-P1.6.1 - Unit Tests

**Description:** Write unit tests for all Phase 1 components.

**Prerequisites:** All T-WGPU-P1.1-5 tasks

**Deliverable:** Tests in device/tests/

**Acceptance Criteria:**
- [ ] Instance creation tests
- [ ] Adapter scoring tests with mock data
- [ ] Feature negotiation tests
- [ ] Limit negotiation tests
- [ ] Error scope tests
- [ ] Capability tier tests
- [ ] 80%+ code coverage

**Estimate:** 8 hours

---

### T-WGPU-P1.6.2 - Integration Tests

**Description:** Write integration tests for full device initialization.

**Prerequisites:** T-WGPU-P1.6.1

**Deliverable:** Integration tests in tests/

**Acceptance Criteria:**
- [ ] Full initialization sequence test
- [ ] Device lost recovery test (if simulable)
- [ ] Queue submission test
- [ ] Cross-backend test matrix (CI)
- [ ] WebGPU test (browser or wasm-pack)

**Estimate:** 6 hours

---

## Task Dependencies

```
T-WGPU-P1.1.1 --> T-WGPU-P1.1.2
T-WGPU-P1.1.1 --> T-WGPU-P1.1.3
T-WGPU-P1.1.1 --> T-WGPU-P1.2.1

T-WGPU-P1.2.1 --> T-WGPU-P1.2.2
T-WGPU-P1.2.1 --> T-WGPU-P1.2.3
T-WGPU-P1.2.1 --> T-WGPU-P1.2.4

T-WGPU-P1.2.2 + T-WGPU-P1.2.3 + T-WGPU-P1.2.4 --> T-WGPU-P1.2.5

T-WGPU-P1.2.5 --> T-WGPU-P1.3.1

T-WGPU-P1.3.1 --> T-WGPU-P1.3.2
T-WGPU-P1.3.1 --> T-WGPU-P1.3.3
T-WGPU-P1.3.1 --> T-WGPU-P1.3.4
T-WGPU-P1.3.1 --> T-WGPU-P1.3.5

T-WGPU-P1.3.1 --> T-WGPU-P1.4.1
T-WGPU-P1.4.1 --> T-WGPU-P1.4.2
T-WGPU-P1.4.1 --> T-WGPU-P1.4.3

T-WGPU-P1.2.4 --> T-WGPU-P1.5.1
T-WGPU-P1.5.1 + T-WGPU-P1.3.1 --> T-WGPU-P1.5.2

All --> T-WGPU-P1.6.1 --> T-WGPU-P1.6.2
```

---

*End of PHASE_1_CORE_TODO.md*
