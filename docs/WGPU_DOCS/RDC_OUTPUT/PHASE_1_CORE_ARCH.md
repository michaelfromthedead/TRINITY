# PHASE 1: CORE - Architecture

**Scope:** Device initialization, adapter selection, queue management
**Duration:** 2-3 weeks
**Dependencies:** None
**Produces:** Foundation for all subsequent phases

---

## Overview

Phase 1 establishes the wgpu entry points and device management infrastructure. This is the foundational layer upon which all rendering operations depend.

### Covered Content (from MASTER.md Part I)

- Chapter 1: The wgpu Object Model
  - 1.1 Instance (creation, backend selection, flags)
  - 1.2 Adapter (enumeration, properties, limits, features, power preference)
  - 1.3 Device (creation, features, limits, lost handling, error scopes)
  - 1.4 Queue (submission, writes, synchronization, batching)

---

## Architectural Decisions

### ADR-001: Multi-Backend Support Strategy

**Context:** TRINITY must run on Vulkan, Metal, DX12, WebGPU, and OpenGL.

**Decision:** Use wgpu's backend abstraction with automatic selection, preferring PRIMARY backends (Vulkan/Metal/DX12) over SECONDARY (OpenGL).

**Rationale:** wgpu handles backend differences; TRINITY focuses on capability tiers.

**Consequences:**
- Simplified codebase (no per-backend code paths in most modules)
- Some features unavailable on lower-tier backends
- Testing required on all target platforms

---

### ADR-002: Capability Tier System

**Context:** Hardware varies from WebGL2 mobile to RTX workstations.

**Decision:** Implement four capability tiers (Minimal/Standard/Advanced/Full) detected at adapter enumeration.

**Rationale:** Enables graceful degradation without runtime feature checks scattered through codebase.

**Consequences:**
- Render path selection is tier-based
- Feature availability queries centralized in CapabilityManager
- Some code paths never executed on low-tier hardware

---

### ADR-003: Device Lost Recovery

**Context:** GPU devices can be lost due to driver crashes, TDR, or power events.

**Decision:** Implement DeviceManager with lost callback and recovery strategy (recreate device, rebuild pipelines).

**Rationale:** Production engines must handle device loss gracefully.

**Consequences:**
- All resources must be tracked for rebuild
- Pipeline cache must support reload
- Transient state discarded on recovery

---

## Component Breakdown

### 1. Instance Management

```
TrinityInstance
├── wgpu::Instance
├── backends: Backends (PRIMARY | SECONDARY)
├── flags: InstanceFlags (VALIDATION, DEBUG)
└── select_adapter() -> Adapter
```

**Responsibilities:**
- Entry point creation
- Backend selection based on platform
- Adapter enumeration and scoring

### 2. Adapter Selection

```
AdapterSelector
├── scoring_function: fn(&AdapterInfo) -> u32
├── blacklist: Vec<VendorId>
├── vendor_preference: Option<VendorId>
└── select() -> Option<Adapter>
```

**Scoring Criteria:**
1. DeviceType (DiscreteGpu > IntegratedGpu > Cpu)
2. Feature availability (higher tier = higher score)
3. Limits (larger = higher score)
4. Vendor preference (optional boost)

### 3. Device Management

```
TrinityDevice
├── device: Arc<wgpu::Device>
├── queue: Arc<wgpu::Queue>
├── capabilities: CapabilityManager
├── error_handler: ErrorHandler
├── frame_count: AtomicU64
└── lost_callback: Option<Box<dyn Fn()>>
```

**Responsibilities:**
- Device creation with negotiated features/limits
- Error scope management
- Device lost handling and recovery
- Frame lifecycle tracking

### 4. Queue Management

```
TrinityQueue
├── queue: Arc<wgpu::Queue>
├── batcher: SubmissionBatcher
└── sync: QueueSync
```

**Responsibilities:**
- Command buffer submission
- Direct buffer/texture writes
- Submission batching (size/time thresholds)
- Work completion tracking

### 5. Capability Detection

```
CapabilityManager
├── tier: CapabilityTier
├── features: wgpu::Features
├── limits: wgpu::Limits
└── report() -> CapabilityReport
```

**Tier Detection Logic:**
- Full: RAY_TRACING + RAY_QUERY
- Advanced: TEXTURE_BINDING_ARRAY + MULTI_DRAW_INDIRECT_COUNT + large workgroup
- Standard: 8K textures + 256 workgroup + 128MB storage
- Minimal: Everything else

---

## Module Structure

```
crates/renderer-backend/src/device/
├── mod.rs              # Module exports
├── instance.rs         # TrinityInstance
├── adapter.rs          # AdapterSelector, scoring
├── device.rs           # TrinityDevice, DeviceManager
├── queue.rs            # TrinityQueue, SubmissionBatcher
├── capabilities.rs     # CapabilityManager, CapabilityTier
├── errors.rs           # ErrorScope, error handling
├── limits.rs           # LimitRequirements, negotiation
└── tests/
    ├── instance_tests.rs
    ├── adapter_tests.rs
    └── device_tests.rs
```

---

## Testing Strategy

### Unit Tests

1. **Instance creation** - Verify all backends available on platform
2. **Adapter scoring** - Test scoring function with mock AdapterInfo
3. **Capability detection** - Test tier assignment for various feature sets
4. **Limit negotiation** - Test capping to adapter limits
5. **Error scopes** - Test push/pop and error capture

### Integration Tests

1. **Full initialization** - Instance -> Adapter -> Device -> Queue
2. **Device lost simulation** - Trigger lost callback, verify recovery
3. **Cross-platform** - CI matrix for Vulkan, Metal, DX12

### Blackbox Tests

1. **Adapter enumeration** - List all adapters on system
2. **Feature detection** - Query and report all features
3. **Limit inspection** - Dump all limits to log

---

## Performance Considerations

1. **Adapter Selection** - One-time cost at startup, can take 100ms+
2. **Device Creation** - One-time cost, pipeline compilation dominates
3. **Queue Submission** - Minimize by batching (SubmissionBatcher)
4. **Error Scopes** - Minimal overhead when validation disabled

---

## Dependencies

### External Crates

- `wgpu` - Core GPU abstraction
- `pollster` - Async adapter/device request blocking
- `log` - Logging
- `thiserror` - Error types

### Internal Dependencies

- None (this is the foundation)

---

## Deliverables Checklist

- [ ] TrinityInstance with multi-backend support
- [ ] AdapterSelector with scoring algorithm
- [ ] TrinityDevice with error handling
- [ ] TrinityQueue with submission batching
- [ ] CapabilityManager with tier detection
- [ ] Device lost recovery mechanism
- [ ] Unit tests (80%+ coverage)
- [ ] Integration tests
- [ ] Documentation

---

*End of PHASE_1_CORE_ARCH.md*
