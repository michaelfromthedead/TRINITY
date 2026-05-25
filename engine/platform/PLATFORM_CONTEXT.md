# PLATFORM_CONTEXT.md — Platform Layer

> **Purpose**: Complete implementation reference for the engine/platform/ layer.  
> Read this file and ONLY this file when implementing platform systems.

---

## 1. Architecture Summary

The platform layer is the absolute foundation of the engine — Layer 0/1 in bootstrap. It abstracts OS services, windowing, input devices, file I/O, threading, timers, GPU access (RHI), and audio backends behind clean interfaces so all higher layers are platform-agnostic.

**Core Subsystems (5):**
1. **OS Abstraction** — File I/O (sync/async/mmap), threading primitives, virtual memory, dynamic libraries, process/system info, locale, high-res timing
2. **Window & Display** — Window creation/lifecycle, multi-monitor, DPI, fullscreen, cursor, HDR, VRR (FreeSync/G-Sync), low latency (Reflex/Anti-Lag)
3. **Render Hardware Interface (RHI)** — Unified GPU abstraction across Vulkan/D3D12/Metal/WebGPU: device, queues, buffers, textures, pipelines, shaders, command buffers, swap chain, barriers, ray tracing, mesh shaders
4. **Audio API** — Audio device enumeration, backend abstraction (WASAPI/CoreAudio/AAudio/ALSA/PulseAudio/XAudio2/WebAudio), spatial audio (Windows Sonic/Tempest 3D)
5. **Input API** — Keyboard, mouse, gamepad, touch, pen, XR controllers, XR hand tracking, haptics (rumble, adaptive triggers, HD rumble)

**Architectural Principles:**
- **Zero-Cost Abstraction** — Compile-time polymorphism where possible
- **Minimal Surface Area** — Expose only what higher layers need
- **Explicit Resource Ownership** — Clear ownership, deterministic lifetimes
- **Fail-Safe Defaults** — Sensible defaults, graceful degradation

**Bootstrap Position (Layer 0/1):**
```
Layer 0: Metaclasses + Foundation bootstrap (Python imports)
Layer 1: Platform registration <- THIS LAYER
Layer 2: Core Systems (Memory, Math, Task, ECS)
Layer 3: Engine Loop (Bootstrap, Scheduler, World)
Layer 4+: Rendering, Simulation, Animation, Audio, Gameplay...
```

Platform resources register first via `ResourceMeta` -> Foundation Registry, making them discoverable by all higher layers via `registry.subclasses(GPUDevice)` etc.

**Platform Targets:**
| Platform | OS | Graphics | Audio | Input |
|----------|-----|----------|-------|-------|
| Windows | Win32 | Vulkan, D3D12 | WASAPI, XAudio2 | Raw Input, XInput |
| Linux | POSIX | Vulkan | ALSA, PulseAudio | evdev, SDL |
| macOS | Darwin | Metal | Core Audio | HID, SDL |
| iOS | Darwin | Metal | Core Audio | Touch, Gamepad |
| Android | Linux | Vulkan | AAudio | Touch, Gamepad |
| Web | Browser | WebGPU | Web Audio | Gamepad API |
| PS5 | Proprietary | AGC | Tempest 3D | DualSense |
| Xbox | Windows | D3D12 | XAudio2 | Xbox Controller |
| Switch | Proprietary | NVN | nn::audio | Joy-Con |

**Threading Model:**
- File I/O callbacks -> dedicated I/O threads
- Audio callbacks -> high-priority audio thread
- Input polling -> main thread
- RHI command recording -> thread-safe; submission -> not thread-safe

---

## 2. Decorators

### 2.1 ECS Core (Foundation Tier 0)

#### @resource (ecs_core.py)
```python
@resource(name: str = None)
```
- **Steps:** TAG(resource=True, resource_name), REGISTER(ecs_core)
- **After-Apply:** `_resource=True`, `_resource_name`, `_resource_id` (auto), `_resource_priority=100`, `_resource_dependencies=()`
- **Singleton enforcement** via `ResourceMeta.__call__()`
- **Platform usage:** ALL platform subsystems are Resources (Window, GPUDevice, FileSystem, ThreadPool, InputManager, AudioDevice, Timer)

#### @system (ecs_core.py)
```python
@system(phase: str = "update")
```
- **Steps:** TAG(system=True, system_phase), REGISTER(ecs_core)
- **After-Apply:** `_system=True`, `_system_phase=SystemPhase[phase]`, `_system_id`, `_reads=()`, `_writes=()`, `_exclusive=False`, `_priority=0`, `_can_parallelize=True`
- **Platform usage:** InputSystem (phase="pre_physics"), WindowSystem (phase="pre_update")

#### @component (ecs_core.py)
```python
@component(name: str = None)
```
- **Steps:** TAG(component=True, component_name), REGISTER(ecs_core)
- **After-Apply:** `_component=True`, `_component_id`, `_field_types={}`, `_field_descriptors={}`, `_field_offsets={}`, `_field_defaults={}`
- **Platform usage:** WindowConfig, DisplayInfo, InputState components

#### @event (ecs_core.py)
```python
@event
```
- **Steps:** TAG(event=True), REGISTER(ecs_core)
- **After-Apply:** `_event=True`, `_event_id`, `_event_name`, `_event_fields={}`, `_event_priority=0`, `_event_channels=()`, `_event_pooled=False`
- **Platform usage:** WindowResized, InputEvent, DeviceConnected, FileIOComplete events

### 2.2 Lifecycle Decorators (Tier 7)

| Decorator | Signature | After-Apply | Platform Use |
|-----------|-----------|-------------|--------------|
| @on_add | `@on_add(component)` | `_on_add_component`, `_lifecycle_hook="add"` | Component attached to entity |
| @on_remove | `@on_remove(component)` | `_on_remove_component`, `_lifecycle_hook="remove"` | Component removed |
| @on_spawn | `@on_spawn()` | `_on_spawn=True` | Entity created |
| @on_despawn | `@on_despawn()` | `_on_despawn=True` | Entity destroyed |

### 2.3 Scheduling Decorators (Tier 5)

#### @phase (scheduling.py)
```python
@phase(name: str, after: tuple = (), before: tuple = ())
```
- **After-Apply:** `_phase=True`, `_phase_name`, `_phase_after`, `_phase_before`
- **Platform usage:** Define INPUT phase before SIMULATION

#### @parallel (scheduling.py)
```python
@parallel(chunk_size: int = 64, min_batch: int = 256)
```
- **After-Apply:** `_parallel=True`, `_parallel_chunk_size`, `_parallel_min_batch`
- **Excludes:** @exclusive

#### @exclusive (scheduling.py)
```python
@exclusive
```
- **After-Apply:** `_exclusive=True`
- **Excludes:** @parallel
- **Platform usage:** Window event pump must be exclusive (main thread only)

#### @fixed (scheduling.py)
```python
@fixed(hz: int = 60)
```
- **After-Apply:** `_fixed=True`, `_fixed_hz`, `_fixed_delta=1.0/hz`
- **Excludes:** @throttle

#### @job (scheduling.py)
```python
@job(priority: int = 0, affinity: str = "any", stack_size: int = 65536)
```
- **After-Apply:** `_job=True`, `_job_priority`, `_job_affinity`, `_job_stack_size`
- **Platform usage:** File I/O jobs, asset loading jobs

#### @async_system (scheduling.py)
```python
@async_system
```
- **After-Apply:** `_async_system=True`, `_is_coroutine`
- **Platform usage:** Async file I/O system, async network system

#### @throttle (scheduling.py)
```python
@throttle(max_hz: float = None, max_ms: float = None)
```
- **After-Apply:** `_throttle=True`, `_throttle_max_hz`, `_throttle_max_ms`
- **Platform usage:** Throttle input polling, display enumeration

### 2.4 Memory Decorators (Tier 3)

#### @pooled (memory.py)
```python
@pooled(initial_size: int = 1024, grow_factor: float = 2.0, max_size: int = None)
```
- **After-Apply:** `_pooled=True`, `_pool_config=PoolConfig(...)`
- **Methods Added:** `release()` -> return to pool

#### @packed (memory.py)
```python
@packed(layout: str = "soa")  # "soa" | "aos" | "hybrid"
```
- **After-Apply:** `_packed=True`, `_packed_layout`, `_packed_config`

#### @aligned (memory.py)
```python
@aligned(bytes: int = 64)  # Must be power of 2
```
- **After-Apply:** `_aligned=True`, `_aligned_bytes`
- **Platform usage:** GPU buffers need 256-byte alignment, cache-line alignment for threading

#### @arena (memory.py)
```python
@arena(name: str = "default")
```
- **After-Apply:** `_arena=True`, `_arena_name`
- **Platform usage:** Per-frame arena for transient platform data

#### @budget (memory.py)
```python
@budget(category: str = "gameplay", max_bytes: int = None, warn_at: float = 0.8)
```
- **After-Apply:** `_budget=True`, `_budget_category`, `_budget_max_bytes`, `_budget_warn_at`
- **Platform usage:** GPU memory budget, console fixed memory budgets

#### @allocator (memory.py)
```python
@allocator(type: str, size: int, thread_safe: bool = False)
```
- **After-Apply:** `_allocator=True`, `_allocator_type` ("linear"|"pool"|"stack"|"buddy"|"tlsf"), `_allocator_size`, `_allocator_thread_safe`
- **Platform usage:** Platform allocators wrapping OS virtual memory

#### @atomic (memory.py)
```python
@atomic
```
- **After-Apply:** `_atomic=True`, `_atomic_lock`
- **Methods Added:** `atomic_load()`, `atomic_store()`, `atomic_exchange()`, `compare_exchange()`, `fetch_add()`, `fetch_sub()`
- **Platform usage:** Thread-safe platform state (connection counts, memory counters)

### 2.5 Data Flow Decorators (Tier 4)

#### @serializable (data_flow.py)
```python
@serializable(format: str = "binary", version: int = 1)
```
- **After-Apply:** `_serializable=True`, `_serializable_format`, `_serializable_version`, `_serializable_fields`
- **Methods Added:** `serialize()`, `deserialize()` (classmethods)
- **Platform usage:** Window state save/restore, input mapping persistence

#### @snapshot (data_flow.py)
```python
@snapshot(history_frames: int = 60)
```
- **After-Apply:** `_snapshot=True`, `_snapshot_history_frames`
- **Methods Added:** `snapshot_save()`, `snapshot_restore(frame)`
- **Requires:** @serializable

### 2.6 Development Decorators (Tier 6)

#### @profile (dev.py)
```python
@profile(name: str = None, warn_ms: float = None, track_allocations: bool = False)
```
- **After-Apply:** `_profiled=True`, `_profile_name`
- **Methods Added:** `profile_stats()`, `profile_reset()`
- **Platform usage:** Profile GPU submission, file I/O latency

#### @gpu_profile (dev.py)
```python
@gpu_profile(category: str, include_memory: bool = False)
```
- **After-Apply:** `_gpu_profiled=True`, `_gpu_profile_category`, `_gpu_profile_include_memory`
- **Platform usage:** GPU timestamp queries, pipeline stats

#### @trace (dev.py)
```python
@trace(level: str = "debug")  # "debug" | "info" | "warn"
```
- **After-Apply:** `_traced=True`, `_trace_level`
- **Platform usage:** Trace platform initialization, device enumeration

#### @reloadable (dev.py)
```python
@reloadable(enabled: bool = True, preserve: list = [], reinitialize: list = [], validate: callable = None)
```
- **After-Apply:** `_reloadable=True`, `_reload_preserve`, `_reload_reinitialize`, `_reload_validate`
- **Platform usage:** Hot-reload shaders, dynamic library reload

#### @bench (dev.py)
```python
@bench(iterations: int = 1000, warmup: int = 100)
```
- **After-Apply:** `_bench=True`, `_bench_iterations`, `_bench_warmup`
- **Platform usage:** Benchmark allocator performance, I/O throughput

#### @invariant (dev.py)
```python
@invariant(check: callable, when: str = "debug")  # "debug" | "always"
```
- **After-Apply:** `_invariants=[{check, when}, ...]`
- **Platform usage:** Assert GPU device valid, window handle not null

### 2.7 Debug & Safety Decorators (Tier 10-11)

#### @reads / @writes (debug_safety.py)
```python
@reads(Component1, Component2, ...)
@writes(Component1, Component2, ...)
```
- **After-Apply:** `_reads_components=tuple`, `_writes_components=tuple`
- **Platform usage:** Declare platform system data access for parallelization safety

#### @track_changes (debug_safety.py)
```python
@track_changes(fields: list = None)  # None = all fields
```
- **After-Apply:** `_tracked=True`, `_tracked_fields`
- **Platform usage:** Track window state changes, GPU memory changes

### 2.8 Security Decorators (Tier 8)

#### @validated (security.py)
```python
@validated(rules: list)  # List of callables
```
- **After-Apply:** `_validated=True`, `_validation_rules`
- **Platform usage:** Validate input values, file paths (prevent traversal)

#### @rate_limited (security.py)
```python
@rate_limited(max_per_second: float = 10.0, per: str = "player")
```
- **After-Apply:** `_rate_limited=True`, `_rate_limit_max`, `_rate_limit_per`
- **Platform usage:** Rate-limit file operations, input event generation

#### @encrypted (security.py)
```python
@encrypted
```
- **After-Apply:** `_encrypted=True`
- **Platform usage:** Encrypt save data, credential storage

---

## 3. Metaclasses

### ResourceMeta (PRIMARY for platform)
```python
class ResourceMeta(EngineMeta):
    _registry: dict[int, type]
    _instances: dict[int, Any]  # Singleton storage
    _next_id: int
```
- **__new__:** Assigns `_resource_id`, `_resource_name`, `_resource_priority=100`, `_resource_dependencies=()`
- **__call__:** Singleton enforcement -- returns existing instance or creates new
- **Key Methods:**
  - `get_by_id(id)` -> type
  - `get_by_name(name)` -> type
  - `all_resources()` -> list[type]
  - `get_instance(cls)` -> instance
  - `has_instance(cls)` -> bool
  - `initialize_all()` -> topologically sorts by dependencies+priority, initializes in order
  - `shutdown_all()` -> shutdown in reverse order, calls `shutdown()` on each
  - `reset_instance(cls)` -> remove for re-creation
  - `get_or_create(cls)` -> get or create with dependency validation
- **Platform usage:** ALL platform subsystems are Resources. `initialize_all()` boots the platform layer. `shutdown_all()` tears it down.

### SystemMeta (for platform systems)
```python
class SystemMeta(EngineMeta):
    _registry: dict[int, type]
    _phases: dict[SystemPhase, list[int]]
```
- **__new__:** Assigns `_system_id`, validates @reads/@writes, analyzes dependencies, checks parallelization
- **Key Methods:**
  - `get_phase_order(phase)` -> topologically sorted systems for a phase
  - `get_parallel_groups(phase)` -> groups of non-conflicting systems
- **Platform usage:** InputSystem, WindowSystem, FileWatchSystem register here

### ComponentMeta (for platform data)
```python
class ComponentMeta(EngineMeta):
    _registry: dict[int, type]
```
- **__new__:** Processes fields from `Annotated[]` hints, installs descriptor chains, validates
- **Key Methods:**
  - `_process_fields(cls)` -> extract field types, offsets, defaults, descriptors
  - `_install_descriptors(cls)` -> build descriptor chains from decorators
- **Platform usage:** WindowConfig, DisplayInfo, InputMapping components

### EventMeta (for platform events)
- **Platform usage:** WindowResized, InputReceived, DeviceConnected, FileIOComplete events

### ProtocolMeta (for RHI)
- **Purpose:** Define network/GPU wire formats with version negotiation
- **Required attribute:** `_protocol_version: int`
- **Platform usage:** RHI abstraction protocol -- unified GPU command interface across backends

---

## 4. Descriptors

### TrackedDescriptor (tracking.py)
```python
TrackedDescriptor(field_offset: int = 0, use_bitmask: bool = False)
```
- **descriptor_id:** "tracked"
- **accepts_inner:** ("storage", "validated", "range")
- **accepts_outer:** ("networked", "observable", "cached")
- **post_set():** If value changed -> `obj._dirty_mask |= 1 << offset` (bitmask) or `obj._dirty_fields.add(name)` (set)
- **Helpers:** `is_dirty(obj, field)`, `get_dirty_fields(obj)`, `clear_dirty(obj)`
- **Platform usage:** Track window size/position changes, GPU memory allocation changes, input state changes -> feeds Foundation Tracker

### ObservableDescriptor (observable.py)
```python
ObservableDescriptor()
```
- **descriptor_id:** "observable"
- **accepts_inner:** ("tracked", "storage", "validated", "range")
- **post_set():** Notifies all registered observers: `callback(obj, name, old_value, new_value)`
- **Helpers:** `add_observer(obj, field, callback)`, `remove_observer()`, `clear_observers()`
- **Platform usage:** Observe window resize -> update render target, observe input config -> rebind keys

### ValidatedDescriptor (validation.py)
```python
ValidatedDescriptor(validators: list = None)
```
- **descriptor_id:** "validated"
- **pre_set():** Runs all validators, raises ValueError on failure
- **Platform usage:** Validate resolution (min/max), validate file paths (no traversal), validate input ranges

### RangeDescriptor (validation.py)
```python
RangeDescriptor(min_val: float, max_val: float, clamp: bool = True)
```
- **descriptor_id:** "range"
- **pre_set():** Clamps or raises ValueError if out of range
- **Platform usage:** Clamp volume levels, brightness, gamma, mouse sensitivity

### TypeDescriptor (validation.py)
```python
TypeDescriptor(expected_type: type = None, coerce: bool = False)
```
- **descriptor_id:** "typed"
- **pre_set():** Type check or coerce
- **Platform usage:** Ensure GPU handles are correct type

### ProfiledDescriptor
- **Platform usage:** Profile file I/O descriptor access timing -> EventLog

### LoggedDescriptor (AuditDescriptor)
- **Platform usage:** Audit log all platform configuration changes -> EventLog

---

## 5. Foundation Integration Points

### Foundation Registry -> Platform Resource Discovery
```
ResourceMeta.__new__() -> registry.register(cls, name="platform.Window")
    -> registry.subclasses(GPUDevice)  # Higher layers discover GPU backends
    -> registry.instances(InputDevice)  # Find all connected input devices
    -> registry.metadata(Window)         # Query window capabilities
```

### Foundation Tracker -> Platform State Changes
```
Window.size = (1920, 1080)
    -> TrackedDescriptor.post_set()
    -> Tracker.mark_dirty(window, "size")
    -> Tracker fires subscriptions (renderer gets notified)
    -> Per-frame: Tracker.all_dirty() collects all platform changes
```

### Foundation EventLog -> Platform Operations
```
@system(phase="input")
@trace
class InputSystem(System):
    def update(self, dt):
        # EventLog.record(Event("InputSystem.update", depth=0))
        for event in poll_events():
            self.process(event)
            # EventLog.record(Event("process_input", root_cause="InputSystem.update"))
```

### Foundation Mirror -> Platform Introspection
```python
from foundation.mirror import mirror

m = mirror(gpu_device)
m.get("memory_allocated")      # Read GPU memory
m.set("debug_mode", True)      # Set debug flag
m.get_path("capabilities.supports_ray_tracing")  # Nested access
m.schema_hash()                # Detect config drift
```

### Foundation Bridge -> ShellLang World Sync
```python
adapter = TrinityWorldAdapter()
adapter.add_instance(gpu_device)  # Trinity GPU -> ShellLang entity
adapter.add_instance(window)      # Trinity Window -> ShellLang entity
# ShellLang can now query: Window.all.where(fullscreen=True)
```

### Foundation Capabilities -> Platform Permissions
```python
Capabilities.has(mod, "file_write")    # Can mod write files?
Capabilities.has(mod, "gpu_access")    # Can mod access GPU?
Capabilities.has(mod, "network")       # Can mod open sockets?
```

---

## 6. Architecture Spec Details

### 6.1 OS Abstraction

#### File I/O
| Operation | Sync | Async | Memory-Mapped |
|-----------|------|-------|---------------|
| Open | `open(path, mode)` | `async_open(path, mode, callback)` | `mmap(path, offset, size)` |
| Read | `read(handle, buf, size)` | `async_read(handle, buf, callback)` | Direct pointer access |
| Write | `write(handle, buf, size)` | `async_write(handle, buf, callback)` | Direct pointer write |
| Close | `close(handle)` | `async_close(handle, callback)` | `munmap(ptr)` |

**Async Backends:** Completion Ports (Windows), io_uring (Linux), kqueue (macOS)
**File Watching:** Directory monitoring for hot-reload support
**Path Utilities:** Normalization, platform path conversion, safe path validation

#### Threading Primitives
| Primitive | Purpose | Platform Mapping |
|-----------|---------|-----------------|
| Thread | OS thread | `CreateThread` / `pthread_create` |
| Mutex | Mutual exclusion | `CRITICAL_SECTION` / `pthread_mutex` |
| RWLock | Shared read / exclusive write | `SRWLOCK` / `pthread_rwlock` |
| Semaphore | Counting/binary sync | `CreateSemaphore` / `sem_init` |
| CondVar | Wait/signal/broadcast | `CONDITION_VARIABLE` / `pthread_cond` |
| Barrier | Synchronization point | `SYNCHRONIZATION_BARRIER` / `pthread_barrier` |
| Atomics | Lock-free operations | CAS, Fetch-Add, memory ordering |
| TLS | Thread-local storage | `TlsAlloc` / `__thread` |

#### Virtual Memory
| Operation | Purpose | Windows | Linux |
|-----------|---------|---------|-------|
| Reserve | Reserve address space | `VirtualAlloc(MEM_RESERVE)` | `mmap(PROT_NONE)` |
| Commit | Back with physical pages | `VirtualAlloc(MEM_COMMIT)` | `mmap(PROT_READ\|WRITE)` |
| Decommit | Release pages, keep reservation | `VirtualFree(MEM_DECOMMIT)` | `madvise(DONTNEED)` |
| Protect | Set R/W/X flags | `VirtualProtect` | `mprotect` |
| Large Pages | 2MB/1GB huge pages | Large page support | hugetlbfs |

#### Dynamic Libraries
- **Load:** Runtime binding of .dll/.so/.dylib
- **Symbol Lookup:** `GetProcAddress` / `dlsym`
- **Unload:** Reference-counted
- **Hot Reload:** Code patching for development

#### System Info
- CPU count, total memory, cache line sizes
- Environment variables, paths, working directory
- Locale (language, region)
- High-resolution timing: `QueryPerformanceCounter` / `clock_gettime`

### 6.2 Window & Display

#### Window Lifecycle
```
Create -> Show -> [Resize | Minimize | Maximize | Fullscreen] -> Hide -> Destroy
                                    |
                        Event Pump (main thread)
                        Cursor control (show/hide/confine/custom)
                        DPI awareness (per-monitor)
```

#### Display Management
- **Enumeration:** All monitors + adapters
- **Modes:** Resolution + refresh rate enumeration
- **Multi-Monitor:** Arrangement detection, primary monitor
- **DPI:** Per-monitor DPI handling, scale factor

#### HDR & Advanced Display
| Feature | Description |
|---------|-------------|
| HDR Query | Detect display HDR support |
| Color Space | sRGB, scRGB, HDR10, PQ |
| HDR Metadata | Mastering display luminance info |
| VRR | FreeSync, G-Sync, HDMI 2.1 VRR |
| Low Latency | NVIDIA Reflex, AMD Anti-Lag |

### 6.3 Render Hardware Interface (RHI)

#### Supported Backends
| Backend | Platforms | Features |
|---------|-----------|----------|
| Vulkan | Windows, Linux, Android | Dynamic rendering, sync2, descriptor indexing, RT, mesh shaders |
| D3D12 | Windows, Xbox | Enhanced barriers, work graphs, mesh shaders, DXR 1.1 |
| Metal | macOS, iOS | Argument buffers, mesh shaders (3.0+), RT (3.0+), bindless |
| WebGPU | Web | Compute, limited RT |

#### RHI Core Objects
```
Adapter -> Device -> Queues (Graphics, Compute, Transfer)
                      |
                      +- Resources: Buffer, Texture, Sampler, View (SRV/UAV/RTV/DSV), Heap
                      +- Pipeline: Shader, PSO (Graphics/Compute/RT), Root Signature, Input Layout
                      +- Commands: Command List, Command Queue, Indirect, Timestamps
                      +- Binding: Descriptor Heap, Bindless Arrays, Push Constants
                      +- Sync: Fence (CPU-GPU), Semaphore (GPU-GPU), Barrier
```

#### Resource Types
| Resource | Key Properties |
|----------|---------------|
| Buffer | size, usage (Vertex/Index/Constant/Storage/Indirect), memory type |
| Texture | dimensions, mip levels, array size, format, type, usage, samples |
| Sampler | filter, address mode, LOD, anisotropy, comparison |
| SRV | Shader read view |
| UAV | Compute read/write view |
| RTV | Render target view |
| DSV | Depth stencil view |

#### Pipeline State
| Pipeline | Components |
|----------|-----------|
| Graphics | VS+PS, input layout, topology, rasterizer, depth/stencil, blend, root sig |
| Compute | CS, root signature |
| Raytracing | Ray gen, miss, hit groups, recursion depth, root sigs |
| Mesh | Task (optional) + Mesh + Pixel shaders, state |

#### Command Recording
- Barrier management (transition, UAV, aliasing)
- Render pass lifecycle (begin/end)
- State setting (pipeline, viewport, scissor, blend factor, stencil ref)
- Binding (descriptor heap, constants, CBV, SRV, UAV, descriptor table)
- Draw calls (vertex, indexed, indirect, instanced)
- Compute dispatch
- Copy (buffer<->buffer, texture<->texture, buffer<->texture)
- Ray tracing (BLAS/TLAS build, dispatch rays)
- Mesh shader dispatch

#### Presentation
| Property | Options |
|----------|---------|
| Present Mode | Immediate, VSync, Mailbox |
| Color Space | sRGB, scRGB, HDR10, PQ |
| Buffer Count | 2 (double) or 3 (triple) |
| HDR Metadata | Mastering display info for tone mapping |

#### Synchronization
- **Fence:** CPU-GPU sync with monotonic values
- **Barrier:** State transitions (e.g., RENDER_TARGET -> SHADER_RESOURCE)

### 6.4 Audio API

#### Audio Device
```python
AudioDevice:
    enumerate(type) -> list[AudioDeviceInfo]
    get_default(type) -> AudioDeviceInfo
    open(info, callback, buffer_size) -> AudioStream
```

**AudioDeviceInfo:** name, type (output/input), channels, sample_rate, format

#### Backends
| Backend | Platform | Notes |
|---------|----------|-------|
| WASAPI | Windows | Low-latency, exclusive mode |
| Core Audio | macOS, iOS | Low-latency, spatial |
| AAudio | Android | Low-latency, performance mode |
| ALSA | Linux | Direct hardware |
| PulseAudio | Linux | Desktop integration |
| XAudio2 | Windows, Xbox | Game-focused |
| Web Audio | Browser | JS interop |

#### Spatial Audio
- Windows Sonic / Dolby Atmos (Windows)
- Tempest 3D Audio (PS5)
- Apple Spatial Audio (AirPods/HomePod)

### 6.5 Input API

#### Device Types
| Device | Events | Extras |
|--------|--------|--------|
| Keyboard | key_down, key_pressed, key_released | Scan codes + virtual keys |
| Mouse | position, delta, scroll, buttons | Raw input (no acceleration) |
| Gamepad | axes, triggers, buttons | Deadzone, vibration |
| Touch | id, position, pressure, phase | Multi-touch |
| Pen | position, pressure, tilt | Stylus support |
| XR Controller | pose, velocity, buttons, triggers | Haptics |
| XR Hand | 25 joints per hand, pinch/grip | Gesture detection |

#### Haptics
| Type | Description | Platform |
|------|-------------|----------|
| Rumble | Frequency-based vibration | All gamepads |
| Adaptive Trigger | Position + strength resistance | DualSense |
| HD Rumble | Fine-grained haptics | Joy-Con |

### 6.6 Platform-Specific Concerns

#### Console
- **Certification:** TRC (PlayStation), XR (Xbox), Lotcheck (Nintendo)
- **Memory:** Fixed budgets, no overcommit
- **GPU:** Fixed hardware, known capabilities
- **User:** Platform profiles, sign-in states
- **Save:** Platform cloud, mandatory backup
- **Achievements:** Trophies / Gamerscore / Badges

#### Mobile
- **App Lifecycle:** Background/foreground transitions
- **Thermal:** Throttling detection + response
- **Battery:** Power mode queries, optimization hints
- **Safe Areas:** Notch, rounded corners, gesture areas
- **Permissions:** Storage, camera, microphone runtime requests

#### Desktop
- **Multi-GPU:** SLI, CrossFire, explicit multi-adapter
- **Overlays:** Steam, Discord, GeForce Experience
- **Capture:** Screenshot, video recording
- **Shader Cache:** PSO cache, precompilation

#### Suspend & Resume
- **State Preservation:** GPU context save/restore
- **Audio Pause:** Stream handling during suspend
- **Network:** Reconnection after resume
- **Quick Resume:** Xbox Series instant resume

#### Platform Services
- **Store:** In-app purchases, DLC, entitlements
- **Social:** Friends, invites, rich presence
- **Cloud:** Save sync, cross-platform progress
- **Voice:** Platform voice integration
- **Streaming:** PS Remote Play, Steam Link

---

## 7. Decorator Stacks

### @platform_adaptive -- Battery/LOD/Streaming
```python
from trinity.decorators.builtin_stacks.platform import platform_adaptive

@platform_adaptive(lod_levels=3)
class TerrainRenderer(Component):
    detail_level: int = 2
```
**Combines:** @battery_aware + @lod(levels) + @streamable(priority="normal")
**Purpose:** Automatically adapt quality to platform capabilities and power state

### Custom Platform Stacks (to be defined)

#### platform_resource -- Standard Platform Resource
```python
# Proposed composition:
@resource
@serializable(format="binary")
@track_changes
@profile
@trace
class PlatformResourceBase(Resource): ...
```
**Combines:** @resource + @serializable + @track_changes + @profile + @trace
**Purpose:** Standard pattern for all platform resources (tracked, profiled, serializable for session save)

#### gpu_resource -- GPU-Managed Resource
```python
# Proposed composition:
@resource
@budget(category="gpu")
@aligned(bytes=256)
@track_changes
@gpu_profile(category="memory")
class GPUResourceBase(Resource): ...
```
**Combines:** @resource + @budget + @aligned(256) + @track_changes + @gpu_profile
**Purpose:** GPU resources with memory budgeting, alignment, and profiling

#### platform_system -- Standard Platform System
```python
# Proposed composition:
@system(phase="pre_update")
@profile
@trace
@exclusive  # Most platform systems are main-thread
class PlatformSystemBase(System): ...
```
**Combines:** @system + @profile + @trace + @exclusive
**Purpose:** Platform systems that must run on main thread, with profiling

#### async_io_system -- Async I/O System
```python
# Proposed composition:
@system(phase="pre_update")
@async_system
@profile
@trace
class AsyncIOBase(System): ...
```
**Combines:** @system + @async_system + @profile + @trace
**Purpose:** Asynchronous file I/O and network systems

---

## 8. TODO Checklist (from GAME_ENGINE_INTEGRATION_TODO.md section 2)

### 2.1 OS Abstraction
- [ ] Implement window management (create, resize, fullscreen, multi-monitor)
- [ ] Implement input polling (keyboard, mouse, gamepad raw events)
- [ ] Implement file I/O abstraction (sync reads/writes)
- [ ] Implement async file I/O (completion ports / io_uring / kqueue)
- [ ] Implement memory-mapped file access
- [ ] Implement file watching (directory monitoring for hot-reload)
- [ ] Implement path utilities (normalization, platform conversion, safe validation)
- [ ] Implement threading primitives (threads, mutexes, RWLocks, semaphores, condition variables, barriers, atomics, TLS)
- [ ] Implement virtual memory (reserve, commit, decommit, protect, large pages, query)
- [ ] Implement dynamic library loading (load, symbol lookup, unload, hot reload)
- [ ] Implement process/system info (CPU count, memory, cache sizes, env vars)
- [ ] Implement high-resolution timing (monotonic clock, ticks, frequency)
- [ ] Register ALL platform Resources via ResourceMeta -> Foundation Registry
- [ ] Wire @native decorator to mark platform-specific implementations
- [ ] Wire Foundation Tracker -> platform state change tracking
- [ ] Wire Foundation EventLog -> platform operation logging

### 2.2 Render Hardware Interface (RHI)
- [ ] Define RHI abstraction protocol using ProtocolMeta
- [ ] Implement adapter enumeration and feature query
- [ ] Implement device creation and resource lifecycle
- [ ] Implement Vulkan backend (or initial backend choice)
- [ ] Implement GPU resource management (buffers, textures, samplers, views, heaps)
- [ ] Implement pipeline state objects (graphics, compute, raytracing, mesh)
- [ ] Implement command buffer recording and submission
- [ ] Implement swap chain management (create, present, resize)
- [ ] Implement synchronization (fences, barriers, semaphores)
- [ ] Implement descriptor binding (descriptor heaps, bindless, push constants)
- [ ] Implement ray tracing support (BLAS, TLAS, dispatch rays)
- [ ] Implement mesh shader support (task + mesh dispatch)
- [ ] Register GPU device as Resource in Foundation Registry
- [ ] Wire @gpu_buffer, @gpu_kernel decorators to RHI resources
- [ ] Wire @gpu_profile to RHI timestamp queries
- [ ] Wire Foundation Mirror -> GPU resource introspection

### 2.3 Platform Services
- [ ] Implement timer/clock (high-resolution, monotonic, ticks_per_second)
- [ ] Implement dynamic library loading (for plugins/mods)
- [ ] Wire @target decorator for platform-conditional compilation
- [ ] Implement display management (enumeration, modes, multi-monitor, DPI)
- [ ] Implement HDR support (query, color space, metadata)
- [ ] Implement VRR support (FreeSync, G-Sync, HDMI 2.1)
- [ ] Implement low-latency support (Reflex, Anti-Lag)
- [ ] Implement audio device enumeration and backend selection
- [ ] Implement input device enumeration (keyboard, mouse, gamepad, touch, XR)
- [ ] Implement haptic feedback (rumble, adaptive triggers, HD rumble)
- [ ] Implement cursor management (show, hide, confine, custom cursors)
- [ ] Implement clipboard access (copy/paste text, images)
- [ ] Implement suspend/resume handling (GPU save/restore, audio pause, network reconnect)
- [ ] Implement platform service integration (store, social, cloud, voice, streaming)

### Console-Specific
- [ ] Implement certification compliance (TRC, XR, Lotcheck)
- [ ] Implement fixed memory budget management
- [ ] Implement platform user management (profiles, sign-in)
- [ ] Implement platform save system (cloud, backup)
- [ ] Implement achievements/trophies integration

### Mobile-Specific
- [ ] Implement app lifecycle (background/foreground)
- [ ] Implement thermal throttling detection/response
- [ ] Implement battery optimization
- [ ] Implement safe area detection
- [ ] Implement runtime permission requests

---

## 9. Directory Structure

```
engine/platform/
├── __init__.py                    # Public API exports: file_system(), thread_pool(), 
│                                  #   virtual_memory(), create_window(), poll_events(),
│                                  #   create_graphics_device(), get_audio_device(),
│                                  #   input(), services(), time_seconds(), time_ticks()
├── os/
│   ├── __init__.py
│   ├── file_system.py             # Sync/async/mmap file I/O + path utilities
│   ├── file_watcher.py            # Directory monitoring for hot-reload
│   ├── threading.py               # Thread, Mutex, RWLock, Semaphore, CondVar, Barrier
│   ├── atomics.py                 # Atomic operations, CAS, memory ordering
│   ├── virtual_memory.py          # Reserve/commit/decommit/protect/large pages
│   ├── dynamic_library.py         # Load/symbol lookup/unload/hot reload
│   ├── system_info.py             # CPU count, memory, cache, env, locale
│   └── timing.py                  # High-res monotonic clock, ticks, frequency
├── window/
│   ├── __init__.py
│   ├── window.py                  # Window creation, lifecycle, resize, fullscreen
│   ├── display.py                 # Monitor enumeration, modes, multi-monitor, DPI
│   ├── cursor.py                  # Show/hide/confine/custom cursors
│   ├── hdr.py                     # HDR query, color space, metadata
│   └── vrr.py                     # VRR support (FreeSync, G-Sync, HDMI 2.1)
├── rhi/
│   ├── __init__.py
│   ├── device.py                  # Adapter enumeration, device creation, feature query
│   ├── resources.py               # Buffer, Texture, Sampler, View, Heap
│   ├── pipeline.py                # Graphics/Compute/RT/Mesh PSOs, root signature
│   ├── commands.py                # Command list recording, barriers, draw/dispatch
│   ├── swapchain.py               # Swap chain create/present/resize
│   ├── sync.py                    # Fence, Semaphore, Barrier
│   ├── binding.py                 # Descriptor heaps, bindless, push constants
│   ├── raytracing.py              # BLAS, TLAS, dispatch rays
│   ├── mesh_shaders.py            # Task + Mesh shader dispatch
│   └── backends/
│       ├── __init__.py
│       ├── vulkan.py              # Vulkan backend implementation
│       ├── d3d12.py               # D3D12 backend implementation
│       ├── metal.py               # Metal backend implementation
│       └── webgpu.py              # WebGPU backend implementation
├── audio/
│   ├── __init__.py
│   ├── audio_device.py            # Device enumeration, open, callback
│   ├── spatial.py                 # Windows Sonic, Tempest 3D, Apple Spatial
│   └── backends/
│       ├── __init__.py
│       ├── wasapi.py              # Windows WASAPI
│       ├── core_audio.py          # macOS/iOS Core Audio
│       ├── aaudio.py              # Android AAudio
│       ├── alsa.py                # Linux ALSA
│       ├── pulse.py               # Linux PulseAudio
│       ├── xaudio2.py             # Windows/Xbox XAudio2
│       └── web_audio.py           # Browser Web Audio
├── input/
│   ├── __init__.py
│   ├── input_manager.py           # Device enumeration, event polling, dispatch
│   ├── keyboard.py                # Key down/pressed/released, scan codes
│   ├── mouse.py                   # Position, delta, scroll, buttons, raw input
│   ├── gamepad.py                 # Axes, triggers, buttons, deadzone
│   ├── touch.py                   # Multi-touch (id, position, pressure, phase)
│   ├── pen.py                     # Pressure-sensitive stylus
│   ├── haptics.py                 # Rumble, adaptive triggers, HD rumble
│   └── xr_input.py               # XR controllers (pose, buttons), XR hands (25 joints)
├── services/
│   ├── __init__.py
│   ├── platform_detect.py         # Compile-time + runtime platform detection
│   ├── permissions.py             # Runtime permission requests (mobile)
│   ├── app_lifecycle.py           # Suspend/resume, background/foreground
│   ├── store.py                   # IAP, DLC, entitlements
│   ├── social.py                  # Friends, invites, presence
│   ├── cloud.py                   # Cloud saves, cross-platform sync
│   ├── achievements.py            # Trophies, Gamerscore, Badges
│   └── streaming.py               # Remote Play, Steam Link
└── gpu/
    ├── __init__.py
    └── low_latency.py             # NVIDIA Reflex, AMD Anti-Lag
```

---

## 10. Canonical Usage Examples

### Example 1: Platform Window Resource
```python
from typing import Annotated
from trinity.base import Resource
from trinity.descriptors.tracking import TrackedDescriptor
from trinity.descriptors.observable import ObservableDescriptor
from trinity.descriptors.validation import RangeDescriptor

@resource
@serializable(format="binary")
@track_changes
class Window(Resource):
    """Platform window -- singleton, tracked, observable."""
    _resource_priority = 10  # Early init
    _resource_dependencies = ()

    width: Annotated[int, TrackedDescriptor(), RangeDescriptor(min_val=320, max_val=7680)]
    height: Annotated[int, TrackedDescriptor(), RangeDescriptor(min_val=240, max_val=4320)]
    fullscreen: Annotated[bool, TrackedDescriptor(), ObservableDescriptor()]
    vsync: Annotated[bool, TrackedDescriptor()]
    title: str = "Game"

    def initialize(self, config: dict):
        """Create OS window handle."""
        self.width = config.get("width", 1920)
        self.height = config.get("height", 1080)
        self.fullscreen = config.get("fullscreen", False)
        self._handle = _create_native_window(self.width, self.height, self.title)

    def shutdown(self):
        """Destroy OS window handle."""
        _destroy_native_window(self._handle)
        self._handle = None
```

### Example 2: GPU Device Resource
```python
@resource
@budget(category="gpu")
@track_changes
@gpu_profile(category="device", include_memory=True)
class GPUDevice(Resource):
    """RHI GPU device -- singleton, budgeted, profiled."""
    _resource_priority = 20  # After Window
    _resource_dependencies = (Window,)

    memory_allocated: Annotated[int, TrackedDescriptor()]
    memory_budget: Annotated[int, TrackedDescriptor()]
    adapter_name: str = ""
    supports_raytracing: bool = False
    supports_mesh_shaders: bool = False

    def initialize(self):
        adapter = self._enumerate_adapters()[0]  # Best adapter
        self.adapter_name = adapter.name
        self.memory_budget = adapter.dedicated_memory
        self.supports_raytracing = adapter.features.raytracing
        self.supports_mesh_shaders = adapter.features.mesh_shaders
        self._device = _create_device(adapter)

    def shutdown(self):
        _destroy_device(self._device)
```

### Example 3: Input System
```python
@system(phase="pre_physics")
@exclusive  # Main thread only
@profile
@trace
@reads(InputState)
@writes(InputState)
class InputSystem(System):
    """Polls platform input devices, updates InputState components."""

    def execute(self, dt: float):
        for event in poll_native_events():
            match event.type:
                case "key_down":
                    self._handle_key(event)
                case "mouse_move":
                    self._handle_mouse(event)
                case "gamepad_axis":
                    self._handle_gamepad(event)
                case "touch":
                    self._handle_touch(event)

    def _handle_key(self, event):
        # Update InputState component on entity
        pass
```

### Example 4: Async File I/O Resource
```python
@resource
@profile
@trace
class FileSystem(Resource):
    """Platform file I/O abstraction -- sync, async, mmap."""
    _resource_priority = 5  # Very early (before GPU)

    def read_sync(self, path: str) -> bytes:
        """Synchronous file read."""
        pass

    async def read_async(self, path: str) -> bytes:
        """Async file read (io_uring / IOCP / kqueue)."""
        pass

    def mmap_read(self, path: str, offset: int = 0, size: int = 0):
        """Memory-mapped file access."""
        pass

    def watch(self, path: str, callback):
        """Watch directory for changes (hot-reload)."""
        pass

    def shutdown(self):
        """Close all handles, stop watchers."""
        pass
```

### Example 5: Platform Bootstrap
```python
def bootstrap_platform():
    """Layer 1 bootstrap -- initialize all platform resources in priority order."""
    # ResourceMeta.initialize_all() handles ordering:
    # 1. FileSystem (priority=5)   -- file access first
    # 2. Window (priority=10)      -- OS window
    # 3. GPUDevice (priority=20)   -- needs Window
    # 4. AudioDevice (priority=25) -- audio backend
    # 5. InputManager (priority=30)-- input devices
    # 6. Timer (priority=35)       -- high-res clock
    # 7. PlatformServices (priority=40) -- store, social, cloud
    ResourceMeta.initialize_all()
```

---

## 11. Integration Patterns

### Pattern 1: Resource Registration -> Discovery
```python
# PLATFORM DEFINES:
@resource
class GPUDevice(Resource): ...

@resource
class VulkanDevice(GPUDevice): ...

@resource
class D3D12Device(GPUDevice): ...

# HIGHER LAYER DISCOVERS:
gpu_backends = registry.subclasses(GPUDevice)   # [VulkanDevice, D3D12Device]
gpu = registry.instances(GPUDevice)[0]           # Active GPU device
```

### Pattern 2: Platform State -> Foundation Tracker -> Subscribers
```python
# Window resizes:
window.width = 2560
# -> TrackedDescriptor.post_set() fires
# -> Tracker.mark_dirty(window, "width")
# -> Renderer's subscription fires: on_window_resize(old, new)
# -> Renderer recreates swap chain
```

### Pattern 3: Platform Operations -> EventLog Audit Trail
```python
# Every platform operation logged:
EventLog.record("file_open", {"path": "/assets/mesh.bin", "mode": "read"})
EventLog.record("gpu_alloc", {"type": "texture", "size": 4194304, "format": "BC7"})
EventLog.record("input_device", {"type": "gamepad", "action": "connected", "id": 0})

# Queryable:
file_ops = EventLog.events_for_operation("file_open")
gpu_allocs = EventLog.events_for_operation("gpu_alloc")
```

### Pattern 4: Platform -> Bridge -> ShellLang Debugging
```python
# Live debugging via ShellLang:
# > QUERY Window WHERE fullscreen == True
# > MUTATE Window SET vsync = False
# > QUERY GPUDevice WHERE memory_allocated > 1000000000
# > QUERY InputDevice WHERE type == "gamepad"
```

### Pattern 5: Error Handling -- Result<T> Pattern
```python
class Result(Generic[T]):
    """Platform error handling pattern."""
    value: Optional[T]
    error: Optional[ErrorCode]

    def ok(self) -> bool:
        return self.error is None

# Usage:
result = file_system.read_sync("/assets/texture.bin")
if result.ok():
    data = result.value
else:
    log.error(f"File read failed: {result.error}")
```

### Pattern 6: RAII Resource Lifetime
```python
class ScopedGPUBuffer:
    """RAII wrapper for GPU buffer -- auto-destroys on scope exit."""
    def __init__(self, device, size, usage):
        self._buffer = device.create_buffer(size, usage)

    def __del__(self):
        if self._buffer:
            self._buffer.destroy()

    def __enter__(self):
        return self._buffer

    def __exit__(self, *args):
        self._buffer.destroy()
        self._buffer = None
```

---

## 12. Quick Reference Tables

### Platform Resource Priority Order (Bootstrap)
| Priority | Resource | Dependencies | Purpose |
|----------|----------|-------------|---------|
| 5 | FileSystem | -- | File access (needed by all) |
| 10 | Window | -- | OS window |
| 15 | Display | Window | Monitor/DPI management |
| 20 | GPUDevice | Window | GPU + RHI |
| 25 | AudioDevice | -- | Audio backend |
| 30 | InputManager | Window | Input devices |
| 35 | Timer | -- | High-res clock |
| 40 | PlatformServices | -- | Store, social, cloud |

### SystemPhase Enum
| Phase | Value | Platform Systems |
|-------|-------|-----------------|
| PRE_PHYSICS | 0 | InputSystem |
| PHYSICS | 1 | -- |
| POST_PHYSICS | 2 | -- |
| PRE_UPDATE | 3 | WindowSystem, FileWatchSystem |
| UPDATE | 4 | -- |
| POST_UPDATE | 5 | -- |
| PRE_RENDER | 6 | -- |
| RENDER | 7 | SwapChainPresent |

### Platform Decorators Summary
| Decorator | Tier | Platform Role |
|-----------|------|---------------|
| @resource | 0 | ALL platform subsystems |
| @system | 0 | Platform tick systems |
| @component | 0 | Platform config data |
| @event | 0 | Platform events |
| @exclusive | 5 | Main-thread enforcement |
| @async_system | 5 | Async I/O systems |
| @job | 5 | File/asset loading jobs |
| @fixed | 5 | Fixed-rate platform polling |
| @throttle | 5 | Rate-limited operations |
| @pooled | 3 | Platform object pools |
| @aligned | 3 | GPU buffer alignment |
| @budget | 3 | GPU/console memory limits |
| @allocator | 3 | Platform memory allocators |
| @atomic | 3 | Thread-safe platform state |
| @serializable | 4 | Platform config persistence |
| @profile | 6 | Platform perf measurement |
| @gpu_profile | 6 | GPU timestamp/memory |
| @trace | 6 | Platform operation logging |
| @reloadable | 6 | Shader/library hot-reload |
| @bench | 6 | Platform benchmarks |
| @invariant | 6 | Platform safety checks |
| @track_changes | 11 | Dirty flags -> Foundation |
| @validated | 8 | Input/path validation |
| @rate_limited | 8 | Operation rate limiting |

### RHI Object Hierarchy
```
Adapter
  +- Device
       +- Queue (Graphics)
       +- Queue (Compute)
       +- Queue (Transfer)
       +- Heap
       |   +- Buffer
       |   +- Texture
       |   +- Sampler
       +- Pipeline State
       |   +- Graphics PSO
       |   +- Compute PSO
       |   +- RT PSO
       |   +- Mesh PSO
       +- Command List
       +- Fence
       +- Semaphore
       +- Swap Chain
```

### Descriptor Stacking for Platform Fields
```
StorageDescriptor           <- innermost (raw storage)
  -> RangeDescriptor         <- value bounds (resolution limits)
    -> ValidatedDescriptor   <- custom validators
      -> TrackedDescriptor   <- dirty flags -> Foundation Tracker
        -> ObservableDescriptor <- subscriber notifications
```

### Error Handling
```python
Result<T> = { value: T | None, error: ErrorCode | None }
result.ok() -> bool
# Use for ALL platform operations that can fail
```

### Constants (from trinity/constants.py)
| Constant | Value | Platform Use |
|----------|-------|-------------|
| CACHE_LINE_BYTES | 64 | Thread-safe alignment |
| DEFAULT_POOL_SIZE | 1024 | Platform object pools |
| DEFAULT_RESOURCE_PRIORITY | 100 | Resource init order |
| DEFAULT_PHYSICS_HZ | 60 | Fixed-rate input polling |

---

*End of PLATFORM_CONTEXT.md -- This file is the sole reference for implementing engine/platform/.*
