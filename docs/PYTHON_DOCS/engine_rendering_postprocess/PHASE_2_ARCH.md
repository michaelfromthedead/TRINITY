# PHASE 2 ARCH: GPU Integration

## Objective

Connect the stub `execute()` methods to real GPU resources via RHI command list integration. This phase replaces `None` placeholder buffers with actual GPU allocations and records commands to the frame graph.

## Architecture Decisions

### AD-1: RHI Abstraction Layer

**Decision**: GPU execution flows through the existing RHI (Rendering Hardware Interface) abstraction.

**Rationale**: The investigation shows existing dependencies on `engine.rendering.framegraph`. GPU work should use the same abstraction layer for consistency.

**Integration Points**:
- `FrameGraph.add_pass()` for registering GPU work
- `PassNode` for defining resource dependencies
- `ResourceFormat` for texture formats

### AD-2: Buffer Allocation Strategy

**Decision**: Replace `None` placeholders with proper GPU buffer allocation during effect initialization.

**Rationale**: Current stub methods return `None` for `_ao_buffer`, `_output_buffer`, `_motion_buffer`. These must become real GPU resources.

**Allocation Points**:
- Effect `__init__()`: Allocate persistent buffers
- `execute()`: Use pre-allocated buffers or allocate transient buffers
- `IntermediateTargetManager`: Manages ping-pong targets

### AD-3: Intermediate Target Management

**Decision**: Use existing `IntermediateTargetManager` for ping-pong buffer allocation.

**Rationale**: The PostProcessStack already has this infrastructure (per investigation line 35). Connect it to real GPU resources.

**Pattern**:
```python
class IntermediateTargetManager:
    def acquire_target(self, width, height, format) -> GPUTexture:
        # Allocate or reuse from pool
    
    def release_target(self, target: GPUTexture):
        # Return to pool for reuse
```

### AD-4: Command List Recording Pattern

**Decision**: Each `execute()` method records commands to a command list, does not submit immediately.

**Rationale**: Frame graph batches and reorders GPU work. Individual effects should not submit directly.

**Pattern**:
```python
def execute(self, input_buffer, output_buffer, cmd_list):
    cmd_list.set_pipeline(self.pipeline)
    cmd_list.set_bind_group(0, self.bind_group)
    cmd_list.dispatch(width // 8, height // 8, 1)
    # Does NOT call cmd_list.submit()
```

### AD-5: Pass Node Resource Declaration

**Decision**: Each effect declares resource dependencies via `PassNode` read/write semantics.

**Rationale**: Frame graph needs to know which resources each pass reads and writes for automatic barrier insertion.

**Pattern**:
```python
def add_to_frame_graph(self, frame_graph, input_resource):
    pass_node = frame_graph.add_pass("SSAO")
    pass_node.read(input_resource.depth)
    pass_node.read(input_resource.normal)
    pass_node.write(self._ao_buffer)
    return pass_node
```

### AD-6: Effect Pipeline Initialization

**Decision**: Pipeline state objects (PSOs) are created at effect initialization, not per-frame.

**Rationale**: PSO creation is expensive. Effects should cache pipelines.

**Initialization**:
```python
class SSAOEffect:
    def __init__(self, device, shader_module):
        self.pipeline = device.create_compute_pipeline(
            shader=shader_module,
            layout=self.bind_group_layout
        )
```

### AD-7: Settings-to-Uniform Translation

**Decision**: Settings dataclasses translate to GPU uniform buffers.

**Rationale**: Settings like `SSAOSettings`, `BloomSettings` contain parameters that shaders need. These must be uploaded to GPU.

**Pattern**:
```python
@dataclass
class SSAOSettings:
    radius: float
    bias: float
    intensity: float
    
    def to_uniform_data(self) -> bytes:
        return struct.pack('fff', self.radius, self.bias, self.intensity)
```

## Components to Modify

### Effects Returning Placeholder Buffers

| Effect | Current Return | Required Change |
|--------|----------------|-----------------|
| SSAO.calculate() | `self._ao_buffer` (None) | Return allocated GPUTexture |
| HBAO.calculate() | `self._ao_buffer` (None) | Return allocated GPUTexture |
| GTAO.calculate() | `self._ao_buffer` (None) | Return allocated GPUTexture |
| UpscalingEffect.upscale() | `self._output_buffer` (None) | Return allocated GPUTexture |
| TAA.apply() | placeholder history | Return blended result |
| MotionBlur.apply_blur() | `self._motion_buffer` (None) | Return allocated GPUTexture |
| NearFieldDOF.blur() | placeholder | Return allocated GPUTexture |
| FarFieldDOF.blur() | placeholder | Return allocated GPUTexture |

### Frame Graph Integration

The investigation shows existing frame graph integration (line 35, 167). Verify and complete:

- [ ] `add_to_frame_graph()` correctly declares read/write dependencies
- [ ] `PassFlags` (COMPUTE, GRAPHICS) correctly set
- [ ] Resource lifetime managed by frame graph

### IntermediateTargetManager

Currently manages ping-pong targets conceptually. Connect to real GPU:

- [ ] `acquire_target()` allocates from GPU pool
- [ ] `release_target()` returns to pool
- [ ] Format conversion handled

## Dependencies

### Required RHI Components

| Component | Purpose |
|-----------|---------|
| `Device` | GPU device handle for resource creation |
| `CommandList` | Records GPU commands |
| `ComputePipeline` | Pipeline state for compute shaders |
| `BindGroup` | Resource binding for shaders |
| `Texture` | GPU texture resource |
| `Buffer` | GPU buffer resource |
| `Sampler` | Texture sampling state |

### Required from Frame Graph

| Component | Purpose |
|-----------|---------|
| `FrameGraph` | Orchestrates pass execution |
| `PassNode` | Declares resource dependencies |
| `ResourceFormat` | Texture format specification |

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| RHI API not finalized | High | Define minimal interface, implement adapter |
| Resource lifetime bugs | Medium | Use frame graph for automatic lifetime |
| Performance regression | Medium | Profile before/after, optimize hot paths |
| Barrier placement wrong | Medium | Let frame graph handle barriers automatically |

## Deliverables

1. Modified effect classes with real GPU buffer allocation
2. Pipeline initialization code for each effect
3. Command list recording in all `execute()` methods
4. Verified frame graph integration
5. IntermediateTargetManager connected to GPU pool
6. Settings-to-uniform translation for all settings classes
