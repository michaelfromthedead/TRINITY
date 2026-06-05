# PHASE 2 TODO: GPU Integration

## Objective

Connect stub `execute()` methods to real GPU resources via RHI command list integration.

---

## T-PP-2.1: Buffer Allocation Infrastructure

### Tasks

- [ ] **T-PP-2.1.1**: Implement GPUTexturePool
  - Pool of reusable GPU textures by format and size
  - Acquire/release semantics
  - Automatic resize on demand

- [ ] **T-PP-2.1.2**: Connect IntermediateTargetManager to GPUTexturePool
  - `acquire_target()` calls pool.acquire()
  - `release_target()` calls pool.release()
  - Track in-flight targets per frame

- [ ] **T-PP-2.1.3**: Define standard texture formats
  - AO buffer: R8_UNORM or R16_FLOAT
  - Color buffer: RGBA16_FLOAT
  - Motion buffer: RG16_FLOAT
  - Depth buffer: D32_FLOAT

### Acceptance Criteria

- Pool allocates textures on demand
- Pool reuses textures when possible
- No memory leaks across frames

---

## T-PP-2.2: SSAO GPU Integration

**File**: `ambient_occlusion.py`

### Tasks

- [ ] **T-PP-2.2.1**: Allocate AO buffer in SSAO.__init__()
  - Format: R8_UNORM
  - Size: screen resolution (or half)

- [ ] **T-PP-2.2.2**: Create SSAO compute pipeline
  - Bind shader module (Phase 3 provides shader)
  - Define bind group layout: depth, normal, kernel, output

- [ ] **T-PP-2.2.3**: Upload SSAO kernel to GPU buffer
  - Generate hemisphere samples (existing code)
  - Upload to structured buffer

- [ ] **T-PP-2.2.4**: Implement SSAO.calculate() command recording
  - Bind depth texture
  - Bind normal texture
  - Bind kernel buffer
  - Bind output UAV
  - Dispatch compute

- [ ] **T-PP-2.2.5**: Return allocated AO buffer (not None)

### Acceptance Criteria

- SSAO.calculate() records GPU commands
- Returns valid GPUTexture (not None)
- Frame graph tracks AO buffer lifetime

---

## T-PP-2.3: Bloom GPU Integration

**File**: `bloom.py`

### Tasks

- [ ] **T-PP-2.3.1**: Allocate bloom mip chain
  - 8 mip levels
  - Format: RGBA16_FLOAT

- [ ] **T-PP-2.3.2**: Create threshold compute pipeline
  - Soft-knee threshold shader (Phase 3)

- [ ] **T-PP-2.3.3**: Create downsample compute pipeline
  - 13-tap downsample shader

- [ ] **T-PP-2.3.4**: Create blur compute pipeline
  - Separable Gaussian shader

- [ ] **T-PP-2.3.5**: Create upsample compute pipeline
  - Bilinear upsample with additive blend

- [ ] **T-PP-2.3.6**: Implement bloom execute() command recording
  - Threshold pass
  - Downsample passes (chain)
  - Blur passes (per mip)
  - Upsample passes (chain)
  - Final composite

### Acceptance Criteria

- Bloom pipeline executes all passes
- Mip chain correctly generated
- Final output composited to destination

---

## T-PP-2.4: Tonemapping GPU Integration

**File**: `tonemapping.py`

### Tasks

- [ ] **T-PP-2.4.1**: Create tonemap compute pipeline
  - Parameterized by operator type
  - Or: separate pipeline per operator

- [ ] **T-PP-2.4.2**: Upload tonemap settings as uniform buffer
  - Operator parameters (white point, shoulder, toe, etc.)

- [ ] **T-PP-2.4.3**: Implement tonemap execute() command recording
  - Bind input HDR texture
  - Bind settings uniform
  - Bind output LDR texture
  - Dispatch

### Acceptance Criteria

- All 8 tonemap operators supported
- Settings correctly uploaded
- Output is LDR (0-1 range)

---

## T-PP-2.5: Color Grading GPU Integration

**File**: `color_grading.py`

### Tasks

- [ ] **T-PP-2.5.1**: Upload LUT3D as 3D texture
  - 32x32x32 or 64x64x64 grid
  - Trilinear sampling enabled

- [ ] **T-PP-2.5.2**: Upload color grading matrices as uniform
  - White balance matrix
  - Channel mixer matrix
  - Lift/gamma/gain parameters

- [ ] **T-PP-2.5.3**: Create color grading compute pipeline
  - Apply all transforms in order

- [ ] **T-PP-2.5.4**: Implement color grading execute() command recording
  - Bind input texture
  - Bind LUT 3D texture
  - Bind settings uniform
  - Dispatch

### Acceptance Criteria

- LUT loaded and sampled correctly
- All color transforms applied
- Order matches CPU implementation

---

## T-PP-2.6: TAA GPU Integration

**File**: `antialiasing.py`

### Tasks

- [ ] **T-PP-2.6.1**: Allocate history buffer
  - Format: RGBA16_FLOAT
  - Persistent across frames

- [ ] **T-PP-2.6.2**: Upload jitter sequence as uniform
  - Current frame jitter offset

- [ ] **T-PP-2.6.3**: Create TAA compute pipeline
  - History reprojection
  - Neighborhood clamping
  - Temporal blend

- [ ] **T-PP-2.6.4**: Implement TAA.apply() command recording
  - Bind current frame
  - Bind motion vectors
  - Bind history
  - Bind output
  - Dispatch
  - Copy output to history

### Acceptance Criteria

- History correctly reprojected
- Ghosting minimized via clamping
- History buffer persists across frames

---

## T-PP-2.7: DOF GPU Integration

**File**: `dof.py`

### Tasks

- [ ] **T-PP-2.7.1**: Allocate CoC buffer
  - Format: R16_FLOAT
  - Per-pixel circle of confusion

- [ ] **T-PP-2.7.2**: Upload bokeh kernel as structured buffer
  - Kernel shape (disk, polygon, anamorphic)

- [ ] **T-PP-2.7.3**: Create CoC compute pipeline
  - Calculate CoC from depth

- [ ] **T-PP-2.7.4**: Create DOF blur compute pipeline
  - Near field blur
  - Far field blur
  - Composite

- [ ] **T-PP-2.7.5**: Implement DOF.blur() command recording

### Acceptance Criteria

- CoC calculated per pixel
- Blur strength varies with CoC
- Near/far field correctly separated

---

## T-PP-2.8: Exposure GPU Integration

**File**: `exposure.py`

### Tasks

- [ ] **T-PP-2.8.1**: Allocate histogram buffer
  - 256 bins
  - Atomic increment support

- [ ] **T-PP-2.8.2**: Create histogram compute pipeline
  - Parallel histogram generation

- [ ] **T-PP-2.8.3**: Create average luminance compute pipeline
  - Reduce histogram to single value

- [ ] **T-PP-2.8.4**: Implement eye adaptation GPU readback
  - Read previous frame average
  - Compute adaptation
  - Upload new exposure

### Acceptance Criteria

- Histogram generated on GPU
- Average luminance calculated
- Exposure adapts over time

---

## T-PP-2.9: Motion Blur GPU Integration

**File**: `motion_blur.py`

### Tasks

- [ ] **T-PP-2.9.1**: Allocate velocity buffer
  - Format: RG16_FLOAT
  - Per-pixel velocity

- [ ] **T-PP-2.9.2**: Allocate tile max buffer
  - Reduced resolution
  - Max velocity per tile

- [ ] **T-PP-2.9.3**: Create tile max compute pipeline
  - Reduce to per-tile max velocity

- [ ] **T-PP-2.9.4**: Create motion blur compute pipeline
  - Sample along velocity direction

- [ ] **T-PP-2.9.5**: Implement motion_blur.apply_blur() command recording

### Acceptance Criteria

- Velocity calculated per pixel
- Tile max reduces correctly
- Blur follows motion vectors

---

## T-PP-2.10: Upscaling GPU Integration

**File**: `upscaling.py`

### Tasks

- [ ] **T-PP-2.10.1**: Allocate upscaled output buffer
  - Full resolution
  - Format: RGBA16_FLOAT

- [ ] **T-PP-2.10.2**: Create bilinear upscale pipeline
  - Simple bilinear for fallback

- [ ] **T-PP-2.10.3**: Create FSR1 pipeline
  - EASU pass (edge-aware upscale)
  - RCAS pass (sharpening)

- [ ] **T-PP-2.10.4**: Implement upscale() command recording
  - Route to correct upscaler based on settings

### Acceptance Criteria

- Upscaling works at various scale factors
- FSR1 quality better than bilinear
- Output at target resolution

---

## T-PP-2.11: Frame Graph Integration Verification

### Tasks

- [ ] **T-PP-2.11.1**: Verify all effects call add_to_frame_graph()
  - Check each effect class

- [ ] **T-PP-2.11.2**: Verify PassNode read/write declarations
  - All inputs declared as read
  - All outputs declared as write

- [ ] **T-PP-2.11.3**: Verify PassFlags correct
  - COMPUTE for compute shaders
  - GRAPHICS for raster passes

- [ ] **T-PP-2.11.4**: Verify effect ordering by priority
  - Frame graph respects priority order

### Acceptance Criteria

- Frame graph builds valid execution order
- No missing resource barriers
- No resource aliasing errors

---

## Summary

| Task Group | Tasks | Priority |
|------------|-------|----------|
| T-PP-2.1 Buffer Infrastructure | 3 | Critical |
| T-PP-2.2 SSAO | 5 | High |
| T-PP-2.3 Bloom | 6 | High |
| T-PP-2.4 Tonemapping | 3 | High |
| T-PP-2.5 Color Grading | 4 | High |
| T-PP-2.6 TAA | 4 | High |
| T-PP-2.7 DOF | 5 | Medium |
| T-PP-2.8 Exposure | 4 | Medium |
| T-PP-2.9 Motion Blur | 5 | Medium |
| T-PP-2.10 Upscaling | 4 | Medium |
| T-PP-2.11 Frame Graph | 4 | High |

**Total Tasks**: 47
