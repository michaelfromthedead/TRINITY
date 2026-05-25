# Phase 6: Temporal AA & Upscaling -- Architecture

**Phase ID**: PHASE_6 | **Tasks**: 13 (3 [x], 3 [~], 7 [-])

---

## Module Structure

### Python: `antialiasing.py` (733 lines)

```
TAASettings (dataclass)
  - jitter_sequence_length (8/16/32)
  - clamp_mode (BOX, AABB, RELAXED_AABB, VARIANCE_CLIP)
  - blend_factor, sharpen, use_ycocg, depth_threshold

FXAASettings (dataclass) -- stub
SMAASettings (dataclass) -- stub

AAEffect (PostProcessEffect)
  - execute() -> stub (calls apply methods that return input)

--- TAA ---

JitterSequence
  - halton(index, base) -> Halton sequence value
    - base=2 for X jitter, base=3 for Y jitter
    - Radical inverse via digit reversal
    - **Working**: correct Halton(2,3) generation
  - generate_sequence(length) -> list of (x, y) jitter offsets
    - **Working**: correct jitter offsets in [-0.5, 0.5] range
  - current_jitter: property returning current frame's offset
  - sequence_complete: property checking if sequence looped

TAAEffect
  - __init__(settings)
  - apply(frame, history, velocity, settings) -> stub (returns frame unchanged)
  - _apply_jitter(projection_matrix) -> jittered projection
    - Applies sub-pixel offset to projection matrix
    - Jitter = offset * 2 / resolution for NDC space offset
    - **Working**: correct jitter application
  - reproject_history(history, velocity) -> reprojected UV
    - history_uv = current_uv - velocity
    - velocity scaling correct for jitter offset
    - history_initialized: handles first frame
    - **Working**: correct reprojection logic
  - neighborhood_clamp(history, color, depth, settings) -> AABB clamp
    - _compute_aabb(3x3 neighborhood) -> min, max
    - _clip_to_aabb(color, aabb) -> clamped color
    - **Working**: correct AABB computation
  - temporal_blend(history, current, blend_factor) -> blended result
    - result = lerp(history, current, blend_factor)
    - **Working**: correct blend
  - sharpen(color, strength) -> sharpened result
    - **Working**: correct sharpening filter

--- FXAA (Stub) ---

FXAAEffect
  - apply(color) -> stub (returns color unchanged)

--- SMAA (Stub) ---

SMAAPass
  - edge_detection() -> stub
  - blend_weight() -> stub
  - neighborhood_blend() -> stub
```

### Python: `upscaling.py` (886 lines)

```
UpscalingSettings (dataclass)
  - method: UPSCALE_METHOD (BILINEAR, FSR1, CAS, FSR2, DLSS, XESS)
  - target_resolution: (width, height) tuple
  - sharpness, quality_preset

UpscalingEffect (PostProcessEffect)
  - __init__(settings)
  - execute() -> stub (calls upscale() methods that return input)
  - _detect_available_upscalers() -> auto-detection chain
    - **Working**: correct fallback chain structure

--- Spatial Upscalers ---

SpatialUpscaler (ABC)
  - upscale(input, output_resolution) -> upscaled output
  - name: str property
  - is_available: bool property

BilinearUpscaler(SpatialUpscaler)
  - upscale() -> stub (returns input, no actual resize)
  - is_available: True

FSR1Upscaler(SpatialUpscaler)
  - upscale() -> stub (returns input, no actual resize)
  - EASU(edge adaptive) -- stub
  - RCAS(robust contrast adaptive) -- stub
  - is_available: True

CASUpscaler(SpatialUpscaler)
  - upscale() -> stub (returns input)
  - is_available: True

--- Temporal Upscalers ---

TemporalUpscaler (ABC)
  - initialize() -> bool
  - evaluate(input, output) -> bool
  - get_optimal_render_resolution(output_resolution) -> (width, height)
  - is_available: bool
  - name: str

DLSSUpscaler(TemporalUpscaler)
  - initialize() -> stub (returns False, simulating no SDK)
  - evaluate() -> stub (returns False)
  - NGX method stubs
  - is_available: False (always unavailable -- no SDK DLLs)

FSR2Upscaler(TemporalUpscaler)
  - initialize() -> stub (returns False)
  - evaluate() -> stub (returns False)
  - FidelityFX API stubs
  - is_available: False

XeSSUpscaler(TemporalUpscaler)
  - initialize() -> stub (returns False)
  - evaluate() -> stub (returns False)
  - DP4a intrinsics stubs
  - is_available: False
```

### Rust: `post_process.rs`

```
create_taa_pass(index, input, history, output) -> IrPass
  - PassType::Compute, name="taa"
  - DispatchSource::Direct { 1, 1, 1 }
  - reads: [input, history]
  - writes: [output, history]  (history updated in place)
  - tags: ["post-process", "taa"]
```

---

## Data Flow

```
Color Grading Output
  |
  v
[Jitter: Halton(2,3)] -> sub-pixel offset applied to projection matrix
  |
  v
[TAA Reprojection] -- history UV = current UV - velocity
  |
  v
[TAA Neighborhood Clamp] -- AABB clip (STUB)
  |
  v
[TAA Temporal Blend] -- lerp(history, current, blend)
  |
  v
[TAA Sharpen] -- post-AA sharpening
  |
  v
[AA Mode Check]
  |-- SMAA (STUB) / FXAA (STUB) -- optional second AA pass
  |
  v
[Spatial / Temporal Upscale]
  |-- Auto-detect: DLSS -> XeSS -> FSR2 -> (FSR1/CAS/Bilinear fallback)
  |
  v
LDR Output (final)
```

**Key observation**: TAA has the most complete structural implementation -- Halton jitter works, reprojection logic works, AABB clamp works, temporal blend works. However, all TAA pixel-modification methods return input unchanged. The upscaling subsystem is structurally correct with proper ABC hierarchy, fallback chain, and SDK wrapper stubs.

---

## What Exists ([x] = 3)

| Task | Component | State |
|------|-----------|-------|
| T-PP-6.1 | Halton(2,3) jitter | Working: correct Halton sequence, configurable length, applied to projection matrix |
| T-PP-6.1b | Configurable jitter length | Working: 8/16/32 options, quality presets set per level |
| T-PP-6.5 | UpscalerPlugin ABC | Complete: SpatialUpscaler + TemporalUpscaler ABCs with all methods |

## What Is Partial ([~] = 3)

| Task | Component | Gap |
|------|-----------|-----|
| T-PP-6.1a | History reprojection | reproject_history() logic correct, apply() returns input unchanged |
| T-PP-6.2 | Neighborhood clamp + blend | AABB computation correct, apply() returns input, no variance clipping |
| T-PP-6.5a | SDK wrappers + auto-detection | DLSS/FSR2/XeSS wrappers structural, all evaluate() stubs return False/no-op |

## What Is Missing ([-] = 7)

| Task | Component | Reason |
|------|-----------|--------|
| T-PP-6.2a | YCoCg clipping + quality clamp | YCoCg not implemented. Variance clipping (Salvi 2012) not implemented. |
| T-PP-6.2b | Disocclusion detection | depth_threshold setting exists, no code reads or uses it |
| T-PP-6.3 | TSR Lanczos upsampling | TSR does not exist anywhere in codebase |
| T-PP-6.3a | TSR input scale | No TSR implementation |
| T-PP-6.3b | TSR exposure-weighted accumulation | No TSR implementation |
| T-PP-6.4 | TSR adaptive sharpening | No TSR implementation. CAS exists but is stub returning input. |
| T-PP-6.6 | Phase 6 integration tests | None. No jitter/taa/upscaler verification. |

---

## Halton Jitter Detail (Working)

```python
def halton(index, base):
    """Halton sequence value for given index and base."""
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result

# Halton(2,3) for jitter:
# X: halton(frame_index, 2)  -- base 2 (binary digit reversal)
# Y: halton(frame_index, 3)  -- base 3 (ternary digit reversal)
# Offset range: [-0.5, 0.5] after centering

def generate_sequence(length):
    return [(halton(i, 2) - 0.5, halton(i, 3) - 0.5) for i in range(length)]
```

## TAA Reprojection Detail (Working, returns input)

```python
def reproject_history(self, history, velocity):
    """Reproject history buffer using velocity buffer."""
    if not self.history_initialized:
        self.history_initialized = True
        return history  # First frame: use current frame
    
    # Reproject: history UV = current UV - velocity
    # velocity is motion vector in UV space from current to previous
    for y in range(height):
        for x in range(width):
            uv = vec2(x/width, y/height)
            prev_uv = uv - velocity[y, x]  # motion vector
            # Sample history at prev_uv
            reprojected[y, x] = sample(history, prev_uv)
    
    return reprojected
```

## Auto-Detection Chain (Working Structure)

```python
def _detect_available_upscalers(self):
    upscalers = []
    
    # Try temporal upscalers (best quality first)
    d = DLSSUpscaler()
    if d.initialize():
        upscalers.append(d)   # DLSS > XeSS > FSR2
    
    x = XeSSUpscaler()
    if x.initialize():
        upscalers.append(x)
    
    f2 = FSR2Upscaler()
    if f2.initialize():
        upscalers.append(f2)
    
    # Fallback to spatial upscalers
    f1 = FSR1Upscaler()
    upscalers.append(f1)      # Always available (CPU stub)
    
    c = CASUpscaler()
    upscalers.append(c)
    
    b = BilinearUpscaler()
    upscalers.append(b)       # Last resort
    
    return upscalers
# NOTE: All initialize() calls return False -- always falls through to FSR1
```

---

## Rust Frame Graph Contract

```rust
create_taa_pass(index, input, history, output) -> IrPass
  // Dispatch: 1x1x1 (placeholder -- must override for full-resolution)
  // Access: reads input + history, writes output + history
  //         history is both read and written (read-modify-write)
  // Lifetime: history is Transient (per-chain), output is the LDR final target
```

## TAA Pipeline Pass Plan (Planned, Not Implemented)

```
Frame N:
  Pass 1: Apply Halton jitter to projection matrix (CPU, done in Python)
  Pass 2: Compute motion vectors (velocity buffer) -- from G2 GPU-Driven
  Pass 3: Reproject history -- sample history buffer at prev UV
  Pass 4: Neighborhood clamp -- AABB clip history sample
  Pass 5: Temporal blend -- lerp(history, current, blend_factor)
  Pass 6: Sharpen -- post-AA sharpening filter
  Pass 7: Write history back for frame N+1

Frame N+1:
  Repeat with new jitter offset
```
