# Phase 2: Bloom & Lens Effects -- Architecture

**Phase ID**: PHASE_2 | **Tasks**: 12 (1 [x], 2 [~], 9 [-])

---

## Module Structure

### Python: `bloom.py` (840 lines)

```
BloomSettings (dataclass)
  - threshold, knee, radius, intensity, lens_dirt_texture
  - quality_levels: mip_count (3/5/6/8), blur_type, sample_count
  
BloomEffect (PostProcessEffect)
  - __init__(settings)
  - execute() -> loops mips calling downsample/upsample/blur (all stubs)

BloomThreshold
  - extract_bright_passes(color, threshold, knee) -> soft-knee extraction
  - Formula: max(rgb - threshold, 0) with knee smoothstep transition
  - CPU reference: works for individual pixels

BloomDownsample
  - downsample(high_res, level) -> creates low_res buffer (zero-initialized stub)
  - Quality: 3/5/6/8 mip levels per preset
  - No actual downsample filtering -- returns zero buffer

BloomBlur
  - GaussianBlur (separable, pre-calculated 5x5 weights/offsets)
  - KawaseBlur (5-point cross, 4 iterations/2 iterations per quality)
  - BoxBlur (3-tap horizontal + vertical)
  - All CPU reference implementations
  - blur() methods perform actual convolution on numpy arrays

BloomUpsample
  - upsample_and_accumulate(low_res, high_res) -> stub (returns high_res unchanged)
  - No tent filter, no composite
```

### Rust: `post_process.rs`

```
create_bloom_pass(index, input, output, width, height) -> IrPass
  - PassType::Compute, name="bloom"
  - DispatchSource::Direct { width/8, height/8, 1 }
  - reads: [input], writes: [output]
  - tags: ["post-process", "bloom"]

Used in chain: tonemap_output -> bloom -> bloom_output -> TAA
```

---

## Data Flow

```
HDR Scene Color (after exposure)
  |
  v
[BloomThreshold] -- soft knee extraction
  |
  v
[Downsample Pyramid] -- 5 levels (High), 7 levels (Ultra)
  |  Level 0 -> Level 1 -> Level 2 -> Level 3 -> Level 4
  |
  v
[Gaussian Blur per level] -- separable 5x5 (High+Ultra)
  |
  v
[Upsample + Accumulate] -- bottom-up blend (STUB)
  |  Level 4 -> Level 3 -> Level 2 -> Level 1 -> Level 0
  |
  v
[Composite] -- hdr_scene + bloom_intensity * bloom (STUB)
  |
  v
To DOF / Next effect
```

**Key observation**: Bloom pipeline architecture is structurally complete. Threshold algorithm works (CPU). Blur kernels are implemented (CPU). Downsample/upsample/composite are stubs that return zero/input unchanged.

---

## What Exists ([x] = 1)

| Task | Component | State |
|------|-----------|-------|
| T-PP-2.1 | Bloom bright-pass + downsample | BloomThreshold works (CPU). Rust IR pass builder exists. Downsample is stub. |

## What Is Partial ([~] = 2)

| Task | Component | Gap |
|------|-----------|-----|
| T-PP-2.1a | Separable 5x5 Gaussian | CPU reference convolution works. Not GPU-ready. No WGSL shader. |
| T-PP-2.2 | Upsample-and-blur | BloomUpsample exists but returns input unchanged. Kawase/Box blur CPU impls exist. |

## What Is Missing ([-] = 9)

| Task | Component | Reason |
|------|-----------|--------|
| T-PP-2.2a | Upsample composite | No composite: hdr_scene + bloom_intensity * upsampled_bloom |
| T-PP-2.3 | Lens flare | No implementation anywhere |
| T-PP-2.3a | Anamorphic streaks | No implementation |
| T-PP-2.3b | Quality levels for lens flare | No implementation |
| T-PP-2.4 | Chromatic aberration | No implementation |
| T-PP-2.4a | Anamorphic distortion ratio | No implementation |
| T-PP-2.4b | CA quality levels | No implementation |
| T-PP-2.5 | Phase 2 integration tests | None |
| T-PP-2.5a | CA offset validation | None |

---

## Missing Effects Detail

### Lens Flare (T-PP-2.3)
- No module, no class, no shader
- `BloomSettings.lens_dirt` field exists but is a texture overlay, not lens flare ghosts
- Would need: ghost generation, halo, anamorphic streaks, chromatic shift

### Chromatic Aberration (T-PP-2.4)
- No module, no class, no shader
- Referenced only in camera effects (lens distortion) and XR compositor -- unrelated
- Would need: per-channel radial offset, fringe suppression, quality levels

---

## Interfaces

### Rust Frame Graph Contract

```rust
create_bloom_pass(index, input, output, width, height) -> IrPass
  // Dispatch: width/8 x height/8 (derived from resolution)
  // Access: reads input, writes output
  // Lifetime: Transient intermediate
```

### Bloom Pipeline Pass Plan (planned, not implemented)

```
Pass 1: Bright-pass extraction     -> R16G16B16A16_FLOAT  (full res)
Pass 2-6: Downsample 5 levels      -> R16G16B16A16_FLOAT  (1/2, 1/4, 1/8, 1/16, 1/32 res)
Pass 7-11: Blur per level          -> R16G16B16A16_FLOAT  (same res per level)
Pass 12-16: Upsample + accumulate  -> R16G16B16A16_FLOAT  (bottom-up)
Pass 17: Composite                 -> R16G16B16A16_FLOAT  (full res)
```

Currently only 1 Rust pass exists (bloom) with placeholder dispatch. Would need ~17 passes for a full GPU bloom pipeline.
