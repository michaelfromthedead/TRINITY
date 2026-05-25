# Phase 1: HDR Pipeline & Core Tonemapping -- Architecture

**Phase ID**: PHASE_1 | **Tasks**: 16 (8 [x], 3 [~], 5 [-])

---

## Module Structure

### Python: `tonemapping.py` (694 lines)

```
TonemapSettings (dataclass)
  - operator: TonemapOperator enum
  - exposure_multiplier, white_point, output_gamma
  
TonemapOperator (enum)
  - REINHARD, REINHARD_EXTENDED, ACES_FULL, ACES_FITTED
  - AGX, FILMIC, NEUTRAL, CUSTOM_CURVE

TonemappingEffect (PostProcessEffect)
  - __init__(settings, operator)
  - execute() -> stub (empty)
  - _apply_operator(color) -> dispatch to operator classes

--- Operator classes (CPU reference implementations) ---

Reinhard:  color / (color + 1.0)
ReinhardExtended:  x * (1 + x/white^2) / (1 + x)
ACES:  sRGB->ACEScg matrix * fitted curve * ACEScg->sRGB
ACESFitted:  (x*(a*x+b))/(x*(c*x+d)+e), a=2.51, b=0.03, c=2.43, d=0.59, e=0.14
AgX:  log-encoded gamut mapping curve
Filmic (Hable):  (x*(a*x+b))/(x*(c*x+d)+e) with Hable's coefficients
Neutral:  balanced S-curve
CustomCurve:  user-provided lookup table
```

### Python: `exposure.py` (590 lines)

```
ExposureSettings (dataclass)
  - mode: ExposureMode (MANUAL, AUTO_AVERAGE, HISTOGRAM)
  - manual_ev, metering_mode, histogram_bins, eye_adaptation_speed

ManualExposure
  - calculate(luminance) -> fixed EV value
AutoAverage
  - calculate(luminance) -> log2(average_luminance)
  - needs luminance buffer or CPU compute input
HistogramExposure
  - calculate(histogram) -> percentile-based weighted average
  - bins: numpy array, bucket calculation
EyeAdaptation
  - update(current_exposure, dt) -> EMA-smoothed exposure
  - formula: 1 - exp(-speed * dt)
  - asymmetric: speed_up=3.0s, speed_down=1.0s

ExposureEffect (PostProcessEffect)
  - execute() -> stub (empty)
```

### Rust: `post_process.rs`

```
create_tonemap_pass(index, input, output) -> IrPass
  - PassType::Compute, name="tonemap"
  - DispatchSource::Direct { 1, 1, 1 }
  - reads: [input], writes: [output]
  - tags: ["post-process", "tonemap"]

create_post_process_chain() -> (Vec<IrPass>, Vec<IrResource>)
  - Pass 0: Tonemap (hdr_input -> tonemap_output)
  - Transient resources: tonemap_output (RGBA16F)
```

---

## Data Flow

```
HDR Scene Color (RGBA16F)
  |
  v
[ExposureEffect] -> exposure_multiplier (CPU uniform)
  |
  v
[TonemappingEffect] -> _apply_operator()
  |                   |-> Reinhard: color/(color+1)
  |                   |-> ACES: matrix * curve * matrix
  |                   |-> AgX: log-encoding curve
  |                   |-> Filmic: Hable curve
  |
  v
LDR Output (RGBA8_UNORM)
```

**Key observation**: Exposure and tonemapping are CPU-only. The Rust IR pass has placeholder dispatch (1x1x1). No GPU implementation exists.

---

## What Exists ([x] = 8)

| Task | Component | State |
|------|-----------|-------|
| T-PP-1.1 | PostProcessStack | Complete: 1,776 lines, quality presets, executor, volumes |
| T-PP-1.1a | EffectPriority IntEnum | Complete: 9 positions with quality toggle wiring |
| T-PP-1.1b | Quality presets | Complete: 4 presets, intermediate target manager |
| T-PP-1.2 | HDR framebuffer resources | Complete: format configs, Rust transient resources |
| T-PP-1.3 | ACES filmic tonemap | Complete: ACESFitted + ACES full, CPU Python |
| T-PP-1.3a | TonemapSettings config | Complete: operator select, white point, exposure |
| T-PP-1.4 | 8 tonemap operators | Complete: all 8 operators as CPU Python classes |
| T-PP-1.5a | Eye adaptation EMA | Complete: asymmetric speed, exponential smoothing |

## What Is Partial ([~] = 3)

| Task | Component | Gap |
|------|-----------|-----|
| T-PP-1.2a | Transient buffer aliasing | Ping-pong pool (size 2) exists, no phase-level aliasing groups |
| T-PP-1.2b | Resource lifetime analysis | Transient lifetime set, no explicit non-overlap analysis |
| T-PP-1.5b | Quality bin/cull variation | Presets define bins up to 512, but constants cap at 256 |

## What Is Missing ([-] = 5)

| Task | Component | Reason |
|------|-----------|--------|
| T-PP-1.3 | ACES WGSL shader | No shader exists |
| T-PP-1.4a | Operator reference validation | No reference image validation |
| T-PP-1.5 | Auto-exposure histogram compute | CPU only, no compute shader |
| T-PP-1.6 | Vignette | No implementation anywhere |
| T-PP-1.7 / 1.7a | Integration tests | No integration tests, no budget validation |

---

## Interfaces

### Rust-to-Shader Contract (planned, not implemented)

```rust
// Uniform buffer for tonemapping parameters
struct TonemapParams {
    exposure_multiplier: f32,
    white_point: f32,
    _pad: [f32; 2],
    operator: u32,  // TonemapOperator enum
}

// Shader entry: @compute @workgroup_size(16, 16)
// Input: hdr_texture @binding(0)
// Output: ldr_texture @binding(1)
// Uniform: tonemap_params @binding(2)
```

### Rust Frame Graph Contract (implemented)

```rust
create_tonemap_pass(index, hdr_input, ldr_output) -> IrPass
  // Dispatch: 1x1x1 (placeholder, must override at recording time)
  // Access: reads hdr_input, writes ldr_output
  // Lifetime: Transient intermediate resources
```
