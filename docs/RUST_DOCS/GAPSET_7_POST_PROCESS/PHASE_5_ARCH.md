# Phase 5: Color Grading -- Architecture

**Phase ID**: PHASE_5 | **Tasks**: 7 (2 [x], 1 [~], 4 [-])

---

## Module Structure

### Python: `color_grading.py` (681 lines)

```
ColorGradingSettings (dataclass)
  - lut_path, lut_size (16/32/48/64)
  - white_balance_temperature, white_balance_tint
  - contrast, saturation, vibrance
  - lift, gamma, gain (RGB triplets)
  - channel_mixer (4x4 matrix), split_toning (shadows/highlights)

ColorGradingEffect (PostProcessEffect)
  - execute() -> stub (empty, no-op)

--- Core Components ---

LUT3D
  - data: numpy array of shape (size, size, size, 3)
  - size: 16, 32, 48, or 64
  
  - from_cube(filepath) -> LUT3D
    - Parses: TITLE, LUT_3D_SIZE, DOMAIN_MIN, DOMAIN_MAX
    - Reads N^3 rows of floating-point RGB values
    - Reshapes into 3D grid: [R][G][B] -> RGB
    - **Working**: correct parser implementation
    - Validation: checks row count matches size^3, skips comments (#)
    
  - from_identity(size) -> LUT3D
    - Generates identity grid: output[R][G][B] = (R, G, B) / (size-1)
    - **Working**: correct identity generation
    
  - sample(r, g, b) -> (r', g', b')
    - Trilinear interpolation between 8 nearest grid points
    - Find floor/ceil indices, fractional weights
    - Interpolate R-axis, then G-axis, then B-axis
    - **Working**: correct trilinear CPU interpolation
    
  - cache: dict
    - get_cached(filepath) -> LUT3D (loads once, reuses)
    - **Working**: file-based caching
    
  - apply(color) -> transformed color via LUT sample
    - **Working**: pixel-level LUT application

--- Pipeline Components ---

WhiteBalance
  - apply(color, temperature, tint) -> white-balanced color
  - Von Kries transform with Bradford chromatic adaptation
  - **Working**: CPU color temperature correction

LiftGammaGain
  - apply(color, lift, gamma, gain) -> LGG-transformed color
  - Formula: color * gain + lift, then pow(1/gamma)
  - **Working**: CPU LGG pipeline

ContrastSettings
  - apply(color, contrast, pivot=0.5) -> contrast-adjusted color
  - **Working**: CPU contrast

SaturationSettings
  - apply(color, saturation) -> saturation-adjusted color
  - **Working**: CPU saturation

ChannelMixer
  - apply(color, matrix) -> channel-mixed color
  - 4x4 matrix multiplication (RGBA)
  - **Working**: CPU channel mixing

SplitToning
  - apply(color, shadows, highlights, balance) -> split-toned color
  - **Working**: CPU split toning

ColorGradingStack
  - process(color, settings) -> full pipeline:
    WhiteBalance -> ChannelMixer -> LiftGammaGain ->
    Contrast -> Saturation/Vibrance -> SplitToning -> LUT
  - **Working**: complete CPU pipeline
```

---

## Data Flow

```
LDR Output (post-tonemapping, display-referred sRGB)
  |
  v
[WhiteBalance] -- Von Kries Bradford chromatic adaptation
  |  temp_offset = temperature - D65 (6500K)
  |  tint_offset = tint * 0.01
  |  Apply diagonal RGB scaling in LMS cone space
  |
  v
[ChannelMixer] -- 4x4 matrix per-channel mixing
  |  RGBA = matrix * RGBA
  |
  v
[LiftGammaGain] -- shadows/midtones/highlights
  |  color = color * gain + lift
  |  color = pow(color, 1/gamma)
  |
  v
[Contrast] -- S-curve around pivot
  |  color = (color - pivot) * contrast + pivot
  |
  v
[Saturation / Vibrance]
  |  sat: adjust channel spread from luminance
  |  vib: non-linear boost in low-sat regions
  |
  v
[SplitToning] -- shadow/highlight color grading
  |  shadows weighted by (1 - luminance)
  |  highlights weighted by luminance
  |
  v
[LUT Application] -- 3D LUT with trilinear interpolation
  |  Sample 8 nearest grid points, tri-weight blend
  |
  v
To Anti-Aliasing / Next effect
```

**Key observation**: Color grading is the **most complete** post-process subsystem. The .cube parser, identity LUT, trilinear interpolation, white balance, LGG, contrast, saturation, channel mixer, and split toning all work as CPU reference implementations. `ColorGradingEffect.execute()` is empty -- the pipeline is defined in `ColorGradingStack.process()` but not wired into the effect system.

---

## What Exists ([x] = 2)

| Task | Component | State |
|------|-----------|-------|
| T-PP-5.1 | .cube LUT parser | Working: TITLE, LUT_3D_SIZE, DOMAIN_MIN/MAX, data rows, validation |
| T-PP-5.1b | Identity LUT + cache | Working: from_identity(), file-based set_cache()/get_cached() |

## What Is Partial ([~] = 1)

| Task | Component | Gap |
|------|-----------|-----|
| T-PP-5.2 | LUT application + trilinear | CPU trilinear works. ColorGradingStack pipeline works CPU. No GPU shader. No execute() wiring. |

## What Is Missing ([-] = 4)

| Task | Component | Reason |
|------|-----------|--------|
| T-PP-5.1a | 2D atlas conversion | No GPU texture layout. LUT data stays as numpy array. |
| T-PP-5.2a | Atlas layout + dither | No atlas, no dither, quality-LUT-size mapping not connected |
| T-PP-5.2b | Performance budget | No measurement infrastructure |
| T-PP-5.3 | Phase 5 integration tests | None. Most testable subsystem (parser + interpolation + pipeline) has no test coverage. |

---

## .cube Parser Detail (Working)

The parser handles the standard .cube format:

```
TITLE "My LUT"
LUT_3D_SIZE 32
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0

# Data rows (N^3 = 32^3 = 32768 rows)
0.000000 0.000000 0.000000
0.000000 0.000000 0.032258
...
```

**Parser logic:**
1. Read lines, split into tokens
2. `TITLE` captures the string
3. `LUT_3D_SIZE N` sets grid dimension
4. `DOMAIN_MIN/MAX` sets input range (default: 0-1)
5. `#` lines are skipped as comments
6. All other tokens form RGB data rows
7. Validate: total rows must equal N^3
8. Reshape: `data = rows.reshape(N, N, N, 3)`

## LUT Application Detail (Working)

```python
def sample(self, r, g, b):
    # Normalize to grid coordinates
    size = self.size
    r = r * (size - 1)
    g = g * (size - 1)
    b = b * (size - 1)
    
    # Floor/ceil indices
    ri, gi, bi = int(r), int(g), int(b)
    rf, gf, bf = r - ri, g - gi, b - bi
    
    # 8 nearest grid points
    c000 = self.data[ri,   gi,   bi]
    c100 = self.data[ri+1, gi,   bi]
    c010 = self.data[ri,   gi+1, bi]
    c110 = self.data[ri+1, gi+1, bi]
    c001 = self.data[ri,   gi,   bi+1]
    c101 = self.data[ri+1, gi,   bi+1]
    c011 = self.data[ri,   gi+1, bi+1]
    c111 = self.data[ri+1, gi+1, bi+1]
    
    # Trilinear interpolation
    # R-axis
    c00 = lerp(c000, c100, rf)
    c10 = lerp(c010, c110, rf)
    c01 = lerp(c001, c101, rf)
    c11 = lerp(c011, c111, rf)
    # G-axis
    c0 = lerp(c00, c10, gf)
    c1 = lerp(c01, c11, gf)
    # B-axis
    return lerp(c0, c1, bf)
```

---

## ColorGradingStack Pipeline (Working)

```python
def process(self, color, settings):
    color = WhiteBalance.apply(color, settings.temperature, settings.tint)
    color = ChannelMixer.apply(color, settings.channel_mixer)
    color = LiftGammaGain.apply(color, settings.lift, settings.gamma, settings.gain)
    color = Contrast.apply(color, settings.contrast)
    color = Saturation.apply(color, settings.saturation)
    color = SplitToning.apply(color, settings.shadow_color, settings.highlight_color)
    color = self.lut.sample(color[0], color[1], color[2])
    return color
```

---

## Interfaces (GPU -- Planned, Not Implemented)

The 2D atlas conversion (T-PP-5.1a) would pack the 3D LUT into a 2D texture for hardware bilinear filtering:

```
LUT size 32^3 -> Atlas size 32*32 x 32 = 1024 x 32
- N rows of NxN squares stacked vertically
- texture.Sample(uv) interpolates between adjacent grid points
- GPU does hardware trilinear via bilinear on 2D + manual third axis
```
