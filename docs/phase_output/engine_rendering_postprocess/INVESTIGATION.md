# engine/rendering/postprocess Investigation

**Lines**: 8,861
**Classification**: PARTIAL (Real architecture with stub GPU execution)

## Summary

This subsystem is a comprehensive post-processing framework with REAL algorithmic implementations at the CPU/Python level, but GPU execution paths return placeholder values. The architecture is production-quality with correct mathematical formulas, proper effect chaining, quality presets, and frame graph integration. However, actual GPU commands are not recorded -- execute methods prepare data structures but return `None` or placeholder buffers.

## File Analysis

| File | Lines | Classification | Key Components |
|------|-------|----------------|----------------|
| postprocess_stack.py | 1,776 | REAL | PostProcessStack, PostProcessEffect (ABC), IntermediateTargetManager, PostProcessVolume, QualityPresets, ExecutionFlags |
| upscaling.py | 886 | PARTIAL | FSR1/FSR2/DLSS/XeSS abstractions, UpscalingEffect -- class structure complete but upscale() returns `_output_buffer` (None) |
| bloom.py | 840 | REAL | BloomThreshold (soft-knee math), BloomDownsample (mip chain), BloomBlur (Gaussian/Kawase/Box REAL algorithms), BloomUpsample |
| antialiasing.py | 733 | PARTIAL | JitterSequence (REAL Halton), FXAA/SMAA/TAA class structure -- TAA.apply() has placeholder history blend |
| tonemapping.py | 694 | REAL | Reinhard, ReinhardExtended, ACES, ACESFitted, AgX, Filmic, CustomCurve -- all tonemap_value() methods have REAL math |
| color_grading.py | 681 | REAL | WhiteBalance, LiftGammaGain, ContrastSettings, SaturationSettings, LUT3D (trilinear interpolation), ColorGradingStack.apply() |
| ambient_occlusion.py | 673 | PARTIAL | SSAOKernel (REAL hemisphere generation), SSAO/HBAO/GTAO class structure, BilateralFilter -- calculate() returns placeholder |
| dof.py | 652 | PARTIAL | CircleOfConfusion (REAL optics math), BokehShape (REAL kernel generation), NearFieldDOF/FarFieldDOF -- blur() returns placeholder |
| exposure.py | 590 | REAL | luminance_to_ev()/ev_to_exposure() (REAL), ManualExposure, AutoExposure (metering kernels), HistogramExposure, EyeAdaptation (temporal smoothing) |
| motion_blur.py | 550 | PARTIAL | CameraMotionBlur (matrix multiply REAL), TileMaxVelocity structure, ObjectMotionBlur -- apply_blur() returns placeholder |

## Key Findings

### What This Subsystem ACTUALLY Implements

1. **PostProcessStack Framework (REAL)**
   - Effect ordering by priority
   - Quality preset system (Low/Medium/High/Ultra)
   - Execution flags (SKIP_IF_DISABLED, SKIP_ON_FIRST_FRAME, FORCE_ASYNC)
   - Ping-pong intermediate target management
   - PostProcessVolume with spatial blending (box/sphere shapes)
   - Frame graph integration via `add_to_frame_graph()`

2. **Tonemap Operators (REAL)**
   - 8 operators with mathematically correct implementations
   - ACES color space matrices (sRGB -> ACEScg)
   - AgX log-space encoding with look transforms
   - Filmic S-curve with shoulder/toe controls
   - Custom artist-defined curves with Hermite interpolation

3. **Bloom Pipeline (REAL)**
   - Soft threshold with knee parameters
   - Mip chain generation (up to 8 levels)
   - Three blur algorithms: Gaussian (separable), Kawase (5-point cross), Box
   - Per-mip intensity and scatter settings

4. **Color Grading (REAL)**
   - White balance temperature/tint to RGB conversion
   - Lift/Gamma/Gain per-channel adjustment
   - Shadow/Midtone/Highlight separation with blend weights
   - Saturation with vibrance (protects already-saturated colors)
   - 3D LUT loading (.cube format) with trilinear interpolation
   - Channel mixer matrix

5. **Exposure Control (REAL)**
   - Luminance-to-EV conversion (ISO 12232:2006)
   - Center-weighted, spot, and matrix metering kernels
   - Histogram-based percentile exposure
   - Eye adaptation temporal smoothing with asymmetric speeds

6. **DOF Optics (REAL)**
   - Circle of Confusion calculation from aperture/focal length/sensor size
   - Hyperfocal distance computation
   - Bokeh kernel generation (disk, polygon with blade curvature, anamorphic)
   - AutoFocus with smooth transition

7. **TAA/AA (PARTIAL)**
   - Halton jitter sequence generation (REAL)
   - Projection matrix jitter application (REAL)
   - History reprojection/clamping (STUB -- returns placeholder)

8. **AO/Motion Blur (PARTIAL)**
   - SSAO hemisphere kernel generation (REAL)
   - HBAO direction generation (REAL)
   - Bilateral filter weight formula (REAL)
   - Tile-based velocity structure (REAL)
   - Actual sampling and blur (STUB)

### What Is STUB

All `execute()` methods that would record GPU commands return placeholder values:
- `_ao_buffer`, `_output_buffer`, `_motion_buffer` are `None`
- GPU shader dispatch is not implemented
- No RHI command list recording

## Evidence

### REAL: Tonemapping Math (tonemapping.py:296-307)
```python
def _rrt_odt(self, x: float) -> float:
    """Apply RRT + ODT approximation."""
    a = x * (x + 0.0245786) - 0.000090537
    b = x * (0.983729 * x + 0.4329510) + 0.238081
    return a / b if b != 0 else 0.0
```

### REAL: Bloom Gaussian Blur (bloom.py:439-507)
```python
def _gaussian_blur(self, source, target, iterations, width, height):
    # Full separable convolution implementation
    for _ in range(iterations):
        # Horizontal pass: convolve each row
        for y in range(height):
            for x in range(width):
                # ... actual convolution math
        # Vertical pass: convolve each column
        # ... actual convolution math
    return result
```

### REAL: Circle of Confusion (dof.py:166-208)
```python
def calculate(self, depth: float, image_width: int) -> float:
    # Hyperfocal distance
    hyperfocal = self.focal_length + (self.focal_length ** 2) / (self.aperture * coc_mm)
    # CoC calculation with magnification
    magnification = focal_m / (focus_m - focal_m)
    coc_m = abs(depth - focus_m) * magnification * (self.focal_length / self.aperture) / depth / 1000.0
    # Convert to pixels
    return min(coc_pixels, self.max_coc_radius)
```

### REAL: Halton Jitter Sequence (antialiasing.py:190-225)
```python
def _generate_halton(self, count: int) -> List[Tuple[float, float]]:
    samples = []
    for i in range(count):
        x = self._halton(i + 1, 2) - 0.5  # Base 2
        y = self._halton(i + 1, 3) - 0.5  # Base 3
        samples.append((x, y))
    return samples

def _halton(self, index: int, base: int) -> float:
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i = i // base
        f /= base
    return result
```

### STUB: AO Calculate (ambient_occlusion.py:192-210)
```python
def calculate(self, depth_buffer, normal_buffer, settings, projection):
    """Calculate SSAO."""
    return self._ao_buffer  # Returns None
```

### STUB: Upscaler Execute (upscaling.py:256-270)
```python
def upscale(self, color_buffer, settings):
    """Upscale using bilinear filtering."""
    return self._output_buffer  # Returns None
```

## Architecture Quality

- Well-structured ABC hierarchy with `PostProcessEffect[T]` generic
- Settings use `@dataclass` with validation in `__post_init__`
- Proper separation: settings, processors, effects
- Constants factored to `constants.py` module
- Frame graph integration via `add_to_frame_graph()` and `PassNode`
- Temporal effects handle `is_first_frame` correctly
- Quality presets define active effect sets and per-effect configs

## Dependencies

- `engine.rendering.framegraph.frame_graph.FrameGraph`
- `engine.rendering.framegraph.pass_node.PassNode, PassFlags`
- `engine.rendering.framegraph.resource_manager.ResourceFormat`
- `./constants` module for magic numbers

## Recommendations

1. GPU execution is completely missing -- would need RHI command list integration
2. Shader sources not present (would need WGSL/HLSL)
3. Test coverage could verify CPU-side math (tonemapping, blur, LUT)
4. Good candidate for Rust port -- algorithms are well-defined
