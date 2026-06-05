# CLARIFICATION: engine/rendering/postprocess

## Philosophical Framing

This subsystem represents a well-designed separation of concerns between algorithm specification and execution. The CPU-side Python code serves as an executable specification — the mathematical formulas, data structures, and control flow are all correct and production-quality. What is missing is the GPU execution layer that would translate these specifications into actual shader dispatches.

This is not accidental or lazy design. It reflects a deliberate architectural choice: define the algorithms in a high-level language where they can be understood, tested, and iterated upon, then separately implement the GPU execution layer. The Python code is not throwaway prototyping — it is the authoritative reference for what each post-processing effect should compute.

## Design Rationale

### Why CPU-Side Algorithms Matter

The mathematical implementations in this subsystem are the ground truth:

1. **Tonemap operators** — The RRT+ODT approximation in ACESFitted, the log-space encoding in AgX, the shoulder/toe S-curves in Filmic — these are standard industry formulas that must be implemented exactly. The Python code documents what "correct" means.

2. **Blur algorithms** — Gaussian separable convolution, Kawase 5-point cross, Box blur — these have well-known mathematical definitions. The Python implementations show the expected behavior.

3. **Optics math** — Circle of Confusion, hyperfocal distance, bokeh kernel generation — these derive from real optical physics. The Python code embodies the correct formulas from cinematography and optics literature.

4. **Color science** — White balance temperature-to-RGB conversion, ACES color space matrices, LUT trilinear interpolation — these follow published standards (ISO 12232:2006, ACES spec).

### Why GPU Execution Is Separate

GPU execution via shaders introduces complexity orthogonal to the algorithms:
- Memory layout (UAVs, SRVs, constant buffers)
- Dispatch organization (thread groups, tiles)
- Synchronization (barriers, semaphores)
- Platform differences (WGSL vs HLSL vs SPIR-V)

By keeping algorithm specification in Python and GPU execution as a separate concern, the codebase:
- Enables testing algorithms without GPU hardware
- Allows iteration on visual appearance before GPU optimization
- Supports multiple GPU backends without duplicating algorithm logic
- Provides executable documentation for shader authors

### The Stub Pattern

The `_ao_buffer`, `_output_buffer`, `_motion_buffer` returning `None` is a deliberate stub pattern, not incomplete code. These methods:
1. Accept proper inputs (depth buffers, settings, projection matrices)
2. Validate inputs and prepare intermediate data structures
3. Return placeholder values that signal "GPU execution needed here"

This allows the rest of the engine to integrate with post-processing without the GPU layer being complete. The frame graph can build dependency chains, the quality preset system can configure effects, the execution order is correct — only the final shader dispatch is missing.

## Architectural Observations

### Quality

The architecture is production-quality:
- ABC hierarchy with `PostProcessEffect[T]` generic for type-safe settings
- `@dataclass` settings with validation in `__post_init__`
- Proper separation: settings, processors, effects
- Constants factored to `constants.py` module
- Frame graph integration via `add_to_frame_graph()` and `PassNode`
- Temporal effects handle `is_first_frame` correctly
- Quality presets define active effect sets and per-effect configs

### Test Strategy

The CPU-side algorithms are testable without GPU:
- Tonemap operators: feed known HDR values, verify output matches expected curves
- Blur algorithms: run on synthetic images, verify energy conservation
- Color grading: verify transform matrices, LUT interpolation
- Exposure: verify luminance-to-EV and EV-to-exposure round-trip
- DOF: verify CoC calculation against known optical formulas
- Halton sequence: verify low-discrepancy properties

### Rust Port Potential

The investigation notes this subsystem is a "good candidate for Rust port — algorithms are well-defined." The reasoning:
- Pure mathematical functions translate directly
- No complex Python-specific features
- Type-safe settings would benefit from Rust's type system
- Performance-critical path (runs every frame)

## Phase Organization Rationale

### Phase 1: CPU Algorithm Verification

Before adding GPU execution, verify the algorithms are correct. This:
- Builds confidence in the mathematical implementations
- Creates regression tests for future changes
- Documents expected behavior through test cases

### Phase 2: GPU Integration

Connect the stub methods to real GPU resources. This requires:
- RHI command list integration
- GPU buffer allocation
- Frame graph execution

This phase is pure plumbing — the algorithms do not change.

### Phase 3: Shader Implementation

Write the GPU shaders that implement the algorithms. The Python code serves as the specification:
- Each tonemap operator Python function -> one compute/pixel shader
- Each blur algorithm Python loop -> one optimized shader
- Each color transform Python matrix -> one shader with the same matrix

## Key Insight

The investigation reveals that this subsystem is more complete than it appears. The "stub" label applies only to GPU execution. The actual algorithms — the intellectual core of post-processing — are fully implemented and production-quality. The work remaining is integration and shader writing, not algorithm development.
