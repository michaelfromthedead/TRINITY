# PHASE 6 TODO: Rendering Pipeline

## Overview

Phase 6 integrates and validates the XR rendering pipeline. The implementation is production-ready; this phase focuses on GPU integration, performance profiling, and quality tuning.

## Tasks

### T-XR-6.1: Stereo Renderer Integration

**Priority**: Critical
**Effort**: Large (24 hours)
**Dependencies**: T-XR-1.1 (OpenXR bindings), RHI layer

**Description**: Integrate stereo rendering with RHI and validate on hardware.

**Subtasks**:
- [ ] T-XR-6.1.1: Implement multi-view rendering path
- [ ] T-XR-6.1.2: Implement instanced rendering fallback
- [ ] T-XR-6.1.3: Implement sequential rendering fallback
- [ ] T-XR-6.1.4: Test view matrix accuracy per eye
- [ ] T-XR-6.1.5: Test projection matrix with IPD
- [ ] T-XR-6.1.6: Profile draw call reduction per method

**Acceptance Criteria**:
- [ ] Multi-view reduces draw calls by 40%+
- [ ] Instanced reduces draw calls by 50%+
- [ ] View separation matches physical IPD
- [ ] Factory selects best available method

**Files**:
- `engine/xr/rendering/stereo.py`

---

### T-XR-6.2: Foveated Rendering Integration

**Priority**: Critical
**Effort**: Large (24 hours)
**Dependencies**: T-XR-2.4 (eye tracking), RHI VRS support

**Description**: Integrate foveated rendering with VRS hardware.

**Subtasks**:
- [ ] T-XR-6.2.1: Generate VRS image from foveation config
- [ ] T-XR-6.2.2: Upload VRS image to GPU
- [ ] T-XR-6.2.3: Test fixed foveation at screen center
- [ ] T-XR-6.2.4: Test dynamic foveation following gaze
- [ ] T-XR-6.2.5: Measure pixel savings per mode
- [ ] T-XR-6.2.6: Tune region radii for invisibility

**Acceptance Criteria**:
- [ ] VRS image applied per frame
- [ ] Dynamic foveation tracks gaze smoothly
- [ ] Pixel savings >30% with fixed foveation
- [ ] Foveation boundaries not noticeable

**Files**:
- `engine/xr/rendering/foveated.py`

---

### T-XR-6.3: Compositor Layer Rendering

**Priority**: High
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-6.1

**Description**: Validate compositor layer blending and ordering.

**Subtasks**:
- [ ] T-XR-6.3.1: Test projection layer rendering
- [ ] T-XR-6.3.2: Test quad layer positioning and sizing
- [ ] T-XR-6.3.3: Test cylinder layer (curved UI)
- [ ] T-XR-6.3.4: Test layer priority ordering
- [ ] T-XR-6.3.5: Test blend mode correctness (alpha, additive)
- [ ] T-XR-6.3.6: Test head-locked vs world-locked layers

**Acceptance Criteria**:
- [ ] Quad layers render at correct world position
- [ ] Cylinder layers curve correctly
- [ ] Layer ordering matches priority
- [ ] Blend modes produce expected results

**Files**:
- `engine/xr/rendering/compositor.py`

---

### T-XR-6.4: Reprojection Validation

**Priority**: High
**Effort**: Large (24 hours)
**Dependencies**: T-XR-6.1

**Description**: Validate reprojection modes handle frame drops.

**Subtasks**:
- [ ] T-XR-6.4.1: Test ATW rotation correction
- [ ] T-XR-6.4.2: Test ATW rotation clamping (prevent artifacts)
- [ ] T-XR-6.4.3: Test ASW motion vector analysis
- [ ] T-XR-6.4.4: Test ASW frame synthesis
- [ ] T-XR-6.4.5: Test hybrid mode blending
- [ ] T-XR-6.4.6: Measure reprojection latency

**Acceptance Criteria**:
- [ ] ATW corrects head rotation during dropped frame
- [ ] ATW does not cause warping artifacts
- [ ] ASW synthesizes convincing frames
- [ ] Reprojection completes in <2ms

**Files**:
- `engine/xr/rendering/reprojection.py`

---

### T-XR-6.5: Hidden Area Mesh Integration

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-6.1, device mesh data

**Description**: Integrate hidden area mesh with render pipeline.

**Subtasks**:
- [ ] T-XR-6.5.1: Load mesh data from device/runtime
- [ ] T-XR-6.5.2: Test stencil-based masking
- [ ] T-XR-6.5.3: Test depth-based masking
- [ ] T-XR-6.5.4: Test combined masking
- [ ] T-XR-6.5.5: Measure pixel savings per device
- [ ] T-XR-6.5.6: Validate visible area calculation

**Acceptance Criteria**:
- [ ] Hidden pixels not shaded (early-Z or stencil reject)
- [ ] Visible area ratio matches device spec
- [ ] 10-15% pixel reduction achieved

**Files**:
- `engine/xr/rendering/hidden_area.py`

---

### T-XR-6.6: Quality Preset Tuning

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-6.1 through T-XR-6.5

**Description**: Tune quality presets for different hardware tiers.

**Subtasks**:
- [ ] T-XR-6.6.1: Test low_quality preset (mobile VR)
- [ ] T-XR-6.6.2: Test medium_quality preset (typical desktop)
- [ ] T-XR-6.6.3: Test high_quality preset (high-end desktop)
- [ ] T-XR-6.6.4: Test ultra_quality preset (maximum quality)
- [ ] T-XR-6.6.5: Profile frame time per preset per device
- [ ] T-XR-6.6.6: Auto-quality scaling based on frame time

**Acceptance Criteria**:
- [ ] Each preset maintains target frame time on appropriate hardware
- [ ] Quality difference visible between presets
- [ ] Auto-scaling adjusts quality to maintain 90Hz

**Files**:
- `engine/xr/rendering/__init__.py`

---

### T-XR-6.7: Frame Timing Analysis

**Priority**: High
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-6.1 through T-XR-6.4

**Description**: Profile and optimize frame timing.

**Subtasks**:
- [ ] T-XR-6.7.1: Instrument frame time breakdown
- [ ] T-XR-6.7.2: Identify bottlenecks per pipeline stage
- [ ] T-XR-6.7.3: Optimize slow stages
- [ ] T-XR-6.7.4: Test late latch timing
- [ ] T-XR-6.7.5: Test compositor submit timing
- [ ] T-XR-6.7.6: Measure motion-to-photon latency

**Acceptance Criteria**:
- [ ] Frame time breakdown available per stage
- [ ] Total frame time <11ms at 90Hz
- [ ] Motion-to-photon latency <20ms

**Files**:
- `engine/xr/rendering/__init__.py`

---

### T-XR-6.8: Passthrough Integration

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: T-XR-6.3

**Description**: Integrate passthrough rendering for AR/MR modes.

**Subtasks**:
- [ ] T-XR-6.8.1: Request passthrough layer from runtime
- [ ] T-XR-6.8.2: Configure passthrough as background layer
- [ ] T-XR-6.8.3: Test passthrough opacity control
- [ ] T-XR-6.8.4: Test passthrough color correction
- [ ] T-XR-6.8.5: Test virtual object occlusion
- [ ] T-XR-6.8.6: Measure passthrough latency

**Acceptance Criteria**:
- [ ] Passthrough visible behind virtual content
- [ ] Opacity adjustable 0-100%
- [ ] Virtual objects occlude passthrough correctly
- [ ] Passthrough latency acceptable (<30ms)

**Files**:
- `engine/xr/rendering/compositor.py`

---

### T-XR-6.9: Rendering Unit Tests

**Priority**: Medium
**Effort**: Medium (16 hours)
**Dependencies**: None

**Description**: Add unit tests for rendering calculations.

**Subtasks**:
- [ ] T-XR-6.9.1: Test view matrix calculation
- [ ] T-XR-6.9.2: Test projection matrix calculation
- [ ] T-XR-6.9.3: Test VRS region assignment
- [ ] T-XR-6.9.4: Test pose prediction math
- [ ] T-XR-6.9.5: Test layer ordering
- [ ] T-XR-6.9.6: Test hidden area containment

**Acceptance Criteria**:
- [ ] >85% code coverage on rendering core
- [ ] Matrix math verified against known-good results
- [ ] Region assignment verified geometrically

**Files**:
- `engine/xr/rendering/tests/` (new directory)

---

## Phase 6 Completion Criteria

- [ ] Stereo rendering integrated with RHI
- [ ] Foveated rendering reduces GPU load
- [ ] Compositor layers render correctly
- [ ] Reprojection handles dropped frames
- [ ] Hidden area mesh saves pixels
- [ ] Quality presets tuned for hardware tiers
- [ ] Frame timing meets 90Hz target
- [ ] Passthrough works for AR/MR modes
- [ ] Unit tests cover rendering calculations

## Estimated Total Effort

| Task | Effort |
|------|--------|
| T-XR-6.1: Stereo Renderer | 24 hours |
| T-XR-6.2: Foveated Rendering | 24 hours |
| T-XR-6.3: Compositor Layers | 16 hours |
| T-XR-6.4: Reprojection | 24 hours |
| T-XR-6.5: Hidden Area Mesh | 16 hours |
| T-XR-6.6: Quality Presets | 16 hours |
| T-XR-6.7: Frame Timing | 16 hours |
| T-XR-6.8: Passthrough | 16 hours |
| T-XR-6.9: Unit Tests | 16 hours |
| **Total** | **168 hours** |
