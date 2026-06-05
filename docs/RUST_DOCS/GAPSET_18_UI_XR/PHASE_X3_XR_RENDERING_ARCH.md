# Phase X3: XR Rendering — Architecture

**Tasks:** T-XR-3.1 through T-XR-3.6 (6 tasks)
**Effort:** 19-26 days
**Status:** ✅ COMPLETE (per PROJECT.md verification)

---

## 1. Overview

Phase X3 implements XR-specific rendering: stereo rendering pipeline, foveated rendering, reprojection (ATW/ASW), compositor layers, and hidden area mesh optimization.

---

## 2. Stereo Rendering (`rendering/stereo.py`)

### Per-Eye View Matrices
```
left_view = inverse(hmd_pose) * translate(-ipd/2, 0, 0)
right_view = inverse(hmd_pose) * translate(+ipd/2, 0, 0)
```

IPD (interpupillary distance) from device profile.

### Rendering Modes
| Mode | Description | GPU Requirement |
|------|-------------|-----------------|
| Multiview | Single draw, both eyes | VK_KHR_multiview |
| Instanced | Instanced draw, eye index | Standard |
| Sequential | Two passes, one per eye | Fallback |

### Performance Target
- Frame time ≤11.1ms at 90Hz
- Render scale configurable: 0.5x-2.0x

---

## 3. Foveated Rendering (`rendering/foveated.py`)

### Fixed Foveated Rendering
Quality levels 0-4, pre-configured region sizes.

### Dynamic Foveated Rendering
Eye tracking gaze follows fovea center.

### Regions
| Region | Quality | Typical Coverage |
|--------|---------|------------------|
| Fovea | Full | 5° radius |
| Parafoveal | Reduced | 15° radius |
| Peripheral | Lowest | Remainder |

`@foveated_region` decorator configures per-object quality.

---

## 4. Reprojection (`rendering/reprojection.py`)

### Asynchronous Time Warp (ATW)
Warps last rendered frame to latest predicted HMD pose.

### Asynchronous Space Warp (ASW)
- Generates motion vectors
- Interpolates new frame from previous two
- Reduces render load when dropping frames

### Latency Target
Pose-to-photon <20ms.

---

## 5. Compositor Layers (`rendering/compositor.py`)

### Layer Types
| Layer | Purpose |
|-------|---------|
| Scene | Main 3D world rendering |
| UI Overlay | 2D/3D UI panels |
| Passthrough | Camera feed for AR |
| Quad | Fixed UI panels in space |

### Layer Properties
- Independent resolution per layer
- Independent refresh rate
- Ordering and blending

---

## 6. Hidden Area Mesh (`rendering/hidden_area.py`)

### Purpose
Lens-induced distortion makes screen corners invisible. Hidden area mesh culls these pixels.

### Savings
Typically 15-25% of pixels culled.

### Implementation
- Per-eye mesh from device profile
- Early-Z rejection in fragment shader
- `TrackedDescriptor` for runtime updates

---

## 7. Dependencies

- Phase X1: XR Runtime
- Phase X2: HMD pose
- S14: wgpu RHI backend
- S1: Frame Graph backend
- S15: Rust math library
