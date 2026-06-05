# PHASE 6 ARCHITECTURE: Rendering Pipeline

## Phase Overview

Phase 6 implements the XR-specific rendering pipeline optimizations required for comfortable 90Hz/120Hz performance. This phase covers stereo rendering, foveated rendering, compositor layer management, reprojection (ATW/ASW), and hidden area mesh optimization. The rendering pipeline must achieve 11ms frame time consistently.

## Architectural Decisions

### ADR-XR-050: Stereo Rendering Strategy

**Context**: XR requires rendering two viewpoints (left/right eye) efficiently.

**Decision**: Implement three stereo methods with runtime selection:
1. **Multi-View (OVR_multiview2)**: Single draw call, GPU selects eye via gl_ViewID
2. **Instanced**: Two instances per draw, select eye via gl_InstanceID
3. **Sequential**: Traditional two-pass rendering

**Consequences**:
- Multi-view: 40% fewer draw calls, requires GPU support
- Instanced: 50% draw call reduction, wider GPU support
- Sequential: Works everywhere, 2x draw calls

### ADR-XR-051: Foveated Rendering Architecture

**Context**: Peripheral vision has lower acuity; rendering full resolution everywhere wastes GPU.

**Decision**: Implement Variable Rate Shading (VRS) with three modes:
1. **Fixed**: Static fovea at screen center
2. **Dynamic**: Fovea follows eye gaze
3. **Contrast-Adaptive**: Higher rate for high-contrast regions

**Consequences**:
- Peripheral pixels reduced by 4-16x
- Dynamic requires eye tracking
- Contrast-adaptive maintains edge detail

### ADR-XR-052: Foveation Region Model

**Context**: Shading rate must vary smoothly from fovea to periphery.

**Decision**: Define three concentric regions:
| Region | Radius | Shading Rate |
|--------|--------|--------------|
| Fovea | 5 deg | 1x (full) |
| Parafoveal | 20 deg | 2x (half) |
| Peripheral | 55 deg | 4x (quarter) |

**Consequences**:
- Matches human visual acuity falloff
- Clear boundaries for VRS tile assignment
- Configurable radii per application

### ADR-XR-053: Compositor Layer System

**Context**: XR rendering involves multiple layers (3D scene, UI, passthrough) that must composite correctly.

**Decision**: Implement five layer types:
1. **PROJECTION**: Main stereo 3D scene
2. **QUAD**: Flat 2D panels in 3D space
3. **CYLINDER**: Curved UI surfaces
4. **CUBEMAP**: Environment maps
5. **EQUIRECT**: 360-degree panoramas

With priority-based ordering and blend modes.

**Consequences**:
- UI rendered at native resolution (no aliasing)
- Passthrough as background layer
- Flexible layer composition

### ADR-XR-054: Reprojection Strategy

**Context**: Frame drops cause judder; reprojection synthesizes frames from previous data.

**Decision**: Implement three reprojection modes:
1. **ATW (Asynchronous Timewarp)**: Rotation-only correction
2. **ASW (Asynchronous Spacewarp)**: Translation via motion vectors
3. **Hybrid**: ATW + partial ASW for different regions

**Consequences**:
- ATW handles head rotation during frame drop
- ASW handles translation (walking) artifacts
- Hybrid balances quality vs latency

### ADR-XR-055: Pose Prediction Model

**Context**: Rendered frame displays 20ms+ after rendering; prediction compensates.

**Decision**: Implement prediction with configurable method:
1. **Linear**: Extrapolate velocity
2. **Quadratic**: Extrapolate with acceleration
3. **Kalman**: Filtered prediction for noisy tracking

**Consequences**:
- Reduces perceived latency
- Kalman smooths jittery tracking
- Prediction horizon configurable

### ADR-XR-056: Hidden Area Mesh Optimization

**Context**: VR lenses have invisible corner regions; rendering there wastes GPU.

**Decision**: Implement Hidden Area Mesh (HAM) with three methods:
1. **Stencil**: Mark hidden pixels in stencil buffer
2. **Depth**: Write far depth to reject via early-Z
3. **Combined**: Stencil + depth for maximum rejection

**Consequences**:
- 10-15% pixel reduction on typical HMDs
- Early-Z rejection before fragment shader
- Per-HMD mesh geometry

## Component Specifications

### Stereo Rendering System

```
StereoRenderer (Abstract Base)
├── render(scene, left_view, right_view)
├── get_view_matrix(eye) -> Mat4
├── get_projection_matrix(eye) -> Mat4
└── Properties
    ├── ipd: float (meters)
    ├── method: StereoMethod
    └── projection_type: ProjectionType

MultiViewStereoRenderer(StereoRenderer)
├── Requires: OVR_multiview2 extension
├── Shader: Uses gl_ViewID for eye selection
├── Render: Single draw call renders both eyes
└── Pros: Minimum draw calls, GPU handles view selection

InstancedStereoRenderer(StereoRenderer)
├── Requires: Instancing support
├── Shader: Uses gl_InstanceID % 2 for eye selection
├── Render: 2 instances per draw call
└── Pros: Wider GPU support than multi-view

SequentialStereoRenderer(StereoRenderer)
├── Requires: Nothing special
├── Render: Two separate render passes
└── Pros: Universal compatibility

StereoConfig
├── ipd_meters: float (default 0.063)
├── near_plane: float (default 0.1)
├── far_plane: float (default 1000.0)
├── ipd_mode: IPDMode (HARDWARE, SOFTWARE, WORLD_SCALE)
└── projection_type: ProjectionType (SYMMETRIC, CANTED, ASYMMETRIC)

StereoMethod Enum
├── MULTI_VIEW
├── INSTANCED
└── SEQUENTIAL

create_stereo_renderer(config) -> StereoRenderer
```

### Foveated Rendering System

```
FoveatedRenderer (Abstract Base)
├── update_gaze(gaze_direction) -> None
├── get_vrs_image() -> Array[int, W, H]
├── get_pixel_savings() -> float
└── Properties
    ├── type: FoveationType
    ├── fovea_radius: float (degrees)
    ├── parafoveal_radius: float (degrees)
    └── peripheral_radius: float (degrees)

FixedFoveatedRenderer(FoveatedRenderer)
├── Gaze: Fixed at screen center
├── VRS: Static shading rate map
└── Use: No eye tracking available

DynamicFoveatedRenderer(FoveatedRenderer)
├── Gaze: Updated from eye tracker
├── VRS: Recalculated when gaze moves
├── Smoothing: Gaze position smoothed to reduce jitter
└── Use: Eye tracking available

ContrastAdaptiveFoveatedRenderer(FoveatedRenderer)
├── Gaze: Follows eye or screen center
├── Contrast: Analyze previous frame for edges
├── VRS: Higher rate where contrast high
└── Use: Maintain edge detail in periphery

FoveationConfig
├── fovea_radius: float (default 5.0 deg)
├── parafoveal_radius: float (default 20.0 deg)
├── peripheral_radius: float (default 55.0 deg)
├── gaze_smoothing: float (0-1)
└── contrast_threshold: float

ShadingRate Enum (VRS rates)
├── FULL (1x1)
├── HALF_X (2x1)
├── HALF_Y (1x2)
├── HALF (2x2)
├── QUARTER_X (4x2)
├── QUARTER_Y (2x4)
└── QUARTER (4x4)

FoveationRegion Enum
├── FOVEA (full rate)
├── PARAFOVEAL (half rate)
└── PERIPHERAL (quarter rate)

create_foveated_renderer(config) -> FoveatedRenderer
```

### Compositor System

```
XRCompositor
├── Layer Management
│   ├── add_layer(layer) -> int (layer id)
│   ├── remove_layer(layer_id) -> bool
│   ├── reorder_layers() -> None (by priority)
│   └── get_layers() -> List[CompositorLayer]
├── Rendering
│   ├── begin_frame() -> FrameState
│   ├── submit_layer(layer_id, textures) -> None
│   └── end_frame() -> None
├── Metrics
│   ├── frame_time_ms: float
│   ├── submit_time_ms: float
│   └── wait_time_ms: float
└── Configuration
    ├── max_layers: int (default 16)
    ├── depth_testing: bool
    └── chromatic_aberration_correction: bool

CompositorLayer (Abstract Base)
├── layer_id: int
├── layer_type: LayerType
├── priority: int (lower = rendered first)
├── blend_mode: BlendMode
├── flags: LayerFlags
└── is_visible: bool

ProjectionLayer(CompositorLayer)
├── Main stereo 3D scene
├── textures: Tuple[Texture, Texture] (left, right)
└── fov: FieldOfView

QuadLayer(CompositorLayer)
├── Flat 2D panel in 3D space
├── position: Vec3
├── orientation: Quat
├── size: Vec2 (meters)
└── texture: Texture

CylinderLayer(CompositorLayer)
├── Curved UI surface
├── position: Vec3
├── orientation: Quat
├── radius: float
├── central_angle: float
└── aspect_ratio: float

LayerType Enum
├── PROJECTION
├── QUAD
├── CYLINDER
├── CUBEMAP
└── EQUIRECT

BlendMode Enum
├── OPAQUE
├── ALPHA_BLEND
├── PREMULTIPLIED
└── ADDITIVE

LayerFlags Enum
├── HEAD_LOCKED
├── WORLD_LOCKED
├── DEPTH_TEST
└── STATIC

create_compositor(config) -> XRCompositor
```

### Reprojection System

```
Reprojection (Abstract Base)
├── reproject(frame, new_pose, old_pose) -> Frame
├── get_rotation_delta(new_pose, old_pose) -> Quat
├── get_translation_delta(new_pose, old_pose) -> Vec3
└── Properties
    ├── mode: ReprojectionMode
    ├── prediction_method: PredictionMethod
    └── is_enabled: bool

ATWReprojection(Reprojection)
├── Rotation-Only Correction
│   ├── Calculate rotation delta between poses
│   ├── Apply rotation to rendered frame
│   └── Clamp rotation to prevent artifacts
├── Late Latch: Integrate pose at last moment
└── Use: Handles head rotation during frame drop

ASWReprojection(Reprojection)
├── Translation Correction
│   ├── Analyze motion vectors from previous frames
│   ├── Estimate translation from motion
│   ├── Synthesize new frame via displacement
│   └── Handle disocclusion (newly visible regions)
├── Motion Caching: Reuse motion analysis
└── Use: Handles walking/translation artifacts

HybridReprojection(Reprojection)
├── Combined ATW + ASW
│   ├── ATW for rotation everywhere
│   ├── ASW for high-motion regions
│   └── Blend based on motion magnitude
└── Use: Best quality, highest cost

ReprojectionConfig
├── target_frame_time_ms: float (11.11 for 90Hz)
├── photon_time_offset_ms: float (time to display)
├── prediction_horizon_ms: float (how far to predict)
├── atw_rotation_limit: float (radians, max correction)
└── asw_motion_threshold: float (pixels/frame)

ReprojectionMode Enum
├── NONE
├── ATW
├── ASW
└── HYBRID

PredictionMethod Enum
├── NONE
├── LINEAR
├── QUADRATIC
└── KALMAN

create_reprojection(config) -> Reprojection
```

### Hidden Area Mesh System

```
HiddenAreaMask (Abstract Base)
├── apply_mask(render_context) -> None
├── clear_mask(render_context) -> None
├── get_visible_area_ratio() -> float
└── Properties
    ├── type: HiddenAreaType
    ├── left_mesh: MeshData
    └── right_mesh: MeshData

StencilHiddenAreaMask(HiddenAreaMask)
├── Apply: Write stencil value to hidden regions
├── Test: Reject pixels where stencil matches
└── Performance: Fragment shader never runs

DepthHiddenAreaMask(HiddenAreaMask)
├── Apply: Write far depth to hidden regions
├── Test: Early-Z rejects hidden pixels
└── Performance: Fragment shader may partially run

CombinedHiddenAreaMask(HiddenAreaMask)
├── Apply: Both stencil and depth
├── Test: Stencil first, then depth
└── Performance: Maximum rejection

HiddenAreaConfig
├── stencil_reference: int (stencil value)
├── depth_value: float (typically 1.0)
├── padding_pixels: int (expand mask slightly)
└── mesh_format: MeshFormat

MeshData
├── vertices: List[Vec2] (normalized device coords)
├── indices: List[int] (triangle list)
├── vertex_count: int
└── triangle_count: int

HiddenAreaType Enum
├── NONE
├── STENCIL
├── DEPTH
├── MESH
└── COMBINED

MeshFormat Enum
├── TRIANGLE_LIST
├── TRIANGLE_FAN
└── LINE_LOOP

create_hidden_area_mask(config, left_mesh, right_mesh) -> HiddenAreaMask
```

### Unified Render Pipeline

```
XRRenderPipeline
├── Components
│   ├── stereo: StereoRenderer
│   ├── foveated: FoveatedRenderer
│   ├── compositor: XRCompositor
│   ├── reprojection: Reprojection
│   └── hidden_area: HiddenAreaMask
├── Frame Lifecycle
│   ├── begin_frame() -> FrameState
│   ├── begin_eye(eye) -> ViewState
│   ├── render_scene(scene)
│   ├── end_eye(eye)
│   ├── submit_frame()
│   └── end_frame()
├── Settings
│   ├── resolution_scale: float (0.5 - 2.0)
│   ├── refresh_rate: RefreshRate
│   ├── display_mode: DisplayMode
│   └── quality_preset: QualityPreset
└── Metrics
    ├── frame_time_ms: float
    ├── gpu_time_ms: float
    ├── pixel_count: int
    └── draw_call_count: int

XRRenderSettings (Resource)
├── Quality Presets
│   ├── low_quality(): 70% resolution, sequential, fixed fov
│   ├── medium_quality(): 100% resolution, instanced, ATW
│   ├── high_quality(): 120% resolution, multi-view, hybrid
│   └── ultra_quality(): 150% resolution, contrast-adaptive, ASW
├── Stereo Settings
├── Foveation Settings
├── Reprojection Settings
└── Hidden Area Settings

RefreshRate Enum
├── HZ_72
├── HZ_80
├── HZ_90
├── HZ_120
└── HZ_144

DisplayMode Enum
├── VR (full immersion)
├── AR (passthrough + virtual)
├── MR (blended)
└── SPECTATOR (2D monitor output)

create_xr_render_pipeline(quality) -> XRRenderPipeline
```

## Integration Points

### Dependencies (Incoming)
- Phase 1: Runtime provides view info and frame timing
- RHI: GPU texture and buffer management
- Scene: Render commands

### Dependents (Outgoing)
- Phase 2: Foveated rendering consumes eye tracking gaze
- Phase 5: Compositor renders UI layers

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     XRRenderPipeline                         │
│                                                              │
│  begin_frame()                                               │
│       │                                                      │
│       ▼                                                      │
│  ┌───────────────┐                                          │
│  │ HiddenAreaMask│ Apply stencil/depth mask                 │
│  └───────┬───────┘                                          │
│          │                                                   │
│          ▼                                                   │
│  ┌───────────────┐  ┌───────────────┐                       │
│  │ Left Eye      │  │ Right Eye     │                       │
│  │               │  │               │                       │
│  │ ┌───────────┐ │  │ ┌───────────┐ │                       │
│  │ │ Foveated  │ │  │ │ Foveated  │ │ Apply VRS per region  │
│  │ │ Rendering │ │  │ │ Rendering │ │                       │
│  │ └─────┬─────┘ │  │ └─────┬─────┘ │                       │
│  │       │       │  │       │       │                       │
│  │       ▼       │  │       ▼       │                       │
│  │  Scene Draw   │  │  Scene Draw   │                       │
│  └───────┬───────┘  └───────┬───────┘                       │
│          │                  │                                │
│          └────────┬─────────┘                                │
│                   │                                          │
│                   ▼                                          │
│          ┌───────────────┐                                  │
│          │   Compositor  │ Merge layers, apply UI          │
│          └───────┬───────┘                                  │
│                  │                                          │
│                  ▼                                          │
│          ┌───────────────┐                                  │
│          │ Reprojection  │ ATW/ASW if needed               │
│          └───────┬───────┘                                  │
│                  │                                          │
│  submit_frame()  ▼                                          │
│          ┌───────────────┐                                  │
│          │   Display     │                                  │
│          └───────────────┘                                  │
└─────────────────────────────────────────────────────────────┘
```

## Performance Requirements

| Component | Budget | Notes |
|-----------|--------|-------|
| Frame Time | 11.11ms | 90Hz target |
| Hidden Area Apply | <0.1ms | Stencil write |
| Foveated Setup | <0.2ms | VRS image update |
| Scene Render | 6-8ms | Main GPU work |
| Compositor | <0.5ms | Layer merge |
| Reprojection | <2ms | ASW fallback |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Frame drops at 90Hz | High | High | Auto quality scaling, reprojection |
| Foveation visible | Medium | Medium | Smooth region transitions |
| Reprojection artifacts | Medium | Medium | Clamp correction magnitude |
| Hidden area mesh wrong | Low | Medium | Per-HMD mesh data |
