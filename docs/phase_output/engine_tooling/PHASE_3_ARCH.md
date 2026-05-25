# PHASE 3 ARCHITECTURE: Visual Content Creation Tools

## Phase Overview

Phase 3 implements the visual content creation tools: Level Editor, Material Editor, Animation Tools, and Terrain Tools. These tools enable artists and designers to create game content.

## Components

### 1. Level Editor (engine/tooling/leveleditor/)

**Purpose**: Scene construction and object management

**Architecture**:
```
LevelEditor
    |
    +-- HierarchyTree
    |       +-- HierarchyNode (parent/children, depth, path)
    |       +-- HierarchyFolder (organizational)
    |       +-- HierarchyGroup (transform-affecting)
    |       +-- DragDropOperation (reparent, reorder, copy, link)
    |       +-- HierarchyFilter (visible, selected, layer, name, type)
    |
    +-- PlacementTool
    |       +-- PlacementMode (SINGLE, PAINT_BRUSH, SCATTER, FOLIAGE, SPLINE)
    |       +-- BrushSettings (radius, falloff, density)
    |       +-- ScatterSettings (min_distance, alignment, clustering)
    |       +-- Algorithms:
    |               +-- Poisson disk sampling (blue noise)
    |               +-- Grid jitter sampling
    |               +-- Cluster sampling
    |               +-- Hermite spline interpolation
    |
    +-- PrefabSystem
    |       +-- PrefabAsset (template with components, children)
    |       +-- PrefabInstance (instantiated with overrides)
    |       +-- PrefabVariant (derived prefabs)
    |       +-- PrefabOverride (VALUE, ADD_COMPONENT, REMOVE_COMPONENT)
    |       +-- Circular reference detection
    |
    +-- SnappingSystem
    |       +-- GridSnap (world/local/custom)
    |       +-- SurfaceSnap (raycast-based)
    |       +-- VertexSnap (mesh vertex)
    |       +-- EdgeSnap (midpoint/perpendicular)
    |       +-- PivotSnap (object center)
    |       +-- SnapPriority (VERTEX_FIRST, SURFACE_FIRST, GRID_FIRST, NEAREST)
    |
    +-- LayerManager
    |       +-- Layer (visibility, locking, color, settings)
    |       +-- LayerMask (bitfield operations)
    |
    +-- MeasurementTools
    |       +-- DistanceMeasurement (point-to-point, cumulative)
    |       +-- AngleMeasurement (3-point, surface normal)
    |       +-- AreaMeasurement (shoelace formula)
    |
    +-- BookmarkManager
    |       +-- CameraBookmark (position, rotation, FOV)
    |       +-- BookmarkCategory
    |
    +-- AlignmentTool / DistributionTool
            +-- AlignAxis (X, Y, Z, XY, XZ, YZ, ALL)
            +-- AlignReference (FIRST, LAST, CENTER, SELECTION_BOUNDS)
```

**Key Algorithms**:
- **Poisson Disk Sampling**: Blue-noise distribution with grid acceleration for scatter placement
- **Hermite Spline**: Smooth curve evaluation for spline-based placement
- **Point-to-Line Distance**: Precision snapping calculations

### 2. Material Editor (engine/tooling/material_editor/)

**Purpose**: Node-based material authoring with shader compilation

**Architecture**:
```
MaterialEditor
    |
    +-- MaterialGraph
    |       +-- Node management (add, remove, duplicate)
    |       +-- Connection management (connect, disconnect)
    |       +-- Validation (cycles, types, completeness)
    |       +-- Serialization (to_dict, from_dict, JSON)
    |       +-- Callbacks (node_added, connection_added, state_changed)
    |
    +-- MaterialNode (ABC)
    |       +-- NodePin (DataType, Direction)
    |       +-- NodeCategory (INPUT, MATH, TEXTURE, UTILITY, PBR, OUTPUT)
    |       +-- Node Types (40+):
    |               +-- Input: Constant, Constant2/3/4, Parameter
    |               +-- Math: Add, Subtract, Multiply, Lerp, Clamp, Power, etc.
    |               +-- Texture: TextureSample, UV, TilingOffset, NormalMap
    |               +-- Utility: Time, WorldPosition, ViewDirection, Split, Combine
    |               +-- PBR: Fresnel, GGX, Lambert, BRDF
    |               +-- Output: PBROutput, UnlitOutput
    |
    +-- MaterialCompiler
    |       +-- ShaderGenerator (ABC)
    |               +-- HLSLGenerator
    |               +-- GLSLGenerator
    |               +-- MetalGenerator
    |       +-- Compilation Pipeline:
    |               1. Validate graph
    |               2. Collect parameters/textures
    |               3. Generate code (topological order)
    |               4. Apply optimization passes
    |
    +-- ConnectionValidator
    |       +-- Type compatibility rules
    |       +-- Implicit/explicit conversions
    |       +-- Cycle detection
    |
    +-- MaterialParameters
    |       +-- ParameterType (Scalar, Vector, Color, Texture, Bool, Int)
    |       +-- ParameterRange (min, max, step, soft limits)
    |       +-- ParameterSemantic (Roughness, Metallic, Albedo, Normal, etc.)
    |
    +-- MaterialInstances
    |       +-- MaterialDefinition (base material)
    |       +-- MaterialInstance (parameter overrides)
    |
    +-- MaterialLibrary
    |       +-- LibraryItem (type, category, tags, ratings)
    |       +-- SearchFilter (query, categories, favorites)
    |       +-- Default categories (Metals, Woods, Stones, etc.)
    |
    +-- MaterialPreview
            +-- PreviewRenderer (ABC)
            +-- LightingPreset (Studio, Outdoor, Indoor, Dramatic)
            +-- Camera controls (orbit, zoom, pan)
            +-- Display options (wireframe, UV grid, normals)
```

**Data Types**:
| Type | Wire Color | Conversions |
|------|-----------|-------------|
| Execution | White | None |
| Bool | Red | Int, Float |
| Int | Cyan | Float, Bool |
| Float | Green | Int (truncate), Bool |
| Float2 | Light Green | Float (x) |
| Float3 | Yellow | Float2 (xy), Float (x) |
| Float4 | Pink | Float3 (xyz), Float (x) |
| Texture | Orange | None |

### 3. Animation Tools (engine/tooling/animation_tools/)

**Purpose**: Animation editing and state machine authoring

**Architecture**:
```
AnimationTools
    |
    +-- AnimGraphEditor
    |       +-- GraphNode (ABC)
    |               +-- StateNode (animation state)
    |               +-- TransitionNode (conditions, blend duration)
    |               +-- BlendNode (linear, additive, mesh-space)
    |               +-- BlendSpace1D/2D (sample interpolation)
    |       +-- Parameter management
    |       +-- Connection validation
    |
    +-- Sequencer
    |       +-- Timeline (frame rate, snapping, markers, loop ranges)
    |       +-- AnimationTrack[T] (generic keyframe track)
    |               +-- TransformTrack (position, rotation, scale)
    |               +-- SkeletalTrack (per-bone animation)
    |               +-- CameraTrack (FOV keyframes)
    |               +-- EventTrack (time ranges)
    |               +-- AudioTrack (clip management)
    |               +-- PropertyTrack (generic properties)
    |       +-- SequencerPlayback (ONCE, LOOP, PING_PONG, CLAMP)
    |
    +-- CurveEditor
    |       +-- TangentHandle (slope, length, normalization)
    |       +-- CurveKey (tangent mode: AUTO, FREE, LINEAR, FLAT, WEIGHTED, BREAK)
    |       +-- EasingFunction (30+ functions: bounce, elastic, back, etc.)
    |       +-- BezierCurve (De Casteljau's algorithm)
    |       +-- HermiteCurve (basis functions)
    |
    +-- PoseEditor
    |       +-- AnimPose (per-bone transforms with weights)
    |       +-- AdditivePose (delta from reference)
    |       +-- PoseLibrary (category-based organization)
    |       +-- Bone selection, mirroring, blending
    |
    +-- IKSetup
    |       +-- IKBone (constraints: hinge, ball-socket, angle limits)
    |       +-- IKEffector (position/rotation weights)
    |       +-- IKPoleVector (plane orientation)
    |       +-- Solver configs: TwoBone, FABRIK, CCD
    |       +-- Cone constraint implementation
    |
    +-- SkeletonEditor
    |       +-- Socket (attachment points)
    |       +-- VirtualBone (MIDPOINT, LOOK_AT, COPY, DISTANCE)
    |       +-- RetargetMapping (translation/rotation modes)
    |       +-- BoneMirrorPair (left/right mapping)
    |
    +-- MontageEditor
    |       +-- MontageSection (loop config, links, branches)
    |       +-- SectionLink (conditional branching)
    |       +-- AnimSlot (bone-filtered with priorities)
    |
    +-- NotifiesEditor
    |       +-- AnimNotify (ABC)
    |               +-- AnimNotifyState (duration-based)
    |               +-- SoundNotify (volume, pitch, bone attachment)
    |               +-- ParticleNotify (socket attachment)
    |               +-- FootstepNotify (surface detection)
    |
    +-- PreviewScene
            +-- GroundSettings (grid, reflections)
            +-- LightingSettings (directional, ambient, sky)
            +-- PreviewProp (bone/socket attachment)
            +-- PreviewPlayback (seek, loop, speed)
```

### 4. Terrain Tools (engine/tooling/terrain/)

**Purpose**: Landscape sculpting, painting, and decoration

**Architecture**:
```
TerrainTools
    |
    +-- TerrainSculptTool
    |       +-- SculptMode (RAISE, LOWER, SMOOTH, FLATTEN, NOISE, LEVEL, STAMP)
    |       +-- BrushShape (CIRCLE, SQUARE, CUSTOM)
    |       +-- FalloffCurve (LINEAR, SMOOTH, SPHERE, TIP, FLAT)
    |       +-- TerrainBrush (influence calculation)
    |       +-- TerrainData (heightmap, dirty chunk tracking)
    |       +-- SculptOperation (undo/redo records)
    |
    +-- TerrainPaintTool
    |       +-- PaintMode (PAINT, ERASE, BLEND, REPLACE)
    |       +-- LayerBlendMode (HEIGHT, SLOPE, NOISE, COMBINED, MANUAL)
    |       +-- TerrainMask (ABC)
    |               +-- HeightMask (height range, feathering)
    |               +-- SlopeMask (slope angle)
    |               +-- NoiseMask (FBM with octaves)
    |       +-- PaintLayer (splatmap weight storage)
    |       +-- Weight normalization (sum to 1.0)
    |
    +-- TerrainImportExport
    |       +-- HeightmapImporter (RAW_8BIT, RAW_16BIT, RAW_32BIT, PNG)
    |       +-- TerrainExporter (RAW, OBJ mesh, JSON)
    |       +-- Auto-dimension detection
    |
    +-- TerrainMaterialManager
    |       +-- TerrainMaterialLayer (PBR textures, UV settings)
    |       +-- MaterialBlendSettings
    |       +-- BlendMode (LINEAR, HEIGHT_BASED, SLOPE_BASED, NOISE_BASED, SHARP)
    |       +-- auto_paint_by_slope(), auto_paint_by_height()
    |
    +-- FoliagePlacementTool
    |       +-- FoliageType (GRASS, TREE, BUSH, ROCK, FLOWER, DEBRIS)
    |       +-- FoliageInstance (position, rotation, scale, LOD)
    |       +-- FoliageLayer (density map, settings)
    |       +-- FoliageDensityBrush
    |       +-- Slope/height constraints
    |
    +-- TerrainLODSystem
    |       +-- LODLevel (LOD0-LOD4: full to 1/16 resolution)
    |       +-- TerrainChunk (height data, neighbor references)
    |       +-- ChunkState (UNLOADED, LOADING, LOADED, STREAMING)
    |       +-- Priority-based chunk loading
    |       +-- Neighbor stitching
    |
    +-- ErosionSimulator
            +-- ErosionType (HYDRAULIC, THERMAL, COMBINED)
            +-- HydraulicErosionParams (inertia, sediment capacity, gravity)
            +-- ThermalErosionParams (talus angle, erosion rate)
            +-- WaterDroplet simulation
            +-- Thermal relaxation
```

## Data Flow

### Material Compilation Flow
```
MaterialGraph
    -> ConnectionValidator.validate()
    -> MaterialCompiler.collect_parameters()
    -> topological_sort(nodes)
    -> For each node:
        -> node.generate_code(inputs, output_vars)
    -> ShaderGenerator.generate_vertex_shader()
    -> ShaderGenerator.generate_pixel_shader()
    -> apply_optimizations()
    -> CompiledMaterial
```

### Animation Graph Evaluation Flow
```
AnimGraphEditor
    -> Find active state
    -> Evaluate transitions (conditions)
    -> If transitioning:
        -> BlendNode.get_blend_weights()
        -> Interpolate poses
    -> Apply IK chains
    -> Fire notify events
    -> Output final pose
```

### Terrain Sculpt Flow
```
Mouse Input
    -> TerrainSculptTool.begin_stroke()
    -> For each sample point:
        -> TerrainBrush.get_influence(x, y)
        -> Apply sculpt mode:
            RAISE: height += strength * falloff
            SMOOTH: height = average(neighbors)
            FLATTEN: height = target_height
            NOISE: height += random * strength
        -> Mark dirty chunks
    -> TerrainSculptTool.end_stroke()
    -> UndoSystem.record(SculptOperation)
```

## Integration Points

### Foundation Integration
- Tracker: All tools integrate with undo system via `@track_changes`
- Mirror: Property editing through reflection

### Renderer Integration
- Material preview requires render backend
- Terrain LOD system integrates with streaming
- Animation preview needs skeletal mesh rendering

### Scene Integration
- Level editor manages scene hierarchy
- Prefabs interact with scene instantiation
- Terrain integrates with collision/physics

## Thread Safety

| Component | Strategy |
|-----------|----------|
| MaterialGraph | Not thread-safe (single editor) |
| MaterialCompiler | Thread-safe (parallel compilation) |
| TerrainData | Chunk-level locking |
| FoliageLayer | Instance-level operations atomic |
| AnimGraphEditor | Not thread-safe (single editor) |

## Testing Strategy

### Unit Tests
- Poisson sampling distribution quality
- Material node code generation
- Spline interpolation accuracy
- Erosion simulation stability

### Integration Tests
- Full material compile and preview
- Animation graph state transitions
- Terrain import/export round-trip
- Prefab instantiation with overrides

### Visual Tests
- Material appearance verification
- Animation playback quality
- Terrain sculpt results
- Foliage distribution patterns
