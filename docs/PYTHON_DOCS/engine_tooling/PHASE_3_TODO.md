# PHASE 3 TODO: Visual Content Creation Tools

## Overview

Phase 3 implements the visual content creation tools. These tools enable artists and designers to create game content visually.

---

## 1. Level Editor

### 1.1 Hierarchy System
- [ ] **T1.1.1**: Test deep copy with ID regeneration
  - Acceptance: Duplicated nodes have unique IDs
  - Acceptance: References updated correctly
  - File: `engine/tooling/leveleditor/hierarchy.py`

- [ ] **T1.1.2**: Verify circular reference prevention
  - Acceptance: Cannot parent node to its own descendant
  - Acceptance: Error message descriptive

- [ ] **T1.1.3**: Implement hierarchy filtering
  - Acceptance: Filter by visibility, selection, layer
  - Acceptance: Filter by name pattern
  - Acceptance: Filter by component type

### 1.2 Placement System
- [ ] **T1.2.1**: Test Poisson disk sampling quality
  - Acceptance: Blue noise distribution
  - Acceptance: Minimum distance respected
  - Acceptance: Performance < 100ms for 1000 points
  - File: `engine/tooling/leveleditor/placement.py`

- [ ] **T1.2.2**: Verify spline placement
  - Acceptance: Objects placed along Hermite spline
  - Acceptance: Spacing configurable
  - Acceptance: Rotation follows spline tangent

- [ ] **T1.2.3**: Test foliage brush
  - Acceptance: Density painting works
  - Acceptance: Slope/height constraints honored
  - Acceptance: Random rotation/scale applied

### 1.3 Prefab System
- [ ] **T1.3.1**: Test prefab instantiation
  - Acceptance: Instance created from asset
  - Acceptance: Components copied correctly
  - File: `engine/tooling/leveleditor/prefabs.py`

- [ ] **T1.3.2**: Verify override system
  - Acceptance: Property overrides persist
  - Acceptance: Component add/remove overrides work
  - Acceptance: Child add/remove overrides work

- [ ] **T1.3.3**: Test prefab variant inheritance
  - Acceptance: Variant inherits from parent
  - Acceptance: Parent changes propagate
  - Acceptance: Variant overrides preserved

- [ ] **T1.3.4**: Implement nested prefab support
  - Acceptance: Prefabs can contain prefab instances
  - Acceptance: Circular reference detection works

### 1.4 Snapping System
- [ ] **T1.4.1**: Test grid snapping
  - Acceptance: Position snaps to grid
  - Acceptance: Custom grid sizes work
  - Acceptance: Local/world grid toggle
  - File: `engine/tooling/leveleditor/snapping.py`

- [ ] **T1.4.2**: Implement surface snapping
  - Acceptance: Objects snap to surfaces
  - Acceptance: Normal alignment option
  - Integration: Requires raycast from scene

- [ ] **T1.4.3**: Test vertex/edge snapping
  - Acceptance: Snaps to mesh vertices
  - Acceptance: Edge midpoint snapping
  - Integration: Requires MeshProvider protocol

### 1.5 Layer System
- [ ] **T1.5.1**: Test layer visibility
  - Acceptance: Hidden layers not rendered
  - Acceptance: Visibility toggle works
  - File: `engine/tooling/leveleditor/layers.py`

- [ ] **T1.5.2**: Implement layer locking
  - Acceptance: Locked layers not selectable
  - Acceptance: Lock toggle works

### 1.6 Measurements
- [ ] **T1.6.1**: Test distance measurement
  - Acceptance: Point-to-point distance correct
  - Acceptance: Units conversion works
  - File: `engine/tooling/leveleditor/measurements.py`

- [ ] **T1.6.2**: Verify area measurement
  - Acceptance: Polygon area correct (shoelace)
  - Acceptance: Perimeter calculated

---

## 2. Material Editor

### 2.1 Node System
- [ ] **T2.1.1**: Test all math nodes
  - Acceptance: Add, Subtract, Multiply, Divide work
  - Acceptance: Lerp, Clamp, Saturate work
  - Acceptance: Power, Dot, Cross, Normalize work
  - File: `engine/tooling/material_editor/material_nodes.py`

- [ ] **T2.1.2**: Test texture sampling
  - Acceptance: TextureSample node works
  - Acceptance: UV input respected
  - Acceptance: Sampler settings applied

- [ ] **T2.1.3**: Test PBR nodes
  - Acceptance: Fresnel calculation correct
  - Acceptance: GGX distribution correct
  - Acceptance: BRDF output matches reference

### 2.2 Graph Editor
- [ ] **T2.2.1**: Test connection validation
  - Acceptance: Type mismatches rejected
  - Acceptance: Cycles detected
  - Acceptance: Valid connections created
  - File: `engine/tooling/material_editor/material_graph.py`

- [ ] **T2.2.2**: Verify topological evaluation
  - Acceptance: Nodes evaluated in dependency order
  - Acceptance: Output node evaluated last

- [ ] **T2.2.3**: Test serialization
  - Acceptance: Graph saves to JSON
  - Acceptance: Graph loads correctly
  - Acceptance: All node types handled

### 2.3 Compiler
- [ ] **T2.3.1**: Test HLSL generation
  - Acceptance: Valid HLSL output
  - Acceptance: Variable names unique
  - Acceptance: Texture bindings correct
  - File: `engine/tooling/material_editor/material_compiler.py`

- [ ] **T2.3.2**: Test GLSL generation
  - Acceptance: Valid GLSL output
  - Acceptance: Uniform declarations correct

- [ ] **T2.3.3**: Test Metal generation
  - Acceptance: Valid Metal output
  - Acceptance: Argument buffers correct

### 2.4 Parameters
- [ ] **T2.4.1**: Test parameter types
  - Acceptance: Scalar with range works
  - Acceptance: Vector with components works
  - Acceptance: Color picker works
  - Acceptance: Texture selection works
  - File: `engine/tooling/material_editor/material_parameters.py`

- [ ] **T2.4.2**: Verify parameter semantics
  - Acceptance: Roughness mapped to 0-1
  - Acceptance: Metallic mapped to 0-1
  - Acceptance: Normal map format detected

### 2.5 Instances
- [ ] **T2.5.1**: Test material instancing
  - Acceptance: Instance created from definition
  - Acceptance: Parameter overrides work
  - Acceptance: Base material changes reflect
  - File: `engine/tooling/material_editor/material_instances.py`

### 2.6 Preview
- [ ] **T2.6.1**: Integrate preview renderer
  - Acceptance: Material renders on sphere
  - Acceptance: Lighting presets work
  - Acceptance: Camera controls work
  - Integration: Requires render backend
  - File: `engine/tooling/material_editor/material_preview.py`

---

## 3. Animation Tools

### 3.1 Anim Graph
- [ ] **T3.1.1**: Test state machine transitions
  - Acceptance: Conditions evaluated correctly
  - Acceptance: Blend duration respected
  - Acceptance: Exit time triggers work
  - File: `engine/tooling/animation_tools/anim_graph_editor.py`

- [ ] **T3.1.2**: Test blend spaces
  - Acceptance: 1D blend interpolates correctly
  - Acceptance: 2D blend triangulates correctly
  - Acceptance: Sample weights sum to 1.0

### 3.2 Sequencer
- [ ] **T3.2.1**: Test timeline management
  - Acceptance: Frame rate conversion correct
  - Acceptance: Snapping to frames works
  - Acceptance: Loop range respected
  - File: `engine/tooling/animation_tools/sequencer.py`

- [ ] **T3.2.2**: Test keyframe interpolation
  - Acceptance: Linear interpolation works
  - Acceptance: Slerp for rotations
  - Acceptance: Custom curves applied

- [ ] **T3.2.3**: Test playback modes
  - Acceptance: ONCE plays once
  - Acceptance: LOOP loops correctly
  - Acceptance: PING_PONG reverses

### 3.3 Curve Editor
- [ ] **T3.3.1**: Test Bezier evaluation
  - Acceptance: De Casteljau's algorithm correct
  - Acceptance: Tangent handles work
  - File: `engine/tooling/animation_tools/curve_editor.py`

- [ ] **T3.3.2**: Test easing functions
  - Acceptance: All 30+ easings work
  - Acceptance: Bounce, elastic, back correct

### 3.4 IK Setup
- [ ] **T3.4.1**: Test IK solvers
  - Acceptance: Two-bone solver works
  - Acceptance: FABRIK solver works
  - Acceptance: CCD solver works
  - File: `engine/tooling/animation_tools/ik_setup.py`

- [ ] **T3.4.2**: Test IK constraints
  - Acceptance: Angle limits respected
  - Acceptance: Cone constraint works
  - Acceptance: Pole vector works

### 3.5 Montages
- [ ] **T3.5.1**: Test section playback
  - Acceptance: Sections play in sequence
  - Acceptance: Looping sections work
  - File: `engine/tooling/animation_tools/montage_editor.py`

- [ ] **T3.5.2**: Test branching
  - Acceptance: Conditional branches work
  - Acceptance: Link transitions work

### 3.6 Notifies
- [ ] **T3.6.1**: Test notify firing
  - Acceptance: Sound notify fires
  - Acceptance: Particle notify fires
  - Acceptance: Custom notify fires
  - File: `engine/tooling/animation_tools/notifies_editor.py`

---

## 4. Terrain Tools

### 4.1 Sculpting
- [ ] **T4.1.1**: Test sculpt modes
  - Acceptance: RAISE increases height
  - Acceptance: LOWER decreases height
  - Acceptance: SMOOTH averages neighbors
  - Acceptance: FLATTEN targets height
  - File: `engine/tooling/terrain/sculpt_tools.py`

- [ ] **T4.1.2**: Test brush falloff
  - Acceptance: LINEAR falloff correct
  - Acceptance: SMOOTH falloff (Hermite) correct
  - Acceptance: SPHERE falloff correct

- [ ] **T4.1.3**: Test undo integration
  - Acceptance: Sculpt operations undoable
  - Acceptance: Height data restored

### 4.2 Painting
- [ ] **T4.2.1**: Test paint modes
  - Acceptance: PAINT adds weight
  - Acceptance: ERASE removes weight
  - Acceptance: BLEND redistributes
  - File: `engine/tooling/terrain/paint_tools.py`

- [ ] **T4.2.2**: Test mask types
  - Acceptance: HeightMask filters by height
  - Acceptance: SlopeMask filters by slope
  - Acceptance: NoiseMask adds variation

- [ ] **T4.2.3**: Verify weight normalization
  - Acceptance: Layer weights sum to 1.0
  - Acceptance: No negative weights

### 4.3 Import/Export
- [ ] **T4.3.1**: Test RAW import
  - Acceptance: 8/16/32 bit RAW loads
  - Acceptance: Dimensions detected
  - File: `engine/tooling/terrain/terrain_import.py`

- [ ] **T4.3.2**: Test OBJ export
  - Acceptance: Valid OBJ mesh output
  - Acceptance: UVs included
  - Acceptance: Normals correct

### 4.4 Materials
- [ ] **T4.4.1**: Test layer blending
  - Acceptance: HEIGHT_BASED blends by height
  - Acceptance: SLOPE_BASED blends by slope
  - Acceptance: SHARP blend has hard edges
  - File: `engine/tooling/terrain/terrain_materials.py`

- [ ] **T4.4.2**: Test auto-painting
  - Acceptance: auto_paint_by_slope works
  - Acceptance: auto_paint_by_height works

### 4.5 Foliage
- [ ] **T4.5.1**: Test density painting
  - Acceptance: Instances placed by density
  - Acceptance: Slope constraints honored
  - Acceptance: Height constraints honored
  - File: `engine/tooling/terrain/foliage_tools.py`

- [ ] **T4.5.2**: Test LOD system
  - Acceptance: LOD levels calculated by distance
  - Acceptance: Instances update on camera move

### 4.6 LOD System
- [ ] **T4.6.1**: Test chunk loading
  - Acceptance: Chunks load by priority
  - Acceptance: Max loaded chunks respected
  - File: `engine/tooling/terrain/terrain_lod.py`

- [ ] **T4.6.2**: Test LOD transitions
  - Acceptance: LOD selection by distance
  - Acceptance: Resolution decimation correct

### 4.7 Erosion
- [ ] **T4.7.1**: Test hydraulic erosion
  - Acceptance: Droplet simulation runs
  - Acceptance: Erosion/deposition visible
  - Acceptance: Reasonable performance (50K droplets < 5s)
  - File: `engine/tooling/terrain/erosion_tools.py`

- [ ] **T4.7.2**: Test thermal erosion
  - Acceptance: Material transfers down slope
  - Acceptance: Talus angle respected

---

## Integration Tests

### I1. Full Material Pipeline
- [ ] **I1.1**: Create material, compile, preview
  - Steps: Add nodes, connect, compile HLSL, preview
  - Acceptance: Material renders correctly

### I2. Animation Playback
- [ ] **I2.1**: Create animation, play in preview
  - Steps: Add keyframes, set up curves, play
  - Acceptance: Animation plays smoothly

### I3. Terrain Complete Workflow
- [ ] **I3.1**: Sculpt, paint, add foliage
  - Steps: Sculpt height, paint layers, place foliage
  - Acceptance: Complete terrain visible

### I4. Prefab Round-Trip
- [ ] **I4.1**: Create, save, load prefab
  - Steps: Create hierarchy, save prefab, load instance
  - Acceptance: Instance matches original

---

## Performance Targets

| Metric | Target | Test Method |
|--------|--------|-------------|
| Poisson sampling (1000) | < 100ms | Benchmark |
| Material compile | < 500ms | Benchmark simple material |
| Terrain sculpt | 60 FPS | Brush interaction |
| Animation evaluate | < 1ms | Benchmark state machine |
| Foliage update (10K) | < 10ms | LOD update cycle |

---

## Dependencies

### Required Before Phase 3
- Phase 1: Undo system, logging
- Phase 2: Asset import (textures, meshes)

### Integration Points
- Render backend for previews
- Scene system for level editor
- Physics for surface snapping

### Blocks Phase 4+
- Visual scripting may reference materials
- Debug tools need scene access
