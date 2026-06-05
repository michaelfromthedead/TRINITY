# PHASE 3 TODO: WGSL Code Generation

## Source File
`engine/rendering/demoscene/wgsl_codegen.py` (645 lines)

---

## T-3.1: SDF Primitive Function Templates

**Description**: Verify all 7 SDF primitives emit correct WGSL functions.

**Primitives**:
| Function | Formula | Expected Output |
|----------|---------|-----------------|
| sdSphere | `length(p) - r` | Correct sphere distance |
| sdBox | `length(max(abs(p) - b, 0.0))` | Exterior + interior box |
| sdTorus | 2D ring distance in XZ + Y | Correct torus |
| sdCylinder | Capped cylinder | Height and radius correct |
| sdCone | Angle-based | Matches IQ reference |
| sdPlane | Dot product with normal | Infinite plane |
| sdCapsule | Line segment distance | Two endpoints + radius |

**Acceptance Criteria**:
- [ ] Each template compiles as valid WGSL
- [ ] Math matches Inigo Quilez reference implementations
- [ ] Edge cases (zero radius, etc.) handled without NaN/Inf

---

## T-3.2: Domain Operation Emission

**Description**: Verify domain transformation functions emit correctly.

**Operations**:
- domain_repeat(p, cell_size) - Infinite repetition
- domain_mirror(p, axis) - Axis-aligned mirroring
- domain_kifs(p, folds) - Kaleidoscopic folding
- domain_twist(p, amount) - Helical twist
- domain_bend(p, amount) - Curvature
- domain_stretch(p, factors) - Non-uniform scale

**Acceptance Criteria**:
- [ ] Each operation emits valid WGSL function
- [ ] Position input/output preserved
- [ ] Parameters correctly interpolated into template

---

## T-3.3: KIFS Distance Compensation

**Description**: Verify domain_kifs_compensation() generates correct WGSL.

**Expected Code**:
```wgsl
fn domain_kifs_compensation(folds: f32) -> f32 {
    let safe_folds = max(abs(folds), 1.0);
    let angle = 6.283185307179586 / safe_folds;
    let half_angle = angle * 0.5;
    let per_fold = cos(half_angle);
    var comp: f32 = 1.0;
    for (var i = 0u; i < u32(safe_folds); i = i + 1u) {
        comp *= per_fold;
    }
    return max(comp, 1e-8);
}
```

**Acceptance Criteria**:
- [ ] Loop generates correct fold accumulation
- [ ] safe_folds prevents division by zero
- [ ] Final max() prevents zero compensation
- [ ] 2*PI constant precise to 15+ digits

---

## T-3.4: Stretch Distance Compensation

**Description**: Verify stretch compensation calculates minimum scale factor.

**Acceptance Criteria**:
- [ ] Non-uniform scaling extracts min scale
- [ ] Compensation = 1.0 / min(factors)
- [ ] Handles negative scale factors

---

## T-3.5: PBR Material Struct

**Description**: Verify Material struct definition emits correctly.

**Expected Fields**:
```wgsl
struct Material {
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    emissive: vec3<f32>,
    ambient_occlusion: f32,
}
```

**Acceptance Criteria**:
- [ ] All 5 PBR properties present
- [ ] Types match WGSL vec3<f32> and f32
- [ ] Struct padding considered for GPU layout

---

## T-3.6: scene_material() Switch Function

**Description**: Verify material lookup switch emits for all materials.

**Acceptance Criteria**:
- [ ] Switch generates case for each MaterialNode
- [ ] Material IDs match order in SceneGraph.materials
- [ ] Default case provides fallback material
- [ ] Values interpolated from MaterialNode properties

---

## T-3.7: sd_scene() Entry Point

**Description**: Verify scene entry point generates correctly.

**Structure**:
1. Position variable initialization
2. Domain pipeline application (in order)
3. Primitive SDF evaluations
4. CSG combinations
5. Return vec2(distance * compensation, material_id)

**Acceptance Criteria**:
- [ ] Pipeline applied in SceneGraph.pipeline order
- [ ] All primitives evaluated
- [ ] CSG operations combine distances correctly
- [ ] Compensation factor multiplied into final distance
- [ ] Material ID encoded in y component

---

## T-3.8: Pipeline Expression Builder

**Description**: Verify domain transformation chain builds correctly.

**Test Cases**:
- Empty pipeline: position unchanged
- Single op: `pos = domain_twist(pos, 2.0)`
- Multiple ops: chained in order
- Mixed ops: different op types combined

**Acceptance Criteria**:
- [ ] Order preserved from SceneGraph.pipeline
- [ ] Each op receives output of previous
- [ ] Variable naming consistent
- [ ] No duplicate function definitions

---

## T-3.9: Compensation Factor Calculation

**Description**: Verify overall compensation factor computed for sphere tracing.

**Rules**:
- Isometric ops (repeat, mirror, twist, bend): factor = 1.0
- KIFS: factor = domain_kifs_compensation(folds)
- Stretch: factor = min(stretch_factors)
- Multiple non-isometric: multiply factors together

**Acceptance Criteria**:
- [ ] Isometric-only scenes have no compensation overhead
- [ ] KIFS compensation called when KifsNode present
- [ ] Stretch compensation computed from StretchNode.factors
- [ ] Multiple compensations multiply correctly

---

## T-3.10: Complete WGSL Output

**Description**: Verify full shader output compiles and runs.

**Output Structure**:
1. SDF primitive functions (only those used)
2. Domain operation functions (only those used)
3. Compensation functions (if needed)
4. Material struct and scene_material()
5. sd_scene() entry point

**Acceptance Criteria**:
- [ ] Output is valid WGSL syntax
- [ ] wgpu/dawn compiler accepts output
- [ ] Unused functions not emitted (dead code elimination)
- [ ] No duplicate function definitions
- [ ] Consistent indentation and formatting
