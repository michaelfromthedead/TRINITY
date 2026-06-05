# PHASE 1 TODO: AST Node System

## Source File
`engine/rendering/demoscene/ast_nodes.py` (244 lines)

---

## T-1.1: Base ExprNode Implementation

**Description**: Verify base ExprNode class provides complete traversal infrastructure.

**Acceptance Criteria**:
- [ ] walk() yields self first, then recursively yields from all children
- [ ] children() returns tuple of direct child nodes (not deep)
- [ ] pretty() returns formatted multi-line string representation
- [ ] label() returns concise single-line identifier

**Verification**: Unit tests for each method on sample node tree.

---

## T-1.2: Primitive Value Nodes

**Description**: Validate leaf nodes representing primitive values.

**Nodes**:
- FloatNode: Single float value
- Vec3Node: 3D vector (x, y, z)
- PositionNode: Position variable reference

**Acceptance Criteria**:
- [ ] Each node is frozen dataclass
- [ ] children() returns empty tuple for leaf nodes
- [ ] WGSL emission produces valid literals (e.g., "vec3<f32>(1.0, 2.0, 3.0)")

---

## T-1.3: Domain Operation Nodes

**Description**: Verify domain transformation nodes.

**Nodes**:
- RepeatNode: Infinite repetition with cell size
- CellIdNode: Current cell index access
- MirrorNode: Axis-aligned mirroring
- KifsNode: Kaleidoscopic iterated function system (non-isometric)
- TwistNode: Helical twist around axis
- BendNode: Curvature along axis
- StretchNode: Non-uniform scaling (non-isometric)

**Acceptance Criteria**:
- [ ] Each node stores transformation parameters
- [ ] KifsNode and StretchNode flagged as requiring distance compensation
- [ ] children() returns position input if applicable

---

## T-1.4: SDF Primitive Nodes

**Description**: Validate geometric shape nodes.

**Nodes**:
- SphereNode: radius parameter
- BoxNode: half-extents vec3
- TorusNode: major/minor radii
- CylinderNode: height/radius
- ConeNode: height/angle
- PlaneNode: normal/offset
- CapsuleNode: length/radius

**Acceptance Criteria**:
- [ ] Each node stores geometry parameters
- [ ] Parameters match Inigo Quilez function signatures
- [ ] Nodes reference position after domain transformation chain

---

## T-1.5: CSG Combine Nodes

**Description**: Verify boolean operation nodes.

**Nodes**:
- UnionNode: min(d1, d2)
- IntersectionNode: max(d1, d2)
- SubtractionNode: max(d1, -d2)

**Acceptance Criteria**:
- [ ] Each node holds two child SDF expressions
- [ ] children() returns both operands
- [ ] Smooth variants (with blend radius) supported if present

---

## T-1.6: MaterialNode PBR Properties

**Description**: Validate PBR material representation.

**Properties**:
- albedo: RGB color (vec3)
- roughness: 0.0-1.0 float
- metallic: 0.0-1.0 float
- emissive: RGB emission (vec3)
- ambient_occlusion: 0.0-1.0 float

**Acceptance Criteria**:
- [ ] All properties have sensible defaults
- [ ] Material can be associated with specific primitives
- [ ] Multiple materials supported per scene

---

## T-1.7: SceneGraph Container

**Description**: Verify root container holds complete scene.

**Contents**:
- primitives: tuple of SdfPrimitiveNode
- pipeline: tuple of DomainOpNode (ordered transformations)
- materials: tuple of MaterialNode
- name: optional scene identifier

**Acceptance Criteria**:
- [ ] children() returns pipeline + primitives
- [ ] deep_label() produces full scene summary
- [ ] Empty scene (no primitives) handled gracefully
