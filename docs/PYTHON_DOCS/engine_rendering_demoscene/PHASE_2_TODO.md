# PHASE 2 TODO: AST Builder and DSL Parsing

## Source File
`engine/rendering/demoscene/ast_builder.py` (241 lines)

---

## T-2.1: Multi-Dispatch Walk Method

**Description**: Verify AstBuilder.walk() correctly dispatches based on input type.

**Dispatch Cases**:
- ExprNode: Return unchanged
- dict: Lookup type key, dispatch to appropriate builder
- list: Recursively walk, return list of results
- callable: Attempt lambda disassembly
- DSL object: Convert to dict, then walk

**Acceptance Criteria**:
- [ ] ExprNode pass-through preserves node identity
- [ ] Dict with "type": "sdSphere" dispatches to SphereNode builder
- [ ] List of primitives returns list of primitive nodes
- [ ] Lambda `lambda p: sdSphere(p, 1.0)` produces SphereNode
- [ ] Unknown types raise descriptive error

---

## T-2.2: Composition Dispatch Table

**Description**: Verify _COMPOSITION_DISPATCH maps domain operations correctly.

**Mappings Required**:
- "repeat" -> RepeatNode
- "mirror" -> MirrorNode
- "kifs" -> KifsNode
- "twist" -> TwistNode
- "bend" -> BendNode
- "stretch" -> StretchNode

**Acceptance Criteria**:
- [ ] Each domain op name maps to correct node constructor
- [ ] Parameters extracted from dict and passed correctly
- [ ] Missing parameters raise clear errors

---

## T-2.3: Primitive Dispatch Table

**Description**: Verify _PRIMITIVE_DISPATCH maps SDF functions correctly.

**Mappings Required**:
- "sdSphere" -> SphereNode
- "sdBox" -> BoxNode
- "sdTorus" -> TorusNode
- "sdCylinder" -> CylinderNode
- "sdCone" -> ConeNode
- "sdPlane" -> PlaneNode
- "sdCapsule" -> CapsuleNode

**Acceptance Criteria**:
- [ ] Inigo Quilez naming convention preserved
- [ ] Each primitive maps to correct node type
- [ ] Geometry parameters extracted and validated

---

## T-2.4: Marker Dispatch Table

**Description**: Verify _MARKER_DISPATCH handles explicit type markers.

**Purpose**: Allow explicit node construction via type strings when needed.

**Acceptance Criteria**:
- [ ] Fallback dispatch for types not in other tables
- [ ] Extensible for custom node types
- [ ] Clear error for unknown type markers

---

## T-2.5: Lambda Disassembly

**Description**: Verify _disassemble_lambda() extracts SDF expressions from Python lambdas.

**Test Cases**:
- `lambda p: sdSphere(p, 1.0)` -> SphereNode
- `lambda p: sdBox(domain_twist(p, 2.0), vec3(1,1,1))` -> BoxNode with TwistNode chain
- `lambda p: union(sdSphere(p, 1.0), sdBox(p, vec3(1,1,1)))` -> UnionNode

**Acceptance Criteria**:
- [ ] inspect.getsource() retrieves lambda source
- [ ] ast.parse() correctly parses source
- [ ] Lambda with Call body extracts function call
- [ ] ValueError raised for compiled-only code
- [ ] None returned for lambda without Call body

---

## T-2.6: Recursive AST Building from Calls

**Description**: Verify _build_ast_from_call() recursively processes nested function calls.

**Test Cases**:
- Numeric literal 1.0 -> FloatNode(1.0)
- Name 'p' -> PositionNode()
- Call sdSphere(p, 1.0) -> SphereNode(PositionNode(), FloatNode(1.0))
- Nested domain_twist(p, 2.0) -> TwistNode(PositionNode(), FloatNode(2.0))

**Acceptance Criteria**:
- [ ] Numeric literals produce FloatNode
- [ ] Position variable 'p' produces PositionNode
- [ ] Function calls dispatch through primitive/composition tables
- [ ] Nested calls produce correct tree structure
- [ ] vec3(x, y, z) calls produce Vec3Node

---

## T-2.7: walk_composition Method

**Description**: Verify walk_composition() handles domain operation pipelines.

**Purpose**: Build pipeline from sequential domain operation specifications.

**Acceptance Criteria**:
- [ ] List of operations produces ordered pipeline tuple
- [ ] Each operation resolves via _COMPOSITION_DISPATCH
- [ ] Pipeline order preserved for code generation
