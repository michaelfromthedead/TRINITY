# PHASE 3 TODO: Material Graph and Code Generation

## T-MAT-3.1: Validate Math Node Code Generation

**Description**: Verify all math nodes generate correct GLSL.

**Tasks**:
- [ ] Test Add node: `a + b`
- [ ] Test Subtract node: `a - b`
- [ ] Test Multiply node: `a * b`
- [ ] Test Divide node: `a / b` (with zero-division handling)
- [ ] Test Lerp node: `mix(a, b, t)`
- [ ] Test Clamp node: `clamp(x, min, max)`
- [ ] Test Power node: `pow(base, exp)`
- [ ] Test Min, Max, Abs, Sqrt, Frac, Floor, Ceil

**Acceptance Criteria**:
- Generated GLSL compiles without errors
- Output matches expected mathematical operation
- Type promotion works correctly (float * vec3 = vec3)
- Edge cases handled (divide by zero, negative sqrt)

---

## T-MAT-3.2: Validate Texture Node Code Generation

**Description**: Ensure TextureSampleNode and UVNode generate correct sampling code.

**Tasks**:
- [ ] Test TextureSampleNode with default UVs
- [ ] Test TextureSampleNode with custom UV input
- [ ] Test all output channels (rgba, rgb, r, g, b, a)
- [ ] Test UVNode provides correct coordinate variable

**Acceptance Criteria**:
- Generated code: `texture(tex_name, uv)`
- Channel extraction: `texture(...).rgb`, `.r`, etc.
- Default UV is `v_uv`
- Custom UV input is correctly wired

---

## T-MAT-3.3: Validate Utility Node Code Generation

**Description**: Verify utility nodes generate correct GLSL.

**Tasks**:
- [ ] Test OneMinus node: `1.0 - x`
- [ ] Test ComponentMask node: swizzle operations
- [ ] Test AppendNode: vector construction

**Acceptance Criteria**:
- OneMinus generates `1.0 - input`
- ComponentMask generates correct swizzle (`.xy`, `.rgb`, etc.)
- AppendNode constructs vectors from components

---

## T-MAT-3.4: Validate OutputNode Integration

**Description**: Confirm OutputNode correctly terminates the graph and assigns PBR outputs.

**Tasks**:
- [ ] Test all output slots (base_color, metallic, roughness, normal, emissive, ao, opacity)
- [ ] Verify output variable naming convention
- [ ] Test partial connections (not all slots filled)
- [ ] Verify default values for unconnected slots

**Acceptance Criteria**:
- Connected slots generate assignment statements
- Unconnected slots use sensible defaults
- Output variables match shader expectations
- All 7 PBR outputs are supported

---

## T-MAT-3.5: Validate MaterialGraph Cycle Detection

**Description**: Ensure graph validation correctly detects and rejects cycles.

**Tasks**:
- [ ] Create valid acyclic graph, verify passes validation
- [ ] Create simple cycle (A -> B -> A), verify rejection
- [ ] Create complex cycle (A -> B -> C -> A), verify rejection
- [ ] Test self-loop (A -> A), verify rejection

**Acceptance Criteria**:
- Acyclic graphs pass validation
- Any cycle raises clear error with cycle path
- Validation runs in O(V+E) time
- Error message identifies nodes in cycle

---

## T-MAT-3.6: Validate GraphCompiler Output

**Description**: Verify GraphCompiler produces valid, complete GLSL.

**Tasks**:
- [ ] Test uniform declaration generation
- [ ] Test sampler declaration generation
- [ ] Test code ordering matches topological sort
- [ ] Test complete shader structure (uniforms, samplers, main function)

**Acceptance Criteria**:
- Uniforms declared before use
- Samplers declared before use
- Node code in dependency order
- Generated shader compiles with glslang

---

## T-MAT-3.7: Validate Type Checking

**Description**: Ensure port connections enforce type compatibility.

**Tasks**:
- [ ] Test compatible connections (float -> float, vec3 -> vec3)
- [ ] Test implicit promotion (float -> vec3)
- [ ] Test incompatible connections (vec3 -> float without mask)
- [ ] Verify error messages for type mismatches

**Acceptance Criteria**:
- Compatible types connect without error
- Implicit promotion follows GLSL rules
- Incompatible types rejected at build time
- Errors identify the problematic connection

---

## T-MAT-3.8: End-to-End Graph Compilation Test

**Description**: Build a complete material graph and verify the full compilation pipeline.

**Tasks**:
- [ ] Create a realistic PBR material graph (texture, math, output)
- [ ] Compile to GLSL
- [ ] Verify GLSL compiles with glslang
- [ ] Compare output against hand-written reference

**Acceptance Criteria**:
- Complete graph compiles without error
- Generated GLSL is syntactically valid
- Generated GLSL is semantically equivalent to reference
- No dead code or unused declarations
