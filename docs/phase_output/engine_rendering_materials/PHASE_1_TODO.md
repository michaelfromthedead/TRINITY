# PHASE 1 TODO: Core Material System Validation

## T-MAT-1.1: Validate MaterialTemplate API

**Description**: Verify MaterialTemplate correctly defines parameter schemas and default values.

**Tasks**:
- [ ] Write unit tests for MaterialTemplate creation with various parameter types
- [ ] Test parameter schema validation (type checking, range validation)
- [ ] Verify default value assignment and retrieval
- [ ] Test serialization/deserialization of templates

**Acceptance Criteria**:
- All parameter types (float, vec2, vec3, vec4, int, bool) are supported
- Invalid parameter values are rejected with clear errors
- Templates can be saved and loaded without data loss

---

## T-MAT-1.2: Validate MaterialInstance Override Semantics

**Description**: Confirm MaterialInstance correctly overrides template parameters while maintaining weak reference integrity.

**Tasks**:
- [ ] Test instance creation from template
- [ ] Verify parameter override behavior (instance value takes precedence)
- [ ] Test weak reference behavior when template is invalidated
- [ ] Verify dirty flag propagation on parameter change

**Acceptance Criteria**:
- Instances inherit all template parameters by default
- Overridden parameters return instance values
- Weak reference becomes invalid when template is destroyed
- DirtyFlags.PARAMETERS is set on any parameter change

---

## T-MAT-1.3: Validate MaterialFunction Dependencies

**Description**: Ensure MaterialFunction dependency tracking is correct and prevents circular dependencies.

**Tasks**:
- [ ] Test function creation with embedded GLSL code
- [ ] Verify dependency declaration and retrieval
- [ ] Test circular dependency detection
- [ ] Validate topological sort of function dependencies

**Acceptance Criteria**:
- Functions can declare dependencies on other functions
- Circular dependencies raise clear errors
- Dependencies are resolved in correct order for shader inclusion

---

## T-MAT-1.4: Validate MaterialLayer Composition

**Description**: Verify MaterialLayer stacking and blend modes work correctly.

**Tasks**:
- [ ] Test layer creation with various blend modes
- [ ] Verify layer ordering and priority
- [ ] Test layer enable/disable toggle
- [ ] Validate blend mode application (alpha, additive, multiply)

**Acceptance Criteria**:
- Layers composite in declared order
- Blend modes produce expected visual results
- Disabled layers are excluded from composition
- Layer parameters can be animated

---

## T-MAT-1.5: Validate MaterialSystem Registry

**Description**: Confirm MaterialSystem correctly manages templates, instances, and hot-reload.

**Tasks**:
- [ ] Test template registration and lookup
- [ ] Test instance registration and automatic garbage collection
- [ ] Verify hot-reload invalidates affected instances
- [ ] Test concurrent access patterns

**Acceptance Criteria**:
- Templates are retrievable by name after registration
- Unused instances are garbage collected
- Hot-reload triggers DirtyFlags.SHADER on affected instances
- No race conditions under concurrent access

---

## T-MAT-1.6: Validate DirtyFlags Integration

**Description**: Ensure dirty flag system correctly tracks changes across the material hierarchy.

**Tasks**:
- [ ] Test flag setting on parameter change
- [ ] Test flag setting on texture rebind
- [ ] Test flag propagation from template to instances
- [ ] Verify flag clearing after GPU upload

**Acceptance Criteria**:
- PARAMETERS flag set on scalar changes
- TEXTURES flag set on texture binding changes
- SHADER flag set on template recompilation
- Flags can be atomically cleared by renderer
