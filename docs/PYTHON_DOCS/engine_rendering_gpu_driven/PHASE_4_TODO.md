# PHASE 4 TODO: Indirect Draw and Bindless Resources

## Summary

Validate and prepare the indirect draw system (indirect_draw.py, 661 lines), bindless resources (bindless.py, 786 lines), and instance batching (instancing.py, 736 lines) for GPU command buffer execution.

---

## T-INDIR-4.1: Validate DrawIndexedIndirectArgs Layout

**Description**: Verify indirect draw arguments match GPU API specifications.

**Acceptance Criteria**:
- [ ] Layout matches VkDrawIndexedIndirectCommand (5 x u32)
- [ ] Field order is correct (index_count, instance_count, first_index, vertex_offset, first_instance)
- [ ] Struct size is 20 bytes
- [ ] Test serialization to bytes
- [ ] Verify alignment requirements

**Files**: `engine/rendering/gpu_driven/indirect_draw.py` lines 50-100

**Estimate**: 1 hour

---

## T-INDIR-4.2: Validate Draw Command Generation

**Description**: Verify draw commands are generated correctly from batches.

**Acceptance Criteria**:
- [ ] Test single batch (single draw command)
- [ ] Test multiple batches (multiple draw commands)
- [ ] Verify index_count matches mesh index count
- [ ] Verify instance_count matches batch size
- [ ] Verify first_index offset is correct
- [ ] Test empty batch handling

**Files**: `engine/rendering/gpu_driven/indirect_draw.py` lines 200-350

**Estimate**: 2 hours

---

## T-INDIR-4.3: Validate Draw Command Compaction

**Description**: Verify contiguous draw ranges are merged correctly.

**Acceptance Criteria**:
- [ ] Test contiguous batches (should merge)
- [ ] Test non-contiguous batches (should not merge)
- [ ] Test different materials (should not merge)
- [ ] Verify merged command has correct index_count sum
- [ ] Verify no commands lost during compaction
- [ ] Benchmark compaction ratio (merged / original)

**Files**: `engine/rendering/gpu_driven/indirect_draw.py` lines 549-636

**Estimate**: 2.5 hours

---

## T-BIND-4.4: Validate Generational Resource Handles

**Description**: Verify generational handles detect stale references.

**Acceptance Criteria**:
- [ ] Allocate handle, verify is_valid returns true
- [ ] Free handle, verify is_valid returns false
- [ ] Reallocate same index, old handle should be invalid
- [ ] Test generation wraparound (if applicable)
- [ ] Test free list reuse
- [ ] Test handle serialization

**Files**: `engine/rendering/gpu_driven/bindless.py` lines 50-150

**Estimate**: 2 hours

---

## T-BIND-4.5: Validate Descriptor Heap Allocation

**Description**: Verify descriptor heap allocation and deallocation is leak-free.

**Acceptance Criteria**:
- [ ] Allocate until heap full, verify correct error
- [ ] Free all, verify all slots available
- [ ] Allocate/free cycle, verify no leaks
- [ ] Test fragmentation resistance
- [ ] Verify descriptor data is preserved
- [ ] Test concurrent allocation patterns

**Files**: `engine/rendering/gpu_driven/bindless.py` lines 200-350

**Estimate**: 2 hours

---

## T-BIND-4.6: Validate Material Resource Table

**Description**: Verify PBR material bindings are correct.

**Acceptance Criteria**:
- [ ] Test material registration with all texture handles
- [ ] Test material lookup by ID
- [ ] Test material update (replace texture handle)
- [ ] Test material removal
- [ ] Verify handle validation on material access
- [ ] Test default material fallback

**Files**: `engine/rendering/gpu_driven/bindless.py` lines 500-650

**Estimate**: 2 hours

---

## T-BIND-4.7: Validate Buffer Device Address (BDA)

**Description**: Verify buffer device address storage and access.

**Acceptance Criteria**:
- [ ] Test BDA allocation for buffer
- [ ] Test BDA with offset calculation
- [ ] Test BDA validation (within buffer bounds)
- [ ] Verify 64-bit address storage
- [ ] Test multiple buffers in same heap
- [ ] Document GPU pointer alignment requirements

**Files**: `engine/rendering/gpu_driven/bindless.py` lines 350-500

**Estimate**: 1.5 hours

---

## T-INST-4.8: Validate Instance Batching

**Description**: Verify instances are grouped correctly by mesh+material.

**Acceptance Criteria**:
- [ ] Test single mesh, single material (one batch)
- [ ] Test single mesh, multiple materials (multiple batches)
- [ ] Test multiple meshes, single material (multiple batches)
- [ ] Verify all instances are in exactly one batch
- [ ] Test empty instance list
- [ ] Benchmark batching performance

**Files**: `engine/rendering/gpu_driven/instancing.py` lines 200-400

**Estimate**: 2 hours

---

## T-INST-4.9: Validate Mat4x4 TRS Construction

**Description**: Verify transform matrix construction from translation, rotation, scale.

**Acceptance Criteria**:
- [ ] Test identity transform
- [ ] Test translation only
- [ ] Test rotation only (quaternion)
- [ ] Test scale only (uniform and non-uniform)
- [ ] Test combined TRS
- [ ] Verify matrix multiplication order

**Files**: `engine/rendering/gpu_driven/instancing.py` lines 50-150

**Estimate**: 1.5 hours

---

## T-INST-4.10: Validate CulledInstanceBatcher

**Description**: Verify instance batching integrates with culling output.

**Acceptance Criteria**:
- [ ] Test with visibility mask (some culled)
- [ ] Test all visible (no culling)
- [ ] Test all culled (empty output)
- [ ] Verify culled instances excluded from batches
- [ ] Test mixed LOD levels
- [ ] Verify correct instance count in draw commands

**Files**: `engine/rendering/gpu_driven/instancing.py` lines 500-650

**Estimate**: 2 hours

---

## T-INDIR-4.11: Create GPU Buffer Layout Specification

**Description**: Document buffer layouts for GPU indirect draw and bindless.

**Acceptance Criteria**:
- [ ] Define indirect draw buffer layout
- [ ] Define instance data buffer layout
- [ ] Define descriptor heap layout
- [ ] Define material table buffer layout
- [ ] Create WGSL struct definitions
- [ ] Document binding group configuration

**Output**: `docs/gpu_driven/indirect_bindless_spec.md`

**Estimate**: 2 hours

---

## T-INDIR-4.12: Write WGSL Indirect/Bindless Shaders

**Description**: Create WGSL shaders for bindless material access.

**Acceptance Criteria**:
- [ ] Bindless texture sampling function
- [ ] Material lookup from instance ID
- [ ] Instance data buffer access
- [ ] Indirect draw compatible vertex shader
- [ ] Proper binding array declarations
- [ ] Test bindless texture fallback

**Output**: `engine/rendering/shaders/bindless/`

**Estimate**: 4 hours

---

## Phase 4 Totals

| Category | Count |
|----------|-------|
| Tasks | 12 |
| Estimated Hours | 24.5 |
| Test Coverage | 10 validation tasks |
| Documentation | 1 spec task |
| Implementation | 1 shader task |
