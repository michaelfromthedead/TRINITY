# PHASE 1 TODO: GPU Cloth Completion

## Objective

Implement actual GPU compute shader dispatch for cloth simulation, replacing the current PARTIAL STUB in `gpu_cloth.py`.

---

## Task 1: wgpu Device Acquisition

**File**: `engine/simulation/cloth/gpu_cloth.py`

**Description**: Implement device and queue acquisition from renderer-backend.

**Acceptance Criteria**:
- [ ] `GPUClothSolver.__init__()` accepts renderer-backend device reference
- [ ] Graceful fallback to CPU solver when GPU unavailable
- [ ] Device capabilities validated (compute shader support)

---

## Task 2: WGSL Shader Conversion

**Files**: `engine/simulation/cloth/gpu_cloth.py`, new `engine/simulation/cloth/shaders/` directory

**Description**: Convert existing GLSL shader templates to WGSL format.

**Shaders to Convert**:
- [ ] `INTEGRATION_SHADER_TEMPLATE` -> `integration.wgsl`
- [ ] `DISTANCE_CONSTRAINT_SHADER_TEMPLATE` -> `distance_constraint.wgsl`
- [ ] `VELOCITY_UPDATE_SHADER_TEMPLATE` -> `velocity_update.wgsl`

**Acceptance Criteria**:
- [ ] Each shader compiles without errors via wgpu
- [ ] Shader logic matches CPU PBD implementation in `cloth_simulation.py`
- [ ] Workgroup sizes configurable (default 256)

---

## Task 3: Buffer Management

**File**: `engine/simulation/cloth/gpu_cloth.py`

**Description**: Implement GPU buffer creation, upload, and download.

**Buffers Required**:
- [ ] `positions_current`: vec4<f32> array
- [ ] `positions_predicted`: vec4<f32> array (double-buffering)
- [ ] `velocities`: vec4<f32> array
- [ ] `inverse_masses`: f32 array
- [ ] `distance_constraints`: packed constraint struct array
- [ ] `params`: uniform buffer for simulation parameters

**Acceptance Criteria**:
- [ ] `upload_mesh()` transfers ClothMesh data to GPU
- [ ] `download_positions()` retrieves positions for CPU collision
- [ ] Buffer reuse on subsequent frames (no per-frame allocation)

---

## Task 4: Compute Pipeline Creation

**File**: `engine/simulation/cloth/gpu_cloth.py`

**Description**: Create compute pipelines for each simulation stage.

**Pipelines**:
- [ ] Integration pipeline (gravity, external forces)
- [ ] Distance constraint pipeline (iterative projection)
- [ ] Velocity update pipeline

**Acceptance Criteria**:
- [ ] Pipelines created once at initialization
- [ ] Bind group layouts match buffer definitions
- [ ] Pipeline caching for shader recompilation avoidance

---

## Task 5: Compute Dispatch Implementation

**File**: `engine/simulation/cloth/gpu_cloth.py`

**Description**: Implement `GPUClothSolver.step()` with actual compute dispatch.

**Dispatch Sequence**:
1. [ ] Dispatch integration shader
2. [ ] For each constraint iteration: dispatch distance constraint shader
3. [ ] Dispatch velocity update shader
4. [ ] Submit command buffer and wait for completion

**Acceptance Criteria**:
- [ ] Removes explicit "does NOT simulate" warning
- [ ] Iteration count configurable (default matches CPU: 4)
- [ ] GPU queue submit properly synchronized

---

## Task 6: CPU Collision Integration

**File**: `engine/simulation/cloth/gpu_cloth.py`

**Description**: Integrate GPU solver with existing CPU collision handler.

**Workflow**:
1. [ ] After velocity update, download positions to CPU
2. [ ] Call `ClothCollisionHandler.resolve_collisions()` on CPU
3. [ ] Upload corrected positions back to GPU

**Acceptance Criteria**:
- [ ] Collision corrections applied correctly
- [ ] No position discontinuities between frames
- [ ] Self-collision handled via existing spatial hash code

---

## Task 7: Interface Compatibility

**File**: `engine/simulation/cloth/gpu_cloth.py`

**Description**: Ensure GPU solver is drop-in replacement for CPU solver.

**Interface Methods**:
- [ ] `step(dt: float)` matches `ClothSimulation.step()` signature
- [ ] `get_positions()` returns current particle positions
- [ ] `set_external_forces()` applies wind/gravity

**Acceptance Criteria**:
- [ ] Factory function selects GPU vs CPU based on availability
- [ ] Switching solvers mid-simulation preserves state
- [ ] All ClothMesh creation methods work with GPU solver

---

## Task 8: Performance Validation

**Description**: Benchmark GPU solver against CPU baseline.

**Benchmarks**:
- [ ] 1K particles: GPU should be faster or equal
- [ ] 10K particles: GPU must be at least 5x faster
- [ ] 50K particles: GPU must handle where CPU cannot

**Acceptance Criteria**:
- [ ] Benchmark script in `tests/benchmarks/`
- [ ] Results logged with particle count, iteration count, and timing
- [ ] No visual artifacts compared to CPU reference

---

## Dependencies

- renderer-backend wgpu device availability
- WGSL shader compilation support in wgpu
- Existing `cloth_collision.py` ClothCollisionHandler

## Estimated Effort

| Task | Complexity | Estimate |
|------|------------|----------|
| Task 1: Device Acquisition | Low | 2 hours |
| Task 2: Shader Conversion | Medium | 4 hours |
| Task 3: Buffer Management | Medium | 4 hours |
| Task 4: Pipeline Creation | Medium | 3 hours |
| Task 5: Dispatch Implementation | High | 6 hours |
| Task 6: Collision Integration | Medium | 4 hours |
| Task 7: Interface Compatibility | Low | 2 hours |
| Task 8: Performance Validation | Medium | 4 hours |
| **Total** | | **29 hours** |
