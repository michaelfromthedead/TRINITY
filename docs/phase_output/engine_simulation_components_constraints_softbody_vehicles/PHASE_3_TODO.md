# PHASE 3 TODO: Soft Body Simulation

**Scope**: engine/simulation/softbody (5 files, ~3,283 lines)  
**Priority**: Medium — Specialized physics for deformables

---

## soft_body_pbd.py (766 lines)

### T-SOFT-PBD1.1: Validate Distance Constraint Projection
- [ ] Test two-particle distance constraint convergence
- [ ] Verify inverse mass weighting
- [ ] Test compliance parameter effect
- [ ] Acceptance: Distance converges to rest length within 1e-4

### T-SOFT-PBD1.2: Test Volume Preservation Constraint
- [ ] Compute volume before and after deformation
- [ ] Verify volume gradient computation
- [ ] Test global volume constraint convergence
- [ ] Acceptance: Volume error < 1% after 1000 substeps

### T-SOFT-PBD1.3: Validate Strain Limiting
- [ ] Test maximum stretch ratio enforcement
- [ ] Verify no over-extension beyond limit
- [ ] Test strain distribution across mesh
- [ ] Acceptance: No edge exceeds strain limit

### T-SOFT-PBD1.4: Test Gauss-Seidel Iteration
- [ ] Verify constraint order affects convergence
- [ ] Test iteration count vs. accuracy tradeoff
- [ ] Compare to Jacobi iteration
- [ ] Acceptance: Gauss-Seidel converges faster than Jacobi

### T-SOFT-PBD1.5: Benchmark PBD Performance
- [ ] Time per iteration for 1K, 10K, 100K particles
- [ ] Profile hot spots
- [ ] Test SIMD vectorization benefit
- [ ] Acceptance: <1ms for 10K particles at 10 iterations

---

## fem_solver.py (724 lines)

### T-SOFT-FEM1.1: Validate Deformation Gradient Computation
- [ ] Test F = Ds * Dm_inv formula
- [ ] Verify identity F for undeformed mesh
- [ ] Test with known deformation
- [ ] Acceptance: F matches expected for test cases

### T-SOFT-FEM1.2: Test Neo-Hookean Material
- [ ] Verify strain energy formula
- [ ] Test stress P = dW/dF
- [ ] Compare to analytical solution for uniaxial stretch
- [ ] Acceptance: Stress matches Neo-Hookean theory

### T-SOFT-FEM1.3: Test Corotational Material
- [ ] Verify polar decomposition extracts R
- [ ] Test linear stress in rotation frame
- [ ] Compare to Neo-Hookean for small deformation
- [ ] Acceptance: R is orthogonal, stress matches theory

### T-SOFT-FEM1.4: Test St. Venant-Kirchhoff Material
- [ ] Verify Green strain computation
- [ ] Test second Piola-Kirchhoff stress
- [ ] Test inversion handling (known limitation)
- [ ] Acceptance: Stress matches StVK theory for small strain

### T-SOFT-FEM1.5: Validate Newton Iteration Convergence
- [ ] Test convergence for simple deformation
- [ ] Monitor residual reduction per iteration
- [ ] Test line search backtracking
- [ ] Acceptance: Converge in <5 iterations for typical cases

### T-SOFT-FEM1.6: Test Polar Decomposition Stability
- [ ] Test with nearly singular F
- [ ] Test with inverted elements
- [ ] Verify SVD fallback for degenerate cases
- [ ] Acceptance: No numerical explosion for degenerate F

---

## shape_matching.py (621 lines)

### T-SOFT-SM1.1: Validate Apq/Aqq Matrix Construction
- [ ] Test formula: `Apq = sum(m * outer(p - cm, q - rest_cm))`
- [ ] Verify rest pose produces identity A
- [ ] Test with known rigid rotation
- [ ] Acceptance: Matrices match expected for test cases

### T-SOFT-SM1.2: Test SVD Polar Decomposition
- [ ] Verify R = U * V^T is orthogonal
- [ ] Test det(R) = 1 (proper rotation)
- [ ] Test with reflection (flip handling)
- [ ] Acceptance: R is proper rotation for all test cases

### T-SOFT-SM1.3: Validate Cluster Generation
- [ ] Test cluster radius parameter
- [ ] Verify overlap percentage
- [ ] Test cluster coverage (all vertices in at least one cluster)
- [ ] Acceptance: Clusters cover mesh with specified overlap

### T-SOFT-SM1.4: Test Skinning Weight Blending
- [ ] Verify weights sum to 1 per vertex
- [ ] Test smooth transition between clusters
- [ ] Test stiffness parameter effect
- [ ] Acceptance: Smooth deformation without discontinuities

### T-SOFT-SM1.5: Benchmark Shape Matching Performance
- [ ] Time per frame for 10, 50, 100 clusters
- [ ] Compare to per-vertex matching
- [ ] Profile SVD computation
- [ ] Acceptance: O(clusters) scaling demonstrated

---

## muscle.py (590 lines)

### T-SOFT-M1.1: Validate Force-Length Relationship
- [ ] Test Gaussian curve shape
- [ ] Verify peak at optimal fiber length
- [ ] Test width parameter effect
- [ ] Acceptance: Force-length curve matches Hill model

### T-SOFT-M1.2: Test Force-Velocity Relationship
- [ ] Test concentric contraction (velocity < 0)
- [ ] Test eccentric contraction (velocity > 0)
- [ ] Verify Hill equation: `fv = (V_max + v) / (V_max - v/a_rel)`
- [ ] Acceptance: Force-velocity curve matches Hill model

### T-SOFT-M1.3: Validate Pennation Angle
- [ ] Test fiber orientation effect
- [ ] Verify force projection to tendon axis
- [ ] Test pennation angle change with length
- [ ] Acceptance: Pennation correctly modifies force direction

### T-SOFT-M1.4: Test Series Elastic Element
- [ ] Verify tendon stiffness contribution
- [ ] Test tendon force-length relationship
- [ ] Test interaction with contractile element
- [ ] Acceptance: Series elastic behavior matches model

### T-SOFT-M1.5: Test Activation Signal
- [ ] Test activation range (0-1)
- [ ] Test activation dynamics (rise/fall time)
- [ ] Verify force scaling with activation
- [ ] Acceptance: Activation correctly scales active force

---

## deformable_mesh.py (582 lines)

### T-SOFT-DM1.1: Validate Barycentric Embedding
- [ ] Test barycentric coordinate computation
- [ ] Verify coordinates sum to 1
- [ ] Test vertex inside tetrahedron detection
- [ ] Acceptance: All surface vertices correctly embedded

### T-SOFT-DM1.2: Test Surface Position Update
- [ ] Verify position interpolation from tet vertices
- [ ] Test with known tet deformation
- [ ] Compare to expected surface position
- [ ] Acceptance: Surface tracks tet mesh correctly

### T-SOFT-DM1.3: Validate Normal Recomputation
- [ ] Test face normal calculation after deformation
- [ ] Verify vertex normal averaging
- [ ] Test normal direction (outward)
- [ ] Acceptance: Normals correct for deformed mesh

### T-SOFT-DM1.4: Test Collision Proxy Generation
- [ ] Verify proxy mesh generated
- [ ] Test proxy accuracy vs. complexity tradeoff
- [ ] Test proxy update frequency
- [ ] Acceptance: Collision proxy tracks surface

---

## Integration Tasks

### T-SOFT-INT1: Solver Selection Interface
- [ ] Design unified soft body interface
- [ ] Test solver switching at runtime
- [ ] Document solver tradeoffs
- [ ] Acceptance: Clean API for solver selection

### T-SOFT-INT2: GPU Acceleration Path
- [ ] Identify parallelization opportunities
- [ ] Design constraint batch structure
- [ ] Document GPU kernel requirements
- [ ] Acceptance: Architecture ready for GPU port

### T-SOFT-INT3: Collision Integration
- [ ] Test soft body vs. rigid body collision
- [ ] Test soft body vs. soft body collision
- [ ] Verify collision response stability
- [ ] Acceptance: Stable collision with all body types

---

## Completion Criteria

Phase 3 is complete when:
1. All T-SOFT-*.* tasks pass acceptance criteria
2. All 5 softbody files have >80% test coverage
3. Volume preservation error < 1% for PBD
4. FEM converges in < 5 Newton iterations
5. Shape matching is O(clusters) verified
6. Muscle force matches Hill model within 5%
