# Investigation: engine/simulation/softbody

## Summary
The softbody module contains 3,629 lines of substantive, production-quality soft body physics implementation. All advertised features (FEM solver, shape matching, PBD constraints, muscle simulation, deformable mesh skinning) are fully implemented with real physics algorithms, not stubs. This is a genuine physics simulation codebase.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 83 | REAL | Clean exports of all components |
| `config.py` | 263 | REAL | Material presets (rubber, muscle, fat, jelly, skin, foam, clay), solver configs |
| `fem_solver.py` | 724 | REAL | Complete FEM with Neo-Hookean, Corotational, St. Venant-Kirchhoff materials |
| `shape_matching.py` | 621 | REAL | Muller et al. shape matching with clustered deformation |
| `soft_body_pbd.py` | 766 | REAL | Position-Based Dynamics with volume, strain, edge, collision constraints |
| `muscle.py` | 590 | REAL | Hill-type muscle model with force-length/velocity curves |
| `deformable_mesh.py` | 582 | REAL | Tet-embedded surface extraction and skinning |

## Softbody Components
1. **FEM Solver** - Finite Element Method for accurate tetrahedral deformation
2. **Shape Matching** - Fast geometric deformation (Muller 2005)
3. **PBD** - Position-Based Dynamics for interactive simulation
4. **Muscle Simulation** - Hill-type contraction with fiber tracking
5. **Deformable Mesh** - Surface embedding and GPU skinning
6. **Material Models** - Neo-Hookean, Corotational, St. Venant-Kirchhoff

## Implementation Analysis

### Real FEM? **YES**
- `TetrahedralMesh` with proper rest/deformed vertex tracking
- Deformation gradient computation: `F = Ds @ inv_Dm`
- First Piola-Kirchhoff stress computation
- Lame parameter calculation from Young's modulus and Poisson ratio
- Three material models: Neo-Hookean, Corotational, St. Venant-Kirchhoff
- Inversion handling via SVD singular value clamping
- Semi-implicit Euler integration

### Real Mass-Spring (Edge Constraints)? **YES** (within PBD)
- `EdgeLengthConstraint` maintains rest length between particles
- PBD projection: correction = stiffness * C / w_sum
- Separate compression_stiffness option

### Real Shape Matching? **YES**
- Optimal rigid transformation via polar decomposition of Apq matrix
- SVD-based rotation extraction with proper reflection handling
- Clustered shape matching with overlapping regions
- Linear transform mode for stretch/shear
- Aqq matrix precomputation with regularization for degenerate configs

### Real PBD? **YES**
- `VolumeConstraint` - tetrahedral volume preservation with gradient-based projection
- `StrainLimitConstraint` - SVD-based deformation gradient clamping
- `EdgeLengthConstraint` - distance constraints
- `CollisionConstraint` - plane/sphere collision response
- Iterative substep solver with configurable iterations

### Real Muscle? **YES**
- Hill-type force-length relationship (Gaussian curve at optimal length)
- Force-velocity relationship with concentric/eccentric distinction
- Muscle fibers with direction tracking
- Volume preservation during contraction (radial bulging)
- `MuscleGroup` and `MuscleController` for coordinated activation
- Antagonist inhibition

## Verdict
**REAL IMPLEMENTATION** - Complete, production-quality soft body physics

## Evidence

### FEM Neo-Hookean Stress (fem_solver.py:258-259)
```python
# P = mu * F - mu * F^{-T} + lam * ln(J) * F^{-T}
P = mu * F - mu * F_inv_T + lam * math.log(J) * F_inv_T
```

### Shape Matching Polar Decomposition (shape_matching.py:194-202)
```python
U, sigma, Vt = np.linalg.svd(Apq)
R = U @ Vt

# Ensure proper rotation (det = 1, not -1 for reflection)
if np.linalg.det(R) < 0:
    U[:, -1] *= -1
    R = U @ Vt
```

### PBD Volume Constraint Gradient (soft_body_pbd.py:123-125)
```python
grad0 = -np.cross(d2 - d1, d3 - d1) / 6.0
grad1 = np.cross(d2, d3) / 6.0
grad2 = np.cross(d3, d1) / 6.0
grad3 = np.cross(d1, d2) / 6.0
```

### Hill Muscle Force-Length (muscle.py:291-292)
```python
# Width parameter from config instead of hardcoded 0.45
f_L = math.exp(-((L_norm - 1.0) ** 2) / MUSCLE_FORCE_LENGTH_WIDTH)
```

### Corotational Material (fem_solver.py:302-330)
```python
def _polar_decomposition(self, F: Matrix3x3) -> Tuple[Matrix3x3, Matrix3x3]:
    if self.use_svd:
        U, sigma, Vt = np.linalg.svd(F)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        S = Vt.T @ np.diag(sigma) @ Vt
```

## Quality Indicators
- Proper numpy typing with NDArray annotations
- Dataclasses for clean data structures
- Abstract base classes for material models and constraints
- Configuration externalized to config.py
- Singularity handling (degenerate tetrahedra, inverted elements)
- Volume preservation in multiple forms (PBD constraint, muscle bulging)
- Real physics papers referenced (Muller 2005 for shape matching)
