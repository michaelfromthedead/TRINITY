// SPDX-License-Identifier: MIT
//
// sdf_domain.wgsl -- Domain deformation operations for SDF ray marching.
//
// These functions transform the domain (coordinate space) before SDF
// evaluation, enabling repetition, symmetry, folding, and non-linear
// deformations. Most deformations are isometries (mirror, twist). KIFS and stretch are NOT isometries -- see per-function docs.
//
// Convention: return vec3<f32> for the transformed position. Callers
// apply the transformed position to their SDF primitive directly.
//
// Reference: Inigo Quilez -- Domain Deformations
// https://iquilezles.org/articles/domaindeform/

// =============================================================================
// T-DEMO-1.22: Domain Repetition
// =============================================================================

/// Tiles space by folding p into the cell [0, c) then centering around origin.
/// The returned position is suitable for SDF evaluation at any cell.
///   p        -- input position
///   c        -- cell size along each axis
///   returns  -- position within the centered cell
fn domain_repeat(p: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    // Euclidean modulo expanded form (naga v24 does not support mod() built-in)
    return (p - c * floor(p / c)) - 0.5 * c;
}

/// Returns the integer cell index of p under tiling with cell size c.
/// Use this for pseudo-random variation between cells (seed a hash with
/// the cell ID before evaluating the SDF).
///   p        -- input position
///   c        -- cell size along each axis
///   returns  -- integer cell index (as f32 for WGSL compatibility)
fn domain_cell_id(p: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    return floor(p / c + 0.5);
}

// =============================================================================
// T-DEMO-1.23: Domain Mirroring
// =============================================================================

/// Mirrors space across the YZ plane (x = 0). Points with negative x are
/// reflected to positive x, creating bilateral symmetry.
fn domain_mirror_x(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(abs(p.x), p.y, p.z);
}

/// Mirrors space across the XZ plane (y = 0). Points with negative y are
/// reflected to positive y.
fn domain_mirror_y(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(p.x, abs(p.y), p.z);
}

/// Mirrors space across the XY plane (z = 0). Points with negative z are
/// reflected to positive z.
fn domain_mirror_z(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(p.x, p.y, abs(p.z));
}

// =============================================================================
// T-DEMO-1.24: Kaleidoscopic Fold (KIFS)
// =============================================================================

/// Kaleidoscopic fold: creates rotational symmetry by repeatedly reflecting
/// the xy-plane across fold boundaries and rotating by the fold angle.
/// After `folds` steps, the domain has rotational symmetry of order `folds`.
///
/// The fold is NOT an isometry. abs() reduces distances, so callers
/// should compensate for the non-uniform metric.
/// Combine with domain_stretch or explicit scaling for KIFS fractals.
///
///   p           -- input position
///   folds       -- number of fold segments (e.g., 6 for hexagonal symmetry)
///   returns     -- folded position
fn domain_kifs(p: vec3<f32>, folds: f32) -> vec3<f32> {
    let safe_folds = max(abs(folds), 1.0);
    let angle = 6.283185307179586 / safe_folds; // 2 * PI / safe_folds
    var q = p;
    for (var i = 0u; i < u32(safe_folds); i = i + 1u) {
        // Fold: reflect the half-plane across the x-axis
        q = vec3<f32>(abs(q.x), abs(q.y), q.z);
        // Rotate by one fold angle
        let ca = cos(angle);
        let sa = sin(angle);
        let rx = ca * q.x - sa * q.y;
        let ry = sa * q.x + ca * q.y;
        q = vec3<f32>(rx, ry, q.z);
    }
    return q;
}

// =============================================================================
// T-DEMO-1.25: Twist
// =============================================================================

/// Twists space by rotating the xz-plane proportional to the y-coordinate.
/// The twist angle at height y is k * y radians.
///   p        -- input position
///   k        -- twist rate (radians per unit height)
///   returns  -- twisted position
fn domain_twist(p: vec3<f32>, k: f32) -> vec3<f32> {
    let c = cos(k * p.y);
    let s = sin(k * p.y);
    return vec3<f32>(
        c * p.x - s * p.z,
        p.y,
        s * p.x + c * p.z
    );
}

// =============================================================================
// T-DEMO-1.26: Bend
// =============================================================================

/// Bends the coordinate axes along a circular arc in the xz-plane.
/// The bend maps straight lines parallel to the z-axis to circular arcs
/// of radius r. The x-coordinate becomes radial distance from the arc center.
///
///   p        -- input position
///   r        -- bend radius (larger = gentler curve)
///   returns  -- bent position
fn domain_bend(p: vec3<f32>, r: f32) -> vec3<f32> {
    // Avoid division by zero: clamp radius to epsilon
    let safe_r = max(abs(r), 1e-8);
    let theta = p.x / safe_r;
    let c = cos(theta);
    let s = sin(theta);
    // Bent position (identity for r near zero)
    let bent = vec3<f32>(
        -safe_r + (safe_r + p.z) * c,
        p.y,
        (safe_r + p.z) * s
    );
    // Return p when r is near zero, otherwise the bent position
    return select(bent, p, abs(r) < 1e-8);
}

// =============================================================================
// T-DEMO-1.27: Stretch (Anisotropic Scaling)
// =============================================================================

/// Stretches or compresses space along the x-axis, with inverse scaling
/// on y and z (uniform inverse scaling, determinant = 1/s).
/// The SDF distance should be compensated by min(s, 1/s) due to
/// the non-uniform metric.
///   p        -- input position
///   s        -- stretch factor (>1 stretches, <1 compresses, 1 is identity)
///   returns  -- stretched position
fn domain_stretch_x(p: vec3<f32>, s: f32) -> vec3<f32> {
    return vec3<f32>(p.x * s, p.y / s, p.z / s);
}

/// Stretches or compresses space along the y-axis (uniform inverse scaling, determinant = 1/s).
fn domain_stretch_y(p: vec3<f32>, s: f32) -> vec3<f32> {
    return vec3<f32>(p.x / s, p.y * s, p.z / s);
}

/// Stretches or compresses space along the z-axis (uniform inverse scaling, determinant = 1/s).
fn domain_stretch_z(p: vec3<f32>, s: f32) -> vec3<f32> {
    return vec3<f32>(p.x / s, p.y / s, p.z * s);
}
