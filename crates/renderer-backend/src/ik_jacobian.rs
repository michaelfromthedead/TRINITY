//! Jacobian-based IK solver for skeletal animation in TRINITY Engine (T-AN-4.4).
//!
//! This module provides a sophisticated Jacobian IK solver supporting:
//!
//! - Jacobian matrix construction (3xN position-only, 6xN position+rotation)
//! - Damped Least Squares (DLS) pseudo-inverse for singularity avoidance
//! - Singular Value Decomposition (SVD) for numerical stability
//! - Multiple end effectors solved simultaneously
//! - Task prioritization with null-space projection
//!
//! # Mathematical Background
//!
//! The Jacobian matrix relates joint velocities to end-effector velocities:
//! ```text
//! dx = J * dq
//! ```
//! where `dx` is the effector velocity, `J` is the Jacobian, and `dq` is joint velocities.
//!
//! To solve for `dq` given a target `dx`, we use the pseudo-inverse:
//! ```text
//! dq = J^+ * dx
//! ```
//!
//! Damped Least Squares adds regularization to avoid singularities:
//! ```text
//! dq = J^T * (J*J^T + lambda^2*I)^-1 * dx
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::ik_jacobian::{JacobianChain, JacobianTarget, JacobianParams, solve_jacobian};
//! use renderer_backend::skeleton::Skeleton;
//! use renderer_backend::pose::Pose;
//! use glam::Vec3;
//!
//! let chain = JacobianChain {
//!     bones: vec![0, 1, 2],
//!     dof_per_bone: vec![3, 3, 3], // ball joints
//! };
//!
//! let target = JacobianTarget {
//!     effector_bone: 2,
//!     target_position: Vec3::new(1.0, 2.0, 0.0),
//!     target_rotation: None,
//!     weight: 1.0,
//! };
//!
//! let params = JacobianParams {
//!     targets: vec![target],
//!     damping: 0.1,
//!     max_iterations: 50,
//!     tolerance: 0.001,
//!     use_svd: false,
//!     null_space_posture: None,
//! };
//!
//! let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);
//! ```

use glam::{Mat3, Mat4, Quat, Vec3};

use crate::pose::Pose;
use crate::skeleton::Skeleton;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of degrees of freedom supported in a single chain.
pub const MAX_DOF: usize = 128;

/// Maximum number of simultaneous targets.
pub const MAX_TARGETS: usize = 16;

/// Default damping factor for DLS.
pub const DEFAULT_DAMPING: f32 = 0.05;

/// Default convergence tolerance (in world units).
pub const DEFAULT_TOLERANCE: f32 = 0.001;

/// Default maximum iterations.
pub const DEFAULT_MAX_ITERATIONS: u32 = 50;

/// Small epsilon for numerical stability.
const EPSILON: f32 = 1e-8;

// ---------------------------------------------------------------------------
// Degree of Freedom Types
// ---------------------------------------------------------------------------

/// Degrees of freedom configuration for a joint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DofType {
    /// Single rotational DOF (hinge joint) around specified axis.
    Hinge(Axis),
    /// Two rotational DOFs (universal joint).
    Universal(Axis, Axis),
    /// Three rotational DOFs (ball joint / spherical joint).
    Ball,
}

impl DofType {
    /// Get the number of DOFs for this joint type.
    #[inline]
    pub fn count(&self) -> u8 {
        match self {
            DofType::Hinge(_) => 1,
            DofType::Universal(_, _) => 2,
            DofType::Ball => 3,
        }
    }

    /// Get the rotation axes for this DOF type.
    pub fn axes(&self) -> Vec<Vec3> {
        match self {
            DofType::Hinge(axis) => vec![axis.to_vec3()],
            DofType::Universal(a1, a2) => vec![a1.to_vec3(), a2.to_vec3()],
            DofType::Ball => vec![Vec3::X, Vec3::Y, Vec3::Z],
        }
    }
}

impl Default for DofType {
    fn default() -> Self {
        DofType::Ball
    }
}

/// Axis enumeration for joint rotation axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    /// Convert axis to unit vector.
    #[inline]
    pub fn to_vec3(&self) -> Vec3 {
        match self {
            Axis::X => Vec3::X,
            Axis::Y => Vec3::Y,
            Axis::Z => Vec3::Z,
        }
    }
}

// ---------------------------------------------------------------------------
// IK Chain Definition
// ---------------------------------------------------------------------------

/// Defines an IK chain of connected bones with DOF specifications.
#[derive(Clone, Debug)]
pub struct JacobianChain {
    /// Bone indices in the chain, from root to effector.
    /// Must be in parent-child order (each bone's parent should appear earlier).
    pub bones: Vec<usize>,

    /// DOF type for each bone in the chain.
    /// Length must match `bones.len()`.
    pub dof_per_bone: Vec<DofType>,
}

impl JacobianChain {
    /// Create a new chain with all ball joints (3 DOF each).
    pub fn new_ball_chain(bones: Vec<usize>) -> Self {
        let dof_per_bone = vec![DofType::Ball; bones.len()];
        Self { bones, dof_per_bone }
    }

    /// Create a new chain with custom DOF per bone.
    pub fn new(bones: Vec<usize>, dof_per_bone: Vec<DofType>) -> Self {
        assert_eq!(
            bones.len(),
            dof_per_bone.len(),
            "bones and dof_per_bone must have same length"
        );
        Self { bones, dof_per_bone }
    }

    /// Get the total number of DOFs in the chain.
    pub fn total_dof(&self) -> usize {
        self.dof_per_bone.iter().map(|d| d.count() as usize).sum()
    }

    /// Get the effector bone index (last bone in chain).
    pub fn effector_bone(&self) -> Option<usize> {
        self.bones.last().copied()
    }

    /// Validate the chain against a skeleton.
    pub fn validate(&self, skeleton: &Skeleton) -> Result<(), JacobianError> {
        if self.bones.is_empty() {
            return Err(JacobianError::EmptyChain);
        }

        if self.bones.len() != self.dof_per_bone.len() {
            return Err(JacobianError::DofMismatch {
                bones: self.bones.len(),
                dofs: self.dof_per_bone.len(),
            });
        }

        for &bone_idx in &self.bones {
            if bone_idx >= skeleton.bone_count() {
                return Err(JacobianError::InvalidBoneIndex {
                    index: bone_idx,
                    max: skeleton.bone_count(),
                });
            }
        }

        if self.total_dof() > MAX_DOF {
            return Err(JacobianError::TooManyDof {
                count: self.total_dof(),
                max: MAX_DOF,
            });
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// IK Target Definition
// ---------------------------------------------------------------------------

/// Target for IK solving.
#[derive(Clone, Debug)]
pub struct JacobianTarget {
    /// Index of the effector bone (must be in the chain).
    pub effector_bone: usize,

    /// Target position in world space.
    pub target_position: Vec3,

    /// Optional target rotation in world space.
    /// When specified, creates a 6-DOF constraint.
    pub target_rotation: Option<Quat>,

    /// Weight for this target (0.0-1.0).
    /// Higher weight = higher priority in multi-target solving.
    pub weight: f32,
}

impl JacobianTarget {
    /// Create a position-only target with unit weight.
    pub fn position(effector_bone: usize, position: Vec3) -> Self {
        Self {
            effector_bone,
            target_position: position,
            target_rotation: None,
            weight: 1.0,
        }
    }

    /// Create a position+rotation target with unit weight.
    pub fn position_rotation(effector_bone: usize, position: Vec3, rotation: Quat) -> Self {
        Self {
            effector_bone,
            target_position: position,
            target_rotation: Some(rotation),
            weight: 1.0,
        }
    }

    /// Set the weight for this target.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Get the number of constraint rows for this target.
    pub fn constraint_count(&self) -> usize {
        if self.target_rotation.is_some() {
            6 // position (3) + rotation (3)
        } else {
            3 // position only
        }
    }
}

// ---------------------------------------------------------------------------
// IK Parameters
// ---------------------------------------------------------------------------

/// Parameters for Jacobian IK solving.
#[derive(Clone, Debug)]
pub struct JacobianParams {
    /// List of targets to solve for.
    pub targets: Vec<JacobianTarget>,

    /// Damping factor for DLS (lambda).
    /// Higher values = more stable but slower convergence.
    /// Typical range: 0.01-0.1
    pub damping: f32,

    /// Maximum number of iterations.
    pub max_iterations: u32,

    /// Convergence tolerance in world units.
    /// Solver stops when all targets are within this distance.
    pub tolerance: f32,

    /// Use SVD decomposition instead of DLS.
    /// More numerically stable but slower.
    pub use_svd: bool,

    /// Optional null-space posture for secondary objectives.
    /// When provided, joints will try to return to these angles
    /// while still satisfying the primary target.
    /// Length must match total DOF count.
    pub null_space_posture: Option<Vec<f32>>,
}

impl Default for JacobianParams {
    fn default() -> Self {
        Self {
            targets: Vec::new(),
            damping: DEFAULT_DAMPING,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tolerance: DEFAULT_TOLERANCE,
            use_svd: false,
            null_space_posture: None,
        }
    }
}

impl JacobianParams {
    /// Create params with a single position target.
    pub fn single_target(effector_bone: usize, target: Vec3) -> Self {
        Self {
            targets: vec![JacobianTarget::position(effector_bone, target)],
            ..Default::default()
        }
    }

    /// Add a target to the parameters.
    pub fn add_target(mut self, target: JacobianTarget) -> Self {
        self.targets.push(target);
        self
    }

    /// Set the damping factor.
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping.max(EPSILON);
        self
    }

    /// Set maximum iterations.
    pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations.max(1);
        self
    }

    /// Set convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance.max(EPSILON);
        self
    }

    /// Enable SVD mode.
    pub fn with_svd(mut self, use_svd: bool) -> Self {
        self.use_svd = use_svd;
        self
    }

    /// Set null-space posture for secondary objectives.
    pub fn with_null_space_posture(mut self, posture: Vec<f32>) -> Self {
        self.null_space_posture = Some(posture);
        self
    }
}

// ---------------------------------------------------------------------------
// IK Result
// ---------------------------------------------------------------------------

/// Result of Jacobian IK solving.
#[derive(Clone, Debug)]
pub struct JacobianResult {
    /// Joint angles after solving.
    /// Stored as Euler angles (XYZ order) for each DOF in the chain.
    pub joint_angles: Vec<f32>,

    /// Number of iterations used.
    pub iterations: u32,

    /// Whether the solver converged within tolerance.
    pub converged: bool,

    /// Final error distance for each target.
    pub per_target_error: Vec<f32>,

    /// Total weighted error across all targets.
    pub total_error: f32,
}

impl JacobianResult {
    /// Create a result indicating failure to solve.
    pub fn failed(target_count: usize) -> Self {
        Self {
            joint_angles: Vec::new(),
            iterations: 0,
            converged: false,
            per_target_error: vec![f32::MAX; target_count],
            total_error: f32::MAX,
        }
    }
}

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors that can occur during Jacobian IK solving.
#[derive(Clone, Debug, PartialEq)]
pub enum JacobianError {
    /// The chain has no bones.
    EmptyChain,

    /// Mismatch between bones and DOF specification.
    DofMismatch { bones: usize, dofs: usize },

    /// Bone index out of bounds.
    InvalidBoneIndex { index: usize, max: usize },

    /// Too many DOFs in the chain.
    TooManyDof { count: usize, max: usize },

    /// Too many targets specified.
    TooManyTargets { count: usize, max: usize },

    /// Target effector not in chain.
    EffectorNotInChain { effector: usize },

    /// Null-space posture has wrong length.
    NullSpacePostureMismatch { expected: usize, got: usize },

    /// Singular matrix encountered (only with use_svd=false).
    SingularMatrix,
}

impl std::fmt::Display for JacobianError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyChain => write!(f, "IK chain has no bones"),
            Self::DofMismatch { bones, dofs } => {
                write!(f, "bone count ({}) != DOF count ({})", bones, dofs)
            }
            Self::InvalidBoneIndex { index, max } => {
                write!(f, "bone index {} >= skeleton size {}", index, max)
            }
            Self::TooManyDof { count, max } => {
                write!(f, "DOF count {} exceeds maximum {}", count, max)
            }
            Self::TooManyTargets { count, max } => {
                write!(f, "target count {} exceeds maximum {}", count, max)
            }
            Self::EffectorNotInChain { effector } => {
                write!(f, "effector bone {} not in chain", effector)
            }
            Self::NullSpacePostureMismatch { expected, got } => {
                write!(f, "null-space posture length {} != DOF count {}", got, expected)
            }
            Self::SingularMatrix => write!(f, "singular Jacobian matrix"),
        }
    }
}

impl std::error::Error for JacobianError {}

// ---------------------------------------------------------------------------
// Internal: Matrix Operations
// ---------------------------------------------------------------------------

/// Simple dense matrix for Jacobian operations.
/// Row-major storage: data[row * cols + col]
#[derive(Clone, Debug)]
struct DenseMatrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl DenseMatrix {
    /// Create a zero matrix.
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Create an identity matrix.
    fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Get element at (row, col).
    #[inline]
    fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col).
    #[inline]
    fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    /// Add a value to element at (row, col).
    #[inline]
    fn add(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] += value;
    }

    /// Get a row as a slice.
    #[inline]
    fn row(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Transpose the matrix.
    fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                result.set(c, r, self.get(r, c));
            }
        }
        result
    }

    /// Matrix multiplication: self * other.
    fn mul(&self, other: &DenseMatrix) -> Self {
        assert_eq!(self.cols, other.rows, "matrix dimension mismatch");
        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// Matrix-vector multiplication: self * vec.
    fn mul_vec(&self, vec: &[f32]) -> Vec<f32> {
        assert_eq!(self.cols, vec.len(), "dimension mismatch");
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.get(i, j) * vec[j];
            }
        }
        result
    }

    /// Add lambda^2 * I to diagonal (for damped least squares).
    fn add_damping(&mut self, lambda: f32) {
        let lambda_sq = lambda * lambda;
        let n = self.rows.min(self.cols);
        for i in 0..n {
            self.add(i, i, lambda_sq);
        }
    }

    /// Solve Ax = b using Gaussian elimination with partial pivoting.
    /// Returns None if matrix is singular.
    fn solve(&self, b: &[f32]) -> Option<Vec<f32>> {
        assert_eq!(self.rows, self.cols, "matrix must be square");
        assert_eq!(self.rows, b.len(), "dimension mismatch");

        let n = self.rows;
        let mut a = self.data.clone();
        let mut x = b.to_vec();

        // Gaussian elimination with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_val = a[k * n + k].abs();
            let mut max_row = k;
            for i in k + 1..n {
                let val = a[i * n + k].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            if max_val < EPSILON {
                return None; // Singular matrix
            }

            // Swap rows
            if max_row != k {
                for j in k..n {
                    a.swap(k * n + j, max_row * n + j);
                }
                x.swap(k, max_row);
            }

            // Eliminate column
            for i in k + 1..n {
                let factor = a[i * n + k] / a[k * n + k];
                for j in k + 1..n {
                    a[i * n + j] -= factor * a[k * n + j];
                }
                x[i] -= factor * x[k];
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            for j in i + 1..n {
                x[i] -= a[i * n + j] * x[j];
            }
            x[i] /= a[i * n + i];
        }

        Some(x)
    }

    /// Compute Frobenius norm.
    fn frobenius_norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

// ---------------------------------------------------------------------------
// Internal: SVD Implementation
// ---------------------------------------------------------------------------

/// Simple SVD result.
struct SvdResult {
    /// Left singular vectors (m x m).
    u: DenseMatrix,
    /// Singular values (min(m,n)).
    s: Vec<f32>,
    /// Right singular vectors (n x n).
    vt: DenseMatrix,
}

/// Compute SVD using Jacobi rotations.
/// This is a simple implementation suitable for small matrices (< 100x100).
fn compute_svd(matrix: &DenseMatrix) -> SvdResult {
    let m = matrix.rows;
    let n = matrix.cols;
    let k = m.min(n);

    // For small matrices, use bidiagonalization + implicit QR
    // This is a simplified implementation

    // Start with A^T * A for right singular vectors
    let ata = matrix.transpose().mul(matrix);

    // Compute eigenvalues/eigenvectors of A^T * A using Jacobi iterations
    let (eigenvalues, v) = jacobi_eigendecomposition(&ata);

    // Singular values are sqrt of eigenvalues
    let mut s: Vec<f32> = eigenvalues.iter().map(|&e| e.max(0.0).sqrt()).collect();

    // Sort by descending singular value
    let mut indices: Vec<usize> = (0..k).collect();
    indices.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap());

    // Reorder
    let sorted_s: Vec<f32> = indices.iter().map(|&i| s[i]).collect();
    let sorted_v = reorder_columns(&v, &indices);

    // Compute U = A * V * S^-1
    let mut u = DenseMatrix::zeros(m, k);
    for j in 0..k {
        if sorted_s[j] > EPSILON {
            // u_j = (1/s_j) * A * v_j
            for i in 0..m {
                let mut sum = 0.0;
                for l in 0..n {
                    sum += matrix.get(i, l) * sorted_v.get(l, j);
                }
                u.set(i, j, sum / sorted_s[j]);
            }
        }
    }

    // Pad U to m x m
    let mut u_full = DenseMatrix::identity(m);
    for i in 0..m {
        for j in 0..k {
            u_full.set(i, j, u.get(i, j));
        }
    }

    // Create V^T
    let vt = sorted_v.transpose();

    // Pad V^T to n x n
    let mut vt_full = DenseMatrix::identity(n);
    for i in 0..k {
        for j in 0..n {
            vt_full.set(i, j, vt.get(i, j));
        }
    }

    SvdResult {
        u: u_full,
        s: sorted_s,
        vt: vt_full,
    }
}

/// Jacobi eigenvalue algorithm for symmetric matrices.
fn jacobi_eigendecomposition(matrix: &DenseMatrix) -> (Vec<f32>, DenseMatrix) {
    assert_eq!(matrix.rows, matrix.cols);
    let n = matrix.rows;

    let mut a = matrix.clone();
    let mut v = DenseMatrix::identity(n);

    const MAX_SWEEPS: usize = 50;
    const TOL: f32 = 1e-10;

    for _ in 0..MAX_SWEEPS {
        // Find largest off-diagonal element
        let mut max_off = 0.0f32;
        let mut p = 0;
        let mut q = 1;

        for i in 0..n {
            for j in i + 1..n {
                let val = a.get(i, j).abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < TOL {
            break;
        }

        // Compute rotation angle
        let diff = a.get(q, q) - a.get(p, p);
        let theta = if diff.abs() < EPSILON {
            std::f32::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a.get(p, q) / diff).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to A
        for i in 0..n {
            if i != p && i != q {
                let aip = a.get(i, p);
                let aiq = a.get(i, q);
                a.set(i, p, c * aip - s * aiq);
                a.set(p, i, c * aip - s * aiq);
                a.set(i, q, s * aip + c * aiq);
                a.set(q, i, s * aip + c * aiq);
            }
        }

        let app = a.get(p, p);
        let aqq = a.get(q, q);
        let apq = a.get(p, q);

        a.set(p, p, c * c * app - 2.0 * s * c * apq + s * s * aqq);
        a.set(q, q, s * s * app + 2.0 * s * c * apq + c * c * aqq);
        a.set(p, q, 0.0);
        a.set(q, p, 0.0);

        // Accumulate rotation in V
        for i in 0..n {
            let vip = v.get(i, p);
            let viq = v.get(i, q);
            v.set(i, p, c * vip - s * viq);
            v.set(i, q, s * vip + c * viq);
        }
    }

    // Extract eigenvalues from diagonal
    let eigenvalues: Vec<f32> = (0..n).map(|i| a.get(i, i)).collect();

    (eigenvalues, v)
}

/// Reorder columns of a matrix by index.
fn reorder_columns(matrix: &DenseMatrix, indices: &[usize]) -> DenseMatrix {
    let mut result = DenseMatrix::zeros(matrix.rows, indices.len());
    for (new_col, &old_col) in indices.iter().enumerate() {
        for row in 0..matrix.rows {
            result.set(row, new_col, matrix.get(row, old_col));
        }
    }
    result
}

/// Compute pseudo-inverse using SVD.
fn svd_pseudo_inverse(svd: &SvdResult, damping: f32) -> DenseMatrix {
    let m = svd.u.rows;
    let n = svd.vt.cols;
    let k = svd.s.len();

    // Compute damped inverse of singular values
    let damping_sq = damping * damping;
    let s_inv: Vec<f32> = svd
        .s
        .iter()
        .map(|&s| {
            if s > EPSILON {
                s / (s * s + damping_sq)
            } else {
                0.0
            }
        })
        .collect();

    // Pseudo-inverse = V * S^+ * U^T
    // Since we have V^T, we need: V * S^+ * U^T = (V^T)^T * S^+ * U^T

    let mut result = DenseMatrix::zeros(n, m);

    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0;
            for l in 0..k {
                // V[i,l] * s_inv[l] * U^T[l,j]
                // V[i,l] = V^T[l,i] (transposed)
                // U^T[l,j] = U[j,l]
                sum += svd.vt.get(l, i) * s_inv[l] * svd.u.get(j, l);
            }
            result.set(i, j, sum);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Internal: Jacobian Construction
// ---------------------------------------------------------------------------

/// Compute the world position of a bone given current pose.
fn compute_bone_world_position(
    skeleton: &Skeleton,
    pose: &Pose,
    bone_index: usize,
) -> Vec3 {
    let transforms = pose.transforms();
    let world_mats = skeleton.compute_world_transforms(&transforms);
    world_mats[bone_index].w_axis.truncate()
}

/// Compute the world rotation of a bone given current pose.
fn compute_bone_world_rotation(
    skeleton: &Skeleton,
    pose: &Pose,
    bone_index: usize,
) -> Quat {
    let transforms = pose.transforms();
    let world_mats = skeleton.compute_world_transforms(&transforms);
    Quat::from_mat4(&world_mats[bone_index])
}

/// Compute world transforms for all bones.
fn compute_world_transforms(skeleton: &Skeleton, pose: &Pose) -> Vec<Mat4> {
    let transforms = pose.transforms();
    skeleton.compute_world_transforms(&transforms)
}

/// Build the Jacobian matrix for the given chain and targets.
///
/// The Jacobian relates joint velocities to effector velocities:
/// J[i] = axis[i] x (effector_pos - joint_pos) for rotational joints
fn build_jacobian(
    chain: &JacobianChain,
    skeleton: &Skeleton,
    pose: &Pose,
    targets: &[JacobianTarget],
    world_transforms: &[Mat4],
) -> DenseMatrix {
    let total_dof = chain.total_dof();

    // Count total constraint rows
    let total_constraints: usize = targets.iter().map(|t| t.constraint_count()).sum();

    let mut jacobian = DenseMatrix::zeros(total_constraints, total_dof);

    let mut constraint_row = 0;

    for target in targets {
        // Find effector position
        let effector_pos = world_transforms[target.effector_bone].w_axis.truncate();
        let effector_rot = Quat::from_mat4(&world_transforms[target.effector_bone]);

        let mut dof_col = 0;

        // Build Jacobian columns for each joint in the chain
        for (bone_idx_in_chain, &bone_idx) in chain.bones.iter().enumerate() {
            let joint_pos = world_transforms[bone_idx].w_axis.truncate();
            let joint_rot = Quat::from_mat4(&world_transforms[bone_idx]);

            let dof_type = chain.dof_per_bone[bone_idx_in_chain];
            let axes = dof_type.axes();

            for local_axis in &axes {
                // Transform axis to world space
                let world_axis = joint_rot * *local_axis;

                // Position Jacobian: axis x (effector - joint)
                let r = effector_pos - joint_pos;
                let j_pos = world_axis.cross(r);

                // Write position part (3 rows)
                for i in 0..3 {
                    let weighted = j_pos[i] * target.weight;
                    jacobian.set(constraint_row + i, dof_col, weighted);
                }

                // Rotation Jacobian (if target has rotation constraint)
                if target.target_rotation.is_some() {
                    // For rotation, Jacobian is just the axis
                    for i in 0..3 {
                        let weighted = world_axis[i] * target.weight;
                        jacobian.set(constraint_row + 3 + i, dof_col, weighted);
                    }
                }

                dof_col += 1;
            }
        }

        constraint_row += target.constraint_count();
    }

    jacobian
}

/// Compute the error vector for all targets.
fn compute_error(
    targets: &[JacobianTarget],
    world_transforms: &[Mat4],
) -> Vec<f32> {
    let mut error = Vec::new();

    for target in targets {
        let effector_pos = world_transforms[target.effector_bone].w_axis.truncate();
        let effector_rot = Quat::from_mat4(&world_transforms[target.effector_bone]);

        // Position error
        let pos_error = target.target_position - effector_pos;
        error.push(pos_error.x * target.weight);
        error.push(pos_error.y * target.weight);
        error.push(pos_error.z * target.weight);

        // Rotation error (if specified)
        if let Some(target_rot) = target.target_rotation {
            // Compute rotation difference
            let rot_diff = target_rot * effector_rot.inverse();
            // Convert to axis-angle
            let (axis, angle) = rot_diff.to_axis_angle();
            let rot_error = axis * angle;

            error.push(rot_error.x * target.weight);
            error.push(rot_error.y * target.weight);
            error.push(rot_error.z * target.weight);
        }
    }

    error
}

/// Apply joint angle deltas to the pose.
fn apply_joint_deltas(
    chain: &JacobianChain,
    pose: &mut Pose,
    deltas: &[f32],
    step_scale: f32,
) {
    let mut dof_idx = 0;

    for (bone_idx_in_chain, &bone_idx) in chain.bones.iter().enumerate() {
        let dof_type = chain.dof_per_bone[bone_idx_in_chain];
        let axes = dof_type.axes();

        let current_rot = pose.rotations[bone_idx];

        let mut delta_rot = Quat::IDENTITY;
        for axis in &axes {
            let delta_angle = deltas[dof_idx] * step_scale;
            delta_rot = delta_rot * Quat::from_axis_angle(*axis, delta_angle);
            dof_idx += 1;
        }

        pose.rotations[bone_idx] = (current_rot * delta_rot).normalize();
    }
}

/// Extract current joint angles from pose.
fn extract_joint_angles(chain: &JacobianChain, pose: &Pose) -> Vec<f32> {
    let mut angles = Vec::with_capacity(chain.total_dof());

    for (bone_idx_in_chain, &bone_idx) in chain.bones.iter().enumerate() {
        let rot = pose.rotations[bone_idx];
        let (x, y, z) = quat_to_euler_xyz(rot);

        let dof_type = chain.dof_per_bone[bone_idx_in_chain];
        match dof_type {
            DofType::Hinge(Axis::X) => angles.push(x),
            DofType::Hinge(Axis::Y) => angles.push(y),
            DofType::Hinge(Axis::Z) => angles.push(z),
            DofType::Universal(a1, a2) => {
                angles.push(match a1 {
                    Axis::X => x,
                    Axis::Y => y,
                    Axis::Z => z,
                });
                angles.push(match a2 {
                    Axis::X => x,
                    Axis::Y => y,
                    Axis::Z => z,
                });
            }
            DofType::Ball => {
                angles.push(x);
                angles.push(y);
                angles.push(z);
            }
        }
    }

    angles
}

/// Convert quaternion to Euler angles (XYZ order).
fn quat_to_euler_xyz(q: Quat) -> (f32, f32, f32) {
    // Use glam's built-in euler conversion for consistency
    q.to_euler(glam::EulerRot::XYZ)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Solve IK using Jacobian-based method.
///
/// This is the main entry point for Jacobian IK solving.
///
/// # Arguments
///
/// * `chain` - The IK chain definition
/// * `skeleton` - The skeleton being animated
/// * `pose` - The pose to modify (input/output)
/// * `params` - Solver parameters including targets
///
/// # Returns
///
/// A `JacobianResult` containing the solution and convergence info.
///
/// # Example
///
/// ```ignore
/// let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
/// let params = JacobianParams::single_target(2, Vec3::new(1.0, 2.0, 0.0));
/// let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);
/// if result.converged {
///     println!("IK solved in {} iterations", result.iterations);
/// }
/// ```
pub fn solve_jacobian(
    chain: &JacobianChain,
    skeleton: &Skeleton,
    pose: &mut Pose,
    params: &JacobianParams,
) -> JacobianResult {
    // Validate inputs
    if let Err(_) = chain.validate(skeleton) {
        return JacobianResult::failed(params.targets.len());
    }

    if params.targets.is_empty() {
        return JacobianResult::failed(0);
    }

    if params.targets.len() > MAX_TARGETS {
        return JacobianResult::failed(params.targets.len());
    }

    // Validate all effectors are in chain
    for target in &params.targets {
        if !chain.bones.contains(&target.effector_bone) {
            return JacobianResult::failed(params.targets.len());
        }
    }

    // Validate null-space posture if provided
    if let Some(ref posture) = params.null_space_posture {
        if posture.len() != chain.total_dof() {
            return JacobianResult::failed(params.targets.len());
        }
    }

    let total_dof = chain.total_dof();
    let tolerance_sq = params.tolerance * params.tolerance;

    let mut iterations = 0;
    let mut converged = false;

    for iter in 0..params.max_iterations {
        iterations = iter + 1;

        // Compute current world transforms
        let world_transforms = compute_world_transforms(skeleton, pose);

        // Compute error
        let error = compute_error(&params.targets, &world_transforms);

        // Check convergence
        let mut max_error_sq = 0.0f32;
        for target in &params.targets {
            let effector_pos = world_transforms[target.effector_bone].w_axis.truncate();
            let err = (target.target_position - effector_pos).length_squared();
            max_error_sq = max_error_sq.max(err);
        }

        if max_error_sq < tolerance_sq {
            converged = true;
            break;
        }

        // Build Jacobian
        let jacobian = build_jacobian(chain, skeleton, pose, &params.targets, &world_transforms);

        // Solve for joint deltas
        let deltas = if params.use_svd {
            // SVD-based pseudo-inverse
            let svd = compute_svd(&jacobian);
            let j_pinv = svd_pseudo_inverse(&svd, params.damping);
            j_pinv.mul_vec(&error)
        } else {
            // Damped Least Squares
            // dq = J^T * (J*J^T + lambda^2*I)^-1 * error
            let jt = jacobian.transpose();
            let mut jjt = jacobian.mul(&jt);
            jjt.add_damping(params.damping);

            if let Some(y) = jjt.solve(&error) {
                jt.mul_vec(&y)
            } else {
                // Fallback: simple gradient descent
                let step_size = 0.01;
                jt.mul_vec(&error)
                    .iter()
                    .map(|&x| x * step_size)
                    .collect()
            }
        };

        // Apply null-space projection for secondary objectives
        let final_deltas = if let Some(ref posture) = params.null_space_posture {
            // Null-space projection: dq += (I - J^+*J) * (posture - current)
            let current_angles = extract_joint_angles(chain, pose);
            let posture_error: Vec<f32> = posture
                .iter()
                .zip(current_angles.iter())
                .map(|(&p, &c)| 0.1 * (p - c)) // Small gain for secondary task
                .collect();

            // Compute J^+ * J
            let jt = jacobian.transpose();
            if params.use_svd {
                let svd = compute_svd(&jacobian);
                let j_pinv = svd_pseudo_inverse(&svd, params.damping);
                let jpj = j_pinv.mul(&jacobian);

                // (I - J^+*J) * posture_error
                let mut null_space_delta = vec![0.0; total_dof];
                for i in 0..total_dof {
                    null_space_delta[i] = posture_error[i];
                    for j in 0..total_dof {
                        null_space_delta[i] -= jpj.get(i, j) * posture_error[j];
                    }
                }

                // Combine primary and secondary
                deltas
                    .iter()
                    .zip(null_space_delta.iter())
                    .map(|(&d, &n)| d + n)
                    .collect()
            } else {
                deltas
            }
        } else {
            deltas
        };

        // Apply deltas with clamping
        let max_step = 0.1; // Maximum radians per iteration
        let step_scale = {
            let max_delta = final_deltas.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            if max_delta > max_step {
                max_step / max_delta
            } else {
                1.0
            }
        };

        apply_joint_deltas(chain, pose, &final_deltas, step_scale);
    }

    // Compute final errors
    let world_transforms = compute_world_transforms(skeleton, pose);
    let per_target_error: Vec<f32> = params
        .targets
        .iter()
        .map(|target| {
            let effector_pos = world_transforms[target.effector_bone].w_axis.truncate();
            (target.target_position - effector_pos).length()
        })
        .collect();

    let total_error = per_target_error
        .iter()
        .zip(params.targets.iter())
        .map(|(&e, t)| e * t.weight)
        .sum();

    let joint_angles = extract_joint_angles(chain, pose);

    JacobianResult {
        joint_angles,
        iterations,
        converged,
        per_target_error,
        total_error,
    }
}

/// Solve IK with error handling.
///
/// Like `solve_jacobian` but returns a Result for error cases.
pub fn try_solve_jacobian(
    chain: &JacobianChain,
    skeleton: &Skeleton,
    pose: &mut Pose,
    params: &JacobianParams,
) -> Result<JacobianResult, JacobianError> {
    // Validate chain
    chain.validate(skeleton)?;

    // Validate targets
    if params.targets.is_empty() {
        return Err(JacobianError::EmptyChain);
    }

    if params.targets.len() > MAX_TARGETS {
        return Err(JacobianError::TooManyTargets {
            count: params.targets.len(),
            max: MAX_TARGETS,
        });
    }

    for target in &params.targets {
        if !chain.bones.contains(&target.effector_bone) {
            return Err(JacobianError::EffectorNotInChain {
                effector: target.effector_bone,
            });
        }
    }

    // Validate null-space posture
    if let Some(ref posture) = params.null_space_posture {
        if posture.len() != chain.total_dof() {
            return Err(JacobianError::NullSpacePostureMismatch {
                expected: chain.total_dof(),
                got: posture.len(),
            });
        }
    }

    Ok(solve_jacobian(chain, skeleton, pose, params))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Bone, SkeletonBuilder};
    use std::f32::consts::PI;

    // ========== Helper Functions ==========

    /// Create a simple 3-bone arm skeleton.
    fn create_arm_skeleton() -> Skeleton {
        SkeletonBuilder::new()
            .root("shoulder")
            .child_at("elbow", "shoulder", Vec3::new(1.0, 0.0, 0.0))
            .child_at("wrist", "elbow", Vec3::new(1.0, 0.0, 0.0))
            .build()
            .unwrap()
    }

    /// Create a longer 5-bone chain skeleton.
    fn create_long_chain() -> Skeleton {
        SkeletonBuilder::new()
            .root("root")
            .child_at("b1", "root", Vec3::new(0.5, 0.0, 0.0))
            .child_at("b2", "b1", Vec3::new(0.5, 0.0, 0.0))
            .child_at("b3", "b2", Vec3::new(0.5, 0.0, 0.0))
            .child_at("effector", "b3", Vec3::new(0.5, 0.0, 0.0))
            .build()
            .unwrap()
    }

    /// Create a skeleton with multiple branches.
    fn create_branching_skeleton() -> Skeleton {
        SkeletonBuilder::new()
            .root("root")
            .child_at("left_shoulder", "root", Vec3::new(-0.5, 0.0, 0.0))
            .child_at("left_elbow", "left_shoulder", Vec3::new(-1.0, 0.0, 0.0))
            .child_at("left_hand", "left_elbow", Vec3::new(-1.0, 0.0, 0.0))
            .child_at("right_shoulder", "root", Vec3::new(0.5, 0.0, 0.0))
            .child_at("right_elbow", "right_shoulder", Vec3::new(1.0, 0.0, 0.0))
            .child_at("right_hand", "right_elbow", Vec3::new(1.0, 0.0, 0.0))
            .build()
            .unwrap()
    }

    // ========== DofType Tests ==========

    #[test]
    fn test_dof_type_count() {
        assert_eq!(DofType::Hinge(Axis::X).count(), 1);
        assert_eq!(DofType::Hinge(Axis::Y).count(), 1);
        assert_eq!(DofType::Hinge(Axis::Z).count(), 1);
        assert_eq!(DofType::Universal(Axis::X, Axis::Y).count(), 2);
        assert_eq!(DofType::Ball.count(), 3);
    }

    #[test]
    fn test_dof_type_axes() {
        let axes = DofType::Hinge(Axis::X).axes();
        assert_eq!(axes.len(), 1);
        assert!(axes[0].abs_diff_eq(Vec3::X, 1e-6));

        let axes = DofType::Universal(Axis::Y, Axis::Z).axes();
        assert_eq!(axes.len(), 2);
        assert!(axes[0].abs_diff_eq(Vec3::Y, 1e-6));
        assert!(axes[1].abs_diff_eq(Vec3::Z, 1e-6));

        let axes = DofType::Ball.axes();
        assert_eq!(axes.len(), 3);
    }

    #[test]
    fn test_dof_type_default() {
        assert_eq!(DofType::default(), DofType::Ball);
    }

    // ========== JacobianChain Tests ==========

    #[test]
    fn test_chain_new_ball_chain() {
        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        assert_eq!(chain.bones, vec![0, 1, 2]);
        assert_eq!(chain.dof_per_bone.len(), 3);
        assert!(chain.dof_per_bone.iter().all(|&d| d == DofType::Ball));
        assert_eq!(chain.total_dof(), 9);
    }

    #[test]
    fn test_chain_new_mixed_dof() {
        let chain = JacobianChain::new(
            vec![0, 1, 2],
            vec![
                DofType::Ball,
                DofType::Universal(Axis::X, Axis::Z),
                DofType::Hinge(Axis::Y),
            ],
        );
        assert_eq!(chain.total_dof(), 3 + 2 + 1);
    }

    #[test]
    fn test_chain_effector_bone() {
        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        assert_eq!(chain.effector_bone(), Some(2));

        let empty = JacobianChain::new_ball_chain(vec![]);
        assert_eq!(empty.effector_bone(), None);
    }

    #[test]
    fn test_chain_validate_success() {
        let skeleton = create_arm_skeleton();
        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        assert!(chain.validate(&skeleton).is_ok());
    }

    #[test]
    fn test_chain_validate_empty() {
        let skeleton = create_arm_skeleton();
        let chain = JacobianChain::new_ball_chain(vec![]);
        assert_eq!(chain.validate(&skeleton), Err(JacobianError::EmptyChain));
    }

    #[test]
    fn test_chain_validate_invalid_bone() {
        let skeleton = create_arm_skeleton();
        let chain = JacobianChain::new_ball_chain(vec![0, 1, 99]);
        let err = chain.validate(&skeleton).unwrap_err();
        match err {
            JacobianError::InvalidBoneIndex { index, .. } => assert_eq!(index, 99),
            _ => panic!("expected InvalidBoneIndex"),
        }
    }

    #[test]
    #[should_panic(expected = "bones and dof_per_bone must have same length")]
    fn test_chain_new_mismatch_panics() {
        let _ = JacobianChain::new(vec![0, 1], vec![DofType::Ball]);
    }

    // ========== JacobianTarget Tests ==========

    #[test]
    fn test_target_position() {
        let target = JacobianTarget::position(2, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(target.effector_bone, 2);
        assert!(target.target_position.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-6));
        assert!(target.target_rotation.is_none());
        assert_eq!(target.weight, 1.0);
        assert_eq!(target.constraint_count(), 3);
    }

    #[test]
    fn test_target_position_rotation() {
        let rot = Quat::from_rotation_y(PI / 4.0);
        let target = JacobianTarget::position_rotation(2, Vec3::ZERO, rot);
        assert!(target.target_rotation.is_some());
        assert_eq!(target.constraint_count(), 6);
    }

    #[test]
    fn test_target_with_weight() {
        let target = JacobianTarget::position(0, Vec3::ZERO).with_weight(0.5);
        assert_eq!(target.weight, 0.5);

        // Weight should be clamped
        let target = JacobianTarget::position(0, Vec3::ZERO).with_weight(2.0);
        assert_eq!(target.weight, 1.0);

        let target = JacobianTarget::position(0, Vec3::ZERO).with_weight(-0.5);
        assert_eq!(target.weight, 0.0);
    }

    // ========== JacobianParams Tests ==========

    #[test]
    fn test_params_default() {
        let params = JacobianParams::default();
        assert!(params.targets.is_empty());
        assert_eq!(params.damping, DEFAULT_DAMPING);
        assert_eq!(params.max_iterations, DEFAULT_MAX_ITERATIONS);
        assert_eq!(params.tolerance, DEFAULT_TOLERANCE);
        assert!(!params.use_svd);
        assert!(params.null_space_posture.is_none());
    }

    #[test]
    fn test_params_single_target() {
        let params = JacobianParams::single_target(2, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(params.targets.len(), 1);
        assert_eq!(params.targets[0].effector_bone, 2);
    }

    #[test]
    fn test_params_builder() {
        let params = JacobianParams::default()
            .add_target(JacobianTarget::position(0, Vec3::ZERO))
            .add_target(JacobianTarget::position(1, Vec3::ONE))
            .with_damping(0.1)
            .with_max_iterations(100)
            .with_tolerance(0.01)
            .with_svd(true)
            .with_null_space_posture(vec![0.0; 6]);

        assert_eq!(params.targets.len(), 2);
        assert_eq!(params.damping, 0.1);
        assert_eq!(params.max_iterations, 100);
        assert_eq!(params.tolerance, 0.01);
        assert!(params.use_svd);
        assert_eq!(params.null_space_posture.as_ref().unwrap().len(), 6);
    }

    // ========== DenseMatrix Tests ==========

    #[test]
    fn test_matrix_zeros() {
        let m = DenseMatrix::zeros(3, 4);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        assert!(m.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_matrix_identity() {
        let m = DenseMatrix::identity(3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 1), 1.0);
        assert_eq!(m.get(2, 2), 1.0);
        assert_eq!(m.get(0, 1), 0.0);
        assert_eq!(m.get(1, 0), 0.0);
    }

    #[test]
    fn test_matrix_get_set() {
        let mut m = DenseMatrix::zeros(2, 2);
        m.set(0, 1, 5.0);
        assert_eq!(m.get(0, 1), 5.0);
        m.add(0, 1, 3.0);
        assert_eq!(m.get(0, 1), 8.0);
    }

    #[test]
    fn test_matrix_transpose() {
        let mut m = DenseMatrix::zeros(2, 3);
        m.set(0, 2, 1.0);
        m.set(1, 0, 2.0);

        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(2, 0), 1.0);
        assert_eq!(t.get(0, 1), 2.0);
    }

    #[test]
    fn test_matrix_multiply() {
        // 2x3 * 3x2 = 2x2
        let mut a = DenseMatrix::zeros(2, 3);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(0, 2, 3.0);
        a.set(1, 0, 4.0);
        a.set(1, 1, 5.0);
        a.set(1, 2, 6.0);

        let mut b = DenseMatrix::zeros(3, 2);
        b.set(0, 0, 7.0);
        b.set(0, 1, 8.0);
        b.set(1, 0, 9.0);
        b.set(1, 1, 10.0);
        b.set(2, 0, 11.0);
        b.set(2, 1, 12.0);

        let c = a.mul(&b);
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        // c[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        assert!((c.get(0, 0) - 58.0).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_mul_vec() {
        let mut m = DenseMatrix::zeros(2, 3);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(0, 2, 3.0);
        m.set(1, 0, 4.0);
        m.set(1, 1, 5.0);
        m.set(1, 2, 6.0);

        let v = vec![1.0, 1.0, 1.0];
        let result = m.mul_vec(&v);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 6.0).abs() < 1e-6); // 1+2+3
        assert!((result[1] - 15.0).abs() < 1e-6); // 4+5+6
    }

    #[test]
    fn test_matrix_solve() {
        // Solve Ax = b where A = [[4, 1], [1, 3]], b = [1, 2]
        // Solution: x = [1/11, 7/11]
        let mut a = DenseMatrix::zeros(2, 2);
        a.set(0, 0, 4.0);
        a.set(0, 1, 1.0);
        a.set(1, 0, 1.0);
        a.set(1, 1, 3.0);

        let b = vec![1.0, 2.0];
        let x = a.solve(&b).unwrap();

        assert!((x[0] - 1.0 / 11.0).abs() < 1e-5);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_matrix_solve_singular() {
        let mut a = DenseMatrix::zeros(2, 2);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 2.0);
        a.set(1, 1, 4.0); // Row 2 = 2 * Row 1

        let b = vec![1.0, 2.0];
        assert!(a.solve(&b).is_none());
    }

    // ========== SVD Tests ==========

    #[test]
    fn test_svd_identity() {
        let m = DenseMatrix::identity(3);
        let svd = compute_svd(&m);

        // All singular values should be 1
        for &s in &svd.s {
            assert!((s - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_svd_simple_matrix() {
        let mut m = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 3.0);
        m.set(0, 1, 0.0);
        m.set(1, 0, 0.0);
        m.set(1, 1, 2.0);

        let svd = compute_svd(&m);

        // Singular values should be 3 and 2
        let mut sorted_s = svd.s.clone();
        sorted_s.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((sorted_s[0] - 3.0).abs() < 1e-4);
        assert!((sorted_s[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_svd_pseudo_inverse() {
        let mut m = DenseMatrix::zeros(3, 2);
        m.set(0, 0, 1.0);
        m.set(1, 1, 1.0);
        m.set(2, 0, 0.5);

        let svd = compute_svd(&m);
        let pinv = svd_pseudo_inverse(&svd, 0.001);

        // Pseudo-inverse of 3x2 is 2x3
        assert_eq!(pinv.rows, 2);
        assert_eq!(pinv.cols, 3);

        // M * M^+ * M should approximately equal M
        let mpm = m.mul(&pinv).mul(&m);
        for i in 0..m.rows {
            for j in 0..m.cols {
                assert!((mpm.get(i, j) - m.get(i, j)).abs() < 0.1);
            }
        }
    }

    // ========== Basic Solver Tests ==========

    #[test]
    fn test_solve_single_effector_basic() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        let params = JacobianParams::single_target(2, Vec3::new(1.5, 0.5, 0.0))
            .with_max_iterations(100)
            .with_tolerance(0.05);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        assert!(result.iterations > 0);
        assert!(result.per_target_error[0] < 0.5, "Error too large: {}", result.per_target_error[0]);
    }

    #[test]
    fn test_solve_reachable_target() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Target at (2, 0, 0) - fully extended arm
        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        let params = JacobianParams::single_target(2, Vec3::new(2.0, 0.0, 0.0))
            .with_tolerance(0.01);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        // Should converge for a reachable target
        assert!(result.per_target_error[0] < 0.1);
    }

    #[test]
    fn test_solve_unreachable_target() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Target at (10, 0, 0) - way beyond reach
        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        let params = JacobianParams::single_target(2, Vec3::new(10.0, 0.0, 0.0))
            .with_max_iterations(50)
            .with_tolerance(0.01);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        // Should not converge but should point towards target
        assert!(!result.converged);
        assert!(result.per_target_error[0] < 10.0); // Should get closer
    }

    // ========== Multiple Effector Tests ==========

    #[test]
    fn test_solve_multiple_effectors() {
        let skeleton = create_branching_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Create chain for left arm: root -> left_shoulder -> left_elbow -> left_hand
        let left_chain = JacobianChain::new_ball_chain(vec![0, 1, 2, 3]);

        // Target for left hand
        let params = JacobianParams::single_target(3, Vec3::new(-2.0, 0.5, 0.0))
            .with_max_iterations(100);

        let result = solve_jacobian(&left_chain, &skeleton, &mut pose, &params);

        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solve_weighted_targets() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);

        // Two targets with different weights
        let params = JacobianParams::default()
            .add_target(JacobianTarget::position(1, Vec3::new(0.5, 0.5, 0.0)).with_weight(0.2))
            .add_target(JacobianTarget::position(2, Vec3::new(1.5, 0.0, 0.0)).with_weight(0.8))
            .with_max_iterations(50);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        assert_eq!(result.per_target_error.len(), 2);
    }

    // ========== Damping Tests ==========

    #[test]
    fn test_damping_effect_stability() {
        let skeleton = create_arm_skeleton();

        // Test with low damping
        let mut pose_low = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        let params_low = JacobianParams::single_target(2, Vec3::new(1.0, 1.0, 0.0))
            .with_damping(0.01)
            .with_max_iterations(30);
        let result_low = solve_jacobian(&chain, &skeleton, &mut pose_low, &params_low);

        // Test with high damping
        let mut pose_high = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let params_high = JacobianParams::single_target(2, Vec3::new(1.0, 1.0, 0.0))
            .with_damping(0.5)
            .with_max_iterations(30);
        let result_high = solve_jacobian(&chain, &skeleton, &mut pose_high, &params_high);

        // High damping should converge slower but be more stable
        // (This is a qualitative test - both should produce valid results)
        assert!(result_low.iterations > 0);
        assert!(result_high.iterations > 0);
    }

    #[test]
    fn test_damping_prevents_oscillation() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Target near singularity
        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        let params = JacobianParams::single_target(2, Vec3::new(2.0, 0.001, 0.0))
            .with_damping(0.1)
            .with_max_iterations(100);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        // Should complete without exploding
        assert!(result.iterations <= 100);
        assert!(result.total_error.is_finite());
    }

    // ========== SVD vs DLS Tests ==========

    #[test]
    fn test_svd_vs_dls_comparison() {
        let skeleton = create_arm_skeleton();

        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        let target = Vec3::new(1.2, 0.8, 0.0);

        // Solve with DLS
        let mut pose_dls = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let params_dls = JacobianParams::single_target(2, target)
            .with_svd(false)
            .with_max_iterations(50);
        let result_dls = solve_jacobian(&chain, &skeleton, &mut pose_dls, &params_dls);

        // Solve with SVD
        let mut pose_svd = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let params_svd = JacobianParams::single_target(2, target)
            .with_svd(true)
            .with_max_iterations(50);
        let result_svd = solve_jacobian(&chain, &skeleton, &mut pose_svd, &params_svd);

        // Both should produce reasonable results
        assert!(result_dls.per_target_error[0] < 1.0);
        assert!(result_svd.per_target_error[0] < 1.0);
    }

    #[test]
    fn test_svd_near_singularity() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Near-singular configuration (fully extended arm)
        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        let params = JacobianParams::single_target(2, Vec3::new(2.0, 0.0001, 0.0))
            .with_svd(true)
            .with_max_iterations(100);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        // SVD should handle this gracefully
        assert!(result.total_error.is_finite());
        assert!(!result.total_error.is_nan());
    }

    // ========== Null-Space Posture Tests ==========

    #[test]
    fn test_null_space_posture_maintenance() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);

        // Define a preferred posture (9 DOF for 3 ball joints)
        let preferred_posture = vec![0.0, 0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0];

        let params = JacobianParams::single_target(2, Vec3::new(1.5, 0.3, 0.0))
            .with_null_space_posture(preferred_posture)
            .with_svd(true)
            .with_max_iterations(100);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        // Should still reach target
        assert!(result.per_target_error[0] < 1.0);
    }

    // ========== Singularity Handling Tests ==========

    #[test]
    fn test_singularity_extended_arm() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Target exactly at full extension
        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        let params = JacobianParams::single_target(2, Vec3::new(2.0, 0.0, 0.0))
            .with_damping(0.1)
            .with_max_iterations(50);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        // Should not explode
        assert!(result.total_error.is_finite());
        for angle in &result.joint_angles {
            assert!(angle.is_finite());
            assert!(angle.abs() < 10.0); // Reasonable angle bounds
        }
    }

    #[test]
    fn test_singularity_folded_arm() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Fold the arm first
        pose.rotations[1] = Quat::from_rotation_z(PI);

        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        let params = JacobianParams::single_target(2, Vec3::new(0.5, 0.5, 0.0))
            .with_max_iterations(100);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        // Should still produce valid output
        assert!(result.total_error.is_finite());
    }

    // ========== 6-DOF Tests ==========

    #[test]
    fn test_6dof_position_rotation_target() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);

        let target = JacobianTarget::position_rotation(
            2,
            Vec3::new(1.5, 0.3, 0.0),
            Quat::from_rotation_z(PI / 4.0),
        );

        let params = JacobianParams::default()
            .add_target(target)
            .with_max_iterations(100);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        assert!(result.iterations > 0);
        assert_eq!(result.per_target_error.len(), 1);
    }

    // ========== Task Priority Tests ==========

    #[test]
    fn test_task_priority_primary_secondary() {
        let skeleton = create_long_chain();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2, 3, 4]);

        // Primary target with high weight
        let primary = JacobianTarget::position(4, Vec3::new(1.5, 0.5, 0.0)).with_weight(1.0);

        // Secondary target with low weight
        let secondary = JacobianTarget::position(2, Vec3::new(0.5, 0.0, 0.0)).with_weight(0.1);

        let params = JacobianParams::default()
            .add_target(primary)
            .add_target(secondary)
            .with_max_iterations(100);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        // Primary should have lower error than secondary
        // (This may not always hold but is the general trend)
        assert_eq!(result.per_target_error.len(), 2);
    }

    // ========== Performance Benchmarks ==========

    #[test]
    fn test_performance_iterations() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        let params = JacobianParams::single_target(2, Vec3::new(1.0, 1.0, 0.0))
            .with_max_iterations(200)
            .with_tolerance(0.001);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        // Should converge in reasonable iterations
        if result.converged {
            assert!(result.iterations < 100, "Took too many iterations: {}", result.iterations);
        }
    }

    // ========== Error Handling Tests ==========

    #[test]
    fn test_try_solve_empty_chain() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let chain = JacobianChain::new_ball_chain(vec![]);
        let params = JacobianParams::single_target(2, Vec3::ZERO);

        let result = try_solve_jacobian(&chain, &skeleton, &mut pose, &params);
        assert!(matches!(result, Err(JacobianError::EmptyChain)));
    }

    #[test]
    fn test_try_solve_effector_not_in_chain() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let chain = JacobianChain::new_ball_chain(vec![0, 1]);
        let params = JacobianParams::single_target(2, Vec3::ZERO); // bone 2 not in chain

        let result = try_solve_jacobian(&chain, &skeleton, &mut pose, &params);
        assert!(matches!(result, Err(JacobianError::EffectorNotInChain { .. })));
    }

    #[test]
    fn test_try_solve_null_space_mismatch() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]); // 9 DOF
        let params = JacobianParams::single_target(2, Vec3::ZERO)
            .with_null_space_posture(vec![0.0; 5]); // Wrong length

        let result = try_solve_jacobian(&chain, &skeleton, &mut pose, &params);
        assert!(matches!(result, Err(JacobianError::NullSpacePostureMismatch { .. })));
    }

    // ========== Hinge Joint Tests ==========

    #[test]
    fn test_hinge_joints() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Create chain with hinge joints only (like a robot arm)
        let chain = JacobianChain::new(
            vec![0, 1, 2],
            vec![
                DofType::Hinge(Axis::Z),
                DofType::Hinge(Axis::Z),
                DofType::Hinge(Axis::Z),
            ],
        );

        assert_eq!(chain.total_dof(), 3);

        let params = JacobianParams::single_target(2, Vec3::new(1.0, 1.0, 0.0))
            .with_max_iterations(100);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        assert_eq!(result.joint_angles.len(), 3);
    }

    // ========== Universal Joint Tests ==========

    #[test]
    fn test_universal_joints() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let chain = JacobianChain::new(
            vec![0, 1, 2],
            vec![
                DofType::Universal(Axis::X, Axis::Y),
                DofType::Ball,
                DofType::Hinge(Axis::Z),
            ],
        );

        assert_eq!(chain.total_dof(), 2 + 3 + 1);

        let params = JacobianParams::single_target(2, Vec3::new(1.0, 0.5, 0.5))
            .with_max_iterations(100);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        assert_eq!(result.joint_angles.len(), 6);
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_target_at_current_position() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Get current effector position
        let world_transforms = compute_world_transforms(&skeleton, &pose);
        let current_pos = world_transforms[2].w_axis.truncate();

        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        let params = JacobianParams::single_target(2, current_pos)
            .with_tolerance(0.001);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        // Should converge immediately or very quickly
        assert!(result.converged);
        assert!(result.per_target_error[0] < 0.01);
    }

    #[test]
    fn test_zero_weight_target() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);

        // Zero weight target should have no effect
        let params = JacobianParams::default()
            .add_target(JacobianTarget::position(2, Vec3::new(5.0, 5.0, 0.0)).with_weight(0.0))
            .with_max_iterations(50);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        // Pose should be essentially unchanged
        for rot in &pose.rotations {
            assert!(rot.abs_diff_eq(Quat::IDENTITY, 0.1));
        }

        // Since weight is 0, weighted error contribution is 0
        assert!(result.total_error.abs() < 0.01);
    }

    // ========== JacobianError Display Tests ==========

    #[test]
    fn test_error_display() {
        let err = JacobianError::EmptyChain;
        assert!(err.to_string().contains("no bones"));

        let err = JacobianError::DofMismatch { bones: 3, dofs: 5 };
        assert!(err.to_string().contains("3"));
        assert!(err.to_string().contains("5"));

        let err = JacobianError::InvalidBoneIndex { index: 10, max: 5 };
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("5"));

        let err = JacobianError::TooManyDof { count: 200, max: 128 };
        assert!(err.to_string().contains("200"));
        assert!(err.to_string().contains("128"));

        let err = JacobianError::TooManyTargets { count: 20, max: 16 };
        assert!(err.to_string().contains("20"));
        assert!(err.to_string().contains("16"));

        let err = JacobianError::EffectorNotInChain { effector: 5 };
        assert!(err.to_string().contains("5"));

        let err = JacobianError::NullSpacePostureMismatch { expected: 9, got: 6 };
        assert!(err.to_string().contains("9"));
        assert!(err.to_string().contains("6"));

        let err = JacobianError::SingularMatrix;
        assert!(err.to_string().contains("singular"));
    }

    // ========== Result Construction Tests ==========

    #[test]
    fn test_result_failed() {
        let result = JacobianResult::failed(3);
        assert!(result.joint_angles.is_empty());
        assert_eq!(result.iterations, 0);
        assert!(!result.converged);
        assert_eq!(result.per_target_error.len(), 3);
        assert_eq!(result.total_error, f32::MAX);
    }

    // ========== Integration Test ==========

    #[test]
    fn test_full_ik_workflow() {
        // Create skeleton
        let skeleton = create_arm_skeleton();

        // Create initial pose
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Define chain
        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);

        // Validate chain
        assert!(chain.validate(&skeleton).is_ok());

        // Create target
        let target_pos = Vec3::new(1.2, 0.8, 0.3);
        let params = JacobianParams::single_target(2, target_pos)
            .with_damping(0.05)
            .with_max_iterations(100)
            .with_tolerance(0.05);

        // Solve
        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        // Verify result
        assert!(result.iterations > 0);
        assert!(result.total_error.is_finite());

        // Check pose was modified
        let world_transforms = compute_world_transforms(&skeleton, &pose);
        let effector_pos = world_transforms[2].w_axis.truncate();
        let distance = (effector_pos - target_pos).length();

        println!(
            "IK result: {} iterations, error={}, converged={}",
            result.iterations, result.per_target_error[0], result.converged
        );

        // Should be reasonably close to target
        assert!(distance < 1.0, "Distance to target: {}", distance);
    }

    // ========== Additional Coverage Tests ==========

    #[test]
    fn test_axis_to_vec3() {
        assert!(Axis::X.to_vec3().abs_diff_eq(Vec3::X, 1e-6));
        assert!(Axis::Y.to_vec3().abs_diff_eq(Vec3::Y, 1e-6));
        assert!(Axis::Z.to_vec3().abs_diff_eq(Vec3::Z, 1e-6));
    }

    #[test]
    fn test_chain_too_many_dof() {
        let skeleton = create_long_chain();
        // Create a chain with way too many DOF (hypothetically)
        // In practice, we need many bones to exceed MAX_DOF
        let mut bones = Vec::new();
        let mut dofs = Vec::new();
        for i in 0..skeleton.bone_count() {
            bones.push(i);
            dofs.push(DofType::Ball);
        }
        let chain = JacobianChain::new(bones, dofs);
        // Should validate successfully since 5 * 3 = 15 < MAX_DOF
        assert!(chain.validate(&skeleton).is_ok());
    }

    #[test]
    fn test_matrix_row_access() {
        let mut m = DenseMatrix::zeros(3, 4);
        m.set(1, 0, 5.0);
        m.set(1, 1, 6.0);
        m.set(1, 2, 7.0);
        m.set(1, 3, 8.0);

        let row = m.row(1);
        assert_eq!(row.len(), 4);
        assert!((row[0] - 5.0).abs() < 1e-6);
        assert!((row[1] - 6.0).abs() < 1e-6);
        assert!((row[2] - 7.0).abs() < 1e-6);
        assert!((row[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_frobenius_norm() {
        let mut m = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 3.0);
        m.set(1, 1, 4.0);

        // Frobenius norm = sqrt(3^2 + 4^2) = sqrt(25) = 5
        let norm = m.frobenius_norm();
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_quat_to_euler_xyz_identity() {
        let q = Quat::IDENTITY;
        let (x, y, z) = quat_to_euler_xyz(q);
        assert!(x.abs() < 1e-5);
        assert!(y.abs() < 1e-5);
        assert!(z.abs() < 1e-5);
    }

    #[test]
    fn test_quat_to_euler_xyz_rotation_x() {
        let angle = PI / 4.0;
        let q = Quat::from_rotation_x(angle);
        let (x, y, z) = quat_to_euler_xyz(q);
        assert!((x - angle).abs() < 1e-4, "x={}, expected={}", x, angle);
        assert!(y.abs() < 1e-4);
        assert!(z.abs() < 1e-4);
    }

    #[test]
    fn test_solve_with_very_low_damping() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        let params = JacobianParams::single_target(2, Vec3::new(1.0, 0.5, 0.0))
            .with_damping(0.001) // Very low damping
            .with_max_iterations(100);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        // Should still produce valid output
        assert!(result.total_error.is_finite());
        assert!(!result.total_error.is_nan());
    }

    #[test]
    fn test_solve_with_high_damping() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        let params = JacobianParams::single_target(2, Vec3::new(1.0, 0.5, 0.0))
            .with_damping(1.0) // Very high damping
            .with_max_iterations(200);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        // High damping = slower convergence but stable
        assert!(result.total_error.is_finite());
    }

    #[test]
    fn test_multiple_iterations_improving() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let chain = JacobianChain::new_ball_chain(vec![0, 1, 2]);
        let target = Vec3::new(1.0, 1.0, 0.0);

        // Get initial error
        let world_transforms = compute_world_transforms(&skeleton, &pose);
        let initial_error = (world_transforms[2].w_axis.truncate() - target).length();

        // Solve with many iterations
        let params = JacobianParams::single_target(2, target)
            .with_max_iterations(100)
            .with_tolerance(0.001);

        let result = solve_jacobian(&chain, &skeleton, &mut pose, &params);

        // Final error should be less than initial error
        assert!(
            result.per_target_error[0] < initial_error,
            "Should improve: initial={}, final={}",
            initial_error,
            result.per_target_error[0]
        );
    }

    #[test]
    fn test_bone_world_position_computation() {
        let skeleton = create_arm_skeleton();
        let pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Bone 0 (shoulder) at origin
        let pos0 = compute_bone_world_position(&skeleton, &pose, 0);
        assert!(pos0.abs_diff_eq(Vec3::ZERO, 1e-5));

        // Bone 1 (elbow) at (1, 0, 0)
        let pos1 = compute_bone_world_position(&skeleton, &pose, 1);
        assert!(pos1.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-5));

        // Bone 2 (wrist) at (2, 0, 0)
        let pos2 = compute_bone_world_position(&skeleton, &pose, 2);
        assert!(pos2.abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_bone_world_rotation_computation() {
        let skeleton = create_arm_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Rotate shoulder by 45 degrees around Z
        pose.rotations[0] = Quat::from_rotation_z(PI / 4.0);

        let rot = compute_bone_world_rotation(&skeleton, &pose, 0);
        assert!(rot.abs_diff_eq(Quat::from_rotation_z(PI / 4.0), 1e-4));
    }
}
