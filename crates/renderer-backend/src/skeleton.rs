//! Skeleton asset for skeletal animation in TRINITY Engine (T-AN-1.1).
//!
//! This module provides the foundational Skeleton asset type that represents
//! a bone hierarchy for skeletal animation. It supports:
//!
//! - Bone hierarchy with parent-child relationships
//! - Local and world space transform computation
//! - Inverse bind matrices for skinning
//! - Validation for cycles, invalid indices, and duplicates
//! - JSON serialization via serde
//!
//! # Memory Layout
//!
//! Skeletons are CPU-side assets that get converted to GPU-friendly formats
//! (see `skinning::JointData`) for runtime animation.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::skeleton::{Skeleton, Bone, Transform};
//! use glam::{Vec3, Quat, Mat4};
//!
//! // Create a simple skeleton with root and child bones
//! let mut skeleton = Skeleton::new();
//!
//! // Add root bone (no parent)
//! let root = Bone {
//!     name: "root".to_string(),
//!     parent_index: None,
//!     local_transform: Transform::IDENTITY,
//!     inverse_bind_matrix: Mat4::IDENTITY,
//! };
//! skeleton.add_bone(root);
//!
//! // Add child bone
//! let spine = Bone {
//!     name: "spine".to_string(),
//!     parent_index: Some(0),
//!     local_transform: Transform::from_position(Vec3::new(0.0, 1.0, 0.0)),
//!     inverse_bind_matrix: Mat4::IDENTITY,
//! };
//! skeleton.add_bone(spine);
//!
//! // Validate hierarchy
//! skeleton.validate().expect("valid skeleton");
//!
//! // Compute world transforms from local poses
//! let poses = vec![Transform::IDENTITY, Transform::IDENTITY];
//! let world_transforms = skeleton.compute_world_transforms(&poses);
//! ```

use std::collections::HashMap;
use std::fmt;

use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of bones supported per skeleton.
pub const MAX_BONES: usize = 256;

/// Maximum hierarchy depth for parent traversal.
pub const MAX_HIERARCHY_DEPTH: usize = 64;

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

/// A decomposed transform representing position, rotation, and scale.
///
/// This is the canonical representation for bone transforms in local space.
/// It can be converted to/from Mat4 for world space operations.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    /// Position in local space.
    pub position: Vec3,
    /// Rotation as a unit quaternion.
    pub rotation: Quat,
    /// Non-uniform scale.
    pub scale: Vec3,
}

impl Transform {
    /// Identity transform: no translation, no rotation, uniform scale of 1.
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// Create a transform from position only (identity rotation and scale).
    #[inline]
    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    /// Create a transform from rotation only (zero position, uniform scale).
    #[inline]
    pub fn from_rotation(rotation: Quat) -> Self {
        Self {
            position: Vec3::ZERO,
            rotation,
            scale: Vec3::ONE,
        }
    }

    /// Create a transform from position and rotation (uniform scale).
    #[inline]
    pub fn from_position_rotation(position: Vec3, rotation: Quat) -> Self {
        Self {
            position,
            rotation,
            scale: Vec3::ONE,
        }
    }

    /// Create a transform from all components.
    #[inline]
    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    /// Create a transform from a uniform scale factor.
    #[inline]
    pub fn from_scale(scale: f32) -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::splat(scale),
        }
    }

    /// Create a transform from non-uniform scale.
    #[inline]
    pub fn from_scale_vec(scale: Vec3) -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale,
        }
    }

    /// Convert this transform to a 4x4 matrix.
    ///
    /// The matrix is constructed as: Translation * Rotation * Scale
    #[inline]
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    /// Attempt to decompose a 4x4 matrix into a Transform.
    ///
    /// This may fail for matrices with shear or non-uniform negative scales.
    /// Returns None if decomposition fails.
    pub fn from_matrix(matrix: Mat4) -> Option<Self> {
        // Extract scale from column magnitudes
        let scale_x = matrix.x_axis.truncate().length();
        let scale_y = matrix.y_axis.truncate().length();
        let scale_z = matrix.z_axis.truncate().length();

        if scale_x < f32::EPSILON || scale_y < f32::EPSILON || scale_z < f32::EPSILON {
            return None;
        }

        let scale = Vec3::new(scale_x, scale_y, scale_z);

        // Normalize rotation columns
        let rot_x = matrix.x_axis.truncate() / scale_x;
        let rot_y = matrix.y_axis.truncate() / scale_y;
        let rot_z = matrix.z_axis.truncate() / scale_z;

        // Build rotation matrix and convert to quaternion
        let rotation_mat = Mat4::from_cols(
            rot_x.extend(0.0),
            rot_y.extend(0.0),
            rot_z.extend(0.0),
            Vec3::ZERO.extend(1.0),
        );

        let rotation = Quat::from_mat4(&rotation_mat);

        // Extract translation
        let position = matrix.w_axis.truncate();

        Some(Self {
            position,
            rotation,
            scale,
        })
    }

    /// Multiply two transforms (parent * child = world).
    ///
    /// This combines transforms as if `self` is the parent and `other` is the child.
    #[inline]
    pub fn mul_transform(&self, other: &Transform) -> Transform {
        // Combined transform: parent_T * parent_R * parent_S * child_T * child_R * child_S
        // For proper hierarchical transforms:
        // - Child position is transformed by parent's full TRS
        // - Child rotation is combined with parent rotation
        // - Child scale is combined with parent scale

        let rotated_scaled_pos = self.rotation * (self.scale * other.position);
        let combined_position = self.position + rotated_scaled_pos;
        let combined_rotation = self.rotation * other.rotation;
        let combined_scale = self.scale * other.scale;

        Transform {
            position: combined_position,
            rotation: combined_rotation,
            scale: combined_scale,
        }
    }

    /// Compute the inverse of this transform.
    ///
    /// Returns None if the scale contains zero components.
    pub fn inverse(&self) -> Option<Transform> {
        if self.scale.x.abs() < f32::EPSILON
            || self.scale.y.abs() < f32::EPSILON
            || self.scale.z.abs() < f32::EPSILON
        {
            return None;
        }

        let inv_scale = Vec3::ONE / self.scale;
        let inv_rotation = self.rotation.inverse();
        let inv_position = inv_rotation * (-self.position * inv_scale);

        Some(Transform {
            position: inv_position,
            rotation: inv_rotation,
            scale: inv_scale,
        })
    }

    /// Linearly interpolate between two transforms.
    ///
    /// Uses spherical linear interpolation (slerp) for rotation.
    #[inline]
    pub fn lerp(&self, other: &Transform, t: f32) -> Transform {
        Transform {
            position: self.position.lerp(other.position, t),
            rotation: self.rotation.slerp(other.rotation, t),
            scale: self.scale.lerp(other.scale, t),
        }
    }

    /// Check if this transform is approximately equal to another.
    #[inline]
    pub fn approx_eq(&self, other: &Transform, epsilon: f32) -> bool {
        self.position.abs_diff_eq(other.position, epsilon)
            && (self.rotation.dot(other.rotation).abs() > 1.0 - epsilon)
            && self.scale.abs_diff_eq(other.scale, epsilon)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<Transform> for Mat4 {
    fn from(t: Transform) -> Self {
        t.to_matrix()
    }
}

// ---------------------------------------------------------------------------
// Bone
// ---------------------------------------------------------------------------

/// A single bone in a skeleton hierarchy.
///
/// Each bone stores:
/// - A unique name for identification
/// - An optional parent index (None for root bones)
/// - A local-space transform relative to the parent
/// - An inverse bind matrix for skinning computation
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bone {
    /// Unique name of this bone.
    pub name: String,

    /// Index of the parent bone, or None if this is a root bone.
    pub parent_index: Option<usize>,

    /// Transform relative to the parent bone.
    /// For root bones, this is relative to the skeleton's origin.
    pub local_transform: Transform,

    /// Inverse of the bind-pose world transform.
    /// Used to transform vertices from model space to bone space.
    #[serde(with = "mat4_serde")]
    pub inverse_bind_matrix: Mat4,
}

impl Bone {
    /// Create a new bone with the given name and identity transforms.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parent_index: None,
            local_transform: Transform::IDENTITY,
            inverse_bind_matrix: Mat4::IDENTITY,
        }
    }

    /// Create a root bone (no parent).
    pub fn root(name: impl Into<String>) -> Self {
        Self::new(name)
    }

    /// Create a child bone with the given parent index.
    pub fn with_parent(mut self, parent_index: usize) -> Self {
        self.parent_index = Some(parent_index);
        self
    }

    /// Set the local transform.
    pub fn with_local_transform(mut self, transform: Transform) -> Self {
        self.local_transform = transform;
        self
    }

    /// Set the inverse bind matrix.
    pub fn with_inverse_bind_matrix(mut self, matrix: Mat4) -> Self {
        self.inverse_bind_matrix = matrix;
        self
    }

    /// Check if this is a root bone (no parent).
    #[inline]
    pub fn is_root(&self) -> bool {
        self.parent_index.is_none()
    }
}

/// Custom serde for Mat4 (column-major array).
mod mat4_serde {
    use glam::Mat4;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(mat: &Mat4, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        mat.to_cols_array().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Mat4, D::Error>
    where
        D: Deserializer<'de>,
    {
        let arr = <[f32; 16]>::deserialize(deserializer)?;
        Ok(Mat4::from_cols_array(&arr))
    }
}

// ---------------------------------------------------------------------------
// SkeletonError
// ---------------------------------------------------------------------------

/// Errors that can occur during skeleton validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkeletonError {
    /// A bone references a parent that would create a cycle in the hierarchy.
    CircularParentReference {
        /// Index of the bone that creates the cycle.
        bone_index: usize,
        /// Name of the bone.
        bone_name: String,
        /// The cycle path as a list of bone indices.
        cycle_path: Vec<usize>,
    },

    /// A bone references a parent index that doesn't exist.
    InvalidParentIndex {
        /// Index of the bone with the invalid parent.
        bone_index: usize,
        /// Name of the bone.
        bone_name: String,
        /// The invalid parent index.
        parent_index: usize,
    },

    /// Two or more bones share the same name.
    DuplicateBoneName {
        /// The duplicated name.
        name: String,
        /// Indices of bones with this name.
        indices: Vec<usize>,
    },

    /// A root bone is not at index 0 (optional strict validation).
    RootNotAtIndexZero {
        /// Index of the root bone.
        root_index: usize,
        /// Name of the root bone.
        root_name: String,
    },

    /// Parent index references a bone that comes after this bone.
    /// This violates topological ordering.
    ParentNotBeforeChild {
        /// Index of the child bone.
        bone_index: usize,
        /// Name of the child bone.
        bone_name: String,
        /// Index of the parent bone.
        parent_index: usize,
    },

    /// The skeleton exceeds the maximum bone count.
    TooManyBones {
        /// Number of bones in the skeleton.
        count: usize,
        /// Maximum allowed.
        max: usize,
    },

    /// The skeleton hierarchy is too deep.
    HierarchyTooDeep {
        /// Bone index where depth exceeded.
        bone_index: usize,
        /// Depth at that bone.
        depth: usize,
        /// Maximum allowed depth.
        max_depth: usize,
    },

    /// Inverse bind matrix is degenerate (not invertible).
    DegenerateInverseBindMatrix {
        /// Index of the bone.
        bone_index: usize,
        /// Name of the bone.
        bone_name: String,
    },
}

impl fmt::Display for SkeletonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CircularParentReference {
                bone_index,
                bone_name,
                cycle_path,
            } => {
                write!(
                    f,
                    "circular parent reference at bone {} '{}': cycle {:?}",
                    bone_index, bone_name, cycle_path
                )
            }
            Self::InvalidParentIndex {
                bone_index,
                bone_name,
                parent_index,
            } => {
                write!(
                    f,
                    "invalid parent index {} for bone {} '{}'",
                    parent_index, bone_index, bone_name
                )
            }
            Self::DuplicateBoneName { name, indices } => {
                write!(
                    f,
                    "duplicate bone name '{}' at indices {:?}",
                    name, indices
                )
            }
            Self::RootNotAtIndexZero {
                root_index,
                root_name,
            } => {
                write!(
                    f,
                    "root bone '{}' at index {}, expected index 0",
                    root_name, root_index
                )
            }
            Self::ParentNotBeforeChild {
                bone_index,
                bone_name,
                parent_index,
            } => {
                write!(
                    f,
                    "bone {} '{}' has parent at index {}, which is not before it",
                    bone_index, bone_name, parent_index
                )
            }
            Self::TooManyBones { count, max } => {
                write!(f, "skeleton has {} bones, maximum is {}", count, max)
            }
            Self::HierarchyTooDeep {
                bone_index,
                depth,
                max_depth,
            } => {
                write!(
                    f,
                    "hierarchy too deep at bone {}: depth {} exceeds max {}",
                    bone_index, depth, max_depth
                )
            }
            Self::DegenerateInverseBindMatrix {
                bone_index,
                bone_name,
            } => {
                write!(
                    f,
                    "degenerate inverse bind matrix at bone {} '{}'",
                    bone_index, bone_name
                )
            }
        }
    }
}

impl std::error::Error for SkeletonError {}

// ---------------------------------------------------------------------------
// Skeleton
// ---------------------------------------------------------------------------

/// A skeletal hierarchy for animation.
///
/// The skeleton stores bones in a flat array with parent indices forming
/// a tree structure. Multiple root bones are supported (forest structure).
///
/// For efficient hierarchy traversal, bones should be stored in topological
/// order (parent before children). The [`add_bone`](Self::add_bone) method
/// enforces this ordering.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Skeleton {
    /// All bones in the skeleton, stored in topological order.
    bones: Vec<Bone>,

    /// Map from bone name to index for O(1) lookup.
    #[serde(skip)]
    name_to_index: HashMap<String, usize>,

    /// Indices of root bones (bones with no parent).
    root_indices: Vec<usize>,
}

impl Skeleton {
    /// Create a new empty skeleton.
    pub fn new() -> Self {
        Self {
            bones: Vec::new(),
            name_to_index: HashMap::new(),
            root_indices: Vec::new(),
        }
    }

    /// Create a skeleton with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bones: Vec::with_capacity(capacity),
            name_to_index: HashMap::with_capacity(capacity),
            root_indices: Vec::new(),
        }
    }

    /// Add a bone to the skeleton.
    ///
    /// Returns the index of the newly added bone.
    ///
    /// # Panics
    ///
    /// Panics if the parent index (if specified) is >= the current bone count.
    /// Bones must be added in topological order (parent before children).
    pub fn add_bone(&mut self, bone: Bone) -> usize {
        let index = self.bones.len();

        // Validate parent index
        if let Some(parent_idx) = bone.parent_index {
            assert!(
                parent_idx < index,
                "parent index {} must be less than bone index {}",
                parent_idx,
                index
            );
        }

        // Track root bones
        if bone.is_root() {
            self.root_indices.push(index);
        }

        // Update name index
        self.name_to_index.insert(bone.name.clone(), index);

        self.bones.push(bone);
        index
    }

    /// Get a bone by index.
    #[inline]
    pub fn bone(&self, index: usize) -> Option<&Bone> {
        self.bones.get(index)
    }

    /// Get a mutable reference to a bone by index.
    #[inline]
    pub fn bone_mut(&mut self, index: usize) -> Option<&mut Bone> {
        self.bones.get_mut(index)
    }

    /// Get a bone by name.
    #[inline]
    pub fn bone_by_name(&self, name: &str) -> Option<&Bone> {
        self.name_to_index.get(name).and_then(|&i| self.bones.get(i))
    }

    /// Get the index of a bone by name.
    #[inline]
    pub fn bone_index(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }

    /// Get the parent index of a bone.
    #[inline]
    pub fn parent(&self, index: usize) -> Option<usize> {
        self.bones.get(index).and_then(|b| b.parent_index)
    }

    /// Get all direct children of a bone.
    ///
    /// This is O(n) where n is the total bone count.
    pub fn children(&self, index: usize) -> Vec<usize> {
        self.bones
            .iter()
            .enumerate()
            .filter(|(_, bone)| bone.parent_index == Some(index))
            .map(|(i, _)| i)
            .collect()
    }

    /// Get the number of bones in the skeleton.
    #[inline]
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Check if the skeleton is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bones.is_empty()
    }

    /// Get all root bone indices.
    #[inline]
    pub fn root_indices(&self) -> &[usize] {
        &self.root_indices
    }

    /// Get the number of root bones.
    #[inline]
    pub fn root_count(&self) -> usize {
        self.root_indices.len()
    }

    /// Iterate over all bones.
    #[inline]
    pub fn bones(&self) -> &[Bone] {
        &self.bones
    }

    /// Get the depth of a bone in the hierarchy.
    ///
    /// Root bones have depth 0.
    pub fn bone_depth(&self, index: usize) -> Option<usize> {
        let mut depth = 0;
        let mut current = index;

        loop {
            if let Some(bone) = self.bones.get(current) {
                match bone.parent_index {
                    Some(parent) => {
                        depth += 1;
                        current = parent;
                    }
                    None => return Some(depth),
                }
            } else {
                return None;
            }

            // Safety check for cycles
            if depth > MAX_HIERARCHY_DEPTH {
                return None;
            }
        }
    }

    /// Compute world-space transforms from local poses.
    ///
    /// # Arguments
    ///
    /// * `local_poses` - Local-space transforms for each bone. Must have
    ///   the same length as the bone count.
    ///
    /// # Returns
    ///
    /// A vector of world-space transformation matrices, one per bone.
    ///
    /// # Panics
    ///
    /// Panics if `local_poses.len() != self.bone_count()`.
    pub fn compute_world_transforms(&self, local_poses: &[Transform]) -> Vec<Mat4> {
        assert_eq!(
            local_poses.len(),
            self.bones.len(),
            "local_poses length {} must match bone count {}",
            local_poses.len(),
            self.bones.len()
        );

        let mut world_transforms = Vec::with_capacity(self.bones.len());

        for (i, bone) in self.bones.iter().enumerate() {
            let local_mat = local_poses[i].to_matrix();

            let world_mat = match bone.parent_index {
                Some(parent_idx) => {
                    // Parent must come before child due to topological ordering
                    debug_assert!(parent_idx < i);
                    world_transforms[parent_idx] * local_mat
                }
                None => local_mat,
            };

            world_transforms.push(world_mat);
        }

        world_transforms
    }

    /// Compute skinning matrices from local poses.
    ///
    /// Skinning matrix = world_transform * inverse_bind_matrix
    ///
    /// This transforms vertices from their original (bind pose) position
    /// to their animated position.
    ///
    /// # Arguments
    ///
    /// * `local_poses` - Local-space transforms for each bone.
    ///
    /// # Returns
    ///
    /// A vector of skinning matrices, one per bone.
    pub fn compute_skinning_matrices(&self, local_poses: &[Transform]) -> Vec<Mat4> {
        let world_transforms = self.compute_world_transforms(local_poses);

        world_transforms
            .iter()
            .zip(self.bones.iter())
            .map(|(world, bone)| *world * bone.inverse_bind_matrix)
            .collect()
    }

    /// Compute bind-pose world transforms.
    ///
    /// Uses the local_transform of each bone to compute world transforms.
    pub fn compute_bind_pose_world_transforms(&self) -> Vec<Mat4> {
        let local_poses: Vec<Transform> = self.bones.iter().map(|b| b.local_transform).collect();
        self.compute_world_transforms(&local_poses)
    }

    /// Compute and cache inverse bind matrices from bind pose.
    ///
    /// Call this after setting up the skeleton hierarchy and local transforms
    /// to automatically compute inverse bind matrices.
    ///
    /// # Returns
    ///
    /// `Err` if any world transform is not invertible.
    pub fn compute_inverse_bind_matrices(&mut self) -> Result<(), SkeletonError> {
        let world_transforms = self.compute_bind_pose_world_transforms();

        for (i, world) in world_transforms.iter().enumerate() {
            let det = world.determinant();
            if det.abs() < f32::EPSILON {
                return Err(SkeletonError::DegenerateInverseBindMatrix {
                    bone_index: i,
                    bone_name: self.bones[i].name.clone(),
                });
            }
            self.bones[i].inverse_bind_matrix = world.inverse();
        }

        Ok(())
    }

    /// Validate the skeleton hierarchy.
    ///
    /// Checks for:
    /// - Circular parent references
    /// - Invalid parent indices
    /// - Duplicate bone names
    /// - Excessive bone count
    /// - Excessive hierarchy depth
    ///
    /// # Returns
    ///
    /// `Ok(())` if the skeleton is valid, or the first error found.
    pub fn validate(&self) -> Result<(), SkeletonError> {
        // Check bone count
        if self.bones.len() > MAX_BONES {
            return Err(SkeletonError::TooManyBones {
                count: self.bones.len(),
                max: MAX_BONES,
            });
        }

        // Check for duplicate names
        let mut name_indices: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, bone) in self.bones.iter().enumerate() {
            name_indices
                .entry(&bone.name)
                .or_insert_with(Vec::new)
                .push(i);
        }
        for (name, indices) in name_indices {
            if indices.len() > 1 {
                return Err(SkeletonError::DuplicateBoneName {
                    name: name.to_string(),
                    indices,
                });
            }
        }

        // Check each bone
        for (i, bone) in self.bones.iter().enumerate() {
            // Check parent index validity
            if let Some(parent_idx) = bone.parent_index {
                if parent_idx >= self.bones.len() {
                    return Err(SkeletonError::InvalidParentIndex {
                        bone_index: i,
                        bone_name: bone.name.clone(),
                        parent_index: parent_idx,
                    });
                }

                // Check topological ordering
                if parent_idx >= i {
                    return Err(SkeletonError::ParentNotBeforeChild {
                        bone_index: i,
                        bone_name: bone.name.clone(),
                        parent_index: parent_idx,
                    });
                }
            }

            // Check hierarchy depth (also detects cycles)
            let mut depth = 0;
            let mut current = i;
            let mut visited = vec![false; self.bones.len()];

            loop {
                if visited[current] {
                    // Found a cycle - collect the path
                    let mut cycle_path = vec![current];
                    let mut trace = i;
                    while trace != current {
                        if let Some(parent) = self.bones[trace].parent_index {
                            cycle_path.push(trace);
                            trace = parent;
                        } else {
                            break;
                        }
                    }
                    return Err(SkeletonError::CircularParentReference {
                        bone_index: i,
                        bone_name: bone.name.clone(),
                        cycle_path,
                    });
                }

                visited[current] = true;

                match self.bones[current].parent_index {
                    Some(parent) => {
                        depth += 1;
                        if depth > MAX_HIERARCHY_DEPTH {
                            return Err(SkeletonError::HierarchyTooDeep {
                                bone_index: i,
                                depth,
                                max_depth: MAX_HIERARCHY_DEPTH,
                            });
                        }
                        current = parent;
                    }
                    None => break,
                }
            }
        }

        Ok(())
    }

    /// Validate with strict root-at-zero requirement.
    ///
    /// This is a stricter validation that requires the first root bone
    /// to be at index 0.
    pub fn validate_strict(&self) -> Result<(), SkeletonError> {
        self.validate()?;

        // Check that first root is at index 0
        if !self.root_indices.is_empty() && self.root_indices[0] != 0 {
            return Err(SkeletonError::RootNotAtIndexZero {
                root_index: self.root_indices[0],
                root_name: self.bones[self.root_indices[0]].name.clone(),
            });
        }

        Ok(())
    }

    /// Rebuild internal indices after deserialization.
    ///
    /// Call this after deserializing a skeleton to rebuild the
    /// `name_to_index` map and `root_indices` vector.
    pub fn rebuild_indices(&mut self) {
        self.name_to_index.clear();
        self.root_indices.clear();

        for (i, bone) in self.bones.iter().enumerate() {
            self.name_to_index.insert(bone.name.clone(), i);
            if bone.is_root() {
                self.root_indices.push(i);
            }
        }
    }

    /// Get all descendants of a bone (children, grandchildren, etc.).
    pub fn descendants(&self, index: usize) -> Vec<usize> {
        let mut result = Vec::new();
        let mut stack = self.children(index);

        while let Some(child) = stack.pop() {
            result.push(child);
            stack.extend(self.children(child));
        }

        result
    }

    /// Get the path from root to a bone.
    ///
    /// Returns the bone indices from root to the specified bone (inclusive).
    pub fn path_to_root(&self, index: usize) -> Vec<usize> {
        let mut path = vec![index];
        let mut current = index;

        while let Some(bone) = self.bones.get(current) {
            if let Some(parent) = bone.parent_index {
                path.push(parent);
                current = parent;
            } else {
                break;
            }
        }

        path.reverse();
        path
    }

    /// Find the common ancestor of two bones.
    ///
    /// Returns `None` if the bones are in different trees (no common ancestor).
    pub fn common_ancestor(&self, a: usize, b: usize) -> Option<usize> {
        let path_a = self.path_to_root(a);
        let path_b = self.path_to_root(b);

        // Find last common element
        let mut ancestor = None;
        for (pa, pb) in path_a.iter().zip(path_b.iter()) {
            if pa == pb {
                ancestor = Some(*pa);
            } else {
                break;
            }
        }

        ancestor
    }

    /// Create a JSON representation of this skeleton.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Create a skeleton from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let mut skeleton: Self = serde_json::from_str(json)?;
        skeleton.rebuild_indices();
        Ok(skeleton)
    }
}

impl PartialEq for Skeleton {
    fn eq(&self, other: &Self) -> bool {
        self.bones == other.bones
    }
}

// ---------------------------------------------------------------------------
// SkeletonBuilder
// ---------------------------------------------------------------------------

/// Builder for creating skeletons with a fluent API.
#[derive(Default)]
pub struct SkeletonBuilder {
    skeleton: Skeleton,
}

impl SkeletonBuilder {
    /// Create a new skeleton builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a root bone.
    pub fn root(mut self, name: impl Into<String>) -> Self {
        let bone = Bone::new(name);
        self.skeleton.add_bone(bone);
        self
    }

    /// Add a child bone to the specified parent.
    pub fn child(mut self, name: impl Into<String>, parent_name: &str) -> Self {
        let parent_index = self
            .skeleton
            .bone_index(parent_name)
            .expect("parent bone must exist");

        let bone = Bone::new(name).with_parent(parent_index);
        self.skeleton.add_bone(bone);
        self
    }

    /// Add a child bone at a specific position relative to parent.
    pub fn child_at(
        mut self,
        name: impl Into<String>,
        parent_name: &str,
        position: Vec3,
    ) -> Self {
        let parent_index = self
            .skeleton
            .bone_index(parent_name)
            .expect("parent bone must exist");

        let bone = Bone::new(name)
            .with_parent(parent_index)
            .with_local_transform(Transform::from_position(position));
        self.skeleton.add_bone(bone);
        self
    }

    /// Build the skeleton, computing inverse bind matrices.
    pub fn build(mut self) -> Result<Skeleton, SkeletonError> {
        self.skeleton.compute_inverse_bind_matrices()?;
        self.skeleton.validate()?;
        Ok(self.skeleton)
    }

    /// Build without computing inverse bind matrices.
    pub fn build_unchecked(self) -> Skeleton {
        self.skeleton
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // ===== Transform Tests =====

    #[test]
    fn test_transform_identity() {
        let t = Transform::IDENTITY;
        assert_eq!(t.position, Vec3::ZERO);
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn test_transform_from_position() {
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let t = Transform::from_position(pos);
        assert_eq!(t.position, pos);
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn test_transform_from_rotation() {
        let rot = Quat::from_rotation_y(PI / 2.0);
        let t = Transform::from_rotation(rot);
        assert_eq!(t.position, Vec3::ZERO);
        assert!(t.rotation.abs_diff_eq(rot, 1e-6));
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn test_transform_new() {
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let rot = Quat::from_rotation_z(PI / 4.0);
        let scale = Vec3::new(2.0, 2.0, 2.0);
        let t = Transform::new(pos, rot, scale);
        assert_eq!(t.position, pos);
        assert!(t.rotation.abs_diff_eq(rot, 1e-6));
        assert_eq!(t.scale, scale);
    }

    #[test]
    fn test_transform_to_matrix_identity() {
        let t = Transform::IDENTITY;
        let m = t.to_matrix();
        assert!(m.abs_diff_eq(Mat4::IDENTITY, 1e-6));
    }

    #[test]
    fn test_transform_to_matrix_translation() {
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let t = Transform::from_position(pos);
        let m = t.to_matrix();
        let expected = Mat4::from_translation(pos);
        assert!(m.abs_diff_eq(expected, 1e-6));
    }

    #[test]
    fn test_transform_to_matrix_rotation() {
        let rot = Quat::from_rotation_y(PI / 2.0);
        let t = Transform::from_rotation(rot);
        let m = t.to_matrix();
        let expected = Mat4::from_quat(rot);
        assert!(m.abs_diff_eq(expected, 1e-6));
    }

    #[test]
    fn test_transform_to_matrix_scale() {
        let scale = Vec3::new(2.0, 3.0, 4.0);
        let t = Transform::from_scale_vec(scale);
        let m = t.to_matrix();
        let expected = Mat4::from_scale(scale);
        assert!(m.abs_diff_eq(expected, 1e-6));
    }

    #[test]
    fn test_transform_from_matrix_identity() {
        let m = Mat4::IDENTITY;
        let t = Transform::from_matrix(m).unwrap();
        assert!(t.approx_eq(&Transform::IDENTITY, 1e-5));
    }

    #[test]
    fn test_transform_from_matrix_translation() {
        let pos = Vec3::new(5.0, -3.0, 2.0);
        let m = Mat4::from_translation(pos);
        let t = Transform::from_matrix(m).unwrap();
        assert!(t.position.abs_diff_eq(pos, 1e-5));
    }

    #[test]
    fn test_transform_roundtrip_matrix() {
        let original = Transform::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_x(PI / 3.0),
            Vec3::new(1.5, 2.0, 0.5),
        );
        let m = original.to_matrix();
        let recovered = Transform::from_matrix(m).unwrap();
        assert!(recovered.approx_eq(&original, 1e-4));
    }

    #[test]
    fn test_transform_mul_identity() {
        let t = Transform::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_y(PI / 4.0),
            Vec3::new(2.0, 2.0, 2.0),
        );
        let result = t.mul_transform(&Transform::IDENTITY);
        assert!(result.approx_eq(&t, 1e-5));
    }

    #[test]
    fn test_transform_mul_translation() {
        let parent = Transform::from_position(Vec3::new(10.0, 0.0, 0.0));
        let child = Transform::from_position(Vec3::new(5.0, 0.0, 0.0));
        let result = parent.mul_transform(&child);
        assert!(result.position.abs_diff_eq(Vec3::new(15.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_transform_mul_rotation() {
        let parent = Transform::from_rotation(Quat::from_rotation_y(PI / 2.0));
        let child = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));
        let result = parent.mul_transform(&child);
        // Rotating (1,0,0) by 90 degrees around Y gives (0,0,-1)
        assert!(result.position.abs_diff_eq(Vec3::new(0.0, 0.0, -1.0), 1e-5));
    }

    #[test]
    fn test_transform_inverse() {
        let t = Transform::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_z(PI / 6.0),
            Vec3::new(2.0, 2.0, 2.0),
        );
        let inv = t.inverse().unwrap();
        let result = t.mul_transform(&inv);
        assert!(result.approx_eq(&Transform::IDENTITY, 1e-4));
    }

    #[test]
    fn test_transform_inverse_zero_scale() {
        let t = Transform::from_scale_vec(Vec3::new(0.0, 1.0, 1.0));
        assert!(t.inverse().is_none());
    }

    #[test]
    fn test_transform_lerp() {
        let a = Transform::from_position(Vec3::ZERO);
        let b = Transform::from_position(Vec3::new(10.0, 0.0, 0.0));
        let mid = a.lerp(&b, 0.5);
        assert!(mid.position.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_transform_default() {
        let t = Transform::default();
        assert_eq!(t, Transform::IDENTITY);
    }

    // ===== Bone Tests =====

    #[test]
    fn test_bone_new() {
        let bone = Bone::new("test");
        assert_eq!(bone.name, "test");
        assert!(bone.is_root());
        assert_eq!(bone.local_transform, Transform::IDENTITY);
        assert!(bone.inverse_bind_matrix.abs_diff_eq(Mat4::IDENTITY, 1e-6));
    }

    #[test]
    fn test_bone_root() {
        let bone = Bone::root("root");
        assert!(bone.is_root());
        assert_eq!(bone.parent_index, None);
    }

    #[test]
    fn test_bone_with_parent() {
        let bone = Bone::new("child").with_parent(0);
        assert!(!bone.is_root());
        assert_eq!(bone.parent_index, Some(0));
    }

    #[test]
    fn test_bone_with_local_transform() {
        let t = Transform::from_position(Vec3::new(1.0, 2.0, 3.0));
        let bone = Bone::new("bone").with_local_transform(t);
        assert_eq!(bone.local_transform, t);
    }

    #[test]
    fn test_bone_with_inverse_bind_matrix() {
        let m = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        let bone = Bone::new("bone").with_inverse_bind_matrix(m);
        assert!(bone.inverse_bind_matrix.abs_diff_eq(m, 1e-6));
    }

    // ===== Skeleton Tests =====

    #[test]
    fn test_skeleton_new() {
        let skeleton = Skeleton::new();
        assert_eq!(skeleton.bone_count(), 0);
        assert!(skeleton.is_empty());
    }

    #[test]
    fn test_skeleton_add_bone() {
        let mut skeleton = Skeleton::new();
        let idx = skeleton.add_bone(Bone::new("root"));
        assert_eq!(idx, 0);
        assert_eq!(skeleton.bone_count(), 1);
        assert!(!skeleton.is_empty());
    }

    #[test]
    fn test_skeleton_add_multiple_bones() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.add_bone(Bone::new("child").with_parent(0));
        skeleton.add_bone(Bone::new("grandchild").with_parent(1));

        assert_eq!(skeleton.bone_count(), 3);
        assert_eq!(skeleton.root_count(), 1);
    }

    #[test]
    fn test_skeleton_bone_by_index() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::new("root"));

        let bone = skeleton.bone(0).unwrap();
        assert_eq!(bone.name, "root");

        assert!(skeleton.bone(1).is_none());
    }

    #[test]
    fn test_skeleton_bone_by_name() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::new("root"));
        skeleton.add_bone(Bone::new("child").with_parent(0));

        let bone = skeleton.bone_by_name("child").unwrap();
        assert_eq!(bone.parent_index, Some(0));

        assert!(skeleton.bone_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_skeleton_bone_index() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::new("root"));
        skeleton.add_bone(Bone::new("child").with_parent(0));

        assert_eq!(skeleton.bone_index("root"), Some(0));
        assert_eq!(skeleton.bone_index("child"), Some(1));
        assert_eq!(skeleton.bone_index("nonexistent"), None);
    }

    #[test]
    fn test_skeleton_parent() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.add_bone(Bone::new("child").with_parent(0));

        assert_eq!(skeleton.parent(0), None);
        assert_eq!(skeleton.parent(1), Some(0));
        assert_eq!(skeleton.parent(99), None);
    }

    #[test]
    fn test_skeleton_children() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.add_bone(Bone::new("child1").with_parent(0));
        skeleton.add_bone(Bone::new("child2").with_parent(0));
        skeleton.add_bone(Bone::new("grandchild").with_parent(1));

        let root_children = skeleton.children(0);
        assert_eq!(root_children.len(), 2);
        assert!(root_children.contains(&1));
        assert!(root_children.contains(&2));

        let child1_children = skeleton.children(1);
        assert_eq!(child1_children.len(), 1);
        assert!(child1_children.contains(&3));

        assert!(skeleton.children(3).is_empty());
    }

    #[test]
    fn test_skeleton_root_indices() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root1"));
        skeleton.add_bone(Bone::new("child").with_parent(0));
        skeleton.add_bone(Bone::root("root2")); // Second root

        assert_eq!(skeleton.root_indices(), &[0, 2]);
        assert_eq!(skeleton.root_count(), 2);
    }

    #[test]
    fn test_skeleton_bone_depth() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.add_bone(Bone::new("child").with_parent(0));
        skeleton.add_bone(Bone::new("grandchild").with_parent(1));
        skeleton.add_bone(Bone::new("great_grandchild").with_parent(2));

        assert_eq!(skeleton.bone_depth(0), Some(0));
        assert_eq!(skeleton.bone_depth(1), Some(1));
        assert_eq!(skeleton.bone_depth(2), Some(2));
        assert_eq!(skeleton.bone_depth(3), Some(3));
        assert_eq!(skeleton.bone_depth(99), None);
    }

    #[test]
    fn test_skeleton_descendants() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.add_bone(Bone::new("child1").with_parent(0));
        skeleton.add_bone(Bone::new("child2").with_parent(0));
        skeleton.add_bone(Bone::new("grandchild").with_parent(1));

        let desc = skeleton.descendants(0);
        assert_eq!(desc.len(), 3);
        assert!(desc.contains(&1));
        assert!(desc.contains(&2));
        assert!(desc.contains(&3));
    }

    #[test]
    fn test_skeleton_path_to_root() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.add_bone(Bone::new("child").with_parent(0));
        skeleton.add_bone(Bone::new("grandchild").with_parent(1));

        let path = skeleton.path_to_root(2);
        assert_eq!(path, vec![0, 1, 2]);

        let root_path = skeleton.path_to_root(0);
        assert_eq!(root_path, vec![0]);
    }

    #[test]
    fn test_skeleton_common_ancestor() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.add_bone(Bone::new("left").with_parent(0));
        skeleton.add_bone(Bone::new("right").with_parent(0));
        skeleton.add_bone(Bone::new("left_child").with_parent(1));

        assert_eq!(skeleton.common_ancestor(1, 2), Some(0));
        assert_eq!(skeleton.common_ancestor(3, 2), Some(0));
        assert_eq!(skeleton.common_ancestor(3, 1), Some(1));
    }

    #[test]
    fn test_skeleton_common_ancestor_different_trees() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root1"));
        skeleton.add_bone(Bone::root("root2"));
        skeleton.add_bone(Bone::new("child1").with_parent(0));
        skeleton.add_bone(Bone::new("child2").with_parent(1));

        assert_eq!(skeleton.common_ancestor(2, 3), None);
    }

    // ===== World Transform Tests =====

    #[test]
    fn test_compute_world_transforms_single_bone() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));

        let poses = vec![Transform::from_position(Vec3::new(1.0, 2.0, 3.0))];
        let world = skeleton.compute_world_transforms(&poses);

        assert_eq!(world.len(), 1);
        let expected = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        assert!(world[0].abs_diff_eq(expected, 1e-5));
    }

    #[test]
    fn test_compute_world_transforms_hierarchy() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.add_bone(Bone::new("child").with_parent(0));

        let poses = vec![
            Transform::from_position(Vec3::new(10.0, 0.0, 0.0)),
            Transform::from_position(Vec3::new(5.0, 0.0, 0.0)),
        ];
        let world = skeleton.compute_world_transforms(&poses);

        // Child world = parent world * child local
        // = translate(10,0,0) * translate(5,0,0) = translate(15,0,0)
        let expected = Mat4::from_translation(Vec3::new(15.0, 0.0, 0.0));
        assert!(world[1].abs_diff_eq(expected, 1e-5));
    }

    #[test]
    fn test_compute_world_transforms_rotation_hierarchy() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.add_bone(Bone::new("child").with_parent(0));

        let poses = vec![
            Transform::from_rotation(Quat::from_rotation_y(PI / 2.0)),
            Transform::from_position(Vec3::new(1.0, 0.0, 0.0)),
        ];
        let world = skeleton.compute_world_transforms(&poses);

        // Child at (1,0,0) rotated 90 deg around Y becomes (0,0,-1)
        let child_pos = world[1].w_axis.truncate();
        assert!(child_pos.abs_diff_eq(Vec3::new(0.0, 0.0, -1.0), 1e-5));
    }

    #[test]
    fn test_compute_skinning_matrices() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(
            Bone::new("root").with_inverse_bind_matrix(Mat4::from_translation(Vec3::new(
                -1.0, 0.0, 0.0,
            ))),
        );

        let poses = vec![Transform::from_position(Vec3::new(2.0, 0.0, 0.0))];
        let skinning = skeleton.compute_skinning_matrices(&poses);

        // Skinning = world * inverse_bind
        // = translate(2,0,0) * translate(-1,0,0) = translate(1,0,0)
        let expected = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        assert!(skinning[0].abs_diff_eq(expected, 1e-5));
    }

    #[test]
    fn test_compute_bind_pose_world_transforms() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(
            Bone::root("root").with_local_transform(Transform::from_position(Vec3::new(
                0.0, 1.0, 0.0,
            ))),
        );
        skeleton.add_bone(
            Bone::new("child")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 2.0, 0.0))),
        );

        let world = skeleton.compute_bind_pose_world_transforms();

        assert!(world[0]
            .w_axis
            .truncate()
            .abs_diff_eq(Vec3::new(0.0, 1.0, 0.0), 1e-5));
        assert!(world[1]
            .w_axis
            .truncate()
            .abs_diff_eq(Vec3::new(0.0, 3.0, 0.0), 1e-5));
    }

    #[test]
    fn test_compute_inverse_bind_matrices() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(
            Bone::root("root").with_local_transform(Transform::from_position(Vec3::new(
                0.0, 1.0, 0.0,
            ))),
        );

        skeleton.compute_inverse_bind_matrices().unwrap();

        let bone = skeleton.bone(0).unwrap();
        // Inverse of translate(0,1,0) is translate(0,-1,0)
        let expected = Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0));
        assert!(bone.inverse_bind_matrix.abs_diff_eq(expected, 1e-5));
    }

    // ===== Validation Tests =====

    #[test]
    fn test_validate_empty_skeleton() {
        let skeleton = Skeleton::new();
        assert!(skeleton.validate().is_ok());
    }

    #[test]
    fn test_validate_simple_skeleton() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.add_bone(Bone::new("child").with_parent(0));
        assert!(skeleton.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_parent_index() {
        let mut skeleton = Skeleton::new();
        skeleton.bones.push(Bone {
            name: "bone".to_string(),
            parent_index: Some(999), // Invalid
            local_transform: Transform::IDENTITY,
            inverse_bind_matrix: Mat4::IDENTITY,
        });

        let err = skeleton.validate().unwrap_err();
        match err {
            SkeletonError::InvalidParentIndex { parent_index, .. } => {
                assert_eq!(parent_index, 999);
            }
            _ => panic!("expected InvalidParentIndex error"),
        }
    }

    #[test]
    fn test_validate_duplicate_names() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::new("duplicate"));
        skeleton.bones.push(Bone::new("duplicate")); // Bypass add_bone to create duplicate

        let err = skeleton.validate().unwrap_err();
        match err {
            SkeletonError::DuplicateBoneName { name, indices } => {
                assert_eq!(name, "duplicate");
                assert_eq!(indices, vec![0, 1]);
            }
            _ => panic!("expected DuplicateBoneName error"),
        }
    }

    #[test]
    fn test_validate_parent_not_before_child() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.bones.push(Bone {
            name: "child".to_string(),
            parent_index: Some(2), // Points to a later bone
            local_transform: Transform::IDENTITY,
            inverse_bind_matrix: Mat4::IDENTITY,
        });
        skeleton.bones.push(Bone {
            name: "later".to_string(),
            parent_index: Some(0),
            local_transform: Transform::IDENTITY,
            inverse_bind_matrix: Mat4::IDENTITY,
        });

        let err = skeleton.validate().unwrap_err();
        match err {
            SkeletonError::ParentNotBeforeChild { bone_index, .. } => {
                assert_eq!(bone_index, 1);
            }
            _ => panic!("expected ParentNotBeforeChild error"),
        }
    }

    #[test]
    fn test_validate_too_many_bones() {
        let mut skeleton = Skeleton::new();
        for i in 0..=MAX_BONES {
            let bone = if i == 0 {
                Bone::root(format!("bone_{}", i))
            } else {
                Bone::new(format!("bone_{}", i)).with_parent(0)
            };
            skeleton.bones.push(bone);
        }

        let err = skeleton.validate().unwrap_err();
        match err {
            SkeletonError::TooManyBones { count, max } => {
                assert_eq!(count, MAX_BONES + 1);
                assert_eq!(max, MAX_BONES);
            }
            _ => panic!("expected TooManyBones error"),
        }
    }

    #[test]
    fn test_validate_strict_root_not_at_zero() {
        let mut skeleton = Skeleton::new();
        skeleton.bones.push(Bone::new("child").with_parent(1)); // Child first
        skeleton.bones.push(Bone::root("root")); // Root second
        skeleton.root_indices.push(1);

        // Regular validation should fail for parent ordering
        assert!(skeleton.validate().is_err());
    }

    // ===== Serialization Tests =====

    #[test]
    fn test_skeleton_json_roundtrip() {
        let mut original = Skeleton::new();
        original.add_bone(
            Bone::root("root").with_local_transform(Transform::from_position(Vec3::new(
                0.0, 1.0, 0.0,
            ))),
        );
        original.add_bone(
            Bone::new("child")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 2.0, 0.0))),
        );

        let json = original.to_json().unwrap();
        let recovered = Skeleton::from_json(&json).unwrap();

        assert_eq!(original.bone_count(), recovered.bone_count());
        assert_eq!(
            original.bone(0).unwrap().name,
            recovered.bone(0).unwrap().name
        );
        assert_eq!(
            original.bone(1).unwrap().parent_index,
            recovered.bone(1).unwrap().parent_index
        );
    }

    #[test]
    fn test_skeleton_rebuild_indices() {
        let mut skeleton = Skeleton::new();
        skeleton.bones.push(Bone::root("root"));
        skeleton.bones.push(Bone::new("child").with_parent(0));
        // name_to_index and root_indices are empty

        skeleton.rebuild_indices();

        assert_eq!(skeleton.bone_index("root"), Some(0));
        assert_eq!(skeleton.bone_index("child"), Some(1));
        assert_eq!(skeleton.root_indices(), &[0]);
    }

    #[test]
    fn test_transform_json_roundtrip() {
        let original = Transform::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_y(PI / 4.0),
            Vec3::new(1.5, 1.5, 1.5),
        );

        let json = serde_json::to_string(&original).unwrap();
        let recovered: Transform = serde_json::from_str(&json).unwrap();

        assert!(recovered.approx_eq(&original, 1e-5));
    }

    #[test]
    fn test_bone_json_roundtrip() {
        let original = Bone::new("test")
            .with_parent(0)
            .with_local_transform(Transform::from_position(Vec3::new(1.0, 2.0, 3.0)))
            .with_inverse_bind_matrix(Mat4::from_translation(Vec3::new(-1.0, -2.0, -3.0)));

        let json = serde_json::to_string(&original).unwrap();
        let recovered: Bone = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered.name, original.name);
        assert_eq!(recovered.parent_index, original.parent_index);
    }

    // ===== Builder Tests =====

    #[test]
    fn test_skeleton_builder() {
        let skeleton = SkeletonBuilder::new()
            .root("root")
            .child("spine", "root")
            .child("head", "spine")
            .build_unchecked();

        assert_eq!(skeleton.bone_count(), 3);
        assert_eq!(skeleton.bone(1).unwrap().parent_index, Some(0));
        assert_eq!(skeleton.bone(2).unwrap().parent_index, Some(1));
    }

    #[test]
    fn test_skeleton_builder_with_positions() {
        let skeleton = SkeletonBuilder::new()
            .root("root")
            .child_at("arm", "root", Vec3::new(1.0, 0.0, 0.0))
            .build_unchecked();

        let arm = skeleton.bone_by_name("arm").unwrap();
        assert!(arm
            .local_transform
            .position
            .abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_skeleton_builder_build() {
        let result = SkeletonBuilder::new()
            .root("root")
            .child("child", "root")
            .build();

        assert!(result.is_ok());
        let skeleton = result.unwrap();
        // Inverse bind matrices should be computed
        let root = skeleton.bone(0).unwrap();
        assert!(root.inverse_bind_matrix.abs_diff_eq(Mat4::IDENTITY, 1e-5));
    }

    // ===== Inverse Bind Matrix Tests =====

    #[test]
    fn test_inverse_bind_matrix_identity_hierarchy() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.add_bone(Bone::new("child").with_parent(0));

        skeleton.compute_inverse_bind_matrices().unwrap();

        // Both should have identity inverse bind matrices
        assert!(skeleton.bone(0).unwrap().inverse_bind_matrix.abs_diff_eq(Mat4::IDENTITY, 1e-5));
        assert!(skeleton.bone(1).unwrap().inverse_bind_matrix.abs_diff_eq(Mat4::IDENTITY, 1e-5));
    }

    #[test]
    fn test_inverse_bind_matrix_translated_hierarchy() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(
            Bone::root("root").with_local_transform(Transform::from_position(Vec3::new(
                0.0, 1.0, 0.0,
            ))),
        );
        skeleton.add_bone(
            Bone::new("child")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 2.0, 0.0))),
        );

        skeleton.compute_inverse_bind_matrices().unwrap();

        // Root world pos = (0, 1, 0), inverse = (0, -1, 0)
        let root_inv = skeleton.bone(0).unwrap().inverse_bind_matrix;
        let expected_root = Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0));
        assert!(root_inv.abs_diff_eq(expected_root, 1e-5));

        // Child world pos = (0, 3, 0), inverse = (0, -3, 0)
        let child_inv = skeleton.bone(1).unwrap().inverse_bind_matrix;
        let expected_child = Mat4::from_translation(Vec3::new(0.0, -3.0, 0.0));
        assert!(child_inv.abs_diff_eq(expected_child, 1e-5));
    }

    #[test]
    fn test_inverse_bind_matrix_degenerate() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(
            Bone::root("root")
                .with_local_transform(Transform::from_scale_vec(Vec3::new(0.0, 1.0, 1.0))),
        );

        let err = skeleton.compute_inverse_bind_matrices().unwrap_err();
        match err {
            SkeletonError::DegenerateInverseBindMatrix { bone_name, .. } => {
                assert_eq!(bone_name, "root");
            }
            _ => panic!("expected DegenerateInverseBindMatrix error"),
        }
    }

    // ===== Error Display Tests =====

    #[test]
    fn test_skeleton_error_display() {
        let err = SkeletonError::CircularParentReference {
            bone_index: 5,
            bone_name: "test".to_string(),
            cycle_path: vec![3, 4, 5],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("circular"));
        assert!(msg.contains("test"));

        let err = SkeletonError::InvalidParentIndex {
            bone_index: 1,
            bone_name: "child".to_string(),
            parent_index: 999,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("invalid parent"));
        assert!(msg.contains("999"));

        let err = SkeletonError::DuplicateBoneName {
            name: "dup".to_string(),
            indices: vec![0, 5],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("duplicate"));
        assert!(msg.contains("dup"));

        let err = SkeletonError::TooManyBones {
            count: 300,
            max: 256,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("300"));
        assert!(msg.contains("256"));
    }

    // ===== Additional Edge Case Tests =====

    #[test]
    fn test_skeleton_bone_mut() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::new("root"));

        if let Some(bone) = skeleton.bone_mut(0) {
            bone.local_transform = Transform::from_position(Vec3::new(5.0, 0.0, 0.0));
        }

        assert!(skeleton.bone(0).unwrap().local_transform.position.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_skeleton_bones_slice() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("a"));
        skeleton.add_bone(Bone::new("b").with_parent(0));
        skeleton.add_bone(Bone::new("c").with_parent(0));

        let bones = skeleton.bones();
        assert_eq!(bones.len(), 3);
        assert_eq!(bones[0].name, "a");
        assert_eq!(bones[1].name, "b");
        assert_eq!(bones[2].name, "c");
    }

    #[test]
    fn test_transform_from_scale() {
        let t = Transform::from_scale(2.0);
        assert_eq!(t.scale, Vec3::splat(2.0));
        assert_eq!(t.position, Vec3::ZERO);
        assert_eq!(t.rotation, Quat::IDENTITY);
    }

    #[test]
    fn test_skeleton_with_capacity() {
        let skeleton = Skeleton::with_capacity(100);
        assert_eq!(skeleton.bone_count(), 0);
    }

    #[test]
    fn test_skeleton_equality() {
        let mut s1 = Skeleton::new();
        s1.add_bone(Bone::root("root"));

        let mut s2 = Skeleton::new();
        s2.add_bone(Bone::root("root"));

        assert_eq!(s1, s2);

        s2.add_bone(Bone::new("child").with_parent(0));
        assert_ne!(s1, s2);
    }

    #[test]
    #[should_panic(expected = "parent index")]
    fn test_add_bone_invalid_parent_panics() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::new("child").with_parent(0)); // No bones exist yet
    }
}
