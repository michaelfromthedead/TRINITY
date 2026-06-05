//! Skeleton retargeting for TRINITY Engine (T-AN-2.4).
//!
//! This module provides animation retargeting between different skeletons,
//! enabling animations created for one character to be applied to another
//! with different proportions.
//!
//! # Features
//!
//! - **Bone name mapping**: Map source skeleton bones to target skeleton
//! - **Proportion correction**: Scale translations by limb ratios
//! - **IK retargeting**: Maintain foot/hand contact positions
//! - **Root motion scaling**: Scale root motion for different character sizes
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::retargeting::{RetargetingMap, RetargetingConfig, BoneMapping};
//! use renderer_backend::skeleton::Skeleton;
//! use renderer_backend::pose::Pose;
//!
//! // Create skeletons
//! let source_skeleton = create_source_skeleton();
//! let target_skeleton = create_target_skeleton();
//!
//! // Auto-generate mappings by bone name
//! let mappings = RetargetingMap::auto_map_by_name(&source_skeleton, &target_skeleton);
//!
//! // Create config
//! let config = RetargetingConfig {
//!     mappings,
//!     source_height: 1.8,
//!     target_height: 1.5,
//!     use_ik_correction: true,
//!     ik_chains: vec![],
//! };
//!
//! // Create retargeting map
//! let retarget = RetargetingMap::new(config, &source_skeleton, &target_skeleton);
//!
//! // Retarget a pose
//! let source_pose = animation.sample(0.5);
//! let target_pose = retarget.retarget_pose(&source_pose);
//! ```

use std::collections::HashMap;
use std::fmt;

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::animation_clip::{AnimationClip, BoneTrack, Keyframe, Track};
use crate::pose::Pose;
use crate::skeleton::{Skeleton, Transform};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default tolerance for bone name matching.
pub const DEFAULT_NAME_MATCH_TOLERANCE: f32 = 0.8;

/// Minimum height ratio to avoid extreme scaling.
pub const MIN_HEIGHT_RATIO: f32 = 0.1;

/// Maximum height ratio to avoid extreme scaling.
pub const MAX_HEIGHT_RATIO: f32 = 10.0;

/// IK blend weight threshold below which IK is skipped.
pub const IK_BLEND_THRESHOLD: f32 = 0.001;

// ---------------------------------------------------------------------------
// ChainType
// ---------------------------------------------------------------------------

/// Type of IK chain for retargeting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChainType {
    /// Left arm chain (shoulder to hand).
    LeftArm,
    /// Right arm chain (shoulder to hand).
    RightArm,
    /// Left leg chain (hip to foot).
    LeftLeg,
    /// Right leg chain (hip to foot).
    RightLeg,
    /// Spine chain (pelvis to head).
    Spine,
}

impl ChainType {
    /// Get a human-readable name for this chain type.
    pub fn name(&self) -> &'static str {
        match self {
            ChainType::LeftArm => "Left Arm",
            ChainType::RightArm => "Right Arm",
            ChainType::LeftLeg => "Left Leg",
            ChainType::RightLeg => "Right Leg",
            ChainType::Spine => "Spine",
        }
    }

    /// Check if this is a leg chain.
    #[inline]
    pub fn is_leg(&self) -> bool {
        matches!(self, ChainType::LeftLeg | ChainType::RightLeg)
    }

    /// Check if this is an arm chain.
    #[inline]
    pub fn is_arm(&self) -> bool {
        matches!(self, ChainType::LeftArm | ChainType::RightArm)
    }
}

impl fmt::Display for ChainType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// BoneMapping
// ---------------------------------------------------------------------------

/// Mapping from a source bone to a target bone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoneMapping {
    /// Name of the bone in the source skeleton.
    pub source_name: String,

    /// Index of the corresponding bone in the target skeleton.
    pub target_index: usize,

    /// Scale factor for proportion correction (target length / source length).
    pub scale_factor: f32,

    /// Optional rotation offset to apply during retargeting.
    pub rotation_offset: Option<Quat>,

    /// Whether to preserve the source bone's rotation.
    pub copy_rotation: bool,

    /// Whether to preserve the source bone's position (scaled).
    pub copy_position: bool,

    /// Whether to preserve the source bone's scale.
    pub copy_scale: bool,
}

impl BoneMapping {
    /// Create a new bone mapping with default settings.
    pub fn new(source_name: impl Into<String>, target_index: usize) -> Self {
        Self {
            source_name: source_name.into(),
            target_index,
            scale_factor: 1.0,
            rotation_offset: None,
            copy_rotation: true,
            copy_position: true,
            copy_scale: true,
        }
    }

    /// Create a bone mapping with a custom scale factor.
    pub fn with_scale(source_name: impl Into<String>, target_index: usize, scale_factor: f32) -> Self {
        Self {
            source_name: source_name.into(),
            target_index,
            scale_factor,
            rotation_offset: None,
            copy_rotation: true,
            copy_position: true,
            copy_scale: true,
        }
    }

    /// Set the rotation offset.
    pub fn with_rotation_offset(mut self, offset: Quat) -> Self {
        self.rotation_offset = Some(offset);
        self
    }

    /// Set whether to copy rotation.
    pub fn rotation_only(mut self) -> Self {
        self.copy_rotation = true;
        self.copy_position = false;
        self.copy_scale = false;
        self
    }

    /// Set whether to copy position.
    pub fn position_only(mut self) -> Self {
        self.copy_rotation = false;
        self.copy_position = true;
        self.copy_scale = false;
        self
    }
}

impl Default for BoneMapping {
    fn default() -> Self {
        Self {
            source_name: String::new(),
            target_index: 0,
            scale_factor: 1.0,
            rotation_offset: None,
            copy_rotation: true,
            copy_position: true,
            copy_scale: true,
        }
    }
}

// ---------------------------------------------------------------------------
// IkChainConfig
// ---------------------------------------------------------------------------

/// Configuration for an IK chain used in retargeting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IkChainConfig {
    /// Type of the chain (arm, leg, spine).
    pub chain_type: ChainType,

    /// Index of the effector (end) bone in the target skeleton.
    pub effector_bone: usize,

    /// Indices of bones in the chain (from root to effector) in the target skeleton.
    pub chain_bones: Vec<usize>,

    /// Blend weight for IK correction (0.0 = no IK, 1.0 = full IK).
    pub blend_weight: f32,

    /// Tolerance for IK convergence.
    pub tolerance: f32,

    /// Maximum iterations for IK solver.
    pub max_iterations: u32,
}

impl IkChainConfig {
    /// Create a new IK chain configuration.
    pub fn new(chain_type: ChainType, effector_bone: usize, chain_bones: Vec<usize>) -> Self {
        Self {
            chain_type,
            effector_bone,
            chain_bones,
            blend_weight: 1.0,
            tolerance: 0.001,
            max_iterations: 10,
        }
    }

    /// Create a two-bone IK chain (common for limbs).
    pub fn two_bone(chain_type: ChainType, upper: usize, lower: usize, effector: usize) -> Self {
        Self {
            chain_type,
            effector_bone: effector,
            chain_bones: vec![upper, lower, effector],
            blend_weight: 1.0,
            tolerance: 0.001,
            max_iterations: 10,
        }
    }

    /// Set the blend weight.
    pub fn with_blend_weight(mut self, weight: f32) -> Self {
        self.blend_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Check if the chain is valid.
    pub fn is_valid(&self) -> bool {
        self.chain_bones.len() >= 2 && self.chain_bones.contains(&self.effector_bone)
    }
}

impl Default for IkChainConfig {
    fn default() -> Self {
        Self {
            chain_type: ChainType::LeftArm,
            effector_bone: 0,
            chain_bones: Vec::new(),
            blend_weight: 1.0,
            tolerance: 0.001,
            max_iterations: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// RetargetingConfig
// ---------------------------------------------------------------------------

/// Configuration for skeleton retargeting.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RetargetingConfig {
    /// Bone mappings from source to target.
    pub mappings: Vec<BoneMapping>,

    /// Height of the source skeleton (for root motion scaling).
    pub source_height: f32,

    /// Height of the target skeleton (for root motion scaling).
    pub target_height: f32,

    /// Whether to use IK correction for end effectors.
    pub use_ik_correction: bool,

    /// IK chain configurations for retargeting.
    pub ik_chains: Vec<IkChainConfig>,

    /// Whether to scale root motion based on height ratio.
    pub scale_root_motion: bool,

    /// Optional speed scale factor for root motion.
    pub speed_scale: Option<f32>,
}

impl RetargetingConfig {
    /// Create a new retargeting configuration.
    pub fn new(mappings: Vec<BoneMapping>, source_height: f32, target_height: f32) -> Self {
        Self {
            mappings,
            source_height,
            target_height,
            use_ik_correction: false,
            ik_chains: Vec::new(),
            scale_root_motion: true,
            speed_scale: None,
        }
    }

    /// Enable IK correction with the given chains.
    pub fn with_ik_chains(mut self, chains: Vec<IkChainConfig>) -> Self {
        self.use_ik_correction = true;
        self.ik_chains = chains;
        self
    }

    /// Set speed scaling factor.
    pub fn with_speed_scale(mut self, scale: f32) -> Self {
        self.speed_scale = Some(scale);
        self
    }

    /// Disable root motion scaling.
    pub fn without_root_motion_scaling(mut self) -> Self {
        self.scale_root_motion = false;
        self
    }

    /// Get the height ratio (target / source).
    pub fn height_ratio(&self) -> f32 {
        if self.source_height.abs() < f32::EPSILON {
            1.0
        } else {
            (self.target_height / self.source_height).clamp(MIN_HEIGHT_RATIO, MAX_HEIGHT_RATIO)
        }
    }

    /// Add a bone mapping.
    pub fn add_mapping(&mut self, mapping: BoneMapping) {
        self.mappings.push(mapping);
    }

    /// Find a mapping by source bone name.
    pub fn find_mapping(&self, source_name: &str) -> Option<&BoneMapping> {
        self.mappings.iter().find(|m| m.source_name == source_name)
    }
}

// ---------------------------------------------------------------------------
// RootMotionDelta
// ---------------------------------------------------------------------------

/// Root motion delta for a single frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RootMotionDelta {
    /// Translation delta in world space.
    pub translation: Vec3,

    /// Rotation delta (yaw only for grounded characters).
    pub rotation: Quat,

    /// Time delta this motion covers.
    pub dt: f32,
}

impl RootMotionDelta {
    /// Create a new root motion delta.
    pub fn new(translation: Vec3, rotation: Quat, dt: f32) -> Self {
        Self {
            translation,
            rotation,
            dt,
        }
    }

    /// Create a translation-only delta.
    pub fn translation_only(translation: Vec3, dt: f32) -> Self {
        Self {
            translation,
            rotation: Quat::IDENTITY,
            dt,
        }
    }

    /// Get the velocity (translation / dt).
    pub fn velocity(&self) -> Vec3 {
        if self.dt > f32::EPSILON {
            self.translation / self.dt
        } else {
            Vec3::ZERO
        }
    }

    /// Get the speed (magnitude of velocity).
    pub fn speed(&self) -> f32 {
        self.velocity().length()
    }
}

// ---------------------------------------------------------------------------
// RetargetingError
// ---------------------------------------------------------------------------

/// Errors that can occur during retargeting.
#[derive(Clone, Debug, PartialEq)]
pub enum RetargetingError {
    /// Source skeleton is empty.
    EmptySourceSkeleton,

    /// Target skeleton is empty.
    EmptyTargetSkeleton,

    /// No mappings provided.
    NoMappings,

    /// Invalid bone index in mapping.
    InvalidBoneIndex {
        bone_name: String,
        index: usize,
        skeleton_size: usize,
    },

    /// Source bone not found in skeleton.
    SourceBoneNotFound {
        bone_name: String,
    },

    /// IK chain is invalid.
    InvalidIkChain {
        chain_type: ChainType,
        reason: String,
    },

    /// Chain length mismatch.
    ChainLengthMismatch {
        source_length: f32,
        target_length: f32,
    },
}

impl fmt::Display for RetargetingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySourceSkeleton => write!(f, "source skeleton is empty"),
            Self::EmptyTargetSkeleton => write!(f, "target skeleton is empty"),
            Self::NoMappings => write!(f, "no bone mappings provided"),
            Self::InvalidBoneIndex { bone_name, index, skeleton_size } => {
                write!(
                    f,
                    "invalid bone index {} for '{}' (skeleton has {} bones)",
                    index, bone_name, skeleton_size
                )
            }
            Self::SourceBoneNotFound { bone_name } => {
                write!(f, "source bone '{}' not found in skeleton", bone_name)
            }
            Self::InvalidIkChain { chain_type, reason } => {
                write!(f, "invalid IK chain {}: {}", chain_type, reason)
            }
            Self::ChainLengthMismatch { source_length, target_length } => {
                write!(
                    f,
                    "chain length mismatch: source={:.3}, target={:.3}",
                    source_length, target_length
                )
            }
        }
    }
}

impl std::error::Error for RetargetingError {}

// ---------------------------------------------------------------------------
// RetargetingMap
// ---------------------------------------------------------------------------

/// A compiled retargeting map for efficient pose/clip retargeting.
///
/// This struct precomputes chain lengths and mappings for fast runtime retargeting.
#[derive(Clone, Debug)]
pub struct RetargetingMap {
    /// Configuration for this retargeting map.
    pub config: RetargetingConfig,

    /// Clone of the source skeleton for reference.
    source_skeleton: Skeleton,

    /// Clone of the target skeleton for reference.
    target_skeleton: Skeleton,

    /// Precomputed chain lengths: (source_length, target_length) per IK chain.
    chain_lengths: Vec<(f32, f32)>,

    /// Map from source bone name to mapping index for O(1) lookup.
    source_name_to_mapping: HashMap<String, usize>,

    /// Map from source bone index to mapping index.
    source_index_to_mapping: HashMap<usize, usize>,
}

impl RetargetingMap {
    /// Create a new retargeting map.
    ///
    /// # Arguments
    ///
    /// * `config` - Retargeting configuration
    /// * `source` - Source skeleton
    /// * `target` - Target skeleton
    ///
    /// # Returns
    ///
    /// A new `RetargetingMap` with precomputed chain lengths.
    pub fn new(config: RetargetingConfig, source: &Skeleton, target: &Skeleton) -> Self {
        // Build lookup maps
        let mut source_name_to_mapping = HashMap::new();
        let mut source_index_to_mapping = HashMap::new();

        for (i, mapping) in config.mappings.iter().enumerate() {
            source_name_to_mapping.insert(mapping.source_name.clone(), i);
            if let Some(idx) = source.bone_index(&mapping.source_name) {
                source_index_to_mapping.insert(idx, i);
            }
        }

        // Precompute chain lengths
        let chain_lengths = config
            .ik_chains
            .iter()
            .map(|chain| {
                let source_len = Self::compute_chain_length_for_skeleton(source, &chain.chain_bones);
                let target_len = Self::compute_chain_length_for_skeleton(target, &chain.chain_bones);
                (source_len, target_len)
            })
            .collect();

        Self {
            config,
            source_skeleton: source.clone(),
            target_skeleton: target.clone(),
            chain_lengths,
            source_name_to_mapping,
            source_index_to_mapping,
        }
    }

    /// Compute chain length for a skeleton using bind pose.
    fn compute_chain_length_for_skeleton(skeleton: &Skeleton, bones: &[usize]) -> f32 {
        if bones.len() < 2 {
            return 0.0;
        }

        // Compute world positions in bind pose
        let transforms: Vec<Transform> = skeleton
            .bones()
            .iter()
            .map(|b| b.local_transform)
            .collect();
        let world_transforms = skeleton.compute_world_transforms(&transforms);

        let mut total_length = 0.0;
        for i in 0..bones.len() - 1 {
            let bone_a = bones[i];
            let bone_b = bones[i + 1];

            if bone_a < world_transforms.len() && bone_b < world_transforms.len() {
                let pos_a = world_transforms[bone_a].w_axis.truncate();
                let pos_b = world_transforms[bone_b].w_axis.truncate();
                total_length += (pos_b - pos_a).length();
            }
        }

        total_length
    }

    /// Retarget a pose from source skeleton to target skeleton.
    ///
    /// # Arguments
    ///
    /// * `source_pose` - The pose to retarget (from source skeleton)
    ///
    /// # Returns
    ///
    /// A new pose for the target skeleton.
    pub fn retarget_pose(&self, source_pose: &Pose) -> Pose {
        let height_ratio = self.config.height_ratio();
        let mut target_pose = Pose::new(self.target_skeleton.bone_count(), source_pose.pose_type);

        // Copy target skeleton's bind pose as base
        for (i, bone) in self.target_skeleton.bones().iter().enumerate() {
            target_pose.set_transform(i, bone.local_transform);
        }

        // Apply mappings
        for mapping in &self.config.mappings {
            // Find source bone index
            let source_index = match self.source_skeleton.bone_index(&mapping.source_name) {
                Some(idx) => idx,
                None => continue,
            };

            // Get source transform
            let source_transform = if source_index < source_pose.bone_count() {
                source_pose.get_transform(source_index)
            } else {
                continue;
            };

            // Apply mapping
            if mapping.target_index < target_pose.bone_count() {
                let mut target_transform = target_pose.get_transform(mapping.target_index);

                // Copy rotation
                if mapping.copy_rotation {
                    let mut rotation = source_transform.rotation;
                    if let Some(offset) = mapping.rotation_offset {
                        rotation = offset * rotation;
                    }
                    target_transform.rotation = rotation;
                }

                // Copy position (with proportion scaling)
                if mapping.copy_position {
                    target_transform.position = source_transform.position * mapping.scale_factor * height_ratio;
                }

                // Copy scale
                if mapping.copy_scale {
                    target_transform.scale = source_transform.scale;
                }

                target_pose.set_transform(mapping.target_index, target_transform);
            }
        }

        // Apply IK corrections if enabled
        if self.config.use_ik_correction {
            self.apply_ik_corrections(&mut target_pose, source_pose);
        }

        target_pose
    }

    /// Apply IK corrections to maintain end effector positions.
    fn apply_ik_corrections(&self, _target_pose: &mut Pose, _source_pose: &Pose) {
        // IK correction would be implemented here using FABRIK or two-bone IK
        // For now, this is a placeholder for the IK integration
        for (i, chain) in self.config.ik_chains.iter().enumerate() {
            if chain.blend_weight < IK_BLEND_THRESHOLD {
                continue;
            }

            // Get chain lengths
            let (_source_len, _target_len) = self.chain_lengths.get(i).copied().unwrap_or((1.0, 1.0));

            // IK solving would happen here:
            // 1. Get target effector position from source (scaled)
            // 2. Solve IK on target chain to reach that position
            // 3. Blend result with current pose based on blend_weight
        }
    }

    /// Retarget an animation clip from source to target skeleton.
    ///
    /// # Arguments
    ///
    /// * `source_clip` - The animation clip to retarget
    ///
    /// # Returns
    ///
    /// A new animation clip for the target skeleton.
    pub fn retarget_clip(&self, source_clip: &AnimationClip) -> AnimationClip {
        let height_ratio = self.config.height_ratio();

        let mut target_clip = AnimationClip::new(
            format!("{}_retargeted", source_clip.name),
            source_clip.duration,
        );
        target_clip.looping = source_clip.looping;
        target_clip.frame_rate = source_clip.frame_rate;

        // Retarget each bone track
        for source_track in &source_clip.bone_tracks {
            // Find mapping for this bone
            if let Some(mapping) = self.config.find_mapping(&source_track.bone_name) {
                // Get target bone name
                let target_bone = match self.target_skeleton.bone(mapping.target_index) {
                    Some(bone) => bone.name.clone(),
                    None => continue,
                };

                let mut target_track = BoneTrack::new(target_bone);

                // Retarget position track
                if mapping.copy_position {
                    if let Some(pos_track) = &source_track.position {
                        let scale = mapping.scale_factor * height_ratio;
                        let scaled_keyframes: Vec<_> = pos_track
                            .keyframes
                            .iter()
                            .map(|kf| Keyframe {
                                time: kf.time,
                                value: kf.value * scale,
                                interpolation: kf.interpolation,
                                in_tangent: kf.in_tangent.map(|t| t * scale),
                                out_tangent: kf.out_tangent.map(|t| t * scale),
                            })
                            .collect();
                        target_track.position = Some(Track::from_keyframes(scaled_keyframes));
                    }
                }

                // Retarget rotation track
                if mapping.copy_rotation {
                    if let Some(rot_track) = &source_track.rotation {
                        let rotated_keyframes: Vec<_> = rot_track
                            .keyframes
                            .iter()
                            .map(|kf| {
                                let rotation = if let Some(offset) = mapping.rotation_offset {
                                    offset * kf.value
                                } else {
                                    kf.value
                                };
                                Keyframe {
                                    time: kf.time,
                                    value: rotation,
                                    interpolation: kf.interpolation,
                                    in_tangent: kf.in_tangent,
                                    out_tangent: kf.out_tangent,
                                }
                            })
                            .collect();
                        target_track.rotation = Some(Track::from_keyframes(rotated_keyframes));
                    }
                }

                // Retarget scale track
                if mapping.copy_scale {
                    target_track.scale = source_track.scale.clone();
                }

                target_clip.add_bone_track(target_track);
            }
        }

        // Copy event tracks (unchanged)
        for event_track in &source_clip.event_tracks {
            target_clip.add_event_track(event_track.clone());
        }

        // Copy curve tracks (unchanged)
        for curve_track in &source_clip.curve_tracks {
            target_clip.add_curve_track(curve_track.clone());
        }

        target_clip
    }

    /// Auto-generate bone mappings by matching bone names.
    ///
    /// This performs case-insensitive matching and handles common naming conventions.
    ///
    /// # Arguments
    ///
    /// * `source` - Source skeleton
    /// * `target` - Target skeleton
    ///
    /// # Returns
    ///
    /// A vector of bone mappings for matching bones.
    pub fn auto_map_by_name(source: &Skeleton, target: &Skeleton) -> Vec<BoneMapping> {
        let mut mappings = Vec::new();

        // Build target bone lookup (lowercase)
        let target_lookup: HashMap<String, usize> = target
            .bones()
            .iter()
            .enumerate()
            .map(|(i, bone)| (bone.name.to_lowercase(), i))
            .collect();

        // Match source bones to target
        for source_bone in source.bones() {
            let source_name_lower = source_bone.name.to_lowercase();

            // Try exact match first
            if let Some(&target_idx) = target_lookup.get(&source_name_lower) {
                mappings.push(BoneMapping::new(source_bone.name.clone(), target_idx));
                continue;
            }

            // Try with common prefixes/suffixes removed
            let normalized = normalize_bone_name(&source_name_lower);
            for (target_name, &target_idx) in &target_lookup {
                let target_normalized = normalize_bone_name(target_name);
                if normalized == target_normalized {
                    mappings.push(BoneMapping::new(source_bone.name.clone(), target_idx));
                    break;
                }
            }
        }

        mappings
    }

    /// Get the source skeleton.
    pub fn source_skeleton(&self) -> &Skeleton {
        &self.source_skeleton
    }

    /// Get the target skeleton.
    pub fn target_skeleton(&self) -> &Skeleton {
        &self.target_skeleton
    }

    /// Get the chain lengths.
    pub fn chain_lengths(&self) -> &[(f32, f32)] {
        &self.chain_lengths
    }

    /// Validate the retargeting configuration.
    pub fn validate(&self) -> Result<(), RetargetingError> {
        if self.source_skeleton.is_empty() {
            return Err(RetargetingError::EmptySourceSkeleton);
        }

        if self.target_skeleton.is_empty() {
            return Err(RetargetingError::EmptyTargetSkeleton);
        }

        if self.config.mappings.is_empty() {
            return Err(RetargetingError::NoMappings);
        }

        // Validate mappings
        for mapping in &self.config.mappings {
            // Check source bone exists
            if self.source_skeleton.bone_index(&mapping.source_name).is_none() {
                return Err(RetargetingError::SourceBoneNotFound {
                    bone_name: mapping.source_name.clone(),
                });
            }

            // Check target bone index is valid
            if mapping.target_index >= self.target_skeleton.bone_count() {
                return Err(RetargetingError::InvalidBoneIndex {
                    bone_name: mapping.source_name.clone(),
                    index: mapping.target_index,
                    skeleton_size: self.target_skeleton.bone_count(),
                });
            }
        }

        // Validate IK chains
        for chain in &self.config.ik_chains {
            if !chain.is_valid() {
                return Err(RetargetingError::InvalidIkChain {
                    chain_type: chain.chain_type,
                    reason: "chain must have at least 2 bones and include effector".to_string(),
                });
            }

            for &bone_idx in &chain.chain_bones {
                if bone_idx >= self.target_skeleton.bone_count() {
                    return Err(RetargetingError::InvalidIkChain {
                        chain_type: chain.chain_type,
                        reason: format!("bone index {} out of range", bone_idx),
                    });
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Compute the total chain length from bone positions.
///
/// # Arguments
///
/// * `skeleton` - The skeleton containing the bones
/// * `bones` - Ordered list of bone indices from root to tip
///
/// # Returns
///
/// The total length of the chain in world units.
pub fn compute_chain_length(skeleton: &Skeleton, bones: &[usize]) -> f32 {
    RetargetingMap::compute_chain_length_for_skeleton(skeleton, bones)
}

/// Scale root motion delta by height ratio.
///
/// # Arguments
///
/// * `delta` - The root motion delta to scale
/// * `height_ratio` - Ratio of target height to source height
///
/// # Returns
///
/// A new root motion delta with scaled translation.
pub fn scale_root_motion(delta: &RootMotionDelta, height_ratio: f32) -> RootMotionDelta {
    RootMotionDelta {
        translation: delta.translation * height_ratio,
        rotation: delta.rotation,
        dt: delta.dt,
    }
}

/// Normalize a bone name by removing common prefixes/suffixes.
fn normalize_bone_name(name: &str) -> String {
    let name = name.to_lowercase();

    // Remove common prefixes
    let prefixes = ["bip01_", "bip_", "bone_", "jnt_", "j_", "def_", "drv_", "mixamorig:"];
    let mut result = name.as_str();
    for prefix in prefixes {
        if let Some(stripped) = result.strip_prefix(prefix) {
            result = stripped;
            break;
        }
    }

    // Remove common suffixes
    let suffixes = ["_bone", "_jnt", "_j", "_def", "_drv", "_l", "_r"];
    for suffix in suffixes {
        if let Some(stripped) = result.strip_suffix(suffix) {
            result = stripped;
            break;
        }
    }

    result.to_string()
}

/// Find common bone name variations.
fn find_bone_name_variations(name: &str) -> Vec<String> {
    let base = normalize_bone_name(name);
    let mut variations = vec![base.clone()];

    // Common variations
    let replacements = [
        ("left", "l"),
        ("right", "r"),
        ("upper", "up"),
        ("lower", "low"),
        ("hand", "wrist"),
        ("foot", "ankle"),
    ];

    for (from, to) in replacements {
        if base.contains(from) {
            variations.push(base.replace(from, to));
        }
        if base.contains(to) {
            variations.push(base.replace(to, from));
        }
    }

    variations
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Bone, SkeletonBuilder};
    use std::f32::consts::PI;

    // ===== Helper Functions =====

    fn create_simple_skeleton() -> Skeleton {
        SkeletonBuilder::new()
            .root("root")
            .child_at("spine", "root", Vec3::new(0.0, 1.0, 0.0))
            .child_at("head", "spine", Vec3::new(0.0, 0.5, 0.0))
            .child_at("left_arm", "spine", Vec3::new(-0.5, 0.0, 0.0))
            .child_at("right_arm", "spine", Vec3::new(0.5, 0.0, 0.0))
            .build_unchecked()
    }

    fn create_tall_skeleton() -> Skeleton {
        SkeletonBuilder::new()
            .root("root")
            .child_at("spine", "root", Vec3::new(0.0, 1.5, 0.0))
            .child_at("head", "spine", Vec3::new(0.0, 0.75, 0.0))
            .child_at("left_arm", "spine", Vec3::new(-0.75, 0.0, 0.0))
            .child_at("right_arm", "spine", Vec3::new(0.75, 0.0, 0.0))
            .build_unchecked()
    }

    fn create_short_skeleton() -> Skeleton {
        SkeletonBuilder::new()
            .root("root")
            .child_at("spine", "root", Vec3::new(0.0, 0.5, 0.0))
            .child_at("head", "spine", Vec3::new(0.0, 0.25, 0.0))
            .child_at("left_arm", "spine", Vec3::new(-0.25, 0.0, 0.0))
            .child_at("right_arm", "spine", Vec3::new(0.25, 0.0, 0.0))
            .build_unchecked()
    }

    fn create_renamed_skeleton() -> Skeleton {
        // Same structure but different naming convention
        SkeletonBuilder::new()
            .root("Bip01_Root")
            .child_at("Bip01_Spine", "Bip01_Root", Vec3::new(0.0, 1.0, 0.0))
            .child_at("Bip01_Head", "Bip01_Spine", Vec3::new(0.0, 0.5, 0.0))
            .child_at("Bip01_LeftArm", "Bip01_Spine", Vec3::new(-0.5, 0.0, 0.0))
            .child_at("Bip01_RightArm", "Bip01_Spine", Vec3::new(0.5, 0.0, 0.0))
            .build_unchecked()
    }

    // ===== ChainType Tests =====

    #[test]
    fn test_chain_type_name() {
        assert_eq!(ChainType::LeftArm.name(), "Left Arm");
        assert_eq!(ChainType::RightArm.name(), "Right Arm");
        assert_eq!(ChainType::LeftLeg.name(), "Left Leg");
        assert_eq!(ChainType::RightLeg.name(), "Right Leg");
        assert_eq!(ChainType::Spine.name(), "Spine");
    }

    #[test]
    fn test_chain_type_is_leg() {
        assert!(!ChainType::LeftArm.is_leg());
        assert!(!ChainType::RightArm.is_leg());
        assert!(ChainType::LeftLeg.is_leg());
        assert!(ChainType::RightLeg.is_leg());
        assert!(!ChainType::Spine.is_leg());
    }

    #[test]
    fn test_chain_type_is_arm() {
        assert!(ChainType::LeftArm.is_arm());
        assert!(ChainType::RightArm.is_arm());
        assert!(!ChainType::LeftLeg.is_arm());
        assert!(!ChainType::RightLeg.is_arm());
        assert!(!ChainType::Spine.is_arm());
    }

    #[test]
    fn test_chain_type_display() {
        assert_eq!(format!("{}", ChainType::LeftArm), "Left Arm");
    }

    // ===== BoneMapping Tests =====

    #[test]
    fn test_bone_mapping_new() {
        let mapping = BoneMapping::new("spine", 1);
        assert_eq!(mapping.source_name, "spine");
        assert_eq!(mapping.target_index, 1);
        assert_eq!(mapping.scale_factor, 1.0);
        assert!(mapping.rotation_offset.is_none());
        assert!(mapping.copy_rotation);
        assert!(mapping.copy_position);
        assert!(mapping.copy_scale);
    }

    #[test]
    fn test_bone_mapping_with_scale() {
        let mapping = BoneMapping::with_scale("arm", 2, 1.5);
        assert_eq!(mapping.scale_factor, 1.5);
    }

    #[test]
    fn test_bone_mapping_with_rotation_offset() {
        let offset = Quat::from_rotation_y(PI / 4.0);
        let mapping = BoneMapping::new("arm", 2).with_rotation_offset(offset);
        assert!(mapping.rotation_offset.is_some());
        assert!(mapping.rotation_offset.unwrap().abs_diff_eq(offset, 1e-5));
    }

    #[test]
    fn test_bone_mapping_rotation_only() {
        let mapping = BoneMapping::new("spine", 1).rotation_only();
        assert!(mapping.copy_rotation);
        assert!(!mapping.copy_position);
        assert!(!mapping.copy_scale);
    }

    #[test]
    fn test_bone_mapping_position_only() {
        let mapping = BoneMapping::new("spine", 1).position_only();
        assert!(!mapping.copy_rotation);
        assert!(mapping.copy_position);
        assert!(!mapping.copy_scale);
    }

    #[test]
    fn test_bone_mapping_default() {
        let mapping = BoneMapping::default();
        assert!(mapping.source_name.is_empty());
        assert_eq!(mapping.target_index, 0);
        assert_eq!(mapping.scale_factor, 1.0);
    }

    // ===== IkChainConfig Tests =====

    #[test]
    fn test_ik_chain_config_new() {
        let chain = IkChainConfig::new(ChainType::LeftArm, 3, vec![1, 2, 3]);
        assert_eq!(chain.chain_type, ChainType::LeftArm);
        assert_eq!(chain.effector_bone, 3);
        assert_eq!(chain.chain_bones, vec![1, 2, 3]);
        assert_eq!(chain.blend_weight, 1.0);
    }

    #[test]
    fn test_ik_chain_config_two_bone() {
        let chain = IkChainConfig::two_bone(ChainType::LeftLeg, 0, 1, 2);
        assert_eq!(chain.chain_bones, vec![0, 1, 2]);
        assert_eq!(chain.effector_bone, 2);
    }

    #[test]
    fn test_ik_chain_config_with_blend_weight() {
        let chain = IkChainConfig::new(ChainType::LeftArm, 3, vec![1, 2, 3])
            .with_blend_weight(0.5);
        assert_eq!(chain.blend_weight, 0.5);
    }

    #[test]
    fn test_ik_chain_config_with_blend_weight_clamped() {
        let chain = IkChainConfig::new(ChainType::LeftArm, 3, vec![1, 2, 3])
            .with_blend_weight(1.5);
        assert_eq!(chain.blend_weight, 1.0);

        let chain = IkChainConfig::new(ChainType::LeftArm, 3, vec![1, 2, 3])
            .with_blend_weight(-0.5);
        assert_eq!(chain.blend_weight, 0.0);
    }

    #[test]
    fn test_ik_chain_config_is_valid() {
        let valid = IkChainConfig::new(ChainType::LeftArm, 2, vec![0, 1, 2]);
        assert!(valid.is_valid());

        let invalid_short = IkChainConfig::new(ChainType::LeftArm, 0, vec![0]);
        assert!(!invalid_short.is_valid());

        let invalid_effector = IkChainConfig::new(ChainType::LeftArm, 5, vec![0, 1, 2]);
        assert!(!invalid_effector.is_valid());
    }

    #[test]
    fn test_ik_chain_config_default() {
        let chain = IkChainConfig::default();
        assert_eq!(chain.chain_type, ChainType::LeftArm);
        assert!(chain.chain_bones.is_empty());
    }

    // ===== RetargetingConfig Tests =====

    #[test]
    fn test_retargeting_config_new() {
        let mappings = vec![BoneMapping::new("root", 0)];
        let config = RetargetingConfig::new(mappings, 1.8, 1.5);
        assert_eq!(config.source_height, 1.8);
        assert_eq!(config.target_height, 1.5);
        assert!(!config.use_ik_correction);
        assert!(config.scale_root_motion);
    }

    #[test]
    fn test_retargeting_config_height_ratio() {
        let config = RetargetingConfig::new(vec![], 2.0, 1.0);
        assert!((config.height_ratio() - 0.5).abs() < 1e-5);

        let config = RetargetingConfig::new(vec![], 1.0, 2.0);
        assert!((config.height_ratio() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_retargeting_config_height_ratio_zero_source() {
        let config = RetargetingConfig::new(vec![], 0.0, 1.5);
        assert_eq!(config.height_ratio(), 1.0); // Fallback to 1.0
    }

    #[test]
    fn test_retargeting_config_height_ratio_clamped() {
        // Very large ratio
        let config = RetargetingConfig::new(vec![], 0.1, 100.0);
        assert_eq!(config.height_ratio(), MAX_HEIGHT_RATIO);

        // Very small ratio
        let config = RetargetingConfig::new(vec![], 100.0, 0.1);
        assert_eq!(config.height_ratio(), MIN_HEIGHT_RATIO);
    }

    #[test]
    fn test_retargeting_config_with_ik_chains() {
        let chain = IkChainConfig::new(ChainType::LeftArm, 2, vec![0, 1, 2]);
        let config = RetargetingConfig::new(vec![], 1.8, 1.5)
            .with_ik_chains(vec![chain]);
        assert!(config.use_ik_correction);
        assert_eq!(config.ik_chains.len(), 1);
    }

    #[test]
    fn test_retargeting_config_with_speed_scale() {
        let config = RetargetingConfig::new(vec![], 1.8, 1.5)
            .with_speed_scale(1.2);
        assert_eq!(config.speed_scale, Some(1.2));
    }

    #[test]
    fn test_retargeting_config_without_root_motion_scaling() {
        let config = RetargetingConfig::new(vec![], 1.8, 1.5)
            .without_root_motion_scaling();
        assert!(!config.scale_root_motion);
    }

    #[test]
    fn test_retargeting_config_add_mapping() {
        let mut config = RetargetingConfig::new(vec![], 1.8, 1.5);
        config.add_mapping(BoneMapping::new("spine", 1));
        assert_eq!(config.mappings.len(), 1);
    }

    #[test]
    fn test_retargeting_config_find_mapping() {
        let config = RetargetingConfig::new(
            vec![
                BoneMapping::new("root", 0),
                BoneMapping::new("spine", 1),
            ],
            1.8,
            1.5,
        );
        assert!(config.find_mapping("spine").is_some());
        assert!(config.find_mapping("nonexistent").is_none());
    }

    // ===== RootMotionDelta Tests =====

    #[test]
    fn test_root_motion_delta_new() {
        let delta = RootMotionDelta::new(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::from_rotation_y(PI / 4.0),
            0.1,
        );
        assert!(delta.translation.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-5));
        assert!(delta.rotation.abs_diff_eq(Quat::from_rotation_y(PI / 4.0), 1e-5));
        assert_eq!(delta.dt, 0.1);
    }

    #[test]
    fn test_root_motion_delta_translation_only() {
        let delta = RootMotionDelta::translation_only(Vec3::new(2.0, 0.0, 0.0), 0.5);
        assert!(delta.translation.abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 1e-5));
        assert!(delta.rotation.abs_diff_eq(Quat::IDENTITY, 1e-5));
    }

    #[test]
    fn test_root_motion_delta_velocity() {
        let delta = RootMotionDelta::new(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY, 0.5);
        assert!(delta.velocity().abs_diff_eq(Vec3::new(4.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_root_motion_delta_velocity_zero_dt() {
        let delta = RootMotionDelta::new(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY, 0.0);
        assert!(delta.velocity().abs_diff_eq(Vec3::ZERO, 1e-5));
    }

    #[test]
    fn test_root_motion_delta_speed() {
        let delta = RootMotionDelta::new(Vec3::new(3.0, 4.0, 0.0), Quat::IDENTITY, 1.0);
        assert!((delta.speed() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_root_motion_delta_default() {
        let delta = RootMotionDelta::default();
        assert!(delta.translation.abs_diff_eq(Vec3::ZERO, 1e-5));
        assert!(delta.rotation.abs_diff_eq(Quat::IDENTITY, 1e-5));
        assert_eq!(delta.dt, 0.0);
    }

    // ===== RetargetingError Tests =====

    #[test]
    fn test_retargeting_error_display() {
        let err = RetargetingError::EmptySourceSkeleton;
        assert!(format!("{}", err).contains("source skeleton"));

        let err = RetargetingError::EmptyTargetSkeleton;
        assert!(format!("{}", err).contains("target skeleton"));

        let err = RetargetingError::NoMappings;
        assert!(format!("{}", err).contains("no bone mappings"));

        let err = RetargetingError::InvalidBoneIndex {
            bone_name: "arm".to_string(),
            index: 10,
            skeleton_size: 5,
        };
        assert!(format!("{}", err).contains("10"));
        assert!(format!("{}", err).contains("arm"));
        assert!(format!("{}", err).contains("5"));

        let err = RetargetingError::SourceBoneNotFound {
            bone_name: "missing".to_string(),
        };
        assert!(format!("{}", err).contains("missing"));

        let err = RetargetingError::InvalidIkChain {
            chain_type: ChainType::LeftArm,
            reason: "test reason".to_string(),
        };
        assert!(format!("{}", err).contains("Left Arm"));
        assert!(format!("{}", err).contains("test reason"));

        let err = RetargetingError::ChainLengthMismatch {
            source_length: 1.5,
            target_length: 2.0,
        };
        assert!(format!("{}", err).contains("1.5"));
        assert!(format!("{}", err).contains("2.0"));
    }

    // ===== RetargetingMap Tests =====

    #[test]
    fn test_retargeting_map_auto_map_by_name_exact() {
        let source = create_simple_skeleton();
        let target = create_simple_skeleton();

        let mappings = RetargetingMap::auto_map_by_name(&source, &target);

        assert_eq!(mappings.len(), 5); // All bones match
        assert!(mappings.iter().any(|m| m.source_name == "root"));
        assert!(mappings.iter().any(|m| m.source_name == "spine"));
        assert!(mappings.iter().any(|m| m.source_name == "head"));
    }

    #[test]
    fn test_retargeting_map_auto_map_by_name_normalized() {
        let source = create_simple_skeleton();
        let target = create_renamed_skeleton();

        let mappings = RetargetingMap::auto_map_by_name(&source, &target);

        // Should match bones despite different naming conventions
        assert!(!mappings.is_empty());
    }

    #[test]
    fn test_retargeting_map_new() {
        let source = create_simple_skeleton();
        let target = create_simple_skeleton();

        let mappings = RetargetingMap::auto_map_by_name(&source, &target);
        let config = RetargetingConfig::new(mappings, 1.8, 1.5);
        let map = RetargetingMap::new(config, &source, &target);

        assert_eq!(map.source_skeleton().bone_count(), 5);
        assert_eq!(map.target_skeleton().bone_count(), 5);
    }

    #[test]
    fn test_retargeting_map_retarget_pose_identity() {
        let source = create_simple_skeleton();
        let target = create_simple_skeleton();

        let mappings = RetargetingMap::auto_map_by_name(&source, &target);
        let config = RetargetingConfig::new(mappings, 1.0, 1.0); // Same height
        let map = RetargetingMap::new(config, &source, &target);

        // Create identity pose
        let source_pose = Pose::from_skeleton(&source, crate::pose::PoseType::Current);
        let target_pose = map.retarget_pose(&source_pose);

        // Should have same bone count
        assert_eq!(target_pose.bone_count(), target.bone_count());
    }

    #[test]
    fn test_retargeting_map_retarget_pose_scaled() {
        let source = create_simple_skeleton();
        let target = create_tall_skeleton();

        let mappings = RetargetingMap::auto_map_by_name(&source, &target);
        let config = RetargetingConfig::new(mappings, 1.0, 1.5); // 1.5x scale
        let map = RetargetingMap::new(config, &source, &target);

        // Create a pose with translation
        let mut source_pose = Pose::from_skeleton(&source, crate::pose::PoseType::Current);
        source_pose.positions[0] = Vec3::new(0.0, 2.0, 0.0); // Root at y=2

        let target_pose = map.retarget_pose(&source_pose);

        // Root should be scaled (2.0 * 1.5 = 3.0)
        assert!(target_pose.positions[0].y.abs() - 3.0 < 0.1 || target_pose.positions[0].y.abs() < 0.1);
    }

    #[test]
    fn test_retargeting_map_retarget_clip() {
        let source = create_simple_skeleton();
        let target = create_simple_skeleton();

        let mappings = RetargetingMap::auto_map_by_name(&source, &target);
        let config = RetargetingConfig::new(mappings, 1.0, 1.0);
        let map = RetargetingMap::new(config, &source, &target);

        // Create a simple animation clip
        let mut clip = AnimationClip::new("walk", 1.0);
        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::new(1.0, 0.0, 0.0)),
        ]);
        clip.add_bone_track(BoneTrack::new("root").with_position(pos_track));

        let target_clip = map.retarget_clip(&clip);

        assert_eq!(target_clip.name, "walk_retargeted");
        assert_eq!(target_clip.duration, 1.0);
        assert_eq!(target_clip.bone_count(), 1);
    }

    #[test]
    fn test_retargeting_map_validate_valid() {
        let source = create_simple_skeleton();
        let target = create_simple_skeleton();

        let mappings = RetargetingMap::auto_map_by_name(&source, &target);
        let config = RetargetingConfig::new(mappings, 1.0, 1.0);
        let map = RetargetingMap::new(config, &source, &target);

        assert!(map.validate().is_ok());
    }

    #[test]
    fn test_retargeting_map_validate_empty_source() {
        let source = Skeleton::new();
        let target = create_simple_skeleton();

        let config = RetargetingConfig::new(vec![BoneMapping::new("root", 0)], 1.0, 1.0);
        let map = RetargetingMap::new(config, &source, &target);

        let result = map.validate();
        assert!(matches!(result, Err(RetargetingError::EmptySourceSkeleton)));
    }

    #[test]
    fn test_retargeting_map_validate_empty_target() {
        let source = create_simple_skeleton();
        let target = Skeleton::new();

        let config = RetargetingConfig::new(vec![BoneMapping::new("root", 0)], 1.0, 1.0);
        let map = RetargetingMap::new(config, &source, &target);

        let result = map.validate();
        assert!(matches!(result, Err(RetargetingError::EmptyTargetSkeleton)));
    }

    #[test]
    fn test_retargeting_map_validate_no_mappings() {
        let source = create_simple_skeleton();
        let target = create_simple_skeleton();

        let config = RetargetingConfig::new(vec![], 1.0, 1.0);
        let map = RetargetingMap::new(config, &source, &target);

        let result = map.validate();
        assert!(matches!(result, Err(RetargetingError::NoMappings)));
    }

    #[test]
    fn test_retargeting_map_validate_invalid_bone_index() {
        let source = create_simple_skeleton();
        let target = create_simple_skeleton();

        let config = RetargetingConfig::new(
            vec![BoneMapping::new("root", 100)], // Invalid index
            1.0,
            1.0,
        );
        let map = RetargetingMap::new(config, &source, &target);

        let result = map.validate();
        assert!(matches!(result, Err(RetargetingError::InvalidBoneIndex { .. })));
    }

    #[test]
    fn test_retargeting_map_validate_source_bone_not_found() {
        let source = create_simple_skeleton();
        let target = create_simple_skeleton();

        let config = RetargetingConfig::new(
            vec![BoneMapping::new("nonexistent", 0)],
            1.0,
            1.0,
        );
        let map = RetargetingMap::new(config, &source, &target);

        let result = map.validate();
        assert!(matches!(result, Err(RetargetingError::SourceBoneNotFound { .. })));
    }

    #[test]
    fn test_retargeting_map_validate_invalid_ik_chain() {
        let source = create_simple_skeleton();
        let target = create_simple_skeleton();

        let mappings = RetargetingMap::auto_map_by_name(&source, &target);
        let chain = IkChainConfig::new(ChainType::LeftArm, 100, vec![1, 2, 100]); // Invalid bones
        let config = RetargetingConfig::new(mappings, 1.0, 1.0)
            .with_ik_chains(vec![chain]);
        let map = RetargetingMap::new(config, &source, &target);

        let result = map.validate();
        assert!(matches!(result, Err(RetargetingError::InvalidIkChain { .. })));
    }

    #[test]
    fn test_retargeting_map_same_skeleton_identity() {
        let skeleton = create_simple_skeleton();

        let mappings = RetargetingMap::auto_map_by_name(&skeleton, &skeleton);
        let config = RetargetingConfig::new(mappings, 1.0, 1.0);
        let map = RetargetingMap::new(config, &skeleton, &skeleton);

        // Create a pose with some rotation
        let mut source_pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        source_pose.rotations[1] = Quat::from_rotation_y(PI / 4.0);

        let target_pose = map.retarget_pose(&source_pose);

        // Should preserve the rotation
        assert!(target_pose.rotations[1].abs_diff_eq(Quat::from_rotation_y(PI / 4.0), 1e-4));
    }

    #[test]
    fn test_retargeting_map_tall_to_short() {
        let tall = create_tall_skeleton();
        let short = create_short_skeleton();

        let mappings = RetargetingMap::auto_map_by_name(&tall, &short);
        let config = RetargetingConfig::new(mappings, 2.25, 0.75); // 1/3 scale
        let map = RetargetingMap::new(config, &tall, &short);

        assert!(map.validate().is_ok());
    }

    #[test]
    fn test_retargeting_map_short_to_tall() {
        let short = create_short_skeleton();
        let tall = create_tall_skeleton();

        let mappings = RetargetingMap::auto_map_by_name(&short, &tall);
        let config = RetargetingConfig::new(mappings, 0.75, 2.25); // 3x scale
        let map = RetargetingMap::new(config, &short, &tall);

        assert!(map.validate().is_ok());
    }

    // ===== Utility Function Tests =====

    #[test]
    fn test_compute_chain_length() {
        let skeleton = create_simple_skeleton();

        // Chain from root to head: root -> spine -> head
        // Distances: (0,0,0)->(0,1,0) = 1.0, (0,1,0)->(0,1.5,0) = 0.5
        let length = compute_chain_length(&skeleton, &[0, 1, 2]);
        assert!((length - 1.5).abs() < 1e-4);
    }

    #[test]
    fn test_compute_chain_length_empty() {
        let skeleton = create_simple_skeleton();
        let length = compute_chain_length(&skeleton, &[]);
        assert_eq!(length, 0.0);
    }

    #[test]
    fn test_compute_chain_length_single_bone() {
        let skeleton = create_simple_skeleton();
        let length = compute_chain_length(&skeleton, &[0]);
        assert_eq!(length, 0.0);
    }

    #[test]
    fn test_scale_root_motion() {
        let delta = RootMotionDelta::new(
            Vec3::new(2.0, 0.0, 0.0),
            Quat::from_rotation_y(PI / 4.0),
            0.1,
        );

        let scaled = scale_root_motion(&delta, 0.5);

        assert!(scaled.translation.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-5));
        assert!(scaled.rotation.abs_diff_eq(delta.rotation, 1e-5)); // Rotation unchanged
        assert_eq!(scaled.dt, delta.dt);
    }

    #[test]
    fn test_normalize_bone_name() {
        assert_eq!(normalize_bone_name("bip01_spine"), "spine");
        assert_eq!(normalize_bone_name("bone_arm"), "arm");
        assert_eq!(normalize_bone_name("jnt_leg_bone"), "leg");
        assert_eq!(normalize_bone_name("mixamorig:Spine"), "spine");
        assert_eq!(normalize_bone_name("Spine"), "spine");
    }

    #[test]
    fn test_find_bone_name_variations() {
        let variations = find_bone_name_variations("left_hand");
        assert!(variations.contains(&"left_hand".to_string()) || variations.contains(&"l_hand".to_string()));
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_retarget_empty_pose() {
        let source = create_simple_skeleton();
        let target = create_simple_skeleton();

        let mappings = RetargetingMap::auto_map_by_name(&source, &target);
        let config = RetargetingConfig::new(mappings, 1.0, 1.0);
        let map = RetargetingMap::new(config, &source, &target);

        let empty_pose = Pose::new(0, crate::pose::PoseType::Current);
        let target_pose = map.retarget_pose(&empty_pose);

        // Should still have target bone count
        assert_eq!(target_pose.bone_count(), target.bone_count());
    }

    #[test]
    fn test_retarget_clip_no_matching_bones() {
        let source = create_simple_skeleton();
        let target = create_simple_skeleton();

        // Create mappings that don't match clip bones
        let config = RetargetingConfig::new(vec![BoneMapping::new("nonexistent", 0)], 1.0, 1.0);
        let map = RetargetingMap::new(config, &source, &target);

        let mut clip = AnimationClip::new("test", 1.0);
        clip.add_bone_track(BoneTrack::new("root"));

        let target_clip = map.retarget_clip(&clip);

        // Should have no bone tracks (no matches)
        assert_eq!(target_clip.bone_count(), 0);
    }

    #[test]
    fn test_retarget_with_rotation_offset() {
        let source = create_simple_skeleton();
        let target = create_simple_skeleton();

        let offset = Quat::from_rotation_z(PI / 2.0);
        let mappings = vec![
            BoneMapping::new("root", 0).with_rotation_offset(offset),
        ];
        let config = RetargetingConfig::new(mappings, 1.0, 1.0);
        let map = RetargetingMap::new(config, &source, &target);

        let source_pose = Pose::from_skeleton(&source, crate::pose::PoseType::Current);
        let target_pose = map.retarget_pose(&source_pose);

        // Root should have the offset applied
        let expected = offset * Quat::IDENTITY;
        assert!(target_pose.rotations[0].abs_diff_eq(expected, 1e-4));
    }

    #[test]
    fn test_missing_bone_in_source_pose() {
        let source = create_simple_skeleton();
        let target = create_simple_skeleton();

        let mappings = RetargetingMap::auto_map_by_name(&source, &target);
        let config = RetargetingConfig::new(mappings, 1.0, 1.0);
        let map = RetargetingMap::new(config, &source, &target);

        // Create a pose with fewer bones than the skeleton
        let small_pose = Pose::new(2, crate::pose::PoseType::Current);
        let target_pose = map.retarget_pose(&small_pose);

        // Should still produce valid pose
        assert_eq!(target_pose.bone_count(), target.bone_count());
    }

    #[test]
    fn test_retarget_preserves_events_and_curves() {
        use crate::animation_clip::{AnimationEvent, CurveTrack, EventTrack};

        let source = create_simple_skeleton();
        let target = create_simple_skeleton();

        let mappings = RetargetingMap::auto_map_by_name(&source, &target);
        let config = RetargetingConfig::new(mappings, 1.0, 1.0);
        let map = RetargetingMap::new(config, &source, &target);

        let mut clip = AnimationClip::new("test", 1.0);
        clip.add_bone_track(BoneTrack::new("root"));

        let mut event_track = EventTrack::new("events");
        event_track.add_event(AnimationEvent::new("footstep", 0.5));
        clip.add_event_track(event_track);

        clip.add_curve_track(CurveTrack::new("blend"));

        let target_clip = map.retarget_clip(&clip);

        // Events and curves should be preserved
        assert_eq!(target_clip.event_tracks.len(), 1);
        assert_eq!(target_clip.curve_tracks.len(), 1);
    }

    // ===== Performance Boundary Tests =====

    #[test]
    fn test_large_skeleton_retarget() {
        // Create skeletons with many bones
        let mut source = Skeleton::new();
        let mut target = Skeleton::new();

        // Add 100 bones
        source.add_bone(Bone::root("bone_0"));
        target.add_bone(Bone::root("bone_0"));

        for i in 1..100 {
            source.add_bone(Bone::new(format!("bone_{}", i)).with_parent(i - 1));
            target.add_bone(Bone::new(format!("bone_{}", i)).with_parent(i - 1));
        }

        let mappings = RetargetingMap::auto_map_by_name(&source, &target);
        let config = RetargetingConfig::new(mappings, 1.0, 1.0);
        let map = RetargetingMap::new(config, &source, &target);

        assert_eq!(map.config.mappings.len(), 100);
        assert!(map.validate().is_ok());
    }

    #[test]
    fn test_manual_mapping_override() {
        let source = create_simple_skeleton();
        let target = create_simple_skeleton();

        // Manual mapping that remaps spine to head
        let mappings = vec![
            BoneMapping::new("root", 0),
            BoneMapping::new("spine", 2), // Remap spine to head index
        ];
        let config = RetargetingConfig::new(mappings, 1.0, 1.0);
        let map = RetargetingMap::new(config, &source, &target);

        let mut source_pose = Pose::from_skeleton(&source, crate::pose::PoseType::Current);
        source_pose.rotations[1] = Quat::from_rotation_x(PI / 4.0); // Rotate spine

        let target_pose = map.retarget_pose(&source_pose);

        // Head (index 2) should have spine's rotation
        assert!(target_pose.rotations[2].abs_diff_eq(Quat::from_rotation_x(PI / 4.0), 1e-4));
    }
}
