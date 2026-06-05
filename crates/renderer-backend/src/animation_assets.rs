//! Animation asset registration and metadata for TRINITY Engine (T-AN-1.5).
//!
//! This module provides the asset system integration for animation assets:
//!
//! - `SkeletonAssetMeta` - Metadata for skeleton assets (.skel)
//! - `AnimationClipAssetMeta` - Metadata for animation clips (.anim, .fbx, .glb)
//! - `MotionDatabaseAssetMeta` - Metadata for motion matching databases (.mmdb)
//! - `AnimationAssetRegistry` - Central registry for all animation assets
//!
//! # Architecture
//!
//! ```text
//! AnimationAssetRegistry
//! ├── skeletons: HashMap<String, SkeletonAssetMeta>
//! ├── clips: HashMap<String, AnimationClipAssetMeta>
//! └── motion_databases: HashMap<String, MotionDatabaseAssetMeta>
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::animation_assets::{
//!     AnimationAssetRegistry, AnimationAssetType,
//!     SkeletonAssetMeta, AnimationClipAssetMeta,
//! };
//! use std::path::PathBuf;
//!
//! let mut registry = AnimationAssetRegistry::new();
//!
//! // Register a skeleton
//! let skel_meta = SkeletonAssetMeta {
//!     path: PathBuf::from("assets/characters/hero.skel"),
//!     bone_count: 65,
//!     hash: 0xDEADBEEF,
//!     last_modified: 1716681600,
//! };
//! registry.register_skeleton("hero", skel_meta);
//!
//! // Register an animation clip
//! let clip_meta = AnimationClipAssetMeta {
//!     path: PathBuf::from("assets/animations/hero_walk.anim"),
//!     duration: 1.0,
//!     bone_count: 65,
//!     event_count: 4,
//!     hash: 0xCAFEBABE,
//!     last_modified: 1716681600,
//! };
//! registry.register_clip("hero_walk", clip_meta);
//!
//! // Query assets by extension
//! let anim_files = registry.list_by_extension("anim");
//! ```

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// File extension for skeleton assets.
pub const SKELETON_EXTENSION: &str = "skel";

/// File extension for native animation clips.
pub const ANIM_EXTENSION: &str = "anim";

/// File extension for FBX animation files.
pub const FBX_EXTENSION: &str = "fbx";

/// File extension for glTF binary files (can contain animations).
pub const GLB_EXTENSION: &str = "glb";

/// File extension for glTF JSON files (can contain animations).
pub const GLTF_EXTENSION: &str = "gltf";

/// File extension for motion matching databases.
pub const MMDB_EXTENSION: &str = "mmdb";

/// All supported skeleton extensions.
pub const SKELETON_EXTENSIONS: &[&str] = &[SKELETON_EXTENSION];

/// All supported animation clip extensions.
pub const ANIMATION_EXTENSIONS: &[&str] = &[ANIM_EXTENSION, FBX_EXTENSION, GLB_EXTENSION, GLTF_EXTENSION];

/// All supported motion database extensions.
pub const MOTION_DB_EXTENSIONS: &[&str] = &[MMDB_EXTENSION];

// ---------------------------------------------------------------------------
// AnimationAssetType
// ---------------------------------------------------------------------------

/// Type of animation asset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnimationAssetType {
    /// Skeleton hierarchy asset (.skel).
    Skeleton,

    /// Animation clip asset (.anim, .fbx, .glb, .gltf).
    AnimationClip,

    /// Motion matching database (.mmdb).
    MotionDatabase,
}

impl AnimationAssetType {
    /// Get the primary file extension for this asset type.
    #[inline]
    pub fn primary_extension(&self) -> &'static str {
        match self {
            Self::Skeleton => SKELETON_EXTENSION,
            Self::AnimationClip => ANIM_EXTENSION,
            Self::MotionDatabase => MMDB_EXTENSION,
        }
    }

    /// Get all supported extensions for this asset type.
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Self::Skeleton => SKELETON_EXTENSIONS,
            Self::AnimationClip => ANIMATION_EXTENSIONS,
            Self::MotionDatabase => MOTION_DB_EXTENSIONS,
        }
    }

    /// Check if a file extension matches this asset type.
    pub fn matches_extension(&self, ext: &str) -> bool {
        let ext_lower = ext.to_lowercase();
        self.extensions().iter().any(|e| *e == ext_lower)
    }

    /// Determine asset type from a file extension.
    ///
    /// Returns `None` if the extension is not recognized.
    pub fn from_extension(ext: &str) -> Option<Self> {
        let ext_lower = ext.to_lowercase();

        if SKELETON_EXTENSIONS.contains(&ext_lower.as_str()) {
            Some(Self::Skeleton)
        } else if ANIMATION_EXTENSIONS.contains(&ext_lower.as_str()) {
            Some(Self::AnimationClip)
        } else if MOTION_DB_EXTENSIONS.contains(&ext_lower.as_str()) {
            Some(Self::MotionDatabase)
        } else {
            None
        }
    }

    /// Determine asset type from a file path.
    pub fn from_path(path: &std::path::Path) -> Option<Self> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(Self::from_extension)
    }
}

impl fmt::Display for AnimationAssetType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Skeleton => write!(f, "skeleton"),
            Self::AnimationClip => write!(f, "animation_clip"),
            Self::MotionDatabase => write!(f, "motion_database"),
        }
    }
}

impl Default for AnimationAssetType {
    fn default() -> Self {
        Self::AnimationClip
    }
}

// ---------------------------------------------------------------------------
// SkeletonAssetMeta
// ---------------------------------------------------------------------------

/// Metadata for a skeleton asset.
///
/// Stores information needed for asset management without loading
/// the full skeleton data into memory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkeletonAssetMeta {
    /// Path to the skeleton file.
    pub path: PathBuf,

    /// Number of bones in the skeleton.
    pub bone_count: usize,

    /// Content hash for change detection.
    pub hash: u64,

    /// Last modification timestamp (Unix epoch seconds).
    pub last_modified: u64,
}

impl SkeletonAssetMeta {
    /// Create new skeleton metadata.
    pub fn new(path: impl Into<PathBuf>, bone_count: usize) -> Self {
        Self {
            path: path.into(),
            bone_count,
            hash: 0,
            last_modified: 0,
        }
    }

    /// Set the content hash.
    #[inline]
    pub fn with_hash(mut self, hash: u64) -> Self {
        self.hash = hash;
        self
    }

    /// Set the last modified timestamp.
    #[inline]
    pub fn with_last_modified(mut self, timestamp: u64) -> Self {
        self.last_modified = timestamp;
        self
    }

    /// Get the file extension.
    pub fn extension(&self) -> Option<&str> {
        self.path.extension().and_then(|ext| ext.to_str())
    }

    /// Check if the metadata matches the expected extension.
    pub fn has_valid_extension(&self) -> bool {
        self.extension()
            .map(|ext| SKELETON_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
            .unwrap_or(false)
    }

    /// Get the asset name (filename without extension).
    pub fn asset_name(&self) -> Option<&str> {
        self.path.file_stem().and_then(|s| s.to_str())
    }

    /// Validate the metadata.
    pub fn validate(&self) -> Result<(), AnimationAssetError> {
        if self.bone_count == 0 {
            return Err(AnimationAssetError::InvalidBoneCount {
                asset_name: self.asset_name().unwrap_or("unknown").to_string(),
                count: 0,
            });
        }

        if self.bone_count > crate::skeleton::MAX_BONES {
            return Err(AnimationAssetError::TooManyBones {
                asset_name: self.asset_name().unwrap_or("unknown").to_string(),
                count: self.bone_count,
                max: crate::skeleton::MAX_BONES,
            });
        }

        Ok(())
    }
}

impl Default for SkeletonAssetMeta {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            bone_count: 0,
            hash: 0,
            last_modified: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// AnimationClipAssetMeta
// ---------------------------------------------------------------------------

/// Metadata for an animation clip asset.
///
/// Stores information needed for asset management and quick filtering
/// without loading the full animation data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationClipAssetMeta {
    /// Path to the animation file.
    pub path: PathBuf,

    /// Duration of the animation in seconds.
    pub duration: f32,

    /// Number of animated bones.
    pub bone_count: usize,

    /// Number of animation events.
    pub event_count: usize,

    /// Content hash for change detection.
    pub hash: u64,

    /// Last modification timestamp (Unix epoch seconds).
    pub last_modified: u64,
}

impl AnimationClipAssetMeta {
    /// Create new animation clip metadata.
    pub fn new(path: impl Into<PathBuf>, duration: f32, bone_count: usize) -> Self {
        Self {
            path: path.into(),
            duration,
            bone_count,
            event_count: 0,
            hash: 0,
            last_modified: 0,
        }
    }

    /// Set the event count.
    #[inline]
    pub fn with_event_count(mut self, count: usize) -> Self {
        self.event_count = count;
        self
    }

    /// Set the content hash.
    #[inline]
    pub fn with_hash(mut self, hash: u64) -> Self {
        self.hash = hash;
        self
    }

    /// Set the last modified timestamp.
    #[inline]
    pub fn with_last_modified(mut self, timestamp: u64) -> Self {
        self.last_modified = timestamp;
        self
    }

    /// Get the file extension.
    pub fn extension(&self) -> Option<&str> {
        self.path.extension().and_then(|ext| ext.to_str())
    }

    /// Check if the metadata matches a supported extension.
    pub fn has_valid_extension(&self) -> bool {
        self.extension()
            .map(|ext| ANIMATION_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
            .unwrap_or(false)
    }

    /// Get the asset name (filename without extension).
    pub fn asset_name(&self) -> Option<&str> {
        self.path.file_stem().and_then(|s| s.to_str())
    }

    /// Check if this is an FBX file.
    #[inline]
    pub fn is_fbx(&self) -> bool {
        self.extension()
            .map(|ext| ext.eq_ignore_ascii_case(FBX_EXTENSION))
            .unwrap_or(false)
    }

    /// Check if this is a glTF file (binary or JSON).
    #[inline]
    pub fn is_gltf(&self) -> bool {
        self.extension()
            .map(|ext| {
                ext.eq_ignore_ascii_case(GLB_EXTENSION) || ext.eq_ignore_ascii_case(GLTF_EXTENSION)
            })
            .unwrap_or(false)
    }

    /// Check if this is a native animation file.
    #[inline]
    pub fn is_native(&self) -> bool {
        self.extension()
            .map(|ext| ext.eq_ignore_ascii_case(ANIM_EXTENSION))
            .unwrap_or(false)
    }

    /// Validate the metadata.
    pub fn validate(&self) -> Result<(), AnimationAssetError> {
        if self.duration < 0.0 {
            return Err(AnimationAssetError::InvalidDuration {
                asset_name: self.asset_name().unwrap_or("unknown").to_string(),
                duration: self.duration,
            });
        }

        if self.bone_count > crate::animation_clip::MAX_ANIMATED_BONES {
            return Err(AnimationAssetError::TooManyAnimatedBones {
                asset_name: self.asset_name().unwrap_or("unknown").to_string(),
                count: self.bone_count,
                max: crate::animation_clip::MAX_ANIMATED_BONES,
            });
        }

        if self.event_count > crate::animation_clip::MAX_EVENTS_PER_CLIP {
            return Err(AnimationAssetError::TooManyEvents {
                asset_name: self.asset_name().unwrap_or("unknown").to_string(),
                count: self.event_count,
                max: crate::animation_clip::MAX_EVENTS_PER_CLIP,
            });
        }

        Ok(())
    }
}

impl Default for AnimationClipAssetMeta {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            duration: 0.0,
            bone_count: 0,
            event_count: 0,
            hash: 0,
            last_modified: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// MotionDatabaseAssetMeta
// ---------------------------------------------------------------------------

/// Metadata for a motion matching database asset.
///
/// Motion databases contain pre-processed animation data optimized
/// for real-time motion matching queries.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MotionDatabaseAssetMeta {
    /// Path to the motion database file.
    pub path: PathBuf,

    /// Number of motion clips in the database.
    pub clip_count: usize,

    /// Total number of poses across all clips.
    pub pose_count: usize,

    /// Number of feature dimensions per pose.
    pub feature_dimensions: usize,

    /// Content hash for change detection.
    pub hash: u64,

    /// Last modification timestamp (Unix epoch seconds).
    pub last_modified: u64,
}

impl MotionDatabaseAssetMeta {
    /// Create new motion database metadata.
    pub fn new(path: impl Into<PathBuf>, clip_count: usize, pose_count: usize) -> Self {
        Self {
            path: path.into(),
            clip_count,
            pose_count,
            feature_dimensions: 0,
            hash: 0,
            last_modified: 0,
        }
    }

    /// Set the feature dimensions.
    #[inline]
    pub fn with_feature_dimensions(mut self, dims: usize) -> Self {
        self.feature_dimensions = dims;
        self
    }

    /// Set the content hash.
    #[inline]
    pub fn with_hash(mut self, hash: u64) -> Self {
        self.hash = hash;
        self
    }

    /// Set the last modified timestamp.
    #[inline]
    pub fn with_last_modified(mut self, timestamp: u64) -> Self {
        self.last_modified = timestamp;
        self
    }

    /// Get the file extension.
    pub fn extension(&self) -> Option<&str> {
        self.path.extension().and_then(|ext| ext.to_str())
    }

    /// Check if the metadata matches the expected extension.
    pub fn has_valid_extension(&self) -> bool {
        self.extension()
            .map(|ext| MOTION_DB_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
            .unwrap_or(false)
    }

    /// Get the asset name (filename without extension).
    pub fn asset_name(&self) -> Option<&str> {
        self.path.file_stem().and_then(|s| s.to_str())
    }

    /// Validate the metadata.
    pub fn validate(&self) -> Result<(), AnimationAssetError> {
        if self.clip_count == 0 {
            return Err(AnimationAssetError::EmptyMotionDatabase {
                asset_name: self.asset_name().unwrap_or("unknown").to_string(),
            });
        }

        if self.pose_count == 0 {
            return Err(AnimationAssetError::NoPosesInDatabase {
                asset_name: self.asset_name().unwrap_or("unknown").to_string(),
            });
        }

        Ok(())
    }
}

impl Default for MotionDatabaseAssetMeta {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            clip_count: 0,
            pose_count: 0,
            feature_dimensions: 0,
            hash: 0,
            last_modified: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// AnimationAssetError
// ---------------------------------------------------------------------------

/// Errors that can occur during animation asset operations.
#[derive(Clone, Debug, PartialEq)]
pub enum AnimationAssetError {
    /// Asset not found in registry.
    AssetNotFound {
        asset_type: AnimationAssetType,
        name: String,
    },

    /// Asset already registered.
    AssetAlreadyRegistered {
        asset_type: AnimationAssetType,
        name: String,
    },

    /// Invalid bone count (zero or too many).
    InvalidBoneCount {
        asset_name: String,
        count: usize,
    },

    /// Too many bones in skeleton.
    TooManyBones {
        asset_name: String,
        count: usize,
        max: usize,
    },

    /// Invalid animation duration.
    InvalidDuration {
        asset_name: String,
        duration: f32,
    },

    /// Too many animated bones.
    TooManyAnimatedBones {
        asset_name: String,
        count: usize,
        max: usize,
    },

    /// Too many animation events.
    TooManyEvents {
        asset_name: String,
        count: usize,
        max: usize,
    },

    /// Empty motion database (no clips).
    EmptyMotionDatabase {
        asset_name: String,
    },

    /// Motion database has no poses.
    NoPosesInDatabase {
        asset_name: String,
    },

    /// Unsupported file extension.
    UnsupportedExtension {
        path: PathBuf,
        extension: String,
    },

    /// Hash mismatch during validation.
    HashMismatch {
        asset_name: String,
        expected: u64,
        actual: u64,
    },
}

impl fmt::Display for AnimationAssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AssetNotFound { asset_type, name } => {
                write!(f, "{} asset '{}' not found in registry", asset_type, name)
            }
            Self::AssetAlreadyRegistered { asset_type, name } => {
                write!(f, "{} asset '{}' is already registered", asset_type, name)
            }
            Self::InvalidBoneCount { asset_name, count } => {
                write!(f, "invalid bone count {} in '{}'", count, asset_name)
            }
            Self::TooManyBones { asset_name, count, max } => {
                write!(
                    f,
                    "too many bones in '{}': {} (max {})",
                    asset_name, count, max
                )
            }
            Self::InvalidDuration { asset_name, duration } => {
                write!(f, "invalid duration {} in '{}'", duration, asset_name)
            }
            Self::TooManyAnimatedBones { asset_name, count, max } => {
                write!(
                    f,
                    "too many animated bones in '{}': {} (max {})",
                    asset_name, count, max
                )
            }
            Self::TooManyEvents { asset_name, count, max } => {
                write!(
                    f,
                    "too many events in '{}': {} (max {})",
                    asset_name, count, max
                )
            }
            Self::EmptyMotionDatabase { asset_name } => {
                write!(f, "motion database '{}' has no clips", asset_name)
            }
            Self::NoPosesInDatabase { asset_name } => {
                write!(f, "motion database '{}' has no poses", asset_name)
            }
            Self::UnsupportedExtension { path, extension } => {
                write!(
                    f,
                    "unsupported file extension '{}' for path '{}'",
                    extension,
                    path.display()
                )
            }
            Self::HashMismatch { asset_name, expected, actual } => {
                write!(
                    f,
                    "hash mismatch for '{}': expected {:016x}, got {:016x}",
                    asset_name, expected, actual
                )
            }
        }
    }
}

impl std::error::Error for AnimationAssetError {}

// ---------------------------------------------------------------------------
// AnimationAssetRegistry
// ---------------------------------------------------------------------------

/// Central registry for animation assets.
///
/// Provides registration, lookup, and querying of animation asset metadata.
/// This is the main entry point for managing animation assets in the engine.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AnimationAssetRegistry {
    /// Registered skeleton assets.
    skeletons: HashMap<String, SkeletonAssetMeta>,

    /// Registered animation clip assets.
    clips: HashMap<String, AnimationClipAssetMeta>,

    /// Registered motion database assets.
    motion_databases: HashMap<String, MotionDatabaseAssetMeta>,
}

impl AnimationAssetRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            skeletons: HashMap::new(),
            clips: HashMap::new(),
            motion_databases: HashMap::new(),
        }
    }

    /// Create a registry with pre-allocated capacity.
    pub fn with_capacity(skeletons: usize, clips: usize, motion_dbs: usize) -> Self {
        Self {
            skeletons: HashMap::with_capacity(skeletons),
            clips: HashMap::with_capacity(clips),
            motion_databases: HashMap::with_capacity(motion_dbs),
        }
    }

    // ===== Skeleton Registration =====

    /// Register a skeleton asset.
    pub fn register_skeleton(&mut self, name: &str, meta: SkeletonAssetMeta) {
        self.skeletons.insert(name.to_string(), meta);
    }

    /// Register a skeleton with validation.
    pub fn register_skeleton_validated(
        &mut self,
        name: &str,
        meta: SkeletonAssetMeta,
    ) -> Result<(), AnimationAssetError> {
        meta.validate()?;

        if self.skeletons.contains_key(name) {
            return Err(AnimationAssetError::AssetAlreadyRegistered {
                asset_type: AnimationAssetType::Skeleton,
                name: name.to_string(),
            });
        }

        self.skeletons.insert(name.to_string(), meta);
        Ok(())
    }

    /// Get a skeleton asset by name.
    #[inline]
    pub fn get_skeleton(&self, name: &str) -> Option<&SkeletonAssetMeta> {
        self.skeletons.get(name)
    }

    /// Get a mutable skeleton asset by name.
    #[inline]
    pub fn get_skeleton_mut(&mut self, name: &str) -> Option<&mut SkeletonAssetMeta> {
        self.skeletons.get_mut(name)
    }

    /// Remove a skeleton asset.
    #[inline]
    pub fn unregister_skeleton(&mut self, name: &str) -> Option<SkeletonAssetMeta> {
        self.skeletons.remove(name)
    }

    /// Check if a skeleton is registered.
    #[inline]
    pub fn has_skeleton(&self, name: &str) -> bool {
        self.skeletons.contains_key(name)
    }

    /// Get the number of registered skeletons.
    #[inline]
    pub fn skeleton_count(&self) -> usize {
        self.skeletons.len()
    }

    /// Iterate over all registered skeletons.
    #[inline]
    pub fn skeletons(&self) -> impl Iterator<Item = (&String, &SkeletonAssetMeta)> {
        self.skeletons.iter()
    }

    /// Get all skeleton names.
    pub fn skeleton_names(&self) -> Vec<&str> {
        self.skeletons.keys().map(|s| s.as_str()).collect()
    }

    // ===== Animation Clip Registration =====

    /// Register an animation clip asset.
    pub fn register_clip(&mut self, name: &str, meta: AnimationClipAssetMeta) {
        self.clips.insert(name.to_string(), meta);
    }

    /// Register an animation clip with validation.
    pub fn register_clip_validated(
        &mut self,
        name: &str,
        meta: AnimationClipAssetMeta,
    ) -> Result<(), AnimationAssetError> {
        meta.validate()?;

        if self.clips.contains_key(name) {
            return Err(AnimationAssetError::AssetAlreadyRegistered {
                asset_type: AnimationAssetType::AnimationClip,
                name: name.to_string(),
            });
        }

        self.clips.insert(name.to_string(), meta);
        Ok(())
    }

    /// Get an animation clip asset by name.
    #[inline]
    pub fn get_clip(&self, name: &str) -> Option<&AnimationClipAssetMeta> {
        self.clips.get(name)
    }

    /// Get a mutable animation clip asset by name.
    #[inline]
    pub fn get_clip_mut(&mut self, name: &str) -> Option<&mut AnimationClipAssetMeta> {
        self.clips.get_mut(name)
    }

    /// Remove an animation clip asset.
    #[inline]
    pub fn unregister_clip(&mut self, name: &str) -> Option<AnimationClipAssetMeta> {
        self.clips.remove(name)
    }

    /// Check if a clip is registered.
    #[inline]
    pub fn has_clip(&self, name: &str) -> bool {
        self.clips.contains_key(name)
    }

    /// Get the number of registered clips.
    #[inline]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Iterate over all registered clips.
    #[inline]
    pub fn clips(&self) -> impl Iterator<Item = (&String, &AnimationClipAssetMeta)> {
        self.clips.iter()
    }

    /// Get all clip names.
    pub fn clip_names(&self) -> Vec<&str> {
        self.clips.keys().map(|s| s.as_str()).collect()
    }

    // ===== Motion Database Registration =====

    /// Register a motion database asset.
    pub fn register_motion_database(&mut self, name: &str, meta: MotionDatabaseAssetMeta) {
        self.motion_databases.insert(name.to_string(), meta);
    }

    /// Register a motion database with validation.
    pub fn register_motion_database_validated(
        &mut self,
        name: &str,
        meta: MotionDatabaseAssetMeta,
    ) -> Result<(), AnimationAssetError> {
        meta.validate()?;

        if self.motion_databases.contains_key(name) {
            return Err(AnimationAssetError::AssetAlreadyRegistered {
                asset_type: AnimationAssetType::MotionDatabase,
                name: name.to_string(),
            });
        }

        self.motion_databases.insert(name.to_string(), meta);
        Ok(())
    }

    /// Get a motion database asset by name.
    #[inline]
    pub fn get_motion_database(&self, name: &str) -> Option<&MotionDatabaseAssetMeta> {
        self.motion_databases.get(name)
    }

    /// Get a mutable motion database asset by name.
    #[inline]
    pub fn get_motion_database_mut(&mut self, name: &str) -> Option<&mut MotionDatabaseAssetMeta> {
        self.motion_databases.get_mut(name)
    }

    /// Remove a motion database asset.
    #[inline]
    pub fn unregister_motion_database(&mut self, name: &str) -> Option<MotionDatabaseAssetMeta> {
        self.motion_databases.remove(name)
    }

    /// Check if a motion database is registered.
    #[inline]
    pub fn has_motion_database(&self, name: &str) -> bool {
        self.motion_databases.contains_key(name)
    }

    /// Get the number of registered motion databases.
    #[inline]
    pub fn motion_database_count(&self) -> usize {
        self.motion_databases.len()
    }

    /// Iterate over all registered motion databases.
    #[inline]
    pub fn motion_databases(&self) -> impl Iterator<Item = (&String, &MotionDatabaseAssetMeta)> {
        self.motion_databases.iter()
    }

    /// Get all motion database names.
    pub fn motion_database_names(&self) -> Vec<&str> {
        self.motion_databases.keys().map(|s| s.as_str()).collect()
    }

    // ===== Query Methods =====

    /// List all assets with the given file extension.
    ///
    /// Returns asset names (not paths) that have files with the specified extension.
    pub fn list_by_extension(&self, ext: &str) -> Vec<&str> {
        let ext_lower = ext.to_lowercase();
        let mut results = Vec::new();

        // Check skeletons
        for (name, meta) in &self.skeletons {
            if let Some(file_ext) = meta.extension() {
                if file_ext.eq_ignore_ascii_case(&ext_lower) {
                    results.push(name.as_str());
                }
            }
        }

        // Check clips
        for (name, meta) in &self.clips {
            if let Some(file_ext) = meta.extension() {
                if file_ext.eq_ignore_ascii_case(&ext_lower) {
                    results.push(name.as_str());
                }
            }
        }

        // Check motion databases
        for (name, meta) in &self.motion_databases {
            if let Some(file_ext) = meta.extension() {
                if file_ext.eq_ignore_ascii_case(&ext_lower) {
                    results.push(name.as_str());
                }
            }
        }

        results
    }

    /// List all assets of a specific type.
    pub fn list_by_type(&self, asset_type: AnimationAssetType) -> Vec<&str> {
        match asset_type {
            AnimationAssetType::Skeleton => self.skeleton_names(),
            AnimationAssetType::AnimationClip => self.clip_names(),
            AnimationAssetType::MotionDatabase => self.motion_database_names(),
        }
    }

    /// Find clips with duration in the given range.
    pub fn clips_by_duration(&self, min_duration: f32, max_duration: f32) -> Vec<&str> {
        self.clips
            .iter()
            .filter(|(_, meta)| meta.duration >= min_duration && meta.duration <= max_duration)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Find clips with at least the specified number of bones.
    pub fn clips_by_min_bones(&self, min_bones: usize) -> Vec<&str> {
        self.clips
            .iter()
            .filter(|(_, meta)| meta.bone_count >= min_bones)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Find clips that have events.
    pub fn clips_with_events(&self) -> Vec<&str> {
        self.clips
            .iter()
            .filter(|(_, meta)| meta.event_count > 0)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Find assets modified after the given timestamp.
    pub fn assets_modified_after(&self, timestamp: u64) -> Vec<(&str, AnimationAssetType)> {
        let mut results = Vec::new();

        for (name, meta) in &self.skeletons {
            if meta.last_modified > timestamp {
                results.push((name.as_str(), AnimationAssetType::Skeleton));
            }
        }

        for (name, meta) in &self.clips {
            if meta.last_modified > timestamp {
                results.push((name.as_str(), AnimationAssetType::AnimationClip));
            }
        }

        for (name, meta) in &self.motion_databases {
            if meta.last_modified > timestamp {
                results.push((name.as_str(), AnimationAssetType::MotionDatabase));
            }
        }

        results
    }

    /// Get total asset count across all types.
    #[inline]
    pub fn total_asset_count(&self) -> usize {
        self.skeletons.len() + self.clips.len() + self.motion_databases.len()
    }

    /// Check if the registry is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.skeletons.is_empty() && self.clips.is_empty() && self.motion_databases.is_empty()
    }

    /// Clear all registered assets.
    pub fn clear(&mut self) {
        self.skeletons.clear();
        self.clips.clear();
        self.motion_databases.clear();
    }

    // ===== Serialization =====

    /// Serialize the registry to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Merge another registry into this one.
    ///
    /// Assets from `other` will overwrite assets with the same name in `self`.
    pub fn merge(&mut self, other: AnimationAssetRegistry) {
        self.skeletons.extend(other.skeletons);
        self.clips.extend(other.clips);
        self.motion_databases.extend(other.motion_databases);
    }

    /// Get statistics about the registry.
    pub fn stats(&self) -> RegistryStats {
        let total_bones: usize = self.skeletons.values().map(|s| s.bone_count).sum();
        let total_duration: f32 = self.clips.values().map(|c| c.duration).sum();
        let total_events: usize = self.clips.values().map(|c| c.event_count).sum();
        let total_poses: usize = self.motion_databases.values().map(|m| m.pose_count).sum();

        RegistryStats {
            skeleton_count: self.skeletons.len(),
            clip_count: self.clips.len(),
            motion_database_count: self.motion_databases.len(),
            total_bones,
            total_animation_duration: total_duration,
            total_events,
            total_poses,
        }
    }
}

// ---------------------------------------------------------------------------
// RegistryStats
// ---------------------------------------------------------------------------

/// Statistics about an AnimationAssetRegistry.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RegistryStats {
    /// Number of registered skeletons.
    pub skeleton_count: usize,

    /// Number of registered animation clips.
    pub clip_count: usize,

    /// Number of registered motion databases.
    pub motion_database_count: usize,

    /// Total bones across all skeletons.
    pub total_bones: usize,

    /// Total animation duration across all clips (seconds).
    pub total_animation_duration: f32,

    /// Total events across all clips.
    pub total_events: usize,

    /// Total poses across all motion databases.
    pub total_poses: usize,
}

impl fmt::Display for RegistryStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AnimationAssetRegistry: {} skeletons ({} bones), {} clips ({:.1}s, {} events), {} motion DBs ({} poses)",
            self.skeleton_count,
            self.total_bones,
            self.clip_count,
            self.total_animation_duration,
            self.total_events,
            self.motion_database_count,
            self.total_poses
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===== AnimationAssetType Tests =====

    #[test]
    fn test_asset_type_primary_extension() {
        assert_eq!(AnimationAssetType::Skeleton.primary_extension(), "skel");
        assert_eq!(AnimationAssetType::AnimationClip.primary_extension(), "anim");
        assert_eq!(AnimationAssetType::MotionDatabase.primary_extension(), "mmdb");
    }

    #[test]
    fn test_asset_type_extensions() {
        assert_eq!(AnimationAssetType::Skeleton.extensions(), &["skel"]);
        assert_eq!(AnimationAssetType::AnimationClip.extensions(), &["anim", "fbx", "glb", "gltf"]);
        assert_eq!(AnimationAssetType::MotionDatabase.extensions(), &["mmdb"]);
    }

    #[test]
    fn test_asset_type_matches_extension() {
        assert!(AnimationAssetType::Skeleton.matches_extension("skel"));
        assert!(AnimationAssetType::Skeleton.matches_extension("SKEL"));
        assert!(!AnimationAssetType::Skeleton.matches_extension("anim"));

        assert!(AnimationAssetType::AnimationClip.matches_extension("anim"));
        assert!(AnimationAssetType::AnimationClip.matches_extension("fbx"));
        assert!(AnimationAssetType::AnimationClip.matches_extension("glb"));
        assert!(AnimationAssetType::AnimationClip.matches_extension("gltf"));
        assert!(AnimationAssetType::AnimationClip.matches_extension("FBX"));
    }

    #[test]
    fn test_asset_type_from_extension() {
        assert_eq!(
            AnimationAssetType::from_extension("skel"),
            Some(AnimationAssetType::Skeleton)
        );
        assert_eq!(
            AnimationAssetType::from_extension("anim"),
            Some(AnimationAssetType::AnimationClip)
        );
        assert_eq!(
            AnimationAssetType::from_extension("fbx"),
            Some(AnimationAssetType::AnimationClip)
        );
        assert_eq!(
            AnimationAssetType::from_extension("glb"),
            Some(AnimationAssetType::AnimationClip)
        );
        assert_eq!(
            AnimationAssetType::from_extension("mmdb"),
            Some(AnimationAssetType::MotionDatabase)
        );
        assert_eq!(AnimationAssetType::from_extension("unknown"), None);
    }

    #[test]
    fn test_asset_type_from_path() {
        assert_eq!(
            AnimationAssetType::from_path(&PathBuf::from("hero.skel")),
            Some(AnimationAssetType::Skeleton)
        );
        assert_eq!(
            AnimationAssetType::from_path(&PathBuf::from("/assets/walk.anim")),
            Some(AnimationAssetType::AnimationClip)
        );
        assert_eq!(
            AnimationAssetType::from_path(&PathBuf::from("model.fbx")),
            Some(AnimationAssetType::AnimationClip)
        );
        assert_eq!(
            AnimationAssetType::from_path(&PathBuf::from("no_extension")),
            None
        );
    }

    #[test]
    fn test_asset_type_display() {
        assert_eq!(format!("{}", AnimationAssetType::Skeleton), "skeleton");
        assert_eq!(format!("{}", AnimationAssetType::AnimationClip), "animation_clip");
        assert_eq!(format!("{}", AnimationAssetType::MotionDatabase), "motion_database");
    }

    #[test]
    fn test_asset_type_default() {
        assert_eq!(AnimationAssetType::default(), AnimationAssetType::AnimationClip);
    }

    // ===== SkeletonAssetMeta Tests =====

    #[test]
    fn test_skeleton_meta_new() {
        let meta = SkeletonAssetMeta::new("hero.skel", 65);
        assert_eq!(meta.path, PathBuf::from("hero.skel"));
        assert_eq!(meta.bone_count, 65);
        assert_eq!(meta.hash, 0);
        assert_eq!(meta.last_modified, 0);
    }

    #[test]
    fn test_skeleton_meta_builder() {
        let meta = SkeletonAssetMeta::new("hero.skel", 65)
            .with_hash(0xDEADBEEF)
            .with_last_modified(1716681600);

        assert_eq!(meta.hash, 0xDEADBEEF);
        assert_eq!(meta.last_modified, 1716681600);
    }

    #[test]
    fn test_skeleton_meta_extension() {
        let meta = SkeletonAssetMeta::new("assets/hero.skel", 65);
        assert_eq!(meta.extension(), Some("skel"));

        let no_ext = SkeletonAssetMeta::new("assets/hero", 65);
        assert_eq!(no_ext.extension(), None);
    }

    #[test]
    fn test_skeleton_meta_has_valid_extension() {
        let valid = SkeletonAssetMeta::new("hero.skel", 65);
        assert!(valid.has_valid_extension());

        let invalid = SkeletonAssetMeta::new("hero.fbx", 65);
        assert!(!invalid.has_valid_extension());
    }

    #[test]
    fn test_skeleton_meta_asset_name() {
        let meta = SkeletonAssetMeta::new("assets/characters/hero.skel", 65);
        assert_eq!(meta.asset_name(), Some("hero"));
    }

    #[test]
    fn test_skeleton_meta_validate() {
        let valid = SkeletonAssetMeta::new("hero.skel", 65);
        assert!(valid.validate().is_ok());

        let zero_bones = SkeletonAssetMeta::new("hero.skel", 0);
        assert!(zero_bones.validate().is_err());

        let too_many = SkeletonAssetMeta::new("hero.skel", 500);
        assert!(too_many.validate().is_err());
    }

    #[test]
    fn test_skeleton_meta_default() {
        let meta = SkeletonAssetMeta::default();
        assert!(meta.path.as_os_str().is_empty());
        assert_eq!(meta.bone_count, 0);
    }

    // ===== AnimationClipAssetMeta Tests =====

    #[test]
    fn test_clip_meta_new() {
        let meta = AnimationClipAssetMeta::new("walk.anim", 1.5, 65);
        assert_eq!(meta.path, PathBuf::from("walk.anim"));
        assert_eq!(meta.duration, 1.5);
        assert_eq!(meta.bone_count, 65);
        assert_eq!(meta.event_count, 0);
    }

    #[test]
    fn test_clip_meta_builder() {
        let meta = AnimationClipAssetMeta::new("walk.anim", 1.5, 65)
            .with_event_count(4)
            .with_hash(0xCAFEBABE)
            .with_last_modified(1716681600);

        assert_eq!(meta.event_count, 4);
        assert_eq!(meta.hash, 0xCAFEBABE);
        assert_eq!(meta.last_modified, 1716681600);
    }

    #[test]
    fn test_clip_meta_extension() {
        let meta = AnimationClipAssetMeta::new("walk.anim", 1.0, 65);
        assert_eq!(meta.extension(), Some("anim"));
    }

    #[test]
    fn test_clip_meta_has_valid_extension() {
        assert!(AnimationClipAssetMeta::new("walk.anim", 1.0, 65).has_valid_extension());
        assert!(AnimationClipAssetMeta::new("walk.fbx", 1.0, 65).has_valid_extension());
        assert!(AnimationClipAssetMeta::new("walk.glb", 1.0, 65).has_valid_extension());
        assert!(AnimationClipAssetMeta::new("walk.gltf", 1.0, 65).has_valid_extension());
        assert!(!AnimationClipAssetMeta::new("walk.skel", 1.0, 65).has_valid_extension());
    }

    #[test]
    fn test_clip_meta_is_fbx() {
        assert!(AnimationClipAssetMeta::new("walk.fbx", 1.0, 65).is_fbx());
        assert!(AnimationClipAssetMeta::new("walk.FBX", 1.0, 65).is_fbx());
        assert!(!AnimationClipAssetMeta::new("walk.anim", 1.0, 65).is_fbx());
    }

    #[test]
    fn test_clip_meta_is_gltf() {
        assert!(AnimationClipAssetMeta::new("walk.glb", 1.0, 65).is_gltf());
        assert!(AnimationClipAssetMeta::new("walk.gltf", 1.0, 65).is_gltf());
        assert!(AnimationClipAssetMeta::new("walk.GLB", 1.0, 65).is_gltf());
        assert!(!AnimationClipAssetMeta::new("walk.fbx", 1.0, 65).is_gltf());
    }

    #[test]
    fn test_clip_meta_is_native() {
        assert!(AnimationClipAssetMeta::new("walk.anim", 1.0, 65).is_native());
        assert!(!AnimationClipAssetMeta::new("walk.fbx", 1.0, 65).is_native());
    }

    #[test]
    fn test_clip_meta_validate() {
        let valid = AnimationClipAssetMeta::new("walk.anim", 1.5, 65);
        assert!(valid.validate().is_ok());

        let negative_duration = AnimationClipAssetMeta::new("walk.anim", -1.0, 65);
        assert!(negative_duration.validate().is_err());

        let too_many_bones = AnimationClipAssetMeta::new("walk.anim", 1.0, 500);
        assert!(too_many_bones.validate().is_err());

        let too_many_events = AnimationClipAssetMeta::new("walk.anim", 1.0, 65)
            .with_event_count(2000);
        assert!(too_many_events.validate().is_err());
    }

    #[test]
    fn test_clip_meta_default() {
        let meta = AnimationClipAssetMeta::default();
        assert!(meta.path.as_os_str().is_empty());
        assert_eq!(meta.duration, 0.0);
        assert_eq!(meta.bone_count, 0);
    }

    // ===== MotionDatabaseAssetMeta Tests =====

    #[test]
    fn test_motion_db_meta_new() {
        let meta = MotionDatabaseAssetMeta::new("locomotion.mmdb", 10, 5000);
        assert_eq!(meta.path, PathBuf::from("locomotion.mmdb"));
        assert_eq!(meta.clip_count, 10);
        assert_eq!(meta.pose_count, 5000);
        assert_eq!(meta.feature_dimensions, 0);
    }

    #[test]
    fn test_motion_db_meta_builder() {
        let meta = MotionDatabaseAssetMeta::new("locomotion.mmdb", 10, 5000)
            .with_feature_dimensions(64)
            .with_hash(0x12345678)
            .with_last_modified(1716681600);

        assert_eq!(meta.feature_dimensions, 64);
        assert_eq!(meta.hash, 0x12345678);
        assert_eq!(meta.last_modified, 1716681600);
    }

    #[test]
    fn test_motion_db_meta_has_valid_extension() {
        assert!(MotionDatabaseAssetMeta::new("db.mmdb", 1, 100).has_valid_extension());
        assert!(!MotionDatabaseAssetMeta::new("db.anim", 1, 100).has_valid_extension());
    }

    #[test]
    fn test_motion_db_meta_validate() {
        let valid = MotionDatabaseAssetMeta::new("db.mmdb", 10, 5000);
        assert!(valid.validate().is_ok());

        let no_clips = MotionDatabaseAssetMeta::new("db.mmdb", 0, 5000);
        assert!(no_clips.validate().is_err());

        let no_poses = MotionDatabaseAssetMeta::new("db.mmdb", 10, 0);
        assert!(no_poses.validate().is_err());
    }

    // ===== AnimationAssetError Tests =====

    #[test]
    fn test_error_display() {
        let err = AnimationAssetError::AssetNotFound {
            asset_type: AnimationAssetType::Skeleton,
            name: "hero".to_string(),
        };
        assert!(format!("{}", err).contains("skeleton"));
        assert!(format!("{}", err).contains("hero"));

        let err = AnimationAssetError::TooManyBones {
            asset_name: "hero".to_string(),
            count: 500,
            max: 256,
        };
        assert!(format!("{}", err).contains("500"));
        assert!(format!("{}", err).contains("256"));

        let err = AnimationAssetError::HashMismatch {
            asset_name: "walk".to_string(),
            expected: 0xDEAD,
            actual: 0xBEEF,
        };
        assert!(format!("{}", err).contains("walk"));
    }

    // ===== AnimationAssetRegistry Tests =====

    #[test]
    fn test_registry_new() {
        let registry = AnimationAssetRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.total_asset_count(), 0);
    }

    #[test]
    fn test_registry_with_capacity() {
        let registry = AnimationAssetRegistry::with_capacity(10, 100, 5);
        assert!(registry.is_empty());
    }

    // ===== Skeleton Registration Tests =====

    #[test]
    fn test_register_skeleton() {
        let mut registry = AnimationAssetRegistry::new();
        let meta = SkeletonAssetMeta::new("hero.skel", 65);

        registry.register_skeleton("hero", meta.clone());

        assert!(registry.has_skeleton("hero"));
        assert_eq!(registry.skeleton_count(), 1);
        assert_eq!(registry.get_skeleton("hero").unwrap().bone_count, 65);
    }

    #[test]
    fn test_register_skeleton_validated() {
        let mut registry = AnimationAssetRegistry::new();

        // Valid registration
        let valid = SkeletonAssetMeta::new("hero.skel", 65);
        assert!(registry.register_skeleton_validated("hero", valid).is_ok());

        // Duplicate registration
        let duplicate = SkeletonAssetMeta::new("hero2.skel", 65);
        assert!(registry.register_skeleton_validated("hero", duplicate).is_err());

        // Invalid metadata
        let invalid = SkeletonAssetMeta::new("invalid.skel", 0);
        assert!(registry.register_skeleton_validated("invalid", invalid).is_err());
    }

    #[test]
    fn test_get_skeleton_mut() {
        let mut registry = AnimationAssetRegistry::new();
        registry.register_skeleton("hero", SkeletonAssetMeta::new("hero.skel", 65));

        if let Some(meta) = registry.get_skeleton_mut("hero") {
            meta.bone_count = 70;
        }

        assert_eq!(registry.get_skeleton("hero").unwrap().bone_count, 70);
    }

    #[test]
    fn test_unregister_skeleton() {
        let mut registry = AnimationAssetRegistry::new();
        registry.register_skeleton("hero", SkeletonAssetMeta::new("hero.skel", 65));

        let removed = registry.unregister_skeleton("hero");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().bone_count, 65);
        assert!(!registry.has_skeleton("hero"));
    }

    #[test]
    fn test_skeleton_names() {
        let mut registry = AnimationAssetRegistry::new();
        registry.register_skeleton("hero", SkeletonAssetMeta::new("hero.skel", 65));
        registry.register_skeleton("enemy", SkeletonAssetMeta::new("enemy.skel", 45));

        let names = registry.skeleton_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"hero"));
        assert!(names.contains(&"enemy"));
    }

    // ===== Clip Registration Tests =====

    #[test]
    fn test_register_clip() {
        let mut registry = AnimationAssetRegistry::new();
        let meta = AnimationClipAssetMeta::new("walk.anim", 1.5, 65);

        registry.register_clip("walk", meta);

        assert!(registry.has_clip("walk"));
        assert_eq!(registry.clip_count(), 1);
        assert_eq!(registry.get_clip("walk").unwrap().duration, 1.5);
    }

    #[test]
    fn test_register_clip_validated() {
        let mut registry = AnimationAssetRegistry::new();

        let valid = AnimationClipAssetMeta::new("walk.anim", 1.5, 65);
        assert!(registry.register_clip_validated("walk", valid).is_ok());

        let duplicate = AnimationClipAssetMeta::new("walk2.anim", 1.0, 65);
        assert!(registry.register_clip_validated("walk", duplicate).is_err());

        let invalid = AnimationClipAssetMeta::new("invalid.anim", -1.0, 65);
        assert!(registry.register_clip_validated("invalid", invalid).is_err());
    }

    #[test]
    fn test_unregister_clip() {
        let mut registry = AnimationAssetRegistry::new();
        registry.register_clip("walk", AnimationClipAssetMeta::new("walk.anim", 1.5, 65));

        let removed = registry.unregister_clip("walk");
        assert!(removed.is_some());
        assert!(!registry.has_clip("walk"));
    }

    // ===== Motion Database Registration Tests =====

    #[test]
    fn test_register_motion_database() {
        let mut registry = AnimationAssetRegistry::new();
        let meta = MotionDatabaseAssetMeta::new("locomotion.mmdb", 10, 5000);

        registry.register_motion_database("locomotion", meta);

        assert!(registry.has_motion_database("locomotion"));
        assert_eq!(registry.motion_database_count(), 1);
    }

    #[test]
    fn test_register_motion_database_validated() {
        let mut registry = AnimationAssetRegistry::new();

        let valid = MotionDatabaseAssetMeta::new("locomotion.mmdb", 10, 5000);
        assert!(registry.register_motion_database_validated("locomotion", valid).is_ok());

        let invalid = MotionDatabaseAssetMeta::new("empty.mmdb", 0, 5000);
        assert!(registry.register_motion_database_validated("empty", invalid).is_err());
    }

    // ===== Query Tests =====

    #[test]
    fn test_list_by_extension() {
        let mut registry = AnimationAssetRegistry::new();

        registry.register_skeleton("hero", SkeletonAssetMeta::new("hero.skel", 65));
        registry.register_clip("walk", AnimationClipAssetMeta::new("walk.anim", 1.0, 65));
        registry.register_clip("run", AnimationClipAssetMeta::new("run.anim", 0.8, 65));
        registry.register_clip("attack", AnimationClipAssetMeta::new("attack.fbx", 0.5, 65));

        let skel_assets = registry.list_by_extension("skel");
        assert_eq!(skel_assets.len(), 1);
        assert!(skel_assets.contains(&"hero"));

        let anim_assets = registry.list_by_extension("anim");
        assert_eq!(anim_assets.len(), 2);
        assert!(anim_assets.contains(&"walk"));
        assert!(anim_assets.contains(&"run"));

        let fbx_assets = registry.list_by_extension("fbx");
        assert_eq!(fbx_assets.len(), 1);
        assert!(fbx_assets.contains(&"attack"));

        // Case insensitive
        let anim_upper = registry.list_by_extension("ANIM");
        assert_eq!(anim_upper.len(), 2);
    }

    #[test]
    fn test_list_by_type() {
        let mut registry = AnimationAssetRegistry::new();

        registry.register_skeleton("hero", SkeletonAssetMeta::new("hero.skel", 65));
        registry.register_skeleton("enemy", SkeletonAssetMeta::new("enemy.skel", 45));
        registry.register_clip("walk", AnimationClipAssetMeta::new("walk.anim", 1.0, 65));

        let skeletons = registry.list_by_type(AnimationAssetType::Skeleton);
        assert_eq!(skeletons.len(), 2);

        let clips = registry.list_by_type(AnimationAssetType::AnimationClip);
        assert_eq!(clips.len(), 1);

        let motion_dbs = registry.list_by_type(AnimationAssetType::MotionDatabase);
        assert!(motion_dbs.is_empty());
    }

    #[test]
    fn test_clips_by_duration() {
        let mut registry = AnimationAssetRegistry::new();

        registry.register_clip("short", AnimationClipAssetMeta::new("short.anim", 0.5, 65));
        registry.register_clip("medium", AnimationClipAssetMeta::new("medium.anim", 1.5, 65));
        registry.register_clip("long", AnimationClipAssetMeta::new("long.anim", 3.0, 65));

        let short_clips = registry.clips_by_duration(0.0, 1.0);
        assert_eq!(short_clips.len(), 1);
        assert!(short_clips.contains(&"short"));

        let medium_clips = registry.clips_by_duration(1.0, 2.0);
        assert_eq!(medium_clips.len(), 1);
        assert!(medium_clips.contains(&"medium"));

        let all_clips = registry.clips_by_duration(0.0, 10.0);
        assert_eq!(all_clips.len(), 3);
    }

    #[test]
    fn test_clips_by_min_bones() {
        let mut registry = AnimationAssetRegistry::new();

        registry.register_clip("simple", AnimationClipAssetMeta::new("simple.anim", 1.0, 20));
        registry.register_clip("complex", AnimationClipAssetMeta::new("complex.anim", 1.0, 65));

        let complex_clips = registry.clips_by_min_bones(50);
        assert_eq!(complex_clips.len(), 1);
        assert!(complex_clips.contains(&"complex"));
    }

    #[test]
    fn test_clips_with_events() {
        let mut registry = AnimationAssetRegistry::new();

        registry.register_clip("no_events", AnimationClipAssetMeta::new("no_events.anim", 1.0, 65));
        registry.register_clip(
            "with_events",
            AnimationClipAssetMeta::new("with_events.anim", 1.0, 65).with_event_count(4),
        );

        let event_clips = registry.clips_with_events();
        assert_eq!(event_clips.len(), 1);
        assert!(event_clips.contains(&"with_events"));
    }

    #[test]
    fn test_assets_modified_after() {
        let mut registry = AnimationAssetRegistry::new();

        registry.register_skeleton(
            "old",
            SkeletonAssetMeta::new("old.skel", 65).with_last_modified(1000),
        );
        registry.register_skeleton(
            "new",
            SkeletonAssetMeta::new("new.skel", 65).with_last_modified(2000),
        );
        registry.register_clip(
            "recent",
            AnimationClipAssetMeta::new("recent.anim", 1.0, 65).with_last_modified(1500),
        );

        let modified = registry.assets_modified_after(1200);
        assert_eq!(modified.len(), 2);

        let has_new = modified.iter().any(|(name, _)| *name == "new");
        let has_recent = modified.iter().any(|(name, _)| *name == "recent");
        assert!(has_new);
        assert!(has_recent);
    }

    #[test]
    fn test_total_asset_count() {
        let mut registry = AnimationAssetRegistry::new();

        registry.register_skeleton("hero", SkeletonAssetMeta::new("hero.skel", 65));
        registry.register_clip("walk", AnimationClipAssetMeta::new("walk.anim", 1.0, 65));
        registry.register_clip("run", AnimationClipAssetMeta::new("run.anim", 0.8, 65));
        registry.register_motion_database("locomotion", MotionDatabaseAssetMeta::new("locomotion.mmdb", 10, 5000));

        assert_eq!(registry.total_asset_count(), 4);
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut registry = AnimationAssetRegistry::new();

        registry.register_skeleton("hero", SkeletonAssetMeta::new("hero.skel", 65));
        registry.register_clip("walk", AnimationClipAssetMeta::new("walk.anim", 1.0, 65));

        registry.clear();

        assert!(registry.is_empty());
        assert_eq!(registry.skeleton_count(), 0);
        assert_eq!(registry.clip_count(), 0);
    }

    #[test]
    fn test_merge() {
        let mut registry1 = AnimationAssetRegistry::new();
        registry1.register_skeleton("hero", SkeletonAssetMeta::new("hero.skel", 65));

        let mut registry2 = AnimationAssetRegistry::new();
        registry2.register_clip("walk", AnimationClipAssetMeta::new("walk.anim", 1.0, 65));
        registry2.register_skeleton("enemy", SkeletonAssetMeta::new("enemy.skel", 45));

        registry1.merge(registry2);

        assert_eq!(registry1.skeleton_count(), 2);
        assert_eq!(registry1.clip_count(), 1);
        assert!(registry1.has_skeleton("hero"));
        assert!(registry1.has_skeleton("enemy"));
        assert!(registry1.has_clip("walk"));
    }

    #[test]
    fn test_stats() {
        let mut registry = AnimationAssetRegistry::new();

        registry.register_skeleton("hero", SkeletonAssetMeta::new("hero.skel", 65));
        registry.register_skeleton("enemy", SkeletonAssetMeta::new("enemy.skel", 45));
        registry.register_clip(
            "walk",
            AnimationClipAssetMeta::new("walk.anim", 1.5, 65).with_event_count(4),
        );
        registry.register_clip(
            "run",
            AnimationClipAssetMeta::new("run.anim", 0.8, 65).with_event_count(2),
        );
        registry.register_motion_database("locomotion", MotionDatabaseAssetMeta::new("locomotion.mmdb", 10, 5000));

        let stats = registry.stats();

        assert_eq!(stats.skeleton_count, 2);
        assert_eq!(stats.clip_count, 2);
        assert_eq!(stats.motion_database_count, 1);
        assert_eq!(stats.total_bones, 110); // 65 + 45
        assert!((stats.total_animation_duration - 2.3).abs() < 0.001); // 1.5 + 0.8
        assert_eq!(stats.total_events, 6); // 4 + 2
        assert_eq!(stats.total_poses, 5000);
    }

    #[test]
    fn test_stats_display() {
        let stats = RegistryStats {
            skeleton_count: 2,
            clip_count: 10,
            motion_database_count: 1,
            total_bones: 130,
            total_animation_duration: 15.5,
            total_events: 42,
            total_poses: 10000,
        };

        let display = format!("{}", stats);
        assert!(display.contains("2 skeletons"));
        assert!(display.contains("130 bones"));
        assert!(display.contains("10 clips"));
        assert!(display.contains("15.5s"));
        assert!(display.contains("42 events"));
        assert!(display.contains("10000 poses"));
    }

    // ===== Serialization Tests =====

    #[test]
    fn test_registry_json_roundtrip() {
        let mut original = AnimationAssetRegistry::new();

        original.register_skeleton(
            "hero",
            SkeletonAssetMeta::new("hero.skel", 65)
                .with_hash(0xDEADBEEF)
                .with_last_modified(1716681600),
        );
        original.register_clip(
            "walk",
            AnimationClipAssetMeta::new("walk.anim", 1.5, 65)
                .with_event_count(4)
                .with_hash(0xCAFEBABE),
        );
        original.register_motion_database(
            "locomotion",
            MotionDatabaseAssetMeta::new("locomotion.mmdb", 10, 5000)
                .with_feature_dimensions(64),
        );

        let json = original.to_json().unwrap();
        let recovered = AnimationAssetRegistry::from_json(&json).unwrap();

        assert_eq!(recovered.skeleton_count(), original.skeleton_count());
        assert_eq!(recovered.clip_count(), original.clip_count());
        assert_eq!(recovered.motion_database_count(), original.motion_database_count());

        let hero = recovered.get_skeleton("hero").unwrap();
        assert_eq!(hero.bone_count, 65);
        assert_eq!(hero.hash, 0xDEADBEEF);

        let walk = recovered.get_clip("walk").unwrap();
        assert_eq!(walk.duration, 1.5);
        assert_eq!(walk.event_count, 4);

        let locomotion = recovered.get_motion_database("locomotion").unwrap();
        assert_eq!(locomotion.clip_count, 10);
        assert_eq!(locomotion.feature_dimensions, 64);
    }

    #[test]
    fn test_skeleton_meta_json_roundtrip() {
        let original = SkeletonAssetMeta::new("hero.skel", 65)
            .with_hash(0xABCD1234)
            .with_last_modified(1716681600);

        let json = serde_json::to_string(&original).unwrap();
        let recovered: SkeletonAssetMeta = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered.path, original.path);
        assert_eq!(recovered.bone_count, original.bone_count);
        assert_eq!(recovered.hash, original.hash);
        assert_eq!(recovered.last_modified, original.last_modified);
    }

    #[test]
    fn test_clip_meta_json_roundtrip() {
        let original = AnimationClipAssetMeta::new("walk.anim", 1.5, 65)
            .with_event_count(4)
            .with_hash(0x12345678)
            .with_last_modified(1716681600);

        let json = serde_json::to_string(&original).unwrap();
        let recovered: AnimationClipAssetMeta = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered.path, original.path);
        assert_eq!(recovered.duration, original.duration);
        assert_eq!(recovered.bone_count, original.bone_count);
        assert_eq!(recovered.event_count, original.event_count);
    }

    // ===== Iterator Tests =====

    #[test]
    fn test_skeletons_iterator() {
        let mut registry = AnimationAssetRegistry::new();
        registry.register_skeleton("a", SkeletonAssetMeta::new("a.skel", 10));
        registry.register_skeleton("b", SkeletonAssetMeta::new("b.skel", 20));

        let total_bones: usize = registry.skeletons().map(|(_, meta)| meta.bone_count).sum();
        assert_eq!(total_bones, 30);
    }

    #[test]
    fn test_clips_iterator() {
        let mut registry = AnimationAssetRegistry::new();
        registry.register_clip("a", AnimationClipAssetMeta::new("a.anim", 1.0, 10));
        registry.register_clip("b", AnimationClipAssetMeta::new("b.anim", 2.0, 20));

        let total_duration: f32 = registry.clips().map(|(_, meta)| meta.duration).sum();
        assert!((total_duration - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_motion_databases_iterator() {
        let mut registry = AnimationAssetRegistry::new();
        registry.register_motion_database("a", MotionDatabaseAssetMeta::new("a.mmdb", 5, 1000));
        registry.register_motion_database("b", MotionDatabaseAssetMeta::new("b.mmdb", 10, 2000));

        let total_poses: usize = registry.motion_databases().map(|(_, meta)| meta.pose_count).sum();
        assert_eq!(total_poses, 3000);
    }

    // ===== Edge Cases =====

    #[test]
    fn test_empty_extension_lookup() {
        let registry = AnimationAssetRegistry::new();
        let results = registry.list_by_extension("anim");
        assert!(results.is_empty());
    }

    #[test]
    fn test_nonexistent_asset_lookup() {
        let registry = AnimationAssetRegistry::new();
        assert!(registry.get_skeleton("nonexistent").is_none());
        assert!(registry.get_clip("nonexistent").is_none());
        assert!(registry.get_motion_database("nonexistent").is_none());
    }

    #[test]
    fn test_unregister_nonexistent() {
        let mut registry = AnimationAssetRegistry::new();
        assert!(registry.unregister_skeleton("nonexistent").is_none());
        assert!(registry.unregister_clip("nonexistent").is_none());
        assert!(registry.unregister_motion_database("nonexistent").is_none());
    }

    #[test]
    fn test_overwrite_registration() {
        let mut registry = AnimationAssetRegistry::new();

        registry.register_skeleton("hero", SkeletonAssetMeta::new("hero.skel", 65));
        registry.register_skeleton("hero", SkeletonAssetMeta::new("hero_v2.skel", 70));

        assert_eq!(registry.skeleton_count(), 1);
        assert_eq!(registry.get_skeleton("hero").unwrap().bone_count, 70);
    }
}
