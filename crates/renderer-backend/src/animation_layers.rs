//! Animation layer system for TRINITY Engine (T-AN-5.5).
//!
//! This module provides a hierarchical animation layer system with:
//!
//! - **AnimationLayer**: Individual layers with blend modes and masks
//! - **LayerMask**: Per-bone masks with weight gradients and preset names
//! - **LayerStack**: Ordered collection of layers with evaluation and caching
//! - **Blend Modes**: Override, Additive, and Multiply modes
//!
//! # Architecture
//!
//! ```text
//! LayerStack
//! +-- layers: Vec<LayerEntry>
//! |   +-- layer: AnimationLayer
//! |   |   +-- name: String
//! |   |   +-- index: usize
//! |   |   +-- blend_mode: LayerBlendMode
//! |   |   +-- weight: f32 (0.0 - 1.0)
//! |   |   +-- mask: Option<LayerMask>
//! |   |   +-- active: bool
//! |   |   +-- source: LayerSource
//! |   +-- cached_pose: Option<Pose>
//! |   +-- dirty: bool
//! +-- bone_count: usize
//! +-- evaluation_order: Vec<usize>
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::animation_layers::{
//!     LayerStack, AnimationLayer, LayerMask, LayerBlendMode, MaskPreset,
//! };
//!
//! // Create a layer stack
//! let mut stack = LayerStack::new(64); // 64 bones
//!
//! // Add base locomotion layer
//! stack.add_layer(AnimationLayer::new("locomotion", LayerBlendMode::Override));
//!
//! // Add upper body layer with mask
//! let upper_mask = LayerMask::from_preset(MaskPreset::UpperBody, 64);
//! stack.add_layer(
//!     AnimationLayer::new("upper_body", LayerBlendMode::Override)
//!         .with_mask(upper_mask)
//! );
//!
//! // Add additive breathing layer
//! stack.add_layer(
//!     AnimationLayer::new("breathing", LayerBlendMode::Additive)
//!         .with_weight(0.5)
//! );
//!
//! // Evaluate all layers
//! let final_pose = stack.evaluate(&layer_poses);
//! ```

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::pose::{lerp_vec3, nlerp_quat, Pose, PoseType};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of layers in a stack.
pub const MAX_LAYERS: usize = 32;

/// Maximum number of bones per mask.
pub const MAX_MASK_BONES: usize = 512;

/// Default weight for new layers.
pub const DEFAULT_LAYER_WEIGHT: f32 = 1.0;

// ---------------------------------------------------------------------------
// LayerBlendMode
// ---------------------------------------------------------------------------

/// Blend mode for animation layers.
///
/// Controls how a layer's pose combines with the poses below it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerBlendMode {
    /// Replace base pose with layer pose (weighted).
    /// At weight 1.0, completely replaces the base.
    /// At weight 0.5, blends 50% with base.
    #[default]
    Override,

    /// Add layer delta to base pose.
    /// Layer positions/scales are added, rotations are multiplied.
    /// Useful for breathing, procedural effects, hit reactions.
    Additive,

    /// Scale base pose by layer factors.
    /// Positions are multiplied, rotations are scaled, scales are multiplied.
    /// Useful for strength modifiers, scale effects.
    Multiply,
}

impl LayerBlendMode {
    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Override => "Override",
            Self::Additive => "Additive",
            Self::Multiply => "Multiply",
        }
    }

    /// Check if this mode is additive.
    #[inline]
    pub fn is_additive(&self) -> bool {
        matches!(self, Self::Additive)
    }

    /// Check if this mode requires special handling.
    #[inline]
    pub fn is_override(&self) -> bool {
        matches!(self, Self::Override)
    }
}

// ---------------------------------------------------------------------------
// MaskPreset
// ---------------------------------------------------------------------------

/// Preset mask configurations for common use cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaskPreset {
    /// Upper body only (spine, arms, head).
    UpperBody,
    /// Lower body only (hips, legs).
    LowerBody,
    /// Left arm only.
    LeftArm,
    /// Right arm only.
    RightArm,
    /// Left leg only.
    LeftLeg,
    /// Right leg only.
    RightLeg,
    /// Head and neck only.
    Head,
    /// Spine chain only.
    Spine,
    /// Hands only.
    Hands,
    /// Feet only.
    Feet,
    /// All bones.
    FullBody,
}

impl MaskPreset {
    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::UpperBody => "Upper Body",
            Self::LowerBody => "Lower Body",
            Self::LeftArm => "Left Arm",
            Self::RightArm => "Right Arm",
            Self::LeftLeg => "Left Leg",
            Self::RightLeg => "Right Leg",
            Self::Head => "Head",
            Self::Spine => "Spine",
            Self::Hands => "Hands",
            Self::Feet => "Feet",
            Self::FullBody => "Full Body",
        }
    }

    /// Get standard bone index ranges for a 64-bone humanoid skeleton.
    ///
    /// This is a common layout:
    /// - 0: Root
    /// - 1-4: Spine (including pelvis)
    /// - 5-7: Head/Neck
    /// - 8-15: Left arm (shoulder to fingers)
    /// - 16-23: Right arm
    /// - 24-31: Left leg (hip to toes)
    /// - 32-39: Right leg
    /// - 40-63: Additional bones (fingers, twist, etc.)
    pub fn default_bone_indices(&self, bone_count: usize) -> Vec<usize> {
        let max_idx = bone_count;
        match self {
            Self::UpperBody => {
                // Spine (1-4), Head (5-7), Arms (8-23)
                (1..24.min(max_idx)).collect()
            }
            Self::LowerBody => {
                // Root (0), Pelvis (part of spine), Legs (24-39)
                let mut indices = vec![0];
                indices.extend(24..40.min(max_idx));
                indices
            }
            Self::LeftArm => (8..16.min(max_idx)).collect(),
            Self::RightArm => (16..24.min(max_idx)).collect(),
            Self::LeftLeg => (24..32.min(max_idx)).collect(),
            Self::RightLeg => (32..40.min(max_idx)).collect(),
            Self::Head => (5..8.min(max_idx)).collect(),
            Self::Spine => (1..5.min(max_idx)).collect(),
            Self::Hands => {
                // Assuming fingers are at end of arm chains (14-15, 22-23)
                let mut indices = Vec::new();
                if max_idx > 14 {
                    indices.push(14);
                }
                if max_idx > 15 {
                    indices.push(15);
                }
                if max_idx > 22 {
                    indices.push(22);
                }
                if max_idx > 23 {
                    indices.push(23);
                }
                indices
            }
            Self::Feet => {
                // Assuming feet are at end of leg chains (30-31, 38-39)
                let mut indices = Vec::new();
                if max_idx > 30 {
                    indices.push(30);
                }
                if max_idx > 31 {
                    indices.push(31);
                }
                if max_idx > 38 {
                    indices.push(38);
                }
                if max_idx > 39 {
                    indices.push(39);
                }
                indices
            }
            Self::FullBody => (0..max_idx).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// LayerMask
// ---------------------------------------------------------------------------

/// A bone mask for animation layers.
///
/// Defines which bones are affected by a layer and with what weight.
/// Supports gradual falloff for smooth blending at mask boundaries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerMask {
    /// Weight per bone (0.0 = not affected, 1.0 = fully affected).
    pub bone_weights: Vec<f32>,

    /// Optional preset this mask was created from.
    pub preset: Option<MaskPreset>,

    /// Human-readable name for this mask.
    pub name: Option<String>,
}

impl LayerMask {
    /// Create a new mask for the given bone count with all weights at 0.
    pub fn new(bone_count: usize) -> Self {
        Self {
            bone_weights: vec![0.0; bone_count],
            preset: None,
            name: None,
        }
    }

    /// Create a mask with all bones fully weighted.
    pub fn full(bone_count: usize) -> Self {
        Self {
            bone_weights: vec![1.0; bone_count],
            preset: Some(MaskPreset::FullBody),
            name: Some("Full Body".to_string()),
        }
    }

    /// Create a mask from a preset.
    pub fn from_preset(preset: MaskPreset, bone_count: usize) -> Self {
        let mut mask = Self::new(bone_count);
        mask.preset = Some(preset);
        mask.name = Some(preset.name().to_string());

        for idx in preset.default_bone_indices(bone_count) {
            if idx < bone_count {
                mask.bone_weights[idx] = 1.0;
            }
        }

        mask
    }

    /// Create a mask from a list of bone indices.
    pub fn from_indices(indices: &[usize], bone_count: usize) -> Self {
        let mut mask = Self::new(bone_count);
        for &idx in indices {
            if idx < bone_count {
                mask.bone_weights[idx] = 1.0;
            }
        }
        mask
    }

    /// Create a mask from bone indices with specific weights.
    pub fn from_weighted_indices(indices: &[(usize, f32)], bone_count: usize) -> Self {
        let mut mask = Self::new(bone_count);
        for &(idx, weight) in indices {
            if idx < bone_count {
                mask.bone_weights[idx] = weight.clamp(0.0, 1.0);
            }
        }
        mask
    }

    /// Set the mask name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Get the number of bones in this mask.
    #[inline]
    pub fn bone_count(&self) -> usize {
        self.bone_weights.len()
    }

    /// Check if the mask is empty (no bones affected).
    pub fn is_empty(&self) -> bool {
        self.bone_weights.iter().all(|&w| w <= 0.0)
    }

    /// Check if all bones are fully weighted.
    pub fn is_full(&self) -> bool {
        self.bone_weights.iter().all(|&w| w >= 1.0)
    }

    /// Get the weight for a specific bone.
    #[inline]
    pub fn get_weight(&self, bone_index: usize) -> f32 {
        self.bone_weights.get(bone_index).copied().unwrap_or(0.0)
    }

    /// Set the weight for a specific bone.
    #[inline]
    pub fn set_weight(&mut self, bone_index: usize, weight: f32) {
        if bone_index < self.bone_weights.len() {
            self.bone_weights[bone_index] = weight.clamp(0.0, 1.0);
        }
    }

    /// Set weight for a range of bones.
    pub fn set_range(&mut self, start: usize, end: usize, weight: f32) {
        let weight = weight.clamp(0.0, 1.0);
        for idx in start..end.min(self.bone_weights.len()) {
            self.bone_weights[idx] = weight;
        }
    }

    /// Add falloff around mask boundaries.
    ///
    /// For each bone at the edge of the mask, applies a gradual weight
    /// falloff to the specified number of neighboring bones.
    pub fn add_falloff(&mut self, falloff_bones: usize) {
        if falloff_bones == 0 || self.bone_weights.is_empty() {
            return;
        }

        // Find edges (bones where weight transitions from 0 to >0 or vice versa)
        let original = self.bone_weights.clone();
        let len = original.len();

        for i in 0..len {
            if original[i] <= 0.0 {
                continue;
            }

            // Apply falloff to neighboring zero-weight bones
            for offset in 1..=falloff_bones {
                let falloff_weight = 1.0 - (offset as f32 / (falloff_bones + 1) as f32);

                // Forward
                if i + offset < len && original[i + offset] <= 0.0 {
                    self.bone_weights[i + offset] =
                        self.bone_weights[i + offset].max(falloff_weight * original[i]);
                }

                // Backward
                if i >= offset && original[i - offset] <= 0.0 {
                    self.bone_weights[i - offset] =
                        self.bone_weights[i - offset].max(falloff_weight * original[i]);
                }
            }
        }
    }

    /// Compute the union of two masks (max weight per bone).
    pub fn union(&self, other: &LayerMask) -> LayerMask {
        let len = self.bone_weights.len().max(other.bone_weights.len());
        let mut result = LayerMask::new(len);

        for i in 0..len {
            let a = self.bone_weights.get(i).copied().unwrap_or(0.0);
            let b = other.bone_weights.get(i).copied().unwrap_or(0.0);
            result.bone_weights[i] = a.max(b);
        }

        result
    }

    /// Compute the intersection of two masks (min weight per bone).
    pub fn intersection(&self, other: &LayerMask) -> LayerMask {
        let len = self.bone_weights.len().min(other.bone_weights.len());
        let mut result = LayerMask::new(len);

        for i in 0..len {
            let a = self.bone_weights[i];
            let b = other.bone_weights[i];
            result.bone_weights[i] = a.min(b);
        }

        result
    }

    /// Compute the difference of two masks (subtract other from self).
    pub fn subtract(&self, other: &LayerMask) -> LayerMask {
        let len = self.bone_weights.len();
        let mut result = LayerMask::new(len);

        for i in 0..len {
            let a = self.bone_weights[i];
            let b = other.bone_weights.get(i).copied().unwrap_or(0.0);
            result.bone_weights[i] = (a - b).max(0.0);
        }

        result
    }

    /// Invert the mask (1.0 - weight for each bone).
    pub fn invert(&self) -> LayerMask {
        let mut result = LayerMask::new(self.bone_weights.len());
        for i in 0..self.bone_weights.len() {
            result.bone_weights[i] = 1.0 - self.bone_weights[i];
        }
        result
    }

    /// Scale all weights by a factor.
    pub fn scale(&mut self, factor: f32) {
        for w in &mut self.bone_weights {
            *w = (*w * factor).clamp(0.0, 1.0);
        }
    }

    /// Count the number of affected bones (weight > 0).
    pub fn affected_bone_count(&self) -> usize {
        self.bone_weights.iter().filter(|&&w| w > 0.0).count()
    }

    /// Get indices of affected bones.
    pub fn affected_indices(&self) -> Vec<usize> {
        self.bone_weights
            .iter()
            .enumerate()
            .filter(|(_, &w)| w > 0.0)
            .map(|(i, _)| i)
            .collect()
    }

    /// Resize the mask to a new bone count.
    pub fn resize(&mut self, new_bone_count: usize) {
        self.bone_weights.resize(new_bone_count, 0.0);
    }
}

impl Default for LayerMask {
    fn default() -> Self {
        Self::new(0)
    }
}

// ---------------------------------------------------------------------------
// LayerSource
// ---------------------------------------------------------------------------

/// The source of animation data for a layer.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerSource {
    /// No source (pose is provided externally).
    None,

    /// State machine by index.
    StateMachine(usize),

    /// Blend tree by index.
    BlendTree(usize),

    /// Direct clip reference by index.
    Clip(usize),

    /// Sub-graph reference by index.
    SubGraph(usize),
}

impl Default for LayerSource {
    fn default() -> Self {
        Self::None
    }
}

impl LayerSource {
    /// Check if this source has a value.
    #[inline]
    pub fn is_some(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Get the state machine index if this is a state machine source.
    #[inline]
    pub fn state_machine(&self) -> Option<usize> {
        match self {
            Self::StateMachine(idx) => Some(*idx),
            _ => None,
        }
    }

    /// Get the blend tree index if this is a blend tree source.
    #[inline]
    pub fn blend_tree(&self) -> Option<usize> {
        match self {
            Self::BlendTree(idx) => Some(*idx),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// AnimationLayer
// ---------------------------------------------------------------------------

/// An animation layer in the layer stack.
///
/// Layers are evaluated from bottom to top, with each layer's pose
/// blended into the accumulated result according to its blend mode,
/// weight, and mask.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationLayer {
    /// Human-readable layer name.
    pub name: String,

    /// Layer index in the stack (0 = base layer).
    pub index: usize,

    /// How this layer blends with layers below.
    pub blend_mode: LayerBlendMode,

    /// Blend weight (0.0 = no effect, 1.0 = full effect).
    pub weight: f32,

    /// Optional bone mask for selective blending.
    pub mask: Option<LayerMask>,

    /// Whether this layer is active.
    pub active: bool,

    /// Source of animation data for this layer.
    pub source: LayerSource,

    /// Optional sync group for time synchronization.
    pub sync_group: Option<String>,
}

impl AnimationLayer {
    /// Create a new layer with the given name and blend mode.
    pub fn new(name: impl Into<String>, blend_mode: LayerBlendMode) -> Self {
        Self {
            name: name.into(),
            index: 0,
            blend_mode,
            weight: DEFAULT_LAYER_WEIGHT,
            mask: None,
            active: true,
            source: LayerSource::None,
            sync_group: None,
        }
    }

    /// Create a base (override) layer.
    pub fn base(name: impl Into<String>) -> Self {
        Self::new(name, LayerBlendMode::Override)
    }

    /// Create an additive layer.
    pub fn additive(name: impl Into<String>) -> Self {
        Self::new(name, LayerBlendMode::Additive)
    }

    /// Create a multiply layer.
    pub fn multiply(name: impl Into<String>) -> Self {
        Self::new(name, LayerBlendMode::Multiply)
    }

    /// Set the blend weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the bone mask.
    pub fn with_mask(mut self, mask: LayerMask) -> Self {
        self.mask = Some(mask);
        self
    }

    /// Set the layer source.
    pub fn with_source(mut self, source: LayerSource) -> Self {
        self.source = source;
        self
    }

    /// Set a state machine as the source.
    pub fn with_state_machine(mut self, index: usize) -> Self {
        self.source = LayerSource::StateMachine(index);
        self
    }

    /// Set a blend tree as the source.
    pub fn with_blend_tree(mut self, index: usize) -> Self {
        self.source = LayerSource::BlendTree(index);
        self
    }

    /// Set a sync group.
    pub fn with_sync_group(mut self, group: impl Into<String>) -> Self {
        self.sync_group = Some(group.into());
        self
    }

    /// Deactivate the layer.
    pub fn inactive(mut self) -> Self {
        self.active = false;
        self
    }

    /// Set the layer index.
    pub fn with_index(mut self, index: usize) -> Self {
        self.index = index;
        self
    }

    /// Check if this layer affects a specific bone.
    #[inline]
    pub fn affects_bone(&self, bone_index: usize) -> bool {
        match &self.mask {
            None => true,
            Some(mask) => mask.get_weight(bone_index) > 0.0,
        }
    }

    /// Get the effective weight for a specific bone.
    ///
    /// Combines layer weight with mask weight.
    #[inline]
    pub fn effective_weight(&self, bone_index: usize) -> f32 {
        if !self.active {
            return 0.0;
        }
        match &self.mask {
            None => self.weight,
            Some(mask) => self.weight * mask.get_weight(bone_index),
        }
    }

    /// Check if this layer should be evaluated.
    #[inline]
    pub fn should_evaluate(&self) -> bool {
        self.active && self.weight > 0.0
    }
}

impl Default for AnimationLayer {
    fn default() -> Self {
        Self::new("default", LayerBlendMode::Override)
    }
}

// ---------------------------------------------------------------------------
// LayerEntry
// ---------------------------------------------------------------------------

/// Internal entry for a layer in the stack with caching support.
#[derive(Clone, Debug)]
struct LayerEntry {
    /// The animation layer configuration.
    layer: AnimationLayer,

    /// Cached output pose from last evaluation.
    cached_pose: Option<Pose>,

    /// Whether this layer needs re-evaluation.
    dirty: bool,

    /// Frame number when cache was last updated.
    cache_frame: u64,
}

impl LayerEntry {
    /// Create a new entry for the given layer.
    fn new(layer: AnimationLayer) -> Self {
        Self {
            layer,
            cached_pose: None,
            dirty: true,
            cache_frame: 0,
        }
    }

    /// Mark the layer as dirty (needs re-evaluation).
    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Clear the dirty flag.
    fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Invalidate the cache.
    fn invalidate_cache(&mut self) {
        self.cached_pose = None;
        self.dirty = true;
    }
}

// ---------------------------------------------------------------------------
// LayerStack
// ---------------------------------------------------------------------------

/// A stack of animation layers for hierarchical pose blending.
///
/// The stack evaluates layers from bottom (index 0) to top, blending
/// each layer's pose into the accumulated result. The base layer (index 0)
/// is always evaluated first as the starting pose.
#[derive(Clone, Debug)]
pub struct LayerStack {
    /// Ordered list of layer entries.
    entries: Vec<LayerEntry>,

    /// Number of bones in the skeleton.
    bone_count: usize,

    /// Current evaluation frame number.
    frame: u64,

    /// Optional name for debugging.
    pub name: Option<String>,
}

impl LayerStack {
    /// Create a new empty layer stack.
    pub fn new(bone_count: usize) -> Self {
        Self {
            entries: Vec::new(),
            bone_count,
            frame: 0,
            name: None,
        }
    }

    /// Set the stack name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Get the number of bones.
    #[inline]
    pub fn bone_count(&self) -> usize {
        self.bone_count
    }

    /// Get the number of layers.
    #[inline]
    pub fn layer_count(&self) -> usize {
        self.entries.len()
    }

    /// Check if the stack is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // -------------------------------------------------------------------------
    // Layer Management
    // -------------------------------------------------------------------------

    /// Add a layer to the top of the stack.
    ///
    /// Returns the index of the added layer.
    pub fn add_layer(&mut self, mut layer: AnimationLayer) -> usize {
        let index = self.entries.len();
        layer.index = index;
        self.entries.push(LayerEntry::new(layer));
        index
    }

    /// Insert a layer at a specific position.
    ///
    /// All layers above are shifted up.
    pub fn insert_layer(&mut self, position: usize, mut layer: AnimationLayer) {
        let position = position.min(self.entries.len());
        layer.index = position;
        self.entries.insert(position, LayerEntry::new(layer));

        // Update indices of shifted layers
        for i in (position + 1)..self.entries.len() {
            self.entries[i].layer.index = i;
        }
    }

    /// Remove a layer by index.
    ///
    /// Returns the removed layer, or None if index is invalid.
    pub fn remove_layer(&mut self, index: usize) -> Option<AnimationLayer> {
        if index >= self.entries.len() {
            return None;
        }

        let entry = self.entries.remove(index);

        // Update indices of shifted layers
        for i in index..self.entries.len() {
            self.entries[i].layer.index = i;
        }

        Some(entry.layer)
    }

    /// Move a layer to a new position.
    pub fn move_layer(&mut self, from: usize, to: usize) {
        if from >= self.entries.len() || to >= self.entries.len() || from == to {
            return;
        }

        let entry = self.entries.remove(from);
        self.entries.insert(to, entry);

        // Update all indices
        for (i, entry) in self.entries.iter_mut().enumerate() {
            entry.layer.index = i;
        }
    }

    /// Swap two layers.
    pub fn swap_layers(&mut self, a: usize, b: usize) {
        if a < self.entries.len() && b < self.entries.len() && a != b {
            self.entries.swap(a, b);
            self.entries[a].layer.index = a;
            self.entries[b].layer.index = b;
        }
    }

    /// Get a layer by index.
    pub fn get_layer(&self, index: usize) -> Option<&AnimationLayer> {
        self.entries.get(index).map(|e| &e.layer)
    }

    /// Get a mutable layer by index.
    pub fn get_layer_mut(&mut self, index: usize) -> Option<&mut AnimationLayer> {
        self.entries.get_mut(index).map(|e| {
            e.dirty = true;
            &mut e.layer
        })
    }

    /// Find a layer by name.
    pub fn find_layer(&self, name: &str) -> Option<usize> {
        self.entries.iter().position(|e| e.layer.name == name)
    }

    /// Set layer weight.
    pub fn set_layer_weight(&mut self, index: usize, weight: f32) -> bool {
        if let Some(entry) = self.entries.get_mut(index) {
            entry.layer.weight = weight.clamp(0.0, 1.0);
            entry.dirty = true;
            true
        } else {
            false
        }
    }

    /// Set layer active state.
    pub fn set_layer_active(&mut self, index: usize, active: bool) -> bool {
        if let Some(entry) = self.entries.get_mut(index) {
            entry.layer.active = active;
            entry.dirty = true;
            true
        } else {
            false
        }
    }

    /// Get the base layer (index 0).
    pub fn base_layer(&self) -> Option<&AnimationLayer> {
        self.get_layer(0)
    }

    /// Get mutable base layer.
    pub fn base_layer_mut(&mut self) -> Option<&mut AnimationLayer> {
        self.get_layer_mut(0)
    }

    /// Iterate over all layers.
    pub fn layers(&self) -> impl Iterator<Item = &AnimationLayer> {
        self.entries.iter().map(|e| &e.layer)
    }

    /// Iterate over active layers only.
    pub fn active_layers(&self) -> impl Iterator<Item = &AnimationLayer> {
        self.entries
            .iter()
            .filter(|e| e.layer.should_evaluate())
            .map(|e| &e.layer)
    }

    // -------------------------------------------------------------------------
    // Dirty Flags
    // -------------------------------------------------------------------------

    /// Mark a layer as dirty.
    pub fn mark_dirty(&mut self, index: usize) {
        if let Some(entry) = self.entries.get_mut(index) {
            entry.mark_dirty();
        }
    }

    /// Mark all layers as dirty.
    pub fn mark_all_dirty(&mut self) {
        for entry in &mut self.entries {
            entry.mark_dirty();
        }
    }

    /// Check if any layer is dirty.
    pub fn is_dirty(&self) -> bool {
        self.entries.iter().any(|e| e.dirty)
    }

    /// Clear all dirty flags.
    pub fn clear_dirty(&mut self) {
        for entry in &mut self.entries {
            entry.clear_dirty();
        }
    }

    /// Invalidate all caches.
    pub fn invalidate_caches(&mut self) {
        for entry in &mut self.entries {
            entry.invalidate_cache();
        }
    }

    /// Check if a specific layer is dirty.
    pub fn is_layer_dirty(&self, index: usize) -> bool {
        self.entries.get(index).map(|e| e.dirty).unwrap_or(false)
    }

    // -------------------------------------------------------------------------
    // Evaluation
    // -------------------------------------------------------------------------

    /// Evaluate all layers and return the final blended pose.
    ///
    /// The `layer_poses` slice should contain one pose per layer.
    /// Layers are blended bottom-to-top according to their blend modes.
    pub fn evaluate(&mut self, layer_poses: &[Pose]) -> Pose {
        self.frame += 1;

        if self.entries.is_empty() || self.bone_count == 0 {
            return Pose::new(self.bone_count, PoseType::Current);
        }

        // Start with identity pose
        let mut result = Pose::new(self.bone_count, PoseType::Current);
        let mut has_base = false;

        for (i, entry) in self.entries.iter().enumerate() {
            if !entry.layer.should_evaluate() {
                continue;
            }

            // Get the pose for this layer
            let layer_pose = match layer_poses.get(i) {
                Some(pose) => pose,
                None => continue,
            };

            // Blend based on mode
            if !has_base && entry.layer.blend_mode == LayerBlendMode::Override {
                // First override layer becomes the base
                result = self.apply_layer_override(&result, layer_pose, &entry.layer, true);
                has_base = true;
            } else {
                // Subsequent layers blend according to their mode
                result = match entry.layer.blend_mode {
                    LayerBlendMode::Override => {
                        self.apply_layer_override(&result, layer_pose, &entry.layer, false)
                    }
                    LayerBlendMode::Additive => {
                        self.apply_layer_additive(&result, layer_pose, &entry.layer)
                    }
                    LayerBlendMode::Multiply => {
                        self.apply_layer_multiply(&result, layer_pose, &entry.layer)
                    }
                };
            }
        }

        result
    }

    /// Evaluate with cached layer poses.
    ///
    /// This version uses internally cached poses when available.
    /// Call `set_layer_pose` to update cached poses.
    pub fn evaluate_cached(&mut self) -> Pose {
        self.frame += 1;

        if self.entries.is_empty() || self.bone_count == 0 {
            return Pose::new(self.bone_count, PoseType::Current);
        }

        let mut result = Pose::new(self.bone_count, PoseType::Current);
        let mut has_base = false;

        // Collect poses first to avoid borrow issues
        let poses_and_layers: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.layer.should_evaluate() && e.cached_pose.is_some())
            .map(|e| (e.cached_pose.as_ref().unwrap().clone(), e.layer.clone()))
            .collect();

        for (layer_pose, layer) in poses_and_layers {
            if !has_base && layer.blend_mode == LayerBlendMode::Override {
                result = self.apply_layer_override(&result, &layer_pose, &layer, true);
                has_base = true;
            } else {
                result = match layer.blend_mode {
                    LayerBlendMode::Override => {
                        self.apply_layer_override(&result, &layer_pose, &layer, false)
                    }
                    LayerBlendMode::Additive => {
                        self.apply_layer_additive(&result, &layer_pose, &layer)
                    }
                    LayerBlendMode::Multiply => {
                        self.apply_layer_multiply(&result, &layer_pose, &layer)
                    }
                };
            }
        }

        result
    }

    /// Set the cached pose for a layer.
    pub fn set_layer_pose(&mut self, index: usize, pose: Pose) {
        if let Some(entry) = self.entries.get_mut(index) {
            entry.cached_pose = Some(pose);
            entry.cache_frame = self.frame;
            entry.dirty = false;
        }
    }

    /// Get the cached pose for a layer.
    pub fn get_layer_pose(&self, index: usize) -> Option<&Pose> {
        self.entries.get(index).and_then(|e| e.cached_pose.as_ref())
    }

    // -------------------------------------------------------------------------
    // Blend Helpers
    // -------------------------------------------------------------------------

    /// Apply override blend mode.
    fn apply_layer_override(
        &self,
        base: &Pose,
        layer: &Pose,
        layer_cfg: &AnimationLayer,
        is_first: bool,
    ) -> Pose {
        let bone_count = base.bone_count().min(layer.bone_count()).min(self.bone_count);
        let mut result = Pose::new(bone_count, PoseType::Current);

        for i in 0..bone_count {
            let w = layer_cfg.effective_weight(i);

            if w <= 0.0 {
                // No effect from this layer
                result.positions[i] = base.positions[i];
                result.rotations[i] = base.rotations[i];
                result.scales[i] = base.scales[i];
            } else if w >= 1.0 || is_first {
                // Full replacement
                result.positions[i] = layer.positions[i];
                result.rotations[i] = layer.rotations[i];
                result.scales[i] = layer.scales[i];
            } else {
                // Weighted blend
                result.positions[i] = lerp_vec3(base.positions[i], layer.positions[i], w);
                result.rotations[i] = nlerp_quat(base.rotations[i], layer.rotations[i], w);
                result.scales[i] = lerp_vec3(base.scales[i], layer.scales[i], w);
            }
        }

        result
    }

    /// Apply additive blend mode.
    fn apply_layer_additive(
        &self,
        base: &Pose,
        layer: &Pose,
        layer_cfg: &AnimationLayer,
    ) -> Pose {
        let bone_count = base.bone_count().min(layer.bone_count()).min(self.bone_count);
        let mut result = Pose::new(bone_count, PoseType::Current);

        for i in 0..bone_count {
            let w = layer_cfg.effective_weight(i);

            if w <= 0.0 {
                result.positions[i] = base.positions[i];
                result.rotations[i] = base.rotations[i];
                result.scales[i] = base.scales[i];
            } else {
                // Additive: add position/scale, multiply rotation
                result.positions[i] = base.positions[i] + layer.positions[i] * w;

                // For rotation: interpolate from identity to layer rotation, then multiply
                let additive_rot = nlerp_quat(Quat::IDENTITY, layer.rotations[i], w);
                result.rotations[i] = (base.rotations[i] * additive_rot).normalize();

                result.scales[i] = base.scales[i] + layer.scales[i] * w;
            }
        }

        result
    }

    /// Apply multiply blend mode.
    fn apply_layer_multiply(
        &self,
        base: &Pose,
        layer: &Pose,
        layer_cfg: &AnimationLayer,
    ) -> Pose {
        let bone_count = base.bone_count().min(layer.bone_count()).min(self.bone_count);
        let mut result = Pose::new(bone_count, PoseType::Current);

        for i in 0..bone_count {
            let w = layer_cfg.effective_weight(i);

            if w <= 0.0 {
                result.positions[i] = base.positions[i];
                result.rotations[i] = base.rotations[i];
                result.scales[i] = base.scales[i];
            } else if w >= 1.0 {
                // Full multiply
                result.positions[i] = Vec3::new(
                    base.positions[i].x * layer.positions[i].x,
                    base.positions[i].y * layer.positions[i].y,
                    base.positions[i].z * layer.positions[i].z,
                );

                // For rotation multiply: apply layer rotation scaled
                result.rotations[i] = (base.rotations[i] * layer.rotations[i]).normalize();

                result.scales[i] = Vec3::new(
                    base.scales[i].x * layer.scales[i].x,
                    base.scales[i].y * layer.scales[i].y,
                    base.scales[i].z * layer.scales[i].z,
                );
            } else {
                // Partial multiply - blend between base and multiplied result
                let mult_pos = Vec3::new(
                    base.positions[i].x * layer.positions[i].x,
                    base.positions[i].y * layer.positions[i].y,
                    base.positions[i].z * layer.positions[i].z,
                );
                result.positions[i] = lerp_vec3(base.positions[i], mult_pos, w);

                let mult_rot = (base.rotations[i] * layer.rotations[i]).normalize();
                result.rotations[i] = nlerp_quat(base.rotations[i], mult_rot, w);

                let mult_scale = Vec3::new(
                    base.scales[i].x * layer.scales[i].x,
                    base.scales[i].y * layer.scales[i].y,
                    base.scales[i].z * layer.scales[i].z,
                );
                result.scales[i] = lerp_vec3(base.scales[i], mult_scale, w);
            }
        }

        result
    }

    /// Set the bone count and resize all mask if needed.
    pub fn set_bone_count(&mut self, bone_count: usize) {
        self.bone_count = bone_count;
        for entry in &mut self.entries {
            if let Some(ref mut mask) = entry.layer.mask {
                mask.resize(bone_count);
            }
        }
    }

    /// Get the current frame number.
    #[inline]
    pub fn frame(&self) -> u64 {
        self.frame
    }
}

impl Default for LayerStack {
    fn default() -> Self {
        Self::new(0)
    }
}

// ---------------------------------------------------------------------------
// LayerStackBuilder
// ---------------------------------------------------------------------------

/// Builder for creating layer stacks with a fluent API.
pub struct LayerStackBuilder {
    stack: LayerStack,
}

impl LayerStackBuilder {
    /// Create a new builder with the given bone count.
    pub fn new(bone_count: usize) -> Self {
        Self {
            stack: LayerStack::new(bone_count),
        }
    }

    /// Set the stack name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.stack.name = Some(name.into());
        self
    }

    /// Add a layer.
    pub fn layer(mut self, layer: AnimationLayer) -> Self {
        self.stack.add_layer(layer);
        self
    }

    /// Add a base override layer.
    pub fn base_layer(mut self, name: impl Into<String>) -> Self {
        self.stack.add_layer(AnimationLayer::base(name));
        self
    }

    /// Add an additive layer.
    pub fn additive_layer(mut self, name: impl Into<String>, weight: f32) -> Self {
        self.stack
            .add_layer(AnimationLayer::additive(name).with_weight(weight));
        self
    }

    /// Add a masked override layer.
    pub fn masked_layer(
        mut self,
        name: impl Into<String>,
        preset: MaskPreset,
        weight: f32,
    ) -> Self {
        let mask = LayerMask::from_preset(preset, self.stack.bone_count);
        self.stack.add_layer(
            AnimationLayer::base(name)
                .with_mask(mask)
                .with_weight(weight),
        );
        self
    }

    /// Build the layer stack.
    pub fn build(self) -> LayerStack {
        self.stack
    }
}

impl Default for LayerStackBuilder {
    fn default() -> Self {
        Self::new(64)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // =========================================================================
    // LayerBlendMode Tests
    // =========================================================================

    #[test]
    fn test_blend_mode_default() {
        assert_eq!(LayerBlendMode::default(), LayerBlendMode::Override);
    }

    #[test]
    fn test_blend_mode_name() {
        assert_eq!(LayerBlendMode::Override.name(), "Override");
        assert_eq!(LayerBlendMode::Additive.name(), "Additive");
        assert_eq!(LayerBlendMode::Multiply.name(), "Multiply");
    }

    #[test]
    fn test_blend_mode_is_additive() {
        assert!(!LayerBlendMode::Override.is_additive());
        assert!(LayerBlendMode::Additive.is_additive());
        assert!(!LayerBlendMode::Multiply.is_additive());
    }

    #[test]
    fn test_blend_mode_is_override() {
        assert!(LayerBlendMode::Override.is_override());
        assert!(!LayerBlendMode::Additive.is_override());
        assert!(!LayerBlendMode::Multiply.is_override());
    }

    // =========================================================================
    // MaskPreset Tests
    // =========================================================================

    #[test]
    fn test_mask_preset_name() {
        assert_eq!(MaskPreset::UpperBody.name(), "Upper Body");
        assert_eq!(MaskPreset::LowerBody.name(), "Lower Body");
        assert_eq!(MaskPreset::LeftArm.name(), "Left Arm");
        assert_eq!(MaskPreset::Head.name(), "Head");
        assert_eq!(MaskPreset::FullBody.name(), "Full Body");
    }

    #[test]
    fn test_mask_preset_default_indices_upper_body() {
        let indices = MaskPreset::UpperBody.default_bone_indices(64);
        // Should include spine (1-4), head (5-7), arms (8-23)
        assert!(indices.contains(&1));
        assert!(indices.contains(&5));
        assert!(indices.contains(&10));
        assert!(!indices.contains(&0)); // Not root
        assert!(!indices.contains(&30)); // Not legs
    }

    #[test]
    fn test_mask_preset_default_indices_lower_body() {
        let indices = MaskPreset::LowerBody.default_bone_indices(64);
        assert!(indices.contains(&0)); // Root
        assert!(indices.contains(&25)); // Left leg
        assert!(indices.contains(&35)); // Right leg
    }

    #[test]
    fn test_mask_preset_full_body() {
        let indices = MaskPreset::FullBody.default_bone_indices(10);
        assert_eq!(indices.len(), 10);
        for i in 0..10 {
            assert!(indices.contains(&i));
        }
    }

    #[test]
    fn test_mask_preset_respects_bone_count() {
        // With fewer bones, presets should not exceed bone count
        let indices = MaskPreset::LeftArm.default_bone_indices(5);
        for idx in &indices {
            assert!(*idx < 5);
        }
    }

    // =========================================================================
    // LayerMask Tests
    // =========================================================================

    #[test]
    fn test_layer_mask_new() {
        let mask = LayerMask::new(10);
        assert_eq!(mask.bone_count(), 10);
        assert!(mask.is_empty());
        for i in 0..10 {
            assert_eq!(mask.get_weight(i), 0.0);
        }
    }

    #[test]
    fn test_layer_mask_full() {
        let mask = LayerMask::full(5);
        assert_eq!(mask.bone_count(), 5);
        assert!(mask.is_full());
        for i in 0..5 {
            assert_eq!(mask.get_weight(i), 1.0);
        }
    }

    #[test]
    fn test_layer_mask_from_preset() {
        let mask = LayerMask::from_preset(MaskPreset::Head, 64);
        assert_eq!(mask.preset, Some(MaskPreset::Head));
        assert_eq!(mask.name, Some("Head".to_string()));

        // Head bones (5-7) should be weighted
        assert!(mask.get_weight(5) > 0.0);
        assert!(mask.get_weight(6) > 0.0);
        assert!(mask.get_weight(7) > 0.0);

        // Other bones should not be weighted
        assert_eq!(mask.get_weight(0), 0.0);
        assert_eq!(mask.get_weight(10), 0.0);
    }

    #[test]
    fn test_layer_mask_from_indices() {
        let mask = LayerMask::from_indices(&[0, 2, 4], 5);
        assert_eq!(mask.get_weight(0), 1.0);
        assert_eq!(mask.get_weight(1), 0.0);
        assert_eq!(mask.get_weight(2), 1.0);
        assert_eq!(mask.get_weight(3), 0.0);
        assert_eq!(mask.get_weight(4), 1.0);
    }

    #[test]
    fn test_layer_mask_from_weighted_indices() {
        let mask = LayerMask::from_weighted_indices(&[(0, 1.0), (1, 0.5), (2, 0.25)], 5);
        assert_eq!(mask.get_weight(0), 1.0);
        assert_eq!(mask.get_weight(1), 0.5);
        assert_eq!(mask.get_weight(2), 0.25);
        assert_eq!(mask.get_weight(3), 0.0);
    }

    #[test]
    fn test_layer_mask_set_weight() {
        let mut mask = LayerMask::new(5);
        mask.set_weight(2, 0.75);
        assert_eq!(mask.get_weight(2), 0.75);

        // Clamps to valid range
        mask.set_weight(3, 2.0);
        assert_eq!(mask.get_weight(3), 1.0);

        mask.set_weight(4, -0.5);
        assert_eq!(mask.get_weight(4), 0.0);
    }

    #[test]
    fn test_layer_mask_set_range() {
        let mut mask = LayerMask::new(10);
        mask.set_range(2, 5, 0.8);

        assert_eq!(mask.get_weight(0), 0.0);
        assert_eq!(mask.get_weight(1), 0.0);
        assert_eq!(mask.get_weight(2), 0.8);
        assert_eq!(mask.get_weight(3), 0.8);
        assert_eq!(mask.get_weight(4), 0.8);
        assert_eq!(mask.get_weight(5), 0.0);
    }

    #[test]
    fn test_layer_mask_union() {
        let a = LayerMask::from_indices(&[0, 1, 2], 5);
        let b = LayerMask::from_indices(&[2, 3, 4], 5);
        let result = a.union(&b);

        assert_eq!(result.get_weight(0), 1.0);
        assert_eq!(result.get_weight(1), 1.0);
        assert_eq!(result.get_weight(2), 1.0);
        assert_eq!(result.get_weight(3), 1.0);
        assert_eq!(result.get_weight(4), 1.0);
    }

    #[test]
    fn test_layer_mask_intersection() {
        let a = LayerMask::from_indices(&[0, 1, 2], 5);
        let b = LayerMask::from_indices(&[1, 2, 3], 5);
        let result = a.intersection(&b);

        assert_eq!(result.get_weight(0), 0.0);
        assert_eq!(result.get_weight(1), 1.0);
        assert_eq!(result.get_weight(2), 1.0);
        assert_eq!(result.get_weight(3), 0.0);
    }

    #[test]
    fn test_layer_mask_subtract() {
        let a = LayerMask::from_indices(&[0, 1, 2, 3], 5);
        let b = LayerMask::from_indices(&[2, 3], 5);
        let result = a.subtract(&b);

        assert_eq!(result.get_weight(0), 1.0);
        assert_eq!(result.get_weight(1), 1.0);
        assert_eq!(result.get_weight(2), 0.0);
        assert_eq!(result.get_weight(3), 0.0);
    }

    #[test]
    fn test_layer_mask_invert() {
        let mask = LayerMask::from_indices(&[0, 2], 4);
        let inverted = mask.invert();

        assert_eq!(inverted.get_weight(0), 0.0);
        assert_eq!(inverted.get_weight(1), 1.0);
        assert_eq!(inverted.get_weight(2), 0.0);
        assert_eq!(inverted.get_weight(3), 1.0);
    }

    #[test]
    fn test_layer_mask_scale() {
        let mut mask = LayerMask::full(3);
        mask.scale(0.5);

        assert_eq!(mask.get_weight(0), 0.5);
        assert_eq!(mask.get_weight(1), 0.5);
        assert_eq!(mask.get_weight(2), 0.5);
    }

    #[test]
    fn test_layer_mask_affected_bone_count() {
        let mask = LayerMask::from_indices(&[0, 2, 4], 10);
        assert_eq!(mask.affected_bone_count(), 3);
    }

    #[test]
    fn test_layer_mask_affected_indices() {
        let mask = LayerMask::from_indices(&[1, 3, 5], 7);
        let affected = mask.affected_indices();
        assert_eq!(affected, vec![1, 3, 5]);
    }

    #[test]
    fn test_layer_mask_resize() {
        let mut mask = LayerMask::from_indices(&[0, 1], 3);
        mask.resize(5);

        assert_eq!(mask.bone_count(), 5);
        assert_eq!(mask.get_weight(0), 1.0);
        assert_eq!(mask.get_weight(1), 1.0);
        assert_eq!(mask.get_weight(2), 0.0);
        assert_eq!(mask.get_weight(3), 0.0);
        assert_eq!(mask.get_weight(4), 0.0);
    }

    #[test]
    fn test_layer_mask_add_falloff() {
        let mut mask = LayerMask::from_indices(&[2], 5);
        mask.add_falloff(1);

        // Bone 2 should be fully weighted
        assert_eq!(mask.get_weight(2), 1.0);
        // Neighbors should have falloff
        assert!(mask.get_weight(1) > 0.0 && mask.get_weight(1) < 1.0);
        assert!(mask.get_weight(3) > 0.0 && mask.get_weight(3) < 1.0);
    }

    #[test]
    fn test_layer_mask_out_of_bounds() {
        let mask = LayerMask::new(5);
        // Out of bounds returns 0
        assert_eq!(mask.get_weight(100), 0.0);
    }

    #[test]
    fn test_layer_mask_with_name() {
        let mask = LayerMask::new(5).with_name("Custom Mask");
        assert_eq!(mask.name, Some("Custom Mask".to_string()));
    }

    // =========================================================================
    // LayerSource Tests
    // =========================================================================

    #[test]
    fn test_layer_source_default() {
        assert_eq!(LayerSource::default(), LayerSource::None);
    }

    #[test]
    fn test_layer_source_is_some() {
        assert!(!LayerSource::None.is_some());
        assert!(LayerSource::StateMachine(0).is_some());
        assert!(LayerSource::BlendTree(0).is_some());
    }

    #[test]
    fn test_layer_source_getters() {
        assert_eq!(LayerSource::StateMachine(5).state_machine(), Some(5));
        assert_eq!(LayerSource::StateMachine(5).blend_tree(), None);

        assert_eq!(LayerSource::BlendTree(3).blend_tree(), Some(3));
        assert_eq!(LayerSource::BlendTree(3).state_machine(), None);
    }

    // =========================================================================
    // AnimationLayer Tests
    // =========================================================================

    #[test]
    fn test_animation_layer_new() {
        let layer = AnimationLayer::new("test", LayerBlendMode::Additive);
        assert_eq!(layer.name, "test");
        assert_eq!(layer.blend_mode, LayerBlendMode::Additive);
        assert_eq!(layer.weight, 1.0);
        assert!(layer.active);
        assert!(layer.mask.is_none());
    }

    #[test]
    fn test_animation_layer_base() {
        let layer = AnimationLayer::base("locomotion");
        assert_eq!(layer.blend_mode, LayerBlendMode::Override);
    }

    #[test]
    fn test_animation_layer_additive() {
        let layer = AnimationLayer::additive("breathing");
        assert_eq!(layer.blend_mode, LayerBlendMode::Additive);
    }

    #[test]
    fn test_animation_layer_multiply() {
        let layer = AnimationLayer::multiply("strength");
        assert_eq!(layer.blend_mode, LayerBlendMode::Multiply);
    }

    #[test]
    fn test_animation_layer_with_weight() {
        let layer = AnimationLayer::base("test").with_weight(0.5);
        assert_eq!(layer.weight, 0.5);

        // Should clamp
        let layer2 = AnimationLayer::base("test").with_weight(2.0);
        assert_eq!(layer2.weight, 1.0);

        let layer3 = AnimationLayer::base("test").with_weight(-0.5);
        assert_eq!(layer3.weight, 0.0);
    }

    #[test]
    fn test_animation_layer_with_mask() {
        let mask = LayerMask::from_preset(MaskPreset::UpperBody, 64);
        let layer = AnimationLayer::base("upper").with_mask(mask);
        assert!(layer.mask.is_some());
    }

    #[test]
    fn test_animation_layer_with_source() {
        let layer = AnimationLayer::base("test")
            .with_source(LayerSource::StateMachine(0));
        assert_eq!(layer.source, LayerSource::StateMachine(0));
    }

    #[test]
    fn test_animation_layer_with_state_machine() {
        let layer = AnimationLayer::base("test").with_state_machine(5);
        assert_eq!(layer.source, LayerSource::StateMachine(5));
    }

    #[test]
    fn test_animation_layer_with_blend_tree() {
        let layer = AnimationLayer::base("test").with_blend_tree(3);
        assert_eq!(layer.source, LayerSource::BlendTree(3));
    }

    #[test]
    fn test_animation_layer_inactive() {
        let layer = AnimationLayer::base("test").inactive();
        assert!(!layer.active);
    }

    #[test]
    fn test_animation_layer_affects_bone() {
        // No mask - affects all
        let layer = AnimationLayer::base("test");
        assert!(layer.affects_bone(0));
        assert!(layer.affects_bone(100));

        // With mask
        let mask = LayerMask::from_indices(&[0, 2], 5);
        let masked = AnimationLayer::base("test").with_mask(mask);
        assert!(masked.affects_bone(0));
        assert!(!masked.affects_bone(1));
        assert!(masked.affects_bone(2));
    }

    #[test]
    fn test_animation_layer_effective_weight() {
        let layer = AnimationLayer::base("test").with_weight(0.8);
        assert_eq!(layer.effective_weight(0), 0.8);

        // Inactive layer
        let inactive = AnimationLayer::base("test").inactive();
        assert_eq!(inactive.effective_weight(0), 0.0);

        // With mask
        let mask = LayerMask::from_weighted_indices(&[(0, 0.5), (1, 1.0)], 3);
        let masked = AnimationLayer::base("test").with_mask(mask).with_weight(0.8);
        assert!((masked.effective_weight(0) - 0.4).abs() < 1e-6); // 0.8 * 0.5
        assert!((masked.effective_weight(1) - 0.8).abs() < 1e-6); // 0.8 * 1.0
        assert_eq!(masked.effective_weight(2), 0.0); // 0.8 * 0.0
    }

    #[test]
    fn test_animation_layer_should_evaluate() {
        let active = AnimationLayer::base("test");
        assert!(active.should_evaluate());

        let inactive = AnimationLayer::base("test").inactive();
        assert!(!inactive.should_evaluate());

        let zero_weight = AnimationLayer::base("test").with_weight(0.0);
        assert!(!zero_weight.should_evaluate());
    }

    // =========================================================================
    // LayerStack Tests
    // =========================================================================

    #[test]
    fn test_layer_stack_new() {
        let stack = LayerStack::new(64);
        assert_eq!(stack.bone_count(), 64);
        assert_eq!(stack.layer_count(), 0);
        assert!(stack.is_empty());
    }

    #[test]
    fn test_layer_stack_add_layer() {
        let mut stack = LayerStack::new(64);
        let idx = stack.add_layer(AnimationLayer::base("locomotion"));

        assert_eq!(idx, 0);
        assert_eq!(stack.layer_count(), 1);
        assert!(!stack.is_empty());

        let idx2 = stack.add_layer(AnimationLayer::additive("breathing"));
        assert_eq!(idx2, 1);
        assert_eq!(stack.layer_count(), 2);
    }

    #[test]
    fn test_layer_stack_insert_layer() {
        let mut stack = LayerStack::new(64);
        stack.add_layer(AnimationLayer::base("a"));
        stack.add_layer(AnimationLayer::base("c"));
        stack.insert_layer(1, AnimationLayer::base("b"));

        assert_eq!(stack.get_layer(0).unwrap().name, "a");
        assert_eq!(stack.get_layer(1).unwrap().name, "b");
        assert_eq!(stack.get_layer(2).unwrap().name, "c");

        // Check indices were updated
        assert_eq!(stack.get_layer(0).unwrap().index, 0);
        assert_eq!(stack.get_layer(1).unwrap().index, 1);
        assert_eq!(stack.get_layer(2).unwrap().index, 2);
    }

    #[test]
    fn test_layer_stack_remove_layer() {
        let mut stack = LayerStack::new(64);
        stack.add_layer(AnimationLayer::base("a"));
        stack.add_layer(AnimationLayer::base("b"));
        stack.add_layer(AnimationLayer::base("c"));

        let removed = stack.remove_layer(1);
        assert_eq!(removed.unwrap().name, "b");
        assert_eq!(stack.layer_count(), 2);

        assert_eq!(stack.get_layer(0).unwrap().name, "a");
        assert_eq!(stack.get_layer(1).unwrap().name, "c");

        // Check indices
        assert_eq!(stack.get_layer(0).unwrap().index, 0);
        assert_eq!(stack.get_layer(1).unwrap().index, 1);
    }

    #[test]
    fn test_layer_stack_remove_invalid() {
        let mut stack = LayerStack::new(64);
        stack.add_layer(AnimationLayer::base("a"));

        assert!(stack.remove_layer(5).is_none());
    }

    #[test]
    fn test_layer_stack_move_layer() {
        let mut stack = LayerStack::new(64);
        stack.add_layer(AnimationLayer::base("a"));
        stack.add_layer(AnimationLayer::base("b"));
        stack.add_layer(AnimationLayer::base("c"));

        stack.move_layer(0, 2);

        assert_eq!(stack.get_layer(0).unwrap().name, "b");
        assert_eq!(stack.get_layer(1).unwrap().name, "c");
        assert_eq!(stack.get_layer(2).unwrap().name, "a");
    }

    #[test]
    fn test_layer_stack_swap_layers() {
        let mut stack = LayerStack::new(64);
        stack.add_layer(AnimationLayer::base("a"));
        stack.add_layer(AnimationLayer::base("b"));

        stack.swap_layers(0, 1);

        assert_eq!(stack.get_layer(0).unwrap().name, "b");
        assert_eq!(stack.get_layer(1).unwrap().name, "a");
    }

    #[test]
    fn test_layer_stack_find_layer() {
        let mut stack = LayerStack::new(64);
        stack.add_layer(AnimationLayer::base("locomotion"));
        stack.add_layer(AnimationLayer::additive("breathing"));

        assert_eq!(stack.find_layer("locomotion"), Some(0));
        assert_eq!(stack.find_layer("breathing"), Some(1));
        assert_eq!(stack.find_layer("unknown"), None);
    }

    #[test]
    fn test_layer_stack_set_layer_weight() {
        let mut stack = LayerStack::new(64);
        stack.add_layer(AnimationLayer::base("test"));

        assert!(stack.set_layer_weight(0, 0.5));
        assert_eq!(stack.get_layer(0).unwrap().weight, 0.5);

        assert!(!stack.set_layer_weight(5, 0.5)); // Invalid index
    }

    #[test]
    fn test_layer_stack_set_layer_active() {
        let mut stack = LayerStack::new(64);
        stack.add_layer(AnimationLayer::base("test"));

        assert!(stack.set_layer_active(0, false));
        assert!(!stack.get_layer(0).unwrap().active);

        assert!(!stack.set_layer_active(5, false)); // Invalid index
    }

    #[test]
    fn test_layer_stack_base_layer() {
        let mut stack = LayerStack::new(64);
        assert!(stack.base_layer().is_none());

        stack.add_layer(AnimationLayer::base("base"));
        assert!(stack.base_layer().is_some());
        assert_eq!(stack.base_layer().unwrap().name, "base");
    }

    #[test]
    fn test_layer_stack_iterate_layers() {
        let mut stack = LayerStack::new(64);
        stack.add_layer(AnimationLayer::base("a"));
        stack.add_layer(AnimationLayer::base("b"));
        stack.add_layer(AnimationLayer::base("c").inactive());

        let names: Vec<_> = stack.layers().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b", "c"]);

        let active_names: Vec<_> = stack.active_layers().map(|l| l.name.as_str()).collect();
        assert_eq!(active_names, vec!["a", "b"]);
    }

    #[test]
    fn test_layer_stack_dirty_flags() {
        let mut stack = LayerStack::new(64);
        stack.add_layer(AnimationLayer::base("test"));

        assert!(stack.is_dirty());
        assert!(stack.is_layer_dirty(0));

        stack.clear_dirty();
        assert!(!stack.is_dirty());
        assert!(!stack.is_layer_dirty(0));

        stack.mark_dirty(0);
        assert!(stack.is_dirty());

        stack.clear_dirty();
        stack.mark_all_dirty();
        assert!(stack.is_dirty());
    }

    #[test]
    fn test_layer_stack_invalidate_caches() {
        let mut stack = LayerStack::new(64);
        stack.add_layer(AnimationLayer::base("test"));
        stack.set_layer_pose(0, Pose::new(64, PoseType::Current));

        assert!(stack.get_layer_pose(0).is_some());

        stack.invalidate_caches();
        assert!(stack.get_layer_pose(0).is_none());
    }

    // =========================================================================
    // Layer Evaluation Tests
    // =========================================================================

    #[test]
    fn test_layer_stack_evaluate_empty() {
        let mut stack = LayerStack::new(5);
        let result = stack.evaluate(&[]);
        assert_eq!(result.bone_count(), 5);
    }

    #[test]
    fn test_layer_stack_evaluate_single_layer() {
        let mut stack = LayerStack::new(2);
        stack.add_layer(AnimationLayer::base("base"));

        let mut pose = Pose::new(2, PoseType::Current);
        pose.positions[0] = Vec3::new(1.0, 0.0, 0.0);
        pose.rotations[0] = Quat::from_rotation_y(PI / 4.0);

        let result = stack.evaluate(&[pose]);

        assert_eq!(result.bone_count(), 2);
        assert!(result.positions[0].abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-6));
        assert!(result.rotations[0].abs_diff_eq(Quat::from_rotation_y(PI / 4.0), 1e-5));
    }

    #[test]
    fn test_layer_stack_evaluate_override_blend() {
        let mut stack = LayerStack::new(2);
        stack.add_layer(AnimationLayer::base("base"));
        stack.add_layer(AnimationLayer::base("overlay").with_weight(0.5));

        let mut base_pose = Pose::new(2, PoseType::Current);
        base_pose.positions[0] = Vec3::new(0.0, 0.0, 0.0);

        let mut overlay_pose = Pose::new(2, PoseType::Current);
        overlay_pose.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let result = stack.evaluate(&[base_pose, overlay_pose]);

        // Should be 50% blend: (0 + 10) * 0.5 = 5
        assert!(result.positions[0].abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_layer_stack_evaluate_additive_blend() {
        let mut stack = LayerStack::new(2);
        stack.add_layer(AnimationLayer::base("base"));
        stack.add_layer(AnimationLayer::additive("additive").with_weight(1.0));

        let mut base_pose = Pose::new(2, PoseType::Current);
        base_pose.positions[0] = Vec3::new(5.0, 0.0, 0.0);

        let mut additive_pose = Pose::new(2, PoseType::Current);
        additive_pose.positions[0] = Vec3::new(3.0, 0.0, 0.0);

        let result = stack.evaluate(&[base_pose, additive_pose]);

        // Should add: 5 + 3 = 8
        assert!(result.positions[0].abs_diff_eq(Vec3::new(8.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_layer_stack_evaluate_additive_partial_weight() {
        let mut stack = LayerStack::new(2);
        stack.add_layer(AnimationLayer::base("base"));
        stack.add_layer(AnimationLayer::additive("additive").with_weight(0.5));

        let mut base_pose = Pose::new(2, PoseType::Current);
        base_pose.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let mut additive_pose = Pose::new(2, PoseType::Current);
        additive_pose.positions[0] = Vec3::new(4.0, 0.0, 0.0);

        let result = stack.evaluate(&[base_pose, additive_pose]);

        // Should add partial: 10 + (4 * 0.5) = 12
        assert!(result.positions[0].abs_diff_eq(Vec3::new(12.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_layer_stack_evaluate_multiply_blend() {
        let mut stack = LayerStack::new(2);
        stack.add_layer(AnimationLayer::base("base"));
        stack.add_layer(AnimationLayer::multiply("mult").with_weight(1.0));

        let mut base_pose = Pose::new(2, PoseType::Current);
        base_pose.positions[0] = Vec3::new(2.0, 3.0, 4.0);
        base_pose.scales[0] = Vec3::new(1.0, 1.0, 1.0);

        let mut mult_pose = Pose::new(2, PoseType::Current);
        mult_pose.positions[0] = Vec3::new(2.0, 2.0, 2.0);
        mult_pose.scales[0] = Vec3::new(2.0, 2.0, 2.0);

        let result = stack.evaluate(&[base_pose, mult_pose]);

        // Positions multiplied: (2*2, 3*2, 4*2) = (4, 6, 8)
        assert!(result.positions[0].abs_diff_eq(Vec3::new(4.0, 6.0, 8.0), 1e-6));
        // Scales multiplied: (1*2, 1*2, 1*2) = (2, 2, 2)
        assert!(result.scales[0].abs_diff_eq(Vec3::new(2.0, 2.0, 2.0), 1e-6));
    }

    #[test]
    fn test_layer_stack_evaluate_with_mask() {
        let mut stack = LayerStack::new(4);

        // Base layer
        stack.add_layer(AnimationLayer::base("base"));

        // Masked overlay - only affects bones 0 and 1
        let mask = LayerMask::from_indices(&[0, 1], 4);
        stack.add_layer(
            AnimationLayer::base("overlay")
                .with_mask(mask)
                .with_weight(1.0),
        );

        let mut base_pose = Pose::new(4, PoseType::Current);
        for i in 0..4 {
            base_pose.positions[i] = Vec3::new(i as f32, 0.0, 0.0);
        }

        let mut overlay_pose = Pose::new(4, PoseType::Current);
        for i in 0..4 {
            overlay_pose.positions[i] = Vec3::new(100.0, 0.0, 0.0);
        }

        let result = stack.evaluate(&[base_pose, overlay_pose]);

        // Bones 0-1 should be from overlay (masked in)
        assert!(result.positions[0].abs_diff_eq(Vec3::new(100.0, 0.0, 0.0), 1e-6));
        assert!(result.positions[1].abs_diff_eq(Vec3::new(100.0, 0.0, 0.0), 1e-6));

        // Bones 2-3 should be from base (not masked)
        assert!(result.positions[2].abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 1e-6));
        assert!(result.positions[3].abs_diff_eq(Vec3::new(3.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_layer_stack_evaluate_inactive_layer() {
        let mut stack = LayerStack::new(2);
        stack.add_layer(AnimationLayer::base("base"));
        stack.add_layer(AnimationLayer::base("overlay").inactive());

        let mut base_pose = Pose::new(2, PoseType::Current);
        base_pose.positions[0] = Vec3::new(5.0, 0.0, 0.0);

        let mut overlay_pose = Pose::new(2, PoseType::Current);
        overlay_pose.positions[0] = Vec3::new(100.0, 0.0, 0.0);

        let result = stack.evaluate(&[base_pose, overlay_pose]);

        // Inactive overlay should be skipped
        assert!(result.positions[0].abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_layer_stack_evaluate_zero_weight_layer() {
        let mut stack = LayerStack::new(2);
        stack.add_layer(AnimationLayer::base("base"));
        stack.add_layer(AnimationLayer::base("overlay").with_weight(0.0));

        let mut base_pose = Pose::new(2, PoseType::Current);
        base_pose.positions[0] = Vec3::new(5.0, 0.0, 0.0);

        let mut overlay_pose = Pose::new(2, PoseType::Current);
        overlay_pose.positions[0] = Vec3::new(100.0, 0.0, 0.0);

        let result = stack.evaluate(&[base_pose, overlay_pose]);

        // Zero weight layer should be skipped
        assert!(result.positions[0].abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_layer_stack_evaluate_multi_layer() {
        let mut stack = LayerStack::new(2);
        stack.add_layer(AnimationLayer::base("base"));
        stack.add_layer(AnimationLayer::additive("add1").with_weight(1.0));
        stack.add_layer(AnimationLayer::additive("add2").with_weight(1.0));

        let mut base_pose = Pose::new(2, PoseType::Current);
        base_pose.positions[0] = Vec3::new(1.0, 0.0, 0.0);

        let mut add1_pose = Pose::new(2, PoseType::Current);
        add1_pose.positions[0] = Vec3::new(2.0, 0.0, 0.0);

        let mut add2_pose = Pose::new(2, PoseType::Current);
        add2_pose.positions[0] = Vec3::new(3.0, 0.0, 0.0);

        let result = stack.evaluate(&[base_pose, add1_pose, add2_pose]);

        // 1 + 2 + 3 = 6
        assert!(result.positions[0].abs_diff_eq(Vec3::new(6.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_layer_stack_evaluate_cached() {
        let mut stack = LayerStack::new(2);
        stack.add_layer(AnimationLayer::base("base"));
        stack.add_layer(AnimationLayer::additive("additive"));

        // Set cached poses
        let mut base_pose = Pose::new(2, PoseType::Current);
        base_pose.positions[0] = Vec3::new(5.0, 0.0, 0.0);
        stack.set_layer_pose(0, base_pose);

        let mut add_pose = Pose::new(2, PoseType::Current);
        add_pose.positions[0] = Vec3::new(3.0, 0.0, 0.0);
        stack.set_layer_pose(1, add_pose);

        let result = stack.evaluate_cached();

        // 5 + 3 = 8
        assert!(result.positions[0].abs_diff_eq(Vec3::new(8.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_layer_stack_frame_counter() {
        let mut stack = LayerStack::new(2);
        stack.add_layer(AnimationLayer::base("test"));

        assert_eq!(stack.frame(), 0);
        stack.evaluate(&[Pose::new(2, PoseType::Current)]);
        assert_eq!(stack.frame(), 1);
        stack.evaluate(&[Pose::new(2, PoseType::Current)]);
        assert_eq!(stack.frame(), 2);
    }

    #[test]
    fn test_layer_stack_set_bone_count() {
        let mut stack = LayerStack::new(32);
        let mask = LayerMask::full(32);
        stack.add_layer(AnimationLayer::base("test").with_mask(mask));

        stack.set_bone_count(64);

        assert_eq!(stack.bone_count(), 64);
        assert_eq!(stack.get_layer(0).unwrap().mask.as_ref().unwrap().bone_count(), 64);
    }

    // =========================================================================
    // LayerStackBuilder Tests
    // =========================================================================

    #[test]
    fn test_layer_stack_builder() {
        let stack = LayerStackBuilder::new(64)
            .name("character")
            .base_layer("locomotion")
            .additive_layer("breathing", 0.5)
            .masked_layer("upper_body", MaskPreset::UpperBody, 1.0)
            .build();

        assert_eq!(stack.name, Some("character".to_string()));
        assert_eq!(stack.bone_count(), 64);
        assert_eq!(stack.layer_count(), 3);

        assert_eq!(stack.get_layer(0).unwrap().name, "locomotion");
        assert_eq!(stack.get_layer(1).unwrap().name, "breathing");
        assert_eq!(stack.get_layer(1).unwrap().weight, 0.5);
        assert_eq!(stack.get_layer(2).unwrap().name, "upper_body");
        assert!(stack.get_layer(2).unwrap().mask.is_some());
    }

    #[test]
    fn test_layer_stack_builder_default() {
        let builder = LayerStackBuilder::default();
        let stack = builder.build();
        assert_eq!(stack.bone_count(), 64);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_empty_mask_evaluation() {
        let mut stack = LayerStack::new(4);
        stack.add_layer(AnimationLayer::base("base"));

        // Empty mask = no bones affected
        let empty_mask = LayerMask::new(4);
        stack.add_layer(
            AnimationLayer::base("overlay")
                .with_mask(empty_mask)
                .with_weight(1.0),
        );

        let mut base_pose = Pose::new(4, PoseType::Current);
        base_pose.positions[0] = Vec3::new(5.0, 0.0, 0.0);

        let mut overlay_pose = Pose::new(4, PoseType::Current);
        overlay_pose.positions[0] = Vec3::new(100.0, 0.0, 0.0);

        let result = stack.evaluate(&[base_pose, overlay_pose]);

        // Empty mask means no effect from overlay
        assert!(result.positions[0].abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_single_bone_mask() {
        let mut stack = LayerStack::new(4);
        stack.add_layer(AnimationLayer::base("base"));

        let mask = LayerMask::from_indices(&[1], 4);
        stack.add_layer(
            AnimationLayer::base("overlay")
                .with_mask(mask)
                .with_weight(1.0),
        );

        let base_pose = Pose::new(4, PoseType::Current);

        let mut overlay_pose = Pose::new(4, PoseType::Current);
        overlay_pose.positions[1] = Vec3::new(10.0, 0.0, 0.0);

        let result = stack.evaluate(&[base_pose, overlay_pose]);

        // Only bone 1 should be affected
        assert!(result.positions[0].abs_diff_eq(Vec3::ZERO, 1e-6));
        assert!(result.positions[1].abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), 1e-6));
        assert!(result.positions[2].abs_diff_eq(Vec3::ZERO, 1e-6));
    }

    #[test]
    fn test_layer_ordering_matters() {
        let mut stack = LayerStack::new(2);

        // Order: base -> additive
        stack.add_layer(AnimationLayer::base("base"));
        stack.add_layer(AnimationLayer::additive("additive"));

        let mut base_pose = Pose::new(2, PoseType::Current);
        base_pose.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let mut add_pose = Pose::new(2, PoseType::Current);
        add_pose.positions[0] = Vec3::new(5.0, 0.0, 0.0);

        let result = stack.evaluate(&[base_pose.clone(), add_pose.clone()]);
        assert!(result.positions[0].abs_diff_eq(Vec3::new(15.0, 0.0, 0.0), 1e-6));

        // Swap order: additive -> base
        let mut stack2 = LayerStack::new(2);
        stack2.add_layer(AnimationLayer::additive("additive"));
        stack2.add_layer(AnimationLayer::base("base"));

        // Now base layer will override after additive, which does nothing initially
        let result2 = stack2.evaluate(&[add_pose, base_pose]);
        // First layer is additive but there's no base yet, so it starts from identity
        // Then base layer overrides
        assert!(result2.positions[0].abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_rotation_blending() {
        let mut stack = LayerStack::new(1);
        stack.add_layer(AnimationLayer::base("base"));
        stack.add_layer(AnimationLayer::base("overlay").with_weight(0.5));

        let mut base_pose = Pose::new(1, PoseType::Current);
        base_pose.rotations[0] = Quat::IDENTITY;

        let mut overlay_pose = Pose::new(1, PoseType::Current);
        overlay_pose.rotations[0] = Quat::from_rotation_y(PI / 2.0);

        let result = stack.evaluate(&[base_pose, overlay_pose]);

        // Should be halfway rotation
        let expected = Quat::from_rotation_y(PI / 4.0);
        assert!(result.rotations[0].abs_diff_eq(expected, 1e-5));
    }

    #[test]
    fn test_additive_rotation() {
        let mut stack = LayerStack::new(1);
        stack.add_layer(AnimationLayer::base("base"));
        stack.add_layer(AnimationLayer::additive("additive"));

        let mut base_pose = Pose::new(1, PoseType::Current);
        base_pose.rotations[0] = Quat::from_rotation_y(PI / 4.0);

        let mut add_pose = Pose::new(1, PoseType::Current);
        add_pose.rotations[0] = Quat::from_rotation_y(PI / 4.0);

        let result = stack.evaluate(&[base_pose, add_pose]);

        // Additive rotation: base * additive = PI/4 + PI/4 = PI/2
        let expected = Quat::from_rotation_y(PI / 2.0);
        assert!(result.rotations[0].abs_diff_eq(expected, 1e-5));
    }

    #[test]
    fn test_missing_layer_poses() {
        let mut stack = LayerStack::new(2);
        stack.add_layer(AnimationLayer::base("base"));
        stack.add_layer(AnimationLayer::additive("additive"));
        stack.add_layer(AnimationLayer::additive("another"));

        // Only provide poses for first two layers
        let base_pose = Pose::new(2, PoseType::Current);
        let add_pose = Pose::new(2, PoseType::Current);

        // Should not panic, just skip missing layer
        let result = stack.evaluate(&[base_pose, add_pose]);
        assert_eq!(result.bone_count(), 2);
    }

    #[test]
    fn test_serialization_layer_mask() {
        let mask = LayerMask::from_preset(MaskPreset::UpperBody, 32);
        let json = serde_json::to_string(&mask).unwrap();
        let recovered: LayerMask = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered.bone_count(), mask.bone_count());
        assert_eq!(recovered.preset, mask.preset);
        for i in 0..32 {
            assert_eq!(recovered.get_weight(i), mask.get_weight(i));
        }
    }

    #[test]
    fn test_serialization_animation_layer() {
        let layer = AnimationLayer::base("test")
            .with_weight(0.75)
            .with_state_machine(5)
            .with_sync_group("locomotion");

        let json = serde_json::to_string(&layer).unwrap();
        let recovered: AnimationLayer = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered.name, "test");
        assert_eq!(recovered.weight, 0.75);
        assert_eq!(recovered.source, LayerSource::StateMachine(5));
        assert_eq!(recovered.sync_group, Some("locomotion".to_string()));
    }
}
