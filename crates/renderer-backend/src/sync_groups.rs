//! Animation synchronization groups for runtime animation control (T-AN-5.6).
//!
//! This module provides animation sync groups for coordinating multiple animations:
//!
//! - **SyncGroup**: Container for synchronized state machines/layers
//! - **SyncMarker**: Named timing markers within animations (e.g., foot contacts)
//! - **SyncTrack**: Collection of markers for a single animation clip
//! - **Phase synchronization**: Align animations at sync markers
//! - **SyncGroupManager**: Manages multiple sync groups
//! - **Foot sync**: Special handling for walk/run cycle blending
//!
//! # Architecture
//!
//! ```text
//! SyncGroupManager
//! +-- groups: HashMap<SyncGroupId, SyncGroup>
//! |   +-- SyncGroup
//! |       +-- id: SyncGroupId
//! |       +-- name: String
//! |       +-- members: Vec<SyncMember>
//! |       +-- master: Option<usize>
//! |       +-- sync_mode: SyncGroupMode
//! |
//! +-- tracks: HashMap<ClipId, SyncTrack>
//!     +-- SyncTrack
//!         +-- clip_id: usize
//!         +-- markers: Vec<SyncMarker>
//!             +-- name: String
//!             +-- normalized_time: f32
//!             +-- marker_type: MarkerType
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::sync_groups::{
//!     SyncGroupManager, SyncGroup, SyncMarker, SyncTrack,
//!     SyncGroupMode, MarkerType,
//! };
//!
//! // Create a sync group manager
//! let mut manager = SyncGroupManager::new();
//!
//! // Create sync tracks with foot markers for animations
//! let mut walk_track = SyncTrack::new(0); // clip index 0 = walk
//! walk_track.add_marker(SyncMarker::foot_contact("left_foot_down", 0.0));
//! walk_track.add_marker(SyncMarker::foot_contact("right_foot_down", 0.5));
//! manager.add_track(walk_track);
//!
//! let mut run_track = SyncTrack::new(1); // clip index 1 = run
//! run_track.add_marker(SyncMarker::foot_contact("left_foot_down", 0.0));
//! run_track.add_marker(SyncMarker::foot_contact("right_foot_down", 0.5));
//! manager.add_track(run_track);
//!
//! // Create a sync group for locomotion
//! let locomotion_group = SyncGroup::new("locomotion")
//!     .with_mode(SyncGroupMode::PhaseLocked);
//! let group_id = manager.add_group(locomotion_group);
//!
//! // Register layers/state machines with the group
//! manager.register_member(group_id, 0); // layer 0
//! manager.register_member(group_id, 1); // layer 1
//!
//! // Update each frame
//! manager.update(0.016);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::TAU;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of sync groups per manager.
pub const MAX_SYNC_GROUPS: usize = 64;

/// Maximum number of members per sync group.
pub const MAX_MEMBERS_PER_GROUP: usize = 16;

/// Maximum number of markers per sync track.
pub const MAX_MARKERS_PER_TRACK: usize = 64;

/// Default phase correction rate (per second).
pub const DEFAULT_PHASE_CORRECTION_RATE: f32 = 5.0;

/// Epsilon for phase comparisons.
pub const PHASE_EPSILON: f32 = 1e-6;

/// Threshold for considering phases "close enough".
pub const PHASE_SNAP_THRESHOLD: f32 = 0.01;

// ---------------------------------------------------------------------------
// SyncGroupId
// ---------------------------------------------------------------------------

/// Unique identifier for a sync group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SyncGroupId(pub u32);

impl SyncGroupId {
    /// Create a new sync group ID.
    #[inline]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    #[inline]
    pub fn value(&self) -> u32 {
        self.0
    }

    /// Invalid/null group ID.
    pub const INVALID: SyncGroupId = SyncGroupId(u32::MAX);
}

impl Default for SyncGroupId {
    fn default() -> Self {
        Self::INVALID
    }
}

// ---------------------------------------------------------------------------
// MarkerType
// ---------------------------------------------------------------------------

/// Type of sync marker for semantic categorization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarkerType {
    /// Generic sync marker.
    #[default]
    Generic,

    /// Foot contact marker (for foot sync).
    FootContact,

    /// Foot lift marker (foot leaving ground).
    FootLift,

    /// Beat marker (for music sync).
    Beat,

    /// Event marker (triggers gameplay events).
    Event,

    /// Loop point marker.
    LoopPoint,
}

impl MarkerType {
    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Generic => "Generic",
            Self::FootContact => "Foot Contact",
            Self::FootLift => "Foot Lift",
            Self::Beat => "Beat",
            Self::Event => "Event",
            Self::LoopPoint => "Loop Point",
        }
    }

    /// Check if this is a foot-related marker.
    #[inline]
    pub fn is_foot_marker(&self) -> bool {
        matches!(self, Self::FootContact | Self::FootLift)
    }
}

// ---------------------------------------------------------------------------
// SyncMarker
// ---------------------------------------------------------------------------

/// A synchronization marker within an animation.
///
/// Markers define key timing points in animations that should be aligned
/// when blending between different clips.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyncMarker {
    /// Marker name (e.g., "left_foot_down", "right_foot_down").
    pub name: String,

    /// Normalized time position (0.0 - 1.0).
    pub normalized_time: f32,

    /// Type of marker for semantic categorization.
    pub marker_type: MarkerType,

    /// Optional bone index associated with this marker.
    pub bone_index: Option<usize>,

    /// Optional user data for custom extensions.
    pub user_data: Option<i32>,
}

impl SyncMarker {
    /// Create a new sync marker.
    ///
    /// # Arguments
    ///
    /// * `name` - Marker name for identification
    /// * `normalized_time` - Position in the clip (0.0 - 1.0)
    pub fn new(name: impl Into<String>, normalized_time: f32) -> Self {
        Self {
            name: name.into(),
            normalized_time: normalized_time.clamp(0.0, 1.0),
            marker_type: MarkerType::Generic,
            bone_index: None,
            user_data: None,
        }
    }

    /// Create a foot contact marker.
    pub fn foot_contact(name: impl Into<String>, normalized_time: f32) -> Self {
        Self {
            name: name.into(),
            normalized_time: normalized_time.clamp(0.0, 1.0),
            marker_type: MarkerType::FootContact,
            bone_index: None,
            user_data: None,
        }
    }

    /// Create a foot lift marker.
    pub fn foot_lift(name: impl Into<String>, normalized_time: f32) -> Self {
        Self {
            name: name.into(),
            normalized_time: normalized_time.clamp(0.0, 1.0),
            marker_type: MarkerType::FootLift,
            bone_index: None,
            user_data: None,
        }
    }

    /// Create a beat marker.
    pub fn beat(name: impl Into<String>, normalized_time: f32) -> Self {
        Self {
            name: name.into(),
            normalized_time: normalized_time.clamp(0.0, 1.0),
            marker_type: MarkerType::Beat,
            bone_index: None,
            user_data: None,
        }
    }

    /// Create an event marker.
    pub fn event(name: impl Into<String>, normalized_time: f32) -> Self {
        Self {
            name: name.into(),
            normalized_time: normalized_time.clamp(0.0, 1.0),
            marker_type: MarkerType::Event,
            bone_index: None,
            user_data: None,
        }
    }

    /// Set the marker type.
    pub fn with_type(mut self, marker_type: MarkerType) -> Self {
        self.marker_type = marker_type;
        self
    }

    /// Set the bone index.
    pub fn with_bone(mut self, bone_index: usize) -> Self {
        self.bone_index = Some(bone_index);
        self
    }

    /// Set user data.
    pub fn with_user_data(mut self, data: i32) -> Self {
        self.user_data = Some(data);
        self
    }
}

impl Default for SyncMarker {
    fn default() -> Self {
        Self::new("default", 0.0)
    }
}

// ---------------------------------------------------------------------------
// SyncTrack
// ---------------------------------------------------------------------------

/// A collection of sync markers for a single animation clip.
///
/// Tracks store all timing markers for an animation, enabling phase
/// synchronization with other animations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyncTrack {
    /// Index of the animation clip this track belongs to.
    pub clip_id: usize,

    /// Sync markers sorted by normalized time.
    pub markers: Vec<SyncMarker>,

    /// Clip duration in seconds (for time <-> normalized conversions).
    pub duration: f32,

    /// Whether markers are currently sorted.
    sorted: bool,

    /// Optional name for debugging.
    pub name: Option<String>,
}

impl SyncTrack {
    /// Create a new sync track for a clip.
    ///
    /// # Arguments
    ///
    /// * `clip_id` - Index of the animation clip
    pub fn new(clip_id: usize) -> Self {
        Self {
            clip_id,
            markers: Vec::new(),
            duration: 1.0,
            sorted: true,
            name: None,
        }
    }

    /// Create a sync track with preallocated capacity.
    pub fn with_capacity(clip_id: usize, capacity: usize) -> Self {
        Self {
            clip_id,
            markers: Vec::with_capacity(capacity),
            duration: 1.0,
            sorted: true,
            name: None,
        }
    }

    /// Set the clip duration.
    pub fn with_duration(mut self, duration: f32) -> Self {
        self.duration = duration.max(0.001);
        self
    }

    /// Set the track name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add a marker to the track.
    pub fn add_marker(&mut self, marker: SyncMarker) {
        self.markers.push(marker);
        self.sorted = false;
    }

    /// Remove a marker by name.
    ///
    /// Returns true if a marker was removed.
    pub fn remove_marker(&mut self, name: &str) -> bool {
        let initial_len = self.markers.len();
        self.markers.retain(|m| m.name != name);
        self.markers.len() < initial_len
    }

    /// Ensure markers are sorted by normalized time.
    pub fn sort_markers(&mut self) {
        if !self.sorted {
            self.markers
                .sort_by(|a, b| a.normalized_time.partial_cmp(&b.normalized_time).unwrap());
            self.sorted = true;
        }
    }

    /// Get the number of markers.
    #[inline]
    pub fn marker_count(&self) -> usize {
        self.markers.len()
    }

    /// Check if the track has markers.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.markers.is_empty()
    }

    /// Find a marker by name.
    pub fn find_marker(&self, name: &str) -> Option<&SyncMarker> {
        self.markers.iter().find(|m| m.name == name)
    }

    /// Find markers by type.
    pub fn find_markers_by_type(&self, marker_type: MarkerType) -> Vec<&SyncMarker> {
        self.markers
            .iter()
            .filter(|m| m.marker_type == marker_type)
            .collect()
    }

    /// Get foot contact markers only.
    pub fn foot_contacts(&self) -> Vec<&SyncMarker> {
        self.find_markers_by_type(MarkerType::FootContact)
    }

    /// Get the nearest marker before a given normalized time.
    pub fn marker_before(&mut self, normalized_time: f32) -> Option<&SyncMarker> {
        self.sort_markers();
        self.markers
            .iter()
            .rev()
            .find(|m| m.normalized_time <= normalized_time)
    }

    /// Get the nearest marker after a given normalized time.
    pub fn marker_after(&mut self, normalized_time: f32) -> Option<&SyncMarker> {
        self.sort_markers();
        self.markers
            .iter()
            .find(|m| m.normalized_time > normalized_time)
    }

    /// Get markers surrounding a normalized time.
    ///
    /// Returns (previous_marker, next_marker) where either can be None.
    pub fn surrounding_markers(
        &mut self,
        normalized_time: f32,
    ) -> (Option<&SyncMarker>, Option<&SyncMarker>) {
        self.sort_markers();

        let before = self
            .markers
            .iter()
            .rev()
            .find(|m| m.normalized_time <= normalized_time);
        let after = self
            .markers
            .iter()
            .find(|m| m.normalized_time > normalized_time);

        (before, after)
    }

    /// Interpolate marker position between two named markers.
    ///
    /// Returns the interpolated normalized time.
    pub fn interpolate_between(
        &self,
        from_marker: &str,
        to_marker: &str,
        t: f32,
    ) -> Option<f32> {
        let from = self.find_marker(from_marker)?;
        let to = self.find_marker(to_marker)?;

        Some(from.normalized_time + (to.normalized_time - from.normalized_time) * t.clamp(0.0, 1.0))
    }

    /// Convert normalized time to absolute time in seconds.
    #[inline]
    pub fn to_absolute_time(&self, normalized_time: f32) -> f32 {
        normalized_time * self.duration
    }

    /// Convert absolute time to normalized time.
    #[inline]
    pub fn to_normalized_time(&self, absolute_time: f32) -> f32 {
        if self.duration > PHASE_EPSILON {
            absolute_time / self.duration
        } else {
            0.0
        }
    }

    /// Auto-detect markers from animation events.
    ///
    /// This is a placeholder for integration with the animation event system.
    pub fn auto_detect_from_events(&mut self, events: &[(f32, String)]) {
        for (time, name) in events {
            let normalized = self.to_normalized_time(*time);
            let marker_type = if name.contains("foot") || name.contains("step") {
                if name.contains("down") || name.contains("contact") {
                    MarkerType::FootContact
                } else if name.contains("up") || name.contains("lift") {
                    MarkerType::FootLift
                } else {
                    MarkerType::Generic
                }
            } else if name.contains("beat") {
                MarkerType::Beat
            } else {
                MarkerType::Event
            };

            self.add_marker(SyncMarker {
                name: name.clone(),
                normalized_time: normalized,
                marker_type,
                bone_index: None,
                user_data: None,
            });
        }
    }
}

impl Default for SyncTrack {
    fn default() -> Self {
        Self::new(0)
    }
}

// ---------------------------------------------------------------------------
// SyncGroupMode
// ---------------------------------------------------------------------------

/// Synchronization mode for a sync group.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SyncGroupMode {
    /// Leader mode: one member sets the pace, others follow.
    #[default]
    Leader,

    /// Follower mode: all members follow an external time source.
    Follower,

    /// Phase-locked mode: all members maintain the same phase.
    PhaseLocked,

    /// Marker-aligned mode: members align at matching markers.
    MarkerAligned,

    /// Independent mode: no synchronization (group just tracks phase).
    Independent,
}

impl SyncGroupMode {
    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Leader => "Leader",
            Self::Follower => "Follower",
            Self::PhaseLocked => "Phase-locked",
            Self::MarkerAligned => "Marker-aligned",
            Self::Independent => "Independent",
        }
    }

    /// Check if this mode requires a master member.
    #[inline]
    pub fn requires_master(&self) -> bool {
        matches!(self, Self::Leader)
    }

    /// Check if this mode synchronizes phase.
    #[inline]
    pub fn synchronizes_phase(&self) -> bool {
        !matches!(self, Self::Independent)
    }
}

// ---------------------------------------------------------------------------
// SyncMember
// ---------------------------------------------------------------------------

/// A member of a sync group (state machine, layer, or custom).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyncMember {
    /// Unique ID within the group.
    pub id: usize,

    /// Current phase (0.0 - 1.0).
    pub phase: f32,

    /// Phase velocity (change per second).
    pub phase_velocity: f32,

    /// Current clip index being played.
    pub current_clip: Option<usize>,

    /// Weight for blending (0.0 - 1.0).
    pub weight: f32,

    /// Whether this member is active.
    pub active: bool,

    /// Target phase for correction.
    target_phase: f32,

    /// Phase correction amount per second.
    correction_rate: f32,

    /// Optional name for debugging.
    pub name: Option<String>,
}

impl SyncMember {
    /// Create a new sync member.
    pub fn new(id: usize) -> Self {
        Self {
            id,
            phase: 0.0,
            phase_velocity: 1.0,
            current_clip: None,
            weight: 1.0,
            active: true,
            target_phase: 0.0,
            correction_rate: DEFAULT_PHASE_CORRECTION_RATE,
            name: None,
        }
    }

    /// Set the member name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the phase correction rate.
    pub fn with_correction_rate(mut self, rate: f32) -> Self {
        self.correction_rate = rate.max(0.0);
        self
    }

    /// Set the current clip.
    pub fn with_clip(mut self, clip: usize) -> Self {
        self.current_clip = Some(clip);
        self
    }

    /// Set the phase.
    pub fn set_phase(&mut self, phase: f32) {
        self.phase = wrap_phase(phase);
        self.target_phase = self.phase;
    }

    /// Set the target phase for smooth correction.
    pub fn set_target_phase(&mut self, target: f32) {
        self.target_phase = wrap_phase(target);
    }

    /// Update phase toward target with smooth correction.
    ///
    /// Returns the phase delta applied this frame.
    pub fn update_phase_correction(&mut self, dt: f32) -> f32 {
        let phase_diff = phase_difference(self.phase, self.target_phase);

        if phase_diff.abs() < PHASE_SNAP_THRESHOLD {
            // Close enough, snap to target
            let delta = phase_diff;
            self.phase = self.target_phase;
            delta
        } else {
            // Smooth correction
            let max_correction = self.correction_rate * dt;
            let correction = phase_diff.clamp(-max_correction, max_correction);
            self.phase = wrap_phase(self.phase + correction);
            correction
        }
    }

    /// Advance phase by delta time.
    pub fn advance_phase(&mut self, dt: f32) {
        self.phase = wrap_phase(self.phase + self.phase_velocity * dt);
    }
}

impl Default for SyncMember {
    fn default() -> Self {
        Self::new(0)
    }
}

// ---------------------------------------------------------------------------
// SyncGroup
// ---------------------------------------------------------------------------

/// A group of synchronized animation sources.
///
/// Sync groups coordinate the phase of multiple state machines, layers,
/// or other animation sources to maintain consistent timing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyncGroup {
    /// Unique group ID.
    pub id: SyncGroupId,

    /// Human-readable name.
    pub name: String,

    /// Group members.
    pub members: Vec<SyncMember>,

    /// Index of the master member (if any).
    pub master: Option<usize>,

    /// Synchronization mode.
    pub sync_mode: SyncGroupMode,

    /// Global phase for the group (0.0 - 1.0).
    pub group_phase: f32,

    /// Global phase velocity.
    pub group_phase_velocity: f32,

    /// Phase correction rate for members.
    pub correction_rate: f32,

    /// Whether the group is enabled.
    pub enabled: bool,

    /// Marker name to use for alignment (if MarkerAligned mode).
    pub alignment_marker: Option<String>,
}

impl SyncGroup {
    /// Create a new sync group.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: SyncGroupId::INVALID,
            name: name.into(),
            members: Vec::new(),
            master: None,
            sync_mode: SyncGroupMode::Leader,
            group_phase: 0.0,
            group_phase_velocity: 1.0,
            correction_rate: DEFAULT_PHASE_CORRECTION_RATE,
            enabled: true,
            alignment_marker: None,
        }
    }

    /// Set the sync mode.
    pub fn with_mode(mut self, mode: SyncGroupMode) -> Self {
        self.sync_mode = mode;
        self
    }

    /// Set the correction rate.
    pub fn with_correction_rate(mut self, rate: f32) -> Self {
        self.correction_rate = rate.max(0.0);
        self
    }

    /// Set the alignment marker (for MarkerAligned mode).
    pub fn with_alignment_marker(mut self, marker: impl Into<String>) -> Self {
        self.alignment_marker = Some(marker.into());
        self
    }

    /// Disable the group.
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Add a member to the group.
    ///
    /// Returns the index of the added member.
    pub fn add_member(&mut self, member: SyncMember) -> usize {
        let index = self.members.len();
        self.members.push(member);

        // First member becomes master by default in Leader mode
        if self.master.is_none() && self.sync_mode == SyncGroupMode::Leader {
            self.master = Some(index);
        }

        index
    }

    /// Remove a member by ID.
    ///
    /// Returns true if a member was removed.
    pub fn remove_member(&mut self, member_id: usize) -> bool {
        let initial_len = self.members.len();
        self.members.retain(|m| m.id != member_id);

        // Update master if needed
        if let Some(master_idx) = self.master {
            if master_idx >= self.members.len() {
                self.master = if self.members.is_empty() {
                    None
                } else {
                    Some(0)
                };
            }
        }

        self.members.len() < initial_len
    }

    /// Get a member by index.
    pub fn get_member(&self, index: usize) -> Option<&SyncMember> {
        self.members.get(index)
    }

    /// Get a mutable member by index.
    pub fn get_member_mut(&mut self, index: usize) -> Option<&mut SyncMember> {
        self.members.get_mut(index)
    }

    /// Find a member by ID.
    pub fn find_member(&self, member_id: usize) -> Option<&SyncMember> {
        self.members.iter().find(|m| m.id == member_id)
    }

    /// Find a member index by ID.
    pub fn find_member_index(&self, member_id: usize) -> Option<usize> {
        self.members.iter().position(|m| m.id == member_id)
    }

    /// Get the master member if any.
    pub fn get_master(&self) -> Option<&SyncMember> {
        self.master.and_then(|idx| self.members.get(idx))
    }

    /// Set the master member by index.
    pub fn set_master(&mut self, index: usize) -> bool {
        if index < self.members.len() {
            self.master = Some(index);
            true
        } else {
            false
        }
    }

    /// Get the number of members.
    #[inline]
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Check if the group is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Get active member count.
    pub fn active_member_count(&self) -> usize {
        self.members.iter().filter(|m| m.active).count()
    }

    /// Update the group phase and synchronize members.
    pub fn update(&mut self, dt: f32) {
        if !self.enabled || self.is_empty() {
            return;
        }

        match self.sync_mode {
            SyncGroupMode::Leader => self.update_leader_mode(dt),
            SyncGroupMode::Follower => self.update_follower_mode(dt),
            SyncGroupMode::PhaseLocked => self.update_phase_locked_mode(dt),
            SyncGroupMode::MarkerAligned => self.update_marker_aligned_mode(dt),
            SyncGroupMode::Independent => self.update_independent_mode(dt),
        }
    }

    /// Leader mode: master advances, followers sync to master.
    fn update_leader_mode(&mut self, dt: f32) {
        // Get master phase
        let master_phase = if let Some(master_idx) = self.master {
            if let Some(master) = self.members.get_mut(master_idx) {
                master.advance_phase(dt);
                master.phase
            } else {
                return;
            }
        } else {
            // No master - advance group phase
            self.group_phase = wrap_phase(self.group_phase + self.group_phase_velocity * dt);
            self.group_phase
        };

        // Sync followers to master
        for (i, member) in self.members.iter_mut().enumerate() {
            if !member.active {
                continue;
            }
            if Some(i) != self.master {
                member.set_target_phase(master_phase);
                member.update_phase_correction(dt);
            }
        }

        self.group_phase = master_phase;
    }

    /// Follower mode: all members follow external time.
    fn update_follower_mode(&mut self, dt: f32) {
        // Group phase is set externally, sync all members
        for member in &mut self.members {
            if member.active {
                member.set_target_phase(self.group_phase);
                member.update_phase_correction(dt);
            }
        }
    }

    /// Phase-locked mode: all members maintain the same phase.
    fn update_phase_locked_mode(&mut self, dt: f32) {
        // Advance group phase
        self.group_phase = wrap_phase(self.group_phase + self.group_phase_velocity * dt);

        // Sync all members to group phase
        for member in &mut self.members {
            if member.active {
                member.phase = self.group_phase;
                member.target_phase = self.group_phase;
            }
        }
    }

    /// Marker-aligned mode: align members at matching markers.
    fn update_marker_aligned_mode(&mut self, dt: f32) {
        // Advance group phase
        self.group_phase = wrap_phase(self.group_phase + self.group_phase_velocity * dt);

        // Members advance independently but correct toward marker alignment
        for member in &mut self.members {
            if member.active {
                member.advance_phase(dt);
                // Marker alignment would be handled by the manager with track info
            }
        }
    }

    /// Independent mode: members advance independently.
    fn update_independent_mode(&mut self, dt: f32) {
        // Advance group phase for reference
        self.group_phase = wrap_phase(self.group_phase + self.group_phase_velocity * dt);

        // Each member advances independently
        for member in &mut self.members {
            if member.active {
                member.advance_phase(dt);
            }
        }
    }

    /// Set the group phase and sync all members.
    pub fn set_phase(&mut self, phase: f32) {
        self.group_phase = wrap_phase(phase);
        for member in &mut self.members {
            member.set_phase(self.group_phase);
        }
    }

    /// Set the group phase velocity.
    pub fn set_phase_velocity(&mut self, velocity: f32) {
        self.group_phase_velocity = velocity;
        for member in &mut self.members {
            member.phase_velocity = velocity;
        }
    }

    /// Compute weighted average phase across active members.
    pub fn weighted_average_phase(&self) -> f32 {
        let mut total_weight = 0.0;
        let mut weighted_sum_x = 0.0;
        let mut weighted_sum_y = 0.0;

        for member in &self.members {
            if !member.active || member.weight <= 0.0 {
                continue;
            }

            // Use circular averaging to handle wrap-around
            let angle = member.phase * TAU;
            weighted_sum_x += member.weight * angle.cos();
            weighted_sum_y += member.weight * angle.sin();
            total_weight += member.weight;
        }

        if total_weight < PHASE_EPSILON {
            return 0.0;
        }

        // Convert back to phase
        let avg_angle = (weighted_sum_y / total_weight).atan2(weighted_sum_x / total_weight);
        wrap_phase(avg_angle / TAU)
    }

    /// Reset the group to initial state.
    pub fn reset(&mut self) {
        self.group_phase = 0.0;
        for member in &mut self.members {
            member.phase = 0.0;
            member.target_phase = 0.0;
        }
    }
}

impl Default for SyncGroup {
    fn default() -> Self {
        Self::new("default")
    }
}

// ---------------------------------------------------------------------------
// FootSyncState
// ---------------------------------------------------------------------------

/// State for foot synchronization tracking.
///
/// Tracks gait phase for blending between walk/run while maintaining
/// consistent foot timing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FootSyncState {
    /// Current gait phase (0.0 - 1.0, where 0.0 = left foot down, 0.5 = right foot down).
    pub gait_phase: f32,

    /// Phase velocity (gait cycles per second).
    pub gait_velocity: f32,

    /// Left foot phase (0.0 = fully down, 0.5 = fully up, 1.0 = fully down again).
    pub left_foot_phase: f32,

    /// Right foot phase.
    pub right_foot_phase: f32,

    /// Whether left foot is currently planted.
    pub left_planted: bool,

    /// Whether right foot is currently planted.
    pub right_planted: bool,

    /// Stride length (for synchronizing different walk speeds).
    pub stride_length: f32,
}

impl FootSyncState {
    /// Create a new foot sync state.
    pub fn new() -> Self {
        Self {
            gait_phase: 0.0,
            gait_velocity: 1.0,
            left_foot_phase: 0.0,
            right_foot_phase: 0.5, // Right foot is offset by half cycle
            left_planted: true,
            right_planted: false,
            stride_length: 1.0,
        }
    }

    /// Create from a specific gait phase.
    pub fn from_phase(phase: f32) -> Self {
        let wrapped = wrap_phase(phase);
        Self {
            gait_phase: wrapped,
            gait_velocity: 1.0,
            left_foot_phase: wrapped,
            right_foot_phase: wrap_phase(wrapped + 0.5),
            left_planted: wrapped < 0.25 || wrapped > 0.75,
            right_planted: wrapped >= 0.25 && wrapped <= 0.75,
            stride_length: 1.0,
        }
    }

    /// Update the gait phase.
    pub fn update(&mut self, dt: f32) {
        self.gait_phase = wrap_phase(self.gait_phase + self.gait_velocity * dt);

        // Update individual foot phases
        self.left_foot_phase = self.gait_phase;
        self.right_foot_phase = wrap_phase(self.gait_phase + 0.5);

        // Update planted states (foot is planted in first and last quarter of its cycle)
        self.left_planted = self.left_foot_phase < 0.25 || self.left_foot_phase > 0.75;
        self.right_planted = self.right_foot_phase < 0.25 || self.right_foot_phase > 0.75;
    }

    /// Set the gait phase directly.
    pub fn set_phase(&mut self, phase: f32) {
        self.gait_phase = wrap_phase(phase);
        self.left_foot_phase = self.gait_phase;
        self.right_foot_phase = wrap_phase(self.gait_phase + 0.5);
        self.update_planted_states();
    }

    /// Update planted states based on current phases.
    fn update_planted_states(&mut self) {
        self.left_planted = self.left_foot_phase < 0.25 || self.left_foot_phase > 0.75;
        self.right_planted = self.right_foot_phase < 0.25 || self.right_foot_phase > 0.75;
    }

    /// Check if both feet are planted (double support phase).
    #[inline]
    pub fn is_double_support(&self) -> bool {
        self.left_planted && self.right_planted
    }

    /// Check if exactly one foot is planted (single support phase).
    #[inline]
    pub fn is_single_support(&self) -> bool {
        self.left_planted != self.right_planted
    }

    /// Get the planted foot (0 = left, 1 = right, -1 = both or neither).
    pub fn planted_foot(&self) -> i32 {
        match (self.left_planted, self.right_planted) {
            (true, false) => 0,
            (false, true) => 1,
            _ => -1,
        }
    }

    /// Blend between two foot sync states.
    pub fn blend(&self, other: &FootSyncState, t: f32) -> FootSyncState {
        let t = t.clamp(0.0, 1.0);

        // Use circular interpolation for phases
        let gait_phase = circular_lerp(self.gait_phase, other.gait_phase, t);

        FootSyncState {
            gait_phase,
            gait_velocity: self.gait_velocity + (other.gait_velocity - self.gait_velocity) * t,
            left_foot_phase: circular_lerp(self.left_foot_phase, other.left_foot_phase, t),
            right_foot_phase: circular_lerp(self.right_foot_phase, other.right_foot_phase, t),
            left_planted: if t < 0.5 {
                self.left_planted
            } else {
                other.left_planted
            },
            right_planted: if t < 0.5 {
                self.right_planted
            } else {
                other.right_planted
            },
            stride_length: self.stride_length + (other.stride_length - self.stride_length) * t,
        }
    }
}

impl Default for FootSyncState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SyncGroupManager
// ---------------------------------------------------------------------------

/// Manager for multiple sync groups.
///
/// Provides centralized management of sync groups and sync tracks,
/// handling creation, updates, and queries.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SyncGroupManager {
    /// All sync groups indexed by ID.
    pub groups: HashMap<SyncGroupId, SyncGroup>,

    /// Sync tracks for animation clips.
    pub tracks: HashMap<usize, SyncTrack>,

    /// Next available group ID.
    next_group_id: u32,

    /// Whether manager is enabled.
    pub enabled: bool,
}

impl SyncGroupManager {
    /// Create a new sync group manager.
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
            tracks: HashMap::new(),
            next_group_id: 0,
            enabled: true,
        }
    }

    /// Create with preallocated capacity.
    pub fn with_capacity(groups_capacity: usize, tracks_capacity: usize) -> Self {
        Self {
            groups: HashMap::with_capacity(groups_capacity),
            tracks: HashMap::with_capacity(tracks_capacity),
            next_group_id: 0,
            enabled: true,
        }
    }

    // -------------------------------------------------------------------------
    // Group Management
    // -------------------------------------------------------------------------

    /// Create and add a new sync group.
    ///
    /// Returns the assigned group ID.
    pub fn add_group(&mut self, mut group: SyncGroup) -> SyncGroupId {
        let id = SyncGroupId::new(self.next_group_id);
        self.next_group_id += 1;

        group.id = id;
        self.groups.insert(id, group);
        id
    }

    /// Remove a sync group.
    ///
    /// Returns the removed group if it existed.
    pub fn remove_group(&mut self, id: SyncGroupId) -> Option<SyncGroup> {
        self.groups.remove(&id)
    }

    /// Get a group by ID.
    pub fn get_group(&self, id: SyncGroupId) -> Option<&SyncGroup> {
        self.groups.get(&id)
    }

    /// Get a mutable group by ID.
    pub fn get_group_mut(&mut self, id: SyncGroupId) -> Option<&mut SyncGroup> {
        self.groups.get_mut(&id)
    }

    /// Find a group by name.
    pub fn find_group(&self, name: &str) -> Option<&SyncGroup> {
        self.groups.values().find(|g| g.name == name)
    }

    /// Find a group ID by name.
    pub fn find_group_id(&self, name: &str) -> Option<SyncGroupId> {
        self.groups
            .iter()
            .find(|(_, g)| g.name == name)
            .map(|(id, _)| *id)
    }

    /// Get the number of groups.
    #[inline]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Check if a group exists.
    #[inline]
    pub fn has_group(&self, id: SyncGroupId) -> bool {
        self.groups.contains_key(&id)
    }

    /// Get all group IDs.
    pub fn group_ids(&self) -> Vec<SyncGroupId> {
        self.groups.keys().copied().collect()
    }

    // -------------------------------------------------------------------------
    // Member Management
    // -------------------------------------------------------------------------

    /// Register a member with a group.
    ///
    /// Returns the member index within the group.
    pub fn register_member(&mut self, group_id: SyncGroupId, member_id: usize) -> Option<usize> {
        let group = self.groups.get_mut(&group_id)?;
        let member = SyncMember::new(member_id);
        Some(group.add_member(member))
    }

    /// Unregister a member from a group.
    ///
    /// Returns true if the member was removed.
    pub fn unregister_member(&mut self, group_id: SyncGroupId, member_id: usize) -> bool {
        if let Some(group) = self.groups.get_mut(&group_id) {
            group.remove_member(member_id)
        } else {
            false
        }
    }

    /// Set a member's phase.
    pub fn set_member_phase(
        &mut self,
        group_id: SyncGroupId,
        member_id: usize,
        phase: f32,
    ) -> bool {
        if let Some(group) = self.groups.get_mut(&group_id) {
            if let Some(member_idx) = group.find_member_index(member_id) {
                if let Some(member) = group.get_member_mut(member_idx) {
                    member.set_phase(phase);
                    return true;
                }
            }
        }
        false
    }

    /// Set a member's current clip.
    pub fn set_member_clip(
        &mut self,
        group_id: SyncGroupId,
        member_id: usize,
        clip_id: usize,
    ) -> bool {
        if let Some(group) = self.groups.get_mut(&group_id) {
            if let Some(member_idx) = group.find_member_index(member_id) {
                if let Some(member) = group.get_member_mut(member_idx) {
                    member.current_clip = Some(clip_id);
                    return true;
                }
            }
        }
        false
    }

    /// Set a member's weight.
    pub fn set_member_weight(
        &mut self,
        group_id: SyncGroupId,
        member_id: usize,
        weight: f32,
    ) -> bool {
        if let Some(group) = self.groups.get_mut(&group_id) {
            if let Some(member_idx) = group.find_member_index(member_id) {
                if let Some(member) = group.get_member_mut(member_idx) {
                    member.weight = weight.clamp(0.0, 1.0);
                    return true;
                }
            }
        }
        false
    }

    // -------------------------------------------------------------------------
    // Track Management
    // -------------------------------------------------------------------------

    /// Add a sync track.
    pub fn add_track(&mut self, track: SyncTrack) {
        self.tracks.insert(track.clip_id, track);
    }

    /// Remove a sync track.
    ///
    /// Returns the removed track if it existed.
    pub fn remove_track(&mut self, clip_id: usize) -> Option<SyncTrack> {
        self.tracks.remove(&clip_id)
    }

    /// Get a track by clip ID.
    pub fn get_track(&self, clip_id: usize) -> Option<&SyncTrack> {
        self.tracks.get(&clip_id)
    }

    /// Get a mutable track by clip ID.
    pub fn get_track_mut(&mut self, clip_id: usize) -> Option<&mut SyncTrack> {
        self.tracks.get_mut(&clip_id)
    }

    /// Check if a track exists.
    #[inline]
    pub fn has_track(&self, clip_id: usize) -> bool {
        self.tracks.contains_key(&clip_id)
    }

    /// Get the number of tracks.
    #[inline]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    // -------------------------------------------------------------------------
    // Update
    // -------------------------------------------------------------------------

    /// Update all sync groups.
    pub fn update(&mut self, dt: f32) {
        if !self.enabled {
            return;
        }

        for group in self.groups.values_mut() {
            group.update(dt);
        }
    }

    /// Update a specific group.
    pub fn update_group(&mut self, id: SyncGroupId, dt: f32) {
        if let Some(group) = self.groups.get_mut(&id) {
            group.update(dt);
        }
    }

    // -------------------------------------------------------------------------
    // Phase Synchronization
    // -------------------------------------------------------------------------

    /// Compute phase offset between two clips at matching markers.
    ///
    /// Returns the phase offset needed to align clip B to clip A at the specified marker.
    pub fn compute_phase_offset(
        &self,
        clip_a: usize,
        clip_b: usize,
        marker_name: &str,
    ) -> Option<f32> {
        let track_a = self.tracks.get(&clip_a)?;
        let track_b = self.tracks.get(&clip_b)?;

        let marker_a = track_a.find_marker(marker_name)?;
        let marker_b = track_b.find_marker(marker_name)?;

        Some(phase_difference(marker_a.normalized_time, marker_b.normalized_time))
    }

    /// Align a member to another member at a matching marker.
    pub fn align_to_marker(
        &mut self,
        group_id: SyncGroupId,
        source_member_id: usize,
        target_member_id: usize,
        marker_name: &str,
    ) -> bool {
        // Get clips and phases from both members
        let (source_clip, source_phase, target_clip) = {
            let group = match self.groups.get(&group_id) {
                Some(g) => g,
                None => return false,
            };
            let source = match group.find_member(source_member_id) {
                Some(m) => m,
                None => return false,
            };
            let target = match group.find_member(target_member_id) {
                Some(m) => m,
                None => return false,
            };
            let source_clip = match source.current_clip {
                Some(c) => c,
                None => return false,
            };
            let target_clip = match target.current_clip {
                Some(c) => c,
                None => return false,
            };
            (source_clip, source.phase, target_clip)
        };

        // Compute offset
        let offset = match self.compute_phase_offset(source_clip, target_clip, marker_name) {
            Some(o) => o,
            None => return false,
        };

        // Apply offset to target
        let target_phase = wrap_phase(source_phase + offset);
        self.set_member_phase(group_id, target_member_id, target_phase)
    }

    /// Get the sync state for a member at a given phase.
    pub fn query_sync_state(
        &self,
        group_id: SyncGroupId,
        member_id: usize,
    ) -> Option<SyncQueryResult> {
        let group = self.groups.get(&group_id)?;
        let member = group.find_member(member_id)?;

        let track = member.current_clip.and_then(|c| self.tracks.get(&c));

        Some(SyncQueryResult {
            phase: member.phase,
            phase_velocity: member.phase_velocity,
            group_phase: group.group_phase,
            current_clip: member.current_clip,
            has_track: track.is_some(),
            marker_count: track.map(|t| t.marker_count()).unwrap_or(0),
        })
    }

    /// Reset all groups.
    pub fn reset(&mut self) {
        for group in self.groups.values_mut() {
            group.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// SyncQueryResult
// ---------------------------------------------------------------------------

/// Result of a sync state query.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyncQueryResult {
    /// Member's current phase.
    pub phase: f32,

    /// Member's phase velocity.
    pub phase_velocity: f32,

    /// Group's current phase.
    pub group_phase: f32,

    /// Currently playing clip.
    pub current_clip: Option<usize>,

    /// Whether a sync track exists for the current clip.
    pub has_track: bool,

    /// Number of markers in the current track.
    pub marker_count: usize,
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Wrap a phase value to [0, 1).
#[inline]
pub fn wrap_phase(phase: f32) -> f32 {
    let wrapped = phase - phase.floor();
    if wrapped < 0.0 {
        wrapped + 1.0
    } else {
        wrapped
    }
}

/// Compute the shortest signed difference between two phases.
///
/// Returns a value in [-0.5, 0.5].
#[inline]
pub fn phase_difference(from: f32, to: f32) -> f32 {
    let diff = wrap_phase(to) - wrap_phase(from);
    if diff > 0.5 {
        diff - 1.0
    } else if diff < -0.5 {
        diff + 1.0
    } else {
        diff
    }
}

/// Circular linear interpolation between two phases.
#[inline]
pub fn circular_lerp(a: f32, b: f32, t: f32) -> f32 {
    let diff = phase_difference(a, b);
    wrap_phase(a + diff * t)
}

/// Convert speed to gait cycles per second (for foot sync).
///
/// Typical values: walk ~1.0 cycle/s, run ~2.0 cycles/s.
#[inline]
pub fn speed_to_gait_frequency(speed: f32, stride_length: f32) -> f32 {
    if stride_length > PHASE_EPSILON {
        speed / (stride_length * 2.0) // Two steps per gait cycle
    } else {
        1.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Utility Function Tests
    // =========================================================================

    #[test]
    fn test_wrap_phase_positive() {
        assert!((wrap_phase(0.0) - 0.0).abs() < PHASE_EPSILON);
        assert!((wrap_phase(0.5) - 0.5).abs() < PHASE_EPSILON);
        assert!((wrap_phase(0.99) - 0.99).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_wrap_phase_overflow() {
        assert!((wrap_phase(1.0) - 0.0).abs() < PHASE_EPSILON);
        assert!((wrap_phase(1.5) - 0.5).abs() < PHASE_EPSILON);
        assert!((wrap_phase(2.3) - 0.3).abs() < PHASE_EPSILON);
        assert!((wrap_phase(5.7) - 0.7).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_wrap_phase_negative() {
        let wrapped = wrap_phase(-0.3);
        assert!((wrapped - 0.7).abs() < PHASE_EPSILON);

        let wrapped2 = wrap_phase(-1.5);
        assert!((wrapped2 - 0.5).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_phase_difference_simple() {
        // Forward difference
        let diff = phase_difference(0.2, 0.5);
        assert!((diff - 0.3).abs() < PHASE_EPSILON);

        // Backward difference
        let diff2 = phase_difference(0.5, 0.2);
        assert!((diff2 - (-0.3)).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_phase_difference_wraparound() {
        // Shorter path goes backward across 0
        let diff = phase_difference(0.9, 0.1);
        assert!((diff - 0.2).abs() < PHASE_EPSILON);

        // Shorter path goes forward across 0
        let diff2 = phase_difference(0.1, 0.9);
        assert!((diff2 - (-0.2)).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_phase_difference_same() {
        let diff = phase_difference(0.5, 0.5);
        assert!((diff - 0.0).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_circular_lerp_simple() {
        let lerped = circular_lerp(0.2, 0.8, 0.5);
        assert!((lerped - 0.5).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_circular_lerp_wraparound() {
        // Lerp from 0.9 to 0.1 should go through 0/1
        let lerped = circular_lerp(0.9, 0.1, 0.5);
        assert!((lerped - 0.0).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_circular_lerp_edges() {
        let start = circular_lerp(0.3, 0.7, 0.0);
        assert!((start - 0.3).abs() < PHASE_EPSILON);

        let end = circular_lerp(0.3, 0.7, 1.0);
        assert!((end - 0.7).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_speed_to_gait_frequency() {
        // Walking at 1.4 m/s with 0.7m stride should be ~1 cycle/s
        let freq = speed_to_gait_frequency(1.4, 0.7);
        assert!((freq - 1.0).abs() < 0.01);
    }

    // =========================================================================
    // MarkerType Tests
    // =========================================================================

    #[test]
    fn test_marker_type_default() {
        assert_eq!(MarkerType::default(), MarkerType::Generic);
    }

    #[test]
    fn test_marker_type_is_foot_marker() {
        assert!(MarkerType::FootContact.is_foot_marker());
        assert!(MarkerType::FootLift.is_foot_marker());
        assert!(!MarkerType::Generic.is_foot_marker());
        assert!(!MarkerType::Beat.is_foot_marker());
    }

    #[test]
    fn test_marker_type_names() {
        assert_eq!(MarkerType::FootContact.name(), "Foot Contact");
        assert_eq!(MarkerType::Beat.name(), "Beat");
    }

    // =========================================================================
    // SyncMarker Tests
    // =========================================================================

    #[test]
    fn test_sync_marker_new() {
        let marker = SyncMarker::new("test", 0.5);
        assert_eq!(marker.name, "test");
        assert!((marker.normalized_time - 0.5).abs() < PHASE_EPSILON);
        assert_eq!(marker.marker_type, MarkerType::Generic);
        assert!(marker.bone_index.is_none());
    }

    #[test]
    fn test_sync_marker_clamping() {
        let marker_low = SyncMarker::new("low", -0.5);
        assert!((marker_low.normalized_time - 0.0).abs() < PHASE_EPSILON);

        let marker_high = SyncMarker::new("high", 1.5);
        assert!((marker_high.normalized_time - 1.0).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_marker_foot_contact() {
        let marker = SyncMarker::foot_contact("left_foot_down", 0.25);
        assert_eq!(marker.marker_type, MarkerType::FootContact);
        assert!((marker.normalized_time - 0.25).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_marker_foot_lift() {
        let marker = SyncMarker::foot_lift("right_foot_up", 0.75);
        assert_eq!(marker.marker_type, MarkerType::FootLift);
    }

    #[test]
    fn test_sync_marker_beat() {
        let marker = SyncMarker::beat("beat_1", 0.0);
        assert_eq!(marker.marker_type, MarkerType::Beat);
    }

    #[test]
    fn test_sync_marker_event() {
        let marker = SyncMarker::event("attack_hit", 0.5);
        assert_eq!(marker.marker_type, MarkerType::Event);
    }

    #[test]
    fn test_sync_marker_with_bone() {
        let marker = SyncMarker::foot_contact("left_foot", 0.0).with_bone(5);
        assert_eq!(marker.bone_index, Some(5));
    }

    #[test]
    fn test_sync_marker_with_user_data() {
        let marker = SyncMarker::event("custom", 0.5).with_user_data(42);
        assert_eq!(marker.user_data, Some(42));
    }

    // =========================================================================
    // SyncTrack Tests
    // =========================================================================

    #[test]
    fn test_sync_track_new() {
        let track = SyncTrack::new(5);
        assert_eq!(track.clip_id, 5);
        assert!(track.is_empty());
        assert_eq!(track.marker_count(), 0);
    }

    #[test]
    fn test_sync_track_with_duration() {
        let track = SyncTrack::new(0).with_duration(2.5);
        assert!((track.duration - 2.5).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_track_with_name() {
        let track = SyncTrack::new(0).with_name("walk_cycle");
        assert_eq!(track.name, Some("walk_cycle".to_string()));
    }

    #[test]
    fn test_sync_track_add_marker() {
        let mut track = SyncTrack::new(0);
        track.add_marker(SyncMarker::new("m1", 0.3));
        track.add_marker(SyncMarker::new("m2", 0.7));

        assert_eq!(track.marker_count(), 2);
        assert!(!track.is_empty());
    }

    #[test]
    fn test_sync_track_remove_marker() {
        let mut track = SyncTrack::new(0);
        track.add_marker(SyncMarker::new("m1", 0.3));
        track.add_marker(SyncMarker::new("m2", 0.7));

        assert!(track.remove_marker("m1"));
        assert_eq!(track.marker_count(), 1);
        assert!(!track.remove_marker("nonexistent"));
    }

    #[test]
    fn test_sync_track_find_marker() {
        let mut track = SyncTrack::new(0);
        track.add_marker(SyncMarker::new("target", 0.5));

        let found = track.find_marker("target");
        assert!(found.is_some());
        assert!((found.unwrap().normalized_time - 0.5).abs() < PHASE_EPSILON);

        assert!(track.find_marker("nonexistent").is_none());
    }

    #[test]
    fn test_sync_track_find_markers_by_type() {
        let mut track = SyncTrack::new(0);
        track.add_marker(SyncMarker::foot_contact("lf", 0.0));
        track.add_marker(SyncMarker::foot_contact("rf", 0.5));
        track.add_marker(SyncMarker::beat("b1", 0.25));

        let foot_markers = track.find_markers_by_type(MarkerType::FootContact);
        assert_eq!(foot_markers.len(), 2);
    }

    #[test]
    fn test_sync_track_foot_contacts() {
        let mut track = SyncTrack::new(0);
        track.add_marker(SyncMarker::foot_contact("lf", 0.0));
        track.add_marker(SyncMarker::foot_contact("rf", 0.5));
        track.add_marker(SyncMarker::event("other", 0.25));

        let contacts = track.foot_contacts();
        assert_eq!(contacts.len(), 2);
    }

    #[test]
    fn test_sync_track_sort_markers() {
        let mut track = SyncTrack::new(0);
        track.add_marker(SyncMarker::new("m3", 0.9));
        track.add_marker(SyncMarker::new("m1", 0.1));
        track.add_marker(SyncMarker::new("m2", 0.5));

        track.sort_markers();

        assert!((track.markers[0].normalized_time - 0.1).abs() < PHASE_EPSILON);
        assert!((track.markers[1].normalized_time - 0.5).abs() < PHASE_EPSILON);
        assert!((track.markers[2].normalized_time - 0.9).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_track_marker_before() {
        let mut track = SyncTrack::new(0);
        track.add_marker(SyncMarker::new("a", 0.2));
        track.add_marker(SyncMarker::new("b", 0.6));

        let before = track.marker_before(0.4);
        assert!(before.is_some());
        assert_eq!(before.unwrap().name, "a");

        let before_start = track.marker_before(0.1);
        assert!(before_start.is_none());
    }

    #[test]
    fn test_sync_track_marker_after() {
        let mut track = SyncTrack::new(0);
        track.add_marker(SyncMarker::new("a", 0.2));
        track.add_marker(SyncMarker::new("b", 0.6));

        let after = track.marker_after(0.3);
        assert!(after.is_some());
        assert_eq!(after.unwrap().name, "b");

        let after_end = track.marker_after(0.8);
        assert!(after_end.is_none());
    }

    #[test]
    fn test_sync_track_surrounding_markers() {
        let mut track = SyncTrack::new(0);
        track.add_marker(SyncMarker::new("a", 0.2));
        track.add_marker(SyncMarker::new("b", 0.6));

        let (before, after) = track.surrounding_markers(0.4);
        assert!(before.is_some());
        assert!(after.is_some());
        assert_eq!(before.unwrap().name, "a");
        assert_eq!(after.unwrap().name, "b");
    }

    #[test]
    fn test_sync_track_interpolate_between() {
        let mut track = SyncTrack::new(0);
        track.add_marker(SyncMarker::new("a", 0.2));
        track.add_marker(SyncMarker::new("b", 0.6));

        let mid = track.interpolate_between("a", "b", 0.5);
        assert!(mid.is_some());
        assert!((mid.unwrap() - 0.4).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_track_time_conversions() {
        let track = SyncTrack::new(0).with_duration(2.0);

        assert!((track.to_absolute_time(0.5) - 1.0).abs() < PHASE_EPSILON);
        assert!((track.to_normalized_time(1.5) - 0.75).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_track_auto_detect_from_events() {
        let mut track = SyncTrack::new(0).with_duration(1.0);
        let events = vec![
            (0.0, "left_foot_down".to_string()),
            (0.5, "right_foot_contact".to_string()),
            (0.25, "beat_1".to_string()),
        ];

        track.auto_detect_from_events(&events);

        assert_eq!(track.marker_count(), 3);
        assert!(track.find_marker("left_foot_down").is_some());
    }

    // =========================================================================
    // SyncGroupMode Tests
    // =========================================================================

    #[test]
    fn test_sync_group_mode_default() {
        assert_eq!(SyncGroupMode::default(), SyncGroupMode::Leader);
    }

    #[test]
    fn test_sync_group_mode_requires_master() {
        assert!(SyncGroupMode::Leader.requires_master());
        assert!(!SyncGroupMode::PhaseLocked.requires_master());
        assert!(!SyncGroupMode::Independent.requires_master());
    }

    #[test]
    fn test_sync_group_mode_synchronizes_phase() {
        assert!(SyncGroupMode::Leader.synchronizes_phase());
        assert!(SyncGroupMode::PhaseLocked.synchronizes_phase());
        assert!(!SyncGroupMode::Independent.synchronizes_phase());
    }

    // =========================================================================
    // SyncMember Tests
    // =========================================================================

    #[test]
    fn test_sync_member_new() {
        let member = SyncMember::new(5);
        assert_eq!(member.id, 5);
        assert!((member.phase - 0.0).abs() < PHASE_EPSILON);
        assert!(member.active);
    }

    #[test]
    fn test_sync_member_with_name() {
        let member = SyncMember::new(0).with_name("layer_0");
        assert_eq!(member.name, Some("layer_0".to_string()));
    }

    #[test]
    fn test_sync_member_with_clip() {
        let member = SyncMember::new(0).with_clip(3);
        assert_eq!(member.current_clip, Some(3));
    }

    #[test]
    fn test_sync_member_set_phase() {
        let mut member = SyncMember::new(0);
        member.set_phase(0.7);
        assert!((member.phase - 0.7).abs() < PHASE_EPSILON);
        assert!((member.target_phase - 0.7).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_member_set_phase_wrapping() {
        let mut member = SyncMember::new(0);
        member.set_phase(1.3);
        assert!((member.phase - 0.3).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_member_advance_phase() {
        let mut member = SyncMember::new(0);
        member.phase_velocity = 1.0;
        member.advance_phase(0.25);
        assert!((member.phase - 0.25).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_member_advance_phase_wrapping() {
        let mut member = SyncMember::new(0);
        member.phase = 0.9;
        member.phase_velocity = 1.0;
        member.advance_phase(0.2);
        assert!((member.phase - 0.1).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_member_phase_correction() {
        let mut member = SyncMember::new(0);
        member.correction_rate = 10.0;
        member.phase = 0.0;
        member.set_target_phase(0.5);

        // After one update, should move toward target
        member.update_phase_correction(0.1);
        assert!(member.phase > 0.0);
        assert!(member.phase < 0.5);
    }

    #[test]
    fn test_sync_member_phase_correction_snap() {
        let mut member = SyncMember::new(0);
        member.phase = 0.5;
        member.set_target_phase(0.505); // Within snap threshold

        member.update_phase_correction(0.1);
        assert!((member.phase - 0.505).abs() < PHASE_EPSILON);
    }

    // =========================================================================
    // SyncGroup Tests
    // =========================================================================

    #[test]
    fn test_sync_group_new() {
        let group = SyncGroup::new("locomotion");
        assert_eq!(group.name, "locomotion");
        assert!(group.is_empty());
        assert!(group.enabled);
    }

    #[test]
    fn test_sync_group_with_mode() {
        let group = SyncGroup::new("test").with_mode(SyncGroupMode::PhaseLocked);
        assert_eq!(group.sync_mode, SyncGroupMode::PhaseLocked);
    }

    #[test]
    fn test_sync_group_add_member() {
        let mut group = SyncGroup::new("test");
        let idx = group.add_member(SyncMember::new(0));
        assert_eq!(idx, 0);
        assert_eq!(group.member_count(), 1);
        assert!(!group.is_empty());
    }

    #[test]
    fn test_sync_group_first_member_becomes_master() {
        let mut group = SyncGroup::new("test").with_mode(SyncGroupMode::Leader);
        group.add_member(SyncMember::new(0));
        assert_eq!(group.master, Some(0));
    }

    #[test]
    fn test_sync_group_remove_member() {
        let mut group = SyncGroup::new("test");
        group.add_member(SyncMember::new(0));
        group.add_member(SyncMember::new(1));

        assert!(group.remove_member(0));
        assert_eq!(group.member_count(), 1);
        assert!(!group.remove_member(99));
    }

    #[test]
    fn test_sync_group_get_member() {
        let mut group = SyncGroup::new("test");
        group.add_member(SyncMember::new(42));

        let member = group.get_member(0);
        assert!(member.is_some());
        assert_eq!(member.unwrap().id, 42);
    }

    #[test]
    fn test_sync_group_find_member() {
        let mut group = SyncGroup::new("test");
        group.add_member(SyncMember::new(10));
        group.add_member(SyncMember::new(20));

        let found = group.find_member(20);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, 20);

        assert!(group.find_member(99).is_none());
    }

    #[test]
    fn test_sync_group_find_member_index() {
        let mut group = SyncGroup::new("test");
        group.add_member(SyncMember::new(10));
        group.add_member(SyncMember::new(20));

        assert_eq!(group.find_member_index(10), Some(0));
        assert_eq!(group.find_member_index(20), Some(1));
        assert_eq!(group.find_member_index(99), None);
    }

    #[test]
    fn test_sync_group_set_master() {
        let mut group = SyncGroup::new("test");
        group.add_member(SyncMember::new(0));
        group.add_member(SyncMember::new(1));

        assert!(group.set_master(1));
        assert_eq!(group.master, Some(1));

        assert!(!group.set_master(99)); // Out of bounds
    }

    #[test]
    fn test_sync_group_active_member_count() {
        let mut group = SyncGroup::new("test");
        group.add_member(SyncMember::new(0));
        let mut inactive = SyncMember::new(1);
        inactive.active = false;
        group.add_member(inactive);

        assert_eq!(group.active_member_count(), 1);
    }

    #[test]
    fn test_sync_group_set_phase() {
        let mut group = SyncGroup::new("test");
        group.add_member(SyncMember::new(0));
        group.add_member(SyncMember::new(1));

        group.set_phase(0.5);

        assert!((group.group_phase - 0.5).abs() < PHASE_EPSILON);
        assert!((group.get_member(0).unwrap().phase - 0.5).abs() < PHASE_EPSILON);
        assert!((group.get_member(1).unwrap().phase - 0.5).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_group_set_phase_velocity() {
        let mut group = SyncGroup::new("test");
        group.add_member(SyncMember::new(0));

        group.set_phase_velocity(2.0);

        assert!((group.group_phase_velocity - 2.0).abs() < PHASE_EPSILON);
        assert!((group.get_member(0).unwrap().phase_velocity - 2.0).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_group_weighted_average_phase() {
        let mut group = SyncGroup::new("test");

        let mut m1 = SyncMember::new(0);
        m1.phase = 0.2;
        m1.weight = 1.0;

        let mut m2 = SyncMember::new(1);
        m2.phase = 0.4;
        m2.weight = 1.0;

        group.add_member(m1);
        group.add_member(m2);

        let avg = group.weighted_average_phase();
        // Should be approximately 0.3
        assert!((avg - 0.3).abs() < 0.05);
    }

    #[test]
    fn test_sync_group_reset() {
        let mut group = SyncGroup::new("test");
        group.add_member(SyncMember::new(0));
        group.group_phase = 0.5;
        group.get_member_mut(0).unwrap().phase = 0.7;

        group.reset();

        assert!((group.group_phase - 0.0).abs() < PHASE_EPSILON);
        assert!((group.get_member(0).unwrap().phase - 0.0).abs() < PHASE_EPSILON);
    }

    // =========================================================================
    // SyncGroup Update Mode Tests
    // =========================================================================

    #[test]
    fn test_sync_group_update_leader_mode() {
        let mut group = SyncGroup::new("test").with_mode(SyncGroupMode::Leader);

        let mut master = SyncMember::new(0);
        master.phase_velocity = 1.0;
        group.add_member(master);

        let follower = SyncMember::new(1);
        group.add_member(follower);

        group.update(0.1);

        // Master should advance
        let master_phase = group.get_member(0).unwrap().phase;
        assert!((master_phase - 0.1).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_group_update_phase_locked_mode() {
        let mut group = SyncGroup::new("test").with_mode(SyncGroupMode::PhaseLocked);
        group.group_phase_velocity = 1.0;

        group.add_member(SyncMember::new(0));
        group.add_member(SyncMember::new(1));

        group.update(0.25);

        // All members should be at the same phase
        let phase_0 = group.get_member(0).unwrap().phase;
        let phase_1 = group.get_member(1).unwrap().phase;
        let group_phase = group.group_phase;

        assert!((phase_0 - 0.25).abs() < PHASE_EPSILON);
        assert!((phase_1 - 0.25).abs() < PHASE_EPSILON);
        assert!((group_phase - 0.25).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_group_update_independent_mode() {
        let mut group = SyncGroup::new("test").with_mode(SyncGroupMode::Independent);

        let mut m1 = SyncMember::new(0);
        m1.phase_velocity = 1.0;

        let mut m2 = SyncMember::new(1);
        m2.phase_velocity = 2.0;

        group.add_member(m1);
        group.add_member(m2);

        group.update(0.1);

        // Members advance independently
        let phase_0 = group.get_member(0).unwrap().phase;
        let phase_1 = group.get_member(1).unwrap().phase;

        assert!((phase_0 - 0.1).abs() < PHASE_EPSILON);
        assert!((phase_1 - 0.2).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_group_disabled_no_update() {
        let mut group = SyncGroup::new("test")
            .with_mode(SyncGroupMode::PhaseLocked)
            .disabled();
        group.group_phase_velocity = 1.0;
        group.add_member(SyncMember::new(0));

        group.update(0.5);

        // Phase should not have advanced
        assert!((group.group_phase - 0.0).abs() < PHASE_EPSILON);
    }

    // =========================================================================
    // FootSyncState Tests
    // =========================================================================

    #[test]
    fn test_foot_sync_state_new() {
        let state = FootSyncState::new();
        assert!((state.gait_phase - 0.0).abs() < PHASE_EPSILON);
        assert!(state.left_planted);
        assert!(!state.right_planted);
    }

    #[test]
    fn test_foot_sync_state_from_phase() {
        let state = FootSyncState::from_phase(0.5);
        assert!((state.gait_phase - 0.5).abs() < PHASE_EPSILON);
        assert!((state.right_foot_phase - 0.0).abs() < PHASE_EPSILON); // Offset by 0.5
    }

    #[test]
    fn test_foot_sync_state_update() {
        let mut state = FootSyncState::new();
        state.gait_velocity = 1.0;
        state.update(0.25);

        assert!((state.gait_phase - 0.25).abs() < PHASE_EPSILON);
        assert!((state.left_foot_phase - 0.25).abs() < PHASE_EPSILON);
        assert!((state.right_foot_phase - 0.75).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_foot_sync_state_planted_states() {
        let mut state = FootSyncState::new();

        // At phase 0, left foot should be planted
        state.set_phase(0.0);
        assert!(state.left_planted);
        assert!(!state.right_planted);

        // At phase 0.5, right foot should be planted
        state.set_phase(0.5);
        assert!(!state.left_planted);
        assert!(state.right_planted);
    }

    #[test]
    fn test_foot_sync_state_is_double_support() {
        let mut state = FootSyncState::new();
        state.left_planted = true;
        state.right_planted = true;
        assert!(state.is_double_support());

        state.right_planted = false;
        assert!(!state.is_double_support());
    }

    #[test]
    fn test_foot_sync_state_is_single_support() {
        let mut state = FootSyncState::new();
        state.left_planted = true;
        state.right_planted = false;
        assert!(state.is_single_support());

        state.left_planted = true;
        state.right_planted = true;
        assert!(!state.is_single_support());
    }

    #[test]
    fn test_foot_sync_state_planted_foot() {
        let mut state = FootSyncState::new();

        state.left_planted = true;
        state.right_planted = false;
        assert_eq!(state.planted_foot(), 0);

        state.left_planted = false;
        state.right_planted = true;
        assert_eq!(state.planted_foot(), 1);

        state.left_planted = true;
        state.right_planted = true;
        assert_eq!(state.planted_foot(), -1);
    }

    #[test]
    fn test_foot_sync_state_blend() {
        let state_a = FootSyncState::from_phase(0.0);
        let state_b = FootSyncState::from_phase(0.5);

        let blended = state_a.blend(&state_b, 0.5);
        assert!((blended.gait_phase - 0.25).abs() < 0.05);
    }

    // =========================================================================
    // SyncGroupManager Tests
    // =========================================================================

    #[test]
    fn test_sync_group_manager_new() {
        let manager = SyncGroupManager::new();
        assert_eq!(manager.group_count(), 0);
        assert_eq!(manager.track_count(), 0);
        assert!(manager.enabled);
    }

    #[test]
    fn test_sync_group_manager_add_group() {
        let mut manager = SyncGroupManager::new();
        let group = SyncGroup::new("locomotion");
        let id = manager.add_group(group);

        assert_ne!(id, SyncGroupId::INVALID);
        assert_eq!(manager.group_count(), 1);
        assert!(manager.has_group(id));
    }

    #[test]
    fn test_sync_group_manager_remove_group() {
        let mut manager = SyncGroupManager::new();
        let id = manager.add_group(SyncGroup::new("test"));

        let removed = manager.remove_group(id);
        assert!(removed.is_some());
        assert_eq!(manager.group_count(), 0);
    }

    #[test]
    fn test_sync_group_manager_get_group() {
        let mut manager = SyncGroupManager::new();
        let id = manager.add_group(SyncGroup::new("test"));

        let group = manager.get_group(id);
        assert!(group.is_some());
        assert_eq!(group.unwrap().name, "test");
    }

    #[test]
    fn test_sync_group_manager_find_group() {
        let mut manager = SyncGroupManager::new();
        manager.add_group(SyncGroup::new("locomotion"));
        manager.add_group(SyncGroup::new("combat"));

        let found = manager.find_group("combat");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "combat");

        assert!(manager.find_group("nonexistent").is_none());
    }

    #[test]
    fn test_sync_group_manager_find_group_id() {
        let mut manager = SyncGroupManager::new();
        let id = manager.add_group(SyncGroup::new("test"));

        let found_id = manager.find_group_id("test");
        assert_eq!(found_id, Some(id));
    }

    #[test]
    fn test_sync_group_manager_register_member() {
        let mut manager = SyncGroupManager::new();
        let id = manager.add_group(SyncGroup::new("test"));

        let member_idx = manager.register_member(id, 42);
        assert!(member_idx.is_some());
        assert_eq!(member_idx.unwrap(), 0);

        let group = manager.get_group(id).unwrap();
        assert_eq!(group.member_count(), 1);
    }

    #[test]
    fn test_sync_group_manager_unregister_member() {
        let mut manager = SyncGroupManager::new();
        let id = manager.add_group(SyncGroup::new("test"));
        manager.register_member(id, 42);

        assert!(manager.unregister_member(id, 42));
        assert_eq!(manager.get_group(id).unwrap().member_count(), 0);
    }

    #[test]
    fn test_sync_group_manager_set_member_phase() {
        let mut manager = SyncGroupManager::new();
        let id = manager.add_group(SyncGroup::new("test"));
        manager.register_member(id, 0);

        assert!(manager.set_member_phase(id, 0, 0.7));

        let group = manager.get_group(id).unwrap();
        let member = group.find_member(0).unwrap();
        assert!((member.phase - 0.7).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_group_manager_set_member_clip() {
        let mut manager = SyncGroupManager::new();
        let id = manager.add_group(SyncGroup::new("test"));
        manager.register_member(id, 0);

        assert!(manager.set_member_clip(id, 0, 5));

        let member = manager.get_group(id).unwrap().find_member(0).unwrap();
        assert_eq!(member.current_clip, Some(5));
    }

    #[test]
    fn test_sync_group_manager_add_track() {
        let mut manager = SyncGroupManager::new();
        let track = SyncTrack::new(5);
        manager.add_track(track);

        assert!(manager.has_track(5));
        assert_eq!(manager.track_count(), 1);
    }

    #[test]
    fn test_sync_group_manager_remove_track() {
        let mut manager = SyncGroupManager::new();
        manager.add_track(SyncTrack::new(5));

        let removed = manager.remove_track(5);
        assert!(removed.is_some());
        assert!(!manager.has_track(5));
    }

    #[test]
    fn test_sync_group_manager_get_track() {
        let mut manager = SyncGroupManager::new();
        let mut track = SyncTrack::new(5);
        track.add_marker(SyncMarker::new("m1", 0.5));
        manager.add_track(track);

        let found = manager.get_track(5);
        assert!(found.is_some());
        assert_eq!(found.unwrap().marker_count(), 1);
    }

    #[test]
    fn test_sync_group_manager_update() {
        let mut manager = SyncGroupManager::new();
        let id = manager.add_group(
            SyncGroup::new("test").with_mode(SyncGroupMode::PhaseLocked),
        );
        manager.register_member(id, 0);
        manager.get_group_mut(id).unwrap().group_phase_velocity = 1.0;

        manager.update(0.25);

        let group = manager.get_group(id).unwrap();
        assert!((group.group_phase - 0.25).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_group_manager_compute_phase_offset() {
        let mut manager = SyncGroupManager::new();

        let mut track_a = SyncTrack::new(0);
        track_a.add_marker(SyncMarker::foot_contact("lf", 0.0));

        let mut track_b = SyncTrack::new(1);
        track_b.add_marker(SyncMarker::foot_contact("lf", 0.25));

        manager.add_track(track_a);
        manager.add_track(track_b);

        let offset = manager.compute_phase_offset(0, 1, "lf");
        assert!(offset.is_some());
        assert!((offset.unwrap() - 0.25).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_group_manager_query_sync_state() {
        let mut manager = SyncGroupManager::new();
        let id = manager.add_group(SyncGroup::new("test"));
        manager.register_member(id, 0);
        manager.set_member_phase(id, 0, 0.3);
        manager.set_member_clip(id, 0, 5);

        let mut track = SyncTrack::new(5);
        track.add_marker(SyncMarker::new("m1", 0.5));
        manager.add_track(track);

        let result = manager.query_sync_state(id, 0);
        assert!(result.is_some());

        let state = result.unwrap();
        assert!((state.phase - 0.3).abs() < PHASE_EPSILON);
        assert_eq!(state.current_clip, Some(5));
        assert!(state.has_track);
        assert_eq!(state.marker_count, 1);
    }

    #[test]
    fn test_sync_group_manager_reset() {
        let mut manager = SyncGroupManager::new();
        let id = manager.add_group(SyncGroup::new("test"));
        manager.register_member(id, 0);
        manager.set_member_phase(id, 0, 0.5);
        manager.get_group_mut(id).unwrap().group_phase = 0.5;

        manager.reset();

        let group = manager.get_group(id).unwrap();
        assert!((group.group_phase - 0.0).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_sync_group_manager_group_ids() {
        let mut manager = SyncGroupManager::new();
        let id1 = manager.add_group(SyncGroup::new("a"));
        let id2 = manager.add_group(SyncGroup::new("b"));

        let ids = manager.group_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    // =========================================================================
    // SyncGroupId Tests
    // =========================================================================

    #[test]
    fn test_sync_group_id() {
        let id = SyncGroupId::new(42);
        assert_eq!(id.value(), 42);
    }

    #[test]
    fn test_sync_group_id_invalid() {
        let invalid = SyncGroupId::INVALID;
        assert_eq!(invalid.value(), u32::MAX);
    }

    #[test]
    fn test_sync_group_id_default() {
        let id = SyncGroupId::default();
        assert_eq!(id, SyncGroupId::INVALID);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_empty_group_update() {
        let mut group = SyncGroup::new("empty");
        group.update(0.1); // Should not panic
    }

    #[test]
    fn test_single_member_group() {
        let mut group = SyncGroup::new("single").with_mode(SyncGroupMode::Leader);
        group.add_member(SyncMember::new(0));

        group.update(0.1);
        // Single member is master and follower
        assert!((group.get_member(0).unwrap().phase - 0.1).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_phase_wraparound_during_update() {
        let mut group = SyncGroup::new("test").with_mode(SyncGroupMode::PhaseLocked);
        group.group_phase_velocity = 1.0;
        group.group_phase = 0.9;
        group.add_member(SyncMember::new(0));

        group.update(0.2);

        // Phase should wrap to 0.1
        assert!((group.group_phase - 0.1).abs() < PHASE_EPSILON);
    }

    #[test]
    fn test_manager_invalid_group_operations() {
        let mut manager = SyncGroupManager::new();
        let invalid_id = SyncGroupId::INVALID;

        assert!(manager.get_group(invalid_id).is_none());
        assert!(!manager.unregister_member(invalid_id, 0));
        assert!(!manager.set_member_phase(invalid_id, 0, 0.5));
    }

    #[test]
    fn test_track_with_zero_duration() {
        let track = SyncTrack::new(0).with_duration(0.0);
        // Duration should be clamped to minimum
        assert!(track.duration >= 0.001);
    }

    #[test]
    fn test_marker_interpolation_nonexistent() {
        let track = SyncTrack::new(0);
        let result = track.interpolate_between("a", "b", 0.5);
        assert!(result.is_none());
    }
}
