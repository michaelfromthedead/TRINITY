//! Animation event types for the event system (T-AN-1.6).
//!
//! This module provides typed animation events that integrate with the event log
//! for tracking animation state changes, notifies, and ragdoll activation.
//!
//! # Architecture
//!
//! ```text
//! AnimationEventLog
//! +-- AnimationNotify[]     # Gameplay callbacks (footsteps, sounds)
//! +-- StateTransition[]     # State machine transitions
//! +-- RagdollActivated[]    # Physics ragdoll activation
//! ```
//!
//! # Causal Tracking
//!
//! Events support causal chain tracking via `causal_id`. When an event
//! causes another event, the caused event references the original via
//! its `causal_id` field. This enables debugging and replay analysis.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::animation_events::{AnimationEventLog, AnimationNotify, StateTransition};
//!
//! let mut log = AnimationEventLog::new();
//!
//! // Emit a state transition
//! let transition_id = log.emit(StateTransition {
//!     entity_id: 42,
//!     from_state: "idle".to_string(),
//!     to_state: "walk".to_string(),
//!     blend_time: 0.2,
//!     timestamp: 1000,
//!     causal_id: None,
//! });
//!
//! // Emit a notify caused by the transition
//! log.emit(AnimationNotify {
//!     entity_id: 42,
//!     clip_name: "walk".to_string(),
//!     notify_name: "footstep".to_string(),
//!     time: 0.25,
//!     timestamp: 1250,
//!     causal_id: Some(transition_id),
//! });
//!
//! // Query events
//! let recent = log.events_since(1000);
//! let chain = log.find_causal_chain(transition_id);
//! ```

use std::any::Any;
use std::fmt;

use glam::Vec3;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of events to retain in the log before pruning.
pub const MAX_EVENT_LOG_SIZE: usize = 4096;

/// Default prune threshold (remove oldest 25% when full).
pub const PRUNE_RATIO: f32 = 0.25;

// ---------------------------------------------------------------------------
// AnimationEventKind
// ---------------------------------------------------------------------------

/// Discriminant for animation event types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnimationEventKind {
    /// Animation notify (gameplay callback).
    Notify,
    /// State machine transition.
    StateTransition,
    /// Ragdoll physics activation.
    RagdollActivated,
    /// Custom event type.
    Custom,
}

impl fmt::Display for AnimationEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Notify => write!(f, "Notify"),
            Self::StateTransition => write!(f, "StateTransition"),
            Self::RagdollActivated => write!(f, "RagdollActivated"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

// ---------------------------------------------------------------------------
// AnimationEventTrait
// ---------------------------------------------------------------------------

/// Trait for polymorphic animation events.
///
/// All animation event types implement this trait to enable:
/// - Runtime type identification via `kind()`
/// - Entity and timestamp queries
/// - Causal chain tracking
/// - Type-erased storage in the event log
pub trait AnimationEventTrait: Send + Sync + fmt::Debug {
    /// Returns the event kind discriminant.
    fn kind(&self) -> AnimationEventKind;

    /// Returns the entity ID this event applies to.
    fn entity_id(&self) -> u64;

    /// Returns the timestamp when this event occurred.
    fn timestamp(&self) -> u64;

    /// Returns the causal event ID if this event was caused by another.
    fn causal_id(&self) -> Option<u64>;

    /// Sets the causal ID for this event.
    fn set_causal_id(&mut self, id: Option<u64>);

    /// Returns a descriptive name for this event.
    fn name(&self) -> &str;

    /// Downcast to concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Downcast to concrete type (mutable).
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Clone this event into a boxed trait object.
    fn clone_boxed(&self) -> Box<dyn AnimationEventTrait>;
}

// ---------------------------------------------------------------------------
// AnimationNotify
// ---------------------------------------------------------------------------

/// Animation notify event for gameplay callbacks.
///
/// Notifies are fired at specific times during animation playback and
/// are used for:
/// - Footstep sounds
/// - Particle effects (dust, sparks)
/// - Weapon swing sounds
/// - Impact timing
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationNotify {
    /// Entity this notify applies to.
    pub entity_id: u64,

    /// Name of the animation clip that fired this notify.
    pub clip_name: String,

    /// Name of the notify (e.g., "footstep_left", "attack_start").
    pub notify_name: String,

    /// Time within the clip when the notify fired.
    pub time: f32,

    /// Global timestamp when the event occurred.
    pub timestamp: u64,

    /// ID of the event that caused this notify, if any.
    pub causal_id: Option<u64>,
}

impl AnimationNotify {
    /// Create a new animation notify.
    #[inline]
    pub fn new(
        entity_id: u64,
        clip_name: impl Into<String>,
        notify_name: impl Into<String>,
        time: f32,
        timestamp: u64,
    ) -> Self {
        Self {
            entity_id,
            clip_name: clip_name.into(),
            notify_name: notify_name.into(),
            time,
            timestamp,
            causal_id: None,
        }
    }

    /// Set the causal ID.
    #[inline]
    pub fn with_causal_id(mut self, id: u64) -> Self {
        self.causal_id = Some(id);
        self
    }
}

impl Default for AnimationNotify {
    fn default() -> Self {
        Self {
            entity_id: 0,
            clip_name: String::new(),
            notify_name: String::new(),
            time: 0.0,
            timestamp: 0,
            causal_id: None,
        }
    }
}

impl AnimationEventTrait for AnimationNotify {
    #[inline]
    fn kind(&self) -> AnimationEventKind {
        AnimationEventKind::Notify
    }

    #[inline]
    fn entity_id(&self) -> u64 {
        self.entity_id
    }

    #[inline]
    fn timestamp(&self) -> u64 {
        self.timestamp
    }

    #[inline]
    fn causal_id(&self) -> Option<u64> {
        self.causal_id
    }

    #[inline]
    fn set_causal_id(&mut self, id: Option<u64>) {
        self.causal_id = id;
    }

    #[inline]
    fn name(&self) -> &str {
        &self.notify_name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn AnimationEventTrait> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// StateTransition
// ---------------------------------------------------------------------------

/// State machine transition event.
///
/// Fired when an animation state machine transitions between states.
/// Tracks the source and destination states along with blend timing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StateTransition {
    /// Entity this transition applies to.
    pub entity_id: u64,

    /// Name of the source state.
    pub from_state: String,

    /// Name of the destination state.
    pub to_state: String,

    /// Blend duration in seconds.
    pub blend_time: f32,

    /// Global timestamp when the transition started.
    pub timestamp: u64,

    /// ID of the event that caused this transition, if any.
    pub causal_id: Option<u64>,
}

impl StateTransition {
    /// Create a new state transition.
    #[inline]
    pub fn new(
        entity_id: u64,
        from_state: impl Into<String>,
        to_state: impl Into<String>,
        blend_time: f32,
        timestamp: u64,
    ) -> Self {
        Self {
            entity_id,
            from_state: from_state.into(),
            to_state: to_state.into(),
            blend_time,
            timestamp,
            causal_id: None,
        }
    }

    /// Set the causal ID.
    #[inline]
    pub fn with_causal_id(mut self, id: u64) -> Self {
        self.causal_id = Some(id);
        self
    }

    /// Check if this is a self-transition (from == to).
    #[inline]
    pub fn is_self_transition(&self) -> bool {
        self.from_state == self.to_state
    }
}

impl Default for StateTransition {
    fn default() -> Self {
        Self {
            entity_id: 0,
            from_state: String::new(),
            to_state: String::new(),
            blend_time: 0.0,
            timestamp: 0,
            causal_id: None,
        }
    }
}

impl AnimationEventTrait for StateTransition {
    #[inline]
    fn kind(&self) -> AnimationEventKind {
        AnimationEventKind::StateTransition
    }

    #[inline]
    fn entity_id(&self) -> u64 {
        self.entity_id
    }

    #[inline]
    fn timestamp(&self) -> u64 {
        self.timestamp
    }

    #[inline]
    fn causal_id(&self) -> Option<u64> {
        self.causal_id
    }

    #[inline]
    fn set_causal_id(&mut self, id: Option<u64>) {
        self.causal_id = id;
    }

    fn name(&self) -> &str {
        &self.to_state
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn AnimationEventTrait> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// RagdollActivated
// ---------------------------------------------------------------------------

/// Ragdoll physics activation event.
///
/// Fired when an entity's animation is blended into ragdoll physics,
/// typically upon death or significant impact. Records the impact
/// parameters for physics simulation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RagdollActivated {
    /// Entity being ragdolled.
    pub entity_id: u64,

    /// Blend time from animation to physics (seconds).
    pub blend_time: f32,

    /// Force vector of the impact that triggered ragdoll.
    pub impact_force: Vec3,

    /// World-space point where the impact occurred.
    pub impact_point: Vec3,

    /// Global timestamp when ragdoll was activated.
    pub timestamp: u64,

    /// ID of the event that caused this activation, if any.
    pub causal_id: Option<u64>,
}

impl RagdollActivated {
    /// Create a new ragdoll activation event.
    #[inline]
    pub fn new(
        entity_id: u64,
        blend_time: f32,
        impact_force: Vec3,
        impact_point: Vec3,
        timestamp: u64,
    ) -> Self {
        Self {
            entity_id,
            blend_time,
            impact_force,
            impact_point,
            timestamp,
            causal_id: None,
        }
    }

    /// Set the causal ID.
    #[inline]
    pub fn with_causal_id(mut self, id: u64) -> Self {
        self.causal_id = Some(id);
        self
    }

    /// Get the magnitude of the impact force.
    #[inline]
    pub fn impact_magnitude(&self) -> f32 {
        self.impact_force.length()
    }

    /// Get the normalized impact direction.
    #[inline]
    pub fn impact_direction(&self) -> Vec3 {
        self.impact_force.normalize_or_zero()
    }
}

impl Default for RagdollActivated {
    fn default() -> Self {
        Self {
            entity_id: 0,
            blend_time: 0.0,
            impact_force: Vec3::ZERO,
            impact_point: Vec3::ZERO,
            timestamp: 0,
            causal_id: None,
        }
    }
}

impl AnimationEventTrait for RagdollActivated {
    #[inline]
    fn kind(&self) -> AnimationEventKind {
        AnimationEventKind::RagdollActivated
    }

    #[inline]
    fn entity_id(&self) -> u64 {
        self.entity_id
    }

    #[inline]
    fn timestamp(&self) -> u64 {
        self.timestamp
    }

    #[inline]
    fn causal_id(&self) -> Option<u64> {
        self.causal_id
    }

    #[inline]
    fn set_causal_id(&mut self, id: Option<u64>) {
        self.causal_id = id;
    }

    fn name(&self) -> &str {
        "ragdoll"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn AnimationEventTrait> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// CustomEvent
// ---------------------------------------------------------------------------

/// Custom animation event with arbitrary string payload.
///
/// Used for application-specific events that don't fit the predefined types.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomEvent {
    /// Entity this event applies to.
    pub entity_id: u64,

    /// Custom event name.
    pub event_name: String,

    /// Optional string payload.
    pub payload: Option<String>,

    /// Global timestamp when the event occurred.
    pub timestamp: u64,

    /// ID of the event that caused this, if any.
    pub causal_id: Option<u64>,
}

impl CustomEvent {
    /// Create a new custom event.
    #[inline]
    pub fn new(entity_id: u64, event_name: impl Into<String>, timestamp: u64) -> Self {
        Self {
            entity_id,
            event_name: event_name.into(),
            payload: None,
            timestamp,
            causal_id: None,
        }
    }

    /// Set the payload.
    #[inline]
    pub fn with_payload(mut self, payload: impl Into<String>) -> Self {
        self.payload = Some(payload.into());
        self
    }

    /// Set the causal ID.
    #[inline]
    pub fn with_causal_id(mut self, id: u64) -> Self {
        self.causal_id = Some(id);
        self
    }
}

impl Default for CustomEvent {
    fn default() -> Self {
        Self {
            entity_id: 0,
            event_name: String::new(),
            payload: None,
            timestamp: 0,
            causal_id: None,
        }
    }
}

impl AnimationEventTrait for CustomEvent {
    #[inline]
    fn kind(&self) -> AnimationEventKind {
        AnimationEventKind::Custom
    }

    #[inline]
    fn entity_id(&self) -> u64 {
        self.entity_id
    }

    #[inline]
    fn timestamp(&self) -> u64 {
        self.timestamp
    }

    #[inline]
    fn causal_id(&self) -> Option<u64> {
        self.causal_id
    }

    #[inline]
    fn set_causal_id(&mut self, id: Option<u64>) {
        self.causal_id = id;
    }

    fn name(&self) -> &str {
        &self.event_name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn AnimationEventTrait> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// StoredEvent
// ---------------------------------------------------------------------------

/// An event stored in the log with its assigned ID.
struct StoredEvent {
    /// Unique ID assigned to this event.
    id: u64,
    /// The event data.
    event: Box<dyn AnimationEventTrait>,
}

impl fmt::Debug for StoredEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StoredEvent")
            .field("id", &self.id)
            .field("event", &self.event)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// AnimationEventLog
// ---------------------------------------------------------------------------

/// Event log for tracking animation events.
///
/// The log stores events in chronological order and supports:
/// - Event emission with automatic ID assignment
/// - Timestamp-based queries
/// - Causal chain reconstruction
/// - Entity-filtered queries
/// - Automatic pruning when capacity is exceeded
///
/// # Thread Safety
///
/// The log itself is not thread-safe. Use external synchronization
/// (e.g., `Mutex<AnimationEventLog>`) for concurrent access.
#[derive(Default)]
pub struct AnimationEventLog {
    /// Stored events in chronological order.
    events: Vec<StoredEvent>,

    /// Counter for assigning unique event IDs.
    causal_counter: u64,

    /// Maximum number of events before pruning.
    max_size: usize,
}

impl AnimationEventLog {
    /// Create a new empty event log.
    #[inline]
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            causal_counter: 0,
            max_size: MAX_EVENT_LOG_SIZE,
        }
    }

    /// Create a log with custom capacity.
    #[inline]
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            events: Vec::with_capacity(max_size.min(1024)),
            causal_counter: 0,
            max_size,
        }
    }

    /// Emit an event to the log, returning its assigned ID.
    ///
    /// The ID can be used as a `causal_id` for subsequent events
    /// to establish causal relationships.
    pub fn emit<E: AnimationEventTrait + 'static>(&mut self, event: E) -> u64 {
        // Prune if at capacity
        if self.events.len() >= self.max_size {
            self.prune();
        }

        let id = self.causal_counter;
        self.causal_counter += 1;

        self.events.push(StoredEvent {
            id,
            event: Box::new(event),
        });

        id
    }

    /// Emit a boxed event to the log.
    pub fn emit_boxed(&mut self, event: Box<dyn AnimationEventTrait>) -> u64 {
        if self.events.len() >= self.max_size {
            self.prune();
        }

        let id = self.causal_counter;
        self.causal_counter += 1;

        self.events.push(StoredEvent { id, event });

        id
    }

    /// Get all events that occurred since the given timestamp.
    pub fn events_since(&self, timestamp: u64) -> Vec<&dyn AnimationEventTrait> {
        self.events
            .iter()
            .filter(|stored| stored.event.timestamp() >= timestamp)
            .map(|stored| stored.event.as_ref())
            .collect()
    }

    /// Get all events in a timestamp range [start, end).
    pub fn events_in_range(&self, start: u64, end: u64) -> Vec<&dyn AnimationEventTrait> {
        self.events
            .iter()
            .filter(|stored| {
                let ts = stored.event.timestamp();
                ts >= start && ts < end
            })
            .map(|stored| stored.event.as_ref())
            .collect()
    }

    /// Find the causal chain starting from the given event ID.
    ///
    /// Returns a list of event IDs in causal order (from root cause to effect).
    /// The returned vector includes the given ID if it exists in the log.
    pub fn find_causal_chain(&self, event_id: u64) -> Vec<u64> {
        // Build a reverse map: ID -> causal_id
        let mut chain = Vec::new();
        let mut current_id = Some(event_id);

        // Walk backwards through causal chain
        while let Some(id) = current_id {
            chain.push(id);

            // Find the event with this ID
            let event = self.events.iter().find(|stored| stored.id == id);

            current_id = event.and_then(|stored| stored.event.causal_id());

            // Guard against infinite loops
            if chain.len() > self.events.len() {
                break;
            }
        }

        // Reverse to get root-to-effect order
        chain.reverse();
        chain
    }

    /// Find all events caused by the given event ID (direct descendants).
    pub fn find_caused_by(&self, event_id: u64) -> Vec<&dyn AnimationEventTrait> {
        self.events
            .iter()
            .filter(|stored| stored.event.causal_id() == Some(event_id))
            .map(|stored| stored.event.as_ref())
            .collect()
    }

    /// Get events for a specific entity.
    pub fn events_for_entity(&self, entity_id: u64) -> Vec<&dyn AnimationEventTrait> {
        self.events
            .iter()
            .filter(|stored| stored.event.entity_id() == entity_id)
            .map(|stored| stored.event.as_ref())
            .collect()
    }

    /// Get events of a specific kind.
    pub fn events_of_kind(&self, kind: AnimationEventKind) -> Vec<&dyn AnimationEventTrait> {
        self.events
            .iter()
            .filter(|stored| stored.event.kind() == kind)
            .map(|stored| stored.event.as_ref())
            .collect()
    }

    /// Get the event with the given ID.
    pub fn get(&self, event_id: u64) -> Option<&dyn AnimationEventTrait> {
        self.events
            .iter()
            .find(|stored| stored.id == event_id)
            .map(|stored| stored.event.as_ref())
    }

    /// Get the ID of the most recently emitted event.
    #[inline]
    pub fn last_event_id(&self) -> Option<u64> {
        self.events.last().map(|stored| stored.id)
    }

    /// Get the most recently emitted event.
    #[inline]
    pub fn last_event(&self) -> Option<&dyn AnimationEventTrait> {
        self.events.last().map(|stored| stored.event.as_ref())
    }

    /// Get the number of events in the log.
    #[inline]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if the log is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Clear all events from the log.
    #[inline]
    pub fn clear(&mut self) {
        self.events.clear();
        // Note: causal_counter is NOT reset to avoid ID reuse
    }

    /// Get the current causal counter value.
    #[inline]
    pub fn causal_counter(&self) -> u64 {
        self.causal_counter
    }

    /// Prune oldest events to make room for new ones.
    fn prune(&mut self) {
        let prune_count = (self.events.len() as f32 * PRUNE_RATIO).ceil() as usize;
        if prune_count > 0 && prune_count < self.events.len() {
            self.events.drain(0..prune_count);
        }
    }

    /// Iterate over all events.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (u64, &dyn AnimationEventTrait)> {
        self.events.iter().map(|stored| (stored.id, stored.event.as_ref()))
    }

    /// Count events by kind.
    pub fn count_by_kind(&self) -> std::collections::HashMap<AnimationEventKind, usize> {
        let mut counts = std::collections::HashMap::new();
        for stored in &self.events {
            *counts.entry(stored.event.kind()).or_insert(0) += 1;
        }
        counts
    }

    /// Get statistics about the event log.
    pub fn stats(&self) -> EventLogStats {
        let counts = self.count_by_kind();

        let oldest_timestamp = self
            .events
            .first()
            .map(|s| s.event.timestamp())
            .unwrap_or(0);

        let newest_timestamp = self
            .events
            .last()
            .map(|s| s.event.timestamp())
            .unwrap_or(0);

        EventLogStats {
            total_events: self.events.len(),
            notify_count: *counts.get(&AnimationEventKind::Notify).unwrap_or(&0),
            transition_count: *counts.get(&AnimationEventKind::StateTransition).unwrap_or(&0),
            ragdoll_count: *counts.get(&AnimationEventKind::RagdollActivated).unwrap_or(&0),
            custom_count: *counts.get(&AnimationEventKind::Custom).unwrap_or(&0),
            oldest_timestamp,
            newest_timestamp,
            causal_counter: self.causal_counter,
        }
    }
}

impl fmt::Debug for AnimationEventLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnimationEventLog")
            .field("len", &self.events.len())
            .field("causal_counter", &self.causal_counter)
            .field("max_size", &self.max_size)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// EventLogStats
// ---------------------------------------------------------------------------

/// Statistics about an event log.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EventLogStats {
    /// Total number of events.
    pub total_events: usize,
    /// Number of notify events.
    pub notify_count: usize,
    /// Number of state transition events.
    pub transition_count: usize,
    /// Number of ragdoll activation events.
    pub ragdoll_count: usize,
    /// Number of custom events.
    pub custom_count: usize,
    /// Timestamp of oldest event.
    pub oldest_timestamp: u64,
    /// Timestamp of newest event.
    pub newest_timestamp: u64,
    /// Current causal counter value.
    pub causal_counter: u64,
}

impl EventLogStats {
    /// Get the time span covered by events in the log.
    #[inline]
    pub fn time_span(&self) -> u64 {
        self.newest_timestamp.saturating_sub(self.oldest_timestamp)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Downcast an event to AnimationNotify.
pub fn as_notify(event: &dyn AnimationEventTrait) -> Option<&AnimationNotify> {
    event.as_any().downcast_ref::<AnimationNotify>()
}

/// Downcast an event to StateTransition.
pub fn as_state_transition(event: &dyn AnimationEventTrait) -> Option<&StateTransition> {
    event.as_any().downcast_ref::<StateTransition>()
}

/// Downcast an event to RagdollActivated.
pub fn as_ragdoll_activated(event: &dyn AnimationEventTrait) -> Option<&RagdollActivated> {
    event.as_any().downcast_ref::<RagdollActivated>()
}

/// Downcast an event to CustomEvent.
pub fn as_custom_event(event: &dyn AnimationEventTrait) -> Option<&CustomEvent> {
    event.as_any().downcast_ref::<CustomEvent>()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===== AnimationEventKind Tests =====

    #[test]
    fn test_event_kind_display() {
        assert_eq!(format!("{}", AnimationEventKind::Notify), "Notify");
        assert_eq!(format!("{}", AnimationEventKind::StateTransition), "StateTransition");
        assert_eq!(format!("{}", AnimationEventKind::RagdollActivated), "RagdollActivated");
        assert_eq!(format!("{}", AnimationEventKind::Custom), "Custom");
    }

    #[test]
    fn test_event_kind_equality() {
        assert_eq!(AnimationEventKind::Notify, AnimationEventKind::Notify);
        assert_ne!(AnimationEventKind::Notify, AnimationEventKind::StateTransition);
    }

    // ===== AnimationNotify Tests =====

    #[test]
    fn test_animation_notify_new() {
        let notify = AnimationNotify::new(42, "walk", "footstep", 0.5, 1000);
        assert_eq!(notify.entity_id, 42);
        assert_eq!(notify.clip_name, "walk");
        assert_eq!(notify.notify_name, "footstep");
        assert_eq!(notify.time, 0.5);
        assert_eq!(notify.timestamp, 1000);
        assert!(notify.causal_id.is_none());
    }

    #[test]
    fn test_animation_notify_with_causal_id() {
        let notify = AnimationNotify::new(1, "run", "step", 0.0, 100)
            .with_causal_id(5);
        assert_eq!(notify.causal_id, Some(5));
    }

    #[test]
    fn test_animation_notify_default() {
        let notify = AnimationNotify::default();
        assert_eq!(notify.entity_id, 0);
        assert!(notify.clip_name.is_empty());
        assert!(notify.notify_name.is_empty());
        assert_eq!(notify.time, 0.0);
        assert_eq!(notify.timestamp, 0);
        assert!(notify.causal_id.is_none());
    }

    #[test]
    fn test_animation_notify_trait_impl() {
        let notify = AnimationNotify::new(42, "walk", "footstep", 0.5, 1000);

        assert_eq!(notify.kind(), AnimationEventKind::Notify);
        assert_eq!(notify.entity_id(), 42);
        assert_eq!(notify.timestamp(), 1000);
        assert!(notify.causal_id().is_none());
        assert_eq!(notify.name(), "footstep");
    }

    #[test]
    fn test_animation_notify_set_causal_id() {
        let mut notify = AnimationNotify::new(1, "a", "b", 0.0, 0);
        notify.set_causal_id(Some(10));
        assert_eq!(notify.causal_id(), Some(10));
        notify.set_causal_id(None);
        assert!(notify.causal_id().is_none());
    }

    #[test]
    fn test_animation_notify_clone_boxed() {
        let notify = AnimationNotify::new(42, "walk", "footstep", 0.5, 1000);
        let boxed = notify.clone_boxed();
        assert_eq!(boxed.entity_id(), 42);
        assert_eq!(boxed.kind(), AnimationEventKind::Notify);
    }

    #[test]
    fn test_animation_notify_as_any() {
        let notify = AnimationNotify::new(42, "walk", "footstep", 0.5, 1000);
        let any_ref = notify.as_any();
        let downcast = any_ref.downcast_ref::<AnimationNotify>();
        assert!(downcast.is_some());
        assert_eq!(downcast.unwrap().entity_id, 42);
    }

    // ===== StateTransition Tests =====

    #[test]
    fn test_state_transition_new() {
        let trans = StateTransition::new(1, "idle", "walk", 0.2, 500);
        assert_eq!(trans.entity_id, 1);
        assert_eq!(trans.from_state, "idle");
        assert_eq!(trans.to_state, "walk");
        assert_eq!(trans.blend_time, 0.2);
        assert_eq!(trans.timestamp, 500);
        assert!(trans.causal_id.is_none());
    }

    #[test]
    fn test_state_transition_with_causal_id() {
        let trans = StateTransition::new(1, "a", "b", 0.1, 100)
            .with_causal_id(3);
        assert_eq!(trans.causal_id, Some(3));
    }

    #[test]
    fn test_state_transition_is_self_transition() {
        let self_trans = StateTransition::new(1, "idle", "idle", 0.0, 0);
        assert!(self_trans.is_self_transition());

        let normal_trans = StateTransition::new(1, "idle", "walk", 0.2, 0);
        assert!(!normal_trans.is_self_transition());
    }

    #[test]
    fn test_state_transition_default() {
        let trans = StateTransition::default();
        assert_eq!(trans.entity_id, 0);
        assert!(trans.from_state.is_empty());
        assert!(trans.to_state.is_empty());
        assert_eq!(trans.blend_time, 0.0);
        assert_eq!(trans.timestamp, 0);
        assert!(trans.causal_id.is_none());
    }

    #[test]
    fn test_state_transition_trait_impl() {
        let trans = StateTransition::new(10, "run", "jump", 0.15, 2000);

        assert_eq!(trans.kind(), AnimationEventKind::StateTransition);
        assert_eq!(trans.entity_id(), 10);
        assert_eq!(trans.timestamp(), 2000);
        assert!(trans.causal_id().is_none());
        assert_eq!(trans.name(), "jump");
    }

    #[test]
    fn test_state_transition_clone_boxed() {
        let trans = StateTransition::new(5, "a", "b", 0.1, 100);
        let boxed = trans.clone_boxed();
        assert_eq!(boxed.entity_id(), 5);
        assert_eq!(boxed.kind(), AnimationEventKind::StateTransition);
    }

    // ===== RagdollActivated Tests =====

    #[test]
    fn test_ragdoll_activated_new() {
        let ragdoll = RagdollActivated::new(
            99,
            0.3,
            Vec3::new(100.0, 50.0, 0.0),
            Vec3::new(1.0, 2.0, 3.0),
            3000,
        );

        assert_eq!(ragdoll.entity_id, 99);
        assert_eq!(ragdoll.blend_time, 0.3);
        assert_eq!(ragdoll.impact_force, Vec3::new(100.0, 50.0, 0.0));
        assert_eq!(ragdoll.impact_point, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(ragdoll.timestamp, 3000);
        assert!(ragdoll.causal_id.is_none());
    }

    #[test]
    fn test_ragdoll_activated_with_causal_id() {
        let ragdoll = RagdollActivated::new(1, 0.1, Vec3::ZERO, Vec3::ZERO, 100)
            .with_causal_id(7);
        assert_eq!(ragdoll.causal_id, Some(7));
    }

    #[test]
    fn test_ragdoll_activated_impact_magnitude() {
        let ragdoll = RagdollActivated::new(
            1,
            0.1,
            Vec3::new(3.0, 4.0, 0.0), // magnitude = 5
            Vec3::ZERO,
            0,
        );
        assert!((ragdoll.impact_magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_ragdoll_activated_impact_direction() {
        let ragdoll = RagdollActivated::new(
            1,
            0.1,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::ZERO,
            0,
        );
        let dir = ragdoll.impact_direction();
        assert!((dir - Vec3::X).length() < 1e-5);
    }

    #[test]
    fn test_ragdoll_activated_impact_direction_zero() {
        let ragdoll = RagdollActivated::new(1, 0.1, Vec3::ZERO, Vec3::ZERO, 0);
        let dir = ragdoll.impact_direction();
        assert_eq!(dir, Vec3::ZERO);
    }

    #[test]
    fn test_ragdoll_activated_default() {
        let ragdoll = RagdollActivated::default();
        assert_eq!(ragdoll.entity_id, 0);
        assert_eq!(ragdoll.blend_time, 0.0);
        assert_eq!(ragdoll.impact_force, Vec3::ZERO);
        assert_eq!(ragdoll.impact_point, Vec3::ZERO);
        assert_eq!(ragdoll.timestamp, 0);
        assert!(ragdoll.causal_id.is_none());
    }

    #[test]
    fn test_ragdoll_activated_trait_impl() {
        let ragdoll = RagdollActivated::new(42, 0.2, Vec3::X, Vec3::Y, 5000);

        assert_eq!(ragdoll.kind(), AnimationEventKind::RagdollActivated);
        assert_eq!(ragdoll.entity_id(), 42);
        assert_eq!(ragdoll.timestamp(), 5000);
        assert!(ragdoll.causal_id().is_none());
        assert_eq!(ragdoll.name(), "ragdoll");
    }

    #[test]
    fn test_ragdoll_activated_clone_boxed() {
        let ragdoll = RagdollActivated::new(8, 0.1, Vec3::ONE, Vec3::ZERO, 200);
        let boxed = ragdoll.clone_boxed();
        assert_eq!(boxed.entity_id(), 8);
        assert_eq!(boxed.kind(), AnimationEventKind::RagdollActivated);
    }

    // ===== CustomEvent Tests =====

    #[test]
    fn test_custom_event_new() {
        let event = CustomEvent::new(5, "my_event", 1500);
        assert_eq!(event.entity_id, 5);
        assert_eq!(event.event_name, "my_event");
        assert!(event.payload.is_none());
        assert_eq!(event.timestamp, 1500);
        assert!(event.causal_id.is_none());
    }

    #[test]
    fn test_custom_event_with_payload() {
        let event = CustomEvent::new(1, "test", 0)
            .with_payload("some data");
        assert_eq!(event.payload, Some("some data".to_string()));
    }

    #[test]
    fn test_custom_event_with_causal_id() {
        let event = CustomEvent::new(1, "test", 0)
            .with_causal_id(12);
        assert_eq!(event.causal_id, Some(12));
    }

    #[test]
    fn test_custom_event_default() {
        let event = CustomEvent::default();
        assert_eq!(event.entity_id, 0);
        assert!(event.event_name.is_empty());
        assert!(event.payload.is_none());
        assert_eq!(event.timestamp, 0);
        assert!(event.causal_id.is_none());
    }

    #[test]
    fn test_custom_event_trait_impl() {
        let event = CustomEvent::new(77, "custom_thing", 8000);

        assert_eq!(event.kind(), AnimationEventKind::Custom);
        assert_eq!(event.entity_id(), 77);
        assert_eq!(event.timestamp(), 8000);
        assert!(event.causal_id().is_none());
        assert_eq!(event.name(), "custom_thing");
    }

    // ===== AnimationEventLog Basic Tests =====

    #[test]
    fn test_event_log_new() {
        let log = AnimationEventLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert_eq!(log.causal_counter(), 0);
    }

    #[test]
    fn test_event_log_with_capacity() {
        let log = AnimationEventLog::with_capacity(100);
        assert!(log.is_empty());
        assert_eq!(log.max_size, 100);
    }

    #[test]
    fn test_event_log_emit() {
        let mut log = AnimationEventLog::new();

        let id1 = log.emit(AnimationNotify::new(1, "walk", "step", 0.0, 100));
        assert_eq!(id1, 0);
        assert_eq!(log.len(), 1);

        let id2 = log.emit(StateTransition::new(1, "idle", "walk", 0.2, 200));
        assert_eq!(id2, 1);
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_event_log_emit_boxed() {
        let mut log = AnimationEventLog::new();
        let event: Box<dyn AnimationEventTrait> = Box::new(
            AnimationNotify::new(1, "run", "step", 0.1, 50)
        );
        let id = log.emit_boxed(event);
        assert_eq!(id, 0);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_event_log_get() {
        let mut log = AnimationEventLog::new();
        let id = log.emit(AnimationNotify::new(42, "walk", "footstep", 0.5, 1000));

        let event = log.get(id).unwrap();
        assert_eq!(event.entity_id(), 42);
        assert_eq!(event.kind(), AnimationEventKind::Notify);

        assert!(log.get(999).is_none());
    }

    #[test]
    fn test_event_log_last_event() {
        let mut log = AnimationEventLog::new();
        assert!(log.last_event().is_none());
        assert!(log.last_event_id().is_none());

        log.emit(AnimationNotify::new(1, "a", "b", 0.0, 100));
        log.emit(StateTransition::new(2, "c", "d", 0.1, 200));

        assert_eq!(log.last_event_id(), Some(1));
        let last = log.last_event().unwrap();
        assert_eq!(last.entity_id(), 2);
        assert_eq!(last.kind(), AnimationEventKind::StateTransition);
    }

    #[test]
    fn test_event_log_clear() {
        let mut log = AnimationEventLog::new();
        log.emit(AnimationNotify::new(1, "a", "b", 0.0, 100));
        log.emit(StateTransition::new(2, "c", "d", 0.1, 200));

        assert_eq!(log.len(), 2);
        let counter_before = log.causal_counter();

        log.clear();

        assert!(log.is_empty());
        // Counter should NOT be reset
        assert_eq!(log.causal_counter(), counter_before);
    }

    // ===== Event Query Tests =====

    #[test]
    fn test_events_since() {
        let mut log = AnimationEventLog::new();
        log.emit(AnimationNotify::new(1, "a", "b", 0.0, 100));
        log.emit(AnimationNotify::new(2, "c", "d", 0.0, 200));
        log.emit(AnimationNotify::new(3, "e", "f", 0.0, 300));

        let events = log.events_since(200);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].entity_id(), 2);
        assert_eq!(events[1].entity_id(), 3);
    }

    #[test]
    fn test_events_since_none() {
        let mut log = AnimationEventLog::new();
        log.emit(AnimationNotify::new(1, "a", "b", 0.0, 100));

        let events = log.events_since(500);
        assert!(events.is_empty());
    }

    #[test]
    fn test_events_in_range() {
        let mut log = AnimationEventLog::new();
        log.emit(AnimationNotify::new(1, "a", "b", 0.0, 100));
        log.emit(AnimationNotify::new(2, "c", "d", 0.0, 200));
        log.emit(AnimationNotify::new(3, "e", "f", 0.0, 300));
        log.emit(AnimationNotify::new(4, "g", "h", 0.0, 400));

        let events = log.events_in_range(200, 400);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].entity_id(), 2);
        assert_eq!(events[1].entity_id(), 3);
    }

    #[test]
    fn test_events_for_entity() {
        let mut log = AnimationEventLog::new();
        log.emit(AnimationNotify::new(1, "a", "b", 0.0, 100));
        log.emit(AnimationNotify::new(2, "c", "d", 0.0, 200));
        log.emit(AnimationNotify::new(1, "e", "f", 0.0, 300));

        let events = log.events_for_entity(1);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].timestamp(), 100);
        assert_eq!(events[1].timestamp(), 300);
    }

    #[test]
    fn test_events_of_kind() {
        let mut log = AnimationEventLog::new();
        log.emit(AnimationNotify::new(1, "a", "b", 0.0, 100));
        log.emit(StateTransition::new(2, "c", "d", 0.1, 200));
        log.emit(AnimationNotify::new(3, "e", "f", 0.0, 300));

        let notifies = log.events_of_kind(AnimationEventKind::Notify);
        assert_eq!(notifies.len(), 2);

        let transitions = log.events_of_kind(AnimationEventKind::StateTransition);
        assert_eq!(transitions.len(), 1);

        let ragdolls = log.events_of_kind(AnimationEventKind::RagdollActivated);
        assert!(ragdolls.is_empty());
    }

    // ===== Causal Chain Tests =====

    #[test]
    fn test_find_causal_chain_simple() {
        let mut log = AnimationEventLog::new();

        // Root event
        let id0 = log.emit(StateTransition::new(1, "idle", "walk", 0.2, 100));

        // Caused by id0
        let notify = AnimationNotify::new(1, "walk", "footstep", 0.25, 125)
            .with_causal_id(id0);
        let id1 = log.emit(notify);

        let chain = log.find_causal_chain(id1);
        assert_eq!(chain, vec![id0, id1]);
    }

    #[test]
    fn test_find_causal_chain_long() {
        let mut log = AnimationEventLog::new();

        let id0 = log.emit(CustomEvent::new(1, "trigger", 100));

        let mut prev_id = id0;
        for i in 1..=5 {
            let event = CustomEvent::new(1, format!("event_{}", i), 100 + i * 10)
                .with_causal_id(prev_id);
            prev_id = log.emit(event);
        }

        let chain = log.find_causal_chain(prev_id);
        assert_eq!(chain.len(), 6);
        assert_eq!(chain[0], id0);
        assert_eq!(chain[5], prev_id);
    }

    #[test]
    fn test_find_causal_chain_no_cause() {
        let mut log = AnimationEventLog::new();
        let id = log.emit(AnimationNotify::new(1, "a", "b", 0.0, 100));

        let chain = log.find_causal_chain(id);
        assert_eq!(chain, vec![id]);
    }

    #[test]
    fn test_find_causal_chain_nonexistent() {
        let log = AnimationEventLog::new();
        let chain = log.find_causal_chain(999);
        assert_eq!(chain, vec![999]);
    }

    #[test]
    fn test_find_caused_by() {
        let mut log = AnimationEventLog::new();

        let root_id = log.emit(StateTransition::new(1, "idle", "attack", 0.1, 100));

        // Two events caused by root
        log.emit(AnimationNotify::new(1, "attack", "swing_start", 0.0, 110).with_causal_id(root_id));
        log.emit(AnimationNotify::new(1, "attack", "swing_hit", 0.3, 130).with_causal_id(root_id));

        // Unrelated event
        log.emit(AnimationNotify::new(2, "walk", "step", 0.0, 120));

        let caused = log.find_caused_by(root_id);
        assert_eq!(caused.len(), 2);
    }

    // ===== Iteration Tests =====

    #[test]
    fn test_event_log_iter() {
        let mut log = AnimationEventLog::new();
        log.emit(AnimationNotify::new(1, "a", "b", 0.0, 100));
        log.emit(StateTransition::new(2, "c", "d", 0.1, 200));

        let collected: Vec<_> = log.iter().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].0, 0); // ID
        assert_eq!(collected[0].1.entity_id(), 1);
        assert_eq!(collected[1].0, 1); // ID
        assert_eq!(collected[1].1.entity_id(), 2);
    }

    // ===== Statistics Tests =====

    #[test]
    fn test_count_by_kind() {
        let mut log = AnimationEventLog::new();
        log.emit(AnimationNotify::new(1, "a", "b", 0.0, 100));
        log.emit(AnimationNotify::new(2, "c", "d", 0.0, 200));
        log.emit(StateTransition::new(3, "e", "f", 0.1, 300));
        log.emit(RagdollActivated::new(4, 0.2, Vec3::X, Vec3::ZERO, 400));

        let counts = log.count_by_kind();
        assert_eq!(*counts.get(&AnimationEventKind::Notify).unwrap(), 2);
        assert_eq!(*counts.get(&AnimationEventKind::StateTransition).unwrap(), 1);
        assert_eq!(*counts.get(&AnimationEventKind::RagdollActivated).unwrap(), 1);
    }

    #[test]
    fn test_event_log_stats() {
        let mut log = AnimationEventLog::new();
        log.emit(AnimationNotify::new(1, "a", "b", 0.0, 100));
        log.emit(StateTransition::new(2, "c", "d", 0.1, 200));
        log.emit(RagdollActivated::new(3, 0.2, Vec3::X, Vec3::ZERO, 300));
        log.emit(CustomEvent::new(4, "custom", 400));

        let stats = log.stats();
        assert_eq!(stats.total_events, 4);
        assert_eq!(stats.notify_count, 1);
        assert_eq!(stats.transition_count, 1);
        assert_eq!(stats.ragdoll_count, 1);
        assert_eq!(stats.custom_count, 1);
        assert_eq!(stats.oldest_timestamp, 100);
        assert_eq!(stats.newest_timestamp, 400);
        assert_eq!(stats.causal_counter, 4);
    }

    #[test]
    fn test_event_log_stats_empty() {
        let log = AnimationEventLog::new();
        let stats = log.stats();
        assert_eq!(stats.total_events, 0);
        assert_eq!(stats.time_span(), 0);
    }

    #[test]
    fn test_event_log_stats_time_span() {
        let mut log = AnimationEventLog::new();
        log.emit(AnimationNotify::new(1, "a", "b", 0.0, 1000));
        log.emit(AnimationNotify::new(2, "c", "d", 0.0, 5000));

        let stats = log.stats();
        assert_eq!(stats.time_span(), 4000);
    }

    // ===== Pruning Tests =====

    #[test]
    fn test_event_log_prune_on_capacity() {
        let mut log = AnimationEventLog::with_capacity(10);

        // Fill to capacity
        for i in 0..10 {
            log.emit(AnimationNotify::new(i, "clip", "notify", 0.0, i));
        }
        assert_eq!(log.len(), 10);

        // Add one more - should trigger prune
        log.emit(AnimationNotify::new(10, "clip", "notify", 0.0, 10));

        // Should have pruned ~25% = 2-3 events
        assert!(log.len() < 10);
        assert!(log.len() >= 7);
    }

    #[test]
    fn test_event_log_causal_counter_survives_prune() {
        let mut log = AnimationEventLog::with_capacity(5);

        for i in 0..5 {
            log.emit(AnimationNotify::new(i, "a", "b", 0.0, i));
        }

        let counter_before = log.causal_counter();

        // Trigger prune
        log.emit(AnimationNotify::new(99, "a", "b", 0.0, 99));

        // Counter should have incremented
        assert!(log.causal_counter() > counter_before);
    }

    // ===== Downcast Helper Tests =====

    #[test]
    fn test_as_notify() {
        let notify = AnimationNotify::new(1, "walk", "step", 0.0, 100);
        let boxed: Box<dyn AnimationEventTrait> = Box::new(notify);

        let downcasted = as_notify(boxed.as_ref());
        assert!(downcasted.is_some());
        assert_eq!(downcasted.unwrap().clip_name, "walk");
    }

    #[test]
    fn test_as_notify_wrong_type() {
        let trans = StateTransition::new(1, "a", "b", 0.1, 100);
        let boxed: Box<dyn AnimationEventTrait> = Box::new(trans);

        let downcasted = as_notify(boxed.as_ref());
        assert!(downcasted.is_none());
    }

    #[test]
    fn test_as_state_transition() {
        let trans = StateTransition::new(5, "idle", "run", 0.25, 500);
        let boxed: Box<dyn AnimationEventTrait> = Box::new(trans);

        let downcasted = as_state_transition(boxed.as_ref());
        assert!(downcasted.is_some());
        assert_eq!(downcasted.unwrap().from_state, "idle");
        assert_eq!(downcasted.unwrap().to_state, "run");
    }

    #[test]
    fn test_as_ragdoll_activated() {
        let ragdoll = RagdollActivated::new(3, 0.5, Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO, 750);
        let boxed: Box<dyn AnimationEventTrait> = Box::new(ragdoll);

        let downcasted = as_ragdoll_activated(boxed.as_ref());
        assert!(downcasted.is_some());
        assert_eq!(downcasted.unwrap().blend_time, 0.5);
    }

    #[test]
    fn test_as_custom_event() {
        let event = CustomEvent::new(7, "special", 900).with_payload("data");
        let boxed: Box<dyn AnimationEventTrait> = Box::new(event);

        let downcasted = as_custom_event(boxed.as_ref());
        assert!(downcasted.is_some());
        assert_eq!(downcasted.unwrap().payload, Some("data".to_string()));
    }

    // ===== Debug Formatting Tests =====

    #[test]
    fn test_animation_notify_debug() {
        let notify = AnimationNotify::new(1, "walk", "step", 0.5, 100);
        let debug_str = format!("{:?}", notify);
        assert!(debug_str.contains("AnimationNotify"));
        assert!(debug_str.contains("walk"));
    }

    #[test]
    fn test_state_transition_debug() {
        let trans = StateTransition::new(1, "idle", "run", 0.2, 200);
        let debug_str = format!("{:?}", trans);
        assert!(debug_str.contains("StateTransition"));
        assert!(debug_str.contains("idle"));
        assert!(debug_str.contains("run"));
    }

    #[test]
    fn test_ragdoll_activated_debug() {
        let ragdoll = RagdollActivated::new(1, 0.3, Vec3::X, Vec3::Y, 300);
        let debug_str = format!("{:?}", ragdoll);
        assert!(debug_str.contains("RagdollActivated"));
    }

    #[test]
    fn test_event_log_debug() {
        let mut log = AnimationEventLog::new();
        log.emit(AnimationNotify::new(1, "a", "b", 0.0, 100));

        let debug_str = format!("{:?}", log);
        assert!(debug_str.contains("AnimationEventLog"));
        assert!(debug_str.contains("len"));
    }

    // ===== Serialization Tests =====

    #[test]
    fn test_animation_notify_serde() {
        let notify = AnimationNotify::new(42, "walk", "footstep", 0.5, 1000)
            .with_causal_id(5);

        let json = serde_json::to_string(&notify).unwrap();
        let recovered: AnimationNotify = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered, notify);
    }

    #[test]
    fn test_state_transition_serde() {
        let trans = StateTransition::new(10, "idle", "attack", 0.15, 2000);

        let json = serde_json::to_string(&trans).unwrap();
        let recovered: StateTransition = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered, trans);
    }

    #[test]
    fn test_ragdoll_activated_serde() {
        let ragdoll = RagdollActivated::new(
            99,
            0.3,
            Vec3::new(100.0, 50.0, 25.0),
            Vec3::new(1.0, 2.0, 3.0),
            3000,
        );

        let json = serde_json::to_string(&ragdoll).unwrap();
        let recovered: RagdollActivated = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered, ragdoll);
    }

    #[test]
    fn test_custom_event_serde() {
        let event = CustomEvent::new(5, "my_event", 500)
            .with_payload("test_data")
            .with_causal_id(3);

        let json = serde_json::to_string(&event).unwrap();
        let recovered: CustomEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered, event);
    }

    #[test]
    fn test_event_kind_serde() {
        let kinds = vec![
            AnimationEventKind::Notify,
            AnimationEventKind::StateTransition,
            AnimationEventKind::RagdollActivated,
            AnimationEventKind::Custom,
        ];

        for kind in kinds {
            let json = serde_json::to_string(&kind).unwrap();
            let recovered: AnimationEventKind = serde_json::from_str(&json).unwrap();
            assert_eq!(recovered, kind);
        }
    }

    // ===== Thread Safety Compilation Tests =====
    // These tests verify that the types implement Send + Sync as required

    #[test]
    fn test_animation_notify_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AnimationNotify>();
    }

    #[test]
    fn test_state_transition_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StateTransition>();
    }

    #[test]
    fn test_ragdoll_activated_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RagdollActivated>();
    }

    #[test]
    fn test_custom_event_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CustomEvent>();
    }
}
