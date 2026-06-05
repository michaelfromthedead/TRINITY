//! Animation state machine for runtime animation control (T-AN-5.2).
//!
//! This module provides a full-featured animation state machine system with:
//!
//! - **State definitions**: States with associated clips, blend trees, or sub-graphs
//! - **Transition system**: Conditional transitions with blend time and curves
//! - **Transition queue**: Queue transitions without interrupting current blends
//! - **Wildcard transitions**: "From any state" transitions for global interrupts
//! - **Condition evaluation**: Comparison and logical operators on parameters
//!
//! # Architecture
//!
//! ```text
//! StateMachine
//! +-- states: Vec<AnimationState>
//! |   +-- name: String
//! |   +-- content: StateContent (Clip|BlendTree|SubGraph)
//! |   +-- on_enter: Option<String> (callback name)
//! |   +-- on_exit: Option<String>
//! |   +-- speed_multiplier: f32
//! +-- transitions: Vec<Transition>
//! |   +-- from_state: Option<usize> (None = wildcard)
//! |   +-- to_state: usize
//! |   +-- condition: TransitionCondition
//! |   +-- blend_time: f32
//! |   +-- blend_curve: BlendCurve
//! |   +-- sync_mode: SyncMode
//! |   +-- priority: i32
//! +-- entry_state: usize
//! +-- current_state: usize
//! +-- transition_progress: Option<TransitionProgress>
//! +-- queued_transition: Option<usize>
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::state_machine::{
//!     StateMachine, AnimationState, StateContent,
//!     Transition, TransitionCondition, CompareOp,
//!     ParameterSet, ParameterValue,
//! };
//!
//! // Create states
//! let idle = AnimationState::new("idle", StateContent::Clip(0));
//! let walk = AnimationState::new("walk", StateContent::Clip(1));
//! let run = AnimationState::new("run", StateContent::Clip(2));
//!
//! // Create state machine
//! let mut sm = StateMachine::new();
//! sm.add_state(idle);
//! sm.add_state(walk);
//! sm.add_state(run);
//!
//! // Add transitions
//! sm.add_transition(Transition::new(Some(0), 1)
//!     .with_condition(TransitionCondition::parameter("speed", CompareOp::Greater, 0.1))
//!     .with_blend_time(0.2));
//!
//! sm.add_transition(Transition::new(Some(1), 2)
//!     .with_condition(TransitionCondition::parameter("speed", CompareOp::Greater, 5.0))
//!     .with_blend_time(0.3));
//!
//! // Wildcard transition (from any state to hurt)
//! sm.add_transition(Transition::wildcard(3) // hurt state
//!     .with_condition(TransitionCondition::parameter("damaged", CompareOp::Equal, true))
//!     .with_priority(100));
//!
//! // Update with parameters
//! let mut params = ParameterSet::new();
//! params.set_float("speed", 3.0);
//! sm.update(&params, 0.016);
//! ```

use serde::{Deserialize, Serialize};

use crate::animation_graph::{ComparisonOp, ParameterValue};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of states in a state machine.
pub const MAX_STATES: usize = 256;

/// Maximum number of transitions in a state machine.
pub const MAX_TRANSITIONS: usize = 1024;

/// Maximum transition queue depth.
pub const MAX_TRANSITION_QUEUE: usize = 4;

/// Default blend time for transitions (seconds).
pub const DEFAULT_BLEND_TIME: f32 = 0.2;

/// Priority for wildcard transitions (default).
pub const WILDCARD_BASE_PRIORITY: i32 = 1000;

// ---------------------------------------------------------------------------
// StateContent
// ---------------------------------------------------------------------------

/// The content type of an animation state.
///
/// A state can play a single clip, a blend tree, or delegate to a nested
/// state machine (sub-graph) for complex hierarchical behavior.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StateContent {
    /// Play a single animation clip by index.
    Clip(usize),

    /// Evaluate a blend tree by index.
    BlendTree(usize),

    /// Evaluate a nested state machine by index.
    SubGraph(usize),

    /// Empty state (useful for transitions or placeholder).
    Empty,
}

impl StateContent {
    /// Check if this is an empty state.
    #[inline]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Get the clip index if this is a clip state.
    #[inline]
    pub fn clip_index(&self) -> Option<usize> {
        match self {
            Self::Clip(idx) => Some(*idx),
            _ => None,
        }
    }

    /// Get the blend tree index if this is a blend tree state.
    #[inline]
    pub fn blend_tree_index(&self) -> Option<usize> {
        match self {
            Self::BlendTree(idx) => Some(*idx),
            _ => None,
        }
    }

    /// Get the sub-graph index if this is a sub-graph state.
    #[inline]
    pub fn sub_graph_index(&self) -> Option<usize> {
        match self {
            Self::SubGraph(idx) => Some(*idx),
            _ => None,
        }
    }
}

impl Default for StateContent {
    fn default() -> Self {
        Self::Empty
    }
}

// ---------------------------------------------------------------------------
// AnimationState
// ---------------------------------------------------------------------------

/// An animation state in the state machine.
///
/// States represent distinct animation behaviors (idle, walk, run, attack).
/// Each state has associated content (clip or blend tree) and optional
/// enter/exit callbacks for gameplay hooks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationState {
    /// Human-readable state name.
    pub name: String,

    /// The animation content for this state.
    pub clip_or_tree: StateContent,

    /// Callback name to invoke when entering this state.
    pub on_enter: Option<String>,

    /// Callback name to invoke when exiting this state.
    pub on_exit: Option<String>,

    /// Speed multiplier for animation playback in this state.
    pub speed_multiplier: f32,

    /// Whether this state loops its content.
    pub looping: bool,

    /// Tags for this state (e.g., "grounded", "airborne").
    pub tags: Vec<String>,
}

impl AnimationState {
    /// Create a new state with the given name and content.
    pub fn new(name: impl Into<String>, content: StateContent) -> Self {
        Self {
            name: name.into(),
            clip_or_tree: content,
            on_enter: None,
            on_exit: None,
            speed_multiplier: 1.0,
            looping: true,
            tags: Vec::new(),
        }
    }

    /// Create a clip state.
    #[inline]
    pub fn clip(name: impl Into<String>, clip_index: usize) -> Self {
        Self::new(name, StateContent::Clip(clip_index))
    }

    /// Create a blend tree state.
    #[inline]
    pub fn blend_tree(name: impl Into<String>, tree_index: usize) -> Self {
        Self::new(name, StateContent::BlendTree(tree_index))
    }

    /// Create a sub-graph state.
    #[inline]
    pub fn sub_graph(name: impl Into<String>, graph_index: usize) -> Self {
        Self::new(name, StateContent::SubGraph(graph_index))
    }

    /// Create an empty state.
    #[inline]
    pub fn empty(name: impl Into<String>) -> Self {
        Self::new(name, StateContent::Empty)
    }

    /// Set the enter callback.
    pub fn with_on_enter(mut self, callback: impl Into<String>) -> Self {
        self.on_enter = Some(callback.into());
        self
    }

    /// Set the exit callback.
    pub fn with_on_exit(mut self, callback: impl Into<String>) -> Self {
        self.on_exit = Some(callback.into());
        self
    }

    /// Set the speed multiplier.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed_multiplier = speed;
        self
    }

    /// Set whether this state loops.
    pub fn with_looping(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }

    /// Add a tag to this state.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Check if this state has a specific tag.
    #[inline]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

impl Default for AnimationState {
    fn default() -> Self {
        Self::empty("default")
    }
}

// ---------------------------------------------------------------------------
// CompareOp
// ---------------------------------------------------------------------------

/// Comparison operators for transition conditions.
///
/// These are used to compare parameter values against thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompareOp {
    /// Equal to.
    Equal,
    /// Not equal to.
    NotEqual,
    /// Less than.
    Less,
    /// Greater than.
    Greater,
    /// Less than or equal to.
    LessEqual,
    /// Greater than or equal to.
    GreaterEqual,
}

impl CompareOp {
    /// Get the operator symbol.
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::Less => "<",
            Self::Greater => ">",
            Self::LessEqual => "<=",
            Self::GreaterEqual => ">=",
        }
    }

    /// Evaluate this operator on two float values.
    #[inline]
    pub fn evaluate_float(&self, a: f32, b: f32) -> bool {
        match self {
            Self::Equal => (a - b).abs() < 1e-6,
            Self::NotEqual => (a - b).abs() >= 1e-6,
            Self::Less => a < b,
            Self::Greater => a > b,
            Self::LessEqual => a <= b,
            Self::GreaterEqual => a >= b,
        }
    }

    /// Evaluate this operator on two integer values.
    #[inline]
    pub fn evaluate_int(&self, a: i32, b: i32) -> bool {
        match self {
            Self::Equal => a == b,
            Self::NotEqual => a != b,
            Self::Less => a < b,
            Self::Greater => a > b,
            Self::LessEqual => a <= b,
            Self::GreaterEqual => a >= b,
        }
    }

    /// Evaluate this operator on two boolean values.
    #[inline]
    pub fn evaluate_bool(&self, a: bool, b: bool) -> bool {
        match self {
            Self::Equal => a == b,
            Self::NotEqual => a != b,
            // Boolean comparisons: true > false
            Self::Less => !a && b,
            Self::Greater => a && !b,
            Self::LessEqual => !a || b,
            Self::GreaterEqual => a || !b,
        }
    }
}

impl From<ComparisonOp> for CompareOp {
    fn from(op: ComparisonOp) -> Self {
        match op {
            ComparisonOp::Equal => Self::Equal,
            ComparisonOp::NotEqual => Self::NotEqual,
            ComparisonOp::Less => Self::Less,
            ComparisonOp::Greater => Self::Greater,
            ComparisonOp::LessEqual => Self::LessEqual,
            ComparisonOp::GreaterEqual => Self::GreaterEqual,
        }
    }
}

// ---------------------------------------------------------------------------
// TransitionCondition
// ---------------------------------------------------------------------------

/// A condition that must be satisfied for a transition to occur.
///
/// Conditions form a tree with parameter comparisons at the leaves
/// and logical operators (AND, OR, NOT) at the branches.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TransitionCondition {
    /// Always true. Used for unconditional transitions.
    Always,

    /// Never true. Used to disable transitions.
    Never,

    /// Compare a parameter against a value.
    Parameter {
        /// Parameter name to compare.
        name: String,
        /// Comparison operator.
        op: CompareOp,
        /// Value to compare against.
        value: ConditionValue,
    },

    /// Check if a trigger parameter has fired.
    Trigger {
        /// Trigger parameter name.
        name: String,
    },

    /// Check if the current state's animation has completed.
    /// Only valid for non-looping states.
    StateComplete,

    /// Check if current playback time exceeds a threshold.
    TimeExceeds {
        /// Time threshold in seconds.
        threshold: f32,
    },

    /// Logical AND: both conditions must be true.
    And(Box<TransitionCondition>, Box<TransitionCondition>),

    /// Logical OR: at least one condition must be true.
    Or(Box<TransitionCondition>, Box<TransitionCondition>),

    /// Logical NOT: inverts the condition.
    Not(Box<TransitionCondition>),
}

impl TransitionCondition {
    /// Create an always-true condition.
    #[inline]
    pub fn always() -> Self {
        Self::Always
    }

    /// Create a never-true condition.
    #[inline]
    pub fn never() -> Self {
        Self::Never
    }

    /// Create a parameter comparison condition.
    pub fn parameter(
        name: impl Into<String>,
        op: CompareOp,
        value: impl Into<ConditionValue>,
    ) -> Self {
        Self::Parameter {
            name: name.into(),
            op,
            value: value.into(),
        }
    }

    /// Create a float parameter comparison.
    pub fn float_param(name: impl Into<String>, op: CompareOp, threshold: f32) -> Self {
        Self::parameter(name, op, ConditionValue::Float(threshold))
    }

    /// Create an int parameter comparison.
    pub fn int_param(name: impl Into<String>, op: CompareOp, threshold: i32) -> Self {
        Self::parameter(name, op, ConditionValue::Int(threshold))
    }

    /// Create a bool parameter comparison.
    pub fn bool_param(name: impl Into<String>, value: bool) -> Self {
        Self::parameter(name, CompareOp::Equal, ConditionValue::Bool(value))
    }

    /// Create a trigger condition.
    pub fn trigger(name: impl Into<String>) -> Self {
        Self::Trigger { name: name.into() }
    }

    /// Create a state complete condition.
    #[inline]
    pub fn state_complete() -> Self {
        Self::StateComplete
    }

    /// Create a time exceeds condition.
    pub fn time_exceeds(threshold: f32) -> Self {
        Self::TimeExceeds { threshold }
    }

    /// Create an AND condition.
    pub fn and(a: TransitionCondition, b: TransitionCondition) -> Self {
        Self::And(Box::new(a), Box::new(b))
    }

    /// Create an OR condition.
    pub fn or(a: TransitionCondition, b: TransitionCondition) -> Self {
        Self::Or(Box::new(a), Box::new(b))
    }

    /// Create a NOT condition.
    pub fn not(condition: TransitionCondition) -> Self {
        Self::Not(Box::new(condition))
    }

    /// Evaluate this condition against a parameter set and state context.
    pub fn evaluate(&self, params: &ParameterSet, context: &EvaluationContext) -> bool {
        match self {
            Self::Always => true,
            Self::Never => false,

            Self::Parameter { name, op, value } => {
                if let Some(param_value) = params.get(name) {
                    match (param_value, value) {
                        (ParameterValue::Float(a), ConditionValue::Float(b)) => {
                            op.evaluate_float(*a, *b)
                        }
                        (ParameterValue::Int(a), ConditionValue::Int(b)) => {
                            op.evaluate_int(*a, *b)
                        }
                        (ParameterValue::Bool(a), ConditionValue::Bool(b)) => {
                            op.evaluate_bool(*a, *b)
                        }
                        // Type mismatch: try numeric conversion
                        (ParameterValue::Float(a), ConditionValue::Int(b)) => {
                            op.evaluate_float(*a, *b as f32)
                        }
                        (ParameterValue::Int(a), ConditionValue::Float(b)) => {
                            op.evaluate_float(*a as f32, *b)
                        }
                        _ => false, // Incompatible types
                    }
                } else {
                    false // Parameter not found
                }
            }

            Self::Trigger { name } => params.is_trigger_fired(name),

            Self::StateComplete => context.state_complete,

            Self::TimeExceeds { threshold } => context.current_time >= *threshold,

            Self::And(a, b) => a.evaluate(params, context) && b.evaluate(params, context),

            Self::Or(a, b) => a.evaluate(params, context) || b.evaluate(params, context),

            Self::Not(c) => !c.evaluate(params, context),
        }
    }

    /// Check if this condition is always true.
    #[inline]
    pub fn is_always(&self) -> bool {
        matches!(self, Self::Always)
    }

    /// Check if this condition requires a trigger.
    pub fn requires_trigger(&self) -> bool {
        match self {
            Self::Trigger { .. } => true,
            Self::And(a, b) | Self::Or(a, b) => a.requires_trigger() || b.requires_trigger(),
            Self::Not(c) => c.requires_trigger(),
            _ => false,
        }
    }
}

impl Default for TransitionCondition {
    fn default() -> Self {
        Self::Always
    }
}

// ---------------------------------------------------------------------------
// ConditionValue
// ---------------------------------------------------------------------------

/// A value used in transition conditions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ConditionValue {
    Float(f32),
    Int(i32),
    Bool(bool),
}

impl From<f32> for ConditionValue {
    fn from(v: f32) -> Self {
        Self::Float(v)
    }
}

impl From<i32> for ConditionValue {
    fn from(v: i32) -> Self {
        Self::Int(v)
    }
}

impl From<bool> for ConditionValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

// ---------------------------------------------------------------------------
// EvaluationContext
// ---------------------------------------------------------------------------

/// Context information for condition evaluation.
///
/// Provides state-specific information like completion status and timing.
#[derive(Clone, Debug, Default)]
pub struct EvaluationContext {
    /// Whether the current state's animation has completed.
    pub state_complete: bool,

    /// Current playback time in the state (seconds).
    pub current_time: f32,

    /// Normalized progress through the current clip (0.0 to 1.0).
    pub normalized_time: f32,

    /// Index of the current state.
    pub current_state: usize,
}

impl EvaluationContext {
    /// Create a new evaluation context.
    pub fn new(current_state: usize) -> Self {
        Self {
            state_complete: false,
            current_time: 0.0,
            normalized_time: 0.0,
            current_state,
        }
    }

    /// Set the completion status.
    pub fn with_complete(mut self, complete: bool) -> Self {
        self.state_complete = complete;
        self
    }

    /// Set the current time.
    pub fn with_time(mut self, time: f32, normalized: f32) -> Self {
        self.current_time = time;
        self.normalized_time = normalized;
        self
    }
}

// ---------------------------------------------------------------------------
// BlendCurve
// ---------------------------------------------------------------------------

/// Easing curve for transition blending.
///
/// Controls how the blend weight changes over the transition duration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlendCurve {
    /// Linear interpolation (constant rate).
    #[default]
    Linear,

    /// Ease in (slow start, fast end).
    EaseIn,

    /// Ease out (fast start, slow end).
    EaseOut,

    /// Ease in and out (slow start and end).
    EaseInOut,

    /// Smooth step (Hermite interpolation).
    SmoothStep,

    /// Smoother step (Ken Perlin's improved version).
    SmootherStep,

    /// Instant snap (no blending, immediate switch).
    Instant,
}

impl BlendCurve {
    /// Apply this curve to a normalized progress value (0.0 to 1.0).
    pub fn apply(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
            Self::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                }
            }
            Self::SmoothStep => t * t * (3.0 - 2.0 * t),
            Self::SmootherStep => t * t * t * (t * (t * 6.0 - 15.0) + 10.0),
            Self::Instant => if t > 0.0 { 1.0 } else { 0.0 },
        }
    }

    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::EaseIn => "Ease In",
            Self::EaseOut => "Ease Out",
            Self::EaseInOut => "Ease In/Out",
            Self::SmoothStep => "Smooth Step",
            Self::SmootherStep => "Smoother Step",
            Self::Instant => "Instant",
        }
    }
}

// ---------------------------------------------------------------------------
// SyncMode
// ---------------------------------------------------------------------------

/// Synchronization mode for transitions.
///
/// Controls how the source and destination animations are synchronized
/// during a transition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SyncMode {
    /// Freeze the source animation, start target from beginning.
    #[default]
    FreezeSource,

    /// Sync target animation to source's normalized time.
    SyncToSource,

    /// Start target from the beginning, source continues playing.
    Crossfade,

    /// Both animations continue independently.
    Independent,

    /// Snap source to target's foot phase.
    FootSync,
}

impl SyncMode {
    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::FreezeSource => "Freeze Source",
            Self::SyncToSource => "Sync to Source",
            Self::Crossfade => "Crossfade",
            Self::Independent => "Independent",
            Self::FootSync => "Foot Sync",
        }
    }
}

// ---------------------------------------------------------------------------
// Transition
// ---------------------------------------------------------------------------

/// A transition between states in the state machine.
///
/// Transitions define how the state machine moves between states,
/// including the conditions that trigger them and the blending behavior.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transition {
    /// Source state index. None means "from any state" (wildcard).
    pub from_state: Option<usize>,

    /// Destination state index.
    pub to_state: usize,

    /// Condition that must be satisfied for this transition.
    pub condition: TransitionCondition,

    /// Duration of the transition blend (seconds).
    pub blend_time: f32,

    /// Easing curve for the blend.
    pub blend_curve: BlendCurve,

    /// How to synchronize source and target animations.
    pub sync_mode: SyncMode,

    /// Priority for transition selection (higher = more priority).
    /// Wildcard transitions should have higher base priority.
    pub priority: i32,

    /// Whether this transition can interrupt an in-progress transition.
    pub can_interrupt: bool,

    /// Whether this transition consumes triggers it uses.
    pub consume_trigger: bool,

    /// Optional name for debugging.
    pub name: Option<String>,

    /// Whether this transition is enabled.
    pub enabled: bool,
}

impl Transition {
    /// Create a new transition from one state to another.
    ///
    /// # Arguments
    ///
    /// * `from_state` - Source state index, or None for wildcard.
    /// * `to_state` - Destination state index.
    pub fn new(from_state: Option<usize>, to_state: usize) -> Self {
        let priority = if from_state.is_none() {
            WILDCARD_BASE_PRIORITY
        } else {
            0
        };

        Self {
            from_state,
            to_state,
            condition: TransitionCondition::Always,
            blend_time: DEFAULT_BLEND_TIME,
            blend_curve: BlendCurve::Linear,
            sync_mode: SyncMode::FreezeSource,
            priority,
            can_interrupt: from_state.is_none(), // Wildcards can interrupt by default
            consume_trigger: true,
            name: None,
            enabled: true,
        }
    }

    /// Create a wildcard transition (from any state).
    ///
    /// Wildcard transitions are useful for global interrupts like
    /// damage reactions or death animations.
    pub fn wildcard(to_state: usize) -> Self {
        Self::new(None, to_state)
    }

    /// Create a direct transition between two specific states.
    pub fn direct(from_state: usize, to_state: usize) -> Self {
        Self::new(Some(from_state), to_state)
    }

    /// Set the transition condition.
    pub fn with_condition(mut self, condition: TransitionCondition) -> Self {
        self.condition = condition;
        self
    }

    /// Set the blend time.
    pub fn with_blend_time(mut self, time: f32) -> Self {
        self.blend_time = time.max(0.0);
        self
    }

    /// Set the blend curve.
    pub fn with_blend_curve(mut self, curve: BlendCurve) -> Self {
        self.blend_curve = curve;
        self
    }

    /// Set the sync mode.
    pub fn with_sync_mode(mut self, mode: SyncMode) -> Self {
        self.sync_mode = mode;
        self
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Enable interrupt capability.
    pub fn interruptible(mut self) -> Self {
        self.can_interrupt = true;
        self
    }

    /// Disable interrupt capability.
    pub fn non_interruptible(mut self) -> Self {
        self.can_interrupt = false;
        self
    }

    /// Set whether triggers are consumed.
    pub fn with_consume_trigger(mut self, consume: bool) -> Self {
        self.consume_trigger = consume;
        self
    }

    /// Set an optional name for debugging.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Disable this transition.
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Check if this is a wildcard transition.
    #[inline]
    pub fn is_wildcard(&self) -> bool {
        self.from_state.is_none()
    }

    /// Check if this transition applies from the given state.
    #[inline]
    pub fn applies_from(&self, state: usize) -> bool {
        match self.from_state {
            None => true, // Wildcard
            Some(from) => from == state,
        }
    }

    /// Check if this is a self-transition (same source and destination).
    #[inline]
    pub fn is_self_transition(&self) -> bool {
        match self.from_state {
            Some(from) => from == self.to_state,
            None => false, // Wildcards are never self-transitions
        }
    }

    /// Evaluate whether this transition should fire.
    pub fn should_fire(
        &self,
        current_state: usize,
        params: &ParameterSet,
        context: &EvaluationContext,
    ) -> bool {
        if !self.enabled {
            return false;
        }
        if !self.applies_from(current_state) {
            return false;
        }
        self.condition.evaluate(params, context)
    }
}

impl Default for Transition {
    fn default() -> Self {
        Self::new(Some(0), 0)
    }
}

// ---------------------------------------------------------------------------
// TransitionProgress
// ---------------------------------------------------------------------------

/// Progress of an in-flight transition.
///
/// Tracks the blending state between source and destination states.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionProgress {
    /// Index of the transition being executed.
    pub transition_index: usize,

    /// Source state index.
    pub from_state: usize,

    /// Destination state index.
    pub to_state: usize,

    /// Elapsed time since transition started (seconds).
    pub elapsed: f32,

    /// Total transition duration (seconds).
    pub duration: f32,

    /// The blend curve being used.
    pub blend_curve: BlendCurve,

    /// Source animation's time when transition started.
    pub source_start_time: f32,

    /// Target animation's start time.
    pub target_start_time: f32,

    /// Sync mode for this transition.
    pub sync_mode: SyncMode,
}

impl TransitionProgress {
    /// Create a new transition progress tracker.
    pub fn new(
        transition_index: usize,
        from_state: usize,
        to_state: usize,
        duration: f32,
        blend_curve: BlendCurve,
        sync_mode: SyncMode,
    ) -> Self {
        Self {
            transition_index,
            from_state,
            to_state,
            elapsed: 0.0,
            duration,
            blend_curve,
            source_start_time: 0.0,
            target_start_time: 0.0,
            sync_mode,
        }
    }

    /// Get the normalized progress (0.0 to 1.0).
    #[inline]
    pub fn normalized_progress(&self) -> f32 {
        if self.duration <= 0.0 {
            1.0
        } else {
            (self.elapsed / self.duration).clamp(0.0, 1.0)
        }
    }

    /// Get the blend weight after applying the curve.
    #[inline]
    pub fn blend_weight(&self) -> f32 {
        self.blend_curve.apply(self.normalized_progress())
    }

    /// Check if the transition is complete.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.elapsed >= self.duration
    }

    /// Advance the transition by delta time.
    ///
    /// Returns true if the transition completed this frame.
    pub fn advance(&mut self, dt: f32) -> bool {
        let was_complete = self.is_complete();
        self.elapsed += dt;
        !was_complete && self.is_complete()
    }
}

// ---------------------------------------------------------------------------
// ParameterSet
// ---------------------------------------------------------------------------

/// A set of animation parameters for state machine evaluation.
///
/// This is a simplified parameter container specifically for state machine
/// condition evaluation. For full animation graph parameters, see
/// `animation_graph::AnimationParameter`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ParameterSet {
    /// Named parameters with their values.
    parameters: std::collections::HashMap<String, ParameterValue>,

    /// Triggers that have fired this frame.
    fired_triggers: std::collections::HashSet<String>,
}

impl ParameterSet {
    /// Create an empty parameter set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a parameter value.
    pub fn set(&mut self, name: impl Into<String>, value: ParameterValue) {
        self.parameters.insert(name.into(), value);
    }

    /// Set a float parameter.
    pub fn set_float(&mut self, name: impl Into<String>, value: f32) {
        self.set(name, ParameterValue::Float(value));
    }

    /// Set an int parameter.
    pub fn set_int(&mut self, name: impl Into<String>, value: i32) {
        self.set(name, ParameterValue::Int(value));
    }

    /// Set a bool parameter.
    pub fn set_bool(&mut self, name: impl Into<String>, value: bool) {
        self.set(name, ParameterValue::Bool(value));
    }

    /// Fire a trigger parameter.
    pub fn fire_trigger(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.fired_triggers.insert(name.clone());
        self.parameters.insert(name, ParameterValue::Trigger);
    }

    /// Get a parameter value.
    pub fn get(&self, name: &str) -> Option<&ParameterValue> {
        self.parameters.get(name)
    }

    /// Get a float parameter value.
    pub fn get_float(&self, name: &str) -> Option<f32> {
        self.get(name).and_then(|v| v.as_float())
    }

    /// Get an int parameter value.
    pub fn get_int(&self, name: &str) -> Option<i32> {
        self.get(name).and_then(|v| v.as_int())
    }

    /// Get a bool parameter value.
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.get(name).and_then(|v| v.as_bool())
    }

    /// Check if a trigger has fired.
    pub fn is_trigger_fired(&self, name: &str) -> bool {
        self.fired_triggers.contains(name)
    }

    /// Reset all triggers (called at end of frame).
    pub fn reset_triggers(&mut self) {
        self.fired_triggers.clear();
    }

    /// Consume a specific trigger.
    pub fn consume_trigger(&mut self, name: &str) {
        self.fired_triggers.remove(name);
    }

    /// Clear all parameters.
    pub fn clear(&mut self) {
        self.parameters.clear();
        self.fired_triggers.clear();
    }

    /// Get the number of parameters.
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Check if there are no parameters.
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Iterate over all parameters.
    pub fn iter_params(&self) -> impl Iterator<Item = (&String, &ParameterValue)> {
        self.parameters.iter()
    }

    /// Iterate over all fired triggers.
    pub fn iter_triggers(&self) -> impl Iterator<Item = &String> {
        self.fired_triggers.iter()
    }

    /// Merge another ParameterSet into this one.
    /// Parameters from `other` take precedence.
    pub fn merge_from(&mut self, other: &ParameterSet) {
        for (name, value) in other.iter_params() {
            self.parameters.insert(name.clone(), value.clone());
        }
        for trigger in other.iter_triggers() {
            self.fired_triggers.insert(trigger.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// StateCallback
// ---------------------------------------------------------------------------

/// A callback that was triggered during state machine update.
///
/// These are collected during update and can be processed by the
/// gameplay system afterward.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateCallback {
    /// The callback name (as specified in AnimationState).
    pub name: String,

    /// The type of callback (enter or exit).
    pub callback_type: CallbackType,

    /// The state index that triggered this callback.
    pub state_index: usize,
}

/// Type of state callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallbackType {
    /// Called when entering a state.
    Enter,
    /// Called when exiting a state.
    Exit,
}

// ---------------------------------------------------------------------------
// StateMachine
// ---------------------------------------------------------------------------

/// Animation state machine for controlling animation flow.
///
/// The state machine manages transitions between animation states based
/// on parameter conditions, supporting complex animation behaviors like
/// locomotion systems, combat systems, and character controllers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateMachine {
    /// All states in the state machine.
    pub states: Vec<AnimationState>,

    /// All transitions between states.
    pub transitions: Vec<Transition>,

    /// Index of the entry state (where the machine starts).
    pub entry_state: usize,

    /// Index of the current active state.
    pub current_state: usize,

    /// In-progress transition, if any.
    pub transition_progress: Option<TransitionProgress>,

    /// Queued transitions waiting for current transition to complete.
    queued_transitions: Vec<usize>,

    /// Time spent in the current state (seconds).
    state_time: f32,

    /// Callbacks triggered during the last update.
    #[serde(skip)]
    pending_callbacks: Vec<StateCallback>,

    /// Optional name for debugging.
    pub name: Option<String>,

    /// Whether the state machine is enabled.
    pub enabled: bool,
}

impl StateMachine {
    /// Create a new empty state machine.
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            transitions: Vec::new(),
            entry_state: 0,
            current_state: 0,
            transition_progress: None,
            queued_transitions: Vec::new(),
            state_time: 0.0,
            pending_callbacks: Vec::new(),
            name: None,
            enabled: true,
        }
    }

    /// Create a state machine with a name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    // -------------------------------------------------------------------------
    // State Management
    // -------------------------------------------------------------------------

    /// Add a state to the state machine.
    ///
    /// Returns the index of the added state.
    pub fn add_state(&mut self, state: AnimationState) -> usize {
        let index = self.states.len();
        self.states.push(state);
        index
    }

    /// Get a state by index.
    pub fn get_state(&self, index: usize) -> Option<&AnimationState> {
        self.states.get(index)
    }

    /// Get a mutable state by index.
    pub fn get_state_mut(&mut self, index: usize) -> Option<&mut AnimationState> {
        self.states.get_mut(index)
    }

    /// Find a state by name.
    pub fn find_state(&self, name: &str) -> Option<usize> {
        self.states.iter().position(|s| s.name == name)
    }

    /// Get the current state.
    pub fn get_current_state(&self) -> &AnimationState {
        &self.states[self.current_state]
    }

    /// Get the number of states.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Set the entry state by index.
    pub fn set_entry_state(&mut self, index: usize) {
        if index < self.states.len() {
            self.entry_state = index;
        }
    }

    /// Reset to the entry state.
    pub fn reset(&mut self) {
        // Exit current state
        if let Some(on_exit) = &self.states[self.current_state].on_exit {
            self.pending_callbacks.push(StateCallback {
                name: on_exit.clone(),
                callback_type: CallbackType::Exit,
                state_index: self.current_state,
            });
        }

        self.current_state = self.entry_state;
        self.state_time = 0.0;
        self.transition_progress = None;
        self.queued_transitions.clear();

        // Enter entry state
        if let Some(on_enter) = &self.states[self.entry_state].on_enter {
            self.pending_callbacks.push(StateCallback {
                name: on_enter.clone(),
                callback_type: CallbackType::Enter,
                state_index: self.entry_state,
            });
        }
    }

    // -------------------------------------------------------------------------
    // Transition Management
    // -------------------------------------------------------------------------

    /// Add a transition to the state machine.
    ///
    /// Returns the index of the added transition.
    pub fn add_transition(&mut self, transition: Transition) -> usize {
        let index = self.transitions.len();
        self.transitions.push(transition);
        index
    }

    /// Get a transition by index.
    pub fn get_transition(&self, index: usize) -> Option<&Transition> {
        self.transitions.get(index)
    }

    /// Get the number of transitions.
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// Check if currently transitioning.
    pub fn is_transitioning(&self) -> bool {
        self.transition_progress.is_some()
    }

    /// Get the current transition progress, if any.
    pub fn current_transition(&self) -> Option<&TransitionProgress> {
        self.transition_progress.as_ref()
    }

    /// Get the blend weight if transitioning (0.0 = source, 1.0 = target).
    pub fn transition_blend_weight(&self) -> f32 {
        self.transition_progress
            .as_ref()
            .map(|p| p.blend_weight())
            .unwrap_or(0.0)
    }

    // -------------------------------------------------------------------------
    // Transition Evaluation
    // -------------------------------------------------------------------------

    /// Evaluate all transitions and find the best one to take.
    ///
    /// Returns the index of the highest-priority valid transition.
    pub fn evaluate_transitions(&self, params: &ParameterSet) -> Option<usize> {
        let context = EvaluationContext::new(self.current_state)
            .with_time(self.state_time, 0.0);

        let mut best_transition: Option<(usize, i32)> = None;

        for (i, transition) in self.transitions.iter().enumerate() {
            if !transition.enabled {
                continue;
            }

            // Check if we're in a transition and this one can't interrupt
            if self.is_transitioning() && !transition.can_interrupt {
                continue;
            }

            // Check if this transition applies from current state
            if !transition.applies_from(self.current_state) {
                continue;
            }

            // Evaluate the condition
            if !transition.condition.evaluate(params, &context) {
                continue;
            }

            // Compare priorities
            match best_transition {
                None => best_transition = Some((i, transition.priority)),
                Some((_, best_priority)) if transition.priority > best_priority => {
                    best_transition = Some((i, transition.priority));
                }
                _ => {}
            }
        }

        best_transition.map(|(idx, _)| idx)
    }

    /// Start a transition by index.
    ///
    /// Returns true if the transition was started.
    pub fn start_transition(&mut self, transition_idx: usize) -> bool {
        let transition = match self.transitions.get(transition_idx) {
            Some(t) => t.clone(),
            None => return false,
        };

        // Can't start if we're transitioning and this can't interrupt
        if self.is_transitioning() && !transition.can_interrupt {
            // Queue it instead
            if self.queued_transitions.len() < MAX_TRANSITION_QUEUE {
                self.queued_transitions.push(transition_idx);
            }
            return false;
        }

        // Exit callbacks for current state (if not already transitioning)
        if !self.is_transitioning() {
            if let Some(on_exit) = &self.states[self.current_state].on_exit {
                self.pending_callbacks.push(StateCallback {
                    name: on_exit.clone(),
                    callback_type: CallbackType::Exit,
                    state_index: self.current_state,
                });
            }
        }

        // Create transition progress
        self.transition_progress = Some(TransitionProgress::new(
            transition_idx,
            self.current_state,
            transition.to_state,
            transition.blend_time,
            transition.blend_curve,
            transition.sync_mode,
        ));

        true
    }

    /// Force an immediate state change without blending.
    ///
    /// Useful for teleporting or respawning where blending doesn't make sense.
    pub fn force_state(&mut self, state_index: usize) {
        if state_index >= self.states.len() {
            return;
        }

        // Exit current state
        if let Some(on_exit) = &self.states[self.current_state].on_exit {
            self.pending_callbacks.push(StateCallback {
                name: on_exit.clone(),
                callback_type: CallbackType::Exit,
                state_index: self.current_state,
            });
        }

        // Clear any in-progress transition
        self.transition_progress = None;
        self.queued_transitions.clear();

        // Enter new state
        self.current_state = state_index;
        self.state_time = 0.0;

        if let Some(on_enter) = &self.states[state_index].on_enter {
            self.pending_callbacks.push(StateCallback {
                name: on_enter.clone(),
                callback_type: CallbackType::Enter,
                state_index,
            });
        }
    }

    // -------------------------------------------------------------------------
    // Update
    // -------------------------------------------------------------------------

    /// Update the state machine with the given parameters and delta time.
    ///
    /// This evaluates transitions, advances any in-progress transition,
    /// and updates state timing.
    pub fn update(&mut self, params: &ParameterSet, dt: f32) {
        if !self.enabled || self.states.is_empty() {
            return;
        }

        // Advance state time
        self.state_time += dt;

        // If we have an in-progress transition, advance it
        if let Some(ref mut progress) = self.transition_progress {
            let completed = progress.advance(dt);

            if completed {
                // Transition complete - enter new state
                let to_state = progress.to_state;
                self.current_state = to_state;
                self.state_time = 0.0;
                self.transition_progress = None;

                // Enter callback
                if let Some(on_enter) = &self.states[to_state].on_enter {
                    self.pending_callbacks.push(StateCallback {
                        name: on_enter.clone(),
                        callback_type: CallbackType::Enter,
                        state_index: to_state,
                    });
                }

                // Process queued transitions
                if let Some(queued_idx) = self.queued_transitions.pop() {
                    self.start_transition(queued_idx);
                }
            }
        } else {
            // No transition in progress - evaluate for new transitions
            if let Some(transition_idx) = self.evaluate_transitions(params) {
                self.start_transition(transition_idx);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Callbacks
    // -------------------------------------------------------------------------

    /// Drain pending callbacks from the last update.
    pub fn drain_callbacks(&mut self) -> Vec<StateCallback> {
        std::mem::take(&mut self.pending_callbacks)
    }

    /// Check if there are pending callbacks.
    pub fn has_pending_callbacks(&self) -> bool {
        !self.pending_callbacks.is_empty()
    }

    // -------------------------------------------------------------------------
    // Querying
    // -------------------------------------------------------------------------

    /// Get the time spent in the current state.
    pub fn state_time(&self) -> f32 {
        self.state_time
    }

    /// Get all transitions from a specific state.
    pub fn transitions_from(&self, state: usize) -> impl Iterator<Item = (usize, &Transition)> {
        self.transitions
            .iter()
            .enumerate()
            .filter(move |(_, t)| t.applies_from(state))
    }

    /// Get all wildcard transitions.
    pub fn wildcard_transitions(&self) -> impl Iterator<Item = (usize, &Transition)> {
        self.transitions
            .iter()
            .enumerate()
            .filter(|(_, t)| t.is_wildcard())
    }

    /// Check if a direct path exists between two states.
    pub fn has_path(&self, from: usize, to: usize) -> bool {
        self.transitions
            .iter()
            .any(|t| t.from_state == Some(from) && t.to_state == to)
    }

    /// Get the destination state if currently transitioning.
    pub fn transition_target(&self) -> Option<usize> {
        self.transition_progress.as_ref().map(|p| p.to_state)
    }

    // -------------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------------

    /// Validate the state machine for common issues.
    ///
    /// Returns a list of warnings/errors found.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.states.is_empty() {
            issues.push("State machine has no states".to_string());
            return issues;
        }

        if self.entry_state >= self.states.len() {
            issues.push(format!(
                "Entry state index {} is out of bounds (max {})",
                self.entry_state,
                self.states.len() - 1
            ));
        }

        // Check transitions reference valid states
        for (i, t) in self.transitions.iter().enumerate() {
            if let Some(from) = t.from_state {
                if from >= self.states.len() {
                    issues.push(format!(
                        "Transition {} references invalid from_state {}",
                        i, from
                    ));
                }
            }
            if t.to_state >= self.states.len() {
                issues.push(format!(
                    "Transition {} references invalid to_state {}",
                    i, t.to_state
                ));
            }
        }

        // Check for unreachable states
        let mut reachable = vec![false; self.states.len()];
        reachable[self.entry_state] = true;
        let mut changed = true;
        while changed {
            changed = false;
            for t in &self.transitions {
                let from_reachable = match t.from_state {
                    Some(from) => reachable.get(from).copied().unwrap_or(false),
                    None => reachable.iter().any(|&r| r), // Wildcard: any reachable state
                };
                if from_reachable {
                    if let Some(r) = reachable.get_mut(t.to_state) {
                        if !*r {
                            *r = true;
                            changed = true;
                        }
                    }
                }
            }
        }

        for (i, &r) in reachable.iter().enumerate() {
            if !r {
                issues.push(format!(
                    "State '{}' (index {}) is unreachable",
                    self.states[i].name, i
                ));
            }
        }

        // Check for dead-end states (no outgoing transitions)
        for (i, state) in self.states.iter().enumerate() {
            let has_outgoing = self
                .transitions
                .iter()
                .any(|t| t.applies_from(i) || t.is_wildcard());
            if !has_outgoing {
                issues.push(format!(
                    "State '{}' (index {}) has no outgoing transitions",
                    state.name, i
                ));
            }
        }

        issues
    }
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StateMachineBuilder
// ---------------------------------------------------------------------------

/// Builder for creating state machines with a fluent API.
pub struct StateMachineBuilder {
    machine: StateMachine,
}

impl StateMachineBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            machine: StateMachine::new(),
        }
    }

    /// Set the machine name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.machine.name = Some(name.into());
        self
    }

    /// Add a state.
    pub fn state(mut self, state: AnimationState) -> Self {
        self.machine.add_state(state);
        self
    }

    /// Add a clip state.
    pub fn clip_state(mut self, name: impl Into<String>, clip_index: usize) -> Self {
        self.machine.add_state(AnimationState::clip(name, clip_index));
        self
    }

    /// Add a transition.
    pub fn transition(mut self, transition: Transition) -> Self {
        self.machine.add_transition(transition);
        self
    }

    /// Add a direct transition between states.
    pub fn direct_transition(
        mut self,
        from: usize,
        to: usize,
        condition: TransitionCondition,
    ) -> Self {
        self.machine
            .add_transition(Transition::direct(from, to).with_condition(condition));
        self
    }

    /// Add a wildcard transition.
    pub fn wildcard_transition(mut self, to: usize, condition: TransitionCondition) -> Self {
        self.machine
            .add_transition(Transition::wildcard(to).with_condition(condition));
        self
    }

    /// Set the entry state.
    pub fn entry_state(mut self, index: usize) -> Self {
        self.machine.entry_state = index;
        self
    }

    /// Build the state machine.
    pub fn build(self) -> StateMachine {
        self.machine
    }
}

impl Default for StateMachineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // State Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_state_creation() {
        let state = AnimationState::clip("idle", 0);
        assert_eq!(state.name, "idle");
        assert_eq!(state.clip_or_tree, StateContent::Clip(0));
        assert_eq!(state.speed_multiplier, 1.0);
        assert!(state.looping);
    }

    #[test]
    fn test_state_with_callbacks() {
        let state = AnimationState::clip("attack", 5)
            .with_on_enter("on_attack_start")
            .with_on_exit("on_attack_end")
            .with_speed(1.5)
            .with_looping(false);

        assert_eq!(state.on_enter, Some("on_attack_start".to_string()));
        assert_eq!(state.on_exit, Some("on_attack_end".to_string()));
        assert_eq!(state.speed_multiplier, 1.5);
        assert!(!state.looping);
    }

    #[test]
    fn test_state_tags() {
        let state = AnimationState::clip("jump", 3)
            .with_tag("airborne")
            .with_tag("locomotion");

        assert!(state.has_tag("airborne"));
        assert!(state.has_tag("locomotion"));
        assert!(!state.has_tag("grounded"));
    }

    #[test]
    fn test_state_content_types() {
        assert_eq!(StateContent::Clip(0).clip_index(), Some(0));
        assert_eq!(StateContent::BlendTree(1).blend_tree_index(), Some(1));
        assert_eq!(StateContent::SubGraph(2).sub_graph_index(), Some(2));
        assert!(StateContent::Empty.is_empty());
    }

    #[test]
    fn test_state_lookup() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));
        sm.add_state(AnimationState::clip("walk", 1));
        sm.add_state(AnimationState::clip("run", 2));

        assert_eq!(sm.find_state("idle"), Some(0));
        assert_eq!(sm.find_state("walk"), Some(1));
        assert_eq!(sm.find_state("run"), Some(2));
        assert_eq!(sm.find_state("nonexistent"), None);
    }

    // -------------------------------------------------------------------------
    // CompareOp Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compare_op_float() {
        assert!(CompareOp::Equal.evaluate_float(1.0, 1.0));
        assert!(!CompareOp::Equal.evaluate_float(1.0, 2.0));
        assert!(CompareOp::NotEqual.evaluate_float(1.0, 2.0));
        assert!(CompareOp::Less.evaluate_float(1.0, 2.0));
        assert!(!CompareOp::Less.evaluate_float(2.0, 1.0));
        assert!(CompareOp::Greater.evaluate_float(2.0, 1.0));
        assert!(CompareOp::LessEqual.evaluate_float(1.0, 1.0));
        assert!(CompareOp::LessEqual.evaluate_float(1.0, 2.0));
        assert!(CompareOp::GreaterEqual.evaluate_float(2.0, 1.0));
        assert!(CompareOp::GreaterEqual.evaluate_float(1.0, 1.0));
    }

    #[test]
    fn test_compare_op_int() {
        assert!(CompareOp::Equal.evaluate_int(5, 5));
        assert!(CompareOp::NotEqual.evaluate_int(5, 6));
        assert!(CompareOp::Less.evaluate_int(3, 5));
        assert!(CompareOp::Greater.evaluate_int(7, 5));
    }

    #[test]
    fn test_compare_op_bool() {
        assert!(CompareOp::Equal.evaluate_bool(true, true));
        assert!(CompareOp::Equal.evaluate_bool(false, false));
        assert!(CompareOp::NotEqual.evaluate_bool(true, false));
        // true > false
        assert!(CompareOp::Greater.evaluate_bool(true, false));
        assert!(!CompareOp::Greater.evaluate_bool(false, true));
    }

    // -------------------------------------------------------------------------
    // Condition Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_condition_always() {
        let cond = TransitionCondition::always();
        let params = ParameterSet::new();
        let ctx = EvaluationContext::default();
        assert!(cond.evaluate(&params, &ctx));
    }

    #[test]
    fn test_condition_never() {
        let cond = TransitionCondition::never();
        let params = ParameterSet::new();
        let ctx = EvaluationContext::default();
        assert!(!cond.evaluate(&params, &ctx));
    }

    #[test]
    fn test_condition_float_param() {
        let cond = TransitionCondition::float_param("speed", CompareOp::Greater, 5.0);
        let ctx = EvaluationContext::default();

        let mut params = ParameterSet::new();
        params.set_float("speed", 3.0);
        assert!(!cond.evaluate(&params, &ctx));

        params.set_float("speed", 7.0);
        assert!(cond.evaluate(&params, &ctx));
    }

    #[test]
    fn test_condition_bool_param() {
        let cond = TransitionCondition::bool_param("is_grounded", true);
        let ctx = EvaluationContext::default();

        let mut params = ParameterSet::new();
        params.set_bool("is_grounded", false);
        assert!(!cond.evaluate(&params, &ctx));

        params.set_bool("is_grounded", true);
        assert!(cond.evaluate(&params, &ctx));
    }

    #[test]
    fn test_condition_trigger() {
        let cond = TransitionCondition::trigger("jump");
        let ctx = EvaluationContext::default();

        let mut params = ParameterSet::new();
        assert!(!cond.evaluate(&params, &ctx));

        params.fire_trigger("jump");
        assert!(cond.evaluate(&params, &ctx));
    }

    #[test]
    fn test_condition_and() {
        let cond = TransitionCondition::and(
            TransitionCondition::float_param("speed", CompareOp::Greater, 0.0),
            TransitionCondition::bool_param("is_grounded", true),
        );
        let ctx = EvaluationContext::default();

        let mut params = ParameterSet::new();
        params.set_float("speed", 5.0);
        params.set_bool("is_grounded", false);
        assert!(!cond.evaluate(&params, &ctx)); // speed > 0 but not grounded

        params.set_bool("is_grounded", true);
        assert!(cond.evaluate(&params, &ctx)); // both true
    }

    #[test]
    fn test_condition_or() {
        let cond = TransitionCondition::or(
            TransitionCondition::trigger("jump"),
            TransitionCondition::bool_param("force_jump", true),
        );
        let ctx = EvaluationContext::default();

        let mut params = ParameterSet::new();
        assert!(!cond.evaluate(&params, &ctx));

        params.fire_trigger("jump");
        assert!(cond.evaluate(&params, &ctx));

        params.reset_triggers();
        params.set_bool("force_jump", true);
        assert!(cond.evaluate(&params, &ctx));
    }

    #[test]
    fn test_condition_not() {
        let cond = TransitionCondition::not(TransitionCondition::bool_param("is_dead", true));
        let ctx = EvaluationContext::default();

        let mut params = ParameterSet::new();
        params.set_bool("is_dead", false);
        assert!(cond.evaluate(&params, &ctx));

        params.set_bool("is_dead", true);
        assert!(!cond.evaluate(&params, &ctx));
    }

    #[test]
    fn test_condition_state_complete() {
        let cond = TransitionCondition::state_complete();
        let mut params = ParameterSet::new();

        let ctx_incomplete = EvaluationContext::new(0).with_complete(false);
        assert!(!cond.evaluate(&params, &ctx_incomplete));

        let ctx_complete = EvaluationContext::new(0).with_complete(true);
        assert!(cond.evaluate(&mut params, &ctx_complete));
    }

    #[test]
    fn test_condition_time_exceeds() {
        let cond = TransitionCondition::time_exceeds(2.0);
        let params = ParameterSet::new();

        let ctx_early = EvaluationContext::new(0).with_time(1.5, 0.5);
        assert!(!cond.evaluate(&params, &ctx_early));

        let ctx_late = EvaluationContext::new(0).with_time(2.5, 0.8);
        assert!(cond.evaluate(&params, &ctx_late));
    }

    #[test]
    fn test_complex_condition() {
        // ((speed > 5) AND is_grounded) OR jump_trigger
        let cond = TransitionCondition::or(
            TransitionCondition::and(
                TransitionCondition::float_param("speed", CompareOp::Greater, 5.0),
                TransitionCondition::bool_param("is_grounded", true),
            ),
            TransitionCondition::trigger("jump"),
        );
        let ctx = EvaluationContext::default();

        let mut params = ParameterSet::new();
        params.set_float("speed", 3.0);
        params.set_bool("is_grounded", true);
        assert!(!cond.evaluate(&params, &ctx)); // speed too low

        params.set_float("speed", 7.0);
        assert!(cond.evaluate(&params, &ctx)); // speed high + grounded

        params.set_bool("is_grounded", false);
        assert!(!cond.evaluate(&params, &ctx)); // not grounded

        params.fire_trigger("jump");
        assert!(cond.evaluate(&params, &ctx)); // trigger fires regardless
    }

    // -------------------------------------------------------------------------
    // BlendCurve Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_blend_curve_linear() {
        let curve = BlendCurve::Linear;
        assert!((curve.apply(0.0) - 0.0).abs() < 1e-6);
        assert!((curve.apply(0.5) - 0.5).abs() < 1e-6);
        assert!((curve.apply(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_curve_ease_in() {
        let curve = BlendCurve::EaseIn;
        assert!((curve.apply(0.0) - 0.0).abs() < 1e-6);
        assert!(curve.apply(0.5) < 0.5); // Slow start
        assert!((curve.apply(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_curve_ease_out() {
        let curve = BlendCurve::EaseOut;
        assert!((curve.apply(0.0) - 0.0).abs() < 1e-6);
        assert!(curve.apply(0.5) > 0.5); // Fast start
        assert!((curve.apply(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_curve_smooth_step() {
        let curve = BlendCurve::SmoothStep;
        assert!((curve.apply(0.0) - 0.0).abs() < 1e-6);
        assert!((curve.apply(0.5) - 0.5).abs() < 1e-6); // Symmetric
        assert!((curve.apply(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_curve_instant() {
        let curve = BlendCurve::Instant;
        assert!((curve.apply(0.0) - 0.0).abs() < 1e-6);
        assert!((curve.apply(0.01) - 1.0).abs() < 1e-6);
        assert!((curve.apply(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_curve_clamping() {
        let curve = BlendCurve::Linear;
        assert!((curve.apply(-0.5) - 0.0).abs() < 1e-6);
        assert!((curve.apply(1.5) - 1.0).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // Transition Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transition_creation() {
        let t = Transition::direct(0, 1);
        assert_eq!(t.from_state, Some(0));
        assert_eq!(t.to_state, 1);
        assert!(!t.is_wildcard());
        assert!(t.enabled);
    }

    #[test]
    fn test_wildcard_transition() {
        let t = Transition::wildcard(5);
        assert_eq!(t.from_state, None);
        assert_eq!(t.to_state, 5);
        assert!(t.is_wildcard());
        assert!(t.can_interrupt); // Wildcards can interrupt by default
        assert_eq!(t.priority, WILDCARD_BASE_PRIORITY);
    }

    #[test]
    fn test_transition_applies_from() {
        let direct = Transition::direct(2, 3);
        assert!(direct.applies_from(2));
        assert!(!direct.applies_from(0));
        assert!(!direct.applies_from(3));

        let wildcard = Transition::wildcard(5);
        assert!(wildcard.applies_from(0));
        assert!(wildcard.applies_from(5));
        assert!(wildcard.applies_from(100));
    }

    #[test]
    fn test_self_transition() {
        let self_trans = Transition::direct(3, 3);
        assert!(self_trans.is_self_transition());

        let normal = Transition::direct(3, 4);
        assert!(!normal.is_self_transition());

        let wildcard = Transition::wildcard(3);
        assert!(!wildcard.is_self_transition());
    }

    #[test]
    fn test_transition_priority() {
        let low = Transition::direct(0, 1).with_priority(10);
        let high = Transition::direct(0, 1).with_priority(100);
        assert!(high.priority > low.priority);
    }

    // -------------------------------------------------------------------------
    // TransitionProgress Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transition_progress() {
        let mut progress = TransitionProgress::new(0, 0, 1, 1.0, BlendCurve::Linear, SyncMode::FreezeSource);

        assert!(!progress.is_complete());
        assert!((progress.blend_weight() - 0.0).abs() < 1e-6);

        progress.advance(0.5);
        assert!(!progress.is_complete());
        assert!((progress.blend_weight() - 0.5).abs() < 1e-6);

        let completed = progress.advance(0.6);
        assert!(completed);
        assert!(progress.is_complete());
        assert!((progress.blend_weight() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_transition_progress_with_curve() {
        let mut progress =
            TransitionProgress::new(0, 0, 1, 1.0, BlendCurve::EaseIn, SyncMode::FreezeSource);

        progress.elapsed = 0.5;
        let weight = progress.blend_weight();
        // EaseIn: weight should be less than linear (0.5)
        assert!(weight < 0.5);
    }

    // -------------------------------------------------------------------------
    // StateMachine Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_state_machine_creation() {
        let sm = StateMachine::new();
        assert_eq!(sm.state_count(), 0);
        assert_eq!(sm.transition_count(), 0);
        assert!(!sm.is_transitioning());
    }

    #[test]
    fn test_add_states() {
        let mut sm = StateMachine::new();
        let idx0 = sm.add_state(AnimationState::clip("idle", 0));
        let idx1 = sm.add_state(AnimationState::clip("walk", 1));

        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(sm.state_count(), 2);
        assert_eq!(sm.get_state(0).unwrap().name, "idle");
        assert_eq!(sm.get_state(1).unwrap().name, "walk");
    }

    #[test]
    fn test_simple_transition() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));
        sm.add_state(AnimationState::clip("walk", 1));

        sm.add_transition(
            Transition::direct(0, 1).with_condition(TransitionCondition::float_param(
                "speed",
                CompareOp::Greater,
                0.1,
            )),
        );

        let mut params = ParameterSet::new();
        params.set_float("speed", 0.0);

        // Should stay in idle
        sm.update(&params, 0.016);
        assert_eq!(sm.current_state, 0);
        assert!(!sm.is_transitioning());

        // Should start transition to walk
        params.set_float("speed", 1.0);
        sm.update(&params, 0.016);
        assert!(sm.is_transitioning());
        assert_eq!(sm.transition_target(), Some(1));

        // Complete the transition
        sm.update(&params, 1.0);
        assert!(!sm.is_transitioning());
        assert_eq!(sm.current_state, 1);
    }

    #[test]
    fn test_conditional_transition() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("walk", 0));
        sm.add_state(AnimationState::clip("run", 1));

        sm.add_transition(
            Transition::direct(0, 1)
                .with_condition(TransitionCondition::and(
                    TransitionCondition::float_param("speed", CompareOp::Greater, 5.0),
                    TransitionCondition::bool_param("is_grounded", true),
                ))
                .with_blend_time(0.3),
        );

        let mut params = ParameterSet::new();
        params.set_float("speed", 7.0);
        params.set_bool("is_grounded", false);

        // Condition not met (not grounded)
        sm.update(&params, 0.016);
        assert!(!sm.is_transitioning());

        // Now grounded
        params.set_bool("is_grounded", true);
        sm.update(&params, 0.016);
        assert!(sm.is_transitioning());
    }

    #[test]
    fn test_wildcard_transition_any_to_b() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));
        sm.add_state(AnimationState::clip("walk", 1));
        sm.add_state(AnimationState::clip("hurt", 2));

        // Wildcard: any state -> hurt when damaged
        sm.add_transition(
            Transition::wildcard(2)
                .with_condition(TransitionCondition::trigger("damaged"))
                .with_priority(1000)
                .with_blend_time(0.1),
        );

        let mut params = ParameterSet::new();

        // Start in idle
        assert_eq!(sm.current_state, 0);

        // Trigger damage
        params.fire_trigger("damaged");
        sm.update(&params, 0.016);

        assert!(sm.is_transitioning());
        assert_eq!(sm.transition_target(), Some(2));
    }

    #[test]
    fn test_state_machine_transition_priority() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));
        sm.add_state(AnimationState::clip("walk", 1));
        sm.add_state(AnimationState::clip("run", 2));

        // Low priority: idle -> walk
        sm.add_transition(
            Transition::direct(0, 1)
                .with_condition(TransitionCondition::always())
                .with_priority(10),
        );

        // High priority: idle -> run
        sm.add_transition(
            Transition::direct(0, 2)
                .with_condition(TransitionCondition::always())
                .with_priority(100),
        );

        let params = ParameterSet::new();
        sm.update(&params, 0.016);

        // Should pick the higher priority (run)
        assert!(sm.is_transitioning());
        assert_eq!(sm.transition_target(), Some(2));
    }

    #[test]
    fn test_transition_queue_dont_interrupt() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));
        sm.add_state(AnimationState::clip("walk", 1));
        sm.add_state(AnimationState::clip("run", 2));

        // Non-interruptible transitions
        sm.add_transition(
            Transition::direct(0, 1)
                .with_condition(TransitionCondition::always())
                .with_blend_time(0.5)
                .non_interruptible(),
        );
        sm.add_transition(
            Transition::direct(1, 2)
                .with_condition(TransitionCondition::always())
                .with_blend_time(0.3)
                .non_interruptible(),
        );

        let params = ParameterSet::new();

        // Start first transition
        sm.update(&params, 0.016);
        assert!(sm.is_transitioning());
        assert_eq!(sm.transition_target(), Some(1));

        // Mid-transition, the queue should have second transition
        sm.update(&params, 0.2);
        assert!(sm.is_transitioning());
        // Still going to state 1
        assert_eq!(sm.transition_target(), Some(1));
    }

    #[test]
    fn test_blend_time_and_curve() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("a", 0));
        sm.add_state(AnimationState::clip("b", 1));

        sm.add_transition(
            Transition::direct(0, 1)
                .with_condition(TransitionCondition::always())
                .with_blend_time(1.0)
                .with_blend_curve(BlendCurve::EaseInOut),
        );

        let params = ParameterSet::new();
        sm.update(&params, 0.0);

        // Start transition
        sm.update(&params, 0.001);
        assert!(sm.is_transitioning());

        let progress = sm.current_transition().unwrap();
        assert_eq!(progress.blend_curve, BlendCurve::EaseInOut);
        assert!((progress.duration - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_entry_exit_callbacks() {
        let mut sm = StateMachine::new();
        sm.add_state(
            AnimationState::clip("idle", 0)
                .with_on_enter("enter_idle")
                .with_on_exit("exit_idle"),
        );
        sm.add_state(
            AnimationState::clip("walk", 1)
                .with_on_enter("enter_walk")
                .with_on_exit("exit_walk"),
        );

        sm.add_transition(
            Transition::direct(0, 1)
                .with_condition(TransitionCondition::always())
                .with_blend_time(0.1),
        );

        let params = ParameterSet::new();

        // Trigger transition (should fire exit_idle)
        sm.update(&params, 0.016);
        let callbacks = sm.drain_callbacks();
        assert_eq!(callbacks.len(), 1);
        assert_eq!(callbacks[0].name, "exit_idle");
        assert_eq!(callbacks[0].callback_type, CallbackType::Exit);

        // Complete transition (should fire enter_walk)
        sm.update(&params, 1.0);
        let callbacks = sm.drain_callbacks();
        assert_eq!(callbacks.len(), 1);
        assert_eq!(callbacks[0].name, "enter_walk");
        assert_eq!(callbacks[0].callback_type, CallbackType::Enter);
    }

    #[test]
    fn test_state_machine_self_transition() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("attack", 0));

        // Self-transition for combo attacks
        sm.add_transition(
            Transition::direct(0, 0)
                .with_condition(TransitionCondition::trigger("attack"))
                .with_blend_time(0.1),
        );

        let mut params = ParameterSet::new();
        params.fire_trigger("attack");

        sm.update(&params, 0.016);
        assert!(sm.is_transitioning());
        assert_eq!(sm.transition_target(), Some(0));
    }

    #[test]
    fn test_no_valid_transition() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));
        sm.add_state(AnimationState::clip("walk", 1));

        sm.add_transition(
            Transition::direct(0, 1).with_condition(TransitionCondition::float_param(
                "speed",
                CompareOp::Greater,
                100.0,
            )),
        );

        let mut params = ParameterSet::new();
        params.set_float("speed", 1.0);

        sm.update(&params, 0.016);
        assert!(!sm.is_transitioning());
        assert_eq!(sm.current_state, 0);
    }

    #[test]
    fn test_force_state() {
        let mut sm = StateMachine::new();
        sm.add_state(
            AnimationState::clip("idle", 0)
                .with_on_exit("exit_idle"),
        );
        sm.add_state(
            AnimationState::clip("walk", 1)
                .with_on_enter("enter_walk"),
        );

        sm.force_state(1);
        assert_eq!(sm.current_state, 1);
        assert!(!sm.is_transitioning());

        let callbacks = sm.drain_callbacks();
        assert_eq!(callbacks.len(), 2);
        // Exit old state, enter new state
    }

    #[test]
    fn test_reset() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));
        sm.add_state(AnimationState::clip("walk", 1));

        sm.current_state = 1;
        sm.state_time = 5.0;

        sm.reset();
        assert_eq!(sm.current_state, 0);
        assert!((sm.state_time - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_disabled_transition() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));
        sm.add_state(AnimationState::clip("walk", 1));

        sm.add_transition(
            Transition::direct(0, 1)
                .with_condition(TransitionCondition::always())
                .disabled(),
        );

        let params = ParameterSet::new();
        sm.update(&params, 0.016);

        assert!(!sm.is_transitioning());
    }

    #[test]
    fn test_validation_no_states() {
        let sm = StateMachine::new();
        let issues = sm.validate();
        assert!(!issues.is_empty());
        assert!(issues[0].contains("no states"));
    }

    #[test]
    fn test_validation_invalid_entry() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));
        sm.entry_state = 5;

        let issues = sm.validate();
        assert!(issues.iter().any(|i| i.contains("Entry state")));
    }

    #[test]
    fn test_validation_invalid_transition_refs() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));
        sm.add_transition(Transition::direct(0, 5)); // Invalid to_state

        let issues = sm.validate();
        assert!(issues.iter().any(|i| i.contains("invalid to_state")));
    }

    #[test]
    fn test_validation_unreachable_state() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));
        sm.add_state(AnimationState::clip("unreachable", 1));
        // No transitions to state 1

        let issues = sm.validate();
        assert!(issues.iter().any(|i| i.contains("unreachable")));
    }

    // -------------------------------------------------------------------------
    // ParameterSet Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parameter_set() {
        let mut params = ParameterSet::new();

        params.set_float("speed", 5.0);
        params.set_int("combo", 3);
        params.set_bool("grounded", true);

        assert_eq!(params.get_float("speed"), Some(5.0));
        assert_eq!(params.get_int("combo"), Some(3));
        assert_eq!(params.get_bool("grounded"), Some(true));
        assert_eq!(params.get_float("nonexistent"), None);
    }

    #[test]
    fn test_parameter_set_triggers() {
        let mut params = ParameterSet::new();

        assert!(!params.is_trigger_fired("jump"));

        params.fire_trigger("jump");
        assert!(params.is_trigger_fired("jump"));

        params.consume_trigger("jump");
        assert!(!params.is_trigger_fired("jump"));
    }

    #[test]
    fn test_parameter_set_reset_triggers() {
        let mut params = ParameterSet::new();
        params.fire_trigger("a");
        params.fire_trigger("b");

        assert!(params.is_trigger_fired("a"));
        assert!(params.is_trigger_fired("b"));

        params.reset_triggers();
        assert!(!params.is_trigger_fired("a"));
        assert!(!params.is_trigger_fired("b"));
    }

    // -------------------------------------------------------------------------
    // Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder() {
        let sm = StateMachineBuilder::new()
            .name("locomotion")
            .clip_state("idle", 0)
            .clip_state("walk", 1)
            .clip_state("run", 2)
            .direct_transition(
                0,
                1,
                TransitionCondition::float_param("speed", CompareOp::Greater, 0.1),
            )
            .direct_transition(
                1,
                2,
                TransitionCondition::float_param("speed", CompareOp::Greater, 5.0),
            )
            .wildcard_transition(0, TransitionCondition::float_param("speed", CompareOp::Less, 0.1))
            .entry_state(0)
            .build();

        assert_eq!(sm.name, Some("locomotion".to_string()));
        assert_eq!(sm.state_count(), 3);
        assert_eq!(sm.transition_count(), 3);
        assert_eq!(sm.entry_state, 0);
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_zero_blend_time() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("a", 0));
        sm.add_state(AnimationState::clip("b", 1));

        sm.add_transition(
            Transition::direct(0, 1)
                .with_condition(TransitionCondition::always())
                .with_blend_time(0.0),
        );

        let params = ParameterSet::new();
        sm.update(&params, 0.001);

        // With zero blend time, should complete immediately
        assert!(!sm.is_transitioning());
        assert_eq!(sm.current_state, 1);
    }

    #[test]
    fn test_instant_blend_curve() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("a", 0));
        sm.add_state(AnimationState::clip("b", 1));

        sm.add_transition(
            Transition::direct(0, 1)
                .with_condition(TransitionCondition::always())
                .with_blend_time(1.0)
                .with_blend_curve(BlendCurve::Instant),
        );

        let params = ParameterSet::new();
        sm.update(&params, 0.016);

        let weight = sm.transition_blend_weight();
        assert!((weight - 1.0).abs() < 1e-6); // Instant snap
    }

    #[test]
    fn test_disabled_state_machine() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("a", 0));
        sm.add_state(AnimationState::clip("b", 1));
        sm.add_transition(Transition::direct(0, 1).with_condition(TransitionCondition::always()));

        sm.enabled = false;
        let params = ParameterSet::new();
        sm.update(&params, 0.016);

        assert!(!sm.is_transitioning());
        assert_eq!(sm.current_state, 0);
    }

    #[test]
    fn test_multiple_triggers_same_frame() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));
        sm.add_state(AnimationState::clip("attack", 1));
        sm.add_state(AnimationState::clip("jump", 2));

        sm.add_transition(
            Transition::direct(0, 1)
                .with_condition(TransitionCondition::trigger("attack"))
                .with_priority(10),
        );
        sm.add_transition(
            Transition::direct(0, 2)
                .with_condition(TransitionCondition::trigger("jump"))
                .with_priority(20),
        );

        let mut params = ParameterSet::new();
        params.fire_trigger("attack");
        params.fire_trigger("jump");

        sm.update(&params, 0.016);

        // Should pick jump (higher priority)
        assert!(sm.is_transitioning());
        assert_eq!(sm.transition_target(), Some(2));
    }

    #[test]
    fn test_transitions_from_state() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("a", 0));
        sm.add_state(AnimationState::clip("b", 1));
        sm.add_state(AnimationState::clip("c", 2));

        sm.add_transition(Transition::direct(0, 1));
        sm.add_transition(Transition::direct(0, 2));
        sm.add_transition(Transition::direct(1, 2));
        sm.add_transition(Transition::wildcard(0));

        let from_0: Vec<_> = sm.transitions_from(0).collect();
        assert_eq!(from_0.len(), 3); // 2 direct + 1 wildcard

        let wildcards: Vec<_> = sm.wildcard_transitions().collect();
        assert_eq!(wildcards.len(), 1);
    }

    #[test]
    fn test_has_path() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("a", 0));
        sm.add_state(AnimationState::clip("b", 1));
        sm.add_state(AnimationState::clip("c", 2));

        sm.add_transition(Transition::direct(0, 1));
        sm.add_transition(Transition::direct(1, 2));

        assert!(sm.has_path(0, 1));
        assert!(sm.has_path(1, 2));
        assert!(!sm.has_path(0, 2)); // No direct path
        assert!(!sm.has_path(2, 0)); // No path back
    }

    #[test]
    fn test_state_time_tracking() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));

        let params = ParameterSet::new();

        sm.update(&params, 0.5);
        assert!((sm.state_time() - 0.5).abs() < 1e-6);

        sm.update(&params, 0.3);
        assert!((sm.state_time() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_interruptible_transition() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));
        sm.add_state(AnimationState::clip("walk", 1));
        sm.add_state(AnimationState::clip("hurt", 2));

        // Non-interruptible walk transition
        sm.add_transition(
            Transition::direct(0, 1)
                .with_condition(TransitionCondition::always())
                .with_blend_time(1.0)
                .non_interruptible()
                .with_priority(10),
        );

        // Interruptible hurt transition (wildcard)
        sm.add_transition(
            Transition::wildcard(2)
                .with_condition(TransitionCondition::trigger("hurt"))
                .with_priority(1000)
                .interruptible(),
        );

        let mut params = ParameterSet::new();

        // Start walking
        sm.update(&params, 0.016);
        assert!(sm.is_transitioning());
        assert_eq!(sm.transition_target(), Some(1));

        // Get hurt mid-transition
        params.fire_trigger("hurt");
        sm.update(&params, 0.1);

        // Should have interrupted and started hurt transition
        assert!(sm.is_transitioning());
        assert_eq!(sm.transition_target(), Some(2));
    }
}
