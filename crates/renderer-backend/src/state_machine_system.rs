//! Animation state machine system integration for TRINITY Engine (T-AN-5.7).
//!
//! This module provides the complete animation controller system that ties together:
//!
//! - **StateMachineInstance**: Runtime instance of a state machine definition
//! - **AnimationController**: Owns multiple state machines, layers, and global parameters
//! - **ParameterBinding**: Bind external values to animation parameters
//! - **TransitionEvaluator**: Priority-based transition selection with triggers
//! - **BlendTreeEvaluator**: Recursive blend tree evaluation with pose caching
//! - **OutputPoseComputation**: Final pose computation with layer blending
//!
//! # Architecture
//!
//! ```text
//! AnimationController
//! +-- state_machines: Vec<StateMachineInstance>
//! |   +-- definition: StateMachine
//! |   +-- current_state: usize
//! |   +-- active_transition: Option<ActiveTransition>
//! |   +-- parameters: ParameterSet (local)
//! |   +-- state_time: f32
//! +-- layers: LayerStack
//! +-- global_params: ParameterSet
//! +-- parameter_bindings: Vec<ParameterBinding>
//! +-- blend_tree_cache: HashMap<usize, CachedPose>
//! +-- output_pose: Pose
//!
//! ParameterBinding
//! +-- source: BindingSource (External | Expression | Constant)
//! +-- target: String (parameter name)
//! +-- modifier: Option<ValueModifier>
//!
//! TransitionEvaluator
//! +-- evaluate_all(state_machine, params) -> Option<TransitionIndex>
//! +-- check_wildcards(params) -> Option<TransitionIndex>
//! +-- consume_triggers(params, transition)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::state_machine_system::{
//!     AnimationController, StateMachineInstance, ParameterBinding,
//! };
//! use renderer_backend::state_machine::{StateMachine, AnimationState, Transition};
//!
//! // Create a state machine definition
//! let mut sm = StateMachine::new();
//! sm.add_state(AnimationState::clip("idle", 0));
//! sm.add_state(AnimationState::clip("walk", 1));
//! sm.add_transition(Transition::direct(0, 1)
//!     .with_condition(TransitionCondition::float_param("speed", CompareOp::Greater, 0.1)));
//!
//! // Create controller with the state machine
//! let mut controller = AnimationController::new(64);
//! controller.add_state_machine(sm);
//!
//! // Bind external speed value
//! controller.bind_parameter("speed", BindingSource::External);
//!
//! // Update each frame
//! controller.set_float("speed", player.velocity.length());
//! let pose = controller.update(delta_time);
//! ```

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::animation_layers::{AnimationLayer, LayerBlendMode, LayerStack};
use crate::blend_tree::{BlendTree1D, BlendTree2D};
use crate::pose::{lerp_vec3, nlerp_quat, Pose, PoseType};
use crate::state_machine::{
    AnimationState, BlendCurve, CallbackType, EvaluationContext, ParameterSet, StateCallback,
    StateContent, StateMachine, SyncMode, Transition, TransitionCondition,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of state machines per controller.
pub const MAX_STATE_MACHINES: usize = 16;

/// Maximum number of parameter bindings.
pub const MAX_BINDINGS: usize = 128;

/// Maximum expression depth for parameter evaluation.
pub const MAX_EXPRESSION_DEPTH: usize = 8;

/// Epsilon for weight comparisons.
pub const WEIGHT_EPSILON: f32 = 1e-6;

/// Default update rate for controller.
pub const DEFAULT_UPDATE_RATE: f32 = 60.0;

// ---------------------------------------------------------------------------
// StateMachineSystemError
// ---------------------------------------------------------------------------

/// Errors that can occur in the state machine system.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StateMachineSystemError {
    /// State machine not found.
    StateMachineNotFound { index: usize },

    /// State not found.
    StateNotFound { name: String },

    /// Parameter not found.
    ParameterNotFound { name: String },

    /// Invalid state index.
    InvalidStateIndex { index: usize, max: usize },

    /// Circular binding dependency.
    CircularBinding { parameter: String },

    /// Expression evaluation failed.
    ExpressionError { expression: String, reason: String },

    /// Blend tree not found.
    BlendTreeNotFound { index: usize },

    /// Layer not found.
    LayerNotFound { index: usize },

    /// Maximum state machines exceeded.
    MaxStateMachinesExceeded,

    /// Invalid transition.
    InvalidTransition { from: usize, to: usize },
}

impl std::fmt::Display for StateMachineSystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StateMachineNotFound { index } => {
                write!(f, "state machine not found: index {}", index)
            }
            Self::StateNotFound { name } => write!(f, "state not found: {}", name),
            Self::ParameterNotFound { name } => write!(f, "parameter not found: {}", name),
            Self::InvalidStateIndex { index, max } => {
                write!(f, "invalid state index {}, max is {}", index, max)
            }
            Self::CircularBinding { parameter } => {
                write!(f, "circular binding dependency for parameter '{}'", parameter)
            }
            Self::ExpressionError { expression, reason } => {
                write!(f, "expression error in '{}': {}", expression, reason)
            }
            Self::BlendTreeNotFound { index } => {
                write!(f, "blend tree not found: index {}", index)
            }
            Self::LayerNotFound { index } => write!(f, "layer not found: index {}", index),
            Self::MaxStateMachinesExceeded => write!(f, "maximum state machines exceeded"),
            Self::InvalidTransition { from, to } => {
                write!(f, "invalid transition from {} to {}", from, to)
            }
        }
    }
}

impl std::error::Error for StateMachineSystemError {}

// ---------------------------------------------------------------------------
// ActiveTransition
// ---------------------------------------------------------------------------

/// An active transition in progress.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActiveTransition {
    /// Index of the transition being executed.
    pub transition_index: usize,

    /// Source state index.
    pub from_state: usize,

    /// Destination state index.
    pub to_state: usize,

    /// Elapsed time since transition started.
    pub elapsed: f32,

    /// Total transition duration.
    pub duration: f32,

    /// Blend curve for this transition.
    pub blend_curve: BlendCurve,

    /// Sync mode for this transition.
    pub sync_mode: SyncMode,

    /// Source state's time when transition started.
    pub source_time: f32,

    /// Target state's time when transition started.
    pub target_time: f32,

    /// Cached pose from source state.
    pub source_pose: Option<Pose>,

    /// Cached pose from target state.
    pub target_pose: Option<Pose>,
}

impl ActiveTransition {
    /// Create a new active transition.
    pub fn new(
        transition_index: usize,
        from_state: usize,
        to_state: usize,
        duration: f32,
        blend_curve: BlendCurve,
        sync_mode: SyncMode,
        source_time: f32,
    ) -> Self {
        Self {
            transition_index,
            from_state,
            to_state,
            elapsed: 0.0,
            duration,
            blend_curve,
            sync_mode,
            source_time,
            target_time: 0.0,
            source_pose: None,
            target_pose: None,
        }
    }

    /// Get the normalized progress (0.0 to 1.0).
    #[inline]
    pub fn progress(&self) -> f32 {
        if self.duration <= 0.0 {
            1.0
        } else {
            (self.elapsed / self.duration).clamp(0.0, 1.0)
        }
    }

    /// Get the blend weight after applying the curve.
    #[inline]
    pub fn blend_weight(&self) -> f32 {
        self.blend_curve.apply(self.progress())
    }

    /// Check if the transition is complete.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.elapsed >= self.duration
    }

    /// Advance the transition by delta time.
    pub fn advance(&mut self, dt: f32) -> bool {
        let was_complete = self.is_complete();
        self.elapsed += dt;
        !was_complete && self.is_complete()
    }
}

// ---------------------------------------------------------------------------
// StateMachineInstance
// ---------------------------------------------------------------------------

/// Runtime instance of a state machine definition.
///
/// Maintains the current state, active transitions, and local parameters
/// for a single state machine instance.
#[derive(Clone, Debug)]
pub struct StateMachineInstance {
    /// The state machine definition.
    pub definition: StateMachine,

    /// Current state index.
    pub current_state: usize,

    /// Active transition if any.
    pub active_transition: Option<ActiveTransition>,

    /// Local parameter values (can override global).
    pub parameters: ParameterSet,

    /// Time accumulated in current state.
    pub state_time: f32,

    /// Whether this instance is enabled.
    pub enabled: bool,

    /// Weight of this state machine in blending.
    pub weight: f32,

    /// Callbacks triggered during last update.
    pending_callbacks: Vec<StateCallback>,

    /// Cached output pose from last evaluation.
    cached_pose: Option<Pose>,

    /// Frame number when cache was last updated.
    /// Note: Reserved for future cache invalidation optimization.
    #[allow(dead_code)]
    cache_frame: u64,

    /// Optional name for debugging.
    pub name: Option<String>,
}

impl StateMachineInstance {
    /// Create a new state machine instance.
    pub fn new(definition: StateMachine) -> Self {
        let entry_state = definition.entry_state;
        Self {
            definition,
            current_state: entry_state,
            active_transition: None,
            parameters: ParameterSet::new(),
            state_time: 0.0,
            enabled: true,
            weight: 1.0,
            pending_callbacks: Vec::new(),
            cached_pose: None,
            cache_frame: 0,
            name: None,
        }
    }

    /// Set the instance name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the instance weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Get the current state.
    pub fn get_current_state(&self) -> Option<&AnimationState> {
        self.definition.states.get(self.current_state)
    }

    /// Get the current state name.
    pub fn current_state_name(&self) -> &str {
        self.definition
            .states
            .get(self.current_state)
            .map(|s| s.name.as_str())
            .unwrap_or("unknown")
    }

    /// Check if currently transitioning.
    #[inline]
    pub fn is_transitioning(&self) -> bool {
        self.active_transition.is_some()
    }

    /// Get the transition target state index.
    pub fn transition_target(&self) -> Option<usize> {
        self.active_transition.as_ref().map(|t| t.to_state)
    }

    /// Get the current blend weight if transitioning.
    pub fn transition_weight(&self) -> f32 {
        self.active_transition
            .as_ref()
            .map(|t| t.blend_weight())
            .unwrap_or(0.0)
    }

    /// Reset to entry state.
    pub fn reset(&mut self) {
        // Exit current state
        if let Some(on_exit) = self
            .definition
            .states
            .get(self.current_state)
            .and_then(|s| s.on_exit.clone())
        {
            self.pending_callbacks.push(StateCallback {
                name: on_exit,
                callback_type: CallbackType::Exit,
                state_index: self.current_state,
            });
        }

        self.current_state = self.definition.entry_state;
        self.active_transition = None;
        self.state_time = 0.0;
        self.cached_pose = None;

        // Enter entry state
        if let Some(on_enter) = self
            .definition
            .states
            .get(self.definition.entry_state)
            .and_then(|s| s.on_enter.clone())
        {
            self.pending_callbacks.push(StateCallback {
                name: on_enter,
                callback_type: CallbackType::Enter,
                state_index: self.definition.entry_state,
            });
        }
    }

    /// Force state change without transition.
    pub fn force_state(&mut self, state_index: usize) -> bool {
        if state_index >= self.definition.states.len() {
            return false;
        }

        // Exit current state
        if let Some(on_exit) = self
            .definition
            .states
            .get(self.current_state)
            .and_then(|s| s.on_exit.clone())
        {
            self.pending_callbacks.push(StateCallback {
                name: on_exit,
                callback_type: CallbackType::Exit,
                state_index: self.current_state,
            });
        }

        self.current_state = state_index;
        self.active_transition = None;
        self.state_time = 0.0;
        self.cached_pose = None;

        // Enter new state
        if let Some(on_enter) = self
            .definition
            .states
            .get(state_index)
            .and_then(|s| s.on_enter.clone())
        {
            self.pending_callbacks.push(StateCallback {
                name: on_enter,
                callback_type: CallbackType::Enter,
                state_index,
            });
        }

        true
    }

    /// Force state change by name.
    pub fn force_state_by_name(&mut self, name: &str) -> bool {
        if let Some(idx) = self.definition.find_state(name) {
            self.force_state(idx)
        } else {
            false
        }
    }

    /// Drain pending callbacks.
    pub fn drain_callbacks(&mut self) -> Vec<StateCallback> {
        std::mem::take(&mut self.pending_callbacks)
    }

    /// Set a local parameter.
    pub fn set_float(&mut self, name: &str, value: f32) {
        self.parameters.set_float(name, value);
    }

    /// Set a local bool parameter.
    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.parameters.set_bool(name, value);
    }

    /// Set a local int parameter.
    pub fn set_int(&mut self, name: &str, value: i32) {
        self.parameters.set_int(name, value);
    }

    /// Fire a trigger.
    pub fn fire_trigger(&mut self, name: &str) {
        self.parameters.fire_trigger(name);
    }
}

// ---------------------------------------------------------------------------
// BindingSource
// ---------------------------------------------------------------------------

/// Source of a parameter binding value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BindingSource {
    /// External value set manually.
    External,

    /// Constant value.
    Constant(f32),

    /// Boolean constant.
    ConstantBool(bool),

    /// Integer constant.
    ConstantInt(i32),

    /// Expression combining other parameters.
    Expression(String),

    /// Another parameter (copy).
    Parameter(String),

    /// Time-based oscillation.
    Oscillator {
        frequency: f32,
        amplitude: f32,
        offset: f32,
    },

    /// Random value in range.
    Random { min: f32, max: f32 },

    /// Smoothed external value.
    Smoothed { source: String, smoothing: f32 },
}

// ---------------------------------------------------------------------------
// ValueModifier
// ---------------------------------------------------------------------------

/// Modifier applied to parameter values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ValueModifier {
    /// Multiply by factor.
    Scale(f32),

    /// Add offset.
    Offset(f32),

    /// Clamp to range.
    Clamp { min: f32, max: f32 },

    /// Remap from one range to another.
    Remap {
        in_min: f32,
        in_max: f32,
        out_min: f32,
        out_max: f32,
    },

    /// Absolute value.
    Abs,

    /// Smooth over time.
    Smooth(f32),

    /// Threshold (0 or 1).
    Threshold(f32),
}

impl ValueModifier {
    /// Apply this modifier to a value.
    pub fn apply(&self, value: f32) -> f32 {
        match self {
            Self::Scale(factor) => value * factor,
            Self::Offset(offset) => value + offset,
            Self::Clamp { min, max } => value.clamp(*min, *max),
            Self::Remap {
                in_min,
                in_max,
                out_min,
                out_max,
            } => {
                let t = (value - in_min) / (in_max - in_min);
                out_min + t * (out_max - out_min)
            }
            Self::Abs => value.abs(),
            Self::Smooth(_) => value, // Smoothing requires state
            Self::Threshold(threshold) => {
                if value >= *threshold {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ParameterBinding
// ---------------------------------------------------------------------------

/// Binding from an external source to an animation parameter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterBinding {
    /// Target parameter name.
    pub target: String,

    /// Source of the value.
    pub source: BindingSource,

    /// Optional modifier to apply.
    pub modifier: Option<ValueModifier>,

    /// Whether this binding is active.
    pub active: bool,

    /// Current smoothed value (for Smooth modifier).
    #[serde(skip)]
    smoothed_value: f32,
}

impl ParameterBinding {
    /// Create a new binding.
    pub fn new(target: impl Into<String>, source: BindingSource) -> Self {
        Self {
            target: target.into(),
            source,
            modifier: None,
            active: true,
            smoothed_value: 0.0,
        }
    }

    /// Create an external binding.
    pub fn external(target: impl Into<String>) -> Self {
        Self::new(target, BindingSource::External)
    }

    /// Create a constant binding.
    pub fn constant(target: impl Into<String>, value: f32) -> Self {
        Self::new(target, BindingSource::Constant(value))
    }

    /// Create a parameter copy binding.
    pub fn copy(target: impl Into<String>, source: impl Into<String>) -> Self {
        Self::new(target, BindingSource::Parameter(source.into()))
    }

    /// Add a modifier.
    pub fn with_modifier(mut self, modifier: ValueModifier) -> Self {
        self.modifier = Some(modifier);
        self
    }

    /// Disable the binding.
    pub fn inactive(mut self) -> Self {
        self.active = false;
        self
    }

    /// Evaluate this binding and return the value.
    pub fn evaluate(&mut self, params: &ParameterSet, dt: f32, time: f32) -> Option<f32> {
        if !self.active {
            return None;
        }

        let raw_value = match &self.source {
            BindingSource::External => return None, // External values set manually
            BindingSource::Constant(v) => *v,
            BindingSource::ConstantBool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            BindingSource::ConstantInt(i) => *i as f32,
            BindingSource::Expression(_) => return None, // Expressions need separate handling
            BindingSource::Parameter(name) => params.get_float(name).unwrap_or(0.0),
            BindingSource::Oscillator {
                frequency,
                amplitude,
                offset,
            } => (time * frequency * std::f32::consts::TAU).sin() * amplitude + offset,
            BindingSource::Random { min, max } => {
                // Simple LCG for deterministic "randomness"
                let seed = (time * 1000.0) as u32;
                let rand = ((seed.wrapping_mul(1103515245).wrapping_add(12345)) % 0x7FFFFFFF) as f32
                    / 0x7FFFFFFF as f32;
                min + rand * (max - min)
            }
            BindingSource::Smoothed { source, smoothing } => {
                let target = params.get_float(source).unwrap_or(0.0);
                self.smoothed_value += (target - self.smoothed_value) * (1.0 - smoothing) * dt * 60.0;
                self.smoothed_value
            }
        };

        let value = match &self.modifier {
            Some(modifier) => modifier.apply(raw_value),
            None => raw_value,
        };

        Some(value)
    }
}

// ---------------------------------------------------------------------------
// TransitionEvaluator
// ---------------------------------------------------------------------------

/// Evaluates transitions for state machines.
///
/// Handles priority-based selection, wildcard transitions, and trigger consumption.
pub struct TransitionEvaluator;

impl TransitionEvaluator {
    /// Evaluate all outgoing transitions from the current state.
    ///
    /// Returns the index of the highest-priority valid transition, if any.
    pub fn evaluate(
        machine: &StateMachine,
        current_state: usize,
        params: &ParameterSet,
        state_time: f32,
        state_complete: bool,
        in_transition: bool,
    ) -> Option<usize> {
        let context = EvaluationContext {
            state_complete,
            current_time: state_time,
            normalized_time: 0.0, // Would need clip duration for this
            current_state,
        };

        let mut best_transition: Option<(usize, i32)> = None;

        for (i, transition) in machine.transitions.iter().enumerate() {
            if !transition.enabled {
                continue;
            }

            // Check if we can interrupt current transition
            if in_transition && !transition.can_interrupt {
                continue;
            }

            // Check if this transition applies from current state
            if !transition.applies_from(current_state) {
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

    /// Get all wildcard transitions that are currently valid.
    pub fn get_valid_wildcards(
        machine: &StateMachine,
        params: &ParameterSet,
        context: &EvaluationContext,
    ) -> Vec<usize> {
        machine
            .transitions
            .iter()
            .enumerate()
            .filter(|(_, t)| t.is_wildcard() && t.enabled && t.condition.evaluate(params, context))
            .map(|(i, _)| i)
            .collect()
    }

    /// Consume triggers used by a transition.
    pub fn consume_triggers(transition: &Transition, params: &mut ParameterSet) {
        if transition.consume_trigger {
            Self::consume_condition_triggers(&transition.condition, params);
        }
    }

    /// Recursively consume triggers from a condition.
    fn consume_condition_triggers(condition: &TransitionCondition, params: &mut ParameterSet) {
        match condition {
            TransitionCondition::Trigger { name } => {
                params.consume_trigger(name);
            }
            TransitionCondition::And(a, b) | TransitionCondition::Or(a, b) => {
                Self::consume_condition_triggers(a, params);
                Self::consume_condition_triggers(b, params);
            }
            TransitionCondition::Not(c) => {
                Self::consume_condition_triggers(c, params);
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// BlendTreeEvaluator
// ---------------------------------------------------------------------------

/// Evaluates blend trees recursively with pose caching.
#[derive(Clone, Debug)]
pub struct BlendTreeEvaluator {
    /// Cached poses by blend tree index.
    cache: HashMap<usize, CachedPose>,

    /// Current frame number.
    frame: u64,
}

/// A cached pose from blend tree evaluation.
/// Note: Currently unused but prepared for future caching optimization.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct CachedPose {
    /// The computed pose.
    pose: Pose,

    /// Frame when this was computed.
    frame: u64,

    /// Parameter hash for cache invalidation.
    param_hash: u64,
}

impl BlendTreeEvaluator {
    /// Create a new blend tree evaluator.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            frame: 0,
        }
    }

    /// Advance to the next frame.
    pub fn advance_frame(&mut self) {
        self.frame += 1;
    }

    /// Clear the cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Evaluate a 1D blend tree.
    pub fn evaluate_1d(
        &mut self,
        tree: &mut BlendTree1D,
        param_value: f32,
        bone_count: usize,
        clip_sampler: &dyn Fn(usize, f32, usize) -> Pose,
    ) -> Pose {
        let weights = tree.evaluate(param_value);

        if weights.is_empty() {
            return Pose::new(bone_count, PoseType::Current);
        }

        if weights.len() == 1 {
            let (clip_idx, _) = weights[0];
            return clip_sampler(clip_idx, 0.0, bone_count);
        }

        // Blend the poses
        self.blend_weighted_poses(&weights, bone_count, clip_sampler)
    }

    /// Evaluate a 2D blend tree.
    pub fn evaluate_2d(
        &mut self,
        tree: &mut BlendTree2D,
        param_x: f32,
        param_y: f32,
        bone_count: usize,
        clip_sampler: &dyn Fn(usize, f32, usize) -> Pose,
    ) -> Pose {
        let weights = tree.evaluate(param_x, param_y);

        if weights.is_empty() {
            return Pose::new(bone_count, PoseType::Current);
        }

        if weights.len() == 1 {
            let (clip_idx, _) = weights[0];
            return clip_sampler(clip_idx, 0.0, bone_count);
        }

        self.blend_weighted_poses(&weights, bone_count, clip_sampler)
    }

    /// Blend poses with the given weights.
    fn blend_weighted_poses(
        &self,
        weights: &[(usize, f32)],
        bone_count: usize,
        clip_sampler: &dyn Fn(usize, f32, usize) -> Pose,
    ) -> Pose {
        let mut result = Pose::new(bone_count, PoseType::Current);

        // Collect weighted poses
        let poses: Vec<(Pose, f32)> = weights
            .iter()
            .filter(|(_, w)| *w > WEIGHT_EPSILON)
            .map(|(idx, w)| (clip_sampler(*idx, 0.0, bone_count), *w))
            .collect();

        if poses.is_empty() {
            return result;
        }

        if poses.len() == 1 {
            return poses[0].0.clone();
        }

        // Blend positions and scales linearly
        for i in 0..bone_count {
            let mut pos = Vec3::ZERO;
            let mut scale = Vec3::ZERO;

            for (pose, weight) in &poses {
                if i < pose.bone_count() {
                    pos += pose.positions[i] * *weight;
                    scale += pose.scales[i] * *weight;
                }
            }

            result.positions[i] = pos;
            result.scales[i] = scale;

            // Blend rotations
            if i < poses[0].0.bone_count() {
                let mut rot = poses[0].0.rotations[i];
                let mut accumulated_weight = poses[0].1;

                for (pose, weight) in poses.iter().skip(1) {
                    if i < pose.bone_count() && *weight > WEIGHT_EPSILON {
                        let blend_factor = *weight / (accumulated_weight + *weight);
                        rot = nlerp_quat(rot, pose.rotations[i], blend_factor);
                        accumulated_weight += *weight;
                    }
                }

                result.rotations[i] = rot.normalize();
            }
        }

        result
    }
}

impl Default for BlendTreeEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// OutputPoseComputation
// ---------------------------------------------------------------------------

/// Computes the final output pose from state machine evaluation.
pub struct OutputPoseComputation;

impl OutputPoseComputation {
    /// Evaluate the current state's pose.
    pub fn evaluate_state_pose(
        state: &AnimationState,
        state_time: f32,
        bone_count: usize,
        clip_sampler: &dyn Fn(usize, f32, usize) -> Pose,
        blend_tree_sampler: &dyn Fn(usize, &ParameterSet, usize) -> Pose,
        params: &ParameterSet,
    ) -> Pose {
        match &state.clip_or_tree {
            StateContent::Clip(idx) => clip_sampler(*idx, state_time, bone_count),
            StateContent::BlendTree(idx) => blend_tree_sampler(*idx, params, bone_count),
            StateContent::SubGraph(_) => Pose::new(bone_count, PoseType::Current), // Sub-graphs evaluated separately
            StateContent::Empty => Pose::new(bone_count, PoseType::Current),
        }
    }

    /// Blend poses during a transition.
    pub fn blend_transition(
        source_pose: &Pose,
        target_pose: &Pose,
        weight: f32,
        _sync_mode: SyncMode,
    ) -> Pose {
        let bone_count = source_pose.bone_count().min(target_pose.bone_count());
        let mut result = Pose::new(bone_count, PoseType::Current);

        for i in 0..bone_count {
            result.positions[i] = lerp_vec3(source_pose.positions[i], target_pose.positions[i], weight);
            result.rotations[i] = nlerp_quat(source_pose.rotations[i], target_pose.rotations[i], weight);
            result.scales[i] = lerp_vec3(source_pose.scales[i], target_pose.scales[i], weight);
        }

        result
    }

    /// Combine poses across layers.
    pub fn combine_layers(layer_poses: &[(Pose, &AnimationLayer)], bone_count: usize) -> Pose {
        if layer_poses.is_empty() {
            return Pose::new(bone_count, PoseType::Current);
        }

        let mut result = Pose::new(bone_count, PoseType::Current);
        let mut has_base = false;

        for (pose, layer) in layer_poses {
            if !layer.active || layer.weight <= 0.0 {
                continue;
            }

            match layer.blend_mode {
                LayerBlendMode::Override => {
                    if !has_base {
                        // First override layer becomes base
                        result = pose.clone();
                        has_base = true;
                    } else {
                        // Blend with existing
                        for i in 0..bone_count.min(pose.bone_count()) {
                            let w = layer.effective_weight(i);
                            if w > 0.0 {
                                result.positions[i] =
                                    lerp_vec3(result.positions[i], pose.positions[i], w);
                                result.rotations[i] =
                                    nlerp_quat(result.rotations[i], pose.rotations[i], w);
                                result.scales[i] =
                                    lerp_vec3(result.scales[i], pose.scales[i], w);
                            }
                        }
                    }
                }
                LayerBlendMode::Additive => {
                    for i in 0..bone_count.min(pose.bone_count()) {
                        let w = layer.effective_weight(i);
                        if w > 0.0 {
                            result.positions[i] += pose.positions[i] * w;
                            let additive_rot = nlerp_quat(Quat::IDENTITY, pose.rotations[i], w);
                            result.rotations[i] =
                                (result.rotations[i] * additive_rot).normalize();
                            result.scales[i] += pose.scales[i] * w;
                        }
                    }
                }
                LayerBlendMode::Multiply => {
                    for i in 0..bone_count.min(pose.bone_count()) {
                        let w = layer.effective_weight(i);
                        if w >= 1.0 {
                            result.positions[i] = Vec3::new(
                                result.positions[i].x * pose.positions[i].x,
                                result.positions[i].y * pose.positions[i].y,
                                result.positions[i].z * pose.positions[i].z,
                            );
                            result.rotations[i] =
                                (result.rotations[i] * pose.rotations[i]).normalize();
                            result.scales[i] = Vec3::new(
                                result.scales[i].x * pose.scales[i].x,
                                result.scales[i].y * pose.scales[i].y,
                                result.scales[i].z * pose.scales[i].z,
                            );
                        } else if w > 0.0 {
                            let mult_pos = Vec3::new(
                                result.positions[i].x * pose.positions[i].x,
                                result.positions[i].y * pose.positions[i].y,
                                result.positions[i].z * pose.positions[i].z,
                            );
                            result.positions[i] = lerp_vec3(result.positions[i], mult_pos, w);

                            let mult_rot =
                                (result.rotations[i] * pose.rotations[i]).normalize();
                            result.rotations[i] = nlerp_quat(result.rotations[i], mult_rot, w);

                            let mult_scale = Vec3::new(
                                result.scales[i].x * pose.scales[i].x,
                                result.scales[i].y * pose.scales[i].y,
                                result.scales[i].z * pose.scales[i].z,
                            );
                            result.scales[i] = lerp_vec3(result.scales[i], mult_scale, w);
                        }
                    }
                }
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// AnimationController
// ---------------------------------------------------------------------------

/// Main animation controller that orchestrates state machines, layers, and blending.
///
/// The controller is the central hub for animation evaluation. It:
/// - Manages multiple state machine instances
/// - Maintains a layer stack for pose blending
/// - Handles global parameter storage and bindings
/// - Produces the final output pose each frame
#[derive(Clone, Debug)]
pub struct AnimationController {
    /// State machine instances.
    pub state_machines: Vec<StateMachineInstance>,

    /// Layer stack for final blending.
    pub layers: LayerStack,

    /// Global animation parameters.
    pub global_params: ParameterSet,

    /// Parameter bindings.
    pub bindings: Vec<ParameterBinding>,

    /// Blend tree evaluator.
    blend_tree_evaluator: BlendTreeEvaluator,

    /// Number of bones in the skeleton.
    pub bone_count: usize,

    /// Current frame number.
    frame: u64,

    /// Total elapsed time.
    total_time: f32,

    /// Whether the controller is enabled.
    pub enabled: bool,

    /// Cached output pose.
    output_pose: Option<Pose>,

    /// Pending callbacks from all state machines.
    pending_callbacks: Vec<(usize, StateCallback)>,

    /// Optional name for debugging.
    pub name: Option<String>,
}

impl AnimationController {
    /// Create a new animation controller.
    pub fn new(bone_count: usize) -> Self {
        Self {
            state_machines: Vec::new(),
            layers: LayerStack::new(bone_count),
            global_params: ParameterSet::new(),
            bindings: Vec::new(),
            blend_tree_evaluator: BlendTreeEvaluator::new(),
            bone_count,
            frame: 0,
            total_time: 0.0,
            enabled: true,
            output_pose: None,
            pending_callbacks: Vec::new(),
            name: None,
        }
    }

    /// Set the controller name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Get the frame number.
    #[inline]
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Get total elapsed time.
    #[inline]
    pub fn total_time(&self) -> f32 {
        self.total_time
    }

    // -------------------------------------------------------------------------
    // State Machine Management
    // -------------------------------------------------------------------------

    /// Add a state machine.
    pub fn add_state_machine(&mut self, definition: StateMachine) -> usize {
        let idx = self.state_machines.len();
        self.state_machines.push(StateMachineInstance::new(definition));
        idx
    }

    /// Add a state machine instance.
    pub fn add_state_machine_instance(&mut self, instance: StateMachineInstance) -> usize {
        let idx = self.state_machines.len();
        self.state_machines.push(instance);
        idx
    }

    /// Get a state machine by index.
    pub fn get_state_machine(&self, index: usize) -> Option<&StateMachineInstance> {
        self.state_machines.get(index)
    }

    /// Get a mutable state machine by index.
    pub fn get_state_machine_mut(&mut self, index: usize) -> Option<&mut StateMachineInstance> {
        self.state_machines.get_mut(index)
    }

    /// Get the number of state machines.
    pub fn state_machine_count(&self) -> usize {
        self.state_machines.len()
    }

    // -------------------------------------------------------------------------
    // Layer Management
    // -------------------------------------------------------------------------

    /// Add a layer.
    pub fn add_layer(&mut self, layer: AnimationLayer) -> usize {
        self.layers.add_layer(layer)
    }

    /// Get a layer by index.
    pub fn get_layer(&self, index: usize) -> Option<&AnimationLayer> {
        self.layers.get_layer(index)
    }

    /// Get a mutable layer by index.
    pub fn get_layer_mut(&mut self, index: usize) -> Option<&mut AnimationLayer> {
        self.layers.get_layer_mut(index)
    }

    /// Set layer weight.
    pub fn set_layer_weight(&mut self, index: usize, weight: f32) -> bool {
        self.layers.set_layer_weight(index, weight)
    }

    /// Set layer active state.
    pub fn set_layer_active(&mut self, index: usize, active: bool) -> bool {
        self.layers.set_layer_active(index, active)
    }

    // -------------------------------------------------------------------------
    // Parameter Management
    // -------------------------------------------------------------------------

    /// Set a float parameter.
    pub fn set_float(&mut self, name: &str, value: f32) {
        self.global_params.set_float(name, value);
    }

    /// Set a bool parameter.
    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.global_params.set_bool(name, value);
    }

    /// Set an int parameter.
    pub fn set_int(&mut self, name: &str, value: i32) {
        self.global_params.set_int(name, value);
    }

    /// Fire a trigger.
    pub fn fire_trigger(&mut self, name: &str) {
        self.global_params.fire_trigger(name);
    }

    /// Get a float parameter.
    pub fn get_float(&self, name: &str) -> Option<f32> {
        self.global_params.get_float(name)
    }

    /// Get a bool parameter.
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.global_params.get_bool(name)
    }

    /// Get an int parameter.
    pub fn get_int(&self, name: &str) -> Option<i32> {
        self.global_params.get_int(name)
    }

    /// Check if a trigger is fired.
    pub fn is_trigger_fired(&self, name: &str) -> bool {
        self.global_params.is_trigger_fired(name)
    }

    /// Reset all triggers.
    pub fn reset_triggers(&mut self) {
        self.global_params.reset_triggers();
    }

    // -------------------------------------------------------------------------
    // Bindings
    // -------------------------------------------------------------------------

    /// Add a parameter binding.
    pub fn add_binding(&mut self, binding: ParameterBinding) {
        self.bindings.push(binding);
    }

    /// Bind an external parameter.
    pub fn bind_parameter(&mut self, name: &str, source: BindingSource) {
        self.bindings.push(ParameterBinding::new(name, source));
    }

    /// Remove a binding by target name.
    pub fn remove_binding(&mut self, target: &str) {
        self.bindings.retain(|b| b.target != target);
    }

    /// Update all bindings.
    fn update_bindings(&mut self, dt: f32) {
        for binding in &mut self.bindings {
            if let Some(value) = binding.evaluate(&self.global_params, dt, self.total_time) {
                self.global_params.set_float(&binding.target, value);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Update
    // -------------------------------------------------------------------------

    /// Update the controller and return the output pose.
    ///
    /// This is the main update function that:
    /// 1. Updates bindings
    /// 2. Updates all state machines
    /// 3. Evaluates transitions
    /// 4. Computes blended poses
    /// 5. Combines layers
    /// 6. Returns the final pose
    pub fn update(&mut self, dt: f32) -> Pose {
        if !self.enabled {
            return self
                .output_pose
                .clone()
                .unwrap_or_else(|| Pose::new(self.bone_count, PoseType::Current));
        }

        self.frame += 1;
        self.total_time += dt;
        self.blend_tree_evaluator.advance_frame();

        // Update bindings
        self.update_bindings(dt);

        // Update each state machine
        for (sm_idx, sm) in self.state_machines.iter_mut().enumerate() {
            if !sm.enabled {
                continue;
            }

            // Merge global params with local params
            let mut merged_params = self.global_params.clone();
            merged_params.merge_from(&sm.parameters);

            // Advance state time
            sm.state_time += dt;

            // Handle active transition
            if let Some(ref mut transition) = sm.active_transition {
                let completed = transition.advance(dt);

                if completed {
                    // Transition complete - enter new state
                    let to_state = transition.to_state;
                    sm.current_state = to_state;
                    sm.state_time = 0.0;
                    sm.active_transition = None;
                    sm.cached_pose = None;

                    // Enter callback
                    if let Some(on_enter) = sm
                        .definition
                        .states
                        .get(to_state)
                        .and_then(|s| s.on_enter.clone())
                    {
                        sm.pending_callbacks.push(StateCallback {
                            name: on_enter,
                            callback_type: CallbackType::Enter,
                            state_index: to_state,
                        });
                    }
                }
            } else {
                // Evaluate transitions
                if let Some(transition_idx) =
                    TransitionEvaluator::evaluate(
                        &sm.definition,
                        sm.current_state,
                        &merged_params,
                        sm.state_time,
                        false,
                        false,
                    )
                {
                    // Start the transition
                    let transition = &sm.definition.transitions[transition_idx];

                    // Consume triggers
                    TransitionEvaluator::consume_triggers(transition, &mut merged_params);

                    // Exit callback
                    if let Some(on_exit) = sm
                        .definition
                        .states
                        .get(sm.current_state)
                        .and_then(|s| s.on_exit.clone())
                    {
                        sm.pending_callbacks.push(StateCallback {
                            name: on_exit,
                            callback_type: CallbackType::Exit,
                            state_index: sm.current_state,
                        });
                    }

                    let mut active = ActiveTransition::new(
                        transition_idx,
                        sm.current_state,
                        transition.to_state,
                        transition.blend_time,
                        transition.blend_curve,
                        transition.sync_mode,
                        sm.state_time,
                    );

                    // Advance the transition immediately to handle large dt
                    let completed = active.advance(dt);
                    if completed {
                        // Transition completes in the same frame
                        sm.current_state = active.to_state;
                        sm.state_time = 0.0;
                        sm.cached_pose = None;

                        // Enter callback for target state
                        if let Some(on_enter) = sm
                            .definition
                            .states
                            .get(active.to_state)
                            .and_then(|s| s.on_enter.clone())
                        {
                            sm.pending_callbacks.push(StateCallback {
                                name: on_enter,
                                callback_type: CallbackType::Enter,
                                state_index: active.to_state,
                            });
                        }
                    } else {
                        sm.active_transition = Some(active);
                    }
                }
            }

            // Collect callbacks
            let callbacks = sm.drain_callbacks();
            for callback in callbacks {
                self.pending_callbacks.push((sm_idx, callback));
            }
        }

        // Reset triggers at end of frame
        self.global_params.reset_triggers();
        for sm in &mut self.state_machines {
            sm.parameters.reset_triggers();
        }

        // Compute output pose
        let output = self.compute_output_pose();
        self.output_pose = Some(output.clone());
        output
    }

    /// Compute the final output pose from all state machines and layers.
    fn compute_output_pose(&self) -> Pose {
        if self.state_machines.is_empty() {
            return Pose::new(self.bone_count, PoseType::Current);
        }

        // Simple case: single state machine
        if self.state_machines.len() == 1 {
            return self.evaluate_state_machine_pose(0);
        }

        // Multiple state machines - blend by weight
        let mut poses: Vec<(Pose, f32)> = Vec::new();
        let mut total_weight = 0.0;

        for (i, sm) in self.state_machines.iter().enumerate() {
            if !sm.enabled || sm.weight <= 0.0 {
                continue;
            }

            let pose = self.evaluate_state_machine_pose(i);
            poses.push((pose, sm.weight));
            total_weight += sm.weight;
        }

        if poses.is_empty() {
            return Pose::new(self.bone_count, PoseType::Current);
        }

        if poses.len() == 1 {
            return poses[0].0.clone();
        }

        // Normalize weights and blend
        let mut result = Pose::new(self.bone_count, PoseType::Current);

        for i in 0..self.bone_count {
            let mut pos = Vec3::ZERO;
            let mut scale = Vec3::ZERO;

            for (pose, weight) in &poses {
                let normalized_weight = *weight / total_weight;
                if i < pose.bone_count() {
                    pos += pose.positions[i] * normalized_weight;
                    scale += pose.scales[i] * normalized_weight;
                }
            }

            result.positions[i] = pos;
            result.scales[i] = scale;

            // Blend rotations
            if i < poses[0].0.bone_count() {
                let mut rot = poses[0].0.rotations[i];
                let mut accumulated_weight = poses[0].1 / total_weight;

                for (pose, weight) in poses.iter().skip(1) {
                    let normalized_weight = *weight / total_weight;
                    if i < pose.bone_count() && normalized_weight > WEIGHT_EPSILON {
                        let blend_factor =
                            normalized_weight / (accumulated_weight + normalized_weight);
                        rot = nlerp_quat(rot, pose.rotations[i], blend_factor);
                        accumulated_weight += normalized_weight;
                    }
                }

                result.rotations[i] = rot.normalize();
            }
        }

        result
    }

    /// Evaluate a single state machine's pose.
    fn evaluate_state_machine_pose(&self, sm_index: usize) -> Pose {
        let sm = match self.state_machines.get(sm_index) {
            Some(sm) => sm,
            None => return Pose::new(self.bone_count, PoseType::Current),
        };

        let state = match sm.definition.states.get(sm.current_state) {
            Some(s) => s,
            None => return Pose::new(self.bone_count, PoseType::Current),
        };

        // Sample current state pose
        let current_pose = self.sample_state_pose(state, sm.state_time);

        // If transitioning, blend with target state
        if let Some(ref transition) = sm.active_transition {
            let target_state = sm.definition.states.get(transition.to_state);

            if let Some(target) = target_state {
                let target_pose = self.sample_state_pose(target, transition.elapsed);
                let weight = transition.blend_weight();

                return OutputPoseComputation::blend_transition(
                    &current_pose,
                    &target_pose,
                    weight,
                    transition.sync_mode,
                );
            }
        }

        current_pose
    }

    /// Sample a pose from a state.
    fn sample_state_pose(&self, state: &AnimationState, time: f32) -> Pose {
        // Create a basic pose based on the state content
        // In a real implementation, this would sample from actual clip data
        let mut pose = Pose::new(self.bone_count, PoseType::Current);

        match &state.clip_or_tree {
            StateContent::Clip(_clip_idx) => {
                // Simple oscillation for testing
                let t = time * state.speed_multiplier;
                let offset = (t * std::f32::consts::PI * 2.0).sin() * 0.1;

                if self.bone_count > 0 {
                    pose.positions[0] = Vec3::new(0.0, offset, 0.0);
                }
            }
            StateContent::BlendTree(_tree_idx) => {
                // Would evaluate blend tree here
            }
            StateContent::SubGraph(_graph_idx) => {
                // Would evaluate sub-graph here
            }
            StateContent::Empty => {}
        }

        pose
    }

    // -------------------------------------------------------------------------
    // Callbacks
    // -------------------------------------------------------------------------

    /// Drain pending callbacks from all state machines.
    pub fn drain_callbacks(&mut self) -> Vec<(usize, StateCallback)> {
        std::mem::take(&mut self.pending_callbacks)
    }

    /// Check if there are pending callbacks.
    pub fn has_pending_callbacks(&self) -> bool {
        !self.pending_callbacks.is_empty()
    }

    // -------------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------------

    /// Reset all state machines to their entry states.
    pub fn reset(&mut self) {
        for sm in &mut self.state_machines {
            sm.reset();
        }
        self.output_pose = None;
        self.total_time = 0.0;
        self.frame = 0;
    }

    /// Reset triggers and prepare for next frame.
    pub fn end_frame(&mut self) {
        self.global_params.reset_triggers();
        for sm in &mut self.state_machines {
            sm.parameters.reset_triggers();
        }
    }

    // -------------------------------------------------------------------------
    // Query
    // -------------------------------------------------------------------------

    /// Get the current state name for a state machine.
    pub fn get_current_state_name(&self, sm_index: usize) -> Option<&str> {
        self.state_machines
            .get(sm_index)
            .map(|sm| sm.current_state_name())
    }

    /// Check if a state machine is transitioning.
    pub fn is_transitioning(&self, sm_index: usize) -> bool {
        self.state_machines
            .get(sm_index)
            .map(|sm| sm.is_transitioning())
            .unwrap_or(false)
    }

    /// Get the last computed output pose.
    pub fn get_output_pose(&self) -> Option<&Pose> {
        self.output_pose.as_ref()
    }

    /// Validate all state machines.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        for (i, sm) in self.state_machines.iter().enumerate() {
            let sm_issues = sm.definition.validate();
            for issue in sm_issues {
                issues.push(format!("State machine {}: {}", i, issue));
            }
        }

        issues
    }
}

impl Default for AnimationController {
    fn default() -> Self {
        Self::new(64)
    }
}

// ---------------------------------------------------------------------------
// AnimationControllerBuilder
// ---------------------------------------------------------------------------

/// Builder for creating animation controllers.
pub struct AnimationControllerBuilder {
    controller: AnimationController,
}

impl AnimationControllerBuilder {
    /// Create a new builder.
    pub fn new(bone_count: usize) -> Self {
        Self {
            controller: AnimationController::new(bone_count),
        }
    }

    /// Set the controller name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.controller.name = Some(name.into());
        self
    }

    /// Add a state machine.
    pub fn state_machine(mut self, definition: StateMachine) -> Self {
        self.controller.add_state_machine(definition);
        self
    }

    /// Add a state machine instance.
    pub fn state_machine_instance(mut self, instance: StateMachineInstance) -> Self {
        self.controller.add_state_machine_instance(instance);
        self
    }

    /// Add a layer.
    pub fn layer(mut self, layer: AnimationLayer) -> Self {
        self.controller.add_layer(layer);
        self
    }

    /// Add a binding.
    pub fn binding(mut self, binding: ParameterBinding) -> Self {
        self.controller.add_binding(binding);
        self
    }

    /// Set a float parameter.
    pub fn float_param(mut self, name: impl Into<String>, value: f32) -> Self {
        self.controller.global_params.set_float(name, value);
        self
    }

    /// Set a bool parameter.
    pub fn bool_param(mut self, name: impl Into<String>, value: bool) -> Self {
        self.controller.global_params.set_bool(name, value);
        self
    }

    /// Build the controller.
    pub fn build(self) -> AnimationController {
        self.controller
    }
}

impl Default for AnimationControllerBuilder {
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
    use crate::state_machine::CompareOp;

    // =========================================================================
    // Helper Functions
    // =========================================================================

    fn create_simple_state_machine() -> StateMachine {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));
        sm.add_state(AnimationState::clip("walk", 1));
        sm.add_state(AnimationState::clip("run", 2));

        sm.add_transition(
            Transition::direct(0, 1)
                .with_condition(TransitionCondition::float_param("speed", CompareOp::Greater, 0.1))
                .with_blend_time(0.2),
        );

        sm.add_transition(
            Transition::direct(1, 2)
                .with_condition(TransitionCondition::float_param("speed", CompareOp::Greater, 5.0))
                .with_blend_time(0.3),
        );

        sm.add_transition(
            Transition::direct(1, 0)
                .with_condition(TransitionCondition::float_param("speed", CompareOp::Less, 0.1))
                .with_blend_time(0.2),
        );

        sm
    }

    fn create_trigger_state_machine() -> StateMachine {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("idle", 0));
        sm.add_state(AnimationState::clip("attack", 1).with_looping(false));
        sm.add_state(AnimationState::clip("hurt", 2).with_looping(false));

        sm.add_transition(
            Transition::direct(0, 1)
                .with_condition(TransitionCondition::trigger("attack"))
                .with_blend_time(0.1),
        );

        sm.add_transition(
            Transition::wildcard(2)
                .with_condition(TransitionCondition::trigger("hurt"))
                .with_priority(100)
                .with_blend_time(0.05),
        );

        sm
    }

    // =========================================================================
    // ActiveTransition Tests
    // =========================================================================

    #[test]
    fn test_active_transition_new() {
        let trans = ActiveTransition::new(0, 0, 1, 0.5, BlendCurve::Linear, SyncMode::FreezeSource, 0.0);

        assert_eq!(trans.transition_index, 0);
        assert_eq!(trans.from_state, 0);
        assert_eq!(trans.to_state, 1);
        assert_eq!(trans.duration, 0.5);
        assert!(!trans.is_complete());
    }

    #[test]
    fn test_active_transition_progress() {
        let mut trans = ActiveTransition::new(0, 0, 1, 1.0, BlendCurve::Linear, SyncMode::FreezeSource, 0.0);

        assert!((trans.progress() - 0.0).abs() < WEIGHT_EPSILON);
        assert!((trans.blend_weight() - 0.0).abs() < WEIGHT_EPSILON);

        trans.elapsed = 0.5;
        assert!((trans.progress() - 0.5).abs() < WEIGHT_EPSILON);
        assert!((trans.blend_weight() - 0.5).abs() < WEIGHT_EPSILON);

        trans.elapsed = 1.0;
        assert!((trans.progress() - 1.0).abs() < WEIGHT_EPSILON);
        assert!((trans.blend_weight() - 1.0).abs() < WEIGHT_EPSILON);
    }

    #[test]
    fn test_active_transition_advance() {
        let mut trans = ActiveTransition::new(0, 0, 1, 0.5, BlendCurve::Linear, SyncMode::FreezeSource, 0.0);

        let completed = trans.advance(0.3);
        assert!(!completed);
        assert!(!trans.is_complete());

        let completed = trans.advance(0.3);
        assert!(completed);
        assert!(trans.is_complete());
    }

    #[test]
    fn test_active_transition_blend_curves() {
        let linear = ActiveTransition::new(0, 0, 1, 1.0, BlendCurve::Linear, SyncMode::FreezeSource, 0.0);
        let ease_in = ActiveTransition::new(0, 0, 1, 1.0, BlendCurve::EaseIn, SyncMode::FreezeSource, 0.0);
        let ease_out = ActiveTransition::new(0, 0, 1, 1.0, BlendCurve::EaseOut, SyncMode::FreezeSource, 0.0);

        // At midpoint
        let mut t_linear = linear.clone();
        t_linear.elapsed = 0.5;

        let mut t_ease_in = ease_in.clone();
        t_ease_in.elapsed = 0.5;

        let mut t_ease_out = ease_out.clone();
        t_ease_out.elapsed = 0.5;

        assert!((t_linear.blend_weight() - 0.5).abs() < WEIGHT_EPSILON);
        assert!(t_ease_in.blend_weight() < 0.5); // Ease in is slower at start
        assert!(t_ease_out.blend_weight() > 0.5); // Ease out is faster at start
    }

    // =========================================================================
    // StateMachineInstance Tests
    // =========================================================================

    #[test]
    fn test_state_machine_instance_new() {
        let sm = create_simple_state_machine();
        let instance = StateMachineInstance::new(sm);

        assert_eq!(instance.current_state, 0);
        assert!(!instance.is_transitioning());
        assert!(instance.enabled);
        assert_eq!(instance.weight, 1.0);
    }

    #[test]
    fn test_state_machine_instance_with_name() {
        let sm = create_simple_state_machine();
        let instance = StateMachineInstance::new(sm).with_name("locomotion");

        assert_eq!(instance.name, Some("locomotion".to_string()));
    }

    #[test]
    fn test_state_machine_instance_current_state_name() {
        let sm = create_simple_state_machine();
        let instance = StateMachineInstance::new(sm);

        assert_eq!(instance.current_state_name(), "idle");
    }

    #[test]
    fn test_state_machine_instance_force_state() {
        let sm = create_simple_state_machine();
        let mut instance = StateMachineInstance::new(sm);

        assert!(instance.force_state(1));
        assert_eq!(instance.current_state, 1);
        assert_eq!(instance.current_state_name(), "walk");

        assert!(!instance.force_state(100)); // Invalid index
    }

    #[test]
    fn test_state_machine_instance_force_state_by_name() {
        let sm = create_simple_state_machine();
        let mut instance = StateMachineInstance::new(sm);

        assert!(instance.force_state_by_name("run"));
        assert_eq!(instance.current_state, 2);

        assert!(!instance.force_state_by_name("nonexistent"));
    }

    #[test]
    fn test_state_machine_instance_reset() {
        let sm = create_simple_state_machine();
        let mut instance = StateMachineInstance::new(sm);

        instance.force_state(2);
        instance.state_time = 5.0;

        instance.reset();

        assert_eq!(instance.current_state, 0);
        assert_eq!(instance.state_time, 0.0);
    }

    #[test]
    fn test_state_machine_instance_local_params() {
        let sm = create_simple_state_machine();
        let mut instance = StateMachineInstance::new(sm);

        instance.set_float("speed", 3.0);
        instance.set_bool("grounded", true);
        instance.set_int("combo", 2);

        assert_eq!(instance.parameters.get_float("speed"), Some(3.0));
        assert_eq!(instance.parameters.get_bool("grounded"), Some(true));
        assert_eq!(instance.parameters.get_int("combo"), Some(2));
    }

    #[test]
    fn test_state_machine_instance_triggers() {
        let sm = create_trigger_state_machine();
        let mut instance = StateMachineInstance::new(sm);

        assert!(!instance.parameters.is_trigger_fired("attack"));

        instance.fire_trigger("attack");
        assert!(instance.parameters.is_trigger_fired("attack"));
    }

    // =========================================================================
    // BindingSource Tests
    // =========================================================================

    #[test]
    fn test_binding_source_constant() {
        let mut binding = ParameterBinding::constant("test", 5.0);
        let params = ParameterSet::new();

        let value = binding.evaluate(&params, 0.016, 0.0);
        assert_eq!(value, Some(5.0));
    }

    #[test]
    fn test_binding_source_parameter_copy() {
        let mut binding = ParameterBinding::copy("target", "source");
        let mut params = ParameterSet::new();
        params.set_float("source", 3.5);

        let value = binding.evaluate(&params, 0.016, 0.0);
        assert_eq!(value, Some(3.5));
    }

    #[test]
    fn test_binding_source_oscillator() {
        let mut binding = ParameterBinding::new(
            "test",
            BindingSource::Oscillator {
                frequency: 1.0,
                amplitude: 1.0,
                offset: 0.0,
            },
        );
        let params = ParameterSet::new();

        // At time 0, sin(0) = 0
        let value = binding.evaluate(&params, 0.016, 0.0);
        assert!(value.unwrap().abs() < WEIGHT_EPSILON);

        // At time 0.25 (quarter period), sin(PI/2) = 1
        let value = binding.evaluate(&params, 0.016, 0.25);
        assert!((value.unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_binding_external() {
        let mut binding = ParameterBinding::external("test");
        let params = ParameterSet::new();

        // External bindings return None (value set manually)
        let value = binding.evaluate(&params, 0.016, 0.0);
        assert_eq!(value, None);
    }

    // =========================================================================
    // ValueModifier Tests
    // =========================================================================

    #[test]
    fn test_value_modifier_scale() {
        let modifier = ValueModifier::Scale(2.0);
        assert_eq!(modifier.apply(5.0), 10.0);
    }

    #[test]
    fn test_value_modifier_offset() {
        let modifier = ValueModifier::Offset(3.0);
        assert_eq!(modifier.apply(5.0), 8.0);
    }

    #[test]
    fn test_value_modifier_clamp() {
        let modifier = ValueModifier::Clamp { min: 0.0, max: 10.0 };
        assert_eq!(modifier.apply(-5.0), 0.0);
        assert_eq!(modifier.apply(5.0), 5.0);
        assert_eq!(modifier.apply(15.0), 10.0);
    }

    #[test]
    fn test_value_modifier_remap() {
        let modifier = ValueModifier::Remap {
            in_min: 0.0,
            in_max: 10.0,
            out_min: 0.0,
            out_max: 100.0,
        };
        assert_eq!(modifier.apply(0.0), 0.0);
        assert_eq!(modifier.apply(5.0), 50.0);
        assert_eq!(modifier.apply(10.0), 100.0);
    }

    #[test]
    fn test_value_modifier_abs() {
        let modifier = ValueModifier::Abs;
        assert_eq!(modifier.apply(-5.0), 5.0);
        assert_eq!(modifier.apply(5.0), 5.0);
    }

    #[test]
    fn test_value_modifier_threshold() {
        let modifier = ValueModifier::Threshold(5.0);
        assert_eq!(modifier.apply(3.0), 0.0);
        assert_eq!(modifier.apply(5.0), 1.0);
        assert_eq!(modifier.apply(7.0), 1.0);
    }

    // =========================================================================
    // ParameterBinding Tests
    // =========================================================================

    #[test]
    fn test_parameter_binding_with_modifier() {
        let mut binding = ParameterBinding::constant("test", 5.0)
            .with_modifier(ValueModifier::Scale(2.0));
        let params = ParameterSet::new();

        let value = binding.evaluate(&params, 0.016, 0.0);
        assert_eq!(value, Some(10.0));
    }

    #[test]
    fn test_parameter_binding_inactive() {
        let mut binding = ParameterBinding::constant("test", 5.0).inactive();
        let params = ParameterSet::new();

        let value = binding.evaluate(&params, 0.016, 0.0);
        assert_eq!(value, None);
    }

    // =========================================================================
    // TransitionEvaluator Tests
    // =========================================================================

    #[test]
    fn test_transition_evaluator_basic() {
        let sm = create_simple_state_machine();
        let mut params = ParameterSet::new();

        // No transition when speed is 0
        let result = TransitionEvaluator::evaluate(&sm, 0, &params, 0.0, false, false);
        assert!(result.is_none());

        // Should transition when speed > 0.1
        params.set_float("speed", 1.0);
        let result = TransitionEvaluator::evaluate(&sm, 0, &params, 0.0, false, false);
        assert_eq!(result, Some(0)); // First transition (idle -> walk)
    }

    #[test]
    fn test_transition_evaluator_priority() {
        let mut sm = StateMachine::new();
        sm.add_state(AnimationState::clip("a", 0));
        sm.add_state(AnimationState::clip("b", 1));
        sm.add_state(AnimationState::clip("c", 2));

        sm.add_transition(
            Transition::direct(0, 1)
                .with_condition(TransitionCondition::always())
                .with_priority(10),
        );
        sm.add_transition(
            Transition::direct(0, 2)
                .with_condition(TransitionCondition::always())
                .with_priority(100),
        );

        let params = ParameterSet::new();
        let result = TransitionEvaluator::evaluate(&sm, 0, &params, 0.0, false, false);

        // Should pick higher priority (index 1, to state c)
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_transition_evaluator_wildcards() {
        let sm = create_trigger_state_machine();
        let mut params = ParameterSet::new();
        params.fire_trigger("hurt");

        let context = EvaluationContext::new(0);
        let wildcards = TransitionEvaluator::get_valid_wildcards(&sm, &params, &context);

        assert_eq!(wildcards.len(), 1);
    }

    #[test]
    fn test_transition_evaluator_consume_triggers() {
        let sm = create_trigger_state_machine();
        let mut params = ParameterSet::new();
        params.fire_trigger("attack");

        assert!(params.is_trigger_fired("attack"));

        let transition = &sm.transitions[0];
        TransitionEvaluator::consume_triggers(transition, &mut params);

        assert!(!params.is_trigger_fired("attack"));
    }

    // =========================================================================
    // BlendTreeEvaluator Tests
    // =========================================================================

    #[test]
    fn test_blend_tree_evaluator_new() {
        let evaluator = BlendTreeEvaluator::new();
        assert_eq!(evaluator.frame, 0);
        assert!(evaluator.cache.is_empty());
    }

    #[test]
    fn test_blend_tree_evaluator_advance_frame() {
        let mut evaluator = BlendTreeEvaluator::new();
        evaluator.advance_frame();
        assert_eq!(evaluator.frame, 1);
    }

    #[test]
    fn test_blend_tree_evaluator_1d() {
        let mut evaluator = BlendTreeEvaluator::new();
        let mut tree = BlendTree1D::new("speed");
        tree.add_clip(0.0, 0);
        tree.add_clip(10.0, 1);

        let clip_sampler = |clip_idx: usize, _time: f32, bone_count: usize| -> Pose {
            let mut pose = Pose::new(bone_count, PoseType::Current);
            if bone_count > 0 {
                pose.positions[0] = Vec3::new(clip_idx as f32, 0.0, 0.0);
            }
            pose
        };

        // At midpoint, should blend 50/50
        let pose = evaluator.evaluate_1d(&mut tree, 5.0, 2, &clip_sampler);
        assert!((pose.positions[0].x - 0.5).abs() < 0.01);
    }

    // =========================================================================
    // OutputPoseComputation Tests
    // =========================================================================

    #[test]
    fn test_output_pose_blend_transition() {
        let mut source = Pose::new(2, PoseType::Current);
        source.positions[0] = Vec3::new(0.0, 0.0, 0.0);

        let mut target = Pose::new(2, PoseType::Current);
        target.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let blended = OutputPoseComputation::blend_transition(&source, &target, 0.5, SyncMode::Crossfade);

        assert!((blended.positions[0].x - 5.0).abs() < WEIGHT_EPSILON);
    }

    #[test]
    fn test_output_pose_combine_layers_single() {
        let pose = Pose::new(2, PoseType::Current);
        let layer = AnimationLayer::base("test");

        let layer_poses = vec![(pose.clone(), &layer)];
        let result = OutputPoseComputation::combine_layers(&layer_poses, 2);

        assert_eq!(result.bone_count(), 2);
    }

    #[test]
    fn test_output_pose_combine_layers_override() {
        let mut base = Pose::new(2, PoseType::Current);
        base.positions[0] = Vec3::new(0.0, 0.0, 0.0);

        let mut overlay = Pose::new(2, PoseType::Current);
        overlay.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let base_layer = AnimationLayer::base("base");
        let overlay_layer = AnimationLayer::base("overlay").with_weight(0.5);

        let layer_poses = vec![(base, &base_layer), (overlay, &overlay_layer)];
        let result = OutputPoseComputation::combine_layers(&layer_poses, 2);

        assert!((result.positions[0].x - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_output_pose_combine_layers_additive() {
        let mut base = Pose::new(2, PoseType::Current);
        base.positions[0] = Vec3::new(5.0, 0.0, 0.0);

        let mut additive = Pose::new(2, PoseType::Current);
        additive.positions[0] = Vec3::new(3.0, 0.0, 0.0);

        let base_layer = AnimationLayer::base("base");
        let additive_layer = AnimationLayer::additive("additive");

        let layer_poses = vec![(base, &base_layer), (additive, &additive_layer)];
        let result = OutputPoseComputation::combine_layers(&layer_poses, 2);

        // Should add: 5 + 3 = 8
        assert!((result.positions[0].x - 8.0).abs() < WEIGHT_EPSILON);
    }

    // =========================================================================
    // AnimationController Tests
    // =========================================================================

    #[test]
    fn test_animation_controller_new() {
        let controller = AnimationController::new(64);
        assert_eq!(controller.bone_count, 64);
        assert!(controller.state_machines.is_empty());
        assert!(controller.enabled);
    }

    #[test]
    fn test_animation_controller_add_state_machine() {
        let mut controller = AnimationController::new(64);
        let sm = create_simple_state_machine();

        let idx = controller.add_state_machine(sm);
        assert_eq!(idx, 0);
        assert_eq!(controller.state_machine_count(), 1);
    }

    #[test]
    fn test_animation_controller_parameters() {
        let mut controller = AnimationController::new(64);

        controller.set_float("speed", 5.0);
        controller.set_bool("grounded", true);
        controller.set_int("combo", 2);

        assert_eq!(controller.get_float("speed"), Some(5.0));
        assert_eq!(controller.get_bool("grounded"), Some(true));
        assert_eq!(controller.get_int("combo"), Some(2));
    }

    #[test]
    fn test_animation_controller_triggers() {
        let mut controller = AnimationController::new(64);

        assert!(!controller.is_trigger_fired("attack"));

        controller.fire_trigger("attack");
        assert!(controller.is_trigger_fired("attack"));

        controller.reset_triggers();
        assert!(!controller.is_trigger_fired("attack"));
    }

    #[test]
    fn test_animation_controller_update_basic() {
        let mut controller = AnimationController::new(64);
        let sm = create_simple_state_machine();
        controller.add_state_machine(sm);

        let pose = controller.update(0.016);
        assert_eq!(pose.bone_count(), 64);
    }

    #[test]
    fn test_animation_controller_transition() {
        let mut controller = AnimationController::new(64);
        let sm = create_simple_state_machine();
        controller.add_state_machine(sm);

        // Start in idle
        assert_eq!(controller.get_current_state_name(0), Some("idle"));

        // Set speed to trigger transition
        controller.set_float("speed", 3.0);
        controller.update(0.016);

        // Should be transitioning to walk
        assert!(controller.is_transitioning(0));

        // Complete the transition
        for _ in 0..20 {
            controller.update(0.016);
        }

        assert!(!controller.is_transitioning(0));
        assert_eq!(controller.get_current_state_name(0), Some("walk"));
    }

    #[test]
    fn test_animation_controller_trigger_transition() {
        let mut controller = AnimationController::new(64);
        let sm = create_trigger_state_machine();
        controller.add_state_machine(sm);

        assert_eq!(controller.get_current_state_name(0), Some("idle"));

        controller.fire_trigger("attack");
        controller.update(0.016);

        assert!(controller.is_transitioning(0));

        // Trigger should be consumed
        assert!(!controller.is_trigger_fired("attack"));
    }

    #[test]
    fn test_animation_controller_wildcard_transition() {
        let mut controller = AnimationController::new(64);
        let sm = create_trigger_state_machine();
        controller.add_state_machine(sm);

        // Force to attack state
        controller.get_state_machine_mut(0).unwrap().force_state(1);
        assert_eq!(controller.get_current_state_name(0), Some("attack"));

        // Wildcard should trigger from any state
        controller.fire_trigger("hurt");
        controller.update(0.016);

        // Should be transitioning to hurt
        assert!(controller.is_transitioning(0));
    }

    #[test]
    fn test_animation_controller_bindings() {
        let mut controller = AnimationController::new(64);

        controller.add_binding(ParameterBinding::constant("test", 5.0));
        controller.update(0.016);

        // Binding should have set the parameter
        assert_eq!(controller.get_float("test"), Some(5.0));
    }

    #[test]
    fn test_animation_controller_reset() {
        let mut controller = AnimationController::new(64);
        let sm = create_simple_state_machine();
        controller.add_state_machine(sm);

        controller.set_float("speed", 3.0);
        controller.update(0.5);
        controller.update(0.5);

        assert_ne!(controller.frame(), 0);

        controller.reset();

        assert_eq!(controller.frame(), 0);
        assert_eq!(controller.get_current_state_name(0), Some("idle"));
    }

    #[test]
    fn test_animation_controller_layers() {
        let mut controller = AnimationController::new(64);

        let layer_idx = controller.add_layer(AnimationLayer::base("locomotion"));
        assert_eq!(layer_idx, 0);

        assert!(controller.set_layer_weight(0, 0.5));
        assert_eq!(controller.get_layer(0).unwrap().weight, 0.5);

        assert!(controller.set_layer_active(0, false));
        assert!(!controller.get_layer(0).unwrap().active);
    }

    #[test]
    fn test_animation_controller_validate() {
        let mut controller = AnimationController::new(64);

        // Empty state machine should produce issues
        let sm = StateMachine::new();
        controller.add_state_machine(sm);

        let issues = controller.validate();
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_animation_controller_callbacks() {
        let mut controller = AnimationController::new(64);

        let mut sm = StateMachine::new();
        sm.add_state(
            AnimationState::clip("idle", 0)
                .with_on_enter("enter_idle")
                .with_on_exit("exit_idle"),
        );
        sm.add_state(AnimationState::clip("walk", 1).with_on_enter("enter_walk"));

        sm.add_transition(
            Transition::direct(0, 1)
                .with_condition(TransitionCondition::float_param("speed", CompareOp::Greater, 0.1))
                .with_blend_time(0.1),
        );

        controller.add_state_machine(sm);

        // Trigger transition
        controller.set_float("speed", 3.0);
        controller.update(0.016);

        // Should have exit callback
        let callbacks = controller.drain_callbacks();
        assert!(!callbacks.is_empty());
        assert_eq!(callbacks[0].1.name, "exit_idle");
        assert_eq!(callbacks[0].1.callback_type, CallbackType::Exit);

        // Complete transition
        for _ in 0..10 {
            controller.update(0.016);
        }

        // Should have enter callback
        let callbacks = controller.drain_callbacks();
        assert!(callbacks.iter().any(|(_, c)| c.name == "enter_walk"));
    }

    #[test]
    fn test_animation_controller_multiple_state_machines() {
        let mut controller = AnimationController::new(64);

        let sm1 = create_simple_state_machine();
        let sm2 = create_trigger_state_machine();

        controller.add_state_machine(sm1);
        controller.add_state_machine(sm2);

        assert_eq!(controller.state_machine_count(), 2);

        // Both should be in their initial states
        assert_eq!(controller.get_current_state_name(0), Some("idle"));
        assert_eq!(controller.get_current_state_name(1), Some("idle"));

        let pose = controller.update(0.016);
        assert_eq!(pose.bone_count(), 64);
    }

    #[test]
    fn test_animation_controller_disabled() {
        let mut controller = AnimationController::new(64);
        let sm = create_simple_state_machine();
        controller.add_state_machine(sm);

        controller.update(0.016);
        let frame_before = controller.frame();

        controller.enabled = false;
        controller.update(0.016);

        // Frame shouldn't advance when disabled
        assert_eq!(controller.frame(), frame_before);
    }

    // =========================================================================
    // AnimationControllerBuilder Tests
    // =========================================================================

    #[test]
    fn test_controller_builder() {
        let controller = AnimationControllerBuilder::new(64)
            .name("character")
            .state_machine(create_simple_state_machine())
            .layer(AnimationLayer::base("locomotion"))
            .float_param("speed", 0.0)
            .bool_param("grounded", true)
            .build();

        assert_eq!(controller.name, Some("character".to_string()));
        assert_eq!(controller.state_machine_count(), 1);
        assert!(controller.layers.layer_count() >= 1);
        assert_eq!(controller.get_float("speed"), Some(0.0));
        assert_eq!(controller.get_bool("grounded"), Some(true));
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_locomotion_system() {
        let mut controller = AnimationController::new(64);
        let sm = create_simple_state_machine();
        controller.add_state_machine(sm);

        // Add oscillator binding for speed (simulating player input)
        controller.add_binding(ParameterBinding::new(
            "speed",
            BindingSource::Oscillator {
                frequency: 0.5,
                amplitude: 4.0,
                offset: 2.0,
            },
        ));

        // Simulate multiple frames
        for _ in 0..100 {
            let _pose = controller.update(0.016);
        }

        // Controller should have updated
        assert!(controller.frame() >= 100);
    }

    #[test]
    fn test_combat_system_triggers() {
        let mut controller = AnimationController::new(64);
        let sm = create_trigger_state_machine();
        controller.add_state_machine(sm);

        // Fire attack trigger
        controller.fire_trigger("attack");
        controller.update(0.016);

        assert!(controller.is_transitioning(0));

        // Complete the transition
        for _ in 0..20 {
            controller.update(0.016);
        }

        // Interrupt with hurt trigger (wildcard)
        controller.fire_trigger("hurt");
        controller.update(0.016);

        // Should immediately start transitioning to hurt
        assert!(controller.is_transitioning(0));
    }

    #[test]
    fn test_parameter_value_propagation() {
        let mut controller = AnimationController::new(64);
        let sm = create_simple_state_machine();
        controller.add_state_machine(sm);

        // Add binding that copies global param to local
        controller.add_binding(ParameterBinding::copy("local_speed", "speed"));
        controller.set_float("speed", 5.0);

        controller.update(0.016);

        // Local param should have copied value
        assert_eq!(controller.get_float("local_speed"), Some(5.0));
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_empty_controller() {
        let mut controller = AnimationController::new(64);
        let pose = controller.update(0.016);

        assert_eq!(pose.bone_count(), 64);
    }

    #[test]
    fn test_zero_bone_count() {
        let mut controller = AnimationController::new(0);
        let sm = create_simple_state_machine();
        controller.add_state_machine(sm);

        let pose = controller.update(0.016);
        assert_eq!(pose.bone_count(), 0);
    }

    #[test]
    fn test_very_small_dt() {
        let mut controller = AnimationController::new(64);
        let sm = create_simple_state_machine();
        controller.add_state_machine(sm);

        for _ in 0..1000 {
            controller.update(0.0001);
        }

        assert!(controller.total_time() > 0.0);
    }

    #[test]
    fn test_very_large_dt() {
        let mut controller = AnimationController::new(64);
        let sm = create_simple_state_machine();
        controller.add_state_machine(sm);

        controller.set_float("speed", 3.0);
        controller.update(10.0); // Large dt should complete transitions immediately

        // Transition should be complete
        assert!(!controller.is_transitioning(0));
    }

    #[test]
    fn test_state_machine_weight_zero() {
        let mut controller = AnimationController::new(64);
        let sm = create_simple_state_machine();
        let mut instance = StateMachineInstance::new(sm).with_weight(0.0);
        instance.enabled = true;

        controller.add_state_machine_instance(instance);

        let pose = controller.update(0.016);
        assert_eq!(pose.bone_count(), 64);
    }

    #[test]
    fn test_disabled_state_machine() {
        let mut controller = AnimationController::new(64);
        let sm = create_simple_state_machine();
        controller.add_state_machine(sm);

        controller.get_state_machine_mut(0).unwrap().enabled = false;
        controller.set_float("speed", 3.0);
        controller.update(0.016);

        // Should not transition when disabled
        assert!(!controller.is_transitioning(0));
    }
}
