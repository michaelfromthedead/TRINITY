//! Animation graph container for runtime animation system (T-AN-5.1).
//!
//! This module provides the runtime animation graph system with support for:
//!
//! - **Graph structure**: Layers, entry points, parameter mapping
//! - **Parameter system**: Float, int, bool, Vec3, and trigger parameters
//! - **Layer system**: Multiple layers with blend modes and bone masks
//! - **Dirty-flag evaluation**: Only re-evaluate when parameters change
//! - **Serialization**: Binary format for fast loading, debug format for inspection
//!
//! # Architecture
//!
//! ```text
//! AnimationGraph
//! ├── parameters: Vec<AnimationParameter>
//! │   ├── name: String
//! │   ├── value: ParameterValue (Float|Int|Bool|Vec3|Trigger)
//! │   └── dirty: bool
//! ├── layers: Vec<AnimationLayer>
//! │   ├── name: String
//! │   ├── weight: f32
//! │   ├── blend_mode: LayerBlendMode
//! │   ├── bone_mask: Option<Vec<bool>>
//! │   └── root_node: usize
//! ├── nodes: Vec<AnimationNode>
//! └── dirty: bool (global dirty flag)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::animation_graph::{
//!     AnimationGraph, AnimationParameter, ParameterValue,
//!     AnimationLayer, LayerBlendMode, AnimationNode,
//! };
//!
//! // Create a simple animation graph
//! let mut graph = AnimationGraph::new();
//!
//! // Add parameters
//! graph.add_parameter("speed", ParameterValue::Float(0.0));
//! graph.add_parameter("is_running", ParameterValue::Bool(false));
//! graph.add_parameter("jump", ParameterValue::Trigger);
//!
//! // Add a base layer
//! let base_layer = AnimationLayer::new("base", LayerBlendMode::Override);
//! graph.add_layer(base_layer);
//!
//! // Set parameter values (marks graph as dirty)
//! graph.set_parameter("speed", ParameterValue::Float(5.0));
//! graph.trigger("jump");
//!
//! // Evaluate if dirty
//! if graph.is_dirty() {
//!     let pose = graph.evaluate(0.016);
//!     graph.clear_dirty();
//! }
//! ```

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::pose::{lerp_vec3, nlerp_quat, Pose, PoseType};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of layers supported per graph.
pub const MAX_LAYERS: usize = 16;

/// Maximum number of parameters per graph.
pub const MAX_PARAMETERS: usize = 256;

/// Maximum number of nodes per graph.
pub const MAX_NODES: usize = 1024;

/// Binary format magic number for animation graphs.
pub const GRAPH_MAGIC: u32 = 0x54414E47; // "TANG" (Trinity ANimation Graph)

/// Binary format version.
pub const GRAPH_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// ParameterValue
// ---------------------------------------------------------------------------

/// A parameter value that can drive animation graph evaluation.
///
/// Parameters are the primary interface for gameplay code to control
/// animation behavior. They can be set from scripts, AI, or physics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    /// Floating-point parameter (e.g., speed, blend weight).
    Float(f32),

    /// Integer parameter (e.g., animation index, state ID).
    Int(i32),

    /// Boolean parameter (e.g., is_grounded, is_attacking).
    Bool(bool),

    /// 3D vector parameter (e.g., velocity, target direction).
    Vec3(Vec3),

    /// One-shot trigger that auto-resets after evaluation.
    /// Triggers are consumed during graph evaluation.
    Trigger,
}

impl ParameterValue {
    /// Get the type name for debugging.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Float(_) => "float",
            Self::Int(_) => "int",
            Self::Bool(_) => "bool",
            Self::Vec3(_) => "vec3",
            Self::Trigger => "trigger",
        }
    }

    /// Check if this is a trigger parameter.
    #[inline]
    pub fn is_trigger(&self) -> bool {
        matches!(self, Self::Trigger)
    }

    /// Try to get as float.
    #[inline]
    pub fn as_float(&self) -> Option<f32> {
        match self {
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as int.
    #[inline]
    pub fn as_int(&self) -> Option<i32> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as bool.
    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as Vec3.
    #[inline]
    pub fn as_vec3(&self) -> Option<Vec3> {
        match self {
            Self::Vec3(v) => Some(*v),
            _ => None,
        }
    }

    /// Get the default value for each type.
    pub fn default_for_type(type_name: &str) -> Option<Self> {
        match type_name {
            "float" => Some(Self::Float(0.0)),
            "int" => Some(Self::Int(0)),
            "bool" => Some(Self::Bool(false)),
            "vec3" => Some(Self::Vec3(Vec3::ZERO)),
            "trigger" => Some(Self::Trigger),
            _ => None,
        }
    }
}

impl Default for ParameterValue {
    fn default() -> Self {
        Self::Float(0.0)
    }
}

// ---------------------------------------------------------------------------
// AnimationParameter
// ---------------------------------------------------------------------------

/// A named parameter in the animation graph.
///
/// Parameters are identified by name for designer-friendly workflow,
/// but can be looked up by index for runtime performance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationParameter {
    /// Human-readable parameter name.
    pub name: String,

    /// Current parameter value.
    pub value: ParameterValue,

    /// Whether this parameter has been modified since last evaluation.
    /// Used for dirty-flag optimization.
    pub dirty: bool,
}

impl AnimationParameter {
    /// Create a new parameter with the given name and value.
    pub fn new(name: impl Into<String>, value: ParameterValue) -> Self {
        Self {
            name: name.into(),
            value,
            dirty: true, // New parameters start dirty
        }
    }

    /// Create a float parameter.
    #[inline]
    pub fn float(name: impl Into<String>, value: f32) -> Self {
        Self::new(name, ParameterValue::Float(value))
    }

    /// Create an int parameter.
    #[inline]
    pub fn int(name: impl Into<String>, value: i32) -> Self {
        Self::new(name, ParameterValue::Int(value))
    }

    /// Create a bool parameter.
    #[inline]
    pub fn bool(name: impl Into<String>, value: bool) -> Self {
        Self::new(name, ParameterValue::Bool(value))
    }

    /// Create a Vec3 parameter.
    #[inline]
    pub fn vec3(name: impl Into<String>, value: Vec3) -> Self {
        Self::new(name, ParameterValue::Vec3(value))
    }

    /// Create a trigger parameter.
    #[inline]
    pub fn trigger(name: impl Into<String>) -> Self {
        // Triggers start as "not fired" (dirty = false means not triggered)
        Self {
            name: name.into(),
            value: ParameterValue::Trigger,
            dirty: false,
        }
    }

    /// Set the parameter value and mark as dirty.
    pub fn set(&mut self, value: ParameterValue) {
        // Only mark dirty if value actually changed
        if self.value != value {
            self.value = value;
            self.dirty = true;
        }
    }

    /// Fire a trigger parameter.
    ///
    /// Returns true if this is a trigger and it was fired.
    pub fn fire(&mut self) -> bool {
        if self.value.is_trigger() {
            self.dirty = true;
            true
        } else {
            false
        }
    }

    /// Clear the dirty flag.
    #[inline]
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Reset a trigger after it has been consumed.
    pub fn reset_trigger(&mut self) {
        if self.value.is_trigger() {
            self.dirty = false;
        }
    }
}

// ---------------------------------------------------------------------------
// LayerBlendMode
// ---------------------------------------------------------------------------

/// How a layer blends with layers below it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerBlendMode {
    /// Completely replace the pose from lower layers.
    /// Used for base locomotion layers.
    #[default]
    Override,

    /// Add to the pose from lower layers.
    /// Used for hit reactions, breathing, procedural effects.
    Additive,

    /// Additive blend only on masked bones.
    /// Used for upper body overlays, facial animation.
    MaskedAdditive,
}

impl LayerBlendMode {
    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Override => "Override",
            Self::Additive => "Additive",
            Self::MaskedAdditive => "MaskedAdditive",
        }
    }

    /// Check if this mode uses additive blending.
    #[inline]
    pub fn is_additive(&self) -> bool {
        matches!(self, Self::Additive | Self::MaskedAdditive)
    }

    /// Check if this mode uses bone masking.
    #[inline]
    pub fn uses_mask(&self) -> bool {
        matches!(self, Self::MaskedAdditive)
    }
}

// ---------------------------------------------------------------------------
// AnimationLayer
// ---------------------------------------------------------------------------

/// A layer in the animation graph.
///
/// Layers are evaluated in order (0 = bottom, n = top) and blended
/// together according to their blend mode and weight.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationLayer {
    /// Human-readable layer name.
    pub name: String,

    /// Blend weight for this layer (0.0 = no effect, 1.0 = full effect).
    pub weight: f32,

    /// How this layer blends with layers below.
    pub blend_mode: LayerBlendMode,

    /// Optional per-bone mask. If Some, only bones with true are affected.
    /// Length must match skeleton bone count.
    pub bone_mask: Option<Vec<bool>>,

    /// Index of the root node in the graph's node array.
    pub root_node: usize,

    /// Whether this layer is enabled.
    pub enabled: bool,
}

impl AnimationLayer {
    /// Create a new layer with the given name and blend mode.
    pub fn new(name: impl Into<String>, blend_mode: LayerBlendMode) -> Self {
        Self {
            name: name.into(),
            weight: 1.0,
            blend_mode,
            bone_mask: None,
            root_node: 0,
            enabled: true,
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

    /// Create a masked additive layer with the given bone mask.
    pub fn masked_additive(name: impl Into<String>, bone_mask: Vec<bool>) -> Self {
        Self {
            name: name.into(),
            weight: 1.0,
            blend_mode: LayerBlendMode::MaskedAdditive,
            bone_mask: Some(bone_mask),
            root_node: 0,
            enabled: true,
        }
    }

    /// Set the blend weight (clamped to [0, 1]).
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the root node index.
    pub fn with_root_node(mut self, node_index: usize) -> Self {
        self.root_node = node_index;
        self
    }

    /// Set the bone mask.
    pub fn with_bone_mask(mut self, mask: Vec<bool>) -> Self {
        self.bone_mask = Some(mask);
        self
    }

    /// Disable this layer.
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Check if a bone is affected by this layer.
    ///
    /// Returns true if:
    /// - No bone mask is set (all bones affected)
    /// - Bone mask exists and the bone is masked in
    #[inline]
    pub fn affects_bone(&self, bone_index: usize) -> bool {
        match &self.bone_mask {
            None => true,
            Some(mask) => mask.get(bone_index).copied().unwrap_or(false),
        }
    }

    /// Get the effective weight for a specific bone.
    ///
    /// Returns 0.0 if the bone is masked out.
    #[inline]
    pub fn effective_weight(&self, bone_index: usize) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        if self.affects_bone(bone_index) {
            self.weight
        } else {
            0.0
        }
    }
}

impl Default for AnimationLayer {
    fn default() -> Self {
        Self::new("default", LayerBlendMode::Override)
    }
}

// ---------------------------------------------------------------------------
// AnimationNodeType
// ---------------------------------------------------------------------------

/// The type of animation node in the graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AnimationNodeType {
    /// Plays a single animation clip.
    Clip {
        /// Index into the animation clip library.
        clip_index: usize,
        /// Current playback time.
        time: f32,
        /// Playback speed multiplier.
        speed: f32,
        /// Whether to loop the clip.
        looping: bool,
    },

    /// Blends between two animations based on a parameter.
    Blend1D {
        /// Child node indices.
        children: Vec<usize>,
        /// Blend positions for each child.
        positions: Vec<f32>,
        /// Parameter name to read blend value from.
        parameter: String,
    },

    /// 2D blend space (e.g., locomotion by speed and direction).
    Blend2D {
        /// Child node indices.
        children: Vec<usize>,
        /// 2D positions for each child (x, y).
        positions: Vec<(f32, f32)>,
        /// X-axis parameter name.
        parameter_x: String,
        /// Y-axis parameter name.
        parameter_y: String,
    },

    /// State machine node with transitions.
    StateMachine {
        /// Current state index.
        current_state: usize,
        /// State nodes.
        states: Vec<usize>,
        /// Transition conditions (from_state, to_state, condition).
        transitions: Vec<StateTransition>,
    },

    /// Additive overlay on top of base pose.
    AdditiveBlend {
        /// Base pose node.
        base: usize,
        /// Additive pose node.
        additive: usize,
        /// Blend weight (0 = base only, 1 = full additive).
        weight: f32,
    },

    /// References another node (for reuse).
    Reference {
        /// Index of the referenced node.
        target: usize,
    },

    /// Outputs an identity pose.
    Identity,
}

impl Default for AnimationNodeType {
    fn default() -> Self {
        Self::Identity
    }
}

// ---------------------------------------------------------------------------
// StateTransition
// ---------------------------------------------------------------------------

/// A transition between states in a state machine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StateTransition {
    /// Source state index.
    pub from_state: usize,

    /// Destination state index.
    pub to_state: usize,

    /// Parameter that triggers this transition.
    pub trigger_parameter: Option<String>,

    /// Condition to check (parameter_name, operator, threshold).
    pub condition: Option<TransitionCondition>,

    /// Transition duration in seconds.
    pub duration: f32,

    /// Whether this transition can interrupt other transitions.
    pub can_interrupt: bool,
}

impl StateTransition {
    /// Create a new transition between states.
    pub fn new(from_state: usize, to_state: usize) -> Self {
        Self {
            from_state,
            to_state,
            trigger_parameter: None,
            condition: None,
            duration: 0.2,
            can_interrupt: false,
        }
    }

    /// Add a trigger parameter requirement.
    pub fn with_trigger(mut self, parameter: impl Into<String>) -> Self {
        self.trigger_parameter = Some(parameter.into());
        self
    }

    /// Add a condition.
    pub fn with_condition(mut self, condition: TransitionCondition) -> Self {
        self.condition = Some(condition);
        self
    }

    /// Set the transition duration.
    pub fn with_duration(mut self, duration: f32) -> Self {
        self.duration = duration.max(0.0);
        self
    }

    /// Allow this transition to interrupt others.
    pub fn interruptible(mut self) -> Self {
        self.can_interrupt = true;
        self
    }
}

// ---------------------------------------------------------------------------
// TransitionCondition
// ---------------------------------------------------------------------------

/// A condition for state machine transitions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionCondition {
    /// Parameter name to compare.
    pub parameter: String,

    /// Comparison operator.
    pub operator: ComparisonOp,

    /// Threshold value to compare against.
    pub threshold: f32,
}

impl TransitionCondition {
    /// Create a new condition.
    pub fn new(parameter: impl Into<String>, operator: ComparisonOp, threshold: f32) -> Self {
        Self {
            parameter: parameter.into(),
            operator,
            threshold,
        }
    }

    /// Check if the condition is satisfied.
    pub fn evaluate(&self, value: f32) -> bool {
        match self.operator {
            ComparisonOp::Less => value < self.threshold,
            ComparisonOp::LessEqual => value <= self.threshold,
            ComparisonOp::Greater => value > self.threshold,
            ComparisonOp::GreaterEqual => value >= self.threshold,
            ComparisonOp::Equal => (value - self.threshold).abs() < 1e-6,
            ComparisonOp::NotEqual => (value - self.threshold).abs() >= 1e-6,
        }
    }
}

// ---------------------------------------------------------------------------
// ComparisonOp
// ---------------------------------------------------------------------------

/// Comparison operators for transition conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComparisonOp {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
}

impl ComparisonOp {
    /// Get the operator symbol.
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::Equal => "==",
            Self::NotEqual => "!=",
        }
    }
}

// ---------------------------------------------------------------------------
// AnimationNode
// ---------------------------------------------------------------------------

/// A node in the animation graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationNode {
    /// Optional name for debugging.
    pub name: Option<String>,

    /// The type-specific node data.
    pub node_type: AnimationNodeType,

    /// Cached output pose (if evaluated this frame).
    #[serde(skip)]
    pub cached_pose: Option<Pose>,

    /// Frame number when cache was last updated.
    #[serde(skip)]
    pub cache_frame: u64,
}

impl AnimationNode {
    /// Create a new node with the given type.
    pub fn new(node_type: AnimationNodeType) -> Self {
        Self {
            name: None,
            node_type,
            cached_pose: None,
            cache_frame: 0,
        }
    }

    /// Create a clip playback node.
    pub fn clip(clip_index: usize) -> Self {
        Self::new(AnimationNodeType::Clip {
            clip_index,
            time: 0.0,
            speed: 1.0,
            looping: true,
        })
    }

    /// Create an identity node.
    pub fn identity() -> Self {
        Self::new(AnimationNodeType::Identity)
    }

    /// Create a 1D blend node.
    pub fn blend_1d(children: Vec<usize>, positions: Vec<f32>, parameter: impl Into<String>) -> Self {
        Self::new(AnimationNodeType::Blend1D {
            children,
            positions,
            parameter: parameter.into(),
        })
    }

    /// Create a 2D blend node.
    pub fn blend_2d(
        children: Vec<usize>,
        positions: Vec<(f32, f32)>,
        parameter_x: impl Into<String>,
        parameter_y: impl Into<String>,
    ) -> Self {
        Self::new(AnimationNodeType::Blend2D {
            children,
            positions,
            parameter_x: parameter_x.into(),
            parameter_y: parameter_y.into(),
        })
    }

    /// Create a state machine node.
    pub fn state_machine(states: Vec<usize>, transitions: Vec<StateTransition>) -> Self {
        Self::new(AnimationNodeType::StateMachine {
            current_state: 0,
            states,
            transitions,
        })
    }

    /// Create an additive blend node.
    pub fn additive_blend(base: usize, additive: usize, weight: f32) -> Self {
        Self::new(AnimationNodeType::AdditiveBlend {
            base,
            additive,
            weight: weight.clamp(0.0, 1.0),
        })
    }

    /// Set the node name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Invalidate the cached pose.
    pub fn invalidate_cache(&mut self) {
        self.cached_pose = None;
    }
}

impl Default for AnimationNode {
    fn default() -> Self {
        Self::identity()
    }
}

// ---------------------------------------------------------------------------
// AnimationGraph
// ---------------------------------------------------------------------------

/// Runtime animation graph for skeletal animation control.
///
/// The graph contains parameters, layers, and nodes that together
/// define how animations are selected, blended, and evaluated.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AnimationGraph {
    /// Parameters that control graph evaluation.
    pub parameters: Vec<AnimationParameter>,

    /// Animation layers (evaluated bottom to top).
    pub layers: Vec<AnimationLayer>,

    /// Graph nodes (blend trees, state machines, clips).
    pub nodes: Vec<AnimationNode>,

    /// Global dirty flag (any parameter changed).
    #[serde(skip)]
    pub dirty: bool,

    /// Current evaluation frame number.
    #[serde(skip)]
    pub frame: u64,

    /// Number of bones (for pose allocation).
    pub bone_count: usize,

    /// Optional graph name for debugging.
    pub name: Option<String>,
}

impl AnimationGraph {
    /// Create a new empty animation graph.
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
            layers: Vec::new(),
            nodes: Vec::new(),
            dirty: true,
            frame: 0,
            bone_count: 0,
            name: None,
        }
    }

    /// Create a graph with the specified bone count.
    pub fn with_bone_count(bone_count: usize) -> Self {
        Self {
            bone_count,
            ..Self::new()
        }
    }

    /// Set the graph name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    // -------------------------------------------------------------------------
    // Parameter Management
    // -------------------------------------------------------------------------

    /// Add a parameter to the graph.
    ///
    /// Returns the parameter index.
    pub fn add_parameter(&mut self, name: impl Into<String>, value: ParameterValue) -> usize {
        let index = self.parameters.len();
        self.parameters.push(AnimationParameter::new(name, value));
        self.dirty = true;
        index
    }

    /// Get a parameter by name.
    pub fn get_parameter(&self, name: &str) -> Option<&ParameterValue> {
        self.parameters
            .iter()
            .find(|p| p.name == name)
            .map(|p| &p.value)
    }

    /// Get a parameter by index.
    pub fn get_parameter_by_index(&self, index: usize) -> Option<&ParameterValue> {
        self.parameters.get(index).map(|p| &p.value)
    }

    /// Find the index of a parameter by name.
    pub fn find_parameter(&self, name: &str) -> Option<usize> {
        self.parameters.iter().position(|p| p.name == name)
    }

    /// Set a parameter value by name.
    ///
    /// Returns true if the parameter was found and updated.
    pub fn set_parameter(&mut self, name: &str, value: ParameterValue) -> bool {
        if let Some(param) = self.parameters.iter_mut().find(|p| p.name == name) {
            param.set(value);
            if param.dirty {
                self.dirty = true;
            }
            true
        } else {
            false
        }
    }

    /// Set a parameter value by index.
    ///
    /// Returns true if the parameter was found and updated.
    pub fn set_parameter_by_index(&mut self, index: usize, value: ParameterValue) -> bool {
        if let Some(param) = self.parameters.get_mut(index) {
            param.set(value);
            if param.dirty {
                self.dirty = true;
            }
            true
        } else {
            false
        }
    }

    /// Fire a trigger parameter by name.
    ///
    /// Returns true if the trigger was found and fired.
    pub fn trigger(&mut self, name: &str) -> bool {
        if let Some(param) = self.parameters.iter_mut().find(|p| p.name == name) {
            if param.fire() {
                self.dirty = true;
                return true;
            }
        }
        false
    }

    /// Fire a trigger parameter by index.
    pub fn trigger_by_index(&mut self, index: usize) -> bool {
        if let Some(param) = self.parameters.get_mut(index) {
            if param.fire() {
                self.dirty = true;
                return true;
            }
        }
        false
    }

    /// Get a float parameter value (convenience method).
    pub fn get_float(&self, name: &str) -> Option<f32> {
        self.get_parameter(name).and_then(|v| v.as_float())
    }

    /// Get a bool parameter value (convenience method).
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.get_parameter(name).and_then(|v| v.as_bool())
    }

    /// Get an int parameter value (convenience method).
    pub fn get_int(&self, name: &str) -> Option<i32> {
        self.get_parameter(name).and_then(|v| v.as_int())
    }

    /// Get a Vec3 parameter value (convenience method).
    pub fn get_vec3(&self, name: &str) -> Option<Vec3> {
        self.get_parameter(name).and_then(|v| v.as_vec3())
    }

    /// Set a float parameter (convenience method).
    pub fn set_float(&mut self, name: &str, value: f32) -> bool {
        self.set_parameter(name, ParameterValue::Float(value))
    }

    /// Set a bool parameter (convenience method).
    pub fn set_bool(&mut self, name: &str, value: bool) -> bool {
        self.set_parameter(name, ParameterValue::Bool(value))
    }

    /// Set an int parameter (convenience method).
    pub fn set_int(&mut self, name: &str, value: i32) -> bool {
        self.set_parameter(name, ParameterValue::Int(value))
    }

    /// Set a Vec3 parameter (convenience method).
    pub fn set_vec3(&mut self, name: &str, value: Vec3) -> bool {
        self.set_parameter(name, ParameterValue::Vec3(value))
    }

    // -------------------------------------------------------------------------
    // Layer Management
    // -------------------------------------------------------------------------

    /// Add a layer to the graph.
    ///
    /// Returns the layer index.
    pub fn add_layer(&mut self, layer: AnimationLayer) -> usize {
        let index = self.layers.len();
        self.layers.push(layer);
        self.dirty = true;
        index
    }

    /// Get a layer by index.
    pub fn get_layer(&self, index: usize) -> Option<&AnimationLayer> {
        self.layers.get(index)
    }

    /// Get a mutable layer by index.
    pub fn get_layer_mut(&mut self, index: usize) -> Option<&mut AnimationLayer> {
        self.dirty = true;
        self.layers.get_mut(index)
    }

    /// Find a layer by name.
    pub fn find_layer(&self, name: &str) -> Option<usize> {
        self.layers.iter().position(|l| l.name == name)
    }

    /// Set layer weight by index.
    pub fn set_layer_weight(&mut self, index: usize, weight: f32) -> bool {
        if let Some(layer) = self.layers.get_mut(index) {
            layer.weight = weight.clamp(0.0, 1.0);
            self.dirty = true;
            true
        } else {
            false
        }
    }

    /// Enable or disable a layer.
    pub fn set_layer_enabled(&mut self, index: usize, enabled: bool) -> bool {
        if let Some(layer) = self.layers.get_mut(index) {
            layer.enabled = enabled;
            self.dirty = true;
            true
        } else {
            false
        }
    }

    // -------------------------------------------------------------------------
    // Node Management
    // -------------------------------------------------------------------------

    /// Add a node to the graph.
    ///
    /// Returns the node index.
    pub fn add_node(&mut self, node: AnimationNode) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);
        index
    }

    /// Get a node by index.
    pub fn get_node(&self, index: usize) -> Option<&AnimationNode> {
        self.nodes.get(index)
    }

    /// Get a mutable node by index.
    pub fn get_node_mut(&mut self, index: usize) -> Option<&mut AnimationNode> {
        self.nodes.get_mut(index)
    }

    // -------------------------------------------------------------------------
    // Dirty Flag Management
    // -------------------------------------------------------------------------

    /// Check if any parameter has changed since last evaluation.
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clear all dirty flags.
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
        for param in &mut self.parameters {
            // Reset triggers after they've been consumed
            param.reset_trigger();
            param.clear_dirty();
        }
    }

    /// Get list of dirty parameters.
    pub fn dirty_parameters(&self) -> Vec<&str> {
        self.parameters
            .iter()
            .filter(|p| p.dirty)
            .map(|p| p.name.as_str())
            .collect()
    }

    /// Invalidate all node caches.
    pub fn invalidate_caches(&mut self) {
        for node in &mut self.nodes {
            node.invalidate_cache();
        }
    }

    // -------------------------------------------------------------------------
    // Evaluation
    // -------------------------------------------------------------------------

    /// Evaluate the animation graph and return the final blended pose.
    ///
    /// This evaluates all enabled layers bottom-to-top and blends them
    /// according to their blend modes and weights.
    pub fn evaluate(&mut self, dt: f32) -> Pose {
        self.frame += 1;
        let frame = self.frame;

        // Start with identity pose
        let mut result = Pose::new(self.bone_count, PoseType::Current);

        if self.layers.is_empty() || self.bone_count == 0 {
            return result;
        }

        // Evaluate each layer
        let mut first_layer = true;
        for layer_idx in 0..self.layers.len() {
            // Extract layer data before mutable borrow
            let (enabled, weight, root_node, blend_mode) = {
                let layer = &self.layers[layer_idx];
                (layer.enabled, layer.weight, layer.root_node, layer.blend_mode)
            };

            if !enabled || weight <= 0.0 {
                continue;
            }

            // Evaluate the layer's root node
            let layer_pose = self.evaluate_node(root_node, dt, frame);

            // Blend with accumulated result
            if first_layer && blend_mode == LayerBlendMode::Override {
                // First override layer: direct copy with weight
                result = if weight >= 1.0 {
                    layer_pose
                } else {
                    // Blend from identity to layer pose
                    Pose::new(self.bone_count, PoseType::Current).blend(&layer_pose, weight)
                };
                first_layer = false;
            } else {
                // Subsequent layers: blend according to mode
                result = self.blend_layer(&result, &layer_pose, layer_idx);
            }
        }

        result
    }

    /// Evaluate a single node in the graph.
    fn evaluate_node(&mut self, node_index: usize, dt: f32, frame: u64) -> Pose {
        // Check if we have a cached result for this frame
        if let Some(node) = self.nodes.get(node_index) {
            if node.cache_frame == frame {
                if let Some(ref pose) = node.cached_pose {
                    return pose.clone();
                }
            }
        }

        // Need to evaluate
        let result = self.evaluate_node_uncached(node_index, dt, frame);

        // Cache the result
        if let Some(node) = self.nodes.get_mut(node_index) {
            node.cached_pose = Some(result.clone());
            node.cache_frame = frame;
        }

        result
    }

    /// Evaluate a node without caching (internal).
    fn evaluate_node_uncached(&mut self, node_index: usize, dt: f32, frame: u64) -> Pose {
        let node_type = match self.nodes.get(node_index) {
            Some(node) => node.node_type.clone(),
            None => return Pose::new(self.bone_count, PoseType::Current),
        };

        match node_type {
            AnimationNodeType::Identity => Pose::new(self.bone_count, PoseType::Current),

            AnimationNodeType::Clip { clip_index: _, time: _, speed: _, looping: _ } => {
                // TODO: Sample actual clip when clip library is connected
                // For now, return identity pose
                Pose::new(self.bone_count, PoseType::Current)
            }

            AnimationNodeType::Blend1D {
                children,
                positions,
                parameter,
            } => {
                if children.is_empty() {
                    return Pose::new(self.bone_count, PoseType::Current);
                }

                let blend_value = self.get_float(&parameter).unwrap_or(0.0);
                self.evaluate_blend_1d(&children, &positions, blend_value, dt, frame)
            }

            AnimationNodeType::Blend2D {
                children,
                positions,
                parameter_x,
                parameter_y,
            } => {
                if children.is_empty() {
                    return Pose::new(self.bone_count, PoseType::Current);
                }

                let x = self.get_float(&parameter_x).unwrap_or(0.0);
                let y = self.get_float(&parameter_y).unwrap_or(0.0);
                self.evaluate_blend_2d(&children, &positions, x, y, dt, frame)
            }

            AnimationNodeType::StateMachine {
                current_state,
                states,
                transitions: _,
            } => {
                // Evaluate current state
                if let Some(&state_node) = states.get(current_state) {
                    self.evaluate_node(state_node, dt, frame)
                } else {
                    Pose::new(self.bone_count, PoseType::Current)
                }
            }

            AnimationNodeType::AdditiveBlend {
                base,
                additive,
                weight,
            } => {
                let base_pose = self.evaluate_node(base, dt, frame);
                if weight <= 0.0 {
                    return base_pose;
                }

                let additive_pose = self.evaluate_node(additive, dt, frame);
                base_pose.blend_additive(&additive_pose, weight)
            }

            AnimationNodeType::Reference { target } => {
                self.evaluate_node(target, dt, frame)
            }
        }
    }

    /// Evaluate a 1D blend between children.
    fn evaluate_blend_1d(
        &mut self,
        children: &[usize],
        positions: &[f32],
        value: f32,
        dt: f32,
        frame: u64,
    ) -> Pose {
        if children.len() == 1 {
            return self.evaluate_node(children[0], dt, frame);
        }

        // Find the two children to blend between
        let mut left_idx = 0;
        let mut right_idx = 0;

        for (i, &pos) in positions.iter().enumerate() {
            if pos <= value {
                left_idx = i;
            }
            if pos >= value {
                right_idx = i;
                break;
            }
        }

        // If value is outside range, clamp to nearest
        if value <= positions[0] {
            return self.evaluate_node(children[0], dt, frame);
        }
        if value >= positions[positions.len() - 1] {
            return self.evaluate_node(children[children.len() - 1], dt, frame);
        }

        // Calculate blend factor
        let left_pos = positions[left_idx];
        let right_pos = positions[right_idx];
        let t = if (right_pos - left_pos).abs() > 1e-6 {
            (value - left_pos) / (right_pos - left_pos)
        } else {
            0.0
        };

        let left_pose = self.evaluate_node(children[left_idx], dt, frame);
        let right_pose = self.evaluate_node(children[right_idx], dt, frame);

        left_pose.blend(&right_pose, t)
    }

    /// Evaluate a 2D blend between children.
    fn evaluate_blend_2d(
        &mut self,
        children: &[usize],
        positions: &[(f32, f32)],
        x: f32,
        y: f32,
        dt: f32,
        frame: u64,
    ) -> Pose {
        if children.len() == 1 {
            return self.evaluate_node(children[0], dt, frame);
        }

        // Simple inverse distance weighting for 2D blend
        let mut weights: Vec<f32> = Vec::with_capacity(children.len());
        let mut total_weight = 0.0;

        for &(px, py) in positions {
            let dx = x - px;
            let dy = y - py;
            let dist_sq = dx * dx + dy * dy;

            // Inverse distance (with minimum to avoid division by zero)
            let w = 1.0 / (dist_sq + 0.001);
            weights.push(w);
            total_weight += w;
        }

        // Normalize weights
        if total_weight > 0.0 {
            for w in &mut weights {
                *w /= total_weight;
            }
        }

        // Blend all poses
        let mut result = Pose::new(self.bone_count, PoseType::Current);
        let mut accumulated_weight = 0.0;

        for (i, &child_idx) in children.iter().enumerate() {
            let w = weights[i];
            if w <= 1e-6 {
                continue;
            }

            let pose = self.evaluate_node(child_idx, dt, frame);

            if accumulated_weight == 0.0 {
                result = pose;
            } else {
                let blend_factor = w / (accumulated_weight + w);
                result = result.blend(&pose, blend_factor);
            }
            accumulated_weight += w;
        }

        result
    }

    /// Blend a layer pose with the accumulated result.
    fn blend_layer(&self, base: &Pose, layer_pose: &Pose, layer_idx: usize) -> Pose {
        let layer = &self.layers[layer_idx];

        match layer.blend_mode {
            LayerBlendMode::Override => {
                // Per-bone weighted blend
                self.blend_with_mask(base, layer_pose, layer)
            }
            LayerBlendMode::Additive => {
                // Convert to additive and apply
                let mut additive = layer_pose.clone();
                additive.pose_type = PoseType::Additive;
                base.blend_additive(&additive, layer.weight)
            }
            LayerBlendMode::MaskedAdditive => {
                // Additive blend with bone mask
                self.blend_additive_masked(base, layer_pose, layer)
            }
        }
    }

    /// Blend two poses with a bone mask.
    fn blend_with_mask(&self, base: &Pose, overlay: &Pose, layer: &AnimationLayer) -> Pose {
        let bone_count = base.bone_count().min(overlay.bone_count());
        let mut result = Pose::new(bone_count, PoseType::Current);

        for i in 0..bone_count {
            let w = layer.effective_weight(i);
            if w <= 0.0 {
                result.positions[i] = base.positions[i];
                result.rotations[i] = base.rotations[i];
                result.scales[i] = base.scales[i];
            } else if w >= 1.0 {
                result.positions[i] = overlay.positions[i];
                result.rotations[i] = overlay.rotations[i];
                result.scales[i] = overlay.scales[i];
            } else {
                result.positions[i] = lerp_vec3(base.positions[i], overlay.positions[i], w);
                result.rotations[i] = nlerp_quat(base.rotations[i], overlay.rotations[i], w);
                result.scales[i] = lerp_vec3(base.scales[i], overlay.scales[i], w);
            }
        }

        result
    }

    /// Additive blend with bone mask.
    fn blend_additive_masked(&self, base: &Pose, additive: &Pose, layer: &AnimationLayer) -> Pose {
        let bone_count = base.bone_count().min(additive.bone_count());
        let mut result = Pose::new(bone_count, PoseType::Current);

        for i in 0..bone_count {
            let w = layer.effective_weight(i);
            if w <= 0.0 {
                result.positions[i] = base.positions[i];
                result.rotations[i] = base.rotations[i];
                result.scales[i] = base.scales[i];
            } else {
                // Additive blend
                result.positions[i] = base.positions[i] + additive.positions[i] * w;
                let additive_rot = nlerp_quat(Quat::IDENTITY, additive.rotations[i], w);
                result.rotations[i] = (base.rotations[i] * additive_rot).normalize();
                result.scales[i] = base.scales[i] + additive.scales[i] * w;
            }
        }

        result
    }

    // -------------------------------------------------------------------------
    // Serialization
    // -------------------------------------------------------------------------

    /// Serialize the graph to a binary format for fast loading.
    pub fn to_binary(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GRAPH_MAGIC.to_le_bytes());
        data.extend_from_slice(&GRAPH_VERSION.to_le_bytes());
        data.extend_from_slice(&(self.bone_count as u32).to_le_bytes());

        // Parameters count
        data.extend_from_slice(&(self.parameters.len() as u32).to_le_bytes());

        // Layers count
        data.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());

        // Nodes count
        data.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());

        // Serialize parameters (simplified - just names and types for now)
        for param in &self.parameters {
            let name_bytes = param.name.as_bytes();
            data.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            data.extend_from_slice(name_bytes);

            // Type tag
            let type_tag: u8 = match &param.value {
                ParameterValue::Float(_) => 0,
                ParameterValue::Int(_) => 1,
                ParameterValue::Bool(_) => 2,
                ParameterValue::Vec3(_) => 3,
                ParameterValue::Trigger => 4,
            };
            data.push(type_tag);

            // Value
            match &param.value {
                ParameterValue::Float(v) => data.extend_from_slice(&v.to_le_bytes()),
                ParameterValue::Int(v) => data.extend_from_slice(&v.to_le_bytes()),
                ParameterValue::Bool(v) => data.push(if *v { 1 } else { 0 }),
                ParameterValue::Vec3(v) => {
                    data.extend_from_slice(&v.x.to_le_bytes());
                    data.extend_from_slice(&v.y.to_le_bytes());
                    data.extend_from_slice(&v.z.to_le_bytes());
                }
                ParameterValue::Trigger => {} // No value to store
            }
        }

        data
    }

    /// Deserialize the graph from binary format.
    pub fn from_binary(data: &[u8]) -> Result<Self, GraphLoadError> {
        if data.len() < 16 {
            return Err(GraphLoadError::InvalidHeader);
        }

        // Read header
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != GRAPH_MAGIC {
            return Err(GraphLoadError::InvalidMagic(magic));
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != GRAPH_VERSION {
            return Err(GraphLoadError::UnsupportedVersion(version));
        }

        let bone_count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;

        let param_count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;

        let mut offset = 20; // Skip to after header
        let mut parameters = Vec::with_capacity(param_count);

        for _ in 0..param_count {
            if offset + 2 > data.len() {
                return Err(GraphLoadError::UnexpectedEof);
            }

            let name_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;

            if offset + name_len > data.len() {
                return Err(GraphLoadError::UnexpectedEof);
            }

            let name = String::from_utf8_lossy(&data[offset..offset + name_len]).to_string();
            offset += name_len;

            if offset >= data.len() {
                return Err(GraphLoadError::UnexpectedEof);
            }

            let type_tag = data[offset];
            offset += 1;

            let value = match type_tag {
                0 => {
                    if offset + 4 > data.len() {
                        return Err(GraphLoadError::UnexpectedEof);
                    }
                    let v = f32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ]);
                    offset += 4;
                    ParameterValue::Float(v)
                }
                1 => {
                    if offset + 4 > data.len() {
                        return Err(GraphLoadError::UnexpectedEof);
                    }
                    let v = i32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ]);
                    offset += 4;
                    ParameterValue::Int(v)
                }
                2 => {
                    if offset >= data.len() {
                        return Err(GraphLoadError::UnexpectedEof);
                    }
                    let v = data[offset] != 0;
                    offset += 1;
                    ParameterValue::Bool(v)
                }
                3 => {
                    if offset + 12 > data.len() {
                        return Err(GraphLoadError::UnexpectedEof);
                    }
                    let x = f32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ]);
                    let y = f32::from_le_bytes([
                        data[offset + 4],
                        data[offset + 5],
                        data[offset + 6],
                        data[offset + 7],
                    ]);
                    let z = f32::from_le_bytes([
                        data[offset + 8],
                        data[offset + 9],
                        data[offset + 10],
                        data[offset + 11],
                    ]);
                    offset += 12;
                    ParameterValue::Vec3(Vec3::new(x, y, z))
                }
                4 => ParameterValue::Trigger,
                _ => return Err(GraphLoadError::InvalidParameterType(type_tag)),
            };

            parameters.push(AnimationParameter::new(name, value));
        }

        Ok(Self {
            parameters,
            layers: Vec::new(),
            nodes: Vec::new(),
            dirty: true,
            frame: 0,
            bone_count,
            name: None,
        })
    }

    /// Serialize to a human-readable debug format.
    pub fn to_debug_string(&self) -> String {
        let mut s = String::new();

        s.push_str(&format!(
            "AnimationGraph: {} ({} bones)\n",
            self.name.as_deref().unwrap_or("unnamed"),
            self.bone_count
        ));
        s.push_str(&format!("  Dirty: {}\n", self.dirty));
        s.push_str(&format!("  Frame: {}\n", self.frame));

        s.push_str("\n  Parameters:\n");
        for (i, param) in self.parameters.iter().enumerate() {
            s.push_str(&format!(
                "    [{}] {}: {:?} (dirty: {})\n",
                i, param.name, param.value, param.dirty
            ));
        }

        s.push_str("\n  Layers:\n");
        for (i, layer) in self.layers.iter().enumerate() {
            s.push_str(&format!(
                "    [{}] {}: {:?} weight={:.2} enabled={} root_node={}\n",
                i,
                layer.name,
                layer.blend_mode,
                layer.weight,
                layer.enabled,
                layer.root_node
            ));
        }

        s.push_str("\n  Nodes:\n");
        for (i, node) in self.nodes.iter().enumerate() {
            let name = node.name.as_deref().unwrap_or("unnamed");
            s.push_str(&format!("    [{}] {}: {:?}\n", i, name, node.node_type));
        }

        s
    }
}

// ---------------------------------------------------------------------------
// GraphLoadError
// ---------------------------------------------------------------------------

/// Errors that can occur when loading an animation graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphLoadError {
    /// Invalid header structure.
    InvalidHeader,
    /// Invalid magic number in header.
    InvalidMagic(u32),
    /// Unsupported format version.
    UnsupportedVersion(u32),
    /// Unexpected end of file.
    UnexpectedEof,
    /// Invalid parameter type tag.
    InvalidParameterType(u8),
}

impl std::fmt::Display for GraphLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHeader => write!(f, "invalid animation graph header"),
            Self::InvalidMagic(m) => write!(f, "invalid magic number: 0x{:08X}", m),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version: {}", v),
            Self::UnexpectedEof => write!(f, "unexpected end of file"),
            Self::InvalidParameterType(t) => write!(f, "invalid parameter type: {}", t),
        }
    }
}

impl std::error::Error for GraphLoadError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // ===== ParameterValue Tests =====

    #[test]
    fn test_parameter_value_float() {
        let v = ParameterValue::Float(3.14);
        assert_eq!(v.type_name(), "float");
        assert_eq!(v.as_float(), Some(3.14));
        assert_eq!(v.as_int(), None);
        assert!(!v.is_trigger());
    }

    #[test]
    fn test_parameter_value_int() {
        let v = ParameterValue::Int(42);
        assert_eq!(v.type_name(), "int");
        assert_eq!(v.as_int(), Some(42));
        assert_eq!(v.as_float(), None);
    }

    #[test]
    fn test_parameter_value_bool() {
        let v = ParameterValue::Bool(true);
        assert_eq!(v.type_name(), "bool");
        assert_eq!(v.as_bool(), Some(true));

        let v2 = ParameterValue::Bool(false);
        assert_eq!(v2.as_bool(), Some(false));
    }

    #[test]
    fn test_parameter_value_vec3() {
        let v = ParameterValue::Vec3(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(v.type_name(), "vec3");
        assert_eq!(v.as_vec3(), Some(Vec3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn test_parameter_value_trigger() {
        let v = ParameterValue::Trigger;
        assert_eq!(v.type_name(), "trigger");
        assert!(v.is_trigger());
        assert_eq!(v.as_float(), None);
    }

    #[test]
    fn test_parameter_value_default() {
        assert_eq!(ParameterValue::default(), ParameterValue::Float(0.0));
    }

    #[test]
    fn test_parameter_value_default_for_type() {
        assert_eq!(
            ParameterValue::default_for_type("float"),
            Some(ParameterValue::Float(0.0))
        );
        assert_eq!(
            ParameterValue::default_for_type("bool"),
            Some(ParameterValue::Bool(false))
        );
        assert_eq!(ParameterValue::default_for_type("unknown"), None);
    }

    // ===== AnimationParameter Tests =====

    #[test]
    fn test_animation_parameter_new() {
        let p = AnimationParameter::new("speed", ParameterValue::Float(5.0));
        assert_eq!(p.name, "speed");
        assert_eq!(p.value, ParameterValue::Float(5.0));
        assert!(p.dirty); // New parameters start dirty
    }

    #[test]
    fn test_animation_parameter_convenience_constructors() {
        let f = AnimationParameter::float("speed", 1.5);
        assert_eq!(f.value.as_float(), Some(1.5));

        let i = AnimationParameter::int("state", 3);
        assert_eq!(i.value.as_int(), Some(3));

        let b = AnimationParameter::bool("grounded", true);
        assert_eq!(b.value.as_bool(), Some(true));

        let v = AnimationParameter::vec3("velocity", Vec3::X);
        assert_eq!(v.value.as_vec3(), Some(Vec3::X));

        let t = AnimationParameter::trigger("jump");
        assert!(t.value.is_trigger());
        assert!(!t.dirty); // Triggers start not-fired
    }

    #[test]
    fn test_animation_parameter_set() {
        let mut p = AnimationParameter::float("speed", 0.0);
        p.clear_dirty();
        assert!(!p.dirty);

        p.set(ParameterValue::Float(5.0));
        assert!(p.dirty);
        assert_eq!(p.value.as_float(), Some(5.0));
    }

    #[test]
    fn test_animation_parameter_set_no_change() {
        let mut p = AnimationParameter::float("speed", 5.0);
        p.clear_dirty();

        // Setting same value should not dirty
        p.set(ParameterValue::Float(5.0));
        assert!(!p.dirty);
    }

    #[test]
    fn test_animation_parameter_fire_trigger() {
        let mut p = AnimationParameter::trigger("jump");
        assert!(!p.dirty);

        assert!(p.fire());
        assert!(p.dirty);
    }

    #[test]
    fn test_animation_parameter_fire_non_trigger() {
        let mut p = AnimationParameter::float("speed", 0.0);
        p.clear_dirty();

        assert!(!p.fire()); // Should return false for non-triggers
        assert!(!p.dirty);
    }

    #[test]
    fn test_animation_parameter_reset_trigger() {
        let mut p = AnimationParameter::trigger("jump");
        p.fire();
        assert!(p.dirty);

        p.reset_trigger();
        assert!(!p.dirty);
    }

    // ===== LayerBlendMode Tests =====

    #[test]
    fn test_layer_blend_mode_default() {
        assert_eq!(LayerBlendMode::default(), LayerBlendMode::Override);
    }

    #[test]
    fn test_layer_blend_mode_name() {
        assert_eq!(LayerBlendMode::Override.name(), "Override");
        assert_eq!(LayerBlendMode::Additive.name(), "Additive");
        assert_eq!(LayerBlendMode::MaskedAdditive.name(), "MaskedAdditive");
    }

    #[test]
    fn test_layer_blend_mode_is_additive() {
        assert!(!LayerBlendMode::Override.is_additive());
        assert!(LayerBlendMode::Additive.is_additive());
        assert!(LayerBlendMode::MaskedAdditive.is_additive());
    }

    #[test]
    fn test_layer_blend_mode_uses_mask() {
        assert!(!LayerBlendMode::Override.uses_mask());
        assert!(!LayerBlendMode::Additive.uses_mask());
        assert!(LayerBlendMode::MaskedAdditive.uses_mask());
    }

    // ===== AnimationLayer Tests =====

    #[test]
    fn test_animation_layer_new() {
        let layer = AnimationLayer::new("base", LayerBlendMode::Override);
        assert_eq!(layer.name, "base");
        assert_eq!(layer.weight, 1.0);
        assert_eq!(layer.blend_mode, LayerBlendMode::Override);
        assert!(layer.bone_mask.is_none());
        assert_eq!(layer.root_node, 0);
        assert!(layer.enabled);
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
    fn test_animation_layer_masked_additive() {
        let mask = vec![true, true, false, false];
        let layer = AnimationLayer::masked_additive("upper_body", mask.clone());
        assert_eq!(layer.blend_mode, LayerBlendMode::MaskedAdditive);
        assert_eq!(layer.bone_mask, Some(mask));
    }

    #[test]
    fn test_animation_layer_with_weight() {
        let layer = AnimationLayer::base("test").with_weight(0.5);
        assert_eq!(layer.weight, 0.5);

        // Clamping
        let layer2 = AnimationLayer::base("test").with_weight(2.0);
        assert_eq!(layer2.weight, 1.0);

        let layer3 = AnimationLayer::base("test").with_weight(-0.5);
        assert_eq!(layer3.weight, 0.0);
    }

    #[test]
    fn test_animation_layer_with_root_node() {
        let layer = AnimationLayer::base("test").with_root_node(5);
        assert_eq!(layer.root_node, 5);
    }

    #[test]
    fn test_animation_layer_disabled() {
        let layer = AnimationLayer::base("test").disabled();
        assert!(!layer.enabled);
    }

    #[test]
    fn test_animation_layer_affects_bone() {
        // No mask - affects all
        let layer = AnimationLayer::base("test");
        assert!(layer.affects_bone(0));
        assert!(layer.affects_bone(100));

        // With mask
        let mask = vec![true, false, true];
        let masked = AnimationLayer::masked_additive("test", mask);
        assert!(masked.affects_bone(0));
        assert!(!masked.affects_bone(1));
        assert!(masked.affects_bone(2));
        assert!(!masked.affects_bone(3)); // Out of bounds returns false
    }

    #[test]
    fn test_animation_layer_effective_weight() {
        let layer = AnimationLayer::base("test").with_weight(0.8);
        assert_eq!(layer.effective_weight(0), 0.8);

        let disabled = AnimationLayer::base("test").disabled();
        assert_eq!(disabled.effective_weight(0), 0.0);

        let mask = vec![true, false];
        let masked = AnimationLayer::masked_additive("test", mask).with_weight(0.5);
        assert_eq!(masked.effective_weight(0), 0.5);
        assert_eq!(masked.effective_weight(1), 0.0);
    }

    // ===== TransitionCondition Tests =====

    #[test]
    fn test_transition_condition_evaluate() {
        let cond = TransitionCondition::new("speed", ComparisonOp::Greater, 5.0);
        assert!(cond.evaluate(6.0));
        assert!(!cond.evaluate(4.0));
        assert!(!cond.evaluate(5.0));

        let cond2 = TransitionCondition::new("speed", ComparisonOp::GreaterEqual, 5.0);
        assert!(cond2.evaluate(5.0));
        assert!(cond2.evaluate(6.0));

        let cond3 = TransitionCondition::new("speed", ComparisonOp::Less, 5.0);
        assert!(cond3.evaluate(4.0));
        assert!(!cond3.evaluate(5.0));

        let cond4 = TransitionCondition::new("speed", ComparisonOp::Equal, 5.0);
        assert!(cond4.evaluate(5.0));
        assert!(!cond4.evaluate(5.001));

        let cond5 = TransitionCondition::new("speed", ComparisonOp::NotEqual, 5.0);
        assert!(!cond5.evaluate(5.0));
        assert!(cond5.evaluate(4.0));
    }

    // ===== ComparisonOp Tests =====

    #[test]
    fn test_comparison_op_symbol() {
        assert_eq!(ComparisonOp::Less.symbol(), "<");
        assert_eq!(ComparisonOp::LessEqual.symbol(), "<=");
        assert_eq!(ComparisonOp::Greater.symbol(), ">");
        assert_eq!(ComparisonOp::GreaterEqual.symbol(), ">=");
        assert_eq!(ComparisonOp::Equal.symbol(), "==");
        assert_eq!(ComparisonOp::NotEqual.symbol(), "!=");
    }

    // ===== AnimationNode Tests =====

    #[test]
    fn test_animation_node_identity() {
        let node = AnimationNode::identity();
        assert!(matches!(node.node_type, AnimationNodeType::Identity));
        assert!(node.name.is_none());
        assert!(node.cached_pose.is_none());
    }

    #[test]
    fn test_animation_node_clip() {
        let node = AnimationNode::clip(5);
        match &node.node_type {
            AnimationNodeType::Clip {
                clip_index,
                time,
                speed,
                looping,
            } => {
                assert_eq!(*clip_index, 5);
                assert_eq!(*time, 0.0);
                assert_eq!(*speed, 1.0);
                assert!(*looping);
            }
            _ => panic!("expected Clip node"),
        }
    }

    #[test]
    fn test_animation_node_blend_1d() {
        let node = AnimationNode::blend_1d(vec![0, 1, 2], vec![0.0, 0.5, 1.0], "speed");
        match &node.node_type {
            AnimationNodeType::Blend1D {
                children,
                positions,
                parameter,
            } => {
                assert_eq!(children.len(), 3);
                assert_eq!(positions.len(), 3);
                assert_eq!(parameter, "speed");
            }
            _ => panic!("expected Blend1D node"),
        }
    }

    #[test]
    fn test_animation_node_with_name() {
        let node = AnimationNode::identity().with_name("idle");
        assert_eq!(node.name, Some("idle".to_string()));
    }

    #[test]
    fn test_animation_node_invalidate_cache() {
        let mut node = AnimationNode::identity();
        node.cached_pose = Some(Pose::new(2, PoseType::Current));
        assert!(node.cached_pose.is_some());

        node.invalidate_cache();
        assert!(node.cached_pose.is_none());
    }

    // ===== AnimationGraph Tests =====

    #[test]
    fn test_animation_graph_new() {
        let graph = AnimationGraph::new();
        assert!(graph.parameters.is_empty());
        assert!(graph.layers.is_empty());
        assert!(graph.nodes.is_empty());
        assert!(graph.dirty);
        assert_eq!(graph.bone_count, 0);
    }

    #[test]
    fn test_animation_graph_with_bone_count() {
        let graph = AnimationGraph::with_bone_count(50);
        assert_eq!(graph.bone_count, 50);
    }

    #[test]
    fn test_animation_graph_add_parameter() {
        let mut graph = AnimationGraph::new();
        let idx = graph.add_parameter("speed", ParameterValue::Float(0.0));
        assert_eq!(idx, 0);
        assert_eq!(graph.parameters.len(), 1);
    }

    #[test]
    fn test_animation_graph_get_parameter() {
        let mut graph = AnimationGraph::new();
        graph.add_parameter("speed", ParameterValue::Float(5.0));

        assert_eq!(graph.get_parameter("speed"), Some(&ParameterValue::Float(5.0)));
        assert_eq!(graph.get_parameter("unknown"), None);
    }

    #[test]
    fn test_animation_graph_get_parameter_by_index() {
        let mut graph = AnimationGraph::new();
        graph.add_parameter("speed", ParameterValue::Float(5.0));

        assert_eq!(
            graph.get_parameter_by_index(0),
            Some(&ParameterValue::Float(5.0))
        );
        assert_eq!(graph.get_parameter_by_index(1), None);
    }

    #[test]
    fn test_animation_graph_set_parameter() {
        let mut graph = AnimationGraph::new();
        graph.add_parameter("speed", ParameterValue::Float(0.0));
        graph.clear_dirty();

        assert!(graph.set_parameter("speed", ParameterValue::Float(5.0)));
        assert!(graph.dirty);
        assert_eq!(graph.get_float("speed"), Some(5.0));

        assert!(!graph.set_parameter("unknown", ParameterValue::Float(1.0)));
    }

    #[test]
    fn test_animation_graph_trigger() {
        let mut graph = AnimationGraph::new();
        graph.add_parameter("jump", ParameterValue::Trigger);
        graph.clear_dirty();

        assert!(graph.trigger("jump"));
        assert!(graph.dirty);

        // Non-trigger parameter
        graph.add_parameter("speed", ParameterValue::Float(0.0));
        assert!(!graph.trigger("speed"));
    }

    #[test]
    fn test_animation_graph_convenience_getters() {
        let mut graph = AnimationGraph::new();
        graph.add_parameter("speed", ParameterValue::Float(5.0));
        graph.add_parameter("state", ParameterValue::Int(3));
        graph.add_parameter("grounded", ParameterValue::Bool(true));
        graph.add_parameter("velocity", ParameterValue::Vec3(Vec3::X));

        assert_eq!(graph.get_float("speed"), Some(5.0));
        assert_eq!(graph.get_int("state"), Some(3));
        assert_eq!(graph.get_bool("grounded"), Some(true));
        assert_eq!(graph.get_vec3("velocity"), Some(Vec3::X));
    }

    #[test]
    fn test_animation_graph_convenience_setters() {
        let mut graph = AnimationGraph::new();
        graph.add_parameter("speed", ParameterValue::Float(0.0));
        graph.add_parameter("state", ParameterValue::Int(0));
        graph.add_parameter("grounded", ParameterValue::Bool(false));
        graph.add_parameter("velocity", ParameterValue::Vec3(Vec3::ZERO));

        assert!(graph.set_float("speed", 5.0));
        assert!(graph.set_int("state", 3));
        assert!(graph.set_bool("grounded", true));
        assert!(graph.set_vec3("velocity", Vec3::X));

        assert_eq!(graph.get_float("speed"), Some(5.0));
        assert_eq!(graph.get_int("state"), Some(3));
        assert_eq!(graph.get_bool("grounded"), Some(true));
        assert_eq!(graph.get_vec3("velocity"), Some(Vec3::X));
    }

    #[test]
    fn test_animation_graph_add_layer() {
        let mut graph = AnimationGraph::new();
        let idx = graph.add_layer(AnimationLayer::base("locomotion"));
        assert_eq!(idx, 0);
        assert_eq!(graph.layers.len(), 1);
    }

    #[test]
    fn test_animation_graph_get_layer() {
        let mut graph = AnimationGraph::new();
        graph.add_layer(AnimationLayer::base("locomotion"));

        assert!(graph.get_layer(0).is_some());
        assert_eq!(graph.get_layer(0).unwrap().name, "locomotion");
        assert!(graph.get_layer(1).is_none());
    }

    #[test]
    fn test_animation_graph_find_layer() {
        let mut graph = AnimationGraph::new();
        graph.add_layer(AnimationLayer::base("locomotion"));
        graph.add_layer(AnimationLayer::additive("breathing"));

        assert_eq!(graph.find_layer("locomotion"), Some(0));
        assert_eq!(graph.find_layer("breathing"), Some(1));
        assert_eq!(graph.find_layer("unknown"), None);
    }

    #[test]
    fn test_animation_graph_set_layer_weight() {
        let mut graph = AnimationGraph::new();
        graph.add_layer(AnimationLayer::base("test"));
        graph.clear_dirty();

        assert!(graph.set_layer_weight(0, 0.5));
        assert!(graph.dirty);
        assert_eq!(graph.get_layer(0).unwrap().weight, 0.5);

        assert!(!graph.set_layer_weight(1, 0.5));
    }

    #[test]
    fn test_animation_graph_set_layer_enabled() {
        let mut graph = AnimationGraph::new();
        graph.add_layer(AnimationLayer::base("test"));

        assert!(graph.set_layer_enabled(0, false));
        assert!(!graph.get_layer(0).unwrap().enabled);
    }

    #[test]
    fn test_animation_graph_add_node() {
        let mut graph = AnimationGraph::new();
        let idx = graph.add_node(AnimationNode::identity());
        assert_eq!(idx, 0);
        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn test_animation_graph_dirty_flag() {
        let mut graph = AnimationGraph::new();
        assert!(graph.is_dirty());

        graph.clear_dirty();
        assert!(!graph.is_dirty());

        graph.add_parameter("speed", ParameterValue::Float(0.0));
        assert!(graph.is_dirty());
    }

    #[test]
    fn test_animation_graph_dirty_parameters() {
        let mut graph = AnimationGraph::new();
        graph.add_parameter("speed", ParameterValue::Float(0.0));
        graph.add_parameter("state", ParameterValue::Int(0));
        graph.clear_dirty();

        graph.set_float("speed", 5.0);
        let dirty = graph.dirty_parameters();
        assert_eq!(dirty.len(), 1);
        assert!(dirty.contains(&"speed"));
    }

    #[test]
    fn test_animation_graph_evaluate_empty() {
        let mut graph = AnimationGraph::with_bone_count(10);
        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 10);
    }

    #[test]
    fn test_animation_graph_evaluate_single_layer() {
        let mut graph = AnimationGraph::with_bone_count(5);
        graph.add_node(AnimationNode::identity());
        graph.add_layer(AnimationLayer::base("locomotion").with_root_node(0));

        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 5);
        assert_eq!(pose.pose_type, PoseType::Current);
    }

    #[test]
    fn test_animation_graph_evaluate_multiple_layers() {
        let mut graph = AnimationGraph::with_bone_count(5);
        graph.add_node(AnimationNode::identity());
        graph.add_layer(AnimationLayer::base("base").with_root_node(0));
        graph.add_layer(AnimationLayer::additive("additive").with_root_node(0));

        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 5);
    }

    #[test]
    fn test_animation_graph_evaluate_disabled_layer() {
        let mut graph = AnimationGraph::with_bone_count(5);
        graph.add_node(AnimationNode::identity());
        graph.add_layer(AnimationLayer::base("disabled").disabled());

        let pose = graph.evaluate(0.016);
        // Should still return identity pose
        assert_eq!(pose.bone_count(), 5);
    }

    #[test]
    fn test_animation_graph_evaluate_zero_weight_layer() {
        let mut graph = AnimationGraph::with_bone_count(5);
        graph.add_node(AnimationNode::identity());
        graph.add_layer(AnimationLayer::base("zero").with_weight(0.0));

        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 5);
    }

    #[test]
    fn test_animation_graph_evaluate_with_bone_mask() {
        let mut graph = AnimationGraph::with_bone_count(4);
        graph.add_node(AnimationNode::identity());

        let mask = vec![true, true, false, false];
        graph.add_layer(
            AnimationLayer::masked_additive("upper", mask)
                .with_root_node(0)
                .with_weight(1.0),
        );

        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 4);
    }

    #[test]
    fn test_animation_graph_frame_counter() {
        let mut graph = AnimationGraph::with_bone_count(5);
        graph.add_node(AnimationNode::identity());
        graph.add_layer(AnimationLayer::base("test"));

        assert_eq!(graph.frame, 0);
        graph.evaluate(0.016);
        assert_eq!(graph.frame, 1);
        graph.evaluate(0.016);
        assert_eq!(graph.frame, 2);
    }

    // ===== Blend Tests =====

    #[test]
    fn test_animation_graph_blend_1d_single_child() {
        let mut graph = AnimationGraph::with_bone_count(2);
        graph.add_parameter("speed", ParameterValue::Float(0.5));
        graph.add_node(AnimationNode::identity().with_name("idle"));
        graph.add_node(AnimationNode::blend_1d(vec![0], vec![0.0], "speed"));
        graph.add_layer(AnimationLayer::base("test").with_root_node(1));

        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 2);
    }

    #[test]
    fn test_animation_graph_blend_1d_multiple_children() {
        let mut graph = AnimationGraph::with_bone_count(2);
        graph.add_parameter("speed", ParameterValue::Float(0.5));
        graph.add_node(AnimationNode::identity().with_name("idle"));
        graph.add_node(AnimationNode::identity().with_name("walk"));
        graph.add_node(AnimationNode::identity().with_name("run"));
        graph.add_node(AnimationNode::blend_1d(vec![0, 1, 2], vec![0.0, 0.5, 1.0], "speed"));
        graph.add_layer(AnimationLayer::base("test").with_root_node(3));

        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 2);
    }

    #[test]
    fn test_animation_graph_blend_2d() {
        let mut graph = AnimationGraph::with_bone_count(2);
        graph.add_parameter("move_x", ParameterValue::Float(0.0));
        graph.add_parameter("move_y", ParameterValue::Float(0.0));
        graph.add_node(AnimationNode::identity());
        graph.add_node(AnimationNode::identity());
        graph.add_node(AnimationNode::blend_2d(
            vec![0, 1],
            vec![(0.0, 0.0), (1.0, 1.0)],
            "move_x",
            "move_y",
        ));
        graph.add_layer(AnimationLayer::base("test").with_root_node(2));

        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 2);
    }

    #[test]
    fn test_animation_graph_additive_blend_node() {
        let mut graph = AnimationGraph::with_bone_count(2);
        graph.add_node(AnimationNode::identity().with_name("base"));
        graph.add_node(AnimationNode::identity().with_name("additive"));
        graph.add_node(AnimationNode::additive_blend(0, 1, 0.5));
        graph.add_layer(AnimationLayer::base("test").with_root_node(2));

        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 2);
    }

    #[test]
    fn test_animation_graph_reference_node() {
        let mut graph = AnimationGraph::with_bone_count(2);
        graph.add_node(AnimationNode::identity().with_name("target"));
        graph.add_node(AnimationNode::new(AnimationNodeType::Reference { target: 0 }));
        graph.add_layer(AnimationLayer::base("test").with_root_node(1));

        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 2);
    }

    // ===== Serialization Tests =====

    #[test]
    fn test_animation_graph_to_binary_empty() {
        let graph = AnimationGraph::with_bone_count(10);
        let data = graph.to_binary();

        // Should have header
        assert!(data.len() >= 16);

        // Check magic
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(magic, GRAPH_MAGIC);
    }

    #[test]
    fn test_animation_graph_binary_roundtrip() {
        let mut graph = AnimationGraph::with_bone_count(50);
        graph.add_parameter("speed", ParameterValue::Float(5.0));
        graph.add_parameter("state", ParameterValue::Int(3));
        graph.add_parameter("grounded", ParameterValue::Bool(true));
        graph.add_parameter("velocity", ParameterValue::Vec3(Vec3::new(1.0, 2.0, 3.0)));
        graph.add_parameter("jump", ParameterValue::Trigger);

        let data = graph.to_binary();
        let recovered = AnimationGraph::from_binary(&data).expect("should parse");

        assert_eq!(recovered.bone_count, 50);
        assert_eq!(recovered.parameters.len(), 5);
        assert_eq!(recovered.get_float("speed"), Some(5.0));
        assert_eq!(recovered.get_int("state"), Some(3));
        assert_eq!(recovered.get_bool("grounded"), Some(true));
        assert_eq!(recovered.get_vec3("velocity"), Some(Vec3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn test_animation_graph_from_binary_invalid_magic() {
        let data = vec![0, 0, 0, 0, 1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0];
        let result = AnimationGraph::from_binary(&data);
        assert!(matches!(result, Err(GraphLoadError::InvalidMagic(_))));
    }

    #[test]
    fn test_animation_graph_from_binary_invalid_header() {
        let data = vec![0; 8]; // Too short
        let result = AnimationGraph::from_binary(&data);
        assert!(matches!(result, Err(GraphLoadError::InvalidHeader)));
    }

    #[test]
    fn test_animation_graph_to_debug_string() {
        let mut graph = AnimationGraph::with_bone_count(10).with_name("test_graph");
        graph.add_parameter("speed", ParameterValue::Float(5.0));
        graph.add_layer(AnimationLayer::base("locomotion"));
        graph.add_node(AnimationNode::identity().with_name("idle"));

        let debug = graph.to_debug_string();
        assert!(debug.contains("test_graph"));
        assert!(debug.contains("10 bones"));
        assert!(debug.contains("speed"));
        assert!(debug.contains("locomotion"));
        assert!(debug.contains("idle"));
    }

    // ===== JSON Serialization Tests =====

    #[test]
    fn test_animation_graph_json_roundtrip() {
        let mut graph = AnimationGraph::with_bone_count(20);
        graph.add_parameter("speed", ParameterValue::Float(5.0));
        graph.add_layer(AnimationLayer::base("base"));
        graph.add_node(AnimationNode::identity());

        let json = serde_json::to_string(&graph).expect("serialize");
        let recovered: AnimationGraph = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(recovered.bone_count, 20);
        assert_eq!(recovered.parameters.len(), 1);
        assert_eq!(recovered.layers.len(), 1);
        assert_eq!(recovered.nodes.len(), 1);
    }

    #[test]
    fn test_parameter_value_json_roundtrip() {
        let values = vec![
            ParameterValue::Float(3.14),
            ParameterValue::Int(-42),
            ParameterValue::Bool(true),
            ParameterValue::Vec3(Vec3::new(1.0, 2.0, 3.0)),
            ParameterValue::Trigger,
        ];

        for v in values {
            let json = serde_json::to_string(&v).expect("serialize");
            let recovered: ParameterValue = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(v, recovered);
        }
    }

    #[test]
    fn test_layer_blend_mode_json_roundtrip() {
        let modes = vec![
            LayerBlendMode::Override,
            LayerBlendMode::Additive,
            LayerBlendMode::MaskedAdditive,
        ];

        for mode in modes {
            let json = serde_json::to_string(&mode).expect("serialize");
            let recovered: LayerBlendMode = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(mode, recovered);
        }
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_empty_graph_evaluate() {
        let mut graph = AnimationGraph::new();
        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 0);
    }

    #[test]
    fn test_graph_with_no_nodes() {
        let mut graph = AnimationGraph::with_bone_count(5);
        graph.add_layer(AnimationLayer::base("test").with_root_node(999)); // Invalid node

        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 5);
    }

    #[test]
    fn test_all_layers_disabled() {
        let mut graph = AnimationGraph::with_bone_count(5);
        graph.add_node(AnimationNode::identity());
        graph.add_layer(AnimationLayer::base("test1").disabled());
        graph.add_layer(AnimationLayer::base("test2").disabled());

        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 5);
    }

    #[test]
    fn test_state_machine_node() {
        let mut graph = AnimationGraph::with_bone_count(2);
        graph.add_node(AnimationNode::identity().with_name("idle"));
        graph.add_node(AnimationNode::identity().with_name("walk"));
        graph.add_node(AnimationNode::state_machine(
            vec![0, 1],
            vec![StateTransition::new(0, 1).with_trigger("start_walk")],
        ));
        graph.add_layer(AnimationLayer::base("test").with_root_node(2));

        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 2);
    }

    #[test]
    fn test_invalidate_caches() {
        let mut graph = AnimationGraph::with_bone_count(2);
        graph.add_node(AnimationNode::identity());
        graph.add_layer(AnimationLayer::base("test").with_root_node(0));

        // Evaluate to populate cache
        graph.evaluate(0.016);
        assert!(graph.nodes[0].cached_pose.is_some());

        // Invalidate
        graph.invalidate_caches();
        assert!(graph.nodes[0].cached_pose.is_none());
    }

    #[test]
    fn test_node_cache_reuse() {
        let mut graph = AnimationGraph::with_bone_count(2);
        graph.add_node(AnimationNode::identity());
        graph.add_layer(AnimationLayer::base("test").with_root_node(0));

        // First evaluation
        graph.evaluate(0.016);
        let frame1 = graph.nodes[0].cache_frame;

        // Second evaluation in same frame shouldn't increment
        // (Actually it does increment the graph frame each evaluate)
        graph.evaluate(0.016);
        let frame2 = graph.nodes[0].cache_frame;

        // Each evaluate increments the frame
        assert_eq!(frame2, frame1 + 1);
    }

    #[test]
    fn test_large_bone_count() {
        let mut graph = AnimationGraph::with_bone_count(256);
        graph.add_node(AnimationNode::identity());
        graph.add_layer(AnimationLayer::base("test"));

        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 256);
    }

    #[test]
    fn test_many_parameters() {
        let mut graph = AnimationGraph::new();
        for i in 0..100 {
            graph.add_parameter(format!("param_{}", i), ParameterValue::Float(i as f32));
        }

        assert_eq!(graph.parameters.len(), 100);
        assert_eq!(graph.get_float("param_50"), Some(50.0));
    }

    #[test]
    fn test_many_layers() {
        let mut graph = AnimationGraph::with_bone_count(5);
        graph.add_node(AnimationNode::identity());

        for i in 0..16 {
            graph.add_layer(
                AnimationLayer::base(format!("layer_{}", i))
                    .with_weight(0.1)
                    .with_root_node(0),
            );
        }

        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 5);
    }

    #[test]
    fn test_blend_1d_edge_values() {
        let mut graph = AnimationGraph::with_bone_count(2);
        graph.add_parameter("blend", ParameterValue::Float(-1.0)); // Below range
        graph.add_node(AnimationNode::identity());
        graph.add_node(AnimationNode::identity());
        graph.add_node(AnimationNode::blend_1d(vec![0, 1], vec![0.0, 1.0], "blend"));
        graph.add_layer(AnimationLayer::base("test").with_root_node(2));

        // Should clamp to first child
        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 2);

        // Set above range
        graph.set_float("blend", 2.0);
        let pose2 = graph.evaluate(0.016);
        assert_eq!(pose2.bone_count(), 2);
    }

    #[test]
    fn test_blend_2d_single_child() {
        let mut graph = AnimationGraph::with_bone_count(2);
        graph.add_parameter("x", ParameterValue::Float(0.5));
        graph.add_parameter("y", ParameterValue::Float(0.5));
        graph.add_node(AnimationNode::identity());
        graph.add_node(AnimationNode::blend_2d(vec![0], vec![(0.0, 0.0)], "x", "y"));
        graph.add_layer(AnimationLayer::base("test").with_root_node(1));

        let pose = graph.evaluate(0.016);
        assert_eq!(pose.bone_count(), 2);
    }

    #[test]
    fn test_clear_dirty_resets_triggers() {
        let mut graph = AnimationGraph::new();
        graph.add_parameter("jump", ParameterValue::Trigger);

        graph.trigger("jump");
        assert!(graph.parameters[0].dirty);

        graph.clear_dirty();
        assert!(!graph.parameters[0].dirty);
    }
}
