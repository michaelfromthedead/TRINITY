//! FlowForge Node Execution System
//!
//! This module provides runtime execution for FlowForge visual scripting nodes.
//! It includes:
//! - NodeExecutor trait for node behavior
//! - ExecutionContext for runtime state
//! - NodeRegistry for node type management
//! - 40+ built-in node types across 7 categories
//!
//! # Categories
//!
//! - **Events (6)**: OnStart, OnTick, OnKeyPressed, OnKeyReleased, OnCollision, CustomEvent
//! - **Actions (8)**: SetPosition, SetRotation, SetScale, PlaySound, SpawnEntity, DestroyEntity, SendMessage, SetVariable
//! - **Conditions (6)**: Compare, IsNull, IsEqual, IsGreater, IsLess, InRange
//! - **Math (10)**: Add, Subtract, Multiply, Divide, Abs, Sin, Cos, Lerp, Clamp, Random
//! - **Flow (6)**: Sequence, Branch, ForLoop, WhileLoop, DoOnce, Delay
//! - **ECS (6)**: GetComponent, SetComponent, HasComponent, AddComponent, RemoveComponent, QueryEntities
//! - **Debug (4)**: Print, Log, Assert, Breakpoint

use crate::flowforge::{Node, NodeId, PortId, PropertyValue, PortType};
use std::collections::HashMap;
use std::any::Any;

// ---------------------------------------------------------------------------
// Runtime Value Type
// ---------------------------------------------------------------------------

/// A runtime value that can flow through node ports.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Value {
    /// No value / null.
    None,
    /// Boolean value.
    Bool(bool),
    /// Integer value.
    Int(i32),
    /// Floating-point value.
    Float(f32),
    /// String value.
    String(String),
    /// 3D vector (x, y, z).
    Vec3([f32; 3]),
    /// Entity ID reference.
    Entity(u64),
    /// List of values.
    List(Vec<Value>),
}

impl Default for Value {
    fn default() -> Self {
        Value::None
    }
}

impl Value {
    /// Convert to bool, with type coercion.
    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::String(s) => !s.is_empty(),
            Value::None => false,
            _ => true,
        }
    }

    /// Convert to int, with type coercion.
    pub fn as_int(&self) -> i32 {
        match self {
            Value::Int(i) => *i,
            Value::Float(f) => *f as i32,
            Value::Bool(b) => if *b { 1 } else { 0 },
            Value::String(s) => s.parse().unwrap_or(0),
            _ => 0,
        }
    }

    /// Convert to float, with type coercion.
    pub fn as_float(&self) -> f32 {
        match self {
            Value::Float(f) => *f,
            Value::Int(i) => *i as f32,
            Value::Bool(b) => if *b { 1.0 } else { 0.0 },
            Value::String(s) => s.parse().unwrap_or(0.0),
            _ => 0.0,
        }
    }

    /// Convert to string representation.
    pub fn as_string(&self) -> String {
        match self {
            Value::String(s) => s.clone(),
            Value::Bool(b) => b.to_string(),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Vec3(v) => format!("({}, {}, {})", v[0], v[1], v[2]),
            Value::Entity(e) => format!("Entity({})", e),
            Value::None => "None".to_string(),
            Value::List(l) => format!("[{} items]", l.len()),
        }
    }

    /// Get as Vec3, returns [0,0,0] if not a Vec3.
    pub fn as_vec3(&self) -> [f32; 3] {
        match self {
            Value::Vec3(v) => *v,
            _ => [0.0, 0.0, 0.0],
        }
    }

    /// Get as entity ID, returns 0 if not an entity.
    pub fn as_entity(&self) -> u64 {
        match self {
            Value::Entity(e) => *e,
            Value::Int(i) => *i as u64,
            _ => 0,
        }
    }

    /// Check if value is None.
    pub fn is_none(&self) -> bool {
        matches!(self, Value::None)
    }

    /// Get the PortType corresponding to this value.
    pub fn port_type(&self) -> PortType {
        match self {
            Value::None => PortType::Any,
            Value::Bool(_) => PortType::Bool,
            Value::Int(_) => PortType::Int,
            Value::Float(_) => PortType::Float,
            Value::String(_) => PortType::String,
            Value::Vec3(_) => PortType::Vec3,
            Value::Entity(_) => PortType::Entity,
            Value::List(_) => PortType::Any,
        }
    }
}

impl From<PropertyValue> for Value {
    fn from(pv: PropertyValue) -> Self {
        match pv {
            PropertyValue::Bool(b) => Value::Bool(b),
            PropertyValue::Int(i) => Value::Int(i),
            PropertyValue::Float(f) => Value::Float(f),
            PropertyValue::String(s) => Value::String(s),
            PropertyValue::Vec3(v) => Value::Vec3(v),
            PropertyValue::Color(c) => Value::Vec3([c[0], c[1], c[2]]),
            PropertyValue::Enum(i) => Value::Int(i as i32),
        }
    }
}

// ---------------------------------------------------------------------------
// Execution Context
// ---------------------------------------------------------------------------

/// Runtime context for node execution.
#[derive(Debug, Default)]
pub struct ExecutionContext {
    /// Input values by port ID.
    inputs: HashMap<PortId, Value>,
    /// Output values by port ID.
    outputs: HashMap<PortId, Value>,
    /// Global variables.
    variables: HashMap<String, Value>,
    /// Current delta time (for OnTick).
    pub delta_time: f32,
    /// Current frame number.
    pub frame: u64,
    /// Events that have fired this frame.
    events: HashMap<String, bool>,
    /// Keys currently pressed.
    keys_pressed: HashMap<String, bool>,
    /// Keys released this frame.
    keys_released: HashMap<String, bool>,
    /// DoOnce state per node.
    do_once_state: HashMap<NodeId, bool>,
    /// Delay timers per node.
    delay_timers: HashMap<NodeId, f32>,
    /// Loop iteration counters.
    loop_counters: HashMap<NodeId, i32>,
    /// Debug log output.
    pub debug_log: Vec<String>,
    /// Random seed for deterministic random.
    random_seed: u64,
    /// Entity components (simplified ECS mock).
    components: HashMap<(u64, String), Value>,
}

impl ExecutionContext {
    /// Create a new execution context.
    pub fn new() -> Self {
        Self {
            random_seed: 12345,
            ..Default::default()
        }
    }

    /// Set an input value.
    pub fn set_input(&mut self, port_id: PortId, value: Value) {
        self.inputs.insert(port_id, value);
    }

    /// Get an input value.
    pub fn get_input(&self, port_id: PortId) -> Value {
        self.inputs.get(&port_id).cloned().unwrap_or(Value::None)
    }

    /// Set an output value.
    pub fn set_output(&mut self, port_id: PortId, value: Value) {
        self.outputs.insert(port_id, value);
    }

    /// Get an output value.
    pub fn get_output(&self, port_id: PortId) -> Value {
        self.outputs.get(&port_id).cloned().unwrap_or(Value::None)
    }

    /// Clear all inputs and outputs (between node executions).
    pub fn clear_io(&mut self) {
        self.inputs.clear();
        self.outputs.clear();
    }

    /// Set a variable.
    pub fn set_variable(&mut self, name: &str, value: Value) {
        self.variables.insert(name.to_string(), value);
    }

    /// Get a variable.
    pub fn get_variable(&self, name: &str) -> Value {
        self.variables.get(name).cloned().unwrap_or(Value::None)
    }

    /// Fire an event.
    pub fn fire_event(&mut self, name: &str) {
        self.events.insert(name.to_string(), true);
    }

    /// Check if an event fired this frame.
    pub fn event_fired(&self, name: &str) -> bool {
        *self.events.get(name).unwrap_or(&false)
    }

    /// Clear events (called at frame end).
    pub fn clear_events(&mut self) {
        self.events.clear();
        self.keys_released.clear();
    }

    /// Set key pressed state.
    pub fn set_key_pressed(&mut self, key: &str, pressed: bool) {
        if !pressed && *self.keys_pressed.get(key).unwrap_or(&false) {
            self.keys_released.insert(key.to_string(), true);
        }
        self.keys_pressed.insert(key.to_string(), pressed);
    }

    /// Check if a key is pressed.
    pub fn is_key_pressed(&self, key: &str) -> bool {
        *self.keys_pressed.get(key).unwrap_or(&false)
    }

    /// Check if a key was released this frame.
    pub fn is_key_released(&self, key: &str) -> bool {
        *self.keys_released.get(key).unwrap_or(&false)
    }

    /// Get DoOnce state for a node.
    pub fn get_do_once(&self, node_id: NodeId) -> bool {
        *self.do_once_state.get(&node_id).unwrap_or(&false)
    }

    /// Set DoOnce state for a node.
    pub fn set_do_once(&mut self, node_id: NodeId, done: bool) {
        self.do_once_state.insert(node_id, done);
    }

    /// Get delay timer for a node.
    pub fn get_delay_timer(&self, node_id: NodeId) -> f32 {
        *self.delay_timers.get(&node_id).unwrap_or(&0.0)
    }

    /// Set delay timer for a node.
    pub fn set_delay_timer(&mut self, node_id: NodeId, time: f32) {
        self.delay_timers.insert(node_id, time);
    }

    /// Get loop counter for a node.
    pub fn get_loop_counter(&self, node_id: NodeId) -> i32 {
        *self.loop_counters.get(&node_id).unwrap_or(&0)
    }

    /// Set loop counter for a node.
    pub fn set_loop_counter(&mut self, node_id: NodeId, count: i32) {
        self.loop_counters.insert(node_id, count);
    }

    /// Generate a pseudo-random float [0, 1).
    pub fn random(&mut self) -> f32 {
        // Simple LCG for determinism
        self.random_seed = self.random_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.random_seed >> 33) as f32) / (u32::MAX as f32)
    }

    /// Add debug log message.
    pub fn log(&mut self, message: String) {
        self.debug_log.push(message);
    }

    /// Set a component value (mock ECS).
    pub fn set_component(&mut self, entity: u64, component: &str, value: Value) {
        self.components.insert((entity, component.to_string()), value);
    }

    /// Get a component value (mock ECS).
    pub fn get_component(&self, entity: u64, component: &str) -> Value {
        self.components.get(&(entity, component.to_string())).cloned().unwrap_or(Value::None)
    }

    /// Check if entity has a component.
    pub fn has_component(&self, entity: u64, component: &str) -> bool {
        self.components.contains_key(&(entity, component.to_string()))
    }

    /// Remove a component.
    pub fn remove_component(&mut self, entity: u64, component: &str) {
        self.components.remove(&(entity, component.to_string()));
    }
}

// ---------------------------------------------------------------------------
// Execution Result
// ---------------------------------------------------------------------------

/// Result of node execution.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionResult {
    /// Node executed successfully, continue to next flow port(s).
    Continue(Vec<PortId>),
    /// Node is waiting (e.g., Delay not complete).
    Waiting,
    /// Error during execution.
    Error(String),
}

impl ExecutionResult {
    /// Create a single-flow continue result.
    pub fn flow(port_id: PortId) -> Self {
        ExecutionResult::Continue(vec![port_id])
    }

    /// Create a no-flow continue result (pure data nodes).
    pub fn done() -> Self {
        ExecutionResult::Continue(vec![])
    }

    /// Create an error result.
    pub fn error(msg: impl Into<String>) -> Self {
        ExecutionResult::Error(msg.into())
    }
}

// ---------------------------------------------------------------------------
// Node Executor Trait
// ---------------------------------------------------------------------------

/// Trait for node execution behavior.
pub trait NodeExecutor: Send + Sync {
    /// Get the node type name (e.g., "Add", "Branch").
    fn node_type(&self) -> &'static str;

    /// Execute the node and return which flow ports to activate.
    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult;

    /// Check if this is an event node (can start execution chains).
    fn is_event(&self) -> bool {
        false
    }

    /// Check if this node should fire this frame (for event nodes).
    fn should_fire(&self, _node: &Node, _ctx: &ExecutionContext) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Event Nodes (6)
// ---------------------------------------------------------------------------

/// OnStart - fires once at the beginning.
pub struct OnStartExecutor;

impl NodeExecutor for OnStartExecutor {
    fn node_type(&self) -> &'static str { "OnStart" }

    fn execute(&self, node: &Node, _ctx: &mut ExecutionContext) -> ExecutionResult {
        if let Some(port) = node.outputs.first() {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }

    fn is_event(&self) -> bool { true }

    fn should_fire(&self, _node: &Node, ctx: &ExecutionContext) -> bool {
        ctx.frame == 0
    }
}

/// OnTick - fires every frame.
pub struct OnTickExecutor;

impl NodeExecutor for OnTickExecutor {
    fn node_type(&self) -> &'static str { "OnTick" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        // Set DeltaTime output
        if let Some(dt_port) = node.outputs.iter().find(|p| p.name == "DeltaTime") {
            ctx.set_output(dt_port.id, Value::Float(ctx.delta_time));
        }
        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else if let Some(port) = node.outputs.first() {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }

    fn is_event(&self) -> bool { true }

    fn should_fire(&self, _node: &Node, _ctx: &ExecutionContext) -> bool {
        true // Always fires
    }
}

/// OnKeyPressed - fires when a key is pressed.
pub struct OnKeyPressedExecutor;

impl NodeExecutor for OnKeyPressedExecutor {
    fn node_type(&self) -> &'static str { "OnKeyPressed" }

    fn execute(&self, node: &Node, _ctx: &mut ExecutionContext) -> ExecutionResult {
        if let Some(port) = node.outputs.first() {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }

    fn is_event(&self) -> bool { true }

    fn should_fire(&self, node: &Node, ctx: &ExecutionContext) -> bool {
        if let Some(prop) = node.get_property("Key") {
            if let PropertyValue::String(key) = &prop.value {
                return ctx.is_key_pressed(key);
            }
        }
        false
    }
}

/// OnKeyReleased - fires when a key is released.
pub struct OnKeyReleasedExecutor;

impl NodeExecutor for OnKeyReleasedExecutor {
    fn node_type(&self) -> &'static str { "OnKeyReleased" }

    fn execute(&self, node: &Node, _ctx: &mut ExecutionContext) -> ExecutionResult {
        if let Some(port) = node.outputs.first() {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }

    fn is_event(&self) -> bool { true }

    fn should_fire(&self, node: &Node, ctx: &ExecutionContext) -> bool {
        if let Some(prop) = node.get_property("Key") {
            if let PropertyValue::String(key) = &prop.value {
                return ctx.is_key_released(key);
            }
        }
        false
    }
}

/// OnCollision - fires when collision occurs.
pub struct OnCollisionExecutor;

impl NodeExecutor for OnCollisionExecutor {
    fn node_type(&self) -> &'static str { "OnCollision" }

    fn execute(&self, node: &Node, _ctx: &mut ExecutionContext) -> ExecutionResult {
        // In a real implementation, collision data would be passed in
        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }

    fn is_event(&self) -> bool { true }

    fn should_fire(&self, _node: &Node, ctx: &ExecutionContext) -> bool {
        ctx.event_fired("Collision")
    }
}

/// CustomEvent - fires when a named event occurs.
pub struct CustomEventExecutor;

impl NodeExecutor for CustomEventExecutor {
    fn node_type(&self) -> &'static str { "CustomEvent" }

    fn execute(&self, node: &Node, _ctx: &mut ExecutionContext) -> ExecutionResult {
        if let Some(port) = node.outputs.first() {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }

    fn is_event(&self) -> bool { true }

    fn should_fire(&self, node: &Node, ctx: &ExecutionContext) -> bool {
        if let Some(prop) = node.get_property("EventName") {
            if let PropertyValue::String(name) = &prop.value {
                return ctx.event_fired(name);
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Action Nodes (8)
// ---------------------------------------------------------------------------

/// SetPosition - sets entity position.
pub struct SetPositionExecutor;

impl NodeExecutor for SetPositionExecutor {
    fn node_type(&self) -> &'static str { "SetPosition" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let entity_port = node.inputs.iter().find(|p| p.name == "Entity");
        let pos_port = node.inputs.iter().find(|p| p.name == "Position");

        if let (Some(ep), Some(pp)) = (entity_port, pos_port) {
            let entity = ctx.get_input(ep.id).as_entity();
            let position = ctx.get_input(pp.id).as_vec3();
            ctx.set_component(entity, "Position", Value::Vec3(position));
        }

        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

/// SetRotation - sets entity rotation.
pub struct SetRotationExecutor;

impl NodeExecutor for SetRotationExecutor {
    fn node_type(&self) -> &'static str { "SetRotation" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let entity_port = node.inputs.iter().find(|p| p.name == "Entity");
        let rot_port = node.inputs.iter().find(|p| p.name == "Rotation");

        if let (Some(ep), Some(rp)) = (entity_port, rot_port) {
            let entity = ctx.get_input(ep.id).as_entity();
            let rotation = ctx.get_input(rp.id).as_vec3();
            ctx.set_component(entity, "Rotation", Value::Vec3(rotation));
        }

        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

/// SetScale - sets entity scale.
pub struct SetScaleExecutor;

impl NodeExecutor for SetScaleExecutor {
    fn node_type(&self) -> &'static str { "SetScale" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let entity_port = node.inputs.iter().find(|p| p.name == "Entity");
        let scale_port = node.inputs.iter().find(|p| p.name == "Scale");

        if let (Some(ep), Some(sp)) = (entity_port, scale_port) {
            let entity = ctx.get_input(ep.id).as_entity();
            let scale = ctx.get_input(sp.id).as_vec3();
            ctx.set_component(entity, "Scale", Value::Vec3(scale));
        }

        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

/// PlaySound - plays a sound.
pub struct PlaySoundExecutor;

impl NodeExecutor for PlaySoundExecutor {
    fn node_type(&self) -> &'static str { "PlaySound" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        if let Some(prop) = node.get_property("Sound") {
            if let PropertyValue::String(sound) = &prop.value {
                ctx.log(format!("Playing sound: {}", sound));
            }
        }

        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

/// SpawnEntity - creates a new entity.
pub struct SpawnEntityExecutor;

impl NodeExecutor for SpawnEntityExecutor {
    fn node_type(&self) -> &'static str { "SpawnEntity" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        // Generate a new entity ID (simplified)
        let new_entity = ctx.frame * 1000 + (ctx.random() * 1000.0) as u64;

        if let Some(port) = node.outputs.iter().find(|p| p.name == "Entity") {
            ctx.set_output(port.id, Value::Entity(new_entity));
        }

        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

/// DestroyEntity - destroys an entity.
pub struct DestroyEntityExecutor;

impl NodeExecutor for DestroyEntityExecutor {
    fn node_type(&self) -> &'static str { "DestroyEntity" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        if let Some(port) = node.inputs.iter().find(|p| p.name == "Entity") {
            let entity = ctx.get_input(port.id).as_entity();
            ctx.log(format!("Destroying entity: {}", entity));
            // In real implementation, would mark entity for destruction
        }

        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

/// SendMessage - sends a message/event.
pub struct SendMessageExecutor;

impl NodeExecutor for SendMessageExecutor {
    fn node_type(&self) -> &'static str { "SendMessage" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        if let Some(prop) = node.get_property("Message") {
            if let PropertyValue::String(msg) = &prop.value {
                ctx.fire_event(msg);
            }
        }

        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

/// SetVariable - sets a global variable.
pub struct SetVariableExecutor;

impl NodeExecutor for SetVariableExecutor {
    fn node_type(&self) -> &'static str { "SetVariable" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let name = node.get_property("Name")
            .and_then(|p| if let PropertyValue::String(s) = &p.value { Some(s.clone()) } else { None })
            .unwrap_or_else(|| "Unnamed".to_string());

        if let Some(port) = node.inputs.iter().find(|p| p.name == "Value") {
            let value = ctx.get_input(port.id);
            ctx.set_variable(&name, value);
        }

        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

// ---------------------------------------------------------------------------
// Condition Nodes (6)
// ---------------------------------------------------------------------------

/// Compare - compares two values.
pub struct CompareExecutor;

impl NodeExecutor for CompareExecutor {
    fn node_type(&self) -> &'static str { "Compare" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let a_port = node.inputs.iter().find(|p| p.name == "A");
        let b_port = node.inputs.iter().find(|p| p.name == "B");

        if let (Some(ap), Some(bp)) = (a_port, b_port) {
            let a = ctx.get_input(ap.id).as_float();
            let b = ctx.get_input(bp.id).as_float();

            if let Some(port) = node.outputs.iter().find(|p| p.name == "Less") {
                ctx.set_output(port.id, Value::Bool(a < b));
            }
            if let Some(port) = node.outputs.iter().find(|p| p.name == "Equal") {
                ctx.set_output(port.id, Value::Bool((a - b).abs() < f32::EPSILON));
            }
            if let Some(port) = node.outputs.iter().find(|p| p.name == "Greater") {
                ctx.set_output(port.id, Value::Bool(a > b));
            }
        }

        ExecutionResult::done()
    }
}

/// IsNull - checks if a value is null.
pub struct IsNullExecutor;

impl NodeExecutor for IsNullExecutor {
    fn node_type(&self) -> &'static str { "IsNull" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        if let Some(port) = node.inputs.iter().find(|p| p.name == "Value") {
            let value = ctx.get_input(port.id);
            let is_null = value.is_none();

            if let Some(out) = node.outputs.iter().find(|p| p.name == "Result") {
                ctx.set_output(out.id, Value::Bool(is_null));
            }
        }

        ExecutionResult::done()
    }
}

/// IsEqual - checks if two values are equal.
pub struct IsEqualExecutor;

impl NodeExecutor for IsEqualExecutor {
    fn node_type(&self) -> &'static str { "IsEqual" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let a_port = node.inputs.iter().find(|p| p.name == "A");
        let b_port = node.inputs.iter().find(|p| p.name == "B");

        if let (Some(ap), Some(bp)) = (a_port, b_port) {
            let a = ctx.get_input(ap.id);
            let b = ctx.get_input(bp.id);
            let equal = a == b;

            if let Some(out) = node.outputs.iter().find(|p| p.name == "Result") {
                ctx.set_output(out.id, Value::Bool(equal));
            }
        }

        ExecutionResult::done()
    }
}

/// IsGreater - checks if A > B.
pub struct IsGreaterExecutor;

impl NodeExecutor for IsGreaterExecutor {
    fn node_type(&self) -> &'static str { "IsGreater" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let a_port = node.inputs.iter().find(|p| p.name == "A");
        let b_port = node.inputs.iter().find(|p| p.name == "B");

        if let (Some(ap), Some(bp)) = (a_port, b_port) {
            let a = ctx.get_input(ap.id).as_float();
            let b = ctx.get_input(bp.id).as_float();

            if let Some(out) = node.outputs.iter().find(|p| p.name == "Result") {
                ctx.set_output(out.id, Value::Bool(a > b));
            }
        }

        ExecutionResult::done()
    }
}

/// IsLess - checks if A < B.
pub struct IsLessExecutor;

impl NodeExecutor for IsLessExecutor {
    fn node_type(&self) -> &'static str { "IsLess" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let a_port = node.inputs.iter().find(|p| p.name == "A");
        let b_port = node.inputs.iter().find(|p| p.name == "B");

        if let (Some(ap), Some(bp)) = (a_port, b_port) {
            let a = ctx.get_input(ap.id).as_float();
            let b = ctx.get_input(bp.id).as_float();

            if let Some(out) = node.outputs.iter().find(|p| p.name == "Result") {
                ctx.set_output(out.id, Value::Bool(a < b));
            }
        }

        ExecutionResult::done()
    }
}

/// InRange - checks if a value is within a range.
pub struct InRangeExecutor;

impl NodeExecutor for InRangeExecutor {
    fn node_type(&self) -> &'static str { "InRange" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let value_port = node.inputs.iter().find(|p| p.name == "Value");
        let min_port = node.inputs.iter().find(|p| p.name == "Min");
        let max_port = node.inputs.iter().find(|p| p.name == "Max");

        if let (Some(vp), Some(minp), Some(maxp)) = (value_port, min_port, max_port) {
            let value = ctx.get_input(vp.id).as_float();
            let min = ctx.get_input(minp.id).as_float();
            let max = ctx.get_input(maxp.id).as_float();

            if let Some(out) = node.outputs.iter().find(|p| p.name == "Result") {
                ctx.set_output(out.id, Value::Bool(value >= min && value <= max));
            }
        }

        ExecutionResult::done()
    }
}

// ---------------------------------------------------------------------------
// Math Nodes (10)
// ---------------------------------------------------------------------------

/// Add - adds two values.
pub struct AddExecutor;

impl NodeExecutor for AddExecutor {
    fn node_type(&self) -> &'static str { "Add" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let a_port = node.inputs.iter().find(|p| p.name == "A");
        let b_port = node.inputs.iter().find(|p| p.name == "B");

        if let (Some(ap), Some(bp)) = (a_port, b_port) {
            let a = ctx.get_input(ap.id).as_float();
            let b = ctx.get_input(bp.id).as_float();
            let result = a + b;

            if let Some(out) = node.outputs.iter().find(|p| p.name == "Result") {
                ctx.set_output(out.id, Value::Float(result));
            }
        }

        ExecutionResult::done()
    }
}

/// Subtract - subtracts two values.
pub struct SubtractExecutor;

impl NodeExecutor for SubtractExecutor {
    fn node_type(&self) -> &'static str { "Subtract" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let a_port = node.inputs.iter().find(|p| p.name == "A");
        let b_port = node.inputs.iter().find(|p| p.name == "B");

        if let (Some(ap), Some(bp)) = (a_port, b_port) {
            let a = ctx.get_input(ap.id).as_float();
            let b = ctx.get_input(bp.id).as_float();
            let result = a - b;

            if let Some(out) = node.outputs.iter().find(|p| p.name == "Result") {
                ctx.set_output(out.id, Value::Float(result));
            }
        }

        ExecutionResult::done()
    }
}

/// Multiply - multiplies two values.
pub struct MultiplyExecutor;

impl NodeExecutor for MultiplyExecutor {
    fn node_type(&self) -> &'static str { "Multiply" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let a_port = node.inputs.iter().find(|p| p.name == "A");
        let b_port = node.inputs.iter().find(|p| p.name == "B");

        if let (Some(ap), Some(bp)) = (a_port, b_port) {
            let a = ctx.get_input(ap.id).as_float();
            let b = ctx.get_input(bp.id).as_float();
            let result = a * b;

            if let Some(out) = node.outputs.iter().find(|p| p.name == "Result") {
                ctx.set_output(out.id, Value::Float(result));
            }
        }

        ExecutionResult::done()
    }
}

/// Divide - divides two values.
pub struct DivideExecutor;

impl NodeExecutor for DivideExecutor {
    fn node_type(&self) -> &'static str { "Divide" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let a_port = node.inputs.iter().find(|p| p.name == "A");
        let b_port = node.inputs.iter().find(|p| p.name == "B");

        if let (Some(ap), Some(bp)) = (a_port, b_port) {
            let a = ctx.get_input(ap.id).as_float();
            let b = ctx.get_input(bp.id).as_float();
            let result = if b.abs() < f32::EPSILON { 0.0 } else { a / b };

            if let Some(out) = node.outputs.iter().find(|p| p.name == "Result") {
                ctx.set_output(out.id, Value::Float(result));
            }
        }

        ExecutionResult::done()
    }
}

/// Abs - absolute value.
pub struct AbsExecutor;

impl NodeExecutor for AbsExecutor {
    fn node_type(&self) -> &'static str { "Abs" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        if let Some(port) = node.inputs.iter().find(|p| p.name == "Value") {
            let value = ctx.get_input(port.id).as_float();
            let result = value.abs();

            if let Some(out) = node.outputs.iter().find(|p| p.name == "Result") {
                ctx.set_output(out.id, Value::Float(result));
            }
        }

        ExecutionResult::done()
    }
}

/// Sin - sine function.
pub struct SinExecutor;

impl NodeExecutor for SinExecutor {
    fn node_type(&self) -> &'static str { "Sin" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        if let Some(port) = node.inputs.iter().find(|p| p.name == "Angle") {
            let angle = ctx.get_input(port.id).as_float();
            let result = angle.sin();

            if let Some(out) = node.outputs.iter().find(|p| p.name == "Result") {
                ctx.set_output(out.id, Value::Float(result));
            }
        }

        ExecutionResult::done()
    }
}

/// Cos - cosine function.
pub struct CosExecutor;

impl NodeExecutor for CosExecutor {
    fn node_type(&self) -> &'static str { "Cos" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        if let Some(port) = node.inputs.iter().find(|p| p.name == "Angle") {
            let angle = ctx.get_input(port.id).as_float();
            let result = angle.cos();

            if let Some(out) = node.outputs.iter().find(|p| p.name == "Result") {
                ctx.set_output(out.id, Value::Float(result));
            }
        }

        ExecutionResult::done()
    }
}

/// Lerp - linear interpolation.
pub struct LerpExecutor;

impl NodeExecutor for LerpExecutor {
    fn node_type(&self) -> &'static str { "Lerp" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let a_port = node.inputs.iter().find(|p| p.name == "A");
        let b_port = node.inputs.iter().find(|p| p.name == "B");
        let t_port = node.inputs.iter().find(|p| p.name == "T");

        if let (Some(ap), Some(bp), Some(tp)) = (a_port, b_port, t_port) {
            let a = ctx.get_input(ap.id).as_float();
            let b = ctx.get_input(bp.id).as_float();
            let t = ctx.get_input(tp.id).as_float().clamp(0.0, 1.0);
            let result = a + (b - a) * t;

            if let Some(out) = node.outputs.iter().find(|p| p.name == "Result") {
                ctx.set_output(out.id, Value::Float(result));
            }
        }

        ExecutionResult::done()
    }
}

/// Clamp - clamps value to range.
pub struct ClampExecutor;

impl NodeExecutor for ClampExecutor {
    fn node_type(&self) -> &'static str { "Clamp" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let value_port = node.inputs.iter().find(|p| p.name == "Value");
        let min_port = node.inputs.iter().find(|p| p.name == "Min");
        let max_port = node.inputs.iter().find(|p| p.name == "Max");

        if let (Some(vp), Some(minp), Some(maxp)) = (value_port, min_port, max_port) {
            let value = ctx.get_input(vp.id).as_float();
            let min = ctx.get_input(minp.id).as_float();
            let max = ctx.get_input(maxp.id).as_float();
            let result = value.clamp(min, max);

            if let Some(out) = node.outputs.iter().find(|p| p.name == "Result") {
                ctx.set_output(out.id, Value::Float(result));
            }
        }

        ExecutionResult::done()
    }
}

/// Random - generates random value [0, 1).
pub struct RandomExecutor;

impl NodeExecutor for RandomExecutor {
    fn node_type(&self) -> &'static str { "Random" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let result = ctx.random();

        if let Some(out) = node.outputs.iter().find(|p| p.name == "Value") {
            ctx.set_output(out.id, Value::Float(result));
        }

        ExecutionResult::done()
    }
}

// ---------------------------------------------------------------------------
// Flow Control Nodes (6)
// ---------------------------------------------------------------------------

/// Sequence - executes outputs in order.
pub struct SequenceExecutor;

impl NodeExecutor for SequenceExecutor {
    fn node_type(&self) -> &'static str { "Sequence" }

    fn execute(&self, node: &Node, _ctx: &mut ExecutionContext) -> ExecutionResult {
        // Return all flow output ports
        let flow_ports: Vec<PortId> = node.outputs.iter()
            .filter(|p| p.port_type == PortType::Flow)
            .map(|p| p.id)
            .collect();

        ExecutionResult::Continue(flow_ports)
    }
}

/// Branch - conditional branch.
pub struct BranchExecutor;

impl NodeExecutor for BranchExecutor {
    fn node_type(&self) -> &'static str { "Branch" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let condition = node.inputs.iter()
            .find(|p| p.name == "Condition")
            .map(|p| ctx.get_input(p.id).as_bool())
            .unwrap_or(false);

        let target_port = if condition {
            node.outputs.iter().find(|p| p.name == "True")
        } else {
            node.outputs.iter().find(|p| p.name == "False")
        };

        if let Some(port) = target_port {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

/// ForLoop - iterates a fixed number of times.
pub struct ForLoopExecutor;

impl NodeExecutor for ForLoopExecutor {
    fn node_type(&self) -> &'static str { "ForLoop" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let count = node.inputs.iter()
            .find(|p| p.name == "Count")
            .map(|p| ctx.get_input(p.id).as_int())
            .unwrap_or(0);

        let current = ctx.get_loop_counter(node.id);

        if current < count {
            // Set Index output
            if let Some(port) = node.outputs.iter().find(|p| p.name == "Index") {
                ctx.set_output(port.id, Value::Int(current));
            }

            ctx.set_loop_counter(node.id, current + 1);

            // Execute loop body
            if let Some(port) = node.outputs.iter().find(|p| p.name == "Loop") {
                return ExecutionResult::flow(port.id);
            }
        } else {
            // Loop complete
            ctx.set_loop_counter(node.id, 0);
            if let Some(port) = node.outputs.iter().find(|p| p.name == "Completed") {
                return ExecutionResult::flow(port.id);
            }
        }

        ExecutionResult::done()
    }
}

/// WhileLoop - iterates while condition is true.
pub struct WhileLoopExecutor;

impl NodeExecutor for WhileLoopExecutor {
    fn node_type(&self) -> &'static str { "WhileLoop" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let condition = node.inputs.iter()
            .find(|p| p.name == "Condition")
            .map(|p| ctx.get_input(p.id).as_bool())
            .unwrap_or(false);

        if condition {
            if let Some(port) = node.outputs.iter().find(|p| p.name == "Loop") {
                return ExecutionResult::flow(port.id);
            }
        } else {
            if let Some(port) = node.outputs.iter().find(|p| p.name == "Completed") {
                return ExecutionResult::flow(port.id);
            }
        }

        ExecutionResult::done()
    }
}

/// DoOnce - executes only once until reset.
pub struct DoOnceExecutor;

impl NodeExecutor for DoOnceExecutor {
    fn node_type(&self) -> &'static str { "DoOnce" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let reset = node.get_property("Reset")
            .and_then(|p| if let PropertyValue::Bool(b) = p.value { Some(b) } else { None })
            .unwrap_or(false);

        if reset {
            ctx.set_do_once(node.id, false);
        }

        if !ctx.get_do_once(node.id) {
            ctx.set_do_once(node.id, true);
            if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
                return ExecutionResult::flow(port.id);
            }
        }

        ExecutionResult::done()
    }
}

/// Delay - waits for a duration.
pub struct DelayExecutor;

impl NodeExecutor for DelayExecutor {
    fn node_type(&self) -> &'static str { "Delay" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let duration = node.inputs.iter()
            .find(|p| p.name == "Duration")
            .map(|p| ctx.get_input(p.id).as_float())
            .unwrap_or(1.0);

        let current_time = ctx.get_delay_timer(node.id);
        let new_time = current_time + ctx.delta_time;

        if new_time >= duration {
            ctx.set_delay_timer(node.id, 0.0);
            if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
                return ExecutionResult::flow(port.id);
            }
            ExecutionResult::done()
        } else {
            ctx.set_delay_timer(node.id, new_time);
            ExecutionResult::Waiting
        }
    }
}

// ---------------------------------------------------------------------------
// ECS Nodes (6)
// ---------------------------------------------------------------------------

/// GetComponent - gets a component value.
pub struct GetComponentExecutor;

impl NodeExecutor for GetComponentExecutor {
    fn node_type(&self) -> &'static str { "GetComponent" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let entity = node.inputs.iter()
            .find(|p| p.name == "Entity")
            .map(|p| ctx.get_input(p.id).as_entity())
            .unwrap_or(0);

        let component_name = node.get_property("Component")
            .and_then(|p| if let PropertyValue::String(s) = &p.value { Some(s.clone()) } else { None })
            .unwrap_or_else(|| "Position".to_string());

        let value = ctx.get_component(entity, &component_name);

        if let Some(out) = node.outputs.iter().find(|p| p.name == "Value") {
            ctx.set_output(out.id, value);
        }

        ExecutionResult::done()
    }
}

/// SetComponent - sets a component value.
pub struct SetComponentExecutor;

impl NodeExecutor for SetComponentExecutor {
    fn node_type(&self) -> &'static str { "SetComponent" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let entity = node.inputs.iter()
            .find(|p| p.name == "Entity")
            .map(|p| ctx.get_input(p.id).as_entity())
            .unwrap_or(0);

        let component_name = node.get_property("Component")
            .and_then(|p| if let PropertyValue::String(s) = &p.value { Some(s.clone()) } else { None })
            .unwrap_or_else(|| "Position".to_string());

        if let Some(port) = node.inputs.iter().find(|p| p.name == "Value") {
            let value = ctx.get_input(port.id);
            ctx.set_component(entity, &component_name, value);
        }

        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

/// HasComponent - checks if entity has a component.
pub struct HasComponentExecutor;

impl NodeExecutor for HasComponentExecutor {
    fn node_type(&self) -> &'static str { "HasComponent" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let entity = node.inputs.iter()
            .find(|p| p.name == "Entity")
            .map(|p| ctx.get_input(p.id).as_entity())
            .unwrap_or(0);

        let component_name = node.get_property("Component")
            .and_then(|p| if let PropertyValue::String(s) = &p.value { Some(s.clone()) } else { None })
            .unwrap_or_else(|| "Position".to_string());

        let has = ctx.has_component(entity, &component_name);

        if let Some(out) = node.outputs.iter().find(|p| p.name == "Result") {
            ctx.set_output(out.id, Value::Bool(has));
        }

        ExecutionResult::done()
    }
}

/// AddComponent - adds a component to an entity.
pub struct AddComponentExecutor;

impl NodeExecutor for AddComponentExecutor {
    fn node_type(&self) -> &'static str { "AddComponent" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let entity = node.inputs.iter()
            .find(|p| p.name == "Entity")
            .map(|p| ctx.get_input(p.id).as_entity())
            .unwrap_or(0);

        let component_name = node.get_property("Component")
            .and_then(|p| if let PropertyValue::String(s) = &p.value { Some(s.clone()) } else { None })
            .unwrap_or_else(|| "Position".to_string());

        // Initialize with default value
        ctx.set_component(entity, &component_name, Value::None);

        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

/// RemoveComponent - removes a component from an entity.
pub struct RemoveComponentExecutor;

impl NodeExecutor for RemoveComponentExecutor {
    fn node_type(&self) -> &'static str { "RemoveComponent" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let entity = node.inputs.iter()
            .find(|p| p.name == "Entity")
            .map(|p| ctx.get_input(p.id).as_entity())
            .unwrap_or(0);

        let component_name = node.get_property("Component")
            .and_then(|p| if let PropertyValue::String(s) = &p.value { Some(s.clone()) } else { None })
            .unwrap_or_else(|| "Position".to_string());

        ctx.remove_component(entity, &component_name);

        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

/// QueryEntities - queries entities with components.
pub struct QueryEntitiesExecutor;

impl NodeExecutor for QueryEntitiesExecutor {
    fn node_type(&self) -> &'static str { "QueryEntities" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        // In a real implementation, this would query the ECS
        // For now, return an empty list
        if let Some(out) = node.outputs.iter().find(|p| p.name == "Entities") {
            ctx.set_output(out.id, Value::List(vec![]));
        }

        ExecutionResult::done()
    }
}

// ---------------------------------------------------------------------------
// Debug Nodes (4)
// ---------------------------------------------------------------------------

/// Print - prints a value to debug output.
pub struct PrintExecutor;

impl NodeExecutor for PrintExecutor {
    fn node_type(&self) -> &'static str { "Print" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        if let Some(port) = node.inputs.iter().find(|p| p.name == "Value") {
            let value = ctx.get_input(port.id);
            ctx.log(format!("PRINT: {}", value.as_string()));
        }

        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

/// Log - logs a message with level.
pub struct LogExecutor;

impl NodeExecutor for LogExecutor {
    fn node_type(&self) -> &'static str { "Log" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let level = node.get_property("Level")
            .and_then(|p| if let PropertyValue::Enum(i) = p.value { Some(i) } else { None })
            .unwrap_or(0);

        let level_str = match level {
            0 => "INFO",
            1 => "WARN",
            2 => "ERROR",
            _ => "DEBUG",
        };

        if let Some(port) = node.inputs.iter().find(|p| p.name == "Message") {
            let message = ctx.get_input(port.id).as_string();
            ctx.log(format!("[{}] {}", level_str, message));
        }

        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

/// Assert - asserts a condition is true.
pub struct AssertExecutor;

impl NodeExecutor for AssertExecutor {
    fn node_type(&self) -> &'static str { "Assert" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        let condition = node.inputs.iter()
            .find(|p| p.name == "Condition")
            .map(|p| ctx.get_input(p.id).as_bool())
            .unwrap_or(true);

        if !condition {
            let message = node.get_property("Message")
                .and_then(|p| if let PropertyValue::String(s) = &p.value { Some(s.clone()) } else { None })
                .unwrap_or_else(|| "Assertion failed".to_string());

            ctx.log(format!("ASSERT FAILED: {}", message));
            return ExecutionResult::error(message);
        }

        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

/// Breakpoint - pauses execution for debugging.
pub struct BreakpointExecutor;

impl NodeExecutor for BreakpointExecutor {
    fn node_type(&self) -> &'static str { "Breakpoint" }

    fn execute(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        ctx.log(format!("BREAKPOINT hit at node {}", node.id));

        // In a real debugger, this would pause execution
        if let Some(port) = node.outputs.iter().find(|p| p.name == "Out") {
            ExecutionResult::flow(port.id)
        } else {
            ExecutionResult::done()
        }
    }
}

// ---------------------------------------------------------------------------
// Node Registry
// ---------------------------------------------------------------------------

/// Registry of all node executors.
pub struct NodeRegistry {
    executors: HashMap<String, Box<dyn NodeExecutor>>,
}

impl NodeRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            executors: HashMap::new(),
        }
    }

    /// Create a registry with all built-in node types.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();

        // Events (6)
        registry.register(Box::new(OnStartExecutor));
        registry.register(Box::new(OnTickExecutor));
        registry.register(Box::new(OnKeyPressedExecutor));
        registry.register(Box::new(OnKeyReleasedExecutor));
        registry.register(Box::new(OnCollisionExecutor));
        registry.register(Box::new(CustomEventExecutor));

        // Actions (8)
        registry.register(Box::new(SetPositionExecutor));
        registry.register(Box::new(SetRotationExecutor));
        registry.register(Box::new(SetScaleExecutor));
        registry.register(Box::new(PlaySoundExecutor));
        registry.register(Box::new(SpawnEntityExecutor));
        registry.register(Box::new(DestroyEntityExecutor));
        registry.register(Box::new(SendMessageExecutor));
        registry.register(Box::new(SetVariableExecutor));

        // Conditions (6)
        registry.register(Box::new(CompareExecutor));
        registry.register(Box::new(IsNullExecutor));
        registry.register(Box::new(IsEqualExecutor));
        registry.register(Box::new(IsGreaterExecutor));
        registry.register(Box::new(IsLessExecutor));
        registry.register(Box::new(InRangeExecutor));

        // Math (10)
        registry.register(Box::new(AddExecutor));
        registry.register(Box::new(SubtractExecutor));
        registry.register(Box::new(MultiplyExecutor));
        registry.register(Box::new(DivideExecutor));
        registry.register(Box::new(AbsExecutor));
        registry.register(Box::new(SinExecutor));
        registry.register(Box::new(CosExecutor));
        registry.register(Box::new(LerpExecutor));
        registry.register(Box::new(ClampExecutor));
        registry.register(Box::new(RandomExecutor));

        // Flow (6)
        registry.register(Box::new(SequenceExecutor));
        registry.register(Box::new(BranchExecutor));
        registry.register(Box::new(ForLoopExecutor));
        registry.register(Box::new(WhileLoopExecutor));
        registry.register(Box::new(DoOnceExecutor));
        registry.register(Box::new(DelayExecutor));

        // ECS (6)
        registry.register(Box::new(GetComponentExecutor));
        registry.register(Box::new(SetComponentExecutor));
        registry.register(Box::new(HasComponentExecutor));
        registry.register(Box::new(AddComponentExecutor));
        registry.register(Box::new(RemoveComponentExecutor));
        registry.register(Box::new(QueryEntitiesExecutor));

        // Debug (4)
        registry.register(Box::new(PrintExecutor));
        registry.register(Box::new(LogExecutor));
        registry.register(Box::new(AssertExecutor));
        registry.register(Box::new(BreakpointExecutor));

        registry
    }

    /// Register a node executor.
    pub fn register(&mut self, executor: Box<dyn NodeExecutor>) {
        self.executors.insert(executor.node_type().to_string(), executor);
    }

    /// Get an executor by node type name.
    pub fn get(&self, node_type: &str) -> Option<&dyn NodeExecutor> {
        self.executors.get(node_type).map(|e| e.as_ref())
    }

    /// Execute a node by its type.
    pub fn execute_node(&self, node: &Node, ctx: &mut ExecutionContext) -> ExecutionResult {
        if let Some(executor) = self.get(&node.name) {
            executor.execute(node, ctx)
        } else {
            ExecutionResult::error(format!("Unknown node type: {}", node.name))
        }
    }

    /// Get all registered node type names.
    pub fn node_types(&self) -> Vec<&str> {
        self.executors.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of registered node types.
    pub fn len(&self) -> usize {
        self.executors.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.executors.is_empty()
    }

    /// Get all event node executors.
    pub fn event_executors(&self) -> Vec<&dyn NodeExecutor> {
        self.executors.values()
            .map(|e| e.as_ref())
            .filter(|e| e.is_event())
            .collect()
    }
}

impl Default for NodeRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowforge::{NodeTemplate, PortTemplate, PropertyTemplate, NodeGraph, PortDirection, Port};

    // Helper to create a test node with ports
    fn make_test_node(name: &str, inputs: &[(&str, PortType)], outputs: &[(&str, PortType)]) -> Node {
        let mut port_id = 1;
        Node {
            id: 1,
            name: name.to_string(),
            category: "Test".to_string(),
            position: (0.0, 0.0),
            inputs: inputs.iter().map(|(n, t)| {
                let id = port_id;
                port_id += 1;
                Port::new(id, *n, *t, PortDirection::Input)
            }).collect(),
            outputs: outputs.iter().map(|(n, t)| {
                let id = port_id;
                port_id += 1;
                Port::new(id, *n, *t, PortDirection::Output)
            }).collect(),
            properties: vec![],
        }
    }

    // -------------------------------------------------------------------------
    // Value Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_value_as_bool() {
        assert!(Value::Bool(true).as_bool());
        assert!(!Value::Bool(false).as_bool());
        assert!(Value::Int(1).as_bool());
        assert!(!Value::Int(0).as_bool());
        assert!(Value::Float(1.0).as_bool());
        assert!(!Value::Float(0.0).as_bool());
        assert!(Value::String("hello".to_string()).as_bool());
        assert!(!Value::String("".to_string()).as_bool());
        assert!(!Value::None.as_bool());
    }

    #[test]
    fn test_value_as_int() {
        assert_eq!(Value::Int(42).as_int(), 42);
        assert_eq!(Value::Float(3.7).as_int(), 3);
        assert_eq!(Value::Bool(true).as_int(), 1);
        assert_eq!(Value::Bool(false).as_int(), 0);
        assert_eq!(Value::String("123".to_string()).as_int(), 123);
        assert_eq!(Value::None.as_int(), 0);
    }

    #[test]
    fn test_value_as_float() {
        assert_eq!(Value::Float(3.14).as_float(), 3.14);
        assert_eq!(Value::Int(42).as_float(), 42.0);
        assert_eq!(Value::Bool(true).as_float(), 1.0);
        assert_eq!(Value::String("2.5".to_string()).as_float(), 2.5);
    }

    #[test]
    fn test_value_as_string() {
        assert_eq!(Value::String("hello".to_string()).as_string(), "hello");
        assert_eq!(Value::Int(42).as_string(), "42");
        assert_eq!(Value::Bool(true).as_string(), "true");
    }

    #[test]
    fn test_value_as_vec3() {
        assert_eq!(Value::Vec3([1.0, 2.0, 3.0]).as_vec3(), [1.0, 2.0, 3.0]);
        assert_eq!(Value::None.as_vec3(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_value_as_entity() {
        assert_eq!(Value::Entity(123).as_entity(), 123);
        assert_eq!(Value::Int(456).as_entity(), 456);
        assert_eq!(Value::None.as_entity(), 0);
    }

    #[test]
    fn test_value_is_none() {
        assert!(Value::None.is_none());
        assert!(!Value::Bool(false).is_none());
        assert!(!Value::Int(0).is_none());
    }

    #[test]
    fn test_value_port_type() {
        assert_eq!(Value::Bool(true).port_type(), PortType::Bool);
        assert_eq!(Value::Int(1).port_type(), PortType::Int);
        assert_eq!(Value::Float(1.0).port_type(), PortType::Float);
        assert_eq!(Value::String("".to_string()).port_type(), PortType::String);
        assert_eq!(Value::Vec3([0.0; 3]).port_type(), PortType::Vec3);
        assert_eq!(Value::Entity(0).port_type(), PortType::Entity);
    }

    #[test]
    fn test_value_from_property() {
        assert_eq!(Value::from(PropertyValue::Bool(true)), Value::Bool(true));
        assert_eq!(Value::from(PropertyValue::Int(42)), Value::Int(42));
        assert_eq!(Value::from(PropertyValue::Float(3.14)), Value::Float(3.14));
    }

    // -------------------------------------------------------------------------
    // ExecutionContext Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_context_io() {
        let mut ctx = ExecutionContext::new();
        ctx.set_input(1, Value::Float(5.0));
        assert_eq!(ctx.get_input(1), Value::Float(5.0));
        assert_eq!(ctx.get_input(999), Value::None);

        ctx.set_output(2, Value::Bool(true));
        assert_eq!(ctx.get_output(2), Value::Bool(true));

        ctx.clear_io();
        assert_eq!(ctx.get_input(1), Value::None);
        assert_eq!(ctx.get_output(2), Value::None);
    }

    #[test]
    fn test_context_variables() {
        let mut ctx = ExecutionContext::new();
        ctx.set_variable("score", Value::Int(100));
        assert_eq!(ctx.get_variable("score"), Value::Int(100));
        assert_eq!(ctx.get_variable("missing"), Value::None);
    }

    #[test]
    fn test_context_events() {
        let mut ctx = ExecutionContext::new();
        assert!(!ctx.event_fired("Jump"));

        ctx.fire_event("Jump");
        assert!(ctx.event_fired("Jump"));

        ctx.clear_events();
        assert!(!ctx.event_fired("Jump"));
    }

    #[test]
    fn test_context_keys() {
        let mut ctx = ExecutionContext::new();
        assert!(!ctx.is_key_pressed("Space"));

        ctx.set_key_pressed("Space", true);
        assert!(ctx.is_key_pressed("Space"));

        ctx.set_key_pressed("Space", false);
        assert!(!ctx.is_key_pressed("Space"));
        assert!(ctx.is_key_released("Space"));

        ctx.clear_events();
        assert!(!ctx.is_key_released("Space"));
    }

    #[test]
    fn test_context_do_once() {
        let mut ctx = ExecutionContext::new();
        assert!(!ctx.get_do_once(1));

        ctx.set_do_once(1, true);
        assert!(ctx.get_do_once(1));

        ctx.set_do_once(1, false);
        assert!(!ctx.get_do_once(1));
    }

    #[test]
    fn test_context_delay_timer() {
        let mut ctx = ExecutionContext::new();
        assert_eq!(ctx.get_delay_timer(1), 0.0);

        ctx.set_delay_timer(1, 0.5);
        assert_eq!(ctx.get_delay_timer(1), 0.5);
    }

    #[test]
    fn test_context_loop_counter() {
        let mut ctx = ExecutionContext::new();
        assert_eq!(ctx.get_loop_counter(1), 0);

        ctx.set_loop_counter(1, 5);
        assert_eq!(ctx.get_loop_counter(1), 5);
    }

    #[test]
    fn test_context_random() {
        let mut ctx = ExecutionContext::new();
        let r1 = ctx.random();
        let r2 = ctx.random();
        assert!(r1 >= 0.0 && r1 < 1.0);
        assert!(r2 >= 0.0 && r2 < 1.0);
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_context_log() {
        let mut ctx = ExecutionContext::new();
        ctx.log("Test message".to_string());
        assert_eq!(ctx.debug_log.len(), 1);
        assert_eq!(ctx.debug_log[0], "Test message");
    }

    #[test]
    fn test_context_components() {
        let mut ctx = ExecutionContext::new();
        assert!(!ctx.has_component(1, "Position"));

        ctx.set_component(1, "Position", Value::Vec3([1.0, 2.0, 3.0]));
        assert!(ctx.has_component(1, "Position"));
        assert_eq!(ctx.get_component(1, "Position"), Value::Vec3([1.0, 2.0, 3.0]));

        ctx.remove_component(1, "Position");
        assert!(!ctx.has_component(1, "Position"));
    }

    // -------------------------------------------------------------------------
    // ExecutionResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_execution_result_flow() {
        let result = ExecutionResult::flow(42);
        assert!(matches!(result, ExecutionResult::Continue(ports) if ports == vec![42]));
    }

    #[test]
    fn test_execution_result_done() {
        let result = ExecutionResult::done();
        assert!(matches!(result, ExecutionResult::Continue(ports) if ports.is_empty()));
    }

    #[test]
    fn test_execution_result_error() {
        let result = ExecutionResult::error("test error");
        assert!(matches!(result, ExecutionResult::Error(msg) if msg == "test error"));
    }

    // -------------------------------------------------------------------------
    // Event Node Tests (6)
    // -------------------------------------------------------------------------

    #[test]
    fn test_on_start_executor() {
        let executor = OnStartExecutor;
        assert_eq!(executor.node_type(), "OnStart");
        assert!(executor.is_event());

        let mut ctx = ExecutionContext::new();
        ctx.frame = 0;
        assert!(executor.should_fire(&make_test_node("OnStart", &[], &[("Out", PortType::Flow)]), &ctx));

        ctx.frame = 1;
        assert!(!executor.should_fire(&make_test_node("OnStart", &[], &[("Out", PortType::Flow)]), &ctx));
    }

    #[test]
    fn test_on_tick_executor() {
        let executor = OnTickExecutor;
        assert_eq!(executor.node_type(), "OnTick");
        assert!(executor.is_event());

        let mut ctx = ExecutionContext::new();
        ctx.delta_time = 0.016;

        let node = make_test_node("OnTick", &[], &[("Out", PortType::Flow), ("DeltaTime", PortType::Float)]);
        let result = executor.execute(&node, &mut ctx);

        assert!(matches!(result, ExecutionResult::Continue(_)));
        // Check DeltaTime output was set
        let dt_port = node.outputs.iter().find(|p| p.name == "DeltaTime").unwrap();
        assert_eq!(ctx.get_output(dt_port.id).as_float(), 0.016);
    }

    #[test]
    fn test_on_key_pressed_executor() {
        let executor = OnKeyPressedExecutor;
        assert_eq!(executor.node_type(), "OnKeyPressed");

        let mut node = make_test_node("OnKeyPressed", &[], &[("Out", PortType::Flow)]);
        node.properties.push(crate::flowforge::NodeProperty::string("Key", "Space"));

        let mut ctx = ExecutionContext::new();
        assert!(!executor.should_fire(&node, &ctx));

        ctx.set_key_pressed("Space", true);
        assert!(executor.should_fire(&node, &ctx));
    }

    #[test]
    fn test_on_key_released_executor() {
        let executor = OnKeyReleasedExecutor;
        assert_eq!(executor.node_type(), "OnKeyReleased");

        let mut node = make_test_node("OnKeyReleased", &[], &[("Out", PortType::Flow)]);
        node.properties.push(crate::flowforge::NodeProperty::string("Key", "Space"));

        let mut ctx = ExecutionContext::new();
        ctx.set_key_pressed("Space", true);
        ctx.set_key_pressed("Space", false);
        assert!(executor.should_fire(&node, &ctx));
    }

    #[test]
    fn test_on_collision_executor() {
        let executor = OnCollisionExecutor;
        assert_eq!(executor.node_type(), "OnCollision");

        let mut ctx = ExecutionContext::new();
        assert!(!executor.should_fire(&make_test_node("OnCollision", &[], &[]), &ctx));

        ctx.fire_event("Collision");
        assert!(executor.should_fire(&make_test_node("OnCollision", &[], &[]), &ctx));
    }

    #[test]
    fn test_custom_event_executor() {
        let executor = CustomEventExecutor;
        assert_eq!(executor.node_type(), "CustomEvent");

        let mut node = make_test_node("CustomEvent", &[], &[("Out", PortType::Flow)]);
        node.properties.push(crate::flowforge::NodeProperty::string("EventName", "MyEvent"));

        let mut ctx = ExecutionContext::new();
        assert!(!executor.should_fire(&node, &ctx));

        ctx.fire_event("MyEvent");
        assert!(executor.should_fire(&node, &ctx));
    }

    // -------------------------------------------------------------------------
    // Action Node Tests (8)
    // -------------------------------------------------------------------------

    #[test]
    fn test_set_position_executor() {
        let executor = SetPositionExecutor;
        assert_eq!(executor.node_type(), "SetPosition");

        let node = make_test_node("SetPosition",
            &[("In", PortType::Flow), ("Entity", PortType::Entity), ("Position", PortType::Vec3)],
            &[("Out", PortType::Flow)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::Entity(1));
        ctx.set_input(node.inputs[2].id, Value::Vec3([10.0, 20.0, 30.0]));

        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_component(1, "Position"), Value::Vec3([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_set_rotation_executor() {
        let executor = SetRotationExecutor;
        assert_eq!(executor.node_type(), "SetRotation");

        let node = make_test_node("SetRotation",
            &[("In", PortType::Flow), ("Entity", PortType::Entity), ("Rotation", PortType::Vec3)],
            &[("Out", PortType::Flow)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::Entity(1));
        ctx.set_input(node.inputs[2].id, Value::Vec3([0.0, 90.0, 0.0]));

        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_component(1, "Rotation"), Value::Vec3([0.0, 90.0, 0.0]));
    }

    #[test]
    fn test_set_scale_executor() {
        let executor = SetScaleExecutor;
        assert_eq!(executor.node_type(), "SetScale");

        let node = make_test_node("SetScale",
            &[("In", PortType::Flow), ("Entity", PortType::Entity), ("Scale", PortType::Vec3)],
            &[("Out", PortType::Flow)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::Entity(1));
        ctx.set_input(node.inputs[2].id, Value::Vec3([2.0, 2.0, 2.0]));

        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_component(1, "Scale"), Value::Vec3([2.0, 2.0, 2.0]));
    }

    #[test]
    fn test_play_sound_executor() {
        let executor = PlaySoundExecutor;
        assert_eq!(executor.node_type(), "PlaySound");

        let mut node = make_test_node("PlaySound", &[("In", PortType::Flow)], &[("Out", PortType::Flow)]);
        node.properties.push(crate::flowforge::NodeProperty::string("Sound", "explosion.wav"));

        let mut ctx = ExecutionContext::new();
        executor.execute(&node, &mut ctx);
        assert!(ctx.debug_log.iter().any(|s| s.contains("explosion.wav")));
    }

    #[test]
    fn test_spawn_entity_executor() {
        let executor = SpawnEntityExecutor;
        assert_eq!(executor.node_type(), "SpawnEntity");

        let node = make_test_node("SpawnEntity",
            &[("In", PortType::Flow)],
            &[("Out", PortType::Flow), ("Entity", PortType::Entity)]);

        let mut ctx = ExecutionContext::new();
        executor.execute(&node, &mut ctx);

        let entity_port = node.outputs.iter().find(|p| p.name == "Entity").unwrap();
        let entity = ctx.get_output(entity_port.id).as_entity();
        assert!(entity > 0);
    }

    #[test]
    fn test_destroy_entity_executor() {
        let executor = DestroyEntityExecutor;
        assert_eq!(executor.node_type(), "DestroyEntity");

        let node = make_test_node("DestroyEntity",
            &[("In", PortType::Flow), ("Entity", PortType::Entity)],
            &[("Out", PortType::Flow)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::Entity(42));

        executor.execute(&node, &mut ctx);
        assert!(ctx.debug_log.iter().any(|s| s.contains("42")));
    }

    #[test]
    fn test_send_message_executor() {
        let executor = SendMessageExecutor;
        assert_eq!(executor.node_type(), "SendMessage");

        let mut node = make_test_node("SendMessage", &[("In", PortType::Flow)], &[("Out", PortType::Flow)]);
        node.properties.push(crate::flowforge::NodeProperty::string("Message", "PlayerDied"));

        let mut ctx = ExecutionContext::new();
        executor.execute(&node, &mut ctx);
        assert!(ctx.event_fired("PlayerDied"));
    }

    #[test]
    fn test_set_variable_executor() {
        let executor = SetVariableExecutor;
        assert_eq!(executor.node_type(), "SetVariable");

        let mut node = make_test_node("SetVariable",
            &[("In", PortType::Flow), ("Value", PortType::Any)],
            &[("Out", PortType::Flow)]);
        node.properties.push(crate::flowforge::NodeProperty::string("Name", "Score"));

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::Int(100));

        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_variable("Score"), Value::Int(100));
    }

    // -------------------------------------------------------------------------
    // Condition Node Tests (6)
    // -------------------------------------------------------------------------

    #[test]
    fn test_compare_executor() {
        let executor = CompareExecutor;
        assert_eq!(executor.node_type(), "Compare");

        let node = make_test_node("Compare",
            &[("A", PortType::Float), ("B", PortType::Float)],
            &[("Less", PortType::Bool), ("Equal", PortType::Bool), ("Greater", PortType::Bool)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Float(5.0));
        ctx.set_input(node.inputs[1].id, Value::Float(10.0));

        executor.execute(&node, &mut ctx);

        let less = node.outputs.iter().find(|p| p.name == "Less").unwrap();
        let equal = node.outputs.iter().find(|p| p.name == "Equal").unwrap();
        let greater = node.outputs.iter().find(|p| p.name == "Greater").unwrap();

        assert!(ctx.get_output(less.id).as_bool());
        assert!(!ctx.get_output(equal.id).as_bool());
        assert!(!ctx.get_output(greater.id).as_bool());
    }

    #[test]
    fn test_is_null_executor() {
        let executor = IsNullExecutor;
        assert_eq!(executor.node_type(), "IsNull");

        let node = make_test_node("IsNull",
            &[("Value", PortType::Any)],
            &[("Result", PortType::Bool)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::None);
        executor.execute(&node, &mut ctx);
        assert!(ctx.get_output(node.outputs[0].id).as_bool());

        ctx.set_input(node.inputs[0].id, Value::Int(0));
        executor.execute(&node, &mut ctx);
        assert!(!ctx.get_output(node.outputs[0].id).as_bool());
    }

    #[test]
    fn test_is_equal_executor() {
        let executor = IsEqualExecutor;
        assert_eq!(executor.node_type(), "IsEqual");

        let node = make_test_node("IsEqual",
            &[("A", PortType::Any), ("B", PortType::Any)],
            &[("Result", PortType::Bool)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Int(5));
        ctx.set_input(node.inputs[1].id, Value::Int(5));
        executor.execute(&node, &mut ctx);
        assert!(ctx.get_output(node.outputs[0].id).as_bool());

        ctx.set_input(node.inputs[1].id, Value::Int(10));
        executor.execute(&node, &mut ctx);
        assert!(!ctx.get_output(node.outputs[0].id).as_bool());
    }

    #[test]
    fn test_is_greater_executor() {
        let executor = IsGreaterExecutor;
        assert_eq!(executor.node_type(), "IsGreater");

        let node = make_test_node("IsGreater",
            &[("A", PortType::Float), ("B", PortType::Float)],
            &[("Result", PortType::Bool)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Float(10.0));
        ctx.set_input(node.inputs[1].id, Value::Float(5.0));
        executor.execute(&node, &mut ctx);
        assert!(ctx.get_output(node.outputs[0].id).as_bool());
    }

    #[test]
    fn test_is_less_executor() {
        let executor = IsLessExecutor;
        assert_eq!(executor.node_type(), "IsLess");

        let node = make_test_node("IsLess",
            &[("A", PortType::Float), ("B", PortType::Float)],
            &[("Result", PortType::Bool)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Float(3.0));
        ctx.set_input(node.inputs[1].id, Value::Float(5.0));
        executor.execute(&node, &mut ctx);
        assert!(ctx.get_output(node.outputs[0].id).as_bool());
    }

    #[test]
    fn test_in_range_executor() {
        let executor = InRangeExecutor;
        assert_eq!(executor.node_type(), "InRange");

        let node = make_test_node("InRange",
            &[("Value", PortType::Float), ("Min", PortType::Float), ("Max", PortType::Float)],
            &[("Result", PortType::Bool)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Float(5.0));
        ctx.set_input(node.inputs[1].id, Value::Float(0.0));
        ctx.set_input(node.inputs[2].id, Value::Float(10.0));
        executor.execute(&node, &mut ctx);
        assert!(ctx.get_output(node.outputs[0].id).as_bool());

        ctx.set_input(node.inputs[0].id, Value::Float(15.0));
        executor.execute(&node, &mut ctx);
        assert!(!ctx.get_output(node.outputs[0].id).as_bool());
    }

    // -------------------------------------------------------------------------
    // Math Node Tests (10)
    // -------------------------------------------------------------------------

    #[test]
    fn test_add_executor() {
        let executor = AddExecutor;
        assert_eq!(executor.node_type(), "Add");

        let node = make_test_node("Add",
            &[("A", PortType::Float), ("B", PortType::Float)],
            &[("Result", PortType::Float)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Float(3.0));
        ctx.set_input(node.inputs[1].id, Value::Float(4.0));
        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_output(node.outputs[0].id).as_float(), 7.0);
    }

    #[test]
    fn test_subtract_executor() {
        let executor = SubtractExecutor;
        assert_eq!(executor.node_type(), "Subtract");

        let node = make_test_node("Subtract",
            &[("A", PortType::Float), ("B", PortType::Float)],
            &[("Result", PortType::Float)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Float(10.0));
        ctx.set_input(node.inputs[1].id, Value::Float(3.0));
        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_output(node.outputs[0].id).as_float(), 7.0);
    }

    #[test]
    fn test_multiply_executor() {
        let executor = MultiplyExecutor;
        assert_eq!(executor.node_type(), "Multiply");

        let node = make_test_node("Multiply",
            &[("A", PortType::Float), ("B", PortType::Float)],
            &[("Result", PortType::Float)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Float(3.0));
        ctx.set_input(node.inputs[1].id, Value::Float(4.0));
        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_output(node.outputs[0].id).as_float(), 12.0);
    }

    #[test]
    fn test_divide_executor() {
        let executor = DivideExecutor;
        assert_eq!(executor.node_type(), "Divide");

        let node = make_test_node("Divide",
            &[("A", PortType::Float), ("B", PortType::Float)],
            &[("Result", PortType::Float)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Float(12.0));
        ctx.set_input(node.inputs[1].id, Value::Float(4.0));
        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_output(node.outputs[0].id).as_float(), 3.0);
    }

    #[test]
    fn test_divide_by_zero() {
        let executor = DivideExecutor;
        let node = make_test_node("Divide",
            &[("A", PortType::Float), ("B", PortType::Float)],
            &[("Result", PortType::Float)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Float(10.0));
        ctx.set_input(node.inputs[1].id, Value::Float(0.0));
        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_output(node.outputs[0].id).as_float(), 0.0);
    }

    #[test]
    fn test_abs_executor() {
        let executor = AbsExecutor;
        assert_eq!(executor.node_type(), "Abs");

        let node = make_test_node("Abs",
            &[("Value", PortType::Float)],
            &[("Result", PortType::Float)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Float(-5.0));
        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_output(node.outputs[0].id).as_float(), 5.0);
    }

    #[test]
    fn test_sin_executor() {
        let executor = SinExecutor;
        assert_eq!(executor.node_type(), "Sin");

        let node = make_test_node("Sin",
            &[("Angle", PortType::Float)],
            &[("Result", PortType::Float)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Float(0.0));
        executor.execute(&node, &mut ctx);
        assert!((ctx.get_output(node.outputs[0].id).as_float() - 0.0).abs() < 0.001);

        ctx.set_input(node.inputs[0].id, Value::Float(std::f32::consts::PI / 2.0));
        executor.execute(&node, &mut ctx);
        assert!((ctx.get_output(node.outputs[0].id).as_float() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cos_executor() {
        let executor = CosExecutor;
        assert_eq!(executor.node_type(), "Cos");

        let node = make_test_node("Cos",
            &[("Angle", PortType::Float)],
            &[("Result", PortType::Float)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Float(0.0));
        executor.execute(&node, &mut ctx);
        assert!((ctx.get_output(node.outputs[0].id).as_float() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_lerp_executor() {
        let executor = LerpExecutor;
        assert_eq!(executor.node_type(), "Lerp");

        let node = make_test_node("Lerp",
            &[("A", PortType::Float), ("B", PortType::Float), ("T", PortType::Float)],
            &[("Result", PortType::Float)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Float(0.0));
        ctx.set_input(node.inputs[1].id, Value::Float(10.0));
        ctx.set_input(node.inputs[2].id, Value::Float(0.5));
        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_output(node.outputs[0].id).as_float(), 5.0);
    }

    #[test]
    fn test_clamp_executor() {
        let executor = ClampExecutor;
        assert_eq!(executor.node_type(), "Clamp");

        let node = make_test_node("Clamp",
            &[("Value", PortType::Float), ("Min", PortType::Float), ("Max", PortType::Float)],
            &[("Result", PortType::Float)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Float(15.0));
        ctx.set_input(node.inputs[1].id, Value::Float(0.0));
        ctx.set_input(node.inputs[2].id, Value::Float(10.0));
        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_output(node.outputs[0].id).as_float(), 10.0);

        ctx.set_input(node.inputs[0].id, Value::Float(-5.0));
        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_output(node.outputs[0].id).as_float(), 0.0);
    }

    #[test]
    fn test_random_executor() {
        let executor = RandomExecutor;
        assert_eq!(executor.node_type(), "Random");

        let node = make_test_node("Random", &[], &[("Value", PortType::Float)]);

        let mut ctx = ExecutionContext::new();
        executor.execute(&node, &mut ctx);
        let r1 = ctx.get_output(node.outputs[0].id).as_float();
        assert!(r1 >= 0.0 && r1 < 1.0);
    }

    // -------------------------------------------------------------------------
    // Flow Control Node Tests (6)
    // -------------------------------------------------------------------------

    #[test]
    fn test_sequence_executor() {
        let executor = SequenceExecutor;
        assert_eq!(executor.node_type(), "Sequence");

        let node = make_test_node("Sequence",
            &[("In", PortType::Flow)],
            &[("1", PortType::Flow), ("2", PortType::Flow), ("3", PortType::Flow)]);

        let mut ctx = ExecutionContext::new();
        let result = executor.execute(&node, &mut ctx);

        if let ExecutionResult::Continue(ports) = result {
            assert_eq!(ports.len(), 3);
        } else {
            panic!("Expected Continue result");
        }
    }

    #[test]
    fn test_branch_executor_true() {
        let executor = BranchExecutor;
        assert_eq!(executor.node_type(), "Branch");

        let node = make_test_node("Branch",
            &[("In", PortType::Flow), ("Condition", PortType::Bool)],
            &[("True", PortType::Flow), ("False", PortType::Flow)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::Bool(true));
        let result = executor.execute(&node, &mut ctx);

        if let ExecutionResult::Continue(ports) = result {
            assert_eq!(ports.len(), 1);
            let true_port = node.outputs.iter().find(|p| p.name == "True").unwrap();
            assert_eq!(ports[0], true_port.id);
        } else {
            panic!("Expected Continue result");
        }
    }

    #[test]
    fn test_branch_executor_false() {
        let executor = BranchExecutor;
        let node = make_test_node("Branch",
            &[("In", PortType::Flow), ("Condition", PortType::Bool)],
            &[("True", PortType::Flow), ("False", PortType::Flow)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::Bool(false));
        let result = executor.execute(&node, &mut ctx);

        if let ExecutionResult::Continue(ports) = result {
            let false_port = node.outputs.iter().find(|p| p.name == "False").unwrap();
            assert_eq!(ports[0], false_port.id);
        } else {
            panic!("Expected Continue result");
        }
    }

    #[test]
    fn test_for_loop_executor() {
        let executor = ForLoopExecutor;
        assert_eq!(executor.node_type(), "ForLoop");

        let node = make_test_node("ForLoop",
            &[("In", PortType::Flow), ("Count", PortType::Int)],
            &[("Loop", PortType::Flow), ("Index", PortType::Int), ("Completed", PortType::Flow)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::Int(3));

        // First iteration
        let result = executor.execute(&node, &mut ctx);
        assert!(matches!(result, ExecutionResult::Continue(_)));
        let index_port = node.outputs.iter().find(|p| p.name == "Index").unwrap();
        assert_eq!(ctx.get_output(index_port.id).as_int(), 0);

        // Second iteration
        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_output(index_port.id).as_int(), 1);

        // Third iteration
        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_output(index_port.id).as_int(), 2);

        // Loop should complete
        let result = executor.execute(&node, &mut ctx);
        if let ExecutionResult::Continue(ports) = result {
            let completed_port = node.outputs.iter().find(|p| p.name == "Completed").unwrap();
            assert_eq!(ports[0], completed_port.id);
        }
    }

    #[test]
    fn test_while_loop_executor() {
        let executor = WhileLoopExecutor;
        assert_eq!(executor.node_type(), "WhileLoop");

        let node = make_test_node("WhileLoop",
            &[("In", PortType::Flow), ("Condition", PortType::Bool)],
            &[("Loop", PortType::Flow), ("Completed", PortType::Flow)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::Bool(true));

        let result = executor.execute(&node, &mut ctx);
        if let ExecutionResult::Continue(ports) = result {
            let loop_port = node.outputs.iter().find(|p| p.name == "Loop").unwrap();
            assert_eq!(ports[0], loop_port.id);
        }

        ctx.set_input(node.inputs[1].id, Value::Bool(false));
        let result = executor.execute(&node, &mut ctx);
        if let ExecutionResult::Continue(ports) = result {
            let completed_port = node.outputs.iter().find(|p| p.name == "Completed").unwrap();
            assert_eq!(ports[0], completed_port.id);
        }
    }

    #[test]
    fn test_do_once_executor() {
        let executor = DoOnceExecutor;
        assert_eq!(executor.node_type(), "DoOnce");

        let mut node = make_test_node("DoOnce",
            &[("In", PortType::Flow)],
            &[("Out", PortType::Flow)]);
        node.properties.push(crate::flowforge::NodeProperty::bool("Reset", false));

        let mut ctx = ExecutionContext::new();

        // First execution should pass
        let result = executor.execute(&node, &mut ctx);
        assert!(matches!(result, ExecutionResult::Continue(ports) if !ports.is_empty()));

        // Second execution should not pass
        let result = executor.execute(&node, &mut ctx);
        assert!(matches!(result, ExecutionResult::Continue(ports) if ports.is_empty()));
    }

    #[test]
    fn test_delay_executor() {
        let executor = DelayExecutor;
        assert_eq!(executor.node_type(), "Delay");

        let node = make_test_node("Delay",
            &[("In", PortType::Flow), ("Duration", PortType::Float)],
            &[("Out", PortType::Flow)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::Float(1.0));
        ctx.delta_time = 0.5;

        // First tick - should be waiting
        let result = executor.execute(&node, &mut ctx);
        assert!(matches!(result, ExecutionResult::Waiting));

        // Second tick - should complete
        let result = executor.execute(&node, &mut ctx);
        assert!(matches!(result, ExecutionResult::Continue(_)));
    }

    // -------------------------------------------------------------------------
    // ECS Node Tests (6)
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_component_executor() {
        let executor = GetComponentExecutor;
        assert_eq!(executor.node_type(), "GetComponent");

        let mut node = make_test_node("GetComponent",
            &[("Entity", PortType::Entity)],
            &[("Value", PortType::Any)]);
        node.properties.push(crate::flowforge::NodeProperty::string("Component", "Health"));

        let mut ctx = ExecutionContext::new();
        ctx.set_component(1, "Health", Value::Int(100));
        ctx.set_input(node.inputs[0].id, Value::Entity(1));

        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_output(node.outputs[0].id), Value::Int(100));
    }

    #[test]
    fn test_set_component_executor() {
        let executor = SetComponentExecutor;
        assert_eq!(executor.node_type(), "SetComponent");

        let mut node = make_test_node("SetComponent",
            &[("In", PortType::Flow), ("Entity", PortType::Entity), ("Value", PortType::Any)],
            &[("Out", PortType::Flow)]);
        node.properties.push(crate::flowforge::NodeProperty::string("Component", "Health"));

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::Entity(1));
        ctx.set_input(node.inputs[2].id, Value::Int(50));

        executor.execute(&node, &mut ctx);
        assert_eq!(ctx.get_component(1, "Health"), Value::Int(50));
    }

    #[test]
    fn test_has_component_executor() {
        let executor = HasComponentExecutor;
        assert_eq!(executor.node_type(), "HasComponent");

        let mut node = make_test_node("HasComponent",
            &[("Entity", PortType::Entity)],
            &[("Result", PortType::Bool)]);
        node.properties.push(crate::flowforge::NodeProperty::string("Component", "Health"));

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Entity(1));

        executor.execute(&node, &mut ctx);
        assert!(!ctx.get_output(node.outputs[0].id).as_bool());

        ctx.set_component(1, "Health", Value::Int(100));
        executor.execute(&node, &mut ctx);
        assert!(ctx.get_output(node.outputs[0].id).as_bool());
    }

    #[test]
    fn test_add_component_executor() {
        let executor = AddComponentExecutor;
        assert_eq!(executor.node_type(), "AddComponent");

        let mut node = make_test_node("AddComponent",
            &[("In", PortType::Flow), ("Entity", PortType::Entity)],
            &[("Out", PortType::Flow)]);
        node.properties.push(crate::flowforge::NodeProperty::string("Component", "Velocity"));

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::Entity(1));

        assert!(!ctx.has_component(1, "Velocity"));
        executor.execute(&node, &mut ctx);
        assert!(ctx.has_component(1, "Velocity"));
    }

    #[test]
    fn test_remove_component_executor() {
        let executor = RemoveComponentExecutor;
        assert_eq!(executor.node_type(), "RemoveComponent");

        let mut node = make_test_node("RemoveComponent",
            &[("In", PortType::Flow), ("Entity", PortType::Entity)],
            &[("Out", PortType::Flow)]);
        node.properties.push(crate::flowforge::NodeProperty::string("Component", "Velocity"));

        let mut ctx = ExecutionContext::new();
        ctx.set_component(1, "Velocity", Value::Vec3([1.0, 0.0, 0.0]));
        ctx.set_input(node.inputs[1].id, Value::Entity(1));

        assert!(ctx.has_component(1, "Velocity"));
        executor.execute(&node, &mut ctx);
        assert!(!ctx.has_component(1, "Velocity"));
    }

    #[test]
    fn test_query_entities_executor() {
        let executor = QueryEntitiesExecutor;
        assert_eq!(executor.node_type(), "QueryEntities");

        let node = make_test_node("QueryEntities",
            &[],
            &[("Entities", PortType::Any)]);

        let mut ctx = ExecutionContext::new();
        executor.execute(&node, &mut ctx);
        let result = ctx.get_output(node.outputs[0].id);
        assert!(matches!(result, Value::List(_)));
    }

    // -------------------------------------------------------------------------
    // Debug Node Tests (4)
    // -------------------------------------------------------------------------

    #[test]
    fn test_print_executor() {
        let executor = PrintExecutor;
        assert_eq!(executor.node_type(), "Print");

        let node = make_test_node("Print",
            &[("In", PortType::Flow), ("Value", PortType::Any)],
            &[("Out", PortType::Flow)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::String("Hello World".to_string()));

        executor.execute(&node, &mut ctx);
        assert!(ctx.debug_log.iter().any(|s| s.contains("Hello World")));
    }

    #[test]
    fn test_log_executor() {
        let executor = LogExecutor;
        assert_eq!(executor.node_type(), "Log");

        let mut node = make_test_node("Log",
            &[("In", PortType::Flow), ("Message", PortType::String)],
            &[("Out", PortType::Flow)]);
        node.properties.push(crate::flowforge::NodeProperty::enum_prop(
            "Level",
            vec!["INFO".to_string(), "WARN".to_string(), "ERROR".to_string()],
            1
        ));

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::String("Warning message".to_string()));

        executor.execute(&node, &mut ctx);
        assert!(ctx.debug_log.iter().any(|s| s.contains("WARN")));
    }

    #[test]
    fn test_assert_executor_pass() {
        let executor = AssertExecutor;
        assert_eq!(executor.node_type(), "Assert");

        let node = make_test_node("Assert",
            &[("In", PortType::Flow), ("Condition", PortType::Bool)],
            &[("Out", PortType::Flow)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::Bool(true));

        let result = executor.execute(&node, &mut ctx);
        assert!(matches!(result, ExecutionResult::Continue(_)));
    }

    #[test]
    fn test_assert_executor_fail() {
        let executor = AssertExecutor;

        let mut node = make_test_node("Assert",
            &[("In", PortType::Flow), ("Condition", PortType::Bool)],
            &[("Out", PortType::Flow)]);
        node.properties.push(crate::flowforge::NodeProperty::string("Message", "Test failed"));

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[1].id, Value::Bool(false));

        let result = executor.execute(&node, &mut ctx);
        assert!(matches!(result, ExecutionResult::Error(_)));
    }

    #[test]
    fn test_breakpoint_executor() {
        let executor = BreakpointExecutor;
        assert_eq!(executor.node_type(), "Breakpoint");

        let node = make_test_node("Breakpoint",
            &[("In", PortType::Flow)],
            &[("Out", PortType::Flow)]);

        let mut ctx = ExecutionContext::new();
        let result = executor.execute(&node, &mut ctx);

        assert!(matches!(result, ExecutionResult::Continue(_)));
        assert!(ctx.debug_log.iter().any(|s| s.contains("BREAKPOINT")));
    }

    // -------------------------------------------------------------------------
    // NodeRegistry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_registry_new() {
        let registry = NodeRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_with_builtins() {
        let registry = NodeRegistry::with_builtins();
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 46); // 6 + 8 + 6 + 10 + 6 + 6 + 4 = 46

        // Check all categories are registered
        assert!(registry.get("OnStart").is_some());
        assert!(registry.get("SetPosition").is_some());
        assert!(registry.get("Compare").is_some());
        assert!(registry.get("Add").is_some());
        assert!(registry.get("Branch").is_some());
        assert!(registry.get("GetComponent").is_some());
        assert!(registry.get("Print").is_some());
    }

    #[test]
    fn test_registry_register() {
        let mut registry = NodeRegistry::new();
        registry.register(Box::new(AddExecutor));
        assert_eq!(registry.len(), 1);
        assert!(registry.get("Add").is_some());
    }

    #[test]
    fn test_registry_execute_node() {
        let registry = NodeRegistry::with_builtins();
        let node = make_test_node("Add",
            &[("A", PortType::Float), ("B", PortType::Float)],
            &[("Result", PortType::Float)]);

        let mut ctx = ExecutionContext::new();
        ctx.set_input(node.inputs[0].id, Value::Float(2.0));
        ctx.set_input(node.inputs[1].id, Value::Float(3.0));

        let result = registry.execute_node(&node, &mut ctx);
        assert!(matches!(result, ExecutionResult::Continue(_)));
        assert_eq!(ctx.get_output(node.outputs[0].id).as_float(), 5.0);
    }

    #[test]
    fn test_registry_execute_unknown_node() {
        let registry = NodeRegistry::with_builtins();
        let node = make_test_node("UnknownNode", &[], &[]);

        let mut ctx = ExecutionContext::new();
        let result = registry.execute_node(&node, &mut ctx);
        assert!(matches!(result, ExecutionResult::Error(_)));
    }

    #[test]
    fn test_registry_node_types() {
        let registry = NodeRegistry::with_builtins();
        let types = registry.node_types();
        assert!(types.contains(&"Add"));
        assert!(types.contains(&"Branch"));
        assert!(types.contains(&"OnStart"));
    }

    #[test]
    fn test_registry_event_executors() {
        let registry = NodeRegistry::with_builtins();
        let events = registry.event_executors();
        assert_eq!(events.len(), 6);

        for executor in events {
            assert!(executor.is_event());
        }
    }

    #[test]
    fn test_registry_default() {
        let registry = NodeRegistry::default();
        assert!(!registry.is_empty());
    }
}
