//! FlowForge Node Graph Editor
//!
//! A visual node-graph scripting system for TRINITY. This module provides:
//! - Core types for nodes, ports, and connections
//! - NodeGraph container with topological sorting and cycle detection
//! - Node templates and categories for organizing node palettes
//! - GraphEditor UI for rendering and interacting with the graph
//!
//! # Architecture
//!
//! ```text
//! NodeCategory         NodeTemplate         Node
//! ============         ============         ====
//! "Math"               "Add"           -->  Node { id: 1, ... }
//! "Logic"              "Branch"        -->  Node { id: 2, ... }
//! "Flow"               "Sequence"      -->  Node { id: 3, ... }
//!
//! NodeGraph
//! =========
//! nodes: HashMap<NodeId, Node>
//! connections: Vec<Connection>
//!
//! GraphEditor
//! ===========
//! Renders NodeGraph via UIContext trait
//! Returns GraphAction events
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use renderer_backend::flowforge::{NodeGraph, NodeTemplate, GraphEditor};
//!
//! let mut graph = NodeGraph::new();
//! let template = NodeTemplate::new("Add", "Math")
//!     .with_input("A", PortType::Float)
//!     .with_input("B", PortType::Float)
//!     .with_output("Result", PortType::Float);
//!
//! let node_id = graph.add_node(&template);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Core Types
// ---------------------------------------------------------------------------

/// Unique identifier for a node in the graph.
pub type NodeId = u64;

/// Unique identifier for a port on a node.
pub type PortId = u64;

/// The data type that flows through a port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PortType {
    /// Execution flow (control flow between nodes).
    Flow,
    /// Boolean value.
    Bool,
    /// Integer value.
    Int,
    /// Floating-point value.
    Float,
    /// 3D vector (x, y, z).
    Vec3,
    /// String value.
    String,
    /// Entity reference.
    Entity,
    /// Wildcard - compatible with any type.
    Any,
}

impl PortType {
    /// Check if this type is compatible with another type for connection.
    pub fn is_compatible_with(&self, other: &PortType) -> bool {
        match (self, other) {
            // Any is compatible with everything
            (PortType::Any, _) | (_, PortType::Any) => true,
            // Same types are compatible
            (a, b) if a == b => true,
            // Numeric types can be implicitly converted
            (PortType::Int, PortType::Float) | (PortType::Float, PortType::Int) => true,
            // Everything else is incompatible
            _ => false,
        }
    }

    /// Get a display name for the type.
    pub fn display_name(&self) -> &'static str {
        match self {
            PortType::Flow => "Flow",
            PortType::Bool => "Bool",
            PortType::Int => "Int",
            PortType::Float => "Float",
            PortType::Vec3 => "Vec3",
            PortType::String => "String",
            PortType::Entity => "Entity",
            PortType::Any => "Any",
        }
    }

    /// Get a color associated with this type (for UI rendering).
    pub fn color(&self) -> [f32; 4] {
        match self {
            PortType::Flow => [1.0, 1.0, 1.0, 1.0],    // White
            PortType::Bool => [0.8, 0.2, 0.2, 1.0],    // Red
            PortType::Int => [0.2, 0.8, 0.2, 1.0],     // Green
            PortType::Float => [0.2, 0.6, 1.0, 1.0],   // Blue
            PortType::Vec3 => [1.0, 0.8, 0.2, 1.0],    // Yellow
            PortType::String => [0.9, 0.4, 0.9, 1.0],  // Magenta
            PortType::Entity => [0.4, 0.9, 0.9, 1.0],  // Cyan
            PortType::Any => [0.7, 0.7, 0.7, 1.0],     // Gray
        }
    }
}

/// Direction of a port (input or output).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PortDirection {
    /// Input port - receives data.
    Input,
    /// Output port - sends data.
    Output,
}

/// A port on a node that can be connected to other ports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Port {
    /// Unique identifier for this port.
    pub id: PortId,
    /// Display name.
    pub name: String,
    /// The type of data this port handles.
    pub port_type: PortType,
    /// Whether this is an input or output port.
    pub direction: PortDirection,
}

impl Port {
    /// Create a new port.
    pub fn new(id: PortId, name: impl Into<String>, port_type: PortType, direction: PortDirection) -> Self {
        Self {
            id,
            name: name.into(),
            port_type,
            direction,
        }
    }

    /// Check if this port can connect to another port.
    pub fn can_connect_to(&self, other: &Port) -> bool {
        // Can only connect input to output and vice versa
        if self.direction == other.direction {
            return false;
        }
        // Types must be compatible
        self.port_type.is_compatible_with(&other.port_type)
    }
}

// ---------------------------------------------------------------------------
// Node Properties
// ---------------------------------------------------------------------------

/// A property value that can be set on a node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyValue {
    /// Boolean value.
    Bool(bool),
    /// Integer value.
    Int(i32),
    /// Float value.
    Float(f32),
    /// String value.
    String(String),
    /// 3D vector.
    Vec3([f32; 3]),
    /// Color (RGBA).
    Color([f32; 4]),
    /// Enum selection (index into options).
    Enum(usize),
}

impl Default for PropertyValue {
    fn default() -> Self {
        PropertyValue::Bool(false)
    }
}

/// A property on a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeProperty {
    /// Property name.
    pub name: String,
    /// Current value.
    pub value: PropertyValue,
    /// For enum properties, the available options.
    pub enum_options: Option<Vec<String>>,
}

impl NodeProperty {
    /// Create a new boolean property.
    pub fn bool(name: impl Into<String>, value: bool) -> Self {
        Self {
            name: name.into(),
            value: PropertyValue::Bool(value),
            enum_options: None,
        }
    }

    /// Create a new integer property.
    pub fn int(name: impl Into<String>, value: i32) -> Self {
        Self {
            name: name.into(),
            value: PropertyValue::Int(value),
            enum_options: None,
        }
    }

    /// Create a new float property.
    pub fn float(name: impl Into<String>, value: f32) -> Self {
        Self {
            name: name.into(),
            value: PropertyValue::Float(value),
            enum_options: None,
        }
    }

    /// Create a new string property.
    pub fn string(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: PropertyValue::String(value.into()),
            enum_options: None,
        }
    }

    /// Create a new vec3 property.
    pub fn vec3(name: impl Into<String>, value: [f32; 3]) -> Self {
        Self {
            name: name.into(),
            value: PropertyValue::Vec3(value),
            enum_options: None,
        }
    }

    /// Create a new color property.
    pub fn color(name: impl Into<String>, value: [f32; 4]) -> Self {
        Self {
            name: name.into(),
            value: PropertyValue::Color(value),
            enum_options: None,
        }
    }

    /// Create a new enum property.
    pub fn enum_prop(name: impl Into<String>, options: Vec<String>, selected: usize) -> Self {
        Self {
            name: name.into(),
            value: PropertyValue::Enum(selected),
            enum_options: Some(options),
        }
    }
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// A node in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique identifier.
    pub id: NodeId,
    /// Display name.
    pub name: String,
    /// Category (e.g., "Math", "Logic", "Flow").
    pub category: String,
    /// Position in the graph editor (x, y).
    pub position: (f32, f32),
    /// Input ports.
    pub inputs: Vec<Port>,
    /// Output ports.
    pub outputs: Vec<Port>,
    /// Node properties.
    pub properties: Vec<NodeProperty>,
}

impl Node {
    /// Get an input port by ID.
    pub fn get_input(&self, port_id: PortId) -> Option<&Port> {
        self.inputs.iter().find(|p| p.id == port_id)
    }

    /// Get an output port by ID.
    pub fn get_output(&self, port_id: PortId) -> Option<&Port> {
        self.outputs.iter().find(|p| p.id == port_id)
    }

    /// Get any port by ID.
    pub fn get_port(&self, port_id: PortId) -> Option<&Port> {
        self.get_input(port_id).or_else(|| self.get_output(port_id))
    }

    /// Get a property by name.
    pub fn get_property(&self, name: &str) -> Option<&NodeProperty> {
        self.properties.iter().find(|p| p.name == name)
    }

    /// Get a mutable property by name.
    pub fn get_property_mut(&mut self, name: &str) -> Option<&mut NodeProperty> {
        self.properties.iter_mut().find(|p| p.name == name)
    }

    /// Check if this node has any flow inputs.
    pub fn has_flow_input(&self) -> bool {
        self.inputs.iter().any(|p| p.port_type == PortType::Flow)
    }

    /// Check if this node has any flow outputs.
    pub fn has_flow_output(&self) -> bool {
        self.outputs.iter().any(|p| p.port_type == PortType::Flow)
    }
}

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

/// A connection between two ports.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Connection {
    /// Source node ID.
    pub from_node: NodeId,
    /// Source port ID (must be an output port).
    pub from_port: PortId,
    /// Target node ID.
    pub to_node: NodeId,
    /// Target port ID (must be an input port).
    pub to_port: PortId,
}

impl Connection {
    /// Create a new connection.
    pub fn new(from_node: NodeId, from_port: PortId, to_node: NodeId, to_port: PortId) -> Self {
        Self {
            from_node,
            from_port,
            to_node,
            to_port,
        }
    }

    /// Create a connection from tuples.
    pub fn from_tuples(from: (NodeId, PortId), to: (NodeId, PortId)) -> Self {
        Self {
            from_node: from.0,
            from_port: from.1,
            to_node: to.0,
            to_port: to.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Error when creating a connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionError {
    /// Source node not found.
    SourceNodeNotFound(NodeId),
    /// Target node not found.
    TargetNodeNotFound(NodeId),
    /// Source port not found.
    SourcePortNotFound(NodeId, PortId),
    /// Target port not found.
    TargetPortNotFound(NodeId, PortId),
    /// Source port is not an output.
    SourceNotOutput(NodeId, PortId),
    /// Target port is not an input.
    TargetNotInput(NodeId, PortId),
    /// Port types are incompatible.
    IncompatibleTypes(PortType, PortType),
    /// Self-connection (same node).
    SelfConnection(NodeId),
    /// Connection already exists.
    AlreadyConnected,
    /// Connection would create a cycle.
    WouldCreateCycle,
    /// Target input already has a connection.
    InputAlreadyConnected(NodeId, PortId),
}

impl std::fmt::Display for ConnectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectionError::SourceNodeNotFound(id) => {
                write!(f, "Source node {} not found", id)
            }
            ConnectionError::TargetNodeNotFound(id) => {
                write!(f, "Target node {} not found", id)
            }
            ConnectionError::SourcePortNotFound(node, port) => {
                write!(f, "Source port {} on node {} not found", port, node)
            }
            ConnectionError::TargetPortNotFound(node, port) => {
                write!(f, "Target port {} on node {} not found", port, node)
            }
            ConnectionError::SourceNotOutput(node, port) => {
                write!(f, "Port {} on node {} is not an output", port, node)
            }
            ConnectionError::TargetNotInput(node, port) => {
                write!(f, "Port {} on node {} is not an input", port, node)
            }
            ConnectionError::IncompatibleTypes(from, to) => {
                write!(f, "Incompatible port types: {} -> {}", from.display_name(), to.display_name())
            }
            ConnectionError::SelfConnection(id) => {
                write!(f, "Cannot connect node {} to itself", id)
            }
            ConnectionError::AlreadyConnected => {
                write!(f, "Connection already exists")
            }
            ConnectionError::WouldCreateCycle => {
                write!(f, "Connection would create a cycle")
            }
            ConnectionError::InputAlreadyConnected(node, port) => {
                write!(f, "Input port {} on node {} already has a connection", port, node)
            }
        }
    }
}

impl std::error::Error for ConnectionError {}

/// Error when a cycle is detected in the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleError {
    /// Nodes involved in the cycle.
    pub nodes: Vec<NodeId>,
}

impl std::fmt::Display for CycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cycle detected involving nodes: {:?}", self.nodes)
    }
}

impl std::error::Error for CycleError {}

/// Graph validation error.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    /// Cycle detected.
    Cycle(CycleError),
    /// Disconnected input (required input has no connection).
    DisconnectedInput { node: NodeId, port: PortId, port_name: String },
    /// Invalid connection.
    InvalidConnection(Connection, ConnectionError),
    /// Orphaned node (no connections at all).
    OrphanedNode(NodeId),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::Cycle(e) => write!(f, "{}", e),
            GraphError::DisconnectedInput { node, port_name, .. } => {
                write!(f, "Input '{}' on node {} is not connected", port_name, node)
            }
            GraphError::InvalidConnection(conn, err) => {
                write!(f, "Invalid connection from {}:{} to {}:{}: {}",
                       conn.from_node, conn.from_port, conn.to_node, conn.to_port, err)
            }
            GraphError::OrphanedNode(id) => {
                write!(f, "Node {} has no connections", id)
            }
        }
    }
}

impl std::error::Error for GraphError {}

// ---------------------------------------------------------------------------
// Node Templates
// ---------------------------------------------------------------------------

/// Template for a port on a node template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortTemplate {
    /// Display name.
    pub name: String,
    /// Port type.
    pub port_type: PortType,
}

impl PortTemplate {
    /// Create a new port template.
    pub fn new(name: impl Into<String>, port_type: PortType) -> Self {
        Self {
            name: name.into(),
            port_type,
        }
    }
}

/// Template for a property on a node template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyTemplate {
    /// Property name.
    pub name: String,
    /// Default value.
    pub default_value: PropertyValue,
    /// Enum options (for enum properties).
    pub enum_options: Option<Vec<String>>,
}

impl PropertyTemplate {
    /// Create a boolean property template.
    pub fn bool(name: impl Into<String>, default: bool) -> Self {
        Self {
            name: name.into(),
            default_value: PropertyValue::Bool(default),
            enum_options: None,
        }
    }

    /// Create an integer property template.
    pub fn int(name: impl Into<String>, default: i32) -> Self {
        Self {
            name: name.into(),
            default_value: PropertyValue::Int(default),
            enum_options: None,
        }
    }

    /// Create a float property template.
    pub fn float(name: impl Into<String>, default: f32) -> Self {
        Self {
            name: name.into(),
            default_value: PropertyValue::Float(default),
            enum_options: None,
        }
    }

    /// Create a string property template.
    pub fn string(name: impl Into<String>, default: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            default_value: PropertyValue::String(default.into()),
            enum_options: None,
        }
    }

    /// Create an enum property template.
    pub fn enum_prop(name: impl Into<String>, options: Vec<String>, default: usize) -> Self {
        Self {
            name: name.into(),
            default_value: PropertyValue::Enum(default),
            enum_options: Some(options),
        }
    }
}

/// Template for creating nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTemplate {
    /// Node name.
    pub name: String,
    /// Category.
    pub category: String,
    /// Input port templates.
    pub inputs: Vec<PortTemplate>,
    /// Output port templates.
    pub outputs: Vec<PortTemplate>,
    /// Property templates.
    pub properties: Vec<PropertyTemplate>,
}

impl NodeTemplate {
    /// Create a new node template.
    pub fn new(name: impl Into<String>, category: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            category: category.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            properties: Vec::new(),
        }
    }

    /// Add an input port.
    pub fn with_input(mut self, name: impl Into<String>, port_type: PortType) -> Self {
        self.inputs.push(PortTemplate::new(name, port_type));
        self
    }

    /// Add an output port.
    pub fn with_output(mut self, name: impl Into<String>, port_type: PortType) -> Self {
        self.outputs.push(PortTemplate::new(name, port_type));
        self
    }

    /// Add a property.
    pub fn with_property(mut self, template: PropertyTemplate) -> Self {
        self.properties.push(template);
        self
    }

    /// Add a flow input (convenience method).
    pub fn with_flow_input(self) -> Self {
        self.with_input("In", PortType::Flow)
    }

    /// Add a flow output (convenience method).
    pub fn with_flow_output(self) -> Self {
        self.with_output("Out", PortType::Flow)
    }
}

/// A category of node templates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCategory {
    /// Category name.
    pub name: String,
    /// Templates in this category.
    pub templates: Vec<NodeTemplate>,
}

impl NodeCategory {
    /// Create a new category.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            templates: Vec::new(),
        }
    }

    /// Add a template to this category.
    pub fn with_template(mut self, template: NodeTemplate) -> Self {
        self.templates.push(template);
        self
    }
}

// ---------------------------------------------------------------------------
// Node Graph
// ---------------------------------------------------------------------------

/// Container for nodes and their connections.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeGraph {
    /// Nodes in the graph.
    nodes: HashMap<NodeId, Node>,
    /// Connections between nodes.
    connections: Vec<Connection>,
    /// Next node ID to assign.
    next_node_id: NodeId,
    /// Next port ID to assign.
    next_port_id: PortId,
}

impl NodeGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            connections: Vec::new(),
            next_node_id: 1,
            next_port_id: 1,
        }
    }

    /// Add a node from a template.
    pub fn add_node(&mut self, template: &NodeTemplate) -> NodeId {
        self.add_node_at(template, (0.0, 0.0))
    }

    /// Add a node from a template at a specific position.
    pub fn add_node_at(&mut self, template: &NodeTemplate, position: (f32, f32)) -> NodeId {
        let node_id = self.next_node_id;
        self.next_node_id += 1;

        let inputs: Vec<Port> = template.inputs.iter().map(|t| {
            let port_id = self.next_port_id;
            self.next_port_id += 1;
            Port::new(port_id, &t.name, t.port_type, PortDirection::Input)
        }).collect();

        let outputs: Vec<Port> = template.outputs.iter().map(|t| {
            let port_id = self.next_port_id;
            self.next_port_id += 1;
            Port::new(port_id, &t.name, t.port_type, PortDirection::Output)
        }).collect();

        let properties: Vec<NodeProperty> = template.properties.iter().map(|t| {
            NodeProperty {
                name: t.name.clone(),
                value: t.default_value.clone(),
                enum_options: t.enum_options.clone(),
            }
        }).collect();

        let node = Node {
            id: node_id,
            name: template.name.clone(),
            category: template.category.clone(),
            position,
            inputs,
            outputs,
            properties,
        };

        self.nodes.insert(node_id, node);
        node_id
    }

    /// Remove a node and all its connections.
    pub fn remove_node(&mut self, id: NodeId) {
        self.nodes.remove(&id);
        self.connections.retain(|c| c.from_node != id && c.to_node != id);
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Get a mutable node by ID.
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    /// Get all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// Get all connections.
    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// Get node count.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get connection count.
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Create a connection between two ports.
    pub fn connect(
        &mut self,
        from: (NodeId, PortId),
        to: (NodeId, PortId),
    ) -> Result<(), ConnectionError> {
        let (from_node_id, from_port_id) = from;
        let (to_node_id, to_port_id) = to;

        // Check for self-connection
        if from_node_id == to_node_id {
            return Err(ConnectionError::SelfConnection(from_node_id));
        }

        // Get source node and port
        let from_node = self.nodes.get(&from_node_id)
            .ok_or(ConnectionError::SourceNodeNotFound(from_node_id))?;
        let from_port = from_node.get_output(from_port_id)
            .ok_or(ConnectionError::SourcePortNotFound(from_node_id, from_port_id))?;

        if from_port.direction != PortDirection::Output {
            return Err(ConnectionError::SourceNotOutput(from_node_id, from_port_id));
        }

        // Get target node and port
        let to_node = self.nodes.get(&to_node_id)
            .ok_or(ConnectionError::TargetNodeNotFound(to_node_id))?;
        let to_port = to_node.get_input(to_port_id)
            .ok_or(ConnectionError::TargetPortNotFound(to_node_id, to_port_id))?;

        if to_port.direction != PortDirection::Input {
            return Err(ConnectionError::TargetNotInput(to_node_id, to_port_id));
        }

        // Check type compatibility
        if !from_port.port_type.is_compatible_with(&to_port.port_type) {
            return Err(ConnectionError::IncompatibleTypes(
                from_port.port_type,
                to_port.port_type,
            ));
        }

        // Check if connection already exists
        let connection = Connection::new(from_node_id, from_port_id, to_node_id, to_port_id);
        if self.connections.contains(&connection) {
            return Err(ConnectionError::AlreadyConnected);
        }

        // Check if input already has a connection
        if self.connections.iter().any(|c| c.to_node == to_node_id && c.to_port == to_port_id) {
            return Err(ConnectionError::InputAlreadyConnected(to_node_id, to_port_id));
        }

        // Check for cycles (would adding this connection create a cycle?)
        if self.would_create_cycle(from_node_id, to_node_id) {
            return Err(ConnectionError::WouldCreateCycle);
        }

        self.connections.push(connection);
        Ok(())
    }

    /// Remove a connection.
    pub fn disconnect(&mut self, from: (NodeId, PortId), to: (NodeId, PortId)) {
        let connection = Connection::from_tuples(from, to);
        self.connections.retain(|c| c != &connection);
    }

    /// Check if adding an edge from -> to would create a cycle.
    fn would_create_cycle(&self, from_node: NodeId, to_node: NodeId) -> bool {
        // If to_node can reach from_node, adding from->to creates a cycle
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(to_node);

        while let Some(current) = queue.pop_front() {
            if current == from_node {
                return true;
            }
            if visited.insert(current) {
                // Find all nodes that current connects to
                for conn in &self.connections {
                    if conn.from_node == current {
                        queue.push_back(conn.to_node);
                    }
                }
            }
        }
        false
    }

    /// Validate the graph for errors.
    pub fn validate(&self) -> Vec<GraphError> {
        let mut errors = Vec::new();

        // Check for cycles
        if let Err(cycle_error) = self.topological_sort() {
            errors.push(GraphError::Cycle(cycle_error));
        }

        // Check for invalid connections
        for conn in &self.connections {
            if let Err(e) = self.validate_connection(conn) {
                errors.push(GraphError::InvalidConnection(conn.clone(), e));
            }
        }

        // Check for orphaned nodes (optional - some graphs allow them)
        for (&id, node) in &self.nodes {
            let has_connection = self.connections.iter().any(|c| {
                c.from_node == id || c.to_node == id
            });
            // Only report as orphaned if node has ports (otherwise it's intentional)
            if !has_connection && (!node.inputs.is_empty() || !node.outputs.is_empty()) {
                // Note: This is a warning, not an error in many cases
                // errors.push(GraphError::OrphanedNode(id));
            }
        }

        errors
    }

    /// Validate a single connection.
    fn validate_connection(&self, conn: &Connection) -> Result<(), ConnectionError> {
        let from_node = self.nodes.get(&conn.from_node)
            .ok_or(ConnectionError::SourceNodeNotFound(conn.from_node))?;
        let from_port = from_node.get_output(conn.from_port)
            .ok_or(ConnectionError::SourcePortNotFound(conn.from_node, conn.from_port))?;

        let to_node = self.nodes.get(&conn.to_node)
            .ok_or(ConnectionError::TargetNodeNotFound(conn.to_node))?;
        let to_port = to_node.get_input(conn.to_port)
            .ok_or(ConnectionError::TargetPortNotFound(conn.to_node, conn.to_port))?;

        if !from_port.port_type.is_compatible_with(&to_port.port_type) {
            return Err(ConnectionError::IncompatibleTypes(
                from_port.port_type,
                to_port.port_type,
            ));
        }

        Ok(())
    }

    /// Topological sort of nodes. Returns error if cycle detected.
    pub fn topological_sort(&self) -> Result<Vec<NodeId>, CycleError> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        // Initialize
        for &id in self.nodes.keys() {
            in_degree.insert(id, 0);
            adjacency.insert(id, Vec::new());
        }

        // Build adjacency list and compute in-degrees
        for conn in &self.connections {
            adjacency.get_mut(&conn.from_node).map(|v| v.push(conn.to_node));
            *in_degree.get_mut(&conn.to_node).unwrap() += 1;
        }

        // Start with nodes that have no incoming edges
        let mut queue: VecDeque<NodeId> = in_degree.iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut sorted = Vec::new();

        while let Some(node) = queue.pop_front() {
            sorted.push(node);

            if let Some(neighbors) = adjacency.get(&node) {
                for &neighbor in neighbors {
                    let deg = in_degree.get_mut(&neighbor).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // If not all nodes are in sorted, there's a cycle
        if sorted.len() != self.nodes.len() {
            let cycle_nodes: Vec<NodeId> = in_degree.iter()
                .filter(|(_, &deg)| deg > 0)
                .map(|(&id, _)| id)
                .collect();
            return Err(CycleError { nodes: cycle_nodes });
        }

        Ok(sorted)
    }

    /// Get connections to a specific node's input port.
    pub fn connections_to(&self, node_id: NodeId, port_id: PortId) -> Vec<&Connection> {
        self.connections.iter()
            .filter(|c| c.to_node == node_id && c.to_port == port_id)
            .collect()
    }

    /// Get connections from a specific node's output port.
    pub fn connections_from(&self, node_id: NodeId, port_id: PortId) -> Vec<&Connection> {
        self.connections.iter()
            .filter(|c| c.from_node == node_id && c.from_port == port_id)
            .collect()
    }

    /// Get all connections involving a node.
    pub fn connections_for_node(&self, node_id: NodeId) -> Vec<&Connection> {
        self.connections.iter()
            .filter(|c| c.from_node == node_id || c.to_node == node_id)
            .collect()
    }

    /// Serialize the graph to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a graph from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Clear the graph.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.connections.clear();
    }
}

// ---------------------------------------------------------------------------
// Graph Actions (events from GraphEditor)
// ---------------------------------------------------------------------------

/// Actions that can occur during graph editing.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphAction {
    /// No action occurred.
    None,
    /// A node was added.
    NodeAdded(NodeId),
    /// A node was removed.
    NodeRemoved(NodeId),
    /// A connection was created.
    Connected(Connection),
    /// A connection was removed.
    Disconnected(Connection),
    /// A node was moved.
    NodeMoved { id: NodeId, position: (f32, f32) },
    /// A node property was changed.
    PropertyChanged {
        node: NodeId,
        property: String,
        value: PropertyValue,
    },
    /// A node was selected.
    NodeSelected(NodeId),
    /// Selection was cleared.
    SelectionCleared,
    /// Multiple actions occurred (batch).
    Batch(Vec<GraphAction>),
}

impl GraphAction {
    /// Check if this is no action.
    pub fn is_none(&self) -> bool {
        matches!(self, GraphAction::None)
    }

    /// Check if any action occurred.
    pub fn is_some(&self) -> bool {
        !self.is_none()
    }
}

// ---------------------------------------------------------------------------
// Graph Editor
// ---------------------------------------------------------------------------

/// State for the graph editor UI.
#[derive(Debug, Clone, Default)]
pub struct GraphEditor {
    /// Available node categories (palette).
    categories: Vec<NodeCategory>,
    /// Currently selected node.
    selected_node: Option<NodeId>,
    /// Node being dragged.
    dragging_node: Option<NodeId>,
    /// Port being connected from.
    connecting_from: Option<(NodeId, PortId, PortDirection)>,
    /// Pan offset.
    pan_offset: (f32, f32),
    /// Zoom level.
    zoom: f32,
    /// Whether the palette is open.
    palette_open: bool,
    /// Search filter for palette.
    palette_search: String,
    /// Grid size for snapping.
    grid_size: f32,
    /// Whether to show grid.
    show_grid: bool,
}

impl GraphEditor {
    /// Create a new graph editor.
    pub fn new() -> Self {
        Self {
            categories: Vec::new(),
            selected_node: None,
            dragging_node: None,
            connecting_from: None,
            pan_offset: (0.0, 0.0),
            zoom: 1.0,
            palette_open: false,
            palette_search: String::new(),
            grid_size: 20.0,
            show_grid: true,
        }
    }

    /// Set the node palette.
    pub fn set_palette(&mut self, categories: Vec<NodeCategory>) {
        self.categories = categories;
    }

    /// Get the palette categories.
    pub fn categories(&self) -> &[NodeCategory] {
        &self.categories
    }

    /// Get the currently selected node.
    pub fn selected_node(&self) -> Option<NodeId> {
        self.selected_node
    }

    /// Set the selected node.
    pub fn select_node(&mut self, id: Option<NodeId>) {
        self.selected_node = id;
    }

    /// Get the pan offset.
    pub fn pan_offset(&self) -> (f32, f32) {
        self.pan_offset
    }

    /// Set the pan offset.
    pub fn set_pan_offset(&mut self, offset: (f32, f32)) {
        self.pan_offset = offset;
    }

    /// Get the zoom level.
    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    /// Set the zoom level.
    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom.clamp(0.1, 4.0);
    }

    /// Toggle the palette.
    pub fn toggle_palette(&mut self) {
        self.palette_open = !self.palette_open;
    }

    /// Check if palette is open.
    pub fn is_palette_open(&self) -> bool {
        self.palette_open
    }

    /// Set palette open state.
    pub fn set_palette_open(&mut self, open: bool) {
        self.palette_open = open;
    }

    /// Render the graph editor and return any actions.
    pub fn render<C: crate::egui_adapter::UIContext>(
        &mut self,
        ctx: &mut C,
        graph: &mut NodeGraph,
    ) -> GraphAction {
        let mut actions = Vec::new();

        // Render toolbar
        ctx.horizontal(|ctx| {
            if ctx.button("Add Node") {
                self.palette_open = true;
            }
            ctx.separator();
            if ctx.button("Zoom In") {
                self.zoom = (self.zoom * 1.1).min(4.0);
            }
            if ctx.button("Zoom Out") {
                self.zoom = (self.zoom / 1.1).max(0.1);
            }
            if ctx.button("Reset View") {
                self.pan_offset = (0.0, 0.0);
                self.zoom = 1.0;
            }
            ctx.separator();
            let mut show_grid = self.show_grid;
            if ctx.checkbox("Grid", &mut show_grid) {
                self.show_grid = show_grid;
            }
        });

        ctx.separator();

        // Render palette if open
        if self.palette_open {
            ctx.collapsing("Node Palette", |ctx| {
                ctx.text_edit("Search", &mut self.palette_search);
                ctx.separator();

                for category in &self.categories {
                    ctx.collapsing(&category.name, |ctx| {
                        for template in &category.templates {
                            let matches_search = self.palette_search.is_empty()
                                || template.name.to_lowercase().contains(&self.palette_search.to_lowercase());

                            if matches_search {
                                if ctx.button(&template.name) {
                                    let node_id = graph.add_node_at(template, self.pan_offset);
                                    actions.push(GraphAction::NodeAdded(node_id));
                                    self.selected_node = Some(node_id);
                                }
                            }
                        }
                    });
                }
            });
            ctx.separator();
        }

        // Render nodes
        ctx.collapsing("Nodes", |ctx| {
            let node_ids: Vec<NodeId> = graph.nodes.keys().copied().collect();

            for node_id in node_ids {
                if let Some(node) = graph.get_node(node_id) {
                    let is_selected = self.selected_node == Some(node_id);
                    let header = format!("{} [{}]", node.name, node_id);

                    ctx.collapsing(&header, |ctx| {
                        // Position
                        ctx.horizontal(|ctx| {
                            ctx.label(&format!("Position: ({:.1}, {:.1})", node.position.0, node.position.1));
                        });

                        // Inputs
                        if !node.inputs.is_empty() {
                            ctx.label("Inputs:");
                            for port in &node.inputs {
                                ctx.horizontal(|ctx| {
                                    ctx.label(&format!("  {} ({})", port.name, port.port_type.display_name()));
                                });
                            }
                        }

                        // Outputs
                        if !node.outputs.is_empty() {
                            ctx.label("Outputs:");
                            for port in &node.outputs {
                                ctx.horizontal(|ctx| {
                                    ctx.label(&format!("  {} ({})", port.name, port.port_type.display_name()));
                                });
                            }
                        }

                        // Properties
                        if !node.properties.is_empty() {
                            ctx.separator();
                            ctx.label("Properties:");
                        }
                    });

                    // Selection button
                    if ctx.button(if is_selected { "* Selected *" } else { "Select" }) {
                        if is_selected {
                            self.selected_node = None;
                            actions.push(GraphAction::SelectionCleared);
                        } else {
                            self.selected_node = Some(node_id);
                            actions.push(GraphAction::NodeSelected(node_id));
                        }
                    }

                    // Edit properties of selected node
                    if is_selected {
                        if let Some(node) = graph.get_node_mut(node_id) {
                            for prop in &mut node.properties {
                                let prop_name = prop.name.clone();
                                let changed = match &mut prop.value {
                                    PropertyValue::Bool(v) => ctx.checkbox(&prop_name, v),
                                    PropertyValue::Int(v) => {
                                        let mut f = *v as f32;
                                        let changed = ctx.slider(&prop_name, &mut f, -100.0, 100.0);
                                        *v = f as i32;
                                        changed
                                    }
                                    PropertyValue::Float(v) => ctx.slider(&prop_name, v, -100.0, 100.0),
                                    PropertyValue::String(v) => ctx.text_edit(&prop_name, v),
                                    PropertyValue::Vec3(_) => {
                                        ctx.label(&format!("{}: Vec3", prop_name));
                                        false
                                    }
                                    PropertyValue::Color(c) => ctx.color_edit(&prop_name, c),
                                    PropertyValue::Enum(idx) => {
                                        if let Some(options) = &prop.enum_options {
                                            let opts: Vec<&str> = options.iter().map(|s| s.as_str()).collect();
                                            ctx.combo(&prop_name, idx, &opts)
                                        } else {
                                            false
                                        }
                                    }
                                };
                                if changed {
                                    actions.push(GraphAction::PropertyChanged {
                                        node: node_id,
                                        property: prop_name,
                                        value: prop.value.clone(),
                                    });
                                }
                            }
                        }

                        ctx.spacing();
                        if ctx.button("Delete Node") {
                            graph.remove_node(node_id);
                            self.selected_node = None;
                            actions.push(GraphAction::NodeRemoved(node_id));
                        }
                    }

                    ctx.separator();
                }
            }
        });

        // Render connections
        ctx.collapsing("Connections", |ctx| {
            ctx.label(&format!("Total: {}", graph.connection_count()));
            for conn in graph.connections() {
                let from_name = graph.get_node(conn.from_node)
                    .and_then(|n| n.get_output(conn.from_port))
                    .map(|p| p.name.as_str())
                    .unwrap_or("?");
                let to_name = graph.get_node(conn.to_node)
                    .and_then(|n| n.get_input(conn.to_port))
                    .map(|p| p.name.as_str())
                    .unwrap_or("?");

                ctx.horizontal(|ctx| {
                    ctx.label(&format!(
                        "{}:{} -> {}:{}",
                        conn.from_node, from_name,
                        conn.to_node, to_name
                    ));
                });
            }
        });

        // Render validation errors
        let errors = graph.validate();
        if !errors.is_empty() {
            ctx.separator();
            ctx.collapsing("Validation Errors", |ctx| {
                for error in &errors {
                    ctx.label(&format!("- {}", error));
                }
            });
        }

        // Combine actions
        match actions.len() {
            0 => GraphAction::None,
            1 => actions.remove(0),
            _ => GraphAction::Batch(actions),
        }
    }

    /// Find a template by name in the palette.
    pub fn find_template(&self, name: &str) -> Option<&NodeTemplate> {
        for category in &self.categories {
            for template in &category.templates {
                if template.name == name {
                    return Some(template);
                }
            }
        }
        None
    }

    /// Snap a position to the grid.
    pub fn snap_to_grid(&self, pos: (f32, f32)) -> (f32, f32) {
        if self.show_grid && self.grid_size > 0.0 {
            (
                (pos.0 / self.grid_size).round() * self.grid_size,
                (pos.1 / self.grid_size).round() * self.grid_size,
            )
        } else {
            pos
        }
    }
}

// ---------------------------------------------------------------------------
// Standard Node Templates
// ---------------------------------------------------------------------------

/// Create standard math node templates.
pub fn math_category() -> NodeCategory {
    NodeCategory::new("Math")
        .with_template(
            NodeTemplate::new("Add", "Math")
                .with_input("A", PortType::Float)
                .with_input("B", PortType::Float)
                .with_output("Result", PortType::Float)
        )
        .with_template(
            NodeTemplate::new("Subtract", "Math")
                .with_input("A", PortType::Float)
                .with_input("B", PortType::Float)
                .with_output("Result", PortType::Float)
        )
        .with_template(
            NodeTemplate::new("Multiply", "Math")
                .with_input("A", PortType::Float)
                .with_input("B", PortType::Float)
                .with_output("Result", PortType::Float)
        )
        .with_template(
            NodeTemplate::new("Divide", "Math")
                .with_input("A", PortType::Float)
                .with_input("B", PortType::Float)
                .with_output("Result", PortType::Float)
        )
        .with_template(
            NodeTemplate::new("Abs", "Math")
                .with_input("Value", PortType::Float)
                .with_output("Result", PortType::Float)
        )
        .with_template(
            NodeTemplate::new("Sin", "Math")
                .with_input("Angle", PortType::Float)
                .with_output("Result", PortType::Float)
        )
        .with_template(
            NodeTemplate::new("Cos", "Math")
                .with_input("Angle", PortType::Float)
                .with_output("Result", PortType::Float)
        )
        .with_template(
            NodeTemplate::new("Lerp", "Math")
                .with_input("A", PortType::Float)
                .with_input("B", PortType::Float)
                .with_input("T", PortType::Float)
                .with_output("Result", PortType::Float)
        )
        .with_template(
            NodeTemplate::new("Clamp", "Math")
                .with_input("Value", PortType::Float)
                .with_input("Min", PortType::Float)
                .with_input("Max", PortType::Float)
                .with_output("Result", PortType::Float)
        )
        .with_template(
            NodeTemplate::new("Random", "Math")
                .with_output("Value", PortType::Float)
        )
}

/// Create standard logic node templates.
pub fn logic_category() -> NodeCategory {
    NodeCategory::new("Logic")
        .with_template(
            NodeTemplate::new("And", "Logic")
                .with_input("A", PortType::Bool)
                .with_input("B", PortType::Bool)
                .with_output("Result", PortType::Bool)
        )
        .with_template(
            NodeTemplate::new("Or", "Logic")
                .with_input("A", PortType::Bool)
                .with_input("B", PortType::Bool)
                .with_output("Result", PortType::Bool)
        )
        .with_template(
            NodeTemplate::new("Not", "Logic")
                .with_input("Value", PortType::Bool)
                .with_output("Result", PortType::Bool)
        )
        .with_template(
            NodeTemplate::new("Compare", "Logic")
                .with_input("A", PortType::Float)
                .with_input("B", PortType::Float)
                .with_output("Less", PortType::Bool)
                .with_output("Equal", PortType::Bool)
                .with_output("Greater", PortType::Bool)
                .with_property(PropertyTemplate::enum_prop(
                    "Mode",
                    vec!["<".to_string(), "<=".to_string(), "==".to_string(), ">=".to_string(), ">".to_string()],
                    2
                ))
        )
        .with_template(
            NodeTemplate::new("Select", "Logic")
                .with_input("Condition", PortType::Bool)
                .with_input("True", PortType::Any)
                .with_input("False", PortType::Any)
                .with_output("Result", PortType::Any)
        )
}

/// Create standard flow control node templates.
pub fn flow_category() -> NodeCategory {
    NodeCategory::new("Flow")
        .with_template(
            NodeTemplate::new("Sequence", "Flow")
                .with_flow_input()
                .with_output("1", PortType::Flow)
                .with_output("2", PortType::Flow)
                .with_output("3", PortType::Flow)
        )
        .with_template(
            NodeTemplate::new("Branch", "Flow")
                .with_flow_input()
                .with_input("Condition", PortType::Bool)
                .with_output("True", PortType::Flow)
                .with_output("False", PortType::Flow)
        )
        .with_template(
            NodeTemplate::new("ForLoop", "Flow")
                .with_flow_input()
                .with_input("Count", PortType::Int)
                .with_output("Loop", PortType::Flow)
                .with_output("Index", PortType::Int)
                .with_output("Completed", PortType::Flow)
        )
        .with_template(
            NodeTemplate::new("WhileLoop", "Flow")
                .with_flow_input()
                .with_input("Condition", PortType::Bool)
                .with_output("Loop", PortType::Flow)
                .with_output("Completed", PortType::Flow)
        )
        .with_template(
            NodeTemplate::new("DoOnce", "Flow")
                .with_flow_input()
                .with_output("Out", PortType::Flow)
                .with_property(PropertyTemplate::bool("Reset", false))
        )
        .with_template(
            NodeTemplate::new("Delay", "Flow")
                .with_flow_input()
                .with_input("Duration", PortType::Float)
                .with_flow_output()
        )
}

/// Create standard variable node templates.
pub fn variable_category() -> NodeCategory {
    NodeCategory::new("Variables")
        .with_template(
            NodeTemplate::new("GetFloat", "Variables")
                .with_output("Value", PortType::Float)
                .with_property(PropertyTemplate::string("Name", "MyFloat"))
        )
        .with_template(
            NodeTemplate::new("SetFloat", "Variables")
                .with_flow_input()
                .with_input("Value", PortType::Float)
                .with_flow_output()
                .with_property(PropertyTemplate::string("Name", "MyFloat"))
        )
        .with_template(
            NodeTemplate::new("GetBool", "Variables")
                .with_output("Value", PortType::Bool)
                .with_property(PropertyTemplate::string("Name", "MyBool"))
        )
        .with_template(
            NodeTemplate::new("SetBool", "Variables")
                .with_flow_input()
                .with_input("Value", PortType::Bool)
                .with_flow_output()
                .with_property(PropertyTemplate::string("Name", "MyBool"))
        )
        .with_template(
            NodeTemplate::new("GetVec3", "Variables")
                .with_output("Value", PortType::Vec3)
                .with_property(PropertyTemplate::string("Name", "MyVec3"))
        )
        .with_template(
            NodeTemplate::new("SetVec3", "Variables")
                .with_flow_input()
                .with_input("Value", PortType::Vec3)
                .with_flow_output()
                .with_property(PropertyTemplate::string("Name", "MyVec3"))
        )
}

/// Create standard event node templates.
pub fn event_category() -> NodeCategory {
    NodeCategory::new("Events")
        .with_template(
            NodeTemplate::new("OnStart", "Events")
                .with_flow_output()
        )
        .with_template(
            NodeTemplate::new("OnTick", "Events")
                .with_output("Out", PortType::Flow)
                .with_output("DeltaTime", PortType::Float)
        )
        .with_template(
            NodeTemplate::new("OnKeyPressed", "Events")
                .with_flow_output()
                .with_property(PropertyTemplate::string("Key", "Space"))
        )
        .with_template(
            NodeTemplate::new("OnCollision", "Events")
                .with_output("Out", PortType::Flow)
                .with_output("Other", PortType::Entity)
        )
        .with_template(
            NodeTemplate::new("CustomEvent", "Events")
                .with_flow_output()
                .with_property(PropertyTemplate::string("EventName", "MyEvent"))
        )
}

/// Create all standard node categories.
pub fn standard_categories() -> Vec<NodeCategory> {
    vec![
        math_category(),
        logic_category(),
        flow_category(),
        variable_category(),
        event_category(),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // PortType Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_port_type_compatibility_same() {
        assert!(PortType::Float.is_compatible_with(&PortType::Float));
        assert!(PortType::Int.is_compatible_with(&PortType::Int));
        assert!(PortType::Bool.is_compatible_with(&PortType::Bool));
        assert!(PortType::String.is_compatible_with(&PortType::String));
        assert!(PortType::Vec3.is_compatible_with(&PortType::Vec3));
        assert!(PortType::Entity.is_compatible_with(&PortType::Entity));
        assert!(PortType::Flow.is_compatible_with(&PortType::Flow));
    }

    #[test]
    fn test_port_type_compatibility_any() {
        assert!(PortType::Any.is_compatible_with(&PortType::Float));
        assert!(PortType::Any.is_compatible_with(&PortType::Bool));
        assert!(PortType::Any.is_compatible_with(&PortType::String));
        assert!(PortType::Float.is_compatible_with(&PortType::Any));
        assert!(PortType::Bool.is_compatible_with(&PortType::Any));
    }

    #[test]
    fn test_port_type_compatibility_numeric() {
        assert!(PortType::Int.is_compatible_with(&PortType::Float));
        assert!(PortType::Float.is_compatible_with(&PortType::Int));
    }

    #[test]
    fn test_port_type_incompatibility() {
        assert!(!PortType::Bool.is_compatible_with(&PortType::Float));
        assert!(!PortType::String.is_compatible_with(&PortType::Int));
        assert!(!PortType::Entity.is_compatible_with(&PortType::Vec3));
        assert!(!PortType::Flow.is_compatible_with(&PortType::Bool));
    }

    #[test]
    fn test_port_type_display_name() {
        assert_eq!(PortType::Flow.display_name(), "Flow");
        assert_eq!(PortType::Float.display_name(), "Float");
        assert_eq!(PortType::Any.display_name(), "Any");
    }

    #[test]
    fn test_port_type_color() {
        let color = PortType::Float.color();
        assert_eq!(color.len(), 4);
        assert!(color[3] > 0.0); // Alpha should be positive
    }

    // -------------------------------------------------------------------------
    // Port Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_port_creation() {
        let port = Port::new(1, "Value", PortType::Float, PortDirection::Input);
        assert_eq!(port.id, 1);
        assert_eq!(port.name, "Value");
        assert_eq!(port.port_type, PortType::Float);
        assert_eq!(port.direction, PortDirection::Input);
    }

    #[test]
    fn test_port_can_connect() {
        let output = Port::new(1, "Out", PortType::Float, PortDirection::Output);
        let input = Port::new(2, "In", PortType::Float, PortDirection::Input);
        assert!(output.can_connect_to(&input));
        assert!(input.can_connect_to(&output));
    }

    #[test]
    fn test_port_cannot_connect_same_direction() {
        let out1 = Port::new(1, "Out1", PortType::Float, PortDirection::Output);
        let out2 = Port::new(2, "Out2", PortType::Float, PortDirection::Output);
        assert!(!out1.can_connect_to(&out2));
    }

    #[test]
    fn test_port_cannot_connect_incompatible_types() {
        let output = Port::new(1, "Out", PortType::String, PortDirection::Output);
        let input = Port::new(2, "In", PortType::Bool, PortDirection::Input);
        assert!(!output.can_connect_to(&input));
    }

    // -------------------------------------------------------------------------
    // NodeProperty Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_property_bool() {
        let prop = NodeProperty::bool("Enabled", true);
        assert_eq!(prop.name, "Enabled");
        assert_eq!(prop.value, PropertyValue::Bool(true));
    }

    #[test]
    fn test_property_int() {
        let prop = NodeProperty::int("Count", 42);
        assert_eq!(prop.name, "Count");
        assert_eq!(prop.value, PropertyValue::Int(42));
    }

    #[test]
    fn test_property_float() {
        let prop = NodeProperty::float("Speed", 1.5);
        assert_eq!(prop.name, "Speed");
        assert_eq!(prop.value, PropertyValue::Float(1.5));
    }

    #[test]
    fn test_property_string() {
        let prop = NodeProperty::string("Name", "Player");
        assert_eq!(prop.name, "Name");
        assert_eq!(prop.value, PropertyValue::String("Player".to_string()));
    }

    #[test]
    fn test_property_vec3() {
        let prop = NodeProperty::vec3("Position", [1.0, 2.0, 3.0]);
        assert_eq!(prop.name, "Position");
        assert_eq!(prop.value, PropertyValue::Vec3([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_property_color() {
        let prop = NodeProperty::color("Tint", [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(prop.name, "Tint");
        assert_eq!(prop.value, PropertyValue::Color([1.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_property_enum() {
        let prop = NodeProperty::enum_prop("Mode", vec!["A".to_string(), "B".to_string()], 1);
        assert_eq!(prop.name, "Mode");
        assert_eq!(prop.value, PropertyValue::Enum(1));
        assert!(prop.enum_options.is_some());
    }

    // -------------------------------------------------------------------------
    // Node Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_node_get_ports() {
        let node = Node {
            id: 1,
            name: "Test".to_string(),
            category: "Test".to_string(),
            position: (0.0, 0.0),
            inputs: vec![Port::new(10, "In", PortType::Float, PortDirection::Input)],
            outputs: vec![Port::new(20, "Out", PortType::Float, PortDirection::Output)],
            properties: vec![],
        };

        assert!(node.get_input(10).is_some());
        assert!(node.get_input(20).is_none());
        assert!(node.get_output(20).is_some());
        assert!(node.get_output(10).is_none());
        assert!(node.get_port(10).is_some());
        assert!(node.get_port(20).is_some());
        assert!(node.get_port(30).is_none());
    }

    #[test]
    fn test_node_get_property() {
        let mut node = Node {
            id: 1,
            name: "Test".to_string(),
            category: "Test".to_string(),
            position: (0.0, 0.0),
            inputs: vec![],
            outputs: vec![],
            properties: vec![NodeProperty::float("Speed", 1.0)],
        };

        assert!(node.get_property("Speed").is_some());
        assert!(node.get_property("Missing").is_none());
        assert!(node.get_property_mut("Speed").is_some());
    }

    #[test]
    fn test_node_has_flow() {
        let node_with_flow = Node {
            id: 1,
            name: "Test".to_string(),
            category: "Test".to_string(),
            position: (0.0, 0.0),
            inputs: vec![Port::new(1, "In", PortType::Flow, PortDirection::Input)],
            outputs: vec![Port::new(2, "Out", PortType::Flow, PortDirection::Output)],
            properties: vec![],
        };

        assert!(node_with_flow.has_flow_input());
        assert!(node_with_flow.has_flow_output());

        let node_without_flow = Node {
            id: 2,
            name: "Math".to_string(),
            category: "Math".to_string(),
            position: (0.0, 0.0),
            inputs: vec![Port::new(3, "A", PortType::Float, PortDirection::Input)],
            outputs: vec![Port::new(4, "Result", PortType::Float, PortDirection::Output)],
            properties: vec![],
        };

        assert!(!node_without_flow.has_flow_input());
        assert!(!node_without_flow.has_flow_output());
    }

    // -------------------------------------------------------------------------
    // Connection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_connection_new() {
        let conn = Connection::new(1, 10, 2, 20);
        assert_eq!(conn.from_node, 1);
        assert_eq!(conn.from_port, 10);
        assert_eq!(conn.to_node, 2);
        assert_eq!(conn.to_port, 20);
    }

    #[test]
    fn test_connection_from_tuples() {
        let conn = Connection::from_tuples((1, 10), (2, 20));
        assert_eq!(conn.from_node, 1);
        assert_eq!(conn.from_port, 10);
        assert_eq!(conn.to_node, 2);
        assert_eq!(conn.to_port, 20);
    }

    // -------------------------------------------------------------------------
    // NodeTemplate Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_node_template_builder() {
        let template = NodeTemplate::new("Add", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float)
            .with_property(PropertyTemplate::bool("Saturate", false));

        assert_eq!(template.name, "Add");
        assert_eq!(template.category, "Math");
        assert_eq!(template.inputs.len(), 2);
        assert_eq!(template.outputs.len(), 1);
        assert_eq!(template.properties.len(), 1);
    }

    #[test]
    fn test_node_template_flow() {
        let template = NodeTemplate::new("Branch", "Flow")
            .with_flow_input()
            .with_flow_output();

        assert_eq!(template.inputs.len(), 1);
        assert_eq!(template.inputs[0].port_type, PortType::Flow);
        assert_eq!(template.outputs.len(), 1);
        assert_eq!(template.outputs[0].port_type, PortType::Flow);
    }

    // -------------------------------------------------------------------------
    // NodeCategory Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_node_category_builder() {
        let category = NodeCategory::new("Math")
            .with_template(NodeTemplate::new("Add", "Math"))
            .with_template(NodeTemplate::new("Sub", "Math"));

        assert_eq!(category.name, "Math");
        assert_eq!(category.templates.len(), 2);
    }

    // -------------------------------------------------------------------------
    // NodeGraph Basic Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_graph_new() {
        let graph = NodeGraph::new();
        assert!(graph.is_empty());
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.connection_count(), 0);
    }

    #[test]
    fn test_graph_add_node() {
        let mut graph = NodeGraph::new();
        let template = NodeTemplate::new("Add", "Math")
            .with_input("A", PortType::Float)
            .with_output("Result", PortType::Float);

        let node_id = graph.add_node(&template);
        assert_eq!(graph.node_count(), 1);

        let node = graph.get_node(node_id).unwrap();
        assert_eq!(node.name, "Add");
        assert_eq!(node.inputs.len(), 1);
        assert_eq!(node.outputs.len(), 1);
    }

    #[test]
    fn test_graph_add_node_at() {
        let mut graph = NodeGraph::new();
        let template = NodeTemplate::new("Add", "Math");

        let node_id = graph.add_node_at(&template, (100.0, 200.0));
        let node = graph.get_node(node_id).unwrap();
        assert_eq!(node.position, (100.0, 200.0));
    }

    #[test]
    fn test_graph_remove_node() {
        let mut graph = NodeGraph::new();
        let template = NodeTemplate::new("Add", "Math");
        let node_id = graph.add_node(&template);

        assert_eq!(graph.node_count(), 1);
        graph.remove_node(node_id);
        assert_eq!(graph.node_count(), 0);
        assert!(graph.get_node(node_id).is_none());
    }

    #[test]
    fn test_graph_remove_node_with_connections() {
        let mut graph = NodeGraph::new();
        let template1 = NodeTemplate::new("A", "Test")
            .with_output("Out", PortType::Float);
        let template2 = NodeTemplate::new("B", "Test")
            .with_input("In", PortType::Float);

        let node1 = graph.add_node(&template1);
        let node2 = graph.add_node(&template2);

        let out_port = graph.get_node(node1).unwrap().outputs[0].id;
        let in_port = graph.get_node(node2).unwrap().inputs[0].id;
        graph.connect((node1, out_port), (node2, in_port)).unwrap();

        assert_eq!(graph.connection_count(), 1);
        graph.remove_node(node1);
        assert_eq!(graph.connection_count(), 0);
    }

    // -------------------------------------------------------------------------
    // Connection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_graph_connect_success() {
        let mut graph = NodeGraph::new();
        let template1 = NodeTemplate::new("A", "Test")
            .with_output("Out", PortType::Float);
        let template2 = NodeTemplate::new("B", "Test")
            .with_input("In", PortType::Float);

        let node1 = graph.add_node(&template1);
        let node2 = graph.add_node(&template2);

        let out_port = graph.get_node(node1).unwrap().outputs[0].id;
        let in_port = graph.get_node(node2).unwrap().inputs[0].id;

        let result = graph.connect((node1, out_port), (node2, in_port));
        assert!(result.is_ok());
        assert_eq!(graph.connection_count(), 1);
    }

    #[test]
    fn test_graph_connect_self_connection() {
        let mut graph = NodeGraph::new();
        let template = NodeTemplate::new("A", "Test")
            .with_input("In", PortType::Float)
            .with_output("Out", PortType::Float);

        let node = graph.add_node(&template);
        let in_port = graph.get_node(node).unwrap().inputs[0].id;
        let out_port = graph.get_node(node).unwrap().outputs[0].id;

        let result = graph.connect((node, out_port), (node, in_port));
        assert!(matches!(result, Err(ConnectionError::SelfConnection(_))));
    }

    #[test]
    fn test_graph_connect_incompatible_types() {
        let mut graph = NodeGraph::new();
        let template1 = NodeTemplate::new("A", "Test")
            .with_output("Out", PortType::Bool);
        let template2 = NodeTemplate::new("B", "Test")
            .with_input("In", PortType::String);

        let node1 = graph.add_node(&template1);
        let node2 = graph.add_node(&template2);

        let out_port = graph.get_node(node1).unwrap().outputs[0].id;
        let in_port = graph.get_node(node2).unwrap().inputs[0].id;

        let result = graph.connect((node1, out_port), (node2, in_port));
        assert!(matches!(result, Err(ConnectionError::IncompatibleTypes(_, _))));
    }

    #[test]
    fn test_graph_connect_already_connected() {
        let mut graph = NodeGraph::new();
        let template1 = NodeTemplate::new("A", "Test")
            .with_output("Out", PortType::Float);
        let template2 = NodeTemplate::new("B", "Test")
            .with_input("In", PortType::Float);

        let node1 = graph.add_node(&template1);
        let node2 = graph.add_node(&template2);

        let out_port = graph.get_node(node1).unwrap().outputs[0].id;
        let in_port = graph.get_node(node2).unwrap().inputs[0].id;

        graph.connect((node1, out_port), (node2, in_port)).unwrap();
        let result = graph.connect((node1, out_port), (node2, in_port));
        assert!(matches!(result, Err(ConnectionError::AlreadyConnected)));
    }

    #[test]
    fn test_graph_connect_input_already_connected() {
        let mut graph = NodeGraph::new();
        let template1 = NodeTemplate::new("A", "Test")
            .with_output("Out", PortType::Float);
        let template2 = NodeTemplate::new("B", "Test")
            .with_output("Out", PortType::Float);
        let template3 = NodeTemplate::new("C", "Test")
            .with_input("In", PortType::Float);

        let node1 = graph.add_node(&template1);
        let node2 = graph.add_node(&template2);
        let node3 = graph.add_node(&template3);

        let out_port1 = graph.get_node(node1).unwrap().outputs[0].id;
        let out_port2 = graph.get_node(node2).unwrap().outputs[0].id;
        let in_port = graph.get_node(node3).unwrap().inputs[0].id;

        graph.connect((node1, out_port1), (node3, in_port)).unwrap();
        let result = graph.connect((node2, out_port2), (node3, in_port));
        assert!(matches!(result, Err(ConnectionError::InputAlreadyConnected(_, _))));
    }

    #[test]
    fn test_graph_disconnect() {
        let mut graph = NodeGraph::new();
        let template1 = NodeTemplate::new("A", "Test")
            .with_output("Out", PortType::Float);
        let template2 = NodeTemplate::new("B", "Test")
            .with_input("In", PortType::Float);

        let node1 = graph.add_node(&template1);
        let node2 = graph.add_node(&template2);

        let out_port = graph.get_node(node1).unwrap().outputs[0].id;
        let in_port = graph.get_node(node2).unwrap().inputs[0].id;

        graph.connect((node1, out_port), (node2, in_port)).unwrap();
        assert_eq!(graph.connection_count(), 1);

        graph.disconnect((node1, out_port), (node2, in_port));
        assert_eq!(graph.connection_count(), 0);
    }

    // -------------------------------------------------------------------------
    // Cycle Detection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_graph_no_cycle() {
        let mut graph = NodeGraph::new();
        let template = NodeTemplate::new("Node", "Test")
            .with_input("In", PortType::Float)
            .with_output("Out", PortType::Float);

        let node1 = graph.add_node(&template);
        let node2 = graph.add_node(&template);
        let node3 = graph.add_node(&template);

        // Linear chain: 1 -> 2 -> 3
        let out1 = graph.get_node(node1).unwrap().outputs[0].id;
        let in2 = graph.get_node(node2).unwrap().inputs[0].id;
        let out2 = graph.get_node(node2).unwrap().outputs[0].id;
        let in3 = graph.get_node(node3).unwrap().inputs[0].id;

        graph.connect((node1, out1), (node2, in2)).unwrap();
        graph.connect((node2, out2), (node3, in3)).unwrap();

        let sorted = graph.topological_sort();
        assert!(sorted.is_ok());
        let sorted = sorted.unwrap();
        assert_eq!(sorted.len(), 3);
        // node1 should come before node2, node2 before node3
        let pos1 = sorted.iter().position(|&id| id == node1).unwrap();
        let pos2 = sorted.iter().position(|&id| id == node2).unwrap();
        let pos3 = sorted.iter().position(|&id| id == node3).unwrap();
        assert!(pos1 < pos2);
        assert!(pos2 < pos3);
    }

    #[test]
    fn test_graph_cycle_detection_prevents_connection() {
        let mut graph = NodeGraph::new();
        let template = NodeTemplate::new("Node", "Test")
            .with_input("In", PortType::Float)
            .with_output("Out", PortType::Float);

        let node1 = graph.add_node(&template);
        let node2 = graph.add_node(&template);

        // Create 1 -> 2
        let out1 = graph.get_node(node1).unwrap().outputs[0].id;
        let in2 = graph.get_node(node2).unwrap().inputs[0].id;
        graph.connect((node1, out1), (node2, in2)).unwrap();

        // Try to create 2 -> 1 (would create cycle)
        let out2 = graph.get_node(node2).unwrap().outputs[0].id;
        let in1 = graph.get_node(node1).unwrap().inputs[0].id;
        let result = graph.connect((node2, out2), (node1, in1));
        assert!(matches!(result, Err(ConnectionError::WouldCreateCycle)));
    }

    // -------------------------------------------------------------------------
    // Topological Sort Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_topological_sort_empty() {
        let graph = NodeGraph::new();
        let sorted = graph.topological_sort().unwrap();
        assert!(sorted.is_empty());
    }

    #[test]
    fn test_topological_sort_single_node() {
        let mut graph = NodeGraph::new();
        let template = NodeTemplate::new("Single", "Test");
        let node_id = graph.add_node(&template);

        let sorted = graph.topological_sort().unwrap();
        assert_eq!(sorted, vec![node_id]);
    }

    #[test]
    fn test_topological_sort_disconnected() {
        let mut graph = NodeGraph::new();
        let template = NodeTemplate::new("Node", "Test");

        let node1 = graph.add_node(&template);
        let node2 = graph.add_node(&template);

        let sorted = graph.topological_sort().unwrap();
        assert_eq!(sorted.len(), 2);
        assert!(sorted.contains(&node1));
        assert!(sorted.contains(&node2));
    }

    // -------------------------------------------------------------------------
    // Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_empty_graph() {
        let graph = NodeGraph::new();
        let errors = graph.validate();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_simple_graph() {
        let mut graph = NodeGraph::new();
        let template1 = NodeTemplate::new("A", "Test")
            .with_output("Out", PortType::Float);
        let template2 = NodeTemplate::new("B", "Test")
            .with_input("In", PortType::Float);

        let node1 = graph.add_node(&template1);
        let node2 = graph.add_node(&template2);

        let out_port = graph.get_node(node1).unwrap().outputs[0].id;
        let in_port = graph.get_node(node2).unwrap().inputs[0].id;
        graph.connect((node1, out_port), (node2, in_port)).unwrap();

        let errors = graph.validate();
        assert!(errors.is_empty());
    }

    // -------------------------------------------------------------------------
    // Connection Query Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_connections_to() {
        let mut graph = NodeGraph::new();
        let template1 = NodeTemplate::new("A", "Test")
            .with_output("Out", PortType::Float);
        let template2 = NodeTemplate::new("B", "Test")
            .with_input("In", PortType::Float);

        let node1 = graph.add_node(&template1);
        let node2 = graph.add_node(&template2);

        let out_port = graph.get_node(node1).unwrap().outputs[0].id;
        let in_port = graph.get_node(node2).unwrap().inputs[0].id;
        graph.connect((node1, out_port), (node2, in_port)).unwrap();

        let conns = graph.connections_to(node2, in_port);
        assert_eq!(conns.len(), 1);
        assert_eq!(conns[0].from_node, node1);
    }

    #[test]
    fn test_connections_from() {
        let mut graph = NodeGraph::new();
        let template1 = NodeTemplate::new("A", "Test")
            .with_output("Out", PortType::Float);
        let template2 = NodeTemplate::new("B", "Test")
            .with_input("In", PortType::Float);

        let node1 = graph.add_node(&template1);
        let node2 = graph.add_node(&template2);

        let out_port = graph.get_node(node1).unwrap().outputs[0].id;
        let in_port = graph.get_node(node2).unwrap().inputs[0].id;
        graph.connect((node1, out_port), (node2, in_port)).unwrap();

        let conns = graph.connections_from(node1, out_port);
        assert_eq!(conns.len(), 1);
        assert_eq!(conns[0].to_node, node2);
    }

    #[test]
    fn test_connections_for_node() {
        let mut graph = NodeGraph::new();
        let template = NodeTemplate::new("Node", "Test")
            .with_input("In", PortType::Float)
            .with_output("Out", PortType::Float);

        let node1 = graph.add_node(&template);
        let node2 = graph.add_node(&template);
        let node3 = graph.add_node(&template);

        // 1 -> 2 -> 3
        let out1 = graph.get_node(node1).unwrap().outputs[0].id;
        let in2 = graph.get_node(node2).unwrap().inputs[0].id;
        let out2 = graph.get_node(node2).unwrap().outputs[0].id;
        let in3 = graph.get_node(node3).unwrap().inputs[0].id;

        graph.connect((node1, out1), (node2, in2)).unwrap();
        graph.connect((node2, out2), (node3, in3)).unwrap();

        let conns = graph.connections_for_node(node2);
        assert_eq!(conns.len(), 2); // One incoming, one outgoing
    }

    // -------------------------------------------------------------------------
    // Serialization Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_graph_serialization() {
        let mut graph = NodeGraph::new();
        let template = NodeTemplate::new("Add", "Math")
            .with_input("A", PortType::Float)
            .with_output("Result", PortType::Float);

        graph.add_node_at(&template, (100.0, 200.0));

        let json = graph.to_json().unwrap();
        let parsed = NodeGraph::from_json(&json).unwrap();

        assert_eq!(parsed.node_count(), 1);
        let node = parsed.nodes().next().unwrap();
        assert_eq!(node.name, "Add");
        assert_eq!(node.position, (100.0, 200.0));
    }

    // -------------------------------------------------------------------------
    // GraphEditor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_graph_editor_new() {
        let editor = GraphEditor::new();
        assert!(editor.selected_node().is_none());
        assert_eq!(editor.zoom(), 1.0);
        assert_eq!(editor.pan_offset(), (0.0, 0.0));
        assert!(!editor.is_palette_open());
    }

    #[test]
    fn test_graph_editor_set_palette() {
        let mut editor = GraphEditor::new();
        editor.set_palette(standard_categories());
        assert!(!editor.categories().is_empty());
    }

    #[test]
    fn test_graph_editor_find_template() {
        let mut editor = GraphEditor::new();
        editor.set_palette(standard_categories());

        assert!(editor.find_template("Add").is_some());
        assert!(editor.find_template("Branch").is_some());
        assert!(editor.find_template("NonExistent").is_none());
    }

    #[test]
    fn test_graph_editor_selection() {
        let mut editor = GraphEditor::new();
        assert!(editor.selected_node().is_none());

        editor.select_node(Some(42));
        assert_eq!(editor.selected_node(), Some(42));

        editor.select_node(None);
        assert!(editor.selected_node().is_none());
    }

    #[test]
    fn test_graph_editor_zoom() {
        let mut editor = GraphEditor::new();
        assert_eq!(editor.zoom(), 1.0);

        editor.set_zoom(2.0);
        assert_eq!(editor.zoom(), 2.0);

        // Clamp to max
        editor.set_zoom(10.0);
        assert_eq!(editor.zoom(), 4.0);

        // Clamp to min
        editor.set_zoom(0.01);
        assert_eq!(editor.zoom(), 0.1);
    }

    #[test]
    fn test_graph_editor_pan() {
        let mut editor = GraphEditor::new();
        assert_eq!(editor.pan_offset(), (0.0, 0.0));

        editor.set_pan_offset((100.0, -50.0));
        assert_eq!(editor.pan_offset(), (100.0, -50.0));
    }

    #[test]
    fn test_graph_editor_palette() {
        let mut editor = GraphEditor::new();
        assert!(!editor.is_palette_open());

        editor.toggle_palette();
        assert!(editor.is_palette_open());

        editor.toggle_palette();
        assert!(!editor.is_palette_open());

        editor.set_palette_open(true);
        assert!(editor.is_palette_open());
    }

    #[test]
    fn test_graph_editor_snap_to_grid() {
        let mut editor = GraphEditor::new();
        editor.show_grid = true;
        editor.grid_size = 20.0;

        assert_eq!(editor.snap_to_grid((15.0, 25.0)), (20.0, 20.0));
        assert_eq!(editor.snap_to_grid((10.0, 10.0)), (20.0, 20.0));
        assert_eq!(editor.snap_to_grid((5.0, 5.0)), (0.0, 0.0));

        editor.show_grid = false;
        assert_eq!(editor.snap_to_grid((15.0, 25.0)), (15.0, 25.0));
    }

    // -------------------------------------------------------------------------
    // GraphAction Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_graph_action_none() {
        let action = GraphAction::None;
        assert!(action.is_none());
        assert!(!action.is_some());
    }

    #[test]
    fn test_graph_action_node_added() {
        let action = GraphAction::NodeAdded(42);
        assert!(!action.is_none());
        assert!(action.is_some());
    }

    // -------------------------------------------------------------------------
    // Standard Categories Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_standard_categories() {
        let categories = standard_categories();
        assert_eq!(categories.len(), 5);

        // Check math category
        let math = categories.iter().find(|c| c.name == "Math");
        assert!(math.is_some());
        assert!(math.unwrap().templates.len() >= 5);

        // Check logic category
        let logic = categories.iter().find(|c| c.name == "Logic");
        assert!(logic.is_some());

        // Check flow category
        let flow = categories.iter().find(|c| c.name == "Flow");
        assert!(flow.is_some());
    }

    #[test]
    fn test_math_category() {
        let category = math_category();
        assert_eq!(category.name, "Math");

        let add = category.templates.iter().find(|t| t.name == "Add");
        assert!(add.is_some());
        let add = add.unwrap();
        assert_eq!(add.inputs.len(), 2);
        assert_eq!(add.outputs.len(), 1);
    }

    #[test]
    fn test_logic_category() {
        let category = logic_category();
        let branch = category.templates.iter().find(|t| t.name == "Select");
        assert!(branch.is_some());
    }

    #[test]
    fn test_flow_category() {
        let category = flow_category();
        let seq = category.templates.iter().find(|t| t.name == "Sequence");
        assert!(seq.is_some());

        let branch = category.templates.iter().find(|t| t.name == "Branch");
        assert!(branch.is_some());
        let branch = branch.unwrap();
        // Branch should have flow input and condition input
        assert!(branch.inputs.iter().any(|p| p.port_type == PortType::Flow));
        assert!(branch.inputs.iter().any(|p| p.port_type == PortType::Bool));
    }

    #[test]
    fn test_variable_category() {
        let category = variable_category();
        let get_float = category.templates.iter().find(|t| t.name == "GetFloat");
        assert!(get_float.is_some());
        let get_float = get_float.unwrap();
        assert!(get_float.properties.len() > 0);
    }

    #[test]
    fn test_event_category() {
        let category = event_category();
        let on_start = category.templates.iter().find(|t| t.name == "OnStart");
        assert!(on_start.is_some());
        let on_start = on_start.unwrap();
        assert!(on_start.outputs.iter().any(|p| p.port_type == PortType::Flow));
    }
}
