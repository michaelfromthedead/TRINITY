//! FlowForge Sub-Graph Macros System (T-TL-8.4)
//!
//! This module provides reusable sub-graphs as macro nodes for FlowForge.
//! It includes:
//! - MacroDefinition for defining reusable node compositions
//! - MacroInstance for embedding macros in graphs
//! - MacroLibrary for macro management
//! - Standard macro library with utility, math, and logic macros
//!
//! # Architecture
//!
//! ```text
//! MacroLibrary                MacroDefinition               MacroInstance
//! ============                ===============               =============
//! macros: HashMap<>  --->    inner_graph: NodeGraph   --->  definition_id
//! categories: Vec<>          inputs: Vec<MacroPort>         input_mappings
//!                            outputs: Vec<MacroPort>        output_mappings
//!
//! Workflow:
//! 1. Define macro with inner_graph
//! 2. Register in library
//! 3. Create instance in target graph
//! 4. expand() to inline nodes or execute as unit
//! 5. collapse() to create macro from selected nodes
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use renderer_backend::flowforge_macros::{MacroLibrary, MacroDefinition, MacroPort};
//! use renderer_backend::flowforge::{NodeGraph, PortType};
//!
//! let mut library = MacroLibrary::new();
//!
//! // Create a Clamp01 macro (clamps to 0-1 range)
//! let clamp_macro = MacroDefinition::new("Clamp01", "Utility")
//!     .with_description("Clamps a value to the 0-1 range")
//!     .with_input("Value", PortType::Float)
//!     .with_output("Result", PortType::Float);
//!
//! library.register(clamp_macro);
//!
//! // Create instance in target graph
//! let instance = library.create_instance("Clamp01").unwrap();
//! ```

use crate::flowforge::{
    Connection, ConnectionError, Node, NodeGraph, NodeId, NodeTemplate, Port, PortDirection,
    PortId, PortType, PropertyTemplate,
};
use crate::flowforge_nodes::Value;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Error types for macro operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MacroError {
    /// Macro not found in library.
    MacroNotFound(String),
    /// Invalid macro definition.
    InvalidDefinition(String),
    /// Node not found in graph.
    NodeNotFound(NodeId),
    /// Port not found.
    PortNotFound(PortId),
    /// Invalid connection during expansion.
    ConnectionError(String),
    /// Graph cycle would be created.
    CycleDetected,
    /// Cannot collapse selected nodes into macro.
    CannotCollapse(String),
    /// Version mismatch.
    VersionMismatch { expected: u32, found: u32 },
}

impl std::fmt::Display for MacroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MacroError::MacroNotFound(id) => write!(f, "Macro '{}' not found", id),
            MacroError::InvalidDefinition(msg) => write!(f, "Invalid macro definition: {}", msg),
            MacroError::NodeNotFound(id) => write!(f, "Node {} not found", id),
            MacroError::PortNotFound(id) => write!(f, "Port {} not found", id),
            MacroError::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            MacroError::CycleDetected => write!(f, "Cycle detected in macro graph"),
            MacroError::CannotCollapse(msg) => write!(f, "Cannot collapse to macro: {}", msg),
            MacroError::VersionMismatch { expected, found } => {
                write!(f, "Version mismatch: expected {}, found {}", expected, found)
            }
        }
    }
}

impl std::error::Error for MacroError {}

// ---------------------------------------------------------------------------
// Macro Port
// ---------------------------------------------------------------------------

/// A port definition for a macro's external interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroPort {
    /// Port name.
    pub name: String,
    /// Port data type.
    pub port_type: PortType,
    /// Default value if not connected.
    pub default_value: Option<Value>,
    /// Description for documentation.
    pub description: String,
}

impl MacroPort {
    /// Create a new macro port.
    pub fn new(name: impl Into<String>, port_type: PortType) -> Self {
        Self {
            name: name.into(),
            port_type,
            default_value: None,
            description: String::new(),
        }
    }

    /// Set the default value.
    pub fn with_default(mut self, value: Value) -> Self {
        self.default_value = Some(value);
        self
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

// ---------------------------------------------------------------------------
// Macro Definition
// ---------------------------------------------------------------------------

/// Definition of a reusable sub-graph macro.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroDefinition {
    /// Unique identifier for the macro.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Description of what the macro does.
    pub description: String,
    /// Category for organization (e.g., "Utility", "Math", "Logic").
    pub category: String,
    /// The internal node graph that implements the macro.
    pub inner_graph: NodeGraph,
    /// Input ports exposed to the outer graph.
    pub inputs: Vec<MacroPort>,
    /// Output ports exposed to the outer graph.
    pub outputs: Vec<MacroPort>,
    /// Version number for compatibility tracking.
    pub version: u32,
    /// Internal node IDs that map to input ports (index corresponds to inputs vec).
    input_node_ids: Vec<NodeId>,
    /// Internal node IDs that map to output ports (index corresponds to outputs vec).
    output_node_ids: Vec<NodeId>,
}

impl MacroDefinition {
    /// Create a new macro definition.
    pub fn new(name: impl Into<String>, category: impl Into<String>) -> Self {
        let name_str = name.into();
        Self {
            id: name_str.to_lowercase().replace(' ', "_"),
            name: name_str,
            description: String::new(),
            category: category.into(),
            inner_graph: NodeGraph::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            version: 1,
            input_node_ids: Vec::new(),
            output_node_ids: Vec::new(),
        }
    }

    /// Set the macro ID explicitly.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add an input port.
    pub fn with_input(mut self, name: impl Into<String>, port_type: PortType) -> Self {
        self.inputs.push(MacroPort::new(name, port_type));
        self
    }

    /// Add an input port with default value.
    pub fn with_input_default(
        mut self,
        name: impl Into<String>,
        port_type: PortType,
        default: Value,
    ) -> Self {
        self.inputs
            .push(MacroPort::new(name, port_type).with_default(default));
        self
    }

    /// Add an output port.
    pub fn with_output(mut self, name: impl Into<String>, port_type: PortType) -> Self {
        self.outputs.push(MacroPort::new(name, port_type));
        self
    }

    /// Set the version.
    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Set the inner graph.
    pub fn with_inner_graph(mut self, graph: NodeGraph) -> Self {
        self.inner_graph = graph;
        self
    }

    /// Get the number of input ports.
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Get the number of output ports.
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Validate the macro definition.
    pub fn validate(&self) -> Result<(), MacroError> {
        if self.id.is_empty() {
            return Err(MacroError::InvalidDefinition("ID cannot be empty".into()));
        }
        if self.name.is_empty() {
            return Err(MacroError::InvalidDefinition("Name cannot be empty".into()));
        }

        // Check for cycles in inner graph
        if self.inner_graph.topological_sort().is_err() {
            return Err(MacroError::CycleDetected);
        }

        Ok(())
    }

    /// Convert to a NodeTemplate for use in graph editors.
    pub fn to_template(&self) -> NodeTemplate {
        let mut template = NodeTemplate::new(&self.name, format!("Macro/{}", self.category));

        for input in &self.inputs {
            template = template.with_input(&input.name, input.port_type);
        }

        for output in &self.outputs {
            template = template.with_output(&output.name, output.port_type);
        }

        template
    }
}

// ---------------------------------------------------------------------------
// Macro Instance
// ---------------------------------------------------------------------------

/// An instance of a macro embedded in a graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroInstance {
    /// ID of the macro definition this is an instance of.
    pub definition_id: String,
    /// Node ID of this instance in the containing graph.
    pub instance_id: NodeId,
    /// Mappings from macro input ports to external connections.
    pub input_mappings: HashMap<PortId, Connection>,
    /// Mappings from macro output ports to external connections.
    pub output_mappings: HashMap<PortId, Connection>,
    /// Version of the macro this instance was created from.
    pub version: u32,
}

impl MacroInstance {
    /// Create a new macro instance.
    pub fn new(definition_id: impl Into<String>, instance_id: NodeId) -> Self {
        Self {
            definition_id: definition_id.into(),
            instance_id,
            input_mappings: HashMap::new(),
            output_mappings: HashMap::new(),
            version: 1,
        }
    }

    /// Set the version.
    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Map an input port to a connection.
    pub fn map_input(&mut self, port_id: PortId, connection: Connection) {
        self.input_mappings.insert(port_id, connection);
    }

    /// Map an output port to a connection.
    pub fn map_output(&mut self, port_id: PortId, connection: Connection) {
        self.output_mappings.insert(port_id, connection);
    }

    /// Get input mapping for a port.
    pub fn get_input_mapping(&self, port_id: PortId) -> Option<&Connection> {
        self.input_mappings.get(&port_id)
    }

    /// Get output mapping for a port.
    pub fn get_output_mapping(&self, port_id: PortId) -> Option<&Connection> {
        self.output_mappings.get(&port_id)
    }

    /// Check if all required inputs are mapped.
    pub fn inputs_complete(&self, definition: &MacroDefinition) -> bool {
        // All inputs without defaults must be mapped
        for (i, input) in definition.inputs.iter().enumerate() {
            let port_id = i as PortId + 1; // Simplified port ID mapping
            if input.default_value.is_none() && !self.input_mappings.contains_key(&port_id) {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Macro Library
// ---------------------------------------------------------------------------

/// Library of macro definitions for management and lookup.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MacroLibrary {
    /// Registered macros by ID.
    macros: HashMap<String, MacroDefinition>,
    /// Available categories.
    categories: Vec<String>,
}

impl MacroLibrary {
    /// Create a new empty library.
    pub fn new() -> Self {
        Self {
            macros: HashMap::new(),
            categories: Vec::new(),
        }
    }

    /// Create a library with standard macros.
    pub fn with_standard_macros() -> Self {
        let mut library = Self::new();

        // Utility macros
        for macro_def in utility_macros() {
            library.register(macro_def);
        }

        // Math macros
        for macro_def in math_macros() {
            library.register(macro_def);
        }

        // Logic macros
        for macro_def in logic_macros() {
            library.register(macro_def);
        }

        library
    }

    /// Register a macro definition.
    pub fn register(&mut self, macro_def: MacroDefinition) {
        // Add category if new
        if !self.categories.contains(&macro_def.category) {
            self.categories.push(macro_def.category.clone());
        }
        self.macros.insert(macro_def.id.clone(), macro_def);
    }

    /// Unregister a macro by ID.
    pub fn unregister(&mut self, id: &str) -> Option<MacroDefinition> {
        self.macros.remove(id)
    }

    /// Get a macro by ID.
    pub fn get(&self, id: &str) -> Option<&MacroDefinition> {
        self.macros.get(id)
    }

    /// Get a mutable reference to a macro by ID.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut MacroDefinition> {
        self.macros.get_mut(id)
    }

    /// Check if a macro exists.
    pub fn contains(&self, id: &str) -> bool {
        self.macros.contains_key(id)
    }

    /// List all macros in a category.
    pub fn list_by_category(&self, category: &str) -> Vec<&MacroDefinition> {
        self.macros
            .values()
            .filter(|m| m.category == category)
            .collect()
    }

    /// List all categories.
    pub fn categories(&self) -> &[String] {
        &self.categories
    }

    /// Get all macro IDs.
    pub fn macro_ids(&self) -> Vec<&str> {
        self.macros.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of registered macros.
    pub fn len(&self) -> usize {
        self.macros.len()
    }

    /// Check if the library is empty.
    pub fn is_empty(&self) -> bool {
        self.macros.is_empty()
    }

    /// Create an instance of a macro.
    pub fn create_instance(&self, macro_id: &str) -> Option<MacroInstance> {
        let macro_def = self.macros.get(macro_id)?;
        Some(MacroInstance::new(macro_id, 0).with_version(macro_def.version))
    }

    /// Expand a macro instance into a target graph.
    ///
    /// This inlines all nodes from the macro's inner graph into the target graph,
    /// remapping node IDs and connections.
    pub fn expand(
        &self,
        instance: &MacroInstance,
        graph: &mut NodeGraph,
    ) -> Result<Vec<NodeId>, MacroError> {
        let macro_def = self
            .macros
            .get(&instance.definition_id)
            .ok_or_else(|| MacroError::MacroNotFound(instance.definition_id.clone()))?;

        // Check version compatibility
        if instance.version != macro_def.version {
            return Err(MacroError::VersionMismatch {
                expected: macro_def.version,
                found: instance.version,
            });
        }

        // Clone nodes from inner graph into target graph
        let mut node_id_map: HashMap<NodeId, NodeId> = HashMap::new();
        let mut port_id_map: HashMap<PortId, PortId> = HashMap::new();
        let mut new_node_ids = Vec::new();

        for node in macro_def.inner_graph.nodes() {
            // Create template from node
            let template = node_to_template(node);
            let new_id = graph.add_node_at(&template, node.position);

            node_id_map.insert(node.id, new_id);
            new_node_ids.push(new_id);

            // Map port IDs
            if let Some(new_node) = graph.get_node(new_id) {
                for (old_port, new_port) in node.inputs.iter().zip(new_node.inputs.iter()) {
                    port_id_map.insert(old_port.id, new_port.id);
                }
                for (old_port, new_port) in node.outputs.iter().zip(new_node.outputs.iter()) {
                    port_id_map.insert(old_port.id, new_port.id);
                }
            }
        }

        // Recreate connections with new IDs
        for conn in macro_def.inner_graph.connections() {
            let new_from_node = *node_id_map.get(&conn.from_node).unwrap();
            let new_from_port = *port_id_map.get(&conn.from_port).unwrap();
            let new_to_node = *node_id_map.get(&conn.to_node).unwrap();
            let new_to_port = *port_id_map.get(&conn.to_port).unwrap();

            if let Err(e) = graph.connect((new_from_node, new_from_port), (new_to_node, new_to_port))
            {
                return Err(MacroError::ConnectionError(format!("{}", e)));
            }
        }

        Ok(new_node_ids)
    }

    /// Collapse selected nodes into a new macro definition.
    ///
    /// The nodes must form a connected subgraph with clear input and output boundaries.
    pub fn collapse(
        &self,
        nodes: &[NodeId],
        graph: &NodeGraph,
    ) -> Result<MacroDefinition, MacroError> {
        if nodes.is_empty() {
            return Err(MacroError::CannotCollapse(
                "No nodes selected".to_string(),
            ));
        }

        let node_set: std::collections::HashSet<NodeId> = nodes.iter().copied().collect();

        // Create new inner graph
        let mut inner_graph = NodeGraph::new();
        let mut node_id_map: HashMap<NodeId, NodeId> = HashMap::new();
        let mut port_id_map: HashMap<PortId, PortId> = HashMap::new();

        // Copy selected nodes
        for &node_id in nodes {
            let node = graph
                .get_node(node_id)
                .ok_or(MacroError::NodeNotFound(node_id))?;

            let template = node_to_template(node);
            let new_id = inner_graph.add_node_at(&template, node.position);
            node_id_map.insert(node_id, new_id);

            // Map port IDs
            if let Some(new_node) = inner_graph.get_node(new_id) {
                for (old_port, new_port) in node.inputs.iter().zip(new_node.inputs.iter()) {
                    port_id_map.insert(old_port.id, new_port.id);
                }
                for (old_port, new_port) in node.outputs.iter().zip(new_node.outputs.iter()) {
                    port_id_map.insert(old_port.id, new_port.id);
                }
            }
        }

        // Copy internal connections (connections where both nodes are in selection)
        for conn in graph.connections() {
            if node_set.contains(&conn.from_node) && node_set.contains(&conn.to_node) {
                let new_from_node = *node_id_map.get(&conn.from_node).unwrap();
                let new_from_port = *port_id_map.get(&conn.from_port).unwrap();
                let new_to_node = *node_id_map.get(&conn.to_node).unwrap();
                let new_to_port = *port_id_map.get(&conn.to_port).unwrap();

                let _ =
                    inner_graph.connect((new_from_node, new_from_port), (new_to_node, new_to_port));
            }
        }

        // Identify input ports (ports that receive connections from outside selection)
        let mut inputs = Vec::new();
        for conn in graph.connections() {
            if !node_set.contains(&conn.from_node) && node_set.contains(&conn.to_node) {
                if let Some(node) = graph.get_node(conn.to_node) {
                    if let Some(port) = node.get_input(conn.to_port) {
                        inputs.push(MacroPort::new(&port.name, port.port_type));
                    }
                }
            }
        }

        // Identify output ports (ports that send connections to outside selection)
        let mut outputs = Vec::new();
        for conn in graph.connections() {
            if node_set.contains(&conn.from_node) && !node_set.contains(&conn.to_node) {
                if let Some(node) = graph.get_node(conn.from_node) {
                    if let Some(port) = node.get_output(conn.from_port) {
                        outputs.push(MacroPort::new(&port.name, port.port_type));
                    }
                }
            }
        }

        // Also add unconnected inputs/outputs from boundary nodes
        for &node_id in nodes {
            if let Some(node) = graph.get_node(node_id) {
                // Check for unconnected inputs
                for port in &node.inputs {
                    let has_internal_connection = graph
                        .connections()
                        .iter()
                        .any(|c| c.to_node == node_id && c.to_port == port.id);
                    if !has_internal_connection
                        && !inputs.iter().any(|p| p.name == port.name)
                        && port.port_type != PortType::Flow
                    {
                        inputs.push(MacroPort::new(&port.name, port.port_type));
                    }
                }

                // Check for unconnected outputs
                for port in &node.outputs {
                    let has_internal_connection = graph
                        .connections()
                        .iter()
                        .any(|c| c.from_node == node_id && c.from_port == port.id);
                    if !has_internal_connection
                        && !outputs.iter().any(|p| p.name == port.name)
                        && port.port_type != PortType::Flow
                    {
                        outputs.push(MacroPort::new(&port.name, port.port_type));
                    }
                }
            }
        }

        let macro_def = MacroDefinition {
            id: format!("collapsed_macro_{}", nodes.len()),
            name: format!("Collapsed ({})", nodes.len()),
            description: format!("Macro collapsed from {} nodes", nodes.len()),
            category: "Custom".to_string(),
            inner_graph,
            inputs,
            outputs,
            version: 1,
            input_node_ids: Vec::new(),
            output_node_ids: Vec::new(),
        };

        macro_def.validate()?;
        Ok(macro_def)
    }

    /// Find macros by name (partial match).
    pub fn search(&self, query: &str) -> Vec<&MacroDefinition> {
        let query_lower = query.to_lowercase();
        self.macros
            .values()
            .filter(|m| {
                m.name.to_lowercase().contains(&query_lower)
                    || m.description.to_lowercase().contains(&query_lower)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Convert a Node to a NodeTemplate.
fn node_to_template(node: &Node) -> NodeTemplate {
    let mut template = NodeTemplate::new(&node.name, &node.category);

    for port in &node.inputs {
        template = template.with_input(&port.name, port.port_type);
    }

    for port in &node.outputs {
        template = template.with_output(&port.name, port.port_type);
    }

    for prop in &node.properties {
        let prop_template = match &prop.value {
            crate::flowforge::PropertyValue::Bool(v) => PropertyTemplate::bool(&prop.name, *v),
            crate::flowforge::PropertyValue::Int(v) => PropertyTemplate::int(&prop.name, *v),
            crate::flowforge::PropertyValue::Float(v) => PropertyTemplate::float(&prop.name, *v),
            crate::flowforge::PropertyValue::String(v) => PropertyTemplate::string(&prop.name, v),
            crate::flowforge::PropertyValue::Enum(v) => PropertyTemplate::enum_prop(
                &prop.name,
                prop.enum_options.clone().unwrap_or_default(),
                *v,
            ),
            _ => continue,
        };
        template = template.with_property(prop_template);
    }

    template
}

// ---------------------------------------------------------------------------
// Standard Macro Library
// ---------------------------------------------------------------------------

/// Create standard utility macros.
pub fn utility_macros() -> Vec<MacroDefinition> {
    vec![
        // Remap: maps a value from one range to another
        MacroDefinition::new("Remap", "Utility")
            .with_id("remap")
            .with_description("Maps a value from one range to another")
            .with_input("Value", PortType::Float)
            .with_input_default("InMin", PortType::Float, Value::Float(0.0))
            .with_input_default("InMax", PortType::Float, Value::Float(1.0))
            .with_input_default("OutMin", PortType::Float, Value::Float(0.0))
            .with_input_default("OutMax", PortType::Float, Value::Float(1.0))
            .with_output("Result", PortType::Float),
        // Clamp01: clamps value to 0-1 range
        MacroDefinition::new("Clamp01", "Utility")
            .with_id("clamp01")
            .with_description("Clamps a value to the 0-1 range")
            .with_input("Value", PortType::Float)
            .with_output("Result", PortType::Float),
        // Normalize: normalizes a Vec3
        MacroDefinition::new("Normalize", "Utility")
            .with_id("normalize")
            .with_description("Normalizes a Vec3 to unit length")
            .with_input("Vector", PortType::Vec3)
            .with_output("Result", PortType::Vec3),
        // Distance: distance between two Vec3
        MacroDefinition::new("Distance", "Utility")
            .with_id("distance")
            .with_description("Calculates distance between two points")
            .with_input("A", PortType::Vec3)
            .with_input("B", PortType::Vec3)
            .with_output("Result", PortType::Float),
    ]
}

/// Create standard math macros.
pub fn math_macros() -> Vec<MacroDefinition> {
    vec![
        // Vec3Add: adds two Vec3
        MacroDefinition::new("Vec3Add", "Math")
            .with_id("vec3_add")
            .with_description("Adds two Vec3 vectors")
            .with_input("A", PortType::Vec3)
            .with_input("B", PortType::Vec3)
            .with_output("Result", PortType::Vec3),
        // Vec3Dot: dot product
        MacroDefinition::new("Vec3Dot", "Math")
            .with_id("vec3_dot")
            .with_description("Calculates dot product of two Vec3")
            .with_input("A", PortType::Vec3)
            .with_input("B", PortType::Vec3)
            .with_output("Result", PortType::Float),
        // Vec3Cross: cross product
        MacroDefinition::new("Vec3Cross", "Math")
            .with_id("vec3_cross")
            .with_description("Calculates cross product of two Vec3")
            .with_input("A", PortType::Vec3)
            .with_input("B", PortType::Vec3)
            .with_output("Result", PortType::Vec3),
        // Mat4Multiply: matrix multiplication placeholder
        MacroDefinition::new("Mat4Multiply", "Math")
            .with_id("mat4_multiply")
            .with_description("Multiplies two 4x4 matrices")
            .with_input("A", PortType::Any)
            .with_input("B", PortType::Any)
            .with_output("Result", PortType::Any),
    ]
}

/// Create standard logic macros.
pub fn logic_macros() -> Vec<MacroDefinition> {
    vec![
        // Toggle: flip-flop state
        MacroDefinition::new("Toggle", "Logic")
            .with_id("toggle")
            .with_description("Toggles a boolean state on each trigger")
            .with_input("Trigger", PortType::Flow)
            .with_output("Out", PortType::Flow)
            .with_output("State", PortType::Bool),
        // Pulse: outputs flow on rising edge
        MacroDefinition::new("Pulse", "Logic")
            .with_id("pulse")
            .with_description("Outputs flow pulse on rising edge of condition")
            .with_input("Condition", PortType::Bool)
            .with_output("Out", PortType::Flow),
        // Debounce: ignores rapid triggers
        MacroDefinition::new("Debounce", "Logic")
            .with_id("debounce")
            .with_description("Ignores triggers within cooldown period")
            .with_input("Trigger", PortType::Flow)
            .with_input_default("Cooldown", PortType::Float, Value::Float(0.5))
            .with_output("Out", PortType::Flow),
        // Throttle: limits trigger rate
        MacroDefinition::new("Throttle", "Logic")
            .with_id("throttle")
            .with_description("Limits trigger rate to interval")
            .with_input("Trigger", PortType::Flow)
            .with_input_default("Interval", PortType::Float, Value::Float(0.1))
            .with_output("Out", PortType::Flow),
    ]
}

/// Get all standard macro categories.
pub fn standard_macro_categories() -> Vec<(&'static str, Vec<MacroDefinition>)> {
    vec![
        ("Utility", utility_macros()),
        ("Math", math_macros()),
        ("Logic", logic_macros()),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowforge::{NodeGraph, NodeTemplate, PortType};

    // -------------------------------------------------------------------------
    // MacroPort Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_macro_port_new() {
        let port = MacroPort::new("Value", PortType::Float);
        assert_eq!(port.name, "Value");
        assert_eq!(port.port_type, PortType::Float);
        assert!(port.default_value.is_none());
        assert!(port.description.is_empty());
    }

    #[test]
    fn test_macro_port_with_default() {
        let port = MacroPort::new("Value", PortType::Float).with_default(Value::Float(0.5));
        assert_eq!(port.default_value, Some(Value::Float(0.5)));
    }

    #[test]
    fn test_macro_port_with_description() {
        let port = MacroPort::new("Value", PortType::Float).with_description("Input value");
        assert_eq!(port.description, "Input value");
    }

    // -------------------------------------------------------------------------
    // MacroDefinition Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_macro_definition_new() {
        let macro_def = MacroDefinition::new("Add", "Math");
        assert_eq!(macro_def.id, "add");
        assert_eq!(macro_def.name, "Add");
        assert_eq!(macro_def.category, "Math");
        assert_eq!(macro_def.version, 1);
        assert!(macro_def.inputs.is_empty());
        assert!(macro_def.outputs.is_empty());
    }

    #[test]
    fn test_macro_definition_with_id() {
        let macro_def = MacroDefinition::new("Custom Add", "Math").with_id("custom_add_v2");
        assert_eq!(macro_def.id, "custom_add_v2");
    }

    #[test]
    fn test_macro_definition_with_description() {
        let macro_def = MacroDefinition::new("Add", "Math").with_description("Adds two numbers");
        assert_eq!(macro_def.description, "Adds two numbers");
    }

    #[test]
    fn test_macro_definition_with_inputs_outputs() {
        let macro_def = MacroDefinition::new("Add", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);

        assert_eq!(macro_def.input_count(), 2);
        assert_eq!(macro_def.output_count(), 1);
        assert_eq!(macro_def.inputs[0].name, "A");
        assert_eq!(macro_def.inputs[1].name, "B");
        assert_eq!(macro_def.outputs[0].name, "Result");
    }

    #[test]
    fn test_macro_definition_with_input_default() {
        let macro_def = MacroDefinition::new("Clamp", "Math")
            .with_input("Value", PortType::Float)
            .with_input_default("Min", PortType::Float, Value::Float(0.0))
            .with_input_default("Max", PortType::Float, Value::Float(1.0));

        assert_eq!(macro_def.input_count(), 3);
        assert!(macro_def.inputs[0].default_value.is_none());
        assert_eq!(macro_def.inputs[1].default_value, Some(Value::Float(0.0)));
        assert_eq!(macro_def.inputs[2].default_value, Some(Value::Float(1.0)));
    }

    #[test]
    fn test_macro_definition_with_version() {
        let macro_def = MacroDefinition::new("Add", "Math").with_version(2);
        assert_eq!(macro_def.version, 2);
    }

    #[test]
    fn test_macro_definition_validate_valid() {
        let macro_def = MacroDefinition::new("Add", "Math")
            .with_input("A", PortType::Float)
            .with_output("Result", PortType::Float);

        assert!(macro_def.validate().is_ok());
    }

    #[test]
    fn test_macro_definition_validate_empty_id() {
        let mut macro_def = MacroDefinition::new("Add", "Math");
        macro_def.id = String::new();

        assert!(matches!(
            macro_def.validate(),
            Err(MacroError::InvalidDefinition(_))
        ));
    }

    #[test]
    fn test_macro_definition_validate_empty_name() {
        let mut macro_def = MacroDefinition::new("", "Math");
        macro_def.id = "valid_id".to_string();

        assert!(matches!(
            macro_def.validate(),
            Err(MacroError::InvalidDefinition(_))
        ));
    }

    #[test]
    fn test_macro_definition_to_template() {
        let macro_def = MacroDefinition::new("Add", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);

        let template = macro_def.to_template();
        assert_eq!(template.name, "Add");
        assert_eq!(template.category, "Macro/Math");
        assert_eq!(template.inputs.len(), 2);
        assert_eq!(template.outputs.len(), 1);
    }

    // -------------------------------------------------------------------------
    // MacroInstance Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_macro_instance_new() {
        let instance = MacroInstance::new("add", 42);
        assert_eq!(instance.definition_id, "add");
        assert_eq!(instance.instance_id, 42);
        assert!(instance.input_mappings.is_empty());
        assert!(instance.output_mappings.is_empty());
    }

    #[test]
    fn test_macro_instance_with_version() {
        let instance = MacroInstance::new("add", 1).with_version(3);
        assert_eq!(instance.version, 3);
    }

    #[test]
    fn test_macro_instance_map_input() {
        let mut instance = MacroInstance::new("add", 1);
        let conn = Connection::new(10, 20, 1, 1);
        instance.map_input(1, conn.clone());

        assert_eq!(instance.get_input_mapping(1), Some(&conn));
        assert!(instance.get_input_mapping(2).is_none());
    }

    #[test]
    fn test_macro_instance_map_output() {
        let mut instance = MacroInstance::new("add", 1);
        let conn = Connection::new(1, 1, 10, 20);
        instance.map_output(1, conn.clone());

        assert_eq!(instance.get_output_mapping(1), Some(&conn));
    }

    #[test]
    fn test_macro_instance_inputs_complete() {
        let macro_def = MacroDefinition::new("Test", "Test")
            .with_input("Required", PortType::Float)
            .with_input_default("Optional", PortType::Float, Value::Float(0.0));

        let mut instance = MacroInstance::new("test", 1);
        assert!(!instance.inputs_complete(&macro_def));

        instance.map_input(1, Connection::new(0, 0, 1, 1));
        assert!(instance.inputs_complete(&macro_def));
    }

    // -------------------------------------------------------------------------
    // MacroLibrary Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_macro_library_new() {
        let library = MacroLibrary::new();
        assert!(library.is_empty());
        assert_eq!(library.len(), 0);
    }

    #[test]
    fn test_macro_library_register() {
        let mut library = MacroLibrary::new();
        let macro_def = MacroDefinition::new("Add", "Math");

        library.register(macro_def);
        assert_eq!(library.len(), 1);
        assert!(library.contains("add"));
    }

    #[test]
    fn test_macro_library_get() {
        let mut library = MacroLibrary::new();
        library.register(MacroDefinition::new("Add", "Math"));

        assert!(library.get("add").is_some());
        assert!(library.get("nonexistent").is_none());
    }

    #[test]
    fn test_macro_library_get_mut() {
        let mut library = MacroLibrary::new();
        library.register(MacroDefinition::new("Add", "Math"));

        if let Some(macro_def) = library.get_mut("add") {
            macro_def.description = "Modified".to_string();
        }

        assert_eq!(library.get("add").unwrap().description, "Modified");
    }

    #[test]
    fn test_macro_library_unregister() {
        let mut library = MacroLibrary::new();
        library.register(MacroDefinition::new("Add", "Math"));

        let removed = library.unregister("add");
        assert!(removed.is_some());
        assert!(!library.contains("add"));
    }

    #[test]
    fn test_macro_library_list_by_category() {
        let mut library = MacroLibrary::new();
        library.register(MacroDefinition::new("Add", "Math"));
        library.register(MacroDefinition::new("Sub", "Math"));
        library.register(MacroDefinition::new("And", "Logic"));

        let math_macros = library.list_by_category("Math");
        assert_eq!(math_macros.len(), 2);

        let logic_macros = library.list_by_category("Logic");
        assert_eq!(logic_macros.len(), 1);
    }

    #[test]
    fn test_macro_library_categories() {
        let mut library = MacroLibrary::new();
        library.register(MacroDefinition::new("Add", "Math"));
        library.register(MacroDefinition::new("And", "Logic"));

        let categories = library.categories();
        assert!(categories.contains(&"Math".to_string()));
        assert!(categories.contains(&"Logic".to_string()));
    }

    #[test]
    fn test_macro_library_macro_ids() {
        let mut library = MacroLibrary::new();
        library.register(MacroDefinition::new("Add", "Math"));
        library.register(MacroDefinition::new("Sub", "Math"));

        let ids = library.macro_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"add"));
        assert!(ids.contains(&"sub"));
    }

    #[test]
    fn test_macro_library_create_instance() {
        let mut library = MacroLibrary::new();
        library.register(MacroDefinition::new("Add", "Math").with_version(2));

        let instance = library.create_instance("add");
        assert!(instance.is_some());
        let instance = instance.unwrap();
        assert_eq!(instance.definition_id, "add");
        assert_eq!(instance.version, 2);

        assert!(library.create_instance("nonexistent").is_none());
    }

    #[test]
    fn test_macro_library_search() {
        let mut library = MacroLibrary::new();
        library.register(MacroDefinition::new("Vec3Add", "Math").with_description("Add vectors"));
        library.register(MacroDefinition::new("Vec3Sub", "Math").with_description("Subtract"));
        library.register(MacroDefinition::new("Lerp", "Math"));

        let results = library.search("vec");
        assert_eq!(results.len(), 2);

        let results = library.search("add");
        assert_eq!(results.len(), 1);

        let results = library.search("vectors");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_macro_library_with_standard_macros() {
        let library = MacroLibrary::with_standard_macros();
        assert!(!library.is_empty());

        // Check utility macros
        assert!(library.contains("remap"));
        assert!(library.contains("clamp01"));
        assert!(library.contains("normalize"));
        assert!(library.contains("distance"));

        // Check math macros
        assert!(library.contains("vec3_add"));
        assert!(library.contains("vec3_dot"));
        assert!(library.contains("vec3_cross"));
        assert!(library.contains("mat4_multiply"));

        // Check logic macros
        assert!(library.contains("toggle"));
        assert!(library.contains("pulse"));
        assert!(library.contains("debounce"));
        assert!(library.contains("throttle"));
    }

    // -------------------------------------------------------------------------
    // Expand Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_expand_simple_macro() {
        let mut library = MacroLibrary::new();

        // Create a macro with one node
        let mut inner_graph = NodeGraph::new();
        let template = NodeTemplate::new("Add", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);
        inner_graph.add_node(&template);

        let macro_def = MacroDefinition::new("SimpleAdd", "Math")
            .with_inner_graph(inner_graph)
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);

        library.register(macro_def);

        let instance = library.create_instance("simpleadd").unwrap();
        let mut target_graph = NodeGraph::new();

        let result = library.expand(&instance, &mut target_graph);
        assert!(result.is_ok());

        let new_nodes = result.unwrap();
        assert_eq!(new_nodes.len(), 1);
        assert_eq!(target_graph.node_count(), 1);
    }

    #[test]
    fn test_expand_macro_with_connections() {
        let mut library = MacroLibrary::new();

        // Create a macro with two connected nodes
        let mut inner_graph = NodeGraph::new();

        let add_template = NodeTemplate::new("Add", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);

        let abs_template = NodeTemplate::new("Abs", "Math")
            .with_input("Value", PortType::Float)
            .with_output("Result", PortType::Float);

        let add_id = inner_graph.add_node(&add_template);
        let abs_id = inner_graph.add_node(&abs_template);

        let add_out = inner_graph.get_node(add_id).unwrap().outputs[0].id;
        let abs_in = inner_graph.get_node(abs_id).unwrap().inputs[0].id;
        let _ = inner_graph.connect((add_id, add_out), (abs_id, abs_in));

        let macro_def = MacroDefinition::new("AddAbs", "Math").with_inner_graph(inner_graph);
        library.register(macro_def);

        let instance = library.create_instance("addabs").unwrap();
        let mut target_graph = NodeGraph::new();

        let result = library.expand(&instance, &mut target_graph);
        assert!(result.is_ok());
        assert_eq!(target_graph.node_count(), 2);
        assert_eq!(target_graph.connection_count(), 1);
    }

    #[test]
    fn test_expand_macro_not_found() {
        let library = MacroLibrary::new();
        let instance = MacroInstance::new("nonexistent", 1);
        let mut target_graph = NodeGraph::new();

        let result = library.expand(&instance, &mut target_graph);
        assert!(matches!(result, Err(MacroError::MacroNotFound(_))));
    }

    #[test]
    fn test_expand_version_mismatch() {
        let mut library = MacroLibrary::new();
        library.register(MacroDefinition::new("Add", "Math").with_version(2));

        let instance = MacroInstance::new("add", 1).with_version(1); // Wrong version
        let mut target_graph = NodeGraph::new();

        let result = library.expand(&instance, &mut target_graph);
        assert!(matches!(
            result,
            Err(MacroError::VersionMismatch {
                expected: 2,
                found: 1
            })
        ));
    }

    // -------------------------------------------------------------------------
    // Collapse Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_collapse_single_node() {
        let library = MacroLibrary::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("Add", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);

        let node_id = graph.add_node(&template);

        let result = library.collapse(&[node_id], &graph);
        assert!(result.is_ok());

        let macro_def = result.unwrap();
        assert_eq!(macro_def.inner_graph.node_count(), 1);
        assert!(macro_def.inputs.len() >= 2); // A and B
        assert!(macro_def.outputs.len() >= 1); // Result
    }

    #[test]
    fn test_collapse_connected_nodes() {
        let library = MacroLibrary::new();
        let mut graph = NodeGraph::new();

        let add_template = NodeTemplate::new("Add", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);

        let abs_template = NodeTemplate::new("Abs", "Math")
            .with_input("Value", PortType::Float)
            .with_output("Result", PortType::Float);

        let add_id = graph.add_node(&add_template);
        let abs_id = graph.add_node(&abs_template);

        let add_out = graph.get_node(add_id).unwrap().outputs[0].id;
        let abs_in = graph.get_node(abs_id).unwrap().inputs[0].id;
        let _ = graph.connect((add_id, add_out), (abs_id, abs_in));

        let result = library.collapse(&[add_id, abs_id], &graph);
        assert!(result.is_ok());

        let macro_def = result.unwrap();
        assert_eq!(macro_def.inner_graph.node_count(), 2);
        assert_eq!(macro_def.inner_graph.connection_count(), 1);
    }

    #[test]
    fn test_collapse_empty_selection() {
        let library = MacroLibrary::new();
        let graph = NodeGraph::new();

        let result = library.collapse(&[], &graph);
        assert!(matches!(result, Err(MacroError::CannotCollapse(_))));
    }

    #[test]
    fn test_collapse_node_not_found() {
        let library = MacroLibrary::new();
        let graph = NodeGraph::new();

        let result = library.collapse(&[999], &graph);
        assert!(matches!(result, Err(MacroError::NodeNotFound(999))));
    }

    // -------------------------------------------------------------------------
    // Standard Macros Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_utility_macros() {
        let macros = utility_macros();
        assert_eq!(macros.len(), 4);

        let names: Vec<&str> = macros.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"Remap"));
        assert!(names.contains(&"Clamp01"));
        assert!(names.contains(&"Normalize"));
        assert!(names.contains(&"Distance"));
    }

    #[test]
    fn test_math_macros() {
        let macros = math_macros();
        assert_eq!(macros.len(), 4);

        let names: Vec<&str> = macros.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"Vec3Add"));
        assert!(names.contains(&"Vec3Dot"));
        assert!(names.contains(&"Vec3Cross"));
        assert!(names.contains(&"Mat4Multiply"));
    }

    #[test]
    fn test_logic_macros() {
        let macros = logic_macros();
        assert_eq!(macros.len(), 4);

        let names: Vec<&str> = macros.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"Toggle"));
        assert!(names.contains(&"Pulse"));
        assert!(names.contains(&"Debounce"));
        assert!(names.contains(&"Throttle"));
    }

    #[test]
    fn test_standard_macro_categories() {
        let categories = standard_macro_categories();
        assert_eq!(categories.len(), 3);

        let category_names: Vec<&str> = categories.iter().map(|(name, _)| *name).collect();
        assert!(category_names.contains(&"Utility"));
        assert!(category_names.contains(&"Math"));
        assert!(category_names.contains(&"Logic"));
    }

    #[test]
    fn test_remap_macro_definition() {
        let macros = utility_macros();
        let remap = macros.iter().find(|m| m.id == "remap").unwrap();

        assert_eq!(remap.input_count(), 5);
        assert_eq!(remap.output_count(), 1);

        // Check defaults
        assert!(remap.inputs[1].default_value.is_some()); // InMin
        assert!(remap.inputs[2].default_value.is_some()); // InMax
        assert!(remap.inputs[3].default_value.is_some()); // OutMin
        assert!(remap.inputs[4].default_value.is_some()); // OutMax
    }

    #[test]
    fn test_debounce_macro_definition() {
        let macros = logic_macros();
        let debounce = macros.iter().find(|m| m.id == "debounce").unwrap();

        assert_eq!(debounce.input_count(), 2);
        assert_eq!(debounce.inputs[0].port_type, PortType::Flow);
        assert_eq!(debounce.inputs[1].port_type, PortType::Float);
        assert_eq!(
            debounce.inputs[1].default_value,
            Some(Value::Float(0.5))
        );
    }

    // -------------------------------------------------------------------------
    // MacroError Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_macro_error_display() {
        let err = MacroError::MacroNotFound("test".to_string());
        assert!(err.to_string().contains("test"));

        let err = MacroError::VersionMismatch {
            expected: 2,
            found: 1,
        };
        assert!(err.to_string().contains("2"));
        assert!(err.to_string().contains("1"));

        let err = MacroError::CycleDetected;
        assert!(err.to_string().contains("Cycle"));
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_full_workflow_register_create_expand() {
        let mut library = MacroLibrary::new();

        // Create a simple macro
        let mut inner_graph = NodeGraph::new();
        let template = NodeTemplate::new("Add", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);
        inner_graph.add_node(&template);

        let macro_def = MacroDefinition::new("TestAdd", "Test")
            .with_inner_graph(inner_graph)
            .with_input("X", PortType::Float)
            .with_input("Y", PortType::Float)
            .with_output("Sum", PortType::Float)
            .with_description("Adds X and Y");

        // Register
        library.register(macro_def);
        assert!(library.contains("testadd"));

        // Create instance
        let instance = library.create_instance("testadd").unwrap();
        assert_eq!(instance.definition_id, "testadd");

        // Expand into target graph
        let mut target = NodeGraph::new();
        let new_nodes = library.expand(&instance, &mut target).unwrap();
        assert_eq!(new_nodes.len(), 1);
        assert_eq!(target.node_count(), 1);
    }

    #[test]
    fn test_full_workflow_collapse_and_expand() {
        let library = MacroLibrary::new();
        let mut source_graph = NodeGraph::new();

        // Create two connected nodes
        let add_template = NodeTemplate::new("Add", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);

        let mul_template = NodeTemplate::new("Multiply", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);

        let add_id = source_graph.add_node(&add_template);
        let mul_id = source_graph.add_node(&mul_template);

        let add_out = source_graph.get_node(add_id).unwrap().outputs[0].id;
        let mul_in = source_graph.get_node(mul_id).unwrap().inputs[0].id;
        let _ = source_graph.connect((add_id, add_out), (mul_id, mul_in));

        // Collapse to macro
        let macro_def = library.collapse(&[add_id, mul_id], &source_graph).unwrap();
        assert_eq!(macro_def.inner_graph.node_count(), 2);
        assert_eq!(macro_def.inner_graph.connection_count(), 1);

        // Register and expand
        let mut library = MacroLibrary::new();
        library.register(macro_def);

        let instance = library
            .create_instance("collapsed_macro_2")
            .unwrap();
        let mut target = NodeGraph::new();

        let result = library.expand(&instance, &mut target);
        assert!(result.is_ok());
        assert_eq!(target.node_count(), 2);
        assert_eq!(target.connection_count(), 1);
    }

    #[test]
    fn test_macro_template_in_graph() {
        let library = MacroLibrary::with_standard_macros();

        let clamp_macro = library.get("clamp01").unwrap();
        let template = clamp_macro.to_template();

        let mut graph = NodeGraph::new();
        let node_id = graph.add_node(&template);

        let node = graph.get_node(node_id).unwrap();
        assert_eq!(node.name, "Clamp01");
        assert_eq!(node.category, "Macro/Utility");
        assert_eq!(node.inputs.len(), 1);
        assert_eq!(node.outputs.len(), 1);
    }
}
