//! FlowForge Python Bytecode Compiler
//!
//! This module compiles FlowForge node graphs to executable Python code.
//! It provides:
//! - GraphCompiler for code generation
//! - CompiledGraph output with metadata
//! - Node-to-Python translation for all node types
//! - Variable scoping and unique naming
//! - Comprehensive error handling with suggestions

use crate::flowforge::{
    Connection, Node, NodeGraph, NodeId, PortId, PortType, PropertyValue,
};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Compile Error
// ---------------------------------------------------------------------------

/// Error during graph compilation.
#[derive(Debug, Clone, PartialEq)]
pub struct CompileError {
    /// Node ID where error occurred.
    pub node_id: Option<NodeId>,
    /// Error message.
    pub message: String,
    /// Suggestion for fixing the error.
    pub suggestion: Option<String>,
}

impl CompileError {
    /// Create a new compile error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            node_id: None,
            message: message.into(),
            suggestion: None,
        }
    }

    /// Create a compile error with node context.
    pub fn at_node(node_id: NodeId, message: impl Into<String>) -> Self {
        Self {
            node_id: Some(node_id),
            message: message.into(),
            suggestion: None,
        }
    }

    /// Add a suggestion to the error.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(node_id) = self.node_id {
            write!(f, "[Node {}] {}", node_id, self.message)?;
        } else {
            write!(f, "{}", self.message)?;
        }
        if let Some(ref suggestion) = self.suggestion {
            write!(f, " (suggestion: {})", suggestion)?;
        }
        Ok(())
    }
}

impl std::error::Error for CompileError {}

// ---------------------------------------------------------------------------
// Compile Metadata
// ---------------------------------------------------------------------------

/// Metadata about the compiled graph.
#[derive(Debug, Clone, Default)]
pub struct CompileMetadata {
    /// Total number of nodes compiled.
    pub node_count: usize,
    /// Total number of connections processed.
    pub connection_count: usize,
    /// Number of event handlers generated.
    pub event_count: usize,
    /// Number of variables used.
    pub variable_count: usize,
    /// Total lines of generated code.
    pub line_count: usize,
    /// Warnings during compilation.
    pub warnings: Vec<String>,
}

// ---------------------------------------------------------------------------
// Compiled Graph
// ---------------------------------------------------------------------------

/// Result of compiling a node graph.
#[derive(Debug, Clone)]
pub struct CompiledGraph {
    /// Generated Python source code.
    pub python_source: String,
    /// Entry point function names (event handlers).
    pub entry_points: Vec<String>,
    /// Required Python imports.
    pub dependencies: Vec<String>,
    /// Compilation metadata.
    pub metadata: CompileMetadata,
}

impl CompiledGraph {
    /// Create a new compiled graph.
    pub fn new(
        python_source: String,
        entry_points: Vec<String>,
        dependencies: Vec<String>,
        metadata: CompileMetadata,
    ) -> Self {
        Self {
            python_source,
            entry_points,
            dependencies,
            metadata,
        }
    }

    /// Get the full Python source with imports.
    pub fn full_source(&self) -> String {
        let mut result = String::new();

        // Add imports
        if !self.dependencies.is_empty() {
            for dep in &self.dependencies {
                result.push_str(dep);
                result.push('\n');
            }
            result.push('\n');
        }

        result.push_str(&self.python_source);
        result
    }
}

// ---------------------------------------------------------------------------
// Graph Compiler
// ---------------------------------------------------------------------------

/// Compiles FlowForge node graphs to Python code.
#[derive(Debug)]
pub struct GraphCompiler {
    /// Current indentation level.
    indent_level: usize,
    /// Generated output.
    output: String,
    /// Counter for temporary variable names.
    temp_var_counter: usize,
    /// Mapping from (node_id, port_id) to variable names.
    port_variables: HashMap<(NodeId, PortId), String>,
    /// Set of required imports.
    imports: HashSet<String>,
    /// Generated function names.
    functions: Vec<String>,
    /// Warnings collected during compilation.
    warnings: Vec<String>,
    /// Node execution order (topologically sorted).
    execution_order: Vec<NodeId>,
}

impl Default for GraphCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphCompiler {
    /// Create a new graph compiler.
    pub fn new() -> Self {
        Self {
            indent_level: 0,
            output: String::new(),
            temp_var_counter: 0,
            port_variables: HashMap::new(),
            imports: HashSet::new(),
            functions: Vec::new(),
            warnings: Vec::new(),
            execution_order: Vec::new(),
        }
    }

    /// Reset the compiler state.
    fn reset(&mut self) {
        self.indent_level = 0;
        self.output.clear();
        self.temp_var_counter = 0;
        self.port_variables.clear();
        self.imports.clear();
        self.functions.clear();
        self.warnings.clear();
        self.execution_order.clear();
    }

    /// Compile a node graph to a CompiledGraph.
    pub fn compile(&mut self, graph: &NodeGraph) -> Result<CompiledGraph, CompileError> {
        self.reset();

        // Validate the graph
        self.validate_graph(graph)?;

        // Get topologically sorted execution order
        self.execution_order = graph.topological_sort().map_err(|e| {
            CompileError::new(format!("Cycle detected in graph: {}", e))
                .with_suggestion("Remove circular connections")
        })?;

        // Generate Python code
        let python_source = self.generate_python(graph)?;

        // Count lines
        let line_count = python_source.lines().count();

        // Build metadata
        let metadata = CompileMetadata {
            node_count: graph.node_count(),
            connection_count: graph.connection_count(),
            event_count: self.functions.len(),
            variable_count: self.port_variables.len(),
            line_count,
            warnings: self.warnings.clone(),
        };

        Ok(CompiledGraph::new(
            python_source,
            self.functions.clone(),
            self.imports.iter().cloned().collect(),
            metadata,
        ))
    }

    /// Generate Python source code from a node graph.
    pub fn generate_python(&mut self, graph: &NodeGraph) -> Result<String, CompileError> {
        // Find all event nodes (entry points)
        let event_nodes: Vec<NodeId> = graph
            .nodes()
            .filter(|n| self.is_event_node(n))
            .map(|n| n.id)
            .collect();

        if event_nodes.is_empty() {
            self.warnings
                .push("No event nodes found; graph has no entry points".to_string());
        }

        // Generate runtime context class
        self.emit_line("# FlowForge Generated Python Code");
        self.emit_line("# Auto-generated - do not edit manually");
        self.emit_line("");
        self.emit_runtime_context();

        // Generate event handler functions
        for event_node_id in &event_nodes {
            if let Some(node) = graph.get_node(*event_node_id) {
                self.generate_event_handler(graph, node)?;
            }
        }

        // Generate main execution function
        self.generate_main_function(&event_nodes);

        Ok(self.output.clone())
    }

    /// Validate the graph before compilation.
    fn validate_graph(&self, graph: &NodeGraph) -> Result<(), CompileError> {
        // Check for empty graph
        if graph.is_empty() {
            return Ok(()); // Empty graph is valid, just produces no code
        }

        // Check for disconnected inputs on critical nodes
        for node in graph.nodes() {
            // Flow nodes require flow inputs
            if self.is_flow_node(node) && !self.is_event_node(node) {
                let has_flow_input = node
                    .inputs
                    .iter()
                    .any(|p| p.port_type == PortType::Flow);
                if has_flow_input {
                    let flow_input = node
                        .inputs
                        .iter()
                        .find(|p| p.port_type == PortType::Flow)
                        .unwrap();
                    let has_connection = graph
                        .connections()
                        .iter()
                        .any(|c| c.to_node == node.id && c.to_port == flow_input.id);
                    if !has_connection {
                        // Not an error, but a warning
                        // self.warnings.push(format!("Node {} has no flow input connection", node.name));
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate the runtime context class.
    fn emit_runtime_context(&mut self) {
        self.imports.insert("import math".to_string());
        self.imports.insert("import random".to_string());

        self.emit_line("class FlowForgeContext:");
        self.indent();
        self.emit_line("\"\"\"Runtime context for FlowForge execution.\"\"\"");
        self.emit_line("");
        self.emit_line("def __init__(self):");
        self.indent();
        self.emit_line("self.variables = {}");
        self.emit_line("self.components = {}");
        self.emit_line("self.events = set()");
        self.emit_line("self.delta_time = 0.016");
        self.emit_line("self.frame = 0");
        self.emit_line("self.do_once_state = set()");
        self.emit_line("self.delay_timers = {}");
        self.emit_line("self.loop_counters = {}");
        self.emit_line("self.debug_log = []");
        self.dedent();
        self.emit_line("");

        // Variable accessors
        self.emit_line("def get_variable(self, name, default=None):");
        self.indent();
        self.emit_line("return self.variables.get(name, default)");
        self.dedent();
        self.emit_line("");

        self.emit_line("def set_variable(self, name, value):");
        self.indent();
        self.emit_line("self.variables[name] = value");
        self.dedent();
        self.emit_line("");

        // Component accessors
        self.emit_line("def get_component(self, entity, component):");
        self.indent();
        self.emit_line("return self.components.get((entity, component))");
        self.dedent();
        self.emit_line("");

        self.emit_line("def set_component(self, entity, component, value):");
        self.indent();
        self.emit_line("self.components[(entity, component)] = value");
        self.dedent();
        self.emit_line("");

        self.emit_line("def has_component(self, entity, component):");
        self.indent();
        self.emit_line("return (entity, component) in self.components");
        self.dedent();
        self.emit_line("");

        self.emit_line("def remove_component(self, entity, component):");
        self.indent();
        self.emit_line("self.components.pop((entity, component), None)");
        self.dedent();
        self.emit_line("");

        // Debug logging
        self.emit_line("def log(self, message):");
        self.indent();
        self.emit_line("self.debug_log.append(str(message))");
        self.dedent();
        self.emit_line("");

        // Random
        self.emit_line("def random_float(self):");
        self.indent();
        self.emit_line("return random.random()");
        self.dedent();

        self.dedent();
        self.emit_line("");
        self.emit_line("");
    }

    /// Generate an event handler function.
    fn generate_event_handler(
        &mut self,
        graph: &NodeGraph,
        event_node: &Node,
    ) -> Result<(), CompileError> {
        let func_name = self.make_event_function_name(event_node);
        self.functions.push(func_name.clone());

        self.emit_line(&format!("def {}(ctx):", func_name));
        self.indent();
        self.emit_line(&format!(
            "\"\"\"Event handler for {} node.\"\"\"",
            event_node.name
        ));

        // Generate code for the event chain
        self.generate_node_chain(graph, event_node)?;

        self.dedent();
        self.emit_line("");

        Ok(())
    }

    /// Generate code for a node and its flow successors.
    fn generate_node_chain(&mut self, graph: &NodeGraph, node: &Node) -> Result<(), CompileError> {
        // Generate code for this node
        self.generate_node_code(graph, node)?;

        // Find flow output ports and their connections
        let flow_outputs: Vec<&crate::flowforge::Port> = node
            .outputs
            .iter()
            .filter(|p| p.port_type == PortType::Flow)
            .collect();

        for flow_port in flow_outputs {
            // Find connections from this flow port
            let connections: Vec<&Connection> = graph
                .connections()
                .iter()
                .filter(|c| c.from_node == node.id && c.from_port == flow_port.id)
                .collect();

            for conn in connections {
                if let Some(next_node) = graph.get_node(conn.to_node) {
                    self.generate_node_chain(graph, next_node)?;
                }
            }
        }

        Ok(())
    }

    /// Generate Python code for a single node.
    fn generate_node_code(&mut self, graph: &NodeGraph, node: &Node) -> Result<(), CompileError> {
        // Get input values from connected nodes
        let inputs = self.resolve_inputs(graph, node);

        match node.name.as_str() {
            // Event nodes - just comments, actual handler generated separately
            "OnStart" => {
                self.emit_line("# OnStart event");
            }
            "OnTick" => {
                self.emit_line("# OnTick event");
                // Set DeltaTime output
                if let Some(dt_port) = node.outputs.iter().find(|p| p.name == "DeltaTime") {
                    let var = self.get_or_create_var(node.id, dt_port.id);
                    self.emit_line(&format!("{} = ctx.delta_time", var));
                }
            }
            "OnKeyPressed" | "OnKeyReleased" | "OnCollision" | "CustomEvent" => {
                self.emit_line(&format!("# {} event", node.name));
            }

            // Math nodes
            "Add" => {
                let a = inputs.get("A").cloned().unwrap_or_else(|| "0.0".to_string());
                let b = inputs.get("B").cloned().unwrap_or_else(|| "0.0".to_string());
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Result") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!("{} = {} + {}", var, a, b));
                }
            }
            "Subtract" => {
                let a = inputs.get("A").cloned().unwrap_or_else(|| "0.0".to_string());
                let b = inputs.get("B").cloned().unwrap_or_else(|| "0.0".to_string());
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Result") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!("{} = {} - {}", var, a, b));
                }
            }
            "Multiply" => {
                let a = inputs.get("A").cloned().unwrap_or_else(|| "0.0".to_string());
                let b = inputs.get("B").cloned().unwrap_or_else(|| "0.0".to_string());
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Result") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!("{} = {} * {}", var, a, b));
                }
            }
            "Divide" => {
                let a = inputs.get("A").cloned().unwrap_or_else(|| "0.0".to_string());
                let b = inputs.get("B").cloned().unwrap_or_else(|| "1.0".to_string());
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Result") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!(
                        "{} = {} / {} if {} != 0 else 0.0",
                        var, a, b, b
                    ));
                }
            }
            "Abs" => {
                let val = inputs.get("Value").cloned().unwrap_or_else(|| "0.0".to_string());
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Result") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!("{} = abs({})", var, val));
                }
            }
            "Sin" => {
                let angle = inputs
                    .get("Angle")
                    .cloned()
                    .unwrap_or_else(|| "0.0".to_string());
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Result") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!("{} = math.sin({})", var, angle));
                }
            }
            "Cos" => {
                let angle = inputs
                    .get("Angle")
                    .cloned()
                    .unwrap_or_else(|| "0.0".to_string());
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Result") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!("{} = math.cos({})", var, angle));
                }
            }
            "Lerp" => {
                let a = inputs.get("A").cloned().unwrap_or_else(|| "0.0".to_string());
                let b = inputs.get("B").cloned().unwrap_or_else(|| "0.0".to_string());
                let t = inputs.get("T").cloned().unwrap_or_else(|| "0.0".to_string());
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Result") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!(
                        "{} = {} + ({} - {}) * max(0.0, min(1.0, {}))",
                        var, a, b, a, t
                    ));
                }
            }
            "Clamp" => {
                let val = inputs.get("Value").cloned().unwrap_or_else(|| "0.0".to_string());
                let min = inputs.get("Min").cloned().unwrap_or_else(|| "0.0".to_string());
                let max = inputs.get("Max").cloned().unwrap_or_else(|| "1.0".to_string());
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Result") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!(
                        "{} = max({}, min({}, {}))",
                        var, min, max, val
                    ));
                }
            }
            "Random" => {
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Value") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!("{} = ctx.random_float()", var));
                }
            }

            // Logic nodes
            "And" => {
                let a = inputs.get("A").cloned().unwrap_or_else(|| "False".to_string());
                let b = inputs.get("B").cloned().unwrap_or_else(|| "False".to_string());
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Result") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!("{} = {} and {}", var, a, b));
                }
            }
            "Or" => {
                let a = inputs.get("A").cloned().unwrap_or_else(|| "False".to_string());
                let b = inputs.get("B").cloned().unwrap_or_else(|| "False".to_string());
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Result") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!("{} = {} or {}", var, a, b));
                }
            }
            "Not" => {
                let val = inputs
                    .get("Value")
                    .cloned()
                    .unwrap_or_else(|| "False".to_string());
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Result") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!("{} = not {}", var, val));
                }
            }
            "Compare" => {
                let a = inputs.get("A").cloned().unwrap_or_else(|| "0.0".to_string());
                let b = inputs.get("B").cloned().unwrap_or_else(|| "0.0".to_string());
                if let Some(less_port) = node.outputs.iter().find(|p| p.name == "Less") {
                    let var = self.get_or_create_var(node.id, less_port.id);
                    self.emit_line(&format!("{} = {} < {}", var, a, b));
                }
                if let Some(equal_port) = node.outputs.iter().find(|p| p.name == "Equal") {
                    let var = self.get_or_create_var(node.id, equal_port.id);
                    self.emit_line(&format!("{} = abs({} - {}) < 1e-6", var, a, b));
                }
                if let Some(greater_port) = node.outputs.iter().find(|p| p.name == "Greater") {
                    let var = self.get_or_create_var(node.id, greater_port.id);
                    self.emit_line(&format!("{} = {} > {}", var, a, b));
                }
            }
            "Select" => {
                let cond = inputs
                    .get("Condition")
                    .cloned()
                    .unwrap_or_else(|| "False".to_string());
                let t = inputs
                    .get("True")
                    .cloned()
                    .unwrap_or_else(|| "None".to_string());
                let f = inputs
                    .get("False")
                    .cloned()
                    .unwrap_or_else(|| "None".to_string());
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Result") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!("{} = {} if {} else {}", var, t, cond, f));
                }
            }

            // Flow control
            "Sequence" => {
                self.emit_line("# Sequence - execute all branches");
            }
            "Branch" => {
                let cond = inputs
                    .get("Condition")
                    .cloned()
                    .unwrap_or_else(|| "False".to_string());
                self.emit_line(&format!("if {}:", cond));
                self.indent();
                // True branch will be filled by flow connections
                self.generate_flow_branch(graph, node, "True")?;
                self.dedent();
                self.emit_line("else:");
                self.indent();
                // False branch
                self.generate_flow_branch(graph, node, "False")?;
                self.dedent();
            }
            "ForLoop" => {
                let count = inputs
                    .get("Count")
                    .cloned()
                    .unwrap_or_else(|| "0".to_string());
                let idx_var = if let Some(idx_port) = node.outputs.iter().find(|p| p.name == "Index")
                {
                    self.get_or_create_var(node.id, idx_port.id)
                } else {
                    self.new_temp_var()
                };
                self.emit_line(&format!("for {} in range(int({})):", idx_var, count));
                self.indent();
                // Loop body
                self.generate_flow_branch(graph, node, "Loop")?;
                self.dedent();
                // Completed branch
                self.emit_line("# Loop completed");
            }
            "WhileLoop" => {
                let cond = inputs
                    .get("Condition")
                    .cloned()
                    .unwrap_or_else(|| "False".to_string());
                self.emit_line(&format!("while {}:", cond));
                self.indent();
                self.generate_flow_branch(graph, node, "Loop")?;
                self.dedent();
            }
            "DoOnce" => {
                let node_key = format!("node_{}", node.id);
                self.emit_line(&format!("if '{}' not in ctx.do_once_state:", node_key));
                self.indent();
                self.emit_line(&format!("ctx.do_once_state.add('{}')", node_key));
                // Will continue to Out flow
                self.dedent();
            }
            "Delay" => {
                let duration = inputs
                    .get("Duration")
                    .cloned()
                    .unwrap_or_else(|| "1.0".to_string());
                self.emit_line(&format!(
                    "# Delay for {} seconds (async not supported in static compile)",
                    duration
                ));
                self.warnings.push(format!(
                    "Delay node {} requires async runtime, compiled as pass-through",
                    node.id
                ));
            }

            // Variable nodes
            "GetFloat" | "GetBool" | "GetVec3" => {
                let name = self.get_property_string(node, "Name", "MyVar");
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Value") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    let default = match node.name.as_str() {
                        "GetFloat" => "0.0",
                        "GetBool" => "False",
                        "GetVec3" => "(0.0, 0.0, 0.0)",
                        _ => "None",
                    };
                    self.emit_line(&format!(
                        "{} = ctx.get_variable('{}', {})",
                        var, name, default
                    ));
                }
            }
            "SetFloat" | "SetBool" | "SetVec3" => {
                let name = self.get_property_string(node, "Name", "MyVar");
                let val = inputs
                    .get("Value")
                    .cloned()
                    .unwrap_or_else(|| "None".to_string());
                self.emit_line(&format!("ctx.set_variable('{}', {})", name, val));
            }

            // ECS nodes
            "GetComponent" => {
                let entity = inputs
                    .get("Entity")
                    .cloned()
                    .unwrap_or_else(|| "0".to_string());
                let component = self.get_property_string(node, "Component", "Position");
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Value") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!(
                        "{} = ctx.get_component({}, '{}')",
                        var, entity, component
                    ));
                }
            }
            "SetComponent" => {
                let entity = inputs
                    .get("Entity")
                    .cloned()
                    .unwrap_or_else(|| "0".to_string());
                let val = inputs
                    .get("Value")
                    .cloned()
                    .unwrap_or_else(|| "None".to_string());
                let component = self.get_property_string(node, "Component", "Position");
                self.emit_line(&format!(
                    "ctx.set_component({}, '{}', {})",
                    entity, component, val
                ));
            }
            "HasComponent" => {
                let entity = inputs
                    .get("Entity")
                    .cloned()
                    .unwrap_or_else(|| "0".to_string());
                let component = self.get_property_string(node, "Component", "Position");
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Result") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!(
                        "{} = ctx.has_component({}, '{}')",
                        var, entity, component
                    ));
                }
            }
            "AddComponent" => {
                let entity = inputs
                    .get("Entity")
                    .cloned()
                    .unwrap_or_else(|| "0".to_string());
                let component = self.get_property_string(node, "Component", "Position");
                self.emit_line(&format!(
                    "ctx.set_component({}, '{}', None)",
                    entity, component
                ));
            }
            "RemoveComponent" => {
                let entity = inputs
                    .get("Entity")
                    .cloned()
                    .unwrap_or_else(|| "0".to_string());
                let component = self.get_property_string(node, "Component", "Position");
                self.emit_line(&format!(
                    "ctx.remove_component({}, '{}')",
                    entity, component
                ));
            }
            "QueryEntities" => {
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Entities") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!("{} = []  # Query not implemented in static compile", var));
                }
            }

            // Action nodes
            "SetPosition" => {
                let entity = inputs
                    .get("Entity")
                    .cloned()
                    .unwrap_or_else(|| "0".to_string());
                let pos = inputs
                    .get("Position")
                    .cloned()
                    .unwrap_or_else(|| "(0.0, 0.0, 0.0)".to_string());
                self.emit_line(&format!(
                    "ctx.set_component({}, 'Position', {})",
                    entity, pos
                ));
            }
            "SetRotation" => {
                let entity = inputs
                    .get("Entity")
                    .cloned()
                    .unwrap_or_else(|| "0".to_string());
                let rot = inputs
                    .get("Rotation")
                    .cloned()
                    .unwrap_or_else(|| "(0.0, 0.0, 0.0)".to_string());
                self.emit_line(&format!(
                    "ctx.set_component({}, 'Rotation', {})",
                    entity, rot
                ));
            }
            "SetScale" => {
                let entity = inputs
                    .get("Entity")
                    .cloned()
                    .unwrap_or_else(|| "0".to_string());
                let scale = inputs
                    .get("Scale")
                    .cloned()
                    .unwrap_or_else(|| "(1.0, 1.0, 1.0)".to_string());
                self.emit_line(&format!(
                    "ctx.set_component({}, 'Scale', {})",
                    entity, scale
                ));
            }
            "SpawnEntity" => {
                if let Some(out_port) = node.outputs.iter().find(|p| p.name == "Entity") {
                    let var = self.get_or_create_var(node.id, out_port.id);
                    self.emit_line(&format!(
                        "{} = ctx.frame * 1000 + int(ctx.random_float() * 1000)",
                        var
                    ));
                }
            }
            "DestroyEntity" => {
                let entity = inputs
                    .get("Entity")
                    .cloned()
                    .unwrap_or_else(|| "0".to_string());
                self.emit_line(&format!("ctx.log(f'Destroying entity: {}')", entity));
            }
            "SendMessage" => {
                let msg = self.get_property_string(node, "Message", "Event");
                self.emit_line(&format!("ctx.events.add('{}')", msg));
            }
            "SetVariable" => {
                let name = self.get_property_string(node, "Name", "MyVar");
                let val = inputs
                    .get("Value")
                    .cloned()
                    .unwrap_or_else(|| "None".to_string());
                self.emit_line(&format!("ctx.set_variable('{}', {})", name, val));
            }
            "PlaySound" => {
                let sound = self.get_property_string(node, "Sound", "sound.wav");
                self.emit_line(&format!("ctx.log('Playing sound: {}')", sound));
            }

            // Debug nodes
            "Print" => {
                let val = inputs
                    .get("Value")
                    .cloned()
                    .unwrap_or_else(|| "''".to_string());
                self.emit_line(&format!("ctx.log(f'PRINT: {{{}}}')", val));
            }
            "Log" => {
                let msg = inputs
                    .get("Message")
                    .cloned()
                    .unwrap_or_else(|| "''".to_string());
                let level = self.get_property_enum(node, "Level", &["INFO", "WARN", "ERROR"], 0);
                self.emit_line(&format!("ctx.log(f'[{}] {{{}}}')", level, msg));
            }
            "Assert" => {
                let cond = inputs
                    .get("Condition")
                    .cloned()
                    .unwrap_or_else(|| "True".to_string());
                let msg = self.get_property_string(node, "Message", "Assertion failed");
                self.emit_line(&format!("assert {}, '{}'", cond, msg));
            }
            "Breakpoint" => {
                self.emit_line(&format!("ctx.log('BREAKPOINT hit at node {}')", node.id));
                self.emit_line("# breakpoint() - uncomment for debugging");
            }

            // Unknown node
            _ => {
                self.emit_line(&format!("# Unknown node type: {}", node.name));
                self.warnings.push(format!("Unknown node type: {}", node.name));
            }
        }

        Ok(())
    }

    /// Generate code for a specific flow branch.
    fn generate_flow_branch(
        &mut self,
        graph: &NodeGraph,
        node: &Node,
        port_name: &str,
    ) -> Result<(), CompileError> {
        if let Some(flow_port) = node.outputs.iter().find(|p| p.name == port_name) {
            let connections: Vec<&Connection> = graph
                .connections()
                .iter()
                .filter(|c| c.from_node == node.id && c.from_port == flow_port.id)
                .collect();

            if connections.is_empty() {
                self.emit_line("pass");
            } else {
                for conn in connections {
                    if let Some(next_node) = graph.get_node(conn.to_node) {
                        self.generate_node_code(graph, next_node)?;
                    }
                }
            }
        } else {
            self.emit_line("pass");
        }
        Ok(())
    }

    /// Resolve input values for a node from connections.
    fn resolve_inputs(&mut self, graph: &NodeGraph, node: &Node) -> HashMap<String, String> {
        let mut inputs = HashMap::new();

        for input_port in &node.inputs {
            if input_port.port_type == PortType::Flow {
                continue; // Skip flow ports
            }

            // Find connection to this input
            if let Some(conn) = graph
                .connections()
                .iter()
                .find(|c| c.to_node == node.id && c.to_port == input_port.id)
            {
                // Get the variable name for the source output
                if let Some(var) = self.port_variables.get(&(conn.from_node, conn.from_port)) {
                    inputs.insert(input_port.name.clone(), var.clone());
                } else {
                    // Create a reference to the source node's output
                    let var = self.get_or_create_var(conn.from_node, conn.from_port);
                    inputs.insert(input_port.name.clone(), var);
                }
            } else {
                // No connection - use default based on type
                let default = self.get_default_for_type(&input_port.port_type);
                inputs.insert(input_port.name.clone(), default);
            }
        }

        inputs
    }

    /// Get default value for a port type.
    fn get_default_for_type(&self, port_type: &PortType) -> String {
        match port_type {
            PortType::Flow => "None".to_string(),
            PortType::Bool => "False".to_string(),
            PortType::Int => "0".to_string(),
            PortType::Float => "0.0".to_string(),
            PortType::Vec3 => "(0.0, 0.0, 0.0)".to_string(),
            PortType::String => "''".to_string(),
            PortType::Entity => "0".to_string(),
            PortType::Any => "None".to_string(),
        }
    }

    /// Generate the main execution function.
    fn generate_main_function(&mut self, event_nodes: &[NodeId]) {
        self.emit_line("def execute_graph(ctx):");
        self.indent();
        self.emit_line("\"\"\"Execute all event handlers.\"\"\"");

        if self.functions.is_empty() {
            self.emit_line("pass  # No event handlers");
        } else {
            for func in &self.functions.clone() {
                self.emit_line(&format!("{}(ctx)", func));
            }
        }

        self.dedent();
        self.emit_line("");

        // Generate test main
        self.emit_line("if __name__ == '__main__':");
        self.indent();
        self.emit_line("ctx = FlowForgeContext()");
        self.emit_line("execute_graph(ctx)");
        self.emit_line("print('Execution complete')");
        self.emit_line("print(f'Debug log: {ctx.debug_log}')");
        self.dedent();
    }

    /// Get or create a variable for a port.
    fn get_or_create_var(&mut self, node_id: NodeId, port_id: PortId) -> String {
        if let Some(var) = self.port_variables.get(&(node_id, port_id)) {
            return var.clone();
        }

        let var = self.new_temp_var();
        self.port_variables.insert((node_id, port_id), var.clone());
        var
    }

    /// Generate a new temporary variable name.
    fn new_temp_var(&mut self) -> String {
        let var = format!("_tmp_{}", self.temp_var_counter);
        self.temp_var_counter += 1;
        var
    }

    /// Make a function name for an event node.
    fn make_event_function_name(&self, node: &Node) -> String {
        let base_name = match node.name.as_str() {
            "OnStart" => "on_start",
            "OnTick" => "on_tick",
            "OnKeyPressed" => {
                let key = self.get_property_string(node, "Key", "Space");
                return format!("on_key_pressed_{}", key.to_lowercase());
            }
            "OnKeyReleased" => {
                let key = self.get_property_string(node, "Key", "Space");
                return format!("on_key_released_{}", key.to_lowercase());
            }
            "OnCollision" => "on_collision",
            "CustomEvent" => {
                let event = self.get_property_string(node, "EventName", "Event");
                return format!("on_custom_{}", event.to_lowercase());
            }
            _ => "on_unknown",
        };
        format!("{}_{}", base_name, node.id)
    }

    /// Get a string property value from a node.
    fn get_property_string(&self, node: &Node, name: &str, default: &str) -> String {
        node.get_property(name)
            .and_then(|p| {
                if let PropertyValue::String(s) = &p.value {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| default.to_string())
    }

    /// Get an enum property value from a node.
    fn get_property_enum(&self, node: &Node, name: &str, options: &[&str], default: usize) -> String {
        let idx = node
            .get_property(name)
            .and_then(|p| {
                if let PropertyValue::Enum(i) = p.value {
                    Some(i)
                } else {
                    None
                }
            })
            .unwrap_or(default);

        options.get(idx).unwrap_or(&options[0]).to_string()
    }

    /// Check if a node is an event node.
    fn is_event_node(&self, node: &Node) -> bool {
        matches!(
            node.name.as_str(),
            "OnStart"
                | "OnTick"
                | "OnKeyPressed"
                | "OnKeyReleased"
                | "OnCollision"
                | "CustomEvent"
        )
    }

    /// Check if a node has flow ports.
    fn is_flow_node(&self, node: &Node) -> bool {
        node.inputs.iter().any(|p| p.port_type == PortType::Flow)
            || node.outputs.iter().any(|p| p.port_type == PortType::Flow)
    }

    // -------------------------------------------------------------------------
    // Output helpers
    // -------------------------------------------------------------------------

    /// Emit a line with current indentation.
    fn emit_line(&mut self, line: &str) {
        let indent = "    ".repeat(self.indent_level);
        self.output.push_str(&indent);
        self.output.push_str(line);
        self.output.push('\n');
    }

    /// Increase indentation.
    fn indent(&mut self) {
        self.indent_level += 1;
    }

    /// Decrease indentation.
    fn dedent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level -= 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowforge::{NodeTemplate, PortType, PropertyTemplate, NodeGraph};

    // Helper to create a simple test graph
    fn create_simple_math_graph() -> NodeGraph {
        let mut graph = NodeGraph::new();

        let add_template = NodeTemplate::new("Add", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);

        graph.add_node(&add_template);
        graph
    }

    // -------------------------------------------------------------------------
    // CompileError Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compile_error_new() {
        let err = CompileError::new("test error");
        assert_eq!(err.message, "test error");
        assert!(err.node_id.is_none());
        assert!(err.suggestion.is_none());
    }

    #[test]
    fn test_compile_error_at_node() {
        let err = CompileError::at_node(42, "node error");
        assert_eq!(err.node_id, Some(42));
        assert_eq!(err.message, "node error");
    }

    #[test]
    fn test_compile_error_with_suggestion() {
        let err = CompileError::new("error").with_suggestion("fix it");
        assert_eq!(err.suggestion, Some("fix it".to_string()));
    }

    #[test]
    fn test_compile_error_display() {
        let err = CompileError::at_node(42, "test error").with_suggestion("fix it");
        let display = format!("{}", err);
        assert!(display.contains("Node 42"));
        assert!(display.contains("test error"));
        assert!(display.contains("fix it"));
    }

    // -------------------------------------------------------------------------
    // CompileMetadata Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compile_metadata_default() {
        let meta = CompileMetadata::default();
        assert_eq!(meta.node_count, 0);
        assert_eq!(meta.connection_count, 0);
        assert_eq!(meta.event_count, 0);
        assert_eq!(meta.variable_count, 0);
        assert_eq!(meta.line_count, 0);
        assert!(meta.warnings.is_empty());
    }

    // -------------------------------------------------------------------------
    // CompiledGraph Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compiled_graph_new() {
        let graph = CompiledGraph::new(
            "print('hello')".to_string(),
            vec!["main".to_string()],
            vec!["import math".to_string()],
            CompileMetadata::default(),
        );

        assert_eq!(graph.python_source, "print('hello')");
        assert_eq!(graph.entry_points, vec!["main"]);
        assert_eq!(graph.dependencies, vec!["import math"]);
    }

    #[test]
    fn test_compiled_graph_full_source() {
        let graph = CompiledGraph::new(
            "print('hello')".to_string(),
            vec![],
            vec!["import math".to_string(), "import random".to_string()],
            CompileMetadata::default(),
        );

        let full = graph.full_source();
        assert!(full.contains("import math"));
        assert!(full.contains("import random"));
        assert!(full.contains("print('hello')"));
    }

    // -------------------------------------------------------------------------
    // GraphCompiler Basic Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compiler_new() {
        let compiler = GraphCompiler::new();
        assert_eq!(compiler.indent_level, 0);
        assert!(compiler.output.is_empty());
        assert_eq!(compiler.temp_var_counter, 0);
    }

    #[test]
    fn test_compiler_default() {
        let compiler = GraphCompiler::default();
        assert_eq!(compiler.indent_level, 0);
    }

    #[test]
    fn test_compiler_reset() {
        let mut compiler = GraphCompiler::new();
        compiler.indent_level = 5;
        compiler.temp_var_counter = 10;
        compiler.output = "some code".to_string();

        compiler.reset();

        assert_eq!(compiler.indent_level, 0);
        assert_eq!(compiler.temp_var_counter, 0);
        assert!(compiler.output.is_empty());
    }

    #[test]
    fn test_compiler_compile_empty_graph() {
        let mut compiler = GraphCompiler::new();
        let graph = NodeGraph::new();

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(compiled.entry_points.is_empty());
        assert!(compiled.metadata.node_count == 0);
    }

    // -------------------------------------------------------------------------
    // Math Node Compilation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compile_add_node() {
        let mut compiler = GraphCompiler::new();
        let graph = create_simple_math_graph();

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
        // No event nodes, so no entry points
    }

    #[test]
    fn test_compile_math_chain() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        // Create: OnStart -> Add(3, 4) -> print
        let on_start = NodeTemplate::new("OnStart", "Events")
            .with_output("Out", PortType::Flow);

        let add = NodeTemplate::new("Add", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);

        let print_template = NodeTemplate::new("Print", "Debug")
            .with_input("In", PortType::Flow)
            .with_input("Value", PortType::Any)
            .with_output("Out", PortType::Flow);

        let start_id = graph.add_node(&on_start);
        let add_id = graph.add_node(&add);
        let print_id = graph.add_node(&print_template);

        // Connect flow: OnStart.Out -> Print.In
        let start_out = graph.get_node(start_id).unwrap().outputs[0].id;
        let print_in = graph.get_node(print_id).unwrap().inputs[0].id;
        graph.connect((start_id, start_out), (print_id, print_in)).unwrap();

        // Connect data: Add.Result -> Print.Value
        let add_out = graph.get_node(add_id).unwrap().outputs[0].id;
        let print_val = graph.get_node(print_id).unwrap().inputs[1].id;
        graph.connect((add_id, add_out), (print_id, print_val)).unwrap();

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(!compiled.entry_points.is_empty());
        assert!(compiled.python_source.contains("on_start"));
    }

    #[test]
    fn test_compile_subtract_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("Subtract", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_multiply_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("Multiply", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_divide_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("Divide", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_sin_cos_nodes() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let sin_template = NodeTemplate::new("Sin", "Math")
            .with_input("Angle", PortType::Float)
            .with_output("Result", PortType::Float);

        let cos_template = NodeTemplate::new("Cos", "Math")
            .with_input("Angle", PortType::Float)
            .with_output("Result", PortType::Float);

        graph.add_node(&sin_template);
        graph.add_node(&cos_template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(compiled.dependencies.iter().any(|d| d.contains("math")));
    }

    #[test]
    fn test_compile_lerp_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("Lerp", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_input("T", PortType::Float)
            .with_output("Result", PortType::Float);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_clamp_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("Clamp", "Math")
            .with_input("Value", PortType::Float)
            .with_input("Min", PortType::Float)
            .with_input("Max", PortType::Float)
            .with_output("Result", PortType::Float);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_random_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("Random", "Math")
            .with_output("Value", PortType::Float);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Flow Control Compilation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compile_branch_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let on_start = NodeTemplate::new("OnStart", "Events")
            .with_output("Out", PortType::Flow);

        let branch = NodeTemplate::new("Branch", "Flow")
            .with_input("In", PortType::Flow)
            .with_input("Condition", PortType::Bool)
            .with_output("True", PortType::Flow)
            .with_output("False", PortType::Flow);

        let start_id = graph.add_node(&on_start);
        let branch_id = graph.add_node(&branch);

        let start_out = graph.get_node(start_id).unwrap().outputs[0].id;
        let branch_in = graph.get_node(branch_id).unwrap().inputs[0].id;
        graph.connect((start_id, start_out), (branch_id, branch_in)).unwrap();

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(compiled.python_source.contains("if"));
        assert!(compiled.python_source.contains("else"));
    }

    #[test]
    fn test_compile_for_loop_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let on_start = NodeTemplate::new("OnStart", "Events")
            .with_output("Out", PortType::Flow);

        let for_loop = NodeTemplate::new("ForLoop", "Flow")
            .with_input("In", PortType::Flow)
            .with_input("Count", PortType::Int)
            .with_output("Loop", PortType::Flow)
            .with_output("Index", PortType::Int)
            .with_output("Completed", PortType::Flow);

        let start_id = graph.add_node(&on_start);
        let loop_id = graph.add_node(&for_loop);

        let start_out = graph.get_node(start_id).unwrap().outputs[0].id;
        let loop_in = graph.get_node(loop_id).unwrap().inputs[0].id;
        graph.connect((start_id, start_out), (loop_id, loop_in)).unwrap();

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(compiled.python_source.contains("for"));
        assert!(compiled.python_source.contains("range"));
    }

    #[test]
    fn test_compile_while_loop_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let on_start = NodeTemplate::new("OnStart", "Events")
            .with_output("Out", PortType::Flow);

        let while_loop = NodeTemplate::new("WhileLoop", "Flow")
            .with_input("In", PortType::Flow)
            .with_input("Condition", PortType::Bool)
            .with_output("Loop", PortType::Flow)
            .with_output("Completed", PortType::Flow);

        let start_id = graph.add_node(&on_start);
        let loop_id = graph.add_node(&while_loop);

        let start_out = graph.get_node(start_id).unwrap().outputs[0].id;
        let loop_in = graph.get_node(loop_id).unwrap().inputs[0].id;
        graph.connect((start_id, start_out), (loop_id, loop_in)).unwrap();

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(compiled.python_source.contains("while"));
    }

    #[test]
    fn test_compile_sequence_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("Sequence", "Flow")
            .with_input("In", PortType::Flow)
            .with_output("1", PortType::Flow)
            .with_output("2", PortType::Flow)
            .with_output("3", PortType::Flow);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Event Handler Compilation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compile_on_start_event() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("OnStart", "Events")
            .with_output("Out", PortType::Flow);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(!compiled.entry_points.is_empty());
        assert!(compiled.entry_points[0].contains("on_start"));
    }

    #[test]
    fn test_compile_on_tick_event() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("OnTick", "Events")
            .with_output("Out", PortType::Flow)
            .with_output("DeltaTime", PortType::Float);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(compiled.python_source.contains("delta_time"));
    }

    #[test]
    fn test_compile_on_key_pressed_event() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("OnKeyPressed", "Events")
            .with_output("Out", PortType::Flow)
            .with_property(PropertyTemplate::string("Key", "Space"));

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(compiled.entry_points[0].contains("key_pressed"));
    }

    #[test]
    fn test_compile_custom_event() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("CustomEvent", "Events")
            .with_output("Out", PortType::Flow)
            .with_property(PropertyTemplate::string("EventName", "MyEvent"));

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(compiled.entry_points[0].contains("custom"));
    }

    // -------------------------------------------------------------------------
    // Variable Scoping Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_temp_var_generation() {
        let mut compiler = GraphCompiler::new();

        let var1 = compiler.new_temp_var();
        let var2 = compiler.new_temp_var();
        let var3 = compiler.new_temp_var();

        assert_eq!(var1, "_tmp_0");
        assert_eq!(var2, "_tmp_1");
        assert_eq!(var3, "_tmp_2");
    }

    #[test]
    fn test_get_or_create_var() {
        let mut compiler = GraphCompiler::new();

        let var1 = compiler.get_or_create_var(1, 10);
        let var2 = compiler.get_or_create_var(1, 10); // Same node/port
        let var3 = compiler.get_or_create_var(2, 20); // Different node/port

        assert_eq!(var1, var2); // Should reuse variable
        assert_ne!(var1, var3); // Should be different
    }

    // -------------------------------------------------------------------------
    // ECS Node Compilation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compile_get_component() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("GetComponent", "ECS")
            .with_input("Entity", PortType::Entity)
            .with_output("Value", PortType::Any)
            .with_property(PropertyTemplate::string("Component", "Health"));

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_set_component() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("SetComponent", "ECS")
            .with_input("In", PortType::Flow)
            .with_input("Entity", PortType::Entity)
            .with_input("Value", PortType::Any)
            .with_output("Out", PortType::Flow)
            .with_property(PropertyTemplate::string("Component", "Health"));

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_has_component() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("HasComponent", "ECS")
            .with_input("Entity", PortType::Entity)
            .with_output("Result", PortType::Bool)
            .with_property(PropertyTemplate::string("Component", "Health"));

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Error Handling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compile_unknown_node_warning() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        // Create a node with an unknown type
        let template = NodeTemplate::new("UnknownNodeType", "Test");
        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(!compiled.metadata.warnings.is_empty());
    }

    #[test]
    fn test_compile_delay_warning() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let on_start = NodeTemplate::new("OnStart", "Events")
            .with_output("Out", PortType::Flow);

        let delay = NodeTemplate::new("Delay", "Flow")
            .with_input("In", PortType::Flow)
            .with_input("Duration", PortType::Float)
            .with_output("Out", PortType::Flow);

        let start_id = graph.add_node(&on_start);
        let delay_id = graph.add_node(&delay);

        let start_out = graph.get_node(start_id).unwrap().outputs[0].id;
        let delay_in = graph.get_node(delay_id).unwrap().inputs[0].id;
        graph.connect((start_id, start_out), (delay_id, delay_in)).unwrap();

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(compiled.metadata.warnings.iter().any(|w| w.contains("Delay")));
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compile_full_program() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        // OnStart -> SetVariable("score", 0) -> Print(score)
        let on_start = NodeTemplate::new("OnStart", "Events")
            .with_output("Out", PortType::Flow);

        let set_var = NodeTemplate::new("SetVariable", "Variables")
            .with_input("In", PortType::Flow)
            .with_input("Value", PortType::Any)
            .with_output("Out", PortType::Flow)
            .with_property(PropertyTemplate::string("Name", "score"));

        let print_template = NodeTemplate::new("Print", "Debug")
            .with_input("In", PortType::Flow)
            .with_input("Value", PortType::Any)
            .with_output("Out", PortType::Flow);

        let start_id = graph.add_node(&on_start);
        let set_id = graph.add_node(&set_var);
        let print_id = graph.add_node(&print_template);

        // Connect OnStart -> SetVariable
        let start_out = graph.get_node(start_id).unwrap().outputs[0].id;
        let set_in = graph.get_node(set_id).unwrap().inputs[0].id;
        graph.connect((start_id, start_out), (set_id, set_in)).unwrap();

        // Connect SetVariable -> Print
        let set_out = graph.get_node(set_id).unwrap().outputs[0].id;
        let print_in = graph.get_node(print_id).unwrap().inputs[0].id;
        graph.connect((set_id, set_out), (print_id, print_in)).unwrap();

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert_eq!(compiled.entry_points.len(), 1);
        assert!(compiled.python_source.contains("set_variable"));
        assert!(compiled.python_source.contains("ctx.log"));
    }

    #[test]
    fn test_compile_generates_valid_python() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let on_start = NodeTemplate::new("OnStart", "Events")
            .with_output("Out", PortType::Flow);

        graph.add_node(&on_start);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        let source = compiled.full_source();

        // Check basic structure
        assert!(source.contains("class FlowForgeContext"));
        assert!(source.contains("def __init__(self)"));
        assert!(source.contains("def execute_graph(ctx)"));
        assert!(source.contains("if __name__ == '__main__'"));
    }

    #[test]
    fn test_compile_metadata() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let on_start = NodeTemplate::new("OnStart", "Events")
            .with_output("Out", PortType::Flow);

        let add = NodeTemplate::new("Add", "Math")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Result", PortType::Float);

        graph.add_node(&on_start);
        graph.add_node(&add);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert_eq!(compiled.metadata.node_count, 2);
        assert_eq!(compiled.metadata.event_count, 1);
        assert!(compiled.metadata.line_count > 0);
    }

    // -------------------------------------------------------------------------
    // Debug Node Compilation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compile_print_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("Print", "Debug")
            .with_input("In", PortType::Flow)
            .with_input("Value", PortType::Any)
            .with_output("Out", PortType::Flow);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_assert_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let on_start = NodeTemplate::new("OnStart", "Events")
            .with_output("Out", PortType::Flow);

        let assert_template = NodeTemplate::new("Assert", "Debug")
            .with_input("In", PortType::Flow)
            .with_input("Condition", PortType::Bool)
            .with_output("Out", PortType::Flow)
            .with_property(PropertyTemplate::string("Message", "Test assertion"));

        let start_id = graph.add_node(&on_start);
        let assert_id = graph.add_node(&assert_template);

        let start_out = graph.get_node(start_id).unwrap().outputs[0].id;
        let assert_in = graph.get_node(assert_id).unwrap().inputs[0].id;
        graph.connect((start_id, start_out), (assert_id, assert_in)).unwrap();

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(compiled.python_source.contains("assert"));
    }

    #[test]
    fn test_compile_breakpoint_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("Breakpoint", "Debug")
            .with_input("In", PortType::Flow)
            .with_output("Out", PortType::Flow);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Logic Node Compilation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compile_and_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("And", "Logic")
            .with_input("A", PortType::Bool)
            .with_input("B", PortType::Bool)
            .with_output("Result", PortType::Bool);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_or_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("Or", "Logic")
            .with_input("A", PortType::Bool)
            .with_input("B", PortType::Bool)
            .with_output("Result", PortType::Bool);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_not_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("Not", "Logic")
            .with_input("Value", PortType::Bool)
            .with_output("Result", PortType::Bool);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_compare_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("Compare", "Logic")
            .with_input("A", PortType::Float)
            .with_input("B", PortType::Float)
            .with_output("Less", PortType::Bool)
            .with_output("Equal", PortType::Bool)
            .with_output("Greater", PortType::Bool);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_select_node() {
        let mut compiler = GraphCompiler::new();
        let mut graph = NodeGraph::new();

        let template = NodeTemplate::new("Select", "Logic")
            .with_input("Condition", PortType::Bool)
            .with_input("True", PortType::Any)
            .with_input("False", PortType::Any)
            .with_output("Result", PortType::Any);

        graph.add_node(&template);

        let result = compiler.compile(&graph);
        assert!(result.is_ok());
    }
}
