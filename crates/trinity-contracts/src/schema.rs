//! Schema extraction for contract constraints.
//!
//! Converts contract constraints to JSON schemas for data generation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A constraint schema for data generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSchema {
    /// Schema version.
    pub version: String,
    /// Parameter schemas.
    pub parameters: HashMap<String, ParamSchema>,
    /// Return value schema.
    pub returns: Option<ParamSchema>,
}

impl ConstraintSchema {
    /// Create a new empty schema.
    pub fn new() -> Self {
        Self {
            version: "1.0".to_string(),
            parameters: HashMap::new(),
            returns: None,
        }
    }

    /// Add a parameter schema.
    pub fn add_param(&mut self, name: impl Into<String>, schema: ParamSchema) {
        self.parameters.insert(name.into(), schema);
    }

    /// Set return schema.
    pub fn set_returns(&mut self, schema: ParamSchema) {
        self.returns = Some(schema);
    }

    /// Convert to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parse from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl Default for ConstraintSchema {
    fn default() -> Self {
        Self::new()
    }
}

/// Schema for a single parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamSchema {
    /// Parameter type.
    #[serde(rename = "type")]
    pub param_type: SchemaType,
    /// Constraints on the parameter.
    pub constraints: Vec<Constraint>,
}

impl ParamSchema {
    /// Create a new parameter schema.
    pub fn new(param_type: SchemaType) -> Self {
        Self {
            param_type,
            constraints: Vec::new(),
        }
    }

    /// Add a constraint.
    pub fn constraint(mut self, c: Constraint) -> Self {
        self.constraints.push(c);
        self
    }
}

/// Schema type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SchemaType {
    /// Integer type.
    Integer,
    /// Float type.
    Float,
    /// String type.
    String,
    /// Boolean type.
    Boolean,
    /// Array type.
    Array,
    /// Object type.
    Object,
    /// Any type.
    Any,
}

/// A constraint on a parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// Constraint kind.
    pub kind: ConstraintKind,
    /// Constraint value.
    pub value: serde_json::Value,
}

impl Constraint {
    /// Create a minimum constraint.
    pub fn min(value: i64) -> Self {
        Self {
            kind: ConstraintKind::Min,
            value: serde_json::Value::Number(value.into()),
        }
    }

    /// Create a maximum constraint.
    pub fn max(value: i64) -> Self {
        Self {
            kind: ConstraintKind::Max,
            value: serde_json::Value::Number(value.into()),
        }
    }

    /// Create a non-zero constraint.
    pub fn non_zero() -> Self {
        Self {
            kind: ConstraintKind::NonZero,
            value: serde_json::Value::Bool(true),
        }
    }

    /// Create a non-empty constraint.
    pub fn non_empty() -> Self {
        Self {
            kind: ConstraintKind::NonEmpty,
            value: serde_json::Value::Bool(true),
        }
    }

    /// Create an enum constraint.
    pub fn one_of(values: Vec<serde_json::Value>) -> Self {
        Self {
            kind: ConstraintKind::OneOf,
            value: serde_json::Value::Array(values),
        }
    }

    /// Create a pattern constraint.
    pub fn pattern(regex: impl Into<String>) -> Self {
        Self {
            kind: ConstraintKind::Pattern,
            value: serde_json::Value::String(regex.into()),
        }
    }
}

/// Constraint kind.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintKind {
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Non-zero.
    NonZero,
    /// Non-empty.
    NonEmpty,
    /// One of values.
    OneOf,
    /// Regex pattern.
    Pattern,
    /// Custom expression.
    Custom,
}

/// Parse a requires expression into constraints.
pub fn parse_constraint(expr: &str) -> Vec<Constraint> {
    let mut constraints = Vec::new();

    // Simple pattern matching
    if expr.contains("> 0") || expr.contains(">= 1") {
        constraints.push(Constraint::min(1));
    }
    if expr.contains("< 0") || expr.contains("<= -1") {
        constraints.push(Constraint::max(-1));
    }
    if expr.contains("!= 0") {
        constraints.push(Constraint::non_zero());
    }
    if expr.contains("is_empty()") && expr.contains('!') {
        constraints.push(Constraint::non_empty());
    }

    // Range patterns: x >= N && x <= M
    if let Some(min) = extract_min_bound(expr) {
        constraints.push(Constraint::min(min));
    }
    if let Some(max) = extract_max_bound(expr) {
        constraints.push(Constraint::max(max));
    }

    constraints
}

fn extract_min_bound(expr: &str) -> Option<i64> {
    // Look for >= N or > N patterns
    for pattern in &[">= ", "> "] {
        if let Some(idx) = expr.find(pattern) {
            let after = &expr[idx + pattern.len()..];
            let num_str: String = after.chars().take_while(|c| c.is_ascii_digit() || *c == '-').collect();
            if let Ok(n) = num_str.parse::<i64>() {
                return Some(if pattern == &"> " { n + 1 } else { n });
            }
        }
    }
    None
}

fn extract_max_bound(expr: &str) -> Option<i64> {
    // Look for <= N or < N patterns
    for pattern in &["<= ", "< "] {
        if let Some(idx) = expr.find(pattern) {
            let after = &expr[idx + pattern.len()..];
            let num_str: String = after.chars().take_while(|c| c.is_ascii_digit() || *c == '-').collect();
            if let Ok(n) = num_str.parse::<i64>() {
                return Some(if pattern == &"< " { n - 1 } else { n });
            }
        }
    }
    None
}

/// Infer schema type from Rust type name.
pub fn infer_type(rust_type: &str) -> SchemaType {
    match rust_type {
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => SchemaType::Integer,
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => SchemaType::Integer,
        "f32" | "f64" => SchemaType::Float,
        "bool" => SchemaType::Boolean,
        "String" | "&str" | "str" => SchemaType::String,
        s if s.starts_with("Vec<") || s.starts_with('[') => SchemaType::Array,
        _ => SchemaType::Any,
    }
}

/// Contract table for storing extracted schemas.
#[derive(Debug, Default)]
pub struct ContractTable {
    /// Contracts by function name.
    contracts: HashMap<String, ConstraintSchema>,
}

impl ContractTable {
    /// Create a new empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a contract schema.
    pub fn store(&mut self, function: impl Into<String>, schema: ConstraintSchema) {
        self.contracts.insert(function.into(), schema);
    }

    /// Get a contract schema.
    pub fn get(&self, function: &str) -> Option<&ConstraintSchema> {
        self.contracts.get(function)
    }

    /// List all functions.
    pub fn functions(&self) -> Vec<&str> {
        self.contracts.keys().map(|s| s.as_str()).collect()
    }

    /// Get contract count.
    pub fn len(&self) -> usize {
        self.contracts.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.contracts.is_empty()
    }

    /// Export all schemas to JSON.
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.contracts)
    }

    /// Import schemas from JSON.
    pub fn import_json(&mut self, json: &str) -> Result<(), serde_json::Error> {
        let imported: HashMap<String, ConstraintSchema> = serde_json::from_str(json)?;
        self.contracts.extend(imported);
        Ok(())
    }
}
