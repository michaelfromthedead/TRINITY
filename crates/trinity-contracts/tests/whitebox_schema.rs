//! Whitebox tests for schema extraction.

use trinity_contracts::schema::{
    infer_type, parse_constraint, Constraint, ConstraintKind, ConstraintSchema, ContractTable,
    ParamSchema, SchemaType,
};

// ==================== ConstraintSchema ====================

#[test]
fn test_schema_new() {
    let schema = ConstraintSchema::new();
    assert_eq!(schema.version, "1.0");
    assert!(schema.parameters.is_empty());
}

#[test]
fn test_schema_add_param() {
    let mut schema = ConstraintSchema::new();
    schema.add_param("x", ParamSchema::new(SchemaType::Integer));
    assert!(schema.parameters.contains_key("x"));
}

#[test]
fn test_schema_set_returns() {
    let mut schema = ConstraintSchema::new();
    schema.set_returns(ParamSchema::new(SchemaType::Boolean));
    assert!(schema.returns.is_some());
}

#[test]
fn test_schema_to_json() {
    let mut schema = ConstraintSchema::new();
    schema.add_param("x", ParamSchema::new(SchemaType::Integer));
    let json = schema.to_json().unwrap();
    assert!(json.contains("\"x\""));
    assert!(json.contains("integer"));
}

#[test]
fn test_schema_from_json() {
    let json = r#"{"version":"1.0","parameters":{},"returns":null}"#;
    let schema = ConstraintSchema::from_json(json).unwrap();
    assert_eq!(schema.version, "1.0");
}

// ==================== ParamSchema ====================

#[test]
fn test_param_schema_new() {
    let schema = ParamSchema::new(SchemaType::Integer);
    assert_eq!(schema.param_type, SchemaType::Integer);
    assert!(schema.constraints.is_empty());
}

#[test]
fn test_param_schema_constraint() {
    let schema = ParamSchema::new(SchemaType::Integer)
        .constraint(Constraint::min(0))
        .constraint(Constraint::max(100));
    assert_eq!(schema.constraints.len(), 2);
}

// ==================== SchemaType ====================

#[test]
fn test_schema_types() {
    let types = vec![
        SchemaType::Integer,
        SchemaType::Float,
        SchemaType::String,
        SchemaType::Boolean,
        SchemaType::Array,
        SchemaType::Object,
        SchemaType::Any,
    ];
    assert_eq!(types.len(), 7);
}

// ==================== Constraint ====================

#[test]
fn test_constraint_min() {
    let c = Constraint::min(5);
    assert_eq!(c.kind, ConstraintKind::Min);
}

#[test]
fn test_constraint_max() {
    let c = Constraint::max(100);
    assert_eq!(c.kind, ConstraintKind::Max);
}

#[test]
fn test_constraint_non_zero() {
    let c = Constraint::non_zero();
    assert_eq!(c.kind, ConstraintKind::NonZero);
}

#[test]
fn test_constraint_non_empty() {
    let c = Constraint::non_empty();
    assert_eq!(c.kind, ConstraintKind::NonEmpty);
}

#[test]
fn test_constraint_pattern() {
    let c = Constraint::pattern(r"^\d+$");
    assert_eq!(c.kind, ConstraintKind::Pattern);
}

// ==================== parse_constraint ====================

#[test]
fn test_parse_positive() {
    let constraints = parse_constraint("x > 0");
    assert!(constraints.iter().any(|c| c.kind == ConstraintKind::Min));
}

#[test]
fn test_parse_non_zero() {
    let constraints = parse_constraint("x != 0");
    assert!(constraints.iter().any(|c| c.kind == ConstraintKind::NonZero));
}

#[test]
fn test_parse_non_empty() {
    let constraints = parse_constraint("!s.is_empty()");
    assert!(constraints.iter().any(|c| c.kind == ConstraintKind::NonEmpty));
}

#[test]
fn test_parse_range() {
    let constraints = parse_constraint("x >= 10 && x <= 20");
    assert!(constraints.len() >= 2);
}

// ==================== infer_type ====================

#[test]
fn test_infer_integer() {
    assert_eq!(infer_type("i32"), SchemaType::Integer);
    assert_eq!(infer_type("u64"), SchemaType::Integer);
}

#[test]
fn test_infer_float() {
    assert_eq!(infer_type("f32"), SchemaType::Float);
    assert_eq!(infer_type("f64"), SchemaType::Float);
}

#[test]
fn test_infer_string() {
    assert_eq!(infer_type("String"), SchemaType::String);
    assert_eq!(infer_type("&str"), SchemaType::String);
}

#[test]
fn test_infer_boolean() {
    assert_eq!(infer_type("bool"), SchemaType::Boolean);
}

#[test]
fn test_infer_array() {
    assert_eq!(infer_type("Vec<i32>"), SchemaType::Array);
}

// ==================== ContractTable ====================

#[test]
fn test_table_new() {
    let table = ContractTable::new();
    assert!(table.is_empty());
}

#[test]
fn test_table_store_get() {
    let mut table = ContractTable::new();
    table.store("my_func", ConstraintSchema::new());
    assert!(table.get("my_func").is_some());
    assert!(table.get("other").is_none());
}

#[test]
fn test_table_functions() {
    let mut table = ContractTable::new();
    table.store("func_a", ConstraintSchema::new());
    table.store("func_b", ConstraintSchema::new());
    assert_eq!(table.len(), 2);
}

#[test]
fn test_table_export_json() {
    let mut table = ContractTable::new();
    table.store("test", ConstraintSchema::new());
    let json = table.export_json().unwrap();
    assert!(json.contains("test"));
}
