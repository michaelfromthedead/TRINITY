//! Blackbox tests for schema extraction with real use cases.

use trinity_contracts::schema::{
    infer_type, parse_constraint, Constraint, ConstraintSchema, ContractTable, ParamSchema,
    SchemaType,
};

#[test]
fn test_full_schema_workflow() {
    // Build a schema for a clamp function
    let mut schema = ConstraintSchema::new();

    // Add parameters
    schema.add_param(
        "value",
        ParamSchema::new(SchemaType::Integer),
    );
    schema.add_param(
        "min",
        ParamSchema::new(SchemaType::Integer),
    );
    schema.add_param(
        "max",
        ParamSchema::new(SchemaType::Integer)
            .constraint(Constraint::min(0)),
    );

    // Add return
    schema.set_returns(ParamSchema::new(SchemaType::Integer));

    // Serialize
    let json = schema.to_json().unwrap();
    assert!(json.contains("value"));
    assert!(json.contains("min"));
    assert!(json.contains("max"));

    // Deserialize
    let parsed = ConstraintSchema::from_json(&json).unwrap();
    assert_eq!(parsed.parameters.len(), 3);
}

#[test]
fn test_contract_table_workflow() {
    let mut table = ContractTable::new();

    // Store multiple contracts
    let mut div_schema = ConstraintSchema::new();
    div_schema.add_param(
        "a",
        ParamSchema::new(SchemaType::Integer),
    );
    div_schema.add_param(
        "b",
        ParamSchema::new(SchemaType::Integer)
            .constraint(Constraint::non_zero()),
    );
    table.store("safe_divide", div_schema);

    let mut sqrt_schema = ConstraintSchema::new();
    sqrt_schema.add_param(
        "x",
        ParamSchema::new(SchemaType::Float)
            .constraint(Constraint::min(0)),
    );
    table.store("sqrt", sqrt_schema);

    // Verify storage
    assert_eq!(table.len(), 2);
    assert!(table.get("safe_divide").is_some());
    assert!(table.get("sqrt").is_some());

    // Export and verify
    let json = table.export_json().unwrap();
    assert!(json.contains("safe_divide"));
    assert!(json.contains("sqrt"));
}

#[test]
fn test_constraint_parsing_from_requires() {
    // Common contract patterns
    let positive = parse_constraint("x > 0");
    assert!(!positive.is_empty());

    let bounded = parse_constraint("age >= 0 && age <= 150");
    assert!(bounded.len() >= 2);

    let non_empty = parse_constraint("!name.is_empty()");
    assert!(!non_empty.is_empty());

    let divisor = parse_constraint("divisor != 0");
    assert!(!divisor.is_empty());
}

#[test]
fn test_type_inference() {
    // Common Rust types
    assert_eq!(infer_type("i32"), SchemaType::Integer);
    assert_eq!(infer_type("f64"), SchemaType::Float);
    assert_eq!(infer_type("String"), SchemaType::String);
    assert_eq!(infer_type("bool"), SchemaType::Boolean);
    assert_eq!(infer_type("Vec<u8>"), SchemaType::Array);

    // Unknown types
    assert_eq!(infer_type("MyCustomType"), SchemaType::Any);
}

#[test]
fn test_schema_roundtrip() {
    let mut original = ConstraintSchema::new();
    original.add_param(
        "input",
        ParamSchema::new(SchemaType::String)
            .constraint(Constraint::non_empty())
            .constraint(Constraint::pattern(r"^[a-z]+$")),
    );

    let json = original.to_json().unwrap();
    let restored = ConstraintSchema::from_json(&json).unwrap();

    assert_eq!(restored.parameters.len(), original.parameters.len());
    let input = restored.parameters.get("input").unwrap();
    assert_eq!(input.param_type, SchemaType::String);
    assert_eq!(input.constraints.len(), 2);
}

#[test]
fn test_complex_contract() {
    let mut schema = ConstraintSchema::new();

    // Function: sort(arr: Vec<i32>, ascending: bool) -> Vec<i32>
    schema.add_param(
        "arr",
        ParamSchema::new(SchemaType::Array),
    );
    schema.add_param(
        "ascending",
        ParamSchema::new(SchemaType::Boolean),
    );
    schema.set_returns(ParamSchema::new(SchemaType::Array));

    assert_eq!(schema.parameters.len(), 2);
    assert!(schema.returns.is_some());
}
