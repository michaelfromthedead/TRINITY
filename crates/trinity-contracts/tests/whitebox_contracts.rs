//! Whitebox tests for trinity-contracts.

use trinity_contracts::{
    AlgebraicProperty, Constraint, Contract, ContractResult, ContractSchema,
    ContractViolation, LayoutConstraint, ViolationKind,
};

// ==================== Contract ====================

#[test]
fn test_contract_new() {
    let contract = Contract::new("my_func");
    assert_eq!(contract.function, "my_func");
    assert!(contract.requires.is_empty());
    assert!(contract.ensures.is_empty());
}

#[test]
fn test_contract_requires() {
    let contract = Contract::new("func")
        .requires(Constraint::new("x > 0"));
    
    assert_eq!(contract.requires.len(), 1);
    assert_eq!(contract.requires[0].expression, "x > 0");
}

#[test]
fn test_contract_ensures() {
    let contract = Contract::new("func")
        .ensures(Constraint::new("result > 0"));
    
    assert_eq!(contract.ensures.len(), 1);
    assert_eq!(contract.ensures[0].expression, "result > 0");
}

#[test]
fn test_contract_invariant() {
    let contract = Contract::new("func")
        .invariant(Constraint::new("self.len() > 0"));
    
    assert_eq!(contract.invariants.len(), 1);
}

#[test]
fn test_contract_layout() {
    let contract = Contract::new("func")
        .layout(LayoutConstraint::new().size(16).align(8));
    
    assert!(contract.layout.is_some());
    assert_eq!(contract.layout.as_ref().unwrap().size, Some(16));
    assert_eq!(contract.layout.as_ref().unwrap().align, Some(8));
}

// ==================== Constraint ====================

#[test]
fn test_constraint_new() {
    let constraint = Constraint::new("x > 0");
    assert_eq!(constraint.expression, "x > 0");
    assert!(constraint.message.is_none());
    assert!(constraint.is_satisfied);
}

#[test]
fn test_constraint_message() {
    let constraint = Constraint::new("x > 0")
        .message("x must be positive");
    
    assert_eq!(constraint.message, Some("x must be positive".to_string()));
}

#[test]
fn test_constraint_satisfied() {
    let constraint = Constraint::new("x > 0").satisfied(false);
    assert!(!constraint.is_satisfied);
}

// ==================== LayoutConstraint ====================

#[test]
fn test_layout_constraint_new() {
    let layout = LayoutConstraint::new();
    assert!(layout.size.is_none());
    assert!(layout.align.is_none());
}

#[test]
fn test_layout_constraint_size() {
    let layout = LayoutConstraint::new().size(64);
    assert_eq!(layout.size, Some(64));
}

#[test]
fn test_layout_constraint_align() {
    let layout = LayoutConstraint::new().align(16);
    assert_eq!(layout.align, Some(16));
}

// ==================== ContractResult ====================

#[test]
fn test_contract_result_new() {
    let result = ContractResult::new();
    assert!(result.passed);
    assert!(result.violations.is_empty());
}

#[test]
fn test_contract_result_add_violation() {
    let mut result = ContractResult::new();
    result.add_violation(ContractViolation {
        kind: ViolationKind::Precondition,
        constraint: "x > 0".to_string(),
        message: None,
    });
    
    assert!(!result.passed);
    assert_eq!(result.violations.len(), 1);
}

#[test]
fn test_contract_result_has_violations() {
    let mut result = ContractResult::new();
    assert!(!result.has_violations());
    
    result.add_violation(ContractViolation {
        kind: ViolationKind::Postcondition,
        constraint: "result != 0".to_string(),
        message: Some("Result was zero".to_string()),
    });
    
    assert!(result.has_violations());
}

// ==================== ViolationKind ====================

#[test]
fn test_violation_kinds() {
    let kinds = vec![
        ViolationKind::Precondition,
        ViolationKind::Postcondition,
        ViolationKind::Invariant,
        ViolationKind::Layout,
    ];
    assert_eq!(kinds.len(), 4);
}

// ==================== AlgebraicProperty ====================

#[test]
fn test_algebraic_properties() {
    let props = vec![
        AlgebraicProperty::Commutative,
        AlgebraicProperty::Associative,
        AlgebraicProperty::Idempotent,
        AlgebraicProperty::Identity,
        AlgebraicProperty::Inverse,
    ];
    assert_eq!(props.len(), 5);
}

// ==================== ContractSchema ====================

#[test]
fn test_schema_new() {
    let schema = ContractSchema::new();
    assert_eq!(schema.version, "1.0");
    assert!(schema.contracts.is_empty());
}

#[test]
fn test_schema_add() {
    let mut schema = ContractSchema::new();
    schema.add(Contract::new("func1"));
    schema.add(Contract::new("func2"));
    
    assert_eq!(schema.contracts.len(), 2);
}

#[test]
fn test_schema_to_json() {
    let mut schema = ContractSchema::new();
    schema.add(Contract::new("test_func"));
    
    let json = schema.to_json().unwrap();
    
    assert!(json.contains("test_func"));
    assert!(json.contains("version"));
}

#[test]
fn test_schema_from_json() {
    let json = r#"{
        "version": "1.0",
        "contracts": [
            {"function": "my_func", "requires": [], "ensures": [], "invariants": [], "layout": null}
        ]
    }"#;
    
    let schema = ContractSchema::from_json(json).unwrap();
    
    assert_eq!(schema.contracts.len(), 1);
    assert_eq!(schema.contracts[0].function, "my_func");
}
