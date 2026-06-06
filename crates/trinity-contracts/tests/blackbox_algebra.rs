//! Blackbox tests for algebraic properties with real operations.

use trinity_contracts::algebra::{
    verify_associative, verify_commutative, verify_idempotent, verify_identity, verify_involutory,
    Property, PropertySpec, PropertyTestGenerator,
};

// Real math operations
fn add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }
fn mul(a: i32, b: i32) -> i32 { a.wrapping_mul(b) }
fn max(a: i32, b: i32) -> i32 { a.max(b) }
fn min(a: i32, b: i32) -> i32 { a.min(b) }
fn negate(a: i32) -> i32 { -a }
fn identity(a: i32) -> i32 { a }

#[test]
fn test_addition_properties() {
    // Addition is commutative
    assert!(verify_commutative(add, 3, 5));
    assert!(verify_commutative(add, -7, 12));
    
    // Addition is associative
    assert!(verify_associative(add, 1, 2, 3));
    assert!(verify_associative(add, -5, 10, -3));
    
    // Addition has identity 0
    assert!(verify_identity(add, 42, 0));
    assert!(verify_identity(add, -100, 0));
}

#[test]
fn test_multiplication_properties() {
    // Multiplication is commutative
    assert!(verify_commutative(mul, 3, 5));
    
    // Multiplication is associative
    assert!(verify_associative(mul, 2, 3, 4));
    
    // Multiplication has identity 1
    assert!(verify_identity(mul, 42, 1));
}

#[test]
fn test_max_min_properties() {
    // Max is commutative
    assert!(verify_commutative(max, 3, 5));
    
    // Max is associative
    assert!(verify_associative(max, 1, 5, 3));
    
    // Max is idempotent
    assert!(verify_idempotent(max, 5));
    
    // Min is commutative
    assert!(verify_commutative(min, 3, 5));
    
    // Min is idempotent
    assert!(verify_idempotent(min, 5));
}

#[test]
fn test_negation_properties() {
    // Negation is involutory
    assert!(verify_involutory(negate, 5));
    assert!(verify_involutory(negate, -42));
    
    // Identity is involutory
    assert!(verify_involutory(identity, 5));
}

#[test]
fn test_property_spec_workflow() {
    // Define properties for addition
    let add_spec = PropertySpec::new("add")
        .property(Property::Commutative)
        .property(Property::Associative)
        .property(Property::Identity)
        .with_identity("0");
    
    assert!(add_spec.has_property(Property::Commutative));
    assert!(add_spec.has_property(Property::Associative));
    assert!(add_spec.has_property(Property::Identity));
    assert_eq!(add_spec.identity, Some("0".to_string()));
}

#[test]
fn test_generator_workflow() {
    let mut gen = PropertyTestGenerator::new();
    
    // Add specs for common operations
    gen.add(
        PropertySpec::new("add")
            .property(Property::Commutative)
            .property(Property::Associative),
    );
    
    gen.add(
        PropertySpec::new("max")
            .property(Property::Commutative)
            .property(Property::Idempotent),
    );
    
    // Generate module
    let module = gen.generate_module("math_properties", "i32");
    
    assert!(module.contains("mod math_properties"));
    assert!(module.contains("test_add_commutative"));
    assert!(module.contains("test_max_idempotent"));
}

#[test]
fn test_property_parse_all() {
    let props = [
        "commutative",
        "associative",
        "idempotent",
        "identity",
        "inverse",
        "distributive",
        "involutory",
        "monotonic",
    ];
    
    for name in &props {
        assert!(
            Property::parse(name).is_some(),
            "Failed to parse: {}",
            name
        );
    }
}
