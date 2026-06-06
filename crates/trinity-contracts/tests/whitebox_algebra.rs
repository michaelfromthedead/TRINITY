//! Whitebox tests for algebraic properties.

use trinity_contracts::algebra::{
    verify_associative, verify_commutative, verify_idempotent, verify_identity, verify_involutory,
    Property, PropertySpec, PropertyTestGenerator,
};

// ==================== Property ====================

#[test]
fn test_property_variants() {
    let props = vec![
        Property::Commutative,
        Property::Associative,
        Property::Idempotent,
        Property::Identity,
        Property::Inverse,
        Property::Distributive,
        Property::Involutory,
        Property::Monotonic,
    ];
    assert_eq!(props.len(), 8);
}

#[test]
fn test_property_display() {
    assert_eq!(format!("{}", Property::Commutative), "commutative");
    assert_eq!(format!("{}", Property::Associative), "associative");
}

#[test]
fn test_property_parse() {
    assert_eq!(Property::parse("commutative"), Some(Property::Commutative));
    assert_eq!(Property::parse("ASSOCIATIVE"), Some(Property::Associative));
    assert_eq!(Property::parse("unknown"), None);
}

#[test]
fn test_property_description() {
    let desc = Property::Commutative.description();
    assert!(desc.contains("f(a, b)"));
}

// ==================== PropertySpec ====================

#[test]
fn test_property_spec_new() {
    let spec = PropertySpec::new("add");
    assert_eq!(spec.function, "add");
    assert!(spec.properties.is_empty());
}

#[test]
fn test_property_spec_property() {
    let spec = PropertySpec::new("add")
        .property(Property::Commutative)
        .property(Property::Associative);
    assert_eq!(spec.properties.len(), 2);
}

#[test]
fn test_property_spec_with_identity() {
    let spec = PropertySpec::new("add").with_identity("0");
    assert_eq!(spec.identity, Some("0".to_string()));
}

#[test]
fn test_property_spec_with_inverse() {
    let spec = PropertySpec::new("add").with_inverse("negate");
    assert_eq!(spec.inverse_fn, Some("negate".to_string()));
}

#[test]
fn test_property_spec_has_property() {
    let spec = PropertySpec::new("add").property(Property::Commutative);
    assert!(spec.has_property(Property::Commutative));
    assert!(!spec.has_property(Property::Associative));
}

// ==================== PropertyTestGenerator ====================

#[test]
fn test_generator_new() {
    let gen = PropertyTestGenerator::new();
    assert!(gen.generate_module("test", "i32").contains("mod test"));
}

#[test]
fn test_gen_commutative_test() {
    let code = PropertyTestGenerator::gen_commutative_test("add", "i32");
    assert!(code.contains("test_add_commutative"));
    assert!(code.contains("proptest!"));
}

#[test]
fn test_gen_associative_test() {
    let code = PropertyTestGenerator::gen_associative_test("add", "i32");
    assert!(code.contains("test_add_associative"));
    assert!(code.contains("a:") && code.contains("b:") && code.contains("c:"));
}

#[test]
fn test_gen_idempotent_test() {
    let code = PropertyTestGenerator::gen_idempotent_test("max", "i32");
    assert!(code.contains("test_max_idempotent"));
}

#[test]
fn test_gen_identity_test() {
    let code = PropertyTestGenerator::gen_identity_test("add", "i32", "0");
    assert!(code.contains("test_add_identity"));
    assert!(code.contains("0"));
}

#[test]
fn test_gen_involutory_test() {
    let code = PropertyTestGenerator::gen_involutory_test("negate", "i32");
    assert!(code.contains("test_negate_involutory"));
}

#[test]
fn test_generate_tests() {
    let mut gen = PropertyTestGenerator::new();
    let spec = PropertySpec::new("add")
        .property(Property::Commutative)
        .property(Property::Associative);
    gen.add(spec.clone());
    
    let tests = gen.generate_tests(&spec, "i32");
    assert!(tests.contains("commutative"));
    assert!(tests.contains("associative"));
}

#[test]
fn test_generate_module() {
    let mut gen = PropertyTestGenerator::new();
    gen.add(PropertySpec::new("add").property(Property::Commutative));
    
    let module = gen.generate_module("algebra_tests", "i32");
    assert!(module.contains("mod algebra_tests"));
    assert!(module.contains("use proptest"));
}

// ==================== verify_* functions ====================

#[test]
fn test_verify_commutative_true() {
    let add = |a: i32, b: i32| a + b;
    assert!(verify_commutative(add, 3, 5));
}

#[test]
fn test_verify_commutative_false() {
    let sub = |a: i32, b: i32| a - b;
    assert!(!verify_commutative(sub, 3, 5));
}

#[test]
fn test_verify_associative_true() {
    let add = |a: i32, b: i32| a.wrapping_add(b);
    assert!(verify_associative(add, 1, 2, 3));
}

#[test]
fn test_verify_idempotent_true() {
    let max = |a: i32, b: i32| a.max(b);
    assert!(verify_idempotent(max, 5));
}

#[test]
fn test_verify_identity_true() {
    let add = |a: i32, b: i32| a + b;
    assert!(verify_identity(add, 5, 0));
}

#[test]
fn test_verify_involutory_true() {
    let negate = |a: i32| -a;
    assert!(verify_involutory(negate, 5));
}
