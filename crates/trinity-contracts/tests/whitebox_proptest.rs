//! Whitebox tests for property test generation.

use trinity_contracts::proptest::{
    parse_ensures, parse_requires, ParsedConstraint, PropertyTest, RangeHint,
    StrategyHint, TestModuleGenerator,
};

// ==================== RangeHint ====================

#[test]
fn test_range_hint_new() {
    let hint = RangeHint::new();
    assert!(hint.min.is_none());
    assert!(hint.max.is_none());
}

#[test]
fn test_range_hint_min_max() {
    let hint = RangeHint::new().min(0).max(100);
    assert_eq!(hint.min, Some(0));
    assert_eq!(hint.max, Some(100));
}

#[test]
fn test_range_hint_to_i32_range() {
    let hint = RangeHint::new().min(10).max(20);
    let range = hint.to_i32_range();
    assert_eq!(range.start, 10);
    assert_eq!(range.end, 20);
}

// ==================== StrategyHint ====================

#[test]
fn test_strategy_hint_variants() {
    let hints = vec![
        StrategyHint::Any,
        StrategyHint::Positive,
        StrategyHint::Negative,
        StrategyHint::NonZero,
        StrategyHint::NonEmpty,
    ];
    assert_eq!(hints.len(), 5);
}

// ==================== ParsedConstraint ====================

#[test]
fn test_parsed_constraint_new() {
    let constraint = ParsedConstraint::new("x", "x > 0");
    assert_eq!(constraint.param, "x");
    assert_eq!(constraint.expression, "x > 0");
}

#[test]
fn test_infer_hint_positive() {
    let constraint = ParsedConstraint::new("x", "x > 0");
    assert!(matches!(constraint.hint, StrategyHint::Positive));
}

#[test]
fn test_infer_hint_negative() {
    let constraint = ParsedConstraint::new("x", "x < 0");
    assert!(matches!(constraint.hint, StrategyHint::Negative));
}

#[test]
fn test_infer_hint_nonzero() {
    let constraint = ParsedConstraint::new("x", "x != 0");
    assert!(matches!(constraint.hint, StrategyHint::NonZero));
}

#[test]
fn test_infer_hint_nonempty() {
    let constraint = ParsedConstraint::new("s", "!s.is_empty()");
    assert!(matches!(constraint.hint, StrategyHint::NonEmpty));
}

#[test]
fn test_to_strategy_code_any() {
    let constraint = ParsedConstraint {
        param: "x".to_string(),
        hint: StrategyHint::Any,
        expression: "true".to_string(),
    };
    assert!(constraint.to_strategy_code("i32").contains("any"));
}

#[test]
fn test_to_strategy_code_positive() {
    let constraint = ParsedConstraint::new("x", "x > 0");
    let code = constraint.to_strategy_code("i32");
    assert!(code.contains("1.."));
}

// ==================== PropertyTest ====================

#[test]
fn test_property_test_new() {
    let test = PropertyTest::new("test_add", "add");
    assert_eq!(test.name, "test_add");
    assert_eq!(test.function, "add");
}

#[test]
fn test_property_test_param() {
    let test = PropertyTest::new("test_func", "func")
        .param(ParsedConstraint::new("x", "x > 0"));
    assert_eq!(test.params.len(), 1);
}

#[test]
fn test_property_test_postcondition() {
    let test = PropertyTest::new("test_func", "func")
        .postcondition("result > 0");
    assert_eq!(test.postconditions.len(), 1);
}

#[test]
fn test_property_test_generate_code() {
    let test = PropertyTest::new("test_double", "double")
        .param(ParsedConstraint::new("x", "x > 0"))
        .postcondition("result > x");
    let code = test.generate_code();
    
    assert!(code.contains("#[test]"));
    assert!(code.contains("proptest!"));
    assert!(code.contains("prop_assert!"));
}

// ==================== TestModuleGenerator ====================

#[test]
fn test_generator_new() {
    let gen = TestModuleGenerator::new("test_module");
    assert_eq!(gen.test_count(), 0);
}

#[test]
fn test_generator_add_test() {
    let mut gen = TestModuleGenerator::new("test_module");
    gen.add_test(PropertyTest::new("test1", "func1"));
    gen.add_test(PropertyTest::new("test2", "func2"));
    assert_eq!(gen.test_count(), 2);
}

#[test]
fn test_generator_generate() {
    let mut gen = TestModuleGenerator::new("contract_tests");
    gen.add_test(PropertyTest::new("test_add", "add"));
    let code = gen.generate();
    
    assert!(code.contains("mod contract_tests"));
    assert!(code.contains("use proptest::prelude::*"));
}

// ==================== parse_* functions ====================

#[test]
fn test_parse_requires() {
    let constraint = parse_requires("x", "x > 0");
    assert_eq!(constraint.param, "x");
}

#[test]
fn test_parse_ensures() {
    let result = parse_ensures("*result > 0");
    assert_eq!(result, "result > 0");
}

#[test]
fn test_parse_ensures_method() {
    let result = parse_ensures("result.is_valid()");
    assert!(result.contains("result"));
}
