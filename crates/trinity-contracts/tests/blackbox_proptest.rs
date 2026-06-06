//! Blackbox tests for property test generation with real usage.

use trinity_contracts::proptest::{
    ParsedConstraint, PropertyTest, RangeHint, StrategyHint, TestModuleGenerator,
};

#[test]
fn test_full_workflow() {
    // Create constraints for a function
    let mut gen = TestModuleGenerator::new("math_tests");

    // Test for an add function
    let add_test = PropertyTest::new("test_add_commutative", "add")
        .param(ParsedConstraint::new("a", "true"))
        .param(ParsedConstraint::new("b", "true"))
        .postcondition("add(a, b) == add(b, a)");

    gen.add_test(add_test);

    // Test for a multiply function
    let mul_test = PropertyTest::new("test_mul_by_zero", "multiply")
        .param(ParsedConstraint::new("x", "true"))
        .postcondition("multiply(x, 0) == 0");

    gen.add_test(mul_test);

    // Generate module
    let code = gen.generate();

    assert!(code.contains("mod math_tests"));
    assert!(code.contains("test_add_commutative"));
    assert!(code.contains("test_mul_by_zero"));
}

#[test]
fn test_constraint_based_strategies() {
    // Positive constraint
    let pos = ParsedConstraint::new("x", "x > 0");
    assert!(matches!(pos.hint, StrategyHint::Positive));

    // Non-zero constraint
    let nonzero = ParsedConstraint::new("y", "y != 0");
    assert!(matches!(nonzero.hint, StrategyHint::NonZero));

    // Non-empty constraint
    let nonempty = ParsedConstraint::new("s", "!s.is_empty()");
    assert!(matches!(nonempty.hint, StrategyHint::NonEmpty));
}

#[test]
fn test_range_strategies() {
    let range = RangeHint::new().min(1).max(100);

    let constraint = ParsedConstraint {
        param: "x".to_string(),
        hint: StrategyHint::Range(range),
        expression: "x >= 1 && x <= 100".to_string(),
    };

    let code = constraint.to_strategy_code("i32");
    assert!(code.contains("1"));
    assert!(code.contains("100"));
}

#[test]
fn test_generated_test_structure() {
    let test = PropertyTest::new("test_division", "safe_div")
        .param(ParsedConstraint::new("a", "true"))
        .param(ParsedConstraint::new("b", "b != 0"))
        .postcondition("safe_div(a, b) * b + a % b == a");

    let code = test.generate_code();

    // Check structure
    assert!(code.contains("#[test]"));
    assert!(code.contains("fn test_division"));
    assert!(code.contains("proptest!"));
    assert!(code.contains("let result"));
    assert!(code.contains("prop_assert!"));
}

#[test]
fn test_multiple_params_test() {
    let test = PropertyTest::new("test_clamp", "clamp")
        .param(ParsedConstraint::new("value", "true"))
        .param(ParsedConstraint::new("min", "true"))
        .param(ParsedConstraint::new("max", "max >= min"))
        .postcondition("result >= min")
        .postcondition("result <= max");

    assert_eq!(test.params.len(), 3);
    assert_eq!(test.postconditions.len(), 2);
}

#[test]
fn test_oneof_strategy() {
    let constraint = ParsedConstraint {
        param: "op".to_string(),
        hint: StrategyHint::OneOf(vec![
            "\"add\"".to_string(),
            "\"sub\"".to_string(),
            "\"mul\"".to_string(),
        ]),
        expression: "op in operations".to_string(),
    };

    let code = constraint.to_strategy_code("&str");
    assert!(code.contains("prop_oneof!"));
    assert!(code.contains("add"));
}

#[test]
fn test_custom_strategy() {
    let constraint = ParsedConstraint {
        param: "v".to_string(),
        hint: StrategyHint::Custom("vec![1, 2, 3, 4, 5].prop_shuffle()".to_string()),
        expression: "v is shuffled".to_string(),
    };

    let code = constraint.to_strategy_code("Vec<i32>");
    assert!(code.contains("prop_shuffle"));
}
