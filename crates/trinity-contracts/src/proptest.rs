//! Property test generation utilities for contracts.
//!
//! Provides helpers for generating proptest-based property tests
//! from contract specifications.

use std::ops::Range;

/// Strategy hint for generating test inputs.
#[derive(Debug, Clone)]
pub enum StrategyHint {
    /// Any value of the type.
    Any,
    /// Value in a range.
    Range(RangeHint),
    /// Positive numbers only.
    Positive,
    /// Negative numbers only.
    Negative,
    /// Non-zero values.
    NonZero,
    /// Non-empty collections/strings.
    NonEmpty,
    /// Value from a set.
    OneOf(Vec<String>),
    /// Custom strategy expression.
    Custom(String),
}

/// Range hint for numeric types.
#[derive(Debug, Clone)]
pub struct RangeHint {
    /// Minimum value (inclusive).
    pub min: Option<i64>,
    /// Maximum value (inclusive).
    pub max: Option<i64>,
}

impl RangeHint {
    /// Create a new range hint.
    pub fn new() -> Self {
        Self { min: None, max: None }
    }

    /// Set minimum.
    pub fn min(mut self, min: i64) -> Self {
        self.min = Some(min);
        self
    }

    /// Set maximum.
    pub fn max(mut self, max: i64) -> Self {
        self.max = Some(max);
        self
    }

    /// Convert to i32 range.
    pub fn to_i32_range(&self) -> Range<i32> {
        let min = self.min.unwrap_or(i32::MIN as i64) as i32;
        let max = self.max.unwrap_or(i32::MAX as i64) as i32;
        min..max
    }

    /// Convert to i64 range.
    pub fn to_i64_range(&self) -> Range<i64> {
        let min = self.min.unwrap_or(i64::MIN);
        let max = self.max.unwrap_or(i64::MAX);
        min..max
    }
}

impl Default for RangeHint {
    fn default() -> Self {
        Self::new()
    }
}

/// Parsed constraint for property testing.
#[derive(Debug, Clone)]
pub struct ParsedConstraint {
    /// Parameter name.
    pub param: String,
    /// Strategy hint derived from constraint.
    pub hint: StrategyHint,
    /// Original expression.
    pub expression: String,
}

impl ParsedConstraint {
    /// Create a new parsed constraint.
    pub fn new(param: impl Into<String>, expression: impl Into<String>) -> Self {
        let expr = expression.into();
        let hint = Self::infer_hint(&expr);
        Self {
            param: param.into(),
            hint,
            expression: expr,
        }
    }

    /// Infer strategy hint from expression.
    fn infer_hint(expr: &str) -> StrategyHint {
        // Simple pattern matching on common constraints
        if expr.contains("> 0") || expr.contains(">= 1") {
            StrategyHint::Positive
        } else if expr.contains("< 0") || expr.contains("<= -1") {
            StrategyHint::Negative
        } else if expr.contains("!= 0") {
            StrategyHint::NonZero
        } else if expr.contains("is_empty()") && expr.contains('!') {
            StrategyHint::NonEmpty
        } else {
            StrategyHint::Any
        }
    }

    /// Generate proptest strategy code.
    pub fn to_strategy_code(&self, type_name: &str) -> String {
        match &self.hint {
            StrategyHint::Any => format!("any::<{}>()", type_name),
            StrategyHint::Positive => {
                if type_name.starts_with('i') {
                    format!("1..{}::MAX", type_name)
                } else {
                    format!("1..{}::MAX", type_name)
                }
            }
            StrategyHint::Negative => format!("{}::MIN..-1", type_name),
            StrategyHint::NonZero => {
                format!(
                    "prop_oneof![{}::MIN..-1, 1..{}::MAX]",
                    type_name, type_name
                )
            }
            StrategyHint::NonEmpty => "\"[a-z]+\"".to_string(),
            StrategyHint::Range(r) => {
                let min = r.min.map(|v| v.to_string()).unwrap_or_else(|| format!("{}::MIN", type_name));
                let max = r.max.map(|v| v.to_string()).unwrap_or_else(|| format!("{}::MAX", type_name));
                format!("{}..{}", min, max)
            }
            StrategyHint::OneOf(values) => {
                format!("prop_oneof![{}]", values.join(", "))
            }
            StrategyHint::Custom(s) => s.clone(),
        }
    }
}

/// Property test specification.
#[derive(Debug, Clone)]
pub struct PropertyTest {
    /// Test name.
    pub name: String,
    /// Function being tested.
    pub function: String,
    /// Parameter constraints.
    pub params: Vec<ParsedConstraint>,
    /// Postconditions to verify.
    pub postconditions: Vec<String>,
}

impl PropertyTest {
    /// Create a new property test.
    pub fn new(name: impl Into<String>, function: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            function: function.into(),
            params: Vec::new(),
            postconditions: Vec::new(),
        }
    }

    /// Add a parameter constraint.
    pub fn param(mut self, constraint: ParsedConstraint) -> Self {
        self.params.push(constraint);
        self
    }

    /// Add a postcondition.
    pub fn postcondition(mut self, expr: impl Into<String>) -> Self {
        self.postconditions.push(expr.into());
        self
    }

    /// Generate proptest code.
    pub fn generate_code(&self) -> String {
        let mut code = String::new();

        code.push_str("#[test]\n");
        code.push_str(&format!("fn {}() {{\n", self.name));
        code.push_str("    proptest!(|(");

        // Parameters
        let param_list: Vec<String> = self
            .params
            .iter()
            .map(|p| format!("{} in {}", p.param, p.to_strategy_code("i32")))
            .collect();
        code.push_str(&param_list.join(", "));

        code.push_str(")| {\n");
        code.push_str(&format!("        let result = {}({});\n", self.function, 
            self.params.iter().map(|p| p.param.as_str()).collect::<Vec<_>>().join(", ")));

        // Postconditions
        for post in &self.postconditions {
            code.push_str(&format!("        prop_assert!({});\n", post));
        }

        code.push_str("    });\n");
        code.push_str("}\n");

        code
    }
}

/// Test module generator.
#[derive(Debug, Default)]
pub struct TestModuleGenerator {
    /// Module name.
    module_name: String,
    /// Tests to generate.
    tests: Vec<PropertyTest>,
}

impl TestModuleGenerator {
    /// Create a new generator.
    pub fn new(module_name: impl Into<String>) -> Self {
        Self {
            module_name: module_name.into(),
            tests: Vec::new(),
        }
    }

    /// Add a property test.
    pub fn add_test(&mut self, test: PropertyTest) {
        self.tests.push(test);
    }

    /// Generate the complete module.
    pub fn generate(&self) -> String {
        let mut code = String::new();

        code.push_str(&format!("mod {} {{\n", self.module_name));
        code.push_str("    use super::*;\n");
        code.push_str("    use proptest::prelude::*;\n\n");

        for test in &self.tests {
            for line in test.generate_code().lines() {
                code.push_str("    ");
                code.push_str(line);
                code.push('\n');
            }
            code.push('\n');
        }

        code.push_str("}\n");
        code
    }

    /// Get test count.
    pub fn test_count(&self) -> usize {
        self.tests.len()
    }
}

/// Parse a requires clause into a constraint.
pub fn parse_requires(param: &str, expr: &str) -> ParsedConstraint {
    ParsedConstraint::new(param, expr)
}

/// Parse an ensures clause for property assertions.
pub fn parse_ensures(expr: &str) -> String {
    // Convert result references
    expr.replace("*result", "result")
        .replace("result.", "&result.")
}
