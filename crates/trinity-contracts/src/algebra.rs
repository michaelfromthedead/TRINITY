//! Algebraic properties for contract verification.
//!
//! Provides support for verifying algebraic properties like
//! commutativity, associativity, idempotence, etc.

use std::fmt;

/// Algebraic property that can be verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Property {
    /// f(a, b) == f(b, a)
    Commutative,
    /// f(f(a, b), c) == f(a, f(b, c))
    Associative,
    /// f(a, a) == a
    Idempotent,
    /// f(a, identity) == a
    Identity,
    /// f(a, inverse(a)) == identity
    Inverse,
    /// f(a, f(b, c)) == f(f(a, b), f(a, c))
    Distributive,
    /// f(f(a)) == a
    Involutory,
    /// a <= b implies f(a) <= f(b)
    Monotonic,
}

impl fmt::Display for Property {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Property::Commutative => write!(f, "commutative"),
            Property::Associative => write!(f, "associative"),
            Property::Idempotent => write!(f, "idempotent"),
            Property::Identity => write!(f, "identity"),
            Property::Inverse => write!(f, "inverse"),
            Property::Distributive => write!(f, "distributive"),
            Property::Involutory => write!(f, "involutory"),
            Property::Monotonic => write!(f, "monotonic"),
        }
    }
}

impl Property {
    /// Parse from string.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "commutative" => Some(Property::Commutative),
            "associative" => Some(Property::Associative),
            "idempotent" => Some(Property::Idempotent),
            "identity" => Some(Property::Identity),
            "inverse" => Some(Property::Inverse),
            "distributive" => Some(Property::Distributive),
            "involutory" => Some(Property::Involutory),
            "monotonic" => Some(Property::Monotonic),
            _ => None,
        }
    }

    /// Get property description.
    pub fn description(&self) -> &'static str {
        match self {
            Property::Commutative => "f(a, b) == f(b, a)",
            Property::Associative => "f(f(a, b), c) == f(a, f(b, c))",
            Property::Idempotent => "f(a, a) == a",
            Property::Identity => "f(a, identity) == a",
            Property::Inverse => "f(a, inverse(a)) == identity",
            Property::Distributive => "f(a, f(b, c)) == f(f(a, b), f(a, c))",
            Property::Involutory => "f(f(a)) == a",
            Property::Monotonic => "a <= b implies f(a) <= f(b)",
        }
    }
}

/// Property specification for a function.
#[derive(Debug, Clone)]
pub struct PropertySpec {
    /// Function name.
    pub function: String,
    /// Properties to verify.
    pub properties: Vec<Property>,
    /// Identity element (if applicable).
    pub identity: Option<String>,
    /// Inverse function (if applicable).
    pub inverse_fn: Option<String>,
}

impl PropertySpec {
    /// Create a new property spec.
    pub fn new(function: impl Into<String>) -> Self {
        Self {
            function: function.into(),
            properties: Vec::new(),
            identity: None,
            inverse_fn: None,
        }
    }

    /// Add a property.
    pub fn property(mut self, prop: Property) -> Self {
        self.properties.push(prop);
        self
    }

    /// Set identity element.
    pub fn with_identity(mut self, identity: impl Into<String>) -> Self {
        self.identity = Some(identity.into());
        self
    }

    /// Set inverse function.
    pub fn with_inverse(mut self, inverse_fn: impl Into<String>) -> Self {
        self.inverse_fn = Some(inverse_fn.into());
        self
    }

    /// Check if has property.
    pub fn has_property(&self, prop: Property) -> bool {
        self.properties.contains(&prop)
    }
}

/// Test generator for algebraic properties.
pub struct PropertyTestGenerator {
    /// Test cases to generate.
    specs: Vec<PropertySpec>,
}

impl PropertyTestGenerator {
    /// Create a new generator.
    pub fn new() -> Self {
        Self { specs: Vec::new() }
    }

    /// Add a property spec.
    pub fn add(&mut self, spec: PropertySpec) {
        self.specs.push(spec);
    }

    /// Generate test code for commutativity.
    pub fn gen_commutative_test(func: &str, type_name: &str) -> String {
        format!(
            r#"#[test]
fn test_{}_commutative() {{
    proptest!(|(a: {}, b: {})| {{
        prop_assert_eq!({}(a, b), {}(b, a));
    }});
}}"#,
            func, type_name, type_name, func, func
        )
    }

    /// Generate test code for associativity.
    pub fn gen_associative_test(func: &str, type_name: &str) -> String {
        format!(
            r#"#[test]
fn test_{}_associative() {{
    proptest!(|(a: {}, b: {}, c: {})| {{
        prop_assert_eq!({}({}(a, b), c), {}(a, {}(b, c)));
    }});
}}"#,
            func, type_name, type_name, type_name, func, func, func, func
        )
    }

    /// Generate test code for idempotence.
    pub fn gen_idempotent_test(func: &str, type_name: &str) -> String {
        format!(
            r#"#[test]
fn test_{}_idempotent() {{
    proptest!(|(a: {})| {{
        prop_assert_eq!({}(a, a), a);
    }});
}}"#,
            func, type_name, func
        )
    }

    /// Generate test code for identity.
    pub fn gen_identity_test(func: &str, type_name: &str, identity: &str) -> String {
        format!(
            r#"#[test]
fn test_{}_identity() {{
    proptest!(|(a: {})| {{
        prop_assert_eq!({}(a, {}), a);
    }});
}}"#,
            func, type_name, func, identity
        )
    }

    /// Generate test code for involution.
    pub fn gen_involutory_test(func: &str, type_name: &str) -> String {
        format!(
            r#"#[test]
fn test_{}_involutory() {{
    proptest!(|(a: {})| {{
        prop_assert_eq!({}({}(a)), a);
    }});
}}"#,
            func, type_name, func, func
        )
    }

    /// Generate all tests for a spec.
    pub fn generate_tests(&self, spec: &PropertySpec, type_name: &str) -> String {
        let mut tests = String::new();

        for prop in &spec.properties {
            let test = match prop {
                Property::Commutative => {
                    Self::gen_commutative_test(&spec.function, type_name)
                }
                Property::Associative => {
                    Self::gen_associative_test(&spec.function, type_name)
                }
                Property::Idempotent => {
                    Self::gen_idempotent_test(&spec.function, type_name)
                }
                Property::Identity => {
                    if let Some(ref id) = spec.identity {
                        Self::gen_identity_test(&spec.function, type_name, id)
                    } else {
                        continue;
                    }
                }
                Property::Involutory => {
                    Self::gen_involutory_test(&spec.function, type_name)
                }
                _ => continue,
            };

            tests.push_str(&test);
            tests.push_str("\n\n");
        }

        tests
    }

    /// Generate module with all tests.
    pub fn generate_module(&self, module_name: &str, type_name: &str) -> String {
        let mut code = String::new();

        code.push_str(&format!("mod {} {{\n", module_name));
        code.push_str("    use super::*;\n");
        code.push_str("    use proptest::prelude::*;\n\n");

        for spec in &self.specs {
            let tests = self.generate_tests(spec, type_name);
            for line in tests.lines() {
                code.push_str("    ");
                code.push_str(line);
                code.push('\n');
            }
        }

        code.push_str("}\n");
        code
    }
}

impl Default for PropertyTestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Verify commutativity at runtime.
pub fn verify_commutative<T, F>(f: F, a: T, b: T) -> bool
where
    T: Clone + PartialEq,
    F: Fn(T, T) -> T,
{
    f(a.clone(), b.clone()) == f(b, a)
}

/// Verify associativity at runtime.
pub fn verify_associative<T, F>(f: F, a: T, b: T, c: T) -> bool
where
    T: Clone + PartialEq,
    F: Fn(T, T) -> T,
{
    f(f(a.clone(), b.clone()), c.clone()) == f(a, f(b, c))
}

/// Verify idempotence at runtime.
pub fn verify_idempotent<T, F>(f: F, a: T) -> bool
where
    T: Clone + PartialEq,
    F: Fn(T, T) -> T,
{
    f(a.clone(), a.clone()) == a
}

/// Verify identity at runtime.
pub fn verify_identity<T, F>(f: F, a: T, identity: T) -> bool
where
    T: Clone + PartialEq,
    F: Fn(T, T) -> T,
{
    f(a.clone(), identity) == a
}

/// Verify involution at runtime.
pub fn verify_involutory<T, F>(f: F, a: T) -> bool
where
    T: Clone + PartialEq,
    F: Fn(T) -> T,
{
    f(f(a.clone())) == a
}
