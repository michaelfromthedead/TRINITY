//! Contract annotations for Trinity.
//!
//! Provides `#[contract]` attribute macro for:
//! - Runtime precondition/postcondition checks
//! - Property-based test generation
//! - Contract schema extraction

pub mod runtime;

pub use runtime::{
    check_ensures, check_invariant, check_requires, debug_ensures, debug_invariant,
    debug_requires, CheckKind, CheckResult, ContractChecker, InvariantGuard,
};
pub use trinity_contracts_macros::contract;

use serde::{Deserialize, Serialize};

/// A contract definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contract {
    /// Function name.
    pub function: String,
    /// Preconditions (requires).
    pub requires: Vec<Constraint>,
    /// Postconditions (ensures).
    pub ensures: Vec<Constraint>,
    /// Invariants.
    pub invariants: Vec<Constraint>,
    /// Layout constraints.
    pub layout: Option<LayoutConstraint>,
}

impl Contract {
    /// Create a new empty contract.
    pub fn new(function: impl Into<String>) -> Self {
        Self {
            function: function.into(),
            requires: Vec::new(),
            ensures: Vec::new(),
            invariants: Vec::new(),
            layout: None,
        }
    }

    /// Add a precondition.
    pub fn requires(mut self, constraint: Constraint) -> Self {
        self.requires.push(constraint);
        self
    }

    /// Add a postcondition.
    pub fn ensures(mut self, constraint: Constraint) -> Self {
        self.ensures.push(constraint);
        self
    }

    /// Add an invariant.
    pub fn invariant(mut self, constraint: Constraint) -> Self {
        self.invariants.push(constraint);
        self
    }

    /// Set layout constraint.
    pub fn layout(mut self, layout: LayoutConstraint) -> Self {
        self.layout = Some(layout);
        self
    }

    /// Check all preconditions.
    pub fn check_requires(&self) -> ContractResult {
        let mut result = ContractResult::new();
        for req in &self.requires {
            if !req.is_satisfied {
                result.add_violation(ContractViolation {
                    kind: ViolationKind::Precondition,
                    constraint: req.expression.clone(),
                    message: req.message.clone(),
                });
            }
        }
        result
    }

    /// Check all postconditions.
    pub fn check_ensures(&self) -> ContractResult {
        let mut result = ContractResult::new();
        for ens in &self.ensures {
            if !ens.is_satisfied {
                result.add_violation(ContractViolation {
                    kind: ViolationKind::Postcondition,
                    constraint: ens.expression.clone(),
                    message: ens.message.clone(),
                });
            }
        }
        result
    }
}

/// A constraint expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// The constraint expression as a string.
    pub expression: String,
    /// Optional human-readable message.
    pub message: Option<String>,
    /// Whether the constraint is currently satisfied.
    #[serde(skip)]
    pub is_satisfied: bool,
}

impl Constraint {
    /// Create a new constraint.
    pub fn new(expression: impl Into<String>) -> Self {
        Self {
            expression: expression.into(),
            message: None,
            is_satisfied: true,
        }
    }

    /// Add a message.
    pub fn message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Mark as satisfied or not.
    pub fn satisfied(mut self, is_satisfied: bool) -> Self {
        self.is_satisfied = is_satisfied;
        self
    }
}

/// Layout constraints for struct alignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConstraint {
    /// Required size in bytes.
    pub size: Option<usize>,
    /// Required alignment in bytes.
    pub align: Option<usize>,
}

impl LayoutConstraint {
    /// Create a new layout constraint.
    pub fn new() -> Self {
        Self {
            size: None,
            align: None,
        }
    }

    /// Set size.
    pub fn size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }

    /// Set alignment.
    pub fn align(mut self, align: usize) -> Self {
        self.align = Some(align);
        self
    }
}

impl Default for LayoutConstraint {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of contract validation.
#[derive(Debug, Clone, Default)]
pub struct ContractResult {
    /// Whether all constraints passed.
    pub passed: bool,
    /// Violations found.
    pub violations: Vec<ContractViolation>,
}

impl ContractResult {
    /// Create a new passing result.
    pub fn new() -> Self {
        Self {
            passed: true,
            violations: Vec::new(),
        }
    }

    /// Add a violation.
    pub fn add_violation(&mut self, violation: ContractViolation) {
        self.passed = false;
        self.violations.push(violation);
    }

    /// Check if any violations occurred.
    pub fn has_violations(&self) -> bool {
        !self.violations.is_empty()
    }
}

/// A contract violation.
#[derive(Debug, Clone)]
pub struct ContractViolation {
    /// Kind of violation.
    pub kind: ViolationKind,
    /// The constraint that was violated.
    pub constraint: String,
    /// Optional message.
    pub message: Option<String>,
}

/// Kind of contract violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationKind {
    /// Precondition violation.
    Precondition,
    /// Postcondition violation.
    Postcondition,
    /// Invariant violation.
    Invariant,
    /// Layout violation.
    Layout,
}

/// Algebraic properties for functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgebraicProperty {
    /// f(a, b) == f(b, a)
    Commutative,
    /// f(f(a, b), c) == f(a, f(b, c))
    Associative,
    /// f(a, a) == a
    Idempotent,
    /// f(a, id) == a
    Identity,
    /// f(a, inv(a)) == id
    Inverse,
}

/// Contract schema for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractSchema {
    /// Version.
    pub version: String,
    /// Contracts in the schema.
    pub contracts: Vec<Contract>,
}

impl ContractSchema {
    /// Create a new schema.
    pub fn new() -> Self {
        Self {
            version: "1.0".to_string(),
            contracts: Vec::new(),
        }
    }

    /// Add a contract.
    pub fn add(&mut self, contract: Contract) {
        self.contracts.push(contract);
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parse from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl Default for ContractSchema {
    fn default() -> Self {
        Self::new()
    }
}
