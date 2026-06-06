//! Runtime contract checking utilities.
//!
//! Provides functions for runtime contract verification with
//! detailed error messages and optional logging.

use std::fmt;

/// Result of a runtime contract check.
#[derive(Debug, Clone)]
pub struct CheckResult {
    /// Whether the check passed.
    pub passed: bool,
    /// The constraint expression.
    pub constraint: String,
    /// Error message if failed.
    pub message: Option<String>,
    /// Function name.
    pub function: String,
    /// Check kind.
    pub kind: CheckKind,
}

impl CheckResult {
    /// Create a passing result.
    pub fn pass(constraint: impl Into<String>, function: impl Into<String>, kind: CheckKind) -> Self {
        Self {
            passed: true,
            constraint: constraint.into(),
            message: None,
            function: function.into(),
            kind,
        }
    }

    /// Create a failing result.
    pub fn fail(
        constraint: impl Into<String>,
        function: impl Into<String>,
        kind: CheckKind,
        message: impl Into<String>,
    ) -> Self {
        Self {
            passed: false,
            constraint: constraint.into(),
            message: Some(message.into()),
            function: function.into(),
            kind,
        }
    }
}

/// Kind of runtime check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckKind {
    /// Precondition check.
    Precondition,
    /// Postcondition check.
    Postcondition,
    /// Invariant check.
    Invariant,
}

impl fmt::Display for CheckKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckKind::Precondition => write!(f, "Precondition"),
            CheckKind::Postcondition => write!(f, "Postcondition"),
            CheckKind::Invariant => write!(f, "Invariant"),
        }
    }
}

/// Check a precondition at runtime.
#[inline]
pub fn check_requires(condition: bool, constraint: &str, function: &str) {
    if !condition {
        panic!(
            "Contract violation in `{}`: precondition `{}` failed",
            function, constraint
        );
    }
}

/// Check a postcondition at runtime.
#[inline]
pub fn check_ensures(condition: bool, constraint: &str, function: &str) {
    if !condition {
        panic!(
            "Contract violation in `{}`: postcondition `{}` failed",
            function, constraint
        );
    }
}

/// Check an invariant at runtime.
#[inline]
pub fn check_invariant(condition: bool, constraint: &str, function: &str) {
    if !condition {
        panic!(
            "Contract violation in `{}`: invariant `{}` failed",
            function, constraint
        );
    }
}

/// Debug-only precondition check.
#[inline]
pub fn debug_requires(condition: bool, constraint: &str, function: &str) {
    debug_assert!(
        condition,
        "Contract violation in `{}`: precondition `{}` failed",
        function,
        constraint
    );
}

/// Debug-only postcondition check.
#[inline]
pub fn debug_ensures(condition: bool, constraint: &str, function: &str) {
    debug_assert!(
        condition,
        "Contract violation in `{}`: postcondition `{}` failed",
        function,
        constraint
    );
}

/// Debug-only invariant check.
#[inline]
pub fn debug_invariant(condition: bool, constraint: &str, function: &str) {
    debug_assert!(
        condition,
        "Contract violation in `{}`: invariant `{}` failed",
        function,
        constraint
    );
}

/// Contract checker that collects violations.
#[derive(Debug, Default)]
pub struct ContractChecker {
    /// Function being checked.
    function: String,
    /// Collected violations.
    violations: Vec<CheckResult>,
    /// Whether to panic on first violation.
    panic_on_first: bool,
}

impl ContractChecker {
    /// Create a new checker for a function.
    pub fn new(function: impl Into<String>) -> Self {
        Self {
            function: function.into(),
            violations: Vec::new(),
            panic_on_first: false,
        }
    }

    /// Set to panic on first violation.
    pub fn panic_on_first(mut self) -> Self {
        self.panic_on_first = true;
        self
    }

    /// Check a precondition.
    pub fn requires(&mut self, condition: bool, constraint: &str) -> &mut Self {
        if !condition {
            let result = CheckResult::fail(
                constraint,
                &self.function,
                CheckKind::Precondition,
                format!("Precondition `{}` failed", constraint),
            );
            if self.panic_on_first {
                panic!("{}", result.message.as_ref().unwrap());
            }
            self.violations.push(result);
        }
        self
    }

    /// Check a postcondition.
    pub fn ensures(&mut self, condition: bool, constraint: &str) -> &mut Self {
        if !condition {
            let result = CheckResult::fail(
                constraint,
                &self.function,
                CheckKind::Postcondition,
                format!("Postcondition `{}` failed", constraint),
            );
            if self.panic_on_first {
                panic!("{}", result.message.as_ref().unwrap());
            }
            self.violations.push(result);
        }
        self
    }

    /// Check an invariant.
    pub fn invariant(&mut self, condition: bool, constraint: &str) -> &mut Self {
        if !condition {
            let result = CheckResult::fail(
                constraint,
                &self.function,
                CheckKind::Invariant,
                format!("Invariant `{}` failed", constraint),
            );
            if self.panic_on_first {
                panic!("{}", result.message.as_ref().unwrap());
            }
            self.violations.push(result);
        }
        self
    }

    /// Get all violations.
    pub fn violations(&self) -> &[CheckResult] {
        &self.violations
    }

    /// Check if any violations occurred.
    pub fn has_violations(&self) -> bool {
        !self.violations.is_empty()
    }

    /// Panic if any violations occurred.
    pub fn assert_valid(&self) {
        if !self.violations.is_empty() {
            let messages: Vec<&str> = self
                .violations
                .iter()
                .filter_map(|v| v.message.as_deref())
                .collect();
            panic!(
                "Contract violations in `{}`:\n  {}",
                self.function,
                messages.join("\n  ")
            );
        }
    }
}

/// Guard for checking invariants on drop.
pub struct InvariantGuard<F: FnOnce() -> bool> {
    check: Option<F>,
    constraint: String,
    function: String,
}

impl<F: FnOnce() -> bool> InvariantGuard<F> {
    /// Create a new invariant guard.
    pub fn new(constraint: impl Into<String>, function: impl Into<String>, check: F) -> Self {
        Self {
            check: Some(check),
            constraint: constraint.into(),
            function: function.into(),
        }
    }
}

impl<F: FnOnce() -> bool> Drop for InvariantGuard<F> {
    fn drop(&mut self) {
        if let Some(check) = self.check.take() {
            if !check() {
                panic!(
                    "Contract violation in `{}`: invariant `{}` failed on exit",
                    self.function, self.constraint
                );
            }
        }
    }
}
