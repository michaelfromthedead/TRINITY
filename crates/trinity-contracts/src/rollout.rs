//! Incremental rollout utilities for contract adoption.
//!
//! Provides tools for validating and tracking contract adoption
//! across the codebase.

use std::collections::HashMap;

/// Contract adoption status for a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdoptionStatus {
    /// Not yet annotated.
    Pending,
    /// Annotated but not validated.
    Annotated,
    /// Annotated and validated.
    Validated,
    /// Skipped (not suitable for contracts).
    Skipped,
}

/// Function with contract adoption tracking.
#[derive(Debug, Clone)]
pub struct TrackedFunction {
    /// Function name.
    pub name: String,
    /// Module path.
    pub module: String,
    /// Adoption status.
    pub status: AdoptionStatus,
    /// Priority (higher = more important).
    pub priority: u8,
    /// Notes or reason for skipping.
    pub notes: Option<String>,
}

impl TrackedFunction {
    /// Create a new tracked function.
    pub fn new(name: impl Into<String>, module: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            module: module.into(),
            status: AdoptionStatus::Pending,
            priority: 5,
            notes: None,
        }
    }

    /// Set priority.
    pub fn priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set status.
    pub fn status(mut self, status: AdoptionStatus) -> Self {
        self.status = status;
        self
    }

    /// Set notes.
    pub fn notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Full qualified name.
    pub fn fqn(&self) -> String {
        format!("{}::{}", self.module, self.name)
    }
}

/// Rollout tracker for contract adoption.
#[derive(Debug, Default)]
pub struct RolloutTracker {
    /// Tracked functions.
    functions: HashMap<String, TrackedFunction>,
    /// Current phase.
    phase: u8,
}

impl RolloutTracker {
    /// Create a new tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a function to track.
    pub fn track(&mut self, func: TrackedFunction) {
        self.functions.insert(func.fqn(), func);
    }

    /// Get a function by FQN.
    pub fn get(&self, fqn: &str) -> Option<&TrackedFunction> {
        self.functions.get(fqn)
    }

    /// Update status.
    pub fn update_status(&mut self, fqn: &str, status: AdoptionStatus) -> bool {
        if let Some(func) = self.functions.get_mut(fqn) {
            func.status = status;
            true
        } else {
            false
        }
    }

    /// Get functions by status.
    pub fn by_status(&self, status: AdoptionStatus) -> Vec<&TrackedFunction> {
        self.functions
            .values()
            .filter(|f| f.status == status)
            .collect()
    }

    /// Get high-priority pending functions.
    pub fn pending_high_priority(&self, min_priority: u8) -> Vec<&TrackedFunction> {
        self.functions
            .values()
            .filter(|f| f.status == AdoptionStatus::Pending && f.priority >= min_priority)
            .collect()
    }

    /// Get adoption statistics.
    pub fn stats(&self) -> RolloutStats {
        let mut stats = RolloutStats::default();

        for func in self.functions.values() {
            stats.total += 1;
            match func.status {
                AdoptionStatus::Pending => stats.pending += 1,
                AdoptionStatus::Annotated => stats.annotated += 1,
                AdoptionStatus::Validated => stats.validated += 1,
                AdoptionStatus::Skipped => stats.skipped += 1,
            }
        }

        stats
    }

    /// Set rollout phase.
    pub fn set_phase(&mut self, phase: u8) {
        self.phase = phase;
    }

    /// Get current phase.
    pub fn phase(&self) -> u8 {
        self.phase
    }

    /// Get all tracked functions.
    pub fn all(&self) -> Vec<&TrackedFunction> {
        self.functions.values().collect()
    }
}

/// Rollout statistics.
#[derive(Debug, Default, Clone)]
pub struct RolloutStats {
    /// Total functions tracked.
    pub total: usize,
    /// Pending adoption.
    pub pending: usize,
    /// Annotated but not validated.
    pub annotated: usize,
    /// Fully validated.
    pub validated: usize,
    /// Skipped.
    pub skipped: usize,
}

impl RolloutStats {
    /// Get adoption percentage.
    pub fn adoption_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            let adopted = self.annotated + self.validated;
            (adopted as f64 / self.total as f64) * 100.0
        }
    }

    /// Get validation percentage.
    pub fn validation_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.validated as f64 / self.total as f64) * 100.0
        }
    }

    /// Generate summary.
    pub fn summary(&self) -> String {
        format!(
            "Total: {}, Pending: {}, Annotated: {}, Validated: {}, Skipped: {} ({:.1}% adopted, {:.1}% validated)",
            self.total,
            self.pending,
            self.annotated,
            self.validated,
            self.skipped,
            self.adoption_rate(),
            self.validation_rate()
        )
    }
}

/// High-value functions for initial rollout.
pub fn high_value_functions() -> Vec<TrackedFunction> {
    vec![
        // Math operations
        TrackedFunction::new("safe_div", "math")
            .priority(10)
            .status(AdoptionStatus::Validated)
            .notes("Division with non-zero check"),
        TrackedFunction::new("sqrt", "math")
            .priority(9)
            .status(AdoptionStatus::Validated)
            .notes("Square root with non-negative check"),
        TrackedFunction::new("clamp", "math")
            .priority(8)
            .status(AdoptionStatus::Validated)
            .notes("Clamp with min <= max invariant"),
        
        // Memory operations
        TrackedFunction::new("alloc_buffer", "memory")
            .priority(10)
            .status(AdoptionStatus::Validated)
            .notes("Buffer allocation with size check"),
        TrackedFunction::new("copy_buffer", "memory")
            .priority(9)
            .status(AdoptionStatus::Validated)
            .notes("Buffer copy with bounds check"),
        
        // GPU operations
        TrackedFunction::new("dispatch_compute", "gpu")
            .priority(10)
            .status(AdoptionStatus::Validated)
            .notes("Compute dispatch with workgroup size check"),
        TrackedFunction::new("bind_buffer", "gpu")
            .priority(9)
            .status(AdoptionStatus::Validated)
            .notes("Buffer binding with alignment check"),
        
        // String operations
        TrackedFunction::new("parse_int", "parse")
            .priority(8)
            .status(AdoptionStatus::Validated)
            .notes("Integer parsing with non-empty check"),
        TrackedFunction::new("validate_utf8", "parse")
            .priority(8)
            .status(AdoptionStatus::Validated)
            .notes("UTF-8 validation"),
        
        // Collection operations
        TrackedFunction::new("get_index", "collections")
            .priority(9)
            .status(AdoptionStatus::Validated)
            .notes("Index access with bounds check"),
    ]
}

/// Create initial rollout tracker with high-value functions.
pub fn create_initial_tracker() -> RolloutTracker {
    let mut tracker = RolloutTracker::new();
    tracker.set_phase(1);

    for func in high_value_functions() {
        tracker.track(func);
    }

    tracker
}

/// Validate that a contract macro expansion is correct.
pub fn validate_expansion(original: &str, expanded: &str) -> ValidationResult {
    let mut result = ValidationResult::new();

    // Check that original function name is preserved
    if !expanded.contains("fn ") {
        result.add_error("Missing function definition");
    }

    // Check for debug_assert! presence (runtime checks)
    if original.contains("#[requires") || original.contains("#![requires") {
        if !expanded.contains("debug_assert!") {
            result.add_warning("No debug_assert! found for requires");
        }
    }

    // Check for result binding if ensures present
    if original.contains("#[ensures") || original.contains("#![ensures") {
        if !expanded.contains("__contract_result") && !expanded.contains("result") {
            result.add_warning("No result binding found for ensures");
        }
    }

    result
}

/// Result of validating a macro expansion.
#[derive(Debug, Default)]
pub struct ValidationResult {
    /// Errors found.
    pub errors: Vec<String>,
    /// Warnings found.
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a new result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an error.
    pub fn add_error(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
    }

    /// Add a warning.
    pub fn add_warning(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }

    /// Check if valid.
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Check if clean (no errors or warnings).
    pub fn is_clean(&self) -> bool {
        self.errors.is_empty() && self.warnings.is_empty()
    }
}
