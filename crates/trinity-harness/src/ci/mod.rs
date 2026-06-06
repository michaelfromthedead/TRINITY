//! CI workflow generation and validation.
//!
//! Provides utilities for generating GitHub Actions workflows
//! that use trinity-harness for incremental testing.

/// GitHub Actions workflow configuration.
#[derive(Debug, Clone)]
pub struct WorkflowConfig {
    /// Workflow name.
    pub name: String,
    /// Trigger branches.
    pub branches: Vec<String>,
    /// Whether to enable workflow_dispatch.
    pub manual_trigger: bool,
    /// Cache configuration.
    pub cache_enabled: bool,
    /// Rust version.
    pub rust_version: String,
    /// Python version.
    pub python_version: String,
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            name: "Trinity Harness".to_string(),
            branches: vec!["master".to_string(), "main".to_string()],
            manual_trigger: true,
            cache_enabled: true,
            rust_version: "stable".to_string(),
            python_version: "3.13".to_string(),
        }
    }
}

impl WorkflowConfig {
    /// Create a new workflow config.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set trigger branches.
    pub fn branches(mut self, branches: Vec<String>) -> Self {
        self.branches = branches;
        self
    }

    /// Enable or disable manual trigger.
    pub fn manual_trigger(mut self, enabled: bool) -> Self {
        self.manual_trigger = enabled;
        self
    }

    /// Enable or disable caching.
    pub fn cache(mut self, enabled: bool) -> Self {
        self.cache_enabled = enabled;
        self
    }

    /// Set Rust version.
    pub fn rust_version(mut self, version: impl Into<String>) -> Self {
        self.rust_version = version.into();
        self
    }

    /// Set Python version.
    pub fn python_version(mut self, version: impl Into<String>) -> Self {
        self.python_version = version.into();
        self
    }
}

/// A workflow step.
#[derive(Debug, Clone)]
pub struct WorkflowStep {
    /// Step name.
    pub name: String,
    /// Step ID (optional).
    pub id: Option<String>,
    /// Run command (optional).
    pub run: Option<String>,
    /// Uses action (optional).
    pub uses: Option<String>,
    /// Condition (optional).
    pub condition: Option<String>,
}

impl WorkflowStep {
    /// Create a run step.
    pub fn run(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            id: None,
            run: Some(command.into()),
            uses: None,
            condition: None,
        }
    }

    /// Create a uses step.
    pub fn uses(name: impl Into<String>, action: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            id: None,
            run: None,
            uses: Some(action.into()),
            condition: None,
        }
    }

    /// Set step ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set condition.
    pub fn with_condition(mut self, condition: impl Into<String>) -> Self {
        self.condition = Some(condition.into());
        self
    }
}

/// Generate workflow steps for harness integration.
pub fn generate_harness_steps() -> Vec<WorkflowStep> {
    vec![
        WorkflowStep::uses("Checkout", "actions/checkout@v4"),
        WorkflowStep::run("Build Harness", "cargo build -p trinity-harness --release"),
        WorkflowStep::run("Query Stale Tests", "echo 'Querying stale tests...'")
            .with_id("query"),
        WorkflowStep::run("Run Stale Tests", "cargo test -p trinity-harness")
            .with_condition("steps.query.outputs.stale_count != '0'"),
        WorkflowStep::run("Update State", "echo 'Updating state...'"),
    ]
}

/// Validate a workflow configuration.
pub fn validate_workflow(config: &WorkflowConfig) -> ValidationResult {
    let mut result = ValidationResult::new();

    if config.name.is_empty() {
        result.add_error("Workflow name is required");
    }

    if config.branches.is_empty() {
        result.add_error("At least one trigger branch is required");
    }

    if config.rust_version.is_empty() {
        result.add_warning("Rust version not specified, using 'stable'");
    }

    if config.python_version.is_empty() {
        result.add_warning("Python version not specified");
    }

    result
}

/// Result of workflow validation.
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    /// Whether validation passed.
    pub is_valid: bool,
    /// Errors found.
    pub errors: Vec<String>,
    /// Warnings found.
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a new result.
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add an error.
    pub fn add_error(&mut self, error: impl Into<String>) {
        self.is_valid = false;
        self.errors.push(error.into());
    }

    /// Add a warning.
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }

    /// Check if validation passed.
    pub fn passed(&self) -> bool {
        self.is_valid && self.errors.is_empty()
    }
}

/// Generate YAML for a workflow.
pub fn generate_yaml(config: &WorkflowConfig) -> String {
    let mut yaml = String::new();

    yaml.push_str(&format!("name: {}\n\n", config.name));

    yaml.push_str("on:\n");
    yaml.push_str("  push:\n");
    yaml.push_str("    branches:\n");
    for branch in &config.branches {
        yaml.push_str(&format!("      - {}\n", branch));
    }

    yaml.push_str("  pull_request:\n");
    yaml.push_str("    branches:\n");
    for branch in &config.branches {
        yaml.push_str(&format!("      - {}\n", branch));
    }

    if config.manual_trigger {
        yaml.push_str("  workflow_dispatch:\n");
    }

    yaml.push_str("\njobs:\n");
    yaml.push_str("  harness:\n");
    yaml.push_str("    runs-on: ubuntu-latest\n");
    yaml.push_str("    steps:\n");

    for step in generate_harness_steps() {
        yaml.push_str(&format!("      - name: {}\n", step.name));
        if let Some(uses) = &step.uses {
            yaml.push_str(&format!("        uses: {}\n", uses));
        }
        if let Some(run) = &step.run {
            yaml.push_str(&format!("        run: {}\n", run));
        }
        if let Some(condition) = &step.condition {
            yaml.push_str(&format!("        if: {}\n", condition));
        }
    }

    yaml
}
