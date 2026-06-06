//! Documentation generation for trinity-harness.
//!
//! Provides utilities for generating documentation about
//! daemon operation, CI integration, and CLI commands.

/// Documentation about daemon operation.
pub fn daemon_docs() -> String {
    r#"# Trinity Harness Daemon

## Overview

The trinity-harness daemon monitors code changes and maintains test state
across your project. It runs continuously in the background, tracking which
code has changed and which tests need to be re-run.

## Starting the Daemon

```bash
trinity-harness daemon
```

Options:
- `--verbose`: Enable verbose logging
- `--poll-interval <ms>`: Set polling interval (default: 1000ms)
- `--debounce <ms>`: Set debounce time (default: 100ms)

## How It Works

1. **File Watching**: Monitors source files for changes
2. **State Tracking**: Maintains state (Green/Red/Dirty/Untested) per node
3. **Event Processing**: Processes file changes and triggers state transitions
4. **Staleness Propagation**: Marks dependent code as stale when dependencies change

## State Machine

- **Untested**: Node has never been tested
- **Green**: All tests passing
- **Red**: Tests failing
- **Dirty**: Code changed since last test

## Events

- `FileCreated`: New file detected
- `FileModified`: File content changed
- `FileDeleted`: File removed
- `StateChanged`: Node state transition
- `Started/Stopped`: Daemon lifecycle

## Notifications

Subscribe to state changes:
- `state`: State change events
- `files`: File change events

Webhook support available for CI integration.
"#.to_string()
}

/// Documentation about CI integration.
pub fn ci_docs() -> String {
    r#"# CI Integration

## GitHub Actions Workflow

Trinity-harness provides a GitHub Actions workflow at `.github/workflows/harness.yml`.

## Workflow Steps

1. **Checkout**: Clone repository with full history
2. **Setup**: Install Rust and Python
3. **Cache**: Cache Cargo dependencies
4. **Build Harness**: Build trinity-harness
5. **Query Stale Tests**: Identify tests that need running
6. **Run Stale Tests**: Execute only changed tests
7. **Update State**: Record test results

## Configuration

```yaml
name: Trinity Harness

on:
  push:
    branches: [master, main]
  pull_request:
  workflow_dispatch:
    inputs:
      force_full:
        description: 'Force full test run'
        default: 'false'
```

## Manual Trigger

Run the workflow manually with:
- Go to Actions tab
- Select "Trinity Harness" workflow
- Click "Run workflow"
- Optionally enable "Force full test run"

## Incremental Testing

The workflow only runs tests for changed code:
1. Query which nodes are Dirty/Untested/Red
2. Run tests only for those nodes
3. Update state based on results

## Best Practices

- Keep test mappings up to date
- Use `workflow_dispatch` for full runs when needed
- Review test coverage reports
- Monitor for flaky tests
"#.to_string()
}

/// Documentation about CLI commands.
pub fn cli_docs() -> String {
    r#"# CLI Commands

## Available Commands

### `daemon`

Start the background daemon.

```bash
trinity-harness daemon [options]
```

Options:
- `--verbose`, `-v`: Enable verbose output
- `--project-root <path>`: Set project root (default: current directory)

### `query needs-testing`

List nodes that need testing.

```bash
trinity-harness query needs-testing [options]
```

Options:
- `--format <text|json>`: Output format (default: text)

Output:
```
Nodes needing testing: 5/100
  Dirty func_a (src/lib.rs)
  Untested func_b (src/main.rs)
  Red test_c (tests/test.rs)
```

### `run-stale`

Run tests only for stale nodes.

```bash
trinity-harness run-stale [options]
```

This runs:
1. Cargo tests for affected Rust code
2. Pytest for affected Python code

### `update-from-results`

Update state from test results.

```bash
trinity-harness update-from-results [results-file]
```

If no file is provided, runs tests and updates state automatically.

## Exit Codes

- `0`: Success
- `1`: Error (check message for details)

## Examples

```bash
# Start daemon in verbose mode
trinity-harness daemon --verbose

# Query stale tests as JSON
trinity-harness query needs-testing --format json

# Run only what's needed
trinity-harness run-stale

# Update state from CI results
trinity-harness update-from-results ci-results.json
```
"#.to_string()
}

/// Generate all documentation.
pub fn generate_all() -> Documentation {
    Documentation {
        daemon: daemon_docs(),
        ci: ci_docs(),
        cli: cli_docs(),
    }
}

/// Documentation collection.
#[derive(Debug, Clone)]
pub struct Documentation {
    /// Daemon documentation.
    pub daemon: String,
    /// CI documentation.
    pub ci: String,
    /// CLI documentation.
    pub cli: String,
}

impl Documentation {
    /// Generate combined documentation.
    pub fn combined(&self) -> String {
        format!(
            "{}\n\n---\n\n{}\n\n---\n\n{}",
            self.daemon, self.ci, self.cli
        )
    }

    /// Get word count.
    pub fn word_count(&self) -> usize {
        self.daemon.split_whitespace().count()
            + self.ci.split_whitespace().count()
            + self.cli.split_whitespace().count()
    }

    /// Get section count.
    pub fn section_count(&self) -> usize {
        self.daemon.matches("## ").count()
            + self.ci.matches("## ").count()
            + self.cli.matches("## ").count()
    }
}

/// Documentation section.
#[derive(Debug, Clone)]
pub struct DocSection {
    /// Section title.
    pub title: String,
    /// Section content.
    pub content: String,
    /// Subsections.
    pub subsections: Vec<DocSection>,
}

impl DocSection {
    /// Create a new section.
    pub fn new(title: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            content: content.into(),
            subsections: Vec::new(),
        }
    }

    /// Add a subsection.
    pub fn with_subsection(mut self, section: DocSection) -> Self {
        self.subsections.push(section);
        self
    }

    /// Render to markdown.
    pub fn to_markdown(&self, level: usize) -> String {
        let prefix = "#".repeat(level);
        let mut md = format!("{} {}\n\n{}\n", prefix, self.title, self.content);

        for sub in &self.subsections {
            md.push_str(&sub.to_markdown(level + 1));
        }

        md
    }
}

/// Validate documentation.
pub fn validate_docs(docs: &Documentation) -> DocValidation {
    let mut result = DocValidation::new();

    // Check daemon docs
    if docs.daemon.is_empty() {
        result.add_error("Daemon documentation is empty");
    }
    if !docs.daemon.contains("## ") {
        result.add_warning("Daemon docs missing sections");
    }

    // Check CI docs
    if docs.ci.is_empty() {
        result.add_error("CI documentation is empty");
    }
    if !docs.ci.contains("## ") {
        result.add_warning("CI docs missing sections");
    }

    // Check CLI docs
    if docs.cli.is_empty() {
        result.add_error("CLI documentation is empty");
    }
    if !docs.cli.contains("## ") {
        result.add_warning("CLI docs missing sections");
    }

    result
}

/// Documentation validation result.
#[derive(Debug, Clone, Default)]
pub struct DocValidation {
    /// Whether validation passed.
    pub is_valid: bool,
    /// Errors found.
    pub errors: Vec<String>,
    /// Warnings found.
    pub warnings: Vec<String>,
}

impl DocValidation {
    /// Create a new validation result.
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
