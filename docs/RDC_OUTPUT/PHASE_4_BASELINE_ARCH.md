# PHASE 4: Baseline Run — Architecture

**Duration:** 1 day
**Depends On:** Phase 3 (Test Mapping)

---

## Overview

Run ALL tests once. Mark passing nodes GREEN, failing nodes RED. Establish baseline state.

## Components

### 4.1 Test Runner Integration

```rust
pub struct TestRunner {
    db: HarnessDb,
}

impl TestRunner {
    pub fn run_all(&self) -> Result<TestResults> {
        // Run Rust tests
        let rust_results = self.run_cargo_test()?;
        
        // Run Python tests
        let python_results = self.run_pytest()?;
        
        // Process results
        self.process_results(&rust_results, &python_results)
    }
    
    fn run_cargo_test(&self) -> Result<RustTestResults> {
        let output = Command::new("cargo")
            .args(["test", "--", "--format=json"])
            .output()?;
        parse_cargo_test_json(&output.stdout)
    }
    
    fn run_pytest(&self) -> Result<PythonTestResults> {
        let output = Command::new("uv")
            .args(["run", "pytest", "--json-report", "-q"])
            .output()?;
        parse_pytest_json(&output.stdout)
    }
}
```

### 4.2 State Assignment

```rust
pub fn assign_state(&self, test_result: &TestResult) {
    let target_nodes = self.db.get_test_targets(test_result.test_id)?;
    
    for node_id in target_nodes {
        let event = if test_result.passed {
            CodeEvent::TestsPassed
        } else {
            CodeEvent::TestsFailed
        };
        
        self.db.transition_state(node_id, event)?;
    }
}
```

## Acceptance Criteria

- [ ] All Rust tests run
- [ ] All Python tests run
- [ ] Results parsed from JSON output
- [ ] Each test result mapped to target nodes
- [ ] Passing tests → nodes marked GREEN
- [ ] Failing tests → nodes marked RED
- [ ] Baseline state stored in database
