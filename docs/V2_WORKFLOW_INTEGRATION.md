# V2 Workflow Integration

**Started:** 2026-06-05
**Status:** DESIGN PHASE
**Depends On:** V2_SUPERSTATE_VISION.md, V2_SUPERSQLITE_PERSISTENCE.md

---

## Executive Summary

This document specifies the **engine** that keeps the code state system running. It covers:

- **Event Sources** — What generates events (file changes, commits, test results)
- **Event Processing** — How events trigger state transitions
- **Propagation Rules** — How staleness spreads through dependencies
- **CI Pipeline** — What runs automatically
- **Daemon Mode** — Long-running background service

The workflow integration is the bridge between the physical world (files, tests, commits) and the state model (the CodeGraph with statecharts).

---

## Part 1: Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         WORKFLOW INTEGRATION                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          EVENT SOURCES                                  │ │
│  │                                                                         │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │ │
│  │  │ FILE WATCHER │  │  SUPERFOSSIL │  │  CI PIPELINE │  │   MANUAL   │ │ │
│  │  │              │  │              │  │              │  │            │ │ │
│  │  │ inotify/kq   │  │ commit hooks │  │ test results │  │ CLI/API    │ │ │
│  │  │ notify crate │  │ pre/post     │  │ lint results │  │ commands   │ │ │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │ │
│  │         │                 │                 │                │        │ │
│  └─────────┼─────────────────┼─────────────────┼────────────────┼────────┘ │
│            │                 │                 │                │          │
│            └─────────────────┴─────────────────┴────────────────┘          │
│                                      │                                      │
│                                      ▼                                      │
│                         ┌────────────────────────┐                         │
│                         │     EVENT INGESTER     │                         │
│                         │                        │                         │
│                         │ • Normalize events     │                         │
│                         │ • Debounce duplicates  │                         │
│                         │ • Validate payloads    │                         │
│                         │ • Assign sequence      │                         │
│                         └───────────┬────────────┘                         │
│                                     │                                      │
│                                     ▼                                      │
│                         ┌────────────────────────┐                         │
│                         │    SUPERSQLITE         │                         │
│                         │    code_events         │                         │
│                         │    (append-only log)   │                         │
│                         └───────────┬────────────┘                         │
│                                     │                                      │
│            ┌────────────────────────┼────────────────────────┐             │
│            │                        │                        │             │
│            ▼                        ▼                        ▼             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐     │
│  │  STATE UPDATER   │    │   PROPAGATION    │    │  NOTIFICATION    │     │
│  │                  │    │     ENGINE       │    │    SERVICE       │     │
│  │ • Update node    │    │                  │    │                  │     │
│  │   current_state  │    │ • BFS traversal  │    │ • Pub/sub        │     │
│  │ • Write history  │    │ • Staleness      │    │ • UI updates     │     │
│  │ • Record event   │    │   spreading      │    │ • Webhooks       │     │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Event Sources

### 2.1 File Watcher

The file watcher monitors the filesystem for changes and generates events.

```rust
use notify::{Watcher, RecursiveMode, watcher, DebouncedEvent};
use std::sync::mpsc::channel;
use std::time::Duration;
use std::path::{Path, PathBuf};
use std::fs;

pub struct FileWatcher {
    db: HarnessDb,
    parsers: ParserRegistry,
    roots: Vec<PathBuf>,
}

impl FileWatcher {
    pub fn new(db: HarnessDb, roots: Vec<PathBuf>) -> Self {
        Self {
            db,
            parsers: ParserRegistry::new(),
            roots,
        }
    }
    
    /// Start watching — blocks forever
    pub fn run(&self) -> Result<()> {
        let (tx, rx) = channel();
        
        // 100ms debounce to batch rapid saves
        let mut watcher = watcher(tx, Duration::from_millis(100))?;
        
        for root in &self.roots {
            println!("Watching: {:?}", root);
            watcher.watch(root, RecursiveMode::Recursive)?;
        }
        
        loop {
            match rx.recv() {
                Ok(event) => {
                    if let Err(e) = self.handle_event(event) {
                        eprintln!("Error handling event: {:?}", e);
                    }
                }
                Err(e) => {
                    eprintln!("Watch error: {:?}", e);
                }
            }
        }
    }
    
    fn handle_event(&self, event: DebouncedEvent) -> Result<()> {
        match event {
            DebouncedEvent::Write(path) => {
                self.on_file_modified(&path)?;
            }
            DebouncedEvent::Create(path) => {
                self.on_file_created(&path)?;
            }
            DebouncedEvent::Remove(path) => {
                self.on_file_removed(&path)?;
            }
            DebouncedEvent::Rename(from, to) => {
                self.on_file_removed(&from)?;
                self.on_file_created(&to)?;
            }
            DebouncedEvent::Chmod(_) => {
                // Ignore permission changes
            }
            DebouncedEvent::Rescan => {
                self.full_rescan()?;
            }
            _ => {}
        }
        Ok(())
    }
    
    fn on_file_modified(&self, path: &Path) -> Result<()> {
        let lang = Language::from_path(path);
        if lang == Language::Unknown {
            return Ok(()); // Not a source file
        }
        
        // Skip non-existent files (deleted between event and handler)
        if !path.exists() {
            return Ok(());
        }
        
        // Read and hash
        let source = fs::read_to_string(path)?;
        let new_hash = blake3::hash(source.as_bytes());
        
        // Get existing nodes for this file
        let existing = self.db.get_nodes_by_file(path)?;
        
        // Quick check: if hash unchanged, skip
        if existing.len() == 1 && existing[0].hashes.full == new_hash.as_bytes() {
            return Ok(()); // No real change
        }
        
        // Parse the file
        let new_units = self.parsers.parse_file(path, &source, lang)?;
        
        // Diff: find what changed
        let changes = self.diff_units(&existing, &new_units);
        
        // Process changes
        for change in changes {
            match change {
                UnitChange::Added(unit) => {
                    self.db.upsert_node(&unit)?;
                    self.db.append_event(&CodeEventBuilder::new(EventType::NodeCreated)
                        .node_id(&unit.node_id)
                        .build())?;
                }
                UnitChange::Removed(node_id) => {
                    self.db.delete_node(&node_id)?;
                    self.db.append_event(&CodeEventBuilder::new(EventType::NodeDeleted)
                        .node_id(&node_id)
                        .build())?;
                }
                UnitChange::Modified { node_id, change_type, old, new } => {
                    self.db.upsert_node(&new)?;
                    
                    let event_type = match change_type {
                        ChangeType::Signature => EventType::SignatureChanged,
                        ChangeType::Body => EventType::BodyChanged,
                        ChangeType::Layout => EventType::LayoutChanged,
                        ChangeType::Full => EventType::SourceChanged,
                    };
                    
                    self.db.append_event(&CodeEventBuilder::new(event_type)
                        .node_id(&node_id)
                        .payload(json!({
                            "old_hash": hex::encode(old.hashes.full),
                            "new_hash": hex::encode(new.hashes.full),
                        }))
                        .build())?;
                }
            }
        }
        
        Ok(())
    }
    
    fn on_file_created(&self, path: &Path) -> Result<()> {
        // Same as modified for new files
        self.on_file_modified(path)
    }
    
    fn on_file_removed(&self, path: &Path) -> Result<()> {
        let nodes = self.db.get_nodes_by_file(path)?;
        
        for node in nodes {
            self.db.delete_node(&node.node_id)?;
            self.db.append_event(&CodeEventBuilder::new(EventType::NodeDeleted)
                .node_id(&node.node_id)
                .payload(json!({ "file": path.to_string_lossy() }))
                .build())?;
        }
        
        Ok(())
    }
    
    fn full_rescan(&self) -> Result<()> {
        println!("Performing full rescan...");
        
        // Walk all source files
        for root in &self.roots {
            for entry in walkdir::WalkDir::new(root)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
            {
                let path = entry.path();
                if Language::from_path(path) != Language::Unknown {
                    self.on_file_modified(path)?;
                }
            }
        }
        
        // Emit rescan complete event
        self.db.append_event(&CodeEventBuilder::new(EventType::FullReparse)
            .payload(json!({ "roots": self.roots }))
            .build())?;
        
        Ok(())
    }
    
    fn diff_units(&self, old: &[CodeNode], new: &[CodeUnit]) -> Vec<UnitChange> {
        let mut changes = Vec::new();
        
        let old_map: HashMap<&str, &CodeNode> = old.iter()
            .map(|n| (n.node_id.as_str(), n))
            .collect();
        
        let new_map: HashMap<&str, &CodeUnit> = new.iter()
            .map(|u| (u.node_id.as_str(), u))
            .collect();
        
        // Find added and modified
        for (id, new_unit) in &new_map {
            if let Some(old_node) = old_map.get(id) {
                // Exists — check for changes
                let change_type = if old_node.hashes.signature != new_unit.hashes.signature {
                    Some(ChangeType::Signature)
                } else if old_node.hashes.layout != new_unit.hashes.layout {
                    Some(ChangeType::Layout)
                } else if old_node.hashes.body != new_unit.hashes.body {
                    Some(ChangeType::Body)
                } else if old_node.hashes.full != new_unit.hashes.full {
                    Some(ChangeType::Full)
                } else {
                    None // No change
                };
                
                if let Some(ct) = change_type {
                    changes.push(UnitChange::Modified {
                        node_id: id.to_string(),
                        change_type: ct,
                        old: (*old_node).clone(),
                        new: (*new_unit).clone(),
                    });
                }
            } else {
                // New unit
                changes.push(UnitChange::Added((*new_unit).clone()));
            }
        }
        
        // Find removed
        for (id, _) in &old_map {
            if !new_map.contains_key(id) {
                changes.push(UnitChange::Removed(id.to_string()));
            }
        }
        
        changes
    }
}

#[derive(Debug, Clone)]
enum UnitChange {
    Added(CodeUnit),
    Removed(String),
    Modified {
        node_id: String,
        change_type: ChangeType,
        old: CodeNode,
        new: CodeUnit,
    },
}

#[derive(Debug, Clone, Copy)]
enum ChangeType {
    Signature, // Function signature, struct fields changed
    Body,      // Only implementation changed
    Layout,    // Memory layout changed (struct alignment)
    Full,      // Other changes
}
```

### 2.2 Superfossil Hooks

Integration with the Superfossil VCS for commit events.

```rust
use superfossil_sys::{
    fossil_main, fossil_capture_start, fossil_capture_end,
    db_get_connection, db_set_connection,
};

pub struct SuperfossilIntegration {
    db: HarnessDb,
    fossil_db_path: PathBuf,
}

impl SuperfossilIntegration {
    pub fn new(db: HarnessDb, fossil_db_path: PathBuf) -> Self {
        Self { db, fossil_db_path }
    }
    
    /// Called by Superfossil's pre-commit hook
    /// Returns false to block the commit
    pub fn pre_commit(&self) -> Result<bool> {
        println!("Running pre-commit checks...");
        
        // 1. Check for Rust/WGSL layout mismatches
        let mismatches = self.db.find_layout_mismatches()?;
        if !mismatches.is_empty() {
            eprintln!("ERROR: Layout mismatches detected!");
            eprintln!("These Rust structs have different memory layouts than their WGSL counterparts:");
            eprintln!();
            for m in &mismatches {
                eprintln!("  Struct: {}", m.struct_name);
                eprintln!("    Rust:  {} ({} bytes, align {})", m.rust_file, m.rust_size, m.rust_align);
                eprintln!("    WGSL:  {} ({} bytes, align {})", m.wgsl_file, m.wgsl_size, m.wgsl_align);
                eprintln!();
            }
            eprintln!("Fix the layouts before committing.");
            return Ok(false);
        }
        
        // 2. Check for quarantined code
        let quarantined = self.db.nodes_in_state(CodeState::Quarantined)?;
        if !quarantined.is_empty() {
            eprintln!("WARNING: Committing with quarantined code:");
            for n in &quarantined {
                eprintln!("  - {}", n.qualified_name.as_deref().unwrap_or(&n.name));
            }
            eprintln!();
            // Warning only, don't block
        }
        
        // 3. Check for RED tests
        let red = self.db.nodes_in_state(CodeState::TestedRed)?;
        if !red.is_empty() {
            eprintln!("WARNING: Committing with failing tests:");
            for n in &red {
                eprintln!("  - {}", n.qualified_name.as_deref().unwrap_or(&n.name));
            }
            eprintln!();
            // Warning only, don't block (might be intentional WIP)
        }
        
        Ok(true)
    }
    
    /// Called by Superfossil's post-commit hook
    pub fn post_commit(&self, commit_info: &CommitInfo) -> Result<()> {
        println!("Processing commit: {}", commit_info.hash);
        
        // Record commit event
        self.db.append_event(&CodeEventBuilder::new(EventType::Commit)
            .payload(json!({
                "commit_hash": commit_info.hash,
                "author": commit_info.author,
                "message": commit_info.message,
                "timestamp": commit_info.timestamp,
                "files_changed": commit_info.files,
            }))
            .build())?;
        
        // The file watcher should have already processed the changes,
        // but we emit a CommitComplete event for correlation
        self.db.append_event(&CodeEventBuilder::new(EventType::CommitComplete)
            .correlation_id(&commit_info.hash)
            .build())?;
        
        Ok(())
    }
    
    /// Get list of files changed since last sync
    pub fn get_changed_files(&self) -> Result<Vec<PathBuf>> {
        // Use Fossil's diffing
        fossil_capture_start();
        let args = vec!["fossil", "changes", "--classify"];
        let argc = args.len() as i32;
        let argv: Vec<*mut i8> = args.iter()
            .map(|s| CString::new(*s).unwrap().into_raw())
            .collect();
        
        unsafe {
            fossil_main(argc, argv.as_ptr() as *mut *mut i8);
        }
        
        let mut len: i32 = 0;
        let output = unsafe {
            let ptr = fossil_capture_end(&mut len);
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        };
        
        // Parse output
        let files: Vec<PathBuf> = output
            .lines()
            .filter_map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    Some(PathBuf::from(parts[1]))
                } else {
                    None
                }
            })
            .collect();
        
        Ok(files)
    }
    
    /// Get commit history for a file
    pub fn get_file_history(&self, path: &Path) -> Result<Vec<CommitInfo>> {
        // Use Fossil's finfo
        fossil_capture_start();
        let path_str = path.to_string_lossy();
        let args = vec!["fossil", "finfo", "-b", &path_str];
        // ... similar capture pattern
        
        // Parse and return
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct CommitInfo {
    pub hash: String,
    pub author: String,
    pub message: String,
    pub timestamp: String,
    pub files: Vec<String>,
}
```

### 2.3 CI Pipeline Integration

Ingesting results from CI runs (tests, lints, etc.).

```rust
use serde::{Deserialize, Serialize};

pub struct CIPipelineIntegration {
    db: HarnessDb,
}

impl CIPipelineIntegration {
    pub fn new(db: HarnessDb) -> Self {
        Self { db }
    }
    
    // =========================================================================
    // TEST RESULTS
    // =========================================================================
    
    /// Ingest test results from cargo test JSON output
    pub fn ingest_cargo_test_results(&self, json_path: &Path) -> Result<IngestReport> {
        let content = fs::read_to_string(json_path)?;
        let results: CargoTestOutput = serde_json::from_str(&content)?;
        
        let mut report = IngestReport::default();
        
        for test in results.tests {
            // Map test name to code unit
            let node_id = self.map_test_to_node(&test.name)?;
            
            // Record test result
            let outcome = match test.status.as_str() {
                "ok" => TestOutcome::Passed,
                "failed" => TestOutcome::Failed,
                "ignored" => TestOutcome::Skipped,
                _ => TestOutcome::Error,
            };
            
            self.db.insert_test_result(&TestResult {
                node_id: node_id.clone(),
                test_name: test.name.clone(),
                outcome: outcome.clone(),
                duration_ms: test.exec_time_ms,
                error_message: test.message.clone(),
                stdout: test.stdout.clone(),
                ..Default::default()
            })?;
            
            // Emit event
            let event_type = match outcome {
                TestOutcome::Passed => EventType::TestsPassed,
                TestOutcome::Failed => EventType::TestsFailed,
                TestOutcome::Skipped => EventType::TestsSkipped,
                TestOutcome::Error => EventType::TestsError,
            };
            
            self.db.append_event(&CodeEventBuilder::new(event_type)
                .node_id(&node_id)
                .payload(json!({
                    "test_name": test.name,
                    "duration_ms": test.exec_time_ms,
                    "message": test.message,
                }))
                .build())?;
            
            report.tests_processed += 1;
            match outcome {
                TestOutcome::Passed => report.passed += 1,
                TestOutcome::Failed => report.failed += 1,
                TestOutcome::Skipped => report.skipped += 1,
                TestOutcome::Error => report.errors += 1,
            }
        }
        
        Ok(report)
    }
    
    /// Ingest pytest results (for Python code)
    pub fn ingest_pytest_results(&self, json_path: &Path) -> Result<IngestReport> {
        let content = fs::read_to_string(json_path)?;
        let results: PytestOutput = serde_json::from_str(&content)?;
        
        // Similar processing...
        todo!()
    }
    
    // =========================================================================
    // LINT RESULTS
    // =========================================================================
    
    /// Ingest clippy output
    pub fn ingest_clippy_results(&self, json_path: &Path) -> Result<IngestReport> {
        let content = fs::read_to_string(json_path)?;
        let mut report = IngestReport::default();
        
        for line in content.lines() {
            if line.is_empty() { continue; }
            
            let msg: ClippyMessage = serde_json::from_str(line)?;
            
            if let Some(span) = msg.spans.first() {
                // Find the node at this location
                let node_id = self.find_node_at(&span.file_name, span.line_start)?;
                
                if let Some(id) = node_id {
                    self.db.append_event(&CodeEventBuilder::new(EventType::LintWarning)
                        .node_id(&id)
                        .payload(json!({
                            "lint_code": msg.code.as_ref().map(|c| &c.code),
                            "message": msg.message,
                            "level": msg.level,
                            "file": span.file_name,
                            "line": span.line_start,
                        }))
                        .build())?;
                    
                    report.lints_processed += 1;
                }
            }
        }
        
        Ok(report)
    }
    
    /// Ingest mypy results (for Python)
    pub fn ingest_mypy_results(&self, output_path: &Path) -> Result<IngestReport> {
        // Parse mypy output format
        todo!()
    }
    
    // =========================================================================
    // HELPERS
    // =========================================================================
    
    /// Map a test name like "tests::math::test_add" to a node_id
    fn map_test_to_node(&self, test_name: &str) -> Result<String> {
        // Strategy 1: Direct lookup by qualified name
        if let Some(node) = self.db.get_node_by_qualified_name(test_name)? {
            // It's a test node — find what it tests
            if let Some(tested) = self.db.get_tested_by(&node.node_id)?.first() {
                return Ok(tested.clone());
            }
            return Ok(node.node_id);
        }
        
        // Strategy 2: Pattern matching
        // "test_foo" likely tests "foo"
        if test_name.starts_with("test_") {
            let target = &test_name[5..]; // Remove "test_"
            if let Some(node) = self.db.search_node_by_name(target)? {
                return Ok(node.node_id);
            }
        }
        
        // Strategy 3: File-based (blackbox_foo.rs tests foo.rs)
        // ...
        
        // Fallback: Create a synthetic test node
        Ok(format!("test:{}", test_name))
    }
    
    fn find_node_at(&self, file: &str, line: u32) -> Result<Option<String>> {
        self.db.query_row(r#"
            SELECT node_id FROM code_nodes
            WHERE file_path = ?1
              AND span_start_line <= ?2
              AND span_end_line >= ?2
            ORDER BY (span_end_line - span_start_line) ASC
            LIMIT 1
        "#, params![file, line], |row| row.get(0)).optional()
    }
    
    // =========================================================================
    // SELECTIVE TESTING
    // =========================================================================
    
    /// Get list of tests that need to run (test stale code only)
    pub fn get_required_tests(&self) -> Result<Vec<String>> {
        // Find all stale nodes
        let stale = self.db.stale_nodes()?;
        
        let mut test_names = HashSet::new();
        
        for node in stale {
            // Find tests that cover this node
            let tests = self.db.get_tests_for(&node.node_id)?;
            for test in tests {
                test_names.insert(test.test_name);
            }
        }
        
        Ok(test_names.into_iter().collect())
    }
    
    /// Generate cargo test filter for stale tests
    pub fn get_cargo_test_filter(&self) -> Result<String> {
        let tests = self.get_required_tests()?;
        
        if tests.is_empty() {
            return Ok(String::new());
        }
        
        // cargo test accepts patterns
        // For exact matching, we'd need to run multiple times
        // or use a regex filter
        Ok(tests.join(" "))
    }
}

#[derive(Debug, Default)]
pub struct IngestReport {
    pub tests_processed: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub errors: usize,
    pub lints_processed: usize,
}

#[derive(Debug, Deserialize)]
struct CargoTestOutput {
    tests: Vec<CargoTestResult>,
}

#[derive(Debug, Deserialize)]
struct CargoTestResult {
    name: String,
    status: String,
    exec_time_ms: Option<u64>,
    message: Option<String>,
    stdout: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ClippyMessage {
    message: String,
    code: Option<ClippyCode>,
    level: String,
    spans: Vec<ClippySpan>,
}

#[derive(Debug, Deserialize)]
struct ClippyCode {
    code: String,
}

#[derive(Debug, Deserialize)]
struct ClippySpan {
    file_name: String,
    line_start: u32,
    line_end: u32,
}
```

### 2.4 Manual Events API

For programmatic or CLI-triggered events.

```rust
pub struct ManualEventApi {
    db: HarnessDb,
}

impl ManualEventApi {
    /// Mark a node as QA approved
    pub fn qa_approve(&self, node_id: &str, approver: &str) -> Result<()> {
        self.db.append_event(&CodeEventBuilder::new(EventType::QaApproved)
            .node_id(node_id)
            .payload(json!({
                "approver": approver,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }))
            .build())?;
        Ok(())
    }
    
    /// Quarantine a node
    pub fn quarantine(&self, node_id: &str, reason: &str) -> Result<()> {
        self.db.append_event(&CodeEventBuilder::new(EventType::Quarantine)
            .node_id(node_id)
            .payload(json!({ "reason": reason }))
            .build())?;
        Ok(())
    }
    
    /// Release from quarantine
    pub fn release(&self, node_id: &str) -> Result<()> {
        self.db.append_event(&CodeEventBuilder::new(EventType::Release)
            .node_id(node_id)
            .build())?;
        Ok(())
    }
    
    /// Force a full reparse
    pub fn trigger_reparse(&self) -> Result<()> {
        self.db.append_event(&CodeEventBuilder::new(EventType::FullReparse)
            .build())?;
        Ok(())
    }
    
    /// Mark tests as passed (manual override)
    pub fn mark_tests_passed(&self, node_id: &str, reason: &str) -> Result<()> {
        self.db.append_event(&CodeEventBuilder::new(EventType::TestsPassed)
            .node_id(node_id)
            .payload(json!({
                "manual": true,
                "reason": reason,
            }))
            .build())?;
        Ok(())
    }
}
```

---

## Part 3: Event Processing

The event processor reads from the event stream and updates state.

### 3.1 Main Processor Loop

```rust
pub struct EventProcessor {
    db: HarnessDb,
    cursor_name: String,
    propagation: PropagationEngine,
    notification: NotificationService,
}

impl EventProcessor {
    pub fn new(db: HarnessDb) -> Self {
        Self {
            db: db.clone(),
            cursor_name: "event_processor".to_string(),
            propagation: PropagationEngine::new(db.clone()),
            notification: NotificationService::new(db.clone()),
        }
    }
    
    /// Main processing loop — runs forever
    pub async fn run(&self) -> Result<()> {
        println!("Event processor starting...");
        
        loop {
            let batch_result = self.process_batch().await;
            
            match batch_result {
                Ok(0) => {
                    // No events — wait a bit
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                Ok(n) => {
                    println!("Processed {} events", n);
                }
                Err(e) => {
                    eprintln!("Error processing events: {:?}", e);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }
    
    async fn process_batch(&self) -> Result<usize> {
        let cursor = self.db.get_cursor(&self.cursor_name)?;
        let events = self.db.read_events(cursor, 100)?;
        
        if events.is_empty() {
            return Ok(0);
        }
        
        for event in &events {
            self.process_one(event).await?;
        }
        
        // Update cursor
        if let Some(last) = events.last() {
            self.db.set_cursor(&self.cursor_name, last.sequence)?;
        }
        
        Ok(events.len())
    }
    
    async fn process_one(&self, event: &StoredEvent) -> Result<()> {
        match event.event_type {
            // =================================================================
            // CHANGE EVENTS — Trigger state transitions + propagation
            // =================================================================
            EventType::SignatureChanged => {
                if let Some(node_id) = &event.node_id {
                    // Signature change is significant — mark stale + propagate
                    self.transition_state(node_id, CodeState::StaleDirect, event)?;
                    self.propagation.propagate(node_id, PropagationType::Full, event.sequence).await?;
                }
            }
            
            EventType::LayoutChanged => {
                if let Some(node_id) = &event.node_id {
                    // Layout change — also check for Rust/WGSL mismatch
                    self.transition_state(node_id, CodeState::StaleDirect, event)?;
                    self.propagation.propagate(node_id, PropagationType::Full, event.sequence).await?;
                    
                    // Check for layout mismatches
                    self.check_layout_mismatches(node_id, event.sequence)?;
                }
            }
            
            EventType::BodyChanged => {
                if let Some(node_id) = &event.node_id {
                    // Body-only change — mark stale but DON'T propagate
                    // (Dependents care about signature, not body)
                    self.transition_state(node_id, CodeState::StaleDirect, event)?;
                }
            }
            
            EventType::SourceChanged => {
                if let Some(node_id) = &event.node_id {
                    // Generic change — full propagation
                    self.transition_state(node_id, CodeState::StaleDirect, event)?;
                    self.propagation.propagate(node_id, PropagationType::Full, event.sequence).await?;
                }
            }
            
            // =================================================================
            // DEPENDENCY EVENTS — Cascaded from propagation
            // =================================================================
            EventType::DependencyChanged => {
                if let Some(node_id) = &event.node_id {
                    let current = self.db.get_node(node_id)?;
                    if let Some(node) = current {
                        // Only transition if current state is "good"
                        if node.current_state.is_good() {
                            self.transition_state(node_id, CodeState::StaleTransitive, event)?;
                        }
                    }
                }
            }
            
            EventType::TransitiveDependencyChanged => {
                if let Some(node_id) = &event.node_id {
                    let current = self.db.get_node(node_id)?;
                    if let Some(node) = current {
                        if node.current_state.is_good() {
                            self.transition_state(node_id, CodeState::StaleDeep, event)?;
                        }
                    }
                }
            }
            
            // =================================================================
            // TEST EVENTS — State transitions
            // =================================================================
            EventType::TestsPassed => {
                if let Some(node_id) = &event.node_id {
                    self.transition_state(node_id, CodeState::TestedGreen, event)?;
                    self.db.update_node_tested_at(node_id)?;
                }
            }
            
            EventType::TestsFailed => {
                if let Some(node_id) = &event.node_id {
                    self.transition_state(node_id, CodeState::TestedRed, event)?;
                    self.db.update_node_tested_at(node_id)?;
                }
            }
            
            EventType::TestsSkipped => {
                if let Some(node_id) = &event.node_id {
                    self.transition_state(node_id, CodeState::TestedSkipped, event)?;
                }
            }
            
            // =================================================================
            // QA EVENTS
            // =================================================================
            EventType::QaApproved => {
                if let Some(node_id) = &event.node_id {
                    // Only QA approve if currently green
                    let current = self.db.get_node(node_id)?;
                    if let Some(node) = current {
                        if node.current_state == CodeState::TestedGreen {
                            self.transition_state(node_id, CodeState::QaApproved, event)?;
                        }
                    }
                }
            }
            
            EventType::Quarantine => {
                if let Some(node_id) = &event.node_id {
                    self.transition_state(node_id, CodeState::Quarantined, event)?;
                }
            }
            
            EventType::Release => {
                if let Some(node_id) = &event.node_id {
                    // Return to last known state before quarantine
                    let history = self.db.state_timeline(node_id)?;
                    let last_good = history.iter()
                        .rev()
                        .find(|h| h.state != "quarantined")
                        .map(|h| CodeState::from_str(&h.state))
                        .unwrap_or(Ok(CodeState::Untouched))?;
                    
                    self.transition_state(node_id, last_good, event)?;
                }
            }
            
            // =================================================================
            // LIFECYCLE EVENTS
            // =================================================================
            EventType::NodeCreated => {
                if let Some(node_id) = &event.node_id {
                    self.transition_state(node_id, CodeState::Untouched, event)?;
                }
            }
            
            EventType::NodeDeleted => {
                // Node already deleted from DB by file watcher
                // Just record in history
                if let Some(node_id) = &event.node_id {
                    self.db.record_state_change(
                        node_id,
                        CodeState::Deleted,
                        Some(event.sequence),
                        Some(event.event_type.as_str()),
                    )?;
                }
            }
            
            // =================================================================
            // NOTIFICATION EVENTS
            // =================================================================
            EventType::LayoutMismatchDetected => {
                // Notify about critical issue
                self.notification.send(NotificationLevel::Critical, 
                    "Layout mismatch detected", 
                    event.payload.clone()
                )?;
            }
            
            _ => {
                // Unknown or unhandled event type
            }
        }
        
        Ok(())
    }
    
    fn transition_state(&self, node_id: &str, new_state: CodeState, event: &StoredEvent) -> Result<()> {
        self.db.record_state_change(
            node_id,
            new_state,
            Some(event.sequence),
            Some(event.event_type.as_str()),
        )?;
        
        // Notify
        self.notification.send(NotificationLevel::Info,
            &format!("{} -> {:?}", node_id, new_state),
            None,
        )?;
        
        Ok(())
    }
    
    fn check_layout_mismatches(&self, node_id: &str, causation: i64) -> Result<()> {
        // Get the node
        let node = self.db.get_node(node_id)?;
        if let Some(n) = node {
            if n.kind == "rust_struct" || n.kind == "wgsl_struct" {
                // Check for matching struct in other language
                let mismatches = self.db.find_layout_mismatches()?;
                
                for m in mismatches {
                    if m.rust_node_id == node_id || m.wgsl_node_id == node_id {
                        self.db.append_event(&CodeEventBuilder::new(EventType::LayoutMismatchDetected)
                            .node_id(&m.rust_node_id)
                            .causation_id(causation)
                            .payload(json!({
                                "rust_node": m.rust_node_id,
                                "wgsl_node": m.wgsl_node_id,
                                "struct_name": m.struct_name,
                                "rust_size": m.rust_size,
                                "wgsl_size": m.wgsl_size,
                            }))
                            .build())?;
                    }
                }
            }
        }
        
        Ok(())
    }
}

impl CodeState {
    fn is_good(&self) -> bool {
        matches!(self, 
            CodeState::TestedGreen | 
            CodeState::QaApproved
        )
    }
}
```

### 3.2 Propagation Engine

```rust
pub struct PropagationEngine {
    db: HarnessDb,
}

#[derive(Debug, Clone, Copy)]
pub enum PropagationType {
    /// Full propagation — signature/layout changed
    Full,
    /// Shallow propagation — only direct dependents
    Shallow,
}

impl PropagationEngine {
    pub fn new(db: HarnessDb) -> Self {
        Self { db }
    }
    
    /// Propagate staleness from an origin node to its dependents
    pub async fn propagate(
        &self,
        origin: &str,
        prop_type: PropagationType,
        causation: i64,
    ) -> Result<PropagationResult> {
        let mut result = PropagationResult::default();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        // Get direct dependents
        let direct = self.db.dependents(origin, 1)?;
        for dep in direct {
            queue.push_back((dep, 1)); // (node_id, depth)
        }
        
        let max_depth = match prop_type {
            PropagationType::Full => 100,
            PropagationType::Shallow => 1,
        };
        
        while let Some((node_id, depth)) = queue.pop_front() {
            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id.clone());
            
            // Determine staleness level
            let staleness = if depth == 1 {
                CodeState::StaleTransitive
            } else {
                CodeState::StaleDeep
            };
            
            // Check current state
            let current = self.db.get_node(&node_id)?;
            if let Some(node) = current {
                // Don't downgrade a worse state
                if node.current_state.severity() >= staleness.severity() {
                    continue;
                }
                
                // Don't propagate to quarantined nodes
                if node.current_state == CodeState::Quarantined {
                    continue;
                }
            }
            
            // Emit event for this node
            let event_type = if depth == 1 {
                EventType::DependencyChanged
            } else {
                EventType::TransitiveDependencyChanged
            };
            
            self.db.append_event(&CodeEventBuilder::new(event_type)
                .node_id(&node_id)
                .causation_id(causation)
                .payload(json!({
                    "origin": origin,
                    "depth": depth,
                }))
                .build())?;
            
            result.nodes_affected += 1;
            
            // Continue propagation if not at max depth
            if depth < max_depth {
                let next_deps = self.db.dependents(&node_id, 1)?;
                for dep in next_deps {
                    if !visited.contains(&dep) {
                        queue.push_back((dep, depth + 1));
                    }
                }
            }
        }
        
        result.max_depth = visited.iter()
            .filter_map(|_| Some(1)) // Would need to track depth
            .max()
            .unwrap_or(0);
        
        Ok(result)
    }
}

#[derive(Debug, Default)]
pub struct PropagationResult {
    pub nodes_affected: usize,
    pub max_depth: usize,
}

impl CodeState {
    fn severity(&self) -> u8 {
        match self {
            CodeState::TestedGreen => 0,
            CodeState::QaApproved => 0,
            CodeState::Untouched => 1,
            CodeState::Unknown => 2,
            CodeState::TestedSkipped => 3,
            CodeState::StaleDeep => 4,
            CodeState::StaleTransitive => 5,
            CodeState::StaleDirect => 6,
            CodeState::Changed => 7,
            CodeState::TestedRed => 8,
            CodeState::Quarantined => 9,
            CodeState::Deleted => 10,
        }
    }
}
```

---

## Part 4: Notification Service

```rust
pub struct NotificationService {
    db: HarnessDb,
}

#[derive(Debug, Clone, Copy)]
pub enum NotificationLevel {
    Debug,
    Info,
    Warning,
    Critical,
}

impl NotificationService {
    pub fn new(db: HarnessDb) -> Self {
        Self { db }
    }
    
    /// Send a notification (pub/sub)
    pub fn send(&self, level: NotificationLevel, message: &str, payload: Option<Value>) -> Result<()> {
        let notification = json!({
            "level": format!("{:?}", level),
            "message": message,
            "payload": payload,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        
        self.db.conn.execute(
            "SELECT mem_pubsub_publish('notifications', ?1)",
            params![notification.to_string()],
        )?;
        
        // Also write to events for persistence
        if matches!(level, NotificationLevel::Warning | NotificationLevel::Critical) {
            self.db.append_event(&CodeEventBuilder::new(EventType::Notification)
                .payload(notification)
                .build())?;
        }
        
        Ok(())
    }
    
    /// Subscribe to notifications (for UI)
    pub fn subscribe(&self) -> Result<i64> {
        let sub_id: i64 = self.db.conn.query_row(
            "SELECT mem_pubsub_subscribe('notifications')",
            [],
            |row| row.get(0),
        )?;
        Ok(sub_id)
    }
}
```

---

## Part 5: Daemon Mode

The harness daemon runs all components together.

```rust
use tokio::signal;

pub struct HarnessDaemon {
    db: HarnessDb,
    config: DaemonConfig,
}

#[derive(Debug, Clone)]
pub struct DaemonConfig {
    pub watch_paths: Vec<PathBuf>,
    pub fossil_db: Option<PathBuf>,
    pub enable_file_watcher: bool,
    pub enable_event_processor: bool,
}

impl HarnessDaemon {
    pub fn new(db: HarnessDb, config: DaemonConfig) -> Self {
        Self { db, config }
    }
    
    pub async fn run(&self) -> Result<()> {
        println!("Starting harness daemon...");
        println!("  Watch paths: {:?}", self.config.watch_paths);
        println!("  Fossil DB: {:?}", self.config.fossil_db);
        
        let mut handles = vec![];
        
        // Start file watcher
        if self.config.enable_file_watcher {
            let fw = FileWatcher::new(self.db.clone(), self.config.watch_paths.clone());
            let handle = tokio::spawn(async move {
                if let Err(e) = fw.run() {
                    eprintln!("File watcher error: {:?}", e);
                }
            });
            handles.push(handle);
            println!("  File watcher: started");
        }
        
        // Start event processor
        if self.config.enable_event_processor {
            let ep = EventProcessor::new(self.db.clone());
            let handle = tokio::spawn(async move {
                if let Err(e) = ep.run().await {
                    eprintln!("Event processor error: {:?}", e);
                }
            });
            handles.push(handle);
            println!("  Event processor: started");
        }
        
        println!("Daemon running. Press Ctrl+C to stop.");
        
        // Wait for shutdown signal
        signal::ctrl_c().await?;
        
        println!("Shutting down...");
        
        // Cancel all tasks
        for handle in handles {
            handle.abort();
        }
        
        Ok(())
    }
}
```

---

## Part 6: CI Pipeline Scripts

### 6.1 CI Configuration

```yaml
# .superfossil/ci.yaml

name: TRINITY Harness Pipeline

stages:
  - name: parse
    description: Parse all source files and update code graph
    
  - name: lint
    description: Run linters and ingest results
    
  - name: test-stale
    description: Run tests only for stale code
    
  - name: report
    description: Generate state report

jobs:
  parse:
    stage: parse
    script:
      - harness daemon --once  # Parse all files, process events, exit
      - harness check-layouts   # Fail on Rust/WGSL mismatch
    artifacts:
      - brain.db

  lint:
    stage: lint
    script:
      - cargo clippy --message-format=json > clippy.json 2>&1 || true
      - harness ingest-lint clippy.json
      - uv run ruff check engine/ --output-format=json > ruff.json || true
      - harness ingest-lint ruff.json --format=ruff
    artifacts:
      - clippy.json
      - ruff.json

  test-rust:
    stage: test-stale
    script:
      - |
        FILTER=$(harness query-tests --stale --lang=rust)
        if [ -n "$FILTER" ]; then
          cargo test $FILTER -- --format=json > rust-tests.json
          harness ingest-tests rust-tests.json
        else
          echo "No stale Rust tests"
        fi
    artifacts:
      - rust-tests.json

  test-python:
    stage: test-stale
    script:
      - |
        TESTS=$(harness query-tests --stale --lang=python)
        if [ -n "$TESTS" ]; then
          uv run pytest $TESTS --json-report > python-tests.json
          harness ingest-tests python-tests.json --format=pytest
        else
          echo "No stale Python tests"
        fi
    artifacts:
      - python-tests.json

  report:
    stage: report
    script:
      - harness status --json > state-report.json
      - harness metrics --format=prometheus > metrics.txt
      - |
        echo "## Code State Summary"
        harness status --summary
    artifacts:
      - state-report.json
      - metrics.txt
```

### 6.2 CI Helper Script

```bash
#!/bin/bash
# scripts/ci-harness.sh

set -e

COMMAND=$1
shift

case $COMMAND in
    init)
        # Initialize the harness for CI
        harness init --db=brain.db
        harness parse --all
        ;;
    
    check)
        # Pre-merge checks
        harness check-layouts || {
            echo "FAIL: Layout mismatches detected"
            exit 1
        }
        
        QUARANTINED=$(harness query --state=quarantined --count)
        if [ "$QUARANTINED" -gt 0 ]; then
            echo "WARN: $QUARANTINED quarantined nodes"
        fi
        ;;
    
    test-stale)
        # Run only stale tests
        LANG=${1:-all}
        
        TESTS=$(harness query-tests --stale --lang=$LANG)
        
        if [ -z "$TESTS" ]; then
            echo "No stale tests for $LANG"
            exit 0
        fi
        
        echo "Running stale tests: $TESTS"
        
        case $LANG in
            rust|all)
                cargo test $TESTS -- --format=json 2>&1 | tee rust-tests.json
                harness ingest-tests rust-tests.json
                ;;
            python)
                uv run pytest $TESTS --json-report
                harness ingest-tests .report.json --format=pytest
                ;;
        esac
        ;;
    
    report)
        echo "=== Code State Report ==="
        harness status --summary
        echo ""
        echo "=== Stale Nodes ==="
        harness query --state=stale_direct --limit=20
        echo ""
        echo "=== Layout Mismatches ==="
        harness check-layouts --report
        ;;
    
    *)
        echo "Usage: ci-harness.sh {init|check|test-stale|report}"
        exit 1
        ;;
esac
```

---

## Part 7: Event Types Reference

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    // === SOURCE CHANGE EVENTS ===
    /// Source code changed (generic)
    SourceChanged,
    /// Function/struct signature changed (public interface)
    SignatureChanged,
    /// Only implementation body changed (not signature)
    BodyChanged,
    /// Memory layout changed (struct alignment/size)
    LayoutChanged,
    
    // === DEPENDENCY EVENTS ===
    /// A direct dependency changed
    DependencyChanged,
    /// A transitive dependency changed (2+ hops)
    TransitiveDependencyChanged,
    
    // === TEST EVENTS ===
    /// Tests ran and passed
    TestsPassed,
    /// Tests ran and failed
    TestsFailed,
    /// Tests were skipped
    TestsSkipped,
    /// Tests had an error (not failure)
    TestsError,
    
    // === QA EVENTS ===
    /// Passed QA/adversarial review
    QaApproved,
    /// Flagged by QA review
    QaFlagged,
    
    // === LIFECYCLE EVENTS ===
    /// Put node in quarantine
    Quarantine,
    /// Released from quarantine
    Release,
    /// Marked as deprecated
    Deprecated,
    
    // === GRAPH EVENTS ===
    /// New node created
    NodeCreated,
    /// Node deleted
    NodeDeleted,
    /// Edge created
    EdgeCreated,
    /// Edge deleted
    EdgeDeleted,
    
    // === VCS EVENTS ===
    /// Superfossil commit completed
    Commit,
    /// Commit processing complete
    CommitComplete,
    
    // === LINT EVENTS ===
    /// Linter warning
    LintWarning,
    /// Linter error
    LintError,
    
    // === SYSTEM EVENTS ===
    /// Full reparse triggered
    FullReparse,
    /// Propagation completed
    PropagationComplete,
    /// Layout mismatch detected
    LayoutMismatchDetected,
    /// Generic notification
    Notification,
}

impl EventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SourceChanged => "source_changed",
            Self::SignatureChanged => "signature_changed",
            Self::BodyChanged => "body_changed",
            Self::LayoutChanged => "layout_changed",
            Self::DependencyChanged => "dependency_changed",
            Self::TransitiveDependencyChanged => "transitive_dependency_changed",
            Self::TestsPassed => "tests_passed",
            Self::TestsFailed => "tests_failed",
            Self::TestsSkipped => "tests_skipped",
            Self::TestsError => "tests_error",
            Self::QaApproved => "qa_approved",
            Self::QaFlagged => "qa_flagged",
            Self::Quarantine => "quarantine",
            Self::Release => "release",
            Self::Deprecated => "deprecated",
            Self::NodeCreated => "node_created",
            Self::NodeDeleted => "node_deleted",
            Self::EdgeCreated => "edge_created",
            Self::EdgeDeleted => "edge_deleted",
            Self::Commit => "commit",
            Self::CommitComplete => "commit_complete",
            Self::LintWarning => "lint_warning",
            Self::LintError => "lint_error",
            Self::FullReparse => "full_reparse",
            Self::PropagationComplete => "propagation_complete",
            Self::LayoutMismatchDetected => "layout_mismatch_detected",
            Self::Notification => "notification",
        }
    }
}
```

---

## Part 8: State Transition Diagram

```
                                  ┌─────────────┐
                                  │   UNKNOWN   │
                                  └──────┬──────┘
                                         │
                                         │ NodeCreated
                                         ▼
                                  ┌─────────────┐
                     ┌────────────│  UNTOUCHED  │────────────┐
                     │            └──────┬──────┘            │
                     │                   │                   │
              TestsPassed         SourceChanged         TestsFailed
                     │                   │                   │
                     ▼                   ▼                   ▼
              ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
              │   TESTED    │     │   STALE     │     │   TESTED    │
              │    GREEN    │     │   DIRECT    │     │    RED      │
              └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
                     │                   │                   │
         ┌───────────┼───────────────────┼───────────────────┤
         │           │                   │                   │
    QaApproved  SourceChanged      TestsPassed          TestsPassed
         │           │                   │                   │
         ▼           ▼                   ▼                   ▼
   ┌───────────┐                  ┌─────────────┐     ┌─────────────┐
   │    QA     │                  │   TESTED    │     │   TESTED    │
   │ APPROVED  │                  │    GREEN    │     │    GREEN    │
   └─────┬─────┘                  └─────────────┘     └─────────────┘
         │
    SourceChanged
         │
         ▼
   ┌─────────────┐
   │   STALE     │
   │   DIRECT    │
   └─────────────┘


   ═══════════════════════════════════════════════════════════════════

   PROPAGATION:

   Node A [GREEN] ──(SourceChanged)──▶ Node A [STALE_DIRECT]
                                              │
                                              │ DependencyChanged (to B)
                                              ▼
                                       Node B [GREEN] ──▶ Node B [STALE_TRANSITIVE]
                                                                │
                                              TransitiveDependencyChanged (to C)
                                                                ▼
                                                         Node C [GREEN] ──▶ Node C [STALE_DEEP]

   ═══════════════════════════════════════════════════════════════════

   QUARANTINE (special):

   ANY STATE ──(Quarantine)──▶ QUARANTINED ──(Release)──▶ LAST_KNOWN_GOOD
```

---

## Appendix A: Event Builder

```rust
pub struct CodeEventBuilder {
    event_type: EventType,
    node_id: Option<String>,
    payload: Option<Value>,
    idempotency_key: Option<String>,
    correlation_id: Option<String>,
    causation_id: Option<i64>,
}

impl CodeEventBuilder {
    pub fn new(event_type: EventType) -> Self {
        Self {
            event_type,
            node_id: None,
            payload: None,
            idempotency_key: None,
            correlation_id: None,
            causation_id: None,
        }
    }
    
    pub fn node_id(mut self, id: &str) -> Self {
        self.node_id = Some(id.to_string());
        self
    }
    
    pub fn payload(mut self, payload: Value) -> Self {
        self.payload = Some(payload);
        self
    }
    
    pub fn idempotency_key(mut self, key: &str) -> Self {
        self.idempotency_key = Some(key.to_string());
        self
    }
    
    pub fn correlation_id(mut self, id: &str) -> Self {
        self.correlation_id = Some(id.to_string());
        self
    }
    
    pub fn causation_id(mut self, id: i64) -> Self {
        self.causation_id = Some(id);
        self
    }
    
    pub fn build(self) -> StoredEvent {
        StoredEvent {
            sequence: 0, // Assigned by DB
            timestamp: chrono::Utc::now().to_rfc3339(),
            event_type: self.event_type,
            node_id: self.node_id,
            payload: self.payload,
            idempotency_key: self.idempotency_key,
            correlation_id: self.correlation_id,
            causation_id: self.causation_id,
        }
    }
}
```

---

## Appendix B: Configuration

```toml
# harness.toml

[daemon]
watch_paths = ["crates/", "engine/", "tests/"]
debounce_ms = 100
enable_file_watcher = true
enable_event_processor = true

[database]
path = "brain.db"
cache_size_mb = 64
wal_mode = true

[superfossil]
enabled = true
db_path = ".superfossil/repo.fossil"
pre_commit_checks = true

[ci]
auto_test_stale = true
fail_on_layout_mismatch = true
warn_on_quarantine = true

[notifications]
enable_pubsub = true
webhook_url = ""
```

---

**The workflow integration is the heartbeat of the system. Events flow in, state updates flow out, and the entire codebase stays synchronized.**
