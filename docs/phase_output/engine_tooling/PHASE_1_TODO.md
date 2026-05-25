# PHASE 1 TODO: Core Editor Infrastructure

## Overview

Phase 1 establishes the foundational infrastructure for all editor tooling. This phase must be completed before other phases can proceed, as all subsequent modules depend on undo, logging, and application shell systems.

---

## 1. Editor Application Shell

### 1.1 DockingManager Integration
- [ ] **T1.1.1**: Wire DockingManager.load_layout() to actual file persistence
  - Acceptance: Layout restored on editor restart
  - File: `engine/tooling/editor/app_shell.py`
  
- [ ] **T1.1.2**: Implement panel resize handles
  - Acceptance: Panels can be resized by dragging edges
  - Acceptance: Resize constraints respected (min_width, min_height)

- [ ] **T1.1.3**: Add floating panel support
  - Acceptance: Panels can be undocked to floating windows
  - Acceptance: Floating panels can be re-docked

### 1.2 Tab Management
- [ ] **T1.2.1**: Implement tab reordering via drag
  - Acceptance: Tabs can be dragged within TabGroup
  - Acceptance: Tabs can be dragged between TabGroups

- [ ] **T1.2.2**: Add dirty state indicators
  - Acceptance: Modified tabs show visual indicator (asterisk or dot)
  - Acceptance: Close prompt when closing dirty tab

### 1.3 Menu System
- [ ] **T1.3.1**: Implement keyboard shortcut display
  - Acceptance: Shortcuts shown aligned in menu items
  - Acceptance: Platform-appropriate modifier display (Cmd vs Ctrl)

- [ ] **T1.3.2**: Add recent files submenu
  - Acceptance: Last 10 files tracked and displayed
  - Acceptance: File existence validated on display

### 1.4 Status Bar
- [ ] **T1.4.1**: Implement progress indicator integration
  - Acceptance: Long operations show progress in status bar
  - Acceptance: Progress can be cancelled via status bar

---

## 2. Undo System

### 2.1 Foundation Integration
- [ ] **T2.1.1**: Complete Foundation Tracker subscription
  - Acceptance: Changes from Tracker automatically create undo entries
  - Acceptance: Tracker and UndoSystem stay synchronized
  - File: `engine/tooling/undo/undo_system.py`

- [ ] **T2.1.2**: Implement Mirror-based SetFieldCommand
  - Acceptance: Field changes use Mirror for get/set
  - Acceptance: Type validation via Mirror schema
  - File: `engine/tooling/undo/command_pattern.py`

### 2.2 Transaction System
- [ ] **T2.2.1**: Complete savepoint rollback
  - Acceptance: Partial rollback to named savepoint works
  - Acceptance: Commands after savepoint are undone
  - File: `engine/tooling/undo/transaction.py`

- [ ] **T2.2.2**: Add nested transaction visualization
  - Acceptance: Nesting level exposed for UI display
  - Acceptance: Active transactions queryable

### 2.3 History View
- [ ] **T2.3.1**: Wire HistoryView to UndoSystem
  - Acceptance: Branching history created when undoing and making changes
  - Acceptance: Navigate to any history node
  - File: `engine/tooling/undo/history_view.py`

- [ ] **T2.3.2**: Implement history pruning
  - Acceptance: Old branches can be deleted
  - Acceptance: Memory usage bounded

### 2.4 Dirty Tracking
- [ ] **T2.4.1**: Implement document state tracking
  - Acceptance: Each open document has dirty state
  - Acceptance: Save clears dirty state
  - File: `engine/tooling/undo/dirty_tracking.py`

- [ ] **T2.4.2**: Add unsaved changes dialog
  - Acceptance: Close prompts for unsaved documents
  - Acceptance: Batch save option for multiple dirty documents

---

## 3. Console System

### 3.1 CVar System
- [ ] **T3.1.1**: Implement CVar change callbacks
  - Acceptance: Callbacks fired on value change
  - Acceptance: Old and new values passed to callback
  - File: `engine/tooling/console/cvar_system.py`

- [ ] **T3.1.2**: Add CVar auto-completion
  - Acceptance: Tab completes CVar names
  - Acceptance: Prefix matching with multiple suggestions

- [ ] **T3.1.3**: Implement CVar persistence
  - Acceptance: ARCHIVE CVars saved to JSON
  - Acceptance: Values restored on startup

### 3.2 Command System
- [ ] **T3.2.1**: Add command history persistence
  - Acceptance: History saved between sessions
  - Acceptance: Configurable max history size
  - File: `engine/tooling/console/command_history.py`

- [ ] **T3.2.2**: Implement reverse search
  - Acceptance: Ctrl+R searches command history
  - Acceptance: Incremental search updates results

### 3.3 Console UI
- [ ] **T3.3.1**: Add console output filtering
  - Acceptance: Filter by output type (ERROR, WARNING, INFO)
  - Acceptance: Pattern-based filtering
  - File: `engine/tooling/console/console_ui.py`

- [ ] **T3.3.2**: Implement console log export
  - Acceptance: Export console log to file
  - Acceptance: Timestamp formatting in export

---

## 4. Logging System

### 4.1 Targets
- [ ] **T4.1.1**: Implement file rotation
  - Acceptance: Files rotate at max_size
  - Acceptance: Old files deleted at max_files
  - File: `engine/tooling/logging/log_targets.py`

- [ ] **T4.1.2**: Add TLS support for NetworkTarget
  - Acceptance: SSL/TLS connection option
  - Acceptance: Certificate validation configurable

- [ ] **T4.1.3**: Implement RingBuffer search
  - Acceptance: Search by pattern, level, category
  - Acceptance: Return matching entries for crash report

### 4.2 Filters
- [ ] **T4.2.1**: Test rate limiting with sliding windows
  - Acceptance: Burst allowed, then throttled
  - Acceptance: Per-category:level tracking
  - File: `engine/tooling/logging/log_filter.py`

- [ ] **T4.2.2**: Verify deduplication window
  - Acceptance: Duplicate messages suppressed
  - Acceptance: Count shown on final message

### 4.3 Structured Logging
- [ ] **T4.3.1**: Implement span export
  - Acceptance: Spans exportable to tracing backend
  - Acceptance: OpenTelemetry format compatibility
  - File: `engine/tooling/logging/structured_log.py`

- [ ] **T4.3.2**: Add context propagation
  - Acceptance: Trace IDs propagate across async calls
  - Acceptance: Baggage items preserved

---

## 5. Crash Reporting

### 5.1 Exception Capture
- [ ] **T5.1.1**: Verify exception chaining capture
  - Acceptance: __cause__ and __context__ captured
  - Acceptance: Full chain in crash report
  - File: `engine/tooling/crash/crash_reporter.py`

- [ ] **T5.1.2**: Add local variable capture
  - Acceptance: Local vars in each stack frame
  - Acceptance: Sensitive data filtering option

### 5.2 Upload System
- [ ] **T5.2.1**: Test retry with exponential backoff
  - Acceptance: Retries on 429/503
  - Acceptance: Backoff increases between retries
  - File: `engine/tooling/crash/crash_upload.py`

- [ ] **T5.2.2**: Implement upload queue persistence
  - Acceptance: Failed uploads queued to disk
  - Acceptance: Retry on next startup

### 5.3 Analytics
- [ ] **T5.3.1**: Verify pattern detection
  - Acceptance: Stack patterns detected across crashes
  - Acceptance: Version correlation identified
  - File: `engine/tooling/crash/crash_analytics.py`

- [ ] **T5.3.2**: Implement trend reporting
  - Acceptance: Hourly/daily crash trends
  - Acceptance: Regression detection

### 5.4 Symbol Server
- [ ] **T5.4.1**: Test symbol file loading
  - Acceptance: JSON and text formats loaded
  - Acceptance: Address resolution correct
  - File: `engine/tooling/crash/symbol_server.py`

- [ ] **T5.4.2**: Verify cache eviction
  - Acceptance: LRU eviction at max size
  - Acceptance: TTL expiration respected

---

## 6. Hot Reload

### 6.1 Module Watching
- [ ] **T6.1.1**: Test debouncing behavior
  - Acceptance: Rapid changes coalesced
  - Acceptance: Single event after debounce period
  - File: `engine/tooling/hotreload/module_watcher.py`

- [ ] **T6.1.2**: Verify exclude patterns
  - Acceptance: __pycache__ ignored
  - Acceptance: Custom patterns respected

### 6.2 State Preservation
- [ ] **T6.2.1**: Test all preservation strategies
  - Acceptance: SERIALIZER preserves Foundation objects
  - Acceptance: MIRROR preserves decorated classes
  - Acceptance: PICKLE handles Python objects
  - File: `engine/tooling/hotreload/state_preservation.py`

- [ ] **T6.2.2**: Verify transient field handling
  - Acceptance: Transient fields reset on reload
  - Acceptance: Non-transient fields preserved

### 6.3 Schema Validation
- [ ] **T6.3.1**: Test breaking change detection
  - Acceptance: Field removal detected
  - Acceptance: Type narrowing detected
  - File: `engine/tooling/hotreload/schema_hash.py`

- [ ] **T6.3.2**: Verify migration hints
  - Acceptance: Hints generated for breaking changes
  - Acceptance: Clear action recommendations

### 6.4 Dependency Tracking
- [ ] **T6.4.1**: Test cascade reload order
  - Acceptance: Dependencies reloaded first
  - Acceptance: Topological sort correct
  - File: `engine/tooling/hotreload/dependency_tracker.py`

- [ ] **T6.4.2**: Verify cycle detection
  - Acceptance: Circular imports detected
  - Acceptance: Error reported with cycle path

---

## Integration Tests

### I1. Full Undo Cycle
- [ ] **I1.1**: Test undo/redo with Foundation objects
  - Steps: Create object, modify fields, undo, verify original state, redo, verify modified state
  - Acceptance: State correctly restored at each step

### I2. Console to CVar
- [ ] **I2.1**: Test CVar modification via console
  - Steps: Set CVar via command, verify value changed, verify callback fired
  - Acceptance: Full round-trip works

### I3. Hot Reload with Undo
- [ ] **I3.1**: Test undo after hot reload
  - Steps: Make changes, reload module, undo changes
  - Acceptance: Undo works correctly after reload

### I4. Crash During Operation
- [ ] **I4.1**: Test crash report capture during long operation
  - Steps: Trigger crash during build, verify crash report complete
  - Acceptance: Operation context included in report

---

## Performance Targets

| Metric | Target | Test Method |
|--------|--------|-------------|
| Undo stack push | < 1ms | Benchmark 1000 operations |
| Log write (async) | < 10us | Benchmark 10000 messages |
| CVar lookup | < 100us | Benchmark 1000 lookups |
| Hot reload (small) | < 100ms | Benchmark single file reload |
| Crash report gen | < 50ms | Benchmark exception capture |

---

## Dependencies

### Required Before Phase 1
- Foundation Tracker implementation
- Foundation Mirror implementation
- Platform file watcher implementation

### Blocks Phase 2+
- All Phase 2-6 modules depend on:
  - Logging (error reporting)
  - Undo (reversible operations)
  - Console (debug commands)
  - Hot reload (development iteration)
