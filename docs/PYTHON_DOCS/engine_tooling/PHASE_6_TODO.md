# PHASE 6 TODO: Localization and Polish

## Overview

Phase 6 completes the tooling subsystem with localization support and polishes existing systems for production readiness.

---

## 1. Localization System

### 1.1 Text Extraction
- [ ] **T1.1.1**: Test Python pattern extraction
  - Acceptance: _(), localize(), tr() patterns found
  - Acceptance: ngettext() plurals detected
  - File: `engine/tooling/localization/text_extraction.py`

- [ ] **T1.1.2**: Test C++ pattern extraction
  - Acceptance: TR() patterns found
  - Acceptance: LOCTEXT() with context found

- [ ] **T1.1.3**: Test JavaScript pattern extraction
  - Acceptance: t() and i18n() patterns found

- [ ] **T1.1.4**: Test JSON asset extraction
  - Acceptance: text, label, title keys found
  - Acceptance: Nested objects traversed

- [ ] **T1.1.5**: Implement YAML extraction
  - Acceptance: YAML files parsed
  - Acceptance: Same keys as JSON detected
  - Note: Currently not implemented

### 1.2 String Tables
- [ ] **T1.2.1**: Test string entry management
  - Acceptance: Add/update/delete entries
  - Acceptance: Translations per language
  - File: `engine/tooling/localization/string_table.py`

- [ ] **T1.2.2**: Test plural forms
  - Acceptance: English ONE/OTHER works
  - Acceptance: Slavic rules work
  - Acceptance: Arabic rules work

- [ ] **T1.2.3**: Test category organization
  - Acceptance: Strings grouped by category
  - Acceptance: Category queries work

- [ ] **T1.2.4**: Test search functionality
  - Acceptance: Search by key
  - Acceptance: Search by text content

### 1.3 Translation Memory
- [ ] **T1.3.1**: Test exact matching
  - Acceptance: Identical source returns exact match
  - Acceptance: Quality ranking works
  - File: `engine/tooling/localization/translation_memory.py`

- [ ] **T1.3.2**: Test fuzzy matching
  - Acceptance: Similar text returns matches
  - Acceptance: Similarity score correct
  - Acceptance: Threshold filtering works

- [ ] **T1.3.3**: Test context boost
  - Acceptance: Same context increases score
  - Acceptance: +10% boost applied

- [ ] **T1.3.4**: Test usage tracking
  - Acceptance: Usage count incremented
  - Acceptance: Frequently used ranked higher

### 1.4 Workflow
- [ ] **T1.4.1**: Test extraction step
  - Acceptance: New strings detected
  - Acceptance: Removed strings flagged
  - File: `engine/tooling/localization/loc_workflow.py`

- [ ] **T1.4.2**: Test translation tasks
  - Acceptance: Tasks created per string
  - Acceptance: Assignment works
  - Acceptance: Completion updates state

- [ ] **T1.4.3**: Test validators
  - Acceptance: Empty translation error
  - Acceptance: Placeholder mismatch error
  - Acceptance: Untranslated warning
  - Acceptance: Length warning

- [ ] **T1.4.4**: Test external export
  - Acceptance: Format suitable for translators
  - Acceptance: Import updates strings

### 1.5 Dashboard
- [ ] **T1.5.1**: Test progress calculation
  - Acceptance: Completion percentage correct
  - Acceptance: Word count accurate
  - File: `engine/tooling/localization/loc_dashboard.py`

- [ ] **T1.5.2**: Test missing strings
  - Acceptance: Missing per language
  - Acceptance: Priority sorting works

- [ ] **T1.5.3**: Test report export
  - Acceptance: Text report readable
  - Acceptance: JSON report valid
  - Acceptance: CSV importable

### 1.6 Preview
- [ ] **T1.6.1**: Test preview modes
  - Acceptance: NORMAL shows translations
  - Acceptance: PSEUDO_LOC transforms text
  - Acceptance: KEYS_ONLY shows keys
  - Acceptance: MISSING_ONLY highlights missing
  - Acceptance: LONG_TEXT doubles length
  - File: `engine/tooling/localization/loc_preview.py`

- [ ] **T1.6.2**: Implement accent map for pseudo-loc
  - Acceptance: Characters replaced with accented versions
  - Note: Currently identity map

- [ ] **T1.6.3**: Test language switcher
  - Acceptance: Available languages listed
  - Acceptance: Cycle works
  - Acceptance: Preview updates

### 1.7 Missing Features
- [ ] **T1.7.1**: Add TMX import/export
  - Acceptance: TMX files importable
  - Acceptance: TMX files exportable
  - Note: Industry standard format

- [ ] **T1.7.2**: Add XLIFF support
  - Acceptance: XLIFF files importable
  - Acceptance: XLIFF files exportable
  - Note: Industry standard format

---

## 2. Cross-Module Polish

### 2.1 Error Handling
- [ ] **T2.1.1**: Audit exception handling in editor
  - Acceptance: No unhandled exceptions crash editor
  - Acceptance: User-friendly error messages
  - Files: `engine/tooling/editor/*.py`

- [ ] **T2.1.2**: Audit exception handling in build
  - Acceptance: Build failures reported clearly
  - Acceptance: Partial results saved
  - Files: `engine/tooling/build/*.py`

- [ ] **T2.1.3**: Audit exception handling in profiler
  - Acceptance: Profiler doesn't affect execution
  - Acceptance: Errors logged, not thrown
  - Files: `engine/tooling/profiling/*.py`

### 2.2 Logging Consistency
- [ ] **T2.2.1**: Verify all modules use LogSystem
  - Acceptance: No direct print() calls
  - Acceptance: Appropriate log levels

- [ ] **T2.2.2**: Add structured logging where missing
  - Acceptance: Key-value pairs for important events
  - Acceptance: Trace IDs for async operations

### 2.3 Thread Safety Audit
- [ ] **T2.3.1**: Verify singleton thread safety
  - Acceptance: All singletons use RLock
  - Acceptance: No race conditions in tests

- [ ] **T2.3.2**: Verify collection thread safety
  - Acceptance: Concurrent access handled
  - Acceptance: Iteration during modification safe

### 2.4 Memory Management
- [ ] **T2.4.1**: Verify weakref usage
  - Acceptance: Object trackers use weakrefs
  - Acceptance: No memory leaks in long sessions

- [ ] **T2.4.2**: Verify cleanup methods
  - Acceptance: reset_instance() on all singletons
  - Acceptance: clear() methods work

---

## 3. Integration Completeness

### 3.1 Foundation Integration
- [ ] **T3.1.1**: Verify Tracker integration
  - Acceptance: All undo operations use Tracker
  - Acceptance: Dirty state synchronized

- [ ] **T3.1.2**: Verify Mirror integration
  - Acceptance: Property editing uses Mirror
  - Acceptance: Serialization uses Mirror

### 3.2 Platform Integration
- [ ] **T3.2.1**: Verify file watcher integration
  - Acceptance: Hot reload uses platform watcher
  - Acceptance: Asset browser updates on changes

### 3.3 Renderer Integration Points
- [ ] **T3.3.1**: Document preview renderer interface
  - Acceptance: Material preview interface documented
  - Acceptance: Debug draw interface documented

- [ ] **T3.3.2**: Document GPU profiler interface
  - Acceptance: Timestamp query interface documented
  - Acceptance: VRAM tracking interface documented

---

## 4. Documentation

### 4.1 API Documentation
- [ ] **T4.1.1**: Verify all public classes have docstrings
  - Acceptance: Module-level docstrings
  - Acceptance: Class-level docstrings
  - Acceptance: Method-level docstrings

- [ ] **T4.1.2**: Add usage examples
  - Acceptance: Example in module docstring
  - Acceptance: Common patterns documented

### 4.2 Architecture Documentation
- [ ] **T4.2.1**: Create component diagram
  - Acceptance: All 20 modules shown
  - Acceptance: Dependencies indicated

- [ ] **T4.2.2**: Create data flow diagrams
  - Acceptance: Build pipeline flow
  - Acceptance: Editor command flow
  - Acceptance: Profiler data flow

---

## 5. Test Coverage

### 5.1 Coverage Goals
- [ ] **T5.1.1**: Achieve 80% coverage on core modules
  - Priority: undo, console, logging, crash
  - Acceptance: 80%+ line coverage

- [ ] **T5.1.2**: Achieve 100% coverage on critical paths
  - Priority: Build cache invalidation
  - Priority: Undo/redo consistency
  - Priority: Serialization round-trips

### 5.2 Integration Tests
- [ ] **T5.2.1**: Full editor workflow test
  - Steps: Create scene, edit, save, load
  - Acceptance: All state preserved

- [ ] **T5.2.2**: Full build workflow test
  - Steps: Import, cook, package, extract
  - Acceptance: Runnable package

- [ ] **T5.2.3**: Full localization workflow test
  - Steps: Extract, translate, import, preview
  - Acceptance: Translations display correctly

---

## 6. Performance Optimization

### 6.1 Profiling
- [ ] **T6.1.1**: Profile editor startup
  - Target: < 2 seconds
  - Acceptance: Bottlenecks identified and addressed

- [ ] **T6.1.2**: Profile large scene operations
  - Target: Smooth interaction with 10K objects
  - Acceptance: No frame drops during selection

### 6.2 Optimization
- [ ] **T6.2.1**: Optimize asset search
  - Target: < 100ms for 100K assets
  - Acceptance: Index-based lookup

- [ ] **T6.2.2**: Optimize undo stack
  - Target: < 1ms per operation
  - Acceptance: Command merging effective

---

## Integration Tests

### I1. Full Localization Pipeline
- [ ] **I1.1**: Extract, translate, validate, preview
  - Steps: Run extraction, add translations, validate, preview
  - Acceptance: Complete workflow functional

### I2. Cross-Module Integration
- [ ] **I2.1**: Build with localization
  - Steps: Build project with multiple languages
  - Acceptance: All languages packaged

### I3. Editor Session
- [ ] **I3.1**: Full editing session
  - Steps: Create content, edit, undo/redo, save, reload
  - Acceptance: All state preserved correctly

---

## Performance Targets

| Metric | Target | Test Method |
|--------|--------|-------------|
| String table load | < 100ms (10K strings) | Benchmark |
| TM fuzzy search | < 50ms | Benchmark |
| Pseudo-loc transform | < 1ms per string | Benchmark |
| Dashboard refresh | < 200ms | Benchmark |

---

## Dependencies

### Required Before Phase 6
- Phase 1-5 complete
- All core systems functional

### External
- Translation tools for workflow testing
- Native speakers for localization QA

---

## Completion Criteria

Phase 6 is complete when:
1. All localization features implemented and tested
2. 80% test coverage achieved on core modules
3. All documentation complete
4. Performance targets met
5. No critical bugs in production scenarios
