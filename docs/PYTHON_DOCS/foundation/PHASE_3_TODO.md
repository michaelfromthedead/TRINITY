# PHASE 3 TODO: Layer 3 Interactive Systems

## T-FND-3.1: Inspector Core

### Description
Implement object visualization with pluggable view system.

### Tasks
- [ ] Implement Inspector singleton class
- [ ] Implement inspect(obj) method
- [ ] Implement register_view(type, view) method
- [ ] Implement default_view(obj) fallback
- [ ] Define View protocol with render() and supports()
- [ ] Define UIContext protocol for rendering state
- [ ] Implement view lookup by type priority
- [ ] Handle cycle detection via visited set

### Acceptance Criteria
- inspect() renders objects to string
- Custom views override defaults for specific types
- Most specific type match wins priority
- Cycles detected and handled (ellipsis or reference)
- Max depth prevents infinite recursion
- UIContext tracks nesting correctly

---

## T-FND-3.2: Default Views

### Description
Implement built-in views for common types.

### Tasks
- [ ] Implement PrimitiveView for int, str, bool, None
- [ ] Implement CollectionView for list, dict, set, tuple
- [ ] Implement ObjectView using mirror for general objects
- [ ] Implement ExceptionView for tracebacks
- [ ] Register all default views on module load
- [ ] Ensure default views handle nested objects

### Acceptance Criteria
- Primitives render as their repr
- Collections render with elements
- Objects render attributes via mirror
- Exceptions render full traceback
- Nested structures render recursively
- Empty collections render correctly

---

## T-FND-3.3: History View

### Description
Implement visualization of object change history.

### Tasks
- [ ] Implement HistoryView class
- [ ] Query ChangeTracker for object history
- [ ] Render timeline of modifications
- [ ] Show old value -> new value for each change
- [ ] Include timestamps
- [ ] Handle objects with no history
- [ ] Limit history depth (configurable)

### Acceptance Criteria
- History renders as timeline
- Each change shows field, old value, new value
- Timestamps formatted readably
- Objects with no history show "No history"
- Recent changes appear first
- Long history truncated with "..."

---

## T-FND-3.4: Causality View

### Description
Implement visualization of event causal chains.

### Tasks
- [ ] Implement CausalityView class
- [ ] Query EventLog for event chain
- [ ] Render tree structure showing ancestry
- [ ] Show immediate_parent at each level
- [ ] Show root_cause at top
- [ ] Display event depth
- [ ] Handle circular references in event chains

### Acceptance Criteria
- Causal chain renders as tree
- Root cause at top of tree
- Each event shows parent relationship
- Depth numbers accurate
- ASCII art tree readable
- Handles events with no chain

---

## T-FND-3.5: Provenance View

### Description
Implement visualization of value derivation DAG.

### Tasks
- [ ] Implement ProvenanceView class
- [ ] Query provenance module for derivation_tree
- [ ] Render DAG structure showing inputs
- [ ] Show transitive dependencies
- [ ] Handle cycles in derivation graph
- [ ] Display source values at leaves
- [ ] Indicate computed vs source nodes

### Acceptance Criteria
- Derivation renders as DAG
- All inputs shown for computed values
- Transitive dependencies included
- Cycles shown with reference markers
- Leaf nodes show source values
- Computed nodes show computation

---

## T-FND-3.6: Shell Core

### Description
Implement interactive Python REPL with namespace.

### Tasks
- [ ] Implement Shell class
- [ ] Implement __init__(namespace) with default namespace
- [ ] Implement execute(code) method
- [ ] Implement add_to_namespace(name, obj) method
- [ ] Implement get_namespace() method
- [ ] Detect expression vs statement
- [ ] Use eval() for expressions, exec() for statements
- [ ] Pre-import foundation modules

### Acceptance Criteria
- Expressions return evaluated value
- Statements modify namespace
- Namespace persists across execute() calls
- Foundation modules available
- add_to_namespace injects bindings
- get_namespace returns current state

---

## T-FND-3.7: Shell Error Handling

### Description
Implement robust error handling for shell execution.

### Tasks
- [ ] Catch all exceptions during execute()
- [ ] Format traceback for display
- [ ] Return error info without crashing shell
- [ ] Handle SyntaxError specially
- [ ] Handle NameError with suggestions
- [ ] Preserve exception for programmatic access

### Acceptance Criteria
- Exceptions don't crash shell
- Traceback formatted readably
- SyntaxError shows line and column
- NameError suggests similar names
- execute() returns error info dict
- Original exception accessible

---

## T-FND-3.8: Capability Enum and Set

### Description
Implement capability primitives.

### Tasks
- [ ] Define Capability flag enum
- [ ] Include: READ, WRITE, CREATE, DELETE, EXECUTE, SPAWN, NETWORK, FILESYSTEM
- [ ] Implement CapabilitySet immutable class
- [ ] Implement grant(cap) returning new set
- [ ] Implement revoke(cap) returning new set
- [ ] Implement has(cap) method
- [ ] Implement __contains__ for `in` syntax
- [ ] Implement __and__, __or__ for set operations

### Acceptance Criteria
- All capabilities defined
- CapabilitySet is immutable
- grant returns new set with cap added
- revoke returns new set with cap removed
- has checks capability presence
- Set operations return new CapabilitySet

---

## T-FND-3.9: Secure Context

### Description
Implement capability context manager.

### Tasks
- [ ] Implement SecureContext class
- [ ] Implement __init__(capabilities)
- [ ] Implement __enter__ pushing to stack
- [ ] Implement __exit__ popping from stack
- [ ] Implement restrict(caps) for nested restriction
- [ ] Use _current_capabilities context variable
- [ ] Default capabilities: empty set

### Acceptance Criteria
- Context manager syntax works
- Capabilities active during context
- Exit restores previous capabilities
- restrict() creates more limited context
- Nested contexts stack correctly
- Thread-safe via contextvars

---

## T-FND-3.10: Capability Decorator

### Description
Implement @require_capability decorator.

### Tasks
- [ ] Implement require_capability(*caps) decorator
- [ ] Check _current_capabilities has all required caps
- [ ] Raise CapabilityError if missing
- [ ] Include required caps in error message
- [ ] Preserve function metadata
- [ ] Document caps in function.__required_capabilities__

### Acceptance Criteria
- Decorator enforces capability check
- CapabilityError raised for missing caps
- Error message lists missing capabilities
- Function metadata preserved
- __required_capabilities__ attribute set
- Works with async functions

---

## T-FND-3.11: Secure Shell

### Description
Implement capability-enforced code execution.

### Tasks
- [ ] Implement SecureShell class extending Shell
- [ ] Implement __init__(namespace, capabilities)
- [ ] Implement execute(code) with capability checks
- [ ] Parse code to AST before execution
- [ ] Walk AST checking for restricted operations
- [ ] Map operations to required capabilities
- [ ] Implement with_restricted(caps) method

### Acceptance Criteria
- Allowed operations execute normally
- Disallowed operations raise CapabilityError
- AST check happens before execution
- All restricted patterns detected:
  - FILESYSTEM: open, pathlib
  - NETWORK: socket, urllib
  - EXECUTE: subprocess, os.system
  - SPAWN: multiprocessing, threading
- with_restricted creates more limited shell

---

## T-FND-3.12: Capability Operation Mapping

### Description
Define which capabilities are required for which operations.

### Tasks
- [ ] Define FILESYSTEM_OPERATIONS list (open, Path, etc.)
- [ ] Define NETWORK_OPERATIONS list (socket, urllib, etc.)
- [ ] Define EXECUTE_OPERATIONS list (subprocess, os.system, etc.)
- [ ] Define SPAWN_OPERATIONS list (Process, Thread, etc.)
- [ ] Document each mapping with rationale
- [ ] Support combined capability requirements

### Acceptance Criteria
- All dangerous operations mapped
- Combined requirements supported (e.g., FILESYSTEM + WRITE)
- Mapping documented in code
- Easy to extend for new operations
- No false positives (legitimate code allowed)
