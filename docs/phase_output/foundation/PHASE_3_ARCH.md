# PHASE 3 ARCHITECTURE: Layer 3 Interactive Systems

## Overview

Phase 3 implements interactive debugging and security systems. These modules provide object visualization, live code execution, and capability-based security enforcement. They depend on Layers 0-2.

## Components

### inspector.py (282 lines)

Object visualization with pluggable views.

**Classes**:
- `Inspector`: Singleton managing object inspection
  - `inspect(obj)`: Render object with appropriate view
  - `register_view(type, view)`: Add custom view for type
  - `default_view(obj)`: Fallback visualization

- `View`: Protocol for custom visualizers
  - `render(obj, context) -> str`: Generate visualization
  - `supports(obj) -> bool`: Check if view handles object

- `UIContext`: Protocol for rendering context
  - `indent_level`: Current nesting depth
  - `max_depth`: Maximum recursion depth
  - `visited`: Set of visited object IDs (cycle prevention)

**Default Views**:
- Primitive view (int, str, bool, None)
- Collection view (list, dict, set)
- Object view (general objects via mirror)
- Exception view (tracebacks)

**Design**: Views registered by type priority. Most specific type match wins.

### inspector_views.py (440 lines)

Specialized views for debugging.

**Classes**:
- `HistoryView`: Visualizes object change history
  - Shows timeline of modifications
  - Integrates with ChangeTracker
  - Displays old/new values

- `CausalityView`: Visualizes event causal chains
  - Tree rendering of event ancestry
  - Integrates with EventLog
  - Shows immediate_parent, root_cause, depth

- `ProvenanceView`: Visualizes value derivation
  - DAG rendering of computation inputs
  - Integrates with provenance module
  - Shows transitive dependencies

**Rendering Format**: ASCII art trees suitable for terminal output.

### shell.py (203 lines)

Interactive Python REPL with namespace.

**Classes**:
- `Shell`: Interactive execution environment
  - `__init__(namespace)`: Initialize with globals
  - `execute(code)`: Run code string
  - `add_to_namespace(name, obj)`: Inject binding
  - `get_namespace()`: Return current namespace

**Execution Model**:
- Expression: `eval()` returns value
- Statement: `exec()` modifies namespace
- Detection: Try eval first, fall back to exec

**Namespace Setup**:
- Foundation modules pre-imported
- Helper functions for common operations
- History via readline integration

**Error Handling**:
- Catches all exceptions
- Formats traceback for display
- Does not crash shell on errors

### secure_shell.py (249 lines)

Capability-enforced code execution.

**Classes**:
- `SecureShell`: Shell with capability checks
  - `__init__(namespace, capabilities)`: Initialize with allowed caps
  - `execute(code)`: Run with capability enforcement
  - `with_restricted(caps)`: Create more restricted shell

**Enforcement Points**:
- FILESYSTEM: file operations (open, pathlib)
- NETWORK: socket, urllib, requests
- EXECUTE: subprocess, os.system
- SPAWN: multiprocessing, threading

**Implementation**:
1. Parse code to AST
2. Walk AST checking for restricted operations
3. Raise CapabilityError if operation not allowed
4. Execute only if all checks pass

**Sandboxing Limitations**:
- Not a security sandbox (no syscall filtering)
- Defense in depth against accidental misuse
- User-provided code assumed semi-trusted

### capabilities.py (363 lines)

Capability-based security primitives.

**Enums**:
- `Capability`: Flag enum
  - READ
  - WRITE
  - CREATE
  - DELETE
  - EXECUTE
  - SPAWN
  - NETWORK
  - FILESYSTEM

**Classes**:
- `CapabilitySet`: Immutable set of capabilities
  - `grant(cap) -> CapabilitySet`: Returns new set with cap added
  - `revoke(cap) -> CapabilitySet`: Returns new set with cap removed
  - `has(cap) -> bool`: Check capability presence
  - `__contains__(cap)`: Support `cap in caps` syntax
  - `__and__`, `__or__`: Set operations returning new sets

- `SecureContext`: Context manager for capability scope
  - `__init__(capabilities)`: Set active capabilities
  - `__enter__`: Push to capability stack
  - `__exit__`: Pop from capability stack
  - `restrict(caps)`: Create nested context with fewer caps

**Decorator**:
- `@require_capability(*caps)`: Enforce caps before function call
  - Raises CapabilityError if missing
  - Documents required capabilities in function metadata

**Context Variable**:
- `_current_capabilities`: Thread-safe capability stack
- Default: Empty set (no capabilities)
- Functions check current context for caps

## Data Flow

```
Object Inspection:
  obj -> Inspector.inspect() -> View lookup (by type) -> View.render() -> String

History Visualization:
  obj -> HistoryView -> ChangeTracker query -> Timeline rendering

Causal Visualization:
  event -> CausalityView -> EventLog query -> Tree rendering

Code Execution:
  code -> Shell.execute() -> eval/exec -> Result

Secure Execution:
  code -> SecureShell.execute() -> AST check -> Capability verify -> exec
              |
              v (on failure)
         CapabilityError

Capability Context:
  with SecureContext(caps):
      # _current_capabilities = caps
      @require_capability(CAP)
      def operation():
          # Checks _current_capabilities.has(CAP)
```

## Dependencies

Layer 3 depends on:
- Layer 2: tracker (HistoryView), query (potential integration)
- Layer 1: registry (type lookup)
- Layer 0: mirror (object introspection), eventlog (CausalityView), provenance (ProvenanceView)
- stdlib: ast, contextvars, enum, typing

## Security Model

### Defense in Depth

1. **Capability Declaration**: Functions declare required caps via decorator
2. **Context Scoping**: Code runs with limited caps via SecureContext
3. **AST Checking**: Restricted operations detected before execution
4. **Audit Trail**: Operations recorded via EventLog

### Threat Model

- **In Scope**: Accidental misuse, AI-generated code, user scripts
- **Out of Scope**: Malicious bytecode, native extensions, kernel exploits

### Capability Combinations

| Operation | Required Capabilities |
|-----------|----------------------|
| Read file | FILESYSTEM, READ |
| Write file | FILESYSTEM, WRITE |
| HTTP request | NETWORK, READ/WRITE |
| Spawn process | EXECUTE, SPAWN |
| Create entity | CREATE |
| Delete entity | DELETE |

## Testing Strategy

### Inspector Tests
- Default views render correctly for each type
- Custom views registered and used
- Cycle detection prevents infinite recursion
- Max depth respected
- View priority by type specificity

### Inspector Views Tests
- HistoryView shows correct timeline
- CausalityView renders correct tree structure
- ProvenanceView shows correct DAG
- All views handle empty/missing data

### Shell Tests
- Expression evaluation returns value
- Statement execution modifies namespace
- Namespace injection works
- Errors caught and formatted
- History maintained

### SecureShell Tests
- Allowed operations execute
- Disallowed operations raise CapabilityError
- AST check covers all restricted patterns
- Nested restriction works

### Capabilities Tests
- CapabilitySet immutability
- grant/revoke return new sets
- has() checks correctly
- SecureContext pushes/pops correctly
- @require_capability enforces
- Nested contexts restrict correctly
