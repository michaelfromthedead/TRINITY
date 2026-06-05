# PHASE 2 ARCHITECTURE: AST Builder and DSL Parsing

## Overview

Phase 2 implements the multi-dispatch walker that transforms various input representations into the AST node system.

## Architectural Decisions

### AD-2.1: Multi-Dispatch Walker Pattern

**Decision**: AstBuilder.walk() uses dispatch tables to handle dicts, lists, ExprNodes, callables, and DSL objects uniformly.

**Rationale**:
- Single entry point simplifies caller code
- Type-based dispatch is explicit and extensible
- New input formats require only dispatch table updates

**Dispatch Order**:
1. ExprNode - pass through unchanged
2. dict - lookup "type" key in dispatch tables
3. list - recursively walk and collect
4. callable - attempt lambda disassembly
5. DSL object - extract to dict then walk

### AD-2.2: Dispatch Table Organization

**Decision**: Three separate dispatch tables for different node categories.

**Tables**:
- `_COMPOSITION_DISPATCH`: Domain operations (repeat, mirror, kifs, twist, bend, stretch)
- `_PRIMITIVE_DISPATCH`: SDF functions (sdSphere, sdBox, sdTorus, etc.)
- `_MARKER_DISPATCH`: Type string markers for explicit node construction

**Rationale**:
- Separation prevents naming collisions
- Clear mapping from DSL vocabulary to node constructors
- Priority ordering: primitives checked before generic markers

### AD-2.3: Lambda Disassembly via Python AST

**Decision**: Parse Python lambdas using `ast` module to extract SDF expressions.

**Rationale**:
- Enables natural Python syntax: `lambda p: sdSphere(domain_twist(p, 2.0), 1.0)`
- Leverages Python's own parser for correctness
- Avoids custom expression parser

**Requirements**:
- Source code must be available at runtime (no .pyc-only)
- Lambda must have Call as body (not arbitrary expressions)

**Implementation Flow**:
```
lambda fn
  -> inspect.getsource(fn)
  -> ast.parse(source)
  -> find Lambda node with Call body
  -> _build_ast_from_call() recursively
```

### AD-2.4: Recursive AST Building from Call Nodes

**Decision**: `_build_ast_from_call()` recursively processes Python AST Call nodes.

**Rationale**:
- Nested function calls map naturally to nested AST nodes
- Arguments extracted and converted to appropriate node types
- Function name maps via dispatch tables to node constructor

**Handling**:
- Numeric literals -> FloatNode
- Name 'p' -> PositionNode
- Calls -> recursive dispatch through tables

## Dependencies

- Python `ast` module for lambda parsing
- Python `inspect` module for source retrieval
- Python `textwrap` for source normalization

## Interfaces

### Input
- Dict structures with "type" keys
- Python lambdas encoding SDF expressions
- Lists of the above
- Existing ExprNode instances (pass-through)
- DSL objects with to_dict() or similar

### Output
- ExprNode tree ready for code generation

## Error Handling

- `ValueError` for uninspectable lambdas (compiled code)
- `SyntaxError` passthrough for malformed source
- `None` return for lambdas without Call body
