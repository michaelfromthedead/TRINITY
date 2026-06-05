# PHASE 1 ARCHITECTURE: AST Node System

## Overview

Phase 1 establishes the foundational AST node system that represents SDF scenes as traversable trees.

## Architectural Decisions

### AD-1.1: Frozen Dataclasses for All Nodes

**Decision**: Use `@dataclass(frozen=True)` for all AST nodes.

**Rationale**:
- Immutability prevents accidental state changes during traversal
- Enables hash-based caching and memoization
- Thread-safe by construction
- Aligns with functional programming patterns common in compilers

**Consequences**:
- Node modification requires creating new instances
- Children must be tuples, not lists

### AD-1.2: Base ExprNode Protocol

**Decision**: All nodes inherit from ExprNode providing walk(), children(), pretty(), label() methods.

**Rationale**:
- Uniform traversal interface for code generator
- Pretty-printing enables debugging and visualization
- Label provides concise node identification

**Implementation**:
```python
class ExprNode:
    def walk(self): ...      # Generator yielding self and all descendants
    def children(self): ...  # Direct child nodes
    def pretty(self): ...    # Multi-line formatted string
    def label(self): ...     # Single-line identifier
```

### AD-1.3: Node Category Hierarchy

**Decision**: Organize nodes into distinct categories:
- Primitive nodes (FloatNode, Vec3Node, PositionNode)
- Domain operation nodes (RepeatNode, MirrorNode, KifsNode, TwistNode, BendNode, StretchNode)
- SDF primitive nodes (SphereNode, BoxNode, TorusNode, etc.)
- CSG combine nodes (UnionNode, IntersectionNode, SubtractionNode)
- Container nodes (MaterialNode, SceneGraph)

**Rationale**:
- Clear semantic grouping aids code generation dispatch
- Each category has distinct code emission patterns
- Extensibility within categories

### AD-1.4: SceneGraph as Root Container

**Decision**: SceneGraph holds primitives tuple, pipeline tuple (domain ops), and materials tuple.

**Rationale**:
- Single entry point for code generation
- Pipeline represents ordered sequence of domain transformations
- Materials collected for switch-based material lookup

**Structure**:
```
SceneGraph
  ├── pipeline: (DomainOpNode, ...)     # Applied in order
  ├── primitives: (SdfPrimitiveNode, ...) # Rendered shapes
  └── materials: (MaterialNode, ...)     # Material definitions
```

## Dependencies

- Python dataclasses module
- No external dependencies (pure Python)

## Interfaces

### Input
- Node constructors accept typed parameters

### Output
- Nodes expose traversal methods for code generation
- Pretty-print for debugging
