# PHASE 4 ARCHITECTURE: Scripting and Logic Systems

## Phase Overview

Phase 4 implements the visual scripting system (FlowForge), enabling designers to create gameplay logic without traditional programming.

## Components

### 1. Visual Scripting System (engine/tooling/visual_scripting/)

**Purpose**: Node-based game logic authoring with bytecode execution

**Architecture**:
```
VisualScripting ("FlowForge")
    |
    +-- DataTypes
    |       +-- BlueprintType (ABC)
    |               +-- BoolType, IntType, FloatType, StringType
    |               +-- Vector2Type, Vector3Type
    |               +-- RotatorType, TransformType
    |               +-- ObjectType, ActorType, ComponentType
    |               +-- ArrayType, MapType, SetType
    |               +-- ExecutionType, WildcardType
    |       +-- TypeColor (wire colors)
    |       +-- Conversion rules matrix
    |
    +-- NodeTypes
    |       +-- Node (ABC)
    |       +-- NodePin (direction, kind, type)
    |       +-- NodeMetadata (display, category, color, flags)
    |       +-- NodeCategory (18 categories)
    |       +-- Node Implementations:
    |               +-- Events: BeginPlay, Tick, InputAction, CustomEvent
    |               +-- Flow: Branch, Sequence, ForLoop, WhileLoop, ForEach
    |               +-- Flow: Switch, Gate, DoOnce, FlipFlop
    |               +-- Functions: PureFunction, CallFunction
    |               +-- Variables: GetVariable, SetVariable
    |               +-- Macros: Macro, MacroInput, MacroOutput
    |               +-- Literals: Bool, Int, Float, String, Vector
    |               +-- Utility: PrintString, Delay
    |       +-- NodeRegistry (search, factory)
    |
    +-- GraphEditor
    |       +-- BlueprintGraph (nodes, connections)
    |       +-- Connection (source/target pins)
    |       +-- ViewState (zoom, pan, transforms)
    |       +-- UndoStack (max 100 actions)
    |       +-- ClipboardData (copy/paste)
    |       +-- Selection modes (REPLACE, ADD, REMOVE, TOGGLE)
    |       +-- Box selection, hit testing
    |       +-- Minimap navigation
    |
    +-- BlueprintCompiler
    |       +-- OpCode (35 bytecode operations)
    |       +-- BytecodeInstruction (opcode + operand)
    |       +-- ConstantPool (strings, numbers, IDs)
    |       +-- CompiledFunction (instructions, locals)
    |       +-- CompiledBlueprint (functions, source map)
    |       +-- Optimization levels:
    |               +-- NONE: No optimization
    |               +-- BASIC: NOP removal
    |               +-- STANDARD: Constant folding
    |               +-- AGGRESSIVE: Dead code elimination
    |       +-- BytecodeSerializer (binary format "BPBC")
    |
    +-- BlueprintRuntime
    |       +-- BlueprintVM (stack-based execution)
    |       +-- LatentAction (delays, timers)
    |       +-- EventDispatcher (subscription, queuing)
    |       +-- Safety limits (max_iterations, max_stack)
    |       +-- Frame statistics
    |
    +-- ExecutionContext
    |       +-- VariableScope (LOCAL, INSTANCE, CLASS, GLOBAL)
    |       +-- Variable (name, type, value, scope)
    |       +-- StackFrame (function, locals, return value)
    |       +-- ExecutionState (IDLE, RUNNING, PAUSED, WAITING)
    |       +-- ExecutionContextPool (object reuse)
    |
    +-- BlueprintDebug
    |       +-- Breakpoint (conditional, hit count)
    |       +-- StepMode (INTO, OVER, OUT)
    |       +-- WatchExpression
    |       +-- ExecutionHistory
    |       +-- NodeProfile (timing per node)
    |
    +-- NodeLibrary
    |       +-- CategoryInfo (icon, color, description)
    |       +-- NodeEntry (metadata, factory)
    |       +-- SearchResult (relevance ranking)
    |
    +-- BlueprintSerializer
            +-- SerializationFormat (BINARY, JSON)
            +-- BlueprintHeader (version, checksum)
            +-- IncrementalSerializer
```

**Bytecode Operations (OpCode)**:
| Category | Operations |
|----------|------------|
| Control | NOP, HALT, JUMP, JUMP_IF, JUMP_IF_NOT, CALL, RETURN |
| Stack | PUSH, POP, DUP, SWAP |
| Variables | LOAD_LOCAL, STORE_LOCAL, LOAD_GLOBAL, STORE_GLOBAL |
| Arithmetic | ADD, SUB, MUL, DIV, MOD, NEG |
| Comparison | EQ, NE, LT, LE, GT, GE |
| Logical | AND, OR, NOT |
| Objects | GET_PROPERTY, SET_PROPERTY, CALL_METHOD |
| Special | LATENT_BEGIN, LATENT_END |

**Node Categories (18)**:
- EVENT: BeginPlay, Tick, InputAction, CustomEvent
- FLOW_CONTROL: Branch, Sequence, Loop variants
- FUNCTION: Pure/impure function calls
- VARIABLE: Get/Set with scopes
- MATH: Arithmetic, vector math
- STRING: String operations
- VECTOR: Vector2/3 operations
- TRANSFORM: Position, rotation, scale
- OBJECT: Object references
- ARRAY: Array manipulation
- UTILITY: Print, delay, cast
- MACRO: Reusable subgraphs
- CUSTOM: User-defined nodes
- DEBUG: Debug utilities
- AI: AI-specific nodes
- PHYSICS: Physics queries
- AUDIO: Audio playback
- UI: Widget interactions

## Execution Model

### Node Evaluation
```
Pull-based Data Flow:
    When a data input is needed:
        -> Evaluate source node (if pure)
        -> Return cached value (if impure already executed)

Push-based Execution Flow:
    Event fires (BeginPlay, Tick, etc.)
        -> Execute event node
        -> Follow execution pin to next node
        -> Execute that node
        -> Repeat until no execution pin
```

### Latent Actions
```
Delay Node:
    1. Schedule LatentAction with duration
    2. Return, saving execution state
    3. On tick: check if delay elapsed
    4. Resume execution from saved state
```

### Variable Scoping
```
Resolution Order:
    LOCAL (function-level)
        -> TEMPORARY (expression-level)
        -> INSTANCE (per-object)
        -> CLASS (shared by instances)
        -> GLOBAL (engine-wide)

Shadowing:
    Inner scope overrides outer scope
    No warning on shadowing
```

## Compilation Pipeline

### Graph to Bytecode
```
1. Validate graph (cycles, types)
2. Find entry points (event nodes)
3. For each entry point:
    a. Create function context
    b. Collect reachable nodes
    c. Topological sort
    d. Generate bytecode per node:
        - Evaluate inputs
        - Generate operation
        - Assign outputs
    e. Resolve jump labels
4. Build constant pool
5. Serialize to binary
```

### Bytecode Format
```
Header (16 bytes):
    Magic: "BPBC" (4 bytes)
    Version: uint32
    Function count: uint32
    Constant pool offset: uint32

Constant Pool:
    Type tag (1 byte) + length (4 bytes) + data

Function Table:
    Entry offset (4 bytes) per function

Instructions:
    OpCode (1 byte) + Operand (4 bytes)
```

## Debug Infrastructure

### Breakpoints
```python
class Breakpoint:
    node_id: str           # Node to break on
    pin_id: str | None     # Specific pin (optional)
    condition: str | None  # Expression to evaluate
    hit_count: int         # Current hits
    target_count: int      # Break after N hits (0 = always)
    log_only: bool         # Log instead of break
```

### Stepping
```
Step Into: Execute next node, follow execution pins
Step Over: Execute node, skip pure evaluations
Step Out: Run until current function returns
```

### Watch Expressions
```
Expression syntax:
    variable_name
    object.property
    array[index]
    function(args)

Updates on:
    - Step
    - Breakpoint hit
    - Explicit refresh
```

## Integration Points

### Engine Events
```python
# Event handlers
runtime.begin_play()          # Map load
runtime.end_play()            # Map unload
runtime.tick(delta_time)      # Frame update
runtime.input_action(name)    # Input event
runtime.input_axis(name, val) # Axis input
runtime.custom_event(name, *) # Custom events
```

### Object System
```python
# Object references
ObjectRef(class_name, object_id)

# Property access
GET_PROPERTY: obj.property
SET_PROPERTY: obj.property = value

# Method calls
CALL_METHOD: obj.method(args)
```

### Component Integration
```
Blueprints can:
    - Get/set component properties
    - Call component methods
    - Spawn/destroy actors
    - Interact with physics
    - Play audio/particles
```

## Thread Safety

| Component | Strategy |
|-----------|----------|
| BlueprintGraph | Not thread-safe (editor only) |
| BlueprintCompiler | Thread-safe (parallel compile) |
| BlueprintVM | Per-instance (no sharing) |
| ExecutionContextPool | Thread-safe pool |
| NodeRegistry | Read-only after init |

## Configuration

### VM Limits
```python
BlueprintVM(
    max_iterations=100000,    # Loop guard
    max_stack_depth=1000,     # Recursion guard
    latent_budget_ms=16.0,    # Per-frame latent budget
)
```

### Compiler Options
```python
CompilerOptions(
    optimization_level=OptimizationLevel.STANDARD,
    generate_debug_info=True,
    validate_types=True,
    allow_pure_speculation=True,
)
```

## Testing Strategy

### Unit Tests
- Type conversion correctness
- Node code generation
- Bytecode encoding/decoding
- Jump label resolution

### Integration Tests
- Full compile-execute cycle
- Latent action resume
- Event dispatch
- Breakpoint hitting

### Performance Tests
- Tight loop execution
- Many-node graphs
- Deep recursion
- High-frequency events
