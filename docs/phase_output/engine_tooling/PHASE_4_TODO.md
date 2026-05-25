# PHASE 4 TODO: Scripting and Logic Systems

## Overview

Phase 4 implements the visual scripting system (FlowForge). This enables gameplay programming through a node-based interface similar to Unreal Engine Blueprints.

---

## 1. Data Types

### 1.1 Primitive Types
- [ ] **T1.1.1**: Test all primitive type conversions
  - Acceptance: Bool <-> Int <-> Float conversions work
  - Acceptance: Precision loss warnings for Float->Int
  - File: `engine/tooling/visual_scripting/data_types.py`

- [ ] **T1.1.2**: Test string type operations
  - Acceptance: String concatenation works
  - Acceptance: String formatting works
  - Acceptance: String to numeric conversion

### 1.2 Vector Types
- [ ] **T1.2.1**: Test vector math operations
  - Acceptance: Magnitude calculation correct
  - Acceptance: Normalize works (handles zero)
  - Acceptance: Dot and cross products correct

- [ ] **T1.2.2**: Test vector swizzling
  - Acceptance: .xy, .xyz extractions work
  - Acceptance: Float broadcast to vector works

### 1.3 Transform Types
- [ ] **T1.3.1**: Test transform composition
  - Acceptance: Position, rotation, scale combine
  - Acceptance: Forward/right/up vectors correct

- [ ] **T1.3.2**: Test rotator conversion
  - Acceptance: Degrees <-> radians correct
  - Acceptance: Euler to quaternion conversion

### 1.4 Collection Types
- [ ] **T1.4.1**: Test array operations
  - Acceptance: Push, pop, get, set work
  - Acceptance: ForEach iteration works
  - Acceptance: Find/contains work
  - File: `engine/tooling/visual_scripting/data_types.py`

- [ ] **T1.4.2**: Test map operations
  - Acceptance: Key-value storage works
  - Acceptance: Contains key check works
  - Acceptance: Iteration works

### 1.5 Type System
- [ ] **T1.5.1**: Test can_connect_types matrix
  - Acceptance: Compatible types connect
  - Acceptance: Incompatible types rejected
  - Acceptance: Wildcard type matches all

- [ ] **T1.5.2**: Verify wire colors
  - Acceptance: Each type has distinct color
  - Acceptance: Colors match documentation

---

## 2. Node Types

### 2.1 Event Nodes
- [ ] **T2.1.1**: Test BeginPlay event
  - Acceptance: Fires on map load
  - Acceptance: Execution continues from output
  - File: `engine/tooling/visual_scripting/node_types.py`

- [ ] **T2.1.2**: Test Tick event
  - Acceptance: Fires each frame
  - Acceptance: Delta time input correct

- [ ] **T2.1.3**: Test InputAction event
  - Acceptance: Fires on input binding
  - Acceptance: Action name parameter works

### 2.2 Flow Control Nodes
- [ ] **T2.2.1**: Test Branch node
  - Acceptance: True branch taken when condition true
  - Acceptance: False branch taken when condition false

- [ ] **T2.2.2**: Test Sequence node
  - Acceptance: All outputs execute in order
  - Acceptance: Short-circuit if any fails

- [ ] **T2.2.3**: Test ForLoop node
  - Acceptance: Loop body executes N times
  - Acceptance: Index variable correct
  - Acceptance: Break pin exits loop

- [ ] **T2.2.4**: Test ForEachLoop node
  - Acceptance: Iterates array elements
  - Acceptance: Element variable correct
  - Acceptance: Index variable correct

- [ ] **T2.2.5**: Test WhileLoop node
  - Acceptance: Loop while condition true
  - Acceptance: Iteration limit prevents infinite

- [ ] **T2.2.6**: Test Switch node
  - Acceptance: Correct case selected
  - Acceptance: Default case works

- [ ] **T2.2.7**: Test Gate node
  - Acceptance: Open/close controls flow
  - Acceptance: Toggle flips state

- [ ] **T2.2.8**: Test DoOnce node
  - Acceptance: Executes only first time
  - Acceptance: Reset pin re-enables

- [ ] **T2.2.9**: Test FlipFlop node
  - Acceptance: Alternates A/B outputs
  - Acceptance: State persists

### 2.3 Function Nodes
- [ ] **T2.3.1**: Test CallFunction node
  - Acceptance: Function called with args
  - Acceptance: Return value captured

- [ ] **T2.3.2**: Test PureFunction node
  - Acceptance: No execution pins
  - Acceptance: Evaluates on demand

### 2.4 Variable Nodes
- [ ] **T2.4.1**: Test GetVariable node
  - Acceptance: Returns current value
  - Acceptance: All scopes accessible

- [ ] **T2.4.2**: Test SetVariable node
  - Acceptance: Updates value
  - Acceptance: Output returns new value

### 2.5 Macro Nodes
- [ ] **T2.5.1**: Test Macro expansion
  - Acceptance: Macro contents inlined
  - Acceptance: Inputs/outputs connected
  - Acceptance: Recursive macros prevented

---

## 3. Graph Editor

### 3.1 Graph Operations
- [ ] **T3.1.1**: Test node creation
  - Acceptance: Nodes added to graph
  - Acceptance: Positions recorded
  - File: `engine/tooling/visual_scripting/graph_editor.py`

- [ ] **T3.1.2**: Test node deletion
  - Acceptance: Nodes removed
  - Acceptance: Connections cleaned up

- [ ] **T3.1.3**: Test node duplication
  - Acceptance: Copy with unique IDs
  - Acceptance: Internal connections preserved

### 3.2 Connection Operations
- [ ] **T3.2.1**: Test connection creation
  - Acceptance: Valid connections created
  - Acceptance: Invalid connections rejected
  - Acceptance: Type validation runs

- [ ] **T3.2.2**: Test connection deletion
  - Acceptance: Connection removed
  - Acceptance: Both ends updated

- [ ] **T3.2.3**: Test cycle detection
  - Acceptance: Cycles prevented
  - Acceptance: Error message descriptive

### 3.3 Selection
- [ ] **T3.3.1**: Test selection modes
  - Acceptance: REPLACE clears and selects
  - Acceptance: ADD adds to selection
  - Acceptance: REMOVE removes from selection
  - Acceptance: TOGGLE toggles selection

- [ ] **T3.3.2**: Test box selection
  - Acceptance: Nodes in rect selected
  - Acceptance: Partial overlap handled

### 3.4 Undo/Redo
- [ ] **T3.4.1**: Test undo operations
  - Acceptance: Node create undoable
  - Acceptance: Node delete undoable
  - Acceptance: Connection undoable
  - Acceptance: Move undoable

- [ ] **T3.4.2**: Complete undo implementation
  - Acceptance: Full node serialization for undo
  - Note: Currently partial implementation
  - File: `engine/tooling/visual_scripting/graph_editor.py`

### 3.5 View Controls
- [ ] **T3.5.1**: Test zoom
  - Acceptance: Zoom range 0.1 to 5.0
  - Acceptance: Zoom to cursor point

- [ ] **T3.5.2**: Test pan
  - Acceptance: Pan updates offset
  - Acceptance: Coordinate conversion correct

- [ ] **T3.5.3**: Test minimap
  - Acceptance: Minimap shows all nodes
  - Acceptance: Click navigates to position

---

## 4. Compiler

### 4.1 Bytecode Generation
- [ ] **T4.1.1**: Test instruction encoding
  - Acceptance: OpCode + operand correct
  - Acceptance: Endianness consistent
  - File: `engine/tooling/visual_scripting/blueprint_compiler.py`

- [ ] **T4.1.2**: Test constant pool
  - Acceptance: Strings deduplicated
  - Acceptance: Numbers stored correctly
  - Acceptance: References indexed

### 4.2 Control Flow Compilation
- [ ] **T4.2.1**: Test branch compilation
  - Acceptance: Jump instructions correct
  - Acceptance: Label resolution works

- [ ] **T4.2.2**: Test loop compilation
  - Acceptance: For loop bytecode correct
  - Acceptance: While loop bytecode correct
  - Acceptance: Break jumps to end

- [ ] **T4.2.3**: Test sequence compilation
  - Acceptance: Multiple paths generated
  - Acceptance: Order preserved

### 4.3 Optimization
- [ ] **T4.3.1**: Implement constant folding
  - Acceptance: 1 + 2 -> 3 at compile time
  - Acceptance: String concat folded
  - Note: Currently placeholder

- [ ] **T4.3.2**: Implement dead code elimination
  - Acceptance: Unreachable code removed
  - Acceptance: Unused variables removed
  - Note: Currently placeholder

### 4.4 Serialization
- [ ] **T4.4.1**: Test binary serialization
  - Acceptance: Bytecode saves correctly
  - Acceptance: Bytecode loads correctly
  - Acceptance: Version checking works

---

## 5. Runtime

### 5.1 VM Execution
- [ ] **T5.1.1**: Test stack operations
  - Acceptance: Push/pop work
  - Acceptance: Stack overflow handled
  - File: `engine/tooling/visual_scripting/blueprint_runtime.py`

- [ ] **T5.1.2**: Test arithmetic operations
  - Acceptance: ADD, SUB, MUL, DIV correct
  - Acceptance: Division by zero handled

- [ ] **T5.1.3**: Test comparison operations
  - Acceptance: EQ, NE, LT, GT, LE, GE correct
  - Acceptance: Type coercion works

### 5.2 Latent Actions
- [ ] **T5.2.1**: Test Delay node
  - Acceptance: Execution pauses for duration
  - Acceptance: Resumes after delay
  - Acceptance: Multiple delays concurrent

- [ ] **T5.2.2**: Test latent action cancellation
  - Acceptance: Pending actions cancellable
  - Acceptance: No crash on orphaned actions

### 5.3 Events
- [ ] **T5.3.1**: Test event dispatch
  - Acceptance: Events fire handlers
  - Acceptance: Multiple handlers work
  - Acceptance: Event queue ordering

- [ ] **T5.3.2**: Test custom events
  - Acceptance: Custom event definition
  - Acceptance: Fire custom event
  - Acceptance: Parameters passed

### 5.4 Safety
- [ ] **T5.4.1**: Test iteration limit
  - Acceptance: Infinite loops stopped
  - Acceptance: Error reported

- [ ] **T5.4.2**: Test stack limit
  - Acceptance: Deep recursion stopped
  - Acceptance: Error reported

---

## 6. Debugger

### 6.1 Breakpoints
- [ ] **T6.1.1**: Test basic breakpoints
  - Acceptance: Execution stops at breakpoint
  - Acceptance: State inspectable
  - File: `engine/tooling/visual_scripting/blueprint_debug.py`

- [ ] **T6.1.2**: Test conditional breakpoints
  - Acceptance: Only breaks when condition true
  - Acceptance: Hit count tracking works

### 6.2 Stepping
- [ ] **T6.2.1**: Test step into
  - Acceptance: Steps to next node
  - Acceptance: Follows execution flow

- [ ] **T6.2.2**: Test step over
  - Acceptance: Skips function internals
  - Acceptance: Returns at same level

- [ ] **T6.2.3**: Test step out
  - Acceptance: Runs until function returns
  - Acceptance: Stops at caller

### 6.3 Watch Expressions
- [ ] **T6.3.1**: Test variable watching
  - Acceptance: Variable values shown
  - Acceptance: Updates on step

- [ ] **T6.3.2**: Test expression evaluation
  - Acceptance: Complex expressions evaluate
  - Acceptance: Errors shown for invalid

---

## 7. Node Library

### 7.1 Search
- [ ] **T7.1.1**: Test search functionality
  - Acceptance: Finds nodes by name
  - Acceptance: Finds nodes by keyword
  - Acceptance: Category filtering works
  - File: `engine/tooling/visual_scripting/node_library.py`

- [ ] **T7.1.2**: Test relevance ranking
  - Acceptance: Exact matches first
  - Acceptance: Partial matches lower

### 7.2 Custom Nodes
- [ ] **T7.2.1**: Test custom node registration
  - Acceptance: Custom nodes appear in library
  - Acceptance: Custom nodes compilable

---

## 8. Serialization

### 8.1 Graph Persistence
- [ ] **T8.1.1**: Test JSON save/load
  - Acceptance: Graph saves to JSON
  - Acceptance: Graph loads correctly
  - Acceptance: All node types handled
  - File: `engine/tooling/visual_scripting/blueprint_serializer.py`

- [ ] **T8.1.2**: Test binary save/load
  - Acceptance: Smaller than JSON
  - Acceptance: Faster load time

- [ ] **T8.1.3**: Test version compatibility
  - Acceptance: Old versions loadable
  - Acceptance: Migration path exists

---

## Integration Tests

### I1. Complete Script
- [ ] **I1.1**: Create, compile, execute script
  - Steps: Create nodes, connect, compile, run
  - Acceptance: Script executes correctly

### I2. Debug Session
- [ ] **I2.1**: Set breakpoint, step through
  - Steps: Add breakpoint, run, step, inspect
  - Acceptance: Full debug session works

### I3. Latent Workflow
- [ ] **I3.1**: Test delay across frames
  - Steps: Delay node with 1 second, verify timing
  - Acceptance: Delay accurate within 1 frame

---

## Performance Targets

| Metric | Target | Test Method |
|--------|--------|-------------|
| Compile (100 nodes) | < 100ms | Benchmark |
| Execute (tight loop) | > 1M ops/sec | Benchmark |
| VM startup | < 1ms | Benchmark |
| Memory per context | < 1KB | Profile |

---

## Dependencies

### Required Before Phase 4
- Phase 1: Undo system (graph editing)
- Phase 3: Material system (for material-related nodes)

### Integration Points
- Actor/component system for object nodes
- Input system for input events
- Physics for physics nodes
- Audio for audio nodes

### Blocks Phase 5+
- Debug tools may integrate with script debugging
- Profiler may profile script execution
