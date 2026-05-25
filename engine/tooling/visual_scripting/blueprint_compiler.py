"""
FlowForge Blueprint Compiler - Compile blueprints to bytecode or native code.

Provides compilation capabilities:
- Blueprint to bytecode compilation
- Bytecode optimization passes
- Optional native code generation hooks
- Validation and error checking
- Incremental compilation support
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from .data_types import BlueprintType, TYPE_REGISTRY
from .graph_editor import BlueprintGraph, Connection
from .node_types import Node, NodeCategory, Pin, PinKind


class OpCode(IntEnum):
    """Bytecode operation codes."""
    NOP = 0x00
    HALT = 0x01

    # Stack operations
    PUSH_CONST = 0x10
    PUSH_VAR = 0x11
    POP = 0x12
    DUP = 0x13
    SWAP = 0x14

    # Variable operations
    LOAD_LOCAL = 0x20
    STORE_LOCAL = 0x21
    LOAD_INSTANCE = 0x22
    STORE_INSTANCE = 0x23
    LOAD_GLOBAL = 0x24
    STORE_GLOBAL = 0x25

    # Control flow
    JUMP = 0x30
    JUMP_IF = 0x31
    JUMP_IF_NOT = 0x32
    CALL = 0x33
    RETURN = 0x34
    YIELD = 0x35

    # Node operations
    EXEC_NODE = 0x40
    LOAD_PIN = 0x41
    STORE_PIN = 0x42

    # Arithmetic
    ADD = 0x50
    SUB = 0x51
    MUL = 0x52
    DIV = 0x53
    MOD = 0x54
    NEG = 0x55

    # Comparison
    EQ = 0x60
    NE = 0x61
    LT = 0x62
    LE = 0x63
    GT = 0x64
    GE = 0x65

    # Logical
    AND = 0x70
    OR = 0x71
    NOT = 0x72

    # Special
    DEBUG_BREAK = 0xF0
    TRACE = 0xF1


@dataclass
class BytecodeInstruction:
    """A single bytecode instruction."""
    opcode: OpCode
    operand: Any = None
    source_node_id: Optional[str] = None
    source_line: int = 0

    def encode(self) -> bytes:
        """Encode to binary."""
        # Simple encoding: 1 byte opcode + 4 byte operand index
        op_byte = struct.pack("B", self.opcode)
        if self.operand is not None:
            if isinstance(self.operand, int):
                operand_bytes = struct.pack("<i", self.operand)
            elif isinstance(self.operand, float):
                operand_bytes = struct.pack("<f", self.operand)
            else:
                operand_bytes = struct.pack("<i", 0)
        else:
            operand_bytes = struct.pack("<i", 0)
        return op_byte + operand_bytes

    @classmethod
    def decode(cls, data: bytes, offset: int = 0) -> Tuple[BytecodeInstruction, int]:
        """Decode from binary."""
        opcode = OpCode(struct.unpack("B", data[offset:offset + 1])[0])
        operand = struct.unpack("<i", data[offset + 1:offset + 5])[0]
        return cls(opcode=opcode, operand=operand), 5


@dataclass
class ConstantPool:
    """Pool of constants used in bytecode."""
    strings: List[str] = field(default_factory=list)
    numbers: List[float] = field(default_factory=list)
    node_ids: List[str] = field(default_factory=list)
    type_names: List[str] = field(default_factory=list)

    _string_index: Dict[str, int] = field(default_factory=dict)
    _number_index: Dict[float, int] = field(default_factory=dict)
    _node_index: Dict[str, int] = field(default_factory=dict)

    def add_string(self, s: str) -> int:
        """Add a string constant, return index."""
        if s in self._string_index:
            return self._string_index[s]
        idx = len(self.strings)
        self.strings.append(s)
        self._string_index[s] = idx
        return idx

    def add_number(self, n: float) -> int:
        """Add a number constant, return index."""
        if n in self._number_index:
            return self._number_index[n]
        idx = len(self.numbers)
        self.numbers.append(n)
        self._number_index[n] = idx
        return idx

    def add_node_id(self, node_id: str) -> int:
        """Add a node ID, return index."""
        if node_id in self._node_index:
            return self._node_index[node_id]
        idx = len(self.node_ids)
        self.node_ids.append(node_id)
        self._node_index[node_id] = idx
        return idx

    def add_type(self, type_name: str) -> int:
        """Add a type name, return index."""
        if type_name in self.type_names:
            return self.type_names.index(type_name)
        idx = len(self.type_names)
        self.type_names.append(type_name)
        return idx


@dataclass
class CompiledFunction:
    """A compiled function/event."""
    name: str
    entry_node_id: str
    instructions: List[BytecodeInstruction] = field(default_factory=list)
    local_variables: List[str] = field(default_factory=list)
    parameter_count: int = 0

    def get_size(self) -> int:
        """Get the size in bytes."""
        return len(self.instructions) * 5  # 5 bytes per instruction


@dataclass
class CompiledBlueprint:
    """A fully compiled blueprint."""
    source_id: str
    source_name: str
    version: str = "1.0"
    compile_time: float = 0.0

    constant_pool: ConstantPool = field(default_factory=ConstantPool)
    functions: List[CompiledFunction] = field(default_factory=list)

    # Debug info
    source_map: Dict[int, str] = field(default_factory=dict)  # instruction -> node_id
    variable_names: Dict[int, str] = field(default_factory=dict)

    # Metadata
    checksum: str = ""
    optimization_level: int = 0

    def get_function(self, name: str) -> Optional[CompiledFunction]:
        """Get a function by name."""
        for func in self.functions:
            if func.name == name:
                return func
        return None

    def get_total_instructions(self) -> int:
        """Get total instruction count."""
        return sum(len(f.instructions) for f in self.functions)


class OptimizationLevel(Enum):
    """Optimization levels."""
    NONE = 0
    BASIC = 1
    STANDARD = 2
    AGGRESSIVE = 3


@dataclass
class CompilerError:
    """A compilation error."""
    message: str
    node_id: Optional[str] = None
    pin_id: Optional[str] = None
    severity: str = "error"  # "error", "warning", "info"

    def __str__(self) -> str:
        location = f" at node {self.node_id}" if self.node_id else ""
        return f"[{self.severity.upper()}]{location}: {self.message}"


class CompilationResult:
    """Result of compilation."""

    def __init__(self):
        self.success = False
        self.blueprint: Optional[CompiledBlueprint] = None
        self.errors: List[CompilerError] = []
        self.warnings: List[CompilerError] = []
        self.compile_time: float = 0.0
        self.optimization_stats: Dict[str, Any] = {}

    def add_error(
        self,
        message: str,
        node_id: Optional[str] = None,
        pin_id: Optional[str] = None
    ) -> None:
        """Add a compilation error."""
        self.errors.append(CompilerError(message, node_id, pin_id, "error"))

    def add_warning(
        self,
        message: str,
        node_id: Optional[str] = None,
        pin_id: Optional[str] = None
    ) -> None:
        """Add a compilation warning."""
        self.warnings.append(CompilerError(message, node_id, pin_id, "warning"))


class BlueprintCompiler:
    """Compiler for blueprint graphs."""

    def __init__(
        self,
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    ):
        self.optimization_level = optimization_level
        self._current_graph: Optional[BlueprintGraph] = None
        self._current_result: Optional[CompilationResult] = None
        self._current_function: Optional[CompiledFunction] = None
        self._label_addresses: Dict[str, int] = {}
        self._pending_jumps: List[Tuple[int, str]] = []

    def compile(self, graph: BlueprintGraph) -> CompilationResult:
        """Compile a blueprint graph."""
        start_time = time.time()
        result = CompilationResult()
        self._current_graph = graph
        self._current_result = result

        # Validate graph first
        validation_errors = graph.validate()
        for error in validation_errors:
            result.add_warning(error)

        try:
            compiled = CompiledBlueprint(
                source_id=graph.id,
                source_name=graph.name,
                compile_time=start_time
            )

            # Compile each entry point as a function
            for entry_id in graph.entry_points:
                entry_node = graph.get_node(entry_id)
                if entry_node:
                    func = self._compile_function(entry_node, compiled)
                    if func:
                        compiled.functions.append(func)

            # Apply optimizations
            if self.optimization_level != OptimizationLevel.NONE:
                self._optimize(compiled, result)

            # Calculate checksum
            compiled.checksum = self._calculate_checksum(compiled)
            compiled.optimization_level = self.optimization_level.value

            result.blueprint = compiled
            result.success = len(result.errors) == 0

        except Exception as e:
            result.add_error(f"Compilation failed: {str(e)}")
            result.success = False

        result.compile_time = time.time() - start_time
        self._current_graph = None
        self._current_result = None

        return result

    def _compile_function(
        self,
        entry_node: Node,
        compiled: CompiledBlueprint
    ) -> Optional[CompiledFunction]:
        """Compile a single function/event."""
        func_name = entry_node.get_metadata().display_name
        func = CompiledFunction(
            name=func_name,
            entry_node_id=entry_node.id
        )
        self._current_function = func
        self._label_addresses.clear()
        self._pending_jumps.clear()

        # Compile the execution flow
        self._compile_node_chain(entry_node, compiled)

        # Add halt at end
        func.instructions.append(BytecodeInstruction(OpCode.HALT))

        # Resolve jumps
        self._resolve_jumps(func)

        self._current_function = None
        return func

    def _compile_node_chain(
        self,
        node: Node,
        compiled: CompiledBlueprint,
        visited: Optional[Set[str]] = None
    ) -> None:
        """Compile a chain of connected nodes."""
        if visited is None:
            visited = set()

        if node.id in visited:
            return
        visited.add(node.id)

        func = self._current_function
        if not func:
            return

        # Record label for this node
        self._label_addresses[node.id] = len(func.instructions)

        # Compile input value resolution
        self._compile_input_resolution(node, compiled)

        # Compile node execution
        node_idx = compiled.constant_pool.add_node_id(node.id)
        func.instructions.append(BytecodeInstruction(
            OpCode.EXEC_NODE,
            operand=node_idx,
            source_node_id=node.id
        ))

        # Handle flow control nodes specially
        meta = node.get_metadata()
        if meta.category == NodeCategory.FLOW_CONTROL:
            self._compile_flow_control(node, compiled, visited)
        else:
            # Follow single execution output
            for conn in self._current_graph.get_outgoing_connections(node.id):
                if conn.is_execution:
                    next_node = self._current_graph.get_node(conn.target_node_id)
                    if next_node:
                        self._compile_node_chain(next_node, compiled, visited)
                    break

    def _compile_input_resolution(
        self,
        node: Node,
        compiled: CompiledBlueprint
    ) -> None:
        """Compile code to resolve input pin values."""
        func = self._current_function
        if not func:
            return

        for pin in node.input_pins.values():
            if pin.kind == PinKind.DATA and pin.is_connected:
                # Find source connection
                for conn in self._current_graph.get_incoming_connections(node.id):
                    if conn.target_pin_id == pin.id:
                        source_node = self._current_graph.get_node(conn.source_node_id)
                        if source_node and source_node.get_metadata().is_pure:
                            # Inline pure function call
                            self._compile_pure_node(source_node, compiled)

                        # Load from source pin
                        pin_name = f"{conn.source_node_id}.{conn.source_pin_id}"
                        pin_idx = compiled.constant_pool.add_string(pin_name)
                        func.instructions.append(BytecodeInstruction(
                            OpCode.LOAD_PIN,
                            operand=pin_idx
                        ))

                        # Store to target pin
                        target_name = f"{node.id}.{pin.id}"
                        target_idx = compiled.constant_pool.add_string(target_name)
                        func.instructions.append(BytecodeInstruction(
                            OpCode.STORE_PIN,
                            operand=target_idx
                        ))
                        break

    def _compile_pure_node(
        self,
        node: Node,
        compiled: CompiledBlueprint
    ) -> None:
        """Compile a pure (side-effect-free) node."""
        # Pure nodes just need their inputs resolved and execution
        self._compile_input_resolution(node, compiled)

        node_idx = compiled.constant_pool.add_node_id(node.id)
        self._current_function.instructions.append(BytecodeInstruction(
            OpCode.EXEC_NODE,
            operand=node_idx,
            source_node_id=node.id
        ))

    def _compile_flow_control(
        self,
        node: Node,
        compiled: CompiledBlueprint,
        visited: Set[str]
    ) -> None:
        """Compile flow control node (branch, loop, etc.)."""
        func = self._current_function
        if not func:
            return

        node_name = node.get_metadata().display_name

        if "Branch" in node_name:
            self._compile_branch(node, compiled, visited)
        elif "For Loop" in node_name:
            self._compile_for_loop(node, compiled, visited)
        elif "While Loop" in node_name:
            self._compile_while_loop(node, compiled, visited)
        elif "Sequence" in node_name:
            self._compile_sequence(node, compiled, visited)
        else:
            # Default: follow first execution output
            for conn in self._current_graph.get_outgoing_connections(node.id):
                if conn.is_execution:
                    next_node = self._current_graph.get_node(conn.target_node_id)
                    if next_node:
                        self._compile_node_chain(next_node, compiled, visited)
                    break

    def _compile_branch(
        self,
        node: Node,
        compiled: CompiledBlueprint,
        visited: Set[str]
    ) -> None:
        """Compile a branch node."""
        func = self._current_function

        # Find True and False outputs
        true_conn = None
        false_conn = None

        for conn in self._current_graph.get_outgoing_connections(node.id):
            if conn.is_execution:
                source_pin = None
                for pin in node.output_pins.values():
                    if pin.id == conn.source_pin_id:
                        source_pin = pin
                        break
                if source_pin:
                    if source_pin.name == "True":
                        true_conn = conn
                    elif source_pin.name == "False":
                        false_conn = conn

        # Jump if condition is false
        false_label = f"{node.id}_false"
        end_label = f"{node.id}_end"

        # Add conditional jump (will be resolved later)
        jump_idx = len(func.instructions)
        func.instructions.append(BytecodeInstruction(OpCode.JUMP_IF_NOT, operand=0))
        self._pending_jumps.append((jump_idx, false_label))

        # Compile True branch
        if true_conn:
            true_node = self._current_graph.get_node(true_conn.target_node_id)
            if true_node:
                self._compile_node_chain(true_node, compiled, visited.copy())

        # Jump to end after True branch
        end_jump_idx = len(func.instructions)
        func.instructions.append(BytecodeInstruction(OpCode.JUMP, operand=0))
        self._pending_jumps.append((end_jump_idx, end_label))

        # False branch label
        self._label_addresses[false_label] = len(func.instructions)

        # Compile False branch
        if false_conn:
            false_node = self._current_graph.get_node(false_conn.target_node_id)
            if false_node:
                self._compile_node_chain(false_node, compiled, visited.copy())

        # End label
        self._label_addresses[end_label] = len(func.instructions)

    def _compile_for_loop(
        self,
        node: Node,
        compiled: CompiledBlueprint,
        visited: Set[str]
    ) -> None:
        """Compile a for loop node."""
        func = self._current_function

        loop_start = f"{node.id}_loop"
        loop_end = f"{node.id}_end"

        # Loop start label
        self._label_addresses[loop_start] = len(func.instructions)

        # Check condition (index <= last)
        # This is simplified - real implementation would evaluate the condition
        func.instructions.append(BytecodeInstruction(OpCode.EXEC_NODE, operand=compiled.constant_pool.add_node_id(node.id)))

        # Conditional jump to end
        jump_idx = len(func.instructions)
        func.instructions.append(BytecodeInstruction(OpCode.JUMP_IF_NOT, operand=0))
        self._pending_jumps.append((jump_idx, loop_end))

        # Compile loop body
        for conn in self._current_graph.get_outgoing_connections(node.id):
            if conn.is_execution:
                source_pin = None
                for pin in node.output_pins.values():
                    if pin.id == conn.source_pin_id:
                        source_pin = pin
                        break
                if source_pin and source_pin.name == "LoopBody":
                    body_node = self._current_graph.get_node(conn.target_node_id)
                    if body_node:
                        self._compile_node_chain(body_node, compiled, visited.copy())
                    break

        # Jump back to start
        loop_jump_idx = len(func.instructions)
        func.instructions.append(BytecodeInstruction(OpCode.JUMP, operand=0))
        self._pending_jumps.append((loop_jump_idx, loop_start))

        # End label
        self._label_addresses[loop_end] = len(func.instructions)

        # Compile completed branch
        for conn in self._current_graph.get_outgoing_connections(node.id):
            if conn.is_execution:
                source_pin = None
                for pin in node.output_pins.values():
                    if pin.id == conn.source_pin_id:
                        source_pin = pin
                        break
                if source_pin and source_pin.name == "Completed":
                    completed_node = self._current_graph.get_node(conn.target_node_id)
                    if completed_node:
                        self._compile_node_chain(completed_node, compiled, visited)
                    break

    def _compile_while_loop(
        self,
        node: Node,
        compiled: CompiledBlueprint,
        visited: Set[str]
    ) -> None:
        """Compile a while loop node."""
        # Similar to for loop but with condition check
        self._compile_for_loop(node, compiled, visited)

    def _compile_sequence(
        self,
        node: Node,
        compiled: CompiledBlueprint,
        visited: Set[str]
    ) -> None:
        """Compile a sequence node."""
        # Execute each output in order
        outputs = sorted(
            [conn for conn in self._current_graph.get_outgoing_connections(node.id) if conn.is_execution],
            key=lambda c: c.source_pin_id
        )

        for conn in outputs:
            next_node = self._current_graph.get_node(conn.target_node_id)
            if next_node:
                self._compile_node_chain(next_node, compiled, visited.copy())

    def _resolve_jumps(self, func: CompiledFunction) -> None:
        """Resolve pending jump addresses."""
        for jump_idx, label in self._pending_jumps:
            if label in self._label_addresses:
                target_addr = self._label_addresses[label]
                func.instructions[jump_idx].operand = target_addr
            else:
                self._current_result.add_warning(f"Unresolved jump label: {label}")

    def _optimize(
        self,
        compiled: CompiledBlueprint,
        result: CompilationResult
    ) -> None:
        """Apply optimization passes."""
        stats = {
            "removed_nops": 0,
            "folded_constants": 0,
            "eliminated_dead_code": 0
        }

        if self.optimization_level.value >= OptimizationLevel.BASIC.value:
            stats["removed_nops"] = self._remove_nops(compiled)

        if self.optimization_level.value >= OptimizationLevel.STANDARD.value:
            stats["folded_constants"] = self._fold_constants(compiled)

        if self.optimization_level.value >= OptimizationLevel.AGGRESSIVE.value:
            stats["eliminated_dead_code"] = self._eliminate_dead_code(compiled)

        result.optimization_stats = stats

    def _remove_nops(self, compiled: CompiledBlueprint) -> int:
        """Remove NOP instructions."""
        count = 0
        for func in compiled.functions:
            original_len = len(func.instructions)
            func.instructions = [i for i in func.instructions if i.opcode != OpCode.NOP]
            count += original_len - len(func.instructions)
        return count

    def _fold_constants(self, compiled: CompiledBlueprint) -> int:
        """Fold constant expressions."""
        # Simplified - would need full expression analysis
        return 0

    def _eliminate_dead_code(self, compiled: CompiledBlueprint) -> int:
        """Eliminate unreachable code."""
        # Simplified - would need control flow analysis
        return 0

    def _calculate_checksum(self, compiled: CompiledBlueprint) -> str:
        """Calculate a checksum for the compiled blueprint."""
        data = json.dumps({
            "source_id": compiled.source_id,
            "functions": [f.name for f in compiled.functions],
            "instruction_count": compiled.get_total_instructions(),
            "constants": len(compiled.constant_pool.strings) + len(compiled.constant_pool.numbers)
        }).encode()
        return hashlib.md5(data).hexdigest()


class BytecodeSerializer:
    """Serializer for compiled blueprints."""

    MAGIC = b"BPBC"  # Blueprint ByteCode
    VERSION = 1

    def serialize(self, compiled: CompiledBlueprint) -> bytes:
        """Serialize a compiled blueprint to bytes."""
        parts = []

        # Header
        parts.append(self.MAGIC)
        parts.append(struct.pack("<H", self.VERSION))

        # Metadata
        name_bytes = compiled.source_name.encode("utf-8")
        parts.append(struct.pack("<H", len(name_bytes)))
        parts.append(name_bytes)

        # Constant pool
        parts.append(self._serialize_constant_pool(compiled.constant_pool))

        # Functions
        parts.append(struct.pack("<H", len(compiled.functions)))
        for func in compiled.functions:
            parts.append(self._serialize_function(func))

        return b"".join(parts)

    def _serialize_constant_pool(self, pool: ConstantPool) -> bytes:
        """Serialize the constant pool."""
        parts = []

        # Strings
        parts.append(struct.pack("<H", len(pool.strings)))
        for s in pool.strings:
            s_bytes = s.encode("utf-8")
            parts.append(struct.pack("<H", len(s_bytes)))
            parts.append(s_bytes)

        # Numbers
        parts.append(struct.pack("<H", len(pool.numbers)))
        for n in pool.numbers:
            parts.append(struct.pack("<d", n))

        return b"".join(parts)

    def _serialize_function(self, func: CompiledFunction) -> bytes:
        """Serialize a function."""
        parts = []

        # Name
        name_bytes = func.name.encode("utf-8")
        parts.append(struct.pack("<H", len(name_bytes)))
        parts.append(name_bytes)

        # Instructions
        parts.append(struct.pack("<I", len(func.instructions)))
        for inst in func.instructions:
            parts.append(inst.encode())

        return b"".join(parts)

    def deserialize(self, data: bytes) -> Optional[CompiledBlueprint]:
        """Deserialize a compiled blueprint from bytes."""
        offset = 0

        # Check magic
        if data[offset:offset + 4] != self.MAGIC:
            return None
        offset += 4

        # Version
        version = struct.unpack("<H", data[offset:offset + 2])[0]
        if version != self.VERSION:
            return None
        offset += 2

        compiled = CompiledBlueprint(source_id="", source_name="")

        # Name
        name_len = struct.unpack("<H", data[offset:offset + 2])[0]
        offset += 2
        compiled.source_name = data[offset:offset + name_len].decode("utf-8")
        offset += name_len

        # Constant pool
        pool, offset = self._deserialize_constant_pool(data, offset)
        compiled.constant_pool = pool

        # Functions
        func_count = struct.unpack("<H", data[offset:offset + 2])[0]
        offset += 2
        for _ in range(func_count):
            func, offset = self._deserialize_function(data, offset)
            compiled.functions.append(func)

        return compiled

    def _deserialize_constant_pool(
        self,
        data: bytes,
        offset: int
    ) -> Tuple[ConstantPool, int]:
        """Deserialize the constant pool."""
        pool = ConstantPool()

        # Strings
        str_count = struct.unpack("<H", data[offset:offset + 2])[0]
        offset += 2
        for _ in range(str_count):
            str_len = struct.unpack("<H", data[offset:offset + 2])[0]
            offset += 2
            pool.strings.append(data[offset:offset + str_len].decode("utf-8"))
            offset += str_len

        # Numbers
        num_count = struct.unpack("<H", data[offset:offset + 2])[0]
        offset += 2
        for _ in range(num_count):
            pool.numbers.append(struct.unpack("<d", data[offset:offset + 8])[0])
            offset += 8

        return pool, offset

    def _deserialize_function(
        self,
        data: bytes,
        offset: int
    ) -> Tuple[CompiledFunction, int]:
        """Deserialize a function."""
        # Name
        name_len = struct.unpack("<H", data[offset:offset + 2])[0]
        offset += 2
        name = data[offset:offset + name_len].decode("utf-8")
        offset += name_len

        func = CompiledFunction(name=name, entry_node_id="")

        # Instructions
        inst_count = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        for _ in range(inst_count):
            inst, size = BytecodeInstruction.decode(data, offset)
            func.instructions.append(inst)
            offset += size

        return func, offset


# Convenience function
def compile_blueprint(
    graph: BlueprintGraph,
    optimization: OptimizationLevel = OptimizationLevel.STANDARD
) -> CompilationResult:
    """Compile a blueprint graph."""
    compiler = BlueprintCompiler(optimization)
    return compiler.compile(graph)
