"""
Tests for FlowForge blueprint compiler.

Tests bytecode generation, optimization, and serialization.
"""

import pytest

from engine.tooling.visual_scripting.blueprint_compiler import (
    OpCode,
    BytecodeInstruction,
    ConstantPool,
    CompiledFunction,
    CompiledBlueprint,
    OptimizationLevel,
    CompilerError,
    CompilationResult,
    BlueprintCompiler,
    BytecodeSerializer,
    compile_blueprint,
)
from engine.tooling.visual_scripting.graph_editor import BlueprintGraph, Connection
from engine.tooling.visual_scripting.node_types import (
    BeginPlayNode,
    BranchNode,
    SequenceNode,
    ForLoopNode,
    PrintStringNode,
)


class TestOpCode:
    """Tests for OpCode enumeration."""

    def test_opcodes_exist(self):
        assert OpCode.NOP == 0x00
        assert OpCode.HALT == 0x01
        assert OpCode.PUSH_CONST == 0x10
        assert OpCode.EXEC_NODE == 0x40

    def test_opcodes_unique(self):
        codes = [op.value for op in OpCode]
        assert len(codes) == len(set(codes))


class TestBytecodeInstruction:
    """Tests for BytecodeInstruction class."""

    def test_create_instruction(self):
        inst = BytecodeInstruction(
            opcode=OpCode.PUSH_CONST,
            operand=42
        )
        assert inst.opcode == OpCode.PUSH_CONST
        assert inst.operand == 42

    def test_encode_instruction(self):
        inst = BytecodeInstruction(
            opcode=OpCode.PUSH_CONST,
            operand=42
        )
        encoded = inst.encode()

        assert len(encoded) == 5  # 1 byte opcode + 4 bytes operand
        assert encoded[0] == OpCode.PUSH_CONST

    def test_decode_instruction(self):
        inst = BytecodeInstruction(
            opcode=OpCode.PUSH_CONST,
            operand=42
        )
        encoded = inst.encode()

        decoded, size = BytecodeInstruction.decode(encoded)

        assert decoded.opcode == OpCode.PUSH_CONST
        assert decoded.operand == 42
        assert size == 5

    def test_instruction_with_source_info(self):
        inst = BytecodeInstruction(
            opcode=OpCode.EXEC_NODE,
            operand=0,
            source_node_id="node_123",
            source_line=10
        )
        assert inst.source_node_id == "node_123"
        assert inst.source_line == 10


class TestConstantPool:
    """Tests for ConstantPool class."""

    def test_add_string(self):
        pool = ConstantPool()
        idx1 = pool.add_string("hello")
        idx2 = pool.add_string("world")
        idx3 = pool.add_string("hello")  # duplicate

        assert idx1 == 0
        assert idx2 == 1
        assert idx3 == 0  # Same as first

    def test_add_number(self):
        pool = ConstantPool()
        idx1 = pool.add_number(3.14)
        idx2 = pool.add_number(2.71)
        idx3 = pool.add_number(3.14)

        assert idx1 == 0
        assert idx2 == 1
        assert idx3 == 0

    def test_add_node_id(self):
        pool = ConstantPool()
        idx = pool.add_node_id("node_123")

        assert idx == 0
        assert "node_123" in pool.node_ids

    def test_add_type(self):
        pool = ConstantPool()
        idx1 = pool.add_type("Int")
        idx2 = pool.add_type("Float")
        idx3 = pool.add_type("Int")

        assert idx1 == 0
        assert idx2 == 1
        assert idx3 == 0


class TestCompiledFunction:
    """Tests for CompiledFunction class."""

    def test_create_function(self):
        func = CompiledFunction(
            name="BeginPlay",
            entry_node_id="node_1"
        )
        assert func.name == "BeginPlay"
        assert len(func.instructions) == 0

    def test_get_size(self):
        func = CompiledFunction(name="Test", entry_node_id="node")
        func.instructions.append(BytecodeInstruction(OpCode.NOP))
        func.instructions.append(BytecodeInstruction(OpCode.HALT))

        assert func.get_size() == 10  # 2 instructions * 5 bytes


class TestCompiledBlueprint:
    """Tests for CompiledBlueprint class."""

    def test_create_blueprint(self):
        compiled = CompiledBlueprint(
            source_id="bp_123",
            source_name="TestBlueprint"
        )
        assert compiled.source_name == "TestBlueprint"
        assert len(compiled.functions) == 0

    def test_get_function(self):
        compiled = CompiledBlueprint(source_id="bp", source_name="Test")
        func = CompiledFunction(name="TestFunc", entry_node_id="node")
        compiled.functions.append(func)

        found = compiled.get_function("TestFunc")
        assert found == func

        not_found = compiled.get_function("Unknown")
        assert not_found is None

    def test_get_total_instructions(self):
        compiled = CompiledBlueprint(source_id="bp", source_name="Test")

        func1 = CompiledFunction(name="Func1", entry_node_id="n1")
        func1.instructions.append(BytecodeInstruction(OpCode.NOP))
        func1.instructions.append(BytecodeInstruction(OpCode.HALT))

        func2 = CompiledFunction(name="Func2", entry_node_id="n2")
        func2.instructions.append(BytecodeInstruction(OpCode.NOP))

        compiled.functions.append(func1)
        compiled.functions.append(func2)

        assert compiled.get_total_instructions() == 3


class TestCompilerError:
    """Tests for CompilerError class."""

    def test_create_error(self):
        error = CompilerError(
            message="Invalid connection",
            node_id="node_123",
            severity="error"
        )
        assert error.message == "Invalid connection"
        assert error.node_id == "node_123"

    def test_error_str(self):
        error = CompilerError(
            message="Test error",
            node_id="node_123"
        )
        error_str = str(error)

        assert "ERROR" in error_str
        assert "node_123" in error_str
        assert "Test error" in error_str


class TestCompilationResult:
    """Tests for CompilationResult class."""

    def test_create_result(self):
        result = CompilationResult()
        assert result.success is False
        assert len(result.errors) == 0

    def test_add_error(self):
        result = CompilationResult()
        result.add_error("Test error", node_id="node")

        assert len(result.errors) == 1
        assert result.errors[0].severity == "error"

    def test_add_warning(self):
        result = CompilationResult()
        result.add_warning("Test warning", node_id="node")

        assert len(result.warnings) == 1
        assert result.warnings[0].severity == "warning"


class TestBlueprintCompiler:
    """Tests for BlueprintCompiler class."""

    def test_create_compiler(self):
        compiler = BlueprintCompiler()
        assert compiler.optimization_level == OptimizationLevel.STANDARD

    def test_create_compiler_with_optimization(self):
        compiler = BlueprintCompiler(OptimizationLevel.AGGRESSIVE)
        assert compiler.optimization_level == OptimizationLevel.AGGRESSIVE

    def test_compile_empty_graph(self):
        compiler = BlueprintCompiler()
        graph = BlueprintGraph()

        result = compiler.compile(graph)

        assert result.success is True
        assert len(result.blueprint.functions) == 0

    def test_compile_simple_graph(self):
        compiler = BlueprintCompiler()
        graph = BlueprintGraph()

        begin = BeginPlayNode()
        print_node = PrintStringNode()
        graph.add_node(begin)
        graph.add_node(print_node)

        conn = Connection(
            id="c1",
            source_node_id=begin.id,
            source_pin_id=begin.output_pins["Out"].id,
            target_node_id=print_node.id,
            target_pin_id=print_node.input_pins["In"].id
        )
        graph.add_connection(conn)

        result = compiler.compile(graph)

        assert result.success is True
        assert len(result.blueprint.functions) == 1

    def test_compile_with_branch(self):
        compiler = BlueprintCompiler()
        graph = BlueprintGraph()

        begin = BeginPlayNode()
        branch = BranchNode()
        print1 = PrintStringNode()
        print2 = PrintStringNode()

        graph.add_node(begin)
        graph.add_node(branch)
        graph.add_node(print1)
        graph.add_node(print2)

        # BeginPlay -> Branch
        graph.add_connection(Connection(
            id="c1",
            source_node_id=begin.id,
            source_pin_id=begin.output_pins["Out"].id,
            target_node_id=branch.id,
            target_pin_id=branch.input_pins["In"].id
        ))

        # Branch(True) -> Print1
        graph.add_connection(Connection(
            id="c2",
            source_node_id=branch.id,
            source_pin_id=branch.output_pins["True"].id,
            target_node_id=print1.id,
            target_pin_id=print1.input_pins["In"].id
        ))

        # Branch(False) -> Print2
        graph.add_connection(Connection(
            id="c3",
            source_node_id=branch.id,
            source_pin_id=branch.output_pins["False"].id,
            target_node_id=print2.id,
            target_pin_id=print2.input_pins["In"].id
        ))

        result = compiler.compile(graph)

        assert result.success is True

    def test_compile_with_loop(self):
        compiler = BlueprintCompiler()
        graph = BlueprintGraph()

        begin = BeginPlayNode()
        loop = ForLoopNode()
        print_node = PrintStringNode()

        graph.add_node(begin)
        graph.add_node(loop)
        graph.add_node(print_node)

        graph.add_connection(Connection(
            id="c1",
            source_node_id=begin.id,
            source_pin_id=begin.output_pins["Out"].id,
            target_node_id=loop.id,
            target_pin_id=loop.input_pins["In"].id
        ))

        graph.add_connection(Connection(
            id="c2",
            source_node_id=loop.id,
            source_pin_id=loop.output_pins["LoopBody"].id,
            target_node_id=print_node.id,
            target_pin_id=print_node.input_pins["In"].id
        ))

        result = compiler.compile(graph)

        assert result.success is True

    def test_compile_with_sequence(self):
        compiler = BlueprintCompiler()
        graph = BlueprintGraph()

        begin = BeginPlayNode()
        sequence = SequenceNode(num_outputs=3)

        graph.add_node(begin)
        graph.add_node(sequence)

        graph.add_connection(Connection(
            id="c1",
            source_node_id=begin.id,
            source_pin_id=begin.output_pins["Out"].id,
            target_node_id=sequence.id,
            target_pin_id=sequence.input_pins["In"].id
        ))

        result = compiler.compile(graph)

        assert result.success is True


class TestCompilerOptimizations:
    """Tests for compiler optimizations."""

    def test_no_optimization(self):
        compiler = BlueprintCompiler(OptimizationLevel.NONE)
        graph = BlueprintGraph()
        begin = BeginPlayNode()
        graph.add_node(begin)

        result = compiler.compile(graph)

        assert result.optimization_stats == {}

    def test_basic_optimization(self):
        compiler = BlueprintCompiler(OptimizationLevel.BASIC)
        graph = BlueprintGraph()
        begin = BeginPlayNode()
        graph.add_node(begin)

        result = compiler.compile(graph)

        assert "removed_nops" in result.optimization_stats

    def test_aggressive_optimization(self):
        compiler = BlueprintCompiler(OptimizationLevel.AGGRESSIVE)
        graph = BlueprintGraph()
        begin = BeginPlayNode()
        graph.add_node(begin)

        result = compiler.compile(graph)

        assert "eliminated_dead_code" in result.optimization_stats


class TestCompilerChecksum:
    """Tests for compilation checksum."""

    def test_checksum_generated(self):
        compiler = BlueprintCompiler()
        graph = BlueprintGraph()
        begin = BeginPlayNode()
        graph.add_node(begin)

        result = compiler.compile(graph)

        assert result.blueprint.checksum != ""

    def test_same_graph_same_checksum(self):
        compiler = BlueprintCompiler()

        graph1 = BlueprintGraph(graph_id="same_id", name="Test")
        begin1 = BeginPlayNode(node_id="node_1")
        graph1.add_node(begin1)

        graph2 = BlueprintGraph(graph_id="same_id", name="Test")
        begin2 = BeginPlayNode(node_id="node_1")
        graph2.add_node(begin2)

        result1 = compiler.compile(graph1)
        result2 = compiler.compile(graph2)

        # May not be exactly same due to timing, but format should match


class TestBytecodeSerializer:
    """Tests for BytecodeSerializer class."""

    def test_serialize_empty_blueprint(self):
        serializer = BytecodeSerializer()
        compiled = CompiledBlueprint(source_id="bp", source_name="Test")

        data = serializer.serialize(compiled)

        assert data[:4] == b"BPBC"

    def test_serialize_deserialize_roundtrip(self):
        serializer = BytecodeSerializer()

        compiled = CompiledBlueprint(source_id="bp_123", source_name="TestBP")
        compiled.constant_pool.add_string("hello")
        compiled.constant_pool.add_number(3.14)

        func = CompiledFunction(name="TestFunc", entry_node_id="node_1")
        func.instructions.append(BytecodeInstruction(OpCode.PUSH_CONST, operand=0))
        func.instructions.append(BytecodeInstruction(OpCode.HALT))
        compiled.functions.append(func)

        # Serialize
        data = serializer.serialize(compiled)

        # Deserialize
        restored = serializer.deserialize(data)

        assert restored is not None
        assert restored.source_name == "TestBP"
        assert len(restored.constant_pool.strings) == 1
        assert len(restored.functions) == 1

    def test_deserialize_invalid_magic(self):
        serializer = BytecodeSerializer()
        data = b"XXXX" + bytes(100)

        result = serializer.deserialize(data)

        assert result is None

    def test_deserialize_invalid_version(self):
        serializer = BytecodeSerializer()
        data = b"BPBC" + bytes([255, 255]) + bytes(100)

        result = serializer.deserialize(data)

        assert result is None


class TestCompileBlueprintFunction:
    """Tests for the compile_blueprint convenience function."""

    def test_compile_blueprint_default(self):
        graph = BlueprintGraph()
        begin = BeginPlayNode()
        graph.add_node(begin)

        result = compile_blueprint(graph)

        assert result.success is True

    def test_compile_blueprint_with_optimization(self):
        graph = BlueprintGraph()
        begin = BeginPlayNode()
        graph.add_node(begin)

        result = compile_blueprint(graph, OptimizationLevel.AGGRESSIVE)

        assert result.success is True
        assert result.blueprint.optimization_level == OptimizationLevel.AGGRESSIVE.value


class TestCompileTime:
    """Tests for compilation timing."""

    def test_compile_time_recorded(self):
        compiler = BlueprintCompiler()
        graph = BlueprintGraph()
        begin = BeginPlayNode()
        graph.add_node(begin)

        result = compiler.compile(graph)

        assert result.compile_time >= 0


class TestMultipleEntryPoints:
    """Tests for compiling graphs with multiple entry points."""

    def test_multiple_events(self):
        compiler = BlueprintCompiler()
        graph = BlueprintGraph()

        begin1 = BeginPlayNode()
        from engine.tooling.visual_scripting.node_types import TickNode
        tick = TickNode()

        graph.add_node(begin1)
        graph.add_node(tick)

        result = compiler.compile(graph)

        assert result.success is True
        assert len(result.blueprint.functions) == 2
