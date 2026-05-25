"""
Comprehensive unit tests for the Shell system.
Tests code execution, namespace management, context binding, and history.
"""
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from foundation.shell import shell, Shell, ExecutionResult


@pytest.fixture(autouse=True)
def reset_shell():
    """Reset shell state before each test."""
    shell.reset_namespace()
    shell.clear_history()
    shell.unbind()
    yield


class TestExecution:
    """Tests for code execution functionality."""

    def test_execute_expression(self):
        """Expressions should return their computed value."""
        result = shell.execute("1 + 1")
        assert result.success is True
        assert result.value == 2

    def test_execute_statement(self):
        """Statements should execute and modify namespace."""
        result = shell.execute("x = 42")
        assert result.success is True
        assert shell.namespace.get("x") == 42

    def test_execute_invalid_syntax(self):
        """Invalid syntax should result in failure."""
        result = shell.execute("if if if")
        assert result.success is False
        assert result.error is not None

    def test_execute_runtime_error(self):
        """Runtime errors should be caught and reported."""
        result = shell.execute("1 / 0")
        assert result.success is False
        assert result.error_type == "ZeroDivisionError"

    def test_execute_name_error(self):
        """Undefined name access should be caught."""
        result = shell.execute("undefined_variable")
        assert result.success is False
        assert result.error_type == "NameError"

    def test_execute_type_error(self):
        """Type errors should be caught."""
        result = shell.execute("'string' + 5")
        assert result.success is False
        assert result.error_type == "TypeError"

    def test_execute_multiline(self):
        """Multiline statements should work."""
        code = """
def add(a, b):
    return a + b
"""
        result = shell.execute(code)
        assert result.success is True

        result2 = shell.execute("add(2, 3)")
        assert result2.value == 5

    def test_execute_class_definition(self):
        """Class definitions should work."""
        code = """
class Counter:
    def __init__(self):
        self.value = 0
    def inc(self):
        self.value += 1
"""
        result = shell.execute(code)
        assert result.success is True
        assert "Counter" in shell.namespace

    def test_execute_import(self):
        """Import statements should work."""
        result = shell.execute("import math")
        assert result.success is True
        assert "math" in shell.namespace


class TestNamespace:
    """Tests for namespace management."""

    def test_default_namespace_has_mirror(self):
        """Namespace should have mirror available."""
        assert "mirror" in shell.namespace

    def test_default_namespace_has_registry(self):
        """Namespace should have registry available."""
        assert "registry" in shell.namespace

    def test_default_namespace_has_tracker(self):
        """Namespace should have tracker available."""
        assert "tracker" in shell.namespace

    def test_default_namespace_has_inspector(self):
        """Namespace should have inspector available."""
        assert "inspector" in shell.namespace

    def test_default_namespace_has_serializer(self):
        """Namespace should have serializer available."""
        assert "serializer" in shell.namespace

    def test_default_namespace_has_shell(self):
        """Namespace should have shell reference."""
        assert "shell" in shell.namespace
        assert shell.namespace["shell"] is shell

    def test_convenience_functions(self):
        """Namespace should have convenience functions."""
        assert "inspect" in shell.namespace
        assert "save" in shell.namespace
        assert "load" in shell.namespace
        assert "copy" in shell.namespace
        assert "types" in shell.namespace
        assert "undo" in shell.namespace
        assert "redo" in shell.namespace
        assert "instances" in shell.namespace
        assert "dirty" in shell.namespace

    def test_reset_namespace(self):
        """reset_namespace should clear user-defined variables."""
        shell.execute("custom_var = 123")
        assert "custom_var" in shell.namespace
        shell.reset_namespace()
        assert "custom_var" not in shell.namespace

    def test_reset_preserves_core_systems(self):
        """reset_namespace should preserve core systems."""
        shell.execute("custom = 'test'")
        shell.reset_namespace()

        assert "mirror" in shell.namespace
        assert "registry" in shell.namespace
        assert "tracker" in shell.namespace

    def test_namespace_property_returns_dict(self):
        """namespace property should return a dict."""
        assert isinstance(shell.namespace, dict)


class TestResultVariables:
    """Tests for automatic result variable management."""

    def test_underscore_stores_last_result(self):
        """_ should store the last expression result."""
        shell.execute("10 + 5")
        assert shell.namespace.get("_") == 15

    def test_double_underscore_stores_second_last(self):
        """__ should store the second to last result."""
        shell.execute("1")
        shell.execute("2")
        assert shell.namespace.get("__") == 1
        assert shell.namespace.get("_") == 2

    def test_triple_underscore_stores_third_last(self):
        """___ should store the third to last result."""
        shell.execute("1")
        shell.execute("2")
        shell.execute("3")
        assert shell.namespace.get("___") == 1
        assert shell.namespace.get("__") == 2
        assert shell.namespace.get("_") == 3

    def test_none_result_not_stored(self):
        """None results should not update result variables."""
        shell.execute("10")
        assert shell.namespace.get("_") == 10

        shell.execute("x = 5")  # Statement returns None
        # _ should still be 10
        assert shell.namespace.get("_") == 10

    def test_result_vars_chain_correctly(self):
        """Result variables should chain through multiple executions."""
        shell.execute("100")
        shell.execute("200")
        shell.execute("300")
        shell.execute("400")

        assert shell.namespace.get("_") == 400
        assert shell.namespace.get("__") == 300
        assert shell.namespace.get("___") == 200


class TestContextBinding:
    """Tests for object context binding."""

    def test_bind_object(self):
        """bind() should set object as 'self' in namespace."""
        obj = {"value": 42}
        shell.bind(obj)
        assert shell.bound_object is obj
        assert shell.namespace.get("self") is obj

    def test_unbind_object(self):
        """unbind() should remove 'self' from namespace."""
        shell.bind({"x": 1})
        shell.unbind()
        assert shell.bound_object is None
        assert "self" not in shell.namespace

    def test_access_bound_object(self):
        """Should be able to access bound object via 'self'."""
        class Obj:
            value = 100

        obj = Obj()
        shell.bind(obj)
        result = shell.execute("self.value")
        assert result.value == 100

    def test_modify_bound_object(self):
        """Should be able to modify bound object."""
        class Obj:
            value = 0

        obj = Obj()
        shell.bind(obj)
        shell.execute("self.value = 999")
        assert obj.value == 999

    def test_rebind_replaces_previous(self):
        """Rebinding should replace the previous bound object."""
        obj1 = {"id": 1}
        obj2 = {"id": 2}

        shell.bind(obj1)
        shell.bind(obj2)

        assert shell.bound_object is obj2
        assert shell.namespace.get("self") is obj2

    def test_bound_object_survives_reset(self):
        """Bound object should be restored after namespace reset."""
        obj = {"persistent": True}
        shell.bind(obj)
        shell.reset_namespace()

        assert shell.namespace.get("self") is obj

    def test_unbind_after_reset(self):
        """Unbinding after reset should work correctly."""
        shell.bind({"x": 1})
        shell.reset_namespace()
        shell.unbind()

        assert "self" not in shell.namespace


class TestHistory:
    """Tests for command history management."""

    def test_history_records_input(self):
        """History should record executed commands."""
        shell.execute("1 + 1")
        shell.execute("2 + 2")
        assert len(shell.history) == 2
        assert "1 + 1" in shell.history
        assert "2 + 2" in shell.history

    def test_clear_history(self):
        """clear_history should empty the history."""
        shell.execute("x = 1")
        shell.execute("y = 2")
        shell.clear_history()
        assert len(shell.history) == 0

    def test_history_records_failed_commands(self):
        """History should record commands even if they fail."""
        shell.execute("invalid syntax here")
        assert len(shell.history) == 1

    def test_history_property_returns_copy(self):
        """history property should return a copy, not the original."""
        shell.execute("x = 1")
        history = shell.history
        history.append("should not affect original")
        assert "should not affect original" not in shell.history

    def test_save_and_load_history(self):
        """Should be able to save and load history."""
        shell.execute("a = 1")
        shell.execute("b = 2")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name

        try:
            shell.save_history(temp_path)
            shell.clear_history()
            assert len(shell.history) == 0

            shell.load_history(temp_path)
            assert "a = 1" in shell.history
            assert "b = 2" in shell.history
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_history_appends(self):
        """load_history should append to existing history."""
        shell.execute("existing")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("loaded_command\n")
            temp_path = f.name

        try:
            shell.load_history(temp_path)
            assert "existing" in shell.history
            assert "loaded_command" in shell.history
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """Loading nonexistent file should not raise."""
        shell.load_history("/nonexistent/path/to/file.txt")
        # Should complete without error


class TestOutputCapture:
    """Tests for stdout capture during execution."""

    def test_print_captured(self):
        """print output should be captured in result.output."""
        result = shell.execute("print('hello')")
        assert "hello" in result.output

    def test_multiple_prints_captured(self):
        """Multiple prints should all be captured."""
        result = shell.execute("print('a'); print('b'); print('c')")
        assert "a" in result.output
        assert "b" in result.output
        assert "c" in result.output

    def test_output_not_mixed_with_value(self):
        """Output and return value should be separate.

        Note: Multi-statement lines are treated as statements (exec),
        not expressions (eval), so value is None. We test with separate calls.
        """
        # First execute print (statement)
        result1 = shell.execute("print('output')")
        assert "output" in result1.output

        # Then execute expression (returns value)
        result2 = shell.execute("42")
        assert result2.value == 42
        assert result2.output == ""  # Expression has no print output

    def test_error_output_captured(self):
        """Output before error should still be captured."""
        result = shell.execute("print('before'); 1/0")
        assert "before" in result.output
        assert result.success is False


class TestExecutionResult:
    """Tests for the ExecutionResult dataclass."""

    def test_result_dataclass_success(self):
        """ExecutionResult should store success cases."""
        result = ExecutionResult(
            success=True,
            value=42,
            output="",
            error=None,
            error_type=None
        )
        assert result.success is True
        assert result.value == 42

    def test_result_dataclass_failure(self):
        """ExecutionResult should store failure cases."""
        result = ExecutionResult(
            success=False,
            value=None,
            output="",
            error="Division by zero",
            error_type="ZeroDivisionError"
        )
        assert result.success is False
        assert "zero" in result.error.lower()
        assert result.error_type == "ZeroDivisionError"

    def test_result_defaults(self):
        """ExecutionResult should have sensible defaults."""
        result = ExecutionResult(success=True)
        assert result.value is None
        assert result.output == ""
        assert result.error is None
        assert result.error_type is None


class TestShellInstances:
    """Tests for multiple Shell instances."""

    def test_new_shell_instance(self):
        """Creating a new Shell should work independently."""
        new_shell = Shell()
        assert isinstance(new_shell, Shell)
        assert new_shell is not shell

    def test_instances_have_separate_namespaces(self):
        """Different Shell instances should have separate namespaces."""
        shell1 = Shell()
        shell2 = Shell()

        shell1.execute("x = 1")
        shell2.execute("x = 2")

        assert shell1.namespace.get("x") == 1
        assert shell2.namespace.get("x") == 2

    def test_instances_have_separate_histories(self):
        """Different Shell instances should have separate histories."""
        shell1 = Shell()
        shell2 = Shell()

        shell1.execute("command1")
        shell2.execute("command2")

        assert "command1" in shell1.history
        assert "command1" not in shell2.history
        assert "command2" in shell2.history
        assert "command2" not in shell1.history


class TestConvenienceFunctions:
    """Tests for built-in convenience functions."""

    def test_inspect_function(self):
        """inspect() convenience function should work."""
        result = shell.execute("inspect({'test': 1})")
        assert result.success is True
        # Should return an InspectorPanel
        assert result.value is not None

    def test_types_function(self):
        """types() convenience function should work."""
        result = shell.execute("types()")
        assert result.success is True
        # Should return something iterable

    def test_copy_function(self):
        """copy() convenience function should work."""
        shell.execute("original = {'a': 1}")
        result = shell.execute("copy(original)")
        assert result.success is True
        assert result.value == {'a': 1}
        # Should be a different object
        assert result.value is not shell.namespace.get("original")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string_execution(self):
        """Empty string should execute without error."""
        result = shell.execute("")
        assert result.success is True

    def test_whitespace_only_execution(self):
        """Whitespace-only string should execute without error."""
        result = shell.execute("   \n\t   ")
        assert result.success is True

    def test_long_expression(self):
        """Long expressions should work."""
        expr = " + ".join(["1"] * 100)
        result = shell.execute(expr)
        assert result.success is True
        assert result.value == 100

    def test_unicode_support(self):
        """Unicode should work in code."""
        result = shell.execute("greeting = 'Hello'")
        assert result.success is True
        assert shell.namespace.get("greeting") == "Hello"

    def test_complex_numbers(self):
        """Complex numbers should work."""
        result = shell.execute("(1 + 2j) * (3 + 4j)")
        assert result.success is True
        assert result.value == (1 + 2j) * (3 + 4j)

    def test_generator_expression(self):
        """Generator expressions should work."""
        result = shell.execute("list(x*2 for x in range(5))")
        assert result.success is True
        assert result.value == [0, 2, 4, 6, 8]

    def test_lambda_execution(self):
        """Lambda expressions should work."""
        result = shell.execute("(lambda x: x * 2)(5)")
        assert result.success is True
        assert result.value == 10

    def test_list_comprehension(self):
        """List comprehensions should work."""
        result = shell.execute("[x**2 for x in range(5)]")
        assert result.success is True
        assert result.value == [0, 1, 4, 9, 16]

    def test_dict_comprehension(self):
        """Dict comprehensions should work."""
        result = shell.execute("{x: x**2 for x in range(3)}")
        assert result.success is True
        assert result.value == {0: 0, 1: 1, 2: 4}


class TestInteractionWithCoreSystems:
    """Tests for interaction with core foundation systems."""

    def test_use_mirror(self):
        """Should be able to use mirror from shell.

        Note: Multi-line code blocks are treated as statements (exec),
        so we need to execute in separate steps to get return values.
        """
        # Define class first (statement)
        shell.execute("""
class TestClass:
    value = 10
""")
        # Create mirror (statement)
        shell.execute("m = mirror(TestClass())")

        # Now get the value (expression)
        result = shell.execute("m.type_name")
        assert result.success is True
        assert result.value == "TestClass"

    def test_use_inspector(self):
        """Should be able to use inspector from shell."""
        result = shell.execute("inspector.inspect({'x': 1})")
        assert result.success is True

    def test_undo_redo_available(self):
        """undo and redo functions should be callable."""
        # Just test they exist and are callable
        assert callable(shell.namespace.get("undo"))
        assert callable(shell.namespace.get("redo"))
