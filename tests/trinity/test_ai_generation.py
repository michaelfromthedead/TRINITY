"""Tests for AI generation decorators (Tier 9)."""

from __future__ import annotations

import pytest

from trinity.decorators.ai_generation import (
    VALID_PATTERN_CATEGORIES,
    complexity,
    constraints,
    example,
    generates,
    pattern,
    pure,
    stub,
)
from trinity.decorators.ops import Op, decompose, expand


# ============================================================================
# TestExample
# ============================================================================


class TestExample:
    """Tests for @example decorator."""

    def test_single_example_stored(self):
        """Single example should be stored in _examples list."""

        @example(inputs={"x": 1, "y": 2}, output=3, description="Simple case")
        def add(x, y):
            return x + y

        assert hasattr(add, "_examples")
        assert len(add._examples) == 1
        assert add._examples[0] == {
            "inputs": {"x": 1, "y": 2},
            "output": 3,
            "description": "Simple case",
        }

    def test_multiple_examples_accumulate(self):
        """Multiple @example decorators should accumulate.

        Note: Decorators are applied bottom-to-top, so Third is added first.
        """

        @example(inputs={"x": 1, "y": 2}, output=3, description="First")
        @example(inputs={"x": 0, "y": 0}, output=0, description="Second")
        @example(inputs={"x": -1, "y": 1}, output=0, description="Third")
        def add(x, y):
            return x + y

        assert len(add._examples) == 3
        # Decorators applied bottom-to-top: Third, Second, First
        assert add._examples[0]["description"] == "Third"
        assert add._examples[1]["description"] == "Second"
        assert add._examples[2]["description"] == "First"

    def test_inputs_output_description_stored(self):
        """All example parameters should be stored correctly."""

        @example(
            inputs={"data": [1, 2, 3]},
            output=6,
            description="Sum of list",
        )
        def sum_list(data):
            return sum(data)

        ex = sum_list._examples[0]
        assert ex["inputs"] == {"data": [1, 2, 3]}
        assert ex["output"] == 6
        assert ex["description"] == "Sum of list"

    def test_example_default_description(self):
        """Description should default to empty string."""

        @example(inputs={"x": 1}, output=1)
        def identity(x):
            return x

        assert identity._examples[0]["description"] == ""

    def test_missing_inputs_raises_error(self):
        """Missing inputs parameter should raise ValueError."""
        with pytest.raises(ValueError, match="inputs must be a dict"):

            @example(output=1)
            def func():
                pass

    def test_missing_output_raises_error(self):
        """Missing output parameter should raise ValueError."""
        with pytest.raises(ValueError, match="output is required"):

            @example(inputs={"x": 1})
            def func():
                pass

    def test_non_dict_inputs_raises_error(self):
        """Non-dict inputs should raise ValueError."""
        with pytest.raises(ValueError, match="inputs must be a dict"):

            @example(inputs=[1, 2], output=3)  # type: ignore
            def func():
                pass


# ============================================================================
# TestConstraints
# ============================================================================


class TestConstraints:
    """Tests for @constraints decorator."""

    def test_rules_list_stored(self):
        """Rules list should be stored in _constraints."""

        @constraints(rules=["must handle None", "no side effects"])
        def process(data):
            pass

        assert hasattr(process, "_constraints")
        assert process._constraints == ["must handle None", "no side effects"]

    def test_accumulation_when_stacked(self):
        """Multiple @constraints should accumulate rules."""

        @constraints(rules=["rule1", "rule2"])
        @constraints(rules=["rule3"])
        def func():
            pass

        assert len(func._constraints) == 3
        assert "rule1" in func._constraints
        assert "rule2" in func._constraints
        assert "rule3" in func._constraints

    def test_empty_rules_raises_error(self):
        """Empty rules list should raise ValueError."""
        with pytest.raises(ValueError, match="rules must be a non-empty list"):

            @constraints(rules=[])
            def func():
                pass

    def test_non_list_rules_raises_error(self):
        """Non-list rules should raise ValueError."""
        with pytest.raises(ValueError, match="rules must be a non-empty list"):

            @constraints(rules="not a list")  # type: ignore
            def func():
                pass


# ============================================================================
# TestStub
# ============================================================================


class TestStub:
    """Tests for @stub decorator."""

    def test_hints_stored(self):
        """Implementation hints should be stored."""

        @stub(
            signature_only=False,
            implementation_hints=["use binary search", "handle empty list"],
        )
        def find_item(items, target):
            pass

        assert find_item._stub is True
        assert find_item._stub_signature_only is False
        assert find_item._stub_hints == ["use binary search", "handle empty list"]

    def test_signature_only_flag_default(self):
        """signature_only should default to True."""

        @stub()
        def func():
            pass

        assert func._stub_signature_only is True
        assert func._stub_hints == []

    def test_signature_only_flag_custom(self):
        """signature_only can be set to False."""

        @stub(signature_only=False)
        def func():
            pass

        assert func._stub_signature_only is False

    def test_stub_marker_set(self):
        """_stub marker should always be set."""

        @stub()
        def func():
            pass

        assert func._stub is True


# ============================================================================
# TestPattern
# ============================================================================


class TestPattern:
    """Tests for @pattern decorator."""

    def test_category_validation_invalid_raises(self):
        """Invalid category should raise ValueError."""
        with pytest.raises(ValueError, match="category must be one of"):

            @pattern(name="Singleton", category="invalid")
            class MyClass:
                pass

    def test_name_category_stored(self):
        """Name and category should be stored."""

        @pattern(name="Observer", category="behavioral")
        class EventSystem:
            pass

        assert EventSystem._pattern is True
        assert EventSystem._pattern_name == "Observer"
        assert EventSystem._pattern_category == "behavioral"

    def test_empty_name_raises_error(self):
        """Empty name should raise ValueError."""
        with pytest.raises(ValueError, match="name must be non-empty"):

            @pattern(name="", category="creational")
            class MyClass:
                pass

    def test_all_valid_categories(self):
        """All valid categories should work."""
        for category in VALID_PATTERN_CATEGORIES:

            @pattern(name="TestPattern", category=category)
            class TestClass:
                pass

            assert TestClass._pattern_category == category


# ============================================================================
# TestComplexity
# ============================================================================


class TestComplexity:
    """Tests for @complexity decorator."""

    def test_time_space_strings_stored(self):
        """Time and space complexity should be stored."""

        @complexity(time="O(n log n)", space="O(n)")
        def merge_sort(items):
            pass

        assert merge_sort._complexity is True
        assert merge_sort._complexity_time == "O(n log n)"
        assert merge_sort._complexity_space == "O(n)"

    def test_empty_time_raises_error(self):
        """Empty time string should raise ValueError."""
        with pytest.raises(ValueError, match="time must be a non-empty string"):

            @complexity(time="", space="O(1)")
            def func():
                pass

    def test_empty_space_raises_error(self):
        """Empty space string should raise ValueError."""
        with pytest.raises(ValueError, match="space must be a non-empty string"):

            @complexity(time="O(1)", space="")
            def func():
                pass

    def test_complexity_marker_set(self):
        """_complexity marker should be set."""

        @complexity(time="O(1)", space="O(1)")
        def func():
            pass

        assert func._complexity is True


# ============================================================================
# TestGenerates
# ============================================================================


class TestGenerates:
    """Tests for @generates decorator."""

    def test_output_type_stored(self):
        """output_type should be stored."""

        @generates(output_type=str, count=5)
        def generate_strings():
            pass

        assert generate_strings._generates is True
        assert generate_strings._generates_output_type is str

    def test_count_default_is_one(self):
        """count should default to 1."""

        @generates(output_type=int)
        def generate_number():
            pass

        assert generate_number._generates_count == 1

    def test_count_many(self):
        """count can be 'many'."""

        @generates(output_type=str, count="many")
        def generate_names():
            pass

        assert generate_names._generates_count == "many"

    def test_invalid_count_raises_error(self):
        """Invalid count should raise ValueError."""
        with pytest.raises(ValueError, match="count must be a positive int or 'many'"):

            @generates(output_type=str, count="invalid")  # type: ignore
            def func():
                pass

    def test_negative_count_raises_error(self):
        """Negative count should raise ValueError."""
        with pytest.raises(ValueError, match="count must be positive"):

            @generates(output_type=str, count=-1)
            def func():
                pass

    def test_zero_count_raises_error(self):
        """Zero count should raise ValueError."""
        with pytest.raises(ValueError, match="count must be positive"):

            @generates(output_type=str, count=0)
            def func():
                pass

    def test_generates_marker_set(self):
        """_generates marker should be set."""

        @generates(output_type=str)
        def func():
            pass

        assert func._generates is True

    def test_missing_output_type_raises_error(self):
        """Missing output_type parameter should raise ValueError."""
        with pytest.raises(ValueError, match="output_type is required"):

            @generates(count=5)  # type: ignore
            def func():
                pass


# ============================================================================
# TestPure
# ============================================================================


class TestPure:
    """Tests for @pure decorator."""

    def test_marker_works_with_parens(self):
        """@pure() with parentheses should work."""

        @pure()
        def add(x, y):
            return x + y

        assert add._pure is True

    def test_marker_set(self):
        """_pure marker should be set to True."""

        @pure()
        def identity(x):
            return x

        assert identity._pure is True

    def test_pure_on_class(self):
        """@pure should work on classes."""

        @pure()
        class PureData:
            pass

        assert PureData._pure is True


# ============================================================================
# TestAiGenerationIntrospection
# ============================================================================


class TestAiGenerationIntrospection:
    """Tests for introspection of AI generation decorators."""

    @pytest.mark.parametrize(
        "decorator_func",
        [example, constraints, stub, pattern, complexity, generates, pure],
    )
    def test_decompose_all_decorators(self, decorator_func):
        """All AI generation decorators should decompose correctly."""
        steps = decompose(decorator_func)

        assert isinstance(steps, list)
        # For decorators with callable steps builders, decompose returns empty
        # until applied. This is expected behavior.

    @pytest.mark.parametrize(
        "decorator_func",
        [example, constraints, stub, pattern, complexity, generates, pure],
    )
    def test_expand_all_decorators(self, decorator_func):
        """All AI generation decorators should expand correctly."""
        expanded = expand(decorator_func)

        assert isinstance(expanded, str)

    @pytest.mark.parametrize(
        "decorator_func,params",
        [
            (example, {"inputs": {"x": 1}, "output": 1, "description": "test"}),
            (constraints, {"rules": ["rule1"]}),
            (stub, {"signature_only": True, "implementation_hints": []}),
            (pattern, {"name": "Singleton", "category": "creational"}),
            (complexity, {"time": "O(1)", "space": "O(1)"}),
            (generates, {"output_type": str, "count": 1}),
            (pure, {}),
        ],
    )
    def test_all_have_register_ai_generation(self, decorator_func, params):
        """All AI generation decorators should have REGISTER(ai_generation) step when applied."""

        @decorator_func(**params)
        def func():
            pass

        # Check that the function has been registered
        assert hasattr(func, "_registries")
        assert "ai_generation" in func._registries


# ============================================================================
# Integration Tests
# ============================================================================


class TestAiGenerationIntegration:
    """Integration tests for AI generation decorators."""

    def test_multiple_decorators_on_same_function(self):
        """Multiple AI generation decorators should work together."""

        @pure()
        @complexity(time="O(n)", space="O(1)")
        @example(inputs={"items": [1, 2, 3]}, output=6)
        def sum_items(items):
            return sum(items)

        assert sum_items._pure is True
        assert sum_items._complexity is True
        assert sum_items._complexity_time == "O(n)"
        assert len(sum_items._examples) == 1

    def test_stub_with_pattern(self):
        """@stub and @pattern should work together."""

        @pattern(name="Factory", category="creational")
        @stub(implementation_hints=["use registry pattern"])
        class WidgetFactory:
            pass

        assert WidgetFactory._pattern is True
        assert WidgetFactory._stub is True
        assert "use registry pattern" in WidgetFactory._stub_hints

    def test_generates_with_complexity(self):
        """@generates and @complexity should work together."""

        @complexity(time="O(n)", space="O(n)")
        @generates(output_type=list, count="many")
        def generate_permutations(items):
            pass

        assert generate_permutations._complexity is True
        assert generate_permutations._generates is True
        assert generate_permutations._generates_count == "many"
