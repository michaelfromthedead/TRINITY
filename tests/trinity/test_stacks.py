"""Comprehensive tests for trinity.decorators.stacks module."""

import pytest

from trinity.decorators.stacks import Stack, stack, parameterized_stack, _noop


# --- Test decorators ---

def add_marker_a(cls):
    cls._marker_a = True
    return cls


def add_marker_b(cls):
    cls._marker_b = True
    return cls


def append_order(label):
    """Factory: decorator that appends label to cls._order."""
    def decorator(cls):
        if not hasattr(cls, "_order"):
            cls._order = []
        cls._order.append(label)
        return cls
    decorator.__name__ = f"append_{label}"
    decorator.__qualname__ = f"append_{label}"
    return decorator


# =============================================================================
# TestStack
# =============================================================================

class TestStack:
    def test_applies_decorators_in_reverse_order(self):
        d1 = append_order("first")
        d2 = append_order("second")
        d3 = append_order("third")
        s = Stack(d1, d2, d3)

        @s
        class C:
            pass

        # Reverse order means d3 applied first, then d2, then d1
        assert C._order == ["third", "second", "first"]

    def test_single_decorator(self):
        s = Stack(add_marker_a)

        @s
        class C:
            pass

        assert C._marker_a is True

    def test_empty_stack_is_identity(self):
        s = Stack()

        class Original:
            pass

        result = s(Original)
        assert result is Original

    def test_expand_returns_names(self):
        s = Stack(add_marker_a, add_marker_b, name="test")
        names = s.expand()
        assert names == ["add_marker_a", "add_marker_b"]

    def test_decorators_property(self):
        s = Stack(add_marker_a, add_marker_b)
        assert isinstance(s.decorators, tuple)
        assert s.decorators == (add_marker_a, add_marker_b)

    def test_repr_shows_name_and_count(self):
        s = Stack(add_marker_a, add_marker_b, name="my_stack")
        r = repr(s)
        assert "my_stack" in r
        assert "2" in r

    def test_len_returns_count(self):
        s = Stack(add_marker_a, add_marker_b, add_marker_a)
        assert len(s) == 3

    def test_add_composes_stacks(self):
        s1 = Stack(add_marker_a)
        s2 = Stack(add_marker_b)
        combined = s1 + s2
        assert isinstance(combined, Stack)
        assert len(combined) == 2

        @combined
        class C:
            pass

        assert C._marker_a is True
        assert C._marker_b is True

    def test_add_preserves_order(self):
        d1 = append_order("a")
        d2 = append_order("b")
        d3 = append_order("c")
        s1 = Stack(d1, d2)
        s2 = Stack(d3)
        combined = s1 + s2

        @combined
        class C:
            pass

        # s1 decorators come before s2, applied in reverse: c, b, a
        assert C._order == ["c", "b", "a"]

    def test_name_default(self):
        s = Stack(add_marker_a, add_marker_b)
        r = repr(s)
        assert r == "Stack(2 decorators)"

    def test_add_non_stack_raises_type_error(self):
        s = Stack(add_marker_a)
        with pytest.raises(TypeError):
            s + "not a stack"

    def test_call_decorator_returns_none_raises(self):
        def bad_decorator(cls):
            return None  # forgot to return cls

        s = Stack(bad_decorator)
        with pytest.raises(TypeError, match="returned None"):
            s(type("Dummy", (), {}))

    def test_expand_with_lambda(self):
        lam = lambda cls: cls  # noqa: E731
        s = Stack(lam)
        names = s.expand()
        assert len(names) == 1
        # Lambda has __name__ == "<lambda>"
        assert names[0] == "<lambda>"

    def test_add_combined_name(self):
        s1 = Stack(add_marker_a, name="alpha")
        s2 = Stack(add_marker_b, name="beta")
        combined = s1 + s2
        assert repr(combined) == "alpha+beta(2 decorators)"

    def test_add_combined_name_one_missing(self):
        s1 = Stack(add_marker_a, name="alpha")
        s2 = Stack(add_marker_b)
        combined = s1 + s2
        assert repr(combined) == "Stack(2 decorators)"


# =============================================================================
# TestStackFunction
# =============================================================================

class TestStackFunction:
    def test_creates_stack(self):
        s = stack(add_marker_a, add_marker_b)
        assert isinstance(s, Stack)

    def test_passes_name(self):
        s = stack(add_marker_a, name="foo")
        assert "foo" in repr(s)


# =============================================================================
# TestParameterizedStack
# =============================================================================

class TestParameterizedStack:
    def test_basic_parameterized_stack(self):
        @parameterized_stack
        def my_stack():
            return stack(add_marker_a, add_marker_b, name="my_stack")

        result = my_stack()
        assert isinstance(result, Stack)

    def test_validates_return_type(self):
        @parameterized_stack
        def bad_stack():
            return "not a stack"

        with pytest.raises(TypeError, match="bad_stack must return a Stack, got str"):
            bad_stack()

    def test_sets_flag(self):
        @parameterized_stack
        def my_stack():
            return stack(add_marker_a, name="s")

        assert my_stack._is_parameterized_stack is True

    def test_preserves_docstring(self):
        @parameterized_stack
        def my_stack():
            """My docstring."""
            return stack(add_marker_a, name="s")

        assert my_stack.__doc__ == "My docstring."

    def test_with_parameters(self):
        @parameterized_stack
        def my_stack(use_b=False):
            decorators = [add_marker_a]
            if use_b:
                decorators.append(add_marker_b)
            return stack(*decorators, name="param")

        s1 = my_stack()
        assert len(s1) == 1

        s2 = my_stack(use_b=True)
        assert len(s2) == 2

        @s2
        class C:
            pass

        assert C._marker_a is True
        assert C._marker_b is True


# =============================================================================
# TestNoop
# =============================================================================

class TestNoop:
    def test_noop_returns_cls_unchanged(self):
        class MyClass:
            pass

        result = _noop(MyClass)
        assert result is MyClass

    def test_noop_returns_function_unchanged(self):
        def my_func():
            return 42

        result = _noop(my_func)
        assert result is my_func
        assert result() == 42

    def test_noop_in_stack_is_identity(self):
        s = Stack(_noop, _noop, _noop)

        class Original:
            x = 1

        result = s(Original)
        assert result is Original
        assert result.x == 1
