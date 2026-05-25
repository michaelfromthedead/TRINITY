import pytest
from trinity.decorators.registry import registry, StackSpec
from trinity.decorators.stacks import Stack, stack, parameterized_stack, _noop


@pytest.fixture(autouse=True)
def _clean_stacks():
    """Snapshot and restore the stacks dict so tests don't leak state."""
    saved = dict(registry._stacks)
    yield
    registry._stacks.clear()
    registry._stacks.update(saved)


# ---------------------------------------------------------------------------
# StackSpec dataclass
# ---------------------------------------------------------------------------

class TestStackSpec:
    def test_dataclass_fields(self):
        spec = StackSpec(
            name="test",
            stack_fn=lambda: None,
            decorators=["a", "b"],
            domain="test",
            doc="some doc",
            parameterized=False,
        )
        assert spec.name == "test"
        assert spec.domain == "test"
        assert spec.parameterized is False
        assert spec.decorators == ["a", "b"]
        assert spec.doc == "some doc"


# ---------------------------------------------------------------------------
# Stack registration
# ---------------------------------------------------------------------------

class TestStackRegistry:
    def test_register_and_retrieve(self):
        def my_stack():
            return stack(_noop, name="my")

        registry.register_stack("reg_test_1", my_stack, domain="test", doc="A test stack")
        spec = registry.get_stack("reg_test_1")
        assert spec.name == "reg_test_1"
        assert spec.domain == "test"
        assert spec.doc == "A test stack"
        assert spec.parameterized is False
        # _noop decorator should appear in the decorators list
        assert "_noop" in spec.decorators

    def test_get_stack_not_found(self):
        with pytest.raises(KeyError, match="nonexistent_stack_xyz"):
            registry.get_stack("nonexistent_stack_xyz")

    def test_duplicate_registration_raises(self):
        def s():
            return stack()

        registry.register_stack("dup_test", s, domain="test")
        with pytest.raises(ValueError, match="already registered"):
            registry.register_stack("dup_test", s, domain="test")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            registry.register_stack("", lambda: stack(), domain="test")

    def test_whitespace_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            registry.register_stack("   ", lambda: stack(), domain="test")

    def test_all_stacks_contains_registered(self):
        def s():
            return stack()

        registry.register_stack("all_test_1", s, domain="d1")
        registry.register_stack("all_test_2", s, domain="d2")
        all_s = registry.all_stacks()
        names = [sp.name for sp in all_s]
        assert "all_test_1" in names
        assert "all_test_2" in names

    def test_expand_stack_returns_decorator_names(self):
        def simple():
            return stack(_noop, _noop, name="double_noop")

        registry.register_stack("expand_test_1", simple, domain="test")
        result = registry.expand_stack("expand_test_1")
        assert result == ["_noop", "_noop"]

    def test_expand_stack_empty_when_no_decorators(self):
        def empty():
            return stack()

        registry.register_stack("expand_empty", empty, domain="test")
        assert registry.expand_stack("expand_empty") == []

    def test_parameterized_detection(self):
        @parameterized_stack
        def my_param(x: int = 1) -> Stack:
            return stack(_noop)

        registry.register_stack("param_det_1", my_param, domain="test")
        spec = registry.get_stack("param_det_1")
        assert spec.parameterized is True
        # Parameterized stacks don't eagerly resolve decorators
        assert spec.decorators == []

    def test_non_parameterized_requiring_args_becomes_parameterized(self):
        """A stack_fn that requires args but lacks the marker gets auto-detected."""
        def needs_args(n: int):
            return stack(*([_noop] * n))

        registry.register_stack("auto_param_1", needs_args, domain="test")
        spec = registry.get_stack("auto_param_1")
        # Should have been promoted to parameterized because calling with no args raises TypeError
        assert spec.parameterized is True

    def test_doc_falls_back_to_docstring(self):
        def documented():
            """My docstring."""
            return stack()

        registry.register_stack("doc_test_1", documented, domain="test")
        spec = registry.get_stack("doc_test_1")
        assert spec.doc == "My docstring."
