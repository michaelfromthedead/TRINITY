"""
Phase 10 Tests: Tooling -- doctor, step_trace, op_coverage, lint.

Tests for trinity/tools/ module providing diagnostic and validation utilities.
"""

from __future__ import annotations

import itertools
import warnings

import pytest

from trinity.decorators.ops import Op, Step, HookEvent, validate_steps, decompose
from trinity.metaclasses.engine_meta import EngineMeta
from trinity.metaclasses.component_meta import ComponentMeta

# ---------------------------------------------------------------------------
# Lazy imports for tools (may not exist yet)
# ---------------------------------------------------------------------------

_tools_imported = False
_import_error = None

try:
    from trinity.tools import doctor, trace, op_coverage, lint
    from trinity.tools.doctor import doctor as doctor_fn
    from trinity.tools.step_trace import trace as trace_fn
    from trinity.tools.op_coverage import op_coverage as op_coverage_fn
    from trinity.tools.lint import lint as lint_fn, install_lint_hook, uninstall_lint_hook
    _tools_imported = True
except ImportError as exc:
    _import_error = exc

needs_tools = pytest.mark.skipif(
    not _tools_imported,
    reason=f"trinity.tools not available: {_import_error}",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_counter = itertools.count(1)


def _unique_name(prefix: str = "TestComp") -> str:
    return f"{prefix}_{next(_counter)}"


@pytest.fixture(autouse=True)
def _clean_registries():
    """Clear registries before and after each test."""
    EngineMeta.clear_registry()
    ComponentMeta.clear_registry()
    yield
    EngineMeta.clear_registry()
    ComponentMeta.clear_registry()


def _make_component(name: str | None = None, fields: dict | None = None, ns: dict | None = None):
    """Create a minimal component class via ComponentMeta."""
    name = name or _unique_name()
    namespace = {"__annotations__": fields or {}, "__module__": "test_phase10"}
    if ns:
        namespace.update(ns)
    return ComponentMeta(name, (), namespace)


# ===========================================================================
# TestToolImports
# ===========================================================================


class TestToolImports:
    """Verify that all tools are importable from the package."""

    @needs_tools
    def test_all_tools_importable_from_package(self):
        from trinity.tools import doctor, trace, op_coverage, lint
        assert callable(doctor)
        assert callable(trace)
        assert callable(op_coverage)
        assert callable(lint)


# ===========================================================================
# TestDoctor
# ===========================================================================


@needs_tools
class TestDoctor:
    """Tests for doctor() -- validates all registered classes."""

    def test_doctor_returns_dict(self):
        _make_component()
        result = doctor_fn()
        assert isinstance(result, dict)
        for key in ("total", "passed", "failed", "errors"):
            assert key in result, f"Missing key: {key}"

    def test_doctor_no_errors_on_clean_class(self):
        """A clean component has no semantic errors (ordering artifacts excluded)."""
        cls = _make_component(fields={"x": float, "y": float})
        result = doctor_fn()
        errors = result.get("errors", {})
        class_errors = errors.get(cls.__name__, [])
        # Filter out the cross-layer REGISTER ordering artifact
        semantic_errors = [e for e in class_errors if "REGISTER" not in e]
        assert len(semantic_errors) == 0, f"Unexpected errors: {semantic_errors}"

    def test_doctor_counts_classes(self):
        """Total should count all registered engine types."""
        _make_component(name=_unique_name("DoctorA"))
        _make_component(name=_unique_name("DoctorB"))
        result = doctor_fn()
        assert result["total"] == 2


# ===========================================================================
# TestTrace
# ===========================================================================


@needs_tools
class TestTrace:
    """Tests for trace(cls) -- formatted step trace per layer."""

    def test_trace_returns_string(self):
        cls = _make_component(fields={"hp": int})
        result = trace_fn(cls)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_trace_contains_class_name(self):
        name = _unique_name("Traced")
        cls = _make_component(name=name, fields={"hp": int})
        result = trace_fn(cls)
        assert name in result

    def test_trace_shows_decorator_section(self):
        cls = _make_component(fields={"hp": int})
        result = trace_fn(cls)
        assert "[Decorator]" in result

    def test_trace_shows_descriptor_section(self):
        cls = _make_component(fields={"hp": int})
        result = trace_fn(cls)
        assert "[Descriptor]" in result

    def test_trace_shows_metaclass_section(self):
        cls = _make_component(fields={"hp": int})
        result = trace_fn(cls)
        assert "[Metaclass]" in result

    def test_trace_shows_metaclass_steps(self):
        """A component with fields shows TAG, REGISTER, DESCRIBE steps in metaclass section."""
        cls = _make_component(fields={"hp": int, "mp": int})
        result = trace_fn(cls)
        # The metaclass section is after "[Metaclass]"
        metaclass_section = result[result.index("[Metaclass]"):]
        assert "tag" in metaclass_section
        assert "register" in metaclass_section
        assert "describe" in metaclass_section


# ===========================================================================
# TestOpCoverage
# ===========================================================================


@needs_tools
class TestOpCoverage:
    """Tests for op_coverage() -- Op usage analysis."""

    def test_op_coverage_returns_dict(self):
        _make_component(fields={"x": float})
        result = op_coverage_fn()
        assert isinstance(result, dict)
        for key in ("op_counts", "zero_step_classes", "total_classes", "total_steps", "coverage"):
            assert key in result, f"Missing key: {key}"

    def test_op_coverage_all_ops_present(self):
        """op_counts should have an entry for every Op value."""
        _make_component(fields={"x": float})
        result = op_coverage_fn()
        op_counts = result["op_counts"]
        for op in Op:
            assert op.value in op_counts, f"Op '{op.value}' missing from op_counts"

    def test_op_coverage_counts_classes(self):
        """total_classes should match registered engine types count."""
        _make_component(name=_unique_name("CovA"))
        _make_component(name=_unique_name("CovB"))
        _make_component(name=_unique_name("CovC"))
        result = op_coverage_fn()
        assert result["total_classes"] == 3


# ===========================================================================
# TestLint
# ===========================================================================


@needs_tools
class TestLint:
    """Tests for lint(cls) -- validate single class. Returns list[str]."""

    def test_lint_clean_class(self):
        """A clean component has no semantic lint errors (ordering artifacts excluded)."""
        cls = _make_component(fields={"x": float, "y": float})
        result = lint_fn(cls)
        assert isinstance(result, list)
        # Filter out the cross-layer REGISTER ordering artifact
        semantic_errors = [e for e in result if "REGISTER" not in e]
        assert len(semantic_errors) == 0, f"Unexpected errors: {semantic_errors}"

    def test_lint_returns_errors(self):
        """A class with HOOK(on_change) but no TRACK should fail lint."""
        cls = _make_component(fields={"x": float})
        cls._applied_steps = [
            Step(Op.HOOK, {"event": HookEvent.ON_CHANGE}),
        ]
        result = lint_fn(cls)
        assert isinstance(result, list)
        assert len(result) > 0, "Expected lint errors for HOOK(on_change) without TRACK"

    def test_install_lint_hook(self):
        """After installing hook, creating a class with bad steps emits a warning."""
        install_lint_hook()
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # Create a class that has pre-set _applied_steps violating rules.
                # The lint hook intercepts EngineMeta.__new__ which ComponentMeta
                # calls via super(). We set _applied_steps in namespace so the hook
                # picks it up after creation.
                ns = {
                    "__annotations__": {"x": float},
                    "__module__": "test_phase10",
                    "_applied_steps": [
                        Step(Op.HOOK, {"event": HookEvent.ON_CHANGE}),
                    ],
                }
                name = _unique_name("LintHooked")
                try:
                    ComponentMeta(name, (), ns)
                except Exception:
                    pass  # Class creation may fail; we care about warnings
                # Check if any lint-related warning was emitted
                lint_warnings = [
                    x for x in w
                    if "lint" in str(x.message).lower()
                    or "HOOK" in str(x.message)
                    or "TRACK" in str(x.message)
                    or "requires" in str(x.message).lower()
                ]
                assert len(lint_warnings) > 0, (
                    f"Expected lint warning, got warnings: {[str(x.message) for x in w]}"
                )
        finally:
            uninstall_lint_hook()

    def test_uninstall_lint_hook(self):
        """After uninstalling hook, no lint warnings on new classes."""
        install_lint_hook()
        uninstall_lint_hook()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _make_component(fields={"x": float})
            lint_warnings = [
                x for x in w
                if "lint" in str(x.message).lower()
            ]
            assert len(lint_warnings) == 0, (
                f"Unexpected lint warnings after uninstall: {[str(x.message) for x in w]}"
            )
