"""
Tests for CompilationResult bridge deserialization (T-FG-7.7).

Tests the wiring between Python CompilationResult and Rust frame graph
compilation output, verifying that all fields are correctly populated
from the bridge JSON.
"""

import pytest

from engine.rendering.framegraph import CompilationResult, CompileError


class TestCompileError:
    """Test CompileError dataclass."""

    def test_compile_error_default_values(self):
        """Test that CompileError has sensible defaults."""
        error = CompileError()

        assert error.pass_name == ""
        assert error.phase == ""
        assert error.message == ""

    def test_compile_error_with_values(self):
        """Test CompileError with explicit values."""
        error = CompileError(
            pass_name="GBuffer",
            phase="barrier_insertion",
            message="Resource not in expected state",
        )

        assert error.pass_name == "GBuffer"
        assert error.phase == "barrier_insertion"
        assert error.message == "Resource not in expected state"

    def test_compile_error_equality(self):
        """Test CompileError dataclass equality."""
        error1 = CompileError(
            pass_name="Pass1",
            phase="validation",
            message="Test error",
        )
        error2 = CompileError(
            pass_name="Pass1",
            phase="validation",
            message="Test error",
        )

        assert error1 == error2


class TestCompilationResultFields:
    """Test CompilationResult T-FG-7.7 fields."""

    def test_memory_savings_percent_default(self):
        """Test memory_savings_percent defaults to 0.0."""
        result = CompilationResult()

        assert result.memory_savings_percent == 0.0

    def test_memory_savings_percent_explicit(self):
        """Test setting memory_savings_percent explicitly."""
        result = CompilationResult(memory_savings_percent=42.5)

        assert result.memory_savings_percent == 42.5

    def test_errors_default_empty(self):
        """Test errors list defaults to empty."""
        result = CompilationResult()

        assert result.errors == []
        assert isinstance(result.errors, list)

    def test_errors_explicit(self):
        """Test setting errors list explicitly."""
        errors = [
            CompileError(
                pass_name="PassA",
                phase="culling",
                message="Orphaned resource",
            ),
        ]
        result = CompilationResult(errors=errors)

        assert len(result.errors) == 1
        assert result.errors[0].pass_name == "PassA"

    def test_pass_count_and_culled_count(self):
        """Test pass_count and culled_count fields."""
        result = CompilationResult(
            pass_count=10,
            culled_count=3,
        )

        assert result.pass_count == 10
        assert result.culled_count == 3


class TestFromBridgeJson:
    """Test CompilationResult.from_bridge_json() deserialization."""

    def test_from_bridge_json_minimal(self):
        """Test parsing minimal bridge JSON output."""
        data = {
            "passes": [],
            "barriers": [],
            "async_passes": [],
            "cull_stats": {},
            "validation": {"valid": True},
        }

        result = CompilationResult.from_bridge_json(data)

        assert result.success is True
        assert result.execution_order == []
        assert result.barrier_count == 0
        assert result.async_pass_count == 0
        assert result.memory_savings_percent == 0.0
        assert result.errors == []

    def test_from_bridge_json_with_passes(self):
        """Test parsing bridge JSON with pass list."""
        data = {
            "passes": [
                {"name": "GBuffer"},
                {"name": "Lighting"},
                {"name": "PostProcess"},
            ],
            "barriers": [{"id": 1}, {"id": 2}],
            "async_passes": [{"name": "AsyncCompute"}],
            "cull_stats": {
                "passes_total": 5,
                "culled_pass_count": 2,
            },
            "validation": {"valid": True},
        }

        result = CompilationResult.from_bridge_json(data)

        assert result.success is True
        assert result.execution_order == ["GBuffer", "Lighting", "PostProcess"]
        assert result.barrier_count == 2
        assert result.async_pass_count == 1
        assert result.pass_count == 5
        assert result.culled_count == 2

    def test_from_bridge_json_with_memory_savings(self):
        """Test parsing memory_savings_percent from cull_stats."""
        data = {
            "passes": [{"name": "Pass1"}],
            "barriers": [],
            "async_passes": [],
            "cull_stats": {
                "passes_total": 3,
                "passes_eliminated": 1,
                "memory_savings_percent": 35.7,
            },
            "validation": {"valid": True},
        }

        result = CompilationResult.from_bridge_json(data)

        assert result.memory_savings_percent == 35.7
        # Also test alternate key name
        assert result.culled_count == 1

    def test_from_bridge_json_with_errors(self):
        """Test parsing errors list from bridge JSON."""
        data = {
            "passes": [{"name": "GBuffer"}],
            "barriers": [],
            "async_passes": [],
            "cull_stats": {},
            "validation": {"valid": False},
            "errors": [
                {
                    "pass_name": "GBuffer",
                    "phase": "resource_validation",
                    "message": "Resource 'albedo' not found",
                },
                {
                    "pass_name": "Lighting",
                    "phase": "barrier_insertion",
                    "message": "Invalid state transition",
                },
            ],
        }

        result = CompilationResult.from_bridge_json(data)

        assert result.success is False
        assert len(result.errors) == 2

        assert result.errors[0].pass_name == "GBuffer"
        assert result.errors[0].phase == "resource_validation"
        assert result.errors[0].message == "Resource 'albedo' not found"

        assert result.errors[1].pass_name == "Lighting"
        assert result.errors[1].phase == "barrier_insertion"

    def test_from_bridge_json_forward_compatible(self):
        """Test that missing T-FG-7.7 fields use defaults (forward compat)."""
        # Simulate older bridge output without T-FG-7.7 fields
        data = {
            "passes": [{"name": "Pass1"}],
            "barriers": [],
            "async_passes": [],
            "cull_stats": {
                "passes_total": 1,
            },
            "validation": {"valid": True},
            # No 'errors' key, no 'memory_savings_percent'
        }

        result = CompilationResult.from_bridge_json(data)

        assert result.success is True
        assert result.memory_savings_percent == 0.0
        assert result.errors == []

    def test_from_bridge_json_malformed_error_skipped(self):
        """Test that malformed error entries are handled gracefully."""
        data = {
            "passes": [],
            "barriers": [],
            "async_passes": [],
            "cull_stats": {},
            "validation": {"valid": True},
            "errors": [
                {"pass_name": "Valid", "phase": "test", "message": "OK"},
                "not a dict",  # Should be skipped
                123,  # Should be skipped
                {"pass_name": "AlsoValid"},  # Missing keys get defaults
            ],
        }

        result = CompilationResult.from_bridge_json(data)

        # Only dict entries are parsed
        assert len(result.errors) == 2
        assert result.errors[0].pass_name == "Valid"
        assert result.errors[1].pass_name == "AlsoValid"
        assert result.errors[1].phase == ""  # Default
        assert result.errors[1].message == ""  # Default

    def test_from_bridge_json_validation_not_dict(self):
        """Test handling when validation is not a dict."""
        data = {
            "passes": [],
            "barriers": [],
            "async_passes": [],
            "cull_stats": {},
            "validation": True,  # Not a dict
        }

        result = CompilationResult.from_bridge_json(data)

        # Falls back to True
        assert result.success is True

    def test_from_bridge_json_complete_realistic(self):
        """Test realistic complete bridge output."""
        data = {
            "passes": [
                {"name": "GBuffer", "pass_type": "Graphics"},
                {"name": "Lighting", "pass_type": "Compute"},
                {"name": "PostProcess", "pass_type": "Graphics"},
            ],
            "barriers": [
                {"resource": "albedo", "before": "RENDER_TARGET", "after": "SHADER_RESOURCE"},
                {"resource": "hdr", "before": "UAV", "after": "SHADER_RESOURCE"},
            ],
            "async_passes": [],
            "cull_stats": {
                "passes_total": 5,
                "culled_pass_count": 2,
                "memory_savings_percent": 18.3,
            },
            "validation": {"valid": True},
            "errors": [],
        }

        result = CompilationResult.from_bridge_json(data)

        assert result.success is True
        assert result.execution_order == ["GBuffer", "Lighting", "PostProcess"]
        assert result.barrier_count == 2
        assert result.pass_count == 5
        assert result.culled_count == 2
        assert result.memory_savings_percent == 18.3
        assert result.errors == []


class TestFromBridgeJsonEdgeCases:
    """Edge case tests for from_bridge_json."""

    def test_empty_dict(self):
        """Test parsing empty dict."""
        result = CompilationResult.from_bridge_json({})

        assert result.success is True
        assert result.execution_order == []
        assert result.memory_savings_percent == 0.0
        assert result.errors == []

    def test_passes_with_missing_name(self):
        """Test passes without name key are filtered out."""
        data = {
            "passes": [
                {"name": "Valid"},
                {"no_name": "Invalid"},
                {"name": "AlsoValid"},
            ],
            "barriers": [],
            "async_passes": [],
            "cull_stats": {},
            "validation": {"valid": True},
        }

        result = CompilationResult.from_bridge_json(data)

        assert result.execution_order == ["Valid", "AlsoValid"]

    def test_high_memory_savings(self):
        """Test memory savings at upper bound."""
        data = {
            "passes": [],
            "barriers": [],
            "async_passes": [],
            "cull_stats": {
                "memory_savings_percent": 100.0,
            },
            "validation": {"valid": True},
        }

        result = CompilationResult.from_bridge_json(data)

        assert result.memory_savings_percent == 100.0

    def test_many_errors(self):
        """Test parsing many errors."""
        errors = [
            {
                "pass_name": f"Pass{i}",
                "phase": "validation",
                "message": f"Error {i}",
            }
            for i in range(10)
        ]

        data = {
            "passes": [],
            "barriers": [],
            "async_passes": [],
            "cull_stats": {},
            "validation": {"valid": False},
            "errors": errors,
        }

        result = CompilationResult.from_bridge_json(data)

        assert len(result.errors) == 10
        assert result.errors[5].pass_name == "Pass5"
        assert result.errors[5].message == "Error 5"


class TestCompilationResultIntegration:
    """Integration tests for CompilationResult with FrameGraph."""

    def test_python_fallback_has_default_memory_savings(self):
        """Test Python fallback path leaves memory_savings_percent at 0."""
        from engine.rendering.framegraph import FrameGraph, PassFlags

        fg = FrameGraph()
        fg.add_graphics_pass("Test").set_flag(PassFlags.SIDE_EFFECTS)

        result = fg.compile()

        # Python path doesn't compute memory savings
        assert result.memory_savings_percent == 0.0
        assert result.errors == []

    def test_python_fallback_has_pass_count(self):
        """Test Python fallback populates pass_count and culled_count.

        This test forces the Python fallback by mocking _omega to be unavailable.
        """
        from unittest.mock import patch
        from engine.rendering.framegraph import FrameGraph, ResourceState

        fg = FrameGraph()

        # Create passes - one will be culled
        tex1 = fg.create_texture("tex1")
        tex2 = fg.create_texture("tex2")
        backbuffer = fg.import_external("bb", None, is_backbuffer=True)

        pass1 = fg.add_graphics_pass("Used")
        pass1.add_color_attachment(tex1)

        pass2 = fg.add_graphics_pass("AlsoUsed")
        pass2.read(tex1, ResourceState.SHADER_RESOURCE)
        pass2.add_color_attachment(backbuffer)

        pass3 = fg.add_graphics_pass("Unused")
        pass3.add_color_attachment(tex2)  # tex2 never read

        # Force Python fallback by making _omega unavailable
        with patch.dict("sys.modules", {"_omega": None}):
            # Need to also patch the import inside the method
            with patch.object(
                fg, "_try_compile_via_rust", return_value=None
            ):
                result = fg.compile()

        assert result.pass_count == 3
        assert result.culled_count == 1
        assert "Unused" in result.culled_passes
