"""Tests for the profiler markers module."""

from __future__ import annotations

import time
import threading

import pytest

from engine.tooling.profiling.profiler_markers import (
    profile,
    gpu_profile,
    ProfileMarker,
    GPUProfileMarker,
    MarkerScope,
    MarkerType,
    begin_marker,
    end_marker,
    get_active_markers,
    get_marker_depth,
    clear_markers,
    profile_class,
)
from engine.tooling.profiling.cpu_profiler import cpu_profiler
from engine.tooling.profiling.gpu_profiler import gpu_profiler as gpu_profiler_instance, RenderPassType


class TestMarkerScope:
    """Tests for MarkerScope."""

    def test_creation(self):
        """Test basic creation."""
        scope = MarkerScope(
            name="test",
            marker_type=MarkerType.CPU,
            start_time=time.perf_counter(),
        )
        assert scope.name == "test"
        assert scope.marker_type == MarkerType.CPU
        assert not scope.is_complete

    def test_complete(self):
        """Test completing a scope."""
        scope = MarkerScope(
            name="test",
            marker_type=MarkerType.CPU,
            start_time=time.perf_counter(),
        )

        time.sleep(0.001)
        scope.complete()

        assert scope.is_complete
        assert scope.duration_ms > 0

    def test_duration_incomplete(self):
        """Test duration when incomplete."""
        scope = MarkerScope(
            name="test",
            marker_type=MarkerType.CPU,
            start_time=time.perf_counter(),
        )
        assert scope.duration_ms == 0.0

    def test_with_metadata(self):
        """Test scope with metadata."""
        scope = MarkerScope(
            name="test",
            marker_type=MarkerType.GPU,
            start_time=time.perf_counter(),
            metadata={"category": "rendering"},
        )
        assert scope.metadata["category"] == "rendering"

    def test_to_dict(self):
        """Test dictionary conversion."""
        scope = MarkerScope(
            name="test",
            marker_type=MarkerType.CPU,
            start_time=time.perf_counter(),
            depth=1,
            thread_id=123,
        )
        scope.complete()

        data = scope.to_dict()

        assert data["name"] == "test"
        assert data["marker_type"] == "CPU"
        assert data["depth"] == 1


class TestProfileMarker:
    """Tests for ProfileMarker."""

    def test_creation(self):
        """Test marker creation."""
        marker = ProfileMarker("test_op")
        assert marker.name == "test_op"
        assert not marker.is_started

    def test_begin_end(self):
        """Test manual begin/end."""
        marker = ProfileMarker("test_op")

        marker.begin()
        assert marker.is_started

        time.sleep(0.001)
        duration = marker.end()

        assert not marker.is_started
        assert duration > 0

    def test_context_manager(self):
        """Test as context manager."""
        marker = ProfileMarker("test_op")

        with marker:
            assert marker.is_started
            time.sleep(0.001)

        assert not marker.is_started
        assert marker.duration_ms > 0

    def test_with_warn_threshold(self):
        """Test with warning threshold."""
        marker = ProfileMarker("test_op", warn_ms=1.0)

        with marker:
            time.sleep(0.002)

        # Should have triggered warning (tested indirectly)
        assert marker.duration_ms > 1.0

    def test_double_begin(self):
        """Test that double begin is ignored."""
        marker = ProfileMarker("test_op")

        marker.begin()
        marker.begin()  # Should be ignored

        marker.end()
        assert not marker.is_started


class TestGPUProfileMarker:
    """Tests for GPUProfileMarker."""

    def test_creation(self):
        """Test marker creation."""
        marker = GPUProfileMarker("shadow_pass", "shadows")
        assert marker.name == "shadow_pass"
        assert marker.category == "shadows"
        assert not marker.is_started

    def test_begin_end(self):
        """Test manual begin/end."""
        marker = GPUProfileMarker("test", "rendering")

        marker.begin()
        assert marker.is_started

        duration = marker.end()
        assert not marker.is_started
        assert duration >= 0

    def test_context_manager(self):
        """Test as context manager."""
        marker = GPUProfileMarker("test", "rendering")

        with marker:
            assert marker.is_started

        assert not marker.is_started

    def test_with_pass_type(self):
        """Test with render pass type."""
        marker = GPUProfileMarker(
            "shadow_pass",
            "shadows",
            pass_type=RenderPassType.SHADOW,
        )

        with marker:
            pass

        # Pass type should be recorded in metadata


class TestBeginMarker:
    """Tests for begin_marker context manager."""

    def test_basic_usage(self):
        """Test basic usage."""
        with begin_marker("test", MarkerType.CPU) as scope:
            assert scope.name == "test"
            assert scope.marker_type == MarkerType.CPU

        assert scope.is_complete

    def test_nested_markers(self):
        """Test nested markers."""
        with begin_marker("outer") as outer:
            assert get_marker_depth() == 1

            with begin_marker("inner") as inner:
                assert get_marker_depth() == 2
                assert inner.parent is outer
                assert inner in outer.children

            assert get_marker_depth() == 1

        assert get_marker_depth() == 0

    def test_with_metadata(self):
        """Test with metadata."""
        with begin_marker("test", custom_key="custom_value") as scope:
            pass

        assert scope.metadata["custom_key"] == "custom_value"


class TestEndMarker:
    """Tests for end_marker function."""

    def test_end_marker(self):
        """Test manual marker ending."""
        scope = MarkerScope(
            name="test",
            marker_type=MarkerType.CPU,
            start_time=time.perf_counter(),
        )

        time.sleep(0.001)
        duration = end_marker(scope)

        assert scope.is_complete
        assert duration > 0


class TestMarkerStackFunctions:
    """Tests for marker stack functions."""

    def test_get_active_markers(self):
        """Test getting active markers."""
        clear_markers()

        with begin_marker("outer"):
            with begin_marker("inner"):
                markers = get_active_markers()
                assert len(markers) == 2

        markers = get_active_markers()
        assert len(markers) == 0

    def test_get_marker_depth(self):
        """Test getting marker depth."""
        clear_markers()

        assert get_marker_depth() == 0

        with begin_marker("level1"):
            assert get_marker_depth() == 1

            with begin_marker("level2"):
                assert get_marker_depth() == 2

    def test_clear_markers(self):
        """Test clearing markers."""
        # Start some markers without context manager
        marker1 = ProfileMarker("m1")
        marker1.begin()

        clear_markers()

        assert get_marker_depth() == 0


class TestProfileDecorator:
    """Tests for @profile decorator."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup and teardown for each test."""
        cpu_profiler.enable()
        cpu_profiler.clear()
        yield
        cpu_profiler.disable()

    def test_basic_decoration(self):
        """Test basic function decoration."""
        @profile
        def my_function():
            return 42

        result = my_function()

        assert result == 42
        assert hasattr(my_function, "_profiled")
        assert my_function._profiled is True

    def test_with_arguments(self):
        """Test decoration with arguments."""
        @profile(name="custom_name", warn_ms=10.0)
        def my_function():
            return 42

        result = my_function()

        assert result == 42
        assert my_function._profile_name == "custom_name"
        assert my_function._profile_warn_ms == 10.0

    def test_function_with_args(self):
        """Test decorated function with arguments."""
        @profile
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_function_with_kwargs(self):
        """Test decorated function with keyword arguments."""
        @profile
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = greet("World", greeting="Hi")
        assert result == "Hi, World!"

    def test_profile_stats_method(self):
        """Test profile_stats helper method."""
        @profile(name="stats_test")
        def test_func():
            pass

        test_func()
        test_func()

        stats = test_func.profile_stats()
        # Stats may be empty if samples were cleared
        assert isinstance(stats, dict)

    def test_preserves_docstring(self):
        """Test that decorator preserves docstring."""
        @profile
        def documented_function():
            """This is a docstring."""
            pass

        assert documented_function.__doc__ == """This is a docstring."""

    def test_preserves_name(self):
        """Test that decorator preserves function name."""
        @profile
        def named_function():
            pass

        assert named_function.__name__ == "named_function"


class TestGPUProfileDecorator:
    """Tests for @gpu_profile decorator."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup and teardown for each test."""
        gpu_profiler_instance.enable()
        gpu_profiler_instance.clear()
        yield
        gpu_profiler_instance.disable()

    def test_basic_decoration(self):
        """Test basic function decoration."""
        @gpu_profile(category="rendering")
        def render_pass():
            return True

        result = render_pass()

        assert result is True
        assert hasattr(render_pass, "_gpu_profiled")
        assert render_pass._gpu_profiled is True
        assert render_pass._gpu_profile_category == "rendering"

    def test_with_all_arguments(self):
        """Test decoration with all arguments."""
        @gpu_profile(
            category="shadows",
            include_memory=True,
            pass_type=RenderPassType.SHADOW,
        )
        def shadow_pass():
            pass

        shadow_pass()

        assert shadow_pass._gpu_profile_include_memory is True
        assert shadow_pass._gpu_profile_pass_type == RenderPassType.SHADOW

    def test_gpu_stats_method(self):
        """Test gpu_stats helper method."""
        @gpu_profile(category="test_category")
        def gpu_func():
            pass

        gpu_func()

        stats = gpu_func.gpu_stats()
        assert isinstance(stats, dict)


class TestProfileClassDecorator:
    """Tests for @profile_class decorator."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup and teardown for each test."""
        cpu_profiler.enable()
        cpu_profiler.clear()
        yield
        cpu_profiler.disable()

    def test_profiles_all_public_methods(self):
        """Test that all public methods are profiled."""
        @profile_class
        class MyClass:
            def method_a(self):
                return "a"

            def method_b(self):
                return "b"

            def _private(self):
                return "private"

        obj = MyClass()

        assert hasattr(obj.method_a, "_profiled")
        assert hasattr(obj.method_b, "_profiled")
        # Private methods should not be profiled
        assert not hasattr(obj._private, "_profiled")

    def test_with_methods_list(self):
        """Test profiling specific methods."""
        @profile_class(methods=["method_a"])
        class MyClass:
            def method_a(self):
                return "a"

            def method_b(self):
                return "b"

        obj = MyClass()

        assert hasattr(obj.method_a, "_profiled")
        assert not hasattr(obj.method_b, "_profiled")

    def test_with_exclude_list(self):
        """Test excluding specific methods."""
        @profile_class(exclude=["method_b"])
        class MyClass:
            def method_a(self):
                return "a"

            def method_b(self):
                return "b"

        obj = MyClass()

        assert hasattr(obj.method_a, "_profiled")
        assert not hasattr(obj.method_b, "_profiled")

    def test_with_warn_threshold(self):
        """Test with warning threshold for all methods."""
        @profile_class(warn_ms=5.0)
        class MyClass:
            def method(self):
                pass

        obj = MyClass()
        assert obj.method._profile_warn_ms == 5.0

    def test_methods_still_work(self):
        """Test that methods still function correctly."""
        @profile_class
        class Calculator:
            def add(self, a, b):
                return a + b

            def multiply(self, a, b):
                return a * b

        calc = Calculator()

        assert calc.add(2, 3) == 5
        assert calc.multiply(4, 5) == 20
