"""Tests for build pipeline orchestration."""
import pytest
import time
import threading
from engine.tooling.build.build_pipeline import (
    BuildStage,
    BuildStageStatus,
    BuildStageResult,
    BuildDependency,
    BuildGraph,
    BuildExecutor,
    SequentialBuildExecutor,
    ParallelBuildExecutor,
    BuildPipeline,
    create_compile_stage,
    create_link_stage,
)


class TestBuildStageStatus:
    """Tests for BuildStageStatus enum."""

    def test_all_statuses_exist(self):
        """Test all stage statuses exist."""
        assert BuildStageStatus.PENDING
        assert BuildStageStatus.QUEUED
        assert BuildStageStatus.RUNNING
        assert BuildStageStatus.COMPLETED
        assert BuildStageStatus.FAILED
        assert BuildStageStatus.SKIPPED
        assert BuildStageStatus.CANCELLED


class TestBuildStageResult:
    """Tests for BuildStageResult dataclass."""

    def test_success_result(self):
        """Test creating success result."""
        result = BuildStageResult(
            success=True,
            elapsed_time=1.5,
            output="Build complete",
        )
        assert result.success is True
        assert result.elapsed_time == 1.5
        assert result.error is None

    def test_failure_result(self):
        """Test creating failure result."""
        result = BuildStageResult(
            success=False,
            elapsed_time=0.5,
            error="Compilation error",
        )
        assert result.success is False
        assert result.error == "Compilation error"


class TestBuildStage:
    """Tests for BuildStage dataclass."""

    def test_stage_creation(self):
        """Test creating a build stage."""
        def execute_fn(context):
            return BuildStageResult(success=True, elapsed_time=0)

        stage = BuildStage(name="compile", execute=execute_fn)
        assert stage.name == "compile"
        assert stage.status == BuildStageStatus.PENDING
        assert stage.result is None

    def test_stage_with_dependencies(self):
        """Test stage with dependencies."""
        stage = BuildStage(
            name="link",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
            dependencies={"compile"},
        )
        assert "compile" in stage.dependencies

    def test_stage_reset(self):
        """Test resetting stage state."""
        stage = BuildStage(
            name="test",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
        )
        stage.status = BuildStageStatus.COMPLETED
        stage.result = BuildStageResult(success=True, elapsed_time=1.0)

        stage.reset()
        assert stage.status == BuildStageStatus.PENDING
        assert stage.result is None


class TestBuildDependency:
    """Tests for BuildDependency dataclass."""

    def test_required_dependency(self):
        """Test required dependency."""
        dep = BuildDependency(
            source="compile",
            target="link",
            dependency_type="required",
        )
        assert dep.is_satisfied({"compile"})
        assert not dep.is_satisfied(set())

    def test_optional_dependency(self):
        """Test optional dependency."""
        dep = BuildDependency(
            source="docs",
            target="package",
            dependency_type="optional",
        )
        assert dep.is_satisfied(set())  # Optional always satisfied

    def test_conditional_dependency(self):
        """Test conditional dependency."""
        condition_met = True
        dep = BuildDependency(
            source="test",
            target="deploy",
            dependency_type="conditional",
            condition=lambda: condition_met,
        )
        assert dep.is_satisfied({"test"})


class TestBuildGraph:
    """Tests for BuildGraph."""

    def test_add_stage(self):
        """Test adding stages to graph."""
        graph = BuildGraph()
        stage = BuildStage(
            name="compile",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
        )
        graph.add_stage(stage)
        assert graph.get_stage("compile") is not None

    def test_remove_stage(self):
        """Test removing stages from graph."""
        graph = BuildGraph()
        stage = BuildStage(
            name="compile",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
        )
        graph.add_stage(stage)
        result = graph.remove_stage("compile")
        assert result is True
        assert graph.get_stage("compile") is None

    def test_get_dependencies(self):
        """Test getting stage dependencies."""
        graph = BuildGraph()
        compile_stage = BuildStage(
            name="compile",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
        )
        link_stage = BuildStage(
            name="link",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
            dependencies={"compile"},
        )
        graph.add_stage(compile_stage)
        graph.add_stage(link_stage)

        deps = graph.get_dependencies("link")
        assert "compile" in deps

    def test_get_dependents(self):
        """Test getting stage dependents."""
        graph = BuildGraph()
        compile_stage = BuildStage(
            name="compile",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
        )
        link_stage = BuildStage(
            name="link",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
            dependencies={"compile"},
        )
        graph.add_stage(compile_stage)
        graph.add_stage(link_stage)

        dependents = graph.get_dependents("compile")
        assert "link" in dependents

    def test_get_ready_stages(self):
        """Test getting ready stages."""
        graph = BuildGraph()
        compile_stage = BuildStage(
            name="compile",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
        )
        link_stage = BuildStage(
            name="link",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
            dependencies={"compile"},
        )
        graph.add_stage(compile_stage)
        graph.add_stage(link_stage)

        # Initially only compile is ready
        ready = graph.get_ready_stages(set())
        assert len(ready) == 1
        assert ready[0].name == "compile"

        # After compile, link is ready
        ready = graph.get_ready_stages({"compile"})
        assert len(ready) == 1
        assert ready[0].name == "link"

    def test_topological_sort(self):
        """Test topological sorting."""
        graph = BuildGraph()
        for name in ["a", "b", "c"]:
            deps = set()
            if name == "b":
                deps = {"a"}
            if name == "c":
                deps = {"b"}
            stage = BuildStage(
                name=name,
                execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
                dependencies=deps,
            )
            graph.add_stage(stage)

        order = graph.topological_sort()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_circular_dependency_detection(self):
        """Test detecting circular dependencies."""
        graph = BuildGraph()
        stage_a = BuildStage(
            name="a",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
            dependencies={"b"},
        )
        stage_b = BuildStage(
            name="b",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
            dependencies={"a"},
        )
        graph.add_stage(stage_a)
        graph.add_stage(stage_b)

        with pytest.raises(ValueError, match="[Cc]ircular"):
            graph.topological_sort()

    def test_validate_missing_dependency(self):
        """Test validating missing dependencies."""
        graph = BuildGraph()
        stage = BuildStage(
            name="link",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
            dependencies={"compile"},  # compile doesn't exist
        )
        graph.add_stage(stage)

        errors = graph.validate()
        assert len(errors) > 0


class TestSequentialBuildExecutor:
    """Tests for SequentialBuildExecutor."""

    def test_execute_single_stage(self):
        """Test executing single stage."""
        graph = BuildGraph()
        executed = []

        def execute_fn(context):
            executed.append("compile")
            return BuildStageResult(success=True, elapsed_time=0.1)

        stage = BuildStage(name="compile", execute=execute_fn)
        graph.add_stage(stage)

        executor = SequentialBuildExecutor()
        results = executor.execute(graph, {})

        assert "compile" in executed
        assert results["compile"].success is True

    def test_execute_with_dependencies(self):
        """Test executing with dependencies."""
        graph = BuildGraph()
        order = []

        def make_execute(name):
            def execute_fn(context):
                order.append(name)
                return BuildStageResult(success=True, elapsed_time=0.1)
            return execute_fn

        compile_stage = BuildStage(name="compile", execute=make_execute("compile"))
        link_stage = BuildStage(
            name="link",
            execute=make_execute("link"),
            dependencies={"compile"},
        )
        graph.add_stage(compile_stage)
        graph.add_stage(link_stage)

        executor = SequentialBuildExecutor()
        results = executor.execute(graph, {})

        assert order == ["compile", "link"]

    def test_dependency_failure_skips_dependents(self):
        """Test that failed dependencies skip dependents."""
        graph = BuildGraph()

        compile_stage = BuildStage(
            name="compile",
            execute=lambda ctx: BuildStageResult(success=False, elapsed_time=0.1, error="Error"),
        )
        link_stage = BuildStage(
            name="link",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0.1),
            dependencies={"compile"},
        )
        graph.add_stage(compile_stage)
        graph.add_stage(link_stage)

        executor = SequentialBuildExecutor()
        results = executor.execute(graph, {})

        assert results["compile"].success is False
        assert results["link"].success is False
        assert graph.get_stage("link").status == BuildStageStatus.SKIPPED

    def test_cancel(self):
        """Test cancelling execution."""
        graph = BuildGraph()
        start_event = threading.Event()
        cancel_event = threading.Event()

        def slow_execute(context):
            start_event.set()
            cancel_event.wait()
            return BuildStageResult(success=True, elapsed_time=1.0)

        stage = BuildStage(name="slow", execute=slow_execute)
        graph.add_stage(stage)

        executor = SequentialBuildExecutor()

        def run_build():
            executor.execute(graph, {})

        thread = threading.Thread(target=run_build)
        thread.start()

        start_event.wait()
        executor.cancel()
        cancel_event.set()
        thread.join(timeout=2.0)


class TestParallelBuildExecutor:
    """Tests for ParallelBuildExecutor."""

    def test_parallel_execution(self):
        """Test parallel execution of independent stages."""
        graph = BuildGraph()
        start_times = {}

        def make_execute(name):
            def execute_fn(context):
                start_times[name] = time.time()
                time.sleep(0.1)
                return BuildStageResult(success=True, elapsed_time=0.1)
            return execute_fn

        # Two independent stages
        stage_a = BuildStage(name="a", execute=make_execute("a"))
        stage_b = BuildStage(name="b", execute=make_execute("b"))
        graph.add_stage(stage_a)
        graph.add_stage(stage_b)

        executor = ParallelBuildExecutor(max_workers=2)
        results = executor.execute(graph, {})

        assert results["a"].success is True
        assert results["b"].success is True

        # Should start at roughly the same time
        time_diff = abs(start_times["a"] - start_times["b"])
        assert time_diff < 0.05  # Within 50ms

    def test_respects_dependencies(self):
        """Test parallel execution respects dependencies."""
        graph = BuildGraph()
        order = []
        lock = threading.Lock()

        def make_execute(name):
            def execute_fn(context):
                with lock:
                    order.append(name)
                time.sleep(0.05)
                return BuildStageResult(success=True, elapsed_time=0.05)
            return execute_fn

        stage_a = BuildStage(name="a", execute=make_execute("a"))
        stage_b = BuildStage(
            name="b",
            execute=make_execute("b"),
            dependencies={"a"},
        )
        graph.add_stage(stage_a)
        graph.add_stage(stage_b)

        executor = ParallelBuildExecutor(max_workers=2)
        executor.execute(graph, {})

        assert order.index("a") < order.index("b")


class TestBuildPipeline:
    """Tests for BuildPipeline."""

    def test_add_stage(self):
        """Test adding stage to pipeline."""
        pipeline = BuildPipeline()
        stage = pipeline.add_stage(
            name="compile",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
        )
        assert stage.name == "compile"

    def test_remove_stage(self):
        """Test removing stage from pipeline."""
        pipeline = BuildPipeline()
        pipeline.add_stage(
            name="compile",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
        )
        result = pipeline.remove_stage("compile")
        assert result is True

    def test_set_context(self):
        """Test setting context values."""
        pipeline = BuildPipeline()
        pipeline.set_context("output_dir", "/build")
        assert pipeline.get_context("output_dir") == "/build"

    def test_run_pipeline(self):
        """Test running pipeline."""
        pipeline = BuildPipeline()
        executed = False

        def execute_fn(context):
            nonlocal executed
            executed = True
            return BuildStageResult(success=True, elapsed_time=0)

        pipeline.add_stage(name="compile", execute=execute_fn)
        results = pipeline.run()

        assert executed
        assert results["compile"].success is True

    def test_event_callbacks(self):
        """Test event callbacks."""
        pipeline = BuildPipeline()
        events = []

        def on_build_started(build_id):
            events.append(("started", build_id))

        def on_build_completed(build_id, results):
            events.append(("completed", build_id))

        pipeline.on("build_started", on_build_started)
        pipeline.on("build_completed", on_build_completed)
        pipeline.add_stage(
            name="test",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
        )

        pipeline.run()

        assert len(events) == 2
        assert events[0][0] == "started"
        assert events[1][0] == "completed"

    def test_is_running(self):
        """Test is_running state."""
        pipeline = BuildPipeline()
        start_event = threading.Event()
        continue_event = threading.Event()

        def slow_execute(context):
            start_event.set()
            continue_event.wait()
            return BuildStageResult(success=True, elapsed_time=0)

        pipeline.add_stage(name="slow", execute=slow_execute)

        def run_build():
            pipeline.run()

        thread = threading.Thread(target=run_build)
        thread.start()

        start_event.wait()
        assert pipeline.is_running() is True

        continue_event.set()
        thread.join(timeout=2.0)
        assert pipeline.is_running() is False

    def test_get_status(self):
        """Test getting stage statuses."""
        pipeline = BuildPipeline()
        pipeline.add_stage(
            name="compile",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
        )
        pipeline.add_stage(
            name="link",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
            dependencies={"compile"},
        )

        status = pipeline.get_status()
        assert "compile" in status
        assert "link" in status

    def test_validate(self):
        """Test pipeline validation."""
        pipeline = BuildPipeline()
        pipeline.add_stage(
            name="link",
            execute=lambda ctx: BuildStageResult(success=True, elapsed_time=0),
            dependencies={"compile"},  # Missing dependency
        )

        errors = pipeline.validate()
        assert len(errors) > 0


class TestStageFactories:
    """Tests for stage factory functions."""

    def test_create_compile_stage(self):
        """Test creating compile stage."""
        stage = create_compile_stage(
            name="compile_main",
            files=["main.cpp", "util.cpp"],
            output_dir="obj",
            compiler_flags=["-O2"],
        )
        assert stage.name == "compile_main"
        assert "files" in stage.metadata

    def test_create_link_stage(self):
        """Test creating link stage."""
        stage = create_link_stage(
            name="link_main",
            object_files=["main.o", "util.o"],
            output_path="main.exe",
            linker_flags=["-lpthread"],
            dependencies={"compile_main"},
        )
        assert stage.name == "link_main"
        assert "compile_main" in stage.dependencies
