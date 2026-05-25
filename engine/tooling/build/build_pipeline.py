"""Build pipeline orchestration with stages, dependencies, and parallelism.

Provides a directed acyclic graph (DAG) based build pipeline that supports
parallel execution, dependency management, and progress tracking.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import asyncio
import concurrent.futures
import threading
import time
import uuid


class BuildStageStatus(Enum):
    """Status of a build stage."""
    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    CANCELLED = auto()


@dataclass
class BuildStageResult:
    """Result of executing a build stage."""
    success: bool
    elapsed_time: float
    output: Any = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)


@dataclass
class BuildStage:
    """A single stage in the build pipeline."""
    name: str
    execute: Callable[..., BuildStageResult]
    dependencies: Set[str] = field(default_factory=set)
    status: BuildStageStatus = BuildStageStatus.PENDING
    result: Optional[BuildStageResult] = None
    priority: int = 0
    can_skip: bool = False
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.id = str(uuid.uuid4())

    def reset(self) -> None:
        """Reset the stage to pending state."""
        self.status = BuildStageStatus.PENDING
        self.result = None
        self.retry_count = 0


@dataclass
class BuildDependency:
    """Represents a dependency between build stages."""
    source: str
    target: str
    dependency_type: str = "required"  # required, optional, conditional
    condition: Optional[Callable[[], bool]] = None

    def is_satisfied(self, completed_stages: Set[str]) -> bool:
        """Check if this dependency is satisfied."""
        if self.dependency_type == "optional":
            return True
        if self.dependency_type == "conditional" and self.condition:
            return self.condition()
        return self.source in completed_stages


class BuildGraph:
    """Directed acyclic graph for build dependencies."""

    def __init__(self):
        self._stages: Dict[str, BuildStage] = {}
        self._dependencies: Dict[str, Set[str]] = {}  # stage -> set of dependencies
        self._dependents: Dict[str, Set[str]] = {}    # stage -> set of dependents
        self._lock = threading.Lock()

    def add_stage(self, stage: BuildStage) -> None:
        """Add a stage to the graph."""
        with self._lock:
            self._stages[stage.name] = stage
            self._dependencies[stage.name] = set(stage.dependencies)
            if stage.name not in self._dependents:
                self._dependents[stage.name] = set()

            # Update dependents for each dependency
            for dep in stage.dependencies:
                if dep not in self._dependents:
                    self._dependents[dep] = set()
                self._dependents[dep].add(stage.name)

    def remove_stage(self, name: str) -> bool:
        """Remove a stage from the graph."""
        with self._lock:
            if name not in self._stages:
                return False

            # Remove from dependents lists
            for dep in self._dependencies[name]:
                if dep in self._dependents:
                    self._dependents[dep].discard(name)

            # Remove dependents
            for dependent in self._dependents.get(name, set()):
                if dependent in self._dependencies:
                    self._dependencies[dependent].discard(name)

            del self._stages[name]
            del self._dependencies[name]
            if name in self._dependents:
                del self._dependents[name]

            return True

    def get_stage(self, name: str) -> Optional[BuildStage]:
        """Get a stage by name."""
        return self._stages.get(name)

    def get_all_stages(self) -> List[BuildStage]:
        """Get all stages."""
        return list(self._stages.values())

    def get_dependencies(self, name: str) -> Set[str]:
        """Get dependencies of a stage."""
        return set(self._dependencies.get(name, set()))

    def get_dependents(self, name: str) -> Set[str]:
        """Get dependents of a stage."""
        return set(self._dependents.get(name, set()))

    def get_ready_stages(self, completed: Set[str]) -> List[BuildStage]:
        """Get stages that are ready to execute."""
        ready = []
        for name, stage in self._stages.items():
            # Stage must be pending and not already completed
            if stage.status == BuildStageStatus.PENDING and name not in completed:
                deps = self._dependencies.get(name, set())
                if deps.issubset(completed):
                    ready.append(stage)
        # Sort by priority (higher priority first)
        ready.sort(key=lambda s: s.priority, reverse=True)
        return ready

    def topological_sort(self) -> List[str]:
        """Return stages in topological order."""
        in_degree = {name: len(deps) for name, deps in self._dependencies.items()}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort by priority to get consistent ordering
            queue.sort(key=lambda n: self._stages[n].priority if n in self._stages else 0, reverse=True)
            name = queue.pop(0)
            result.append(name)

            for dependent in self._dependents.get(name, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._stages):
            raise ValueError("Circular dependency detected in build graph")

        return result

    def validate(self) -> List[str]:
        """Validate the graph and return any errors."""
        errors = []

        # Check for missing dependencies
        for name, deps in self._dependencies.items():
            for dep in deps:
                if dep not in self._stages:
                    errors.append(f"Stage '{name}' depends on non-existent stage '{dep}'")

        # Check for circular dependencies
        try:
            self.topological_sort()
        except ValueError as e:
            errors.append(str(e))

        return errors

    def reset(self) -> None:
        """Reset all stages to pending state."""
        for stage in self._stages.values():
            stage.reset()


class BuildExecutor(ABC):
    """Abstract base class for build executors."""

    @abstractmethod
    def execute(self, graph: BuildGraph, context: Dict[str, Any]) -> Dict[str, BuildStageResult]:
        """Execute the build graph."""
        pass

    @abstractmethod
    def cancel(self) -> None:
        """Cancel the current build."""
        pass


class SequentialBuildExecutor(BuildExecutor):
    """Executes build stages sequentially."""

    def __init__(self):
        self._cancelled = False

    def execute(self, graph: BuildGraph, context: Dict[str, Any]) -> Dict[str, BuildStageResult]:
        """Execute stages in topological order."""
        self._cancelled = False
        results: Dict[str, BuildStageResult] = {}
        completed: Set[str] = set()

        try:
            order = graph.topological_sort()
        except ValueError as e:
            # Return error for all stages
            error_result = BuildStageResult(success=False, elapsed_time=0, error=str(e))
            for stage in graph.get_all_stages():
                results[stage.name] = error_result
                stage.status = BuildStageStatus.FAILED
            return results

        for name in order:
            if self._cancelled:
                stage = graph.get_stage(name)
                if stage:
                    stage.status = BuildStageStatus.CANCELLED
                    results[name] = BuildStageResult(success=False, elapsed_time=0, error="Cancelled")
                continue

            stage = graph.get_stage(name)
            if not stage:
                continue

            # Check if dependencies succeeded
            deps_failed = any(
                results.get(dep, BuildStageResult(success=False, elapsed_time=0)).success is False
                for dep in graph.get_dependencies(name)
            )

            if deps_failed:
                stage.status = BuildStageStatus.SKIPPED
                results[name] = BuildStageResult(success=False, elapsed_time=0, error="Dependencies failed")
                continue

            # Execute the stage
            stage.status = BuildStageStatus.RUNNING
            start_time = time.time()

            try:
                result = stage.execute(context)
                stage.result = result
                results[name] = result

                if result.success:
                    stage.status = BuildStageStatus.COMPLETED
                    completed.add(name)
                else:
                    stage.status = BuildStageStatus.FAILED

            except Exception as e:
                elapsed = time.time() - start_time
                result = BuildStageResult(success=False, elapsed_time=elapsed, error=str(e))
                stage.result = result
                stage.status = BuildStageStatus.FAILED
                results[name] = result

        return results

    def cancel(self) -> None:
        """Cancel the build."""
        self._cancelled = True


class ParallelBuildExecutor(BuildExecutor):
    """Executes build stages in parallel where possible."""

    def __init__(self, max_workers: int = 4):
        self._max_workers = max_workers
        self._cancelled = False
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._lock = threading.Lock()

    def execute(self, graph: BuildGraph, context: Dict[str, Any]) -> Dict[str, BuildStageResult]:
        """Execute stages in parallel respecting dependencies."""
        self._cancelled = False
        results: Dict[str, BuildStageResult] = {}
        completed: Set[str] = set()
        failed: Set[str] = set()

        # Validate graph
        errors = graph.validate()
        if errors:
            error_msg = "; ".join(errors)
            error_result = BuildStageResult(success=False, elapsed_time=0, error=error_msg)
            for stage in graph.get_all_stages():
                results[stage.name] = error_result
                stage.status = BuildStageStatus.FAILED
            return results

        # Create thread pool
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
        futures: Dict[concurrent.futures.Future, str] = {}

        try:
            while len(completed) + len(failed) < len(graph.get_all_stages()):
                if self._cancelled:
                    break

                # Find ready stages
                ready = graph.get_ready_stages(completed)

                # Filter out stages already in progress or failed
                in_progress = {name for name in futures.values()}
                ready = [s for s in ready if s.name not in in_progress and s.name not in failed]

                # Submit ready stages
                for stage in ready:
                    # Check if dependencies failed
                    deps = graph.get_dependencies(stage.name)
                    if any(d in failed for d in deps):
                        stage.status = BuildStageStatus.SKIPPED
                        failed.add(stage.name)
                        results[stage.name] = BuildStageResult(
                            success=False, elapsed_time=0, error="Dependencies failed"
                        )
                        continue

                    stage.status = BuildStageStatus.QUEUED
                    future = self._executor.submit(self._execute_stage, stage, context)
                    futures[future] = stage.name

                # Wait for at least one to complete
                if futures:
                    done, _ = concurrent.futures.wait(
                        futures.keys(),
                        timeout=0.1,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for future in done:
                        name = futures.pop(future)
                        stage = graph.get_stage(name)

                        try:
                            result = future.result()
                            results[name] = result
                            if stage:
                                stage.result = result

                            if result.success:
                                completed.add(name)
                                if stage:
                                    stage.status = BuildStageStatus.COMPLETED
                            else:
                                failed.add(name)
                                if stage:
                                    stage.status = BuildStageStatus.FAILED

                        except Exception as e:
                            result = BuildStageResult(success=False, elapsed_time=0, error=str(e))
                            results[name] = result
                            failed.add(name)
                            if stage:
                                stage.result = result
                                stage.status = BuildStageStatus.FAILED
                else:
                    # No stages ready and none in progress - check for deadlock
                    remaining = set(s.name for s in graph.get_all_stages()) - completed - failed
                    if remaining:
                        # All remaining stages have unsatisfied dependencies
                        for name in remaining:
                            stage = graph.get_stage(name)
                            if stage:
                                stage.status = BuildStageStatus.SKIPPED
                            results[name] = BuildStageResult(
                                success=False, elapsed_time=0, error="Unsatisfied dependencies"
                            )
                        break

            # Handle cancelled stages
            if self._cancelled:
                for stage in graph.get_all_stages():
                    if stage.status in (BuildStageStatus.PENDING, BuildStageStatus.QUEUED):
                        stage.status = BuildStageStatus.CANCELLED
                        results[stage.name] = BuildStageResult(
                            success=False, elapsed_time=0, error="Cancelled"
                        )

        finally:
            self._executor.shutdown(wait=True)
            self._executor = None

        return results

    def _execute_stage(self, stage: BuildStage, context: Dict[str, Any]) -> BuildStageResult:
        """Execute a single stage."""
        stage.status = BuildStageStatus.RUNNING
        start_time = time.time()

        try:
            result = stage.execute(context)
            result.elapsed_time = time.time() - start_time
            return result
        except Exception as e:
            return BuildStageResult(
                success=False,
                elapsed_time=time.time() - start_time,
                error=str(e)
            )

    def cancel(self) -> None:
        """Cancel the build."""
        self._cancelled = True
        if self._executor:
            self._executor.shutdown(wait=False)


class BuildPipeline:
    """High-level build pipeline management."""

    def __init__(self, executor: Optional[BuildExecutor] = None):
        self._graph = BuildGraph()
        self._executor = executor or ParallelBuildExecutor()
        self._context: Dict[str, Any] = {}
        self._callbacks: Dict[str, List[Callable]] = {
            "stage_started": [],
            "stage_completed": [],
            "build_started": [],
            "build_completed": [],
        }
        self._current_build: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def graph(self) -> BuildGraph:
        """Get the build graph."""
        return self._graph

    def add_stage(
        self,
        name: str,
        execute: Callable[..., BuildStageResult],
        dependencies: Optional[Set[str]] = None,
        priority: int = 0,
        **kwargs
    ) -> BuildStage:
        """Add a stage to the pipeline."""
        stage = BuildStage(
            name=name,
            execute=execute,
            dependencies=dependencies or set(),
            priority=priority,
            **kwargs
        )
        self._graph.add_stage(stage)
        return stage

    def remove_stage(self, name: str) -> bool:
        """Remove a stage from the pipeline."""
        return self._graph.remove_stage(name)

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self._context.get(key, default)

    def on(self, event: str, callback: Callable) -> None:
        """Register an event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, *args, **kwargs) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception:
                pass  # Don't let callback errors affect build

    def run(self, build_id: Optional[str] = None) -> Dict[str, BuildStageResult]:
        """Run the build pipeline."""
        with self._lock:
            if self._current_build:
                raise RuntimeError("A build is already in progress")
            self._current_build = build_id or str(uuid.uuid4())

        try:
            self._emit("build_started", self._current_build)
            self._graph.reset()
            results = self._executor.execute(self._graph, self._context)
            self._emit("build_completed", self._current_build, results)
            return results
        finally:
            with self._lock:
                self._current_build = None

    def cancel(self) -> None:
        """Cancel the current build."""
        self._executor.cancel()

    def is_running(self) -> bool:
        """Check if a build is currently running."""
        with self._lock:
            return self._current_build is not None

    def get_status(self) -> Dict[str, BuildStageStatus]:
        """Get the status of all stages."""
        return {stage.name: stage.status for stage in self._graph.get_all_stages()}

    def validate(self) -> List[str]:
        """Validate the pipeline configuration."""
        return self._graph.validate()


# Utility functions for creating common stage types
def create_compile_stage(
    name: str,
    files: List[str],
    output_dir: str,
    compiler_flags: List[str],
    dependencies: Optional[Set[str]] = None
) -> BuildStage:
    """Create a compilation stage."""
    def execute(context: Dict[str, Any]) -> BuildStageResult:
        # Placeholder implementation
        return BuildStageResult(
            success=True,
            elapsed_time=0,
            artifacts=[f"{output_dir}/{f}.o" for f in files]
        )

    return BuildStage(
        name=name,
        execute=execute,
        dependencies=dependencies or set(),
        metadata={"files": files, "output_dir": output_dir, "flags": compiler_flags}
    )


def create_link_stage(
    name: str,
    object_files: List[str],
    output_path: str,
    linker_flags: List[str],
    dependencies: Optional[Set[str]] = None
) -> BuildStage:
    """Create a linking stage."""
    def execute(context: Dict[str, Any]) -> BuildStageResult:
        # Placeholder implementation
        return BuildStageResult(
            success=True,
            elapsed_time=0,
            artifacts=[output_path]
        )

    return BuildStage(
        name=name,
        execute=execute,
        dependencies=dependencies or set(),
        metadata={"objects": object_files, "output": output_path, "flags": linker_flags}
    )
