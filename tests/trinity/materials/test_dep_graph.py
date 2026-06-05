"""Tests for the MaterialDepGraph bidirectional dependency graph (T-MAT-2.6).

Verifies:
- Edge recording for material compilations
- BFS invalidation set computation
- Material-to-material dependencies
- Concurrent access (basic threading test)
- Graph clearing and re-recording
- Transitive closure correctness

Acceptance Criteria:
1. Inserting 10 materials produces correct adjacency
2. BFS returns transitive closure
3. Lock contention < 1us under concurrent reads (basic threading test)
"""

from __future__ import annotations

import pytest
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from trinity.materials.dep_graph import MaterialDepGraph


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def graph() -> MaterialDepGraph:
    """Create a fresh MaterialDepGraph."""
    return MaterialDepGraph()


@pytest.fixture
def sample_materials() -> dict[str, set[str]]:
    """Sample material -> includes mapping for testing."""
    return {
        "materials/gold.wgsl": {"shaders/pbr/brdf.wgsl", "shaders/common/color.wgsl"},
        "materials/silver.wgsl": {"shaders/pbr/brdf.wgsl", "shaders/common/color.wgsl"},
        "materials/copper.wgsl": {"shaders/pbr/brdf.wgsl", "shaders/common/math.wgsl"},
        "materials/glass.wgsl": {"shaders/pbr/refraction.wgsl", "shaders/common/color.wgsl"},
        "materials/water.wgsl": {"shaders/pbr/refraction.wgsl", "shaders/water/caustics.wgsl"},
    }


@pytest.fixture
def ten_materials() -> dict[str, set[str]]:
    """Ten materials with varying include patterns for acceptance test."""
    return {
        "materials/mat_0.wgsl": {"includes/common.wgsl", "includes/utils.wgsl"},
        "materials/mat_1.wgsl": {"includes/common.wgsl"},
        "materials/mat_2.wgsl": {"includes/utils.wgsl", "includes/math.wgsl"},
        "materials/mat_3.wgsl": {"includes/common.wgsl", "includes/math.wgsl"},
        "materials/mat_4.wgsl": {"includes/pbr.wgsl"},
        "materials/mat_5.wgsl": {"includes/pbr.wgsl", "includes/common.wgsl"},
        "materials/mat_6.wgsl": {"includes/color.wgsl"},
        "materials/mat_7.wgsl": {"includes/color.wgsl", "includes/utils.wgsl"},
        "materials/mat_8.wgsl": {"includes/math.wgsl", "includes/color.wgsl"},
        "materials/mat_9.wgsl": {"includes/common.wgsl", "includes/pbr.wgsl", "includes/color.wgsl"},
    }


# =============================================================================
# Suite A: Basic Edge Recording
# =============================================================================


class TestEdgeRecording:
    """Tests for recording material compilation edges."""

    def test_record_single_material(self, graph: MaterialDepGraph) -> None:
        """Record a single material with includes."""
        material = Path("materials/gold.wgsl")
        includes = {Path("shaders/brdf.wgsl"), Path("shaders/color.wgsl")}

        graph.record_material_compilation(material, includes)

        assert graph.material_count() == 1
        assert graph.include_count() == 2

    def test_record_multiple_materials(
        self, graph: MaterialDepGraph, sample_materials: dict
    ) -> None:
        """Record multiple materials with shared includes."""
        for mat, incs in sample_materials.items():
            graph.record_material_compilation(
                Path(mat), {Path(inc) for inc in incs}
            )

        assert graph.material_count() == 5
        # Count unique includes
        all_includes = set()
        for incs in sample_materials.values():
            all_includes.update(incs)
        assert graph.include_count() == len(all_includes)

    def test_record_ten_materials_adjacency(
        self, graph: MaterialDepGraph, ten_materials: dict
    ) -> None:
        """Acceptance: Inserting 10 materials produces correct adjacency."""
        for mat, incs in ten_materials.items():
            graph.record_material_compilation(
                Path(mat), {Path(inc) for inc in incs}
            )

        assert graph.material_count() == 10

        # Verify adjacency for common.wgsl (used by mat_0, 1, 3, 5, 9)
        common_users = graph.get_include_materials(Path("includes/common.wgsl"))
        expected = {
            Path("materials/mat_0.wgsl").resolve(),
            Path("materials/mat_1.wgsl").resolve(),
            Path("materials/mat_3.wgsl").resolve(),
            Path("materials/mat_5.wgsl").resolve(),
            Path("materials/mat_9.wgsl").resolve(),
        }
        assert common_users == expected

    def test_recompilation_clears_old_edges(self, graph: MaterialDepGraph) -> None:
        """Re-recording a material clears old edges first."""
        material = Path("materials/test.wgsl")

        # First compilation
        graph.record_material_compilation(
            material, {Path("includes/a.wgsl"), Path("includes/b.wgsl")}
        )
        assert len(graph.get_material_includes(material)) == 2

        # Recompilation with different includes
        graph.record_material_compilation(
            material, {Path("includes/c.wgsl")}
        )
        includes = graph.get_material_includes(material)
        assert len(includes) == 1
        assert Path("includes/c.wgsl").resolve() in includes

        # Old includes should no longer reference the material
        a_users = graph.get_include_materials(Path("includes/a.wgsl"))
        assert material.resolve() not in a_users

    def test_empty_includes(self, graph: MaterialDepGraph) -> None:
        """Material with no includes."""
        material = Path("materials/simple.wgsl")
        graph.record_material_compilation(material, set())

        assert graph.material_count() == 1
        assert graph.include_count() == 0
        assert len(graph.get_material_includes(material)) == 0


# =============================================================================
# Suite B: BFS Invalidation Set
# =============================================================================


class TestBFSInvalidation:
    """Tests for broadest invalidation set computation."""

    def test_direct_include_change(
        self, graph: MaterialDepGraph, sample_materials: dict
    ) -> None:
        """Changing an include invalidates materials using it."""
        for mat, incs in sample_materials.items():
            graph.record_material_compilation(
                Path(mat), {Path(inc) for inc in incs}
            )

        # brdf.wgsl is used by gold, silver, copper
        affected = graph.broadest_invalidation_set(Path("shaders/pbr/brdf.wgsl"))

        expected = {
            Path("materials/gold.wgsl").resolve(),
            Path("materials/silver.wgsl").resolve(),
            Path("materials/copper.wgsl").resolve(),
        }
        assert affected == expected

    def test_isolated_include_change(
        self, graph: MaterialDepGraph, sample_materials: dict
    ) -> None:
        """Include used by single material."""
        for mat, incs in sample_materials.items():
            graph.record_material_compilation(
                Path(mat), {Path(inc) for inc in incs}
            )

        # caustics.wgsl only used by water
        affected = graph.broadest_invalidation_set(
            Path("shaders/water/caustics.wgsl")
        )
        assert affected == {Path("materials/water.wgsl").resolve()}

    def test_unused_include_change(self, graph: MaterialDepGraph) -> None:
        """Changing an unused include affects nothing."""
        graph.record_material_compilation(
            Path("materials/test.wgsl"), {Path("includes/used.wgsl")}
        )

        affected = graph.broadest_invalidation_set(Path("includes/unused.wgsl"))
        assert affected == set()

    def test_transitive_closure(self, graph: MaterialDepGraph) -> None:
        """Acceptance: BFS returns transitive closure via material deps."""
        # Set up a chain: include -> mat_a -> mat_b -> mat_c
        graph.record_material_compilation(
            Path("materials/mat_a.wgsl"), {Path("includes/base.wgsl")}
        )
        graph.record_material_compilation(
            Path("materials/mat_b.wgsl"), set()
        )
        graph.record_material_compilation(
            Path("materials/mat_c.wgsl"), set()
        )

        # Add material dependencies
        graph.record_material_dependency(
            Path("materials/mat_b.wgsl"), Path("materials/mat_a.wgsl")
        )
        graph.record_material_dependency(
            Path("materials/mat_c.wgsl"), Path("materials/mat_b.wgsl")
        )

        # Changing base.wgsl should invalidate all three materials
        affected = graph.broadest_invalidation_set(Path("includes/base.wgsl"))

        expected = {
            Path("materials/mat_a.wgsl").resolve(),
            Path("materials/mat_b.wgsl").resolve(),
            Path("materials/mat_c.wgsl").resolve(),
        }
        assert affected == expected

    def test_diamond_dependency(self, graph: MaterialDepGraph) -> None:
        """Diamond pattern in dependencies."""
        #        include
        #         / \
        #       A     B
        #         \ /
        #          C
        graph.record_material_compilation(
            Path("materials/A.wgsl"), {Path("includes/shared.wgsl")}
        )
        graph.record_material_compilation(
            Path("materials/B.wgsl"), {Path("includes/shared.wgsl")}
        )
        graph.record_material_compilation(
            Path("materials/C.wgsl"), set()
        )

        graph.record_material_dependency(
            Path("materials/C.wgsl"), Path("materials/A.wgsl")
        )
        graph.record_material_dependency(
            Path("materials/C.wgsl"), Path("materials/B.wgsl")
        )

        affected = graph.broadest_invalidation_set(Path("includes/shared.wgsl"))

        expected = {
            Path("materials/A.wgsl").resolve(),
            Path("materials/B.wgsl").resolve(),
            Path("materials/C.wgsl").resolve(),
        }
        assert affected == expected

    def test_material_change_propagates(self, graph: MaterialDepGraph) -> None:
        """Changing a material itself propagates to dependents."""
        graph.record_material_compilation(
            Path("materials/base.wgsl"), set()
        )
        graph.record_material_compilation(
            Path("materials/derived.wgsl"), set()
        )
        graph.record_material_dependency(
            Path("materials/derived.wgsl"), Path("materials/base.wgsl")
        )

        affected = graph.broadest_invalidation_set(Path("materials/base.wgsl"))

        expected = {
            Path("materials/base.wgsl").resolve(),
            Path("materials/derived.wgsl").resolve(),
        }
        assert affected == expected


# =============================================================================
# Suite C: Material Dependencies
# =============================================================================


class TestMaterialDependencies:
    """Tests for material-to-material dependencies."""

    def test_record_dependency(self, graph: MaterialDepGraph) -> None:
        """Record a single material dependency."""
        mat_a = Path("materials/A.wgsl")
        mat_b = Path("materials/B.wgsl")

        graph.record_material_compilation(mat_a, set())
        graph.record_material_compilation(mat_b, set())
        graph.record_material_dependency(mat_b, mat_a)

        # B depends on A
        deps = graph.get_material_dependencies(mat_b)
        assert mat_a.resolve() in deps

        # A has B as dependent
        dependents = graph.get_material_dependents(mat_a)
        assert mat_b.resolve() in dependents

    def test_multiple_dependencies(self, graph: MaterialDepGraph) -> None:
        """Material with multiple dependencies."""
        materials = [Path(f"materials/mat_{i}.wgsl") for i in range(4)]
        for mat in materials:
            graph.record_material_compilation(mat, set())

        # mat_3 depends on mat_0, mat_1, mat_2
        for i in range(3):
            graph.record_material_dependency(materials[3], materials[i])

        deps = graph.get_material_dependencies(materials[3])
        assert len(deps) == 3

    def test_deep_dependency_chain(self, graph: MaterialDepGraph) -> None:
        """Chain of material dependencies."""
        depth = 10
        materials = [Path(f"materials/level_{i}.wgsl") for i in range(depth)]

        for mat in materials:
            graph.record_material_compilation(mat, set())

        for i in range(1, depth):
            graph.record_material_dependency(materials[i], materials[i - 1])

        # Changing level_0 should affect all materials
        affected = graph.broadest_invalidation_set(materials[0])
        assert len(affected) == depth


# =============================================================================
# Suite D: Concurrent Access
# =============================================================================


class TestConcurrentAccess:
    """Tests for thread-safe concurrent access."""

    def test_concurrent_reads(self, graph: MaterialDepGraph) -> None:
        """Basic threading test: concurrent reads don't deadlock or corrupt."""
        # Populate graph first
        for i in range(100):
            graph.record_material_compilation(
                Path(f"materials/mat_{i}.wgsl"),
                {Path(f"includes/inc_{j}.wgsl") for j in range(5)}
            )

        num_threads = 10
        reads_per_thread = 1000
        read_counts: list[int] = []
        errors: list[Exception] = []

        def read_operations() -> int:
            count = 0
            try:
                for _ in range(reads_per_thread):
                    result = graph.get_include_materials(Path("includes/inc_0.wgsl"))
                    # Verify result is consistent (100 materials use this include)
                    assert len(result) == 100
                    count += 1
            except Exception as e:
                errors.append(e)
            return count

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(read_operations) for _ in range(num_threads)]
            for future in as_completed(futures):
                read_counts.append(future.result())

        # All threads should complete all reads without errors
        assert not errors, f"Concurrent reads caused errors: {errors}"
        assert sum(read_counts) == num_threads * reads_per_thread

    def test_concurrent_reads_and_writes(self, graph: MaterialDepGraph) -> None:
        """Concurrent reads and writes don't corrupt state."""
        num_threads = 8
        operations_per_thread = 500
        errors: list[Exception] = []

        def mixed_operations(thread_id: int) -> None:
            try:
                for i in range(operations_per_thread):
                    mat = Path(f"materials/t{thread_id}_m{i}.wgsl")
                    includes = {
                        Path(f"includes/shared.wgsl"),
                        Path(f"includes/t{thread_id}.wgsl"),
                    }

                    # Write
                    graph.record_material_compilation(mat, includes)

                    # Read
                    graph.get_include_materials(Path("includes/shared.wgsl"))
                    graph.broadest_invalidation_set(Path("includes/shared.wgsl"))

            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=mixed_operations, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent operations caused errors: {errors}"

        # Verify graph integrity
        assert graph.material_count() == num_threads * operations_per_thread

    def test_concurrent_invalidation(self, graph: MaterialDepGraph) -> None:
        """Concurrent BFS invalidation calls are safe."""
        # Set up a complex graph
        for i in range(50):
            graph.record_material_compilation(
                Path(f"materials/mat_{i}.wgsl"),
                {Path(f"includes/common.wgsl"), Path(f"includes/inc_{i % 10}.wgsl")}
            )

        results: list[set[Path]] = []
        lock = threading.Lock()

        def invalidate_operation() -> None:
            affected = graph.broadest_invalidation_set(Path("includes/common.wgsl"))
            with lock:
                results.append(affected)

        threads = [
            threading.Thread(target=invalidate_operation)
            for _ in range(20)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be identical
        first_result = results[0]
        assert len(first_result) == 50
        for result in results[1:]:
            assert result == first_result


# =============================================================================
# Suite E: Graph Clearing and Re-recording
# =============================================================================


class TestClearingAndRerecording:
    """Tests for clearing and re-recording graph data."""

    def test_clear_empties_graph(self, graph: MaterialDepGraph) -> None:
        """Clear removes all edges."""
        graph.record_material_compilation(
            Path("materials/test.wgsl"),
            {Path("includes/a.wgsl"), Path("includes/b.wgsl")}
        )
        graph.record_material_dependency(
            Path("materials/test.wgsl"), Path("materials/base.wgsl")
        )

        graph.clear()

        assert graph.material_count() == 0
        assert graph.include_count() == 0

    def test_rerecord_after_clear(self, graph: MaterialDepGraph) -> None:
        """Graph works correctly after clear and re-record."""
        # Initial recording
        graph.record_material_compilation(
            Path("materials/old.wgsl"), {Path("includes/old.wgsl")}
        )

        # Clear
        graph.clear()

        # Re-record different data
        graph.record_material_compilation(
            Path("materials/new.wgsl"), {Path("includes/new.wgsl")}
        )

        assert graph.material_count() == 1
        affected = graph.broadest_invalidation_set(Path("includes/new.wgsl"))
        assert Path("materials/new.wgsl").resolve() in affected

    def test_remove_single_material(self, graph: MaterialDepGraph) -> None:
        """Remove a single material from the graph."""
        graph.record_material_compilation(
            Path("materials/a.wgsl"), {Path("includes/shared.wgsl")}
        )
        graph.record_material_compilation(
            Path("materials/b.wgsl"), {Path("includes/shared.wgsl")}
        )

        graph.remove_material(Path("materials/a.wgsl"))

        assert graph.material_count() == 1
        shared_users = graph.get_include_materials(Path("includes/shared.wgsl"))
        assert len(shared_users) == 1
        assert Path("materials/b.wgsl").resolve() in shared_users

    def test_remove_cleans_dependencies(self, graph: MaterialDepGraph) -> None:
        """Removing material cleans up dependency edges."""
        graph.record_material_compilation(Path("materials/a.wgsl"), set())
        graph.record_material_compilation(Path("materials/b.wgsl"), set())
        graph.record_material_dependency(
            Path("materials/b.wgsl"), Path("materials/a.wgsl")
        )

        graph.remove_material(Path("materials/b.wgsl"))

        # a.wgsl should have no dependents
        dependents = graph.get_material_dependents(Path("materials/a.wgsl"))
        assert len(dependents) == 0


# =============================================================================
# Suite F: Edge Iteration
# =============================================================================


class TestEdgeIteration:
    """Tests for edge iteration functionality."""

    def test_iterate_edges(self, graph: MaterialDepGraph) -> None:
        """Iterate over all edges in the graph."""
        graph.record_material_compilation(
            Path("materials/a.wgsl"),
            {Path("includes/1.wgsl"), Path("includes/2.wgsl")}
        )
        graph.record_material_dependency(
            Path("materials/b.wgsl"), Path("materials/a.wgsl")
        )

        edges = list(graph.edges())

        include_edges = [e for e in edges if e[2] == "include"]
        depends_edges = [e for e in edges if e[2] == "depends"]

        assert len(include_edges) == 2
        assert len(depends_edges) == 1

    def test_all_materials(self, graph: MaterialDepGraph) -> None:
        """Get all materials in the graph."""
        materials = {Path(f"materials/mat_{i}.wgsl") for i in range(5)}
        for mat in materials:
            graph.record_material_compilation(mat, set())

        result = graph.all_materials()
        assert len(result) == 5

    def test_all_includes(self, graph: MaterialDepGraph) -> None:
        """Get all includes in the graph."""
        includes = {Path(f"includes/inc_{i}.wgsl") for i in range(3)}
        graph.record_material_compilation(Path("materials/test.wgsl"), includes)

        result = graph.all_includes()
        assert len(result) == 3


# =============================================================================
# Suite G: Repr and String Output
# =============================================================================


class TestStringRepresentation:
    """Tests for string representation."""

    def test_repr_empty(self, graph: MaterialDepGraph) -> None:
        """Repr of empty graph."""
        result = repr(graph)
        assert "materials=0" in result
        assert "includes=0" in result

    def test_repr_populated(self, graph: MaterialDepGraph) -> None:
        """Repr of populated graph."""
        graph.record_material_compilation(
            Path("materials/test.wgsl"),
            {Path("includes/a.wgsl"), Path("includes/b.wgsl")}
        )

        result = repr(graph)
        assert "materials=1" in result
        assert "includes=2" in result
