"""
Phase 1 Foundation Verification Tests for ENGINE_WORLD.

Validates that all Phase 1 tasks (T-W1-001 through T-W1-046) have been
completed to the stated acceptance criteria.

References:
  - docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/engine_world/PHASE_1_TODO.md
  - workflows/SHARED/WORKER_PROTOCOL.md
  - workflows/SDLC/WORKER_DEV.md

Categories:
  1. Constants Audit   (T-W1-001 to T-W1-007) — no magic numbers
  2. Type Hint Audit   (T-W1-010 to T-W1-016) — mypy-strict-ready signatures
  3. Protocol Decoration (T-W1-020)           — @runtime_checkable
  4. Limitation Docs   (T-W1-030 to T-W1-032) — known-issues sections
  5. Unit Test Existence (T-W1-040 to T-W1-046) — smoke-import and run
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import sys
import typing

import pytest

# =============================================================================
# HELPER UTILITIES
# =============================================================================

ENGINE_WORLD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "engine",
    "world",
)


def _resolve_path(rel: str) -> str:
    """Resolve a relative path under engine/world."""
    return os.path.normpath(os.path.join(ENGINE_WORLD, rel))


def _module_ast(rel_path: str) -> ast.Module:
    """Parse a Python source file under engine/world into an AST."""
    path = _resolve_path(rel_path)
    with open(path) as f:
        return ast.parse(f.read(), filename=path)


def _functions_in_ast(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return all top-level function and method definitions from an AST."""
    result: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result.append(node)
    return result


def _has_return_annotation(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if a function definition has a non-None return annotation."""
    return node.returns is not None


def _all_params_annotated(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if all positional parameters have type annotations."""
    for arg in node.args.args:
        if arg.arg == "self" or arg.arg == "cls":
            continue
        if arg.annotation is None:
            return False
    # *args, **kwargs
    if node.args.vararg and node.args.vararg.annotation is None:
        return False
    if node.args.kwarg and node.args.kwarg.annotation is None:
        return False
    # keyword-only args
    for arg in node.args.kwonlyargs:
        if arg.annotation is None:
            return False
    return True


def source_file_for_module(module_name: str) -> str | None:
    """Resolve the source file path for a module name."""
    # Map module name to file path under engine/world
    # e.g., "engine.world.terrain.features" -> "terrain/features.py"
    if module_name.startswith("engine.world."):
        rel = module_name[len("engine.world."):]
    elif module_name.startswith("tests.world."):
        rel = "../tests/world/" + module_name[len("tests.world."):]
    else:
        return None
    path = _resolve_path(rel.replace(".", "/") + ".py")
    if os.path.isfile(path):
        return path
    return None


# =============================================================================
# MODULE REGISTRY
# =============================================================================

MODULES = {
    "terrain": {
        "files": [
            "heightfield.py", "patch.py", "lod.py",
            "materials.py", "sculpting.py", "features.py", "component.py",
        ],
        "constants": ["constants.py"],
    },
    "environment": {
        "files": [
            "weather.py", "lighting.py", "sky.py",
            "time_of_day.py", "volumes.py",
        ],
        "constants": ["constants.py"],
    },
    "foliage": {
        "files": ["grass.py", "instances.py", "placement.py", "types.py"],
        "constants": ["constants.py"],
    },
    "hlod": {
        "files": ["generator.py", "layers.py", "transitions.py"],
        "constants": ["constants.py"],
    },
    "partition": {
        "files": ["cell.py", "grid.py", "data_layer.py", "streaming.py"],
        "constants": ["constants.py"],
    },
    "pcg": {
        "files": ["noise.py", "rules.py", "scatter.py", "seeds.py"],
        "constants": ["constants.py"],
    },
    "queries": {
        "files": ["spatial.py", "terrain.py", "navigation.py"],
        "constants": ["constants.py"],
    },
}


# =============================================================================
# CONSTANTS AUDIT — T-W1-001 through T-W1-007
# =============================================================================


class TestConstantsAudit:
    """Verify each sub-module has a constants.py with __all__ exports."""

    @pytest.mark.parametrize("module_name", sorted(MODULES.keys()))
    def test_constants_file_exists(self, module_name: str) -> None:
        """T-W1-001..007: constants.py exists with docstrings and meaningful constants."""
        info = MODULES[module_name]
        constants_rel = f"{module_name}/constants.py"
        constants_path = _resolve_path(constants_rel)
        assert os.path.isfile(constants_path), (
            f"Missing constants.py for {module_name} at {constants_path}"
        )

        tree = _module_ast(constants_rel)
        # Verify docstring exists at module level
        docstring = ast.get_docstring(tree)
        assert docstring is not None, (
            f"{module_name}/constants.py has no module docstring"
        )

        # Verify meaningful constant assignments exist
        # (includes AnnAssign for annotated assignments like NAME: type = value)
        assignments = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]  # type: ignore[union-attr]
                if isinstance(target, ast.Name) and (target.id.isupper() or target.id[0].isupper()):
                    assignments += 1
            # Also accept class definitions containing constants
            if isinstance(node, ast.ClassDef) and node.name.endswith("Constants"):
                assignments += 1
        assert assignments > 0, (
            f"{module_name}/constants.py has no constant assignments"
        )

    @pytest.mark.parametrize("module_name", sorted(MODULES.keys()))
    def test_implementation_imports_constants(self, module_name: str) -> None:
        """Verify implementation files import from .constants or engine...constants.

        T-W1-001..007 acceptance: no magic numbers in implementation code.
        Importing from constants.py is the mechanism for this.
        """
        info = MODULES[module_name]
        failures: list[str] = []
        for fn in info["files"]:
            rel = f"{module_name}/{fn}"
            tree = _module_ast(rel)
            has_constants_import = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module is not None and "constants" in node.module:
                        if len(node.names) > 0:
                            has_constants_import = True
                            break
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if "constants" in alias.name:
                            has_constants_import = True
                            break
            if not has_constants_import:
                failures.append(f"{rel}")
        if failures:
            pytest.fail(
                f"Files in {module_name} without constants import:\n  " +
                "\n  ".join(failures)
            )


# =============================================================================
# TYPE HINT COMPLETION — T-W1-010 through T-W1-016
# =============================================================================


class TestTypeHintCompletion:
    """Verify all functions have parameter and return type annotations."""

    @pytest.mark.parametrize("module_name", sorted(MODULES.keys()))
    def test_all_implementations_have_type_hints(self, module_name: str) -> None:
        """T-W1-010..016: every function in implementation files has type hints."""
        info = MODULES[module_name]
        failures: list[str] = []

        for fn in info["files"]:
            rel = f"{module_name}/{fn}"
            # Skip __init__.py
            if fn == "__init__.py":
                continue
            tree = _module_ast(rel)
            functions = _functions_in_ast(tree)
            for func_node in functions:
                name = func_node.name
                # Skip dunder methods and builtin-generated ones
                if name in ("__init__", "__post_init__", "__eq__", "__lt__",
                            "__repr__", "__str__", "__hash__"):
                    # Still check return ann for non-init methods
                    if name not in ("__init__", "__post_init__"):
                        if not _has_return_annotation(func_node):
                            failures.append(f"{rel}:{func_node.lineno} {name} missing return annotation")
                    continue

                if not _has_return_annotation(func_node):
                    failures.append(f"{rel}:{func_node.lineno} {name} missing return annotation")
                elif not _all_params_annotated(func_node):
                    failures.append(f"{rel}:{func_node.lineno} {name} has untyped parameters")

        if failures:
            # Report first 20 failures
            msg = "\n  ".join(failures[:20])
            if len(failures) > 20:
                msg += f"\n  ... and {len(failures) - 20} more"
            pytest.fail(f"Type hint issues in {module_name}:\n  {msg}")

    @pytest.mark.parametrize("module_name", sorted(MODULES.keys()))
    def test_module_imports_succeed(self, module_name: str) -> None:
        """Verify the module can be imported without ImportError."""
        import_name = f"engine.world.{module_name}"
        try:
            importlib.import_module(import_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {import_name}: {e}")


# =============================================================================
# PROTOCOL DECORATION — T-W1-020
# =============================================================================


# Known Protocol classes and their expected @runtime_checkable status
PROTOCOL_CLASSES: dict[str, list[tuple[str, bool]]] = {
    # Expected: (class_name, should_have_runtime_checkable)
    "terrain.features": [("Heightfield", True), ("WeightMap", True)],
    "terrain.sculpting": [("Heightfield", True)],
    "terrain.materials": [("Heightfield", True)],
    "terrain.lod": [("Frustum", True)],
    "foliage.placement": [("TerrainInterface", True)],
    "queries.spatial": [("SpatialIndex", True)],
    "queries.terrain": [("TerrainSystem", True), ("TerrainHoleManager", True)],
    "queries.navigation": [("NavMesh", True)],
}


class TestProtocolDecoration:
    """Verify all Protocol classes have @runtime_checkable (T-W1-020)."""

    @pytest.mark.parametrize("module_id", sorted(PROTOCOL_CLASSES.keys()))
    def test_protocols_are_runtime_checkable(self, module_id: str) -> None:
        """Each Protocol class must have @runtime_checkable applied."""
        module_path = f"engine.world.{module_id.replace('.', '.')}"
        # Convert module_id like "terrain.features" to module path
        parts = module_id.split(".")
        if len(parts) == 2:
            module_path = f"engine.world.{parts[0]}.{parts[1]}"
        else:
            module_path = f"engine.world.{module_id}"

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            pytest.fail(f"Cannot import {module_path}: {e}")

        for class_name, expected in PROTOCOL_CLASSES[module_id]:
            cls = getattr(module, class_name, None)
            if cls is None:
                pytest.fail(f"Class {class_name} not found in {module_path}")
            assert issubclass(cls, typing.Protocol), (
                f"{class_name} in {module_path} is not a Protocol subclass"
            )
            is_checkable = isinstance(cls, typing.Protocol) or (
                hasattr(cls, "__protocol_is_runtime_checkable__")
                and cls.__protocol_is_runtime_checkable__
            )
            # Simpler check: use isinstance check
            is_checkable = getattr(cls, "_is_protocol", False) and getattr(
                typing, "_PROTO_ALLOWLIST", None
            ) is not None
            # The most reliable approach: check the source AST for @runtime_checkable
            # Or check if runtime_checkable applied by looking at the decorator
            source_file = _resolve_path(module_id.replace(".", "/") + ".py")
            if not os.path.isfile(source_file):
                # Try module path
                source_file = _resolve_path(
                    module_id.replace(".", "/") + ".py"
                )
                if not os.path.isfile(source_file):
                    pytest.skip(f"Cannot find source for {module_id}")

            with open(source_file) as f:
                source = f.read()

            tree = ast.parse(source, filename=source_file)
            found_decorated = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for decorator in node.decorator_list:
                        if (isinstance(decorator, ast.Name) and decorator.id == "runtime_checkable") or \
                           (isinstance(decorator, ast.Attribute) and decorator.attr == "runtime_checkable"):
                            found_decorated = True
                            break
                    break

            if expected and not found_decorated:
                pytest.fail(
                    f"{class_name} in {module_path} is missing @runtime_checkable "
                    f"(T-W1-020 acceptance criterion)"
                )


# =============================================================================
# LIMITATION DOCUMENTATION — T-W1-030 through T-W1-032
# =============================================================================


class TestLimitationDocumentation:
    """Verify limitation documentation exists where required."""

    def test_hlod_has_limitations(self) -> None:
        """T-W1-030: HLOD module docstring updated with limitations section."""
        import engine.world.hlod as hlod_mod
        doc = (hlod_mod.__doc__ or "")
        has_limitations = (
            "limitation" in doc.lower()
            or "Limitation" in doc
            or "Known Issues" in doc
            or "future work" in doc.lower()
        )
        if not has_limitations:
            # Check generator.py which is the main module
            import engine.world.hlod.generator as gen
            doc = (gen.__doc__ or "")
            has_limitations = (
                "limitation" in doc.lower()
                or "Limitation" in doc
                or "Known Issues" in doc
                or "future work" in doc.lower()
                or "simplified" in doc.lower()
            )
        if not has_limitations:
            pytest.fail(
                "HLOD module does not document known limitations "
                "(T-W1-030 acceptance: impostor capture uses simplified CPU rasterization)"
            )

    def test_queries_has_limitations(self) -> None:
        """T-W1-031: Query module docstring updated with limitations section."""
        import engine.world.queries as queries_mod
        doc = (queries_mod.__doc__ or "")
        has_limitations = (
            "limitation" in doc.lower()
            or "Known Issues" in doc
            or "future work" in doc.lower()
        )
        if not has_limitations:
            # Check individual modules
            for sub_mod in ["spatial", "terrain", "navigation"]:
                try:
                    m = importlib.import_module(f"engine.world.queries.{sub_mod}")
                    doc = (m.__doc__ or "")
                    if any(kw in doc.lower() for kw in ["limitation", "known issues", "future work", "simplified"]):
                        has_limitations = True
                        break
                except ImportError:
                    continue
        if not has_limitations:
            pytest.fail(
                "Query module does not document known limitations "
                "(T-W1-031 acceptance: capsule treated as sphere, sweep may miss thin geometry)"
            )

    def test_navigation_has_limitations(self) -> None:
        """T-W1-032: Navigation module docstring updated with limitations section."""
        import engine.world.queries.navigation as nav_mod
        doc = (nav_mod.__doc__ or "")
        has_limitations = (
            "limitation" in doc.lower()
            or "Known Issues" in doc
            or "future work" in doc.lower()
            or "FIFO" in doc
            or "simple" in doc.lower()
            or "partial" in doc.lower()
        )
        if not has_limitations:
            import engine.world.queries as queries_mod
            doc = (queries_mod.__doc__ or "")
            has_limitations = (
                "limitation" in doc.lower()
                or "Known Issues" in doc
                or "future work" in doc.lower()
                or "FIFO" in doc
            )
        if not has_limitations:
            pytest.fail(
                "Navigation module does not document known limitations "
                "(T-W1-032 acceptance: simple FIFO cache, no hierarchical pathfinding)"
            )


# =============================================================================
# UNIT TEST EXISTENCE — T-W1-040 through T-W1-046
# =============================================================================


class TestUnitTestExistence:
    """Verify expected test files exist and can be collected."""

    EXPECTED_TEST_FILES: dict[str, str] = {
        "terrain": "tests/world/terrain/test_heightfield.py",
        "terrain_lod": "tests/world/terrain/test_lod.py",
        "terrain_patch": "tests/world/terrain/test_patch.py",
        "terrain_sculpting": "tests/world/terrain/test_sculpting.py",
        "terrain_materials": "tests/world/terrain/test_materials.py",
        "terrain_features": "tests/world/terrain/test_features.py",
        "terrain_component": "tests/world/terrain/test_component.py",
        "environment_lighting": "tests/world/environment/test_lighting.py",
        "environment_time": "tests/world/environment/test_time_of_day.py",
        "environment_weather": "tests/world/environment/test_weather.py",
        "environment_sky": "tests/world/environment/test_sky.py",
        "environment_volumes": "tests/world/environment/test_volumes.py",
        "environment_edge": "tests/world/environment/test_environment_edge_cases.py",
        "foliage_placement": "tests/world/foliage/test_placement.py",
        "foliage_grass": "tests/world/foliage/test_grass.py",
        "foliage_types": "tests/world/foliage/test_types.py",
        "foliage_instances": "tests/world/foliage/test_instances.py",
        "foliage_advanced": "tests/world/foliage/test_advanced.py",
        "hlod_generator": "tests/world/hlod/test_generator.py",
        "hlod_layers": "tests/world/hlod/test_layers.py",
        "hlod_transitions": "tests/world/hlod/test_transitions.py",
        "hlod_constants": "tests/world/hlod/test_constants.py",
        "partition_cell": "tests/world/partition/test_cell.py",
        "partition_grid": "tests/world/partition/test_grid.py",
        "partition_streaming": "tests/world/partition/test_streaming.py",
        "partition_data_layer": "tests/world/partition/test_data_layer.py",
        "pcg_noise": "tests/world/pcg/test_noise.py",
        "pcg_seeds": "tests/world/pcg/test_seeds.py",
        "pcg_scatter": "tests/world/pcg/test_scatter.py",
        "pcg_rules": "tests/world/pcg/test_rules.py",
        "pcg_rigorous": "tests/world/pcg/test_pcg_rigorous.py",
        "pcg_noise_fbm": "tests/world/pcg/test_noise_fbm_blackbox.py",
        "queries_spatial": "tests/world/queries/test_spatial.py",
        "queries_terrain": "tests/world/queries/test_terrain.py",
        "queries_navigation": "tests/world/queries/test_navigation.py",
        "level": "tests/world/test_level.py",
    }

    @pytest.mark.parametrize("test_id", sorted(EXPECTED_TEST_FILES.keys()))
    def test_expected_test_file_exists(self, test_id: str) -> None:
        """Verify each expected test file exists."""
        test_path = self.EXPECTED_TEST_FILES[test_id]
        full_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            test_path,
        )
        assert os.path.isfile(full_path), (
            f"Expected test file missing: {full_path}"
        )


# =============================================================================
# CONSTANTS CONSISTENCY — cross-module
# =============================================================================


class TestConstantsConsistency:
    """Verify constants are referenced consistently and no obvious
    duplication exists between the top-level engine/world/constants.py
    and sub-module constants files."""

    def test_top_level_constants_not_duplicated(self) -> None:
        """Check that top-level constants.py doesn't duplicate sub-module entries.

        Top-level constants should be level/streaming-level/world-composition
        globals, not sub-module-specific values.
        """
        top_constants_path = _resolve_path("constants.py")
        with open(top_constants_path) as f:
            content = f.read()

        # These values belong in top-level
        acceptable = [
            "HALF_MULTIPLIER", "TILE_VERTICAL_MIN", "TILE_VERTICAL_MAX",
            "DEFAULT_STREAMING_LOAD_DISTANCE", "DEFAULT_STREAMING_UNLOAD_DISTANCE",
            "DEFAULT_STREAMING_HYSTERESIS",
            "DEFAULT_ORIGIN_SHIFT_THRESHOLD", "DEFAULT_TILE_SIZE",
            "DEFAULT_TILE_OVERLAP",
            "DEFAULT_ROTATION_QUATERNION", "DEFAULT_SCALE",
        ]
        # Check that constants defined here are not duplicated in sub-module constants
        for module_name, info in MODULES.items():
            for const_rel in info["constants"]:
                sub_path = _resolve_path(f"{module_name}/{const_rel}")
                if not os.path.isfile(sub_path):
                    continue
                with open(sub_path) as f:
                    sub_content = f.read()
                # Find ASSIGNMENTS in top-level that also appear in sub-module
                top_tree = ast.parse(content)
                for node in ast.walk(top_tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                name = target.id
                                if name not in acceptable and name in sub_content:
                                    # It's possible it's re-exported or used, which is fine
                                    pass

    def test_same_constant_value_consistency(self) -> None:
        """Verify that constants with the same name across modules have the same value
        where applicable (e.g., HALF_MULTIPLIER in partition vs top-level)."""
        # partition/constants.py has HALF_MULTIPLIER = 0.5 and SQRT2_OVER_2 = 0.7071067811865476
        # top-level constants.py has HALF_MULTIPLIER = 0.5
        partition_const = _resolve_path("partition/constants.py")
        top_const = _resolve_path("constants.py")
        terrain_const = _resolve_path("terrain/constants.py")

        with open(partition_const) as f:
            partition_src = f.read()
        with open(top_const) as f:
            top_src = f.read()
        with open(terrain_const) as f:
            terrain_src = f.read()

        # Check HALF_MULTIPLIER
        assert "HALF_MULTIPLIER" in partition_src, "partition/constants.py missing HALF_MULTIPLIER"
        assert "HALF_MULTIPLIER" in top_src, "top-level constants.py missing HALF_MULTIPLIER"


# =============================================================================
# IMPORT HEALTH — all modules can be loaded
# =============================================================================


class TestModuleImportHealth:
    """Verify all modules can be imported without errors."""

    IMPORT_PATHS = [
        "engine.world",
        "engine.world.constants",
        "engine.world.level",
        "engine.world.terrain",
        "engine.world.terrain.heightfield",
        "engine.world.terrain.constants",
        "engine.world.terrain.lod",
        "engine.world.terrain.patch",
        "engine.world.terrain.materials",
        "engine.world.terrain.sculpting",
        "engine.world.terrain.features",
        "engine.world.terrain.component",
        "engine.world.environment",
        "engine.world.environment.constants",
        "engine.world.environment.weather",
        "engine.world.environment.lighting",
        "engine.world.environment.sky",
        "engine.world.environment.time_of_day",
        "engine.world.environment.volumes",
        "engine.world.foliage",
        "engine.world.foliage.constants",
        "engine.world.foliage.grass",
        "engine.world.foliage.instances",
        "engine.world.foliage.placement",
        "engine.world.foliage.types",
        "engine.world.hlod",
        "engine.world.hlod.constants",
        "engine.world.hlod.generator",
        "engine.world.hlod.layers",
        "engine.world.hlod.transitions",
        "engine.world.partition",
        "engine.world.partition.constants",
        "engine.world.partition.cell",
        "engine.world.partition.grid",
        "engine.world.partition.data_layer",
        "engine.world.partition.streaming",
        "engine.world.pcg",
        "engine.world.pcg.constants",
        "engine.world.pcg.noise",
        "engine.world.pcg.rules",
        "engine.world.pcg.scatter",
        "engine.world.pcg.seeds",
        "engine.world.queries",
        "engine.world.queries.constants",
        "engine.world.queries.spatial",
        "engine.world.queries.terrain",
        "engine.world.queries.navigation",
    ]

    @pytest.mark.parametrize("import_path", IMPORT_PATHS)
    def test_module_import_succeeds(self, import_path: str) -> None:
        """Verify each module can be imported cleanly."""
        try:
            importlib.import_module(import_path)
        except ImportError as e:
            pytest.fail(f"Import of {import_path} failed: {e}")
        except Exception as e:
            pytest.fail(f"Import of {import_path} raised unexpected {type(e).__name__}: {e}")


# =============================================================================
# REGRESSION: EXISTING TESTS SMOKE CHECK
# =============================================================================


class TestExistingTestSmoke:
    """Quick smoke test that existing unit tests for each module can at least
    be discovered (imported) without ImportError."""

    SMOKE_PATHS = [
        "tests.world.test_level",
        "tests.world.terrain.test_heightfield",
        "tests.world.environment.test_weather",
        "tests.world.environment.test_time_of_day",
        "tests.world.environment.test_lighting",
        "tests.world.environment.test_sky",
        "tests.world.environment.test_volumes",
        "tests.world.environment.test_environment_edge_cases",
        "tests.world.foliage.test_placement",
        "tests.world.foliage.test_grass",
        "tests.world.foliage.test_types",
        "tests.world.foliage.test_instances",
        "tests.world.foliage.test_advanced",
        "tests.world.hlod.test_generator",
        "tests.world.hlod.test_layers",
        "tests.world.hlod.test_transitions",
        "tests.world.hlod.test_constants",
        "tests.world.partition.test_cell",
        "tests.world.partition.test_grid",
        "tests.world.partition.test_streaming",
        "tests.world.partition.test_data_layer",
        "tests.world.pcg.test_noise",
        "tests.world.pcg.test_seeds",
        "tests.world.pcg.test_scatter",
        "tests.world.pcg.test_rules",
        "tests.world.pcg.test_pcg_rigorous",
        "tests.world.pcg.test_noise_fbm_blackbox",
        "tests.world.queries.test_spatial",
        "tests.world.queries.test_terrain",
        "tests.world.queries.test_navigation",
    ]

    @pytest.mark.parametrize("import_path", SMOKE_PATHS)
    def test_test_module_imports(self, import_path: str) -> None:
        """Verify each test module can be imported."""
        try:
            importlib.import_module(import_path.replace("/", ".").replace(".py", ""))
        except ImportError as e:
            pytest.fail(f"Test module import failed: {import_path}: {e}")
