"""Tests for incremental re-parsing (P2 Phase 2.3.4).

Covers:
- Detecting changed files via mtime comparison
- Re-parsing only changed files
- Preserving user-adjusted node positions for unchanged nodes
"""

from __future__ import annotations

import os
import time
import textwrap
from pathlib import Path

import pytest

from flowforge_backend.ast_parser.graph_types import (
    GraphNode,
    GraphEdge,
    NodeGraph,
    NodePosition,
    SourceLocation,
)
from flowforge_backend.ast_parser.incremental import (
    IncrementalParser,
    detect_changed_files,
    reparse_changed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMPONENT_SOURCE = textwrap.dedent("""\
    from trinity import component

    @component
    class Position:
        x: float = 0.0
        y: float = 0.0
""")

SYSTEM_SOURCE = textwrap.dedent("""\
    from trinity import system

    @system
    class Movement:
        speed: float = 1.0
""")

UPDATED_COMPONENT_SOURCE = textwrap.dedent("""\
    from trinity import component

    @component
    class Position:
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0
""")


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests: detect_changed_files
# ---------------------------------------------------------------------------


class TestDetectChangedFiles:
    """Tests for detect_changed_files()."""

    def test_all_new_files_detected(self, tmp_path: Path) -> None:
        """With empty previous state every file should be reported."""
        _write_file(tmp_path / "a.py", COMPONENT_SOURCE)
        _write_file(tmp_path / "b.py", SYSTEM_SOURCE)

        changed = detect_changed_files(str(tmp_path), {})
        assert len(changed) == 2

    def test_unchanged_files_not_reported(self, tmp_path: Path) -> None:
        """Files whose mtime matches previous state should be skipped."""
        file_a = tmp_path / "a.py"
        _write_file(file_a, COMPONENT_SOURCE)

        mtime = os.path.getmtime(str(file_a))
        previous = {str(file_a): mtime}

        changed = detect_changed_files(str(tmp_path), previous)
        assert changed == []

    def test_modified_file_detected(self, tmp_path: Path) -> None:
        """A file with a newer mtime should be reported as changed."""
        file_a = tmp_path / "a.py"
        _write_file(file_a, COMPONENT_SOURCE)

        # Record mtime, then bump it
        old_mtime = os.path.getmtime(str(file_a))
        previous = {str(file_a): old_mtime}

        # Ensure mtime advances (some filesystems have 1s granularity)
        time.sleep(0.05)
        _write_file(file_a, UPDATED_COMPONENT_SOURCE)

        changed = detect_changed_files(str(tmp_path), previous)
        assert str(file_a) in changed

    def test_nonexistent_directory_returns_empty(self) -> None:
        changed = detect_changed_files("/nonexistent/path", {})
        assert changed == []


# ---------------------------------------------------------------------------
# Tests: reparse_changed
# ---------------------------------------------------------------------------


class TestReparseChanged:
    """Tests for reparse_changed()."""

    def test_no_changes_returns_previous(self, tmp_path: Path) -> None:
        """When changed_files is empty, previous graph is returned as-is."""
        prev = NodeGraph(nodes=[], edges=[])
        result = reparse_changed(str(tmp_path), prev, [])
        assert result is prev

    def test_changed_file_produces_new_graph(self, tmp_path: Path) -> None:
        """Re-parsing a changed file should produce an updated graph."""
        _write_file(tmp_path / "comp.py", COMPONENT_SOURCE)

        # Build an initial (empty-ish) previous graph
        prev = NodeGraph(nodes=[], edges=[])

        result = reparse_changed(
            str(tmp_path), prev, [str(tmp_path / "comp.py")]
        )
        # The new graph should contain the Position component node
        names = [n.name for n in result.nodes]
        assert "Position" in names


# ---------------------------------------------------------------------------
# Tests: position preservation
# ---------------------------------------------------------------------------


class TestPositionPreservation:
    """Unchanged nodes should keep their user-adjusted positions."""

    def test_positions_preserved_for_unchanged_nodes(self, tmp_path: Path) -> None:
        """Nodes from unchanged files should retain their previous position."""
        comp_file = tmp_path / "comp.py"
        sys_file = tmp_path / "sys.py"
        _write_file(comp_file, COMPONENT_SOURCE)
        _write_file(sys_file, SYSTEM_SOURCE)

        # Do a full initial parse
        parser = IncrementalParser()
        graph = parser.parse_directory(str(tmp_path))

        # Find the Position node and give it a custom position
        position_node = None
        for node in graph.nodes:
            if node.name == "Position":
                position_node = node
                break

        assert position_node is not None, "Position node should exist"

        # Simulate user dragging the node
        custom_pos = NodePosition(x=999.0, y=888.0)
        position_node.position = custom_pos

        # Now modify only sys.py so comp.py is unchanged
        time.sleep(0.05)
        _write_file(sys_file, textwrap.dedent("""\
            from trinity import system

            @system
            class Movement:
                speed: float = 2.0
                accel: float = 0.5
        """))

        # Re-parse incrementally.  Since IncrementalParser already
        # recorded mtimes, the second call detects sys.py as changed.
        graph2 = parser.parse_directory(str(tmp_path))

        # Find Position node in the new graph
        pos_node2 = None
        for node in graph2.nodes:
            if node.name == "Position":
                pos_node2 = node
                break

        assert pos_node2 is not None, "Position node should still exist"
        # The preserved position should match what was set
        assert pos_node2.position.x == custom_pos.x
        assert pos_node2.position.y == custom_pos.y


# ---------------------------------------------------------------------------
# Tests: IncrementalParser
# ---------------------------------------------------------------------------


class TestIncrementalParser:
    """Tests for the IncrementalParser class."""

    def test_first_parse_is_full(self, tmp_path: Path) -> None:
        _write_file(tmp_path / "a.py", COMPONENT_SOURCE)

        parser = IncrementalParser()
        assert parser.previous_graph is None

        graph = parser.parse_directory(str(tmp_path))
        assert parser.previous_graph is not None
        assert len(graph.nodes) > 0

    def test_reset_clears_state(self, tmp_path: Path) -> None:
        _write_file(tmp_path / "a.py", COMPONENT_SOURCE)

        parser = IncrementalParser()
        parser.parse_directory(str(tmp_path))
        assert parser.previous_graph is not None

        parser.reset()
        assert parser.previous_graph is None
        assert parser.file_mtimes == {}
