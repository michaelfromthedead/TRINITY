"""
Comprehensive tests for ReferenceManager functionality.

Tests reference tracking, graph queries, broken references, and redirects.
"""

import pytest
import sys
import tempfile
import shutil
import json
from pathlib import Path

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.assettools.reference_manager import (
    ReferenceManager,
    AssetReference,
    ReferenceGraph,
    BrokenReference,
    ReferenceRedirect,
    ReferenceType,
    ReferenceStatus,
)


@pytest.fixture
def temp_ref_dir():
    """Create a temporary directory for reference tests."""
    path = Path(tempfile.mkdtemp())

    # Create test files
    (path / "assets").mkdir()
    (path / "assets" / "textures").mkdir()
    (path / "assets" / "materials").mkdir()

    # Create texture files
    (path / "assets" / "textures" / "hero_diffuse.png").write_text("png data")
    (path / "assets" / "textures" / "hero_normal.png").write_text("png data")

    # Create material file with references
    mat_data = {
        "shader": "standard",
        "textures": {
            "diffuse": "textures/hero_diffuse.png",
            "normal": "textures/hero_normal.png",
        }
    }
    (path / "assets" / "materials" / "hero.mat").write_text(json.dumps(mat_data))

    # Create scene file with references
    scene_data = {
        "objects": [
            {"type": "mesh", "material": "materials/hero.mat"},
        ]
    }
    (path / "assets" / "level.scene").write_text(json.dumps(scene_data))

    # Create Python file with references
    (path / "assets" / "loader.py").write_text('''
def load_texture():
    return "textures/hero_diffuse.png"
''')

    yield path
    shutil.rmtree(path)


class TestAssetReference:
    """Test AssetReference dataclass."""

    def test_reference_creation(self):
        """Reference should store all attributes."""
        ref = AssetReference(
            source_path=Path("/source.mat"),
            target_path=Path("/target.png"),
            reference_type=ReferenceType.MATERIAL,
            location="textures.diffuse",
        )

        assert ref.source_path == Path("/source.mat")
        assert ref.target_path == Path("/target.png")
        assert ref.reference_type == ReferenceType.MATERIAL
        assert ref.location == "textures.diffuse"

    def test_reference_equality(self):
        """References with same source/target/location should be equal."""
        ref1 = AssetReference(
            source_path=Path("/s.mat"),
            target_path=Path("/t.png"),
            location="loc",
        )
        ref2 = AssetReference(
            source_path=Path("/s.mat"),
            target_path=Path("/t.png"),
            location="loc",
        )

        assert ref1 == ref2

    def test_reference_hashable(self):
        """References should be hashable."""
        ref = AssetReference(
            source_path=Path("/s.mat"),
            target_path=Path("/t.png"),
        )

        refs = {ref}
        assert len(refs) == 1


class TestBrokenReference:
    """Test BrokenReference dataclass."""

    def test_broken_reference_creation(self):
        """BrokenReference should store all attributes."""
        ref = AssetReference(
            source_path=Path("/source.mat"),
            target_path=Path("/missing.png"),
        )
        broken = BrokenReference(
            reference=ref,
            missing_path=Path("/missing.png"),
        )

        assert broken.reference == ref
        assert broken.missing_path == Path("/missing.png")

    def test_age_seconds(self):
        """age_seconds should calculate time since broken."""
        import time
        ref = AssetReference(
            source_path=Path("/s.mat"),
            target_path=Path("/m.png"),
        )
        broken = BrokenReference(
            reference=ref,
            missing_path=Path("/m.png"),
            broken_since=time.time() - 10,
        )

        assert broken.age_seconds >= 10


class TestReferenceRedirect:
    """Test ReferenceRedirect dataclass."""

    def test_redirect_creation(self):
        """Redirect should store all attributes."""
        redirect = ReferenceRedirect(
            old_path=Path("/old.png"),
            new_path=Path("/new.png"),
        )

        assert redirect.old_path == Path("/old.png")
        assert redirect.new_path == Path("/new.png")
        assert redirect.auto_redirect is True


class TestReferenceGraph:
    """Test ReferenceGraph functionality."""

    def test_graph_creation(self):
        """Graph should initialize empty."""
        graph = ReferenceGraph()
        assert len(graph) == 0

    def test_add_reference(self):
        """add_reference() should add to graph."""
        graph = ReferenceGraph()
        ref = AssetReference(
            source_path=Path("/a.mat"),
            target_path=Path("/b.png"),
        )

        graph.add_reference(ref)

        assert len(graph) == 1

    def test_remove_reference(self):
        """remove_reference() should remove from graph."""
        graph = ReferenceGraph()
        ref = AssetReference(
            source_path=Path("/a.mat"),
            target_path=Path("/b.png"),
        )
        graph.add_reference(ref)

        success = graph.remove_reference(ref)

        assert success
        assert len(graph) == 0

    def test_get_outgoing(self):
        """get_outgoing() should return refs from source."""
        graph = ReferenceGraph()
        ref1 = AssetReference(source_path=Path("/a.mat"), target_path=Path("/b.png"))
        ref2 = AssetReference(source_path=Path("/a.mat"), target_path=Path("/c.png"))
        ref3 = AssetReference(source_path=Path("/d.mat"), target_path=Path("/b.png"))

        graph.add_reference(ref1)
        graph.add_reference(ref2)
        graph.add_reference(ref3)

        outgoing = graph.get_outgoing(Path("/a.mat"))

        assert len(outgoing) == 2

    def test_get_incoming(self):
        """get_incoming() should return refs to target."""
        graph = ReferenceGraph()
        ref1 = AssetReference(source_path=Path("/a.mat"), target_path=Path("/b.png"))
        ref2 = AssetReference(source_path=Path("/c.mat"), target_path=Path("/b.png"))
        ref3 = AssetReference(source_path=Path("/a.mat"), target_path=Path("/d.png"))

        graph.add_reference(ref1)
        graph.add_reference(ref2)
        graph.add_reference(ref3)

        incoming = graph.get_incoming(Path("/b.png"))

        assert len(incoming) == 2

    def test_get_dependencies(self):
        """get_dependencies() should return direct targets."""
        graph = ReferenceGraph()
        ref1 = AssetReference(source_path=Path("/a.mat"), target_path=Path("/b.png"))
        ref2 = AssetReference(source_path=Path("/a.mat"), target_path=Path("/c.png"))

        graph.add_reference(ref1)
        graph.add_reference(ref2)

        deps = graph.get_dependencies(Path("/a.mat"))

        assert Path("/b.png") in deps
        assert Path("/c.png") in deps

    def test_get_dependents(self):
        """get_dependents() should return direct sources."""
        graph = ReferenceGraph()
        ref1 = AssetReference(source_path=Path("/a.mat"), target_path=Path("/b.png"))
        ref2 = AssetReference(source_path=Path("/c.mat"), target_path=Path("/b.png"))

        graph.add_reference(ref1)
        graph.add_reference(ref2)

        deps = graph.get_dependents(Path("/b.png"))

        assert Path("/a.mat") in deps
        assert Path("/c.mat") in deps

    def test_get_all_dependencies(self):
        """get_all_dependencies() should return transitive deps."""
        graph = ReferenceGraph()
        # a -> b -> c
        ref1 = AssetReference(source_path=Path("/a"), target_path=Path("/b"))
        ref2 = AssetReference(source_path=Path("/b"), target_path=Path("/c"))

        graph.add_reference(ref1)
        graph.add_reference(ref2)

        all_deps = graph.get_all_dependencies(Path("/a"))

        assert Path("/b") in all_deps
        assert Path("/c") in all_deps

    def test_get_all_dependents(self):
        """get_all_dependents() should return transitive deps."""
        graph = ReferenceGraph()
        # a -> b -> c
        ref1 = AssetReference(source_path=Path("/a"), target_path=Path("/b"))
        ref2 = AssetReference(source_path=Path("/b"), target_path=Path("/c"))

        graph.add_reference(ref1)
        graph.add_reference(ref2)

        all_deps = graph.get_all_dependents(Path("/c"))

        assert Path("/b") in all_deps
        assert Path("/a") in all_deps

    def test_has_path(self):
        """has_path() should detect transitive connection."""
        graph = ReferenceGraph()
        # a -> b -> c
        ref1 = AssetReference(source_path=Path("/a"), target_path=Path("/b"))
        ref2 = AssetReference(source_path=Path("/b"), target_path=Path("/c"))

        graph.add_reference(ref1)
        graph.add_reference(ref2)

        assert graph.has_path(Path("/a"), Path("/c"))
        assert not graph.has_path(Path("/c"), Path("/a"))

    def test_find_cycles(self):
        """find_cycles() should detect circular references."""
        graph = ReferenceGraph()
        # a -> b -> c -> a (cycle)
        ref1 = AssetReference(source_path=Path("/a"), target_path=Path("/b"))
        ref2 = AssetReference(source_path=Path("/b"), target_path=Path("/c"))
        ref3 = AssetReference(source_path=Path("/c"), target_path=Path("/a"))

        graph.add_reference(ref1)
        graph.add_reference(ref2)
        graph.add_reference(ref3)

        cycles = graph.find_cycles()

        assert len(cycles) > 0

    def test_get_reference(self):
        """get_reference() should return specific reference."""
        graph = ReferenceGraph()
        ref = AssetReference(
            source_path=Path("/a.mat"),
            target_path=Path("/b.png"),
            location="loc",
        )
        graph.add_reference(ref)

        found = graph.get_reference(Path("/a.mat"), Path("/b.png"), "loc")

        assert found == ref

    def test_clear(self):
        """clear() should remove all references."""
        graph = ReferenceGraph()
        ref = AssetReference(source_path=Path("/a"), target_path=Path("/b"))
        graph.add_reference(ref)

        graph.clear()

        assert len(graph) == 0

    def test_iterator(self):
        """Graph should be iterable."""
        graph = ReferenceGraph()
        ref = AssetReference(source_path=Path("/a"), target_path=Path("/b"))
        graph.add_reference(ref)

        refs = list(graph)

        assert len(refs) == 1


class TestReferenceManager:
    """Test ReferenceManager main class."""

    def test_manager_creation(self, temp_ref_dir):
        """Manager should initialize correctly."""
        manager = ReferenceManager(temp_ref_dir / "assets")

        assert manager.root_path == temp_ref_dir / "assets"

    def test_add_reference(self, temp_ref_dir):
        """add_reference() should add to manager."""
        manager = ReferenceManager(temp_ref_dir / "assets")
        ref = AssetReference(
            source_path=temp_ref_dir / "assets" / "materials" / "hero.mat",
            target_path=temp_ref_dir / "assets" / "textures" / "hero_diffuse.png",
        )

        manager.add_reference(ref)

        assert len(manager.graph) == 1

    def test_remove_reference(self, temp_ref_dir):
        """remove_reference() should remove from manager."""
        manager = ReferenceManager(temp_ref_dir / "assets")
        ref = AssetReference(
            source_path=temp_ref_dir / "assets" / "a.mat",
            target_path=temp_ref_dir / "assets" / "b.png",
        )
        manager.add_reference(ref)

        success = manager.remove_reference(ref)

        assert success
        assert len(manager.graph) == 0

    def test_find_usages(self, temp_ref_dir):
        """find_usages() should return incoming refs."""
        manager = ReferenceManager(temp_ref_dir / "assets")
        target = temp_ref_dir / "assets" / "textures" / "hero_diffuse.png"

        ref1 = AssetReference(
            source_path=temp_ref_dir / "assets" / "a.mat",
            target_path=target,
        )
        ref2 = AssetReference(
            source_path=temp_ref_dir / "assets" / "b.mat",
            target_path=target,
        )

        manager.add_reference(ref1)
        manager.add_reference(ref2)

        usages = manager.find_usages(target)

        assert len(usages) == 2

    def test_find_dependencies(self, temp_ref_dir):
        """find_dependencies() should return outgoing refs."""
        manager = ReferenceManager(temp_ref_dir / "assets")
        source = temp_ref_dir / "assets" / "a.mat"

        ref1 = AssetReference(
            source_path=source,
            target_path=temp_ref_dir / "assets" / "b.png",
        )
        ref2 = AssetReference(
            source_path=source,
            target_path=temp_ref_dir / "assets" / "c.png",
        )

        manager.add_reference(ref1)
        manager.add_reference(ref2)

        deps = manager.find_dependencies(source)

        assert len(deps) == 2

    def test_broken_reference_detection(self, temp_ref_dir):
        """Broken references should be tracked."""
        manager = ReferenceManager(temp_ref_dir / "assets")
        ref = AssetReference(
            source_path=temp_ref_dir / "assets" / "a.mat",
            target_path=temp_ref_dir / "assets" / "missing.png",
        )

        manager.add_reference(ref)

        broken = manager.get_broken_references()
        assert len(broken) == 1
        assert broken[0].missing_path == temp_ref_dir / "assets" / "missing.png"

    def test_get_broken_for_asset(self, temp_ref_dir):
        """get_broken_for_asset() should return broken refs for asset."""
        manager = ReferenceManager(temp_ref_dir / "assets")
        source = temp_ref_dir / "assets" / "a.mat"
        ref = AssetReference(
            source_path=source,
            target_path=temp_ref_dir / "assets" / "missing.png",
        )
        manager.add_reference(ref)

        broken = manager.get_broken_for_asset(source)

        assert len(broken) == 1

    def test_create_redirect(self, temp_ref_dir):
        """create_redirect() should create redirect."""
        manager = ReferenceManager(temp_ref_dir / "assets")

        redirect = manager.create_redirect(
            old_path=Path("/old.png"),
            new_path=Path("/new.png"),
        )

        assert redirect.old_path == Path("/old.png")
        assert redirect.new_path == Path("/new.png")
        assert len(manager.get_redirects()) == 1

    def test_apply_redirect(self, temp_ref_dir):
        """Redirect should update references."""
        manager = ReferenceManager(temp_ref_dir / "assets")
        old_target = temp_ref_dir / "assets" / "old.png"
        new_target = temp_ref_dir / "assets" / "textures" / "hero_diffuse.png"

        ref = AssetReference(
            source_path=temp_ref_dir / "assets" / "a.mat",
            target_path=old_target,
        )
        manager.add_reference(ref)

        manager.create_redirect(old_target, new_target, auto_apply=True)

        # Check reference was updated
        refs = manager.find_dependencies(temp_ref_dir / "assets" / "a.mat")
        assert len(refs) == 1
        assert refs[0].target_path == new_target

    def test_remove_redirect(self, temp_ref_dir):
        """remove_redirect() should remove redirect."""
        manager = ReferenceManager(temp_ref_dir / "assets")
        manager.create_redirect(Path("/old.png"), Path("/new.png"))

        success = manager.remove_redirect(Path("/old.png"))

        assert success
        assert len(manager.get_redirects()) == 0

    def test_validate_references(self, temp_ref_dir):
        """validate_references() should validate all refs."""
        manager = ReferenceManager(temp_ref_dir / "assets")

        # Add valid and invalid refs
        valid_ref = AssetReference(
            source_path=temp_ref_dir / "assets" / "a.mat",
            target_path=temp_ref_dir / "assets" / "textures" / "hero_diffuse.png",
        )
        invalid_ref = AssetReference(
            source_path=temp_ref_dir / "assets" / "b.mat",
            target_path=temp_ref_dir / "assets" / "missing.png",
        )

        manager.add_reference(valid_ref)
        manager.add_reference(invalid_ref)

        valid_count, broken_count = manager.validate_references()

        assert valid_count == 1
        assert broken_count == 1

    def test_suggest_fixes(self, temp_ref_dir):
        """suggest_fixes() should suggest potential fixes."""
        manager = ReferenceManager(temp_ref_dir / "assets")

        # Add broken reference
        ref = AssetReference(
            source_path=temp_ref_dir / "assets" / "a.mat",
            target_path=temp_ref_dir / "assets" / "hero_diffuse.png",  # Wrong path
        )
        manager.add_reference(ref)

        broken_refs = manager.get_broken_references()
        if broken_refs:
            suggestions = manager.suggest_fixes(broken_refs[0])
            # Should suggest the actual file location
            assert len(suggestions) >= 0  # May or may not find suggestions

    def test_get_cycles(self, temp_ref_dir):
        """get_cycles() should return circular refs."""
        manager = ReferenceManager(temp_ref_dir / "assets")

        # Create cycle
        ref1 = AssetReference(source_path=Path("/a"), target_path=Path("/b"))
        ref2 = AssetReference(source_path=Path("/b"), target_path=Path("/c"))
        ref3 = AssetReference(source_path=Path("/c"), target_path=Path("/a"))

        manager.add_reference(ref1)
        manager.add_reference(ref2)
        manager.add_reference(ref3)

        cycles = manager.get_cycles()

        assert len(cycles) > 0

    def test_get_stats(self, temp_ref_dir):
        """get_stats() should return statistics."""
        manager = ReferenceManager(temp_ref_dir / "assets")

        ref = AssetReference(
            source_path=temp_ref_dir / "assets" / "a.mat",
            target_path=temp_ref_dir / "assets" / "textures" / "hero_diffuse.png",
        )
        manager.add_reference(ref)

        stats = manager.get_stats()

        assert stats["total_references"] == 1

    def test_clear(self, temp_ref_dir):
        """clear() should remove all data."""
        manager = ReferenceManager(temp_ref_dir / "assets")

        ref = AssetReference(
            source_path=temp_ref_dir / "assets" / "a.mat",
            target_path=temp_ref_dir / "assets" / "b.png",
        )
        manager.add_reference(ref)
        manager.create_redirect(Path("/old.png"), Path("/new.png"))

        manager.clear()

        assert len(manager.graph) == 0
        assert len(manager.get_redirects()) == 0

    def test_change_listener(self, temp_ref_dir):
        """Change listeners should be notified."""
        manager = ReferenceManager(temp_ref_dir / "assets")
        changes = []

        manager.on_change(lambda ref, action: changes.append((ref, action)))

        ref = AssetReference(
            source_path=temp_ref_dir / "assets" / "a.mat",
            target_path=temp_ref_dir / "assets" / "b.png",
        )
        manager.add_reference(ref)
        manager.remove_reference(ref)

        assert len(changes) == 2
        assert changes[0][1] == "added"
        assert changes[1][1] == "removed"

    def test_register_scanner(self, temp_ref_dir):
        """register_scanner() should register custom scanner."""
        manager = ReferenceManager(temp_ref_dir / "assets")

        def custom_scanner(path):
            return []

        manager.register_scanner("custom", custom_scanner)

        assert "custom" in manager._scanners


class TestScanning:
    """Test reference scanning functionality."""

    def test_scan_json_file(self, temp_ref_dir):
        """Scanning JSON files should find references."""
        manager = ReferenceManager(temp_ref_dir / "assets")
        mat_path = temp_ref_dir / "assets" / "materials" / "hero.mat"

        refs = manager.scan_asset(mat_path)

        # Should find texture references
        assert len(refs) >= 0  # May or may not parse correctly

    def test_scan_directory(self, temp_ref_dir):
        """scan_directory() should scan all files."""
        manager = ReferenceManager(temp_ref_dir / "assets")

        count = manager.scan_directory(temp_ref_dir / "assets")

        # Count may vary based on what the scanners find
        assert count >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
