"""Bidirectional dependency graph for material shader hot-reload.

This module provides a thread-safe dependency graph that tracks relationships
between materials and their include files, enabling efficient hot-reload
invalidation. It extends the basic DepGraph from includes.py with:

1. Material-level tracking (not just includes)
2. Bidirectional edges for efficient traversal
3. BFS-based broadest invalidation set computation
4. RLock-guarded concurrent access support

Example::

    graph = MaterialDepGraph()

    # Record material compilation with its includes
    graph.record_material_compilation(
        material=Path("materials/gold.wgsl"),
        includes={Path("shaders/pbr/brdf.wgsl"), Path("shaders/common/color.wgsl")}
    )

    # Get all materials affected by a change
    affected = graph.broadest_invalidation_set(Path("shaders/common/math.wgsl"))
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Dict, Set, Optional, Iterator


@dataclass
class MaterialDepGraph:
    """Bidirectional dependency graph for material hot-reload.

    Tracks three types of relationships:
    - Include -> Materials: Which materials use a given include file
    - Material -> Includes: Which includes a material depends on
    - Material -> Dependents: Material-to-material dependencies

    All operations are thread-safe via RLock for concurrent reads/writes.

    Attributes:
        include_to_materials: Forward edges from include files to materials.
        material_to_includes: Backward edges from materials to their includes.
        material_to_dependents: Material dependency relationships.
    """

    # Forward edges: include -> materials that use it
    include_to_materials: Dict[Path, Set[Path]] = field(default_factory=dict)

    # Backward edges: material -> includes it uses
    material_to_includes: Dict[Path, Set[Path]] = field(default_factory=dict)

    # Material dependencies: material -> other materials it depends on
    material_to_dependents: Dict[Path, Set[Path]] = field(default_factory=dict)

    # Reverse of material_to_dependents: material -> materials that depend on it
    material_dependents_of: Dict[Path, Set[Path]] = field(default_factory=dict)

    # Thread-safe lock for concurrent access
    _lock: RLock = field(default_factory=RLock)

    def record_material_compilation(
        self,
        material: Path,
        includes: Set[Path]
    ) -> None:
        """Record edges when a material is compiled.

        Updates the dependency graph with the include relationships discovered
        during material compilation. Clears old edges for the material first
        to handle recompilation correctly.

        Args:
            material: Path to the material file being compiled.
            includes: Set of include file paths used by this material.
        """
        with self._lock:
            material = self._normalize_path(material)
            includes = {self._normalize_path(inc) for inc in includes}

            # Clear old edges for this material
            old_includes = self.material_to_includes.get(material, set())
            for inc in old_includes:
                if inc in self.include_to_materials:
                    self.include_to_materials[inc].discard(material)
                    # Clean up empty sets
                    if not self.include_to_materials[inc]:
                        del self.include_to_materials[inc]

            # Add new edges
            self.material_to_includes[material] = includes.copy()
            for inc in includes:
                if inc not in self.include_to_materials:
                    self.include_to_materials[inc] = set()
                self.include_to_materials[inc].add(material)

    def record_material_dependency(
        self,
        material: Path,
        depends_on: Path
    ) -> None:
        """Record that a material depends on another material.

        This tracks material-to-material dependencies, separate from
        include dependencies. Used for materials that reference or
        extend other materials.

        Args:
            material: The dependent material.
            depends_on: The material being depended upon.
        """
        with self._lock:
            material = self._normalize_path(material)
            depends_on = self._normalize_path(depends_on)

            # Forward edge: material depends on depends_on
            if material not in self.material_to_dependents:
                self.material_to_dependents[material] = set()
            self.material_to_dependents[material].add(depends_on)

            # Reverse edge: depends_on has material as dependent
            if depends_on not in self.material_dependents_of:
                self.material_dependents_of[depends_on] = set()
            self.material_dependents_of[depends_on].add(material)

    def broadest_invalidation_set(self, changed_path: Path) -> Set[Path]:
        """BFS to find all materials affected by a file change.

        Computes the transitive closure of all materials that need
        recompilation when a file (include or material) changes.

        The algorithm:
        1. Start with the changed path
        2. If it's an include, find all materials using it
        3. For each affected material, find materials depending on it
        4. Continue BFS until no new materials are found

        Args:
            changed_path: Path to the file that changed.

        Returns:
            Set of all material paths that need recompilation.
        """
        with self._lock:
            changed_path = self._normalize_path(changed_path)
            affected: Set[Path] = set()
            queue: deque[Path] = deque([changed_path])
            visited: Set[Path] = {changed_path}

            while queue:
                path = queue.popleft()

                # If it's an include file, add all materials that use it
                if path in self.include_to_materials:
                    for mat in self.include_to_materials[path]:
                        if mat not in visited:
                            visited.add(mat)
                            affected.add(mat)
                            queue.append(mat)

                # If it's a material, it's affected (unless it's the root include)
                if path in self.material_to_includes:
                    affected.add(path)

                # Check materials that depend on this material
                if path in self.material_dependents_of:
                    for dep in self.material_dependents_of[path]:
                        if dep not in visited:
                            visited.add(dep)
                            affected.add(dep)
                            queue.append(dep)

            return affected

    def get_material_includes(self, material: Path) -> Set[Path]:
        """Get all includes used by a material.

        Args:
            material: Path to the material file.

        Returns:
            Set of include paths used by this material.
        """
        with self._lock:
            material = self._normalize_path(material)
            return self.material_to_includes.get(material, set()).copy()

    def get_include_materials(self, include: Path) -> Set[Path]:
        """Get all materials that use an include.

        Args:
            include: Path to the include file.

        Returns:
            Set of material paths using this include.
        """
        with self._lock:
            include = self._normalize_path(include)
            return self.include_to_materials.get(include, set()).copy()

    def get_material_dependents(self, material: Path) -> Set[Path]:
        """Get materials that depend on this material.

        Args:
            material: Path to the material file.

        Returns:
            Set of material paths depending on this material.
        """
        with self._lock:
            material = self._normalize_path(material)
            return self.material_dependents_of.get(material, set()).copy()

    def get_material_dependencies(self, material: Path) -> Set[Path]:
        """Get materials that this material depends on.

        Args:
            material: Path to the material file.

        Returns:
            Set of material paths this material depends on.
        """
        with self._lock:
            material = self._normalize_path(material)
            return self.material_to_dependents.get(material, set()).copy()

    def remove_material(self, material: Path) -> None:
        """Remove a material and all its edges from the graph.

        Args:
            material: Path to the material to remove.
        """
        with self._lock:
            material = self._normalize_path(material)

            # Remove include edges
            if material in self.material_to_includes:
                for inc in self.material_to_includes[material]:
                    if inc in self.include_to_materials:
                        self.include_to_materials[inc].discard(material)
                        if not self.include_to_materials[inc]:
                            del self.include_to_materials[inc]
                del self.material_to_includes[material]

            # Remove material dependency edges (forward)
            if material in self.material_to_dependents:
                for dep in self.material_to_dependents[material]:
                    if dep in self.material_dependents_of:
                        self.material_dependents_of[dep].discard(material)
                        if not self.material_dependents_of[dep]:
                            del self.material_dependents_of[dep]
                del self.material_to_dependents[material]

            # Remove material dependency edges (reverse)
            if material in self.material_dependents_of:
                for dep in self.material_dependents_of[material]:
                    if dep in self.material_to_dependents:
                        self.material_to_dependents[dep].discard(material)
                        if not self.material_to_dependents[dep]:
                            del self.material_to_dependents[dep]
                del self.material_dependents_of[material]

    def clear(self) -> None:
        """Clear all edges from the graph."""
        with self._lock:
            self.include_to_materials.clear()
            self.material_to_includes.clear()
            self.material_to_dependents.clear()
            self.material_dependents_of.clear()

    def material_count(self) -> int:
        """Get the number of materials in the graph.

        Returns:
            Count of unique materials.
        """
        with self._lock:
            return len(self.material_to_includes)

    def include_count(self) -> int:
        """Get the number of includes in the graph.

        Returns:
            Count of unique include files.
        """
        with self._lock:
            return len(self.include_to_materials)

    def all_materials(self) -> Set[Path]:
        """Get all materials in the graph.

        Returns:
            Set of all material paths.
        """
        with self._lock:
            return set(self.material_to_includes.keys())

    def all_includes(self) -> Set[Path]:
        """Get all includes in the graph.

        Returns:
            Set of all include file paths.
        """
        with self._lock:
            return set(self.include_to_materials.keys())

    def edges(self) -> Iterator[tuple[Path, Path, str]]:
        """Iterate over all edges in the graph.

        Yields:
            Tuples of (source, target, edge_type) where edge_type is
            'include' for material->include edges or 'depends' for
            material->material edges.
        """
        with self._lock:
            # Material to include edges
            for material, includes in self.material_to_includes.items():
                for inc in includes:
                    yield (material, inc, "include")

            # Material to material edges
            for material, deps in self.material_to_dependents.items():
                for dep in deps:
                    yield (material, dep, "depends")

    def _normalize_path(self, path: Path) -> Path:
        """Normalize a path for consistent graph keys.

        Args:
            path: Path to normalize.

        Returns:
            Resolved absolute path.
        """
        if isinstance(path, str):
            path = Path(path)
        try:
            return path.resolve()
        except (OSError, ValueError):
            # Fallback for paths that can't be resolved
            return path

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"MaterialDepGraph("
                f"materials={self.material_count()}, "
                f"includes={self.include_count()})"
            )
