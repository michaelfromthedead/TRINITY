"""
ReferenceManager - Track asset references, find usages, and redirect references.

Provides comprehensive asset reference tracking:
- Build and query reference graphs
- Find all usages of an asset
- Redirect references when assets move
- Detect and report broken references
- Integration with ContentStore for hash-based tracking
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Protocol, Union
from weakref import WeakValueDictionary

from trinity.decorators.dev import editor


class ReferenceType(Enum):
    """Types of asset references."""

    DIRECT = auto()  # Direct file reference
    EMBEDDED = auto()  # Embedded within another asset
    SOFT = auto()  # Soft/lazy reference (loaded on demand)
    MATERIAL = auto()  # Material texture reference
    PREFAB = auto()  # Prefab component reference
    SCENE = auto()  # Scene object reference
    SCRIPT = auto()  # Script/code reference
    ANIMATION = auto()  # Animation clip reference


class ReferenceStatus(Enum):
    """Status of a reference."""

    VALID = auto()  # Reference target exists
    BROKEN = auto()  # Target not found
    REDIRECTED = auto()  # Reference has been redirected
    PENDING = auto()  # Not yet validated


@dataclass
class AssetReference:
    """Represents a reference from one asset to another.

    Attributes:
        source_path: Path of the asset containing the reference
        target_path: Path of the referenced asset
        reference_type: Type of reference
        status: Current status of the reference
        location: Location within source (e.g., line number, property path)
        content_hash: ContentStore hash of the target (for tracking moves)
        metadata: Additional reference metadata
    """

    source_path: Path
    target_path: Path
    reference_type: ReferenceType = ReferenceType.DIRECT
    status: ReferenceStatus = ReferenceStatus.PENDING
    location: Optional[str] = None
    content_hash: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.source_path, self.target_path, self.location))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AssetReference):
            return (
                self.source_path == other.source_path
                and self.target_path == other.target_path
                and self.location == other.location
            )
        return False


@dataclass
class BrokenReference:
    """Represents a broken reference that needs attention.

    Attributes:
        reference: The broken reference
        missing_path: The path that could not be found
        suggested_fixes: List of potential fix paths
        last_known_hash: Last known content hash of target
        broken_since: Timestamp when reference was detected broken
    """

    reference: AssetReference
    missing_path: Path
    suggested_fixes: list[Path] = field(default_factory=list)
    last_known_hash: Optional[str] = None
    broken_since: float = field(default_factory=time.time)

    @property
    def age_seconds(self) -> float:
        """How long the reference has been broken."""
        return time.time() - self.broken_since


@dataclass
class ReferenceRedirect:
    """Represents a reference redirect/remap.

    Attributes:
        old_path: Original path
        new_path: New path
        created_at: When redirect was created
        applied_count: Number of references updated
        auto_redirect: Whether to auto-apply to new broken refs
    """

    old_path: Path
    new_path: Path
    created_at: float = field(default_factory=time.time)
    applied_count: int = 0
    auto_redirect: bool = True


class ReferenceGraph:
    """Graph structure for tracking asset references.

    Provides efficient queries for:
    - What does asset X reference?
    - What references asset X?
    - Is there a path between assets A and B?

    Attributes:
        _forward: Map from source -> set of targets
        _reverse: Map from target -> set of sources
        _references: All references indexed by (source, target, location)
    """

    def __init__(self) -> None:
        self._forward: dict[Path, set[Path]] = {}
        self._reverse: dict[Path, set[Path]] = {}
        self._references: dict[tuple[Path, Path, Optional[str]], AssetReference] = {}

    def add_reference(self, ref: AssetReference) -> None:
        """Add a reference to the graph."""
        key = (ref.source_path, ref.target_path, ref.location)

        if key in self._references:
            return  # Already exists

        self._references[key] = ref

        # Update forward edges
        if ref.source_path not in self._forward:
            self._forward[ref.source_path] = set()
        self._forward[ref.source_path].add(ref.target_path)

        # Update reverse edges
        if ref.target_path not in self._reverse:
            self._reverse[ref.target_path] = set()
        self._reverse[ref.target_path].add(ref.source_path)

    def remove_reference(self, ref: AssetReference) -> bool:
        """Remove a reference from the graph."""
        key = (ref.source_path, ref.target_path, ref.location)

        if key not in self._references:
            return False

        del self._references[key]

        # Check if there are other references with same source/target
        other_refs = [
            r for r in self._references.values()
            if r.source_path == ref.source_path and r.target_path == ref.target_path
        ]

        if not other_refs:
            # No other refs, remove from forward/reverse
            if ref.source_path in self._forward:
                self._forward[ref.source_path].discard(ref.target_path)
                if not self._forward[ref.source_path]:
                    del self._forward[ref.source_path]

            if ref.target_path in self._reverse:
                self._reverse[ref.target_path].discard(ref.source_path)
                if not self._reverse[ref.target_path]:
                    del self._reverse[ref.target_path]

        return True

    def get_outgoing(self, source: Path) -> list[AssetReference]:
        """Get all references from a source asset."""
        return [
            ref for ref in self._references.values()
            if ref.source_path == source
        ]

    def get_incoming(self, target: Path) -> list[AssetReference]:
        """Get all references to a target asset."""
        return [
            ref for ref in self._references.values()
            if ref.target_path == target
        ]

    def get_dependencies(self, source: Path) -> set[Path]:
        """Get all direct dependencies of an asset."""
        return self._forward.get(source, set()).copy()

    def get_dependents(self, target: Path) -> set[Path]:
        """Get all assets that depend on the target."""
        return self._reverse.get(target, set()).copy()

    def get_all_dependencies(self, source: Path, visited: Optional[set[Path]] = None) -> set[Path]:
        """Get all transitive dependencies of an asset."""
        if visited is None:
            visited = set()

        if source in visited:
            return set()

        visited.add(source)
        deps = set()

        for target in self._forward.get(source, set()):
            deps.add(target)
            deps.update(self.get_all_dependencies(target, visited))

        return deps

    def get_all_dependents(self, target: Path, visited: Optional[set[Path]] = None) -> set[Path]:
        """Get all assets that transitively depend on the target."""
        if visited is None:
            visited = set()

        if target in visited:
            return set()

        visited.add(target)
        deps = set()

        for source in self._reverse.get(target, set()):
            deps.add(source)
            deps.update(self.get_all_dependents(source, visited))

        return deps

    def has_path(self, source: Path, target: Path) -> bool:
        """Check if there's a path from source to target."""
        return target in self.get_all_dependencies(source)

    def find_cycles(self) -> list[list[Path]]:
        """Find all cycles in the reference graph."""
        cycles: list[list[Path]] = []
        visited: set[Path] = set()
        rec_stack: set[Path] = set()
        path: list[Path] = []

        def dfs(node: Path) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._forward.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])

            path.pop()
            rec_stack.remove(node)

        for node in self._forward:
            if node not in visited:
                dfs(node)

        return cycles

    def get_reference(self, source: Path, target: Path, location: Optional[str] = None) -> Optional[AssetReference]:
        """Get a specific reference."""
        return self._references.get((source, target, location))

    def get_all_references(self) -> list[AssetReference]:
        """Get all references in the graph."""
        return list(self._references.values())

    def clear(self) -> None:
        """Clear all references."""
        self._forward.clear()
        self._reverse.clear()
        self._references.clear()

    def __len__(self) -> int:
        return len(self._references)

    def __iter__(self) -> Iterator[AssetReference]:
        return iter(self._references.values())


class ContentStoreProtocol(Protocol):
    """Protocol for ContentStore integration."""

    def put(self, obj: Any) -> Any:
        """Store object, return content hash."""
        ...

    def has(self, hash: Any) -> bool:
        """Check if hash exists."""
        ...


@editor(category="Assets")
class ReferenceManager:
    """Manages asset references across the project.

    Provides:
    - Reference graph building and queries
    - Broken reference detection
    - Reference redirection
    - Usage tracking
    - Integration with ContentStore for hash-based tracking

    Attributes:
        root_path: Root directory for assets
        graph: The reference graph
        content_store: ContentStore for hash tracking
        _broken_refs: Currently broken references
        _redirects: Active redirects
        _scanners: Registered reference scanners by extension
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        content_store: Optional[ContentStoreProtocol] = None,
    ) -> None:
        """Initialize the reference manager.

        Args:
            root_path: Root directory for assets
            content_store: ContentStore for hash-based tracking
        """
        self.root_path = Path(root_path).resolve()
        self.graph = ReferenceGraph()
        self.content_store = content_store

        self._broken_refs: dict[tuple[Path, Path], BrokenReference] = {}
        self._redirects: dict[Path, ReferenceRedirect] = {}
        self._scanners: dict[str, Callable[[Path], list[AssetReference]]] = {}
        self._change_listeners: list[Callable[[AssetReference, str], None]] = []

        # Register default scanners
        self._register_default_scanners()

    def scan_asset(self, path: Union[str, Path]) -> list[AssetReference]:
        """Scan an asset for references.

        Args:
            path: Path to the asset

        Returns:
            List of discovered references
        """
        path = Path(path)

        if not path.exists():
            return []

        ext = path.suffix.lstrip(".").lower()
        scanner = self._scanners.get(ext)

        if scanner:
            refs = scanner(path)
            for ref in refs:
                self.add_reference(ref)
            return refs

        return []

    def scan_directory(
        self,
        directory: Optional[Union[str, Path]] = None,
        recursive: bool = True,
    ) -> int:
        """Scan a directory for references.

        Args:
            directory: Directory to scan (defaults to root_path)
            recursive: Whether to scan subdirectories

        Returns:
            Number of references found
        """
        directory = Path(directory) if directory else self.root_path
        count = 0

        if recursive:
            for path in directory.rglob("*"):
                if path.is_file():
                    refs = self.scan_asset(path)
                    count += len(refs)
        else:
            for path in directory.iterdir():
                if path.is_file():
                    refs = self.scan_asset(path)
                    count += len(refs)

        return count

    def add_reference(self, ref: AssetReference) -> None:
        """Add a reference to the manager.

        Args:
            ref: The reference to add
        """
        # Validate the reference
        ref.status = self._validate_reference(ref)

        # Add to graph
        self.graph.add_reference(ref)

        # Track if broken
        if ref.status == ReferenceStatus.BROKEN:
            self._add_broken_reference(ref)

        # Notify listeners
        self._notify_change(ref, "added")

    def remove_reference(self, ref: AssetReference) -> bool:
        """Remove a reference.

        Args:
            ref: The reference to remove

        Returns:
            True if removed, False if not found
        """
        if self.graph.remove_reference(ref):
            # Remove from broken refs if present
            key = (ref.source_path, ref.target_path)
            self._broken_refs.pop(key, None)

            self._notify_change(ref, "removed")
            return True
        return False

    def find_usages(self, path: Union[str, Path]) -> list[AssetReference]:
        """Find all usages of an asset.

        Args:
            path: Path to the asset

        Returns:
            List of references to this asset
        """
        return self.graph.get_incoming(Path(path))

    def find_dependencies(self, path: Union[str, Path]) -> list[AssetReference]:
        """Find all dependencies of an asset.

        Args:
            path: Path to the asset

        Returns:
            List of references from this asset
        """
        return self.graph.get_outgoing(Path(path))

    def get_broken_references(self) -> list[BrokenReference]:
        """Get all broken references."""
        return list(self._broken_refs.values())

    def get_broken_for_asset(self, path: Union[str, Path]) -> list[BrokenReference]:
        """Get broken references for a specific asset."""
        path = Path(path)
        return [
            br for br in self._broken_refs.values()
            if br.reference.source_path == path or br.reference.target_path == path
        ]

    def create_redirect(
        self,
        old_path: Union[str, Path],
        new_path: Union[str, Path],
        auto_apply: bool = True,
    ) -> ReferenceRedirect:
        """Create a reference redirect.

        Args:
            old_path: Original path
            new_path: New path
            auto_apply: Whether to auto-apply to existing broken refs

        Returns:
            The created redirect
        """
        old_path = Path(old_path)
        new_path = Path(new_path)

        redirect = ReferenceRedirect(
            old_path=old_path,
            new_path=new_path,
            auto_redirect=auto_apply,
        )

        self._redirects[old_path] = redirect

        if auto_apply:
            self._apply_redirect(redirect)

        return redirect

    def apply_redirect(self, old_path: Union[str, Path]) -> int:
        """Apply a redirect to all affected references.

        Args:
            old_path: Original path to redirect from

        Returns:
            Number of references updated
        """
        old_path = Path(old_path)
        redirect = self._redirects.get(old_path)

        if redirect:
            return self._apply_redirect(redirect)
        return 0

    def remove_redirect(self, old_path: Union[str, Path]) -> bool:
        """Remove a redirect.

        Args:
            old_path: Path the redirect is from

        Returns:
            True if removed, False if not found
        """
        old_path = Path(old_path)
        if old_path in self._redirects:
            del self._redirects[old_path]
            return True
        return False

    def get_redirects(self) -> list[ReferenceRedirect]:
        """Get all active redirects."""
        return list(self._redirects.values())

    def validate_references(self) -> tuple[int, int]:
        """Validate all references.

        Returns:
            Tuple of (valid_count, broken_count)
        """
        valid = 0
        broken = 0

        for ref in self.graph:
            ref.status = self._validate_reference(ref)

            if ref.status == ReferenceStatus.VALID:
                valid += 1
                # Remove from broken if was broken
                key = (ref.source_path, ref.target_path)
                self._broken_refs.pop(key, None)
            elif ref.status == ReferenceStatus.BROKEN:
                broken += 1
                self._add_broken_reference(ref)

        return valid, broken

    def suggest_fixes(
        self,
        broken_ref: BrokenReference,
        max_suggestions: int = 5,
    ) -> list[Path]:
        """Suggest fixes for a broken reference.

        Args:
            broken_ref: The broken reference
            max_suggestions: Maximum number of suggestions

        Returns:
            List of potential fix paths
        """
        missing = broken_ref.missing_path
        suggestions: list[tuple[float, Path]] = []

        # Search for files with same name
        name = missing.name
        for path in self.root_path.rglob(name):
            if path.is_file():
                # Score based on similarity of path
                score = self._path_similarity(missing, path)
                suggestions.append((score, path))

        # If we have a content hash, look for matching files
        if broken_ref.last_known_hash and self.content_store:
            # Content store lookup would go here
            pass

        # Sort by score and return top suggestions
        suggestions.sort(key=lambda x: x[0], reverse=True)
        return [path for _, path in suggestions[:max_suggestions]]

    def register_scanner(
        self,
        extension: str,
        scanner: Callable[[Path], list[AssetReference]],
    ) -> None:
        """Register a reference scanner for a file extension.

        Args:
            extension: File extension (without dot)
            scanner: Function that returns references from a file
        """
        self._scanners[extension.lower()] = scanner

    def on_change(
        self,
        callback: Callable[[AssetReference, str], None],
    ) -> None:
        """Register a change listener.

        Args:
            callback: Function receiving (reference, action)
        """
        self._change_listeners.append(callback)

    def get_cycles(self) -> list[list[Path]]:
        """Find circular reference chains."""
        return self.graph.find_cycles()

    def get_stats(self) -> dict[str, Any]:
        """Get reference statistics."""
        refs = list(self.graph)

        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}

        for ref in refs:
            type_name = ref.reference_type.name
            by_type[type_name] = by_type.get(type_name, 0) + 1

            status_name = ref.status.name
            by_status[status_name] = by_status.get(status_name, 0) + 1

        return {
            "total_references": len(refs),
            "broken_references": len(self._broken_refs),
            "active_redirects": len(self._redirects),
            "by_type": by_type,
            "by_status": by_status,
        }

    def clear(self) -> None:
        """Clear all references and redirects."""
        self.graph.clear()
        self._broken_refs.clear()
        self._redirects.clear()

    def _validate_reference(self, ref: AssetReference) -> ReferenceStatus:
        """Validate a reference and return its status."""
        # Check if redirected
        if ref.target_path in self._redirects:
            return ReferenceStatus.REDIRECTED

        # Check if target exists
        if ref.target_path.exists():
            return ReferenceStatus.VALID

        return ReferenceStatus.BROKEN

    def _add_broken_reference(self, ref: AssetReference) -> None:
        """Add a broken reference to tracking."""
        key = (ref.source_path, ref.target_path)

        if key not in self._broken_refs:
            broken = BrokenReference(
                reference=ref,
                missing_path=ref.target_path,
                last_known_hash=ref.content_hash,
            )
            broken.suggested_fixes = self.suggest_fixes(broken)
            self._broken_refs[key] = broken

    def _apply_redirect(self, redirect: ReferenceRedirect) -> int:
        """Apply a redirect to affected references."""
        count = 0

        for ref in list(self.graph):
            if ref.target_path == redirect.old_path:
                # Create new reference with new path
                new_ref = AssetReference(
                    source_path=ref.source_path,
                    target_path=redirect.new_path,
                    reference_type=ref.reference_type,
                    status=ReferenceStatus.REDIRECTED,
                    location=ref.location,
                    content_hash=ref.content_hash,
                    metadata=ref.metadata.copy(),
                )

                # Remove old, add new
                self.graph.remove_reference(ref)
                self.graph.add_reference(new_ref)

                # Remove from broken refs
                key = (ref.source_path, ref.target_path)
                self._broken_refs.pop(key, None)

                count += 1

        redirect.applied_count = count
        return count

    def _path_similarity(self, path1: Path, path2: Path) -> float:
        """Calculate similarity score between two paths."""
        parts1 = path1.parts
        parts2 = path2.parts

        # Count matching parts from the end
        matches = 0
        for p1, p2 in zip(reversed(parts1), reversed(parts2)):
            if p1 == p2:
                matches += 1
            else:
                break

        # Return ratio of matching parts
        return matches / max(len(parts1), len(parts2))

    def _notify_change(self, ref: AssetReference, action: str) -> None:
        """Notify listeners of a reference change."""
        for listener in self._change_listeners:
            try:
                listener(ref, action)
            except Exception:
                pass

    def _register_default_scanners(self) -> None:
        """Register default reference scanners."""
        # Scene files (JSON-based)
        self._scanners["scene"] = self._scan_json_refs
        self._scanners["prefab"] = self._scan_json_refs

        # Material files
        self._scanners["mat"] = self._scan_json_refs

        # Python scripts
        self._scanners["py"] = self._scan_python_refs

    def _scan_json_refs(self, path: Path) -> list[AssetReference]:
        """Scan a JSON file for asset references."""
        refs: list[AssetReference] = []

        try:
            with open(path, "r") as f:
                data = json.load(f)

            def scan_value(value: Any, location: str) -> None:
                if isinstance(value, str):
                    # Check if it looks like a path
                    if "/" in value or "\\" in value:
                        potential_path = self.root_path / value
                        if potential_path.suffix:
                            refs.append(AssetReference(
                                source_path=path,
                                target_path=potential_path,
                                reference_type=ReferenceType.DIRECT,
                                location=location,
                            ))
                elif isinstance(value, dict):
                    for k, v in value.items():
                        scan_value(v, f"{location}.{k}")
                elif isinstance(value, list):
                    for i, v in enumerate(value):
                        scan_value(v, f"{location}[{i}]")

            scan_value(data, "")

        except Exception:
            pass

        return refs

    def _scan_python_refs(self, path: Path) -> list[AssetReference]:
        """Scan a Python file for asset references."""
        refs: list[AssetReference] = []

        try:
            with open(path, "r") as f:
                content = f.read()

            # Look for string literals that might be paths
            import re
            pattern = r'["\']([^"\']+\.(png|jpg|fbx|obj|wav|ogg|glb|gltf))["\']'

            for match in re.finditer(pattern, content, re.IGNORECASE):
                potential_path = match.group(1)
                line_num = content[:match.start()].count("\n") + 1

                refs.append(AssetReference(
                    source_path=path,
                    target_path=self.root_path / potential_path,
                    reference_type=ReferenceType.SCRIPT,
                    location=f"line:{line_num}",
                ))

        except Exception:
            pass

        return refs


__all__ = [
    "ReferenceType",
    "ReferenceStatus",
    "AssetReference",
    "BrokenReference",
    "ReferenceRedirect",
    "ReferenceGraph",
    "ReferenceManager",
]
