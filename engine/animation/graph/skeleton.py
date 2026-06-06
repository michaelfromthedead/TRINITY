"""Skeleton and bone hierarchy for the animation graph.

Defines the Bone and Skeleton classes that form a hierarchical tree
structure used for skeletal animation, IK chain queries, and
skeleton-level validation.

Architecture
------------
Bone
    name: str              -- unique bone identifier
    parent: Bone | None    -- reference to parent (None for roots)
    children: list[Bone]   -- ordered list of child bones
    bind_pose: Transform   -- local-space rest pose relative to parent

    add_child(bone)     -- add child with automatic parent linkage
    remove_child(bone)  -- detach child from this bone

Skeleton
    _bones: dict[str, Bone]       -- O(1) name lookup
    root_bones: list[Bone]        -- entries with no parent
    bone_count: int               -- total number of bones

    get_bone(name)      -- O(1) dict lookup
    get_chain(a, b)     -- ordered chain start-to-end for IK
    add_bone(...)       -- build hierarchy incrementally
    validate(...)       -- no orphans, single-connected forest (returns errors)
    is_valid(...)       -- convenience bool wrapper around validate()
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional, Sequence

from engine.core.math.transform import Transform


class Bone:
    """A single bone in a skeleton hierarchy.

    Bones form a tree via ``parent`` and ``children`` references.
    Each bone stores its **local** bind pose (relative to its parent).

    Attributes
    ----------
    name : str
        Unique human-readable bone identifier (e.g. ``"spine_01"``).
    parent : Bone or None
        Parent bone in the hierarchy.  ``None`` for root bones.
    children : list[Bone]
        Direct child bones.  Order is insertion order.
    bind_pose : Transform
        Local-space bind-pose transform (translation / rotation / scale)
        relative to the parent bone.

    Raises
    ------
    ValueError
        If ``name`` is an empty string.
    """

    def __init__(
        self,
        name: str,
        parent: Optional[Bone] = None,
        bind_pose: Optional[Transform] = None,
    ) -> None:
        if not name:
            raise ValueError("Bone name must be a non-empty string")
        self.name = name
        self.parent = parent
        self.children: List[Bone] = []
        self.bind_pose = bind_pose if bind_pose is not None else Transform.identity()

    # -- convenience properties ------------------------------------------------

    @property
    def is_root(self) -> bool:
        """True when this bone has no parent (i.e. it is a root bone)."""
        return self.parent is None

    @property
    def depth(self) -> int:
        """Number of steps from this bone to the root (root depth == 0)."""
        d = 0
        p = self.parent
        while p is not None:
            d += 1
            p = p.parent
        return d

    # -- hierarchy navigation --------------------------------------------------

    def get_root(self) -> Bone:
        """Walk up to the root of the tree this bone belongs to."""
        r: Bone = self
        while r.parent is not None:
            r = r.parent
        return r

    def get_siblings(self) -> List[Bone]:
        """Return all other bones that share this bone's parent."""
        if self.parent is None:
            return []
        return [b for b in self.parent.children if b is not self]

    def is_ancestor_of(self, other: Bone) -> bool:
        """Return True when ``other`` is a descendant of this bone."""
        p = other.parent
        while p is not None:
            if p is self:
                return True
            p = p.parent
        return False

    def is_descendant_of(self, ancestor: Bone) -> bool:
        """Return True when this bone is a descendant of ``ancestor``."""
        return ancestor.is_ancestor_of(self)

    def get_descendants(self) -> List[Bone]:
        """Return all descendant bones in breadth-first order (excludes self)."""
        result: List[Bone] = []
        queue: deque[Bone] = deque(self.children)
        while queue:
            b = queue.popleft()
            result.append(b)
            queue.extend(b.children)
        return result

    # -- child management -----------------------------------------------------

    def add_child(self, bone: Bone) -> None:
        """Add a bone as a child of this bone.

        Sets the child's parent reference to this bone and appends
        to this bone's children list. If the bone already has a parent,
        it is first removed from that parent's children list.

        Parameters
        ----------
        bone : Bone
            The bone to add as a child.

        Raises
        ------
        ValueError
            If attempting to add self as a child (cycle), or if *bone*
            is already a child of this bone.
        """
        if bone is self:
            raise ValueError("Cannot add a bone as its own child")
        if bone in self.children:
            raise ValueError(f"Bone '{bone.name}' is already a child of '{self.name}'")

        # Remove from previous parent if any
        if bone.parent is not None:
            bone.parent.children = [c for c in bone.parent.children if c is not bone]

        bone.parent = self
        self.children.append(bone)

    def remove_child(self, bone: Bone) -> bool:
        """Remove a bone from this bone's children.

        Clears the child's parent reference if removal succeeds.
        Does **not** recursively remove the child's descendants;
        they remain attached to the removed bone.

        Parameters
        ----------
        bone : Bone
            The bone to remove from children.

        Returns
        -------
        bool
            ``True`` if the bone was found and removed, ``False`` otherwise.
        """
        if bone not in self.children:
            return False

        self.children = [c for c in self.children if c is not bone]
        bone.parent = None
        return True

    # -- copy / representation ------------------------------------------------

    def copy(self) -> Bone:
        """Create a deep copy of this bone and all its descendants."""
        return deepcopy(self)

    def __repr__(self) -> str:
        parent_str = f", parent={self.parent.name}" if self.parent else ", root"
        return f"Bone('{self.name}'{parent_str})"


class Skeleton:
    """A hierarchy of bones for skeletal animation.

    The skeleton maintains a **forest** of bones (one or more trees).
    The primary storage is a name-to-Bone dictionary for O(1) lookup,
    plus a separate list of root bones for traversal entry points.

    Parameters
    ----------
    name : str
        Skeleton identifier (e.g. ``"humanoid"``, ``"spider"``).
    """

    def __init__(self, name: str = "skeleton") -> None:
        if not name:
            raise ValueError("Skeleton name must be a non-empty string")
        self.name = name
        self._bones: Dict[str, Bone] = {}
        self._root_bones: List[Bone] = []

    # -- read-only property views ---------------------------------------------

    @property
    def bone_count(self) -> int:
        """Total number of bones registered with this skeleton."""
        return len(self._bones)

    @property
    def bones(self) -> Dict[str, Bone]:
        """Read-only view of the name-to-Bone mapping."""
        return dict(self._bones)

    @property
    def root_bones(self) -> List[Bone]:
        """Read-only view of root bones (bones with no parent)."""
        return list(self._root_bones)

    # -- bone management ------------------------------------------------------

    def add_bone(
        self,
        name: str,
        parent_name: Optional[str] = None,
        bind_pose: Optional[Transform] = None,
    ) -> Bone:
        """Create a bone and insert it into the hierarchy.

        Parameters
        ----------
        name : str
            Unique bone name.
        parent_name : str or None
            Name of the parent bone.  ``None`` creates a root bone.
        bind_pose : Transform or None
            Local-space bind pose.  Identity when omitted.

        Returns
        -------
        Bone
            The newly created bone (already linked into the skeleton).

        Raises
        ------
        ValueError
            If *name* already exists, or *parent_name* is not ``None`` and
            does not name a registered bone.
        """
        if name in self._bones:
            raise ValueError(f"Bone '{name}' already exists in skeleton '{self.name}'")

        parent: Optional[Bone] = None
        if parent_name is not None:
            parent = self._bones.get(parent_name)
            if parent is None:
                raise ValueError(
                    f"Parent bone '{parent_name}' not found in skeleton '{self.name}'"
                )

        bone = Bone(name=name, parent=parent, bind_pose=bind_pose)

        self._bones[name] = bone
        if parent is not None:
            parent.children.append(bone)
        else:
            self._root_bones.append(bone)

        return bone

    def remove_bone(self, name: str) -> bool:
        """Remove a bone and adopt its children into the grandparent.

        When a bone is removed its children are re-parented to the removed
        bone's parent (or become roots if the removed bone was itself a root).
        Returns ``True`` when the bone was found and removed.
        """
        bone = self._bones.get(name)
        if bone is None:
            return False

        # Re-parent children to bone's parent (or None for roots)
        for child in list(bone.children):
            child.parent = bone.parent
            if bone.parent is not None:
                bone.parent.children.append(child)
            else:
                self._root_bones.append(child)
        bone.children.clear()

        # Detach from parent
        if bone.parent is not None:
            bone.parent.children = [c for c in bone.parent.children if c is not bone]
        else:
            self._root_bones = [r for r in self._root_bones if r is not bone]

        del self._bones[name]
        return True

    def has_bone(self, name: str) -> bool:
        """Return True when a bone with *name* exists in the skeleton."""
        return name in self._bones

    def get_bone(self, name: str) -> Optional[Bone]:
        """Look up a bone by name (O(1) via dictionary).

        Parameters
        ----------
        name : str
            Bone name.

        Returns
        -------
        Bone or None
            The bone, or ``None`` when not found.
        """
        return self._bones.get(name)

    def get_child_names(self, name: str) -> List[str]:
        """Return the names of all direct children of *name*."""
        bone = self._bones.get(name)
        if bone is None:
            return []
        return [c.name for c in bone.children]

    def get_parent_name(self, name: str) -> Optional[str]:
        """Return the name of the parent of *name*, or ``None``."""
        bone = self._bones.get(name)
        if bone is None or bone.parent is None:
            return None
        return bone.parent.name

    def get_ancestor_names(self, name: str) -> List[str]:
        """Return the chain of ancestor names from *name* to root (inclusive)."""
        bone = self._bones.get(name)
        if bone is None:
            return []
        result = [bone.name]
        p = bone.parent
        while p is not None:
            result.append(p.name)
            p = p.parent
        return result

    def get_depth(self, name: str) -> int:
        """Return the depth of *name* (root depth == 0).

        Returns -1 when the bone is not found.
        """
        bone = self._bones.get(name)
        if bone is None:
            return -1
        return bone.depth

    def get_max_depth(self) -> int:
        """Return the maximum bone depth in the skeleton."""
        if not self._bones:
            return -1
        return max(b.depth for b in self._bones.values())

    # -- chain query (IK integration) -----------------------------------------

    def get_chain(self, start_name: str, end_name: str) -> List[Bone]:
        """Return the ordered bone chain from *start_name* to *end_name*.

        The chain always runs **from** *start* **to** *end* following the
        hierarchy.  When *start* and *end* belong to different trees the
        result is an empty list.

        This method is the primary integration point for IK solvers that
        need an ordered list of bones forming a kinematic chain.

        Parameters
        ----------
        start_name : str
            Name of the start bone (closest to root / upstream).
        end_name : str
            Name of the end bone (farthest from root / downstream).

        Returns
        -------
        list[Bone]
            Ordered chain from *start* to *end*, or ``[]`` when there
            is no valid path between the two bones.
        """
        start = self._bones.get(start_name)
        end = self._bones.get(end_name)
        if start is None or end is None:
            return []

        if start is end:
            return [start]

        # Walk each bone up to its root, recording the path.
        start_path: List[Bone] = []
        b: Optional[Bone] = start
        while b is not None:
            start_path.append(b)
            b = b.parent

        end_path: List[Bone] = []
        b = end
        while b is not None:
            end_path.append(b)
            b = b.parent

        # Find the lowest common ancestor (first bone in end_path that
        # also appears in start_path).
        start_names = {bone.name for bone in start_path}

        common: Optional[Bone] = None
        idx_in_end = -1
        for i, bone in enumerate(end_path):
            if bone.name in start_names:
                common = bone
                idx_in_end = i
                break

        if common is None:
            return []  # disconnected trees

        # Index of common ancestor within start_path.
        idx_in_start = next(
            i for i, bone in enumerate(start_path) if bone.name == common.name
        )

        chain: List[Bone] = []

        # Segment 1: start -> ... -> bone just before common
        for i in range(idx_in_start):
            chain.append(start_path[i])

        # Segment 2: the common bone itself
        chain.append(common)

        # Segment 3: from the bone after common -> end
        for i in range(idx_in_end - 1, -1, -1):
            chain.append(end_path[i])

        return chain

    # -- validation -----------------------------------------------------------

    def validate(self) -> List[str]:
        """Check skeleton integrity.

        Returns a list of human-readable error messages.  An empty list
        means the skeleton is structurally valid.

        Checks performed
        -----------------
        * At least one root bone exists (when skeleton is non-empty).
        * Every entry in the name dict also appears in the root or
          children references.
        * No parent cycles via parent references (defensive).
        """
        errors: List[str] = []

        if not self._bones:
            errors.append("Skeleton is empty -- no bones registered")
            return errors

        # -- root bones -------------------------------------------------------
        if not self._root_bones:
            errors.append(
                "Skeleton has no root bones: every bone has a parent "
                "(would form a cycle or be unreachable)"
            )

        # -- consistency: every registered bone appears in roots or children --
        reachable: set[str] = set()
        for r in self._root_bones:
            reachable.add(r.name)
            queue: List[Bone] = list(r.children)
            while queue:
                child = queue.pop(0)
                reachable.add(child.name)
                queue.extend(child.children)

        for name in self._bones:
            if name not in reachable:
                errors.append(
                    f"Orphan bone '{name}' is registered but not reachable "
                    "from any root"
                )

        # -- parent back-reference consistency --------------------------------
        for name, bone in self._bones.items():
            if bone.parent is not None and bone.parent.name not in self._bones:
                errors.append(
                    f"Bone '{name}' references parent '{bone.parent.name}' "
                    "that is not registered in this skeleton"
                )
            for child in bone.children:
                if child.name not in self._bones:
                    errors.append(
                        f"Bone '{name}' lists child '{child.name}' "
                        "that is not registered in this skeleton"
                    )

        # -- cycle detection (via parent references) --------------------------
        for name in self._bones:
            visited: set[str] = set()
            b = self._bones[name]
            while b is not None:
                if b.name in visited:
                    errors.append(
                        f"Cycle detected involving bone '{b.name}' "
                        f"while walking up from '{name}'"
                    )
                    break
                visited.add(b.name)
                b = b.parent

        return errors

    def is_valid(self) -> bool:
        """Check if the skeleton is structurally valid.

        This is a convenience wrapper around ``validate()`` that returns
        a boolean instead of a list of error messages.

        Returns
        -------
        bool
            ``True`` when the skeleton passes all validation checks,
            ``False`` otherwise.
        """
        return len(self.validate()) == 0

    # -- bulk operations ------------------------------------------------------

    def add_bones_from_flat(
        self,
        definitions: Sequence[tuple[str, Optional[str]]],
        bind_poses: Optional[Dict[str, Transform]] = None,
    ) -> None:
        """Add bones from a flat list of (name, parent_name) pairs.

        Bones are added in list order so parents must appear before
        their children.

        Parameters
        ----------
        definitions : sequence of (str, str or None)
            Each element is ``(name, parent_name)``.  ``parent_name``
            may be ``None`` for root bones.
        bind_poses : dict of (str -> Transform) or None
            Optional per-bone bind poses keyed by bone name.
        """
        for name, parent_name in definitions:
            pose = bind_poses.get(name) if bind_poses else None
            self.add_bone(name=name, parent_name=parent_name, bind_pose=pose)

    def copy(self, name: Optional[str] = None) -> Skeleton:
        """Create an independent deep copy of this skeleton.

        Parameters
        ----------
        name : str or None
            Optional new name for the copy.  When ``None`` the original
            name is preserved.
        """
        result: Skeleton = deepcopy(self)
        if name is not None:
            result.name = name
        return result

    def __contains__(self, name: str) -> bool:
        return name in self._bones

    def __len__(self) -> int:
        return len(self._bones)

    def __iter__(self):
        return iter(self._bones.values())

    def __repr__(self) -> str:
        return f"Skeleton('{self.name}', bones={self.bone_count})"
