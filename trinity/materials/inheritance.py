"""Material Inheritance Model for TRINITY Material DSL (T-MAT-5.2).

This module provides advanced inheritance resolution for Material classes,
supporting texture slot inheritance, surface method chaining via super(),
and proper MRO resolution for multiple inheritance scenarios.

Key Features:
- InheritanceResolver: Walks MRO to collect inherited texture slots and WGSL
- resolve_parent_surface(cls): Get parent's compiled surface() WGSL
- inline_super_call(child_wgsl, parent_wgsl): Replace super() marker with parent code
- merge_texture_declarations(child, parent): Combine texture slots with override support
- Diamond inheritance support with proper MRO ordering

Example::

    from trinity.materials import Material, MaterialMeta, surface
    from trinity.materials.textures import Texture2D
    from trinity.materials.inheritance import InheritanceResolver

    class BaseMaterial(Material, metaclass=MaterialMeta):
        albedo = Texture2D(default="white", srgb=True)

        @surface
        def surface(self, ctx, out):
            out.base_color = ctx.sample(self.albedo, ctx.uv).xyz
            out.roughness = 0.5

    class MetalMaterial(BaseMaterial, metaclass=MaterialMeta):
        roughness_map = Texture2D(default="gray")

        @surface
        def surface(self, ctx, out):
            super().surface(ctx, out)  # Apply base shader
            out.metallic = 0.9
            out.roughness *= ctx.sample(self.roughness_map, ctx.uv).r

    # Inspect inheritance
    resolver = InheritanceResolver(MetalMaterial)
    print(resolver.get_inheritance_chain())  # [MetalMaterial, BaseMaterial]
    print(resolver.get_all_textures())       # {albedo, roughness_map}
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from trinity.materials.textures import TextureDescriptor


# =============================================================================
# EXCEPTIONS
# =============================================================================


class InheritanceError(Exception):
    """Base exception for inheritance-related errors."""
    pass


class NoParentSurfaceError(InheritanceError):
    """Raised when super().surface() is called but no parent has a surface method."""
    pass


class CircularInheritanceError(InheritanceError):
    """Raised when circular inheritance is detected (shouldn't happen with Python's MRO)."""
    pass


class DiamondConflictError(InheritanceError):
    """Raised when diamond inheritance creates unresolvable conflicts."""
    pass


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class InheritanceInfo:
    """Information about a material's inheritance structure.

    Attributes:
        material_class: The material class this info describes
        parent_material: Direct parent with surface method, or None
        inheritance_chain: List of materials from this class to root
        all_textures: All texture bindings (own + inherited)
        own_textures: Textures defined directly on this class
        inherited_textures: Textures inherited from parent classes
        has_super_call: Whether surface() calls super().surface()
        parent_wgsl: WGSL source from parent material, or None
        mro_materials: All material classes in MRO order
    """
    material_class: type
    parent_material: Optional[type] = None
    inheritance_chain: List[type] = field(default_factory=list)
    all_textures: Dict[str, Any] = field(default_factory=dict)
    own_textures: Dict[str, Any] = field(default_factory=dict)
    inherited_textures: Dict[str, Any] = field(default_factory=dict)
    has_super_call: bool = False
    parent_wgsl: Optional[str] = None
    mro_materials: List[type] = field(default_factory=list)


@dataclass
class TextureMergeResult:
    """Result of merging texture declarations from parent and child.

    Attributes:
        merged: Combined texture bindings (child overrides parent)
        from_parent: Textures that came from parent(s)
        from_child: Textures defined on child
        overridden: Textures where child overrode parent
    """
    merged: Dict[str, Any] = field(default_factory=dict)
    from_parent: Dict[str, Any] = field(default_factory=dict)
    from_child: Dict[str, Any] = field(default_factory=dict)
    overridden: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SuperCallInfo:
    """Information about a super().surface() call in the AST.

    Attributes:
        lineno: Line number where super() appears
        col_offset: Column offset of the call
        has_args: Whether arguments were passed
        arg_count: Number of arguments
    """
    lineno: int
    col_offset: int
    has_args: bool = True
    arg_count: int = 2


# =============================================================================
# INHERITANCE RESOLVER
# =============================================================================


class InheritanceResolver:
    """Resolves inheritance relationships for Material classes.

    This class walks the MRO (Method Resolution Order) to collect inherited
    texture slots, find parent surface methods, and build inheritance chains.

    Example::

        resolver = InheritanceResolver(MyMaterial)

        # Get inheritance chain
        chain = resolver.get_inheritance_chain()  # [MyMaterial, Parent, Grandparent]

        # Get all textures including inherited
        textures = resolver.get_all_textures()

        # Get parent's WGSL
        parent_wgsl = resolver.get_parent_wgsl()

        # Check if super() is called
        if resolver.has_super_call():
            print("Material extends parent shader")
    """

    def __init__(self, material_class: type):
        """Initialize resolver for a material class.

        Args:
            material_class: The Material subclass to analyze
        """
        self.material_class = material_class
        self._info: Optional[InheritanceInfo] = None
        self._resolved = False

    def resolve(self) -> InheritanceInfo:
        """Perform full inheritance resolution.

        Returns:
            InheritanceInfo with all resolved information

        Raises:
            InheritanceError: If resolution fails
        """
        if self._resolved and self._info:
            return self._info

        info = InheritanceInfo(material_class=self.material_class)

        # Build MRO list of material classes
        info.mro_materials = self._get_material_mro()

        # Find parent with surface method
        info.parent_material = self._find_parent_material()

        # Build inheritance chain
        info.inheritance_chain = self._build_inheritance_chain()

        # Collect textures
        texture_result = self._collect_textures()
        info.all_textures = texture_result.merged
        info.own_textures = texture_result.from_child
        info.inherited_textures = texture_result.from_parent

        # Check for super() call
        info.has_super_call = self._detect_super_call()

        # Get parent WGSL if available
        if info.parent_material:
            info.parent_wgsl = getattr(info.parent_material, "_wgsl_source", None)

        self._info = info
        self._resolved = True
        return info

    def get_inheritance_chain(self) -> List[type]:
        """Get the chain of material classes in inheritance order.

        Returns:
            List of material classes from this class to root parent.
            Only includes classes with surface methods.
        """
        if hasattr(self.material_class, "get_inheritance_chain"):
            return self.material_class.get_inheritance_chain()
        return self._build_inheritance_chain()

    def get_parent_material(self) -> Optional[type]:
        """Get the first parent class with a surface method.

        Returns:
            Parent material class, or None if no parent has surface().
        """
        return getattr(self.material_class, "_parent_material", None) or self._find_parent_material()

    def get_parent_wgsl(self) -> Optional[str]:
        """Get the compiled WGSL from the parent material.

        Returns:
            Parent WGSL source, or None if no parent.
        """
        parent = self.get_parent_material()
        if parent:
            return getattr(parent, "_wgsl_source", None)
        return None

    def get_all_textures(self) -> Dict[str, Any]:
        """Get all texture bindings (own + inherited).

        Returns:
            Dict mapping texture names to descriptors.
        """
        return getattr(self.material_class, "_texture_bindings", {})

    def get_own_textures(self) -> Dict[str, Any]:
        """Get textures defined directly on this class.

        Returns:
            Dict mapping texture names to descriptors.
        """
        if hasattr(self.material_class, "get_own_textures"):
            return self.material_class.get_own_textures()

        all_tex = self.get_all_textures()
        inherited = self.get_inherited_textures()
        return {k: v for k, v in all_tex.items() if k not in inherited}

    def get_inherited_textures(self) -> Dict[str, Any]:
        """Get textures inherited from parent classes.

        Returns:
            Dict mapping texture names to descriptors.
        """
        return getattr(self.material_class, "_inherited_textures", {})

    def has_super_call(self) -> bool:
        """Check if the surface method calls super().surface().

        Returns:
            True if super() is called in surface().
        """
        return getattr(self.material_class, "_has_super_call", False) or self._detect_super_call()

    def _get_material_mro(self) -> List[type]:
        """Get MRO filtered to only Material subclasses."""
        from trinity.materials.dsl import Material

        result = []
        for cls in self.material_class.__mro__:
            if cls is Material:
                continue  # Skip base Material class
            if isinstance(cls, type) and issubclass(cls, Material):
                result.append(cls)
        return result

    def _find_parent_material(self) -> Optional[type]:
        """Find the first parent class with a surface method."""
        for base in self.material_class.__bases__:
            if hasattr(base, "_surface_method") and base._surface_method:
                return base
            if hasattr(base, "surface") and callable(getattr(base, "surface", None)):
                # Check if it's a real implementation, not the base Material.surface
                method = getattr(base, "surface")
                # Skip if it's just the stub from Material base
                if hasattr(method, "_is_surface") or base.__name__ != "Material":
                    return base
        return None

    def _build_inheritance_chain(self) -> List[type]:
        """Build the chain of materials with surface methods."""
        chain = [self.material_class]
        current = self.material_class

        while True:
            parent = getattr(current, "_parent_material", None)
            if parent is None:
                # Try to find it
                for base in current.__bases__:
                    if hasattr(base, "_surface_method") and base._surface_method:
                        parent = base
                        break

            if parent is None:
                break

            chain.append(parent)
            current = parent

        return chain

    def _collect_textures(self) -> TextureMergeResult:
        """Collect and merge texture declarations from inheritance tree."""
        from trinity.materials.textures import TextureDescriptor

        result = TextureMergeResult()

        # Collect from parent classes first (in MRO order)
        for base in reversed(self.material_class.__mro__[1:]):  # Skip self
            if hasattr(base, "_texture_bindings"):
                for name, desc in base._texture_bindings.items():
                    if name not in result.from_parent:
                        result.from_parent[name] = desc

        # Collect own textures
        for name in dir(self.material_class):
            try:
                attr = getattr(self.material_class, name)
                if isinstance(attr, TextureDescriptor):
                    if name in result.from_parent:
                        result.overridden[name] = attr
                    result.from_child[name] = attr
            except AttributeError:
                continue

        # Merge: parent textures, then override with child
        result.merged = dict(result.from_parent)
        result.merged.update(result.from_child)

        return result

    def _detect_super_call(self) -> bool:
        """Detect if surface() contains a super().surface() call."""
        surface_method = getattr(self.material_class, "_surface_method", None)
        if surface_method is None:
            surface_method = getattr(self.material_class, "surface", None)

        if surface_method is None:
            return False

        try:
            source = textwrap.dedent(inspect.getsource(surface_method))
            tree = ast.parse(source)

            detector = SuperCallDetector()
            detector.visit(tree)
            return detector.has_super_surface_call

        except (OSError, SyntaxError):
            return False


# =============================================================================
# AST VISITOR FOR SUPER() DETECTION
# =============================================================================


class SuperCallDetector(ast.NodeVisitor):
    """AST visitor that detects super().surface() calls."""

    def __init__(self):
        self.has_super_surface_call = False
        self.super_calls: List[SuperCallInfo] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Check for super().surface(ctx, out) pattern."""
        if self._is_super_surface_call(node):
            self.has_super_surface_call = True
            self.super_calls.append(SuperCallInfo(
                lineno=node.lineno,
                col_offset=node.col_offset,
                has_args=len(node.args) > 0,
                arg_count=len(node.args),
            ))

        self.generic_visit(node)

    def _is_super_surface_call(self, node: ast.Call) -> bool:
        """Check if a Call node is super().surface(...)."""
        if not isinstance(node.func, ast.Attribute):
            return False

        # Check if calling .surface method
        if node.func.attr != "surface":
            return False

        # Check if the object is a super() call
        if not isinstance(node.func.value, ast.Call):
            return False

        # Check if it's super()
        if isinstance(node.func.value.func, ast.Name):
            return node.func.value.func.id == "super"

        return False


# =============================================================================
# WGSL INLINING FUNCTIONS
# =============================================================================


def resolve_parent_surface(cls: type) -> Optional[str]:
    """Get the parent's surface() WGSL for a material class.

    Args:
        cls: Material class to find parent surface for

    Returns:
        Parent's WGSL source, or None if no parent
    """
    parent = getattr(cls, "_parent_material", None)
    if parent is None:
        resolver = InheritanceResolver(cls)
        parent = resolver.get_parent_material()

    if parent:
        return getattr(parent, "_wgsl_source", None)
    return None


def inline_super_call(child_wgsl: str, parent_wgsl: str, parent_name: str = "Parent") -> str:
    """Replace super() marker comment with actual parent WGSL.

    This function finds the marker comment left by PythonToWGSLTranslator
    when it encounters super().surface() and replaces it with the parent's
    surface shader body.

    Args:
        child_wgsl: Child's WGSL with super() marker
        parent_wgsl: Parent's WGSL to inline
        parent_name: Name of parent class for comments

    Returns:
        WGSL with super() replaced by parent code

    Example::

        child_wgsl = '''
            /* super().surface() -> Parent */;
            out.metallic = 0.9;
        '''
        parent_wgsl = '''
            out.roughness = 0.5;
        '''
        result = inline_super_call(child_wgsl, parent_wgsl, "BaseMaterial")
        # Result:
        # // BEGIN parent material (BaseMaterial)
        #     out.roughness = 0.5;
        # // END parent material
        # out.metallic = 0.9;
    """
    if not parent_wgsl:
        return child_wgsl

    # Prepare parent code with proper indentation
    parent_lines = parent_wgsl.strip().split("\n")
    indented_parent = "\n".join(f"    {line}" for line in parent_lines if line.strip())

    # Look for marker patterns
    super_marker = f"/* super().surface() -> {parent_name} */"

    replacement = (
        f"// BEGIN parent material ({parent_name})\n"
        f"{indented_parent}\n"
        f"    // END parent material"
    )

    # Try different marker formats
    patterns = [
        f"    {super_marker};",  # Indented with semicolon
        f"{super_marker};",     # No indent, with semicolon
        super_marker,           # Just the marker
    ]

    for pattern in patterns:
        if pattern in child_wgsl:
            return child_wgsl.replace(pattern, replacement)

    # If no marker found, return original
    return child_wgsl


def merge_texture_declarations(
    child_cls: type,
    parent_cls: type,
) -> Dict[str, Any]:
    """Merge texture declarations from child and parent classes.

    Child textures override parent textures with the same name.

    Args:
        child_cls: Child material class
        parent_cls: Parent material class

    Returns:
        Merged dict of texture name -> descriptor
    """
    merged = {}

    # Start with parent textures
    parent_bindings = getattr(parent_cls, "_texture_bindings", {})
    merged.update(parent_bindings)

    # Also include parent's inherited textures
    parent_inherited = getattr(parent_cls, "_inherited_textures", {})
    for name, desc in parent_inherited.items():
        if name not in merged:
            merged[name] = desc

    # Override with child textures
    child_bindings = getattr(child_cls, "_texture_bindings", {})
    for name, desc in child_bindings.items():
        merged[name] = desc

    return merged


# =============================================================================
# MRO WALKER
# =============================================================================


class MROWalker:
    """Walks the Method Resolution Order for material classes.

    Provides utilities for traversing the MRO and collecting information
    from parent classes in the correct order.

    Example::

        walker = MROWalker(DiamondChildMaterial)

        # Iterate over all material parents
        for cls in walker.walk_materials():
            print(cls.__name__)

        # Get linearized texture declarations
        textures = walker.collect_textures()

        # Check for diamond inheritance
        if walker.has_diamond_inheritance():
            print("Warning: Diamond inheritance detected")
    """

    def __init__(self, material_class: type):
        """Initialize walker for a material class.

        Args:
            material_class: Material class to walk
        """
        self.material_class = material_class
        self._mro = list(material_class.__mro__)

    def walk_materials(self) -> List[type]:
        """Get all Material subclasses in MRO order.

        Returns:
            List of Material subclasses (excluding base Material class)
        """
        from trinity.materials.dsl import Material

        result = []
        for cls in self._mro:
            if cls is Material or cls is object:
                continue
            if isinstance(cls, type) and issubclass(cls, Material):
                result.append(cls)
        return result

    def walk_parents(self) -> List[type]:
        """Get parent Material classes (excluding self).

        Returns:
            List of parent Material classes in MRO order
        """
        materials = self.walk_materials()
        if materials and materials[0] is self.material_class:
            return materials[1:]
        return materials

    def collect_textures(self) -> Dict[str, Any]:
        """Collect all texture bindings in MRO order.

        Later classes in MRO (closer to current class) override earlier ones.

        Returns:
            Merged dict of texture bindings
        """
        merged = {}

        # Walk in reverse MRO order so later classes override
        for cls in reversed(self.walk_materials()):
            bindings = getattr(cls, "_texture_bindings", {})
            merged.update(bindings)

        return merged

    def collect_wgsl_chain(self) -> List[Tuple[type, str]]:
        """Collect WGSL sources from all materials in chain.

        Returns:
            List of (class, wgsl_source) tuples
        """
        result = []
        for cls in self.walk_materials():
            wgsl = getattr(cls, "_wgsl_source", "")
            if wgsl:
                result.append((cls, wgsl))
        return result

    def has_diamond_inheritance(self) -> bool:
        """Check if the inheritance tree has diamond inheritance.

        Diamond inheritance occurs when a class inherits from two or more
        classes that have a common ancestor (other than Material/object).

        Returns:
            True if diamond inheritance is detected
        """
        from trinity.materials.dsl import Material

        # Count occurrences of each class in the combined base MROs
        seen: Dict[type, int] = {}

        for base in self.material_class.__bases__:
            if not isinstance(base, type) or not issubclass(base, Material):
                continue
            if base is Material:
                continue

            for cls in base.__mro__:
                if cls is Material or cls is object:
                    continue
                if isinstance(cls, type) and issubclass(cls, Material):
                    seen[cls] = seen.get(cls, 0) + 1

        # Diamond exists if any class appears more than once
        return any(count > 1 for count in seen.values())

    def get_diamond_ancestors(self) -> List[type]:
        """Get the shared ancestor classes in diamond inheritance.

        Returns:
            List of classes that are inherited through multiple paths
        """
        from trinity.materials.dsl import Material

        seen: Dict[type, int] = {}

        for base in self.material_class.__bases__:
            if not isinstance(base, type) or not issubclass(base, Material):
                continue
            if base is Material:
                continue

            for cls in base.__mro__:
                if cls is Material or cls is object:
                    continue
                if isinstance(cls, type) and issubclass(cls, Material):
                    seen[cls] = seen.get(cls, 0) + 1

        return [cls for cls, count in seen.items() if count > 1]


# =============================================================================
# COMBINED WGSL OUTPUT
# =============================================================================


def generate_combined_wgsl(material_class: type, include_comments: bool = True) -> str:
    """Generate WGSL that includes all inherited surface logic.

    This function walks the inheritance chain and produces WGSL that shows
    the full resolved shader including all super() inlining.

    Args:
        material_class: Material class to generate WGSL for
        include_comments: Add comments showing inheritance structure

    Returns:
        Complete WGSL for the material's surface shader
    """
    wgsl = getattr(material_class, "_wgsl_source", "")

    if not include_comments:
        return wgsl

    # Add inheritance info as comments
    resolver = InheritanceResolver(material_class)
    chain = resolver.get_inheritance_chain()

    header = f"// Material: {material_class.__name__}\n"
    if len(chain) > 1:
        chain_names = " -> ".join(cls.__name__ for cls in chain)
        header += f"// Inheritance: {chain_names}\n"

    if resolver.has_super_call():
        header += "// Uses super().surface() to extend parent shader\n"

    textures = resolver.get_all_textures()
    if textures:
        header += f"// Texture slots: {', '.join(textures.keys())}\n"

    header += "\n"

    return header + wgsl


def validate_inheritance(material_class: type) -> List[str]:
    """Validate the inheritance structure of a material class.

    Checks for common issues:
    - super() call without parent surface
    - Unresolved texture conflicts
    - Deep inheritance chains (performance warning)

    Args:
        material_class: Material class to validate

    Returns:
        List of warning/error messages (empty if valid)
    """
    issues = []
    resolver = InheritanceResolver(material_class)

    # Check for super() without parent
    if resolver.has_super_call():
        parent = resolver.get_parent_material()
        if parent is None:
            issues.append(
                f"ERROR: {material_class.__name__} calls super().surface() "
                "but has no parent with a surface method"
            )
        elif not getattr(parent, "_wgsl_source", None):
            issues.append(
                f"WARNING: {material_class.__name__} calls super().surface() "
                f"but parent {parent.__name__} has no compiled WGSL"
            )

    # Check inheritance depth
    chain = resolver.get_inheritance_chain()
    if len(chain) > 5:
        issues.append(
            f"WARNING: Deep inheritance chain ({len(chain)} levels) may impact "
            "shader compilation performance"
        )

    # Check for diamond inheritance
    walker = MROWalker(material_class)
    if walker.has_diamond_inheritance():
        ancestors = walker.get_diamond_ancestors()
        names = ", ".join(cls.__name__ for cls in ancestors)
        issues.append(
            f"WARNING: Diamond inheritance detected. Shared ancestors: {names}. "
            "Texture slots from first base in MRO take precedence."
        )

    # Check for compilation errors
    error = getattr(material_class, "_compilation_error", None)
    if error:
        issues.append(f"ERROR: Compilation failed: {error}")

    return issues


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Exceptions
    "InheritanceError",
    "NoParentSurfaceError",
    "CircularInheritanceError",
    "DiamondConflictError",
    # Data classes
    "InheritanceInfo",
    "TextureMergeResult",
    "SuperCallInfo",
    # Main classes
    "InheritanceResolver",
    "SuperCallDetector",
    "MROWalker",
    # Functions
    "resolve_parent_surface",
    "inline_super_call",
    "merge_texture_declarations",
    "generate_combined_wgsl",
    "validate_inheritance",
]
