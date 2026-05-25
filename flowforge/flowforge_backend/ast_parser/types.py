"""Type definitions for AST parsing in FlowForge.

This module defines dataclasses used to represent parsed Python code elements
including methods, parameters, classes, and other AST constructs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
from typing_extensions import Self


class TrinityDecoratorType(Enum):
    """Types of Trinity decorators that can be detected."""
    COMPONENT = auto()
    SYSTEM = auto()
    RESOURCE = auto()
    EVENT = auto()
    UNKNOWN = auto()


@dataclass
class ImportInfo:
    """Information about a single import.

    Represents either:
    - A module import: `import X` or `import X as Y`
    - A from import: `from X import Y` or `from X import Y as Z`

    Attributes:
        name: The imported name (module name or imported identifier)
        module: Source module for from-imports, None for direct imports
        alias: The alias if using `as`, None otherwise
        is_from_import: True if this is a from-import, False for direct import
        level: Relative import level (0 = absolute, 1 = `.`, 2 = `..`, etc.)
        lineno: Line number where the import appears
        col_offset: Column offset where the import appears
    """
    name: str
    module: Optional[str] = None
    alias: Optional[str] = None
    is_from_import: bool = False
    level: int = 0
    lineno: int = 0
    col_offset: int = 0

    @property
    def effective_name(self) -> str:
        """Get the name as it would be used in code (alias if present, else name)."""
        return self.alias if self.alias else self.name

    @property
    def is_relative(self) -> bool:
        """Check if this is a relative import."""
        return self.level > 0

    @property
    def full_module_path(self) -> str:
        """Get the full module path for the import.

        For direct imports, returns the module name.
        For from imports, returns module.name or just module.
        For relative imports, includes the leading dots.
        """
        prefix = "." * self.level if self.level > 0 else ""
        if self.module:
            return f"{prefix}{self.module}"
        return f"{prefix}{self.name}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "name": self.name,
            "is_from_import": self.is_from_import,
            "effective_name": self.effective_name,
            "lineno": self.lineno,
        }
        if self.module is not None:
            result["module"] = self.module
        if self.alias is not None:
            result["alias"] = self.alias
        if self.level > 0:
            result["level"] = self.level
        return result


@dataclass
class TrinityImportInfo:
    """Tracks Trinity-specific import information.

    Used to detect and track imports from the trinity module,
    including any aliases that might be used for decorators.

    Attributes:
        decorator_name: The decorator name as imported (e.g., 'component')
        alias: The alias used in code if aliased, else same as decorator_name
        decorator_type: The type of Trinity decorator this represents
    """
    decorator_name: str
    alias: str
    decorator_type: TrinityDecoratorType

    @classmethod
    def from_name(cls, name: str, alias: Optional[str] = None) -> Self:
        """Create TrinityImportInfo from a decorator name.

        Args:
            name: The decorator name (component, system, resource, event)
            alias: Optional alias used in import

        Returns:
            New TrinityImportInfo instance
        """
        decorator_map = {
            "component": TrinityDecoratorType.COMPONENT,
            "system": TrinityDecoratorType.SYSTEM,
            "resource": TrinityDecoratorType.RESOURCE,
            "event": TrinityDecoratorType.EVENT,
        }
        decorator_type = decorator_map.get(name, TrinityDecoratorType.UNKNOWN)
        return cls(
            decorator_name=name,
            alias=alias if alias else name,
            decorator_type=decorator_type,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decorator_name": self.decorator_name,
            "alias": self.alias,
            "decorator_type": self.decorator_type.name.lower(),
        }


class MethodKind(Enum):
    """Classification of method types."""
    REGULAR = auto()      # Regular instance method with self
    CLASS_METHOD = auto() # Class method with cls (decorated with @classmethod)
    STATIC_METHOD = auto() # Static method (decorated with @staticmethod)


@dataclass(frozen=True)
class ParameterDef:
    """Represents a parameter definition in a method signature.

    Attributes:
        name: The parameter name
        type_annotation: The type annotation as a string, or None if untyped
        default_value: The default value as a string representation, or None
    """
    name: str
    type_annotation: Optional[str] = None
    default_value: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {"name": self.name}
        if self.type_annotation is not None:
            result["type_annotation"] = self.type_annotation
        if self.default_value is not None:
            result["default_value"] = self.default_value
        return result


@dataclass(frozen=True)
class FieldDef:
    """Represents a class field definition extracted from AST.

    Handles annotated assignments in class bodies such as:
        name: str
        x: float = 0.0
        items: list[Item] = field(default_factory=list)

    Attributes:
        name: The field name (target of the assignment)
        type_annotation: String representation of the type annotation
        default_value: String representation of the default value, or None
        line_number: The source line number where the field is defined
    """
    name: str
    type_annotation: str
    default_value: Optional[str] = None
    line_number: int = 0

    def has_default(self) -> bool:
        """Check if this field has a default value."""
        return self.default_value is not None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result: dict = {
            "name": self.name,
            "type_annotation": self.type_annotation,
            "line_number": self.line_number,
        }
        if self.default_value is not None:
            result["default_value"] = self.default_value
        return result


@dataclass(frozen=True)
class QueryComponentInfo:
    """Information about components extracted from a Query[...] type annotation.

    Attributes:
        component_types: List of component type names from the Query generic
        raw_annotation: The original full type annotation string
    """
    component_types: tuple[str, ...]
    raw_annotation: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "component_types": list(self.component_types),
            "raw_annotation": self.raw_annotation,
        }


@dataclass(frozen=True)
class MethodDef:
    """Represents a method definition extracted from a class.

    Attributes:
        name: The method name
        parameters: List of parameter definitions (excluding self/cls)
        return_type: The return type annotation as a string, or None
        docstring: The method's docstring, or None if absent
        line_number: The line number where the method is defined
        kind: The type of method (regular, class method, static method)
        query_info: For System methods, info about Query[...] parameters
        decorators: List of decorator names applied to this method
    """
    name: str
    parameters: tuple[ParameterDef, ...]
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    line_number: int = 0
    kind: MethodKind = MethodKind.REGULAR
    query_info: Optional[QueryComponentInfo] = None
    decorators: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "parameters": [p.to_dict() for p in self.parameters],
            "line_number": self.line_number,
            "kind": self.kind.name.lower(),
        }
        if self.return_type is not None:
            result["return_type"] = self.return_type
        if self.docstring is not None:
            result["docstring"] = self.docstring
        if self.query_info is not None:
            result["query_info"] = self.query_info.to_dict()
        if self.decorators:
            result["decorators"] = list(self.decorators)
        return result


@dataclass(frozen=True)
class ClassDef:
    """Represents a class definition extracted from source code.

    Attributes:
        name: The class name
        bases: Tuple of base class names
        fields: Tuple of field definitions (annotated class attributes)
        methods: Tuple of method definitions
        docstring: The class's docstring, or None if absent
        line_number: The line number where the class is defined
        decorators: List of decorator names applied to this class
        is_system: True if this class inherits from System (Trinity ECS)
    """
    name: str
    bases: tuple[str, ...] = field(default_factory=tuple)
    fields: tuple[FieldDef, ...] = field(default_factory=tuple)
    methods: tuple[MethodDef, ...] = field(default_factory=tuple)
    docstring: Optional[str] = None
    line_number: int = 0
    decorators: tuple[str, ...] = field(default_factory=tuple)
    is_system: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "bases": list(self.bases),
            "fields": [f.to_dict() for f in self.fields],
            "methods": [m.to_dict() for m in self.methods],
            "line_number": self.line_number,
            "is_system": self.is_system,
        }
        if self.docstring is not None:
            result["docstring"] = self.docstring
        if self.decorators:
            result["decorators"] = list(self.decorators)
        return result


@dataclass(frozen=True)
class ModuleDef:
    """Represents a parsed Python module.

    Attributes:
        classes: Tuple of class definitions in the module
        docstring: The module's docstring, or None if absent
        imports: Tuple of import statements (as strings)
    """
    classes: tuple[ClassDef, ...] = field(default_factory=tuple)
    docstring: Optional[str] = None
    imports: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "classes": [c.to_dict() for c in self.classes],
        }
        if self.docstring is not None:
            result["docstring"] = self.docstring
        if self.imports:
            result["imports"] = list(self.imports)
        return result


# =============================================================================
# Import Definitions
# =============================================================================

@dataclass(frozen=True)
class ImportDef:
    """Represents an import statement for Trinity ECS parsing.

    Attributes:
        module: The module being imported (e.g., "trinity.ecs")
        names: Tuple of imported names (e.g., ("Component", "System"))
        is_from_import: True if this is a 'from X import Y' statement
        line_number: The source line number of this import
    """
    module: str
    names: tuple[str, ...] = field(default_factory=tuple)
    is_from_import: bool = False
    line_number: int = 0

    def imports_name(self, name: str) -> bool:
        """Check if this import brings in a specific name."""
        if self.is_from_import:
            return name in self.names
        return name == self.module.split(".")[-1]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "module": self.module,
            "names": list(self.names),
            "is_from_import": self.is_from_import,
            "line_number": self.line_number,
        }


# -----------------------------------------------------------------------------
# Trinity-specific definition types for @component, @system, @resource, @event
# -----------------------------------------------------------------------------


@dataclass
class DecoratorArgs:
    """Represents arguments passed to a Trinity decorator.

    Handles three forms:
    - Simple: @component -> no args
    - Called: @component() -> empty args
    - With args: @component(priority=1) -> keyword args

    Attributes:
        positional: List of positional arguments.
        keyword: Dictionary of keyword arguments.
    """
    positional: list[Any] = field(default_factory=list)
    keyword: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "positional": self.positional,
            "keyword": self.keyword,
        }


@dataclass
class TrinityDef:
    """Base class for Trinity definition types.

    Represents a class decorated with a Trinity decorator (@component, @system,
    @resource, @event).

    Attributes:
        name: The class name.
        decorator_type: The type of Trinity decorator applied.
        docstring: The class docstring if present.
        line_number: Source line number where the class is defined.
        decorator_args: Arguments passed to the decorator.
        fields: Tuple of field definitions.
        methods: Tuple of method definitions.
        source_file: The source file path (set by the parser).
    """
    name: str
    decorator_type: TrinityDecoratorType
    docstring: Optional[str] = None
    line_number: int = 0
    decorator_args: DecoratorArgs = field(default_factory=DecoratorArgs)
    fields: tuple[FieldDef, ...] = field(default_factory=tuple)
    methods: tuple[MethodDef, ...] = field(default_factory=tuple)
    source_file: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "name": self.name,
            "decorator_type": self.decorator_type.name.lower(),
            "line_number": self.line_number,
            "fields": [f.to_dict() for f in self.fields],
            "methods": [m.to_dict() for m in self.methods],
        }
        if self.docstring is not None:
            result["docstring"] = self.docstring
        if self.decorator_args.positional or self.decorator_args.keyword:
            result["decorator_args"] = self.decorator_args.to_dict()
        if self.source_file is not None:
            result["source_file"] = self.source_file
        return result


@dataclass
class ComponentDef(TrinityDef):
    """Represents a @component decorated class.

    Components are data containers that can be attached to entities.
    They hold the data/state but no logic.
    """

    def __post_init__(self) -> None:
        if self.decorator_type == TrinityDecoratorType.UNKNOWN:
            object.__setattr__(self, 'decorator_type', TrinityDecoratorType.COMPONENT)


@dataclass
class SystemDef(TrinityDef):
    """Represents a @system decorated class.

    Systems process entities with specific component queries.
    They contain the logic that operates on components.

    Attributes:
        queries: Tuple of query type names this system operates on,
                 extracted from Query[...] type annotations.
    """
    queries: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.decorator_type == TrinityDecoratorType.UNKNOWN:
            object.__setattr__(self, 'decorator_type', TrinityDecoratorType.SYSTEM)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        if self.queries:
            result["queries"] = list(self.queries)
        return result


@dataclass
class ResourceDef(TrinityDef):
    """Represents a @resource decorated class.

    Resources are singleton shared data containers accessible
    globally throughout the application.

    Attributes:
        is_singleton: Whether this resource is a singleton (default True).
    """
    is_singleton: bool = True

    def __post_init__(self) -> None:
        if self.decorator_type == TrinityDecoratorType.UNKNOWN:
            object.__setattr__(self, 'decorator_type', TrinityDecoratorType.RESOURCE)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        result["is_singleton"] = self.is_singleton
        return result


@dataclass
class EventDef(TrinityDef):
    """Represents a @event decorated class.

    Events are signals/triggers with optional payload data.
    They enable decoupled communication between systems.

    Attributes:
        payload_fields: Tuple of fields that make up the event payload.
    """
    payload_fields: tuple[FieldDef, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.decorator_type == TrinityDecoratorType.UNKNOWN:
            object.__setattr__(self, 'decorator_type', TrinityDecoratorType.EVENT)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        if self.payload_fields:
            result["payload_fields"] = [f.to_dict() for f in self.payload_fields]
        return result


@dataclass
class ParseResult:
    """Result of parsing a Python source file for Trinity definitions.

    Attributes:
        source_file: The path to the parsed source file.
        components: List of parsed ComponentDef instances.
        systems: List of parsed SystemDef instances.
        resources: List of parsed ResourceDef instances.
        events: List of parsed EventDef instances.
        imports: List of parsed ImportInfo instances.
        errors: List of parsing errors encountered.
    """
    source_file: str
    components: list[ComponentDef] = field(default_factory=list)
    systems: list[SystemDef] = field(default_factory=list)
    resources: list[ResourceDef] = field(default_factory=list)
    events: list[EventDef] = field(default_factory=list)
    imports: list[ImportDef] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def all_definitions(self) -> list[TrinityDef]:
        """Return all Trinity definitions in a single list."""
        result: list[TrinityDef] = []
        result.extend(self.components)
        result.extend(self.systems)
        result.extend(self.resources)
        result.extend(self.events)
        return result

    @property
    def has_errors(self) -> bool:
        """Check if any parsing errors occurred."""
        return len(self.errors) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_file": self.source_file,
            "components": [c.to_dict() for c in self.components],
            "systems": [s.to_dict() for s in self.systems],
            "resources": [r.to_dict() for r in self.resources],
            "events": [e.to_dict() for e in self.events],
            "imports": [i.to_dict() for i in self.imports],
            "errors": self.errors,
        }
