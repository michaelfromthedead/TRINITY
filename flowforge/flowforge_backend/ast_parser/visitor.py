"""AST Visitor for Trinity game engine code parsing.

This module provides the TrinityASTVisitor class for parsing Python code
and extracting structured information about classes, methods, and their
relationships to Trinity ECS concepts.

It includes special handling for Trinity decorators:
- @component: Data containers attached to entities
- @system: Logic that processes entities with specific components
- @resource: Singleton shared data containers
- @event: Signals/triggers with optional payload data
"""

from __future__ import annotations

import ast
import re
from typing import Any, Optional

from .constants import (
    COMPONENT_BASE_CLASSES,
    DEFAULT_SOURCE_NAME,
    EVENT_BASE_CLASSES,
    RESOURCE_BASE_CLASSES,
    SYSTEM_BASE_CLASSES,
    TRINITY_DECORATOR_NAMES,
    TRINITY_MODULE,
    UNKNOWN_ANNOTATION,
    UNKNOWN_DECORATOR,
    UNKNOWN_DEFAULT,
)
from .types import (
    ClassDef,
    ComponentDef,
    DecoratorArgs,
    EventDef,
    FieldDef,
    ImportDef,
    ImportInfo,
    MethodDef,
    MethodKind,
    ModuleDef,
    ParameterDef,
    ParseResult,
    QueryComponentInfo,
    ResourceDef,
    SystemDef,
    TrinityDecoratorType,
    TrinityDef,
    TrinityImportInfo,
)


# =============================================================================
# CONSTANTS - Visitor-specific mappings
# =============================================================================

# Mapping from decorator names to TrinityDecoratorType enum
# (Uses enum values, unlike the string mapping in constants.py)
DECORATOR_TYPE_MAP = {
    "component": TrinityDecoratorType.COMPONENT,
    "system": TrinityDecoratorType.SYSTEM,
    "resource": TrinityDecoratorType.RESOURCE,
    "event": TrinityDecoratorType.EVENT,
}


class ImportTracker:
    """Tracks and manages import information for a module.

    This class maintains a registry of all imports encountered during
    AST traversal and provides methods to resolve names to their source
    modules and detect Trinity decorators.

    NOTE: This class is currently NOT used by TrinityASTVisitor, which has
    its own internal _trinity_imports dict. This is a cleaner abstraction
    that should either be integrated into TrinityASTVisitor or removed.
    See TODO in TrinityASTVisitor for details.
    """

    def __init__(self) -> None:
        """Initialize an empty import tracker."""
        self.imports: list[ImportInfo] = []
        self.trinity_imports: list[TrinityImportInfo] = []
        self._name_to_import: dict[str, ImportInfo] = {}
        self._name_to_trinity: dict[str, TrinityImportInfo] = {}
        self._type_imports: dict[str, str] = {}

    def add_import(self, import_info: ImportInfo) -> None:
        """Add an import to the tracker."""
        self.imports.append(import_info)
        self._name_to_import[import_info.effective_name] = import_info
        self._track_type_import(import_info)

    def add_trinity_import(self, trinity_info: TrinityImportInfo) -> None:
        """Add a Trinity-specific import to the tracker."""
        self.trinity_imports.append(trinity_info)
        self._name_to_trinity[trinity_info.alias] = trinity_info

    def _track_type_import(self, import_info: ImportInfo) -> None:
        """Track imports that may be used for type annotations."""
        typing_modules = {"typing", "typing_extensions", "collections.abc"}
        if import_info.module in typing_modules or import_info.name in typing_modules:
            source = import_info.module if import_info.is_from_import else import_info.name
            self._type_imports[import_info.effective_name] = source or import_info.name
        elif import_info.module == "dataclasses" or import_info.name == "dataclasses":
            self._type_imports[import_info.effective_name] = "dataclasses"
        elif import_info.is_from_import:
            module = import_info.module or ""
            if import_info.is_relative:
                module = "." * import_info.level + module
            self._type_imports[import_info.effective_name] = module

    def resolve_name(self, name: str) -> Optional[ImportInfo]:
        """Resolve a name to its import information."""
        return self._name_to_import.get(name)

    def resolve_module(self, name: str) -> Optional[str]:
        """Resolve a name to its source module."""
        info = self._name_to_import.get(name)
        return info.module if info and info.is_from_import else (info.name if info else None)

    def resolve_type_source(self, type_name: str) -> Optional[str]:
        """Resolve a type name to its source module."""
        return self._type_imports.get(type_name)

    def is_trinity_decorator(self, name: str) -> bool:
        """Check if a name refers to a Trinity decorator."""
        return name in self._name_to_trinity

    def get_trinity_decorator_type(self, name: str) -> Optional[TrinityDecoratorType]:
        """Get the Trinity decorator type for a name."""
        info = self._name_to_trinity.get(name)
        return info.decorator_type if info else None

    def is_trinity_module_imported(self) -> bool:
        """Check if the trinity module itself is imported."""
        return any(not i.is_from_import and i.name == TRINITY_MODULE for i in self.imports)

    def get_trinity_module_alias(self) -> Optional[str]:
        """Get the alias for the trinity module if imported directly."""
        for info in self.imports:
            if not info.is_from_import and info.name == TRINITY_MODULE:
                return info.effective_name
        return None

    def is_qualified_trinity_decorator(self, qualified_name: str) -> bool:
        """Check if a qualified name is a Trinity decorator."""
        if "." not in qualified_name:
            return self.is_trinity_decorator(qualified_name)
        parts = qualified_name.split(".", 1)
        if len(parts) != 2:
            return False
        alias = self.get_trinity_module_alias()
        return alias and parts[0] == alias and parts[1] in TRINITY_DECORATOR_NAMES

    def get_qualified_trinity_decorator_type(self, qualified_name: str) -> Optional[TrinityDecoratorType]:
        """Get the decorator type for a qualified Trinity decorator name."""
        if "." not in qualified_name:
            return self.get_trinity_decorator_type(qualified_name)
        parts = qualified_name.split(".", 1)
        if len(parts) != 2:
            return None
        alias = self.get_trinity_module_alias()
        if alias and parts[0] == alias and parts[1] in TRINITY_DECORATOR_NAMES:
            return TrinityImportInfo.from_name(parts[1]).decorator_type
        return None

    def list_all_imports(self) -> list[ImportInfo]:
        """Get all tracked imports."""
        return list(self.imports)

    def list_trinity_imports(self) -> list[TrinityImportInfo]:
        """Get all Trinity-specific imports."""
        return list(self.trinity_imports)

    def list_type_imports(self) -> dict[str, str]:
        """Get all type-related imports."""
        return dict(self._type_imports)

    def to_dict(self) -> dict[str, Any]:
        """Convert tracker state to dictionary for serialization."""
        return {
            "imports": [imp.to_dict() for imp in self.imports],
            "trinity_imports": [ti.to_dict() for ti in self.trinity_imports],
            "type_imports": self._type_imports,
        }


class TrinityASTVisitor(ast.NodeVisitor):
    """AST visitor specialized for parsing Trinity game engine code.

    This visitor extracts structured information from Python source code,
    with special handling for Trinity ECS patterns such as System classes
    and Query[...] type annotations.

    It also detects Trinity decorators (@component, @system, @resource, @event)
    and creates the appropriate definition types.

    Example:
        visitor = TrinityASTVisitor()
        module_def = visitor.parse(source_code)
        for class_def in module_def.classes:
            print(f"Class: {class_def.name}")
            for method in class_def.methods:
                print(f"  Method: {method.name}")

        # Get Trinity definitions
        trinity_result = visitor.get_trinity_result()
        for component in trinity_result.components:
            print(f"Component: {component.name}")

    Attributes:
        source_file: Path to the source file being parsed (optional).

    TODO: Architectural improvements needed:
        1. Consolidate ImportInfo (types.py:25) and ImportDef (types.py:335)
           into a single type - they serve similar purposes.
        2. Consider using ImportTracker class instead of internal _trinity_imports
           dict for cleaner separation of concerns.
        3. Consider deprecating parse() -> ModuleDef in favor of just
           get_trinity_result() -> ParseResult since that's the main use case.
    """

    # Pattern for extracting component types from Query[A, B, C] annotations
    _QUERY_PATTERN = re.compile(r"Query\[([^\]]+)\]")

    def __init__(self, source_file: Optional[str] = None) -> None:
        """Initialize the visitor.

        Args:
            source_file: Optional path to the source file being parsed.
        """
        self.source_file = source_file
        self.current_file = source_file  # Alias for compatibility
        self._classes: list[ClassDef] = []
        self._module_docstring: Optional[str] = None
        self._imports: list[str] = []
        self._import_defs: list[ImportDef] = []  # Structured import definitions
        self._errors: list[str] = []

        # Trinity-specific tracking
        self._components: list[ComponentDef] = []
        self._systems: list[SystemDef] = []
        self._resources: list[ResourceDef] = []
        self._events: list[EventDef] = []
        self._trinity_imports: dict[str, TrinityImportInfo] = {}

    @property
    def components(self) -> list[ComponentDef]:
        """Get the list of extracted component definitions."""
        return self._components

    @property
    def systems(self) -> list[SystemDef]:
        """Get the list of extracted system definitions."""
        return self._systems

    @property
    def resources(self) -> list[ResourceDef]:
        """Get the list of extracted resource definitions."""
        return self._resources

    @property
    def events(self) -> list[EventDef]:
        """Get the list of extracted event definitions."""
        return self._events

    @property
    def imports(self) -> list[ImportDef]:
        """Get the list of structured import definitions."""
        return self._import_defs

    @property
    def errors(self) -> list[str]:
        """Get the list of parsing errors."""
        return self._errors

    def get_trinity_result(self) -> ParseResult:
        """Get the parse result containing Trinity definitions.

        Returns:
            ParseResult containing all parsed Trinity definitions.
        """
        return ParseResult(
            source_file=self.source_file or "",
            components=self._components,
            systems=self._systems,
            resources=self._resources,
            events=self._events,
            imports=self._import_defs,
            errors=self._errors,
        )

    def parse(self, source: str, file_path: Optional[str] = None) -> ModuleDef:
        """Parse Python source code and return a ModuleDef.

        Args:
            source: Python source code as a string
            file_path: Optional file path for context (updates current_file)

        Returns:
            ModuleDef containing all extracted information

        Raises:
            SyntaxError: If the source code has syntax errors
        """
        # Update current file if provided
        if file_path is not None:
            self.source_file = file_path
            self.current_file = file_path

        # Reset all state
        self._classes = []
        self._module_docstring = None
        self._imports = []
        self._import_defs = []
        self._errors = []
        self._components = []
        self._systems = []
        self._resources = []
        self._events = []
        self._trinity_imports = {}

        tree = ast.parse(source)

        # Extract module docstring
        self._module_docstring = ast.get_docstring(tree)

        # Visit all nodes
        self.visit(tree)

        return ModuleDef(
            classes=tuple(self._classes),
            docstring=self._module_docstring,
            imports=tuple(self._imports),
        )

    def visit_Import(self, node: ast.Import) -> None:
        """Visit an import statement and track Trinity module imports."""
        for alias in node.names:
            if alias.asname:
                self._imports.append(f"import {alias.name} as {alias.asname}")
            else:
                self._imports.append(f"import {alias.name}")

            # Create structured ImportDef
            import_name = alias.asname if alias.asname else alias.name.split(".")[-1]
            self._import_defs.append(ImportDef(
                module=alias.name,
                names=(import_name,),
                is_from_import=False,
                line_number=node.lineno,
            ))

            # Track trinity module imports
            if alias.name == "trinity" or alias.name.startswith("trinity."):
                effective_name = alias.asname or alias.name
                # If importing the whole module, we need to handle trinity.component, etc.
                self._trinity_imports[effective_name] = TrinityImportInfo.from_name(
                    alias.name, alias.asname
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit a from...import statement and track Trinity decorator imports."""
        module = node.module or ""
        names = []
        name_list: list[str] = []
        for alias in node.names:
            if alias.asname:
                names.append(f"{alias.name} as {alias.asname}")
                name_list.append(alias.asname)
            else:
                names.append(alias.name)
                name_list.append(alias.name)
        level = "." * node.level
        self._imports.append(f"from {level}{module} import {', '.join(names)}")

        # Create structured ImportDef
        self._import_defs.append(ImportDef(
            module=f"{level}{module}" if level else module,
            names=tuple(name_list),
            is_from_import=True,
            line_number=node.lineno,
        ))

        # Track Trinity decorator imports
        if module == "trinity" or module.startswith("trinity."):
            for alias in node.names:
                name = alias.name
                effective_name = alias.asname or name
                if name in TRINITY_DECORATOR_NAMES:
                    self._trinity_imports[effective_name] = TrinityImportInfo.from_name(
                        name, alias.asname
                    )

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition and check for Trinity decorators.

        This method examines each class for Trinity decorators and creates
        the appropriate definition type (ComponentDef, SystemDef, etc.).

        Handles three decorator forms:
        - Simple: @component
        - Called: @component()
        - With args: @component(priority=1)

        Args:
            node: The ClassDef AST node to visit.
        """
        # Extract base class names
        bases = tuple(self._annotation_to_string(base) for base in node.bases)

        # Check if this is a System class (inherits from System or has @system decorator)
        is_system = bool(SYSTEM_BASE_CLASSES & set(bases))

        # Check for Trinity decorators early to determine if this is a System
        trinity_decorator = self._find_trinity_decorator(node.decorator_list)
        if trinity_decorator is not None:
            decorator_type, _ = trinity_decorator
            if decorator_type == TrinityDecoratorType.SYSTEM:
                is_system = True

        # Extract decorator names (as strings)
        decorators = tuple(self._decorator_to_string(d) for d in node.decorator_list)

        # Extract fields
        fields = self._extract_fields(node.body)

        # Extract methods (with Query extraction enabled for system classes)
        methods = self._extract_methods(node.body, is_system=is_system)

        # Get docstring
        docstring = ast.get_docstring(node)

        # Create the standard ClassDef
        class_def = ClassDef(
            name=node.name,
            bases=bases,
            fields=fields,
            methods=methods,
            docstring=docstring,
            line_number=node.lineno,
            decorators=decorators,
            is_system=is_system,
        )
        self._classes.append(class_def)

        # Use the already-found Trinity decorator (don't search again)
        if trinity_decorator is not None:
            decorator_type, decorator_args = trinity_decorator
            try:
                trinity_def = self._create_trinity_def(
                    decorator_type=decorator_type,
                    name=node.name,
                    docstring=docstring,
                    line_number=node.lineno,
                    decorator_args=decorator_args,
                    fields=fields,
                    methods=methods,
                )
                self._add_trinity_def(trinity_def)
            except Exception as e:
                self._errors.append(
                    f"Error parsing Trinity class '{node.name}' at line {node.lineno}: {e}"
                )

        # Continue visiting nested classes
        self.generic_visit(node)

    def _find_trinity_decorator(
        self, decorators: list[ast.expr]
    ) -> tuple[TrinityDecoratorType, DecoratorArgs] | None:
        """Find a Trinity decorator in the decorator list.

        Searches through decorators to find @component, @system, @resource,
        or @event. Handles various decorator forms:
        - Simple name: @component
        - Called without args: @component()
        - Called with args: @component(priority=1)
        - Attribute access: @trinity.component

        Args:
            decorators: List of decorator AST nodes.

        Returns:
            Tuple of (TrinityDecoratorType, DecoratorArgs) if found, None otherwise.
        """
        for decorator in decorators:
            result = self._parse_trinity_decorator(decorator)
            if result is not None:
                return result
        return None

    def _parse_trinity_decorator(
        self, decorator: ast.expr
    ) -> tuple[TrinityDecoratorType, DecoratorArgs] | None:
        """Parse a single decorator node to check if it's a Trinity decorator.

        Handles three forms:
        - ast.Name: @component
        - ast.Call with ast.Name func: @component() or @component(priority=1)
        - ast.Attribute: @trinity.component

        Args:
            decorator: The decorator AST node.

        Returns:
            Tuple of (TrinityDecoratorType, DecoratorArgs) if it's a Trinity
            decorator, None otherwise.
        """
        # Simple decorator: @component
        if isinstance(decorator, ast.Name):
            name = decorator.id
            if name in TRINITY_DECORATOR_NAMES:
                return DECORATOR_TYPE_MAP[name], DecoratorArgs()
            # Check if it's an aliased import
            if name in self._trinity_imports:
                info = self._trinity_imports[name]
                if info.decorator_type != TrinityDecoratorType.UNKNOWN:
                    return info.decorator_type, DecoratorArgs()

        # Called decorator: @component() or @component(priority=1)
        elif isinstance(decorator, ast.Call):
            func = decorator.func

            # @component(...) where component is a simple name
            if isinstance(func, ast.Name):
                name = func.id
                if name in TRINITY_DECORATOR_NAMES:
                    args = self._extract_decorator_args(decorator)
                    return DECORATOR_TYPE_MAP[name], args
                # Check aliased imports
                if name in self._trinity_imports:
                    info = self._trinity_imports[name]
                    if info.decorator_type != TrinityDecoratorType.UNKNOWN:
                        args = self._extract_decorator_args(decorator)
                        return info.decorator_type, args

            # @trinity.component(...) where trinity.component is an attribute
            elif isinstance(func, ast.Attribute):
                attr_name = func.attr
                if attr_name in TRINITY_DECORATOR_NAMES:
                    # Verify the value is 'trinity' or an alias of it
                    if self._is_trinity_module_reference(func.value):
                        args = self._extract_decorator_args(decorator)
                        return DECORATOR_TYPE_MAP[attr_name], args

        # Attribute access without call: @trinity.component
        elif isinstance(decorator, ast.Attribute):
            attr_name = decorator.attr
            if attr_name in TRINITY_DECORATOR_NAMES:
                if self._is_trinity_module_reference(decorator.value):
                    return DECORATOR_TYPE_MAP[attr_name], DecoratorArgs()

        return None

    def _is_trinity_module_reference(self, node: ast.expr) -> bool:
        """Check if an AST node refers to the trinity module.

        Args:
            node: The AST expression node to check.

        Returns:
            True if the node refers to 'trinity' or an alias of it.
        """
        if isinstance(node, ast.Name):
            name = node.id
            # Check for 'trinity' or any alias that imports trinity
            if name == "trinity":
                return True
            if name in self._trinity_imports:
                info = self._trinity_imports[name]
                return info.decorator_name == "trinity" or info.alias == "trinity"
        return False

    def _extract_decorator_args(self, call_node: ast.Call) -> DecoratorArgs:
        """Extract arguments from a decorator call.

        Args:
            call_node: The Call AST node representing the decorator call.

        Returns:
            DecoratorArgs with positional and keyword arguments.
        """
        positional: list[Any] = []
        keyword: dict[str, Any] = {}

        # Extract positional arguments
        for arg in call_node.args:
            value = self._eval_constant(arg)
            positional.append(value)

        # Extract keyword arguments
        for kw in call_node.keywords:
            if kw.arg is not None:
                value = self._eval_constant(kw.value)
                keyword[kw.arg] = value

        return DecoratorArgs(positional=positional, keyword=keyword)

    def _eval_constant(self, node: ast.expr) -> Any:
        """Safely evaluate a constant expression from AST.

        Args:
            node: The AST expression node.

        Returns:
            The evaluated value, or a string representation for complex expressions.
        """
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.List):
            return [self._eval_constant(elt) for elt in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_constant(elt) for elt in node.elts)
        if isinstance(node, ast.Dict):
            return {
                self._eval_constant(k) if k else None: self._eval_constant(v)
                for k, v in zip(node.keys, node.values)
            }
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self._eval_constant(node.operand)
            if isinstance(value, (int, float)):
                return -value

        # For complex expressions, return the AST dump as a string
        return ast.unparse(node)

    def _create_trinity_def(
        self,
        decorator_type: TrinityDecoratorType,
        name: str,
        docstring: Optional[str],
        line_number: int,
        decorator_args: DecoratorArgs,
        fields: tuple[FieldDef, ...],
        methods: tuple[MethodDef, ...],
    ) -> TrinityDef:
        """Create the appropriate TrinityDef subclass.

        Args:
            decorator_type: The type of Trinity decorator.
            name: The class name.
            docstring: The class docstring.
            line_number: Source line number.
            decorator_args: Arguments passed to the decorator.
            fields: Tuple of field definitions.
            methods: Tuple of method definitions.

        Returns:
            The appropriate TrinityDef subclass instance.
        """
        base_kwargs: dict[str, Any] = {
            "name": name,
            "decorator_type": decorator_type,
            "docstring": docstring,
            "line_number": line_number,
            "decorator_args": decorator_args,
            "fields": fields,
            "methods": methods,
            "source_file": self.source_file,
        }

        if decorator_type == TrinityDecoratorType.COMPONENT:
            return ComponentDef(**base_kwargs)

        elif decorator_type == TrinityDecoratorType.SYSTEM:
            # Extract queries from method signatures (look for Query[...] parameters)
            queries = self._extract_queries_from_methods(methods)
            return SystemDef(**base_kwargs, queries=queries)

        elif decorator_type == TrinityDecoratorType.RESOURCE:
            is_singleton = decorator_args.keyword.get("singleton", True)
            return ResourceDef(**base_kwargs, is_singleton=is_singleton)

        elif decorator_type == TrinityDecoratorType.EVENT:
            # Event fields are also the payload
            return EventDef(**base_kwargs, payload_fields=fields)

        else:
            return TrinityDef(**base_kwargs)

    def _extract_queries_from_methods(
        self, methods: tuple[MethodDef, ...]
    ) -> tuple[str, ...]:
        """Extract query types from method signatures.

        Looks for parameters with Query[...] type annotations and extracts
        the component types from them.

        Args:
            methods: Tuple of method definitions.

        Returns:
            Tuple of query type names (component types).
        """
        queries: list[str] = []

        for method in methods:
            if method.query_info is not None:
                for comp_type in method.query_info.component_types:
                    if comp_type not in queries:
                        queries.append(comp_type)

        return tuple(queries)

    def _add_trinity_def(self, trinity_def: TrinityDef) -> None:
        """Add a Trinity definition to the appropriate list.

        Args:
            trinity_def: The Trinity definition to add.
        """
        if isinstance(trinity_def, ComponentDef):
            self._components.append(trinity_def)
        elif isinstance(trinity_def, SystemDef):
            self._systems.append(trinity_def)
        elif isinstance(trinity_def, ResourceDef):
            self._resources.append(trinity_def)
        elif isinstance(trinity_def, EventDef):
            self._events.append(trinity_def)

    def _extract_fields(self, class_body: list[ast.stmt]) -> tuple[FieldDef, ...]:
        """Extract field definitions from a class body.

        Fields are annotated assignments in the class body, such as:
            name: str
            x: float = 0.0
            items: list[Item] = field(default_factory=list)

        Args:
            class_body: List of AST statements from the class body

        Returns:
            Tuple of FieldDef objects representing the extracted fields
        """
        fields: list[FieldDef] = []

        for stmt in class_body:
            if not isinstance(stmt, ast.AnnAssign):
                continue

            # Only handle simple name targets (not subscripts or attributes)
            if not isinstance(stmt.target, ast.Name):
                continue

            name = stmt.target.id
            type_annotation = self._annotation_to_string(stmt.annotation)

            default_value = None
            if stmt.value is not None:
                default_value = self._default_to_string(stmt.value)

            field_def = FieldDef(
                name=name,
                type_annotation=type_annotation,
                default_value=default_value,
                line_number=stmt.lineno,
            )
            fields.append(field_def)

        return tuple(fields)

    def _extract_methods(
        self, class_body: list[ast.stmt], *, is_system: bool = False
    ) -> tuple[MethodDef, ...]:
        """Extract method definitions from a class body.

        Args:
            class_body: List of AST statements from the class body
            is_system: Whether this class is a Trinity System (enables Query extraction)

        Returns:
            Tuple of MethodDef objects representing the extracted methods
        """
        methods: list[MethodDef] = []

        for stmt in class_body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            method_def = self._extract_single_method(stmt, is_system=is_system)
            methods.append(method_def)

        return tuple(methods)

    def _extract_single_method(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        is_system: bool = False,
    ) -> MethodDef:
        """Extract a single method definition from a function node.

        Args:
            node: The AST FunctionDef or AsyncFunctionDef node
            is_system: Whether the containing class is a Trinity System

        Returns:
            MethodDef representing the extracted method
        """
        # Extract decorator names and determine method kind
        decorators = tuple(self._decorator_to_string(d) for d in node.decorator_list)
        kind = self._determine_method_kind(decorators)

        # Extract parameters
        parameters, query_info = self._extract_parameters(
            node.args, kind=kind, is_system=is_system
        )

        # Extract return type
        return_type = None
        if node.returns is not None:
            return_type = self._annotation_to_string(node.returns)

        # Extract docstring
        docstring = ast.get_docstring(node)

        return MethodDef(
            name=node.name,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            line_number=node.lineno,
            kind=kind,
            query_info=query_info,
            decorators=decorators,
        )

    def _determine_method_kind(self, decorators: tuple[str, ...]) -> MethodKind:
        """Determine the method kind based on decorators.

        Args:
            decorators: Tuple of decorator names

        Returns:
            MethodKind indicating the type of method
        """
        for decorator in decorators:
            if decorator == "classmethod":
                return MethodKind.CLASS_METHOD
            if decorator == "staticmethod":
                return MethodKind.STATIC_METHOD
        return MethodKind.REGULAR

    def _extract_parameters(
        self,
        args: ast.arguments,
        *,
        kind: MethodKind,
        is_system: bool = False,
    ) -> tuple[tuple[ParameterDef, ...], Optional[QueryComponentInfo]]:
        """Extract parameter definitions from function arguments.

        This method handles:
        - Regular positional arguments
        - *args and **kwargs
        - Default values
        - Type annotations
        - Skipping self/cls for instance/class methods

        Args:
            args: The AST arguments node
            kind: The method kind (determines if self/cls should be skipped)
            is_system: Whether to extract Query info from type annotations

        Returns:
            Tuple of (parameters, query_info) where query_info is set if
            a Query[...] annotation is found in a System method
        """
        parameters: list[ParameterDef] = []
        query_info: Optional[QueryComponentInfo] = None

        # Calculate how many parameters have defaults
        # defaults apply to the last N positional args
        num_defaults = len(args.defaults)
        num_args = len(args.args)
        first_default_idx = num_args - num_defaults

        # Determine which parameters to skip (self/cls)
        skip_first = kind in (MethodKind.REGULAR, MethodKind.CLASS_METHOD)

        # Process positional arguments
        for idx, arg in enumerate(args.args):
            # Skip self/cls parameter
            if idx == 0 and skip_first:
                continue

            param, param_query_info = self._extract_single_parameter(
                arg, args.defaults, idx, first_default_idx, is_system=is_system
            )
            parameters.append(param)

            # Track Query info (use first one found)
            if param_query_info is not None and query_info is None:
                query_info = param_query_info

        # Process positional-only arguments (Python 3.8+)
        for idx, arg in enumerate(args.posonlyargs):
            # Skip self/cls parameter (would be first)
            if idx == 0 and skip_first and not args.args:
                continue

            param, param_query_info = self._extract_single_parameter(
                arg, [], -1, -1, is_system=is_system
            )
            parameters.append(param)

            if param_query_info is not None and query_info is None:
                query_info = param_query_info

        # Process keyword-only arguments
        for idx, arg in enumerate(args.kwonlyargs):
            default = args.kw_defaults[idx] if idx < len(args.kw_defaults) else None
            type_annotation = None
            if arg.annotation is not None:
                type_annotation = self._annotation_to_string(arg.annotation)

            default_value = None
            if default is not None:
                default_value = self._default_to_string(default)

            param = ParameterDef(
                name=arg.arg,
                type_annotation=type_annotation,
                default_value=default_value,
            )
            parameters.append(param)

            # Check for Query in annotation
            if is_system and type_annotation is not None:
                param_query_info = self._extract_query_info(type_annotation)
                if param_query_info is not None and query_info is None:
                    query_info = param_query_info

        # Process *args
        if args.vararg is not None:
            type_annotation = None
            if args.vararg.annotation is not None:
                type_annotation = self._annotation_to_string(args.vararg.annotation)

            param = ParameterDef(
                name=f"*{args.vararg.arg}",
                type_annotation=type_annotation,
                default_value=None,
            )
            parameters.append(param)

        # Process **kwargs
        if args.kwarg is not None:
            type_annotation = None
            if args.kwarg.annotation is not None:
                type_annotation = self._annotation_to_string(args.kwarg.annotation)

            param = ParameterDef(
                name=f"**{args.kwarg.arg}",
                type_annotation=type_annotation,
                default_value=None,
            )
            parameters.append(param)

        return tuple(parameters), query_info

    def _extract_single_parameter(
        self,
        arg: ast.arg,
        defaults: list[ast.expr],
        idx: int,
        first_default_idx: int,
        *,
        is_system: bool = False,
    ) -> tuple[ParameterDef, Optional[QueryComponentInfo]]:
        """Extract a single parameter definition.

        Args:
            arg: The AST arg node
            defaults: List of default value expressions
            idx: Index of this argument in the argument list
            first_default_idx: Index of the first argument with a default
            is_system: Whether to extract Query info from type annotations

        Returns:
            Tuple of (ParameterDef, optional QueryComponentInfo)
        """
        type_annotation = None
        if arg.annotation is not None:
            type_annotation = self._annotation_to_string(arg.annotation)

        # Calculate default value if any
        default_value = None
        if first_default_idx >= 0 and idx >= first_default_idx:
            default_idx = idx - first_default_idx
            if default_idx < len(defaults):
                default_value = self._default_to_string(defaults[default_idx])

        param = ParameterDef(
            name=arg.arg,
            type_annotation=type_annotation,
            default_value=default_value,
        )

        # Extract Query info if this is a System method
        query_info = None
        if is_system and type_annotation is not None:
            query_info = self._extract_query_info(type_annotation)

        return param, query_info

    def _extract_query_info(self, annotation: str) -> Optional[QueryComponentInfo]:
        """Extract component types from a Query[...] type annotation.

        Args:
            annotation: The type annotation string

        Returns:
            QueryComponentInfo if the annotation contains Query[...], else None
        """
        match = self._QUERY_PATTERN.search(annotation)
        if match is None:
            return None

        # Extract the content inside Query[...]
        inner = match.group(1)

        # Split by comma and clean up whitespace
        components = tuple(comp.strip() for comp in inner.split(","))

        return QueryComponentInfo(
            component_types=components,
            raw_annotation=annotation,
        )

    def _annotation_to_string(self, node: ast.expr) -> str:
        """Convert an AST annotation node to its string representation.

        Args:
            node: An AST expression node representing a type annotation

        Returns:
            String representation of the annotation
        """
        # Python 3.9+ has ast.unparse
        try:
            return ast.unparse(node)
        except AttributeError:
            # Fallback for older Python versions
            return self._unparse_annotation(node)

    def _unparse_annotation(self, node: ast.expr) -> str:
        """Manually unparse an annotation node (fallback for Python < 3.9).

        Args:
            node: An AST expression node

        Returns:
            String representation
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Attribute):
            value = self._unparse_annotation(node.value)
            return f"{value}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            value = self._unparse_annotation(node.value)
            slice_val = self._unparse_annotation(node.slice)
            return f"{value}[{slice_val}]"
        elif isinstance(node, ast.Tuple):
            elements = ", ".join(self._unparse_annotation(e) for e in node.elts)
            return elements
        elif isinstance(node, ast.List):
            elements = ", ".join(self._unparse_annotation(e) for e in node.elts)
            return f"[{elements}]"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Union type with | operator (Python 3.10+)
            left = self._unparse_annotation(node.left)
            right = self._unparse_annotation(node.right)
            return f"{left} | {right}"
        elif isinstance(node, ast.Call):
            # Handle callable types and other complex constructs
            func = self._unparse_annotation(node.func)
            args = ", ".join(self._unparse_annotation(a) for a in node.args)
            return f"{func}({args})"
        else:
            # Fallback: try to get source segment or return placeholder
            return UNKNOWN_ANNOTATION

    def _default_to_string(self, node: ast.expr) -> str:
        """Convert a default value AST node to its string representation.

        Args:
            node: An AST expression node representing a default value

        Returns:
            String representation of the default value
        """
        # Use ast.unparse for Python 3.9+
        try:
            return ast.unparse(node)
        except AttributeError:
            return self._unparse_default(node)

    def _unparse_default(self, node: ast.expr) -> str:
        """Manually unparse a default value node (fallback for Python < 3.9).

        Args:
            node: An AST expression node

        Returns:
            String representation
        """
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            elements = ", ".join(self._unparse_default(e) for e in node.elts)
            return f"[{elements}]"
        elif isinstance(node, ast.Dict):
            pairs = []
            for k, v in zip(node.keys, node.values):
                key = self._unparse_default(k) if k is not None else "**"
                val = self._unparse_default(v)
                pairs.append(f"{key}: {val}")
            return "{" + ", ".join(pairs) + "}"
        elif isinstance(node, ast.Tuple):
            elements = ", ".join(self._unparse_default(e) for e in node.elts)
            return f"({elements})"
        elif isinstance(node, ast.Set):
            elements = ", ".join(self._unparse_default(e) for e in node.elts)
            return "{" + elements + "}"
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return f"-{self._unparse_default(node.operand)}"
            elif isinstance(node.op, ast.UAdd):
                return f"+{self._unparse_default(node.operand)}"
            elif isinstance(node.op, ast.Not):
                return f"not {self._unparse_default(node.operand)}"
        elif isinstance(node, ast.Call):
            func = self._unparse_default(node.func)
            args = ", ".join(self._unparse_default(a) for a in node.args)
            return f"{func}({args})"
        elif isinstance(node, ast.Attribute):
            value = self._unparse_default(node.value)
            return f"{value}.{node.attr}"

        return UNKNOWN_DEFAULT

    def _decorator_to_string(self, node: ast.expr) -> str:
        """Convert a decorator AST node to its string representation.

        Args:
            node: An AST expression node representing a decorator

        Returns:
            String representation of the decorator (without @)
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._decorator_to_string(node.value)
            return f"{value}.{node.attr}"
        elif isinstance(node, ast.Call):
            func = self._decorator_to_string(node.func)
            # Include arguments in decorator representation
            try:
                args_str = ast.unparse(node)[len(func) + 1:-1]  # Remove func name and parens
                return f"{func}({args_str})"
            except AttributeError:
                # Fallback for older Python
                args = []
                for arg in node.args:
                    args.append(self._unparse_default(arg))
                for kw in node.keywords:
                    args.append(f"{kw.arg}={self._unparse_default(kw.value)}")
                return f"{func}({', '.join(args)})"
        else:
            try:
                return ast.unparse(node)
            except AttributeError:
                return UNKNOWN_DECORATOR


# =============================================================================
# Module-level convenience functions
# =============================================================================


def parse_source(source: str, source_file: str = DEFAULT_SOURCE_NAME) -> ParseResult:
    """Parse Python source code and extract Trinity definitions.

    This is a convenience function that creates a TrinityASTVisitor,
    parses the source, and returns a ParseResult with all Trinity
    definitions categorized by type.

    Args:
        source: Python source code as a string
        source_file: Optional path to the source file for error messages

    Returns:
        ParseResult containing components, systems, resources, and events

    Raises:
        SyntaxError: If the source code has syntax errors

    Example:
        result = parse_source('''
        from trinity import component

        @component
        class Position:
            x: float = 0.0
            y: float = 0.0
        ''')
        for comp in result.components:
            print(f"Component: {comp.name}")
    """
    visitor = TrinityASTVisitor(source_file=source_file)
    visitor.parse(source)
    return visitor.get_trinity_result()


def parse_file(file_path: str) -> ParseResult:
    """Parse a Python source file and extract Trinity definitions.

    This is a convenience function that reads a file and calls parse_source.

    Args:
        file_path: Path to the Python source file

    Returns:
        ParseResult containing components, systems, resources, and events

    Raises:
        FileNotFoundError: If the file does not exist
        SyntaxError: If the source code has syntax errors

    Example:
        result = parse_file("/path/to/game/components.py")
        for comp in result.components:
            print(f"Component: {comp.name}")
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        return parse_source(source, source_file=file_path)
    except OSError as e:
        result = ParseResult(source_file=file_path)
        result.errors.append(f"Could not read file: {e}")
        return result
