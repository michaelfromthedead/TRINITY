"""
Debuggable Decorator - Auto-generates debug UI for decorated classes.

Provides:
- @debuggable class decorator for marking classes for auto-inspection
- DebugField for field-level metadata (min/max, step, readonly, widget)
- DebugSection for grouping fields into collapsible sections
- DebugConfig for class-level debug UI configuration
- Integration with AutoInspector from debug_ui.py
- Dynamic visibility control via show_if conditions
- Custom widget overrides per field

Usage:
    @debuggable(title="My Component", expanded=True)
    class MyComponent:
        health: int = field(default=100, metadata=debug_field(min=0, max=100, step=1))
        name: str = field(default="", metadata=debug_field(widget="text"))
        _internal: float = 0.0  # Hidden by default (underscore prefix)
"""
from __future__ import annotations

import functools
import inspect
import weakref
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from engine.tooling.editor.app_shell import editor, reloadable

# Import debug_ui components
from engine.tooling.editor.debug_ui import (
    AutoInspector,
    CheckboxWidget,
    CollapsibleSection,
    Color,
    ColorPickerWidget,
    DebugUI,
    DebugUIContext,
    DropdownWidget,
    FloatInputWidget,
    FloatSliderWidget,
    IntInputWidget,
    IntSliderWidget,
    PropertyBinding,
    PropertyPanel,
    TextInputWidget,
    Vec2,
    Vec2InputWidget,
    Vec3,
    Vec3InputWidget,
    Vec4,
    Vec4InputWidget,
    Widget,
    WidgetConfig,
    WidgetRegistry,
    WidgetType,
)


T = TypeVar("T")


# =============================================================================
# Debug Field Configuration
# =============================================================================


class WidgetHint(Enum):
    """Hints for which widget type to use for a field."""
    AUTO = auto()  # Let registry choose based on type
    SLIDER = auto()  # Force slider for numeric
    INPUT = auto()  # Force input box for numeric
    TEXT = auto()  # Text input
    TEXT_AREA = auto()  # Multi-line text
    CHECKBOX = auto()  # Boolean checkbox
    DROPDOWN = auto()  # Dropdown/combo
    COLOR = auto()  # Color picker
    VEC2 = auto()  # 2D vector
    VEC3 = auto()  # 3D vector
    VEC4 = auto()  # 4D vector
    BUTTON = auto()  # Action button
    LABEL = auto()  # Read-only label
    CUSTOM = auto()  # Custom widget class


@dataclass
class DebugFieldConfig:
    """Configuration for a single debuggable field."""
    # Display
    label: Optional[str] = None  # Display label (defaults to field name)
    tooltip: str = ""  # Tooltip text
    order: int = 0  # Display order (lower = first)

    # Constraints
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    precision: int = 3  # Decimal places for floats

    # Widget control
    widget: WidgetHint = WidgetHint.AUTO
    custom_widget: Optional[Type[Widget]] = None
    widget_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Visibility
    hidden: bool = False
    readonly: bool = False
    show_if: Optional[Callable[[Any], bool]] = None  # Dynamic visibility

    # Grouping
    section: Optional[str] = None  # Section name to group into

    # Validation
    validator: Optional[Callable[[Any], bool]] = None
    on_change: Optional[Callable[[Any, Any], None]] = None  # (old, new)

    # Dropdown options
    choices: Optional[List[Any]] = None
    choice_labels: Optional[Dict[Any, str]] = None


def debug_field(
    label: Optional[str] = None,
    tooltip: str = "",
    order: int = 0,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    step: Optional[Union[int, float]] = None,
    precision: int = 3,
    widget: WidgetHint = WidgetHint.AUTO,
    custom_widget: Optional[Type[Widget]] = None,
    widget_kwargs: Optional[Dict[str, Any]] = None,
    hidden: bool = False,
    readonly: bool = False,
    show_if: Optional[Callable[[Any], bool]] = None,
    section: Optional[str] = None,
    validator: Optional[Callable[[Any], bool]] = None,
    on_change: Optional[Callable[[Any, Any], None]] = None,
    choices: Optional[List[Any]] = None,
    choice_labels: Optional[Dict[Any, str]] = None,
) -> Dict[str, Any]:
    """
    Create debug field metadata for use with dataclass field().

    Example:
        @debuggable()
        @dataclass
        class Player:
            health: int = field(
                default=100,
                metadata=debug_field(min_value=0, max_value=100, step=5)
            )
    """
    config = DebugFieldConfig(
        label=label,
        tooltip=tooltip,
        order=order,
        min_value=min_value,
        max_value=max_value,
        step=step,
        precision=precision,
        widget=widget,
        custom_widget=custom_widget,
        widget_kwargs=widget_kwargs or {},
        hidden=hidden,
        readonly=readonly,
        show_if=show_if,
        section=section,
        validator=validator,
        on_change=on_change,
        choices=choices,
        choice_labels=choice_labels,
    )
    return {"debug_field": config}


# =============================================================================
# Debug Section Configuration
# =============================================================================


@dataclass
class DebugSectionConfig:
    """Configuration for a debug UI section."""
    name: str
    label: Optional[str] = None  # Display label (defaults to name)
    expanded: bool = True
    order: int = 0  # Section display order
    icon: str = ""
    collapsible: bool = True
    show_if: Optional[Callable[[Any], bool]] = None


def debug_section(
    name: str,
    label: Optional[str] = None,
    expanded: bool = True,
    order: int = 0,
    icon: str = "",
    collapsible: bool = True,
    show_if: Optional[Callable[[Any], bool]] = None,
) -> DebugSectionConfig:
    """Create a section configuration."""
    return DebugSectionConfig(
        name=name,
        label=label,
        expanded=expanded,
        order=order,
        icon=icon,
        collapsible=collapsible,
        show_if=show_if,
    )


# =============================================================================
# Debug Config (Class-level)
# =============================================================================


@dataclass
class DebugConfig:
    """Class-level debug UI configuration."""
    title: Optional[str] = None  # Panel title (defaults to class name)
    expanded: bool = True
    auto_sync: bool = True  # Auto-sync widget values
    exclude_private: bool = True  # Exclude _private fields
    exclude_dunder: bool = True  # Exclude __dunder__ fields
    exclude_fields: Set[str] = field(default_factory=set)
    include_fields: Optional[Set[str]] = None  # If set, only these fields
    sections: List[DebugSectionConfig] = field(default_factory=list)
    default_section: Optional[str] = None  # Section for unsectioned fields
    icon: str = ""
    category: str = "General"  # Category for grouping in UI

    def add_section(self, section: DebugSectionConfig) -> None:
        """Add a section configuration."""
        self.sections.append(section)

    def get_section(self, name: str) -> Optional[DebugSectionConfig]:
        """Get section by name."""
        for section in self.sections:
            return section if section.name == name else None
        return None


# =============================================================================
# Debuggable Class Registry
# =============================================================================


@editor(category="DebugUI")
@reloadable()
class DebuggableRegistry:
    """Registry of all debuggable classes and their configurations."""
    __slots__ = ("_classes", "_configs", "_panels", "_inspector_cache")

    _instance: Optional["DebuggableRegistry"] = None

    def __init__(self):
        self._classes: Dict[type, DebugConfig] = {}
        self._configs: Dict[type, Dict[str, DebugFieldConfig]] = {}
        self._panels: Dict[int, "DebuggablePanel"] = {}
        self._inspector_cache: Dict[type, List[Tuple[str, type, DebugFieldConfig]]] = {}

    @classmethod
    def get_instance(cls) -> "DebuggableRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def register(
        self,
        cls: type,
        config: DebugConfig,
        field_configs: Dict[str, DebugFieldConfig],
    ) -> None:
        """Register a debuggable class."""
        self._classes[cls] = config
        self._configs[cls] = field_configs
        # Clear cache for this class
        self._inspector_cache.pop(cls, None)

    def unregister(self, cls: type) -> bool:
        """Unregister a debuggable class."""
        if cls in self._classes:
            del self._classes[cls]
            self._configs.pop(cls, None)
            self._inspector_cache.pop(cls, None)
            return True
        return False

    def is_debuggable(self, cls_or_obj: Union[type, Any]) -> bool:
        """Check if class or instance is debuggable."""
        cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)
        return cls in self._classes

    def get_config(self, cls: type) -> Optional[DebugConfig]:
        """Get debug config for class."""
        return self._classes.get(cls)

    def get_field_configs(self, cls: type) -> Dict[str, DebugFieldConfig]:
        """Get field configs for class."""
        return self._configs.get(cls, {})

    def get_field_config(self, cls: type, field_name: str) -> Optional[DebugFieldConfig]:
        """Get config for specific field."""
        configs = self._configs.get(cls, {})
        return configs.get(field_name)

    def list_classes(self) -> List[type]:
        """List all registered debuggable classes."""
        return list(self._classes.keys())

    def list_classes_by_category(self, category: str) -> List[type]:
        """List classes in a specific category."""
        return [
            cls for cls, config in self._classes.items()
            if config.category == category
        ]

    def get_categories(self) -> Set[str]:
        """Get all registered categories."""
        return {config.category for config in self._classes.values()}

    def get_inspectable_fields(
        self, cls: type
    ) -> List[Tuple[str, type, DebugFieldConfig]]:
        """Get list of (field_name, field_type, config) for inspection."""
        if cls in self._inspector_cache:
            return self._inspector_cache[cls]

        config = self._classes.get(cls)
        field_configs = self._configs.get(cls, {})

        if config is None:
            return []

        result = []
        hints = {}
        try:
            hints = get_type_hints(cls)
        except (TypeError, NameError):
            pass

        # Gather all field names
        field_names = set(hints.keys())

        # For dataclasses, also check fields()
        if is_dataclass(cls):
            for f in fields(cls):
                field_names.add(f.name)

        # For regular classes, check __dict__ of class
        if hasattr(cls, "__annotations__"):
            field_names.update(cls.__annotations__.keys())

        for name in field_names:
            # Check exclusions
            if config.exclude_private and name.startswith("_") and not name.startswith("__"):
                continue
            if config.exclude_dunder and name.startswith("__"):
                continue
            if name in config.exclude_fields:
                continue
            if config.include_fields is not None and name not in config.include_fields:
                continue

            field_type = hints.get(name, Any)
            field_cfg = field_configs.get(name, DebugFieldConfig())

            if field_cfg.hidden:
                continue

            result.append((name, field_type, field_cfg))

        # Sort by order, then alphabetically
        result.sort(key=lambda x: (x[2].order, x[0]))

        self._inspector_cache[cls] = result
        return result

    def get_panel(self, obj: Any) -> Optional["DebuggablePanel"]:
        """Get cached panel for object."""
        return self._panels.get(id(obj))

    def cache_panel(self, obj: Any, panel: "DebuggablePanel") -> None:
        """Cache panel for object."""
        self._panels[id(obj)] = panel

    def clear_panel_cache(self) -> None:
        """Clear all cached panels."""
        self._panels.clear()


# =============================================================================
# Debuggable Panel
# =============================================================================


@editor(category="DebugUI")
@reloadable()
class DebuggablePanel:
    """Panel generated from a @debuggable class instance."""
    __slots__ = (
        "_target_ref", "_cls", "_config", "_field_configs",
        "_widgets", "_sections", "_bindings", "_property_panel",
        "_dirty_fields", "_on_change_callbacks"
    )

    def __init__(
        self,
        target: Any,
        config: DebugConfig,
        field_configs: Dict[str, DebugFieldConfig],
    ):
        self._target_ref = weakref.ref(target)
        self._cls = type(target)
        self._config = config
        self._field_configs = field_configs
        self._widgets: Dict[str, Widget] = {}
        self._sections: Dict[str, CollapsibleSection] = {}
        self._bindings: Dict[str, PropertyBinding] = {}
        self._property_panel: Optional[PropertyPanel] = None
        self._dirty_fields: Set[str] = set()
        self._on_change_callbacks: Dict[str, List[Callable[[Any, Any], None]]] = {}

        self._build_panel()

    @property
    def target(self) -> Optional[Any]:
        """Get target object."""
        return self._target_ref()

    @property
    def config(self) -> DebugConfig:
        """Get debug config."""
        return self._config

    @property
    def title(self) -> str:
        """Get panel title."""
        return self._config.title or self._cls.__name__

    @property
    def widgets(self) -> Dict[str, Widget]:
        """Get all widgets by field name."""
        return dict(self._widgets)

    @property
    def sections(self) -> Dict[str, CollapsibleSection]:
        """Get all sections by name."""
        return dict(self._sections)

    @property
    def property_panel(self) -> Optional[PropertyPanel]:
        """Get underlying property panel."""
        return self._property_panel

    def _build_panel(self) -> None:
        """Build the panel widgets from config."""
        registry = DebuggableRegistry.get_instance()
        inspectable = registry.get_inspectable_fields(self._cls)

        # Create property panel
        self._property_panel = PropertyPanel(
            title=self.title,
            target=self.target,
            auto_sync=self._config.auto_sync,
        )

        # Create sections
        for section_cfg in self._config.sections:
            section = CollapsibleSection(
                title=section_cfg.label or section_cfg.name,
                expanded=section_cfg.expanded,
                icon=section_cfg.icon,
            )
            self._sections[section_cfg.name] = section

        # Create default section if needed
        if self._config.default_section and self._config.default_section not in self._sections:
            self._sections[self._config.default_section] = CollapsibleSection(
                title=self._config.default_section,
                expanded=True,
            )

        # Create widgets for each field
        for field_name, field_type, field_cfg in inspectable:
            widget = self._create_widget_for_field(field_name, field_type, field_cfg)
            if widget is None:
                continue

            self._widgets[field_name] = widget

            # Create binding
            target = self.target
            if target is not None:
                value = getattr(target, field_name, None)
                widget.set_value(value)

                binding = self._property_panel.add_property(
                    field_name,
                    widget,
                    readonly=field_cfg.readonly,
                )
                self._bindings[field_name] = binding

                # Set up on_change callback
                if field_cfg.on_change:
                    self._register_on_change(field_name, field_cfg.on_change)

                # Set up value change tracking
                original_on_change = widget.on_change
                def make_change_handler(fname: str, original: Optional[Callable]):
                    def handler(new_value: Any):
                        self._on_field_changed(fname, new_value)
                        if original:
                            original(new_value)
                    return handler
                widget.on_change = make_change_handler(field_name, original_on_change)

            # Add to section if specified
            section_name = field_cfg.section or self._config.default_section
            if section_name and section_name in self._sections:
                self._sections[section_name].add_child(widget)

    def _create_widget_for_field(
        self,
        field_name: str,
        field_type: type,
        field_cfg: DebugFieldConfig,
    ) -> Optional[Widget]:
        """Create appropriate widget for field."""
        label = field_cfg.label or field_name.replace("_", " ").title()

        # Custom widget override
        if field_cfg.custom_widget is not None:
            kwargs = {"label": label, **field_cfg.widget_kwargs}
            return field_cfg.custom_widget(**kwargs)

        # Get initial value
        target = self.target
        initial_value = None
        if target is not None:
            try:
                initial_value = getattr(target, field_name, None)
            except (AttributeError, TypeError):
                pass

        # Determine widget based on hint and type
        hint = field_cfg.widget

        # Handle enums -> dropdown
        origin = get_origin(field_type)
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            return self._create_enum_widget(field_type, label, initial_value, field_cfg)

        # Handle explicit choices -> dropdown
        if field_cfg.choices is not None:
            return self._create_choices_widget(label, initial_value, field_cfg)

        # Widget hint overrides
        if hint == WidgetHint.LABEL:
            from engine.tooling.editor.debug_ui import LabelWidget
            return LabelWidget(str(initial_value) if initial_value else "")

        if hint == WidgetHint.COLOR:
            return ColorPickerWidget(label, initial_value if isinstance(initial_value, Color) else Color())

        if hint == WidgetHint.VEC2:
            return Vec2InputWidget(label, initial_value if isinstance(initial_value, Vec2) else Vec2())

        if hint == WidgetHint.VEC3:
            return Vec3InputWidget(label, initial_value if isinstance(initial_value, Vec3) else Vec3())

        if hint == WidgetHint.VEC4:
            return Vec4InputWidget(label, initial_value if isinstance(initial_value, Vec4) else Vec4())

        # Type-based widget selection
        if field_type == bool or hint == WidgetHint.CHECKBOX:
            return CheckboxWidget(label, bool(initial_value) if initial_value is not None else False)

        if field_type == str or hint == WidgetHint.TEXT:
            return TextInputWidget(label, str(initial_value) if initial_value else "")

        if field_type == int:
            return self._create_int_widget(label, initial_value, field_cfg, hint)

        if field_type == float:
            return self._create_float_widget(label, initial_value, field_cfg, hint)

        # Vector types
        if field_type == Vec2:
            return Vec2InputWidget(label, initial_value if isinstance(initial_value, Vec2) else Vec2())

        if field_type == Vec3:
            return Vec3InputWidget(label, initial_value if isinstance(initial_value, Vec3) else Vec3())

        if field_type == Vec4:
            return Vec4InputWidget(label, initial_value if isinstance(initial_value, Vec4) else Vec4())

        if field_type == Color:
            return ColorPickerWidget(label, initial_value if isinstance(initial_value, Color) else Color())

        # Fallback to text input
        return TextInputWidget(label, str(initial_value) if initial_value else "")

    def _create_int_widget(
        self,
        label: str,
        initial_value: Any,
        field_cfg: DebugFieldConfig,
        hint: WidgetHint,
    ) -> Widget:
        """Create integer widget."""
        value = int(initial_value) if initial_value is not None else 0
        min_val = int(field_cfg.min_value) if field_cfg.min_value is not None else None
        max_val = int(field_cfg.max_value) if field_cfg.max_value is not None else None
        step_val = int(field_cfg.step) if field_cfg.step is not None else 1

        # Decide slider vs input
        use_slider = hint == WidgetHint.SLIDER
        if hint == WidgetHint.AUTO and min_val is not None and max_val is not None:
            use_slider = (max_val - min_val) <= 1000

        if use_slider and min_val is not None and max_val is not None:
            return IntSliderWidget(label, value, min_val, max_val, step_val)
        else:
            return IntInputWidget(label, value, min_val, max_val, step_val)

    def _create_float_widget(
        self,
        label: str,
        initial_value: Any,
        field_cfg: DebugFieldConfig,
        hint: WidgetHint,
    ) -> Widget:
        """Create float widget."""
        value = float(initial_value) if initial_value is not None else 0.0
        min_val = float(field_cfg.min_value) if field_cfg.min_value is not None else None
        max_val = float(field_cfg.max_value) if field_cfg.max_value is not None else None
        step_val = float(field_cfg.step) if field_cfg.step is not None else 0.1
        precision = field_cfg.precision

        # Decide slider vs input
        use_slider = hint == WidgetHint.SLIDER
        if hint == WidgetHint.AUTO and min_val is not None and max_val is not None:
            use_slider = (max_val - min_val) <= 100

        if use_slider and min_val is not None and max_val is not None:
            return FloatSliderWidget(label, value, min_val, max_val, step_val, precision)
        else:
            return FloatInputWidget(label, value, min_val, max_val, step_val, precision)

    def _create_enum_widget(
        self,
        enum_type: Type[Enum],
        label: str,
        initial_value: Any,
        field_cfg: DebugFieldConfig,
    ) -> Widget:
        """Create dropdown for enum type."""
        options = [e.name for e in enum_type]
        selected = 0
        if initial_value is not None:
            try:
                selected = options.index(initial_value.name)
            except (ValueError, AttributeError):
                pass
        return DropdownWidget(label, options, selected)

    def _create_choices_widget(
        self,
        label: str,
        initial_value: Any,
        field_cfg: DebugFieldConfig,
    ) -> Widget:
        """Create dropdown from explicit choices."""
        choices = field_cfg.choices or []
        labels = field_cfg.choice_labels or {}
        options = [labels.get(c, str(c)) for c in choices]

        selected = 0
        if initial_value is not None and initial_value in choices:
            selected = choices.index(initial_value)

        return DropdownWidget(label, options, selected)

    def _on_field_changed(self, field_name: str, new_value: Any) -> None:
        """Handle field value change."""
        self._dirty_fields.add(field_name)

        # Call registered callbacks
        callbacks = self._on_change_callbacks.get(field_name, [])
        target = self.target
        if target is not None:
            old_value = getattr(target, field_name, None)
            for cb in callbacks:
                try:
                    cb(old_value, new_value)
                except Exception:
                    pass

        # Update visibility of dependent fields
        self._update_visibility()

    def _register_on_change(
        self, field_name: str, callback: Callable[[Any, Any], None]
    ) -> None:
        """Register on_change callback for field."""
        if field_name not in self._on_change_callbacks:
            self._on_change_callbacks[field_name] = []
        self._on_change_callbacks[field_name].append(callback)

    def _update_visibility(self) -> None:
        """Update visibility of fields based on show_if conditions."""
        target = self.target
        if target is None:
            return

        registry = DebuggableRegistry.get_instance()
        inspectable = registry.get_inspectable_fields(self._cls)

        for field_name, _, field_cfg in inspectable:
            if field_cfg.show_if is not None:
                widget = self._widgets.get(field_name)
                if widget is not None:
                    try:
                        visible = field_cfg.show_if(target)
                        widget.visible = visible
                    except Exception:
                        pass

        # Update section visibility
        for section_cfg in self._config.sections:
            if section_cfg.show_if is not None:
                section = self._sections.get(section_cfg.name)
                if section is not None:
                    try:
                        visible = section_cfg.show_if(target)
                        section.visible = visible
                    except Exception:
                        pass

    def sync_from_target(self) -> None:
        """Update all widgets from target values."""
        target = self.target
        if target is None:
            return

        for field_name, widget in self._widgets.items():
            try:
                value = getattr(target, field_name, None)
                widget.set_value(value)
            except (AttributeError, TypeError):
                pass

        self._dirty_fields.clear()
        self._update_visibility()

    def sync_to_target(self) -> None:
        """Update target from widget values."""
        target = self.target
        if target is None:
            return

        for field_name, binding in self._bindings.items():
            if not binding.readonly:
                binding.sync_to_target()

    def get_widget(self, field_name: str) -> Optional[Widget]:
        """Get widget for field."""
        return self._widgets.get(field_name)

    def get_section(self, section_name: str) -> Optional[CollapsibleSection]:
        """Get section by name."""
        return self._sections.get(section_name)

    def set_field_visible(self, field_name: str, visible: bool) -> None:
        """Set field visibility."""
        widget = self._widgets.get(field_name)
        if widget is not None:
            widget.visible = visible

    def set_field_enabled(self, field_name: str, enabled: bool) -> None:
        """Set field enabled state."""
        widget = self._widgets.get(field_name)
        if widget is not None:
            widget.enabled = enabled

    def set_section_expanded(self, section_name: str, expanded: bool) -> None:
        """Set section expanded state."""
        section = self._sections.get(section_name)
        if section is not None:
            section.expanded = expanded

    def expand_all(self) -> None:
        """Expand all sections."""
        for section in self._sections.values():
            section.expand()

    def collapse_all(self) -> None:
        """Collapse all sections."""
        for section in self._sections.values():
            section.collapse()

    def is_dirty(self) -> bool:
        """Check if any field has been modified."""
        return len(self._dirty_fields) > 0

    def get_dirty_fields(self) -> Set[str]:
        """Get names of modified fields."""
        return set(self._dirty_fields)

    def clear_dirty(self) -> None:
        """Clear dirty state."""
        self._dirty_fields.clear()

    def render(self, ctx: DebugUIContext) -> None:
        """Render the panel."""
        if self._property_panel is not None:
            self._property_panel.render(ctx)


# =============================================================================
# Debuggable Inspector (extends AutoInspector)
# =============================================================================


@editor(category="DebugUI")
@reloadable()
class DebuggableInspector:
    """Inspector that handles @debuggable classes with full metadata."""
    __slots__ = ("_registry", "_base_inspector", "_panels")

    def __init__(self, widget_registry: Optional[WidgetRegistry] = None):
        self._registry = DebuggableRegistry.get_instance()
        self._base_inspector = AutoInspector(widget_registry)
        self._panels: Dict[int, DebuggablePanel] = {}

    @property
    def base_inspector(self) -> AutoInspector:
        """Get underlying AutoInspector."""
        return self._base_inspector

    @property
    def widget_registry(self) -> WidgetRegistry:
        """Get widget registry."""
        return self._base_inspector.registry

    def inspect(self, obj: Any, title: Optional[str] = None) -> Union[DebuggablePanel, PropertyPanel]:
        """
        Inspect an object.

        Returns DebuggablePanel for @debuggable classes, PropertyPanel for others.
        """
        cls = type(obj)

        # Check if debuggable
        if self._registry.is_debuggable(cls):
            return self._inspect_debuggable(obj, title)
        else:
            return self._base_inspector.inspect_object(obj, title)

    def _inspect_debuggable(self, obj: Any, title: Optional[str] = None) -> DebuggablePanel:
        """Inspect a @debuggable object."""
        obj_id = id(obj)

        # Return cached panel if exists
        if obj_id in self._panels:
            panel = self._panels[obj_id]
            panel.sync_from_target()
            return panel

        cls = type(obj)
        config = self._registry.get_config(cls)
        field_configs = self._registry.get_field_configs(cls)

        if config is None:
            config = DebugConfig()

        if title:
            config = DebugConfig(
                title=title,
                expanded=config.expanded,
                auto_sync=config.auto_sync,
                exclude_private=config.exclude_private,
                exclude_dunder=config.exclude_dunder,
                exclude_fields=config.exclude_fields,
                include_fields=config.include_fields,
                sections=config.sections,
                default_section=config.default_section,
                icon=config.icon,
                category=config.category,
            )

        panel = DebuggablePanel(obj, config, field_configs)
        self._panels[obj_id] = panel
        return panel

    def clear_cache(self) -> None:
        """Clear cached panels."""
        self._panels.clear()
        self._base_inspector.clear_cache()

    def remove_cached(self, obj: Any) -> bool:
        """Remove specific object from cache."""
        obj_id = id(obj)
        if obj_id in self._panels:
            del self._panels[obj_id]
            return True
        return self._base_inspector.remove_cached(obj)


# =============================================================================
# Debuggable Decorator
# =============================================================================


def debuggable(
    title: Optional[str] = None,
    expanded: bool = True,
    auto_sync: bool = True,
    exclude_private: bool = True,
    exclude_dunder: bool = True,
    exclude_fields: Optional[Set[str]] = None,
    include_fields: Optional[Set[str]] = None,
    sections: Optional[List[DebugSectionConfig]] = None,
    default_section: Optional[str] = None,
    icon: str = "",
    category: str = "General",
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator marking a class for auto-generated debug UI.

    Example:
        @debuggable(title="Player Stats", category="Game")
        @dataclass
        class Player:
            name: str = ""
            health: int = field(
                default=100,
                metadata=debug_field(min_value=0, max_value=100)
            )

    Args:
        title: Panel title (defaults to class name)
        expanded: Initial expanded state
        auto_sync: Auto-sync widget values with target
        exclude_private: Exclude _private fields
        exclude_dunder: Exclude __dunder__ fields
        exclude_fields: Set of field names to exclude
        include_fields: If set, only include these fields
        sections: List of section configurations
        default_section: Section name for unsectioned fields
        icon: Panel icon
        category: Category for grouping in UI

    Returns:
        Decorated class with debug UI metadata
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Build config
        config = DebugConfig(
            title=title,
            expanded=expanded,
            auto_sync=auto_sync,
            exclude_private=exclude_private,
            exclude_dunder=exclude_dunder,
            exclude_fields=exclude_fields or set(),
            include_fields=include_fields,
            sections=sections or [],
            default_section=default_section,
            icon=icon,
            category=category,
        )

        # Extract field configs from dataclass fields or annotations
        field_configs: Dict[str, DebugFieldConfig] = {}

        if is_dataclass(cls):
            for f in fields(cls):
                if f.metadata and "debug_field" in f.metadata:
                    field_configs[f.name] = f.metadata["debug_field"]

        # Also check class-level __debug_fields__ if defined
        if hasattr(cls, "__debug_fields__"):
            for name, cfg in cls.__debug_fields__.items():
                if isinstance(cfg, DebugFieldConfig):
                    field_configs[name] = cfg
                elif isinstance(cfg, dict):
                    field_configs[name] = DebugFieldConfig(**cfg)

        # Register with registry
        registry = DebuggableRegistry.get_instance()
        registry.register(cls, config, field_configs)

        # Add markers to class
        cls._debuggable = True
        cls._debug_config = config

        return cls

    return decorator


# =============================================================================
# Helper Functions
# =============================================================================


def is_debuggable(cls_or_obj: Union[type, Any]) -> bool:
    """Check if class or instance is decorated with @debuggable."""
    registry = DebuggableRegistry.get_instance()
    return registry.is_debuggable(cls_or_obj)


def get_debug_panel(obj: Any, title: Optional[str] = None) -> Optional[DebuggablePanel]:
    """
    Get debug panel for a @debuggable object.

    Returns None if object is not @debuggable.
    """
    if not is_debuggable(obj):
        return None

    inspector = DebuggableInspector()
    return inspector.inspect(obj, title)


def create_debug_ui_for(
    obj: Any,
    debug_ui: DebugUI,
    panel_id: Optional[str] = None,
) -> Optional[Union[DebuggablePanel, PropertyPanel]]:
    """
    Create and register debug UI for an object.

    Args:
        obj: Object to inspect
        debug_ui: DebugUI manager instance
        panel_id: Optional panel ID for registration

    Returns:
        Created panel or None
    """
    inspector = DebuggableInspector(debug_ui.registry)
    panel = inspector.inspect(obj)

    if panel_id and isinstance(panel, DebuggablePanel):
        debug_ui._panels[panel_id] = panel.property_panel
        if panel.property_panel:
            debug_ui._root.add_child(panel.property_panel)

    return panel


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Core decorator
    "debuggable",
    "debug_field",
    "debug_section",
    # Configuration classes
    "DebugConfig",
    "DebugFieldConfig",
    "DebugSectionConfig",
    "WidgetHint",
    # Panel and inspector
    "DebuggablePanel",
    "DebuggableInspector",
    "DebuggableRegistry",
    # Helper functions
    "is_debuggable",
    "get_debug_panel",
    "create_debug_ui_for",
]
