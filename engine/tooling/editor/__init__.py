"""
Editor Framework - Main editor application infrastructure.

This module provides the core editor framework including:
- Application shell with docking, tabs, and panels
- 3D/2D viewport with camera controls and render modes
- Multi-selection system with marquee and grouping
- Transform gizmos (translate, rotate, scale, universal)
- Editor modes (Select, Paint, Sculpt, Placement, Sequence)
- Command pattern for undoable actions
- Keyboard shortcut manager with customization
- User preferences system with categories
- Plugin manager with hot-loading capability
- Debug UI framework (imgui/egui-style)

All editor classes use the @editor decorator and integrate with
Foundation's Tracker for undo/redo and Mirror for property inspection.
"""

from engine.tooling.editor.app_shell import (
    EditorApplication,
    DockingManager,
    Panel,
    PanelPosition,
    Tab,
    TabGroup,
    MenuBar,
    MenuItem,
    ToolBar,
    StatusBar,
)

from engine.tooling.editor.viewport import (
    Viewport,
    Camera,
    CameraMode,
    RenderMode,
    ViewportOverlay,
    GridSettings,
    ViewportInput,
)

from engine.tooling.editor.selection import (
    SelectionManager,
    Selection,
    SelectionSet,
    SelectionFilter,
    MarqueeSelection,
    SelectionGroup,
    PickingResult,
)

from engine.tooling.editor.gizmos import (
    GizmoManager,
    Gizmo,
    GizmoType,
    GizmoSpace,
    TranslateGizmo,
    RotateGizmo,
    ScaleGizmo,
    UniversalGizmo,
    GizmoConstraint,
)

from engine.tooling.editor.modes import (
    EditorMode,
    ModeManager,
    SelectMode,
    PaintMode,
    SculptMode,
    PlacementMode,
    SequenceMode,
    ModeContext,
    ModeTool,
)

from engine.tooling.editor.commands import (
    Command,
    CommandManager,
    CommandBatch,
    TransformCommand,
    CreateCommand,
    DeleteCommand,
    PropertyCommand,
    ReparentCommand,
    CompositeCommand,
)

from engine.tooling.editor.shortcuts import (
    ShortcutManager,
    Shortcut,
    ShortcutContext,
    KeyBinding,
    KeyModifiers,
    ShortcutConflict,
)

from engine.tooling.editor.preferences import (
    PreferencesManager,
    PreferenceCategory,
    Preference,
    PreferenceType,
    PreferenceValidator,
    PreferencesPage,
)

from engine.tooling.editor.plugins import (
    PluginManager,
    Plugin,
    PluginState,
    PluginDependency,
    PluginExtensionPoint,
    PluginManifest,
)

from engine.tooling.editor.script_reload import (
    ScriptReloader,
    ScriptState,
    ModuleSwapper,
    StateSerializer,
    ReloadState,
    ReloadStrategy,
    ReloadResult,
    ReloadError,
    ReloadErrorType,
    ExecutionCheckpoint,
    ModuleBackup,
    reloadable_section,
    get_script_reloader,
)

from engine.tooling.editor.debug_ui import (
    # Core types
    WidgetType,
    UIState,
    Vec2,
    Vec3,
    Vec4,
    Color,
    WidgetStyle,
    WidgetConfig,
    UIEvent,
    # Base widgets
    Widget,
    ContainerWidget,
    # Concrete widgets
    LabelWidget,
    TextInputWidget,
    IntSliderWidget,
    FloatSliderWidget,
    IntInputWidget,
    FloatInputWidget,
    CheckboxWidget,
    DropdownWidget,
    ColorPickerWidget,
    Vec2InputWidget,
    Vec3InputWidget,
    Vec4InputWidget,
    ButtonWidget,
    SeparatorWidget,
    # Containers
    CollapsibleSection,
    PropertyPanel,
    PropertyBinding,
    # Registry and inspection
    WidgetRegistry,
    AutoInspector,
    # Context and manager
    DebugUIContext,
    DebugUI,
)

from engine.tooling.editor.debuggable import (
    # Core decorator
    debuggable,
    debug_field,
    debug_section,
    # Configuration classes
    DebugConfig,
    DebugFieldConfig,
    DebugSectionConfig,
    WidgetHint,
    # Panel and inspector
    DebuggablePanel,
    DebuggableInspector,
    DebuggableRegistry,
    # Helper functions
    is_debuggable,
    get_debug_panel,
    create_debug_ui_for,
)

__all__ = [
    # App Shell
    "EditorApplication",
    "DockingManager",
    "Panel",
    "PanelPosition",
    "Tab",
    "TabGroup",
    "MenuBar",
    "MenuItem",
    "ToolBar",
    "StatusBar",
    # Viewport
    "Viewport",
    "Camera",
    "CameraMode",
    "RenderMode",
    "ViewportOverlay",
    "GridSettings",
    "ViewportInput",
    # Selection
    "SelectionManager",
    "Selection",
    "SelectionSet",
    "SelectionFilter",
    "MarqueeSelection",
    "SelectionGroup",
    "PickingResult",
    # Gizmos
    "GizmoManager",
    "Gizmo",
    "GizmoType",
    "GizmoSpace",
    "TranslateGizmo",
    "RotateGizmo",
    "ScaleGizmo",
    "UniversalGizmo",
    "GizmoConstraint",
    # Modes
    "EditorMode",
    "ModeManager",
    "SelectMode",
    "PaintMode",
    "SculptMode",
    "PlacementMode",
    "SequenceMode",
    "ModeContext",
    "ModeTool",
    # Commands
    "Command",
    "CommandManager",
    "CommandBatch",
    "TransformCommand",
    "CreateCommand",
    "DeleteCommand",
    "PropertyCommand",
    "ReparentCommand",
    "CompositeCommand",
    # Shortcuts
    "ShortcutManager",
    "Shortcut",
    "ShortcutContext",
    "KeyBinding",
    "KeyModifiers",
    "ShortcutConflict",
    # Preferences
    "PreferencesManager",
    "PreferenceCategory",
    "Preference",
    "PreferenceType",
    "PreferenceValidator",
    "PreferencesPage",
    # Plugins
    "PluginManager",
    "Plugin",
    "PluginState",
    "PluginDependency",
    "PluginExtensionPoint",
    "PluginManifest",
    # Script Reload
    "ScriptReloader",
    "ScriptState",
    "ModuleSwapper",
    "StateSerializer",
    "ReloadState",
    "ReloadStrategy",
    "ReloadResult",
    "ReloadError",
    "ReloadErrorType",
    "ExecutionCheckpoint",
    "ModuleBackup",
    "reloadable_section",
    "get_script_reloader",
    # Debug UI - Core types
    "WidgetType",
    "UIState",
    "Vec2",
    "Vec3",
    "Vec4",
    "Color",
    "WidgetStyle",
    "WidgetConfig",
    "UIEvent",
    # Debug UI - Base widgets
    "Widget",
    "ContainerWidget",
    # Debug UI - Concrete widgets
    "LabelWidget",
    "TextInputWidget",
    "IntSliderWidget",
    "FloatSliderWidget",
    "IntInputWidget",
    "FloatInputWidget",
    "CheckboxWidget",
    "DropdownWidget",
    "ColorPickerWidget",
    "Vec2InputWidget",
    "Vec3InputWidget",
    "Vec4InputWidget",
    "ButtonWidget",
    "SeparatorWidget",
    # Debug UI - Containers
    "CollapsibleSection",
    "PropertyPanel",
    "PropertyBinding",
    # Debug UI - Registry and inspection
    "WidgetRegistry",
    "AutoInspector",
    # Debug UI - Context and manager
    "DebugUIContext",
    "DebugUI",
    # Debuggable decorator and inspector
    "debuggable",
    "debug_field",
    "debug_section",
    "DebugConfig",
    "DebugFieldConfig",
    "DebugSectionConfig",
    "WidgetHint",
    "DebuggablePanel",
    "DebuggableInspector",
    "DebuggableRegistry",
    "is_debuggable",
    "get_debug_panel",
    "create_debug_ui_for",
]
