"""
Shortcuts - Keyboard shortcut manager with customization.

Provides:
- Shortcut registration and lookup
- Context-based shortcuts (global, viewport, panel-specific)
- Key binding with modifiers
- Conflict detection and resolution
- Shortcut customization and persistence
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any, Callable, Optional

from engine.tooling.editor.app_shell import editor, reloadable


class KeyModifiers(Flag):
    """Modifier keys for shortcuts."""
    NONE = 0
    CTRL = auto()
    SHIFT = auto()
    ALT = auto()
    META = auto()  # Cmd on Mac, Win on Windows

    @classmethod
    def from_string(cls, s: str) -> "KeyModifiers":
        """Parse modifiers from string like 'Ctrl+Shift'."""
        mods = cls.NONE
        s = s.lower()
        if "ctrl" in s:
            mods |= cls.CTRL
        if "shift" in s:
            mods |= cls.SHIFT
        if "alt" in s:
            mods |= cls.ALT
        if "meta" in s or "cmd" in s or "win" in s:
            mods |= cls.META
        return mods

    def to_string(self) -> str:
        """Convert modifiers to display string."""
        parts = []
        if KeyModifiers.CTRL in self:
            parts.append("Ctrl")
        if KeyModifiers.SHIFT in self:
            parts.append("Shift")
        if KeyModifiers.ALT in self:
            parts.append("Alt")
        if KeyModifiers.META in self:
            parts.append("Meta")
        return "+".join(parts) if parts else ""


@editor(category="Shortcuts")
@reloadable()
class KeyBinding:
    """A key binding consisting of a key and modifiers."""
    __slots__ = ("key", "modifiers")

    def __init__(self, key: str, modifiers: KeyModifiers = KeyModifiers.NONE):
        self.key = key.upper() if len(key) == 1 else key
        self.modifiers = modifiers

    @classmethod
    def from_string(cls, binding_str: str) -> "KeyBinding":
        """
        Parse a binding string like 'Ctrl+Shift+S' or 'F5'.

        Supports:
        - Single keys: 'A', 'F1', 'Space', 'Enter', 'Escape'
        - Modifiers: 'Ctrl+A', 'Shift+F1', 'Ctrl+Shift+S'
        """
        parts = [p.strip() for p in binding_str.split("+")]
        key = parts[-1]

        modifier_parts = parts[:-1]
        mods = KeyModifiers.NONE
        for part in modifier_parts:
            part_lower = part.lower()
            if part_lower == "ctrl":
                mods |= KeyModifiers.CTRL
            elif part_lower == "shift":
                mods |= KeyModifiers.SHIFT
            elif part_lower == "alt":
                mods |= KeyModifiers.ALT
            elif part_lower in ("meta", "cmd", "win"):
                mods |= KeyModifiers.META

        return cls(key, mods)

    def to_string(self) -> str:
        """Convert to display string."""
        parts = []
        if KeyModifiers.CTRL in self.modifiers:
            parts.append("Ctrl")
        if KeyModifiers.SHIFT in self.modifiers:
            parts.append("Shift")
        if KeyModifiers.ALT in self.modifiers:
            parts.append("Alt")
        if KeyModifiers.META in self.modifiers:
            parts.append("Meta")
        parts.append(self.key)
        return "+".join(parts)

    def matches(self, key: str, modifiers: KeyModifiers) -> bool:
        """Check if this binding matches the given key and modifiers."""
        key_match = self.key.upper() == key.upper()
        mod_match = self.modifiers == modifiers
        return key_match and mod_match

    def __hash__(self) -> int:
        return hash((self.key.upper(), self.modifiers))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, KeyBinding):
            return self.key.upper() == other.key.upper() and self.modifiers == other.modifiers
        return False


@editor(category="Shortcuts")
@reloadable()
class ShortcutContext:
    """Context for shortcut activation."""
    __slots__ = ("id", "name", "parent_id", "priority")

    def __init__(self, id: str, name: str = "", parent_id: Optional[str] = None,
                 priority: int = 0):
        self.id = id
        self.name = name or id
        self.parent_id = parent_id
        self.priority = priority


# Predefined contexts
CONTEXT_GLOBAL = ShortcutContext("global", "Global", priority=0)
CONTEXT_VIEWPORT = ShortcutContext("viewport", "Viewport", "global", priority=10)
CONTEXT_HIERARCHY = ShortcutContext("hierarchy", "Hierarchy", "global", priority=10)
CONTEXT_INSPECTOR = ShortcutContext("inspector", "Inspector", "global", priority=10)
CONTEXT_CONTENT_BROWSER = ShortcutContext("content_browser", "Content Browser", "global", priority=10)
CONTEXT_CONSOLE = ShortcutContext("console", "Console", "global", priority=20)
CONTEXT_TEXT_EDIT = ShortcutContext("text_edit", "Text Editing", "global", priority=30)


@editor(category="Shortcuts")
@reloadable()
class Shortcut:
    """A shortcut binding an action to a key combination."""
    __slots__ = ("id", "name", "description", "binding", "action",
                 "context", "enabled", "category", "is_default",
                 "allows_repeat")

    def __init__(self, id: str, name: str, binding: Optional[KeyBinding] = None,
                 action: Optional[Callable] = None, description: str = "",
                 context: Optional[ShortcutContext] = None, enabled: bool = True,
                 category: str = "General", allows_repeat: bool = False):
        self.id = id
        self.name = name
        self.description = description
        self.binding = binding
        self.action = action
        self.context = context or CONTEXT_GLOBAL
        self.enabled = enabled
        self.category = category
        self.is_default = True
        self.allows_repeat = allows_repeat

    def execute(self) -> bool:
        """Execute the shortcut action. Returns True if executed."""
        if self.enabled and self.action:
            self.action()
            return True
        return False

    def set_binding(self, binding: Optional[KeyBinding]) -> None:
        """Set the key binding."""
        self.binding = binding
        self.is_default = False

    def reset_to_default(self, default_binding: Optional[KeyBinding]) -> None:
        """Reset to default binding."""
        self.binding = default_binding
        self.is_default = True


@editor(category="Shortcuts")
@reloadable()
class ShortcutConflict:
    """Represents a conflict between shortcuts."""
    __slots__ = ("binding", "shortcuts", "context")

    def __init__(self, binding: KeyBinding, shortcuts: list[Shortcut],
                 context: ShortcutContext):
        self.binding = binding
        self.shortcuts = shortcuts
        self.context = context


@editor(category="Shortcuts")
@reloadable(preserve=["_shortcuts", "_contexts"])
class ShortcutManager:
    """Manages keyboard shortcuts."""
    __slots__ = ("_shortcuts", "_contexts", "_active_contexts",
                 "_defaults", "on_shortcut_triggered", "on_conflict")

    def __init__(self):
        self._shortcuts: dict[str, Shortcut] = {}
        self._contexts: dict[str, ShortcutContext] = {}
        self._active_contexts: set[str] = {"global"}
        self._defaults: dict[str, KeyBinding] = {}
        self.on_shortcut_triggered: Optional[Callable[[Shortcut], None]] = None
        self.on_conflict: Optional[Callable[[ShortcutConflict], None]] = None

        # Register predefined contexts
        for ctx in [CONTEXT_GLOBAL, CONTEXT_VIEWPORT, CONTEXT_HIERARCHY,
                    CONTEXT_INSPECTOR, CONTEXT_CONTENT_BROWSER,
                    CONTEXT_CONSOLE, CONTEXT_TEXT_EDIT]:
            self.register_context(ctx)

    @property
    def shortcuts(self) -> list[Shortcut]:
        """Get all shortcuts."""
        return list(self._shortcuts.values())

    @property
    def contexts(self) -> list[ShortcutContext]:
        """Get all contexts."""
        return list(self._contexts.values())

    def register_context(self, context: ShortcutContext) -> None:
        """Register a shortcut context."""
        self._contexts[context.id] = context

    def unregister_context(self, context_id: str) -> bool:
        """Unregister a context."""
        if context_id != "global":  # Can't remove global
            return self._contexts.pop(context_id, None) is not None
        return False

    def set_active_context(self, context_id: str) -> None:
        """Set the currently active context."""
        self._active_contexts = {"global"}
        if context_id in self._contexts:
            self._active_contexts.add(context_id)
            # Also add parent contexts
            ctx = self._contexts[context_id]
            while ctx.parent_id:
                self._active_contexts.add(ctx.parent_id)
                ctx = self._contexts.get(ctx.parent_id)
                if not ctx:
                    break

    def add_active_context(self, context_id: str) -> None:
        """Add a context to active contexts."""
        if context_id in self._contexts:
            self._active_contexts.add(context_id)

    def remove_active_context(self, context_id: str) -> None:
        """Remove a context from active contexts."""
        if context_id != "global":
            self._active_contexts.discard(context_id)

    @property
    def active_contexts(self) -> set[str]:
        """Get active context IDs."""
        return set(self._active_contexts)

    def register(self, shortcut: Shortcut) -> None:
        """Register a shortcut."""
        self._shortcuts[shortcut.id] = shortcut
        if shortcut.binding:
            self._defaults[shortcut.id] = shortcut.binding

    def unregister(self, shortcut_id: str) -> Optional[Shortcut]:
        """Unregister a shortcut."""
        self._defaults.pop(shortcut_id, None)
        return self._shortcuts.pop(shortcut_id, None)

    def get(self, shortcut_id: str) -> Optional[Shortcut]:
        """Get a shortcut by ID."""
        return self._shortcuts.get(shortcut_id)

    def set_binding(self, shortcut_id: str, binding: Optional[KeyBinding]) -> bool:
        """Set binding for a shortcut. Returns True if successful."""
        shortcut = self._shortcuts.get(shortcut_id)
        if shortcut:
            shortcut.set_binding(binding)
            return True
        return False

    def reset_binding(self, shortcut_id: str) -> bool:
        """Reset a shortcut to its default binding."""
        shortcut = self._shortcuts.get(shortcut_id)
        if shortcut:
            default = self._defaults.get(shortcut_id)
            shortcut.reset_to_default(default)
            return True
        return False

    def reset_all_bindings(self) -> None:
        """Reset all shortcuts to defaults."""
        for sid in self._shortcuts:
            self.reset_binding(sid)

    def find_conflicts(self) -> list[ShortcutConflict]:
        """Find all shortcut conflicts."""
        conflicts = []

        # Group shortcuts by binding and context
        binding_map: dict[tuple[KeyBinding, str], list[Shortcut]] = {}

        for shortcut in self._shortcuts.values():
            if shortcut.binding and shortcut.enabled:
                key = (shortcut.binding, shortcut.context.id)
                if key not in binding_map:
                    binding_map[key] = []
                binding_map[key].append(shortcut)

        # Find conflicts (multiple shortcuts with same binding in same context)
        for (binding, ctx_id), shortcuts in binding_map.items():
            if len(shortcuts) > 1:
                ctx = self._contexts.get(ctx_id, CONTEXT_GLOBAL)
                conflicts.append(ShortcutConflict(binding, shortcuts, ctx))

        return conflicts

    def find_by_binding(self, binding: KeyBinding,
                        context_id: Optional[str] = None) -> list[Shortcut]:
        """Find shortcuts by key binding."""
        results = []
        for shortcut in self._shortcuts.values():
            if shortcut.binding and shortcut.binding == binding:
                if context_id is None or shortcut.context.id == context_id:
                    results.append(shortcut)
        return results

    def on_key_down(self, key: str, modifiers: KeyModifiers) -> bool:
        """
        Handle key down event. Returns True if a shortcut was triggered.

        Finds the highest priority matching shortcut in active contexts.
        """
        matching = []

        for shortcut in self._shortcuts.values():
            if not shortcut.enabled or not shortcut.binding:
                continue

            # Check if binding matches
            if not shortcut.binding.matches(key, modifiers):
                continue

            # Check if context is active
            if shortcut.context.id not in self._active_contexts:
                continue

            matching.append(shortcut)

        if not matching:
            return False

        # Check for conflicts
        if len(matching) > 1:
            # Sort by context priority (higher first)
            matching.sort(key=lambda s: s.context.priority, reverse=True)

            # If top shortcuts have same priority, it's a conflict
            top_priority = matching[0].context.priority
            same_priority = [s for s in matching if s.context.priority == top_priority]

            if len(same_priority) > 1 and self.on_conflict:
                binding = same_priority[0].binding
                if binding:
                    conflict = ShortcutConflict(binding, same_priority,
                                                same_priority[0].context)
                    self.on_conflict(conflict)

        # Execute highest priority shortcut
        shortcut = matching[0]
        if shortcut.execute():
            if self.on_shortcut_triggered:
                self.on_shortcut_triggered(shortcut)
            return True

        return False

    def get_shortcuts_for_context(self, context_id: str) -> list[Shortcut]:
        """Get all shortcuts for a specific context."""
        return [s for s in self._shortcuts.values()
                if s.context.id == context_id]

    def get_shortcuts_by_category(self) -> dict[str, list[Shortcut]]:
        """Get shortcuts grouped by category."""
        categories: dict[str, list[Shortcut]] = {}
        for shortcut in self._shortcuts.values():
            if shortcut.category not in categories:
                categories[shortcut.category] = []
            categories[shortcut.category].append(shortcut)
        return categories

    def save_customizations(self) -> dict:
        """Save custom bindings for persistence."""
        customizations = {}
        for sid, shortcut in self._shortcuts.items():
            if not shortcut.is_default:
                binding_str = shortcut.binding.to_string() if shortcut.binding else None
                customizations[sid] = binding_str
        return customizations

    def load_customizations(self, customizations: dict) -> None:
        """Load custom bindings."""
        for sid, binding_str in customizations.items():
            shortcut = self._shortcuts.get(sid)
            if shortcut:
                if binding_str:
                    binding = KeyBinding.from_string(binding_str)
                    shortcut.set_binding(binding)
                else:
                    shortcut.set_binding(None)

    def register_action(self, id: str, name: str, binding_str: str,
                        action: Callable[[], None], category: str = "General",
                        description: str = "",
                        context: Optional[ShortcutContext] = None) -> Shortcut:
        """Convenience method to register an action with shortcut."""
        binding = KeyBinding.from_string(binding_str) if binding_str else None
        shortcut = Shortcut(
            id=id,
            name=name,
            binding=binding,
            action=action,
            category=category,
            description=description,
            context=context or CONTEXT_GLOBAL
        )
        self.register(shortcut)
        return shortcut

    def get_display_string(self, shortcut_id: str) -> str:
        """Get display string for a shortcut's binding."""
        shortcut = self._shortcuts.get(shortcut_id)
        if shortcut and shortcut.binding:
            return shortcut.binding.to_string()
        return ""
