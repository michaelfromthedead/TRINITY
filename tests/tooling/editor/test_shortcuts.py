"""
Comprehensive tests for the Shortcut system.

Tests cover:
- Key bindings with modifiers
- Shortcut registration and lookup
- Context-based shortcuts
- Conflict detection
- Shortcut customization
- Shortcut persistence
"""
import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.editor.shortcuts import (
    ShortcutManager,
    Shortcut,
    ShortcutContext,
    KeyBinding,
    KeyModifiers,
    ShortcutConflict,
    CONTEXT_GLOBAL,
    CONTEXT_VIEWPORT,
)


class TestKeyModifiers:
    """Tests for KeyModifiers flags."""

    def test_modifiers_none(self):
        """NONE modifier is empty."""
        assert KeyModifiers.NONE.value == 0

    def test_modifiers_combinations(self):
        """Modifiers can be combined."""
        combo = KeyModifiers.CTRL | KeyModifiers.SHIFT
        assert KeyModifiers.CTRL in combo
        assert KeyModifiers.SHIFT in combo
        assert KeyModifiers.ALT not in combo

    def test_modifiers_from_string(self):
        """Modifiers can be parsed from string."""
        mods = KeyModifiers.from_string("Ctrl+Shift")
        assert KeyModifiers.CTRL in mods
        assert KeyModifiers.SHIFT in mods
        assert KeyModifiers.ALT not in mods

    def test_modifiers_from_string_case_insensitive(self):
        """Modifier parsing is case insensitive."""
        mods = KeyModifiers.from_string("CTRL+shift+Alt")
        assert KeyModifiers.CTRL in mods
        assert KeyModifiers.SHIFT in mods
        assert KeyModifiers.ALT in mods

    def test_modifiers_to_string(self):
        """Modifiers can be converted to string."""
        mods = KeyModifiers.CTRL | KeyModifiers.SHIFT
        s = mods.to_string()
        assert "Ctrl" in s
        assert "Shift" in s


class TestKeyBinding:
    """Tests for KeyBinding class."""

    def test_binding_creation(self):
        """KeyBinding should be created properly."""
        binding = KeyBinding("S", KeyModifiers.CTRL)
        assert binding.key == "S"
        assert KeyModifiers.CTRL in binding.modifiers

    def test_binding_key_uppercase(self):
        """Single char keys are uppercased."""
        binding = KeyBinding("a")
        assert binding.key == "A"

    def test_binding_from_string_simple(self):
        """KeyBinding can be parsed from simple string."""
        binding = KeyBinding.from_string("F5")
        assert binding.key == "F5"
        assert binding.modifiers == KeyModifiers.NONE

    def test_binding_from_string_with_modifiers(self):
        """KeyBinding can be parsed with modifiers."""
        binding = KeyBinding.from_string("Ctrl+Shift+S")
        assert binding.key == "S"
        assert KeyModifiers.CTRL in binding.modifiers
        assert KeyModifiers.SHIFT in binding.modifiers

    def test_binding_to_string(self):
        """KeyBinding can be converted to string."""
        binding = KeyBinding("S", KeyModifiers.CTRL | KeyModifiers.SHIFT)
        s = binding.to_string()
        assert "Ctrl" in s
        assert "Shift" in s
        assert "S" in s

    def test_binding_matches(self):
        """KeyBinding can check for matches."""
        binding = KeyBinding("S", KeyModifiers.CTRL)

        assert binding.matches("S", KeyModifiers.CTRL) is True
        assert binding.matches("s", KeyModifiers.CTRL) is True  # Case insensitive
        assert binding.matches("S", KeyModifiers.NONE) is False
        assert binding.matches("A", KeyModifiers.CTRL) is False

    def test_binding_equality(self):
        """KeyBindings can be compared."""
        b1 = KeyBinding("S", KeyModifiers.CTRL)
        b2 = KeyBinding("s", KeyModifiers.CTRL)  # Different case
        b3 = KeyBinding("S", KeyModifiers.SHIFT)

        assert b1 == b2
        assert b1 != b3

    def test_binding_hash(self):
        """KeyBindings can be hashed."""
        b1 = KeyBinding("S", KeyModifiers.CTRL)
        b2 = KeyBinding("s", KeyModifiers.CTRL)

        # Same hash for equal bindings
        assert hash(b1) == hash(b2)

        # Can be used in sets
        s = {b1}
        assert b2 in s


class TestShortcutContext:
    """Tests for ShortcutContext class."""

    def test_context_creation(self):
        """ShortcutContext should be created properly."""
        ctx = ShortcutContext("my_context", "My Context", priority=10)
        assert ctx.id == "my_context"
        assert ctx.name == "My Context"
        assert ctx.priority == 10

    def test_context_with_parent(self):
        """ShortcutContext can have parent."""
        ctx = ShortcutContext("child", "Child", parent_id="parent")
        assert ctx.parent_id == "parent"


class TestShortcut:
    """Tests for Shortcut class."""

    def test_shortcut_creation(self):
        """Shortcut should be created properly."""
        binding = KeyBinding.from_string("Ctrl+S")
        shortcut = Shortcut("save", "Save", binding)

        assert shortcut.id == "save"
        assert shortcut.name == "Save"
        assert shortcut.binding == binding
        assert shortcut.enabled is True

    def test_shortcut_execute(self):
        """Shortcut executes action."""
        executed = []
        binding = KeyBinding.from_string("Ctrl+S")
        shortcut = Shortcut("save", "Save", binding, action=lambda: executed.append(True))

        assert shortcut.execute() is True
        assert len(executed) == 1

    def test_shortcut_execute_disabled(self):
        """Disabled shortcut doesn't execute."""
        executed = []
        shortcut = Shortcut("test", "Test", action=lambda: executed.append(True), enabled=False)

        assert shortcut.execute() is False
        assert len(executed) == 0

    def test_shortcut_set_binding(self):
        """Shortcut binding can be changed."""
        shortcut = Shortcut("test", "Test")
        new_binding = KeyBinding.from_string("Ctrl+T")

        shortcut.set_binding(new_binding)
        assert shortcut.binding == new_binding
        assert shortcut.is_default is False

    def test_shortcut_reset_to_default(self):
        """Shortcut can be reset to default."""
        default = KeyBinding.from_string("Ctrl+S")
        shortcut = Shortcut("save", "Save", default)
        shortcut.set_binding(KeyBinding.from_string("Ctrl+Shift+S"))

        shortcut.reset_to_default(default)
        assert shortcut.binding == default
        assert shortcut.is_default is True


class TestShortcutConflict:
    """Tests for ShortcutConflict class."""

    def test_conflict_creation(self):
        """ShortcutConflict should be created properly."""
        binding = KeyBinding.from_string("Ctrl+S")
        s1 = Shortcut("save", "Save", binding)
        s2 = Shortcut("search", "Search", binding)

        conflict = ShortcutConflict(binding, [s1, s2], CONTEXT_GLOBAL)

        assert conflict.binding == binding
        assert len(conflict.shortcuts) == 2


class TestShortcutManager:
    """Tests for ShortcutManager class."""

    def test_manager_creation(self):
        """ShortcutManager should be created with default contexts."""
        manager = ShortcutManager()

        # Should have predefined contexts
        assert len(manager.contexts) >= 2  # global + others

    def test_manager_register_shortcut(self):
        """ShortcutManager can register shortcuts."""
        manager = ShortcutManager()
        binding = KeyBinding.from_string("Ctrl+S")
        shortcut = Shortcut("save", "Save", binding)

        manager.register(shortcut)
        assert manager.get("save") == shortcut

    def test_manager_unregister_shortcut(self):
        """ShortcutManager can unregister shortcuts."""
        manager = ShortcutManager()
        shortcut = Shortcut("test", "Test")
        manager.register(shortcut)

        removed = manager.unregister("test")
        assert removed == shortcut
        assert manager.get("test") is None

    def test_manager_set_binding(self):
        """ShortcutManager can set binding for shortcut."""
        manager = ShortcutManager()
        shortcut = Shortcut("test", "Test")
        manager.register(shortcut)

        binding = KeyBinding.from_string("Ctrl+T")
        assert manager.set_binding("test", binding) is True
        assert shortcut.binding == binding

    def test_manager_reset_binding(self):
        """ShortcutManager can reset binding to default."""
        manager = ShortcutManager()
        default = KeyBinding.from_string("Ctrl+S")
        shortcut = Shortcut("save", "Save", default)
        manager.register(shortcut)

        shortcut.set_binding(KeyBinding.from_string("Ctrl+Shift+S"))
        manager.reset_binding("save")

        assert shortcut.binding == default

    def test_manager_reset_all(self):
        """ShortcutManager can reset all bindings."""
        manager = ShortcutManager()
        s1 = Shortcut("s1", "S1", KeyBinding.from_string("F1"))
        s2 = Shortcut("s2", "S2", KeyBinding.from_string("F2"))

        manager.register(s1)
        manager.register(s2)

        s1.set_binding(KeyBinding.from_string("F3"))
        s2.set_binding(KeyBinding.from_string("F4"))

        manager.reset_all_bindings()

        assert s1.binding.key == "F1"
        assert s2.binding.key == "F2"

    def test_manager_find_by_binding(self):
        """ShortcutManager can find shortcuts by binding."""
        manager = ShortcutManager()
        binding = KeyBinding.from_string("Ctrl+S")
        shortcut = Shortcut("save", "Save", binding)
        manager.register(shortcut)

        found = manager.find_by_binding(binding)
        assert shortcut in found

    def test_manager_find_conflicts(self):
        """ShortcutManager detects conflicts."""
        manager = ShortcutManager()
        binding = KeyBinding.from_string("Ctrl+S")
        s1 = Shortcut("save", "Save", binding, context=CONTEXT_GLOBAL)
        s2 = Shortcut("search", "Search", binding, context=CONTEXT_GLOBAL)

        manager.register(s1)
        manager.register(s2)

        conflicts = manager.find_conflicts()
        assert len(conflicts) >= 1

    def test_manager_no_conflict_different_context(self):
        """Different contexts don't conflict."""
        manager = ShortcutManager()
        binding = KeyBinding.from_string("Ctrl+S")

        ctx1 = ShortcutContext("ctx1", "Context 1")
        ctx2 = ShortcutContext("ctx2", "Context 2")
        manager.register_context(ctx1)
        manager.register_context(ctx2)

        s1 = Shortcut("s1", "S1", binding, context=ctx1)
        s2 = Shortcut("s2", "S2", binding, context=ctx2)

        manager.register(s1)
        manager.register(s2)

        conflicts = manager.find_conflicts()
        assert len(conflicts) == 0

    def test_manager_on_key_down(self):
        """ShortcutManager handles key down events."""
        manager = ShortcutManager()
        executed = []

        shortcut = Shortcut(
            "test", "Test",
            KeyBinding.from_string("Ctrl+S"),
            action=lambda: executed.append(True),
            context=CONTEXT_GLOBAL
        )
        manager.register(shortcut)

        result = manager.on_key_down("S", KeyModifiers.CTRL)
        assert result is True
        assert len(executed) == 1

    def test_manager_on_key_down_no_match(self):
        """Non-matching key returns False."""
        manager = ShortcutManager()
        result = manager.on_key_down("X", KeyModifiers.NONE)
        assert result is False

    def test_manager_context_activation(self):
        """Context affects shortcut triggering."""
        manager = ShortcutManager()
        executed_global = []
        executed_viewport = []

        s_global = Shortcut(
            "global", "Global",
            KeyBinding.from_string("F1"),
            action=lambda: executed_global.append(True),
            context=CONTEXT_GLOBAL
        )
        s_viewport = Shortcut(
            "viewport", "Viewport",
            KeyBinding.from_string("F2"),
            action=lambda: executed_viewport.append(True),
            context=CONTEXT_VIEWPORT
        )

        manager.register(s_global)
        manager.register(s_viewport)

        # Global context only
        manager.set_active_context("global")

        manager.on_key_down("F1", KeyModifiers.NONE)
        assert len(executed_global) == 1

        manager.on_key_down("F2", KeyModifiers.NONE)
        assert len(executed_viewport) == 0  # Not active

        # Activate viewport context
        manager.add_active_context("viewport")
        manager.on_key_down("F2", KeyModifiers.NONE)
        assert len(executed_viewport) == 1

    def test_manager_context_priority(self):
        """Higher priority context wins on conflict."""
        manager = ShortcutManager()
        low_exec = []
        high_exec = []

        ctx_low = ShortcutContext("low", "Low", priority=1)
        ctx_high = ShortcutContext("high", "High", priority=10)
        manager.register_context(ctx_low)
        manager.register_context(ctx_high)

        binding = KeyBinding.from_string("F1")
        s_low = Shortcut("low", "Low", binding, action=lambda: low_exec.append(True), context=ctx_low)
        s_high = Shortcut("high", "High", binding, action=lambda: high_exec.append(True), context=ctx_high)

        manager.register(s_low)
        manager.register(s_high)

        manager.add_active_context("low")
        manager.add_active_context("high")

        manager.on_key_down("F1", KeyModifiers.NONE)

        # High priority should win
        assert len(high_exec) == 1
        assert len(low_exec) == 0

    def test_manager_shortcut_triggered_callback(self):
        """ShortcutManager triggers callback on shortcut execution."""
        manager = ShortcutManager()
        triggered = []
        manager.on_shortcut_triggered = lambda s: triggered.append(s.id)

        shortcut = Shortcut(
            "test", "Test",
            KeyBinding.from_string("F1"),
            action=lambda: None
        )
        manager.register(shortcut)

        manager.on_key_down("F1", KeyModifiers.NONE)
        assert "test" in triggered

    def test_manager_get_shortcuts_by_category(self):
        """ShortcutManager can group by category."""
        manager = ShortcutManager()
        s1 = Shortcut("s1", "S1", category="File")
        s2 = Shortcut("s2", "S2", category="File")
        s3 = Shortcut("s3", "S3", category="Edit")

        manager.register(s1)
        manager.register(s2)
        manager.register(s3)

        categories = manager.get_shortcuts_by_category()
        assert len(categories["File"]) == 2
        assert len(categories["Edit"]) == 1

    def test_manager_save_load_customizations(self):
        """ShortcutManager can save and load customizations."""
        manager = ShortcutManager()
        default = KeyBinding.from_string("Ctrl+S")
        shortcut = Shortcut("save", "Save", default)
        manager.register(shortcut)

        # Customize
        shortcut.set_binding(KeyBinding.from_string("Ctrl+Shift+S"))

        # Save
        customizations = manager.save_customizations()
        assert "save" in customizations

        # Reset and reload
        shortcut.reset_to_default(default)
        manager.load_customizations(customizations)

        assert shortcut.binding.to_string() == "Ctrl+Shift+S"

    def test_manager_register_action_convenience(self):
        """ShortcutManager has convenience method for actions."""
        manager = ShortcutManager()
        executed = []

        shortcut = manager.register_action(
            "save", "Save",
            "Ctrl+S",
            lambda: executed.append(True),
            category="File"
        )

        assert shortcut.id == "save"
        assert shortcut.binding.key == "S"
        assert shortcut.category == "File"

    def test_manager_get_display_string(self):
        """ShortcutManager provides display string for shortcuts."""
        manager = ShortcutManager()
        shortcut = Shortcut("save", "Save", KeyBinding.from_string("Ctrl+Shift+S"))
        manager.register(shortcut)

        display = manager.get_display_string("save")
        assert "Ctrl" in display
        assert "Shift" in display
        assert "S" in display

    def test_manager_register_unregister_context(self):
        """Contexts can be registered and unregistered."""
        manager = ShortcutManager()
        ctx = ShortcutContext("custom", "Custom")

        manager.register_context(ctx)
        assert any(c.id == "custom" for c in manager.contexts)

        manager.unregister_context("custom")
        assert not any(c.id == "custom" for c in manager.contexts)

    def test_manager_cannot_remove_global_context(self):
        """Global context cannot be unregistered."""
        manager = ShortcutManager()
        result = manager.unregister_context("global")
        assert result is False
