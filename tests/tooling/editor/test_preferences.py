"""
Comprehensive tests for the Preferences system.

Tests cover:
- Preference creation and types
- Preference validation
- Categories and pages
- Save/load preferences
- Preference change notifications
- Restart-required preferences
"""
import pytest
import sys
import tempfile
import os
from enum import Enum

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.editor.preferences import (
    PreferencesManager,
    PreferenceCategory,
    Preference,
    PreferenceType,
    PreferenceValidator,
    PreferencesPage,
)


class ThemeEnum(Enum):
    """Test enum for preference tests."""
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"


class TestPreferenceValidator:
    """Tests for PreferenceValidator class."""

    def test_validator_creation(self):
        """Validator should be created with defaults."""
        validator = PreferenceValidator()
        assert validator.min_value is None
        assert validator.max_value is None

    def test_validator_range(self):
        """Validator can check range."""
        validator = PreferenceValidator()
        validator.set_range(0, 100)

        is_valid, _ = validator.validate(50)
        assert is_valid is True

        is_valid, error = validator.validate(-10)
        assert is_valid is False
        assert ">=0" in error or ">= 0" in error

        is_valid, error = validator.validate(150)
        assert is_valid is False
        assert "<=100" in error or "<= 100" in error

    def test_validator_allowed_values(self):
        """Validator can check allowed values."""
        validator = PreferenceValidator()
        validator.set_allowed_values({"small", "medium", "large"})

        is_valid, _ = validator.validate("medium")
        assert is_valid is True

        is_valid, _ = validator.validate("huge")
        assert is_valid is False

    def test_validator_pattern(self):
        """Validator can check regex pattern."""
        validator = PreferenceValidator()
        validator.set_pattern(r"^[a-z]+$")

        is_valid, _ = validator.validate("hello")
        assert is_valid is True

        is_valid, _ = validator.validate("Hello123")
        assert is_valid is False

    def test_validator_custom(self):
        """Validator can use custom function."""
        validator = PreferenceValidator()
        validator.set_custom(lambda x: x % 2 == 0, "Must be even")

        is_valid, _ = validator.validate(4)
        assert is_valid is True

        is_valid, error = validator.validate(3)
        assert is_valid is False
        assert "even" in error.lower()

    def test_validator_chaining(self):
        """Validator methods can be chained."""
        validator = PreferenceValidator()
        result = validator.set_range(0, 100).set_allowed_values({10, 20, 30})

        assert result == validator


class TestPreference:
    """Tests for Preference class."""

    def test_preference_creation_bool(self):
        """Bool preference is detected automatically."""
        pref = Preference("show_grid", "Show Grid", True)
        assert pref.pref_type == PreferenceType.BOOL
        assert pref.default_value is True
        assert pref.value is True

    def test_preference_creation_int(self):
        """Int preference is detected automatically."""
        pref = Preference("font_size", "Font Size", 12)
        assert pref.pref_type == PreferenceType.INT

    def test_preference_creation_float(self):
        """Float preference is detected automatically."""
        pref = Preference("zoom_level", "Zoom Level", 1.0)
        assert pref.pref_type == PreferenceType.FLOAT

    def test_preference_creation_string(self):
        """String preference is detected automatically."""
        pref = Preference("project_name", "Project Name", "My Project")
        assert pref.pref_type == PreferenceType.STRING

    def test_preference_creation_enum(self):
        """Enum preference is detected automatically."""
        pref = Preference("theme", "Theme", ThemeEnum.DARK)
        assert pref.pref_type == PreferenceType.ENUM

    def test_preference_value_change(self):
        """Preference value can be changed."""
        pref = Preference("value", "Value", 10)

        pref.value = 20
        assert pref.value == 20

    def test_preference_set_value_with_validation(self):
        """Preference validates value on set."""
        validator = PreferenceValidator().set_range(0, 100)
        pref = Preference("value", "Value", 50, validator=validator)

        assert pref.set_value(75) is True
        assert pref.value == 75

        assert pref.set_value(150) is False
        assert pref.value == 75  # Unchanged

    def test_preference_reset_to_default(self):
        """Preference can be reset to default."""
        pref = Preference("value", "Value", 10)
        pref.value = 50

        pref.reset_to_default()
        assert pref.value == 10

    def test_preference_is_modified(self):
        """Preference tracks modification state."""
        pref = Preference("value", "Value", 10)

        assert pref.is_modified() is False

        pref.value = 20
        assert pref.is_modified() is True

        pref.reset_to_default()
        assert pref.is_modified() is False

    def test_preference_change_callback(self):
        """Preference triggers callback on change."""
        changes = []
        pref = Preference("value", "Value", 10)
        pref.on_changed = lambda old, new: changes.append((old, new))

        pref.value = 20

        assert len(changes) == 1
        assert changes[0] == (10, 20)


class TestPreferenceCategory:
    """Tests for PreferenceCategory class."""

    def test_category_creation(self):
        """Category should be created properly."""
        cat = PreferenceCategory("editor", "Editor Settings", order=5)
        assert cat.id == "editor"
        assert cat.name == "Editor Settings"
        assert cat.order == 5

    def test_category_preferences(self):
        """Category tracks preference IDs."""
        cat = PreferenceCategory("general", "General")

        cat.add_preference_id("pref1")
        cat.add_preference_id("pref2")

        assert "pref1" in cat.preference_ids
        assert "pref2" in cat.preference_ids

    def test_category_remove_preference(self):
        """Category can remove preference IDs."""
        cat = PreferenceCategory("general", "General")
        cat.add_preference_id("pref1")

        cat.remove_preference_id("pref1")
        assert "pref1" not in cat.preference_ids


class TestPreferencesPage:
    """Tests for PreferencesPage class."""

    def test_page_creation(self):
        """Page should be created properly."""
        page = PreferencesPage("general", "General Settings", order=0)
        assert page.id == "general"
        assert page.title == "General Settings"

    def test_page_categories(self):
        """Page can hold category IDs."""
        page = PreferencesPage("general", "General")

        page.add_category("editor")
        page.add_category("appearance")

        assert "editor" in page.categories
        assert "appearance" in page.categories


class TestPreferencesManager:
    """Tests for PreferencesManager class."""

    def test_manager_creation(self):
        """PreferencesManager should be created with default categories."""
        manager = PreferencesManager()

        # Should have default categories
        assert len(manager.categories) > 0
        assert any(c.id == "general" for c in manager.categories)

    def test_manager_register_preference(self):
        """PreferencesManager can register preferences."""
        manager = PreferencesManager()
        pref = Preference("show_fps", "Show FPS", True, category="general")

        manager.register(pref)
        assert manager.get("show_fps") == pref

    def test_manager_unregister_preference(self):
        """PreferencesManager can unregister preferences."""
        manager = PreferencesManager()
        pref = Preference("test", "Test", True)
        manager.register(pref)

        removed = manager.unregister("test")
        assert removed == pref
        assert manager.get("test") is None

    def test_manager_get_value(self):
        """PreferencesManager can get values by ID."""
        manager = PreferencesManager()
        pref = Preference("font_size", "Font Size", 12)
        manager.register(pref)

        assert manager.get_value("font_size") == 12
        assert manager.get_value("nonexistent", default=42) == 42

    def test_manager_set_value(self):
        """PreferencesManager can set values by ID."""
        manager = PreferencesManager()
        pref = Preference("font_size", "Font Size", 12)
        manager.register(pref)

        assert manager.set_value("font_size", 14) is True
        assert pref.value == 14

    def test_manager_reset_preference(self):
        """PreferencesManager can reset individual preferences."""
        manager = PreferencesManager()
        pref = Preference("value", "Value", 10)
        manager.register(pref)

        pref.value = 50
        manager.reset_preference("value")

        assert pref.value == 10

    def test_manager_reset_all(self):
        """PreferencesManager can reset all preferences."""
        manager = PreferencesManager()
        p1 = Preference("v1", "V1", 10)
        p2 = Preference("v2", "V2", 20)

        manager.register(p1)
        manager.register(p2)

        p1.value = 100
        p2.value = 200

        manager.reset_all()

        assert p1.value == 10
        assert p2.value == 20

    def test_manager_reset_category(self):
        """PreferencesManager can reset category."""
        manager = PreferencesManager()
        p1 = Preference("v1", "V1", 10, category="general")
        p2 = Preference("v2", "V2", 20, category="editor")

        manager.register(p1)
        manager.register(p2)

        p1.value = 100
        p2.value = 200

        count = manager.reset_category("general")

        assert count == 1
        assert p1.value == 10
        assert p2.value == 200  # Unchanged

    def test_manager_is_dirty(self):
        """PreferencesManager tracks dirty state."""
        manager = PreferencesManager()
        pref = Preference("value", "Value", 10)
        manager.register(pref)

        assert manager.is_dirty is False

        pref.value = 20
        assert manager.is_dirty is True

    def test_manager_change_callback(self):
        """PreferencesManager triggers callback on change."""
        manager = PreferencesManager()
        changes = []
        manager.on_preference_changed = lambda id, old, new: changes.append((id, old, new))

        pref = Preference("value", "Value", 10)
        manager.register(pref)

        pref.value = 20

        assert len(changes) == 1
        assert changes[0] == ("value", 10, 20)

    def test_manager_restart_required(self):
        """PreferencesManager tracks restart-required changes."""
        manager = PreferencesManager()
        pref = Preference("renderer", "Renderer", "opengl", requires_restart=True)
        manager.register(pref)

        assert manager.has_restart_changes is False

        pref.value = "vulkan"
        assert manager.has_restart_changes is True
        assert "renderer" in manager.restart_change_ids

    def test_manager_save_load(self):
        """PreferencesManager can save and load preferences."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Save
            manager1 = PreferencesManager(temp_path)
            pref = Preference("font_size", "Font Size", 12)
            manager1.register(pref)
            pref.value = 16
            manager1.save()

            # Load
            manager2 = PreferencesManager(temp_path)
            pref2 = Preference("font_size", "Font Size", 12)
            manager2.register(pref2)
            manager2.load()

            assert pref2.value == 16
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_manager_export_import_dict(self):
        """PreferencesManager can export/import to dict."""
        manager = PreferencesManager()
        p1 = Preference("v1", "V1", 10)
        p2 = Preference("v2", "V2", "hello")

        manager.register(p1)
        manager.register(p2)

        p1.value = 50
        p2.value = "world"

        data = manager.export_to_dict()
        assert data["v1"] == 50
        assert data["v2"] == "world"

        p1.reset_to_default()
        p2.reset_to_default()

        count = manager.import_from_dict(data)
        assert count == 2
        assert p1.value == 50
        assert p2.value == "world"

    def test_manager_convenience_bool(self):
        """PreferencesManager has convenience method for bool."""
        manager = PreferencesManager()
        pref = manager.register_bool("show_fps", "Show FPS", True)

        assert pref.pref_type == PreferenceType.BOOL
        assert manager.get("show_fps") == pref

    def test_manager_convenience_int(self):
        """PreferencesManager has convenience method for int."""
        manager = PreferencesManager()
        pref = manager.register_int("font_size", "Font Size", 12, min_val=6, max_val=72)

        assert pref.pref_type == PreferenceType.INT
        assert pref.validator is not None

        # Validation should work
        assert pref.set_value(100) is False

    def test_manager_convenience_float(self):
        """PreferencesManager has convenience method for float."""
        manager = PreferencesManager()
        pref = manager.register_float("zoom", "Zoom", 1.0, min_val=0.1, max_val=10.0)

        assert pref.pref_type == PreferenceType.FLOAT

    def test_manager_convenience_string(self):
        """PreferencesManager has convenience method for string."""
        manager = PreferencesManager()
        pref = manager.register_string("username", "Username", "user")

        assert pref.pref_type == PreferenceType.STRING

    def test_manager_convenience_path(self):
        """PreferencesManager has convenience method for path."""
        manager = PreferencesManager()
        pref = manager.register_path("project_dir", "Project Directory", "/home/user/projects")

        assert pref.pref_type == PreferenceType.PATH
        assert pref.widget_hint == "path"

    def test_manager_convenience_color(self):
        """PreferencesManager has convenience method for color."""
        manager = PreferencesManager()
        pref = manager.register_color("bg_color", "Background Color", (0.2, 0.2, 0.2, 1.0))

        assert pref.pref_type == PreferenceType.COLOR
        assert pref.widget_hint == "color"

    def test_manager_get_preferences_in_category(self):
        """PreferencesManager can get preferences by category."""
        manager = PreferencesManager()
        p1 = Preference("v1", "V1", 10, category="editor")
        p2 = Preference("v2", "V2", 20, category="editor")
        p3 = Preference("v3", "V3", 30, category="general")

        manager.register(p1)
        manager.register(p2)
        manager.register(p3)

        editor_prefs = manager.get_preferences_in_category("editor")
        assert len(editor_prefs) == 2
        assert p1 in editor_prefs
        assert p2 in editor_prefs

    def test_manager_categories_sorted_by_order(self):
        """Categories are sorted by order."""
        manager = PreferencesManager()
        manager.register_category(PreferenceCategory("z", "Z", order=100))
        manager.register_category(PreferenceCategory("a", "A", order=1))

        categories = manager.categories
        # Default categories have various orders, but check our custom ones
        assert categories[-1].id == "z" or categories[-1].order >= 100

    def test_manager_pages(self):
        """PreferencesManager can manage pages."""
        manager = PreferencesManager()
        page = PreferencesPage("settings", "Settings", order=0)
        page.add_category("general")

        manager.register_page(page)

        assert manager.get_page("settings") == page
        assert page in manager.pages

    def test_manager_save_cleared_dirty(self):
        """Save clears dirty and restart flags."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            manager = PreferencesManager(temp_path)
            pref = Preference("value", "Value", 10, requires_restart=True)
            manager.register(pref)

            pref.value = 20
            assert manager.is_dirty is True
            assert manager.has_restart_changes is True

            manager.save()

            assert manager.is_dirty is False
            assert manager.has_restart_changes is False
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_manager_save_load_callbacks(self):
        """PreferencesManager triggers save/load callbacks."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            manager = PreferencesManager(temp_path)
            saved = []
            loaded = []
            manager.on_preferences_saved = lambda: saved.append(True)
            manager.on_preferences_loaded = lambda: loaded.append(True)

            pref = Preference("value", "Value", 10)
            manager.register(pref)

            manager.save()
            assert len(saved) == 1

            manager.load()
            assert len(loaded) == 1
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
