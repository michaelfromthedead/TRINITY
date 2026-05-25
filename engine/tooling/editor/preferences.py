"""
Preferences - User preferences system with categories.

Provides:
- Preference management with type validation
- Categorized preferences for organization
- Save/load preferences to file
- Preference change notifications
- Custom validators for complex types
"""
from __future__ import annotations

import json
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar, Union

from engine.tooling.editor.app_shell import editor, reloadable

T = TypeVar('T')


class PreferenceType(Enum):
    """Types of preferences."""
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    ENUM = auto()
    COLOR = auto()
    PATH = auto()
    LIST = auto()
    DICT = auto()


@editor(category="Preferences")
@reloadable()
class PreferenceValidator:
    """Validates preference values."""
    __slots__ = ("min_value", "max_value", "allowed_values", "pattern",
                 "custom_validator", "error_message")

    def __init__(self):
        self.min_value: Optional[Union[int, float]] = None
        self.max_value: Optional[Union[int, float]] = None
        self.allowed_values: Optional[set[Any]] = None
        self.pattern: Optional[str] = None
        self.custom_validator: Optional[Callable[[Any], bool]] = None
        self.error_message: str = "Invalid value"

    def set_range(self, min_val: Union[int, float],
                  max_val: Union[int, float]) -> "PreferenceValidator":
        """Set min/max range for numeric values."""
        self.min_value = min_val
        self.max_value = max_val
        return self

    def set_allowed_values(self, values: set[Any]) -> "PreferenceValidator":
        """Set allowed values."""
        self.allowed_values = values
        return self

    def set_pattern(self, pattern: str) -> "PreferenceValidator":
        """Set regex pattern for string values."""
        self.pattern = pattern
        return self

    def set_custom(self, validator: Callable[[Any], bool],
                   error_message: str = "") -> "PreferenceValidator":
        """Set custom validation function."""
        self.custom_validator = validator
        if error_message:
            self.error_message = error_message
        return self

    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a value. Returns (is_valid, error_message)."""
        # Range check
        if self.min_value is not None and value < self.min_value:
            return False, f"Value must be >= {self.min_value}"
        if self.max_value is not None and value > self.max_value:
            return False, f"Value must be <= {self.max_value}"

        # Allowed values check
        if self.allowed_values is not None and value not in self.allowed_values:
            return False, f"Value must be one of {self.allowed_values}"

        # Pattern check
        if self.pattern is not None:
            import re
            if not re.match(self.pattern, str(value)):
                return False, f"Value must match pattern {self.pattern}"

        # Custom validation
        if self.custom_validator is not None:
            if not self.custom_validator(value):
                return False, self.error_message

        return True, ""


@editor(category="Preferences")
@reloadable()
class Preference(Generic[T]):
    """A single preference setting."""
    __slots__ = ("id", "name", "description", "pref_type", "default_value",
                 "_value", "category", "validator", "widget_hint",
                 "hidden", "requires_restart", "on_changed", "_manager_ref")

    def __init__(self, id: str, name: str, default_value: T,
                 pref_type: Optional[PreferenceType] = None,
                 description: str = "", category: str = "General",
                 validator: Optional[PreferenceValidator] = None,
                 widget_hint: str = "", hidden: bool = False,
                 requires_restart: bool = False):
        self.id = id
        self.name = name
        self.description = description
        self.default_value = default_value
        self._value = default_value
        self.category = category
        self.validator = validator
        self.widget_hint = widget_hint
        self.hidden = hidden
        self.requires_restart = requires_restart
        self.on_changed = None
        self._manager_ref = None

        # Infer type if not specified
        if pref_type is None:
            if isinstance(default_value, bool):
                self.pref_type = PreferenceType.BOOL
            elif isinstance(default_value, int):
                self.pref_type = PreferenceType.INT
            elif isinstance(default_value, float):
                self.pref_type = PreferenceType.FLOAT
            elif isinstance(default_value, str):
                self.pref_type = PreferenceType.STRING
            elif isinstance(default_value, Enum):
                self.pref_type = PreferenceType.ENUM
            elif isinstance(default_value, (list, tuple)):
                self.pref_type = PreferenceType.LIST
            elif isinstance(default_value, dict):
                self.pref_type = PreferenceType.DICT
            else:
                self.pref_type = PreferenceType.STRING
        else:
            self.pref_type = pref_type

    @property
    def value(self) -> T:
        """Get the preference value."""
        return self._value

    @value.setter
    def value(self, new_value: T) -> None:
        """Set the preference value."""
        self.set_value(new_value)

    def set_value(self, new_value: T, validate: bool = True) -> bool:
        """Set value with optional validation. Returns True if successful."""
        if validate and self.validator:
            is_valid, error = self.validator.validate(new_value)
            if not is_valid:
                return False

        old_value = self._value
        self._value = new_value

        if self.on_changed and old_value != new_value:
            self.on_changed(old_value, new_value)

        return True

    def reset_to_default(self) -> None:
        """Reset to default value."""
        self.set_value(self.default_value, validate=False)

    def is_modified(self) -> bool:
        """Check if value differs from default."""
        return self._value != self.default_value


@editor(category="Preferences")
@reloadable()
class PreferenceCategory:
    """A category for organizing preferences."""
    __slots__ = ("id", "name", "description", "icon", "order",
                 "parent_id", "_preferences")

    def __init__(self, id: str, name: str, description: str = "",
                 icon: str = "", order: int = 0,
                 parent_id: Optional[str] = None):
        self.id = id
        self.name = name
        self.description = description
        self.icon = icon
        self.order = order
        self.parent_id = parent_id
        self._preferences: list[str] = []

    def add_preference_id(self, pref_id: str) -> None:
        """Add a preference ID to this category."""
        if pref_id not in self._preferences:
            self._preferences.append(pref_id)

    def remove_preference_id(self, pref_id: str) -> None:
        """Remove a preference ID from this category."""
        if pref_id in self._preferences:
            self._preferences.remove(pref_id)

    @property
    def preference_ids(self) -> list[str]:
        """Get preference IDs in this category."""
        return list(self._preferences)


@editor(category="Preferences")
@reloadable()
class PreferencesPage:
    """A page in the preferences dialog."""
    __slots__ = ("id", "title", "icon", "categories", "order")

    def __init__(self, id: str, title: str, icon: str = "", order: int = 0):
        self.id = id
        self.title = title
        self.icon = icon
        self.categories: list[str] = []
        self.order = order

    def add_category(self, category_id: str) -> None:
        """Add a category to this page."""
        if category_id not in self.categories:
            self.categories.append(category_id)


@editor(category="Preferences")
@reloadable(preserve=["_preferences", "_categories"])
class PreferencesManager:
    """Manages user preferences."""
    __slots__ = ("_preferences", "_categories", "_pages", "_dirty",
                 "_file_path", "on_preference_changed", "on_preferences_saved",
                 "on_preferences_loaded", "_pending_restart_changes", "__weakref__")

    def __init__(self, file_path: Optional[str] = None):
        self._preferences: dict[str, Preference] = {}
        self._categories: dict[str, PreferenceCategory] = {}
        self._pages: dict[str, PreferencesPage] = {}
        self._dirty: bool = False
        self._file_path = file_path
        self._pending_restart_changes: list[str] = []
        self.on_preference_changed: Optional[Callable[[str, Any, Any], None]] = None
        self.on_preferences_saved: Optional[Callable[[], None]] = None
        self.on_preferences_loaded: Optional[Callable[[], None]] = None

        # Create default categories
        self._create_default_categories()

    def _create_default_categories(self) -> None:
        """Create default preference categories."""
        defaults = [
            PreferenceCategory("general", "General", order=0),
            PreferenceCategory("appearance", "Appearance", order=1),
            PreferenceCategory("editor", "Editor", order=2),
            PreferenceCategory("viewport", "Viewport", order=3),
            PreferenceCategory("input", "Input", order=4),
            PreferenceCategory("performance", "Performance", order=5),
            PreferenceCategory("advanced", "Advanced", order=100),
        ]
        for cat in defaults:
            self._categories[cat.id] = cat

    @property
    def is_dirty(self) -> bool:
        """Check if preferences have unsaved changes."""
        return self._dirty

    @property
    def has_restart_changes(self) -> bool:
        """Check if there are changes requiring restart."""
        return len(self._pending_restart_changes) > 0

    @property
    def restart_change_ids(self) -> list[str]:
        """Get IDs of preferences with pending restart changes."""
        return list(self._pending_restart_changes)

    def register(self, preference: Preference) -> None:
        """Register a preference."""
        self._preferences[preference.id] = preference
        preference._manager_ref = weakref.ref(self)

        # Wire up change notification
        original_callback = preference.on_changed
        def on_change(old_val: Any, new_val: Any) -> None:
            self._dirty = True
            if preference.requires_restart:
                if preference.id not in self._pending_restart_changes:
                    self._pending_restart_changes.append(preference.id)
            if self.on_preference_changed:
                self.on_preference_changed(preference.id, old_val, new_val)
            if original_callback:
                original_callback(old_val, new_val)
        preference.on_changed = on_change

        # Add to category
        cat = self._categories.get(preference.category)
        if cat:
            cat.add_preference_id(preference.id)
        elif preference.category:
            # Create category if it doesn't exist
            cat = PreferenceCategory(preference.category, preference.category.title())
            self._categories[cat.id] = cat
            cat.add_preference_id(preference.id)

    def unregister(self, preference_id: str) -> Optional[Preference]:
        """Unregister a preference."""
        pref = self._preferences.pop(preference_id, None)
        if pref:
            cat = self._categories.get(pref.category)
            if cat:
                cat.remove_preference_id(preference_id)
        return pref

    def get(self, preference_id: str) -> Optional[Preference]:
        """Get a preference by ID."""
        return self._preferences.get(preference_id)

    def get_value(self, preference_id: str, default: Any = None) -> Any:
        """Get a preference value by ID."""
        pref = self._preferences.get(preference_id)
        return pref.value if pref else default

    def set_value(self, preference_id: str, value: Any) -> bool:
        """Set a preference value by ID. Returns True if successful."""
        pref = self._preferences.get(preference_id)
        if pref:
            return pref.set_value(value)
        return False

    def reset_preference(self, preference_id: str) -> bool:
        """Reset a preference to default."""
        pref = self._preferences.get(preference_id)
        if pref:
            pref.reset_to_default()
            return True
        return False

    def reset_all(self) -> None:
        """Reset all preferences to defaults."""
        for pref in self._preferences.values():
            pref.reset_to_default()
        self._dirty = True

    def reset_category(self, category_id: str) -> int:
        """Reset all preferences in a category. Returns count reset."""
        cat = self._categories.get(category_id)
        if not cat:
            return 0

        count = 0
        for pref_id in cat.preference_ids:
            if self.reset_preference(pref_id):
                count += 1
        return count

    # Category management
    def register_category(self, category: PreferenceCategory) -> None:
        """Register a preference category."""
        self._categories[category.id] = category

    def get_category(self, category_id: str) -> Optional[PreferenceCategory]:
        """Get a category by ID."""
        return self._categories.get(category_id)

    @property
    def categories(self) -> list[PreferenceCategory]:
        """Get all categories sorted by order."""
        return sorted(self._categories.values(), key=lambda c: c.order)

    def get_preferences_in_category(self, category_id: str) -> list[Preference]:
        """Get all preferences in a category."""
        cat = self._categories.get(category_id)
        if not cat:
            return []
        return [self._preferences[pid] for pid in cat.preference_ids
                if pid in self._preferences]

    # Page management
    def register_page(self, page: PreferencesPage) -> None:
        """Register a preferences page."""
        self._pages[page.id] = page

    def get_page(self, page_id: str) -> Optional[PreferencesPage]:
        """Get a page by ID."""
        return self._pages.get(page_id)

    @property
    def pages(self) -> list[PreferencesPage]:
        """Get all pages sorted by order."""
        return sorted(self._pages.values(), key=lambda p: p.order)

    # Persistence
    def save(self, file_path: Optional[str] = None) -> bool:
        """Save preferences to file. Returns True if successful."""
        path = file_path or self._file_path
        if not path:
            return False

        try:
            data = {}
            for pref_id, pref in self._preferences.items():
                # Only save modified preferences
                if pref.is_modified():
                    value = pref.value
                    # Handle special types
                    if pref.pref_type == PreferenceType.ENUM and isinstance(value, Enum):
                        value = value.name
                    data[pref_id] = value

            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

            self._dirty = False
            self._pending_restart_changes.clear()

            if self.on_preferences_saved:
                self.on_preferences_saved()

            return True
        except Exception:
            return False

    def load(self, file_path: Optional[str] = None) -> bool:
        """Load preferences from file. Returns True if successful."""
        path = file_path or self._file_path
        if not path or not Path(path).exists():
            return False

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            for pref_id, value in data.items():
                pref = self._preferences.get(pref_id)
                if pref:
                    # Handle enum conversion
                    if pref.pref_type == PreferenceType.ENUM:
                        if isinstance(pref.default_value, Enum):
                            enum_type = type(pref.default_value)
                            try:
                                value = enum_type[value]
                            except KeyError:
                                continue
                    pref.set_value(value, validate=False)

            self._dirty = False

            if self.on_preferences_loaded:
                self.on_preferences_loaded()

            return True
        except Exception:
            return False

    def export_to_dict(self) -> dict[str, Any]:
        """Export all preference values to a dictionary."""
        return {pid: pref.value for pid, pref in self._preferences.items()}

    def import_from_dict(self, data: dict[str, Any]) -> int:
        """Import preference values from dictionary. Returns count imported."""
        count = 0
        for pref_id, value in data.items():
            if self.set_value(pref_id, value):
                count += 1
        return count

    # Convenience methods for common preference types
    def register_bool(self, id: str, name: str, default: bool,
                      category: str = "General", **kwargs) -> Preference[bool]:
        """Register a boolean preference."""
        pref = Preference(id, name, default, PreferenceType.BOOL,
                          category=category, **kwargs)
        self.register(pref)
        return pref

    def register_int(self, id: str, name: str, default: int,
                     category: str = "General", min_val: Optional[int] = None,
                     max_val: Optional[int] = None, **kwargs) -> Preference[int]:
        """Register an integer preference."""
        validator = None
        if min_val is not None or max_val is not None:
            validator = PreferenceValidator()
            if min_val is not None:
                validator.min_value = min_val
            if max_val is not None:
                validator.max_value = max_val
        pref = Preference(id, name, default, PreferenceType.INT,
                          category=category, validator=validator, **kwargs)
        self.register(pref)
        return pref

    def register_float(self, id: str, name: str, default: float,
                       category: str = "General", min_val: Optional[float] = None,
                       max_val: Optional[float] = None, **kwargs) -> Preference[float]:
        """Register a float preference."""
        validator = None
        if min_val is not None or max_val is not None:
            validator = PreferenceValidator()
            if min_val is not None:
                validator.min_value = min_val
            if max_val is not None:
                validator.max_value = max_val
        pref = Preference(id, name, default, PreferenceType.FLOAT,
                          category=category, validator=validator, **kwargs)
        self.register(pref)
        return pref

    def register_string(self, id: str, name: str, default: str,
                        category: str = "General", **kwargs) -> Preference[str]:
        """Register a string preference."""
        pref = Preference(id, name, default, PreferenceType.STRING,
                          category=category, **kwargs)
        self.register(pref)
        return pref

    def register_path(self, id: str, name: str, default: str,
                      category: str = "General", **kwargs) -> Preference[str]:
        """Register a path preference."""
        pref = Preference(id, name, default, PreferenceType.PATH,
                          category=category, widget_hint="path", **kwargs)
        self.register(pref)
        return pref

    def register_color(self, id: str, name: str,
                       default: tuple[float, float, float, float],
                       category: str = "General", **kwargs) -> Preference:
        """Register a color preference."""
        pref = Preference(id, name, default, PreferenceType.COLOR,
                          category=category, widget_hint="color", **kwargs)
        self.register(pref)
        return pref
