"""
Comprehensive tests for Screen Reader accessibility support.

Tests cover:
- ARIA roles
- ARIA properties
- ARIA states
- Live regions
- Focus announcements
- AccessibilityManager singleton
- Role management
- Property management
- State management
- State change announcements
- Live region management
- Announcement handlers
- Focus tracking
- Widget cleanup
"""

import pytest
import sys
from pathlib import Path

# Add engine to path
engine_path = Path(__file__).parent.parent.parent.parent.parent / "engine"
sys.path.insert(0, str(engine_path.parent))

from engine.ui.accessibility.screen_reader import (
    AriaRole,
    AriaProperty,
    AriaState,
    AriaLiveRegion,
    FocusAnnouncement,
    LiveRegionPoliteness,
    AccessibilityManager,
)


class TestAriaRole:
    """Test AriaRole enum."""

    def test_widget_roles_exist(self):
        """Test widget roles are defined."""
        assert AriaRole.BUTTON is not None
        assert AriaRole.CHECKBOX is not None
        assert AriaRole.SLIDER is not None
        assert AriaRole.TEXTBOX is not None

    def test_composite_roles_exist(self):
        """Test composite widget roles are defined."""
        assert AriaRole.MENU is not None
        assert AriaRole.MENUBAR is not None
        assert AriaRole.TABLIST is not None
        assert AriaRole.TREE is not None

    def test_landmark_roles_exist(self):
        """Test landmark roles are defined."""
        assert AriaRole.BANNER is not None
        assert AriaRole.MAIN is not None
        assert AriaRole.NAVIGATION is not None
        assert AriaRole.SEARCH is not None

    def test_live_region_roles_exist(self):
        """Test live region roles are defined."""
        assert AriaRole.ALERT is not None
        assert AriaRole.LOG is not None
        assert AriaRole.STATUS is not None

    def test_document_structure_roles_exist(self):
        """Test document structure roles are defined."""
        assert AriaRole.ARTICLE is not None
        assert AriaRole.HEADING is not None
        assert AriaRole.LIST is not None
        assert AriaRole.TABLE is not None


class TestAriaProperty:
    """Test AriaProperty class."""

    def test_default_property(self):
        """Test default property values."""
        prop = AriaProperty()
        assert prop.label is None
        assert prop.description is None
        assert prop.valuemin is None

    def test_set_label(self):
        """Test setting label."""
        prop = AriaProperty(label="Submit Button")
        assert prop.label == "Submit Button"

    def test_set_value_range(self):
        """Test setting value range."""
        prop = AriaProperty(
            valuemin=0.0,
            valuemax=100.0,
            valuenow=50.0,
        )
        assert prop.valuemin == 0.0
        assert prop.valuemax == 100.0
        assert prop.valuenow == 50.0

    def test_set_value_text(self):
        """Test setting value text."""
        prop = AriaProperty(valuetext="50 percent")
        assert prop.valuetext == "50 percent"

    def test_set_description(self):
        """Test setting description."""
        prop = AriaProperty(description="Click to submit the form")
        assert prop.description == "Click to submit the form"

    def test_get_accessible_name(self):
        """Test getting accessible name."""
        prop = AriaProperty(label="Save File")
        assert prop.get_accessible_name() == "Save File"

    def test_get_accessible_name_from_valuetext(self):
        """Test getting accessible name from valuetext."""
        prop = AriaProperty(valuetext="Volume: 75%")
        assert prop.get_accessible_name() == "Volume: 75%"

    def test_get_accessible_description(self):
        """Test getting accessible description."""
        prop = AriaProperty(description="Saves the current file")
        assert prop.get_accessible_description() == "Saves the current file"

    def test_structure_properties(self):
        """Test structure properties."""
        prop = AriaProperty(
            level=2,
            setsize=5,
            posinset=3,
        )
        assert prop.level == 2
        assert prop.setsize == 5
        assert prop.posinset == 3

    def test_relationship_properties(self):
        """Test relationship properties."""
        prop = AriaProperty(
            controls="menu1",
            owns="submenu",
            flowto="next-section",
        )
        assert prop.controls == "menu1"
        assert prop.owns == "submenu"

    def test_modal_property(self):
        """Test modal property."""
        prop = AriaProperty(modal=True)
        assert prop.modal is True


class TestAriaState:
    """Test AriaState class."""

    def test_default_state(self):
        """Test default state values."""
        state = AriaState()
        assert state.checked is None
        assert state.selected is False
        assert state.expanded is None
        assert state.disabled is False

    def test_checked_state(self):
        """Test checked state."""
        state = AriaState(checked=True)
        assert state.checked is True

    def test_checked_mixed(self):
        """Test mixed checked state."""
        state = AriaState(checked=None)  # None represents mixed
        assert state.checked is None

    def test_expanded_state(self):
        """Test expanded state."""
        state = AriaState(expanded=True)
        assert state.expanded is True

    def test_disabled_state(self):
        """Test disabled state."""
        state = AriaState(disabled=True)
        assert state.disabled is True

    def test_hidden_state(self):
        """Test hidden state."""
        state = AriaState(hidden=True)
        assert state.hidden is True

    def test_busy_state(self):
        """Test busy state."""
        state = AriaState(busy=True)
        assert state.busy is True

    def test_pressed_state(self):
        """Test pressed state."""
        state = AriaState(pressed=True)
        assert state.pressed is True

    def test_invalid_state(self):
        """Test invalid state."""
        state = AriaState(invalid=True)
        assert state.invalid is True

    def test_required_state(self):
        """Test required state."""
        state = AriaState(required=True)
        assert state.required is True

    def test_grabbed_state(self):
        """Test grabbed state (drag and drop)."""
        state = AriaState(grabbed=True)
        assert state.grabbed is True

    def test_has_state_change(self):
        """Test detecting state changes."""
        state1 = AriaState(checked=False)
        state2 = AriaState(checked=True)
        assert state2.has_state_change(state1) is True

    def test_no_state_change(self):
        """Test no change detection."""
        state1 = AriaState(checked=True)
        state2 = AriaState(checked=True)
        assert state2.has_state_change(state1) is False


class TestLiveRegionPoliteness:
    """Test LiveRegionPoliteness enum."""

    def test_off_level(self):
        """Test OFF level exists."""
        assert LiveRegionPoliteness.OFF is not None

    def test_polite_level(self):
        """Test POLITE level exists."""
        assert LiveRegionPoliteness.POLITE is not None

    def test_assertive_level(self):
        """Test ASSERTIVE level exists."""
        assert LiveRegionPoliteness.ASSERTIVE is not None


class TestAriaLiveRegion:
    """Test AriaLiveRegion class."""

    def test_default_live_region(self):
        """Test default live region."""
        region = AriaLiveRegion()
        assert region.politeness == LiveRegionPoliteness.POLITE
        assert region.atomic is False

    def test_assertive_region(self):
        """Test assertive live region."""
        region = AriaLiveRegion(politeness=LiveRegionPoliteness.ASSERTIVE)
        assert region.politeness == LiveRegionPoliteness.ASSERTIVE

    def test_atomic_region(self):
        """Test atomic live region."""
        region = AriaLiveRegion(atomic=True)
        assert region.atomic is True

    def test_announce(self):
        """Test announcing through region."""
        region = AriaLiveRegion()
        region.announce("New notification")
        pending = region.get_pending()
        assert "New notification" in pending

    def test_announce_off_does_nothing(self):
        """Test OFF politeness doesn't announce."""
        region = AriaLiveRegion(politeness=LiveRegionPoliteness.OFF)
        region.announce("This should not be queued")
        pending = region.get_pending()
        assert len(pending) == 0

    def test_get_pending_clears(self):
        """Test get_pending clears the queue."""
        region = AriaLiveRegion()
        region.announce("Message 1")
        region.announce("Message 2")
        pending = region.get_pending()
        assert len(pending) == 2
        pending2 = region.get_pending()
        assert len(pending2) == 0

    def test_clear(self):
        """Test clearing pending announcements."""
        region = AriaLiveRegion()
        region.announce("Message")
        region.clear()
        pending = region.get_pending()
        assert len(pending) == 0


class TestFocusAnnouncement:
    """Test FocusAnnouncement class."""

    def test_basic_announcement(self):
        """Test basic focus announcement."""
        announcement = FocusAnnouncement(
            widget_id="btn1",
            role=AriaRole.BUTTON,
            name="Submit",
        )
        text = announcement.build_announcement()
        assert "Submit" in text
        assert "button" in text

    def test_announcement_with_checked_state(self):
        """Test announcement with checked state."""
        announcement = FocusAnnouncement(
            widget_id="chk1",
            role=AriaRole.CHECKBOX,
            name="Enable notifications",
            state=AriaState(checked=True),
        )
        text = announcement.build_announcement()
        assert "checked" in text

    def test_announcement_with_expanded_state(self):
        """Test announcement with expanded state."""
        announcement = FocusAnnouncement(
            widget_id="menu1",
            role=AriaRole.BUTTON,
            name="Menu",
            state=AriaState(expanded=True),
        )
        text = announcement.build_announcement()
        assert "expanded" in text

    def test_announcement_with_collapsed_state(self):
        """Test announcement with collapsed state."""
        announcement = FocusAnnouncement(
            widget_id="menu1",
            role=AriaRole.BUTTON,
            name="Menu",
            state=AriaState(expanded=False),
        )
        text = announcement.build_announcement()
        assert "collapsed" in text

    def test_announcement_with_position(self):
        """Test announcement with position."""
        announcement = FocusAnnouncement(
            widget_id="item1",
            role=AriaRole.LISTITEM,
            name="Item",
            position="3 of 10",
        )
        text = announcement.build_announcement()
        assert "3 of 10" in text

    def test_announcement_with_hint(self):
        """Test announcement with hint."""
        announcement = FocusAnnouncement(
            widget_id="slider1",
            role=AriaRole.SLIDER,
            name="Volume",
            hint="Use arrow keys to adjust",
        )
        text = announcement.build_announcement()
        assert "arrow keys" in text

    def test_announcement_disabled_state(self):
        """Test announcement with disabled state."""
        announcement = FocusAnnouncement(
            widget_id="btn1",
            role=AriaRole.BUTTON,
            name="Submit",
            state=AriaState(disabled=True),
        )
        text = announcement.build_announcement()
        assert "disabled" in text

    def test_announcement_mixed_checkbox(self):
        """Test announcement for mixed checkbox."""
        announcement = FocusAnnouncement(
            widget_id="chk1",
            role=AriaRole.CHECKBOX,
            name="Select all",
            state=AriaState(checked=None),  # Mixed
        )
        text = announcement.build_announcement()
        assert "mixed" in text


class TestAccessibilityManager:
    """Test AccessibilityManager singleton."""

    def setup_method(self):
        """Reset singleton before each test."""
        AccessibilityManager.reset_instance()

    def test_singleton(self):
        """Test singleton pattern."""
        mgr1 = AccessibilityManager()
        mgr2 = AccessibilityManager()
        assert mgr1 is mgr2

    def test_enabled_by_default(self):
        """Test accessibility is enabled by default."""
        mgr = AccessibilityManager()
        assert mgr.enabled is True

    def test_disable(self):
        """Test disabling accessibility."""
        mgr = AccessibilityManager()
        mgr.enabled = False
        assert mgr.enabled is False


class TestAccessibilityManagerRoles:
    """Test AccessibilityManager role management."""

    def setup_method(self):
        """Reset singleton before each test."""
        AccessibilityManager.reset_instance()

    def test_set_role(self):
        """Test setting widget role."""
        mgr = AccessibilityManager()
        mgr.set_role("widget1", AriaRole.BUTTON)
        assert mgr.get_role("widget1") == AriaRole.BUTTON

    def test_get_role_not_found(self):
        """Test getting nonexistent role."""
        mgr = AccessibilityManager()
        assert mgr.get_role("nonexistent") is None

    def test_remove_role(self):
        """Test removing widget role."""
        mgr = AccessibilityManager()
        mgr.set_role("widget1", AriaRole.BUTTON)
        mgr.remove_role("widget1")
        assert mgr.get_role("widget1") is None


class TestAccessibilityManagerProperties:
    """Test AccessibilityManager property management."""

    def setup_method(self):
        """Reset singleton before each test."""
        AccessibilityManager.reset_instance()

    def test_set_property(self):
        """Test setting widget property."""
        mgr = AccessibilityManager()
        prop = AriaProperty(label="Test Button")
        mgr.set_property("widget1", prop)
        assert mgr.get_property("widget1") is prop

    def test_get_property_not_found(self):
        """Test getting nonexistent property."""
        mgr = AccessibilityManager()
        assert mgr.get_property("nonexistent") is None

    def test_update_property(self):
        """Test updating widget property."""
        mgr = AccessibilityManager()
        prop = AriaProperty(label="Old Label")
        mgr.set_property("widget1", prop)
        mgr.update_property("widget1", label="New Label")
        updated = mgr.get_property("widget1")
        assert updated.label == "New Label"

    def test_update_property_creates_if_missing(self):
        """Test update creates property if missing."""
        mgr = AccessibilityManager()
        mgr.update_property("widget1", label="Created Label")
        prop = mgr.get_property("widget1")
        assert prop is not None
        assert prop.label == "Created Label"

    def test_remove_property(self):
        """Test removing widget property."""
        mgr = AccessibilityManager()
        mgr.set_property("widget1", AriaProperty(label="Test"))
        mgr.remove_property("widget1")
        assert mgr.get_property("widget1") is None


class TestAccessibilityManagerStates:
    """Test AccessibilityManager state management."""

    def setup_method(self):
        """Reset singleton before each test."""
        AccessibilityManager.reset_instance()

    def test_set_state(self):
        """Test setting widget state."""
        mgr = AccessibilityManager()
        state = AriaState(checked=True)
        mgr.set_state("widget1", state)
        assert mgr.get_state("widget1") is state

    def test_get_state_not_found(self):
        """Test getting nonexistent state."""
        mgr = AccessibilityManager()
        assert mgr.get_state("nonexistent") is None

    def test_update_state(self):
        """Test updating widget state."""
        mgr = AccessibilityManager()
        state = AriaState(checked=False)
        mgr.set_state("widget1", state)
        mgr.update_state("widget1", checked=True)
        updated = mgr.get_state("widget1")
        assert updated.checked is True

    def test_update_state_creates_if_missing(self):
        """Test update creates state if missing."""
        mgr = AccessibilityManager()
        mgr.update_state("widget1", disabled=True)
        state = mgr.get_state("widget1")
        assert state is not None
        assert state.disabled is True

    def test_remove_state(self):
        """Test removing widget state."""
        mgr = AccessibilityManager()
        mgr.set_state("widget1", AriaState(checked=True))
        mgr.remove_state("widget1")
        assert mgr.get_state("widget1") is None


class TestAccessibilityManagerLiveRegions:
    """Test AccessibilityManager live region management."""

    def setup_method(self):
        """Reset singleton before each test."""
        AccessibilityManager.reset_instance()

    def test_create_live_region(self):
        """Test creating a live region."""
        mgr = AccessibilityManager()
        region = mgr.create_live_region("notifications")
        assert region is not None
        assert mgr.get_live_region("notifications") is region

    def test_create_assertive_region(self):
        """Test creating assertive region."""
        mgr = AccessibilityManager()
        region = mgr.create_live_region(
            "alerts",
            politeness=LiveRegionPoliteness.ASSERTIVE,
        )
        assert region.politeness == LiveRegionPoliteness.ASSERTIVE

    def test_announce_live(self):
        """Test announcing through live region."""
        mgr = AccessibilityManager()
        mgr.create_live_region("status")
        announcements = []

        def handler(msg, pol):
            announcements.append((msg, pol))

        mgr.add_announcement_handler(handler)
        mgr.announce_live("status", "Operation complete")
        assert len(announcements) >= 1

    def test_announce_polite(self):
        """Test polite announcement."""
        mgr = AccessibilityManager()
        announcements = []

        def handler(msg, pol):
            announcements.append((msg, pol))

        mgr.add_announcement_handler(handler)
        mgr.announce_polite("New message received")
        assert len(announcements) == 1
        assert announcements[0][1] == LiveRegionPoliteness.POLITE

    def test_announce_assertive(self):
        """Test assertive announcement."""
        mgr = AccessibilityManager()
        announcements = []

        def handler(msg, pol):
            announcements.append((msg, pol))

        mgr.add_announcement_handler(handler)
        mgr.announce_assertive("Error occurred!")
        assert len(announcements) == 1
        assert announcements[0][1] == LiveRegionPoliteness.ASSERTIVE

    def test_remove_live_region(self):
        """Test removing a live region."""
        mgr = AccessibilityManager()
        mgr.create_live_region("temp")
        mgr.remove_live_region("temp")
        assert mgr.get_live_region("temp") is None


class TestAccessibilityManagerFocus:
    """Test AccessibilityManager focus tracking."""

    def setup_method(self):
        """Reset singleton before each test."""
        AccessibilityManager.reset_instance()

    def test_announce_focus(self):
        """Test announcing focus."""
        mgr = AccessibilityManager()
        mgr.set_role("btn1", AriaRole.BUTTON)
        mgr.set_property("btn1", AriaProperty(label="Submit"))
        announcements = []

        def handler(msg, pol):
            announcements.append(msg)

        mgr.add_announcement_handler(handler)
        mgr.announce_focus("btn1")
        assert mgr.current_focus == "btn1"
        assert len(announcements) >= 1

    def test_focus_history(self):
        """Test focus history tracking."""
        mgr = AccessibilityManager()
        mgr.set_role("btn1", AriaRole.BUTTON)
        mgr.set_property("btn1", AriaProperty(label="Button 1"))
        mgr.set_role("btn2", AriaRole.BUTTON)
        mgr.set_property("btn2", AriaProperty(label="Button 2"))
        mgr.announce_focus("btn1")
        mgr.announce_focus("btn2")
        history = mgr.get_focus_history()
        assert "btn1" in history
        assert "btn2" in history

    def test_clear_focus(self):
        """Test clearing focus."""
        mgr = AccessibilityManager()
        mgr.set_role("btn1", AriaRole.BUTTON)
        mgr.set_property("btn1", AriaProperty(label="Button"))
        mgr.announce_focus("btn1")
        mgr.clear_focus()
        assert mgr.current_focus is None


class TestAccessibilityManagerAnnouncementHandlers:
    """Test AccessibilityManager announcement handlers."""

    def setup_method(self):
        """Reset singleton before each test."""
        AccessibilityManager.reset_instance()

    def test_add_handler(self):
        """Test adding announcement handler."""
        mgr = AccessibilityManager()
        received = []

        def handler(msg, pol):
            received.append(msg)

        mgr.add_announcement_handler(handler)
        mgr.announce_polite("Test message")
        assert len(received) == 1

    def test_remove_handler(self):
        """Test removing announcement handler."""
        mgr = AccessibilityManager()
        received = []

        def handler(msg, pol):
            received.append(msg)

        mgr.add_announcement_handler(handler)
        mgr.remove_announcement_handler(handler)
        mgr.announce_polite("Test message")
        assert len(received) == 0

    def test_multiple_handlers(self):
        """Test multiple announcement handlers."""
        mgr = AccessibilityManager()
        received1 = []
        received2 = []

        def handler1(msg, pol):
            received1.append(msg)

        def handler2(msg, pol):
            received2.append(msg)

        mgr.add_announcement_handler(handler1)
        mgr.add_announcement_handler(handler2)
        mgr.announce_polite("Test message")
        assert len(received1) == 1
        assert len(received2) == 1


class TestAccessibilityManagerCleanup:
    """Test AccessibilityManager cleanup methods."""

    def setup_method(self):
        """Reset singleton before each test."""
        AccessibilityManager.reset_instance()

    def test_remove_widget(self):
        """Test removing all data for a widget."""
        mgr = AccessibilityManager()
        mgr.set_role("widget1", AriaRole.BUTTON)
        mgr.set_property("widget1", AriaProperty(label="Test"))
        mgr.set_state("widget1", AriaState(disabled=False))
        mgr.remove_widget("widget1")
        assert mgr.get_role("widget1") is None
        assert mgr.get_property("widget1") is None
        assert mgr.get_state("widget1") is None

    def test_clear(self):
        """Test clearing all data."""
        mgr = AccessibilityManager()
        mgr.set_role("widget1", AriaRole.BUTTON)
        mgr.create_live_region("notifications")
        mgr.clear()
        assert mgr.get_role("widget1") is None
        assert mgr.get_live_region("notifications") is None
        assert mgr.current_focus is None


class TestAccessibilityManagerScreenReader:
    """Test AccessibilityManager screen reader detection."""

    def setup_method(self):
        """Reset singleton before each test."""
        AccessibilityManager.reset_instance()

    def test_screen_reader_not_active_by_default(self):
        """Test screen reader not active by default."""
        mgr = AccessibilityManager()
        assert mgr.screen_reader_active is False

    def test_set_screen_reader_active(self):
        """Test setting screen reader active."""
        mgr = AccessibilityManager()
        mgr.set_screen_reader_active(True, "NVDA")
        assert mgr.screen_reader_active is True
        assert mgr.screen_reader_name == "NVDA"

    def test_detect_screen_reader(self):
        """Test detect_screen_reader returns boolean."""
        mgr = AccessibilityManager()
        result = mgr.detect_screen_reader()
        assert isinstance(result, bool)
