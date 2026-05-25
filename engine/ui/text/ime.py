"""
Input Method Editor (IME) support for complex script input.

Provides:
- IMEHandler class for managing IME state
- IME state tracking (composing, committed)
- Composition string handling
- Candidate list management
- IME window positioning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator
from weakref import WeakMethod, ref


class IMEEventType(Enum):
    """Types of IME events."""
    COMPOSITION_START = auto()    # IME composition started
    COMPOSITION_UPDATE = auto()   # Composition string changed
    COMPOSITION_END = auto()      # Composition committed or cancelled
    CANDIDATE_SHOW = auto()       # Candidate list displayed
    CANDIDATE_HIDE = auto()       # Candidate list hidden
    CANDIDATE_SELECT = auto()     # Candidate selected
    MODE_CHANGE = auto()          # IME mode changed (e.g., Hiragana to Katakana)


class IMEState(Enum):
    """Current state of the IME."""
    INACTIVE = auto()       # IME not active
    ACTIVE = auto()         # IME active but not composing
    COMPOSING = auto()      # Currently composing text
    SELECTING = auto()      # Selecting from candidate list


@dataclass
class IMEEvent:
    """An IME-related event.

    Attributes:
        event_type: Type of the event
        composition: Current composition string (if applicable)
        committed_text: Committed text (if applicable)
        cursor_position: Cursor position within composition
        candidate_index: Selected candidate index (if applicable)
        timestamp: Event timestamp
    """
    event_type: IMEEventType
    composition: str = ""
    committed_text: str = ""
    cursor_position: int = 0
    candidate_index: int = -1
    timestamp: float = 0.0

    @property
    def is_composition_event(self) -> bool:
        """Check if this is a composition-related event."""
        return self.event_type in (
            IMEEventType.COMPOSITION_START,
            IMEEventType.COMPOSITION_UPDATE,
            IMEEventType.COMPOSITION_END,
        )

    @property
    def is_candidate_event(self) -> bool:
        """Check if this is a candidate-related event."""
        return self.event_type in (
            IMEEventType.CANDIDATE_SHOW,
            IMEEventType.CANDIDATE_HIDE,
            IMEEventType.CANDIDATE_SELECT,
        )


@dataclass
class CompositionString:
    """Represents the current IME composition.

    Attributes:
        text: The composition text
        cursor: Cursor position within the composition
        selection_start: Start of selection (if any)
        selection_end: End of selection (if any)
        clauses: List of clause ranges for multi-clause composition
        target_clause: Index of the target clause (being converted)
        reading: Reading/pronunciation (for languages like Japanese)
    """
    text: str = ""
    cursor: int = 0
    selection_start: int = 0
    selection_end: int = 0
    clauses: list[tuple[int, int]] = field(default_factory=list)
    target_clause: int = -1
    reading: str = ""

    @property
    def is_empty(self) -> bool:
        """Check if composition is empty."""
        return len(self.text) == 0

    @property
    def has_selection(self) -> bool:
        """Check if there's a selection within the composition."""
        return self.selection_start != self.selection_end

    @property
    def selection_length(self) -> int:
        """Get the length of the selection."""
        return abs(self.selection_end - self.selection_start)

    def get_clause_text(self, index: int) -> str:
        """Get text for a specific clause.

        Args:
            index: Clause index

        Returns:
            Clause text or empty string if invalid
        """
        if 0 <= index < len(self.clauses):
            start, end = self.clauses[index]
            return self.text[start:end]
        return ""

    def get_target_clause_text(self) -> str:
        """Get text of the target (active) clause."""
        if self.target_clause >= 0:
            return self.get_clause_text(self.target_clause)
        return self.text

    def clear(self) -> None:
        """Clear the composition."""
        self.text = ""
        self.cursor = 0
        self.selection_start = 0
        self.selection_end = 0
        self.clauses.clear()
        self.target_clause = -1
        self.reading = ""

    def set_text(self, text: str, cursor: int = -1) -> None:
        """Set composition text.

        Args:
            text: New composition text
            cursor: Cursor position (-1 for end)
        """
        self.text = text
        self.cursor = cursor if cursor >= 0 else len(text)
        self.selection_start = self.cursor
        self.selection_end = self.cursor

    def insert(self, text: str) -> None:
        """Insert text at cursor position.

        Args:
            text: Text to insert
        """
        before = self.text[:self.cursor]
        after = self.text[self.cursor:]
        self.text = before + text + after
        self.cursor += len(text)
        self.selection_start = self.cursor
        self.selection_end = self.cursor

    def delete_selection(self) -> str:
        """Delete selected text.

        Returns:
            Deleted text
        """
        if not self.has_selection:
            return ""

        start = min(self.selection_start, self.selection_end)
        end = max(self.selection_start, self.selection_end)

        deleted = self.text[start:end]
        self.text = self.text[:start] + self.text[end:]
        self.cursor = start
        self.selection_start = start
        self.selection_end = start

        return deleted


@dataclass
class IMECandidate:
    """A single IME candidate for conversion.

    Attributes:
        text: The candidate text
        label: Optional label (e.g., "1", "a")
        reading: Pronunciation/reading
        annotation: Additional info (e.g., word type)
        score: Relevance score (higher = more relevant)
    """
    text: str
    label: str = ""
    reading: str = ""
    annotation: str = ""
    score: float = 0.0

    def __str__(self) -> str:
        """Get string representation."""
        if self.label:
            return f"{self.label}. {self.text}"
        return self.text


@dataclass
class CandidateList:
    """List of IME candidates for conversion.

    Attributes:
        candidates: List of candidates
        selected_index: Currently selected candidate
        page_size: Number of candidates per page
        page_start: Index of first candidate on current page
    """
    candidates: list[IMECandidate] = field(default_factory=list)
    selected_index: int = 0
    page_size: int = 9
    page_start: int = 0

    @property
    def is_empty(self) -> bool:
        """Check if candidate list is empty."""
        return len(self.candidates) == 0

    @property
    def count(self) -> int:
        """Get total number of candidates."""
        return len(self.candidates)

    @property
    def page_count(self) -> int:
        """Get total number of pages."""
        if self.page_size <= 0:
            return 1
        return (len(self.candidates) + self.page_size - 1) // self.page_size

    @property
    def current_page(self) -> int:
        """Get current page number (0-indexed)."""
        if self.page_size <= 0:
            return 0
        return self.page_start // self.page_size

    @property
    def selected(self) -> IMECandidate | None:
        """Get currently selected candidate."""
        if 0 <= self.selected_index < len(self.candidates):
            return self.candidates[self.selected_index]
        return None

    @property
    def page_candidates(self) -> list[IMECandidate]:
        """Get candidates for the current page."""
        end = min(self.page_start + self.page_size, len(self.candidates))
        return self.candidates[self.page_start:end]

    def add(self, candidate: IMECandidate) -> None:
        """Add a candidate to the list.

        Args:
            candidate: Candidate to add
        """
        self.candidates.append(candidate)

    def clear(self) -> None:
        """Clear all candidates."""
        self.candidates.clear()
        self.selected_index = 0
        self.page_start = 0

    def select_next(self) -> bool:
        """Select the next candidate.

        Returns:
            True if selection changed
        """
        if self.selected_index < len(self.candidates) - 1:
            self.selected_index += 1
            self._update_page()
            return True
        return False

    def select_previous(self) -> bool:
        """Select the previous candidate.

        Returns:
            True if selection changed
        """
        if self.selected_index > 0:
            self.selected_index -= 1
            self._update_page()
            return True
        return False

    def select_by_label(self, label: str) -> bool:
        """Select candidate by label.

        Args:
            label: Label to match

        Returns:
            True if candidate was found and selected
        """
        for i, candidate in enumerate(self.candidates):
            if candidate.label == label:
                self.selected_index = i
                self._update_page()
                return True
        return False

    def select_by_index(self, index: int) -> bool:
        """Select candidate by index.

        Args:
            index: Index to select

        Returns:
            True if valid index
        """
        if 0 <= index < len(self.candidates):
            self.selected_index = index
            self._update_page()
            return True
        return False

    def next_page(self) -> bool:
        """Move to the next page.

        Returns:
            True if page changed
        """
        next_start = self.page_start + self.page_size
        if next_start < len(self.candidates):
            self.page_start = next_start
            self.selected_index = self.page_start
            return True
        return False

    def previous_page(self) -> bool:
        """Move to the previous page.

        Returns:
            True if page changed
        """
        if self.page_start > 0:
            self.page_start = max(0, self.page_start - self.page_size)
            self.selected_index = self.page_start
            return True
        return False

    def _update_page(self) -> None:
        """Update page to show selected candidate."""
        if self.selected_index < self.page_start:
            self.page_start = (self.selected_index // self.page_size) * self.page_size
        elif self.selected_index >= self.page_start + self.page_size:
            self.page_start = (self.selected_index // self.page_size) * self.page_size

    def __iter__(self) -> Iterator[IMECandidate]:
        """Iterate over all candidates."""
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)


@dataclass
class IMEWindowPosition:
    """Position and size information for IME windows.

    Attributes:
        x: X coordinate for the composition window
        y: Y coordinate for the composition window
        width: Width of the composition area
        height: Height of a line (for candidate window positioning)
        caret_x: X position of the caret within composition
        caret_y: Y position of the caret
    """
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    caret_x: float = 0.0
    caret_y: float = 0.0

    @property
    def candidate_window_position(self) -> tuple[float, float]:
        """Get recommended position for candidate window.

        Returns:
            (x, y) position below the composition
        """
        return (self.x, self.y + self.height)

    def set_from_rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        """Set position from a rectangle.

        Args:
            x: X coordinate
            y: Y coordinate
            width: Width
            height: Height
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def set_caret(self, x: float, y: float) -> None:
        """Set caret position.

        Args:
            x: Caret X coordinate
            y: Caret Y coordinate
        """
        self.caret_x = x
        self.caret_y = y


# Type alias for event callbacks
IMEEventCallback = Callable[[IMEEvent], None]


class IMEHandler:
    """Manages IME state and events for text input.

    Handles the lifecycle of IME composition and provides
    callbacks for UI integration.
    """

    def __init__(self) -> None:
        """Initialize the IME handler."""
        self._state = IMEState.INACTIVE
        self._composition = CompositionString()
        self._candidates = CandidateList()
        self._window_position = IMEWindowPosition()
        self._mode: str = ""  # IME mode (e.g., "hiragana", "katakana")

        # Event callbacks - using weak references to prevent memory leaks
        # Strong callbacks are kept alive by caller
        self._strong_callbacks: list[IMEEventCallback] = []
        # Weak callbacks are automatically removed when referent is garbage collected
        self._weak_callbacks: list[ref[IMEEventCallback] | WeakMethod] = []

        # Platform-specific handle
        self._platform_handle: Any = None

    @property
    def state(self) -> IMEState:
        """Get current IME state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Check if IME is active."""
        return self._state != IMEState.INACTIVE

    @property
    def is_composing(self) -> bool:
        """Check if currently composing."""
        return self._state == IMEState.COMPOSING

    @property
    def is_selecting(self) -> bool:
        """Check if selecting from candidates."""
        return self._state == IMEState.SELECTING

    @property
    def composition(self) -> CompositionString:
        """Get current composition."""
        return self._composition

    @property
    def candidates(self) -> CandidateList:
        """Get candidate list."""
        return self._candidates

    @property
    def window_position(self) -> IMEWindowPosition:
        """Get IME window position info."""
        return self._window_position

    @property
    def mode(self) -> str:
        """Get current IME mode."""
        return self._mode

    def activate(self) -> None:
        """Activate the IME."""
        if self._state == IMEState.INACTIVE:
            self._state = IMEState.ACTIVE

    def deactivate(self) -> None:
        """Deactivate the IME."""
        if self._state != IMEState.INACTIVE:
            self._cancel_composition()
            self._state = IMEState.INACTIVE

    def start_composition(self) -> None:
        """Start a new composition."""
        if self._state == IMEState.INACTIVE:
            self.activate()

        self._composition.clear()
        self._candidates.clear()
        self._state = IMEState.COMPOSING

        self._emit_event(IMEEvent(
            event_type=IMEEventType.COMPOSITION_START,
        ))

    def update_composition(
        self,
        text: str,
        cursor: int = -1,
        clauses: list[tuple[int, int]] | None = None,
        target_clause: int = -1,
    ) -> None:
        """Update the composition string.

        Args:
            text: New composition text
            cursor: Cursor position (-1 for end)
            clauses: List of clause ranges
            target_clause: Index of target clause
        """
        self._composition.set_text(text, cursor)

        if clauses:
            self._composition.clauses = list(clauses)
        else:
            self._composition.clauses = [(0, len(text))] if text else []

        self._composition.target_clause = target_clause

        self._emit_event(IMEEvent(
            event_type=IMEEventType.COMPOSITION_UPDATE,
            composition=text,
            cursor_position=self._composition.cursor,
        ))

    def commit_composition(self, text: str | None = None) -> str:
        """Commit the composition and return the committed text.

        Args:
            text: Text to commit (or current composition if None)

        Returns:
            Committed text
        """
        committed = text if text is not None else self._composition.text

        self._emit_event(IMEEvent(
            event_type=IMEEventType.COMPOSITION_END,
            committed_text=committed,
        ))

        self._composition.clear()
        self._candidates.clear()
        self._state = IMEState.ACTIVE

        return committed

    def _cancel_composition(self) -> None:
        """Cancel the current composition without committing."""
        if self._state in (IMEState.COMPOSING, IMEState.SELECTING):
            self._emit_event(IMEEvent(
                event_type=IMEEventType.COMPOSITION_END,
                committed_text="",  # Empty = cancelled
            ))

            self._composition.clear()
            self._candidates.clear()

    def show_candidates(self, candidates: list[IMECandidate]) -> None:
        """Show candidate list.

        Args:
            candidates: List of candidates to show
        """
        self._candidates.clear()
        for candidate in candidates:
            self._candidates.add(candidate)

        if candidates:
            self._state = IMEState.SELECTING
            self._emit_event(IMEEvent(
                event_type=IMEEventType.CANDIDATE_SHOW,
            ))

    def hide_candidates(self) -> None:
        """Hide the candidate list."""
        if not self._candidates.is_empty:
            self._candidates.clear()
            self._state = IMEState.COMPOSING

            self._emit_event(IMEEvent(
                event_type=IMEEventType.CANDIDATE_HIDE,
            ))

    def select_candidate(self, index: int) -> IMECandidate | None:
        """Select a candidate by index.

        Args:
            index: Candidate index

        Returns:
            Selected candidate or None if invalid
        """
        if self._candidates.select_by_index(index):
            selected = self._candidates.selected

            self._emit_event(IMEEvent(
                event_type=IMEEventType.CANDIDATE_SELECT,
                candidate_index=index,
            ))

            return selected
        return None

    def select_candidate_by_label(self, label: str) -> IMECandidate | None:
        """Select a candidate by label.

        Args:
            label: Candidate label (e.g., "1", "a")

        Returns:
            Selected candidate or None if not found
        """
        if self._candidates.select_by_label(label):
            return self._candidates.selected
        return None

    def set_mode(self, mode: str) -> None:
        """Set the IME mode.

        Args:
            mode: Mode identifier (e.g., "hiragana", "katakana", "alphanumeric")
        """
        if mode != self._mode:
            self._mode = mode
            self._emit_event(IMEEvent(
                event_type=IMEEventType.MODE_CHANGE,
            ))

    def set_window_position(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        """Set IME window position.

        Args:
            x: X coordinate
            y: Y coordinate
            width: Width
            height: Height
        """
        self._window_position.set_from_rect(x, y, width, height)

    def set_caret_position(self, x: float, y: float) -> None:
        """Set caret position within composition.

        Args:
            x: Caret X coordinate
            y: Caret Y coordinate
        """
        self._window_position.set_caret(x, y)

    def add_callback(self, callback: IMEEventCallback, weak: bool = False) -> None:
        """Add an event callback.

        Args:
            callback: Function to call on IME events
            weak: If True, use weak reference (callback won't prevent GC)
        """
        if weak:
            # Create appropriate weak reference based on callback type
            if hasattr(callback, '__self__'):
                # It's a bound method
                weak_ref = WeakMethod(callback)  # type: ignore
            else:
                weak_ref = ref(callback)
            self._weak_callbacks.append(weak_ref)
        else:
            if callback not in self._strong_callbacks:
                self._strong_callbacks.append(callback)

    def remove_callback(self, callback: IMEEventCallback) -> bool:
        """Remove an event callback.

        Args:
            callback: Callback to remove

        Returns:
            True if callback was found and removed
        """
        if callback in self._strong_callbacks:
            self._strong_callbacks.remove(callback)
            return True
        return False

    def _emit_event(self, event: IMEEvent) -> None:
        """Emit an event to all callbacks.

        Args:
            event: Event to emit
        """
        # Emit to strong callbacks
        for callback in self._strong_callbacks:
            try:
                callback(event)
            except Exception:
                pass  # Don't let callback errors break IME

        # Emit to weak callbacks, cleaning up dead references
        live_weak_callbacks: list[ref[IMEEventCallback] | WeakMethod] = []
        for weak_ref in self._weak_callbacks:
            callback = weak_ref()
            if callback is not None:
                try:
                    callback(event)
                    live_weak_callbacks.append(weak_ref)
                except Exception:
                    pass  # Don't let callback errors break IME
        self._weak_callbacks = live_weak_callbacks

    def handle_key(self, key: str, modifiers: int = 0) -> bool:
        """Handle a key press during composition.

        Args:
            key: Key that was pressed
            modifiers: Modifier key flags

        Returns:
            True if the key was consumed by IME
        """
        if not self.is_active:
            return False

        # Handle candidate navigation
        if self.is_selecting:
            if key == "Up":
                self._candidates.select_previous()
                return True
            elif key == "Down":
                self._candidates.select_next()
                return True
            elif key == "PageUp":
                self._candidates.previous_page()
                return True
            elif key == "PageDown":
                self._candidates.next_page()
                return True
            elif key == "Enter" or key == "Return":
                if self._candidates.selected:
                    text = self._candidates.selected.text
                    self.commit_composition(text)
                    return True
            elif key == "Escape":
                self.hide_candidates()
                return True
            elif len(key) == 1 and key.isdigit():
                # Number key selection (1-9)
                index = int(key) - 1
                if 0 <= index < len(self._candidates.page_candidates):
                    real_index = self._candidates.page_start + index
                    if self.select_candidate(real_index):
                        text = self._candidates.selected.text
                        self.commit_composition(text)
                        return True

        # Handle composition keys
        if self.is_composing:
            if key == "Escape":
                self._cancel_composition()
                self._state = IMEState.ACTIVE
                return True
            elif key == "Enter" or key == "Return":
                if not self._composition.is_empty:
                    self.commit_composition()
                    return True
            elif key == "Backspace":
                if self._composition.cursor > 0:
                    # Delete character before cursor
                    text = self._composition.text
                    cursor = self._composition.cursor
                    new_text = text[:cursor - 1] + text[cursor:]
                    self.update_composition(new_text, cursor - 1)
                    return True
            elif key == "Delete":
                if self._composition.cursor < len(self._composition.text):
                    # Delete character after cursor
                    text = self._composition.text
                    cursor = self._composition.cursor
                    new_text = text[:cursor] + text[cursor + 1:]
                    self.update_composition(new_text, cursor)
                    return True
            elif key == "Left":
                if self._composition.cursor > 0:
                    self._composition.cursor -= 1
                    return True
            elif key == "Right":
                if self._composition.cursor < len(self._composition.text):
                    self._composition.cursor += 1
                    return True
            elif key == "Home":
                self._composition.cursor = 0
                return True
            elif key == "End":
                self._composition.cursor = len(self._composition.text)
                return True

        return False

    def handle_character(self, char: str) -> bool:
        """Handle a character input during composition.

        Args:
            char: Character that was typed

        Returns:
            True if the character was consumed by IME
        """
        if not self.is_active:
            return False

        if not self.is_composing:
            self.start_composition()

        # Insert character into composition
        self._composition.insert(char)

        self._emit_event(IMEEvent(
            event_type=IMEEventType.COMPOSITION_UPDATE,
            composition=self._composition.text,
            cursor_position=self._composition.cursor,
        ))

        return True

    def reset(self) -> None:
        """Reset IME to initial state."""
        self._cancel_composition()
        self._composition.clear()
        self._candidates.clear()
        self._state = IMEState.INACTIVE
        self._mode = ""
        # Note: callbacks are intentionally NOT cleared on reset
        # to allow reuse of the handler with existing listeners

    def set_platform_handle(self, handle: Any) -> None:
        """Set platform-specific IME handle.

        Args:
            handle: Platform IME handle
        """
        self._platform_handle = handle

    @property
    def platform_handle(self) -> Any:
        """Get platform-specific IME handle."""
        return self._platform_handle
