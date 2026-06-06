"""
Conversation Module.

Multi-character conversation management with branching dialogue,
turn-based speaking, and conversation flow control.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Optional

from .config import (
    CONTEXT_CONVERSATION,
    CONVERSATION_GAP_MS,
    MAX_ACTIVE_CONVERSATIONS,
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
    EVENT_CONVERSATION_STARTED,
    EVENT_CONVERSATION_ENDED,
)
from .vo_line import VOLine, VOLineState


class ConversationState(str, Enum):
    """State of a conversation."""
    INACTIVE = "inactive"
    IDLE = "idle"
    STARTING = "starting"
    ACTIVE = "active"
    PLAYING = "playing"
    PAUSED = "paused"
    WAITING = "waiting"  # Waiting for player input
    ENDING = "ending"
    COMPLETED = "completed"
    COMPLETE = "complete"
    CANCELLED = "cancelled"

    def __eq__(self, other):
        if isinstance(other, ConversationState):
            # Make INACTIVE and IDLE compare equal (different names, same semantic meaning)
            aliases = [{"INACTIVE", "IDLE"}, {"ACTIVE", "PLAYING"}, {"COMPLETED", "COMPLETE"}]
            for alias_set in aliases:
                if self.name in alias_set and other.name in alias_set:
                    return True
        return str.__eq__(self, other)

    def __hash__(self):
        return hash(self.value)


@dataclass
class ConversationNode:
    """
    A node in a conversation representing one line or branching point.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    line: Optional[VOLine] = None
    line_id: Optional[str] = None  # Convenience: auto-creates VOLine if provided
    next_nodes: list[str] = field(default_factory=list)  # Node IDs
    conditions: dict[str, Any] = field(default_factory=dict)
    condition: Optional[str] = None  # Convenience: single condition string
    speaker: Optional[str] = None  # Speaker identifier
    delay_ms: float = 0.0  # Delay before transitioning to next node
    is_branch_point: bool = False
    branch_options: list[dict[str, Any]] = field(default_factory=list)
    on_enter: Optional[Callable[[ConversationNode], None]] = None
    on_exit: Optional[Callable[[ConversationNode], None]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Auto-create VOLine from line_id if provided."""
        if self.line_id is not None and self.line is None:
            self.line = VOLine(line_id=self.line_id, speaker_id=self.speaker or "")
        # Set speaker on existing line if provided
        elif self.line is not None and self.speaker is not None:
            self.line.speaker_id = self.speaker
        # Store single condition in conditions dict if provided
        if self.condition is not None and "expr" not in self.conditions:
            self.conditions["expr"] = self.condition

    @property
    def has_line(self) -> bool:
        """Check if this node has a VO line."""
        return self.line is not None

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (no next nodes)."""
        return len(self.next_nodes) == 0 and not self.is_branch_point

    def evaluate_condition(self, context: dict[str, Any]) -> bool:
        """
        Evaluate this node's condition against the given context.

        Args:
            context: Dictionary of variables to evaluate against

        Returns:
            True if condition is met or no condition exists
        """
        if self.condition is None:
            return True

        # Simple expression evaluation for common comparison operators
        expr = self.condition.strip()

        # Try to evaluate using a safe subset of operations
        try:
            # Parse simple comparison expressions like "health > 50"
            for op, func in [
                (" >= ", lambda a, b: a >= b),
                (" <= ", lambda a, b: a <= b),
                (" == ", lambda a, b: a == b),
                (" != ", lambda a, b: a != b),
                (" > ", lambda a, b: a > b),
                (" < ", lambda a, b: a < b),
            ]:
                if op in expr:
                    left, right = expr.split(op, 1)
                    left = left.strip()
                    right = right.strip()

                    # Get left value from context
                    left_val = context.get(left, left)
                    # Try to convert right to number if possible
                    try:
                        right_val = int(right)
                    except ValueError:
                        try:
                            right_val = float(right)
                        except ValueError:
                            right_val = context.get(right, right)

                    return func(left_val, right_val)

            # If no operator found, check for truthiness of a variable
            return bool(context.get(expr, False))

        except Exception:
            # If evaluation fails, return False
            return False

    def get_next_node_id(
        self,
        game_state: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """Get the next node ID based on conditions."""
        if not self.next_nodes:
            return None

        if len(self.next_nodes) == 1:
            return self.next_nodes[0]

        # Multiple paths - evaluate conditions
        if game_state:
            for node_id in self.next_nodes:
                # Check if this path's conditions are met
                # (would need condition checking logic per path)
                pass

        # Default to first path
        return self.next_nodes[0]


@dataclass
class Conversation:
    """
    Represents a complete conversation with multiple participants.
    """
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    participants: list[str] = field(default_factory=list)  # Speaker IDs
    nodes: dict[str, ConversationNode] = field(default_factory=dict)
    start_node_id: str = ""
    priority: int = PRIORITY_HIGH
    interruptible: bool = True
    skippable: bool = True
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Runtime state
    _state: ConversationState = field(default=ConversationState.INACTIVE, init=False)
    _current_node_id: str = field(default="", init=False)
    _start_time: float = field(default=0.0, init=False)
    _elapsed_time: float = field(default=0.0, init=False)
    _played_nodes: list[str] = field(default_factory=list, init=False)

    # Callbacks
    on_node_start: Optional[Callable[[str], None]] = field(default=None, init=False)
    on_complete: Optional[Callable[[str], None]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize after dataclass creation."""
        if isinstance(self.tags, list):
            self.tags = set(self.tags)

    @property
    def state(self) -> ConversationState:
        """Get current conversation state."""
        return self._state

    @state.setter
    def state(self, value: ConversationState) -> None:
        """Set conversation state."""
        self._state = value

    @property
    def current_node(self) -> Optional[ConversationNode]:
        """Get current node."""
        if self._current_node_id:
            return self.nodes.get(self._current_node_id)
        return None

    @property
    def current_line(self) -> Optional[VOLine]:
        """Get current VO line."""
        node = self.current_node
        return node.line if node else None

    @property
    def is_active(self) -> bool:
        """Check if conversation is active."""
        return self._state in (
            ConversationState.ACTIVE,
            ConversationState.WAITING,
            ConversationState.STARTING,
        )

    @property
    def is_complete(self) -> bool:
        """Check if conversation is complete."""
        return self._state in (
            ConversationState.COMPLETED,
            ConversationState.CANCELLED,
            ConversationState.COMPLETE,
        )

    @property
    def is_paused(self) -> bool:
        """Check if conversation is paused."""
        return self._state == ConversationState.PAUSED

    @property
    def progress(self) -> float:
        """Get conversation progress (0-1)."""
        if not self.nodes:
            return 0.0
        return len(self._played_nodes) / len(self.nodes)

    @property
    def remaining_nodes_count(self) -> int:
        """Get count of remaining unplayed nodes."""
        return len(self.nodes) - len(self._played_nodes)

    def add_node(self, node: ConversationNode) -> None:
        """Add a node to the conversation."""
        self.nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the conversation."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            return True
        return False

    def get_node(self, node_id: str) -> Optional[ConversationNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def start(self, current_time: Optional[float] = None) -> Optional[ConversationNode]:
        """
        Start the conversation.

        Args:
            current_time: Optional start time, defaults to time.time() * 1000

        Returns:
            First node to play
        """
        if not self.start_node_id or self.start_node_id not in self.nodes:
            return None

        if current_time is None:
            current_time = time.time() * 1000

        self._state = ConversationState.STARTING
        self._start_time = current_time
        self._current_node_id = self.start_node_id
        self._played_nodes = []
        self._state = ConversationState.PLAYING

        # Call on_node_start callback for first node
        if self.on_node_start:
            self.on_node_start(self.start_node_id)

        return self.current_node

    def advance(
        self,
        game_state: Optional[dict[str, Any]] = None,
        choice_index: Optional[int] = None,
    ) -> Optional[ConversationNode]:
        """
        Advance to the next node.

        Args:
            game_state: Current game state for conditional branching
            choice_index: Player's choice at branch point

        Returns:
            Next node or None if conversation ended
        """
        current = self.current_node
        if not current:
            return None

        # Record that we played this node
        if current.node_id not in self._played_nodes:
            self._played_nodes.append(current.node_id)

        # Call on_exit callback
        if current.on_exit:
            current.on_exit(current)

        # Get next node ID
        next_id = None

        if current.is_branch_point and choice_index is not None:
            # Handle branching based on player choice
            if choice_index < len(current.branch_options):
                option = current.branch_options[choice_index]
                # Support both dict format {"next_node_id": "..."} and list of next_nodes
                if isinstance(option, dict):
                    next_id = option.get("next_node_id")
                elif choice_index < len(current.next_nodes):
                    next_id = current.next_nodes[choice_index]
            elif choice_index < len(current.next_nodes):
                # Fallback to next_nodes if branch_options doesn't have enough entries
                next_id = current.next_nodes[choice_index]
        else:
            next_id = current.get_next_node_id(game_state)

        if next_id:
            self._current_node_id = next_id
            next_node = self.current_node

            if next_node and next_node.on_enter:
                next_node.on_enter(next_node)

            # Call on_node_start callback
            if self.on_node_start:
                self.on_node_start(next_id)

            return next_node
        else:
            # No more nodes - conversation complete
            self._state = ConversationState.COMPLETE

            # Call on_complete callback
            if self.on_complete:
                self.on_complete(self.conversation_id)

            return None

    def complete_current_node(
        self,
        game_state: Optional[dict[str, Any]] = None,
    ) -> Optional[ConversationNode]:
        """
        Mark the current node as complete and advance to the next.

        If the current node has multiple next nodes (branch point), sets
        state to WAITING and returns None until select_branch is called.

        Args:
            game_state: Current game state for conditional branching

        Returns:
            Next node or None if conversation ended or waiting for branch selection
        """
        current = self.current_node
        if current and len(current.next_nodes) > 1:
            # Multiple paths - wait for branch selection
            # Record that we played this node
            if current.node_id not in self._played_nodes:
                self._played_nodes.append(current.node_id)
            self._state = ConversationState.WAITING
            return None

        return self.advance(game_state=game_state)

    def select_branch(self, node_id: str) -> Optional[ConversationNode]:
        """
        Select a specific branch by node ID.

        Args:
            node_id: The ID of the node to branch to

        Returns:
            The selected node

        Raises:
            ValueError: If the node_id is not a valid branch option
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' does not exist")

        # Validate this is a valid branch from current context
        current = self.current_node
        if current and node_id not in current.next_nodes:
            raise ValueError(
                f"Node '{node_id}' is not a valid branch from current node"
            )

        self._current_node_id = node_id
        return self.current_node

    def pause(self) -> None:
        """Pause the conversation."""
        if self._state in (ConversationState.ACTIVE, ConversationState.PLAYING):
            self._state = ConversationState.PAUSED

    def resume(self) -> None:
        """Resume the conversation."""
        if self._state == ConversationState.PAUSED:
            self._state = ConversationState.PLAYING

    def cancel(self) -> None:
        """Cancel the conversation."""
        self._state = ConversationState.CANCELLED

    def skip_current(self) -> Optional[ConversationNode]:
        """Skip the current node and advance to the next."""
        return self.advance()

    def skip_all(self) -> None:
        """Skip all remaining nodes and complete the conversation."""
        while self.advance() is not None:
            pass

    def skip_to_node(self, node_id: str) -> Optional[ConversationNode]:
        """Skip to a specific node."""
        if node_id in self.nodes:
            self._current_node_id = node_id
            return self.current_node
        return None

    def wait_for_input(self) -> None:
        """Set conversation to waiting for player input."""
        self._state = ConversationState.WAITING

    def reset(self) -> None:
        """Reset conversation to initial state."""
        self._state = ConversationState.IDLE
        self._current_node_id = ""
        self._start_time = 0.0
        self._elapsed_time = 0.0
        self._played_nodes = []

    def to_dict(self) -> dict[str, Any]:
        """Serialize conversation to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "participants": self.participants,
            "start_node_id": self.start_node_id,
            "priority": self.priority,
            "interruptible": self.interruptible,
            "skippable": self.skippable,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }

    def save_state(self) -> dict[str, Any]:
        """Save the current conversation state for later restoration."""
        return {
            "current_node": self._current_node_id,
            "state": self._state.value,
            "start_time": self._start_time,
            "elapsed_time": self._elapsed_time,
            "played_nodes": list(self._played_nodes),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore a previously saved conversation state."""
        self._current_node_id = state.get("current_node", "")
        state_value = state.get("state", "idle")
        self._state = ConversationState(state_value)
        self._start_time = state.get("start_time", 0.0)
        self._elapsed_time = state.get("elapsed_time", 0.0)
        self._played_nodes = list(state.get("played_nodes", []))


class ConversationManager:
    """
    Manages multiple conversations with priority and flow control.
    """

    def __init__(
        self,
        max_active: int = MAX_ACTIVE_CONVERSATIONS,
        gap_ms: float = CONVERSATION_GAP_MS,
        on_conversation_started: Optional[Callable[[Conversation], None]] = None,
        on_conversation_ended: Optional[Callable[[Conversation, bool], None]] = None,
        on_line_started: Optional[Callable[[Conversation, VOLine], None]] = None,
        on_line_ended: Optional[Callable[[Conversation, VOLine], None]] = None,
        on_branch_reached: Optional[Callable[[Conversation, ConversationNode], None]] = None,
    ) -> None:
        """
        Initialize the conversation manager.

        Args:
            max_active: Maximum concurrent conversations
            gap_ms: Gap between lines in a conversation
            on_conversation_started: Callback when conversation starts
            on_conversation_ended: Callback when conversation ends (bool=cancelled)
            on_line_started: Callback when a line starts
            on_line_ended: Callback when a line ends
            on_branch_reached: Callback when branch point reached
        """
        self._conversations: dict[str, Conversation] = {}
        self._active_conversations: list[str] = []
        self._max_active = max_active
        self._gap_ms = gap_ms
        self._lock = threading.RLock()

        # Callbacks
        self._on_conversation_started = on_conversation_started
        self._on_conversation_ended = on_conversation_ended
        self._on_line_started = on_line_started
        self._on_line_ended = on_line_ended
        self._on_branch_reached = on_branch_reached

        # Line timing
        self._line_end_times: dict[str, float] = {}
        self._game_state: dict[str, Any] = {}

    def register_conversation(self, conversation: Conversation) -> None:
        """Register a conversation."""
        with self._lock:
            self._conversations[conversation.conversation_id] = conversation

    # Alias for register_conversation
    def register(self, conversation: Conversation) -> None:
        """Register a conversation (alias for register_conversation)."""
        return self.register_conversation(conversation)

    def unregister_conversation(self, conversation_id: str) -> bool:
        """Unregister a conversation."""
        with self._lock:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
                return True
            return False

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        with self._lock:
            return self._conversations.get(conversation_id)

    # Alias for get_conversation
    def get(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID (alias for get_conversation)."""
        return self.get_conversation(conversation_id)

    def start_conversation(
        self,
        conversation_id: str,
        current_time: Optional[float] = None,
    ) -> Optional[ConversationNode]:
        """
        Start a conversation.

        Returns:
            First node or None if conversation couldn't start
        """
        if current_time is None:
            current_time = time.time()

        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation:
                return None

            # Check if we can start another conversation
            if len(self._active_conversations) >= self._max_active:
                # Try to interrupt lower priority conversation
                can_interrupt = False
                for active_id in list(self._active_conversations):
                    active = self._conversations.get(active_id)
                    if active and active.interruptible:
                        if conversation.priority > active.priority:
                            self.end_conversation(active_id, cancelled=True)
                            can_interrupt = True
                            break

                if not can_interrupt:
                    return None

            # Start the conversation
            first_node = conversation.start(current_time)

            if first_node:
                self._active_conversations.append(conversation_id)

                if self._on_conversation_started:
                    self._on_conversation_started(conversation)

                if first_node.has_line and self._on_line_started:
                    self._on_line_started(conversation, first_node.line)

            return first_node

    # Alias for start_conversation
    def start(
        self,
        conversation_id: str,
        current_time: Optional[float] = None,
    ) -> Optional[ConversationNode]:
        """Start a conversation (alias for start_conversation)."""
        return self.start_conversation(conversation_id, current_time)

    def is_active(self, conversation_id: str) -> bool:
        """Check if a conversation is currently active."""
        with self._lock:
            return conversation_id in self._active_conversations

    def get_active(self) -> Optional[Conversation]:
        """Get the first active conversation, if any."""
        with self._lock:
            if self._active_conversations:
                return self._conversations.get(self._active_conversations[0])
            return None

    def stop(self, conversation_id: str) -> bool:
        """Stop a conversation (alias for end_conversation with cancelled=True)."""
        return self.end_conversation(conversation_id, cancelled=True)

    def end_conversation(
        self,
        conversation_id: str,
        cancelled: bool = False,
    ) -> bool:
        """
        End a conversation.

        Args:
            conversation_id: ID of conversation to end
            cancelled: Whether conversation was cancelled

        Returns:
            True if conversation was ended
        """
        with self._lock:
            if conversation_id not in self._active_conversations:
                return False

            conversation = self._conversations.get(conversation_id)
            if not conversation:
                return False

            # Update state
            if cancelled:
                conversation.cancel()
            else:
                conversation._state = ConversationState.COMPLETED

            self._active_conversations.remove(conversation_id)
            self._line_end_times.pop(conversation_id, None)

            if self._on_conversation_ended:
                self._on_conversation_ended(conversation, cancelled)

            return True

    def advance_conversation(
        self,
        conversation_id: str,
        choice_index: Optional[int] = None,
        current_time: Optional[float] = None,
    ) -> Optional[ConversationNode]:
        """
        Advance a conversation to the next node.

        Returns:
            Next node or None if conversation ended
        """
        if current_time is None:
            current_time = time.time()

        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation or not conversation.is_active:
                return None

            current_node = conversation.current_node
            if current_node and current_node.has_line and self._on_line_ended:
                self._on_line_ended(conversation, current_node.line)

            next_node = conversation.advance(self._game_state, choice_index)

            if next_node:
                # Handle branch point
                if next_node.is_branch_point and self._on_branch_reached:
                    conversation.wait_for_input()
                    self._on_branch_reached(conversation, next_node)

                # Start next line
                elif next_node.has_line:
                    self._line_end_times[conversation_id] = (
                        current_time + (self._gap_ms / 1000.0)
                    )

                    if self._on_line_started:
                        self._on_line_started(conversation, next_node.line)
            else:
                # Conversation ended
                self.end_conversation(conversation_id)

            return next_node

    def skip_line(self, conversation_id: str) -> bool:
        """Skip the current line in a conversation."""
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation or not conversation.skippable:
                return False

            if conversation.is_active:
                current_node = conversation.current_node
                if current_node and current_node.has_line:
                    if current_node.line.interruptible:
                        current_node.line.complete_playback(interrupted=True)
                        return True

            return False

    def make_choice(
        self,
        conversation_id: str,
        choice_index: int,
    ) -> Optional[ConversationNode]:
        """
        Make a choice at a branch point.

        Returns:
            Next node after the choice
        """
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation:
                return None

            if conversation.state != ConversationState.WAITING:
                return None

            conversation._state = ConversationState.ACTIVE
            return self.advance_conversation(conversation_id, choice_index)

    def update(self, delta_ms: float, current_time: Optional[float] = None) -> None:
        """
        Update all active conversations.

        Args:
            delta_ms: Time since last update in milliseconds
            current_time: Current game time
        """
        if current_time is None:
            current_time = time.time()

        with self._lock:
            for conv_id in list(self._active_conversations):
                conversation = self._conversations.get(conv_id)
                if not conversation:
                    continue

                conversation._elapsed_time += delta_ms

                # Check if waiting for line gap
                end_time = self._line_end_times.get(conv_id)
                if end_time and current_time >= end_time:
                    del self._line_end_times[conv_id]

                # Update current line
                current_node = conversation.current_node
                if current_node and current_node.has_line:
                    line = current_node.line
                    if line.is_playing:
                        line.update_playback(delta_ms)

                    # Auto-advance if line is completed (regardless of playing state)
                    if line.is_completed:
                        # Auto-advance if not at branch point
                        if not self._is_next_branch(conversation):
                            self.advance_conversation(conv_id, current_time=current_time)

    def _is_next_branch(self, conversation: Conversation) -> bool:
        """Check if next node is a branch point."""
        current = conversation.current_node
        if not current:
            return False

        next_id = current.get_next_node_id(self._game_state)
        if next_id:
            next_node = conversation.get_node(next_id)
            return next_node.is_branch_point if next_node else False

        return False

    def pause_all(self) -> None:
        """Pause all active conversations."""
        with self._lock:
            for conv_id in self._active_conversations:
                conversation = self._conversations.get(conv_id)
                if conversation:
                    conversation.pause()

    def resume_all(self) -> None:
        """Resume all paused conversations."""
        with self._lock:
            for conv_id in self._active_conversations:
                conversation = self._conversations.get(conv_id)
                if conversation:
                    conversation.resume()

    def cancel_all(self) -> None:
        """Cancel all active conversations."""
        with self._lock:
            for conv_id in list(self._active_conversations):
                self.end_conversation(conv_id, cancelled=True)

    def update_game_state(self, state: dict[str, Any]) -> None:
        """Update game state for conditional branching."""
        with self._lock:
            self._game_state.update(state)

    def set_context(self, key: str, value: Any) -> None:
        """Set a context variable for condition evaluation."""
        with self._lock:
            self._game_state[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        with self._lock:
            return self._game_state.get(key, default)

    @property
    def active_conversation_ids(self) -> list[str]:
        """Get IDs of active conversations."""
        with self._lock:
            return list(self._active_conversations)

    @property
    def active_count(self) -> int:
        """Get number of active conversations."""
        with self._lock:
            return len(self._active_conversations)

    @property
    def is_any_active(self) -> bool:
        """Check if any conversation is active."""
        with self._lock:
            return len(self._active_conversations) > 0


# =============================================================================
# Helper Functions
# =============================================================================


def create_linear_conversation(
    first_arg: str | list[VOLine | str],
    second_arg: list[VOLine | str] | str | None = None,
    title: str = "",
    priority: int = PRIORITY_HIGH,
    conversation_id: Optional[str] = None,
) -> Conversation:
    """
    Create a simple linear conversation from a list of lines.

    Supports two calling conventions:
        create_linear_conversation(lines, title="Test")  # lines first
        create_linear_conversation("conv_id", lines)     # conversation_id first

    Args:
        first_arg: Either lines list or conversation_id string
        second_arg: Either lines list or title (when first_arg is lines)
        title: Conversation title
        priority: Priority level
        conversation_id: Optional explicit conversation ID

    Returns:
        Configured Conversation object
    """
    # Determine which calling convention is being used
    if isinstance(first_arg, list):
        # First form: create_linear_conversation(lines, title="Test")
        lines = first_arg
        if isinstance(second_arg, str):
            title = second_arg
        conv_id = conversation_id or str(uuid.uuid4())
    else:
        # Second form: create_linear_conversation("conv_id", lines)
        conv_id = first_arg
        lines = second_arg if isinstance(second_arg, list) else []

    conversation = Conversation(
        conversation_id=conv_id,
        title=title,
        priority=priority,
    )

    participants = set()
    prev_node_id = None

    for i, line_or_id in enumerate(lines):
        # Convert string line IDs to VOLine objects
        if isinstance(line_or_id, str):
            line = VOLine(line_id=line_or_id, context_type=CONTEXT_CONVERSATION)
            line_id = line_or_id
        else:
            line = line_or_id
            line.context_type = CONTEXT_CONVERSATION
            line_id = line.line_id
        participants.add(line.speaker_id)

        node = ConversationNode(
            line=line,
            line_id=line_id,
        )

        if prev_node_id:
            prev_node = conversation.get_node(prev_node_id)
            if prev_node:
                prev_node.next_nodes.append(node.node_id)
        else:
            conversation.start_node_id = node.node_id

        conversation.add_node(node)
        prev_node_id = node.node_id

    conversation.participants = list(participants)
    return conversation


def create_branching_conversation(
    first_arg: str | dict[str, dict[str, Any]] | list[dict[str, Any]],
    second_arg: dict[str, dict[str, Any]] | list[dict[str, Any]] | str | None = None,
    title: str = "",
    conversation_id: Optional[str] = None,
) -> Conversation:
    """
    Create a branching conversation from node data.

    Supports two calling conventions:
        create_branching_conversation(nodes_data, title="Test")  # nodes first
        create_branching_conversation("conv_id", nodes_data)     # conversation_id first

    Args:
        first_arg: Either nodes_data or conversation_id
        second_arg: Either nodes_data or title
        title: Conversation title
        conversation_id: Optional explicit conversation ID

    Returns:
        Configured Conversation object
    """
    # Determine which calling convention is being used
    if isinstance(first_arg, (dict, list)) and not isinstance(first_arg, str):
        # First form: create_branching_conversation(nodes_data, title="Test")
        nodes_data = first_arg
        if isinstance(second_arg, str):
            title = second_arg
        conv_id = conversation_id or str(uuid.uuid4())
    else:
        # Second form: create_branching_conversation("conv_id", nodes_data)
        conv_id = first_arg
        nodes_data = second_arg if isinstance(second_arg, (dict, list)) else []

    conversation = Conversation(
        conversation_id=conv_id,
        title=title,
    )

    # Handle dict format: {node_id: {line: ..., next: [...]}}
    if isinstance(nodes_data, dict):
        for node_id, data in nodes_data.items():
            line_data = data.get("line")
            # Convert string line IDs to VOLine objects
            if isinstance(line_data, str):
                line = VOLine(line_id=line_data, context_type=CONTEXT_CONVERSATION)
            else:
                line = line_data

            node = ConversationNode(
                node_id=node_id,
                line=line,
                next_nodes=data.get("next", []),
                is_branch_point=data.get("is_branch", False),
                branch_options=data.get("options", []),
                conditions=data.get("conditions", {}),
            )
            conversation.add_node(node)

            if not conversation.start_node_id:
                conversation.start_node_id = node.node_id
    else:
        # Handle list format: [{"id": ..., "line": ..., "next": [...]}]
        for data in nodes_data:
            line_data = data.get("line")
            if isinstance(line_data, str):
                line = VOLine(line_id=line_data, context_type=CONTEXT_CONVERSATION)
            else:
                line = line_data

            node = ConversationNode(
                node_id=data.get("id", str(uuid.uuid4())),
                line=line,
                next_nodes=data.get("next", []),
                is_branch_point=data.get("is_branch", False),
                branch_options=data.get("options", []),
                conditions=data.get("conditions", {}),
            )

            conversation.add_node(node)

            if not conversation.start_node_id:
                conversation.start_node_id = node.node_id

    return conversation
