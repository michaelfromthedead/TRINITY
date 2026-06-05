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
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    WAITING = "waiting"  # Waiting for player input
    ENDING = "ending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

    # Aliases (same value as primary state)
    IDLE = "inactive"      # Alias for INACTIVE
    PLAYING = "active"     # Alias for ACTIVE
    COMPLETE = "completed" # Alias for COMPLETED


@dataclass
class ConversationNode:
    """
    A node in a conversation representing one line or branching point.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    line: Optional[VOLine] = None
    line_id: Optional[str] = None  # Alternative to line object
    speaker: Optional[str] = None  # Speaker ID/name
    condition: Optional[str] = None  # Condition expression
    delay_ms: float = 0.0  # Delay before next node
    next_nodes: list[str] = field(default_factory=list)  # Node IDs
    conditions: dict[str, Any] = field(default_factory=dict)
    is_branch_point: bool = False
    branch_options: list[dict[str, Any]] = field(default_factory=list)
    on_enter: Optional[Callable[[ConversationNode], None]] = None
    on_exit: Optional[Callable[[ConversationNode], None]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_line(self) -> bool:
        """Check if this node has a VO line."""
        return self.line is not None or self.line_id is not None

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (no next nodes)."""
        return len(self.next_nodes) == 0 and not self.is_branch_point

    def evaluate_condition(self, context: dict[str, Any]) -> bool:
        """
        Evaluate the node's condition against a context.

        Args:
            context: Dictionary of variable names to values

        Returns:
            True if condition passes (or no condition), False otherwise
        """
        if not self.condition:
            return True

        try:
            # Parse simple conditions like "health > 50"
            # Support basic comparisons
            condition = self.condition.strip()

            # Try to evaluate as a simple expression
            # Replace variable names with context values
            for var_name, var_value in context.items():
                condition = condition.replace(var_name, repr(var_value))

            # Safely evaluate the condition
            return bool(eval(condition, {"__builtins__": {}}, {}))
        except Exception:
            return True  # Default to True on parse errors

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
        )

    @property
    def progress(self) -> float:
        """Get conversation progress (0-1)."""
        if not self.nodes:
            return 0.0
        return len(self._played_nodes) / len(self.nodes)

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

    def start(self, current_time: float) -> Optional[ConversationNode]:
        """
        Start the conversation.

        Returns:
            First node to play
        """
        if not self.start_node_id or self.start_node_id not in self.nodes:
            return None

        self._state = ConversationState.STARTING
        self._start_time = current_time
        self._current_node_id = self.start_node_id
        self._played_nodes = []
        self._state = ConversationState.ACTIVE

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

            return next_node
        else:
            # No more nodes - conversation complete
            self._state = ConversationState.COMPLETED
            return None

    def pause(self) -> None:
        """Pause the conversation."""
        if self._state in (ConversationState.ACTIVE, ConversationState.PLAYING):
            self._state = ConversationState.PAUSED

    def resume(self) -> None:
        """Resume the conversation."""
        if self._state == ConversationState.PAUSED:
            self._state = ConversationState.ACTIVE

    def cancel(self) -> None:
        """Cancel the conversation."""
        self._state = ConversationState.CANCELLED

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
        self._state = ConversationState.INACTIVE
        self._current_node_id = ""
        self._start_time = 0.0
        self._elapsed_time = 0.0
        self._played_nodes = []

    @property
    def is_paused(self) -> bool:
        """Check if conversation is paused."""
        return self._state == ConversationState.PAUSED

    @property
    def remaining_nodes_count(self) -> int:
        """Get count of remaining nodes to play."""
        return len(self.nodes) - len(self._played_nodes)

    # Callbacks
    on_node_start: Optional[Callable[[str], None]] = None
    on_complete: Optional[Callable[[str], None]] = None

    def start(self, current_time: Optional[float] = None) -> Optional[ConversationNode]:
        """
        Start the conversation.

        Args:
            current_time: Optional current time (defaults to time.time())

        Returns:
            First node to play
        """
        if current_time is None:
            current_time = time.time()

        if not self.start_node_id or self.start_node_id not in self.nodes:
            # If no start node, use first added node
            if self.nodes:
                self.start_node_id = next(iter(self.nodes.keys()))
            else:
                return None

        self._state = ConversationState.ACTIVE
        self._start_time = current_time
        self._current_node_id = self.start_node_id
        self._played_nodes = []

        # Call on_node_start callback
        if self.on_node_start:
            self.on_node_start(self._current_node_id)

        return self.current_node

    def complete_current_node(self) -> Optional[ConversationNode]:
        """
        Complete the current node and advance to the next.

        Returns:
            Next node or None if conversation is complete
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

        # Get next node
        next_id = current.get_next_node_id()

        if next_id and next_id in self.nodes:
            self._current_node_id = next_id
            next_node = self.current_node

            # Call on_enter callback
            if next_node and next_node.on_enter:
                next_node.on_enter(next_node)

            # Call on_node_start callback
            if self.on_node_start:
                self.on_node_start(self._current_node_id)

            return next_node
        else:
            # No more nodes - conversation complete
            self._state = ConversationState.COMPLETED
            if self.on_complete:
                self.on_complete(self.conversation_id)
            return None

    def select_branch(self, branch_id: str) -> Optional[ConversationNode]:
        """
        Select a branch at a branching point.

        Args:
            branch_id: The node ID of the branch to select

        Returns:
            The selected node

        Raises:
            ValueError: If branch_id is not a valid next node
        """
        current = self.current_node
        if not current:
            raise ValueError("No current node")

        if branch_id not in current.next_nodes and branch_id not in self.nodes:
            raise ValueError(f"Invalid branch: {branch_id}")

        if branch_id not in self.nodes:
            raise KeyError(f"Node {branch_id} does not exist")

        self._current_node_id = branch_id

        # Call callbacks
        next_node = self.current_node
        if next_node and next_node.on_enter:
            next_node.on_enter(next_node)

        if self.on_node_start:
            self.on_node_start(self._current_node_id)

        return next_node

    def go_to_node(self, node_id: str) -> None:
        """
        Go to a specific node (only allowed for branching, not linear).

        Raises:
            ValueError: If trying to go back in a linear conversation
        """
        if node_id in self._played_nodes:
            raise ValueError("Cannot go back to already played node in linear conversation")
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        self._current_node_id = node_id

    def skip_current(self) -> Optional[ConversationNode]:
        """
        Skip the current node and advance to the next.

        Returns:
            Next node or None if conversation is complete
        """
        return self.complete_current_node()

    def skip_all(self) -> None:
        """Skip all remaining nodes and complete the conversation."""
        # Mark all nodes as played
        self._played_nodes = list(self.nodes.keys())
        self._state = ConversationState.COMPLETED
        if self.on_complete:
            self.on_complete(self.conversation_id)

    def save_state(self) -> dict[str, Any]:
        """
        Save the current conversation state for later restoration.

        Returns:
            Dictionary containing state information
        """
        return {
            "conversation_id": self.conversation_id,
            "current_node": self._current_node_id,
            "current_node_id": self._current_node_id,
            "state": self._state.value,
            "played_nodes": list(self._played_nodes),
            "elapsed_time": self._elapsed_time,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """
        Restore conversation state from a saved state.

        Args:
            state: State dictionary from save_state()
        """
        self._current_node_id = state.get("current_node") or state.get("current_node_id", "")
        state_value = state.get("state", "inactive")
        # Map state value to enum
        for s in ConversationState:
            if s.value == state_value:
                self._state = s
                break
        self._played_nodes = list(state.get("played_nodes", []))
        self._elapsed_time = state.get("elapsed_time", 0.0)

    def set_context(self, key: str, value: Any) -> None:
        """
        Set a context variable.

        Args:
            key: Variable name
            value: Variable value
        """
        self.metadata[f"context_{key}"] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get a context variable.

        Args:
            key: Variable name
            default: Default value if not found

        Returns:
            The variable value or default
        """
        return self.metadata.get(f"context_{key}", default)

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

    # Alias for test API
    def register(self, conversation: Conversation) -> None:
        """Register a conversation (alias for register_conversation)."""
        self.register_conversation(conversation)

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

    # Alias for test API
    def get(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID (alias for get_conversation)."""
        return self.get_conversation(conversation_id)

    def is_active(self, conversation_id: str) -> bool:
        """Check if a conversation is currently active."""
        with self._lock:
            return conversation_id in self._active_conversations

    def get_active(self) -> Optional[Conversation]:
        """Get the currently active conversation (first one if multiple)."""
        with self._lock:
            if self._active_conversations:
                return self._conversations.get(self._active_conversations[0])
            return None

    def stop(self, conversation_id: str) -> bool:
        """Stop a conversation (alias for end_conversation with cancelled=True)."""
        return self.end_conversation(conversation_id, cancelled=True)

    def start(self, conversation_id: str, current_time: Optional[float] = None) -> Optional[ConversationNode]:
        """Start a conversation (alias for start_conversation)."""
        return self.start_conversation(conversation_id, current_time)

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

                    # Check for auto-advance after update or if already completed
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

    def set_context(self, key: str, value: Any) -> None:
        """
        Set a context variable in the game state.

        Args:
            key: Variable name
            value: Variable value
        """
        with self._lock:
            self._game_state[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get a context variable from the game state.

        Args:
            key: Variable name
            default: Default value if not found

        Returns:
            The variable value or default
        """
        with self._lock:
            return self._game_state.get(key, default)


# =============================================================================
# Helper Functions
# =============================================================================


def create_linear_conversation(
    conversation_id_or_lines: Any,
    lines_or_title: Any = None,
    title: str = "",
    priority: int = PRIORITY_HIGH,
) -> Conversation:
    """
    Create a simple linear conversation from a list of lines.

    Can be called as:
        create_linear_conversation(lines)
        create_linear_conversation(lines, conversation_id=..., title=..., priority=...)
        create_linear_conversation(conversation_id, lines)  # Test API

    Args:
        conversation_id_or_lines: Either conversation ID (str) or list of lines/line_ids
        lines_or_title: Either lines list or title (for backward compat)
        title: Conversation title
        priority: Priority level

    Returns:
        Configured Conversation object
    """
    # Parse arguments - support both signatures
    if isinstance(conversation_id_or_lines, str) and lines_or_title is not None:
        # New test API: create_linear_conversation("conv_id", ["line_1", "line_2"])
        conversation_id = conversation_id_or_lines
        lines = lines_or_title
    elif isinstance(conversation_id_or_lines, list):
        # Original API: create_linear_conversation(lines, ...)
        lines = conversation_id_or_lines
        conversation_id = lines_or_title if isinstance(lines_or_title, str) else str(uuid.uuid4())
    else:
        # Fallback
        lines = []
        conversation_id = str(uuid.uuid4())

    conversation = Conversation(
        conversation_id=conversation_id,
        title=title,
        priority=priority,
    )

    participants = set()
    prev_node_id = None

    for i, line_item in enumerate(lines):
        # Support both VOLine objects and line_id strings
        if isinstance(line_item, str):
            # It's a line ID string
            node = ConversationNode(
                line_id=line_item,
            )
        else:
            # It's a VOLine object
            line_item.context_type = CONTEXT_CONVERSATION
            participants.add(line_item.speaker_id)
            node = ConversationNode(
                line=line_item,
                line_id=getattr(line_item, 'line_id', None),
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
    conversation_id_or_data: Any,
    structure_or_title: Any = None,
    title: str = "",
) -> Conversation:
    """
    Create a branching conversation from node data.

    Can be called as:
        create_branching_conversation(nodes_data)
        create_branching_conversation(conversation_id, structure_dict)  # Test API

    Args:
        conversation_id_or_data: Either conversation ID (str) or list/dict of node data
        structure_or_title: Either structure dict or title
        title: Conversation title

    Returns:
        Configured Conversation object
    """
    # Parse arguments - support both signatures
    if isinstance(conversation_id_or_data, str) and structure_or_title is not None:
        # New test API: create_branching_conversation("conv_id", {structure_dict})
        conversation_id = conversation_id_or_data
        nodes_data = structure_or_title
    elif isinstance(conversation_id_or_data, (list, dict)):
        # Original API
        nodes_data = conversation_id_or_data
        conversation_id = structure_or_title if isinstance(structure_or_title, str) else str(uuid.uuid4())
    else:
        nodes_data = {}
        conversation_id = str(uuid.uuid4())

    conversation = Conversation(
        conversation_id=conversation_id,
        title=title,
    )

    # Handle dict format: {"node_id": {"line": ..., "next": [...]}, ...}
    if isinstance(nodes_data, dict):
        for node_id, data in nodes_data.items():
            node = ConversationNode(
                node_id=node_id,
                line=data.get("line") if isinstance(data.get("line"), VOLine) else None,
                line_id=data.get("line") if isinstance(data.get("line"), str) else None,
                next_nodes=data.get("next", []),
                is_branch_point=len(data.get("next", [])) > 1,
                branch_options=data.get("options", []),
                conditions=data.get("conditions", {}),
            )

            conversation.add_node(node)

            if not conversation.start_node_id:
                conversation.start_node_id = node_id
    else:
        # Handle list format
        for data in nodes_data:
            node = ConversationNode(
                node_id=data.get("id", str(uuid.uuid4())),
                line=data.get("line"),
                next_nodes=data.get("next", []),
                is_branch_point=data.get("is_branch", False),
                branch_options=data.get("options", []),
                conditions=data.get("conditions", {}),
            )

            conversation.add_node(node)

            if not conversation.start_node_id:
                conversation.start_node_id = node.node_id

    return conversation
