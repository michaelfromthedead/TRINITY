"""
Whitebox tests for Conversation module.

Tests ConversationNode, Conversation state machine, ConversationManager,
branching, and helper functions.
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch

from engine.audio.dialogue.conversation import (
    ConversationNode,
    Conversation,
    ConversationState,
    ConversationManager,
    create_linear_conversation,
    create_branching_conversation,
)
from engine.audio.dialogue.vo_line import VOLine, VOLineState
from engine.audio.dialogue.config import (
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
    MAX_ACTIVE_CONVERSATIONS,
    CONVERSATION_GAP_MS,
)


# =============================================================================
# ConversationState Tests
# =============================================================================


class TestConversationState:
    """Tests for ConversationState enum."""

    def test_all_states_exist(self):
        """Test all required states are defined."""
        assert ConversationState.INACTIVE.value == "inactive"
        assert ConversationState.STARTING.value == "starting"
        assert ConversationState.ACTIVE.value == "active"
        assert ConversationState.PAUSED.value == "paused"
        assert ConversationState.WAITING.value == "waiting"
        assert ConversationState.ENDING.value == "ending"
        assert ConversationState.COMPLETED.value == "completed"
        assert ConversationState.CANCELLED.value == "cancelled"


# =============================================================================
# ConversationNode Tests
# =============================================================================


class TestConversationNode:
    """Tests for ConversationNode dataclass."""

    def test_initialization(self):
        """Test ConversationNode initializes correctly."""
        node = ConversationNode()

        assert node.node_id is not None
        assert node.line is None
        assert node.next_nodes == []
        assert node.conditions == {}
        assert node.is_branch_point is False
        assert node.branch_options == []

    def test_custom_initialization(self):
        """Test ConversationNode with custom values."""
        line = VOLine(text="Hello")
        node = ConversationNode(
            node_id="node_1",
            line=line,
            next_nodes=["node_2", "node_3"],
            is_branch_point=True,
            branch_options=[{"text": "Option 1"}],
        )

        assert node.node_id == "node_1"
        assert node.line is line
        assert len(node.next_nodes) == 2
        assert node.is_branch_point is True

    def test_has_line_true(self):
        """Test has_line returns True when line exists."""
        node = ConversationNode(line=VOLine())
        assert node.has_line is True

    def test_has_line_false(self):
        """Test has_line returns False when no line."""
        node = ConversationNode()
        assert node.has_line is False

    def test_is_terminal_true(self):
        """Test is_terminal returns True for terminal node."""
        node = ConversationNode(next_nodes=[], is_branch_point=False)
        assert node.is_terminal is True

    def test_is_terminal_false_has_next(self):
        """Test is_terminal returns False when has next nodes."""
        node = ConversationNode(next_nodes=["next"])
        assert node.is_terminal is False

    def test_is_terminal_false_branch_point(self):
        """Test is_terminal returns False for branch point."""
        node = ConversationNode(is_branch_point=True)
        assert node.is_terminal is False

    def test_get_next_node_id_empty(self):
        """Test get_next_node_id returns None for empty."""
        node = ConversationNode()
        assert node.get_next_node_id() is None

    def test_get_next_node_id_single(self):
        """Test get_next_node_id returns single next node."""
        node = ConversationNode(next_nodes=["node_2"])
        assert node.get_next_node_id() == "node_2"

    def test_get_next_node_id_multiple(self):
        """Test get_next_node_id returns first for multiple."""
        node = ConversationNode(next_nodes=["node_2", "node_3"])
        assert node.get_next_node_id() == "node_2"

    def test_callbacks(self):
        """Test on_enter and on_exit callbacks."""
        on_enter = MagicMock()
        on_exit = MagicMock()
        node = ConversationNode(on_enter=on_enter, on_exit=on_exit)

        node.on_enter(node)
        node.on_exit(node)

        on_enter.assert_called_once_with(node)
        on_exit.assert_called_once_with(node)


# =============================================================================
# Conversation Basic Tests
# =============================================================================


class TestConversationBasic:
    """Basic tests for Conversation dataclass."""

    def test_initialization(self):
        """Test Conversation initializes correctly."""
        conv = Conversation()

        assert conv.conversation_id is not None
        assert conv.title == ""
        assert conv.participants == []
        assert conv.nodes == {}
        assert conv.start_node_id == ""
        assert conv.state == ConversationState.INACTIVE

    def test_custom_initialization(self):
        """Test Conversation with custom values."""
        conv = Conversation(
            conversation_id="conv_1",
            title="Test Conversation",
            participants=["npc1", "npc2"],
            priority=PRIORITY_HIGH,
            interruptible=False,
            skippable=False,
            tags={"main_story"},
        )

        assert conv.conversation_id == "conv_1"
        assert conv.title == "Test Conversation"
        assert len(conv.participants) == 2
        assert conv.priority == PRIORITY_HIGH
        assert conv.interruptible is False
        assert conv.skippable is False

    def test_post_init_tags_list_to_set(self):
        """Test __post_init__ converts tags list to set."""
        conv = Conversation(tags=["tag1", "tag2"])
        assert isinstance(conv.tags, set)

    def test_state_property(self):
        """Test state property getter and setter."""
        conv = Conversation()
        conv.state = ConversationState.ACTIVE

        assert conv.state == ConversationState.ACTIVE


# =============================================================================
# Conversation Node Management Tests
# =============================================================================


class TestConversationNodeManagement:
    """Tests for Conversation node management."""

    def test_add_node(self):
        """Test add_node adds node to conversation."""
        conv = Conversation()
        node = ConversationNode(node_id="node_1")

        conv.add_node(node)

        assert "node_1" in conv.nodes
        assert conv.nodes["node_1"] is node

    def test_remove_node(self):
        """Test remove_node removes node."""
        conv = Conversation()
        node = ConversationNode(node_id="node_1")
        conv.add_node(node)

        result = conv.remove_node("node_1")

        assert result is True
        assert "node_1" not in conv.nodes

    def test_remove_node_not_found(self):
        """Test remove_node returns False for missing node."""
        conv = Conversation()

        result = conv.remove_node("missing")

        assert result is False

    def test_get_node(self):
        """Test get_node retrieves node."""
        conv = Conversation()
        node = ConversationNode(node_id="node_1")
        conv.add_node(node)

        result = conv.get_node("node_1")

        assert result is node

    def test_get_node_not_found(self):
        """Test get_node returns None for missing node."""
        conv = Conversation()

        result = conv.get_node("missing")

        assert result is None


# =============================================================================
# Conversation State Property Tests
# =============================================================================


class TestConversationStateProperties:
    """Tests for Conversation state properties."""

    def test_current_node_none(self):
        """Test current_node returns None when not set."""
        conv = Conversation()
        assert conv.current_node is None

    def test_current_node_valid(self):
        """Test current_node returns current node."""
        conv = Conversation()
        node = ConversationNode(node_id="node_1")
        conv.add_node(node)
        conv._current_node_id = "node_1"

        assert conv.current_node is node

    def test_current_line_none(self):
        """Test current_line returns None when no current node."""
        conv = Conversation()
        assert conv.current_line is None

    def test_current_line_valid(self):
        """Test current_line returns line from current node."""
        conv = Conversation()
        line = VOLine(text="Hello")
        node = ConversationNode(node_id="node_1", line=line)
        conv.add_node(node)
        conv._current_node_id = "node_1"

        assert conv.current_line is line

    def test_is_active_true(self):
        """Test is_active returns True for active states."""
        conv = Conversation()

        for state in [ConversationState.ACTIVE, ConversationState.WAITING, ConversationState.STARTING]:
            conv.state = state
            assert conv.is_active is True

    def test_is_active_false(self):
        """Test is_active returns False for inactive states."""
        conv = Conversation()

        for state in [ConversationState.INACTIVE, ConversationState.COMPLETED]:
            conv.state = state
            assert conv.is_active is False

    def test_is_complete_true(self):
        """Test is_complete returns True for completed states."""
        conv = Conversation()

        for state in [ConversationState.COMPLETED, ConversationState.CANCELLED]:
            conv.state = state
            assert conv.is_complete is True

    def test_is_complete_false(self):
        """Test is_complete returns False for incomplete states."""
        conv = Conversation()
        conv.state = ConversationState.ACTIVE
        assert conv.is_complete is False

    def test_progress_empty(self):
        """Test progress returns 0 for empty conversation."""
        conv = Conversation()
        assert conv.progress == 0.0

    def test_progress_calculation(self):
        """Test progress calculation."""
        conv = Conversation()
        for i in range(4):
            conv.add_node(ConversationNode(node_id=f"node_{i}"))
        conv._played_nodes = ["node_0", "node_1"]

        assert conv.progress == 0.5


# =============================================================================
# Conversation Flow Tests
# =============================================================================


class TestConversationFlow:
    """Tests for Conversation flow control."""

    def test_start(self):
        """Test start initiates conversation."""
        conv = Conversation()
        node = ConversationNode(node_id="start")
        conv.add_node(node)
        conv.start_node_id = "start"

        result = conv.start(100.0)

        assert result is node
        assert conv.state == ConversationState.ACTIVE
        assert conv._start_time == 100.0

    def test_start_no_start_node(self):
        """Test start returns None without start node."""
        conv = Conversation()

        result = conv.start(100.0)

        assert result is None

    def test_start_invalid_start_node(self):
        """Test start returns None for invalid start node."""
        conv = Conversation(start_node_id="missing")

        result = conv.start(100.0)

        assert result is None

    def test_advance_linear(self):
        """Test advance moves to next node."""
        conv = Conversation()
        node1 = ConversationNode(node_id="node_1", next_nodes=["node_2"])
        node2 = ConversationNode(node_id="node_2")
        conv.add_node(node1)
        conv.add_node(node2)
        conv.start_node_id = "node_1"
        conv.start(100.0)

        result = conv.advance()

        assert result is node2
        assert "node_1" in conv._played_nodes

    def test_advance_terminal(self):
        """Test advance completes at terminal node."""
        conv = Conversation()
        node = ConversationNode(node_id="node_1", next_nodes=[])
        conv.add_node(node)
        conv.start_node_id = "node_1"
        conv.start(100.0)

        result = conv.advance()

        assert result is None
        assert conv.state == ConversationState.COMPLETED

    def test_advance_branch_with_choice(self):
        """Test advance with branch choice."""
        conv = Conversation()
        branch = ConversationNode(
            node_id="branch",
            is_branch_point=True,
            branch_options=[
                {"next_node_id": "option_a"},
                {"next_node_id": "option_b"},
            ],
            next_nodes=["option_a", "option_b"],
        )
        option_a = ConversationNode(node_id="option_a")
        option_b = ConversationNode(node_id="option_b")

        conv.add_node(branch)
        conv.add_node(option_a)
        conv.add_node(option_b)
        conv.start_node_id = "branch"
        conv.start(100.0)

        result = conv.advance(choice_index=1)

        assert result is option_b

    def test_advance_calls_callbacks(self):
        """Test advance calls node callbacks."""
        on_exit = MagicMock()
        on_enter = MagicMock()

        conv = Conversation()
        node1 = ConversationNode(node_id="node_1", next_nodes=["node_2"], on_exit=on_exit)
        node2 = ConversationNode(node_id="node_2", on_enter=on_enter)
        conv.add_node(node1)
        conv.add_node(node2)
        conv.start_node_id = "node_1"
        conv.start(100.0)

        conv.advance()

        on_exit.assert_called_once()
        on_enter.assert_called_once()

    def test_pause(self):
        """Test pause pauses active conversation."""
        conv = Conversation()
        conv.state = ConversationState.ACTIVE

        conv.pause()

        assert conv.state == ConversationState.PAUSED

    def test_pause_not_active(self):
        """Test pause does nothing when not active."""
        conv = Conversation()
        conv.state = ConversationState.INACTIVE

        conv.pause()

        assert conv.state == ConversationState.INACTIVE

    def test_resume(self):
        """Test resume resumes paused conversation."""
        conv = Conversation()
        conv.state = ConversationState.PAUSED

        conv.resume()

        assert conv.state == ConversationState.ACTIVE

    def test_resume_not_paused(self):
        """Test resume does nothing when not paused."""
        conv = Conversation()
        conv.state = ConversationState.ACTIVE

        conv.resume()

        assert conv.state == ConversationState.ACTIVE

    def test_cancel(self):
        """Test cancel cancels conversation."""
        conv = Conversation()
        conv.state = ConversationState.ACTIVE

        conv.cancel()

        assert conv.state == ConversationState.CANCELLED

    def test_skip_to_node(self):
        """Test skip_to_node jumps to specified node."""
        conv = Conversation()
        node1 = ConversationNode(node_id="node_1")
        node2 = ConversationNode(node_id="node_2")
        conv.add_node(node1)
        conv.add_node(node2)
        conv.start_node_id = "node_1"
        conv.start(100.0)

        result = conv.skip_to_node("node_2")

        assert result is node2
        assert conv._current_node_id == "node_2"

    def test_skip_to_node_invalid(self):
        """Test skip_to_node returns None for invalid node."""
        conv = Conversation()
        conv.state = ConversationState.ACTIVE

        result = conv.skip_to_node("missing")

        assert result is None

    def test_wait_for_input(self):
        """Test wait_for_input sets waiting state."""
        conv = Conversation()

        conv.wait_for_input()

        assert conv.state == ConversationState.WAITING

    def test_reset(self):
        """Test reset resets conversation to initial state."""
        conv = Conversation()
        conv.state = ConversationState.ACTIVE
        conv._current_node_id = "some_node"
        conv._played_nodes = ["node_1", "node_2"]
        conv._elapsed_time = 5000.0

        conv.reset()

        assert conv.state == ConversationState.INACTIVE
        assert conv._current_node_id == ""
        assert conv._played_nodes == []
        assert conv._elapsed_time == 0.0


# =============================================================================
# Conversation Serialization Tests
# =============================================================================


class TestConversationSerialization:
    """Tests for Conversation serialization."""

    def test_to_dict(self):
        """Test to_dict returns serializable dictionary."""
        conv = Conversation(
            conversation_id="conv_1",
            title="Test",
            participants=["npc1"],
            start_node_id="start",
            priority=PRIORITY_HIGH,
            tags={"main"},
        )

        data = conv.to_dict()

        assert data["conversation_id"] == "conv_1"
        assert data["title"] == "Test"
        assert data["participants"] == ["npc1"]
        assert data["start_node_id"] == "start"
        assert data["priority"] == PRIORITY_HIGH
        assert "main" in data["tags"]


# =============================================================================
# ConversationManager Basic Tests
# =============================================================================


class TestConversationManagerBasic:
    """Basic tests for ConversationManager."""

    def test_initialization(self):
        """Test ConversationManager initializes correctly."""
        manager = ConversationManager()

        assert manager.active_count == 0
        assert manager.is_any_active is False
        assert manager.active_conversation_ids == []

    def test_custom_initialization(self):
        """Test ConversationManager with custom parameters."""
        callback = MagicMock()
        manager = ConversationManager(
            max_active=2,
            gap_ms=500.0,
            on_conversation_started=callback,
        )

        assert manager._max_active == 2
        assert manager._gap_ms == 500.0

    def test_register_conversation(self):
        """Test register_conversation adds conversation."""
        manager = ConversationManager()
        conv = Conversation(conversation_id="conv_1")

        manager.register_conversation(conv)

        assert manager.get_conversation("conv_1") is conv

    def test_unregister_conversation(self):
        """Test unregister_conversation removes conversation."""
        manager = ConversationManager()
        conv = Conversation(conversation_id="conv_1")
        manager.register_conversation(conv)

        result = manager.unregister_conversation("conv_1")

        assert result is True
        assert manager.get_conversation("conv_1") is None

    def test_unregister_conversation_not_found(self):
        """Test unregister_conversation returns False for missing."""
        manager = ConversationManager()

        result = manager.unregister_conversation("missing")

        assert result is False


# =============================================================================
# ConversationManager Start/End Tests
# =============================================================================


class TestConversationManagerStartEnd:
    """Tests for ConversationManager start/end operations."""

    def test_start_conversation(self):
        """Test start_conversation starts registered conversation."""
        callback = MagicMock()
        manager = ConversationManager(on_conversation_started=callback)

        conv = Conversation(conversation_id="conv_1")
        node = ConversationNode(node_id="start")
        conv.add_node(node)
        conv.start_node_id = "start"

        manager.register_conversation(conv)
        result = manager.start_conversation("conv_1")

        assert result is node
        assert manager.active_count == 1
        callback.assert_called_once_with(conv)

    def test_start_conversation_not_registered(self):
        """Test start_conversation returns None for unregistered."""
        manager = ConversationManager()

        result = manager.start_conversation("missing")

        assert result is None

    def test_start_conversation_at_max(self):
        """Test start_conversation fails at max active."""
        manager = ConversationManager(max_active=1)

        # Start first conversation
        conv1 = Conversation(conversation_id="conv_1")
        node1 = ConversationNode(node_id="start1")
        conv1.add_node(node1)
        conv1.start_node_id = "start1"
        manager.register_conversation(conv1)
        manager.start_conversation("conv_1")

        # Try to start second
        conv2 = Conversation(conversation_id="conv_2", priority=PRIORITY_NORMAL)
        node2 = ConversationNode(node_id="start2")
        conv2.add_node(node2)
        conv2.start_node_id = "start2"
        manager.register_conversation(conv2)

        result = manager.start_conversation("conv_2")

        assert result is None

    def test_start_conversation_interrupts_lower_priority(self):
        """Test start_conversation can interrupt lower priority."""
        manager = ConversationManager(max_active=1)

        # Start low priority conversation
        conv1 = Conversation(conversation_id="conv_1", priority=PRIORITY_NORMAL, interruptible=True)
        node1 = ConversationNode(node_id="start1")
        conv1.add_node(node1)
        conv1.start_node_id = "start1"
        manager.register_conversation(conv1)
        manager.start_conversation("conv_1")

        # Start high priority conversation
        conv2 = Conversation(conversation_id="conv_2", priority=PRIORITY_HIGH)
        node2 = ConversationNode(node_id="start2")
        conv2.add_node(node2)
        conv2.start_node_id = "start2"
        manager.register_conversation(conv2)

        result = manager.start_conversation("conv_2")

        assert result is not None
        assert conv1.state == ConversationState.CANCELLED

    def test_end_conversation(self):
        """Test end_conversation ends active conversation."""
        callback = MagicMock()
        manager = ConversationManager(on_conversation_ended=callback)

        conv = Conversation(conversation_id="conv_1")
        node = ConversationNode(node_id="start")
        conv.add_node(node)
        conv.start_node_id = "start"

        manager.register_conversation(conv)
        manager.start_conversation("conv_1")

        result = manager.end_conversation("conv_1")

        assert result is True
        assert manager.active_count == 0
        assert conv.state == ConversationState.COMPLETED
        callback.assert_called_with(conv, False)

    def test_end_conversation_cancelled(self):
        """Test end_conversation with cancelled flag."""
        manager = ConversationManager()

        conv = Conversation(conversation_id="conv_1")
        node = ConversationNode(node_id="start")
        conv.add_node(node)
        conv.start_node_id = "start"

        manager.register_conversation(conv)
        manager.start_conversation("conv_1")

        manager.end_conversation("conv_1", cancelled=True)

        assert conv.state == ConversationState.CANCELLED


# =============================================================================
# ConversationManager Advance Tests
# =============================================================================


class TestConversationManagerAdvance:
    """Tests for ConversationManager advance operations."""

    def test_advance_conversation(self):
        """Test advance_conversation moves to next node."""
        callback = MagicMock()
        manager = ConversationManager(on_line_ended=callback)

        conv = Conversation(conversation_id="conv_1")
        line = VOLine(text="Hello")
        node1 = ConversationNode(node_id="node_1", line=line, next_nodes=["node_2"])
        node2 = ConversationNode(node_id="node_2")
        conv.add_node(node1)
        conv.add_node(node2)
        conv.start_node_id = "node_1"

        manager.register_conversation(conv)
        manager.start_conversation("conv_1")

        result = manager.advance_conversation("conv_1")

        assert result is node2
        callback.assert_called_once()

    def test_advance_conversation_ends_at_terminal(self):
        """Test advance_conversation ends at terminal node."""
        manager = ConversationManager()

        conv = Conversation(conversation_id="conv_1")
        node = ConversationNode(node_id="node_1", next_nodes=[])
        conv.add_node(node)
        conv.start_node_id = "node_1"

        manager.register_conversation(conv)
        manager.start_conversation("conv_1")

        result = manager.advance_conversation("conv_1")

        assert result is None
        assert manager.active_count == 0

    def test_skip_line(self):
        """Test skip_line skips current line."""
        manager = ConversationManager()

        conv = Conversation(conversation_id="conv_1", skippable=True)
        line = VOLine(text="Hello", interruptible=True)
        line.state = VOLineState.PLAYING
        node = ConversationNode(node_id="node_1", line=line)
        conv.add_node(node)
        conv.start_node_id = "node_1"

        manager.register_conversation(conv)
        manager.start_conversation("conv_1")

        result = manager.skip_line("conv_1")

        assert result is True
        assert line.state == VOLineState.INTERRUPTED

    def test_skip_line_not_skippable(self):
        """Test skip_line fails for non-skippable conversation."""
        manager = ConversationManager()

        conv = Conversation(conversation_id="conv_1", skippable=False)
        node = ConversationNode(node_id="node_1", line=VOLine())
        conv.add_node(node)
        conv.start_node_id = "node_1"

        manager.register_conversation(conv)
        manager.start_conversation("conv_1")

        result = manager.skip_line("conv_1")

        assert result is False


# =============================================================================
# ConversationManager Choice Tests
# =============================================================================


class TestConversationManagerChoice:
    """Tests for ConversationManager choice handling."""

    def test_make_choice(self):
        """Test make_choice selects branch option."""
        manager = ConversationManager()

        conv = Conversation(conversation_id="conv_1")
        branch = ConversationNode(
            node_id="branch",
            is_branch_point=True,
            branch_options=[{"next_node_id": "a"}, {"next_node_id": "b"}],
            next_nodes=["a", "b"],
        )
        option_a = ConversationNode(node_id="a")
        option_b = ConversationNode(node_id="b")

        conv.add_node(branch)
        conv.add_node(option_a)
        conv.add_node(option_b)
        conv.start_node_id = "branch"

        manager.register_conversation(conv)
        manager.start_conversation("conv_1")
        conv.wait_for_input()

        result = manager.make_choice("conv_1", 1)

        assert result is option_b

    def test_make_choice_not_waiting(self):
        """Test make_choice fails when not waiting."""
        manager = ConversationManager()

        conv = Conversation(conversation_id="conv_1")
        node = ConversationNode(node_id="node")
        conv.add_node(node)
        conv.start_node_id = "node"

        manager.register_conversation(conv)
        manager.start_conversation("conv_1")
        # Not calling wait_for_input()

        result = manager.make_choice("conv_1", 0)

        assert result is None


# =============================================================================
# ConversationManager Update Tests
# =============================================================================


class TestConversationManagerUpdate:
    """Tests for ConversationManager update functionality."""

    def test_update_advances_elapsed_time(self):
        """Test update advances elapsed time."""
        manager = ConversationManager()

        conv = Conversation(conversation_id="conv_1")
        node = ConversationNode(node_id="node", line=VOLine(duration_ms=1000.0))
        conv.add_node(node)
        conv.start_node_id = "node"

        manager.register_conversation(conv)
        manager.start_conversation("conv_1")

        manager.update(100.0)

        assert conv._elapsed_time == 100.0

    def test_update_auto_advances(self):
        """Test update auto-advances completed lines."""
        manager = ConversationManager()

        conv = Conversation(conversation_id="conv_1")
        line = VOLine(duration_ms=100.0)
        line.state = VOLineState.PLAYING
        line.playback_position_ms = 100.0  # At end

        node1 = ConversationNode(node_id="node_1", line=line, next_nodes=["node_2"])
        node2 = ConversationNode(node_id="node_2")
        conv.add_node(node1)
        conv.add_node(node2)
        conv.start_node_id = "node_1"

        manager.register_conversation(conv)
        manager.start_conversation("conv_1")

        # Complete the line
        line.complete_playback()

        manager.update(10.0)

        assert conv._current_node_id == "node_2"


# =============================================================================
# ConversationManager Bulk Operations Tests
# =============================================================================


class TestConversationManagerBulkOperations:
    """Tests for ConversationManager bulk operations."""

    def test_pause_all(self):
        """Test pause_all pauses all active conversations."""
        manager = ConversationManager(max_active=2)

        for i in range(2):
            conv = Conversation(conversation_id=f"conv_{i}")
            node = ConversationNode(node_id=f"node_{i}")
            conv.add_node(node)
            conv.start_node_id = f"node_{i}"
            manager.register_conversation(conv)
            manager.start_conversation(f"conv_{i}")

        manager.pause_all()

        for conv_id in manager.active_conversation_ids:
            conv = manager.get_conversation(conv_id)
            assert conv.state == ConversationState.PAUSED

    def test_resume_all(self):
        """Test resume_all resumes all paused conversations."""
        manager = ConversationManager(max_active=2)

        for i in range(2):
            conv = Conversation(conversation_id=f"conv_{i}")
            node = ConversationNode(node_id=f"node_{i}")
            conv.add_node(node)
            conv.start_node_id = f"node_{i}"
            manager.register_conversation(conv)
            manager.start_conversation(f"conv_{i}")

        manager.pause_all()
        manager.resume_all()

        for conv_id in manager.active_conversation_ids:
            conv = manager.get_conversation(conv_id)
            assert conv.state == ConversationState.ACTIVE

    def test_cancel_all(self):
        """Test cancel_all cancels all active conversations."""
        manager = ConversationManager(max_active=2)

        for i in range(2):
            conv = Conversation(conversation_id=f"conv_{i}")
            node = ConversationNode(node_id=f"node_{i}")
            conv.add_node(node)
            conv.start_node_id = f"node_{i}"
            manager.register_conversation(conv)
            manager.start_conversation(f"conv_{i}")

        manager.cancel_all()

        assert manager.active_count == 0

    def test_update_game_state(self):
        """Test update_game_state updates game state dict."""
        manager = ConversationManager()

        manager.update_game_state({"level": 5, "quest": "main"})

        assert manager._game_state["level"] == 5
        assert manager._game_state["quest"] == "main"


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestConversationHelperFunctions:
    """Tests for conversation helper functions."""

    def test_create_linear_conversation(self):
        """Test create_linear_conversation creates linear flow."""
        lines = [
            VOLine(text="Line 1", speaker_id="npc1"),
            VOLine(text="Line 2", speaker_id="npc2"),
            VOLine(text="Line 3", speaker_id="npc1"),
        ]

        conv = create_linear_conversation(lines, title="Test")

        assert conv.title == "Test"
        assert len(conv.nodes) == 3
        assert conv.start_node_id is not None

        # Verify linear linking
        start_node = conv.get_node(conv.start_node_id)
        assert start_node is not None
        assert len(start_node.next_nodes) == 1

    def test_create_linear_conversation_participants(self):
        """Test create_linear_conversation extracts participants."""
        lines = [
            VOLine(speaker_id="npc1"),
            VOLine(speaker_id="npc2"),
            VOLine(speaker_id="npc1"),
        ]

        conv = create_linear_conversation(lines)

        assert "npc1" in conv.participants
        assert "npc2" in conv.participants

    def test_create_branching_conversation(self):
        """Test create_branching_conversation creates branching flow."""
        nodes_data = [
            {"id": "start", "line": VOLine(text="Start"), "next": ["branch"]},
            {
                "id": "branch",
                "line": None,
                "is_branch": True,
                "options": [
                    {"text": "Option A", "next_node_id": "a"},
                    {"text": "Option B", "next_node_id": "b"},
                ],
                "next": ["a", "b"],
            },
            {"id": "a", "line": VOLine(text="A"), "next": []},
            {"id": "b", "line": VOLine(text="B"), "next": []},
        ]

        conv = create_branching_conversation(nodes_data, title="Branching")

        assert conv.title == "Branching"
        assert len(conv.nodes) == 4
        assert conv.start_node_id == "start"

        branch_node = conv.get_node("branch")
        assert branch_node.is_branch_point is True
        assert len(branch_node.branch_options) == 2


# =============================================================================
# ConversationManager Thread Safety Tests
# =============================================================================


class TestConversationManagerThreadSafety:
    """Thread safety tests for ConversationManager."""

    def test_concurrent_start_end(self):
        """Test concurrent start/end operations."""
        manager = ConversationManager(max_active=10)

        # Register conversations
        for i in range(20):
            conv = Conversation(conversation_id=f"conv_{i}")
            node = ConversationNode(node_id=f"node_{i}")
            conv.add_node(node)
            conv.start_node_id = f"node_{i}"
            manager.register_conversation(conv)

        def start_conversations():
            for i in range(10):
                manager.start_conversation(f"conv_{i}")
                time.sleep(0.001)

        def end_conversations():
            for i in range(10):
                manager.end_conversation(f"conv_{i}")
                time.sleep(0.001)

        t1 = threading.Thread(target=start_conversations)
        t2 = threading.Thread(target=end_conversations)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Should complete without deadlock

    def test_concurrent_update(self):
        """Test concurrent update operations."""
        manager = ConversationManager(max_active=5)

        # Start some conversations
        for i in range(5):
            conv = Conversation(conversation_id=f"conv_{i}")
            node = ConversationNode(node_id=f"node_{i}", line=VOLine(duration_ms=1000.0))
            conv.add_node(node)
            conv.start_node_id = f"node_{i}"
            manager.register_conversation(conv)
            manager.start_conversation(f"conv_{i}")

        def update_loop():
            for _ in range(50):
                manager.update(10.0)
                time.sleep(0.001)

        threads = [threading.Thread(target=update_loop) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without deadlock
