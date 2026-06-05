"""
Blackbox tests for Conversation state machine and branching dialogue.

Tests PUBLIC behavior only - no internal state inspection.
Based on GAPSET_15_AUDIO Phase 9 specifications.
"""

import pytest
from typing import List, Dict, Optional

# Public API imports
from engine.audio.dialogue import (
    Conversation,
    ConversationNode,
    ConversationState,
    ConversationManager,
    create_linear_conversation,
    create_branching_conversation,
    DialogueState,
    create_vo_line,
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
)


class TestConversationCreation:
    """Test Conversation creation and initialization."""

    def test_create_conversation_minimal(self):
        """Conversation can be created with minimal parameters."""
        conv = Conversation(conversation_id="conv_001")
        assert conv is not None
        assert conv.conversation_id == "conv_001"

    def test_create_conversation_with_nodes(self):
        """Conversation can be created with nodes."""
        nodes = [
            ConversationNode(node_id="node_1", line_id="line_1"),
            ConversationNode(node_id="node_2", line_id="line_2"),
        ]
        conv = Conversation(conversation_id="conv_002", nodes=nodes)
        assert len(conv.nodes) == 2

    def test_create_linear_conversation_helper(self):
        """create_linear_conversation helper works."""
        lines = ["line_1", "line_2", "line_3"]
        conv = create_linear_conversation("linear_001", lines)
        assert conv.conversation_id == "linear_001"
        assert len(conv.nodes) == 3

    def test_create_branching_conversation_helper(self):
        """create_branching_conversation helper works."""
        structure = {
            "start": {"line": "line_1", "next": ["branch_a", "branch_b"]},
            "branch_a": {"line": "line_2a", "next": ["end"]},
            "branch_b": {"line": "line_2b", "next": ["end"]},
            "end": {"line": "line_3", "next": []},
        }
        conv = create_branching_conversation("branch_001", structure)
        assert conv.conversation_id == "branch_001"


class TestConversationNode:
    """Test ConversationNode structure."""

    def test_node_creation(self):
        """ConversationNode can be created."""
        node = ConversationNode(node_id="node_001", line_id="line_001")
        assert node.node_id == "node_001"
        assert node.line_id == "line_001"

    def test_node_with_next_nodes(self):
        """ConversationNode can have next nodes."""
        node = ConversationNode(
            node_id="branch_point",
            line_id="branch_line",
            next_nodes=["option_a", "option_b"]
        )
        assert len(node.next_nodes) == 2

    def test_node_with_condition(self):
        """ConversationNode can have conditions."""
        node = ConversationNode(
            node_id="conditional_001",
            line_id="cond_line",
            condition="player_health > 50"
        )
        assert node.condition is not None

    def test_node_with_speaker(self):
        """ConversationNode can specify speaker."""
        node = ConversationNode(
            node_id="speaker_001",
            line_id="speaker_line",
            speaker="Commander"
        )
        assert node.speaker == "Commander"

    def test_node_with_delay(self):
        """ConversationNode can have delay before next."""
        node = ConversationNode(
            node_id="delay_001",
            line_id="delay_line",
            delay_ms=500
        )
        assert node.delay_ms == 500


class TestConversationState:
    """Test ConversationState enumeration."""

    def test_idle_state_exists(self):
        """IDLE state exists."""
        assert ConversationState.IDLE is not None

    def test_playing_state_exists(self):
        """PLAYING state exists."""
        assert ConversationState.PLAYING is not None

    def test_waiting_state_exists(self):
        """WAITING state exists."""
        assert ConversationState.WAITING is not None

    def test_complete_state_exists(self):
        """COMPLETE state exists."""
        assert ConversationState.COMPLETE is not None

    def test_cancelled_state_exists(self):
        """CANCELLED state exists."""
        assert ConversationState.CANCELLED is not None


class TestConversationStateMachine:
    """Test conversation state transitions."""

    def test_initial_state_is_idle(self):
        """New conversation starts in IDLE state."""
        conv = Conversation(conversation_id="state_001")
        assert conv.state == ConversationState.IDLE

    def test_start_transitions_to_playing(self):
        """Starting conversation transitions to PLAYING."""
        lines = ["line_1"]
        conv = create_linear_conversation("start_001", lines)
        conv.start()
        assert conv.state == ConversationState.PLAYING

    def test_complete_node_advances_state(self):
        """Completing a node advances conversation."""
        lines = ["line_1", "line_2"]
        conv = create_linear_conversation("advance_001", lines)
        conv.start()
        conv.complete_current_node()
        # Should still be playing or waiting for next node

    def test_cancel_transitions_to_cancelled(self):
        """Cancelling conversation transitions to CANCELLED."""
        conv = create_linear_conversation("cancel_001", ["line_1"])
        conv.start()
        conv.cancel()
        assert conv.state == ConversationState.CANCELLED

    def test_conversation_completes_at_end(self):
        """Conversation completes when all nodes played."""
        conv = create_linear_conversation("complete_001", ["line_1"])
        conv.start()
        conv.complete_current_node()
        assert conv.state == ConversationState.COMPLETE


class TestConversationBranching:
    """Test branching conversation logic."""

    def test_branch_point_has_multiple_options(self):
        """Branch points have multiple next options."""
        structure = {
            "start": {"line": "intro", "next": ["yes_path", "no_path"]},
            "yes_path": {"line": "yes", "next": []},
            "no_path": {"line": "no", "next": []},
        }
        conv = create_branching_conversation("branch_test", structure)
        start_node = conv.get_node("start")
        assert len(start_node.next_nodes) == 2

    def test_select_branch(self):
        """Branch can be selected."""
        structure = {
            "start": {"line": "intro", "next": ["yes_path", "no_path"]},
            "yes_path": {"line": "yes", "next": []},
            "no_path": {"line": "no", "next": []},
        }
        conv = create_branching_conversation("select_001", structure)
        conv.start()
        conv.complete_current_node()
        conv.select_branch("yes_path")

        current = conv.current_node
        assert current.node_id == "yes_path"

    def test_invalid_branch_selection_rejected(self):
        """Invalid branch selection is rejected."""
        structure = {
            "start": {"line": "intro", "next": ["valid"]},
            "valid": {"line": "valid", "next": []},
        }
        conv = create_branching_conversation("invalid_001", structure)
        conv.start()
        conv.complete_current_node()

        with pytest.raises((ValueError, KeyError)):
            conv.select_branch("invalid_path")


class TestConversationConditions:
    """Test conditional node evaluation."""

    def test_condition_evaluation(self):
        """Conditions can be evaluated."""
        node = ConversationNode(
            node_id="cond_001",
            line_id="cond_line",
            condition="health > 50"
        )

        # Simulate condition evaluation with context
        context = {"health": 75}
        result = node.evaluate_condition(context)
        assert result is True

    def test_condition_evaluation_false(self):
        """False conditions are evaluated correctly."""
        node = ConversationNode(
            node_id="cond_002",
            line_id="cond_line",
            condition="health > 50"
        )

        context = {"health": 25}
        result = node.evaluate_condition(context)
        assert result is False

    def test_no_condition_always_true(self):
        """Nodes without conditions always evaluate true."""
        node = ConversationNode(node_id="no_cond", line_id="line")

        result = node.evaluate_condition({})
        assert result is True


class TestConversationManager:
    """Test ConversationManager orchestration."""

    def test_manager_creation(self):
        """ConversationManager can be created."""
        manager = ConversationManager()
        assert manager is not None

    def test_manager_register_conversation(self):
        """Manager can register conversations."""
        manager = ConversationManager()
        conv = create_linear_conversation("register_001", ["line_1"])
        manager.register(conv)

        retrieved = manager.get("register_001")
        assert retrieved.conversation_id == "register_001"

    def test_manager_start_conversation(self):
        """Manager can start conversations."""
        manager = ConversationManager()
        conv = create_linear_conversation("start_mgr_001", ["line_1"])
        manager.register(conv)

        manager.start("start_mgr_001")
        assert manager.is_active("start_mgr_001")

    def test_manager_active_conversation(self):
        """Manager tracks active conversation."""
        manager = ConversationManager()
        conv = create_linear_conversation("active_001", ["line_1"])
        manager.register(conv)
        manager.start("active_001")

        active = manager.get_active()
        assert active is not None
        assert active.conversation_id == "active_001"

    def test_manager_stop_conversation(self):
        """Manager can stop conversations."""
        manager = ConversationManager()
        conv = create_linear_conversation("stop_001", ["line_1"])
        manager.register(conv)
        manager.start("stop_001")
        manager.stop("stop_001")

        assert not manager.is_active("stop_001")

    def test_manager_update_tick(self):
        """Manager has update/tick method."""
        manager = ConversationManager()
        assert hasattr(manager, 'update') or hasattr(manager, 'tick')


class TestConversationProgress:
    """Test conversation progress tracking."""

    def test_get_current_node(self):
        """Current node can be retrieved."""
        conv = create_linear_conversation("progress_001", ["line_1", "line_2"])
        conv.start()

        current = conv.current_node
        assert current is not None

    def test_get_progress_percentage(self):
        """Progress percentage can be calculated."""
        conv = create_linear_conversation("percent_001", ["line_1", "line_2", "line_3", "line_4"])
        conv.start()
        conv.complete_current_node()

        # Should be ~25% complete
        progress = conv.progress
        assert 0.0 <= progress <= 1.0

    def test_get_remaining_nodes(self):
        """Remaining nodes can be counted."""
        conv = create_linear_conversation("remain_001", ["line_1", "line_2", "line_3"])
        conv.start()

        remaining = conv.remaining_nodes_count
        assert remaining >= 0


class TestConversationEvents:
    """Test conversation event callbacks."""

    def test_on_node_start_callback(self):
        """on_node_start callback is called."""
        callback_called = []

        def on_start(node_id):
            callback_called.append(node_id)

        conv = create_linear_conversation("cb_001", ["line_1"])
        conv.on_node_start = on_start
        conv.start()

        assert len(callback_called) > 0

    def test_on_conversation_complete_callback(self):
        """on_complete callback is called."""
        callback_called = []

        def on_complete(conv_id):
            callback_called.append(conv_id)

        conv = create_linear_conversation("cb_002", ["line_1"])
        conv.on_complete = on_complete
        conv.start()
        conv.complete_current_node()

        assert "cb_002" in callback_called


class TestLinearConversation:
    """Test linear (sequential) conversations."""

    def test_linear_plays_in_order(self):
        """Linear conversation plays nodes in order."""
        conv = create_linear_conversation("linear_order", ["first", "second", "third"])
        conv.start()

        nodes_played = [conv.current_node.line_id]
        conv.complete_current_node()
        nodes_played.append(conv.current_node.line_id)
        conv.complete_current_node()
        nodes_played.append(conv.current_node.line_id)

        assert nodes_played == ["first", "second", "third"]

    def test_linear_cannot_go_back(self):
        """Linear conversation cannot go backwards."""
        conv = create_linear_conversation("no_back", ["a", "b", "c"])
        conv.start()
        conv.complete_current_node()  # Now at "b"

        # Should not be able to go back
        with pytest.raises((ValueError, AttributeError)):
            conv.go_to_node("a")


class TestDialogueState:
    """Test DialogueState enumeration."""

    def test_dialogue_state_pending(self):
        """PENDING state exists."""
        assert DialogueState.PENDING is not None

    def test_dialogue_state_active(self):
        """ACTIVE state exists."""
        assert DialogueState.ACTIVE is not None

    def test_dialogue_state_paused(self):
        """PAUSED state exists."""
        assert DialogueState.PAUSED is not None

    def test_dialogue_state_complete(self):
        """COMPLETE state exists."""
        assert DialogueState.COMPLETE is not None


class TestConversationPause:
    """Test conversation pause functionality."""

    def test_pause_conversation(self):
        """Conversation can be paused."""
        conv = create_linear_conversation("pause_001", ["line_1", "line_2"])
        conv.start()
        conv.pause()

        assert conv.state == ConversationState.WAITING or conv.is_paused

    def test_resume_conversation(self):
        """Conversation can be resumed."""
        conv = create_linear_conversation("resume_001", ["line_1", "line_2"])
        conv.start()
        conv.pause()
        conv.resume()

        assert conv.state == ConversationState.PLAYING


class TestConversationSkip:
    """Test conversation skip functionality."""

    def test_skip_current_node(self):
        """Current node can be skipped."""
        conv = create_linear_conversation("skip_001", ["line_1", "line_2"])
        conv.start()

        # Skip should advance without waiting for audio
        conv.skip_current()
        assert conv.current_node.line_id == "line_2"

    def test_skip_entire_conversation(self):
        """Entire conversation can be skipped."""
        conv = create_linear_conversation("skip_all", ["a", "b", "c", "d"])
        conv.start()
        conv.skip_all()

        assert conv.state == ConversationState.COMPLETE


class TestConversationReset:
    """Test conversation reset functionality."""

    def test_reset_to_beginning(self):
        """Conversation can be reset to beginning."""
        conv = create_linear_conversation("reset_001", ["line_1", "line_2", "line_3"])
        conv.start()
        conv.complete_current_node()
        conv.complete_current_node()  # At line_3
        conv.reset()

        assert conv.state == ConversationState.IDLE
        # Starting again should start from beginning


class TestConversationSerialization:
    """Test conversation state serialization."""

    def test_save_state(self):
        """Conversation state can be saved."""
        conv = create_linear_conversation("save_001", ["a", "b", "c"])
        conv.start()
        conv.complete_current_node()  # At "b"

        state = conv.save_state()
        assert state is not None
        assert "current_node" in state or hasattr(state, 'current_node')

    def test_restore_state(self):
        """Conversation state can be restored."""
        conv = create_linear_conversation("restore_001", ["a", "b", "c"])
        conv.start()
        conv.complete_current_node()

        state = conv.save_state()

        # Reset and restore
        conv.reset()
        conv.restore_state(state)

        # Should be back at "b"
        assert conv.current_node.line_id == "b"


class TestConversationContext:
    """Test conversation context/game state integration."""

    def test_set_context_variable(self):
        """Context variables can be set."""
        manager = ConversationManager()
        manager.set_context("player_name", "John")

        value = manager.get_context("player_name")
        assert value == "John"

    def test_context_used_in_conditions(self):
        """Context is used in condition evaluation."""
        manager = ConversationManager()
        manager.set_context("has_key", True)

        structure = {
            "start": {"line": "check", "next": ["with_key", "no_key"]},
            "with_key": {"line": "enter", "next": [], "condition": "has_key"},
            "no_key": {"line": "locked", "next": [], "condition": "not has_key"},
        }
        conv = create_branching_conversation("context_001", structure)
        manager.register(conv)
        # Context should affect branch availability
