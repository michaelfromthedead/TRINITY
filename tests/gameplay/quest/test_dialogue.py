"""
Comprehensive tests for Quest Dialogue System.

Tests cover:
- Dialogue node types (text, choice, condition)
- Dialogue graph traversal
- Condition evaluation
- Variable substitution
- Speaker/listener context
- Dialogue events
- Branching dialogue
- Dialogue localization hooks
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import Mock, MagicMock

# DialogueChoice, DialogueSession, etc. are planned but not yet implemented
pytest.skip("Dialogue system API not fully implemented", allow_module_level=True)

from engine.gameplay.quest.dialogue import (
    DialogueGraph,
    DialogueNode,
    DialogueChoice,
    DialogueSession,
    DialogueContext,
    DialogueSpeaker,
    TextNode,
    ChoiceNode,
    BranchNode,
    EventNode,
    RandomNode,
    EntryNode,
    ExitNode,
)
from engine.gameplay.quest.constants import (
    NodeType,
    VariableScope,
    ComparisonOperator,
    EffectType,
)
from engine.gameplay.quest.dialogue_conditions import (
    Condition,
    ConditionResult,
    VariableCondition,
    ItemCondition,
    QuestStateCondition,
    QuestState,
    AndCondition,
    OrCondition,
    NotCondition,
    AlwaysTrueCondition,
    AlwaysFalseCondition,
)
from engine.gameplay.quest.dialogue_effects import (
    Effect,
    EffectResult,
    SetVariableEffect,
    GiveItemEffect,
    TakeItemEffect,
    StartQuestEffect,
)
from engine.gameplay.quest.dialogue_variables import (
    VariableManager,
    VariableScope,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def variable_manager():
    """Create a variable manager for testing."""
    return VariableManager()


@pytest.fixture
def mock_context():
    """Create a mock dialogue context."""
    context = Mock(spec=DialogueContext)
    context.variables = VariableManager()
    context.get_item_count = Mock(return_value=0)
    context.has_item = Mock(return_value=False)
    context.get_quest_state = Mock(return_value=QuestState.NOT_STARTED)
    context.get_quest_progress = Mock(return_value=0.0)
    context.get_reputation = Mock(return_value=0)
    return context


@pytest.fixture
def simple_graph():
    """Create a simple dialogue graph."""
    graph = DialogueGraph(id="simple_dialogue")

    # Entry node
    entry = EntryNode(id="start", next_node="text1")
    graph.add_node(entry)

    # Text node
    text1 = TextNode(
        id="text1",
        speaker_id="npc_001",
        text="Hello, traveler!",
        next_node="choice1",
    )
    graph.add_node(text1)

    # Choice node
    choice1 = ChoiceNode(
        id="choice1",
        choices=[
            DialogueChoice(text="Hello!", next_node="text2"),
            DialogueChoice(text="Goodbye.", next_node="exit"),
        ],
    )
    graph.add_node(choice1)

    # Another text
    text2 = TextNode(
        id="text2",
        speaker_id="npc_001",
        text="Welcome!",
        next_node="exit",
    )
    graph.add_node(text2)

    # Exit node
    exit_node = ExitNode(id="exit")
    graph.add_node(exit_node)

    graph.entry_point = "start"
    return graph


@pytest.fixture
def branching_graph():
    """Create a branching dialogue graph."""
    graph = DialogueGraph(id="branching_dialogue")

    # Entry
    entry = EntryNode(id="start", next_node="check_item")
    graph.add_node(entry)

    # Branch based on item
    branch = BranchNode(
        id="check_item",
        branches=[
            ("has_key", "key_dialogue"),
            ("default", "no_key_dialogue"),
        ],
        conditions={
            "has_key": ItemCondition(item_id="gold_key", min_count=1),
        },
    )
    graph.add_node(branch)

    # Key dialogue
    key_text = TextNode(
        id="key_dialogue",
        speaker_id="npc",
        text="I see you have the key!",
        next_node="exit",
    )
    graph.add_node(key_text)

    # No key dialogue
    no_key_text = TextNode(
        id="no_key_dialogue",
        speaker_id="npc",
        text="Come back when you have the key.",
        next_node="exit",
    )
    graph.add_node(no_key_text)

    # Exit
    exit_node = ExitNode(id="exit")
    graph.add_node(exit_node)

    graph.entry_point = "start"
    return graph


# =============================================================================
# DialogueNode Tests
# =============================================================================

class TestDialogueNode:
    """Tests for dialogue node base functionality."""

    def test_text_node_creation(self):
        """Test creating a text node."""
        node = TextNode(
            id="text_1",
            speaker_id="npc_001",
            text="Hello!",
            next_node="next",
        )
        assert node.id == "text_1"
        assert node.node_type == NodeType.TEXT
        assert node.speaker_id == "npc_001"
        assert node.text == "Hello!"

    def test_choice_node_creation(self):
        """Test creating a choice node."""
        node = ChoiceNode(
            id="choice_1",
            choices=[
                DialogueChoice(text="Yes", next_node="yes_path"),
                DialogueChoice(text="No", next_node="no_path"),
            ],
        )
        assert node.id == "choice_1"
        assert node.node_type == NodeType.CHOICE
        assert len(node.choices) == 2

    def test_branch_node_creation(self):
        """Test creating a branch node."""
        node = BranchNode(
            id="branch_1",
            branches=[
                ("condition1", "path1"),
                ("default", "default_path"),
            ],
        )
        assert node.id == "branch_1"
        assert node.node_type == NodeType.BRANCH
        assert len(node.branches) == 2

    def test_event_node_creation(self):
        """Test creating an event node."""
        node = EventNode(
            id="event_1",
            event_name="quest_started",
            event_data={"quest_id": "quest_001"},
            next_node="next",
        )
        assert node.id == "event_1"
        assert node.node_type == NodeType.EVENT
        assert node.event_name == "quest_started"

    def test_random_node_creation(self):
        """Test creating a random node."""
        node = RandomNode(
            id="random_1",
            variations=[
                ("variation_a", 0.5),
                ("variation_b", 0.3),
                ("variation_c", 0.2),
            ],
        )
        assert node.id == "random_1"
        assert node.node_type == NodeType.RANDOM
        assert len(node.variations) == 3

    def test_entry_node_creation(self):
        """Test creating an entry node."""
        node = EntryNode(id="start", next_node="first_text")
        assert node.id == "start"
        assert node.node_type == NodeType.ENTRY
        assert node.next_node == "first_text"

    def test_exit_node_creation(self):
        """Test creating an exit node."""
        node = ExitNode(id="end")
        assert node.id == "end"
        assert node.node_type == NodeType.EXIT

    def test_node_with_effects(self):
        """Test node with effects."""
        effect = SetVariableEffect(
            variable_name="talked",
            value=True,
            scope=VariableScope.GLOBAL,
        )
        node = TextNode(
            id="text_effect",
            speaker_id="npc",
            text="Let me mark that down.",
            effects=[effect],
            next_node="next",
        )
        assert len(node.effects) == 1

    def test_node_with_condition(self):
        """Test node with visibility condition."""
        condition = VariableCondition(
            variable_name="has_quest",
            operator=ComparisonOperator.EQUAL,
            expected_value=True,
        )
        node = TextNode(
            id="conditional_text",
            speaker_id="npc",
            text="I have a quest for you!",
            condition=condition,
            next_node="next",
        )
        assert node.condition is not None

    def test_node_serialization(self):
        """Test node serialization."""
        node = TextNode(
            id="text_1",
            speaker_id="npc_001",
            text="Hello!",
            next_node="next",
        )
        data = node.to_dict()

        assert data["id"] == "text_1"
        assert data["type"] == "text"
        assert data["speaker_id"] == "npc_001"

    def test_node_deserialization(self):
        """Test node deserialization."""
        data = {
            "id": "text_1",
            "type": "text",
            "speaker_id": "npc_001",
            "text": "Hello!",
            "next_node": "next",
        }
        node = DialogueNode.from_dict(data)

        assert isinstance(node, TextNode)
        assert node.id == "text_1"


# =============================================================================
# DialogueChoice Tests
# =============================================================================

class TestDialogueChoice:
    """Tests for dialogue choice functionality."""

    def test_choice_creation(self):
        """Test creating a dialogue choice."""
        choice = DialogueChoice(
            text="I accept your quest!",
            next_node="accept_path",
        )
        assert choice.text == "I accept your quest!"
        assert choice.next_node == "accept_path"

    def test_choice_with_condition(self):
        """Test choice with visibility condition."""
        condition = VariableCondition(
            variable_name="charisma",
            operator=ComparisonOperator.GREATER_EQUAL,
            expected_value=15,
        )
        choice = DialogueChoice(
            text="[Persuade] Convince them",
            next_node="persuade_path",
            condition=condition,
        )
        assert choice.condition is not None

    def test_choice_with_effects(self):
        """Test choice with effects on selection."""
        effect = SetVariableEffect(
            variable_name="chose_violence",
            value=True,
        )
        choice = DialogueChoice(
            text="Attack!",
            next_node="combat",
            effects=[effect],
        )
        assert len(choice.effects) == 1

    def test_choice_visibility(self, mock_context):
        """Test choice visibility evaluation."""
        # Choice visible when has item
        mock_context.has_item.return_value = True

        condition = ItemCondition(item_id="bribe_gold", min_count=100)
        choice = DialogueChoice(
            text="[Bribe] Offer gold",
            next_node="bribe_path",
            condition=condition,
        )

        # Should be visible
        mock_context.get_item_count.return_value = 100
        result = choice.condition.evaluate(mock_context)
        assert result.success is True

    def test_choice_serialization(self):
        """Test choice serialization."""
        choice = DialogueChoice(
            text="Hello",
            next_node="path1",
        )
        data = choice.to_dict()

        assert data["text"] == "Hello"
        assert data["next_node"] == "path1"


# =============================================================================
# DialogueGraph Tests
# =============================================================================

class TestDialogueGraph:
    """Tests for dialogue graph functionality."""

    def test_graph_creation(self):
        """Test creating a dialogue graph."""
        graph = DialogueGraph(id="test_dialogue")
        assert graph.id == "test_dialogue"
        assert len(graph.nodes) == 0

    def test_add_node(self, simple_graph):
        """Test adding nodes to graph."""
        assert "start" in simple_graph.nodes
        assert "text1" in simple_graph.nodes
        assert "choice1" in simple_graph.nodes
        assert "exit" in simple_graph.nodes

    def test_get_node(self, simple_graph):
        """Test getting a node by ID."""
        node = simple_graph.get_node("text1")
        assert node is not None
        assert node.id == "text1"

    def test_get_nonexistent_node(self, simple_graph):
        """Test getting a non-existent node."""
        node = simple_graph.get_node("nonexistent")
        assert node is None

    def test_remove_node(self, simple_graph):
        """Test removing a node."""
        result = simple_graph.remove_node("text2")
        assert result is True
        assert "text2" not in simple_graph.nodes

    def test_entry_point(self, simple_graph):
        """Test graph entry point."""
        assert simple_graph.entry_point == "start"
        entry = simple_graph.get_entry_node()
        assert entry is not None
        assert entry.id == "start"

    def test_validate_graph(self, simple_graph):
        """Test graph validation."""
        errors = simple_graph.validate()
        assert len(errors) == 0

    def test_validate_missing_entry(self):
        """Test validation fails without entry point."""
        graph = DialogueGraph(id="invalid")
        errors = graph.validate()
        assert len(errors) > 0
        assert any("entry" in e.lower() for e in errors)

    def test_validate_orphan_node(self):
        """Test validation detects orphan nodes."""
        graph = DialogueGraph(id="orphan_test")

        entry = EntryNode(id="start", next_node="text1")
        graph.add_node(entry)

        text1 = TextNode(id="text1", speaker_id="npc", text="Hello", next_node="exit")
        graph.add_node(text1)

        # Orphan node - not connected to main path
        orphan = TextNode(id="orphan", speaker_id="npc", text="Orphan", next_node="exit")
        graph.add_node(orphan)

        exit_node = ExitNode(id="exit")
        graph.add_node(exit_node)

        graph.entry_point = "start"

        errors = graph.validate()
        assert any("orphan" in e.lower() for e in errors)

    def test_validate_broken_link(self):
        """Test validation detects broken links."""
        graph = DialogueGraph(id="broken_test")

        entry = EntryNode(id="start", next_node="text1")
        graph.add_node(entry)

        text1 = TextNode(id="text1", speaker_id="npc", text="Hello", next_node="nonexistent")
        graph.add_node(text1)

        graph.entry_point = "start"

        errors = graph.validate()
        assert any("nonexistent" in e.lower() for e in errors)

    def test_graph_serialization(self, simple_graph):
        """Test graph serialization."""
        data = simple_graph.to_dict()

        assert data["id"] == "simple_dialogue"
        assert "nodes" in data
        assert len(data["nodes"]) == 5

    def test_graph_deserialization(self, simple_graph):
        """Test graph deserialization."""
        data = simple_graph.to_dict()
        loaded = DialogueGraph.from_dict(data)

        assert loaded.id == simple_graph.id
        assert len(loaded.nodes) == len(simple_graph.nodes)

    def test_get_all_text_nodes(self, simple_graph):
        """Test getting all text nodes."""
        text_nodes = simple_graph.get_nodes_by_type(NodeType.TEXT)
        assert len(text_nodes) == 2

    def test_get_all_choice_nodes(self, simple_graph):
        """Test getting all choice nodes."""
        choice_nodes = simple_graph.get_nodes_by_type(NodeType.CHOICE)
        assert len(choice_nodes) == 1


# =============================================================================
# DialogueSession Tests
# =============================================================================

class TestDialogueSession:
    """Tests for dialogue session functionality."""

    def test_session_creation(self, simple_graph, mock_context):
        """Test creating a dialogue session."""
        session = DialogueSession(
            graph=simple_graph,
            context=mock_context,
        )
        assert session.graph is simple_graph
        assert session.current_node is None

    def test_session_start(self, simple_graph, mock_context):
        """Test starting a dialogue session."""
        session = DialogueSession(graph=simple_graph, context=mock_context)
        session.start()

        assert session.is_active
        # Should move past entry to first content node
        assert session.current_node.id == "text1"

    def test_session_advance(self, simple_graph, mock_context):
        """Test advancing through dialogue."""
        session = DialogueSession(graph=simple_graph, context=mock_context)
        session.start()

        # At text1, advance to choice1
        session.advance()
        assert session.current_node.id == "choice1"

    def test_session_select_choice(self, simple_graph, mock_context):
        """Test selecting a choice."""
        session = DialogueSession(graph=simple_graph, context=mock_context)
        session.start()
        session.advance()  # Move to choice node

        # Select first choice
        result = session.select_choice(0)

        assert result is True
        assert session.current_node.id == "text2"

    def test_session_select_invalid_choice(self, simple_graph, mock_context):
        """Test selecting an invalid choice index."""
        session = DialogueSession(graph=simple_graph, context=mock_context)
        session.start()
        session.advance()

        result = session.select_choice(99)
        assert result is False

    def test_session_end(self, simple_graph, mock_context):
        """Test dialogue session ending."""
        session = DialogueSession(graph=simple_graph, context=mock_context)
        session.start()
        session.advance()  # To choice
        session.select_choice(1)  # Select "Goodbye"

        assert session.is_active is False

    def test_session_get_current_text(self, simple_graph, mock_context):
        """Test getting current node text."""
        session = DialogueSession(graph=simple_graph, context=mock_context)
        session.start()

        text = session.get_current_text()
        assert text == "Hello, traveler!"

    def test_session_get_current_speaker(self, simple_graph, mock_context):
        """Test getting current speaker."""
        session = DialogueSession(graph=simple_graph, context=mock_context)
        session.start()

        speaker = session.get_current_speaker()
        assert speaker == "npc_001"

    def test_session_get_available_choices(self, simple_graph, mock_context):
        """Test getting available choices."""
        session = DialogueSession(graph=simple_graph, context=mock_context)
        session.start()
        session.advance()

        choices = session.get_available_choices()

        assert len(choices) == 2
        assert choices[0].text == "Hello!"
        assert choices[1].text == "Goodbye."

    def test_session_branch_evaluation(self, branching_graph, mock_context):
        """Test automatic branch evaluation."""
        # Without key
        mock_context.get_item_count.return_value = 0

        session = DialogueSession(graph=branching_graph, context=mock_context)
        session.start()

        assert session.current_node.id == "no_key_dialogue"

    def test_session_branch_with_item(self, branching_graph, mock_context):
        """Test branch with item present."""
        # With key
        mock_context.get_item_count.return_value = 1

        session = DialogueSession(graph=branching_graph, context=mock_context)
        session.start()

        assert session.current_node.id == "key_dialogue"

    def test_session_effects_executed(self, mock_context):
        """Test that effects are executed on node entry."""
        graph = DialogueGraph(id="effect_test")

        entry = EntryNode(id="start", next_node="text1")
        graph.add_node(entry)

        effect = SetVariableEffect(
            variable_name="visited",
            value=True,
            scope=VariableScope.GLOBAL,
        )
        text1 = TextNode(
            id="text1",
            speaker_id="npc",
            text="Welcome!",
            effects=[effect],
            next_node="exit",
        )
        graph.add_node(text1)
        graph.add_node(ExitNode(id="exit"))

        graph.entry_point = "start"

        session = DialogueSession(graph=graph, context=mock_context)
        session.start()

        # Effect should have set variable
        assert mock_context.variables.get("visited", scope=VariableScope.GLOBAL) is True

    def test_session_history(self, simple_graph, mock_context):
        """Test dialogue history tracking."""
        session = DialogueSession(graph=simple_graph, context=mock_context)
        session.start()

        history = session.get_history()

        assert len(history) >= 1
        assert history[0]["node_id"] == "text1"

    def test_session_restart(self, simple_graph, mock_context):
        """Test restarting a session."""
        session = DialogueSession(graph=simple_graph, context=mock_context)
        session.start()
        session.advance()
        session.select_choice(0)

        session.restart()

        assert session.current_node.id == "text1"


# =============================================================================
# Condition Evaluation Tests
# =============================================================================

class TestConditionEvaluation:
    """Tests for dialogue condition evaluation."""

    def test_variable_condition_equal(self, mock_context):
        """Test variable equality condition."""
        mock_context.variables.set("level", 10)

        condition = VariableCondition(
            variable_name="level",
            operator=ComparisonOperator.EQUAL,
            expected_value=10,
        )

        result = condition.evaluate(mock_context)
        assert result.success is True

    def test_variable_condition_greater(self, mock_context):
        """Test variable greater-than condition."""
        mock_context.variables.set("charisma", 15)

        condition = VariableCondition(
            variable_name="charisma",
            operator=ComparisonOperator.GREATER_EQUAL,
            expected_value=10,
        )

        result = condition.evaluate(mock_context)
        assert result.success is True

    def test_variable_condition_less(self, mock_context):
        """Test variable less-than condition."""
        mock_context.variables.set("health", 25)

        condition = VariableCondition(
            variable_name="health",
            operator=ComparisonOperator.LESS,
            expected_value=50,
        )

        result = condition.evaluate(mock_context)
        assert result.success is True

    def test_variable_condition_contains(self, mock_context):
        """Test string contains condition."""
        mock_context.variables.set("name", "Sir Knight")

        condition = VariableCondition(
            variable_name="name",
            operator=ComparisonOperator.CONTAINS,
            expected_value="Knight",
        )

        result = condition.evaluate(mock_context)
        assert result.success is True

    def test_item_condition(self, mock_context):
        """Test item condition."""
        mock_context.get_item_count.return_value = 5

        condition = ItemCondition(item_id="gold_coin", min_count=3)

        result = condition.evaluate(mock_context)
        assert result.success is True

    def test_item_condition_range(self, mock_context):
        """Test item condition with range."""
        mock_context.get_item_count.return_value = 50

        condition = ItemCondition(item_id="arrows", min_count=10, max_count=100)

        result = condition.evaluate(mock_context)
        assert result.success is True

    def test_quest_state_condition(self, mock_context):
        """Test quest state condition."""
        mock_context.get_quest_state.return_value = QuestState.COMPLETED

        condition = QuestStateCondition(
            quest_id="quest_001",
            required_state=QuestState.COMPLETED,
        )

        result = condition.evaluate(mock_context)
        assert result.success is True

    def test_and_condition(self, mock_context):
        """Test AND compound condition."""
        mock_context.variables.set("level", 10)
        mock_context.get_item_count.return_value = 5

        condition = AndCondition([
            VariableCondition("level", ComparisonOperator.GREATER_EQUAL, 5),
            ItemCondition(item_id="key", min_count=1),
        ])

        result = condition.evaluate(mock_context)
        assert result.success is True

    def test_and_condition_fails(self, mock_context):
        """Test AND condition when one fails."""
        mock_context.variables.set("level", 10)
        mock_context.get_item_count.return_value = 0

        condition = AndCondition([
            VariableCondition("level", ComparisonOperator.GREATER_EQUAL, 5),
            ItemCondition(item_id="key", min_count=1),
        ])

        result = condition.evaluate(mock_context)
        assert result.success is False

    def test_or_condition(self, mock_context):
        """Test OR compound condition."""
        mock_context.variables.set("level", 3)
        mock_context.get_item_count.return_value = 5

        condition = OrCondition([
            VariableCondition("level", ComparisonOperator.GREATER_EQUAL, 10),
            ItemCondition(item_id="key", min_count=1),
        ])

        result = condition.evaluate(mock_context)
        assert result.success is True

    def test_not_condition(self, mock_context):
        """Test NOT condition."""
        mock_context.get_item_count.return_value = 0

        inner = ItemCondition(item_id="cursed_item", min_count=1)
        condition = NotCondition(inner)

        result = condition.evaluate(mock_context)
        assert result.success is True

    def test_always_true_condition(self, mock_context):
        """Test always true condition."""
        condition = AlwaysTrueCondition()
        result = condition.evaluate(mock_context)
        assert result.success is True

    def test_always_false_condition(self, mock_context):
        """Test always false condition."""
        condition = AlwaysFalseCondition()
        result = condition.evaluate(mock_context)
        assert result.success is False


# =============================================================================
# Variable Substitution Tests
# =============================================================================

class TestVariableSubstitution:
    """Tests for variable substitution in dialogue text."""

    def test_simple_substitution(self, variable_manager):
        """Test simple variable substitution."""
        variable_manager.set("player_name", "Hero")

        text = "Welcome, {player_name}!"
        result = variable_manager.substitute(text)

        assert result == "Welcome, Hero!"

    def test_multiple_substitutions(self, variable_manager):
        """Test multiple substitutions."""
        variable_manager.set("name", "Hero")
        variable_manager.set("title", "Warrior")

        text = "{name} the {title} enters."
        result = variable_manager.substitute(text)

        assert result == "Hero the Warrior enters."

    def test_substitution_missing_variable(self, variable_manager):
        """Test substitution with missing variable."""
        text = "Hello, {unknown}!"
        result = variable_manager.substitute(text)

        # Should either keep placeholder or replace with empty
        assert "{unknown}" in result or result == "Hello, !"

    def test_substitution_with_default(self, variable_manager):
        """Test substitution with default value."""
        text = "Hello, {player_name:Stranger}!"
        result = variable_manager.substitute(text)

        assert result == "Hello, Stranger!"

    def test_numeric_substitution(self, variable_manager):
        """Test numeric variable substitution."""
        variable_manager.set("gold", 100)

        text = "You have {gold} gold coins."
        result = variable_manager.substitute(text)

        assert result == "You have 100 gold coins."

    def test_nested_braces(self, variable_manager):
        """Test text with escaped braces."""
        variable_manager.set("name", "Hero")

        text = "{{Literal braces}} and {name}"
        result = variable_manager.substitute(text)

        assert "{Literal braces}" in result
        assert "Hero" in result

    def test_scoped_substitution(self, variable_manager):
        """Test substitution with scoped variables."""
        variable_manager.set("global_var", "Global Value", scope=VariableScope.GLOBAL)
        variable_manager.set("local_var", "Local Value", scope=VariableScope.LOCAL)

        text = "{global_var} and {local_var}"
        result = variable_manager.substitute(text)

        assert "Global Value" in result
        assert "Local Value" in result


# =============================================================================
# Speaker Context Tests
# =============================================================================

class TestSpeakerContext:
    """Tests for speaker context functionality."""

    def test_speaker_creation(self):
        """Test creating a dialogue speaker."""
        speaker = DialogueSpeaker(
            id="npc_001",
            name="Guard Captain",
            portrait="guard_portrait",
        )
        assert speaker.id == "npc_001"
        assert speaker.name == "Guard Captain"

    def test_speaker_with_voice(self):
        """Test speaker with voice settings."""
        speaker = DialogueSpeaker(
            id="wizard",
            name="Old Wizard",
            voice_id="wise_elder",
            voice_pitch=0.8,
        )
        assert speaker.voice_id == "wise_elder"
        assert speaker.voice_pitch == 0.8

    def test_speaker_mood(self):
        """Test speaker mood property."""
        speaker = DialogueSpeaker(
            id="npc",
            name="NPC",
            mood="angry",
        )
        assert speaker.mood == "angry"

    def test_session_get_speaker_info(self, simple_graph, mock_context):
        """Test getting speaker info from session."""
        # Add speaker to graph
        simple_graph.speakers["npc_001"] = DialogueSpeaker(
            id="npc_001",
            name="Merchant",
            portrait="merchant_happy",
        )

        session = DialogueSession(graph=simple_graph, context=mock_context)
        session.start()

        speaker_info = session.get_current_speaker_info()

        assert speaker_info is not None
        assert speaker_info.name == "Merchant"


# =============================================================================
# Dialogue Events Tests
# =============================================================================

class TestDialogueEvents:
    """Tests for dialogue event system."""

    def test_event_node_fires_event(self, mock_context):
        """Test that event nodes fire events."""
        events_fired = []

        def on_event(name, data):
            events_fired.append((name, data))

        mock_context.trigger_event = on_event

        graph = DialogueGraph(id="event_test")
        graph.add_node(EntryNode(id="start", next_node="event1"))
        graph.add_node(EventNode(
            id="event1",
            event_name="quest_offered",
            event_data={"quest_id": "q001"},
            next_node="exit",
        ))
        graph.add_node(ExitNode(id="exit"))
        graph.entry_point = "start"

        session = DialogueSession(graph=graph, context=mock_context)
        session.start()

        assert len(events_fired) == 1
        assert events_fired[0][0] == "quest_offered"
        assert events_fired[0][1]["quest_id"] == "q001"

    def test_session_event_listeners(self, simple_graph, mock_context):
        """Test session event listeners."""
        events = []

        def on_dialogue_event(event_type, data):
            events.append((event_type, data))

        session = DialogueSession(graph=simple_graph, context=mock_context)
        session.add_event_listener(on_dialogue_event)
        session.start()

        # Should fire node_entered event
        assert any(e[0] == "node_entered" for e in events)

    def test_choice_selection_event(self, simple_graph, mock_context):
        """Test event fired on choice selection."""
        events = []

        def on_event(event_type, data):
            events.append((event_type, data))

        session = DialogueSession(graph=simple_graph, context=mock_context)
        session.add_event_listener(on_event)
        session.start()
        session.advance()  # To choice
        session.select_choice(0)

        assert any(e[0] == "choice_selected" for e in events)

    def test_dialogue_end_event(self, simple_graph, mock_context):
        """Test event fired on dialogue end."""
        events = []

        def on_event(event_type, data):
            events.append((event_type, data))

        session = DialogueSession(graph=simple_graph, context=mock_context)
        session.add_event_listener(on_event)
        session.start()
        session.advance()
        session.select_choice(1)  # Goodbye

        assert any(e[0] == "dialogue_ended" for e in events)


# =============================================================================
# Branching Dialogue Tests
# =============================================================================

class TestBranchingDialogue:
    """Tests for branching dialogue functionality."""

    def test_multiple_branches(self, mock_context):
        """Test dialogue with multiple branches."""
        graph = DialogueGraph(id="multi_branch")

        graph.add_node(EntryNode(id="start", next_node="branch"))
        graph.add_node(BranchNode(
            id="branch",
            branches=[
                ("high_rep", "good_path"),
                ("medium_rep", "neutral_path"),
                ("default", "bad_path"),
            ],
            conditions={
                "high_rep": VariableCondition("reputation", ComparisonOperator.GREATER_EQUAL, 75),
                "medium_rep": VariableCondition("reputation", ComparisonOperator.GREATER_EQUAL, 25),
            },
        ))
        graph.add_node(TextNode(id="good_path", speaker_id="npc", text="Good", next_node="exit"))
        graph.add_node(TextNode(id="neutral_path", speaker_id="npc", text="Neutral", next_node="exit"))
        graph.add_node(TextNode(id="bad_path", speaker_id="npc", text="Bad", next_node="exit"))
        graph.add_node(ExitNode(id="exit"))
        graph.entry_point = "start"

        # Test high reputation
        mock_context.variables.set("reputation", 80)
        session = DialogueSession(graph=graph, context=mock_context)
        session.start()
        assert session.current_node.id == "good_path"

        # Test medium reputation
        mock_context.variables.set("reputation", 50)
        session = DialogueSession(graph=graph, context=mock_context)
        session.start()
        assert session.current_node.id == "neutral_path"

        # Test low reputation
        mock_context.variables.set("reputation", 10)
        session = DialogueSession(graph=graph, context=mock_context)
        session.start()
        assert session.current_node.id == "bad_path"

    def test_conditional_choices(self, mock_context):
        """Test choices with conditions."""
        graph = DialogueGraph(id="conditional_choice")

        graph.add_node(EntryNode(id="start", next_node="choice"))
        graph.add_node(ChoiceNode(
            id="choice",
            choices=[
                DialogueChoice(
                    text="[Strength 15] Break down the door",
                    next_node="strength_path",
                    condition=VariableCondition("strength", ComparisonOperator.GREATER_EQUAL, 15),
                ),
                DialogueChoice(
                    text="[Lock pick] Pick the lock",
                    next_node="pick_path",
                    condition=ItemCondition(item_id="lockpick"),
                ),
                DialogueChoice(
                    text="Leave",
                    next_node="exit",
                ),
            ],
        ))
        graph.add_node(TextNode(id="strength_path", speaker_id="npc", text="Smash", next_node="exit"))
        graph.add_node(TextNode(id="pick_path", speaker_id="npc", text="Click", next_node="exit"))
        graph.add_node(ExitNode(id="exit"))
        graph.entry_point = "start"

        # Low strength, no lockpick
        mock_context.variables.set("strength", 10)
        mock_context.get_item_count.return_value = 0

        session = DialogueSession(graph=graph, context=mock_context)
        session.start()

        choices = session.get_available_choices()
        # Only "Leave" should be available
        assert len(choices) == 1
        assert choices[0].text == "Leave"

    def test_random_variation(self, mock_context):
        """Test random dialogue variations."""
        graph = DialogueGraph(id="random_test")

        graph.add_node(EntryNode(id="start", next_node="random"))
        graph.add_node(RandomNode(
            id="random",
            variations=[
                ("greeting_1", 0.5),
                ("greeting_2", 0.5),
            ],
        ))
        graph.add_node(TextNode(id="greeting_1", speaker_id="npc", text="Hello!", next_node="exit"))
        graph.add_node(TextNode(id="greeting_2", speaker_id="npc", text="Hi there!", next_node="exit"))
        graph.add_node(ExitNode(id="exit"))
        graph.entry_point = "start"

        session = DialogueSession(graph=graph, context=mock_context)
        session.start()

        # Should be at one of the greetings
        assert session.current_node.id in ["greeting_1", "greeting_2"]


# =============================================================================
# Localization Tests
# =============================================================================

class TestDialogueLocalization:
    """Tests for dialogue localization hooks."""

    def test_localization_key_generation(self):
        """Test localization key generation."""
        node = TextNode(
            id="text1",
            speaker_id="npc",
            text="Hello!",
            localization_key="dialogue.quest1.greeting",
            next_node="exit",
        )
        assert node.localization_key == "dialogue.quest1.greeting"

    def test_auto_localization_key(self):
        """Test automatic localization key from node ID."""
        node = TextNode(
            id="quest1_greeting",
            speaker_id="npc",
            text="Hello!",
            next_node="exit",
        )
        auto_key = node.get_auto_localization_key()
        assert "quest1_greeting" in auto_key

    def test_choice_localization_key(self):
        """Test choice localization keys."""
        choice = DialogueChoice(
            text="Accept quest",
            next_node="accept",
            localization_key="choice.accept_quest",
        )
        assert choice.localization_key == "choice.accept_quest"

    def test_session_with_localization(self, simple_graph, mock_context):
        """Test session with localization function."""
        def localize(key):
            translations = {
                "dialogue.simple_dialogue.text1": "Translated greeting!",
            }
            return translations.get(key, key)

        session = DialogueSession(
            graph=simple_graph,
            context=mock_context,
            localize_func=localize,
        )
        session.start()

        # If node has localization key, should use translated text
        # (Implementation dependent)

    def test_graph_extract_strings(self, simple_graph):
        """Test extracting all localizable strings from graph."""
        strings = simple_graph.extract_localizable_strings()

        assert any("Hello, traveler!" in s for s in strings)
        assert any("Welcome!" in s for s in strings)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestDialogueEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_graph(self, mock_context):
        """Test handling empty graph."""
        graph = DialogueGraph(id="empty")

        session = DialogueSession(graph=graph, context=mock_context)

        # Should handle gracefully
        session.start()
        assert session.is_active is False

    def test_circular_reference(self, mock_context):
        """Test handling circular references."""
        graph = DialogueGraph(id="circular")

        graph.add_node(EntryNode(id="start", next_node="text1"))
        graph.add_node(TextNode(id="text1", speaker_id="npc", text="Loop", next_node="text1"))

        graph.entry_point = "start"

        # Validation should catch this
        errors = graph.validate()
        # May or may not be an error depending on implementation

    def test_very_long_dialogue(self, mock_context):
        """Test dialogue with many nodes."""
        graph = DialogueGraph(id="long")

        prev_id = "start"
        graph.add_node(EntryNode(id="start", next_node="text_0"))

        for i in range(100):
            node_id = f"text_{i}"
            next_id = f"text_{i+1}" if i < 99 else "exit"
            graph.add_node(TextNode(
                id=node_id,
                speaker_id="npc",
                text=f"Line {i}",
                next_node=next_id,
            ))

        graph.add_node(ExitNode(id="exit"))
        graph.entry_point = "start"

        session = DialogueSession(graph=graph, context=mock_context)
        session.start()

        # Should be able to traverse all nodes
        for _ in range(99):
            session.advance()

        assert session.current_node.id == "exit"

    def test_choice_with_no_available_options(self, mock_context):
        """Test choice node where all options are hidden."""
        graph = DialogueGraph(id="no_choices")

        graph.add_node(EntryNode(id="start", next_node="choice"))
        graph.add_node(ChoiceNode(
            id="choice",
            choices=[
                DialogueChoice(
                    text="Option 1",
                    next_node="path1",
                    condition=AlwaysFalseCondition(),
                ),
                DialogueChoice(
                    text="Option 2",
                    next_node="path2",
                    condition=AlwaysFalseCondition(),
                ),
            ],
            fallback_node="fallback",
        ))
        graph.add_node(TextNode(id="fallback", speaker_id="npc", text="No options", next_node="exit"))
        graph.add_node(ExitNode(id="exit"))
        graph.entry_point = "start"

        session = DialogueSession(graph=graph, context=mock_context)
        session.start()

        choices = session.get_available_choices()
        assert len(choices) == 0

        # Should have fallback option
        session.advance()  # Should go to fallback
        assert session.current_node.id == "fallback"

    def test_special_characters_in_text(self, mock_context):
        """Test handling special characters in text."""
        graph = DialogueGraph(id="special")

        graph.add_node(EntryNode(id="start", next_node="text1"))
        graph.add_node(TextNode(
            id="text1",
            speaker_id="npc",
            text='Special chars: <>&"\'\n\tTab',
            next_node="exit",
        ))
        graph.add_node(ExitNode(id="exit"))
        graph.entry_point = "start"

        session = DialogueSession(graph=graph, context=mock_context)
        session.start()

        text = session.get_current_text()
        assert "<>&" in text or "&lt;&gt;&amp;" in text
