"""
Dialogue Graph System.

Provides the core dialogue graph structure with node types:
- TextNode: NPC speech/narration
- ChoiceNode: Player choice options
- BranchNode: Conditional branching
- EventNode: Trigger game events
- RandomNode: Random variation selection
- Entry/Exit points for dialogue flow
"""

from __future__ import annotations

import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

from .constants import (
    DEFAULT_AUTO_ADVANCE_DELAY,
    DEFAULT_CHOICE_TIMEOUT,
    DEFAULT_LANGUAGE,
    DEFAULT_TYPING_SPEED,
    MAX_CHOICES_PER_NODE,
    MAX_CONNECTIONS_PER_NODE,
    MAX_NODES_PER_GRAPH,
    MAX_TRAVERSAL_DEPTH,
    NodeType,
    PortraitPosition,
    STRING_KEY_PREFIX,
    STRING_KEY_SEPARATOR,
)
from .dialogue_conditions import Condition, ConditionResult, AlwaysTrueCondition
from .dialogue_effects import Effect, EffectBatch, EffectResult


# =============================================================================
# Localization Support
# =============================================================================

class LocalizationProvider(Protocol):
    """Protocol for localization string lookup."""

    def get_string(
        self,
        key: str,
        language: str = DEFAULT_LANGUAGE,
        **kwargs: Any
    ) -> str:
        """
        Get a localized string.

        Args:
            key: The string key.
            language: The language code.
            **kwargs: Format arguments.

        Returns:
            The localized string.
        """
        ...


class DefaultLocalizationProvider:
    """Default localization provider that returns the key."""

    def __init__(self, strings: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Initialize with optional string table.

        Args:
            strings: Dict of {language: {key: value}}.
        """
        self._strings = strings or {}

    def get_string(
        self,
        key: str,
        language: str = DEFAULT_LANGUAGE,
        **kwargs: Any
    ) -> str:
        """Get a localized string."""
        lang_strings = self._strings.get(language, {})
        text = lang_strings.get(key, key)

        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError as e:
                # Log missing format key but return partially formatted text
                import logging
                logging.getLogger(__name__).warning(
                    f"Missing format key in localization string '{key}': {e}"
                )
            except ValueError as e:
                # Log malformed format string
                import logging
                logging.getLogger(__name__).warning(
                    f"Invalid format string in localization key '{key}': {e}"
                )

        return text

    def add_string(
        self,
        key: str,
        value: str,
        language: str = DEFAULT_LANGUAGE
    ) -> None:
        """Add a string to the table."""
        if language not in self._strings:
            self._strings[language] = {}
        self._strings[language][key] = value


# =============================================================================
# Dialogue Context
# =============================================================================

@dataclass
class DialogueState:
    """Current state of dialogue execution."""
    current_node_id: Optional[str] = None
    visited_nodes: Set[str] = field(default_factory=set)
    choice_history: List[Tuple[str, int]] = field(default_factory=list)
    is_active: bool = False
    current_speaker: Optional[str] = None
    current_text: str = ""


# =============================================================================
# Node Base Class
# =============================================================================

@dataclass
class DialogueNode(ABC):
    """
    Abstract base class for all dialogue nodes.

    Each node represents a step in the dialogue flow.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    @abstractmethod
    def node_type(self) -> NodeType:
        """Get the node type."""
        pass

    @abstractmethod
    def get_next_nodes(self, context: Any) -> List[str]:
        """
        Get the IDs of the next possible nodes.

        Args:
            context: The dialogue context.

        Returns:
            List of node IDs.
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueNode":
        """Deserialize from dictionary."""
        pass


# =============================================================================
# Text Node
# =============================================================================

@dataclass
class TextNode(DialogueNode):
    """
    Node displaying NPC speech or narration.

    Supports localized text, speaker portraits, and typing animation.
    """
    text: str = ""
    text_key: Optional[str] = None  # For localization
    speaker: Optional[str] = None
    speaker_portrait: Optional[str] = None
    portrait_position: PortraitPosition = PortraitPosition.LEFT
    typing_speed: float = DEFAULT_TYPING_SPEED
    auto_advance: bool = False
    auto_advance_delay: float = DEFAULT_AUTO_ADVANCE_DELAY
    next_node: Optional[str] = None
    effects: List[Effect] = field(default_factory=list)

    @property
    def node_type(self) -> NodeType:
        return NodeType.TEXT

    def get_text(
        self,
        localization: Optional[LocalizationProvider] = None,
        language: str = DEFAULT_LANGUAGE,
        **format_args: Any
    ) -> str:
        """
        Get the display text, optionally localized.

        Args:
            localization: Localization provider.
            language: Language code.
            **format_args: Format arguments.

        Returns:
            The display text.
        """
        if self.text_key and localization:
            return localization.get_string(
                self.text_key,
                language,
                **format_args
            )
        return self.text.format(**format_args) if format_args else self.text

    def get_next_nodes(self, context: Any) -> List[str]:
        """Get next node (single path)."""
        return [self.next_node] if self.next_node else []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "type": "text",
            "id": self.id,
            "text": self.text,
            "tags": self.tags,
            "metadata": self.metadata,
        }

        if self.text_key:
            result["text_key"] = self.text_key
        if self.speaker:
            result["speaker"] = self.speaker
        if self.speaker_portrait:
            result["speaker_portrait"] = self.speaker_portrait
        if self.portrait_position != PortraitPosition.LEFT:
            result["portrait_position"] = self.portrait_position.name
        if self.typing_speed != DEFAULT_TYPING_SPEED:
            result["typing_speed"] = self.typing_speed
        if self.auto_advance:
            result["auto_advance"] = self.auto_advance
            result["auto_advance_delay"] = self.auto_advance_delay
        if self.next_node:
            result["next_node"] = self.next_node
        if self.effects:
            result["effects"] = [e.to_dict() for e in self.effects]

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextNode":
        """Deserialize from dictionary."""
        from .dialogue_effects import effect_from_dict

        portrait_pos = PortraitPosition.LEFT
        if "portrait_position" in data:
            portrait_pos = PortraitPosition[data["portrait_position"]]

        effects = []
        if "effects" in data:
            effects = [effect_from_dict(e) for e in data["effects"]]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            text=data.get("text", ""),
            text_key=data.get("text_key"),
            speaker=data.get("speaker"),
            speaker_portrait=data.get("speaker_portrait"),
            portrait_position=portrait_pos,
            typing_speed=data.get("typing_speed", DEFAULT_TYPING_SPEED),
            auto_advance=data.get("auto_advance", False),
            auto_advance_delay=data.get("auto_advance_delay", DEFAULT_AUTO_ADVANCE_DELAY),
            next_node=data.get("next_node"),
            effects=effects,
            tags=data.get("tags", {}),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Choice Node
# =============================================================================

@dataclass
class Choice:
    """A single choice option."""
    text: str
    text_key: Optional[str] = None
    target_node: str = ""
    condition: Optional[Condition] = None
    effects: List[Effect] = field(default_factory=list)
    enabled: bool = True
    visible_when_disabled: bool = True

    def is_available(self, context: Any) -> bool:
        """Check if choice is available."""
        if not self.enabled:
            return False
        if self.condition:
            result = self.condition.evaluate(context)
            return result.success
        return True

    def get_text(
        self,
        localization: Optional[LocalizationProvider] = None,
        language: str = DEFAULT_LANGUAGE,
        **format_args: Any
    ) -> str:
        """Get the display text."""
        if self.text_key and localization:
            return localization.get_string(
                self.text_key,
                language,
                **format_args
            )
        return self.text.format(**format_args) if format_args else self.text

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "text": self.text,
            "target_node": self.target_node,
        }

        if self.text_key:
            result["text_key"] = self.text_key
        if self.condition:
            result["condition"] = self.condition.to_dict()
        if self.effects:
            result["effects"] = [e.to_dict() for e in self.effects]
        if not self.enabled:
            result["enabled"] = self.enabled
        if not self.visible_when_disabled:
            result["visible_when_disabled"] = self.visible_when_disabled

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Choice":
        """Deserialize from dictionary."""
        from .dialogue_conditions import condition_from_dict
        from .dialogue_effects import effect_from_dict

        condition = None
        if "condition" in data:
            condition = condition_from_dict(data["condition"])

        effects = []
        if "effects" in data:
            effects = [effect_from_dict(e) for e in data["effects"]]

        return cls(
            text=data.get("text", ""),
            text_key=data.get("text_key"),
            target_node=data.get("target_node", ""),
            condition=condition,
            effects=effects,
            enabled=data.get("enabled", True),
            visible_when_disabled=data.get("visible_when_disabled", True),
        )


@dataclass
class ChoiceNode(DialogueNode):
    """
    Node presenting player with choices.

    Supports conditional choices, timeouts, and default selections.
    """
    prompt: str = ""
    prompt_key: Optional[str] = None
    choices: List[Choice] = field(default_factory=list)
    timeout: float = DEFAULT_CHOICE_TIMEOUT
    timeout_choice: int = 0  # Index of choice when timeout expires
    speaker: Optional[str] = None

    def __post_init__(self):
        """Validate parameters."""
        if len(self.choices) > MAX_CHOICES_PER_NODE:
            raise ValueError(
                f"Too many choices: {len(self.choices)} > {MAX_CHOICES_PER_NODE}"
            )

    @property
    def node_type(self) -> NodeType:
        return NodeType.CHOICE

    def get_available_choices(self, context: Any) -> List[Tuple[int, Choice]]:
        """
        Get list of available choices.

        Args:
            context: The dialogue context.

        Returns:
            List of (index, choice) tuples.
        """
        available = []
        for i, choice in enumerate(self.choices):
            if choice.is_available(context):
                available.append((i, choice))
            elif choice.visible_when_disabled:
                # Include disabled but visible choices
                available.append((i, choice))
        return available

    def get_next_nodes(self, context: Any) -> List[str]:
        """Get all possible next nodes."""
        return [c.target_node for c in self.choices if c.target_node]

    def select_choice(
        self,
        index: int,
        context: Any
    ) -> Tuple[Optional[str], List[EffectResult]]:
        """
        Select a choice and execute its effects.

        Args:
            index: The choice index.
            context: The effect context.

        Returns:
            Tuple of (next_node_id, effect_results).
        """
        if not 0 <= index < len(self.choices):
            return None, []

        choice = self.choices[index]

        if not choice.is_available(context):
            return None, []

        # Execute choice effects
        results = []
        for effect in choice.effects:
            result = effect.execute(context)
            results.append(result)

        return choice.target_node, results

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "type": "choice",
            "id": self.id,
            "prompt": self.prompt,
            "choices": [c.to_dict() for c in self.choices],
            "tags": self.tags,
            "metadata": self.metadata,
        }

        if self.prompt_key:
            result["prompt_key"] = self.prompt_key
        if self.timeout != DEFAULT_CHOICE_TIMEOUT:
            result["timeout"] = self.timeout
            result["timeout_choice"] = self.timeout_choice
        if self.speaker:
            result["speaker"] = self.speaker

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChoiceNode":
        """Deserialize from dictionary."""
        choices = [Choice.from_dict(c) for c in data.get("choices", [])]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            prompt=data.get("prompt", ""),
            prompt_key=data.get("prompt_key"),
            choices=choices,
            timeout=data.get("timeout", DEFAULT_CHOICE_TIMEOUT),
            timeout_choice=data.get("timeout_choice", 0),
            speaker=data.get("speaker"),
            tags=data.get("tags", {}),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Branch Node
# =============================================================================

@dataclass
class Branch:
    """A single branch option with condition."""
    condition: Condition
    target_node: str
    priority: int = 0  # Higher priority evaluated first

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "condition": self.condition.to_dict(),
            "target_node": self.target_node,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Branch":
        """Deserialize from dictionary."""
        from .dialogue_conditions import condition_from_dict

        return cls(
            condition=condition_from_dict(data["condition"]),
            target_node=data["target_node"],
            priority=data.get("priority", 0),
        )


@dataclass
class BranchNode(DialogueNode):
    """
    Node for conditional branching.

    Evaluates conditions in priority order and takes the first matching branch.
    """
    branches: List[Branch] = field(default_factory=list)
    default_node: Optional[str] = None

    @property
    def node_type(self) -> NodeType:
        return NodeType.BRANCH

    def evaluate(self, context: Any) -> Tuple[Optional[str], ConditionResult]:
        """
        Evaluate branches and return the selected path.

        Args:
            context: The dialogue context.

        Returns:
            Tuple of (target_node_id, condition_result).
        """
        # Sort by priority (highest first)
        sorted_branches = sorted(
            self.branches,
            key=lambda b: b.priority,
            reverse=True
        )

        for branch in sorted_branches:
            result = branch.condition.evaluate(context)
            if result.success:
                return branch.target_node, result

        # No branch matched, use default
        if self.default_node:
            return self.default_node, ConditionResult(
                success=True,
                message="Using default branch"
            )

        return None, ConditionResult(
            success=False,
            message="No branch condition matched"
        )

    def get_next_nodes(self, context: Any) -> List[str]:
        """Get all possible next nodes."""
        nodes = [b.target_node for b in self.branches]
        if self.default_node:
            nodes.append(self.default_node)
        return nodes

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "type": "branch",
            "id": self.id,
            "branches": [b.to_dict() for b in self.branches],
            "tags": self.tags,
            "metadata": self.metadata,
        }

        if self.default_node:
            result["default_node"] = self.default_node

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BranchNode":
        """Deserialize from dictionary."""
        branches = [Branch.from_dict(b) for b in data.get("branches", [])]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            branches=branches,
            default_node=data.get("default_node"),
            tags=data.get("tags", {}),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Event Node
# =============================================================================

@dataclass
class EventNode(DialogueNode):
    """
    Node that triggers game events.

    Used for cutscenes, animations, gameplay triggers, etc.
    """
    event_name: str = ""
    event_data: Dict[str, Any] = field(default_factory=dict)
    effects: List[Effect] = field(default_factory=list)
    wait_for_completion: bool = False
    next_node: Optional[str] = None

    @property
    def node_type(self) -> NodeType:
        return NodeType.EVENT

    def execute(self, context: Any) -> List[EffectResult]:
        """
        Execute the event node.

        Args:
            context: The effect context.

        Returns:
            List of effect results.
        """
        results = []
        for effect in self.effects:
            result = effect.execute(context)
            results.append(result)
        return results

    def get_next_nodes(self, context: Any) -> List[str]:
        """Get next node."""
        return [self.next_node] if self.next_node else []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "type": "event",
            "id": self.id,
            "event_name": self.event_name,
            "event_data": self.event_data,
            "effects": [e.to_dict() for e in self.effects],
            "wait_for_completion": self.wait_for_completion,
            "tags": self.tags,
            "metadata": self.metadata,
        }

        if self.next_node:
            result["next_node"] = self.next_node

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventNode":
        """Deserialize from dictionary."""
        from .dialogue_effects import effect_from_dict

        effects = [effect_from_dict(e) for e in data.get("effects", [])]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            event_name=data.get("event_name", ""),
            event_data=data.get("event_data", {}),
            effects=effects,
            wait_for_completion=data.get("wait_for_completion", False),
            next_node=data.get("next_node"),
            tags=data.get("tags", {}),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Random Node
# =============================================================================

@dataclass
class RandomOption:
    """A random selection option with weight."""
    target_node: str
    weight: float = 1.0
    condition: Optional[Condition] = None

    def __post_init__(self):
        """Validate parameters."""
        if self.weight <= 0:
            raise ValueError("weight must be > 0")

    def is_available(self, context: Any) -> bool:
        """Check if option is available."""
        if self.condition:
            result = self.condition.evaluate(context)
            return result.success
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "target_node": self.target_node,
            "weight": self.weight,
        }
        if self.condition:
            result["condition"] = self.condition.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RandomOption":
        """Deserialize from dictionary."""
        from .dialogue_conditions import condition_from_dict

        condition = None
        if "condition" in data:
            condition = condition_from_dict(data["condition"])

        return cls(
            target_node=data["target_node"],
            weight=data.get("weight", 1.0),
            condition=condition,
        )


@dataclass
class RandomNode(DialogueNode):
    """
    Node that randomly selects from weighted options.

    Useful for variety in NPC responses.
    """
    options: List[RandomOption] = field(default_factory=list)
    seed: Optional[int] = None  # For deterministic selection
    fallback_node: Optional[str] = None

    @property
    def node_type(self) -> NodeType:
        return NodeType.RANDOM

    def select(self, context: Any) -> Optional[str]:
        """
        Select a random option.

        Args:
            context: The dialogue context.

        Returns:
            Selected node ID or None.
        """
        # Filter available options
        available = [
            opt for opt in self.options
            if opt.is_available(context)
        ]

        if not available:
            return self.fallback_node

        # Calculate total weight
        total_weight = sum(opt.weight for opt in available)

        # Use seed if provided
        rng = random.Random(self.seed) if self.seed else random

        # Select based on weight
        choice_val = rng.uniform(0, total_weight)
        cumulative = 0.0

        for opt in available:
            cumulative += opt.weight
            if choice_val <= cumulative:
                return opt.target_node

        # Fallback (shouldn't reach here)
        return available[-1].target_node if available else self.fallback_node

    def get_next_nodes(self, context: Any) -> List[str]:
        """Get all possible next nodes."""
        nodes = [opt.target_node for opt in self.options]
        if self.fallback_node:
            nodes.append(self.fallback_node)
        return list(set(nodes))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "type": "random",
            "id": self.id,
            "options": [o.to_dict() for o in self.options],
            "tags": self.tags,
            "metadata": self.metadata,
        }

        if self.seed is not None:
            result["seed"] = self.seed
        if self.fallback_node:
            result["fallback_node"] = self.fallback_node

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RandomNode":
        """Deserialize from dictionary."""
        options = [RandomOption.from_dict(o) for o in data.get("options", [])]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            options=options,
            seed=data.get("seed"),
            fallback_node=data.get("fallback_node"),
            tags=data.get("tags", {}),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Entry/Exit Nodes
# =============================================================================

@dataclass
class EntryNode(DialogueNode):
    """
    Entry point for a dialogue.

    Can have conditions for availability.
    """
    name: str = "default"
    condition: Optional[Condition] = None
    next_node: Optional[str] = None
    priority: int = 0  # For selecting between multiple valid entries

    @property
    def node_type(self) -> NodeType:
        return NodeType.ENTRY

    def is_available(self, context: Any) -> bool:
        """Check if entry point is available."""
        if self.condition:
            result = self.condition.evaluate(context)
            return result.success
        return True

    def get_next_nodes(self, context: Any) -> List[str]:
        """Get next node."""
        return [self.next_node] if self.next_node else []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "type": "entry",
            "id": self.id,
            "name": self.name,
            "priority": self.priority,
            "tags": self.tags,
            "metadata": self.metadata,
        }

        if self.condition:
            result["condition"] = self.condition.to_dict()
        if self.next_node:
            result["next_node"] = self.next_node

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntryNode":
        """Deserialize from dictionary."""
        from .dialogue_conditions import condition_from_dict

        condition = None
        if "condition" in data:
            condition = condition_from_dict(data["condition"])

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "default"),
            condition=condition,
            next_node=data.get("next_node"),
            priority=data.get("priority", 0),
            tags=data.get("tags", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExitNode(DialogueNode):
    """
    Exit point for a dialogue.

    Can trigger effects on exit.
    """
    name: str = "default"
    effects: List[Effect] = field(default_factory=list)
    next_dialogue: Optional[str] = None  # Chain to another dialogue

    @property
    def node_type(self) -> NodeType:
        return NodeType.EXIT

    def execute(self, context: Any) -> List[EffectResult]:
        """Execute exit effects."""
        results = []
        for effect in self.effects:
            result = effect.execute(context)
            results.append(result)
        return results

    def get_next_nodes(self, context: Any) -> List[str]:
        """Exit nodes have no next node in current dialogue."""
        return []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "type": "exit",
            "id": self.id,
            "name": self.name,
            "effects": [e.to_dict() for e in self.effects],
            "tags": self.tags,
            "metadata": self.metadata,
        }

        if self.next_dialogue:
            result["next_dialogue"] = self.next_dialogue

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExitNode":
        """Deserialize from dictionary."""
        from .dialogue_effects import effect_from_dict

        effects = [effect_from_dict(e) for e in data.get("effects", [])]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "default"),
            effects=effects,
            next_dialogue=data.get("next_dialogue"),
            tags=data.get("tags", {}),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Node Factory
# =============================================================================

NODE_TYPES: Dict[str, type] = {
    "text": TextNode,
    "choice": ChoiceNode,
    "branch": BranchNode,
    "event": EventNode,
    "random": RandomNode,
    "entry": EntryNode,
    "exit": ExitNode,
}


def node_from_dict(data: Dict[str, Any]) -> DialogueNode:
    """
    Create a node from a dictionary.

    Args:
        data: Dictionary representation of a node.

    Returns:
        DialogueNode instance.

    Raises:
        ValueError: If node type is unknown.
    """
    node_type = data.get("type")

    if node_type not in NODE_TYPES:
        raise ValueError(f"Unknown node type: {node_type}")

    return NODE_TYPES[node_type].from_dict(data)


# =============================================================================
# Dialogue Graph
# =============================================================================

@dataclass
class DialogueGraph:
    """
    A complete dialogue graph containing nodes and connections.

    Provides methods for traversal, validation, and serialization.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    nodes: Dict[str, DialogueNode] = field(default_factory=dict)
    default_entry: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    localization: Optional[LocalizationProvider] = field(
        default=None,
        repr=False
    )

    def __post_init__(self):
        """Validate the graph."""
        if len(self.nodes) > MAX_NODES_PER_GRAPH:
            raise ValueError(
                f"Too many nodes: {len(self.nodes)} > {MAX_NODES_PER_GRAPH}"
            )

    def add_node(self, node: DialogueNode) -> None:
        """
        Add a node to the graph.

        Args:
            node: The node to add.

        Raises:
            ValueError: If node ID already exists or max nodes reached.
        """
        if len(self.nodes) >= MAX_NODES_PER_GRAPH:
            raise ValueError(f"Maximum nodes ({MAX_NODES_PER_GRAPH}) reached")

        if node.id in self.nodes:
            raise ValueError(f"Node ID already exists: {node.id}")

        self.nodes[node.id] = node

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the graph.

        Args:
            node_id: The node ID to remove.

        Returns:
            True if removed, False if not found.
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
            return True
        return False

    def get_node(self, node_id: str) -> Optional[DialogueNode]:
        """
        Get a node by ID.

        Args:
            node_id: The node ID.

        Returns:
            The node or None if not found.
        """
        return self.nodes.get(node_id)

    def get_entry_points(self) -> List[EntryNode]:
        """
        Get all entry points.

        Returns:
            List of entry nodes sorted by priority.
        """
        entries = [
            node for node in self.nodes.values()
            if isinstance(node, EntryNode)
        ]
        return sorted(entries, key=lambda n: n.priority, reverse=True)

    def get_entry_point(
        self,
        name: str = "default",
        context: Any = None
    ) -> Optional[EntryNode]:
        """
        Get an entry point by name.

        Args:
            name: The entry point name.
            context: Optional context for availability check.

        Returns:
            The entry node or None.
        """
        for node in self.get_entry_points():
            if node.name == name:
                if context is None or node.is_available(context):
                    return node
        return None

    def get_best_entry(self, context: Any) -> Optional[EntryNode]:
        """
        Get the best available entry point.

        Args:
            context: The dialogue context.

        Returns:
            The highest priority available entry.
        """
        for entry in self.get_entry_points():
            if entry.is_available(context):
                return entry
        return None

    def get_exit_points(self) -> List[ExitNode]:
        """Get all exit points."""
        return [
            node for node in self.nodes.values()
            if isinstance(node, ExitNode)
        ]

    def validate(self) -> List[str]:
        """
        Validate the graph structure.

        Returns:
            List of validation error messages.
        """
        errors = []

        # Check for at least one entry point
        entries = self.get_entry_points()
        if not entries:
            errors.append("Graph has no entry points")

        # Check for at least one exit point
        exits = self.get_exit_points()
        if not exits:
            errors.append("Graph has no exit points")

        # Check for broken connections
        for node_id, node in self.nodes.items():
            next_nodes = node.get_next_nodes(None)
            for next_id in next_nodes:
                if next_id and next_id not in self.nodes:
                    errors.append(
                        f"Node '{node_id}' references non-existent node '{next_id}'"
                    )

        # Check for unreachable nodes
        reachable = self._get_reachable_nodes()
        for node_id in self.nodes:
            if node_id not in reachable:
                node = self.nodes[node_id]
                if not isinstance(node, EntryNode):
                    errors.append(f"Node '{node_id}' is unreachable")

        return errors

    def _get_reachable_nodes(self) -> Set[str]:
        """Get all nodes reachable from entry points."""
        reachable: Set[str] = set()
        to_visit: List[str] = []

        # Start from all entry points
        for entry in self.get_entry_points():
            to_visit.append(entry.id)

        # BFS traversal
        while to_visit:
            node_id = to_visit.pop(0)
            if node_id in reachable:
                continue

            reachable.add(node_id)
            node = self.nodes.get(node_id)
            if node:
                next_nodes = node.get_next_nodes(None)
                for next_id in next_nodes:
                    if next_id and next_id not in reachable:
                        to_visit.append(next_id)

        return reachable

    def get_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = MAX_TRAVERSAL_DEPTH
    ) -> Optional[List[str]]:
        """
        Find a path between two nodes.

        Args:
            start_id: Starting node ID.
            end_id: Target node ID.
            max_depth: Maximum search depth.

        Returns:
            List of node IDs forming a path, or None if no path exists.
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return None

        visited: Set[str] = set()
        queue: List[List[str]] = [[start_id]]

        while queue:
            path = queue.pop(0)

            if len(path) > max_depth:
                continue

            current_id = path[-1]

            if current_id == end_id:
                return path

            if current_id in visited:
                continue

            visited.add(current_id)
            node = self.nodes.get(current_id)

            if node:
                next_nodes = node.get_next_nodes(None)
                for next_id in next_nodes:
                    if next_id and next_id not in visited:
                        queue.append(path + [next_id])

        return None

    def iterate_nodes(
        self,
        node_type: Optional[type] = None
    ) -> Iterator[DialogueNode]:
        """
        Iterate over nodes, optionally filtered by type.

        Args:
            node_type: Optional node type to filter by.

        Yields:
            Dialogue nodes.
        """
        for node in self.nodes.values():
            if node_type is None or isinstance(node, node_type):
                yield node

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": {
                node_id: node.to_dict()
                for node_id, node in self.nodes.items()
            },
            "default_entry": self.default_entry,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueGraph":
        """Deserialize from dictionary."""
        nodes = {}
        for node_id, node_data in data.get("nodes", {}).items():
            nodes[node_id] = node_from_dict(node_data)

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            nodes=nodes,
            default_entry=data.get("default_entry", "default"),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Dialogue Graph Builder
# =============================================================================

class DialogueGraphBuilder:
    """
    Builder for constructing dialogue graphs fluently.
    """

    def __init__(self, name: str = "", graph_id: Optional[str] = None):
        """
        Initialize the builder.

        Args:
            name: The graph name.
            graph_id: Optional graph ID.
        """
        self._graph = DialogueGraph(
            id=graph_id or str(uuid.uuid4()),
            name=name
        )
        self._current_node: Optional[DialogueNode] = None

    def set_description(self, description: str) -> "DialogueGraphBuilder":
        """Set the graph description."""
        self._graph.description = description
        return self

    def add_entry(
        self,
        name: str = "default",
        node_id: Optional[str] = None,
        condition: Optional[Condition] = None,
        priority: int = 0
    ) -> "DialogueGraphBuilder":
        """Add an entry point."""
        node = EntryNode(
            id=node_id or str(uuid.uuid4()),
            name=name,
            condition=condition,
            priority=priority
        )
        self._graph.add_node(node)
        self._current_node = node
        return self

    def add_text(
        self,
        text: str,
        speaker: Optional[str] = None,
        node_id: Optional[str] = None,
        **kwargs: Any
    ) -> "DialogueGraphBuilder":
        """Add a text node."""
        node = TextNode(
            id=node_id or str(uuid.uuid4()),
            text=text,
            speaker=speaker,
            **kwargs
        )
        self._graph.add_node(node)

        # Connect from previous node if exists
        if self._current_node:
            self._connect_to(node.id)

        self._current_node = node
        return self

    def add_choice(
        self,
        prompt: str,
        choices: List[Tuple[str, str]],  # (text, target_node_id)
        node_id: Optional[str] = None,
        **kwargs: Any
    ) -> "DialogueGraphBuilder":
        """Add a choice node."""
        choice_objs = [
            Choice(text=text, target_node=target)
            for text, target in choices
        ]

        node = ChoiceNode(
            id=node_id or str(uuid.uuid4()),
            prompt=prompt,
            choices=choice_objs,
            **kwargs
        )
        self._graph.add_node(node)

        if self._current_node:
            self._connect_to(node.id)

        self._current_node = node
        return self

    def add_branch(
        self,
        branches: List[Tuple[Condition, str]],  # (condition, target_node_id)
        default: Optional[str] = None,
        node_id: Optional[str] = None
    ) -> "DialogueGraphBuilder":
        """Add a branch node."""
        branch_objs = [
            Branch(condition=cond, target_node=target)
            for cond, target in branches
        ]

        node = BranchNode(
            id=node_id or str(uuid.uuid4()),
            branches=branch_objs,
            default_node=default
        )
        self._graph.add_node(node)

        if self._current_node:
            self._connect_to(node.id)

        self._current_node = node
        return self

    def add_exit(
        self,
        name: str = "default",
        node_id: Optional[str] = None,
        effects: Optional[List[Effect]] = None
    ) -> "DialogueGraphBuilder":
        """Add an exit point."""
        node = ExitNode(
            id=node_id or str(uuid.uuid4()),
            name=name,
            effects=effects or []
        )
        self._graph.add_node(node)

        if self._current_node:
            self._connect_to(node.id)

        self._current_node = node
        return self

    def connect(self, from_id: str, to_id: str) -> "DialogueGraphBuilder":
        """
        Connect two nodes.

        Only works for nodes with a single next_node attribute.
        """
        from_node = self._graph.get_node(from_id)
        if from_node and hasattr(from_node, "next_node"):
            from_node.next_node = to_id
        return self

    def _connect_to(self, target_id: str) -> None:
        """Connect current node to target."""
        if self._current_node and hasattr(self._current_node, "next_node"):
            self._current_node.next_node = target_id

    def build(self) -> DialogueGraph:
        """Build and return the graph."""
        return self._graph
