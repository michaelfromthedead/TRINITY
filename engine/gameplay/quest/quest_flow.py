"""
Quest Flow Module.

Provides flow types for quest progression:
- Sequential: Objectives must be completed in order
- Parallel: Objectives can be completed in any order
- Branching: Player choices affect quest path
- Optional: Bonus objectives that don't affect completion
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .objectives import Objective

__all__ = [
    "FlowType",
    "FlowState",
    "QuestFlow",
    "SequentialFlow",
    "ParallelFlow",
    "BranchingFlow",
    "OptionalFlow",
    "MixedFlow",
    "FlowNode",
    "BranchCondition",
    "FlowBuilder",
]


class FlowType(Enum):
    """Types of quest flows."""

    SEQUENTIAL = auto()  # Complete in order
    PARALLEL = auto()  # Complete in any order
    BRANCHING = auto()  # Player choice affects path
    OPTIONAL = auto()  # Bonus objectives
    MIXED = auto()  # Combination of flow types


class FlowState(Enum):
    """Flow execution states."""

    PENDING = auto()  # Not yet started
    ACTIVE = auto()  # Currently executing
    COMPLETE = auto()  # Successfully finished
    FAILED = auto()  # Flow failed
    SKIPPED = auto()  # Skipped (branching)


@dataclass
class BranchCondition:
    """
    Condition for branching paths.

    Evaluates whether a branch should be taken based on game state.
    """

    id: str
    description: str
    check: Callable[[dict[str, Any]], bool] | None = None
    required_flags: set[str] = field(default_factory=set)
    required_items: dict[str, int] = field(default_factory=dict)
    required_reputation: dict[str, int] = field(default_factory=dict)
    choice_text: str = ""  # Text shown to player for manual choice

    def evaluate(self, context: dict[str, Any]) -> bool:
        """
        Evaluate the condition against a context.

        Args:
            context: Game state context containing flags, items, reputation, etc.

        Returns:
            True if condition is met
        """
        # Check custom function first
        if self.check is not None:
            if not self.check(context):
                return False

        # Check required flags
        flags = context.get("flags", set())
        if not self.required_flags.issubset(flags):
            return False

        # Check required items
        items = context.get("items", {})
        for item_id, required in self.required_items.items():
            if items.get(item_id, 0) < required:
                return False

        # Check required reputation
        reputation = context.get("reputation", {})
        for faction_id, required in self.required_reputation.items():
            if reputation.get(faction_id, 0) < required:
                return False

        return True


@dataclass
class FlowNode:
    """
    A node in the quest flow graph.

    Represents a single step or group of steps in quest progression.
    """

    id: str
    objectives: list[Objective] = field(default_factory=list)
    flow_type: FlowType = FlowType.SEQUENTIAL
    state: FlowState = FlowState.PENDING
    children: list[FlowNode] = field(default_factory=list)
    branch_conditions: dict[str, BranchCondition] = field(default_factory=dict)
    optional: bool = False
    on_enter: Callable[[FlowNode], None] | None = None
    on_exit: Callable[[FlowNode], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if this node is complete."""
        return self.state == FlowState.COMPLETE

    @property
    def is_active(self) -> bool:
        """Check if this node is active."""
        return self.state == FlowState.ACTIVE

    @property
    def progress(self) -> float:
        """Calculate progress through this node."""
        if not self.objectives:
            if self.children:
                return sum(c.progress for c in self.children) / len(self.children)
            return 0.0

        return sum(o.progress for o in self.objectives) / len(self.objectives)

    def activate(self) -> None:
        """Activate this node."""
        if self.state != FlowState.PENDING:
            return

        self.state = FlowState.ACTIVE

        if self.on_enter:
            self.on_enter(self)

        # Activate objectives based on flow type
        if self.flow_type == FlowType.SEQUENTIAL:
            if self.objectives:
                self.objectives[0].activate()
        else:
            for obj in self.objectives:
                obj.activate()

    def complete(self) -> None:
        """Mark this node as complete."""
        self.state = FlowState.COMPLETE
        if self.on_exit:
            self.on_exit(self)

    def fail(self) -> None:
        """Mark this node as failed."""
        self.state = FlowState.FAILED
        if self.on_exit:
            self.on_exit(self)

    def skip(self) -> None:
        """Skip this node (for branching)."""
        self.state = FlowState.SKIPPED

    def add_objective(self, objective: Objective) -> None:
        """Add an objective to this node."""
        self.objectives.append(objective)

    def add_child(self, node: FlowNode) -> None:
        """Add a child node."""
        self.children.append(node)

    def set_branch_condition(self, child_id: str, condition: BranchCondition) -> None:
        """Set condition for a child branch."""
        self.branch_conditions[child_id] = condition


class QuestFlow(ABC):
    """
    Base class for quest flow controllers.

    Manages the progression through quest objectives and handles
    state transitions based on the flow type.
    """

    def __init__(self, quest_id: str) -> None:
        self.quest_id = quest_id
        self._root: FlowNode | None = None
        self._current_node: FlowNode | None = None
        self._state = FlowState.PENDING
        self._on_complete: Callable[[QuestFlow], None] | None = None
        self._on_fail: Callable[[QuestFlow], None] | None = None
        self._context: dict[str, Any] = {}

    @property
    def state(self) -> FlowState:
        """Get flow state."""
        return self._state

    @property
    def is_complete(self) -> bool:
        """Check if flow is complete."""
        return self._state == FlowState.COMPLETE

    @property
    def is_failed(self) -> bool:
        """Check if flow has failed."""
        return self._state == FlowState.FAILED

    @property
    @abstractmethod
    def flow_type(self) -> FlowType:
        """Get the flow type."""
        ...

    @property
    def progress(self) -> float:
        """Get overall progress."""
        if self._root is None:
            return 0.0
        return self._root.progress

    def set_context(self, context: dict[str, Any]) -> None:
        """Set the evaluation context for branching."""
        self._context = context

    def update_context(self, updates: dict[str, Any]) -> None:
        """Update the evaluation context."""
        self._context.update(updates)

    def set_root(self, node: FlowNode) -> None:
        """Set the root node."""
        self._root = node

    def set_complete_callback(self, callback: Callable[[QuestFlow], None]) -> None:
        """Set callback for flow completion."""
        self._on_complete = callback

    def set_fail_callback(self, callback: Callable[[QuestFlow], None]) -> None:
        """Set callback for flow failure."""
        self._on_fail = callback

    def start(self) -> None:
        """Start the flow."""
        if self._root is None:
            return

        self._state = FlowState.ACTIVE
        self._current_node = self._root
        self._root.activate()

    @abstractmethod
    def update(self) -> None:
        """Update the flow state based on objective progress."""
        ...

    def _complete(self) -> None:
        """Mark flow as complete."""
        self._state = FlowState.COMPLETE
        if self._on_complete:
            self._on_complete(self)

    def _fail(self) -> None:
        """Mark flow as failed."""
        self._state = FlowState.FAILED
        if self._on_fail:
            self._on_fail(self)

    def get_active_objectives(self) -> list[Objective]:
        """Get currently active objectives."""
        if self._current_node is None:
            return []
        return [o for o in self._current_node.objectives if o.is_active]

    def serialize(self) -> dict[str, Any]:
        """Serialize flow state."""
        return {
            "quest_id": self.quest_id,
            "state": self._state.name,
            "context": self._context,
        }


class SequentialFlow(QuestFlow):
    """
    Sequential quest flow.

    Objectives must be completed in a specific order.
    The next objective only becomes active when the previous one completes.
    """

    def __init__(self, quest_id: str) -> None:
        super().__init__(quest_id)
        self._current_index = 0

    @property
    def flow_type(self) -> FlowType:
        return FlowType.SEQUENTIAL

    @property
    def current_index(self) -> int:
        """Get current objective index."""
        return self._current_index

    def update(self) -> None:
        """Update sequential flow."""
        if self._state != FlowState.ACTIVE:
            return
        if self._root is None:
            return

        objectives = self._root.objectives

        if not objectives:
            self._complete()
            return

        # Check current objective
        if self._current_index < len(objectives):
            current = objectives[self._current_index]

            if current.is_complete:
                # Move to next objective
                self._current_index += 1

                if self._current_index < len(objectives):
                    objectives[self._current_index].activate()
                else:
                    # All objectives complete
                    self._root.complete()
                    self._complete()

            elif current.is_failed:
                self._root.fail()
                self._fail()


class ParallelFlow(QuestFlow):
    """
    Parallel quest flow.

    All objectives are active simultaneously and can be completed
    in any order. Flow completes when all objectives are done.
    """

    def __init__(self, quest_id: str, require_all: bool = True) -> None:
        super().__init__(quest_id)
        self.require_all = require_all  # If False, any one completion is enough

    @property
    def flow_type(self) -> FlowType:
        return FlowType.PARALLEL

    def update(self) -> None:
        """Update parallel flow."""
        if self._state != FlowState.ACTIVE:
            return
        if self._root is None:
            return

        objectives = self._root.objectives
        if not objectives:
            self._complete()
            return

        required = [o for o in objectives if not o.optional]
        optional = [o for o in objectives if o.optional]

        if self.require_all:
            # All required must be complete
            if all(o.is_complete for o in required):
                self._root.complete()
                self._complete()
            elif any(o.is_failed for o in required):
                self._root.fail()
                self._fail()
        else:
            # Any one completion is enough
            if any(o.is_complete for o in required):
                self._root.complete()
                self._complete()
            elif all(o.is_failed for o in required):
                self._root.fail()
                self._fail()


class BranchingFlow(QuestFlow):
    """
    Branching quest flow.

    Player choices or conditions determine which path the quest takes.
    Different branches can lead to different outcomes and rewards.
    """

    def __init__(self, quest_id: str, auto_advance: bool = False) -> None:
        super().__init__(quest_id)
        self._chosen_branch: str | None = None
        self._branch_history: list[str] = []
        self._auto_advance = auto_advance  # Auto-select first available branch if True

    @property
    def flow_type(self) -> FlowType:
        return FlowType.BRANCHING

    @property
    def chosen_branch(self) -> str | None:
        """Get the chosen branch ID."""
        return self._chosen_branch

    @property
    def branch_history(self) -> list[str]:
        """Get history of branch choices."""
        return self._branch_history.copy()

    def get_available_branches(self) -> list[tuple[str, BranchCondition]]:
        """
        Get branches available based on current context.

        Returns:
            List of (branch_id, condition) tuples for available branches
        """
        if self._current_node is None:
            return []

        available = []
        for child in self._current_node.children:
            if child.id in self._current_node.branch_conditions:
                condition = self._current_node.branch_conditions[child.id]
                if condition.evaluate(self._context):
                    available.append((child.id, condition))
            else:
                # No condition means always available
                available.append((child.id, BranchCondition(
                    id=child.id,
                    description="",
                )))

        return available

    def choose_branch(self, branch_id: str) -> bool:
        """
        Choose a branch path.

        Args:
            branch_id: ID of the branch to take

        Returns:
            True if branch was successfully chosen
        """
        if self._current_node is None:
            return False

        # Find the branch
        for child in self._current_node.children:
            if child.id == branch_id:
                # Check condition if present
                if branch_id in self._current_node.branch_conditions:
                    condition = self._current_node.branch_conditions[branch_id]
                    if not condition.evaluate(self._context):
                        return False

                # Mark other branches as skipped
                for other in self._current_node.children:
                    if other.id != branch_id:
                        other.skip()

                # Activate chosen branch
                self._chosen_branch = branch_id
                self._branch_history.append(branch_id)
                self._current_node = child
                child.activate()
                return True

        return False

    def auto_choose_branch(self) -> bool:
        """
        Automatically choose a branch based on conditions.

        Returns:
            True if a branch was automatically chosen
        """
        available = self.get_available_branches()
        if available:
            return self.choose_branch(available[0][0])
        return False

    def update(self) -> None:
        """Update branching flow."""
        if self._state != FlowState.ACTIVE:
            return
        if self._current_node is None:
            return

        # Check if current node's objectives are complete
        objectives = self._current_node.objectives
        if objectives:
            required = [o for o in objectives if not o.optional]
            if all(o.is_complete for o in required):
                self._current_node.complete()

                # Check for child branches
                if self._current_node.children:
                    # Auto-select first eligible branch if enabled, otherwise wait for player choice
                    if self._auto_advance:
                        self.auto_choose_branch()
                    # Otherwise wait for explicit choose_branch() call
                else:
                    # No more branches, flow complete
                    self._complete()

            elif any(o.is_failed for o in required):
                self._current_node.fail()
                self._fail()
        else:
            # No objectives, just check children
            if self._current_node.children:
                if not self._chosen_branch:
                    # Auto-select first branch if enabled, otherwise wait for player choice
                    if self._auto_advance:
                        self.auto_choose_branch()
                    # Otherwise wait for explicit choose_branch() call
            else:
                self._current_node.complete()
                self._complete()


class OptionalFlow(QuestFlow):
    """
    Optional quest flow.

    Contains bonus objectives that don't affect quest completion
    but provide additional rewards or story content.
    """

    def __init__(self, quest_id: str, base_flow: QuestFlow) -> None:
        super().__init__(quest_id)
        self._base_flow = base_flow
        self._optional_objectives: list[Objective] = []
        self._bonus_rewards: list[Any] = []

    @property
    def flow_type(self) -> FlowType:
        return FlowType.OPTIONAL

    @property
    def base_flow(self) -> QuestFlow:
        """Get the base flow."""
        return self._base_flow

    @property
    def optional_objectives(self) -> list[Objective]:
        """Get optional objectives."""
        return self._optional_objectives

    @property
    def optional_progress(self) -> float:
        """Get progress on optional objectives."""
        if not self._optional_objectives:
            return 0.0
        return sum(o.progress for o in self._optional_objectives) / len(self._optional_objectives)

    @property
    def completed_optional_count(self) -> int:
        """Get count of completed optional objectives."""
        return sum(1 for o in self._optional_objectives if o.is_complete)

    def add_optional_objective(
        self,
        objective: Objective,
        bonus_reward: Any = None
    ) -> None:
        """Add an optional objective with optional bonus reward."""
        objective.optional = True
        self._optional_objectives.append(objective)
        if bonus_reward is not None:
            self._bonus_rewards.append(bonus_reward)

    def start(self) -> None:
        """Start the flow."""
        super().start()
        self._base_flow.start()

        # Activate optional objectives
        for obj in self._optional_objectives:
            obj.activate()

    def update(self) -> None:
        """Update the flow."""
        if self._state != FlowState.ACTIVE:
            return

        # Update base flow
        self._base_flow.update()

        # Check base flow completion
        if self._base_flow.is_complete:
            self._complete()
        elif self._base_flow.is_failed:
            self._fail()

    def get_earned_bonus_rewards(self) -> list[Any]:
        """Get bonus rewards for completed optional objectives."""
        earned = []
        for i, obj in enumerate(self._optional_objectives):
            if obj.is_complete and i < len(self._bonus_rewards):
                earned.append(self._bonus_rewards[i])
        return earned


class MixedFlow(QuestFlow):
    """
    Mixed quest flow.

    Combines multiple flow types into a complex quest structure.
    Supports nested flows and complex progression paths.
    """

    def __init__(self, quest_id: str) -> None:
        super().__init__(quest_id)
        self._nodes: dict[str, FlowNode] = {}
        self._node_order: list[str] = []
        self._active_nodes: set[str] = set()

    @property
    def flow_type(self) -> FlowType:
        return FlowType.MIXED

    def add_node(self, node: FlowNode, after: str | None = None) -> None:
        """
        Add a node to the flow.

        Args:
            node: The flow node to add
            after: Optional ID of node this should come after
        """
        self._nodes[node.id] = node

        if after is not None and after in self._nodes:
            # Add as child of the specified node
            self._nodes[after].add_child(node)
        else:
            self._node_order.append(node.id)

    def start(self) -> None:
        """Start the mixed flow."""
        self._state = FlowState.ACTIVE

        # Activate first-level nodes based on their flow type
        for node_id in self._node_order:
            node = self._nodes[node_id]
            if node.flow_type == FlowType.PARALLEL:
                node.activate()
                self._active_nodes.add(node_id)
            elif not self._active_nodes:  # First sequential node
                node.activate()
                self._active_nodes.add(node_id)
                if node.flow_type == FlowType.SEQUENTIAL:
                    break

    def update(self) -> None:
        """Update mixed flow."""
        if self._state != FlowState.ACTIVE:
            return

        completed_nodes: list[str] = []
        failed = False

        # Update all active nodes
        for node_id in list(self._active_nodes):
            node = self._nodes[node_id]

            # Check objectives
            if node.objectives:
                if node.flow_type == FlowType.SEQUENTIAL:
                    self._update_sequential_node(node)
                else:
                    self._update_parallel_node(node)

            # Check completion
            if node.is_complete:
                completed_nodes.append(node_id)
            elif node.state == FlowState.FAILED:
                failed = True
                break

        # Handle failures
        if failed:
            self._fail()
            return

        # Handle completions
        for node_id in completed_nodes:
            self._active_nodes.discard(node_id)
            node = self._nodes[node_id]

            # Activate children
            for child in node.children:
                child.activate()
                self._active_nodes.add(child.id)

            # Activate next sequential node
            if node.flow_type == FlowType.SEQUENTIAL:
                idx = self._node_order.index(node_id) if node_id in self._node_order else -1
                if idx >= 0 and idx + 1 < len(self._node_order):
                    next_id = self._node_order[idx + 1]
                    if next_id not in self._active_nodes:
                        self._nodes[next_id].activate()
                        self._active_nodes.add(next_id)

        # Check if all nodes complete
        if not self._active_nodes:
            all_complete = all(
                n.is_complete or n.state == FlowState.SKIPPED
                for n in self._nodes.values()
            )
            if all_complete:
                self._complete()

    def _update_sequential_node(self, node: FlowNode) -> None:
        """Update a sequential node."""
        for i, obj in enumerate(node.objectives):
            if obj.is_active:
                if obj.is_complete:
                    if i + 1 < len(node.objectives):
                        node.objectives[i + 1].activate()
                    else:
                        node.complete()
                elif obj.is_failed and not obj.optional:
                    node.fail()
                break

    def _update_parallel_node(self, node: FlowNode) -> None:
        """Update a parallel node."""
        required = [o for o in node.objectives if not o.optional]
        if all(o.is_complete for o in required):
            node.complete()
        elif any(o.is_failed for o in required):
            node.fail()


class FlowBuilder:
    """
    Builder for constructing quest flows.

    Provides a fluent interface for creating complex quest flows.
    """

    def __init__(self, quest_id: str) -> None:
        self.quest_id = quest_id
        self._root: FlowNode | None = None
        self._current_node: FlowNode | None = None
        self._node_stack: list[FlowNode] = []

    def sequential(self, node_id: str) -> FlowBuilder:
        """Start a sequential flow section."""
        node = FlowNode(id=node_id, flow_type=FlowType.SEQUENTIAL)
        self._add_node(node)
        return self

    def parallel(self, node_id: str) -> FlowBuilder:
        """Start a parallel flow section."""
        node = FlowNode(id=node_id, flow_type=FlowType.PARALLEL)
        self._add_node(node)
        return self

    def branching(self, node_id: str) -> FlowBuilder:
        """Start a branching flow section."""
        node = FlowNode(id=node_id, flow_type=FlowType.BRANCHING)
        self._add_node(node)
        return self

    def optional(self, node_id: str) -> FlowBuilder:
        """Start an optional flow section."""
        node = FlowNode(id=node_id, flow_type=FlowType.OPTIONAL, optional=True)
        self._add_node(node)
        return self

    def _add_node(self, node: FlowNode) -> None:
        """Add a node to the flow."""
        if self._root is None:
            self._root = node
            self._current_node = node
        elif self._current_node is not None:
            self._current_node.add_child(node)
            self._node_stack.append(self._current_node)
            self._current_node = node

    def objective(self, objective: Objective) -> FlowBuilder:
        """Add an objective to the current node."""
        if self._current_node is not None:
            self._current_node.add_objective(objective)
        return self

    def branch(
        self,
        branch_id: str,
        condition: BranchCondition | None = None
    ) -> FlowBuilder:
        """Add a branch to the current node."""
        if self._current_node is not None:
            branch_node = FlowNode(id=branch_id, flow_type=FlowType.SEQUENTIAL)
            self._current_node.add_child(branch_node)
            if condition is not None:
                self._current_node.set_branch_condition(branch_id, condition)
        return self

    def end(self) -> FlowBuilder:
        """End the current flow section."""
        if self._node_stack:
            self._current_node = self._node_stack.pop()
        return self

    def on_enter(self, callback: Callable[[FlowNode], None]) -> FlowBuilder:
        """Set enter callback for current node."""
        if self._current_node is not None:
            self._current_node.on_enter = callback
        return self

    def on_exit(self, callback: Callable[[FlowNode], None]) -> FlowBuilder:
        """Set exit callback for current node."""
        if self._current_node is not None:
            self._current_node.on_exit = callback
        return self

    def build_sequential(self) -> SequentialFlow:
        """Build a sequential flow."""
        flow = SequentialFlow(self.quest_id)
        if self._root is not None:
            flow.set_root(self._root)
        return flow

    def build_parallel(self, require_all: bool = True) -> ParallelFlow:
        """Build a parallel flow."""
        flow = ParallelFlow(self.quest_id, require_all)
        if self._root is not None:
            flow.set_root(self._root)
        return flow

    def build_branching(self) -> BranchingFlow:
        """Build a branching flow."""
        flow = BranchingFlow(self.quest_id)
        if self._root is not None:
            flow.set_root(self._root)
        return flow

    def build_mixed(self) -> MixedFlow:
        """Build a mixed flow."""
        flow = MixedFlow(self.quest_id)
        if self._root is not None:
            flow.add_node(self._root)
        return flow
