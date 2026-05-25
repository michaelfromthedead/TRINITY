"""
AI Debugging - Perception visualization, behavior tree viewer, blackboard display.

Provides tools for debugging AI systems including:
- Perception visualization (sight cones, hearing radius)
- Behavior tree state visualization
- Blackboard key-value display
- AI pause/step functionality
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from engine.gameplay.ai.behavior_tree import BehaviorTree, BTNode
    from engine.gameplay.ai.blackboard import Blackboard

logger = logging.getLogger(__name__)


class AIDebugState(Enum):
    """State of AI debugging."""
    RUNNING = auto()
    PAUSED = auto()
    STEPPING = auto()


class PerceptionType(Enum):
    """Types of perception to visualize."""
    SIGHT = auto()
    HEARING = auto()
    TOUCH = auto()
    SMELL = auto()
    CUSTOM = auto()


@dataclass
class AIDebugConfig:
    """
    Configuration for AI debugging system.

    All numeric constants are defined here to avoid magic numbers.
    """
    # Default perception ranges (units)
    default_sight_range: float = 20.0      # Default sight perception range
    default_hearing_range: float = 15.0    # Default hearing perception range
    default_sight_angle: float = 90.0      # Default sight cone angle (degrees)
    default_hearing_angle: float = 360.0   # Hearing is omnidirectional

    # Default colors (RGBA, 0.0-1.0)
    sight_color: Tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.3)   # Yellow, transparent
    hearing_color: Tuple[float, float, float, float] = (0.0, 1.0, 1.0, 0.2) # Cyan, more transparent
    custom_color: Tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.3)  # Orange

    # Build restrictions
    allow_in_shipping: bool = False  # AI debugging disabled in shipping


@dataclass
class PerceptionVisual:
    """Visual representation of a perception sense."""
    perception_type: PerceptionType
    color: Tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.5)  # RGBA
    range: float = 10.0
    angle: float = 90.0  # Degrees, for cone-based senses
    show_range: bool = True
    show_direction: bool = True
    show_detected: bool = True


@dataclass
class BTNodeDebugInfo:
    """Debug information for a behavior tree node."""
    name: str
    node_type: str
    status: str
    is_active: bool
    depth: int
    execution_count: int = 0
    last_result: Optional[str] = None
    children: List["BTNodeDebugInfo"] = field(default_factory=list)


@dataclass
class BlackboardDebugInfo:
    """Debug information for blackboard entries."""
    key: str
    value: Any
    namespace: str
    value_type: str
    timestamp: float
    ttl: Optional[float] = None


class AIDebugger:
    """
    Debugger for AI systems.

    SECURITY: This debugger is automatically disabled in shipping builds
    to prevent AI manipulation exploits.

    Provides visualization and control for:
    - Perception (sight cones, hearing radius)
    - Behavior trees (active nodes, execution flow)
    - Blackboards (key-value pairs)
    - AI execution (pause, step, override)
    """

    def __init__(self, config: Optional[AIDebugConfig] = None) -> None:
        self._config = config or AIDebugConfig()
        self._state = AIDebugState.RUNNING
        self._enabled = True

        # Visualization settings
        self._show_perception: Set[Any] = set()
        self._show_behavior_tree: Set[Any] = set()
        self._show_blackboard: Set[Any] = set()
        self._show_all_perception = False
        self._show_all_behavior_trees = False
        self._show_all_blackboards = False

        # Perception settings - using config values
        self._perception_configs: Dict[Any, List[PerceptionVisual]] = {}
        self._default_sight = PerceptionVisual(
            perception_type=PerceptionType.SIGHT,
            color=self._config.sight_color,
            range=self._config.default_sight_range,
            angle=self._config.default_sight_angle,
        )
        self._default_hearing = PerceptionVisual(
            perception_type=PerceptionType.HEARING,
            color=self._config.hearing_color,
            range=self._config.default_hearing_range,
            angle=self._config.default_hearing_angle,
        )

        # State overrides
        self._state_overrides: Dict[Any, str] = {}

        # Step control
        self._step_pending = False
        self._step_entity: Optional[Any] = None

        # Callbacks
        self._state_callbacks: List[Callable[[AIDebugState], None]] = []

        # Check build restrictions (after all fields initialized)
        self._build_allowed = self._check_build_allowed()

    def _check_build_allowed(self) -> bool:
        """Check if AI debugging is allowed in this build."""
        import os

        if os.environ.get("GAME_BUILD_TYPE", "").upper() == "SHIPPING":
            if not self._config.allow_in_shipping:
                logger.info("AIDebugger disabled - shipping build")
                return False
        if os.environ.get("SHIPPING") == "1":
            if not self._config.allow_in_shipping:
                return False

        return True

    @property
    def config(self) -> AIDebugConfig:
        """Get the AI debug configuration."""
        return self._config

    @property
    def state(self) -> AIDebugState:
        """Get the current AI debug state."""
        return self._state

    @property
    def enabled(self) -> bool:
        """Check if AI debugging is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable AI debugging."""
        if value and not self._build_allowed:
            logger.warning("Cannot enable AI debugger - not allowed in this build")
            return
        self._enabled = value
        if not value:
            self._show_perception.clear()
            self._show_behavior_tree.clear()
            self._show_blackboard.clear()
            self._show_all_perception = False
            self._show_all_behavior_trees = False
            self._show_all_blackboards = False

    # =========================================================================
    # Perception Debugging
    # =========================================================================

    def show_perception(
        self,
        entity: Any,
        perception_types: Optional[List[PerceptionType]] = None,
    ) -> None:
        """
        Enable perception visualization for an entity.

        Args:
            entity: The entity to visualize
            perception_types: Types to show (default: all)
        """
        self._show_perception.add(entity)

        if perception_types:
            visuals = []
            for ptype in perception_types:
                if ptype == PerceptionType.SIGHT:
                    visuals.append(self._default_sight)
                elif ptype == PerceptionType.HEARING:
                    visuals.append(self._default_hearing)
                else:
                    visuals.append(PerceptionVisual(
                        perception_type=ptype,
                        color=self._config.custom_color,
                    ))
            self._perception_configs[entity] = visuals

        logger.debug("Perception debug enabled for %s", entity)

    def hide_perception(self, entity: Any) -> None:
        """Hide perception visualization for an entity."""
        self._show_perception.discard(entity)
        self._perception_configs.pop(entity, None)

    def set_perception_config(
        self,
        entity: Any,
        config: List[PerceptionVisual],
    ) -> None:
        """Set custom perception visualization config."""
        self._perception_configs[entity] = config

    def get_perception_config(
        self,
        entity: Any,
    ) -> List[PerceptionVisual]:
        """Get perception visualization config for an entity."""
        if entity in self._perception_configs:
            return self._perception_configs[entity]
        return [self._default_sight, self._default_hearing]

    def is_perception_shown(self, entity: Any) -> bool:
        """Check if perception is shown for an entity."""
        return self._show_all_perception or entity in self._show_perception

    def show_all_perception(self, enabled: bool = True) -> None:
        """Enable perception visualization for all entities."""
        self._show_all_perception = enabled

    # =========================================================================
    # Behavior Tree Debugging
    # =========================================================================

    def show_behavior_tree(self, entity: Any) -> None:
        """Enable behavior tree visualization for an entity."""
        self._show_behavior_tree.add(entity)
        logger.debug("Behavior tree debug enabled for %s", entity)

    def hide_behavior_tree(self, entity: Any) -> None:
        """Hide behavior tree visualization for an entity."""
        self._show_behavior_tree.discard(entity)

    def is_behavior_tree_shown(self, entity: Any) -> bool:
        """Check if behavior tree is shown for an entity."""
        return self._show_all_behavior_trees or entity in self._show_behavior_tree

    def show_all_behavior_trees(self, enabled: bool = True) -> None:
        """Enable behavior tree visualization for all entities."""
        self._show_all_behavior_trees = enabled

    def get_behavior_tree_info(
        self,
        entity: Any,
        tree: Optional[Any] = None,
    ) -> Optional[BTNodeDebugInfo]:
        """
        Get debug information for an entity's behavior tree.

        Args:
            entity: The entity to inspect
            tree: Optional explicit behavior tree (otherwise auto-detected)

        Returns:
            Root node debug info with children, or None if not found.
        """
        if tree is None:
            tree = self._get_entity_behavior_tree(entity)

        if tree is None:
            return None

        return self._build_bt_debug_info(tree.root if hasattr(tree, "root") else tree)

    def _get_entity_behavior_tree(self, entity: Any) -> Optional[Any]:
        """Get behavior tree from entity. Override for actual implementation."""
        if hasattr(entity, "behavior_tree"):
            return entity.behavior_tree
        if hasattr(entity, "bt"):
            return entity.bt
        if hasattr(entity, "ai") and hasattr(entity.ai, "behavior_tree"):
            return entity.ai.behavior_tree
        return None

    def _build_bt_debug_info(
        self,
        node: Any,
        depth: int = 0,
    ) -> BTNodeDebugInfo:
        """Recursively build debug info for a behavior tree node."""
        info = BTNodeDebugInfo(
            name=getattr(node, "name", node.__class__.__name__),
            node_type=getattr(node, "node_type", type(node).__name__),
            status=str(getattr(node, "status", "UNKNOWN")),
            is_active=getattr(node, "_status", None) is not None,
            depth=depth,
        )

        # Add children
        children = getattr(node, "_children", None) or getattr(node, "children", None)
        if children:
            for child in children:
                info.children.append(self._build_bt_debug_info(child, depth + 1))

        # Single child (decorator nodes)
        child = getattr(node, "_child", None) or getattr(node, "child", None)
        if child and not children:
            info.children.append(self._build_bt_debug_info(child, depth + 1))

        return info

    # =========================================================================
    # Blackboard Debugging
    # =========================================================================

    def show_blackboard(self, entity: Any) -> None:
        """Enable blackboard visualization for an entity."""
        self._show_blackboard.add(entity)
        logger.debug("Blackboard debug enabled for %s", entity)

    def hide_blackboard(self, entity: Any) -> None:
        """Hide blackboard visualization for an entity."""
        self._show_blackboard.discard(entity)

    def is_blackboard_shown(self, entity: Any) -> bool:
        """Check if blackboard is shown for an entity."""
        return self._show_all_blackboards or entity in self._show_blackboard

    def show_all_blackboards(self, enabled: bool = True) -> None:
        """Enable blackboard visualization for all entities."""
        self._show_all_blackboards = enabled

    def get_blackboard_info(
        self,
        entity: Any,
        blackboard: Optional[Any] = None,
    ) -> List[BlackboardDebugInfo]:
        """
        Get debug information for an entity's blackboard.

        Args:
            entity: The entity to inspect
            blackboard: Optional explicit blackboard (otherwise auto-detected)

        Returns:
            List of blackboard entry info.
        """
        if blackboard is None:
            blackboard = self._get_entity_blackboard(entity)

        if blackboard is None:
            return []

        return self._build_blackboard_debug_info(blackboard)

    def _get_entity_blackboard(self, entity: Any) -> Optional[Any]:
        """Get blackboard from entity. Override for actual implementation."""
        if hasattr(entity, "blackboard"):
            return entity.blackboard
        if hasattr(entity, "_blackboard"):
            return entity._blackboard
        if hasattr(entity, "ai") and hasattr(entity.ai, "blackboard"):
            return entity.ai.blackboard
        return None

    def _build_blackboard_debug_info(
        self,
        blackboard: Any,
    ) -> List[BlackboardDebugInfo]:
        """Build debug info for a blackboard."""
        entries = []

        # Handle our Blackboard class
        if hasattr(blackboard, "_data"):
            for key_str, entry in blackboard._data.items():
                parts = key_str.split(":", 1)
                namespace = parts[0] if len(parts) > 1 else "default"
                key_name = parts[1] if len(parts) > 1 else parts[0]

                entries.append(BlackboardDebugInfo(
                    key=key_name,
                    value=entry.value,
                    namespace=namespace,
                    value_type=type(entry.value).__name__,
                    timestamp=entry.timestamp,
                    ttl=entry.ttl,
                ))
        # Generic dict-like blackboard
        elif hasattr(blackboard, "items"):
            for key, value in blackboard.items():
                entries.append(BlackboardDebugInfo(
                    key=str(key),
                    value=value,
                    namespace="default",
                    value_type=type(value).__name__,
                    timestamp=0.0,
                ))

        return entries

    # =========================================================================
    # AI Control
    # =========================================================================

    def pause_ai(self) -> None:
        """Pause all AI execution."""
        self._state = AIDebugState.PAUSED
        logger.info("AI paused")
        self._notify_state_callbacks(self._state)

    def resume_ai(self) -> None:
        """Resume AI execution."""
        self._state = AIDebugState.RUNNING
        self._step_pending = False
        self._step_entity = None
        logger.info("AI resumed")
        self._notify_state_callbacks(self._state)

    def step_ai(self, entity: Optional[Any] = None) -> None:
        """
        Step AI execution (single tick).

        Args:
            entity: Specific entity to step (None for all)
        """
        if self._state != AIDebugState.PAUSED:
            logger.warning("Cannot step - AI is not paused")
            return

        self._step_pending = True
        self._step_entity = entity
        self._state = AIDebugState.STEPPING

        logger.debug("AI step requested for %s", entity or "all")
        self._notify_state_callbacks(self._state)

    def should_tick(self, entity: Any) -> bool:
        """
        Check if an entity's AI should tick.

        Called by AI systems to check if they should execute.
        """
        if not self._enabled:
            return True

        if self._state == AIDebugState.RUNNING:
            return True

        if self._state == AIDebugState.STEPPING:
            if self._step_pending:
                if self._step_entity is None or self._step_entity == entity:
                    return True

        return False

    def consume_step(self) -> bool:
        """
        Consume a pending step.

        Called after AI tick to consume the step.
        Returns True if step was consumed.
        """
        if self._step_pending:
            self._step_pending = False
            self._step_entity = None
            self._state = AIDebugState.PAUSED
            self._notify_state_callbacks(self._state)
            return True
        return False

    def override_state(self, entity: Any, state: str) -> None:
        """
        Override the AI state for an entity.

        Args:
            entity: Entity to override
            state: State name to force
        """
        self._state_overrides[entity] = state
        logger.info("AI state override: %s -> %s", entity, state)

    def clear_state_override(self, entity: Any) -> None:
        """Clear state override for an entity."""
        self._state_overrides.pop(entity, None)

    def get_state_override(self, entity: Any) -> Optional[str]:
        """Get state override for an entity."""
        return self._state_overrides.get(entity)

    def has_state_override(self, entity: Any) -> bool:
        """Check if entity has a state override."""
        return entity in self._state_overrides

    # =========================================================================
    # Callbacks
    # =========================================================================

    def add_state_callback(
        self,
        callback: Callable[[AIDebugState], None],
    ) -> None:
        """Add a callback for AI debug state changes."""
        self._state_callbacks.append(callback)

    def remove_state_callback(
        self,
        callback: Callable[[AIDebugState], None],
    ) -> bool:
        """Remove a state callback."""
        try:
            self._state_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def _notify_state_callbacks(self, state: AIDebugState) -> None:
        """Notify state callbacks."""
        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error("State callback error: %s", e)

    # =========================================================================
    # Console Commands
    # =========================================================================

    def cmd_ai_debug(self, entity_id: Optional[str] = None) -> str:
        """Console command: ai.debug [entity]"""
        if entity_id:
            return f"AI debug for {entity_id}: use ai.perception, ai.bt, ai.bb"
        return "Usage: ai.debug <entity> - Show AI debug for entity"

    def cmd_ai_pause(self) -> str:
        """Console command: ai.pause"""
        self.pause_ai()
        return "AI paused"

    def cmd_ai_resume(self) -> str:
        """Console command: ai.resume"""
        self.resume_ai()
        return "AI resumed"

    def cmd_ai_step(self) -> str:
        """Console command: ai.step"""
        self.step_ai()
        return "AI stepping"


# =============================================================================
# Singleton instance
# =============================================================================

_ai_debugger: Optional[AIDebugger] = None


def get_ai_debugger() -> AIDebugger:
    """Get the global AI debugger instance."""
    global _ai_debugger
    if _ai_debugger is None:
        _ai_debugger = AIDebugger()
    return _ai_debugger


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "AIDebugConfig",
    "AIDebugger",
    "AIDebugState",
    "BlackboardDebugInfo",
    "BTNodeDebugInfo",
    "get_ai_debugger",
    "PerceptionType",
    "PerceptionVisual",
]
