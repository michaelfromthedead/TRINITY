"""
AI Event Definitions for EventLog integration.

Defines typed AI events for behavior trees, GOAP, utility AI, and blackboard
that integrate with Foundation EventLog for replay and debugging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from foundation import Event, EventLog, get_event_log, get_current_tick


# =============================================================================
# AI Event Base
# =============================================================================


@dataclass
class AIEvent:
    """Base class for AI events with common fields."""
    entity_id: int
    timestamp: float = field(default_factory=time.time)

    def to_foundation_event(self, operation: str) -> Event:
        """Convert to a Foundation Event for recording."""
        return Event(
            tick=get_current_tick(),
            operation=operation,
            entity=self.entity_id,
            operation_args=self._to_args(),
        )

    def _to_args(self) -> Dict[str, Any]:
        """Convert event fields to operation args dict."""
        return {
            k: v for k, v in self.__dict__.items()
            if k not in ('entity_id', 'timestamp')
        }


# =============================================================================
# Behavior Tree Events
# =============================================================================


@dataclass
class BTNodeEntered(AIEvent):
    """Fired when a behavior tree node begins execution."""
    bt_name: str = ""
    node_name: str = ""
    node_type: str = ""

    def _to_args(self) -> Dict[str, Any]:
        return {
            'bt_name': self.bt_name,
            'node_name': self.node_name,
            'node_type': self.node_type,
            'timestamp': self.timestamp,
        }


@dataclass
class BTNodeExited(AIEvent):
    """Fired when a behavior tree node completes execution."""
    bt_name: str = ""
    node_name: str = ""
    result: str = ""  # SUCCESS, FAILURE, RUNNING

    def _to_args(self) -> Dict[str, Any]:
        return {
            'bt_name': self.bt_name,
            'node_name': self.node_name,
            'result': self.result,
            'timestamp': self.timestamp,
        }


# =============================================================================
# GOAP Events
# =============================================================================


@dataclass
class GOAPPlanCreated(AIEvent):
    """Fired when GOAP creates a new plan."""
    goal: str = ""
    actions: List[str] = field(default_factory=list)
    cost: float = 0.0

    def _to_args(self) -> Dict[str, Any]:
        return {
            'goal': self.goal,
            'actions': self.actions,
            'cost': self.cost,
            'timestamp': self.timestamp,
        }


@dataclass
class GOAPActionExecuted(AIEvent):
    """Fired when a GOAP action is executed."""
    action: str = ""
    success: bool = True

    def _to_args(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'success': self.success,
            'timestamp': self.timestamp,
        }


# =============================================================================
# Utility AI Events
# =============================================================================


@dataclass
class UtilityScoreComputed(AIEvent):
    """Fired when a utility consideration score is computed."""
    consideration: str = ""
    score: float = 0.0

    def _to_args(self) -> Dict[str, Any]:
        return {
            'consideration': self.consideration,
            'score': self.score,
            'timestamp': self.timestamp,
        }


@dataclass
class UtilityActionSelected(AIEvent):
    """Fired when a utility AI action is selected."""
    action: str = ""
    score: float = 0.0
    all_scores: Dict[str, float] = field(default_factory=dict)

    def _to_args(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'score': self.score,
            'all_scores': self.all_scores,
            'timestamp': self.timestamp,
        }


# =============================================================================
# Blackboard Events
# =============================================================================


@dataclass
class BlackboardValueChanged(AIEvent):
    """Fired when a blackboard value changes."""
    key: str = ""
    old_value: Any = None
    new_value: Any = None

    def _to_args(self) -> Dict[str, Any]:
        return {
            'key': self.key,
            'old_value': repr(self.old_value),
            'new_value': repr(self.new_value),
            'timestamp': self.timestamp,
        }


# =============================================================================
# Causal Chain Builder
# =============================================================================


class CausalChain:
    """
    Tracks causal relationships between AI events.

    Used to link parent BT nodes to child nodes for replay.
    """

    def __init__(self) -> None:
        self._chain: List[Event] = []
        self._parent_stack: List[str] = []

    def push_parent(self, operation: str) -> None:
        """Push a parent operation onto the stack."""
        self._parent_stack.append(operation)

    def pop_parent(self) -> Optional[str]:
        """Pop the current parent operation."""
        if self._parent_stack:
            return self._parent_stack.pop()
        return None

    def current_parent(self) -> Optional[str]:
        """Get the current parent operation."""
        return self._parent_stack[-1] if self._parent_stack else None

    def add_event(self, event: Event) -> None:
        """Add an event to the chain."""
        if self._parent_stack:
            event.immediate_parent = self._parent_stack[-1]
        self._chain.append(event)

    def get_chain(self) -> List[Event]:
        """Get all events in the chain."""
        return list(self._chain)

    def clear(self) -> None:
        """Clear the causal chain."""
        self._chain.clear()
        self._parent_stack.clear()


# =============================================================================
# AI EventLog Integration
# =============================================================================


class AIEventLogger:
    """
    Centralized logger for AI events.

    Integrates with Foundation EventLog while providing
    AI-specific helpers and causal chain tracking.
    """

    def __init__(self, event_log: Optional[EventLog] = None) -> None:
        self._event_log = event_log or get_event_log()
        self._causal_chain = CausalChain()
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Check if logging is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable logging."""
        self._enabled = value

    @property
    def causal_chain(self) -> CausalChain:
        """Get the causal chain tracker."""
        return self._causal_chain

    def log_bt_node_entered(
        self,
        entity_id: int,
        bt_name: str,
        node_name: str,
        node_type: str,
    ) -> None:
        """Log a behavior tree node entry event."""
        if not self._enabled:
            return

        event_data = BTNodeEntered(
            entity_id=entity_id,
            bt_name=bt_name,
            node_name=node_name,
            node_type=node_type,
        )
        foundation_event = event_data.to_foundation_event("BT.NodeEntered")
        foundation_event.immediate_parent = self._causal_chain.current_parent()
        self._causal_chain.push_parent(f"BT.{node_name}")
        self._event_log.record(foundation_event)

    def log_bt_node_exited(
        self,
        entity_id: int,
        bt_name: str,
        node_name: str,
        result: str,
    ) -> None:
        """Log a behavior tree node exit event."""
        if not self._enabled:
            return

        self._causal_chain.pop_parent()
        event_data = BTNodeExited(
            entity_id=entity_id,
            bt_name=bt_name,
            node_name=node_name,
            result=result,
        )
        foundation_event = event_data.to_foundation_event("BT.NodeExited")
        foundation_event.immediate_parent = self._causal_chain.current_parent()
        self._event_log.record(foundation_event)

    def log_goap_plan_created(
        self,
        entity_id: int,
        goal: str,
        actions: List[str],
        cost: float,
    ) -> None:
        """Log a GOAP plan creation event."""
        if not self._enabled:
            return

        event_data = GOAPPlanCreated(
            entity_id=entity_id,
            goal=goal,
            actions=actions,
            cost=cost,
        )
        foundation_event = event_data.to_foundation_event("GOAP.PlanCreated")
        self._event_log.record(foundation_event)

    def log_goap_action_executed(
        self,
        entity_id: int,
        action: str,
        success: bool,
    ) -> None:
        """Log a GOAP action execution event."""
        if not self._enabled:
            return

        event_data = GOAPActionExecuted(
            entity_id=entity_id,
            action=action,
            success=success,
        )
        foundation_event = event_data.to_foundation_event("GOAP.ActionExecuted")
        self._event_log.record(foundation_event)

    def log_utility_score_computed(
        self,
        entity_id: int,
        consideration: str,
        score: float,
    ) -> None:
        """Log a utility consideration score event."""
        if not self._enabled:
            return

        event_data = UtilityScoreComputed(
            entity_id=entity_id,
            consideration=consideration,
            score=score,
        )
        foundation_event = event_data.to_foundation_event("Utility.ScoreComputed")
        self._event_log.record(foundation_event)

    def log_utility_action_selected(
        self,
        entity_id: int,
        action: str,
        score: float,
        all_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log a utility action selection event."""
        if not self._enabled:
            return

        event_data = UtilityActionSelected(
            entity_id=entity_id,
            action=action,
            score=score,
            all_scores=all_scores or {},
        )
        foundation_event = event_data.to_foundation_event("Utility.ActionSelected")
        self._event_log.record(foundation_event)

    def log_blackboard_value_changed(
        self,
        entity_id: int,
        key: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        """Log a blackboard value change event."""
        if not self._enabled:
            return

        event_data = BlackboardValueChanged(
            entity_id=entity_id,
            key=key,
            old_value=old_value,
            new_value=new_value,
        )
        foundation_event = event_data.to_foundation_event("Blackboard.ValueChanged")
        self._event_log.record(foundation_event)

    def get_events_for_entity(self, entity_id: int) -> List[Event]:
        """Get all AI events for a specific entity."""
        return self._event_log.events_for_entity(entity_id)

    def get_bt_events(self) -> List[Event]:
        """Get all behavior tree events."""
        entered = self._event_log.events_for_operation("BT.NodeEntered")
        exited = self._event_log.events_for_operation("BT.NodeExited")
        return sorted(entered + exited, key=lambda e: e.tick)

    def get_goap_events(self) -> List[Event]:
        """Get all GOAP events."""
        plans = self._event_log.events_for_operation("GOAP.PlanCreated")
        actions = self._event_log.events_for_operation("GOAP.ActionExecuted")
        return sorted(plans + actions, key=lambda e: e.tick)

    def get_utility_events(self) -> List[Event]:
        """Get all utility AI events."""
        scores = self._event_log.events_for_operation("Utility.ScoreComputed")
        selections = self._event_log.events_for_operation("Utility.ActionSelected")
        return sorted(scores + selections, key=lambda e: e.tick)

    def get_blackboard_events(self) -> List[Event]:
        """Get all blackboard events."""
        return self._event_log.events_for_operation("Blackboard.ValueChanged")

    def clear(self) -> None:
        """Clear all events."""
        self._causal_chain.clear()


# Global AI event logger instance
_ai_event_logger: Optional[AIEventLogger] = None


def get_ai_event_logger() -> AIEventLogger:
    """Get the global AI event logger instance."""
    global _ai_event_logger
    if _ai_event_logger is None:
        _ai_event_logger = AIEventLogger()
    return _ai_event_logger


def set_ai_event_logger(logger: AIEventLogger) -> None:
    """Set the global AI event logger instance."""
    global _ai_event_logger
    _ai_event_logger = logger


# =============================================================================
# AI Decision Replay
# =============================================================================


class AIDecisionReplay:
    """
    Reconstructs AI decision paths from recorded events.

    Enables debugging by replaying the exact sequence of
    decisions that led to a particular outcome.
    """

    def __init__(self, event_log: Optional[EventLog] = None) -> None:
        self._event_log = event_log or get_event_log()

    def reconstruct_bt_execution(
        self,
        entity_id: int,
        start_tick: Optional[int] = None,
        end_tick: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct a behavior tree execution sequence.

        Returns a list of decision steps with node entries and exits.
        """
        events = self._event_log.events_for_entity(entity_id)

        # Filter by tick range
        if start_tick is not None:
            events = [e for e in events if e.tick >= start_tick]
        if end_tick is not None:
            events = [e for e in events if e.tick <= end_tick]

        # Filter to BT events only
        bt_events = [
            e for e in events
            if e.operation in ("BT.NodeEntered", "BT.NodeExited")
        ]

        # Build decision path
        path: List[Dict[str, Any]] = []
        for event in bt_events:
            step = {
                'tick': event.tick,
                'operation': event.operation,
                'node_name': event.operation_args.get('node_name', ''),
                'node_type': event.operation_args.get('node_type', ''),
                'result': event.operation_args.get('result', ''),
                'parent': event.immediate_parent,
            }
            path.append(step)

        return path

    def reconstruct_goap_plan_execution(
        self,
        entity_id: int,
        start_tick: Optional[int] = None,
        end_tick: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct a GOAP plan execution sequence.

        Returns the plan and each action execution.
        """
        events = self._event_log.events_for_entity(entity_id)

        # Filter by tick range
        if start_tick is not None:
            events = [e for e in events if e.tick >= start_tick]
        if end_tick is not None:
            events = [e for e in events if e.tick <= end_tick]

        # Filter to GOAP events only
        goap_events = [
            e for e in events
            if e.operation in ("GOAP.PlanCreated", "GOAP.ActionExecuted")
        ]

        # Build execution sequence
        sequence: List[Dict[str, Any]] = []
        for event in goap_events:
            step = {
                'tick': event.tick,
                'operation': event.operation,
            }
            if event.operation == "GOAP.PlanCreated":
                step['goal'] = event.operation_args.get('goal', '')
                step['actions'] = event.operation_args.get('actions', [])
                step['cost'] = event.operation_args.get('cost', 0.0)
            else:
                step['action'] = event.operation_args.get('action', '')
                step['success'] = event.operation_args.get('success', False)
            sequence.append(step)

        return sequence

    def reconstruct_utility_decision(
        self,
        entity_id: int,
        tick: int,
    ) -> Dict[str, Any]:
        """
        Reconstruct a utility AI decision at a specific tick.

        Returns the scores and selected action.
        """
        events = self._event_log.events_where(entity=entity_id, tick=tick)

        utility_events = [
            e for e in events
            if e.operation.startswith("Utility.")
        ]

        result: Dict[str, Any] = {
            'tick': tick,
            'scores': {},
            'selected_action': None,
            'selected_score': 0.0,
        }

        for event in utility_events:
            if event.operation == "Utility.ScoreComputed":
                consideration = event.operation_args.get('consideration', '')
                score = event.operation_args.get('score', 0.0)
                result['scores'][consideration] = score
            elif event.operation == "Utility.ActionSelected":
                result['selected_action'] = event.operation_args.get('action', '')
                result['selected_score'] = event.operation_args.get('score', 0.0)
                result['all_scores'] = event.operation_args.get('all_scores', {})

        return result

    def get_blackboard_history(
        self,
        entity_id: int,
        key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get the history of blackboard changes for an entity.

        Optionally filter by key.
        """
        events = self._event_log.events_for_entity(entity_id)
        bb_events = [
            e for e in events
            if e.operation == "Blackboard.ValueChanged"
        ]

        if key is not None:
            bb_events = [
                e for e in bb_events
                if e.operation_args.get('key') == key
            ]

        history: List[Dict[str, Any]] = []
        for event in bb_events:
            history.append({
                'tick': event.tick,
                'key': event.operation_args.get('key', ''),
                'old_value': event.operation_args.get('old_value'),
                'new_value': event.operation_args.get('new_value'),
                'timestamp': event.operation_args.get('timestamp'),
            })

        return history


__all__ = [
    # Event types
    "AIEvent",
    "BTNodeEntered",
    "BTNodeExited",
    "GOAPPlanCreated",
    "GOAPActionExecuted",
    "UtilityScoreComputed",
    "UtilityActionSelected",
    "BlackboardValueChanged",
    # Causal chain
    "CausalChain",
    # Logger
    "AIEventLogger",
    "get_ai_event_logger",
    "set_ai_event_logger",
    # Replay
    "AIDecisionReplay",
]
