"""
Automated gameplay testing with bots.

Provides infrastructure for automated playtesting using
AI-controlled bots and scripted behaviors.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union


class BotActionType(Enum):
    """Types of bot actions."""

    MOVE = auto()
    JUMP = auto()
    ATTACK = auto()
    USE = auto()
    INTERACT = auto()
    WAIT = auto()
    LOOK = auto()
    CROUCH = auto()
    SPRINT = auto()
    RELOAD = auto()
    SWITCH_WEAPON = auto()
    OPEN_MENU = auto()
    NAVIGATE_TO = auto()
    CUSTOM = auto()


@dataclass
class BotAction:
    """
    An action that a bot can perform.

    Represents a single atomic action in the game.
    """

    action_type: BotActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.action_type.name,
            "parameters": self.parameters,
            "duration": self.duration,
            "weight": self.weight,
        }

    @classmethod
    def move(cls, direction: Tuple[float, float, float], duration: float = 0.1) -> "BotAction":
        """Create a move action."""
        return cls(
            action_type=BotActionType.MOVE,
            parameters={"direction": direction},
            duration=duration,
        )

    @classmethod
    def jump(cls) -> "BotAction":
        """Create a jump action."""
        return cls(action_type=BotActionType.JUMP, duration=0.5)

    @classmethod
    def attack(cls, target: Optional[int] = None) -> "BotAction":
        """Create an attack action."""
        return cls(
            action_type=BotActionType.ATTACK,
            parameters={"target": target},
            duration=0.2,
        )

    @classmethod
    def navigate_to(cls, position: Tuple[float, float, float]) -> "BotAction":
        """Create a navigation action."""
        return cls(
            action_type=BotActionType.NAVIGATE_TO,
            parameters={"position": position},
        )

    @classmethod
    def wait(cls, duration: float) -> "BotAction":
        """Create a wait action."""
        return cls(action_type=BotActionType.WAIT, duration=duration)

    @classmethod
    def custom(cls, name: str, **parameters) -> "BotAction":
        """Create a custom action."""
        return cls(
            action_type=BotActionType.CUSTOM,
            parameters={"name": name, **parameters},
        )


class BotBehavior(ABC):
    """
    Base class for bot behaviors.

    Defines how a bot decides what actions to take.
    """

    name: str = "base"

    @abstractmethod
    def get_next_action(self, bot: "GameBot", world_state: Dict[str, Any]) -> BotAction:
        """Determine the next action for the bot."""
        pass

    def on_start(self, bot: "GameBot") -> None:
        """Called when behavior starts."""
        pass

    def on_end(self, bot: "GameBot") -> None:
        """Called when behavior ends."""
        pass


class RandomWalkBehavior(BotBehavior):
    """Bot behavior that walks randomly."""

    name = "random_walk"

    def __init__(self, move_duration: float = 1.0):
        self.move_duration = move_duration

    def get_next_action(self, bot: "GameBot", world_state: Dict[str, Any]) -> BotAction:
        """Pick a random direction and move."""
        direction = (
            random.uniform(-1, 1),
            0,
            random.uniform(-1, 1),
        )
        return BotAction.move(direction, self.move_duration)


class ExplorationBehavior(BotBehavior):
    """Bot behavior that explores the map."""

    name = "exploration"

    def __init__(self):
        self._visited_positions: Set[Tuple[int, int, int]] = set()
        self._target_position: Optional[Tuple[float, float, float]] = None

    def get_next_action(self, bot: "GameBot", world_state: Dict[str, Any]) -> BotAction:
        """Navigate to unexplored areas."""
        current_pos = world_state.get("position", (0, 0, 0))
        grid_pos = (int(current_pos[0]), int(current_pos[1]), int(current_pos[2]))
        self._visited_positions.add(grid_pos)

        # Find new target if needed
        if self._target_position is None or self._reached_target(current_pos):
            self._target_position = self._find_exploration_target(world_state)

        if self._target_position:
            return BotAction.navigate_to(self._target_position)

        return BotAction.wait(0.5)

    def _reached_target(self, current_pos: Tuple[float, float, float]) -> bool:
        """Check if target reached."""
        if self._target_position is None:
            return True
        distance = sum((a - b) ** 2 for a, b in zip(current_pos, self._target_position)) ** 0.5
        return distance < 1.0

    def _find_exploration_target(
        self,
        world_state: Dict[str, Any],
    ) -> Optional[Tuple[float, float, float]]:
        """Find a new exploration target."""
        # Simple random target within bounds
        bounds = world_state.get("world_bounds", ((-100, 100), (0, 10), (-100, 100)))
        return (
            random.uniform(bounds[0][0], bounds[0][1]),
            random.uniform(bounds[1][0], bounds[1][1]),
            random.uniform(bounds[2][0], bounds[2][1]),
        )


class CombatBehavior(BotBehavior):
    """Bot behavior for combat testing."""

    name = "combat"

    def __init__(self, aggression: float = 0.8):
        self.aggression = aggression

    def get_next_action(self, bot: "GameBot", world_state: Dict[str, Any]) -> BotAction:
        """Engage in combat with nearby enemies."""
        enemies = world_state.get("nearby_enemies", [])

        if enemies and random.random() < self.aggression:
            target = enemies[0]
            return BotAction.attack(target.get("id"))

        # Move towards enemies if none in range
        if enemies:
            target_pos = enemies[0].get("position")
            if target_pos:
                return BotAction.navigate_to(target_pos)

        # Random movement when no enemies
        return BotAction.move((random.uniform(-1, 1), 0, random.uniform(-1, 1)))


class ScriptedBehavior(BotBehavior):
    """Bot behavior that follows a script."""

    name = "scripted"

    def __init__(self, actions: List[BotAction]):
        self.actions = actions
        self._current_index = 0

    def get_next_action(self, bot: "GameBot", world_state: Dict[str, Any]) -> BotAction:
        """Return the next scripted action."""
        if self._current_index >= len(self.actions):
            self._current_index = 0  # Loop

        action = self.actions[self._current_index]
        self._current_index += 1
        return action


class GameBot:
    """
    A game bot for automated testing.

    Simulates player input and behavior for testing purposes.
    """

    def __init__(
        self,
        name: str,
        behavior: BotBehavior,
        entity_id: Optional[int] = None,
    ):
        self.name = name
        self.behavior = behavior
        self.entity_id = entity_id
        self._active = False
        self._actions_performed: List[BotAction] = []
        self._metrics: Dict[str, Any] = {
            "actions_count": 0,
            "distance_traveled": 0.0,
            "enemies_killed": 0,
            "deaths": 0,
            "time_active": 0.0,
        }
        self._last_position: Optional[Tuple[float, float, float]] = None

    def start(self) -> None:
        """Start the bot."""
        self._active = True
        self.behavior.on_start(self)

    def stop(self) -> None:
        """Stop the bot."""
        self._active = False
        self.behavior.on_end(self)

    @property
    def is_active(self) -> bool:
        """Check if bot is active."""
        return self._active

    def update(self, world_state: Dict[str, Any], delta_time: float) -> Optional[BotAction]:
        """
        Update the bot and get next action.

        Args:
            world_state: Current world state
            delta_time: Time since last update

        Returns:
            Next action to perform, or None
        """
        if not self._active:
            return None

        # Update metrics
        self._metrics["time_active"] += delta_time
        current_pos = world_state.get("position")
        if current_pos and self._last_position:
            distance = sum(
                (a - b) ** 2 for a, b in zip(current_pos, self._last_position)
            ) ** 0.5
            self._metrics["distance_traveled"] += distance
        self._last_position = current_pos

        # Get next action from behavior
        action = self.behavior.get_next_action(self, world_state)

        if action:
            self._actions_performed.append(action)
            self._metrics["actions_count"] += 1

        return action

    def record_kill(self) -> None:
        """Record an enemy kill."""
        self._metrics["enemies_killed"] += 1

    def record_death(self) -> None:
        """Record a death."""
        self._metrics["deaths"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get bot metrics."""
        return self._metrics.copy()

    def get_action_history(self) -> List[BotAction]:
        """Get action history."""
        return self._actions_performed.copy()


class BotController:
    """
    Controller for managing multiple bots.

    Coordinates bot spawning, updating, and metrics collection.
    """

    def __init__(self):
        self._bots: Dict[str, GameBot] = {}
        self._behaviors: Dict[str, Type[BotBehavior]] = {
            "random_walk": RandomWalkBehavior,
            "exploration": ExplorationBehavior,
            "combat": CombatBehavior,
            "scripted": ScriptedBehavior,
        }

    def register_behavior(self, name: str, behavior_class: Type[BotBehavior]) -> None:
        """Register a custom behavior."""
        self._behaviors[name] = behavior_class

    def create_bot(
        self,
        name: str,
        behavior_type: str = "random_walk",
        behavior_args: Optional[Dict[str, Any]] = None,
    ) -> GameBot:
        """Create and register a bot."""
        behavior_class = self._behaviors.get(behavior_type, RandomWalkBehavior)
        behavior = behavior_class(**(behavior_args or {}))

        bot = GameBot(name=name, behavior=behavior)
        self._bots[name] = bot
        return bot

    def get_bot(self, name: str) -> Optional[GameBot]:
        """Get a bot by name."""
        return self._bots.get(name)

    def remove_bot(self, name: str) -> None:
        """Remove a bot."""
        if name in self._bots:
            self._bots[name].stop()
            del self._bots[name]

    def start_all(self) -> None:
        """Start all bots."""
        for bot in self._bots.values():
            bot.start()

    def stop_all(self) -> None:
        """Stop all bots."""
        for bot in self._bots.values():
            bot.stop()

    def update_all(
        self,
        world_state: Dict[str, Any],
        delta_time: float,
    ) -> Dict[str, Optional[BotAction]]:
        """Update all bots and return their actions."""
        actions = {}
        for name, bot in self._bots.items():
            actions[name] = bot.update(world_state, delta_time)
        return actions

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all bots."""
        return {name: bot.get_metrics() for name, bot in self._bots.items()}


@dataclass
class PlaytestEvent:
    """An event recorded during playtesting."""

    timestamp: float
    event_type: str
    bot_name: str
    data: Dict[str, Any] = field(default_factory=dict)


class PlaytestRecorder:
    """
    Records playtest events and actions.

    Captures bot actions, game events, and metrics during testing.
    """

    def __init__(self):
        self._events: List[PlaytestEvent] = []
        self._start_time: float = 0.0
        self._recording = False

    def start(self) -> None:
        """Start recording."""
        self._start_time = time.time()
        self._recording = True

    def stop(self) -> None:
        """Stop recording."""
        self._recording = False

    def record_event(
        self,
        event_type: str,
        bot_name: str,
        **data,
    ) -> None:
        """Record an event."""
        if not self._recording:
            return

        event = PlaytestEvent(
            timestamp=time.time() - self._start_time,
            event_type=event_type,
            bot_name=bot_name,
            data=data,
        )
        self._events.append(event)

    def record_action(self, bot_name: str, action: BotAction) -> None:
        """Record a bot action."""
        self.record_event(
            "action",
            bot_name,
            action_type=action.action_type.name,
            parameters=action.parameters,
        )

    def get_events(self) -> List[PlaytestEvent]:
        """Get all recorded events."""
        return self._events.copy()

    def export(self, path: str) -> None:
        """Export events to file."""
        import json

        data = {
            "events": [
                {
                    "timestamp": e.timestamp,
                    "type": e.event_type,
                    "bot": e.bot_name,
                    "data": e.data,
                }
                for e in self._events
            ]
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


@dataclass
class PlaytestSession:
    """
    A playtest session with bots.

    Manages a complete playtest session including setup,
    execution, and reporting.
    """

    name: str
    duration: float = 60.0  # Seconds
    bot_count: int = 1
    behavior_type: str = "exploration"
    seed: Optional[int] = None

    _controller: Optional[BotController] = field(default=None, repr=False)
    _recorder: Optional[PlaytestRecorder] = field(default=None, repr=False)
    _results: Dict[str, Any] = field(default_factory=dict, repr=False)

    def setup(self) -> None:
        """Set up the playtest session."""
        if self.seed is not None:
            random.seed(self.seed)

        self._controller = BotController()
        self._recorder = PlaytestRecorder()

        # Create bots
        for i in range(self.bot_count):
            self._controller.create_bot(
                name=f"Bot_{i}",
                behavior_type=self.behavior_type,
            )

    def run(self, world_update_func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run the playtest session.

        Args:
            world_update_func: Function to update world state

        Returns:
            Session results
        """
        if not self._controller:
            self.setup()

        self._recorder.start()
        self._controller.start_all()

        start_time = time.time()
        world_state = {}
        delta_time = 1 / 60  # 60 FPS

        while time.time() - start_time < self.duration:
            # Update bots
            actions = self._controller.update_all(world_state, delta_time)

            # Record actions
            for bot_name, action in actions.items():
                if action:
                    self._recorder.record_action(bot_name, action)

            # Update world (would integrate with game engine)
            world_state = world_update_func(world_state)

            time.sleep(delta_time)

        self._controller.stop_all()
        self._recorder.stop()

        # Collect results
        self._results = {
            "name": self.name,
            "duration": time.time() - start_time,
            "bot_count": self.bot_count,
            "metrics": self._controller.get_all_metrics(),
            "event_count": len(self._recorder.get_events()),
        }

        return self._results

    def get_results(self) -> Dict[str, Any]:
        """Get session results."""
        return self._results.copy()


class PlaytestReporter:
    """
    Generates reports from playtest sessions.

    Creates human-readable reports and analytics from playtest data.
    """

    def __init__(self):
        self._sessions: List[Dict[str, Any]] = []

    def add_session(self, results: Dict[str, Any]) -> None:
        """Add session results."""
        self._sessions.append(results)

    def generate_report(self) -> str:
        """Generate a text report."""
        lines = ["Playtest Report", "=" * 50, ""]

        for i, session in enumerate(self._sessions):
            lines.append(f"Session {i + 1}: {session.get('name', 'Unnamed')}")
            lines.append(f"  Duration: {session.get('duration', 0):.1f}s")
            lines.append(f"  Bots: {session.get('bot_count', 0)}")
            lines.append(f"  Events: {session.get('event_count', 0)}")

            metrics = session.get("metrics", {})
            for bot_name, bot_metrics in metrics.items():
                lines.append(f"  {bot_name}:")
                lines.append(f"    Actions: {bot_metrics.get('actions_count', 0)}")
                lines.append(f"    Distance: {bot_metrics.get('distance_traveled', 0):.1f}")
                lines.append(f"    Kills: {bot_metrics.get('enemies_killed', 0)}")
                lines.append(f"    Deaths: {bot_metrics.get('deaths', 0)}")

            lines.append("")

        return "\n".join(lines)

    def export_json(self, path: str) -> None:
        """Export report as JSON."""
        import json

        with open(path, "w") as f:
            json.dump({"sessions": self._sessions}, f, indent=2)


def create_bot(
    name: str,
    behavior: Union[str, BotBehavior] = "random_walk",
    **kwargs,
) -> GameBot:
    """
    Create a game bot.

    Args:
        name: Bot name
        behavior: Behavior type or instance
        **kwargs: Additional bot parameters

    Returns:
        Created bot
    """
    if isinstance(behavior, str):
        behaviors = {
            "random_walk": RandomWalkBehavior,
            "exploration": ExplorationBehavior,
            "combat": CombatBehavior,
        }
        behavior_class = behaviors.get(behavior, RandomWalkBehavior)
        behavior = behavior_class()

    return GameBot(name=name, behavior=behavior, **kwargs)


def run_playtest(
    duration: float = 60.0,
    bot_count: int = 1,
    behavior: str = "exploration",
    world_update_func: Optional[Callable] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a playtest session.

    Args:
        duration: Session duration in seconds
        bot_count: Number of bots
        behavior: Bot behavior type
        world_update_func: World update function
        **kwargs: Additional session parameters

    Returns:
        Session results
    """
    if world_update_func is None:
        # Default no-op world update
        world_update_func = lambda state: state

    session = PlaytestSession(
        name=kwargs.get("name", "Playtest"),
        duration=duration,
        bot_count=bot_count,
        behavior_type=behavior,
    )

    return session.run(world_update_func)
