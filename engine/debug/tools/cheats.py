"""
Cheat System - God mode, teleport, spawn, and other debug cheats.

Provides a CheatManager for registering and executing cheat commands
that integrate with the console system.

SECURITY NOTE: This module contains debug functionality that should ONLY
be available in non-shipping builds. Use require_debug_build() to guard
access in production code.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

if TYPE_CHECKING:
    from engine.core.math.vec import Vec3

logger = logging.getLogger(__name__)


# =============================================================================
# Build Configuration
# =============================================================================

@dataclass
class CheatConfig:
    """Configuration for cheat system behavior and limits."""
    # Enable/disable cheats globally based on build type
    allow_in_shipping: bool = False
    allow_in_demo: bool = False
    allow_in_development: bool = True
    allow_in_debug: bool = True

    # Multiplayer restrictions
    allow_in_multiplayer: bool = False

    # Logging
    log_all_cheats: bool = True


# Global configuration - can be overridden by build system
_cheat_config = CheatConfig()


def get_cheat_config() -> CheatConfig:
    """Get the global cheat configuration."""
    return _cheat_config


def set_cheat_config(config: CheatConfig) -> None:
    """Set the global cheat configuration."""
    global _cheat_config
    _cheat_config = config


def is_shipping_build() -> bool:
    """
    Check if this is a shipping build.

    Checks environment variables and build configuration to determine
    if cheats should be disabled.
    """
    # Check environment variable (set by build system)
    if os.environ.get("GAME_BUILD_TYPE", "").upper() == "SHIPPING":
        return True
    if os.environ.get("NDEBUG") == "1":
        return True
    if os.environ.get("SHIPPING") == "1":
        return True

    # Try to import build config if available
    try:
        from engine.tooling.build.build_config import (
            ConfigurationManager,
            ConfigurationPreset,
        )
        # Check active build configuration
        manager = ConfigurationManager()
        active = manager.get_active()
        if active and active.preset == ConfigurationPreset.SHIPPING:
            return True
    except ImportError:
        pass

    return False


def is_debug_build() -> bool:
    """Check if this is a debug build."""
    if os.environ.get("GAME_BUILD_TYPE", "").upper() == "DEBUG":
        return True
    if os.environ.get("DEBUG") == "1":
        return True
    if os.environ.get("_DEBUG") == "1":
        return True
    return False


def cheats_allowed() -> bool:
    """
    Check if cheats are allowed in the current build.

    Returns False for shipping builds unless explicitly overridden.
    """
    config = get_cheat_config()

    if is_shipping_build():
        return config.allow_in_shipping

    if is_debug_build():
        return config.allow_in_debug

    # Default to development behavior
    return config.allow_in_development


def require_debug_build(func: Callable) -> Callable:
    """
    Decorator that prevents function execution in shipping builds.

    Use this to guard cheat-related functions that should never
    execute in production.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not cheats_allowed():
            logger.warning(
                "Cheat function '%s' blocked - not allowed in this build",
                func.__name__
            )
            return None
        return func(*args, **kwargs)

    return wrapper


class CheatFlags(Flag):
    """Flags controlling cheat behavior and requirements."""
    NONE = 0
    REQUIRES_DEBUG_BUILD = auto()
    REQUIRES_DEVELOPER = auto()
    REQUIRES_ADMIN = auto()
    PERSISTENT = auto()  # Survives level changes
    REPLICATED = auto()  # Syncs across network
    LOGGED = auto()  # Always log activation
    DISABLED_IN_MULTIPLAYER = auto()


@dataclass
class CheatCommand:
    """A registered cheat command."""
    name: str
    handler: Callable[..., Any]
    description: str = ""
    flags: CheatFlags = CheatFlags.NONE
    aliases: List[str] = field(default_factory=list)
    parameters: List[Tuple[str, type, Optional[Any]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Cheat command name cannot be empty")


@dataclass
class CheatState:
    """
    Current state of cheat modes for an entity.

    All state changes are tracked and can be observed via callbacks
    registered on the CheatManager.
    """
    god_mode: bool = False
    fly_mode: bool = False
    ghost_mode: bool = False
    infinite_ammo: bool = False
    invisible: bool = False
    speed_multiplier: float = 1.0
    damage_multiplier: float = 1.0

    def is_any_active(self) -> bool:
        """Check if any cheat mode is currently active."""
        return (
            self.god_mode or
            self.fly_mode or
            self.ghost_mode or
            self.infinite_ammo or
            self.invisible or
            self.speed_multiplier != 1.0 or
            self.damage_multiplier != 1.0
        )


class CheatManager:
    """
    Manages cheat commands and cheat state for debugging.

    SECURITY: This class automatically disables itself in shipping builds.
    Use cheats_allowed() to check if cheats are available.

    Integrates with the console system to provide commands like:
    - god: Toggle invulnerability
    - fly: Toggle flying/noclip mode
    - ghost: Toggle collision
    - teleport X Y Z: Move to location
    - spawn Actor: Create entity
    - kill [Target]: Destroy entity
    - sethealth Value: Modify health
    - give Item [Count]: Add to inventory
    - infiniteammo: Toggle infinite ammo
    """

    def __init__(self) -> None:
        self._commands: Dict[str, CheatCommand] = {}
        self._aliases: Dict[str, str] = {}
        self._entity_states: Dict[Any, CheatState] = {}
        self._global_state = CheatState()
        self._enabled = True
        self._is_multiplayer = False
        self._callbacks: Dict[str, List[Callable[..., None]]] = {}

        # Apply build-type restrictions
        if not cheats_allowed():
            self._enabled = False
            logger.info("CheatManager disabled - not allowed in this build type")

        # Register built-in cheats
        self._register_builtin_cheats()

    def _register_builtin_cheats(self) -> None:
        """Register all built-in cheat commands."""
        self.register(
            CheatCommand(
                name="god",
                handler=self._cmd_god,
                description="Toggle invulnerability",
                flags=CheatFlags.LOGGED,
                aliases=["godmode", "invincible"],
            )
        )
        self.register(
            CheatCommand(
                name="fly",
                handler=self._cmd_fly,
                description="Toggle flying mode (pass through geometry)",
                flags=CheatFlags.LOGGED,
                aliases=["noclip", "flymode"],
            )
        )
        self.register(
            CheatCommand(
                name="ghost",
                handler=self._cmd_ghost,
                description="Toggle ghost mode (no collision)",
                flags=CheatFlags.LOGGED,
                aliases=["nocollision", "ghostmode"],
            )
        )
        self.register(
            CheatCommand(
                name="teleport",
                handler=self._cmd_teleport,
                description="Teleport to location",
                flags=CheatFlags.LOGGED,
                aliases=["tp", "goto"],
                parameters=[
                    ("x", float, None),
                    ("y", float, None),
                    ("z", float, None),
                ],
            )
        )
        self.register(
            CheatCommand(
                name="spawn",
                handler=self._cmd_spawn,
                description="Spawn an entity at current location",
                flags=CheatFlags.LOGGED,
                aliases=["create", "summon"],
                parameters=[
                    ("actor_type", str, None),
                    ("x", float, None),
                    ("y", float, None),
                    ("z", float, None),
                ],
            )
        )
        self.register(
            CheatCommand(
                name="kill",
                handler=self._cmd_kill,
                description="Kill/destroy target entity",
                flags=CheatFlags.LOGGED,
                aliases=["destroy", "remove"],
                parameters=[
                    ("target", str, "self"),
                ],
            )
        )
        self.register(
            CheatCommand(
                name="sethealth",
                handler=self._cmd_set_health,
                description="Set health value",
                flags=CheatFlags.LOGGED,
                aliases=["hp", "health"],
                parameters=[
                    ("value", float, None),
                ],
            )
        )
        self.register(
            CheatCommand(
                name="give",
                handler=self._cmd_give_item,
                description="Give item to inventory",
                flags=CheatFlags.LOGGED,
                aliases=["additem", "giveitem"],
                parameters=[
                    ("item_type", str, None),
                    ("count", int, 1),
                ],
            )
        )
        self.register(
            CheatCommand(
                name="infiniteammo",
                handler=self._cmd_infinite_ammo,
                description="Toggle infinite ammo",
                flags=CheatFlags.LOGGED | CheatFlags.PERSISTENT,
                aliases=["unlimitedammo", "ammo"],
            )
        )

    @property
    def enabled(self) -> bool:
        """Check if cheats are enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """
        Enable or disable cheats.

        Note: Cannot enable cheats in shipping builds even if explicitly requested.
        """
        if value and not cheats_allowed():
            logger.warning("Cannot enable cheats - not allowed in this build type")
            return

        self._enabled = value
        if not value:
            # Disable all active cheats
            self._global_state = CheatState()
            self._entity_states.clear()
            logger.info("Cheats disabled - all cheat states reset")

    def set_multiplayer_mode(self, is_multiplayer: bool) -> None:
        """
        Set whether the game is in multiplayer mode.

        Many cheats are disabled in multiplayer to prevent unfair advantages.
        """
        self._is_multiplayer = is_multiplayer
        config = get_cheat_config()

        if is_multiplayer and not config.allow_in_multiplayer:
            logger.info("Multiplayer mode enabled - cheats restricted")

    @property
    def is_multiplayer(self) -> bool:
        """Check if currently in multiplayer mode."""
        return self._is_multiplayer

    def register(self, command: CheatCommand) -> None:
        """Register a cheat command."""
        if command.name in self._commands:
            logger.warning("Overwriting existing cheat command: %s", command.name)

        self._commands[command.name] = command

        for alias in command.aliases:
            self._aliases[alias] = command.name

        logger.debug("Registered cheat: %s", command.name)

    def unregister(self, name: str) -> bool:
        """Unregister a cheat command."""
        if name not in self._commands:
            return False

        command = self._commands[name]
        del self._commands[name]

        for alias in command.aliases:
            if alias in self._aliases:
                del self._aliases[alias]

        return True

    def execute(
        self,
        command: str,
        *args: Any,
        entity: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[bool, str]:
        """
        Execute a cheat command.

        Returns:
            Tuple of (success, message)
        """
        if not self._enabled:
            return False, "Cheats are disabled"

        # Resolve alias
        cmd_name = self._aliases.get(command, command)

        if cmd_name not in self._commands:
            return False, f"Unknown cheat command: {command}"

        cmd = self._commands[cmd_name]

        # Check flags
        if CheatFlags.REQUIRES_DEBUG_BUILD in cmd.flags:
            if not is_debug_build():
                return False, f"Cheat '{cmd_name}' requires debug build"

        config = get_cheat_config()
        if CheatFlags.DISABLED_IN_MULTIPLAYER in cmd.flags:
            if self._is_multiplayer and not config.allow_in_multiplayer:
                return False, f"Cheat '{cmd_name}' is disabled in multiplayer"

        try:
            result = cmd.handler(*args, entity=entity, **kwargs)

            if CheatFlags.LOGGED in cmd.flags:
                logger.info(
                    "Cheat executed: %s %s (entity=%s)",
                    cmd_name,
                    args,
                    entity,
                )

            # Notify callbacks
            self._notify_callbacks(cmd_name, entity, args, kwargs)

            return True, str(result) if result else "OK"
        except Exception as e:
            logger.error("Cheat execution failed: %s - %s", cmd_name, e)
            return False, str(e)

    def get_state(self, entity: Optional[Any] = None) -> CheatState:
        """Get cheat state for an entity or global state."""
        if entity is None:
            return self._global_state
        return self._entity_states.get(entity, CheatState())

    def _get_or_create_state(self, entity: Optional[Any]) -> CheatState:
        """Get or create cheat state for an entity."""
        if entity is None:
            return self._global_state
        if entity not in self._entity_states:
            self._entity_states[entity] = CheatState()
        return self._entity_states[entity]

    def add_callback(
        self,
        cheat_name: str,
        callback: Callable[..., None],
    ) -> None:
        """Add a callback for when a cheat is executed."""
        if cheat_name not in self._callbacks:
            self._callbacks[cheat_name] = []
        self._callbacks[cheat_name].append(callback)

    def remove_callback(
        self,
        cheat_name: str,
        callback: Callable[..., None],
    ) -> bool:
        """Remove a cheat callback."""
        if cheat_name not in self._callbacks:
            return False
        try:
            self._callbacks[cheat_name].remove(callback)
            return True
        except ValueError:
            return False

    def _notify_callbacks(
        self,
        cheat_name: str,
        entity: Optional[Any],
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Notify callbacks of cheat execution."""
        callbacks = self._callbacks.get(cheat_name, [])
        for callback in callbacks:
            try:
                callback(cheat_name, entity, args, kwargs)
            except Exception as e:
                logger.error("Cheat callback error: %s", e)

    def list_commands(self) -> List[CheatCommand]:
        """List all registered cheat commands."""
        return list(self._commands.values())

    # =========================================================================
    # Built-in Command Handlers
    # =========================================================================

    def god_mode(self, enabled: bool, entity: Optional[Any] = None) -> bool:
        """Enable or disable god mode (invulnerability)."""
        state = self._get_or_create_state(entity)
        state.god_mode = enabled
        return enabled

    def fly_mode(self, enabled: bool, entity: Optional[Any] = None) -> bool:
        """Enable or disable fly mode (pass through geometry)."""
        state = self._get_or_create_state(entity)
        state.fly_mode = enabled
        return enabled

    def ghost_mode(self, enabled: bool, entity: Optional[Any] = None) -> bool:
        """Enable or disable ghost mode (no collision)."""
        state = self._get_or_create_state(entity)
        state.ghost_mode = enabled
        return enabled

    def teleport(
        self,
        x: float,
        y: float,
        z: float,
        entity: Optional[Any] = None,
    ) -> Tuple[float, float, float]:
        """
        Teleport entity to location.

        Returns the target position.
        """
        # Actual teleportation would be handled by the entity system
        # This just validates and returns the target
        logger.info("Teleport: entity=%s -> (%f, %f, %f)", entity, x, y, z)
        return (x, y, z)

    def spawn(
        self,
        actor_type: str,
        position: Optional[Tuple[float, float, float]] = None,
    ) -> Optional[Any]:
        """
        Spawn an entity at the given position.

        Returns the spawned entity or None if failed.
        """
        # Actual spawning would be handled by the entity system
        logger.info("Spawn: type=%s, position=%s", actor_type, position)
        return None  # Placeholder - would return spawned entity

    def kill(self, target: Any) -> bool:
        """
        Kill/destroy the target entity.

        Returns True if destroyed.
        """
        # Actual destruction would be handled by the entity system
        logger.info("Kill: target=%s", target)
        return True

    def set_health(self, value: float, entity: Optional[Any] = None) -> float:
        """
        Set health value for entity.

        Returns the new health value.
        """
        # Actual health modification would be handled by health system
        logger.info("SetHealth: entity=%s, value=%f", entity, value)
        return value

    def give_item(
        self,
        item_type: str,
        count: int = 1,
        entity: Optional[Any] = None,
    ) -> int:
        """
        Give items to entity's inventory.

        Returns the number of items given.
        """
        # Actual item giving would be handled by inventory system
        logger.info(
            "GiveItem: entity=%s, item=%s, count=%d",
            entity,
            item_type,
            count,
        )
        return count

    def infinite_ammo(
        self,
        enabled: bool,
        entity: Optional[Any] = None,
    ) -> bool:
        """Enable or disable infinite ammo."""
        state = self._get_or_create_state(entity)
        state.infinite_ammo = enabled
        return enabled

    # =========================================================================
    # Console Command Handlers (wrap public methods)
    # =========================================================================

    def _cmd_god(
        self,
        enabled: Optional[bool] = None,
        entity: Optional[Any] = None,
    ) -> str:
        """Console handler for god command."""
        state = self.get_state(entity)
        if enabled is None:
            enabled = not state.god_mode
        self.god_mode(enabled, entity)
        return f"God mode {'enabled' if enabled else 'disabled'}"

    def _cmd_fly(
        self,
        enabled: Optional[bool] = None,
        entity: Optional[Any] = None,
    ) -> str:
        """Console handler for fly command."""
        state = self.get_state(entity)
        if enabled is None:
            enabled = not state.fly_mode
        self.fly_mode(enabled, entity)
        return f"Fly mode {'enabled' if enabled else 'disabled'}"

    def _cmd_ghost(
        self,
        enabled: Optional[bool] = None,
        entity: Optional[Any] = None,
    ) -> str:
        """Console handler for ghost command."""
        state = self.get_state(entity)
        if enabled is None:
            enabled = not state.ghost_mode
        self.ghost_mode(enabled, entity)
        return f"Ghost mode {'enabled' if enabled else 'disabled'}"

    def _cmd_teleport(
        self,
        x: float,
        y: float,
        z: float,
        entity: Optional[Any] = None,
    ) -> str:
        """Console handler for teleport command."""
        pos = self.teleport(x, y, z, entity)
        return f"Teleported to ({pos[0]}, {pos[1]}, {pos[2]})"

    def _cmd_spawn(
        self,
        actor_type: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        entity: Optional[Any] = None,
    ) -> str:
        """Console handler for spawn command."""
        position = None
        if x is not None and y is not None and z is not None:
            position = (x, y, z)
        result = self.spawn(actor_type, position)
        return f"Spawned {actor_type}" if result else f"Failed to spawn {actor_type}"

    def _cmd_kill(
        self,
        target: str = "self",
        entity: Optional[Any] = None,
    ) -> str:
        """Console handler for kill command."""
        actual_target = entity if target == "self" else target
        if self.kill(actual_target):
            return f"Killed {target}"
        return f"Failed to kill {target}"

    def _cmd_set_health(
        self,
        value: float,
        entity: Optional[Any] = None,
    ) -> str:
        """Console handler for sethealth command."""
        new_health = self.set_health(value, entity)
        return f"Health set to {new_health}"

    def _cmd_give_item(
        self,
        item_type: str,
        count: int = 1,
        entity: Optional[Any] = None,
    ) -> str:
        """Console handler for give command."""
        given = self.give_item(item_type, count, entity)
        return f"Gave {given}x {item_type}"

    def _cmd_infinite_ammo(
        self,
        enabled: Optional[bool] = None,
        entity: Optional[Any] = None,
    ) -> str:
        """Console handler for infiniteammo command."""
        state = self.get_state(entity)
        if enabled is None:
            enabled = not state.infinite_ammo
        self.infinite_ammo(enabled, entity)
        return f"Infinite ammo {'enabled' if enabled else 'disabled'}"


# =============================================================================
# Singleton instance
# =============================================================================

_cheat_manager: Optional[CheatManager] = None


def get_cheat_manager() -> CheatManager:
    """
    Get the global cheat manager instance.

    Note: In shipping builds, the manager will be disabled by default
    and cheat execution will be blocked.
    """
    global _cheat_manager
    if _cheat_manager is None:
        _cheat_manager = CheatManager()
    return _cheat_manager


def reset_cheat_manager() -> None:
    """
    Reset the global cheat manager.

    Primarily used for testing purposes.
    """
    global _cheat_manager
    _cheat_manager = None


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "CheatCommand",
    "CheatConfig",
    "CheatFlags",
    "CheatManager",
    "CheatState",
    "cheats_allowed",
    "get_cheat_config",
    "get_cheat_manager",
    "is_debug_build",
    "is_shipping_build",
    "require_debug_build",
    "reset_cheat_manager",
    "set_cheat_config",
]
