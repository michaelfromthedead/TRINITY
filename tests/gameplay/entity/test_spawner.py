"""
Comprehensive tests for the Entity Spawner System.

Note: spawner.py source file doesn't exist yet, so these tests are designed
based on the spawner configuration constants and expected spawner functionality
for a UE5-inspired game engine.

Tests for:
- Spawn point configuration
- Spawn rules (cooldown, limits, conditions)
- Wave spawning
- Pool-based spawning
- Spawn events and callbacks
- Spawn location selection
"""
from __future__ import annotations

import pytest
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from unittest.mock import Mock, MagicMock, patch, call
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto

from engine.gameplay.entity.actor import Actor, Transform, Pawn, Character
from engine.gameplay.entity.prefab import (
    PrefabRegistry,
    PrefabInstantiator,
    register_prefab,
    spawn_prefab,
)
from engine.gameplay.entity.constants import (
    DEFAULT_SPAWN_QUEUE_SIZE,
    DEFAULT_DESTROY_QUEUE_SIZE,
    DEFAULT_SPAWN_BATCH_SIZE,
    DEFAULT_DESTROY_BATCH_SIZE,
    DEFAULT_ENTITY_POOL_SIZE,
    DEFAULT_ENTITY_POOL_MAX_SIZE,
    DEFAULT_ENTITY_POOL_GROW_FACTOR,
    DEFERRED_SPAWN_PRIORITY,
    DEFERRED_DESTROY_PRIORITY,
)


# =============================================================================
# SPAWNER IMPLEMENTATION FOR TESTING
# =============================================================================
# Since spawner.py doesn't exist, we create a minimal implementation
# that the tests will verify against. In practice, this would be the
# actual spawner implementation.


class SpawnCondition:
    """Base class for spawn conditions."""

    def is_satisfied(self, spawner: "Spawner", context: Dict[str, Any]) -> bool:
        """Check if the condition is satisfied."""
        return True


class CooldownCondition(SpawnCondition):
    """Spawn condition based on cooldown time."""

    def __init__(self, cooldown_seconds: float):
        self.cooldown_seconds = cooldown_seconds
        # Initialize to negative infinity so cooldown is initially satisfied
        self.last_spawn_time = float('-inf')

    def is_satisfied(self, spawner: "Spawner", context: Dict[str, Any]) -> bool:
        current_time = context.get("current_time", time.time())
        return current_time - self.last_spawn_time >= self.cooldown_seconds

    def on_spawn(self, context: Dict[str, Any]) -> None:
        self.last_spawn_time = context.get("current_time", time.time())


class MaxActiveCondition(SpawnCondition):
    """Spawn condition based on maximum active entities."""

    def __init__(self, max_active: int):
        self.max_active = max_active

    def is_satisfied(self, spawner: "Spawner", context: Dict[str, Any]) -> bool:
        return len(spawner.active_entities) < self.max_active


class WaveCondition(SpawnCondition):
    """Spawn condition for wave-based spawning."""

    def __init__(self, entities_per_wave: int):
        self.entities_per_wave = entities_per_wave
        self.spawned_in_wave = 0

    def is_satisfied(self, spawner: "Spawner", context: Dict[str, Any]) -> bool:
        return self.spawned_in_wave < self.entities_per_wave

    def on_spawn(self, context: Dict[str, Any]) -> None:
        self.spawned_in_wave += 1

    def reset_wave(self) -> None:
        self.spawned_in_wave = 0


@dataclass
class SpawnPoint:
    """A spawn point configuration."""

    name: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    radius: float = 0.0
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)
    weight: float = 1.0
    max_spawns: int = -1  # -1 = unlimited
    spawn_count: int = 0


@dataclass
class SpawnRequest:
    """A request to spawn an entity."""

    prefab_name: str
    spawn_point: Optional[SpawnPoint] = None
    transform: Optional[Transform] = None
    overrides: Optional[Dict[str, Any]] = None
    callback: Optional[Callable[[Actor], None]] = None
    priority: int = DEFERRED_SPAWN_PRIORITY


class EntityPool:
    """Object pool for reusing entities."""

    def __init__(
        self,
        prefab_name: str,
        initial_size: int = DEFAULT_ENTITY_POOL_SIZE,
        max_size: int = DEFAULT_ENTITY_POOL_MAX_SIZE,
        grow_factor: float = DEFAULT_ENTITY_POOL_GROW_FACTOR,
    ):
        self.prefab_name = prefab_name
        self.initial_size = initial_size
        self.max_size = max_size
        self.grow_factor = grow_factor
        self._available: List[Actor] = []
        self._in_use: Set[int] = set()
        self._total_created = 0

    def acquire(self) -> Optional[Actor]:
        """Acquire an entity from the pool."""
        if self._available:
            entity = self._available.pop()
            self._in_use.add(entity.entity_id)
            return entity

        if self._total_created < self.max_size:
            entity = spawn_prefab(self.prefab_name)
            if entity:
                self._total_created += 1
                self._in_use.add(entity.entity_id)
            return entity

        return None

    def release(self, entity: Actor) -> bool:
        """Return an entity to the pool."""
        if entity.entity_id not in self._in_use:
            return False

        self._in_use.discard(entity.entity_id)
        if len(self._available) < self.max_size:
            # Reset entity state
            entity.position = (0.0, 0.0, 0.0)
            self._available.append(entity)
            return True
        return False

    @property
    def available_count(self) -> int:
        return len(self._available)

    @property
    def in_use_count(self) -> int:
        return len(self._in_use)

    @property
    def total_count(self) -> int:
        return self._total_created


class Spawner:
    """Entity spawner with spawn points and rules."""

    def __init__(self, name: str):
        self.name = name
        self._spawn_points: Dict[str, SpawnPoint] = {}
        self._conditions: List[SpawnCondition] = []
        self._active_entities: Dict[int, Actor] = {}
        self._pending_spawns: deque[SpawnRequest] = deque(maxlen=DEFAULT_SPAWN_QUEUE_SIZE)
        self._pools: Dict[str, EntityPool] = {}
        self._callbacks: Dict[str, List[Callable]] = {
            "on_spawn": [],
            "on_destroy": [],
            "on_wave_start": [],
            "on_wave_end": [],
        }
        self._enabled = True
        self._spawn_count = 0
        self._current_wave = 0

    # =========================================================================
    # SPAWN POINTS
    # =========================================================================

    def add_spawn_point(self, spawn_point: SpawnPoint) -> None:
        """Add a spawn point."""
        self._spawn_points[spawn_point.name] = spawn_point

    def remove_spawn_point(self, name: str) -> Optional[SpawnPoint]:
        """Remove a spawn point."""
        return self._spawn_points.pop(name, None)

    def get_spawn_point(self, name: str) -> Optional[SpawnPoint]:
        """Get a spawn point by name."""
        return self._spawn_points.get(name)

    def get_spawn_points(self, tags: Optional[Set[str]] = None) -> List[SpawnPoint]:
        """Get spawn points, optionally filtered by tags."""
        points = list(self._spawn_points.values())
        if tags:
            points = [p for p in points if p.tags & tags]
        return [p for p in points if p.enabled]

    # =========================================================================
    # CONDITIONS
    # =========================================================================

    def add_condition(self, condition: SpawnCondition) -> None:
        """Add a spawn condition."""
        self._conditions.append(condition)

    def remove_condition(self, condition: SpawnCondition) -> bool:
        """Remove a spawn condition."""
        try:
            self._conditions.remove(condition)
            return True
        except ValueError:
            return False

    def can_spawn(self, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if spawning is allowed."""
        if not self._enabled:
            return False
        ctx = context or {"current_time": time.time()}
        return all(c.is_satisfied(self, ctx) for c in self._conditions)

    # =========================================================================
    # SPAWNING
    # =========================================================================

    def spawn(
        self,
        prefab_name: str,
        spawn_point: Optional[str] = None,
        transform: Optional[Transform] = None,
        overrides: Optional[Dict[str, Any]] = None,
        immediate: bool = True,
        use_pool: bool = False,
    ) -> Optional[Actor]:
        """Spawn an entity."""
        context = {"current_time": time.time()}

        if not self.can_spawn(context):
            return None

        # Get spawn point
        sp = None
        if spawn_point:
            sp = self._spawn_points.get(spawn_point)
            if sp and (sp.max_spawns >= 0 and sp.spawn_count >= sp.max_spawns):
                return None

        # Determine spawn transform
        spawn_transform = transform
        if not spawn_transform and sp:
            spawn_transform = Transform(position=sp.position, rotation=sp.rotation)

        # Spawn from pool or create new
        entity = None
        if use_pool and prefab_name in self._pools:
            entity = self._pools[prefab_name].acquire()
            if entity and spawn_transform:
                entity.position = spawn_transform.position
                entity.rotation = spawn_transform.rotation
        else:
            if immediate:
                entity = spawn_prefab(
                    prefab_name,
                    position=spawn_transform.position if spawn_transform else None,
                    rotation=spawn_transform.rotation if spawn_transform else None,
                    overrides=overrides,
                    immediate=True,
                )
            else:
                request = SpawnRequest(
                    prefab_name=prefab_name,
                    spawn_point=sp,
                    transform=spawn_transform,
                    overrides=overrides,
                )
                self._pending_spawns.append(request)
                return None

        if entity:
            self._register_entity(entity, sp, context)

        return entity

    def _register_entity(
        self,
        entity: Actor,
        spawn_point: Optional[SpawnPoint],
        context: Dict[str, Any],
    ) -> None:
        """Register a spawned entity."""
        self._active_entities[entity.entity_id] = entity
        self._spawn_count += 1

        if spawn_point:
            spawn_point.spawn_count += 1

        # Notify conditions
        for condition in self._conditions:
            if hasattr(condition, "on_spawn"):
                condition.on_spawn(context)

        # Fire callbacks
        for callback in self._callbacks["on_spawn"]:
            try:
                callback(entity)
            except Exception:
                pass

    def despawn(self, entity: Actor, return_to_pool: bool = False) -> bool:
        """Despawn an entity."""
        if entity.entity_id not in self._active_entities:
            return False

        del self._active_entities[entity.entity_id]

        # Return to pool if applicable
        if return_to_pool:
            for pool in self._pools.values():
                if pool.release(entity):
                    break
        else:
            entity.destroy()

        # Fire callbacks
        for callback in self._callbacks["on_destroy"]:
            try:
                callback(entity)
            except Exception:
                pass

        return True

    # =========================================================================
    # DEFERRED SPAWNING
    # =========================================================================

    def process_pending(self) -> List[Actor]:
        """Process pending spawn requests."""
        spawned = []
        batch_size = min(DEFAULT_SPAWN_BATCH_SIZE, len(self._pending_spawns))

        for _ in range(batch_size):
            if not self._pending_spawns:
                break

            request = self._pending_spawns.popleft()
            context = {"current_time": time.time()}

            if not self.can_spawn(context):
                continue

            entity = spawn_prefab(
                request.prefab_name,
                position=request.transform.position if request.transform else None,
                rotation=request.transform.rotation if request.transform else None,
                overrides=request.overrides,
                immediate=True,
            )

            if entity:
                self._register_entity(entity, request.spawn_point, context)
                spawned.append(entity)

                if request.callback:
                    try:
                        request.callback(entity)
                    except Exception:
                        pass

        return spawned

    # =========================================================================
    # WAVE SPAWNING
    # =========================================================================

    def start_wave(self, wave_number: int = -1) -> None:
        """Start a spawn wave."""
        if wave_number >= 0:
            self._current_wave = wave_number
        else:
            self._current_wave += 1

        # Reset wave conditions
        for condition in self._conditions:
            if isinstance(condition, WaveCondition):
                condition.reset_wave()

        # Fire callbacks
        for callback in self._callbacks["on_wave_start"]:
            try:
                callback(self._current_wave)
            except Exception:
                pass

    def end_wave(self) -> None:
        """End the current spawn wave."""
        for callback in self._callbacks["on_wave_end"]:
            try:
                callback(self._current_wave)
            except Exception:
                pass

    # =========================================================================
    # POOLING
    # =========================================================================

    def create_pool(
        self,
        prefab_name: str,
        initial_size: int = DEFAULT_ENTITY_POOL_SIZE,
        max_size: int = DEFAULT_ENTITY_POOL_MAX_SIZE,
    ) -> EntityPool:
        """Create an entity pool for a prefab."""
        pool = EntityPool(
            prefab_name=prefab_name,
            initial_size=initial_size,
            max_size=max_size,
        )
        self._pools[prefab_name] = pool
        return pool

    def get_pool(self, prefab_name: str) -> Optional[EntityPool]:
        """Get a pool by prefab name."""
        return self._pools.get(prefab_name)

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_spawn(self, callback: Callable[[Actor], None]) -> None:
        """Register a spawn callback."""
        self._callbacks["on_spawn"].append(callback)

    def on_destroy(self, callback: Callable[[Actor], None]) -> None:
        """Register a destroy callback."""
        self._callbacks["on_destroy"].append(callback)

    def on_wave_start(self, callback: Callable[[int], None]) -> None:
        """Register a wave start callback."""
        self._callbacks["on_wave_start"].append(callback)

    def on_wave_end(self, callback: Callable[[int], None]) -> None:
        """Register a wave end callback."""
        self._callbacks["on_wave_end"].append(callback)

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def active_entities(self) -> Dict[int, Actor]:
        """Get all active spawned entities."""
        return self._active_entities

    @property
    def spawn_count(self) -> int:
        """Get total spawn count."""
        return self._spawn_count

    @property
    def current_wave(self) -> int:
        """Get current wave number."""
        return self._current_wave

    @property
    def enabled(self) -> bool:
        """Check if spawner is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the spawner."""
        self._enabled = value


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_state():
    """Reset state before each test."""
    PrefabRegistry.reset_instance()
    PrefabInstantiator.reset_instance()
    Actor.reset_entity_ids()
    yield
    PrefabRegistry.reset_instance()
    PrefabInstantiator.reset_instance()


@pytest.fixture
def spawner():
    """Create a basic spawner."""
    return Spawner("TestSpawner")


@pytest.fixture
def enemy_prefab():
    """Register an enemy prefab for testing."""
    return register_prefab("enemy", Character, tags={"enemy"})


@pytest.fixture
def spawn_point():
    """Create a basic spawn point."""
    return SpawnPoint(
        name="spawn_1",
        position=(10.0, 0.0, 10.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )


# =============================================================================
# SPAWN POINT TESTS
# =============================================================================


class TestSpawnPoint:
    """Tests for SpawnPoint configuration."""

    def test_spawn_point_creation(self):
        """Test creating a spawn point."""
        sp = SpawnPoint(name="test_point")
        assert sp.name == "test_point"
        assert sp.enabled is True

    def test_spawn_point_with_position(self):
        """Test spawn point with custom position."""
        sp = SpawnPoint(
            name="positioned",
            position=(100.0, 50.0, 100.0),
        )
        assert sp.position == (100.0, 50.0, 100.0)

    def test_spawn_point_with_rotation(self):
        """Test spawn point with custom rotation."""
        sp = SpawnPoint(
            name="rotated",
            rotation=(0.0, 0.707, 0.0, 0.707),
        )
        assert sp.rotation == (0.0, 0.707, 0.0, 0.707)

    def test_spawn_point_with_radius(self):
        """Test spawn point with spawn radius."""
        sp = SpawnPoint(name="area", radius=5.0)
        assert sp.radius == 5.0

    def test_spawn_point_with_tags(self):
        """Test spawn point with tags."""
        sp = SpawnPoint(name="tagged", tags={"outdoor", "safe"})
        assert "outdoor" in sp.tags
        assert "safe" in sp.tags

    def test_spawn_point_with_weight(self):
        """Test spawn point with selection weight."""
        sp = SpawnPoint(name="weighted", weight=2.5)
        assert sp.weight == 2.5

    def test_spawn_point_max_spawns(self):
        """Test spawn point with max spawns limit."""
        sp = SpawnPoint(name="limited", max_spawns=10)
        assert sp.max_spawns == 10

    def test_spawn_point_unlimited_spawns(self):
        """Test spawn point with unlimited spawns."""
        sp = SpawnPoint(name="unlimited", max_spawns=-1)
        assert sp.max_spawns == -1


class TestSpawnerSpawnPoints:
    """Tests for spawner spawn point management."""

    def test_add_spawn_point(self, spawner, spawn_point):
        """Test adding a spawn point to spawner."""
        spawner.add_spawn_point(spawn_point)
        assert spawner.get_spawn_point("spawn_1") is spawn_point

    def test_remove_spawn_point(self, spawner, spawn_point):
        """Test removing a spawn point."""
        spawner.add_spawn_point(spawn_point)
        removed = spawner.remove_spawn_point("spawn_1")
        assert removed is spawn_point
        assert spawner.get_spawn_point("spawn_1") is None

    def test_remove_nonexistent_spawn_point(self, spawner):
        """Test removing nonexistent spawn point."""
        result = spawner.remove_spawn_point("nonexistent")
        assert result is None

    def test_get_spawn_points_all(self, spawner):
        """Test getting all spawn points."""
        spawner.add_spawn_point(SpawnPoint(name="sp1"))
        spawner.add_spawn_point(SpawnPoint(name="sp2"))
        spawner.add_spawn_point(SpawnPoint(name="sp3"))
        points = spawner.get_spawn_points()
        assert len(points) == 3

    def test_get_spawn_points_by_tags(self, spawner):
        """Test getting spawn points filtered by tags."""
        spawner.add_spawn_point(SpawnPoint(name="outdoor1", tags={"outdoor"}))
        spawner.add_spawn_point(SpawnPoint(name="outdoor2", tags={"outdoor", "safe"}))
        spawner.add_spawn_point(SpawnPoint(name="indoor", tags={"indoor"}))

        outdoor_points = spawner.get_spawn_points(tags={"outdoor"})
        assert len(outdoor_points) == 2

    def test_get_spawn_points_excludes_disabled(self, spawner):
        """Test disabled spawn points are excluded."""
        spawner.add_spawn_point(SpawnPoint(name="enabled"))
        spawner.add_spawn_point(SpawnPoint(name="disabled", enabled=False))
        points = spawner.get_spawn_points()
        assert len(points) == 1
        assert points[0].name == "enabled"


# =============================================================================
# SPAWN CONDITION TESTS
# =============================================================================


class TestCooldownCondition:
    """Tests for CooldownCondition."""

    def test_cooldown_initially_satisfied(self, spawner):
        """Test cooldown is satisfied initially."""
        condition = CooldownCondition(cooldown_seconds=1.0)
        context = {"current_time": 0.0}
        assert condition.is_satisfied(spawner, context) is True

    def test_cooldown_not_satisfied_during_cooldown(self, spawner):
        """Test cooldown is not satisfied during cooldown period."""
        condition = CooldownCondition(cooldown_seconds=1.0)
        condition.last_spawn_time = 0.0
        context = {"current_time": 0.5}
        assert condition.is_satisfied(spawner, context) is False

    def test_cooldown_satisfied_after_period(self, spawner):
        """Test cooldown is satisfied after period."""
        condition = CooldownCondition(cooldown_seconds=1.0)
        condition.last_spawn_time = 0.0
        context = {"current_time": 1.5}
        assert condition.is_satisfied(spawner, context) is True

    def test_cooldown_on_spawn_updates_time(self, spawner):
        """Test on_spawn updates last spawn time."""
        condition = CooldownCondition(cooldown_seconds=1.0)
        context = {"current_time": 5.0}
        condition.on_spawn(context)
        assert condition.last_spawn_time == 5.0


class TestMaxActiveCondition:
    """Tests for MaxActiveCondition."""

    def test_max_active_satisfied_when_under_limit(self, spawner):
        """Test condition satisfied when under limit."""
        condition = MaxActiveCondition(max_active=10)
        assert condition.is_satisfied(spawner, {}) is True

    def test_max_active_not_satisfied_at_limit(self, spawner, enemy_prefab):
        """Test condition not satisfied at limit."""
        condition = MaxActiveCondition(max_active=2)
        spawner.add_condition(condition)

        # Spawn up to limit
        spawner.spawn("enemy")
        spawner.spawn("enemy")

        # Now at limit
        assert condition.is_satisfied(spawner, {}) is False

    def test_max_active_satisfied_after_despawn(self, spawner, enemy_prefab):
        """Test condition satisfied after despawning."""
        condition = MaxActiveCondition(max_active=2)
        spawner.add_condition(condition)

        entity1 = spawner.spawn("enemy")
        spawner.spawn("enemy")

        spawner.despawn(entity1)
        assert condition.is_satisfied(spawner, {}) is True


class TestWaveCondition:
    """Tests for WaveCondition."""

    def test_wave_initially_satisfied(self, spawner):
        """Test wave condition initially satisfied."""
        condition = WaveCondition(entities_per_wave=5)
        assert condition.is_satisfied(spawner, {}) is True

    def test_wave_not_satisfied_at_limit(self, spawner):
        """Test wave condition not satisfied at limit."""
        condition = WaveCondition(entities_per_wave=2)
        condition.on_spawn({})
        condition.on_spawn({})
        assert condition.is_satisfied(spawner, {}) is False

    def test_wave_reset(self, spawner):
        """Test resetting wave condition."""
        condition = WaveCondition(entities_per_wave=2)
        condition.on_spawn({})
        condition.on_spawn({})
        condition.reset_wave()
        assert condition.is_satisfied(spawner, {}) is True


class TestSpawnerConditions:
    """Tests for spawner condition management."""

    def test_add_condition(self, spawner):
        """Test adding a condition."""
        condition = CooldownCondition(cooldown_seconds=1.0)
        spawner.add_condition(condition)
        assert condition in spawner._conditions

    def test_remove_condition(self, spawner):
        """Test removing a condition."""
        condition = CooldownCondition(cooldown_seconds=1.0)
        spawner.add_condition(condition)
        result = spawner.remove_condition(condition)
        assert result is True
        assert condition not in spawner._conditions

    def test_remove_nonexistent_condition(self, spawner):
        """Test removing nonexistent condition."""
        condition = CooldownCondition(cooldown_seconds=1.0)
        result = spawner.remove_condition(condition)
        assert result is False

    def test_can_spawn_all_satisfied(self, spawner):
        """Test can_spawn when all conditions satisfied."""
        spawner.add_condition(CooldownCondition(cooldown_seconds=0.0))
        spawner.add_condition(MaxActiveCondition(max_active=100))
        assert spawner.can_spawn() is True

    def test_can_spawn_one_not_satisfied(self, spawner):
        """Test can_spawn when one condition not satisfied."""
        satisfied = CooldownCondition(cooldown_seconds=0.0)
        not_satisfied = MaxActiveCondition(max_active=0)
        spawner.add_condition(satisfied)
        spawner.add_condition(not_satisfied)
        assert spawner.can_spawn() is False

    def test_can_spawn_disabled(self, spawner):
        """Test can_spawn when spawner disabled."""
        spawner.enabled = False
        assert spawner.can_spawn() is False


# =============================================================================
# BASIC SPAWNING TESTS
# =============================================================================


class TestBasicSpawning:
    """Tests for basic spawning functionality."""

    def test_spawn_basic(self, spawner, enemy_prefab):
        """Test basic entity spawning."""
        entity = spawner.spawn("enemy")
        assert entity is not None
        assert isinstance(entity, Character)

    def test_spawn_at_spawn_point(self, spawner, enemy_prefab, spawn_point):
        """Test spawning at a spawn point."""
        spawner.add_spawn_point(spawn_point)
        entity = spawner.spawn("enemy", spawn_point="spawn_1")
        assert entity is not None
        assert entity.position == spawn_point.position

    def test_spawn_with_transform(self, spawner, enemy_prefab):
        """Test spawning with custom transform."""
        transform = Transform(position=(50.0, 0.0, 50.0))
        entity = spawner.spawn("enemy", transform=transform)
        assert entity.position == (50.0, 0.0, 50.0)

    def test_spawn_with_overrides(self, spawner, enemy_prefab):
        """Test spawning with property overrides."""
        entity = spawner.spawn("enemy", overrides={"max_walk_speed": 20.0})
        assert entity.max_walk_speed == 20.0

    def test_spawn_increments_count(self, spawner, enemy_prefab):
        """Test spawn count is incremented."""
        assert spawner.spawn_count == 0
        spawner.spawn("enemy")
        assert spawner.spawn_count == 1
        spawner.spawn("enemy")
        assert spawner.spawn_count == 2

    def test_spawn_registers_entity(self, spawner, enemy_prefab):
        """Test spawned entity is registered."""
        entity = spawner.spawn("enemy")
        assert entity.entity_id in spawner.active_entities

    def test_spawn_blocked_by_conditions(self, spawner, enemy_prefab):
        """Test spawn blocked when conditions not met."""
        spawner.add_condition(MaxActiveCondition(max_active=0))
        entity = spawner.spawn("enemy")
        assert entity is None

    def test_spawn_point_max_spawns(self, spawner, enemy_prefab):
        """Test spawn point respects max spawns."""
        sp = SpawnPoint(name="limited", max_spawns=2)
        spawner.add_spawn_point(sp)

        spawner.spawn("enemy", spawn_point="limited")
        spawner.spawn("enemy", spawn_point="limited")
        entity = spawner.spawn("enemy", spawn_point="limited")

        assert entity is None


class TestDespawning:
    """Tests for entity despawning."""

    def test_despawn_entity(self, spawner, enemy_prefab):
        """Test despawning an entity."""
        entity = spawner.spawn("enemy")
        result = spawner.despawn(entity)
        assert result is True
        assert entity.entity_id not in spawner.active_entities

    def test_despawn_nonexistent(self, spawner, enemy_prefab):
        """Test despawning non-tracked entity."""
        entity = spawn_prefab("enemy")  # Spawn outside spawner
        result = spawner.despawn(entity)
        assert result is False

    def test_despawn_return_to_pool(self, spawner, enemy_prefab):
        """Test despawning with return to pool."""
        pool = spawner.create_pool("enemy", initial_size=5, max_size=10)
        entity = spawner.spawn("enemy", use_pool=True)
        initial_available = pool.available_count

        spawner.despawn(entity, return_to_pool=True)
        assert pool.available_count > initial_available


# =============================================================================
# DEFERRED SPAWNING TESTS
# =============================================================================


class TestDeferredSpawning:
    """Tests for deferred spawning."""

    def test_spawn_deferred(self, spawner, enemy_prefab):
        """Test deferred spawn returns None."""
        result = spawner.spawn("enemy", immediate=False)
        assert result is None

    def test_spawn_deferred_queued(self, spawner, enemy_prefab):
        """Test deferred spawn is queued."""
        spawner.spawn("enemy", immediate=False)
        spawner.spawn("enemy", immediate=False)
        assert len(spawner._pending_spawns) == 2

    def test_process_pending(self, spawner, enemy_prefab):
        """Test processing pending spawns."""
        spawner.spawn("enemy", immediate=False)
        spawner.spawn("enemy", immediate=False)
        spawner.spawn("enemy", immediate=False)

        created = spawner.process_pending()
        assert len(created) == 3

    def test_process_pending_batch_limit(self, spawner, enemy_prefab):
        """Test pending spawns are processed in batches."""
        for _ in range(DEFAULT_SPAWN_BATCH_SIZE + 10):
            spawner.spawn("enemy", immediate=False)

        created1 = spawner.process_pending()
        assert len(created1) == DEFAULT_SPAWN_BATCH_SIZE

        created2 = spawner.process_pending()
        assert len(created2) == 10

    def test_process_pending_respects_conditions(self, spawner, enemy_prefab):
        """Test pending spawns respect conditions."""
        spawner.add_condition(MaxActiveCondition(max_active=2))

        for _ in range(5):
            spawner.spawn("enemy", immediate=False)

        created = spawner.process_pending()
        assert len(created) == 2


# =============================================================================
# WAVE SPAWNING TESTS
# =============================================================================


class TestWaveSpawning:
    """Tests for wave-based spawning."""

    def test_start_wave(self, spawner):
        """Test starting a wave."""
        spawner.start_wave(1)
        assert spawner.current_wave == 1

    def test_start_wave_auto_increment(self, spawner):
        """Test wave auto-increments."""
        spawner.start_wave()
        assert spawner.current_wave == 1
        spawner.start_wave()
        assert spawner.current_wave == 2

    def test_wave_resets_conditions(self, spawner, enemy_prefab):
        """Test wave start resets wave conditions."""
        condition = WaveCondition(entities_per_wave=2)
        spawner.add_condition(condition)

        spawner.spawn("enemy")
        spawner.spawn("enemy")
        assert not condition.is_satisfied(spawner, {})

        spawner.start_wave()
        assert condition.is_satisfied(spawner, {})

    def test_wave_callback_on_start(self, spawner):
        """Test wave start callback is fired."""
        callback = Mock()
        spawner.on_wave_start(callback)
        spawner.start_wave(5)
        callback.assert_called_once_with(5)

    def test_wave_callback_on_end(self, spawner):
        """Test wave end callback is fired."""
        callback = Mock()
        spawner.on_wave_end(callback)
        spawner.start_wave(3)
        spawner.end_wave()
        callback.assert_called_once_with(3)

    def test_wave_spawning_limit(self, spawner, enemy_prefab):
        """Test wave spawning respects per-wave limit."""
        condition = WaveCondition(entities_per_wave=3)
        spawner.add_condition(condition)

        spawner.start_wave()
        spawned = []
        for _ in range(5):
            entity = spawner.spawn("enemy")
            if entity:
                spawned.append(entity)

        assert len(spawned) == 3


# =============================================================================
# POOL-BASED SPAWNING TESTS
# =============================================================================


class TestEntityPool:
    """Tests for EntityPool."""

    def test_pool_creation(self, enemy_prefab):
        """Test creating an entity pool."""
        pool = EntityPool(
            prefab_name="enemy",
            initial_size=10,
            max_size=100,
        )
        assert pool.prefab_name == "enemy"
        assert pool.initial_size == 10
        assert pool.max_size == 100

    def test_pool_acquire(self, enemy_prefab):
        """Test acquiring from pool."""
        pool = EntityPool(prefab_name="enemy")
        entity = pool.acquire()
        assert entity is not None
        assert pool.in_use_count == 1

    def test_pool_release(self, enemy_prefab):
        """Test releasing to pool."""
        pool = EntityPool(prefab_name="enemy")
        entity = pool.acquire()
        result = pool.release(entity)
        assert result is True
        assert pool.available_count == 1

    def test_pool_reuse(self, enemy_prefab):
        """Test pool reuses entities."""
        pool = EntityPool(prefab_name="enemy")
        entity1 = pool.acquire()
        pool.release(entity1)
        entity2 = pool.acquire()
        assert entity1 is entity2

    def test_pool_max_size(self, enemy_prefab):
        """Test pool respects max size."""
        pool = EntityPool(prefab_name="enemy", max_size=2)
        pool.acquire()
        pool.acquire()
        entity = pool.acquire()
        assert entity is None

    def test_pool_release_nonacquired(self, enemy_prefab):
        """Test releasing non-acquired entity fails."""
        pool = EntityPool(prefab_name="enemy")
        other_entity = spawn_prefab("enemy")
        result = pool.release(other_entity)
        assert result is False

    def test_pool_counts(self, enemy_prefab):
        """Test pool count properties."""
        pool = EntityPool(prefab_name="enemy")
        pool.acquire()
        pool.acquire()
        e3 = pool.acquire()
        pool.release(e3)

        assert pool.in_use_count == 2
        assert pool.available_count == 1
        assert pool.total_count == 3


class TestSpawnerPooling:
    """Tests for spawner pooling integration."""

    def test_create_pool(self, spawner, enemy_prefab):
        """Test creating a pool through spawner."""
        pool = spawner.create_pool("enemy", initial_size=5, max_size=20)
        assert pool is not None
        assert spawner.get_pool("enemy") is pool

    def test_spawn_from_pool(self, spawner, enemy_prefab):
        """Test spawning from pool."""
        spawner.create_pool("enemy")
        entity = spawner.spawn("enemy", use_pool=True)
        assert entity is not None

    def test_spawn_from_pool_reuses(self, spawner, enemy_prefab):
        """Test pool reuses despawned entities."""
        pool = spawner.create_pool("enemy")
        entity1 = spawner.spawn("enemy", use_pool=True)
        spawner.despawn(entity1, return_to_pool=True)
        entity2 = spawner.spawn("enemy", use_pool=True)
        assert entity1 is entity2


# =============================================================================
# CALLBACK TESTS
# =============================================================================


class TestSpawnerCallbacks:
    """Tests for spawner callbacks."""

    def test_on_spawn_callback(self, spawner, enemy_prefab):
        """Test on_spawn callback is fired."""
        callback = Mock()
        spawner.on_spawn(callback)
        entity = spawner.spawn("enemy")
        callback.assert_called_once_with(entity)

    def test_on_destroy_callback(self, spawner, enemy_prefab):
        """Test on_destroy callback is fired."""
        callback = Mock()
        spawner.on_destroy(callback)
        entity = spawner.spawn("enemy")
        spawner.despawn(entity)
        callback.assert_called_once_with(entity)

    def test_multiple_callbacks(self, spawner, enemy_prefab):
        """Test multiple callbacks are all fired."""
        callback1 = Mock()
        callback2 = Mock()
        spawner.on_spawn(callback1)
        spawner.on_spawn(callback2)

        spawner.spawn("enemy")

        callback1.assert_called_once()
        callback2.assert_called_once()

    def test_callback_error_doesnt_break_spawn(self, spawner, enemy_prefab):
        """Test callback error doesn't prevent spawn."""
        bad_callback = Mock(side_effect=RuntimeError("Callback error"))
        good_callback = Mock()
        spawner.on_spawn(bad_callback)
        spawner.on_spawn(good_callback)

        entity = spawner.spawn("enemy")

        assert entity is not None
        good_callback.assert_called_once()


# =============================================================================
# SPAWN LOCATION SELECTION TESTS
# =============================================================================


class TestSpawnLocationSelection:
    """Tests for spawn location selection."""

    def test_spawn_at_specific_point(self, spawner, enemy_prefab):
        """Test spawning at a specific spawn point."""
        sp1 = SpawnPoint(name="point1", position=(10.0, 0.0, 10.0))
        sp2 = SpawnPoint(name="point2", position=(20.0, 0.0, 20.0))
        spawner.add_spawn_point(sp1)
        spawner.add_spawn_point(sp2)

        entity = spawner.spawn("enemy", spawn_point="point2")
        assert entity.position == (20.0, 0.0, 20.0)

    def test_spawn_with_transform_override(self, spawner, enemy_prefab, spawn_point):
        """Test transform overrides spawn point."""
        spawner.add_spawn_point(spawn_point)
        transform = Transform(position=(100.0, 0.0, 100.0))

        entity = spawner.spawn("enemy", transform=transform)
        assert entity.position == (100.0, 0.0, 100.0)

    def test_get_spawn_points_weighted(self, spawner):
        """Test spawn points have weights for selection."""
        sp1 = SpawnPoint(name="low", weight=1.0)
        sp2 = SpawnPoint(name="high", weight=10.0)
        spawner.add_spawn_point(sp1)
        spawner.add_spawn_point(sp2)

        points = spawner.get_spawn_points()
        weights = {p.name: p.weight for p in points}
        assert weights["high"] > weights["low"]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestSpawnerIntegration:
    """Integration tests for spawner system."""

    def test_wave_based_enemy_spawning(self, spawner, enemy_prefab):
        """Test complete wave-based spawning scenario."""
        # Configure spawner
        spawner.add_condition(WaveCondition(entities_per_wave=5))
        spawner.add_spawn_point(SpawnPoint(name="spawn_a", position=(10.0, 0.0, 10.0)))
        spawner.add_spawn_point(SpawnPoint(name="spawn_b", position=(20.0, 0.0, 20.0)))

        # Wave 1
        spawner.start_wave(1)
        wave1_spawns = []
        for _ in range(10):
            entity = spawner.spawn("enemy")
            if entity:
                wave1_spawns.append(entity)
        assert len(wave1_spawns) == 5

        # Wave 2
        spawner.start_wave(2)
        wave2_spawns = []
        for _ in range(10):
            entity = spawner.spawn("enemy")
            if entity:
                wave2_spawns.append(entity)
        assert len(wave2_spawns) == 5

    def test_pool_based_spawning_performance(self, spawner, enemy_prefab):
        """Test pool improves spawn/despawn cycle."""
        pool = spawner.create_pool("enemy", max_size=50)

        # Spawn and despawn cycle
        for _ in range(3):
            entities = []
            for _ in range(10):
                e = spawner.spawn("enemy", use_pool=True)
                if e:
                    entities.append(e)

            for e in entities:
                spawner.despawn(e, return_to_pool=True)

        # Pool should be reusing entities
        assert pool.available_count > 0
        assert pool.total_count <= 10  # Shouldn't keep creating new ones

    def test_mixed_immediate_and_deferred(self, spawner, enemy_prefab):
        """Test mixing immediate and deferred spawns."""
        immediate = spawner.spawn("enemy", immediate=True)
        spawner.spawn("enemy", immediate=False)
        spawner.spawn("enemy", immediate=False)

        assert immediate is not None
        assert len(spawner.active_entities) == 1

        deferred = spawner.process_pending()
        assert len(deferred) == 2
        assert len(spawner.active_entities) == 3


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestSpawnerEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_spawner_disabled(self, spawner, enemy_prefab):
        """Test spawning when spawner is disabled."""
        spawner.enabled = False
        entity = spawner.spawn("enemy")
        assert entity is None

    def test_spawn_nonexistent_prefab(self, spawner):
        """Test spawning nonexistent prefab."""
        entity = spawner.spawn("nonexistent")
        assert entity is None

    def test_spawn_nonexistent_spawn_point(self, spawner, enemy_prefab):
        """Test spawning at nonexistent spawn point."""
        entity = spawner.spawn("enemy", spawn_point="nonexistent")
        assert entity is not None  # Should still spawn, just without position

    def test_thread_safe_spawning(self, spawner, enemy_prefab):
        """Test thread-safe spawning."""
        entities = []
        lock = threading.Lock()

        def spawn_entity():
            entity = spawner.spawn("enemy")
            if entity:
                with lock:
                    entities.append(entity)

        threads = [threading.Thread(target=spawn_entity) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All spawns should succeed
        assert len(entities) == 20
        # All should have unique IDs
        ids = [e.entity_id for e in entities]
        assert len(ids) == len(set(ids))
