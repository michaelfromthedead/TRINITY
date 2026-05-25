"""
FlowForge E2E Test Fixture - Trinity Decorated Classes

This file contains sample Python code with Trinity decorators for testing
the FlowForge desktop application's ability to:
- Parse Python files with Trinity decorators
- Create visual nodes from decorated classes
- Handle different Trinity types (component, system, resource, event)

The classes here represent a simple game entity system with:
- Position and Velocity components
- Health and Damage resources
- A Movement system
- Collision and Damage events
"""

from dataclasses import dataclass
from typing import Optional, List, Callable
from enum import Enum


# =============================================================================
# Mock Trinity Decorators (for testing without actual Trinity runtime)
# =============================================================================

def component(cls=None, *, name: str = None):
    """Mark a class as a Trinity Component."""
    def decorator(cls):
        cls.__trinity_type__ = 'component'
        cls.__trinity_name__ = name or cls.__name__
        return cls
    return decorator(cls) if cls else decorator


def system(cls=None, *, name: str = None, priority: int = 0):
    """Mark a class as a Trinity System."""
    def decorator(cls):
        cls.__trinity_type__ = 'system'
        cls.__trinity_name__ = name or cls.__name__
        cls.__trinity_priority__ = priority
        return cls
    return decorator(cls) if cls else decorator


def resource(cls=None, *, name: str = None, singleton: bool = True):
    """Mark a class as a Trinity Resource."""
    def decorator(cls):
        cls.__trinity_type__ = 'resource'
        cls.__trinity_name__ = name or cls.__name__
        cls.__trinity_singleton__ = singleton
        return cls
    return decorator(cls) if cls else decorator


def event(cls=None, *, name: str = None):
    """Mark a class as a Trinity Event."""
    def decorator(cls):
        cls.__trinity_type__ = 'event'
        cls.__trinity_name__ = name or cls.__name__
        return cls
    return decorator(cls) if cls else decorator


# =============================================================================
# Components - Data attached to entities
# =============================================================================

@component
@dataclass
class Position:
    """2D position component for entities."""
    x: float = 0.0
    y: float = 0.0


@component
@dataclass
class Velocity:
    """Velocity component for moving entities."""
    dx: float = 0.0
    dy: float = 0.0


@component
@dataclass
class Sprite:
    """Visual representation component."""
    texture_id: str = "default"
    width: int = 32
    height: int = 32
    visible: bool = True
    layer: int = 0


@component
@dataclass
class Collider:
    """Collision detection component."""
    width: float = 1.0
    height: float = 1.0
    is_trigger: bool = False
    collision_mask: int = 0xFFFF


@component(name="PlayerTag")
class Player:
    """Tag component to mark player entities."""
    player_id: int = 0
    name: str = "Player"


@component
@dataclass
class Health:
    """Health component for damageable entities."""
    current: float = 100.0
    maximum: float = 100.0
    regeneration_rate: float = 0.0
    invulnerable: bool = False

    def take_damage(self, amount: float) -> float:
        """Apply damage and return actual damage taken."""
        if self.invulnerable:
            return 0.0
        actual_damage = min(amount, self.current)
        self.current -= actual_damage
        return actual_damage

    def heal(self, amount: float) -> float:
        """Heal and return actual healing done."""
        missing_health = self.maximum - self.current
        actual_heal = min(amount, missing_health)
        self.current += actual_heal
        return actual_heal

    @property
    def is_alive(self) -> bool:
        return self.current > 0


# =============================================================================
# Resources - Global game state
# =============================================================================

@resource
@dataclass
class GameTime:
    """Global time resource."""
    delta_time: float = 0.0
    total_time: float = 0.0
    time_scale: float = 1.0
    frame_count: int = 0
    is_paused: bool = False


@resource
@dataclass
class GameConfig:
    """Global game configuration resource."""
    screen_width: int = 1920
    screen_height: int = 1080
    gravity: float = 9.81
    debug_mode: bool = False
    max_entities: int = 10000


@resource(singleton=True)
@dataclass
class InputState:
    """Current input state resource."""
    mouse_x: int = 0
    mouse_y: int = 0
    mouse_buttons: int = 0
    keys_pressed: List[str] = None

    def __post_init__(self):
        if self.keys_pressed is None:
            self.keys_pressed = []

    def is_key_pressed(self, key: str) -> bool:
        return key in self.keys_pressed

    def is_mouse_button_pressed(self, button: int) -> bool:
        return bool(self.mouse_buttons & (1 << button))


# =============================================================================
# Events - Signals between systems
# =============================================================================

@event
@dataclass
class CollisionEvent:
    """Event fired when two entities collide."""
    entity_a: int
    entity_b: int
    contact_point_x: float
    contact_point_y: float
    normal_x: float = 0.0
    normal_y: float = 1.0


@event
@dataclass
class DamageEvent:
    """Event fired when an entity takes damage."""
    target_entity: int
    source_entity: Optional[int]
    damage_amount: float
    damage_type: str = "physical"
    is_critical: bool = False


@event(name="EntitySpawned")
@dataclass
class SpawnEvent:
    """Event fired when a new entity is created."""
    entity_id: int
    entity_type: str
    spawn_x: float
    spawn_y: float


@event
@dataclass
class DeathEvent:
    """Event fired when an entity dies."""
    entity_id: int
    killer_entity: Optional[int] = None
    death_cause: str = "unknown"


# =============================================================================
# Systems - Game logic processors
# =============================================================================

@system(priority=100)
class MovementSystem:
    """
    System that updates entity positions based on velocity.
    Runs at high priority to ensure position updates happen first.
    """

    def update(self, entities: List, time: GameTime) -> None:
        """Update positions for all entities with Position and Velocity."""
        if time.is_paused:
            return

        dt = time.delta_time * time.time_scale

        for entity in entities:
            if hasattr(entity, 'position') and hasattr(entity, 'velocity'):
                entity.position.x += entity.velocity.dx * dt
                entity.position.y += entity.velocity.dy * dt


@system(priority=50)
class PhysicsSystem:
    """
    System that applies physics simulation.
    Handles gravity, friction, and basic physics.
    """

    def __init__(self):
        self.gravity_enabled = True

    def update(self, entities: List, time: GameTime, config: GameConfig) -> None:
        """Apply physics to all entities with Velocity."""
        if time.is_paused:
            return

        dt = time.delta_time * time.time_scale

        for entity in entities:
            if hasattr(entity, 'velocity'):
                # Apply gravity
                if self.gravity_enabled:
                    entity.velocity.dy += config.gravity * dt


@system(priority=25)
class CollisionSystem:
    """
    System that detects and resolves collisions between entities.
    Emits CollisionEvent for each detected collision.
    """

    def __init__(self):
        self.collision_callbacks: List[Callable] = []

    def update(self, entities: List) -> List[CollisionEvent]:
        """Check for collisions and emit events."""
        events = []

        for i, entity_a in enumerate(entities):
            if not hasattr(entity_a, 'collider') or not hasattr(entity_a, 'position'):
                continue

            for entity_b in entities[i + 1:]:
                if not hasattr(entity_b, 'collider') or not hasattr(entity_b, 'position'):
                    continue

                if self._check_aabb_collision(entity_a, entity_b):
                    event = CollisionEvent(
                        entity_a=id(entity_a),
                        entity_b=id(entity_b),
                        contact_point_x=(entity_a.position.x + entity_b.position.x) / 2,
                        contact_point_y=(entity_a.position.y + entity_b.position.y) / 2,
                    )
                    events.append(event)

        return events

    def _check_aabb_collision(self, a, b) -> bool:
        """Check axis-aligned bounding box collision."""
        return (
            a.position.x < b.position.x + b.collider.width and
            a.position.x + a.collider.width > b.position.x and
            a.position.y < b.position.y + b.collider.height and
            a.position.y + a.collider.height > b.position.y
        )


@system(priority=10, name="HealthManager")
class HealthSystem:
    """
    System that manages entity health and processes damage.
    Handles regeneration and death detection.
    """

    def update(self, entities: List, time: GameTime) -> List[DeathEvent]:
        """Update health regeneration and check for deaths."""
        death_events = []

        if time.is_paused:
            return death_events

        dt = time.delta_time * time.time_scale

        for entity in entities:
            if not hasattr(entity, 'health'):
                continue

            health = entity.health

            # Apply regeneration
            if health.regeneration_rate > 0 and health.is_alive:
                health.heal(health.regeneration_rate * dt)

            # Check for death
            if not health.is_alive:
                death_events.append(DeathEvent(
                    entity_id=id(entity),
                    death_cause="health_depleted"
                ))

        return death_events

    def process_damage_event(self, event: DamageEvent, entities: dict) -> None:
        """Process a damage event and apply damage to target."""
        target = entities.get(event.target_entity)
        if target and hasattr(target, 'health'):
            damage = event.damage_amount
            if event.is_critical:
                damage *= 2.0
            target.health.take_damage(damage)


@system(priority=5)
class RenderSystem:
    """
    System that handles rendering entities to the screen.
    Runs at low priority to render after all updates.
    """

    def __init__(self):
        self.render_queue: List = []

    def update(self, entities: List) -> None:
        """Collect visible entities for rendering."""
        self.render_queue.clear()

        for entity in entities:
            if hasattr(entity, 'sprite') and hasattr(entity, 'position'):
                if entity.sprite.visible:
                    self.render_queue.append((
                        entity.sprite.layer,
                        entity.position,
                        entity.sprite
                    ))

        # Sort by layer
        self.render_queue.sort(key=lambda x: x[0])

    def render(self, screen) -> None:
        """Render all queued sprites to the screen."""
        for layer, position, sprite in self.render_queue:
            # Rendering logic would go here
            pass


# =============================================================================
# Additional utility types for comprehensive testing
# =============================================================================

class Direction(Enum):
    """Direction enumeration for movement."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@component
@dataclass
class AIController:
    """AI behavior controller component."""
    behavior_tree_id: str = "default_ai"
    target_entity: Optional[int] = None
    aggression_level: float = 0.5
    detection_range: float = 100.0
    attack_cooldown: float = 1.0
    current_state: str = "idle"


@resource
@dataclass
class AudioState:
    """Global audio state resource."""
    master_volume: float = 1.0
    music_volume: float = 0.7
    sfx_volume: float = 1.0
    is_muted: bool = False


@event
@dataclass
class AudioEvent:
    """Event to trigger audio playback."""
    sound_id: str
    volume: float = 1.0
    pitch: float = 1.0
    loop: bool = False
    position_x: Optional[float] = None
    position_y: Optional[float] = None


# =============================================================================
# Entry point for direct execution (testing)
# =============================================================================

if __name__ == "__main__":
    # Print all Trinity-decorated classes for verification
    import inspect

    print("Trinity Decorated Classes Found:")
    print("=" * 50)

    for name, obj in list(globals().items()):
        if inspect.isclass(obj) and hasattr(obj, '__trinity_type__'):
            trinity_type = obj.__trinity_type__
            trinity_name = getattr(obj, '__trinity_name__', name)
            print(f"  [{trinity_type:10}] {trinity_name}")

            # List fields for dataclasses
            if hasattr(obj, '__dataclass_fields__'):
                for field_name, field_info in obj.__dataclass_fields__.items():
                    field_type = field_info.type.__name__ if hasattr(field_info.type, '__name__') else str(field_info.type)
                    print(f"              - {field_name}: {field_type}")

    print("=" * 50)
    print(f"Total: {sum(1 for obj in globals().values() if inspect.isclass(obj) and hasattr(obj, '__trinity_type__'))}")
