"""
Core Particle System Implementation.

Provides particle emitter lifecycle management, particle pooling, and budget control.
Supports CPU, GPU, and Auto simulation modes based on @particle_emitter decorator config.

Architecture:
    ParticleEmitter - Manages spawn, update, render, death lifecycle
    ParticlePool - Ring buffer of particles with max alive limit and recycling
    ParticleBudget - Category-based budgeting for particle counts/memory
    EmitterConfig - Configuration from @particle_emitter decorator
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterator,
    Optional,
    Protocol,
    TypeVar,
)

if TYPE_CHECKING:
    from engine.rendering.particles.particle_modules import ParticleModule


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class SimulationMode(Enum):
    """Particle simulation execution mode."""

    CPU = auto()  # CPU-side simulation (full control, slower for large counts)
    GPU = auto()  # GPU compute shader simulation (fast, limited flexibility)
    AUTO = auto()  # Automatically choose based on particle count and complexity


class EmitterState(Enum):
    """Lifecycle state of a particle emitter."""

    INACTIVE = auto()  # Not active, not emitting
    WARMING_UP = auto()  # Pre-simulating to fill the system
    ACTIVE = auto()  # Actively emitting and simulating
    STOPPING = auto()  # No longer emitting, waiting for particles to die
    STOPPED = auto()  # Fully stopped, can be recycled


class ParticleState(Enum):
    """State of an individual particle."""

    DEAD = 0  # Available for reuse
    ALIVE = 1  # Currently active
    DYING = 2  # In death transition


# Import centralized constants
from engine.rendering.particles.constants import (
    PARTICLE_CONSTANTS,
    DEFAULT_GPU_THRESHOLD,
    DEFAULT_MAX_PARTICLES,
    DEFAULT_WARMUP_TIME,
    DEFAULT_LIFETIME,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class Vec3:
    """Simple 3D vector for particle positions/velocities."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self.__mul__(scalar)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Vec3":
        length = self.length()
        if length < 1e-8:
            return Vec3(0, 0, 0)
        return Vec3(self.x / length, self.y / length, self.z / length)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )


@dataclass
class Vec4:
    """4D vector for colors with alpha."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    @classmethod
    def from_rgb(cls, r: float, g: float, b: float, a: float = 1.0) -> "Vec4":
        return cls(r, g, b, a)

    def lerp(self, other: "Vec4", t: float) -> "Vec4":
        return Vec4(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
            self.w + (other.w - self.w) * t,
        )


@dataclass
class Particle:
    """Individual particle data structure."""

    # Core state
    state: ParticleState = ParticleState.DEAD
    age: float = 0.0
    lifetime: float = 1.0

    # Transform
    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    acceleration: Vec3 = field(default_factory=Vec3)

    # Visual properties
    color: Vec4 = field(default_factory=lambda: Vec4(1, 1, 1, 1))
    size: float = 1.0
    rotation: float = 0.0
    angular_velocity: float = 0.0

    # Custom data slots for modules
    custom_data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_alive(self) -> bool:
        return self.state != ParticleState.DEAD

    @property
    def normalized_age(self) -> float:
        """Age as fraction of lifetime (0-1)."""
        if self.lifetime <= 0:
            return 1.0
        return min(1.0, self.age / self.lifetime)

    def reset(self) -> None:
        """Reset particle to initial state for reuse."""
        self.state = ParticleState.DEAD
        self.age = 0.0
        self.lifetime = 1.0
        self.position = Vec3()
        self.velocity = Vec3()
        self.acceleration = Vec3()
        self.color = Vec4(1, 1, 1, 1)
        self.size = 1.0
        self.rotation = 0.0
        self.angular_velocity = 0.0
        self.custom_data.clear()


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class EmitterConfig:
    """
    Configuration for particle emitter from @particle_emitter decorator.

    Attributes:
        max_particles: Maximum number of alive particles
        simulation: Simulation mode (cpu/gpu/auto)
        budget_category: Optional category for budget tracking
        warmup_time: Time to pre-simulate at startup
        loop: Whether to loop emission
        duration: Total emission duration (0 = infinite)
        prewarm: Pre-warm particle system before first render
    """

    max_particles: int = DEFAULT_MAX_PARTICLES
    simulation: SimulationMode = SimulationMode.AUTO
    budget_category: Optional[str] = None
    warmup_time: float = DEFAULT_WARMUP_TIME
    loop: bool = True
    duration: float = 0.0  # 0 = infinite
    prewarm: bool = False

    @classmethod
    def from_decorator_params(
        cls,
        max_particles: int = DEFAULT_MAX_PARTICLES,
        simulation: str = "auto",
        budget_category: Optional[str] = None,
        **kwargs: Any,
    ) -> "EmitterConfig":
        """Create config from @particle_emitter decorator parameters."""
        sim_mode = {
            "cpu": SimulationMode.CPU,
            "gpu": SimulationMode.GPU,
            "auto": SimulationMode.AUTO,
        }.get(simulation.lower(), SimulationMode.AUTO)

        return cls(
            max_particles=max_particles,
            simulation=sim_mode,
            budget_category=budget_category,
            **kwargs,
        )


# =============================================================================
# PARTICLE POOL
# =============================================================================


class ParticlePool:
    """
    Ring buffer pool of particles with max alive limit and recycling.

    Uses a free list approach for O(1) allocation/deallocation.
    Particles are reused to avoid GC pressure.

    Attributes:
        max_particles: Maximum number of particles in pool
        alive_count: Current number of alive particles
    """

    def __init__(self, max_particles: int = DEFAULT_MAX_PARTICLES) -> None:
        self._max_particles = max_particles
        self._particles: list[Particle] = [
            Particle() for _ in range(max_particles)
        ]
        self._free_indices: list[int] = list(range(max_particles))
        self._alive_indices: set[int] = set()
        self._alive_count = 0
        # O(1) reverse lookup from particle id to index
        self._particle_to_index: dict[int, int] = {
            id(p): i for i, p in enumerate(self._particles)
        }

    @property
    def max_particles(self) -> int:
        return self._max_particles

    @property
    def alive_count(self) -> int:
        return self._alive_count

    @property
    def free_count(self) -> int:
        return len(self._free_indices)

    def allocate(self) -> Optional[Particle]:
        """
        Allocate a particle from the pool.

        Returns:
            Particle if available, None if pool is exhausted
        """
        if not self._free_indices:
            return None

        index = self._free_indices.pop()
        particle = self._particles[index]
        particle.reset()
        particle.state = ParticleState.ALIVE
        self._alive_indices.add(index)
        self._alive_count += 1
        return particle

    def deallocate(self, particle: Particle) -> None:
        """Return a particle to the pool for reuse. O(1) operation."""
        # Use O(1) lookup instead of O(n) list.index()
        particle_id = id(particle)
        if particle_id not in self._particle_to_index:
            return  # Particle not from this pool

        index = self._particle_to_index[particle_id]
        if index in self._alive_indices:
            self._alive_indices.remove(index)
            self._free_indices.append(index)
            particle.state = ParticleState.DEAD
            self._alive_count -= 1

    def iter_alive(self) -> Iterator[Particle]:
        """Iterate over all alive particles."""
        for index in self._alive_indices.copy():
            particle = self._particles[index]
            if particle.is_alive:
                yield particle

    def kill_all(self) -> None:
        """Kill all particles and return them to the pool."""
        for index in self._alive_indices.copy():
            self._particles[index].state = ParticleState.DEAD
            self._free_indices.append(index)
        self._alive_indices.clear()
        self._alive_count = 0

    def compact(self) -> None:
        """Compact the pool by removing dead particles from alive list."""
        dead_indices = []
        for index in self._alive_indices:
            if not self._particles[index].is_alive:
                dead_indices.append(index)

        for index in dead_indices:
            self._alive_indices.remove(index)
            self._free_indices.append(index)
            self._alive_count -= 1


# =============================================================================
# PARTICLE BUDGET
# =============================================================================


@dataclass
class BudgetAllocation:
    """Budget allocation for a single category."""

    category: str
    max_particles: int
    current_particles: int = 0
    max_memory_bytes: int = 0
    current_memory_bytes: int = 0
    priority: int = 1  # Higher = more important, less likely to be culled

    @property
    def particle_usage(self) -> float:
        """Fraction of particle budget used (0-1)."""
        if self.max_particles <= 0:
            return 0.0
        return self.current_particles / self.max_particles

    @property
    def memory_usage(self) -> float:
        """Fraction of memory budget used (0-1)."""
        if self.max_memory_bytes <= 0:
            return 0.0
        return self.current_memory_bytes / self.max_memory_bytes

    def can_allocate(self, count: int = 1) -> bool:
        """Check if we can allocate more particles."""
        return (self.current_particles + count) <= self.max_particles


class ParticleBudget:
    """
    Category-based budget management for particle systems.

    Tracks particle counts and memory usage per category.
    Enforces limits and provides usage statistics.

    Categories allow separate budgets for different effect types:
    - "ambient": Low-priority background particles
    - "gameplay": Medium-priority gameplay effects
    - "critical": High-priority important effects
    """

    # Default budget allocations - use centralized constants
    DEFAULT_BUDGETS: dict[str, tuple[int, int]] = {
        "ambient": PARTICLE_CONSTANTS.BUDGET_AMBIENT,
        "gameplay": PARTICLE_CONSTANTS.BUDGET_GAMEPLAY,
        "critical": PARTICLE_CONSTANTS.BUDGET_CRITICAL,
        "default": PARTICLE_CONSTANTS.BUDGET_DEFAULT,
    }

    def __init__(self) -> None:
        self._allocations: dict[str, BudgetAllocation] = {}
        self._total_limit = PARTICLE_CONSTANTS.BUDGET_TOTAL_LIMIT
        self._total_particles = 0

        # Initialize default budgets
        for category, (max_p, priority) in self.DEFAULT_BUDGETS.items():
            self._allocations[category] = BudgetAllocation(
                category=category,
                max_particles=max_p,
                priority=priority,
            )

    def get_allocation(self, category: Optional[str]) -> BudgetAllocation:
        """Get budget allocation for a category."""
        cat = category or "default"
        if cat not in self._allocations:
            # Create new category with default budget
            self._allocations[cat] = BudgetAllocation(
                category=cat,
                max_particles=self.DEFAULT_BUDGETS["default"][0],
                priority=self.DEFAULT_BUDGETS["default"][1],
            )
        return self._allocations[cat]

    def set_budget(
        self,
        category: str,
        max_particles: int,
        priority: int = 25,
        max_memory_bytes: int = 0,
    ) -> None:
        """Set budget for a category."""
        self._allocations[category] = BudgetAllocation(
            category=category,
            max_particles=max_particles,
            priority=priority,
            max_memory_bytes=max_memory_bytes,
        )

    def request_particles(
        self, category: Optional[str], count: int
    ) -> int:
        """
        Request particle allocation from budget.

        Returns:
            Number of particles actually allocated (may be less than requested)
        """
        allocation = self.get_allocation(category)

        # Check category limit
        available_in_category = max(
            0, allocation.max_particles - allocation.current_particles
        )

        # Check global limit
        available_globally = max(0, self._total_limit - self._total_particles)

        # Allocate the minimum of all constraints
        allocated = min(count, available_in_category, available_globally)

        if allocated > 0:
            allocation.current_particles += allocated
            self._total_particles += allocated

        return allocated

    def release_particles(self, category: Optional[str], count: int) -> None:
        """Release particles back to budget."""
        allocation = self.get_allocation(category)
        release_count = min(count, allocation.current_particles)
        allocation.current_particles -= release_count
        self._total_particles -= release_count

    def get_total_usage(self) -> float:
        """Get overall particle budget usage (0-1)."""
        if self._total_limit <= 0:
            return 0.0
        return self._total_particles / self._total_limit

    def get_category_stats(self) -> dict[str, dict[str, Any]]:
        """Get usage statistics for all categories."""
        return {
            cat: {
                "current": alloc.current_particles,
                "max": alloc.max_particles,
                "usage": alloc.particle_usage,
                "priority": alloc.priority,
            }
            for cat, alloc in self._allocations.items()
        }


# =============================================================================
# PARTICLE EMITTER
# =============================================================================


class ParticleEmitter:
    """
    Particle emitter with lifecycle management.

    Manages the full lifecycle of a particle system:
    - Spawn: Create new particles via spawn modules
    - Update: Apply update modules (physics, forces, etc.)
    - Render: Prepare particles for rendering
    - Death: Handle particle death and recycling

    Attributes:
        config: Emitter configuration
        state: Current emitter lifecycle state
        age: Time since emitter started
    """

    def __init__(
        self,
        config: Optional[EmitterConfig] = None,
        budget: Optional[ParticleBudget] = None,
    ) -> None:
        self._config = config or EmitterConfig()
        self._budget = budget
        self._pool = ParticlePool(self._config.max_particles)
        self._state = EmitterState.INACTIVE
        self._age = 0.0
        self._emission_time = 0.0

        # Module lists
        self._spawn_modules: list["ParticleModule"] = []
        self._update_modules: list["ParticleModule"] = []
        self._render_modules: list["ParticleModule"] = []

        # Simulation mode (resolved from AUTO)
        self._resolved_simulation = self._resolve_simulation_mode()

        # Statistics
        self._spawn_count = 0
        self._death_count = 0
        self._frame_count = 0

        # Optional callbacks
        self._on_particle_spawn: Optional[Callable[[Particle], None]] = None
        self._on_particle_death: Optional[Callable[[Particle], None]] = None
        self._on_emitter_complete: Optional[Callable[[], None]] = None

    @property
    def config(self) -> EmitterConfig:
        return self._config

    @property
    def state(self) -> EmitterState:
        return self._state

    @property
    def age(self) -> float:
        return self._age

    @property
    def alive_count(self) -> int:
        return self._pool.alive_count

    @property
    def simulation_mode(self) -> SimulationMode:
        return self._resolved_simulation

    def _resolve_simulation_mode(self) -> SimulationMode:
        """Resolve AUTO simulation mode to CPU or GPU."""
        if self._config.simulation != SimulationMode.AUTO:
            return self._config.simulation

        # Auto mode: choose based on particle count
        if self._config.max_particles >= DEFAULT_GPU_THRESHOLD:
            return SimulationMode.GPU
        return SimulationMode.CPU

    def add_spawn_module(self, module: "ParticleModule") -> None:
        """Add a module to the spawn phase."""
        self._spawn_modules.append(module)

    def add_update_module(self, module: "ParticleModule") -> None:
        """Add a module to the update phase."""
        self._update_modules.append(module)

    def add_render_module(self, module: "ParticleModule") -> None:
        """Add a module to the render phase."""
        self._render_modules.append(module)

    def start(self) -> None:
        """Start the emitter."""
        if self._state == EmitterState.INACTIVE:
            self._state = EmitterState.WARMING_UP
            self._age = 0.0
            self._emission_time = 0.0

            # Handle prewarm
            if self._config.prewarm and self._config.warmup_time > 0:
                self._prewarm()
            else:
                self._state = EmitterState.ACTIVE

    def _prewarm(self) -> None:
        """Pre-warm the particle system."""
        warmup_dt = 1.0 / PARTICLE_CONSTANTS.PREWARM_FPS
        warmup_steps = int(self._config.warmup_time / warmup_dt)

        for _ in range(warmup_steps):
            self._update_internal(warmup_dt, spawn=True)

        self._state = EmitterState.ACTIVE

    def stop(self, immediate: bool = False) -> None:
        """
        Stop the emitter.

        Args:
            immediate: If True, kill all particles immediately.
                      If False, let existing particles die naturally.
        """
        if immediate:
            self._pool.kill_all()
            self._state = EmitterState.STOPPED
            if self._on_emitter_complete:
                self._on_emitter_complete()
        else:
            self._state = EmitterState.STOPPING

    def update(self, dt: float) -> None:
        """
        Update the particle system.

        Args:
            dt: Delta time in seconds
        """
        if self._state == EmitterState.INACTIVE:
            return

        if self._state == EmitterState.STOPPED:
            return

        self._age += dt
        self._frame_count += 1

        # Check duration
        should_spawn = self._state == EmitterState.ACTIVE
        if self._config.duration > 0 and self._age >= self._config.duration:
            if self._config.loop:
                self._age = 0.0
            else:
                should_spawn = False
                if self._state == EmitterState.ACTIVE:
                    self._state = EmitterState.STOPPING

        self._update_internal(dt, spawn=should_spawn)

        # Check if all particles are dead when stopping
        if self._state == EmitterState.STOPPING and self._pool.alive_count == 0:
            self._state = EmitterState.STOPPED
            if self._on_emitter_complete:
                self._on_emitter_complete()

    def _update_internal(self, dt: float, spawn: bool = True) -> None:
        """Internal update implementation."""
        # Phase 1: Spawn new particles
        if spawn:
            self._spawn_phase(dt)

        # Phase 2: Update existing particles
        self._update_phase(dt)

        # Phase 3: Kill dead particles
        self._death_phase()

        # Compact pool periodically
        if self._frame_count % PARTICLE_CONSTANTS.COMPACT_INTERVAL == 0:
            self._pool.compact()

    def _spawn_phase(self, dt: float) -> None:
        """Execute spawn modules to create new particles."""
        for module in self._spawn_modules:
            if module.is_active_for_lod(0):  # TODO: Pass actual LOD level
                particles_to_spawn = module.get_spawn_count(dt)
                for _ in range(particles_to_spawn):
                    particle = self._spawn_particle()
                    if particle:
                        module.apply_to_particle(particle, dt)

    def _spawn_particle(self) -> Optional[Particle]:
        """Spawn a single particle if budget allows."""
        # Check budget
        if self._budget:
            allocated = self._budget.request_particles(
                self._config.budget_category, 1
            )
            if allocated == 0:
                return None

        particle = self._pool.allocate()
        if particle:
            self._spawn_count += 1
            if self._on_particle_spawn:
                self._on_particle_spawn(particle)
        elif self._budget:
            # Failed to allocate from pool, release budget
            self._budget.release_particles(self._config.budget_category, 1)

        return particle

    def _update_phase(self, dt: float) -> None:
        """Execute update modules on all alive particles."""
        for particle in self._pool.iter_alive():
            # Update age
            particle.age += dt

            # Check for death
            if particle.age >= particle.lifetime:
                particle.state = ParticleState.DYING
                continue

            # Apply update modules
            for module in self._update_modules:
                if module.is_active_for_lod(0):  # TODO: Pass actual LOD level
                    module.apply_to_particle(particle, dt)

            # Basic physics integration
            particle.velocity = particle.velocity + particle.acceleration * dt
            particle.position = particle.position + particle.velocity * dt
            particle.rotation += particle.angular_velocity * dt

    def _death_phase(self) -> None:
        """Handle dying particles."""
        for particle in self._pool.iter_alive():
            if particle.state == ParticleState.DYING:
                if self._on_particle_death:
                    self._on_particle_death(particle)

                self._pool.deallocate(particle)
                self._death_count += 1

                # Release budget
                if self._budget:
                    self._budget.release_particles(
                        self._config.budget_category, 1
                    )

    def iter_particles(self) -> Iterator[Particle]:
        """Iterate over all alive particles for rendering."""
        return self._pool.iter_alive()

    def get_stats(self) -> dict[str, Any]:
        """Get emitter statistics."""
        return {
            "state": self._state.name,
            "age": self._age,
            "alive_count": self._pool.alive_count,
            "spawn_count": self._spawn_count,
            "death_count": self._death_count,
            "frame_count": self._frame_count,
            "simulation_mode": self._resolved_simulation.name,
            "pool_usage": self._pool.alive_count / self._pool.max_particles,
        }


# =============================================================================
# PARTICLE SYSTEM MANAGER
# =============================================================================


class ParticleSystemManager:
    """
    Manages multiple particle emitters.

    Provides centralized budget control, LOD management, and batch operations.
    """

    def __init__(self) -> None:
        self._emitters: dict[str, ParticleEmitter] = {}
        self._budget = ParticleBudget()
        self._current_lod_level = 0

    @property
    def budget(self) -> ParticleBudget:
        return self._budget

    def create_emitter(
        self,
        name: str,
        config: Optional[EmitterConfig] = None,
    ) -> ParticleEmitter:
        """Create and register a new emitter."""
        emitter = ParticleEmitter(config=config, budget=self._budget)
        self._emitters[name] = emitter
        return emitter

    def get_emitter(self, name: str) -> Optional[ParticleEmitter]:
        """Get an emitter by name."""
        return self._emitters.get(name)

    def remove_emitter(self, name: str) -> None:
        """Remove an emitter."""
        if name in self._emitters:
            emitter = self._emitters[name]
            emitter.stop(immediate=True)
            del self._emitters[name]

    def update_all(self, dt: float) -> None:
        """Update all emitters."""
        for emitter in self._emitters.values():
            emitter.update(dt)

    def set_lod_level(self, level: int) -> None:
        """Set the global LOD level for all emitters."""
        self._current_lod_level = level

    def get_total_particle_count(self) -> int:
        """Get total alive particles across all emitters."""
        return sum(e.alive_count for e in self._emitters.values())

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        return {
            "emitter_count": len(self._emitters),
            "total_particles": self.get_total_particle_count(),
            "budget_usage": self._budget.get_total_usage(),
            "category_stats": self._budget.get_category_stats(),
        }


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Enums
    "SimulationMode",
    "EmitterState",
    "ParticleState",
    # Data structures
    "Vec3",
    "Vec4",
    "Particle",
    # Configuration
    "EmitterConfig",
    # Core classes
    "ParticlePool",
    "ParticleBudget",
    "BudgetAllocation",
    "ParticleEmitter",
    "ParticleSystemManager",
]
