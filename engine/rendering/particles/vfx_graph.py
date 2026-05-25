"""
Node-Based VFX Authoring System.

Provides a graph-based interface for creating complex VFX effects by connecting
reusable modules. Supports visual editing workflows and runtime compilation.

Architecture:
    VFXContext - Execution context (Spawn, Update, Render)
    VFXModule - Reusable behavior nodes
    VFXParameter - Exposed controls for tweaking
    VFXEvent - Inter-system communication
    VFXGraph - Container that connects modules and compiles to particle system

Supports @vfx_event decorator for event triggers (spawn, death, collision, custom).
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from engine.rendering.particles.particle_system import (
    EmitterConfig,
    Particle,
    ParticleEmitter,
    ParticleState,
    SimulationMode,
    Vec3,
    Vec4,
)
from engine.rendering.particles.particle_modules import (
    BillboardRenderer,
    BurstEmitter,
    ColorOverLifeModule,
    GravityModule,
    LifetimeModule,
    ModuleStage,
    ParticleModule,
    RateEmitter,
    ShapeEmitter,
    SizeOverLifeModule,
    VelocityModule,
)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class VFXContextType(Enum):
    """VFX graph execution context."""

    SPAWN = auto()  # Particle spawn context
    UPDATE = auto()  # Per-frame update context
    RENDER = auto()  # Rendering context
    EVENT = auto()  # Event handling context
    GLOBAL = auto()  # Global/initialization context


class VFXParameterType(Enum):
    """Types for exposed VFX parameters."""

    FLOAT = auto()
    FLOAT2 = auto()
    FLOAT3 = auto()
    FLOAT4 = auto()
    INT = auto()
    BOOL = auto()
    COLOR = auto()
    CURVE = auto()
    GRADIENT = auto()
    TEXTURE = auto()
    MESH = auto()


class VFXEventTrigger(Enum):
    """VFX event trigger types (from @vfx_event decorator)."""

    SPAWN = auto()  # Triggered when particle spawns
    DEATH = auto()  # Triggered when particle dies
    COLLISION = auto()  # Triggered on particle collision
    CUSTOM = auto()  # Custom trigger


class VFXNodeType(Enum):
    """Types of nodes in VFX graph."""

    # Sources
    EMITTER = auto()  # Particle emitter configuration
    SPAWN_RATE = auto()  # Spawn rate control
    SPAWN_BURST = auto()  # Burst spawn control

    # Generators
    RANDOM = auto()  # Random value generator
    NOISE = auto()  # Noise generator
    CURVE = auto()  # Curve sampler
    GRADIENT = auto()  # Gradient sampler

    # Forces
    GRAVITY = auto()  # Gravity force
    WIND = auto()  # Wind force
    TURBULENCE = auto()  # Turbulence force
    VORTEX = auto()  # Vortex force
    ATTRACTION = auto()  # Point attraction

    # Modifiers
    SIZE = auto()  # Size modifier
    COLOR = auto()  # Color modifier
    ROTATION = auto()  # Rotation modifier
    LIFETIME = auto()  # Lifetime modifier
    VELOCITY = auto()  # Velocity modifier

    # Collision
    COLLISION = auto()  # Collision handling

    # Rendering
    BILLBOARD = auto()  # Billboard renderer
    MESH = auto()  # Mesh particle renderer
    TRAIL = auto()  # Trail renderer

    # Events
    EVENT_SPAWN = auto()  # On spawn event
    EVENT_DEATH = auto()  # On death event
    EVENT_COLLISION = auto()  # On collision event
    EVENT_CUSTOM = auto()  # Custom event

    # Utility
    BRANCH = auto()  # Conditional branch
    SEQUENCE = auto()  # Sequential execution
    PARALLEL = auto()  # Parallel execution


# =============================================================================
# VFX CONTEXT
# =============================================================================


@dataclass
class VFXContext:
    """
    Execution context for VFX graph evaluation.

    Contains the current particle, delta time, and system state.
    """

    context_type: VFXContextType = VFXContextType.UPDATE
    particle: Optional[Particle] = None
    dt: float = 0.0
    time: float = 0.0
    emitter_position: Vec3 = field(default_factory=Vec3)
    emitter_velocity: Vec3 = field(default_factory=Vec3)

    # Per-frame temporaries
    scratch: dict[str, Any] = field(default_factory=dict)

    # Accumulated outputs
    outputs: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# VFX PARAMETER
# =============================================================================


@dataclass
class VFXParameter:
    """
    Exposed parameter for VFX graph.

    Can be tweaked in editor or at runtime.
    """

    name: str
    param_type: VFXParameterType
    default_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    description: str = ""
    category: str = "General"
    visible_in_editor: bool = True

    # Runtime value (may differ from default)
    _value: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._value is None:
            self._value = self.default_value

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, new_value: Any) -> None:
        # Apply constraints
        if self.min_value is not None and new_value < self.min_value:
            new_value = self.min_value
        if self.max_value is not None and new_value > self.max_value:
            new_value = self.max_value
        self._value = new_value

    def reset(self) -> None:
        """Reset to default value."""
        self._value = self.default_value


# =============================================================================
# VFX EVENT
# =============================================================================


@dataclass(frozen=True)
class VFXEventConfig:
    """
    Configuration for VFX event from @vfx_event decorator.

    Attributes:
        trigger: Event trigger type (spawn, death, collision, custom)
    """

    trigger: VFXEventTrigger = VFXEventTrigger.CUSTOM

    @classmethod
    def from_decorator_params(
        cls,
        trigger: str,
        **kwargs: Any,
    ) -> "VFXEventConfig":
        """Create config from @vfx_event decorator parameters."""
        trigger_map = {
            "spawn": VFXEventTrigger.SPAWN,
            "death": VFXEventTrigger.DEATH,
            "collision": VFXEventTrigger.COLLISION,
            "custom": VFXEventTrigger.CUSTOM,
        }
        return cls(trigger=trigger_map.get(trigger.lower(), VFXEventTrigger.CUSTOM))


class VFXEvent:
    """
    VFX event for inter-system communication.

    Events can trigger actions in other VFX systems or game systems.
    """

    def __init__(
        self,
        name: str,
        trigger: VFXEventTrigger = VFXEventTrigger.CUSTOM,
    ) -> None:
        self._name = name
        self._trigger = trigger
        self._handlers: list[Callable[[VFXContext], None]] = []
        self._payload_schema: dict[str, type] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def trigger(self) -> VFXEventTrigger:
        return self._trigger

    def set_payload_schema(self, schema: dict[str, type]) -> None:
        """Define expected payload data types."""
        self._payload_schema = schema

    def add_handler(self, handler: Callable[[VFXContext], None]) -> None:
        """Register an event handler."""
        self._handlers.append(handler)

    def remove_handler(self, handler: Callable[[VFXContext], None]) -> None:
        """Unregister an event handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    def fire(self, context: VFXContext, payload: Optional[dict[str, Any]] = None) -> None:
        """Fire the event with optional payload."""
        if payload:
            context.outputs.update(payload)
        for handler in self._handlers:
            handler(context)


# =============================================================================
# VFX MODULE (NODE)
# =============================================================================


class VFXModule(ABC):
    """
    Base class for VFX graph nodes.

    Modules are connected in a graph and executed in dependency order.
    """

    def __init__(
        self,
        name: str = "",
        node_type: VFXNodeType = VFXNodeType.EMITTER,
    ) -> None:
        self._id = str(uuid.uuid4())[:8]
        self._name = name or f"{node_type.name}_{self._id}"
        self._node_type = node_type
        self._enabled = True

        # Connections
        self._inputs: dict[str, "VFXConnection"] = {}
        self._outputs: dict[str, "VFXConnection"] = {}

        # Parameters
        self._parameters: dict[str, VFXParameter] = {}

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def node_type(self) -> VFXNodeType:
        return self._node_type

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def add_parameter(self, param: VFXParameter) -> None:
        """Add an exposed parameter."""
        self._parameters[param.name] = param

    def get_parameter(self, name: str) -> Optional[VFXParameter]:
        """Get parameter by name."""
        return self._parameters.get(name)

    def set_parameter_value(self, name: str, value: Any) -> None:
        """Set parameter value."""
        if name in self._parameters:
            self._parameters[name].value = value

    def get_input_names(self) -> list[str]:
        """Get list of input slot names."""
        return list(self._inputs.keys())

    def get_output_names(self) -> list[str]:
        """Get list of output slot names."""
        return list(self._outputs.keys())

    def connect_input(self, slot: str, connection: "VFXConnection") -> None:
        """Connect an input slot."""
        self._inputs[slot] = connection

    def connect_output(self, slot: str, connection: "VFXConnection") -> None:
        """Connect an output slot."""
        self._outputs[slot] = connection

    @abstractmethod
    def evaluate(self, context: VFXContext) -> None:
        """Evaluate the module with the given context."""
        pass

    def to_particle_module(self) -> Optional[ParticleModule]:
        """Convert to a ParticleModule if applicable."""
        return None


@dataclass
class VFXConnection:
    """Connection between two VFX modules."""

    source_module_id: str
    source_slot: str
    target_module_id: str
    target_slot: str

    def __hash__(self) -> int:
        return hash(
            (
                self.source_module_id,
                self.source_slot,
                self.target_module_id,
                self.target_slot,
            )
        )


# =============================================================================
# CONCRETE VFX MODULES
# =============================================================================


class VFXEmitterModule(VFXModule):
    """Emitter configuration module."""

    def __init__(
        self,
        max_particles: int = 1000,
        simulation: SimulationMode = SimulationMode.AUTO,
        loop: bool = True,
        duration: float = 0.0,
    ) -> None:
        super().__init__(node_type=VFXNodeType.EMITTER)
        self._max_particles = max_particles
        self._simulation = simulation
        self._loop = loop
        self._duration = duration

        # Add parameters
        self.add_parameter(
            VFXParameter(
                "max_particles",
                VFXParameterType.INT,
                max_particles,
                min_value=1,
                max_value=1000000,
            )
        )
        self.add_parameter(
            VFXParameter("loop", VFXParameterType.BOOL, loop)
        )
        self.add_parameter(
            VFXParameter("duration", VFXParameterType.FLOAT, duration, min_value=0)
        )

    def evaluate(self, context: VFXContext) -> None:
        context.outputs["max_particles"] = self._parameters["max_particles"].value
        context.outputs["loop"] = self._parameters["loop"].value
        context.outputs["duration"] = self._parameters["duration"].value

    def to_emitter_config(self) -> EmitterConfig:
        """Convert to EmitterConfig."""
        return EmitterConfig(
            max_particles=self._parameters["max_particles"].value,
            simulation=self._simulation,
            loop=self._parameters["loop"].value,
            duration=self._parameters["duration"].value,
        )


class VFXSpawnRateModule(VFXModule):
    """Spawn rate control module."""

    def __init__(self, rate: float = 100.0) -> None:
        super().__init__(node_type=VFXNodeType.SPAWN_RATE)
        self.add_parameter(
            VFXParameter(
                "rate",
                VFXParameterType.FLOAT,
                rate,
                min_value=0,
                max_value=100000,
                description="Particles per second",
            )
        )

    def evaluate(self, context: VFXContext) -> None:
        context.outputs["spawn_rate"] = self._parameters["rate"].value

    def to_particle_module(self) -> Optional[ParticleModule]:
        return RateEmitter(rate=self._parameters["rate"].value)


class VFXBurstModule(VFXModule):
    """Burst spawn control module."""

    def __init__(self, count: int = 10, repeat_interval: float = 0.0) -> None:
        super().__init__(node_type=VFXNodeType.SPAWN_BURST)
        self.add_parameter(
            VFXParameter("count", VFXParameterType.INT, count, min_value=1)
        )
        self.add_parameter(
            VFXParameter("repeat_interval", VFXParameterType.FLOAT, repeat_interval, min_value=0)
        )

    def evaluate(self, context: VFXContext) -> None:
        context.outputs["burst_count"] = self._parameters["count"].value
        context.outputs["burst_interval"] = self._parameters["repeat_interval"].value

    def to_particle_module(self) -> Optional[ParticleModule]:
        return BurstEmitter(
            count=self._parameters["count"].value,
            repeat_interval=self._parameters["repeat_interval"].value,
        )


class VFXGravityModule(VFXModule):
    """Gravity force module."""

    def __init__(self, gravity: Vec3 = None) -> None:
        super().__init__(node_type=VFXNodeType.GRAVITY)
        g = gravity or Vec3(0, -9.81, 0)
        self.add_parameter(
            VFXParameter("gravity_x", VFXParameterType.FLOAT, g.x)
        )
        self.add_parameter(
            VFXParameter("gravity_y", VFXParameterType.FLOAT, g.y)
        )
        self.add_parameter(
            VFXParameter("gravity_z", VFXParameterType.FLOAT, g.z)
        )

    def evaluate(self, context: VFXContext) -> None:
        if context.particle:
            gravity = Vec3(
                self._parameters["gravity_x"].value,
                self._parameters["gravity_y"].value,
                self._parameters["gravity_z"].value,
            )
            context.particle.acceleration = context.particle.acceleration + gravity

    def to_particle_module(self) -> Optional[ParticleModule]:
        return GravityModule(
            gravity=Vec3(
                self._parameters["gravity_x"].value,
                self._parameters["gravity_y"].value,
                self._parameters["gravity_z"].value,
            )
        )


class VFXSizeOverLifeModule(VFXModule):
    """Size over lifetime module."""

    def __init__(
        self,
        start_size: float = 1.0,
        end_size: float = 0.0,
        curve: str = "linear",
    ) -> None:
        super().__init__(node_type=VFXNodeType.SIZE)
        self.add_parameter(
            VFXParameter("start_size", VFXParameterType.FLOAT, start_size, min_value=0)
        )
        self.add_parameter(
            VFXParameter("end_size", VFXParameterType.FLOAT, end_size, min_value=0)
        )

    def evaluate(self, context: VFXContext) -> None:
        if context.particle:
            t = context.particle.normalized_age
            start = self._parameters["start_size"].value
            end = self._parameters["end_size"].value
            context.particle.size = start + (end - start) * t

    def to_particle_module(self) -> Optional[ParticleModule]:
        return SizeOverLifeModule(
            start_size=self._parameters["start_size"].value,
            end_size=self._parameters["end_size"].value,
        )


class VFXColorOverLifeModule(VFXModule):
    """Color over lifetime module."""

    def __init__(
        self,
        start_color: Vec4 = None,
        end_color: Vec4 = None,
    ) -> None:
        super().__init__(node_type=VFXNodeType.COLOR)
        sc = start_color or Vec4(1, 1, 1, 1)
        ec = end_color or Vec4(1, 1, 1, 0)

        self.add_parameter(
            VFXParameter("start_color", VFXParameterType.COLOR, sc)
        )
        self.add_parameter(
            VFXParameter("end_color", VFXParameterType.COLOR, ec)
        )

    def evaluate(self, context: VFXContext) -> None:
        if context.particle:
            t = context.particle.normalized_age
            start: Vec4 = self._parameters["start_color"].value
            end: Vec4 = self._parameters["end_color"].value
            context.particle.color = start.lerp(end, t)

    def to_particle_module(self) -> Optional[ParticleModule]:
        return ColorOverLifeModule(
            start_color=self._parameters["start_color"].value,
            end_color=self._parameters["end_color"].value,
        )


class VFXEventModule(VFXModule):
    """Event handling module."""

    def __init__(
        self,
        event_trigger: VFXEventTrigger = VFXEventTrigger.CUSTOM,
        event_name: str = "custom_event",
    ) -> None:
        super().__init__(
            node_type={
                VFXEventTrigger.SPAWN: VFXNodeType.EVENT_SPAWN,
                VFXEventTrigger.DEATH: VFXNodeType.EVENT_DEATH,
                VFXEventTrigger.COLLISION: VFXNodeType.EVENT_COLLISION,
                VFXEventTrigger.CUSTOM: VFXNodeType.EVENT_CUSTOM,
            }.get(event_trigger, VFXNodeType.EVENT_CUSTOM)
        )
        self._event = VFXEvent(event_name, event_trigger)
        self._child_modules: list[VFXModule] = []

    @property
    def event(self) -> VFXEvent:
        return self._event

    def add_child(self, module: VFXModule) -> None:
        """Add a module to execute when event fires."""
        self._child_modules.append(module)

    def evaluate(self, context: VFXContext) -> None:
        # Execute child modules
        for child in self._child_modules:
            if child.enabled:
                child.evaluate(context)

        # Fire the event
        self._event.fire(context)


# =============================================================================
# VFX GRAPH
# =============================================================================


class VFXGraph:
    """
    Node-based VFX graph that compiles to a particle system.

    Connects VFX modules and manages their execution order.
    """

    def __init__(self, name: str = "VFXGraph") -> None:
        self._name = name
        self._id = str(uuid.uuid4())[:8]

        # Graph structure
        self._modules: dict[str, VFXModule] = {}
        self._connections: set[VFXConnection] = set()

        # Categorized modules for execution
        self._spawn_modules: list[VFXModule] = []
        self._update_modules: list[VFXModule] = []
        self._render_modules: list[VFXModule] = []
        self._event_modules: dict[VFXEventTrigger, list[VFXEventModule]] = {
            VFXEventTrigger.SPAWN: [],
            VFXEventTrigger.DEATH: [],
            VFXEventTrigger.COLLISION: [],
            VFXEventTrigger.CUSTOM: [],
        }

        # Exposed parameters (aggregated from modules)
        self._exposed_parameters: dict[str, VFXParameter] = {}

        # Compiled emitter (cached)
        self._compiled_emitter: Optional[ParticleEmitter] = None
        self._dirty = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> str:
        return self._id

    def add_module(self, module: VFXModule) -> str:
        """
        Add a module to the graph.

        Returns:
            Module ID
        """
        self._modules[module.id] = module
        self._dirty = True
        return module.id

    def remove_module(self, module_id: str) -> None:
        """Remove a module from the graph."""
        if module_id in self._modules:
            del self._modules[module_id]
            # Remove connections involving this module
            self._connections = {
                c
                for c in self._connections
                if c.source_module_id != module_id and c.target_module_id != module_id
            }
            self._dirty = True

    def get_module(self, module_id: str) -> Optional[VFXModule]:
        """Get a module by ID."""
        return self._modules.get(module_id)

    def connect(
        self,
        source_id: str,
        source_slot: str,
        target_id: str,
        target_slot: str,
    ) -> bool:
        """
        Connect two modules.

        Returns:
            True if connection was created
        """
        if source_id not in self._modules or target_id not in self._modules:
            return False

        connection = VFXConnection(source_id, source_slot, target_id, target_slot)
        self._connections.add(connection)

        # Update module connections
        self._modules[source_id].connect_output(source_slot, connection)
        self._modules[target_id].connect_input(target_slot, connection)

        self._dirty = True
        return True

    def disconnect(
        self,
        source_id: str,
        source_slot: str,
        target_id: str,
        target_slot: str,
    ) -> bool:
        """
        Disconnect two modules.

        Returns:
            True if connection was removed
        """
        connection = VFXConnection(source_id, source_slot, target_id, target_slot)
        if connection in self._connections:
            self._connections.remove(connection)
            self._dirty = True
            return True
        return False

    def expose_parameter(self, module_id: str, param_name: str, exposed_name: str) -> None:
        """Expose a module parameter at the graph level."""
        module = self._modules.get(module_id)
        if module:
            param = module.get_parameter(param_name)
            if param:
                # Create a linked parameter
                self._exposed_parameters[exposed_name] = param

    def set_parameter(self, name: str, value: Any) -> None:
        """Set an exposed parameter value."""
        if name in self._exposed_parameters:
            self._exposed_parameters[name].value = value

    def _categorize_modules(self) -> None:
        """Categorize modules by execution phase."""
        self._spawn_modules.clear()
        self._update_modules.clear()
        self._render_modules.clear()
        for trigger in self._event_modules:
            self._event_modules[trigger].clear()

        for module in self._modules.values():
            node_type = module.node_type

            # Spawn modules
            if node_type in (
                VFXNodeType.EMITTER,
                VFXNodeType.SPAWN_RATE,
                VFXNodeType.SPAWN_BURST,
            ):
                self._spawn_modules.append(module)

            # Update modules (forces, modifiers)
            elif node_type in (
                VFXNodeType.GRAVITY,
                VFXNodeType.WIND,
                VFXNodeType.TURBULENCE,
                VFXNodeType.VORTEX,
                VFXNodeType.ATTRACTION,
                VFXNodeType.SIZE,
                VFXNodeType.COLOR,
                VFXNodeType.ROTATION,
                VFXNodeType.LIFETIME,
                VFXNodeType.VELOCITY,
                VFXNodeType.COLLISION,
            ):
                self._update_modules.append(module)

            # Render modules
            elif node_type in (
                VFXNodeType.BILLBOARD,
                VFXNodeType.MESH,
                VFXNodeType.TRAIL,
            ):
                self._render_modules.append(module)

            # Event modules
            elif node_type in (
                VFXNodeType.EVENT_SPAWN,
                VFXNodeType.EVENT_DEATH,
                VFXNodeType.EVENT_COLLISION,
                VFXNodeType.EVENT_CUSTOM,
            ):
                if isinstance(module, VFXEventModule):
                    trigger = module.event.trigger
                    self._event_modules[trigger].append(module)

    def compile(self) -> ParticleEmitter:
        """
        Compile the VFX graph to a ParticleEmitter.

        Returns:
            Configured ParticleEmitter ready for use
        """
        if not self._dirty and self._compiled_emitter:
            return self._compiled_emitter

        # Categorize modules
        self._categorize_modules()

        # Find emitter configuration
        emitter_config = EmitterConfig()
        for module in self._spawn_modules:
            if isinstance(module, VFXEmitterModule):
                emitter_config = module.to_emitter_config()
                break

        # Create emitter
        emitter = ParticleEmitter(config=emitter_config)

        # Add spawn modules
        for module in self._spawn_modules:
            pm = module.to_particle_module()
            if pm and pm.stage == ModuleStage.SPAWN:
                emitter.add_spawn_module(pm)

        # Add update modules
        for module in self._update_modules:
            pm = module.to_particle_module()
            if pm:
                emitter.add_update_module(pm)

        # Add render modules
        for module in self._render_modules:
            pm = module.to_particle_module()
            if pm:
                emitter.add_render_module(pm)

        # Set up event handlers
        def make_event_handler(
            event_modules: list[VFXEventModule],
        ) -> Callable[[Particle], None]:
            def handler(particle: Particle) -> None:
                context = VFXContext(
                    context_type=VFXContextType.EVENT,
                    particle=particle,
                )
                for module in event_modules:
                    module.evaluate(context)

            return handler

        if self._event_modules[VFXEventTrigger.SPAWN]:
            emitter._on_particle_spawn = make_event_handler(
                self._event_modules[VFXEventTrigger.SPAWN]
            )

        if self._event_modules[VFXEventTrigger.DEATH]:
            emitter._on_particle_death = make_event_handler(
                self._event_modules[VFXEventTrigger.DEATH]
            )

        self._compiled_emitter = emitter
        self._dirty = False
        return emitter

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary for saving."""
        return {
            "name": self._name,
            "id": self._id,
            "modules": [
                {
                    "id": m.id,
                    "name": m.name,
                    "type": m.node_type.name,
                    "enabled": m.enabled,
                    "parameters": {
                        name: p.value for name, p in m._parameters.items()
                    },
                }
                for m in self._modules.values()
            ],
            "connections": [
                {
                    "source_id": c.source_module_id,
                    "source_slot": c.source_slot,
                    "target_id": c.target_module_id,
                    "target_slot": c.target_slot,
                }
                for c in self._connections
            ],
            "exposed_parameters": list(self._exposed_parameters.keys()),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VFXGraph":
        """Deserialize graph from dictionary."""
        graph = cls(name=data.get("name", "VFXGraph"))

        # Module factory
        module_types: dict[str, type] = {
            "EMITTER": VFXEmitterModule,
            "SPAWN_RATE": VFXSpawnRateModule,
            "SPAWN_BURST": VFXBurstModule,
            "GRAVITY": VFXGravityModule,
            "SIZE": VFXSizeOverLifeModule,
            "COLOR": VFXColorOverLifeModule,
        }

        # Create modules
        for mod_data in data.get("modules", []):
            mod_type = mod_data.get("type")
            if mod_type in module_types:
                module = module_types[mod_type]()
                # Set parameters
                for name, value in mod_data.get("parameters", {}).items():
                    module.set_parameter_value(name, value)
                module.enabled = mod_data.get("enabled", True)
                graph._modules[mod_data["id"]] = module

        # Create connections
        for conn_data in data.get("connections", []):
            graph.connect(
                conn_data["source_id"],
                conn_data["source_slot"],
                conn_data["target_id"],
                conn_data["target_slot"],
            )

        return graph


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Enums
    "VFXContextType",
    "VFXParameterType",
    "VFXEventTrigger",
    "VFXNodeType",
    # Context
    "VFXContext",
    # Parameter
    "VFXParameter",
    # Event
    "VFXEventConfig",
    "VFXEvent",
    # Modules
    "VFXModule",
    "VFXConnection",
    "VFXEmitterModule",
    "VFXSpawnRateModule",
    "VFXBurstModule",
    "VFXGravityModule",
    "VFXSizeOverLifeModule",
    "VFXColorOverLifeModule",
    "VFXEventModule",
    # Graph
    "VFXGraph",
]
