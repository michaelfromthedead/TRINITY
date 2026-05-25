"""
Tests for the VFX Graph system.

Tests:
    - VFXModule creation and parameters
    - VFXGraph assembly and connections
    - Graph compilation to ParticleEmitter
    - Serialization/deserialization
"""

import pytest

from engine.rendering.particles.particle_system import (
    EmitterState,
    ParticleEmitter,
    SimulationMode,
    Vec3,
    Vec4,
)
from engine.rendering.particles.vfx_graph import (
    VFXContextType,
    VFXParameterType,
    VFXEventTrigger,
    VFXNodeType,
    VFXContext,
    VFXParameter,
    VFXEventConfig,
    VFXEvent,
    VFXModule,
    VFXConnection,
    VFXEmitterModule,
    VFXSpawnRateModule,
    VFXBurstModule,
    VFXGravityModule,
    VFXSizeOverLifeModule,
    VFXColorOverLifeModule,
    VFXEventModule,
    VFXGraph,
)


class TestVFXParameter:
    """Test VFXParameter creation and value handling."""

    def test_creation(self):
        """Test parameter creation with defaults."""
        param = VFXParameter(
            name="rate",
            param_type=VFXParameterType.FLOAT,
            default_value=100.0,
        )
        assert param.name == "rate"
        assert param.param_type == VFXParameterType.FLOAT
        assert param.value == 100.0

    def test_value_clamping(self):
        """Test value is clamped to min/max."""
        param = VFXParameter(
            name="rate",
            param_type=VFXParameterType.FLOAT,
            default_value=50.0,
            min_value=0.0,
            max_value=100.0,
        )

        param.value = 150.0
        assert param.value == 100.0  # Clamped to max

        param.value = -10.0
        assert param.value == 0.0  # Clamped to min

    def test_reset(self):
        """Test resetting to default value."""
        param = VFXParameter(
            name="size",
            param_type=VFXParameterType.FLOAT,
            default_value=1.0,
        )
        param.value = 5.0
        param.reset()
        assert param.value == 1.0


class TestVFXEvent:
    """Test VFXEvent handling."""

    def test_creation(self):
        """Test event creation."""
        event = VFXEvent("on_spawn", VFXEventTrigger.SPAWN)
        assert event.name == "on_spawn"
        assert event.trigger == VFXEventTrigger.SPAWN

    def test_handler_registration(self):
        """Test adding and removing handlers."""
        event = VFXEvent("test_event")
        called = []

        def handler(context):
            called.append(True)

        event.add_handler(handler)

        # Fire event
        context = VFXContext()
        event.fire(context)

        assert len(called) == 1

        # Remove and verify not called again
        event.remove_handler(handler)
        event.fire(context)
        assert len(called) == 1

    def test_payload_passing(self):
        """Test payload is passed to context."""
        event = VFXEvent("test_event")
        received_payload = {}

        def handler(context):
            received_payload.update(context.outputs)

        event.add_handler(handler)
        context = VFXContext()
        event.fire(context, payload={"hit_position": Vec3(1, 2, 3)})

        assert "hit_position" in received_payload


class TestVFXModules:
    """Test concrete VFX module implementations."""

    def test_emitter_module(self):
        """Test emitter module to config conversion."""
        module = VFXEmitterModule(
            max_particles=5000,
            simulation=SimulationMode.GPU,
            loop=True,
            duration=10.0,
        )

        config = module.to_emitter_config()

        assert config.max_particles == 5000
        assert config.simulation == SimulationMode.GPU
        assert config.loop is True
        assert config.duration == 10.0

    def test_spawn_rate_module(self):
        """Test spawn rate module converts to RateEmitter."""
        module = VFXSpawnRateModule(rate=500.0)
        pm = module.to_particle_module()

        assert pm is not None
        # Verify it creates a rate emitter with correct rate
        count = pm.get_spawn_count(0.1)  # 0.1s at 500/s = 50
        assert count == 50

    def test_burst_module(self):
        """Test burst module converts to BurstEmitter."""
        module = VFXBurstModule(count=20, repeat_interval=1.0)
        pm = module.to_particle_module()

        assert pm is not None
        # First burst should give 20
        count = pm.get_spawn_count(0.016)
        assert count == 20

    def test_gravity_module(self):
        """Test gravity module to GravityModule conversion."""
        module = VFXGravityModule(gravity=Vec3(0, -20, 0))
        pm = module.to_particle_module()

        assert pm is not None
        assert pm._gravity.y == -20

    def test_size_over_life_module(self):
        """Test size over life module."""
        module = VFXSizeOverLifeModule(start_size=2.0, end_size=0.0)
        pm = module.to_particle_module()

        assert pm is not None
        assert pm._start_size == 2.0
        assert pm._end_size == 0.0

    def test_color_over_life_module(self):
        """Test color over life module."""
        start = Vec4(1, 0, 0, 1)  # Red
        end = Vec4(0, 0, 1, 0)  # Blue transparent
        module = VFXColorOverLifeModule(start_color=start, end_color=end)
        pm = module.to_particle_module()

        assert pm is not None
        assert pm._start_color.x == 1  # Red channel


class TestVFXGraph:
    """Test VFXGraph assembly and compilation."""

    def test_creation(self):
        """Test graph creation."""
        graph = VFXGraph("FireEffect")
        assert graph.name == "FireEffect"

    def test_add_modules(self):
        """Test adding modules to graph."""
        graph = VFXGraph()

        emitter = VFXEmitterModule(max_particles=1000)
        spawn_rate = VFXSpawnRateModule(rate=100)
        gravity = VFXGravityModule()

        graph.add_module(emitter)
        graph.add_module(spawn_rate)
        graph.add_module(gravity)

        assert graph.get_module(emitter.id) is emitter
        assert graph.get_module(spawn_rate.id) is spawn_rate
        assert graph.get_module(gravity.id) is gravity

    def test_remove_module(self):
        """Test removing modules from graph."""
        graph = VFXGraph()
        module = VFXSpawnRateModule()
        module_id = graph.add_module(module)

        graph.remove_module(module_id)

        assert graph.get_module(module_id) is None

    def test_connect_modules(self):
        """Test connecting modules together."""
        graph = VFXGraph()

        emitter = VFXEmitterModule()
        spawn_rate = VFXSpawnRateModule()

        e_id = graph.add_module(emitter)
        s_id = graph.add_module(spawn_rate)

        result = graph.connect(e_id, "out", s_id, "in")
        assert result is True

    def test_compile_basic_graph(self):
        """Test compiling graph to ParticleEmitter."""
        graph = VFXGraph("BasicEffect")

        # Add essential modules
        graph.add_module(VFXEmitterModule(max_particles=500))
        graph.add_module(VFXSpawnRateModule(rate=50))
        graph.add_module(VFXGravityModule(gravity=Vec3(0, -10, 0)))

        # Compile
        emitter = graph.compile()

        # Verify emitter was created with correct config
        assert isinstance(emitter, ParticleEmitter)
        assert emitter.config.max_particles == 500

    def test_compile_with_modifiers(self):
        """Test compilation includes update modules."""
        graph = VFXGraph("ModifiedEffect")

        graph.add_module(VFXEmitterModule(max_particles=100))
        graph.add_module(VFXSpawnRateModule(rate=10))
        graph.add_module(VFXGravityModule())
        graph.add_module(VFXSizeOverLifeModule(start_size=1.0, end_size=0.0))
        graph.add_module(VFXColorOverLifeModule())

        emitter = graph.compile()

        # Verify modules were added
        assert len(emitter._update_modules) >= 2  # At least gravity + size

    def test_compile_caching(self):
        """Test that compile() caches the result."""
        graph = VFXGraph()
        graph.add_module(VFXEmitterModule())

        emitter1 = graph.compile()
        emitter2 = graph.compile()

        assert emitter1 is emitter2  # Same cached instance

        # Modify graph should invalidate cache
        graph.add_module(VFXGravityModule())
        emitter3 = graph.compile()

        assert emitter3 is not emitter1

    def test_expose_parameter(self):
        """Test exposing module parameters at graph level."""
        graph = VFXGraph()

        spawn_rate = VFXSpawnRateModule(rate=100)
        module_id = graph.add_module(spawn_rate)

        graph.expose_parameter(module_id, "rate", "spawn_rate")
        graph.set_parameter("spawn_rate", 200)

        # Parameter should be updated
        assert spawn_rate.get_parameter("rate").value == 200


class TestVFXGraphSerialization:
    """Test VFXGraph serialization and deserialization."""

    def test_to_dict(self):
        """Test graph serialization to dictionary."""
        graph = VFXGraph("TestGraph")
        graph.add_module(VFXEmitterModule(max_particles=1000))
        graph.add_module(VFXSpawnRateModule(rate=100))

        data = graph.to_dict()

        assert data["name"] == "TestGraph"
        assert len(data["modules"]) == 2

    def test_from_dict(self):
        """Test graph deserialization from dictionary."""
        data = {
            "name": "LoadedGraph",
            "modules": [
                {
                    "id": "test_emitter",
                    "name": "Emitter",
                    "type": "EMITTER",
                    "enabled": True,
                    "parameters": {"max_particles": 2000},
                },
                {
                    "id": "test_rate",
                    "name": "SpawnRate",
                    "type": "SPAWN_RATE",
                    "enabled": True,
                    "parameters": {"rate": 150},
                },
            ],
            "connections": [],
            "exposed_parameters": [],
        }

        graph = VFXGraph.from_dict(data)

        assert graph.name == "LoadedGraph"
        assert graph.get_module("test_emitter") is not None
        assert graph.get_module("test_rate") is not None

    def test_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        original = VFXGraph("RoundtripTest")
        original.add_module(VFXEmitterModule(max_particles=3000))
        original.add_module(VFXSpawnRateModule(rate=250))
        original.add_module(VFXGravityModule(Vec3(0, -15, 0)))

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = VFXGraph.from_dict(data)

        assert restored.name == original.name
        # Compile both and verify they produce similar emitters
        emitter_original = original.compile()
        emitter_restored = restored.compile()

        assert emitter_original.config.max_particles == emitter_restored.config.max_particles


class TestVFXEventModule:
    """Test VFXEventModule behavior."""

    def test_event_module_creation(self):
        """Test event module creation."""
        module = VFXEventModule(
            event_trigger=VFXEventTrigger.SPAWN,
            event_name="on_spawn",
        )

        assert module.event.name == "on_spawn"
        assert module.event.trigger == VFXEventTrigger.SPAWN

    def test_event_module_children(self):
        """Test adding child modules to event."""
        event_module = VFXEventModule(
            event_trigger=VFXEventTrigger.DEATH,
            event_name="on_death",
        )

        # Add a child module that should execute on event
        size_module = VFXSizeOverLifeModule()
        event_module.add_child(size_module)

        # Evaluate should process children
        context = VFXContext()
        event_module.evaluate(context)

        # Event should have been fired (handler registration needed to verify)
        # For now, just verify no errors occurred


class TestVFXGraphWithEvents:
    """Test VFXGraph with event modules."""

    def test_compile_with_spawn_event(self):
        """Test compiling graph with spawn event handler."""
        graph = VFXGraph("EventGraph")

        graph.add_module(VFXEmitterModule(max_particles=100))
        graph.add_module(VFXSpawnRateModule(rate=10))

        spawn_event = VFXEventModule(
            event_trigger=VFXEventTrigger.SPAWN,
            event_name="on_spawn",
        )
        graph.add_module(spawn_event)

        emitter = graph.compile()

        # Verify event handler was attached
        assert emitter._on_particle_spawn is not None

    def test_compile_with_death_event(self):
        """Test compiling graph with death event handler."""
        graph = VFXGraph("DeathEventGraph")

        graph.add_module(VFXEmitterModule(max_particles=100))

        death_event = VFXEventModule(
            event_trigger=VFXEventTrigger.DEATH,
            event_name="on_death",
        )
        graph.add_module(death_event)

        emitter = graph.compile()

        # Verify event handler was attached
        assert emitter._on_particle_death is not None
