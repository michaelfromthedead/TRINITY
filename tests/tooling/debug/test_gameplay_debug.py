"""Tests for gameplay debug - AI paths, triggers, zones."""

import pytest
from engine.tooling.debug.gameplay_debug import (
    GameplayDebugger,
    AIVisualization,
    NavMeshDisplay,
    TriggerVolumeVisualizer,
    AIAgent,
    AIState,
    NavMeshPolygon,
    NavMeshConnection,
    TriggerVolume,
    TriggerType,
    Vector3,
)


class TestVector3Gameplay:
    """Tests for Vector3 in gameplay context."""

    def test_to_tuple(self):
        v = Vector3(1, 2, 3)
        assert v.to_tuple() == (1, 2, 3)


class TestAIAgent:
    """Tests for AIAgent class."""

    def test_agent_creation(self):
        agent = AIAgent(
            agent_id="agent_001",
            position=Vector3(0, 0, 0),
            state=AIState.IDLE,
        )
        assert agent.agent_id == "agent_001"
        assert agent.state == AIState.IDLE

    def test_agent_with_path(self):
        path = [Vector3(0, 0, 0), Vector3(5, 0, 0), Vector3(10, 0, 0)]
        agent = AIAgent(
            agent_id="agent",
            position=Vector3(0, 0, 0),
            path=path,
        )
        assert len(agent.path) == 3

    def test_agent_health(self):
        agent = AIAgent(
            agent_id="agent",
            position=Vector3(0, 0, 0),
            health=50.0,
            max_health=100.0,
        )
        assert agent.health == 50.0
        assert agent.max_health == 100.0


class TestAIVisualization:
    """Tests for AIVisualization class."""

    def test_ai_viz_creation(self):
        viz = AIVisualization()
        assert viz.is_enabled is True
        assert viz.agent_count == 0

    def test_ai_viz_enable_disable(self):
        viz = AIVisualization()
        viz.disable()
        assert not viz.is_enabled
        viz.enable()
        assert viz.is_enabled

    def test_register_agent(self):
        viz = AIVisualization()
        agent = AIAgent(agent_id="a1", position=Vector3(0, 0, 0))
        viz.register_agent(agent)
        assert viz.agent_count == 1
        assert viz.get_agent("a1") is agent

    def test_unregister_agent(self):
        viz = AIVisualization()
        agent = AIAgent(agent_id="a1", position=Vector3(0, 0, 0))
        viz.register_agent(agent)
        removed = viz.unregister_agent("a1")
        assert removed is agent
        assert viz.agent_count == 0

    def test_update_agent(self):
        viz = AIVisualization()
        agent = AIAgent(agent_id="a1", position=Vector3(0, 0, 0))
        viz.register_agent(agent)

        result = viz.update_agent(
            "a1",
            position=Vector3(10, 0, 0),
            state=AIState.CHASE,
            health=50.0,
        )
        assert result is True
        assert agent.position.x == 10
        assert agent.state == AIState.CHASE
        assert agent.health == 50.0

    def test_update_agent_not_found(self):
        viz = AIVisualization()
        result = viz.update_agent("nonexistent", position=Vector3(0, 0, 0))
        assert result is False

    def test_get_state_color(self):
        viz = AIVisualization()
        color = viz.get_state_color(AIState.ATTACK)
        assert color == (1.0, 0.0, 0.0, 1.0)  # Red for attack

    def test_set_state_color(self):
        viz = AIVisualization()
        new_color = (0.5, 0.5, 0.5, 1.0)
        viz.set_state_color(AIState.IDLE, new_color)
        assert viz.get_state_color(AIState.IDLE) == new_color

    def test_show_options(self):
        viz = AIVisualization()
        viz.set_show_paths(False)
        viz.set_show_perception(False)
        viz.set_show_targets(False)
        viz.set_show_states(False)
        viz.set_show_health(False)
        # No assertions needed, just verify no errors

    def test_generate_draw_commands(self):
        viz = AIVisualization()
        agent = AIAgent(
            agent_id="a1",
            position=Vector3(0, 0, 0),
            state=AIState.PATROL,
            path=[Vector3(0, 0, 0), Vector3(5, 0, 0)],
            target=Vector3(10, 0, 0),
        )
        viz.register_agent(agent)

        commands = viz.generate_draw_commands()
        assert len(commands) > 0

    def test_generate_draw_commands_disabled(self):
        viz = AIVisualization()
        agent = AIAgent(agent_id="a1", position=Vector3(0, 0, 0))
        viz.register_agent(agent)
        viz.disable()

        commands = viz.generate_draw_commands()
        assert len(commands) == 0

    def test_get_agents_by_state(self):
        viz = AIVisualization()
        viz.register_agent(AIAgent("a1", Vector3(0, 0, 0), state=AIState.IDLE))
        viz.register_agent(AIAgent("a2", Vector3(0, 0, 0), state=AIState.CHASE))
        viz.register_agent(AIAgent("a3", Vector3(0, 0, 0), state=AIState.IDLE))

        idle_agents = viz.get_agents_by_state(AIState.IDLE)
        assert len(idle_agents) == 2

    def test_clear_all_agents(self):
        viz = AIVisualization()
        viz.register_agent(AIAgent("a1", Vector3(0, 0, 0)))
        viz.register_agent(AIAgent("a2", Vector3(0, 0, 0)))
        viz.clear_all_agents()
        assert viz.agent_count == 0


class TestNavMeshPolygon:
    """Tests for NavMeshPolygon class."""

    def test_polygon_creation(self):
        vertices = [Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0.5, 0, 1)]
        polygon = NavMeshPolygon(
            polygon_id=1,
            vertices=vertices,
            area_type="walkable",
        )
        assert polygon.polygon_id == 1
        assert len(polygon.vertices) == 3


class TestNavMeshDisplay:
    """Tests for NavMeshDisplay class."""

    def test_nav_mesh_creation(self):
        display = NavMeshDisplay()
        assert display.is_enabled is True
        assert display.polygon_count == 0

    def test_add_polygon(self):
        display = NavMeshDisplay()
        vertices = [Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0.5, 0, 1)]
        polygon = NavMeshPolygon(polygon_id=1, vertices=vertices)
        display.add_polygon(polygon)
        assert display.polygon_count == 1
        assert display.get_polygon(1) is polygon

    def test_remove_polygon(self):
        display = NavMeshDisplay()
        polygon = NavMeshPolygon(polygon_id=1, vertices=[])
        display.add_polygon(polygon)
        removed = display.remove_polygon(1)
        assert removed is polygon
        assert display.polygon_count == 0

    def test_add_connection(self):
        display = NavMeshDisplay()
        conn = NavMeshConnection(
            from_polygon=1,
            to_polygon=2,
            edge_start=Vector3(0, 0, 0),
            edge_end=Vector3(1, 0, 0),
        )
        display.add_connection(conn)
        assert display.connection_count == 1

    def test_select_polygon(self):
        display = NavMeshDisplay()
        display.select_polygon(5)
        assert display._selected_polygon == 5
        display.select_polygon(None)
        assert display._selected_polygon is None

    def test_show_options(self):
        display = NavMeshDisplay()
        display.set_show_polygons(False)
        display.set_show_connections(False)
        display.set_show_costs(True)

    def test_generate_draw_commands(self):
        display = NavMeshDisplay()
        vertices = [Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0.5, 0, 1)]
        polygon = NavMeshPolygon(polygon_id=1, vertices=vertices)
        display.add_polygon(polygon)

        commands = display.generate_draw_commands()
        assert len(commands) > 0

    def test_generate_draw_commands_disabled(self):
        display = NavMeshDisplay()
        polygon = NavMeshPolygon(polygon_id=1, vertices=[Vector3(0, 0, 0)])
        display.add_polygon(polygon)
        display.disable()

        commands = display.generate_draw_commands()
        assert len(commands) == 0

    def test_clear(self):
        display = NavMeshDisplay()
        display.add_polygon(NavMeshPolygon(polygon_id=1, vertices=[]))
        display.add_connection(NavMeshConnection(from_polygon=1, to_polygon=2))
        display.clear()
        assert display.polygon_count == 0
        assert display.connection_count == 0


class TestTriggerVolume:
    """Tests for TriggerVolume class."""

    def test_volume_creation(self):
        volume = TriggerVolume(
            volume_id="trigger_1",
            trigger_type=TriggerType.BOX,
            position=Vector3(0, 0, 0),
            extents=Vector3(5, 5, 5),
        )
        assert volume.volume_id == "trigger_1"
        assert volume.trigger_type == TriggerType.BOX

    def test_volume_with_events(self):
        volume = TriggerVolume(
            volume_id="trigger",
            trigger_type=TriggerType.SPHERE,
            position=Vector3(0, 0, 0),
            on_enter="OnEnterTrigger",
            on_exit="OnExitTrigger",
        )
        assert volume.on_enter == "OnEnterTrigger"
        assert volume.on_exit == "OnExitTrigger"


class TestTriggerVolumeVisualizer:
    """Tests for TriggerVolumeVisualizer class."""

    def test_visualizer_creation(self):
        viz = TriggerVolumeVisualizer()
        assert viz.is_enabled is True
        assert viz.volume_count == 0

    def test_add_volume(self):
        viz = TriggerVolumeVisualizer()
        volume = TriggerVolume(
            volume_id="t1",
            trigger_type=TriggerType.BOX,
            position=Vector3(0, 0, 0),
        )
        viz.add_volume(volume)
        assert viz.volume_count == 1
        assert viz.get_volume("t1") is volume

    def test_remove_volume(self):
        viz = TriggerVolumeVisualizer()
        volume = TriggerVolume("t1", TriggerType.BOX, Vector3(0, 0, 0))
        viz.add_volume(volume)
        removed = viz.remove_volume("t1")
        assert removed is volume
        assert viz.volume_count == 0

    def test_set_triggered(self):
        viz = TriggerVolumeVisualizer()
        volume = TriggerVolume("t1", TriggerType.BOX, Vector3(0, 0, 0))
        viz.add_volume(volume)

        result = viz.set_triggered("t1", True)
        assert result is True
        assert volume.triggered is True
        assert volume.trigger_count == 1

    def test_set_enabled(self):
        viz = TriggerVolumeVisualizer()
        volume = TriggerVolume("t1", TriggerType.BOX, Vector3(0, 0, 0))
        viz.add_volume(volume)

        viz.set_enabled("t1", False)
        assert volume.enabled is False

    def test_show_options(self):
        viz = TriggerVolumeVisualizer()
        viz.set_show_bounds(False)
        viz.set_show_names(False)
        viz.set_show_events(False)

    def test_generate_draw_commands_box(self):
        viz = TriggerVolumeVisualizer()
        volume = TriggerVolume(
            volume_id="t1",
            trigger_type=TriggerType.BOX,
            position=Vector3(0, 0, 0),
            extents=Vector3(1, 1, 1),
        )
        viz.add_volume(volume)

        commands = viz.generate_draw_commands()
        assert len(commands) > 0
        assert any(cmd.get("type") == "box" for cmd in commands)

    def test_generate_draw_commands_sphere(self):
        viz = TriggerVolumeVisualizer()
        volume = TriggerVolume(
            volume_id="t1",
            trigger_type=TriggerType.SPHERE,
            position=Vector3(0, 0, 0),
            radius=5.0,
        )
        viz.add_volume(volume)

        commands = viz.generate_draw_commands()
        assert any(cmd.get("type") == "sphere" for cmd in commands)

    def test_generate_draw_commands_disabled(self):
        viz = TriggerVolumeVisualizer()
        volume = TriggerVolume("t1", TriggerType.BOX, Vector3(0, 0, 0))
        viz.add_volume(volume)
        viz.disable()

        commands = viz.generate_draw_commands()
        assert len(commands) == 0

    def test_get_triggered_volumes(self):
        viz = TriggerVolumeVisualizer()
        v1 = TriggerVolume("t1", TriggerType.BOX, Vector3(0, 0, 0))
        v2 = TriggerVolume("t2", TriggerType.BOX, Vector3(0, 0, 0))
        viz.add_volume(v1)
        viz.add_volume(v2)
        viz.set_triggered("t1", True)

        triggered = viz.get_triggered_volumes()
        assert len(triggered) == 1
        assert triggered[0].volume_id == "t1"

    def test_get_volumes_by_tag(self):
        viz = TriggerVolumeVisualizer()
        v1 = TriggerVolume("t1", TriggerType.BOX, Vector3(0, 0, 0), tags=["spawn"])
        v2 = TriggerVolume("t2", TriggerType.BOX, Vector3(0, 0, 0), tags=["danger"])
        v3 = TriggerVolume("t3", TriggerType.BOX, Vector3(0, 0, 0), tags=["spawn"])
        viz.add_volume(v1)
        viz.add_volume(v2)
        viz.add_volume(v3)

        spawn_volumes = viz.get_volumes_by_tag("spawn")
        assert len(spawn_volumes) == 2

    def test_clear_all_volumes(self):
        viz = TriggerVolumeVisualizer()
        viz.add_volume(TriggerVolume("t1", TriggerType.BOX, Vector3(0, 0, 0)))
        viz.add_volume(TriggerVolume("t2", TriggerType.BOX, Vector3(0, 0, 0)))
        viz.clear_all_volumes()
        assert viz.volume_count == 0


class TestGameplayDebugger:
    """Tests for GameplayDebugger singleton."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        GameplayDebugger.reset_instance()
        yield
        GameplayDebugger.reset_instance()

    def test_singleton(self):
        d1 = GameplayDebugger.get_instance()
        d2 = GameplayDebugger.get_instance()
        assert d1 is d2

    def test_enable_disable(self):
        debugger = GameplayDebugger.get_instance()
        debugger.enable()
        assert debugger.is_enabled
        debugger.disable()
        assert not debugger.is_enabled

    def test_subsystems_accessible(self):
        debugger = GameplayDebugger.get_instance()
        assert isinstance(debugger.ai_visualization, AIVisualization)
        assert isinstance(debugger.nav_mesh_display, NavMeshDisplay)
        assert isinstance(debugger.trigger_visualizer, TriggerVolumeVisualizer)

    def test_generate_all_draw_commands(self):
        debugger = GameplayDebugger.get_instance()
        debugger.ai_visualization.register_agent(
            AIAgent("a1", Vector3(0, 0, 0))
        )
        debugger.nav_mesh_display.add_polygon(
            NavMeshPolygon(1, [Vector3(0, 0, 0)])
        )
        debugger.trigger_visualizer.add_volume(
            TriggerVolume("t1", TriggerType.BOX, Vector3(0, 0, 0))
        )

        commands = debugger.generate_all_draw_commands()
        assert len(commands) > 0

    def test_generate_all_disabled(self):
        debugger = GameplayDebugger.get_instance()
        debugger.ai_visualization.register_agent(
            AIAgent("a1", Vector3(0, 0, 0))
        )
        debugger.disable()

        commands = debugger.generate_all_draw_commands()
        assert len(commands) == 0

    def test_clear_all(self):
        debugger = GameplayDebugger.get_instance()
        debugger.ai_visualization.register_agent(AIAgent("a1", Vector3(0, 0, 0)))
        debugger.nav_mesh_display.add_polygon(NavMeshPolygon(1, []))
        debugger.trigger_visualizer.add_volume(
            TriggerVolume("t1", TriggerType.BOX, Vector3(0, 0, 0))
        )

        debugger.clear_all()
        assert debugger.ai_visualization.agent_count == 0
        assert debugger.nav_mesh_display.polygon_count == 0
        assert debugger.trigger_visualizer.volume_count == 0
