"""ECS system for crowd animation and rendering.

Integrates crowd simulation, LOD, and GPU rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.core.math import Vec3, Quat, Transform
from engine.core.ecs import Entity, World

from ..crowds.animation_texture import AnimationTextureAtlas
from ..crowds.crowd_renderer import CrowdRenderer, CrowdInstance
from ..crowds.crowd_lod import CrowdLOD, LODLevel
from ..crowds.crowd_behavior import CrowdSimulator, CrowdAgent, AgentState
from engine.animation.config import CROWD_SYSTEM_CONFIG


@dataclass
class CrowdComponent:
    """Component for entities managing crowds.

    Attributes:
        simulator: Crowd behavior simulator
        renderer: GPU crowd renderer
        lod: LOD manager
        enabled: Whether crowd is enabled
        update_rate: Simulation update rate (per second)
        max_visible: Maximum visible instances
    """
    simulator: CrowdSimulator = field(default_factory=CrowdSimulator)
    renderer: CrowdRenderer = field(default_factory=CrowdRenderer)
    lod: CrowdLOD = field(default_factory=CrowdLOD)
    enabled: bool = True
    update_rate: float = CROWD_SYSTEM_CONFIG.DEFAULT_UPDATE_RATE  # Updates per second
    max_visible: int = CROWD_SYSTEM_CONFIG.DEFAULT_MAX_VISIBLE

    # Camera reference for LOD
    camera_position: Vec3 = field(default_factory=Vec3.zero)
    camera_forward: Vec3 = field(default_factory=Vec3.forward)

    # State
    _accumulated_time: float = 0.0
    _synced_instances: dict[int, int] = field(default_factory=dict)  # agent_id -> instance_id

    def add_agent(
        self,
        position: Vec3,
        mesh_id: int = 0,
        material_id: int = 0,
        initial_state: AgentState = AgentState.IDLE
    ) -> int:
        """Add a new crowd agent.

        Args:
            position: World position
            mesh_id: Mesh identifier for rendering
            material_id: Material identifier
            initial_state: Initial behavior state

        Returns:
            Agent ID
        """
        # Create agent
        agent = CrowdAgent(
            position=position,
            current_state=initial_state,
        )
        agent_id = self.simulator.add_agent(agent)

        # Create render instance
        instance = CrowdInstance(
            position=position,
            rotation=agent.get_rotation(),
        )
        instance_id = self.renderer.add_instance(instance, mesh_id, material_id)

        self._synced_instances[agent_id] = instance_id
        return agent_id

    def remove_agent(self, agent_id: int) -> bool:
        """Remove agent and its render instance."""
        if agent_id not in self._synced_instances:
            return False

        instance_id = self._synced_instances.pop(agent_id)
        self.simulator.remove_agent(agent_id)
        self.renderer.remove_instance(instance_id)
        return True

    def get_agent(self, agent_id: int) -> CrowdAgent | None:
        """Get agent by ID."""
        return self.simulator.get_agent(agent_id)

    def set_agent_target(self, agent_id: int, target: Vec3) -> bool:
        """Set movement target for agent."""
        agent = self.simulator.get_agent(agent_id)
        if agent:
            agent.target_position = target
            self.simulator.transition_agent(agent, AgentState.WALKING)
            return True
        return False

    def get_agent_count(self) -> int:
        """Get total agent count."""
        return self.simulator.agent_count

    def get_visible_count(self) -> int:
        """Get visible instance count."""
        return sum(1 for batch in self.renderer.get_batches() for inst in batch.instances if inst.visible)


class CrowdSystem:
    """ECS system for crowd simulation and rendering.

    Manages crowd behavior simulation, LOD updates, and render data preparation.
    """

    def __init__(self):
        self._lod_distances: list[float] = list(CROWD_SYSTEM_CONFIG.DEFAULT_LOD_DISTANCES)
        self._cull_distance: float = CROWD_SYSTEM_CONFIG.DEFAULT_CULL_DISTANCE

    def set_lod_distances(self, distances: list[float]) -> None:
        """Set LOD distance thresholds."""
        self._lod_distances = sorted(distances)

    def set_cull_distance(self, distance: float) -> None:
        """Set maximum render distance."""
        self._cull_distance = distance

    def update(
        self,
        world: World,
        dt: float,
        entity_components: list[tuple[Entity, CrowdComponent]]
    ) -> None:
        """Update all crowd components.

        Args:
            world: ECS world
            dt: Delta time
            entity_components: List of (entity, component) tuples
        """
        for entity, component in entity_components:
            if not component.enabled:
                continue

            self._update_component(component, dt)

    def _update_component(self, component: CrowdComponent, dt: float) -> None:
        """Update single crowd component."""
        # Accumulate time for fixed-rate updates
        component._accumulated_time += dt
        update_interval = 1.0 / component.update_rate

        # Run simulation updates
        while component._accumulated_time >= update_interval:
            component._accumulated_time -= update_interval
            component.simulator.update(update_interval)

        # Sync agent states to render instances
        self._sync_agents_to_instances(component)

        # Update LOD levels
        self._update_lod(component)

        # Cull distant instances
        component.renderer.cull_instances(component.camera_position, self._cull_distance)

        # Update LOD levels for visible instances
        component.renderer.update_lod_levels(component.camera_position, self._lod_distances)

        # Update renderer (advance animation times)
        component.renderer.update(dt)

        # Advance LOD frame counter
        component.lod.advance_frame()

    def _sync_agents_to_instances(self, component: CrowdComponent) -> None:
        """Synchronize agent states to render instances."""
        for agent_id, instance_id in component._synced_instances.items():
            agent = component.simulator.get_agent(agent_id)
            if not agent:
                continue

            # Find instance in renderer
            for batch in component.renderer.get_batches():
                for instance in batch.instances:
                    if instance.instance_id == instance_id:
                        # Update instance from agent
                        instance.position = agent.position
                        instance.rotation = agent.get_rotation()

                        # Update animation based on agent state
                        anim_blend = agent.animation_blend
                        instance.animation_index = anim_blend.get_primary_animation()

                        break

    def _update_lod(self, component: CrowdComponent) -> None:
        """Update LOD levels for all instances."""
        for batch in component.renderer.get_batches():
            for instance in batch.instances:
                distance = instance.distance_to(component.camera_position)
                new_lod = component.lod.get_lod_for_distance(distance, instance.lod_level)
                instance.lod_level = new_lod

    def prepare_render_data(
        self,
        entity_components: list[tuple[Entity, CrowdComponent]]
    ) -> list[tuple[Entity, Any]]:
        """Prepare render data for all crowds.

        Returns:
            List of (entity, render_data) tuples
        """
        result = []

        for entity, component in entity_components:
            if not component.enabled:
                continue

            render_data = component.renderer.prepare_render_data()
            result.append((entity, render_data))

        return result

    def trigger_flee_event(
        self,
        component: CrowdComponent,
        threat_position: Vec3,
        radius: float
    ) -> int:
        """Trigger flee event for agents near threat.

        Args:
            component: Crowd component
            threat_position: Position of threat
            radius: Affect radius

        Returns:
            Number of agents affected
        """
        return component.simulator.trigger_flee(threat_position, radius)

    def spawn_crowd_formation(
        self,
        component: CrowdComponent,
        center: Vec3,
        count: int,
        radius: float,
        mesh_id: int = 0,
        material_id: int = 0,
        formation: str = "circle"
    ) -> list[int]:
        """Spawn agents in a formation.

        Args:
            component: Crowd component
            center: Formation center
            count: Number of agents
            radius: Formation radius
            mesh_id: Mesh ID for rendering
            material_id: Material ID
            formation: Formation type (circle, grid, random)

        Returns:
            List of agent IDs
        """
        import math
        import random

        agent_ids = []

        if formation == "circle":
            for i in range(count):
                angle = (i / count) * math.pi * 2
                pos = Vec3(
                    center.x + math.cos(angle) * radius,
                    center.y,
                    center.z + math.sin(angle) * radius,
                )
                agent_id = component.add_agent(pos, mesh_id, material_id)
                agent_ids.append(agent_id)

        elif formation == "grid":
            side = int(math.ceil(math.sqrt(count)))
            spacing = radius * 2 / max(1, side - 1)
            idx = 0
            for row in range(side):
                for col in range(side):
                    if idx >= count:
                        break
                    pos = Vec3(
                        center.x - radius + col * spacing,
                        center.y,
                        center.z - radius + row * spacing,
                    )
                    agent_id = component.add_agent(pos, mesh_id, material_id)
                    agent_ids.append(agent_id)
                    idx += 1

        elif formation == "random":
            for _ in range(count):
                angle = random.random() * math.pi * 2
                dist = random.random() * radius
                pos = Vec3(
                    center.x + math.cos(angle) * dist,
                    center.y,
                    center.z + math.sin(angle) * dist,
                )
                agent_id = component.add_agent(pos, mesh_id, material_id)
                agent_ids.append(agent_id)

        return agent_ids

    def get_stats(
        self,
        entity_components: list[tuple[Entity, CrowdComponent]]
    ) -> dict[str, Any]:
        """Get aggregate statistics for all crowds.

        Returns:
            Dictionary of statistics
        """
        total_agents = 0
        total_visible = 0
        total_batches = 0

        for _, component in entity_components:
            total_agents += component.get_agent_count()
            total_visible += component.get_visible_count()
            total_batches += component.renderer.batch_count

        return {
            "total_agents": total_agents,
            "total_visible": total_visible,
            "total_batches": total_batches,
            "cull_distance": self._cull_distance,
            "lod_distances": self._lod_distances,
        }
