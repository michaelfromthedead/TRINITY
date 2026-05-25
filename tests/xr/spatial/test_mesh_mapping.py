"""
Tests for mesh mapping module.

Tests the spatial mesh system including:
    - Mesh block creation and updates
    - LOD levels and update modes
    - Bounds computation
    - Raycast against mesh
    - SpatialMeshManager operations
"""

import pytest

from engine.core.math.vec import Vec3
from engine.xr.spatial.mesh_mapping import (
    MeshBlock,
    MeshBounds,
    MeshClassification,
    MeshLODLevel,
    MeshManagerConfig,
    MeshTriangle,
    MeshUpdateMode,
    MeshVertex,
    SpatialMesh,
    SpatialMeshManager,
)


class TestMeshBounds:
    """Tests for MeshBounds class."""

    def test_default_bounds(self):
        bounds = MeshBounds()
        assert bounds.min_point == Vec3.zero()
        assert bounds.max_point == Vec3.zero()

    def test_center(self):
        bounds = MeshBounds(
            min_point=Vec3(0, 0, 0),
            max_point=Vec3(10, 10, 10),
        )
        center = bounds.center
        assert center == Vec3(5, 5, 5)

    def test_size(self):
        bounds = MeshBounds(
            min_point=Vec3(-1, -2, -3),
            max_point=Vec3(1, 2, 3),
        )
        size = bounds.size
        assert size == Vec3(2, 4, 6)

    def test_extents(self):
        bounds = MeshBounds(
            min_point=Vec3(-1, -2, -3),
            max_point=Vec3(1, 2, 3),
        )
        extents = bounds.extents
        assert extents == Vec3(1, 2, 3)

    def test_contains_point_inside(self):
        bounds = MeshBounds(
            min_point=Vec3(0, 0, 0),
            max_point=Vec3(10, 10, 10),
        )
        assert bounds.contains_point(Vec3(5, 5, 5)) is True

    def test_contains_point_outside(self):
        bounds = MeshBounds(
            min_point=Vec3(0, 0, 0),
            max_point=Vec3(10, 10, 10),
        )
        assert bounds.contains_point(Vec3(15, 5, 5)) is False

    def test_contains_point_on_boundary(self):
        bounds = MeshBounds(
            min_point=Vec3(0, 0, 0),
            max_point=Vec3(10, 10, 10),
        )
        assert bounds.contains_point(Vec3(10, 5, 5)) is True

    def test_intersects_overlapping(self):
        bounds1 = MeshBounds(
            min_point=Vec3(0, 0, 0),
            max_point=Vec3(10, 10, 10),
        )
        bounds2 = MeshBounds(
            min_point=Vec3(5, 5, 5),
            max_point=Vec3(15, 15, 15),
        )
        assert bounds1.intersects(bounds2) is True

    def test_intersects_non_overlapping(self):
        bounds1 = MeshBounds(
            min_point=Vec3(0, 0, 0),
            max_point=Vec3(10, 10, 10),
        )
        bounds2 = MeshBounds(
            min_point=Vec3(20, 20, 20),
            max_point=Vec3(30, 30, 30),
        )
        assert bounds1.intersects(bounds2) is False

    def test_expand_to_include(self):
        bounds = MeshBounds(
            min_point=Vec3(0, 0, 0),
            max_point=Vec3(10, 10, 10),
        )
        bounds.expand_to_include(Vec3(15, -5, 5))

        assert bounds.min_point.y == -5
        assert bounds.max_point.x == 15


class TestMeshBlock:
    """Tests for MeshBlock class."""

    def test_empty_block(self):
        block = MeshBlock()
        assert block.vertex_count == 0
        assert block.triangle_count == 0
        assert block.is_dirty is True

    def test_block_with_data(self):
        vertices = [
            MeshVertex(position=Vec3(0, 0, 0)),
            MeshVertex(position=Vec3(1, 0, 0)),
            MeshVertex(position=Vec3(0, 0, 1)),
        ]
        triangles = [MeshTriangle(v0=0, v1=1, v2=2)]

        block = MeshBlock(vertices=vertices, triangles=triangles)
        assert block.vertex_count == 3
        assert block.triangle_count == 1

    def test_compute_bounds(self):
        vertices = [
            MeshVertex(position=Vec3(-1, 0, -1)),
            MeshVertex(position=Vec3(1, 0, -1)),
            MeshVertex(position=Vec3(0, 2, 1)),
        ]
        block = MeshBlock(vertices=vertices)
        block.compute_bounds()

        assert block.bounds.min_point.x == -1
        assert block.bounds.max_point.x == 1
        assert block.bounds.max_point.y == 2

    def test_clear(self):
        vertices = [
            MeshVertex(position=Vec3(0, 0, 0)),
            MeshVertex(position=Vec3(1, 0, 0)),
        ]
        block = MeshBlock(vertices=vertices)
        initial_version = block.version

        block.clear()

        assert block.vertex_count == 0
        assert block.triangle_count == 0
        assert block.version == initial_version + 1


class TestSpatialMesh:
    """Tests for SpatialMesh class."""

    def test_default_mesh(self):
        mesh = SpatialMesh()
        assert mesh.lod_level == MeshLODLevel.MEDIUM
        assert mesh.update_mode == MeshUpdateMode.INCREMENTAL
        assert mesh.is_valid is False

    def test_custom_settings(self):
        mesh = SpatialMesh(
            lod_level=MeshLODLevel.HIGH,
            update_mode=MeshUpdateMode.FULL,
        )
        assert mesh.lod_level == MeshLODLevel.HIGH
        assert mesh.update_mode == MeshUpdateMode.FULL

    def test_add_block(self):
        mesh = SpatialMesh()
        block = MeshBlock(vertices=[
            MeshVertex(position=Vec3(0, 0, 0)),
            MeshVertex(position=Vec3(1, 0, 0)),
            MeshVertex(position=Vec3(0, 0, 1)),
        ])
        block.compute_bounds()

        mesh.add_block(block)

        assert mesh.block_count == 1
        assert mesh.is_valid is True
        assert mesh.get_block(block.block_id) is block

    def test_update_block(self):
        mesh = SpatialMesh()
        block = MeshBlock()
        mesh.add_block(block)

        new_vertices = [
            MeshVertex(position=Vec3(0, 0, 0)),
            MeshVertex(position=Vec3(2, 0, 0)),
            MeshVertex(position=Vec3(1, 0, 2)),
        ]
        new_triangles = [MeshTriangle(v0=0, v1=1, v2=2)]

        result = mesh.update_block(
            block.block_id,
            vertices=new_vertices,
            triangles=new_triangles,
            timestamp=1.0,
        )

        assert result is True
        assert mesh.vertex_count == 3
        assert mesh.triangle_count == 1

    def test_remove_block(self):
        mesh = SpatialMesh()
        block = MeshBlock()
        mesh.add_block(block)

        result = mesh.remove_block(block.block_id)

        assert result is True
        assert mesh.block_count == 0

    def test_get_all_blocks(self):
        mesh = SpatialMesh()
        for _ in range(5):
            mesh.add_block(MeshBlock())

        blocks = mesh.get_all_blocks()
        assert len(blocks) == 5

    def test_get_dirty_blocks(self):
        mesh = SpatialMesh()
        b1 = MeshBlock()
        b2 = MeshBlock()
        mesh.add_block(b1)
        mesh.add_block(b2)

        # Update one block (clears dirty flag)
        mesh.update_block(b1.block_id, [], [], 1.0)

        dirty = mesh.get_dirty_blocks()
        assert len(dirty) == 1
        assert b2 in dirty

    def test_get_blocks_in_bounds(self):
        mesh = SpatialMesh()

        b1 = MeshBlock(vertices=[MeshVertex(position=Vec3(0, 0, 0))])
        b1.compute_bounds()
        mesh.add_block(b1)

        b2 = MeshBlock(vertices=[MeshVertex(position=Vec3(100, 100, 100))])
        b2.compute_bounds()
        mesh.add_block(b2)

        query_bounds = MeshBounds(
            min_point=Vec3(-5, -5, -5),
            max_point=Vec3(5, 5, 5),
        )

        result = mesh.get_blocks_in_bounds(query_bounds)
        assert len(result) == 1
        assert b1 in result

    def test_get_blocks_near(self):
        mesh = SpatialMesh()

        b1 = MeshBlock(vertices=[MeshVertex(position=Vec3(1, 1, 1))])
        b1.compute_bounds()
        mesh.add_block(b1)

        b2 = MeshBlock(vertices=[MeshVertex(position=Vec3(50, 50, 50))])
        b2.compute_bounds()
        mesh.add_block(b2)

        result = mesh.get_blocks_near(Vec3(0, 0, 0), radius=10.0)
        assert len(result) == 1

    def test_raycast_hit(self):
        mesh = SpatialMesh()

        # Create a simple triangle on the floor
        vertices = [
            MeshVertex(position=Vec3(-1, 0, -1)),
            MeshVertex(position=Vec3(1, 0, -1)),
            MeshVertex(position=Vec3(0, 0, 1)),
        ]
        triangles = [MeshTriangle(v0=0, v1=1, v2=2)]
        block = MeshBlock(vertices=vertices, triangles=triangles)
        block.compute_bounds()
        mesh.add_block(block)

        result = mesh.raycast(
            origin=Vec3(0, 5, 0),
            direction=Vec3(0, -1, 0),
        )

        assert result is not None
        hit_point, normal, distance = result
        assert abs(distance - 5.0) < 0.1

    def test_raycast_miss(self):
        mesh = SpatialMesh()

        vertices = [
            MeshVertex(position=Vec3(-1, 0, -1)),
            MeshVertex(position=Vec3(1, 0, -1)),
            MeshVertex(position=Vec3(0, 0, 1)),
        ]
        triangles = [MeshTriangle(v0=0, v1=1, v2=2)]
        block = MeshBlock(vertices=vertices, triangles=triangles)
        block.compute_bounds()
        mesh.add_block(block)

        # Ray pointing up, away from triangle
        result = mesh.raycast(
            origin=Vec3(0, 5, 0),
            direction=Vec3(0, 1, 0),
        )

        assert result is None

    def test_cleanup_distant_blocks(self):
        mesh = SpatialMesh()

        near_block = MeshBlock(vertices=[MeshVertex(position=Vec3(0, 0, 0))])
        near_block.compute_bounds()
        mesh.add_block(near_block)

        far_block = MeshBlock(vertices=[MeshVertex(position=Vec3(100, 100, 100))])
        far_block.compute_bounds()
        mesh.add_block(far_block)

        removed = mesh.cleanup_distant_blocks(
            reference=Vec3(0, 0, 0),
            max_distance=50.0,
        )

        assert removed == 1
        assert mesh.block_count == 1

    def test_clear(self):
        mesh = SpatialMesh()
        mesh.add_block(MeshBlock())
        mesh.add_block(MeshBlock())

        mesh.clear()

        assert mesh.block_count == 0
        assert mesh.is_valid is False

    def test_lod_change_marks_dirty(self):
        mesh = SpatialMesh()
        block = MeshBlock()
        mesh.add_block(block)

        # Update to clear dirty
        mesh.update_block(block.block_id, [], [], 1.0)
        assert block.is_dirty is False

        # Change LOD
        mesh.lod_level = MeshLODLevel.HIGH

        assert block.is_dirty is True


class TestSpatialMeshManager:
    """Tests for SpatialMeshManager class."""

    def test_default_config(self):
        manager = SpatialMeshManager()
        assert manager.config.lod_level == MeshLODLevel.MEDIUM
        assert manager.config.max_distance == 20.0

    def test_custom_config(self):
        config = MeshManagerConfig(
            lod_level=MeshLODLevel.HIGH,
            max_distance=50.0,
            classification_enabled=True,
        )
        manager = SpatialMeshManager(config)
        assert manager.config.lod_level == MeshLODLevel.HIGH
        assert manager.mesh.classification_enabled is True

    def test_start_stop(self):
        manager = SpatialMeshManager()
        assert manager.is_running is False

        manager.start()
        assert manager.is_running is True

        manager.stop()
        assert manager.is_running is False

    def test_set_lod_level(self):
        manager = SpatialMeshManager()
        manager.set_lod_level(MeshLODLevel.ULTRA)
        assert manager.config.lod_level == MeshLODLevel.ULTRA
        assert manager.mesh.lod_level == MeshLODLevel.ULTRA

    def test_set_max_distance(self):
        manager = SpatialMeshManager()
        manager.set_max_distance(30.0)
        assert manager.config.max_distance == 30.0

    def test_set_max_distance_minimum(self):
        manager = SpatialMeshManager()
        manager.set_max_distance(0.5)
        assert manager.config.max_distance == 1.0  # Clamped to minimum

    def test_get_mesh_for_physics(self):
        manager = SpatialMeshManager()
        manager.config.physics_mesh_enabled = True

        mesh = manager.get_mesh_for_physics()
        assert mesh is manager.mesh

    def test_get_mesh_for_physics_disabled(self):
        manager = SpatialMeshManager()
        manager.config.physics_mesh_enabled = False

        mesh = manager.get_mesh_for_physics()
        assert mesh is None

    def test_get_mesh_for_occlusion(self):
        manager = SpatialMeshManager()
        manager.config.occlusion_mesh_enabled = True

        mesh = manager.get_mesh_for_occlusion()
        assert mesh is manager.mesh

    def test_clear(self):
        manager = SpatialMeshManager()
        manager.mesh.add_block(MeshBlock())

        manager.clear()
        assert manager.mesh.block_count == 0

    def test_raycast(self):
        manager = SpatialMeshManager()

        vertices = [
            MeshVertex(position=Vec3(-1, 0, -1)),
            MeshVertex(position=Vec3(1, 0, -1)),
            MeshVertex(position=Vec3(0, 0, 1)),
        ]
        triangles = [MeshTriangle(v0=0, v1=1, v2=2)]
        block = MeshBlock(vertices=vertices, triangles=triangles)
        block.compute_bounds()
        manager.mesh.add_block(block)

        result = manager.raycast(
            origin=Vec3(0, 5, 0),
            direction=Vec3(0, -1, 0),
        )

        assert result is not None
