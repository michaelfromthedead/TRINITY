"""
Tests for foliage instance management (instances.py).

Tests the HISM pattern including:
- FoliageInstance creation and transforms
- Frustum culling
- FoliageCluster management
- HierarchicalInstancedMesh operations
- FoliageManager coordination
- BatchedDescriptor bulk operations
"""

import math

import pytest

from engine.world.foliage.instances import (
    BatchedDescriptor,
    FoliageCluster,
    FoliageInstance,
    FoliageManager,
    Frustum,
    HierarchicalInstancedMesh,
)
from engine.world.foliage.placement import Bounds, PlacementResult
from engine.world.foliage.types import FoliageCategory, FoliageType


# =============================================================================
# FoliageInstance
# =============================================================================


class TestFoliageInstance:
    def test_default_values(self):
        inst = FoliageInstance()
        assert inst.instance_id == 0
        assert inst.foliage_type_id == ""
        assert inst.position == (0.0, 0.0, 0.0)
        assert inst.rotation == (0.0, 0.0, 0.0)
        assert inst.scale == (1.0, 1.0, 1.0)
        assert inst.visible is True
        assert inst.lod_level == 0

    def test_custom_values(self):
        inst = FoliageInstance(
            instance_id=42,
            foliage_type_id="test_plant",
            position=(10.0, 5.0, 20.0),
            rotation=(0.0, 45.0, 0.0),
            scale=(1.5, 1.5, 1.5),
            visible=False,
            lod_level=2,
        )
        assert inst.instance_id == 42
        assert inst.foliage_type_id == "test_plant"
        assert inst.position == (10.0, 5.0, 20.0)
        assert inst.lod_level == 2

    def test_distance_to_origin(self):
        inst = FoliageInstance(position=(3.0, 0.0, 4.0))
        dist = inst.distance_to((0.0, 0.0, 0.0))
        assert abs(dist - 5.0) < 0.001

    def test_distance_to_point(self):
        inst = FoliageInstance(position=(10.0, 0.0, 0.0))
        dist = inst.distance_to((0.0, 0.0, 0.0))
        assert abs(dist - 10.0) < 0.001

    def test_distance_squared_to(self):
        inst = FoliageInstance(position=(3.0, 0.0, 4.0))
        dist_sq = inst.distance_squared_to((0.0, 0.0, 0.0))
        assert abs(dist_sq - 25.0) < 0.001

    def test_from_placement(self):
        placement = PlacementResult(
            position=(10.0, 5.0, 20.0),
            rotation=(0.0, 90.0, 0.0),
            scale=(2.0, 2.0, 2.0),
            foliage_type_id="test_plant",
        )
        inst = FoliageInstance.from_placement(42, placement)
        assert inst.instance_id == 42
        assert inst.position == (10.0, 5.0, 20.0)
        assert inst.rotation == (0.0, 90.0, 0.0)
        assert inst.scale == (2.0, 2.0, 2.0)
        assert inst.foliage_type_id == "test_plant"


# =============================================================================
# Frustum
# =============================================================================


class TestFrustum:
    def test_empty_frustum(self):
        frustum = Frustum()
        assert len(frustum.planes) == 0

    def test_contains_point_no_planes(self):
        frustum = Frustum()
        # No planes = everything is inside
        assert frustum.contains_point((0.0, 0.0, 0.0)) is True

    def test_contains_point_simple_plane(self):
        # Plane equation: n.p + d >= 0 means inside
        # Plane with normal (1, 0, 0) and d=0: x >= 0 is inside
        frustum = Frustum(planes=[(1.0, 0.0, 0.0, 0.0)])
        # Point at positive X should be inside
        assert frustum.contains_point((1.0, 0.0, 0.0)) is True
        # Point at negative X should be outside
        assert frustum.contains_point((-1.0, 0.0, 0.0)) is False

    def test_contains_sphere_inside(self):
        # Plane with normal (1, 0, 0) and d=-10: x >= 10 is inside
        # Actually we want x >= -10 which is x + 10 >= 0
        frustum = Frustum(planes=[(1.0, 0.0, 0.0, 10.0)])
        # Sphere at origin with radius 5 should be inside (0 + 10 = 10 >= 0)
        assert frustum.contains_sphere((0.0, 0.0, 0.0), 5.0) is True

    def test_contains_sphere_intersecting(self):
        # Plane: x >= 0
        frustum = Frustum(planes=[(1.0, 0.0, 0.0, 0.0)])
        # Sphere at x=-3 with radius 5 intersects plane (center dist = -3, but -3 + 5 = 2 > 0)
        assert frustum.contains_sphere((-3.0, 0.0, 0.0), 5.0) is True

    def test_contains_sphere_outside(self):
        # Plane: x >= 0
        frustum = Frustum(planes=[(1.0, 0.0, 0.0, 0.0)])
        # Sphere at x=-10 with radius 2 is outside (-10 + 2 = -8 < 0)
        assert frustum.contains_sphere((-10.0, 0.0, 0.0), 2.0) is False

    def test_contains_bounds_inside(self):
        # Plane: x >= -100 (normal (1,0,0), d=100)
        frustum = Frustum(planes=[(1.0, 0.0, 0.0, 100.0)])
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        assert frustum.contains_bounds(bounds, 0.0, 10.0) is True

    def test_contains_bounds_outside(self):
        # Plane: x >= 100 (normal (1,0,0), d=-100)
        frustum = Frustum(planes=[(1.0, 0.0, 0.0, -100.0)])
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        assert frustum.contains_bounds(bounds, 0.0, 10.0) is False

    def test_from_view_projection_identity(self):
        # Simple identity-like matrix
        matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        frustum = Frustum.from_view_projection(matrix)
        assert len(frustum.planes) == 6


# =============================================================================
# FoliageCluster
# =============================================================================


class TestFoliageCluster:
    def test_creation(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        cluster = FoliageCluster(bounds, "test_plant")
        assert cluster.foliage_type_id == "test_plant"
        assert cluster.instance_count == 0

    def test_add_instance(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        cluster = FoliageCluster(bounds, "test_plant")
        inst = FoliageInstance(instance_id=0, position=(25.0, 0.0, 25.0))
        cluster.add_instance(inst)
        assert cluster.instance_count == 1

    def test_remove_instance(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        cluster = FoliageCluster(bounds, "test_plant")
        inst = FoliageInstance(instance_id=42, position=(25.0, 0.0, 25.0))
        cluster.add_instance(inst)
        assert cluster.remove_instance(42) is True
        assert cluster.instance_count == 0

    def test_remove_instance_not_found(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        cluster = FoliageCluster(bounds, "test_plant")
        assert cluster.remove_instance(999) is False

    def test_get_instances(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        cluster = FoliageCluster(bounds, "test_plant")
        cluster.add_instance(FoliageInstance(instance_id=0))
        cluster.add_instance(FoliageInstance(instance_id=1))
        instances = cluster.get_instances()
        assert len(instances) == 2

    def test_get_visible_instances(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        cluster = FoliageCluster(bounds, "test_plant")
        cluster.add_instance(FoliageInstance(instance_id=0, visible=True))
        cluster.add_instance(FoliageInstance(instance_id=1, visible=False))
        visible = cluster.get_visible_instances()
        assert len(visible) == 1

    def test_visible_count(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        cluster = FoliageCluster(bounds, "test_plant")
        cluster.add_instance(FoliageInstance(instance_id=0, visible=True))
        cluster.add_instance(FoliageInstance(instance_id=1, visible=True))
        assert cluster.visible_count == 2

    def test_cull_all_visible(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        cluster = FoliageCluster(bounds, "test_plant")
        cluster.add_instance(FoliageInstance(instance_id=0, position=(25.0, 5.0, 25.0)))
        cluster.add_instance(FoliageInstance(instance_id=1, position=(10.0, 5.0, 10.0)))

        # Frustum that includes everything
        frustum = Frustum()
        visible = cluster.cull(frustum)
        assert visible == 2

    def test_update_lod(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=200.0, max_z=200.0)
        cluster = FoliageCluster(bounds, "test_plant")
        cluster.add_instance(FoliageInstance(instance_id=0, position=(10.0, 0.0, 0.0)))
        cluster.add_instance(FoliageInstance(instance_id=1, position=(200.0, 0.0, 0.0)))

        lod_distances = [50.0, 150.0, 500.0]
        cluster.update_lod((0.0, 0.0, 0.0), lod_distances)

        instances = cluster.get_instances()
        # First instance at 10 units -> LOD 0 (< 50)
        assert instances[0].lod_level == 0
        # Second instance at 200 units -> LOD 2 (>= 150, < 500)
        assert instances[1].lod_level == 2

    def test_clear(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        cluster = FoliageCluster(bounds, "test_plant")
        cluster.add_instance(FoliageInstance(instance_id=0))
        cluster.clear()
        assert cluster.instance_count == 0


# =============================================================================
# HierarchicalInstancedMesh
# =============================================================================


class TestHierarchicalInstancedMesh:
    def test_creation(self):
        ft = FoliageType(type_id="test_plant")
        hism = HierarchicalInstancedMesh(ft)
        assert hism.foliage_type == ft
        assert hism.total_instances == 0

    def test_add_instance(self):
        ft = FoliageType(type_id="test_plant")
        hism = HierarchicalInstancedMesh(ft)
        placement = PlacementResult(position=(10.0, 0.0, 20.0))
        instance_id = hism.add_instance(placement)
        assert instance_id == 0
        assert hism.total_instances == 1

    def test_add_instances(self):
        ft = FoliageType(type_id="test_plant")
        hism = HierarchicalInstancedMesh(ft)
        placements = [
            PlacementResult(position=(10.0, 0.0, 20.0)),
            PlacementResult(position=(30.0, 0.0, 40.0)),
            PlacementResult(position=(50.0, 0.0, 60.0)),
        ]
        ids = hism.add_instances(placements)
        assert len(ids) == 3
        assert hism.total_instances == 3

    def test_remove_instance(self):
        ft = FoliageType(type_id="test_plant")
        hism = HierarchicalInstancedMesh(ft)
        instance_id = hism.add_instance(PlacementResult(position=(10.0, 0.0, 20.0)))
        assert hism.remove_instance(instance_id) is True
        assert hism.total_instances == 0

    def test_remove_instance_not_found(self):
        ft = FoliageType(type_id="test_plant")
        hism = HierarchicalInstancedMesh(ft)
        assert hism.remove_instance(999) is False

    def test_remove_instances_in_bounds(self):
        ft = FoliageType(type_id="test_plant")
        hism = HierarchicalInstancedMesh(ft, cluster_size=100.0)

        # Add instances
        hism.add_instance(PlacementResult(position=(10.0, 0.0, 10.0)))
        hism.add_instance(PlacementResult(position=(20.0, 0.0, 20.0)))
        hism.add_instance(PlacementResult(position=(150.0, 0.0, 150.0)))

        # Remove within bounds
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=100.0, max_z=100.0)
        removed = hism.remove_instances_in_bounds(bounds)
        assert removed == 2
        assert hism.total_instances == 1

    def test_cluster_creation(self):
        ft = FoliageType(type_id="test_plant")
        hism = HierarchicalInstancedMesh(ft, cluster_size=50.0)

        # Add instances in different clusters
        hism.add_instance(PlacementResult(position=(10.0, 0.0, 10.0)))
        hism.add_instance(PlacementResult(position=(60.0, 0.0, 60.0)))

        assert hism.cluster_count == 2

    def test_update_visibility(self):
        ft = FoliageType(type_id="test_plant")
        hism = HierarchicalInstancedMesh(ft)
        hism.add_instance(PlacementResult(position=(10.0, 0.0, 10.0)))
        hism.add_instance(PlacementResult(position=(20.0, 0.0, 20.0)))

        # Update with empty frustum (everything visible)
        frustum = Frustum()
        hism.update_visibility((0.0, 0.0, 0.0), frustum)
        assert hism.visible_instances == 2

    def test_get_instance_buffer(self):
        ft = FoliageType(type_id="test_plant")
        hism = HierarchicalInstancedMesh(ft)
        hism.add_instance(PlacementResult(position=(10.0, 5.0, 20.0)))
        hism.add_instance(PlacementResult(position=(30.0, 5.0, 40.0)))

        buffer = hism.get_instance_buffer()
        assert len(buffer) == 2
        assert "position" in buffer[0]
        assert "rotation" in buffer[0]
        assert "scale" in buffer[0]
        assert "lod_level" in buffer[0]

    def test_get_instance_buffer_by_lod(self):
        ft = FoliageType(type_id="test_plant", lod_distances=[50.0, 150.0, 500.0])
        hism = HierarchicalInstancedMesh(ft)
        hism.add_instance(PlacementResult(position=(10.0, 0.0, 0.0)))
        hism.add_instance(PlacementResult(position=(100.0, 0.0, 0.0)))

        # Update LOD
        hism.update_visibility((0.0, 0.0, 0.0), Frustum())

        by_lod = hism.get_instance_buffer_by_lod()
        assert len(by_lod) >= 1

    def test_get_clusters_in_bounds(self):
        ft = FoliageType(type_id="test_plant")
        hism = HierarchicalInstancedMesh(ft, cluster_size=50.0)
        hism.add_instance(PlacementResult(position=(25.0, 0.0, 25.0)))
        hism.add_instance(PlacementResult(position=(75.0, 0.0, 75.0)))

        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=60.0, max_z=60.0)
        clusters = hism.get_clusters_in_bounds(bounds)
        assert len(clusters) >= 1

    def test_clear(self):
        ft = FoliageType(type_id="test_plant")
        hism = HierarchicalInstancedMesh(ft)
        hism.add_instance(PlacementResult(position=(10.0, 0.0, 20.0)))
        hism.clear()
        assert hism.total_instances == 0
        assert hism.cluster_count == 0


# =============================================================================
# FoliageManager
# =============================================================================


class TestFoliageManager:
    def test_creation(self):
        manager = FoliageManager()
        assert len(manager.get_registered_types()) == 0

    def test_register_type(self):
        manager = FoliageManager()
        ft = FoliageType(type_id="test_plant")
        manager.register_type(ft)
        assert "test_plant" in manager.get_registered_types()

    def test_register_duplicate_type(self):
        manager = FoliageManager()
        ft = FoliageType(type_id="test_plant")
        manager.register_type(ft)
        with pytest.raises(ValueError, match="already registered"):
            manager.register_type(ft)

    def test_unregister_type(self):
        manager = FoliageManager()
        ft = FoliageType(type_id="test_plant")
        manager.register_type(ft)
        assert manager.unregister_type("test_plant") is True
        assert "test_plant" not in manager.get_registered_types()

    def test_unregister_type_not_found(self):
        manager = FoliageManager()
        assert manager.unregister_type("nonexistent") is False

    def test_add_instances(self):
        manager = FoliageManager()
        ft = FoliageType(type_id="test_plant")
        manager.register_type(ft)

        placements = [
            PlacementResult(position=(10.0, 0.0, 20.0), foliage_type_id="test_plant"),
            PlacementResult(position=(30.0, 0.0, 40.0), foliage_type_id="test_plant"),
        ]
        ids = manager.add_instances("test_plant", placements)
        assert len(ids) == 2
        assert manager.get_total_instances() == 2

    def test_add_instances_unregistered_type(self):
        manager = FoliageManager()
        placements = [PlacementResult()]
        with pytest.raises(KeyError):
            manager.add_instances("nonexistent", placements)

    def test_remove_instances(self):
        manager = FoliageManager()
        ft = FoliageType(type_id="test_plant")
        manager.register_type(ft)
        manager.add_instances("test_plant", [
            PlacementResult(position=(10.0, 0.0, 10.0)),
            PlacementResult(position=(20.0, 0.0, 20.0)),
        ])

        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=100.0, max_z=100.0)
        removed = manager.remove_instances("test_plant", bounds)
        assert removed == 2

    def test_remove_all_instances(self):
        manager = FoliageManager()
        ft1 = FoliageType(type_id="plant_a")
        ft2 = FoliageType(type_id="plant_b")
        manager.register_type(ft1)
        manager.register_type(ft2)

        manager.add_instances("plant_a", [PlacementResult(position=(10.0, 0.0, 10.0))])
        manager.add_instances("plant_b", [PlacementResult(position=(20.0, 0.0, 20.0))])

        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=100.0, max_z=100.0)
        removed = manager.remove_all_instances(bounds)
        assert removed == 2

    def test_update(self):
        manager = FoliageManager()
        ft = FoliageType(type_id="test_plant")
        manager.register_type(ft)
        manager.add_instances("test_plant", [
            PlacementResult(position=(10.0, 0.0, 10.0)),
        ])

        manager.update((0.0, 0.0, 0.0), Frustum())
        assert manager.get_visible_instances() == 1

    def test_get_render_data(self):
        manager = FoliageManager()
        ft = FoliageType(type_id="test_plant")
        manager.register_type(ft)
        manager.add_instances("test_plant", [
            PlacementResult(position=(10.0, 0.0, 10.0)),
        ])

        render_data = manager.get_render_data()
        assert len(render_data) == 1
        assert render_data[0][0] == "test_plant"
        assert len(render_data[0][1]) == 1

    def test_get_render_data_by_lod(self):
        manager = FoliageManager()
        ft = FoliageType(type_id="test_plant")
        manager.register_type(ft)
        manager.add_instances("test_plant", [
            PlacementResult(position=(10.0, 0.0, 10.0)),
        ])

        render_data = manager.get_render_data_by_lod()
        assert len(render_data) >= 1

    def test_get_hism(self):
        manager = FoliageManager()
        ft = FoliageType(type_id="test_plant")
        manager.register_type(ft)

        hism = manager.get_hism("test_plant")
        assert hism is not None
        assert hism.foliage_type == ft

    def test_get_hism_not_found(self):
        manager = FoliageManager()
        assert manager.get_hism("nonexistent") is None

    def test_clear(self):
        manager = FoliageManager()
        ft = FoliageType(type_id="test_plant")
        manager.register_type(ft)
        manager.clear()
        assert len(manager.get_registered_types()) == 0

    def test_clear_type(self):
        manager = FoliageManager()
        ft = FoliageType(type_id="test_plant")
        manager.register_type(ft)
        manager.add_instances("test_plant", [PlacementResult(position=(10.0, 0.0, 10.0))])

        assert manager.clear_type("test_plant") is True
        assert manager.get_total_instances() == 0


# =============================================================================
# BatchedDescriptor
# =============================================================================


class TestBatchedDescriptor:
    def test_add_placements(self):
        manager = FoliageManager()
        ft = FoliageType(type_id="test_plant")
        manager.register_type(ft)

        batch = BatchedDescriptor(manager)
        batch.add("test_plant", [
            PlacementResult(position=(10.0, 0.0, 10.0)),
            PlacementResult(position=(20.0, 0.0, 20.0)),
        ])

        added, removed = batch.execute()
        assert added == 2
        assert removed == 0

    def test_remove_placements(self):
        manager = FoliageManager()
        ft = FoliageType(type_id="test_plant")
        manager.register_type(ft)
        manager.add_instances("test_plant", [
            PlacementResult(position=(10.0, 0.0, 10.0)),
            PlacementResult(position=(20.0, 0.0, 20.0)),
        ])

        batch = BatchedDescriptor(manager)
        batch.remove("test_plant", Bounds(min_x=0.0, min_z=0.0, max_x=100.0, max_z=100.0))

        added, removed = batch.execute()
        assert added == 0
        assert removed == 2

    def test_chained_operations(self):
        manager = FoliageManager()
        ft = FoliageType(type_id="test_plant")
        manager.register_type(ft)

        batch = BatchedDescriptor(manager)
        result = (
            batch
            .add("test_plant", [PlacementResult(position=(10.0, 0.0, 10.0))])
            .add("test_plant", [PlacementResult(position=(20.0, 0.0, 20.0))])
        )

        assert result is batch  # Chaining returns self
        added, removed = batch.execute()
        assert added == 2

    def test_clear_pending(self):
        manager = FoliageManager()
        ft = FoliageType(type_id="test_plant")
        manager.register_type(ft)

        batch = BatchedDescriptor(manager)
        batch.add("test_plant", [PlacementResult(position=(10.0, 0.0, 10.0))])
        batch.clear()

        added, removed = batch.execute()
        assert added == 0

    def test_unregistered_type_skipped(self):
        manager = FoliageManager()

        batch = BatchedDescriptor(manager)
        batch.add("nonexistent", [PlacementResult()])
        batch.remove("nonexistent", Bounds())

        # Should not raise, just skip
        added, removed = batch.execute()
        assert added == 0
        assert removed == 0
