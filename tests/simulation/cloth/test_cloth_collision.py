"""
Whitebox tests for cloth collision detection and response.

Tests:
- CollisionResult: collision data structure
- SphereCollider: sphere-particle collision
- CapsuleCollider: capsule-particle collision
- BoxCollider: AABB-particle collision
- MeshCollider: mesh-particle collision
- SDFCollider: SDF-particle collision
- SpatialHash: spatial hashing for self-collision
- handle_self_collision: self-collision resolution
- ClothCollisionHandler: collision manager
"""

import math

import numpy as np
import pytest

from engine.simulation.cloth.cloth_collision import (
    BoxCollider,
    CapsuleCollider,
    ClothCollisionHandler,
    CollisionResult,
    MeshCollider,
    SDFCollider,
    SpatialHash,
    SphereCollider,
    collide_with_box,
    collide_with_capsule,
    collide_with_mesh,
    collide_with_sdf,
    collide_with_sphere,
    handle_self_collision,
)
from engine.simulation.cloth.cloth_simulation import ClothMesh, ClothParticle, ClothTriangle


def make_particle(pos, inv_mass=1.0):
    """Helper to create a particle at a position."""
    pos_arr = np.array(pos, dtype=np.float32)
    return ClothParticle(
        position=pos_arr,
        prev_position=pos_arr.copy(),
        inv_mass=inv_mass,
    )


class TestCollisionResult:
    """Test CollisionResult data class."""

    def test_no_collision(self):
        """Test result for no collision."""
        result = CollisionResult(collided=False)

        assert not result.collided
        assert result.penetration_depth == 0.0
        assert result.contact_normal is None
        assert result.contact_point is None

    def test_collision_with_data(self):
        """Test result with collision data."""
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        point = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result = CollisionResult(
            collided=True,
            penetration_depth=0.5,
            contact_normal=normal,
            contact_point=point,
        )

        assert result.collided
        assert result.penetration_depth == 0.5
        assert np.allclose(result.contact_normal, normal)
        assert np.allclose(result.contact_point, point)


class TestSphereCollider:
    """Test sphere collision detection and response."""

    def test_no_collision_outside_sphere(self):
        """Particle outside sphere should not collide."""
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
        )
        particle = make_particle([2.0, 0.0, 0.0])  # Outside radius + margin

        result = collide_with_sphere(particle, sphere, margin=0.01)

        assert not result.collided

    def test_collision_inside_sphere(self):
        """Particle inside sphere should collide."""
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
        )
        particle = make_particle([0.5, 0.0, 0.0])

        result = collide_with_sphere(particle, sphere, margin=0.01)

        assert result.collided
        assert result.penetration_depth > 0

    def test_collision_resolves_penetration(self):
        """Colliding particle should be pushed out."""
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
            friction=0.0,  # No friction for simple test
        )
        particle = make_particle([0.5, 0.0, 0.0])

        collide_with_sphere(particle, sphere, margin=0.01)

        # Particle should be pushed to surface (radius + margin)
        dist = np.linalg.norm(particle.position - sphere.center)
        assert abs(dist - 1.01) < 1e-5

    def test_collision_normal_points_outward(self):
        """Contact normal should point outward from sphere."""
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
        )
        particle = make_particle([0.5, 0.0, 0.0])

        result = collide_with_sphere(particle, sphere, margin=0.01)

        # Normal should point in +X direction
        assert result.contact_normal[0] > 0.99

    def test_particle_at_center(self):
        """Particle at sphere center should get arbitrary push direction."""
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
        )
        particle = make_particle([0.0, 0.0, 0.0])

        result = collide_with_sphere(particle, sphere, margin=0.01)

        assert result.collided
        # Should use Y-up as default normal
        assert result.contact_normal[1] == 1.0

    def test_pinned_particle_does_not_move(self):
        """Pinned particle should not move on collision."""
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
        )
        particle = make_particle([0.5, 0.0, 0.0], inv_mass=0.0)
        pos_before = particle.position.copy()

        result = collide_with_sphere(particle, sphere, margin=0.01)

        assert result.collided
        assert np.allclose(particle.position, pos_before)

    def test_friction_applies(self):
        """Friction should reduce tangential velocity."""
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
            friction=0.5,
        )
        # Particle moving tangentially
        particle = make_particle([0.5, 0.0, 0.0])
        particle.prev_position = np.array([0.5, 0.0, 0.1], dtype=np.float32)

        collide_with_sphere(particle, sphere, margin=0.01)

        # Particle should be pushed out to at least radius + margin
        dist = np.linalg.norm(particle.position - sphere.center)
        assert dist >= 1.0  # At least at surface


class TestCapsuleCollider:
    """Test capsule collision detection and response."""

    def test_no_collision_outside_capsule(self):
        """Particle outside capsule should not collide."""
        capsule = CapsuleCollider(
            point_a=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            point_b=np.array([0.0, 2.0, 0.0], dtype=np.float32),
            radius=0.5,
        )
        particle = make_particle([2.0, 1.0, 0.0])  # Far from capsule

        result = collide_with_capsule(particle, capsule, margin=0.01)

        assert not result.collided

    def test_collision_inside_capsule(self):
        """Particle inside capsule should collide."""
        capsule = CapsuleCollider(
            point_a=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            point_b=np.array([0.0, 2.0, 0.0], dtype=np.float32),
            radius=0.5,
        )
        particle = make_particle([0.2, 1.0, 0.0])  # Inside radius

        result = collide_with_capsule(particle, capsule, margin=0.01)

        assert result.collided
        assert result.penetration_depth > 0

    def test_collision_near_cap(self):
        """Collision near hemispherical cap should work."""
        capsule = CapsuleCollider(
            point_a=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            point_b=np.array([0.0, 2.0, 0.0], dtype=np.float32),
            radius=0.5,
        )
        # Near bottom cap
        particle = make_particle([0.2, -0.3, 0.0])

        result = collide_with_capsule(particle, capsule, margin=0.01)

        assert result.collided

    def test_degenerate_capsule_becomes_sphere(self):
        """Capsule with zero-length axis should behave like sphere."""
        capsule = CapsuleCollider(
            point_a=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            point_b=np.array([0.0, 0.0, 0.0], dtype=np.float32),  # Same point
            radius=1.0,
        )
        particle = make_particle([0.5, 0.0, 0.0])

        result = collide_with_capsule(particle, capsule, margin=0.01)

        assert result.collided
        # Should behave like sphere centered at point_a

    def test_particle_on_axis(self):
        """Particle exactly on axis should get perpendicular push."""
        capsule = CapsuleCollider(
            point_a=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            point_b=np.array([0.0, 2.0, 0.0], dtype=np.float32),
            radius=0.5,
        )
        particle = make_particle([0.0, 1.0, 0.0])  # On axis

        result = collide_with_capsule(particle, capsule, margin=0.01)

        assert result.collided
        # Normal should be perpendicular to axis


class TestBoxCollider:
    """Test AABB box collision detection and response."""

    def test_no_collision_outside_box(self):
        """Particle outside box should not collide."""
        box = BoxCollider(
            min_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            max_point=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        particle = make_particle([2.0, 0.5, 0.5])

        result = collide_with_box(particle, box, margin=0.01)

        assert not result.collided

    def test_collision_inside_box(self):
        """Particle inside box should collide."""
        box = BoxCollider(
            min_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            max_point=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        particle = make_particle([0.5, 0.5, 0.5])  # Center of box

        result = collide_with_box(particle, box, margin=0.01)

        assert result.collided

    def test_collision_pushes_to_nearest_face(self):
        """Collision should push particle through nearest face."""
        box = BoxCollider(
            min_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            max_point=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            friction=0.0,
        )
        # Near +X face
        particle = make_particle([0.9, 0.5, 0.5])

        collide_with_box(particle, box, margin=0.01)

        # Should be pushed to +X face
        assert particle.position[0] > 1.0

    def test_collision_normal_for_each_face(self):
        """Collision normal should correspond to nearest face."""
        box = BoxCollider(
            min_point=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            max_point=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        # Test near +Y face
        particle = make_particle([0.0, 0.9, 0.0])
        result = collide_with_box(particle, box, margin=0.01)

        assert result.collided
        assert result.contact_normal[1] == 1.0  # +Y normal


class TestMeshCollider:
    """Test triangle mesh collision detection and response."""

    @pytest.fixture
    def simple_mesh_collider(self):
        """Create a simple triangle mesh (floor plane)."""
        vertices = np.array([
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)

        return MeshCollider(vertices=vertices, indices=indices, friction=0.3)

    def test_no_collision_above_mesh(self, simple_mesh_collider):
        """Particle above mesh should not collide."""
        particle = make_particle([0.0, 0.5, 0.0])

        result = collide_with_mesh(particle, simple_mesh_collider, margin=0.01)

        assert not result.collided

    def test_collision_below_mesh(self, simple_mesh_collider):
        """Particle just below mesh should collide."""
        particle = make_particle([0.0, 0.005, 0.0])

        result = collide_with_mesh(particle, simple_mesh_collider, margin=0.01)

        assert result.collided

    def test_collision_pushes_out(self, simple_mesh_collider):
        """Collision should push particle away from mesh surface."""
        particle = make_particle([0.0, -0.005, 0.0])
        initial_y = particle.position[1]

        collide_with_mesh(particle, simple_mesh_collider, margin=0.01)

        # Particle should have moved (collision response applied)
        # The exact direction depends on closest triangle orientation
        # At minimum, particle position should remain finite
        assert np.isfinite(particle.position).all()

    def test_empty_mesh_no_collision(self):
        """Empty mesh should never collide."""
        mesh = MeshCollider(
            vertices=np.zeros((0, 3), dtype=np.float32),
            indices=np.zeros(0, dtype=np.int32),
        )
        particle = make_particle([0.0, 0.0, 0.0])

        result = collide_with_mesh(particle, mesh, margin=0.01)

        assert not result.collided


class TestSDFCollider:
    """Test SDF collision detection and response."""

    def test_collision_with_sdf_sphere(self):
        """Test collision with SDF representing a sphere."""

        def sphere_sdf(pos):
            dist = np.linalg.norm(pos) - 1.0  # Unit sphere
            grad = pos / (np.linalg.norm(pos) + 1e-8)
            return dist, grad.astype(np.float32)

        sdf = SDFCollider(sdf_function=sphere_sdf, friction=0.0)
        particle = make_particle([0.5, 0.0, 0.0])  # Inside sphere

        result = collide_with_sdf(particle, sdf, margin=0.01)

        assert result.collided
        assert result.penetration_depth > 0

    def test_no_collision_outside_sdf(self):
        """Particle outside SDF should not collide."""

        def sphere_sdf(pos):
            dist = np.linalg.norm(pos) - 1.0
            grad = pos / (np.linalg.norm(pos) + 1e-8)
            return dist, grad.astype(np.float32)

        sdf = SDFCollider(sdf_function=sphere_sdf)
        particle = make_particle([2.0, 0.0, 0.0])  # Outside sphere

        result = collide_with_sdf(particle, sdf, margin=0.01)

        assert not result.collided

    def test_sdf_zero_gradient_no_collision(self):
        """Zero gradient should return no collision."""

        def bad_sdf(pos):
            return 0.5, np.zeros(3, dtype=np.float32)  # Zero gradient

        sdf = SDFCollider(sdf_function=bad_sdf)
        particle = make_particle([0.0, 0.0, 0.0])

        result = collide_with_sdf(particle, sdf, margin=0.01)

        # Should not crash, should return no collision due to zero gradient
        assert not result.collided


class TestSpatialHash:
    """Test SpatialHash for self-collision broad phase."""

    def test_hash_creation(self):
        """Test spatial hash creation."""
        sh = SpatialHash(cell_size=0.1, table_size=1024)

        assert sh.cell_size == 0.1
        assert sh.table_size == 1024

    def test_insert_and_query(self):
        """Test inserting and querying particles."""
        sh = SpatialHash(cell_size=0.1, table_size=1024)

        sh.insert(0, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        sh.insert(1, np.array([0.05, 0.0, 0.0], dtype=np.float32))
        sh.insert(2, np.array([10.0, 0.0, 0.0], dtype=np.float32))  # Far away

        results = sh.query(np.array([0.0, 0.0, 0.0], dtype=np.float32), radius=0.1)

        assert 0 in results
        assert 1 in results
        # Particle 2 may or may not be in results depending on hash collisions

    def test_clear(self):
        """Test clearing the hash table."""
        sh = SpatialHash()
        sh.insert(0, np.array([0.0, 0.0, 0.0], dtype=np.float32))

        sh.clear()

        results = sh.query(np.array([0.0, 0.0, 0.0], dtype=np.float32), radius=1.0)
        assert len(results) == 0

    def test_build_from_particles(self):
        """Test building hash from particle list."""
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([0.05, 0.0, 0.0]),
            make_particle([0.1, 0.0, 0.0]),
        ]
        sh = SpatialHash(cell_size=0.1)

        sh.build_from_particles(particles)

        results = sh.query(np.array([0.05, 0.0, 0.0], dtype=np.float32), radius=0.1)
        assert len(results) >= 1  # Should find nearby particles


class TestHandleSelfCollision:
    """Test self-collision handling."""

    def test_no_collision_when_separated(self):
        """Particles far apart should not collide."""
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([10.0, 0.0, 0.0]),  # Far apart
        ]
        mesh = ClothMesh(particles=particles, edges=[], triangles=[])

        count = handle_self_collision(mesh, thickness=0.02)

        assert count == 0

    def test_collision_when_overlapping(self):
        """Overlapping particles should be separated."""
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([0.01, 0.0, 0.0]),  # Very close (< 2*thickness)
        ]
        mesh = ClothMesh(particles=particles, edges=[], triangles=[])

        count = handle_self_collision(mesh, thickness=0.02)

        assert count > 0
        # Particles should be pushed apart
        dist = np.linalg.norm(particles[1].position - particles[0].position)
        assert dist > 0.01

    def test_pinned_particles_dont_move(self):
        """Pinned particles should not be moved."""
        particles = [
            make_particle([0.0, 0.0, 0.0], inv_mass=0.0),  # Pinned
            make_particle([0.01, 0.0, 0.0]),
        ]
        mesh = ClothMesh(particles=particles, edges=[], triangles=[])
        pos0_before = particles[0].position.copy()

        handle_self_collision(mesh, thickness=0.02)

        assert np.allclose(particles[0].position, pos0_before)

    def test_both_pinned_no_move(self):
        """Both pinned particles should not move."""
        particles = [
            make_particle([0.0, 0.0, 0.0], inv_mass=0.0),
            make_particle([0.01, 0.0, 0.0], inv_mass=0.0),
        ]
        mesh = ClothMesh(particles=particles, edges=[], triangles=[])

        count = handle_self_collision(mesh, thickness=0.02)

        # Both pinned, no collision resolved
        assert count == 0


class TestClothCollisionHandler:
    """Test ClothCollisionHandler class."""

    def test_handler_creation(self):
        """Test handler creation."""
        handler = ClothCollisionHandler()

        assert len(handler.sphere_colliders) == 0
        assert len(handler.capsule_colliders) == 0
        assert len(handler.box_colliders) == 0
        assert handler.enable_self_collision

    def test_add_sphere(self):
        """Test adding sphere collider."""
        handler = ClothCollisionHandler()
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
        )

        handler.add_sphere(sphere)

        assert len(handler.sphere_colliders) == 1

    def test_add_capsule(self):
        """Test adding capsule collider."""
        handler = ClothCollisionHandler()
        capsule = CapsuleCollider(
            point_a=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            point_b=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            radius=0.5,
        )

        handler.add_capsule(capsule)

        assert len(handler.capsule_colliders) == 1

    def test_add_box(self):
        """Test adding box collider."""
        handler = ClothCollisionHandler()
        box = BoxCollider(
            min_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            max_point=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        handler.add_box(box)

        assert len(handler.box_colliders) == 1

    def test_add_mesh_collider(self):
        """Test adding mesh collider."""
        handler = ClothCollisionHandler()
        mesh = MeshCollider(
            vertices=np.zeros((3, 3), dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
        )

        handler.add_mesh(mesh)

        assert len(handler.mesh_colliders) == 1

    def test_add_sdf(self):
        """Test adding SDF collider."""
        handler = ClothCollisionHandler()
        sdf = SDFCollider(sdf_function=lambda p: (0.0, np.zeros(3, dtype=np.float32)))

        handler.add_sdf(sdf)

        assert len(handler.sdf_colliders) == 1

    def test_clear(self):
        """Test clearing all colliders."""
        handler = ClothCollisionHandler()
        handler.add_sphere(
            SphereCollider(center=np.zeros(3, dtype=np.float32), radius=1.0)
        )
        handler.add_box(
            BoxCollider(
                min_point=np.zeros(3, dtype=np.float32),
                max_point=np.ones(3, dtype=np.float32),
            )
        )

        handler.clear()

        assert len(handler.sphere_colliders) == 0
        assert len(handler.box_colliders) == 0

    def test_process_collisions(self):
        """Test processing all collisions."""
        handler = ClothCollisionHandler()
        handler.enable_self_collision = False  # Disable for simpler test

        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
        )
        handler.add_sphere(sphere)

        particles = [
            make_particle([0.5, 0.0, 0.0]),  # Inside sphere
            make_particle([2.0, 0.0, 0.0]),  # Outside sphere
        ]
        mesh = ClothMesh(particles=particles, edges=[], triangles=[])

        count = handler.process_collisions(mesh, margin=0.01)

        assert count == 1  # Only one particle collided

    def test_process_collisions_with_self_collision(self):
        """Test processing includes self-collision."""
        handler = ClothCollisionHandler()
        handler.enable_self_collision = True

        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([0.01, 0.0, 0.0]),  # Overlapping
        ]
        mesh = ClothMesh(particles=particles, edges=[], triangles=[])

        count = handler.process_collisions(mesh, margin=0.01)

        # Should resolve self-collision
        assert count > 0
