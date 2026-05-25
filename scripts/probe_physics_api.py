"""Probe physics module public API signatures for BLACKBOX contract tests."""
import inspect
import engine.simulation.physics as p

targets = [
    'RigidBody', 'PhysicsWorld', 'PhysicsConfig', 'CollisionFilter',
    'AABB', 'SphereShape', 'BoxShape', 'CapsuleShape', 'Ray', 'RayHit',
    'overlap_sphere', 'sweep_sphere', 'raycast', 'raycast_closest',
    'PhysicsMaterial', 'BodyFlags', 'MassProperties', 'ConvexHullShape',
    'CompoundShape', 'MeshShape',
]

for name in targets:
    obj = getattr(p, name, None)
    if obj is None:
        print(f'{name}: NOT FOUND in engine.simulation.physics')
        continue
    try:
        sig = inspect.signature(obj.__init__)
        print(f'{name}{sig}')
    except (ValueError, TypeError):
        try:
            sig = inspect.signature(obj)
            print(f'{name}{sig}')
        except Exception as e:
            print(f'{name}: {e}')
