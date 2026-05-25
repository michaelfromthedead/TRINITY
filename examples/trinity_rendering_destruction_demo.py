"""
Demo: Trinity Pattern - Rendering and Destruction Decorators

This example demonstrates the new Tier 42 (RENDERING) and Tier 43 (DESTRUCTION)
decorators in action.
"""

from trinity.decorators.rendering import (
    gi_contributor,
    material_blend,
    material_domain,
    reflection_probe,
    render_layer,
    shadow_caster,
)
from trinity.decorators.destruction import (
    damage_resistance,
    damage_type,
    destructible,
    fracture,
    joint,
    physics_material,
)


# ============================================================================
# RENDERING EXAMPLES
# ============================================================================


@render_layer(layer="main", order=0)
@shadow_caster(mode="dynamic", resolution_scale=1.5)
@gi_contributor(importance="high", emissive=False)
class Hero:
    """Main character with dynamic shadows and high GI contribution."""

    pass


@material_blend(mode="translucent")
@material_domain(domain="surface")
@gi_contributor(importance="low")
class Glass:
    """Translucent glass surface with minimal GI contribution."""

    pass


@reflection_probe(capture_mode="realtime", resolution=512, update_rate=30.0)
@render_layer(layer="environment", order=-10)
class MirrorSurface:
    """Real-time reflective surface with high-res probe."""

    pass


@shadow_caster(mode="static")
@gi_contributor(importance="critical", emissive=True)
@material_domain(domain="surface")
class LightSource:
    """Emissive light source with static shadows and critical GI."""

    pass


# ============================================================================
# DESTRUCTION EXAMPLES
# ============================================================================


@fracture(pattern="voronoi", min_size=0.15)
@physics_material(friction=0.7, restitution=0.2, density=2.5)
@destructible(health=500.0, fracture_depth=3, debris_lifetime=15.0)
class ConcretePillar:
    """Destructible concrete pillar with voronoi fracture pattern."""

    pass


@damage_resistance(resistances={"fire": 0.9, "physical": 0.3, "explosive": 0.1})
@damage_type(id="fire", base_multiplier=1.5)
@destructible(health=200.0, fracture_depth=2)
class WoodenCrate:
    """Fire-resistant wooden crate."""

    pass


@joint(type="hinge", break_force=1000.0, break_torque=500.0)
@physics_material(friction=0.5, restitution=0.1, density=3.0)
@destructible(health=1000.0, fracture_depth=1)
class MetalDoor:
    """Metal door with hinge joint that breaks under force."""

    pass


@fracture(pattern="radial", min_size=0.05, interior_material="ice_crystal")
@damage_type(id="ice", base_multiplier=0.5)
@physics_material(friction=0.1, restitution=0.9, density=0.9)
@destructible(health=50.0, fracture_depth=4, debris_lifetime=5.0)
class IceSculpture:
    """Fragile ice sculpture with radial fracture."""

    pass


# ============================================================================
# COMBINED RENDERING + DESTRUCTION
# ============================================================================


@render_layer(layer="main", order=5)
@shadow_caster(mode="dynamic", resolution_scale=1.0)
@gi_contributor(importance="medium")
@material_blend(mode="opaque")
@material_domain(domain="surface")
@joint(type="fixed", break_force=5000.0)
@fracture(pattern="voronoi", min_size=0.2)
@damage_resistance(resistances={"physical": 0.7, "fire": 0.5})
@physics_material(friction=0.6, restitution=0.3, density=2.0)
@destructible(health=800.0, fracture_depth=2, debris_lifetime=20.0)
class ReinforcedWall:
    """
    Fully featured reinforced wall with both rendering and destruction.

    Rendering features:
    - Dynamic shadows with standard resolution
    - Medium GI contribution
    - Opaque surface material
    - Rendered in main layer at order 5

    Destruction features:
    - 800 health
    - Voronoi fracture with 2 levels of depth
    - Fixed joint that breaks at 5000N
    - Resistant to physical and fire damage
    - Standard concrete-like physics material
    """

    pass


# ============================================================================
# INSPECTION
# ============================================================================


def inspect_object(obj_class, name: str):
    """Print detailed information about a decorated object."""
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")

    # Applied decorators
    if hasattr(obj_class, "_applied_decorators"):
        print(f"Applied decorators: {', '.join(obj_class._applied_decorators)}")

    # Rendering properties
    if hasattr(obj_class, "_gi_contributor"):
        print(f"  GI: importance={obj_class._gi_importance}, emissive={obj_class._gi_emissive}")

    if hasattr(obj_class, "_shadow_caster"):
        print(f"  Shadow: mode={obj_class._shadow_mode}, scale={obj_class._shadow_resolution_scale}")

    if hasattr(obj_class, "_reflection_probe"):
        print(f"  Reflection: mode={obj_class._reflection_capture_mode}, res={obj_class._reflection_resolution}")

    if hasattr(obj_class, "_material_domain"):
        print(f"  Material: domain={obj_class._material_domain_type}")

    if hasattr(obj_class, "_material_blend"):
        print(f"  Blend: mode={obj_class._material_blend_mode}")

    if hasattr(obj_class, "_render_layer"):
        print(f"  Layer: {obj_class._render_layer_name} (order={obj_class._render_layer_order})")

    # Destruction properties
    if hasattr(obj_class, "_destructible"):
        print(f"  Destructible: health={obj_class._destructible_health}, depth={obj_class._destructible_fracture_depth}")

    if hasattr(obj_class, "_damage_type"):
        print(f"  Damage type: {obj_class._damage_type_id} (x{obj_class._damage_type_multiplier})")

    if hasattr(obj_class, "_damage_resistance"):
        print(f"  Resistances: {obj_class._damage_resistance_values}")

    if hasattr(obj_class, "_fracture"):
        print(f"  Fracture: pattern={obj_class._fracture_pattern}, min_size={obj_class._fracture_min_size}")

    if hasattr(obj_class, "_physics_material"):
        print(f"  Physics: friction={obj_class._physics_friction}, restitution={obj_class._physics_restitution}, density={obj_class._physics_density}")

    if hasattr(obj_class, "_joint"):
        print(f"  Joint: type={obj_class._joint_type}, break_force={obj_class._joint_break_force}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRINITY PATTERN - RENDERING & DESTRUCTION DEMO")
    print("=" * 60)

    # Inspect rendering examples
    inspect_object(Hero, "Hero Character")
    inspect_object(Glass, "Glass Surface")
    inspect_object(MirrorSurface, "Mirror Surface")
    inspect_object(LightSource, "Light Source")

    # Inspect destruction examples
    inspect_object(ConcretePillar, "Concrete Pillar")
    inspect_object(WoodenCrate, "Wooden Crate")
    inspect_object(MetalDoor, "Metal Door")
    inspect_object(IceSculpture, "Ice Sculpture")

    # Inspect combined example
    inspect_object(ReinforcedWall, "Reinforced Wall (Combined)")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
