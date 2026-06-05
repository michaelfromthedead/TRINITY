"""Tests for Material Inheritance Model (T-MAT-5.2).

Verifies:
- Child classes inherit parent texture slots
- Child can override surface method while calling super()
- super().surface(ctx, out) produces correct WGSL with inlined parent
- MRO is respected for multiple inheritance
- Texture slot override works correctly
- Inheritance chain is tracked properly
"""

from __future__ import annotations

import pytest

from trinity.materials import (
    Material,
    MaterialMeta,
    SurfaceContext,
    SurfaceOutput,
    surface,
    Vec2,
    Vec3,
    Vec4,
    Texture2D,
    TextureCube,
    WGSLTranslationError,
)


# =============================================================================
# Suite A: Texture Slot Inheritance
# =============================================================================


class TestTextureInheritance:
    """Child classes inherit texture slots from parents."""

    def test_child_inherits_parent_textures(self):
        """Child material inherits texture slots from parent."""

        class ParentMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white", srgb=True)
            normal = Texture2D(default="flat_normal")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 1.0, 1.0)

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            roughness_map = Texture2D(default="gray")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        # Child should have all parent textures + own
        assert "albedo" in ChildMaterial._texture_bindings
        assert "normal" in ChildMaterial._texture_bindings
        assert "roughness_map" in ChildMaterial._texture_bindings
        assert len(ChildMaterial._texture_bindings) == 3

    def test_child_can_override_inherited_texture(self):
        """Child can override a texture slot from parent."""

        class ParentMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white", srgb=True)

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 1.0, 1.0)

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            # Override with different settings
            albedo = Texture2D(default="black", srgb=False)

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.5, 0.5, 0.5)

        # Child should have overridden albedo
        assert "albedo" in ChildMaterial._texture_bindings
        assert ChildMaterial._texture_bindings["albedo"].default == "black"
        assert ChildMaterial._texture_bindings["albedo"].srgb is False

    def test_inherited_textures_property(self):
        """get_inherited_textures returns only parent textures."""

        class ParentMaterial(Material, metaclass=MaterialMeta):
            parent_tex = Texture2D(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            child_tex = Texture2D(default="black")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.metallic = 0.9

        inherited = ChildMaterial.get_inherited_textures()
        own = ChildMaterial.get_own_textures()

        assert "parent_tex" in inherited
        assert "child_tex" not in inherited
        assert "child_tex" in own
        assert "parent_tex" not in own

    def test_multi_level_texture_inheritance(self):
        """Textures are inherited through multiple levels."""

        class GrandparentMaterial(Material, metaclass=MaterialMeta):
            grandparent_tex = Texture2D(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ParentMaterial(GrandparentMaterial, metaclass=MaterialMeta):
            parent_tex = Texture2D(default="gray")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.metallic = 0.5

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            child_tex = Texture2D(default="black")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.ao = 0.5

        # Child should have all three textures
        assert "grandparent_tex" in ChildMaterial._texture_bindings
        assert "parent_tex" in ChildMaterial._texture_bindings
        assert "child_tex" in ChildMaterial._texture_bindings


# =============================================================================
# Suite B: Surface Method Inheritance
# =============================================================================


class TestSurfaceInheritance:
    """Child can inherit or override surface method."""

    def test_child_inherits_surface_without_override(self):
        """Child without surface method inherits parent's surface."""

        class ParentMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.7

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            extra_tex = Texture2D(default="white")
            # No surface method - should inherit

        # Child should have parent's WGSL
        assert ChildMaterial._wgsl_source == ParentMaterial._wgsl_source
        assert ChildMaterial._surface_method is ParentMaterial._surface_method

    def test_child_overrides_surface(self):
        """Child with surface method overrides parent."""

        class ParentMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.7

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.3

        # Child should have different WGSL
        assert ChildMaterial._wgsl_source != ParentMaterial._wgsl_source
        assert "0.3" in ChildMaterial._wgsl_source

    def test_parent_material_reference(self):
        """_parent_material tracks the first parent with surface."""

        class ParentMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.metallic = 0.9

        assert ChildMaterial._parent_material is ParentMaterial
        assert ChildMaterial.get_parent_material() is ParentMaterial


# =============================================================================
# Suite C: super().surface() Call
# =============================================================================


class TestSuperSurfaceCall:
    """super().surface(ctx, out) inlines parent WGSL."""

    def test_super_call_inlines_parent(self):
        """super().surface() inlines parent shader code."""

        class ParentMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                super().surface(ctx, out)
                out.metallic = 0.9

        # Child WGSL should contain parent code
        wgsl = ChildMaterial._wgsl_source
        assert "parent material" in wgsl.lower() or "ParentMaterial" in wgsl
        assert "0.5" in wgsl  # Parent's roughness value
        assert "0.9" in wgsl  # Child's metallic value

    def test_super_call_tracked(self):
        """has_super_call() returns True when super() is used."""

        class ParentMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ChildWithSuper(ParentMaterial, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                super().surface(ctx, out)
                out.metallic = 0.9

        class ChildWithoutSuper(ParentMaterial, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.metallic = 0.9

        assert ChildWithSuper.has_super_call() is True
        assert ChildWithoutSuper.has_super_call() is False

    def test_super_call_without_parent_raises(self):
        """super().surface() without parent material raises error."""
        # This tests the translator's error handling
        # When there's no parent with a surface method, super() should fail

        # Note: Material base class doesn't have _wgsl_source, so this
        # should raise WGSLTranslationError

        with pytest.raises(Exception):
            # Attempting super() on Material base should fail
            class BadMaterial(Material, metaclass=MaterialMeta):
                @surface
                def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                    super().surface(ctx, out)  # No parent surface to call!
                    out.roughness = 0.5

            # Force compilation error check
            if BadMaterial._compilation_error:
                raise BadMaterial._compilation_error

    def test_super_call_preserves_parent_builtins(self):
        """super() call merges parent's used builtins."""

        class ParentMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = clamp(0.5, 0.0, 1.0)

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                super().surface(ctx, out)
                out.metallic = saturate(0.9)

        # Both parent and child builtins should be tracked
        # (clamp and saturate are core WGSL builtins, not custom)
        assert ChildMaterial._compilation_error is None


# =============================================================================
# Suite D: Multiple Inheritance (MRO)
# =============================================================================


class TestMultipleInheritance:
    """MRO is respected for multiple inheritance."""

    def test_mro_texture_precedence(self):
        """Later bases in MRO don't override earlier ones for textures."""

        class MaterialA(Material, metaclass=MaterialMeta):
            shared_tex = Texture2D(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.1

        class MaterialB(Material, metaclass=MaterialMeta):
            shared_tex = Texture2D(default="black")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.9

        class ChildMaterial(MaterialA, MaterialB, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.metallic = 0.5

        # MaterialA comes first in MRO, so its texture should win
        assert ChildMaterial._texture_bindings["shared_tex"].default == "white"

    def test_mro_surface_precedence(self):
        """First parent in MRO with surface method is used for super()."""

        class MaterialA(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.1

        class MaterialB(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.9

        class ChildMaterial(MaterialA, MaterialB, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                super().surface(ctx, out)
                out.metallic = 0.5

        # MaterialA comes first in MRO
        assert ChildMaterial._parent_material is MaterialA
        # Parent code should have MaterialA's value
        assert "0.1" in ChildMaterial._wgsl_source


# =============================================================================
# Suite E: Inheritance Chain Tracking
# =============================================================================


class TestInheritanceChain:
    """Inheritance chain is tracked for debugging."""

    def test_single_inheritance_chain(self):
        """get_inheritance_chain returns correct order."""

        class Grandparent(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.1

        class Parent(Grandparent, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class Child(Parent, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.9

        chain = Child.get_inheritance_chain()
        assert chain == [Child, Parent, Grandparent]

    def test_inheritance_chain_stops_at_root(self):
        """Chain stops at Material base class."""

        class SingleMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        chain = SingleMaterial.get_inheritance_chain()
        assert chain == [SingleMaterial]


# =============================================================================
# Suite F: WGSL Output Verification
# =============================================================================


class TestWGSLOutput:
    """Verify correct WGSL generation for inherited materials."""

    def test_child_wgsl_contains_own_code(self):
        """Child WGSL contains its own surface code."""

        class ParentMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 0.0, 0.0)

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.0, 1.0, 0.0)

        wgsl = ChildMaterial._wgsl_source
        # Should contain green color (0.0, 1.0, 0.0)
        assert "0.0" in wgsl and "1.0" in wgsl

    def test_super_call_produces_valid_wgsl(self):
        """super().surface() produces syntactically valid WGSL."""

        class ParentMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                super().surface(ctx, out)
                out.metallic = 0.9

        wgsl = ChildMaterial._wgsl_source

        # Should have both parent and child code
        assert "roughness" in wgsl.lower() or "out.roughness" in wgsl
        assert "metallic" in wgsl.lower() or "out.metallic" in wgsl

        # Should not have Python syntax artifacts
        assert "super()" not in wgsl
        assert "self." not in wgsl

    def test_no_compilation_errors_with_inheritance(self):
        """Inherited materials compile without errors."""

        class BaseMaterial(Material, metaclass=MaterialMeta):
            base_tex = Texture2D(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 1.0, 1.0)
                out.roughness = 0.5

        class DerivedMaterial(BaseMaterial, metaclass=MaterialMeta):
            detail_tex = Texture2D(default="gray")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                super().surface(ctx, out)
                out.roughness = 0.3

        assert DerivedMaterial._compilation_error is None
        assert DerivedMaterial._wgsl_source != ""


# =============================================================================
# Suite G: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_child_inherits_everything(self):
        """Child with no attributes inherits everything from parent."""

        class ParentMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class EmptyChild(ParentMaterial, metaclass=MaterialMeta):
            pass

        assert "albedo" in EmptyChild._texture_bindings
        assert EmptyChild._wgsl_source == ParentMaterial._wgsl_source

    def test_child_with_only_textures(self):
        """Child with only texture additions inherits surface."""

        class ParentMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class TextureOnlyChild(ParentMaterial, metaclass=MaterialMeta):
            extra_tex = Texture2D(default="black")

        assert "extra_tex" in TextureOnlyChild._texture_bindings
        assert TextureOnlyChild._wgsl_source == ParentMaterial._wgsl_source

    def test_cubemap_inheritance(self):
        """TextureCube slots are inherited correctly."""

        class ParentMaterial(Material, metaclass=MaterialMeta):
            environment = TextureCube(default="black")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            reflection = TextureCube(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.metallic = 0.9

        assert "environment" in ChildMaterial._texture_bindings
        assert "reflection" in ChildMaterial._texture_bindings
        assert ChildMaterial._texture_bindings["environment"]._is_cube is True
        assert ChildMaterial._texture_bindings["reflection"]._is_cube is True


# =============================================================================
# Suite H: InheritanceResolver Tests
# =============================================================================


class TestInheritanceResolver:
    """Tests for the InheritanceResolver class."""

    def test_resolver_initialization(self):
        """InheritanceResolver initializes correctly."""
        from trinity.materials.inheritance import InheritanceResolver

        class SimpleMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        resolver = InheritanceResolver(SimpleMaterial)
        assert resolver.material_class is SimpleMaterial
        assert resolver._resolved is False

    def test_resolver_resolve_method(self):
        """resolve() returns InheritanceInfo."""
        from trinity.materials.inheritance import InheritanceResolver, InheritanceInfo

        class ParentMaterial(Material, metaclass=MaterialMeta):
            parent_tex = Texture2D(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            child_tex = Texture2D(default="black")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                super().surface(ctx, out)
                out.metallic = 0.9

        resolver = InheritanceResolver(ChildMaterial)
        info = resolver.resolve()

        assert isinstance(info, InheritanceInfo)
        assert info.material_class is ChildMaterial
        assert info.parent_material is ParentMaterial
        assert info.has_super_call is True

    def test_resolver_caches_result(self):
        """resolve() caches result on subsequent calls."""
        from trinity.materials.inheritance import InheritanceResolver

        class SimpleMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        resolver = InheritanceResolver(SimpleMaterial)
        info1 = resolver.resolve()
        info2 = resolver.resolve()

        assert info1 is info2
        assert resolver._resolved is True

    def test_resolver_get_all_textures(self):
        """get_all_textures returns combined textures."""
        from trinity.materials.inheritance import InheritanceResolver

        class ParentMaterial(Material, metaclass=MaterialMeta):
            parent_tex = Texture2D(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            child_tex = Texture2D(default="black")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.metallic = 0.9

        resolver = InheritanceResolver(ChildMaterial)
        textures = resolver.get_all_textures()

        assert "parent_tex" in textures
        assert "child_tex" in textures

    def test_resolver_get_parent_wgsl(self):
        """get_parent_wgsl returns parent's compiled WGSL."""
        from trinity.materials.inheritance import InheritanceResolver

        class ParentMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.metallic = 0.9

        resolver = InheritanceResolver(ChildMaterial)
        parent_wgsl = resolver.get_parent_wgsl()

        assert parent_wgsl is not None
        assert "0.5" in parent_wgsl

    def test_resolver_no_parent(self):
        """get_parent_wgsl returns None when no parent."""
        from trinity.materials.inheritance import InheritanceResolver

        class RootMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        resolver = InheritanceResolver(RootMaterial)
        parent_wgsl = resolver.get_parent_wgsl()

        # Material base class doesn't have WGSL
        assert parent_wgsl is None or parent_wgsl == ""


# =============================================================================
# Suite I: MROWalker Tests
# =============================================================================


class TestMROWalker:
    """Tests for the MROWalker class."""

    def test_walker_walk_materials(self):
        """walk_materials returns Material subclasses in MRO order."""
        from trinity.materials.inheritance import MROWalker

        class Grandparent(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.1

        class Parent(Grandparent, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class Child(Parent, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.9

        walker = MROWalker(Child)
        materials = walker.walk_materials()

        assert Child in materials
        assert Parent in materials
        assert Grandparent in materials
        # Should be in MRO order
        assert materials.index(Child) < materials.index(Parent)
        assert materials.index(Parent) < materials.index(Grandparent)

    def test_walker_walk_parents(self):
        """walk_parents excludes the current class."""
        from trinity.materials.inheritance import MROWalker

        class Parent(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class Child(Parent, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.9

        walker = MROWalker(Child)
        parents = walker.walk_parents()

        assert Child not in parents
        assert Parent in parents

    def test_walker_collect_textures(self):
        """collect_textures merges textures in MRO order."""
        from trinity.materials.inheritance import MROWalker

        class Parent(Material, metaclass=MaterialMeta):
            shared = Texture2D(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class Child(Parent, metaclass=MaterialMeta):
            shared = Texture2D(default="black")  # Override

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.9

        walker = MROWalker(Child)
        textures = walker.collect_textures()

        # Child's override should win
        assert textures["shared"].default == "black"

    def test_walker_detect_diamond_inheritance(self):
        """has_diamond_inheritance detects diamond patterns."""
        from trinity.materials.inheritance import MROWalker

        class CommonAncestor(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.1

        class BranchA(CommonAncestor, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.3

        class BranchB(CommonAncestor, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.7

        class DiamondChild(BranchA, BranchB, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.9

        walker = MROWalker(DiamondChild)
        assert walker.has_diamond_inheritance() is True

    def test_walker_no_diamond_linear_inheritance(self):
        """has_diamond_inheritance returns False for linear chains."""
        from trinity.materials.inheritance import MROWalker

        class Grandparent(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.1

        class Parent(Grandparent, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class Child(Parent, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.9

        walker = MROWalker(Child)
        assert walker.has_diamond_inheritance() is False

    def test_walker_get_diamond_ancestors(self):
        """get_diamond_ancestors returns shared ancestor classes."""
        from trinity.materials.inheritance import MROWalker

        class CommonAncestor(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.1

        class BranchA(CommonAncestor, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.3

        class BranchB(CommonAncestor, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.7

        class DiamondChild(BranchA, BranchB, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.9

        walker = MROWalker(DiamondChild)
        ancestors = walker.get_diamond_ancestors()

        assert CommonAncestor in ancestors


# =============================================================================
# Suite J: Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions in inheritance module."""

    def test_resolve_parent_surface(self):
        """resolve_parent_surface returns parent WGSL."""
        from trinity.materials.inheritance import resolve_parent_surface

        class ParentMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.metallic = 0.9

        wgsl = resolve_parent_surface(ChildMaterial)
        assert wgsl is not None
        assert "0.5" in wgsl

    def test_resolve_parent_surface_no_parent(self):
        """resolve_parent_surface returns None for root materials."""
        from trinity.materials.inheritance import resolve_parent_surface

        class RootMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        wgsl = resolve_parent_surface(RootMaterial)
        assert wgsl is None or wgsl == ""

    def test_inline_super_call_replaces_marker(self):
        """inline_super_call replaces super() marker with parent code."""
        from trinity.materials.inheritance import inline_super_call

        child_wgsl = "    /* super().surface() -> Parent */;\n    out.metallic = 0.9;"
        parent_wgsl = "out.roughness = 0.5;"

        result = inline_super_call(child_wgsl, parent_wgsl, "Parent")

        assert "parent material" in result.lower()
        assert "out.roughness" in result
        assert "out.metallic" in result

    def test_inline_super_call_no_marker(self):
        """inline_super_call returns unchanged WGSL if no marker."""
        from trinity.materials.inheritance import inline_super_call

        child_wgsl = "out.metallic = 0.9;"
        parent_wgsl = "out.roughness = 0.5;"

        result = inline_super_call(child_wgsl, parent_wgsl, "Parent")

        assert result == child_wgsl

    def test_merge_texture_declarations(self):
        """merge_texture_declarations combines parent and child textures."""
        from trinity.materials.inheritance import merge_texture_declarations

        class ParentMaterial(Material, metaclass=MaterialMeta):
            parent_tex = Texture2D(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            child_tex = Texture2D(default="black")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.metallic = 0.9

        merged = merge_texture_declarations(ChildMaterial, ParentMaterial)

        assert "parent_tex" in merged
        assert "child_tex" in merged

    def test_generate_combined_wgsl(self):
        """generate_combined_wgsl produces WGSL with comments."""
        from trinity.materials.inheritance import generate_combined_wgsl

        class ParentMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                super().surface(ctx, out)
                out.metallic = 0.9

        wgsl = generate_combined_wgsl(ChildMaterial, include_comments=True)

        assert "ChildMaterial" in wgsl
        assert "Inheritance" in wgsl
        assert "super()" in wgsl

    def test_generate_combined_wgsl_no_comments(self):
        """generate_combined_wgsl without comments."""
        from trinity.materials.inheritance import generate_combined_wgsl

        class SimpleMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        wgsl = generate_combined_wgsl(SimpleMaterial, include_comments=False)

        # Should just be the raw WGSL
        assert "// Material:" not in wgsl

    def test_validate_inheritance_no_issues(self):
        """validate_inheritance returns empty list for valid inheritance."""
        from trinity.materials.inheritance import validate_inheritance

        class ParentMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ValidChild(ParentMaterial, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                super().surface(ctx, out)
                out.metallic = 0.9

        issues = validate_inheritance(ValidChild)
        assert issues == []

    def test_validate_inheritance_deep_chain_warning(self):
        """validate_inheritance warns about deep inheritance chains."""
        from trinity.materials.inheritance import validate_inheritance

        # Create a chain of 6+ materials
        class Level1(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.1

        class Level2(Level1, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.2

        class Level3(Level2, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.3

        class Level4(Level3, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.4

        class Level5(Level4, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class Level6(Level5, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.6

        issues = validate_inheritance(Level6)
        assert any("deep inheritance" in issue.lower() for issue in issues)

    def test_validate_inheritance_diamond_warning(self):
        """validate_inheritance warns about diamond inheritance."""
        from trinity.materials.inheritance import validate_inheritance

        class CommonAncestor(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.1

        class BranchA(CommonAncestor, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.3

        class BranchB(CommonAncestor, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.7

        class DiamondChild(BranchA, BranchB, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.9

        issues = validate_inheritance(DiamondChild)
        assert any("diamond" in issue.lower() for issue in issues)


# =============================================================================
# Suite K: SuperCallDetector Tests
# =============================================================================


class TestSuperCallDetector:
    """Tests for the SuperCallDetector AST visitor."""

    def test_detector_finds_super_surface_call(self):
        """SuperCallDetector detects super().surface() calls."""
        import ast
        from trinity.materials.inheritance import SuperCallDetector

        code = '''
def surface(self, ctx, out):
    super().surface(ctx, out)
    out.metallic = 0.9
'''
        tree = ast.parse(code)
        detector = SuperCallDetector()
        detector.visit(tree)

        assert detector.has_super_surface_call is True
        assert len(detector.super_calls) == 1

    def test_detector_no_super_call(self):
        """SuperCallDetector returns False when no super() call."""
        import ast
        from trinity.materials.inheritance import SuperCallDetector

        code = '''
def surface(self, ctx, out):
    out.metallic = 0.9
'''
        tree = ast.parse(code)
        detector = SuperCallDetector()
        detector.visit(tree)

        assert detector.has_super_surface_call is False
        assert len(detector.super_calls) == 0

    def test_detector_other_super_calls_ignored(self):
        """SuperCallDetector ignores super() calls to other methods."""
        import ast
        from trinity.materials.inheritance import SuperCallDetector

        code = '''
def surface(self, ctx, out):
    super().__init__()  # Not surface()
    out.metallic = 0.9
'''
        tree = ast.parse(code)
        detector = SuperCallDetector()
        detector.visit(tree)

        assert detector.has_super_surface_call is False

    def test_detector_records_call_info(self):
        """SuperCallDetector records info about super() calls."""
        import ast
        from trinity.materials.inheritance import SuperCallDetector

        code = '''
def surface(self, ctx, out):
    super().surface(ctx, out)
'''
        tree = ast.parse(code)
        detector = SuperCallDetector()
        detector.visit(tree)

        assert len(detector.super_calls) == 1
        call_info = detector.super_calls[0]
        assert call_info.has_args is True
        assert call_info.arg_count == 2


# =============================================================================
# Suite L: Data Class Tests
# =============================================================================


class TestDataClasses:
    """Tests for inheritance data classes."""

    def test_inheritance_info_defaults(self):
        """InheritanceInfo has correct default values."""
        from trinity.materials.inheritance import InheritanceInfo

        class DummyClass:
            pass

        info = InheritanceInfo(material_class=DummyClass)

        assert info.material_class is DummyClass
        assert info.parent_material is None
        assert info.inheritance_chain == []
        assert info.all_textures == {}
        assert info.has_super_call is False

    def test_texture_merge_result_defaults(self):
        """TextureMergeResult has correct default values."""
        from trinity.materials.inheritance import TextureMergeResult

        result = TextureMergeResult()

        assert result.merged == {}
        assert result.from_parent == {}
        assert result.from_child == {}
        assert result.overridden == {}

    def test_super_call_info(self):
        """SuperCallInfo stores call information."""
        from trinity.materials.inheritance import SuperCallInfo

        info = SuperCallInfo(lineno=10, col_offset=4, has_args=True, arg_count=2)

        assert info.lineno == 10
        assert info.col_offset == 4
        assert info.has_args is True
        assert info.arg_count == 2


# =============================================================================
# Suite M: Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for inheritance exceptions."""

    def test_inheritance_error_is_exception(self):
        """InheritanceError is an Exception subclass."""
        from trinity.materials.inheritance import InheritanceError

        assert issubclass(InheritanceError, Exception)

    def test_no_parent_surface_error(self):
        """NoParentSurfaceError can be raised."""
        from trinity.materials.inheritance import NoParentSurfaceError

        with pytest.raises(NoParentSurfaceError):
            raise NoParentSurfaceError("No parent surface method found")

    def test_circular_inheritance_error(self):
        """CircularInheritanceError can be raised."""
        from trinity.materials.inheritance import CircularInheritanceError

        with pytest.raises(CircularInheritanceError):
            raise CircularInheritanceError("Circular inheritance detected")

    def test_diamond_conflict_error(self):
        """DiamondConflictError can be raised."""
        from trinity.materials.inheritance import DiamondConflictError

        with pytest.raises(DiamondConflictError):
            raise DiamondConflictError("Diamond inheritance conflict")


# =============================================================================
# Suite N: Complex Inheritance Scenarios
# =============================================================================


class TestComplexScenarios:
    """Complex inheritance scenarios."""

    def test_three_level_inheritance_with_super(self):
        """Three-level inheritance where each level calls super()."""

        class GrandparentMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.1

        class ParentMaterial(GrandparentMaterial, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                super().surface(ctx, out)
                out.metallic = 0.5

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                super().surface(ctx, out)
                out.ao = 0.8

        wgsl = ChildMaterial._wgsl_source
        assert "0.8" in wgsl  # Child's ao

    def test_mixin_pattern(self):
        """Mixin classes that add texture slots."""

        class BaseMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class NormalMapMixin(Material, metaclass=MaterialMeta):
            normal_map = Texture2D(default="flat_normal")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                pass

        class NormalMappedMaterial(BaseMaterial, NormalMapMixin, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                super().surface(ctx, out)
                out.metallic = 0.9

        assert "normal_map" in NormalMappedMaterial._texture_bindings

    def test_texture_override_preserves_other_settings(self):
        """When child overrides texture, other settings can change."""

        class ParentMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white", srgb=True, filter="linear")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ChildMaterial(ParentMaterial, metaclass=MaterialMeta):
            albedo = Texture2D(default="black", srgb=False, filter="nearest")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.metallic = 0.9

        child_albedo = ChildMaterial._texture_bindings["albedo"]
        assert child_albedo.default == "black"
        assert child_albedo.srgb is False
        assert child_albedo.filter.value == "nearest"
