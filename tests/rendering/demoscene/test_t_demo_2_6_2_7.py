"""
Integration tests for T-DEMO-2.6 (Material Codegen) and T-DEMO-2.7 (Scene Codegen).

Tests the complete WGSL code generation pipeline for demoscene ray marching,
including:
  - Material struct generation with PBR properties
  - scene_material() function with switch/case
  - Complete compute shader generation
  - Ray marching infrastructure (ray generation, marching, normals)
  - PBR lighting calculations
  - Integration of scene_sdf() with material propagation

These tests verify that the generated WGSL is syntactically valid and
contains all required components for wgpu compilation.
"""

from __future__ import annotations

import re

import pytest

from engine.rendering.demoscene.ast_nodes import (
    Axis, BendNode, BoxNode, CameraNode, CapsuleNode, ConeNode, CylinderNode,
    EllipsoidNode, FloatNode, FullSceneNode, KifsNode, LightNode, LightType,
    MaterialNode, MirrorNode, OctahedronNode, PlaneNode, PositionNode,
    PyramidNode, RenderSettingsNode, RepeatNode, RoundedBoxNode, SceneGraph,
    SphereNode, StretchNode, TorusNode, TwistNode, Vec3Node,
)
from engine.rendering.demoscene.material_codegen import (
    MaterialCodegen, generate_material_wgsl, MATERIAL_STRUCT,
    DEFAULT_MATERIAL, PROCEDURAL_PATTERNS, HEIGHT_PALETTE,
)
from engine.rendering.demoscene.scene_codegen import (
    SceneCodegen, generate_scene_wgsl, generate_compute_shader,
    COMPUTE_SHADER_HEADER, BIND_GROUP_0, BIND_GROUP_1_UNIFORMS,
    RAY_GENERATION, RAY_MARCHING, NORMAL_ESTIMATION,
    PBR_LIGHTING, TONE_MAPPING,
)


# =============================================================================
# Test Helpers
# =============================================================================


def _balanced_parens(s: str) -> bool:
    """Check balanced parentheses."""
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def _balanced_braces(s: str) -> bool:
    """Check balanced curly braces."""
    depth = 0
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def _extract_fns(wgsl: str) -> list[str]:
    """Extract all WGSL function names."""
    fns = []
    for line in wgsl.splitlines():
        m = re.match(r"fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", line)
        if m:
            fns.append(m.group(1))
    return fns


def _extract_structs(wgsl: str) -> list[str]:
    """Extract all WGSL struct names."""
    structs = []
    for line in wgsl.splitlines():
        m = re.match(r"struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\{", line)
        if m:
            structs.append(m.group(1))
    return structs


def _simple_scene() -> FullSceneNode:
    """Create a simple test scene with one sphere."""
    graph = SceneGraph(
        primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        name="simple",
    )
    return FullSceneNode(
        scene_graph=graph,
        materials=(
            MaterialNode(
                material_id=0,
                albedo=Vec3Node(0.8, 0.2, 0.2),
            ),
        ),
        name="simple_scene",
    )


def _complex_scene() -> FullSceneNode:
    """Create a complex test scene with multiple primitives and materials."""
    graph = SceneGraph(
        primitives=(
            SphereNode(PositionNode(), FloatNode(1.0)),
            BoxNode(PositionNode(), Vec3Node(0.5, 0.5, 0.5)),
            TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
        ),
        pipeline=(
            TwistNode(PositionNode(), FloatNode(2.0)),
            MirrorNode(PositionNode(), Axis.X),
        ),
        name="complex",
    )
    return FullSceneNode(
        scene_graph=graph,
        materials=(
            MaterialNode(material_id=0, albedo=Vec3Node(0.8, 0.2, 0.2)),
            MaterialNode(material_id=1, albedo=Vec3Node(0.2, 0.8, 0.2)),
            MaterialNode(material_id=2, albedo=Vec3Node(0.2, 0.2, 0.8)),
        ),
        lights=(
            LightNode(
                position=Vec3Node(5.0, 5.0, 5.0),
                color=Vec3Node(1.0, 1.0, 1.0),
                intensity=FloatNode(2.0),
                light_type=LightType.POINT,
            ),
            LightNode(
                position=Vec3Node(0.0, 0.0, 0.0),
                color=Vec3Node(0.8, 0.9, 1.0),
                intensity=FloatNode(1.0),
                light_type=LightType.DIRECTIONAL,
                direction=Vec3Node(0.0, -1.0, 0.5),
            ),
        ),
        name="complex_scene",
    )


# =============================================================================
# T-DEMO-2.6: Material Codegen Tests
# =============================================================================


class TestMaterialNodeCreation:
    """Test MaterialNode AST creation."""

    def test_material_with_defaults(self) -> None:
        mat = MaterialNode(material_id=0, albedo=Vec3Node(0.5, 0.5, 0.5))
        assert mat.material_id == 0
        assert mat.roughness.value == 0.5
        assert mat.metallic.value == 0.0
        assert mat.ambient_occlusion.value == 1.0

    def test_material_with_custom_values(self) -> None:
        mat = MaterialNode(
            material_id=1,
            albedo=Vec3Node(0.8, 0.2, 0.2),
            roughness=FloatNode(0.3),
            metallic=FloatNode(0.9),
            emission=Vec3Node(0.1, 0.0, 0.0),
            ambient_occlusion=FloatNode(0.7),
        )
        assert mat.material_id == 1
        assert mat.roughness.value == 0.3
        assert mat.metallic.value == 0.9
        assert mat.emission.x == 0.1
        assert mat.ambient_occlusion.value == 0.7

    def test_material_label(self) -> None:
        mat = MaterialNode(
            material_id=0,
            albedo=Vec3Node(0.5, 0.5, 0.5),
            roughness=FloatNode(0.3),
            metallic=FloatNode(0.9),
        )
        label = mat.label()
        assert "Material" in label
        assert "id=0" in label
        assert "roughness=0.3" in label
        assert "metallic=0.9" in label


class TestMaterialStructGeneration:
    """Test MATERIAL_STRUCT constant."""

    def test_struct_declaration(self) -> None:
        assert "struct Material" in MATERIAL_STRUCT

    def test_all_pbr_fields_present(self) -> None:
        assert "albedo: vec3<f32>" in MATERIAL_STRUCT
        assert "roughness: f32" in MATERIAL_STRUCT
        assert "metallic: f32" in MATERIAL_STRUCT
        assert "emission: vec3<f32>" in MATERIAL_STRUCT
        assert "ambient_occlusion: f32" in MATERIAL_STRUCT

    def test_balanced_syntax(self) -> None:
        assert _balanced_braces(MATERIAL_STRUCT)
        assert _balanced_parens(MATERIAL_STRUCT)


class TestMaterialCodegenSingleMaterial:
    """Test MaterialCodegen with single material."""

    def test_single_material_no_switch(self) -> None:
        gen = MaterialCodegen()
        mat = MaterialNode(material_id=0, albedo=Vec3Node(0.8, 0.2, 0.2))
        wgsl = gen.generate([mat])
        assert "fn scene_material" in wgsl
        assert "switch" not in wgsl

    def test_single_material_values_present(self) -> None:
        gen = MaterialCodegen()
        mat = MaterialNode(
            material_id=0,
            albedo=Vec3Node(0.5, 0.6, 0.7),
            roughness=FloatNode(0.3),
        )
        wgsl = gen.generate([mat])
        assert "vec3<f32>(0.5, 0.6, 0.7)" in wgsl
        assert "0.3" in wgsl


class TestMaterialCodegenMultipleMaterials:
    """Test MaterialCodegen with multiple materials."""

    def test_two_materials_uses_switch(self) -> None:
        gen = MaterialCodegen()
        mat0 = MaterialNode(material_id=0, albedo=Vec3Node(0.8, 0.2, 0.2))
        mat1 = MaterialNode(material_id=1, albedo=Vec3Node(0.2, 0.8, 0.2))
        wgsl = gen.generate([mat0, mat1])
        assert "switch material_id" in wgsl
        assert "case 0u:" in wgsl
        assert "case 1u:" in wgsl
        assert "default:" in wgsl

    def test_three_materials_all_cases(self) -> None:
        gen = MaterialCodegen()
        mats = [
            MaterialNode(material_id=0, albedo=Vec3Node(0.8, 0.2, 0.2)),
            MaterialNode(material_id=1, albedo=Vec3Node(0.2, 0.8, 0.2)),
            MaterialNode(material_id=2, albedo=Vec3Node(0.2, 0.2, 0.8)),
        ]
        wgsl = gen.generate(mats)
        assert "case 0u:" in wgsl
        assert "case 1u:" in wgsl
        assert "case 2u:" in wgsl

    def test_each_material_albedo_distinct(self) -> None:
        gen = MaterialCodegen()
        mat0 = MaterialNode(material_id=0, albedo=Vec3Node(0.9, 0.1, 0.1))
        mat1 = MaterialNode(material_id=1, albedo=Vec3Node(0.1, 0.9, 0.1))
        wgsl = gen.generate([mat0, mat1])
        assert "vec3<f32>(0.9, 0.1, 0.1)" in wgsl
        assert "vec3<f32>(0.1, 0.9, 0.1)" in wgsl


class TestMaterialCodegenProceduralPatterns:
    """Test procedural pattern generation."""

    def test_include_patterns_flag(self) -> None:
        gen = MaterialCodegen()
        wgsl = gen.generate([], include_patterns=True)
        assert "fn pattern_checker" in wgsl
        assert "fn pattern_stripe" in wgsl
        assert "fn pattern_height_gradient" in wgsl
        assert "fn pattern_radial" in wgsl
        assert "fn material_blend" in wgsl

    def test_height_palette_flag(self) -> None:
        gen = MaterialCodegen()
        wgsl = gen.generate([], include_height_palette=True)
        assert "fn height_palette" in wgsl
        assert "-> Material" in wgsl


class TestMaterialCodegenSyntax:
    """Test WGSL syntax validity."""

    def test_balanced_braces(self) -> None:
        gen = MaterialCodegen()
        mats = [
            MaterialNode(material_id=0, albedo=Vec3Node(0.8, 0.2, 0.2)),
            MaterialNode(material_id=1, albedo=Vec3Node(0.2, 0.8, 0.2)),
        ]
        wgsl = gen.generate(mats)
        assert _balanced_braces(wgsl)

    def test_balanced_parens(self) -> None:
        gen = MaterialCodegen()
        mat = MaterialNode(material_id=0, albedo=Vec3Node(0.5, 0.5, 0.5))
        wgsl = gen.generate([mat])
        assert _balanced_parens(wgsl)

    def test_no_trailing_whitespace(self) -> None:
        gen = MaterialCodegen()
        wgsl = gen.generate([])
        for line in wgsl.split("\n"):
            assert line == line.rstrip()


# =============================================================================
# T-DEMO-2.7: Scene Codegen Tests
# =============================================================================


class TestSceneCodegenBasics:
    """Test basic SceneCodegen functionality."""

    def test_generates_string(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert isinstance(wgsl, str)
        assert len(wgsl) > 0

    def test_includes_header(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "SPDX-License-Identifier: MIT" in wgsl

    def test_includes_scene_name(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "simple_scene" in wgsl


class TestSceneCodegenStructs:
    """Test struct generation in compute shader."""

    def test_material_struct_present(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "struct Material" in wgsl

    def test_uniforms_struct_present(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "struct Uniforms" in wgsl

    def test_rayhit_struct_present(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "struct RayHit" in wgsl


class TestSceneCodegenFunctions:
    """Test required function generation."""

    def test_scene_sdf_present(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "fn scene_sdf" in wgsl

    def test_scene_material_present(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "fn scene_material" in wgsl

    def test_generate_ray_present(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "fn generate_ray" in wgsl

    def test_ray_march_present(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "fn ray_march" in wgsl

    def test_estimate_normal_present(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "fn estimate_normal" in wgsl

    def test_calculate_lighting_present(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "fn calculate_lighting" in wgsl

    def test_tone_map_present(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "fn tone_map_aces" in wgsl

    def test_main_compute_present(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "fn main" in wgsl
        assert "@compute" in wgsl


class TestSceneCodegenPrimitives:
    """Test SDF primitive function generation."""

    def test_sphere_function(self) -> None:
        graph = SceneGraph(primitives=(SphereNode(PositionNode(), FloatNode(1.0)),))
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)
        assert "fn sdf_sphere" in wgsl
        assert "length(p) - r" in wgsl

    def test_box_function(self) -> None:
        graph = SceneGraph(primitives=(BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),))
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)
        assert "fn sdf_box" in wgsl

    def test_torus_function(self) -> None:
        graph = SceneGraph(primitives=(TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),))
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)
        assert "fn sdf_torus" in wgsl

    def test_cylinder_function(self) -> None:
        graph = SceneGraph(primitives=(CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),))
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)
        assert "fn sdf_cylinder" in wgsl

    def test_cone_function(self) -> None:
        graph = SceneGraph(primitives=(ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),))
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)
        assert "fn sdf_cone" in wgsl

    def test_plane_function(self) -> None:
        graph = SceneGraph(primitives=(PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),))
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)
        assert "fn sdf_plane" in wgsl

    def test_capsule_function(self) -> None:
        graph = SceneGraph(primitives=(CapsuleNode(
            PositionNode(),
            Vec3Node(0.0, -1.0, 0.0),
            Vec3Node(0.0, 1.0, 0.0),
            FloatNode(0.5)
        ),))
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)
        assert "fn sdf_capsule" in wgsl


class TestSceneCodegenDomainOps:
    """Test domain operation code generation."""

    def test_repeat_op(self) -> None:
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),),
        )
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)
        assert "fn domain_repeat" in wgsl
        assert "domain_repeat(p," in wgsl

    def test_mirror_op(self) -> None:
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(MirrorNode(PositionNode(), Axis.X),),
        )
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)
        assert "fn domain_mirror_x" in wgsl

    def test_twist_op(self) -> None:
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(2.0)),),
        )
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)
        assert "fn domain_twist" in wgsl

    def test_kifs_with_compensation(self) -> None:
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(KifsNode(PositionNode(), FloatNode(6.0)),),
        )
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)
        assert "fn domain_kifs" in wgsl
        assert "fn domain_kifs_compensation" in wgsl
        assert "domain_kifs_compensation(6.0)" in wgsl

    def test_stretch_with_compensation(self) -> None:
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(StretchNode(PositionNode(), FloatNode(2.0), Axis.X),),
        )
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)
        assert "fn domain_stretch_x" in wgsl
        assert "fn domain_stretch_compensation" in wgsl


class TestSceneCodegenMaterialPropagation:
    """Test material ID propagation in SDF."""

    def test_single_primitive_material_id(self) -> None:
        graph = SceneGraph(primitives=(SphereNode(PositionNode(), FloatNode(1.0)),))
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)
        # Should return vec2 with material_id
        assert "vec2<f32>(sdf_sphere" in wgsl
        assert ", 0.0)" in wgsl

    def test_multiple_primitives_select(self) -> None:
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(0.5, 0.5, 0.5)),
            ),
        )
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)
        # Should use select for material propagation
        assert "select(" in wgsl
        assert "d0" in wgsl
        assert "d1" in wgsl


class TestSceneCodegenLights:
    """Test light contribution generation."""

    def test_default_light_when_empty(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "fn calculate_light_contribution" in wgsl
        # Default light is directional
        assert "normalize(vec3<f32>(1.0, 1.0, 1.0))" in wgsl

    def test_point_light(self) -> None:
        graph = SceneGraph(primitives=(SphereNode(PositionNode(), FloatNode(1.0)),))
        scene = FullSceneNode(
            scene_graph=graph,
            lights=(
                LightNode(
                    position=Vec3Node(3.0, 3.0, 3.0),
                    color=Vec3Node(1.0, 0.9, 0.8),
                    intensity=FloatNode(2.0),
                    light_type=LightType.POINT,
                ),
            ),
        )
        wgsl = generate_compute_shader(scene)
        assert "Light 0: point" in wgsl
        assert "vec3<f32>(3.0, 3.0, 3.0)" in wgsl

    def test_directional_light(self) -> None:
        graph = SceneGraph(primitives=(SphereNode(PositionNode(), FloatNode(1.0)),))
        scene = FullSceneNode(
            scene_graph=graph,
            lights=(
                LightNode(
                    position=Vec3Node(0.0, 0.0, 0.0),
                    color=Vec3Node(1.0, 1.0, 1.0),
                    intensity=FloatNode(1.0),
                    light_type=LightType.DIRECTIONAL,
                    direction=Vec3Node(0.0, -1.0, 0.0),
                ),
            ),
        )
        wgsl = generate_compute_shader(scene)
        assert "Light 0: directional" in wgsl

    def test_multiple_lights(self) -> None:
        scene = _complex_scene()
        wgsl = generate_compute_shader(scene)
        assert "Light 0:" in wgsl
        assert "Light 1:" in wgsl


class TestSceneCodegenPBRLighting:
    """Test PBR lighting function generation."""

    def test_fresnel_schlick(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "fn fresnel_schlick" in wgsl

    def test_distribution_ggx(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "fn distribution_ggx" in wgsl

    def test_geometry_smith(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "fn geometry_smith" in wgsl


class TestSceneCodegenComputeEntry:
    """Test compute shader entry point."""

    def test_workgroup_size(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "@workgroup_size(8, 8, 1)" in wgsl

    def test_custom_workgroup_size(self) -> None:
        graph = SceneGraph(primitives=(SphereNode(PositionNode(), FloatNode(1.0)),))
        scene = FullSceneNode(
            scene_graph=graph,
            settings=RenderSettingsNode(workgroup_size_x=16, workgroup_size_y=16),
        )
        wgsl = generate_compute_shader(scene)
        assert "@workgroup_size(16, 16, 1)" in wgsl

    def test_global_invocation_id(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "@builtin(global_invocation_id)" in wgsl


class TestSceneCodegenBindings:
    """Test bind group and uniform generation."""

    def test_output_texture_binding(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "@group(0) @binding(0)" in wgsl
        assert "texture_storage_2d" in wgsl

    def test_uniforms_binding(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert "@group(1) @binding(0)" in wgsl
        assert "var<uniform> uniforms" in wgsl


class TestSceneCodegenSyntax:
    """Test WGSL syntax validity."""

    def test_simple_scene_balanced_braces(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert _balanced_braces(wgsl)

    def test_simple_scene_balanced_parens(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        assert _balanced_parens(wgsl)

    def test_complex_scene_balanced_braces(self) -> None:
        scene = _complex_scene()
        wgsl = generate_compute_shader(scene)
        assert _balanced_braces(wgsl)

    def test_complex_scene_balanced_parens(self) -> None:
        scene = _complex_scene()
        wgsl = generate_compute_shader(scene)
        assert _balanced_parens(wgsl)

    def test_no_trailing_whitespace(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        for line in wgsl.split("\n"):
            assert line == line.rstrip()


class TestSceneCodegenFunctionOrder:
    """Test that functions are generated in correct order."""

    def test_material_struct_before_scene_material(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        struct_idx = wgsl.index("struct Material")
        fn_idx = wgsl.index("fn scene_material")
        assert struct_idx < fn_idx

    def test_sdf_primitives_before_scene_sdf(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        sdf_sphere_idx = wgsl.index("fn sdf_sphere")
        scene_sdf_idx = wgsl.index("fn scene_sdf")
        assert sdf_sphere_idx < scene_sdf_idx

    def test_scene_sdf_before_ray_march(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        scene_sdf_idx = wgsl.index("fn scene_sdf")
        ray_march_idx = wgsl.index("fn ray_march")
        assert scene_sdf_idx < ray_march_idx

    def test_helpers_before_main(self) -> None:
        scene = _simple_scene()
        wgsl = generate_compute_shader(scene)
        main_idx = wgsl.index("fn main")
        assert wgsl.index("fn generate_ray") < main_idx
        assert wgsl.index("fn ray_march") < main_idx
        assert wgsl.index("fn estimate_normal") < main_idx
        assert wgsl.index("fn calculate_lighting") < main_idx


# =============================================================================
# Integration Tests
# =============================================================================


class TestFullSceneIntegration:
    """Integration tests for complete scene generation."""

    def test_all_primitives_scene(self) -> None:
        """Test scene with all primitive types."""
        graph = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(0.5, 0.5, 0.5)),
                TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.3)),
                CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
                PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(-1.0)),
            ),
        )
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)

        assert "fn sdf_sphere" in wgsl
        assert "fn sdf_box" in wgsl
        assert "fn sdf_torus" in wgsl
        assert "fn sdf_cylinder" in wgsl
        assert "fn sdf_plane" in wgsl
        assert _balanced_braces(wgsl)
        assert _balanced_parens(wgsl)

    def test_all_domain_ops_scene(self) -> None:
        """Test scene with all domain operation types."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
                MirrorNode(PositionNode(), Axis.X),
                KifsNode(PositionNode(), FloatNode(5.0)),
                TwistNode(PositionNode(), FloatNode(1.0)),
                BendNode(PositionNode(), FloatNode(5.0)),
                StretchNode(PositionNode(), FloatNode(2.0), Axis.Y),
            ),
        )
        scene = FullSceneNode(scene_graph=graph)
        wgsl = generate_compute_shader(scene)

        assert "fn domain_repeat" in wgsl
        assert "fn domain_mirror_x" in wgsl
        assert "fn domain_kifs" in wgsl
        assert "fn domain_twist" in wgsl
        assert "fn domain_bend" in wgsl
        assert "fn domain_stretch_y" in wgsl
        assert "fn domain_kifs_compensation" in wgsl
        assert "fn domain_stretch_compensation" in wgsl
        assert _balanced_braces(wgsl)

    def test_many_materials_scene(self) -> None:
        """Test scene with many materials."""
        graph = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
        )
        materials = tuple(
            MaterialNode(material_id=i, albedo=Vec3Node(i / 10.0, 0.5, 0.5))
            for i in range(10)
        )
        scene = FullSceneNode(scene_graph=graph, materials=materials)
        wgsl = generate_compute_shader(scene)

        assert "switch material_id" in wgsl
        for i in range(10):
            assert f"case {i}u:" in wgsl
        assert _balanced_braces(wgsl)

    def test_complex_scene_complete(self) -> None:
        """Test complete complex scene with all features."""
        scene = _complex_scene()
        wgsl = generate_compute_shader(scene)

        # Check all major components
        assert "struct Material" in wgsl
        assert "struct Uniforms" in wgsl
        assert "struct RayHit" in wgsl
        assert "fn scene_sdf" in wgsl
        assert "fn scene_material" in wgsl
        assert "fn generate_ray" in wgsl
        assert "fn ray_march" in wgsl
        assert "fn estimate_normal" in wgsl
        assert "fn calculate_lighting" in wgsl
        assert "fn main" in wgsl
        assert "@compute" in wgsl
        assert _balanced_braces(wgsl)
        assert _balanced_parens(wgsl)


class TestGenerateConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_generate_material_wgsl_returns_string(self) -> None:
        mat = MaterialNode(material_id=0, albedo=Vec3Node(0.5, 0.5, 0.5))
        wgsl = generate_material_wgsl([mat])
        assert isinstance(wgsl, str)
        assert "fn scene_material" in wgsl

    def test_generate_scene_wgsl_returns_string(self) -> None:
        scene = _simple_scene()
        wgsl = generate_scene_wgsl(scene)
        assert isinstance(wgsl, str)
        assert "@compute" in wgsl

    def test_generate_compute_shader_alias(self) -> None:
        scene = _simple_scene()
        wgsl1 = generate_scene_wgsl(scene)
        wgsl2 = generate_compute_shader(scene)
        assert wgsl1 == wgsl2
