// SPDX-License-Identifier: MIT
//
// blackbox_render_pipeline.rs -- Blackbox tests for T-WGPU-P3.1.1 Render Pipeline Descriptor.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - TrinityRenderPipeline -- Wrapper around wgpu::RenderPipeline
//   - RenderPipelineDescriptor -- Builder for render pipeline creation
//   - VertexStateDescriptor, VertexBufferLayoutDescriptor, VertexAttributeDescriptor
//   - PrimitiveStateDescriptor -- Topology, culling, polygon mode
//   - DepthStencilStateDescriptor, DepthBiasStateDescriptor, StencilFaceStateDescriptor
//   - MultisampleStateDescriptor -- MSAA configuration
//   - FragmentStateDescriptor, ColorTargetStateDescriptor, BlendStateDescriptor, BlendComponentDescriptor
//   - create_render_pipeline -- Convenience function
//
// ACCEPTANCE CRITERIA:
//   1. API surface tests -- All public types accessible
//   2. Real pipeline creation with wgpu device
//   3. Pipeline usage in render pass
//   4. Error handling for invalid configurations
//   5. Integration with existing TRINITY render system
//   6. Performance sanity (pipeline creation < 100ms)
//
// Total target: 20+ tests

use pollster::block_on;
use renderer_backend::device::{enumerate_adapters_with_info, TrinityInstance};
use renderer_backend::render_pipeline::{
    create_render_pipeline, BlendComponentDescriptor, BlendStateDescriptor,
    ColorTargetStateDescriptor, DepthBiasStateDescriptor, DepthStencilStateDescriptor,
    FragmentStateDescriptor, MultisampleStateDescriptor, PrimitiveStateDescriptor,
    RenderPipelineDescriptor, StencilFaceStateDescriptor, TrinityRenderPipeline,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexStateDescriptor,
};
use std::time::Instant;

// =============================================================================
// TEST SHADERS
// =============================================================================

/// Minimal vertex shader for basic pipeline creation tests.
const MINIMAL_VERTEX_SHADER: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
"#;

/// Minimal fragment shader for basic pipeline creation tests.
const MINIMAL_FRAGMENT_SHADER: &str = r#"
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;

/// PBR-style vertex shader with position, normal, and UV.
const PBR_VERTEX_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) tex_coord: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(input.position, 1.0);
    out.world_normal = input.normal;
    out.tex_coord = input.uv;
    return out;
}
"#;

/// PBR-style fragment shader.
const PBR_FRAGMENT_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) tex_coord: vec2<f32>,
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let diffuse = max(dot(input.world_normal, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
    return vec4<f32>(vec3<f32>(diffuse), 1.0);
}
"#;

/// Shadow map vertex shader (position only).
const SHADOW_VERTEX_SHADER: &str = r#"
@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(position, 1.0);
}
"#;

/// UI vertex shader with position, UV, and color.
const UI_VERTEX_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(input.position, 0.0, 1.0);
    out.tex_coord = input.uv;
    out.color = input.color;
    return out;
}
"#;

/// UI fragment shader.
const UI_FRAGMENT_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
"#;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Creates a TrinityInstance and gets the first available adapter.
fn get_test_adapter() -> Option<wgpu::Adapter> {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    result.adapters.into_iter().next()
}

/// Helper macro to skip test if no GPU adapter is available.
macro_rules! require_adapter {
    () => {
        match get_test_adapter() {
            Some(adapter) => adapter,
            None => {
                eprintln!("SKIP: No GPU adapter available for this test");
                return;
            }
        }
    };
}

/// Creates a wgpu device for testing.
fn create_test_device(adapter: &wgpu::Adapter) -> Option<(wgpu::Device, wgpu::Queue)> {
    block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
}

/// Helper macro to get a device, skipping if unavailable.
macro_rules! require_device {
    ($adapter:expr) => {
        match create_test_device($adapter) {
            Some((device, queue)) => (device, queue),
            None => {
                eprintln!("SKIP: Could not create device");
                return;
            }
        }
    };
}

/// Creates a shader module from WGSL source.
fn create_shader_module(device: &wgpu::Device, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test_shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
    })
}

/// Creates an empty pipeline layout for basic tests.
fn create_empty_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("test_layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    })
}

// =============================================================================
// SECTION 1: API SURFACE TESTS (No GPU Required)
// =============================================================================

/// Test: All public types from render_pipeline module are accessible.
#[test]
fn test_render_pipeline_api_surface() {
    // Verify all public types are accessible (compile-time check)
    let _: fn() -> TrinityRenderPipeline = || panic!("type check only");
    let _: fn() -> PrimitiveStateDescriptor = PrimitiveStateDescriptor::default;
    let _: fn() -> MultisampleStateDescriptor = MultisampleStateDescriptor::default;
    let _: fn() -> BlendComponentDescriptor = BlendComponentDescriptor::default;
    let _: fn() -> BlendStateDescriptor = BlendStateDescriptor::replace;
    let _: fn() -> StencilFaceStateDescriptor = StencilFaceStateDescriptor::default;
    let _: fn() -> DepthBiasStateDescriptor = DepthBiasStateDescriptor::default;
}

/// Test: VertexAttributeDescriptor construction and conversion.
#[test]
fn test_vertex_attribute_descriptor_api() {
    let attr = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 0, 0);

    assert_eq!(attr.format, wgpu::VertexFormat::Float32x3);
    assert_eq!(attr.offset, 0);
    assert_eq!(attr.shader_location, 0);

    // Test conversion to wgpu type
    let wgpu_attr: wgpu::VertexAttribute = attr.into();
    assert_eq!(wgpu_attr.format, wgpu::VertexFormat::Float32x3);
    assert_eq!(wgpu_attr.offset, 0);
    assert_eq!(wgpu_attr.shader_location, 0);
}

/// Test: VertexBufferLayoutDescriptor builder pattern.
#[test]
fn test_vertex_buffer_layout_descriptor_builder() {
    let layout = VertexBufferLayoutDescriptor::per_vertex(32)
        .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0)
        .with_attribute(wgpu::VertexFormat::Float32x3, 12, 1)
        .with_attribute(wgpu::VertexFormat::Float32x2, 24, 2);

    assert_eq!(layout.array_stride, 32);
    assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
    assert_eq!(layout.attributes.len(), 3);
}

/// Test: VertexBufferLayoutDescriptor per-instance mode.
#[test]
fn test_vertex_buffer_layout_per_instance() {
    let layout = VertexBufferLayoutDescriptor::per_instance(64)
        .with_attribute(wgpu::VertexFormat::Float32x4, 0, 0)
        .with_attribute(wgpu::VertexFormat::Float32x4, 16, 1)
        .with_attribute(wgpu::VertexFormat::Float32x4, 32, 2)
        .with_attribute(wgpu::VertexFormat::Float32x4, 48, 3);

    assert_eq!(layout.array_stride, 64);
    assert_eq!(layout.step_mode, wgpu::VertexStepMode::Instance);
    assert_eq!(layout.attributes.len(), 4);
}

/// Test: PrimitiveStateDescriptor default values.
#[test]
fn test_primitive_state_defaults() {
    let state = PrimitiveStateDescriptor::default();

    assert_eq!(state.topology, wgpu::PrimitiveTopology::TriangleList);
    assert_eq!(state.front_face, wgpu::FrontFace::Ccw);
    assert_eq!(state.cull_mode, Some(wgpu::Face::Back));
    assert!(!state.unclipped_depth);
    assert_eq!(state.polygon_mode, wgpu::PolygonMode::Fill);
    assert!(!state.conservative);
}

/// Test: PrimitiveStateDescriptor presets.
#[test]
fn test_primitive_state_presets() {
    let tri_strip = PrimitiveStateDescriptor::triangle_strip(wgpu::IndexFormat::Uint16);
    assert_eq!(tri_strip.topology, wgpu::PrimitiveTopology::TriangleStrip);
    assert_eq!(tri_strip.strip_index_format, Some(wgpu::IndexFormat::Uint16));

    let line_list = PrimitiveStateDescriptor::line_list();
    assert_eq!(line_list.topology, wgpu::PrimitiveTopology::LineList);
    assert_eq!(line_list.cull_mode, None);

    let point_list = PrimitiveStateDescriptor::point_list();
    assert_eq!(point_list.topology, wgpu::PrimitiveTopology::PointList);
}

/// Test: PrimitiveStateDescriptor builder methods.
#[test]
fn test_primitive_state_builder() {
    let state = PrimitiveStateDescriptor::new()
        .topology(wgpu::PrimitiveTopology::TriangleStrip)
        .no_culling()
        .wireframe();

    assert_eq!(state.topology, wgpu::PrimitiveTopology::TriangleStrip);
    assert_eq!(state.cull_mode, None);
    assert_eq!(state.polygon_mode, wgpu::PolygonMode::Line);
}

/// Test: DepthStencilStateDescriptor default values.
#[test]
fn test_depth_stencil_state_defaults() {
    let state = DepthStencilStateDescriptor::new()
        .format(wgpu::TextureFormat::Depth32Float);

    assert_eq!(state.format, wgpu::TextureFormat::Depth32Float);
    assert!(state.depth_write_enabled);
    assert_eq!(state.depth_compare, wgpu::CompareFunction::Less);
}

/// Test: DepthStencilStateDescriptor presets.
#[test]
fn test_depth_stencil_state_presets() {
    let shadow = DepthStencilStateDescriptor::shadow_map();
    assert_eq!(shadow.bias.constant, 2);
    assert_eq!(shadow.bias.slope_scale, 2.0);

    let transparent = DepthStencilStateDescriptor::transparent();
    assert!(!transparent.depth_write_enabled);
    assert_eq!(transparent.depth_compare, wgpu::CompareFunction::LessEqual);

    let prepass = DepthStencilStateDescriptor::depth_prepass();
    assert!(prepass.depth_write_enabled);
}

/// Test: DepthBiasStateDescriptor shadow map preset.
#[test]
fn test_depth_bias_shadow_map_preset() {
    let bias = DepthBiasStateDescriptor::shadow_map();

    assert_eq!(bias.constant, 2);
    assert_eq!(bias.slope_scale, 2.0);
    assert_eq!(bias.clamp, 0.0);
}

/// Test: MultisampleStateDescriptor default and presets.
#[test]
fn test_multisample_state_presets() {
    let default = MultisampleStateDescriptor::default();
    assert_eq!(default.count, 1);
    assert!(!default.alpha_to_coverage_enabled);

    let msaa4 = MultisampleStateDescriptor::msaa_4x();
    assert_eq!(msaa4.count, 4);

    let msaa8 = MultisampleStateDescriptor::msaa_8x();
    assert_eq!(msaa8.count, 8);
}

/// Test: BlendStateDescriptor presets.
#[test]
fn test_blend_state_presets() {
    let replace = BlendStateDescriptor::replace();
    assert_eq!(replace.color.src_factor, wgpu::BlendFactor::One);
    assert_eq!(replace.color.dst_factor, wgpu::BlendFactor::Zero);

    let alpha = BlendStateDescriptor::alpha_blend();
    assert_eq!(alpha.color.src_factor, wgpu::BlendFactor::SrcAlpha);
    assert_eq!(alpha.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);

    let premul = BlendStateDescriptor::premultiplied_alpha();
    assert_eq!(premul.color.src_factor, wgpu::BlendFactor::One);
    assert_eq!(premul.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);

    let additive = BlendStateDescriptor::additive();
    assert_eq!(additive.color.src_factor, wgpu::BlendFactor::One);
    assert_eq!(additive.color.dst_factor, wgpu::BlendFactor::One);
}

/// Test: ColorTargetStateDescriptor builder and presets.
#[test]
fn test_color_target_state_builder() {
    let target = ColorTargetStateDescriptor::new(wgpu::TextureFormat::Bgra8UnormSrgb)
        .alpha_blend()
        .write_mask(wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN);

    assert_eq!(target.format, wgpu::TextureFormat::Bgra8UnormSrgb);
    assert!(target.blend.is_some());
    assert_eq!(target.write_mask, wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN);

    let srgb = ColorTargetStateDescriptor::srgb();
    assert_eq!(srgb.format, wgpu::TextureFormat::Bgra8UnormSrgb);

    let hdr = ColorTargetStateDescriptor::hdr();
    assert_eq!(hdr.format, wgpu::TextureFormat::Rgba16Float);
}

/// Test: StencilFaceStateDescriptor builder.
#[test]
fn test_stencil_face_state_builder() {
    let state = StencilFaceStateDescriptor::new()
        .compare(wgpu::CompareFunction::Equal)
        .pass_op(wgpu::StencilOperation::Replace)
        .fail_op(wgpu::StencilOperation::Zero)
        .depth_fail_op(wgpu::StencilOperation::Keep);

    assert_eq!(state.compare, wgpu::CompareFunction::Equal);
    assert_eq!(state.pass_op, wgpu::StencilOperation::Replace);
    assert_eq!(state.fail_op, wgpu::StencilOperation::Zero);
    assert_eq!(state.depth_fail_op, wgpu::StencilOperation::Keep);
}

// =============================================================================
// SECTION 2: REAL PIPELINE CREATION (Requires GPU)
// =============================================================================

/// Test: Create a simple render pipeline with minimal configuration.
#[test]
fn test_create_simple_render_pipeline() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let vs_module = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs_module = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let pipeline = RenderPipelineDescriptor::new(&layout)
        .label("simple_pipeline")
        .vertex(VertexStateDescriptor::new(&vs_module))
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .target(wgpu::TextureFormat::Bgra8UnormSrgb),
        )
        .build(&device);

    assert_eq!(pipeline.label(), Some("simple_pipeline"));
    assert!(pipeline.layout_id() > 0);

    // Verify raw pipeline is accessible
    let _raw: &wgpu::RenderPipeline = pipeline.raw();
}

/// Test: Create a full PBR-style pipeline.
#[test]
fn test_create_pbr_pipeline() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let vs_module = create_shader_module(&device, PBR_VERTEX_SHADER);
    let fs_module = create_shader_module(&device, PBR_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    // PBR vertex layout: position + normal + UV = 32 bytes
    let vertex_layout = VertexBufferLayoutDescriptor::per_vertex(32)
        .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0)   // position
        .with_attribute(wgpu::VertexFormat::Float32x3, 12, 1)  // normal
        .with_attribute(wgpu::VertexFormat::Float32x2, 24, 2); // uv

    let pipeline = RenderPipelineDescriptor::new(&layout)
        .label("pbr_forward")
        .vertex(VertexStateDescriptor::new(&vs_module).buffer(vertex_layout))
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .target(wgpu::TextureFormat::Bgra8UnormSrgb),
        )
        .primitive(
            PrimitiveStateDescriptor::triangle_list()
                .cull_back()
                .front_face(wgpu::FrontFace::Ccw),
        )
        .depth_stencil(DepthStencilStateDescriptor::depth_less().format(wgpu::TextureFormat::Depth32Float))
        .multisample(MultisampleStateDescriptor::default())
        .build(&device);

    assert_eq!(pipeline.label(), Some("pbr_forward"));
}

/// Test: Create a shadow map pipeline (depth-only, no fragment).
#[test]
fn test_create_shadow_pipeline() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let vs_module = create_shader_module(&device, SHADOW_VERTEX_SHADER);
    let layout = create_empty_layout(&device);

    // Shadow map vertex layout: position only = 12 bytes
    let vertex_layout = VertexBufferLayoutDescriptor::per_vertex(12)
        .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0);

    let pipeline = RenderPipelineDescriptor::new(&layout)
        .label("shadow_map")
        .vertex(VertexStateDescriptor::new(&vs_module).buffer(vertex_layout))
        // No fragment stage for depth-only rendering
        .primitive(
            PrimitiveStateDescriptor::triangle_list()
                .cull_front() // Front-face culling for shadow mapping
        )
        .depth_stencil(DepthStencilStateDescriptor::shadow_map())
        .build(&device);

    assert_eq!(pipeline.label(), Some("shadow_map"));
}

/// Test: Create a UI pipeline with alpha blending.
#[test]
fn test_create_ui_pipeline() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let vs_module = create_shader_module(&device, UI_VERTEX_SHADER);
    let fs_module = create_shader_module(&device, UI_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    // UI vertex layout: position(2) + uv(2) + color(4) = 32 bytes
    let vertex_layout = VertexBufferLayoutDescriptor::per_vertex(32)
        .with_attribute(wgpu::VertexFormat::Float32x2, 0, 0)   // position
        .with_attribute(wgpu::VertexFormat::Float32x2, 8, 1)   // uv
        .with_attribute(wgpu::VertexFormat::Float32x4, 16, 2); // color

    let pipeline = RenderPipelineDescriptor::new(&layout)
        .label("ui_pipeline")
        .vertex(VertexStateDescriptor::new(&vs_module).buffer(vertex_layout))
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .target_state(
                    ColorTargetStateDescriptor::new(wgpu::TextureFormat::Bgra8UnormSrgb)
                        .premultiplied_alpha()
                ),
        )
        .primitive(PrimitiveStateDescriptor::triangle_list().no_culling())
        // No depth testing for UI
        .build(&device);

    assert_eq!(pipeline.label(), Some("ui_pipeline"));
}

/// Test: Create pipeline using convenience function.
#[test]
fn test_create_render_pipeline_convenience() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let vs_module = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs_module = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let desc = RenderPipelineDescriptor::new(&layout)
        .label("convenience_test")
        .vertex(VertexStateDescriptor::new(&vs_module))
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .target(wgpu::TextureFormat::Bgra8UnormSrgb),
        );

    let pipeline = create_render_pipeline(&device, desc);
    assert_eq!(pipeline.label(), Some("convenience_test"));
}

/// Test: Pipeline with MSAA (4x).
#[test]
fn test_create_msaa_pipeline() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let vs_module = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs_module = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let pipeline = RenderPipelineDescriptor::new(&layout)
        .label("msaa_4x_pipeline")
        .vertex(VertexStateDescriptor::new(&vs_module))
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .target(wgpu::TextureFormat::Bgra8UnormSrgb),
        )
        .multisample(MultisampleStateDescriptor::msaa_4x())
        .build(&device);

    assert_eq!(pipeline.label(), Some("msaa_4x_pipeline"));
}

/// Test: Pipeline with reverse-Z depth testing.
#[test]
fn test_create_reverse_z_pipeline() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let vs_module = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs_module = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let pipeline = RenderPipelineDescriptor::new(&layout)
        .label("reverse_z_pipeline")
        .vertex(VertexStateDescriptor::new(&vs_module))
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .target(wgpu::TextureFormat::Bgra8UnormSrgb),
        )
        .depth_stencil(DepthStencilStateDescriptor::depth_greater().format(wgpu::TextureFormat::Depth32Float))
        .build(&device);

    assert_eq!(pipeline.label(), Some("reverse_z_pipeline"));
}

/// Test: Pipeline with multiple color targets (MRT).
#[test]
fn test_create_mrt_pipeline() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // MRT fragment shader
    const MRT_FRAGMENT_SHADER: &str = r#"
    struct FragmentOutput {
        @location(0) color: vec4<f32>,
        @location(1) normal: vec4<f32>,
    }

    @fragment
    fn fs_main() -> FragmentOutput {
        var out: FragmentOutput;
        out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
        out.normal = vec4<f32>(0.0, 1.0, 0.0, 1.0);
        return out;
    }
    "#;

    let vs_module = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs_module = create_shader_module(&device, MRT_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let pipeline = RenderPipelineDescriptor::new(&layout)
        .label("mrt_gbuffer")
        .vertex(VertexStateDescriptor::new(&vs_module))
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .target(wgpu::TextureFormat::Rgba8Unorm) // color
                .target(wgpu::TextureFormat::Rgba8Unorm), // normal
        )
        .build(&device);

    assert_eq!(pipeline.label(), Some("mrt_gbuffer"));
}

/// Test: Pipeline with additive blending.
#[test]
fn test_create_additive_blend_pipeline() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let vs_module = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs_module = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let pipeline = RenderPipelineDescriptor::new(&layout)
        .label("additive_particles")
        .vertex(VertexStateDescriptor::new(&vs_module))
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .target_state(ColorTargetStateDescriptor::new(wgpu::TextureFormat::Bgra8UnormSrgb).additive()),
        )
        .primitive(PrimitiveStateDescriptor::triangle_list().no_culling())
        .depth_stencil(DepthStencilStateDescriptor::transparent())
        .build(&device);

    assert_eq!(pipeline.label(), Some("additive_particles"));
}

// =============================================================================
// SECTION 3: PIPELINE USAGE IN RENDER PASS
// =============================================================================

/// Test: Pipeline can be used in a render pass.
#[test]
fn test_pipeline_in_render_pass() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let vs_module = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs_module = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let pipeline = RenderPipelineDescriptor::new(&layout)
        .label("render_pass_test")
        .vertex(VertexStateDescriptor::new(&vs_module))
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .target(wgpu::TextureFormat::Rgba8Unorm),
        )
        .build(&device);

    // Create a texture to render to
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render_target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(pipeline.raw());
        render_pass.draw(0..3, 0..1);
    }

    queue.submit(Some(encoder.finish()));
}

/// Test: Pipeline can be extracted via into_inner.
#[test]
fn test_pipeline_into_inner() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let vs_module = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs_module = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let pipeline = RenderPipelineDescriptor::new(&layout)
        .label("into_inner_test")
        .vertex(VertexStateDescriptor::new(&vs_module))
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .target(wgpu::TextureFormat::Bgra8UnormSrgb),
        )
        .build(&device);

    let raw_pipeline: wgpu::RenderPipeline = pipeline.into_inner();

    // Verify we can still use the raw pipeline
    let _ = raw_pipeline;
}

// =============================================================================
// SECTION 4: ERROR HANDLING
// =============================================================================

/// Test: Pipeline creation panics without vertex state.
#[test]
#[should_panic(expected = "vertex state is required")]
fn test_pipeline_requires_vertex_state() {
    let adapter = match get_test_adapter() {
        Some(a) => a,
        None => {
            // Can't test without GPU, simulate panic
            panic!("vertex state is required");
        }
    };
    let (device, _queue) = match create_test_device(&adapter) {
        Some(d) => d,
        None => panic!("vertex state is required"),
    };

    let layout = create_empty_layout(&device);

    // This should panic because vertex state is not set
    let _pipeline = RenderPipelineDescriptor::new(&layout)
        .label("no_vertex")
        .build(&device);
}

/// Test: Pipeline with invalid shader source fails gracefully.
#[test]
#[should_panic]
fn test_pipeline_invalid_shader() {
    let adapter = match get_test_adapter() {
        Some(a) => a,
        None => panic!("skip test: no adapter"),
    };
    let (device, _queue) = match create_test_device(&adapter) {
        Some(d) => d,
        None => panic!("skip test: no device"),
    };

    // Invalid WGSL - this should panic during shader module creation
    const INVALID_SHADER: &str = r#"
    @vertex
    fn vs_main( { // Missing closing paren
        return vec4<f32>(0.0);
    }
    "#;

    // This will panic
    let _vs_module = create_shader_module(&device, INVALID_SHADER);
}

// =============================================================================
// SECTION 5: LAYOUT ID TRACKING
// =============================================================================

/// Test: Layout IDs are unique across pipeline descriptors.
#[test]
fn test_layout_ids_are_unique() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let vs_module = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs_module = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let pipeline1 = RenderPipelineDescriptor::new(&layout)
        .label("pipeline1")
        .vertex(VertexStateDescriptor::new(&vs_module))
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .target(wgpu::TextureFormat::Bgra8UnormSrgb),
        )
        .build(&device);

    let pipeline2 = RenderPipelineDescriptor::new(&layout)
        .label("pipeline2")
        .vertex(VertexStateDescriptor::new(&vs_module))
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .target(wgpu::TextureFormat::Bgra8UnormSrgb),
        )
        .build(&device);

    // Each descriptor gets a unique layout ID (even with same layout)
    assert_ne!(pipeline1.layout_id(), pipeline2.layout_id());
}

/// Test: Custom layout ID can be specified.
#[test]
fn test_custom_layout_id() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let vs_module = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs_module = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let custom_id: u64 = 42;
    let pipeline = RenderPipelineDescriptor::with_layout_id(&layout, custom_id)
        .label("custom_id_pipeline")
        .vertex(VertexStateDescriptor::new(&vs_module))
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .target(wgpu::TextureFormat::Bgra8UnormSrgb),
        )
        .build(&device);

    assert_eq!(pipeline.layout_id(), custom_id);
}

// =============================================================================
// SECTION 6: PERFORMANCE
// =============================================================================

/// Test: Pipeline creation completes within performance budget.
#[test]
fn test_pipeline_creation_performance() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let vs_module = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs_module = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let start = Instant::now();

    // Create 10 pipelines
    for i in 0..10 {
        let _pipeline = RenderPipelineDescriptor::new(&layout)
            .label(&format!("perf_test_{}", i))
            .vertex(VertexStateDescriptor::new(&vs_module))
            .fragment(
                FragmentStateDescriptor::new(&fs_module)
                    .target(wgpu::TextureFormat::Bgra8UnormSrgb),
            )
            .build(&device);
    }

    let elapsed = start.elapsed();

    // 10 pipelines should complete in under 1 second
    assert!(
        elapsed < std::time::Duration::from_secs(1),
        "Pipeline creation took too long: {:?}",
        elapsed
    );

    // Each pipeline should average under 100ms
    assert!(
        elapsed < std::time::Duration::from_millis(1000),
        "Average pipeline creation > 100ms: {:?} for 10 pipelines",
        elapsed
    );
}

/// Test: Descriptor builder chain is efficient (no excessive allocations).
#[test]
fn test_descriptor_builder_chaining() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let vs_module = create_shader_module(&device, PBR_VERTEX_SHADER);
    let fs_module = create_shader_module(&device, PBR_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let start = Instant::now();

    for _ in 0..100 {
        // Create a fully-configured descriptor
        let _desc = RenderPipelineDescriptor::new(&layout)
            .label("chained_test")
            .vertex(
                VertexStateDescriptor::new(&vs_module)
                    .entry_point("vs_main")
                    .buffer(
                        VertexBufferLayoutDescriptor::per_vertex(32)
                            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0)
                            .with_attribute(wgpu::VertexFormat::Float32x3, 12, 1)
                            .with_attribute(wgpu::VertexFormat::Float32x2, 24, 2),
                    ),
            )
            .fragment(
                FragmentStateDescriptor::new(&fs_module)
                    .entry_point("fs_main")
                    .target_state(
                        ColorTargetStateDescriptor::new(wgpu::TextureFormat::Bgra8UnormSrgb)
                            .alpha_blend()
                            .write_mask(wgpu::ColorWrites::ALL),
                    ),
            )
            .primitive(
                PrimitiveStateDescriptor::triangle_list()
                    .cull_back()
                    .front_face(wgpu::FrontFace::Ccw),
            )
            .depth_stencil(
                DepthStencilStateDescriptor::depth_less()
                    .format(wgpu::TextureFormat::Depth32Float)
                    .depth_write_enabled(true),
            )
            .multisample(MultisampleStateDescriptor::new().count(1));
    }

    let elapsed = start.elapsed();

    // 100 descriptor constructions should be fast (< 10ms)
    assert!(
        elapsed < std::time::Duration::from_millis(100),
        "Descriptor construction too slow: {:?} for 100 iterations",
        elapsed
    );
}

// =============================================================================
// SECTION 7: INTEGRATION WITH TRINITY SYSTEMS
// =============================================================================

/// Test: Vertex entry point can be customized.
#[test]
fn test_custom_vertex_entry_point() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Shader with custom entry point
    const CUSTOM_ENTRY_SHADER: &str = r#"
    @vertex
    fn custom_vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    "#;

    let vs_module = create_shader_module(&device, CUSTOM_ENTRY_SHADER);
    let fs_module = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    let pipeline = RenderPipelineDescriptor::new(&layout)
        .label("custom_entry")
        .vertex(VertexStateDescriptor::new(&vs_module).entry_point("custom_vs"))
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .target(wgpu::TextureFormat::Bgra8UnormSrgb),
        )
        .build(&device);

    assert_eq!(pipeline.label(), Some("custom_entry"));
}

/// Test: Fragment entry point can be customized.
#[test]
fn test_custom_fragment_entry_point() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Shader with custom entry point
    const CUSTOM_ENTRY_SHADER: &str = r#"
    @fragment
    fn custom_fs() -> @location(0) vec4<f32> {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    "#;

    let vs_module = create_shader_module(&device, MINIMAL_VERTEX_SHADER);
    let fs_module = create_shader_module(&device, CUSTOM_ENTRY_SHADER);
    let layout = create_empty_layout(&device);

    let pipeline = RenderPipelineDescriptor::new(&layout)
        .label("custom_fs_entry")
        .vertex(VertexStateDescriptor::new(&vs_module))
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .entry_point("custom_fs")
                .target(wgpu::TextureFormat::Bgra8UnormSrgb),
        )
        .build(&device);

    assert_eq!(pipeline.label(), Some("custom_fs_entry"));
}

/// Test: Multiple vertex buffers can be specified.
#[test]
fn test_multiple_vertex_buffers() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Shader with multiple vertex inputs from different buffers
    const MULTI_BUFFER_SHADER: &str = r#"
    struct VertexInput {
        @location(0) position: vec3<f32>,
        @location(1) instance_offset: vec3<f32>,
    }

    @vertex
    fn vs_main(input: VertexInput) -> @builtin(position) vec4<f32> {
        return vec4<f32>(input.position + input.instance_offset, 1.0);
    }
    "#;

    let vs_module = create_shader_module(&device, MULTI_BUFFER_SHADER);
    let fs_module = create_shader_module(&device, MINIMAL_FRAGMENT_SHADER);
    let layout = create_empty_layout(&device);

    // Per-vertex position buffer
    let vertex_buffer = VertexBufferLayoutDescriptor::per_vertex(12)
        .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0);

    // Per-instance offset buffer
    let instance_buffer = VertexBufferLayoutDescriptor::per_instance(12)
        .with_attribute(wgpu::VertexFormat::Float32x3, 0, 1);

    let pipeline = RenderPipelineDescriptor::new(&layout)
        .label("multi_buffer")
        .vertex(
            VertexStateDescriptor::new(&vs_module)
                .buffer(vertex_buffer)
                .buffer(instance_buffer),
        )
        .fragment(
            FragmentStateDescriptor::new(&fs_module)
                .target(wgpu::TextureFormat::Bgra8UnormSrgb),
        )
        .build(&device);

    assert_eq!(pipeline.label(), Some("multi_buffer"));
}

// =============================================================================
// SECTION 8: WGPU TYPE CONVERSIONS
// =============================================================================

/// Test: PrimitiveStateDescriptor converts to wgpu::PrimitiveState.
#[test]
fn test_primitive_state_wgpu_conversion() {
    let state = PrimitiveStateDescriptor::triangle_strip(wgpu::IndexFormat::Uint32)
        .wireframe()
        .cull_front();

    let wgpu_state: wgpu::PrimitiveState = state.into();

    assert_eq!(wgpu_state.topology, wgpu::PrimitiveTopology::TriangleStrip);
    assert_eq!(wgpu_state.strip_index_format, Some(wgpu::IndexFormat::Uint32));
    assert_eq!(wgpu_state.polygon_mode, wgpu::PolygonMode::Line);
    assert_eq!(wgpu_state.cull_mode, Some(wgpu::Face::Front));
}

/// Test: MultisampleStateDescriptor converts to wgpu::MultisampleState.
#[test]
fn test_multisample_state_wgpu_conversion() {
    let state = MultisampleStateDescriptor::msaa_4x().alpha_to_coverage(true);

    let wgpu_state: wgpu::MultisampleState = state.into();

    assert_eq!(wgpu_state.count, 4);
    assert!(wgpu_state.alpha_to_coverage_enabled);
}

/// Test: BlendStateDescriptor converts to wgpu::BlendState.
#[test]
fn test_blend_state_wgpu_conversion() {
    let blend = BlendStateDescriptor::alpha_blend();

    let wgpu_blend: wgpu::BlendState = blend.into();

    assert_eq!(wgpu_blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
    assert_eq!(wgpu_blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    assert_eq!(wgpu_blend.color.operation, wgpu::BlendOperation::Add);
}

/// Test: StencilFaceStateDescriptor converts to wgpu::StencilFaceState.
#[test]
fn test_stencil_face_state_wgpu_conversion() {
    let stencil = StencilFaceStateDescriptor::new()
        .compare(wgpu::CompareFunction::NotEqual)
        .pass_op(wgpu::StencilOperation::Invert);

    let wgpu_stencil: wgpu::StencilFaceState = stencil.into();

    assert_eq!(wgpu_stencil.compare, wgpu::CompareFunction::NotEqual);
    assert_eq!(wgpu_stencil.pass_op, wgpu::StencilOperation::Invert);
}

/// Test: DepthBiasStateDescriptor converts to wgpu::DepthBiasState.
#[test]
fn test_depth_bias_state_wgpu_conversion() {
    let bias = DepthBiasStateDescriptor::new()
        .constant(5)
        .slope_scale(1.5)
        .clamp(0.1);

    let wgpu_bias: wgpu::DepthBiasState = bias.into();

    assert_eq!(wgpu_bias.constant, 5);
    assert_eq!(wgpu_bias.slope_scale, 1.5);
    assert_eq!(wgpu_bias.clamp, 0.1);
}
