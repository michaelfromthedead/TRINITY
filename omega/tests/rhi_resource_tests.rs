//! Buffer, texture, and sampler resource tests.
//!
//! Mirrors tests/platform/rhi/test_resources.py plus Rust-native
//! buffer upload/readback simulation tests.

mod common;

use common::*;

// =========================================================================
// Buffer tests
// =========================================================================

fn make_buffer_device() -> MockDevice {
    create_test_device()
}

#[test]
fn test_buffer_creation_and_properties() {
    let device = make_buffer_device();
    let desc = BufferDesc {
        size: 4096,
        usage: BufferUsage::VERTEX | BufferUsage::INDEX,
        memory_type: MemoryType::Default,
        stride: 16,
    };
    let buffer = device.create_buffer(desc);

    assert!(buffer.is_valid());
    assert!(buffer.handle() > 0);
    assert_eq!(buffer.desc().size, 4096);
    assert_eq!(buffer.desc().stride, 16);
    assert!(buffer.desc().usage.contains(BufferUsage::VERTEX));
    assert!(buffer.desc().usage.contains(BufferUsage::INDEX));
}

#[test]
fn test_buffer_destroy() {
    let device = make_buffer_device();
    let desc = BufferDesc {
        size: 1024,
        usage: BufferUsage::CONSTANT,
        memory_type: MemoryType::Default,
        stride: 0,
    };
    let mut buffer = device.create_buffer(desc);

    assert!(buffer.is_valid());
    buffer.destroy();
    assert!(!buffer.is_valid());
}

#[test]
fn test_buffer_unique_handles() {
    let device = make_buffer_device();
    let desc = BufferDesc {
        size: 1024,
        usage: BufferUsage::STORAGE,
        memory_type: MemoryType::Default,
        stride: 0,
    };
    let b1 = device.create_buffer(desc.clone());
    let b2 = device.create_buffer(desc);

    assert_ne!(b1.handle(), b2.handle());
}

#[test]
fn test_buffer_upload_memory_type() {
    let device = make_buffer_device();
    let desc = BufferDesc {
        size: 256,
        usage: BufferUsage::COPY_SRC,
        memory_type: MemoryType::Upload,
        stride: 0,
    };
    let buffer = device.create_buffer(desc);
    assert!(buffer.is_valid());
    assert_eq!(buffer.desc().memory_type, MemoryType::Upload);
}

#[test]
fn test_buffer_readback_memory_type() {
    let device = make_buffer_device();
    let desc = BufferDesc {
        size: 256,
        usage: BufferUsage::COPY_DST,
        memory_type: MemoryType::Readback,
        stride: 0,
    };
    let buffer = device.create_buffer(desc);
    assert!(buffer.is_valid());
    assert_eq!(buffer.desc().memory_type, MemoryType::Readback);
}

#[test]
fn test_buffer_usage_flags_vertex() {
    let device = make_buffer_device();
    let desc = BufferDesc {
        size: 1024,
        usage: BufferUsage::VERTEX,
        memory_type: MemoryType::Default,
        stride: 0,
    };
    let buffer = device.create_buffer(desc);
    assert!(buffer.desc().usage.contains(BufferUsage::VERTEX));
}

#[test]
fn test_buffer_usage_flags_combined() {
    let device = make_buffer_device();
    let desc = BufferDesc {
        size: 2048,
        usage: BufferUsage::INDEX | BufferUsage::COPY_DST,
        memory_type: MemoryType::Default,
        stride: 0,
    };
    let buffer = device.create_buffer(desc);
    assert!(buffer.desc().usage.contains(BufferUsage::INDEX));
    assert!(buffer.desc().usage.contains(BufferUsage::COPY_DST));
    assert!(!buffer.desc().usage.contains(BufferUsage::CONSTANT));
}

#[test]
fn test_buffer_destroy_idempotent() {
    let device = make_buffer_device();
    let desc = BufferDesc {
        size: 64,
        usage: BufferUsage::INDIRECT,
        memory_type: MemoryType::Default,
        stride: 0,
    };
    let mut buffer = device.create_buffer(desc);
    buffer.destroy();
    buffer.destroy(); // second destroy should not panic
    assert!(!buffer.is_valid());
}

#[test]
fn test_buffer_copy_buffer_recording() {
    let device = make_buffer_device();
    let src = device.create_buffer(BufferDesc {
        size: 2048,
        usage: BufferUsage::COPY_SRC,
        memory_type: MemoryType::Default,
        stride: 0,
    });
    let dst = device.create_buffer(BufferDesc {
        size: 2048,
        usage: BufferUsage::COPY_DST,
        memory_type: MemoryType::Default,
        stride: 0,
    });

    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.copy_buffer(dst.handle(), 0, src.handle(), 0, 1024);
    cmd.end();

    let cmds = cmd.recorded_commands();
    assert_eq!(cmds.len(), 1);
    assert_eq!(cmds[0].cmd_type, "copy_buffer");
}

// =========================================================================
// Texture tests
// =========================================================================

#[test]
fn test_texture_creation_2d() {
    let device = create_test_device();
    let desc = TextureDesc {
        ty: TextureType::Texture2D,
        format: Format::RGBA8Unorm,
        width: 512,
        height: 512,
        depth: 1,
        mip_levels: 4,
        array_size: 1,
        sample_count: SampleCount::X1,
        usage: TextureUsage::SHADER_RESOURCE | TextureUsage::RENDER_TARGET,
    };
    let texture = device.create_texture(desc);

    assert!(texture.is_valid());
    assert!(texture.handle() > 0);
    assert_eq!(texture.desc().width, 512);
    assert_eq!(texture.desc().height, 512);
    assert_eq!(texture.desc().mip_levels, 4);
    assert_eq!(texture.desc().format, Format::RGBA8Unorm);
}

#[test]
fn test_texture_3d_creation() {
    let device = create_test_device();
    let desc = TextureDesc {
        ty: TextureType::Texture3D,
        format: Format::RGBA16Float,
        width: 128,
        height: 128,
        depth: 128,
        mip_levels: 1,
        array_size: 1,
        sample_count: SampleCount::X1,
        usage: TextureUsage::UNORDERED_ACCESS,
    };
    let texture = device.create_texture(desc);

    assert!(texture.is_valid());
    assert_eq!(texture.desc().depth, 128);
}

#[test]
fn test_texture_cube_creation() {
    let device = create_test_device();
    let desc = TextureDesc {
        ty: TextureType::TextureCube,
        format: Format::RGBA8Unorm,
        width: 256,
        height: 256,
        depth: 1,
        mip_levels: 1,
        array_size: 1,
        sample_count: SampleCount::X1,
        usage: TextureUsage::SHADER_RESOURCE,
    };
    let texture = device.create_texture(desc);
    assert!(texture.is_valid());
}

#[test]
fn test_texture_1d_creation() {
    let device = create_test_device();
    let desc = TextureDesc {
        ty: TextureType::Texture1D,
        format: Format::R32Float,
        width: 512,
        height: 1,
        depth: 1,
        mip_levels: 1,
        array_size: 1,
        sample_count: SampleCount::X1,
        usage: TextureUsage::SHADER_RESOURCE,
    };
    let texture = device.create_texture(desc);
    assert!(texture.is_valid());
    assert_eq!(texture.desc().ty, TextureType::Texture1D);
}

#[test]
fn test_texture_array_creation() {
    let device = create_test_device();
    let desc = TextureDesc {
        ty: TextureType::TextureArray,
        format: Format::RGBA8Unorm,
        width: 128,
        height: 128,
        depth: 1,
        mip_levels: 1,
        array_size: 6,
        sample_count: SampleCount::X1,
        usage: TextureUsage::SHADER_RESOURCE,
    };
    let texture = device.create_texture(desc);
    assert!(texture.is_valid());
    assert_eq!(texture.desc().array_size, 6);
}

#[test]
fn test_texture_destroy() {
    let device = create_test_device();
    let desc = TextureDesc {
        ty: TextureType::Texture2D,
        format: Format::R32Float,
        width: 64,
        height: 64,
        depth: 1,
        mip_levels: 1,
        array_size: 1,
        sample_count: SampleCount::X1,
        usage: TextureUsage::SHADER_RESOURCE,
    };
    let mut texture = device.create_texture(desc);

    assert!(texture.is_valid());
    texture.destroy();
    assert!(!texture.is_valid());
}

#[test]
fn test_texture_destroy_idempotent() {
    let device = create_test_device();
    let desc = TextureDesc {
        ty: TextureType::Texture2D,
        format: Format::RGBA8Unorm,
        width: 32,
        height: 32,
        depth: 1,
        mip_levels: 1,
        array_size: 1,
        sample_count: SampleCount::X1,
        usage: TextureUsage::SHADER_RESOURCE,
    };
    let mut texture = device.create_texture(desc);
    texture.destroy();
    texture.destroy(); // second destroy should not panic
    assert!(!texture.is_valid());
}

#[test]
fn test_texture_unique_handles() {
    let device = create_test_device();
    let desc = TextureDesc {
        ty: TextureType::Texture2D,
        format: Format::RGBA8Unorm,
        width: 128,
        height: 128,
        depth: 1,
        mip_levels: 1,
        array_size: 1,
        sample_count: SampleCount::X1,
        usage: TextureUsage::SHADER_RESOURCE,
    };
    let t1 = device.create_texture(desc.clone());
    let t2 = device.create_texture(desc);
    assert_ne!(t1.handle(), t2.handle());
}

#[test]
fn test_texture_usage_flags() {
    let device = create_test_device();
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_RESOURCE;
    let desc = TextureDesc {
        ty: TextureType::Texture2D,
        format: Format::RGBA8Unorm,
        width: 256,
        height: 256,
        depth: 1,
        mip_levels: 1,
        array_size: 1,
        sample_count: SampleCount::X1,
        usage,
    };
    let texture = device.create_texture(desc);

    assert!(texture.desc().usage.contains(TextureUsage::RENDER_TARGET));
    assert!(texture.desc().usage.contains(TextureUsage::SHADER_RESOURCE));
    assert!(!texture.desc().usage.contains(TextureUsage::DEPTH_STENCIL));
}

#[test]
fn test_texture_different_formats() {
    let device = create_test_device();
    let base = TextureDesc {
        ty: TextureType::Texture2D,
        format: Format::RGBA8Unorm,
        width: 128,
        height: 128,
        depth: 1,
        mip_levels: 1,
        array_size: 1,
        sample_count: SampleCount::X1,
        usage: TextureUsage::SHADER_RESOURCE,
    };

    let tex_rgba8 = device.create_texture(TextureDesc { format: Format::RGBA8Unorm, ..base.clone() });
    assert_eq!(tex_rgba8.desc().format, Format::RGBA8Unorm);

    let tex_rgba16f = device.create_texture(TextureDesc { format: Format::RGBA16Float, ..base });
    assert_eq!(tex_rgba16f.desc().format, Format::RGBA16Float);
    assert_ne!(tex_rgba16f.desc().format, tex_rgba8.desc().format);
}

#[test]
fn test_texture_sample_counts() {
    let device = create_test_device();
    let base = TextureDesc {
        ty: TextureType::Texture2D,
        format: Format::RGBA8Unorm,
        width: 256,
        height: 256,
        depth: 1,
        mip_levels: 1,
        array_size: 1,
        sample_count: SampleCount::X1,
        usage: TextureUsage::RENDER_TARGET,
    };

    let tex1 = device.create_texture(TextureDesc { sample_count: SampleCount::X1, ..base.clone() });
    assert_eq!(tex1.desc().sample_count, SampleCount::X1);

    let tex4 = device.create_texture(TextureDesc { sample_count: SampleCount::X4, ..base });
    assert_eq!(tex4.desc().sample_count, SampleCount::X4);
    assert_ne!(tex4.desc().sample_count, tex1.desc().sample_count);
}

// =========================================================================
// Sampler tests
// =========================================================================

#[test]
fn test_sampler_creation() {
    let device = create_test_device();
    let desc = SamplerDesc {
        min_filter: FilterMode::Linear,
        mag_filter: FilterMode::Linear,
        mip_filter: FilterMode::Nearest,
        address_u: AddressMode::Clamp,
        address_v: AddressMode::Wrap,
        address_w: AddressMode::Mirror,
        max_anisotropy: 16,
        compare_op: Some(CompareOp::Less),
        min_lod: 0.0,
        max_lod: 10.0,
        mip_lod_bias: 0.0,
    };
    let sampler = device.create_sampler(desc);

    assert!(sampler.is_valid());
    assert!(sampler.handle() > 0);
    assert_eq!(sampler.desc().min_filter, FilterMode::Linear);
    assert_eq!(sampler.desc().address_u, AddressMode::Clamp);
    assert_eq!(sampler.desc().max_anisotropy, 16);
}

#[test]
fn test_sampler_destroy() {
    let device = create_test_device();
    let desc = SamplerDesc::default();
    let mut sampler = device.create_sampler(desc);

    assert!(sampler.is_valid());
    sampler.destroy();
    assert!(!sampler.is_valid());
}

#[test]
fn test_sampler_unique_handles() {
    let device = create_test_device();
    let desc = SamplerDesc::default();
    let s1 = device.create_sampler(desc.clone());
    let s2 = device.create_sampler(desc);
    assert_ne!(s1.handle(), s2.handle());
}

#[test]
fn test_sampler_default_fields() {
    let desc = SamplerDesc::default();
    assert_eq!(desc.min_filter, FilterMode::Linear);
    assert_eq!(desc.mag_filter, FilterMode::Linear);
    assert_eq!(desc.mip_filter, FilterMode::Linear);
    assert_eq!(desc.address_u, AddressMode::Wrap);
    assert_eq!(desc.max_anisotropy, 1);
    assert!(desc.compare_op.is_none());
}

#[test]
fn test_sampler_all_address_modes() {
    let device = create_test_device();
    for mode in &[AddressMode::Wrap, AddressMode::Clamp, AddressMode::Mirror, AddressMode::Border] {
        let desc = SamplerDesc {
            address_u: *mode,
            address_v: *mode,
            address_w: *mode,
            ..Default::default()
        };
        let sampler = device.create_sampler(desc);
        assert!(sampler.is_valid());
    }
}

#[test]
fn test_sampler_compare_ops() {
    let device = create_test_device();
    for op in &[
        CompareOp::Never,
        CompareOp::Less,
        CompareOp::Equal,
        CompareOp::Always,
    ] {
        let desc = SamplerDesc {
            compare_op: Some(*op),
            ..Default::default()
        };
        let sampler = device.create_sampler(desc);
        assert!(sampler.is_valid());
        assert_eq!(sampler.desc().compare_op, Some(*op));
    }
}
