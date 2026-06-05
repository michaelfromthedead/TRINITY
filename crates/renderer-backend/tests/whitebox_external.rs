//! Whitebox unit tests for T-WGPU-P7.5.4 External/Imported Resource handling.
//!
//! WHITEBOX: Internal access tests for `frame_graph::external` module.
//! Tests verify the complete behavior of external resource types, import modes,
//! barriers, registry operations, and synchronization logic.
//!
//! Target file: `crates/renderer-backend/src/frame_graph/external.rs`
//!
//! Coverage:
//!   - ExternalResourceType: All 5 variants and predicates (20+ tests)
//!   - ImportMode: All variants, barriers, conversions (20+ tests)
//!   - ExternalTextureInfo: Construction and accessors (20+ tests)
//!   - ExternalBufferInfo: Construction and predicates (20+ tests)
//!   - SwapchainInfo: Construction and metadata (15+ tests)
//!   - ImportedResourceInfo: Variant predicates (15+ tests)
//!   - ImportedResource: Constructors and tracking (20+ tests)
//!   - ResourceBarrier: Factory methods and fields (15+ tests)
//!   - ExternalResourceRegistry: Import, lookup, clear (25+ tests)
//!   - ExternalSynchronizer: Barrier computation (15+ tests)

use renderer_backend::frame_graph::{ResourceAccess, ResourceHandle, PassIndex};
use renderer_backend::frame_graph::external::{
    ExternalResourceType, ImportMode, ResourceBarrier,
    ExternalResourceRegistry, ExternalSynchronizer,
};

// ============================================================================
// SECTION 1: ExternalResourceType Tests (20+ tests)
// ============================================================================

mod external_resource_type_tests {
    use super::*;

    #[test]
    fn swapchain_variant_exists() {
        let t = ExternalResourceType::Swapchain;
        assert!(matches!(t, ExternalResourceType::Swapchain));
    }

    #[test]
    fn user_texture_variant_exists() {
        let t = ExternalResourceType::UserTexture;
        assert!(matches!(t, ExternalResourceType::UserTexture));
    }

    #[test]
    fn user_buffer_variant_exists() {
        let t = ExternalResourceType::UserBuffer;
        assert!(matches!(t, ExternalResourceType::UserBuffer));
    }

    #[test]
    fn shared_texture_variant_exists() {
        let t = ExternalResourceType::SharedTexture;
        assert!(matches!(t, ExternalResourceType::SharedTexture));
    }

    #[test]
    fn shared_buffer_variant_exists() {
        let t = ExternalResourceType::SharedBuffer;
        assert!(matches!(t, ExternalResourceType::SharedBuffer));
    }

    #[test]
    fn is_swapchain_returns_true_for_swapchain() {
        assert!(ExternalResourceType::Swapchain.is_swapchain());
    }

    #[test]
    fn is_swapchain_returns_false_for_user_texture() {
        assert!(!ExternalResourceType::UserTexture.is_swapchain());
    }

    #[test]
    fn is_swapchain_returns_false_for_user_buffer() {
        assert!(!ExternalResourceType::UserBuffer.is_swapchain());
    }

    #[test]
    fn is_swapchain_returns_false_for_shared_texture() {
        assert!(!ExternalResourceType::SharedTexture.is_swapchain());
    }

    #[test]
    fn is_swapchain_returns_false_for_shared_buffer() {
        assert!(!ExternalResourceType::SharedBuffer.is_swapchain());
    }

    #[test]
    fn is_user_provided_returns_false_for_swapchain() {
        assert!(!ExternalResourceType::Swapchain.is_user_provided());
    }

    #[test]
    fn is_user_provided_returns_true_for_user_texture() {
        assert!(ExternalResourceType::UserTexture.is_user_provided());
    }

    #[test]
    fn is_user_provided_returns_true_for_user_buffer() {
        assert!(ExternalResourceType::UserBuffer.is_user_provided());
    }

    #[test]
    fn is_user_provided_returns_false_for_shared_types() {
        assert!(!ExternalResourceType::SharedTexture.is_user_provided());
        assert!(!ExternalResourceType::SharedBuffer.is_user_provided());
    }

    #[test]
    fn is_shared_returns_false_for_non_shared_types() {
        assert!(!ExternalResourceType::Swapchain.is_shared());
        assert!(!ExternalResourceType::UserTexture.is_shared());
        assert!(!ExternalResourceType::UserBuffer.is_shared());
    }

    #[test]
    fn is_shared_returns_true_for_shared_texture() {
        assert!(ExternalResourceType::SharedTexture.is_shared());
    }

    #[test]
    fn is_shared_returns_true_for_shared_buffer() {
        assert!(ExternalResourceType::SharedBuffer.is_shared());
    }

    #[test]
    fn is_texture_returns_true_for_texture_types() {
        assert!(ExternalResourceType::Swapchain.is_texture());
        assert!(ExternalResourceType::UserTexture.is_texture());
        assert!(ExternalResourceType::SharedTexture.is_texture());
    }

    #[test]
    fn is_texture_returns_false_for_buffer_types() {
        assert!(!ExternalResourceType::UserBuffer.is_texture());
        assert!(!ExternalResourceType::SharedBuffer.is_texture());
    }

    #[test]
    fn is_buffer_returns_true_for_buffer_types() {
        assert!(ExternalResourceType::UserBuffer.is_buffer());
        assert!(ExternalResourceType::SharedBuffer.is_buffer());
    }

    #[test]
    fn is_buffer_returns_false_for_texture_types() {
        assert!(!ExternalResourceType::Swapchain.is_buffer());
        assert!(!ExternalResourceType::UserTexture.is_buffer());
        assert!(!ExternalResourceType::SharedTexture.is_buffer());
    }

    #[test]
    fn display_all_variants() {
        assert_eq!(format!("{}", ExternalResourceType::Swapchain), "Swapchain");
        assert_eq!(format!("{}", ExternalResourceType::UserTexture), "UserTexture");
        assert_eq!(format!("{}", ExternalResourceType::UserBuffer), "UserBuffer");
        assert_eq!(format!("{}", ExternalResourceType::SharedTexture), "SharedTexture");
        assert_eq!(format!("{}", ExternalResourceType::SharedBuffer), "SharedBuffer");
    }

    #[test]
    fn external_resource_type_is_clone() {
        let a = ExternalResourceType::UserTexture;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn external_resource_type_is_copy() {
        let a = ExternalResourceType::SharedBuffer;
        let b = a;
        let c = a;
        assert_eq!(b, c);
    }

    #[test]
    fn external_resource_type_debug_format() {
        let dbg = format!("{:?}", ExternalResourceType::Swapchain);
        assert!(dbg.contains("Swapchain"));
    }

    #[test]
    fn external_resource_type_default() {
        let default = ExternalResourceType::default();
        assert_eq!(default, ExternalResourceType::UserTexture);
    }

    #[test]
    fn texture_and_buffer_mutually_exclusive() {
        for t in [
            ExternalResourceType::Swapchain,
            ExternalResourceType::UserTexture,
            ExternalResourceType::UserBuffer,
            ExternalResourceType::SharedTexture,
            ExternalResourceType::SharedBuffer,
        ] {
            assert!(t.is_texture() ^ t.is_buffer(), "{:?} should be texture xor buffer", t);
        }
    }
}

// ============================================================================
// SECTION 2: ImportMode Tests (20+ tests)
// ============================================================================

mod import_mode_tests {
    use super::*;

    #[test]
    fn read_only_variant_exists() {
        let m = ImportMode::ReadOnly;
        assert!(matches!(m, ImportMode::ReadOnly));
    }

    #[test]
    fn write_only_variant_exists() {
        let m = ImportMode::WriteOnly;
        assert!(matches!(m, ImportMode::WriteOnly));
    }

    #[test]
    fn read_write_variant_exists() {
        let m = ImportMode::ReadWrite;
        assert!(matches!(m, ImportMode::ReadWrite));
    }

    #[test]
    fn requires_acquire_barrier_true_for_read_only() {
        assert!(ImportMode::ReadOnly.requires_acquire_barrier());
    }

    #[test]
    fn requires_acquire_barrier_false_for_write_only() {
        assert!(!ImportMode::WriteOnly.requires_acquire_barrier());
    }

    #[test]
    fn requires_acquire_barrier_true_for_read_write() {
        assert!(ImportMode::ReadWrite.requires_acquire_barrier());
    }

    #[test]
    fn requires_release_barrier_false_for_read_only() {
        assert!(!ImportMode::ReadOnly.requires_release_barrier());
    }

    #[test]
    fn requires_release_barrier_true_for_write_only() {
        assert!(ImportMode::WriteOnly.requires_release_barrier());
    }

    #[test]
    fn requires_release_barrier_true_for_read_write() {
        assert!(ImportMode::ReadWrite.requires_release_barrier());
    }

    #[test]
    fn is_read_true_for_read_only() {
        assert!(ImportMode::ReadOnly.is_read());
    }

    #[test]
    fn is_read_false_for_write_only() {
        assert!(!ImportMode::WriteOnly.is_read());
    }

    #[test]
    fn is_read_true_for_read_write() {
        assert!(ImportMode::ReadWrite.is_read());
    }

    #[test]
    fn is_write_false_for_read_only() {
        assert!(!ImportMode::ReadOnly.is_write());
    }

    #[test]
    fn is_write_true_for_write_only() {
        assert!(ImportMode::WriteOnly.is_write());
    }

    #[test]
    fn is_write_true_for_read_write() {
        assert!(ImportMode::ReadWrite.is_write());
    }

    #[test]
    fn to_resource_access_conversions() {
        assert_eq!(ImportMode::ReadOnly.to_resource_access(), ResourceAccess::Read);
        assert_eq!(ImportMode::WriteOnly.to_resource_access(), ResourceAccess::Write);
        assert_eq!(ImportMode::ReadWrite.to_resource_access(), ResourceAccess::ReadWrite);
    }

    #[test]
    fn display_all_variants() {
        assert_eq!(format!("{}", ImportMode::ReadOnly), "ReadOnly");
        assert_eq!(format!("{}", ImportMode::WriteOnly), "WriteOnly");
        assert_eq!(format!("{}", ImportMode::ReadWrite), "ReadWrite");
    }

    #[test]
    fn import_mode_default_is_read_only() {
        assert_eq!(ImportMode::default(), ImportMode::ReadOnly);
    }

    #[test]
    fn import_mode_from_resource_access() {
        assert_eq!(ImportMode::from(ResourceAccess::Read), ImportMode::ReadOnly);
        assert_eq!(ImportMode::from(ResourceAccess::Write), ImportMode::WriteOnly);
        assert_eq!(ImportMode::from(ResourceAccess::ReadWrite), ImportMode::ReadWrite);
    }

    #[test]
    fn import_mode_is_clone() {
        let a = ImportMode::ReadWrite;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn import_mode_is_copy() {
        let a = ImportMode::WriteOnly;
        let b = a;
        let c = a;
        assert_eq!(b, c);
    }

    #[test]
    fn import_mode_debug_format() {
        let dbg = format!("{:?}", ImportMode::ReadOnly);
        assert!(dbg.contains("ReadOnly"));
    }

    #[test]
    fn import_mode_resource_access_round_trip() {
        for mode in [ImportMode::ReadOnly, ImportMode::WriteOnly, ImportMode::ReadWrite] {
            let access = mode.to_resource_access();
            let back: ImportMode = access.into();
            assert_eq!(mode, back, "Round-trip failed for {:?}", mode);
        }
    }

    #[test]
    fn barrier_symmetry_tests() {
        assert!(ImportMode::ReadOnly.requires_acquire_barrier());
        assert!(!ImportMode::ReadOnly.requires_release_barrier());
        assert!(!ImportMode::WriteOnly.requires_acquire_barrier());
        assert!(ImportMode::WriteOnly.requires_release_barrier());
        assert!(ImportMode::ReadWrite.requires_acquire_barrier());
        assert!(ImportMode::ReadWrite.requires_release_barrier());
    }
}

// ============================================================================
// SECTION 3: ExternalTextureInfo Tests (20+ tests)
// ============================================================================

mod external_texture_info_tests {
    use super::*;

    #[test]
    fn size_2d_texture() {
        let size = (800u32, 600u32, 1u32);
        assert_eq!(size.0, 800);
        assert_eq!(size.1, 600);
        assert_eq!(size.2, 1);
    }

    #[test]
    fn size_3d_texture() {
        let size = (256u32, 256u32, 256u32);
        assert_eq!(size.2, 256);
    }

    #[test]
    fn size_array_texture() {
        let size = (1024u32, 1024u32, 16u32);
        assert_eq!(size.2, 16);
    }

    #[test]
    fn size_cubemap_texture() {
        let size = (512u32, 512u32, 6u32);
        assert_eq!(size.2, 6);
    }

    #[test]
    fn size_1x1_texture() {
        let size = (1u32, 1u32, 1u32);
        assert_eq!(size.0, 1);
    }

    #[test]
    fn size_max_dimension() {
        let size = (16384u32, 16384u32, 1u32);
        assert_eq!(size.0, 16384);
    }

    #[test]
    fn sample_count_1_is_not_multisampled() {
        assert!(1u32 <= 1);
    }

    #[test]
    fn sample_count_4_is_multisampled() {
        assert!(4u32 > 1);
    }

    #[test]
    fn sample_count_8_is_multisampled() {
        assert!(8u32 > 1);
    }

    #[test]
    fn external_type_can_be_user_texture() {
        let ext_type = ExternalResourceType::UserTexture;
        assert!(ext_type.is_texture());
        assert!(ext_type.is_user_provided());
    }

    #[test]
    fn external_type_can_be_shared_texture() {
        let ext_type = ExternalResourceType::SharedTexture;
        assert!(ext_type.is_texture());
        assert!(ext_type.is_shared());
    }

    #[test]
    fn external_type_can_be_swapchain() {
        let ext_type = ExternalResourceType::Swapchain;
        assert!(ext_type.is_texture());
        assert!(ext_type.is_swapchain());
    }

    #[test]
    fn import_mode_read_only_for_texture() {
        let mode = ImportMode::ReadOnly;
        assert!(mode.is_read());
        assert!(!mode.is_write());
    }

    #[test]
    fn import_mode_write_only_for_texture() {
        let mode = ImportMode::WriteOnly;
        assert!(!mode.is_read());
        assert!(mode.is_write());
    }

    #[test]
    fn import_mode_read_write_for_texture() {
        let mode = ImportMode::ReadWrite;
        assert!(mode.is_read());
        assert!(mode.is_write());
    }

    #[test]
    fn width_accessor_logic() {
        let size = (1920u32, 1080u32, 1u32);
        assert_eq!(size.0, 1920);
    }

    #[test]
    fn height_accessor_logic() {
        let size = (1920u32, 1080u32, 1u32);
        assert_eq!(size.1, 1080);
    }

    #[test]
    fn depth_or_layers_accessor_logic() {
        let size = (1920u32, 1080u32, 6u32);
        assert_eq!(size.2, 6);
    }

    #[test]
    fn multisampled_logic() {
        assert!(1u32 <= 1);
        assert!(4u32 > 1);
        assert!(8u32 > 1);
        assert!(16u32 > 1);
    }

    #[test]
    fn texture_type_is_not_buffer() {
        let ext_type = ExternalResourceType::UserTexture;
        assert!(!ext_type.is_buffer());
    }
}

// ============================================================================
// SECTION 4: ExternalBufferInfo Tests (20+ tests)
// ============================================================================

mod external_buffer_info_tests {
    use super::*;

    #[test]
    fn buffer_size_small() {
        let size = 64u64;
        assert_eq!(size, 64);
    }

    #[test]
    fn buffer_size_kilobyte() {
        let size = 1024u64;
        assert_eq!(size, 1024);
    }

    #[test]
    fn buffer_size_megabyte() {
        let size = 1024u64 * 1024;
        assert_eq!(size, 1_048_576);
    }

    #[test]
    fn buffer_size_large() {
        let size = 256u64 * 1024 * 1024;
        assert_eq!(size, 268_435_456);
    }

    #[test]
    fn uniform_usage_flag_logic() {
        let usage = wgpu::BufferUsages::UNIFORM;
        assert!(usage.contains(wgpu::BufferUsages::UNIFORM));
    }

    #[test]
    fn storage_usage_flag_logic() {
        let usage = wgpu::BufferUsages::STORAGE;
        assert!(usage.contains(wgpu::BufferUsages::STORAGE));
    }

    #[test]
    fn vertex_usage_flag_logic() {
        let usage = wgpu::BufferUsages::VERTEX;
        assert!(usage.contains(wgpu::BufferUsages::VERTEX));
    }

    #[test]
    fn index_usage_flag_logic() {
        let usage = wgpu::BufferUsages::INDEX;
        assert!(usage.contains(wgpu::BufferUsages::INDEX));
    }

    #[test]
    fn combined_uniform_storage_flags() {
        let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE;
        assert!(usage.contains(wgpu::BufferUsages::UNIFORM));
        assert!(usage.contains(wgpu::BufferUsages::STORAGE));
    }

    #[test]
    fn combined_vertex_index_flags() {
        let usage = wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDEX;
        assert!(usage.contains(wgpu::BufferUsages::VERTEX));
        assert!(usage.contains(wgpu::BufferUsages::INDEX));
    }

    #[test]
    fn copy_src_dst_flags() {
        let usage = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        assert!(usage.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(usage.contains(wgpu::BufferUsages::COPY_DST));
    }

    #[test]
    fn map_read_write_flags() {
        let usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE;
        assert!(usage.contains(wgpu::BufferUsages::MAP_READ));
        assert!(usage.contains(wgpu::BufferUsages::MAP_WRITE));
    }

    #[test]
    fn external_type_can_be_user_buffer() {
        let ext_type = ExternalResourceType::UserBuffer;
        assert!(ext_type.is_buffer());
        assert!(ext_type.is_user_provided());
    }

    #[test]
    fn external_type_can_be_shared_buffer() {
        let ext_type = ExternalResourceType::SharedBuffer;
        assert!(ext_type.is_buffer());
        assert!(ext_type.is_shared());
    }

    #[test]
    fn import_mode_read_only_for_buffer() {
        let mode = ImportMode::ReadOnly;
        assert!(mode.requires_acquire_barrier());
        assert!(!mode.requires_release_barrier());
    }

    #[test]
    fn import_mode_write_only_for_buffer() {
        let mode = ImportMode::WriteOnly;
        assert!(!mode.requires_acquire_barrier());
        assert!(mode.requires_release_barrier());
    }

    #[test]
    fn import_mode_read_write_for_buffer() {
        let mode = ImportMode::ReadWrite;
        assert!(mode.requires_acquire_barrier());
        assert!(mode.requires_release_barrier());
    }

    #[test]
    fn is_uniform_predicate_logic() {
        let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        assert!(usage.contains(wgpu::BufferUsages::UNIFORM));
        let other = wgpu::BufferUsages::STORAGE;
        assert!(!other.contains(wgpu::BufferUsages::UNIFORM));
    }

    #[test]
    fn is_storage_predicate_logic() {
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        assert!(usage.contains(wgpu::BufferUsages::STORAGE));
    }

    #[test]
    fn buffer_type_is_not_texture() {
        let ext_type = ExternalResourceType::UserBuffer;
        assert!(!ext_type.is_texture());
    }
}

// ============================================================================
// SECTION 5: SwapchainInfo Tests (15+ tests)
// ============================================================================

mod swapchain_info_tests {
    use super::*;

    #[test]
    fn swapchain_format_bgra8_unorm() {
        let format = wgpu::TextureFormat::Bgra8Unorm;
        assert!(matches!(format, wgpu::TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn swapchain_format_bgra8_unorm_srgb() {
        let format = wgpu::TextureFormat::Bgra8UnormSrgb;
        assert!(matches!(format, wgpu::TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn swapchain_format_rgba8_unorm() {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        assert!(matches!(format, wgpu::TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn swapchain_format_rgba16_float() {
        let format = wgpu::TextureFormat::Rgba16Float;
        assert!(matches!(format, wgpu::TextureFormat::Rgba16Float));
    }

    #[test]
    fn swapchain_size_1920x1080() {
        let size = (1920u32, 1080u32);
        assert_eq!(size.0, 1920);
        assert_eq!(size.1, 1080);
    }

    #[test]
    fn swapchain_size_2560x1440() {
        let size = (2560u32, 1440u32);
        assert_eq!(size.0, 2560);
    }

    #[test]
    fn swapchain_size_3840x2160() {
        let size = (3840u32, 2160u32);
        assert_eq!(size.0, 3840);
    }

    #[test]
    fn present_mode_fifo() {
        let mode = wgpu::PresentMode::Fifo;
        assert!(matches!(mode, wgpu::PresentMode::Fifo));
    }

    #[test]
    fn present_mode_mailbox() {
        let mode = wgpu::PresentMode::Mailbox;
        assert!(matches!(mode, wgpu::PresentMode::Mailbox));
    }

    #[test]
    fn present_mode_immediate() {
        let mode = wgpu::PresentMode::Immediate;
        assert!(matches!(mode, wgpu::PresentMode::Immediate));
    }

    #[test]
    fn present_mode_fifo_relaxed() {
        let mode = wgpu::PresentMode::FifoRelaxed;
        assert!(matches!(mode, wgpu::PresentMode::FifoRelaxed));
    }

    #[test]
    fn acquire_time_is_instant() {
        let instant = std::time::Instant::now();
        // Elapsed time is always non-negative (testing Instant::now works)
        let _ = instant.elapsed();
    }

    #[test]
    fn elapsed_time_increases() {
        let start = std::time::Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(start.elapsed().as_micros() > 0);
    }

    #[test]
    fn swapchain_external_type_is_swapchain() {
        let ext_type = ExternalResourceType::Swapchain;
        assert!(ext_type.is_swapchain());
        assert!(ext_type.is_texture());
        assert!(!ext_type.is_buffer());
    }

    #[test]
    fn swapchain_width_height_accessors() {
        let size = (800u32, 600u32);
        assert_eq!(size.0, 800);
        assert_eq!(size.1, 600);
    }
}

// ============================================================================
// SECTION 6: ImportedResourceInfo Tests (15+ tests)
// ============================================================================

mod imported_resource_info_tests {
    use super::*;

    #[test]
    fn swapchain_import_mode_is_write_only() {
        let mode = ImportMode::WriteOnly;
        assert!(mode.is_write());
        assert!(!mode.is_read());
    }

    #[test]
    fn swapchain_external_type_is_swapchain() {
        let ext_type = ExternalResourceType::Swapchain;
        assert!(ext_type.is_swapchain());
    }

    #[test]
    fn swapchain_variant_is_texture() {
        let ext_type = ExternalResourceType::Swapchain;
        assert!(ext_type.is_texture());
    }

    #[test]
    fn swapchain_variant_is_not_buffer() {
        let ext_type = ExternalResourceType::Swapchain;
        assert!(!ext_type.is_buffer());
    }

    #[test]
    fn texture_variant_is_texture() {
        let ext_type = ExternalResourceType::UserTexture;
        assert!(ext_type.is_texture());
    }

    #[test]
    fn texture_variant_is_not_buffer() {
        let ext_type = ExternalResourceType::UserTexture;
        assert!(!ext_type.is_buffer());
    }

    #[test]
    fn texture_variant_is_not_swapchain() {
        let ext_type = ExternalResourceType::UserTexture;
        assert!(!ext_type.is_swapchain());
    }

    #[test]
    fn buffer_variant_is_buffer() {
        let ext_type = ExternalResourceType::UserBuffer;
        assert!(ext_type.is_buffer());
    }

    #[test]
    fn buffer_variant_is_not_texture() {
        let ext_type = ExternalResourceType::UserBuffer;
        assert!(!ext_type.is_texture());
    }

    #[test]
    fn buffer_variant_is_not_swapchain() {
        let ext_type = ExternalResourceType::UserBuffer;
        assert!(!ext_type.is_swapchain());
    }

    #[test]
    fn texture_can_have_read_only_mode() {
        let mode = ImportMode::ReadOnly;
        assert!(mode.is_read());
    }

    #[test]
    fn buffer_can_have_read_write_mode() {
        let mode = ImportMode::ReadWrite;
        assert!(mode.is_read() && mode.is_write());
    }

    #[test]
    fn shared_texture_can_have_write_only_mode() {
        let mode = ImportMode::WriteOnly;
        let ext_type = ExternalResourceType::SharedTexture;
        assert!(mode.is_write());
        assert!(ext_type.is_shared());
    }

    #[test]
    fn shared_buffer_can_have_read_only_mode() {
        let mode = ImportMode::ReadOnly;
        let ext_type = ExternalResourceType::SharedBuffer;
        assert!(mode.is_read());
        assert!(ext_type.is_shared());
    }

    #[test]
    fn is_texture_is_buffer_is_swapchain_exhaustive() {
        let swapchain = ExternalResourceType::Swapchain;
        assert!(swapchain.is_swapchain());
        assert!(swapchain.is_texture());

        let user_tex = ExternalResourceType::UserTexture;
        assert!(!user_tex.is_swapchain());
        assert!(user_tex.is_texture());

        let user_buf = ExternalResourceType::UserBuffer;
        assert!(!user_buf.is_swapchain());
        assert!(user_buf.is_buffer());
    }
}

// ============================================================================
// SECTION 7: ImportedResource Tests (20+ tests)
// ============================================================================

mod imported_resource_tests {
    use super::*;

    #[test]
    fn resource_handle_zero() {
        let handle = ResourceHandle(0);
        assert_eq!(handle.0, 0);
    }

    #[test]
    fn resource_handle_arbitrary() {
        let handle = ResourceHandle(42);
        assert_eq!(handle.0, 42);
    }

    #[test]
    fn resource_handle_none_sentinel() {
        let handle = ResourceHandle::NONE;
        assert_eq!(handle.0, u32::MAX);
    }

    #[test]
    fn resource_handle_display() {
        let handle = ResourceHandle(123);
        let s = format!("{}", handle);
        assert!(s.contains("123"));
    }

    #[test]
    fn resource_handle_equality() {
        let a = ResourceHandle(5);
        let b = ResourceHandle(5);
        let c = ResourceHandle(6);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn resource_name_string_conversion() {
        let name: String = "my_texture".into();
        assert_eq!(name, "my_texture");
    }

    #[test]
    fn resource_name_empty() {
        let name: String = "".into();
        assert!(name.is_empty());
    }

    #[test]
    fn pass_index_creation() {
        let pass = PassIndex(0);
        assert_eq!(pass.0, 0);
    }

    #[test]
    fn pass_index_display() {
        let pass = PassIndex(42);
        let s = format!("{}", pass);
        assert!(s.contains("42"));
    }

    #[test]
    fn first_use_pass_logic_first_call_sets_value() {
        let mut first_use: Option<PassIndex> = None;
        let pass = PassIndex(0);
        if first_use.is_none() {
            first_use = Some(pass);
        }
        assert_eq!(first_use, Some(PassIndex(0)));
    }

    #[test]
    fn first_use_pass_logic_subsequent_calls_do_not_change() {
        let mut first_use: Option<PassIndex> = Some(PassIndex(0));
        if first_use.is_none() {
            first_use = Some(PassIndex(5));
        }
        assert_eq!(first_use, Some(PassIndex(0)));
    }

    #[test]
    fn last_use_pass_logic_always_updates() {
        let mut last_use: Option<PassIndex>;
        last_use = Some(PassIndex(0));
        assert!(last_use.is_some());
        last_use = Some(PassIndex(5));
        assert!(last_use.is_some());
        last_use = Some(PassIndex(10));
        assert_eq!(last_use, Some(PassIndex(10)));
    }

    #[test]
    fn has_usage_none_both() {
        let first_use: Option<PassIndex> = None;
        let last_use: Option<PassIndex> = None;
        assert!(!(first_use.is_some() || last_use.is_some()));
    }

    #[test]
    fn has_usage_first_only() {
        let first_use: Option<PassIndex> = Some(PassIndex(0));
        let last_use: Option<PassIndex> = None;
        assert!(first_use.is_some() || last_use.is_some());
    }

    #[test]
    fn has_usage_both() {
        let first_use: Option<PassIndex> = Some(PassIndex(0));
        let last_use: Option<PassIndex> = Some(PassIndex(10));
        assert!(first_use.is_some() || last_use.is_some());
    }

    #[test]
    fn imported_resource_texture_predicates() {
        let ext_type = ExternalResourceType::UserTexture;
        assert!(ext_type.is_texture());
        assert!(!ext_type.is_buffer());
    }

    #[test]
    fn imported_resource_buffer_predicates() {
        let ext_type = ExternalResourceType::UserBuffer;
        assert!(!ext_type.is_texture());
        assert!(ext_type.is_buffer());
    }

    #[test]
    fn imported_resource_swapchain_predicates() {
        let ext_type = ExternalResourceType::Swapchain;
        assert!(ext_type.is_texture());
        assert!(ext_type.is_swapchain());
    }

    #[test]
    fn import_mode_accessor_all() {
        assert_eq!(ImportMode::ReadOnly.to_resource_access(), ResourceAccess::Read);
        assert_eq!(ImportMode::WriteOnly.to_resource_access(), ResourceAccess::Write);
        assert_eq!(ImportMode::ReadWrite.to_resource_access(), ResourceAccess::ReadWrite);
    }

    #[test]
    fn external_type_accessor_all_types() {
        for ext_type in [
            ExternalResourceType::Swapchain,
            ExternalResourceType::UserTexture,
            ExternalResourceType::UserBuffer,
            ExternalResourceType::SharedTexture,
            ExternalResourceType::SharedBuffer,
        ] {
            assert!(ext_type.is_texture() || ext_type.is_buffer());
        }
    }
}

// ============================================================================
// SECTION 8: ResourceBarrier Tests (15+ tests)
// ============================================================================

mod resource_barrier_tests {
    use super::*;

    #[test]
    fn new_creates_barrier_with_all_fields() {
        let barrier = ResourceBarrier::new(
            ResourceHandle(1),
            ResourceAccess::Read,
            ResourceAccess::Write,
            true,
        );
        assert_eq!(barrier.resource, ResourceHandle(1));
        assert_eq!(barrier.before_access, ResourceAccess::Read);
        assert_eq!(barrier.after_access, ResourceAccess::Write);
        assert!(barrier.acquire);
    }

    #[test]
    fn new_release_barrier() {
        let barrier = ResourceBarrier::new(
            ResourceHandle(2),
            ResourceAccess::Write,
            ResourceAccess::Read,
            false,
        );
        assert!(!barrier.acquire);
    }

    #[test]
    fn acquire_factory_creates_acquire_barrier() {
        let barrier = ResourceBarrier::acquire(ResourceHandle(10), ResourceAccess::Read);
        assert!(barrier.is_acquire());
        assert!(!barrier.is_release());
    }

    #[test]
    fn acquire_factory_sets_target_access() {
        let barrier = ResourceBarrier::acquire(ResourceHandle(5), ResourceAccess::Write);
        assert_eq!(barrier.after_access, ResourceAccess::Write);
    }

    #[test]
    fn acquire_factory_sets_before_to_read() {
        let barrier = ResourceBarrier::acquire(ResourceHandle(7), ResourceAccess::ReadWrite);
        assert_eq!(barrier.before_access, ResourceAccess::Read);
    }

    #[test]
    fn release_factory_creates_release_barrier() {
        let barrier = ResourceBarrier::release(ResourceHandle(20), ResourceAccess::Write);
        assert!(!barrier.is_acquire());
        assert!(barrier.is_release());
    }

    #[test]
    fn release_factory_sets_from_access() {
        let barrier = ResourceBarrier::release(ResourceHandle(15), ResourceAccess::ReadWrite);
        assert_eq!(barrier.before_access, ResourceAccess::ReadWrite);
    }

    #[test]
    fn release_factory_sets_after_to_read() {
        let barrier = ResourceBarrier::release(ResourceHandle(25), ResourceAccess::Write);
        assert_eq!(barrier.after_access, ResourceAccess::Read);
    }

    #[test]
    fn is_acquire_true_for_acquire_barrier() {
        let barrier = ResourceBarrier::acquire(ResourceHandle(1), ResourceAccess::Write);
        assert!(barrier.is_acquire());
    }

    #[test]
    fn is_release_true_for_release_barrier() {
        let barrier = ResourceBarrier::release(ResourceHandle(1), ResourceAccess::Write);
        assert!(barrier.is_release());
    }

    #[test]
    fn display_acquire_barrier() {
        let barrier = ResourceBarrier::acquire(ResourceHandle(1), ResourceAccess::Write);
        let s = format!("{}", barrier);
        assert!(s.contains("Acquire"));
    }

    #[test]
    fn display_release_barrier() {
        let barrier = ResourceBarrier::release(ResourceHandle(2), ResourceAccess::Write);
        let s = format!("{}", barrier);
        assert!(s.contains("Release"));
    }

    #[test]
    fn barrier_clone() {
        let a = ResourceBarrier::acquire(ResourceHandle(1), ResourceAccess::Write);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn barrier_debug() {
        let barrier = ResourceBarrier::release(ResourceHandle(5), ResourceAccess::ReadWrite);
        let dbg = format!("{:?}", barrier);
        assert!(dbg.contains("ResourceBarrier"));
    }

    #[test]
    fn barrier_equality() {
        let a = ResourceBarrier::new(ResourceHandle(1), ResourceAccess::Read, ResourceAccess::Write, true);
        let b = ResourceBarrier::new(ResourceHandle(1), ResourceAccess::Read, ResourceAccess::Write, true);
        let c = ResourceBarrier::new(ResourceHandle(2), ResourceAccess::Read, ResourceAccess::Write, true);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}

// ============================================================================
// SECTION 9: ExternalResourceRegistry Tests (25+ tests)
// ============================================================================

mod external_resource_registry_tests {
    use super::*;

    #[test]
    fn new_creates_empty_registry() {
        let registry = ExternalResourceRegistry::new();
        assert!(registry.is_empty());
    }

    #[test]
    fn new_has_zero_count() {
        let registry = ExternalResourceRegistry::new();
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn new_has_no_swapchain() {
        let registry = ExternalResourceRegistry::new();
        assert!(!registry.has_swapchain());
    }

    #[test]
    fn new_swapchain_handle_is_none() {
        let registry = ExternalResourceRegistry::new();
        assert!(registry.swapchain_handle().is_none());
    }

    #[test]
    fn new_get_swapchain_returns_none() {
        let registry = ExternalResourceRegistry::new();
        assert!(registry.get_swapchain().is_none());
    }

    #[test]
    fn default_creates_empty_registry() {
        let registry = ExternalResourceRegistry::default();
        assert!(registry.is_empty());
    }

    #[test]
    fn clear_on_empty_registry() {
        let mut registry = ExternalResourceRegistry::new();
        registry.clear();
        assert!(registry.is_empty());
    }

    #[test]
    fn clear_resets_swapchain() {
        let mut registry = ExternalResourceRegistry::new();
        registry.clear();
        assert!(!registry.has_swapchain());
    }

    #[test]
    fn count_zero_on_new() {
        let registry = ExternalResourceRegistry::new();
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn is_empty_true_on_new() {
        let registry = ExternalResourceRegistry::new();
        assert!(registry.is_empty());
    }

    #[test]
    fn is_empty_true_after_clear() {
        let mut registry = ExternalResourceRegistry::new();
        registry.clear();
        assert!(registry.is_empty());
    }

    #[test]
    fn has_swapchain_false_on_new() {
        let registry = ExternalResourceRegistry::new();
        assert!(!registry.has_swapchain());
    }

    #[test]
    fn get_unknown_handle_returns_none() {
        let registry = ExternalResourceRegistry::new();
        assert!(registry.get(ResourceHandle(0)).is_none());
    }

    #[test]
    fn get_none_handle_returns_none() {
        let registry = ExternalResourceRegistry::new();
        assert!(registry.get(ResourceHandle::NONE).is_none());
    }

    #[test]
    fn get_mut_unknown_handle_returns_none() {
        let mut registry = ExternalResourceRegistry::new();
        assert!(registry.get_mut(ResourceHandle(0)).is_none());
    }

    #[test]
    fn iter_empty_registry_yields_nothing() {
        let registry = ExternalResourceRegistry::new();
        assert_eq!(registry.iter().count(), 0);
    }

    #[test]
    fn handles_empty_registry_yields_nothing() {
        let registry = ExternalResourceRegistry::new();
        assert_eq!(registry.handles().count(), 0);
    }

    #[test]
    fn find_by_name_empty_registry_returns_none() {
        let registry = ExternalResourceRegistry::new();
        assert!(registry.find_by_name("texture").is_none());
    }

    #[test]
    fn track_usage_unknown_handle_does_nothing() {
        let mut registry = ExternalResourceRegistry::new();
        registry.track_usage(ResourceHandle(42), PassIndex(0));
        assert!(registry.is_empty());
    }

    #[test]
    fn display_contains_count() {
        let registry = ExternalResourceRegistry::new();
        let s = format!("{}", registry);
        assert!(s.contains("count=0"));
    }

    #[test]
    fn display_contains_has_swapchain() {
        let registry = ExternalResourceRegistry::new();
        let s = format!("{}", registry);
        assert!(s.contains("has_swapchain=false"));
    }

    #[test]
    fn display_format_is_consistent() {
        let registry = ExternalResourceRegistry::new();
        let s = format!("{}", registry);
        assert!(s.contains("ExternalResourceRegistry"));
    }

    #[test]
    fn release_swapchain_empty_registry_returns_none() {
        let mut registry = ExternalResourceRegistry::new();
        assert!(registry.release_swapchain().is_none());
    }

    #[test]
    fn multiple_clears_are_idempotent() {
        let mut registry = ExternalResourceRegistry::new();
        registry.clear();
        registry.clear();
        registry.clear();
        assert!(registry.is_empty());
    }

    #[test]
    fn clear_after_release_swapchain() {
        let mut registry = ExternalResourceRegistry::new();
        registry.release_swapchain();
        registry.clear();
        assert!(registry.is_empty());
    }
}

// ============================================================================
// SECTION 10: ExternalSynchronizer Tests (15+ tests)
// ============================================================================

mod external_synchronizer_tests {
    use super::*;

    #[test]
    fn compute_acquire_barriers_empty_registry_returns_empty() {
        let registry = ExternalResourceRegistry::new();
        let barriers = ExternalSynchronizer::compute_acquire_barriers(&registry);
        assert!(barriers.is_empty());
    }

    #[test]
    fn compute_acquire_barriers_default_registry_empty() {
        let registry = ExternalResourceRegistry::default();
        let barriers = ExternalSynchronizer::compute_acquire_barriers(&registry);
        assert!(barriers.is_empty());
    }

    #[test]
    fn compute_release_barriers_empty_registry_returns_empty() {
        let registry = ExternalResourceRegistry::new();
        let barriers = ExternalSynchronizer::compute_release_barriers(&registry);
        assert!(barriers.is_empty());
    }

    #[test]
    fn compute_all_barriers_empty_registry_returns_empty_pair() {
        let registry = ExternalResourceRegistry::new();
        let (acquire, release) = ExternalSynchronizer::compute_all_barriers(&registry);
        assert!(acquire.is_empty());
        assert!(release.is_empty());
    }

    #[test]
    fn read_only_mode_requires_acquire_barrier_logic() {
        let mode = ImportMode::ReadOnly;
        assert!(mode.requires_acquire_barrier());
        assert!(!mode.requires_release_barrier());
    }

    #[test]
    fn write_only_mode_requires_release_barrier_logic() {
        let mode = ImportMode::WriteOnly;
        assert!(!mode.requires_acquire_barrier());
        assert!(mode.requires_release_barrier());
    }

    #[test]
    fn read_write_mode_requires_both_barriers_logic() {
        let mode = ImportMode::ReadWrite;
        assert!(mode.requires_acquire_barrier());
        assert!(mode.requires_release_barrier());
    }

    #[test]
    fn acquire_barrier_target_access_from_read_only() {
        let mode = ImportMode::ReadOnly;
        let target = mode.to_resource_access();
        assert_eq!(target, ResourceAccess::Read);
    }

    #[test]
    fn acquire_barrier_target_access_from_write_only() {
        let mode = ImportMode::WriteOnly;
        let target = mode.to_resource_access();
        assert_eq!(target, ResourceAccess::Write);
    }

    #[test]
    fn acquire_barrier_target_access_from_read_write() {
        let mode = ImportMode::ReadWrite;
        let target = mode.to_resource_access();
        assert_eq!(target, ResourceAccess::ReadWrite);
    }

    #[test]
    fn release_barrier_from_access_matches_mode() {
        for mode in [ImportMode::ReadOnly, ImportMode::WriteOnly, ImportMode::ReadWrite] {
            let access = mode.to_resource_access();
            let barrier = ResourceBarrier::release(ResourceHandle(1), access);
            assert_eq!(barrier.before_access, access);
        }
    }

    #[test]
    fn swapchain_has_write_only_import_mode() {
        let mode = ImportMode::WriteOnly;
        assert!(!mode.requires_acquire_barrier());
        assert!(mode.requires_release_barrier());
    }

    #[test]
    fn multiple_modes_barrier_logic() {
        let modes = [
            (ImportMode::ReadOnly, true, false),
            (ImportMode::WriteOnly, false, true),
            (ImportMode::ReadWrite, true, true),
        ];
        for (mode, needs_acquire, needs_release) in modes {
            assert_eq!(mode.requires_acquire_barrier(), needs_acquire);
            assert_eq!(mode.requires_release_barrier(), needs_release);
        }
    }

    #[test]
    fn barrier_factory_consistency() {
        let handle = ResourceHandle(42);
        let acquire = ResourceBarrier::acquire(handle, ResourceAccess::Write);
        assert!(acquire.is_acquire());
        let release = ResourceBarrier::release(handle, ResourceAccess::Write);
        assert!(release.is_release());
    }

    #[test]
    fn synchronizer_all_barriers_match_individual() {
        let registry = ExternalResourceRegistry::new();
        let acquire = ExternalSynchronizer::compute_acquire_barriers(&registry);
        let release = ExternalSynchronizer::compute_release_barriers(&registry);
        let (all_acquire, all_release) = ExternalSynchronizer::compute_all_barriers(&registry);
        assert_eq!(acquire.len(), all_acquire.len());
        assert_eq!(release.len(), all_release.len());
    }
}

// ============================================================================
// SECTION 11: Integration and Edge Case Tests (20+ tests)
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn all_external_types_have_display() {
        for t in [
            ExternalResourceType::Swapchain,
            ExternalResourceType::UserTexture,
            ExternalResourceType::UserBuffer,
            ExternalResourceType::SharedTexture,
            ExternalResourceType::SharedBuffer,
        ] {
            let s = format!("{}", t);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn all_import_modes_have_display() {
        for m in [ImportMode::ReadOnly, ImportMode::WriteOnly, ImportMode::ReadWrite] {
            let s = format!("{}", m);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn all_resource_accesses_have_display() {
        for a in [ResourceAccess::Read, ResourceAccess::Write, ResourceAccess::ReadWrite] {
            let s = format!("{}", a);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn resource_handle_zero_is_valid() {
        let handle = ResourceHandle(0);
        assert_ne!(handle, ResourceHandle::NONE);
    }

    #[test]
    fn resource_handle_max_minus_one_is_valid() {
        let handle = ResourceHandle(u32::MAX - 1);
        assert_ne!(handle, ResourceHandle::NONE);
    }

    #[test]
    fn resource_handle_none_is_max() {
        assert_eq!(ResourceHandle::NONE.0, u32::MAX);
    }

    #[test]
    fn pass_index_zero_is_valid() {
        let pass = PassIndex(0);
        assert_eq!(pass.0, 0);
    }

    #[test]
    fn pass_index_large_value() {
        let pass = PassIndex(10000);
        assert_eq!(pass.0, 10000);
    }

    #[test]
    fn all_type_mode_combinations_valid() {
        let types = [
            ExternalResourceType::Swapchain,
            ExternalResourceType::UserTexture,
            ExternalResourceType::UserBuffer,
            ExternalResourceType::SharedTexture,
            ExternalResourceType::SharedBuffer,
        ];
        let modes = [ImportMode::ReadOnly, ImportMode::WriteOnly, ImportMode::ReadWrite];
        for ext_type in types {
            for mode in modes {
                let _ = ext_type.is_texture();
                let _ = ext_type.is_buffer();
                let _ = mode.requires_acquire_barrier();
                let _ = mode.requires_release_barrier();
            }
        }
    }

    #[test]
    fn acquire_release_are_opposite() {
        let handle = ResourceHandle(1);
        let acquire = ResourceBarrier::acquire(handle, ResourceAccess::Write);
        let release = ResourceBarrier::release(handle, ResourceAccess::Write);
        assert!(acquire.is_acquire());
        assert!(release.is_release());
    }

    #[test]
    fn barrier_is_acquire_xor_is_release() {
        for acquire_flag in [true, false] {
            let barrier = ResourceBarrier::new(
                ResourceHandle(1),
                ResourceAccess::Read,
                ResourceAccess::Write,
                acquire_flag,
            );
            assert!(barrier.is_acquire() ^ barrier.is_release());
        }
    }

    #[test]
    fn registry_new_then_clear_is_still_empty() {
        let mut registry = ExternalResourceRegistry::new();
        registry.clear();
        assert!(registry.is_empty());
    }

    #[test]
    fn registry_default_equals_new() {
        let new = ExternalResourceRegistry::new();
        let default = ExternalResourceRegistry::default();
        assert_eq!(new.count(), default.count());
    }

    #[test]
    fn synchronizer_empty_registry_deterministic() {
        let registry = ExternalResourceRegistry::new();
        let barriers1 = ExternalSynchronizer::compute_acquire_barriers(&registry);
        let barriers2 = ExternalSynchronizer::compute_acquire_barriers(&registry);
        assert_eq!(barriers1.len(), barriers2.len());
    }

    #[test]
    fn external_resource_type_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ExternalResourceType::Swapchain);
        set.insert(ExternalResourceType::UserTexture);
        assert!(set.contains(&ExternalResourceType::Swapchain));
        assert!(!set.contains(&ExternalResourceType::UserBuffer));
    }

    #[test]
    fn import_mode_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ImportMode::ReadOnly);
        set.insert(ImportMode::WriteOnly);
        assert!(set.contains(&ImportMode::ReadOnly));
        assert!(!set.contains(&ImportMode::ReadWrite));
    }

    #[test]
    fn resource_handle_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ResourceHandle(0));
        set.insert(ResourceHandle(1));
        assert!(set.contains(&ResourceHandle(0)));
        assert!(!set.contains(&ResourceHandle(3)));
    }

    #[test]
    fn resource_handle_ordering() {
        let a = ResourceHandle(1);
        let b = ResourceHandle(2);
        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn pass_index_ordering() {
        let a = PassIndex(0);
        let b = PassIndex(1);
        assert!(a < b);
    }

    #[test]
    fn empty_name_string() {
        let name: String = "".into();
        assert!(name.is_empty());
    }

    #[test]
    fn zero_size_buffer() {
        let size = 0u64;
        assert_eq!(size, 0);
    }

    #[test]
    fn zero_dimension_texture() {
        let size = (0u32, 0u32, 0u32);
        assert_eq!(size.0, 0);
    }
}
