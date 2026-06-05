//! Whitebox tests for T-WGPU-P2.3.1: Texture Creation
//!
//! These tests have FULL ACCESS to the texture.rs implementation and thoroughly
//! test all public functions, types, and edge cases.
//!
//! Test Categories:
//! - bytes_per_pixel: Format byte sizes
//! - block_info: Compressed format block information
//! - calculate_mip_count: Mip level calculations
//! - estimate_texture_size: Memory estimation
//! - TrinityTextureDescriptor: Descriptor construction and defaults
//! - TrinityTexture: Wrapper struct and accessors
//! - Usage Presets: texture_usages module constants

use renderer_backend::resources::texture::{
    block_info, bytes_per_pixel, calculate_mip_count, calculate_mip_count_3d,
    estimate_texture_size, texture_usages, TextureCreationError, TrinityTextureDescriptor,
};
use wgpu::{AstcBlock, AstcChannel, Extent3d, TextureDimension, TextureFormat, TextureUsages};

// ============================================================================
// CATEGORY 1: bytes_per_pixel Tests (12 tests)
// ============================================================================

mod bytes_per_pixel_tests {
    use super::*;

    // --- 8-bit formats (1 byte) ---
    #[test]
    fn test_r8_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::R8Unorm), 1);
    }

    #[test]
    fn test_r8_snorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::R8Snorm), 1);
    }

    #[test]
    fn test_r8_uint() {
        assert_eq!(bytes_per_pixel(TextureFormat::R8Uint), 1);
    }

    #[test]
    fn test_r8_sint() {
        assert_eq!(bytes_per_pixel(TextureFormat::R8Sint), 1);
    }

    #[test]
    fn test_stencil8() {
        assert_eq!(bytes_per_pixel(TextureFormat::Stencil8), 1);
    }

    // --- 16-bit formats (2 bytes) ---
    #[test]
    fn test_r16_uint() {
        assert_eq!(bytes_per_pixel(TextureFormat::R16Uint), 2);
    }

    #[test]
    fn test_r16_sint() {
        assert_eq!(bytes_per_pixel(TextureFormat::R16Sint), 2);
    }

    #[test]
    fn test_r16_float() {
        assert_eq!(bytes_per_pixel(TextureFormat::R16Float), 2);
    }

    #[test]
    fn test_rg8_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rg8Unorm), 2);
    }

    #[test]
    fn test_rg8_snorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rg8Snorm), 2);
    }

    #[test]
    fn test_rg8_uint() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rg8Uint), 2);
    }

    #[test]
    fn test_rg8_sint() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rg8Sint), 2);
    }

    #[test]
    fn test_depth16_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Depth16Unorm), 2);
    }

    // --- 24-bit / 32-bit formats (4 bytes) ---
    #[test]
    fn test_depth24_plus() {
        // Implementation-dependent, assume 4
        assert_eq!(bytes_per_pixel(TextureFormat::Depth24Plus), 4);
    }

    #[test]
    fn test_r32_uint() {
        assert_eq!(bytes_per_pixel(TextureFormat::R32Uint), 4);
    }

    #[test]
    fn test_r32_sint() {
        assert_eq!(bytes_per_pixel(TextureFormat::R32Sint), 4);
    }

    #[test]
    fn test_r32_float() {
        assert_eq!(bytes_per_pixel(TextureFormat::R32Float), 4);
    }

    #[test]
    fn test_rg16_uint() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rg16Uint), 4);
    }

    #[test]
    fn test_rg16_sint() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rg16Sint), 4);
    }

    #[test]
    fn test_rg16_float() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rg16Float), 4);
    }

    #[test]
    fn test_rgba8_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba8Unorm), 4);
    }

    #[test]
    fn test_rgba8_unorm_srgb() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba8UnormSrgb), 4);
    }

    #[test]
    fn test_rgba8_snorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba8Snorm), 4);
    }

    #[test]
    fn test_rgba8_uint() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba8Uint), 4);
    }

    #[test]
    fn test_rgba8_sint() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba8Sint), 4);
    }

    #[test]
    fn test_bgra8_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Bgra8Unorm), 4);
    }

    #[test]
    fn test_bgra8_unorm_srgb() {
        assert_eq!(bytes_per_pixel(TextureFormat::Bgra8UnormSrgb), 4);
    }

    #[test]
    fn test_rgb9e5_ufloat() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rgb9e5Ufloat), 4);
    }

    #[test]
    fn test_rgb10a2_uint() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rgb10a2Uint), 4);
    }

    #[test]
    fn test_rgb10a2_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rgb10a2Unorm), 4);
    }

    #[test]
    fn test_rg11b10_float() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rg11b10Float), 4);
    }

    #[test]
    fn test_depth32_float() {
        assert_eq!(bytes_per_pixel(TextureFormat::Depth32Float), 4);
    }

    #[test]
    fn test_depth24_plus_stencil8() {
        assert_eq!(bytes_per_pixel(TextureFormat::Depth24PlusStencil8), 4);
    }

    // --- 64-bit formats (8 bytes) ---
    #[test]
    fn test_rg32_uint() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rg32Uint), 8);
    }

    #[test]
    fn test_rg32_sint() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rg32Sint), 8);
    }

    #[test]
    fn test_rg32_float() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rg32Float), 8);
    }

    #[test]
    fn test_rgba16_uint() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba16Uint), 8);
    }

    #[test]
    fn test_rgba16_sint() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba16Sint), 8);
    }

    #[test]
    fn test_rgba16_float() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba16Float), 8);
    }

    #[test]
    fn test_depth32_float_stencil8() {
        assert_eq!(bytes_per_pixel(TextureFormat::Depth32FloatStencil8), 8);
    }

    // --- 128-bit formats (16 bytes) ---
    #[test]
    fn test_rgba32_uint() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba32Uint), 16);
    }

    #[test]
    fn test_rgba32_sint() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba32Sint), 16);
    }

    #[test]
    fn test_rgba32_float() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba32Float), 16);
    }

    // --- BC compressed formats ---
    #[test]
    fn test_bc1_rgba_unorm() {
        // BC1: 8 bytes per 16 pixels, approximated as 1 byte/pixel
        assert_eq!(bytes_per_pixel(TextureFormat::Bc1RgbaUnorm), 1);
    }

    #[test]
    fn test_bc1_rgba_unorm_srgb() {
        assert_eq!(bytes_per_pixel(TextureFormat::Bc1RgbaUnormSrgb), 1);
    }

    #[test]
    fn test_bc2_rgba_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Bc2RgbaUnorm), 1);
    }

    #[test]
    fn test_bc3_rgba_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Bc3RgbaUnorm), 1);
    }

    #[test]
    fn test_bc4_r_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Bc4RUnorm), 1);
    }

    #[test]
    fn test_bc4_r_snorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Bc4RSnorm), 1);
    }

    #[test]
    fn test_bc5_rg_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Bc5RgUnorm), 1);
    }

    #[test]
    fn test_bc5_rg_snorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Bc5RgSnorm), 1);
    }

    #[test]
    fn test_bc6h_rgb_ufloat() {
        assert_eq!(bytes_per_pixel(TextureFormat::Bc6hRgbUfloat), 1);
    }

    #[test]
    fn test_bc6h_rgb_float() {
        assert_eq!(bytes_per_pixel(TextureFormat::Bc6hRgbFloat), 1);
    }

    #[test]
    fn test_bc7_rgba_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Bc7RgbaUnorm), 1);
    }

    #[test]
    fn test_bc7_rgba_unorm_srgb() {
        assert_eq!(bytes_per_pixel(TextureFormat::Bc7RgbaUnormSrgb), 1);
    }

    // --- ETC2 compressed formats ---
    #[test]
    fn test_etc2_rgb8_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Etc2Rgb8Unorm), 1);
    }

    #[test]
    fn test_etc2_rgb8_unorm_srgb() {
        assert_eq!(bytes_per_pixel(TextureFormat::Etc2Rgb8UnormSrgb), 1);
    }

    #[test]
    fn test_etc2_rgb8a1_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Etc2Rgb8A1Unorm), 1);
    }

    #[test]
    fn test_etc2_rgba8_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::Etc2Rgba8Unorm), 1);
    }

    #[test]
    fn test_eac_r11_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::EacR11Unorm), 1);
    }

    #[test]
    fn test_eac_rg11_unorm() {
        assert_eq!(bytes_per_pixel(TextureFormat::EacRg11Unorm), 1);
    }

    // --- ASTC compressed formats ---
    #[test]
    fn test_astc_4x4() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B4x4,
            channel: AstcChannel::Unorm,
        };
        assert_eq!(bytes_per_pixel(format), 1);
    }

    #[test]
    fn test_astc_8x8() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B8x8,
            channel: AstcChannel::UnormSrgb,
        };
        assert_eq!(bytes_per_pixel(format), 1);
    }

    #[test]
    fn test_astc_12x12() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B12x12,
            channel: AstcChannel::Hdr,
        };
        assert_eq!(bytes_per_pixel(format), 1);
    }
}

// ============================================================================
// CATEGORY 2: block_info Tests (15 tests)
// ============================================================================

mod block_info_tests {
    use super::*;

    // --- Uncompressed formats (1x1 blocks) ---
    #[test]
    fn test_block_info_r8_unorm() {
        assert_eq!(block_info(TextureFormat::R8Unorm), (1, 1, 1));
    }

    #[test]
    fn test_block_info_r8_snorm() {
        assert_eq!(block_info(TextureFormat::R8Snorm), (1, 1, 1));
    }

    #[test]
    fn test_block_info_r16_float() {
        assert_eq!(block_info(TextureFormat::R16Float), (2, 1, 1));
    }

    #[test]
    fn test_block_info_rg8_unorm() {
        assert_eq!(block_info(TextureFormat::Rg8Unorm), (2, 1, 1));
    }

    #[test]
    fn test_block_info_r32_float() {
        assert_eq!(block_info(TextureFormat::R32Float), (4, 1, 1));
    }

    #[test]
    fn test_block_info_rgba8_unorm() {
        assert_eq!(block_info(TextureFormat::Rgba8Unorm), (4, 1, 1));
    }

    #[test]
    fn test_block_info_rgba8_unorm_srgb() {
        assert_eq!(block_info(TextureFormat::Rgba8UnormSrgb), (4, 1, 1));
    }

    #[test]
    fn test_block_info_bgra8_unorm() {
        assert_eq!(block_info(TextureFormat::Bgra8Unorm), (4, 1, 1));
    }

    #[test]
    fn test_block_info_rgb10a2_unorm() {
        assert_eq!(block_info(TextureFormat::Rgb10a2Unorm), (4, 1, 1));
    }

    #[test]
    fn test_block_info_rg11b10_float() {
        assert_eq!(block_info(TextureFormat::Rg11b10Float), (4, 1, 1));
    }

    #[test]
    fn test_block_info_rg32_float() {
        assert_eq!(block_info(TextureFormat::Rg32Float), (8, 1, 1));
    }

    #[test]
    fn test_block_info_rgba16_float() {
        assert_eq!(block_info(TextureFormat::Rgba16Float), (8, 1, 1));
    }

    #[test]
    fn test_block_info_rgba32_float() {
        assert_eq!(block_info(TextureFormat::Rgba32Float), (16, 1, 1));
    }

    #[test]
    fn test_block_info_rgba32_uint() {
        assert_eq!(block_info(TextureFormat::Rgba32Uint), (16, 1, 1));
    }

    // --- Depth/stencil formats ---
    #[test]
    fn test_block_info_stencil8() {
        assert_eq!(block_info(TextureFormat::Stencil8), (1, 1, 1));
    }

    #[test]
    fn test_block_info_depth16_unorm() {
        assert_eq!(block_info(TextureFormat::Depth16Unorm), (2, 1, 1));
    }

    #[test]
    fn test_block_info_depth24_plus() {
        assert_eq!(block_info(TextureFormat::Depth24Plus), (4, 1, 1));
    }

    #[test]
    fn test_block_info_depth32_float() {
        assert_eq!(block_info(TextureFormat::Depth32Float), (4, 1, 1));
    }

    #[test]
    fn test_block_info_depth24_plus_stencil8() {
        assert_eq!(block_info(TextureFormat::Depth24PlusStencil8), (4, 1, 1));
    }

    #[test]
    fn test_block_info_depth32_float_stencil8() {
        assert_eq!(block_info(TextureFormat::Depth32FloatStencil8), (8, 1, 1));
    }

    // --- BC compressed formats (4x4 blocks) ---
    #[test]
    fn test_block_info_bc1_rgba_unorm() {
        // BC1: 8 bytes per 4x4 block
        assert_eq!(block_info(TextureFormat::Bc1RgbaUnorm), (8, 4, 4));
    }

    #[test]
    fn test_block_info_bc1_rgba_unorm_srgb() {
        assert_eq!(block_info(TextureFormat::Bc1RgbaUnormSrgb), (8, 4, 4));
    }

    #[test]
    fn test_block_info_bc2_rgba_unorm() {
        // BC2: 16 bytes per 4x4 block
        assert_eq!(block_info(TextureFormat::Bc2RgbaUnorm), (16, 4, 4));
    }

    #[test]
    fn test_block_info_bc3_rgba_unorm() {
        // BC3: 16 bytes per 4x4 block
        assert_eq!(block_info(TextureFormat::Bc3RgbaUnorm), (16, 4, 4));
    }

    #[test]
    fn test_block_info_bc4_r_unorm() {
        // BC4: 8 bytes per 4x4 block
        assert_eq!(block_info(TextureFormat::Bc4RUnorm), (8, 4, 4));
    }

    #[test]
    fn test_block_info_bc4_r_snorm() {
        assert_eq!(block_info(TextureFormat::Bc4RSnorm), (8, 4, 4));
    }

    #[test]
    fn test_block_info_bc5_rg_unorm() {
        // BC5: 16 bytes per 4x4 block
        assert_eq!(block_info(TextureFormat::Bc5RgUnorm), (16, 4, 4));
    }

    #[test]
    fn test_block_info_bc5_rg_snorm() {
        assert_eq!(block_info(TextureFormat::Bc5RgSnorm), (16, 4, 4));
    }

    #[test]
    fn test_block_info_bc6h_rgb_ufloat() {
        // BC6H: 16 bytes per 4x4 block
        assert_eq!(block_info(TextureFormat::Bc6hRgbUfloat), (16, 4, 4));
    }

    #[test]
    fn test_block_info_bc6h_rgb_float() {
        assert_eq!(block_info(TextureFormat::Bc6hRgbFloat), (16, 4, 4));
    }

    #[test]
    fn test_block_info_bc7_rgba_unorm() {
        // BC7: 16 bytes per 4x4 block
        assert_eq!(block_info(TextureFormat::Bc7RgbaUnorm), (16, 4, 4));
    }

    #[test]
    fn test_block_info_bc7_rgba_unorm_srgb() {
        assert_eq!(block_info(TextureFormat::Bc7RgbaUnormSrgb), (16, 4, 4));
    }

    // --- ETC2 compressed formats ---
    #[test]
    fn test_block_info_etc2_rgb8_unorm() {
        // ETC2 RGB: 8 bytes per 4x4 block
        assert_eq!(block_info(TextureFormat::Etc2Rgb8Unorm), (8, 4, 4));
    }

    #[test]
    fn test_block_info_etc2_rgb8a1_unorm() {
        assert_eq!(block_info(TextureFormat::Etc2Rgb8A1Unorm), (8, 4, 4));
    }

    #[test]
    fn test_block_info_etc2_rgba8_unorm() {
        // ETC2 RGBA: 16 bytes per 4x4 block
        assert_eq!(block_info(TextureFormat::Etc2Rgba8Unorm), (16, 4, 4));
    }

    #[test]
    fn test_block_info_eac_r11_unorm() {
        // EAC R11: 8 bytes per 4x4 block
        assert_eq!(block_info(TextureFormat::EacR11Unorm), (8, 4, 4));
    }

    #[test]
    fn test_block_info_eac_r11_snorm() {
        assert_eq!(block_info(TextureFormat::EacR11Snorm), (8, 4, 4));
    }

    #[test]
    fn test_block_info_eac_rg11_unorm() {
        // EAC RG11: 16 bytes per 4x4 block
        assert_eq!(block_info(TextureFormat::EacRg11Unorm), (16, 4, 4));
    }

    #[test]
    fn test_block_info_eac_rg11_snorm() {
        assert_eq!(block_info(TextureFormat::EacRg11Snorm), (16, 4, 4));
    }

    // --- ASTC compressed formats (various block sizes) ---
    #[test]
    fn test_block_info_astc_4x4() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B4x4,
            channel: AstcChannel::Unorm,
        };
        // ASTC: 16 bytes per block, block size varies
        assert_eq!(block_info(format), (16, 4, 4));
    }

    #[test]
    fn test_block_info_astc_5x4() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B5x4,
            channel: AstcChannel::Unorm,
        };
        assert_eq!(block_info(format), (16, 5, 4));
    }

    #[test]
    fn test_block_info_astc_5x5() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B5x5,
            channel: AstcChannel::UnormSrgb,
        };
        assert_eq!(block_info(format), (16, 5, 5));
    }

    #[test]
    fn test_block_info_astc_6x5() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B6x5,
            channel: AstcChannel::Unorm,
        };
        assert_eq!(block_info(format), (16, 6, 5));
    }

    #[test]
    fn test_block_info_astc_6x6() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B6x6,
            channel: AstcChannel::Unorm,
        };
        assert_eq!(block_info(format), (16, 6, 6));
    }

    #[test]
    fn test_block_info_astc_8x5() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B8x5,
            channel: AstcChannel::Unorm,
        };
        assert_eq!(block_info(format), (16, 8, 5));
    }

    #[test]
    fn test_block_info_astc_8x6() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B8x6,
            channel: AstcChannel::Unorm,
        };
        assert_eq!(block_info(format), (16, 8, 6));
    }

    #[test]
    fn test_block_info_astc_8x8() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B8x8,
            channel: AstcChannel::Unorm,
        };
        assert_eq!(block_info(format), (16, 8, 8));
    }

    #[test]
    fn test_block_info_astc_10x5() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B10x5,
            channel: AstcChannel::Unorm,
        };
        assert_eq!(block_info(format), (16, 10, 5));
    }

    #[test]
    fn test_block_info_astc_10x6() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B10x6,
            channel: AstcChannel::Unorm,
        };
        assert_eq!(block_info(format), (16, 10, 6));
    }

    #[test]
    fn test_block_info_astc_10x8() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B10x8,
            channel: AstcChannel::Unorm,
        };
        assert_eq!(block_info(format), (16, 10, 8));
    }

    #[test]
    fn test_block_info_astc_10x10() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B10x10,
            channel: AstcChannel::Unorm,
        };
        assert_eq!(block_info(format), (16, 10, 10));
    }

    #[test]
    fn test_block_info_astc_12x10() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B12x10,
            channel: AstcChannel::Unorm,
        };
        assert_eq!(block_info(format), (16, 12, 10));
    }

    #[test]
    fn test_block_info_astc_12x12() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B12x12,
            channel: AstcChannel::Hdr,
        };
        assert_eq!(block_info(format), (16, 12, 12));
    }

    // --- ASTC HDR channel ---
    #[test]
    fn test_block_info_astc_hdr_channel() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B4x4,
            channel: AstcChannel::Hdr,
        };
        assert_eq!(block_info(format), (16, 4, 4));
    }
}

// ============================================================================
// CATEGORY 3: calculate_mip_count Tests (12 tests)
// ============================================================================

mod calculate_mip_count_tests {
    use super::*;

    // --- Power of 2 textures ---
    #[test]
    fn test_mip_count_1x1() {
        assert_eq!(calculate_mip_count(1, 1), 1);
    }

    #[test]
    fn test_mip_count_2x2() {
        assert_eq!(calculate_mip_count(2, 2), 2);
    }

    #[test]
    fn test_mip_count_4x4() {
        assert_eq!(calculate_mip_count(4, 4), 3);
    }

    #[test]
    fn test_mip_count_8x8() {
        assert_eq!(calculate_mip_count(8, 8), 4);
    }

    #[test]
    fn test_mip_count_16x16() {
        assert_eq!(calculate_mip_count(16, 16), 5);
    }

    #[test]
    fn test_mip_count_32x32() {
        assert_eq!(calculate_mip_count(32, 32), 6);
    }

    #[test]
    fn test_mip_count_64x64() {
        assert_eq!(calculate_mip_count(64, 64), 7);
    }

    #[test]
    fn test_mip_count_128x128() {
        assert_eq!(calculate_mip_count(128, 128), 8);
    }

    #[test]
    fn test_mip_count_256x256() {
        assert_eq!(calculate_mip_count(256, 256), 9);
    }

    #[test]
    fn test_mip_count_512x512() {
        assert_eq!(calculate_mip_count(512, 512), 10);
    }

    #[test]
    fn test_mip_count_1024x1024() {
        assert_eq!(calculate_mip_count(1024, 1024), 11);
    }

    #[test]
    fn test_mip_count_2048x2048() {
        assert_eq!(calculate_mip_count(2048, 2048), 12);
    }

    #[test]
    fn test_mip_count_4096x4096() {
        assert_eq!(calculate_mip_count(4096, 4096), 13);
    }

    #[test]
    fn test_mip_count_8192x8192() {
        assert_eq!(calculate_mip_count(8192, 8192), 14);
    }

    #[test]
    fn test_mip_count_16384x16384() {
        assert_eq!(calculate_mip_count(16384, 16384), 15);
    }

    // --- Non-square textures ---
    #[test]
    fn test_mip_count_512x256() {
        // max = 512, so 10 mips
        assert_eq!(calculate_mip_count(512, 256), 10);
    }

    #[test]
    fn test_mip_count_256x512() {
        assert_eq!(calculate_mip_count(256, 512), 10);
    }

    #[test]
    fn test_mip_count_1024x512() {
        assert_eq!(calculate_mip_count(1024, 512), 11);
    }

    #[test]
    fn test_mip_count_2048x1() {
        // max = 2048, so 12 mips
        assert_eq!(calculate_mip_count(2048, 1), 12);
    }

    #[test]
    fn test_mip_count_1x2048() {
        assert_eq!(calculate_mip_count(1, 2048), 12);
    }

    #[test]
    fn test_mip_count_4096x1024() {
        assert_eq!(calculate_mip_count(4096, 1024), 13);
    }

    // --- Non-power of 2 textures ---
    #[test]
    fn test_mip_count_100x100() {
        // 100 is between 64 (2^6) and 128 (2^7)
        // log2(100) = 6.64, ceil = 7 mips
        assert_eq!(calculate_mip_count(100, 100), 7);
    }

    #[test]
    fn test_mip_count_1920x1080() {
        // 1920 is between 1024 (2^10) and 2048 (2^11)
        // log2(1920) = 10.9, so 11 mips
        assert_eq!(calculate_mip_count(1920, 1080), 11);
    }

    #[test]
    fn test_mip_count_1280x720() {
        // 1280 is between 1024 and 2048
        assert_eq!(calculate_mip_count(1280, 720), 11);
    }

    #[test]
    fn test_mip_count_3840x2160() {
        // 4K: 3840 is between 2048 and 4096
        assert_eq!(calculate_mip_count(3840, 2160), 12);
    }

    #[test]
    fn test_mip_count_300x200() {
        // 300 is between 256 and 512
        assert_eq!(calculate_mip_count(300, 200), 9);
    }

    #[test]
    fn test_mip_count_500x500() {
        // 500 is between 256 and 512
        assert_eq!(calculate_mip_count(500, 500), 9);
    }

    // --- Edge cases ---
    #[test]
    fn test_mip_count_zero_width() {
        // When one dim is zero, max(0, 256) = 256, so mips based on 256
        // Implementation uses max of width and height
        assert_eq!(calculate_mip_count(0, 256), 9);
    }

    #[test]
    fn test_mip_count_zero_height() {
        // max(256, 0) = 256
        assert_eq!(calculate_mip_count(256, 0), 9);
    }

    #[test]
    fn test_mip_count_both_zero() {
        // max(0, 0) = 0, special case returns 1
        assert_eq!(calculate_mip_count(0, 0), 1);
    }

    // --- 3D mip count ---
    #[test]
    fn test_mip_count_3d_64x64x64() {
        assert_eq!(calculate_mip_count_3d(64, 64, 64), 7);
    }

    #[test]
    fn test_mip_count_3d_256x128x64() {
        // max = 256
        assert_eq!(calculate_mip_count_3d(256, 128, 64), 9);
    }

    #[test]
    fn test_mip_count_3d_128x256x512() {
        // max = 512
        assert_eq!(calculate_mip_count_3d(128, 256, 512), 10);
    }

    #[test]
    fn test_mip_count_3d_1x1x1() {
        assert_eq!(calculate_mip_count_3d(1, 1, 1), 1);
    }

    #[test]
    fn test_mip_count_3d_1x1x1024() {
        // max = 1024
        assert_eq!(calculate_mip_count_3d(1, 1, 1024), 11);
    }

    #[test]
    fn test_mip_count_3d_zero() {
        assert_eq!(calculate_mip_count_3d(0, 0, 0), 1);
    }
}

// ============================================================================
// CATEGORY 4: estimate_texture_size Tests (15 tests)
// ============================================================================

mod estimate_texture_size_tests {
    use super::*;

    // --- Simple 2D texture (no mips) ---
    #[test]
    fn test_size_256x256_rgba8_no_mips() {
        let size = estimate_texture_size(
            Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            TextureFormat::Rgba8Unorm,
            1,
            1,
        );
        assert_eq!(size, 256 * 256 * 4);
    }

    #[test]
    fn test_size_512x512_rgba8_no_mips() {
        let size = estimate_texture_size(
            Extent3d { width: 512, height: 512, depth_or_array_layers: 1 },
            TextureFormat::Rgba8Unorm,
            1,
            1,
        );
        assert_eq!(size, 512 * 512 * 4);
    }

    #[test]
    fn test_size_1024x1024_rgba8_no_mips() {
        let size = estimate_texture_size(
            Extent3d { width: 1024, height: 1024, depth_or_array_layers: 1 },
            TextureFormat::Rgba8Unorm,
            1,
            1,
        );
        assert_eq!(size, 1024 * 1024 * 4); // 4 MB
    }

    #[test]
    fn test_size_1024x1024_r8_no_mips() {
        let size = estimate_texture_size(
            Extent3d { width: 1024, height: 1024, depth_or_array_layers: 1 },
            TextureFormat::R8Unorm,
            1,
            1,
        );
        assert_eq!(size, 1024 * 1024 * 1); // 1 MB
    }

    #[test]
    fn test_size_1024x1024_rgba16_no_mips() {
        let size = estimate_texture_size(
            Extent3d { width: 1024, height: 1024, depth_or_array_layers: 1 },
            TextureFormat::Rgba16Float,
            1,
            1,
        );
        assert_eq!(size, 1024 * 1024 * 8); // 8 MB
    }

    #[test]
    fn test_size_1024x1024_rgba32_no_mips() {
        let size = estimate_texture_size(
            Extent3d { width: 1024, height: 1024, depth_or_array_layers: 1 },
            TextureFormat::Rgba32Float,
            1,
            1,
        );
        assert_eq!(size, 1024 * 1024 * 16); // 16 MB
    }

    // --- With mip chain ---
    #[test]
    fn test_size_256x256_rgba8_full_mips() {
        // 256x256 = 262144 (base)
        // 128x128 = 65536, 64x64 = 16384, 32x32 = 4096, 16x16 = 1024,
        // 8x8 = 256, 4x4 = 64, 2x2 = 16, 1x1 = 4
        // Total = 262144 + 65536 + 16384 + 4096 + 1024 + 256 + 64 + 16 + 4 = 349524
        let size = estimate_texture_size(
            Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            TextureFormat::Rgba8Unorm,
            9,
            1,
        );
        assert_eq!(size, 349524);
    }

    #[test]
    fn test_size_1024x1024_rgba8_full_mips() {
        // Full mip chain for 1024x1024 (11 levels)
        // Base 4MB, total ~5.33MB
        let size = estimate_texture_size(
            Extent3d { width: 1024, height: 1024, depth_or_array_layers: 1 },
            TextureFormat::Rgba8Unorm,
            11,
            1,
        );
        // Sum of geometric series: base * (1 - (1/4)^11) / (1 - 1/4) = base * 4/3 * (1 - (1/4)^11)
        // Approximately 5,592,404 bytes
        assert!(size > 5_000_000 && size < 6_000_000);
    }

    #[test]
    fn test_size_with_partial_mips() {
        // Only 3 mip levels (base + 2)
        let size = estimate_texture_size(
            Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            TextureFormat::Rgba8Unorm,
            3,
            1,
        );
        // 256*256*4 + 128*128*4 + 64*64*4 = 262144 + 65536 + 16384 = 344064
        assert_eq!(size, 344064);
    }

    // --- Array textures ---
    #[test]
    fn test_size_256x256_rgba8_4_layers() {
        let size = estimate_texture_size(
            Extent3d { width: 256, height: 256, depth_or_array_layers: 4 },
            TextureFormat::Rgba8Unorm,
            1,
            1, // ignored, uses extent
        );
        assert_eq!(size, 256 * 256 * 4 * 4);
    }

    #[test]
    fn test_size_512x512_rgba8_6_layers_cubemap() {
        let size = estimate_texture_size(
            Extent3d { width: 512, height: 512, depth_or_array_layers: 6 },
            TextureFormat::Rgba8Unorm,
            1,
            1,
        );
        assert_eq!(size, 512 * 512 * 4 * 6);
    }

    #[test]
    fn test_size_256x256_rgba8_4_layers_with_mips() {
        let size = estimate_texture_size(
            Extent3d { width: 256, height: 256, depth_or_array_layers: 4 },
            TextureFormat::Rgba8Unorm,
            9,
            1,
        );
        // 349524 * 4 = 1398096
        assert_eq!(size, 349524 * 4);
    }

    // --- Compressed textures ---
    #[test]
    fn test_size_256x256_bc1() {
        // BC1: 8 bytes per 4x4 block = 0.5 bytes per pixel
        // 64x64 blocks * 8 bytes = 32768
        let size = estimate_texture_size(
            Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            TextureFormat::Bc1RgbaUnorm,
            1,
            1,
        );
        assert_eq!(size, 64 * 64 * 8);
    }

    #[test]
    fn test_size_256x256_bc3() {
        // BC3: 16 bytes per 4x4 block = 1 byte per pixel
        // 64x64 blocks * 16 bytes = 65536
        let size = estimate_texture_size(
            Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            TextureFormat::Bc3RgbaUnorm,
            1,
            1,
        );
        assert_eq!(size, 64 * 64 * 16);
    }

    #[test]
    fn test_size_1024x1024_bc7() {
        // BC7: 16 bytes per 4x4 block
        // 256x256 blocks * 16 bytes = 1048576
        let size = estimate_texture_size(
            Extent3d { width: 1024, height: 1024, depth_or_array_layers: 1 },
            TextureFormat::Bc7RgbaUnorm,
            1,
            1,
        );
        assert_eq!(size, 256 * 256 * 16);
    }

    #[test]
    fn test_size_256x256_bc1_with_mips() {
        // With mip chain
        let size = estimate_texture_size(
            Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            TextureFormat::Bc1RgbaUnorm,
            9,
            1,
        );
        // More than base, less than double
        assert!(size > 32768 && size < 65536);
    }

    // --- 3D textures ---
    #[test]
    fn test_size_64x64x64_rgba8() {
        // For 3D textures, depth_or_array_layers is depth
        let size = estimate_texture_size(
            Extent3d { width: 64, height: 64, depth_or_array_layers: 64 },
            TextureFormat::Rgba8Unorm,
            1,
            1,
        );
        // Note: estimate_texture_size treats depth as layers for 2D arrays
        // For true 3D volume: 64*64*64*4 = 1,048,576
        assert_eq!(size, 64 * 64 * 4 * 64);
    }

    // --- Non-power-of-2 dimensions ---
    #[test]
    fn test_size_100x100_rgba8() {
        let size = estimate_texture_size(
            Extent3d { width: 100, height: 100, depth_or_array_layers: 1 },
            TextureFormat::Rgba8Unorm,
            1,
            1,
        );
        assert_eq!(size, 100 * 100 * 4);
    }

    #[test]
    fn test_size_1920x1080_rgba8() {
        let size = estimate_texture_size(
            Extent3d { width: 1920, height: 1080, depth_or_array_layers: 1 },
            TextureFormat::Rgba8Unorm,
            1,
            1,
        );
        assert_eq!(size, 1920 * 1080 * 4); // ~8.3 MB
    }

    // --- Non-power-of-2 compressed (block alignment) ---
    #[test]
    fn test_size_100x100_bc1() {
        // 100/4 = 25 blocks, rounded up
        // (100 + 3) / 4 = 25 (no rounding needed since exact)
        let size = estimate_texture_size(
            Extent3d { width: 100, height: 100, depth_or_array_layers: 1 },
            TextureFormat::Bc1RgbaUnorm,
            1,
            1,
        );
        assert_eq!(size, 25 * 25 * 8);
    }

    #[test]
    fn test_size_103x103_bc1() {
        // 103/4 rounded up = 26 blocks
        let size = estimate_texture_size(
            Extent3d { width: 103, height: 103, depth_or_array_layers: 1 },
            TextureFormat::Bc1RgbaUnorm,
            1,
            1,
        );
        assert_eq!(size, 26 * 26 * 8);
    }

    // --- Depth formats ---
    #[test]
    fn test_size_1024x1024_depth32() {
        let size = estimate_texture_size(
            Extent3d { width: 1024, height: 1024, depth_or_array_layers: 1 },
            TextureFormat::Depth32Float,
            1,
            1,
        );
        assert_eq!(size, 1024 * 1024 * 4);
    }

    #[test]
    fn test_size_1024x1024_depth24_stencil8() {
        let size = estimate_texture_size(
            Extent3d { width: 1024, height: 1024, depth_or_array_layers: 1 },
            TextureFormat::Depth24PlusStencil8,
            1,
            1,
        );
        assert_eq!(size, 1024 * 1024 * 4);
    }

    #[test]
    fn test_size_1024x1024_depth32_stencil8() {
        let size = estimate_texture_size(
            Extent3d { width: 1024, height: 1024, depth_or_array_layers: 1 },
            TextureFormat::Depth32FloatStencil8,
            1,
            1,
        );
        assert_eq!(size, 1024 * 1024 * 8);
    }
}

// ============================================================================
// CATEGORY 5: TrinityTextureDescriptor Tests (10 tests)
// ============================================================================

mod descriptor_tests {
    use super::*;

    #[test]
    fn test_default_label() {
        let desc = TrinityTextureDescriptor::default();
        assert!(desc.label.is_none());
    }

    #[test]
    fn test_default_size() {
        let desc = TrinityTextureDescriptor::default();
        assert_eq!(desc.size.width, 1);
        assert_eq!(desc.size.height, 1);
        assert_eq!(desc.size.depth_or_array_layers, 1);
    }

    #[test]
    fn test_default_mip_level_count() {
        let desc = TrinityTextureDescriptor::default();
        assert_eq!(desc.mip_level_count, 1);
    }

    #[test]
    fn test_default_sample_count() {
        let desc = TrinityTextureDescriptor::default();
        assert_eq!(desc.sample_count, 1);
    }

    #[test]
    fn test_default_dimension() {
        let desc = TrinityTextureDescriptor::default();
        assert!(matches!(desc.dimension, TextureDimension::D2));
    }

    #[test]
    fn test_default_format() {
        let desc = TrinityTextureDescriptor::default();
        assert_eq!(desc.format, TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn test_default_usage() {
        let desc = TrinityTextureDescriptor::default();
        assert_eq!(desc.usage, texture_usages::SAMPLED);
    }

    #[test]
    fn test_default_view_formats() {
        let desc = TrinityTextureDescriptor::default();
        assert!(desc.view_formats.is_empty());
    }

    #[test]
    fn test_custom_descriptor() {
        let view_formats = &[TextureFormat::Rgba8Unorm];
        let desc = TrinityTextureDescriptor {
            label: Some("test_texture"),
            size: Extent3d { width: 512, height: 512, depth_or_array_layers: 1 },
            mip_level_count: 0, // Auto
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: texture_usages::RENDER_TARGET,
            view_formats,
        };

        assert_eq!(desc.label, Some("test_texture"));
        assert_eq!(desc.size.width, 512);
        assert_eq!(desc.size.height, 512);
        assert_eq!(desc.mip_level_count, 0);
        assert_eq!(desc.format, TextureFormat::Rgba8UnormSrgb);
        assert_eq!(desc.usage, texture_usages::RENDER_TARGET);
        assert_eq!(desc.view_formats.len(), 1);
    }

    #[test]
    fn test_descriptor_with_multiple_view_formats() {
        let view_formats = &[TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb];
        let desc = TrinityTextureDescriptor {
            view_formats,
            ..Default::default()
        };
        assert_eq!(desc.view_formats.len(), 2);
    }

    #[test]
    fn test_descriptor_clone() {
        let desc = TrinityTextureDescriptor {
            label: Some("original"),
            size: Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            ..Default::default()
        };

        let cloned = desc.clone();
        assert_eq!(cloned.label, desc.label);
        assert_eq!(cloned.size.width, desc.size.width);
    }

    #[test]
    fn test_descriptor_debug() {
        let desc = TrinityTextureDescriptor::default();
        let debug_str = format!("{:?}", desc);
        assert!(debug_str.contains("TrinityTextureDescriptor"));
    }

    #[test]
    fn test_descriptor_1d_texture() {
        let desc = TrinityTextureDescriptor {
            dimension: TextureDimension::D1,
            size: Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
            ..Default::default()
        };
        assert!(matches!(desc.dimension, TextureDimension::D1));
    }

    #[test]
    fn test_descriptor_3d_texture() {
        let desc = TrinityTextureDescriptor {
            dimension: TextureDimension::D3,
            size: Extent3d { width: 64, height: 64, depth_or_array_layers: 64 },
            ..Default::default()
        };
        assert!(matches!(desc.dimension, TextureDimension::D3));
    }

    #[test]
    fn test_descriptor_multisampled() {
        let desc = TrinityTextureDescriptor {
            sample_count: 4,
            mip_level_count: 1, // MSAA cannot have mips
            ..Default::default()
        };
        assert_eq!(desc.sample_count, 4);
    }

    #[test]
    fn test_descriptor_array_texture() {
        let desc = TrinityTextureDescriptor {
            size: Extent3d { width: 256, height: 256, depth_or_array_layers: 6 },
            ..Default::default()
        };
        assert_eq!(desc.size.depth_or_array_layers, 6);
    }
}

// ============================================================================
// CATEGORY 6: TrinityTexture (accessors only - no device) (8 tests)
// ============================================================================

mod trinity_texture_tests {
    use super::*;

    // Note: We cannot test TrinityTexture::from_raw without a wgpu device,
    // but we can test the TextureCreationError type

    #[test]
    fn test_error_zero_dimension_display() {
        let err = TextureCreationError::ZeroDimension;
        let msg = err.to_string();
        assert!(msg.contains("0") || msg.contains("zero"));
    }

    #[test]
    fn test_error_empty_usage_display() {
        let err = TextureCreationError::EmptyUsage;
        let msg = err.to_string();
        assert!(msg.contains("usage"));
    }

    #[test]
    fn test_error_multisample_with_mipmaps_display() {
        let err = TextureCreationError::MultisampleWithMipmaps;
        let msg = err.to_string();
        assert!(msg.contains("multisample") || msg.contains("mipmap"));
    }

    #[test]
    fn test_error_invalid_view_format_display() {
        let err = TextureCreationError::InvalidViewFormat(TextureFormat::Rgba8Unorm);
        let msg = err.to_string();
        assert!(msg.contains("view format"));
    }

    #[test]
    fn test_error_debug() {
        let err = TextureCreationError::ZeroDimension;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("ZeroDimension"));
    }

    #[test]
    fn test_error_clone() {
        let err = TextureCreationError::MultisampleWithMipmaps;
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_error_eq() {
        assert_eq!(TextureCreationError::ZeroDimension, TextureCreationError::ZeroDimension);
        assert_ne!(TextureCreationError::ZeroDimension, TextureCreationError::EmptyUsage);
    }

    #[test]
    fn test_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(TextureCreationError::EmptyUsage);
        assert!(err.to_string().contains("usage"));
    }
}

// ============================================================================
// CATEGORY 7: texture_usages Presets Tests (8 tests)
// ============================================================================

mod usage_presets_tests {
    use super::*;

    #[test]
    fn test_sampled_has_texture_binding() {
        assert!(texture_usages::SAMPLED.contains(TextureUsages::TEXTURE_BINDING));
    }

    #[test]
    fn test_sampled_has_copy_dst() {
        assert!(texture_usages::SAMPLED.contains(TextureUsages::COPY_DST));
    }

    #[test]
    fn test_storage_has_storage_binding() {
        assert!(texture_usages::STORAGE.contains(TextureUsages::STORAGE_BINDING));
    }

    #[test]
    fn test_storage_has_copy_dst() {
        assert!(texture_usages::STORAGE.contains(TextureUsages::COPY_DST));
    }

    #[test]
    fn test_render_target_has_render_attachment() {
        assert!(texture_usages::RENDER_TARGET.contains(TextureUsages::RENDER_ATTACHMENT));
    }

    #[test]
    fn test_render_target_has_texture_binding() {
        assert!(texture_usages::RENDER_TARGET.contains(TextureUsages::TEXTURE_BINDING));
    }

    #[test]
    fn test_render_target_copy_has_render_attachment() {
        assert!(texture_usages::RENDER_TARGET_COPY.contains(TextureUsages::RENDER_ATTACHMENT));
    }

    #[test]
    fn test_render_target_copy_has_copy_src() {
        assert!(texture_usages::RENDER_TARGET_COPY.contains(TextureUsages::COPY_SRC));
    }

    #[test]
    fn test_depth_target_same_as_render_target() {
        // DEPTH_TARGET has same flags as RENDER_TARGET
        assert_eq!(texture_usages::DEPTH_TARGET, texture_usages::RENDER_TARGET);
    }

    #[test]
    fn test_full_has_texture_binding() {
        assert!(texture_usages::FULL.contains(TextureUsages::TEXTURE_BINDING));
    }

    #[test]
    fn test_full_has_storage_binding() {
        assert!(texture_usages::FULL.contains(TextureUsages::STORAGE_BINDING));
    }

    #[test]
    fn test_full_has_render_attachment() {
        assert!(texture_usages::FULL.contains(TextureUsages::RENDER_ATTACHMENT));
    }

    #[test]
    fn test_full_has_copy_src() {
        assert!(texture_usages::FULL.contains(TextureUsages::COPY_SRC));
    }

    #[test]
    fn test_full_has_copy_dst() {
        assert!(texture_usages::FULL.contains(TextureUsages::COPY_DST));
    }

    #[test]
    fn test_sampled_does_not_have_render_attachment() {
        assert!(!texture_usages::SAMPLED.contains(TextureUsages::RENDER_ATTACHMENT));
    }

    #[test]
    fn test_storage_does_not_have_render_attachment() {
        assert!(!texture_usages::STORAGE.contains(TextureUsages::RENDER_ATTACHMENT));
    }

    #[test]
    fn test_render_target_does_not_have_storage() {
        assert!(!texture_usages::RENDER_TARGET.contains(TextureUsages::STORAGE_BINDING));
    }
}

// ============================================================================
// Additional Edge Case Tests
// ============================================================================

mod edge_case_tests {
    use super::*;

    // --- Very large textures ---
    #[test]
    fn test_large_texture_mip_count() {
        // 32K texture
        assert_eq!(calculate_mip_count(32768, 32768), 16);
    }

    #[test]
    fn test_large_texture_size_estimation() {
        // 4K texture with mips
        let size = estimate_texture_size(
            Extent3d { width: 4096, height: 4096, depth_or_array_layers: 1 },
            TextureFormat::Rgba8Unorm,
            13,
            1,
        );
        // Should be approximately 89.5 MB
        assert!(size > 85_000_000 && size < 95_000_000);
    }

    // --- Minimum dimension textures ---
    #[test]
    fn test_1x1_texture() {
        let size = estimate_texture_size(
            Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            TextureFormat::Rgba8Unorm,
            1,
            1,
        );
        assert_eq!(size, 4);
    }

    #[test]
    fn test_1x1_rgba32() {
        let size = estimate_texture_size(
            Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            TextureFormat::Rgba32Float,
            1,
            1,
        );
        assert_eq!(size, 16);
    }

    // --- Compressed texture edge cases ---
    #[test]
    fn test_1x1_bc1() {
        // Even 1x1 needs one full block (4x4)
        let size = estimate_texture_size(
            Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            TextureFormat::Bc1RgbaUnorm,
            1,
            1,
        );
        assert_eq!(size, 8); // One 4x4 block = 8 bytes for BC1
    }

    #[test]
    fn test_3x3_bc1() {
        // 3x3 rounds up to one 4x4 block
        let size = estimate_texture_size(
            Extent3d { width: 3, height: 3, depth_or_array_layers: 1 },
            TextureFormat::Bc1RgbaUnorm,
            1,
            1,
        );
        assert_eq!(size, 8);
    }

    #[test]
    fn test_5x5_bc1() {
        // 5x5 rounds up to 2x2 blocks
        let size = estimate_texture_size(
            Extent3d { width: 5, height: 5, depth_or_array_layers: 1 },
            TextureFormat::Bc1RgbaUnorm,
            1,
            1,
        );
        assert_eq!(size, 2 * 2 * 8);
    }

    // --- ASTC edge cases ---
    #[test]
    fn test_astc_12x12_tiny() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B12x12,
            channel: AstcChannel::Unorm,
        };
        // 12x12 texture = exactly one block
        let size = estimate_texture_size(
            Extent3d { width: 12, height: 12, depth_or_array_layers: 1 },
            format,
            1,
            1,
        );
        assert_eq!(size, 16); // One ASTC block
    }

    #[test]
    fn test_astc_12x12_large() {
        let format = TextureFormat::Astc {
            block: AstcBlock::B12x12,
            channel: AstcChannel::Unorm,
        };
        // 1024x1024 with 12x12 blocks
        // (1024 + 11) / 12 = 86 blocks per dimension
        let size = estimate_texture_size(
            Extent3d { width: 1024, height: 1024, depth_or_array_layers: 1 },
            format,
            1,
            1,
        );
        assert_eq!(size, 86 * 86 * 16);
    }

    // --- Mip chain edge cases ---
    #[test]
    fn test_mip_chain_odd_dimensions() {
        // 511x511 texture
        let mips = calculate_mip_count(511, 511);
        assert_eq!(mips, 9); // 511 < 512 = 2^9
    }

    #[test]
    fn test_estimate_size_mip_chain_non_pow2() {
        // 300x300 with full mips
        let mips = calculate_mip_count(300, 300);
        let size = estimate_texture_size(
            Extent3d { width: 300, height: 300, depth_or_array_layers: 1 },
            TextureFormat::Rgba8Unorm,
            mips,
            1,
        );
        // Should be about 1.33x base
        let base = 300 * 300 * 4;
        assert!(size > base && size < base * 2);
    }
}

// ============================================================================
// Format Consistency Tests
// ============================================================================

mod format_consistency_tests {
    use super::*;

    // Verify block_info and bytes_per_pixel are consistent for uncompressed
    #[test]
    fn test_consistency_r8() {
        let (bytes, w, h) = block_info(TextureFormat::R8Unorm);
        assert_eq!(bytes / (w * h), bytes_per_pixel(TextureFormat::R8Unorm));
    }

    #[test]
    fn test_consistency_rgba8() {
        let (bytes, w, h) = block_info(TextureFormat::Rgba8Unorm);
        assert_eq!(bytes / (w * h), bytes_per_pixel(TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn test_consistency_rgba16() {
        let (bytes, w, h) = block_info(TextureFormat::Rgba16Float);
        assert_eq!(bytes / (w * h), bytes_per_pixel(TextureFormat::Rgba16Float));
    }

    #[test]
    fn test_consistency_rgba32() {
        let (bytes, w, h) = block_info(TextureFormat::Rgba32Float);
        assert_eq!(bytes / (w * h), bytes_per_pixel(TextureFormat::Rgba32Float));
    }

    #[test]
    fn test_consistency_depth32() {
        let (bytes, w, h) = block_info(TextureFormat::Depth32Float);
        assert_eq!(bytes / (w * h), bytes_per_pixel(TextureFormat::Depth32Float));
    }
}
