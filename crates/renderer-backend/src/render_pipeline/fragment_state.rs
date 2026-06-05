//! Fragment state and blend descriptors for render pipelines.

// ---------------------------------------------------------------------------
// BlendComponentDescriptor
// ---------------------------------------------------------------------------

/// Describes blending for a single color component (color or alpha).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendComponentDescriptor {
    /// Source blend factor.
    pub src_factor: wgpu::BlendFactor,
    /// Destination blend factor.
    pub dst_factor: wgpu::BlendFactor,
    /// Blend operation.
    pub operation: wgpu::BlendOperation,
}

impl Default for BlendComponentDescriptor {
    fn default() -> Self {
        Self {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::Zero,
            operation: wgpu::BlendOperation::Add,
        }
    }
}

impl BlendComponentDescriptor {
    /// Create with replace blending (no blending).
    pub fn replace() -> Self {
        Self::default()
    }

    /// Create standard alpha blending.
    pub fn alpha_blend() -> Self {
        Self {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        }
    }

    /// Create premultiplied alpha blending.
    pub fn premultiplied_alpha() -> Self {
        Self {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        }
    }

    /// Create additive blending.
    pub fn additive() -> Self {
        Self {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        }
    }
}

impl From<BlendComponentDescriptor> for wgpu::BlendComponent {
    fn from(desc: BlendComponentDescriptor) -> Self {
        wgpu::BlendComponent {
            src_factor: desc.src_factor,
            dst_factor: desc.dst_factor,
            operation: desc.operation,
        }
    }
}

// ---------------------------------------------------------------------------
// BlendStateDescriptor
// ---------------------------------------------------------------------------

/// Describes the blend state for a color target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendStateDescriptor {
    /// Blending for RGB components.
    pub color: BlendComponentDescriptor,
    /// Blending for alpha component.
    pub alpha: BlendComponentDescriptor,
}

impl BlendStateDescriptor {
    /// Create with no blending (replace).
    pub fn replace() -> Self {
        Self {
            color: BlendComponentDescriptor::replace(),
            alpha: BlendComponentDescriptor::replace(),
        }
    }

    /// Create standard alpha blending.
    pub fn alpha_blend() -> Self {
        Self {
            color: BlendComponentDescriptor::alpha_blend(),
            alpha: BlendComponentDescriptor {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }
    }

    /// Create premultiplied alpha blending.
    pub fn premultiplied_alpha() -> Self {
        Self {
            color: BlendComponentDescriptor::premultiplied_alpha(),
            alpha: BlendComponentDescriptor::premultiplied_alpha(),
        }
    }

    /// Create additive blending.
    pub fn additive() -> Self {
        Self {
            color: BlendComponentDescriptor::additive(),
            alpha: BlendComponentDescriptor::additive(),
        }
    }
}

impl From<BlendStateDescriptor> for wgpu::BlendState {
    fn from(desc: BlendStateDescriptor) -> Self {
        wgpu::BlendState {
            color: desc.color.into(),
            alpha: desc.alpha.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// ColorTargetStateDescriptor
// ---------------------------------------------------------------------------

/// Describes a single color target (render target attachment).
#[derive(Debug, Clone, PartialEq)]
pub struct ColorTargetStateDescriptor {
    /// Texture format of the target.
    pub format: wgpu::TextureFormat,
    /// Optional blend state.
    pub blend: Option<BlendStateDescriptor>,
    /// Color write mask.
    pub write_mask: wgpu::ColorWrites,
}

impl ColorTargetStateDescriptor {
    /// Create a new color target with the given format.
    pub fn new(format: wgpu::TextureFormat) -> Self {
        Self {
            format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }
    }

    /// Set the blend state.
    pub fn blend(mut self, blend: BlendStateDescriptor) -> Self {
        self.blend = Some(blend);
        self
    }

    /// Enable alpha blending.
    pub fn alpha_blend(mut self) -> Self {
        self.blend = Some(BlendStateDescriptor::alpha_blend());
        self
    }

    /// Enable premultiplied alpha blending.
    pub fn premultiplied_alpha(mut self) -> Self {
        self.blend = Some(BlendStateDescriptor::premultiplied_alpha());
        self
    }

    /// Enable additive blending.
    pub fn additive(mut self) -> Self {
        self.blend = Some(BlendStateDescriptor::additive());
        self
    }

    /// Set the color write mask.
    pub fn write_mask(mut self, mask: wgpu::ColorWrites) -> Self {
        self.write_mask = mask;
        self
    }

    /// Create an sRGB target with no blending.
    pub fn srgb() -> Self {
        Self::new(wgpu::TextureFormat::Bgra8UnormSrgb)
    }

    /// Create an HDR target (Rgba16Float).
    pub fn hdr() -> Self {
        Self::new(wgpu::TextureFormat::Rgba16Float)
    }
}

// ---------------------------------------------------------------------------
// FragmentStateDescriptor
// ---------------------------------------------------------------------------

/// Describes the fragment stage of a render pipeline.
///
/// # Required Fields
///
/// - `module`: The fragment shader module (required at construction)
///
/// # Optional Fields
///
/// - `entry_point`: Entry function name (default: `"fs_main"`)
/// - `compilation_options`: Shader compilation options (default: empty)
/// - `targets`: Color target states (default: empty)
#[derive(Debug, Clone)]
pub struct FragmentStateDescriptor<'a> {
    /// The fragment shader module.
    pub module: &'a wgpu::ShaderModule,
    /// The entry point function name.
    pub entry_point: &'a str,
    /// Shader compilation options.
    pub compilation_options: wgpu::PipelineCompilationOptions<'a>,
    /// Color render targets.
    pub targets: Vec<Option<ColorTargetStateDescriptor>>,
}

impl<'a> FragmentStateDescriptor<'a> {
    /// Create a new fragment state descriptor with the given shader module.
    ///
    /// Uses `"fs_main"` as the default entry point.
    pub fn new(module: &'a wgpu::ShaderModule) -> Self {
        Self {
            module,
            entry_point: "fs_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: Vec::new(),
        }
    }

    /// Set the entry point function name.
    pub fn entry_point(mut self, entry_point: &'a str) -> Self {
        self.entry_point = entry_point;
        self
    }

    /// Set the shader compilation options.
    pub fn compilation_options(
        mut self,
        options: wgpu::PipelineCompilationOptions<'a>,
    ) -> Self {
        self.compilation_options = options;
        self
    }

    /// Add a color target with the given format.
    pub fn target(mut self, format: wgpu::TextureFormat) -> Self {
        self.targets
            .push(Some(ColorTargetStateDescriptor::new(format)));
        self
    }

    /// Add a color target with a descriptor.
    pub fn target_state(mut self, target: ColorTargetStateDescriptor) -> Self {
        self.targets.push(Some(target));
        self
    }

    /// Add a null color target (for depth-only passes).
    pub fn null_target(mut self) -> Self {
        self.targets.push(None);
        self
    }

    /// Set all color targets at once.
    pub fn targets(mut self, targets: Vec<Option<ColorTargetStateDescriptor>>) -> Self {
        self.targets = targets;
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blend_component_presets() {
        let replace = BlendComponentDescriptor::replace();
        assert_eq!(replace.src_factor, wgpu::BlendFactor::One);
        assert_eq!(replace.dst_factor, wgpu::BlendFactor::Zero);

        let alpha = BlendComponentDescriptor::alpha_blend();
        assert_eq!(alpha.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(alpha.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);

        let additive = BlendComponentDescriptor::additive();
        assert_eq!(additive.src_factor, wgpu::BlendFactor::One);
        assert_eq!(additive.dst_factor, wgpu::BlendFactor::One);
    }

    #[test]
    fn test_blend_state_presets() {
        let alpha = BlendStateDescriptor::alpha_blend();
        assert_eq!(alpha.color.src_factor, wgpu::BlendFactor::SrcAlpha);

        let premul = BlendStateDescriptor::premultiplied_alpha();
        assert_eq!(premul.color.src_factor, wgpu::BlendFactor::One);

        let additive = BlendStateDescriptor::additive();
        assert_eq!(additive.color.dst_factor, wgpu::BlendFactor::One);
    }

    #[test]
    fn test_color_target_builder() {
        let target =
            ColorTargetStateDescriptor::new(wgpu::TextureFormat::Bgra8UnormSrgb).alpha_blend();
        assert_eq!(target.format, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert!(target.blend.is_some());
    }

    #[test]
    fn test_color_target_presets() {
        let srgb = ColorTargetStateDescriptor::srgb();
        assert_eq!(srgb.format, wgpu::TextureFormat::Bgra8UnormSrgb);

        let hdr = ColorTargetStateDescriptor::hdr();
        assert_eq!(hdr.format, wgpu::TextureFormat::Rgba16Float);
    }

    #[test]
    fn test_blend_into_wgpu() {
        let blend = BlendStateDescriptor::alpha_blend();
        let wgpu_blend: wgpu::BlendState = blend.into();
        assert_eq!(wgpu_blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
    }

    // -------------------------------------------------------------------------
    // Additional Whitebox Tests - Blend Factors
    // -------------------------------------------------------------------------

    #[test]
    fn test_all_blend_factors_src() {
        // Test all BlendFactor variants as source factor
        let factors = [
            wgpu::BlendFactor::Zero,
            wgpu::BlendFactor::One,
            wgpu::BlendFactor::Src,
            wgpu::BlendFactor::OneMinusSrc,
            wgpu::BlendFactor::SrcAlpha,
            wgpu::BlendFactor::OneMinusSrcAlpha,
            wgpu::BlendFactor::Dst,
            wgpu::BlendFactor::OneMinusDst,
            wgpu::BlendFactor::DstAlpha,
            wgpu::BlendFactor::OneMinusDstAlpha,
            wgpu::BlendFactor::SrcAlphaSaturated,
            wgpu::BlendFactor::Constant,
            wgpu::BlendFactor::OneMinusConstant,
            wgpu::BlendFactor::Src1,
            wgpu::BlendFactor::OneMinusSrc1,
            wgpu::BlendFactor::Src1Alpha,
            wgpu::BlendFactor::OneMinusSrc1Alpha,
        ];

        for factor in factors {
            let component = BlendComponentDescriptor {
                src_factor: factor,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            };
            assert_eq!(component.src_factor, factor);
        }
    }

    #[test]
    fn test_all_blend_operations() {
        // Test all BlendOperation variants
        let ops = [
            wgpu::BlendOperation::Add,
            wgpu::BlendOperation::Subtract,
            wgpu::BlendOperation::ReverseSubtract,
            wgpu::BlendOperation::Min,
            wgpu::BlendOperation::Max,
        ];

        for op in ops {
            let component = BlendComponentDescriptor {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: op,
            };
            assert_eq!(component.operation, op);
        }
    }

    #[test]
    fn test_write_mask_all() {
        let target = ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm)
            .write_mask(wgpu::ColorWrites::ALL);
        assert_eq!(target.write_mask, wgpu::ColorWrites::ALL);
    }

    #[test]
    fn test_write_mask_none() {
        let target = ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm)
            .write_mask(wgpu::ColorWrites::empty());
        assert_eq!(target.write_mask, wgpu::ColorWrites::empty());
    }

    #[test]
    fn test_write_mask_red_only() {
        let target = ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm)
            .write_mask(wgpu::ColorWrites::RED);
        assert_eq!(target.write_mask, wgpu::ColorWrites::RED);
    }

    #[test]
    fn test_write_mask_green_only() {
        let target = ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm)
            .write_mask(wgpu::ColorWrites::GREEN);
        assert_eq!(target.write_mask, wgpu::ColorWrites::GREEN);
    }

    #[test]
    fn test_write_mask_blue_only() {
        let target = ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm)
            .write_mask(wgpu::ColorWrites::BLUE);
        assert_eq!(target.write_mask, wgpu::ColorWrites::BLUE);
    }

    #[test]
    fn test_write_mask_alpha_only() {
        let target = ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm)
            .write_mask(wgpu::ColorWrites::ALPHA);
        assert_eq!(target.write_mask, wgpu::ColorWrites::ALPHA);
    }

    #[test]
    fn test_write_mask_rgb_no_alpha() {
        let target = ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm)
            .write_mask(wgpu::ColorWrites::COLOR);
        assert_eq!(target.write_mask, wgpu::ColorWrites::COLOR);
    }

    #[test]
    fn test_write_mask_combination() {
        let target = ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm)
            .write_mask(wgpu::ColorWrites::RED | wgpu::ColorWrites::BLUE);
        assert!(target.write_mask.contains(wgpu::ColorWrites::RED));
        assert!(target.write_mask.contains(wgpu::ColorWrites::BLUE));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::ALPHA));
    }

    #[test]
    fn test_premultiplied_alpha_preset() {
        let target = ColorTargetStateDescriptor::srgb().premultiplied_alpha();
        assert!(target.blend.is_some());
        let blend = target.blend.unwrap();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    }

    #[test]
    fn test_additive_blend_target() {
        let target = ColorTargetStateDescriptor::srgb().additive();
        assert!(target.blend.is_some());
        let blend = target.blend.unwrap();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::One);
    }

    #[test]
    fn test_color_vs_alpha_blend_separation() {
        let blend = BlendStateDescriptor::alpha_blend();
        // Color uses SrcAlpha
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        // Alpha uses One (to preserve alpha in framebuffer)
        assert_eq!(blend.alpha.src_factor, wgpu::BlendFactor::One);
    }

    #[test]
    fn test_mrt_single_target() {
        // Multiple Render Targets - single target
        let targets = vec![Some(ColorTargetStateDescriptor::srgb())];
        assert_eq!(targets.len(), 1);
    }

    #[test]
    fn test_mrt_two_targets() {
        // MRT with 2 targets (common for deferred shading)
        let targets = vec![
            Some(ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm)), // Albedo
            Some(ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba16Float)), // Normal
        ];
        assert_eq!(targets.len(), 2);
    }

    #[test]
    fn test_mrt_four_targets() {
        // MRT with 4 targets (G-buffer)
        let targets = vec![
            Some(ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm)),   // Albedo
            Some(ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba16Float)),  // Normal
            Some(ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm)),   // Material
            Some(ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba32Float)),  // Position
        ];
        assert_eq!(targets.len(), 4);
        assert_eq!(targets[0].as_ref().unwrap().format, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(targets[1].as_ref().unwrap().format, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(targets[2].as_ref().unwrap().format, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(targets[3].as_ref().unwrap().format, wgpu::TextureFormat::Rgba32Float);
    }

    #[test]
    fn test_mrt_eight_targets() {
        // MRT with 8 targets (max for most GPUs)
        let targets: Vec<Option<ColorTargetStateDescriptor>> = (0..8)
            .map(|_| Some(ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm)))
            .collect();
        assert_eq!(targets.len(), 8);
    }

    #[test]
    fn test_mrt_with_null_targets() {
        // MRT with some null targets (sparse attachment)
        let targets = vec![
            Some(ColorTargetStateDescriptor::srgb()),
            None,  // Skip attachment 1
            Some(ColorTargetStateDescriptor::hdr()),
            None,  // Skip attachment 3
        ];
        assert_eq!(targets.len(), 4);
        assert!(targets[0].is_some());
        assert!(targets[1].is_none());
        assert!(targets[2].is_some());
        assert!(targets[3].is_none());
    }

    #[test]
    fn test_blend_component_equality() {
        let comp1 = BlendComponentDescriptor::alpha_blend();
        let comp2 = BlendComponentDescriptor::alpha_blend();
        let comp3 = BlendComponentDescriptor::additive();

        assert_eq!(comp1, comp2);
        assert_ne!(comp1, comp3);
    }

    #[test]
    fn test_blend_state_equality() {
        let state1 = BlendStateDescriptor::alpha_blend();
        let state2 = BlendStateDescriptor::alpha_blend();
        let state3 = BlendStateDescriptor::additive();

        assert_eq!(state1, state2);
        assert_ne!(state1, state3);
    }

    #[test]
    fn test_color_target_equality() {
        let target1 = ColorTargetStateDescriptor::srgb();
        let target2 = ColorTargetStateDescriptor::srgb();
        let target3 = ColorTargetStateDescriptor::hdr();

        assert_eq!(target1, target2);
        assert_ne!(target1, target3);
    }

    #[test]
    fn test_blend_component_copy() {
        let comp = BlendComponentDescriptor::alpha_blend();
        let comp_copy = comp;
        assert_eq!(comp, comp_copy);
    }

    #[test]
    fn test_blend_state_copy() {
        let state = BlendStateDescriptor::alpha_blend();
        let state_copy = state;
        assert_eq!(state, state_copy);
    }

    #[test]
    fn test_color_target_clone() {
        let target = ColorTargetStateDescriptor::srgb().alpha_blend();
        let target_clone = target.clone();
        assert_eq!(target, target_clone);
    }

    #[test]
    fn test_blend_component_into_wgpu() {
        let comp = BlendComponentDescriptor {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusDstAlpha,
            operation: wgpu::BlendOperation::Subtract,
        };

        let wgpu_comp: wgpu::BlendComponent = comp.into();

        assert_eq!(wgpu_comp.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(wgpu_comp.dst_factor, wgpu::BlendFactor::OneMinusDstAlpha);
        assert_eq!(wgpu_comp.operation, wgpu::BlendOperation::Subtract);
    }

    #[test]
    fn test_blend_state_into_wgpu() {
        let state = BlendStateDescriptor {
            color: BlendComponentDescriptor::additive(),
            alpha: BlendComponentDescriptor::replace(),
        };

        let wgpu_state: wgpu::BlendState = state.into();

        assert_eq!(wgpu_state.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(wgpu_state.color.dst_factor, wgpu::BlendFactor::One);
        assert_eq!(wgpu_state.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(wgpu_state.alpha.dst_factor, wgpu::BlendFactor::Zero);
    }

    #[test]
    fn test_various_texture_formats() {
        // Test various render target formats
        let formats = [
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureFormat::Rg8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::R16Float,
            wgpu::TextureFormat::Rg16Float,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureFormat::Rg32Float,
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureFormat::Rgb10a2Unorm,
            wgpu::TextureFormat::Rg11b10Float,
        ];

        for format in formats {
            let target = ColorTargetStateDescriptor::new(format);
            assert_eq!(target.format, format);
        }
    }

    #[test]
    fn test_color_target_no_blend() {
        let target = ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm);
        assert!(target.blend.is_none());
    }

    #[test]
    fn test_color_target_custom_blend() {
        let custom_blend = BlendStateDescriptor {
            color: BlendComponentDescriptor {
                src_factor: wgpu::BlendFactor::Constant,
                dst_factor: wgpu::BlendFactor::OneMinusConstant,
                operation: wgpu::BlendOperation::Max,
            },
            alpha: BlendComponentDescriptor::replace(),
        };

        let target = ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm)
            .blend(custom_blend);

        assert!(target.blend.is_some());
        let blend = target.blend.unwrap();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::Constant);
        assert_eq!(blend.color.operation, wgpu::BlendOperation::Max);
    }
}
