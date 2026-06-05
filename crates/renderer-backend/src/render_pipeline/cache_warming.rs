//! Cache warming utilities for render pipeline pre-compilation.

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::PipelineKey;

/// Configuration for cache warming behavior.
#[derive(Clone, Debug)]
pub struct WarmingConfig {
    /// Run compilation in background thread.
    pub background: bool,
    /// Number of parallel compilation threads (default: available CPUs).
    pub parallelism: usize,
}

impl Default for WarmingConfig {
    fn default() -> Self {
        Self {
            background: false,
            parallelism: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4),
        }
    }
}

impl WarmingConfig {
    /// Create a configuration for background warming.
    pub fn background() -> Self {
        Self { background: true, ..Default::default() }
    }

    /// Set the parallelism level.
    pub fn with_parallelism(mut self, parallelism: usize) -> Self {
        self.parallelism = parallelism.max(1);
        self
    }
}

/// Progress information during cache warming.
#[derive(Clone, Debug)]
pub struct WarmingProgress {
    /// Number of pipelines completed.
    pub completed: usize,
    /// Total number of pipelines to warm.
    pub total: usize,
    /// The current key being compiled (None if between compilations).
    pub current_key: Option<PipelineKey>,
}

impl WarmingProgress {
    /// Calculate progress as a fraction (0.0 - 1.0).
    pub fn fraction(&self) -> f64 {
        if self.total == 0 { 1.0 } else { self.completed as f64 / self.total as f64 }
    }

    /// Calculate progress as a percentage (0 - 100).
    pub fn percent(&self) -> u8 {
        (self.fraction() * 100.0) as u8
    }
}

/// Callback function for progress updates.
pub type ProgressCallback = Box<dyn Fn(WarmingProgress) + Send + Sync>;

/// Result of a cache warming operation.
#[derive(Clone, Debug)]
pub struct WarmingResult {
    /// Number of pipelines successfully warmed.
    pub warmed: usize,
    /// Number of pipelines skipped (already in cache).
    pub skipped: usize,
    /// Number of pipelines that failed to compile.
    pub failed: usize,
    /// Total duration of the warming operation.
    pub duration: Duration,
}

impl WarmingResult {
    /// Total number of keys processed.
    pub fn total(&self) -> usize {
        self.warmed + self.skipped + self.failed
    }

    /// Check if all pipelines were successfully processed (warmed or skipped).
    pub fn is_success(&self) -> bool {
        self.failed == 0
    }
}

/// Handle for an asynchronous warming operation.
pub struct WarmingHandle {
    handle: std::thread::JoinHandle<WarmingResult>,
}

impl WarmingHandle {
    /// Wait for the warming operation to complete and return the result.
    pub fn join(self) -> WarmingResult {
        self.handle.join().unwrap_or(WarmingResult {
            warmed: 0, skipped: 0, failed: 0, duration: Duration::ZERO,
        })
    }

    /// Check if the warming operation has finished.
    pub fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }
}

/// Internal function to warm a single key.
fn warm_single_key<F>(cache: &super::RenderPipelineCache, key: &PipelineKey, create_fn: &F) -> bool
where
    F: Fn(&PipelineKey) -> wgpu::RenderPipeline,
{
    if cache.contains(key) {
        return false;
    }
    cache.get_or_create(key, || create_fn(key));
    true
}

impl super::RenderPipelineCache {
    /// Pre-warm the cache with the given pipeline keys.
    pub fn warm_cache<F>(
        &self,
        keys: &[PipelineKey],
        create_fn: F,
        _config: WarmingConfig,
        progress: Option<ProgressCallback>,
    ) -> WarmingResult
    where
        F: Fn(&PipelineKey) -> wgpu::RenderPipeline + Send + Sync,
    {
        let start = Instant::now();
        let total = keys.len();
        let mut warmed = 0;
        let mut skipped = 0;

        for (i, key) in keys.iter().enumerate() {
            if let Some(ref cb) = progress {
                cb(WarmingProgress { completed: i, total, current_key: Some(key.clone()) });
            }
            if self.contains(key) {
                skipped += 1;
            } else {
                self.get_or_create(key, || create_fn(key));
                warmed += 1;
            }
        }

        if let Some(ref cb) = progress {
            cb(WarmingProgress { completed: total, total, current_key: None });
        }

        WarmingResult { warmed, skipped, failed: 0, duration: start.elapsed() }
    }

    /// Warm cache in background, returning immediately.
    pub fn warm_cache_async<F>(
        self: &Arc<Self>,
        keys: Vec<PipelineKey>,
        create_fn: F,
        progress: Option<ProgressCallback>,
    ) -> WarmingHandle
    where
        F: Fn(&PipelineKey) -> wgpu::RenderPipeline + Send + Sync + 'static,
    {
        let cache = Arc::clone(self);
        let progress = progress.map(Arc::new);

        let handle = std::thread::spawn(move || {
            let start = Instant::now();
            let total = keys.len();
            let (mut warmed, mut skipped) = (0, 0);

            for (i, key) in keys.iter().enumerate() {
                if let Some(ref cb) = progress {
                    cb(WarmingProgress { completed: i, total, current_key: Some(key.clone()) });
                }
                if warm_single_key(&cache, key, &create_fn) { warmed += 1; } else { skipped += 1; }
            }

            if let Some(ref cb) = progress {
                cb(WarmingProgress { completed: total, total, current_key: None });
            }

            WarmingResult { warmed, skipped, failed: 0, duration: start.elapsed() }
        });

        WarmingHandle { handle }
    }
}

/// Common TRINITY pipeline key presets.
pub mod common_pipelines {
    use super::super::PipelineKey;

    /// PBR forward rendering pipeline key (4x MSAA, back-face culling, depth test).
    pub fn pbr_forward(vertex_shader_id: u64, fragment_shader_id: u64) -> PipelineKey {
        PipelineKey::new(vertex_shader_id)
            .with_fragment_shader(fragment_shader_id)
            .with_cull_mode(Some(wgpu::Face::Back))
            .with_depth_format(wgpu::TextureFormat::Depth32Float)
            .with_depth_write(true)
            .with_sample_count(4)
    }

    /// Shadow map pipeline key (depth-only, front-face culling).
    pub fn shadow_map(vertex_shader_id: u64) -> PipelineKey {
        PipelineKey::new(vertex_shader_id)
            .with_cull_mode(Some(wgpu::Face::Front))
            .with_depth_format(wgpu::TextureFormat::Depth32Float)
            .with_depth_write(true)
    }

    /// UI pipeline key (alpha blend, no culling, no depth).
    pub fn ui(vertex_shader_id: u64, fragment_shader_id: u64) -> PipelineKey {
        PipelineKey::new(vertex_shader_id)
            .with_fragment_shader(fragment_shader_id)
            .with_cull_mode(None)
            .with_depth_write(false)
            .with_color_targets_hash(ALPHA_BLEND_HASH)
    }

    /// Skybox pipeline key (LessEqual depth, no depth write).
    pub fn skybox(vertex_shader_id: u64, fragment_shader_id: u64) -> PipelineKey {
        PipelineKey::new(vertex_shader_id)
            .with_fragment_shader(fragment_shader_id)
            .with_cull_mode(None)
            .with_depth_format(wgpu::TextureFormat::Depth32Float)
            .with_depth_write(false)
            .with_depth_compare(wgpu::CompareFunction::LessEqual)
    }

    /// Particle pipeline key (additive blend, no depth write).
    pub fn particle(vertex_shader_id: u64, fragment_shader_id: u64) -> PipelineKey {
        PipelineKey::new(vertex_shader_id)
            .with_fragment_shader(fragment_shader_id)
            .with_cull_mode(None)
            .with_depth_format(wgpu::TextureFormat::Depth32Float)
            .with_depth_write(false)
            .with_color_targets_hash(ADDITIVE_BLEND_HASH)
    }

    /// Post-process fullscreen quad pipeline key.
    pub fn fullscreen_quad(vertex_shader_id: u64, fragment_shader_id: u64) -> PipelineKey {
        PipelineKey::new(vertex_shader_id)
            .with_fragment_shader(fragment_shader_id)
            .with_cull_mode(None)
            .with_depth_write(false)
    }

    const ALPHA_BLEND_HASH: u64 = 0x414C_5048_415F_424C;
    const ADDITIVE_BLEND_HASH: u64 = 0x4144_4449_545F_424C;
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // WarmingConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_warming_config_default() {
        let config = WarmingConfig::default();
        assert!(!config.background);
        assert!(config.parallelism >= 1);
    }

    #[test]
    fn test_warming_config_default_parallelism_uses_available() {
        let config = WarmingConfig::default();
        let expected = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        assert_eq!(config.parallelism, expected);
    }

    #[test]
    fn test_warming_config_background() {
        assert!(WarmingConfig::background().background);
    }

    #[test]
    fn test_warming_config_background_preserves_parallelism() {
        let bg_config = WarmingConfig::background();
        let default_config = WarmingConfig::default();
        assert_eq!(bg_config.parallelism, default_config.parallelism);
    }

    #[test]
    fn test_warming_config_with_parallelism() {
        assert_eq!(WarmingConfig::default().with_parallelism(8).parallelism, 8);
        assert_eq!(WarmingConfig::default().with_parallelism(0).parallelism, 1);
    }

    #[test]
    fn test_warming_config_with_parallelism_zero_clamps_to_one() {
        let config = WarmingConfig::default().with_parallelism(0);
        assert_eq!(config.parallelism, 1);
    }

    #[test]
    fn test_warming_config_with_parallelism_one() {
        let config = WarmingConfig::default().with_parallelism(1);
        assert_eq!(config.parallelism, 1);
    }

    #[test]
    fn test_warming_config_with_parallelism_large_value() {
        let config = WarmingConfig::default().with_parallelism(1024);
        assert_eq!(config.parallelism, 1024);
    }

    #[test]
    fn test_warming_config_builder_chaining() {
        let config = WarmingConfig::background()
            .with_parallelism(16);
        assert!(config.background);
        assert_eq!(config.parallelism, 16);
    }

    #[test]
    fn test_warming_config_clone() {
        let config = WarmingConfig::background().with_parallelism(4);
        let cloned = config.clone();
        assert_eq!(config.background, cloned.background);
        assert_eq!(config.parallelism, cloned.parallelism);
    }

    #[test]
    fn test_warming_config_debug() {
        let config = WarmingConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("WarmingConfig"));
        assert!(debug_str.contains("background"));
        assert!(debug_str.contains("parallelism"));
    }

    // -------------------------------------------------------------------------
    // WarmingProgress Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_warming_progress_fraction() {
        assert_eq!(WarmingProgress { completed: 0, total: 10, current_key: None }.fraction(), 0.0);
        assert!((WarmingProgress { completed: 5, total: 10, current_key: None }.fraction() - 0.5).abs() < f64::EPSILON);
        assert_eq!(WarmingProgress { completed: 10, total: 10, current_key: None }.fraction(), 1.0);
        assert_eq!(WarmingProgress { completed: 0, total: 0, current_key: None }.fraction(), 1.0);
    }

    #[test]
    fn test_warming_progress_fraction_zero_total_returns_one() {
        let progress = WarmingProgress { completed: 0, total: 0, current_key: None };
        assert_eq!(progress.fraction(), 1.0);
    }

    #[test]
    fn test_warming_progress_fraction_zero_completed() {
        let progress = WarmingProgress { completed: 0, total: 100, current_key: None };
        assert_eq!(progress.fraction(), 0.0);
    }

    #[test]
    fn test_warming_progress_fraction_half() {
        let progress = WarmingProgress { completed: 50, total: 100, current_key: None };
        assert!((progress.fraction() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_warming_progress_fraction_complete() {
        let progress = WarmingProgress { completed: 100, total: 100, current_key: None };
        assert_eq!(progress.fraction(), 1.0);
    }

    #[test]
    fn test_warming_progress_fraction_quarter() {
        let progress = WarmingProgress { completed: 25, total: 100, current_key: None };
        assert!((progress.fraction() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_warming_progress_percent() {
        assert_eq!(WarmingProgress { completed: 3, total: 4, current_key: None }.percent(), 75);
    }

    #[test]
    fn test_warming_progress_percent_zero_total_returns_100() {
        let progress = WarmingProgress { completed: 0, total: 0, current_key: None };
        assert_eq!(progress.percent(), 100);
    }

    #[test]
    fn test_warming_progress_percent_zero() {
        let progress = WarmingProgress { completed: 0, total: 100, current_key: None };
        assert_eq!(progress.percent(), 0);
    }

    #[test]
    fn test_warming_progress_percent_50() {
        let progress = WarmingProgress { completed: 50, total: 100, current_key: None };
        assert_eq!(progress.percent(), 50);
    }

    #[test]
    fn test_warming_progress_percent_100() {
        let progress = WarmingProgress { completed: 100, total: 100, current_key: None };
        assert_eq!(progress.percent(), 100);
    }

    #[test]
    fn test_warming_progress_percent_truncation() {
        // 33.33...% should truncate to 33
        let progress = WarmingProgress { completed: 1, total: 3, current_key: None };
        assert_eq!(progress.percent(), 33);
    }

    #[test]
    fn test_warming_progress_with_current_key() {
        let key = PipelineKey::new(42);
        let progress = WarmingProgress { completed: 1, total: 5, current_key: Some(key) };
        assert_eq!(progress.current_key.unwrap().vertex_shader_id, 42);
    }

    #[test]
    fn test_warming_progress_without_current_key() {
        let progress = WarmingProgress { completed: 5, total: 10, current_key: None };
        assert!(progress.current_key.is_none());
    }

    #[test]
    fn test_warming_progress_current_key_with_fragment() {
        let key = PipelineKey::new(1).with_fragment_shader(2);
        let progress = WarmingProgress { completed: 0, total: 1, current_key: Some(key) };
        let unwrapped = progress.current_key.unwrap();
        assert_eq!(unwrapped.vertex_shader_id, 1);
        assert_eq!(unwrapped.fragment_shader_id, Some(2));
    }

    #[test]
    fn test_warming_progress_clone() {
        let key = PipelineKey::new(42);
        let progress = WarmingProgress { completed: 5, total: 10, current_key: Some(key) };
        let cloned = progress.clone();
        assert_eq!(progress.completed, cloned.completed);
        assert_eq!(progress.total, cloned.total);
        assert_eq!(progress.current_key, cloned.current_key);
    }

    #[test]
    fn test_warming_progress_debug() {
        let progress = WarmingProgress { completed: 5, total: 10, current_key: None };
        let debug_str = format!("{:?}", progress);
        assert!(debug_str.contains("WarmingProgress"));
        assert!(debug_str.contains("completed"));
        assert!(debug_str.contains("total"));
    }

    // -------------------------------------------------------------------------
    // WarmingResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_warming_result_total() {
        let result = WarmingResult { warmed: 5, skipped: 3, failed: 2, duration: Duration::from_secs(1) };
        assert_eq!(result.total(), 10);
    }

    #[test]
    fn test_warming_result_total_empty() {
        let result = WarmingResult { warmed: 0, skipped: 0, failed: 0, duration: Duration::ZERO };
        assert_eq!(result.total(), 0);
    }

    #[test]
    fn test_warming_result_total_all_warmed() {
        let result = WarmingResult { warmed: 100, skipped: 0, failed: 0, duration: Duration::from_secs(1) };
        assert_eq!(result.total(), 100);
    }

    #[test]
    fn test_warming_result_total_all_skipped() {
        let result = WarmingResult { warmed: 0, skipped: 50, failed: 0, duration: Duration::ZERO };
        assert_eq!(result.total(), 50);
    }

    #[test]
    fn test_warming_result_total_all_failed() {
        let result = WarmingResult { warmed: 0, skipped: 0, failed: 25, duration: Duration::from_millis(500) };
        assert_eq!(result.total(), 25);
    }

    #[test]
    fn test_warming_result_is_success() {
        assert!(WarmingResult { warmed: 5, skipped: 5, failed: 0, duration: Duration::ZERO }.is_success());
        assert!(!WarmingResult { warmed: 5, skipped: 3, failed: 2, duration: Duration::ZERO }.is_success());
    }

    #[test]
    fn test_warming_result_is_success_all_warmed() {
        let result = WarmingResult { warmed: 100, skipped: 0, failed: 0, duration: Duration::from_secs(1) };
        assert!(result.is_success());
    }

    #[test]
    fn test_warming_result_is_success_all_skipped() {
        let result = WarmingResult { warmed: 0, skipped: 100, failed: 0, duration: Duration::ZERO };
        assert!(result.is_success());
    }

    #[test]
    fn test_warming_result_is_success_mixed_warmed_skipped() {
        let result = WarmingResult { warmed: 50, skipped: 50, failed: 0, duration: Duration::from_millis(250) };
        assert!(result.is_success());
    }

    #[test]
    fn test_warming_result_is_success_one_failed() {
        let result = WarmingResult { warmed: 99, skipped: 0, failed: 1, duration: Duration::from_secs(1) };
        assert!(!result.is_success());
    }

    #[test]
    fn test_warming_result_is_success_all_failed() {
        let result = WarmingResult { warmed: 0, skipped: 0, failed: 100, duration: Duration::from_secs(2) };
        assert!(!result.is_success());
    }

    #[test]
    fn test_warming_result_empty() {
        let result = WarmingResult { warmed: 0, skipped: 0, failed: 0, duration: Duration::ZERO };
        assert_eq!(result.total(), 0);
        assert!(result.is_success());
    }

    #[test]
    fn test_warming_result_duration_tracking() {
        let duration = Duration::from_millis(1234);
        let result = WarmingResult { warmed: 10, skipped: 5, failed: 0, duration };
        assert_eq!(result.duration.as_millis(), 1234);
    }

    #[test]
    fn test_warming_result_duration_zero() {
        let result = WarmingResult { warmed: 0, skipped: 0, failed: 0, duration: Duration::ZERO };
        assert_eq!(result.duration.as_nanos(), 0);
    }

    #[test]
    fn test_warming_result_duration_large() {
        let duration = Duration::from_secs(3600); // 1 hour
        let result = WarmingResult { warmed: 1000, skipped: 500, failed: 0, duration };
        assert_eq!(result.duration.as_secs(), 3600);
    }

    #[test]
    fn test_warming_result_clone() {
        let result = WarmingResult {
            warmed: 10,
            skipped: 5,
            failed: 2,
            duration: Duration::from_millis(500),
        };
        let cloned = result.clone();
        assert_eq!(result.warmed, cloned.warmed);
        assert_eq!(result.skipped, cloned.skipped);
        assert_eq!(result.failed, cloned.failed);
        assert_eq!(result.duration, cloned.duration);
    }

    #[test]
    fn test_warming_result_debug() {
        let result = WarmingResult { warmed: 5, skipped: 3, failed: 1, duration: Duration::from_secs(1) };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("WarmingResult"));
        assert!(debug_str.contains("warmed"));
        assert!(debug_str.contains("skipped"));
        assert!(debug_str.contains("failed"));
        assert!(debug_str.contains("duration"));
    }

    // -------------------------------------------------------------------------
    // Common Pipelines Tests - Full Field Verification
    // -------------------------------------------------------------------------

    #[test]
    fn test_common_pipelines_pbr_forward() {
        let key = common_pipelines::pbr_forward(1, 2);
        assert_eq!(key.vertex_shader_id, 1);
        assert_eq!(key.fragment_shader_id, Some(2));
        assert_eq!(key.cull_mode, Some(wgpu::Face::Back));
        assert_eq!(key.depth_format, Some(wgpu::TextureFormat::Depth32Float));
        assert!(key.depth_write);
        assert_eq!(key.sample_count, 4);
    }

    #[test]
    fn test_common_pipelines_pbr_forward_4x_msaa() {
        let key = common_pipelines::pbr_forward(100, 200);
        assert_eq!(key.sample_count, 4);
    }

    #[test]
    fn test_common_pipelines_pbr_forward_back_cull() {
        let key = common_pipelines::pbr_forward(1, 2);
        assert_eq!(key.cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn test_common_pipelines_pbr_forward_depth_test() {
        let key = common_pipelines::pbr_forward(1, 2);
        assert_eq!(key.depth_format, Some(wgpu::TextureFormat::Depth32Float));
        assert!(key.depth_write);
    }

    #[test]
    fn test_common_pipelines_shadow_map() {
        let key = common_pipelines::shadow_map(1);
        assert_eq!(key.vertex_shader_id, 1);
        assert_eq!(key.fragment_shader_id, None);
        assert_eq!(key.cull_mode, Some(wgpu::Face::Front));
        assert!(key.depth_write);
    }

    #[test]
    fn test_common_pipelines_shadow_map_depth_only() {
        let key = common_pipelines::shadow_map(42);
        // Depth-only means no fragment shader
        assert!(key.fragment_shader_id.is_none());
        assert!(key.depth_write);
        assert_eq!(key.depth_format, Some(wgpu::TextureFormat::Depth32Float));
    }

    #[test]
    fn test_common_pipelines_shadow_map_front_cull() {
        let key = common_pipelines::shadow_map(1);
        assert_eq!(key.cull_mode, Some(wgpu::Face::Front));
    }

    #[test]
    fn test_common_pipelines_ui() {
        let key = common_pipelines::ui(1, 2);
        assert_eq!(key.cull_mode, None);
        assert!(!key.depth_write);
    }

    #[test]
    fn test_common_pipelines_ui_alpha_blend() {
        let key = common_pipelines::ui(1, 2);
        // UI uses alpha blend hash
        assert_ne!(key.color_targets_hash, 0);
    }

    #[test]
    fn test_common_pipelines_ui_no_depth() {
        let key = common_pipelines::ui(1, 2);
        assert!(!key.depth_write);
    }

    #[test]
    fn test_common_pipelines_ui_no_cull() {
        let key = common_pipelines::ui(1, 2);
        assert_eq!(key.cull_mode, None);
    }

    #[test]
    fn test_common_pipelines_skybox() {
        let key = common_pipelines::skybox(1, 2);
        assert_eq!(key.cull_mode, None);
        assert!(!key.depth_write);
        assert_eq!(key.depth_compare, wgpu::CompareFunction::LessEqual);
    }

    #[test]
    fn test_common_pipelines_skybox_less_equal_depth() {
        let key = common_pipelines::skybox(1, 2);
        assert_eq!(key.depth_compare, wgpu::CompareFunction::LessEqual);
    }

    #[test]
    fn test_common_pipelines_skybox_no_depth_write() {
        let key = common_pipelines::skybox(1, 2);
        assert!(!key.depth_write);
    }

    #[test]
    fn test_common_pipelines_skybox_has_depth_format() {
        let key = common_pipelines::skybox(1, 2);
        assert_eq!(key.depth_format, Some(wgpu::TextureFormat::Depth32Float));
    }

    #[test]
    fn test_common_pipelines_particle() {
        let key = common_pipelines::particle(1, 2);
        assert_eq!(key.cull_mode, None);
        assert!(!key.depth_write);
    }

    #[test]
    fn test_common_pipelines_particle_additive_blend() {
        let key = common_pipelines::particle(1, 2);
        // Particle uses additive blend hash
        assert_ne!(key.color_targets_hash, 0);
    }

    #[test]
    fn test_common_pipelines_particle_no_depth_write() {
        let key = common_pipelines::particle(1, 2);
        assert!(!key.depth_write);
    }

    #[test]
    fn test_common_pipelines_fullscreen_quad() {
        let key = common_pipelines::fullscreen_quad(1, 2);
        assert_eq!(key.cull_mode, None);
        assert!(!key.depth_write);
    }

    #[test]
    fn test_common_pipelines_fullscreen_quad_no_depth() {
        let key = common_pipelines::fullscreen_quad(1, 2);
        assert!(!key.depth_write);
    }

    #[test]
    fn test_common_pipelines_fullscreen_quad_no_cull() {
        let key = common_pipelines::fullscreen_quad(1, 2);
        assert_eq!(key.cull_mode, None);
    }

    #[test]
    fn test_common_pipelines_unique_keys() {
        let pbr = common_pipelines::pbr_forward(1, 2);
        let shadow = common_pipelines::shadow_map(1);
        let ui = common_pipelines::ui(1, 2);
        let skybox = common_pipelines::skybox(1, 2);
        let particle = common_pipelines::particle(1, 2);
        let fullscreen = common_pipelines::fullscreen_quad(1, 2);
        assert_ne!(pbr, shadow);
        assert_ne!(pbr, ui);
        assert_ne!(pbr, skybox);
        assert_ne!(pbr, particle);
        assert_ne!(pbr, fullscreen);
        assert_ne!(shadow, ui);
    }

    #[test]
    fn test_common_pipelines_all_six_are_distinct() {
        let keys = vec![
            common_pipelines::pbr_forward(1, 2),
            common_pipelines::shadow_map(1),
            common_pipelines::ui(1, 2),
            common_pipelines::skybox(1, 2),
            common_pipelines::particle(1, 2),
            common_pipelines::fullscreen_quad(1, 2),
        ];

        // Verify all 6 keys are distinct from each other
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j], "Pipeline {} and {} should be distinct", i, j);
            }
        }
    }

    #[test]
    fn test_common_pipelines_different_shader_ids() {
        let key1 = common_pipelines::pbr_forward(1, 2);
        let key2 = common_pipelines::pbr_forward(3, 4);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_common_pipelines_same_shader_ids_same_type() {
        let key1 = common_pipelines::pbr_forward(10, 20);
        let key2 = common_pipelines::pbr_forward(10, 20);
        assert_eq!(key1, key2);
    }

    // -------------------------------------------------------------------------
    // WarmingHandle Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_warming_handle_is_finished() {
        let handle = WarmingHandle {
            handle: std::thread::spawn(|| WarmingResult {
                warmed: 0, skipped: 0, failed: 0, duration: Duration::ZERO,
            }),
        };
        std::thread::sleep(Duration::from_millis(10));
        assert!(handle.is_finished());
    }

    #[test]
    fn test_warming_handle_join() {
        let handle = WarmingHandle {
            handle: std::thread::spawn(|| WarmingResult {
                warmed: 5, skipped: 3, failed: 0, duration: Duration::from_millis(100),
            }),
        };
        let result = handle.join();
        assert_eq!(result.warmed, 5);
        assert_eq!(result.skipped, 3);
        assert!(result.is_success());
    }

    #[test]
    fn test_warming_handle_join_returns_correct_values() {
        let handle = WarmingHandle {
            handle: std::thread::spawn(|| WarmingResult {
                warmed: 10,
                skipped: 20,
                failed: 5,
                duration: Duration::from_millis(500),
            }),
        };
        let result = handle.join();
        assert_eq!(result.warmed, 10);
        assert_eq!(result.skipped, 20);
        assert_eq!(result.failed, 5);
        assert!(!result.is_success());
    }

    #[test]
    fn test_warming_handle_is_finished_immediate() {
        let handle = WarmingHandle {
            handle: std::thread::spawn(|| WarmingResult {
                warmed: 0, skipped: 0, failed: 0, duration: Duration::ZERO,
            }),
        };
        // Give thread time to complete
        std::thread::sleep(Duration::from_millis(50));
        assert!(handle.is_finished());
    }

    // -------------------------------------------------------------------------
    // ProgressCallback Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_progress_callback_type() {
        let callback: ProgressCallback = Box::new(|progress| { let _ = progress.percent(); });
        callback(WarmingProgress { completed: 1, total: 2, current_key: None });
    }

    #[test]
    fn test_progress_callback_receives_progress() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = Arc::clone(&call_count);

        let callback: ProgressCallback = Box::new(move |_progress| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        callback(WarmingProgress { completed: 0, total: 10, current_key: None });
        callback(WarmingProgress { completed: 5, total: 10, current_key: None });
        callback(WarmingProgress { completed: 10, total: 10, current_key: None });

        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    // -------------------------------------------------------------------------
    // Thread Safety Tests (compile-time checks)
    // -------------------------------------------------------------------------

    #[test]
    fn test_warming_config_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<WarmingConfig>();
        assert_sync::<WarmingConfig>();
    }

    #[test]
    fn test_warming_progress_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<WarmingProgress>();
        assert_sync::<WarmingProgress>();
    }

    #[test]
    fn test_warming_result_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<WarmingResult>();
        assert_sync::<WarmingResult>();
    }

    #[test]
    fn test_warming_handle_send() {
        fn assert_send<T: Send>() {}
        assert_send::<WarmingHandle>();
    }

    #[test]
    fn test_progress_callback_send_sync_bounds() {
        // ProgressCallback is Box<dyn Fn(WarmingProgress) + Send + Sync>
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<ProgressCallback>();
        assert_sync::<ProgressCallback>();
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_warming_progress_large_total() {
        let progress = WarmingProgress { completed: 500_000, total: 1_000_000, current_key: None };
        assert!((progress.fraction() - 0.5).abs() < f64::EPSILON);
        assert_eq!(progress.percent(), 50);
    }

    #[test]
    fn test_warming_progress_completed_equals_total() {
        let progress = WarmingProgress { completed: 100, total: 100, current_key: None };
        assert_eq!(progress.fraction(), 1.0);
        assert_eq!(progress.percent(), 100);
    }

    #[test]
    fn test_warming_result_large_counts() {
        let result = WarmingResult {
            warmed: 1_000_000,
            skipped: 500_000,
            failed: 100_000,
            duration: Duration::from_secs(3600),
        };
        assert_eq!(result.total(), 1_600_000);
        assert!(!result.is_success());
    }

    #[test]
    fn test_warming_config_parallelism_max() {
        let config = WarmingConfig::default().with_parallelism(usize::MAX);
        assert_eq!(config.parallelism, usize::MAX);
    }

    #[test]
    fn test_warming_result_duration_precision() {
        let duration = Duration::from_nanos(123456789);
        let result = WarmingResult { warmed: 1, skipped: 0, failed: 0, duration };
        assert_eq!(result.duration.as_nanos(), 123456789);
    }

    // -------------------------------------------------------------------------
    // Additional Coverage Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_warming_progress_one_of_many() {
        let progress = WarmingProgress { completed: 1, total: 1000, current_key: None };
        assert!((progress.fraction() - 0.001).abs() < f64::EPSILON);
        assert_eq!(progress.percent(), 0); // 0.1% truncates to 0
    }

    #[test]
    fn test_warming_progress_almost_complete() {
        let progress = WarmingProgress { completed: 999, total: 1000, current_key: None };
        assert!((progress.fraction() - 0.999).abs() < f64::EPSILON);
        assert_eq!(progress.percent(), 99);
    }

    #[test]
    fn test_warming_result_only_failed() {
        let result = WarmingResult { warmed: 0, skipped: 0, failed: 50, duration: Duration::from_secs(10) };
        assert_eq!(result.total(), 50);
        assert!(!result.is_success());
    }

    #[test]
    fn test_common_pipelines_ui_vs_particle_blend_different() {
        let ui = common_pipelines::ui(1, 2);
        let particle = common_pipelines::particle(1, 2);
        // Both use custom blend hashes but different ones
        assert_ne!(ui.color_targets_hash, particle.color_targets_hash);
    }

    #[test]
    fn test_common_pipelines_fullscreen_quad_vs_ui() {
        let ui = common_pipelines::ui(1, 2);
        let fullscreen = common_pipelines::fullscreen_quad(1, 2);
        // UI has alpha blend, fullscreen_quad doesn't set custom blend
        assert_ne!(ui, fullscreen);
    }

    #[test]
    fn test_common_pipelines_skybox_vs_pbr_depth_compare() {
        let pbr = common_pipelines::pbr_forward(1, 2);
        let skybox = common_pipelines::skybox(1, 2);
        // PBR uses default Less, skybox uses LessEqual
        assert_ne!(pbr.depth_compare, skybox.depth_compare);
    }
}
