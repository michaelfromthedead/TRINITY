// WHITEBOX tests for T-WGPU-P1.1.2 (Backend Selection)
//
// WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
// They exercise internal code paths, branch conditions, and edge cases
// that are not visible through the public contract alone.
//
// Implementation under test: crates/renderer-backend/src/device/instance.rs
//   - TrinityInstance::select_backends()
//   - TrinityInstance::try_backend_from_env()
//   - TrinityInstance::platform_default_backends()
//   - TrinityInstance::backend_priority_description()
//
// WHITEBOX coverage plan:
//   - Path A: TRINITY_BACKEND env not set -> returns None from try_backend_from_env
//   - Path B: TRINITY_BACKEND = "vulkan" or "vk" -> returns Backends::VULKAN
//   - Path C: TRINITY_BACKEND = "dx12"/"d3d12"/"directx12" -> DX12 on Windows, fallback elsewhere
//   - Path D: TRINITY_BACKEND = "metal"/"mtl" -> METAL on macOS, fallback elsewhere
//   - Path E: TRINITY_BACKEND = "opengl"/"gl"/"gles" -> returns Backends::GL
//   - Path F: TRINITY_BACKEND = "webgpu"/"browser_webgpu" -> BROWSER_WEBGPU on WASM, fallback elsewhere
//   - Path G: TRINITY_BACKEND = "primary" -> returns Backends::PRIMARY
//   - Path H: TRINITY_BACKEND = "secondary" -> returns Backends::SECONDARY
//   - Path I: TRINITY_BACKEND = "all" -> returns Backends::all()
//   - Path J: TRINITY_BACKEND = invalid value -> returns None (fallback to platform default)
//   - Path K: Case insensitivity verification (VULKAN, Vulkan, vulkan all work)
//   - Path L: platform_default_backends() returns correct backends per platform
//   - Path M: backend_priority_description() returns correct string per platform
//   - Path N: select_backends() prioritizes env over platform default
//   - Path O: Incompatible backend for platform falls back gracefully

use renderer_backend::device::TrinityInstance;
use std::env;
use std::sync::Mutex;

// ============================================================================
// Test Helpers
// ============================================================================

/// Global mutex to ensure tests that modify environment variables run serially.
/// Environment variables are process-global, so parallel tests would race.
static ENV_MUTEX: Mutex<()> = Mutex::new(());

/// RAII guard to safely set/restore environment variables.
/// Ensures tests don't leak state to each other.
/// Also holds the ENV_MUTEX lock to prevent parallel access.
struct EnvGuard {
    key: &'static str,
    original: Option<String>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl EnvGuard {
    /// Set an environment variable, saving the original value.
    /// Acquires the ENV_MUTEX to ensure serial execution.
    fn set(key: &'static str, value: &str) -> Self {
        let lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let original = env::var(key).ok();
        env::set_var(key, value);
        Self { key, original, _lock: lock }
    }

    /// Clear an environment variable, saving the original value.
    /// Acquires the ENV_MUTEX to ensure serial execution.
    fn clear(key: &'static str) -> Self {
        let lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let original = env::var(key).ok();
        env::remove_var(key);
        Self { key, original, _lock: lock }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(val) => env::set_var(self.key, val),
            None => env::remove_var(self.key),
        }
        // _lock is dropped here, releasing the mutex
    }
}

// ============================================================================
// Path A: Environment variable not set
// ============================================================================

#[test]
fn test_select_backends_without_env_uses_platform_default() {
    let _guard = EnvGuard::clear("TRINITY_BACKEND");

    let backends = TrinityInstance::select_backends();

    // Should return platform-specific defaults
    #[cfg(target_arch = "wasm32")]
    assert_eq!(backends, wgpu::Backends::BROWSER_WEBGPU);

    #[cfg(all(not(target_arch = "wasm32"), target_os = "windows"))]
    {
        assert!(backends.contains(wgpu::Backends::VULKAN));
        assert!(backends.contains(wgpu::Backends::DX12));
        assert!(backends.contains(wgpu::Backends::GL));
    }

    #[cfg(all(not(target_arch = "wasm32"), target_os = "macos"))]
    assert_eq!(backends, wgpu::Backends::METAL);

    #[cfg(all(not(target_arch = "wasm32"), target_os = "linux"))]
    {
        assert!(backends.contains(wgpu::Backends::VULKAN));
        assert!(backends.contains(wgpu::Backends::GL));
        // DX12 and Metal should NOT be present on Linux
        assert!(!backends.contains(wgpu::Backends::DX12));
        assert!(!backends.contains(wgpu::Backends::METAL));
    }
}

// ============================================================================
// Path B: Vulkan override (including aliases)
// ============================================================================

#[test]
fn test_env_override_vulkan_lowercase() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "vulkan");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::VULKAN);
}

#[test]
fn test_env_override_vk_alias() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "vk");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::VULKAN);
}

// ============================================================================
// Path C: DX12 override (Windows only, fallback elsewhere)
// ============================================================================

#[test]
#[cfg(target_os = "windows")]
fn test_env_override_dx12_on_windows() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "dx12");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::DX12);
}

#[test]
#[cfg(target_os = "windows")]
fn test_env_override_d3d12_alias_on_windows() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "d3d12");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::DX12);
}

#[test]
#[cfg(target_os = "windows")]
fn test_env_override_directx12_alias_on_windows() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "directx12");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::DX12);
}

#[test]
#[cfg(not(target_os = "windows"))]
fn test_env_override_dx12_on_non_windows_falls_back() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "dx12");

    // DX12 not available, should fall back to platform default
    let backends = TrinityInstance::select_backends();

    #[cfg(target_os = "linux")]
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::GL);

    #[cfg(target_os = "macos")]
    assert_eq!(backends, wgpu::Backends::METAL);

    #[cfg(target_arch = "wasm32")]
    assert_eq!(backends, wgpu::Backends::BROWSER_WEBGPU);
}

// ============================================================================
// Path D: Metal override (macOS only, fallback elsewhere)
// ============================================================================

#[test]
#[cfg(target_os = "macos")]
fn test_env_override_metal_on_macos() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "metal");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::METAL);
}

#[test]
#[cfg(target_os = "macos")]
fn test_env_override_mtl_alias_on_macos() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "mtl");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::METAL);
}

#[test]
#[cfg(not(target_os = "macos"))]
fn test_env_override_metal_on_non_macos_falls_back() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "metal");

    // Metal not available, should fall back to platform default
    let backends = TrinityInstance::select_backends();

    #[cfg(target_os = "linux")]
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::GL);

    #[cfg(target_os = "windows")]
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::GL);

    #[cfg(target_arch = "wasm32")]
    assert_eq!(backends, wgpu::Backends::BROWSER_WEBGPU);
}

// ============================================================================
// Path E: OpenGL override (all aliases)
// ============================================================================

#[test]
fn test_env_override_opengl() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "opengl");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::GL);
}

#[test]
fn test_env_override_gl_alias() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "gl");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::GL);
}

#[test]
fn test_env_override_gles_alias() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "gles");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::GL);
}

// ============================================================================
// Path F: WebGPU override (WASM only, fallback elsewhere)
// ============================================================================

#[test]
#[cfg(target_arch = "wasm32")]
fn test_env_override_webgpu_on_wasm() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "webgpu");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::BROWSER_WEBGPU);
}

#[test]
#[cfg(target_arch = "wasm32")]
fn test_env_override_browser_webgpu_alias_on_wasm() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "browser_webgpu");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::BROWSER_WEBGPU);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_env_override_webgpu_on_non_wasm_falls_back() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "webgpu");

    // WebGPU not available, should fall back to platform default
    let backends = TrinityInstance::select_backends();

    #[cfg(target_os = "linux")]
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::GL);

    #[cfg(target_os = "windows")]
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::GL);

    #[cfg(target_os = "macos")]
    assert_eq!(backends, wgpu::Backends::METAL);
}

// ============================================================================
// Path G: PRIMARY override
// ============================================================================

#[test]
fn test_env_override_primary() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "primary");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::PRIMARY);
}

// ============================================================================
// Path H: SECONDARY override
// ============================================================================

#[test]
fn test_env_override_secondary() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "secondary");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::SECONDARY);
}

// ============================================================================
// Path I: ALL override
// ============================================================================

#[test]
fn test_env_override_all() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "all");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::all());
}

// ============================================================================
// Path J: Invalid value falls back to platform default
// ============================================================================

#[test]
fn test_env_override_invalid_value_falls_back() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "invalid_backend_xyz");

    // Invalid value should fall back to platform defaults
    let backends = TrinityInstance::select_backends();

    #[cfg(target_os = "linux")]
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::GL);

    #[cfg(target_os = "windows")]
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::GL);

    #[cfg(target_os = "macos")]
    assert_eq!(backends, wgpu::Backends::METAL);

    #[cfg(target_arch = "wasm32")]
    assert_eq!(backends, wgpu::Backends::BROWSER_WEBGPU);
}

#[test]
fn test_env_override_empty_string_falls_back() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "");

    // Empty string is invalid, should fall back
    let backends = TrinityInstance::select_backends();

    #[cfg(target_os = "linux")]
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::GL);
}

#[test]
fn test_env_override_whitespace_only_falls_back() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "   ");

    // Whitespace-only is invalid, should fall back
    let backends = TrinityInstance::select_backends();

    #[cfg(target_os = "linux")]
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::GL);
}

#[test]
fn test_env_override_typo_vulckan_falls_back() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "vulckan"); // typo

    // Typo is invalid, should fall back
    let backends = TrinityInstance::select_backends();

    #[cfg(target_os = "linux")]
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::GL);
}

// ============================================================================
// Path K: Case insensitivity verification
// ============================================================================

#[test]
fn test_env_case_insensitive_uppercase() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "VULKAN");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::VULKAN);
}

#[test]
fn test_env_case_insensitive_mixed_case() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "VuLkAn");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::VULKAN);
}

#[test]
fn test_env_case_insensitive_opengl_uppercase() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "OPENGL");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::GL);
}

#[test]
fn test_env_case_insensitive_gl_uppercase() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "GL");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::GL);
}

#[test]
fn test_env_case_insensitive_primary_uppercase() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "PRIMARY");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::PRIMARY);
}

#[test]
fn test_env_case_insensitive_all_uppercase() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "ALL");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::all());
}

// ============================================================================
// Path L: Platform default backends verification
// ============================================================================

#[test]
#[cfg(target_arch = "wasm32")]
fn test_platform_default_wasm() {
    let _guard = EnvGuard::clear("TRINITY_BACKEND");

    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::BROWSER_WEBGPU);
    // WASM should ONLY have BROWSER_WEBGPU
    assert!(!backends.contains(wgpu::Backends::VULKAN));
    assert!(!backends.contains(wgpu::Backends::METAL));
    assert!(!backends.contains(wgpu::Backends::DX12));
    assert!(!backends.contains(wgpu::Backends::GL));
}

#[test]
#[cfg(all(not(target_arch = "wasm32"), target_os = "windows"))]
fn test_platform_default_windows() {
    let _guard = EnvGuard::clear("TRINITY_BACKEND");

    let backends = TrinityInstance::select_backends();
    // Windows should have Vulkan, DX12, and GL (in that priority order)
    assert!(backends.contains(wgpu::Backends::VULKAN));
    assert!(backends.contains(wgpu::Backends::DX12));
    assert!(backends.contains(wgpu::Backends::GL));
    // Metal should NOT be present on Windows
    assert!(!backends.contains(wgpu::Backends::METAL));
}

#[test]
#[cfg(all(not(target_arch = "wasm32"), target_os = "macos"))]
fn test_platform_default_macos() {
    let _guard = EnvGuard::clear("TRINITY_BACKEND");

    let backends = TrinityInstance::select_backends();
    // macOS should ONLY have Metal
    assert_eq!(backends, wgpu::Backends::METAL);
    assert!(!backends.contains(wgpu::Backends::VULKAN));
    assert!(!backends.contains(wgpu::Backends::DX12));
    assert!(!backends.contains(wgpu::Backends::GL));
}

#[test]
#[cfg(all(not(target_arch = "wasm32"), target_os = "linux"))]
fn test_platform_default_linux() {
    let _guard = EnvGuard::clear("TRINITY_BACKEND");

    let backends = TrinityInstance::select_backends();
    // Linux should have Vulkan and GL (in that priority order)
    assert!(backends.contains(wgpu::Backends::VULKAN));
    assert!(backends.contains(wgpu::Backends::GL));
    // DX12 and Metal should NOT be present on Linux
    assert!(!backends.contains(wgpu::Backends::DX12));
    assert!(!backends.contains(wgpu::Backends::METAL));
}

// ============================================================================
// Path M: Backend priority description verification
// ============================================================================

#[test]
#[cfg(target_arch = "wasm32")]
fn test_backend_priority_description_wasm() {
    let desc = TrinityInstance::backend_priority_description();
    assert_eq!(desc, "WASM: BROWSER_WEBGPU only");
}

#[test]
#[cfg(all(not(target_arch = "wasm32"), target_os = "windows"))]
fn test_backend_priority_description_windows() {
    let desc = TrinityInstance::backend_priority_description();
    assert_eq!(desc, "Windows: Vulkan > DX12 > OpenGL");
}

#[test]
#[cfg(all(not(target_arch = "wasm32"), target_os = "macos"))]
fn test_backend_priority_description_macos() {
    let desc = TrinityInstance::backend_priority_description();
    assert_eq!(desc, "macOS: Metal only");
}

#[test]
#[cfg(all(not(target_arch = "wasm32"), target_os = "linux"))]
fn test_backend_priority_description_linux() {
    let desc = TrinityInstance::backend_priority_description();
    assert_eq!(desc, "Linux: Vulkan > OpenGL");
}

// ============================================================================
// Path N: Environment override takes priority over platform default
// ============================================================================

#[test]
#[cfg(target_os = "linux")]
fn test_env_overrides_platform_default_linux() {
    // On Linux, platform default is Vulkan | GL
    // But env override should take priority
    let _guard = EnvGuard::set("TRINITY_BACKEND", "gl");

    let backends = TrinityInstance::select_backends();
    // Should be ONLY GL, not Vulkan | GL
    assert_eq!(backends, wgpu::Backends::GL);
    assert!(!backends.contains(wgpu::Backends::VULKAN));
}

#[test]
#[cfg(target_os = "windows")]
fn test_env_overrides_platform_default_windows() {
    // On Windows, platform default is Vulkan | DX12 | GL
    // But env override should take priority
    let _guard = EnvGuard::set("TRINITY_BACKEND", "vulkan");

    let backends = TrinityInstance::select_backends();
    // Should be ONLY Vulkan
    assert_eq!(backends, wgpu::Backends::VULKAN);
    assert!(!backends.contains(wgpu::Backends::DX12));
    assert!(!backends.contains(wgpu::Backends::GL));
}

// ============================================================================
// Path O: Incompatible backend for platform falls back gracefully
// ============================================================================

#[test]
#[cfg(target_os = "linux")]
fn test_incompatible_dx12_on_linux() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "dx12");

    // DX12 is Windows-only, Linux should fall back to defaults
    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::GL);
    // Verify DX12 is NOT in the result
    assert!(!backends.contains(wgpu::Backends::DX12));
}

#[test]
#[cfg(target_os = "linux")]
fn test_incompatible_metal_on_linux() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "metal");

    // Metal is macOS-only, Linux should fall back to defaults
    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::GL);
    // Verify Metal is NOT in the result
    assert!(!backends.contains(wgpu::Backends::METAL));
}

#[test]
#[cfg(target_os = "linux")]
fn test_incompatible_webgpu_on_linux() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "webgpu");

    // BROWSER_WEBGPU is WASM-only, Linux should fall back to defaults
    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::GL);
    // Verify BROWSER_WEBGPU is NOT in the result
    assert!(!backends.contains(wgpu::Backends::BROWSER_WEBGPU));
}

#[test]
#[cfg(target_os = "windows")]
fn test_incompatible_metal_on_windows() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "metal");

    // Metal is macOS-only, Windows should fall back to defaults
    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::GL);
}

#[test]
#[cfg(target_os = "macos")]
fn test_incompatible_dx12_on_macos() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "dx12");

    // DX12 is Windows-only, macOS should fall back to defaults
    let backends = TrinityInstance::select_backends();
    assert_eq!(backends, wgpu::Backends::METAL);
}

// ============================================================================
// TrinityInstance construction tests
// ============================================================================

#[test]
fn test_instance_new_without_env() {
    let _guard = EnvGuard::clear("TRINITY_BACKEND");

    let instance = TrinityInstance::new();

    // Verify backends match platform defaults
    #[cfg(target_os = "linux")]
    assert_eq!(instance.backends(), wgpu::Backends::VULKAN | wgpu::Backends::GL);

    #[cfg(target_os = "windows")]
    assert_eq!(instance.backends(), wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::GL);

    #[cfg(target_os = "macos")]
    assert_eq!(instance.backends(), wgpu::Backends::METAL);
}

#[test]
fn test_instance_new_with_env_override() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "vulkan");

    let instance = TrinityInstance::new();
    assert_eq!(instance.backends(), wgpu::Backends::VULKAN);
}

#[test]
fn test_instance_with_backends_explicit() {
    // Explicitly set backends, bypassing env and platform logic
    let instance = TrinityInstance::with_backends(wgpu::Backends::all());
    assert_eq!(instance.backends(), wgpu::Backends::all());
}

#[test]
fn test_instance_with_backends_single() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::GL);
    assert_eq!(instance.backends(), wgpu::Backends::GL);
}

#[test]
fn test_instance_default_trait() {
    let _guard = EnvGuard::clear("TRINITY_BACKEND");

    let instance: TrinityInstance = Default::default();

    #[cfg(target_os = "linux")]
    assert_eq!(instance.backends(), wgpu::Backends::VULKAN | wgpu::Backends::GL);
}

// ============================================================================
// Instance inner access tests
// ============================================================================

#[test]
fn test_instance_inner_reference() {
    let instance = TrinityInstance::new();
    let _inner: &wgpu::Instance = instance.inner();
    // Just verify we can get a reference without panic
}

#[test]
fn test_instance_into_inner_ownership() {
    let instance = TrinityInstance::new();
    let _inner: wgpu::Instance = instance.into_inner();
    // Verify ownership transfer works
}

// ============================================================================
// Debug formatting test
// ============================================================================

#[test]
fn test_instance_debug_format() {
    let instance = TrinityInstance::new();
    let debug_str = format!("{:?}", instance);

    // Verify debug output contains expected fields
    assert!(debug_str.contains("TrinityInstance"));
    assert!(debug_str.contains("backends"));
}

// ============================================================================
// Adapter enumeration test (basic smoke test)
// ============================================================================

#[test]
fn test_enumerate_adapters_returns_vec() {
    let _guard = EnvGuard::clear("TRINITY_BACKEND");

    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();

    // Just verify it returns a Vec (may be empty on headless CI)
    // We're testing the API, not the hardware
    let _ = adapters.len();
}

// ============================================================================
// Multiple backend combination tests
// ============================================================================

#[test]
fn test_backends_bitflag_combination() {
    let instance = TrinityInstance::with_backends(
        wgpu::Backends::VULKAN | wgpu::Backends::GL
    );

    let backends = instance.backends();
    assert!(backends.contains(wgpu::Backends::VULKAN));
    assert!(backends.contains(wgpu::Backends::GL));
    assert!(!backends.contains(wgpu::Backends::DX12));
}

// ============================================================================
// Edge case: Very long/unusual env values
// ============================================================================

#[test]
fn test_env_very_long_value_falls_back() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", &"x".repeat(10000));

    // Absurdly long value should fall back gracefully
    let backends = TrinityInstance::select_backends();

    #[cfg(target_os = "linux")]
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::GL);
}

#[test]
fn test_env_special_characters_falls_back() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "vulkan; rm -rf /");

    // Special characters should just be treated as invalid
    let backends = TrinityInstance::select_backends();

    #[cfg(target_os = "linux")]
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::GL);
}

#[test]
fn test_env_unicode_falls_back() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "vulkan\u{1F600}");

    // Unicode characters make it invalid
    let backends = TrinityInstance::select_backends();

    #[cfg(target_os = "linux")]
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::GL);
}

#[test]
fn test_env_numeric_falls_back() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "12345");

    // Pure numeric should fall back
    let backends = TrinityInstance::select_backends();

    #[cfg(target_os = "linux")]
    assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::GL);
}

// ============================================================================
// Test concurrent env access (race condition coverage)
// ============================================================================

#[test]
fn test_select_backends_consistent_result() {
    let _guard = EnvGuard::set("TRINITY_BACKEND", "vulkan");

    // Multiple calls should return consistent results
    let b1 = TrinityInstance::select_backends();
    let b2 = TrinityInstance::select_backends();
    let b3 = TrinityInstance::select_backends();

    assert_eq!(b1, b2);
    assert_eq!(b2, b3);
    assert_eq!(b1, wgpu::Backends::VULKAN);
}
