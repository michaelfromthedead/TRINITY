// Blackbox contract tests for T-WGPU-P1.1.2 Backend Selection
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::device::TrinityInstance`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/device/instance.rs
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_TODO.md (T-WGPU-P1.1.2)
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_ARCH.md (TrinityInstance spec)
//
// Acceptance criteria (T-WGPU-P1.1.2):
//   - Windows: Vulkan > DX12 > OpenGL
//   - macOS: Metal only
//   - Linux: Vulkan > OpenGL
//   - WASM: BROWSER_WEBGPU only
//   - Configurable override via environment variable (TRINITY_BACKEND)
//
// Valid TRINITY_BACKEND values (from contract):
//   - vulkan, dx12, metal, opengl/gl, webgpu
//   - primary, secondary, all
//
// Test design rationale:
//   Equivalence partitioning:
//     - Platform default backends (Linux, Windows, macOS, WASM)
//     - Env var override with valid values
//     - Env var override with invalid values
//   Boundary cases:
//     - Platform-incompatible backend requests (e.g., Metal on Linux)
//     - Empty/whitespace env var values
//   Error cases:
//     - Invalid backend string should fall back to platform default
//     - Unavailable backend should gracefully degrade

use renderer_backend::device::TrinityInstance;
use std::env;
use std::sync::Mutex;

// Mutex to serialize tests that modify environment variables
static ENV_MUTEX: Mutex<()> = Mutex::new(());

/// RAII guard for environment variable manipulation in tests.
/// Restores the original value (or removes the var) when dropped.
struct EnvGuard {
    key: &'static str,
    original: Option<String>,
}

impl EnvGuard {
    /// Set an environment variable, returning a guard that restores it on drop.
    fn set(key: &'static str, value: &str) -> Self {
        let original = env::var(key).ok();
        env::set_var(key, value);
        Self { key, original }
    }

    /// Clear an environment variable, returning a guard that restores it on drop.
    fn clear(key: &'static str) -> Self {
        let original = env::var(key).ok();
        env::remove_var(key);
        Self { key, original }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(val) => env::set_var(self.key, val),
            None => env::remove_var(self.key),
        }
    }
}

// =============================================================================
// 1. Platform Default Backend Selection
// =============================================================================

/// Verifies that on Linux, the default includes Vulkan (primary choice per contract).
///
/// Contract: Linux: Vulkan > OpenGL
#[test]
#[cfg(target_os = "linux")]
fn test_linux_default_includes_vulkan() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::clear("TRINITY_BACKEND");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    // Contract says Linux prefers Vulkan
    assert!(
        backends.contains(wgpu::Backends::VULKAN),
        "Linux default backends should include Vulkan, got {:?}",
        backends
    );
}

/// Verifies that on Windows, the default includes Vulkan and/or DX12 (per contract priority).
///
/// Contract: Windows: Vulkan > DX12 > OpenGL
#[test]
#[cfg(target_os = "windows")]
fn test_windows_default_includes_primary_backends() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::clear("TRINITY_BACKEND");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    // Contract says Windows prefers Vulkan, then DX12
    let has_primary = backends.contains(wgpu::Backends::VULKAN)
        || backends.contains(wgpu::Backends::DX12);

    assert!(
        has_primary,
        "Windows default backends should include Vulkan or DX12, got {:?}",
        backends
    );
}

/// Verifies that on macOS, the default is Metal only.
///
/// Contract: macOS: Metal only
#[test]
#[cfg(target_os = "macos")]
fn test_macos_default_is_metal() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::clear("TRINITY_BACKEND");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    // Contract says macOS uses Metal only
    assert!(
        backends.contains(wgpu::Backends::METAL),
        "macOS default backends should include Metal, got {:?}",
        backends
    );
}

/// Verifies that PRIMARY backends are used by default on non-WASM platforms.
///
/// Contract: Desktop platforms use PRIMARY backends (Vulkan/Metal/DX12).
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_desktop_uses_primary_backends_by_default() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::clear("TRINITY_BACKEND");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    // PRIMARY = Vulkan | Metal | DX12
    let has_primary_backend = backends.contains(wgpu::Backends::VULKAN)
        || backends.contains(wgpu::Backends::METAL)
        || backends.contains(wgpu::Backends::DX12);

    assert!(
        has_primary_backend,
        "Desktop should use PRIMARY backends by default, got {:?}",
        backends
    );
}

// =============================================================================
// 2. TRINITY_BACKEND Environment Variable Override - Valid Values
// =============================================================================

/// Verifies that TRINITY_BACKEND=vulkan selects Vulkan backend.
///
/// Contract: Configurable override via environment variable.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_env_override_vulkan() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "vulkan");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    assert!(
        backends.contains(wgpu::Backends::VULKAN),
        "TRINITY_BACKEND=vulkan should select Vulkan, got {:?}",
        backends
    );
}

/// Verifies that TRINITY_BACKEND=opengl selects OpenGL backend.
///
/// Contract: Configurable override via environment variable.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_env_override_opengl() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "opengl");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    assert!(
        backends.contains(wgpu::Backends::GL),
        "TRINITY_BACKEND=opengl should select GL, got {:?}",
        backends
    );
}

/// Verifies that TRINITY_BACKEND=gl is accepted as alias for opengl.
///
/// Contract: Valid env values: vulkan, dx12, metal, opengl/gl, webgpu
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_env_override_gl_alias() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "gl");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    assert!(
        backends.contains(wgpu::Backends::GL),
        "TRINITY_BACKEND=gl should select GL (alias for opengl), got {:?}",
        backends
    );
}

/// Verifies that TRINITY_BACKEND=primary selects PRIMARY backends.
///
/// Contract: Configurable override via environment variable.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_env_override_primary() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "primary");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    // PRIMARY = Vulkan | Metal | DX12
    let expected = wgpu::Backends::PRIMARY;
    assert_eq!(
        backends, expected,
        "TRINITY_BACKEND=primary should select PRIMARY backends, got {:?}",
        backends
    );
}

/// Verifies that TRINITY_BACKEND=secondary selects SECONDARY (OpenGL) backends.
///
/// Contract: Configurable override via environment variable.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_env_override_secondary() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "secondary");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    // SECONDARY = GL
    assert!(
        backends.contains(wgpu::Backends::GL),
        "TRINITY_BACKEND=secondary should select GL, got {:?}",
        backends
    );
}

/// Verifies that TRINITY_BACKEND=all selects all available backends.
///
/// Contract: Configurable override via environment variable.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_env_override_all() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "all");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    // "all" should include at least PRIMARY backends
    let has_vulkan = backends.contains(wgpu::Backends::VULKAN);
    let has_gl = backends.contains(wgpu::Backends::GL);

    // On Linux, "all" should include both Vulkan and GL
    #[cfg(target_os = "linux")]
    assert!(
        has_vulkan && has_gl,
        "TRINITY_BACKEND=all on Linux should include Vulkan and GL, got {:?}",
        backends
    );

    // On any platform, "all" should be at least PRIMARY
    #[cfg(not(target_os = "linux"))]
    {
        let has_primary = backends.contains(wgpu::Backends::VULKAN)
            || backends.contains(wgpu::Backends::METAL)
            || backends.contains(wgpu::Backends::DX12);
        assert!(
            has_primary,
            "TRINITY_BACKEND=all should include PRIMARY backends, got {:?}",
            backends
        );
    }
}

/// Verifies that TRINITY_BACKEND=dx12 works on Windows.
///
/// Contract: Windows: Vulkan > DX12 > OpenGL
#[test]
#[cfg(target_os = "windows")]
fn test_env_override_dx12_on_windows() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "dx12");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    assert!(
        backends.contains(wgpu::Backends::DX12),
        "TRINITY_BACKEND=dx12 on Windows should select DX12, got {:?}",
        backends
    );
}

/// Verifies that TRINITY_BACKEND=metal works on macOS.
///
/// Contract: macOS: Metal only
#[test]
#[cfg(target_os = "macos")]
fn test_env_override_metal_on_macos() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "metal");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    assert!(
        backends.contains(wgpu::Backends::METAL),
        "TRINITY_BACKEND=metal on macOS should select Metal, got {:?}",
        backends
    );
}

// =============================================================================
// 3. TRINITY_BACKEND Override - Case Insensitivity
// =============================================================================

/// Verifies that TRINITY_BACKEND is case-insensitive (VULKAN, Vulkan, vulkan all work).
///
/// Contract: Environment variable should be user-friendly.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_env_override_case_insensitive_uppercase() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "VULKAN");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    assert!(
        backends.contains(wgpu::Backends::VULKAN),
        "TRINITY_BACKEND=VULKAN (uppercase) should work, got {:?}",
        backends
    );
}

/// Verifies that TRINITY_BACKEND handles mixed case.
///
/// Contract: Environment variable should be user-friendly.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_env_override_case_insensitive_mixed() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "VuLkAn");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    assert!(
        backends.contains(wgpu::Backends::VULKAN),
        "TRINITY_BACKEND=VuLkAn (mixed case) should work, got {:?}",
        backends
    );
}

// =============================================================================
// 4. TRINITY_BACKEND Override - Invalid Values
// =============================================================================

/// Verifies that invalid TRINITY_BACKEND values fall back to platform default.
///
/// Contract: Unknown values should not crash; graceful fallback to platform default.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_env_override_invalid_falls_back_to_default() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "invalid_backend_xyz");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    // Should fall back to platform default (PRIMARY)
    let has_primary = backends.contains(wgpu::Backends::VULKAN)
        || backends.contains(wgpu::Backends::METAL)
        || backends.contains(wgpu::Backends::DX12);

    assert!(
        has_primary,
        "Invalid TRINITY_BACKEND should fall back to platform default, got {:?}",
        backends
    );
}

/// Verifies that empty TRINITY_BACKEND value is treated as unset.
///
/// Contract: Empty env var should behave like unset.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_env_override_empty_string_uses_default() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    // Empty should behave like unset -> platform default
    let has_primary = backends.contains(wgpu::Backends::VULKAN)
        || backends.contains(wgpu::Backends::METAL)
        || backends.contains(wgpu::Backends::DX12);

    assert!(
        has_primary,
        "Empty TRINITY_BACKEND should use platform default, got {:?}",
        backends
    );
}

/// Verifies that whitespace-only TRINITY_BACKEND value is treated as unset.
///
/// Contract: Whitespace-only env var should behave like unset.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_env_override_whitespace_only_uses_default() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "   ");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    // Whitespace should behave like unset -> platform default
    let has_primary = backends.contains(wgpu::Backends::VULKAN)
        || backends.contains(wgpu::Backends::METAL)
        || backends.contains(wgpu::Backends::DX12);

    assert!(
        has_primary,
        "Whitespace TRINITY_BACKEND should use platform default, got {:?}",
        backends
    );
}

// =============================================================================
// 5. Platform-Incompatible Backend Requests
// =============================================================================

/// Verifies that requesting DX12 on non-Windows falls back gracefully.
///
/// Contract: DX12 is Windows-only; non-Windows should fallback.
#[test]
#[cfg(not(any(target_os = "windows", target_arch = "wasm32")))]
fn test_env_override_dx12_on_non_windows_falls_back() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "dx12");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    // DX12 is not available on Linux/macOS; should fall back to platform default
    let has_platform_default = backends.contains(wgpu::Backends::VULKAN)
        || backends.contains(wgpu::Backends::METAL);

    assert!(
        has_platform_default,
        "dx12 on non-Windows should fall back to platform default, got {:?}",
        backends
    );
}

/// Verifies that requesting Metal on non-macOS falls back gracefully.
///
/// Contract: Metal is macOS-only; non-macOS should fallback.
#[test]
#[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
fn test_env_override_metal_on_non_macos_falls_back() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "metal");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    // Metal is not available on Linux/Windows; should fall back to platform default
    let has_platform_default = backends.contains(wgpu::Backends::VULKAN)
        || backends.contains(wgpu::Backends::DX12);

    assert!(
        has_platform_default,
        "metal on non-macOS should fall back to platform default, got {:?}",
        backends
    );
}

/// Verifies that requesting WebGPU on native platform falls back gracefully.
///
/// Contract: WebGPU (BROWSER_WEBGPU) is WASM-only; native should fallback.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_env_override_webgpu_on_native_falls_back() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "webgpu");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    // BROWSER_WEBGPU is not available on native; should fall back to platform default
    let has_platform_default = backends.contains(wgpu::Backends::VULKAN)
        || backends.contains(wgpu::Backends::METAL)
        || backends.contains(wgpu::Backends::DX12);

    assert!(
        has_platform_default,
        "webgpu on native should fall back to platform default, got {:?}",
        backends
    );
}

// =============================================================================
// 6. Backend Selection - Adapter Verification
// =============================================================================

/// Verifies that overriding to Vulkan yields adapters with Vulkan backend.
///
/// Contract: When a specific backend is selected, enumerated adapters should
/// be from that backend (if available).
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_vulkan_override_yields_vulkan_adapters() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "vulkan");

    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();

    // If we have adapters, they should all be Vulkan
    for adapter in &adapters {
        let info = adapter.get_info();
        assert_eq!(
            info.backend,
            wgpu::Backend::Vulkan,
            "With TRINITY_BACKEND=vulkan, adapter '{}' should be Vulkan, got {:?}",
            info.name,
            info.backend
        );
    }
}

/// Verifies that overriding to OpenGL yields adapters with GL backend.
///
/// Contract: When a specific backend is selected, enumerated adapters should
/// be from that backend (if available).
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_opengl_override_yields_gl_adapters() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "opengl");

    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();

    // If we have adapters, they should all be GL
    for adapter in &adapters {
        let info = adapter.get_info();
        assert_eq!(
            info.backend,
            wgpu::Backend::Gl,
            "With TRINITY_BACKEND=opengl, adapter '{}' should be GL, got {:?}",
            info.name,
            info.backend
        );
    }
}

// =============================================================================
// 7. with_backends() vs TRINITY_BACKEND Interaction
// =============================================================================

/// Verifies that with_backends() takes precedence over TRINITY_BACKEND env var.
///
/// Contract: Explicit API should override environment configuration.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_with_backends_overrides_env_var() {
    let _lock = ENV_MUTEX.lock().unwrap();
    // Set env to OpenGL
    let _guard = EnvGuard::set("TRINITY_BACKEND", "opengl");

    // But explicitly request Vulkan via API
    let instance = TrinityInstance::with_backends(wgpu::Backends::VULKAN);
    let backends = instance.backends();

    // API should win over env var
    assert_eq!(
        backends,
        wgpu::Backends::VULKAN,
        "with_backends() should override TRINITY_BACKEND env var"
    );
}

/// Verifies that new() respects TRINITY_BACKEND while with_backends() does not.
///
/// Contract: new() uses env var; with_backends() uses explicit parameter.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_new_respects_env_but_with_backends_does_not() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "opengl");

    // new() should use env var
    let instance_new = TrinityInstance::new();
    assert!(
        instance_new.backends().contains(wgpu::Backends::GL),
        "new() should respect TRINITY_BACKEND=opengl"
    );

    // with_backends() should ignore env var
    let instance_explicit = TrinityInstance::with_backends(wgpu::Backends::VULKAN);
    assert_eq!(
        instance_explicit.backends(),
        wgpu::Backends::VULKAN,
        "with_backends(VULKAN) should ignore TRINITY_BACKEND env var"
    );
}

// =============================================================================
// 8. Backend Query Consistency
// =============================================================================

/// Verifies that backends() returns consistent results across multiple calls.
///
/// Contract: The backend configuration is immutable after construction.
#[test]
fn test_backends_returns_consistent_value() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::clear("TRINITY_BACKEND");

    let instance = TrinityInstance::new();

    let backends1 = instance.backends();
    let backends2 = instance.backends();
    let backends3 = instance.backends();

    assert_eq!(backends1, backends2);
    assert_eq!(backends2, backends3);
}

/// Verifies that backends() on with_backends() instance matches what was requested.
///
/// Contract: backends() returns the Backends the instance was created with.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_backends_matches_with_backends_parameter() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::clear("TRINITY_BACKEND");

    let test_cases = [
        wgpu::Backends::VULKAN,
        wgpu::Backends::GL,
        wgpu::Backends::VULKAN | wgpu::Backends::GL,
        wgpu::Backends::PRIMARY,
        wgpu::Backends::all(),
    ];

    for expected in test_cases {
        let instance = TrinityInstance::with_backends(expected);
        let actual = instance.backends();
        assert_eq!(
            actual, expected,
            "backends() should return exactly what was passed to with_backends()"
        );
    }
}

// =============================================================================
// 9. Edge Cases
// =============================================================================

/// Verifies that creating instances with the same backend multiple times works.
///
/// Contract: Backend selection is reproducible.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_repeated_backend_selection_is_deterministic() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "vulkan");

    let instance1 = TrinityInstance::new();
    let instance2 = TrinityInstance::new();

    assert_eq!(
        instance1.backends(),
        instance2.backends(),
        "Backend selection should be deterministic"
    );
}

/// Verifies that env var with leading/trailing whitespace is trimmed.
///
/// Contract: User-friendly env var parsing.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_env_override_with_whitespace_around_value() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("TRINITY_BACKEND", "  vulkan  ");

    let instance = TrinityInstance::new();
    let backends = instance.backends();

    assert!(
        backends.contains(wgpu::Backends::VULKAN),
        "TRINITY_BACKEND='  vulkan  ' (with whitespace) should work, got {:?}",
        backends
    );
}

// =============================================================================
// 10. Thread Safety of Backend Selection
// =============================================================================

/// Verifies that concurrent instance creation with different env var states is safe.
///
/// Contract: Thread-safe backend selection.
#[test]
fn test_concurrent_instance_creation_thread_safe() {
    use std::thread;

    // Note: We can't safely test env var races, but we can test concurrent
    // instance creation with explicit backends.
    let handles: Vec<_> = (0..4)
        .map(|i| {
            thread::spawn(move || {
                let backends = if i % 2 == 0 {
                    wgpu::Backends::VULKAN
                } else {
                    wgpu::Backends::GL
                };
                let instance = TrinityInstance::with_backends(backends);
                (instance.backends(), backends)
            })
        })
        .collect();

    for handle in handles {
        let (actual, expected) = handle.join().unwrap();
        assert_eq!(
            actual, expected,
            "Each thread should get its requested backends"
        );
    }
}
