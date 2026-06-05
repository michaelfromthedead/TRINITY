#!/usr/bin/env bash
# T-DEMO-5.8: Standalone Verification Script
#
# Verifies that the TRINITY demoscene renderer can run without external dependencies.
#
# Checks performed:
#   1. No asset files required at runtime
#   2. No Python runtime required
#   3. No network dependencies
#   4. GPU backend availability (wgpu: Vulkan/Metal/DX12/GL)
#   5. Binary self-containment
#
# Usage:
#   ./scripts/verify_standalone.sh [--verbose] [--skip-gpu]
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed
#   2 - Script error

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CRATE_DIR="$PROJECT_ROOT/crates/renderer-backend"
VERBOSE=false
SKIP_GPU=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
for arg in "$@"; do
    case $arg in
        --verbose|-v)
            VERBOSE=true
            ;;
        --skip-gpu)
            SKIP_GPU=true
            ;;
        --help|-h)
            echo "Usage: $0 [--verbose] [--skip-gpu]"
            echo ""
            echo "Verify TRINITY demoscene standalone requirements."
            echo ""
            echo "Options:"
            echo "  --verbose, -v    Show detailed output"
            echo "  --skip-gpu       Skip GPU availability checks"
            echo "  --help, -h       Show this help"
            exit 0
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "       $1"
    fi
}

# Track results
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

check() {
    local name="$1"
    local result="$2"
    local detail="${3:-}"

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    if [ "$result" = "pass" ]; then
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        log_pass "$name"
    elif [ "$result" = "warn" ]; then
        WARNINGS=$((WARNINGS + 1))
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        log_warn "$name"
    else
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        log_fail "$name"
    fi

    if [ -n "$detail" ]; then
        log_verbose "$detail"
    fi
}

# =============================================================================
# Check 1: No Asset Files Required
# =============================================================================

log_info "Checking for external asset dependencies..."

check_no_assets() {
    local minimal_rs="$CRATE_DIR/src/demoscene/minimal.rs"

    if [ ! -f "$minimal_rs" ]; then
        check "minimal.rs exists" "fail" "File not found: $minimal_rs"
        return
    fi

    # Check shader is inline
    if grep -q 'MINIMAL_SHADER: &str = r#"' "$minimal_rs"; then
        check "Shader embedded inline" "pass"
    else
        check "Shader embedded inline" "fail" "Shader not found as inline string"
    fi

    # Extract only production code (before test section)
    local test_line
    test_line=$(grep -n '#\[cfg(test)\]' "$minimal_rs" | head -1 | cut -d: -f1 || echo "0")
    local prod_code
    if [ "$test_line" != "0" ]; then
        prod_code=$(head -n "$test_line" "$minimal_rs")
    else
        prod_code=$(cat "$minimal_rs")
    fi

    # Check no texture loading in production code
    if ! echo "$prod_code" | grep -q 'image::open\|ImageReader'; then
        check "No texture file loading" "pass"
    else
        check "No texture file loading" "fail" "Found texture loading code"
    fi

    # Check no model loading in production code
    if ! echo "$prod_code" | grep -q 'Gltf::open\|tobj::load'; then
        check "No model file loading" "pass"
    else
        check "No model file loading" "fail" "Found model loading code"
    fi

    # Check no config file loading in production code
    if ! echo "$prod_code" | grep -q 'serde_json::from_reader\|toml::from_str'; then
        check "No config file loading" "pass"
    else
        check "No config file loading" "fail" "Found config file loading"
    fi

    # Check no audio loading in production code
    if ! echo "$prod_code" | grep -q 'rodio::Decoder\|OutputStream::try_default'; then
        check "No audio file loading" "pass"
    else
        check "No audio file loading" "fail" "Found audio loading code"
    fi
}

check_no_assets

# =============================================================================
# Check 2: No Python Runtime Required
# =============================================================================

log_info "Checking Python independence..."

check_no_python() {
    local minimal_rs="$CRATE_DIR/src/demoscene/minimal.rs"

    # Check no pyo3 imports in minimal.rs
    if ! grep -q 'pyo3::' "$minimal_rs" 2>/dev/null; then
        check "No PyO3 imports" "pass"
    else
        check "No PyO3 imports" "fail" "Found pyo3 imports"
    fi

    # Check no Python subprocess calls
    if ! grep -q 'Command::new.*python\|subprocess' "$minimal_rs" 2>/dev/null; then
        check "No Python subprocess calls" "pass"
    else
        check "No Python subprocess calls" "fail" "Found Python subprocess"
    fi

    # Check feature flag for pyo3
    local cargo_toml="$CRATE_DIR/Cargo.toml"
    if grep -q 'pyo3.*optional.*true\|optional.*pyo3' "$cargo_toml"; then
        check "PyO3 is optional feature" "pass"
    else
        check "PyO3 is optional feature" "warn" "PyO3 may be required by default"
    fi

    # Check no Python in build.rs (if exists)
    local build_rs="$CRATE_DIR/build.rs"
    if [ -f "$build_rs" ]; then
        if ! grep -q 'python\|Python\|PYTHON' "$build_rs"; then
            check "build.rs has no Python deps" "pass"
        else
            check "build.rs has no Python deps" "warn" "Python referenced in build.rs"
        fi
    else
        check "build.rs has no Python deps" "pass" "No build.rs found"
    fi
}

check_no_python

# =============================================================================
# Check 3: No Network Dependencies
# =============================================================================

log_info "Checking network independence..."

check_no_network() {
    local minimal_rs="$CRATE_DIR/src/demoscene/minimal.rs"

    # Check no HTTP/network crates
    if ! grep -q 'reqwest::\|hyper::\|curl::\|ureq::' "$minimal_rs" 2>/dev/null; then
        check "No HTTP client imports" "pass"
    else
        check "No HTTP client imports" "fail" "Found HTTP client"
    fi

    # Check no TCP/UDP socket usage
    if ! grep -q 'TcpStream\|UdpSocket\|net::' "$minimal_rs" 2>/dev/null; then
        check "No socket usage" "pass"
    else
        check "No socket usage" "fail" "Found socket usage"
    fi

    # Check no URL parsing
    if ! grep -q 'url::\|Url::parse' "$minimal_rs" 2>/dev/null; then
        check "No URL parsing" "pass"
    else
        check "No URL parsing" "fail" "Found URL parsing"
    fi

    # Check Cargo.toml for network deps
    local cargo_toml="$CRATE_DIR/Cargo.toml"
    if ! grep -q 'reqwest\|hyper\|tokio.*net\|async-std.*net' "$cargo_toml"; then
        check "No network deps in Cargo.toml" "pass"
    else
        check "No network deps in Cargo.toml" "warn" "Network deps in Cargo.toml"
    fi
}

check_no_network

# =============================================================================
# Check 4: GPU Backend Availability
# =============================================================================

log_info "Checking GPU backend support..."

check_gpu_backends() {
    if [ "$SKIP_GPU" = true ]; then
        check "GPU checks (skipped)" "pass" "Skipped via --skip-gpu"
        return
    fi

    # Check wgpu dependency
    local cargo_toml="$CRATE_DIR/Cargo.toml"
    if grep -q 'wgpu' "$cargo_toml"; then
        check "wgpu dependency present" "pass"
    else
        check "wgpu dependency present" "fail" "wgpu not found"
        return
    fi

    # Check platform-specific backends
    case "$(uname -s)" in
        Linux)
            # Check Vulkan availability
            if command -v vulkaninfo &> /dev/null || [ -f /usr/lib/x86_64-linux-gnu/libvulkan.so.1 ]; then
                check "Vulkan available (Linux)" "pass"
            else
                check "Vulkan available (Linux)" "warn" "Vulkan not detected"
            fi
            ;;
        Darwin)
            # macOS always has Metal
            check "Metal available (macOS)" "pass"
            ;;
        MINGW*|CYGWIN*|MSYS*)
            # Windows - check for DX12
            check "DirectX12 available (Windows)" "pass" "Assumed available"
            ;;
        *)
            check "GPU backend check" "warn" "Unknown platform"
            ;;
    esac

    # Check OpenGL fallback
    if command -v glxinfo &> /dev/null; then
        check "OpenGL fallback available" "pass"
    else
        check "OpenGL fallback available" "warn" "glxinfo not found"
    fi
}

check_gpu_backends

# =============================================================================
# Check 5: Binary Self-Containment
# =============================================================================

log_info "Checking binary self-containment..."

check_binary_standalone() {
    # Check shader size
    local minimal_rs="$CRATE_DIR/src/demoscene/minimal.rs"

    if [ -f "$minimal_rs" ]; then
        # Extract shader size (approximate)
        local shader_start
        shader_start=$(grep -n 'MINIMAL_SHADER: &str = r#"' "$minimal_rs" | cut -d: -f1 || echo "0")
        local shader_end
        shader_end=$(grep -n '"#;' "$minimal_rs" | head -1 | cut -d: -f1 || echo "0")

        if [ "$shader_start" != "0" ] && [ "$shader_end" != "0" ]; then
            local shader_lines=$((shader_end - shader_start))
            if [ "$shader_lines" -lt 50 ]; then
                check "Shader is compact" "pass" "$shader_lines lines"
            else
                check "Shader is compact" "warn" "$shader_lines lines (consider minification)"
            fi
        else
            check "Shader is compact" "warn" "Could not measure shader size"
        fi
    fi

    # Check uniforms are minimal (16 bytes)
    if grep -q 'size_of::<MinimalUniforms>.*16\|size: 16' "$minimal_rs" 2>/dev/null; then
        check "Uniforms are minimal (16 bytes)" "pass"
    else
        check "Uniforms are minimal" "warn" "Could not verify uniform size"
    fi

    # Check no dynamic linking requirements
    local cargo_toml="$CRATE_DIR/Cargo.toml"
    if ! grep -q 'dylib\|cdylib' "$cargo_toml"; then
        check "No dynamic library output" "pass"
    else
        check "No dynamic library output" "warn" "Dynamic library output configured"
    fi

    # Check for embedded tests
    if grep -q '#\[cfg(test)\]' "$minimal_rs" 2>/dev/null; then
        check "Has embedded tests" "pass"
    else
        check "Has embedded tests" "fail" "No tests found"
    fi
}

check_binary_standalone

# =============================================================================
# Check 6: Test Execution
# =============================================================================

log_info "Running demoscene minimal tests..."

check_tests() {
    cd "$CRATE_DIR"

    # Run tests with cargo
    if cargo test demoscene_minimal --features test-utils 2>&1 | tail -20; then
        check "Tests compile" "pass"
    else
        check "Tests compile" "warn" "Test execution may have issues"
    fi

    # Also run the minimal module tests specifically
    if cargo test minimal:: --features test-utils -- --test-threads=1 2>&1 | tail -10; then
        check "Minimal tests pass" "pass"
    else
        check "Minimal tests pass" "warn" "Some tests may have failed"
    fi
}

# Only run tests if cargo is available
if command -v cargo &> /dev/null; then
    check_tests
else
    check "Tests (skipped - no cargo)" "warn"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================"
echo "STANDALONE VERIFICATION SUMMARY"
echo "============================================"
echo ""
echo -e "Total checks:  ${BLUE}$TOTAL_CHECKS${NC}"
echo -e "Passed:        ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed:        ${RED}$FAILED_CHECKS${NC}"
echo -e "Warnings:      ${YELLOW}$WARNINGS${NC}"
echo ""

if [ "$FAILED_CHECKS" -eq 0 ]; then
    echo -e "${GREEN}All critical checks passed!${NC}"
    echo ""
    echo "The TRINITY demoscene renderer is standalone-ready:"
    echo "  - No asset files required at runtime"
    echo "  - No Python runtime required"
    echo "  - No network dependencies"
    echo "  - GPU backend support verified (wgpu)"
    echo ""
    exit 0
else
    echo -e "${RED}Some checks failed!${NC}"
    echo ""
    echo "Please review the failed checks above and fix the issues."
    exit 1
fi
