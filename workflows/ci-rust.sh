#!/usr/bin/env bash
# ============================================================================
# ci-rust.sh — Local Rust CI/CD Pipeline
# ============================================================================
# Usage:
#   ./ci-rust.sh              # Run full pipeline (fmt → lint → test → build → release)
#   ./ci-rust.sh fmt          # Format check only
#   ./ci-rust.sh lint         # Clippy only
#   ./ci-rust.sh test         # Tests only
#   ./ci-rust.sh build        # Release build only
#   ./ci-rust.sh release      # Build + copy artifact to release/
#   ./ci-rust.sh quick        # fmt + lint only (used by pre-commit hook)
#   ./ci-rust.sh gate         # fmt + lint + test + build (used by pre-push hook)
#
# Exit codes:
#   0 = all stages passed
#   1 = stage failure (pipeline halts on first failure)
# ============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
RELEASE_DIR="${PROJECT_ROOT}/release"
CARGO_ARGS="${CARGO_ARGS:-}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Colors (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    RED='' GREEN='' YELLOW='' CYAN='' BOLD='' RESET=''
fi

# --- Helpers ------------------------------------------------------------------
stage_header() {
    echo -e "\n${BOLD}${CYAN}═══════════════════════════════════════════════════${RESET}"
    echo -e "${BOLD}${CYAN}  STAGE: $1${RESET}"
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════${RESET}\n"
}

pass() { echo -e "\n${GREEN}✓ $1 passed${RESET}"; }
fail() { echo -e "\n${RED}✗ $1 FAILED${RESET}"; exit 1; }
warn() { echo -e "${YELLOW}⚠ $1${RESET}"; }
info() { echo -e "${CYAN}→ $1${RESET}"; }

elapsed() {
    local start=$1
    local end=$(date +%s)
    echo "$((end - start))s"
}

# --- Prerequisite check -------------------------------------------------------
check_prereqs() {
    local missing=0
    for cmd in cargo rustfmt; do
        if ! command -v "$cmd" &>/dev/null; then
            warn "Missing: $cmd"
            missing=1
        fi
    done
    if [ $missing -eq 1 ]; then
        echo -e "${RED}Install missing tools before running pipeline.${RESET}"
        exit 1
    fi

    # Verify we're in a Rust project
    if [ ! -f "${PROJECT_ROOT}/Cargo.toml" ]; then
        echo -e "${RED}No Cargo.toml found at project root: ${PROJECT_ROOT}${RESET}"
        exit 1
    fi
}

# --- Stages -------------------------------------------------------------------

stage_fmt() {
    stage_header "FORMAT CHECK"
    local start=$(date +%s)
    cd "$PROJECT_ROOT"

    info "Running cargo fmt --check"
    if cargo fmt --check ${CARGO_ARGS} 2>&1; then
        pass "Format check ($(elapsed $start))"
    else
        echo ""
        warn "Auto-fix with: cargo fmt"
        fail "Format check"
    fi
}

stage_lint() {
    stage_header "LINT (CLIPPY)"
    local start=$(date +%s)
    cd "$PROJECT_ROOT"

    info "Running cargo clippy -- -D warnings"
    if cargo clippy ${CARGO_ARGS} -- -D warnings 2>&1; then
        pass "Clippy ($(elapsed $start))"
    else
        fail "Clippy"
    fi
}

stage_test() {
    stage_header "TEST"
    local start=$(date +%s)
    cd "$PROJECT_ROOT"

    info "Running cargo test"
    if cargo test ${CARGO_ARGS} 2>&1; then
        pass "Tests ($(elapsed $start))"
    else
        fail "Tests"
    fi
}

stage_build() {
    stage_header "BUILD (RELEASE)"
    local start=$(date +%s)
    cd "$PROJECT_ROOT"

    info "Running cargo build --release"
    if cargo build --release ${CARGO_ARGS} 2>&1; then
        pass "Release build ($(elapsed $start))"
    else
        fail "Release build"
    fi
}

stage_release() {
    stage_build

    stage_header "RELEASE ARTIFACT"
    cd "$PROJECT_ROOT"

    # Extract package name from Cargo.toml
    local pkg_name
    pkg_name=$(grep -m1 '^name' Cargo.toml | sed 's/name *= *"\(.*\)"/\1/')

    # Extract version from Cargo.toml
    local pkg_version
    pkg_version=$(grep -m1 '^version' Cargo.toml | sed 's/version *= *"\(.*\)"/\1/')

    local artifact_name="${pkg_name}-v${pkg_version}-${TIMESTAMP}"
    local artifact_dir="${RELEASE_DIR}/${artifact_name}"

    mkdir -p "$artifact_dir"

    # Copy binary
    local binary_path="target/release/${pkg_name}"
    if [ -f "$binary_path" ]; then
        cp "$binary_path" "$artifact_dir/"
        info "Binary: ${artifact_dir}/${pkg_name}"
    else
        # Workspace — copy all binaries from target/release that aren't deps
        local found=0
        for bin in target/release/*; do
            if [ -f "$bin" ] && [ -x "$bin" ] && file "$bin" | grep -q "ELF"; then
                local basename=$(basename "$bin")
                # Skip common build artifacts
                [[ "$basename" == *.d ]] && continue
                [[ "$basename" == build-script-* ]] && continue
                cp "$bin" "$artifact_dir/"
                info "Binary: ${artifact_dir}/${basename}"
                found=1
            fi
        done
        if [ $found -eq 0 ]; then
            warn "No release binaries found (library crate?)"
        fi
    fi

    # Generate build metadata
    cat > "${artifact_dir}/BUILD_META.txt" <<EOF
package: ${pkg_name}
version: ${pkg_version}
timestamp: ${TIMESTAMP}
rustc: $(rustc --version)
commit: $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
branch: $(git branch --show-current 2>/dev/null || echo "unknown")
host: $(hostname)
EOF

    info "Metadata: ${artifact_dir}/BUILD_META.txt"

    # Checksum
    (cd "$artifact_dir" && find . -type f ! -name 'SHA256SUMS' -exec sha256sum {} \; > SHA256SUMS)
    info "Checksums: ${artifact_dir}/SHA256SUMS"

    echo ""
    info "Release artifact: ${artifact_dir}"
    pass "Release artifact"
}

# --- Pipeline runners ---------------------------------------------------------

run_full() {
    echo -e "${BOLD}${CYAN}"
    echo "╔═══════════════════════════════════════════════════╗"
    echo "║         RUST CI/CD — FULL PIPELINE                ║"
    echo "╚═══════════════════════════════════════════════════╝"
    echo -e "${RESET}"

    local pipeline_start=$(date +%s)
    check_prereqs

    stage_fmt
    stage_lint
    stage_test
    stage_release

    echo -e "\n${BOLD}${GREEN}"
    echo "╔═══════════════════════════════════════════════════╗"
    echo "║         ALL STAGES PASSED ✓                       ║"
    echo "╚═══════════════════════════════════════════════════╝"
    echo -e "${RESET}"
    info "Total time: $(elapsed $pipeline_start)"
}

run_quick() {
    check_prereqs
    stage_fmt
    stage_lint
}

run_gate() {
    check_prereqs
    stage_fmt
    stage_lint
    stage_test
    stage_build
}

# --- Entrypoint ---------------------------------------------------------------

case "${1:-full}" in
    fmt|format)  check_prereqs; stage_fmt ;;
    lint|clippy) check_prereqs; stage_lint ;;
    test)        check_prereqs; stage_test ;;
    build)       check_prereqs; stage_build ;;
    release)     check_prereqs; stage_release ;;
    quick)       run_quick ;;
    gate)        run_gate ;;
    full)        run_full ;;
    *)
        echo "Usage: $0 {fmt|lint|test|build|release|quick|gate|full}"
        echo ""
        echo "  fmt      Format check (cargo fmt --check)"
        echo "  lint     Clippy with -D warnings"
        echo "  test     cargo test"
        echo "  build    cargo build --release"
        echo "  release  Build + artifact + metadata + checksums"
        echo "  quick    fmt + lint (pre-commit)"
        echo "  gate     fmt + lint + test + build (pre-push)"
        echo "  full     All stages including release artifact"
        exit 1
        ;;
esac
