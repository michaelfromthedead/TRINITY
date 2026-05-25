#!/usr/bin/env bash
# ============================================================================
# ci-python.sh — Local Python 3.13 CI/CD Pipeline
# ============================================================================
# Usage:
#   ./ci-python.sh              # Run full pipeline (fmt → lint → typecheck → test → build → release)
#   ./ci-python.sh fmt          # Format check only (ruff format)
#   ./ci-python.sh lint         # Lint only (ruff check)
#   ./ci-python.sh typecheck    # Type check only (mypy)
#   ./ci-python.sh test         # Tests only (pytest)
#   ./ci-python.sh build        # Build wheel + sdist
#   ./ci-python.sh release      # Build + copy artifact to release/
#   ./ci-python.sh quick        # fmt + lint only (used by pre-commit hook)
#   ./ci-python.sh gate         # fmt + lint + typecheck + test + build (used by pre-push hook)
#
# Environment:
#   PYTHON        Python binary (default: python3.13, falls back to python3, python)
#   VENV_DIR      Venv path (default: .venv)
#   PYTEST_ARGS   Extra pytest arguments
#   MYPY_ARGS     Extra mypy arguments
#   SRC_DIR       Source directory to check (auto-detected from pyproject.toml or defaults to src/)
#
# Exit codes:
#   0 = all stages passed
#   1 = stage failure (pipeline halts on first failure)
# ============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
RELEASE_DIR="${PROJECT_ROOT}/release"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PYTEST_ARGS="${PYTEST_ARGS:-}"
MYPY_ARGS="${MYPY_ARGS:-}"

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

# --- Python resolution --------------------------------------------------------
resolve_python() {
    if [ -n "${PYTHON:-}" ]; then
        if command -v "$PYTHON" &>/dev/null; then
            echo "$PYTHON"
            return
        fi
    fi

    for candidate in python3.13 python3 python; do
        if command -v "$candidate" &>/dev/null; then
            local ver
            ver=$("$candidate" --version 2>&1 | grep -oP '\d+\.\d+')
            echo "$candidate"
            return
        fi
    done

    echo ""
}

# --- Source dir detection -----------------------------------------------------
detect_src_dir() {
    if [ -n "${SRC_DIR:-}" ]; then
        echo "$SRC_DIR"
        return
    fi

    # Try pyproject.toml [tool.ruff] src setting
    if [ -f "${PROJECT_ROOT}/pyproject.toml" ]; then
        # Look for packages or a name field to guess the source dir
        local pkg
        pkg=$(grep -m1 'name' "${PROJECT_ROOT}/pyproject.toml" 2>/dev/null \
              | sed 's/name *= *"\(.*\)"/\1/' | tr '-' '_' || true)
        if [ -n "$pkg" ] && [ -d "${PROJECT_ROOT}/src/${pkg}" ]; then
            echo "src/${pkg}"
            return
        fi
        if [ -n "$pkg" ] && [ -d "${PROJECT_ROOT}/${pkg}" ]; then
            echo "$pkg"
            return
        fi
    fi

    # Fallback: src/ if it exists, else .
    if [ -d "${PROJECT_ROOT}/src" ]; then
        echo "src"
    else
        echo "."
    fi
}

# --- Prerequisite check -------------------------------------------------------
PYTHON_BIN=""

check_prereqs() {
    PYTHON_BIN=$(resolve_python)
    if [ -z "$PYTHON_BIN" ]; then
        echo -e "${RED}No Python found. Set PYTHON env var or install Python 3.13.${RESET}"
        exit 1
    fi

    local py_ver
    py_ver=$("$PYTHON_BIN" --version 2>&1)
    info "Using: ${py_ver} (${PYTHON_BIN})"

    # Activate venv if it exists
    if [ -f "${VENV_DIR}/bin/activate" ]; then
        source "${VENV_DIR}/bin/activate"
        info "Activated venv: ${VENV_DIR}"
    else
        warn "No venv at ${VENV_DIR} — running against system Python"
        warn "Create one with: ${PYTHON_BIN} -m venv ${VENV_DIR}"
    fi

    # Check for required tools
    local missing=0
    for cmd in ruff pytest mypy; do
        if ! command -v "$cmd" &>/dev/null; then
            warn "Missing: $cmd"
            missing=1
        fi
    done
    if [ $missing -eq 1 ]; then
        warn "Install missing tools: pip install ruff pytest mypy"
        warn "Skipping stages that require missing tools."
    fi

    # Verify project file exists
    if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ] && [ ! -f "${PROJECT_ROOT}/setup.py" ]; then
        warn "No pyproject.toml or setup.py found — some stages may fail"
    fi
}

# --- Stages -------------------------------------------------------------------

stage_fmt() {
    stage_header "FORMAT CHECK (ruff format)"
    local start=$(date +%s)
    cd "$PROJECT_ROOT"

    if ! command -v ruff &>/dev/null; then
        warn "ruff not installed — skipping format check"
        return
    fi

    info "Running ruff format --check"
    if ruff format --check . 2>&1; then
        pass "Format check ($(elapsed $start))"
    else
        echo ""
        warn "Auto-fix with: ruff format ."
        fail "Format check"
    fi
}

stage_lint() {
    stage_header "LINT (ruff check)"
    local start=$(date +%s)
    cd "$PROJECT_ROOT"

    if ! command -v ruff &>/dev/null; then
        warn "ruff not installed — skipping lint"
        return
    fi

    info "Running ruff check"
    if ruff check . 2>&1; then
        pass "Lint ($(elapsed $start))"
    else
        echo ""
        warn "Auto-fix with: ruff check --fix ."
        fail "Lint"
    fi
}

stage_typecheck() {
    stage_header "TYPE CHECK (mypy)"
    local start=$(date +%s)
    cd "$PROJECT_ROOT"

    if ! command -v mypy &>/dev/null; then
        warn "mypy not installed — skipping type check"
        return
    fi

    local src_dir
    src_dir=$(detect_src_dir)
    info "Running mypy ${src_dir} ${MYPY_ARGS}"

    if mypy ${src_dir} ${MYPY_ARGS} 2>&1; then
        pass "Type check ($(elapsed $start))"
    else
        fail "Type check"
    fi
}

stage_test() {
    stage_header "TEST (pytest)"
    local start=$(date +%s)
    cd "$PROJECT_ROOT"

    if ! command -v pytest &>/dev/null; then
        warn "pytest not installed — skipping tests"
        return
    fi

    info "Running pytest ${PYTEST_ARGS}"
    if pytest ${PYTEST_ARGS} 2>&1; then
        pass "Tests ($(elapsed $start))"
    else
        fail "Tests"
    fi
}

stage_build() {
    stage_header "BUILD (wheel + sdist)"
    local start=$(date +%s)
    cd "$PROJECT_ROOT"

    # Ensure build module is available
    if ! "$PYTHON_BIN" -m build --help &>/dev/null 2>&1; then
        warn "python-build not installed. Installing..."
        "$PYTHON_BIN" -m pip install build --quiet
    fi

    # Clean prior builds
    rm -rf "${PROJECT_ROOT}/dist/"

    info "Running python -m build"
    if "$PYTHON_BIN" -m build 2>&1; then
        pass "Build ($(elapsed $start))"
    else
        fail "Build"
    fi
}

stage_release() {
    stage_build

    stage_header "RELEASE ARTIFACT"
    cd "$PROJECT_ROOT"

    # Extract package name and version
    local pkg_name=""
    local pkg_version=""

    if [ -f pyproject.toml ]; then
        pkg_name=$(grep -m1 '^name' pyproject.toml | sed 's/name *= *"\(.*\)"/\1/' || true)
        pkg_version=$(grep -m1 '^version' pyproject.toml | sed 's/version *= *"\(.*\)"/\1/' || true)
    fi

    pkg_name="${pkg_name:-unknown}"
    pkg_version="${pkg_version:-0.0.0}"

    local artifact_name="${pkg_name}-v${pkg_version}-${TIMESTAMP}"
    local artifact_dir="${RELEASE_DIR}/${artifact_name}"

    mkdir -p "$artifact_dir"

    # Copy dist artifacts
    if [ -d "${PROJECT_ROOT}/dist" ]; then
        cp "${PROJECT_ROOT}/dist/"* "$artifact_dir/" 2>/dev/null || true
        for f in "$artifact_dir"/*; do
            info "Artifact: $f"
        done
    else
        warn "No dist/ directory found"
    fi

    # Generate build metadata
    cat > "${artifact_dir}/BUILD_META.txt" <<EOF
package: ${pkg_name}
version: ${pkg_version}
timestamp: ${TIMESTAMP}
python: $("$PYTHON_BIN" --version 2>&1)
pip_freeze: $(pip freeze 2>/dev/null | head -50 || echo "unavailable")
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
    echo "║       PYTHON CI/CD — FULL PIPELINE                ║"
    echo "╚═══════════════════════════════════════════════════╝"
    echo -e "${RESET}"

    local pipeline_start=$(date +%s)
    check_prereqs

    stage_fmt
    stage_lint
    stage_typecheck
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
    stage_typecheck
    stage_test
    stage_build
}

# --- Entrypoint ---------------------------------------------------------------

case "${1:-full}" in
    fmt|format)    check_prereqs; stage_fmt ;;
    lint)          check_prereqs; stage_lint ;;
    typecheck|mypy) check_prereqs; stage_typecheck ;;
    test)          check_prereqs; stage_test ;;
    build)         check_prereqs; stage_build ;;
    release)       check_prereqs; stage_release ;;
    quick)         run_quick ;;
    gate)          run_gate ;;
    full)          run_full ;;
    *)
        echo "Usage: $0 {fmt|lint|typecheck|test|build|release|quick|gate|full}"
        echo ""
        echo "  fmt        Format check (ruff format --check)"
        echo "  lint       Lint (ruff check)"
        echo "  typecheck  Type check (mypy)"
        echo "  test       Tests (pytest)"
        echo "  build      Build wheel + sdist"
        echo "  release    Build + artifact + metadata + checksums"
        echo "  quick      fmt + lint (pre-commit)"
        echo "  gate       fmt + lint + typecheck + test + build (pre-push)"
        echo "  full       All stages including release artifact"
        exit 1
        ;;
esac
