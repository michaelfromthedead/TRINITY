#!/usr/bin/env bash
# ============================================================================
# workflows/install.sh — unified installer
# ============================================================================
# Installs workflow-system infrastructure into the current git repository:
#   Git hooks (pre-commit + pre-push) for language-level CI enforcement.
#
# Usage:
#   bash workflows/install.sh                 # install hooks
#   bash workflows/install.sh hooks           # install hooks (explicit)
#   bash workflows/install.sh reinit          # DESTRUCTIVE: delete .git/, git init, install hooks
#   bash workflows/install.sh reinit --yes    # reinit without confirmation prompt
#   bash workflows/install.sh --help          # help
#
# Note: This installer does NOT modify your project's CLAUDE.md. Workflows are
# self-bootstrapping — reference them directly by trigger phrase (e.g.,
# "SDLC_WORKFLOW", "RDC_WORKFLOW", "RECON_WORKFLOW", "ORGANIZE_WORKFLOW").
# Each workflow's JSON contains its own engagement protocol in `trigger.on_engage`.
#
# Exit codes:
#   0 = success
#   1 = failure (missing files, invalid args, etc.)
# ============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Colors
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    CYAN='\033[0;36m'
    RESET='\033[0m'
else
    GREEN='' YELLOW='' RED='' CYAN='' RESET=''
fi

ok()   { echo -e "${GREEN}✓${RESET} $1"; }
warn() { echo -e "${YELLOW}⚠${RESET} $1"; }
fail() { echo -e "${RED}✗${RESET} $1"; exit 1; }
info() { echo -e "${CYAN}→${RESET} $1"; }

# --- Arg parsing -------------------------------------------------------------
MODE="${1:-hooks}"
SUBMODE="${2:-}"

show_help() {
    grep -E '^#' "$0" | grep -v '#!/' | sed 's/^# \?//'
    exit 0
}

[ "$MODE" = "--help" ] || [ "$MODE" = "-h" ] && show_help

# --- Pre-flight checks -------------------------------------------------------
# reinit creates .git/ itself, so skip the git-repo check for that mode
if [ "$MODE" != "reinit" ] && [ ! -d "${REPO_ROOT}/.git" ]; then
    fail "Not a git repository: ${REPO_ROOT}"
fi

# --- install_hooks -----------------------------------------------------------
install_hooks() {
    info "Installing git hooks"
    bash "${SCRIPT_DIR}/install-hooks.sh" "$REPO_ROOT"
}

# --- reinit_git --------------------------------------------------------------
# DESTRUCTIVE: deletes ${REPO_ROOT}/.git, runs `git init`, then installs hooks.
# Requires explicit "yes" confirmation unless --yes / -y passed.
reinit_git() {
    local skip_confirm="${1:-}"

    info "Re-initializing git repository at: ${REPO_ROOT}"

    if [ -d "${REPO_ROOT}/.git" ]; then
        warn "Existing .git/ will be DELETED at: ${REPO_ROOT}/.git"
        warn "All commit history, branches, stashes, and refs will be lost."
    else
        info "No existing .git/ found — performing fresh git init only."
    fi

    if [ "$skip_confirm" != "--yes" ] && [ "$skip_confirm" != "-y" ]; then
        echo ""
        read -r -p "Type 'yes' to confirm: " confirm
        if [ "$confirm" != "yes" ]; then
            fail "Reinit aborted."
        fi
    fi

    if [ -d "${REPO_ROOT}/.git" ]; then
        rm -rf "${REPO_ROOT}/.git"
        ok "Deleted existing .git/"
    fi

    (cd "$REPO_ROOT" && git init)
    ok "Re-initialized git repository at ${REPO_ROOT}/.git"

    echo ""
    install_hooks
}

# --- Dispatch ----------------------------------------------------------------
case "$MODE" in
    hooks|all|"")
        install_hooks
        ;;
    reinit)
        reinit_git "$SUBMODE"
        ;;
    --help|-h)
        show_help
        ;;
    *)
        fail "Unknown mode: $MODE (use: hooks | reinit)"
        ;;
esac

echo ""
ok "Done."
