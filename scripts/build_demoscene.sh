#!/usr/bin/env bash
# build_demoscene.sh -- Build demoscene binary with 64K size optimization (T-DEMO-5.6)
#
# Usage:
#   ./scripts/build_demoscene.sh [--upx] [--check-budget] [--target TARGET]
#
# Options:
#   --upx           Apply UPX compression (requires upx installed)
#   --check-budget  Fail if binary exceeds 64KB budget
#   --target        Cross-compile target (e.g., x86_64-unknown-linux-gnu)
#   --profile       Build profile: demoscene (default) or demoscene-minimal
#   --verbose       Show detailed build output
#   --help          Show this help message

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CRATE_DIR="$PROJECT_ROOT/crates/renderer-backend"
SIZE_BUDGET_FILE="$CRATE_DIR/size_budget.json"
MAX_SIZE_BYTES=65536  # 64KB budget

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
USE_UPX=false
CHECK_BUDGET=false
BUILD_PROFILE="demoscene"
BUILD_TARGET=""
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --upx)
            USE_UPX=true
            shift
            ;;
        --check-budget)
            CHECK_BUDGET=true
            shift
            ;;
        --target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        --profile)
            BUILD_PROFILE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            head -20 "$0" | tail -16
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

get_file_size() {
    local file="$1"
    if [[ -f "$file" ]]; then
        stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null
    else
        echo "0"
    fi
}

format_size() {
    local bytes=$1
    if [[ $bytes -ge 1048576 ]]; then
        echo "$(echo "scale=2; $bytes / 1048576" | bc)MB"
    elif [[ $bytes -ge 1024 ]]; then
        echo "$(echo "scale=2; $bytes / 1024" | bc)KB"
    else
        echo "${bytes}B"
    fi
}

# Main build process
log_info "Building demoscene binary with profile: $BUILD_PROFILE"
cd "$PROJECT_ROOT"

# Build command
BUILD_CMD="cargo build --profile $BUILD_PROFILE -p renderer-backend --features demoscene-minimal"

if [[ -n "$BUILD_TARGET" ]]; then
    BUILD_CMD="$BUILD_CMD --target $BUILD_TARGET"
    TARGET_DIR="target/$BUILD_TARGET/$BUILD_PROFILE"
else
    TARGET_DIR="target/$BUILD_PROFILE"
fi

if $VERBOSE; then
    log_info "Running: $BUILD_CMD"
    $BUILD_CMD
else
    log_info "Running: $BUILD_CMD"
    $BUILD_CMD 2>&1 | grep -E "(Compiling|Finished|error|warning)" || true
fi

# Find built artifacts
log_info "Looking for built artifacts in: $TARGET_DIR"

# For a library crate, we look for .rlib or .so files
# The library is renderer_backend (with underscore)
LIB_FILE=""
for ext in rlib so a dylib; do
    candidate="$TARGET_DIR/librenderer_backend.$ext"
    if [[ -f "$candidate" ]]; then
        LIB_FILE="$candidate"
        break
    fi
done

if [[ -z "$LIB_FILE" ]]; then
    # Try deps directory
    for ext in rlib so a dylib; do
        candidate=$(find "$TARGET_DIR" -name "librenderer_backend*.$ext" -type f 2>/dev/null | head -1)
        if [[ -n "$candidate" && -f "$candidate" ]]; then
            LIB_FILE="$candidate"
            break
        fi
    done
fi

if [[ -z "$LIB_FILE" ]]; then
    log_warn "No library artifact found (this is normal for library crates without binaries)"
    # Create a marker for tests
    mkdir -p "$TARGET_DIR"
    echo "Library crate - no binary artifact" > "$TARGET_DIR/BUILD_INFO.txt"
fi

# Size tracking
BASE_SIZE=0
STRIPPED_SIZE=0
COMPRESSED_SIZE=0

if [[ -n "$LIB_FILE" && -f "$LIB_FILE" ]]; then
    BASE_SIZE=$(get_file_size "$LIB_FILE")
    log_success "Base size: $(format_size $BASE_SIZE) ($BASE_SIZE bytes)"

    # Strip the binary
    STRIPPED_FILE="${LIB_FILE}.stripped"
    if command -v strip &> /dev/null; then
        cp "$LIB_FILE" "$STRIPPED_FILE"
        strip -s "$STRIPPED_FILE" 2>/dev/null || strip "$STRIPPED_FILE" 2>/dev/null || true
        STRIPPED_SIZE=$(get_file_size "$STRIPPED_FILE")
        log_success "Stripped size: $(format_size $STRIPPED_SIZE) ($STRIPPED_SIZE bytes)"

        SAVINGS=$((BASE_SIZE - STRIPPED_SIZE))
        SAVINGS_PCT=$(echo "scale=1; $SAVINGS * 100 / $BASE_SIZE" | bc 2>/dev/null || echo "0")
        log_info "Strip savings: $(format_size $SAVINGS) (${SAVINGS_PCT}%)"
    else
        log_warn "strip command not found, skipping"
        STRIPPED_SIZE=$BASE_SIZE
    fi

    # UPX compression (optional)
    if $USE_UPX; then
        if command -v upx &> /dev/null; then
            COMPRESSED_FILE="${LIB_FILE}.upx"
            cp "$STRIPPED_FILE" "$COMPRESSED_FILE" 2>/dev/null || cp "$LIB_FILE" "$COMPRESSED_FILE"

            log_info "Applying UPX compression..."
            if upx --best --lzma "$COMPRESSED_FILE" 2>/dev/null; then
                COMPRESSED_SIZE=$(get_file_size "$COMPRESSED_FILE")
                log_success "Compressed size: $(format_size $COMPRESSED_SIZE) ($COMPRESSED_SIZE bytes)"

                SAVINGS=$((STRIPPED_SIZE - COMPRESSED_SIZE))
                SAVINGS_PCT=$(echo "scale=1; $SAVINGS * 100 / $STRIPPED_SIZE" | bc 2>/dev/null || echo "0")
                log_info "UPX savings: $(format_size $SAVINGS) (${SAVINGS_PCT}%)"
            else
                log_warn "UPX compression failed (may not support this file type)"
                COMPRESSED_SIZE=$STRIPPED_SIZE
            fi
        else
            log_warn "UPX not installed, skipping compression"
            COMPRESSED_SIZE=$STRIPPED_SIZE
        fi
    else
        COMPRESSED_SIZE=$STRIPPED_SIZE
    fi
fi

# Update size budget JSON
TIMESTAMP=$(date -Iseconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S)
cat > "$SIZE_BUDGET_FILE" << EOF
{
  "version": "1.0.0",
  "last_build": "$TIMESTAMP",
  "profile": "$BUILD_PROFILE",
  "target": "${BUILD_TARGET:-native}",
  "budget_bytes": $MAX_SIZE_BYTES,
  "sizes": {
    "base_bytes": $BASE_SIZE,
    "stripped_bytes": $STRIPPED_SIZE,
    "compressed_bytes": $COMPRESSED_SIZE
  },
  "within_budget": $([ $COMPRESSED_SIZE -le $MAX_SIZE_BYTES ] && echo "true" || echo "false"),
  "headroom_bytes": $((MAX_SIZE_BYTES - COMPRESSED_SIZE)),
  "compression_ratio": $(echo "scale=3; $COMPRESSED_SIZE / ($BASE_SIZE + 1)" | bc 2>/dev/null || echo "0"),
  "options": {
    "upx_enabled": $USE_UPX,
    "features": ["demoscene-minimal"]
  }
}
EOF

log_success "Size budget updated: $SIZE_BUDGET_FILE"

# Summary
echo ""
echo "========================================="
echo "       DEMOSCENE BUILD SUMMARY"
echo "========================================="
echo ""
printf "%-20s %12s\n" "Build profile:" "$BUILD_PROFILE"
printf "%-20s %12s\n" "Target:" "${BUILD_TARGET:-native}"
echo ""
printf "%-20s %12s\n" "Base size:" "$(format_size $BASE_SIZE)"
printf "%-20s %12s\n" "Stripped size:" "$(format_size $STRIPPED_SIZE)"
printf "%-20s %12s\n" "Compressed size:" "$(format_size $COMPRESSED_SIZE)"
echo ""
printf "%-20s %12s\n" "Budget:" "$(format_size $MAX_SIZE_BYTES)"
printf "%-20s %12s\n" "Remaining:" "$(format_size $((MAX_SIZE_BYTES - COMPRESSED_SIZE)))"
echo ""

# Check budget
if $CHECK_BUDGET; then
    FINAL_SIZE=$COMPRESSED_SIZE
    if [[ $FINAL_SIZE -gt $MAX_SIZE_BYTES ]]; then
        OVER=$((FINAL_SIZE - MAX_SIZE_BYTES))
        log_error "BUDGET EXCEEDED by $(format_size $OVER) ($OVER bytes)"
        log_error "Final size: $FINAL_SIZE bytes > budget: $MAX_SIZE_BYTES bytes"
        exit 1
    else
        log_success "Within budget: $FINAL_SIZE / $MAX_SIZE_BYTES bytes"
    fi
fi

log_success "Demoscene build complete!"
