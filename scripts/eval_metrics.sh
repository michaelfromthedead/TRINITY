#!/bin/bash
# Generate metrics for a Python directory for evaluation reports
# Usage: ./scripts/eval_metrics.sh engine/simulation/cloth

DIR="${1:-.}"

if [ ! -d "$DIR" ]; then
    echo "Error: $DIR is not a directory"
    exit 1
fi

echo "=== Metrics for $DIR ==="
echo ""

# File count
FILES=$(find "$DIR" -name "*.py" -type f | wc -l)
echo "Total files: $FILES"

# Line counts
if [ "$FILES" -gt 0 ]; then
    TOTAL=$(find "$DIR" -name "*.py" -type f -exec cat {} + | wc -l)
    BLANK=$(find "$DIR" -name "*.py" -type f -exec cat {} + | grep -c '^[[:space:]]*$')
    COMMENT=$(find "$DIR" -name "*.py" -type f -exec cat {} + | grep -c '^[[:space:]]*#')
    CODE=$((TOTAL - BLANK - COMMENT))

    echo "Total lines: $TOTAL"
    echo "Blank lines: $BLANK"
    echo "Comment lines: $COMMENT"
    echo "Code lines: $CODE"
fi

echo ""

# Function/class counts
if [ "$FILES" -gt 0 ]; then
    FUNCS=$(find "$DIR" -name "*.py" -type f -exec grep -h '^\s*def ' {} + | wc -l)
    CLASSES=$(find "$DIR" -name "*.py" -type f -exec grep -h '^\s*class ' {} + | wc -l)

    echo "Functions: $FUNCS"
    echo "Classes: $CLASSES"
fi

echo ""

# Stub indicators
echo "=== Stub Indicators ==="
echo "NotImplementedError: $(find "$DIR" -name "*.py" -type f -exec grep -l 'NotImplementedError' {} + 2>/dev/null | wc -l) files"
echo "TODO comments: $(find "$DIR" -name "*.py" -type f -exec grep -l 'TODO' {} + 2>/dev/null | wc -l) files"
echo "FIXME comments: $(find "$DIR" -name "*.py" -type f -exec grep -l 'FIXME' {} + 2>/dev/null | wc -l) files"
echo "pass-only bodies: $(find "$DIR" -name "*.py" -type f -exec grep -c '^\s*pass$' {} + 2>/dev/null | awk -F: '{sum+=$2} END {print sum}')"

echo ""

# File list with line counts
echo "=== File Inventory ==="
find "$DIR" -name "*.py" -type f | while read f; do
    lines=$(wc -l < "$f")
    printf "%-60s %5d lines\n" "$f" "$lines"
done | sort
