"""Extract Rust/crates tasks from RDC TODO/ARCH files into RUST_BACKLOG.md."""
import re
import os
from pathlib import Path
from collections import defaultdict

RDC = Path("docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED")
RUST_KEYWORDS = re.compile(r'crates/|\.rs[\'"\s\)]|pyo3|cargo', re.IGNORECASE)
OUTPUT = Path("docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/RUST_BACKLOG.md")

# Collect all affected files
affected = []
for md in sorted(RDC.rglob("PHASE_*_TODO.md")):
    if RUST_KEYWORDS.search(md.read_text()):
        affected.append(md)
for md in sorted(RDC.rglob("PHASE_*_ARCH.md")):
    if RUST_KEYWORDS.search(md.read_text()):
        affected.append(md)

# Build backlog
backlog_lines = [
    "# RUST BACKLOG — Translocated from Python SDLC TODO/ARCH files\n",
    "Tasks referencing Rust crates, .rs files, pyo3, or cargo.\n",
    "Preserved for future implementation. Removed from Python-focused workflow files.\n\n",
    "---\n\n",
]

removals_by_file = defaultdict(list)

for fpath in sorted(affected):
    rel = str(fpath.relative_to(RDC))
    text = fpath.read_text()
    lines = text.split('\n')

    # Find rust-related lines
    rust_line_nums = set()
    for i, line in enumerate(lines):
        if RUST_KEYWORDS.search(line):
            rust_line_nums.add(i)

    if not rust_line_nums:
        continue

    # Find containing sections for each rust line
    sections_to_extract = set()
    section_starts = [0] + [i for i, l in enumerate(lines) if l.startswith('## ')] + [len(lines)]

    for rust_idx in rust_line_nums:
        for j in range(len(section_starts) - 1):
            if section_starts[j] <= rust_idx < section_starts[j + 1]:
                # Find enclosing task block (### header back to previous ### or ##)
                block_start = rust_idx
                block_end = rust_idx

                # Find start of the task/item block
                for k in range(rust_idx, -1, -1):
                    if lines[k].startswith('### ') or lines[k].startswith('## '):
                        block_start = k
                        break
                    if k > 0 and lines[k].strip() == '---':
                        break

                # Find end of the task/item block
                for k in range(rust_idx, len(lines)):
                    if k > rust_idx and (lines[k].startswith('### ') or lines[k].startswith('## ')):
                        block_end = k
                        break
                    if k > rust_idx and lines[k].strip() == '---':
                        block_end = k + 1
                        break
                else:
                    block_end = len(lines)

                sections_to_extract.add((block_start, block_end))
                break

    if not sections_to_extract:
        continue

    # Merge overlapping sections
    merged = sorted(sections_to_extract)
    merged_sections = [list(merged[0])]
    for start, end in merged[1:]:
        if start <= merged_sections[-1][1]:
            merged_sections[-1][1] = max(merged_sections[-1][1], end)
        else:
            merged_sections.append([start, end])

    # Add to backlog
    backlog_lines.append(f"## {rel}\n\n")
    for start, end in merged_sections:
        for line in lines[start:end]:
            backlog_lines.append(line + '\n')
        backlog_lines.append('\n')
    backlog_lines.append('---\n\n')

    # Track removal ranges (in reverse for deletion)
    removals_by_file[fpath] = [(end, start) for start, end in merged_sections]

# Write backlog
OUTPUT.write_text(''.join(backlog_lines))
print(f"Backlog written: {OUTPUT} ({len(backlog_lines)} lines)")

# Remove from source files
for fpath, removals in removals_by_file.items():
    text = fpath.read_text()
    lines = text.split('\n')
    # Remove in reverse order (end to start) to preserve indices
    for end, start in sorted(removals, reverse=True):
        # Extend: also remove trailing empty lines / --- separators
        while end < len(lines) and (lines[end].strip() == '' or lines[end].strip() == '---'):
            end += 1
        del lines[start:end]
    new_text = '\n'.join(lines)
    # Clean up triple blank lines
    while '\n\n\n\n' in new_text:
        new_text = new_text.replace('\n\n\n\n', '\n\n\n')
    fpath.write_text(new_text)
    print(f"Cleaned: {fpath.relative_to(RDC)} (removed {len(removals)} sections)")

print(f"\nDone. {len(removals_by_file)} files processed.")
