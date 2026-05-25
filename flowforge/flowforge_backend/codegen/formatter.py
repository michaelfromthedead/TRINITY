"""Formatting preservation for code generation.

This module preserves comments, blank lines, and formatting from original
source code when re-generating code from AST (which strips comments and
normalizes whitespace).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional


def _extract_comments(source: str) -> list[tuple[int, str, str]]:
    """Extract comments with their line index and neighboring code context.

    Returns a list of (line_index, comment_text, nearby_code) tuples.
    nearby_code is the stripped content of the closest non-blank, non-comment
    line before the comment (used for matching position in generated code).
    """
    lines = source.split("\n")
    results: list[tuple[int, str, str]] = []

    for i, line in enumerate(lines):
        stripped = line.rstrip()

        # Full-line comment
        if stripped.lstrip().startswith("#"):
            nearby = _find_nearby_code(lines, i)
            results.append((i, stripped, nearby))
            continue

        # Inline comment: code part # comment part
        if "#" in stripped:
            # Avoid false positives inside strings
            code_part, _, comment_part = _split_inline_comment(stripped)
            if comment_part is not None:
                nearby = code_part.strip()
                results.append((i, f"# {comment_part.strip()}", nearby))

    return results


def _split_inline_comment(line: str) -> tuple[str, str, Optional[str]]:
    """Split a line into code and inline comment, respecting strings.

    Returns (code_part, separator, comment_text) or (line, '', None) if
    no inline comment found.
    """
    in_single = False
    in_double = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line):
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i], "#", line[i + 1:]
        i += 1
    return line, "", None


def _find_nearby_code(lines: list[str], comment_idx: int) -> str:
    """Find the nearest non-blank, non-comment code line before a comment."""
    for j in range(comment_idx - 1, -1, -1):
        stripped = lines[j].strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    # If nothing before, look after
    for j in range(comment_idx + 1, len(lines)):
        stripped = lines[j].strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def preserve_comments(original_source: str, generated_source: str) -> str:
    """Re-insert comments from original source into generated code.

    Matches comments to their position by finding the closest code line
    in the generated output that matches the nearby code context from the
    original source.

    Args:
        original_source: The original source code containing comments.
        generated_source: The generated source code (comments stripped by AST).

    Returns:
        Generated source with comments re-inserted at appropriate positions.
    """
    comments = _extract_comments(original_source)
    if not comments:
        return generated_source

    gen_lines = generated_source.split("\n")
    # Track insertions: map gen line index -> list of comment lines to insert before it
    insertions: dict[int, list[str]] = {}
    # Track inline comments: map gen line index -> inline comment to append
    inline: dict[int, str] = {}

    for _orig_idx, comment_text, nearby_code in comments:
        if not nearby_code:
            # No context to match -- prepend to top
            insertions.setdefault(0, []).append(comment_text)
            continue

        # Check if this was an inline comment (nearby_code is the code on the same line)
        is_inline = not comment_text.lstrip().startswith("#") or nearby_code != ""

        best_idx = _find_best_match(gen_lines, nearby_code)

        if best_idx is not None:
            # Determine if comment was a standalone line (before the code) or inline
            stripped_comment = comment_text.strip()
            if stripped_comment.startswith("#") and _was_standalone_comment(
                original_source, comment_text, nearby_code
            ):
                insertions.setdefault(best_idx, []).append(comment_text)
            else:
                # Inline comment
                inline.setdefault(best_idx, "")
                inline[best_idx] = f"  {stripped_comment}"
        else:
            # Could not match -- insert at top as a block comment
            insertions.setdefault(0, []).append(comment_text)

    # Build result
    result: list[str] = []
    for i, line in enumerate(gen_lines):
        # Insert standalone comments before this line
        if i in insertions:
            for c in insertions[i]:
                # Match indentation of the target line
                indent = _get_indent(line)
                result.append(indent + c.strip())
        appended = line
        if i in inline:
            appended = line.rstrip() + inline[i]
        result.append(appended)

    return "\n".join(result)


def _was_standalone_comment(
    source: str, comment_text: str, nearby_code: str
) -> bool:
    """Check if a comment was on its own line (not inline) in the original."""
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped == comment_text.strip():
            return True
    return False


def _find_best_match(
    gen_lines: list[str], nearby_code: str, threshold: float = 0.6
) -> Optional[int]:
    """Find the generated line index that best matches the nearby code context."""
    best_ratio = 0.0
    best_idx: Optional[int] = None
    for i, line in enumerate(gen_lines):
        stripped = line.strip()
        if not stripped:
            continue
        ratio = SequenceMatcher(None, nearby_code, stripped).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_idx = i
    return best_idx


def _get_indent(line: str) -> str:
    """Return the leading whitespace of a line."""
    return line[: len(line) - len(line.lstrip())]


def preserve_blank_lines(original_source: str, generated_source: str) -> str:
    """Preserve blank line structure from original source in generated code.

    Detects blank-line separated sections in the original (e.g., between
    class definitions) and ensures the generated code has similar spacing.

    Args:
        original_source: The original source with intentional blank lines.
        generated_source: The generated source code.

    Returns:
        Generated source with blank line structure restored.
    """
    orig_lines = original_source.split("\n")
    gen_lines = generated_source.split("\n")

    # Build a set of "section boundary" patterns from original:
    # A section boundary is a blank line that sits between two non-blank lines.
    # We record the stripped content of the line AFTER each boundary.
    boundaries: set[str] = set()
    for i, line in enumerate(orig_lines):
        if line.strip() == "" and i > 0 and i + 1 < len(orig_lines):
            # Look for the next non-blank line
            for j in range(i + 1, len(orig_lines)):
                next_stripped = orig_lines[j].strip()
                if next_stripped:
                    boundaries.add(next_stripped)
                    break

    if not boundaries:
        return generated_source

    # Insert blank lines in generated code before lines that match boundaries,
    # but only if there isn't already a blank line there.
    result: list[str] = []
    for i, line in enumerate(gen_lines):
        stripped = line.strip()
        if stripped in boundaries and i > 0:
            # Ensure at least one blank line before this line
            if result and result[-1].strip() != "":
                result.append("")
        result.append(line)

    return "\n".join(result)


def format_code(source: str) -> str:
    """Run basic formatting on Python source code.

    Tries to use ``black`` for formatting. If black is not available,
    applies simple PEP 8 formatting rules:
    - Consistent 4-space indentation (tabs converted to spaces)
    - Trailing whitespace removed
    - Trailing newline ensured

    Args:
        source: The Python source code to format.

    Returns:
        Formatted source code.
    """
    try:
        import black
        from black import Mode, format_str

        mode = Mode(line_length=88, string_normalization=True, is_pyi=False)
        return format_str(source, mode=mode)
    except ImportError:
        pass
    except Exception:
        pass

    # Simple fallback formatter
    return _simple_format(source)


def _simple_format(source: str) -> str:
    """Simple PEP 8 formatter when black is not available.

    Handles:
    - Tab to 4-space conversion
    - Trailing whitespace removal
    - Ensures trailing newline
    - Ensures two blank lines before top-level class/function definitions
    """
    lines = source.split("\n")
    result: list[str] = []

    for i, line in enumerate(lines):
        # Convert tabs to 4 spaces
        line = line.replace("\t", "    ")
        # Remove trailing whitespace
        line = line.rstrip()
        result.append(line)

    # Ensure two blank lines before top-level class/function defs
    final: list[str] = []
    for i, line in enumerate(result):
        stripped = line.lstrip()
        is_toplevel_def = (
            not line.startswith(" ")
            and (stripped.startswith("class ") or stripped.startswith("def "))
        )
        is_decorator = not line.startswith(" ") and stripped.startswith("@")

        if (is_toplevel_def or is_decorator) and final:
            # Check if previous decorator (don't add blanks between decorator and def)
            prev_stripped = final[-1].lstrip() if final else ""
            if not prev_stripped.startswith("@"):
                # Remove existing trailing blanks
                while final and final[-1] == "":
                    final.pop()
                if final:
                    final.append("")
                    final.append("")

        final.append(line)

    text = "\n".join(final)
    # Ensure trailing newline
    if not text.endswith("\n"):
        text += "\n"
    return text
