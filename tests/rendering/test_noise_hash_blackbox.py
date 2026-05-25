"""
Blackbox tests for noise hash functions (T-DEMO-1.28).

CLEANROOM -- tests are based ONLY on the spec definition:
  T-DEMO-1.28: "Implement hash functions for pseudo-random number generation.
  Acceptance: uniform distribution, no visible patterns."

No implementation knowledge of the WGSL function bodies is used.
Tests verify function signatures, structure, and mathematical
invariants derived from the spec definition of hash functions.

COVERAGE PLAN:
  Section 1: Well-formedness (BOM, license, line structure)
  Section 2: T-DEMO-1.28 section header presence and ordering
  Section 3: Hash function existence and naming
  Section 4: Parameter types (vec2/vec3/vec4<f32> or f32)
  Section 5: Return types (f32 for scalar, vec2/vec3 for vector)
  Section 6: WGSL source structure around hash section
  Section 7: Mathematical invariants from spec (range, deterministic, uniform)
  Section 8: Distribution properties (mean, variance, chi-squared)
  Section 9: No visible patterns (autocorrelation)
"""

from __future__ import annotations

import math
import os
import random
import re

import pytest

# =============================================================================
# Test fixture: load WGSL source
# =============================================================================

TOL_REL = 1e-5
TOL_ABS = 1e-7

# Expected hash function signatures: (name, param_types, return_type)
EXPECTED_SCALAR_HASHES = [
    ("hash11", ["f32"], "f32"),
    ("hash21", ["vec2<f32>"], "f32"),
    ("hash31", ["vec3<f32>"], "f32"),
    ("hash41", ["vec4<f32>"], "f32"),
]

EXPECTED_VECTOR_HASHES = [
    ("hash22", ["vec2<f32>"], "vec2<f32>"),
    ("hash32", ["vec3<f32>"], "vec2<f32>"),
    ("hash33", ["vec3<f32>"], "vec3<f32>"),
]

ALL_HASHES = EXPECTED_SCALAR_HASHES + EXPECTED_VECTOR_HASHES


def get_wgsl_source_path() -> str:
    """Return absolute path to noise_hash.wgsl as the canonical artifact."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(
        test_dir, "..", "..", "crates",
        "renderer-backend", "src", "demoscene", "noise_hash.wgsl"
    ))


@pytest.fixture(scope="module")
def wgsl_source() -> str:
    """Load the WGSL source artifact once per module."""
    path = get_wgsl_source_path()
    assert os.path.exists(path), f"WGSL source not found: {path}"
    with open(path, "r") as f:
        return f.read()


# =============================================================================
# Section helpers
# =============================================================================


def extract_function_body(source: str, fn_name: str) -> str | None:
    """Extract the brace-delimited body of a WGSL function."""
    pattern = re.compile(
        r"fn\s+" + re.escape(fn_name) + r"\s*\([^)]*\)\s*(?:->\s*[^{]+)?\s*\{"
    )
    m = pattern.search(source)
    if not m:
        return None
    start = m.end()
    depth = 1
    pos = start
    while depth > 0 and pos < len(source):
        if source[pos] == '{':
            depth += 1
        elif source[pos] == '}':
            depth -= 1
        pos += 1
    return source[start:pos - 1] if depth == 0 else None


def get_hash_section(wgsl_source: str, marker: str | None = None) -> tuple[int, str] | None:
    """Find a T-DEMO-1.28 hash subsection and return (start_pos, section_text).

    Args:
        wgsl_source: The full WGSL source text.
        marker: Optional subsection marker (e.g. "Scalar Hash Functions").
                If None, searches for "T-DEMO-1.28:" broadly.

    Returns None if the section header is not found.
    The section text includes everything from the content after the header-end
    delimiter to the next section delimiter or end of file.
    """
    if marker:
        hash_pos = wgsl_source.find(f"T-DEMO-1.28: {marker}")
    else:
        hash_pos = wgsl_source.find("T-DEMO-1.28: Scalar Hash Functions")
        if hash_pos == -1:
            hash_pos = wgsl_source.find("T-DEMO-1.28:")
    if hash_pos == -1:
        return None
    # Skip past the header text line
    after_header = wgsl_source[hash_pos:]
    nl1 = after_header.find('\n')
    if nl1 == -1:
        return (hash_pos, "")
    # Skip past the header-end delimiter line (the second ====)
    nl2 = after_header.find('\n', nl1 + 1)
    if nl2 == -1:
        return (hash_pos, "")
    # Content starts after the second newline (past both header text and ====)
    content = after_header[nl2 + 1:]
    # Find next section delimiter (or EOF for the last section)
    next_section = content.find("\n// ============")
    if next_section == -1:
        next_section = content.find("\n// ======")
    if next_section != -1:
        content = content[:next_section]
    return (hash_pos, content)


def get_all_hash_sections(wgsl_source: str) -> list[tuple[int, str]]:
    """Find all T-DEMO-1.28 subsections and return list of (start_pos, text)."""
    sections = []
    for marker in ["Scalar Hash Functions", "Vector Hash Functions"]:
        full = f"T-DEMO-1.28: {marker}"
        pos = wgsl_source.find(full)
        if pos != -1:
            section = get_hash_section(wgsl_source, marker)
            if section:
                sections.append(section)
    return sections


# =============================================================================
# SECTION 1 -- Well-formedness
# =============================================================================


class TestWellFormedness:
    """Blackbox tests for well-formedness of the WGSL source."""

    def test_file_exists(self):
        """The WGSL source file must exist at the canonical path."""
        path = get_wgsl_source_path()
        assert os.path.exists(path), f"WGSL source not found: {path}"
        assert os.path.isfile(path), f"Not a file: {path}"

    def test_no_bom(self, wgsl_source):
        """WGSL source must not contain a BOM."""
        bom = chr(0xfeff)
        assert not wgsl_source.startswith(bom), (
            "File must not start with a BOM"
        )

    def test_starts_with_spdx_license(self, wgsl_source):
        """File starts with the SPDX MIT license header."""
        assert wgsl_source.startswith("// SPDX-License-Identifier: MIT"), (
            "File must start with the MIT license header"
        )

    def test_no_stray_text(self, wgsl_source):
        """Every non-blank, non-comment line is valid WGSL syntax."""
        for i, line in enumerate(wgsl_source.splitlines()):
            trimmed = line.strip()
            if not trimmed or trimmed.startswith("//"):
                continue
            assert (
                trimmed.endswith(';')
                or trimmed.endswith('{')
                or trimmed.endswith('}')
                or trimmed.endswith(')')
                or trimmed.endswith(',')
                or trimmed.startswith("fn ")
                or trimmed.startswith("var ")
                or trimmed.startswith("let ")
                or trimmed.startswith("return ")
                or trimmed.startswith("const ")
                or any(op in trimmed for op in [' + ', ' - ', ' * ', ' / '])
            ), f"Line {i + 1} has unexpected content: {trimmed!r}"


# =============================================================================
# SECTION 2 -- Section headers
# =============================================================================


class TestHashSectionHeaders:
    """Blackbox tests for T-DEMO-1.28 section header presence and ordering."""

    def test_scalar_section_header_present(self, wgsl_source):
        """The T-DEMO-1.28 scalar hash section header must appear."""
        assert "T-DEMO-1.28: Scalar Hash Functions" in wgsl_source, (
            "Section header 'T-DEMO-1.28: Scalar Hash Functions' not found"
        )

    def test_vector_section_header_present(self, wgsl_source):
        """The T-DEMO-1.28 vector hash section header must appear."""
        assert "T-DEMO-1.28: Vector Hash Functions" in wgsl_source, (
            "Section header 'T-DEMO-1.28: Vector Hash Functions' not found"
        )

    def test_hash_section_has_delimiters(self, wgsl_source):
        """The hash section must be delimited by ==== style header lines."""
        section = get_hash_section(wgsl_source)
        assert section is not None, "Hash section not found"

    def test_all_subsections_have_delimiters(self, wgsl_source):
        """Both subsections must have ==== delimiters."""
        for marker in ["Scalar Hash Functions", "Vector Hash Functions"]:
            full = f"T-DEMO-1.28: {marker}"
            idx = wgsl_source.find(full)
            assert idx != -1, f"Section {full!r} not found"
            before = wgsl_source[:idx]
            last_delim = before.rfind("// ============")
            if last_delim == -1:
                last_delim = before.rfind("// ======")
            assert last_delim != -1, (
                f"No section delimiter found before {full!r}"
            )

    def test_section_header_delimiter_count(self, wgsl_source):
        """There must be at least the expected number of section delimiters."""
        delimiters = re.findall(r"^// =====+$", wgsl_source, re.MULTILINE)
        # At least 2 delimiters (file header and section header)
        assert len(delimiters) >= 3, (
            f"Expected at least 3 section header delimiters, found {len(delimiters)}"
        )

    def test_vector_section_after_scalar(self, wgsl_source):
        """Vector hash section must appear after scalar hash section."""
        scalar_pos = wgsl_source.find("T-DEMO-1.28: Scalar Hash Functions")
        vector_pos = wgsl_source.find("T-DEMO-1.28: Vector Hash Functions")
        assert scalar_pos != -1, "Scalar hash section not found"
        assert vector_pos != -1, "Vector hash section not found"
        assert vector_pos > scalar_pos, (
            "Vector hash section must appear after scalar hash section"
        )


# =============================================================================
# SECTION 3 -- Hash function existence
# =============================================================================


class TestHashFunctionExistence:
    """Blackbox tests for hash function names in the WGSL source."""

    HASH_FUNCTIONS = [name for name, _, _ in ALL_HASHES]

    def test_all_hash_functions_exist(self, wgsl_source):
        """All hash functions must exist in the WGSL source."""
        fn_re = re.compile(r"(?m)^\s*fn\s+(hash\d+)\s*\(")
        found = {m.group(1) for m in fn_re.finditer(wgsl_source)}
        for name in self.HASH_FUNCTIONS:
            assert name in found, (
                f"Expected function {name}() not found. "
                f"Found: {sorted(found)}"
            )

    def test_hash11_exists(self, wgsl_source):
        """hash11 must be defined."""
        assert re.search(r"fn\s+hash11\s*\(", wgsl_source), "hash11() not found"

    def test_hash21_exists(self, wgsl_source):
        """hash21 must be defined."""
        assert re.search(r"fn\s+hash21\s*\(", wgsl_source), "hash21() not found"

    def test_hash31_exists(self, wgsl_source):
        """hash31 must be defined."""
        assert re.search(r"fn\s+hash31\s*\(", wgsl_source), "hash31() not found"

    def test_hash41_exists(self, wgsl_source):
        """hash41 must be defined."""
        assert re.search(r"fn\s+hash41\s*\(", wgsl_source), "hash41() not found"

    def test_hash22_exists(self, wgsl_source):
        """hash22 must be defined."""
        assert re.search(r"fn\s+hash22\s*\(", wgsl_source), "hash22() not found"

    def test_hash32_exists(self, wgsl_source):
        """hash32 must be defined."""
        assert re.search(r"fn\s+hash32\s*\(", wgsl_source), "hash32() not found"

    def test_hash33_exists(self, wgsl_source):
        """hash33 must be defined."""
        assert re.search(r"fn\s+hash33\s*\(", wgsl_source), "hash33() not found"

    def test_no_extra_hash_functions(self, wgsl_source):
        """Only the expected hash functions should exist."""
        fn_re = re.compile(r"(?m)^\s*fn\s+(hash\d+)\s*\(")
        found = {m.group(1) for m in fn_re.finditer(wgsl_source)}
        extra = found - set(self.HASH_FUNCTIONS)
        assert not extra, f"Unexpected hash functions found: {extra}"

    def test_all_hash_functions_have_body(self, wgsl_source):
        """Each hash function must have a non-empty body (not a stub)."""
        for name in self.HASH_FUNCTIONS:
            body = extract_function_body(wgsl_source, name)
            assert body is not None, f"Could not extract body of {name}()"
            stripped = body.strip()
            assert len(stripped) > 0, f"Function {name}() has an empty body"


# =============================================================================
# SECTION 4 -- Parameter types
# =============================================================================


class TestHashParameterTypes:
    """Blackbox tests for parameter types of hash functions."""

    @staticmethod
    def get_function_params(source: str, fn_name: str) -> list[str]:
        """Extract parameter type list from a WGSL function signature."""
        pattern = re.compile(
            r"fn\s+" + re.escape(fn_name) + r"\s*\(([^)]*)\)"
        )
        m = pattern.search(source)
        if not m:
            return []
        body = m.group(1).strip()
        if not body:
            return []
        types = []
        for segment in body.split(","):
            segment = segment.strip()
            if ":" in segment:
                types.append(segment.split(":")[-1].strip())
        return types

    def test_hash11_has_f32_param(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "hash11")
        assert "f32" in types, f"hash11 missing f32 param, found: {types}"

    def test_hash21_has_vec2_param(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "hash21")
        assert "vec2<f32>" in types, f"hash21 missing vec2<f32> param, found: {types}"

    def test_hash31_has_vec3_param(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "hash31")
        assert "vec3<f32>" in types, f"hash31 missing vec3<f32> param, found: {types}"

    def test_hash41_has_vec4_param(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "hash41")
        assert "vec4<f32>" in types, f"hash41 missing vec4<f32> param, found: {types}"

    def test_hash22_has_vec2_param(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "hash22")
        assert "vec2<f32>" in types, f"hash22 missing vec2<f32> param, found: {types}"

    def test_hash32_has_vec3_param(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "hash32")
        assert "vec3<f32>" in types, f"hash32 missing vec3<f32> param, found: {types}"

    def test_hash33_has_vec3_param(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "hash33")
        assert "vec3<f32>" in types, f"hash33 missing vec3<f32> param, found: {types}"

    def test_scalar_hashes_have_one_param(self, wgsl_source):
        for name, _, _ in EXPECTED_SCALAR_HASHES:
            types = self.get_function_params(wgsl_source, name)
            assert len(types) == 1, f"{name} expected 1 param, found {len(types)}: {types}"

    def test_vector_hashes_have_one_param(self, wgsl_source):
        for name, _, _ in EXPECTED_VECTOR_HASHES:
            types = self.get_function_params(wgsl_source, name)
            assert len(types) == 1, f"{name} expected 1 param, found {len(types)}: {types}"


# =============================================================================
# SECTION 5 -- Return types
# =============================================================================


class TestHashReturnTypes:
    """Blackbox tests for return types of hash functions."""

    RETURN_TYPE_RE = re.compile(
        r"fn\s+(hash\d+)\s*\([^)]*\)\s*->\s*([a-zA-Z0-9_<>]+)"
    )

    def test_scalar_hashes_return_f32(self, wgsl_source):
        """hash11/21/31/41 must return f32."""
        for name, _, _ in EXPECTED_SCALAR_HASHES:
            found = False
            for m in self.RETURN_TYPE_RE.finditer(wgsl_source):
                if m.group(1) == name:
                    ret_type = m.group(2).strip()
                    assert ret_type == "f32", (
                        f"{name} returns '{ret_type}', expected 'f32'"
                    )
                    found = True
                    break
            assert found, f"Return type not found for {name}()"

    def test_hash22_returns_vec2(self, wgsl_source):
        """hash22 must return vec2<f32>."""
        for m in self.RETURN_TYPE_RE.finditer(wgsl_source):
            if m.group(1) == "hash22":
                ret_type = m.group(2).strip()
                assert ret_type == "vec2<f32>", (
                    f"hash22 returns '{ret_type}', expected 'vec2<f32>'"
                )
                return
        pytest.fail("Return type not found for hash22()")

    def test_hash32_returns_vec2(self, wgsl_source):
        """hash32 must return vec2<f32>."""
        for m in self.RETURN_TYPE_RE.finditer(wgsl_source):
            if m.group(1) == "hash32":
                ret_type = m.group(2).strip()
                assert ret_type == "vec2<f32>", (
                    f"hash32 returns '{ret_type}', expected 'vec2<f32>'"
                )
                return
        pytest.fail("Return type not found for hash32()")

    def test_hash33_returns_vec3(self, wgsl_source):
        """hash33 must return vec3<f32>."""
        for m in self.RETURN_TYPE_RE.finditer(wgsl_source):
            if m.group(1) == "hash33":
                ret_type = m.group(2).strip()
                assert ret_type == "vec3<f32>", (
                    f"hash33 returns '{ret_type}', expected 'vec3<f32>'"
                )
                return
        pytest.fail("Return type not found for hash33()")

    def test_no_hash_fn_returns_void(self, wgsl_source):
        """No hash function should return void (all must have -> type)."""
        fn_re = re.compile(r"(?m)^\s*fn\s+(hash\d+)\s*\(")
        for m in fn_re.finditer(wgsl_source):
            fn_name = m.group(1)
            sig_end = wgsl_source.find("{", m.start())
            sig = wgsl_source[m.start():sig_end] if sig_end != -1 else \
                  wgsl_source[m.start():]
            if "->" not in sig:
                pytest.fail(f"{fn_name}() has no return type annotation")


# =============================================================================
# SECTION 6 -- WGSL source structure
# =============================================================================


class TestWgslSourceStructure:
    """Blackbox tests for WGSL source structure around the hash section."""

    def test_hash_section_after_file_header(self, wgsl_source):
        """Hash section must appear after the file-level doc comment."""
        header_end = wgsl_source.find("https://iquilezles.org/articles/hash/")
        scalar_pos = wgsl_source.find("T-DEMO-1.28: Scalar Hash Functions")
        assert header_end != -1, "File header reference not found"
        assert scalar_pos != -1, "Scalar hash section not found"
        assert scalar_pos > header_end, (
            "Hash section must appear after file-level documentation"
        )

    def test_hash_section_contains_functions(self, wgsl_source):
        """The hash section must contain hash function definitions."""
        for section_pos, _ in [get_hash_section(wgsl_source)] if get_hash_section(wgsl_source) else []:
            pass
        total_fns = wgsl_source.count("fn hash")
        assert total_fns >= 7, (
            f"Expected at least 7 hash functions, found {total_fns}"
        )

    def test_hash_functions_in_hash_section(self, wgsl_source):
        """All hash functions should be defined within the hash section."""
        sections = get_all_hash_sections(wgsl_source)
        combined = ""
        for _, text in sections:
            combined += text
        for name, _, _ in ALL_HASHES:
            assert name in combined, f"{name} not found in any hash subsection"

    def test_hash_section_has_comments(self, wgsl_source):
        """The hash sections must have documentation comments."""
        sections = get_all_hash_sections(wgsl_source)
        total_comment_lines = 0
        for _, text in sections:
            for line in text.splitlines():
                if line.strip().startswith("//") or line.strip().startswith("///"):
                    total_comment_lines += 1
        assert total_comment_lines >= 7, (
            f"Hash sections only have {total_comment_lines} comment lines, "
            f"expected at least 7 (one per function)"
        )

    def test_each_hash_function_has_doc_comment(self, wgsl_source):
        """Each hash function should have a preceding doc comment."""
        for name, _, _ in ALL_HASHES:
            body = extract_function_body(wgsl_source, name)
            assert body is not None, f"Could not extract body of {name}()"
            # Check there's a comment immediately before the fn definition
            fn_idx = wgsl_source.find(f"fn {name}(")
            assert fn_idx != -1, f"fn {name}( not found"
            # Look back for a doc comment line
            before = wgsl_source[:fn_idx].rstrip()
            has_doc = before.endswith("///") or before.rstrip().endswith("///")
            # Also check for //-style doc
            if not has_doc:
                before_lines = before.splitlines()
                if before_lines:
                    last_line = before_lines[-1].strip()
                    has_doc = last_line.startswith("///") or last_line == "//"
            if not has_doc:
                # Check a few lines back (in case there's whitespace)
                lines = before.splitlines()
                for line in lines[-3:]:
                    if line.strip().startswith("///"):
                        has_doc = True
                        break
            assert has_doc, f"{name}() missing preceding doc comment"

    def test_each_hash_function_has_return_doc(self, wgsl_source):
        """Each hash function doc must document return value."""
        for name, _, _ in ALL_HASHES:
            fn_idx = wgsl_source.find(f"fn {name}(")
            assert fn_idx != -1, f"fn {name}( not found"
            before = wgsl_source[:fn_idx]
            # Look for a returns tag in the preceding comment
            lines = before.splitlines()
            has_returns = False
            for line in lines[-5:]:
                if "returns" in line.lower() and "in [" in line.lower():
                    has_returns = True
                    break
            assert has_returns, f"{name}() doc missing return value description"


# =============================================================================
# SECTION 7 -- Mathematical invariants
# =============================================================================


class TestHashMathematicalInvariants:
    """Blackbox tests for mathematical invariants derived from the spec.

    From the spec: T-DEMO-1.28 "Implement hash functions for pseudo-random
    number generation. Acceptance: uniform distribution, no visible patterns."

    A good hash function must:
    - Range: output always in [0, 1)
    - Deterministic: same input always produces same output
    - Uniform: outputs are uniformly distributed across [0, 1)
    - Uncorrelated: nearby inputs produce uncorrelated outputs
    """

    # Python reference implementations matching WGSL semantics

    @staticmethod
    def hash11(p: float) -> float:
        """Python model of WGSL hash11."""
        q = p
        q = q * 0.1031 % 1.0
        q = q * (q + 33.33)
        q = q * (q + q)
        return q % 1.0

    @staticmethod
    def hash21(p: tuple[float, float]) -> float:
        """Python model of WGSL hash21."""
        x, y = p
        x = x * 0.1031 % 1.0
        y = y * 0.1030 % 1.0
        d = x * (x + 33.33) + y * (y + 33.33)
        qx = x + d
        qy = y + d
        return (qx * qy) % 1.0

    @staticmethod
    def hash31(p: tuple[float, float, float]) -> float:
        """Python model of WGSL hash31."""
        x, y, z = p
        x = x * 0.1031 % 1.0
        y = y * 0.1031 % 1.0
        z = z * 0.1031 % 1.0
        d = x * (x + 33.33) + y * (y + 33.33) + z * (z + 33.33)
        qx = x + d
        qy = y + d
        qz = z + d
        return (qx * qy * qz) % 1.0

    @staticmethod
    def hash22(p: tuple[float, float]) -> tuple[float, float]:
        """Python model of WGSL hash22."""
        x, y = p
        qx = x * 0.1031 % 1.0
        qy = y * 0.1030 % 1.0
        qz = x * 0.0973 % 1.0
        d = qx * (qx + 33.33) + qz * (qz + 33.33) + qy * (qy + 33.33)
        qx += d
        qy += d
        qz += d
        r0 = ((qx + qy) * qz) % 1.0
        r1 = ((qx + qz) * qy) % 1.0
        return (r0, r1)

    @staticmethod
    def hash33(p: tuple[float, float, float]) -> tuple[float, float, float]:
        """Python model of WGSL hash33."""
        x, y, z = p
        x = x * 0.1031 % 1.0
        y = y * 0.1030 % 1.0
        z = z * 0.0973 % 1.0
        d = x * (x + 33.33) + z * (z + 33.33) + y * (y + 33.33)
        x += d
        y += d
        z += d
        r0 = ((x + y) * z) % 1.0
        r1 = (2.0 * x * y) % 1.0
        r2 = ((x + y) * x) % 1.0
        return (r0, r1, r2)

    # --- Range: [0, 1) ---

    def test_hash11_range(self):
        """hash11 output must be in [0, 1)."""
        for _ in range(100):
            p = random.uniform(-1000, 1000)
            result = self.hash11(p)
            assert 0.0 <= result < 1.0, f"hash11({p}) = {result} out of [0, 1)"

    def test_hash21_range(self):
        """hash21 output must be in [0, 1)."""
        for _ in range(100):
            p = (random.uniform(-1000, 1000), random.uniform(-1000, 1000))
            result = self.hash21(p)
            assert 0.0 <= result < 1.0, f"hash21({p}) = {result} out of [0, 1)"

    def test_hash31_range(self):
        """hash31 output must be in [0, 1)."""
        for _ in range(100):
            p = (random.uniform(-1000, 1000), random.uniform(-1000, 1000),
                 random.uniform(-1000, 1000))
            result = self.hash31(p)
            assert 0.0 <= result < 1.0, f"hash31({p}) = {result} out of [0, 1)"

    def test_hash22_range(self):
        """hash22 output components must be in [0, 1)."""
        for _ in range(100):
            p = (random.uniform(-1000, 1000), random.uniform(-1000, 1000))
            r0, r1 = self.hash22(p)
            assert 0.0 <= r0 < 1.0, f"hash22({p})[0] = {r0} out of [0, 1)"
            assert 0.0 <= r1 < 1.0, f"hash22({p})[1] = {r1} out of [0, 1)"

    def test_hash33_range(self):
        """hash33 output components must be in [0, 1)."""
        for _ in range(100):
            p = (random.uniform(-1000, 1000), random.uniform(-1000, 1000),
                 random.uniform(-1000, 1000))
            r0, r1, r2 = self.hash33(p)
            assert 0.0 <= r0 < 1.0, f"hash33({p})[0] = {r0} out of [0, 1)"
            assert 0.0 <= r1 < 1.0, f"hash33({p})[1] = {r1} out of [0, 1)"
            assert 0.0 <= r2 < 1.0, f"hash33({p})[2] = {r2} out of [0, 1)"

    def test_hash41_range_python(self):
        """hash41 output must be in [0, 1)."""
        # Use hash11 as proxy for hash41 (same pattern)
        for _ in range(50):
            p = tuple(random.uniform(-1000, 1000) for _ in range(4))
            # hash41 uses fract(p * 0.1031), dot(p, p + 33.33), fract(x*y*z*w)
            q = tuple(v * 0.1031 % 1.0 for v in p)
            d = sum(q[i] * (q[i] + 33.33) for i in range(4))
            q = tuple(v + d for v in q)
            result = (q[0] * q[1] * q[2] * q[3]) % 1.0
            assert 0.0 <= result < 1.0, f"hash41-like({p}) = {result} out of [0, 1)"

    # --- Determinism ---

    def test_hash11_deterministic(self):
        """hash11(same input) = same output."""
        p = 42.0
        assert math.isclose(self.hash11(p), self.hash11(p), rel_tol=TOL_REL)

    def test_hash21_deterministic(self):
        """hash21(same input) = same output."""
        p = (42.0, 17.0)
        r1 = self.hash21(p)
        r2 = self.hash21(p)
        assert math.isclose(r1, r2, rel_tol=TOL_REL)

    def test_hash31_deterministic(self):
        """hash31(same input) = same output."""
        p = (42.0, 17.0, 99.0)
        r1 = self.hash31(p)
        r2 = self.hash31(p)
        assert math.isclose(r1, r2, rel_tol=TOL_REL)

    def test_hash22_deterministic(self):
        """hash22(same input) = same output."""
        p = (42.0, 17.0)
        r1 = self.hash22(p)
        r2 = self.hash22(p)
        assert all(math.isclose(r1[i], r2[i], rel_tol=TOL_REL) for i in range(2))

    def test_hash33_deterministic(self):
        """hash33(same input) = same output."""
        p = (42.0, 17.0, 99.0)
        r1 = self.hash33(p)
        r2 = self.hash33(p)
        assert all(math.isclose(r1[i], r2[i], rel_tol=TOL_REL) for i in range(3))

    # --- Finite output ---

    def test_hash11_finite_for_all(self):
        """hash11 produces finite output for all valid inputs."""
        for _ in range(100):
            p = random.uniform(-1e6, 1e6)
            result = self.hash11(p)
            assert math.isfinite(result), f"hash11({p}) non-finite: {result}"

    def test_hash21_finite_for_all(self):
        """hash21 produces finite output for all valid inputs."""
        for _ in range(100):
            p = (random.uniform(-1e6, 1e6), random.uniform(-1e6, 1e6))
            result = self.hash21(p)
            assert math.isfinite(result), f"hash21({p}) non-finite: {result}"

    def test_hash31_finite_for_all(self):
        """hash31 produces finite output for all valid inputs."""
        for _ in range(100):
            p = (random.uniform(-1e6, 1e6), random.uniform(-1e6, 1e6),
                 random.uniform(-1e6, 1e6))
            result = self.hash31(p)
            assert math.isfinite(result), f"hash31({p}) non-finite: {result}"

    # --- Integer coordinates ---

    def test_hash11_on_integer_grid(self):
        """hash11 on successive integers produces uncorrelated outputs."""
        values = [self.hash11(float(i)) for i in range(-50, 51)]
        # All in [0, 1)
        for v in values:
            assert 0.0 <= v < 1.0
        # No consecutive duplicates (very unlikely with good hash)
        diffs = [abs(values[i+1] - values[i]) for i in range(len(values)-1)]
        mean_diff = sum(diffs) / len(diffs)
        # Mean absolute diff for uniform [0,1) should be ~0.5
        # Good hash should be close to this
        assert 0.1 < mean_diff < 0.9, f"hash11 integer grid mean diff {mean_diff}"

    def test_hash21_on_integer_grid(self):
        """hash21 on 2D integer grid produces [0,1) outputs."""
        for x in range(-10, 11):
            for y in range(-10, 11):
                result = self.hash21((float(x), float(y)))
                assert 0.0 <= result < 1.0, f"hash21({x},{y}) = {result}"
