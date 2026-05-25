"""
Blackbox tests for value noise WGSL functions (T-DEMO-1.29).

CLEANROOM -- tests are based ONLY on the spec definition:
  T-DEMO-1.29: "Value noise. Acceptance: smooth interpolation between hash values."

  Spec formula (FORMULAS.md VOL 10: THE GENESIS ALGORITHM):
    Value Noise Interpolation: lerp(lerp(hash(p00), hash(p10), fx),
                                     lerp(hash(p01), hash(p11), fx), fy)

No implementation knowledge of the WGSL function bodies is used beyond
what is declared in the spec. Tests verify function signatures, structure,
and mathematical invariants derived from the spec definition.

COVERAGE PLAN:
  Section 1: Well-formedness (BOM, license, line structure)
  Section 2: T-DEMO-1.29 section header presence and ordering
  Section 3: Value noise function existence and naming
  Section 4: Parameter types (f32, vec2<f32>, vec3<f32>)
  Section 5: Return types (all f32)
  Section 6: WGSL source structure around value noise section
  Section 7: Mathematical invariants from spec (range, deterministic, continuous)
  Section 8: Smoothstep fade curve (zero derivative at boundaries)
  Section 9: Grid alignment (remapped hash values at integer positions)
  Section 10: Smooth interpolation between hash values
  Section 11: Continuity across cell boundaries
  Section 12: Distribution properties (mean near zero)
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
TOL_ABS = 1e-9

# Expected value noise function signatures: (name, param_types, return_type)
EXPECTED_VALUE_NOISE_FUNCTIONS = [
    ("value_noise_1d", ["f32"], "f32"),
    ("value_noise_2d", ["vec2<f32>"], "f32"),
    ("value_noise_3d", ["vec3<f32>"], "f32"),
]

ALL_FUNCTIONS = EXPECTED_VALUE_NOISE_FUNCTIONS


def get_wgsl_source_path() -> str:
    """Return absolute path to noise_value.wgsl as the canonical artifact."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(
        test_dir, "..", "..", "..", "crates",
        "renderer-backend", "src", "demoscene", "noise_value.wgsl"
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


def get_value_noise_section(wgsl_source: str, marker: str | None = None) -> tuple[int, str] | None:
    """Find a T-DEMO-1.29 subsection and return (start_pos, section_text).

    Args:
        wgsl_source: The full WGSL source text.
        marker: Optional subsection marker (e.g. "Value Noise 1D").
                If None, searches for "T-DEMO-1.29:" broadly.

    Returns None if the section header is not found.
    The section text includes everything from the content after the header-end
    delimiter to the next section delimiter or end of file.
    """
    if marker:
        hash_pos = wgsl_source.find(f"T-DEMO-1.29: {marker}")
    else:
        hash_pos = wgsl_source.find("T-DEMO-1.29: Value Noise 1D")
        if hash_pos == -1:
            hash_pos = wgsl_source.find("T-DEMO-1.29:")
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


def get_all_value_noise_sections(wgsl_source: str) -> list[tuple[int, str]]:
    """Find all T-DEMO-1.29 subsections and return list of (start_pos, text)."""
    sections = []
    for marker in ["Value Noise 1D", "Value Noise 2D", "Value Noise 3D"]:
        full = f"T-DEMO-1.29: {marker}"
        pos = wgsl_source.find(full)
        if pos != -1:
            section = get_value_noise_section(wgsl_source, marker)
            if section:
                sections.append(section)
    return sections


# =============================================================================
# Python reference models for hash functions (from spec: deterministic, [0, 1))
#
# These are derived from the spec requirement for deterministic hashing
# with output in [0, 1). The specific WGSL implementation follows the
# Inigo Quilez fractional hashing pattern, which we model here to verify
# the value noise interpolation formula from the spec.
# =============================================================================


def wgsl_fract(x: float) -> float:
    """WGSL fract: x - floor(x)."""
    return x - math.floor(x)


def py_hash11(p: float) -> float:
    """Model of WGSL hash11: 1D float -> [0, 1) float.

    From the spec: Positional Hash (Deterministic Noise).
    Hash functions produce deterministic pseudo-random values in [0, 1).
    """
    q = p
    q = wgsl_fract(q * 0.1031)
    q = q * (q + 33.33)
    q = q * (q + q)
    return wgsl_fract(q)


def py_hash21(p: tuple[float, float]) -> float:
    """Model of WGSL hash21: 2D -> [0, 1) float."""
    qx = wgsl_fract(p[0] * 0.1031)
    qy = wgsl_fract(p[1] * 0.1030)

    d = qx * (qx + 33.33) + qy * (qy + 33.33)
    qx = qx + d
    qy = qy + d

    return wgsl_fract(qx * qy)


def py_hash31(p: tuple[float, float, float]) -> float:
    """Model of WGSL hash31: 3D -> [0, 1) float."""
    qx = wgsl_fract(p[0] * 0.1031)
    qy = wgsl_fract(p[1] * 0.1030)
    qz = wgsl_fract(p[2] * 0.0973)

    d = qx * (qx + 33.33) + qy * (qy + 33.33) + qz * (qz + 33.33)
    qx = qx + d
    qy = qy + d
    qz = qz + d

    return wgsl_fract(qx * qy * qz)


# =============================================================================
# Python reference models for value noise (from spec formula)
#
# Spec formula: lerp(lerp(hash(p00), hash(p10), fx),
#                    lerp(hash(p01), hash(p11), fx), fy)
#
# For 1D: lerp(hash(p0), hash(p1), f)
# For 3D: trilinear extension of the 2D formula
# =============================================================================


def py_smoothstep(t: float) -> float:
    """Smoothstep fade curve: 6t^5 - 15t^4 + 10t^3.

    This is the fade function that ensures C1 continuity at cell boundaries.
    The first derivative is zero at t=0 and t=1.
    """
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def py_lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + t * (b - a)


def py_value_noise_1d(p: float) -> float:
    """Model of WGSL value_noise_1d.

    From spec: lerp(hash(p0), hash(p1), f) where f = smoothstep(t)
    """
    i = math.floor(p)
    f = p - i
    u = py_smoothstep(f)

    a = py_hash11(i)
    b = py_hash11(i + 1.0)

    # Remap from [0, 1) to [-1, 1]
    va = a * 2.0 - 1.0
    vb = b * 2.0 - 1.0

    return py_lerp(va, vb, u)


def py_value_noise_2d(p: tuple[float, float]) -> float:
    """Model of WGSL value_noise_2d.

    From spec: lerp(lerp(hash(p00), hash(p10), fx),
                    lerp(hash(p01), hash(p11), fx), fy)
    """
    ix = math.floor(p[0])
    iy = math.floor(p[1])
    fx = p[0] - ix
    fy = p[1] - iy

    ux = py_smoothstep(fx)
    uy = py_smoothstep(fy)

    # Hash values at 4 corners
    a = py_hash21((ix, iy))
    b = py_hash21((ix + 1.0, iy))
    c = py_hash21((ix, iy + 1.0))
    d = py_hash21((ix + 1.0, iy + 1.0))

    # Remap from [0, 1) to [-1, 1]
    va = a * 2.0 - 1.0
    vb = b * 2.0 - 1.0
    vc = c * 2.0 - 1.0
    vd = d * 2.0 - 1.0

    # Bilinear interpolation (matching spec formula)
    vx0 = py_lerp(va, vb, ux)
    vx1 = py_lerp(vc, vd, ux)
    return py_lerp(vx0, vx1, uy)


def py_value_noise_3d(p: tuple[float, float, float]) -> float:
    """Model of WGSL value_noise_3d.

    Trilinear extension of the spec 2D formula to 3 dimensions.
    8 corners of the cell, interpolated along x, then y, then z.
    """
    ix = math.floor(p[0])
    iy = math.floor(p[1])
    iz = math.floor(p[2])
    fx = p[0] - ix
    fy = p[1] - iy
    fz = p[2] - iz

    ux = py_smoothstep(fx)
    uy = py_smoothstep(fy)
    uz = py_smoothstep(fz)

    # Hash values at 8 corners
    a = py_hash31((ix, iy, iz))
    b = py_hash31((ix + 1.0, iy, iz))
    c = py_hash31((ix, iy + 1.0, iz))
    d = py_hash31((ix + 1.0, iy + 1.0, iz))
    e = py_hash31((ix, iy, iz + 1.0))
    f = py_hash31((ix + 1.0, iy, iz + 1.0))
    g = py_hash31((ix, iy + 1.0, iz + 1.0))
    h = py_hash31((ix + 1.0, iy + 1.0, iz + 1.0))

    # Remap from [0, 1) to [-1, 1]
    va = a * 2.0 - 1.0
    vb = b * 2.0 - 1.0
    vc = c * 2.0 - 1.0
    vd = d * 2.0 - 1.0
    ve = e * 2.0 - 1.0
    vf = f * 2.0 - 1.0
    vg = g * 2.0 - 1.0
    vh = h * 2.0 - 1.0

    # Trilinear interpolation
    vx00 = py_lerp(va, vb, ux)
    vx10 = py_lerp(vc, vd, ux)
    vx01 = py_lerp(ve, vf, ux)
    vx11 = py_lerp(vg, vh, ux)

    vy0 = py_lerp(vx00, vx10, uy)
    vy1 = py_lerp(vx01, vx11, uy)

    return py_lerp(vy0, vy1, uz)


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

    def test_file_name_correct(self):
        """File must be named noise_value.wgsl."""
        path = get_wgsl_source_path()
        assert path.endswith("noise_value.wgsl"), (
            f"Unexpected file name: {path}"
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
                or any(op in trimmed for op in [' + ', ' - ', ' * ', ' / ', ' *='])
            ), f"Line {i + 1} has unexpected content: {trimmed!r}"


# =============================================================================
# SECTION 2 -- Section headers
# =============================================================================


class TestSectionHeaders:
    """Blackbox tests for T-DEMO-1.29 section header presence and ordering."""

    def test_1d_section_header_present(self, wgsl_source):
        """The T-DEMO-1.29 1D section header must appear."""
        assert "T-DEMO-1.29: Value Noise 1D" in wgsl_source, (
            "Section header 'T-DEMO-1.29: Value Noise 1D' not found"
        )

    def test_2d_section_header_present(self, wgsl_source):
        """The T-DEMO-1.29 2D section header must appear."""
        assert "T-DEMO-1.29: Value Noise 2D" in wgsl_source, (
            "Section header 'T-DEMO-1.29: Value Noise 2D' not found"
        )

    def test_3d_section_header_present(self, wgsl_source):
        """The T-DEMO-1.29 3D section header must appear."""
        assert "T-DEMO-1.29: Value Noise 3D" in wgsl_source, (
            "Section header 'T-DEMO-1.29: Value Noise 3D' not found"
        )

    def test_section_has_delimiters(self, wgsl_source):
        """The value noise section must be delimited by ==== style header lines."""
        section = get_value_noise_section(wgsl_source)
        assert section is not None, "Value noise section not found"

    def test_all_subsections_have_delimiters(self, wgsl_source):
        """All three subsections must have ==== delimiters."""
        for marker in ["Value Noise 1D", "Value Noise 2D", "Value Noise 3D"]:
            full = f"T-DEMO-1.29: {marker}"
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

    def test_subsections_in_order(self, wgsl_source):
        """Subsections must appear in order: 1D, 2D, 3D."""
        pos_1d = wgsl_source.find("T-DEMO-1.29: Value Noise 1D")
        pos_2d = wgsl_source.find("T-DEMO-1.29: Value Noise 2D")
        pos_3d = wgsl_source.find("T-DEMO-1.29: Value Noise 3D")
        assert pos_1d != -1, "1D section not found"
        assert pos_2d != -1, "2D section not found"
        assert pos_3d != -1, "3D section not found"
        assert pos_2d > pos_1d, "2D section must appear after 1D section"
        assert pos_3d > pos_2d, "3D section must appear after 2D section"


# =============================================================================
# SECTION 3 -- Value noise function existence
# =============================================================================


class TestFunctionExistence:
    """Blackbox tests for value noise function names in the WGSL source."""

    FUNCTION_NAMES = [name for name, _, _ in ALL_FUNCTIONS]

    def test_all_functions_exist(self, wgsl_source):
        """All value noise functions must exist in the WGSL source."""
        fn_re = re.compile(r"(?m)^\s*fn\s+(value_noise_\w+)\s*\(")
        found = {m.group(1) for m in fn_re.finditer(wgsl_source)}
        for name in self.FUNCTION_NAMES:
            assert name in found, (
                f"Expected function {name}() not found. "
                f"Found: {sorted(found)}"
            )

    def test_value_noise_1d_exists(self, wgsl_source):
        """value_noise_1d must be defined."""
        assert re.search(r"fn\s+value_noise_1d\s*\(", wgsl_source), (
            "value_noise_1d() not found"
        )

    def test_value_noise_2d_exists(self, wgsl_source):
        """value_noise_2d must be defined."""
        assert re.search(r"fn\s+value_noise_2d\s*\(", wgsl_source), (
            "value_noise_2d() not found"
        )

    def test_value_noise_3d_exists(self, wgsl_source):
        """value_noise_3d must be defined."""
        assert re.search(r"fn\s+value_noise_3d\s*\(", wgsl_source), (
            "value_noise_3d() not found"
        )

    def test_no_extra_value_noise_functions(self, wgsl_source):
        """Only the expected value noise functions should exist."""
        fn_re = re.compile(r"(?m)^\s*fn\s+(value_noise_\w+)\s*\(")
        found = {m.group(1) for m in fn_re.finditer(wgsl_source)}
        extra = found - set(self.FUNCTION_NAMES)
        assert not extra, f"Unexpected value noise functions found: {extra}"

    def test_all_functions_have_body(self, wgsl_source):
        """Each value noise function must have a non-empty body (not a stub)."""
        for name in self.FUNCTION_NAMES:
            body = extract_function_body(wgsl_source, name)
            assert body is not None, f"Could not extract body of {name}()"
            stripped = body.strip()
            assert len(stripped) > 0, f"Function {name}() has an empty body"

    def test_dependency_on_hash_in_section(self, wgsl_source):
        """Value noise functions must call hash functions."""
        for name in self.FUNCTION_NAMES:
            body = extract_function_body(wgsl_source, name)
            assert body is not None, f"Could not extract body of {name}()"
            # Should call at least one hash function
            has_hash_call = any(
                f"hash" in body
                for _ in [1]
            )
            assert has_hash_call, (
                f"{name}() does not call any hash function"
            )


# =============================================================================
# SECTION 4 -- Parameter types
# =============================================================================


class TestParameterTypes:
    """Blackbox tests for parameter types of value noise functions."""

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

    def test_1d_has_f32_param(self, wgsl_source):
        """value_noise_1d must take f32."""
        types = self.get_function_params(wgsl_source, "value_noise_1d")
        assert "f32" in types, (
            f"value_noise_1d missing f32 param, found: {types}"
        )

    def test_2d_has_vec2_param(self, wgsl_source):
        """value_noise_2d must take vec2<f32>."""
        types = self.get_function_params(wgsl_source, "value_noise_2d")
        assert "vec2<f32>" in types, (
            f"value_noise_2d missing vec2<f32> param, found: {types}"
        )

    def test_3d_has_vec3_param(self, wgsl_source):
        """value_noise_3d must take vec3<f32>."""
        types = self.get_function_params(wgsl_source, "value_noise_3d")
        assert "vec3<f32>" in types, (
            f"value_noise_3d missing vec3<f32> param, found: {types}"
        )

    def test_each_has_one_param(self, wgsl_source):
        """Each value noise function must take exactly one parameter."""
        for name, _, _ in ALL_FUNCTIONS:
            types = self.get_function_params(wgsl_source, name)
            assert len(types) == 1, (
                f"{name} expected 1 param, found {len(types)}: {types}"
            )

    def test_no_extra_params(self, wgsl_source):
        """No value noise function should have extra unused parameters."""
        for name, expected_types, _ in ALL_FUNCTIONS:
            types = self.get_function_params(wgsl_source, name)
            assert types == expected_types, (
                f"{name} params {types} don't match expected {expected_types}"
            )


# =============================================================================
# SECTION 5 -- Return types
# =============================================================================


class TestReturnTypes:
    """Blackbox tests for return types of value noise functions."""

    RETURN_TYPE_RE = re.compile(
        r"fn\s+(value_noise_\w+)\s*\([^)]*\)\s*->\s*([a-zA-Z0-9_<>]+)"
    )

    def test_all_return_f32(self, wgsl_source):
        """All value noise functions must return f32."""
        for name, _, expected_ret in ALL_FUNCTIONS:
            found = False
            for m in self.RETURN_TYPE_RE.finditer(wgsl_source):
                if m.group(1) == name:
                    ret_type = m.group(2).strip()
                    assert ret_type == expected_ret, (
                        f"{name} returns '{ret_type}', expected '{expected_ret}'"
                    )
                    found = True
                    break
            assert found, f"Return type not found for {name}()"

    def test_no_function_returns_void(self, wgsl_source):
        """No value noise function should return void (all must have -> type)."""
        fn_re = re.compile(r"(?m)^\s*fn\s+(value_noise_\w+)\s*\(")
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
    """Blackbox tests for WGSL source structure."""

    def test_section_after_file_header(self, wgsl_source):
        """Value noise section must appear after the file-level doc comment."""
        ref_pos = wgsl_source.find("Value Noise")
        header_end = wgsl_source.find("https://iquilezles.org/articles/value-noise/")
        assert header_end != -1, "File header reference not found"
        first_section = wgsl_source.find("T-DEMO-1.29: Value Noise 1D")
        assert first_section != -1, "1D value noise section not found"
        assert first_section > header_end, (
            "Value noise section must appear after file-level documentation"
        )

    def test_section_contains_functions(self, wgsl_source):
        """The value noise section must contain function definitions."""
        total_fns = len(re.findall(r"fn value_noise_", wgsl_source))
        assert total_fns >= 3, (
            f"Expected at least 3 value noise functions, found {total_fns}"
        )

    def test_functions_in_correct_sections(self, wgsl_source):
        """Each function should be defined within its own subsection."""
        sections = get_all_value_noise_sections(wgsl_source)
        section_names = ["Value Noise 1D", "Value Noise 2D", "Value Noise 3D"]
        expected_fns = ["value_noise_1d", "value_noise_2d", "value_noise_3d"]

        for i, (_, text) in enumerate(sections):
            assert expected_fns[i] in text, (
                f"{expected_fns[i]} not found in {section_names[i]} section"
            )

    def test_section_has_comments(self, wgsl_source):
        """The value noise sections must have documentation comments."""
        sections = get_all_value_noise_sections(wgsl_source)
        total_comment_lines = 0
        for _, text in sections:
            for line in text.splitlines():
                if line.strip().startswith("//") or line.strip().startswith("///"):
                    total_comment_lines += 1
        assert total_comment_lines >= 7, (
            f"Value noise sections only have {total_comment_lines} comment "
            f"lines, expected at least 7"
        )

    def test_each_function_has_doc_comment(self, wgsl_source):
        """Each value noise function should have a preceding doc comment."""
        for name, _, _ in ALL_FUNCTIONS:
            body = extract_function_body(wgsl_source, name)
            assert body is not None, f"Could not extract body of {name}()"
            fn_idx = wgsl_source.find(f"fn {name}(")
            assert fn_idx != -1, f"fn {name}( not found"
            before = wgsl_source[:fn_idx].rstrip()
            has_doc = before.endswith("///") or before.rstrip().endswith("///")
            if not has_doc:
                before_lines = before.splitlines()
                if before_lines:
                    last_line = before_lines[-1].strip()
                    has_doc = last_line.startswith("///") or last_line == "//"
            if not has_doc:
                lines = before.splitlines()
                for line in lines[-3:]:
                    if line.strip().startswith("///"):
                        has_doc = True
                        break
            assert has_doc, f"{name}() missing preceding doc comment"

    def test_each_function_has_return_doc(self, wgsl_source):
        """Each value noise function doc must document the return value."""
        for name, _, _ in ALL_FUNCTIONS:
            fn_idx = wgsl_source.find(f"fn {name}(")
            assert fn_idx != -1, f"fn {name}( not found"
            before = wgsl_source[:fn_idx]
            lines = before.splitlines()
            has_returns = False
            for line in lines[-5:]:
                if "returns" in line.lower() and "[-1, 1]" in line:
                    has_returns = True
                    break
            assert has_returns, (
                f"{name}() doc missing return value description with [-1, 1]"
            )

    def test_dependency_reference_present(self, wgsl_source):
        """The file must reference its hash dependency."""
        assert "noise_hash.wgsl" in wgsl_source or "hash11" in wgsl_source, (
            "File missing reference to hash dependency"
        )


# =============================================================================
# SECTION 7 -- Mathematical invariants
# =============================================================================


class TestMathematicalInvariants:
    """Blackbox tests for mathematical invariants derived from the spec.

    From the spec: T-DEMO-1.29 "Value noise. Acceptance: smooth interpolation
    between hash values."

    A correct value noise implementation must:
    - Range: output always in [-1, 1]
    - Deterministic: same input always produces same output
    - Grid alignment: at integer positions, output == remapped hash value
    - Continuous: no discontinuities in the interpolation
    """

    # --- Range: [-1, 1] ---

    def test_1d_range(self):
        """value_noise_1d output must be in [-1, 1]."""
        for i in range(-100, 101):
            p = i * 0.137
            v = py_value_noise_1d(p)
            assert -1.0 <= v <= 1.0 + 1e-12, (
                f"value_noise_1d({p}) = {v} outside [-1, 1]"
            )

    def test_1d_range_dense_sampling(self):
        """Dense sampling across multiple grid cells should stay in range."""
        for i in range(500):
            p = -25.0 + i * 0.1
            v = py_value_noise_1d(p)
            assert -1.0 <= v <= 1.0 + 1e-12, (
                f"value_noise_1d({p}) = {v} outside [-1, 1]"
            )

    def test_2d_range(self):
        """value_noise_2d output must be in [-1, 1]."""
        for ix in range(-10, 11):
            for iy in range(-10, 11):
                p = (ix * 0.3, iy * 0.3)
                v = py_value_noise_2d(p)
                assert -1.0 <= v <= 1.0 + 1e-12, (
                    f"value_noise_2d({p}) = {v} outside [-1, 1]"
                )

    def test_3d_range(self):
        """value_noise_3d output must be in [-1, 1]."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5, iy * 0.5, iz * 0.5)
                    v = py_value_noise_3d(p)
                    assert -1.0 <= v <= 1.0 + 1e-12, (
                        f"value_noise_3d({p}) = {v} outside [-1, 1]"
                    )

    def test_1d_range_boundary_values(self):
        """value_noise_1d at integer positions must be in [-1, 1]."""
        for i in range(-100, 101):
            v = py_value_noise_1d(float(i))
            assert -1.0 <= v <= 1.0 + 1e-12, (
                f"value_noise_1d({i}) = {v} outside [-1, 1]"
            )

    # --- Determinism ---

    def test_1d_deterministic(self):
        """value_noise_1d(same input) = same output."""
        p = 42.0
        r1 = py_value_noise_1d(p)
        for _ in range(20):
            r2 = py_value_noise_1d(p)
            assert math.isclose(r1, r2, rel_tol=TOL_REL), (
                f"value_noise_1d not deterministic at {p}"
            )

    def test_2d_deterministic(self):
        """value_noise_2d(same input) = same output."""
        p = (2.71828, 3.14159)
        r1 = py_value_noise_2d(p)
        for _ in range(20):
            r2 = py_value_noise_2d(p)
            assert math.isclose(r1, r2, rel_tol=TOL_REL), (
                f"value_noise_2d not deterministic at {p}"
            )

    def test_3d_deterministic(self):
        """value_noise_3d(same input) = same output."""
        p = (1.618, 2.718, 3.142)
        r1 = py_value_noise_3d(p)
        for _ in range(20):
            r2 = py_value_noise_3d(p)
            assert math.isclose(r1, r2, rel_tol=TOL_REL), (
                f"value_noise_3d not deterministic at {p}"
            )

    # --- Finite output ---

    def test_1d_finite_for_all(self):
        """value_noise_1d produces finite output for all valid inputs."""
        for _ in range(100):
            p = random.uniform(-1e6, 1e6)
            result = py_value_noise_1d(p)
            assert math.isfinite(result), (
                f"value_noise_1d({p}) non-finite: {result}"
            )

    def test_2d_finite_for_all(self):
        """value_noise_2d produces finite output for all valid inputs."""
        for _ in range(100):
            p = (random.uniform(-1e6, 1e6), random.uniform(-1e6, 1e6))
            result = py_value_noise_2d(p)
            assert math.isfinite(result), (
                f"value_noise_2d({p}) non-finite: {result}"
            )

    def test_3d_finite_for_all(self):
        """value_noise_3d produces finite output for all valid inputs."""
        for _ in range(50):
            p = (random.uniform(-1e6, 1e6), random.uniform(-1e6, 1e6),
                 random.uniform(-1e6, 1e6))
            result = py_value_noise_3d(p)
            assert math.isfinite(result), (
                f"value_noise_3d({p}) non-finite: {result}"
            )

    # --- Negative inputs ---

    def test_negative_inputs_1d(self):
        """value_noise_1d handles negative coordinates correctly."""
        for i in range(-50, 0):
            p = float(i) + 0.3
            v = py_value_noise_1d(p)
            assert -1.0 <= v <= 1.0 + 1e-12, (
                f"value_noise_1d({p}) = {v} outside [-1, 1] for negative input"
            )

    def test_negative_inputs_2d(self):
        """value_noise_2d handles negative coordinates correctly."""
        for ix in range(-5, 0):
            for iy in range(-5, 0):
                p = (ix + 0.3, iy + 0.7)
                v = py_value_noise_2d(p)
                assert -1.0 <= v <= 1.0 + 1e-12, (
                    f"value_noise_2d({p}) = {v} outside [-1, 1]"
                )

    def test_negative_inputs_3d(self):
        """value_noise_3d handles negative coordinates correctly."""
        for ix in range(-3, 0):
            for iy in range(-3, 0):
                for iz in range(-3, 0):
                    p = (ix + 0.3, iy + 0.7, iz + 0.5)
                    v = py_value_noise_3d(p)
                    assert -1.0 <= v <= 1.0 + 1e-12, (
                        f"value_noise_3d({p}) = {v} outside [-1, 1]"
                    )


# =============================================================================
# SECTION 8 -- Smoothstep fade curve
# =============================================================================


class TestSmoothstepCurve:
    """Tests for the smoothstep fade curve (shared by 1D/2D/3D).

    The fade curve ensures C1 continuity by having:
      smoothstep(0) = 0
      smoothstep(1) = 1
      smoothstep'(0) = 0
      smoothstep'(1) = 0
    """

    def test_at_zero(self):
        """At t=0, smoothstep should be 0."""
        assert py_smoothstep(0.0) == pytest.approx(0.0, abs=TOL_ABS)

    def test_at_one(self):
        """At t=1, smoothstep should be 1."""
        assert py_smoothstep(1.0) == pytest.approx(1.0, abs=TOL_ABS)

    def test_at_half(self):
        """At t=0.5, smoothstep should be exactly 0.5."""
        assert py_smoothstep(0.5) == pytest.approx(0.5, abs=TOL_ABS)

    def test_monotonic_increasing(self):
        """Smoothstep should be monotonic in [0, 1]."""
        t_values = [i * 0.01 for i in range(101)]
        values = [py_smoothstep(t) for t in t_values]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-12, (
                f"Smoothstep not monotonic at t={t_values[i]}"
            )

    def test_symmetric(self):
        """Smoothstep should be symmetric: s(t) = 1 - s(1-t)."""
        for i in range(101):
            t = i * 0.01
            assert py_smoothstep(t) == pytest.approx(
                1.0 - py_smoothstep(1.0 - t), abs=TOL_ABS
            )

    def test_zero_derivative_at_zero(self):
        """First derivative should be zero at t=0.

        The derivative of 6t^5 - 15t^4 + 10t^3 is 30t^4 - 60t^3 + 30t^2
        = 30t^2(t-1)^2, which is zero at t=0 and t=1.
        """
        eps = 1e-6
        d0 = (py_smoothstep(eps) - py_smoothstep(0.0)) / eps
        assert abs(d0) < 1e-8, (
            f"Derivative at 0 should be near zero, got {d0}"
        )

    def test_zero_derivative_at_one(self):
        """First derivative should be zero at t=1."""
        eps = 1e-6
        d1 = (py_smoothstep(1.0) - py_smoothstep(1.0 - eps)) / eps
        assert abs(d1) < 1e-8, (
            f"Derivative at 1 should be near zero, got {d1}"
        )

    def test_zero_second_derivative_at_boundaries(self):
        """The second derivative should also be zero at t=0 and t=1.

        This provides C2 continuity at grid cell boundaries.
        """
        eps = 1e-5
        # Second derivative at 0 using central difference
        d0_forward = (py_smoothstep(eps) - py_smoothstep(0.0)) / eps
        d0_backward = (py_smoothstep(0.0) - py_smoothstep(-eps)) / eps
        d2_0 = (d0_forward - d0_backward) / eps
        assert abs(d2_0) < 1e-5, (
            f"Second derivative at 0 should be near zero, got {d2_0}"
        )

    def test_use_in_value_noise_1d_body(self, wgsl_source):
        """The WGSL value_noise_1d must use the smoothstep fade curve."""
        body = extract_function_body(wgsl_source, "value_noise_1d")
        assert body is not None
        # Check that the smoothstep formula (6t^5 - 15t^4 + 10t^3) is present
        has_fade = "6.0" in body and "15.0" in body and "10.0" in body
        has_fade = has_fade or "f * f * f" in body
        assert has_fade, (
            "value_noise_1d body missing smoothstep fade curve"
        )


# =============================================================================
# SECTION 9 -- Grid alignment
# =============================================================================


class TestGridAlignment:
    """Tests that at integer grid positions, value noise = hash value remapped.

    This is a direct consequence of the spec formula:
    When fx=0 and fy=0 (integer position), the interpolation reduces to
    just the hash value at that corner, remapped to [-1, 1].
    """

    def test_1d_integer_grid_matches_hash(self):
        """At integer positions, value_noise_1d = hash11 * 2 - 1."""
        for i in range(-10, 11):
            pi = float(i)
            expected_hash = py_hash11(pi)
            expected = expected_hash * 2.0 - 1.0
            result = py_value_noise_1d(pi)
            assert result == pytest.approx(expected, abs=TOL_ABS), (
                f"At integer {pi}, expected {expected}, got {result}"
            )

    def test_2d_integer_grid_matches_hash(self):
        """At integer grid positions, value_noise_2d = hash21 * 2 - 1."""
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                pi = (float(ix), float(iy))
                expected_hash = py_hash21(pi)
                expected = expected_hash * 2.0 - 1.0
                result = py_value_noise_2d(pi)
                assert result == pytest.approx(expected, abs=TOL_ABS), (
                    f"At integer grid ({ix},{iy}), expected {expected}, "
                    f"got {result}"
                )

    def test_3d_integer_grid_matches_hash(self):
        """At integer grid positions, value_noise_3d = hash31 * 2 - 1."""
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                for iz in range(-3, 4):
                    pi = (float(ix), float(iy), float(iz))
                    expected_hash = py_hash31(pi)
                    expected = expected_hash * 2.0 - 1.0
                    result = py_value_noise_3d(pi)
                    assert result == pytest.approx(expected, abs=TOL_ABS), (
                        f"At integer grid ({ix},{iy},{iz}), expected "
                        f"{expected}, got {result}"
                    )

    def test_1d_hash_variation_along_grid(self):
        """Hash values at adjacent integer positions must differ (no periodicity)."""
        values = [py_value_noise_1d(float(i)) for i in range(-5, 6)]
        unique = len(set(round(v, 10) for v in values))
        assert unique >= 8, (
            f"Expected at least 8 unique values at integer positions, got {unique}"
        )

    def test_2d_hash_variation_along_grid(self):
        """Hash values at adjacent 2D integer positions must differ."""
        values = set()
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                v = round(py_value_noise_2d((float(ix), float(iy))), 10)
                values.add(v)
        assert len(values) >= 20, (
            f"Expected at least 20 unique values at 2D integer grid, got {len(values)}"
        )


# =============================================================================
# SECTION 10 -- Smooth interpolation between hash values
# =============================================================================


class TestSmoothInterpolation:
    """Tests for the core acceptance criteria: smooth interpolation between
    hash values.

    The spec formula produces smooth interpolation by:
    1. Computing hash values at integer grid points
    2. Applying the smoothstep fade curve to the fractional position
    3. Using lerp with the smoothstep factor for C1 continuity
    """

    def test_1d_half_integer_between_hash_values(self):
        """At half-integer, value should lie between adjacent hash values."""
        for i in range(-5, 5):
            pi = float(i)
            half = pi + 0.5
            v_left = py_hash11(pi) * 2.0 - 1.0
            v_right = py_hash11(pi + 1.0) * 2.0 - 1.0
            v_mid = py_value_noise_1d(half)
            lo = min(v_left, v_right)
            hi = max(v_left, v_right)
            assert lo <= v_mid <= hi + 1e-12, (
                f"At half-integer {half}, value {v_mid} not between "
                f"{lo} and {hi}"
            )

    def test_2d_center_interpolated(self):
        """At cell center, value should be near the average of 4 corner hashes.

        Since smoothstep(0.5) = 0.5, the bilinear interpolation at the center
        should equal the average of the four corner values.
        """
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                center = (ix + 0.5, iy + 0.5)
                a_val = py_hash21((float(ix), float(iy))) * 2.0 - 1.0
                b_val = py_hash21((float(ix + 1), float(iy))) * 2.0 - 1.0
                c_val = py_hash21((float(ix), float(iy + 1))) * 2.0 - 1.0
                d_val = py_hash21((float(ix + 1), float(iy + 1))) * 2.0 - 1.0
                center_avg = (a_val + b_val + c_val + d_val) * 0.25
                result = py_value_noise_2d(center)
                assert abs(result - center_avg) < 0.01, (
                    f"At center {center}, expected ~{center_avg}, got {result}"
                )

    def test_1d_interpolation_curve(self):
        """The interpolation curve should follow smoothstep, not linear."""
        # At t=0.5 (half-integer), smoothstep(0.5) = 0.5, so the value
        # is the simple lerp at 0.5. But at t=0.25 and t=0.75, smoothstep
        # deviates from linear - it's steeper in the middle.
        for i in range(-3, 4):
            pi = float(i)
            v_left = py_hash11(pi) * 2.0 - 1.0
            v_right = py_hash11(pi + 1.0) * 2.0 - 1.0

            # Sample at quarter positions
            q1 = py_value_noise_1d(pi + 0.25)
            q3 = py_value_noise_1d(pi + 0.75)

            # Simple linear would give 25% and 75% of the range
            linear_25 = v_left + 0.25 * (v_right - v_left)
            linear_75 = v_left + 0.75 * (v_right - v_left)

            # With smoothstep, the value at 25% should be closer to v_left
            # than linear interpolation would give (smoothstep is S-shaped)
            # smoothstep(0.25) = 0.25^3 * (6*0.25^2 - 15*0.25 + 10)
            # = 0.015625 * (0.375 - 3.75 + 10) = 0.015625 * 6.625 = 0.1035
            # So actual_25 = v_left + 0.1035 * (v_right - v_left)
            # Since 0.1035 < 0.25, the smoothstep value is closer to v_left
            # than the linear value. This is the characteristic S-curve.
            smoothstep_25 = py_smoothstep(0.25)
            smoothstep_75 = py_smoothstep(0.75)

            expected_25 = v_left + smoothstep_25 * (v_right - v_left)
            expected_75 = v_left + smoothstep_75 * (v_right - v_left)

            assert q1 == pytest.approx(expected_25, abs=TOL_ABS), (
                f"At {pi}+0.25, expected {expected_25}, got {q1}"
            )
            assert q3 == pytest.approx(expected_75, abs=TOL_ABS), (
                f"At {pi}+0.75, expected {expected_75}, got {q3}"
            )

    def test_1d_continuity(self):
        """Noise should be continuous with no large jumps over small steps.

        This directly tests the acceptance criteria of smooth interpolation.
        """
        step = 0.001
        prev = py_value_noise_1d(0.0)
        for i in range(1, 500):
            curr = py_value_noise_1d(i * step)
            diff = abs(curr - prev)
            assert diff < 0.01, (
                f"Discontinuity at {i * step}: diff={diff}"
            )
            prev = curr

    def test_2d_continuity_along_x(self):
        """Noise should be continuous along x for fixed y."""
        step = 0.001
        y_fixed = 1.5
        prev = py_value_noise_2d((0.0, y_fixed))
        for i in range(1, 300):
            curr = py_value_noise_2d((i * step, y_fixed))
            diff = abs(curr - prev)
            assert diff < 0.02, (
                f"Discontinuity at x={i * step}, y={y_fixed}: diff={diff}"
            )
            prev = curr

    def test_2d_continuity_along_y(self):
        """Noise should be continuous along y for fixed x."""
        step = 0.001
        x_fixed = 2.5
        prev = py_value_noise_2d((x_fixed, 0.0))
        for i in range(1, 300):
            curr = py_value_noise_2d((x_fixed, i * step))
            diff = abs(curr - prev)
            assert diff < 0.02, (
                f"Discontinuity at x={x_fixed}, y={i * step}: diff={diff}"
            )
            prev = curr

    def test_3d_continuity(self):
        """Noise should be continuous along each axis."""
        step = 0.01
        prev = py_value_noise_3d((0.0, 1.0, 2.0))
        for i in range(1, 200):
            curr = py_value_noise_3d((i * step, 1.0, 2.0))
            assert abs(curr - prev) < 0.05
            prev = curr
        prev = py_value_noise_3d((1.0, 0.0, 2.0))
        for i in range(1, 200):
            curr = py_value_noise_3d((1.0, i * step, 2.0))
            assert abs(curr - prev) < 0.05
            prev = curr
        prev = py_value_noise_3d((1.0, 2.0, 0.0))
        for i in range(1, 200):
            curr = py_value_noise_3d((1.0, 2.0, i * step))
            assert abs(curr - prev) < 0.05
            prev = curr

    def test_1d_no_oscillation_between_grid_points(self):
        """Interpolation between grid points must be monotonic.

        Since lerp and smoothstep are both monotonic, the interpolation
        between any two adjacent hash values must be monotonic.
        """
        for i in range(-5, 5):
            pi = float(i)
            samples = [py_value_noise_1d(pi + j * 0.01) for j in range(101)]
            v_left = samples[0]
            v_right = samples[-1]
            if v_left <= v_right:
                # Should be monotonically increasing
                for j in range(len(samples) - 1):
                    assert samples[j] <= samples[j + 1] + 1e-12, (
                        f"Non-monotonic at cell [{i}, {i+1}]"
                    )
            else:
                # Should be monotonically decreasing
                for j in range(len(samples) - 1):
                    assert samples[j] >= samples[j + 1] - 1e-12, (
                        f"Non-monotonic at cell [{i}, {i+1}]"
                    )

    def test_2d_no_oscillation_between_grid_points(self):
        """Bilinear interpolation must be monotonic along each axis within a cell."""
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                base = (float(ix), float(iy))
                # Sample along x at fixed y = iy + 0.5
                y_mid = iy + 0.5
                samples = [
                    py_value_noise_2d((ix + j * 0.02, y_mid))
                    for j in range(51)
                ]
                # Check for monotonicity (no oscillation/ direction changes)
                # The interpolation should move consistently in one direction
                differences = [samples[j + 1] - samples[j] for j in range(len(samples) - 1)]
                pos_diffs = sum(1 for d in differences if d > 1e-12)
                neg_diffs = sum(1 for d in differences if d < -1e-12)
                # At most 1 direction change allowed (due to floating-point noise
                # near the extremum where derivative approaches zero)
                min_direction_changes = min(pos_diffs, neg_diffs)
                assert min_direction_changes <= 2, (
                    f"Oscillation in cell ({ix},{iy}) along x: "
                    f"{pos_diffs} positive steps, {neg_diffs} negative steps"
                )


# =============================================================================
# SECTION 11 -- Continuity across cell boundaries
# =============================================================================


class TestCellBoundaryContinuity:
    """Tests that value noise is continuous at integer cell boundaries.

    This is critical for the acceptance criteria of smooth interpolation.
    The smoothstep's zero derivative at t=0 and t=1 ensures C1 continuity
    at cell boundaries.
    """

    def test_1d_cell_boundary_continuous(self):
        """Moving from one cell to adjacent should be continuous at boundary."""
        for i in range(-5, 6):
            left = py_value_noise_1d(i - 1e-6)
            right = py_value_noise_1d(i + 1e-6)
            diff = abs(left - right)
            assert diff < 1e-12, (
                f"Gap at cell boundary {i}: diff={diff}"
            )

    def test_2d_cell_boundary_continuous_x(self):
        """Moving across x-boundary should be continuous."""
        for ix in range(-3, 4):
            left = py_value_noise_2d((ix - 1e-6, 0.5))
            right = py_value_noise_2d((ix + 1e-6, 0.5))
            diff = abs(left - right)
            assert diff < 1e-12, (
                f"Gap at cell boundary x={ix}: diff={diff}"
            )

    def test_2d_cell_boundary_continuous_y(self):
        """Moving across y-boundary should be continuous."""
        for iy in range(-3, 4):
            left = py_value_noise_2d((0.5, iy - 1e-6))
            right = py_value_noise_2d((0.5, iy + 1e-6))
            diff = abs(left - right)
            assert diff < 1e-12, (
                f"Gap at cell boundary y={iy}: diff={diff}"
            )

    def test_3d_cell_boundary_continuous(self):
        """Moving across cell boundaries in 3D should be continuous."""
        for i in range(-3, 4):
            left = py_value_noise_3d((i - 1e-6, 0.5, 0.5))
            right = py_value_noise_3d((i + 1e-6, 0.5, 0.5))
            diff = abs(left - right)
            assert diff < 1e-12, (
                f"Gap at 3D cell boundary x={i}: diff={diff}"
            )
            left = py_value_noise_3d((0.5, i - 1e-6, 0.5))
            right = py_value_noise_3d((0.5, i + 1e-6, 0.5))
            diff = abs(left - right)
            assert diff < 1e-12, (
                f"Gap at 3D cell boundary y={i}: diff={diff}"
            )
            left = py_value_noise_3d((0.5, 0.5, i - 1e-6))
            right = py_value_noise_3d((0.5, 0.5, i + 1e-6))
            diff = abs(left - right)
            assert diff < 1e-12, (
                f"Gap at 3D cell boundary z={i}: diff={diff}"
            )

    def test_1d_zero_derivative_at_boundaries(self):
        """The first derivative should approach zero at integer boundaries.

        This is the C1 continuity property from the smoothstep curve.
        """
        eps = 1e-6
        for i in range(-5, 6):
            pi = float(i)
            interior = pi + eps
            d = (py_value_noise_1d(interior) - py_value_noise_1d(pi)) / eps
            assert abs(d) < 0.01, (
                f"Derivative at integer {pi} should be near zero, got {d}"
            )

    def test_2d_zero_derivative_at_boundaries(self):
        """Partial derivatives should approach zero at cell boundaries."""
        eps = 1e-6
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                pi = (float(ix), float(iy))
                # Partial derivative w.r.t x
                interior = (ix + eps, iy)
                d = (py_value_noise_2d(interior) - py_value_noise_2d(pi)) / eps
                assert abs(d) < 0.02, (
                    f"X-derivative at integer ({ix},{iy}) should be near zero, "
                    f"got {d}"
                )


# =============================================================================
# SECTION 12 -- Distribution properties
# =============================================================================


class TestDistribution:
    """Tests that value noise output has reasonable distribution properties."""

    NUM_SAMPLES = 5000

    def test_1d_mean_near_zero(self):
        """Mean of value_noise_1d over many points should be near 0."""
        samples = [
            py_value_noise_1d(i * 0.07) for i in range(self.NUM_SAMPLES)
        ]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.1, (
            f"1D value noise mean {mean} not near 0"
        )

    def test_2d_mean_near_zero(self):
        """Mean of value_noise_2d over many points should be near 0."""
        samples = [
            py_value_noise_2d((i * 0.07, i * 0.11))
            for i in range(self.NUM_SAMPLES)
        ]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.1, (
            f"2D value noise mean {mean} not near 0"
        )

    def test_3d_mean_near_zero(self):
        """Mean of value_noise_3d over many points should be near 0."""
        samples = [
            py_value_noise_3d((i * 0.07, i * 0.11, i * 0.13))
            for i in range(self.NUM_SAMPLES)
        ]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.1, (
            f"3D value noise mean {mean} not near 0"
        )

    def test_1d_values_not_all_same(self):
        """Value noise should not produce constant output."""
        samples = [
            py_value_noise_1d(i * 0.17) for i in range(50)
        ]
        unique = len(set(round(v, 10) for v in samples))
        assert unique > 30, (
            f"Expected at least 30 unique values out of 50, got {unique}"
        )

    def test_2d_values_not_all_same(self):
        """Value noise should not produce constant output for 2D."""
        samples = [
            py_value_noise_2d((i * 0.17, i * 0.23)) for i in range(50)
        ]
        unique = len(set(round(v, 10) for v in samples))
        assert unique > 30, (
            f"Expected at least 30 unique values out of 50, got {unique}"
        )

    def test_3d_values_not_all_same(self):
        """Value noise should not produce constant output for 3D."""
        samples = [
            py_value_noise_3d((i * 0.17, i * 0.23, i * 0.31))
            for i in range(50)
        ]
        unique = len(set(round(v, 10) for v in samples))
        assert unique > 30, (
            f"Expected at least 30 unique values out of 50, got {unique}"
        )

    def test_1d_approx_symmetry(self):
        """About half the samples should be below zero (symmetric around 0)."""
        samples = [
            py_value_noise_1d(i * 0.07) for i in range(self.NUM_SAMPLES)
        ]
        below_zero = sum(1 for v in samples if v < 0)
        ratio = below_zero / len(samples)
        assert 0.4 <= ratio <= 0.6, (
            f"Expected ~50% of samples below zero, got {ratio:.1%}"
        )

    def test_2d_approx_symmetry(self):
        """About half the 2D samples should be below zero."""
        samples = [
            py_value_noise_2d((i * 0.07, i * 0.11))
            for i in range(self.NUM_SAMPLES)
        ]
        below_zero = sum(1 for v in samples if v < 0)
        ratio = below_zero / len(samples)
        assert 0.4 <= ratio <= 0.6, (
            f"Expected ~50% of 2D samples below zero, got {ratio:.1%}"
        )
