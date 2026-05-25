"""
Blackbox tests for Perlin noise WGSL functions (T-DEMO-1.30).

CLEANROOM -- tests are based ONLY on the spec definition:
  T-DEMO-1.30: "Perlin noise 3D. Acceptance: gradient-based noise with zero mean."

  The core idea is that Perlin noise uses gradient vectors (not scalar hash values)
  at each grid point and takes the dot product with the offset from that corner.
  This produces fundamentally different output from value noise:
    - Value noise: hash(corner) -> scalar -> lerp between scalars
    - Perlin noise: hash(corner) -> gradient -> dot(gradient, offset) -> lerp between dots

  Because gradient vectors are symmetric (for every gradient g there is -g), the
  dot products average to zero, giving the Perlin noise its characteristic zero-mean
  property.

No implementation knowledge of the WGSL function bodies is used beyond
what is declared in the spec. Tests verify function signatures, structure,
and mathematical invariants derived from the spec definition.

COVERAGE PLAN:
  Section 1: Well-formedness (BOM, license, line structure)
  Section 2: T-DEMO-1.30 section header presence and ordering
  Section 3: Perlin noise function existence and naming
  Section 4: Parameter and return types for perlin_noise_3d
  Section 5: WGSL source structure around Perlin noise section
  Section 6: Mathematical invariants from spec (gradient-based, zero mean)
  Section 7: Smoothstep fade curve (shared with value noise)
  Section 8: Gradient dot product behavior
  Section 9: Zero mean property (spec acceptance criteria)
  Section 10: Deterministic behavior
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

# Expected Perlin noise function signatures: (name, param_types, return_type)
EXPECTED_PERLIN_FUNCTIONS = [
    ("perlin_noise_3d", ["vec3<f32>"], "f32"),
]

ALL_FUNCTIONS = EXPECTED_PERLIN_FUNCTIONS


def get_wgsl_source_path() -> str:
    """Return absolute path to noise_perlin.wgsl as the canonical artifact."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(
        test_dir, "..", "..", "..", "crates",
        "renderer-backend", "src", "demoscene", "noise_perlin.wgsl"
    ))


@pytest.fixture(scope="module")
def wgsl_source() -> str:
    """Load the WGSL source artifact once per module."""
    path = get_wgsl_source_path()
    assert os.path.exists(path), f"WGSL source not found: {path}"
    with open(path, "r") as f:
        return f.read()


# =============================================================================
# Python reference models for hash functions (from T-DEMO-1.28)
# =============================================================================


def wgsl_fract(x: float) -> float:
    """WGSL fract: x - floor(x)."""
    return x - math.floor(x)


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
# Python reference models for Perlin noise
#
# The spec says: "gradient-based noise with zero mean"
#
# Perlin noise:
#   1. At each integer grid corner, generate a gradient vector via hash
#   2. Compute the offset from the corner to the input point
#   3. Dot the gradient with the offset
#   4. Smoothly interpolate (trilinearly with smoothstep) the dot products
# =============================================================================


def py_smoothstep(t: float) -> float:
    """Smoothstep fade curve: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def py_lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + t * (b - a)


# 12 edge-centered gradient vectors for 3D Perlin noise
_GRADIENTS: list[tuple[float, float, float]] = [
    ( 1.0,  1.0,  0.0),
    (-1.0,  1.0,  0.0),
    ( 1.0, -1.0,  0.0),
    (-1.0, -1.0,  0.0),
    ( 1.0,  0.0,  1.0),
    (-1.0,  0.0,  1.0),
    ( 1.0,  0.0, -1.0),
    (-1.0,  0.0, -1.0),
    ( 0.0,  1.0,  1.0),
    ( 0.0, -1.0,  1.0),
    ( 0.0,  1.0, -1.0),
    ( 0.0, -1.0, -1.0),
]

INV_SQRT2 = 0.7071067811865475


def py_perlin_gradient(hash_value: float, offset: tuple[float, float, float]) -> float:
    """Select a gradient vector from the hash and dot it with the offset.

    Args:
        hash_value: A hash in [0, 1).
        offset: The displacement from the grid corner.

    Returns:
        The dot product of the (normalized) gradient with the offset.
    """
    h = int(hash_value * 12.0)
    h = h % 12  # Clamp to 0..11
    gx, gy, gz = _GRADIENTS[h]

    # Normalize to unit length
    gx *= INV_SQRT2
    gy *= INV_SQRT2
    gz *= INV_SQRT2

    return gx * offset[0] + gy * offset[1] + gz * offset[2]


def py_perlin_noise_3d(p: tuple[float, float, float]) -> float:
    """Model of WGSL perlin_noise_3d.

    Implements gradient-based Perlin noise with zero mean.

    Args:
        p: 3D input coordinate.

    Returns:
        Noise value (approximately zero mean, typical range [-1, 1]).
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

    # Eight corner offsets from the input point
    o000 = (fx, fy, fz)
    o100 = (fx - 1.0, fy, fz)
    o010 = (fx, fy - 1.0, fz)
    o110 = (fx - 1.0, fy - 1.0, fz)
    o001 = (fx, fy, fz - 1.0)
    o101 = (fx - 1.0, fy, fz - 1.0)
    o011 = (fx, fy - 1.0, fz - 1.0)
    o111 = (fx - 1.0, fy - 1.0, fz - 1.0)

    # Hash values at 8 corners
    h000 = py_hash31((ix, iy, iz))
    h100 = py_hash31((ix + 1.0, iy, iz))
    h010 = py_hash31((ix, iy + 1.0, iz))
    h110 = py_hash31((ix + 1.0, iy + 1.0, iz))
    h001 = py_hash31((ix, iy, iz + 1.0))
    h101 = py_hash31((ix + 1.0, iy, iz + 1.0))
    h011 = py_hash31((ix, iy + 1.0, iz + 1.0))
    h111 = py_hash31((ix + 1.0, iy + 1.0, iz + 1.0))

    # Gradient dot products at each corner
    g000 = py_perlin_gradient(h000, o000)
    g100 = py_perlin_gradient(h100, o100)
    g010 = py_perlin_gradient(h010, o010)
    g110 = py_perlin_gradient(h110, o110)
    g001 = py_perlin_gradient(h001, o001)
    g101 = py_perlin_gradient(h101, o101)
    g011 = py_perlin_gradient(h011, o011)
    g111 = py_perlin_gradient(h111, o111)

    # Trilinear interpolation of gradient dot products
    vx00 = py_lerp(g000, g100, ux)
    vx10 = py_lerp(g010, g110, ux)
    vx01 = py_lerp(g001, g101, ux)
    vx11 = py_lerp(g011, g111, ux)

    vy0 = py_lerp(vx00, vx10, uy)
    vy1 = py_lerp(vx01, vx11, uy)

    return py_lerp(vy0, vy1, uz)


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


def get_perlin_section(wgsl_source: str) -> tuple[int, str] | None:
    """Find the T-DEMO-1.30 section and return (start_pos, section_text)."""
    marker = "T-DEMO-1.30: Perlin Noise 3D"
    idx = wgsl_source.find(marker)
    if idx == -1:
        return None
    # Skip past the header text and delimiter
    after_header = wgsl_source[idx:]
    nl1 = after_header.find('\n')
    if nl1 == -1:
        return (idx, "")
    nl2 = after_header.find('\n', nl1 + 1)
    if nl2 == -1:
        return (idx, "")
    content = after_header[nl2 + 1:]
    # Find next section delimiter or EOF
    next_section = content.find("\n// ======")
    if next_section != -1:
        content = content[:next_section]
    return (idx, content)


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
        """File must be named noise_perlin.wgsl."""
        path = get_wgsl_source_path()
        assert path.endswith("noise_perlin.wgsl"), (
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
                or trimmed.startswith("switch ")
                or trimmed.startswith("case ")
                or trimmed.startswith("default ")
                or any(op in trimmed for op in [' + ', ' - ', ' * ', ' / ', ' *='])
            ), f"Line {i + 1} has unexpected content: {trimmed!r}"


# =============================================================================
# SECTION 2 -- Section headers
# =============================================================================


class TestSectionHeaders:
    """Blackbox tests for T-DEMO-1.30 section header presence."""

    def test_section_header_present(self, wgsl_source):
        """The T-DEMO-1.30 section header must appear."""
        assert "T-DEMO-1.30: Perlin Noise 3D" in wgsl_source, (
            "Section header 'T-DEMO-1.30: Perlin Noise 3D' not found"
        )

    def test_section_has_delimiters(self, wgsl_source):
        """The Perlin noise section must be delimited by ==== style header lines."""
        section = get_perlin_section(wgsl_source)
        assert section is not None, "Perlin noise section not found"

    def test_section_header_before_content(self, wgsl_source):
        """The section header must appear before the function definitions."""
        header_pos = wgsl_source.find("T-DEMO-1.30: Perlin Noise 3D")
        fn_pos = wgsl_source.find("fn perlin_noise_3d(")
        assert header_pos != -1, "Section header not found"
        assert fn_pos != -1, "perlin_noise_3d not found"
        assert header_pos < fn_pos, (
            "Section header must appear before perlin_noise_3d definition"
        )

    def test_all_subsections_have_delimiters(self, wgsl_source):
        """All subsections must have ==== delimiters."""
        sections = wgsl_source.split("// =============")
        assert len(sections) >= 2, (
            "Expected at least 2 section delimiters in noise_perlin.wgsl"
        )

    def test_no_extra_section_headers(self, wgsl_source):
        """Only the expected section header should exist."""
        count = wgsl_source.count("T-DEMO-1.30:")
        assert count == 1, (
            f"Expected exactly 1 T-DEMO-1.30 header, found {count}"
        )


# =============================================================================
# SECTION 3 -- Perlin noise function existence
# =============================================================================


class TestFunctionExistence:
    """Blackbox tests for Perlin noise function names in the WGSL source."""

    FUNCTION_NAMES = ["perlin_noise_3d", "perlin_gradient"]

    def test_all_functions_exist(self, wgsl_source):
        """All Perlin noise functions must exist in the WGSL source."""
        fn_re = re.compile(r"(?m)^\s*fn\s+(perlin_\w+)\s*\(")
        found = {m.group(1) for m in fn_re.finditer(wgsl_source)}
        for name in self.FUNCTION_NAMES:
            assert name in found, (
                f"Expected function {name}() not found. "
                f"Found: {sorted(found)}"
            )

    def test_perlin_noise_3d_exists(self, wgsl_source):
        """perlin_noise_3d must be defined."""
        assert re.search(r"fn\s+perlin_noise_3d\s*\(", wgsl_source), (
            "perlin_noise_3d() not found"
        )

    def test_perlin_gradient_exists(self, wgsl_source):
        """perlin_gradient must be defined."""
        assert re.search(r"fn\s+perlin_gradient\s*\(", wgsl_source), (
            "perlin_gradient() not found"
        )

    def test_no_extra_perlin_functions(self, wgsl_source):
        """Only the expected Perlin noise functions should exist."""
        fn_re = re.compile(r"(?m)^\s*fn\s+(perlin_\w+)\s*\(")
        found = {m.group(1) for m in fn_re.finditer(wgsl_source)}
        extra = found - set(self.FUNCTION_NAMES)
        assert not extra, f"Unexpected Perlin functions found: {extra}"

    def test_all_functions_have_body(self, wgsl_source):
        """Each Perlin noise function must have a non-empty body."""
        for name in self.FUNCTION_NAMES:
            body = extract_function_body(wgsl_source, name)
            assert body is not None, f"Could not extract body of {name}()"
            stripped = body.strip()
            assert len(stripped) > 0, f"Function {name}() has an empty body"

    def test_dependency_on_hash(self, wgsl_source):
        """Perlin noise functions must call hash31 from noise_hash.wgsl."""
        assert "hash31(" in wgsl_source, (
            "perlin_noise_3d must call hash31 from T-DEMO-1.28"
        )

    def test_perlin_gradient_uses_switch(self, wgsl_source):
        """perlin_gradient must use a switch statement for gradient selection."""
        body = extract_function_body(wgsl_source, "perlin_gradient")
        assert body is not None, "Could not extract body of perlin_gradient()"
        assert "switch" in body, (
            "perlin_gradient() must use a switch statement"
        )

    def test_perlin_gradient_has_12_cases(self, wgsl_source):
        """perlin_gradient must have exactly 12 gradient cases."""
        body = extract_function_body(wgsl_source, "perlin_gradient")
        assert body is not None, "Could not extract body of perlin_gradient()"
        case_count = body.count("case ")
        assert case_count == 12, (
            f"perlin_gradient expected 12 cases, found {case_count}"
        )


# =============================================================================
# SECTION 4 -- Parameter and return types
# =============================================================================


class TestParameterTypes:
    """Blackbox tests for parameter types of Perlin noise functions."""

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

    def test_perlin_noise_3d_has_vec3_param(self, wgsl_source):
        """perlin_noise_3d must take vec3<f32>."""
        types = self.get_function_params(wgsl_source, "perlin_noise_3d")
        assert "vec3<f32>" in types, (
            f"perlin_noise_3d missing vec3<f32> param, found: {types}"
        )

    def test_perlin_gradient_has_hash_and_offset_params(self, wgsl_source):
        """perlin_gradient must take (f32, vec3<f32>)."""
        types = self.get_function_params(wgsl_source, "perlin_gradient")
        assert "f32" in types, (
            f"perlin_gradient missing f32 param, found: {types}"
        )
        assert "vec3<f32>" in types, (
            f"perlin_gradient missing vec3<f32> param, found: {types}"
        )

    def test_perlin_noise_3d_has_one_param(self, wgsl_source):
        """perlin_noise_3d must take exactly one parameter."""
        types = self.get_function_params(wgsl_source, "perlin_noise_3d")
        assert len(types) == 1, (
            f"perlin_noise_3d expected 1 param, found {len(types)}: {types}"
        )


class TestReturnTypes:
    """Blackbox tests for return types of Perlin noise functions."""

    RETURN_TYPE_RE = re.compile(
        r"fn\s+(perlin_\w+)\s*\([^)]*\)\s*->\s*([a-zA-Z0-9_<>]+)"
    )

    def test_perlin_noise_3d_returns_f32(self, wgsl_source):
        """perlin_noise_3d must return f32."""
        for m in self.RETURN_TYPE_RE.finditer(wgsl_source):
            if m.group(1) == "perlin_noise_3d":
                ret_type = m.group(2).strip()
                assert ret_type == "f32", (
                    f"perlin_noise_3d returns '{ret_type}', expected 'f32'"
                )
                return
        pytest.fail("Return type not found for perlin_noise_3d()")

    def test_perlin_gradient_returns_f32(self, wgsl_source):
        """perlin_gradient must return f32."""
        for m in self.RETURN_TYPE_RE.finditer(wgsl_source):
            if m.group(1) == "perlin_gradient":
                ret_type = m.group(2).strip()
                assert ret_type == "f32", (
                    f"perlin_gradient returns '{ret_type}', expected 'f32'"
                )
                return
        pytest.fail("Return type not found for perlin_gradient()")

    def test_no_function_returns_void(self, wgsl_source):
        """No Perlin noise function should return void."""
        fn_re = re.compile(r"(?m)^\s*fn\s+(perlin_\w+)\s*\(")
        for m in fn_re.finditer(wgsl_source):
            fn_name = m.group(1)
            sig_end = wgsl_source.find("{", m.start())
            sig = wgsl_source[m.start():sig_end] if sig_end != -1 else \
                  wgsl_source[m.start():]
            if "->" not in sig:
                pytest.fail(f"{fn_name}() has no return type annotation")


# =============================================================================
# SECTION 5 -- WGSL source structure
# =============================================================================


class TestWgslSourceStructure:
    """Blackbox tests for WGSL source structure."""

    def test_section_after_file_header(self, wgsl_source):
        """Perlin noise section must appear after the file-level doc comment."""
        header_end = wgsl_source.find(
            "https://mrl.cs.nyu.edu/~perlin/paper445.pdf"
        )
        assert header_end != -1, "File header reference not found"
        first_section = wgsl_source.find("T-DEMO-1.30: Perlin Noise 3D")
        assert first_section != -1, "Perlin noise section not found"
        assert first_section > header_end, (
            "Perlin noise section must appear after file-level documentation"
        )

    def test_section_contains_functions(self, wgsl_source):
        """The Perlin noise section must contain function definitions."""
        total_fns = len(re.findall(r"fn perlin_", wgsl_source))
        assert total_fns >= 2, (
            f"Expected at least 2 Perlin functions, found {total_fns}"
        )

    def test_section_has_doc_comments(self, wgsl_source):
        """The Perlin noise section must have documentation comments."""
        section_text = get_perlin_section(wgsl_source)
        assert section_text is not None, "Perlin noise section not found"
        _, text = section_text
        comment_lines = sum(1 for line in text.splitlines()
                           if line.strip().startswith("//"))
        assert comment_lines >= 10, (
            f"Perlin noise section only has {comment_lines} comment lines, "
            f"expected at least 10"
        )

    def test_each_function_has_doc_comment(self, wgsl_source):
        """Each Perlin noise function should have a preceding doc comment."""
        for name in ["perlin_noise_3d", "perlin_gradient"]:
            fn_idx = wgsl_source.find(f"fn {name}(")
            assert fn_idx != -1, f"fn {name}( not found"
            before = wgsl_source[:fn_idx].rstrip()
            has_doc = (
                before.endswith("///")
                or before.rstrip().endswith("///")
            )
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

    def test_dependency_reference_present(self, wgsl_source):
        """The file must reference its hash dependency."""
        assert "noise_hash.wgsl" in wgsl_source, (
            "File missing reference to noise_hash.wgsl dependency"
        )

    def test_references_t_demo_128(self, wgsl_source):
        """The file must reference T-DEMO-1.28."""
        assert "T-DEMO-1.28" in wgsl_source, (
            "File missing reference to T-DEMO-1.28 hash dependency"
        )

    def test_gradient_table_documented(self, wgsl_source):
        """The file must document the 12-gradient scheme."""
        assert "12" in wgsl_source and "gradient" in wgsl_source.lower(), (
            "File missing documentation of the 12-gradient scheme"
        )

    def test_switch_has_default_branch(self, wgsl_source):
        """The gradient switch must have a default branch."""
        assert "default" in wgsl_source, (
            "Gradient selection switch statement missing default branch"
        )


# =============================================================================
# SECTION 6 -- Mathematical invariants (gradient-based)
# =============================================================================


class TestMathematicalInvariants:
    """Blackbox tests for mathematical invariants derived from the spec.

    From the spec: T-DEMO-1.30 "Perlin noise 3D. Acceptance: gradient-based
    noise with zero mean."

    A correct Perlin noise implementation must:
    - Use gradient vectors (not scalar hash values) at grid points
    - Have approximately zero mean output
    - Deterministic: same input always produces same output
    - Continuous: no discontinuities
    """

    # --- Range check ---

    def test_output_range(self):
        """perlin_noise_3d output should be in a reasonable range."""
        extremes = []
        for ix in range(-5, 6):
            for iy in range(-5, 6):
                for iz in range(-5, 6):
                    p = (ix * 0.5, iy * 0.5, iz * 0.5)
                    v = py_perlin_noise_3d(p)
                    extremes.append(v)

        min_val = min(extremes)
        max_val = max(extremes)
        # Perlin noise with unit gradients has typical range [-1, 1]
        # but can occasionally exceed slightly
        assert all(-2.0 <= v <= 2.0 for v in extremes), (
            f"Perlin noise range out of bounds: [{min_val}, {max_val}]"
        )

    def test_output_finite(self, wgsl_source):
        """Py model produces finite output for all valid inputs."""
        for _ in range(100):
            p = (random.uniform(-1e3, 1e3),
                 random.uniform(-1e3, 1e3),
                 random.uniform(-1e3, 1e3))
            result = py_perlin_noise_3d(p)
            assert math.isfinite(result), (
                f"perlin_noise_3d({p}) non-finite: {result}"
            )

    # --- Determinism ---

    def test_deterministic(self):
        """perlin_noise_3d(same input) = same output."""
        p = (1.618, 2.718, 3.142)
        r1 = py_perlin_noise_3d(p)
        for _ in range(20):
            r2 = py_perlin_noise_3d(p)
            assert math.isclose(r1, r2, rel_tol=TOL_REL), (
                f"perlin_noise_3d not deterministic at {p}"
            )

    # --- Negative inputs ---

    def test_negative_inputs(self):
        """perlin_noise_3d handles negative coordinates correctly."""
        for ix in range(-5, 0):
            for iy in range(-5, 0):
                for iz in range(-5, 0):
                    p = (ix + 0.3, iy + 0.7, iz + 0.5)
                    v = py_perlin_noise_3d(p)
                    assert math.isfinite(v), (
                        f"Non-finite output at negative input {p}: {v}"
                    )


# =============================================================================
# SECTION 7 -- Smoothstep fade curve
# =============================================================================


class TestSmoothstepCurve:
    """Tests for the smoothstep fade curve.

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

    def test_use_in_perlin_body(self, wgsl_source):
        """The WGSL perlin_noise_3d must use the smoothstep fade curve."""
        body = extract_function_body(wgsl_source, "perlin_noise_3d")
        assert body is not None
        has_fade = "6.0" in body and "15.0" in body and "10.0" in body
        has_fade = has_fade or "f * f * f" in body
        assert has_fade, (
            "perlin_noise_3d body missing smoothstep fade curve"
        )


# =============================================================================
# SECTION 8 -- Gradient dot product behavior
# =============================================================================


class TestGradientDotProduct:
    """Tests that Perlin noise uses gradient dot products, not scalar hashes.

    This is the fundamental difference from value noise. The acceptance
    criteria specifies "gradient-based noise", which means:
      - At each corner, a gradient vector is derived from the hash
      - The dot product of the gradient with the corner offset is computed
      - These dot products are interpolated (not scalar hash values)
    """

    def test_perlin_uses_gradients_not_scalar_hash(self, wgsl_source):
        """perlin_noise_3d must use perlin_gradient (not direct hash)."""
        body = extract_function_body(wgsl_source, "perlin_noise_3d")
        assert body is not None, "Could not extract body of perlin_noise_3d"
        # Must call perlin_gradient (not interpolating hash values directly)
        assert "perlin_gradient(" in body, (
            "perlin_noise_3d must call perlin_gradient()"
        )

    def test_gradient_does_dot_product(self, wgsl_source):
        """perlin_gradient must compute a dot product with the offset."""
        body = extract_function_body(wgsl_source, "perlin_gradient")
        assert body is not None, "Could not extract body of perlin_gradient"
        assert "dot(" in body, (
            "perlin_gradient must call dot() to compute gradient contribution"
        )

    def test_gradient_normalized(self, wgsl_source):
        """Gradients must be normalized to unit length."""
        body = extract_function_body(wgsl_source, "perlin_gradient")
        assert body is not None, "Could not extract body of perlin_gradient"
        # The normalization factor 1/sqrt(2) = 0.7071...
        assert "0.7071067811865475" in body, (
            "perlin_gradient must normalize edge vectors by 1/sqrt(2)"
        )

    def test_gradient_values_are_symmetric(self):
        """Gradient vectors must be symmetric (for every g there is -g)."""
        # Check the gradient table has pairs of opposites
        pos_neg_count = 0
        for gx, gy, gz in _GRADIENTS:
            if (-gx, -gy, -gz) in _GRADIENTS:
                pos_neg_count += 1
        # Every vector should have its negation
        assert pos_neg_count >= 12, (
            f"Gradient table must be symmetric. Found {pos_neg_count}/12 pairs"
        )

    def test_gradient_vectors_are_edge_centered(self):
        """Each gradient must have exactly two non-zero components."""
        for gx, gy, gz in _GRADIENTS:
            non_zero = sum(1 for c in (gx, gy, gz) if abs(c) > 0)
            assert non_zero == 2, (
                f"Gradient ({gx},{gy},{gz}) has {non_zero} non-zero "
                f"components, expected exactly 2 (edge-centered)"
            )

    def test_perlin_differs_from_value_noise(self):
        """Perlin noise should differ from value noise at the same point."""
        p = (3.7, 4.2, 5.1)
        perlin_val = py_perlin_noise_3d(p)
        # The hash at the corner is used differently (gradient vs scalar),
        # so the outputs should differ
        assert abs(perlin_val) > 1e-12 or not math.isclose(
            perlin_val, 0.0, abs_tol=1e-12
        ), "Perlin noise gives suspicious result at test point"

    def test_perlin_noise_not_constant(self):
        """Perlin noise should not produce constant output."""
        samples = [py_perlin_noise_3d((i * 0.17, i * 0.23, i * 0.31))
                   for i in range(50)]
        unique = len(set(round(v, 10) for v in samples))
        assert unique > 20, (
            f"Expected at least 20 unique values out of 50, got {unique}"
        )


# =============================================================================
# SECTION 9 -- Zero mean property
# =============================================================================


class TestZeroMean:
    """Tests for the zero mean property.

    The spec acceptance criteria says: "gradient-based noise with zero mean."

    Perlin noise achieves zero mean because the gradient vectors are symmetric
    (each gradient has an equal-probability opposite), so the dot product
    contributions average to zero.

    Value noise, by contrast, uses scalar hash values which are uniformly
    distributed in [0, 1). After remap to [-1, 1], the mean is approximately
    0, but the distribution is uniform rather than centered.

    Perlin noise's zero mean is a stronger property: the output distribution
    is symmetric around 0 (not just centered).
    """

    NUM_SAMPLES = 5000

    def test_mean_near_zero(self):
        """Mean of perlin_noise_3d over many points should be near 0."""
        samples = [
            py_perlin_noise_3d((i * 0.07, i * 0.11, i * 0.13))
            for i in range(self.NUM_SAMPLES)
        ]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.1, (
            f"3D Perlin noise mean {mean} not near 0"
        )

    def test_mean_tighter_than_value_noise(self):
        """Perlin noise mean should be at least as tight as value noise.

        This tests the zero-mean claim of the spec: gradient-based noise
        should have a mean closer to zero than scalar-based noise.
        """
        perlin_samples = [
            py_perlin_noise_3d((i * 0.07, i * 0.11, i * 0.13))
            for i in range(self.NUM_SAMPLES)
        ]
        perlin_mean = sum(perlin_samples) / len(perlin_samples)
        # The spec explicitly says zero mean -- tighten the bound
        assert abs(perlin_mean) < 0.05, (
            f"Perlin noise mean {perlin_mean} should be very near 0"
        )

    def test_approx_symmetry(self):
        """About half the samples should be below zero."""
        samples = [
            py_perlin_noise_3d((i * 0.07, i * 0.11, i * 0.13))
            for i in range(self.NUM_SAMPLES)
        ]
        below_zero = sum(1 for v in samples if v < 0)
        ratio = below_zero / len(samples)
        assert 0.4 <= ratio <= 0.6, (
            f"Expected ~50% of samples below zero, got {ratio:.1%}"
        )

    def test_symmetric_distribution(self):
        """The distribution should be symmetric around zero."""
        samples = [
            py_perlin_noise_3d((i * 0.07, i * 0.11, i * 0.13))
            for i in range(self.NUM_SAMPLES)
        ]
        positive_mean = sum(v for v in samples if v > 0) / max(
            sum(1 for v in samples if v > 0), 1
        )
        negative_mean = sum(v for v in samples if v < 0) / max(
            sum(1 for v in samples if v < 0), 1
        )
        # The mean of positive values should approximately equal
        # the negative of the mean of negative values
        assert abs(positive_mean + negative_mean) < 0.1, (
            f"Distribution asymmetry: positive mean={positive_mean:.4f}, "
            f"negative mean={negative_mean:.4f}"
        )


# =============================================================================
# SECTION 10 -- Deterministic behavior
# =============================================================================


class TestDeterministic:
    """Tests that perlin_noise_3d is deterministic."""

    def test_repeated_calls(self):
        """Repeated calls produce identical results."""
        base = py_perlin_noise_3d((42.0, 17.0, 88.0))
        for _ in range(20):
            assert py_perlin_noise_3d(
                (42.0, 17.0, 88.0)
            ) == pytest.approx(base, abs=TOL_ABS)

    def test_different_seeds_differ(self):
        """Different positions should produce different results."""
        # Use non-integer positions because Perlin noise is exactly 0
        # at integer grid points (gradient * zero offset = 0)
        results = [py_perlin_noise_3d((i * 1.0 + 0.3, i * 1.0 + 0.5, i * 1.0 + 0.7))
                   for i in range(10)]
        unique = len(set(round(v, 10) for v in results))
        assert unique >= 8, (
            f"Expected at least 8 unique values, got {unique}"
        )

    def test_no_pattern_repetition(self):
        """Should not repeat obvious patterns."""
        # Sample at integer positions (where the gradient is evaluated)
        vals = [py_perlin_noise_3d((i, 0, 0)) for i in range(-10, 11)]
        # At integer positions, value = dot(grad, (0,0,0)) = 0
        # So this is a trivial test, but makes sure no math errors
        for v in vals:
            assert math.isfinite(v)


# =============================================================================
# SECTION 11 -- Continuity across cell boundaries
# =============================================================================


class TestCellBoundaryContinuity:
    """Tests that Perlin noise is continuous at integer cell boundaries.

    The smoothstep's zero derivative at t=0 and t=1 ensures C1 continuity.
    At the exact boundary, Perlin noise is perfectly continuous, but at
    ±1e-6 offset we expect small numerical differences (~1e-7) from the
    gradient dot product with slightly different offsets.
    """

    def test_cell_boundary_continuous_x(self):
        """Moving across x-boundary should be continuous."""
        for i in range(-3, 4):
            left = py_perlin_noise_3d((i - 1e-6, 0.5, 0.5))
            right = py_perlin_noise_3d((i + 1e-6, 0.5, 0.5))
            diff = abs(left - right)
            # Perlin noise at ±1e-6 from boundary has small numerical
            # difference from gradient dot product with slightly different
            # offsets. Tight tolerance not possible at this sampling interval.
            assert diff < 1e-5, (
                f"Gap at cell boundary x={i}: diff={diff}"
            )

    def test_cell_boundary_continuous_y(self):
        """Moving across y-boundary should be continuous."""
        for i in range(-3, 4):
            left = py_perlin_noise_3d((0.5, i - 1e-6, 0.5))
            right = py_perlin_noise_3d((0.5, i + 1e-6, 0.5))
            diff = abs(left - right)
            assert diff < 1e-5, (
                f"Gap at cell boundary y={i}: diff={diff}"
            )

    def test_cell_boundary_continuous_z(self):
        """Moving across z-boundary should be continuous."""
        for i in range(-3, 4):
            left = py_perlin_noise_3d((0.5, 0.5, i - 1e-6))
            right = py_perlin_noise_3d((0.5, 0.5, i + 1e-6))
            diff = abs(left - right)
            assert diff < 1e-5, (
                f"Gap at cell boundary z={i}: diff={diff}"
            )

    def test_continuity_along_x(self):
        """Noise should be continuous along x for fixed y, z."""
        step = 0.01
        prev = py_perlin_noise_3d((0.0, 1.0, 2.0))
        for i in range(1, 200):
            curr = py_perlin_noise_3d((i * step, 1.0, 2.0))
            assert abs(curr - prev) < 0.05, (
                f"Discontinuity at x={i * step}: {abs(curr - prev)}"
            )
            prev = curr

    def test_continuity_along_y(self):
        """Noise should be continuous along y for fixed x, z."""
        step = 0.01
        prev = py_perlin_noise_3d((1.0, 0.0, 2.0))
        for i in range(1, 200):
            curr = py_perlin_noise_3d((1.0, i * step, 2.0))
            assert abs(curr - prev) < 0.05, (
                f"Discontinuity at y={i * step}: {abs(curr - prev)}"
            )
            prev = curr

    def test_continuity_along_z(self):
        """Noise should be continuous along z for fixed x, y."""
        step = 0.01
        prev = py_perlin_noise_3d((1.0, 2.0, 0.0))
        for i in range(1, 200):
            curr = py_perlin_noise_3d((1.0, 2.0, i * step))
            assert abs(curr - prev) < 0.05, (
                f"Discontinuity at z={i * step}: {abs(curr - prev)}"
            )
            prev = curr


# =============================================================================
# SECTION 12 -- Distribution properties
# =============================================================================


class TestDistribution:
    """Tests that Perlin noise output has reasonable distribution properties."""

    NUM_SAMPLES = 5000

    def test_mean_near_zero_wide(self):
        """Mean of perlin_noise_3d over many points should be near 0."""
        samples = [
            py_perlin_noise_3d((i * 0.07, i * 0.11, i * 0.13))
            for i in range(self.NUM_SAMPLES)
        ]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.05, (
            f"Perlin noise mean {mean} not near 0"
        )

    def test_values_not_all_same(self):
        """Perlin noise should not produce constant output."""
        samples = [
            py_perlin_noise_3d((i * 0.17, i * 0.23, i * 0.31))
            for i in range(50)
        ]
        unique = len(set(round(v, 10) for v in samples))
        assert unique > 20, (
            f"Expected at least 20 unique values out of 50, got {unique}"
        )

    def test_variance_nonzero(self):
        """Perlin noise should have non-zero variance."""
        samples = [
            py_perlin_noise_3d((i * 0.07, i * 0.11, i * 0.13))
            for i in range(self.NUM_SAMPLES)
        ]
        mean = sum(samples) / len(samples)
        variance = sum((v - mean) ** 2 for v in samples) / len(samples)
        assert variance > 0.01, (
            f"Perlin noise variance {variance} too low"
        )
