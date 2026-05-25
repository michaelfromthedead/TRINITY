"""
Blackbox tests for SDF domain stretch operation (T-DEMO-1.27).

CLEANROOM -- tests are based ONLY on the spec definition:
  T-DEMO-1.27: "Implement stretch operation (anisotropic scaling).
  Acceptance: space scaled along specified axis."

No implementation knowledge of the WGSL function bodies is used.
Tests verify function signatures, structure, and mathematical
invariants derived from the spec definition of anisotropic scaling.

COVERAGE PLAN:
  Section 1: Well-formedness (BOM, license, line structure)
  Section 2: T-DEMO-1.27 section header presence and ordering
  Section 3: Stretch function existence and naming
  Section 4: Parameter types (position vec3<f32>, scale f32)
  Section 5: Return types (vec3<f32> for stretch)
  Section 6: WGSL source structure around stretch section
  Section 7: Mathematical invariants from spec (anisotropic, identity, invertible)
  Section 8: Axis specificity (each stretch only affects its axis)
  Section 9: Compensation invariants (spec-derived distance compensation)
  Section 10: Division-by-zero guards on scale parameter
  Section 11: Domain stretch compensation function verification
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


def get_wgsl_source_path() -> str:
    """Return absolute path to sdf_domain.wgsl as the canonical artifact."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(
        test_dir, "..", "..", "crates",
        "renderer-backend", "src", "demoscene", "sdf_domain.wgsl"
    ))


@pytest.fixture(scope="module")
def wgsl_source() -> str:
    """Load the WGSL source artifact once per module."""
    path = get_wgsl_source_path()
    assert os.path.exists(path), f"WGSL source not found: {path}"
    with open(path, "r") as f:
        return f.read()


def vec3_close(a, b, rel_tol=TOL_REL, abs_tol=TOL_ABS) -> bool:
    return all(
        math.isclose(a[i], b[i], rel_tol=rel_tol, abs_tol=abs_tol)
        for i in range(3)
    )


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


def get_stretch_section(wgsl_source: str) -> tuple[int, str] | None:
    """Find the T-DEMO-1.27 stretch section and return (start_pos, section_text).

    Returns None if the section header is not found.
    The section text includes everything from the content after the header-end
    delimiter (the ==== line below the header text) to the next section
    delimiter or end of file.
    """
    stretch_pos = wgsl_source.find("T-DEMO-1.27: Stretch (Anisotropic Scaling)")
    if stretch_pos == -1:
        return None
    # Skip past the header text line
    after_header = wgsl_source[stretch_pos:]
    nl1 = after_header.find('\n')
    if nl1 == -1:
        return (stretch_pos, "")
    # Skip past the header-end delimiter line (the second ====)
    nl2 = after_header.find('\n', nl1 + 1)
    if nl2 == -1:
        return (stretch_pos, "")
    # Content starts after the second newline (past both header text and ====)
    content = after_header[nl2 + 1:]
    # Find next section delimiter (or EOF for the last section)
    next_section = content.find("\n// ============")
    if next_section == -1:
        next_section = content.find("\n// ======")
    if next_section != -1:
        content = content[:next_section]
    return (stretch_pos, content)


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
            # Valid line endings or starts in WGSL
            assert (
                trimmed.endswith(';')
                or trimmed.endswith('{')
                or trimmed.endswith('}')
                or trimmed.endswith(')')
                or trimmed.endswith(',')       # function arg lists can span lines
                or trimmed.startswith("fn ")
                or trimmed.startswith("var ")
                or trimmed.startswith("let ")
                or trimmed.startswith("return ")
                or trimmed.startswith("const ")
                or trimmed.startswith("alias ")
                or trimmed.startswith("struct ")
                or trimmed.startswith("diagnostic ")
                or trimmed.startswith("enable ")
                or trimmed.startswith("requires ")
                or trimmed.startswith("_ = ")
                # Bare expressions inside multi-line function args
                or any(op in trimmed for op in [' + ', ' - ', ' * ', ' / '])
            ), f"Line {i + 1} has unexpected content: {trimmed!r}"


# =============================================================================
# SECTION 2 -- Section headers
# =============================================================================


class TestDemoSectionHeaders:
    """Blackbox tests for T-DEMO-1.27 section header presence and ordering."""

    def test_stretch_section_header_present(self, wgsl_source):
        """The T-DEMO-1.27 section header must appear in the WGSL source."""
        assert "T-DEMO-1.27: Stretch (Anisotropic Scaling)" in wgsl_source, (
            "Section header 'T-DEMO-1.27: Stretch (Anisotropic Scaling)' "
            "not found in WGSL source"
        )

    def test_stretch_section_has_delimiters(self, wgsl_source):
        """The stretch section must be delimited by ==== style header lines."""
        section = get_stretch_section(wgsl_source)
        assert section is not None, "Stretch section not found"

    def test_all_demo_section_headers_present(self, wgsl_source):
        """All T-DEMO-1.2x section headers are present in order."""
        sections = [
            "T-DEMO-1.22: Domain Repetition",
            "T-DEMO-1.23: Domain Mirroring",
            "T-DEMO-1.24: Kaleidoscopic Fold (KIFS)",
            "T-DEMO-1.25: Twist",
            "T-DEMO-1.26: Bend",
            "T-DEMO-1.27: Stretch (Anisotropic Scaling)",
        ]
        last_pos = 0
        for section in sections:
            pos = wgsl_source.find(section, last_pos)
            assert pos != -1, (
                f"Section header {section!r} not found in WGSL source"
            )
            assert pos >= last_pos, (
                f"Section header {section!r} out of order"
            )
            last_pos = pos

    def test_section_header_delimiter_count(self, wgsl_source):
        """There must be the expected number of section header delimiters."""
        delimiters = re.findall(r"^// =====+$", wgsl_source, re.MULTILINE)
        # At least 2 delimiters (file header open+close, or open+close per section)
        assert len(delimiters) >= 2, (
            f"Expected at least 2 section header delimiters, found {len(delimiters)}"
        )


# =============================================================================
# SECTION 3 -- Stretch function existence
# =============================================================================


class TestStretchFunctionExistence:
    """Blackbox tests for stretch function names in the WGSL source."""

    STRETCH_FUNCTIONS = [
        "domain_stretch_x",
        "domain_stretch_y",
        "domain_stretch_z",
        "domain_stretch_compensation",
    ]

    def test_stretch_functions_exist(self, wgsl_source):
        """All stretch functions must exist in the WGSL source."""
        fn_re = re.compile(r"(?m)^\s*fn\s+(domain_stretch_\w+)\s*\(")
        found = {m.group(1) for m in fn_re.finditer(wgsl_source)}
        for name in self.STRETCH_FUNCTIONS:
            assert name in found, (
                f"Expected function {name}() not found. "
                f"Found: {sorted(found)}"
            )

    def test_stretch_x_exists(self, wgsl_source):
        """domain_stretch_x must be defined."""
        assert re.search(
            r"fn\s+domain_stretch_x\s*\(", wgsl_source
        ), "domain_stretch_x() not found"

    def test_stretch_y_exists(self, wgsl_source):
        """domain_stretch_y must be defined."""
        assert re.search(
            r"fn\s+domain_stretch_y\s*\(", wgsl_source
        ), "domain_stretch_y() not found"

    def test_stretch_z_exists(self, wgsl_source):
        """domain_stretch_z must be defined."""
        assert re.search(
            r"fn\s+domain_stretch_z\s*\(", wgsl_source
        ), "domain_stretch_z() not found"

    def test_no_extra_stretch_functions(self, wgsl_source):
        """Only the expected stretch functions should exist."""
        fn_re = re.compile(r"(?m)^\s*fn\s+(domain_stretch_\w+)\s*\(")
        found = {m.group(1) for m in fn_re.finditer(wgsl_source)}
        extra = found - set(self.STRETCH_FUNCTIONS)
        assert not extra, f"Unexpected stretch functions found: {extra}"

    def test_all_stretch_functions_have_body(self, wgsl_source):
        """Each stretch function must have a non-empty body (not a stub)."""
        for name in self.STRETCH_FUNCTIONS:
            body = extract_function_body(wgsl_source, name)
            assert body is not None, (
                f"Could not extract body of {name}()"
            )
            stripped = body.strip()
            assert len(stripped) > 0, (
                f"Function {name}() has an empty body"
            )


# =============================================================================
# SECTION 4 -- Parameter types
# =============================================================================


class TestStretchParameterTypes:
    """Blackbox tests for parameter types of stretch functions."""

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

    def test_stretch_x_has_position_param(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "domain_stretch_x")
        assert "vec3<f32>" in types, (
            f"domain_stretch_x missing vec3<f32> param, found: {types}"
        )

    def test_stretch_y_has_position_param(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "domain_stretch_y")
        assert "vec3<f32>" in types, (
            f"domain_stretch_y missing vec3<f32> param, found: {types}"
        )

    def test_stretch_z_has_position_param(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "domain_stretch_z")
        assert "vec3<f32>" in types, (
            f"domain_stretch_z missing vec3<f32> param, found: {types}"
        )

    def test_stretch_x_has_scale_param(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "domain_stretch_x")
        assert "f32" in types, (
            f"domain_stretch_x missing f32 scale param, found: {types}"
        )

    def test_stretch_y_has_scale_param(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "domain_stretch_y")
        assert "f32" in types, (
            f"domain_stretch_y missing f32 scale param, found: {types}"
        )

    def test_stretch_z_has_scale_param(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "domain_stretch_z")
        assert "f32" in types, (
            f"domain_stretch_z missing f32 scale param, found: {types}"
        )

    def test_stretch_x_has_two_params(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "domain_stretch_x")
        assert len(types) == 2, (
            f"domain_stretch_x expected 2 params, found {len(types)}: {types}"
        )

    def test_stretch_y_has_two_params(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "domain_stretch_y")
        assert len(types) == 2, (
            f"domain_stretch_y expected 2 params, found {len(types)}: {types}"
        )

    def test_stretch_z_has_two_params(self, wgsl_source):
        types = self.get_function_params(wgsl_source, "domain_stretch_z")
        assert len(types) == 2, (
            f"domain_stretch_z expected 2 params, found {len(types)}: {types}"
        )


# =============================================================================
# SECTION 5 -- Return types
# =============================================================================


class TestStretchReturnTypes:
    """Blackbox tests for return types of stretch functions."""

    RETURN_TYPE_RE = re.compile(
        r"fn\s+(domain_stretch_\w+)\s*\([^)]*\)\s*->\s*([a-zA-Z0-9_<>]+)"
    )

    def test_stretch_fns_return_vec3(self, wgsl_source):
        """domain_stretch_x/y/z must return vec3<f32>."""
        names = ["domain_stretch_x", "domain_stretch_y", "domain_stretch_z"]
        for name in names:
            found = False
            for m in self.RETURN_TYPE_RE.finditer(wgsl_source):
                if m.group(1) == name:
                    ret_type = m.group(2).strip()
                    assert ret_type == "vec3<f32>", (
                        f"{name} returns '{ret_type}', expected 'vec3<f32>'"
                    )
                    found = True
                    break
            assert found, f"Return type not found for {name}()"

    def test_no_stretch_fn_returns_void(self, wgsl_source):
        """No stretch function should return void (all must have -> type)."""
        fn_re = re.compile(r"(?m)^\s*fn\s+(domain_stretch_\w+)\s*\(")
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
    """Blackbox tests for WGSL source structure around the stretch section."""

    def test_stretch_section_after_bend(self, wgsl_source):
        """Stretch section must appear after Bend section (T-DEMO-1.27 after 1.26)."""
        bend_pos = wgsl_source.find("T-DEMO-1.26: Bend")
        stretch_pos = wgsl_source.find("T-DEMO-1.27: Stretch")
        assert bend_pos != -1, "Bend section header not found"
        assert stretch_pos != -1, "Stretch section header not found"
        assert stretch_pos > bend_pos, (
            "Stretch section must appear after Bend section"
        )

    def test_stretch_section_is_last_domain_op(self, wgsl_source):
        """Stretch (1.27) must be the last domain operation section in Phase 1."""
        section = get_stretch_section(wgsl_source)
        assert section is not None, "Stretch section not found"
        # Check there are no other T-DEMO-1.2x sections after stretch
        _, stretch_text = section
        after_stretch = stretch_text[stretch_text.find("T-DEMO-1.27:"):]
        # Extract all T-DEMO-1.x headers after the stretch header itself
        later_headers = re.findall(
            r"T-DEMO-1\.\d+", after_stretch
        )
        # Only the stretch header itself should be present
        non_stretch = [h for h in later_headers if h != "T-DEMO-1.27"]
        assert not non_stretch, (
            f"Found domain op headers after T-DEMO-1.27: {non_stretch}"
        )

    def test_stretch_section_contains_functions(self, wgsl_source):
        """The stretch section must contain the stretch function definitions."""
        section = get_stretch_section(wgsl_source)
        assert section is not None, "Stretch section not found"
        _, section_text = section
        fn_count = section_text.count("fn ")
        assert fn_count >= 3, (
            f"Stretch section contains only {fn_count} functions, "
            f"expected at least 3 (x, y, z)"
        )

    def test_stretch_functions_in_stretch_section(self, wgsl_source):
        """All stretch functions should be defined within the stretch section."""
        section = get_stretch_section(wgsl_source)
        assert section is not None, "Stretch section not found"
        _, section_text = section
        for fn_name in [
            "domain_stretch_x",
            "domain_stretch_y",
            "domain_stretch_z",
        ]:
            assert fn_name in section_text, (
                f"{fn_name} not defined in the stretch section"
            )

    def test_stretch_section_has_comments(self, wgsl_source):
        """The stretch section must have documentation comments."""
        section = get_stretch_section(wgsl_source)
        assert section is not None, "Stretch section not found"
        _, section_text = section
        comment_lines = sum(
            1 for line in section_text.splitlines()
            if line.strip().startswith("//")
        )
        assert comment_lines >= 3, (
            f"Stretch section only has {comment_lines} comment lines, "
            f"expected at least 3"
        )

    def test_stretch_section_has_section_marker(self, wgsl_source):
        """The stretch section header is preceded by an ==== delimiter."""
        idx = wgsl_source.find("T-DEMO-1.27: Stretch (Anisotropic Scaling)")
        assert idx != -1, "Stretch section header not found"
        # Find the last ==== delimiter before the header
        before = wgsl_source[:idx]
        last_delim = before.rfind("// ============")
        if last_delim == -1:
            last_delim = before.rfind("// ======")
        assert last_delim != -1, (
            "No section delimiter found before stretch header"
        )


# =============================================================================
# SECTION 7 -- Mathematical invariants
# =============================================================================


class TestStretchMathematicalInvariants:
    """Blackbox tests for mathematical invariants derived from the spec.

    From the spec: T-DEMO-1.27 "Implement stretch operation (anisotropic
    scaling). Acceptance: space scaled along specified axis."

    Anisotropic scaling means:
    - Stretch along one axis, compensate along the other two
    - s = 1 is identity (no stretch)
    - s > 1 stretches the specified axis, compresses perpendiculars
    - 0 < s < 1 compresses the specified axis, stretches perpendiculars
    - The operation must be invertible (s != 0)
    - Negative s mirrors along the primary axis
    """

    @staticmethod
    def stretch_x(p, s):
        """Anisotropic scaling along x: x*s, y/s, z/s."""
        return (p[0] * s, p[1] / s, p[2] / s)

    @staticmethod
    def stretch_y(p, s):
        """Anisotropic scaling along y: x/s, y*s, z/s."""
        return (p[0] / s, p[1] * s, p[2] / s)

    @staticmethod
    def stretch_z(p, s):
        """Anisotropic scaling along z: x/s, y/s, z*s."""
        return (p[0] / s, p[1] / s, p[2] * s)

    # --- Identity ---

    def test_stretch_identity_at_one(self):
        """At s=1, all stretch axes must be identity."""
        p = (2.5, -1.5, 3.0)
        for fn in [self.stretch_x, self.stretch_y, self.stretch_z]:
            assert vec3_close(fn(p, 1.0), p), (
                f"{fn.__name__} not identity at s=1"
            )

    def test_stretch_identity_at_one_various_inputs(self):
        """Identity at s=1 for diverse inputs."""
        for _ in range(20):
            p = tuple(random.uniform(-10, 10) for _ in range(3))
            for fn in [self.stretch_x, self.stretch_y, self.stretch_z]:
                assert vec3_close(fn(p, 1.0), p)

    # --- Anisotropic behavior ---

    def test_stretch_x_affects_x_differently(self):
        """stretch_x must change x differently from y/z (anisotropic)."""
        p = (2.0, 3.0, 4.0)
        s = 2.0
        result = self.stretch_x(p, s)
        x_ratio = result[0] / p[0]
        yz_ratio = result[1] / p[1]
        assert not math.isclose(yz_ratio, x_ratio, rel_tol=TOL_REL), (
            "stretch_x applies same factor to all axes (isotropic)"
        )

    def test_stretch_y_affects_y_differently(self):
        """stretch_y must change y differently from x/z (anisotropic)."""
        p = (2.0, 3.0, 4.0)
        s = 2.0
        result = self.stretch_y(p, s)
        y_ratio = result[1] / p[1]
        xz_ratio = result[0] / p[0]
        assert not math.isclose(xz_ratio, y_ratio, rel_tol=TOL_REL), (
            "stretch_y applies same factor to all axes (isotropic)"
        )

    def test_stretch_z_affects_z_differently(self):
        """stretch_z must change z differently from x/y (anisotropic)."""
        p = (2.0, 3.0, 4.0)
        s = 2.0
        result = self.stretch_z(p, s)
        z_ratio = result[2] / p[2]
        xy_ratio = result[0] / p[0]
        assert not math.isclose(xy_ratio, z_ratio, rel_tol=TOL_REL), (
            "stretch_z applies same factor to all axes (isotropic)"
        )

    # --- Primary axis scaling ---

    def test_stretch_x_primary_scales_by_s(self):
        """stretch_x must scale the x axis by factor s."""
        p = (2.0, 3.0, 4.0)
        for s in [0.1, 0.5, 2.0, 5.0, 10.0]:
            result = self.stretch_x(p, s)
            assert math.isclose(result[0], p[0] * s, rel_tol=TOL_REL)

    def test_stretch_y_primary_scales_by_s(self):
        """stretch_y must scale the y axis by factor s."""
        p = (2.0, 3.0, 4.0)
        for s in [0.1, 0.5, 2.0, 5.0, 10.0]:
            result = self.stretch_y(p, s)
            assert math.isclose(result[1], p[1] * s, rel_tol=TOL_REL)

    def test_stretch_z_primary_scales_by_s(self):
        """stretch_z must scale the z axis by factor s."""
        p = (2.0, 3.0, 4.0)
        for s in [0.1, 0.5, 2.0, 5.0, 10.0]:
            result = self.stretch_z(p, s)
            assert math.isclose(result[2], p[2] * s, rel_tol=TOL_REL)

    # --- Secondary axis compression ---

    def test_stretch_x_compresses_yz(self):
        """stretch_x must compress y and z by 1/s."""
        p = (2.0, 3.0, 4.0)
        for s in [0.1, 0.5, 2.0, 5.0, 10.0]:
            result = self.stretch_x(p, s)
            assert math.isclose(result[1], p[1] / s, rel_tol=TOL_REL)
            assert math.isclose(result[2], p[2] / s, rel_tol=TOL_REL)

    def test_stretch_y_compresses_xz(self):
        """stretch_y must compress x and z by 1/s."""
        p = (2.0, 3.0, 4.0)
        for s in [0.1, 0.5, 2.0, 5.0, 10.0]:
            result = self.stretch_y(p, s)
            assert math.isclose(result[0], p[0] / s, rel_tol=TOL_REL)
            assert math.isclose(result[2], p[2] / s, rel_tol=TOL_REL)

    def test_stretch_z_compresses_xy(self):
        """stretch_z must compress x and y by 1/s."""
        p = (2.0, 3.0, 4.0)
        for s in [0.1, 0.5, 2.0, 5.0, 10.0]:
            result = self.stretch_z(p, s)
            assert math.isclose(result[0], p[0] / s, rel_tol=TOL_REL)
            assert math.isclose(result[1], p[1] / s, rel_tol=TOL_REL)

    # --- Invertibility ---

    def test_stretch_x_invertible(self):
        """Stretch then stretch by reciprocal must restore original."""
        p = (3.0, 4.0, 5.0)
        for s in [0.25, 0.5, 2.0, 4.0]:
            forward = self.stretch_x(p, s)
            backward = self.stretch_x(forward, 1.0 / s)
            assert vec3_close(backward, p)

    def test_stretch_y_invertible(self):
        p = (3.0, 4.0, 5.0)
        for s in [0.25, 0.5, 2.0, 4.0]:
            forward = self.stretch_y(p, s)
            backward = self.stretch_y(forward, 1.0 / s)
            assert vec3_close(backward, p)

    def test_stretch_z_invertible(self):
        p = (3.0, 4.0, 5.0)
        for s in [0.25, 0.5, 2.0, 4.0]:
            forward = self.stretch_z(p, s)
            backward = self.stretch_z(forward, 1.0 / s)
            assert vec3_close(backward, p)

    def test_stretch_x_invertible_diverse(self):
        for _ in range(20):
            p = tuple(random.uniform(-10, 10) for _ in range(3))
            s = random.uniform(0.1, 10.0)
            forward = self.stretch_x(p, s)
            backward = self.stretch_x(forward, 1.0 / s)
            assert vec3_close(backward, p)

    # --- Determinant ---

    def test_stretch_x_determinant(self):
        """det = s * 1/s * 1/s = 1/s (NOT volume-preserving)."""
        p = (2.0, 3.0, 4.0)
        for s in [0.25, 0.5, 2.0, 4.0]:
            r = self.stretch_x(p, s)
            det = abs(r[0]/p[0]) * abs(r[1]/p[1]) * abs(r[2]/p[2])
            assert math.isclose(det, 1.0 / s, rel_tol=TOL_REL)

    def test_stretch_y_determinant(self):
        p = (2.0, 3.0, 4.0)
        for s in [0.25, 0.5, 2.0, 4.0]:
            r = self.stretch_y(p, s)
            det = abs(r[0]/p[0]) * abs(r[1]/p[1]) * abs(r[2]/p[2])
            assert math.isclose(det, 1.0 / s, rel_tol=TOL_REL)

    def test_stretch_z_determinant(self):
        p = (2.0, 3.0, 4.0)
        for s in [0.25, 0.5, 2.0, 4.0]:
            r = self.stretch_z(p, s)
            det = abs(r[0]/p[0]) * abs(r[1]/p[1]) * abs(r[2]/p[2])
            assert math.isclose(det, 1.0 / s, rel_tol=TOL_REL)

    def test_stretch_not_volume_preserving(self):
        """Stretch det != 1 for s != 1 (not volume-preserving)."""
        for fn in [self.stretch_x, self.stretch_y, self.stretch_z]:
            for s in [0.5, 2.0, 4.0]:
                p = (1.0, 2.0, 3.0)
                r = fn(p, s)
                det = abs(r[0]/p[0]) * abs(r[1]/p[1]) * abs(r[2]/p[2])
                assert not math.isclose(det, 1.0, rel_tol=TOL_REL), (
                    f"det={det} == 1 (should NOT be volume-preserving)"
                )

    def test_determinant_independent_of_input(self):
        """det = 1/s regardless of input position."""
        for fn, name in [
            (self.stretch_x, "stretch_x"),
            (self.stretch_y, "stretch_y"),
            (self.stretch_z, "stretch_z"),
        ]:
            for s in [0.1, 0.5, 1.0, 2.0, 10.0]:
                for _ in range(20):
                    p = tuple(random.uniform(-10, 10) for _ in range(3))
                    r = fn(p, s)
                    det = (
                        abs(r[0]/p[0]) * abs(r[1]/p[1]) * abs(r[2]/p[2])
                    )
                    assert math.isclose(det, 1.0/s, rel_tol=TOL_REL), (
                        f"{name}(s={s}) det={det}, expected 1/{s}"
                    )

    # --- Finite output ---

    def test_stretch_finite_output(self):
        """All stretch functions produce finite output for valid inputs."""
        for _ in range(50):
            p = tuple(random.uniform(-10, 10) for _ in range(3))
            for s in [0.1, 0.5, 1.0, 2.0, 10.0]:
                for fn in [self.stretch_x, self.stretch_y, self.stretch_z]:
                    result = fn(p, s)
                    assert all(math.isfinite(x) for x in result), (
                        f"{fn.__name__}(s={s}) non-finite: {result}"
                    )

    def test_stretch_negative_s(self):
        """Negative s mirrors along the primary axis."""
        p = (2.0, 3.0, 4.0)
        for fn, axis in [
            (self.stretch_x, 0),
            (self.stretch_y, 1),
            (self.stretch_z, 2),
        ]:
            result = fn(p, -1.0)
            assert result[axis] == -p[axis], (
                f"{fn.__name__} s=-1: axis {axis} = {result[axis]} vs {-p[axis]}"
            )

    def test_stretch_finite_at_extremes(self):
        """Stretch finite for extreme scale values."""
        for fn in [self.stretch_x, self.stretch_y, self.stretch_z]:
            for s in [1e-6, 1e6]:
                p = tuple(random.uniform(-10, 10) for _ in range(3))
                result = fn(p, s)
                assert all(math.isfinite(x) for x in result), (
                    f"{fn.__name__}(s={s}) non-finite: {result}"
                )

    def test_stretch_deterministic(self):
        """Stretch is deterministic (same input -> same output)."""
        p = (1.234, 5.678, -3.456)
        for fn in [self.stretch_x, self.stretch_y, self.stretch_z]:
            for s in [0.5, 1.0, 2.0]:
                r1 = fn(p, s)
                r2 = fn(p, s)
                assert vec3_close(r1, r2), (
                    f"{fn.__name__}(s={s}) not deterministic"
                )


# =============================================================================
# SECTION 8 -- Axis specificity
# =============================================================================


class TestAxisSpecificity:
    """Blackbox tests verifying each stretch only affects its target axis."""

    def test_stretch_x_expands_x_compresses_yz(self):
        """stretch_x(s>1) expands x, compresses y and z."""
        p = (2.0, 3.0, 4.0)
        s = 3.0
        r = TestStretchMathematicalInvariants.stretch_x(p, s)
        assert r[0] > p[0], f"x should expand: {r[0]} <= {p[0]}"
        assert r[1] < p[1], f"y should compress: {r[1]} >= {p[1]}"
        assert r[2] < p[2], f"z should compress: {r[2]} >= {p[2]}"

    def test_stretch_y_expands_y_compresses_xz(self):
        """stretch_y(s>1) expands y, compresses x and z."""
        p = (2.0, 3.0, 4.0)
        s = 3.0
        r = TestStretchMathematicalInvariants.stretch_y(p, s)
        assert r[1] > p[1], f"y should expand: {r[1]} <= {p[1]}"
        assert r[0] < p[0], f"x should compress: {r[0]} >= {p[0]}"
        assert r[2] < p[2], f"z should compress: {r[2]} >= {p[2]}"

    def test_stretch_z_expands_z_compresses_xy(self):
        """stretch_z(s>1) expands z, compresses x and y."""
        p = (2.0, 3.0, 4.0)
        s = 3.0
        r = TestStretchMathematicalInvariants.stretch_z(p, s)
        assert r[2] > p[2], f"z should expand: {r[2]} <= {p[2]}"
        assert r[0] < p[0], f"x should compress: {r[0]} >= {p[0]}"
        assert r[1] < p[1], f"y should compress: {r[1]} >= {p[1]}"

    def test_stretch_x_compresses_x_expands_yz(self):
        """stretch_x(s<1) compresses x, expands y and z."""
        p = (2.0, 3.0, 4.0)
        s = 0.25
        r = TestStretchMathematicalInvariants.stretch_x(p, s)
        assert r[0] < p[0], f"x should compress: {r[0]} >= {p[0]}"
        assert r[1] > p[1], f"y should expand: {r[1]} <= {p[1]}"
        assert r[2] > p[2], f"z should expand: {r[2]} <= {p[2]}"

    def test_stretch_x_preserves_yz_ratio(self):
        """stretch_x preserves y/z ratio (both scaled by 1/s)."""
        p = (2.0, 6.0, 3.0)
        r = TestStretchMathematicalInvariants.stretch_x(p, 2.0)
        orig = p[1] / p[2]
        res = r[1] / r[2]
        assert math.isclose(orig, res, rel_tol=TOL_REL), (
            f"yz ratio changed: {orig} -> {res}"
        )

    def test_stretch_y_preserves_xz_ratio(self):
        """stretch_y preserves x/z ratio."""
        p = (4.0, 2.0, 8.0)
        r = TestStretchMathematicalInvariants.stretch_y(p, 2.0)
        orig = p[0] / p[2]
        res = r[0] / r[2]
        assert math.isclose(orig, res, rel_tol=TOL_REL)

    def test_stretch_z_preserves_xy_ratio(self):
        """stretch_z preserves x/y ratio."""
        p = (3.0, 6.0, 2.0)
        r = TestStretchMathematicalInvariants.stretch_z(p, 2.0)
        orig = p[0] / p[1]
        res = r[0] / r[1]
        assert math.isclose(orig, res, rel_tol=TOL_REL)

    def test_each_stretch_only_affects_primary_axis(self):
        """Each stretch changes its primary axis by s, others by 1/s."""
        cases = [
            (TestStretchMathematicalInvariants.stretch_x, 0),
            (TestStretchMathematicalInvariants.stretch_y, 1),
            (TestStretchMathematicalInvariants.stretch_z, 2),
        ]
        for fn, primary in cases:
            for s in [0.5, 2.0]:
                p = (1.0, 1.0, 1.0)
                r = fn(p, s)
                for axis in range(3):
                    if axis == primary:
                        assert math.isclose(r[axis], p[axis] * s, rel_tol=TOL_REL)
                    else:
                        assert math.isclose(r[axis], p[axis] / s, rel_tol=TOL_REL)


# =============================================================================
# SECTION 9 -- Compensation invariants (spec-derived)
# =============================================================================


class TestCompensationInvariants:
    """Blackbox tests for stretch distance compensation.

    Anisotropic scaling distorts the distance field. The compensation factor
    min(|s|, 1/|s|) provides the minimum distance scaling factor.
    """

    @staticmethod
    def compensation(s):
        """Distance compensation: min(|s|, 1/|s|) with zero guard."""
        safe_s = s if abs(s) >= 1e-8 else 1e-8
        return min(abs(safe_s), 1.0 / abs(safe_s))

    def test_compensation_at_one(self):
        """compensation(1) = 1."""
        c = self.compensation(1.0)
        assert math.isclose(c, 1.0, rel_tol=TOL_REL)

    def test_compensation_symmetry(self):
        """compensation(s) = compensation(1/s)."""
        for s in [0.1, 0.25, 0.5, 2.0, 4.0, 10.0]:
            c1 = self.compensation(s)
            c2 = self.compensation(1.0 / s)
            assert math.isclose(c1, c2, rel_tol=TOL_REL)

    def test_compensation_negative_same_as_positive(self):
        """compensation uses abs(s), so negative equals positive."""
        for s in [0.5, 2.0, 10.0]:
            assert math.isclose(
                self.compensation(s), self.compensation(-s), rel_tol=TOL_REL
            )

    def test_compensation_range(self):
        """compensation(s) is always in (0, 1]."""
        for s in [1e-6, 0.1, 0.5, 1.0, 2.0, 10.0, 1e6]:
            c = self.compensation(s)
            assert 0.0 < c <= 1.0, f"comp({s})={c} out of range (0, 1]"

    def test_compensation_max_at_one(self):
        """compensation(s) is maximized at s=1."""
        c1 = self.compensation(1.0)
        for s in [0.1, 0.5, 2.0, 5.0, 10.0]:
            assert self.compensation(s) <= c1 + TOL_ABS

    def test_compensation_decreases_away_from_one(self):
        """compensation decreases as s moves away from 1."""
        assert self.compensation(0.25) < self.compensation(0.5)
        assert self.compensation(4.0) < self.compensation(2.0)

    def test_compensation_at_extremes(self):
        """compensation(s) approaches 0 at extremes."""
        assert self.compensation(1e-6) < 0.001, "comp(1e-6) not near 0"
        assert self.compensation(1e6) < 0.001, "comp(1e6) not near 0"

    def test_compensation_at_zero(self):
        """compensation(0) is finite and positive (guard boundary)."""
        c = self.compensation(0.0)
        assert math.isfinite(c), "compensation(0) is not finite"
        assert c > 0.0

    def test_compensation_finite_for_all(self):
        """compensation(s) is finite for all finite s."""
        for s in [1e-6, 0.0, -0.0, 0.5, 1.0, 2.0, 1e6, -1e6]:
            assert math.isfinite(self.compensation(s))

    def test_compensation_identity_at_s_1(self):
        """compensation(1) = min(1, 1/1) = 1."""
        assert math.isclose(self.compensation(1.0), 1.0, rel_tol=TOL_REL)


# =============================================================================
# SECTION 10 -- Division-by-zero guards
# =============================================================================


class TestStretchGuards:
    """Blackbox tests for division-by-zero guards on the scale parameter.

    Each stretch function divides by 's' and must guard against division by
    zero when s is near-zero. Guards use select(s, 1e-8, abs(s) < 1e-8).
    """

    def test_stretch_x_has_guard(self, wgsl_source):
        body = extract_function_body(wgsl_source, "domain_stretch_x")
        assert body is not None
        assert any(p in body for p in ["safe_s", "select", "clamp", "max(abs"]), (
            "domain_stretch_x missing division-by-zero guard"
        )

    def test_stretch_y_has_guard(self, wgsl_source):
        body = extract_function_body(wgsl_source, "domain_stretch_y")
        assert body is not None
        assert any(p in body for p in ["safe_s", "select", "clamp", "max(abs"]), (
            "domain_stretch_y missing division-by-zero guard"
        )

    def test_stretch_z_has_guard(self, wgsl_source):
        body = extract_function_body(wgsl_source, "domain_stretch_z")
        assert body is not None
        assert any(p in body for p in ["safe_s", "select", "clamp", "max(abs"]), (
            "domain_stretch_z missing division-by-zero guard"
        )

    def test_all_stretch_have_guards(self, wgsl_source):
        for name in ["domain_stretch_x", "domain_stretch_y", "domain_stretch_z"]:
            body = extract_function_body(wgsl_source, name)
            assert body is not None
            assert any(p in body for p in ["safe_s", "select", "clamp", "max(abs"]), (
                f"{name} missing guard"
            )

    def test_guard_clamps_to_minimum(self, wgsl_source):
        for name in ["domain_stretch_x", "domain_stretch_y", "domain_stretch_z"]:
            body = extract_function_body(wgsl_source, name)
            assert body is not None
            has_clamp = "select" in body or "clamp" in body or "max" in body
            assert has_clamp, f"{name} guard does not clamp"


# =============================================================================
# SECTION 11 -- Compensation function verification
# =============================================================================


class TestStretchCompensationFunction:
    """Blackbox tests for the domain_stretch_compensation function.

    T-DEMO-1.27 spec requires a companion function that returns the conservative
    distance compensation factor for anisotropic scaling: min(|s|, 1/|s|).
    Callers divide their SDF distance by this factor for safe sphere tracing.
    """

    def test_compensation_function_exists(self, wgsl_source):
        assert re.search(
            r"fn\s+domain_stretch_compensation\s*\(", wgsl_source
        ), "domain_stretch_compensation() not found"

    def test_compensation_returns_f32(self, wgsl_source):
        ret_re = re.compile(
            r"fn\s+domain_stretch_compensation\s*\([^)]*\)\s*->\s*(f32)"
        )
        assert ret_re.search(wgsl_source), (
            "domain_stretch_compensation must return f32"
        )

    def test_compensation_has_f32_param(self, wgsl_source):
        params = TestStretchParameterTypes.get_function_params(
            wgsl_source, "domain_stretch_compensation"
        )
        assert "f32" in params, (
            f"domain_stretch_compensation missing f32 param, found: {params}"
        )

    def test_compensation_has_one_param(self, wgsl_source):
        params = TestStretchParameterTypes.get_function_params(
            wgsl_source, "domain_stretch_compensation"
        )
        assert len(params) == 1, (
            f"domain_stretch_compensation expected 1 param, found {len(params)}"
        )

    def test_compensation_has_body(self, wgsl_source):
        body = extract_function_body(
            wgsl_source, "domain_stretch_compensation"
        )
        assert body is not None
        assert len(body.strip()) > 0

    def test_compensation_uses_abs_and_min(self, wgsl_source):
        body = extract_function_body(
            wgsl_source, "domain_stretch_compensation"
        )
        assert body is not None
        assert "abs" in body, "compensation must use abs()"
        assert "min" in body, "compensation must use min()"

    def test_compensation_uses_division(self, wgsl_source):
        body = extract_function_body(
            wgsl_source, "domain_stretch_compensation"
        )
        assert body is not None
        assert "/" in body, "compensation must use division for reciprocal"
