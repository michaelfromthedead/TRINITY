#!/usr/bin/env python3
"""
Build-time DSL Compilation Script (T-DEMO-5.5).

Compiles Python scene definitions to WGSL shaders for embedding in Rust.

This script is invoked by build.rs during cargo build to:
  1. Import a Python scene module
  2. Extract the SCENE variable (FullSceneNode)
  3. Compile to WGSL using the demoscene codegen
  4. Validate the generated WGSL
  5. Output to target/generated/

Exit codes:
    0 - Success
    1 - Missing scene file
    2 - Import error (Python module issues)
    3 - Missing SCENE variable
    4 - Invalid scene type
    5 - Compilation error
    6 - WGSL validation error
    7 - Output write error
    8 - Invalid arguments

Usage:
    uv run scripts/compile_demo.py scenes/demo.py target/generated/demo.wgsl
    uv run scripts/compile_demo.py --validate scenes/demo.py
    uv run scripts/compile_demo.py --list-primitives scenes/demo.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional


# Exit codes
EXIT_SUCCESS = 0
EXIT_MISSING_SCENE = 1
EXIT_IMPORT_ERROR = 2
EXIT_MISSING_SCENE_VAR = 3
EXIT_INVALID_SCENE_TYPE = 4
EXIT_COMPILATION_ERROR = 5
EXIT_WGSL_VALIDATION_ERROR = 6
EXIT_OUTPUT_WRITE_ERROR = 7
EXIT_INVALID_ARGUMENTS = 8


def setup_python_path() -> None:
    """Add project root to Python path for imports."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Add project root to path if not already present
    str_root = str(project_root)
    if str_root not in sys.path:
        sys.path.insert(0, str_root)


def load_scene_module(scene_path: Path) -> Optional[object]:
    """Load a Python scene module and return its module object.

    Args:
        scene_path: Path to the .py scene file.

    Returns:
        Loaded module object, or None on failure.

    Side effects:
        Sets exit code on failure.
    """
    if not scene_path.exists():
        print(f"Error: Scene file not found: {scene_path}", file=sys.stderr)
        return None

    if not scene_path.suffix == ".py":
        print(f"Error: Scene file must be a .py file: {scene_path}", file=sys.stderr)
        return None

    # Load module dynamically
    module_name = scene_path.stem
    spec = importlib.util.spec_from_file_location(module_name, scene_path)

    if spec is None or spec.loader is None:
        print(f"Error: Could not load module spec for: {scene_path}", file=sys.stderr)
        return None

    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error: Failed to import scene module: {e}", file=sys.stderr)
        return None


def extract_scene(module: object):
    """Extract the SCENE variable from a loaded module.

    Args:
        module: Loaded Python module.

    Returns:
        The SCENE object, or None if not found or invalid type.
    """
    if not hasattr(module, "SCENE"):
        print("Error: Scene module must define a SCENE variable", file=sys.stderr)
        return None

    scene = module.SCENE

    # Import here to avoid circular imports at module level
    from engine.rendering.demoscene.ast_nodes import FullSceneNode

    if not isinstance(scene, FullSceneNode):
        print(
            f"Error: SCENE must be a FullSceneNode, got {type(scene).__name__}",
            file=sys.stderr
        )
        return None

    return scene


def compile_scene_to_wgsl(scene) -> Optional[str]:
    """Compile a FullSceneNode to WGSL.

    Args:
        scene: FullSceneNode to compile.

    Returns:
        WGSL source code string, or None on failure.
    """
    try:
        from engine.rendering.demoscene.scene_codegen import generate_compute_shader

        wgsl = generate_compute_shader(
            scene,
            include_uniforms=True,
            include_bindings=True,
        )

        return wgsl
    except Exception as e:
        print(f"Error: Compilation failed: {e}", file=sys.stderr)
        return None


def validate_wgsl(wgsl: str) -> bool:
    """Validate WGSL source code for basic correctness.

    Performs lightweight validation:
      - Checks for required entry point
      - Checks for balanced braces
      - Checks for scene_sdf function
      - Checks for scene_material function

    Args:
        wgsl: WGSL source code string.

    Returns:
        True if valid, False otherwise.
    """
    errors = []

    # Check for compute entry point
    if "@compute" not in wgsl:
        errors.append("Missing @compute entry point")

    if "fn main(" not in wgsl:
        errors.append("Missing main() function")

    # Check for scene_sdf function
    if "fn scene_sdf(" not in wgsl:
        errors.append("Missing scene_sdf() function")

    # Check for scene_material function
    if "fn scene_material(" not in wgsl:
        errors.append("Missing scene_material() function")

    # Check balanced braces
    open_braces = wgsl.count("{")
    close_braces = wgsl.count("}")
    if open_braces != close_braces:
        errors.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")

    # Check balanced parentheses
    open_parens = wgsl.count("(")
    close_parens = wgsl.count(")")
    if open_parens != close_parens:
        errors.append(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")

    if errors:
        print("WGSL validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return False

    return True


def write_output(wgsl: str, output_path: Path) -> bool:
    """Write WGSL output to file.

    Creates parent directories if needed.

    Args:
        wgsl: WGSL source code string.
        output_path: Output file path.

    Returns:
        True on success, False on failure.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(wgsl, encoding="utf-8")
        return True
    except Exception as e:
        print(f"Error: Failed to write output: {e}", file=sys.stderr)
        return False


def list_primitives(scene) -> None:
    """Print a list of primitives in the scene."""
    print(f"Scene: {scene.name or '(unnamed)'}")
    print(f"Primitives ({len(scene.scene_graph.primitives)}):")

    for i, prim in enumerate(scene.scene_graph.primitives):
        print(f"  [{i}] {prim.label()}")

    print(f"Domain Pipeline ({len(scene.scene_graph.pipeline)}):")
    for i, op in enumerate(scene.scene_graph.pipeline):
        print(f"  [{i}] {op.label()}")

    print(f"Materials ({len(scene.materials)}):")
    for mat in scene.materials:
        print(f"  [{mat.material_id}] {mat.label()}")

    print(f"Lights ({len(scene.lights)}):")
    for i, light in enumerate(scene.lights):
        print(f"  [{i}] {light.label()}")


def get_scene_info(scene) -> dict:
    """Get scene information as a dictionary."""
    return {
        "name": scene.name or "(unnamed)",
        "primitives": len(scene.scene_graph.primitives),
        "domain_ops": len(scene.scene_graph.pipeline),
        "materials": len(scene.materials),
        "lights": len(scene.lights),
        "resolution": f"{scene.settings.width}x{scene.settings.height}",
        "max_steps": scene.settings.max_steps,
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compile Python scene definitions to WGSL shaders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "scene_file",
        type=Path,
        help="Path to the Python scene file (e.g., scenes/demo.py)",
    )

    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        help="Path to write the output WGSL file",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate the scene, don't write output",
    )

    parser.add_argument(
        "--list-primitives",
        action="store_true",
        help="List primitives and scene info",
    )

    parser.add_argument(
        "--skip-wgsl-validation",
        action="store_true",
        help="Skip WGSL validation step",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.validate and not args.list_primitives and args.output_file is None:
        print("Error: Output file required (or use --validate/--list-primitives)", file=sys.stderr)
        return EXIT_INVALID_ARGUMENTS

    # Setup Python path
    setup_python_path()

    # Load scene module
    if args.verbose:
        print(f"Loading scene: {args.scene_file}")

    module = load_scene_module(args.scene_file)
    if module is None:
        return EXIT_MISSING_SCENE if not args.scene_file.exists() else EXIT_IMPORT_ERROR

    # Extract scene
    scene = extract_scene(module)
    if scene is None:
        return EXIT_MISSING_SCENE_VAR

    if args.verbose:
        info = get_scene_info(scene)
        print(f"Scene loaded: {info['name']}")
        print(f"  Primitives: {info['primitives']}")
        print(f"  Domain ops: {info['domain_ops']}")
        print(f"  Materials: {info['materials']}")
        print(f"  Lights: {info['lights']}")

    # List primitives mode
    if args.list_primitives:
        list_primitives(scene)
        return EXIT_SUCCESS

    # Compile to WGSL
    if args.verbose:
        print("Compiling to WGSL...")

    wgsl = compile_scene_to_wgsl(scene)
    if wgsl is None:
        return EXIT_COMPILATION_ERROR

    if args.verbose:
        print(f"Generated {len(wgsl)} bytes of WGSL")

    # Validate WGSL
    if not args.skip_wgsl_validation:
        if args.verbose:
            print("Validating WGSL...")

        if not validate_wgsl(wgsl):
            return EXIT_WGSL_VALIDATION_ERROR

        if args.verbose:
            print("WGSL validation passed")

    # Validate-only mode
    if args.validate:
        if not args.quiet:
            print(f"Scene '{scene.name}' validated successfully")
        return EXIT_SUCCESS

    # Write output
    if args.verbose:
        print(f"Writing output: {args.output_file}")

    if not write_output(wgsl, args.output_file):
        return EXIT_OUTPUT_WRITE_ERROR

    if not args.quiet:
        print(f"Compiled {args.scene_file.name} -> {args.output_file}")

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
