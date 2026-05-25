"""Test configuration for XR UI tests.

Handles isolation of imports to avoid loading the entire engine.xr package
which may have unresolved dependencies during development.
"""

import sys
from pathlib import Path

# Add the engine root to sys.path but prevent the parent xr package
# from being imported by manipulating the import system
engine_root = Path(__file__).parents[3]

# Ensure engine root is in path
if str(engine_root) not in sys.path:
    sys.path.insert(0, str(engine_root))

# Pre-create a stub module for engine.xr to prevent full package loading
# This allows us to import engine.xr.ui.* directly
import types

# Create stub modules to prevent cascade imports
if 'engine' not in sys.modules:
    sys.modules['engine'] = types.ModuleType('engine')
    sys.modules['engine'].__path__ = [str(engine_root / 'engine')]

if 'engine.xr' not in sys.modules:
    xr_module = types.ModuleType('engine.xr')
    xr_module.__path__ = [str(engine_root / 'engine' / 'xr')]
    sys.modules['engine.xr'] = xr_module
