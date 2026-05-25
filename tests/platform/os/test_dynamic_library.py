"""
Tests for dynamic library loading.
"""
import ctypes
import sys
import pytest
from engine.platform.os.dynamic_library import DynamicLibrary


@pytest.fixture
def dl():
    """Create DynamicLibrary instance."""
    return DynamicLibrary()


def test_load_libc(dl):
    """Test loading standard C library."""
    # Try to load libc
    lib_id = dl.load('c')
    assert lib_id is not None
    assert dl.is_loaded(lib_id)

    # Verify we can actually resolve a symbol from libc
    strlen_symbol = dl.get_symbol(lib_id, 'strlen')
    assert strlen_symbol is not None


def test_load_nonexistent(dl):
    """Test loading nonexistent library."""
    lib_id = dl.load('nonexistent_library_xyz')
    assert lib_id is None


def test_get_symbol(dl):
    """Test getting symbol from library."""
    lib_id = dl.load('c')
    assert lib_id is not None

    # Get strlen symbol
    symbol = dl.get_symbol(lib_id, 'strlen')
    assert symbol is not None


def test_has_symbol(dl):
    """Test checking if symbol exists."""
    lib_id = dl.load('c')
    assert lib_id is not None

    assert dl.has_symbol(lib_id, 'strlen')
    assert not dl.has_symbol(lib_id, 'nonexistent_symbol_xyz')


def test_get_function(dl):
    """Test getting typed function from library."""
    lib_id = dl.load('c')
    assert lib_id is not None

    # Get strlen function
    strlen = dl.get_function(
        lib_id,
        'strlen',
        argtypes=[ctypes.c_char_p],
        restype=ctypes.c_size_t
    )

    assert strlen is not None

    # Test calling it
    result = strlen(b"hello")
    assert result == 5


def test_get_library_path(dl):
    """Test getting library path."""
    lib_id = dl.load('c')
    assert lib_id is not None

    path = dl.get_library_path(lib_id)
    assert path is not None
    assert len(path) > 0


def test_reference_counting(dl):
    """Test reference counting."""
    lib_id = dl.load('c')
    assert lib_id is not None
    assert dl.get_ref_count(lib_id) == 1

    # Load again
    lib_id2 = dl.load('c')
    assert lib_id2 == lib_id
    assert dl.get_ref_count(lib_id) == 2

    # Unload once
    result = dl.unload(lib_id)
    assert not result  # Still referenced
    assert dl.is_loaded(lib_id)
    assert dl.get_ref_count(lib_id) == 1

    # Unload again
    result = dl.unload(lib_id)
    assert result  # Actually unloaded
    assert not dl.is_loaded(lib_id)
    assert dl.get_ref_count(lib_id) == 0


def test_unload_invalid(dl):
    """Test unloading invalid library."""
    result = dl.unload('nonexistent')
    assert not result


def test_get_symbol_invalid_library(dl):
    """Test getting symbol from invalid library."""
    symbol = dl.get_symbol('nonexistent', 'strlen')
    assert symbol is None


def test_get_function_invalid_symbol(dl):
    """Test getting nonexistent function."""
    lib_id = dl.load('c')
    assert lib_id is not None

    func = dl.get_function(lib_id, 'nonexistent_function_xyz')
    assert func is None


def test_multiple_libraries(dl):
    """Test loading multiple libraries."""
    lib1 = dl.load('c')
    lib2 = dl.load('m')  # libm (math library)

    assert lib1 is not None
    assert lib2 is not None
    assert lib1 != lib2

    assert dl.is_loaded(lib1)
    assert dl.is_loaded(lib2)


def test_load_modes(dl):
    """Test different loading modes."""
    # RTLD_GLOBAL (default)
    lib_id = dl.load('c', mode=ctypes.RTLD_GLOBAL)
    assert lib_id is not None

    dl.unload(lib_id)

    # RTLD_LOCAL
    lib_id = dl.load('c', mode=ctypes.RTLD_LOCAL)
    assert lib_id is not None


def test_call_libc_functions(dl):
    """Test calling various libc functions."""
    lib_id = dl.load('c')
    assert lib_id is not None

    # Test abs
    abs_func = dl.get_function(
        lib_id,
        'abs',
        argtypes=[ctypes.c_int],
        restype=ctypes.c_int
    )
    assert abs_func is not None
    assert abs_func(-42) == 42

    # Test strlen
    strlen = dl.get_function(
        lib_id,
        'strlen',
        argtypes=[ctypes.c_char_p],
        restype=ctypes.c_size_t
    )
    assert strlen is not None
    assert strlen(b"test") == 4
