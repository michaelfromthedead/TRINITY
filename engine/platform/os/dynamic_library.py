"""
Dynamic library loading and symbol resolution.
"""
import ctypes
import ctypes.util
from typing import Optional, Any, Callable
from dataclasses import dataclass


@dataclass(slots=True)
class LibraryHandle:
    """Handle to a loaded dynamic library."""
    path: str
    handle: ctypes.CDLL
    ref_count: int = 1


class DynamicLibrary:
    """Dynamic library loader with symbol resolution."""

    def __init__(self):
        self._libraries: dict[str, LibraryHandle] = {}

    def load(self, name: str, mode: int = ctypes.RTLD_GLOBAL) -> Optional[str]:
        """
        Load a dynamic library by name or path.

        Args:
            name: Library name (e.g., 'c') or full path
            mode: Loading mode (RTLD_GLOBAL, RTLD_LOCAL, etc.)

        Returns:
            Library identifier for future operations, or None on failure
        """
        try:
            # Check if already loaded
            if name in self._libraries:
                self._libraries[name].ref_count += 1
                return name

            # Try to find library
            lib_path = name
            if not name.startswith('/'):
                # Not an absolute path, try to find it
                found = ctypes.util.find_library(name)
                if found:
                    lib_path = found

            # Load the library
            handle = ctypes.CDLL(lib_path, mode=mode)

            self._libraries[name] = LibraryHandle(
                path=lib_path,
                handle=handle,
                ref_count=1
            )

            return name
        except Exception as e:
            return None

    def unload(self, lib_id: str) -> bool:
        """
        Unload a dynamic library (decrements reference count).

        Args:
            lib_id: Library identifier from load()

        Returns:
            True if unloaded, False if still referenced or error
        """
        if lib_id not in self._libraries:
            return False

        lib = self._libraries[lib_id]
        lib.ref_count -= 1

        if lib.ref_count <= 0:
            # Actually unload
            # Note: ctypes doesn't provide explicit unload
            # The library will be unloaded when the CDLL object is garbage collected
            del self._libraries[lib_id]
            return True

        return False

    def get_symbol(self, lib_id: str, symbol_name: str) -> Optional[Any]:
        """
        Get address of a symbol in the library.

        Args:
            lib_id: Library identifier
            symbol_name: Symbol name to look up

        Returns:
            Symbol object or None if not found
        """
        if lib_id not in self._libraries:
            return None

        try:
            lib = self._libraries[lib_id]
            symbol = getattr(lib.handle, symbol_name)
            return symbol
        except AttributeError:
            return None

    def get_function(self, lib_id: str, func_name: str,
                     argtypes: Optional[list] = None,
                     restype: Optional[Any] = None) -> Optional[Callable]:
        """
        Get a function from the library with type annotations.

        Args:
            lib_id: Library identifier
            func_name: Function name
            argtypes: List of argument types (ctypes types)
            restype: Return type (ctypes type)

        Returns:
            Callable function or None if not found
        """
        symbol = self.get_symbol(lib_id, func_name)
        if symbol is None:
            return None

        if argtypes is not None:
            symbol.argtypes = argtypes
        if restype is not None:
            symbol.restype = restype

        return symbol

    def has_symbol(self, lib_id: str, symbol_name: str) -> bool:
        """
        Check if library has a symbol.

        Args:
            lib_id: Library identifier
            symbol_name: Symbol name to check

        Returns:
            True if symbol exists
        """
        return self.get_symbol(lib_id, symbol_name) is not None

    def get_library_path(self, lib_id: str) -> Optional[str]:
        """
        Get full path to loaded library.

        Args:
            lib_id: Library identifier

        Returns:
            Full path or None if not loaded
        """
        if lib_id not in self._libraries:
            return None
        return self._libraries[lib_id].path

    def is_loaded(self, lib_id: str) -> bool:
        """
        Check if library is currently loaded.

        Args:
            lib_id: Library identifier

        Returns:
            True if loaded
        """
        return lib_id in self._libraries

    def get_ref_count(self, lib_id: str) -> int:
        """
        Get reference count for library.

        Args:
            lib_id: Library identifier

        Returns:
            Reference count or 0 if not loaded
        """
        if lib_id not in self._libraries:
            return 0
        return self._libraries[lib_id].ref_count
