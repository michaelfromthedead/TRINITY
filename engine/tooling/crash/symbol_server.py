"""
Symbol server for debug symbols.

Provides infrastructure for managing and serving debug symbols
for crash report symbolication.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class SymbolInfo:
    """Information about a debug symbol."""

    address: int
    name: str
    module: str = ""
    filename: str = ""
    line_number: int = 0
    offset: int = 0
    size: int = 0
    type: str = "function"  # function, variable, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "address": hex(self.address),
            "name": self.name,
            "module": self.module,
            "filename": self.filename,
            "line_number": self.line_number,
            "offset": self.offset,
            "size": self.size,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolInfo":
        """Create from dictionary."""
        address = data.get("address", 0)
        if isinstance(address, str):
            address = int(address, 16)

        return cls(
            address=address,
            name=data.get("name", ""),
            module=data.get("module", ""),
            filename=data.get("filename", ""),
            line_number=data.get("line_number", 0),
            offset=data.get("offset", 0),
            size=data.get("size", 0),
            type=data.get("type", "function"),
        )

    def format(self) -> str:
        """Format as human-readable string."""
        result = self.name
        if self.filename and self.line_number:
            result += f" at {self.filename}:{self.line_number}"
        elif self.module:
            result += f" in {self.module}"
        if self.offset:
            result += f"+{hex(self.offset)}"
        return result


@dataclass
class ModuleInfo:
    """Information about a loaded module."""

    name: str
    path: str
    base_address: int
    size: int
    build_id: str = ""
    checksum: str = ""
    debug_file: str = ""
    debug_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "base_address": hex(self.base_address),
            "size": self.size,
            "build_id": self.build_id,
            "checksum": self.checksum,
            "debug_file": self.debug_file,
            "debug_id": self.debug_id,
        }


class SymbolCache:
    """
    Cache for symbol lookups.

    Provides efficient caching of symbol information
    to avoid repeated lookups.
    """

    def __init__(self, max_size: int = 10000, ttl: float = 3600.0):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[SymbolInfo, float]] = {}

    def _make_key(self, module: str, address: int) -> str:
        """Create cache key."""
        return f"{module}:{hex(address)}"

    def get(self, module: str, address: int) -> Optional[SymbolInfo]:
        """Get cached symbol info."""
        key = self._make_key(module, address)
        if key in self._cache:
            info, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                return info
            else:
                del self._cache[key]
        return None

    def put(self, module: str, address: int, info: SymbolInfo) -> None:
        """Cache symbol info."""
        # Evict old entries if needed
        if len(self._cache) >= self.max_size:
            self._evict()

        key = self._make_key(module, address)
        self._cache[key] = (info, time.time())

    def _evict(self) -> None:
        """Evict oldest entries."""
        if not self._cache:
            return

        # Remove oldest 10%
        entries = sorted(self._cache.items(), key=lambda x: x[1][1])
        num_to_remove = max(1, len(entries) // 10)

        for key, _ in entries[:num_to_remove]:
            del self._cache[key]

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
        }


class SymbolServer:
    """
    Symbol server for managing and serving debug symbols.

    Features:
    - Symbol file management
    - Address lookup
    - Symbol caching
    - Multiple symbol sources
    """

    def __init__(
        self,
        symbol_paths: Optional[List[str]] = None,
        cache_path: Optional[str] = None,
        enable_cache: bool = True,
    ):
        self.symbol_paths = [Path(p) for p in (symbol_paths or [])]
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = SymbolCache() if enable_cache else None
        self._symbol_files: Dict[str, Dict[int, SymbolInfo]] = {}
        self._modules: Dict[str, ModuleInfo] = {}

    def add_symbol_path(self, path: str) -> None:
        """Add a symbol search path."""
        self.symbol_paths.append(Path(path))

    def register_module(self, module: ModuleInfo) -> None:
        """Register a loaded module."""
        self._modules[module.name] = module

    def load_symbol_file(self, path: str) -> bool:
        """
        Load a symbol file.

        Supports various formats:
        - JSON symbol maps
        - Simple text format (address name)

        Args:
            path: Path to symbol file

        Returns:
            True if loaded successfully
        """
        path = Path(path)
        if not path.exists():
            return False

        try:
            module_name = path.stem

            if path.suffix == ".json":
                return self._load_json_symbols(path, module_name)
            elif path.suffix in (".sym", ".map", ".txt"):
                return self._load_text_symbols(path, module_name)
            else:
                # Try JSON first, then text
                try:
                    return self._load_json_symbols(path, module_name)
                except Exception:
                    return self._load_text_symbols(path, module_name)

        except Exception as e:
            print(f"Error loading symbol file {path}: {e}")
            return False

    def _load_json_symbols(self, path: Path, module_name: str) -> bool:
        """Load symbols from JSON file."""
        with open(path) as f:
            data = json.load(f)

        symbols = {}
        for sym_data in data.get("symbols", []):
            info = SymbolInfo.from_dict(sym_data)
            info.module = module_name
            symbols[info.address] = info

        self._symbol_files[module_name] = symbols
        return True

    def _load_text_symbols(self, path: Path, module_name: str) -> bool:
        """Load symbols from text file."""
        symbols = {}

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse: address name [filename:line]
                parts = line.split(None, 2)
                if len(parts) < 2:
                    continue

                try:
                    address = int(parts[0], 16)
                except ValueError:
                    continue

                name = parts[1]
                filename = ""
                line_num = 0

                if len(parts) > 2:
                    # Try to parse filename:line
                    match = re.match(r"(\S+):(\d+)", parts[2])
                    if match:
                        filename = match.group(1)
                        line_num = int(match.group(2))

                symbols[address] = SymbolInfo(
                    address=address,
                    name=name,
                    module=module_name,
                    filename=filename,
                    line_number=line_num,
                )

        self._symbol_files[module_name] = symbols
        return True

    def resolve(
        self,
        address: int,
        module: Optional[str] = None,
    ) -> Optional[SymbolInfo]:
        """
        Resolve an address to symbol information.

        Args:
            address: Address to resolve
            module: Module name (optional, improves lookup)

        Returns:
            Symbol info or None if not found
        """
        # Check cache
        if self._cache and module:
            cached = self._cache.get(module, address)
            if cached:
                return cached

        result = None

        if module and module in self._symbol_files:
            result = self._find_symbol_in_module(address, module)
        else:
            # Search all modules
            for mod_name in self._symbol_files:
                result = self._find_symbol_in_module(address, mod_name)
                if result:
                    break

        # Try to load symbols from disk if not found
        if not result and module:
            self._try_load_module_symbols(module)
            result = self._find_symbol_in_module(address, module)

        # Cache result
        if result and self._cache and module:
            self._cache.put(module, address, result)

        return result

    def _find_symbol_in_module(
        self,
        address: int,
        module: str,
    ) -> Optional[SymbolInfo]:
        """Find symbol in a specific module."""
        symbols = self._symbol_files.get(module, {})

        # Exact match
        if address in symbols:
            return symbols[address]

        # Find nearest lower address
        best_match = None
        best_address = 0

        for sym_address, info in symbols.items():
            if sym_address <= address and sym_address > best_address:
                best_address = sym_address
                best_match = info

        if best_match:
            # Create copy with offset
            return SymbolInfo(
                address=best_match.address,
                name=best_match.name,
                module=best_match.module,
                filename=best_match.filename,
                line_number=best_match.line_number,
                offset=address - best_match.address,
                size=best_match.size,
                type=best_match.type,
            )

        return None

    def _try_load_module_symbols(self, module: str) -> bool:
        """Try to load symbols for a module from symbol paths."""
        extensions = [".sym", ".json", ".map", ".pdb"]

        for sym_path in self.symbol_paths:
            for ext in extensions:
                file_path = sym_path / f"{module}{ext}"
                if file_path.exists():
                    return self.load_symbol_file(str(file_path))

        return False

    def symbolicate_stack(
        self,
        frames: List[Dict[str, Any]],
    ) -> List[SymbolInfo]:
        """
        Symbolicate a stack trace.

        Args:
            frames: List of stack frames with 'address' and optional 'module'

        Returns:
            List of symbolicated frames
        """
        result = []

        for frame in frames:
            address = frame.get("address", 0)
            if isinstance(address, str):
                address = int(address, 16)

            module = frame.get("module", "")

            symbol = self.resolve(address, module)
            if symbol:
                result.append(symbol)
            else:
                # Create unknown symbol
                result.append(SymbolInfo(
                    address=address,
                    name=f"<unknown>",
                    module=module,
                ))

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        total_symbols = sum(len(s) for s in self._symbol_files.values())
        return {
            "modules_loaded": len(self._symbol_files),
            "total_symbols": total_symbols,
            "registered_modules": len(self._modules),
            "symbol_paths": [str(p) for p in self.symbol_paths],
            "cache_stats": self._cache.stats() if self._cache else None,
        }


# Global server and convenience functions

_server: Optional[SymbolServer] = None


def get_symbol_server() -> SymbolServer:
    """Get or create the global symbol server."""
    global _server
    if _server is None:
        _server = SymbolServer()
    return _server


def resolve_symbol(
    address: int,
    module: Optional[str] = None,
) -> Optional[SymbolInfo]:
    """
    Resolve an address to symbol information.

    Args:
        address: Address to resolve
        module: Module name

    Returns:
        Symbol info or None
    """
    return get_symbol_server().resolve(address, module)


def lookup_address(
    address: Union[int, str],
    module: Optional[str] = None,
) -> str:
    """
    Look up an address and return formatted string.

    Args:
        address: Address to look up
        module: Module name

    Returns:
        Formatted symbol string
    """
    if isinstance(address, str):
        address = int(address, 16)

    symbol = resolve_symbol(address, module)
    if symbol:
        return symbol.format()
    return f"0x{address:x}"


def symbolicate_stack(
    frames: List[Dict[str, Any]],
) -> List[SymbolInfo]:
    """
    Symbolicate a stack trace.

    Args:
        frames: Stack frames to symbolicate

    Returns:
        Symbolicated frames
    """
    return get_symbol_server().symbolicate_stack(frames)
