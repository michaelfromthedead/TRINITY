"""Asset cooking system for platform-specific processing.

Provides asset cooking pipeline that discovers, filters, converts,
compresses, and packages game assets for target platforms.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Tuple
import concurrent.futures
import hashlib
import json
import os
import threading
import time


class AssetCookState(Enum):
    """State of an asset in the cooking pipeline."""
    DISCOVERED = auto()
    FILTERED = auto()
    CONVERTING = auto()
    CONVERTED = auto()
    COMPRESSING = auto()
    COMPRESSED = auto()
    PACKAGED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class CookResult:
    """Result of cooking an asset."""
    success: bool
    source_path: str
    output_path: Optional[str] = None
    output_size: int = 0
    elapsed_time: float = 0.0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetInfo:
    """Information about an asset to be cooked."""
    source_path: str
    asset_type: str
    state: AssetCookState = AssetCookState.DISCOVERED
    content_hash: str = ""
    dependencies: List[str] = field(default_factory=list)
    cook_result: Optional[CookResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute content hash of the asset."""
        if os.path.exists(self.source_path):
            hasher = hashlib.sha256()
            with open(self.source_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            self.content_hash = hasher.hexdigest()
        return self.content_hash


class AssetCooker(ABC):
    """Abstract base class for asset cookers."""

    @property
    @abstractmethod
    def supported_extensions(self) -> Set[str]:
        """Get supported file extensions."""
        pass

    @property
    @abstractmethod
    def asset_type(self) -> str:
        """Get the asset type this cooker handles."""
        pass

    @abstractmethod
    def cook(
        self,
        asset: AssetInfo,
        output_dir: str,
        platform: str,
        config: Dict[str, Any]
    ) -> CookResult:
        """Cook an asset for the target platform."""
        pass

    def can_cook(self, asset: AssetInfo) -> bool:
        """Check if this cooker can handle the asset."""
        ext = Path(asset.source_path).suffix.lower()
        return ext in self.supported_extensions

    def get_dependencies(self, asset: AssetInfo) -> List[str]:
        """Get dependencies of an asset."""
        return []


class TextureCooker(AssetCooker):
    """Cooker for texture assets."""

    @property
    def supported_extensions(self) -> Set[str]:
        return {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tiff", ".exr", ".hdr", ".dds"}

    @property
    def asset_type(self) -> str:
        return "texture"

    def cook(
        self,
        asset: AssetInfo,
        output_dir: str,
        platform: str,
        config: Dict[str, Any]
    ) -> CookResult:
        start_time = time.time()

        try:
            # Get cooking settings
            max_size = config.get("max_texture_size", 4096)
            format_name = config.get("texture_format", "DXT")
            generate_mipmaps = config.get("generate_mipmaps", True)
            compression_quality = config.get("texture_quality", 0.9)

            # Determine output format based on platform
            if platform in ("ios", "android", "switch"):
                output_ext = ".astc"
            elif platform in ("ps5", "xbox"):
                output_ext = ".gnf" if platform == "ps5" else ".dds"
            else:
                output_ext = ".dds"

            # Build output path
            source_name = Path(asset.source_path).stem
            output_path = os.path.join(output_dir, f"{source_name}{output_ext}")

            # Simulate texture processing
            # In real implementation, this would use image processing libraries
            metadata = {
                "original_format": Path(asset.source_path).suffix,
                "output_format": output_ext,
                "max_size": max_size,
                "mipmaps": generate_mipmaps,
                "compression": format_name,
                "quality": compression_quality,
            }

            # Simulate output file creation
            output_size = os.path.getsize(asset.source_path) if os.path.exists(asset.source_path) else 0

            return CookResult(
                success=True,
                source_path=asset.source_path,
                output_path=output_path,
                output_size=output_size,
                elapsed_time=time.time() - start_time,
                metadata=metadata,
            )

        except Exception as e:
            return CookResult(
                success=False,
                source_path=asset.source_path,
                elapsed_time=time.time() - start_time,
                error=str(e),
            )


class MeshCooker(AssetCooker):
    """Cooker for mesh/model assets."""

    @property
    def supported_extensions(self) -> Set[str]:
        return {".fbx", ".obj", ".gltf", ".glb", ".blend", ".dae", ".3ds", ".ply"}

    @property
    def asset_type(self) -> str:
        return "mesh"

    def cook(
        self,
        asset: AssetInfo,
        output_dir: str,
        platform: str,
        config: Dict[str, Any]
    ) -> CookResult:
        start_time = time.time()

        try:
            # Get cooking settings
            optimize_mesh = config.get("optimize_mesh", True)
            generate_lods = config.get("generate_lods", True)
            lod_levels = config.get("lod_levels", 3)
            compress_vertices = config.get("compress_vertices", True)

            # Build output path
            source_name = Path(asset.source_path).stem
            output_path = os.path.join(output_dir, f"{source_name}.mesh")

            metadata = {
                "original_format": Path(asset.source_path).suffix,
                "optimized": optimize_mesh,
                "lod_count": lod_levels if generate_lods else 1,
                "compressed": compress_vertices,
            }

            output_size = os.path.getsize(asset.source_path) if os.path.exists(asset.source_path) else 0

            return CookResult(
                success=True,
                source_path=asset.source_path,
                output_path=output_path,
                output_size=output_size,
                elapsed_time=time.time() - start_time,
                metadata=metadata,
            )

        except Exception as e:
            return CookResult(
                success=False,
                source_path=asset.source_path,
                elapsed_time=time.time() - start_time,
                error=str(e),
            )

    def get_dependencies(self, asset: AssetInfo) -> List[str]:
        """Get texture dependencies from mesh materials."""
        deps = []
        # In real implementation, parse the mesh file for texture references
        return deps


class AudioCooker(AssetCooker):
    """Cooker for audio assets."""

    @property
    def supported_extensions(self) -> Set[str]:
        return {".wav", ".mp3", ".ogg", ".flac", ".aiff", ".wma", ".m4a"}

    @property
    def asset_type(self) -> str:
        return "audio"

    def cook(
        self,
        asset: AssetInfo,
        output_dir: str,
        platform: str,
        config: Dict[str, Any]
    ) -> CookResult:
        start_time = time.time()

        try:
            # Get cooking settings
            quality = config.get("audio_quality", 0.8)
            sample_rate = config.get("sample_rate", 44100)
            channels = config.get("channels", "stereo")

            # Determine output format based on platform
            if platform == "ps5":
                output_ext = ".at9"
            elif platform == "xbox":
                output_ext = ".xma2"
            elif platform == "switch":
                output_ext = ".opus"
            elif platform in ("ios", "android"):
                output_ext = ".aac"
            else:
                output_ext = ".ogg"

            # Build output path
            source_name = Path(asset.source_path).stem
            output_path = os.path.join(output_dir, f"{source_name}{output_ext}")

            metadata = {
                "original_format": Path(asset.source_path).suffix,
                "output_format": output_ext,
                "quality": quality,
                "sample_rate": sample_rate,
                "channels": channels,
            }

            output_size = os.path.getsize(asset.source_path) if os.path.exists(asset.source_path) else 0

            return CookResult(
                success=True,
                source_path=asset.source_path,
                output_path=output_path,
                output_size=output_size,
                elapsed_time=time.time() - start_time,
                metadata=metadata,
            )

        except Exception as e:
            return CookResult(
                success=False,
                source_path=asset.source_path,
                elapsed_time=time.time() - start_time,
                error=str(e),
            )


class ShaderCooker(AssetCooker):
    """Cooker for shader assets."""

    @property
    def supported_extensions(self) -> Set[str]:
        return {".hlsl", ".glsl", ".metal", ".vert", ".frag", ".comp", ".geom", ".tesc", ".tese"}

    @property
    def asset_type(self) -> str:
        return "shader"

    def cook(
        self,
        asset: AssetInfo,
        output_dir: str,
        platform: str,
        config: Dict[str, Any]
    ) -> CookResult:
        start_time = time.time()

        try:
            # Get cooking settings
            optimization_level = config.get("shader_optimization", 2)
            debug_info = config.get("shader_debug_info", False)

            # Determine output format based on platform
            if platform in ("windows", "xbox"):
                output_ext = ".dxbc" if not config.get("use_dxil", True) else ".dxil"
            elif platform in ("ios", "macos"):
                output_ext = ".metallib"
            elif platform in ("linux", "android"):
                output_ext = ".spirv"
            elif platform == "ps5":
                output_ext = ".sb"  # PlayStation shader binary
            elif platform == "switch":
                output_ext = ".nvn"
            else:
                output_ext = ".spirv"

            # Build output path
            source_name = Path(asset.source_path).stem
            output_path = os.path.join(output_dir, f"{source_name}{output_ext}")

            metadata = {
                "original_format": Path(asset.source_path).suffix,
                "output_format": output_ext,
                "optimization_level": optimization_level,
                "debug_info": debug_info,
            }

            output_size = os.path.getsize(asset.source_path) if os.path.exists(asset.source_path) else 0

            return CookResult(
                success=True,
                source_path=asset.source_path,
                output_path=output_path,
                output_size=output_size,
                elapsed_time=time.time() - start_time,
                metadata=metadata,
            )

        except Exception as e:
            return CookResult(
                success=False,
                source_path=asset.source_path,
                elapsed_time=time.time() - start_time,
                error=str(e),
            )

    def get_dependencies(self, asset: AssetInfo) -> List[str]:
        """Get shader include dependencies."""
        deps = []
        # In real implementation, parse #include directives
        return deps


class CookRegistry:
    """Registry for asset cookers."""

    def __init__(self):
        self._cookers: Dict[str, AssetCooker] = {}
        self._extension_map: Dict[str, str] = {}
        self._lock = threading.Lock()

    def register(self, cooker: AssetCooker) -> None:
        """Register a cooker."""
        with self._lock:
            self._cookers[cooker.asset_type] = cooker
            for ext in cooker.supported_extensions:
                self._extension_map[ext.lower()] = cooker.asset_type

    def unregister(self, asset_type: str) -> bool:
        """Unregister a cooker by asset type."""
        with self._lock:
            if asset_type not in self._cookers:
                return False

            cooker = self._cookers[asset_type]
            for ext in cooker.supported_extensions:
                if self._extension_map.get(ext.lower()) == asset_type:
                    del self._extension_map[ext.lower()]

            del self._cookers[asset_type]
            return True

    def get_cooker(self, asset_type: str) -> Optional[AssetCooker]:
        """Get a cooker by asset type."""
        return self._cookers.get(asset_type)

    def get_cooker_for_extension(self, extension: str) -> Optional[AssetCooker]:
        """Get a cooker that handles the given extension."""
        asset_type = self._extension_map.get(extension.lower())
        if asset_type:
            return self._cookers.get(asset_type)
        return None

    def get_all_cookers(self) -> Dict[str, AssetCooker]:
        """Get all registered cookers."""
        return dict(self._cookers)

    def get_supported_extensions(self) -> Set[str]:
        """Get all supported file extensions."""
        return set(self._extension_map.keys())


class CookPipeline:
    """Asset cooking pipeline with parallel processing."""

    def __init__(self, registry: Optional[CookRegistry] = None, max_workers: int = 4):
        self._registry = registry or CookRegistry()
        self._max_workers = max_workers
        self._assets: Dict[str, AssetInfo] = {}
        self._callbacks: Dict[str, List[Callable]] = {
            "asset_discovered": [],
            "asset_cooked": [],
            "cook_started": [],
            "cook_completed": [],
        }
        self._lock = threading.Lock()
        self._cancelled = False

        # Register default cookers
        self._register_default_cookers()

    def _register_default_cookers(self) -> None:
        """Register the default asset cookers."""
        self._registry.register(TextureCooker())
        self._registry.register(MeshCooker())
        self._registry.register(AudioCooker())
        self._registry.register(ShaderCooker())

    @property
    def registry(self) -> CookRegistry:
        """Get the cooker registry."""
        return self._registry

    def on(self, event: str, callback: Callable) -> None:
        """Register an event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, *args, **kwargs) -> None:
        """Emit an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception:
                pass

    def discover(
        self,
        source_dir: str,
        patterns: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[AssetInfo]:
        """Discover assets in a directory."""
        discovered = []
        supported = self._registry.get_supported_extensions()

        source_path = Path(source_dir)
        if not source_path.exists():
            return discovered

        # Determine search pattern
        if recursive:
            files = source_path.rglob("*")
        else:
            files = source_path.glob("*")

        for file_path in files:
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            if ext not in supported:
                continue

            # Apply custom patterns if provided
            if patterns:
                matched = any(file_path.match(p) for p in patterns)
                if not matched:
                    continue

            # Determine asset type
            cooker = self._registry.get_cooker_for_extension(ext)
            if not cooker:
                continue

            asset = AssetInfo(
                source_path=str(file_path),
                asset_type=cooker.asset_type,
            )
            asset.compute_hash()

            with self._lock:
                self._assets[asset.source_path] = asset

            discovered.append(asset)
            self._emit("asset_discovered", asset)

        return discovered

    def filter(
        self,
        assets: List[AssetInfo],
        filter_fn: Callable[[AssetInfo], bool]
    ) -> List[AssetInfo]:
        """Filter assets based on a predicate."""
        filtered = []
        for asset in assets:
            if filter_fn(asset):
                asset.state = AssetCookState.FILTERED
                filtered.append(asset)
            else:
                asset.state = AssetCookState.SKIPPED
        return filtered

    def cook(
        self,
        assets: List[AssetInfo],
        output_dir: str,
        platform: str,
        config: Dict[str, Any]
    ) -> Dict[str, CookResult]:
        """Cook assets for the target platform."""
        self._cancelled = False
        results: Dict[str, CookResult] = {}

        self._emit("cook_started", len(assets))

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {}

            for asset in assets:
                if self._cancelled:
                    break

                cooker = self._registry.get_cooker(asset.asset_type)
                if not cooker:
                    asset.state = AssetCookState.FAILED
                    results[asset.source_path] = CookResult(
                        success=False,
                        source_path=asset.source_path,
                        error=f"No cooker found for asset type: {asset.asset_type}"
                    )
                    continue

                future = executor.submit(
                    self._cook_asset, asset, cooker, output_dir, platform, config
                )
                futures[future] = asset

            for future in concurrent.futures.as_completed(futures):
                if self._cancelled:
                    break

                asset = futures[future]
                try:
                    result = future.result()
                    asset.cook_result = result
                    results[asset.source_path] = result

                    if result.success:
                        asset.state = AssetCookState.CONVERTED
                    else:
                        asset.state = AssetCookState.FAILED

                    self._emit("asset_cooked", asset, result)

                except Exception as e:
                    result = CookResult(
                        success=False,
                        source_path=asset.source_path,
                        error=str(e)
                    )
                    asset.cook_result = result
                    asset.state = AssetCookState.FAILED
                    results[asset.source_path] = result

        self._emit("cook_completed", results)
        return results

    def _cook_asset(
        self,
        asset: AssetInfo,
        cooker: AssetCooker,
        output_dir: str,
        platform: str,
        config: Dict[str, Any]
    ) -> CookResult:
        """Cook a single asset."""
        asset.state = AssetCookState.CONVERTING
        return cooker.cook(asset, output_dir, platform, config)

    def cancel(self) -> None:
        """Cancel the cooking process."""
        self._cancelled = True

    def get_asset(self, source_path: str) -> Optional[AssetInfo]:
        """Get an asset by source path."""
        return self._assets.get(source_path)

    def get_all_assets(self) -> Dict[str, AssetInfo]:
        """Get all discovered assets."""
        return dict(self._assets)

    def clear(self) -> None:
        """Clear all discovered assets."""
        with self._lock:
            self._assets.clear()


def cook_project(
    source_dir: str,
    output_dir: str,
    platform: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, CookResult]:
    """Convenience function to cook an entire project."""
    pipeline = CookPipeline()
    config = config or {}

    # Discover all assets
    assets = pipeline.discover(source_dir, recursive=True)

    # Cook all assets
    return pipeline.cook(assets, output_dir, platform, config)
