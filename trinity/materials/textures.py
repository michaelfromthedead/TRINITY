"""Texture binding model for TRINITY Material DSL (T-MAT-1.5).

This module provides texture descriptor classes that generate WGSL bindings
for 2D textures and cubemaps. Textures are declared as class attributes on
Material subclasses and automatically generate shader bindings during
metaclass compilation.

Key Features:
- Texture2D: 2D texture descriptors with sampling support
- TextureCube: Cubemap texture descriptors for environment/reflection maps
- Automatic WGSL binding generation at class definition time
- sRGB format support for gamma-correct sampling
- Default texture fallbacks (white, black, flat_normal, gray)
- Configurable filtering and addressing modes

Example::

    from trinity.materials import Material, MaterialMeta, surface
    from trinity.materials.textures import Texture2D, TextureCube

    class BrickMaterial(Material, metaclass=MaterialMeta):
        albedo = Texture2D(default="white", srgb=True)
        normal = Texture2D(default="flat_normal")
        roughness = Texture2D(default="gray")
        environment = TextureCube(default="black")

        @surface
        def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
            out.base_color = ctx.sample(self.albedo, ctx.uv).xyz
            out.normal = ctx.sample(self.normal, ctx.uv).xyz * 2.0 - 1.0
            out.roughness = ctx.sample(self.roughness, ctx.uv).r
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


# =============================================================================
# ENUMS: Filter and Address Modes
# =============================================================================


class FilterMode(Enum):
    """Texture filtering modes for sampling."""
    NEAREST = "nearest"
    LINEAR = "linear"

    def to_wgsl(self) -> str:
        """Convert to WGSL sampler filter mode."""
        return self.value


class AddressMode(Enum):
    """Texture addressing modes for out-of-bounds UVs."""
    REPEAT = "repeat"
    MIRROR_REPEAT = "mirror-repeat"
    CLAMP_TO_EDGE = "clamp-to-edge"

    def to_wgsl(self) -> str:
        """Convert to WGSL sampler address mode."""
        return self.value


class TextureFormat(Enum):
    """WGSL texture format specifiers."""
    RGBA8_UNORM = "rgba8unorm"
    RGBA8_UNORM_SRGB = "rgba8unorm-srgb"
    RGBA16_FLOAT = "rgba16float"
    RGBA32_FLOAT = "rgba32float"
    R8_UNORM = "r8unorm"
    RG8_UNORM = "rg8unorm"
    DEPTH32_FLOAT = "depth32float"

    def to_wgsl(self) -> str:
        """Convert to WGSL format string."""
        return self.value


# =============================================================================
# DEFAULT TEXTURES: Common fallback patterns
# =============================================================================


@dataclass(frozen=True)
class DefaultTextureSpec:
    """Specification for a default texture fallback."""
    name: str
    color: Tuple[float, float, float, float]
    description: str


DEFAULT_TEXTURES: Dict[str, DefaultTextureSpec] = {
    "white": DefaultTextureSpec(
        name="white",
        color=(1.0, 1.0, 1.0, 1.0),
        description="1x1 white pixel (base_color default)"
    ),
    "black": DefaultTextureSpec(
        name="black",
        color=(0.0, 0.0, 0.0, 1.0),
        description="1x1 black pixel (metallic default)"
    ),
    "flat_normal": DefaultTextureSpec(
        name="flat_normal",
        color=(0.5, 0.5, 1.0, 1.0),
        description="1x1 flat normal (0.5, 0.5, 1.0) for normal maps"
    ),
    "gray": DefaultTextureSpec(
        name="gray",
        color=(0.5, 0.5, 0.5, 1.0),
        description="1x1 mid-gray (roughness default)"
    ),
    "transparent": DefaultTextureSpec(
        name="transparent",
        color=(0.0, 0.0, 0.0, 0.0),
        description="1x1 fully transparent"
    ),
}


def get_default_texture(name: str) -> Optional[DefaultTextureSpec]:
    """Get a default texture specification by name.

    Args:
        name: Default texture name (white, black, flat_normal, gray, transparent)

    Returns:
        DefaultTextureSpec if found, None otherwise
    """
    return DEFAULT_TEXTURES.get(name)


def is_valid_default(name: str) -> bool:
    """Check if a name is a valid default texture."""
    return name in DEFAULT_TEXTURES


# =============================================================================
# BASE TEXTURE DESCRIPTOR
# =============================================================================


class TextureDescriptor:
    """Base class for texture descriptors.

    Texture descriptors are declared as class attributes on Material subclasses.
    During MaterialMeta compilation, each descriptor generates WGSL binding
    declarations for both the texture and its associated sampler.

    Attributes:
        default: Default texture fallback name ("white", "black", etc.)
        srgb: If True, use sRGB format for automatic gamma conversion
        filter: Texture filtering mode ("nearest" or "linear")
        address: UV address mode ("repeat", "mirror-repeat", "clamp-to-edge")
        path: Optional path to texture file
        binding_index: Binding index assigned by MaterialMeta (set during compilation)
    """

    # Marker attribute for MaterialMeta detection
    _is_texture_descriptor = True
    _is_cube = False

    def __init__(
        self,
        default: str = "white",
        srgb: bool = False,
        filter: str = "linear",
        address: str = "repeat",
        path: Optional[str] = None,
    ):
        """Initialize a texture descriptor.

        Args:
            default: Default texture fallback name. Must be one of:
                "white", "black", "flat_normal", "gray", "transparent"
            srgb: Enable sRGB format for gamma-correct sampling
            filter: Texture filter mode ("nearest" or "linear")
            address: UV address mode ("repeat", "mirror-repeat", "clamp-to-edge")
            path: Optional path to texture file (for runtime loading)

        Raises:
            ValueError: If default, filter, or address is invalid
        """
        # Validate default texture
        if not is_valid_default(default):
            valid = ", ".join(DEFAULT_TEXTURES.keys())
            raise ValueError(f"Invalid default texture '{default}'. Valid: {valid}")

        # Validate and convert filter mode
        if isinstance(filter, str):
            try:
                filter = FilterMode(filter)
            except ValueError:
                valid = ", ".join(f.value for f in FilterMode)
                raise ValueError(f"Invalid filter mode '{filter}'. Valid: {valid}")
        elif isinstance(filter, FilterMode):
            pass
        else:
            raise TypeError(f"filter must be str or FilterMode, got {type(filter)}")

        # Validate and convert address mode
        if isinstance(address, str):
            try:
                address = AddressMode(address)
            except ValueError:
                valid = ", ".join(a.value for a in AddressMode)
                raise ValueError(f"Invalid address mode '{address}'. Valid: {valid}")
        elif isinstance(address, AddressMode):
            pass
        else:
            raise TypeError(f"address must be str or AddressMode, got {type(address)}")

        self.default = default
        self.srgb = srgb
        self.filter = filter
        self.address = address
        self.path = path
        self.binding_index: Optional[int] = None
        self._name: Optional[str] = None

    def __set_name__(self, owner: type, name: str) -> None:
        """Called when descriptor is assigned to a class attribute.

        This magic method captures the attribute name for use in WGSL generation.
        """
        self._name = name

    @property
    def name(self) -> str:
        """Get the attribute name of this descriptor."""
        return self._name or "unnamed_texture"

    def get_format(self) -> TextureFormat:
        """Get the WGSL texture format based on sRGB flag.

        Returns:
            TextureFormat.RGBA8_UNORM_SRGB if srgb is True,
            TextureFormat.RGBA8_UNORM otherwise
        """
        if self.srgb:
            return TextureFormat.RGBA8_UNORM_SRGB
        return TextureFormat.RGBA8_UNORM

    def get_default_spec(self) -> DefaultTextureSpec:
        """Get the default texture specification.

        Returns:
            DefaultTextureSpec for this descriptor's default texture
        """
        spec = get_default_texture(self.default)
        if spec is None:
            # Fallback to white if somehow invalid
            return DEFAULT_TEXTURES["white"]
        return spec

    def generate_wgsl_binding(self, name: str, group: int = 0) -> str:
        """Generate WGSL texture and sampler binding declarations.

        Args:
            name: Variable name for the texture (e.g., "albedo")
            group: Binding group index (default 0)

        Returns:
            WGSL binding declarations as string

        Raises:
            RuntimeError: If binding_index has not been set
        """
        raise NotImplementedError("Subclasses must implement generate_wgsl_binding")

    def generate_sampler_descriptor(self) -> str:
        """Generate WGSL sampler descriptor creation code.

        Returns:
            WGSL SamplerDescriptor initialization
        """
        return (
            f"SamplerDescriptor {{\n"
            f"    address_mode_u: AddressMode::{self.address.name.title()},\n"
            f"    address_mode_v: AddressMode::{self.address.name.title()},\n"
            f"    address_mode_w: AddressMode::{self.address.name.title()},\n"
            f"    mag_filter: FilterMode::{self.filter.name.title()},\n"
            f"    min_filter: FilterMode::{self.filter.name.title()},\n"
            f"    mipmap_filter: FilterMode::{self.filter.name.title()},\n"
            f"}}"
        )

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(default='{self.default}', srgb={self.srgb}, "
            f"filter='{self.filter.value}', address='{self.address.value}')"
        )


# =============================================================================
# TEXTURE2D: 2D Texture Descriptor
# =============================================================================


class Texture2D(TextureDescriptor):
    """Descriptor for 2D textures with WGSL binding generation.

    Generates WGSL bindings for texture_2d<f32> and associated sampler.
    Used for albedo, normal, roughness, metallic, and other 2D texture maps.

    Example::

        class MyMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white", srgb=True)
            normal = Texture2D(default="flat_normal")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = ctx.sample(self.albedo, ctx.uv).xyz

    The above generates WGSL like::

        @group(0) @binding(1) var albedo_tex: texture_2d<f32>;
        @group(0) @binding(2) var albedo_sampler: sampler;
        @group(0) @binding(3) var normal_tex: texture_2d<f32>;
        @group(0) @binding(4) var normal_sampler: sampler;
    """

    _is_cube = False

    def generate_wgsl_binding(self, name: str, group: int = 0) -> str:
        """Generate WGSL texture and sampler binding declarations.

        Args:
            name: Variable name for the texture (e.g., "albedo")
            group: Binding group index (default 0)

        Returns:
            WGSL binding declarations for texture_2d and sampler

        Raises:
            RuntimeError: If binding_index has not been set
        """
        if self.binding_index is None:
            raise RuntimeError(
                f"Texture2D '{name}' has no binding_index. "
                "This is set by MaterialMeta during class creation."
            )

        texture_binding = self.binding_index
        sampler_binding = self.binding_index + 1

        # Generate format comment for sRGB
        format_comment = ""
        if self.srgb:
            format_comment = f" // sRGB: {self.get_format().value}"

        return (
            f"@group({group}) @binding({texture_binding}) "
            f"var {name}_tex: texture_2d<f32>;{format_comment}\n"
            f"@group({group}) @binding({sampler_binding}) "
            f"var {name}_sampler: sampler;"
        )

    def generate_sample_call(self, uv_expr: str) -> str:
        """Generate WGSL textureSample call for this texture.

        Args:
            uv_expr: WGSL expression for UV coordinates

        Returns:
            WGSL textureSample call string
        """
        return f"textureSample({self.name}_tex, {self.name}_sampler, {uv_expr})"


# =============================================================================
# TEXTURECUBE: Cubemap Texture Descriptor
# =============================================================================


class TextureCube(TextureDescriptor):
    """Descriptor for cubemap textures with WGSL binding generation.

    Generates WGSL bindings for texture_cube<f32> and associated sampler.
    Used for environment maps, reflection probes, and skyboxes.

    Example::

        class ReflectiveMaterial(Material, metaclass=MaterialMeta):
            environment = TextureCube(default="black")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                reflection = ctx.sample_cube(self.environment, reflect_dir)
                out.emissive = reflection.xyz * out.metallic

    The above generates WGSL like::

        @group(0) @binding(1) var environment_tex: texture_cube<f32>;
        @group(0) @binding(2) var environment_sampler: sampler;
    """

    _is_cube = True

    def generate_wgsl_binding(self, name: str, group: int = 0) -> str:
        """Generate WGSL texture and sampler binding declarations.

        Args:
            name: Variable name for the texture (e.g., "environment")
            group: Binding group index (default 0)

        Returns:
            WGSL binding declarations for texture_cube and sampler

        Raises:
            RuntimeError: If binding_index has not been set
        """
        if self.binding_index is None:
            raise RuntimeError(
                f"TextureCube '{name}' has no binding_index. "
                "This is set by MaterialMeta during class creation."
            )

        texture_binding = self.binding_index
        sampler_binding = self.binding_index + 1

        # Generate format comment for sRGB
        format_comment = ""
        if self.srgb:
            format_comment = f" // sRGB: {self.get_format().value}"

        return (
            f"@group({group}) @binding({texture_binding}) "
            f"var {name}_tex: texture_cube<f32>;{format_comment}\n"
            f"@group({group}) @binding({sampler_binding}) "
            f"var {name}_sampler: sampler;"
        )

    def generate_sample_call(self, direction_expr: str) -> str:
        """Generate WGSL textureSample call for this cubemap.

        Args:
            direction_expr: WGSL expression for sampling direction

        Returns:
            WGSL textureSample call string
        """
        return f"textureSample({self.name}_tex, {self.name}_sampler, {direction_expr})"


# =============================================================================
# TEXTURE BINDING COLLECTION
# =============================================================================


@dataclass
class TextureBindingSet:
    """Collection of texture bindings for a material.

    This is used by MaterialMeta to track all texture descriptors
    on a material class and assign binding indices.
    """

    bindings: Dict[str, TextureDescriptor] = field(default_factory=dict)
    _next_binding_index: int = 1  # 0 is reserved for uniforms

    def add(self, name: str, descriptor: TextureDescriptor) -> int:
        """Add a texture descriptor and assign its binding index.

        Args:
            name: Attribute name of the texture
            descriptor: TextureDescriptor instance

        Returns:
            The assigned binding index
        """
        binding_idx = self._next_binding_index
        descriptor.binding_index = binding_idx
        descriptor._name = name
        self.bindings[name] = descriptor
        # Each texture uses 2 bindings: texture + sampler
        self._next_binding_index += 2
        return binding_idx

    def get(self, name: str) -> Optional[TextureDescriptor]:
        """Get a texture descriptor by name."""
        return self.bindings.get(name)

    def generate_all_bindings(self, group: int = 0) -> str:
        """Generate WGSL bindings for all textures.

        Args:
            group: Binding group index

        Returns:
            WGSL binding declarations for all textures
        """
        lines = []
        for name, descriptor in self.bindings.items():
            lines.append(descriptor.generate_wgsl_binding(name, group))
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.bindings)

    def __iter__(self):
        return iter(self.bindings.items())

    def __contains__(self, name: str) -> bool:
        return name in self.bindings


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def collect_texture_descriptors(
    namespace: Dict[str, Any]
) -> TextureBindingSet:
    """Collect all TextureDescriptor instances from a class namespace.

    This is called by MaterialMeta during class creation to find all
    texture descriptors and assign binding indices.

    Args:
        namespace: Class namespace dictionary

    Returns:
        TextureBindingSet containing all found descriptors
    """
    binding_set = TextureBindingSet()

    for name, value in namespace.items():
        if isinstance(value, TextureDescriptor):
            binding_set.add(name, value)

    return binding_set


def validate_texture_descriptor(descriptor: TextureDescriptor) -> None:
    """Validate a texture descriptor configuration.

    Args:
        descriptor: TextureDescriptor to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if not is_valid_default(descriptor.default):
        valid = ", ".join(DEFAULT_TEXTURES.keys())
        raise ValueError(
            f"Invalid default texture '{descriptor.default}'. Valid: {valid}"
        )

    if not isinstance(descriptor.filter, FilterMode):
        raise ValueError(f"Invalid filter mode: {descriptor.filter}")

    if not isinstance(descriptor.address, AddressMode):
        raise ValueError(f"Invalid address mode: {descriptor.address}")
