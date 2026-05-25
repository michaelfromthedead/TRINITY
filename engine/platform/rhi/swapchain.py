"""RHI Swapchain for presentation."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import threading


class PresentMode(Enum):
    """Presentation mode."""
    IMMEDIATE = auto()  # No vsync, tearing possible
    VSYNC = auto()      # Traditional vsync
    MAILBOX = auto()    # Triple buffering


class ColorSpace(Enum):
    """Color space."""
    SRGB = auto()
    SCRGB = auto()
    HDR10 = auto()
    PQ = auto()


@dataclass
class SwapchainDesc:
    """Swapchain descriptor."""
    width: int
    height: int
    format: 'Format'
    buffer_count: int = 2
    present_mode: PresentMode = PresentMode.VSYNC
    color_space: ColorSpace = ColorSpace.SRGB


class Swapchain(ABC):
    """Abstract swapchain."""

    @classmethod
    @abstractmethod
    def create(cls, device: 'Device', desc: SwapchainDesc) -> Swapchain:
        """Create swapchain."""
        pass

    @abstractmethod
    def current_texture(self) -> 'Texture':
        """Get current back buffer texture."""
        pass

    @abstractmethod
    def current_index(self) -> int:
        """Get current back buffer index."""
        pass

    @abstractmethod
    def present(self) -> None:
        """Present current back buffer."""
        pass

    @abstractmethod
    def resize(self, width: int, height: int) -> None:
        """Resize swapchain."""
        pass


class NullSwapchain(Swapchain):
    """Null implementation of Swapchain."""

    def __init__(self, device: 'Device', desc: SwapchainDesc):
        self._device = device
        self._desc = desc
        self._current_index = 0
        self._lock = threading.Lock()

        # Create back buffer textures
        from .resources import TextureDesc, TextureType, TextureUsage
        self._textures = []
        for i in range(desc.buffer_count):
            tex_desc = TextureDesc(
                type=TextureType.TEXTURE_2D,
                format=desc.format,
                width=desc.width,
                height=desc.height,
                usage=TextureUsage.RENDER_TARGET
            )
            self._textures.append(device.create_texture(tex_desc))

    @classmethod
    def create(cls, device: 'Device', desc: SwapchainDesc) -> Swapchain:
        """Create swapchain."""
        return cls(device, desc)

    def current_texture(self) -> 'Texture':
        """Get current back buffer texture."""
        with self._lock:
            return self._textures[self._current_index]

    def current_index(self) -> int:
        """Get current back buffer index."""
        with self._lock:
            return self._current_index

    def present(self) -> None:
        """Present current back buffer."""
        with self._lock:
            self._current_index = (self._current_index + 1) % self._desc.buffer_count

    def resize(self, width: int, height: int) -> None:
        """Resize swapchain."""
        with self._lock:
            # Destroy old textures
            for tex in self._textures:
                tex.destroy()

            # Update descriptor
            self._desc.width = width
            self._desc.height = height

            # Recreate textures
            from .resources import TextureDesc, TextureType, TextureUsage
            self._textures = []
            for i in range(self._desc.buffer_count):
                tex_desc = TextureDesc(
                    type=TextureType.TEXTURE_2D,
                    format=self._desc.format,
                    width=width,
                    height=height,
                    usage=TextureUsage.RENDER_TARGET
                )
                self._textures.append(self._device.create_texture(tex_desc))

            self._current_index = 0


# Forward declarations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .device import Device
    from .resources import Texture, Format
