class SurfaceOutput:
    def __init__(self):
        self.albedo = (1.0, 1.0, 1.0)
        self.metallic = 0.0
        self.roughness = 0.5
        self.emission = (0.0, 0.0, 0.0)
        self.alpha = 1.0
        self.normal = (0.0, 0.0, 1.0)


class Material:
    """Base class for user-defined materials. Override surface()."""
    def surface(self, ctx: 'SurfaceContext', out: SurfaceOutput) -> None:
        pass


class SurfaceContext:
    """Provides sampling functions available in material surface shaders."""
    def sample(self, path: str, uv: tuple) -> tuple:
        ...

    def noise(self, pos: tuple, scale: float = 1.0) -> float:
        ...

    def texture(self, path: str, uv: tuple) -> tuple:
        ...


def surface(func):
    """Decorator: marks a method as the material surface shader entry point."""
    func._is_surface = True
    return func
