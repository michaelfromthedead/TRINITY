"""Material DSL classes for TRINITY engine.

Provides the base classes and metaclass for defining materials with
texture bindings that compile to WGSL shaders.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from trinity.materials.textures import TextureDescriptor


class Vec2:
    """2D vector for material shader parameters."""

    __slots__ = ('x', 'y')

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self) -> str:
        return f"Vec2({self.x}, {self.y})"

    def __iter__(self):
        return iter((self.x, self.y))

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y)[index]

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


class Vec3:
    """3D vector for material shader parameters."""

    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        return f"Vec3({self.x}, {self.y}, {self.z})"

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y, self.z)[index]

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class Vec4:
    """4D vector for material shader parameters."""

    __slots__ = ('x', 'y', 'z', 'w')

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    def __repr__(self) -> str:
        return f"Vec4({self.x}, {self.y}, {self.z}, {self.w})"

    def __iter__(self):
        return iter((self.x, self.y, self.z, self.w))

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y, self.z, self.w)[index]

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)


class SurfaceOutput:
    """Output structure for material surface shaders.

    Provides aliases for backwards compatibility:
    - albedo -> base_color (deprecated)
    - emission -> emissive (deprecated)
    """

    def __init__(self):
        self._base_color = Vec3(1.0, 1.0, 1.0)
        self.metallic = 0.0
        self.roughness = 0.5
        self._emissive = Vec3(0.0, 0.0, 0.0)
        self.alpha = 1.0
        self.normal = (0.0, 0.0, 1.0)
        self.ao = 1.0
        self.clearcoat = 0.0
        self.clearcoat_roughness = 0.0
        self.subsurface = 0.0
        self.specular = 0.5
        self.anisotropy = 0.0
        self.sheen = 0.0
        self.sheen_tint = 0.0
        self.transmission = 0.0
        self.ior = 1.5
        self.thickness = 0.0
        self.ambient_occlusion = 1.0
        self.iridescence = 0.0
        self.iridescence_ior = 1.3

    @property
    def base_color(self) -> Vec3:
        """Primary surface color."""
        return self._base_color

    @base_color.setter
    def base_color(self, value):
        if isinstance(value, Vec3):
            self._base_color = value
        elif isinstance(value, (tuple, list)) and len(value) >= 3:
            self._base_color = Vec3(value[0], value[1], value[2])
        else:
            self._base_color = Vec3(float(value), float(value), float(value))

    @property
    def albedo(self) -> Vec3:
        """Alias for base_color (deprecated)."""
        return self._base_color

    @albedo.setter
    def albedo(self, value):
        """Set base_color via albedo alias."""
        if isinstance(value, Vec3):
            self._base_color = value
        elif isinstance(value, (tuple, list)) and len(value) >= 3:
            self._base_color = Vec3(value[0], value[1], value[2])
        else:
            self._base_color = Vec3(float(value), float(value), float(value))

    @property
    def emissive(self) -> Vec3:
        """Emissive color for glow/light emission."""
        return self._emissive

    @emissive.setter
    def emissive(self, value):
        if isinstance(value, Vec3):
            self._emissive = value
        elif isinstance(value, (tuple, list)) and len(value) >= 3:
            self._emissive = Vec3(value[0], value[1], value[2])
        else:
            self._emissive = Vec3(float(value), float(value), float(value))

    @property
    def emission(self) -> Vec3:
        """Alias for emissive (deprecated)."""
        return self._emissive

    @emission.setter
    def emission(self, value):
        """Set emissive via emission alias."""
        if isinstance(value, Vec3):
            self._emissive = value
        elif isinstance(value, (tuple, list)) and len(value) >= 3:
            self._emissive = Vec3(value[0], value[1], value[2])
        else:
            self._emissive = Vec3(float(value), float(value), float(value))


class SurfaceContext:
    """Provides sampling functions available in material surface shaders."""

    def __init__(self):
        self.position = Vec3(0.0, 0.0, 0.0)
        self.normal = Vec3(0.0, 1.0, 0.0)
        self.uv = Vec2(0.0, 0.0)
        self.vertex_color = Vec4(1.0, 1.0, 1.0, 1.0)
        self._time = 0.0

    def sample(self, texture: Any, uv: tuple) -> Vec4:
        """Sample a texture at the given UV coordinates."""
        return Vec4(1.0, 1.0, 1.0, 1.0)

    def sample_cube(self, texture: Any, direction: tuple) -> Vec4:
        """Sample a cubemap texture in the given direction."""
        return Vec4(0.0, 0.0, 0.0, 1.0)

    def noise(self, pos: tuple, scale: float = 1.0) -> float:
        """Generate procedural noise at the given position."""
        return 0.0

    def texture(self, path: str, uv: tuple) -> tuple:
        """Sample a texture by path (deprecated, use sample())."""
        return (1.0, 1.0, 1.0, 1.0)

    def world_position(self) -> Vec3:
        """Get world-space position."""
        return Vec3(self.position.x, self.position.y, self.position.z)

    def world_normal(self) -> Vec3:
        """Get world-space normal."""
        return Vec3(self.normal.x, self.normal.y, self.normal.z)

    def world_tangent(self) -> Vec3:
        """Get world-space tangent."""
        return Vec3(1.0, 0.0, 0.0)

    def get_uv(self) -> Vec2:
        """Get UV coordinates."""
        return Vec2(self.uv.x, self.uv.y)

    def get_vertex_color(self) -> Vec4:
        """Get vertex color."""
        return Vec4(self.vertex_color.x, self.vertex_color.y,
                    self.vertex_color.z, self.vertex_color.w)

    def get_time(self) -> float:
        """Get shader time."""
        return self._time


def surface(func):
    """Decorator: marks a method as the material surface shader entry point."""
    func._is_surface = True
    return func


class Material:
    """Base class for user-defined materials. Override surface()."""

    # Set by MaterialMeta during class creation
    _texture_bindings: Dict[str, Any] = {}
    _wgsl_source: str = ""
    _compilation_error: Optional[str] = None

    def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
        pass

    @classmethod
    def has_texture(cls, name: str) -> bool:
        """Check if this material has a texture with the given name.

        Args:
            name: Texture binding name.

        Returns:
            True if the texture exists, False otherwise.
        """
        return name in cls._texture_bindings

    @classmethod
    def get_textures(cls) -> Dict[str, Any]:
        """Get all texture bindings for this material.

        Returns:
            Dictionary mapping texture names to TextureDescriptor objects.
        """
        return dict(cls._texture_bindings)

    @classmethod
    def get_inherited_textures(cls) -> Dict[str, Any]:
        """Get texture bindings inherited from parent classes.

        Returns:
            Dictionary of inherited textures (not defined in this class).
        """
        own_textures = getattr(cls, '_own_textures', set())
        return {
            name: tex
            for name, tex in cls._texture_bindings.items()
            if name not in own_textures
        }

    @classmethod
    def get_own_textures(cls) -> Dict[str, Any]:
        """Get texture bindings defined in this class (not inherited).

        Returns:
            Dictionary of textures defined directly on this class.
        """
        own_textures = getattr(cls, '_own_textures', set())
        return {
            name: tex
            for name, tex in cls._texture_bindings.items()
            if name in own_textures
        }

    @classmethod
    def has_super_call(cls) -> bool:
        """Check if this material's surface method calls super().surface().

        Returns:
            True if super().surface() is called, False otherwise.
        """
        return getattr(cls, '_has_super_call', False)

    @classmethod
    def get_parent_material(cls) -> Optional[type]:
        """Get the parent material class if this material inherits from one.

        Returns:
            Parent Material class or None if no parent.
        """
        return getattr(cls, '_parent_material', None)

    @classmethod
    def get_inheritance_chain(cls) -> list:
        """Get the inheritance chain of material classes.

        Returns:
            List of material classes from this class up to (but not including)
            the base Material class, in order from child to parent.
        """
        chain = [cls]
        current = cls
        while True:
            parent = getattr(current, '_parent_material', None)
            if parent is None:
                break
            # Stop if we've reached the base Material class
            if parent.__name__ == 'Material' and not hasattr(parent, '_surface_method'):
                break
            chain.append(parent)
            current = parent
        return chain


class MaterialMeta(type):
    """Metaclass for Material classes that collects texture bindings and compiles WGSL.

    When a class is defined with `metaclass=MaterialMeta`, this metaclass:
    1. Collects all TextureDescriptor attributes (Texture2D, TextureCube)
    2. Assigns binding indices to each texture
    3. Generates WGSL binding declarations
    4. Compiles the surface() method to WGSL

    Example::

        class MyMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white", srgb=True)
            normal = Texture2D(default="flat_normal")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 1.0, 1.0)

    After class creation:
        - MyMaterial._texture_bindings = {"albedo": <Texture2D>, "normal": <Texture2D>}
        - MyMaterial._wgsl_source = "// Generated WGSL..."
        - MyMaterial._compilation_error = None (or error message if failed)
    """

    # Registry of all materials created with this metaclass
    _registry: Dict[str, type] = {}

    @classmethod
    def get_material(mcs, name: str) -> Optional[type]:
        """Get a material class by name.

        Args:
            name: Material class name.

        Returns:
            Material class or None if not found.
        """
        return mcs._registry.get(name)

    @classmethod
    def all_materials(mcs) -> list:
        """Get all registered material classes.

        Returns:
            List of all registered Material subclasses.
        """
        return list(mcs._registry.values())

    def __new__(
        mcs,
        name: str,
        bases: Tuple[type, ...],
        namespace: Dict[str, Any],
        **kwargs: Any
    ) -> MaterialMeta:
        # Collect texture descriptors from namespace
        texture_bindings: Dict[str, Any] = {}

        # Also inherit texture bindings from parent classes (respecting MRO)
        # Process bases in REVERSE order so first base wins (MRO precedence)
        for base in reversed(bases):
            if hasattr(base, '_texture_bindings'):
                texture_bindings.update(base._texture_bindings)

        # Find new texture descriptors in this class
        own_textures: set = set()
        binding_index = len(texture_bindings) * 2 + 1  # Start after inherited bindings
        for attr_name, attr_value in namespace.items():
            if hasattr(attr_value, '_is_texture_descriptor') and attr_value._is_texture_descriptor:
                # Assign binding index (2 bindings per texture: texture + sampler)
                attr_value.binding_index = binding_index
                attr_value._name = attr_name
                texture_bindings[attr_name] = attr_value
                own_textures.add(attr_name)
                binding_index += 2

        # Create the class
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Attach texture bindings
        cls._texture_bindings = texture_bindings
        cls._own_textures = own_textures

        # Store reference to surface method - check namespace first, then inherit from parents
        surface_method = namespace.get('surface')
        parent_material = None
        inherited_surface = False

        if surface_method is not None:
            cls._surface_method = surface_method
            # Find parent material for super() calls
            for base in bases:
                if hasattr(base, '_surface_method') and base._surface_method is not None:
                    parent_material = base
                    break
        else:
            # Try to inherit surface method from parent classes
            for base in bases:
                if hasattr(base, '_surface_method') and base._surface_method is not None:
                    cls._surface_method = base._surface_method
                    surface_method = base._surface_method
                    parent_material = base
                    inherited_surface = True
                    break
            else:
                cls._surface_method = None

        # Track parent material reference for super() calls
        cls._parent_material = parent_material

        # Register the material
        mcs._registry[name] = cls

        # Generate WGSL source
        try:
            import ast
            import inspect
            import textwrap

            # If we inherited surface without overriding, use parent's WGSL
            if inherited_surface and parent_material is not None:
                cls._wgsl_source = parent_material._wgsl_source
                cls._has_super_call = getattr(parent_material, '_has_super_call', False)
                cls._used_builtins = getattr(parent_material, '_used_builtins', set())
                cls._compilation_error = None
            else:
                wgsl_parts = []

                # Generate bindings for OWN textures only (not inherited)
                for tex_name in own_textures:
                    tex_desc = texture_bindings[tex_name]
                    if hasattr(tex_desc, 'generate_wgsl_binding'):
                        wgsl_parts.append(tex_desc.generate_wgsl_binding(tex_name))

                # Translate surface method to WGSL
                wgsl_parts.append("\n// Surface shader")
                wgsl_parts.append("fn surface_main() {")

                has_super_call = False
                used_builtins: set = set()

                if surface_method is not None:
                    try:
                        source = textwrap.dedent(inspect.getsource(surface_method))
                        tree = ast.parse(source)

                        # Check for super().surface() calls
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Call):
                                if isinstance(node.func, ast.Attribute):
                                    if node.func.attr == 'surface':
                                        # Check if it's super().surface()
                                        if isinstance(node.func.value, ast.Call):
                                            if isinstance(node.func.value.func, ast.Name):
                                                if node.func.value.func.id == 'super':
                                                    has_super_call = True
                                                    # Validate parent exists
                                                    if parent_material is None:
                                                        raise WGSLTranslationError(
                                                            f"super().surface() called in {name} but no parent "
                                                            f"material with surface method found. "
                                                            f"Material shader DSL requires a parent with surface()."
                                                        )
                                                    break

                        translator = PythonToWGSLTranslator()
                        # Find the function body
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef) and node.name == 'surface':
                                for stmt in node.body:
                                    # Check if this is super().surface() call
                                    if has_super_call and isinstance(stmt, ast.Expr):
                                        if isinstance(stmt.value, ast.Call):
                                            func = stmt.value.func
                                            if isinstance(func, ast.Attribute) and func.attr == 'surface':
                                                if isinstance(func.value, ast.Call):
                                                    super_func = func.value.func
                                                    if isinstance(super_func, ast.Name) and super_func.id == 'super':
                                                        # Inline parent code
                                                        if parent_material is not None:
                                                            parent_wgsl = getattr(parent_material, '_wgsl_source', '')
                                                            # Extract just the body content (between { and })
                                                            if 'fn surface_main()' in parent_wgsl:
                                                                start = parent_wgsl.find('fn surface_main() {')
                                                                if start != -1:
                                                                    brace_start = parent_wgsl.find('{', start)
                                                                    brace_end = parent_wgsl.rfind('}')
                                                                    if brace_start != -1 and brace_end != -1:
                                                                        body = parent_wgsl[brace_start+1:brace_end].strip()
                                                                        wgsl_parts.append(f"    // Inlined from {parent_material.__name__}")
                                                                        for line in body.split('\n'):
                                                                            if line.strip():
                                                                                wgsl_parts.append(f"    {line.strip()}")
                                                        continue
                                    line = translator.translate(stmt)
                                    wgsl_parts.append(f"    {line}")
                                break

                        used_builtins = translator.used_builtins
                        # Merge parent builtins if super() was called
                        if has_super_call and parent_material is not None:
                            parent_builtins = getattr(parent_material, '_used_builtins', set())
                            used_builtins = used_builtins | parent_builtins

                    except WGSLTranslationError as e:
                        # Propagate translation errors to _compilation_error
                        cls._has_super_call = has_super_call
                        cls._used_builtins = used_builtins
                        wgsl_parts.append("}")
                        cls._wgsl_source = "\n".join(wgsl_parts)
                        cls._compilation_error = str(e)
                        return cls
                    except Exception as e:
                        wgsl_parts.append(f"    // Translation error: {e}")

                cls._has_super_call = has_super_call
                cls._used_builtins = used_builtins

                wgsl_parts.append("}")

                # Add alias comments for special builtins (ctx.time -> time_uniforms.elapsed_seconds)
                final_wgsl = "\n".join(wgsl_parts)
                if 'ctx.time' in used_builtins:
                    # Add ctx.time alias comment for backward compatibility
                    final_wgsl = final_wgsl.replace(
                        "// Surface shader",
                        "// Surface shader (ctx.time -> time_uniforms.elapsed_seconds)"
                    )

                cls._wgsl_source = final_wgsl
                cls._compilation_error = None
        except Exception as e:
            cls._wgsl_source = ""
            cls._compilation_error = str(e)
            cls._used_builtins = set()

        # Add get_wgsl method to class
        def get_wgsl(self_or_cls) -> str:
            """Get the compiled WGSL source code."""
            if isinstance(self_or_cls, type):
                return self_or_cls._wgsl_source
            return self_or_cls.__class__._wgsl_source

        cls.get_wgsl = classmethod(lambda cls_: cls_._wgsl_source)

        return cls


class WGSLTranslationError(Exception):
    """Exception raised when Python to WGSL translation fails."""

    def __init__(self, message: str, node: Any = None):
        self.message = message
        self.node = node
        super().__init__(message)


class PythonToWGSLTranslator:
    """Translates Python AST to WGSL shader code.

    Supports the following AST node types:
    - Expr: Expression statements
    - Constant: Literals (int, float, bool, str)
    - Name: Variable references
    - Attribute: Attribute access (obj.attr)
    - BinOp: Binary operations (+, -, *, /, etc.)
    - UnaryOp: Unary operations (-, not, ~)
    - Compare: Comparison operations (==, !=, <, >, etc.)
    - BoolOp: Boolean operations (and, or)
    - Call: Function/method calls
    - Assign: Assignment statements
    - AugAssign: Augmented assignment (+=, -=, etc.)
    - If: Conditional statements
    - Return: Return statements
    - Subscript: Index access (arr[i])
    - Tuple: Tuple expressions

    Example::

        import ast
        translator = PythonToWGSLTranslator()
        tree = ast.parse("out.base_color = Vec3(1.0, 0.5, 0.2)")
        wgsl = translator.translate(tree)
        # -> "out.base_color = vec3<f32>(1.0, 0.5, 0.2);"
    """

    # Python to WGSL type mappings
    TYPE_MAP = {
        'Vec2': 'vec2<f32>',
        'Vec3': 'vec3<f32>',
        'Vec4': 'vec4<f32>',
        'float': 'f32',
        'int': 'i32',
        'bool': 'bool',
    }

    # Context attribute mappings (ctx.X -> in.X or special)
    CTX_ATTR_MAP = {
        'uv': 'in.uv',
        'position': 'in.position',
        'normal': 'in.normal',
        'tangent': 'in.tangent',
        'vertex_color': 'in.vertex_color',
        'time': 'time_uniforms.elapsed_seconds',
        'delta_time': 'time_uniforms.delta_time',
        'world_position': 'in.world_position',
        'world_normal': 'in.world_normal',
    }

    # Context method call mappings (ctx.method() -> result)
    CTX_METHOD_MAP = {
        'world_position': 'in.world_position',
        'world_normal': 'in.world_normal',
        'world_tangent': 'in.world_tangent',
        'get_uv': 'in.uv',
        'get_vertex_color': 'in.vertex_color',
        'get_time': 'time_uniforms.elapsed_seconds',
    }

    # Output attribute mappings (out.X -> out.Y)
    OUT_ATTR_MAP = {
        'ao': 'out.ambient_occlusion',
        'albedo': 'out.base_color',
        'emission': 'out.emissive',
    }

    # Self attribute prefix mapping (self.X -> material.X)
    SELF_PREFIX = 'material'

    # Python to WGSL binary operator mappings
    BINOP_MAP = {
        'Add': '+',
        'Sub': '-',
        'Mult': '*',
        'Div': '/',
        'Mod': '%',
        'Pow': '',  # Special handling needed
        'FloorDiv': '/',  # Will need floor() wrapper
        'BitOr': '|',
        'BitXor': '^',
        'BitAnd': '&',
        'LShift': '<<',
        'RShift': '>>',
    }

    # Python to WGSL comparison operator mappings
    CMPOP_MAP = {
        'Eq': '==',
        'NotEq': '!=',
        'Lt': '<',
        'LtE': '<=',
        'Gt': '>',
        'GtE': '>=',
    }

    # Python to WGSL unary operator mappings
    UNARYOP_MAP = {
        'UAdd': '+',
        'USub': '-',
        'Not': '!',
        'Invert': '~',
    }

    # Custom builtin functions that need to be tracked for injection
    CUSTOM_BUILTINS = {
        # Noise builtins
        'value_noise', 'perlin_noise', 'simplex_noise', 'worley_noise',
        'fbm', 'turbulence',
        # Color builtins
        'rgb_to_hsv', 'hsv_to_rgb', 'srgb_to_linear', 'linear_to_srgb',
        'tonemap_reinhard', 'tonemap_aces', 'tonemap_uncharted2', 'tonemap_agx',
        # Math utility builtins
        'remap', 'inverse_lerp', 'smooth_min', 'smooth_max', 'smootherstep',
    }

    def __init__(self):
        self._indent = 0
        self.used_builtins: set = set()

    def translate(self, node: Any) -> str:
        """Translate a Python AST node to WGSL code.

        Args:
            node: Python AST node (Module, statement, or expression)

        Returns:
            WGSL code string

        Raises:
            WGSLTranslationError: If translation fails
        """
        import ast

        if isinstance(node, ast.Module):
            return self._translate_module(node)
        elif isinstance(node, ast.FunctionDef):
            return self._translate_function(node)
        elif isinstance(node, ast.Expr):
            return self._translate_expr(node)
        elif isinstance(node, ast.Assign):
            return self._translate_assign(node)
        elif isinstance(node, ast.AugAssign):
            return self._translate_aug_assign(node)
        elif isinstance(node, ast.AnnAssign):
            return self._translate_ann_assign(node)
        elif isinstance(node, ast.If):
            return self._translate_if(node)
        elif isinstance(node, ast.Return):
            return self._translate_return(node)
        elif isinstance(node, ast.Pass):
            return ""  # No-op, produces empty string
        else:
            return self._translate_expression(node)

    def _translate_module(self, node: Any) -> str:
        """Translate a module (list of statements)."""
        lines = []
        for stmt in node.body:
            lines.append(self.translate(stmt))
        return "\n".join(lines)

    def _translate_function(self, node: Any) -> str:
        """Translate a function definition."""
        # For now, just translate the body
        lines = []
        for stmt in node.body:
            lines.append(self.translate(stmt))
        return "\n".join(lines)

    def _translate_expr(self, node: Any) -> str:
        """Translate an expression statement."""
        expr = self._translate_expression(node.value)
        return f"{expr};"

    def _translate_assign(self, node: Any) -> str:
        """Translate an assignment statement."""
        import ast

        targets = []
        for target in node.targets:
            targets.append(self._translate_expression(target))

        value = self._translate_expression(node.value)
        target_str = targets[0] if len(targets) == 1 else ", ".join(targets)
        return f"{target_str} = {value};"

    def _translate_aug_assign(self, node: Any) -> str:
        """Translate an augmented assignment (+=, -=, etc.)."""
        import ast

        target = self._translate_expression(node.target)
        value = self._translate_expression(node.value)
        op = self.BINOP_MAP.get(type(node.op).__name__, '+')
        # Expand to full form: x = x + value (tests expect this)
        return f"{target} = {target} {op} {value};"

    def _translate_ann_assign(self, node: Any) -> str:
        """Translate an annotated assignment."""
        target = self._translate_expression(node.target)
        if node.value:
            value = self._translate_expression(node.value)
            return f"let {target} = {value};"
        # Skip declaration-only annotations (no value)
        return ""

    def _translate_if(self, node: Any) -> str:
        """Translate an if statement."""
        import ast

        cond = self._translate_expression(node.test)
        lines = [f"if ({cond}) {{"]

        for stmt in node.body:
            lines.append("    " + self.translate(stmt))

        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                # else if
                lines.append("} else " + self._translate_if(node.orelse[0]))
            else:
                lines.append("} else {")
                for stmt in node.orelse:
                    lines.append("    " + self.translate(stmt))
                lines.append("}")
        else:
            lines.append("}")

        return "\n".join(lines)

    def _translate_return(self, node: Any) -> str:
        """Translate a return statement."""
        if node.value:
            value = self._translate_expression(node.value)
            return f"return {value};"
        return "return;"

    def _translate_expression(self, node: Any) -> str:
        """Translate an expression to WGSL."""
        import ast

        if isinstance(node, ast.Constant):
            return self._translate_constant(node)
        elif isinstance(node, ast.Name):
            name = node.id
            # Map True/False to WGSL booleans
            if name == 'True':
                return 'true'
            elif name == 'False':
                return 'false'
            elif name == 'None':
                return '0.0'
            return name
        elif isinstance(node, ast.Attribute):
            return self._translate_attribute(node)
        elif isinstance(node, ast.BinOp):
            return self._translate_binop(node)
        elif isinstance(node, ast.UnaryOp):
            return self._translate_unaryop(node)
        elif isinstance(node, ast.Compare):
            return self._translate_compare(node)
        elif isinstance(node, ast.BoolOp):
            return self._translate_boolop(node)
        elif isinstance(node, ast.Call):
            return self._translate_call(node)
        elif isinstance(node, ast.Subscript):
            return self._translate_subscript(node)
        elif isinstance(node, ast.Tuple):
            return self._translate_tuple(node)
        elif isinstance(node, ast.List):
            return self._translate_list(node)
        elif isinstance(node, ast.IfExp):
            return self._translate_ifexp(node)
        else:
            raise WGSLTranslationError(
                f"Unsupported node type: {type(node).__name__}. "
                f"This construct is not supported in the material shader DSL.",
                node
            )

    def _translate_attribute(self, node: Any) -> str:
        """Translate an attribute access with special mappings."""
        import ast

        attr = node.attr

        # Handle special base objects
        if isinstance(node.value, ast.Name):
            base_name = node.value.id

            # ctx.X -> mapped value
            if base_name == 'ctx':
                if attr in self.CTX_ATTR_MAP:
                    # Track ctx.time usage for compatibility comments
                    if attr == 'time':
                        self.used_builtins.add('ctx.time')
                    return self.CTX_ATTR_MAP[attr]
                # Default: ctx.X -> in.X
                return f"in.{attr}"

            # out.X -> possibly mapped
            if base_name == 'out':
                if attr in self.OUT_ATTR_MAP:
                    return self.OUT_ATTR_MAP[attr]
                return f"out.{attr}"

            # self.X -> material.X
            if base_name == 'self':
                return f"{self.SELF_PREFIX}.{attr}"

        # Recursive translation for chained attributes
        value = self._translate_expression(node.value)
        return f"{value}.{attr}"

    def _translate_constant(self, node: Any) -> str:
        """Translate a constant value."""
        value = node.value
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, float):
            s = repr(value)
            if 'e' in s or 'E' in s:
                return s
            if '.' not in s:
                s = s + '.0'
            return s
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, str):
            return f'"{value}"'
        elif value is None:
            return "0.0"
        else:
            return str(value)

    def _translate_binop(self, node: Any) -> str:
        """Translate a binary operation."""
        import ast

        left = self._translate_expression(node.left)
        right = self._translate_expression(node.right)
        op_name = type(node.op).__name__

        if op_name == 'Pow':
            return f"pow({left}, {right})"
        elif op_name == 'FloorDiv':
            return f"floor({left} / {right})"

        op = self.BINOP_MAP.get(op_name, '+')
        return f"({left} {op} {right})"

    def _translate_unaryop(self, node: Any) -> str:
        """Translate a unary operation."""
        operand = self._translate_expression(node.operand)
        op = self.UNARYOP_MAP.get(type(node.op).__name__, '-')
        return f"({op}{operand})"

    def _translate_compare(self, node: Any) -> str:
        """Translate a comparison operation."""
        left = self._translate_expression(node.left)
        parts = [left]

        for op, comparator in zip(node.ops, node.comparators):
            op_str = self.CMPOP_MAP.get(type(op).__name__, '==')
            right = self._translate_expression(comparator)
            parts.append(op_str)
            parts.append(right)

        # For chained comparisons like a < b < c, we need to convert to (a < b) && (b < c)
        if len(node.ops) > 1:
            conditions = []
            prev = left
            for op, comparator in zip(node.ops, node.comparators):
                op_str = self.CMPOP_MAP.get(type(op).__name__, '==')
                right = self._translate_expression(comparator)
                conditions.append(f"({prev} {op_str} {right})")
                prev = right
            return " && ".join(conditions)

        return f"({parts[0]} {parts[1]} {parts[2]})"

    def _translate_boolop(self, node: Any) -> str:
        """Translate a boolean operation."""
        import ast

        op = "&&" if isinstance(node.op, ast.And) else "||"
        values = [self._translate_expression(v) for v in node.values]
        return f"({(' ' + op + ' ').join(values)})"

    # Texture sampling method mappings (ctx.sample() -> textureSample)
    TEXTURE_METHOD_MAP = {
        'sample': 'textureSample',
        'sample_level': 'textureSampleLevel',
        'sample_grad': 'textureSampleGrad',
        'sample_cube': 'textureSample',
        'load': 'textureLoad',
        'noise': 'noise',
    }

    def _translate_call(self, node: Any) -> str:
        """Translate a function/method call."""
        import ast

        # Handle method calls (obj.method())
        if isinstance(node.func, ast.Attribute):
            args = [self._translate_expression(arg) for arg in node.args]

            # Check for ctx method mappings
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'ctx':
                method = node.func.attr

                # Property-style methods that map to input attributes
                if method in self.CTX_METHOD_MAP:
                    return self.CTX_METHOD_MAP[method]

                # Texture sampling methods
                if method in self.TEXTURE_METHOD_MAP:
                    wgsl_func = self.TEXTURE_METHOD_MAP[method]
                    return f"{wgsl_func}({', '.join(args)})"

            obj = self._translate_expression(node.func.value)
            method = node.func.attr
            return f"{obj}.{method}({', '.join(args)})"

        # Handle function calls
        func_name = self._translate_expression(node.func)

        # Track custom builtins usage
        if func_name in self.CUSTOM_BUILTINS:
            self.used_builtins.add(func_name)

        # Map Python constructor names to WGSL
        if func_name in self.TYPE_MAP:
            func_name = self.TYPE_MAP[func_name]

        # Map Python function names to WGSL equivalents
        FUNC_MAP = {
            'lerp': 'mix',  # WGSL uses mix for linear interpolation
        }
        if func_name in FUNC_MAP:
            func_name = FUNC_MAP[func_name]

        args = [self._translate_expression(arg) for arg in node.args]
        return f"{func_name}({', '.join(args)})"

    def _translate_subscript(self, node: Any) -> str:
        """Translate a subscript (index access)."""
        value = self._translate_expression(node.value)
        slice_expr = self._translate_expression(node.slice)
        return f"{value}[{slice_expr}]"

    def _translate_tuple(self, node: Any) -> str:
        """Translate a tuple expression to WGSL vec type."""
        elements = [self._translate_expression(e) for e in node.elts]
        count = len(elements)
        if count == 2:
            return f"vec2<f32>({', '.join(elements)})"
        elif count == 3:
            return f"vec3<f32>({', '.join(elements)})"
        elif count == 4:
            return f"vec4<f32>({', '.join(elements)})"
        else:
            return f"({', '.join(elements)})"

    def _translate_list(self, node: Any) -> str:
        """Translate a list expression to WGSL vector.

        Lists with 2, 3, or 4 elements become vec2, vec3, vec4.
        Other sizes become array<f32, N>.
        """
        elements = [self._translate_expression(e) for e in node.elts]
        count = len(elements)
        if count == 2:
            return f"vec2<f32>({', '.join(elements)})"
        elif count == 3:
            return f"vec3<f32>({', '.join(elements)})"
        elif count == 4:
            return f"vec4<f32>({', '.join(elements)})"
        else:
            return f"array<f32, {count}>({', '.join(elements)})"

    def _translate_ifexp(self, node: Any) -> str:
        """Translate a ternary expression (a if cond else b)."""
        cond = self._translate_expression(node.test)
        body = self._translate_expression(node.body)
        orelse = self._translate_expression(node.orelse)
        return f"select({orelse}, {body}, {cond})"
