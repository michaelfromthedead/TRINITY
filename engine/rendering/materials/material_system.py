"""Core material management system.

This module provides the foundational material management infrastructure:
- MaterialTemplate: Base shader definition with parameter schema
- MaterialInstance: Override parameters for a template
- MaterialFunction: Reusable shader snippets
- MaterialLayer: Composable material stacking
- MaterialSystem: Resource managing all materials
"""
from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from weakref import WeakValueDictionary

from engine.core.math.vec import Vec2, Vec3, Vec4

__all__ = [
    "MaterialDomain",
    "BlendMode",
    "ShadingModel",
    "ParameterType",
    "MaterialParameter",
    "MaterialTemplate",
    "MaterialInstance",
    "MaterialFunction",
    "MaterialLayer",
    "MaterialSystem",
    "DirtyFlags",
]


class MaterialDomain(Enum):
    """Material pipeline domain types."""
    SURFACE = "surface"
    DEFERRED_DECAL = "deferred_decal"
    VOLUME = "volume"
    POST_PROCESS = "post_process"
    UI = "ui"


class BlendMode(Enum):
    """Material blend modes."""
    OPAQUE = "opaque"
    MASKED = "masked"
    TRANSLUCENT = "translucent"
    ADDITIVE = "additive"
    MODULATE = "modulate"


class ShadingModel(Enum):
    """Shading model types."""
    UNLIT = "unlit"
    DEFAULT_LIT = "default_lit"
    SUBSURFACE = "subsurface"
    CLEAR_COAT = "clear_coat"
    CLOTH = "cloth"
    HAIR = "hair"
    EYE = "eye"
    FOLIAGE = "foliage"


class ParameterType(Enum):
    """Material parameter types."""
    FLOAT = auto()
    VEC2 = auto()
    VEC3 = auto()
    VEC4 = auto()
    INT = auto()
    BOOL = auto()
    TEXTURE_2D = auto()
    TEXTURE_CUBE = auto()
    SAMPLER = auto()


@dataclass(slots=True)
class DirtyFlags:
    """Tracks which material properties need GPU re-upload."""
    parameters: bool = False
    textures: bool = False
    shader: bool = False

    def mark_all(self) -> None:
        """Mark all flags as dirty."""
        self.parameters = True
        self.textures = True
        self.shader = True

    def clear_all(self) -> None:
        """Clear all dirty flags."""
        self.parameters = False
        self.textures = False
        self.shader = False

    def any_dirty(self) -> bool:
        """Check if any flag is dirty."""
        return self.parameters or self.textures or self.shader


@dataclass(slots=True)
class MaterialParameter:
    """Definition of a material parameter.

    Attributes:
        name: Parameter name (used in shader binding)
        param_type: Type of the parameter
        default_value: Default value when not overridden
        min_value: Optional minimum value for numeric types
        max_value: Optional maximum value for numeric types
        description: Human-readable description
        group: UI grouping for editor
        hidden: Whether to hide in editor UI
    """
    name: str
    param_type: ParameterType
    default_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    description: str = ""
    group: str = "Default"
    hidden: bool = False

    def validate(self, value: Any) -> Tuple[bool, str]:
        """Validate a value against this parameter's constraints.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None:
            return False, f"Parameter {self.name} cannot be None"

        # Type validation
        expected_types = {
            ParameterType.FLOAT: (int, float),
            ParameterType.VEC2: Vec2,
            ParameterType.VEC3: Vec3,
            ParameterType.VEC4: Vec4,
            ParameterType.INT: int,
            ParameterType.BOOL: bool,
        }

        if self.param_type in expected_types:
            expected = expected_types[self.param_type]
            if not isinstance(value, expected):
                return False, (
                    f"Parameter {self.name} expected {expected}, "
                    f"got {type(value).__name__}"
                )

        # Range validation for numeric types
        if self.param_type in (ParameterType.FLOAT, ParameterType.INT):
            if self.min_value is not None and value < self.min_value:
                return False, (
                    f"Parameter {self.name} value {value} "
                    f"below minimum {self.min_value}"
                )
            if self.max_value is not None and value > self.max_value:
                return False, (
                    f"Parameter {self.name} value {value} "
                    f"above maximum {self.max_value}"
                )

        return True, ""

    def clamp(self, value: Any) -> Any:
        """Clamp value to parameter's valid range."""
        if self.param_type not in (ParameterType.FLOAT, ParameterType.INT):
            return value

        result = value
        if self.min_value is not None:
            result = max(result, self.min_value)
        if self.max_value is not None:
            result = min(result, self.max_value)
        return result


class MaterialTemplate:
    """Base shader definition with parameter schema.

    Templates define the shader code and parameter schema that can be
    instantiated into multiple MaterialInstances.

    Attributes:
        template_id: Unique identifier
        name: Human-readable name
        domain: Material domain (surface, volume, etc.)
        blend_mode: Blend mode
        shading_model: Shading model
        parameters: Parameter definitions
        vertex_shader: Vertex shader source or path
        fragment_shader: Fragment shader source or path
        functions: Referenced material functions
        two_sided: Whether to render both sides
        wireframe: Whether to render as wireframe
    """

    __slots__ = (
        "_template_id",
        "_name",
        "_domain",
        "_blend_mode",
        "_shading_model",
        "_parameters",
        "_vertex_shader",
        "_fragment_shader",
        "_functions",
        "_two_sided",
        "_wireframe",
        "_instances",
        "_version",
        "_compiled_variants",
    )

    def __init__(
        self,
        name: str,
        domain: MaterialDomain = MaterialDomain.SURFACE,
        blend_mode: BlendMode = BlendMode.OPAQUE,
        shading_model: ShadingModel = ShadingModel.DEFAULT_LIT,
        vertex_shader: Optional[str] = None,
        fragment_shader: Optional[str] = None,
        two_sided: bool = False,
        wireframe: bool = False,
    ) -> None:
        self._template_id = str(uuid.uuid4())
        self._name = name
        self._domain = domain
        self._blend_mode = blend_mode
        self._shading_model = shading_model
        self._parameters: Dict[str, MaterialParameter] = {}
        self._vertex_shader = vertex_shader
        self._fragment_shader = fragment_shader
        self._functions: List[MaterialFunction] = []
        self._two_sided = two_sided
        self._wireframe = wireframe
        self._instances: WeakValueDictionary[str, MaterialInstance] = (
            WeakValueDictionary()
        )
        self._version = 0
        self._compiled_variants: Dict[int, Any] = {}

    @property
    def template_id(self) -> str:
        return self._template_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def domain(self) -> MaterialDomain:
        return self._domain

    @property
    def blend_mode(self) -> BlendMode:
        return self._blend_mode

    @property
    def shading_model(self) -> ShadingModel:
        return self._shading_model

    @property
    def two_sided(self) -> bool:
        return self._two_sided

    @property
    def wireframe(self) -> bool:
        return self._wireframe

    @property
    def version(self) -> int:
        return self._version

    @property
    def parameters(self) -> Dict[str, MaterialParameter]:
        return self._parameters.copy()

    def add_parameter(self, param: MaterialParameter) -> None:
        """Add a parameter definition to this template."""
        if param.name in self._parameters:
            raise ValueError(f"Parameter {param.name} already exists")
        self._parameters[param.name] = param
        self._version += 1
        self._invalidate_variants()

    def remove_parameter(self, name: str) -> None:
        """Remove a parameter from this template."""
        if name not in self._parameters:
            raise KeyError(f"Parameter {name} not found")
        del self._parameters[name]
        self._version += 1
        self._invalidate_variants()

    def add_function(self, func: MaterialFunction) -> None:
        """Add a material function reference."""
        if func not in self._functions:
            self._functions.append(func)
            self._version += 1

    def create_instance(self, name: Optional[str] = None) -> MaterialInstance:
        """Create a new instance of this template."""
        instance = MaterialInstance(self, name)
        self._instances[instance.instance_id] = instance
        return instance

    def get_instances(self) -> List[MaterialInstance]:
        """Get all live instances of this template."""
        return list(self._instances.values())

    def get_default_values(self) -> Dict[str, Any]:
        """Get default values for all parameters."""
        return {
            name: param.default_value
            for name, param in self._parameters.items()
        }

    def compute_permutation_key(self, features: Set[str]) -> int:
        """Compute a hash key for shader permutation."""
        feature_str = "|".join(sorted(features))
        return hash(feature_str)

    def _invalidate_variants(self) -> None:
        """Invalidate all compiled shader variants."""
        self._compiled_variants.clear()
        for instance in self._instances.values():
            instance._dirty.shader = True

    def __repr__(self) -> str:
        return (
            f"<MaterialTemplate name={self._name!r} "
            f"domain={self._domain.value} blend={self._blend_mode.value}>"
        )


class MaterialInstance:
    """Instance of a MaterialTemplate with parameter overrides.

    Instances share the shader from their template but can override
    parameter values. Changes to parameters trigger dirty flags for
    efficient GPU re-upload.

    Attributes:
        template: Parent template
        instance_id: Unique identifier
        name: Optional instance name
        overrides: Parameter value overrides
        dirty: Dirty flags for GPU sync
    """

    __slots__ = (
        "_template",
        "_instance_id",
        "_name",
        "_overrides",
        "_dirty",
        "__weakref__",  # Allow weak references
        "_features",
        "_layer_stack",
    )

    def __init__(
        self,
        template: MaterialTemplate,
        name: Optional[str] = None,
    ) -> None:
        self._template = template
        self._instance_id = str(uuid.uuid4())
        self._name = name or f"Instance_{self._instance_id[:8]}"
        self._overrides: Dict[str, Any] = {}
        self._dirty = DirtyFlags()
        self._dirty.mark_all()  # New instance needs full upload
        self._features: Set[str] = set()
        self._layer_stack: List[MaterialLayer] = []

    @property
    def template(self) -> MaterialTemplate:
        return self._template

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def dirty(self) -> DirtyFlags:
        return self._dirty

    @property
    def features(self) -> Set[str]:
        return self._features.copy()

    def get_parameter(self, name: str) -> Any:
        """Get effective parameter value (override or default)."""
        if name in self._overrides:
            return self._overrides[name]

        param = self._template._parameters.get(name)
        if param is None:
            raise KeyError(f"Unknown parameter: {name}")
        return param.default_value

    def set_parameter(
        self,
        name: str,
        value: Any,
        validate: bool = True,
        clamp: bool = True,
    ) -> None:
        """Set a parameter override.

        Args:
            name: Parameter name
            value: New value
            validate: Whether to validate the value
            clamp: Whether to clamp numeric values to valid range
        """
        param = self._template._parameters.get(name)
        if param is None:
            raise KeyError(f"Unknown parameter: {name}")

        if clamp:
            value = param.clamp(value)

        if validate:
            is_valid, error = param.validate(value)
            if not is_valid:
                raise ValueError(error)

        # Check if value actually changed
        old_value = self._overrides.get(name)
        if old_value != value:
            self._overrides[name] = value

            # Mark appropriate dirty flag
            if param.param_type in (
                ParameterType.TEXTURE_2D,
                ParameterType.TEXTURE_CUBE,
            ):
                self._dirty.textures = True
            else:
                self._dirty.parameters = True

    def clear_override(self, name: str) -> None:
        """Clear a parameter override, reverting to template default."""
        if name in self._overrides:
            del self._overrides[name]
            self._dirty.parameters = True

    def clear_all_overrides(self) -> None:
        """Clear all parameter overrides."""
        if self._overrides:
            self._overrides.clear()
            self._dirty.mark_all()

    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all effective parameter values."""
        result = self._template.get_default_values()
        result.update(self._overrides)
        return result

    def enable_feature(self, feature: str) -> None:
        """Enable a shader feature for this instance."""
        if feature not in self._features:
            self._features.add(feature)
            self._dirty.shader = True

    def disable_feature(self, feature: str) -> None:
        """Disable a shader feature for this instance."""
        if feature in self._features:
            self._features.discard(feature)
            self._dirty.shader = True

    def push_layer(self, layer: MaterialLayer) -> None:
        """Push a material layer onto the stack."""
        self._layer_stack.append(layer)
        self._dirty.mark_all()

    def pop_layer(self) -> Optional[MaterialLayer]:
        """Pop and return the top material layer."""
        if self._layer_stack:
            layer = self._layer_stack.pop()
            self._dirty.mark_all()
            return layer
        return None

    def get_layers(self) -> List[MaterialLayer]:
        """Get all material layers."""
        return self._layer_stack.copy()

    def get_permutation_key(self) -> int:
        """Get shader permutation key for this instance."""
        return self._template.compute_permutation_key(self._features)

    def clone(self, new_name: Optional[str] = None) -> MaterialInstance:
        """Create a copy of this instance with same overrides."""
        new_instance = self._template.create_instance(new_name)
        new_instance._overrides = self._overrides.copy()
        new_instance._features = self._features.copy()
        new_instance._layer_stack = self._layer_stack.copy()
        return new_instance

    def __repr__(self) -> str:
        return (
            f"<MaterialInstance name={self._name!r} "
            f"template={self._template.name!r} "
            f"overrides={len(self._overrides)}>"
        )


class MaterialFunction:
    """Reusable shader snippet that can be included in materials.

    Functions encapsulate common shader logic (Fresnel, normal blending,
    parallax mapping, etc.) that can be reused across multiple materials.

    Attributes:
        function_id: Unique identifier
        name: Function name (used in shader code)
        description: Human-readable description
        inputs: Input parameter definitions
        outputs: Output parameter definitions
        code: Shader code implementing the function
    """

    __slots__ = (
        "_function_id",
        "_name",
        "_description",
        "_inputs",
        "_outputs",
        "_code",
        "_dependencies",
    )

    def __init__(
        self,
        name: str,
        code: str,
        description: str = "",
    ) -> None:
        self._function_id = str(uuid.uuid4())
        self._name = name
        self._description = description
        self._inputs: List[MaterialParameter] = []
        self._outputs: List[MaterialParameter] = []
        self._code = code
        self._dependencies: List[MaterialFunction] = []

    @property
    def function_id(self) -> str:
        return self._function_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def code(self) -> str:
        return self._code

    @property
    def inputs(self) -> List[MaterialParameter]:
        return self._inputs.copy()

    @property
    def outputs(self) -> List[MaterialParameter]:
        return self._outputs.copy()

    def add_input(self, param: MaterialParameter) -> None:
        """Add an input parameter."""
        self._inputs.append(param)

    def add_output(self, param: MaterialParameter) -> None:
        """Add an output parameter."""
        self._outputs.append(param)

    def add_dependency(self, func: MaterialFunction) -> None:
        """Add a function dependency."""
        if func not in self._dependencies:
            self._dependencies.append(func)

    def get_dependencies(self) -> List[MaterialFunction]:
        """Get all dependencies including transitive ones."""
        result: List[MaterialFunction] = []
        visited: Set[str] = set()

        def collect(fn: MaterialFunction) -> None:
            if fn.function_id in visited:
                return
            visited.add(fn.function_id)
            for dep in fn._dependencies:
                collect(dep)
            result.append(fn)

        for dep in self._dependencies:
            collect(dep)

        return result

    def get_full_code(self) -> str:
        """Get complete code including all dependencies."""
        deps = self.get_dependencies()
        parts = [dep.code for dep in deps]
        parts.append(self._code)
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return f"<MaterialFunction name={self._name!r}>"


@dataclass(slots=True)
class LayerBlendSettings:
    """Settings for how a material layer blends with layers below."""
    blend_weight: float = 1.0
    blend_mode: str = "lerp"  # "lerp", "add", "multiply", "overlay"
    mask_channel: Optional[str] = None  # "r", "g", "b", "a", or None


class MaterialLayer:
    """Composable layer that can be stacked on materials.

    Layers allow for complex material effects by stacking multiple
    material contributions (e.g., base material + dirt + scratches).

    Attributes:
        layer_id: Unique identifier
        name: Layer name
        parameters: Layer-specific parameters
        blend_settings: How this layer blends with below
        enabled: Whether layer is active
    """

    __slots__ = (
        "_layer_id",
        "_name",
        "_parameters",
        "_blend_settings",
        "_enabled",
        "_mask_texture",
    )

    def __init__(
        self,
        name: str,
        blend_settings: Optional[LayerBlendSettings] = None,
    ) -> None:
        self._layer_id = str(uuid.uuid4())
        self._name = name
        self._parameters: Dict[str, Any] = {}
        self._blend_settings = blend_settings or LayerBlendSettings()
        self._enabled = True
        self._mask_texture: Optional[str] = None

    @property
    def layer_id(self) -> str:
        return self._layer_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def blend_settings(self) -> LayerBlendSettings:
        return self._blend_settings

    def set_parameter(self, name: str, value: Any) -> None:
        """Set a layer parameter."""
        self._parameters[name] = value

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a layer parameter."""
        return self._parameters.get(name, default)

    def set_mask_texture(self, texture_path: Optional[str]) -> None:
        """Set the mask texture for this layer."""
        self._mask_texture = texture_path

    def __repr__(self) -> str:
        return (
            f"<MaterialLayer name={self._name!r} "
            f"enabled={self._enabled}>"
        )


class MaterialSystem:
    """Resource managing all materials in the engine.

    The MaterialSystem is the central registry for material templates
    and instances. It handles:
    - Template and instance registration
    - Hot-reload support
    - GPU resource management
    - Material function library

    Attributes:
        templates: Registered material templates
        instances: All material instances
        functions: Material function library
    """

    __slots__ = (
        "_templates",
        "_instances",
        "_functions",
        "_dirty_instances",
        "_hot_reload_enabled",
        "_on_template_change",
    )

    def __init__(self) -> None:
        self._templates: Dict[str, MaterialTemplate] = {}
        self._instances: Dict[str, MaterialInstance] = {}
        self._functions: Dict[str, MaterialFunction] = {}
        self._dirty_instances: Set[str] = set()
        self._hot_reload_enabled = False
        self._on_template_change: List[
            Callable[[MaterialTemplate], None]
        ] = []

    def register_template(self, template: MaterialTemplate) -> None:
        """Register a material template."""
        if template.template_id in self._templates:
            raise ValueError(
                f"Template {template.template_id} already registered"
            )
        self._templates[template.template_id] = template

    def unregister_template(self, template_id: str) -> None:
        """Unregister a material template."""
        if template_id not in self._templates:
            raise KeyError(f"Template {template_id} not found")

        template = self._templates[template_id]
        # Mark all instances as invalid
        for instance in template.get_instances():
            if instance.instance_id in self._instances:
                del self._instances[instance.instance_id]

        del self._templates[template_id]

    def get_template(self, template_id: str) -> Optional[MaterialTemplate]:
        """Get a template by ID."""
        return self._templates.get(template_id)

    def get_template_by_name(self, name: str) -> Optional[MaterialTemplate]:
        """Get a template by name."""
        for template in self._templates.values():
            if template.name == name:
                return template
        return None

    def register_instance(self, instance: MaterialInstance) -> None:
        """Register a material instance for tracking."""
        self._instances[instance.instance_id] = instance
        self._dirty_instances.add(instance.instance_id)

    def unregister_instance(self, instance_id: str) -> None:
        """Unregister a material instance."""
        self._instances.pop(instance_id, None)
        self._dirty_instances.discard(instance_id)

    def get_instance(self, instance_id: str) -> Optional[MaterialInstance]:
        """Get an instance by ID."""
        return self._instances.get(instance_id)

    def register_function(self, func: MaterialFunction) -> None:
        """Register a material function."""
        self._functions[func.name] = func

    def get_function(self, name: str) -> Optional[MaterialFunction]:
        """Get a material function by name."""
        return self._functions.get(name)

    def get_all_templates(self) -> List[MaterialTemplate]:
        """Get all registered templates."""
        return list(self._templates.values())

    def get_all_instances(self) -> List[MaterialInstance]:
        """Get all registered instances."""
        return list(self._instances.values())

    def get_dirty_instances(self) -> List[MaterialInstance]:
        """Get all instances with dirty flags set."""
        result = []
        for instance_id in list(self._dirty_instances):
            instance = self._instances.get(instance_id)
            if instance and instance.dirty.any_dirty():
                result.append(instance)
            else:
                self._dirty_instances.discard(instance_id)
        return result

    def mark_instance_dirty(self, instance_id: str) -> None:
        """Mark an instance as needing GPU update."""
        self._dirty_instances.add(instance_id)

    def clear_dirty_flags(self) -> None:
        """Clear dirty flags for all instances after GPU sync."""
        for instance_id in list(self._dirty_instances):
            instance = self._instances.get(instance_id)
            if instance:
                instance.dirty.clear_all()
        self._dirty_instances.clear()

    def enable_hot_reload(self) -> None:
        """Enable hot-reload support for shaders."""
        self._hot_reload_enabled = True

    def disable_hot_reload(self) -> None:
        """Disable hot-reload support."""
        self._hot_reload_enabled = False

    def on_template_changed(
        self,
        callback: Callable[[MaterialTemplate], None],
    ) -> None:
        """Register callback for template changes (hot-reload)."""
        self._on_template_change.append(callback)

    def notify_template_changed(self, template: MaterialTemplate) -> None:
        """Notify listeners of template change."""
        for callback in self._on_template_change:
            callback(template)

        # Mark all instances of this template dirty
        for instance in template.get_instances():
            instance.dirty.shader = True
            self._dirty_instances.add(instance.instance_id)

    def create_template(
        self,
        name: str,
        domain: MaterialDomain = MaterialDomain.SURFACE,
        blend_mode: BlendMode = BlendMode.OPAQUE,
        shading_model: ShadingModel = ShadingModel.DEFAULT_LIT,
        **kwargs: Any,
    ) -> MaterialTemplate:
        """Create and register a new template."""
        template = MaterialTemplate(
            name=name,
            domain=domain,
            blend_mode=blend_mode,
            shading_model=shading_model,
            **kwargs,
        )
        self.register_template(template)
        return template

    def create_instance(
        self,
        template: Union[str, MaterialTemplate],
        name: Optional[str] = None,
    ) -> MaterialInstance:
        """Create a new instance from a template."""
        if isinstance(template, str):
            tmpl = self.get_template(template)
            if tmpl is None:
                tmpl = self.get_template_by_name(template)
            if tmpl is None:
                raise KeyError(f"Template not found: {template}")
            template = tmpl

        instance = template.create_instance(name)
        self.register_instance(instance)
        return instance

    def __repr__(self) -> str:
        return (
            f"<MaterialSystem templates={len(self._templates)} "
            f"instances={len(self._instances)}>"
        )
