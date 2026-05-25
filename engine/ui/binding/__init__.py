"""
UI Data Binding System.

Provides reactive data binding for MVVM-style UI development.

Core Components:
- Binding: One-way, two-way, one-time bindings between source and target
- MultiBinding: Multiple sources to single target
- PropertyPath: Nested property access (e.g., "address.city")
- BindingContext: Shared resources and default converters
- BindingExpression: Computed bindings from multiple properties

Value Converters:
- IConverter: Interface for synchronous converters
- IAsyncConverter: Interface for async converters
- MultiValueConverter: Combine multiple values
- Built-in converters for common transformations

Validation:
- IValidator: Interface for validators
- IAsyncValidator: Interface for async validators
- ValidationContext: Manage validation across fields
- Built-in validators for common rules

Observable Collections:
- ObservableList: List with change notifications
- ObservableDict: Dict with change notifications
- VirtualizedListView: Efficient virtualization for large lists

Example:
    from engine.ui.binding import (
        Binding, BindingMode, PropertyPath,
        NumberFormatConverter, required, range_validator,
    )

    # Simple one-way binding
    binding = Binding(
        source=player,
        source_path="health",
        target=health_bar,
        target_path="value",
    )
    binding.attach()

    # Two-way binding with validation
    binding = Binding(
        source=settings,
        source_path="volume",
        target=slider,
        target_path="value",
        mode=BindingMode.TWO_WAY,
        validators=[range_validator(0, 100)],
    )
    binding.attach()
"""

# Core binding system
from .binding import (
    # Enums
    BindingMode,
    UpdateSourceTrigger,
    BindingStatus,
    # Core types
    BindingError,
    PropertyPath,
    BindingContext,
    BindingExpression,
    # Base class
    BindingBase,
    # Main binding classes
    Binding,
    MultiBinding,
    BindingGroup,
    # Factory functions
    bind,
    bind_two_way,
    bind_one_time,
    multi_bind,
)

# Value converters
from .converter import (
    # Interfaces
    IConverter,
    IAsyncConverter,
    # Types
    Color,
    Visibility,
    # Built-in converters
    BoolToVisibilityConverter,
    BoolToStringConverter,
    NumberFormatConverter,
    IntegerFormatConverter,
    StringFormatConverter,
    ColorConverter,
    ColorToRgbaConverter,
    InverseBoolConverter,
    NullToBoolConverter,
    EnumToStringConverter,
    PercentageConverter,
    DateTimeFormatConverter,
    # Advanced converters
    ChainedConverter,
    LambdaConverter,
    AsyncLambdaConverter,
    CachedConverter,
    # Multi-value converters
    MultiValueConverter,
    StringConcatConverter,
    MathOperationConverter,
    # Factory functions
    bool_to_visibility,
    number_format,
    string_format,
    color_converter,
    chain,
)

# Validation
from .validation import (
    # Interfaces
    IValidator,
    IAsyncValidator,
    # Types
    ValidationResult,
    ValidationTrigger,
    ValidationSeverity,
    # Built-in validators
    RequiredValidator,
    RangeValidator,
    RegexValidator,
    LengthValidator,
    EmailValidator,
    UrlValidator,
    ChoiceValidator,
    TypeValidator,
    CompareValidator,
    # Custom validators
    CustomValidator,
    AsyncCustomValidator,
    # Composite
    CompositeValidator,
    # Context
    ValidationContext,
    # Factory functions
    required,
    range_validator,
    regex,
    length,
    email,
    custom,
    all_of,
    any_of,
)

# Observable collections
from .observable import (
    CollectionChangeAction,
    CollectionChangeCallback,
    CollectionChangeEvent,
    IObservableCollection,
    ObservableDict,
    ObservableList,
    VirtualizedListView,
)


__all__ = [
    # === Binding Core ===
    # Enums
    "BindingMode",
    "UpdateSourceTrigger",
    "BindingStatus",
    # Core types
    "BindingError",
    "PropertyPath",
    "BindingContext",
    "BindingExpression",
    # Base class
    "BindingBase",
    # Main binding classes
    "Binding",
    "MultiBinding",
    "BindingGroup",
    # Factory functions
    "bind",
    "bind_two_way",
    "bind_one_time",
    "multi_bind",
    # === Converters ===
    # Interfaces
    "IConverter",
    "IAsyncConverter",
    # Types
    "Color",
    "Visibility",
    # Built-in converters
    "BoolToVisibilityConverter",
    "BoolToStringConverter",
    "NumberFormatConverter",
    "IntegerFormatConverter",
    "StringFormatConverter",
    "ColorConverter",
    "ColorToRgbaConverter",
    "InverseBoolConverter",
    "NullToBoolConverter",
    "EnumToStringConverter",
    "PercentageConverter",
    "DateTimeFormatConverter",
    # Advanced converters
    "ChainedConverter",
    "LambdaConverter",
    "AsyncLambdaConverter",
    "CachedConverter",
    # Multi-value converters
    "MultiValueConverter",
    "StringConcatConverter",
    "MathOperationConverter",
    # Factory functions
    "bool_to_visibility",
    "number_format",
    "string_format",
    "color_converter",
    "chain",
    # === Validation ===
    # Interfaces
    "IValidator",
    "IAsyncValidator",
    # Types
    "ValidationResult",
    "ValidationTrigger",
    "ValidationSeverity",
    # Built-in validators
    "RequiredValidator",
    "RangeValidator",
    "RegexValidator",
    "LengthValidator",
    "EmailValidator",
    "UrlValidator",
    "ChoiceValidator",
    "TypeValidator",
    "CompareValidator",
    # Custom validators
    "CustomValidator",
    "AsyncCustomValidator",
    # Composite
    "CompositeValidator",
    # Context
    "ValidationContext",
    # Factory functions
    "required",
    "range_validator",
    "regex",
    "length",
    "email",
    "custom",
    "all_of",
    "any_of",
    # === Observable Collections ===
    "CollectionChangeAction",
    "CollectionChangeCallback",
    "CollectionChangeEvent",
    "IObservableCollection",
    "ObservableDict",
    "ObservableList",
    "VirtualizedListView",
]
