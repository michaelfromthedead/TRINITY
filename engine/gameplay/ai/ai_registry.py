"""
AI Registry - Decorators and utilities for registering AI types with Foundation Registry.

This module provides decorators for registering AI node types (behavior tree nodes,
GOAP actions, utility considerations) with the Foundation Registry for runtime discovery.

Usage:
    from engine.gameplay.ai.ai_registry import bt_node, goap_action, consideration

    @bt_node(type="action")
    class AttackAction(BTNode):
        pass

    @goap_action(preconditions=["has_weapon"], effects=["target_damaged"])
    class AttackGOAPAction(GOAPAction):
        pass

    @consideration(curve="linear")
    class HealthConsideration(Consideration):
        pass
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Type, TypeVar, Union

from foundation import registry, Registry

# Type variable for decorator return types
T = TypeVar("T", bound=type)

# Tag constants for AI types
TAG_BEHAVIOR_TREE = "behavior_tree"
TAG_BT_NODE = "bt_node"
TAG_GOAP_ACTION = "goap_action"
TAG_CONSIDERATION = "consideration"

# Valid BT node types
VALID_BT_NODE_TYPES = frozenset({
    "selector",
    "sequence",
    "parallel",
    "action",
    "condition",
    "decorator",
    "invert",
    "repeat",
    "timeout",
    "cooldown",
    "retry",
    "force_success",
    "force_failure",
    "composite",
    "leaf",
})

# Valid consideration curve types
VALID_CURVE_TYPES = frozenset({
    "linear",
    "quadratic",
    "exponential",
    "logistic",
    "sine",
    "inverse",
    "step",
    "smoothstep",
    "sigmoid",
    "custom",
})


def behavior_tree(
    name: str,
    description: Optional[str] = None,
    track_instances: bool = False,
) -> Callable[[T], T]:
    """
    Decorator to register a class as a behavior tree definition with the Foundation Registry.

    This decorator:
    1. Registers the BT class with the Foundation Registry
    2. Tags it as "behavior_tree"
    3. Stores metadata (name, description, node_count)

    Args:
        name: Unique name for the behavior tree
        description: Human-readable description of what this BT does
        track_instances: If True, track all instances via WeakSet

    Returns:
        Decorated class registered with Foundation Registry

    Example:
        @behavior_tree(name="patrol", description="Patrol AI behavior")
        class PatrolBehavior:
            @classmethod
            def create_root(cls) -> BTNode:
                return Sequence([MoveToWaypoint(), Wait(1.0)])

        # Query all behavior trees:
        >>> from foundation import registry
        >>> registry.query(tag="behavior_tree")
    """
    def decorator(cls: T) -> T:
        # Mark class attributes for BT identification
        cls._behavior_tree = True
        cls._bt_name = name
        cls._bt_description = description or ""

        # Count nodes if the class has a method to do so
        node_count = 0
        if hasattr(cls, "_count_nodes"):
            try:
                node_count = cls._count_nodes()
            except Exception:
                pass

        # Register with Foundation Registry
        registry_name = f"bt.{name}"
        try:
            registry.register(cls, name=registry_name, track_instances=track_instances)
        except ValueError:
            # Already registered - fine in reload scenarios
            pass

        # Add tag for query-based discovery
        registry.add_tag(cls, TAG_BEHAVIOR_TREE)

        # Store metadata
        registry.set_metadata(cls, "bt_name", name)
        registry.set_metadata(cls, "description", description or "")
        registry.set_metadata(cls, "node_count", node_count)

        return cls

    return decorator


def bt_node(
    type: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    track_instances: bool = False,
) -> Callable[[T], T]:
    """
    Decorator to register a class as a behavior tree node with the Foundation Registry.

    Args:
        type: The node type (selector, sequence, action, condition, decorator, etc.)
        name: Optional custom registry name. Defaults to module.classname.
        description: Optional description of the node's purpose.
        track_instances: If True, track all instances via WeakSet.

    Returns:
        Decorated class registered with the Foundation Registry.

    Example:
        @bt_node(type="action")
        class AttackAction(BTNode):
            pass

        # Query all BT nodes:
        >>> from foundation import registry
        >>> registry.query(tag="bt_node")

        # Query action nodes only:
        >>> registry.query(tag="bt_node", node_type="action")
    """
    if type not in VALID_BT_NODE_TYPES:
        valid_types = ", ".join(sorted(VALID_BT_NODE_TYPES))
        raise ValueError(f"Invalid BT node type '{type}'. Valid types: {valid_types}")

    def decorator(cls: T) -> T:
        # Register with Foundation Registry
        registry.register(cls, name=name, track_instances=track_instances)

        # Add BT node tag
        registry.add_tag(cls, TAG_BT_NODE)

        # Add metadata
        registry.set_metadata(cls, "node_type", type)
        if description:
            registry.set_metadata(cls, "description", description)

        # Store decorator info on class for introspection
        cls._bt_node = True
        cls._bt_node_type = type
        cls._bt_description = description

        return cls

    return decorator


def goap_action(
    preconditions: Optional[Sequence[str]] = None,
    effects: Optional[Sequence[str]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    cost: Optional[float] = None,
    track_instances: bool = False,
) -> Callable[[T], T]:
    """
    Decorator to register a class as a GOAP action with the Foundation Registry.

    Args:
        preconditions: List of world state conditions required to execute this action.
        effects: List of world state changes produced by this action.
        name: Optional custom registry name. Defaults to module.classname.
        description: Optional description of the action's purpose.
        cost: Default cost for this action type (can be overridden per instance).
        track_instances: If True, track all instances via WeakSet.

    Returns:
        Decorated class registered with the Foundation Registry.

    Example:
        @goap_action(preconditions=["has_weapon"], effects=["target_damaged"])
        class AttackAction(GOAPAction):
            pass

        # Query all GOAP actions:
        >>> from foundation import registry
        >>> registry.query(tag="goap_action")

        # Query by effect:
        >>> registry.query(tag="goap_action", effects="target_damaged")
    """
    preconds = frozenset(preconditions) if preconditions else frozenset()
    effs = frozenset(effects) if effects else frozenset()

    def decorator(cls: T) -> T:
        # Register with Foundation Registry
        registry.register(cls, name=name, track_instances=track_instances)

        # Add GOAP action tag
        registry.add_tag(cls, TAG_GOAP_ACTION)

        # Add metadata
        registry.set_metadata(cls, "preconditions", preconds)
        registry.set_metadata(cls, "effects", effs)
        if description:
            registry.set_metadata(cls, "description", description)
        if cost is not None:
            registry.set_metadata(cls, "default_cost", cost)

        # Store decorator info on class for introspection
        cls._goap_action = True
        cls._goap_preconditions = preconds
        cls._goap_effects = effs
        cls._goap_description = description
        cls._goap_default_cost = cost

        return cls

    return decorator


def consideration(
    curve: str = "linear",
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    weight: Optional[float] = None,
    track_instances: bool = False,
) -> Callable[[T], T]:
    """
    Decorator to register a class as a utility AI consideration with the Foundation Registry.

    Args:
        curve: The response curve type (linear, exponential, sigmoid, etc.)
        name: Optional custom registry name. Defaults to module.classname.
        description: Optional description of what this consideration evaluates.
        weight: Default weight for this consideration type.
        track_instances: If True, track all instances via WeakSet.

    Returns:
        Decorated class registered with the Foundation Registry.

    Example:
        @consideration(curve="exponential")
        class HealthConsideration(Consideration):
            pass

        # Query all considerations:
        >>> from foundation import registry
        >>> registry.query(tag="consideration")

        # Query by curve type:
        >>> registry.query(tag="consideration", curve_type="exponential")
    """
    if curve not in VALID_CURVE_TYPES:
        valid_curves = ", ".join(sorted(VALID_CURVE_TYPES))
        raise ValueError(f"Invalid curve type '{curve}'. Valid types: {valid_curves}")

    def decorator(cls: T) -> T:
        # Register with Foundation Registry
        registry.register(cls, name=name, track_instances=track_instances)

        # Add consideration tag
        registry.add_tag(cls, TAG_CONSIDERATION)

        # Add metadata
        registry.set_metadata(cls, "curve_type", curve)
        if description:
            registry.set_metadata(cls, "description", description)
        if weight is not None:
            registry.set_metadata(cls, "default_weight", weight)

        # Store decorator info on class for introspection
        cls._consideration = True
        cls._consideration_curve = curve
        cls._consideration_description = description
        cls._consideration_default_weight = weight

        return cls

    return decorator


# =============================================================================
# Query Helpers
# =============================================================================


def get_all_behavior_trees() -> list[type]:
    """Get all registered behavior tree definitions."""
    return registry.query(tag=TAG_BEHAVIOR_TREE)


def get_all_bt_nodes() -> list[type]:
    """Get all registered behavior tree node types."""
    return registry.query(tag=TAG_BT_NODE)


def get_bt_nodes_by_type(node_type: str) -> list[type]:
    """Get all registered behavior tree nodes of a specific type."""
    return registry.query(tag=TAG_BT_NODE, node_type=node_type)


def get_all_goap_actions() -> list[type]:
    """Get all registered GOAP action types."""
    return registry.query(tag=TAG_GOAP_ACTION)


def get_goap_actions_by_effect(effect: str) -> list[type]:
    """Get all GOAP actions that produce a specific effect."""
    return registry.query(tag=TAG_GOAP_ACTION, effects=effect)


def get_goap_actions_by_precondition(precondition: str) -> list[type]:
    """Get all GOAP actions that require a specific precondition."""
    return registry.query(tag=TAG_GOAP_ACTION, preconditions=precondition)


def get_all_considerations() -> list[type]:
    """Get all registered utility AI consideration types."""
    return registry.query(tag=TAG_CONSIDERATION)


def get_considerations_by_curve(curve_type: str) -> list[type]:
    """Get all considerations using a specific curve type."""
    return registry.query(tag=TAG_CONSIDERATION, curve_type=curve_type)


# =============================================================================
# Factory Functions
# =============================================================================


def create_bt_node_from_registry(
    node_type_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Create a behavior tree node instance from the registry.

    Args:
        node_type_name: The registered name of the node type.
        *args: Positional arguments to pass to the constructor.
        **kwargs: Keyword arguments to pass to the constructor.

    Returns:
        A new instance of the requested node type.

    Raises:
        ValueError: If the node type is not found in the registry.
    """
    cls = registry.get(node_type_name)
    if cls is None:
        raise ValueError(f"BT node type '{node_type_name}' not found in registry")
    if not registry.has_tag(cls, TAG_BT_NODE):
        raise ValueError(f"Type '{node_type_name}' is not a registered BT node")
    return cls(*args, **kwargs)


def create_goap_action_from_registry(
    action_type_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Create a GOAP action instance from the registry.

    Args:
        action_type_name: The registered name of the action type.
        *args: Positional arguments to pass to the constructor.
        **kwargs: Keyword arguments to pass to the constructor.

    Returns:
        A new instance of the requested action type.

    Raises:
        ValueError: If the action type is not found in the registry.
    """
    cls = registry.get(action_type_name)
    if cls is None:
        raise ValueError(f"GOAP action type '{action_type_name}' not found in registry")
    if not registry.has_tag(cls, TAG_GOAP_ACTION):
        raise ValueError(f"Type '{action_type_name}' is not a registered GOAP action")
    return cls(*args, **kwargs)


def create_consideration_from_registry(
    consideration_type_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Create a consideration instance from the registry.

    Args:
        consideration_type_name: The registered name of the consideration type.
        *args: Positional arguments to pass to the constructor.
        **kwargs: Keyword arguments to pass to the constructor.

    Returns:
        A new instance of the requested consideration type.

    Raises:
        ValueError: If the consideration type is not found in the registry.
    """
    cls = registry.get(consideration_type_name)
    if cls is None:
        raise ValueError(f"Consideration type '{consideration_type_name}' not found in registry")
    if not registry.has_tag(cls, TAG_CONSIDERATION):
        raise ValueError(f"Type '{consideration_type_name}' is not a registered consideration")
    return cls(*args, **kwargs)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Decorators
    "behavior_tree",
    "bt_node",
    "goap_action",
    "consideration",
    # Query helpers
    "get_all_behavior_trees",
    "get_all_bt_nodes",
    "get_bt_nodes_by_type",
    "get_all_goap_actions",
    "get_goap_actions_by_effect",
    "get_goap_actions_by_precondition",
    "get_all_considerations",
    "get_considerations_by_curve",
    # Factory functions
    "create_bt_node_from_registry",
    "create_goap_action_from_registry",
    "create_consideration_from_registry",
    # Constants
    "TAG_BEHAVIOR_TREE",
    "TAG_BT_NODE",
    "TAG_GOAP_ACTION",
    "TAG_CONSIDERATION",
    "VALID_BT_NODE_TYPES",
    "VALID_CURVE_TYPES",
]
