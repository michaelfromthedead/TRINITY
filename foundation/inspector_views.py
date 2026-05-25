"""
Inspector Views - History and Causality views for entity inspection.

Part of Core Foundation Layer 3. Provides specialized views for
inspecting entity history and causal chains using EventLog data.

Views:
    - HistoryView: Shows all changes to an entity over time
    - CausalityView: Shows causal chains - why things happened
    - RootCauseSummary: Aggregate root cause analysis

Usage:
    from foundation.inspector_views import register_inspector_views
    register_inspector_views()  # Registers views with global inspector

    # Or use views directly
    from foundation.inspector_views import HistoryView, CausalityView
    panel = inspector.inspect(entity)
    panel.set_view("History")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from foundation.eventlog import get_event_log, Event, EventLog
from foundation.constants import HISTORY_FIELD_WIDTH, HISTORY_TICK_WIDTH, INDENT_SPACES
from foundation.provenance import (
    provenance,
    derivation_tree,
    all_provenance,
    ComputedProvenance,
    DerivationNode,
)

if TYPE_CHECKING:
    from foundation.inspector import UIContext


def _collect_events_for_entity(entity_id: int, log: Optional[EventLog] = None) -> list[Event]:
    """
    Collect all events relevant to an entity.

    Finds events where the entity is the subject AND events where
    the entity appears in the changes list.

    Args:
        entity_id: The entity ID to collect events for.
        log: Optional EventLog instance. Uses global log if not provided.

    Returns:
        List of events related to the entity (deduplicated).
    """
    if log is None:
        log = get_event_log()

    # Get events where entity is the subject
    events = log.events_for_entity(entity_id)

    # Also find events where this entity has changes
    all_events = log.all_events()
    events_with_changes = []
    for event in all_events:
        for change in event.changes:
            if change.entity == entity_id:
                if event not in events_with_changes:
                    events_with_changes.append(event)

    # Merge and deduplicate
    combined_events = list(events)
    for event in events_with_changes:
        if event not in combined_events:
            combined_events.append(event)

    return combined_events


class HistoryView:
    """
    Shows all changes to an entity over time.

    Display format:
    tick 5030 | health       | 45 -> 0  | Trap.trigger |
    tick 5012 | health       | 70 -> 45 | Enemy.attack |

    The view displays changes in reverse chronological order (most recent first).
    Each line shows the tick, field name, value transition, and operation name.
    """

    name: str = "History"

    def can_render(self, obj: Any) -> bool:
        """
        Check if this view can render the given object.

        Returns True for any object that has an 'id' attribute,
        as these can potentially have history in the event log.
        """
        return hasattr(obj, 'id')

    def render(self, obj: Any, ctx: 'UIContext') -> str:
        """
        Render history for an entity.

        Args:
            obj: The entity to show history for. Must have an 'id' attribute.
            ctx: The UI context to render into.

        Returns:
            The rendered output string (from ctx.get_output() if available).
        """
        entity_id = getattr(obj, 'id', id(obj))
        combined_events = _collect_events_for_entity(entity_id)

        if not combined_events:
            ctx.text("No history recorded for this entity.")
            return ctx.get_output() if hasattr(ctx, 'get_output') else ""

        # Sort by tick descending (most recent first)
        combined_events.sort(key=lambda e: e.tick, reverse=True)

        # Render as table
        ctx.text("=== Entity History ===")
        for event in combined_events:
            for change in event.changes:
                if change.entity == entity_id:
                    ctx.text(
                        f"tick {event.tick:{HISTORY_TICK_WIDTH}d} | "
                        f"{change.field:{HISTORY_FIELD_WIDTH}s} | "
                        f"{change.old_value} -> {change.new_value} | "
                        f"{event.operation}"
                    )

        return ctx.get_output() if hasattr(ctx, 'get_output') else ""


class CausalityView:
    """
    Shows causal chains - why things happened.

    Display format:
    tick 5030: health 45 -> 0
      +-- operation: Trap.trigger
      +-- direct cause: trap_1 (Entity 23)
      +-- root cause: Monster_G.think (Entity 8)
      +-- depth: 3 operations

    This view helps debug by showing the chain of causation:
    - What operation directly caused a change
    - What was the original root cause in the chain
    - How deep the call chain was
    """

    name: str = "Causality"

    def can_render(self, obj: Any) -> bool:
        """
        Check if this view can render the given object.

        Returns True for any object that has an 'id' attribute.
        """
        return hasattr(obj, 'id')

    def render(self, obj: Any, ctx: 'UIContext') -> str:
        """
        Render causality analysis for an entity.

        Shows each change with its causal chain information including
        the direct cause (immediate parent) and root cause.

        Args:
            obj: The entity to analyze causality for.
            ctx: The UI context to render into.

        Returns:
            The rendered output string.
        """
        entity_id = getattr(obj, 'id', id(obj))
        combined_events = _collect_events_for_entity(entity_id)

        if not combined_events:
            ctx.text("No causal data recorded for this entity.")
            return ctx.get_output() if hasattr(ctx, 'get_output') else ""

        # Sort by tick ascending (chronological order for causality)
        combined_events.sort(key=lambda e: e.tick)

        ctx.text("=== Causality Analysis ===")
        for event in combined_events:
            if event.changes:
                for change in event.changes:
                    if change.entity == entity_id:
                        ctx.text("")
                        ctx.text(
                            f"tick {event.tick}: {change.field} "
                            f"{change.old_value} -> {change.new_value}"
                        )
                        ctx.text(f"  +-- operation: {event.operation}")

                        if event.immediate_parent:
                            parent_str = f"  +-- direct cause: {event.immediate_parent}"
                            if event.immediate_parent_entity is not None:
                                parent_str += f" (Entity {event.immediate_parent_entity})"
                            ctx.text(parent_str)

                        if event.root_cause:
                            root_str = f"  +-- root cause: {event.root_cause}"
                            if event.root_cause_entity is not None:
                                root_str += f" (Entity {event.root_cause_entity})"
                            ctx.text(root_str)

                        ctx.text(f"  +-- depth: {event.depth} operations")

        return ctx.get_output() if hasattr(ctx, 'get_output') else ""


class ProvenanceView:
    """
    Shows provenance and derivation tree for computed fields.

    Display format:
    === Provenance for Entity (id: 12345) ===

    threat_level = 75
      computed by: Player.threat_level
      at tick: 5030
      inputs:
        nearby_enemies: [3, 7]
      derivation tree:
        threat_level = 75
          +-- damage = 50 (from Enemy)
          +-- damage = 25 (from Enemy)

    This view helps understand what data contributed to computed values
    and trace the full derivation chain.
    """

    name: str = "Provenance"

    def can_render(self, obj: Any) -> bool:
        """
        Check if this view can render the given object.

        Returns True for any object that has provenance data stored.
        """
        obj_id = id(obj)
        prov_data = all_provenance()
        for (stored_id, _field), _prov in prov_data.items():
            if stored_id == obj_id:
                return True
        return False

    def render(self, obj: Any, ctx: 'UIContext') -> str:
        """
        Render provenance for all computed fields on an entity.

        Args:
            obj: The entity to show provenance for.
            ctx: The UI context to render into.

        Returns:
            The rendered output string.
        """
        obj_id = id(obj)
        obj_type = type(obj).__name__

        # Find all provenance for this object
        prov_data = all_provenance()
        obj_provenance: list[tuple[str, ComputedProvenance]] = []
        for (stored_id, field_name), prov in prov_data.items():
            if stored_id == obj_id:
                obj_provenance.append((field_name, prov))

        if not obj_provenance:
            ctx.text(f"No provenance recorded for {obj_type}.")
            return ctx.get_output() if hasattr(ctx, 'get_output') else ""

        # Sort by field name for consistent output
        obj_provenance.sort(key=lambda x: x[0])

        ctx.text(f"=== Provenance for {obj_type} (id: {obj_id}) ===")
        ctx.text("")

        for field_name, prov in obj_provenance:
            ctx.text(f"{field_name} = {prov.value}")
            ctx.text(f"  computed by: {prov.computed_by}")
            ctx.text(f"  at tick: {prov.tick}")

            # Show recorded inputs
            if prov.input_summary:
                ctx.text("  inputs:")
                for input_name, input_value in prov.input_summary.items():
                    ctx.text(f"    {input_name}: {input_value}")

            # Show recorded reads
            if prov.reads:
                ctx.text("  reads:")
                for read in prov.reads:
                    ctx.text(f"    {read.field} = {read.value} (from {read.obj_type})")

            # Show derivation tree
            tree = derivation_tree(obj, field_name)
            if tree and tree.children:
                ctx.text("  derivation tree:")
                self._render_tree(tree, ctx, indent=2)

            ctx.text("")

        return ctx.get_output() if hasattr(ctx, 'get_output') else ""

    def _render_tree(
        self,
        node: DerivationNode,
        ctx: 'UIContext',
        indent: int = 0,
        is_root: bool = True
    ) -> None:
        """
        Recursively render the derivation tree.

        Args:
            node: The current node to render.
            ctx: The UI context to render into.
            indent: Current indentation level.
            is_root: Whether this is the root node.
        """
        prefix = " " * (indent * INDENT_SPACES)
        if is_root:
            ctx.text(f"{prefix}{node.field} = {node.value}")
        else:
            source = f" (from {node.source_obj_type})" if node.source_obj_type else ""
            ctx.text(f"{prefix}+-- {node.field} = {node.value}{source}")

        for child in node.children:
            self._render_tree(child, ctx, indent + 1, is_root=False)


@dataclass
class RootCauseSummary:
    """
    Aggregate root cause analysis for an entity.

    Provides summary statistics about what root causes have affected
    an entity, useful for understanding systemic patterns.

    Attributes:
        root_cause: The operation name that was the root cause.
        root_cause_entity: The entity ID of the root cause (if any).
        total_events: Number of events with this root cause.
        total_changes: Total number of field changes caused.
        affected_entities: Set of entity IDs affected by this root cause.
    """

    root_cause: str
    root_cause_entity: Optional[int]
    total_events: int
    total_changes: int
    affected_entities: set[int]

    @classmethod
    def for_entity(
        cls,
        entity_id: int,
        log: Optional[EventLog] = None
    ) -> list['RootCauseSummary']:
        """
        Get root cause summaries for events affecting an entity.

        Analyzes all events related to an entity and groups them by
        root cause, providing aggregate statistics for each.

        Args:
            entity_id: The entity ID to analyze.
            log: Optional EventLog instance. Uses global log if not provided.

        Returns:
            List of RootCauseSummary objects, one per unique root cause.
        """
        events = _collect_events_for_entity(entity_id, log)

        # Group by root cause
        by_root: dict[tuple[Optional[str], Optional[int]], list[Event]] = {}
        for event in events:
            key = (event.root_cause, event.root_cause_entity)
            by_root.setdefault(key, []).append(event)

        summaries = []
        for (root, root_entity), events_list in by_root.items():
            if root is None:
                continue

            affected: set[int] = set()
            total_changes = 0

            for e in events_list:
                for c in e.changes:
                    affected.add(c.entity)
                    total_changes += 1

            summaries.append(cls(
                root_cause=root,
                root_cause_entity=root_entity,
                total_events=len(events_list),
                total_changes=total_changes,
                affected_entities=affected
            ))

        return summaries


def register_inspector_views() -> None:
    """
    Register history, causality, and provenance views with the global inspector.

    Call this function during application initialization to make
    HistoryView, CausalityView, and ProvenanceView available in the inspector.

    Example:
        from foundation.inspector_views import register_inspector_views
        register_inspector_views()

        # Now you can use:
        panel = inspector.inspect(entity)
        panel.set_view("History")
        panel.set_view("Causality")
        panel.set_view("Provenance")
    """
    from foundation.inspector import inspector
    inspector.register_view(HistoryView())
    inspector.register_view(CausalityView())
    inspector.register_view(ProvenanceView())


__all__ = [
    "HistoryView",
    "CausalityView",
    "ProvenanceView",
    "RootCauseSummary",
    "register_inspector_views",
]
