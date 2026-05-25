# Structure Graph

An inspector view for visualizing decorator composition, descriptor chains, and system dependencies.

---

## Purpose

Make invisible structure visible. Code is a graph. Show it.

```
CODE                              GRAPH
────                              ─────

@component                        ┌────────┐
@networked                        │ Player │
class Player:                     └───┬────┘
    health: float                     │
                                 ┌────┴────┐
                         decorators    fields
                              │           │
                        ┌─────┴─────┐     │
                     component  networked health
                         │          │      │
                     ┌───┴───┐   ┌──┴──┐   │
                   DESC  REG  TAG HOOK  descriptors
```

---

## Graph Types

### 1. Class Structure Graph

What: Single decorated class
Shows: Decorators, primitives, fields, descriptor chains
Use: "What does this class do?"

### 2. Primitive Decomposition Graph

What: Single decorator
Shows: Primitives it composes, requirements, conflicts
Use: "What does @networked actually do?"

### 3. Descriptor Chain Graph

What: Single field
Shows: Descriptor wrapping order, intercept points
Use: "What happens when I set player.health?"

### 4. System Dependency Graph

What: ECS systems
Shows: Execution order, data dependencies, phases
Use: "Why does this system run before that one?"

### 5. World Entity Graph

What: Live ECS world
Shows: Entities, components, relationships
Use: "What exists right now?"

---

## Data Model

```python
@dataclass
class Node:
    id: str
    kind: str      # class, decorator, primitive, field, descriptor, system, entity
    label: str     # Display name
    data: dict     # Kind-specific metadata

@dataclass
class Edge:
    source: str
    target: str
    relation: str  # decorated_by, composed_of, has_field, wrapped_by,
                   # depends_on, requires, conflicts, references

@dataclass
class Graph:
    nodes: list[Node]
    edges: list[Edge]
    root: str | None  # Entry point for navigation
```

---

## Extractors

### Class Extractor

```
INPUT:  type (decorated class)
OUTPUT: Graph

PROCESS:
1. Create class node
2. For each decorator on class:
   a. Create decorator node
   b. Edge: class --decorated_by--> decorator
   c. For each primitive in decompose(decorator):
      - Create primitive node
      - Edge: decorator --composed_of--> primitive
   d. For each requirement:
      - Edge: decorator --requires--> other_decorator
3. For each field in mirror(class).fields:
   a. Create field node
   b. Edge: class --has_field--> field
   c. For each descriptor in chain:
      - Create descriptor node
      - Edge: prev --wrapped_by--> descriptor
   d. For each metadata tag:
      - Create tag node
      - Edge: field --tagged--> tag
```

### Decorator Extractor

```
INPUT:  decorator function
OUTPUT: Graph

PROCESS:
1. Create decorator node as root
2. For each primitive in decompose(decorator):
   a. Create primitive node
   b. Edge: decorator --composed_of--> primitive
   c. Add primitive-specific data (registry, event, category, etc.)
3. For each requirement:
   a. Create required decorator node (ghost)
   b. Edge: decorator --requires--> required
4. For each conflict:
   a. Create conflicting decorator node (ghost)
   b. Edge: decorator --conflicts--> conflict
```

### System Extractor

```
INPUT:  World or list of systems
OUTPUT: Graph

PROCESS:
1. For each system:
   a. Create system node
   b. Add phase, reads, writes to data
2. For each system pair (A, B):
   a. If A.writes intersects B.reads:
      - Edge: A --data_to--> B
   b. If @after/@before declares order:
      - Edge: A --runs_before--> B
3. Group by phase
```

### World Extractor

```
INPUT:  Live World instance
OUTPUT: Graph

PROCESS:
1. For each entity:
   a. Create entity node
   b. For each component on entity:
      - Create component node
      - Edge: entity --has--> component
2. For each relation:
   a. Edge: entity_a --relation_type--> entity_b
```

---

## View Integration

```python
class GraphView(View):
    name = "Structure"
    priority = 40

    def can_show(self, obj) -> bool:
        # Classes, decorators, systems, worlds
        return (
            isinstance(obj, type) or
            callable(obj) and hasattr(obj, '_primitives') or
            hasattr(obj, '__systems__') or
            hasattr(obj, 'entities')
        )

    def render(self, obj, ui, inspector):
        graph = extract_graph(obj)

        # Layout
        positions = self.layout(graph)  # Force-directed or hierarchical

        # Render nodes
        for node in graph.nodes:
            pos = positions[node.id]
            style = NODE_STYLES[node.kind]
            clicked = ui.node(pos, node.label, style)
            if clicked:
                inspector.navigate_to(self.resolve(node))

        # Render edges
        for edge in graph.edges:
            style = EDGE_STYLES[edge.relation]
            ui.edge(positions[edge.source], positions[edge.target], style)

        # Legend
        ui.legend(NODE_STYLES, EDGE_STYLES)
```

---

## Shell Integration

```python
# Add to shell namespace
namespace['graph'] = extract_graph

# Usage
>>> graph(Player)
Graph(nodes=12, edges=15, root="Player")

>>> graph(Player).to_mermaid()
# Mermaid diagram string

>>> graph(Player).to_dot()
# Graphviz DOT string

>>> graph(Player).to_json()
# JSON for external tools

>>> inspect(graph(Player))
# Opens graph in visual inspector
```

---

## Export Formats

### Mermaid

```
graph TD
    Player[Player]
    Player -->|decorated_by| component
    Player -->|decorated_by| networked
    Player -->|has_field| health
    health -->|wrapped_by| TrackedDescriptor
```

### Graphviz DOT

```
digraph G {
    Player [shape=box]
    component [shape=ellipse]
    Player -> component [label="decorated_by"]
}
```

### JSON

```json
{
  "nodes": [
    {"id": "Player", "kind": "class", "label": "Player"}
  ],
  "edges": [
    {"source": "Player", "target": "component", "relation": "decorated_by"}
  ]
}
```

### Cytoscape

```json
{
  "elements": {
    "nodes": [{"data": {"id": "Player"}}],
    "edges": [{"data": {"source": "Player", "target": "component"}}]
  }
}
```

---

## Node Styles

```
KIND        SHAPE       COLOR       ICON
──────────────────────────────────────────
class       rectangle   blue        □
decorator   rounded     green       ◎
primitive   diamond     yellow      ◇
field       ellipse     gray        ○
descriptor  hexagon     orange      ⬡
system      rectangle   purple      ▣
entity      circle      cyan        ●
component   rounded     teal        ◉
tag         small       light gray  •
```

---

## Edge Styles

```
RELATION      LINE        ARROW     COLOR
────────────────────────────────────────────
decorated_by  solid       normal    gray
composed_of   dashed      normal    green
has_field     solid       normal    blue
wrapped_by    solid       normal    orange
requires      dotted      open      red
conflicts     dotted      x         red
depends_on    solid       normal    purple
references    dashed      normal    cyan
```

---

## Interaction

```
ACTION              RESULT
─────────────────────────────────────────────────
Click node          Navigate to that object in inspector
Hover node          Show tooltip with details
Right-click node    Context menu (inspect, copy, filter)
Scroll              Zoom in/out
Drag background     Pan
Drag node           Reposition (if layout unlocked)
Double-click        Expand/collapse subtree
Search box          Highlight matching nodes
Filter dropdown     Show/hide node kinds
```

---

## Queries

```python
# Find all nodes of a kind
graph.nodes_of_kind("decorator")

# Find edges by relation
graph.edges_of_relation("requires")

# Subgraph from node
graph.subgraph_from("Player", depth=2)

# Path between nodes
graph.path("Player", "NetworkedDescriptor")

# All ancestors/descendants
graph.ancestors("health")
graph.descendants("networked")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  SHELL                                                                      │
│    │                                                                        │
│    │  graph(obj)                                                           │
│    ▼                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         EXTRACTORS                                    │  │
│  │                                                                       │  │
│  │   ClassExtractor   DecoratorExtractor   SystemExtractor   WorldExt   │  │
│  │         │                  │                   │              │       │  │
│  │         └──────────────────┴───────────────────┴──────────────┘       │  │
│  │                                   │                                   │  │
│  └───────────────────────────────────┼───────────────────────────────────┘  │
│                                      ▼                                      │
│                               ┌────────────┐                                │
│                               │   Graph    │                                │
│                               │            │                                │
│                               │  nodes     │                                │
│                               │  edges     │                                │
│                               └─────┬──────┘                                │
│                                     │                                       │
│              ┌──────────────────────┼──────────────────────┐                │
│              ▼                      ▼                      ▼                │
│        ┌──────────┐          ┌──────────┐          ┌──────────┐            │
│        │ GraphView│          │ Exporters│          │ Queries  │            │
│        │(Inspector)│         │          │          │          │            │
│        │          │          │ mermaid  │          │ path()   │            │
│        │ render() │          │ dot      │          │ filter() │            │
│        │ interact │          │ json     │          │ subgraph │            │
│        └──────────┘          └──────────┘          └──────────┘            │
│                                                                             │
│  DEPENDENCIES:                                                              │
│  • Mirror (field enumeration)                                              │
│  • Registry (type lookup)                                                  │
│  • decompose() (primitive extraction)                                      │
│  • get_descriptor_chain() (descriptor introspection)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Size Estimate

```
Component               Lines
───────────────────────────────
Data model              50
ClassExtractor          80
DecoratorExtractor      50
SystemExtractor         60
WorldExtractor          60
GraphView               120
Exporters               80
Queries                 60
Shell integration       20
───────────────────────────────
TOTAL                   ~580
```

---

## Future Extensions

```
• Live updates (graph reacts to runtime changes)
• Diff view (compare two graphs)
• Time-travel (graph at different snapshots)
• Search across all graphs
• Bookmarks for interesting subgraphs
• Export to external tools (yEd, Gephi)
• VR/AR visualization for large graphs
• AI queries ("show me everything related to networking")
```

---

## Summary

```
WHAT:     Graph view for Viper's decorator/descriptor/ECS architecture
WHY:      Make invisible structure visible
HOW:      Extract graph from Mirror/Registry/decompose, render in Inspector
COST:     ~580 lines
PROVIDES: Visual understanding of code structure
ENABLES:  Navigate by clicking, export for docs, AI can query structure
```

---

*Another Pharo-style view. Code is data. Structure is visible. Click to explore.*
