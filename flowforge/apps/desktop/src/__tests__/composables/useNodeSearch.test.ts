/**
 * useNodeSearch Composable Tests
 *
 * Tests for the node search functionality in the graph canvas.
 * Covers search filtering, highlighting, keyboard navigation, and state management.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { shallowRef, nextTick } from 'vue';
import { useNodeSearch, type SearchResult, type NodeTypeFilter } from '@/composables/useNodeSearch';
import type { LGraph, LGraphCanvas, LGraphNode, NodeId } from '@/litegraph';
import { UI_CONFIG } from '@/config/flowforge.config';

// =============================================================================
// MOCK FACTORIES
// =============================================================================

/**
 * Create a mock LGraphNode for testing.
 */
function createMockNode(overrides: Partial<LGraphNode> = {}): LGraphNode {
  const defaultNode = {
    id: `node_${Math.random().toString(36).substring(7)}` as NodeId,
    type: 'custom/component',
    title: 'TestNode',
    pos: [100, 100] as [number, number],
    size: [200, 100] as [number, number],
    properties: {},
    color: undefined,
    boxcolor: undefined,
  };

  return {
    ...defaultNode,
    ...overrides,
  } as unknown as LGraphNode;
}

/**
 * Create a mock LGraph instance.
 */
function createMockGraph(nodes: LGraphNode[] = []): LGraph {
  return {
    nodes,
    getNodeById: vi.fn((id: NodeId) => nodes.find((n) => n.id === id)),
  } as unknown as LGraph;
}

/**
 * Create a mock LGraphCanvas instance.
 */
function createMockCanvas(): LGraphCanvas {
  return {
    canvas: { width: 1920, height: 1080 },
    ds: { scale: 1, offset: [0, 0] },
    selectNode: vi.fn(),
    setDirty: vi.fn(),
  } as unknown as LGraphCanvas;
}

// =============================================================================
// TEST SUITE
// =============================================================================

describe('useNodeSearch', () => {
  let mockGraph: ReturnType<typeof shallowRef<LGraph | null>>;
  let mockCanvas: ReturnType<typeof shallowRef<LGraphCanvas | null>>;
  let testNodes: LGraphNode[];

  beforeEach(() => {
    // Create test nodes with different types and names
    testNodes = [
      createMockNode({
        id: 'node_1' as NodeId,
        type: 'custom/component',
        title: 'PlayerComponent',
        pos: [100, 100],
        size: [200, 100],
      }),
      createMockNode({
        id: 'node_2' as NodeId,
        type: 'custom/system',
        title: 'MovementSystem',
        pos: [300, 100],
        size: [200, 100],
      }),
      createMockNode({
        id: 'node_3' as NodeId,
        type: 'custom/resource',
        title: 'GameResource',
        pos: [500, 100],
        size: [200, 100],
      }),
      createMockNode({
        id: 'node_4' as NodeId,
        type: 'custom/event',
        title: 'CollisionEvent',
        pos: [700, 100],
        size: [200, 100],
      }),
      createMockNode({
        id: 'node_5' as NodeId,
        type: 'custom/component',
        title: 'EnemyComponent',
        pos: [100, 300],
        size: [200, 100],
      }),
    ];

    mockGraph = shallowRef(createMockGraph(testNodes));
    mockCanvas = shallowRef(createMockCanvas());
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Initial State', () => {
    it('should initialize with empty search state', () => {
      const { query, typeFilter, results, isActive, selectedIndex } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      expect(query.value).toBe('');
      expect(typeFilter.value).toBe('all');
      expect(results.value).toEqual([]);
      expect(isActive.value).toBe(false);
      expect(selectedIndex.value).toBe(0);
    });

    it('should initialize with empty highlighted node IDs', () => {
      const { highlightedNodeIds } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      expect(highlightedNodeIds.value.size).toBe(0);
    });
  });

  describe('Search by Name', () => {
    it('should filter nodes by name (case-insensitive)', () => {
      const { search, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('player');

      expect(results.value).toHaveLength(1);
      expect(results.value[0]?.title).toBe('PlayerComponent');
    });

    it('should find partial matches', () => {
      const { search, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('component');

      expect(results.value).toHaveLength(2);
      expect(results.value.map((r) => r.title)).toContain('PlayerComponent');
      expect(results.value.map((r) => r.title)).toContain('EnemyComponent');
    });

    it('should return empty results for no matches', () => {
      const { search, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('nonexistent');

      expect(results.value).toHaveLength(0);
    });

    it('should search in node type as well as title', () => {
      const { search, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('system');

      expect(results.value).toHaveLength(1);
      expect(results.value[0]?.title).toBe('MovementSystem');
    });

    it('should sort results alphabetically by title', () => {
      const { search, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('component');

      expect(results.value[0]?.title).toBe('EnemyComponent');
      expect(results.value[1]?.title).toBe('PlayerComponent');
    });
  });

  describe('Filter by Type', () => {
    it('should filter nodes by component type', () => {
      const { search, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('', 'component');

      expect(results.value).toHaveLength(2);
      results.value.forEach((r) => {
        expect(r.trinityType).toBe('component');
      });
    });

    it('should filter nodes by system type', () => {
      const { search, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('', 'system');

      expect(results.value).toHaveLength(1);
      expect(results.value[0]?.title).toBe('MovementSystem');
    });

    it('should filter nodes by resource type', () => {
      const { search, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('', 'resource');

      expect(results.value).toHaveLength(1);
      expect(results.value[0]?.title).toBe('GameResource');
    });

    it('should filter nodes by event type', () => {
      const { search, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('', 'event');

      expect(results.value).toHaveLength(1);
      expect(results.value[0]?.title).toBe('CollisionEvent');
    });

    it('should combine text search with type filter', () => {
      const { search, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('player', 'component');

      expect(results.value).toHaveLength(1);
      expect(results.value[0]?.title).toBe('PlayerComponent');
    });

    it('should return no results when type filter excludes all matches', () => {
      const { search, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('player', 'system');

      expect(results.value).toHaveLength(0);
    });
  });

  describe('Highlight Matching Nodes', () => {
    it('should highlight all matching nodes', () => {
      const { search, highlightedNodeIds } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('component');

      expect(highlightedNodeIds.value.size).toBe(2);
      expect(highlightedNodeIds.value.has('node_1' as NodeId)).toBe(true);
      expect(highlightedNodeIds.value.has('node_5' as NodeId)).toBe(true);
    });

    it('should apply highlight color to matching nodes', () => {
      const { search } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('player');

      expect(testNodes[0]!.boxcolor).toBe(UI_CONFIG.search.highlightColor);
    });

    it('should trigger canvas redraw after highlighting', () => {
      const canvas = mockCanvas.value!;
      const { search } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('player');

      expect(canvas.setDirty).toHaveBeenCalledWith(true, true);
    });

    it('should clear previous highlights before applying new ones', () => {
      const { search, highlightedNodeIds } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('player');
      expect(highlightedNodeIds.value.has('node_1' as NodeId)).toBe(true);

      search('enemy');
      expect(highlightedNodeIds.value.has('node_1' as NodeId)).toBe(false);
      expect(highlightedNodeIds.value.has('node_5' as NodeId)).toBe(true);
    });
  });

  describe('Clear Search', () => {
    it('should reset all search state', () => {
      const { search, clearSearch, query, typeFilter, results, isActive, selectedIndex } =
        useNodeSearch({
          graph: mockGraph,
          canvas: mockCanvas,
        });

      search('player', 'component');
      clearSearch();

      expect(query.value).toBe('');
      expect(typeFilter.value).toBe('all');
      expect(results.value).toEqual([]);
      expect(isActive.value).toBe(false);
      expect(selectedIndex.value).toBe(0);
    });

    it('should clear highlights and restore original colors', () => {
      // Set an explicit original color
      const originalBoxcolor = '#original';
      testNodes[0]!.boxcolor = originalBoxcolor;

      const { search, clearSearch, highlightedNodeIds } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('player');
      expect(highlightedNodeIds.value.size).toBe(1);
      expect(testNodes[0]!.boxcolor).toBe('#fbbf24'); // Highlight color

      clearSearch();

      expect(highlightedNodeIds.value.size).toBe(0);
      expect(testNodes[0]!.boxcolor).toBe(originalBoxcolor);
    });
  });

  describe('Keyboard Navigation', () => {
    it('should select next result', () => {
      const { search, selectNextResult, selectedIndex, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('component');
      expect(selectedIndex.value).toBe(0);

      selectNextResult();
      expect(selectedIndex.value).toBe(1);
    });

    it('should wrap around to first result after last', () => {
      const { search, selectNextResult, selectedIndex, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('component');
      expect(results.value).toHaveLength(2);

      selectNextResult(); // index 1
      selectNextResult(); // wrap to index 0

      expect(selectedIndex.value).toBe(0);
    });

    it('should select previous result', () => {
      const { search, selectNextResult, selectPreviousResult, selectedIndex } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('component');
      selectNextResult(); // index 1

      selectPreviousResult();
      expect(selectedIndex.value).toBe(0);
    });

    it('should wrap to last result when going previous from first', () => {
      const { search, selectPreviousResult, selectedIndex, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('component');
      expect(results.value).toHaveLength(2);

      selectPreviousResult();

      expect(selectedIndex.value).toBe(1);
    });

    it('should do nothing when no results', () => {
      const { search, selectNextResult, selectPreviousResult, selectedIndex } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('nonexistent');

      selectNextResult();
      expect(selectedIndex.value).toBe(0);

      selectPreviousResult();
      expect(selectedIndex.value).toBe(0);
    });
  });

  describe('Confirm Selection', () => {
    it('should select and center on the currently selected result', () => {
      const canvas = mockCanvas.value!;
      const { search, confirmSelection } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('player');
      confirmSelection();

      expect(canvas.selectNode).toHaveBeenCalledWith(
        expect.objectContaining({ id: 'node_1' }),
        false
      );
    });

    it('should set isActive to false after confirmation', () => {
      const { search, confirmSelection, isActive } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('player');
      confirmSelection();

      expect(isActive.value).toBe(false);
    });

    it('should do nothing when no results', () => {
      const canvas = mockCanvas.value!;
      const { search, confirmSelection } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('nonexistent');
      confirmSelection();

      expect(canvas.selectNode).not.toHaveBeenCalled();
    });
  });

  describe('Select Node by ID', () => {
    it('should select a specific node by ID', () => {
      const canvas = mockCanvas.value!;
      const { selectNode } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      selectNode('node_2' as NodeId);

      expect(canvas.selectNode).toHaveBeenCalledWith(
        expect.objectContaining({ id: 'node_2' }),
        false
      );
    });

    it('should center view on the selected node', () => {
      const canvas = mockCanvas.value!;
      const { selectNode } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      selectNode('node_1' as NodeId);

      // Canvas offset should be updated to center the node
      expect(canvas.ds.offset).toBeDefined();
      expect(canvas.setDirty).toHaveBeenCalledWith(true, true);
    });

    it('should do nothing for non-existent node', () => {
      const canvas = mockCanvas.value!;
      const { selectNode } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      selectNode('nonexistent' as NodeId);

      expect(canvas.selectNode).not.toHaveBeenCalled();
    });
  });

  describe('Active State', () => {
    it('should set active state', () => {
      const { setActive, isActive } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      setActive(true);
      expect(isActive.value).toBe(true);

      setActive(false);
      expect(isActive.value).toBe(false);
    });

    it('should clear highlights when deactivating', () => {
      const { search, setActive, highlightedNodeIds } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('component');
      expect(highlightedNodeIds.value.size).toBe(2);

      setActive(false);
      expect(highlightedNodeIds.value.size).toBe(0);
    });
  });

  describe('Edge Cases', () => {
    it('should handle null graph gracefully', () => {
      const nullGraph = shallowRef<LGraph | null>(null);
      const { search, results } = useNodeSearch({
        graph: nullGraph,
        canvas: mockCanvas,
      });

      search('player');

      expect(results.value).toEqual([]);
    });

    it('should handle null canvas gracefully', () => {
      const nullCanvas = shallowRef<LGraphCanvas | null>(null);
      const { search, selectNode } = useNodeSearch({
        graph: mockGraph,
        canvas: nullCanvas,
      });

      // Should not throw
      search('player');
      selectNode('node_1' as NodeId);
    });

    it('should handle empty graph', () => {
      const emptyGraph = shallowRef(createMockGraph([]));
      const { search, results } = useNodeSearch({
        graph: emptyGraph,
        canvas: mockCanvas,
      });

      search('anything');

      expect(results.value).toEqual([]);
    });

    it('should handle nodes with missing titles', () => {
      const nodeWithoutTitle = createMockNode({
        id: 'node_no_title' as NodeId,
        type: 'custom/component',
        title: '',
        pos: [100, 100],
      });

      const graphWithTitlelessNode = shallowRef(createMockGraph([nodeWithoutTitle]));
      const { search, results } = useNodeSearch({
        graph: graphWithTitlelessNode,
        canvas: mockCanvas,
      });

      search('', 'component');

      expect(results.value).toHaveLength(1);
      expect(results.value[0]?.title).toBe('Untitled');
    });

    it('should handle whitespace-only query', () => {
      const { search, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('   ');

      expect(results.value).toEqual([]);
    });

    it('should reset selected index when results change', () => {
      const { search, selectNextResult, selectedIndex, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('component');
      selectNextResult(); // index 1
      expect(selectedIndex.value).toBe(1);

      search('player');
      expect(results.value).toHaveLength(1);
      expect(selectedIndex.value).toBe(0);
    });
  });

  describe('Type Filter Reactivity', () => {
    it('should re-search when type filter changes', async () => {
      const { search, typeFilter, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('');
      typeFilter.value = 'component';
      await nextTick();

      expect(results.value).toHaveLength(2);
      results.value.forEach((r) => {
        expect(r.trinityType).toBe('component');
      });
    });
  });

  describe('Search Result Properties', () => {
    it('should include all required properties in results', () => {
      const { search, results } = useNodeSearch({
        graph: mockGraph,
        canvas: mockCanvas,
      });

      search('player');

      expect(results.value).toHaveLength(1);
      const result = results.value[0]!;

      expect(result.id).toBe('node_1');
      expect(result.title).toBe('PlayerComponent');
      expect(result.type).toBe('custom/component');
      expect(result.trinityType).toBe('component');
      expect(result.pos).toEqual([100, 100]);
      expect(result.node).toBeDefined();
    });
  });
});
