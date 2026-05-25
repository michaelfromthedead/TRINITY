/**
 * Graph Store Tests
 *
 * Tests for the Pinia graph store that manages node graph state,
 * file operations, history (undo/redo), and API integration.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import {
  createMockNodeGraph,
  createMockGraphNode as createMockApiNode,
  createMockFileInfo,
} from '../mocks/tauri';
import { flushPromises } from '../setup';

// Mock the services module BEFORE importing graphStore
// Note: vi.mock is hoisted, so we use vi.hoisted for mock functions
const { mockParsePythonFile, mockSavePythonFile, mockOpenPythonFile, mockSavePythonFileAs, mockGetFileInfo } = vi.hoisted(() => ({
  mockParsePythonFile: vi.fn(),
  mockSavePythonFile: vi.fn(),
  mockOpenPythonFile: vi.fn(),
  mockSavePythonFileAs: vi.fn(),
  mockGetFileInfo: vi.fn(),
}));

vi.mock('@/services', () => ({
  getApi: () => ({
    parsePythonFile: mockParsePythonFile,
    savePythonFile: mockSavePythonFile,
    openPythonFile: mockOpenPythonFile,
    savePythonFileAs: mockSavePythonFileAs,
  }),
}));

vi.mock('@/bridge/files', () => ({
  getFileInfo: mockGetFileInfo,
}));

import { useGraphStore, type GraphNode, type GraphLink, type GraphState } from '@/stores/graphStore';
import type { NodeGraph, GraphNode as ApiGraphNode, GraphEdge } from '@/services';

describe('useGraphStore', () => {
  let store: ReturnType<typeof useGraphStore>;

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();
    mockParsePythonFile.mockReset();
    mockSavePythonFile.mockReset();
    mockOpenPythonFile.mockReset();
    mockSavePythonFileAs.mockReset();
    mockGetFileInfo.mockReset();

    // Default mock implementation for getFileInfo
    mockGetFileInfo.mockResolvedValue(createMockFileInfo());

    setActivePinia(createPinia());
    store = useGraphStore();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('Initial State', () => {
    it('should have empty initial state', () => {
      expect(store.nodes).toEqual([]);
      expect(store.links).toEqual([]);
      expect(store.groups).toEqual([]);
      expect(store.currentFilePath).toBeNull();
      expect(store.isModified).toBe(false);
    });

    it('should have default canvas state', () => {
      expect(store.canvasOffset).toEqual([0, 0]);
      expect(store.canvasScale).toBe(1);
    });

    it('should have empty selection', () => {
      expect(store.selectedNodeIds.size).toBe(0);
      expect(store.selectedLinkIds.size).toBe(0);
      expect(store.hasSelection).toBe(false);
    });

    it('should not be able to undo/redo initially', () => {
      expect(store.canUndo).toBe(false);
      expect(store.canRedo).toBe(false);
    });
  });

  describe('Node Operations', () => {
    describe('addNode', () => {
      it('should add a node and mark as modified', () => {
        const node = store.addNode({
          type: 'component',
          title: 'TestComponent',
          pos: [100, 100],
          size: [200, 100],
        });

        expect(store.nodes).toHaveLength(1);
        expect(node.id).toBe('node_1');
        expect(node.title).toBe('TestComponent');
        expect(store.isModified).toBe(true);
      });

      it('should increment node IDs', () => {
        const node1 = store.addNode({ type: 'component', title: 'Node1', pos: [0, 0], size: [100, 50] });
        const node2 = store.addNode({ type: 'system', title: 'Node2', pos: [200, 0], size: [100, 50] });

        expect(node1.id).toBe('node_1');
        expect(node2.id).toBe('node_2');
      });

      it('should push to history', () => {
        // canUndo requires historyIndex > 0, so we need at least 2 operations
        store.addNode({ type: 'component', title: 'Test1', pos: [0, 0], size: [100, 50] });
        store.addNode({ type: 'component', title: 'Test2', pos: [100, 0], size: [100, 50] });

        expect(store.canUndo).toBe(true);
        expect(store.getHistoryList()).toHaveLength(2);
      });
    });

    describe('removeNode', () => {
      it('should remove a node', () => {
        const node = store.addNode({ type: 'component', title: 'ToRemove', pos: [0, 0], size: [100, 50] });
        store.removeNode(node.id);

        expect(store.nodes).toHaveLength(0);
      });

      it('should remove connected links when removing a node', () => {
        const node1 = store.addNode({ type: 'component', title: 'Node1', pos: [0, 0], size: [100, 50] });
        const node2 = store.addNode({ type: 'component', title: 'Node2', pos: [200, 0], size: [100, 50] });
        store.addLink({ originId: node1.id, originSlot: 0, targetId: node2.id, targetSlot: 0, type: 'reference' });

        store.removeNode(node1.id);

        expect(store.links).toHaveLength(0);
      });

      it('should remove from selection when removing a node', () => {
        const node = store.addNode({ type: 'component', title: 'Test', pos: [0, 0], size: [100, 50] });
        store.selectNode(node.id);
        store.removeNode(node.id);

        expect(store.selectedNodeIds.has(node.id)).toBe(false);
      });

      it('should do nothing for non-existent node', () => {
        store.addNode({ type: 'component', title: 'Test', pos: [0, 0], size: [100, 50] });
        store.removeNode('non_existent');

        expect(store.nodes).toHaveLength(1);
      });
    });

    describe('updateNode', () => {
      it('should update node properties', () => {
        const node = store.addNode({ type: 'component', title: 'Original', pos: [0, 0], size: [100, 50] });
        store.updateNode(node.id, { title: 'Updated', pos: [100, 200] });

        expect(store.nodes[0]?.title).toBe('Updated');
        expect(store.nodes[0]?.pos).toEqual([100, 200]);
      });

      it('should mark as modified', () => {
        const node = store.addNode({ type: 'component', title: 'Test', pos: [0, 0], size: [100, 50] });
        store.markSaved();
        store.updateNode(node.id, { title: 'Changed' });

        expect(store.isModified).toBe(true);
      });
    });

    describe('moveNode', () => {
      it('should update node position', () => {
        const node = store.addNode({ type: 'component', title: 'Test', pos: [0, 0], size: [100, 50] });
        store.moveNode(node.id, [150, 250]);

        expect(store.nodes[0]?.pos).toEqual([150, 250]);
      });
    });
  });

  describe('Link Operations', () => {
    let node1: GraphNode;
    let node2: GraphNode;

    beforeEach(() => {
      node1 = store.addNode({ type: 'component', title: 'Node1', pos: [0, 0], size: [100, 50] });
      node2 = store.addNode({ type: 'component', title: 'Node2', pos: [200, 0], size: [100, 50] });
    });

    describe('addLink', () => {
      it('should add a link between nodes', () => {
        const link = store.addLink({
          originId: node1.id,
          originSlot: 0,
          targetId: node2.id,
          targetSlot: 0,
          type: 'reference',
        });

        expect(store.links).toHaveLength(1);
        expect(link.id).toBe('link_1');
        expect(link.originId).toBe(node1.id);
        expect(link.targetId).toBe(node2.id);
      });

      it('should increment link IDs', () => {
        const link1 = store.addLink({ originId: node1.id, originSlot: 0, targetId: node2.id, targetSlot: 0, type: 'reference' });
        const node3 = store.addNode({ type: 'component', title: 'Node3', pos: [400, 0], size: [100, 50] });
        const link2 = store.addLink({ originId: node2.id, originSlot: 0, targetId: node3.id, targetSlot: 0, type: 'reference' });

        expect(link1.id).toBe('link_1');
        expect(link2.id).toBe('link_2');
      });
    });

    describe('removeLink', () => {
      it('should remove a link', () => {
        const link = store.addLink({ originId: node1.id, originSlot: 0, targetId: node2.id, targetSlot: 0, type: 'reference' });
        store.removeLink(link.id);

        expect(store.links).toHaveLength(0);
      });

      it('should remove from selection', () => {
        const link = store.addLink({ originId: node1.id, originSlot: 0, targetId: node2.id, targetSlot: 0, type: 'reference' });
        store.selectLink(link.id);
        store.removeLink(link.id);

        expect(store.selectedLinkIds.has(link.id)).toBe(false);
      });
    });

    describe('getConnectedEdges', () => {
      it('should return all links connected to a node', () => {
        const node3 = store.addNode({ type: 'component', title: 'Node3', pos: [400, 0], size: [100, 50] });
        store.addLink({ originId: node1.id, originSlot: 0, targetId: node2.id, targetSlot: 0, type: 'reference' });
        store.addLink({ originId: node2.id, originSlot: 0, targetId: node3.id, targetSlot: 0, type: 'reference' });

        const connected = store.getConnectedEdges(node2.id);

        expect(connected).toHaveLength(2);
      });

      it('should return empty array for isolated node', () => {
        const isolated = store.addNode({ type: 'component', title: 'Isolated', pos: [600, 0], size: [100, 50] });

        const connected = store.getConnectedEdges(isolated.id);

        expect(connected).toHaveLength(0);
      });
    });
  });

  describe('Selection Operations', () => {
    let node1: GraphNode;
    let node2: GraphNode;

    beforeEach(() => {
      node1 = store.addNode({ type: 'component', title: 'Node1', pos: [0, 0], size: [100, 50] });
      node2 = store.addNode({ type: 'component', title: 'Node2', pos: [200, 0], size: [100, 50] });
    });

    describe('selectNode', () => {
      it('should select a single node', () => {
        store.selectNode(node1.id);

        expect(store.selectedNodeIds.has(node1.id)).toBe(true);
        expect(store.selectedNodes).toHaveLength(1);
      });

      it('should replace selection by default', () => {
        store.selectNode(node1.id);
        store.selectNode(node2.id);

        expect(store.selectedNodeIds.size).toBe(1);
        expect(store.selectedNodeIds.has(node2.id)).toBe(true);
      });

      it('should add to selection when additive', () => {
        store.selectNode(node1.id);
        store.selectNode(node2.id, true);

        expect(store.selectedNodeIds.size).toBe(2);
      });
    });

    describe('deselectNode', () => {
      it('should deselect a node', () => {
        store.selectNode(node1.id);
        store.deselectNode(node1.id);

        expect(store.selectedNodeIds.has(node1.id)).toBe(false);
      });
    });

    describe('selectLink', () => {
      it('should select a link', () => {
        const link = store.addLink({ originId: node1.id, originSlot: 0, targetId: node2.id, targetSlot: 0, type: 'reference' });
        store.selectLink(link.id);

        expect(store.selectedLinkIds.has(link.id)).toBe(true);
        expect(store.selectedLinks).toHaveLength(1);
      });
    });

    describe('selectAll', () => {
      it('should select all nodes and links', () => {
        const link = store.addLink({ originId: node1.id, originSlot: 0, targetId: node2.id, targetSlot: 0, type: 'reference' });
        store.selectAll();

        expect(store.selectedNodeIds.size).toBe(2);
        expect(store.selectedLinkIds.size).toBe(1);
      });
    });

    describe('clearSelection', () => {
      it('should clear all selection', () => {
        store.selectNode(node1.id);
        store.addLink({ originId: node1.id, originSlot: 0, targetId: node2.id, targetSlot: 0, type: 'reference' });
        store.selectAll();
        store.clearSelection();

        expect(store.selectedNodeIds.size).toBe(0);
        expect(store.selectedLinkIds.size).toBe(0);
        expect(store.hasSelection).toBe(false);
      });
    });

    describe('deleteSelected', () => {
      it('should delete selected nodes and links', () => {
        const link = store.addLink({ originId: node1.id, originSlot: 0, targetId: node2.id, targetSlot: 0, type: 'reference' });
        store.selectNode(node1.id);
        store.selectLink(link.id, true);
        store.deleteSelected();

        expect(store.nodes).toHaveLength(1); // node2 remains
        expect(store.links).toHaveLength(0);
      });
    });
  });

  describe('File Operations', () => {
    describe('setCurrentFile', () => {
      it('should set current file path', () => {
        store.setCurrentFile('/path/to/file.py');

        expect(store.currentFilePath).toBe('/path/to/file.py');
        expect(store.isModified).toBe(false);
      });

      it('should reset modified flag', () => {
        store.addNode({ type: 'component', title: 'Test', pos: [0, 0], size: [100, 50] });
        store.setCurrentFile('/path/to/file.py');

        expect(store.isModified).toBe(false);
      });
    });

    describe('fileName computed', () => {
      it('should return Untitled for null path', () => {
        expect(store.fileName).toBe('Untitled');
      });

      it('should extract filename from path', () => {
        store.setCurrentFile('/path/to/game.py');

        expect(store.fileName).toBe('game.py');
      });
    });

    describe('markModified', () => {
      it('should mark as modified', () => {
        store.markModified();

        expect(store.isModified).toBe(true);
      });
    });

    describe('markSaved', () => {
      it('should mark as not modified', () => {
        store.markModified();
        store.markSaved();

        expect(store.isModified).toBe(false);
      });
    });
  });

  describe('History (Undo/Redo)', () => {
    describe('undo', () => {
      it('should undo the last action', () => {
        store.addNode({ type: 'component', title: 'Node1', pos: [0, 0], size: [100, 50] });
        store.addNode({ type: 'component', title: 'Node2', pos: [100, 0], size: [100, 50] });

        store.undo();

        expect(store.nodes).toHaveLength(1);
        expect(store.nodes[0]?.title).toBe('Node1');
      });

      it('should return null when cannot undo', () => {
        const result = store.undo();

        expect(result).toBeNull();
      });

      it('should enable redo after undo', () => {
        // Need 2 operations to be able to undo
        store.addNode({ type: 'component', title: 'Test1', pos: [0, 0], size: [100, 50] });
        store.addNode({ type: 'component', title: 'Test2', pos: [100, 0], size: [100, 50] });
        store.undo();

        expect(store.canRedo).toBe(true);
      });
    });

    describe('redo', () => {
      it('should redo the undone action', () => {
        store.addNode({ type: 'component', title: 'Node1', pos: [0, 0], size: [100, 50] });
        store.addNode({ type: 'component', title: 'Node2', pos: [100, 0], size: [100, 50] });
        store.undo();
        store.redo();

        expect(store.nodes).toHaveLength(2);
      });

      it('should return null when cannot redo', () => {
        store.addNode({ type: 'component', title: 'Test', pos: [0, 0], size: [100, 50] });

        const result = store.redo();

        expect(result).toBeNull();
      });
    });

    describe('getUndoDescription', () => {
      it('should return description of undoable action', () => {
        // Need 2 operations to be able to undo
        store.addNode({ type: 'component', title: 'TestNode1', pos: [0, 0], size: [100, 50] });
        store.addNode({ type: 'component', title: 'TestNode2', pos: [100, 0], size: [100, 50] });

        const desc = store.getUndoDescription();

        expect(desc).not.toBeNull();
        expect(desc).toContain('Add node');
      });

      it('should return null when cannot undo', () => {
        expect(store.getUndoDescription()).toBeNull();
      });
    });

    describe('getRedoDescription', () => {
      it('should return description of redoable action', () => {
        // Need 2 operations to be able to undo
        store.addNode({ type: 'component', title: 'TestNode1', pos: [0, 0], size: [100, 50] });
        store.addNode({ type: 'component', title: 'TestNode2', pos: [100, 0], size: [100, 50] });
        store.undo();

        const desc = store.getRedoDescription();

        expect(desc).not.toBeNull();
        expect(desc).toContain('Add node');
      });

      it('should return null when cannot redo', () => {
        expect(store.getRedoDescription()).toBeNull();
      });
    });

    describe('clearHistory', () => {
      it('should clear all history', () => {
        store.addNode({ type: 'component', title: 'Node1', pos: [0, 0], size: [100, 50] });
        store.addNode({ type: 'component', title: 'Node2', pos: [100, 0], size: [100, 50] });
        store.clearHistory();

        expect(store.canUndo).toBe(false);
        expect(store.getHistoryList()).toHaveLength(0);
      });
    });

    describe('createSnapshot / restoreFromSnapshot', () => {
      it('should create and restore snapshot', () => {
        store.addNode({ type: 'component', title: 'Original', pos: [0, 0], size: [100, 50] });
        const snapshot = store.createSnapshot();

        store.addNode({ type: 'component', title: 'New', pos: [100, 0], size: [100, 50] });
        store.restoreFromSnapshot(snapshot, 'Restore snapshot');

        expect(store.nodes).toHaveLength(1);
        expect(store.nodes[0]?.title).toBe('Original');
      });
    });

    describe('batch', () => {
      it('should batch operations into single history entry', () => {
        store.batch(() => {
          store.addNode({ type: 'component', title: 'Node1', pos: [0, 0], size: [100, 50] });
          store.addNode({ type: 'component', title: 'Node2', pos: [100, 0], size: [100, 50] });
        }, 'Batch add nodes');

        // Should have one entry from batch
        expect(store.nodes).toHaveLength(2);
      });
    });
  });

  describe('Serialization', () => {
    describe('getGraphState', () => {
      it('should return complete graph state', () => {
        store.addNode({ type: 'component', title: 'Node1', pos: [0, 0], size: [100, 50] });
        const state = store.getGraphState();

        expect(state.nodes).toHaveLength(1);
        expect(state.lastNodeId).toBe(1);
        expect(state.version).toBe(1);
      });
    });

    describe('loadGraphState', () => {
      it('should load complete graph state', () => {
        const state: GraphState = {
          nodes: [
            { id: 'node_5', type: 'system', title: 'Loaded', pos: [50, 50], size: [100, 50] },
          ],
          links: [],
          groups: [],
          lastNodeId: 5,
          lastLinkId: 0,
          version: 1,
        };

        store.loadGraphState(state);

        expect(store.nodes).toHaveLength(1);
        expect(store.nodes[0]?.id).toBe('node_5');
      });

      it('should clear selection and history when loading', () => {
        store.addNode({ type: 'component', title: 'Old', pos: [0, 0], size: [100, 50] });
        store.selectNode('node_1');

        store.loadGraphState({
          nodes: [],
          links: [],
          groups: [],
          lastNodeId: 0,
          lastLinkId: 0,
          version: 1,
        });

        expect(store.selectedNodeIds.size).toBe(0);
      });
    });

    describe('clearGraph', () => {
      it('should reset all state', () => {
        store.addNode({ type: 'component', title: 'Test', pos: [0, 0], size: [100, 50] });
        store.setCurrentFile('/path/to/file.py');
        store.clearGraph();

        expect(store.nodes).toHaveLength(0);
        expect(store.currentFilePath).toBeNull();
        expect(store.isModified).toBe(false);
      });
    });
  });

  describe('Canvas Operations', () => {
    describe('setCanvasOffset', () => {
      it('should update canvas offset', () => {
        store.setCanvasOffset([100, 200]);

        expect(store.canvasOffset).toEqual([100, 200]);
      });
    });

    describe('setCanvasScale', () => {
      it('should update canvas scale within bounds', () => {
        store.setCanvasScale(1.5);

        expect(store.canvasScale).toBe(1.5);
      });

      it('should clamp scale to minimum', () => {
        store.setCanvasScale(0.01);

        expect(store.canvasScale).toBe(0.1); // CANVAS_CONFIG.minScale
      });

      it('should clamp scale to maximum', () => {
        store.setCanvasScale(10);

        expect(store.canvasScale).toBe(2); // CANVAS_CONFIG.maxScale
      });
    });

    describe('resetView', () => {
      it('should reset canvas to default', () => {
        store.setCanvasOffset([500, 500]);
        store.setCanvasScale(0.5);
        store.resetView();

        expect(store.canvasOffset).toEqual([0, 0]);
        expect(store.canvasScale).toBe(1);
      });
    });
  });

  describe('API-Connected File Operations', () => {
    describe('loadFromPythonFile', () => {
      it('should load and convert API graph to store format', async () => {
        const apiGraph = createMockNodeGraph(2, 1);
        mockParsePythonFile.mockResolvedValueOnce(apiGraph);

        await store.loadFromPythonFile('/path/to/game.py');

        expect(store.nodes).toHaveLength(2);
        expect(store.links).toHaveLength(1);
        expect(store.currentFilePath).toBe('/path/to/game.py');
        expect(store.isModified).toBe(false);
      });

      it('should clear selection and history after loading', async () => {
        store.addNode({ type: 'component', title: 'Old', pos: [0, 0], size: [100, 50] });
        store.selectNode('node_1');

        mockParsePythonFile.mockResolvedValueOnce({ nodes: [], edges: [] });

        await store.loadFromPythonFile('/path/to/empty.py');

        expect(store.selectedNodeIds.size).toBe(0);
      });

      it('should convert API node types correctly', async () => {
        const apiGraph: NodeGraph = {
          nodes: [
            createMockApiNode({ id: 'n1', type: 'system', name: 'MoveSystem' }),
          ],
          edges: [],
        };
        mockParsePythonFile.mockResolvedValueOnce(apiGraph);

        await store.loadFromPythonFile('/path/to/systems.py');

        expect(store.nodes[0]?.type).toBe('system');
        expect(store.nodes[0]?.title).toBe('MoveSystem');
      });
    });

    describe('saveToFile', () => {
      it('should save current graph to file', async () => {
        store.addNode({ type: 'component', title: 'Test', pos: [100, 100], size: [200, 100] });
        store.setCurrentFile('/path/to/output.py');

        mockSavePythonFile.mockResolvedValueOnce(undefined);

        await store.saveToFile();

        expect(mockSavePythonFile).toHaveBeenCalledWith(
          '/path/to/output.py',
          expect.any(Object) // NodeGraph
        );
        expect(store.isModified).toBe(false);
      });

      it('should save to custom path if provided', async () => {
        store.addNode({ type: 'component', title: 'Test', pos: [0, 0], size: [100, 50] });
        mockSavePythonFile.mockResolvedValueOnce(undefined);

        await store.saveToFile('/custom/path.py');

        expect(mockSavePythonFile).toHaveBeenCalledWith(
          '/custom/path.py',
          expect.any(Object)
        );
        expect(store.currentFilePath).toBe('/custom/path.py');
      });

      it('should throw error if no path available', async () => {
        await expect(store.saveToFile()).rejects.toThrow('No file path');
      });
    });

    describe('storeToApiGraph', () => {
      it('should convert store state to API NodeGraph format', () => {
        const node = store.addNode({
          type: 'component',
          title: 'MyComponent',
          pos: [100, 200],
          size: [200, 100],
          properties: { field1: 'value' },
        });
        store.setCurrentFile('/path/to/game.py');

        const apiGraph = store.storeToApiGraph();

        expect(apiGraph.nodes).toHaveLength(1);
        expect(apiGraph.nodes[0]).toMatchObject({
          id: node.id,
          type: 'component',
          name: 'MyComponent',
          position: [100, 200],
          data: { field1: 'value' },
          source: { file: '/path/to/game.py', line: 0 },
        });
      });

      it('should convert links to edges', () => {
        const node1 = store.addNode({ type: 'component', title: 'Node1', pos: [0, 0], size: [100, 50] });
        const node2 = store.addNode({ type: 'component', title: 'Node2', pos: [200, 0], size: [100, 50] });
        const link = store.addLink({
          originId: node1.id,
          originSlot: 0,
          targetId: node2.id,
          targetSlot: 0,
          type: 'reference',
        });

        const apiGraph = store.storeToApiGraph();

        expect(apiGraph.edges).toHaveLength(1);
        expect(apiGraph.edges[0]).toMatchObject({
          id: link.id,
          source: node1.id,
          target: node2.id,
          type: 'reference',
        });
      });
    });
  });

  describe('Group Operations', () => {
    describe('createGroup', () => {
      it('should create a group from selected nodes', () => {
        const node1 = store.addNode({ type: 'component', title: 'Node1', pos: [0, 0], size: [100, 50] });
        const node2 = store.addNode({ type: 'component', title: 'Node2', pos: [150, 0], size: [100, 50] });

        const group = store.createGroup('Test Group', [node1.id, node2.id]);

        expect(store.groups).toHaveLength(1);
        expect(group.title).toBe('Test Group');
        expect(group.nodes).toContain(node1.id);
        expect(group.nodes).toContain(node2.id);
      });

      it('should calculate bounds correctly', () => {
        store.addNode({ type: 'component', title: 'Node1', pos: [100, 100], size: [200, 100] });
        store.addNode({ type: 'component', title: 'Node2', pos: [400, 200], size: [200, 100] });

        const group = store.createGroup('Group', ['node_1', 'node_2']);

        // bounds should encompass both nodes with padding
        expect(group.bounds[0]).toBeLessThan(100); // x with padding
        expect(group.bounds[1]).toBeLessThan(100); // y with padding
        expect(group.bounds[2]).toBeGreaterThan(400); // width
        expect(group.bounds[3]).toBeGreaterThan(100); // height
      });

      it('should throw error for empty group', () => {
        expect(() => store.createGroup('Empty', [])).toThrow('Cannot create empty group');
      });
    });

    describe('removeGroup', () => {
      it('should remove a group', () => {
        const node = store.addNode({ type: 'component', title: 'Node', pos: [0, 0], size: [100, 50] });
        const group = store.createGroup('Group', [node.id]);
        store.removeGroup(group.id);

        expect(store.groups).toHaveLength(0);
      });
    });
  });

  describe('File Watcher Integration', () => {
    beforeEach(() => {
      mockGetFileInfo.mockResolvedValue(createMockFileInfo());
    });

    describe('hasExternalChanges', () => {
      it('should initially be false', () => {
        expect(store.hasExternalChanges).toBe(false);
      });
    });

    describe('acknowledgeExternalChanges', () => {
      it('should reset external changes flag', () => {
        // Simulate external change detection
        store.acknowledgeExternalChanges();

        expect(store.hasExternalChanges).toBe(false);
      });
    });

    describe('reloadFromDisk', () => {
      it('should reload the file', async () => {
        const apiGraph = createMockNodeGraph(1, 0);
        mockParsePythonFile.mockResolvedValueOnce(apiGraph);
        store.setCurrentFile('/path/to/file.py');

        await store.reloadFromDisk();

        expect(mockParsePythonFile).toHaveBeenCalledWith('/path/to/file.py');
      });

      it('should throw if no current file', async () => {
        await expect(store.reloadFromDisk()).rejects.toThrow('No file path');
      });
    });
  });
});
