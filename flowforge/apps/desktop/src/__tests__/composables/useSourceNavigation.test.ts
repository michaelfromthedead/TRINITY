/**
 * useSourceNavigation Composable Tests
 *
 * Tests for source file navigation functionality.
 * Covers navigation events, highlight management, clipboard, and external integration.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { mount } from '@vue/test-utils';
import { defineComponent, nextTick } from 'vue';
import {
  useSourceNavigation,
  dispatchNavigateToSource,
  dispatchClearHighlight,
  NAVIGATE_TO_SOURCE_EVENT,
  CLEAR_HIGHLIGHT_EVENT,
  EXTERNAL_NAVIGATION_EVENT,
  type SourceLocation,
} from '@/composables/useSourceNavigation';

// =============================================================================
// TEST COMPONENT WRAPPER
// =============================================================================

/**
 * Test component that uses the useSourceNavigation composable.
 * This is needed because the composable uses onMounted/onUnmounted hooks.
 */
function createTestComponent(options = {}) {
  return defineComponent({
    setup() {
      const navigation = useSourceNavigation(options);
      return {
        ...navigation,
      };
    },
    template: '<div></div>',
  });
}

// =============================================================================
// MOCK SETUP
// =============================================================================

// Mock clipboard API
const mockClipboard = {
  writeText: vi.fn().mockResolvedValue(undefined),
};

// Store for event listeners
const eventListeners: Map<string, EventListener[]> = new Map();

// Custom window mock for events
const originalAddEventListener = window.addEventListener;
const originalRemoveEventListener = window.removeEventListener;
const originalDispatchEvent = window.dispatchEvent;

// =============================================================================
// TEST SUITE
// =============================================================================

describe('useSourceNavigation', () => {
  beforeEach(() => {
    // Clear mocks
    vi.clearAllMocks();
    eventListeners.clear();

    // Mock clipboard
    Object.defineProperty(navigator, 'clipboard', {
      value: mockClipboard,
      writable: true,
      configurable: true,
    });
    mockClipboard.writeText.mockResolvedValue(undefined);

    // Track event listeners
    window.addEventListener = vi.fn((type: string, listener: EventListener) => {
      if (!eventListeners.has(type)) {
        eventListeners.set(type, []);
      }
      eventListeners.get(type)!.push(listener);
      originalAddEventListener.call(window, type, listener);
    });

    window.removeEventListener = vi.fn((type: string, listener: EventListener) => {
      const listeners = eventListeners.get(type);
      if (listeners) {
        const index = listeners.indexOf(listener);
        if (index > -1) {
          listeners.splice(index, 1);
        }
      }
      originalRemoveEventListener.call(window, type, listener);
    });

    // Clear singleton state by creating a component and clearing it
    const clearWrapper = mount(createTestComponent());
    (clearWrapper.vm as any).clearHighlight();
    clearWrapper.unmount();
  });

  afterEach(() => {
    // Restore original window methods
    window.addEventListener = originalAddEventListener;
    window.removeEventListener = originalRemoveEventListener;
    window.dispatchEvent = originalDispatchEvent;
  });

  describe('Initial State', () => {
    it('should initialize with no source highlight', () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      expect(vm.currentSource).toBeNull();
      expect(vm.hasHighlight).toBe(false);

      wrapper.unmount();
    });

    it('should auto-start listening for events by default', () => {
      const wrapper = mount(createTestComponent());

      expect(window.addEventListener).toHaveBeenCalledWith(
        NAVIGATE_TO_SOURCE_EVENT,
        expect.any(Function)
      );
      expect(window.addEventListener).toHaveBeenCalledWith(
        CLEAR_HIGHLIGHT_EVENT,
        expect.any(Function)
      );

      wrapper.unmount();
    });

    it('should not auto-start when autoStart is false', () => {
      const wrapper = mount(createTestComponent({ autoStart: false }));

      // Check that the event listeners were NOT added for our events
      const navigateListeners = eventListeners.get(NAVIGATE_TO_SOURCE_EVENT) || [];
      expect(navigateListeners).toHaveLength(0);

      wrapper.unmount();
    });
  });

  describe('Navigate to Source', () => {
    it('should set current source when navigating', () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      vm.navigateToSource('components.py', 42);

      expect(vm.currentSource).toEqual({ file: 'components.py', line: 42 });
      expect(vm.hasHighlight).toBe(true);

      wrapper.unmount();
    });

    it('should call onNavigate callback when provided', () => {
      const onNavigate = vi.fn();
      const wrapper = mount(createTestComponent({ onNavigate }));
      const vm = wrapper.vm as any;

      vm.navigateToSource('systems.py', 100);

      expect(onNavigate).toHaveBeenCalledWith({
        file: 'systems.py',
        line: 100,
      });

      wrapper.unmount();
    });

    it('should update source location on subsequent navigations', () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      vm.navigateToSource('first.py', 10);
      expect(vm.currentSource.file).toBe('first.py');

      vm.navigateToSource('second.py', 20);
      expect(vm.currentSource.file).toBe('second.py');
      expect(vm.currentSource.line).toBe(20);

      wrapper.unmount();
    });
  });

  describe('Clear Highlight', () => {
    it('should clear current source when clearing highlight', () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      vm.navigateToSource('test.py', 50);
      expect(vm.hasHighlight).toBe(true);

      vm.clearHighlight();

      expect(vm.currentSource).toBeNull();
      expect(vm.hasHighlight).toBe(false);

      wrapper.unmount();
    });

    it('should call onClear callback when provided', () => {
      const onClear = vi.fn();
      const wrapper = mount(createTestComponent({ onClear }));
      const vm = wrapper.vm as any;

      vm.navigateToSource('test.py', 50);
      vm.clearHighlight();

      expect(onClear).toHaveBeenCalled();

      wrapper.unmount();
    });
  });

  describe('Format Location', () => {
    it('should format location as file:line', () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      vm.navigateToSource('resources.py', 123);

      expect(vm.formatLocation()).toBe('resources.py:123');

      wrapper.unmount();
    });

    it('should return empty string when no highlight', () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      expect(vm.formatLocation()).toBe('');

      wrapper.unmount();
    });
  });

  describe('Copy to Clipboard', () => {
    it('should copy formatted location to clipboard', async () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      vm.navigateToSource('events.py', 75);
      const result = await vm.copyToClipboard();

      expect(mockClipboard.writeText).toHaveBeenCalledWith('events.py:75');
      expect(result).toBe(true);

      wrapper.unmount();
    });

    it('should return false when no highlight', async () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      const result = await vm.copyToClipboard();

      expect(result).toBe(false);
      expect(mockClipboard.writeText).not.toHaveBeenCalled();

      wrapper.unmount();
    });

    it('should return false when clipboard write fails', async () => {
      mockClipboard.writeText.mockRejectedValueOnce(new Error('Clipboard error'));

      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      vm.navigateToSource('test.py', 1);
      const result = await vm.copyToClipboard();

      expect(result).toBe(false);

      wrapper.unmount();
    });
  });

  describe('Event Handling', () => {
    it('should handle navigation events from window', async () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      // Dispatch navigation event
      dispatchNavigateToSource('handler.py', 200);
      await nextTick();

      expect(vm.currentSource).toEqual({ file: 'handler.py', line: 200 });
      expect(vm.hasHighlight).toBe(true);

      wrapper.unmount();
    });

    it('should handle clear highlight events from window', async () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      vm.navigateToSource('test.py', 1);
      expect(vm.hasHighlight).toBe(true);

      dispatchClearHighlight();
      await nextTick();

      expect(vm.hasHighlight).toBe(false);

      wrapper.unmount();
    });

    it('should ignore events with invalid data', async () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      // Dispatch event with missing data
      const invalidEvent = new CustomEvent(NAVIGATE_TO_SOURCE_EVENT, {
        detail: { file: 'test.py' }, // missing line
      });
      window.dispatchEvent(invalidEvent);
      await nextTick();

      expect(vm.currentSource).toBeNull();

      wrapper.unmount();
    });

    it('should ignore events with null detail', async () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      const nullEvent = new CustomEvent(NAVIGATE_TO_SOURCE_EVENT, {
        detail: null,
      });
      window.dispatchEvent(nullEvent);
      await nextTick();

      expect(vm.currentSource).toBeNull();

      wrapper.unmount();
    });
  });

  describe('Start/Stop Listening', () => {
    it('should start listening for events', () => {
      const wrapper = mount(createTestComponent({ autoStart: false }));
      const vm = wrapper.vm as any;

      vm.start();

      expect(window.addEventListener).toHaveBeenCalledWith(
        NAVIGATE_TO_SOURCE_EVENT,
        expect.any(Function)
      );

      wrapper.unmount();
    });

    it('should stop listening for events', () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      vm.stop();

      expect(window.removeEventListener).toHaveBeenCalledWith(
        NAVIGATE_TO_SOURCE_EVENT,
        expect.any(Function)
      );
      expect(window.removeEventListener).toHaveBeenCalledWith(
        CLEAR_HIGHLIGHT_EVENT,
        expect.any(Function)
      );

      wrapper.unmount();
    });

    it('should not add duplicate listeners when start called multiple times', () => {
      // Clear mocks to get clean count
      (window.addEventListener as any).mockClear();

      const wrapper = mount(createTestComponent({ autoStart: false }));
      const vm = wrapper.vm as any;

      vm.start();

      // Count calls after first start
      const callsAfterFirstStart = (window.addEventListener as any).mock.calls.filter(
        (call: any[]) => call[0] === NAVIGATE_TO_SOURCE_EVENT
      ).length;

      vm.start(); // Should be idempotent - should not add more listeners

      // Count calls after second start
      const callsAfterSecondStart = (window.addEventListener as any).mock.calls.filter(
        (call: any[]) => call[0] === NAVIGATE_TO_SOURCE_EVENT
      ).length;

      expect(callsAfterSecondStart).toBe(callsAfterFirstStart);

      wrapper.unmount();
    });

    it('should not remove listeners when stop called multiple times', () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      vm.stop();
      const firstCallCount = (window.removeEventListener as any).mock.calls.length;

      vm.stop(); // Should be idempotent
      const secondCallCount = (window.removeEventListener as any).mock.calls.length;

      expect(secondCallCount).toBe(firstCallCount);

      wrapper.unmount();
    });
  });

  describe('External Navigation', () => {
    it('should emit external navigation event', () => {
      const dispatchSpy = vi.spyOn(window, 'dispatchEvent');
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      const location: SourceLocation = { file: 'external.py', line: 300 };
      vm.emitExternalNavigation(location);

      expect(dispatchSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          type: EXTERNAL_NAVIGATION_EVENT,
          detail: location,
        })
      );

      wrapper.unmount();
      dispatchSpy.mockRestore();
    });
  });

  describe('Lifecycle', () => {
    it('should clean up event listeners on unmount', () => {
      const wrapper = mount(createTestComponent());

      wrapper.unmount();

      expect(window.removeEventListener).toHaveBeenCalledWith(
        NAVIGATE_TO_SOURCE_EVENT,
        expect.any(Function)
      );
      expect(window.removeEventListener).toHaveBeenCalledWith(
        CLEAR_HIGHLIGHT_EVENT,
        expect.any(Function)
      );
    });
  });

  describe('Utility Functions', () => {
    describe('dispatchNavigateToSource', () => {
      it('should dispatch a navigation event', () => {
        const dispatchSpy = vi.spyOn(window, 'dispatchEvent');

        dispatchNavigateToSource('utils.py', 50);

        expect(dispatchSpy).toHaveBeenCalledWith(
          expect.objectContaining({
            type: NAVIGATE_TO_SOURCE_EVENT,
            detail: { file: 'utils.py', line: 50 },
          })
        );

        dispatchSpy.mockRestore();
      });
    });

    describe('dispatchClearHighlight', () => {
      it('should dispatch a clear highlight event', () => {
        const dispatchSpy = vi.spyOn(window, 'dispatchEvent');

        dispatchClearHighlight();

        expect(dispatchSpy).toHaveBeenCalledWith(
          expect.objectContaining({
            type: CLEAR_HIGHLIGHT_EVENT,
          })
        );

        dispatchSpy.mockRestore();
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle very long file paths', () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      const longPath = 'a'.repeat(500) + '.py';
      vm.navigateToSource(longPath, 1);

      expect(vm.currentSource.file).toBe(longPath);
      expect(vm.formatLocation()).toBe(`${longPath}:1`);

      wrapper.unmount();
    });

    it('should handle zero line number', () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      vm.navigateToSource('zero.py', 0);

      expect(vm.currentSource.line).toBe(0);
      expect(vm.formatLocation()).toBe('zero.py:0');

      wrapper.unmount();
    });

    it('should handle large line numbers', () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      vm.navigateToSource('large.py', 999999);

      expect(vm.currentSource.line).toBe(999999);
      expect(vm.formatLocation()).toBe('large.py:999999');

      wrapper.unmount();
    });

    it('should handle special characters in file paths', () => {
      const wrapper = mount(createTestComponent());
      const vm = wrapper.vm as any;

      vm.navigateToSource('path/with spaces/file-name.py', 10);

      expect(vm.currentSource.file).toBe('path/with spaces/file-name.py');

      wrapper.unmount();
    });
  });

  describe('Singleton State', () => {
    it('should share state between multiple component instances', async () => {
      const wrapper1 = mount(createTestComponent());
      const wrapper2 = mount(createTestComponent());

      const vm1 = wrapper1.vm as any;
      const vm2 = wrapper2.vm as any;

      vm1.navigateToSource('shared.py', 100);
      await nextTick();

      // Both instances should see the same state
      expect(vm2.currentSource).toEqual({ file: 'shared.py', line: 100 });
      expect(vm2.hasHighlight).toBe(true);

      wrapper1.unmount();
      wrapper2.unmount();
    });
  });

  describe('Callback Order', () => {
    it('should update state before calling onNavigate callback', () => {
      let stateAtCallback: SourceLocation | null = null;

      const onNavigate = vi.fn((location: SourceLocation) => {
        stateAtCallback = location;
      });

      const wrapper = mount(createTestComponent({ onNavigate }));
      const vm = wrapper.vm as any;

      vm.navigateToSource('callback.py', 25);

      expect(stateAtCallback).toEqual({ file: 'callback.py', line: 25 });
      expect(vm.currentSource).toEqual(stateAtCallback);

      wrapper.unmount();
    });
  });
});
