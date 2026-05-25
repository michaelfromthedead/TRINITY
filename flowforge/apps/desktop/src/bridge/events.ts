/**
 * Event Subscription Bridge
 *
 * Replaces ComfyUI's WebSocket event system with Tauri events.
 */

import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import type { ExecutionEvent } from '@flowforge/core';

/**
 * Event subscription handle.
 */
export interface EventSubscription {
  unsubscribe: () => void;
}

/**
 * Subscribe to execution events.
 */
export async function subscribeToEvents(
  eventName: string,
  callback: (event: ExecutionEvent) => void
): Promise<EventSubscription> {
  const unlisten = await listen<ExecutionEvent>(eventName, (event) => {
    callback(event.payload);
  });

  return {
    unsubscribe: unlisten,
  };
}

/**
 * Subscribe to all execution-related events.
 */
export async function subscribeToExecution(
  callback: (event: ExecutionEvent) => void
): Promise<EventSubscription> {
  const unlisteners: UnlistenFn[] = [];

  // Subscribe to all execution event types
  const eventTypes = [
    'execution:start',
    'execution:progress',
    'execution:complete',
    'execution:error',
    'execution:cancelled',
    'node:start',
    'node:progress',
    'node:complete',
    'node:error',
    'node:skipped',
  ];

  for (const eventType of eventTypes) {
    const unlisten = await listen<ExecutionEvent>(eventType, (event) => {
      callback(event.payload);
    });
    unlisteners.push(unlisten);
  }

  return {
    unsubscribe: () => {
      for (const unlisten of unlisteners) {
        unlisten();
      }
    },
  };
}

/**
 * One-time event listener.
 */
export async function once<T>(
  eventName: string,
  timeout?: number
): Promise<T> {
  return new Promise((resolve, reject) => {
    let unlistenFn: UnlistenFn | undefined;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;

    if (timeout !== undefined) {
      timeoutId = setTimeout(() => {
        if (unlistenFn !== undefined) {
          unlistenFn();
        }
        reject(new Error(`Event ${eventName} timed out after ${timeout}ms`));
      }, timeout);
    }

    listen<T>(eventName, (event) => {
      if (timeoutId !== undefined) {
        clearTimeout(timeoutId);
      }
      if (unlistenFn !== undefined) {
        unlistenFn();
      }
      resolve(event.payload);
    }).then((fn) => {
      unlistenFn = fn;
    });
  });
}
