/**
 * History Model
 * 
 * Manages the translation history data and persistence.
 * Handles localStorage operations for history items.
 */

import type { HistoryItem } from './types';

const STORAGE_KEY = "duosign_history";
const MAX_HISTORY_ITEMS = 50;

/**
 * HistoryModel class for managing translation history
 */
export class HistoryModel {
  private static instance: HistoryModel;
  private items: HistoryItem[] = [];
  private isInitialized = false;

  private constructor() {}

  /**
   * Get singleton instance
   */
  static getInstance(): HistoryModel {
    if (!HistoryModel.instance) {
      HistoryModel.instance = new HistoryModel();
    }
    return HistoryModel.instance;
  }

  /**
   * Initialize the model by loading history from storage
   */
  initialize(): HistoryItem[] {
    if (this.isInitialized) {
      return this.items;
    }

    if (typeof window === "undefined") {
      return [];
    }

    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as HistoryItem[];
        this.items = parsed.map(item => ({
          ...item,
          timestamp: new Date(item.timestamp)
        }));
      }
    } catch {
      this.items = [];
    }

    this.isInitialized = true;
    return this.items;
  }

  /**
   * Get all history items
   */
  getAll(): HistoryItem[] {
    return [...this.items];
  }

  /**
   * Get a single item by ID
   */
  getById(id: string): HistoryItem | undefined {
    return this.items.find(item => item.id === id);
  }

  /**
   * Add a new history item
   */
  add(text: string): HistoryItem {
    const newItem: HistoryItem = {
      id: crypto.randomUUID(),
      text,
      timestamp: new Date(),
    };

    this.items = [newItem, ...this.items].slice(0, MAX_HISTORY_ITEMS);
    this.persist();
    
    return newItem;
  }

  /**
   * Update an existing item
   */
  update(id: string, updates: Partial<Omit<HistoryItem, 'id'>>): HistoryItem | null {
    const index = this.items.findIndex(item => item.id === id);
    if (index === -1) return null;

    this.items[index] = { ...this.items[index], ...updates };
    this.persist();
    
    return this.items[index];
  }

  /**
   * Delete an item by ID
   */
  delete(id: string): boolean {
    const initialLength = this.items.length;
    this.items = this.items.filter(item => item.id !== id);
    
    if (this.items.length !== initialLength) {
      this.persist();
      return true;
    }
    
    return false;
  }

  /**
   * Clear all history
   */
  clearAll(): void {
    this.items = [];
    if (typeof window !== "undefined") {
      localStorage.removeItem(STORAGE_KEY);
    }
  }

  /**
   * Persist current state to localStorage
   */
  private persist(): void {
    if (typeof window === "undefined") return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(this.items));
  }
}

// Convenience functions for functional usage
export function getHistory(): HistoryItem[] {
  return HistoryModel.getInstance().initialize();
}

export function addHistoryItem(text: string): HistoryItem {
  return HistoryModel.getInstance().add(text);
}

export function clearHistory(): void {
  HistoryModel.getInstance().clearAll();
}

export function getHistoryItem(id: string): HistoryItem | undefined {
  return HistoryModel.getInstance().getById(id);
}
