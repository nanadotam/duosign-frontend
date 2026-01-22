/**
 * History Controller
 * 
 * Controller for managing translation history interactions.
 * Acts as interface between HistoryModel and Views.
 */

import { HistoryModel, type HistoryItem } from '@/models';

/**
 * HistoryController class for handling history-related actions
 */
export class HistoryController {
  private model: HistoryModel;
  private onHistoryChange?: (items: HistoryItem[]) => void;

  constructor(onChange?: (items: HistoryItem[]) => void) {
    this.model = HistoryModel.getInstance();
    this.onHistoryChange = onChange;
  }

  /**
   * Initialize and load history from storage
   */
  initialize(): HistoryItem[] {
    const items = this.model.initialize();
    return items;
  }

  /**
   * Get all history items
   */
  getHistory(): HistoryItem[] {
    return this.model.getAll();
  }

  /**
   * Add a new translation to history
   */
  addEntry(text: string): HistoryItem {
    const item = this.model.add(text);
    this.notifyChange();
    return item;
  }

  /**
   * Select a history item for replay
   */
  selectItem(id: string): HistoryItem | undefined {
    return this.model.getById(id);
  }

  /**
   * Delete a history item
   */
  deleteEntry(id: string): boolean {
    const result = this.model.delete(id);
    if (result) {
      this.notifyChange();
    }
    return result;
  }

  /**
   * Clear all history
   */
  clearAll(): void {
    this.model.clearAll();
    this.notifyChange();
  }

  /**
   * Set the change callback
   */
  setOnChange(callback: (items: HistoryItem[]) => void): void {
    this.onHistoryChange = callback;
  }

  /**
   * Notify listeners of changes
   */
  private notifyChange(): void {
    if (this.onHistoryChange) {
      this.onHistoryChange(this.model.getAll());
    }
  }
}

/**
 * Create a new HistoryController instance
 */
export function createHistoryController(
  onChange?: (items: HistoryItem[]) => void
): HistoryController {
  return new HistoryController(onChange);
}
