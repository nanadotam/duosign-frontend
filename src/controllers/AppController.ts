/**
 * App Controller
 * 
 * Main application controller that orchestrates all other controllers.
 * Manages the overall application state and coordinates between
 * History, Playback, and Translation controllers.
 */

import { HistoryController } from './HistoryController';
import { PlaybackController } from './PlaybackController';
import { TranslationController, type TranslationState } from './TranslationController';
import type { AppState, HistoryItem, PlaybackState } from '@/models';

export interface AppControllerState {
  appState: AppState;
  history: HistoryItem[];
  selectedHistoryItem: HistoryItem | null;
  playback: PlaybackState;
  translation: TranslationState;
}

export interface AppControllerCallbacks {
  onStateChange?: (state: AppControllerState) => void;
  onAppStateChange?: (appState: AppState) => void;
  onHistoryChange?: (history: HistoryItem[]) => void;
  onPlaybackChange?: (playback: PlaybackState) => void;
}

/**
 * AppController class - main orchestrating controller
 */
export class AppController {
  private historyController: HistoryController;
  private playbackController: PlaybackController;
  private translationController: TranslationController;
  
  private appState: AppState = 'HERO';
  private selectedHistoryItem: HistoryItem | null = null;
  private callbacks: AppControllerCallbacks = {};

  constructor(callbacks?: AppControllerCallbacks) {
    this.callbacks = callbacks || {};

    // Initialize sub-controllers with change handlers
    this.historyController = new HistoryController((items) => {
      this.callbacks.onHistoryChange?.(items);
      this.notifyStateChange();
    });

    this.playbackController = new PlaybackController((state) => {
      this.callbacks.onPlaybackChange?.(state);
      this.notifyStateChange();
    });

    this.translationController = new TranslationController((state) => {
      this.handleTranslationStateChange(state);
      this.notifyStateChange();
    });
  }

  /**
   * Initialize the application
   */
  initialize(): AppControllerState {
    const history = this.historyController.initialize();
    return this.getState();
  }

  /**
   * Get current application state
   */
  getState(): AppControllerState {
    return {
      appState: this.appState,
      history: this.historyController.getHistory(),
      selectedHistoryItem: this.selectedHistoryItem,
      playback: this.playbackController.getState(),
      translation: this.translationController.getState(),
    };
  }

  /**
   * Set application state
   */
  setAppState(state: AppState): void {
    this.appState = state;
    this.callbacks.onAppStateChange?.(state);
    this.notifyStateChange();
  }

  // ============ History Actions ============

  /**
   * Submit new translation
   */
  async submitTranslation(text: string): Promise<void> {
    if (!text.trim()) return;

    // Add to history
    const historyItem = this.historyController.addEntry(text);
    this.selectedHistoryItem = historyItem;

    // Submit translation
    const result = await this.translationController.submitTranslation(text);

    // If successful, start playback
    if (result.success) {
      this.playbackController.handlePlay();
    }
  }

  /**
   * Select a history item
   */
  selectHistoryItem(item: HistoryItem): void {
    this.selectedHistoryItem = item;
    this.setAppState('READY');
    this.notifyStateChange();
  }

  /**
   * Clear all history
   */
  clearHistory(): void {
    this.historyController.clearAll();
    this.selectedHistoryItem = null;
    this.notifyStateChange();
  }

  // ============ Playback Actions ============

  /**
   * Toggle play/pause
   */
  handlePlayPause(): void {
    this.playbackController.handlePlayPause();
  }

  /**
   * Restart playback
   */
  handleRestart(): void {
    this.playbackController.handleRestart();
  }

  /**
   * Cycle playback speed
   */
  handleSpeedChange(): void {
    this.playbackController.handleSpeedChange();
  }

  /**
   * Update playback state
   */
  updatePlayback(updates: Partial<PlaybackState>): void {
    this.playbackController.updateState(updates);
  }

  // ============ Translation Actions ============

  /**
   * Retry last translation
   */
  async retryTranslation(): Promise<void> {
    if (this.selectedHistoryItem) {
      await this.submitTranslation(this.selectedHistoryItem.text);
    }
  }

  // ============ Private Methods ============

  /**
   * Handle translation state changes
   */
  private handleTranslationStateChange(state: TranslationState): void {
    switch (state.status) {
      case 'processing':
        this.appState = 'PROCESSING';
        break;
      case 'success':
        this.appState = 'READY';
        break;
      case 'error':
        this.appState = 'ERROR';
        break;
      case 'offline':
        this.appState = 'OFFLINE';
        break;
      default:
        // Keep current state
        break;
    }
    this.callbacks.onAppStateChange?.(this.appState);
  }

  /**
   * Notify listeners of full state change
   */
  private notifyStateChange(): void {
    this.callbacks.onStateChange?.(this.getState());
  }
}

/**
 * Create a new AppController instance
 */
export function createAppController(callbacks?: AppControllerCallbacks): AppController {
  return new AppController(callbacks);
}
