/**
 * Playback Controller
 * 
 * Controller for managing animation playback.
 * Acts as interface between PlaybackModel and Views.
 */

import { PlaybackModel, type PlaybackState, type PlaybackSpeed, PLAYBACK_SPEEDS } from '@/models';

/**
 * PlaybackController class for handling playback actions
 */
export class PlaybackController {
  private model: PlaybackModel;
  private onPlaybackChange?: (state: PlaybackState) => void;

  constructor(onChange?: (state: PlaybackState) => void) {
    this.model = new PlaybackModel();
    this.onPlaybackChange = onChange;
  }

  /**
   * Get current playback state
   */
  getState(): PlaybackState {
    return this.model.getState();
  }

  /**
   * Handle play button click
   */
  handlePlay(): PlaybackState {
    const state = this.model.play();
    this.notifyChange(state);
    return state;
  }

  /**
   * Handle pause button click
   */
  handlePause(): PlaybackState {
    const state = this.model.pause();
    this.notifyChange(state);
    return state;
  }

  /**
   * Handle play/pause toggle
   */
  handlePlayPause(): PlaybackState {
    const state = this.model.togglePlayPause();
    this.notifyChange(state);
    return state;
  }

  /**
   * Handle restart button click
   */
  handleRestart(): PlaybackState {
    const state = this.model.restart();
    this.notifyChange(state);
    return state;
  }

  /**
   * Handle speed change button click
   */
  handleSpeedChange(): PlaybackState {
    const state = this.model.cycleSpeed();
    this.notifyChange(state);
    return state;
  }

  /**
   * Set specific speed
   */
  setSpeed(speed: PlaybackSpeed): PlaybackState {
    const state = this.model.setSpeed(speed);
    this.notifyChange(state);
    return state;
  }

  /**
   * Update playback state
   */
  updateState(updates: Partial<PlaybackState>): PlaybackState {
    const state = this.model.update(updates);
    this.notifyChange(state);
    return state;
  }

  /**
   * Reset playback to initial state
   */
  reset(): PlaybackState {
    const state = this.model.reset();
    this.notifyChange(state);
    return state;
  }

  /**
   * Get available speeds
   */
  getAvailableSpeeds(): PlaybackSpeed[] {
    return [...PLAYBACK_SPEEDS];
  }

  /**
   * Set the change callback
   */
  setOnChange(callback: (state: PlaybackState) => void): void {
    this.onPlaybackChange = callback;
  }

  /**
   * Notify listeners of changes
   */
  private notifyChange(state: PlaybackState): void {
    if (this.onPlaybackChange) {
      this.onPlaybackChange(state);
    }
  }
}

/**
 * Create a new PlaybackController instance
 */
export function createPlaybackController(
  onChange?: (state: PlaybackState) => void
): PlaybackController {
  return new PlaybackController(onChange);
}
