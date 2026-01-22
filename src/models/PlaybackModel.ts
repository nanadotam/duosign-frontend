/**
 * Playback Model
 * 
 * Manages the playback state for sign language animations.
 * Handles speed settings and playback position.
 */

import type { PlaybackState, PlaybackSpeed } from './types';

/**
 * Default playback state
 */
export const DEFAULT_PLAYBACK_STATE: PlaybackState = {
  isPlaying: false,
  speed: 1,
  currentTime: 0,
};

/**
 * Available playback speeds
 */
export const PLAYBACK_SPEEDS: PlaybackSpeed[] = [0.5, 0.75, 1];

/**
 * PlaybackModel class for managing animation playback state
 */
export class PlaybackModel {
  private state: PlaybackState;

  constructor(initialState?: Partial<PlaybackState>) {
    this.state = { ...DEFAULT_PLAYBACK_STATE, ...initialState };
  }

  /**
   * Get current playback state
   */
  getState(): PlaybackState {
    return { ...this.state };
  }

  /**
   * Start or resume playback
   */
  play(): PlaybackState {
    this.state.isPlaying = true;
    return this.getState();
  }

  /**
   * Pause playback
   */
  pause(): PlaybackState {
    this.state.isPlaying = false;
    return this.getState();
  }

  /**
   * Toggle play/pause
   */
  togglePlayPause(): PlaybackState {
    this.state.isPlaying = !this.state.isPlaying;
    return this.getState();
  }

  /**
   * Restart from beginning
   */
  restart(): PlaybackState {
    this.state.currentTime = 0;
    this.state.isPlaying = true;
    return this.getState();
  }

  /**
   * Set playback speed
   */
  setSpeed(speed: PlaybackSpeed): PlaybackState {
    this.state.speed = speed;
    return this.getState();
  }

  /**
   * Cycle to next speed
   */
  cycleSpeed(): PlaybackState {
    const currentIndex = PLAYBACK_SPEEDS.indexOf(this.state.speed);
    const nextIndex = (currentIndex + 1) % PLAYBACK_SPEEDS.length;
    this.state.speed = PLAYBACK_SPEEDS[nextIndex];
    return this.getState();
  }

  /**
   * Set current time position
   */
  setCurrentTime(time: number): PlaybackState {
    this.state.currentTime = Math.max(0, time);
    return this.getState();
  }

  /**
   * Update multiple state properties
   */
  update(updates: Partial<PlaybackState>): PlaybackState {
    this.state = { ...this.state, ...updates };
    return this.getState();
  }

  /**
   * Reset to default state
   */
  reset(): PlaybackState {
    this.state = { ...DEFAULT_PLAYBACK_STATE };
    return this.getState();
  }
}

/**
 * Create a new PlaybackModel instance
 */
export function createPlaybackModel(initialState?: Partial<PlaybackState>): PlaybackModel {
  return new PlaybackModel(initialState);
}
