/**
 * DuoSign Type Definitions
 * 
 * Core type definitions for the application data structures.
 * These types are used across models, controllers, and views.
 */

// Application states representing the UI flow
export type AppState = "HERO" | "PROCESSING" | "READY" | "ERROR" | "OFFLINE";

// Playback speed options (0.5x, 0.75x, 1x)
export type PlaybackSpeed = 0.5 | 0.75 | 1;

/**
 * Represents a single translation history entry
 */
export interface HistoryItem {
  id: string;
  text: string;
  timestamp: Date;
  outputId?: string;
}

/**
 * Represents the current playback state of the sign animation
 */
export interface PlaybackState {
  isPlaying: boolean;
  speed: PlaybackSpeed;
  currentTime: number;
}

/**
 * Request payload for translation
 */
export interface TranslationRequest {
  text: string;
  timestamp: Date;
}

/**
 * Response from translation service
 */
export interface TranslationResult {
  success: boolean;
  outputId?: string;
  error?: string;
}

/**
 * Device identification for persistence
 */
export interface DeviceInfo {
  deviceId: string;
  lastActive: Date;
}
