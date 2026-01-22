export type AppState = "HERO" | "PROCESSING" | "READY" | "ERROR" | "OFFLINE";

export interface HistoryItem {
  id: string;
  text: string;
  timestamp: Date;
  outputId?: string;
}

export interface PlaybackState {
  isPlaying: boolean;
  speed: 0.5 | 0.75 | 1;
  currentTime: number;
}

export interface TranslationRequest {
  text: string;
  timestamp: Date;
}

export interface TranslationResult {
  success: boolean;
  outputId?: string;
  error?: string;
}
