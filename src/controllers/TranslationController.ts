/**
 * Translation Controller
 * 
 * Controller for managing translation requests and responses.
 * Acts as interface between TranslationModel and Views.
 */

import { 
  TranslationModel, 
  type TranslationRequest, 
  type TranslationResult 
} from '@/models';

export type TranslationStatus = 'idle' | 'processing' | 'success' | 'error' | 'offline';

export interface TranslationState {
  status: TranslationStatus;
  currentRequest: TranslationRequest | null;
  result: TranslationResult | null;
  error: string | null;
}

/**
 * TranslationController class for handling translation actions
 */
export class TranslationController {
  private model: TranslationModel;
  private state: TranslationState;
  private onStateChange?: (state: TranslationState) => void;

  constructor(onChange?: (state: TranslationState) => void) {
    this.model = TranslationModel.getInstance();
    this.state = {
      status: 'idle',
      currentRequest: null,
      result: null,
      error: null,
    };
    this.onStateChange = onChange;
  }

  /**
   * Get current translation state
   */
  getState(): TranslationState {
    return { ...this.state };
  }

  /**
   * Get device ID
   */
  getDeviceId(): string {
    return this.model.getDeviceId();
  }

  /**
   * Submit a translation request
   */
  async submitTranslation(text: string): Promise<TranslationResult> {
    // Check online status
    if (!this.model.isOnline()) {
      this.updateState({
        status: 'offline',
        error: 'No internet connection',
      });
      return this.model.createErrorResult('No internet connection');
    }

    // Create and validate request
    const request = this.model.createRequest(text);
    const validation = this.model.validateRequest(request);

    if (!validation.valid) {
      this.updateState({
        status: 'error',
        error: validation.error || 'Invalid request',
      });
      return this.model.createErrorResult(validation.error || 'Invalid request');
    }

    // Set processing state
    this.updateState({
      status: 'processing',
      currentRequest: request,
      error: null,
    });

    try {
      // Simulate API call (replace with real API in production)
      await this.simulateApiCall(request);

      // Success
      const result = this.model.createSuccessResult(crypto.randomUUID());
      this.updateState({
        status: 'success',
        result,
      });

      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Translation failed';
      const result = this.model.createErrorResult(errorMessage);
      
      this.updateState({
        status: 'error',
        result,
        error: errorMessage,
      });

      return result;
    }
  }

  /**
   * Retry the last translation
   */
  async retryTranslation(): Promise<TranslationResult | null> {
    if (!this.state.currentRequest) {
      return null;
    }
    return this.submitTranslation(this.state.currentRequest.text);
  }

  /**
   * Reset to idle state
   */
  reset(): void {
    this.updateState({
      status: 'idle',
      currentRequest: null,
      result: null,
      error: null,
    });
  }

  /**
   * Check if currently processing
   */
  isProcessing(): boolean {
    return this.state.status === 'processing';
  }

  /**
   * Set the change callback
   */
  setOnChange(callback: (state: TranslationState) => void): void {
    this.onStateChange = callback;
  }

  /**
   * Update state and notify listeners
   */
  private updateState(updates: Partial<TranslationState>): void {
    this.state = { ...this.state, ...updates };
    if (this.onStateChange) {
      this.onStateChange(this.getState());
    }
  }

  /**
   * Simulate API call (replace with real implementation)
   */
  private async simulateApiCall(request: TranslationRequest): Promise<void> {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 2500));

    // Check if still online after delay
    if (!this.model.isOnline()) {
      throw new Error('Connection lost');
    }

    // In production, this would make a real API call:
    // const response = await fetch('/api/translate', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({
    //     text: request.text,
    //     deviceId: this.model.getDeviceId(),
    //   }),
    // });
    // if (!response.ok) throw new Error('API error');
    // return response.json();
  }
}

/**
 * Create a new TranslationController instance
 */
export function createTranslationController(
  onChange?: (state: TranslationState) => void
): TranslationController {
  return new TranslationController(onChange);
}
