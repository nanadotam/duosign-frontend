/**
 * Translation Model
 * 
 * Manages translation requests and results.
 * Handles device identification for API requests.
 */

import type { TranslationRequest, TranslationResult, DeviceInfo } from './types';

const DEVICE_ID_KEY = "duosign_device_id";

/**
 * TranslationModel class for managing translation data
 */
export class TranslationModel {
  private static instance: TranslationModel;
  private deviceInfo: DeviceInfo | null = null;

  private constructor() {}

  /**
   * Get singleton instance
   */
  static getInstance(): TranslationModel {
    if (!TranslationModel.instance) {
      TranslationModel.instance = new TranslationModel();
    }
    return TranslationModel.instance;
  }

  /**
   * Get or create device ID for this client
   */
  getDeviceId(): string {
    if (typeof window === "undefined") return "";

    if (this.deviceInfo) {
      return this.deviceInfo.deviceId;
    }

    let deviceId = localStorage.getItem(DEVICE_ID_KEY);
    if (!deviceId) {
      deviceId = crypto.randomUUID();
      localStorage.setItem(DEVICE_ID_KEY, deviceId);
    }

    this.deviceInfo = {
      deviceId,
      lastActive: new Date(),
    };

    return deviceId;
  }

  /**
   * Create a translation request object
   */
  createRequest(text: string): TranslationRequest {
    return {
      text: text.trim(),
      timestamp: new Date(),
    };
  }

  /**
   * Validate translation request
   */
  validateRequest(request: TranslationRequest): { valid: boolean; error?: string } {
    if (!request.text || request.text.trim().length === 0) {
      return { valid: false, error: "Text cannot be empty" };
    }

    if (request.text.length > 500) {
      return { valid: false, error: "Text exceeds maximum length of 500 characters" };
    }

    return { valid: true };
  }

  /**
   * Create a success result
   */
  createSuccessResult(outputId: string): TranslationResult {
    return {
      success: true,
      outputId,
    };
  }

  /**
   * Create an error result
   */
  createErrorResult(error: string): TranslationResult {
    return {
      success: false,
      error,
    };
  }

  /**
   * Check if the device is online
   */
  isOnline(): boolean {
    if (typeof window === "undefined") return true;
    return navigator.onLine;
  }
}

// Convenience functions for functional usage
export function getDeviceId(): string {
  return TranslationModel.getInstance().getDeviceId();
}

export function createTranslationRequest(text: string): TranslationRequest {
  return TranslationModel.getInstance().createRequest(text);
}

export function validateTranslationRequest(request: TranslationRequest): { valid: boolean; error?: string } {
  return TranslationModel.getInstance().validateRequest(request);
}

export function isDeviceOnline(): boolean {
  return TranslationModel.getInstance().isOnline();
}
