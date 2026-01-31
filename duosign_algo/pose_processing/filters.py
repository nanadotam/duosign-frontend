#!/usr/bin/env python3
"""
1€ Filter Implementation for Pose Landmark Smoothing
====================================================

This module implements the 1€ (One Euro) Filter, an adaptive low-pass filter
designed for filtering noisy signals in interactive systems.

Reference:
    Casiez, G., Roussel, N., & Vogel, D. (2012). 
    1€ filter: a simple speed-based low-pass filter for noisy input in interactive systems.
    Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, 2527-2530.

Author: Nana Kwaku Amoako
Date: 2026-01-31
"""

import numpy as np
from typing import Optional


class LowPassFilter:
    """
    First-order low-pass filter using exponential smoothing.
    
    This is the building block for the 1€ filter. It applies exponential
    moving average to smooth a signal.
    
    Attributes:
        last_value (Optional[float]): Previous filtered value
        initialized (bool): Whether filter has been initialized with data
    """
    
    def __init__(self):
        """Initialize an empty low-pass filter."""
        self.last_value: Optional[float] = None
        self.initialized: bool = False
    
    def filter(self, value: float, alpha: float) -> float:
        """
        Apply low-pass filtering to a single value.
        
        Uses exponential moving average: y[i] = α*x[i] + (1-α)*y[i-1]
        
        Args:
            value (float): Current raw measurement
            alpha (float): Smoothing factor in range [0, 1]
                - 0: No change (maximum smoothing)
                - 1: No smoothing (use raw value)
        
        Returns:
            float: Filtered value
        
        Example:
            >>> lpf = LowPassFilter()
            >>> lpf.filter(10.0, 0.5)  # First call, returns raw value
            10.0
            >>> lpf.filter(12.0, 0.5)  # Blend 50% new, 50% old
            11.0
        """
        # First call: initialize with raw value (no history to smooth with)
        if not self.initialized:
            self.last_value = value
            self.initialized = True
            return value
        
        # Exponential moving average: blend current value with history
        # Higher alpha = more responsive (less smoothing)
        # Lower alpha = more smoothing (less responsive)
        filtered = alpha * value + (1.0 - alpha) * self.last_value
        self.last_value = filtered
        
        return filtered
    
    def reset(self):
        """
        Reset filter state.
        
        Call this when starting a new sequence to clear history.
        """
        self.last_value = None
        self.initialized = False


class OneEuroFilter:
    """
    1€ Filter: Adaptive low-pass filter with velocity-based cutoff adjustment.
    
    The 1€ filter dynamically adjusts its cutoff frequency based on signal velocity,
    providing:
        - Low jitter at low speeds (high smoothing)
        - Low lag at high speeds (low smoothing)
    
    This makes it ideal for sign language pose data where both precision (when hands
    are stationary) and responsiveness (during rapid movements) are critical.
    
    Attributes:
        freq (float): Sampling frequency in Hz (typically video FPS)
        min_cutoff (float): Minimum cutoff frequency in Hz
        beta (float): Speed coefficient (controls lag reduction)
        d_cutoff (float): Derivative filter cutoff in Hz
        x_filter (LowPassFilter): Filter for the signal
        dx_filter (LowPassFilter): Filter for the velocity
        last_time (Optional[float]): Timestamp of last sample
    
    Tuning Guide:
        1. Set beta = 0, adjust min_cutoff to eliminate jitter when stationary
        2. Increase beta to reduce lag during fast movements
        3. Typical values: min_cutoff ∈ [0.5, 2.0], beta ∈ [0.001, 0.01]
    """
    
    def __init__(
        self,
        freq: float = 30.0,          # Sampling frequency (Hz) - typically video FPS
        min_cutoff: float = 1.0,     # Minimum cutoff frequency (Hz) - controls jitter
        beta: float = 0.007,         # Speed coefficient - controls lag reduction
        d_cutoff: float = 1.0        # Derivative cutoff (Hz) - smooths velocity estimate
    ):
        """
        Initialize 1€ filter with tuning parameters.
        
        Args:
            freq (float): Sampling frequency in Hz (e.g., 30 for 30 FPS video)
            min_cutoff (float): Minimum cutoff frequency in Hz
                - Lower values = more smoothing = less jitter (but more lag)
                - Typical range: 0.5 to 2.0 Hz
            beta (float): Speed coefficient
                - Higher values = less lag during fast movements
                - Typical range: 0.001 to 0.01
            d_cutoff (float): Derivative filter cutoff in Hz
                - Usually kept at 1.0 Hz
                - Smooths the velocity estimate to prevent oscillations
        
        Example:
            >>> # For fingerspelling (precise, slow movements)
            >>> filter_precise = OneEuroFilter(freq=30, min_cutoff=0.5, beta=0.001)
            >>> 
            >>> # For dynamic signs (balanced)
            >>> filter_balanced = OneEuroFilter(freq=30, min_cutoff=1.0, beta=0.007)
            >>> 
            >>> # For classifiers (fast, sweeping movements)
            >>> filter_responsive = OneEuroFilter(freq=30, min_cutoff=1.5, beta=0.015)
        """
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        # Two low-pass filters: one for position, one for velocity
        self.x_filter = LowPassFilter()   # Filters the actual signal
        self.dx_filter = LowPassFilter()  # Filters the velocity estimate
        
        self.last_time: Optional[float] = None
    
    def alpha(self, cutoff: float) -> float:
        """
        Calculate smoothing factor (alpha) from cutoff frequency.
        
        Derivation:
            τ = 1 / (2π * f_c)  # Time constant
            α = 1 / (1 + τ/T)   # Smoothing factor
            α = 1 / (1 + 1/(2π * f_c * freq))
        
        Args:
            cutoff (float): Cutoff frequency in Hz
        
        Returns:
            float: Smoothing factor between 0 and 1
        """
        tau = 1.0 / (2.0 * np.pi * cutoff)  # Time constant
        te = 1.0 / self.freq                 # Sampling period
        return 1.0 / (1.0 + tau / te)
    
    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        """
        Filter a single value with adaptive smoothing.
        
        Algorithm:
        1. Compute velocity (dx/dt) from change since last sample
        2. Smooth velocity with fixed cutoff (d_cutoff)
        3. Compute adaptive cutoff: f_c = min_cutoff + beta * |velocity|
        4. Smooth position with adaptive cutoff
        
        The key innovation is step 3: the cutoff frequency increases with velocity,
        automatically reducing smoothing (and lag) during fast movements while
        maintaining high smoothing (low jitter) during slow movements.
        
        Args:
            x (float): Current raw measurement
            timestamp (Optional[float]): Optional timestamp in seconds
                If None, assumes constant sampling rate (1/freq)
        
        Returns:
            float: Filtered value
        
        Example:
            >>> filter_obj = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.007)
            >>> 
            >>> # Filter a sequence of noisy measurements
            >>> measurements = [10.0, 10.2, 9.8, 10.1, 15.0, 20.0]  # Sudden jump
            >>> filtered = []
            >>> for i, x in enumerate(measurements):
            ...     filtered.append(filter_obj.filter(x, i/30.0))
            >>> 
            >>> # Result: smooth during slow changes, responsive during fast changes
        """
        # Handle first call: no previous value to compute velocity
        if self.last_time is None:
            self.last_time = timestamp if timestamp is not None else 0.0
            return self.x_filter.filter(x, 1.0)  # No smoothing on first sample
        
        # Compute time delta (for variable frame rates)
        if timestamp is not None:
            dt = timestamp - self.last_time
            self.last_time = timestamp
        else:
            dt = 1.0 / self.freq  # Assume constant sampling rate
        
        # Step 1: Estimate velocity (rate of change)
        # This tells us how fast the signal is moving
        if self.x_filter.last_value is not None:
            dx = (x - self.x_filter.last_value) / dt  # Units: value/second
        else:
            dx = 0.0
        
        # Step 2: Smooth the velocity estimate
        # This prevents noise in velocity from causing cutoff oscillations
        dx_smooth = self.dx_filter.filter(dx, self.alpha(self.d_cutoff))
        
        # Step 3: Compute adaptive cutoff frequency
        # KEY INNOVATION: cutoff increases with speed
        # - Slow movement (small |dx|) → low cutoff → high smoothing → low jitter
        # - Fast movement (large |dx|) → high cutoff → low smoothing → low lag
        cutoff = self.min_cutoff + self.beta * abs(dx_smooth)
        
        # Step 4: Filter the actual signal with adaptive cutoff
        return self.x_filter.filter(x, self.alpha(cutoff))
    
    def reset(self):
        """
        Reset filter state.
        
        Call this when starting a new video sequence to clear history.
        """
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class LandmarkFilter:
    """
    Applies 1€ filter to all coordinates of all landmarks across a video sequence.
    
    This is the main interface for filtering pose data. It creates and manages
    separate 1€ filters for each landmark coordinate (N landmarks × 3 coordinates).
    
    Attributes:
        fps (float): Video frame rate
        min_cutoff (float): Minimum cutoff frequency for 1€ filter
        beta (float): Speed coefficient for 1€ filter
        d_cutoff (float): Derivative cutoff for 1€ filter
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        """
        Initialize landmark filter.
        
        Args:
            fps (float): Video frame rate in Hz
            min_cutoff (float): Jitter reduction parameter (lower = more smoothing)
            beta (float): Lag reduction parameter (higher = less lag at high speeds)
            d_cutoff (float): Derivative filter cutoff (usually 1.0 Hz)
        """
        self.fps = fps
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
    
    def filter_landmarks(
        self,
        landmarks: np.ndarray,      # Shape: (T, N, 3) - T frames, N landmarks, 3 coords
        confidence: np.ndarray       # Shape: (T, N) - confidence per landmark
    ) -> np.ndarray:
        """
        Filter all landmarks across all frames.
        
        Strategy:
        - Create one 1€ filter per landmark per coordinate (N × 3 filters total)
        - Process frames sequentially (temporal coherence is important)
        - Skip filtering for low-confidence landmarks (preserve NaN)
        
        Args:
            landmarks (np.ndarray): Raw landmark coordinates with shape (T, N, 3)
                - T: Number of frames
                - N: Number of landmarks (e.g., 543 for MediaPipe Holistic)
                - 3: Coordinates (x, y, z)
            confidence (np.ndarray): Detection confidence with shape (T, N)
                - Values in range [0, 1]
        
        Returns:
            np.ndarray: Filtered landmarks with same shape as input (T, N, 3)
        
        Example:
            >>> import numpy as np
            >>> 
            >>> # Load pose data
            >>> data = np.load("sign.pose", allow_pickle=True)
            >>> landmarks = data["landmarks"]  # Shape: (100, 543, 3)
            >>> confidence = data["confidence"]  # Shape: (100, 543)
            >>> fps = float(data["fps"])
            >>> 
            >>> # Create filter
            >>> landmark_filter = LandmarkFilter(
            ...     fps=fps,
            ...     min_cutoff=1.0,
            ...     beta=0.007
            ... )
            >>> 
            >>> # Apply filtering
            >>> filtered_landmarks = landmark_filter.filter_landmarks(landmarks, confidence)
            >>> 
            >>> # Result: filtered_landmarks has same shape as landmarks
            >>> assert filtered_landmarks.shape == landmarks.shape
        """
        T, N, _ = landmarks.shape  # T = frames, N = landmarks
        filtered = np.copy(landmarks)
        
        # Create one filter per landmark per coordinate
        # filters[landmark_idx][coord_idx] = OneEuroFilter instance
        filters = [
            [
                OneEuroFilter(
                    freq=self.fps,
                    min_cutoff=self.min_cutoff,
                    beta=self.beta,
                    d_cutoff=self.d_cutoff
                )
                for _ in range(3)  # x, y, z
            ]
            for _ in range(N)  # For each landmark
        ]
        
        # Process frames sequentially (important for temporal coherence)
        for t in range(T):
            timestamp = t / self.fps  # Convert frame index to time
            
            for lm_idx in range(N):
                # Skip if landmark is missing (NaN) or low confidence
                # Threshold of 0.3 is empirically chosen - adjust as needed
                if np.isnan(landmarks[t, lm_idx]).any() or confidence[t, lm_idx] < 0.3:
                    continue
                
                # Filter each coordinate independently
                for coord_idx in range(3):  # x, y, z
                    raw_value = landmarks[t, lm_idx, coord_idx]
                    filtered_value = filters[lm_idx][coord_idx].filter(raw_value, timestamp)
                    filtered[t, lm_idx, coord_idx] = filtered_value
        
        return filtered


# Example usage and testing
if __name__ == "__main__":
    """
    Example: Filter a synthetic noisy signal to demonstrate 1€ filter behavior.
    """
    import matplotlib.pyplot as plt
    
    # Generate test signal: smooth sine wave + noise
    t = np.linspace(0, 2*np.pi, 100)
    clean_signal = np.sin(t)
    noisy_signal = clean_signal + np.random.normal(0, 0.1, len(t))
    
    # Test different parameter settings
    configs = [
        {"min_cutoff": 0.5, "beta": 0.001, "label": "High smoothing (low jitter)"},
        {"min_cutoff": 1.0, "beta": 0.007, "label": "Balanced (recommended)"},
        {"min_cutoff": 2.0, "beta": 0.02, "label": "Low smoothing (low lag)"},
    ]
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, clean_signal, 'g-', label='Clean signal', linewidth=2)
    plt.plot(t, noisy_signal, 'r.', label='Noisy signal', alpha=0.5)
    
    for config in configs:
        filter_obj = OneEuroFilter(freq=30.0, **{k: v for k, v in config.items() if k != 'label'})
        filtered = [filter_obj.filter(x, i/30.0) for i, x in enumerate(noisy_signal)]
        plt.plot(t, filtered, label=config['label'])
    
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('1€ Filter Comparison')
    plt.grid(True, alpha=0.3)
    plt.savefig('one_euro_filter_demo.png')
    print("Demo plot saved to one_euro_filter_demo.png")
