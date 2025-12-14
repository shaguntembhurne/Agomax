"""
Adaptive threshold module for anomaly detection.

Provides streaming and batch threshold computation using rolling statistics.
"""

import numpy as np
from collections import deque
from typing import Tuple, Dict, Any, Optional

from ..exceptions import InvalidDataError


class AdaptiveThreshold:
    """
    Adaptive threshold using rolling mean + k * std.
    
    Maintains a buffer of recent scores and updates threshold dynamically.
    Only adds scores to buffer when they are NOT anomalous to prevent
    threshold poisoning.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        std_multiplier: float = 3.0,
        min_samples: int = 10
    ):
        """
        Initialize adaptive threshold.
        
        Parameters
        ----------
        window_size : int
            Size of rolling window for statistics
        std_multiplier : float
            Multiplier for standard deviation (safety margin)
        min_samples : int
            Minimum samples before threshold is active
        """
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if std_multiplier <= 0:
            raise ValueError("std_multiplier must be positive")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        
        self.window_size = window_size
        self.std_multiplier = std_multiplier
        self.min_samples = min_samples
        
        self.buffer = deque(maxlen=window_size)
        self._ready = False
    
    @property
    def is_ready(self) -> bool:
        """Check if threshold has enough samples."""
        return len(self.buffer) >= self.min_samples
    
    def update(self, score: float) -> Tuple[bool, float]:
        """
        Update threshold with new score and check if anomalous.
        
        Parameters
        ----------
        score : float
            New anomaly score
            
        Returns
        -------
        is_anomaly : bool
            Whether score exceeds threshold
        threshold : float
            Current threshold value
        """
        if not np.isfinite(score):
            raise InvalidDataError(f"Score must be finite, got {score}")
        
        # Compute threshold from current buffer
        if not self.is_ready:
            threshold = float('inf')
        else:
            arr = np.array(self.buffer)
            mu = arr.mean()
            sigma = arr.std()
            
            if sigma == 0:
                threshold = mu
            else:
                threshold = mu + self.std_multiplier * sigma
        
        # Check if anomalous
        is_anomaly = score > threshold
        
        # Update buffer only if NOT anomalous (prevent poisoning)
        if not is_anomaly:
            self.buffer.append(score)
        
        return is_anomaly, threshold
    
    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        return {
            "window_size": self.window_size,
            "std_multiplier": self.std_multiplier,
            "min_samples": self.min_samples,
            "buffer": list(self.buffer),
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore from serialized state."""
        self.window_size = state["window_size"]
        self.std_multiplier = state["std_multiplier"]
        self.min_samples = state["min_samples"]
        self.buffer = deque(state["buffer"], maxlen=self.window_size)
    
    def reset(self) -> None:
        """Reset threshold state."""
        self.buffer.clear()
        self._ready = False


def compute_static_threshold(
    scores: np.ndarray,
    method: str = "percentile",
    percentile: float = 99.7,
    std_multiplier: float = 3.0
) -> float:
    """
    Compute static threshold from training scores.
    
    Parameters
    ----------
    scores : np.ndarray
        Array of anomaly scores from training data
    method : str
        'percentile' or 'std'
    percentile : float
        Percentile value if method='percentile'
    std_multiplier : float
        Std multiplier if method='std'
        
    Returns
    -------
    threshold : float
        Computed threshold value
    """
    if not isinstance(scores, np.ndarray):
        scores = np.asarray(scores, dtype=float)
    
    if scores.ndim != 1:
        raise InvalidDataError("Scores must be 1-dimensional")
    
    if len(scores) == 0:
        raise InvalidDataError("Scores array is empty")
    
    if not np.isfinite(scores).all():
        raise InvalidDataError("Scores contain NaN or Inf values")
    
    if method == "percentile":
        return float(np.percentile(scores, percentile))
    elif method == "std":
        mu = scores.mean()
        sigma = scores.std()
        return float(mu + std_multiplier * sigma)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'percentile' or 'std'")
