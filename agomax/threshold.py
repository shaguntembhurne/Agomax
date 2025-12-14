# ==============================================================================
# threshold.py
# PURPOSE:
#   Adaptive thresholding for anomaly scores using rolling statistics
#   (mean + k * std)
#
#   NOTE:
#   - Operates PER MODEL
#   - Operates PER TIME SERIES
#   - NO event aggregation
# ==============================================================================

import numpy as np


# ==============================================================================
# VALIDATION
# ==============================================================================

def _validate_scores(scores):
    if scores is None:
        raise ValueError("scores is None")

    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 1:
        raise ValueError("scores must be 1D")

    if len(scores) == 0:
        raise ValueError("scores is empty")

    if not np.isfinite(scores).all():
        raise ValueError("scores contains NaN or Inf")

    return scores


# ==============================================================================
# ADAPTIVE THRESHOLD (STREAMING)
# ==============================================================================

from collections import deque

class AdaptiveThreshold:
    def __init__(self, window=50, k=3.0):
        self.window = window
        self.k = k
        self.buffer = deque(maxlen=window)
        self.ready = False

    def update(self, score):
        """
        Update threshold state and return (is_anomaly, threshold).
        Only updates buffer if score is NOT anomalous (to prevent poisoning).
        """
        # 1. Compute current threshold based on PAST history
        if len(self.buffer) < 2:
            # Not enough history yet
            thresh = float('inf')
            self.ready = False
        else:
            # We have history
            series = np.array(self.buffer)
            mu = series.mean()
            sigma = series.std()
            
            if sigma == 0:
                thresh = mu
            else:
                thresh = mu + self.k * sigma
            self.ready = True

        # 2. Check anomaly
        is_anomaly = score > thresh

        # 3. Update history ONLY if normal (prevent threshold explosion)
        if not is_anomaly:
            self.buffer.append(score)

        return is_anomaly, thresh

    def get_state(self):
        return {
            "window": self.window,
            "k": self.k,
            "buffer": list(self.buffer)
        }

    def set_state(self, state):
        self.window = state["window"]
        self.k = state["k"]
        self.buffer = deque(state["buffer"], maxlen=self.window)


# ==============================================================================
# ROLLING THRESHOLD (BATCH)
# ==============================================================================

def rolling_threshold(
    scores,
    window=50,
    k=3.0,
    min_periods=None
):
    """
    Adaptive rolling threshold using mean + k * std.

    Parameters
    ----------
    scores : array-like
        Time-ordered anomaly scores (higher = more anomalous)
    window : int
        Rolling window size
    k : float
        Std multiplier (safety knob)
    min_periods : int or None
        Minimum samples before threshold is valid

    Returns
    -------
    thresholds : np.ndarray
        Adaptive threshold per timestep
    flags : np.ndarray (bool)
        score > threshold
    """

    scores = _validate_scores(scores)

    if window <= 1:
        raise ValueError("window must be > 1")

    if k <= 0:
        raise ValueError("k must be > 0")

    if min_periods is None:
        min_periods = window

    thresholds = np.full_like(scores, fill_value=np.nan)
    flags = np.zeros_like(scores, dtype=bool)

    for i in range(len(scores)):
        start = max(0, i - window)
        window_scores = scores[start:i]

        if len(window_scores) < min_periods:
            continue

        mu = window_scores.mean()
        sigma = window_scores.std()

        if sigma == 0:
            thresh = mu
        else:
            thresh = mu + k * sigma

        thresholds[i] = thresh
        flags[i] = scores[i] > thresh

    return thresholds, flags


# ==============================================================================
# CLI / SELF TEST
# ==============================================================================

if __name__ == "__main__":
    print("[TEST] Rolling threshold self-test")

    rng = np.random.default_rng(42)

    # Simulated normal scores + injected anomaly
    scores = rng.normal(0, 1, 300)
    scores[200:210] += 6.0  # anomaly burst

    thresholds, flags = rolling_threshold(
        scores,
        window=40,
        k=3.0
    )

    print(f"[INFO] Total points       : {len(scores)}")
    print(f"[INFO] Flagged points     : {flags.sum()}")
    print(f"[INFO] First anomaly idx  : {np.where(flags)[0][0]}")

    print("[DONE] threshold.py test completed")
