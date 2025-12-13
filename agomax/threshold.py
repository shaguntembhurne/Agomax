# ==============================================================================
# threshold.py
# PURPOSE:
#   Robust thresholding utilities for anomaly scores
#   Default: 99.7 percentile
#   Optional: MAD
# ==============================================================================

import numpy as np


# ==============================================================================
# VALIDATION
# ==============================================================================

def _validate_scores(scores):
    if scores is None:
        raise ValueError("scores is None")

    scores = np.asarray(scores)

    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array")

    if len(scores) == 0:
        raise ValueError("scores is empty")

    if not np.isfinite(scores).all():
        raise ValueError("scores contains NaN or Inf")

    return scores


# ==============================================================================
# INTERNAL METHODS
# ==============================================================================

def _percentile_threshold(scores, percentile):
    if not (0 < percentile < 100):
        raise ValueError("percentile must be between 0 and 100")

    return float(np.percentile(scores, percentile))


def _mad_threshold(scores, k):
    if k <= 0:
        raise ValueError("k must be > 0")

    median = np.median(scores)
    mad = np.median(np.abs(scores - median))

    if mad == 0:
        return float(median)

    return float(median + k * mad)


# ==============================================================================
# PUBLIC API
# ==============================================================================

def compute_threshold(
    scores,
    method="percentile",
    percentile=99.7,
    k=3.5
):
    """
    Default:
        compute_threshold(scores) -> 99.7 percentile

    Optional:
        compute_threshold(scores, method="mad", k=3.5)
    """
    scores = _validate_scores(scores)

    if method == "percentile":
        return _percentile_threshold(scores, percentile)

    if method == "mad":
        return _mad_threshold(scores, k)

    raise ValueError(
        f"Unknown method '{method}'. "
        "Valid methods: 'percentile' (default), 'mad'"
    )


# ==============================================================================
# CLI / SELF TEST
# ==============================================================================

if __name__ == "__main__":
    print("[TEST] Running threshold self-test")

    # Simulated anomaly scores from NORMAL data
    rng = np.random.default_rng(42)
    scores = rng.normal(loc=0.0, scale=1.0, size=10_000)

    # Default percentile
    p_thresh = compute_threshold(scores)
    print(f"[OK] Percentile threshold (99.7): {p_thresh:.4f}")

    # MAD
    mad_thresh = compute_threshold(scores, method="mad", k=3.5)
    print(f"[OK] MAD threshold (k=3.5): {mad_thresh:.4f}")

    # Sanity check
    p_rate = (scores > p_thresh).mean() * 100
    mad_rate = (scores > mad_thresh).mean() * 100

    print(f"[INFO] Percentile exceed rate: {p_rate:.3f}%")
    print(f"[INFO] MAD exceed rate       : {mad_rate:.3f}%")

    print("[DONE] threshold.py self-test completed")
    
    """ Default (no thinking required)
    thresh = compute_threshold(train_scores)


    → 99.7 percentile automatically

    Explicit MAD (advanced user)
    thresh = compute_threshold(
        train_scores,
        method="mad",
        k=3.5
    )
    """