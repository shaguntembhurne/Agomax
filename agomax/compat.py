"""
Backward compatibility layer.

Provides the old Pipeline API for existing code while using the new implementation.
This module will be deprecated in future versions.
"""

import warnings
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np

from .detector import AgoMaxDetector
from .config import DetectorConfig


class Pipeline:
    """
    DEPRECATED: Use AgoMaxDetector instead.
    
    This class provides backward compatibility with the old Pipeline API.
    It will be removed in a future version.
    """
    
    def __init__(
        self,
        model_dir: str = "models",
        window: int = 50,
        k: float = 3.0,
        vote_threshold: float = 0.5,
        confirm_steps: int = 3,
        cooldown: int = 10,
    ):
        warnings.warn(
            "Pipeline is deprecated and will be removed in v0.2.0. "
            "Use AgoMaxDetector instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Create config from old parameters
        config = DetectorConfig(
            model_dir=Path(model_dir),
            vote_threshold=vote_threshold,
            confirmation_steps=confirm_steps,
            cooldown_steps=cooldown,
        )
        config.threshold.window_size = window
        config.threshold.std_multiplier = k
        
        # Use new implementation
        self._detector = AgoMaxDetector(config)
        self.model_dir = Path(model_dir)
        self.fitted = False
    
    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit pipeline on training data."""
        self._detector.fit(train_df)
        self.fitted = True
    
    def predict(
        self,
        df: pd.DataFrame,
        explain: bool = False
    ) -> tuple:
        """
        Predict anomalies.
        
        Returns
        -------
        If explain=False:
            (confidence, anomalies, events)
        If explain=True:
            (confidence, anomalies, events, explanations)
        """
        result = self._detector.predict(df, explain=explain)
        
        if explain:
            # Convert to old format
            explanations = [
                {
                    "confidence": detail["anomaly_score"],
                    "anomaly": int(detail["is_anomaly"]),
                    "event": int(detail["is_event"]),
                    "scores": detail["model_scores"],
                }
                for detail in result.details
            ]
            return result.scores, result.labels, result.events, explanations
        else:
            return result.scores, result.labels, result.events
    
    def load(self) -> None:
        """Load pipeline from disk."""
        self._detector = AgoMaxDetector.load(self.model_dir)
        self.fitted = True
    
    def _save(self) -> None:
        """Save pipeline to disk (internal)."""
        self._detector.save(self.model_dir)


# Add to __all__ in __init__.py if needed for backward compatibility
__all__ = ["Pipeline"]
