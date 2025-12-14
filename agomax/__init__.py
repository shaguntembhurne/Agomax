"""
Agomax: Production-grade unsupervised anomaly detection for drone telemetry.

This package provides robust, easy-to-use anomaly detection trained on normal
flight data and deployed for real-time or batch inference.

Main API
--------
AgoMaxDetector : Main detector class
    Train, predict, save, and load anomaly detection models

Configuration
-------------
DetectorConfig : Main detector configuration
EnsembleConfig : Model ensemble configuration
PreprocessorConfig : Preprocessing configuration
ThresholdConfig : Threshold configuration

Results
-------
AnomalyResult : Detection result container

Exceptions
----------
AgoMaxError : Base exception
NotFittedError : Model not trained
InvalidDataError : Invalid input data
ModelNotFoundError : Model files not found
ConfigurationError : Invalid configuration
FeatureMismatchError : Feature mismatch

Utilities
---------
load_data : Load data from various sources
save_data : Save data to file

Examples
--------
>>> from agomax import AgoMaxDetector
>>> import pandas as pd
>>> 
>>> # Train on normal data
>>> train_df = pd.read_csv("normal_flights.csv")
>>> detector = AgoMaxDetector()
>>> detector.fit(train_df)
>>> detector.save("models/")
>>> 
>>> # Load and predict
>>> detector = AgoMaxDetector.load("models/")
>>> test_df = pd.read_csv("new_flight.csv")
>>> result = detector.predict(test_df)
>>> print(f"Anomalies: {result.labels.sum()}")
"""

__version__ = "0.1.0"

from .detector import AgoMaxDetector, AnomalyResult
from .config import (
    DetectorConfig,
    EnsembleConfig,
    PreprocessorConfig,
    ThresholdConfig,
)
from .exceptions import (
    AgoMaxError,
    NotFittedError,
    InvalidDataError,
    ModelNotFoundError,
    ConfigurationError,
    FeatureMismatchError,
)
from .utils import load_data, save_data

# Backward compatibility (deprecated)
from .compat import Pipeline

# Public API
__all__ = [
    # Main API
    "AgoMaxDetector",
    "AnomalyResult",
    # Configuration
    "DetectorConfig",
    "EnsembleConfig",
    "PreprocessorConfig",
    "ThresholdConfig",
    # Exceptions
    "AgoMaxError",
    "NotFittedError",
    "InvalidDataError",
    "ModelNotFoundError",
    "ConfigurationError",
    "FeatureMismatchError",
    # Utilities
    "load_data",
    "save_data",
    # Deprecated
    "Pipeline",
    # Version
    "__version__",
]
