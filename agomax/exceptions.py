"""
Exceptions for Agomax package.

This module defines custom exceptions for better error handling and user feedback.
"""


class AgoMaxError(Exception):
    """Base exception for all Agomax errors."""
    pass


class NotFittedError(AgoMaxError):
    """Raised when trying to use a model that hasn't been fitted."""
    def __init__(self, message="Model has not been fitted. Call fit() before using this method."):
        super().__init__(message)


class InvalidDataError(AgoMaxError):
    """Raised when input data is invalid or malformed."""
    pass


class ConfigurationError(AgoMaxError):
    """Raised when configuration parameters are invalid."""
    pass


class ModelNotFoundError(AgoMaxError):
    """Raised when trying to load a model that doesn't exist."""
    def __init__(self, model_dir):
        message = f"Model files not found in directory: {model_dir}. Train and save a model first."
        super().__init__(message)


class FeatureMismatchError(AgoMaxError):
    """Raised when features in inference data don't match training data."""
    def __init__(self, expected, received):
        message = (
            f"Feature mismatch: expected {len(expected)} features, got {len(received)}.\n"
            f"Expected features: {expected}\n"
            f"Received features: {received}"
        )
        super().__init__(message)
