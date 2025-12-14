"""
Data preprocessing module.

Handles numeric coercion, missing values, and feature scaling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, List

from ..exceptions import InvalidDataError, NotFittedError, FeatureMismatchError
from ..config import PreprocessorConfig


class Preprocessor:
    """
    Preprocessor for telemetry data.
    
    Converts all columns to numeric, handles missing/infinite values,
    and applies standard scaling.
    """
    
    def __init__(self, config: Optional[PreprocessorConfig] = None):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        config : PreprocessorConfig, optional
            Preprocessing configuration
        """
        self.config = config or PreprocessorConfig()
        self.scaler = StandardScaler()
        self.feature_names_: Optional[List[str]] = None
        self.fill_values_: Optional[pd.Series] = None
        self._fitted = False
    
    @property
    def is_fitted(self) -> bool:
        """Check if preprocessor has been fitted."""
        return self._fitted
    
    def _coerce_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all columns to numeric, coercing errors to NaN."""
        df = df.copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    
    def _validate_input(self, df: pd.DataFrame, stage: str = "fit"):
        """Validate input DataFrame."""
        if df is None:
            raise InvalidDataError("Input DataFrame is None")
        
        if not isinstance(df, pd.DataFrame):
            raise InvalidDataError(f"Expected pandas DataFrame, got {type(df)}")
        
        if df.empty:
            raise InvalidDataError("Input DataFrame is empty")
        
        if df.shape[0] == 0:
            raise InvalidDataError("Input DataFrame has no rows")
    
    def fit(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessor on training data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
            
        Returns
        -------
        np.ndarray
            Scaled feature matrix
            
        Raises
        ------
        InvalidDataError
            If input data is invalid or has no numeric columns
        """
        self._validate_input(df, "fit")
        
        # Coerce to numeric
        df = self._coerce_numeric(df)
        
        # Handle infinities
        if self.config.handle_inf:
            df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns that are all NaN
        if self.config.drop_null_columns:
            df = df.dropna(axis=1, how="all")
        
        if df.shape[1] == 0:
            raise InvalidDataError(
                "No valid numeric columns found after preprocessing. "
                "Ensure input data contains numeric values."
            )
        
        # Compute fill values
        if self.config.handle_missing == "median":
            self.fill_values_ = df.median()
        elif self.config.handle_missing == "mean":
            self.fill_values_ = df.mean()
        else:  # drop
            df = df.dropna()
            if df.empty:
                raise InvalidDataError("All rows contain missing values")
            self.fill_values_ = df.median()  # fallback for transform
        
        # Fill missing values
        df = df.fillna(self.fill_values_)
        
        # Store feature names
        self.feature_names_ = df.columns.tolist()
        
        # Fit scaler
        X = self.scaler.fit_transform(df.values)
        
        self._fitted = True
        return X
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Parameters
        ----------
        df : pd.DataFrame
            New data to transform
            
        Returns
        -------
        np.ndarray
            Scaled feature matrix
            
        Raises
        ------
        NotFittedError
            If preprocessor hasn't been fitted
        InvalidDataError
            If input data is invalid
        FeatureMismatchError
            If features don't match training data
        """
        if not self._fitted:
            raise NotFittedError("Preprocessor has not been fitted")
        
        self._validate_input(df, "transform")
        
        # Coerce to numeric
        df = self._coerce_numeric(df)
        
        # Handle infinities
        if self.config.handle_inf:
            df = df.replace([np.inf, -np.inf], np.nan)
        
        # Add missing columns with fill values
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = self.fill_values_[col]
        
        # Select only training features in same order
        df = df[self.feature_names_]
        
        # Fill missing values
        df = df.fillna(self.fill_values_)
        
        # Transform
        return self.scaler.transform(df.values)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit preprocessor and transform data in one step."""
        return self.fit(df)
