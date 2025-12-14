"""
Ensemble anomaly detection models.

Combines multiple unsupervised models for robust anomaly detection.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import pairwise_distances

from ..exceptions import NotFittedError, ModelNotFoundError
from ..config import EnsembleConfig


class AnomalyEnsemble:
    """
    Ensemble of unsupervised anomaly detection models.
    
    Combines scoring models (KMeans, LOF, OCSVM) with structural
    explainer models (DBSCAN, OPTICS) for robust detection.
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        """
        Initialize ensemble.
        
        Parameters
        ----------
        config : EnsembleConfig, optional
            Model configuration
        """
        self.config = config or EnsembleConfig()
        
        # Scoring models (contribute to voting)
        self._scorers: Dict = {}
        
        # Explainer models (structural context only)
        self._explainers: Dict = {}
        
        self._fitted = False
    
    @property
    def is_fitted(self) -> bool:
        """Check if ensemble has been fitted."""
        return self._fitted
    
    @property
    def scorer_names(self) -> list:
        """Names of scoring models."""
        return list(self._scorers.keys())
    
    def fit(self, X: np.ndarray) -> 'AnomalyEnsemble':
        """
        Fit ensemble on normal training data.
        
        Parameters
        ----------
        X : np.ndarray
            Training feature matrix (n_samples, n_features)
            Should contain ONLY normal/expected data
            
        Returns
        -------
        self
            Fitted ensemble
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=float)
        
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        
        if X.shape[0] < 10:
            raise ValueError("Need at least 10 samples for training")
        
        params = self.config.to_params_dict()
        
        # Fit scoring models
        self._scorers["kmeans"] = KMeans(
            **params["kmeans"],
            n_init="auto"
        ).fit(X)
        
        self._scorers["lof"] = LocalOutlierFactor(
            **params["lof"],
            novelty=True
        ).fit(X)
        
        self._scorers["ocsvm"] = OneClassSVM(
            **params["ocsvm"]
        ).fit(X)
        
        # Fit explainer models
        self._explainers["dbscan"] = DBSCAN(
            **params["dbscan"]
        ).fit(X)
        
        self._explainers["optics"] = OPTICS(
            **params["optics"]
        ).fit(X)
        
        self._fitted = True
        return self
    
    def score(self, X: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Compute anomaly scores and explanations.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix to score (n_samples, n_features)
            
        Returns
        -------
        scores : Dict[str, np.ndarray]
            Anomaly scores per model (higher = more anomalous)
        explanations : Dict[str, np.ndarray]
            Structural explanations (boolean flags)
            
        Raises
        ------
        NotFittedError
            If ensemble hasn't been fitted
        """
        if not self._fitted:
            raise NotFittedError("Ensemble has not been fitted")
        
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=float)
        
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        
        scores = {}
        explanations = {}
        
        # KMeans: distance to nearest cluster center
        centers = self._scorers["kmeans"].cluster_centers_
        scores["kmeans"] = np.min(
            pairwise_distances(X, centers),
            axis=1
        )
        
        # LOF: negative novelty score
        scores["lof"] = -self._scorers["lof"].score_samples(X)
        
        # OCSVM: negative decision function
        scores["ocsvm"] = -self._scorers["ocsvm"].score_samples(X)
        
        # DBSCAN: outlier flag (handle small sample sizes)
        try:
            explanations["dbscan_outlier"] = (
                self._explainers["dbscan"].fit_predict(X) == -1
            )
        except ValueError:
            # Not enough samples - mark all as non-outliers
            explanations["dbscan_outlier"] = np.zeros(X.shape[0], dtype=bool)
        
        # OPTICS: outlier flag (handle small sample sizes)
        try:
            explanations["optics_outlier"] = (
                self._explainers["optics"].fit_predict(X) == -1
            )
        except ValueError:
            # Not enough samples - mark all as non-outliers
            explanations["optics_outlier"] = np.zeros(X.shape[0], dtype=bool)
        
        return scores, explanations
    
    def save(self, directory: Path) -> None:
        """
        Save ensemble to disk.
        
        Parameters
        ----------
        directory : Path
            Directory to save model files
        """
        if not self._fitted:
            raise NotFittedError("Cannot save unfitted ensemble")
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self._scorers, directory / "scorers.joblib")
        joblib.dump(self._explainers, directory / "explainers.joblib")
        joblib.dump(self.config, directory / "ensemble_config.joblib")
    
    @classmethod
    def load(cls, directory: Path) -> 'AnomalyEnsemble':
        """
        Load ensemble from disk.
        
        Parameters
        ----------
        directory : Path
            Directory containing model files
            
        Returns
        -------
        ensemble : AnomalyEnsemble
            Loaded ensemble
            
        Raises
        ------
        ModelNotFoundError
            If model files don't exist
        """
        directory = Path(directory)
        
        # Check if files exist
        required_files = ["scorers.joblib", "explainers.joblib", "ensemble_config.joblib"]
        missing = [f for f in required_files if not (directory / f).exists()]
        
        if missing:
            raise ModelNotFoundError(directory)
        
        # Load components
        config = joblib.load(directory / "ensemble_config.joblib")
        scorers = joblib.load(directory / "scorers.joblib")
        explainers = joblib.load(directory / "explainers.joblib")
        
        # Create instance
        ensemble = cls(config)
        ensemble._scorers = scorers
        ensemble._explainers = explainers
        ensemble._fitted = True
        
        return ensemble
