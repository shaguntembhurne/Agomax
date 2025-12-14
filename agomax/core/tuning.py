"""
Hyperparameter tuning for anomaly detection models.

Tunes parameters to minimize false positives on normal training data.
"""

import numpy as np
from typing import Dict, Any

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import pairwise_distances

from ..config import EnsembleConfig


class HyperparameterTuner:
    """
    Hyperparameter tuner for ensemble models.
    
    Tunes scoring models (KMeans, LOF, OCSVM) to minimize false positives
    on normal training data. Explainers use stable defaults.
    """
    
    def __init__(
        self,
        max_anomaly_rate: float = 0.01,
        random_state: int = 42
    ):
        """
        Initialize tuner.
        
        Parameters
        ----------
        max_anomaly_rate : float
            Maximum acceptable anomaly rate on training data
        random_state : int
            Random seed for reproducibility
        """
        self.max_anomaly_rate = max_anomaly_rate
        self.random_state = random_state
    
    def _anomaly_rate(self, scores: np.ndarray) -> float:
        """Compute anomaly rate using mean + 3*std threshold."""
        mu = scores.mean()
        sigma = scores.std()
        threshold = mu + 3 * sigma
        return float(np.mean(scores > threshold))
    
    def tune_kmeans(self, X: np.ndarray) -> Dict[str, Any]:
        """Tune KMeans parameters."""
        for n_clusters in [2, 3, 4, 5]:
            model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init="auto"
            ).fit(X)
            
            # Compute distances to cluster centers
            distances = np.min(
                pairwise_distances(X, model.cluster_centers_),
                axis=1
            )
            
            if self._anomaly_rate(distances) <= self.max_anomaly_rate:
                return {"n_clusters": n_clusters, "max_iter": 300}
        
        return {"n_clusters": 2, "max_iter": 300}
    
    def tune_lof(self, X: np.ndarray) -> Dict[str, Any]:
        """Tune LOF parameters."""
        for n_neighbors in [10, 20, 30, 40, 50]:
            model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                novelty=True
            ).fit(X)
            
            scores = -model.score_samples(X)
            
            if self._anomaly_rate(scores) <= self.max_anomaly_rate:
                return {"n_neighbors": n_neighbors, "metric": "euclidean"}
        
        return {"n_neighbors": 20, "metric": "euclidean"}
    
    def tune_ocsvm(self, X: np.ndarray) -> Dict[str, Any]:
        """Tune One-Class SVM parameters."""
        for nu in [0.005, 0.01, 0.02, 0.05]:
            for gamma in ["scale", "auto", 0.1]:
                model = OneClassSVM(
                    nu=nu,
                    gamma=gamma,
                    kernel="rbf"
                ).fit(X)
                
                scores = -model.score_samples(X)
                
                if self._anomaly_rate(scores) <= self.max_anomaly_rate:
                    return {
                        "nu": nu,
                        "gamma": gamma,
                        "kernel": "rbf"
                    }
        
        return {"nu": 0.01, "gamma": "scale", "kernel": "rbf"}
    
    def tune_dbscan(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Get DBSCAN parameters (not tuned for anomaly rate).
        
        Used for structural context only.
        """
        return {"eps": 1.2, "min_samples": 20}
    
    def tune_optics(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Get OPTICS parameters (not tuned for anomaly rate).
        
        Used for structural context only.
        """
        return {"min_samples": 20, "xi": 0.05}
    
    def tune_all(self, X: np.ndarray) -> EnsembleConfig:
        """
        Tune all model parameters.
        
        Parameters
        ----------
        X : np.ndarray
            Normal training data
            
        Returns
        -------
        config : EnsembleConfig
            Tuned configuration
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=float)
        
        if X.shape[0] < 100:
            # Not enough data for tuning, use defaults
            return EnsembleConfig(random_state=self.random_state)
        
        # Tune each model
        kmeans_params = self.tune_kmeans(X)
        lof_params = self.tune_lof(X)
        ocsvm_params = self.tune_ocsvm(X)
        dbscan_params = self.tune_dbscan(X)
        optics_params = self.tune_optics(X)
        
        # Create config
        return EnsembleConfig(
            kmeans_n_clusters=kmeans_params["n_clusters"],
            kmeans_max_iter=kmeans_params["max_iter"],
            lof_n_neighbors=lof_params["n_neighbors"],
            lof_metric=lof_params["metric"],
            ocsvm_nu=ocsvm_params["nu"],
            ocsvm_gamma=ocsvm_params["gamma"],
            ocsvm_kernel=ocsvm_params["kernel"],
            dbscan_eps=dbscan_params["eps"],
            dbscan_min_samples=dbscan_params["min_samples"],
            optics_min_samples=optics_params["min_samples"],
            optics_xi=optics_params["xi"],
            random_state=self.random_state,
        )
