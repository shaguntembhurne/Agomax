"""
Configuration classes for Agomax.

Separates configuration from runtime state using dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class PreprocessorConfig:
    """Configuration for data preprocessing."""
    handle_missing: str = "median"  # 'median', 'mean', 'drop'
    handle_inf: bool = True
    drop_null_columns: bool = True
    
    def __post_init__(self):
        if self.handle_missing not in ["median", "mean", "drop"]:
            raise ValueError(f"Invalid handle_missing: {self.handle_missing}")


@dataclass
class EnsembleConfig:
    """Configuration for the anomaly detection ensemble."""
    
    # KMeans parameters
    kmeans_n_clusters: int = 2
    kmeans_max_iter: int = 300
    
    # LOF parameters
    lof_n_neighbors: int = 20
    lof_metric: str = "euclidean"
    
    # One-Class SVM parameters
    ocsvm_nu: float = 0.01
    ocsvm_gamma: str = "scale"
    ocsvm_kernel: str = "rbf"
    
    # DBSCAN parameters (explainer)
    dbscan_eps: float = 1.2
    dbscan_min_samples: int = 20
    
    # OPTICS parameters (explainer)
    optics_min_samples: int = 20
    optics_xi: float = 0.05
    
    # General
    random_state: int = 42
    
    def to_params_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert config to parameter dictionary for models."""
        return {
            "kmeans": {
                "n_clusters": self.kmeans_n_clusters,
                "max_iter": self.kmeans_max_iter,
                "random_state": self.random_state,
            },
            "lof": {
                "n_neighbors": self.lof_n_neighbors,
                "metric": self.lof_metric,
            },
            "ocsvm": {
                "nu": self.ocsvm_nu,
                "gamma": self.ocsvm_gamma,
                "kernel": self.ocsvm_kernel,
            },
            "dbscan": {
                "eps": self.dbscan_eps,
                "min_samples": self.dbscan_min_samples,
            },
            "optics": {
                "min_samples": self.optics_min_samples,
                "xi": self.optics_xi,
            },
        }


@dataclass
class ThresholdConfig:
    """Configuration for adaptive thresholding."""
    window_size: int = 50
    std_multiplier: float = 3.0
    min_samples: int = 10
    
    def __post_init__(self):
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.std_multiplier <= 0:
            raise ValueError("std_multiplier must be positive")
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1")


@dataclass
class DetectorConfig:
    """Main configuration for AgoMax anomaly detector."""
    
    # Ensemble voting
    vote_threshold: float = 0.5
    
    # Event detection
    confirmation_steps: int = 3
    cooldown_steps: int = 10
    
    # Model directory
    model_dir: Path = field(default_factory=lambda: Path("models"))
    
    # Component configs
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    
    # Auto-tuning
    auto_tune: bool = True
    tune_max_anomaly_rate: float = 0.01
    
    def __post_init__(self):
        if not isinstance(self.model_dir, Path):
            self.model_dir = Path(self.model_dir)
        
        if not (0 <= self.vote_threshold <= 1):
            raise ValueError("vote_threshold must be in [0, 1]")
        
        if self.confirmation_steps < 1:
            raise ValueError("confirmation_steps must be >= 1")
        
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be >= 0")
