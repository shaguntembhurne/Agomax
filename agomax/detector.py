"""
Main API for Agomax anomaly detection.

Provides the user-facing AgoMaxDetector class.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
import joblib

from .config import DetectorConfig
from .exceptions import NotFittedError, ModelNotFoundError, InvalidDataError
from .core.preprocessing import Preprocessor
from .core.ensemble import AnomalyEnsemble
from .core.threshold import AdaptiveThreshold
from .core.tuning import HyperparameterTuner


class AnomalyResult:
    """
    Result from anomaly detection.
    
    Attributes
    ----------
    scores : np.ndarray
        Continuous anomaly scores (0-1, higher = more anomalous)
    labels : np.ndarray
        Binary anomaly labels (0 = normal, 1 = anomaly)
    events : np.ndarray
        Confirmed anomaly events after temporal filtering
    details : Optional[List[Dict]]
        Per-sample explanation details (if explain=True)
    """
    
    def __init__(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        events: np.ndarray,
        details: Optional[List[Dict]] = None
    ):
        self.scores = scores
        self.labels = labels
        self.events = events
        self.details = details
    
    def __repr__(self) -> str:
        n_samples = len(self.scores)
        n_anomalies = int(self.labels.sum())
        n_events = int(self.events.sum())
        return (
            f"AnomalyResult(samples={n_samples}, "
            f"anomalies={n_anomalies}, events={n_events})"
        )


class AgoMaxDetector:
    """
    Unsupervised anomaly detector for drone telemetry.
    
    Uses an ensemble of unsupervised models trained on normal data
    to detect anomalous patterns in new telemetry.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from agomax import AgoMaxDetector
    >>> 
    >>> # Load normal training data
    >>> train_df = pd.read_csv("normal_flights.csv")
    >>> 
    >>> # Create and train detector
    >>> detector = AgoMaxDetector()
    >>> detector.fit(train_df)
    >>> 
    >>> # Save trained model
    >>> detector.save("models/")
    >>> 
    >>> # Load and use
    >>> detector = AgoMaxDetector.load("models/")
    >>> test_df = pd.read_csv("new_flight.csv")
    >>> result = detector.predict(test_df)
    >>> 
    >>> print(f"Found {result.labels.sum()} anomalies")
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize detector.
        
        Parameters
        ----------
        config : DetectorConfig, optional
            Configuration for the detector. If None, uses defaults.
        """
        self.config = config or DetectorConfig()
        
        # Internal components (created during fit)
        self._preprocessor: Optional[Preprocessor] = None
        self._ensemble: Optional[AnomalyEnsemble] = None
        self._thresholds: Dict[str, AdaptiveThreshold] = {}
        
        # Runtime state for event detection
        self._confirm_counter = 0
        self._cooldown_counter = 0
        
        self._fitted = False
    
    @property
    def is_fitted(self) -> bool:
        """Check if detector has been trained."""
        return self._fitted
    
    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        auto_tune: Optional[bool] = None
    ) -> 'AgoMaxDetector':
        """
        Train detector on normal data.
        
        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            Training data containing ONLY normal/expected patterns.
            If DataFrame, numeric columns will be extracted automatically.
            If ndarray, should be shape (n_samples, n_features).
        auto_tune : bool, optional
            Whether to auto-tune hyperparameters. Overrides config setting.
            
        Returns
        -------
        self
            Trained detector
            
        Raises
        ------
        InvalidDataError
            If input data is invalid
            
        Notes
        -----
        Training data should contain only normal behavior. The detector
        learns what "normal" looks like and flags deviations during inference.
        """
        if auto_tune is None:
            auto_tune = self.config.auto_tune
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise InvalidDataError(
                f"Expected DataFrame or ndarray, got {type(data)}"
            )
        
        # Preprocessing
        self._preprocessor = Preprocessor(self.config.preprocessor)
        X_train = self._preprocessor.fit(data)
        
        # Auto-tune if requested
        if auto_tune:
            tuner = HyperparameterTuner(
                max_anomaly_rate=self.config.tune_max_anomaly_rate,
                random_state=self.config.ensemble.random_state
            )
            ensemble_config = tuner.tune_all(X_train)
        else:
            ensemble_config = self.config.ensemble
        
        # Train ensemble
        self._ensemble = AnomalyEnsemble(ensemble_config)
        self._ensemble.fit(X_train)
        
        # Initialize adaptive thresholds with training data
        scores, _ = self._ensemble.score(X_train)
        self._thresholds = {}
        
        for model_name, score_array in scores.items():
            threshold = AdaptiveThreshold(
                window_size=self.config.threshold.window_size,
                std_multiplier=self.config.threshold.std_multiplier,
                min_samples=self.config.threshold.min_samples
            )
            # Pre-fill with training scores (assumed normal)
            for score in score_array[-threshold.window_size:]:
                threshold.buffer.append(float(score))
            
            self._thresholds[model_name] = threshold
        
        self._fitted = True
        return self
    
    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        explain: bool = False
    ) -> AnomalyResult:
        """
        Detect anomalies in new data.
        
        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            New data to analyze
        explain : bool, default=False
            If True, include detailed explanations in result
            
        Returns
        -------
        result : AnomalyResult
            Detection results with scores, labels, events, and optional details
            
        Raises
        ------
        NotFittedError
            If detector hasn't been trained
        InvalidDataError
            If input data is invalid
        """
        if not self._fitted:
            raise NotFittedError(
                "Detector has not been trained. Call fit() first."
            )
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise InvalidDataError(
                f"Expected DataFrame or ndarray, got {type(data)}"
            )
        
        # Preprocess
        X = self._preprocessor.transform(data)
        
        # Score with ensemble
        scores, explanations = self._ensemble.score(X)
        
        n_samples = X.shape[0]
        model_names = self._ensemble.scorer_names
        
        # Per-sample detection
        anomaly_labels = np.zeros(n_samples, dtype=int)
        anomaly_scores = np.zeros(n_samples, dtype=float)
        events = np.zeros(n_samples, dtype=int)
        details_list = [] if explain else None
        
        for i in range(n_samples):
            # Check each model's threshold
            model_flags = []
            model_thresholds = {}
            
            for model_name in model_names:
                score = float(scores[model_name][i])
                is_anomaly, thresh = self._thresholds[model_name].update(score)
                model_flags.append(int(is_anomaly))
                model_thresholds[model_name] = thresh
            
            # Vote aggregation
            vote_ratio = np.mean(model_flags)
            is_anomaly = vote_ratio >= self.config.vote_threshold
            
            anomaly_scores[i] = vote_ratio
            anomaly_labels[i] = int(is_anomaly)
            
            # Event detection (temporal confirmation)
            if self._cooldown_counter > 0:
                self._cooldown_counter -= 1
            elif is_anomaly:
                self._confirm_counter += 1
                if self._confirm_counter >= self.config.confirmation_steps:
                    events[i] = 1
                    self._confirm_counter = 0
                    self._cooldown_counter = self.config.cooldown_steps
            else:
                self._confirm_counter = 0
            
            # Explanations
            if explain:
                detail = {
                    "sample_index": i,
                    "anomaly_score": float(anomaly_scores[i]),
                    "is_anomaly": bool(anomaly_labels[i]),
                    "is_event": bool(events[i]),
                    "model_scores": {m: float(scores[m][i]) for m in model_names},
                    "model_flags": {m: bool(model_flags[j]) for j, m in enumerate(model_names)},
                    "model_thresholds": model_thresholds,
                    "vote_ratio": float(vote_ratio),
                }
                
                # Add top contributors
                if is_anomaly:
                    contributors = [
                        (m, scores[m][i])
                        for j, m in enumerate(model_names)
                        if model_flags[j]
                    ]
                    contributors.sort(key=lambda x: x[1], reverse=True)
                    detail["top_contributors"] = [m for m, _ in contributors]
                else:
                    detail["top_contributors"] = []
                
                details_list.append(detail)
        
        return AnomalyResult(
            scores=anomaly_scores,
            labels=anomaly_labels,
            events=events,
            details=details_list
        )
    
    def save(self, directory: Optional[Union[str, Path]] = None) -> None:
        """
        Save trained detector to disk.
        
        Parameters
        ----------
        directory : str or Path, optional
            Directory to save model files. If None, uses config.model_dir.
            
        Raises
        ------
        NotFittedError
            If detector hasn't been trained
        """
        if not self._fitted:
            raise NotFittedError("Cannot save unfitted detector")
        
        save_dir = Path(directory) if directory else self.config.model_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save components
        joblib.dump(self._preprocessor, save_dir / "preprocessor.joblib")
        self._ensemble.save(save_dir)
        
        # Save threshold states
        threshold_states = {
            name: thresh.get_state()
            for name, thresh in self._thresholds.items()
        }
        joblib.dump(threshold_states, save_dir / "thresholds.joblib")
        
        # Save config
        joblib.dump(self.config, save_dir / "detector_config.joblib")
    
    @classmethod
    def load(cls, directory: Union[str, Path]) -> 'AgoMaxDetector':
        """
        Load trained detector from disk.
        
        Parameters
        ----------
        directory : str or Path
            Directory containing model files
            
        Returns
        -------
        detector : AgoMaxDetector
            Loaded detector
            
        Raises
        ------
        ModelNotFoundError
            If model files don't exist
        """
        load_dir = Path(directory)
        
        # Check required files
        required = [
            "preprocessor.joblib",
            "scorers.joblib",
            "explainers.joblib",
            "thresholds.joblib",
            "detector_config.joblib"
        ]
        missing = [f for f in required if not (load_dir / f).exists()]
        
        if missing:
            raise ModelNotFoundError(load_dir)
        
        # Load config
        config = joblib.load(load_dir / "detector_config.joblib")
        
        # Create instance
        detector = cls(config)
        
        # Load components
        detector._preprocessor = joblib.load(load_dir / "preprocessor.joblib")
        detector._ensemble = AnomalyEnsemble.load(load_dir)
        
        # Load threshold states
        threshold_states = joblib.load(load_dir / "thresholds.joblib")
        detector._thresholds = {}
        for name, state in threshold_states.items():
            thresh = AdaptiveThreshold()
            thresh.set_state(state)
            detector._thresholds[name] = thresh
        
        detector._fitted = True
        return detector
    
    def reset_state(self) -> None:
        """
        Reset runtime state (thresholds, event counters).
        
        Useful when starting analysis of a new flight or data stream.
        """
        self._confirm_counter = 0
        self._cooldown_counter = 0
        
        for thresh in self._thresholds.values():
            thresh.reset()
