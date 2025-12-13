# ==============================================================================
# pipeline.py
# PURPOSE:
#   Glue code for Agomax
#   - Train once on NORMAL data
#   - Save everything
#   - Load and detect anomalies on new data
# ==============================================================================

from pathlib import Path
import joblib
import numpy as np

from .preprocess import Preprocessor
from .tuner import HyperparameterTuner
import agomax.models as models
from .threshold import compute_threshold


class Pipeline:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.prep = None
        self.ensemble = None
        self.thresholds = None
        self.fitted = False

    # ------------------------------------------------------------------
    # TRAIN (NORMAL DATA ONLY)
    # ------------------------------------------------------------------
    def fit(self, train_df):
        """
        Train pipeline on NORMAL data only.
        """
        # 1️⃣ Preprocess
        self.prep = Preprocessor()
        X_train = self.prep.fit(train_df)

        # 2️⃣ Hyperparameter tuning
        tuner = HyperparameterTuner()
        params = tuner.tune_all(X_train)

        # 3️⃣ Train ensemble
        self.ensemble = models.Ensemble(params)
        self.ensemble.fit(X_train)

        # 4️⃣ Learn thresholds per model
        scores = self.ensemble.score(X_train)

        self.thresholds = {
            name: compute_threshold(score)
            for name, score in scores.items()
        }

        self.fitted = True
        self._save()

    # ------------------------------------------------------------------
    # DETECT (TEST / STREAM DATA)
    # ------------------------------------------------------------------
    def predict(self, df):
        """
        Detect anomalies on new data.
        """
        if not self.fitted:
            raise RuntimeError("Pipeline not fitted or loaded")

        X = self.prep.transform(df)
        scores = self.ensemble.score(X)

        # Threshold per model
        flags = np.vstack([
            (scores[name] > self.thresholds[name]).astype(int)
            for name in scores
        ]).T

        # RADD voting
        anomaly_score = flags.mean(axis=1)
        anomaly = (anomaly_score >= 0.4).astype(int)

        return anomaly_score, anomaly

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------
    def _save(self):
        joblib.dump(self.prep, self.model_dir / "preprocessor.joblib")
        self.ensemble.save(self.model_dir)
        joblib.dump(self.thresholds, self.model_dir / "thresholds.joblib")

    def load(self):
        self.prep = joblib.load(self.model_dir / "preprocessor.joblib")
        self.ensemble = models.Ensemble.load(self.model_dir)
        self.thresholds = joblib.load(self.model_dir / "thresholds.joblib")

        self.fitted = True
