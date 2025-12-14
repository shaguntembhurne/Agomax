# ==============================================================================
# pipeline.py
# PURPOSE:
#   Orchestrates Agomax end-to-end with time-aware anomaly decisions
# ==============================================================================

from pathlib import Path
import joblib
import numpy as np
from collections import deque

from .logger import get_logger
log = get_logger("agomax.pipeline")

from .preprocess import Preprocessor
from .tuner import HyperparameterTuner
from .threshold import AdaptiveThreshold
from . import models


class Pipeline:
    def __init__(
        self,
        model_dir="models",
        window=50,
        k=3.0,
        vote_threshold=0.5,
        confirm_steps=3,
        cooldown=10,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.window = window
        self.k = k
        self.vote_threshold = vote_threshold
        self.confirm_steps = confirm_steps
        self.cooldown = cooldown

        self.prep = None
        self.ensemble = None
        self.fitted = False

        # runtime state
        self.thresholds = {}  # AdaptiveThreshold per model
        self.confirm_counter = 0
        self.cooldown_counter = 0

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    def fit(self, train_df):
        log.info("Starting training")

        self.prep = Preprocessor()
        X_train = self.prep.fit(train_df)

        tuner = HyperparameterTuner()
        params = tuner.tune_all(X_train)

        self.ensemble = models.Ensemble(params)
        self.ensemble.fit(X_train)

        # Initialize adaptive thresholds with training data stats
        scores, _ = self.ensemble.score(X_train)
        for name, s_array in scores.items():
            self.thresholds[name] = AdaptiveThreshold(window=self.window, k=self.k)
            # Pre-fill buffer with training scores to avoid cold start
            # We assume training data is mostly normal
            for s in s_array[-self.window:]:
                self.thresholds[name].buffer.append(float(s))

        self.fitted = True
        self._save()
        log.info("Training complete")

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------
    def predict(self, df, explain=False):
        if not self.fitted:
            raise RuntimeError("Pipeline not fitted")

        X = self.prep.transform(df)
        scores, explanations = self.ensemble.score(X)

        n = X.shape[0]
        model_names = list(scores.keys())

        # Ensure thresholds exist (for loaded models)
        for name in model_names:
            if name not in self.thresholds:
                self.thresholds[name] = AdaptiveThreshold(window=self.window, k=self.k)

        anomalies = np.zeros(n, dtype=int)
        confidences = np.zeros(n)
        events = np.zeros(n, dtype=int)

        for i in range(n):
            model_flags = []

            # -------------------------------
            # PER-MODEL SELECTIVE ADAPTATION
            # -------------------------------
            for name in model_names:
                score = float(scores[name][i])
                is_anomaly, _ = self.thresholds[name].update(score)
                model_flags.append(int(is_anomaly))

            vote_ratio = np.mean(model_flags)
            confidences[i] = vote_ratio

            is_anomaly = vote_ratio >= self.vote_threshold
            anomalies[i] = int(is_anomaly)

            # -------------------------------
            # EVENT AGGREGATION
            # -------------------------------
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                continue

            if is_anomaly:
                self.confirm_counter += 1
            else:
                self.confirm_counter = 0

            if self.confirm_counter >= self.confirm_steps:
                events[i] = 1
                self.confirm_counter = 0
                self.cooldown_counter = self.cooldown

        if not explain:
            return confidences, anomalies, events

        # -------------------------------
        # EXPLANATIONS
        # -------------------------------
        explanation_rows = []
        for i in range(n):
            explanation_rows.append({
                "confidence": float(confidences[i]),
                "anomaly": int(anomalies[i]),
                "event": int(events[i]),
                "scores": {m: float(scores[m][i]) for m in model_names},
            })

        return confidences, anomalies, events, explanation_rows

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------
    def _save(self):
        joblib.dump(self.prep, self.model_dir / "preprocessor.joblib")
        self.ensemble.save(self.model_dir)
        
        # Save threshold states
        thresh_states = {name: t.get_state() for name, t in self.thresholds.items()}
        joblib.dump(thresh_states, self.model_dir / "thresholds.joblib")

    def load(self):
        self.prep = joblib.load(self.model_dir / "preprocessor.joblib")
        self.ensemble = models.Ensemble.load(self.model_dir)
        
        # Load threshold states
        thresh_path = self.model_dir / "thresholds.joblib"
        if thresh_path.exists():
            thresh_states = joblib.load(thresh_path)
            self.thresholds = {}
            for name, state in thresh_states.items():
                t = AdaptiveThreshold()
                t.set_state(state)
                self.thresholds[name] = t
        
        self.fitted = True
