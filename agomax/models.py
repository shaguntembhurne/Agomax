# ==============================================================================
# ensemble.py
# PURPOSE:
#   Train + score unsupervised anomaly ensemble
#   Models:
#     - KMeans
#     - DBSCAN
#     - OPTICS
#     - LOF
#     - One-Class SVM
# ==============================================================================

import numpy as np
from pathlib import Path
import joblib

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import pairwise_distances


class Ensemble:
    def __init__(self, params: dict):
        """
        params: dict from HyperparameterTuner.tune_all()
        """
        self.params = params
        self.models = {}
        self.fitted = False

    # ------------------------------------------------------------------
    # FIT (TRAIN)
    # ------------------------------------------------------------------
    def fit(self, X):
        """
        Fit all models on SCALED training data.
        """
        # KMeans
        self.models["kmeans"] = KMeans(
            **self.params["kmeans"],
            n_init="auto",
            random_state=42
        )
        self.models["kmeans"].fit(X)

        # DBSCAN (density reference only)
        self.models["dbscan"] = DBSCAN(**self.params["dbscan"])
        self.models["dbscan"].fit(X)

        # OPTICS
        self.models["optics"] = OPTICS(**self.params["optics"])
        self.models["optics"].fit(X)

        # LOF (novelty mode for inference)
        self.models["lof"] = LocalOutlierFactor(
            **self.params["lof"],
            novelty=True
        )
        self.models["lof"].fit(X)

        # OCSVM
        self.models["ocsvm"] = OneClassSVM(**self.params["ocsvm"])
        self.models["ocsvm"].fit(X)

        self.fitted = True

    # ------------------------------------------------------------------
    # SCORE (TEST / STREAM)
    # ------------------------------------------------------------------
    def score(self, X):
        """
        Return per-model anomaly scores.
        Higher = more anomalous.
        """
        if not self.fitted:
            raise RuntimeError("Ensemble is not fitted")

        scores = {}

        # KMeans distance
        centers = self.models["kmeans"].cluster_centers_
        scores["kmeans"] = np.min(
            pairwise_distances(X, centers),
            axis=1
        )

        # DBSCAN (binary)
        scores["dbscan"] = (
            self.models["dbscan"].fit_predict(X) == -1
        ).astype(float)

        # OPTICS (binary)
        scores["optics"] = (
            self.models["optics"].fit_predict(X) == -1
        ).astype(float)

        # LOF
        scores["lof"] = -self.models["lof"].score_samples(X)

        # OCSVM
        scores["ocsvm"] = -self.models["ocsvm"].score_samples(X)

        return scores

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------
    def save(self, directory: str):
        """
        Save ensemble models to disk.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.models, path / "models.joblib")
        joblib.dump(self.params, path / "params.joblib")

    @classmethod
    def load(cls, directory: str):
        """
        Load ensemble from disk.
        """
        path = Path(directory)

        params = joblib.load(path / "params.joblib")
        models = joblib.load(path / "models.joblib")

        obj = cls(params)
        obj.models = models
        obj.fitted = True
        return obj


# ==============================================================================
# SELF TEST
# ==============================================================================

if __name__ == "__main__":
    print("[TEST] Running ensemble self-test")

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(1000, 8))

    dummy_params = {
        "kmeans": {"n_clusters": 2},
        "dbscan": {"eps": 1.2, "min_samples": 20},
        "optics": {"min_samples": 20, "xi": 0.05},
        "lof": {"n_neighbors": 30},
        "ocsvm": {"nu": 0.01, "gamma": "scale"},
    }

    ens = Ensemble(dummy_params)
    ens.fit(X)

    scores = ens.score(X)
    for k, v in scores.items():
        print(f"[OK] {k}: score shape {v.shape}")

    ens.save("ensemble_store")
    ens2 = Ensemble.load("ensemble_store")

    scores2 = ens2.score(X)
    print("[DONE] ensemble.py self-test completed")
