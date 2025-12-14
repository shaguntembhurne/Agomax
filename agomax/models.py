import numpy as np
from pathlib import Path
import joblib

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import pairwise_distances


class Ensemble:
    def __init__(self, params: dict):
        self.params = params
        self.scorers = {}
        self.explainers = {}
        self.fitted = False

    # ------------------------------------------------------------------
    # FIT (NORMAL DATA ONLY)
    # ------------------------------------------------------------------
    def fit(self, X):
        # ---- SCORERS (vote-capable) ----
        self.scorers["kmeans"] = KMeans(
            **self.params["kmeans"],
            n_init="auto",
            random_state=42
        ).fit(X)

        self.scorers["lof"] = LocalOutlierFactor(
            **self.params["lof"],
            novelty=True
        ).fit(X)

        self.scorers["ocsvm"] = OneClassSVM(
            **self.params["ocsvm"]
        ).fit(X)

        # ---- EXPLAINERS (NO VOTING) ----
        self.explainers["dbscan"] = DBSCAN(
            **self.params["dbscan"]
        ).fit(X)

        self.explainers["optics"] = OPTICS(
            **self.params["optics"]
        ).fit(X)

        self.fitted = True

    # ------------------------------------------------------------------
    # SCORE (INFERENCE)
    # ------------------------------------------------------------------
    def score(self, X):
        if not self.fitted:
            raise RuntimeError("Ensemble not fitted")

        scores = {}
        explanations = {}

        # ---- SCORERS ----
        centers = self.scorers["kmeans"].cluster_centers_
        scores["kmeans"] = np.min(
            pairwise_distances(X, centers), axis=1
        )

        scores["lof"] = -self.scorers["lof"].score_samples(X)
        scores["ocsvm"] = -self.scorers["ocsvm"].score_samples(X)

        # ---- EXPLAINERS ----
        explanations["dbscan_outlier"] = (
            self.explainers["dbscan"].fit_predict(X) == -1
        )

        explanations["optics_outlier"] = (
            self.explainers["optics"].fit_predict(X) == -1
        )

        return scores, explanations

    # ------------------------------------------------------------------
    # SIMPLE DECISION (DEMO ONLY)
    # ------------------------------------------------------------------
    def decide(self, scores, explanations, thresholds):
        """
        thresholds: dict per scorer
        """
        flags = {}

        for k, v in scores.items():
            flags[k] = v > thresholds[k]

        vote_ratio = (
            flags["kmeans"].astype(int) +
            flags["lof"].astype(int) +
            flags["ocsvm"].astype(int)
        ) / 3.0

        # Structural confirmation
        structure_support = (
            explanations["dbscan_outlier"].astype(int) +
            explanations["optics_outlier"].astype(int)
        )

        return vote_ratio, structure_support
        # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------
    def save(self, directory):
        """
        Save ensemble models to disk.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.scorers, path / "scorers.joblib")
        joblib.dump(self.explainers, path / "explainers.joblib")
        joblib.dump(self.params, path / "params.joblib")

    @classmethod
    def load(cls, directory):
        """
        Load ensemble models from disk.
        """
        path = Path(directory)

        params = joblib.load(path / "params.joblib")
        scorers = joblib.load(path / "scorers.joblib")
        explainers = joblib.load(path / "explainers.joblib")

        obj = cls(params)
        obj.scorers = scorers
        obj.explainers = explainers
        obj.fitted = True
        return obj

if __name__ == "__main__":
    print("\n[TEST] Structural usefulness demo\n")

    rng = np.random.default_rng(42)

    # ---- NORMAL TRAIN DATA ----
    X_train = rng.normal(0, 1, size=(1000, 4))

    # ---- TEST DATA ----
    X_test = np.vstack([
        rng.normal(0, 1, size=(20, 4)),      # normal
        rng.normal(5, 0.3, size=(5, 4))      # true anomaly cluster
    ])

    params = {
        "kmeans": {"n_clusters": 2},
        "lof": {"n_neighbors": 20},
        "ocsvm": {"nu": 0.05, "gamma": "scale"},
        "dbscan": {"eps": 0.8, "min_samples": 10},
        "optics": {"min_samples": 10}
    }

    thresholds = {
        "kmeans": 2.5,
        "lof": 1.5,
        "ocsvm": 0.5
    }

    ens = Ensemble(params)
    ens.fit(X_train)

    scores, explanations = ens.score(X_test)
    vote_ratio, structure = ens.decide(scores, explanations, thresholds)

    for i in range(len(X_test)):
        print(
            f"Sample {i:02d} | "
            f"vote={vote_ratio[i]:.2f} | "
            f"struct_support={structure[i]} | "
            f"dbscan={explanations['dbscan_outlier'][i]} | "
            f"optics={explanations['optics_outlier'][i]}"
        )
