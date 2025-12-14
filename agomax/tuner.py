# ==============================================================================
# tuner.py
# PURPOSE:
#   Hyperparameter tuning for Agomax (NORMAL DATA ONLY)
#
#   - Scorers (KMeans, LOF, OCSVM):
#       tuned to minimize false positives on NORMAL data
#
#   - Explainers (DBSCAN, OPTICS):
#       NOT tuned for anomaly rate
#       kept stable for structural context only
# ==============================================================================

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import pairwise_distances


class HyperparameterTuner:
    def __init__(
        self,
        max_anomaly_rate=0.01,
        random_state=42,
    ):
        self.max_anomaly_rate = max_anomaly_rate
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _rate(self, flags):
        return float(np.mean(flags))

    # ------------------------------------------------------------------
    # KMEANS (SCORER)
    # ------------------------------------------------------------------
    def tune_kmeans(self, X):
        best = None

        for k in [2, 3, 4]:
            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init="auto"
            )
            model.fit(X)

            dist = np.min(
                pairwise_distances(X, model.cluster_centers_),
                axis=1
            )

            # static sanity check (NOT runtime logic)
            thresh = np.mean(dist) + 3 * np.std(dist)
            rate = self._rate(dist > thresh)

            if rate <= self.max_anomaly_rate:
                best = {"n_clusters": k}
                break

        return best or {"n_clusters": 2}

    # ------------------------------------------------------------------
    # LOF (SCORER)
    # ------------------------------------------------------------------
    def tune_lof(self, X):
        for k in [20, 30, 40]:
            model = LocalOutlierFactor(
                n_neighbors=k,
                novelty=True
            )
            model.fit(X)

            scores = -model.score_samples(X)
            thresh = np.mean(scores) + 3 * np.std(scores)
            rate = self._rate(scores > thresh)

            if rate <= self.max_anomaly_rate:
                return {"n_neighbors": k}

        return {"n_neighbors": 30}

    # ------------------------------------------------------------------
    # OCSVM (SCORER)
    # ------------------------------------------------------------------
    def tune_ocsvm(self, X):
        for nu in [0.005, 0.01, 0.02]:
            for gamma in ["scale", 0.1]:
                model = OneClassSVM(
                    nu=nu,
                    gamma=gamma
                )
                model.fit(X)

                scores = -model.score_samples(X)
                thresh = np.mean(scores) + 3 * np.std(scores)
                rate = self._rate(scores > thresh)

                if rate <= self.max_anomaly_rate:
                    return {"nu": nu, "gamma": gamma}

        return {"nu": 0.01, "gamma": "scale"}

    # ------------------------------------------------------------------
    # DBSCAN (EXPLAINER ONLY)
    # ------------------------------------------------------------------
    def tune_dbscan(self, X):
        """
        NOT tuned for anomaly rate.
        Just stable defaults for structure.
        """
        return {
            "eps": 1.2,
            "min_samples": 20
        }

    # ------------------------------------------------------------------
    # OPTICS (EXPLAINER ONLY)
    # ------------------------------------------------------------------
    def tune_optics(self, X):
        """
        NOT tuned for anomaly rate.
        Used only for reachability context.
        """
        return {
            "min_samples": 20,
            "xi": 0.05
        }

    # ------------------------------------------------------------------
    # TUNE ALL
    # ------------------------------------------------------------------
    def tune_all(self, X):
        return {
            "kmeans": self.tune_kmeans(X),
            "lof": self.tune_lof(X),
            "ocsvm": self.tune_ocsvm(X),
            "dbscan": self.tune_dbscan(X),
            "optics": self.tune_optics(X),
        }


# ==============================================================================
# SELF TEST
# ==============================================================================

if __name__ == "__main__":
    print("[TEST] Hyperparameter tuner sanity test")

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(3000, 10))

    tuner = HyperparameterTuner()
    params = tuner.tune_all(X)

    for model, p in params.items():
        print(f"[OK] {model}: {p}")

    print("[DONE] tuner.py test completed")
