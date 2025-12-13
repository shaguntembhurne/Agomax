# ==============================================================================
# tuner.py
# PURPOSE:
#   Hyperparameter tuning for unsupervised anomaly models
#   Trains ONLY on NORMAL data
#   Returns best hyperparameters
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
        percentile=99.7,
        random_state=42
    ):
        self.max_anomaly_rate = max_anomaly_rate
        self.percentile = percentile
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _rate(self, flags):
        return np.mean(flags)

    def _percentile_thresh(self, scores):
        return np.percentile(scores, self.percentile)

    # ------------------------------------------------------------------
    # KMEANS
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

            flags = dist > self._percentile_thresh(dist)
            rate = self._rate(flags)

            if rate <= self.max_anomaly_rate:
                best = {"n_clusters": k}
                break

        if best is None:
            best = {"n_clusters": 2}

        return best

    # ------------------------------------------------------------------
    # DBSCAN
    # ------------------------------------------------------------------
    def tune_dbscan(self, X):
        for eps in [0.8, 1.2, 1.6]:
            for ms in [10, 20, 30]:
                model = DBSCAN(eps=eps, min_samples=ms)
                labels = model.fit_predict(X)

                rate = self._rate(labels == -1)
                if rate <= self.max_anomaly_rate:
                    return {"eps": eps, "min_samples": ms}

        return {"eps": 1.2, "min_samples": 20}

    # ------------------------------------------------------------------
    # OPTICS
    # ------------------------------------------------------------------
    def tune_optics(self, X):
        for ms in [10, 20, 30]:
            for xi in [0.03, 0.05, 0.1]:
                model = OPTICS(min_samples=ms, xi=xi)
                labels = model.fit_predict(X)

                rate = self._rate(labels == -1)
                if rate <= self.max_anomaly_rate:
                    return {"min_samples": ms, "xi": xi}

        return {"min_samples": 20, "xi": 0.05}

    # ------------------------------------------------------------------
    # LOF
    # ------------------------------------------------------------------
    def tune_lof(self, X):
        for k in [20, 30, 40]:
            model = LocalOutlierFactor(
                n_neighbors=k,
                novelty=True
            )
            model.fit(X)

            scores = -model.score_samples(X)
            flags = scores > self._percentile_thresh(scores)

            rate = self._rate(flags)
            if rate <= self.max_anomaly_rate:
                return {"n_neighbors": k}

        return {"n_neighbors": 30}

    # ------------------------------------------------------------------
    # OCSVM
    # ------------------------------------------------------------------
    def tune_ocsvm(self, X):
        for nu in [0.005, 0.01, 0.02]:
            for gamma in ["scale", 0.1, 0.01]:
                model = OneClassSVM(
                    nu=nu,
                    gamma=gamma
                )
                model.fit(X)

                scores = -model.score_samples(X)
                flags = scores > self._percentile_thresh(scores)

                rate = self._rate(flags)
                if rate <= self.max_anomaly_rate:
                    return {"nu": nu, "gamma": gamma}

        return {"nu": 0.01, "gamma": "scale"}

    # ------------------------------------------------------------------
    # TUNE ALL
    # ------------------------------------------------------------------
    def tune_all(self, X):
        return {
            "kmeans": self.tune_kmeans(X),
            "dbscan": self.tune_dbscan(X),
            "optics": self.tune_optics(X),
            "lof": self.tune_lof(X),
            "ocsvm": self.tune_ocsvm(X),
        }


# ==============================================================================
# SELF TEST
# ==============================================================================

if __name__ == "__main__":
    print("[TEST] Running hyperparameter tuner self-test")

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(3000, 10))

    tuner = HyperparameterTuner()
    params = tuner.tune_all(X)

    for model, p in params.items():
        print(f"[OK] {model}: {p}")

    print("[DONE] tuner.py self-test completed")
