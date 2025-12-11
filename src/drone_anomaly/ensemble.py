import numpy as np
import pandas as pd
import joblib
import os
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier

class AgomaxEnsemble:
    def __init__(self, model_dir="models/"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # FIX: Lower contamination to reduce False Positives
        self.contamination = 0.01 
        
        # Initialize Models with clearer boundaries
        self.models = {
            "kmeans": KMeans(n_clusters=1, n_init=10),
            "ocsvm": OneClassSVM(nu=self.contamination, gamma='scale'),
            # FIX: Higher neighbors (30) makes LOF less sensitive to tiny jitters
            "lof": LocalOutlierFactor(n_neighbors=30, novelty=True, contamination=self.contamination),
            "dbscan": DBSCAN(eps=0.6, min_samples=5), # Slightly looser eps
            "optics": OPTICS(min_samples=5)
        }
        self.proxies = {} 
        self.feature_stats = {}

    def fit(self, X):
        """Train models and learn 'Normal' feature stats."""
        print("--- Training Agomax Ensemble ---")
        
        # Save Feature Stats for Root Cause Analysis
        self.feature_stats['mean'] = X.mean()
        self.feature_stats['std'] = X.std()
        
        for name, model in self.models.items():
            if name in ["dbscan", "optics"]:
                # Density models: Fit -> Get Labels -> Train KNN Proxy
                labels = model.fit_predict(X)
                # If everything is -1 (noise), force a class 0 to avoid crash
                if np.all(labels == -1): labels[0] = 0 
                    
                proxy = KNeighborsClassifier(n_neighbors=3)
                proxy.fit(X, labels)
                self.proxies[name] = proxy
            elif name == "kmeans":
                model.fit(X)
                # Learn distance threshold (99th percentile - very strict)
                dists = model.transform(X).min(axis=1)
                self.kmeans_thresh_ = np.percentile(dists, 99)
            else:
                model.fit(X)
                
        self.save_models()
        print("Training Complete.")

    def _get_root_cause(self, row):
        """Explains WHY a row is anomalous (Max Z-Score)."""
        z_scores = abs((row - self.feature_stats['mean']) / self.feature_stats['std'])
        return z_scores.idxmax(), z_scores.max()

    def predict(self, X):
        results = pd.DataFrame(index=X.index)
        
        # 1. Collect Votes
        dists = self.models['kmeans'].transform(X).min(axis=1)
        results['kmeans'] = (dists > self.kmeans_thresh_).astype(int)
        
        results['ocsvm'] = (self.models['ocsvm'].predict(X) == -1).astype(int)
        results['lof'] = (self.models['lof'].predict(X) == -1).astype(int)
        
        for name in ['dbscan', 'optics']:
            pred = self.proxies[name].predict(X)
            results[name] = (pred == -1).astype(int)

        # 2. Aggregation (FIX: Stricter Rules)
        results['vote_count'] = results.sum(axis=1)
        
        # Logic: Anomaly if 3+ models agree OR KMeans is VERY confident
        results['is_anomaly'] = (results['vote_count'] >= 3).astype(int)
        
        # 3. Root Cause Analysis
        results['root_cause_feature'] = "None"
        results['severity_score'] = 0.0
        
        anomaly_indices = results[results['is_anomaly'] == 1].index
        if len(anomaly_indices) > 0:
            print(f"Analyzing root cause for {len(anomaly_indices)} detected anomalies...")
            for idx in anomaly_indices:
                row_values = X.loc[idx]
                feat, score = self._get_root_cause(row_values)
                results.at[idx, 'root_cause_feature'] = feat
                results.at[idx, 'severity_score'] = round(score, 2)

        return results

    def save_models(self):
        joblib.dump(self.models, f"{self.model_dir}/models.pkl")
        joblib.dump(self.proxies, f"{self.model_dir}/proxies.pkl")
        joblib.dump(self.feature_stats, f"{self.model_dir}/stats.pkl")
        joblib.dump(self.kmeans_thresh_, f"{self.model_dir}/k_thresh.pkl")

    def load_models(self):
        self.models = joblib.load(f"{self.model_dir}/models.pkl")
        self.proxies = joblib.load(f"{self.model_dir}/proxies.pkl")
        self.feature_stats = joblib.load(f"{self.model_dir}/stats.pkl")
        self.kmeans_thresh_ = joblib.load(f"{self.model_dir}/k_thresh.pkl")