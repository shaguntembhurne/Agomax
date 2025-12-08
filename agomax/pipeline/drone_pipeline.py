import numpy as np
import pandas as pd
import joblib
import os
import logging

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


# ============================================================
#                     LOGGING SETUP
# ============================================================

logger = logging.getLogger("AgomaX")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)

logger.handlers.clear()
logger.addHandler(handler)



# ============================================================
#                     MODEL MANAGER
# ============================================================

class ModelManager:
    def __init__(self, use_mad=False):
        self.use_mad = use_mad
        self.models = {}
        self.thresholds = {}

    # --------------------------------------------------------
    # SAVE / LOAD MODELS
    # --------------------------------------------------------
    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Saving models to: {folder_path}")

        try:
            for name, model in self.models.items():
                joblib.dump(model, f"{folder_path}/{name}.joblib")

            joblib.dump(self.thresholds, f"{folder_path}/thresholds.joblib")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def load(self, folder_path):
        logger.info(f"Loading models from: {folder_path}")

        if not os.path.exists(folder_path):
            logger.warning(f"No model folder found at: {folder_path}")
            return

        try:
            thresh_path = f"{folder_path}/thresholds.joblib"
            if os.path.exists(thresh_path):
                self.thresholds = joblib.load(thresh_path)

            for name in ["kmeans", "lof", "ocsvm", "dbscan", "optics"]:
                model_file = f"{folder_path}/{name}.joblib"
                if os.path.exists(model_file):
                    self.models[name] = joblib.load(model_file)
        except Exception as e:
            logger.error(f"Error loading models: {e}")



    # --------------------------------------------------------
    # THRESHOLD + SCORES
    # --------------------------------------------------------
    def compute_threshold(self, scores):
        threshold = np.percentile(scores, 99.7)

        if self.use_mad:
            median = np.median(scores)
            mad = np.median(np.abs(scores - median))
            threshold = median + 3 * mad

        return threshold

    def compute_scores(self, model, df, name):

        if name == "kmeans":
            distances = model.transform(df)
            return distances.min(axis=1)

        if name == "lof":
            return -model.negative_outlier_factor_

        if name == "ocsvm":
            return -model.decision_function(df)

        if name == "dbscan":
            return (model.labels_ == -1).astype(int)

        if name == "optics":
            reach = model.reachability_
            return np.nan_to_num(reach, nan=0.0, posinf=0.0)

        raise ValueError(f"Unknown model name: {name}")



    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    def train_models(self, df):

        logger.info(f"Training models on {len(df)} samples")

        try:
            kmeans = KMeans(n_clusters=1)
            lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
            ocsvm = OneClassSVM(kernel="rbf")
            dbscan = DBSCAN()
            optics = OPTICS()

            kmeans.fit(df)
            lof.fit(df)
            ocsvm.fit(df)
            dbscan.fit(df)
            optics.fit(df)

        except Exception as e:
            logger.error(f"Error fitting models: {e}")
            return

        scores_kmeans = self.compute_scores(kmeans, df, "kmeans")
        scores_lof    = self.compute_scores(lof, df, "lof")
        scores_ocsvm  = self.compute_scores(ocsvm, df, "ocsvm")
        scores_dbscan = self.compute_scores(dbscan, df, "dbscan")
        scores_optics = self.compute_scores(optics, df, "optics")

        self.thresholds["kmeans"] = self.compute_threshold(scores_kmeans)
        self.thresholds["lof"]    = self.compute_threshold(scores_lof)
        self.thresholds["ocsvm"]  = self.compute_threshold(scores_ocsvm)
        self.thresholds["dbscan"] = self.compute_threshold(scores_dbscan)
        self.thresholds["optics"] = self.compute_threshold(scores_optics)

        self.models["kmeans"] = kmeans
        self.models["lof"]    = lof
        self.models["ocsvm"]  = ocsvm
        self.models["dbscan"] = dbscan
        self.models["optics"] = optics

        logger.info("Model training complete.")



    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------
    def predict(self, df_row):
        scores = {}
        votes = {}

        for name, model in self.models.items():
            score = self.compute_scores(model, df_row, name)[0]
            scores[name] = score
            votes[name] = score > self.thresholds[name]

        anomaly = sum(votes.values()) >= 3

        logger.info(
            f"Prediction → Anomaly={anomaly} | Votes={votes}"
        )

        return anomaly, votes, scores



# ============================================================
#                     DRONE PIPELINE
# ============================================================

class DronePipeline:
    def __init__(self):
        self.data = None
        self.phase_models = {
            "idle": ModelManager(),
            "takeoff": ModelManager(),
            "hover": ModelManager(),
            "cruise": ModelManager(),
            "landing": ModelManager()
        }

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------
    def load(self, path):
        try:
            self.data = pd.read_csv(path)
            logger.info(f"Loaded dataset from {path} with {len(self.data)} rows.")
            return self.data
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------
    def preprocess(self):
        logger.info("Preprocessing data...")
        df = self.data.copy()

        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.debug(f"Skipping non-numeric column: {col}")
                continue

            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
                logger.debug(f"Filled NaN in column: {col}")

        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

        logger.info("Preprocessing complete.")
        return df



    # --------------------------------------------------------
    # PHASE DETECTION
    # --------------------------------------------------------
    def detect_phases(self, df):

        if not {"altitude", "velocity_z", "accel_z"}.issubset(df.columns):
            logger.error("Missing required columns for phase detection.")
            raise Exception("Required columns: altitude, velocity_z, accel_z")

        phases = []

        idle_vel_th = 0.2
        hover_vel_th = 0.5
        takeoff_acc_th = 0.6
        landing_acc_th = -0.6
        cruise_vel_th = 1.0

        altitude = df["altitude"].values
        velocity_z = df["velocity_z"].values
        accel_z = df["accel_z"].values

        altitude_diff = np.diff(altitude, prepend=altitude[0])

        for i in range(len(df)):
            alt_d = altitude_diff[i]
            vel = velocity_z[i]
            acc = accel_z[i]

            if abs(vel) < idle_vel_th and abs(alt_d) < 0.05:
                phases.append("idle")
                continue

            if acc > takeoff_acc_th and alt_d > 0.05:
                phases.append("takeoff")
                continue

            if acc < landing_acc_th and alt_d < -0.05:
                phases.append("landing")
                continue

            if abs(alt_d) < 0.05 and abs(vel) < hover_vel_th:
                phases.append("hover")
                continue

            if abs(vel) > cruise_vel_th:
                phases.append("cruise")
                continue

            phases.append("hover")

        df["phase"] = phases
        return df



    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    def train(self):
        logger.info("Starting training pipeline...")

        df = self.preprocess()
        df = self.detect_phases(df)

        for phase, model_mgr in self.phase_models.items():
            phase_df = df[df["phase"] == phase]

            if len(phase_df) < 20:
                logger.warning(f"Skipping {phase} → not enough data ({len(phase_df)} rows).")
                continue

            # remove phase column
            phase_df = phase_df.drop(columns=["phase"])

            logger.info(f"Training phase: {phase}")
            model_mgr.train_models(phase_df)

        logger.info("All phase models trained.")



    # --------------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------------
    def save_all(self, folder="models"):
        os.makedirs(folder, exist_ok=True)
        logger.info(f"Saving all phase models to: {folder}")

        for phase, model_mgr in self.phase_models.items():
            model_mgr.save(f"{folder}/{phase}")

    def load_all(self, folder="models"):
        logger.info(f"Loading model set from: {folder}")

        for phase, model_mgr in self.phase_models.items():
            model_mgr.load(f"{folder}/{phase}")



    # --------------------------------------------------------
    # DETECT
    # --------------------------------------------------------
    def detect(self, df_row):

        logger.info("Running anomaly detection...")

        df_row = df_row.copy()

        # scale
        num_cols = df_row.select_dtypes(include=["number"]).columns
        df_row[num_cols] = (
            df_row[num_cols] - self.data[num_cols].mean()
        ) / self.data[num_cols].std()

        # detect phase
        temp = self.detect_phases(df_row.copy())
        phase = temp.iloc[0]["phase"]

        logger.info(f"Detected phase: {phase}")

        model_mgr = self.phase_models.get(phase)

        if len(model_mgr.models) == 0:
            logger.warning(f"No trained model for phase: {phase}")
            return False, {}, {}

        # remove non-numeric
        df_row = df_row.drop(columns=["phase"])

        return model_mgr.predict(df_row)
