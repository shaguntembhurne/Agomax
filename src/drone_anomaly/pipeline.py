import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

class DronePipeline:
    def __init__(self, id_column: str = "timestamp", scaler_path: str = "models/scaler.pkl"):
        self.id_column = id_column
        self.data = None
        self.ids = None
        self.scaler = StandardScaler()
        self.scaler_path = scaler_path
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    def load_data(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")
        # FIX: Keep 'None' as a string, don't turn it into NaN
        self.data = pd.read_csv(filepath, na_values=['nan', 'NaN']) 
        return self

    def clean(self):
        """Standard cleaning: Drop NaNs, Duplicates, keep Numerics, REMOVE LABELS."""
        if self.id_column not in self.data.columns:
            print(f"Warning: ID column '{self.id_column}' not found. Using index.")
            self.data[self.id_column] = self.data.index

        # FIX: Prevent Data Leakage (Drop Ground Truth)
        # We also drop 'true_root_cause' so it doesn't mess up numeric filtering
        cols_to_drop = [c for c in ["is_synthetic_anomaly", "true_root_cause"] if c in self.data.columns]
        if cols_to_drop:
            self.data.drop(columns=cols_to_drop, inplace=True)

        # 1. Identify Numeric Columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if self.id_column in numeric_cols:
            numeric_cols.remove(self.id_column)
            
        # 2. Robust Filter (FIXED)
        # Only drop rows if valid NUMERIC data is missing. 
        self.data.dropna(subset=numeric_cols, inplace=True)
        self.data.drop_duplicates(subset=numeric_cols, inplace=True)
        
        # 3. Store IDs separate from features
        self.ids = self.data[self.id_column].reset_index(drop=True)
        self.data = self.data[numeric_cols].reset_index(drop=True)
        
        # Final safety check
        if len(self.data) == 0:
            raise ValueError("Data cleaning removed all rows! Check your CSV format.")
            
        return self

    def fit_transform_scaler(self):
        print("Fitting scaler on training data...")
        # Save columns to ensure order matches later
        self.feature_names_ = self.data.columns.tolist()
        joblib.dump(self.feature_names_, self.scaler_path.replace(".pkl", "_cols.pkl"))
        
        scaled_data = self.scaler.fit_transform(self.data)
        joblib.dump(self.scaler, self.scaler_path)
        
        self.data = pd.DataFrame(scaled_data, columns=self.data.columns)
        return self.data, self.ids

    def transform_scaler(self):
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError("Scaler not found. Run fit_transform_scaler on train data first.")
        
        # print("Loading scaler and transforming new data...") # Optional: Comment out for cleaner real-time logs
        self.scaler = joblib.load(self.scaler_path)
        
        # Validation: Ensure Test columns match Train columns exactly
        train_cols = joblib.load(self.scaler_path.replace(".pkl", "_cols.pkl"))
        self.data = self.data[train_cols]
        
        scaled_data = self.scaler.transform(self.data)
        self.data = pd.DataFrame(scaled_data, columns=train_cols)
        return self.data, self.ids