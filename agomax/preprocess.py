# ==============================================================================
# process.py
# PURPOSE:
#   Clean + preprocess data for ML
#   - Keep only numeric columns (even if stored as strings)
#   - Drop non-convertible columns
#   - Handle NaN / Inf
#   - Scale using StandardScaler
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# PREPROCESSOR
# ==============================================================================
class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.fill_values = None
        self.fitted = False

    @staticmethod
    def _coerce_numeric(df):
        df = df.copy()
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def fit(self, df):
        if df.empty:
            raise ValueError("Empty DataFrame")

        df = self._coerce_numeric(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=1, how="all")

        if df.shape[1] == 0:
            raise ValueError("No numeric columns")

        # compute fill values from TRAIN
        self.fill_values = df.median()
        df = df.fillna(self.fill_values)

        self.feature_cols = df.columns.tolist()
        X = self.scaler.fit_transform(df.values)

        self.fitted = True
        return X

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Not fitted")

        df = self._coerce_numeric(df)
        df = df.replace([np.inf, -np.inf], np.nan)

        # add missing columns
        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = self.fill_values[c]

        df = df[self.feature_cols]
        df = df.fillna(self.fill_values)

        return self.scaler.transform(df.values)
# ==============================================================================
# CLI / SELF TEST
# ==============================================================================

if __name__ == "__main__":
    print("[TEST] Running preprocessing self-test")

    # Example dirty data
    df = pd.DataFrame({
        "altitude": ["10.5", "11.2", "bad", "12.1"],
        "velocity": ["0.1", "0.2", "0.3", "oops"],
        "status": ["OK", "OK", "FAIL", "OK"],
        "battery": [15.9, 15.8, 15.7, 15.6]
    })

    print("\n[RAW DATA]")
    print(df)

    prep = Preprocessor()

    X_train = prep.fit(df)
    print("\n[TRAIN SCALED]")
    print(X_train)

    X_test = prep.transform(df)
    print("\n[TEST SCALED]")
    print(X_test)

    print("\n[FEATURES USED]")
    print(prep.feature_cols)

    print("\n[DONE] process.py self-test completed")
