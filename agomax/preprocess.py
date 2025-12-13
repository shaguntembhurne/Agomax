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
        self.fitted = False

    # --------------------------------------------------------------------------
    # INTERNAL: force numeric conversion
    # --------------------------------------------------------------------------
    @staticmethod
    def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns to numeric where possible.
        Non-convertible columns become NaN.
        """
        numeric_df = df.copy()

        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(
                numeric_df[col],
                errors="coerce"
            )

        return numeric_df

    # --------------------------------------------------------------------------
    # FIT (TRAIN DATA)
    # --------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        # Force numeric conversion
        df_num = self._coerce_numeric(df)

        # Drop columns that are entirely NaN
        df_num = df_num.dropna(axis=1, how="all")

        if df_num.shape[1] == 0:
            raise ValueError("No numeric columns after conversion")

        # Replace inf with NaN, then drop rows with NaN
        df_num = df_num.replace([np.inf, -np.inf], np.nan)
        df_num = df_num.dropna(axis=0)

        self.feature_cols = df_num.columns.tolist()

        X = df_num.values
        X_scaled = self.scaler.fit_transform(X)

        self.fitted = True
        return X_scaled

    # --------------------------------------------------------------------------
    # TRANSFORM (TEST / INFERENCE DATA)
    # --------------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor is not fitted")

        df_num = self._coerce_numeric(df)

        # Ensure same feature set
        missing = set(self.feature_cols) - set(df_num.columns)
        if missing:
            raise ValueError(f"Missing features in input data: {missing}")

        df_num = df_num[self.feature_cols]

        df_num = df_num.replace([np.inf, -np.inf], np.nan)
        df_num = df_num.dropna(axis=0)

        X = df_num.values
        return self.scaler.transform(X)


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
