# ==============================================================================
# test.py
# PURPOSE:
#   End-to-end Agomax test
#   - Generate NORMAL training data
#   - Train + save pipeline
#   - Generate TEST data with anomalies
#   - Detect anomalies + events
# ==============================================================================

import numpy as np
import pandas as pd
from pathlib import Path

from agomax.pipeline import Pipeline


# ------------------------------------------------------------------
# DATA GENERATION
# ------------------------------------------------------------------

def make_train_data(n=1000):
    """Normal flight only"""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "altitude": rng.normal(100, 2, n),
        "velocity_z": rng.normal(0, 0.2, n),
        "roll": rng.normal(0, 1, n),
        "pitch": rng.normal(0, 1, n),
        "yaw": rng.normal(0, 1, n),
        "battery": rng.normal(15.5, 0.1, n),
    })


def make_test_data(n=300):
    """Normal + injected anomaly event"""
    rng = np.random.default_rng(123)

    df = pd.DataFrame({
        "altitude": rng.normal(100, 2, n),
        "velocity_z": rng.normal(0, 0.2, n),
        "roll": rng.normal(0, 1, n),
        "pitch": rng.normal(0, 1, n),
        "yaw": rng.normal(0, 1, n),
        "battery": rng.normal(15.5, 0.1, n),
    })

    # Inject anomaly burst
    df.loc[180:200, "altitude"] -= 15      # altitude loss
    df.loc[180:200, "roll"] += 10          # instability
    df.loc[180:200, "battery"] -= 1.5      # battery sag

    return df


# ------------------------------------------------------------------
# MAIN TEST
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("\n[TEST] Agomax end-to-end demo\n")

    model_dir = Path("agomax_store")
    model_dir.mkdir(exist_ok=True)

    # ---- TRAIN ----
    train_df = make_train_data()
    train_df.to_csv("train.csv", index=False)

    pipeline = Pipeline(
        model_dir=model_dir,
        window=40,
        k=3.0,
        vote_threshold=0.5,
        confirm_steps=3,
        cooldown=10,
    )

    pipeline.fit(train_df)

    print("\n[OK] Training complete")

    # ---- LOAD PIPELINE ----
    pipeline = Pipeline(
        model_dir=model_dir,
        window=40,
        k=3.0,
        vote_threshold=0.5,
        confirm_steps=3,
        cooldown=10,
    )
    pipeline.load()

    print("[OK] Pipeline loaded")

    # ---- TEST ----
    test_df = make_test_data()
    test_df.to_csv("test.csv", index=False)

    confidence, anomaly, event, explanations = pipeline.predict(
        test_df,
        explain=True
    )

    # ---- RESULTS ----
    print("\n[RESULTS]")
    print(f"Total samples      : {len(test_df)}")
    print(f"Anomaly flags      : {anomaly.sum()}")
    print(f"Anomaly EVENTS     : {event.sum()}")

    event_idx = np.where(event == 1)[0]
    if len(event_idx) > 0:
        print(f"First event index  : {event_idx[0]}")
    else:
        print("No events triggered")

    print("\n[SAMPLE EXPLANATION]")
    print(explanations[190])

    print("\n[DONE] test.py completed")
