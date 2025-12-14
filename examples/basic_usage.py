"""
Basic usage example for Agomax.

Demonstrates training on normal data and detecting anomalies.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from agomax import AgoMaxDetector


def generate_normal_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic normal flight telemetry."""
    rng = np.random.default_rng(42)
    
    return pd.DataFrame({
        "altitude": rng.normal(100, 2, n_samples),
        "velocity_z": rng.normal(0, 0.2, n_samples),
        "roll": rng.normal(0, 1, n_samples),
        "pitch": rng.normal(0, 1, n_samples),
        "yaw": rng.normal(0, 1, n_samples),
        "battery": rng.normal(15.5, 0.1, n_samples),
    })


def generate_test_data(n_samples: int = 300) -> pd.DataFrame:
    """Generate test data with injected anomalies."""
    rng = np.random.default_rng(123)
    
    df = pd.DataFrame({
        "altitude": rng.normal(100, 2, n_samples),
        "velocity_z": rng.normal(0, 0.2, n_samples),
        "roll": rng.normal(0, 1, n_samples),
        "pitch": rng.normal(0, 1, n_samples),
        "yaw": rng.normal(0, 1, n_samples),
        "battery": rng.normal(15.5, 0.1, n_samples),
    })
    
    # Inject anomaly event (altitude drop + instability)
    anomaly_start = 180
    anomaly_end = 200
    df.loc[anomaly_start:anomaly_end, "altitude"] -= 15
    df.loc[anomaly_start:anomaly_end, "roll"] += 10
    df.loc[anomaly_start:anomaly_end, "battery"] -= 1.5
    
    return df


def main():
    print("=" * 60)
    print("Agomax Basic Example")
    print("=" * 60)
    
    # Create output directory
    model_dir = Path("example_models")
    model_dir.mkdir(exist_ok=True)
    
    # Generate training data
    print("\n1. Generating normal training data...")
    train_df = generate_normal_data(n_samples=1000)
    print(f"   Training data shape: {train_df.shape}")
    print(f"   Features: {list(train_df.columns)}")
    
    # Train detector
    print("\n2. Training detector...")
    detector = AgoMaxDetector()
    detector.fit(train_df)
    print("   ✓ Training complete")
    
    # Save model
    print("\n3. Saving model...")
    detector.save(model_dir)
    print(f"   ✓ Saved to {model_dir}")
    
    # Load model
    print("\n4. Loading model...")
    detector = AgoMaxDetector.load(model_dir)
    print("   ✓ Model loaded")
    
    # Generate test data
    print("\n5. Generating test data with anomalies...")
    test_df = generate_test_data(n_samples=300)
    print(f"   Test data shape: {test_df.shape}")
    print(f"   Anomaly injected at samples 180-200")
    
    # Predict
    print("\n6. Running detection...")
    result = detector.predict(test_df)
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total samples:     {len(result.scores)}")
    print(f"Anomalies found:   {result.labels.sum()}")
    print(f"Events detected:   {result.events.sum()}")
    print(f"Mean score:        {result.scores.mean():.3f}")
    print(f"Max score:         {result.scores.max():.3f}")
    
    # Show where anomalies were detected
    anomaly_indices = np.where(result.labels == 1)[0]
    if len(anomaly_indices) > 0:
        print(f"\nAnomaly indices: {anomaly_indices[:10]}...")  # Show first 10
    
    # Explain a few samples
    print("\n7. Getting explanations...")
    result_explained = detector.predict(test_df.iloc[180:190], explain=True)
    
    print("\nSample explanations (indices 180-189):")
    for detail in result_explained.details[:5]:
        print(f"\n  Sample {detail['sample_index']}:")
        print(f"    Anomaly: {detail['is_anomaly']}")
        print(f"    Score: {detail['anomaly_score']:.3f}")
        print(f"    Top contributors: {detail['top_contributors']}")
    
    print("\n" + "=" * 60)
    print("✓ Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
