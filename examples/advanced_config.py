"""
Advanced configuration example.

Shows how to customize detector behavior using configuration classes.
"""

import numpy as np
import pandas as pd

from agomax import (
    AgoMaxDetector,
    DetectorConfig,
    EnsembleConfig,
    PreprocessorConfig,
    ThresholdConfig,
)


def generate_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "altitude": rng.normal(100, 2, n),
        "velocity": rng.normal(5, 0.5, n),
        "roll": rng.normal(0, 1, n),
        "battery": rng.normal(15.5, 0.2, n),
    })


def main():
    print("Advanced Configuration Example\n")
    
    # Custom configuration
    config = DetectorConfig(
        # Voting and event detection
        vote_threshold=0.6,  # Require 60% of models to agree
        confirmation_steps=5,  # Require 5 consecutive anomalies for event
        cooldown_steps=15,  # Wait 15 samples after event
        
        # Preprocessing
        preprocessor=PreprocessorConfig(
            handle_missing="median",
            handle_inf=True,
        ),
        
        # Ensemble models
        ensemble=EnsembleConfig(
            kmeans_n_clusters=3,
            lof_n_neighbors=30,
            ocsvm_nu=0.02,
            random_state=42,
        ),
        
        # Adaptive thresholds
        threshold=ThresholdConfig(
            window_size=100,  # Larger window for more stable thresholds
            std_multiplier=3.5,  # More conservative threshold
            min_samples=20,
        ),
        
        # Disable auto-tuning to use manual config
        auto_tune=False,
    )
    
    # Create detector with custom config
    detector = AgoMaxDetector(config)
    
    # Train
    print("Training with custom configuration...")
    train_df = generate_data(n=1000, seed=42)
    detector.fit(train_df)
    print("✓ Training complete\n")
    
    # Predict
    test_df = generate_data(n=100, seed=999)
    result = detector.predict(test_df)
    
    print("Results:")
    print(f"  Anomalies: {result.labels.sum()}")
    print(f"  Mean score: {result.scores.mean():.3f}")
    print(f"  Max score: {result.scores.max():.3f}")
    
    print("\n✓ Example complete!")


if __name__ == "__main__":
    main()
