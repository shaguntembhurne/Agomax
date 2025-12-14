"""
Streaming data example.

Demonstrates real-time anomaly detection on streaming data.
"""

import numpy as np
import pandas as pd
import time

from agomax import AgoMaxDetector


def simulate_telemetry_stream():
    """Simulate streaming telemetry data."""
    rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
    
    # Occasionally inject anomalies
    is_anomaly = rng.random() < 0.05
    
    if is_anomaly:
        altitude = rng.normal(85, 5)  # Abnormal altitude
        roll = rng.normal(15, 5)  # High roll
    else:
        altitude = rng.normal(100, 2)
        roll = rng.normal(0, 1)
    
    return {
        "altitude": altitude,
        "velocity_z": rng.normal(0, 0.2),
        "roll": roll,
        "pitch": rng.normal(0, 1),
        "yaw": rng.normal(0, 1),
        "battery": rng.normal(15.5, 0.1),
    }


def main():
    print("Streaming Anomaly Detection Example\n")
    
    # Generate and train on historical normal data
    print("1. Training on historical data...")
    rng = np.random.default_rng(42)
    train_df = pd.DataFrame({
        "altitude": rng.normal(100, 2, 500),
        "velocity_z": rng.normal(0, 0.2, 500),
        "roll": rng.normal(0, 1, 500),
        "pitch": rng.normal(0, 1, 500),
        "yaw": rng.normal(0, 1, 500),
        "battery": rng.normal(15.5, 0.1, 500),
    })
    
    detector = AgoMaxDetector()
    detector.fit(train_df)
    print("   ✓ Training complete\n")
    
    # Simulate streaming detection
    print("2. Starting streaming detection (10 samples)...\n")
    
    for i in range(10):
        # Get new telemetry sample
        sample = simulate_telemetry_stream()
        sample_df = pd.DataFrame([sample])
        
        # Detect
        result = detector.predict(sample_df, explain=True)
        
        # Display
        status = "🔴 ANOMALY" if result.labels[0] else "🟢 Normal"
        event = " [EVENT]" if result.events[0] else ""
        
        print(f"Sample {i+1:2d}: {status}{event}")
        print(f"  Score: {result.scores[0]:.3f}")
        print(f"  Altitude: {sample['altitude']:.1f}")
        
        if result.details[0]['is_anomaly']:
            print(f"  Contributors: {result.details[0]['top_contributors']}")
        print()
        
        time.sleep(0.5)  # Simulate time delay
    
    print("✓ Streaming example complete!")


if __name__ == "__main__":
    main()
