"""
Integration tests for Agomax.

Tests the complete workflow: train, save, load, predict.
"""

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from agomax import AgoMaxDetector, DetectorConfig


def generate_normal_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic normal flight telemetry."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "altitude": rng.normal(100, 2, n),
        "velocity_z": rng.normal(0, 0.2, n),
        "roll": rng.normal(0, 1, n),
        "pitch": rng.normal(0, 1, n),
        "yaw": rng.normal(0, 1, n),
        "battery": rng.normal(15.5, 0.1, n),
    })


def generate_anomalous_data(n: int = 300, seed: int = 123) -> pd.DataFrame:
    """Generate test data with anomalies."""
    rng = np.random.default_rng(seed)
    
    df = pd.DataFrame({
        "altitude": rng.normal(100, 2, n),
        "velocity_z": rng.normal(0, 0.2, n),
        "roll": rng.normal(0, 1, n),
        "pitch": rng.normal(0, 1, n),
        "yaw": rng.normal(0, 1, n),
        "battery": rng.normal(15.5, 0.1, n),
    })
    
    # Inject anomaly burst
    df.loc[180:200, "altitude"] -= 15
    df.loc[180:200, "roll"] += 10
    df.loc[180:200, "battery"] -= 1.5
    
    return df


def test_basic_workflow():
    """Test basic train -> predict workflow."""
    print("\n[TEST] Basic workflow")
    
    # Generate data
    train_df = generate_normal_data()
    test_df = generate_anomalous_data()
    
    # Train
    detector = AgoMaxDetector()
    detector.fit(train_df)
    
    assert detector.is_fitted, "Detector should be fitted"
    
    # Predict
    result = detector.predict(test_df)
    
    assert len(result.scores) == len(test_df), "Wrong number of predictions"
    assert result.labels.sum() > 0, "Should detect some anomalies"
    assert result.scores.min() >= 0, "Scores should be non-negative"
    assert result.scores.max() <= 1, "Scores should be <= 1"
    
    print(f"  ✓ Detected {result.labels.sum()} anomalies")
    print(f"  ✓ Detected {result.events.sum()} events")


def test_save_load():
    """Test model persistence."""
    print("\n[TEST] Save and load")
    
    train_df = generate_normal_data()
    test_df = generate_anomalous_data()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "models"
        
        # Train and save
        detector1 = AgoMaxDetector()
        detector1.fit(train_df)
        detector1.save(model_dir)
        
        # Load
        detector2 = AgoMaxDetector.load(model_dir)
        
        assert detector2.is_fitted, "Loaded detector should be fitted"
        
        # Compare predictions
        result1 = detector1.predict(test_df)
        
        # Reset state for fair comparison
        detector2.reset_state()
        result2 = detector2.predict(test_df)
        
        # Scores should be very similar (slight differences due to threshold updates)
        score_diff = np.abs(result1.scores - result2.scores).mean()
        assert score_diff < 0.1, f"Predictions differ too much: {score_diff}"
        
        print(f"  ✓ Model saved and loaded successfully")
        print(f"  ✓ Prediction difference: {score_diff:.4f}")


def test_explanations():
    """Test explanation generation."""
    print("\n[TEST] Explanations")
    
    train_df = generate_normal_data()
    test_df = generate_anomalous_data()
    
    detector = AgoMaxDetector()
    detector.fit(train_df)
    
    # Predict with explanations
    result = detector.predict(test_df, explain=True)
    
    assert result.details is not None, "Details should be present"
    assert len(result.details) == len(test_df), "Should have details for all samples"
    
    # Check detail structure
    detail = result.details[0]
    assert "anomaly_score" in detail
    assert "is_anomaly" in detail
    assert "model_scores" in detail
    assert "model_flags" in detail
    assert "top_contributors" in detail
    
    print(f"  ✓ Generated explanations for {len(result.details)} samples")
    
    # Find an anomaly and check contributors
    for detail in result.details:
        if detail["is_anomaly"]:
            print(f"  ✓ Anomaly contributors: {detail['top_contributors']}")
            break


def test_custom_config():
    """Test custom configuration."""
    print("\n[TEST] Custom configuration")
    
    config = DetectorConfig(
        vote_threshold=0.6,
        confirmation_steps=5,
        auto_tune=False,
    )
    
    detector = AgoMaxDetector(config)
    train_df = generate_normal_data()
    detector.fit(train_df)
    
    assert detector.config.vote_threshold == 0.6
    assert detector.config.confirmation_steps == 5
    
    print("  ✓ Custom configuration applied")


def test_numpy_input():
    """Test with numpy arrays instead of DataFrames."""
    print("\n[TEST] Numpy input")
    
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, (500, 6))
    X_test = rng.normal(0, 1, (100, 6))
    
    detector = AgoMaxDetector()
    detector.fit(X_train)
    result = detector.predict(X_test)
    
    assert len(result.scores) == 100
    print("  ✓ Numpy input works")


def test_reset_state():
    """Test state reset functionality."""
    print("\n[TEST] State reset")
    
    train_df = generate_normal_data()
    test_df = generate_anomalous_data()
    
    detector = AgoMaxDetector()
    detector.fit(train_df)
    
    # Predict once
    result1 = detector.predict(test_df)
    
    # Reset and predict again
    detector.reset_state()
    result2 = detector.predict(test_df)
    
    # Results should be very similar after reset (adaptive thresholds may vary slightly)
    score_diff = np.abs(result1.scores - result2.scores).mean()
    assert score_diff < 0.1, f"Results differ too much after reset: {score_diff}"
    
    print(f"  ✓ State reset works (score diff: {score_diff:.4f})")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Agomax Integration Tests")
    print("=" * 60)
    
    test_basic_workflow()
    test_save_load()
    test_explanations()
    test_custom_config()
    test_numpy_input()
    test_reset_state()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
