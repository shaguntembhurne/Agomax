# Agomax

**Production-grade unsupervised anomaly detection for drone telemetry**

Agomax is a Python package that detects anomalies in drone flight data using unsupervised machine learning. Train once on normal flights, then identify deviations in real-time or batch processing—no labeled anomalies required.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why Agomax

Drone operations generate high-dimensional telemetry where anomalies are rare and unlabeled. Supervised approaches fail due to insufficient failure examples. Agomax solves this by:

- **Learning from normal data only** — Train on successful flights
- **Using an ensemble approach** — Combines 5 complementary unsupervised models
- **Adapting to context** — Thresholds adjust to flight patterns  
- **Providing explanations** — Shows which models flagged each anomaly

---

## Installation

```bash
git clone https://github.com/shaguntembhurne/Agomax.git
cd Agomax
pip install -e .
```

**Requirements:** Python 3.8+, NumPy, Pandas, Scikit-learn, SciPy, Joblib

---

## Quick Start

```python
import pandas as pd
from agomax import AgoMaxDetector

# 1. Load normal flight data (no anomalies)
train_df = pd.read_csv("normal_flights.csv")

# 2. Train detector
detector = AgoMaxDetector()
detector.fit(train_df)

# 3. Save for deployment
detector.save("models/drone_detector")

# 4. Load and detect anomalies
detector = AgoMaxDetector.load("models/drone_detector")
test_df = pd.read_csv("new_flight.csv")
result = detector.predict(test_df)

# 5. Access results
print(f"Anomalies detected: {result.labels.sum()}")
print(f"Mean anomaly score: {result.scores.mean():.3f}")
```

---

## How It Works

### Architecture

Agomax uses an ensemble of five unsupervised models to characterize normal behavior:

| Model | Type | Purpose |
|-------|------|---------|
| **KMeans** | Clustering | Distance to normal clusters |
| **LOF** | Density | Local outlier detection |
| **One-Class SVM** | Margin | Decision boundary around normal data |
| **DBSCAN** | Clustering | Structural outlier context |
| **OPTICS** | Clustering | Handles varying densities |

### Training Pipeline

```
Raw Data → Preprocessing → Hyperparameter Tuning → Ensemble Fit → Threshold Learning → Save
```

1. **Preprocessing**: Numeric coercion, handle missing values, standard scaling
2. **Tuning**: Optimize parameters on normal data to minimize false positives
3. **Fitting**: Train all five models on preprocessed data
4. **Thresholding**: Compute per-model thresholds (default: 99.7th percentile)
5. **Persistence**: Save preprocessor, ensemble, and thresholds to disk

### Inference Pipeline

```
New Data → Transform → Score (per-model) → Threshold → Vote → Anomaly + Confidence
```

1. Transform new data using saved preprocessing pipeline
2. Each model computes an anomaly score
3. Compare scores against learned thresholds
4. Aggregate binary decisions via voting (mean of flags)
5. Apply vote threshold (default: 0.4) to classify anomaly
6. Compute confidence score for each prediction

---

## Usage Examples

### Basic Detection

```python
from agomax import AgoMaxDetector
import pandas as pd

# Train
detector = AgoMaxDetector()
detector.fit(train_df)
detector.save("models/")

# Predict
detector = AgoMaxDetector.load("models/")
result = detector.predict(test_df)

print(result.scores)   # Continuous anomaly scores [0-1]
print(result.labels)   # Binary labels: 0=normal, 1=anomaly
print(result.events)   # Confirmed events after temporal filtering
```

### With Explanations

```python
result = detector.predict(test_df, explain=True)

for i, detail in enumerate(result.details):
    if detail['is_anomaly']:
        print(f"\nSample {i}:")
        print(f"  Anomaly Score: {detail['anomaly_score']:.3f}")
        print(f"  Vote Ratio: {detail['vote_ratio']:.3f}")
        print(f"  Top Contributors: {detail['top_contributors']}")
        print(f"  Model Scores: {detail['model_scores']}")
```

### Custom Configuration

```python
from agomax import AgoMaxDetector, DetectorConfig, EnsembleConfig

config = DetectorConfig(
    vote_threshold=0.6,          # 60% of models must agree
    confirmation_steps=5,        # Require 5 consecutive anomalies
    cooldown_steps=15,           # Wait 15 steps after event
    ensemble=EnsembleConfig(
        kmeans_n_clusters=3,
        lof_n_neighbors=30,
        ocsvm_nu=0.02,
    ),
    auto_tune=False,
)

detector = AgoMaxDetector(config)
detector.fit(train_df)
```

### Streaming/Real-Time

```python
detector = AgoMaxDetector.load("models/")

while True:
    sample = get_telemetry_sample()
    sample_df = pd.DataFrame([sample])
    result = detector.predict(sample_df)
    
    if result.labels[0]:
        print(f"ANOMALY: score={result.scores[0]:.3f}")
    if result.events[0]:
        print(f"CONFIRMED EVENT - Take action!")
```

---

## API Reference

### AgoMaxDetector

**Methods**

- `fit(data, auto_tune=True)` — Train on normal data
- `predict(data, explain=False)` — Detect anomalies, returns AnomalyResult
- `save(directory)` — Persist trained detector
- `load(directory)` — Load from disk (classmethod)
- `reset_state()` — Reset adaptive thresholds

### AnomalyResult

**Attributes**

- `scores` (ndarray) — Continuous anomaly scores [0-1]
- `labels` (ndarray) — Binary labels: 0=normal, 1=anomaly
- `events` (ndarray) — Confirmed events after temporal filtering
- `details` (list, optional) — Per-sample explanations

### Configuration Classes

**DetectorConfig**
```python
DetectorConfig(
    vote_threshold=0.5,        # Fraction of models that must agree
    confirmation_steps=3,      # Consecutive anomalies for event
    cooldown_steps=10,         # Wait period after event
    auto_tune=True,            # Auto-tune hyperparameters
)
```

**EnsembleConfig**
```python
EnsembleConfig(
    kmeans_n_clusters=2,
    lof_n_neighbors=20,
    ocsvm_nu=0.01,
    dbscan_eps=1.2,
    optics_min_samples=20,
    random_state=42,
)
```

---

## Best Practices

### Training Data
- Use only normal flights—remove known failures
- Include diverse conditions (altitude, speed, weather)
- Minimum 500+ samples recommended
- Ensure data covers expected operational range

### Tuning for Production
**High false positives:**
- Increase `vote_threshold` (0.6-0.7)
- Increase `confirmation_steps` (5-10)

**Missing anomalies:**
- Decrease `vote_threshold` (0.3-0.4)
- Ensure training data is sufficiently diverse

**Noisy alerts:**
- Increase `confirmation_steps`
- Increase `cooldown_steps`

### Deployment
- Validate on historical data before production
- Monitor false positive rate and adjust thresholds
- Retrain periodically to capture evolving patterns
- Use explanations to understand alert triggers
- Combine with domain rules—treat as decision support

---

## Limitations

### What Agomax Does
- Detects deviations from normal flight patterns
- Handles high-dimensional telemetry without labels
- Adapts to varying flight conditions

### What Agomax Doesn't Do
- Classify anomaly types (only flags normal vs. anomalous)
- Predict failures (reactive, not predictive)
- Handle extreme distribution drift without retraining
- Work with tiny datasets (needs ~500+ normal samples)

### Assumptions
- Training data is predominantly normal
- Features are numeric or convertible to numeric
- Anomalies manifest in telemetry patterns
- Some false positives are acceptable

---

## Package Structure

```
agomax/
├── __init__.py              # Public API exports
├── detector.py              # AgoMaxDetector (main class)
├── config.py                # Configuration dataclasses
├── exceptions.py            # Custom exceptions
├── utils.py                 # Data loading utilities
└── core/
    ├── preprocessing.py     # Data preprocessing
    ├── ensemble.py          # Model ensemble
    ├── threshold.py         # Adaptive thresholds
    └── tuning.py            # Hyperparameter tuning
```

---

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/name`)
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Citation

```bibtex
@software{agomax2024,
  title={Agomax: Production-Grade Anomaly Detection for Drone Telemetry},
  author={Tembhurne, Shagun},
  year={2024},
  url={https://github.com/shaguntembhurne/Agomax}
}
```

---

## Contact

**Shagun Tembhurne**  
GitHub: [@shaguntembhurne](https://github.com/shaguntembhurne)  
Repository: [github.com/shaguntembhurne/Agomax](https://github.com/shaguntembhurne/Agomax)

Built with scikit-learn, pandas, and NumPy.
