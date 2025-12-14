# 🚁 Agomax# Agomax: Unsupervised Drone Anomaly Detection



**Production-grade unsupervised anomaly detection for drone telemetry**Agomax is a production-oriented Python package for detecting anomalies in drone telemetry using classical unsupervised learning. It trains once on NORMAL flight data, persists all learned artifacts, and then loads to detect anomalies on new or streaming data with an ensemble of models and confidence scoring. Explainability is optional and additive, exposing per-model contributions for each prediction.



[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)## Why Agomax

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Drone operations produce high-dimensional telemetry with sparse labels and evolving behavior. Supervised approaches often lack sufficient labeled anomalies and may overfit. Agomax uses an unsupervised ensemble to model normal behavior, flagging deviations robustly without requiring labeled anomalies.

Agomax is a professional Python library for detecting anomalies in drone flight telemetry using unsupervised machine learning. It learns from normal flight patterns and identifies deviations in real-time or batch processing—no labeled anomalies required.

## Architecture

---

```

## 📋 Table of Contentsload.py        → raw file → CSV (no ML logic)

preprocess.py  → numeric coercion + StandardScaler (fit on train, transform on test)

- [Why Agomax?](#-why-agomax)tuner.py       → hyperparameter tuning on NORMAL data only

- [Key Features](#-key-features)models.py      → Ensemble (KMeans, DBSCAN, OPTICS, LOF, OCSVM) + scoring + save/load

- [Installation](#-installation)threshold.py   → thresholds (default 99.7 percentile; optional MAD)

- [Quick Start](#-quick-start)pipeline.py    → orchestration (fit, save, load, predict)

- [How It Works](#-how-it-works)__init__.py    → exposes Pipeline only

- [Usage Guide](#-usage-guide)```

- [Architecture](#-architecture)

- [Configuration](#%EF%B8%8F-configuration)Mermaid diagram:

- [API Reference](#-api-reference)

- [Examples](#-examples)```mermaid

- [Best Practices](#-best-practices)flowchart TD

- [Limitations](#-limitations)    A[Raw file] --> B[load.py\nread + normalize]

- [Contributing](#-contributing)    B --> C[CSV]

- [License](#-license)    C --> D[preprocess.py\ncoerce numeric + scale]

    D -->|X_train| E[tuner.py\nfind params]

---    E --> F[models.py\nfit ensemble]

    F --> G[threshold.py\ncompute per-model thresholds]

## 🎯 Why Agomax?    D -->|X_test| H[pipeline.py\npredict]

    F --> H

Drone operations generate high-dimensional telemetry data (altitude, velocity, orientation, battery, etc.) where:    G --> H

```

- **Anomalies are rare and unlabeled** — You have normal flights, but not enough labeled failures

- **Supervised learning fails** — Insufficient anomaly examples to train classifiers## Modules

- **Manual thresholds break** — Static rules can't adapt to flight conditions

- **Real-time detection matters** — You need to catch issues before they escalate- `load.py`: Ingest arbitrary files and normalize to CSV. No feature or ML logic.

- `preprocess.py`: Convert columns to numeric, drop non-convertible/NaN-only columns, handle NaN/Inf, fit a StandardScaler on train and reuse on inference with feature consistency checks.

Agomax solves this by:- `tuner.py`: Tune hyperparameters for each model using NORMAL data only. Keeps anomaly rate bounded using percentile-based flags on model scores.

- `models.py`: Implements the Ensemble of KMeans, DBSCAN, OPTICS, LOF (novelty), and One-Class SVM. Provides `fit(X)`, `score(X)` returning per-model anomaly scores, and `save/load` for persistence.

1. **Learning from normal data only** — Train on successful flights- `threshold.py`: Robust thresholding utilities. Default is 99.7 percentile of training scores; optional MAD-based threshold.

2. **Using an ensemble** — Combines 5 complementary unsupervised models- `pipeline.py`: Orchestrates training and inference: preprocess → tune → fit ensemble → learn thresholds → save; then load → transform → score → threshold → vote → anomaly + confidence.

3. **Adapting to context** — Thresholds adjust to flight patterns- `__init__.py`: Exposes `Pipeline` for package consumers.

4. **Providing explanations** — Shows which models flagged the anomaly

## Training flow (normal-only)

---

1. Load NORMAL flight data to a DataFrame.

## ✨ Key Features2. `Preprocessor.fit(df)` → numeric coercion, scaling, feature list saved.

3. `HyperparameterTuner.tune_all(X_train)` → per-model params, deterministic.

| Feature | Description |4. `Ensemble.fit(X_train)` → train all models.

|---------|-------------|5. `threshold.compute_threshold(train_scores)` per model, default 99.7 percentile.

| 🎓 **Unsupervised Learning** | No anomaly labels required—learns normal behavior |6. Persist `preprocessor`, `ensemble` models, and `thresholds` to disk.

| 🧠 **Ensemble Detection** | Combines KMeans, LOF, One-Class SVM, DBSCAN, OPTICS |

| 📊 **Explainable Results** | Per-sample explanations with model contributions |## Inference flow

| ⚡ **Real-Time Ready** | Adaptive thresholds for streaming data |

| 💾 **Persistent Models** | Save/load trained detectors for deployment |1. Load the trained pipeline (`Pipeline.load()`).

| 🔧 **Configurable** | Tune sensitivity, voting, event detection |2. Transform new data with the same features (`Preprocessor.transform(df)`).

| 🐍 **Scikit-learn Style** | Simple `fit()` / `predict()` interface |3. Score with the ensemble (`Ensemble.score(X)`), producing per-model scores.

| ✅ **Production Ready** | Robust error handling, input validation, type hints |4. Threshold per model → binary flags.

5. Vote (mean of flags) → anomaly score in [0,1].

---6. Threshold vote at 0.4 → anomaly 0/1.

7. Confidence (lightweight normalization of model scores) → [0,1].

## 📦 Installation8. Optional explainability: per-row details of model scores, flags, and top contributors.



### From Source## Explainability



```bashExplainability is additive and optional. When calling `Pipeline.predict(df, explain=True)`, the function returns a fourth item: a list of per-row dictionaries including:

git clone https://github.com/shaguntembhurne/Agomax.git

cd Agomax- `anomaly`: 0/1

pip install -e .- `confidence`: float in [0,1]

```- `model_scores`: per-model anomaly scores

- `model_flags`: per-model threshold exceed flags (0/1)

### Requirements- `top_contributors`: models contributing to the anomaly (flag==1), sorted by score desc



- Python 3.8+This leverages existing scores and thresholds, adds no heavy compute, and is deterministic.

- NumPy

- Pandas## Example usage

- Scikit-learn

- SciPyTrain and save:

- Joblib

```python

---import pandas as pd

from agomax.pipeline import Pipeline

## 🚀 Quick Start

# Load NORMAL flight data

```pythontrain_df = pd.read_csv("notebooks/train_normal.csv")

import pandas as pd

from agomax import AgoMaxDetectorpipe = Pipeline(model_dir="models/")

pipe.fit(train_df)

# 1. Load normal flight data (no anomalies!)```

train_df = pd.read_csv("normal_flights.csv")

Load and predict:

# 2. Create and train detector

detector = AgoMaxDetector()```python

detector.fit(train_df)import pandas as pd

from agomax.pipeline import Pipeline

# 3. Save for deployment

detector.save("models/drone_detector")# New/stream data

test_df = pd.read_csv("notebooks/test.csv")

# 4. Load and detect anomalies

detector = AgoMaxDetector.load("models/drone_detector")pipe = Pipeline(model_dir="models/")

test_df = pd.read_csv("new_flight.csv")pipe.load()



# 5. Get results# Backward-compatible prediction

result = detector.predict(test_df)anomaly_score, anomaly, confidence = pipe.predict(test_df)



print(f"Anomalies detected: {result.labels.sum()}")# With explainability

print(f"Anomaly events: {result.events.sum()}")anomaly_score, anomaly, confidence, details = pipe.predict(test_df, explain=True)

print(f"Mean anomaly score: {result.scores.mean():.3f}")# details[0] example:

```# {

#   "anomaly": 1,

**That's it!** No feature engineering, no manual thresholds, no labels needed.#   "confidence": 0.82,

#   "model_scores": {"kmeans": 1.23, "lof": 2.1, "ocsvm": 3.8, ...},

---#   "model_flags": {"kmeans": 1, "lof": 1, "ocsvm": 0, ...},

#   "top_contributors": ["ocsvm", "lof"]

## 🔬 How It Works# }

```

### The Problem

## Debugging guide

Given telemetry from normal drone flights, detect when new telemetry exhibits anomalous patterns that could indicate:

- Hardware failures (sensor drift, motor issues)Common failure modes and remedies:

- Software bugs (navigation errors, control instability)

- Environmental hazards (wind gusts, GPS loss)- Import errors: Ensure relative imports (`from .module import ...`) within the package; `__init__.py` must expose `Pipeline` only.

- Operational anomalies (unexpected maneuvers)- Empty/invalid data: `Preprocessor.fit()` will error on empty DataFrame or no numeric columns after coercion.

- Feature mismatch: `Preprocessor.transform()` enforces the same features; missing columns raise clear errors.

### The Solution- NaN/Inf handling: Inputs are cleaned by replacing Inf with NaN and dropping NaN rows.

- LOF novelty mode: LOF must be `novelty=True` for inference; confirmed in `models.py`.

Agomax uses an **ensemble of unsupervised models** to characterize normal behavior:- Persistence paths: Artifacts saved under `model_dir` using `joblib` with explicit filenames; ensure write permissions.

- Thresholding: `threshold.compute_threshold()` validates input shape and finiteness; use 99.7 percentile or MAD.

```- Determinism: `random_state=42` is set for models where applicable (e.g., KMeans) and the tuner.

┌─────────────────────────────────────────────────────────────┐

│                    TRAINING (Normal Data Only)               │## Limitations & future work

└─────────────────────────────────────────────────────────────┘

- Temporal dynamics: Current system treats rows independently; temporal/sequence models (e.g., LSTM/Transformers) could improve detection.

Input Data          Preprocessing        Model Training- Feature engineering: Domain-specific features may enhance separability while retaining unsupervised training.

────────────        ──────────────       ───────────────- Real-world calibration: Thresholds and vote cutoffs may require calibration to balance false positives and operational risk.

┌──────────┐       ┌──────────┐        ┌─────────────┐- Data drift: Retraining cadence and drift detection mechanisms can be added.

│ Telemetry│──────▶│  Numeric │───────▶│   KMeans    │

│   CSV    │       │ Coercion │        │    LOF      │## Why unsupervised + ensemble

│  (normal)│       │ Scaling  │        │   OCSVM     │

└──────────┘       └──────────┘        │   DBSCAN    │- Label scarcity: True anomalies are rare; unsupervised methods learn normal behavior without labels.

                                        │   OPTICS    │- Robustness: Ensembles combine complementary signals (density, clustering, distance, margin) for stability.

                                        └─────────────┘- Interpretability: Each model’s score/flag provides a distinct perspective on deviation.

                                              │

                                              ▼## Production considerations

                                        ┌─────────────┐

                                        │ Save Models │- False positives: Monitor exceed rates; adjust per-model thresholds and vote cutoff (default 0.4) conservatively.

                                        └─────────────┘- Safety first: Treat anomaly flags as risk indicators; integrate with human-in-the-loop workflows where applicable.

- Logging/monitoring: Capture prediction rates and distribution shifts to trigger retraining.

- Reproducibility: Pin environment, persist artifacts, and use deterministic seeds for repeatable outcomes.

┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE (New Data)                      │
└─────────────────────────────────────────────────────────────┘

Input Data          Preprocessing        Scoring
────────────        ──────────────       ────────
┌──────────┐       ┌──────────┐        ┌─────────────┐
│   New    │──────▶│Same Trans│───────▶│ Each model  │
│Telemetry │       │formations│        │ computes    │
└──────────┘       └──────────┘        │ anomaly     │
                                        │ score       │
                                        └─────────────┘
                                              │
                                              ▼
                   Thresholding        Voting & Events
                   ────────────        ────────────────
                   ┌──────────┐        ┌─────────────┐
                   │ Adaptive │───────▶│ Ensemble    │
                   │Thresholds│        │ voting      │
                   │(per model)│        │ → Labels   │
                   └──────────┘        │ → Events    │
                                        └─────────────┘
                                              │
                                              ▼
                                        ┌─────────────┐
                                        │   Result    │
                                        │ scores      │
                                        │ labels      │
                                        │ events      │
                                        │ (optional   │
                                        │ explanations)│
                                        └─────────────┘
```

### Training Pipeline

1. **Preprocessing**: Convert all columns to numeric, handle missing/infinite values, apply standard scaling
2. **Auto-tuning** (optional): Find hyperparameters that minimize false positives on training data
3. **Ensemble Fitting**: Train 5 models on normal data
4. **Threshold Initialization**: Set adaptive thresholds from training score distributions

### Inference Pipeline

1. **Preprocessing**: Apply same transformations as training
2. **Scoring**: Each model computes anomaly score (higher = more anomalous)
3. **Thresholding**: Adaptive per-model thresholds (update only on normal scores)
4. **Voting**: Aggregate binary decisions (vote_ratio = mean of model flags)
5. **Event Detection**: Require consecutive anomalies before confirming event

---

## 📖 Usage Guide

### Basic Training

```python
from agomax import AgoMaxDetector
import pandas as pd

# Load normal flight data
train_df = pd.read_csv("normal_flights.csv")

# Train detector (auto-tunes by default)
detector = AgoMaxDetector()
detector.fit(train_df)

# Save for later
detector.save("models/")
```

**Important**: Training data should contain **ONLY normal behavior**. Remove any known anomalies or failures.

### Detection

```python
from agomax import AgoMaxDetector

# Load trained detector
detector = AgoMaxDetector.load("models/")

# Predict on new data
test_df = pd.read_csv("new_flight.csv")
result = detector.predict(test_df)

# Access results
print(result.scores)   # Continuous anomaly scores (0-1)
print(result.labels)   # Binary labels (0=normal, 1=anomaly)
print(result.events)   # Confirmed events after temporal filtering
```

### Explanations

```python
# Get detailed explanations
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
from agomax import AgoMaxDetector, DetectorConfig, EnsembleConfig, ThresholdConfig

# Create custom configuration
config = DetectorConfig(
    # Voting: require 60% of models to agree
    vote_threshold=0.6,
    
    # Events: need 5 consecutive anomalies
    confirmation_steps=5,
    cooldown_steps=15,
    
    # Ensemble parameters
    ensemble=EnsembleConfig(
        kmeans_n_clusters=3,
        lof_n_neighbors=30,
        ocsvm_nu=0.02,
    ),
    
    # Adaptive thresholds
    threshold=ThresholdConfig(
        window_size=100,        # Larger window = more stable
        std_multiplier=3.5,     # Higher = more conservative
    ),
    
    # Disable auto-tuning to use manual config
    auto_tune=False,
)

detector = AgoMaxDetector(config)
detector.fit(train_df)
```

### Streaming/Real-Time Use

```python
from agomax import AgoMaxDetector
import pandas as pd

detector = AgoMaxDetector.load("models/")

# Process samples one at a time
while True:
    # Get new telemetry sample
    sample = get_telemetry_sample()  # Your function
    sample_df = pd.DataFrame([sample])
    
    # Detect
    result = detector.predict(sample_df)
    
    if result.labels[0]:
        print(f"⚠️ ANOMALY: score={result.scores[0]:.3f}")
        
    if result.events[0]:
        print(f"🚨 CONFIRMED EVENT - Take action!")
```

---

## 🏗️ Architecture

### Package Structure

```
agomax/
├── __init__.py              # Public API exports
├── detector.py              # AgoMaxDetector (main user-facing class)
├── config.py                # Configuration dataclasses
├── exceptions.py            # Custom exceptions
├── utils.py                 # Data loading utilities
├── compat.py                # Backward compatibility (deprecated)
└── core/                    # Internal implementation
    ├── preprocessing.py     # Data preprocessing
    ├── ensemble.py          # Model ensemble
    ├── threshold.py         # Adaptive thresholds
    └── tuning.py            # Hyperparameter tuning
```

### Ensemble Models

| Model | Type | Purpose | Strength |
|-------|------|---------|----------|
| **KMeans** | Clustering | Distance to normal clusters | Fast, interpretable |
| **LOF** | Density | Local outlier factor | Detects local deviations |
| **One-Class SVM** | Margin | Decision boundary around normal data | Robust to outliers |
| **DBSCAN** | Clustering | Structural outlier context | Finds noise points |
| **OPTICS** | Clustering | Reachability-based context | Handles varying densities |

The first three are **scoring models** that vote on anomalies. The last two provide **structural context** for explanations.

### Design Decisions

**Why Unsupervised?**
- Anomalies are rare and often unlabeled
- New failure modes emerge over time
- Supervised learning requires balanced labeled data

**Why Ensemble?**
- Single models have blind spots
- Different models capture different anomaly types
- Voting reduces false positives

**Why Adaptive Thresholds?**
- Flight conditions vary (altitude, speed, weather)
- Static thresholds produce false alarms
- Streaming data requires online adaptation

**Why Event Detection?**
- Single-point anomalies can be noise
- Persistent anomalies indicate real issues
- Cooldown prevents alert spam

---

## ⚙️ Configuration

### DetectorConfig

Main configuration for the detector.

```python
DetectorConfig(
    vote_threshold=0.5,        # Fraction of models that must agree
    confirmation_steps=3,      # Consecutive anomalies for event
    cooldown_steps=10,         # Samples to wait after event
    model_dir="models",        # Where to save models
    auto_tune=True,            # Auto-tune hyperparameters
)
```

### EnsembleConfig

Model ensemble parameters.

```python
EnsembleConfig(
    # KMeans
    kmeans_n_clusters=2,
    kmeans_max_iter=300,
    
    # LOF
    lof_n_neighbors=20,
    lof_metric="euclidean",
    
    # One-Class SVM
    ocsvm_nu=0.01,
    ocsvm_gamma="scale",
    
    # DBSCAN
    dbscan_eps=1.2,
    dbscan_min_samples=20,
    
    # OPTICS
    optics_min_samples=20,
    optics_xi=0.05,
    
    random_state=42,
)
```

### ThresholdConfig

Adaptive threshold settings.

```python
ThresholdConfig(
    window_size=50,       # Rolling window size
    std_multiplier=3.0,   # Std deviation multiplier
    min_samples=10,       # Minimum samples before active
)
```

---

## 📚 API Reference

### AgoMaxDetector

Main detector class.

#### Methods

**`fit(data, auto_tune=True)`**
- Train detector on normal data
- `data`: DataFrame or ndarray
- `auto_tune`: Whether to tune hyperparameters
- Returns: `self`

**`predict(data, explain=False)`**
- Detect anomalies in new data
- `data`: DataFrame or ndarray
- `explain`: Include detailed explanations
- Returns: `AnomalyResult`

**`save(directory)`**
- Save trained detector to disk
- `directory`: Path to save location

**`load(directory)`** (classmethod)
- Load trained detector from disk
- `directory`: Path to model files
- Returns: `AgoMaxDetector`

**`reset_state()`**
- Reset adaptive thresholds and counters
- Use when starting a new flight/stream

### AnomalyResult

Result container from `predict()`.

#### Attributes

- **`scores`** (ndarray): Continuous anomaly scores (0-1)
- **`labels`** (ndarray): Binary anomaly labels (0/1)
- **`events`** (ndarray): Confirmed anomaly events (0/1)
- **`details`** (list, optional): Per-sample explanations

---

## 💡 Examples

### Example 1: Train and Evaluate

```python
from agomax import AgoMaxDetector
import pandas as pd
import numpy as np

# Generate synthetic normal data
rng = np.random.default_rng(42)
train_df = pd.DataFrame({
    'altitude': rng.normal(100, 2, 1000),
    'velocity': rng.normal(5, 0.5, 1000),
    'roll': rng.normal(0, 1, 1000),
    'battery': rng.normal(15.5, 0.1, 1000),
})

# Train
detector = AgoMaxDetector()
detector.fit(train_df)
detector.save("models/")

# Generate test data with anomalies
test_df = train_df.copy()[:200]
test_df.loc[100:110, 'altitude'] -= 20  # Inject anomaly

# Detect
result = detector.predict(test_df)

print(f"Total samples: {len(result.scores)}")
print(f"Anomalies found: {result.labels.sum()}")
print(f"Events detected: {result.events.sum()}")
```

### Example 2: Real-Time Monitoring

See `examples/streaming.py` for a complete real-time example.

### Example 3: Advanced Configuration

See `examples/advanced_config.py` for customization examples.

---

## ✅ Best Practices

### Training Data

- ✅ **Use only normal flights** — Remove any known failures
- ✅ **Include diverse conditions** — Different altitudes, speeds, weather
- ✅ **Sufficient samples** — At least 500+ samples recommended
- ✅ **Representative data** — Covers expected operational range

### Configuration Tuning

- **High false positives?**
  - Increase `vote_threshold` (0.6-0.7)
  - Increase `std_multiplier` (3.5-4.0)
  - Increase `confirmation_steps` (5-10)

- **Missing anomalies?**
  - Decrease `vote_threshold` (0.3-0.4)
  - Decrease `std_multiplier` (2.5-3.0)
  - Ensure training data is diverse

- **Noisy alerts?**
  - Increase `confirmation_steps`
  - Increase `cooldown_steps`

### Deployment

- ✅ **Validate on historical data** — Test before production
- ✅ **Monitor false positive rate** — Track and adjust thresholds
- ✅ **Retrain periodically** — Capture evolving normal patterns
- ✅ **Use explanations** — Understand what triggered alerts
- ✅ **Combine with domain rules** — Use as decision support, not autopilot

---

## ⚠️ Limitations

### What Agomax Does Well

- Detecting deviations from normal flight patterns
- Handling high-dimensional telemetry
- Operating without labeled anomalies
- Adapting to varying flight conditions

### What Agomax Doesn't Do

- **Classify anomaly types** — Only detects "anomalous vs normal"
- **Predict failures** — Reactive detection, not predictive
- **Handle extreme drift** — Requires retraining for new flight modes
- **Work with tiny datasets** — Needs sufficient normal examples (~500+)

### Assumptions

- Training data is predominantly normal
- Features are numeric or convertible to numeric
- Anomalies manifest in telemetry patterns
- Some false positives are acceptable

---

## 🎯 Use Cases

### Ideal For

- Flight anomaly detection
- Health monitoring systems
- Quality control in manufacturing
- Network intrusion detection (IoT)
- Equipment predictive maintenance

### Not Ideal For

- Anomaly classification (what type of failure)
- Time series forecasting
- Supervised learning tasks with labels
- Tiny datasets (<100 samples)

---

## 🗺️ Roadmap

### Current Version (0.1.0)

- ✅ Ensemble-based detection
- ✅ Adaptive thresholds
- ✅ Model persistence
- ✅ Explanations
- ✅ Event detection

### Future Enhancements

- [ ] Temporal models (LSTM, Transformers)
- [ ] Online learning / incremental updates
- [ ] Anomaly type classification
- [ ] AutoML for hyperparameter selection
- [ ] GPU acceleration
- [ ] Dashboard/visualization tools
- [ ] Multi-flight pattern support

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Shagun Tembhurne**

- GitHub: [@shaguntembhurne](https://github.com/shaguntembhurne)
- Repository: [github.com/shaguntembhurne/Agomax](https://github.com/shaguntembhurne/Agomax)

---

## 🙏 Acknowledgments

Built with:
- [scikit-learn](https://scikit-learn.org/) — Machine learning models
- [pandas](https://pandas.pydata.org/) — Data manipulation
- [NumPy](https://numpy.org/) — Numerical computing

---

## 📊 Citation

If you use Agomax in your research or project, please cite:

```bibtex
@software{agomax2024,
  title={Agomax: Production-Grade Anomaly Detection for Drone Telemetry},
  author={Tembhurne, Shagun},
  year={2024},
  url={https://github.com/shaguntembhurne/Agomax}
}
```

---

<div align="center">
  
**Made with ❤️ for safer drone operations**

[⬆ Back to Top](#-agomax)

</div>
