# Agomax Examples

This directory contains examples demonstrating how to use Agomax.

## Examples

### 1. basic_usage.py
Complete workflow demonstrating:
- Training on synthetic normal data
- Saving and loading models
- Running detection
- Getting explanations

Run with:
```bash
python examples/basic_usage.py
```

### 2. advanced_config.py
Shows how to customize detector behavior using configuration classes:
- Custom voting thresholds
- Ensemble parameters
- Adaptive threshold settings
- Disabling auto-tuning

Run with:
```bash
python examples/advanced_config.py
```

### 3. streaming.py
Demonstrates real-time anomaly detection:
- Training on historical data
- Processing samples one at a time
- Simulating streaming telemetry
- Real-time alerting

Run with:
```bash
python examples/streaming.py
```

## Running All Examples

```bash
cd /path/to/Agomax
python examples/basic_usage.py
python examples/advanced_config.py
python examples/streaming.py
```

## Requirements

Make sure Agomax is installed:
```bash
pip install -e .
```
