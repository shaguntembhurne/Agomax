# Agomax Tests

Integration tests for the Agomax package.

## Running Tests

```bash
# From the project root
python tests/test_integration.py
```

## Test Coverage

The integration tests cover:

✅ **Basic Workflow**
- Training on normal data
- Prediction on new data  
- Anomaly detection
- Event detection

✅ **Model Persistence**
- Saving trained models
- Loading saved models
- Prediction consistency

✅ **Explanations**
- Explanation generation
- Detail structure
- Contributor identification

✅ **Configuration**
- Custom configuration
- Parameter validation

✅ **Input Handling**
- NumPy array input
- DataFrame input
- Feature consistency

✅ **State Management**
- State reset functionality
- Threshold adaptation

## Expected Output

```
============================================================
Agomax Integration Tests
============================================================

[TEST] Basic workflow
  ✓ Detected X anomalies
  ✓ Detected X events

[TEST] Save and load
  ✓ Model saved and loaded successfully
  ✓ Prediction difference: X.XXXX

[TEST] Explanations
  ✓ Generated explanations for X samples
  ✓ Anomaly contributors: [...]

[TEST] Custom configuration
  ✓ Custom configuration applied

[TEST] Numpy input
  ✓ Numpy input works

[TEST] State reset
  ✓ State reset works (score diff: X.XXXX)

============================================================
✓ All tests passed!
============================================================
```

## Adding Tests

To add new tests:

1. Add a test function to `test_integration.py`
2. Follow the naming convention: `test_<feature>()`
3. Include print statements for test progress
4. Use assertions to validate behavior
5. Update this README with the new test
