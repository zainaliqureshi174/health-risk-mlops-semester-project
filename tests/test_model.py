"""
Model Validation Tests for CI/CD Pipeline
"""
import pickle
import json
import os
import sys
import numpy as np

def test_model_files_exist():
    """Test that model files exist"""
    print("Testing model file existence...")
    
    model_files = [
        'models/logistic_regression.pkl',
        'models/random_forest.pkl',
        'models/scaler.pkl'
    ]
    
    for file in model_files:
        assert os.path.exists(file), f"Missing model file: {file}"
        file_size = os.path.getsize(file) / (1024 * 1024)  # MB
        print(f"  ✓ {file} ({file_size:.2f} MB)")
    
    print("✅ All model files exist\n")
    return True

def test_model_loadable():
    """Test that models can be loaded"""
    print("Testing model loading...")
    
    # Load logistic regression
    with open('models/logistic_regression.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    print("  ✓ Logistic regression loaded")
    
    # Load random forest
    with open('models/random_forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    print("  ✓ Random forest loaded")
    
    # Load scaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("  ✓ Scaler loaded")
    
    print("✅ All models loadable\n")
    return lr_model, rf_model, scaler

def test_model_prediction():
    """Test that models can make predictions"""
    print("Testing model predictions...")
    
    lr_model, rf_model, scaler = test_model_loadable()
    
    # Create sample input
    sample_input = np.array([[
        50.0,  # pm25
        80.0,  # pm10
        150.0, # aqi
        25.0,  # temperature
        60.0,  # humidity
        2.0,   # precipitation
        75.0,  # heart_rate
        7.0,   # sleep_hours
        98.0,  # spo2
        5      # stress_level
    ]])
    
    # Scale input
    sample_scaled = scaler.transform(sample_input)
    
    # Test logistic regression prediction
    lr_pred = lr_model.predict(sample_scaled)
    lr_proba = lr_model.predict_proba(sample_scaled)
    assert lr_pred[0] in [0, 1], "Invalid prediction"
    assert 0 <= lr_proba[0][1] <= 1, "Invalid probability"
    print(f"  ✓ LR prediction: {lr_pred[0]}, probability: {lr_proba[0][1]:.3f}")
    
    # Test random forest prediction
    rf_pred = rf_model.predict(sample_scaled)
    rf_proba = rf_model.predict_proba(sample_scaled)
    assert rf_pred[0] in [0, 1], "Invalid prediction"
    assert 0 <= rf_proba[0][1] <= 1, "Invalid probability"
    print(f"  ✓ RF prediction: {rf_pred[0]}, probability: {rf_proba[0][1]:.3f}")
    
    print("✅ Model prediction tests passed\n")
    return True

def test_model_performance():
    """Test that model performance meets minimum threshold"""
    print("Testing model performance...")
    
    # Load results
    results_file = 'models/baseline_results.json'
    assert os.path.exists(results_file), "Results file missing"
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Test logistic regression performance
    lr_acc = results['logistic_regression']['accuracy']
    assert lr_acc > 0.80, f"LR accuracy too low: {lr_acc:.2%}"
    print(f"  ✓ Logistic Regression accuracy: {lr_acc:.2%} (>80%)")
    
    # Test random forest performance
    rf_acc = results['random_forest']['accuracy']
    assert rf_acc > 0.80, f"RF accuracy too low: {rf_acc:.2%}"
    print(f"  ✓ Random Forest accuracy: {rf_acc:.2%} (>80%)")
    
    # Test that model isn't overfitting (ROC-AUC reasonable)
    lr_auc = results['logistic_regression']['roc_auc']
    assert lr_auc > 0.5, f"LR ROC-AUC at chance level: {lr_auc:.3f}"
    print(f"  ✓ Logistic Regression ROC-AUC: {lr_auc:.3f}")
    
    print("✅ Model performance tests passed\n")
    return True

def test_federated_model():
    """Test federated model if it exists"""
    print("Testing federated model...")
    
    fed_model_path = 'models/federated_model.pkl'
    if not os.path.exists(fed_model_path):
        print("  ⚠️  Federated model not found (optional)")
        return True
    
    with open(fed_model_path, 'rb') as f:
        fed_model = pickle.load(f)
    print("  ✓ Federated model loaded")
    
    with open('models/federated_scaler.pkl', 'rb') as f:
        fed_scaler = pickle.load(f)
    print("  ✓ Federated scaler loaded")
    
    # Test prediction
    sample_input = np.array([[50, 80, 150, 25, 60, 2, 75, 7, 98, 5]])
    sample_scaled = fed_scaler.transform(sample_input)
    pred = fed_model.predict(sample_scaled)
    assert pred[0] in [0, 1], "Invalid federated prediction"
    print(f"  ✓ Federated prediction: {pred[0]}")
    
    print("✅ Federated model tests passed\n")
    return True

if __name__ == "__main__":
    try:
        test_model_files_exist()
        test_model_loadable()
        test_model_prediction()
        test_model_performance()
        test_federated_model()
        print("="*50)
        print("ALL MODEL TESTS PASSED ✅")
        print("="*50)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
