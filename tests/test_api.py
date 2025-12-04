"""
API Integration Tests for CI/CD Pipeline
"""
import requests
import sys
import time

API_URL = "http://localhost:8000"

def wait_for_api(max_attempts=30):
    """Wait for API to be ready"""
    print("Waiting for API to be ready...")
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                print(f"  ✓ API ready after {attempt + 1} attempts")
                return True
        except:
            time.sleep(1)
    
    print("  ❌ API failed to start")
    return False

def test_health_endpoint():
    """Test API health check"""
    print("\nTesting health endpoint...")
    
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200, f"Health check failed: {response.status_code}"
    
    data = response.json()
    assert data.get('status') == 'healthy', "API not healthy"
    assert data.get('model_loaded') == True, "Model not loaded"
    
    print("  ✓ Health endpoint OK")
    print(f"  ✓ Status: {data}")
    return True

def test_root_endpoint():
    """Test root endpoint"""
    print("\nTesting root endpoint...")
    
    response = requests.get(f"{API_URL}/")
    assert response.status_code == 200, f"Root endpoint failed: {response.status_code}"
    
    data = response.json()
    assert 'message' in data, "Missing message in response"
    assert 'endpoints' in data, "Missing endpoints in response"
    
    print("  ✓ Root endpoint OK")
    return True

def test_prediction_endpoint():
    """Test prediction endpoint with sample data"""
    print("\nTesting prediction endpoint...")
    
    # Test data
    test_cases = [
        {
            "name": "Low risk case",
            "data": {
                "pm25": 30.0,
                "pm10": 50.0,
                "aqi": 100.0,
                "temperature": 22.0,
                "humidity": 55.0,
                "precipitation": 0.0,
                "heart_rate_avg": 70.0,
                "sleep_hours": 8.0,
                "spo2": 99.0,
                "stress_level": 3
            }
        },
        {
            "name": "High risk case",
            "data": {
                "pm25": 150.0,
                "pm10": 200.0,
                "aqi": 250.0,
                "temperature": 30.0,
                "humidity": 80.0,
                "precipitation": 0.0,
                "heart_rate_avg": 95.0,
                "sleep_hours": 4.0,
                "spo2": 94.0,
                "stress_level": 9
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n  Testing: {test_case['name']}")
        
        response = requests.post(
            f"{API_URL}/predict",
            json=test_case['data']
        )
        
        assert response.status_code == 200, f"Prediction failed: {response.status_code}"
        
        result = response.json()
        
        # Validate response structure
        assert 'prediction' in result, "Missing prediction"
        assert 'probability' in result, "Missing probability"
        assert 'risk_level' in result, "Missing risk_level"
        
        # Validate values
        assert result['prediction'] in [0, 1], "Invalid prediction value"
        assert 0 <= result['probability'] <= 1, "Invalid probability"
        assert result['risk_level'] in ['Low', 'Medium', 'High'], "Invalid risk level"
        
        print(f"    ✓ Prediction: {result['prediction']}")
        print(f"    ✓ Probability: {result['probability']:.3f}")
        print(f"    ✓ Risk Level: {result['risk_level']}")
    
    print("\n  ✓ All prediction tests passed")
    return True

def test_invalid_input():
    """Test API handles invalid input correctly"""
    print("\nTesting invalid input handling...")
    
    # Missing required field
    invalid_data = {
        "pm25": 50.0,
        "pm10": 80.0
        # Missing other required fields
    }
    
    response = requests.post(f"{API_URL}/predict", json=invalid_data)
    assert response.status_code == 422, "Should reject invalid input"
    print("  ✓ Correctly rejects invalid input (422)")
    
    # Invalid data types
    invalid_types = {
        "pm25": "not_a_number",
        "pm10": 80.0,
        "aqi": 150.0,
        "temperature": 25.0,
        "humidity": 60.0,
        "precipitation": 2.0,
        "heart_rate_avg": 75.0,
        "sleep_hours": 7.0,
        "spo2": 98.0,
        "stress_level": 5
    }
    
    response = requests.post(f"{API_URL}/predict", json=invalid_types)
    assert response.status_code == 422, "Should reject invalid types"
    print("  ✓ Correctly rejects invalid data types (422)")
    
    return True

def test_api_performance():
    """Test API response time"""
    print("\nTesting API performance...")
    
    test_data = {
        "pm25": 50.0,
        "pm10": 80.0,
        "aqi": 150.0,
        "temperature": 25.0,
        "humidity": 60.0,
        "precipitation": 2.0,
        "heart_rate_avg": 75.0,
        "sleep_hours": 7.0,
        "spo2": 98.0,
        "stress_level": 5
    }
    
    # Measure response time
    start_time = time.time()
    response = requests.post(f"{API_URL}/predict", json=test_data)
    response_time = time.time() - start_time
    
    assert response.status_code == 200, "Prediction failed"
    assert response_time < 1.0, f"Response too slow: {response_time:.3f}s"
    
    print(f"  ✓ Response time: {response_time*1000:.0f}ms (< 1000ms)")
    
    # Test multiple requests
    print("\n  Testing concurrent requests...")
    times = []
    for i in range(10):
        start = time.time()
        response = requests.post(f"{API_URL}/predict", json=test_data)
        times.append(time.time() - start)
        assert response.status_code == 200
    
    avg_time = sum(times) / len(times)
    print(f"  ✓ Average response time (10 requests): {avg_time*1000:.0f}ms")
    
    return True

if __name__ == "__main__":
    try:
        if not wait_for_api():
            print("\n❌ API not available")
            sys.exit(1)
        
        test_health_endpoint()
        test_root_endpoint()
        test_prediction_endpoint()
        test_invalid_input()
        test_api_performance()
        
        print("\n" + "="*50)
        print("ALL API TESTS PASSED ✅")
        print("="*50)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
