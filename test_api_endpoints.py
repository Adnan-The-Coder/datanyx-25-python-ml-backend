"""Test script to verify API endpoints work correctly."""
import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_endpoints():
    print("Testing ML Prediction API Endpoints")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("1. Testing root endpoint GET /")
    try:
        response = requests.get(f"{API_BASE}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "-" * 50)
    
    # Test 2: Health endpoint
    print("2. Testing health endpoint GET /health")
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "-" * 50)
    
    # Test 3: Models endpoint
    print("3. Testing models endpoint GET /models")
    try:
        response = requests.get(f"{API_BASE}/models")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "-" * 50)
    
    # Test 4: Prediction endpoint
    print("4. Testing prediction endpoint POST /predict")
    test_data = {
        "features": [45.0, 25.0, 7.5, 6.0, 5.0, 8.0, 3.0, 3.5, 60.0]
    }
    
    try:
        response = requests.post(f"{API_BASE}/predict", json=test_data)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 50)
    print("API Testing Complete!")

if __name__ == "__main__":
    print("Starting API endpoint tests...")
    print("Make sure the API server is running on http://localhost:8000")
    print("Waiting 2 seconds before starting tests...")
    time.sleep(2)
    test_endpoints()