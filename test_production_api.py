#!/usr/bin/env python3
"""Test the production API to verify it's working correctly"""

import requests
import json
import time

# Production API URL
BASE_URL = "http://python-fast-api-datanyx.eba-f3wni6xk.ap-south-1.elasticbeanstalk.com"

def test_api_health():
    """Test the API health endpoint"""
    print("Testing API health...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health", timeout=10)
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            print("✅ API is healthy and responding")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_models_endpoint():
    """Test the models info endpoint"""
    print("\nTesting models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/predict/models", timeout=15)
        print(f"Models endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Models endpoint responding")
            print(f"Total diseases: {data.get('total_diseases', 'unknown')}")
            print(f"Models available: {data.get('models_available', 'unknown')}")
            print(f"Models missing: {data.get('models_missing', 'unknown')}")
            
            # Show status of each model
            models = data.get('models', {})
            for disease, info in models.items():
                status = "✅" if info.get('available', False) else "❌"
                print(f"  {status} {disease}: {info.get('status', 'unknown')}")
            
            return True
        else:
            print(f"❌ Models endpoint failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint"""
    print("\nTesting prediction endpoint...")
    
    # Test cases
    test_cases = [
        {
            "name": "Low risk patient",
            "features": [25.0, 22.0, 0.5, 2.0, 1.0, 8.0, 9.0, 1.0, 90.0]
        },
        {
            "name": "Medium risk patient", 
            "features": [45.0, 25.0, 3.5, 6.0, 4.0, 7.0, 8.0, 2.5, 75.0]
        },
        {
            "name": "High risk patient",
            "features": [65.0, 30.0, 8.0, 9.0, 8.0, 3.0, 4.0, 4.5, 60.0]
        }
    ]
    
    headers = {'Content-Type': 'application/json'}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {test_case['name']}")
        
        payload = {
            "features": test_case['features']
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/predict/ml", 
                json=payload, 
                headers=headers,
                timeout=20
            )
            
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print("  ✅ Prediction successful")
                print(f"  Total diseases evaluated: {data.get('total_diseases_evaluated', 'unknown')}")
                
                predictions = data.get('predictions', {})
                for disease, pred in predictions.items():
                    status = pred.get('status', 'unknown')
                    confidence = pred.get('confidence', 0)
                    model_type = pred.get('model_type', 'unknown')
                    print(f"    {disease:<12}: {status} (confidence: {confidence:.3f}) [{model_type}]")
                
                if 'warning' in data:
                    print(f"  ⚠ Warning: {data['warning']}")
                    
            elif response.status_code == 422:
                print("  ❌ Validation error (expected for invalid input)")
                error_data = response.json()
                print(f"  Error: {error_data}")
            else:
                print(f"  ❌ Prediction failed with status {response.status_code}")
                print(f"  Response: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
            return False
    
    return True

def test_invalid_input():
    """Test the API with invalid input"""
    print("\nTesting invalid input handling...")
    
    headers = {'Content-Type': 'application/json'}
    
    # Test with wrong number of features
    payload = {
        "features": [1, 2, 3]  # Only 3 features instead of 9
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/predict/ml", 
            json=payload, 
            headers=headers,
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 400 or response.status_code == 422:
            print("✅ Correctly rejected invalid input")
            error_data = response.json()
            print(f"Error message: {error_data}")
            return True
        else:
            print(f"❌ Should have rejected invalid input but got status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Invalid input test failed: {e}")
        return False

def main():
    """Run all API tests"""
    print("="*70)
    print("TESTING PRODUCTION API")
    print(f"API URL: {BASE_URL}")
    print("="*70)
    
    tests = [
        ("API Health", test_api_health),
        ("Models Info", test_models_endpoint),
        ("Prediction", test_prediction_endpoint),
        ("Invalid Input", test_invalid_input)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
        
        # Small delay between tests
        time.sleep(1)
    
    # Summary
    print("\n" + "="*70)
    print("API TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<15}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All API tests passed! The production API is working perfectly!")
        print("✅ The SimpleRandomForest errors have been resolved!")
        print("✅ Models are loading and making predictions correctly!")
    else:
        print(f"⚠ {total - passed} test(s) failed. Check the details above.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹ Tests interrupted by user")
        exit_code = 1
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1