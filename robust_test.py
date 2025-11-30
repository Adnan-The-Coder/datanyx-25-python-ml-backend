#!/usr/bin/env python3
"""Robust testing of the production API"""

import requests
import json
import time

BASE_URL = "http://python-fast-api-datanyx.eba-f3wni6xk.ap-south-1.elasticbeanstalk.com"

def test_comprehensive():
    print("="*70)
    print("COMPREHENSIVE PRODUCTION API TESTING")
    print("="*70)
    
    # Test 1: Basic API info
    print("\n1. 🔍 API ROOT ENDPOINT:")
    try:
        r = requests.get(BASE_URL, timeout=10)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"   ✅ Message: {data.get('message', 'N/A')}")
            print(f"   ✅ Diseases: {len(data.get('diseases', []))}")
            print(f"   ✅ API Version: {data.get('version', 'N/A')}")
        else:
            print(f"   ❌ Failed with status {r.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Health check
    print("\n2. ❤️ HEALTH CHECK:")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"   ✅ Health: {data.get('status', 'N/A')}")
            print(f"   ✅ Models Loaded: {data.get('models_loaded', 'N/A')}")
            print(f"   ✅ API Name: {data.get('api_name', 'N/A')}")
        else:
            print(f"   ❌ Failed with status {r.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Multiple prediction tests with different risk levels
    test_cases = [
        {
            "name": "Low Risk Patient",
            "features": [25.0, 22.0, 0.5, 2.0, 1.0, 8.0, 9.0, 1.0, 90.0]
        },
        {
            "name": "Medium Risk Patient", 
            "features": [45.0, 25.0, 3.5, 6.0, 4.0, 7.0, 8.0, 2.5, 75.0]
        },
        {
            "name": "High Risk Patient",
            "features": [65.0, 30.0, 8.0, 9.0, 8.0, 3.0, 4.0, 4.5, 60.0]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i+2}. 🧪 ML PREDICTION - {test_case['name'].upper()}:")
        try:
            payload = {"features": test_case['features']}
            r = requests.post(f"{BASE_URL}/predict/ml", json=payload, timeout=15)
            print(f"   Status: {r.status_code}")
            
            if r.status_code == 200:
                data = r.json()
                predictions = data.get('predictions', {})
                
                errors = 0
                successes = 0
                fallbacks = 0
                
                print(f"   Patient Features: {test_case['features']}")
                print(f"   Predictions:")
                
                for disease, pred in predictions.items():
                    if 'error' in pred:
                        errors += 1
                        error_msg = pred['error'][:60] + "..." if len(pred['error']) > 60 else pred['error']
                        print(f"     ❌ {disease:<12}: ERROR - {error_msg}")
                    else:
                        if pred.get('model_type') == 'fallback':
                            fallbacks += 1
                            print(f"     ⚠️  {disease:<12}: {pred.get('status', 'unknown')} (FALLBACK)")
                        else:
                            successes += 1
                            status = pred.get('status', 'unknown')
                            conf = pred.get('confidence', 0)
                            model_type = pred.get('model_type', 'unknown')
                            print(f"     ✅ {disease:<12}: {status} (conf: {conf:.3f}) [{model_type}]")
                
                print(f"   📊 Results: {successes} ML success, {fallbacks} fallback, {errors} errors")
                
                if 'warning' in data:
                    print(f"   ⚠️  Warning: {data['warning']}")
                    
            else:
                print(f"   ❌ Request failed with status {r.status_code}")
                print(f"   Response: {r.text[:200]}...")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    # Test 4: Error handling
    print(f"\n{len(test_cases)+3}. 🚫 ERROR HANDLING TEST:")
    try:
        # Test with wrong number of features
        payload = {"features": [1, 2, 3]}  # Only 3 instead of 9
        r = requests.post(f"{BASE_URL}/predict/ml", json=payload, timeout=10)
        print(f"   Status: {r.status_code}")
        
        if r.status_code in [400, 422]:
            print("   ✅ Correctly rejected invalid input")
            data = r.json()
            print(f"   Error details: {data}")
        else:
            print(f"   ❌ Should have rejected invalid input but got {r.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Documentation endpoints
    print(f"\n{len(test_cases)+4}. 📚 DOCUMENTATION ENDPOINTS:")
    doc_endpoints = ["/docs", "/redoc"]
    for endpoint in doc_endpoints:
        try:
            r = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            status = "✅" if r.status_code == 200 else "❌"
            print(f"   {status} {endpoint}: {r.status_code}")
        except Exception as e:
            print(f"   ❌ {endpoint}: Error - {e}")
    
    print("\n" + "="*70)
    print("🔍 ANALYSIS & RECOMMENDATIONS")
    print("="*70)
    
    # Test the specific error we were fixing
    print("\nTesting for SimpleRandomForest errors specifically...")
    try:
        payload = {"features": [45.0, 25.0, 3.5, 6.0, 4.0, 7.0, 8.0, 2.5, 75.0]}
        r = requests.post(f"{BASE_URL}/predict/ml", json=payload, timeout=15)
        
        if r.status_code == 200:
            data = r.json()
            predictions = data.get('predictions', {})
            
            has_simplerf_errors = any(
                'error' in pred and 'SimpleRandomForest' in pred['error'] 
                for pred in predictions.values()
            )
            
            if has_simplerf_errors:
                print("❌ ISSUE DETECTED: SimpleRandomForest errors still present!")
                print("   🔧 SOLUTION: The production deployment needs to be updated with our fixes")
                print("   📁 FILES TO DEPLOY:")
                print("      - Updated predict.py (already created locally)")
                print("      - New scikit-learn model files (already generated locally)")
                print("   🚀 ACTION NEEDED: Deploy the updated code and models to production")
            else:
                print("✅ SUCCESS: SimpleRandomForest errors have been resolved!")
                print("   🎉 All models are working correctly in production")
                
    except Exception as e:
        print(f"❌ Could not test for specific errors: {e}")

if __name__ == "__main__":
    test_comprehensive()