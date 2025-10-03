"""
Test script for the Fraud Detection API.
Run this after starting the Docker container to test the API endpoints.
"""
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_root():
    """Test the root endpoint."""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint."""
    print("\nTesting model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_single_prediction():
    """Test single prediction endpoint."""
    print("\nTesting single prediction endpoint...")
    
    # Sample transaction data
    transaction_data = {
        "V1": -0.7059898246110177,
        "V2": 0.6277668093643811,
        "V3": -0.035994995232166,
        "V4": 0.1806427850874308,
        "V5": 0.4599348239833234,
        "V6": -0.036283158251373,
        "V7": 0.2802046719288935,
        "V8": -0.1841152764576969,
        "V9": 0.0685241005919484,
        "V10": 0.5863629005107058,
        "V11": -0.25233334795008,
        "V12": -1.2299078418984513,
        "V13": 0.4682882741114543,
        "V14": 0.4017355215141967,
        "V15": -0.3078030347127327,
        "V16": -0.1123814085906342,
        "V17": -0.4589679556521681,
        "V18": 0.0405522364190535,
        "V19": -0.9375302972907276,
        "V20": 0.1741002550832633,
        "V21": -0.1256561406066695,
        "V22": -0.1784533927889745,
        "V23": -0.1156088530642112,
        "V24": -0.2434813742463694,
        "V25": -1.156796820313679,
        "V26": 1.148810949147973,
        "V27": 1.0191007119749338,
        "V28": 0.0030985451139533,
        "Time": 0.0037648659393779,
        "Amount": -0.307400143722893
    }
    
    try:
        # Test with default threshold
        response = requests.post(f"{BASE_URL}/predict", json=transaction_data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Test with custom threshold
        response = requests.post(f"{BASE_URL}/predict?threshold=0.3", json=transaction_data)
        print(f"\nWith threshold=0.3:")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\nTesting batch prediction endpoint...")
    
    # Sample batch data
    batch_data = [
        {
            "V1": -0.7059898246110177,
            "V2": 0.6277668093643811,
            "V3": -0.035994995232166,
            "V4": 0.1806427850874308,
            "V5": 0.4599348239833234,
            "V6": -0.036283158251373,
            "V7": 0.2802046719288935,
            "V8": -0.1841152764576969,
            "V9": 0.0685241005919484,
            "V10": 0.5863629005107058,
            "V11": -0.25233334795008,
            "V12": -1.2299078418984513,
            "V13": 0.4682882741114543,
            "V14": 0.4017355215141967,
            "V15": -0.3078030347127327,
            "V16": -0.1123814085906342,
            "V17": -0.4589679556521681,
            "V18": 0.0405522364190535,
            "V19": -0.9375302972907276,
            "V20": 0.1741002550832633,
            "V21": -0.1256561406066695,
            "V22": -0.1784533927889745,
            "V23": -0.1156088530642112,
            "V24": -0.2434813742463694,
            "V25": -1.156796820313679,
            "V26": 1.148810949147973,
            "V27": 1.0191007119749338,
            "V28": 0.0030985451139533,
            "Time": 0.0037648659393779,
            "Amount": -0.307400143722893
        },
        {
            "V1": 1.2759412830021937,
            "V2": -0.1073248615478466,
            "V3": 0.7170780851178733,
            "V4": -0.5045525405422195,
            "V5": -0.364526151781289,
            "V6": 0.4559322575587443,
            "V7": -0.5409714373682498,
            "V8": 0.5189477504464394,
            "V9": 0.2071424413834301,
            "V10": -0.1423030701696631,
            "V11": -0.4692974133206888,
            "V12": -1.3694100936290607,
            "V13": -0.0922488166686573,
            "V14": 0.0211274549207885,
            "V15": 0.7672067998084137,
            "V16": 0.5413750293737937,
            "V17": -0.2297787436567511,
            "V18": -0.5157953917191936,
            "V19": 0.4145627533440857,
            "V20": 0.2001383717347231,
            "V21": -0.1280759689352185,
            "V22": 0.5318352228561054,
            "V23": 1.6655446033495047,
            "V24": -0.1329989149493462,
            "V25": 0.839086256169328,
            "V26": -1.363483481447708,
            "V27": -0.4867922823583477,
            "V28": 0.9548548079374352,
            "Time": 0.7950691916910888,
            "Amount": -0.3455792105666589
        }
    ]
    
    try:
        response = requests.post(f"{BASE_URL}/predict_batch", json=batch_data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("FRAUD DETECTION API TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_health,
        test_root,
        test_model_info,
        test_single_prediction,
        test_batch_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 40)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! API is working correctly.")
    else:
        print("❌ Some tests failed. Check the API logs.")

if __name__ == "__main__":
    main()
