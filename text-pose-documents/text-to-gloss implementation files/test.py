"""
Simple tests for Text-to-Gloss API
"""

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("✅ Health check passed")


def test_simple_conversion():
    """Test simple text conversion"""
    response = requests.post(
        f"{BASE_URL}/api/v1/text-to-gloss",
        json={"text": "I am searching for a doctor"}
    )
    assert response.status_code == 200
    data = response.json()
    
    print(f"\nInput: {data['text']}")
    print(f"Gloss: {data['gloss_string']}")
    print(f"Glosses: {data['glosses']}")
    
    assert "IX-1" in data["gloss_string"]
    assert "DOCTOR" in data["gloss_string"]
    assert "SEARCH" in data["gloss_string"]
    assert len(data["glosses"]) >= 3
    print("✅ Simple conversion passed")


def test_temporal():
    """Test temporal sentence"""
    response = requests.post(
        f"{BASE_URL}/api/v1/text-to-gloss",
        json={"text": "Tomorrow I will go to school"}
    )
    data = response.json()
    
    print(f"\nInput: {data['text']}")
    print(f"Gloss: {data['gloss_string']}")
    
    assert data["gloss_string"].startswith("TOMORROW")
    print("✅ Temporal test passed")


def test_negation():
    """Test negation"""
    response = requests.post(
        f"{BASE_URL}/api/v1/text-to-gloss",
        json={"text": "She doesn't like pizza"}
    )
    data = response.json()
    
    print(f"\nInput: {data['text']}")
    print(f"Gloss: {data['gloss_string']}")
    
    assert data["gloss_string"].endswith("NOT")
    print("✅ Negation test passed")


def test_indexed_output():
    """Test that glosses are properly indexed"""
    response = requests.post(
        f"{BASE_URL}/api/v1/text-to-gloss",
        json={"text": "Hello"}
    )
    data = response.json()
    
    print(f"\nIndexed glosses for lookup:")
    for item in data["glosses"]:
        print(f"  [{item['index']}] {item['gloss']} (from '{item['original_word']}')")
    
    # Check indices are sequential
    indices = [item["index"] for item in data["glosses"]]
    assert indices == list(range(len(indices)))
    print("✅ Indexed output test passed")


def test_batch():
    """Test batch conversion"""
    texts = [
        "Hello",
        "I am searching for a doctor",
        "Tomorrow I will go to school"
    ]
    
    response = requests.post(
        f"{BASE_URL}/api/v1/batch",
        json={"texts": texts}
    )
    data = response.json()
    
    print(f"\nBatch results:")
    for result in data["results"]:
        print(f"  '{result['text']}' → '{result['gloss_string']}'")
    
    assert data["successful"] == 3
    assert data["failed"] == 0
    print("✅ Batch test passed")


if __name__ == "__main__":
    print("Starting tests...\n")
    print("=" * 60)
    
    try:
        test_health()
        test_simple_conversion()
        test_temporal()
        test_negation()
        test_indexed_output()
        test_batch()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
