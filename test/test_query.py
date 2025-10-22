import requests
import json
import time

def test_query():
    """Test querying the RAG system"""
    url = "http://localhost:8288/api/events"
    
    payload = {
        "name": "rag/query_pdf_ai",
        "data": {
            "question": "What is shown in this image?",  # Update this question!
            "top_k": 5,
            "language": "en"
        },
        "user": {}
    }
    
    print("=" * 50)
    print("TESTING RAG QUERY")
    print("=" * 50)
    print(f"Sending request to: {url}")
    print(f"Question: {payload['data']['question']}")
    print()
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        
        # --- FIX ---
        if response.status_code in [200, 201, 202]:
            print("\n✓ Query event sent successfully!")
            print("... Check your FastAPI logs (Terminal 3) or Inngest Dashboard to see the answer.")
        else:
            print(f"\n✗ Query failed!")
            print(f"Error Response: {response.text}")
        # --- END FIX ---
            
    except requests.exceptions.RequestException as e: # More specific exception
        print(f"\n✗ Connection Error: {e}")

if __name__ == "__main__":
    print("\nWaiting a moment for server to be ready...")
    time.sleep(2)
    
    test_query()