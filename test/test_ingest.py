import requests
import json
import time

def test_pdf_ingest():
    """Test ingesting a PDF document"""
    url = "http://localhost:8288/api/events"
    
    payload = {
        "name": "rag/ingest_document",
        "data": {
            "pdf_path": r"sample_data/test.pdf",  # Update this!
            "language": "en",
            "source_id": "test_document_1"
        },
        "user": {}
    }
    
    print("=" * 50)
    print("TESTING PDF INGESTION")
    print("=" * 50)
    print(f"Sending request to: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")

        # --- FIX ---
        if response.status_code in [200, 201, 202]: # 202 is also a valid "accepted"
            print("\n✓ PDF ingestion event sent successfully!")
            print("... Check your FastAPI logs (Terminal 3) or Inngest Dashboard (http://127.0.0.1:8288) to see it run.")
        else:
            print("\n✗ PDF ingestion failed!")
            print(f"Error Response: {response.text}") # Print raw text, not json
        # --- END FIX ---
            
    except requests.exceptions.RequestException as e: # More specific exception
        print(f"\n✗ Connection Error: {e}")

def test_image_ingest():
    """Test ingesting a standalone image"""
    url = "http://localhost:8288/api/events"
    
    payload = {
        "name": "rag/ingest_document",
        "data": {
            "image_path": "sample_data/beyblade.jpg", # Make sure this path is correct!
            # --- FIX: Removed the 'z' ---
            "language": "en",
            # --- END FIX ---
            "source_id": "test_image_1"
        },
        "user": {}
    }
    
    print("\n" + "=" * 50)
    print("TESTING IMAGE INGESTION")
    print("=" * 50)
    print(f"Sending request to: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        
        # --- FIX ---
        if response.status_code in [200, 201, 202]:
            print("\n✓ Image ingestion event sent successfully!")
            print("... Check your FastAPI logs (Terminal 3) or Inngest Dashboard (http://127.0.0.1:8288) to see it run.")
        else:
            print("\n✗ Image ingestion failed!")
            print(f"Error Response: {response.text}") # Print raw text, not json
        # --- END FIX ---
            
    except requests.exceptions.RequestException as e: # More specific exception
        print(f"\n✗ Connection Error: {e}")

if __name__ == "__main__":
    print("\nWaiting a moment for server to be ready...")
    time.sleep(2)
    
    # Test image ingestion
    test_image_ingest()
    
    # Uncomment to test PDF ingestion
    # print("\nWaiting 5 seconds before next test...")
    # time.sleep(5)
    # test_pdf_ingest()