import requests
from qdrant_client import QdrantClient

def check_qdrant():
    """Check if Qdrant is accessible"""
    try:
        client = QdrantClient(url="http://localhost:6333", timeout=5)
        collections = client.get_collections()
        print("✓ Qdrant is running")
        print(f"  Collections: {[c.name for c in collections.collections]}")
        return True
    except Exception as e:
        print(f"✗ Qdrant error: {e}")
        return False

def check_fastapi():
    """Check if FastAPI server is running"""
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("✓ FastAPI server is running")
            return True
    except Exception as e:
        print(f"✗ FastAPI error: {e}")
        return False

def check_openai_key():
    """Check if OpenAI API key is set"""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    key = os.getenv("OPENAI_API_KEY")
    if key and key.startswith("sk-"):
        print("✓ OpenAI API key is set")
        return True
    else:
        print("✗ OpenAI API key not found or invalid")
        return False

if __name__ == "__main__":
    print("=== System Health Check ===\n")
    check_openai_key()
    check_qdrant()
    check_fastapi()