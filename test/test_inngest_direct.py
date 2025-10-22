import inngest
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Inngest client
inngest_client = inngest.Inngest(
    app_id="rag_app",
    is_production=False
)

async def test_image_ingest():
    """Test standalone image ingestion"""
    print("\n" + "=" * 50)
    print("TESTING IMAGE INGESTION")
    print("=" * 50)
    
    event = inngest.Event(
        name="rag/ingest_document",
        data={
            "image_path": "sample_data/beyblade.jpg", 
            "language": "en",
            "source_id": "test_image_1"
        }
    )
    
    print("Sending image ingestion event...")
    try:
        result = await inngest_client.send(event)
        print(f"✓ Event sent successfully. IDs: {result}")
        print("... Check your FastAPI logs (Terminal 3) or Inngest Dashboard to see it run.")
    except Exception as e:
        print(f"✗ Error sending event: {e}")

async def test_query():
    """Test RAG query"""
    print("\n" + "=" * 50)
    print("TESTING RAG QUERY")
    print("=" * 50)
    
    event = inngest.Event(
        name="rag/query_pdf_ai",
        data={
            "question": "What is shown in the beyblade image?",
            "top_k": 5,
            "language": "en"
        }
    )
    
    print("Sending query event...")
    try:
        result = await inngest_client.send(event)
        print(f"✓ Event sent successfully. IDs: {result}")
        print("... Check your FastAPI logs (Terminal 3) or Inngest Dashboard to see the answer.")
    except Exception as e:
        print(f"✗ Error sending event: {e}")

async def main():
    """Run image tests only"""
    print("\n" + "🚀" * 25)
    print("STARTING IMAGE INGESTION TEST")
    print("🚀" * 25)
    print("\nWaiting a moment for servers to be ready...")
    await asyncio.sleep(2)

    # Test 1: Image ingestion
    await test_image_ingest()
    
    # Wait for image to be indexed
    print("\n⏳ Waiting 10 seconds for image to be indexed...")
    await asyncio.sleep(10)
    
    # Test 2: Query about the image
    await test_query()
    
    print("\n" + "✅" * 25)
    print("ALL TESTS COMPLETED")
    print("✅" * 25)
    print("\n📋 Summary:")
    print("  1. ✓ Image ingestion event sent")
    print("  2. ✓ Query event sent")
    print("\n💡 Check your FastAPI logs or Inngest Dashboard for results!")

if __name__ == "__main__":
    asyncio.run(main())