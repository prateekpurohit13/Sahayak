from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

# Check collection exists
collections = client.get_collections()
print("Collections:", [c.name for c in collections.collections])

# Check points in collection
if client.collection_exists("docs"):
    info = client.get_collection("docs")
    print(f"\nCollection 'docs' info:")
    print(f"  Vector size: {info.config.params.vectors.size}")
    print(f"  Points count: {info.points_count}")
    
    # Sample a few points
    points = client.scroll("docs", limit=3, with_payload=True)
    print(f"\nSample points:")
    for point in points[0]:
        print(f"  ID: {point.id}")
        print(f"  Payload: {point.payload}")
        print()