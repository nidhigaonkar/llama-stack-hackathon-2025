from image_processor import ImageProcessor
from vector_store import VectorStore
from llama_agent import LlamaAgent

# Initialize components
processor = ImageProcessor()
agent = LlamaAgent()
vector_store = VectorStore(llama_agent=agent)

# Process a few test images
test_images = processor.process_images()
if test_images:
    # Add images to vector store
    vector_store.add_images(test_images)
    
    # Test search functionality
    test_queries = [
        "Find images with people",
        "Show images from April 2025",
        "Find screenshots"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.search_images(query)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['path']} - {result['caption']}")
else:
    print("No test images found on desktop")
