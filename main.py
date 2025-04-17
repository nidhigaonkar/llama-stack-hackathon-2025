import os
from dotenv import load_dotenv
from image_processor import ImageProcessor
from vector_store import VectorStore
from llama_agent import LlamaAgent
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
from datetime import datetime

class DesktopEventHandler(FileSystemEventHandler):
    def __init__(self, image_processor, vector_store):
        self.image_processor = image_processor
        self.vector_store = vector_store
        self.last_processed = {}

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.png'):
            self.process_new_image(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.png'):
            self.process_new_image(event.src_path)

    def process_new_image(self, image_path):
        # Avoid processing the same image multiple times in quick succession
        current_time = time.time()
        if image_path in self.last_processed:
            if current_time - self.last_processed[image_path] < 5:  # 5 seconds cooldown
                return
        self.last_processed[image_path] = current_time

        try:
            # Process single image
            image_data = self.image_processor.process_single_image(image_path)
            if image_data:
                # Add to vector store
                self.vector_store.add_images([image_data])
                print(f"Processed new image: {image_path}")
        except Exception as e:
            print(f"Error processing new image {image_path}: {str(e)}")

def main():
    # Load environment variables
    load_dotenv()
    
    print("Initializing PNG Cleanup Assistant...")
    
    # Initialize Llama Stack client and agent
    print("Connecting to Llama Stack...")
    llama_agent = LlamaAgent()
    
    # Initialize components
    image_processor = ImageProcessor()
    vector_store = VectorStore(llama_agent=llama_agent)
    
    # Process existing images
    print("Processing existing images...")
    processed_images = image_processor.process_images()
    
    if processed_images:
        print(f"Found {len(processed_images)} existing PNG images")
        vector_store.add_images(processed_images)
    
    # Set up file system observer
    desktop_path = str(Path.home() / "Desktop")
    event_handler = DesktopEventHandler(image_processor, vector_store)
    observer = Observer()
    observer.schedule(event_handler, desktop_path, recursive=False)
    observer.start()
    
    print("\nPNG Cleanup Assistant is running!")
    print("You can now search for and delete PNG images using natural language.")
    print("Example queries:")
    print("- Find images from January 2021")
    print("- Delete all screenshots from last week")
    print("- Find images containing cats")
    print("\nType 'exit' to quit.")
    
    try:
        while True:
            query = input("\nEnter your query: ").strip()
            if query.lower() == 'exit':
                break
            
            # Parse user intent
            intent_analysis = llama_agent.understand_query(query)
            print(f"Understanding query: {intent_analysis}")
            
            # Search for images
            results = vector_store.search_images(query)
            
            if not results:
                print("No matching images found.")
                continue
            
            # Display results
            print("\nFound matching images:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Path: {result['path']}")
                print(f"   Caption: {result['caption']}")
                print(f"   Date: {result['creation_date']} {result['creation_time']}")
                if 'relevance_score' in result:
                    print(f"   Relevance: {result['relevance_score']:.2f}")
            
            # If intent is to delete, confirm with user
            if "delete" in query.lower() or intent_analysis.get("intent") == "delete":
                confirmation_msg = vector_store.confirm_deletion(results, query)
                print(f"\n{confirmation_msg}")
                
                user_confirm = input("Type 'yes' to confirm deletion: ").strip().lower()
                if user_confirm == 'yes':
                    # Delete files
                    for result in results:
                        try:
                            os.remove(result['path'])
                            print(f"Deleted: {result['path']}")
                        except Exception as e:
                            print(f"Error deleting {result['path']}: {str(e)}")
                else:
                    print("Deletion cancelled.")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()
