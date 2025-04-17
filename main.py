import os
from dotenv import load_dotenv
from image_processor import ImageProcessor
from vector_store import VectorStore
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path


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
           caption = self.image_processor.generate_caption(image_path)
           creation_time = datetime.fromtimestamp(os.path.getctime(image_path))
          
           image_data = {
               'path': image_path,
               'caption': caption,
               'creation_date': creation_time.strftime('%Y-%m-%d'),
               'creation_time': creation_time.strftime('%H:%M:%S')
           }
          
           # Add to vector store
           self.vector_store.add_images([image_data])
           print(f"Processed new image: {image_path}")
          
       except Exception as e:
           print(f"Error processing new image {image_path}: {str(e)}")


def main():
   # Load environment variables
   load_dotenv()
  
   print("Initializing Image Search Assistant...")
  
   # Initialize components
   image_processor = ImageProcessor()
   vector_store = VectorStore()
  
   # Process existing images
   print("Processing existing images...")
   processed_images = image_processor.process_images()
   if processed_images:
       print(f"Found {len(processed_images)} existing images")
       vector_store.add_images(processed_images)
  
   # Set up file system observer
   desktop_path = str(Path.home() / "Desktop")
   event_handler = DesktopEventHandler(image_processor, vector_store)
   observer = Observer()
   observer.schedule(event_handler, desktop_path, recursive=False)
   observer.start()
  
   print("\nImage Search Assistant is running!")
   print("You can now search for images using natural language queries.")
   print("Example queries:")
   print("- Show me images from April 1st")
   print("- Find images containing text about chemistry")
   print("- What images do I have from last week?")
   print("\nType 'exit' to quit.")
  
   try:
       while True:
           query = input("\nEnter your search query: ").strip()
           if query.lower() == 'exit':
               break
              
           results = vector_store.search_images(query)
          
           if results:
               print("\nFound images:")
               for i, result in enumerate(results, 1):
                   print(f"\n{i}. Path: {result['path']}")
                   print(f"   Date: {result['creation_date']} {result['creation_time']}")
                   if result['relevance_score']:
                       print(f"   Relevance: {result['relevance_score']:.2f}")
           else:
               print("\nNo images found matching your query.")
              
   except KeyboardInterrupt:
       print("\nShutting down...")
   finally:
       observer.stop()
       observer.join()


if __name__ == "__main__":
   main()

